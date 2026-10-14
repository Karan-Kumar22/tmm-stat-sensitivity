"""Batch accumulation: seeds, manifest, and combination.

A model point lives in one directory. Its ``manifest.json`` locks the
parameters on first use and records every completed batch, so that a rerun
*appends* new batches instead of starting over:

    point_dir/
      manifest.json
      batches/
        cryostat_00000.hepevt
        cryostat_00000_truth.pkl
        dirt_00000.hepevt
        dirt_00000_truth.pkl
        ...
      combined/                      (written by `generate.py combine`)
        cryostat_truth.pkl
        dirt_truth.pkl

Determinism contract: every random stream of batch ``i`` derives from
``SeedSequence(entropy=base_seed, spawn_key=(i, stream))`` — the spawn-based
derivation, not ``base_seed + i``, which can yield correlated streams.
Streams: 0 = cryostat DarkNews, 1 = dirt DarkNews, 2 = forced-decay RNG.
Because the derivation is pure, a crashed batch is regenerated identically;
an orphan batch file (crash between file write and manifest update) is
safely overwritten byte-identically on the next run.

Combination applies the 1/N chunk normalisation: each DarkNews run's weights
sum to the full physical rate, so the unbiased combination of N
identical-neval batches is their average. Which columns are scaled is
declared in config.weights; probabilities (P_inside) are never scaled.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from config import Config

MANIFEST_NAME = "manifest.json"
BATCHES_SUBDIR = "batches"
COMBINED_SUBDIR = "combined"
FORMAT_VERSION = 1

COMPONENTS = ("cryostat", "dirt")

STREAM_CRYOSTAT_DARKNEWS = 0
STREAM_DIRT_DARKNEWS = 1
STREAM_FORCED_DECAY = 2


class ManifestError(RuntimeError):
    """Manifest missing, corrupt, or inconsistent with the requested run."""


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


def spawn_seed_sequence(
    base_seed: int, batch_index: int, stream: int
) -> np.random.SeedSequence:
    """The one true seed derivation: (base_seed, batch, stream) -> SeedSequence."""
    return np.random.SeedSequence(
        entropy=int(base_seed), spawn_key=(int(batch_index), int(stream))
    )


def darknews_seed(base_seed: int, batch_index: int, stream: int) -> int:
    """A positive int32 seed for DarkNews, derived from the SeedSequence."""
    state = spawn_seed_sequence(base_seed, batch_index, stream).generate_state(
        1, dtype=np.uint32
    )[0]
    # map uniformly into [1, 2^31 - 2]: positive, nonzero, fits everywhere
    return int(state) % (2**31 - 2) + 1


def forced_decay_rng(base_seed: int, batch_index: int) -> np.random.Generator:
    """The per-batch Generator for the dirt forced-decay sampling."""
    return np.random.default_rng(
        spawn_seed_sequence(base_seed, batch_index, STREAM_FORCED_DECAY)
    )


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def batches_directory(point_directory: str | Path) -> Path:
    return Path(point_directory) / BATCHES_SUBDIR


def combined_directory(point_directory: str | Path) -> Path:
    return Path(point_directory) / COMBINED_SUBDIR


def hepevt_path(point_directory: str | Path, component: str, index: int) -> Path:
    return batches_directory(point_directory) / f"{component}_{index:05d}.hepevt"


def truth_table_path(point_directory: str | Path, component: str, index: int) -> Path:
    return batches_directory(point_directory) / f"{component}_{index:05d}_truth.pkl"


def darknews_dataframe_path(
    point_directory: str | Path, component: str, index: int
) -> Path:
    return batches_directory(point_directory) / f"{component}_{index:05d}_darknews.pkl"


def darknews_scratch_path(
    point_directory: str | Path, component: str, index: int
) -> Path:
    return Path(point_directory) / "darknews_scratch" / f"{component}_{index:05d}"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def config_snapshot(config: Config) -> dict[str, Any]:
    """A JSON-safe snapshot of every config section (source path excluded).

    Stored in the manifest on first run; later runs must present an equal
    snapshot, which guarantees all batches of a point share identical
    settings (geometry included).
    """
    sections = {}
    for name in ("generation", "columns", "weights", "storage", "hepevt", "geometry"):
        sections[name] = dataclasses.asdict(getattr(config, name))
    # normalise tuples -> lists etc. exactly as JSON storage will
    return json.loads(json.dumps(sections))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".tmp.")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


class Manifest:
    """The per-model-point record of locked parameters and completed batches."""

    def __init__(self, path: Path, data: dict[str, Any]):
        self.path = path
        self.data = data

    # -- creation / loading -------------------------------------------------

    @classmethod
    def create_or_load(
        cls,
        point_directory: str | Path,
        config: Config,
        *,
        mass: float,
        tmm: float,
    ) -> "Manifest":
        """Open the point's manifest, creating it on first use.

        On reuse, the requested (mass, tmm) and the full config snapshot must
        match what the existing batches were generated with — otherwise the
        run is refused: mixing settings within one point would corrupt the
        1/N combination.
        """
        point_directory = Path(point_directory)
        path = point_directory / MANIFEST_NAME
        snapshot = config_snapshot(config)

        if path.is_file():
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                raise ManifestError(f"Corrupt manifest {path}: {exc}") from exc

            if data.get("format_version") != FORMAT_VERSION:
                raise ManifestError(
                    f"Manifest {path} has format_version "
                    f"{data.get('format_version')!r}, expected {FORMAT_VERSION}"
                )
            for name, requested, stored in (
                ("mass_GeV", mass, data.get("mass_GeV")),
                ("tmm_GeV_inv", tmm, data.get("tmm_GeV_inv")),
            ):
                if stored is None or not math.isclose(
                    float(requested), float(stored), rel_tol=1e-9, abs_tol=0.0
                ):
                    raise ManifestError(
                        f"{name} mismatch: requested {requested!r}, manifest "
                        f"has {stored!r} ({path})"
                    )
            # Storage and HEPevt formatting choices do not change the physics
            # or statistics of a batch, so they are allowed to differ between
            # batches; everything else is locked.
            def physics_view(payload: Mapping[str, Any]) -> dict[str, Any]:
                return {
                    key: value
                    for key, value in payload.items()
                    if key not in ("storage", "hepevt")
                }

            if physics_view(data.get("config_snapshot", {})) != physics_view(snapshot):
                raise ManifestError(
                    f"Configuration differs from the one used for the existing "
                    f"batches in {point_directory}. Refusing to mix settings "
                    "within one model point; use a new directory (or restore "
                    "the original config)."
                )
            return cls(path, data)

        data = {
            "format_version": FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mass_GeV": float(mass),
            "tmm_GeV_inv": float(tmm),
            "neval_per_batch": config.generation.neval_per_batch,
            "base_seed": config.generation.base_seed,
            "config_snapshot": snapshot,
            "batches": {component: [] for component in COMPONENTS},
        }
        manifest = cls(path, data)
        manifest.save()
        return manifest

    @classmethod
    def load(cls, point_directory: str | Path) -> "Manifest":
        """Open an existing manifest (status/combine — no parameters needed)."""
        path = Path(point_directory) / MANIFEST_NAME
        if not path.is_file():
            raise ManifestError(f"No manifest found in {point_directory}")
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ManifestError(f"Corrupt manifest {path}: {exc}") from exc
        return cls(path, data)

    # -- accessors ----------------------------------------------------------

    @property
    def mass(self) -> float:
        return float(self.data["mass_GeV"])

    @property
    def tmm(self) -> float:
        return float(self.data["tmm_GeV_inv"])

    @property
    def base_seed(self) -> int:
        return int(self.data["base_seed"])

    @property
    def neval_per_batch(self) -> int:
        return int(self.data["neval_per_batch"])

    def batches(self, component: str) -> list[dict[str, Any]]:
        if component not in COMPONENTS:
            raise ManifestError(f"Unknown component {component!r}")
        return self.data["batches"][component]

    def next_index(self, component: str) -> int:
        return len(self.batches(component))

    # -- mutation -----------------------------------------------------------

    def record_batch(self, component: str, entry: Mapping[str, Any]) -> None:
        """Append a completed batch and persist the manifest atomically.

        Call this only after the batch's files are fully on disk: the
        manifest is the source of truth, and a file without a manifest entry
        is a harmless orphan, while an entry without files would be a lie.
        """
        entries = self.batches(component)
        index = int(entry["index"])
        if index != len(entries):
            raise ManifestError(
                f"Batch index {index} out of order for {component!r}: "
                f"expected {len(entries)}"
            )
        entries.append(dict(entry))
        self.save()

    def save(self) -> None:
        _atomic_write_json(self.path, self.data)


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------


def combine_truth_tables(
    point_directory: str | Path,
    component: str,
    config: Config,
) -> pd.DataFrame:
    """Concatenate all completed batches of a component, applying 1/N.

    Returns the combined truth table; the caller decides where to save it.
    The configured weight columns are divided by the number of batches; all
    other columns (P_inside included) are untouched.
    """
    manifest = Manifest.load(point_directory)
    entries = manifest.batches(component)
    if not entries:
        raise ManifestError(f"No completed {component!r} batches in {point_directory}")

    tables = []
    for entry in entries:
        path = truth_table_path(point_directory, component, int(entry["index"]))
        if not path.is_file():
            raise ManifestError(
                f"Manifest lists batch {entry['index']} but {path} is missing"
            )
        table = pd.read_pickle(path)
        if len(table) != int(entry["kept_rows"]):
            raise ManifestError(
                f"{path} has {len(table)} rows, manifest says {entry['kept_rows']}"
            )
        table = table.copy()
        table["batch_index"] = int(entry["index"])
        tables.append(table)

    combined = pd.concat(tables, ignore_index=True)

    n_batches = len(entries)
    scale_columns = (
        config.weights.cryostat_scale_columns
        if component == "cryostat"
        else config.weights.dirt_scale_columns
    )
    for column in scale_columns:
        if column not in combined.columns:
            raise ManifestError(
                f"Weight column {column!r} configured for scaling is absent "
                f"from the {component} truth tables"
            )
        combined[column] = combined[column] / n_batches

    combined.attrs = {
        "component": component,
        "mass_GeV": manifest.mass,
        "tmm_GeV_inv": manifest.tmm,
        "n_batches": n_batches,
        "neval_per_batch": manifest.neval_per_batch,
        "chunk_normalization": f"weight columns {list(scale_columns)} divided by {n_batches}",
    }
    return combined
