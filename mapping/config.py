"""Configuration for the TMM mapping program.

Separate from the generation package on purpose: generation settings are
locked into each model point's manifest and must never change mid-point,
while the binning here is *expected* to be iterated on during the binning
study (plan section 9, steps 1 and 9). One JSON file, three sections:

  binning       — the truth-cell axes (E_gamma, cos_theta, z)
  cells         — cell-quality rules (minimum effective events for merging)
  truth_columns — column names of the generation package's truth tables

Same design as the generation config: typed frozen dataclasses, every value
validated at load time, `python config.py` prints what was parsed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class ConfigError(ValueError):
    """Raised when the configuration file is missing or inconsistent."""


@dataclass(frozen=True)
class BinningSettings:
    """The truth-cell axes.

    Energy: uniform bins of ``energy_uniform_width_GeV`` over
    [start, stop], then the explicit wide tail edges, optionally closed by a
    single overflow bin reaching to infinity.

    cos(theta): either ``uniform`` over [-1, 1], or ``log`` — bins uniform in
    log10(1 - cos_theta) between the forward edge and 2 (the plan section 6
    recommendation: automatic forward resolution), plus one extra most-forward
    bin covering [1 - forward_edge, 1].

    z: uniform bins in detector-frame cm. The range must cover the full
    liquid cryostat (roughly -93 to +1130 cm), which is wider than the TPC:
    decays happen in all of the liquid, and a map whose z axis stops at the
    TPC would silently drop out-of-TPC truth (plan section 7).

    Explicit mode: if the section contains ``explicit_edges_file`` (a JSON
    file with energy_edges_GeV / costheta_edges / z_edges_cm, path relative
    to the config file), those edges ARE the binning and every other key in
    this section is ignored. This is how the occupancy-driven quantile
    binning (edges at isotropic-source quantiles) is frozen into a file and
    adopted as the analysis binning.
    """

    energy_uniform_start_GeV: float
    energy_uniform_stop_GeV: float
    energy_uniform_width_GeV: float
    energy_tail_edges_GeV: tuple[float, ...]
    include_energy_overflow: bool
    costheta_scheme: str
    number_of_costheta_bins: int
    costheta_log_forward_edge: float
    costheta_piecewise_edges: tuple[float, ...]
    number_of_z_bins: int
    z_min_cm: float
    z_max_cm: float
    explicit_energy_edges: tuple[float, ...] = ()
    explicit_costheta_edges: tuple[float, ...] = ()
    explicit_z_edges: tuple[float, ...] = ()
    explicit_edges_file: str = ""
    # optional frozen merged (adaptive) binning built ON TOP of the base grid
    # defined by the keys above; see merged_binning.load_merged_binning
    merged_cells_file: str = ""

    @staticmethod
    def _from_explicit_file(section: Mapping[str, Any], base: Path) -> "BinningSettings":
        name = str(section["explicit_edges_file"])
        path = base / name
        if not path.is_file():
            raise ConfigError(f"explicit_edges_file not found: {path}")
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ConfigError(f"explicit_edges_file {path} is not valid JSON: {exc}") from exc

        edges: dict[str, tuple[float, ...]] = {}
        for key in ("energy_edges_GeV", "costheta_edges", "z_edges_cm"):
            if key not in payload:
                raise ConfigError(f"explicit_edges_file {path} is missing {key!r}")
            values = tuple(float(v) for v in payload[key])
            if len(values) < 2:
                raise ConfigError(f"{key} in {path} needs at least 2 edges")
            if any(b <= a for a, b in zip(values, values[1:])):
                raise ConfigError(f"{key} in {path} must be strictly increasing")
            edges[key] = values

        return BinningSettings(
            energy_uniform_start_GeV=edges["energy_edges_GeV"][0],
            energy_uniform_stop_GeV=edges["energy_edges_GeV"][-2],
            energy_uniform_width_GeV=1.0,
            energy_tail_edges_GeV=(),
            include_energy_overflow=not np.isfinite(edges["energy_edges_GeV"][-1]),
            costheta_scheme="explicit",
            number_of_costheta_bins=len(edges["costheta_edges"]) - 1,
            costheta_log_forward_edge=1e-3,
            costheta_piecewise_edges=edges["costheta_edges"],
            number_of_z_bins=len(edges["z_edges_cm"]) - 1,
            z_min_cm=edges["z_edges_cm"][0],
            z_max_cm=edges["z_edges_cm"][-1],
            explicit_energy_edges=edges["energy_edges_GeV"],
            explicit_costheta_edges=edges["costheta_edges"],
            explicit_z_edges=edges["z_edges_cm"],
            explicit_edges_file=name,
        )

    @staticmethod
    def from_mapping(section: Mapping[str, Any],
                     base: Path = Path(".")) -> "BinningSettings":
        if "explicit_edges_file" in section:
            return BinningSettings._from_explicit_file(section, base)

        def require(key: str) -> Any:
            if key not in section:
                raise ConfigError(f"Missing key {key!r} in config section 'binning'")
            return section[key]

        start = float(require("energy_uniform_start_GeV"))
        stop = float(require("energy_uniform_stop_GeV"))
        width = float(require("energy_uniform_width_GeV"))
        if width <= 0.0:
            raise ConfigError(f"energy_uniform_width_GeV must be positive, got {width}")
        if not stop > start:
            raise ConfigError("energy_uniform_stop_GeV must exceed the start")

        tail = tuple(float(v) for v in section.get("energy_tail_edges_GeV", ()))
        if any(edge <= stop for edge in tail):
            raise ConfigError("energy_tail_edges_GeV must all exceed the uniform stop")
        if list(tail) != sorted(set(tail)):
            raise ConfigError("energy_tail_edges_GeV must be strictly increasing")

        scheme = str(require("costheta_scheme")).lower()
        if scheme not in ("log", "uniform", "piecewise"):
            raise ConfigError(
                f"costheta_scheme must be 'log', 'uniform' or 'piecewise', got {scheme!r}"
            )
        n_costheta = int(require("number_of_costheta_bins"))
        if n_costheta < 2:
            raise ConfigError("number_of_costheta_bins must be at least 2")
        forward = float(section.get("costheta_log_forward_edge", 1e-3))
        if not 0.0 < forward < 2.0:
            raise ConfigError("costheta_log_forward_edge must be in (0, 2)")

        # The notebook's hand-tuned edges (duplicates removed): 2 bins in
        # [-1,0], 6 in [0,0.8], 9 in [0.8,0.9], 9 in [0.9,1]. Used only when
        # costheta_scheme = 'piecewise'.
        default_piecewise = (
            tuple(np.linspace(-1.0, 0.0, 3))
            + tuple(np.linspace(0.0, 0.8, 7))[1:]
            + tuple(np.linspace(0.8, 0.9, 10))[1:]
            + tuple(np.linspace(0.9, 1.0, 10))[1:]
        )
        piecewise = tuple(
            float(v) for v in section.get("costheta_piecewise_edges", default_piecewise)
        )
        if scheme == "piecewise":
            if len(piecewise) < 3:
                raise ConfigError("costheta_piecewise_edges needs at least 3 edges")
            if list(piecewise) != sorted(set(piecewise)):
                raise ConfigError("costheta_piecewise_edges must be strictly increasing")
            if abs(piecewise[0] + 1.0) > 1e-12 or abs(piecewise[-1] - 1.0) > 1e-12:
                raise ConfigError("costheta_piecewise_edges must span [-1, 1]")

        n_z = int(require("number_of_z_bins"))
        if n_z < 1:
            raise ConfigError("number_of_z_bins must be at least 1")
        z_min = float(require("z_min_cm"))
        z_max = float(require("z_max_cm"))
        if not z_max > z_min:
            raise ConfigError("z_max_cm must exceed z_min_cm")

        merged = str(section.get("merged_cells_file", ""))
        if merged and not (base / merged).is_file():
            raise ConfigError(f"merged_cells_file not found: {base / merged}")

        return BinningSettings(
            energy_uniform_start_GeV=start,
            energy_uniform_stop_GeV=stop,
            energy_uniform_width_GeV=width,
            energy_tail_edges_GeV=tail,
            include_energy_overflow=bool(section.get("include_energy_overflow", True)),
            costheta_scheme=scheme,
            number_of_costheta_bins=n_costheta,
            costheta_log_forward_edge=forward,
            costheta_piecewise_edges=piecewise,
            number_of_z_bins=n_z,
            z_min_cm=z_min,
            z_max_cm=z_max,
            merged_cells_file=merged,
        )


@dataclass(frozen=True)
class CellSettings:
    """Cell-quality rules.

    ``minimum_effective_events``: cells of the *source* (photon) sample with
    fewer effective events than this are merged rather than used — small
    non-zero populations silently produce huge weights (plan section 6).
    """

    minimum_effective_events: float

    @staticmethod
    def from_mapping(section: Mapping[str, Any]) -> "CellSettings":
        minimum = float(section.get("minimum_effective_events", 10.0))
        if minimum < 0.0:
            raise ConfigError("minimum_effective_events must be non-negative")
        return CellSettings(minimum_effective_events=minimum)


@dataclass(frozen=True)
class TruthColumnSettings:
    """Column names of the truth tables produced by the generation package."""

    energy: str
    costheta: str
    z: str
    cryostat_weight: str
    dirt_weight: str

    @staticmethod
    def from_mapping(section: Mapping[str, Any]) -> "TruthColumnSettings":
        return TruthColumnSettings(
            energy=str(section.get("energy", "E_gamma")),
            costheta=str(section.get("costheta", "cos_theta")),
            z=str(section.get("z", "z")),
            cryostat_weight=str(section.get("cryostat_weight", "w_event_rate")),
            dirt_weight=str(section.get("dirt_weight", "closure_weight")),
        )


@dataclass(frozen=True)
class MappingConfig:
    """The full validated mapping configuration."""

    binning: BinningSettings
    cells: CellSettings
    truth_columns: TruthColumnSettings
    # which generated events count as signal when truth maps are built:
    #   "inclusive"          - every event (legacy grid)
    #   "coherent_cryostat"  - cryostat: coherent upscatters only (the
    #                          visible-recoil-proton events are not described
    #                          by the single-photon response); dirt: both
    signal_definition: str = "inclusive"
    source_path: Path = field(compare=False, default=Path("."))

    @staticmethod
    def from_mapping(payload: Mapping[str, Any], source_path: Path) -> "MappingConfig":
        if "binning" not in payload:
            raise ConfigError(f"Missing config section 'binning' in {source_path}")
        definition = str(payload.get("signal_definition", "inclusive"))
        if definition not in ("inclusive", "coherent_cryostat", "coherent_all"):
            raise ConfigError(
                f"signal_definition must be 'inclusive', 'coherent_cryostat' or 'coherent_all', "
                f"got {definition!r}"
            )
        return MappingConfig(
            binning=BinningSettings.from_mapping(payload["binning"],
                                                 base=source_path.parent),
            cells=CellSettings.from_mapping(payload.get("cells", {})),
            truth_columns=TruthColumnSettings.from_mapping(
                payload.get("truth_columns", {})
            ),
            signal_definition=definition,
            source_path=source_path,
        )


def load_config(path: str | Path) -> MappingConfig:
    """Read and validate a mapping configuration file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Configuration file {path} is not valid JSON: {exc}") from exc
    return MappingConfig.from_mapping(payload, source_path=path)


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("config.json")
    configuration = load_config(target)
    print(f"Loaded mapping configuration from {configuration.source_path}")
    for section_name in ("binning", "cells", "truth_columns"):
        print(f"\n[{section_name}]")
        for key, value in vars(getattr(configuration, section_name)).items():
            print(f"  {key} = {value!r}")
