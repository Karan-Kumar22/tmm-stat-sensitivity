"""Tests for batch_store.py — seeds, manifest lifecycle, combination.

    python test_batch_store.py [--config config.json]

Uses a temporary point directory; no DarkNews involved.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import batch_store
from batch_store import Manifest, ManifestError
from config import load_config
from dataframe_tools import atomic_pickle


def fail(message: str) -> None:
    print(f"FAIL  {message}")
    sys.exit(1)


def ok(message: str) -> None:
    print(f"PASS  {message}")


def test_seed_derivation() -> None:
    # deterministic: same inputs, same seed
    a = batch_store.darknews_seed(17001, 3, batch_store.STREAM_DIRT_DARKNEWS)
    b = batch_store.darknews_seed(17001, 3, batch_store.STREAM_DIRT_DARKNEWS)
    if a != b:
        fail("seed derivation is not deterministic")

    # distinct across batches, streams, and base seeds
    seeds = {
        batch_store.darknews_seed(base, index, stream)
        for base in (17001, 17002)
        for index in range(200)
        for stream in (0, 1, 2)
    }
    if len(seeds) != 2 * 200 * 3:
        fail(f"seed collisions: {2*200*3 - len(seeds)} among 1200 derivations")

    # in range for DarkNews
    if not all(1 <= s <= 2**31 - 1 for s in seeds):
        fail("derived seed outside [1, 2^31 - 1]")

    # forced-decay RNGs of different batches produce different streams
    r0 = batch_store.forced_decay_rng(17001, 0).random(4)
    r1 = batch_store.forced_decay_rng(17001, 1).random(4)
    r0_again = batch_store.forced_decay_rng(17001, 0).random(4)
    if np.array_equal(r0, r1):
        fail("forced-decay RNG identical across batches")
    if not np.array_equal(r0, r0_again):
        fail("forced-decay RNG not reproducible")

    ok("seed derivation: deterministic, collision-free (1200 draws), in-range, "
       "independent per batch")


def test_manifest_lifecycle(config) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        point = Path(tmp) / "point"

        # create
        manifest = Manifest.create_or_load(point, config, mass=0.48, tmm=2.5e-6)
        if manifest.next_index("cryostat") != 0:
            fail("fresh manifest should have next_index 0")

        # reload with matching parameters: fine
        again = Manifest.create_or_load(point, config, mass=0.48, tmm=2.5e-6)
        if again.data != manifest.data:
            fail("reloaded manifest differs from created one")

        # wrong mass: refused
        try:
            Manifest.create_or_load(point, config, mass=0.50, tmm=2.5e-6)
            fail("mass mismatch was not refused")
        except ManifestError:
            pass

        # changed config (different neval): refused
        changed = dataclasses.replace(
            config,
            generation=dataclasses.replace(config.generation, neval_per_batch=999),
        )
        try:
            Manifest.create_or_load(point, changed, mass=0.48, tmm=2.5e-6)
            fail("config change was not refused")
        except ManifestError:
            pass

        # record batches; out-of-order refused
        manifest.record_batch(
            "cryostat", {"index": 0, "kept_rows": 3, "darknews_seed": 12345}
        )
        if manifest.next_index("cryostat") != 1:
            fail("next_index did not advance")
        try:
            manifest.record_batch("cryostat", {"index": 5, "kept_rows": 1})
            fail("out-of-order batch index was not refused")
        except ManifestError:
            pass

        # persistence across a fresh load
        reloaded = Manifest.load(point)
        if reloaded.next_index("cryostat") != 1 or reloaded.next_index("dirt") != 0:
            fail("recorded batches not persisted")

    ok("manifest lifecycle: create, reload, parameter/config locks, "
       "ordered batch recording, persistence")


def make_truth_table(rows: int, weight: float, with_dirt_columns: bool) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "E_gamma": np.linspace(0.1, 1.0, rows),
            "cos_theta": np.linspace(0.9, 1.0, rows),
            "z": np.linspace(100.0, 900.0, rows),
            "w_event_rate": np.full(rows, weight),
        }
    )
    if with_dirt_columns:
        table["P_inside"] = np.full(rows, 1.0e-3)
        table["closure_weight"] = table["w_event_rate"] * table["P_inside"]
    return table


def test_combination(config) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        point = Path(tmp) / "point"
        manifest = Manifest.create_or_load(point, config, mass=0.48, tmm=2.5e-6)

        # three dirt batches, each with total w_event_rate = rows * 4.0
        for index in range(3):
            table = make_truth_table(rows=10, weight=4.0, with_dirt_columns=True)
            atomic_pickle(table, batch_store.truth_table_path(point, "dirt", index))
            manifest.record_batch(
                "dirt", {"index": index, "kept_rows": 10, "darknews_seed": 1 + index}
            )

        combined = batch_store.combine_truth_tables(point, "dirt", config)
        if len(combined) != 30:
            fail(f"combined table has {len(combined)} rows, expected 30")

        # each batch summed to 40.0; combined must sum to 40.0, not 120.0
        total = float(combined["w_event_rate"].sum())
        if abs(total - 40.0) > 1e-12:
            fail(f"combined w_event_rate sums to {total}, expected 40.0 (1/N failed)")
        total_closure = float(combined["closure_weight"].sum())
        if abs(total_closure - 40.0e-3) > 1e-15:
            fail(f"combined closure_weight sums to {total_closure}, expected 0.04")

        # P_inside must NOT be scaled
        if not np.allclose(combined["P_inside"], 1.0e-3):
            fail("P_inside was scaled by 1/N — it must not be")

        if combined.attrs["n_batches"] != 3:
            fail("combined attrs missing n_batches")

        # a missing batch file must be detected
        batch_store.truth_table_path(point, "dirt", 1).unlink()
        try:
            batch_store.combine_truth_tables(point, "dirt", config)
            fail("missing batch file was not detected")
        except ManifestError:
            pass

    ok("combination: concatenates, scales w_event_rate and closure_weight by 1/N, "
       "leaves P_inside alone, detects missing files")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=Path(__file__).with_name("config.json"))
    args = parser.parse_args()
    config = load_config(args.config)

    test_seed_derivation()
    test_manifest_lifecycle(config)
    test_combination(config)
    print("\nAll batch-store tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
