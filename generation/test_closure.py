"""Closure tests (plan sections 5.1 and 5.4). These are the blocking checks
that must pass before production-scale generation.

    python test_closure.py --scratch <dir> [--small-point <dir>] [--fast]

Three parts:

A. Forced-decay self-consistency (no DarkNews, fast). For fabricated events,
   decay distances sampled from the *unconstrained* exponential law land
   inside the liquid with frequency Sum(P_inside)/N — i.e. P_inside really is
   the probability the forcing claims to correct for.

B. Chunk normalisation (plan 5.4). One cryostat run at 2x neval must give the
   same physical rate as two combined neval batches (whose weights the
   combination divided by N=2). Checked on the total rate and on the
   weighted E_gamma spectrum shape.

C. Forced-decay closure against DarkNews (plan 5.1/original 5). At a
   long-lifetime model point, generate dirt twice: once letting DarkNews
   propagate the decay naturally (enforce_prompt=False; keep decays that land
   in the liquid; sum w_event_rate) and once with our forcing
   (sum closure_weight). The two rate estimates must agree statistically.

--fast runs only part A.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import batch_store
from batch_store import Manifest
from config import load_config
from darknews_interface import generate_darknews_sample
from dataframe_tools import get_scalar
from decay_volume import (
    force_dirt_decays_inside_cryostat,
    select_cryostat_decays_in_liquid,
)
import generate
import geometry
from test_decay_volume import make_dirt_dataframe


def fail(message: str) -> None:
    print(f"FAIL  {message}")
    sys.exit(1)


def ok(message: str) -> None:
    print(f"PASS  {message}")


def effective_error(weights: np.ndarray) -> float:
    """Absolute MC error on sum(weights): sqrt(sum w^2)."""
    return float(np.sqrt(np.sum(np.asarray(weights) ** 2)))


# ---------------------------------------------------------------------------
# A. Forced-decay self-consistency (no DarkNews)
# ---------------------------------------------------------------------------


def test_forced_decay_self_consistency(config, rng) -> None:
    n = 200_000
    mass, momentum, lam = 0.48, 1.2, 800.0
    ctau0 = lam * mass / momentum
    df = make_dirt_dataframe(
        config, n=1, mass=mass, ctau0=ctau0, momentum=momentum,
        origin=(0.0, 0.0, -2000.0), direction=(0.0, 0.0, 1.0),
    )
    forced = force_dirt_decays_inside_cryostat(df, config, rng)
    p_inside = float(get_scalar(forced, config.columns.inside_probability)[0])

    # natural decays: unconstrained exponential distances along the same ray
    distances = rng.exponential(scale=lam, size=n)
    x = np.zeros(n)
    y = np.zeros(n)
    z = -2000.0 + distances
    inside = geometry.inside_liquid_cryostat(x, y, z, config.geometry)
    observed = int(inside.sum())
    expected = n * p_inside
    sigma = math.sqrt(n * p_inside * (1.0 - p_inside))
    pull = (observed - expected) / sigma
    if abs(pull) > 5.0:
        fail(
            f"natural-decay MC: {observed} inside vs expected {expected:.1f} "
            f"(pull {pull:.1f}) — P_inside is not the natural probability"
        )
    ok(
        f"A: natural-decay MC gives {observed} inside of {n}, expected "
        f"{expected:.1f} +- {sigma:.1f} (pull {pull:+.2f}) — P_inside closes"
    )


# ---------------------------------------------------------------------------
# B. Chunk normalisation
# ---------------------------------------------------------------------------


def weighted_spectrum(values, weights, edges):
    hist, _ = np.histogram(values, bins=edges, weights=weights)
    err2, _ = np.histogram(values, bins=edges, weights=np.asarray(weights) ** 2)
    return hist, err2


def test_chunk_normalisation(small_point: Path, scratch: Path) -> None:
    manifest_small = Manifest.load(small_point)
    if len(manifest_small.batches("cryostat")) < 2:
        fail(f"{small_point} needs >= 2 cryostat batches for part B")
    config_small = generate.config_from_manifest(manifest_small)

    combined = batch_store.combine_truth_tables(small_point, "cryostat", config_small)
    n_batches = combined.attrs["n_batches"]
    neval_small = manifest_small.neval_per_batch

    # one single run at n_batches x neval, same model point
    big_generation = dataclasses.replace(
        config_small.generation, neval_per_batch=n_batches * neval_small
    )
    config_big = dataclasses.replace(config_small, generation=big_generation)
    big_point = scratch / "closure_big_point"
    manifest_big = Manifest.create_or_load(
        big_point, config_big, mass=manifest_small.mass, tmm=manifest_small.tmm
    )
    if manifest_big.next_index("cryostat") == 0:
        print(f"[part B] generating single {n_batches * neval_small}-eval cryostat run ...")
        generate.run_component_batch("cryostat", 0, manifest_big, config_big, big_point)
    big = pd.read_pickle(batch_store.truth_table_path(big_point, "cryostat", 0))

    sum_combined = float(combined["w_event_rate"].sum())
    sum_big = float(big["w_event_rate"].sum())
    err = math.hypot(
        effective_error(combined["w_event_rate"]), effective_error(big["w_event_rate"])
    )
    diff = sum_combined - sum_big
    rel = diff / sum_big
    pull = diff / err if err > 0 else float("inf")
    if abs(rel) > 0.05:
        fail(
            f"chunk normalisation: combined rate {sum_combined:.6g} vs single-run "
            f"rate {sum_big:.6g} ({rel:+.2%}) — the 1/N correction is wrong"
        )

    # weighted E_gamma spectrum shape
    edges = np.linspace(0.0, 4.0, 11)
    h1, e1 = weighted_spectrum(combined["E_gamma"], combined["w_event_rate"], edges)
    h2, e2 = weighted_spectrum(big["E_gamma"], big["w_event_rate"], edges)
    denom = e1 + e2
    mask = denom > 0
    chi2 = float(np.sum((h1[mask] - h2[mask]) ** 2 / denom[mask]))
    ndf = int(mask.sum())
    if chi2 / ndf > 3.0:
        fail(
            f"chunk normalisation: E_gamma spectra disagree "
            f"(chi2/ndf = {chi2:.1f}/{ndf})"
        )
    ok(
        f"B: combined 2x{neval_small} rate {sum_combined:.6g} vs single "
        f"{n_batches * neval_small}-eval rate {sum_big:.6g} "
        f"({rel:+.2%}, pull {pull:+.2f}); E_gamma shape chi2/ndf = "
        f"{chi2:.1f}/{ndf} — 1/N chunk normalisation closes"
    )


# ---------------------------------------------------------------------------
# C. Forced decay vs DarkNews natural propagation
# ---------------------------------------------------------------------------


def test_forced_vs_natural_darknews(config, scratch: Path, mass: float, tmm: float) -> None:
    base_seed = 424242

    print(f"[part C] dirt at (m4={mass}, mu_tr={tmm:g}): natural propagation run ...")
    natural_raw = generate_darknews_sample(
        config,
        experiment=config.generation.dirt_experiment,
        mass=mass, tmm=tmm,
        seed=batch_store.darknews_seed(base_seed, 0, 0),
        enforce_prompt=False,
        scratch_directory=scratch / "closure_natural_scratch",
    )
    natural_inside = select_cryostat_decays_in_liquid(natural_raw, config)
    w_natural = get_scalar(natural_inside, config.columns.event_weight)
    rate_natural = float(w_natural.sum())
    err_natural = effective_error(w_natural)

    print(f"[part C] forced run ...")
    forced_raw = generate_darknews_sample(
        config,
        experiment=config.generation.dirt_experiment,
        mass=mass, tmm=tmm,
        seed=batch_store.darknews_seed(base_seed, 1, 1),
        enforce_prompt=True,
        scratch_directory=scratch / "closure_forced_scratch",
    )
    rng = batch_store.forced_decay_rng(base_seed, 1)
    forced = force_dirt_decays_inside_cryostat(forced_raw, config, rng, mass=mass)
    w_forced = get_scalar(forced, config.columns.dirt_final_weight)
    rate_forced = float(w_forced.sum())
    err_forced = effective_error(w_forced)

    err = math.hypot(err_natural, err_forced)
    diff = rate_natural - rate_forced
    rel = diff / rate_forced if rate_forced else float("inf")
    pull = diff / err if err > 0 else float("inf")

    print(
        f"[part C] natural: {len(natural_inside)}/{len(natural_raw)} events in "
        f"liquid, rate {rate_natural:.6g} +- {err_natural:.2g}"
    )
    print(
        f"[part C] forced:  {len(forced)}/{len(forced_raw)} events with chord, "
        f"rate {rate_forced:.6g} +- {err_forced:.2g}"
    )

    if len(natural_inside) < 50:
        fail(
            "too few natural inside-decays for a meaningful comparison — "
            "choose a longer-lifetime model point"
        )
    if abs(pull) > 5.0 or abs(rel) > 0.25:
        fail(
            f"forced-decay closure: natural rate {rate_natural:.6g} vs forced "
            f"rate {rate_forced:.6g} ({rel:+.1%}, pull {pull:+.1f})"
        )
    ok(
        f"C: natural {rate_natural:.6g} vs forced {rate_forced:.6g} "
        f"({rel:+.1%}, pull {pull:+.2f}) — forced decay closes against "
        "DarkNews propagation"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=Path(__file__).with_name("config.json"))
    parser.add_argument("--scratch", required=True,
                        help="scratch directory for generated closure samples")
    parser.add_argument("--small-point", default=None,
                        help="existing point dir with >=2 cryostat batches (part B); "
                        "defaults to <scratch>/point_test")
    parser.add_argument("--closure-mass", type=float, default=0.48)
    parser.add_argument("--closure-tmm", type=float, default=1.3e-7,
                        help="long-lifetime point for part C")
    parser.add_argument("--neval", type=int, default=2000,
                        help="neval for the part C dirt runs")
    parser.add_argument("--fast", action="store_true", help="only part A (no DarkNews)")
    args = parser.parse_args()

    scratch = Path(args.scratch)
    config = load_config(args.config)
    rng = np.random.default_rng(20260815)

    test_forced_decay_self_consistency(config, rng)
    if args.fast:
        print("\n--fast: parts B and C skipped.")
        return 0

    small_point = Path(args.small_point) if args.small_point else scratch / "point_test"
    test_chunk_normalisation(small_point, scratch)

    tiny_generation = dataclasses.replace(config.generation, neval_per_batch=args.neval)
    tiny = dataclasses.replace(config, generation=tiny_generation)
    test_forced_vs_natural_darknews(tiny, scratch, args.closure_mass, args.closure_tmm)

    print("\nAll closure tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
