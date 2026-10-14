"""Generate batches for a model point until the mapping validates.

The workflow rule made executable: after each new batch, build the truth map
from all batches except the newest, reweight the photon sample, and compare
against the newest batch (always unseen by the map). Stop when

  - overall chi2/ndf <= --threshold  (the mapping is good), or
  - chi2/ndf stopped improving (< --min-gain relative) — the residual is the
    source-statistics floor / test bias, which more target batches cannot
    fix, or
  - --max-batches is reached.

    python converge_point.py --point <dir> --mass M --tmm MU \\
        [--threshold 2.0] [--max-batches 8] [--min-gain 0.10]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from binning import TruthBinning
from dataframe_io import apply_signal_definition
from merged_binning import load_analysis_binning
from config import load_config
from reweight import compute_weights
from truth_map import TruthMap

GENERATION_DIR = Path("/nevis/riverside/data/karan/tmm_generation")


def n_batches_on_disk(point: Path) -> int:
    """Complete batches only: both components' truth tables present."""
    return min(
        len(sorted(point.glob("batches/cryostat_*_truth.pkl"))),
        len(sorted(point.glob("batches/dirt_*_truth.pkl"))),
    )


def repair_incomplete_batches(point: Path) -> None:
    """Trim a half-written batch left by a mid-generation kill.

    Both components must have the same number of recorded batches; the
    longer component's trailing entries and any stray files are removed.
    Deterministic seeding regenerates the dropped batch identically.
    """
    import json

    manifest_path = point / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text())
    lists = manifest.get("batches", {})
    if not lists:
        return
    n = min(len(entries) for entries in lists.values())
    changed = False
    for component, entries in lists.items():
        while len(entries) > n:
            dropped = entries.pop()
            changed = True
            print(f"[converge] repair: dropped incomplete {component} "
                  f"batch {dropped.get('index')}")
    if changed:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    for path in point.glob("batches/*"):
        try:
            index = int(path.name.split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        if index >= n:
            path.unlink()
            print(f"[converge] repair: removed stray file {path.name}")


_DEFAULT_DEFINITION: str | None = None


def _default_signal_definition() -> str:
    """The signal definition from the mapping config.json, read once."""
    global _DEFAULT_DEFINITION
    if _DEFAULT_DEFINITION is None:
        _DEFAULT_DEFINITION = load_config(
            Path(__file__).with_name("config.json")
        ).signal_definition
    return _DEFAULT_DEFINITION


def load_batch(point: Path, index: int, config=None) -> pd.DataFrame:
    definition = (getattr(config, "signal_definition", None)
                  if config is not None else None) or _default_signal_definition()
    frames = []
    for comp, wcol in (("cryostat", "w_event_rate"), ("dirt", "closure_weight")):
        t = pd.read_pickle(point / "batches" / f"{comp}_{index:05d}_truth.pkl")
        t = apply_signal_definition(t, comp, definition)
        frames.append(pd.DataFrame({
            "E_gamma": t.E_gamma, "cos_theta": t.cos_theta,
            "z": t.z, "weight": t[wcol]}))
    return pd.concat(frames, ignore_index=True)


def validation_chi2(point: Path, config, binning, source) -> tuple[float, int, float]:
    """Map from batches 0..N-2, judged by batch N-1. Returns (chi2, ndf, unsup)."""
    n = n_batches_on_disk(point)
    if n < 2:
        raise RuntimeError("need at least 2 batches to validate")
    train = [load_batch(point, i) for i in range(n - 1)]
    judge = load_batch(point, n - 1)

    truth_map = TruthMap.empty(binning)
    for frame in train:
        truth_map.fill(frame.E_gamma.to_numpy(), frame.cos_theta.to_numpy(),
                       frame.z.to_numpy(), frame.weight.to_numpy() / len(train))
    weights = compute_weights(source, truth_map, config)
    w = weights.tmm_weight.to_numpy()

    n_cells = binning.n_cells
    src_cell = binning.cell_index(source.E_gamma.to_numpy(),
                                  source.cos_theta.to_numpy(),
                                  source.z.to_numpy())
    counts_S = np.bincount(src_cell[src_cell >= 0],
                           minlength=n_cells).astype(float)

    chi2 = ndf = 0
    for column, edges in (("E_gamma", binning.energy_edges),
                          ("cos_theta", binning.costheta_edges),
                          ("z", binning.z_edges)):
        finite = edges[np.isfinite(edges)]
        bh, _ = np.histogram(judge[column], bins=finite, weights=judge.weight)
        be2, _ = np.histogram(judge[column], bins=finite, weights=judge.weight**2)
        sh, _ = np.histogram(source[column], bins=finite, weights=w)
        se2, _ = np.histogram(source[column], bins=finite, weights=w**2)
        # train-map statistical variance propagated through the folding:
        # Var(pred in bin) += sum_cells (n_{bin,cell}/N_S,cell)^2 sumw2(cell).
        # Without it the map is treated as exact, errors are underestimated
        # and chi2 acquires a spurious floor (verdicts FLOOR-ed wholesale
        # when the frozen fine binning was adopted).
        values_s = source[column].to_numpy()
        bin_idx = np.searchsorted(finite, values_s, side="right") - 1
        bin_idx[values_s == finite[-1]] = len(finite) - 2
        inside = (bin_idx >= 0) & (bin_idx < len(finite) - 1) & (src_cell >= 0)
        pair = bin_idx[inside] * n_cells + src_cell[inside]
        n_bc = np.bincount(pair, minlength=(len(finite) - 1) * n_cells)
        n_bc = n_bc.reshape(len(finite) - 1, n_cells)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(counts_S > 0, n_bc / counts_S, 0.0)
        map_var = frac**2 @ truth_map.sumw2
        err2 = be2 + se2 + map_var
        # judge only bins that carry physics: >= 0.1% of the point's rate.
        usable = (bh > 1.0e-3 * bh.sum()) & (err2 > 0)
        chi2 += float(((sh[usable] - bh[usable]) ** 2 / err2[usable]).sum())
        ndf += int(usable.sum())
    return chi2, ndf, float(weights.attrs["unsupported_fraction"])


def measure_min_neff(point: Path, binning, rate_floor: float = 1.0e-3):
    """Min per-cell N_eff over rate-carrying cells, from ALL batches.

    Returns (min_neff, n_carrying_cells); (None, 0) when the point has no
    rate-carrying cells at all (kinematically empty corner).
    """
    n = n_batches_on_disk(point)
    frames = [load_batch(point, i) for i in range(n)]
    df = pd.concat(frames, ignore_index=True)
    if not len(df):
        return None, 0
    idx = binning.cell_index(df.E_gamma.to_numpy(), df.cos_theta.to_numpy(),
                             df.z.to_numpy())
    ok = idx >= 0
    w = df.weight.to_numpy()[ok]
    sw = np.bincount(idx[ok], weights=w, minlength=binning.n_cells)
    sw2 = np.bincount(idx[ok], weights=w ** 2, minlength=binning.n_cells)
    total = sw.sum()
    if rate_floor > 0.0:
        carrying = sw >= rate_floor * total
    else:
        carrying = sw > 0.0          # strict mode: every populated cell
    if total <= 0 or not carrying.any():
        return None, 0
    neff = sw[carrying] ** 2 / sw2[carrying]
    return float(neff.min()), int(carrying.sum())


def run_min_neff(point: Path, args, binning) -> int:
    """Generate until min N_eff over rate-carrying cells >= --min-neff."""
    while True:
        n = n_batches_on_disk(point)
        if n < 1:
            add_one_batch(point, args.mass, args.tmm)
            continue
        value, carrying = measure_min_neff(point, binning, args.rate_floor)
        if value is None:
            print(f"[converge] EMPTY: no rate-carrying cells with {n} batches.")
            return 0
        need = int(np.ceil(n * max(0.0, args.min_neff / value - 1.0)))
        print(f"[converge] {point.name}: {n} batches, min N_eff = {value:.1f} "
              f"over {carrying} carrying cells (target {args.min_neff:g}, "
              f"~{need} more batches)", flush=True)
        if value >= args.min_neff:
            print(f"[converge] NEFF-PASS: min N_eff {value:.1f} >= "
                  f"{args.min_neff:g} with {n} batches.")
            return 0
        if n >= args.max_batches:
            print(f"[converge] LIMIT: reached {n} batches with min N_eff "
                  f"{value:.1f} < {args.min_neff:g}.")
            return 1
        add_one_batch(point, args.mass, args.tmm)


def add_one_batch(point: Path, mass: float, tmm: float) -> None:
    command = [
        sys.executable, "generate.py", "run",
        "--config", "config.json",
        "--outdir", str(point),
        "--mass", str(mass), "--tmm", str(tmm),
        "--batches", "1",
    ]
    subprocess.run(command, cwd=GENERATION_DIR, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", required=True)
    parser.add_argument("--mass", type=float, required=True)
    parser.add_argument("--tmm", type=float, required=True)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--min-gain", type=float, default=0.10,
                        help="stop if chi2/ndf improves by less than this fraction")
    parser.add_argument("--min-neff", type=float, default=None,
                        help="statistics mode: generate until every rate-"
                             "carrying cell has at least this N_eff "
                             "(replaces the chi2 stopping rules)")
    parser.add_argument("--rate-floor", type=float, default=1.0e-3,
                        help="a cell is rate-carrying if it holds at least "
                             "this fraction of the point's rate")
    parser.add_argument("--config", default=Path(__file__).with_name("config.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    binning = load_analysis_binning(config)
    source = pd.read_pickle(Path(__file__).with_name("iso_truth.pkl"))
    point = Path(args.point)
    repair_incomplete_batches(point)

    if args.min_neff is not None:
        return run_min_neff(point, args, binning)

    previous = None
    while True:
        n = n_batches_on_disk(point)
        if n < 2:
            print(f"[converge] {n} batches on disk; generating up to 2 first")
            add_one_batch(point, args.mass, args.tmm)
            continue

        chi2, ndf, unsup = validation_chi2(point, config, binning, source)
        if ndf == 0:
            # essentially zero in-detector rate at this parameter point:
            # nothing to validate, nothing to gain from more batches.
            print(f"[converge] EMPTY: no bins above the rate floor with "
                  f"{n} batches; treating as converged.")
            return 0
        value = chi2 / ndf
        print(f"[converge] {point.name}: {n} batches, chi2/ndf = {chi2:.1f}/{ndf} "
              f"= {value:.2f}, unsupported {unsup:.2%}", flush=True)

        if value <= args.threshold:
            print(f"[converge] PASS: chi2/ndf {value:.2f} <= {args.threshold} "
                  f"with {n} batches. Mapping is good; stop generating.")
            return 0
        if previous is not None and value > previous * (1.0 - args.min_gain):
            print(f"[converge] PLATEAU: {previous:.2f} -> {value:.2f} "
                  f"(gain < {args.min_gain:.0%}). Residual is the source floor / "
                  "test bias; more batches will not fix it. Stopping.")
            return 0
        if n >= args.max_batches:
            print(f"[converge] LIMIT: reached {n} batches without passing. "
                  "Investigate before generating more.")
            return 1

        previous = value
        print(f"[converge] adding batch {n} ...", flush=True)
        add_one_batch(point, args.mass, args.tmm)


if __name__ == "__main__":
    raise SystemExit(main())
