"""Re-validate every grid point under the current (adopted) binning.

The stored batches are binning-agnostic; this recomputes, per point, the
standard independent validation (map from batches 0..N-2, judged by batch
N-1) plus the target-side statistics of the full map:

  min_neff  minimum N_eff = (sum w)^2 / sum w^2 over rate-carrying cells
            (>= 0.1% of the point's rate) using ALL batches

No generation, pure post-processing. Output: revalidation.csv.

    python revalidate_grid.py [--workers 16]
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

GRID = Path("/nevis/riverside/data/karan/tmm_points/grid")
MAPPING_DIR = Path(__file__).resolve().parent

_STATE: dict = {}


def _init_worker() -> None:
    from binning import TruthBinning
    from config import load_config

    config = load_config(MAPPING_DIR / "config.json")
    _STATE["config"] = config
    _STATE["binning"] = TruthBinning.from_settings(config.binning)
    _STATE["source"] = pd.read_pickle(MAPPING_DIR / "iso_truth.pkl")


def revalidate(name: str) -> dict:
    from converge_point import load_batch, n_batches_on_disk, validation_chi2

    config, binning, source = (_STATE["config"], _STATE["binning"],
                               _STATE["source"])
    point = GRID / f"point_{name}"
    row = {"name": name, "batches": n_batches_on_disk(point)}
    try:
        chi2, ndf, unsup = validation_chi2(point, config, binning, source)
        row.update(chi2_ndf=(chi2 / ndf if ndf else np.nan), ndf=ndf,
                   unsupported=unsup)
        row["verdict"] = ("EMPTY" if ndf == 0
                          else "PASS" if chi2 / ndf <= 2.0 else "FLOOR")

        frames = [load_batch(point, i) for i in range(row["batches"])]
        df = pd.concat(frames, ignore_index=True)
        idx = binning.cell_index(df.E_gamma.to_numpy(),
                                 df.cos_theta.to_numpy(), df.z.to_numpy())
        ok = idx >= 0
        w = df.weight.to_numpy()[ok]
        sw = np.bincount(idx[ok], weights=w, minlength=binning.n_cells)
        sw2 = np.bincount(idx[ok], weights=w ** 2, minlength=binning.n_cells)
        carrying = sw >= 1.0e-3 * sw.sum()
        if carrying.any():
            neff = sw[carrying] ** 2 / sw2[carrying]
            row["min_neff"] = float(neff.min())
        else:
            row["min_neff"] = np.nan
    except Exception as exc:  # keep the sweep alive; record the failure
        row.update(verdict="ERROR", error=str(exc)[:200])
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    names = sorted(p.name.removeprefix("point_")
                   for p in GRID.glob("point_*") if p.is_dir())
    print(f"[revalidate] {len(names)} points, {args.workers} workers", flush=True)

    fields = ["name", "batches", "chi2_ndf", "ndf", "unsupported",
              "min_neff", "verdict", "error"]
    out = MAPPING_DIR / "revalidation.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_worker) as pool:
            futures = {pool.submit(revalidate, name): name for name in names}
            for future in as_completed(futures):
                writer.writerow(future.result())
                handle.flush()
                done += 1
                if done % 100 == 0:
                    print(f"[revalidate] {done}/{len(names)}", flush=True)

    table = pd.read_csv(out)
    print(table.verdict.value_counts().to_string())
    floor = table[table.verdict == "FLOOR"]
    if len(floor):
        print(f"FLOOR chi2/ndf: median {floor.chi2_ndf.median():.2f}, "
              f"max {floor.chi2_ndf.max():.1f}")
    print(f"points with min_neff < 50: {(table.min_neff < 50).sum()}")
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
