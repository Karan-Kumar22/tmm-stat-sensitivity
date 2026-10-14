"""Folded reco-level signal predictions for every grid point.

For each (m4, mu_tr) point: build the truth map from ALL stored batches
(frozen merged binning), reweight the isotropic response sample, and fold
onto its reconstructed quantities. Output per point:

  n_truth     total predicted rate (events at the DarkNews normalisation)
  n_reco      rate with a reconstructed shower (no selection/BDT)
  spectrum    reco shower-energy histogram (edges below)

All 992 rows -> grid_predictions.csv; spectra -> grid_predictions.npz.

    python predict_grid.py [--workers 16]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

GRID = Path("/nevis/riverside/data/karan/tmm_points/grid")
MAPPING_DIR = Path(__file__).resolve().parent
E_RECO_EDGES = np.linspace(0.0, 2.5, 26)   # reco shower energy [GeV]

_STATE: dict = {}


def _init_worker() -> None:
    from config import load_config
    from merged_binning import load_analysis_binning

    config = load_config(MAPPING_DIR / "config.json")
    binning = load_analysis_binning(config)
    source = pd.read_pickle(MAPPING_DIR / "iso_truth.pkl")
    reco = pd.read_pickle(MAPPING_DIR / "iso_reco.pkl")
    merged = source.merge(reco, on=["run", "subrun", "event"], how="left")
    _STATE.update(
        config=config,
        binning=binning,
        source=source,
        src_cell=binning.cell_index(source.E_gamma.to_numpy(),
                                    source.cos_theta.to_numpy(),
                                    source.z.to_numpy()),
        reco_ke=merged.reco_shower_KE.to_numpy(),
    )


def predict(name: str) -> dict:
    from converge_point import load_batch, n_batches_on_disk
    from reweight import compute_weights
    from truth_map import TruthMap

    binning, config = _STATE["binning"], _STATE["config"]
    point = GRID / f"point_{name}"
    n = n_batches_on_disk(point)
    row = {"name": name, "batches": n}
    try:
        tmap = TruthMap.empty(binning)
        for i in range(n):
            frame = load_batch(point, i)
            tmap.fill(frame.E_gamma.to_numpy(), frame.cos_theta.to_numpy(),
                      frame.z.to_numpy(), frame.weight.to_numpy() / n)
        weights = compute_weights(_STATE["source"], tmap, config)
        w = weights.tmm_weight.to_numpy()
        ke = _STATE["reco_ke"]
        reconstructed = np.isfinite(ke) & (ke > 0)
        spectrum, _ = np.histogram(ke[reconstructed], bins=E_RECO_EDGES,
                                   weights=w[reconstructed])
        row.update(
            n_truth=float(tmap.total_rate),
            n_carried=float(w.sum()),
            n_reco=float(w[reconstructed].sum()),
            unsupported=float(weights.attrs["unsupported_fraction"]),
            spectrum=spectrum,
        )
    except Exception as exc:
        row["error"] = str(exc)[:200]
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    names = sorted(p.name.removeprefix("point_")
                   for p in GRID.glob("point_*") if p.is_dir())
    print(f"[predict] {len(names)} points, {args.workers} workers", flush=True)

    rows, spectra = [], {}
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker) as pool:
        futures = {pool.submit(predict, n): n for n in names}
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            spectrum = row.pop("spectrum", None)
            if spectrum is not None:
                spectra[row["name"]] = spectrum
            rows.append(row)
            if i % 100 == 0:
                print(f"[predict] {i}/{len(names)}", flush=True)

    table = pd.DataFrame(rows).sort_values("name")
    table.to_csv(MAPPING_DIR / "grid_predictions.csv", index=False)
    np.savez_compressed(
        MAPPING_DIR / "grid_predictions.npz",
        edges=E_RECO_EDGES,
        names=np.array(sorted(spectra)),
        spectra=np.array([spectra[k] for k in sorted(spectra)]),
    )
    good = table.n_reco.notna()
    print(f"[predict] done: {int(good.sum())} points OK, "
          f"{int((~good).sum())} errors")
    print(f"[predict] n_reco: min {table.n_reco.min():.3g}, "
          f"median {table.n_reco.median():.3g}, max {table.n_reco.max():.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
