"""Binning study: candidate binnings x scanned model points -> pass/fail.

The decision rule (agreed before running):

  Choose the COARSEST binning that keeps every scanned point below the
  support tolerance; track the energy-overflow occupancy; verify map
  statistics per point with the batch-0-map-vs-batch-1 chi2.

Declared tolerances:
  unsupported rate fraction  < 1%  at every point (source-limited loss)
  rate outside the map edges = 0   (edges must contain everything)
  energy-overflow occupancy  reported (tracked, not a hard cut)
  batch chi2/ndf             ~ 1   (map statistics adequate at 2 batches)

For every candidate and every point the same generated samples are simply
re-histogrammed — no regeneration.

    python binning_scan.py [--points-dir /nevis/riverside/data/karan/tmm_points]
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from binning import TruthBinning
from config import load_config


# ---------------------------------------------------------------------------
# Candidate binnings: variations of BinningSettings around the current one
# ---------------------------------------------------------------------------


def candidates(base):
    """label -> BinningSettings. Ordered roughly coarse -> fine."""
    def vary(**kwargs):
        return dataclasses.replace(base, **kwargs)

    piecewise_fwd = tuple(np.linspace(-1.0, 0.8, 10)) + (0.9, 0.95, 1.0)
    return {
        "uniform10-cos, z5": vary(costheta_scheme="uniform", number_of_costheta_bins=10),
        "uniform20-cos, z5 (current)": vary(costheta_scheme="uniform", number_of_costheta_bins=20),
        "uniform20-cos, z3": vary(costheta_scheme="uniform", number_of_costheta_bins=20,
                                  number_of_z_bins=3),
        "uniform20-cos, z10": vary(costheta_scheme="uniform", number_of_costheta_bins=20,
                                   number_of_z_bins=10),
        "uniform40-cos, z5": vary(costheta_scheme="uniform", number_of_costheta_bins=40),
        "piecewise fwd[.9,.95,1], z5": vary(costheta_scheme="piecewise",
                                            costheta_piecewise_edges=piecewise_fwd),
        "log12-cos, z5": vary(costheta_scheme="log", number_of_costheta_bins=12),
        "uniform20-cos, E50MeV, z5": vary(costheta_scheme="uniform",
                                          number_of_costheta_bins=20,
                                          energy_uniform_width_GeV=0.05),
    }


# ---------------------------------------------------------------------------
# Point loading (directly from batch truth tables; combined/ not required)
# ---------------------------------------------------------------------------


def load_point_batches(point: Path):
    """Per batch index: one frame with (E_gamma, cos_theta, z, weight).

    Batch weights are used raw (each batch alone carries the full physical
    rate); when several batches are merged the caller divides by N.
    """
    frames = {}
    for comp, wcol in (("cryostat", "w_event_rate"), ("dirt", "closure_weight")):
        for path in sorted(point.glob(f"batches/{comp}_*_truth.pkl")):
            index = int(path.name.split("_")[1])
            table = pd.read_pickle(path)
            frame = pd.DataFrame({
                "E_gamma": table["E_gamma"],
                "cos_theta": table["cos_theta"],
                "z": table["z"],
                "weight": table[wcol],
            })
            frames.setdefault(index, []).append(frame)
    return {index: pd.concat(parts, ignore_index=True) for index, parts in frames.items()}


# ---------------------------------------------------------------------------
# Metrics for one (candidate binning, point)
# ---------------------------------------------------------------------------


def audit_point(binning: TruthBinning, batches: dict, source_cells_cache: dict,
                source: pd.DataFrame, minimum: float) -> dict:
    key = id(binning)
    if key not in source_cells_cache:
        source_cells_cache[key] = binning.cell_index(
            source["E_gamma"].to_numpy(),
            source["cos_theta"].to_numpy(),
            source["z"].to_numpy(),
        )
    source_cells = source_cells_cache[key]
    source_counts = np.bincount(source_cells[source_cells >= 0], minlength=binning.n_cells)

    # target from ALL batches (mean of batches = physical rate)
    n_batches = len(batches)
    target = pd.concat(batches.values(), ignore_index=True)
    weights = target["weight"].to_numpy() / n_batches
    cells = binning.cell_index(
        target["E_gamma"].to_numpy(), target["cos_theta"].to_numpy(), target["z"].to_numpy()
    )
    inside = cells >= 0
    outside_rate = float(weights[~inside].sum())
    sumw = np.bincount(cells[inside], weights=weights[inside], minlength=binning.n_cells)
    total = float(sumw.sum()) + outside_rate

    supported = source_counts >= max(minimum, 1.0)
    unsupported_rate = float(sumw[(sumw > 0) & ~supported].sum())

    overflow_rate = 0.0
    if np.isinf(binning.energy_edges[-1]):
        i_e, _, _ = binning.unravel(np.arange(binning.n_cells))
        overflow_rate = float(sumw[i_e == binning.n_energy - 1].sum())

    result = {
        "total_rate": total,
        "outside_fraction": outside_rate / total if total else 0.0,
        "unsupported_fraction": unsupported_rate / total if total else 0.0,
        "overflow_fraction": overflow_rate / total if total else 0.0,
    }

    # batch-0-map vs batch-1 chi2 (map statistics), if two batches exist
    if len(batches) >= 2:
        b0, b1 = batches[0], batches[1]
        cells0 = binning.cell_index(b0["E_gamma"].to_numpy(), b0["cos_theta"].to_numpy(),
                                    b0["z"].to_numpy())
        in0 = cells0 >= 0
        sumw0 = np.bincount(cells0[in0], weights=b0["weight"].to_numpy()[in0],
                            minlength=binning.n_cells)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(supported, sumw0 / source_counts, 0.0)
        w_src = ratio[np.clip(source_cells, 0, None)]
        w_src[source_cells < 0] = 0.0

        chi2 = ndf = 0.0
        for column, edges in (("E_gamma", binning.energy_edges),
                              ("cos_theta", binning.costheta_edges),
                              ("z", binning.z_edges)):
            b_hist, _ = np.histogram(b1[column], bins=edges, weights=b1["weight"])
            b_err2, _ = np.histogram(b1[column], bins=edges, weights=b1["weight"] ** 2)
            s_hist, _ = np.histogram(source[column], bins=edges, weights=w_src)
            s_err2, _ = np.histogram(source[column], bins=edges, weights=w_src**2)
            err2 = b_err2 + s_err2
            usable = (b_hist > 0) & (err2 > 0)
            chi2 += float(((s_hist[usable] - b_hist[usable]) ** 2 / err2[usable]).sum())
            ndf += int(usable.sum())
        result["chi2_ndf"] = chi2 / ndf if ndf else float("nan")
    else:
        result["chi2_ndf"] = float("nan")
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points-dir", default="/nevis/riverside/data/karan/tmm_points")
    parser.add_argument("--config", default=Path(__file__).with_name("config.json"))
    parser.add_argument("--source", default=str(Path(__file__).with_name("iso_truth.pkl")))
    parser.add_argument("--unsupported-tolerance", type=float, default=0.01)
    parser.add_argument("--output", default="binning_scan_results.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    source = pd.read_pickle(args.source)

    points = {}
    for directory in sorted(Path(args.points_dir).glob("point_*")):
        if (directory / "manifest.json").is_file():
            batches = load_point_batches(directory)
            if batches:
                points[directory.name.replace("point_", "")] = batches
    if not points:
        raise SystemExit("no points with batches found")
    print("points:", ", ".join(f"{k} ({len(v)} batches)" for k, v in points.items()), "\n")

    rows = []
    for label, settings in candidates(config.binning).items():
        binning = TruthBinning.from_settings(settings)
        cache: dict = {}
        worst_unsup = worst_chi2 = 0.0
        all_pass = True
        for point_name, batches in points.items():
            metrics = audit_point(binning, batches, cache, source,
                                  config.cells.minimum_effective_events)
            rows.append({"binning": label, "cells": binning.n_cells,
                         "point": point_name, **metrics})
            worst_unsup = max(worst_unsup, metrics["unsupported_fraction"])
            if not np.isnan(metrics["chi2_ndf"]):
                worst_chi2 = max(worst_chi2, metrics["chi2_ndf"])
            if (metrics["unsupported_fraction"] > args.unsupported_tolerance
                    or metrics["outside_fraction"] > 0):
                all_pass = False
        verdict = "PASS" if all_pass else "fail"
        print(f"{label:32s} cells={binning.n_cells:5d} worst unsup "
              f"{worst_unsup:6.2%}  worst chi2/ndf {worst_chi2:5.2f}  {verdict}")

    table = pd.DataFrame(rows)
    table.to_csv(args.output, index=False)
    print(f"\nper-point details -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
