"""Truth-map reweighting: per-event weights for the source sample.

For each truth cell b the ratio

    R(b) = H_T(b) / N_S(b)

turns the source sample's population into the target model point's rate:
assigning every source event the weight R(b of its truth values) makes the
weighted source reproduce H_T cell by cell. The generation density of the
source cancels — only its per-cell population and coverage matter.

Guards (config section ``cells``):

- target rate in cells with NO source events is *unsupported*: those cells
  contribute nothing, and their rate fraction is reported (plan section 7 —
  a standing diagnostic with a stated tolerance).
- cells with fewer source events than ``minimum_effective_events`` produce
  large noisy ratios; they are treated as unsupported as well (weight 0,
  rate reported) rather than silently amplified (plan section 6).

Output: one row per source event —
    run, subrun, event, source_file, cell, tmm_weight
row order identical to the input source table. Global effective-statistics
numbers (the D2 flavour of plan section 4) are printed and stored in attrs.

    python reweight.py --truth-map <point>/combined/truth_map.npz \\
                       --source iso_truth.pkl --output weights_<point>.pkl
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from binning import TruthBinning
from merged_binning import load_analysis_binning
from config import MappingConfig, load_config
from dataframe_io import atomic_pickle
from truth_map import TruthMap


class ReweightError(RuntimeError):
    """Raised on inconsistent inputs."""


def compute_weights(
    source: pd.DataFrame,
    truth_map: TruthMap,
    config: MappingConfig,
) -> pd.DataFrame:
    """The per-event weight table (rows aligned with ``source``)."""
    binning = truth_map.binning
    columns = config.truth_columns

    cells = binning.cell_index(
        source[columns.energy].to_numpy(),
        source[columns.costheta].to_numpy(),
        source[columns.z].to_numpy(),
    )
    inside = cells >= 0
    n_outside = int((~inside).sum())

    source_counts = np.bincount(cells[inside], minlength=binning.n_cells)

    minimum = config.cells.minimum_effective_events
    populated_target = truth_map.sumw > 0.0
    supported = source_counts >= max(minimum, 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(supported, truth_map.sumw / source_counts, 0.0)

    weights = np.zeros(len(source))
    weights[inside] = ratio[cells[inside]]

    # -- diagnostics ---------------------------------------------------------
    target_total = truth_map.total_rate
    unsupported = populated_target & ~supported
    unsupported_rate = float(truth_map.sumw[unsupported].sum())
    carried = float(weights.sum())

    positive = weights > 0.0
    sum_w = float(weights[positive].sum())
    sum_w2 = float((weights[positive] ** 2).sum())
    n_eff = sum_w**2 / sum_w2 if sum_w2 > 0.0 else 0.0
    max_weight_fraction = float(weights.max() / sum_w) if sum_w > 0.0 else 0.0

    result = pd.DataFrame(
        {
            "run": source["run"].to_numpy(),
            "subrun": source["subrun"].to_numpy(),
            "event": source["event"].to_numpy(),
            "source_file": source["source_file"].to_numpy(),
            "cell": cells,
            "tmm_weight": weights,
        }
    )
    result.attrs = {
        "target_metadata": dict(truth_map.metadata),
        "minimum_effective_events": minimum,
        "target_rate": target_total,
        "carried_rate": carried,
        "unsupported_rate": unsupported_rate,
        "unsupported_fraction": unsupported_rate / target_total if target_total else 0.0,
        "unsupported_cells": int(unsupported.sum()),
        "source_events": int(len(source)),
        "source_events_outside_map": n_outside,
        "events_with_nonzero_weight": int(positive.sum()),
        "effective_statistics": n_eff,
        "max_single_weight_fraction": max_weight_fraction,
    }
    return result


def report(weights: pd.DataFrame) -> str:
    a = weights.attrs
    lines = [
        f"target rate                : {a['target_rate']:.6g}",
        f"carried by weighted source : {a['carried_rate']:.6g}",
        f"unsupported target rate    : {a['unsupported_rate']:.3g} "
        f"({a['unsupported_fraction']:.2%} in {a['unsupported_cells']} cells)",
        f"source events              : {a['source_events']:,} "
        f"({a['source_events_outside_map']} outside the map)",
        f"events with weight > 0     : {a['events_with_nonzero_weight']:,}",
        f"effective statistics N_eff : {a['effective_statistics']:.0f}",
        f"largest single-event share : {a['max_single_weight_fraction']:.2%} of the total weight",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-map", required=True, help="target truth_map.npz")
    parser.add_argument(
        "--source", default=str(Path(__file__).with_name("iso_truth.pkl"))
    )
    parser.add_argument("--config", default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", required=True, help="output weight table pickle")
    args = parser.parse_args()

    configuration = load_config(args.config)
    target = TruthMap.load(
        args.truth_map, load_analysis_binning(configuration)
    )
    source = pd.read_pickle(args.source)

    table = compute_weights(source, target, configuration)
    atomic_pickle(table, args.output)
    print(f"Reweighting {args.source} -> {args.truth_map}\n")
    print(report(table))
    print(f"\nsaved -> {args.output}")
