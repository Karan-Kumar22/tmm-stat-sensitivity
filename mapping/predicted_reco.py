"""Predicted reconstructed distributions for one model point.

Combines the per-event reweighting table (reweight.py output) with the
isotropic sample's reconstructed quantities. Every isotropic event carries
the target weight of its truth cell; histogramming *reconstructed* variables
with those weights is the folded prediction

    N_T(reco) = sum_b H_T(b) M(reco | b)

without ever building M explicitly — the source events ARE the response.

No selection is applied (no BDT yet): "reconstructed" here means Wire-Cell
found a shower (reco_showerKE > 0). The unreconstructed fraction is reported
— it is the efficiency loss folded into the prediction.

Outputs: a merged per-event table (truth cell, weight, reco variables) and a
figure with the predicted reco distributions vs their truth counterparts.

    python predicted_reco.py --weights weights_m0.48_mu2.5e-06.pkl
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dataframe_io import atomic_pickle
from source_sample import DEFAULT_FILES, TREE_PATH

RECO_BRANCHES = [
    "run", "subrun", "event",
    "reco_showerKE",
    "reco_showerMomentum",
    "reco_showervtxZ",
]


def extract_reco_file(path: str | Path) -> pd.DataFrame:
    """Per-event reco shower table for one checkout file."""
    import awkward as ak
    import uproot

    path = Path(path)
    with uproot.open(path) as handle:
        arrays = handle[TREE_PATH].arrays(RECO_BRANCHES, library="ak")

    ke = ak.to_numpy(arrays["reco_showerKE"])
    momentum = ak.to_numpy(arrays["reco_showerMomentum"])  # (n, 4): px py pz E
    px, py, pz = momentum[:, 0], momentum[:, 1], momentum[:, 2]
    p = np.sqrt(px**2 + py**2 + pz**2)
    cos = np.divide(pz, p, out=np.full_like(p, np.nan), where=p > 0)

    return pd.DataFrame(
        {
            "run": ak.to_numpy(arrays["run"]),
            "subrun": ak.to_numpy(arrays["subrun"]),
            "event": ak.to_numpy(arrays["event"]),
            "reco_shower_KE": ke,
            "reco_cos_theta": cos,
            "reco_z": ak.to_numpy(arrays["reco_showervtxZ"]),
            "source_file": path.name,
        }
    )


def load_reco(files=DEFAULT_FILES, cache: Path | None = None, rebuild=False) -> pd.DataFrame:
    if cache is not None and cache.is_file() and not rebuild:
        return pd.read_pickle(cache)
    pieces = []
    for path in files:
        print(f"extracting reco from {Path(path).name} ...")
        pieces.append(extract_reco_file(path))
    table = pd.concat(pieces, ignore_index=True)
    if cache is not None:
        atomic_pickle(table, cache)
    return table


def merge_weights_and_reco(weights: pd.DataFrame, reco: pd.DataFrame) -> pd.DataFrame:
    merged = weights.merge(
        reco, on=["run", "subrun", "event", "source_file"], how="left", validate="1:1"
    )
    if merged["reco_shower_KE"].isna().any():
        missing = int(merged["reco_shower_KE"].isna().sum())
        raise RuntimeError(
            f"{missing} weighted events have no reco row — file lists differ?"
        )
    merged["reconstructed"] = merged["reco_shower_KE"] > 0.0
    return merged


def make_figure(merged: pd.DataFrame, truth: pd.DataFrame, output: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUT = "#33404d", "#6b7684"
    TRUTH, RECO = "#8A9BB8", "#C25E32"

    sel = merged[merged["reconstructed"] & (merged["tmm_weight"] > 0)]
    w = sel["tmm_weight"].to_numpy()
    truth_w = truth["tmm_weight"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), facecolor="white")

    panels = [
        (
            axes[0], "shower energy [GeV]",
            truth["E_gamma"], sel["reco_shower_KE"],
            np.linspace(0, 2.5, 26),
        ),
        (
            axes[1], "shower cos θ",
            truth["cos_theta"], sel["reco_cos_theta"],
            np.linspace(-1, 1, 41),
        ),
        (
            axes[2], "vertex z [cm]",
            truth["z"], sel["reco_z"],
            np.linspace(-50, 1100, 24),
        ),
    ]
    for ax, label, truth_values, reco_values, edges in panels:
        ax.hist(truth_values, bins=edges, weights=truth_w, histtype="step",
                lw=1.8, color=TRUTH, label="truth (weighted)")
        ax.hist(reco_values, bins=edges, weights=w, histtype="step",
                lw=1.8, color=RECO, label="predicted reco")
        ax.set_xlabel(label, color=INK)
        ax.set_ylabel("rate / bin", color=INK)
        ax.tick_params(colors=MUT)
        for spine in ax.spines.values():
            spine.set_color("#c9d1d9")
        ax.legend(frameon=False)

    fig.suptitle(title, color=INK)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"figure -> {output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="reweight.py output pickle")
    parser.add_argument("--reco-cache", default=str(Path(__file__).with_name("iso_reco.pkl")))
    parser.add_argument("--source", default=str(Path(__file__).with_name("iso_truth.pkl")))
    parser.add_argument("--output", default=None, help="merged table pickle")
    parser.add_argument("--figure", default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    weights = pd.read_pickle(args.weights)
    reco = load_reco(cache=Path(args.reco_cache), rebuild=args.rebuild)
    merged = merge_weights_and_reco(weights, reco)

    stem = Path(args.weights).stem.replace("weights_", "")
    output = Path(args.output) if args.output else Path(f"predicted_reco_{stem}.pkl")
    figure = Path(args.figure) if args.figure else Path(f"predicted_reco_{stem}.png")

    total = float(merged["tmm_weight"].sum())
    reconstructed = float(merged.loc[merged["reconstructed"], "tmm_weight"].sum())
    print(f"\ntarget rate carried        : {total:.6g}")
    print(f"with a reconstructed shower: {reconstructed:.6g} "
          f"({reconstructed / total:.1%} of the carried rate)")

    # truth side for the overlay: source truth weighted by the same weights
    truth = pd.read_pickle(args.source)[["E_gamma", "cos_theta", "z"]].copy()
    truth["tmm_weight"] = weights["tmm_weight"].to_numpy()
    truth = truth[truth["tmm_weight"] > 0]

    atomic_pickle(merged, output)
    print(f"merged table -> {output}")
    make_figure(
        merged, truth, figure,
        f"Predicted reconstructed distributions — {stem} (no selection)",
    )
