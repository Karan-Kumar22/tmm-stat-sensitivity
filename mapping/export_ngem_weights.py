"""Export per-point TMM weights in the NGEM framework's parquet format.

For a model point: build the truth map from ALL stored batches (frozen
merged binning), compute per-event weights for the isotropic sample, and
write {out}/tmm_iso_reweighting_weights_m{m4}_mu{mu}.parquet with the
schema NGEM's add_tmm_iso_reweighted_events expects:

    run (i32), subrun (i32), event (i32), tmm_weight (f64), tmm_keep (bool)

Weights are at the DarkNews normalisation (6.8e20 POT), matching the
existing flagship file; tmm_keep = (tmm_weight > 0), which reproduces the
flagship convention exactly (kept == weight>0 there).

    python export_ngem_weights.py --point <dir> [--out DIR] [--tags M MU]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from config import load_config
from converge_point import load_batch, n_batches_on_disk
from merged_binning import load_analysis_binning
from reweight import compute_weights
from truth_map import TruthMap

MAPPING_DIR = Path(__file__).resolve().parent


def export_point(point: Path, out_dir: Path, m4_tag: str, mu_tag: str,
                 config=None, binning=None, source=None) -> dict:
    config = config or load_config(MAPPING_DIR / "config.json")
    binning = binning or load_analysis_binning(config)
    if source is None:
        source = pd.read_pickle(MAPPING_DIR / "iso_truth.pkl")

    n = n_batches_on_disk(point)
    tmap = TruthMap.empty(binning)
    for i in range(n):
        frame = load_batch(point, i)
        tmap.fill(frame.E_gamma.to_numpy(), frame.cos_theta.to_numpy(),
                  frame.z.to_numpy(), frame.weight.to_numpy() / n)
    weights = compute_weights(source, tmap, config)
    w = weights.tmm_weight.to_numpy()

    table = pl.DataFrame({
        "run": source.run.to_numpy().astype(np.int32),
        "subrun": source.subrun.to_numpy().astype(np.int32),
        "event": source.event.to_numpy().astype(np.int32),
        "tmm_weight": w.astype(np.float64),
        "tmm_keep": w > 0.0,
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"tmm_iso_reweighting_weights_m{m4_tag}_mu{mu_tag}.parquet"
    table.write_parquet(out)
    return {
        "path": str(out), "batches": n,
        "sum_weight": float(w.sum()), "kept": int((w > 0).sum()),
        "unsupported": float(weights.attrs["unsupported_fraction"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", required=True)
    parser.add_argument("--out", default=str(MAPPING_DIR / "ngem_weights"))
    parser.add_argument("--tags", nargs=2, metavar=("M4", "MU"),
                        help="filename tags; default derived from --point name")
    args = parser.parse_args()

    point = Path(args.point)
    if args.tags:
        m4_tag, mu_tag = args.tags
    else:
        name = point.name.removeprefix("point_")
        m4_tag, mu_tag = name.split("_mu")
        m4_tag = m4_tag.removeprefix("m")
    result = export_point(point, Path(args.out), m4_tag, mu_tag)
    for key, value in result.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
