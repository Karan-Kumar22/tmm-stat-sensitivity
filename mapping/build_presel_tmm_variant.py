"""Build a presel_df_train_vars variant with different TMM signal weights.

Replaces the TMM_iso_reweighted rows of NGEM's preselection dataframe with
rows derived from a given per-event weight parquet (e.g. the coherent-only
grid_v2 flagship). Everything else is byte-identical; train.py consumes the
result via --presel_path.

    ~/TMM_new/.venv-tmm/bin/python build_presel_tmm_variant.py \
        --weights ngem_weights_v2/tmm_iso_reweighting_weights_m0.48_mu2.5e-06.parquet \
        --tag coh0.48 \
        [--out /nevis/riverside/data/karan/ngem/intermediate_file/presel_df_train_vars_tmmcoh.parquet]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import polars as pl

NGEM = Path("/nevis/houston/home/fk2476/uboone_ngem")
INTER = Path("/nevis/riverside/data/karan/ngem/intermediate_file")
sys.path.insert(0, str(NGEM / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True,
                    help="per-event weight parquet (run/subrun/event/tmm_weight/tmm_keep)")
    ap.add_argument("--tag", default="coh0.48",
                    help="m4 tag for the staged weights filename (postprocessing "
                         "resolves the parquet by tag)")
    ap.add_argument("--mu-tag", default="2.5e-06")
    ap.add_argument("--out", default=str(INTER / "presel_df_train_vars_tmmcoh.parquet"))
    args = ap.parse_args()

    # stage the weights where postprocessing expects them, under the tag name
    staged = INTER / f"tmm_iso_reweighting_weights_m{args.tag}_mu{args.mu_tag}.parquet"
    shutil.copyfile(args.weights, staged)
    print(f"[variant] staged {args.weights} -> {staged}")

    from postprocessing import add_tmm_iso_reweighted_events

    base_lf = (pl.scan_parquet(INTER / "presel_df_train_vars.parquet")
               .filter(pl.col("filetype") != "TMM_iso_reweighted"))
    # lazy input -> the function returns ONLY the new TMM rows
    tmm_df = add_tmm_iso_reweighted_events(base_lf, m4_tag=args.tag,
                                           mu_tr_tag=args.mu_tag)
    print(f"[variant] {tmm_df.height} TMM_iso_reweighted rows "
          f"(kept: {int(tmm_df['tmm_keep'].sum()) if 'tmm_keep' in tmm_df.columns else 'n/a'})")

    print("[variant] collecting base dataframe ...")
    base_df = base_lf.collect(engine="streaming")
    out_df = pl.concat([base_df, tmm_df], how="diagonal_relaxed")
    print(f"[variant] total rows {out_df.height} "
          f"({base_df.height} base + {tmm_df.height} TMM)")
    out_df.write_parquet(args.out)
    print(f"[variant] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
