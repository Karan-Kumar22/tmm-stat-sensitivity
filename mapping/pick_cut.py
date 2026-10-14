"""Scan BDT working points for a given predictions parquet and flagship weights.

For each cut: flagship signal s, total background b (test half x2, at the
DarkNews POT 6.8e20), s/sqrt(b), and the Asimov q. Prints a table; pick the
cut for build_bkg_covariance / sensitivity_scan_syst.

    ~/TMM_new/.venv-tmm/bin/python pick_cut.py \
        --predictions ~/uboone_ngem/training_outputs/tmm_coherent/predictions.parquet \
        --flagship ngem_weights_v2/tmm_iso_reweighting_weights_m0.48_mu2.5e-06.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

NGEM = Path("/nevis/houston/home/fk2476/uboone_ngem")
INTER = Path("/nevis/riverside/data/karan/ngem/intermediate_file")
POT_DATA = 2.098e19 + 4.038e19
POT_TARGET = 6.8e20
BKG_TYPES = ["nu_overlay", "nue_overlay", "nc_pi0_overlay", "ext",
             "dirt_overlay", "numucc_pi0_overlay", "numuCC_rad_corrected",
             "NC_coherent_1g_reweighted"]
KEYS = ["filetype", "run", "subrun", "event"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--flagship", required=True)
    args = ap.parse_args()

    presel = pl.scan_parquet(INTER / "presel_df_train_vars.parquet").select(
        KEYS + ["wc_net_weight", "wc_kine_reco_Enu"])
    scores = pl.scan_parquet(args.predictions).select(
        KEYS + ["used_for_testing", "prob_TMM_iso_reweighted"])
    df = (presel.join(scores, on=KEYS, how="inner")
          .filter(pl.col("used_for_testing")).collect())

    bkg = df.filter(pl.col("filetype").is_in(BKG_TYPES))
    b_score = bkg["prob_TMM_iso_reweighted"].to_numpy()
    b_w = bkg["wc_net_weight"].to_numpy() * 2.0 * POT_TARGET / POT_DATA

    iso = df.filter(pl.col("filetype") == "isotropic_one_gamma_overlay")
    fw = (iso.select(["run", "subrun", "event"])
          .join(pl.read_parquet(args.flagship), on=["run", "subrun", "event"],
                how="left")["tmm_weight"].fill_null(0.0).to_numpy() * 2.0)
    s_score = iso["prob_TMM_iso_reweighted"].to_numpy()

    print(f"{'cut':>5} {'s':>8} {'b':>8} {'s/sqrt(b)':>10} {'asimov q':>9}")
    best = (None, -1.0)
    for cut in np.arange(0.50, 0.99, 0.02):
        s = fw[s_score > cut].sum()
        b = b_w[b_score > cut].sum()
        fom = s / np.sqrt(b) if b > 0 else 0.0
        q = 2.0 * ((s + b) * np.log1p(s / b) - s) if b > 0 and s > 0 else 0.0
        print(f"{cut:5.2f} {s:8.1f} {b:8.1f} {fom:10.2f} {q:9.1f}")
        if fom > best[1]:
            best = (float(cut), float(fom))
    print(f"\nbest by s/sqrt(b): cut = {best[0]:.2f} (fom {best[1]:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
