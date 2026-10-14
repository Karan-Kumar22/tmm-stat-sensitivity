"""Sensitivity scan with the background covariance folded in.

chi2(point) = s^T (V_sys + diag(b_total))^{-1} s   (Asimov data = b)

V_sys is assembled from build_bkg_covariance.py output: fractional
covariances (on the MC part of the background) times outer(b_mc, b_mc),
plus the absolute MC-stat matrix. --sources picks which reweightable
sources enter, so systematics can be staged:

  --sources xsec      : All_UBGenie + 11 GENIE unisims + reinteraction
  --sources xsec,flux : + flux
  --sources none      : stat-only closure (must reproduce q/chi2 scan)

    python sensitivity_scan_cov.py --cov systematics/bkg_cov_cut096_shwKE.npz \
           --sources xsec,flux --out-tag _cov_xsecflux
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

NGEM = Path("/nevis/houston/home/fk2476/uboone_ngem")
INTER = Path("/nevis/riverside/data/karan/ngem/intermediate_file")
HERE = Path(__file__).resolve().parent

BKG_TYPES = ["nu_overlay", "nue_overlay", "nc_pi0_overlay", "ext",
             "dirt_overlay", "numucc_pi0_overlay", "numuCC_rad_corrected",
             "NC_coherent_1g_reweighted"]

FLUX_KEYS = ("flux",)
XSEC_KEYS = ("All_UBGenie", "reint", "AxFFCCQEshape", "DecayAngMEC",
             "NormCCCOH", "NormNCCOH", "RPA_CCQE", "ThetaDelta2NRad",
             "Theta_Delta2Npi", "VecFFCCQEshape", "XSecShape_CCMEC",
             "xsr_scc_Fa3", "xsr_scc_Fv3")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov", required=True)
    ap.add_argument("--sources", default="xsec,flux")
    ap.add_argument("--pot", type=float, default=6.8e20)
    ap.add_argument("--cut", type=float, default=0.96)
    ap.add_argument("--var", default="wc_reco_showerKE")
    ap.add_argument("--weights-dir", default=str(HERE / "ngem_weights_v3_cohall"))
    ap.add_argument("--predictions",
                    default=str(NGEM / "training_outputs/tmm_with_new_files/predictions.parquet"))
    ap.add_argument("--out-tag", default="_cov")
    args = ap.parse_args()

    cov = np.load(args.cov)
    bins = cov["bins"]
    b_mc, b_ext, b_total = cov["b_mc"], cov["b_ext"], cov["b_total"]
    assert abs(cov["cut"] - args.cut) < 1e-9 and abs(cov["pot"] - args.pot) < 1e-6

    picked = []
    src = args.sources.lower()
    if src != "none":
        want = set(s.strip() for s in src.split(","))
        for key in cov.files:
            if not key.startswith("frac_cov_"):
                continue
            name = key.removeprefix("frac_cov_")
            grp = ("flux" if any(name.startswith(k) for k in FLUX_KEYS)
                   else "xsec" if any(name.startswith(k) for k in XSEC_KEYS)
                   else None)
            if grp in want:
                picked.append(name)
    frac_tot = sum((cov[f"frac_cov_{n}"] for n in picked),
                   np.zeros((len(bins) - 1, len(bins) - 1)))
    V_sys = frac_tot * np.outer(b_mc, b_mc) + cov["mcstat_cov"]
    if src == "none":
        V_sys = np.zeros_like(V_sys)
    print(f"[cov-scan] sources: {picked or 'NONE'}")
    print(f"[cov-scan] frac sigma/bin (sys only): "
          f"{np.round(np.divide(np.sqrt(np.diag(V_sys)), b_mc, where=b_mc>0), 3)}")

    V_tot = V_sys + np.diag(b_total)
    Vinv = np.linalg.inv(V_tot)

    presel = pl.scan_parquet(INTER / "presel_df_train_vars.parquet").select(
        ["filetype", "run", "subrun", "event", args.var])
    scores = pl.scan_parquet(args.predictions).select(
        ["filetype", "run", "subrun", "event", "used_for_testing",
         "prob_TMM_iso_reweighted"])
    iso = (presel.join(scores, on=["filetype", "run", "subrun", "event"], how="inner")
           .filter(pl.col("used_for_testing")
                   & (pl.col("filetype") == "isotropic_one_gamma_overlay"))
           .collect())
    keys = iso.select(["run", "subrun", "event"])
    sel = iso["prob_TMM_iso_reweighted"].to_numpy() > args.cut
    e_sel = iso[args.var].to_numpy()[sel]
    sig_scale = 2.0 * args.pot / 6.8e20

    rows = []
    for path in sorted(Path(args.weights_dir).glob("tmm_iso_reweighting_weights_*.parquet")):
        tags = path.stem.removeprefix("tmm_iso_reweighting_weights_m")
        m_tag, mu_tag = tags.split("_mu")
        if "check" in m_tag:
            continue
        w = keys.join(pl.read_parquet(path), on=["run", "subrun", "event"],
                      how="left")["tmm_weight"].fill_null(0.0).to_numpy() * sig_scale
        s_hist, _ = np.histogram(e_sel, bins=bins, weights=w[sel])
        rows.append({"mass": float(m_tag), "tmm": float(mu_tag),
                     "n_sig": float(s_hist.sum()),
                     "chi2_cov": float(s_hist @ Vinv @ s_hist)})
        if len(rows) % 200 == 0:
            print(f"[cov-scan] {len(rows)} points", flush=True)

    import pandas as pd
    tab = pd.DataFrame(rows)
    out = HERE / f"sensitivity_scan{args.out_tag}_{args.pot:.3g}.csv"
    tab.to_csv(out, index=False)
    n95 = int((tab.chi2_cov >= 3.84).sum())
    print(f"[cov-scan] excluded at 95% (chi2_cov): {n95}/{len(tab)}")
    print(f"[cov-scan] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
