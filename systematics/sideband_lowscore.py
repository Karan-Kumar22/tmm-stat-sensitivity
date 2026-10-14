"""Far-sideband (low BDT score) data/MC comparison on OPEN data.

Region: BDT score < --cut (default 0.5), shower-KE spectrum. Prediction
= all background channels, test half x2, at the open-data POT the NGEM
weights are normalised to (no rescaling). Data = filetype 'data' rows of
the predictions parquet (open beam-on data, never used in training).
Error band = reweightable + MC-stat covariance from --cov (built with
--below at the same cut). GoF: chi2 = (n-b)^T (V_sys + V_CNP)^{-1} (n-b).

    python sideband_lowscore.py --cov systematics/bkg_cov_sb05_shwKE.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

NGEM = Path("/nevis/houston/home/fk2476/uboone_ngem")
INTER = Path("/nevis/riverside/data/karan/ngem/intermediate_file")
HERE = Path(__file__).resolve().parent
POT_DATA = 2.098e19 + 4.038e19

BKG_TYPES = ["nu_overlay", "nue_overlay", "nc_pi0_overlay", "ext",
             "dirt_overlay", "numucc_pi0_overlay", "numuCC_rad_corrected",
             "NC_coherent_1g_reweighted"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov", required=True)
    ap.add_argument("--cut", type=float, default=0.5)
    ap.add_argument("--var", default="wc_reco_showerKE")
    ap.add_argument("--predictions",
                    default=str(NGEM / "training_outputs/tmm_with_new_files/predictions.parquet"))
    args = ap.parse_args()

    cov = np.load(args.cov)
    bins = cov["bins"]
    b_mc, b_ext = cov["b_mc"], cov["b_ext"]
    # covariance was built at --pot; rescale prediction and V to open-data POT
    scale = POT_DATA / float(cov["pot"])
    b_mc, b_ext = b_mc * scale, b_ext * scale
    b_total = b_mc + b_ext
    frac_tot = sum(cov[k] for k in cov.files if k.startswith("frac_cov_"))
    V_sys = frac_tot * np.outer(b_mc, b_mc) + cov["mcstat_cov"] * scale**2

    presel = pl.scan_parquet(INTER / "presel_df_train_vars.parquet").select(
        ["filetype", "run", "subrun", "event", args.var])
    scores = pl.scan_parquet(args.predictions).select(
        ["filetype", "run", "subrun", "event", "prob_TMM_iso_reweighted"])
    data = (presel.join(scores, on=["filetype", "run", "subrun", "event"], how="inner")
            .filter((pl.col("filetype") == "data")
                    & (pl.col("prob_TMM_iso_reweighted") < args.cut))
            .collect())
    n, _ = np.histogram(data[args.var].to_numpy(), bins=bins)
    print(f"[sb] data events in sideband: {n.sum()} ; predicted {b_total.sum():.1f}")

    # CNP statistical term
    with np.errstate(divide="ignore"):
        v_cnp = np.where((n > 0) & (b_total > 0),
                         3.0 / (1.0 / np.maximum(n, 1e-9) + 2.0 / np.maximum(b_total, 1e-9)),
                         np.maximum(b_total, 1e-9) / 2.0 * 3.0)
    V = V_sys + np.diag(v_cnp)
    d = n - b_total
    chi2 = float(d @ np.linalg.inv(V) @ d)
    ndof = int((b_total > 0).sum())
    from scipy.stats import chi2 as chi2dist
    p = float(chi2dist.sf(chi2, ndof))
    print(f"[sb] GoF chi2/ndof = {chi2:.1f}/{ndof}  p = {p:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Nimbus Roman", "font.size": 17,
                         "axes.labelsize": 18, "xtick.labelsize": 15,
                         "ytick.labelsize": 15, "legend.fontsize": 14})
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 8.5), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.06})
    c = 0.5 * (bins[1:] + bins[:-1])
    wdt = np.diff(bins)
    a1.bar(c, b_total, width=wdt, color="0.85", label="prediction")
    a1.bar(c, b_ext, width=wdt, color="0.65", label="EXT (beam off)")
    band = np.sqrt(np.diag(V_sys))
    a1.bar(c, 2 * band, width=wdt, bottom=b_total - band, color="none",
           edgecolor="tab:red", hatch="///", lw=0,
           label="syst. + MC stat")
    a1.errorbar(c, n, yerr=np.sqrt(np.maximum(n, 1)), fmt="ko", ms=5,
                label=f"open data ({POT_DATA:.2g} POT)")
    a1.set_ylabel("events / bin")
    a1.legend(frameon=False)
    a1.text(0.97, 0.97, f"BDT score $<$ {args.cut}\n"
            f"$\\chi^2$/ndof = {chi2:.1f}/{ndof}, $p$ = {p:.2f}",
            transform=a1.transAxes, ha="right", va="top", fontsize=15)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(b_total > 0, n / b_total, np.nan)
        rerr = np.where(b_total > 0, np.sqrt(np.maximum(n, 1)) / b_total, np.nan)
        rband = np.where(b_total > 0, band / b_total, 0.0)
    a2.bar(c, 2 * rband, width=wdt, bottom=1 - rband, color="tab:red",
           alpha=0.25, lw=0)
    a2.errorbar(c, ratio, yerr=rerr, fmt="ko", ms=5)
    a2.axhline(1.0, color="0.4", lw=1)
    a2.set_ylim(0.5, 1.5)
    a2.set_xlabel(r"Reconstructed $E_\gamma$ [GeV]")
    a2.set_ylabel("data / pred.")
    fig.tight_layout()
    out = HERE / "slides/sideband_lowscore.png"
    fig.savefig(out, dpi=180)
    print(f"[sb] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
