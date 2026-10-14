"""Build the background covariance for the TMM sensitivity from NGEM systematics.

Replaces the flat-20% stand-in of sensitivity_scan.py. For the BDT-selected
reco-Enu spectrum of all NGEM background channels this computes, per source,
a FRACTIONAL covariance matrix on the MC prediction (EXT excluded, as in
NGEM's plot_helpers.make_histogram_plot):

  reweightable (from NGEM src/systematics.py, multisim / unisim universes):
    flux (1000 univ), All_UBGenie (500), reinteraction (1000),
    11 GENIE unisims (AxFFCCQEshape, DecayAngMEC, NormCCCOH, NormNCCOH,
    RPA_CCQE, ThetaDelta2NRad, Theta_Delta2Npi, VecFFCCQEshape,
    XSecShape_CCMEC, xsr_scc_Fa3, xsr_scc_Fv3)
  detector variations (optional, needs a detvar dataframe that carries the
    BDT features; see --detvar): LYAtt, LYDown, LYRayleigh, WireModX,
    WireModYZ, Recomb2, SCE
  MC statistics: diag(sum w^2) of the selected background

Universe weights come from the NGEM "presel_weights_df" parquet. Ours was
never built; sanderson's (June 2026) was made from exactly our ROOT file set
and covers every one of our background events (checked: 0 missing in all 7
MC channels), so it is used by default via --weights.

Selection = the one used in sensitivity_scan.py: test-half events with
prob_TMM_iso_reweighted > --cut, binned in wc_kine_reco_Enu [MeV].

Output: systematics/bkg_cov_cut{cut}.npz with
  bins, b_mc (MC-only counts, test-half x2, at --pot), b_ext, b_total,
  mcstat_cov (absolute), frac_cov_<source> for every reweightable source,
  frac_cov_detvar_<var> if --detvar given, plus a breakdown PNG.

    python build_bkg_covariance.py [--cut 0.95] [--pot 6.8e20]
                                   [--bins 0,200,...] [--detvar PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

NGEM = Path("/nevis/houston/home/fk2476/uboone_ngem")
INTER = Path("/nevis/riverside/data/karan/ngem/intermediate_file")
DEFAULT_WEIGHTS = Path(
    "/nevis/riverside/data/sanderson/ngem/intermediate_files/presel_weights_df.parquet")
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "systematics"

POT_DATA = 2.098e19 + 4.038e19          # open data 4a + 4b: NGEM wc_net_weight normalisation
BKG_TYPES = ["nu_overlay", "nue_overlay", "nc_pi0_overlay", "ext",
             "dirt_overlay", "numucc_pi0_overlay", "numuCC_rad_corrected",
             "NC_coherent_1g_reweighted"]
VAR = "wc_kine_reco_Enu"          # overridden by --var
KEYS = ["filename", "filetype", "run", "subrun", "event"]

sys.path.insert(0, str(NGEM))
from src.systematics import (create_rw_frac_cov_matrices,          # noqa: E402
                             create_detvar_frac_cov_matrices,
                             create_universe_histograms)


def parse_bins(text: str | None) -> np.ndarray:
    if text is None:
        return np.linspace(0.0, 2000.0, 21)
    return np.array([float(x) for x in text.split(",")])


DEFAULT_PREDICTIONS = NGEM / "training_outputs/tmm_with_new_files/predictions.parquet"
PREDICTIONS = DEFAULT_PREDICTIONS


BELOW = False


def load_selected_background(cut: float, pot: float) -> pl.DataFrame:
    """Test-half background events passing the BDT cut, with the columns
    NGEM's covariance code needs (filename/run/subrun/event keys, CV and
    spline weights, reco Enu) and wc_net_weight rescaled to `pot`.
    With BELOW=True the selection is inverted (score < cut): sideband."""
    presel = pl.scan_parquet(INTER / "presel_df_train_vars.parquet").select(
        KEYS + ["wc_net_weight", "wc_weight_cv", "wc_weight_spline", VAR])
    scores = pl.scan_parquet(PREDICTIONS).select(
        KEYS + ["used_for_testing", "prob_TMM_iso_reweighted"])
    score_sel = (pl.col("prob_TMM_iso_reweighted") < cut if BELOW
                 else pl.col("prob_TMM_iso_reweighted") > cut)
    df = (presel.join(scores, on=KEYS, how="inner")
          .filter(pl.col("used_for_testing")
                  & pl.col("filetype").is_in(BKG_TYPES)
                  & score_sel)
          .collect())
    # test half -> x2 ; NGEM POT -> target POT
    scale = 2.0 * pot / POT_DATA
    return df.with_columns((pl.col("wc_net_weight") * scale).alias("wc_net_weight"))


def mc_stat_cov(vals: np.ndarray, weights: np.ndarray, bins: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(bins, vals, side="right") - 1
    ok = (idx >= 0) & (idx < len(bins) - 1)
    return np.diag(np.bincount(idx[ok], weights=weights[ok] ** 2, minlength=len(bins) - 1))


def breakdown_plot(bins, b_mc, frac, mcstat, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = 0.5 * (bins[1:] + bins[:-1])
    fig, ax = plt.subplots(figsize=(8, 5))
    tot_rw = sum(frac.values())
    ax.step(bins, np.r_[np.sqrt(np.diag(tot_rw)), np.nan], where="post",
            color="k", lw=2, label="all reweightable")
    for name, m in frac.items():
        d = np.sqrt(np.diag(m))
        if d.max() > 0.02:
            ax.step(bins, np.r_[d, np.nan], where="post", lw=1, label=name)
    with np.errstate(divide="ignore", invalid="ignore"):
        ms = np.sqrt(np.diag(mcstat)) / b_mc
    ax.step(bins, np.r_[np.nan_to_num(ms), np.nan], where="post", ls="--",
            color="gray", label="MC stat")
    ax.set_xlabel("reco $E_\\nu$ [MeV]")
    ax.set_ylabel("fractional uncertainty on background MC")
    ax.set_ylim(0, min(1.5, 1.2 * np.nanmax(np.sqrt(np.diag(tot_rw))) + 0.05))
    ax.legend(fontsize=7, ncol=2)
    ax2 = ax.twinx()
    ax2.step(bins, np.r_[b_mc, np.nan], where="post", color="tab:blue", alpha=0.3)
    ax2.set_ylabel("background MC events / bin", color="tab:blue")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"[cov] wrote {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cut", type=float, default=0.95, help="BDT score cut")
    ap.add_argument("--pot", type=float, default=6.8e20, help="target POT for counts")
    ap.add_argument("--bins", default=None, help="comma-separated reco Enu edges [MeV]")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                    help="NGEM presel_weights_df parquet with universe weights")
    ap.add_argument("--detvar", default=None,
                    help="detvar dataframe parquet WITH BDT scores already applied "
                         "(column prob_TMM_iso_reweighted); optional")
    ap.add_argument("--no-bootstrap", action="store_true",
                    help="detvar: plain (CV-var) outer product instead of bootstrap")
    ap.add_argument("--tag", default=None, help="output name tag (default cut value)")
    ap.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS),
                    help="BDT predictions parquet (default: tmm_with_new_files)")
    ap.add_argument("--var", default="wc_kine_reco_Enu",
                    help="fit variable (e.g. wc_reco_showerKE, then give --bins in GeV)")
    ap.add_argument("--below", action="store_true",
                    help="invert the score selection (score < cut): sideband")
    args = ap.parse_args()
    global PREDICTIONS, VAR, BELOW
    PREDICTIONS = Path(args.predictions)
    VAR = args.var
    BELOW = args.below

    bins = parse_bins(args.bins)
    OUT_DIR.mkdir(exist_ok=True)
    tag = args.tag or f"cut{args.cut:.2f}"

    df = load_selected_background(args.cut, args.pot)
    vals = df[VAR].to_numpy()
    w = df["wc_net_weight"].to_numpy()
    is_ext = (df["filetype"] == "ext").to_numpy()
    b_total, _ = np.histogram(vals, bins=bins, weights=w)
    b_ext, _ = np.histogram(vals[is_ext], bins=bins, weights=w[is_ext])
    b_mc = b_total - b_ext
    print(f"[cov] selected background (test half x2, {args.pot:.3g} POT): "
          f"{len(df)} events, {b_total.sum():.1f} expected "
          f"({b_ext.sum():.1f} EXT, {b_mc.sum():.1f} MC)")
    print("[cov] per channel:")
    for ft, n, s in (df.group_by("filetype").agg(pl.len(), pl.col("wc_net_weight").sum())
                     .sort("filetype").iter_rows()):
        print(f"        {ft:28s} {n:7d} events  {s:8.2f} expected")
    print("[cov] b_mc per bin:", np.round(b_mc, 2))

    mc = df.filter(pl.col("filetype") != "ext")
    mcstat = mc_stat_cov(mc[VAR].to_numpy(), mc["wc_net_weight"].to_numpy(), bins)

    # reweightable systematics: fractional covariances on the MC prediction.
    # (NGEM expects a LazyFrame here: it calls .collect() on the input.)
    weights_lf = pl.scan_parquet(args.weights)
    frac = create_rw_frac_cov_matrices(mc.lazy(), VAR, bins, weights_df=weights_lf,
                                       net_weight_var="wc_net_weight")

    # keep the per-universe FLUX histograms of the background as well: the
    # same universes move the TMM signal (flux_systematics.py), so the
    # signal+background flux covariance must be built from joint universes.
    keys = ["filename", "run", "subrun", "event"]
    flux_w = (mc.select(keys).lazy()
              .join(weights_lf.select(keys + ["flux_all"]), on=keys, how="inner")
              .collect(engine="streaming"))
    mc_in = mc.join(flux_w.select(keys), on=keys, how="semi")
    flux_hists = create_universe_histograms(
        mc_in[VAR].to_numpy(), bins, flux_w["flux_all"].to_numpy(),
        mc_in["wc_net_weight"].to_numpy(), description="flux (bkg universes)")
    print(f"[cov] stored background flux universes: {flux_hists.shape} "
          f"({len(mc_in)}/{len(mc)} MC events matched)")

    out = {"bins": bins, "b_mc": b_mc, "b_ext": b_ext, "b_total": b_total,
           "mcstat_cov": mcstat, "cut": args.cut, "pot": args.pot,
           "n_events": len(df), "flux_universe_hists": flux_hists}
    for name, m in frac.items():
        out[f"frac_cov_{name}"] = m
    tot = sum(frac.values())
    print("[cov] fractional sigma per bin (all reweightable):",
          np.round(np.sqrt(np.diag(tot)), 3))
    for name, m in frac.items():
        print(f"        {name:24s} max {np.sqrt(np.diag(m)).max():.3f}")

    if args.detvar:
        dv = pl.read_parquet(args.detvar).filter(pl.col("prob_TMM_iso_reweighted") > args.cut)
        dfrac = create_detvar_frac_cov_matrices(dv, VAR, bins,
                                                use_detvar_bootstrapping=not args.no_bootstrap)
        for name, m in dfrac.items():
            out[f"frac_cov_detvar_{name}"] = m
        print("[cov] detvar total fractional sigma:",
              np.round(np.sqrt(np.diag(sum(dfrac.values()))), 3))

    path = OUT_DIR / f"bkg_cov_{tag}.npz"
    np.savez(path, **out)
    print(f"[cov] wrote {path}")
    breakdown_plot(bins, b_mc, frac, mcstat, OUT_DIR / f"bkg_cov_{tag}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
