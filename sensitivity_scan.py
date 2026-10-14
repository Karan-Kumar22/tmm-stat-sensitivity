"""Actual TMM sensitivity scan using the NGEM BDT and backgrounds.

Signal at each of the 992 grid points: the isotropic single-photon events
that pass WC generic preselection, carry a trained TMM-vs-background BDT
score, and are reweighted by this point's truth-map weights (DarkNews
normalisation = 6.8e20 POT). Background: all NGEM MC channels at the same
POT. Test-half events only (upweighted by 2) so BDT scores are unbiased.

Per point: reco-Enu spectrum after the BDT cut -> Asimov discovery/exclusion
q = 2 sum[(s+b) ln(1+s/b) - s], and a systematics-aware chi2 with a flat
fractional background uncertainty. 90% CL contour at q = 2.71.

    python sensitivity_scan.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

NGEM = Path("/nevis/houston/home/fk2476/uboone_ngem")
INTER = Path("/nevis/riverside/data/karan/ngem/intermediate_file")

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("pot", nargs="?", type=float, default=6.8e20)
_ap.add_argument("--weights-dir", default=None)
_ap.add_argument("--predictions", default=None)
_ap.add_argument("--out-tag", default="")
_ap.add_argument("--flat-sys", type=float, default=0.20)
_ap.add_argument("--var", default="wc_kine_reco_Enu")
_ap.add_argument("--emax", type=float, default=2000.0)
_ap.add_argument("--cut", type=float, default=None,
                 help="fixed BDT cut; skips the flagship s/sqrt(b) scan")
_args = _ap.parse_args()
WEIGHTS = (Path(_args.weights_dir) if _args.weights_dir
           else Path(__file__).resolve().parent / "ngem_weights")
POT_TARGET = _args.pot
POT_DATA = 2.098e19 + 4.038e19          # open data 4a + 4nota
BKG_SCALE = POT_TARGET / POT_DATA
SIG_SCALE = POT_TARGET / 6.8e20         # weight parquets are at DarkNews POT
E_BINS = np.linspace(0.0, _args.emax, 21)
FLAT_BKG_SYS = _args.flat_sys

BKG_TYPES = ["nu_overlay", "nue_overlay", "nc_pi0_overlay", "ext",
             "dirt_overlay", "numucc_pi0_overlay", "numuCC_rad_corrected",
             "NC_coherent_1g_reweighted"]


def load_frames():
    presel = pl.scan_parquet(INTER / "presel_df_train_vars.parquet").select(
        ["filetype", "run", "subrun", "event", "wc_net_weight",
         _args.var])
    scores = pl.scan_parquet(
        _args.predictions or NGEM / "training_outputs/tmm_with_new_files/predictions.parquet"
    ).select(["filetype", "run", "subrun", "event", "used_for_testing",
              "prob_TMM_iso_reweighted"])
    df = presel.join(scores, on=["filetype", "run", "subrun", "event"],
                     how="inner").filter(pl.col("used_for_testing")).collect()

    bkg = df.filter(pl.col("filetype").is_in(BKG_TYPES))
    iso = df.filter(pl.col("filetype") == "isotropic_one_gamma_overlay")
    return (
        # backgrounds: test half x2, scaled to target POT
        bkg[_args.var].to_numpy(),
        bkg["wc_net_weight"].to_numpy() * 2.0 * BKG_SCALE,
        bkg["prob_TMM_iso_reweighted"].to_numpy(),
        # iso signal carriers: key + kinematics + score (weights per point)
        iso.select(["run", "subrun", "event"]),
        iso[_args.var].to_numpy(),
        iso["prob_TMM_iso_reweighted"].to_numpy(),
    )


def asimov_q(s, b):
    ok = (b > 0) & (s > 0)
    return float(2.0 * ((s[ok] + b[ok]) * np.log1p(s[ok] / b[ok]) - s[ok]).sum())


def chi2_sys(s, b, fsys):
    ok = b > 0
    return float((s[ok] ** 2 / (b[ok] + (fsys * b[ok]) ** 2)).sum())


def main() -> int:
    bkg_e, bkg_w, bkg_score, iso_keys, iso_e, iso_score = load_frames()
    print(f"[sens] backgrounds: {len(bkg_e)} test events, "
          f"{bkg_w.sum():.0f} expected at {POT_TARGET:.2g} POT")
    print(f"[sens] signal carriers: {len(iso_e)} scored iso events")

    # choose the BDT working point on the flagship signal
    flagship = pl.read_parquet(
        WEIGHTS / "tmm_iso_reweighting_weights_m0.48_mu2.5e-06.parquet")
    fw = iso_keys.join(flagship, on=["run", "subrun", "event"], how="left"
                       )["tmm_weight"].fill_null(0.0).to_numpy() * 2.0 * SIG_SCALE
    best = (None, -1.0)
    for cut in np.arange(0.30, 0.985, 0.01):
        s = fw[iso_score > cut].sum()
        b = bkg_w[bkg_score > cut].sum()
        fom = s / np.sqrt(b) if b > 0 else 0.0
        if fom > best[1]:
            best = (float(cut), float(fom))
    cut = _args.cut if _args.cut is not None else best[0]
    s_tot = fw[iso_score > cut].sum()
    b_tot = bkg_w[bkg_score > cut].sum()
    print(f"[sens] BDT cut {cut:.2f}: flagship s={s_tot:.1f}, b={b_tot:.1f}, "
          f"s/sqrt(b)={best[1]:.2f}")

    sel_b = bkg_score > cut
    b_hist, _ = np.histogram(bkg_e[sel_b], bins=E_BINS, weights=bkg_w[sel_b])
    sel_i = iso_score > cut
    iso_e_sel = iso_e[sel_i]

    rows = []
    for path in sorted(WEIGHTS.glob("tmm_iso_reweighting_weights_*.parquet")):
        tags = path.stem.removeprefix("tmm_iso_reweighting_weights_m")
        m_tag, mu_tag = tags.split("_mu")
        if "check" in m_tag:
            continue
        w = iso_keys.join(pl.read_parquet(path),
                          on=["run", "subrun", "event"], how="left"
                          )["tmm_weight"].fill_null(0.0).to_numpy() * 2.0 * SIG_SCALE
        s_hist, _ = np.histogram(iso_e_sel, bins=E_BINS, weights=w[sel_i])
        rows.append({
            "mass": float(m_tag), "tmm": float(mu_tag),
            "n_sig": float(s_hist.sum()),
            "q_stat": asimov_q(s_hist, b_hist),
            "chi2_sys": chi2_sys(s_hist, b_hist, FLAT_BKG_SYS),
        })
        if len(rows) % 150 == 0:
            print(f"[sens] {len(rows)} points scanned", flush=True)

    import pandas as pd
    table = pd.DataFrame(rows)
    table.to_csv(f"sensitivity_scan{_args.out_tag}_{POT_TARGET:.3g}.csv", index=False)
    for col in ("q_stat", "chi2_sys"):
        n = int((table[col] >= 2.71).sum())
        print(f"[sens] points excluded at 90% CL ({col}): {n}/{len(table)}")
    print(f"[sens] wrote sensitivity_scan{_args.out_tag}_{POT_TARGET:.3g}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
