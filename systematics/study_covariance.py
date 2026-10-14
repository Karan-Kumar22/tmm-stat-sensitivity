"""Study script: build the flux covariance from scratch and apply it.

Follows covariance_howto.pdf section 4 exactly, one step at a time,
printing every checkpoint. Self-contained: only reads the three input
parquets and the signal weight library. Run it, then rewrite it.

    python study_covariance.py
"""

import numpy as np
import polars as pl

# ----------------------------------------------------------------- inputs
EVENTS = "/nevis/riverside/data/karan/ngem/intermediate_file/presel_df_train_vars.parquet"
SCORES = ("/nevis/houston/home/fk2476/uboone_ngem/"
          "training_outputs/tmm_with_new_files/predictions.parquet")
WEIGHTS = ("/nevis/riverside/data/sanderson/ngem/"
           "intermediate_files/presel_weights_df.parquet")
SIGLIB = "/nevis/riverside/data/karan/tmm_mapping/ngem_weights_v3_cohall"
NPZ = "/nevis/riverside/data/karan/tmm_mapping/systematics/bkg_cov_cut096_shwKE.npz"

BKG = ["nu_overlay", "nue_overlay", "nc_pi0_overlay", "ext", "dirt_overlay",
       "numucc_pi0_overlay", "numuCC_rad_corrected", "NC_coherent_1g_reweighted"]
KEYS = ["filename", "filetype", "run", "subrun", "event"]
CUT = 0.96
POT = 6.8e20
POT_OPEN = 2.098e19 + 4.038e19          # the exposure wc_net_weight is normalised to
BINS = np.linspace(0.0, 2.0, 21)        # shower KE [GeV], 20 bins

# ------------------------------------------------- step 1: the spectrum b
# Join events with scores; keep test-half background events above the cut.
# Test half -> x2; open-data POT -> 6.8e20.
events = pl.scan_parquet(EVENTS).select(
    KEYS + ["wc_net_weight", "wc_reco_showerKE"])
scores = pl.scan_parquet(SCORES).select(
    KEYS + ["used_for_testing", "prob_TMM_iso_reweighted"])

df = (events.join(scores, on=KEYS, how="inner")
      .filter(pl.col("used_for_testing")
              & pl.col("filetype").is_in(BKG)
              & (pl.col("prob_TMM_iso_reweighted") > CUT))
      .collect())

w = df["wc_net_weight"].to_numpy() * 2.0 * POT / POT_OPEN
e = df["wc_reco_showerKE"].to_numpy()
b_total, _ = np.histogram(e, bins=BINS, weights=w)

print(f"STEP 1  selected background: {df.height} events, "
      f"sum b = {b_total.sum():.1f}   [checkpoint: 3818, 24.2]")

# --------------------------------------- step 2: the 1000 universe spectra
# EXT carries no generator systematics -> MC only.
# Each MC event has flux_all = a list of 1000 universe weights.
mc = df.filter(pl.col("filetype") != "ext")
joined = (mc.lazy()
          .join(pl.scan_parquet(WEIGHTS).select(
              ["filename", "run", "subrun", "event", "flux_all"]),
              on=["filename", "run", "subrun", "event"], how="inner")
          .collect())
print(f"STEP 2  weights matched for {joined.height}/{mc.height} MC events")

U = np.stack(joined["flux_all"].to_numpy())           # (n_events, 1000)
# standard weight sanitization: broken weights count as "no variation"
U = np.where(np.isfinite(U) & (U >= 0) & (U <= 30), U, 1.0)

# the joined table already carries each event's weight and energy in the
# SAME row order as U -- never realign through a second join
w_mcj = joined["wc_net_weight"].to_numpy() * 2.0 * POT / POT_OPEN
e_mcj = joined["wc_reco_showerKE"].to_numpy()
b_mc, _ = np.histogram(e_mcj, bins=BINS, weights=w_mcj)

bu = np.stack([np.histogram(e_mcj, bins=BINS, weights=w_mcj * U[:, u])[0]
               for u in range(U.shape[1])], axis=1)   # (20 bins, 1000 universes)
print(f"        universe spectra shape: {bu.shape}")

# ------------------------------------------------- step 3: the covariance
d = bu - b_mc[:, None]
V_flux = d @ d.T / bu.shape[1]                        # eq. (3) of the PDF

frac = np.sqrt(np.diag(V_flux)) / np.maximum(b_mc, 1e-9)
print(f"STEP 3  flux fractional sigma per bin, max = {frac.max():.2f}"
      f"   [checkpoint: ~0.37]")

ref = np.load(NPZ)
V_ref = ref["frac_cov_flux"] * np.outer(ref["b_mc"], ref["b_mc"])
dev = np.abs(np.diag(V_flux) - np.diag(V_ref)) / np.maximum(np.diag(V_ref), 1e-12)
print(f"        vs production npz: median diagonal deviation = {np.median(dev):.3f}")

# --------------------------- step 4: assemble V_sys, chi2 for one point
frac_tot = sum(ref[k] for k in ref.files if k.startswith("frac_cov_"))
V_sys = frac_tot * np.outer(ref["b_mc"], ref["b_mc"]) + ref["mcstat_cov"]

# flagship signal spectrum: iso test events, same cut, flagship weights x2
iso = (events.join(scores, on=KEYS, how="inner")
       .filter(pl.col("used_for_testing")
               & (pl.col("filetype") == "isotropic_one_gamma_overlay"))
       .collect())
fw = (iso.select(["run", "subrun", "event"])
      .join(pl.read_parquet(f"{SIGLIB}/tmm_iso_reweighting_weights_m0.48_mu2.5e-06.parquet"),
            on=["run", "subrun", "event"], how="left")
      )["tmm_weight"].fill_null(0.0).to_numpy() * 2.0
sel = iso["prob_TMM_iso_reweighted"].to_numpy() > CUT
s, _ = np.histogram(iso["wc_reco_showerKE"].to_numpy()[sel], bins=BINS,
                    weights=fw[sel])

chi2_stat = float((s**2 / np.maximum(b_total, 1e-9))[b_total > 0].sum())
chi2_closure = float(s @ np.linalg.inv(np.diag(np.maximum(b_total, 1e-9))) @ s)
chi2_sys = float(s @ np.linalg.inv(V_sys + np.diag(b_total)) @ s)
print(f"STEP 4  flagship point: chi2 stat-only = {chi2_stat:.1f}, "
      f"matrix closure = {chi2_closure:.1f}, with systematics = {chi2_sys:.1f}")
print(f"        [checkpoints: closure == stat-only; with-sys is SMALLER]")

# ----------------------------------------- step 5: the limit at 487 MeV
from pathlib import Path
rows = []
for p in sorted(Path(SIGLIB).glob("tmm_iso_reweighting_weights_m0.48688_mu*.parquet")):
    mu = float(p.stem.split("_mu")[-1])
    wmu = (iso.select(["run", "subrun", "event"])
           .join(pl.read_parquet(p), on=["run", "subrun", "event"], how="left")
           )["tmm_weight"].fill_null(0.0).to_numpy() * 2.0
    smu, _ = np.histogram(iso["wc_reco_showerKE"].to_numpy()[sel], bins=BINS,
                          weights=wmu[sel])
    rows.append((mu, float(smu @ np.linalg.inv(np.diag(np.maximum(b_total, 1e-9))) @ smu),
                 float(smu @ np.linalg.inv(V_sys + np.diag(b_total)) @ smu)))
rows.sort()
mus = np.array([r[0] for r in rows])


def crossing(chi2s):
    c = np.array(chi2s)
    i = np.argmax(c >= 3.84)
    if c[i] < 3.84:
        return np.nan
    if i == 0:
        return mus[0]
    return np.exp(np.interp(np.log(3.84), np.log(c[i-1:i+1]), np.log(mus[i-1:i+1])))


lim_stat = crossing([r[1] for r in rows])
lim_sys = crossing([r[2] for r in rows])
print(f"STEP 5  487 MeV limit: stat-only mu = {lim_stat:.2e}, "
      f"with systematics mu = {lim_sys:.2e}  (x{lim_sys/lim_stat:.2f})")
print(f"        [checkpoint: 1.00e-06 -> 1.32e-06, x1.32]")

# =====================================================================
# BUILD-IT-ALL ROADMAP  (implement one per sitting; deviations from the
# CV histogram, NOT the universe mean: V = sum_u outer(b_u - b, b_u - b)/N)
#
# [x] 1  flux            multisim 1000   weights: wc_net_weight            max frac 0.37
# [ ] 2  All_UBGenie     multisim  500   weights: wc_net_weight/wc_weight_cv  0.649
#         (add wc_weight_cv to the EVENTS select first)
# [ ] 3  reint_all       multisim 1000   weights: wc_net_weight            0.179
# [ ] 4  NormNCCOH_UBGenie      unisim /Ncols  GENIE-divided weights       0.661
# [ ] 5  NormCCCOH_UBGenie      unisim /Ncols  GENIE-divided               0.187
# [ ] 6  AxFFCCQEshape, DecayAngMEC, ThetaDelta2NRad, Theta_Delta2Npi,
#        VecFFCCQEshape, XSecShape_CCMEC  unisim /Ncols  GENIE-divided     0.195 each
# [ ] 7  RPA_CCQE_UBGenie       unisim /2      GENIE-divided               0.195
# [ ] 8  xsr_scc_Fa3_SCC, xsr_scc_Fv3_SCC  unisim /10  PLAIN weights       0.092 each
# [ ] 9  MC stat: diag(sum w^2), non-EXT events        == ref["mcstat_cov"]
# [ ] 10 assembly: fractional (floor b_mc at 1e-3, zero non-finite),
#        sum, x outer(b_mc,b_mc), + MC stat; flagship chi2 == 60.2
#
# Gotchas: 4-key weights join (no filetype) so derived samples inherit
# parent-event universes; float32 cast before sanitization; same b_mc on
# both sides of the fractional conversion; diag(b_total) stays separate.
# =====================================================================
