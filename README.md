# TMM single-photon search: statistical-only sensitivity

Code, results and external-limit data for the statistical-only 95% CL
sensitivity of the MicroBooNE HNL transition-magnetic-moment single-photon
search. Snapshot of the analysis state of 2026-10 (runs 4a-4d production,
coherent-everywhere signal definition, shower-energy fit).

## Analysis configuration (frozen)

- Signal: HNL coherent upscattering (cryostat and dirt), DarkNews grid of
  992 points (32 masses x 31 couplings), per-point truth-map reweighting
  of one isotropic single-photon sample.
- Selection: WC generic preselection, tmm_vs_bkg BDT score > 0.96
  (cut from s/sqrt(b) scan at the flagship point m4 = 480 MeV,
  mu_tr = 2.5e-6 GeV^-1, the MiniBooNE best fit).
- Fit variable: reconstructed shower kinetic energy, 20 bins, 0-2 GeV.
- Statistic: chi2 = sum_i s_i^2 / b_i (Asimov, background-only data);
  95% CL at chi2 >= 3.84 (Wilks, 1 dof). Poisson Asimov q in the CSVs
  as a cross-check.
- Exposure: 6.8e20 POT (and 1.11e21 POT variant). Test-half events
  upweighted x2; weights normalised to the open dataset
  (2.098e19 + 4.038e19 POT) and scaled.

## Contents

- `sensitivity_scan.py` - the scan: selects background and signal-carrier
  events, chooses/uses the BDT working point, loops over the 992 weight
  files, writes a CSV of (mass, tmm, n_sig, q_stat, chi2_sys).
- `make_sensitivity_pointstudy.py` - the phase-space figure: our contours
  with the digitised external limits (d_muN = mu_tr/2 convention).
- `make_sensitivity_anatomy.py` - the annotated contour-anatomy figure
  (m^-3/4 decay-probability slope, valley, coherent form-factor wall).
- `external/` - digitisation of the constraint curves from Kamp et al.,
  PRD 107 055009 (arXiv:2206.07100) source figures: extraction code and
  the resulting npz (MiniBooNE EnuQE / cos-theta regions, global
  constraints outline, MINERvA ME FHC/RHC).

## Checkpoints (reproduce these before trusting a rerun)

- Selected background at cut 0.96, 6.8e20 POT: 3818 test-half events,
  24.2 expected (23.3 inside the 0-2 GeV binning).
- Flagship point (m4 = 0.48, mu = 2.5e-6): chi2 = 126.0.
- 95% CL grid crossing at 487 MeV: mu = 1.00e-6 (interpolated).
- Last excluded mass: 827 MeV. MiniBooNE best fit excluded.

## Data dependencies (not in this repository; Nevis paths)

- Event table: /nevis/riverside/data/karan/ngem/intermediate_file/
  presel_df_train_vars.parquet  (NGEM production, 2026-05-23)
- BDT scores: ~fk2476/uboone_ngem/training_outputs/tmm_with_new_files/
  predictions.parquet
- Signal weight library (993 parquets):
  /nevis/riverside/data/karan/tmm_mapping/ngem_weights_v3_cohall/
- Scan outputs (CSVs) and figures:
  /nevis/riverside/data/karan/tmm_mapping/ (sensitivity_scan_v3cohall_shwKE_*.csv,
  regenerated 2026-10-12 with pinned inputs) and tmm_mapping/slides/
- The NGEM framework (Lee Hagaman): github.com/leehagaman/uboone_ngem

Systematic uncertainties are NOT included here; see the systematics
machinery (build_bkg_covariance.py, sensitivity_scan_cov.py) in the
parent analysis area.
