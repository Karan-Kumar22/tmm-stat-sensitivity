# MicroBooNE HNL transition-magnetic-moment single-photon search: analysis code

Code archive of the full analysis chain, from DarkNews event generation
to the statistical-only sensitivity.
Systematic uncertainties are handled separately and are not part of
this archive.
Snapshot of the analysis state of 2026-10 (runs 4a-4d NGEM production,
coherent-everywhere signal definition, shower-energy fit). Code only:
the data products (grids, dataframes, weight libraries) live on the
Nevis cluster at the paths listed at the bottom.

## The chain, directory by directory

### `generation/` - the MC generation package
DarkNews interface, MicroBooNE geometry (cryostat tube+caps, liquid
fill, dirt cone), forced-decay-in-volume machinery, batch bookkeeping
with convergence control, and its test suite (geometry closure, decay
volume, batch store). Produced the 992-point grid (grid_v3: coherent
and incoherent channels, nu_mu and nu_mu_bar, per-event
scattering_regime).

### `mapping/` - truth-map reweighting
Builds the 3D truth map (E_gamma, cos theta, z) per grid point from the
generated events, with the rate-carrying-cell convergence criterion
(converge_point.py, binning.py, merged_binning.py); campaign driver
run_grid.py; reweights the isotropic single-photon sample to each model
point (reweight.py); exports per-point weight parquets in NGEM format
(export_ngem_grid.py / export_ngem_weights.py). Signal definition
applied in dataframe_io.apply_signal_definition, configured in
config.json. Only the scripts in the direct chain to the sensitivity
are archived; study and QA scripts remain on the Nevis cluster.

### top level - sensitivity
- `sensitivity_scan.py`: the stat-only scan. Selects test-half
  background and signal-carrier events at the BDT working point, loops
  over the 992 weight files, writes (mass, tmm, n_sig, q_stat,
  chi2_sys) CSVs.
- `make_sensitivity_pointstudy.py`, `make_sensitivity_anatomy.py`:
  the phase-space figure with external limits and the contour-anatomy
  figure.
- `external/`: digitisation of the constraint curves from Kamp et al.,
  PRD 107 055009 (arXiv:2206.07100) source figures, with the resulting
  npz (MiniBooNE regions, global constraints, MINERvA ME FHC/RHC).

## Analysis configuration (frozen for the stat-only result)

- Signal: HNL coherent upscattering (cryostat and dirt), 992-point
  DarkNews grid, truth-map reweighting of one isotropic 1-gamma sample.
- Selection: WC generic preselection + tmm_vs_bkg BDT score > 0.96
  (s/sqrt(b) optimum at the flagship point m4 = 480 MeV,
  mu_tr = 2.5e-6, the MiniBooNE best fit).
- Fit: reconstructed shower KE, 20 bins, 0-2 GeV;
  chi2 = sum s_i^2/b_i, 95% CL at 3.84 (Wilks, 1 dof);
  Poisson Asimov q as cross-check.
- Exposure: 6.8e20 POT (1.11e21 variant); weights normalised to the
  open dataset (2.098e19 + 4.038e19 POT).

## Checkpoints (reproduce before trusting a rerun)

- Selected background at cut 0.96, 6.8e20: 3818 test-half events,
  24.2 expected (23.3 inside the 0-2 GeV binning).
- Flagship point chi2 (stat-only) = 126.0.
- 95% CL crossing at 487 MeV: mu = 1.00e-6 (stat), 1.32e-6 (+syst).
- Last excluded mass: 827 MeV. MiniBooNE best fit excluded.

## Data dependencies (Nevis paths, not in this repository)

- Generated grid + registries: /nevis/riverside/data/karan/ (grid_v3)
- Event table: /nevis/riverside/data/karan/ngem/intermediate_file/
  presel_df_train_vars.parquet (NGEM production 2026-05-23)
- BDT scores: ~fk2476/uboone_ngem/training_outputs/tmm_with_new_files/
- Signal weight library (993 parquets): tmm_mapping/ngem_weights_v3_cohall/
- Scan CSVs and figures: tmm_mapping/ and tmm_mapping/slides/
- The NGEM framework (Lee Hagaman): github.com/leehagaman/uboone_ngem

NOTE: generation/mapping scripts expect the numpy-2 virtualenv
(~/TMM_new/.venv-tmm) for grid pickles; NGEM-facing scripts run with the
system python.
