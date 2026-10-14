"""BNB flux systematics on the TMM signal, correlated with the backgrounds.

The HNL production rate at every grid point is proportional to the BNB nu_mu
flux that DarkNews integrates over (MicroBooNE_BNB_fullflux.txt, nu_mu only).
NGEM carries 1000 flux multisim universes as per-event weights on the
neutrino-overlay MC; those weights are a function of the true neutrino
(energy, flavour, parent).  We transfer them to the signal through E_nu:

  1. r_u(E) = sum_i w_i f_{u,i} / sum_i w_i  over nu_mu nu_overlay events in
     true-E_nu bin E, for every universe u  -> systematics/flux_ratios.npz
     (the fractional flux variation of universe u at energy E).

  2. Every generated HNL event carries the E_nu of the upscattering neutrino
     (truth tables).  The reweighted signal reco spectrum is linear in the
     truth histogram (reweight.compute_weights: weight = H_T(b)/N_S(b) on a
     fixed support mask), so for a point

        s_u = M . H . r_u,    M[k,b] = sum_{carriers i in reco bin k, cell b} 1/N_S(b)
                              H[b,E] = truth rate in cell b from neutrinos in E bin E

     gives all 1000 universe spectra from one (n_reco x n_E) matrix A = M.H.
     The CV spectrum A.1 must equal the spectrum from the exported
     ngem_weights parquet; this is checked for every point.

Output: systematics/flux_signal_universes_cut{cut}.npz with
  masses, tmms, s_cv (n_pts x n_reco), s_univ (n_pts x n_reco x 1000,
  float32), and the per-point fractional flux sigma on the total signal.

sensitivity_scan_syst.py combines these with the background universes stored
by build_bkg_covariance.py:  C_flux = cov_u(s_u + b_u).

    python flux_systematics.py [--cut 0.95] [--pot 6.8e20]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from config import load_config
from converge_point import n_batches_on_disk
from merged_binning import load_analysis_binning

NGEM = Path("/nevis/houston/home/fk2476/uboone_ngem")
INTER = Path("/nevis/riverside/data/karan/ngem/intermediate_file")
WEIGHTS_DF = Path(
    "/nevis/riverside/data/sanderson/ngem/intermediate_files/presel_weights_df.parquet")
HERE = Path(__file__).resolve().parent
GRID = Path("/nevis/riverside/data/karan/tmm_points/grid")
OUT_DIR = HERE / "systematics"
POT_DARKNEWS = 6.8e20
VAR = "wc_kine_reco_Enu"
RECO_BINS = np.linspace(0.0, 2000.0, 21)                    # MeV, as sensitivity_scan
# true E_nu bins [GeV]: fine where the nu_mu flux peaks, wide in the kaon tail
ENU_EDGES = np.r_[0.0, np.arange(0.2, 3.0, 0.1), 3.5, 4.0, 5.0, 8.0]


# --------------------------------------------------------------------------- step 1
def build_flux_ratios(path: Path) -> dict:
    keys = ["filename", "run", "subrun", "event"]
    nu = (pl.scan_parquet(INTER / "presel_df_train_vars.parquet")
          .filter((pl.col("filetype") == "nu_overlay") & (pl.col("wc_truth_nuPdg") == 14))
          .select(keys + ["wc_truth_nuEnergy", "wc_net_weight"]))
    fw = pl.scan_parquet(WEIGHTS_DF).select(keys + ["flux_all"])
    df = nu.join(fw, on=keys, how="inner").collect(engine="streaming")
    print(f"[flux] {df.height} nu_mu nu_overlay events with flux universes")

    e = df["wc_truth_nuEnergy"].to_numpy() / 1000.0          # MeV -> GeV
    w = df["wc_net_weight"].to_numpy()
    idx = np.searchsorted(ENU_EDGES, e, side="right") - 1
    ok = (idx >= 0) & (idx < len(ENU_EDGES) - 1)
    n_e = len(ENU_EDGES) - 1
    denom = np.bincount(idx[ok], weights=w[ok], minlength=n_e)
    counts = np.bincount(idx[ok], minlength=n_e)

    univ = df["flux_all"].to_numpy()
    n_u = len(univ[0])
    num = np.zeros((n_e, n_u))
    chunk = 50
    for start in range(0, n_u, chunk):
        stop = min(start + chunk, n_u)
        block = np.array([a[start:stop] for a in univ], dtype=np.float32)  # (events, chunk)
        # NGEM sanitisation (systematics.create_universe_histograms)
        block = np.nan_to_num(block, nan=1, posinf=1, neginf=1)
        block[(block > 30) | (block < 0)] = 1
        for j in range(stop - start):
            num[:, start + j] = np.bincount(idx[ok], weights=w[ok] * block[ok, j].astype(np.float64),
                                            minlength=n_e)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(denom[:, None] > 0, num / denom[:, None], 1.0)
    # bins without MC (only possible below 0.2 GeV): borrow the nearest populated bin
    empty = counts == 0
    if empty.any():
        filled = np.where(~empty)[0]
        for i in np.where(empty)[0]:
            ratios[i] = ratios[filled[np.argmin(np.abs(filled - i))]]
    sig = ratios.std(axis=1)
    print("[flux] fractional flux sigma vs E_nu [GeV]:")
    for lo, hi, s, n in zip(ENU_EDGES[:-1], ENU_EDGES[1:], sig, counts):
        print(f"        {lo:4.1f}-{hi:4.1f}  sigma {s:.3f}   ({n} MC events)")
    out = {"enu_edges": ENU_EDGES, "ratios": ratios.astype(np.float32),
           "counts": counts, "sigma": sig}
    np.savez(path, **out)
    print(f"[flux] wrote {path}")
    return out


# --------------------------------------------------------------------------- step 2
def load_truth_with_enu(point: Path, index: int) -> pd.DataFrame:
    from converge_point import _default_signal_definition
    from dataframe_io import apply_signal_definition
    definition = _default_signal_definition()
    frames = []
    for comp, wcol in (("cryostat", "w_event_rate"), ("dirt", "closure_weight")):
        t = pd.read_pickle(point / "batches" / f"{comp}_{index:05d}_truth.pkl")
        t = apply_signal_definition(t, comp, definition)
        frames.append(pd.DataFrame({"E_gamma": t.E_gamma, "cos_theta": t.cos_theta,
                                    "z": t.z, "E_nu": t.E_nu, "weight": t[wcol]}))
    return pd.concat(frames, ignore_index=True)


def signal_carriers(cut: float, pot: float, binning, config, source: pd.DataFrame):
    """Selected test-half iso photons: reco bin, truth cell, and the factor
    2*pot/POT_DN / N_S(cell) that turns a truth-cell rate into reco counts."""
    presel = pl.scan_parquet(INTER / "presel_df_train_vars.parquet").select(
        ["filetype", "run", "subrun", "event", VAR])
    scores = pl.scan_parquet(
        NGEM / "training_outputs/tmm_with_new_files/predictions.parquet"
    ).select(["filetype", "run", "subrun", "event", "used_for_testing",
              "prob_TMM_iso_reweighted"])
    iso = (presel.join(scores, on=["filetype", "run", "subrun", "event"], how="inner")
           .filter(pl.col("used_for_testing")
                   & (pl.col("filetype") == "isotropic_one_gamma_overlay")
                   & (pl.col("prob_TMM_iso_reweighted") > cut))
           .select(["run", "subrun", "event", VAR]).collect())
    # (no pyarrow in this env: build the polars frame from numpy columns)
    src = pl.DataFrame({c: source[c].to_numpy() for c in
                        ["run", "subrun", "event", "E_gamma", "cos_theta", "z"]})
    merged = iso.join(src, on=["run", "subrun", "event"], how="inner")
    if merged.height != iso.height:
        raise RuntimeError(f"{iso.height - merged.height} carriers missing from iso_truth")

    cols = config.truth_columns
    all_cells = binning.cell_index(source[cols.energy].to_numpy(),
                                   source[cols.costheta].to_numpy(),
                                   source[cols.z].to_numpy())
    n_s = np.bincount(all_cells[all_cells >= 0], minlength=binning.n_cells).astype(float)
    supported = n_s >= max(config.cells.minimum_effective_events, 1.0)

    cell = binning.cell_index(merged["E_gamma"].to_numpy(), merged["cos_theta"].to_numpy(),
                              merged["z"].to_numpy())
    k = np.searchsorted(RECO_BINS, merged[VAR].to_numpy(), side="right") - 1
    keep = (cell >= 0) & supported[np.maximum(cell, 0)] & (k >= 0) & (k < len(RECO_BINS) - 1)
    scale = 2.0 * pot / POT_DARKNEWS
    m = np.zeros((len(RECO_BINS) - 1, binning.n_cells))
    np.add.at(m, (k[keep], cell[keep]), scale / n_s[cell[keep]])
    print(f"[flux] {iso.height} signal carriers above cut, {keep.sum()} in supported cells")
    return m, merged.select(["run", "subrun", "event", VAR])


def point_matrix(point: Path, binning, n_e: int) -> np.ndarray:
    """H[b, E]: truth rate per (cell, E_nu bin), batches combined with 1/N."""
    n = n_batches_on_disk(point)
    h = np.zeros((binning.n_cells, n_e))
    for i in range(n):
        t = load_truth_with_enu(point, i)
        cell = binning.cell_index(t.E_gamma.to_numpy(), t.cos_theta.to_numpy(), t.z.to_numpy())
        ie = np.searchsorted(ENU_EDGES, t.E_nu.to_numpy(), side="right") - 1
        ok = (cell >= 0) & (ie >= 0) & (ie < n_e)
        np.add.at(h, (cell[ok], ie[ok]), t.weight.to_numpy()[ok] / n)
    return h


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cut", type=float, default=0.95)
    ap.add_argument("--pot", type=float, default=POT_DARKNEWS)
    ap.add_argument("--rebuild-ratios", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    ratio_path = OUT_DIR / "flux_ratios.npz"
    if ratio_path.exists() and not args.rebuild_ratios:
        fr = dict(np.load(ratio_path))
        print(f"[flux] loaded {ratio_path}")
    else:
        fr = build_flux_ratios(ratio_path)
    ratios = fr["ratios"].astype(np.float64)            # (n_E, n_univ)
    n_e, n_u = ratios.shape

    config = load_config(HERE / "config.json")
    binning = load_analysis_binning(config)
    source = pd.read_pickle(HERE / "iso_truth.pkl")
    m, carriers = signal_carriers(args.cut, args.pot, binning, config, source)

    points = sorted(p for p in GRID.glob("point_m*_mu*") if (p / "batches").is_dir())
    print(f"[flux] {len(points)} grid points")
    masses, tmms, s_cv, s_univ, check = [], [], [], [], []
    reco_e = carriers[VAR].to_numpy()
    for n_done, point in enumerate(points, 1):
        name = point.name.removeprefix("point_m")
        m_tag, mu_tag = name.split("_mu")
        a = m @ point_matrix(point, binning, n_e)         # (n_reco, n_E)
        cv = a.sum(axis=1)
        su = a @ ratios                                    # (n_reco, n_univ)
        # consistency with the exported weights
        wpath = HERE / "ngem_weights" / f"tmm_iso_reweighting_weights_m{m_tag}_mu{mu_tag}.parquet"
        if wpath.exists():
            w = carriers.join(pl.read_parquet(wpath), on=["run", "subrun", "event"], how="left"
                              )["tmm_weight"].fill_null(0.0).to_numpy() * 2.0 * args.pot / POT_DARKNEWS
            ref, _ = np.histogram(reco_e, bins=RECO_BINS, weights=w)
            rel = abs(cv.sum() - ref.sum()) / ref.sum() if ref.sum() > 0 else 0.0
        else:
            rel = np.nan
        check.append(rel)
        masses.append(float(m_tag)); tmms.append(float(mu_tag))
        s_cv.append(cv); s_univ.append(su.astype(np.float32))
        if n_done % 100 == 0 or n_done == len(points):
            print(f"[flux] {n_done}/{len(points)}  last: m={m_tag} mu={mu_tag} "
                  f"S={cv.sum():.2f} flux sigma {su.sum(axis=0).std() / max(cv.sum(), 1e-12):.3f} "
                  f"cv-vs-export {rel:.1e}", flush=True)

    s_cv = np.array(s_cv); s_univ = np.array(s_univ); check = np.array(check)
    tot = s_univ.sum(axis=1)                               # (n_pts, n_univ)
    frac_sigma = np.where(s_cv.sum(axis=1) > 0, tot.std(axis=1) / np.maximum(s_cv.sum(axis=1), 1e-12), 0)
    print(f"[flux] CV-vs-export max relative difference: {np.nanmax(check):.2e}")
    print(f"[flux] signal flux sigma across grid: min {frac_sigma.min():.3f} "
          f"median {np.median(frac_sigma):.3f} max {frac_sigma.max():.3f}")
    out = OUT_DIR / f"flux_signal_universes_cut{args.cut:.2f}.npz"
    np.savez(out, masses=np.array(masses), tmms=np.array(tmms), reco_bins=RECO_BINS,
             s_cv=s_cv, s_univ=s_univ, frac_sigma=frac_sigma, cv_check=check,
             cut=args.cut, pot=args.pot)
    print(f"[flux] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
