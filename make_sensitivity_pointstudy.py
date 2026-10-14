"""Sensitivity in the (m4, d) plane with external limits, nominal BDT vs
the 49 MeV retrain (BDT training-point study).

Same layout as sensitivity_final: Kamp-digitized constraints + MiniBooNE
contours + MINERvA ME curves, d = mu_tr/2, grid-point crossings (no
interpolation), 95% CL (chi2/q >= 3.84). Both MicroBooNE curves use the
Poisson q statistic: for the nominal working point it agrees with the
chi2 within grid spacing, and it is robust to the sparse background
bins that inflate chi2 for the tight retrain cuts.

    python make_sensitivity_pointstudy.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAP = Path("/nevis/riverside/data/karan/tmm_mapping")
EXT = MAP / "external"

MB = np.load(EXT / "miniboone_digitized.npz")
MIN = np.load(EXT / "minerva_digitized.npz")


def grid_limit(csv, col="q_stat", thr=3.84):
    t = pd.read_csv(csv)
    t = t[np.abs(t.mass - 0.48) > 1e-9]
    out = {}
    for m, sub in t.groupby("mass"):
        sub = sub.drop_duplicates("tmm").sort_values("tmm")
        hit = sub[sub[col] >= thr]
        if len(hit):
            out[m] = float(hit.tmm.iloc[0])
    s = pd.Series(out).sort_index()
    return s.index.to_numpy() * 1000.0, s.to_numpy() / 2.0  # d = mu/2


def main():
    plt.rcParams.update({
        "font.family": "Nimbus Roman", "font.size": 18,
        "axes.labelsize": 20, "xtick.labelsize": 16, "ytick.labelsize": 16,
        "legend.fontsize": 13.5,
    })
    fig, ax = plt.subplots(figsize=(12, 8.5))

    # shaded existing constraints (gray outlines from the Kamp figure)
    for k in MB.keys():
        if k.startswith("constraints_"):
            seg = MB[k]
            ax.fill(seg[:, 0], seg[:, 1], color="0.82", lw=0, zorder=0)

    # MiniBooNE preferred regions
    mbstyle = {
        "enuqe_3sigma": ("#c41e6a", "-", r"MiniBooNE $E_\nu^{QE}$ $3\sigma$"),
        "enuqe_95": ("#c41e6a", "--", r"MiniBooNE $E_\nu^{QE}$ 95%"),
        "costheta_3sigma": ("#2e8b22", "-", r"MiniBooNE $\cos\theta$ $3\sigma$"),
        "costheta_95": ("#2e8b22", "--", r"MiniBooNE $\cos\theta$ 95%"),
    }
    seen = set()
    for k in MB.keys():
        base = k.rsplit("_", 1)[0]
        if base not in mbstyle:
            continue
        c, ls, lab = mbstyle[base]
        ax.plot(MB[k][:, 0], MB[k][:, 1], ls, color=c, lw=1.6,
                label=None if base in seen else lab, zorder=3)
        seen.add(base)

    ax.plot(MIN["ME_FHC"][:, 0], MIN["ME_FHC"][:, 1], "-.", color="#ff8c00",
            lw=2, label="MINERvA ME FHC 95% CL", zorder=3)
    ax.plot(MIN["ME_RHC"][:, 0], MIN["ME_RHC"][:, 1], "--", color="#ff8c00",
            lw=2, label="MINERvA ME RHC 95% CL", zorder=3)

    # MicroBooNE curves (grid crossings, Poisson q, 95% CL)
    curves = [
        ("sensitivity_scan_v3cohall_shwKE_6.8e+20.csv", "#1f77b4", "-", 3,
         r"MicroBooNE 95% CL (stat.), $6.8\times10^{20}$ POT"),
        ("sensitivity_scan_pointcontrol_shwKE_6.8e+20.csv", "0.45", "--", 2.2,
         r"BDT retrained at $m_4=480$ MeV"),
        ("sensitivity_scan_pointlow_shwKE_6.8e+20.csv", "black", "--", 2.5,
         r"BDT trained at $m_4=49$ MeV"),
        ("sensitivity_scan_pointhigh_shwKE_6.8e+20.csv", "#d62728", ":", 2.5,
         r"BDT trained at $m_4=987$ MeV"),
    ]
    for csv, color, ls, lw, lab in curves:
        m, d = grid_limit(MAP / csv)
        ax.plot(m, d, ls, color=color, lw=lw, zorder=5, label=lab)

    ax.plot([480], [1.25e-6], "*", color="black", ms=20, zorder=6,
            label="MiniBooNE best fit")

    for txt, x, y, rot in [("LSND", 25.0, 5.3e-6, 0),
                           ("NOMAD", 263.5, 5.9e-6, 0),
                           ("CHARM-II", 119.1, 1.4e-6, 90),
                           ("SuperK", 45.9, 1.07e-6, 0),
                           ("Borexino", 13.8, 1.2e-7, 62),
                           ("SN1987A", 15.3, 4.3e-8, 0)]:
        ax.text(x, y, txt, color="0.35", fontsize=16, rotation=rot,
                ha="center", va="center", zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, 2000)
    ax.set_ylim(2e-8, 1e-5)
    ax.set_xlabel(r"$m_4$ [MeV]")
    ax.set_ylabel(r"$d_{\mu N}$ [GeV$^{-1}$]")
    ax.legend(loc="lower right", ncol=2, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(MAP / "slides/sensitivity_pointstudy.png", dpi=200)
    print("wrote slides/sensitivity_pointstudy.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
