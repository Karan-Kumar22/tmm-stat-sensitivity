"""Annotated anatomy of the sensitivity contour for slides.

Nominal 95% CL limit (Poisson q, grid crossings) with the three regimes
marked: the m^(-3/4) decay-probability slope at low mass, the valley,
and the coherent form-factor wall at high mass.

    python make_sensitivity_anatomy.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAP = Path("/nevis/riverside/data/karan/tmm_mapping")


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
    return s.index.to_numpy() * 1000.0, s.to_numpy()


def main():
    plt.rcParams.update({
        "font.family": "Nimbus Roman", "font.size": 18,
        "axes.labelsize": 20, "xtick.labelsize": 16, "ytick.labelsize": 16,
    })
    m, mu = grid_limit(MAP / "sensitivity_scan_v3cohall_shwKE_6.8e+20.csv")

    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.plot(m, mu, "-", color="#1f77b4", lw=3.5, zorder=5)

    # m^(-3/4) guide anchored at 20 MeV
    mg = np.logspace(np.log10(10), np.log10(220), 50)
    anchor = np.interp(20.0, m, mu)
    ax.plot(mg, anchor * 1.35 * (mg / 20.0) ** (-0.75), "--", color="0.3",
            lw=2, zorder=4)
    ax.text(38, 2.6e-6, r"$\mu \propto m_4^{-3/4}$",
            color="0.2", fontsize=19, rotation=-31)
    ax.text(13, 4.0e-7,
            "HNL lives too long,\n"
            r"decays past the detector:"
            "\n"
            r"$P_{\rm decay} \propto \Gamma \propto \mu^2 m_4^3$",
            fontsize=16, color="0.25")

    # valley
    ax.annotate("best reach:\ndecays inside, still coherent",
                xy=(210, 1.75e-7), xytext=(95, 6.5e-8),
                fontsize=16, color="0.25", ha="center",
                arrowprops=dict(arrowstyle="->", color="0.4"))

    # form-factor wall
    ax.axvspan(827, 2000, color="0.92", zorder=0)
    ax.text(1250, 3.5e-7,
            "coherent form factor:\n"
            r"$Q\,R_{\rm nucleus} \gtrsim 1$,"
            "\nrate collapses,\nno selection can help",
            fontsize=16, color="0.25", ha="center")

    # mu4 arrow: steepness in coupling
    ax.annotate("", xy=(480, 2.4e-6), xytext=(480, 6.2e-7),
                arrowprops=dict(arrowstyle="->", lw=2, color="#c41e6a"))
    ax.text(530, 1.15e-6, r"rate $\propto \mu^4$"
            "\n(one grid step:\n rate $\\times$2.3)",
            fontsize=16, color="#c41e6a")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, 2000)
    ax.set_ylim(3e-8, 1e-5)
    ax.set_xlabel(r"$m_4$ [MeV]")
    ax.set_ylabel(r"$\mu^{\rm tr}_{\mu 4}$ [GeV$^{-1}$] (95% CL)")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(MAP / "slides/sensitivity_anatomy.png", dpi=200)
    print("wrote slides/sensitivity_anatomy.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
