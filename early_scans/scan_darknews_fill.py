"""Fill-in for the two masses skipped by scan_darknews_2.py's Umu4 windows.

m4 = 0.1227, 0.1622 GeV fell in the uncovered 0.1-0.2 GeV gap of get_Umu4.
Generates them at Umu4 = 1e-6 (same as the adjacent <0.1 GeV and >1 GeV
windows), noHC=False only (the rows the tmm_vs_ncqe figure uses).
Same generator settings and containment cut as scan_darknews_2.py.

    ~/TMM_new/.venv-tmm/bin/python scan_darknews_fill.py
"""

import os
import traceback
import numpy as np
import pandas as pd

from DarkNews.GenLauncher import GenLauncher
from DarkNews.geom import (
    z_t, r_t,
    r_c, theta_c, ctheta_c, h, cap_gap,
)

m4_all = np.logspace(np.log10(0.01), np.log10(1.99), 20)
m4_values = [m4_all[9], m4_all[10]]         # 0.1227, 0.1622 GeV
mu_values = np.logspace(-8, -5, 20)
Umu4 = 1e-6
neval = 50_000
experiments = ["microboone", "microboone_dirt"]

output_csv = "darknews_microboone_cryo_dirt_TMM_Umu4_fill.csv"
partial_csv = "darknews_microboone_cryo_dirt_TMM_Umu4_fill_partial.csv"


def inside_microboone_cryostat(df, label="pos_decay"):
    x = df[(label, "1")].to_numpy()
    y = df[(label, "2")].to_numpy()
    z = df[(label, "3")].to_numpy()

    r_polar = np.sqrt(x**2 + y**2)

    mask_tube = (
        (-z_t / 2 < z)
        & (z < z_t / 2)
        & (r_polar < r_t)
    )

    z1 = z - (-z_t / 2 - cap_gap + r_c * ctheta_c)
    r1 = np.sqrt(r_polar**2 + z1**2)
    inc1 = np.arctan2(z1, r_polar)

    mask_cap1 = (
        (r1 < r_c)
        & (inc1 < theta_c)
        & (z1 < -r_c + h)
    )

    z2 = z + (-z_t / 2 - cap_gap - h + r_c)
    r2 = np.sqrt(r_polar**2 + z2**2)
    inc2 = np.arctan2(-z2, r_polar)

    mask_cap2 = (
        (r2 < r_c)
        & (inc2 < theta_c)
        & (z2 > r_c - h)
    )

    fraction_of_height_wLAr = 0.85

    return (
        (mask_tube | mask_cap1 | mask_cap2)
        & (y + r_t < 2 * r_t * fraction_of_height_wLAr)
    )


def run_one_sample(m4, mu_tr, experiment):
    gen = GenLauncher(
        experiment=experiment,
        neval=neval,
        decay_product="photon",
        noHC=False,
        HNLtype="dirac",
        m4=m4,
        mu_tr_mu4=mu_tr,
        UD4=0,
        Umu4=Umu4,
        epsilon=0.0,
        gD=0.0,
    )
    df = gen.run()
    mask = inside_microboone_cryostat(df, label="pos_decay")
    df_sel = df[mask]
    weight_sum = (df_sel["w_event_rate"].sum()
                  if "w_event_rate" in df_sel.columns else np.nan)
    return {
        "m4": m4,
        "mu_tr_mu4": mu_tr,
        "Umu4": Umu4,
        "experiment": experiment,
        "noHC": False,
        "n_generated": len(df),
        "n_selected_decay_in_cryo": len(df_sel),
        "raw_fraction": len(df_sel) / len(df) if len(df) > 0 else 0.0,
        "weight_sum": weight_sum,
    }


done_keys = set()
results = []
if os.path.exists(partial_csv):
    old = pd.read_csv(partial_csv)
    results = old.to_dict("records")
    for _, row in old.iterrows():
        done_keys.add((round(row["m4"], 12), round(row["mu_tr_mu4"], 20),
                       row["experiment"]))
    print(f"Resuming: {len(done_keys)} samples done", flush=True)

n_total = len(m4_values) * len(mu_values) * len(experiments)
for m4 in m4_values:
    for mu_tr in mu_values:
        for experiment in experiments:
            key = (round(m4, 12), round(mu_tr, 20), experiment)
            if key in done_keys:
                continue
            print(f"[{len(results) + 1}/{n_total}] m4={m4:.4e} "
                  f"mu_tr={mu_tr:.4e} {experiment}", flush=True)
            try:
                out = run_one_sample(m4, mu_tr, experiment)
            except Exception:
                traceback.print_exc()
                out = {"m4": m4, "mu_tr_mu4": mu_tr, "Umu4": Umu4,
                       "experiment": experiment, "noHC": False,
                       "n_generated": 0, "n_selected_decay_in_cryo": 0,
                       "raw_fraction": 0.0, "weight_sum": np.nan}
            results.append(out)
            pd.DataFrame(results).to_csv(partial_csv, index=False)

pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"Done: wrote {output_csv} ({len(results)} rows)", flush=True)
