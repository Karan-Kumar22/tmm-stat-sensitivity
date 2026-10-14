import os
import traceback
import numpy as np
import pandas as pd

from DarkNews.GenLauncher import GenLauncher
from DarkNews.geom import (
    z_t, r_t,
    r_c, theta_c, ctheta_c, h, cap_gap,
)

# -----------------------------
# Scan setup
# -----------------------------

m4_values = np.logspace(np.log10(0.01), np.log10(1.99), 20)
mu_values = np.logspace(-8, -5, 20)

neval = 50_000

experiments = ["microboone", "microboone_dirt"]

output_csv = "darknews_microboone_cryo_dirt_TMM_scan_defaultHC.csv"
partial_csv = "darknews_microboone_cryo_dirt_TMM_scan_defaultHC_partial.csv"


# -----------------------------
# Cryostat decay containment cut
# -----------------------------

def inside_microboone_cryostat(df, label="pos_decay"):
    x = df[(label, "1")].to_numpy()
    y = df[(label, "2")].to_numpy()
    z = df[(label, "3")].to_numpy()

    r_polar = np.sqrt(x**2 + y**2)

    # Cryostat cylindrical tube
    mask_tube = (
        (-z_t / 2 < z)
        & (z < z_t / 2)
        & (r_polar < r_t)
    )

    # Front spherical cap
    z1 = z - (-z_t / 2 - cap_gap + r_c * ctheta_c)
    r1 = np.sqrt(r_polar**2 + z1**2)
    inc1 = np.arctan2(z1, r_polar)

    mask_cap1 = (
        (r1 < r_c)
        & (inc1 < theta_c)
        & (z1 < -r_c + h)
    )

    # Back spherical cap
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


# -----------------------------
# Run one DarkNews sample
# -----------------------------

def run_one_sample(m4, mu_tr, experiment):
    model_kwargs = {
        "m4": m4,
        "mu_tr_mu4": mu_tr,
        "UD4": 0,
        "Umu4": 0,
        "epsilon": 0.0,
        "gD": 0.0,
    }

    gen = GenLauncher(
        experiment=experiment,
        neval=neval,
        decay_product="photon",
        HNLtype="dirac",
        **model_kwargs,
    )

    df = gen.run()

    mask = inside_microboone_cryostat(df, label="pos_decay")
    df_sel = df[mask]

    n_generated = len(df)
    n_selected = len(df_sel)

    if "w_event_rate" in df_sel.columns:
        weight_sum = df_sel["w_event_rate"].sum()
    else:
        weight_sum = np.nan

    return {
        "m4": m4,
        "mu_tr": mu_tr,
        "experiment": experiment,
        "n_generated": n_generated,
        "n_selected_decay_in_cryo": n_selected,
        "raw_fraction": n_selected / n_generated if n_generated > 0 else 0.0,
        "weight_sum": weight_sum,
    }


# -----------------------------
# Resume support
# -----------------------------

done_keys = set()
results = []

if os.path.exists(partial_csv):
    old = pd.read_csv(partial_csv)
    results = old.to_dict("records")

    for _, row in old.iterrows():
        done_keys.add((
            round(row["m4"], 12),
            round(row["mu_tr"], 20),
            row["experiment"],
        ))

    print(f"Resuming from {partial_csv}")
    print(f"Already completed samples: {len(done_keys)}", flush=True)


# -----------------------------
# Main scan
# -----------------------------

for m4 in m4_values:
    for mu_tr in mu_values:
        for experiment in experiments:

            key = (
                round(m4, 12),
                round(mu_tr, 20),
                experiment,
            )

            if key in done_keys:
                continue

            print(
                f"Running m4={m4:.4e}, mu={mu_tr:.4e}, "
                f"experiment={experiment}",
                flush=True,
            )

            try:
                out = run_one_sample(
                    m4=m4,
                    mu_tr=mu_tr,
                    experiment=experiment,
                )

                results.append(out)
                done_keys.add(key)

                pd.DataFrame(results).to_csv(partial_csv, index=False)

                print(
                    f"Done: selected={out['n_selected_decay_in_cryo']}, "
                    f"weight_sum={out['weight_sum']:.6e}",
                    flush=True,
                )

            except Exception:
                print(
                    f"FAILED: m4={m4}, mu={mu_tr}, "
                    f"experiment={experiment}",
                    flush=True,
                )
                traceback.print_exc()


# -----------------------------
# Save detailed and summarized results
# -----------------------------

df_detail = pd.DataFrame(results)
df_detail.to_csv(output_csv, index=False)

df_summary = (
    df_detail
    .groupby(["m4", "mu_tr"], as_index=False)
    .agg(
        total_weight=("weight_sum", "sum"),
        total_selected=("n_selected_decay_in_cryo", "sum"),
        total_generated=("n_generated", "sum"),
    )
)

summary_csv = "darknews_microboone_total_weight_grid_defaultHC.csv"
df_summary.to_csv(summary_csv, index=False)

print("\nFinished scan.")
print(f"Detailed file: {output_csv}")
print(f"Summary file: {summary_csv}")


