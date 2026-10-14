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
noHC_flags = [True, False]

output_csv = "darknews_microboone_cryo_dirt_TMM_Umu4_grid_scan.csv"
partial_csv = "darknews_microboone_cryo_dirt_TMM_Umu4_grid_scan_partial.csv"


# -----------------------------
# Mass-dependent Umu4
# -----------------------------

def get_Umu4(m4):
    """
    m4 is in GeV.
    """

    if 0.010 <= m4 <= 0.100:
        return 1e-6

    elif 0.200 <= m4 <= 0.390:
        return 1e-10

    elif 0.400 <= m4 <= 1.000:
        return 1e-7

    elif 1.000 < m4 <= 2.000:
        return 1e-6

    else:
        return None


# -----------------------------
# Cryostat decay containment cut
# -----------------------------

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


# -----------------------------
# Run one DarkNews sample
# -----------------------------

def run_one_sample(m4, mu_tr, Umu4, experiment, noHC):
    model_kwargs = {
        "m4": m4,
        "mu_tr_mu4": mu_tr,   # scanned TMM
        "UD4": 0,
        "Umu4": Umu4,         # mass-dependent mixing
        "epsilon": 0.0,
        "gD": 0.0,
    }

    gen = GenLauncher(
        experiment=experiment,
        neval=neval,
        decay_product="photon",
        noHC=noHC,
        HNLtype="dirac",
        **model_kwargs,
    )

    df = gen.run()

    mask = inside_microboone_cryostat(df, label="pos_decay")
    df_sel = df[mask]

    n_generated = len(df)
    n_selected = len(df_sel)

    weight_sum = (
        df_sel["w_event_rate"].sum()
        if "w_event_rate" in df_sel.columns
        else np.nan
    )

    return {
        "m4": m4,
        "mu_tr_mu4": mu_tr,
        "Umu4": Umu4,
        "experiment": experiment,
        "noHC": noHC,
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
            round(row["mu_tr_mu4"], 20),
            round(row["Umu4"], 20),
            row["experiment"],
            bool(row["noHC"]),
        ))

    print(f"Resuming from {partial_csv}")
    print(f"Already completed samples: {len(done_keys)}", flush=True)


# -----------------------------
# Main scan
# -----------------------------

for m4 in m4_values:

    Umu4 = get_Umu4(m4)

    if Umu4 is None:
        print(f"Skipping m4={m4:.4e}: no Umu4 assigned", flush=True)
        continue

    for mu_tr in mu_values:
        for experiment in experiments:
            for noHC in noHC_flags:

                key = (
                    round(m4, 12),
                    round(mu_tr, 20),
                    round(Umu4, 20),
                    experiment,
                    noHC,
                )

                if key in done_keys:
                    continue

                print(
                    f"Running m4={m4:.4e}, mu_tr={mu_tr:.4e}, "
                    f"Umu4={Umu4:.4e}, experiment={experiment}, noHC={noHC}",
                    flush=True,
                )

                try:
                    out = run_one_sample(
                        m4=m4,
                        mu_tr=mu_tr,
                        Umu4=Umu4,
                        experiment=experiment,
                        noHC=noHC,
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
                        f"FAILED: m4={m4}, mu_tr={mu_tr}, Umu4={Umu4}, "
                        f"experiment={experiment}, noHC={noHC}",
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
    .groupby(["m4", "mu_tr_mu4", "Umu4"], as_index=False)
    .agg(
        total_weight=("weight_sum", "sum"),
        total_selected=("n_selected_decay_in_cryo", "sum"),
        total_generated=("n_generated", "sum"),
    )
)

summary_csv = "darknews_microboone_total_weight_TMM_Umu4_grid_scan.csv"
df_summary.to_csv(summary_csv, index=False)

print("\nFinished scan.")
print(f"Detailed file: {output_csv}")
print(f"Summary file: {summary_csv}")

