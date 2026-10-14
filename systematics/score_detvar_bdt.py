"""Apply the NGEM TMM BDT to a detector-variation dataframe.

The detvar covariance must be evaluated on the SAME selection as the
analysis (BDT score > cut), so the CV and the seven variation samples need a
score. This reproduces train.py's feature pipeline exactly:

  features  = training_outputs/tmm_with_new_files/feature_importances.csv
              (the 'feature' column is in the model's input order)
  x         = float64, values with |x| > 1e10 set to NaN

Run with the xgboost version the model was trained with (3.2.0, in
~/TMM_new/.venv-tmm) to avoid any cross-version scoring differences.

  --validate  score the nominal presel dataframe (sample) and compare with
              predictions.parquet -- must agree to float precision before
              the detvar scoring is trusted.
  --detvar    input detvar parquet (default: our intermediate_file one);
              writes <stem>_scored.parquet with prob_TMM_iso_reweighted added
              (only the columns the covariance code needs are kept).

    ~/TMM_new/.venv-tmm/bin/python score_detvar_bdt.py --validate
    ~/TMM_new/.venv-tmm/bin/python score_detvar_bdt.py --detvar PATH
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

NGEM = Path("/nevis/houston/home/fk2476/uboone_ngem")
INTER = Path("/nevis/riverside/data/karan/ngem/intermediate_file")
MODEL_DIR = NGEM / "training_outputs/tmm_with_new_files"
KEEP = ["filename", "filetype", "vartype", "detailed_run_period", "run", "subrun",
        "event", "wc_net_weight", "wc_weight_cv", "wc_weight_spline",
        "wc_kine_reco_Enu"]


def load_model():
    feats = [r["feature"] for r in csv.DictReader(open(MODEL_DIR / "feature_importances.csv"))]
    model = xgb.XGBClassifier()
    model.load_model(MODEL_DIR / "bdt.json")
    assert model.n_features_in_ == len(feats), (model.n_features_in_, len(feats))
    return model, feats


def score(model, feats, df: pl.DataFrame, chunk: int = 200_000) -> np.ndarray:
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise SystemExit(f"{len(missing)} BDT features missing from input, e.g. {missing[:8]}")
    out = np.empty(df.height)
    for start in range(0, df.height, chunk):
        x = df.slice(start, chunk).select(feats).to_numpy().astype(np.float64)
        x[(x > 1e10) | (x < -1e10)] = np.nan
        out[start:start + len(x)] = model.predict_proba(x)[:, 1]
    return out


def validate(model, feats, n: int = 200_000) -> int:
    keys = ["filename", "filetype", "run", "subrun", "event"]
    presel = (pl.scan_parquet(INTER / "presel_df_train_vars.parquet")
              .filter(pl.col("filetype") == "nu_overlay").head(n).collect())
    ref = pl.scan_parquet(MODEL_DIR / "predictions.parquet").select(
        keys + ["prob_TMM_iso_reweighted"])
    merged = presel.lazy().join(ref, on=keys, how="inner").collect()
    p = score(model, feats, merged)
    r = merged["prob_TMM_iso_reweighted"].to_numpy()
    d = np.abs(p - r)
    print(f"[validate] {len(d)} nu_overlay events: max |dp| = {d.max():.2e}, "
          f"mean = {d.mean():.2e}; fraction > 1e-4: {(d > 1e-4).mean():.2e}")
    above_ref, above_new = (r > 0.95).sum(), (p > 0.95).sum()
    print(f"[validate] events above 0.95: reference {above_ref}, rescored {above_new}")
    return 0 if d.max() < 1e-3 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--detvar", default=str(INTER / "detvar_presel_df_train_vars.parquet"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, feats = load_model()
    if args.validate:
        return validate(model, feats)

    src = Path(args.detvar)
    cols = pl.scan_parquet(src).collect_schema().names()
    need = [c for c in KEEP if c in cols]
    df = pl.read_parquet(src, columns=list(dict.fromkeys(need + feats)))
    print(f"[score] {src.name}: {df.height} rows")
    p = score(model, feats, df)
    out = df.select(need).with_columns(pl.Series("prob_TMM_iso_reweighted", p))
    out_path = Path(args.out) if args.out else src.with_name(src.stem + "_scored.parquet")
    out.write_parquet(out_path)
    print(out.group_by("vartype").agg(pl.len(), (pl.col("prob_TMM_iso_reweighted") > 0.95).sum()
                                      .alias("n_above_0.95")).sort("vartype"))
    print(f"[score] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
