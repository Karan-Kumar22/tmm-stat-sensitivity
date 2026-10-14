"""Export NGEM-format TMM weight parquets for every grid point."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from export_ngem_weights import export_point

GRID = Path("/nevis/riverside/data/karan/tmm_points/grid")
OUT = Path(__file__).resolve().parent / "ngem_weights"
_STATE: dict = {}


def _init() -> None:
    from config import load_config
    from merged_binning import load_analysis_binning

    import os
    config = load_config(Path(os.environ.get("TMM_MAPPING_CONFIG",
        Path(__file__).resolve().parent / "config.json")))
    _STATE.update(config=config, binning=load_analysis_binning(config),
                  source=pd.read_pickle(
                      Path(__file__).resolve().parent / "iso_truth.pkl"))


def one(name: str) -> dict:
    m4_tag, mu_tag = name.removeprefix("m").split("_mu")
    try:
        result = export_point(GRID / f"point_{name}", OUT, m4_tag, mu_tag,
                              config=_STATE["config"], binning=_STATE["binning"],
                              source=_STATE["source"])
        return {"name": name, **{k: result[k] for k in
                                 ("sum_weight", "kept", "unsupported", "batches")}}
    except Exception as exc:
        return {"name": name, "error": str(exc)[:200]}


def main() -> int:
    global GRID, OUT
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points-dir", default=str(GRID))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    GRID = Path(args.points_dir)
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name.removeprefix("point_") for p in GRID.glob("point_*"))
    print(f"[export] {len(names)} points -> {OUT}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=16, initializer=_init) as pool:
        futures = [pool.submit(one, n) for n in names]
        for i, f in enumerate(as_completed(futures), 1):
            rows.append(f.result())
            if i % 100 == 0:
                print(f"[export] {i}/{len(names)}", flush=True)
    table = pd.DataFrame(rows).sort_values("name")
    table.to_csv(OUT / "export_summary.csv", index=False)
    bad = table.get("error")
    n_bad = int(bad.notna().sum()) if bad is not None else 0
    print(f"[export] done: {len(table) - n_bad} OK, {n_bad} errors")
    print(f"[export] signal rate at 6.8e20 POT: min {table.sum_weight.min():.3g}, "
          f"median {table.sum_weight.median():.3g}, max {table.sum_weight.max():.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
