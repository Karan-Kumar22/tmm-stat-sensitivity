"""Full-grid production driver: converge every (m4, mu_tr) point.

The grid: 31 log-spaced masses (0.01..2.0 GeV) x 32 log-spaced couplings
(2e-8..1e-5 GeV^-1) = 992 points. Each point runs converge_point.py (add
batches until the mapping validates; PASS / PLATEAU / EMPTY / LIMIT), with
several points in parallel.

Fully resumable: grid_registry.csv records every finished point; on restart,
finished points are skipped. Stop any time with Ctrl-C / tmux kill-session —
at worst one in-flight batch per worker is regenerated deterministically.

    python run_grid.py [--workers 8] [--registry grid_registry.csv] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

POINTS_DIR = Path("/nevis/riverside/data/karan/tmm_points/grid")   # default; see --points-dir
MAPPING_DIR = Path(__file__).resolve().parent

MASSES = np.logspace(np.log10(0.01), np.log10(2.0), 31)
TMMS = np.logspace(np.log10(2.0e-8), np.log10(1.0e-5), 32)


def point_name(mass: float, tmm: float) -> str:
    return f"m{mass:.5g}_mu{tmm:.5g}"


def run_point(task: tuple[float, float, int, float | None, float, str]) -> dict:
    mass, tmm, max_batches, min_neff, rate_floor, points_dir = task
    name = point_name(mass, tmm)
    point = Path(points_dir) / f"point_{name}"
    log_path = point / "converge.log"
    point.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, str(MAPPING_DIR / "converge_point.py"),
        "--point", str(point), "--mass", f"{mass:.6g}", "--tmm", f"{tmm:.6g}",
        "--max-batches", str(max_batches),
    ]
    if min_neff is not None:
        command += ["--min-neff", f"{min_neff:g}", "--rate-floor", f"{rate_floor:g}"]
    with log_path.open("a") as log:
        result = subprocess.run(command, cwd=MAPPING_DIR, stdout=log,
                                stderr=subprocess.STDOUT)

    # parse the verdict from the log tail
    status, chi2_ndf, batches = "ERROR", "", ""
    try:
        lines = log_path.read_text().splitlines()
        for line in reversed(lines):
            if "[converge]" not in line:
                continue
            for verdict in ("PASS", "PLATEAU", "EMPTY", "LIMIT"):
                if verdict in line:
                    status = verdict
                    break
            if status != "ERROR":
                break
        for line in reversed(lines):
            if "batches, chi2/ndf" in line:
                batches = line.split("batches")[0].split(":")[-1].strip()
                chi2_ndf = line.split("= ")[-1].split(",")[0].strip()
                break
            if "batches, min N_eff" in line:
                batches = line.split("batches")[0].split(":")[-1].strip()
                chi2_ndf = "neff " + line.split("= ")[-1].split(" over")[0].strip()
                break
    except OSError:
        pass
    if result.returncode not in (0, 1):
        status = "ERROR"

    return {
        "name": name, "mass": f"{mass:.6g}", "tmm": f"{tmm:.6g}",
        "status": status, "chi2_ndf": chi2_ndf, "batches": batches,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--registry", default=str(MAPPING_DIR / "grid_registry.csv"))
    parser.add_argument("--limit", type=int, default=None,
                        help="run at most N points this invocation (for testing)")
    parser.add_argument("--max-batches", type=int, default=8,
                        help="per-point batch cap forwarded to converge_point.py")
    parser.add_argument("--retry-plateau-above", type=float, default=None,
                        help="also retry PLATEAU points whose recorded chi2/ndf "
                             "exceeds this value (suspected false plateaus)")
    parser.add_argument("--min-neff", type=float, default=None,
                        help="statistics campaign: forward to converge_point "
                             "(generate until rate-carrying cells reach this N_eff)")
    parser.add_argument("--rate-floor", type=float, default=1.0e-3)
    parser.add_argument("--points-dir", default=str(POINTS_DIR),
                        help="root directory for the point directories "
                             "(default: the original grid)")
    args = parser.parse_args()

    registry = Path(args.registry)
    done: dict[str, dict] = {}
    if registry.is_file():
        with registry.open() as handle:
            for row in csv.DictReader(handle):
                done[row["name"]] = row

    tasks = []
    for mass in MASSES:
        for tmm in TMMS:
            name = point_name(mass, tmm)
            row = done.get(name)
            if row is not None and row["status"] in ("PASS", "PLATEAU", "EMPTY"):
                retry = (args.retry_plateau_above is not None
                         and row["status"] == "PLATEAU" and row["chi2_ndf"]
                         and float(row["chi2_ndf"]) > args.retry_plateau_above)
                if not retry:
                    continue
            tasks.append((float(mass), float(tmm), args.max_batches,
                          args.min_neff, args.rate_floor, args.points_dir))
    if args.limit:
        tasks = tasks[: args.limit]

    total = len(MASSES) * len(TMMS)
    print(f"[grid] {total} points total, {len(done)} recorded, "
          f"{len(tasks)} to run, {args.workers} workers", flush=True)
    if not tasks:
        print("[grid] nothing to do")
        return 0

    fieldnames = ["name", "mass", "tmm", "status", "chi2_ndf", "batches", "finished_at"]
    write_header = not registry.is_file()
    with registry.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        completed = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_point, task): task for task in tasks}
            for future in as_completed(futures):
                row = future.result()
                writer.writerow(row)
                handle.flush()
                completed += 1
                print(f"[grid] {completed}/{len(tasks)}  {row['name']:24s} "
                      f"{row['status']:8s} chi2/ndf {row['chi2_ndf']} "
                      f"({row['batches']} batches)", flush=True)

    print("[grid] invocation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
