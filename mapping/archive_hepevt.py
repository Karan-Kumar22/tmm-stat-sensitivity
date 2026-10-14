"""Archive completed HEPevt batch files per grid point.

For each point under the grid directory: find loose batches/*.hepevt whose
batch is COMPLETE (matching *_truth.pkl exists for that component+index —
generation writes hepevt before truth, so truth present = batch finished),
tar+gzip them into batches/hepevt_<seq>.tar.gz, VERIFY the archive lists
exactly the expected members, and only then delete the loose files.

Safe to run while generation is active (generation only appends new files)
and safe to re-run any time: already-archived files are gone, new loose
files get swept into the next sequence number.

    python archive_hepevt.py [--workers 4] [--point NAME]  (default: all)
"""

from __future__ import annotations

import argparse
import subprocess
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

GRID = Path("/nevis/riverside/data/karan/tmm_points/grid")


def archive_point(point: Path) -> str:
    batches = point / "batches"
    if not batches.is_dir():
        return f"{point.name}: no batches dir"
    loose = sorted(batches.glob("*.hepevt"))
    ready = [f for f in loose
             if f.with_name(f.stem + "_truth.pkl").is_file()]
    if not ready:
        return f"{point.name}: nothing to archive"

    seq = 0
    while (batches / f"hepevt_{seq:03d}.tar.gz").exists():
        seq += 1
    archive = batches / f"hepevt_{seq:03d}.tar.gz"
    tmp = archive.with_suffix(".gz.tmp")

    size_before = sum(f.stat().st_size for f in ready)
    result = subprocess.run(
        ["tar", "-czf", str(tmp)] + [f.name for f in ready],
        cwd=batches, capture_output=True, text=True)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        return f"{point.name}: TAR FAILED: {result.stderr[:200]}"

    # verify: the archive must list exactly the files we put in
    try:
        with tarfile.open(tmp, "r:gz") as handle:
            members = sorted(m.name for m in handle.getmembers())
    except (tarfile.TarError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        return f"{point.name}: VERIFY FAILED: {exc}"
    expected = sorted(f.name for f in ready)
    if members != expected:
        tmp.unlink(missing_ok=True)
        return f"{point.name}: VERIFY FAILED: member list mismatch"

    tmp.rename(archive)
    for f in ready:
        f.unlink()
    size_after = archive.stat().st_size
    return (f"{point.name}: archived {len(ready)} files, "
            f"{size_before/1e9:.2f} -> {size_after/1e9:.2f} GB "
            f"({archive.name})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--point", default=None,
                        help="archive only this point (name without 'point_')")
    args = parser.parse_args()

    if args.point:
        points = [GRID / f"point_{args.point}"]
    else:
        points = sorted(p for p in GRID.glob("point_*") if p.is_dir())
    print(f"[archive] {len(points)} points, {args.workers} workers", flush=True)

    freed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(archive_point, p): p for p in points}
        for i, future in enumerate(as_completed(futures), 1):
            message = future.result()
            print(f"[archive] {i}/{len(points)} {message}", flush=True)
            if "FAILED" in message:
                print("[archive] stopping on failure"); return 1
    print("[archive] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
