"""Geometry self-tests.

Run without arguments for the synthetic tests (no DarkNews needed):

    python test_geometry.py [--config config.json]

Two families of checks:

1. Predicate <-> ray consistency. The liquid volume is implemented twice —
   as a point predicate (inside_liquid_cryostat) and as a ray intersector
   (cryostat_ray_segments). For random rays, points sampled inside the
   returned segments must satisfy the predicate, points sampled outside all
   segments must fail it, and a ray starting at an inside point must begin
   inside a segment (s = 0). Any frame or sign error breaks these.

2. Analytic spot checks. The beam-axis chord length and a few hand-computable
   crossings must match the constants.

Later, once real batches exist (plan section 5.3):

    python test_geometry.py --check-batch batches/cryostat_00000_truth.pkl

verifies that generated cryostat events start inside the liquid volume.
That mode is wired in when the batch format lands.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from config import load_config
import geometry


def fail(message: str) -> None:
    print(f"FAIL  {message}")
    sys.exit(1)


def ok(message: str) -> None:
    print(f"PASS  {message}")


def point_in_segments(s: float, segments: list[tuple[float, float]], edge: float) -> bool:
    """Is ray parameter s strictly inside any segment (away from edges)?"""
    return any(start + edge < s < stop - edge for start, stop in segments)


def test_segment_points_match_predicate(geo, rng, n_rays: int = 20000) -> None:
    """Sample points inside and outside returned segments; compare with predicate.

    Rays start from a shell far outside the cryostat with directions biased
    toward the detector so a good fraction actually hit it.
    """
    # Random origins on a large sphere, aimed at a random point near the origin.
    origins = rng.normal(size=(n_rays, 3))
    origins /= np.linalg.norm(origins, axis=1, keepdims=True)
    origins *= 2000.0
    targets = rng.uniform(-250.0, 250.0, size=(n_rays, 3))
    directions = targets - origins
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    segments = geometry.cryostat_ray_segments(origins, directions, geo)

    edge = 1.0e-6  # stay away from boundaries where predicate strictness differs
    n_hit = 0
    n_inside_checked = 0
    n_outside_checked = 0
    for i, event_segments in enumerate(segments):
        if not event_segments:
            continue
        n_hit += 1

        # points strictly inside each segment must pass the predicate
        for start, stop in event_segments:
            for frac in (0.25, 0.5, 0.75):
                s = start + frac * (stop - start)
                p = origins[i] + s * directions[i]
                inside = geometry.inside_liquid_cryostat(
                    np.array([p[0]]), np.array([p[1]]), np.array([p[2]]), geo
                )[0]
                if not inside:
                    fail(
                        f"ray {i}: point at s={s:.3f} inside segment "
                        f"({start:.3f}, {stop:.3f}) fails the predicate at {p}"
                    )
                n_inside_checked += 1

        # points outside all segments (but on the ray) must fail the predicate
        first_start = event_segments[0][0]
        last_stop = event_segments[-1][1]
        candidates = [first_start - 5.0, last_stop + 5.0]
        # midpoints of gaps between disjoint segments
        for (s0, s1) in zip(event_segments[:-1], event_segments[1:]):
            candidates.append(0.5 * (s0[1] + s1[0]))
        for s in candidates:
            if s <= 0.0 or point_in_segments(s, event_segments, edge):
                continue
            p = origins[i] + s * directions[i]
            inside = geometry.inside_liquid_cryostat(
                np.array([p[0]]), np.array([p[1]]), np.array([p[2]]), geo
            )[0]
            if inside:
                fail(
                    f"ray {i}: point at s={s:.3f} outside all segments "
                    f"{event_segments} passes the predicate at {p}"
                )
            n_outside_checked += 1

    if n_hit < n_rays * 0.05:
        fail(f"only {n_hit}/{n_rays} test rays hit the cryostat — test geometry broken")
    ok(
        f"segment<->predicate consistency: {n_hit} hitting rays, "
        f"{n_inside_checked} inside points, {n_outside_checked} outside points"
    )


def test_rays_from_inside_start_inside(geo, rng, n_points: int = 5000) -> None:
    """A ray whose origin passes the predicate must start inside a segment."""
    # rejection-sample points inside the liquid
    points = []
    while len(points) < n_points:
        cand = np.column_stack(
            [
                rng.uniform(-200.0, 200.0, size=4 * n_points),
                rng.uniform(-200.0, 150.0, size=4 * n_points),
                rng.uniform(-620.0, 620.0, size=4 * n_points),
            ]
        )
        mask = geometry.inside_liquid_cryostat(cand[:, 0], cand[:, 1], cand[:, 2], geo)
        points.extend(cand[mask][: n_points - len(points)])
    points = np.asarray(points)

    directions = rng.normal(size=(len(points), 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    segments = geometry.cryostat_ray_segments(points, directions, geo)
    for i, event_segments in enumerate(segments):
        if not event_segments:
            fail(f"inside point {points[i]} produced no segments")
        start = event_segments[0][0]
        if start > 1.0e-6:
            fail(
                f"inside point {points[i]} first segment starts at s={start:.6f}, "
                "expected 0 (ray from inside must begin inside)"
            )
    ok(f"rays from {len(points)} inside points all begin inside a segment (s=0)")


def test_beam_axis_chord(geo) -> None:
    """Analytic check: chord along the beam axis through the centre."""
    constants = geometry.cryostat_constants(geo)
    origin = np.array([[0.0, 0.0, -5000.0]])
    direction = np.array([[0.0, 0.0, 1.0]])
    segments = geometry.cryostat_ray_segments(origin, direction, geo)[0]

    # The configured cap gap leaves a ~0.37 cm uncovered sliver at each
    # tube/cap junction, so the axial ray crosses three disjoint segments:
    # front cap | tube | back cap.
    if len(segments) != 3:
        fail(f"beam-axis ray expected 3 segments (cap|tube|cap), got {segments}")

    total = sum(stop - start for start, stop in segments)
    tip_to_tip = (
        (constants["back_center_z"] + constants["radius_cap"])
        - (constants["front_center_z"] - constants["radius_cap"])
    )
    expected = tip_to_tip - 2.0 * geo.cryostat_cap_gap_cm
    if abs(total - expected) > 1.0e-6:
        fail(
            f"beam-axis chord length {total:.4f} cm != expected "
            f"{expected:.4f} cm (tip-to-tip {tip_to_tip:.4f} minus two "
            f"{geo.cryostat_cap_gap_cm} cm junction gaps)"
        )
    ok(
        f"beam-axis chord: 3 segments (cap|tube|cap), total {total:.2f} cm = "
        f"tip-to-tip {tip_to_tip:.2f} minus two junction gaps"
    )


def test_liquid_surface(geo) -> None:
    """A vertical ray must exit exactly at the liquid surface."""
    constants = geometry.cryostat_constants(geo)
    origin = np.array([[0.0, -150.0, 0.0]])
    direction = np.array([[0.0, 1.0, 0.0]])
    segments = geometry.cryostat_ray_segments(origin, direction, geo)[0]
    if not segments:
        fail("vertical ray through the centre found no liquid")
    exit_y = origin[0][1] + segments[-1][1]
    if abs(exit_y - constants["liquid_y_max"]) > 1.0e-6:
        fail(
            f"vertical ray exits liquid at y={exit_y:.4f}, "
            f"expected liquid surface {constants['liquid_y_max']:.4f}"
        )
    ok(f"vertical ray exits at the liquid surface y={exit_y:.3f} cm")


def test_downstream_miss(geo) -> None:
    """A ray pointing away from the detector must produce no segments."""
    origin = np.array([[0.0, 0.0, -5000.0]])
    direction = np.array([[0.0, 0.0, -1.0]])
    segments = geometry.cryostat_ray_segments(origin, direction, geo)[0]
    if segments:
        fail(f"ray pointing away from the detector returned segments: {segments}")
    ok("ray pointing away from the detector returns no segments")


def parse_hepevt_positions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Scattering (HNL line) and decay (photon line) vertices from a HEPevt file.

    Returns two (n, 3) arrays in detector-frame cm. Particle-line fields:
    status pdg m1 m2 d1 d2 px py pz E mass x y z t -> x,y,z at indices 11..13.
    """
    lines = path.read_text().splitlines()
    scattering, decay = [], []
    i = 0
    while i < len(lines):
        header = lines[i].split()
        if len(header) != 2:
            fail(f"{path}: malformed header at line {i + 1}: {lines[i]!r}")
        n_particles = int(header[1])
        block = lines[i + 1 : i + 1 + n_particles]
        hnl = block[0].split()
        photon = block[1].split()
        scattering.append([float(hnl[11]), float(hnl[12]), float(hnl[13])])
        decay.append([float(photon[11]), float(photon[12]), float(photon[13])])
        i += 1 + n_particles
    return np.asarray(scattering), np.asarray(decay)


def check_hepevt_batch(path: Path, geo) -> None:
    """Plan section 5.3 on real generated events.

    Decay vertices must be inside the liquid for both components (that is
    what the pipeline selects/forces). Scattering vertices must be inside
    the liquid for cryostat batches and outside the cryostat for dirt
    batches. Any frame or sign error in the detector transform fails this
    immediately.
    """
    component = path.name.split("_")[0]
    if component not in ("cryostat", "dirt"):
        fail(f"cannot infer component from file name {path.name!r}")

    scattering, decay = parse_hepevt_positions(path)
    # back from detector frame to the DarkNews frame the predicate lives in
    offsets = np.array([
        geo.detector_x_offset_cm, geo.detector_y_offset_cm, geo.detector_z_offset_cm,
    ])
    scattering -= offsets
    decay -= offsets

    decay_inside = geometry.inside_liquid_cryostat(
        decay[:, 0], decay[:, 1], decay[:, 2], geo
    )
    if not decay_inside.all():
        fail(
            f"{path.name}: {(~decay_inside).sum()} of {len(decay)} decay "
            "vertices are outside the liquid argon"
        )

    scatter_inside = geometry.inside_liquid_cryostat(
        scattering[:, 0], scattering[:, 1], scattering[:, 2], geo
    )
    if component == "cryostat":
        if not scatter_inside.all():
            fail(
                f"{path.name}: {(~scatter_inside).sum()} of {len(scattering)} "
                "cryostat scattering vertices are outside the liquid "
                "(coordinate transform or generation volume error)"
            )
        ok(
            f"{path.name}: {len(decay)} events — all scattering AND decay "
            "vertices inside the liquid argon"
        )
    else:
        if scatter_inside.any():
            fail(
                f"{path.name}: {scatter_inside.sum()} dirt scattering vertices "
                "claim to be inside the liquid argon"
            )
        ok(
            f"{path.name}: {len(decay)} events — all scattering vertices in the "
            "dirt, all decay vertices inside the liquid argon"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=Path(__file__).with_name("config.json"))
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--check-batch",
        default=None,
        nargs="+",
        help="path(s) to batch .hepevt files; runs the events-start-inside "
        "check (plan section 5.3) instead of the synthetic tests",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    geo = config.geometry
    rng = np.random.default_rng(args.seed)

    if args.check_batch is not None:
        for batch in args.check_batch:
            check_hepevt_batch(Path(batch), geo)
        print("\nAll batch checks passed.")
        return 0

    test_beam_axis_chord(geo)
    test_liquid_surface(geo)
    test_downstream_miss(geo)
    test_rays_from_inside_start_inside(geo, rng)
    test_segment_points_match_predicate(geo, rng)
    print("\nAll geometry tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
