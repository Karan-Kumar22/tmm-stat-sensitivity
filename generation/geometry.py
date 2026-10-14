"""MicroBooNE liquid-argon cryostat geometry.

Everything here is in the DarkNews MicroBooNE frame: the cryostat tube is
centred at the origin with the beam along +z. The cryostat is a horizontal
cylinder ("tube") closed by two spherical caps and filled with liquid argon up
to a constant height — ``liquid_argon_fill_fraction`` of the tube diameter —
so the liquid surface is the horizontal plane y = liquid_y_max.

Two views of the same volume are provided and must agree with each other:

- ``inside_liquid_cryostat(x, y, z, geometry)`` — point predicate. Used to
  keep cryostat events whose sampled HNL decay is in the liquid.
- ``cryostat_ray_segments(origin, direction, geometry)`` — the positive
  ray-parameter intervals a straight line spends inside the liquid. Used to
  force dirt HNL decays onto their chord through the detector.

``test_geometry.py`` checks the two representations against each other on
random points, and checks that generated events start inside the volume
(plan section 5.3).

Ported from the validated routines in
TMM_Analysis/Scripts/tmm_direct_production/tmm_big_program_direct.py.
"""

from __future__ import annotations

import math

import numpy as np

from config import GeometrySettings


def cryostat_constants(geometry: GeometrySettings) -> dict[str, float]:
    """Derived constants of the capped-cylinder cryostat.

    Each spherical cap has radius ``cap_radius`` and half-opening angle
    ``theta``; its centre sits on the z axis so that the cap rim meets the
    tube end (up to the small configured ``gap``). ``height`` is the sagitta
    of the cap — how far it bulges beyond the tube end.

    These are the same centres implied by the DarkNews MicroBooNE geometry.
    """
    radius_tube = geometry.cryostat_tube_radius_cm
    length_tube = geometry.cryostat_tube_length_cm
    radius_cap = geometry.cryostat_cap_radius_cm
    theta = math.radians(geometry.cryostat_cap_theta_deg)
    gap = geometry.cryostat_cap_gap_cm
    fill = geometry.liquid_argon_fill_fraction

    ctheta = math.cos(theta)
    height = radius_cap * (1.0 - ctheta)

    front_center_z = -length_tube / 2.0 - gap + radius_cap * ctheta
    back_center_z = length_tube / 2.0 + gap + height - radius_cap
    front_cap_max_z = front_center_z - radius_cap + height
    back_cap_min_z = back_center_z + radius_cap - height
    liquid_y_max = -radius_tube + 2.0 * radius_tube * fill

    return {
        "radius_tube": radius_tube,
        "length_tube": length_tube,
        "radius_cap": radius_cap,
        "theta": theta,
        "height": height,
        "front_center_z": front_center_z,
        "back_center_z": back_center_z,
        "front_cap_max_z": front_cap_max_z,
        "back_cap_min_z": back_cap_min_z,
        "liquid_y_max": liquid_y_max,
    }


def inside_liquid_cryostat(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    geometry: GeometrySettings,
) -> np.ndarray:
    """Boolean mask: is each point inside the liquid-filled cryostat volume?

    The volume is (tube OR front cap OR back cap) AND below the liquid surface.
    """
    constants = cryostat_constants(geometry)
    rt = constants["radius_tube"]
    lt = constants["length_tube"]
    rc = constants["radius_cap"]

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    radial2 = x**2 + y**2
    tube = (radial2 < rt**2) & (z > -lt / 2.0) & (z < lt / 2.0)

    front_dz = z - constants["front_center_z"]
    front = (radial2 + front_dz**2 < rc**2) & (z < constants["front_cap_max_z"])

    back_dz = z - constants["back_center_z"]
    back = (radial2 + back_dz**2 < rc**2) & (z > constants["back_cap_min_z"])

    liquid = y < constants["liquid_y_max"]
    return (tube | front | back) & liquid


# ---------------------------------------------------------------------------
# Ray / volume intersection.
#
# A ray is  point(s) = origin + s * direction  with s >= 0 and unit
# ``direction``, so the ray parameter s is a distance in cm. Each primitive
# below returns per-event (start, stop) arrays for the interval of s spent
# inside it; an empty intersection is encoded as start=+inf, stop=-inf, which
# survives interval intersection arithmetic harmlessly.
# ---------------------------------------------------------------------------


def _line_coordinate_interval(
    origin: np.ndarray,
    direction: np.ndarray,
    lower: float | None,
    upper: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Interval of s satisfying lower <= origin + s*direction <= upper.

    ``origin``/``direction`` here are single coordinates (e.g. y or z), one
    entry per event. Either bound may be None (unbounded). A ray moving
    parallel to the bounded coordinate (direction ~ 0) is inside for all s or
    for none, depending on where it sits.
    """
    n = len(origin)
    start = np.full(n, -np.inf)
    stop = np.full(n, np.inf)
    eps = 1.0e-15

    moving = np.abs(direction) > eps
    stationary = ~moving

    if lower is not None:
        candidate = np.divide(lower - origin, direction, out=np.zeros(n), where=moving)
        lo = np.where(direction > 0.0, candidate, -np.inf)
        hi = np.where(direction < 0.0, candidate, np.inf)
        start = np.maximum(start, lo)
        stop = np.minimum(stop, hi)
        stop[stationary & (origin < lower)] = -np.inf

    if upper is not None:
        candidate = np.divide(upper - origin, direction, out=np.zeros(n), where=moving)
        lo = np.where(direction < 0.0, candidate, -np.inf)
        hi = np.where(direction > 0.0, candidate, np.inf)
        start = np.maximum(start, lo)
        stop = np.minimum(stop, hi)
        stop[stationary & (origin > upper)] = -np.inf

    return start, stop


def _intersect_intervals(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    return np.maximum(first[0], second[0]), np.minimum(first[1], second[1])


def _sphere_interval(
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    center_z: float,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interval of s inside a sphere centred at (0, 0, center_z).

    With unit direction the quadratic |origin + s*direction|^2 = R^2 reduces
    to s^2 + 2 b s + c = 0 with b = origin.direction, c = |origin|^2 - R^2.
    """
    ox = x0
    oy = y0
    oz = z0 - center_z
    b = ox * dx + oy * dy + oz * dz
    c = ox**2 + oy**2 + oz**2 - radius**2
    discriminant = b**2 - c

    valid = discriminant >= 0.0
    root = np.sqrt(np.clip(discriminant, 0.0, None))
    start = -b - root
    stop = -b + root
    start[~valid] = np.inf
    stop[~valid] = -np.inf
    return start, stop


def _cylinder_interval(
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    radius: float,
    zmin: float,
    zmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interval of s inside a z-axis cylinder of given radius, capped in z."""
    a = dx**2 + dy**2
    b = x0 * dx + y0 * dy
    c = x0**2 + y0**2 - radius**2

    start = np.full(len(x0), np.inf)
    stop = np.full(len(x0), -np.inf)

    moving = a > 1.0e-15
    discriminant = b**2 - a * c
    valid = moving & (discriminant >= 0.0)
    root = np.sqrt(np.clip(discriminant, 0.0, None))
    start[valid] = (-b[valid] - root[valid]) / a[valid]
    stop[valid] = (-b[valid] + root[valid]) / a[valid]

    # A ray parallel to the cylinder axis is inside radially for all s
    # (if it is within the radius at all), so only the z bounds cut it.
    parallel_inside = (~moving) & (c <= 0.0)
    start[parallel_inside] = -np.inf
    stop[parallel_inside] = np.inf

    return _intersect_intervals(
        (start, stop),
        _line_coordinate_interval(z0, dz, zmin, zmax),
    )


def cryostat_ray_segments(
    origin: np.ndarray,
    direction: np.ndarray,
    geometry: GeometrySettings,
) -> list[list[tuple[float, float]]]:
    """All positive ray-distance intervals inside the liquid argon.

    ``origin`` and ``direction`` are (n, 3) arrays; ``direction`` rows must
    be unit vectors. The volume is the union of tube and both caps, cut by
    the liquid surface. Because the configured cap gap is retained, an event
    can in principle have more than one disjoint segment.

    Returns, per event, a sorted list of merged (start, stop) pairs with
    start >= 0 (the part of the chord behind the origin is discarded — the
    HNL flies forward only).
    """
    x0, y0, z0 = origin.T
    dx, dy, dz = direction.T
    constants = cryostat_constants(geometry)

    y_interval = _line_coordinate_interval(
        y0, dy, lower=None, upper=constants["liquid_y_max"]
    )

    tube = _cylinder_interval(
        x0, y0, z0, dx, dy, dz,
        constants["radius_tube"],
        -constants["length_tube"] / 2.0,
        constants["length_tube"] / 2.0,
    )
    tube = _intersect_intervals(tube, y_interval)

    front = _sphere_interval(
        x0, y0, z0, dx, dy, dz,
        constants["front_center_z"],
        constants["radius_cap"],
    )
    front = _intersect_intervals(
        front,
        _line_coordinate_interval(
            z0, dz, lower=None, upper=constants["front_cap_max_z"]
        ),
    )
    front = _intersect_intervals(front, y_interval)

    back = _sphere_interval(
        x0, y0, z0, dx, dy, dz,
        constants["back_center_z"],
        constants["radius_cap"],
    )
    back = _intersect_intervals(
        back,
        _line_coordinate_interval(
            z0, dz, lower=constants["back_cap_min_z"], upper=None
        ),
    )
    back = _intersect_intervals(back, y_interval)

    primitive_intervals = np.stack(
        [np.column_stack(front), np.column_stack(tube), np.column_stack(back)],
        axis=1,
    )

    all_segments: list[list[tuple[float, float]]] = []
    for event_intervals in primitive_intervals:
        valid: list[tuple[float, float]] = []
        for start, stop in event_intervals:
            start = max(float(start), 0.0)
            stop = float(stop)
            if math.isfinite(start) and math.isfinite(stop) and stop > start:
                valid.append((start, stop))

        # Merge overlapping/adjacent intervals from the three primitives into
        # disjoint segments, sorted by entry distance.
        valid.sort(key=lambda pair: pair[0])
        merged: list[tuple[float, float]] = []
        for start, stop in valid:
            if not merged or start > merged[-1][1] + 1.0e-9:
                merged.append((start, stop))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        all_segments.append(merged)

    return all_segments


def to_detector_frame(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    geometry: GeometrySettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Translate DarkNews-frame positions to detector (TPC) coordinates.

    Pure translation: x_det in [0, 256.35], z_det in [0, 1036.8] for points
    inside the TPC. Used when writing HEPevt files and truth tables.
    """
    return (
        np.asarray(x, dtype=float) + geometry.detector_x_offset_cm,
        np.asarray(y, dtype=float) + geometry.detector_y_offset_cm,
        np.asarray(z, dtype=float) + geometry.detector_z_offset_cm,
    )
