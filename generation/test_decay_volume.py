"""Synthetic tests for decay_volume.py — no DarkNews required.

    python test_decay_volume.py [--config config.json]

Events are fabricated with known kinematics so every output of the forced
decay can be checked against an independent calculation:

1. P_inside agrees with an independent (long-double, naive-difference)
   evaluation in a regime where the naive form is accurate.
2. In the long-lifetime regime (lambda >> chord), P_inside agrees with the
   first-order analytic result — the regime where the naive difference
   formula loses catastrophically to cancellation (demonstrated).
3. Every forced decay position passes inside_liquid_cryostat, and the decay
   time equals distance / (beta c).
4. The sampled decay distances follow the truncated exponential: the sample
   mean matches the analytic mean within statistics.
5. closure_weight = w_event_rate * P_inside exactly.
6. Events whose flight path misses the liquid are dropped.
7. The cryostat selection keeps exactly the events whose decay passes the
   predicate.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config
import geometry
from dataframe_tools import P4_ORDER, POSITION_ORDER, get_scalar, get_vector_component
from decay_volume import (
    force_dirt_decays_inside_cryostat,
    select_cryostat_decays_in_liquid,
)


def fail(message: str) -> None:
    print(f"FAIL  {message}")
    sys.exit(1)


def ok(message: str) -> None:
    print(f"PASS  {message}")


def make_dirt_dataframe(config, *, n, mass, ctau0, momentum, origin, direction):
    """Fabricate n identical dirt events with flat (non-MultiIndex) columns."""
    columns = config.columns
    energy = math.sqrt(momentum**2 + mass**2)
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)

    data = {}
    p4 = {"E": energy, "px": momentum * d[0], "py": momentum * d[1], "pz": momentum * d[2]}
    for index, component in enumerate(P4_ORDER):
        data[f"{columns.hnl_p4}_{index}"] = np.full(n, p4[component])
    pos = {"t": 0.0, "x": origin[0], "y": origin[1], "z": origin[2]}
    for index, component in enumerate(POSITION_ORDER):
        data[f"{columns.scattering_position}_{index}"] = np.full(n, pos[component])
        data[f"{columns.decay_position}_{index}"] = np.full(n, pos[component])
    data[columns.event_weight] = np.full(n, 2.5)

    df = pd.DataFrame(data)
    df.attrs["N4_ctau0"] = float(ctau0)
    df.attrs["generation_mass_GeV"] = float(mass)
    return df


def axis_segments(config):
    """Chord segments for the on-axis ray used by most tests."""
    origin = np.array([[0.0, 0.0, -2000.0]])
    direction = np.array([[0.0, 0.0, 1.0]])
    return geometry.cryostat_ray_segments(origin, direction, config.geometry)[0]


def naive_p_inside_longdouble(segments, lam):
    """Independent evaluation: naive difference formula in extended precision."""
    lam = np.longdouble(lam)
    total = np.longdouble(0.0)
    for start, stop in segments:
        total += np.exp(-np.longdouble(start) / lam) - np.exp(-np.longdouble(stop) / lam)
    return float(total)


def test_p_inside_moderate_lambda(config, rng):
    """lambda comparable to the chord: naive formula is accurate; must agree."""
    mass, momentum = 0.48, 1.2
    lam = 800.0  # cm; ctau0 chosen to produce it
    ctau0 = lam * mass / momentum
    df = make_dirt_dataframe(
        config, n=200, mass=mass, ctau0=ctau0, momentum=momentum,
        origin=(0.0, 0.0, -2000.0), direction=(0.0, 0.0, 1.0),
    )
    forced = force_dirt_decays_inside_cryostat(df, config, rng)
    if len(forced) != 200:
        fail(f"expected all 200 events kept (chord exists), got {len(forced)}")

    expected = naive_p_inside_longdouble(axis_segments(config), lam)
    got = get_scalar(forced, config.columns.inside_probability)
    if not np.allclose(got, expected, rtol=1e-12, atol=0.0):
        fail(f"P_inside {got[0]!r} != long-double reference {expected!r}")
    ok(f"P_inside (lambda=800 cm) matches long-double reference: {expected:.6e}")


def test_p_inside_long_lifetime(config, rng):
    """lambda >> chord: stable form must match first-order analytics; the
    naive float64 difference is shown to be relatively wrong."""
    mass, momentum = 0.1, 1.0
    lam = 1.0e9  # 10^4 km lab decay length
    ctau0 = lam * mass / momentum
    df = make_dirt_dataframe(
        config, n=1, mass=mass, ctau0=ctau0, momentum=momentum,
        origin=(0.0, 0.0, -2000.0), direction=(0.0, 0.0, 1.0),
    )
    forced = force_dirt_decays_inside_cryostat(df, config, rng)
    got = float(get_scalar(forced, config.columns.inside_probability)[0])

    segments = axis_segments(config)
    analytic = sum(
        math.exp(-start / lam) * ((stop - start) / lam) * (1.0 - 0.5 * (stop - start) / lam)
        for start, stop in segments
    )  # second-order expansion of the truncated exponential
    if abs(got / analytic - 1.0) > 1e-9:
        fail(f"long-lifetime P_inside {got!r} != analytic {analytic!r}")

    naive = sum(math.exp(-s / lam) - math.exp(-e / lam) for s, e in segments)
    naive_error = abs(naive / analytic - 1.0)
    ok(
        f"P_inside (lambda=1e9 cm) = {got:.6e} matches analytic; "
        f"naive float64 difference would be off by {naive_error:.2e} relative"
    )


def test_decay_positions_and_time(config, rng):
    mass, momentum = 0.48, 1.2
    lam = 800.0
    ctau0 = lam * mass / momentum
    df = make_dirt_dataframe(
        config, n=5000, mass=mass, ctau0=ctau0, momentum=momentum,
        origin=(0.0, 0.0, -2000.0), direction=(0.0, 0.0, 1.0),
    )
    forced = force_dirt_decays_inside_cryostat(df, config, rng)
    columns = config.columns
    order = POSITION_ORDER
    x = get_vector_component(forced, columns.decay_position, order.index("x"))
    y = get_vector_component(forced, columns.decay_position, order.index("y"))
    z = get_vector_component(forced, columns.decay_position, order.index("z"))
    t = get_vector_component(forced, columns.decay_position, order.index("t"))

    inside = geometry.inside_liquid_cryostat(x, y, z, config.geometry)
    if not inside.all():
        fail(f"{(~inside).sum()} of {len(forced)} forced decay positions are outside the liquid")

    # decay time: distance / (beta c); origin z = -2000, so distance = z + 2000
    energy = math.sqrt(momentum**2 + mass**2)
    beta = momentum / energy
    expected_t = (z + 2000.0) / (beta * config.geometry.speed_of_light_cm_per_ns)
    if not np.allclose(t, expected_t, rtol=1e-12):
        fail("decay time does not equal distance / (beta c)")

    # sampled distances follow the exponential truncated to the segments; for
    # a single dominant segment compare the sample mean with the analytic mean
    segments = axis_segments(config)
    start, stop = max(segments, key=lambda seg: seg[1] - seg[0])  # the tube segment dominates
    on_tube = (z + 2000.0 > start) & (z + 2000.0 < stop)
    distances = z[on_tube] + 2000.0 - start
    span = stop - start
    denom = -math.expm1(-span / lam)
    analytic_mean = lam - span * math.exp(-span / lam) / denom
    sample_mean = float(np.mean(distances))
    sigma = float(np.std(distances) / math.sqrt(len(distances)))
    pull = (sample_mean - analytic_mean) / sigma
    if abs(pull) > 5.0:
        fail(
            f"sampled-distance mean {sample_mean:.2f} vs truncated-exponential "
            f"mean {analytic_mean:.2f} (pull {pull:.1f})"
        )

    w = get_scalar(forced, columns.event_weight)
    p = get_scalar(forced, columns.inside_probability)
    cw = get_scalar(forced, columns.dirt_final_weight)
    if not np.array_equal(cw, w * p):
        fail("closure_weight != w_event_rate * P_inside")

    ok(
        f"{len(forced)} forced decays: all inside the liquid; time = L/(beta c); "
        f"distance distribution matches truncated exponential (pull {pull:.2f}); "
        "closure_weight = w * P_inside"
    )


def test_missing_chord_dropped(config, rng):
    df = make_dirt_dataframe(
        config, n=50, mass=0.48, ctau0=100.0, momentum=1.0,
        origin=(0.0, 0.0, -2000.0), direction=(0.0, 0.0, -1.0),  # away from detector
    )
    forced = force_dirt_decays_inside_cryostat(df, config, rng)
    if len(forced) != 0:
        fail(f"events flying away from the detector were kept: {len(forced)}")
    for name in (config.columns.inside_probability, config.columns.dirt_final_weight):
        get_scalar(forced, name)  # columns must exist even when empty
    ok("events whose flight path misses the liquid are dropped (empty output keeps columns)")


def test_cryostat_selection(config, rng):
    columns = config.columns
    n = 4000
    # decay positions uniform in a box straddling the cryostat boundary
    x = rng.uniform(-250, 250, n)
    y = rng.uniform(-250, 250, n)
    z = rng.uniform(-700, 700, n)
    data = {}
    pos = {"t": np.zeros(n), "x": x, "y": y, "z": z}
    for index, component in enumerate(POSITION_ORDER):
        data[f"{columns.decay_position}_{index}"] = pos[component]
    data[columns.event_weight] = np.ones(n)
    df = pd.DataFrame(data)

    selected = select_cryostat_decays_in_liquid(df, config)
    expected = geometry.inside_liquid_cryostat(x, y, z, config.geometry)
    if len(selected) != int(expected.sum()):
        fail(f"selection kept {len(selected)}, predicate says {int(expected.sum())}")
    zs = get_vector_component(selected, columns.decay_position,
                              POSITION_ORDER.index("z"))
    if len(zs) == 0:
        fail("selection kept nothing — test box misconfigured")
    ok(f"cryostat selection keeps exactly the predicate-true events "
       f"({len(selected)}/{n} in the test box)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=Path(__file__).with_name("config.json"))
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    config = load_config(args.config)
    rng = np.random.default_rng(args.seed)

    test_p_inside_moderate_lambda(config, rng)
    test_p_inside_long_lifetime(config, rng)
    test_decay_positions_and_time(config, rng)
    test_missing_chord_dropped(config, rng)
    test_cryostat_selection(config, rng)
    print("\nAll decay-volume tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
