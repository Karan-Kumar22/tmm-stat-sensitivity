"""Decay-volume physics: which HNL decays end up in the liquid argon.

Two operations, one per sample component:

**Cryostat** (``select_cryostat_decays_in_liquid``): DarkNews generated the
sample with ``enforce_prompt=False``, so every event already carries a decay
position sampled from the HNL lifetime. We simply keep the events whose decay
is inside the liquid-filled cryostat; ``w_event_rate`` remains the full weight.

**Dirt** (``force_dirt_decays_inside_cryostat``): the HNL is produced far
upstream, and the natural probability of decaying inside the detector is tiny.
Instead of wasting nearly every generated event, we *force* the decay onto the
chord the flight path cuts through the liquid argon (importance sampling), and
multiply the weight by the probability that nature would have done the same:

    P_inside = sum over chord segments of
               exp(-L_enter/lambda) * (-expm1(-(L_exit - L_enter)/lambda))

with lambda = (p/m) * ctau0 the per-event lab decay length. The expm1 form is
required: the naive exp(-L_enter/lambda) - exp(-L_exit/lambda) cancels
catastrophically when the chord is short compared with lambda, which is the
common regime (long lifetime, small mu_tr). The decay point itself is drawn
from the exponential distribution *truncated to the segment*:

    L = L_enter - lambda * log1p(-u * (1 - exp(-(L_exit-L_enter)/lambda)))

Events whose chord misses the liquid are dropped (P_inside = 0 exactly).
The forced events get:

- ``pos_decay``  overwritten with the sampled decay point (and decay time
  t_scatt + L / (beta c)),
- ``P_inside``       the summed inside-probability (new column),
- ``closure_weight`` = w_event_rate * P_inside (new column) — the physical
  weight of the forced event. Histogramming with closure_weight gives the true
  rate of dirt HNLs decaying in the detector.

Ported from the validated production code; randomness comes from a
numpy Generator supplied by the caller (seeded per batch by batch_store).https://vscode-remote+ssh-002dremote-002bhopper.vscode-resource.vscode-cdn.net/nevis/riverside/data/karan/tmm_mapping/slides/binning_slices_z.png?version%3D1788553934366
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config import Config
from darknews_interface import get_ctau0_cm
from dataframe_tools import (
    POSITION_ORDER,
    extract_p4,
    extract_position,
    get_scalar,
    set_scalar,
    set_vector_component,
)
import geometry


class DecayVolumeError(RuntimeError):
    """Raised when the input dataframe cannot be processed."""


def select_cryostat_decays_in_liquid(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Keep cryostat events whose sampled HNL decay is inside the liquid argon."""
    columns = config.columns
    decay = extract_position(df, columns.decay_position)
    mask = geometry.inside_liquid_cryostat(decay.x, decay.y, decay.z, config.geometry)
    result = df.loc[mask].copy().reset_index(drop=True)
    result.attrs = dict(df.attrs)
    result.attrs["decay_volume_selection"] = "liquid_cryostat"
    result.attrs["decay_volume_input_rows"] = int(len(df))
    return result


def force_dirt_decays_inside_cryostat(
    df: pd.DataFrame,
    config: Config,
    rng: np.random.Generator,
    mass: float | None = None,
) -> pd.DataFrame:
    """Condition dirt-produced HNLs to decay inside the liquid argon.

    ``mass`` defaults to the value stamped into df.attrs by
    generate_darknews_sample. ``rng`` must be a dedicated per-batch Generator
    so results are reproducible from (base seed, batch index).
    """
    columns = config.columns

    if mass is None:
        mass = df.attrs.get("generation_mass_GeV")
    if mass is None or not mass > 0.0:
        raise DecayVolumeError(
            "HNL mass unavailable: pass mass=... or generate the dataframe "
            "through generate_darknews_sample."
        )

    hnl = extract_p4(df, columns.hnl_p4)
    scattering = extract_position(df, columns.scattering_position)

    momentum = hnl.momentum
    direction = np.column_stack(
        [
            np.divide(hnl.px, momentum, out=np.zeros_like(hnl.px), where=momentum > 0),
            np.divide(hnl.py, momentum, out=np.zeros_like(hnl.py), where=momentum > 0),
            np.divide(hnl.pz, momentum, out=np.zeros_like(hnl.pz), where=momentum > 0),
        ]
    )
    origin = np.column_stack([scattering.x, scattering.y, scattering.z])

    segments = geometry.cryostat_ray_segments(origin, direction, config.geometry)

    ctau0 = get_ctau0_cm(df)
    lab_length = np.divide(
        momentum * ctau0,
        mass,
        out=np.zeros_like(momentum),
        where=momentum > 0.0,
    )

    keep_indices: list[int] = []
    sampled_distance: list[float] = []
    inside_probability: list[float] = []

    for i, event_segments in enumerate(segments):
        lam = float(lab_length[i])
        if not event_segments or not math.isfinite(lam) or lam <= 0.0:
            continue

        # Stable per-segment probability of decaying within [start, stop]:
        # survive to the entry point, decay before the exit.
        probabilities = np.array(
            [
                math.exp(-start / lam) * (-math.expm1(-(stop - start) / lam))
                for start, stop in event_segments
            ],
            dtype=float,
        )
        total = float(np.sum(probabilities))
        if not math.isfinite(total) or total <= 0.0:
            continue

        # Pick a segment with probability proportional to its share, then draw
        # the decay distance from the exponential truncated to that segment.
        chosen = int(rng.choice(len(event_segments), p=probabilities / total))
        start, stop = event_segments[chosen]
        one_minus = -math.expm1(-(stop - start) / lam)
        u = float(rng.random())
        distance = start - lam * math.log1p(-u * one_minus)

        keep_indices.append(i)
        sampled_distance.append(distance)
        inside_probability.append(total)

    if not keep_indices:
        empty = df.iloc[0:0].copy()
        set_scalar(empty, columns.inside_probability, np.array([], dtype=float))
        set_scalar(empty, columns.dirt_final_weight, np.array([], dtype=float))
        empty.attrs = dict(df.attrs)
        empty.attrs["forced_decay_volume"] = "liquid_cryostat"
        empty.attrs["forced_decay_input_rows"] = int(len(df))
        return empty

    result = df.iloc[keep_indices].copy().reset_index(drop=True)
    distances = np.asarray(sampled_distance)
    probabilities = np.asarray(inside_probability)

    decay_xyz = origin[keep_indices] + direction[keep_indices] * distances[:, None]

    base = columns.decay_position
    set_vector_component(result, base, POSITION_ORDER.index("x"), decay_xyz[:, 0])
    set_vector_component(result, base, POSITION_ORDER.index("y"), decay_xyz[:, 1])
    set_vector_component(result, base, POSITION_ORDER.index("z"), decay_xyz[:, 2])

    # Decay time: flight time over the sampled distance at the HNL's speed.
    beta = np.divide(
        momentum[keep_indices],
        hnl.energy[keep_indices],
        out=np.zeros(len(keep_indices)),
        where=hnl.energy[keep_indices] > 0.0,
    )
    c_speed = config.geometry.speed_of_light_cm_per_ns
    if config.geometry.time_unit == "s":
        c_speed *= 1.0e9
    delta_t = np.divide(
        distances,
        beta * c_speed,
        out=np.zeros_like(distances),
        where=beta > 0.0,
    )
    set_vector_component(
        result,
        base,
        POSITION_ORDER.index("t"),
        scattering.t[keep_indices] + delta_t,
    )

    set_scalar(result, columns.inside_probability, probabilities)
    event_weight = get_scalar(result, columns.event_weight)
    set_scalar(result, columns.dirt_final_weight, event_weight * probabilities)

    result.attrs = dict(df.attrs)
    result.attrs["forced_decay_volume"] = "liquid_cryostat"
    result.attrs["forced_decay_input_rows"] = int(len(df))
    result.attrs["N4_ctau0"] = float(ctau0)
    return result
