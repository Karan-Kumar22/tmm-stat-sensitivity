"""Command line for batched TMM sample generation.

    python generate.py run     --config config.json --outdir <point_dir> \\
                               --mass 0.48 --tmm 2.5e-6 --batches 2
    python generate.py status  --outdir <point_dir>
    python generate.py combine --outdir <point_dir>

``run`` appends new batches to the model point in <point_dir>: per batch and
component it runs DarkNews, applies the decay-volume physics
(cryostat: keep liquid decays; dirt: forced decay), and writes the outputs
declared in config.storage — a HEPevt file and a truth table per batch —
then records the batch in manifest.json. Interrupting is safe: completed
batches are never redone, and a partially generated batch is regenerated
identically (deterministic seeds).

``status`` and ``combine`` need no config file: the manifest carries a full
config snapshot from which the configuration is reconstructed, so they always
use exactly the settings the batches were generated with.

Truth table columns (rows in HEPevt event order):
    E_gamma    true photon energy [GeV]
    cos_theta  photon direction cosine w.r.t. the beam axis (pz/|p|)
    z          decay vertex z, detector frame [cm]
    w_event_rate            DarkNews physical weight
    P_inside, closure_weight   (dirt only)

HEPevt event block (byte-compatible with the validated production format,
positions in detector frame, momenta GeV, cm, ns):
    <event_id> 5
    1: HNL          at the scattering vertex, daughters = lines 2,3
    2: photon       at the decay vertex, mother = 1        <- propagated
    3: daughter nu  at the decay vertex, mother = 1
    4: incoming nu  at the scattering vertex
    5: fake line: px=w_event_rate, py=P_inside, pz=final weight,
                  E=region flag (0 cryostat, 1 dirt)
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import batch_store
from batch_store import Manifest, ManifestError
from config import Config, ConfigError, load_config
from darknews_interface import GenerationError, generate_darknews_sample
from dataframe_tools import (
    atomic_pickle,
    extract_p4,
    extract_position,
    get_scalar,
)
from decay_volume import (
    force_dirt_decays_inside_cryostat,
    select_cryostat_decays_in_liquid,
)
from geometry import to_detector_frame


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------


def build_truth_table(df: pd.DataFrame, config: Config, component: str) -> pd.DataFrame:
    """The compact per-event table used for map making (and as weight sidecar).

    Beyond the map axes (E_gamma, cos_theta, z) and weights, it carries the
    neutrino-side truth: incoming-neutrino energy and direction, and the HNL
    energy (the daughter neutrino is recoverable as E_hnl - E_gamma).
    """
    columns = config.columns
    photon = extract_p4(df, columns.photon_p4)
    decay = extract_position(df, columns.decay_position)
    incoming = extract_p4(df, columns.incoming_neutrino_p4)
    hnl = extract_p4(df, columns.hnl_p4)

    momentum = photon.momentum
    cos_theta = np.divide(
        photon.pz, momentum, out=np.zeros_like(momentum), where=momentum > 0.0
    )
    nu_momentum = incoming.momentum
    cos_theta_nu = np.divide(
        incoming.pz, nu_momentum, out=np.zeros_like(nu_momentum), where=nu_momentum > 0.0
    )
    _, _, z_detector = to_detector_frame(decay.x, decay.y, decay.z, config.geometry)

    table = pd.DataFrame(
        {
            "E_gamma": photon.energy,
            "cos_theta": cos_theta,
            "z": z_detector,
            "w_event_rate": get_scalar(df, columns.event_weight),
            "E_nu": incoming.energy,
            "cos_theta_nu": cos_theta_nu,
            "E_hnl": hnl.energy,
        }
    )
    # channel tag and recoil kinetic energy: the signal definition (e.g.
    # coherent-only in the cryostat, or a visible-proton veto) is applied as
    # a DOWNSTREAM filter on these columns, never at generation time.
    try:
        regime = np.asarray(get_scalar(df, "scattering_regime", dtype=object)).ravel()
        table["scattering_regime"] = np.where(
            regime == "coherent", 0, np.where(regime == "p-el", 1, -1)
        ).astype(np.int8)
        recoil = extract_p4(df, "P_recoil")
        recoil_mass = np.sqrt(
            np.clip(recoil.energy**2 - recoil.momentum**2, 0.0, None)
        )
        table["recoil_ke"] = recoil.energy - recoil_mass  # GeV
        # projectile flavour: +14 for nu_mu, -14 for nu_mu_bar. DarkNews has
        # no projectile-pdg column; the process string starts "nu(mu) ..."
        # for neutrinos and "nu(mu)~ ..." (tilde) for antineutrinos.
        process = np.asarray(
            get_scalar(df, "underlying_process", dtype=object)
        ).ravel().astype(str)
        table["nu_pdg"] = np.where(
            np.char.startswith(process, "nu(mu)~"), -14, 14
        ).astype(np.int16)
    except (KeyError, ValueError):
        # older DarkNews output without these columns: legacy inclusive table
        pass
    if component == "dirt":
        table["P_inside"] = get_scalar(df, columns.inside_probability)
        table["closure_weight"] = get_scalar(df, columns.dirt_final_weight)

    table.attrs = {
        "component": component,
        "mass_GeV": df.attrs.get("generation_mass_GeV"),
        "tmm_GeV_inv": df.attrs.get("generation_tmm_GeV_inv"),
        "generation_seed": df.attrs.get("generation_seed"),
    }
    return table


def format_particle(
    status: int, pdg: int,
    mother1: int, mother2: int, daughter1: int, daughter2: int,
    px: float, py: float, pz: float, energy: float, mass: float,
    x: float, y: float, z: float, t: float,
    precision: int = 16,
) -> str:
    # %.<p>g drops trailing zeros ("0" instead of 0.0000000000000000e+00),
    # which is where most of the file size goes.
    number = f"%.{precision}g"
    floats = " ".join(
        number % value for value in (px, py, pz, energy, mass, x, y, z, t)
    )
    return (
        f"{status:d} {pdg:d} {mother1:d} {mother2:d} "
        f"{daughter1:d} {daughter2:d} {floats}"
    )


def write_hepevt(
    df: pd.DataFrame,
    config: Config,
    mass: float,
    component: str,
    path: Path,
) -> None:
    """Write the 5-line-per-event HEPevt file (detector coordinates)."""
    columns = config.columns
    hepevt = config.hepevt
    geometry_settings = config.geometry

    hnl = extract_p4(df, columns.hnl_p4)
    photon = extract_p4(df, columns.photon_p4)
    daughter = extract_p4(df, columns.daughter_neutrino_p4)
    incoming = extract_p4(df, columns.incoming_neutrino_p4)
    scattering = extract_position(df, columns.scattering_position)
    decay = extract_position(df, columns.decay_position)

    sx, sy, sz = to_detector_frame(scattering.x, scattering.y, scattering.z, geometry_settings)
    dx, dy, dz = to_detector_frame(decay.x, decay.y, decay.z, geometry_settings)

    # HNL mass from its four-vector where it is on-shell; model mass otherwise.
    m2 = hnl.energy**2 - hnl.momentum**2
    hnl_mass = np.where(m2 > 0.0, np.sqrt(np.clip(m2, 0.0, None)), float(mass))

    w_event = get_scalar(df, columns.event_weight)
    if component == "dirt":
        p_inside = get_scalar(df, columns.inside_probability)
        final_weight = get_scalar(df, columns.dirt_final_weight)
        region_flag = 1.0
    else:
        p_inside = np.ones(len(df))
        final_weight = w_event
        region_flag = 0.0

    mode = hepevt.particle_lines
    precision = hepevt.float_precision
    # In reduced modes the photon is the first (or only) line, so it carries
    # no mother reference; in full mode it points at the HNL line 1.
    photon_mother = 1 if mode == "full" else 0
    lines_per_event = {
        "full": 5, "nu_photon_weight": 3, "photon_weight": 2, "photon": 1
    }[mode]

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".tmp.")
    try:
        with os.fdopen(fd, "w") as handle:
            for i in range(len(df)):
                handle.write(f"{i} {lines_per_event}\n")
                if mode == "full":
                    # 1: HNL at the scattering vertex; decays into lines 2 and 3.
                    handle.write(format_particle(
                        hepevt.hnl_status, hepevt.hnl_pdg, 0, 0, 2, 3,
                        hnl.px[i], hnl.py[i], hnl.pz[i], hnl.energy[i], hnl_mass[i],
                        sx[i], sy[i], sz[i], scattering.t[i],
                        precision=precision,
                    ) + "\n")
                elif mode == "nu_photon_weight":
                    # 1: incoming neutrino at the scattering vertex — carries
                    # the production info (flux energy + vertex).
                    handle.write(format_particle(
                        hepevt.final_status, hepevt.incoming_neutrino_pdg, 0, 0, 0, 0,
                        incoming.px[i], incoming.py[i], incoming.pz[i],
                        incoming.energy[i], 0.0,
                        sx[i], sy[i], sz[i], scattering.t[i],
                        precision=precision,
                    ) + "\n")
                # photon at the decay vertex (the propagated particle).
                handle.write(format_particle(
                    hepevt.final_status, hepevt.photon_pdg, photon_mother, 0, 0, 0,
                    photon.px[i], photon.py[i], photon.pz[i], photon.energy[i], 0.0,
                    dx[i], dy[i], dz[i], decay.t[i],
                    precision=precision,
                ) + "\n")
                if mode == "full":
                    # 3: daughter neutrino at the decay vertex.
                    handle.write(format_particle(
                        hepevt.final_status, hepevt.daughter_neutrino_pdg, 1, 0, 0, 0,
                        daughter.px[i], daughter.py[i], daughter.pz[i], daughter.energy[i], 0.0,
                        dx[i], dy[i], dz[i], decay.t[i],
                        precision=precision,
                    ) + "\n")
                    # 4: incoming neutrino at the scattering vertex.
                    handle.write(format_particle(
                        hepevt.final_status, hepevt.incoming_neutrino_pdg, 0, 0, 0, 0,
                        incoming.px[i], incoming.py[i], incoming.pz[i], incoming.energy[i], 0.0,
                        sx[i], sy[i], sz[i], scattering.t[i],
                        precision=precision,
                    ) + "\n")
                if mode in ("full", "nu_photon_weight", "photon_weight"):
                    # last line: fake particle carrying weights and source flag.
                    handle.write(format_particle(
                        hepevt.fake_status, 0, 0, 0, 0, 0,
                        w_event[i], p_inside[i], final_weight[i], region_flag, 0.0,
                        0.0, 0.0, 0.0, 0.0,
                        precision=precision,
                    ) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if config.storage.compress_hepevt:
            # compress the finished tmp, then move into place atomically
            subprocess.run(["gzip", "-f", tmp_name], check=True)
            os.replace(tmp_name + ".gz", path.with_name(path.name + ".gz"))
        else:
            os.replace(tmp_name, path)
    except BaseException:
        for leftover in (tmp_name, tmp_name + ".gz"):
            try:
                os.unlink(leftover)
            except OSError:
                pass
        raise


# ---------------------------------------------------------------------------
# One batch of one component
# ---------------------------------------------------------------------------


def run_component_batch(
    component: str,
    index: int,
    manifest: Manifest,
    config: Config,
    point_directory: Path,
) -> dict:
    """Generate, process, and store one batch; returns the manifest entry."""
    mass, tmm = manifest.mass, manifest.tmm
    base_seed = manifest.base_seed

    if component == "cryostat":
        experiment = config.generation.cryostat_experiment
        stream = batch_store.STREAM_CRYOSTAT_DARKNEWS
        enforce_prompt = False
        nopelastic = config.generation.cryostat_nopelastic
    else:
        experiment = config.generation.dirt_experiment
        stream = batch_store.STREAM_DIRT_DARKNEWS
        enforce_prompt = config.generation.dirt_enforce_prompt
        nopelastic = config.generation.dirt_nopelastic

    seed = batch_store.darknews_seed(base_seed, index, stream)
    scratch = batch_store.darknews_scratch_path(point_directory, component, index)

    print(f"[{component} batch {index}] DarkNews {experiment}, seed {seed} ...",
          flush=True)
    raw = generate_darknews_sample(
        config,
        experiment=experiment,
        mass=mass,
        tmm=tmm,
        seed=seed,
        enforce_prompt=enforce_prompt,
        scratch_directory=scratch,
        nopelastic=nopelastic,
    )
    raw_rows = len(raw)

    if component == "cryostat":
        kept = select_cryostat_decays_in_liquid(raw, config)
    else:
        rng = batch_store.forced_decay_rng(base_seed, index)
        kept = force_dirt_decays_inside_cryostat(raw, config, rng, mass=mass)
    kept_rows = len(kept)
    del raw

    entry: dict = {
        "index": index,
        "component": component,
        "darknews_seed": seed,
        "raw_rows": raw_rows,
        "kept_rows": kept_rows,
        "sum_w_event_rate": float(
            get_scalar(kept, config.columns.event_weight).sum()
        ) if kept_rows else 0.0,
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if component == "dirt" and kept_rows:
        entry["sum_closure_weight"] = float(
            get_scalar(kept, config.columns.dirt_final_weight).sum()
        )

    if config.storage.write_truth_table:
        table = build_truth_table(kept, config, component)
        truth_path = batch_store.truth_table_path(point_directory, component, index)
        atomic_pickle(table, truth_path)
        entry["truth_table"] = truth_path.name
    if config.storage.write_hepevt:
        hep_path = batch_store.hepevt_path(point_directory, component, index)
        write_hepevt(kept, config, mass, component, hep_path)
        entry["hepevt"] = hep_path.name
    if config.storage.keep_full_darknews_dataframe:
        df_path = batch_store.darknews_dataframe_path(point_directory, component, index)
        atomic_pickle(kept, df_path)
        entry["darknews_dataframe"] = df_path.name

    if not config.generation.keep_darknews_scratch:
        shutil.rmtree(scratch, ignore_errors=True)

    manifest.record_batch(component, entry)
    print(
        f"[{component} batch {index}] {raw_rows} generated -> {kept_rows} kept, "
        f"sum(w) = {entry['sum_w_event_rate']:.6g}",
        flush=True,
    )
    return entry


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def command_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    point_directory = Path(args.outdir)
    manifest = Manifest.create_or_load(
        point_directory, config, mass=args.mass, tmm=args.tmm
    )

    for _ in range(args.batches):
        for component in args.components:
            index = manifest.next_index(component)
            run_component_batch(component, index, manifest, config, point_directory)

    print(f"\nDone. Manifest: {manifest.path}")
    for component in batch_store.COMPONENTS:
        print(f"  {component}: {len(manifest.batches(component))} batches on disk")
    return 0


def config_from_manifest(manifest: Manifest) -> Config:
    """Reconstruct the configuration the batches were generated with."""
    return Config.from_mapping(manifest.data["config_snapshot"], manifest.path)


def command_status(args: argparse.Namespace) -> int:
    manifest = Manifest.load(args.outdir)
    print(f"Model point: mass = {manifest.mass} GeV, mu_tr = {manifest.tmm} GeV^-1")
    print(f"neval per batch = {manifest.neval_per_batch}, "
          f"base seed = {manifest.base_seed}")
    for component in batch_store.COMPONENTS:
        entries = manifest.batches(component)
        print(f"\n{component}: {len(entries)} batches")
        if not entries:
            continue
        sums = [entry["sum_w_event_rate"] for entry in entries]
        print(f"  {'idx':>4} {'raw':>9} {'kept':>8} {'sum(w_event_rate)':>18}")
        for entry in entries:
            print(
                f"  {entry['index']:>4} {entry['raw_rows']:>9} "
                f"{entry['kept_rows']:>8} {entry['sum_w_event_rate']:>18.6g}"
            )
        mean = float(np.mean(sums))
        spread = float(np.std(sums))
        print(f"  per-batch sum(w): mean {mean:.6g}, rms {spread:.3g}"
              " (batches should be statistically compatible)")
    return 0


def command_combine(args: argparse.Namespace) -> int:
    manifest = Manifest.load(args.outdir)
    config = config_from_manifest(manifest)
    out_directory = batch_store.combined_directory(args.outdir)

    for component in batch_store.COMPONENTS:
        if not manifest.batches(component):
            print(f"{component}: no batches, skipped")
            continue
        combined = batch_store.combine_truth_tables(args.outdir, component, config)
        destination = out_directory / f"{component}_truth.pkl"
        atomic_pickle(combined, destination)
        n = combined.attrs["n_batches"]
        total = float(combined["w_event_rate"].sum())
        line = (f"{component}: {len(combined)} events from {n} batches -> "
                f"{destination}  sum(w_event_rate) = {total:.6g}")
        if "closure_weight" in combined.columns:
            line += f", sum(closure_weight) = {float(combined['closure_weight'].sum()):.6g}"
        print(line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="append batches to a model point")
    run.add_argument("--config", required=True)
    run.add_argument("--outdir", required=True, help="model-point directory")
    run.add_argument("--mass", type=float, required=True, help="m4 [GeV]")
    run.add_argument("--tmm", type=float, required=True, help="mu_tr [GeV^-1]")
    run.add_argument("--batches", type=int, default=1,
                     help="number of new batches to add (default 1)")
    run.add_argument("--components", nargs="+", default=list(batch_store.COMPONENTS),
                     choices=batch_store.COMPONENTS,
                     help="which components to generate (default: both)")
    run.set_defaults(func=command_run)

    status = sub.add_parser("status", help="show what is on disk for a point")
    status.add_argument("--outdir", required=True)
    status.set_defaults(func=command_status)

    combine = sub.add_parser(
        "combine", help="build 1/N-normalised combined truth tables"
    )
    combine.add_argument("--outdir", required=True)
    combine.set_defaults(func=command_combine)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except (ConfigError, ManifestError, GenerationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
