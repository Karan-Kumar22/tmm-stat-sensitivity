"""Configuration loading and validation for TMM sample generation.

One JSON file (see ``config.json``) drives the whole generation package. This
module reads it once, checks it, and hands the rest of the code typed objects
instead of raw dictionaries — so a typo in the config fails loudly at startup
with the offending key named, not deep inside a DarkNews run.

Sections:
  generation — DarkNews physics/run settings and the batching parameters
  columns    — names of the DarkNews dataframe columns we read and write
  weights    — which weight columns the 1/N chunk normalisation is applied to
  geometry   — MicroBooNE cryostat constants (DarkNews frame, cm)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when the configuration file is missing or inconsistent."""


def _require(section: Mapping[str, Any], key: str, section_name: str) -> Any:
    if key not in section:
        raise ConfigError(f"Missing key {key!r} in config section {section_name!r}")
    return section[key]


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not value > 0.0:
        raise ConfigError(f"{name} must be positive, got {value}")
    return value


@dataclass(frozen=True)
class GenerationSettings:
    """DarkNews run settings shared by every batch of a model point."""

    neval_per_batch: int
    base_seed: int
    cryostat_experiment: str
    dirt_experiment: str
    hnl_type: str
    decay_products: str
    nu_flavors: tuple[str, ...]
    nint: int
    neval_warmup: int
    nint_warmup: int
    darknews_loglevel: str
    nocoh: bool
    nopelastic: bool
    # per-component channel switches (default to the global flags above).
    # Coherent-only cryostat: proton-elastic upscatters inside the liquid leave
    # a visible recoil proton the single-photon response model cannot describe;
    # dirt upscatters happen underground, so both channels stay valid there.
    cryostat_nopelastic: bool
    dirt_nopelastic: bool
    noHC: bool
    noHF: bool
    zero_standard_mixings: bool
    disable_dark_portal: bool
    dirt_enforce_prompt: bool
    keep_darknews_scratch: bool

    @staticmethod
    def from_mapping(section: Mapping[str, Any]) -> "GenerationSettings":
        name = "generation"
        neval = int(_require(section, "neval_per_batch", name))
        if neval <= 0:
            raise ConfigError(f"neval_per_batch must be positive, got {neval}")
        return GenerationSettings(
            neval_per_batch=neval,
            base_seed=int(_require(section, "base_seed", name)),
            cryostat_experiment=str(_require(section, "cryostat_experiment", name)),
            dirt_experiment=str(_require(section, "dirt_experiment", name)),
            hnl_type=str(_require(section, "hnl_type", name)),
            decay_products=str(_require(section, "decay_products", name)),
            nu_flavors=tuple(section.get("nu_flavors", ("nu_mu",))),
            nint=int(_require(section, "nint", name)),
            neval_warmup=int(_require(section, "neval_warmup", name)),
            nint_warmup=int(_require(section, "nint_warmup", name)),
            darknews_loglevel=str(section.get("darknews_loglevel", "WARNING")),
            nocoh=bool(section.get("nocoh", False)),
            nopelastic=bool(section.get("nopelastic", False)),
            cryostat_nopelastic=bool(
                section.get("cryostat_nopelastic", section.get("nopelastic", False))
            ),
            dirt_nopelastic=bool(
                section.get("dirt_nopelastic", section.get("nopelastic", False))
            ),
            noHC=bool(section.get("noHC", False)),
            noHF=bool(section.get("noHF", False)),
            zero_standard_mixings=bool(section.get("zero_standard_mixings", True)),
            disable_dark_portal=bool(section.get("disable_dark_portal", True)),
            dirt_enforce_prompt=bool(section.get("dirt_enforce_prompt", True)),
            keep_darknews_scratch=bool(section.get("keep_darknews_scratch", False)),
        )


@dataclass(frozen=True)
class ColumnSettings:
    """Names of the DarkNews dataframe columns used by this package."""

    photon_p4: str
    hnl_p4: str
    daughter_neutrino_p4: str
    incoming_neutrino_p4: str
    scattering_position: str
    decay_position: str
    event_weight: str
    inside_probability: str
    dirt_final_weight: str

    @staticmethod
    def from_mapping(section: Mapping[str, Any]) -> "ColumnSettings":
        name = "columns"
        return ColumnSettings(
            photon_p4=str(_require(section, "photon_p4", name)),
            hnl_p4=str(_require(section, "hnl_p4", name)),
            daughter_neutrino_p4=str(_require(section, "daughter_neutrino_p4", name)),
            incoming_neutrino_p4=str(_require(section, "incoming_neutrino_p4", name)),
            scattering_position=str(_require(section, "scattering_position", name)),
            decay_position=str(_require(section, "decay_position", name)),
            event_weight=str(_require(section, "event_weight", name)),
            inside_probability=str(_require(section, "inside_probability", name)),
            dirt_final_weight=str(_require(section, "dirt_final_weight", name)),
        )


@dataclass(frozen=True)
class WeightSettings:
    """Weight columns that receive the 1/N_batches chunk normalisation."""

    cryostat_scale_columns: tuple[str, ...]
    dirt_scale_columns: tuple[str, ...]

    @staticmethod
    def from_mapping(section: Mapping[str, Any]) -> "WeightSettings":
        name = "weights"
        return WeightSettings(
            cryostat_scale_columns=tuple(_require(section, "cryostat_scale_columns", name)),
            dirt_scale_columns=tuple(_require(section, "dirt_scale_columns", name)),
        )


@dataclass(frozen=True)
class StorageSettings:
    """What each generated batch writes to disk.

    The primary products are a HEPevt text file (events, detector coordinates,
    ready for LArSoft) and a compact truth table with one row per event —
    E_gamma, cos_theta, z and the weight columns — written in the same event
    order as the HEPevt file, so the table doubles as the HEPevt weight
    sidecar. The full DarkNews dataframe is kept only if explicitly requested.
    """

    write_hepevt: bool
    write_truth_table: bool
    keep_full_darknews_dataframe: bool
    # gzip each HEPevt file as soon as its batch completes (-> .hepevt.gz);
    # the truth tables stay uncompressed pickles either way
    compress_hepevt: bool = False

    @staticmethod
    def from_mapping(section: Mapping[str, Any]) -> "StorageSettings":
        settings = StorageSettings(
            write_hepevt=bool(section.get("write_hepevt", True)),
            write_truth_table=bool(section.get("write_truth_table", True)),
            keep_full_darknews_dataframe=bool(
                section.get("keep_full_darknews_dataframe", False)
            ),
            compress_hepevt=bool(section.get("compress_hepevt", False)),
        )
        if not (
            settings.write_hepevt
            or settings.write_truth_table
            or settings.keep_full_darknews_dataframe
        ):
            raise ConfigError("storage: at least one output format must be enabled")
        return settings


@dataclass(frozen=True)
class HepevtSettings:
    """PDG codes and status codes used in the HEPevt records.

    The HNL is written with a placeholder PDG code (14) understood by the
    downstream LArSoft configuration; the photon is the only particle that is
    propagated (final_status), the bookkeeping particles carry fake_status.
    """

    photon_pdg: int
    hnl_pdg: int
    daughter_neutrino_pdg: int
    incoming_neutrino_pdg: int
    hnl_status: int
    final_status: int
    fake_status: int
    float_precision: int
    particle_lines: str

    @staticmethod
    def from_mapping(section: Mapping[str, Any]) -> "HepevtSettings":
        precision = int(section.get("float_precision", 16))
        if not 6 <= precision <= 17:
            raise ConfigError(f"float_precision must be in [6, 17], got {precision}")
        particle_lines = str(section.get("particle_lines", "full")).lower()
        if particle_lines not in ("full", "nu_photon_weight", "photon_weight", "photon"):
            raise ConfigError(
                "particle_lines must be 'full' (5 lines: HNL, photon, daughter nu, "
                "incoming nu, weight line), 'nu_photon_weight' (incoming nu at the "
                "scattering vertex + photon + weight line; HNL energy in the truth "
                "table), 'photon_weight' (photon + weight line) or 'photon' "
                f"(photon only), got {particle_lines!r}"
            )
        return HepevtSettings(
            photon_pdg=int(section.get("photon_pdg", 22)),
            hnl_pdg=int(section.get("hnl_pdg", 14)),
            daughter_neutrino_pdg=int(section.get("daughter_neutrino_pdg", 14)),
            incoming_neutrino_pdg=int(section.get("incoming_neutrino_pdg", 14)),
            hnl_status=int(section.get("hnl_status", 1)),
            final_status=int(section.get("final_status", 1)),
            fake_status=int(section.get("fake_status", 0)),
            float_precision=precision,
            particle_lines=particle_lines,
        )


@dataclass(frozen=True)
class GeometrySettings:
    """MicroBooNE cryostat constants in the DarkNews frame (cm).

    The cryostat is a horizontal cylinder along z, centred at the origin,
    closed by two spherical caps, and filled with liquid argon up to
    ``liquid_argon_fill_fraction`` of the tube diameter. The detector offsets
    convert to detector coordinates later (not used during generation).
    """

    cryostat_tube_radius_cm: float
    cryostat_tube_length_cm: float
    cryostat_cap_radius_cm: float
    cryostat_cap_theta_deg: float
    cryostat_cap_gap_cm: float
    liquid_argon_fill_fraction: float
    detector_x_offset_cm: float
    detector_y_offset_cm: float
    detector_z_offset_cm: float
    time_unit: str
    speed_of_light_cm_per_ns: float

    @staticmethod
    def from_mapping(section: Mapping[str, Any]) -> "GeometrySettings":
        name = "geometry"
        fill = float(_require(section, "liquid_argon_fill_fraction", name))
        if not 0.0 < fill <= 1.0:
            raise ConfigError(f"liquid_argon_fill_fraction must be in (0, 1], got {fill}")
        time_unit = str(section.get("time_unit", "ns")).lower()
        if time_unit not in ("ns", "s"):
            raise ConfigError(f"time_unit must be 'ns' or 's', got {time_unit!r}")
        return GeometrySettings(
            cryostat_tube_radius_cm=_positive(
                _require(section, "cryostat_tube_radius_cm", name), "cryostat_tube_radius_cm"
            ),
            cryostat_tube_length_cm=_positive(
                _require(section, "cryostat_tube_length_cm", name), "cryostat_tube_length_cm"
            ),
            cryostat_cap_radius_cm=_positive(
                _require(section, "cryostat_cap_radius_cm", name), "cryostat_cap_radius_cm"
            ),
            cryostat_cap_theta_deg=_positive(
                _require(section, "cryostat_cap_theta_deg", name), "cryostat_cap_theta_deg"
            ),
            cryostat_cap_gap_cm=float(_require(section, "cryostat_cap_gap_cm", name)),
            liquid_argon_fill_fraction=fill,
            detector_x_offset_cm=float(_require(section, "detector_x_offset_cm", name)),
            detector_y_offset_cm=float(_require(section, "detector_y_offset_cm", name)),
            detector_z_offset_cm=float(_require(section, "detector_z_offset_cm", name)),
            time_unit=time_unit,
            speed_of_light_cm_per_ns=_positive(
                _require(section, "speed_of_light_cm_per_ns", name),
                "speed_of_light_cm_per_ns",
            ),
        )


@dataclass(frozen=True)
class Config:
    """The full validated configuration."""

    generation: GenerationSettings
    columns: ColumnSettings
    weights: WeightSettings
    storage: StorageSettings
    hepevt: HepevtSettings
    geometry: GeometrySettings
    source_path: Path = field(compare=False, default=Path("."))

    @staticmethod
    def from_mapping(payload: Mapping[str, Any], source_path: Path) -> "Config":
        for section in ("generation", "columns", "weights", "geometry"):
            if section not in payload:
                raise ConfigError(f"Missing config section {section!r} in {source_path}")
        return Config(
            generation=GenerationSettings.from_mapping(payload["generation"]),
            columns=ColumnSettings.from_mapping(payload["columns"]),
            weights=WeightSettings.from_mapping(payload["weights"]),
            storage=StorageSettings.from_mapping(payload.get("storage", {})),
            hepevt=HepevtSettings.from_mapping(payload.get("hepevt", {})),
            geometry=GeometrySettings.from_mapping(payload["geometry"]),
            source_path=source_path,
        )


def load_config(path: str | Path) -> Config:
    """Read and validate a configuration file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Configuration file {path} is not valid JSON: {exc}") from exc
    return Config.from_mapping(payload, source_path=path)


if __name__ == "__main__":
    # Quick self-check: `python config.py [config.json]` prints the parsed config.
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("config.json")
    configuration = load_config(target)
    print(f"Loaded configuration from {configuration.source_path}")
    for section_name in ("generation", "columns", "weights", "storage", "hepevt", "geometry"):
        print(f"\n[{section_name}]")
        section = getattr(configuration, section_name)
        for key, value in vars(section).items():
            print(f"  {key} = {value!r}")
