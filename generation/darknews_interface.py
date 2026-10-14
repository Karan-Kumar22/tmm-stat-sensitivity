"""Thin wrapper around the DarkNews event generator.

One public function, ``generate_darknews_sample``: run DarkNews for one
(mass, mu_tr) point and one experiment (cryostat or dirt), return the pandas
dataframe of generated events with provenance recorded in ``df.attrs``.

All physics settings come from the validated Config; the only per-call
choices are which experiment, the model point, the seed, and enforce_prompt.

Compatibility notes (inherited from the validated production code):

- The Nevis DarkNews installation expects the keyword ``decay_product``
  (singular) and ``experiment``; other installations use ``decay_products``
  and/or the legacy ``exp``. We try the combinations in order and use the
  first that GenLauncher accepts.
- ``run()`` takes the output directory as ``overwrite_path`` on the Nevis
  installation; a bare ``run(loglevel=...)`` is the fallback.

Also here: small readers for DarkNews-provided metadata that later stages
need — the HNL proper decay length (ctau0) and the POT normalisation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from config import Config


class GenerationError(RuntimeError):
    """Raised when DarkNews cannot be run or returns something unusable."""


def build_common_kwargs(
    config: Config,
    *,
    mass: float,
    tmm: float,
    seed: int,
    enforce_prompt: bool,
    nopelastic: bool | None = None,
) -> dict[str, Any]:
    """GenLauncher keyword set shared by every attempt (model + run settings).

    ``nopelastic`` overrides the global switch per call — used for the
    per-component channel choice (config.generation.cryostat_nopelastic /
    dirt_nopelastic) and for special-purpose single-channel samples.
    """
    generation = config.generation
    if nopelastic is None:
        nopelastic = generation.nopelastic
    common: dict[str, Any] = {
        # model point
        "m4": float(mass),
        "mu_tr_mu4": float(tmm),
        "HNLtype": generation.hnl_type,
        "nu_flavors": list(generation.nu_flavors),
        # integrator statistics
        "neval": int(generation.neval_per_batch),
        "nint": generation.nint,
        "neval_warmup": generation.neval_warmup,
        "nint_warmup": generation.nint_warmup,
        "seed": int(seed),
        # scattering-process switches (all False = everything on)
        "nocoh": generation.nocoh,
        "nopelastic": bool(nopelastic),
        "noHC": generation.noHC,
        "noHF": generation.noHF,
        # decay handling and output form
        "enforce_prompt": bool(enforce_prompt),
        "sample_geometry": True,
        "pandas": True,
        "sparse": 0,
        "make_summary_plots": False,
    }

    if generation.zero_standard_mixings:
        common.update({"Ue4": 0.0, "Umu4": 0.0, "Utau4": 0.0})
    if generation.disable_dark_portal:
        common.update({"gD": 0.0, "epsilon": 0.0})
    return common


def _launcher_attempts(
    common: dict[str, Any],
    experiment: str,
    decay_products: str,
) -> list[dict[str, Any]]:
    """Keyword-name combinations for different DarkNews versions, in order."""
    attempts: list[dict[str, Any]] = []
    for exp_key in ("experiment", "exp"):
        for decay_key in ("decay_product", "decay_products"):
            kwargs = dict(common)
            kwargs[exp_key] = experiment
            kwargs[decay_key] = decay_products
            attempts.append(kwargs)
    return attempts


def generate_darknews_sample(
    config: Config,
    *,
    experiment: str,
    mass: float,
    tmm: float,
    seed: int,
    enforce_prompt: bool,
    scratch_directory: str | Path,
    nopelastic: bool | None = None,
) -> pd.DataFrame:
    """Run DarkNews once and return the generated dataframe.

    ``experiment`` is config.generation.cryostat_experiment or
    .dirt_experiment. DarkNews's own file output goes to
    ``scratch_directory`` (kept or deleted by the caller according to
    config.storage); the dataframe returned in memory is the product.
    """
    try:
        from DarkNews import GenLauncher
    except ImportError as exc:
        raise GenerationError(
            "DarkNews is not importable in this environment."
        ) from exc

    common = build_common_kwargs(
        config, mass=mass, tmm=tmm, seed=seed, enforce_prompt=enforce_prompt,
        nopelastic=nopelastic,
    )

    scratch_directory = Path(scratch_directory)
    scratch_directory.mkdir(parents=True, exist_ok=True)

    launcher = None
    errors: list[str] = []
    for kwargs in _launcher_attempts(
        common,
        experiment=experiment,
        decay_products=config.generation.decay_products,
    ):
        try:
            launcher = GenLauncher(**kwargs)
            break
        except (TypeError, ValueError, AttributeError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    if launcher is None:
        raise GenerationError(
            "Could not initialize DarkNews GenLauncher with any known "
            "argument-name combination. Errors: " + " | ".join(errors)
        )

    try:
        generated = launcher.run(
            loglevel=config.generation.darknews_loglevel,
            overwrite_path=str(scratch_directory),
        )
    except TypeError:
        # Older run() signature without overwrite_path.
        generated = launcher.run(loglevel=config.generation.darknews_loglevel)

    if isinstance(generated, pd.DataFrame):
        result = generated
    elif hasattr(launcher, "df") and isinstance(launcher.df, pd.DataFrame):
        result = launcher.df
    else:
        raise GenerationError("DarkNews completed but returned no dataframe.")

    result = result.copy()
    result.attrs = dict(getattr(result, "attrs", {}))
    result.attrs["generation_experiment"] = experiment
    result.attrs["generation_mass_GeV"] = float(mass)
    result.attrs["generation_tmm_GeV_inv"] = float(tmm)
    result.attrs["generation_neval"] = int(config.generation.neval_per_batch)
    result.attrs["generation_seed"] = int(seed)
    result.attrs["generation_enforce_prompt"] = bool(enforce_prompt)
    return result


def get_ctau0_cm(df: pd.DataFrame) -> float:
    """HNL proper decay length c*tau0 in cm, from DarkNews output metadata.

    The forced-decay code needs this to compute each event's lab decay
    length lambda = (p/m) * ctau0.
    """
    for key in ("N4_ctau0", "N4_ctau0_cm", "ctau0"):
        if key in df.attrs:
            value = float(df.attrs[key])
            if value > 0.0 and math.isfinite(value):
                return value

    model = df.attrs.get("model")
    if model is not None:
        for key in ("N4_ctau0", "ctau0"):
            if hasattr(model, key):
                value = float(getattr(model, key))
                if value > 0.0 and math.isfinite(value):
                    return value

    raise GenerationError(
        "DarkNews output does not carry a positive N4_ctau0 attribute."
    )


def get_source_pot(df: pd.DataFrame) -> float | None:
    """POT the sample's weights are normalised to, if DarkNews recorded it."""
    for key in ("POTs", "POT", "pot", "target_pot"):
        if key in df.attrs:
            try:
                return float(df.attrs[key])
            except (TypeError, ValueError):
                pass

    experiment = df.attrs.get("experiment")
    if experiment is not None:
        for attribute in ("POTS", "POTs", "POT"):
            if hasattr(experiment, attribute):
                try:
                    return float(getattr(experiment, attribute))
                except (TypeError, ValueError):
                    pass
    return None


if __name__ == "__main__":
    # Smoke test without generating anything:
    #   python darknews_interface.py            -> is DarkNews importable? show kwargs
    #   python darknews_interface.py --run      -> tiny real generation (slow-ish)
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).with_name("config.json"))
    parser.add_argument("--mass", type=float, default=0.48)
    parser.add_argument("--tmm", type=float, default=2.5e-6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run", action="store_true", help="run a tiny real generation")
    args = parser.parse_args()

    from config import load_config

    cfg = load_config(args.config)
    kwargs = build_common_kwargs(
        cfg, mass=args.mass, tmm=args.tmm, seed=args.seed, enforce_prompt=False
    )
    print("GenLauncher kwargs that would be used (cryostat, enforce_prompt=False):")
    print(json.dumps(kwargs, indent=2, default=str))

    try:
        import DarkNews  # noqa: F401
        print(f"\nDarkNews import: OK (version {getattr(DarkNews, '__version__', 'unknown')})")
    except ImportError as exc:
        print(f"\nDarkNews import: FAILED ({exc})")
        raise SystemExit(1)

    if args.run:
        import dataclasses
        import tempfile

        tiny_generation = dataclasses.replace(
            cfg.generation, neval_per_batch=2000
        )
        tiny = dataclasses.replace(cfg, generation=tiny_generation)
        with tempfile.TemporaryDirectory() as scratch:
            df = generate_darknews_sample(
                tiny,
                experiment=tiny.generation.cryostat_experiment,
                mass=args.mass,
                tmm=args.tmm,
                seed=args.seed,
                enforce_prompt=False,
                scratch_directory=scratch,
            )
        print(f"\nGenerated {len(df)} events.")
        print(f"ctau0 = {get_ctau0_cm(df):.4g} cm, source POT = {get_source_pot(df)!r}")
        print("Columns (first 12):")
        for column in list(df.columns)[:12]:
            print(f"  {column}")
