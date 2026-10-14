"""Small I/O helpers for the mapping package."""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

_LEGACY_WARNED: set[str] = set()

VALID_SIGNAL_DEFINITIONS = ("inclusive", "coherent_cryostat", "coherent_all")


def apply_signal_definition(frame, component: str, definition: str):
    """Filter one component's truth table according to the signal definition.

    ``coherent_cryostat``: cryostat events must be coherent upscatters
    (``scattering_regime == 0``); dirt keeps both channels (the recoil is
    underground and invisible). ``coherent_all``: both components filtered
    to coherent upscatters. Legacy tables without the regime column pass
    through unchanged, with a one-time warning — they are the old inclusive
    grid and cannot be filtered.
    """
    if definition == "inclusive":
        return frame
    if definition == "coherent_cryostat" and component != "cryostat":
        return frame
    if definition not in ("coherent_cryostat", "coherent_all"):
        raise ValueError(f"unknown signal_definition {definition!r}; "
                         f"expected one of {VALID_SIGNAL_DEFINITIONS}")
    if "scattering_regime" not in frame.columns:
        if "legacy" not in _LEGACY_WARNED:
            _LEGACY_WARNED.add("legacy")
            print("[signal-definition] WARNING: truth table has no "
                  "scattering_regime column (legacy inclusive sample); "
                  "coherent_cryostat filter NOT applied.")
        return frame
    return frame[frame["scattering_regime"] == 0]


def atomic_pickle(obj: Any, path: str | Path) -> None:
    """Write a pickle so ``path`` is either absent or complete, never partial."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".tmp.")
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
