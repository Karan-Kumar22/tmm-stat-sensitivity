"""Access helpers for DarkNews dataframes.

DarkNews returns a pandas DataFrame whose columns are usually a two-level
MultiIndex: four-vectors and positions are four sub-columns per quantity,
e.g. ('P_decay_photon', '0') ... ('P_decay_photon', '3'), while scalars are
('w_event_rate', ''). Some versions/paths produce flat columns instead
('P_decay_photon_0'). These helpers resolve a *logical* name — "component 2
of P_decay_photon" — to whatever physical column the dataframe has, so the
physics code never touches the layout.

Component *meaning* is fixed by the DarkNews convention (``P4_ORDER`` and
``POSITION_ORDER`` below): four-momenta are (E, px, py, pz), positions are
(t, x, y, z).

Ported from the validated production code
(TMM_Analysis/Scripts/tmm_direct_production/tmm_big_program_direct.py).
"""

from __future__ import annotations

import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Sub-column orderings fixed by the DarkNews output convention.
P4_ORDER = ("E", "px", "py", "pz")
POSITION_ORDER = ("t", "x", "y", "z")


class ColumnError(KeyError):
    """A logical column could not be resolved (or resolves ambiguously)."""


def _normalize_subcolumn(value: Any) -> str:
    # Sub-level labels may come back as '0', 0, or 0.0 depending on how the
    # dataframe was constructed or round-tripped; compare them as clean strings.
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _column_tuple(column: Any, nlevels: int) -> tuple[Any, ...]:
    if isinstance(column, tuple):
        result = tuple(column)
    else:
        result = (column,)
    return result + ("",) * (nlevels - len(result))


def find_scalar_column(df: pd.DataFrame, name: str) -> Any:
    """Resolve a scalar quantity (e.g. 'w_event_rate') to its column key."""
    if not isinstance(df.columns, pd.MultiIndex):
        if name in df.columns:
            return name
        raise ColumnError(name)

    exact = tuple([name] + [""] * (df.columns.nlevels - 1))
    if exact in df.columns:
        return exact

    candidates = []
    for column in df.columns:
        values = _column_tuple(column, df.columns.nlevels)
        if str(values[0]) == name and all(str(v) in ("", "nan") for v in values[1:]):
            candidates.append(column)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ColumnError(f"Ambiguous scalar column {name!r}: {candidates}")
    raise ColumnError(name)


def find_vector_column(df: pd.DataFrame, base: str, index: int) -> Any:
    """Resolve component ``index`` of a vector quantity (e.g. P_decay_photon)."""
    if not isinstance(df.columns, pd.MultiIndex):
        for candidate in (
            f"{base}_{index}",
            f"{base}.{index}",
            f"{base}__{index}",
            f"{base}[{index}]",
        ):
            if candidate in df.columns:
                return candidate
        raise ColumnError(f"{base}[{index}]")

    matches = []
    for column in df.columns:
        values = _column_tuple(column, df.columns.nlevels)
        if str(values[0]) != base or len(values) < 2:
            continue
        if _normalize_subcolumn(values[1]) == str(index):
            matches.append(column)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ColumnError(f"Ambiguous vector column {base}[{index}]: {matches}")
    raise ColumnError(f"{base}[{index}]")


def scalar_key(df: pd.DataFrame, name: str) -> Any:
    """Column key under which to *write* a new scalar column."""
    if isinstance(df.columns, pd.MultiIndex):
        return tuple([name] + [""] * (df.columns.nlevels - 1))
    return name


def get_scalar(df: pd.DataFrame, name: str, dtype: Any = float) -> np.ndarray:
    return df[find_scalar_column(df, name)].to_numpy(dtype=dtype)


def set_scalar(df: pd.DataFrame, name: str, values: Any) -> None:
    df[scalar_key(df, name)] = values


def get_vector_component(df: pd.DataFrame, base: str, index: int) -> np.ndarray:
    return df[find_vector_column(df, base, index)].to_numpy(dtype=float)


def set_vector_component(df: pd.DataFrame, base: str, index: int, values: Any) -> None:
    df[find_vector_column(df, base, index)] = values


@dataclass(frozen=True)
class P4:
    energy: np.ndarray
    px: np.ndarray
    py: np.ndarray
    pz: np.ndarray

    @property
    def momentum(self) -> np.ndarray:
        return np.sqrt(self.px**2 + self.py**2 + self.pz**2)


@dataclass(frozen=True)
class Position:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    t: np.ndarray


def extract_p4(df: pd.DataFrame, base: str) -> P4:
    return P4(
        energy=get_vector_component(df, base, P4_ORDER.index("E")),
        px=get_vector_component(df, base, P4_ORDER.index("px")),
        py=get_vector_component(df, base, P4_ORDER.index("py")),
        pz=get_vector_component(df, base, P4_ORDER.index("pz")),
    )


def extract_position(df: pd.DataFrame, base: str) -> Position:
    return Position(
        x=get_vector_component(df, base, POSITION_ORDER.index("x")),
        y=get_vector_component(df, base, POSITION_ORDER.index("y")),
        z=get_vector_component(df, base, POSITION_ORDER.index("z")),
        t=get_vector_component(df, base, POSITION_ORDER.index("t")),
    )


def atomic_pickle(obj: Any, path: str | Path) -> None:
    """Write a pickle so that ``path`` is either absent or complete, never partial.

    Written to a temporary file in the same directory, fsynced, then renamed —
    rename is atomic on POSIX filesystems.
    """
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
