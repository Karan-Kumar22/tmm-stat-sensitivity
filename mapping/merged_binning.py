"""3D merged (adaptive) binning built on the uniform product grid.

Strategy (priority order: keep E, then cos theta, then z):
  1. merge adjacent E rows only if a whole row cannot reach the threshold;
  2. within a row group, merge adjacent cos theta columns (backward first)
     until each block's TOTAL over z reaches the threshold;
  3. inside each (E, cos) block, merge adjacent z slices until every merged
     slice reaches the threshold.

Every resulting cell is a rectangular box in (E, cos theta, z) holding at
least `threshold` isotropic source events. z is merged first in spirit --
a fine (E, cos) cell survives whenever its z-total suffices -- because both
the source and every target are flat in z.

    from merged_binning import build_merged_cells, source_counts
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from binning import TruthBinning


@dataclass(frozen=True)
class MergedCell:
    """One analysis cell: inclusive fine-bin index ranges + physical edges."""

    e_bins: tuple[int, int]      # (first, last) fine E-bin index, inclusive
    c_bins: tuple[int, int]
    z_bins: tuple[int, int]
    e_range: tuple[float, float]  # physical edges [GeV]
    c_range: tuple[float, float]
    z_range: tuple[float, float]  # [cm]
    count: int                    # isotropic source events inside


def source_counts(binning: TruthBinning, source: pd.DataFrame) -> np.ndarray:
    """Fine-grid source occupancy, shape (n_energy, n_costheta, n_z)."""
    idx = binning.cell_index(source.E_gamma.to_numpy(),
                             source.cos_theta.to_numpy(),
                             source.z.to_numpy())
    return np.bincount(idx[idx >= 0], minlength=binning.n_cells).reshape(
        binning.n_energy, binning.n_costheta, binning.n_z)


def _greedy_groups(profile: np.ndarray, threshold: float) -> list[list[int]]:
    """Split [0..n) into adjacent groups each summing to >= threshold."""
    groups: list[list[int]] = []
    current: list[int] = []
    for k in range(len(profile)):
        current.append(k)
        if profile[current].sum() >= threshold:
            groups.append(current)
            current = []
    if current:                      # leftover joins the last complete group
        if groups:
            groups[-1] = groups[-1] + current
        else:
            groups = [current]
    return groups


def build_merged_cells(binning: TruthBinning, counts: np.ndarray,
                       threshold: float = 100.0) -> list[MergedCell]:
    # step 1: E rows, merged from the sparse top downward only when needed
    row_groups: list[list[int]] = []
    current: list[int] = []
    for i in range(binning.n_energy - 1, -1, -1):
        current.append(i)
        if counts[current].sum() >= threshold:
            row_groups.append(sorted(current))
            current = []
    if current:
        row_groups[-1] = sorted(current + row_groups[-1])
    row_groups = row_groups[::-1]

    cells: list[MergedCell] = []
    for rows in row_groups:
        pooled = counts[rows].sum(axis=0)                      # (ncos, nz)
        # step 2: cos columns, judged on their z-total
        col_groups = _greedy_groups(pooled.sum(axis=1), threshold)
        for cols in col_groups:
            zprofile = pooled[cols].sum(axis=0)                # (nz,)
            # step 3: z slices inside the block
            for zg in _greedy_groups(zprofile, threshold):
                cells.append(MergedCell(
                    e_bins=(rows[0], rows[-1]),
                    c_bins=(cols[0], cols[-1]),
                    z_bins=(zg[0], zg[-1]),
                    e_range=(float(binning.energy_edges[rows[0]]),
                             float(binning.energy_edges[rows[-1] + 1])),
                    c_range=(float(binning.costheta_edges[cols[0]]),
                             float(binning.costheta_edges[cols[-1] + 1])),
                    z_range=(float(binning.z_edges[zg[0]]),
                             float(binning.z_edges[zg[-1] + 1])),
                    count=int(zprofile[zg].sum()),
                ))
    return cells


class MergedBinning:
    """A frozen merged binning: the fine base grid + the cell grouping.

    Drop-in for TruthBinning wherever only ``n_cells`` and ``cell_index``
    are used (TruthMap, reweight): ``cell_index`` returns the MERGED cell id
    (0..n_cells-1, -1 outside the base support). The base grid's edges are
    exposed for axis histograms and map-file locking.
    """

    def __init__(self, base: TruthBinning, cells: list[MergedCell],
                 source_file: str = ""):
        self.base = base
        self.cells = cells
        self.source_file = source_file
        # fine-cell -> merged-cell lookup
        lookup = np.full(base.n_cells, -1, dtype=np.int64)
        for merged_id, cell in enumerate(cells):
            for ie in range(cell.e_bins[0], cell.e_bins[1] + 1):
                for ic in range(cell.c_bins[0], cell.c_bins[1] + 1):
                    for iz in range(cell.z_bins[0], cell.z_bins[1] + 1):
                        lookup[(ie * base.n_costheta + ic) * base.n_z + iz] = merged_id
        if (lookup < 0).any():
            raise ValueError("merged cells do not tile the base grid")
        self._lookup = lookup

    # -- TruthBinning-compatible surface ------------------------------------

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def cell_index(self, energy, costheta, z) -> np.ndarray:
        fine = self.base.cell_index(energy, costheta, z)
        merged = np.where(fine >= 0, self._lookup[np.clip(fine, 0, None)], -1)
        return merged.astype(np.int64)

    @property
    def energy_edges(self) -> np.ndarray:
        return self.base.energy_edges

    @property
    def costheta_edges(self) -> np.ndarray:
        return self.base.costheta_edges

    @property
    def z_edges(self) -> np.ndarray:
        return self.base.z_edges


def load_merged_binning(path) -> MergedBinning:
    """Load a frozen merged binning file (see freeze in binning_merged_v1)."""
    import json
    from pathlib import Path

    path = Path(path)
    payload = json.loads(path.read_text())
    base = TruthBinning(
        energy_edges=np.array([float(v) for v in payload["base"]["energy_edges_GeV"]]),
        costheta_edges=np.array(payload["base"]["costheta_edges"], dtype=float),
        z_edges=np.array(payload["base"]["z_edges_cm"], dtype=float),
    )
    cells = []
    for entry in payload["cells"]:
        e = tuple(entry["e_bins"]); c = tuple(entry["c_bins"]); z = tuple(entry["z_bins"])
        cells.append(MergedCell(
            e_bins=e, c_bins=c, z_bins=z,
            e_range=(float(base.energy_edges[e[0]]), float(base.energy_edges[e[1] + 1])),
            c_range=(float(base.costheta_edges[c[0]]), float(base.costheta_edges[c[1] + 1])),
            z_range=(float(base.z_edges[z[0]]), float(base.z_edges[z[1] + 1])),
            count=int(entry["source_events"]),
        ))
    if len(cells) != int(payload["n_cells"]):
        raise ValueError(f"{path}: cell count mismatch")
    return MergedBinning(base, cells, source_file=path.name)


def load_analysis_binning(config):
    """The official analysis binning for a loaded MappingConfig.

    Returns the frozen MergedBinning when the config's binning section names
    a ``merged_cells_file`` (verifying its base grid matches the config's),
    otherwise the plain product-grid TruthBinning.
    """
    base = TruthBinning.from_settings(config.binning)
    name = config.binning.merged_cells_file
    if not name:
        return base
    merged = load_merged_binning(config.source_path.parent / name)
    for label, a, b in (("energy", base.energy_edges, merged.base.energy_edges),
                        ("costheta", base.costheta_edges, merged.base.costheta_edges),
                        ("z", base.z_edges, merged.base.z_edges)):
        finite = np.isfinite(a) & np.isfinite(b)
        if len(a) != len(b) or not np.allclose(a[finite], b[finite], atol=1e-9):
            raise ValueError(
                f"merged_cells_file base {label} edges do not match the "
                "config's binning section — the frozen file and the config "
                "have diverged")
    return merged


if __name__ == "__main__":
    from pathlib import Path

    from config import load_config

    config = load_config(Path(__file__).with_name("config.json"))
    binning = TruthBinning.from_settings(config.binning)
    source = pd.read_pickle(Path(__file__).with_name("iso_truth.pkl"))
    cells = build_merged_cells(binning, source_counts(binning, source))
    counts = np.array([cell.count for cell in cells])
    print(f"{len(cells)} merged cells; events/cell: min {counts.min()}, "
          f"median {np.median(counts):.0f}, max {counts.max()}")
