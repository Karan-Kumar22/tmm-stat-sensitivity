"""Truth-cell binning: turn BinningSettings into edges and cell indices.

The truth map lives on a 3D grid of cells b = (E_gamma, cos_theta, z).
This module owns the two operations everything downstream needs:

- build the bin edges for the three axes from the validated settings;
- map arrays of (E, cos_theta, z) to flat cell indices, with -1 marking
  events outside the map's support (tracked, never silently dropped —
  plan section 7).

Axes
----
Energy: uniform bins over [start, stop], then the configured wide tail
bins, optionally closed by one overflow bin to +infinity (so "overflow"
is a real cell, not a loss).

cos(theta), three schemes:
  'log'       uniform in log10(1 - cos_theta) between the forward edge and
              2, plus one most-forward bin [1 - forward_edge, 1]. Forward
              resolution grows automatically where the signal peaks
              (plan section 6 recommendation).
  'uniform'   uniform over [-1, 1].
  'piecewise' explicit edges (default: the notebook's hand-tuned scheme,
              duplicate edges removed).

z: uniform over [z_min, z_max] in detector-frame cm. The configured range
covers the full liquid cryostat (about -93 to +1130), wider than the TPC,
because decays happen in all of the liquid.

Cell order: flat index = (i_energy * n_costheta + i_costheta) * n_z + i_z,
fixed here once; H_T arrays and the response matrix all use it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import BinningSettings


def _energy_edges(settings: BinningSettings) -> np.ndarray:
    n_uniform = int(round(
        (settings.energy_uniform_stop_GeV - settings.energy_uniform_start_GeV)
        / settings.energy_uniform_width_GeV
    ))
    edges = list(
        np.linspace(
            settings.energy_uniform_start_GeV,
            settings.energy_uniform_stop_GeV,
            n_uniform + 1,
        )
    )
    edges.extend(settings.energy_tail_edges_GeV)
    if settings.include_energy_overflow:
        edges.append(np.inf)
    return np.asarray(edges, dtype=float)


def _costheta_edges(settings: BinningSettings) -> np.ndarray:
    scheme = settings.costheta_scheme
    if scheme == "uniform":
        return np.linspace(-1.0, 1.0, settings.number_of_costheta_bins + 1)
    if scheme == "piecewise":
        return np.asarray(settings.costheta_piecewise_edges, dtype=float)
    # 'log': uniform in log10(1 - cos) from 2 (cos = -1) down to the forward
    # edge, then one extra bin up to cos = +1.
    one_minus = np.logspace(
        np.log10(2.0),
        np.log10(settings.costheta_log_forward_edge),
        settings.number_of_costheta_bins + 1,
    )
    edges = 1.0 - one_minus            # ascending: -1 ... (1 - forward_edge)
    edges[0] = -1.0                    # exact, not 1 - 2 with roundoff
    return np.append(edges, 1.0)


def _z_edges(settings: BinningSettings) -> np.ndarray:
    return np.linspace(settings.z_min_cm, settings.z_max_cm, settings.number_of_z_bins + 1)


def _axis_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index per value; -1 for values outside [edges[0], edges[-1]].

    A value exactly at the upper boundary belongs to the last bin (so
    cos_theta = 1.0 and z = z_max are inside, not overflow).
    """
    values = np.asarray(values, dtype=float)
    index = np.searchsorted(edges, values, side="right") - 1
    index[values == edges[-1]] = len(edges) - 2
    outside = (values < edges[0]) | (values > edges[-1]) | ~np.isfinite(values)
    index[outside] = -1
    return index


@dataclass(frozen=True)
class TruthBinning:
    """The frozen 3D cell grid."""

    energy_edges: np.ndarray
    costheta_edges: np.ndarray
    z_edges: np.ndarray

    @staticmethod
    def from_settings(settings: BinningSettings) -> "TruthBinning":
        if settings.explicit_energy_edges:
            # edges frozen in an explicit_edges_file (occupancy-driven binning)
            binning = TruthBinning(
                energy_edges=np.asarray(settings.explicit_energy_edges, dtype=float),
                costheta_edges=np.asarray(settings.explicit_costheta_edges, dtype=float),
                z_edges=np.asarray(settings.explicit_z_edges, dtype=float),
            )
        else:
            binning = TruthBinning(
                energy_edges=_energy_edges(settings),
                costheta_edges=_costheta_edges(settings),
                z_edges=_z_edges(settings),
            )
        for name, edges in (
            ("energy", binning.energy_edges),
            ("costheta", binning.costheta_edges),
            ("z", binning.z_edges),
        ):
            if not np.all(np.diff(edges) > 0):
                raise ValueError(f"{name} edges are not strictly increasing: {edges}")
        return binning

    # -- sizes --------------------------------------------------------------

    @property
    def n_energy(self) -> int:
        return len(self.energy_edges) - 1

    @property
    def n_costheta(self) -> int:
        return len(self.costheta_edges) - 1

    @property
    def n_z(self) -> int:
        return len(self.z_edges) - 1

    @property
    def n_cells(self) -> int:
        return self.n_energy * self.n_costheta * self.n_z

    # -- indexing -----------------------------------------------------------

    def cell_index(
        self,
        energy: np.ndarray,
        costheta: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        """Flat cell index per event; -1 = outside the map's support."""
        i_e = _axis_index(energy, self.energy_edges)
        i_c = _axis_index(costheta, self.costheta_edges)
        i_z = _axis_index(z, self.z_edges)
        flat = (i_e * self.n_costheta + i_c) * self.n_z + i_z
        flat[(i_e < 0) | (i_c < 0) | (i_z < 0)] = -1
        return flat

    def unravel(self, flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flat cell index -> (i_energy, i_costheta, i_z)."""
        flat = np.asarray(flat)
        i_z = flat % self.n_z
        rest = flat // self.n_z
        i_c = rest % self.n_costheta
        i_e = rest // self.n_costheta
        return i_e, i_c, i_z

    # -- description --------------------------------------------------------

    def describe(self) -> str:
        e = self.energy_edges
        finite_top = e[-2] if np.isinf(e[-1]) else e[-1]
        lines = [
            f"energy  : {self.n_energy} bins, {e[0]:g} .. {finite_top:g} GeV"
            + (" + overflow" if np.isinf(e[-1]) else ""),
            f"costheta: {self.n_costheta} bins, edges "
            f"{self.costheta_edges[0]:+.3f} .. {self.costheta_edges[-1]:+.3f} "
            f"(most forward bin width {1.0 - self.costheta_edges[-2]:.2e})",
            f"z       : {self.n_z} bins, {self.z_edges[0]:g} .. {self.z_edges[-1]:g} cm",
            f"cells   : {self.n_cells}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Edges as plain lists (for saving next to any histogram built on them)."""
        return {
            "energy_edges": list(self.energy_edges),
            "costheta_edges": list(self.costheta_edges),
            "z_edges": list(self.z_edges),
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from config import load_config

    target = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("config.json")
    configuration = load_config(target)
    binning = TruthBinning.from_settings(configuration.binning)
    print(f"Binning from {configuration.source_path} "
          f"(costheta scheme: {configuration.binning.costheta_scheme})\n")
    print(binning.describe())
    print("\ncostheta edges:")
    print(np.array2string(binning.costheta_edges, precision=5, max_line_width=100))
