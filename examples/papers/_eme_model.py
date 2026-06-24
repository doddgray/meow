"""Shared helpers for full-device EME models of the paper examples.

Provides the generic pieces each device-specific EME model reuses:

- :func:`octave_wls` - a dense wavelength grid spanning a full octave (factor 2
  in frequency) about a centre wavelength;
- :func:`scattering_db` - the requested S-matrix element magnitudes in dB from a
  cascaded S-matrix and port map;
- :func:`plot_spectrum` / :func:`plot_propagation` / :func:`plot_annotated_layout`
  - the three plot types (dense S-parameter spectrum, propagating ``|E|`` field,
  pretty section-annotated layout).

Each device module supplies its own geometry, extrusion and mode/port selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def octave_wls(center: float, npts: int) -> np.ndarray:
    """A dense wavelength grid spanning a full octave about ``center`` [um].

    A full octave is a factor of two in frequency, i.e. wavelengths from
    ``center/sqrt(2)`` to ``center*sqrt(2)``.
    """
    return np.linspace(center / np.sqrt(2), center * np.sqrt(2), npts)


def use_agg() -> Any:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def te_tm_indices(modes: list[Any]) -> dict[str, int]:
    """Index of the highest-neff TE and TM mode in a mode set."""
    import meow as mw

    te = max(
        (i for i, m in enumerate(modes) if float(mw.te_fraction(m)) >= 0.5),
        key=lambda i: np.real(modes[i].neff), default=0,
    )
    tm = max(
        (i for i, m in enumerate(modes) if float(mw.te_fraction(m)) < 0.5),
        key=lambda i: np.real(modes[i].neff), default=min(1, len(modes) - 1),
    )
    return {"TE": int(te), "TM": int(tm)}


def db(amp: complex) -> float:
    """Magnitude of a scattering amplitude in dB (floored at -60 dB)."""
    return 20.0 * np.log10(max(abs(complex(amp)), 1e-3))


def scattering_db(
    s_matrix: Any, port_map: dict[str, int], out_idx: int, in_idx: int
) -> float:
    """``|S[right@out_idx, left@in_idx]|`` in dB from a cascaded S-matrix."""
    s = np.asarray(s_matrix)
    return db(s[port_map[f"right@{out_idx}"], port_map[f"left@{in_idx}"]])


def lateral_centroid(mode: Any) -> float:
    """Energy-weighted lateral (meow-x) position of a mode."""
    density = np.abs(mode.Ex) ** 2 + np.abs(mode.Ey) ** 2 + np.abs(mode.Ez) ** 2
    return float(np.sum(mode.cs.mesh.Xx * density) / np.sum(density))


def plot_spectrum(
    wls: np.ndarray,
    series: dict[str, np.ndarray],
    path: Path,
    *,
    title: str,
    center_nm: float | None = None,
    ylabel: str = "transmission [dB]",
) -> None:
    """Dense S-parameter magnitude spectrum (dB) vs wavelength."""
    plt = use_agg()
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    for label, vals in series.items():
        ax.plot(wls * 1000, vals, label=label)
    if center_nm is not None:
        ax.axvline(center_nm, color="0.6", ls=":", lw=1)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel(ylabel)
    ax.grid(visible=True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_propagation(
    field: np.ndarray,
    x_trans: np.ndarray,
    length: float,
    path: Path,
    *,
    title: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    """|E| along the device: propagation z (field axis 0) vs transverse x (axis 1)."""
    plt = use_agg()
    norm = field / (field.max() or 1.0)
    zprop = np.linspace(0.0, length, field.shape[0])
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.pcolormesh(zprop, x_trans, norm.T, shading="auto", cmap="magma")
    ax.set_xlabel("propagation z [um]")
    ax.set_ylabel("lateral x [um]")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="|E| (norm.)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_annotated_layout(
    component: Any,
    path: Path,
    *,
    title: str,
    layer_styles: dict[Any, tuple[str, str]] | None = None,
    dividers: list[tuple[float, str]] | None = None,
    params: str = "",
    ylim: tuple[float, float] | None = None,
) -> None:
    """Pretty top-view layout coloured by layer, with section + parameter labels."""
    plt = use_agg()
    from matplotlib.collections import PolyCollection

    layer_styles = layer_styles or {}
    try:
        dbu = component.layout().dbu
    except AttributeError:
        dbu = None

    def to_verts(p: Any) -> np.ndarray:
        if dbu is not None and hasattr(p, "each_point_hull"):
            return np.asarray([(pt.x * dbu, pt.y * dbu) for pt in p.each_point_hull()])
        return np.asarray(p)

    fig, ax = plt.subplots(figsize=(11, 3.4))
    for layer, polys in component.get_polygons(by="tuple").items():
        key = tuple(layer) if not isinstance(layer, tuple) else layer
        name, col = layer_styles.get(key, (None, "#1f77b4"))
        ax.add_collection(
            PolyCollection([to_verts(p) for p in polys], facecolors=col,
                           edgecolors="k", linewidths=0.2, alpha=0.75, label=name)
        )
    ymax = ylim[1] if ylim else 6.0
    for zz, lbl in dividers or []:
        ax.axvline(zz, color="0.4", ls="--", lw=1)
        ax.text(zz, ymax * 0.9, lbl, ha="center", fontsize=8)
    if params:
        ax.text(0.01, 0.02, params, transform=ax.transAxes, fontsize=8, color="0.3")
    ax.autoscale_view()
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("propagation z [um]")
    ax.set_ylabel("lateral x [um]")
    if any(n for n, _ in layer_styles.values()):
        ax.legend(loc="upper right", fontsize=7)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
