"""Not-to-scale 2.5D and 3D perspective renderers for layered waveguide stacks.

Used by the multi-layer (trident) edge-coupler designer to visualise a vertical
stack of tapering waveguide layers. Each layer is drawn in its own colour (muted
red / green / blue by default) so the stack is legible; the vertical axis is
deliberately **not to scale** (thin films and their spacings are exaggerated) to
make the layering visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.collections import PolyCollection


@dataclass
class StackLayer:
    """One waveguide layer as a tapering ribbon for rendering.

    ``z`` and ``width`` are equal-length arrays giving the layer's lateral full
    width at each propagation coordinate (``width == 0`` where the layer is
    absent); ``y_center`` and ``thickness`` are its vertical placement.
    """

    name: str
    z: np.ndarray
    width: np.ndarray
    y_center: float
    thickness: float
    color: str
    z_start: float = 0.0  # propagation coordinate where the layer begins


def _shade(color: str, factor: float) -> tuple[float, float, float]:
    r, g, b = mcolors.to_rgb(color)
    if factor <= 1.0:
        return (r * factor, g * factor, b * factor)
    t = factor - 1.0
    return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)


def plot_stack_2p5d(
    ax: Any,
    layers: list[StackLayer],
    *,
    lift: float = 3.0,
    skew: float = 0.25,
) -> None:
    """Oblique 2.5D top view of a layer stack (each layer a coloured ribbon).

    Layers are screen-shifted upward in proportion to their vertical position so
    the stack reads as stacked ribbons; each has a darker shaded sidewall band
    (the 2.5D extrusion cue). Not to scale.
    """
    ys = [lay.y_center for lay in layers]
    span = (max(ys) - min(ys)) or 1.0
    for lay in sorted(layers, key=lambda L: L.y_center):  # back (low) to front
        m = lay.width > 1e-6
        if not m.any():
            continue
        z, w = np.asarray(lay.z)[m], np.asarray(lay.width)[m]
        off = lift * (lay.y_center - min(ys)) / span  # screen-y lift
        base = np.column_stack(
            [np.r_[z, z[::-1]], np.r_[w / 2, (-w / 2)[::-1]] + off]
        )
        top = base + np.array([skew * 0.4, 0.35])
        ax.add_collection(
            PolyCollection([base], facecolors=[_shade(lay.color, 0.55)],
                           edgecolors="none", zorder=lay.y_center)
        )
        ax.add_collection(
            PolyCollection([top], facecolors=[mcolors.to_rgb(lay.color)],
                           edgecolors=[_shade(lay.color, 0.4)], linewidths=0.4,
                           zorder=lay.y_center + 0.1, label=lay.name)
        )
    ax.autoscale_view()
    ax.set_xlabel("z (propagation) [um]")
    ax.set_ylabel("lateral x [um]  (layers lifted, not to scale)")


def _slab_faces(
    z: np.ndarray, w: np.ndarray, y0: float, t: float,
    pos_scale: float, thick_scale: float,
) -> list[np.ndarray]:
    """Quad faces of a tapering slab in (X=z, Y=lateral, Z=vertical, not to scale).

    Position (level spacing) and thickness are exaggerated *separately* so the
    thin films stay visible without the inter-layer gaps dominating the view.
    """
    zt = y0 * pos_scale + t * thick_scale / 2
    zb = y0 * pos_scale - t * thick_scale / 2
    faces: list[np.ndarray] = []
    for k in range(len(z) - 1):
        z0, z1 = z[k], z[k + 1]
        wl0, wl1 = w[k] / 2, w[k + 1] / 2
        # top, bottom, and the two outer sidewalls of this segment
        faces.append(np.array([[z0, -wl0, zt], [z1, -wl1, zt],
                               [z1, wl1, zt], [z0, wl0, zt]]))
        faces.append(np.array([[z0, -wl0, zb], [z1, -wl1, zb],
                               [z1, wl1, zb], [z0, wl0, zb]]))
        faces.append(np.array([[z0, wl0, zb], [z1, wl1, zb],
                               [z1, wl1, zt], [z0, wl0, zt]]))
        faces.append(np.array([[z0, -wl0, zb], [z1, -wl1, zb],
                               [z1, -wl1, zt], [z0, -wl0, zt]]))
    return faces


def plot_stack_3d(
    ax: Any,
    layers: list[StackLayer],
    *,
    pos_scale: float = 2.0,
    thick_scale: float = 14.0,
    alpha: float = 0.9,
) -> None:
    """3D perspective of the tapering layer stack (vertical axis not to scale).

    Each layer is an extruded coloured slab. ``pos_scale`` sets the rendered
    inter-layer spacing and ``thick_scale`` exaggerates each film's thickness so
    the (very thin) cores read as solid slabs rather than sheets.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    for lay in layers:
        m = np.asarray(lay.width) > 1e-6
        if m.sum() < 2:
            continue
        z, w = np.asarray(lay.z)[m], np.asarray(lay.width)[m]
        faces = _slab_faces(
            z, w, lay.y_center, lay.thickness, pos_scale, thick_scale
        )
        col = Poly3DCollection(
            faces, facecolors=mcolors.to_rgb(lay.color),
            edgecolors=_shade(lay.color, 0.5), linewidths=0.1, alpha=alpha,
        )
        ax.add_collection3d(col)
    allz = np.concatenate([np.asarray(L.z) for L in layers])
    allw = np.concatenate([np.asarray(L.width) for L in layers])
    wmax = float(max(allw.max(), 1.0))
    ax.set_xlim(float(allz.min()), float(allz.max()))
    ax.set_ylim(-wmax, wmax)
    yspan = [L.y_center for L in layers]
    pad = 0.5 * pos_scale + thick_scale * max(L.thickness for L in layers)
    ax.set_zlim(min(yspan) * pos_scale - pad, max(yspan) * pos_scale + pad)
    ax.set_xlabel("z (propagation) [um]")
    ax.set_ylabel("lateral x [um]")
    ax.set_zlabel("vertical (not to scale)")
    try:
        ax.set_box_aspect((6, 2, 1.2))
        ax.view_init(elev=22, azim=-60)
    except (AttributeError, ValueError):
        pass


DEFAULT_LAYER_COLORS: tuple[str, ...] = ("C3", "C2", "C0")  # muted red / green / blue
