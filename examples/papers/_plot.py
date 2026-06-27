"""Small plotting helpers shared by the paper examples."""

from __future__ import annotations

from typing import Any

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.collections import PolyCollection


def _component_polygons(component: Any) -> list[np.ndarray]:
    """Extract a component's polygons as arrays of ``(x, y)`` vertices."""
    polys = []
    for layer_polys in component.get_polygons().values():
        for p in layer_polys:
            try:
                dbu = component.layout().dbu
                pts = np.asarray(
                    [(pt.x * dbu, pt.y * dbu) for pt in p.each_point_hull()]
                )
            except AttributeError:
                pts = np.asarray(p)
            polys.append(pts)
    return polys


def _shade(color: Any, factor: float) -> tuple[float, float, float]:
    """Darken (``factor`` < 1) or lighten (> 1) an RGB color toward black/white."""
    r, g, b = mcolors.to_rgb(color)
    if factor <= 1.0:
        return (r * factor, g * factor, b * factor)
    t = factor - 1.0
    return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)


def plot_component(
    component: Any,
    ax: Any,
    color: str = "C3",
    *,
    depth: float | None = None,
    skew: float = 0.35,
) -> None:
    """Draw a component's polygons as a 2.5D extruded layout (Magden-style).

    Each waveguide footprint is rendered as a shallow prism: a muted-red (``C3``
    by default) top face lifted by a *not-to-scale* height with shaded sidewalls
    showing beneath it, evoking the angled-sidewall rib cross-section in an
    oblique view (cf. Magden et al. 2018, Fig. 1). The vertical lift dominates
    (the propagation axis is usually plotted far longer than the transverse one,
    so an in-plane skew would be invisible); ``skew`` adds a small lateral shear
    for a hint of perspective.

    Args:
        component: the gdsfactory component to draw.
        ax: the matplotlib axes to draw on.
        color: top-face color (default matplotlib ``"C3"``, a muted red).
        depth: extruded visual height in data (y) units; defaults to ~12% of the
            drawn transverse extent (purely cosmetic, not to scale).
        skew: lateral (x) shear of the lift, as a fraction of ``depth``.
    """
    polys = _component_polygons(component)
    if not polys:
        return
    all_pts = np.vstack(polys)
    y_span = float(all_pts[:, 1].max() - all_pts[:, 1].min()) or 1.0
    if depth is None:
        depth = 0.12 * y_span
    lift = np.array([skew * depth, depth])  # screen offset from base to top face

    top_color = mcolors.to_rgb(color)
    wall_color = _shade(color, 0.55)  # darker red sidewalls

    bases = [np.asarray(p, dtype=float) for p in polys]
    tops = [b + lift for b in bases]

    # draw the (darker) base body first; the lifted top face is drawn on top and
    # offset, so the base peeks out below/aside it as a shaded sidewall band --
    # a clean 2.5D extruded look without per-facet tiling seams.
    ax.add_collection(
        PolyCollection(bases, facecolors=[wall_color], edgecolors="none")
    )
    ax.add_collection(
        PolyCollection(
            tops, facecolors=[top_color], edgecolors=_shade(color, 0.4),
            linewidths=0.3,
        )
    )
    ax.autoscale_view()
    ax.set_xlabel("x (propagation) [um]")
    ax.set_ylabel("y [um]")
