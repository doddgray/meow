"""Small plotting helpers shared by the paper examples."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.collections import PolyCollection


def _poly_area(p: np.ndarray) -> float:
    """Absolute polygon area (shoelace)."""
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _strip_thickness(p: np.ndarray) -> float:
    """Approximate transverse thickness of a (long) waveguide strip.

    For a strip running mostly along the propagation (x) axis, ``area / length``
    recovers the local width even when the strip meanders (so its bounding box
    is much taller than the waveguide is wide).
    """
    x = p[:, 0]
    length = float(x.max() - x.min()) or 1.0
    return _poly_area(p) / length


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
    color: str | Sequence[str] = "C3",
    *,
    sidewall_frac: float = 0.3,
    skew: float = 0.25,
    depth: float | None = None,
) -> None:
    """Draw a component's polygons as a 2.5D extruded layout (Magden-style).

    Each waveguide footprint is the muted top face (``C3`` muted red by default);
    a darker, *thinner* sidewall band sits flush beneath it (a downward
    drop-shadow), evoking the angled-sidewall rib seen in an oblique view (cf.
    Magden et al. 2018, Fig. 1). The band is sized to a fraction of the
    waveguide's own transverse thickness so it never gaps away from -- nor
    overwhelms -- the top face, on both narrow and wide waveguides.

    Args:
        component: the gdsfactory component to draw.
        ax: the matplotlib axes to draw on.
        color: top-face color(s). A single color (default ``"C3"``) applies to
            every waveguide; pass a sequence to give each waveguide layer (in
            polygon order, cycled) its own base color.
        sidewall_frac: sidewall-band thickness as a fraction of the waveguide
            thickness (~1/3 by default); the band is drawn flush below the top.
        skew: small lateral (x) shear of the band, as a fraction of its depth,
            for a hint of perspective.
        depth: explicit band depth in data units (overrides ``sidewall_frac``).
    """
    polys = _component_polygons(component)
    if not polys:
        return
    colors = [color] * len(polys) if isinstance(color, str) else list(color)
    colors = [colors[i % len(colors)] for i in range(len(polys))]

    if depth is None:
        thick = float(np.median([_strip_thickness(p) for p in polys])) or 1.0
        depth = sidewall_frac * thick
    off = np.array([skew * depth, -depth])  # drop-shadow: flush below + slight side

    bases = [np.asarray(p, dtype=float) for p in polys]
    # darker sidewall band first (footprint shifted down so it stays flush under
    # the top face -- no gap), then the top face on top.
    ax.add_collection(
        PolyCollection(
            [b + off for b in bases],
            facecolors=[_shade(c, 0.55) for c in colors], edgecolors="none",
        )
    )
    ax.add_collection(
        PolyCollection(
            bases, facecolors=[mcolors.to_rgb(c) for c in colors],
            edgecolors=[_shade(c, 0.4) for c in colors], linewidths=0.3,
        )
    )
    ax.autoscale_view()
    ax.set_xlabel("x (propagation) [um]")
    ax.set_ylabel("y [um]")
