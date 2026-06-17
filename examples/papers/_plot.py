"""Small plotting helpers shared by the paper examples."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.collections import PolyCollection


def plot_component(component: Any, ax: Any, color: str = "#86b5dc") -> None:
    """Draw a gdsfactory component's polygons onto a matplotlib axes."""
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
    ax.add_collection(
        PolyCollection(polys, facecolors=color, edgecolors="k", linewidths=0.2)
    )
    ax.autoscale_view()
    ax.set_xlabel("x (propagation) [um]")
    ax.set_ylabel("y [um]")
