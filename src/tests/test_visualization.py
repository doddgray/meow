"""Tests for the cross-section visualization tool."""

from collections.abc import Iterator

import matplotlib as mpl

mpl.use("Agg")

import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import QuadMesh
from matplotlib.patches import PathPatch

import meow as mw

gf.gpdk.PDK.activate()

WIDTH = 0.5
THICKNESS = 0.22
SIDEWALL_ANGLE = 20.0


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def _make_cs(sidewall_angle: float = SIDEWALL_ANGLE) -> mw.CrossSection:
    c = gf.components.straight(length=10.0, width=WIDTH)
    extrusions = {
        (1, 0): [
            mw.GdsExtrusionRule(
                material=mw.silicon,
                h_min=0.0,
                h_max=THICKNESS,
                sidewall_angle=sidewall_angle,
                mesh_order=1,
            ),
            mw.GdsExtrusionRule(
                material=mw.silicon_oxide,
                h_min=-0.3,
                h_max=0.0,
                mesh_order=2,
            ),
        ],
    }
    structs = mw.extrude_gds(c, extrusions)
    mesh = mw.Mesh2D(
        x=np.linspace(-1.0, 1.0, 101),
        y=np.linspace(-0.4, 0.5, 46),
    )
    cell = mw.Cell(structures=structs, mesh=mesh, z_min=0.0, z_max=10.0)
    return mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=1.55, T=25.0))


def _structure_paths(ax: plt.Axes) -> list[np.ndarray]:
    return [p.get_path().vertices for p in ax.patches if isinstance(p, PathPatch)]


def test_polygon_style_renders_exact_trapezoid() -> None:
    """The polygon style draws the angled sidewalls exactly (not pixelated)."""
    cs = _make_cs()
    _, ax = plt.subplots()
    cs._visualize(ax=ax, show=False, style="polygons")

    paths = _structure_paths(ax)
    assert len(paths) == 2  # silicon core + oxide slab

    (core,) = [v for v in paths if np.isclose(v[:, 1].max(), THICKNESS)]
    bottom = core[np.isclose(core[:, 1], 0.0), 0]
    top = core[np.isclose(core[:, 1], THICKNESS), 0]
    w_top_expected = WIDTH - 2 * THICKNESS * np.tan(np.deg2rad(SIDEWALL_ANGLE))
    assert np.isclose(bottom.max() - bottom.min(), WIDTH)
    assert np.isclose(top.max() - top.min(), w_top_expected)


def test_polygon_style_is_default() -> None:
    """mw.visualize(cs) uses the exact polygon rendering by default."""
    cs = _make_cs()
    _, ax = plt.subplots()
    mw.visualize(cs, ax=ax, show=False)
    assert len(_structure_paths(ax)) == 2
    assert not [c for c in ax.collections if isinstance(c, QuadMesh)]


def test_polygon_style_clips_to_mesh_bounds() -> None:
    """Structures wider than the mesh are clipped to the mesh bounds."""
    cs = _make_cs()
    _, ax = plt.subplots()
    cs._visualize(ax=ax, show=False, style="polygons")
    x_min, x_max = cs.mesh.x.min(), cs.mesh.x.max()
    y_min, y_max = cs.mesh.y.min(), cs.mesh.y.max()
    for verts in _structure_paths(ax):
        assert np.all(verts[:, 0] >= x_min)
        assert np.all(verts[:, 0] <= x_max)
        assert np.all(verts[:, 1] >= y_min)
        assert np.all(verts[:, 1] <= y_max)


def test_polygon_style_adds_material_colorbar() -> None:
    cs = _make_cs()
    fig, ax = plt.subplots()
    cs._visualize(ax=ax, show=False, style="polygons")
    assert len(fig.axes) == 2  # main axes + colorbar
    labels = [t.get_text() for t in fig.axes[1].get_yticklabels()]
    assert any("air" in label for label in labels)
    assert any("silicon" in label for label in labels)


def test_pixelated_style_still_available() -> None:
    """The original rasterized visualization remains accessible."""
    cs = _make_cs()
    _, ax = plt.subplots()
    cs._visualize(ax=ax, show=False, style="pixelated")
    assert [c for c in ax.collections if isinstance(c, QuadMesh)]
    assert not _structure_paths(ax)


def test_debug_grid_implies_pixelated_style() -> None:
    cs = _make_cs()
    _, ax = plt.subplots()
    cs._visualize(ax=ax, show=False, debug_grid=True)
    assert [c for c in ax.collections if isinstance(c, QuadMesh)]


def test_vertical_sidewalls_render_as_rectangles() -> None:
    """Zero sidewall angle gives rectangular structure patches."""
    cs = _make_cs(sidewall_angle=0.0)
    _, ax = plt.subplots()
    cs._visualize(ax=ax, show=False, style="polygons")
    (core,) = [v for v in _structure_paths(ax) if np.isclose(v[:, 1].max(), THICKNESS)]
    widths = {
        float(np.ptp(core[np.isclose(core[:, 1], y), 0])) for y in (0.0, THICKNESS)
    }
    assert all(np.isclose(w, WIDTH) for w in widths)
