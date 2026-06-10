"""Tests for GDS extrusion with angled sidewalls."""

import gdsfactory as gf
import numpy as np
import pytest

import meow as mw

gf.gpdk.PDK.activate()

WG_LAYER = (1, 0)
WIDTH = 0.5
THICKNESS = 0.22
LENGTH = 10.0


def _straight_structs(
    sidewall_angle: float, width: float = WIDTH
) -> list[mw.Structure3D]:
    """Extrude a gdsfactory straight waveguide with the given sidewall angle."""
    c = gf.components.straight(length=LENGTH, width=width)
    extrusions = {
        WG_LAYER: [
            mw.GdsExtrusionRule(
                material=mw.silicon,
                h_min=0.0,
                h_max=THICKNESS,
                sidewall_angle=sidewall_angle,
                mesh_order=1,
            ),
        ],
    }
    return mw.extrude_gds(c, extrusions)


def _project_mid(structs: list[mw.Structure3D]) -> list[mw.geometries.Geometry2D]:
    """Project the extruded structures onto the waveguide mid cross-section."""
    geoms = []
    for s in structs:
        geoms.extend(s2d.geometry for s2d in s._project(LENGTH / 2))
    return geoms


def _widths(poly: np.ndarray, y: float) -> float:
    """Width of the polygon edge at height y."""
    xs = poly[np.isclose(poly[:, 1], y), 0]
    return float(xs.max() - xs.min())


def test_extrusion_rule_passes_sidewall_angle() -> None:
    structs = _straight_structs(sidewall_angle=10.0)
    assert len(structs) == 1
    geometry = structs[0].geometry
    assert isinstance(geometry, mw.Prism)
    assert geometry.sidewall_angle == 10.0


def test_zero_angle_projects_to_rectangle() -> None:
    """Without a sidewall angle the cross-section stays rectangular."""
    (geom,) = _project_mid(_straight_structs(sidewall_angle=0.0))
    assert isinstance(geom, mw.Rectangle)
    assert np.isclose(geom.x_max - geom.x_min, WIDTH)
    assert np.isclose(geom.y_min, 0.0)
    assert np.isclose(geom.y_max, THICKNESS)


@pytest.mark.parametrize("angle", [5.0, 10.0, 20.0])
def test_positive_angle_projects_to_trapezoid(angle: float) -> None:
    """The cross-section is a trapezoid narrowing with height by tan(angle)."""
    (geom,) = _project_mid(_straight_structs(sidewall_angle=angle))
    assert isinstance(geom, mw.Polygon2D)
    poly = geom.poly
    assert poly.shape == (4, 2)
    assert np.isclose(poly[:, 1].min(), 0.0)
    assert np.isclose(poly[:, 1].max(), THICKNESS)
    w_top_expected = WIDTH - 2 * THICKNESS * np.tan(np.deg2rad(angle))
    assert np.isclose(_widths(poly, 0.0), WIDTH)
    assert np.isclose(_widths(poly, THICKNESS), w_top_expected)


def test_negative_angle_widens_with_height() -> None:
    (geom,) = _project_mid(_straight_structs(sidewall_angle=-10.0))
    assert isinstance(geom, mw.Polygon2D)
    poly = geom.poly
    w_top_expected = WIDTH + 2 * THICKNESS * np.tan(np.deg2rad(10.0))
    assert np.isclose(_widths(poly, 0.0), WIDTH)
    assert np.isclose(_widths(poly, THICKNESS), w_top_expected)


def test_steep_angle_projects_to_triangle() -> None:
    """If the sidewalls meet below the top, the cross-section is a triangle."""
    angle = 60.0
    (geom,) = _project_mid(_straight_structs(sidewall_angle=angle))
    assert isinstance(geom, mw.Polygon2D)
    poly = geom.poly
    assert poly.shape == (3, 2)
    h_apex_expected = 0.5 * WIDTH / np.tan(np.deg2rad(angle))
    assert h_apex_expected < THICKNESS
    assert np.isclose(poly[:, 1].max(), h_apex_expected)
    assert np.isclose(_widths(poly, 0.0), WIDTH)


def test_prism_axis_z_sidewall() -> None:
    """A z-extruded prism slice shrinks with height above its base."""
    prism = mw.Prism(
        poly=np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]),
        h_min=0.0,
        h_max=0.5,
        axis="z",
        sidewall_angle=45.0,
    )
    (geom,) = prism._project(0.25)
    assert isinstance(geom, mw.Polygon2D)
    # 45 degrees -> inset by 0.25 on all sides: 1x1 square becomes 0.5x0.5
    assert np.isclose(geom._shapely_polygon().area, 0.25)
    assert prism._project(-0.1) == []
    assert prism._project(0.6) == []


def test_prism_axis_x_sidewall() -> None:
    """An x-extruded prism slice narrows along y with increasing x."""
    prism = mw.Prism(
        poly=np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]),
        h_min=0.0,
        h_max=0.2,
        axis="x",
        sidewall_angle=10.0,
    )
    (geom,) = prism._project(0.5)
    assert isinstance(geom, mw.Polygon2D)
    poly = geom.poly
    inset = 0.2 * np.tan(np.deg2rad(10.0))
    ys_bottom = poly[np.isclose(poly[:, 0], 0.0), 1]
    ys_top = poly[np.isclose(poly[:, 0], 0.2), 1]
    assert np.isclose(ys_bottom.max() - ys_bottom.min(), 1.0)
    assert np.isclose(ys_top.max() - ys_top.min(), 1.0 - 2 * inset)


def _make_cs(structs: list[mw.Structure3D]) -> mw.CrossSection:
    mesh = mw.Mesh2D(
        x=np.linspace(-1.0, 1.0, 201),
        y=np.linspace(-0.3, 0.5, 81),
    )
    cell = mw.Cell(structures=structs, mesh=mesh, z_min=0.0, z_max=LENGTH)
    env = mw.Environment(wl=1.55, T=25.0)
    return mw.CrossSection.from_cell(cell=cell, env=env)


def test_trapezoid_rasterization() -> None:
    """The rasterized material mask narrows with height as tan(angle)."""
    angle = 15.0
    cs = _make_cs(_straight_structs(sidewall_angle=angle))
    m_full = np.asarray(cs._m_full)
    x_full = cs.mesh.x_full
    y_full = cs.mesh.y_full
    dx_full = x_full[1] - x_full[0]

    def width_at(y: float) -> float:
        j = int(np.argmin(np.abs(y_full - y)))
        return float(np.count_nonzero(m_full[:, j] == 1) * dx_full)

    w_bottom = width_at(0.02)
    w_top = width_at(0.20)
    w_diff_expected = 2 * (0.20 - 0.02) * np.tan(np.deg2rad(angle))
    # rasterization quantizes each sidewall to the half-pixel grid
    assert np.isclose(w_bottom - w_top, w_diff_expected, atol=2 * dx_full)
    assert w_bottom < WIDTH + 2 * dx_full


def test_trapezoid_mode_solve() -> None:
    """Modes of the trapezoidal waveguide solve fine; less material -> lower neff."""
    neffs = {}
    for angle in [0.0, 20.0]:
        cs = _make_cs(_straight_structs(sidewall_angle=angle))
        modes = mw.compute_modes(cs, num_modes=1)
        neffs[angle] = float(np.real(modes[0].neff))
    assert 1.0 < neffs[20.0] < neffs[0.0]
