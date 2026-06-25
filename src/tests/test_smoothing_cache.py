"""The wavelength-independent subpixel-smoothing plan is computed once + reused."""

from __future__ import annotations

from unittest import mock

import numpy as np

import meow as mw
import meow.cross_section as xs


def _cell() -> mw.Cell:
    core = mw.Structure(
        material=mw.IndexMaterial(name="core", n=3.476),
        geometry=mw.Box(
            x_min=-0.27, x_max=0.27, y_min=0.0, y_max=0.22, z_min=-1.0, z_max=2.0
        ),
        mesh_order=5,
    )
    clad = mw.Structure(
        material=mw.IndexMaterial(name="clad", n=1.444),
        geometry=mw.Box(
            x_min=-1.5, x_max=1.5, y_min=-1.0, y_max=1.0, z_min=-1.0, z_max=2.0
        ),
        mesh_order=10,
    )
    mesh = mw.Mesh2D(x=np.linspace(-1.5, 1.5, 81), y=np.linspace(-1.0, 1.0, 61))
    return mw.create_cells([core, clad], mesh, [1.0], z_min=0.0)[0]


def test_cached_smoothing_matches_fresh() -> None:
    """Reusing a cell across wavelengths gives bit-identical smoothed index."""
    cell = _cell()
    wls = [1.50, 1.55, 1.60]
    cached = [
        np.asarray(mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=wl)).nx)
        for wl in wls
    ]
    for i, wl in enumerate(wls):
        fresh_cell = _cell()  # no shared plan cache
        fresh = np.asarray(
            mw.CrossSection.from_cell(cell=fresh_cell, env=mw.Environment(wl=wl)).nx
        )
        assert np.array_equal(cached[i], fresh)


def test_smoothing_plan_computed_once_per_component() -> None:
    """A wavelength sweep on one cell runs the shapely plan once per component."""
    cell = _cell()
    calls = {"n": 0}
    original = xs._smoothing_plan

    def counting(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return original(*args, **kwargs)

    with mock.patch.object(xs, "_smoothing_plan", counting):
        for wl in np.linspace(1.4, 1.7, 6):
            cs = mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=float(wl)))
            _, _, _ = cs.nx, cs.ny, cs.nz
    # 6 wavelengths x 3 components would be 18 without caching; expect exactly 3
    assert calls["n"] == 3


def test_cross_section_keeps_its_cell() -> None:
    """from_cell now actually stores the originating cell (private handle)."""
    cell = _cell()
    cs = mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=1.55))
    assert cs._cell is cell
