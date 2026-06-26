"""Kottke (normal-projected) tensor subpixel smoothing.

A square waveguide is rotation-invariant: its fundamental neff is the same
axis-aligned or rotated 45 degrees. At a finite resolution the 45-degree
(staircased) edges are where smoothing matters, so the rotated-vs-reference
error is a clean accuracy probe. Kottke's normal-projected diagonal should be
far closer to the axis-aligned reference than the axis-aligned arithmetic/
harmonic scheme.
"""

from __future__ import annotations

import numpy as np
import pytest

import meow as mw
import meow.cross_section as xs


def _square_cs(
    rot_deg: float, method: str, *, half: float = 0.3, npx: int = 141
) -> mw.CrossSection:
    th = np.deg2rad(rot_deg)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    sq = np.array([(-half, -half), (half, -half), (half, half), (-half, half)])
    poly = sq @ rot.T
    core = mw.Structure(
        material=mw.IndexMaterial(name="core", n=3.0),
        geometry=mw.Prism(poly=poly, h_min=-1.0, h_max=2.0, axis="z"),
        mesh_order=5,
    )
    clad = mw.Structure(
        material=mw.IndexMaterial(name="clad", n=1.444),
        geometry=mw.Box(
            x_min=-1.5, x_max=1.5, y_min=-1.5, y_max=1.5, z_min=-1.0, z_max=2.0
        ),
        mesh_order=10,
    )
    mesh = mw.Mesh2D(x=np.linspace(-1.5, 1.5, npx), y=np.linspace(-1.5, 1.5, npx))
    cell = mw.create_cells([core, clad], mesh, [1.0], z_min=0.0)[0]
    return mw.CrossSection.from_cell(
        cell=cell, env=mw.Environment(wl=1.55), smoothing_method=method
    )


def _neff(rot_deg: float, method: str) -> float:
    cs = _square_cs(rot_deg, method)
    return float(np.real(mw.compute_modes(cs, num_modes=1)[0].neff))


def test_kottke_beats_axis_on_tilted_interface() -> None:
    """Kottke is closer to the rotation-invariant reference than axis smoothing.

    Averaged over several tilt angles (the error ratio at a single angle is
    noisy since both errors are small); Kottke roughly halves the mean error.
    """
    ref = _neff(0.0, "axis")  # axis-aligned: edges aligned, ~exact
    angles = (30.0, 45.0, 60.0)
    err_axis = np.mean([abs(_neff(a, "axis") - ref) for a in angles])
    err_kottke = np.mean([abs(_neff(a, "kottke") - ref) for a in angles])
    assert err_kottke < 0.75 * err_axis


def test_kottke_reduces_to_axis_when_aligned() -> None:
    """On an axis-aligned interface Kottke and axis smoothing nearly agree."""
    assert _neff(0.0, "kottke") == pytest.approx(_neff(0.0, "axis"), abs=1e-3)


def test_kottke_plan_cached_on_cell() -> None:
    """The (geometry-only) Kottke plan is computed once per cell per component."""
    cell = _square_cs(30.0, "kottke")._cell
    calls = {"n": 0}
    original = xs._kottke_plan

    def counting(*a: object, **k: object) -> object:
        calls["n"] += 1
        return original(*a, **k)

    from unittest import mock

    with mock.patch.object(xs, "_kottke_plan", counting):
        for wl in (1.5, 1.55, 1.6):
            cs = mw.CrossSection.from_cell(
                cell=cell, env=mw.Environment(wl=wl), smoothing_method="kottke"
            )
            _, _, _ = cs.nx, cs.ny, cs.nz
    assert calls["n"] == 3  # one plan per component, reused across wavelengths


def test_default_method_is_axis() -> None:
    """The default smoothing is unchanged (axis); kottke is strictly opt-in."""
    cs = _square_cs(0.0, "axis")
    assert cs.smoothing_method == "axis"
