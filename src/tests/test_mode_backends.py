"""Sanity and agreement tests for the mode-solver backends.

The metrics tests check the backend-agnostic dispersion helpers on the
default (tidy3d) backend; the agreement tests compare the MPB backend
against tidy3d and are skipped when the meep/mpb bindings are missing.
"""

import importlib.util

import numpy as np
import pytest

import meow as mw

HAVE_MPB = importlib.util.find_spec("meep") is not None

SI3N4_N = 1.996
SIO2_N = 1.444
WL = 1.55


def si3n4_structures() -> list[mw.Structure3D]:
    """A rectangular Si3N4 strip (800 nm x 1.5 um) in SiO2."""
    core = mw.Structure(
        material=mw.IndexMaterial(name="test_si3n4", n=SI3N4_N),
        geometry=mw.Box(
            x_min=-0.75, x_max=0.75, y_min=0.0, y_max=0.8, z_min=0.0, z_max=1.0
        ),
        mesh_order=1,
    )
    clad = mw.Structure(
        material=mw.IndexMaterial(name="test_sio2", n=SIO2_N),
        geometry=mw.Box(x_min=-4, x_max=4, y_min=-3, y_max=3.8, z_min=0.0, z_max=1.0),
        mesh_order=9,
    )
    return [core, clad]


def si3n4_mesh(res: float = 0.05) -> mw.Mesh2D:
    return mw.Mesh2D(
        x=np.arange(-2.2, 2.2 + res / 2, res),
        y=np.arange(-1.4, 2.2 + res / 2, res),
    )


def test_metrics_sanity_tidy3d():
    m = mw.dispersion_metrics(si3n4_structures(), WL, si3n4_mesh(0.05), num_modes=2)
    # guided fundamental mode: between cladding and core index
    assert SIO2_N < m.neff < SI3N4_N
    # normal-ish dispersion: group index above phase index, below core index
    assert m.neff < m.group_index < 2.5
    # mode area on the order of the core area (1.2 um^2)
    assert 0.5 < m.effective_area < 3.0
    # fundamental mode of a wide strip is TE
    assert m.te_fraction > 0.8
    assert np.isfinite(m.dispersion_D)
    assert np.isfinite(m.beta2)


def test_effective_area_property():
    mode = mw.solve_mode(si3n4_structures(), WL, si3n4_mesh(0.05), num_modes=2)
    assert mode.effective_area == pytest.approx(mw.effective_area(mode))
    assert mode.effective_area > 0


def test_tidy3d_local_smoothing_detection():
    """Local dielectric smoothing toggles only when ``tidy3d-extras`` is present."""
    from meow.fde.tidy3d import HAS_TIDY3D_EXTRAS, _enable_local_smoothing

    assert isinstance(HAS_TIDY3D_EXTRAS, bool)
    _enable_local_smoothing(local_smoothing=False)  # never raises
    _enable_local_smoothing(local_smoothing=None)  # auto: no-op without extras
    if not HAS_TIDY3D_EXTRAS:
        with pytest.raises(ImportError, match="tidy3d-extras"):
            _enable_local_smoothing(local_smoothing=True)  # explicit, no package


@pytest.mark.skipif(not HAVE_MPB, reason="meep/mpb bindings not installed")
def test_mpb_deterministic_seed():
    """The seeded MPB backend reproduces the same neff (parallel-safe)."""
    structs, mesh = si3n4_structures(), si3n4_mesh(0.06)
    mpb = mw.compute_modes_mpb
    n1 = mw.solve_mode(structs, WL, mesh, num_modes=2, compute_modes=mpb)
    n2 = mw.solve_mode(structs, WL, mesh, num_modes=2, compute_modes=mpb)
    # reproducible well within the parallel consistency tolerance (neff_atol=1e-6)
    assert np.real(n1.neff) == pytest.approx(np.real(n2.neff), abs=1e-7)


@pytest.mark.skipif(not HAVE_MPB, reason="meep/mpb bindings not installed")
def test_mpb_agrees_with_tidy3d_neff():
    structs, mesh = si3n4_structures(), si3n4_mesh(0.05)
    mode_t3d = mw.solve_mode(
        structs, WL, mesh, num_modes=2, compute_modes=mw.compute_modes_tidy3d
    )
    mode_mpb = mw.solve_mode(
        structs, WL, mesh, num_modes=2, compute_modes=mw.compute_modes_mpb
    )
    assert np.real(mode_mpb.neff) == pytest.approx(np.real(mode_t3d.neff), abs=5e-3)
    assert mode_mpb.te_fraction == pytest.approx(mode_t3d.te_fraction, abs=0.1)


@pytest.mark.skipif(not HAVE_MPB, reason="meep/mpb bindings not installed")
def test_mpb_agrees_with_tidy3d_metrics():
    structs, mesh = si3n4_structures(), si3n4_mesh(0.05)
    m_t3d = mw.dispersion_metrics(
        structs, WL, mesh, num_modes=2, compute_modes=mw.compute_modes_tidy3d
    )
    m_mpb = mw.dispersion_metrics(
        structs, WL, mesh, num_modes=2, compute_modes=mw.compute_modes_mpb
    )
    assert m_mpb.neff == pytest.approx(m_t3d.neff, abs=5e-3)
    assert m_mpb.group_index == pytest.approx(m_t3d.group_index, abs=5e-2)
    assert m_mpb.effective_area == pytest.approx(m_t3d.effective_area, rel=0.2)
