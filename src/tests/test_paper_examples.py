"""Tests for the paper-reproduction examples (examples/papers)."""

import os
import sys
from pathlib import Path

import gdsfactory as gf
import numpy as np
import pytest

import meow as mw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MEOW_EXAMPLE_FAST", "1")

gf.gpdk.PDK.activate()

from examples.papers import (  # noqa: E402
    kwolek2026_faquad as kw,
)
from examples.papers import (  # noqa: E402
    magden2018_dichroic as md,
)

# --- Magden 2018: silicon dichroic filter ---


def test_dichroic_filter_layout() -> None:
    c = md.dichroic_filter()
    port_names = {p.name for p in c.ports}
    assert {"in0", "short_pass", "long_pass"} <= port_names
    total_length = md.L1 + md.L2 + md.L3 + md.L4
    assert np.isclose(c.xmax - c.xmin, total_length, atol=1e-3)
    # WGA + 3 WGB segments
    polys = [p for ps in c.get_polygons().values() for p in ps]
    assert len(polys) == 4


def test_dichroic_filter_extrusion() -> None:
    c = md.dichroic_filter()
    structs = md.extrude_filter(c)
    si = [s for s in structs if s.material.name == "silicon"]
    oxide = [s for s in structs if "oxide" in s.material.name]
    assert len(si) == 4
    assert len(oxide) >= 1


def test_phase_matching_cutoff_in_c_band() -> None:
    """The WGA(318nm)/WGB phase-matching point lies in the C-band.

    This is the central design result of the paper (Fig. 1e: cutoff near
    1540 nm). At converged resolution (<= 20 nm) the sign of
    n_WGA - n_WGB must flip between 1500 and 1600 nm.
    """
    mesh = md.mesh2d(res=0.02)
    diffs = []
    for wl in (1.50, 1.60):
        n_a = md.fundamental_neff(md.wga_structures(0.318), wl, mesh=mesh)
        n_b = md.fundamental_neff(md.wgb_structures(), wl, mesh=mesh)
        diffs.append(n_a - n_b)
    assert diffs[0] > 0  # below cutoff: WGA has the higher index
    assert diffs[1] < 0  # above cutoff: WGB has the higher index


def test_analytical_transmission() -> None:
    gammas = np.array([-1e3, 0.0, 1e3])
    t = md.analytical_transmission(gammas)
    assert np.isclose(t[1], 0.5)  # gamma = 0 marks the 3 dB cutoff
    assert t[0] < 1e-6
    assert t[2] > 1 - 1e-6
    assert np.all(np.diff(md.analytical_transmission(np.linspace(-5, 5, 50))) > 0)


def test_device_cells_pipeline() -> None:
    c = md.dichroic_filter()
    cells = md.device_cells(c, cells_per_section=(2, 2, 3, 2), mesh=md.mesh2d(res=0.05))
    assert len(cells) == 9
    assert np.isclose(sum(cell.length for cell in cells), md.L1 + md.L2 + md.L3 + md.L4)


# --- Kwolek 2026: TFLN FAQUAD combiner ---


@pytest.fixture(scope="module")
def calibration() -> tuple[float, float, float]:
    return kw.calibrate(1.55, res=0.06)


def test_calibration(calibration: tuple[float, float, float]) -> None:
    kappa_0, g_0, dbeta_dtw = calibration
    assert kappa_0 > 0
    assert 0.1 < g_0 < 1.5  # evanescent decay length in a plausible range
    assert dbeta_dtw > 0  # wider waveguide -> higher beta


def test_faquad_design_profiles(calibration: tuple[float, float, float]) -> None:
    design = kw.FaquadDesign(*calibration)
    # gap profile: constant g_m in region I, g_c at the end of the cubic
    assert np.isclose(design.gap(0.0), kw.G_M)
    assert np.isclose(design.gap(design.l_m / 2), kw.G_M)
    assert np.isclose(design.gap(design.z_c), kw.G_C, atol=1e-6)
    assert np.isclose(design.gap(design.half_length), kw.G_F)
    # eta from paper Eq. 12
    from scipy.special import gamma as gamma_fn

    eta_expected = 1.0 / (
        design.kappa_m * (design.l_m + (2 / 3) * gamma_fn(1 / 3) * design.z_0)
    )
    assert np.isclose(design.eta, eta_expected)
    # mixing angle: chi(0) = pi/2, antisymmetric about the center
    assert np.isclose(design.chi(0.0), np.pi / 2)
    z = np.linspace(-design.half_length, design.half_length, 31)
    assert np.allclose(np.cos(design.chi(z)), -np.cos(design.chi(-z)), atol=1e-9)
    assert np.all(np.diff(design.chi(z)) >= -1e-9)  # monotonic 0 -> pi
    # top-width difference: antisymmetric and clipped to dtw_max
    dtw = design.dtw(z)
    assert np.allclose(dtw, -dtw[::-1], atol=1e-9)
    assert np.max(np.abs(dtw)) <= design.dtw_max + 1e-12


def test_ln_material_anisotropy() -> None:
    ln = kw.ln_material(1.55)
    assert isinstance(ln, mw.AnisotropicMaterial)
    assert not ln.is_isotropic
    eps = np.real(np.diag(ln.eps))
    ne, no_y, no_z = np.sqrt(eps)
    assert np.isclose(no_y, no_z)
    assert ne < no_y  # negative uniaxial crystal
    assert 2.1 < ne < 2.2
    assert 2.2 < no_y < 2.3


def test_combiner_layout(calibration: tuple[float, float, float]) -> None:
    c = kw.faquad_combiner(*calibration)
    port_names = {p.name for p in c.ports}
    assert {"in_bar", "out_bar", "out_cross"} <= port_names
    polys = [p for ps in c.get_polygons().values() for p in ps]
    assert len(polys) == 2
    design = kw.FaquadDesign(*calibration)
    assert np.isclose(c.xmax - c.xmin, 2 * design.half_length, atol=0.2)


def test_combiner_extrusion_has_angled_sidewalls(
    calibration: tuple[float, float, float],
) -> None:
    c = kw.faquad_combiner(*calibration)
    structs = kw.device_structures(c, 1.55)
    prisms = [s for s in structs if isinstance(s.geometry, mw.Prism)]
    assert len(prisms) == 2
    for s in prisms:
        assert s.geometry.sidewall_angle == kw.SIDEWALL_DEG
        assert isinstance(s.material, mw.AnisotropicMaterial)


def test_sh_stays_in_bar_port(calibration: tuple[float, float, float]) -> None:
    """At the second harmonic the mode is decoupled and stays in its port.

    This is the dichroic behavior of the combiner (paper Fig. 1f): even a
    coarse model shows orders of magnitude between bar and cross at SH.
    """
    c = kw.faquad_combiner(*calibration)
    cells = kw.device_cells(c, 0.775, num_cells=10, res=0.06)
    t_bar, t_cross = kw.bar_cross_transmission(cells, 0.775, num_modes=3)
    # at this very coarse test resolution most of the SH power is lost to
    # discretization, but the bar/cross contrast remains well over an order
    # of magnitude (the full-resolution figures show > 30 dB extinction)
    assert t_bar > 20 * t_cross
