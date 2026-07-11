"""Tests for the bent-waveguide EME coupler examples (examples/papers).

Fast, coarse-resolution checks of the shared bent-coupler engine
(``_bent_coupler``) and the two device models (point / pulley), plus a smoke
test of each designer. The heavy full-example ``main()`` runs are exercised
separately; here we keep every solve small so the suite stays quick.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MEOW_EXAMPLE_RES", "low")

from examples.papers import _bent_coupler as bc  # noqa: E402

RES = 0.06  # coarse resolution for speed


# --- platforms + cross-sections -------------------------------------------
def test_platform_presets() -> None:
    soi = bc.soi_platform(0.22)
    sin = bc.sin_platform(0.78)
    assert soi.thickness == 0.22
    assert "SiN" in sin.name
    assert soi.substrate is None  # buried -> symmetric oxide


def test_bent_cross_section_solves() -> None:
    pf = bc.sin_platform(0.78, buried=True)
    n_bent, _ = bc.isolated_neff(pf, 1.0, 1.55, bend_radius=20.0, res=RES)
    n_straight, _ = bc.isolated_neff(pf, 1.0, 1.55, bend_radius=np.inf, res=RES)
    # guided, and the bend lowers neff below the straight value
    assert 1.45 < n_bent < 2.0
    assert n_bent < n_straight + 1e-6


# --- coupling extraction ---------------------------------------------------
def test_symmetric_split_positive() -> None:
    pf = bc.sin_platform(0.78, buried=True)
    k = bc.symmetric_kappa0(pf, 1.0, 0.4, 1.55, bend_radius=20.0, res=RES)
    assert k > 0


def test_supermode_coupling_self_consistent() -> None:
    """kappa0, delta obey the 2-level identity S = 2 sqrt(kappa0^2 + delta^2)."""
    pf = bc.sin_platform(0.78, buried=True)
    k0, delta = bc.supermode_coupling(pf, 1.5, 0.7, 0.6, 1.55, bend_radius=23.0,
                                      res=RES)
    n_hi, n_lo, _ = bc.supermode_split(pf, 1.5, 0.7, 0.6, 1.55, bend_radius=23.0,
                                       res=RES)
    s = (2 * np.pi / 1.55) * (n_hi - n_lo)
    assert k0 >= 0
    assert np.isclose(s, 2 * np.hypot(k0, delta), rtol=1e-6)


def test_kappa0_decays_with_gap() -> None:
    pf = bc.soi_platform(0.22, buried=True)
    gaps = np.array([0.12, 0.20, 0.30])
    model = bc.calibrate_coupling(pf, 0.45, 0.45, gaps, 1.55, radius=10.0,
                                  mode="point", res=RES)
    assert model.gamma > 0
    assert model.kappa0(0.12) > model.kappa0(0.30)  # monotonic evanescent decay


# --- coupling laws ---------------------------------------------------------
def test_point_cross_power_bounds_and_monotonic() -> None:
    pf = bc.soi_platform(0.22, buried=True)
    model = bc.calibrate_coupling(pf, 0.45, 0.45, np.array([0.12, 0.20, 0.30]),
                                  1.55, radius=10.0, mode="point", res=RES)
    k2 = [bc.point_cross_power(model, g, 10.0) for g in (0.12, 0.20, 0.30)]
    assert all(0.0 <= v <= 1.0 for v in k2)
    assert k2[0] > k2[1] > k2[2]  # tighter gap -> more coupling


def test_pulley_law_and_sinc_limit() -> None:
    k0, delta = 0.02, 0.0
    # phase-matched: full transfer at the beat length
    lpi = np.pi / (2 * k0)
    assert np.isclose(bc.pulley_cross_power(k0, delta, lpi), 1.0, atol=1e-6)
    # small-Lc: exact and sinc forms agree
    assert np.isclose(bc.pulley_cross_power(k0, delta, 1.0),
                      bc.pulley_cross_power_sinc(k0, delta, 1.0), rtol=1e-3)
    # mismatch caps the maximum transfer
    fmax = 0.02**2 / (0.02**2 + 0.05**2)
    peak = max(bc.pulley_cross_power(0.02, 0.05, L) for L in np.linspace(0, 200, 400))
    assert np.isclose(peak, fmax, atol=1e-2)


# --- ring observables ------------------------------------------------------
def test_all_pass_critical_coupling_zero_on_resonance() -> None:
    a = 0.97
    on_res = bc.all_pass_transmission(a, a, np.array([0.0]))[0]  # t = a
    assert on_res < 1e-6


def test_q_and_extraction() -> None:
    qc = bc.q_coupling(0.01, 2.0, 23.0, 1.55)
    assert qc > 0
    assert 0.0 <= bc.extraction_efficiency(qc, 3e6) <= 1.0
    qi = bc.q_intrinsic(0.1, 2.0, 1.55)  # 0.1 dB/cm SiN -> very high Q
    assert qi > 1e6


# --- designer inversions ---------------------------------------------------
def test_design_point_gap_hits_target() -> None:
    pf = bc.soi_platform(0.22, buried=True)
    model = bc.calibrate_coupling(pf, 0.45, 0.45, np.array([0.10, 0.18, 0.28]),
                                  1.55, radius=15.0, mode="point", res=RES)
    target = 0.02
    gap = bc.design_point_gap(model, 15.0, target)
    got = bc.point_cross_power(model, gap, 15.0)
    # either hit the target or saturate at a range endpoint
    assert (abs(got - target) < 0.01) or gap in (0.05, 0.6)


def test_design_pulley_length_reaches_target() -> None:
    k0, delta = 0.03, 0.0  # phase-matched
    length = bc.design_pulley_length(k0, delta, 0.5)
    assert np.isclose(bc.pulley_cross_power(k0, delta, length), 0.5, atol=0.02)


@pytest.mark.slow
def test_point_designer_smoke() -> None:
    from examples.papers import point_coupler_designer as pcd

    d = pcd.design_point_coupler(bc.soi_platform(0.22), 12.0, 1.31, 0.02,
                                 width=0.4, label="t", res_um=RES)
    assert 0.04 < d.gap < 0.6
    assert d.achieved_kappa2 <= 1.0


@pytest.mark.slow
def test_pulley_designer_smoke() -> None:
    from examples.papers import pulley_coupler_designer as pcd

    d = pcd.design_pulley_coupler(bc.sin_platform(0.78), 23.0, 1.55, 0.2,
                                  ring_width=1.5, gap=0.6, label="t", res_um=RES)
    assert d.length > 0
    assert 0.3 < d.bus_width < 2.0
