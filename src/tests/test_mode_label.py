"""Tests for Hermite-Gaussian mode labeling.

These tests solve real multimode cross-sections (Si3N4 and an anisotropic
LiNbO3 core) and check that the HG labeler assigns the physically expected
``(polarization, m, n)`` to each guided mode, that the labels are consistent
with the independent ``te_fraction`` metric, and that label-based filtering can
be used to restrict an EME model to specific modes.
"""

import numpy as np
import pytest

import meow as mw
from meow.mode_label import (
    ModeLabel,
    filter_modes_by_label,
    hermite_gaussian_field,
    label_mode,
    label_mode_candidates,
    label_modes,
)

SIO2_N = 1.444
SI3N4_N = 1.996
# LiNbO3 near 1.55 um: ordinary ~2.21, extraordinary ~2.14 (anisotropic core).
LN_NO = 2.21
LN_NE = 2.14
WL = 1.55


def _waveguide_modes(
    core_material: mw.Material,
    *,
    core_w: float,
    core_h: float,
    num_modes: int,
    res: float = 0.05,
) -> mw.Modes:
    """Solve modes of a rectangular core embedded in SiO2."""
    core = mw.Structure(
        material=core_material,
        geometry=mw.Box(
            x_min=-core_w / 2,
            x_max=core_w / 2,
            y_min=0.0,
            y_max=core_h,
            z_min=0.0,
            z_max=1.0,
        ),
        mesh_order=1,
    )
    clad = mw.Structure(
        material=mw.IndexMaterial(name="ml_sio2", n=SIO2_N),
        geometry=mw.Box(
            x_min=-6, x_max=6, y_min=-3, y_max=3 + core_h, z_min=0.0, z_max=1.0
        ),
        mesh_order=9,
    )
    mesh = mw.Mesh2D(
        x=np.arange(-3.0, 3.0 + res / 2, res),
        y=np.arange(-1.6, 1.6 + core_h + res / 2, res),
    )
    cell = mw.Cell(structures=[core, clad], mesh=mesh, z_min=0.0, z_max=1.0)
    env = mw.Environment(wl=WL, T=25.0)
    cs = mw.CrossSection.from_cell(cell=cell, env=env)
    return mw.compute_modes(cs, num_modes=num_modes)


@pytest.fixture(scope="module")
def si3n4_modes() -> mw.Modes:
    """A Si3N4 strip wide enough to be horizontally multimode."""
    return _waveguide_modes(
        mw.IndexMaterial(name="ml_si3n4", n=SI3N4_N),
        core_w=3.0,
        core_h=0.7,
        num_modes=6,
    )


@pytest.fixture(scope="module")
def linbo3_modes() -> mw.Modes:
    """An anisotropic LiNbO3 core, horizontally multimode."""
    ln = mw.AnisotropicMaterial.from_n(name="ml_linbo3", n=(LN_NO, LN_NE, LN_NO))
    return _waveguide_modes(ln, core_w=2.0, core_h=0.6, num_modes=6)


# --- Hermite-Gaussian template sanity ------------------------------------


def test_hermite_gaussian_fundamental_is_gaussian():
    x = np.linspace(-5, 5, 201)
    y = np.linspace(-4, 4, 161)
    hg = hermite_gaussian_field(x, y, 0, 0, 0.0, 0.0, 1.5, 1.0)
    expected = np.exp(-((x[:, None] / 1.5) ** 2) - (y[None, :] / 1.0) ** 2)
    assert np.allclose(hg, expected)
    # peak at the center, strictly positive everywhere
    assert hg.min() > 0
    assert np.unravel_index(np.argmax(hg), hg.shape) == (100, 80)


def test_hermite_gaussian_parity_and_orthogonality():
    x = np.linspace(-6, 6, 401)  # symmetric grid about 0
    y = np.linspace(-6, 6, 401)
    hg00 = hermite_gaussian_field(x, y, 0, 0, 0.0, 0.0, 1.0, 1.0)
    hg10 = hermite_gaussian_field(x, y, 1, 0, 0.0, 0.0, 1.0, 1.0)
    # order 0 is even along x, order 1 is odd along x (one central node)
    assert np.allclose(hg00, hg00[::-1, :])
    assert np.allclose(hg10, -hg10[::-1, :])
    assert hg10[x > 0].max() > 0
    assert hg10[x < 0].min() < 0
    # distinct orders are orthogonal under the plain L2 inner product
    overlap = np.trapezoid(np.trapezoid(hg00 * hg10, y, axis=1), x, axis=0)
    assert abs(overlap) < 1e-6


# --- Labeling Si3N4 modes ------------------------------------------------


def test_fundamental_is_te00(si3n4_modes: mw.Modes):
    label = label_mode(si3n4_modes[0])
    assert isinstance(label, ModeLabel)
    assert label.name == "TE00"
    assert label.error < 0.05  # clean fit
    assert label.overlap == pytest.approx(1.0 - label.error)


def test_si3n4_first_modes_have_expected_labels(si3n4_modes: mw.Modes):
    labels = label_modes(si3n4_modes)
    names = [l.name for l in labels]
    # the fundamental (highest neff) is the TE00 mode
    assert names[0] == "TE00"
    # the lowest-order guided modes are all present (their exact neff ordering
    # depends on the core aspect ratio, e.g. TM00 vs TE10 can swap)
    assert {"TE00", "TE10", "TM00", "TM10"}.issubset(set(names))
    # confidently-labeled modes fit an HG profile well and are all distinct:
    # the method gives each guided mode its own unambiguous label
    confident = [l.name for l in labels if l.error < 0.15]
    assert len(confident) >= 4
    assert len(set(confident)) == len(confident)


def test_labels_agree_with_te_fraction(si3n4_modes: mw.Modes):
    for mode in si3n4_modes:
        label = label_mode(mode)
        if label.error > 0.15:
            continue  # skip ambiguous / poorly-confined modes
        if label.pol == "TE":
            assert mode.te_fraction > 0.5
        else:
            assert mode.te_fraction < 0.5


def test_widths_track_core_dimensions(si3n4_modes: mw.Modes):
    # the fundamental's fitted envelope should be wider than the (thin) core
    # height and comparable-to / wider-than nothing absurd along x
    label = label_mode(si3n4_modes[0])
    assert 0.1 < label.wy < 3.0
    assert 0.3 < label.wx < 6.0
    # centered roughly on the core (x ~ 0, y ~ core half-height 0.35)
    assert abs(label.x0) < 0.5
    assert 0.0 < label.y0 < 0.9


# --- Anisotropic LiNbO3 --------------------------------------------------


def test_linbo3_fundamental_and_consistency(linbo3_modes: mw.Modes):
    labels = label_modes(linbo3_modes)
    assert labels[0].name == "TE00"
    assert labels[0].error < 0.05
    # neff between cladding and (ordinary) core index for the fundamental
    assert SIO2_N < linbo3_modes[0].neff.real < LN_NO
    # te_fraction consistency on confident labels
    for mode, label in zip(linbo3_modes, labels, strict=True):
        if label.error < 0.15:
            assert (label.pol == "TE") == (mode.te_fraction > 0.5)


# --- Candidate table & options -------------------------------------------


def test_candidates_sorted_and_best_matches(si3n4_modes: mw.Modes):
    candidates = label_mode_candidates(si3n4_modes[1])
    errors = [c.error for c in candidates]
    assert errors == sorted(errors)  # ascending error
    assert candidates[0] == label_mode(si3n4_modes[1])
    # number of candidates = 2 pols * (max_order_x+1) * (max_order_y+1)
    assert len(candidates) == 2 * (4 + 1) * (3 + 1)
    # the TE10 mode exists in the multimode set and is found exactly once
    te10 = [m for m in si3n4_modes if label_mode(m).name == "TE10"]
    assert len(te10) == 1


def test_optimize_improves_or_matches_guess(si3n4_modes: mw.Modes):
    mode = si3n4_modes[0]
    err_opt = label_mode(mode, optimize=True).error
    err_raw = label_mode(mode, optimize=False).error
    # optimization can only reduce (or tie) the best achievable error
    assert err_opt <= err_raw + 1e-9


def test_mode_hg_label_method(si3n4_modes: mw.Modes):
    label = si3n4_modes[0].hg_label()
    assert label.name == "TE00"


# --- Filtering for an EME model ------------------------------------------


def test_filter_te00_unique(si3n4_modes: mw.Modes):
    te00 = filter_modes_by_label(si3n4_modes, pol="TE", m=0, n=0, max_error=0.2)
    assert len(te00) == 1
    assert te00[0] is si3n4_modes[0]


def test_filter_all_tm_modes(si3n4_modes: mw.Modes):
    tm = filter_modes_by_label(si3n4_modes, pol="TM", max_error=0.2)
    assert len(tm) >= 2
    for mode in tm:
        assert mode.te_fraction < 0.5


def test_filtered_modes_build_eme_s_matrix(si3n4_modes: mw.Modes):
    """Label-filtered modes can be fed straight into an EME S-matrix."""
    te_modes = filter_modes_by_label(si3n4_modes, pol="TE", max_error=0.2)
    assert len(te_modes) >= 2
    # two identical cells -> a straight section; only TE modes propagate
    S, port_map = mw.compute_s_matrix([te_modes, te_modes], cell_lengths=[5.0, 5.0])
    n = len(te_modes)
    # left + right input ports, one per selected mode
    assert np.asarray(S).shape[-1] == 2 * n
    # the fundamental transmits cleanly through the straight section
    i_in, i_out = port_map["left@0"], port_map["right@0"]
    transmission = np.abs(np.asarray(S)[..., i_out, i_in]) ** 2
    assert transmission > 0.9
