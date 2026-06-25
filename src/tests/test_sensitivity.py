"""Tests for the modal sensitivity / adjoint kernel (meow.sensitivity).

The perturbation-theory ``d(neff)/d(epsilon)`` kernel is validated against
finite differences of a real mode solve. Lossless silicon-on-insulator
waveguide; the global-index-scaling perturbation uses the actual (subpixel-
smoothed) permittivity so there is no boundary-pixel ambiguity.
"""

from __future__ import annotations

import numpy as np
import pytest

import meow as mw


def _solve(n_core: float, *, scale: float = 1.0, num_modes: int = 1) -> list[mw.Mode]:
    """A silicon-strip waveguide cross-section; all indices scaled by ``scale``."""
    core = mw.Structure(
        material=mw.IndexMaterial(name="core", n=n_core * scale),
        geometry=mw.Box(
            x_min=-0.25, x_max=0.25, y_min=0.0, y_max=0.22, z_min=-1.0, z_max=2.0
        ),
        mesh_order=5,
    )
    clad = mw.Structure(
        material=mw.IndexMaterial(name="clad", n=1.444 * scale),
        geometry=mw.Box(
            x_min=-1.5, x_max=1.5, y_min=-1.0, y_max=1.0, z_min=-1.0, z_max=2.0
        ),
        mesh_order=10,
    )
    mesh = mw.Mesh2D(x=np.linspace(-1.5, 1.5, 141), y=np.linspace(-1.0, 1.0, 101))
    cells = mw.create_cells([core, clad], mesh, [1.0], z_min=0.0)
    cs = mw.CrossSection.from_cell(cell=cells[0], env=mw.Environment(wl=1.55, T=25.0))
    return mw.compute_modes(cs, num_modes=num_modes)


@pytest.fixture(scope="module")
def mode() -> mw.Mode:
    return _solve(3.476)[0]


def test_modal_power_is_normalized(mode: mw.Mode) -> None:
    """meow normalizes modes to unit power, the perturbation denominator."""
    assert mw.modal_power(mode) == pytest.approx(1.0, abs=1e-6)


def test_sensitivity_density_shapes_and_localization(mode: mw.Mode) -> None:
    """The sensitivity maps are mesh-shaped, real, and peak inside the core."""
    s_xx, s_yy, s_zz = mw.neff_sensitivity(mode)
    assert s_xx.shape == np.asarray(mode.Ex).shape
    for s in (s_xx, s_yy, s_zz):
        assert np.isrealobj(s)
        assert np.all(s >= 0.0)
    # the largest Ex sensitivity sits near the waveguide core (|x|<0.25, 0<y<0.22)
    ix, iy = np.unravel_index(int(np.argmax(s_xx)), s_xx.shape)
    assert abs(float(mode.mesh.x_[ix])) < 0.4
    assert -0.1 < float(mode.mesh.y_[iy]) < 0.32


def test_neff_gradient_matches_finite_difference() -> None:
    """The adjoint kernel reproduces FD d(neff)/d(scale) to ~1e-4 (global scaling).

    Scaling every index by ``s`` makes ``depsilon/ds = 2 * epsilon`` at every
    pixel (using the actual smoothed permittivity), so the kernel and FD probe
    exactly the same perturbation.
    """
    modes = _solve(3.476)
    m = modes[0]
    cs = m.cs
    # depsilon/ds = 2 * eps_ii (eps_ii = n_i^2 of the smoothed cross-section)
    grad = mw.neff_gradient(
        m,
        2.0 * np.asarray(cs.nx) ** 2,
        2.0 * np.asarray(cs.ny) ** 2,
        2.0 * np.asarray(cs.nz) ** 2,
    )

    fd = mw.finite_difference_gradient(
        lambda t: _solve(3.476, scale=1.0 + t)[0].neff, step=1e-3
    )
    assert grad.real == pytest.approx(fd.real, rel=2e-3)
    assert abs(grad - fd) / abs(fd) < 2e-3


def test_neff_value_and_grad_jacobian() -> None:
    """neff_value_and_grad assembles the FD-consistent Jacobian from one solve."""
    n_core = 3.476

    def solve(params: np.ndarray) -> list[mw.Mode]:
        return _solve(float(params[0]))

    def eps_jacobian(
        params: np.ndarray, j: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # only one parameter (n_core): d eps / d n_core = 2 n_core in the core
        assert j == 0
        m = solve(params)[0]
        nx = np.real(np.asarray(m.cs.nx))
        core = (np.abs(nx - params[0]) < 0.4).astype(float)
        d = 2.0 * params[0] * core
        return d, d, d

    neffs, jac = mw.neff_value_and_grad(solve, np.array([n_core]), eps_jacobian)
    assert neffs.shape == (1,)
    assert jac.shape == (1, 1)

    fd = mw.finite_difference_gradient(
        lambda t: _solve(n_core + t)[0].neff, step=1e-3
    )
    # A hard core mask mis-handles subpixel-smoothed boundary pixels (eps there is
    # an area-weighted average, so d eps/d n_core != 2 n_core), giving ~5-10% error
    # at this resolution. The high-precision validation is the global-scaling test
    # above (no masking); here we only check the Jacobian is right to ~10%.
    assert jac[0, 0].real == pytest.approx(fd.real, rel=0.1)
