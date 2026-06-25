"""Differentiable level-set / density -> permittivity path for inverse design.

Validates the end-to-end chain ``param -> density -> eps -> modes -> neff`` with
analytic gradients: jax.grad of a neff objective w.r.t. a geometric density
parameter (a soft waveguide half-width) matches finite differences.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import meow as mw
from meow import levelset

DensityFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

WL = 1.55
N_CORE = 3.45
N_CLAD = 1.444
BETA = 40.0  # soft-boundary steepness [1/um]
Y0, HALF_H = 0.11, 0.11  # waveguide vertical centre / half-height [um]


def _mesh() -> mw.Mesh2D:
    return mw.Mesh2D(x=np.linspace(-1.5, 1.5, 121), y=np.linspace(-1.0, 1.0, 81))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _density(half_w: float) -> DensityFn:
    """A soft rectangle of half-width ``half_w`` (smooth in geometry)."""

    def rho(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        sx = _sigmoid(BETA * (half_w - np.abs(x)))
        sy = _sigmoid(BETA * (HALF_H - np.abs(y - Y0)))
        return sx * sy

    return rho


def _ddensity_dw(half_w: float) -> DensityFn:
    """d rho / d(half_w): only the x-sigmoid depends on the width."""

    def drho(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        sx = _sigmoid(BETA * (half_w - np.abs(x)))
        sy = _sigmoid(BETA * (HALF_H - np.abs(y - Y0)))
        return BETA * sx * (1.0 - sx) * sy

    return drho


def _solve(params: np.ndarray) -> list[list[mw.Mode]]:
    cs = levelset.density_cross_section(
        _mesh(), mw.Environment(wl=WL), _density(float(params[0])),
        n_min=N_CLAD, n_max=N_CORE,
    )
    return [mw.compute_modes(cs, num_modes=1)]


def _neff_real(half_w: float) -> float:
    return float(np.real(_solve(np.array([half_w]))[0][0].neff))


def test_density_cross_section_solves() -> None:
    """A density field yields a solvable, physically sensible mode."""
    n = _neff_real(0.25)
    assert N_CLAD < n < N_CORE


def test_tanh_projection_grad_matches_fd() -> None:
    """The analytic projection derivative matches finite differences."""
    rho = np.linspace(0.0, 1.0, 11)
    g = levelset.tanh_projection_grad(rho, beta=8.0, eta=0.5)
    h = 1e-6
    fd = (
        levelset.tanh_projection(rho + h, 8.0) - levelset.tanh_projection(rho - h, 8.0)
    ) / (2 * h)
    np.testing.assert_allclose(g, fd, rtol=1e-5)


def test_autodiff_through_density_matches_fd() -> None:
    """jax.grad of a neff objective w.r.t. the density width matches FD."""

    def eps_jac(params: np.ndarray, j: int) -> list[tuple]:
        assert j == 0
        dxx, dyy, dzz = levelset.eps_jacobian_components(
            _mesh(), _ddensity_dw(float(params[0])), N_CLAD, N_CORE
        )
        return [(dxx, dyy, dzz)]

    f = mw.make_differentiable_neffs(_solve, shape=(1, 1), eps_jacobian=eps_jac)

    def objective(params: jnp.ndarray) -> jnp.ndarray:
        return jnp.real(f(params)[0, 0])

    w0 = 0.25
    grad = float(jax.grad(objective)(jnp.array([w0]))[0])
    h = 2e-3
    fd = (_neff_real(w0 + h) - _neff_real(w0 - h)) / (2 * h)
    # widening the guide raises neff -> positive, and matches FD closely
    assert grad > 0
    assert grad == pytest.approx(fd, rel=5e-2)
