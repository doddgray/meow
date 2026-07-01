"""End-to-end jax.custom_vjp autodiff via the exact sparse eigenpair adjoint.

Validates :func:`meow.make_differentiable_modes`: unlike
:func:`meow.make_differentiable_neffs` (neff only) and
:func:`meow.make_differentiable_objective` (finite differences of the whole
solve), this wraps :mod:`meow.fde.sparse`'s exact bordered/deflated eigenpair
adjoint, so it returns **both** the effective index and the mode field with a
backward pass built from a single eigensolve. Both channels - a neff-mediated
objective and a field/overlap-mediated objective built in plain ``jax.numpy`` -
are checked against finite differences, and the eigensolve call count is
checked to confirm the gradient is *not* paying for extra re-solves (the
distinguishing property vs :func:`meow.make_differentiable_objective`).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import meow as mw
from meow.fde import sparse

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

WL = 1.55
N_CORE, N_CLAD, H = 3.0, 1.444, 0.22
NX, NY = 81, 41
X_, Y_ = np.linspace(-1.2, 1.2, NX), np.linspace(-0.6, 0.8, NY)
XX, YY = np.meshgrid(X_, Y_)
SIGMA = 0.06  # smoothing scale (a couple pixels) - a genuinely differentiable
# geometry boundary, unlike a hard pixel mask (see meow.levelset module docs).
FIELD_SIZE = NX * NY


def _in_core_y() -> np.ndarray:
    return (YY > 0) & (YY < H)


def _rho(w: float) -> np.ndarray:
    core = 1.0 / (1.0 + np.exp(-(w / 2 - np.abs(XX)) / SIGMA))
    return np.where(_in_core_y(), core, 0.0)


def _drho_dw(w: float) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-(w / 2 - np.abs(XX)) / SIGMA))
    d = s * (1 - s) / (2 * SIGMA)
    return np.where(_in_core_y(), d, 0.0)


def _n_of(w: float) -> np.ndarray:
    rho = _rho(w)
    n2 = N_CLAD**2 + rho * (N_CORE**2 - N_CLAD**2)
    return np.sqrt(n2)


def _solve(params: np.ndarray) -> list[list[sparse.ScalarModeSolution]]:
    w = float(params[0])
    return [sparse.solve_scalar_modes_full(_n_of(w), X_, Y_, WL, num_modes=1)]


def _eps_jac(params: np.ndarray, j: int) -> list[np.ndarray]:
    assert j == 0
    w = float(params[0])
    return [(_drho_dw(w) * (N_CORE**2 - N_CLAD**2)).ravel()]


def _neff_np(w: float) -> float:
    return float(_solve(np.array([w]))[0][0].neff)


def _field_np(w: float) -> np.ndarray:
    return _solve(np.array([w]))[0][0].field


@pytest.fixture
def differentiable_modes() -> Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    return mw.make_differentiable_modes(
        _solve, shape=(1, 1), field_size=FIELD_SIZE, eps_jacobian=_eps_jac
    )


def test_neff_channel_grad_matches_fd(
    differentiable_modes: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
) -> None:
    """d(neff)/dw from the exact adjoint matches a finite difference."""
    f = differentiable_modes
    w0 = 0.5

    def objective(params: jnp.ndarray) -> jnp.ndarray:
        neffs, _fields = f(params)
        return neffs[0, 0]

    val, grad = jax.value_and_grad(objective)(jnp.array([w0]))
    assert float(val) == pytest.approx(_neff_np(w0), abs=1e-10)

    h = 1e-3
    fd = (_neff_np(w0 + h) - _neff_np(w0 - h)) / (2 * h)
    assert float(grad[0]) == pytest.approx(fd, rel=1e-4)


def test_field_overlap_channel_grad_matches_fd(
    differentiable_modes: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
) -> None:
    """d(overlap power)/dw - a *field*-mediated objective - matches FD.

    This is the capability make_differentiable_neffs does not have: the mode
    field itself carries a correct gradient, so an ordinary jnp expression of it
    (here mw.mode_overlap_power against a fixed reference field) differentiates
    exactly, with no analytic overlap-sensitivity formula required.
    """
    f = differentiable_modes
    w0 = 0.5
    ref_field = jnp.array(_field_np(0.9))

    def objective(params: jnp.ndarray) -> jnp.ndarray:
        _neffs, fields = f(params)
        return mw.mode_overlap_power(fields[0, 0], ref_field)

    val, grad = jax.value_and_grad(objective)(jnp.array([w0]))

    def obj_np(w: float) -> float:
        return float(np.sum(_field_np(w) * np.asarray(ref_field)) ** 2)

    assert float(val) == pytest.approx(obj_np(w0), abs=1e-8)
    h = 1e-3
    fd = (obj_np(w0 + h) - obj_np(w0 - h)) / (2 * h)
    assert float(grad[0]) == pytest.approx(fd, rel=1e-3)
    assert abs(float(grad[0])) > 1e-3  # a real, non-flat gradient


def test_single_solve_value_and_grad_caches() -> None:
    """value_and_grad triggers a single eigensolve (fwd/bwd share the cache).

    Unlike make_differentiable_objective (2 * n_params re-solves), the exact
    adjoint needs no extra eigensolve regardless of the objective or the number
    of parameters - only cheap linear solves against a pre-factorized operator.
    """
    calls = {"n": 0}

    def counting_solve(params: np.ndarray) -> list[list[sparse.ScalarModeSolution]]:
        calls["n"] += 1
        return _solve(params)

    f = mw.make_differentiable_modes(
        counting_solve, shape=(1, 1), field_size=FIELD_SIZE, eps_jacobian=_eps_jac
    )
    ref_field = jnp.array(_field_np(0.9))

    def objective(params: jnp.ndarray) -> jnp.ndarray:
        neffs, fields = f(params)
        return neffs[0, 0] + mw.mode_overlap_power(fields[0, 0], ref_field)

    _ = jax.value_and_grad(objective)(jnp.array([0.5]))
    assert calls["n"] == 1


def test_multi_param_gradient_reuses_one_factorization() -> None:
    """A multi-parameter Jacobian still costs a single eigensolve per point.

    (Two independent boundary positions of a symmetric slab, both feeding the
    same field-overlap objective.)
    """
    calls = {"n": 0}

    def n_of_two(params: np.ndarray) -> np.ndarray:
        wl, wr = float(params[0]), float(params[1])
        left = 1.0 / (1.0 + np.exp(-(wl - (-XX)) / SIGMA))  # left edge at x=-wl
        right = 1.0 / (1.0 + np.exp(-(wr - XX) / SIGMA))  # right edge at x=+wr
        rho = np.where(_in_core_y(), np.minimum(left, right), 0.0)
        n2 = N_CLAD**2 + rho * (N_CORE**2 - N_CLAD**2)
        return np.sqrt(n2)

    def solve2(params: np.ndarray) -> list[list[sparse.ScalarModeSolution]]:
        calls["n"] += 1
        return [
            sparse.solve_scalar_modes_full(
                n_of_two(params), X_, Y_, WL, num_modes=1
            )
        ]

    def eps_jac2(params: np.ndarray, j: int) -> list[np.ndarray]:
        h = 1e-5
        ep, em = np.array(params, dtype=float), np.array(params, dtype=float)
        ep[j] += h
        em[j] -= h
        np_, nm_ = n_of_two(ep), n_of_two(em)
        return [((np_**2 - nm_**2) / (2 * h)).ravel()]

    f = mw.make_differentiable_modes(
        solve2, shape=(1, 1), field_size=FIELD_SIZE, eps_jacobian=eps_jac2
    )

    def objective(params: jnp.ndarray) -> jnp.ndarray:
        neffs, _fields = f(params)
        return neffs[0, 0]

    p0 = jnp.array([0.4, 0.4])
    _val, grad = jax.value_and_grad(objective)(p0)
    assert calls["n"] == 1  # one solve for both the value and the 2-param grad

    h = 1e-3
    for j in range(2):
        pp, pm = np.array(p0), np.array(p0)
        pp[j] += h
        pm[j] -= h
        fd = (
            float(sparse.solve_scalar_modes_full(n_of_two(pp), X_, Y_, WL)[0].neff)
            - float(sparse.solve_scalar_modes_full(n_of_two(pm), X_, Y_, WL)[0].neff)
        ) / (2 * h)
        assert float(grad[j]) == pytest.approx(fd, rel=1e-3)
