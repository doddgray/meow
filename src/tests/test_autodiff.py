"""End-to-end jax.custom_vjp autodiff of EME objectives via the modal adjoint.

Validates that ``jax.grad`` of a real objective built (in jax) from the
differentiable effective indices matches finite differences of the same
objective re-solved from scratch - i.e. the gradient flows correctly through the
non-differentiable mode solve via the perturbation adjoint.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import meow as mw

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

WL = 1.55


def _cross_section(n_core: float) -> mw.CrossSection:
    core = mw.Structure(
        material=mw.IndexMaterial(name="core", n=float(n_core)),
        geometry=mw.Box(
            x_min=-0.25, x_max=0.25, y_min=0.0, y_max=0.22, z_min=-1.0, z_max=2.0
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
    mesh = mw.Mesh2D(x=np.linspace(-1.5, 1.5, 121), y=np.linspace(-1.0, 1.0, 81))
    cells = mw.create_cells([core, clad], mesh, [1.0], z_min=0.0)
    return mw.CrossSection.from_cell(cell=cells[0], env=mw.Environment(wl=WL, T=25.0))


def _solve(params: np.ndarray) -> list[list[mw.Mode]]:
    return [mw.compute_modes(_cross_section(float(params[0])), num_modes=1)]


def _cross_sections(params: np.ndarray) -> list[mw.CrossSection]:
    return [_cross_section(float(params[0]))]


def _neff_real(n_core: float) -> float:
    return float(np.real(_solve(np.array([n_core]))[0][0].neff))


def test_grad_through_phase_objective_matches_fd() -> None:
    """jax.grad of an interferometer-style phase objective matches finite diff.

    The objective is the transmitted power of a length L of single-mode guide,
    ``|exp(2j*pi*neff*L/wl)|`` weighted into a real phase functional - exactly
    the kind of propagation-constant dependence the adjoint captures.
    """
    f = mw.make_differentiable_neffs(
        _solve, shape=(1, 1), cross_sections=_cross_sections
    )
    length = 8.0

    def objective(params: jnp.ndarray) -> jnp.ndarray:
        neff = f(params)[0, 0]
        phase = 2.0 * jnp.pi * neff * length / WL
        return jnp.real(jnp.cos(phase)) ** 2  # a real phase functional

    p0 = jnp.array([3.476])
    val, grad = jax.value_and_grad(objective)(p0)

    h = 1e-4

    def obj_np(n: float) -> float:
        phase = 2.0 * np.pi * _neff_real(n) * length / WL
        return float(np.cos(phase) ** 2)

    fd = (obj_np(3.476 + h) - obj_np(3.476 - h)) / (2 * h)
    assert float(val) == pytest.approx(obj_np(3.476), abs=1e-6)
    assert float(grad[0]) == pytest.approx(fd, rel=2e-3)


def test_single_solve_value_and_grad_caches() -> None:
    """value_and_grad triggers a single eigensolve (fwd/bwd share the cache)."""
    calls = {"n": 0}

    def counting_solve(params: np.ndarray) -> list[list[mw.Mode]]:
        calls["n"] += 1
        return _solve(params)

    f = mw.make_differentiable_neffs(
        counting_solve, shape=(1, 1), cross_sections=_cross_sections
    )

    def objective(params: jnp.ndarray) -> jnp.ndarray:
        return jnp.real(f(params)[0, 0]) ** 2

    _ = jax.value_and_grad(objective)(jnp.array([3.476]))
    # one eigensolve for the value+gradient (the cross_sections FD does not solve)
    assert calls["n"] == 1


def test_analytic_eps_jacobian_matches_fd() -> None:
    """An analytic eps Jacobian (d eps/d n_core) gives the same gradient as FD."""
    n0 = 3.476

    def eps_jac(params: np.ndarray, j: int) -> list[tuple]:
        assert j == 0
        cs = _cross_section(float(params[0]))
        nx = np.real(np.asarray(cs.nx))
        core = (np.abs(nx - params[0]) < 0.4).astype(float)
        d = 2.0 * params[0] * core
        return [(d, d, d)]

    f = mw.make_differentiable_neffs(_solve, shape=(1, 1), eps_jacobian=eps_jac)

    def objective(params: jnp.ndarray) -> jnp.ndarray:
        return jnp.real(f(params)[0, 0]) ** 2

    grad = float(jax.grad(objective)(jnp.array([n0]))[0])
    h = 1e-4
    fd = (_neff_real(n0 + h) ** 2 - _neff_real(n0 - h) ** 2) / (2 * h)
    # hard-mask analytic eps Jacobian carries boundary-smoothing error (~10%)
    assert grad == pytest.approx(fd, rel=0.1)
