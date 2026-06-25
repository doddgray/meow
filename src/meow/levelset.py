"""Differentiable level-set / density helpers for gradient-based design.

The autodiff in :mod:`meow.sensitivity` turns design parameters into effective
indices with exact gradients - but it needs a *smooth, differentiable* map from
the parameters to the permittivity. A staircased polygon boundary is not
differentiable in geometry; a **density / level-set** representation is. This
module provides that map:

1. a density field ``rho(x, y)`` in ``[0, 1]`` (1 = core material, 0 = cladding),
   optionally sharpened with the standard :func:`tanh_projection` threshold;
2. a linear *permittivity* interpolation ``eps = eps_min + rho (eps_max -
   eps_min)`` (:func:`density_to_eps`) whose derivative ``d eps/d rho`` is the
   constant ``eps_max - eps_min`` - exactly what the modal adjoint needs;
3. helpers to sample a density on meow's ``Ex``/``Ey``/``Ez`` Yee grids and build
   a solvable :class:`meow.CrossSection` from it
   (:func:`density_cross_section`), plus the analytic permittivity Jacobian
   (:func:`eps_jacobian_components`) to feed
   :func:`meow.make_differentiable_neffs`.

Together with the autodiff plumbing this gives an end-to-end differentiable
inverse-design path: ``params -> density -> eps -> modes -> S-matrix ->
objective``, all with analytic gradients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from meow.arrays import ComplexArray2D, FloatArray2D
    from meow.cross_section import CrossSection
    from meow.environment import Environment
    from meow.mesh import Mesh2D

    DensityFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def tanh_projection(
    rho: np.ndarray, beta: float, eta: float = 0.5
) -> np.ndarray:
    """Smooth threshold projection of a density (Wang/Sigmund 2011).

    Sharpens ``rho`` toward 0/1 around the threshold ``eta`` with steepness
    ``beta`` while staying differentiable (``beta -> inf`` is a hard step). Use
    it to control the grayscale of a level-set before mapping to permittivity.

    Args:
        rho: density in ``[0, 1]``.
        beta: projection steepness (``0`` leaves ``rho`` unchanged).
        eta: threshold in ``[0, 1]`` (default ``0.5``).

    Returns:
        The projected density in ``[0, 1]``.
    """
    if beta <= 0:
        return np.asarray(rho, dtype=float)
    num = np.tanh(beta * eta) + np.tanh(beta * (np.asarray(rho, float) - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return num / den


def tanh_projection_grad(
    rho: np.ndarray, beta: float, eta: float = 0.5
) -> np.ndarray:
    """Derivative ``d/d rho`` of :func:`tanh_projection` (for analytic chains)."""
    if beta <= 0:
        return np.ones_like(np.asarray(rho, dtype=float))
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return beta * (1.0 - np.tanh(beta * (np.asarray(rho, float) - eta)) ** 2) / den


def density_to_eps(
    rho: np.ndarray, eps_min: complex, eps_max: complex
) -> np.ndarray:
    """Linear permittivity interpolation ``eps = eps_min + rho (eps_max-eps_min)``.

    The derivative ``d eps/d rho = eps_max - eps_min`` is constant, which keeps
    the modal adjoint exact. Interpolating *permittivity* (not index) is the
    standard SIMP-like material interpolation for electromagnetic inverse design.
    """
    return eps_min + np.asarray(rho, dtype=float) * (eps_max - eps_min)


def density_to_index(
    rho: np.ndarray, n_min: float, n_max: float
) -> np.ndarray:
    """Refractive index from a density via permittivity interpolation."""
    return np.sqrt(density_to_eps(rho, n_min**2, n_max**2))


def sample_components(
    mesh: Mesh2D, density_fn: DensityFn
) -> tuple[FloatArray2D, FloatArray2D, FloatArray2D]:
    """Sample a density function on the ``Ex``/``Ey``/``Ez`` Yee positions."""
    return (
        np.asarray(density_fn(np.asarray(mesh.Xx), np.asarray(mesh.Yx)), dtype=float),
        np.asarray(density_fn(np.asarray(mesh.Xy), np.asarray(mesh.Yy)), dtype=float),
        np.asarray(density_fn(np.asarray(mesh.Xz), np.asarray(mesh.Yz)), dtype=float),
    )


def index_arrays(
    mesh: Mesh2D, density_fn: DensityFn, n_min: float, n_max: float
) -> tuple[ComplexArray2D, ComplexArray2D, ComplexArray2D]:
    """The ``(nx, ny, nz)`` index arrays for a density field on a mesh."""
    rx, ry, rz = sample_components(mesh, density_fn)
    return (
        density_to_index(rx, n_min, n_max).astype(np.complex128),
        density_to_index(ry, n_min, n_max).astype(np.complex128),
        density_to_index(rz, n_min, n_max).astype(np.complex128),
    )


def density_cross_section(
    mesh: Mesh2D,
    env: Environment,
    density_fn: DensityFn,
    *,
    n_min: float,
    n_max: float,
) -> CrossSection:
    """Build a solvable :class:`meow.CrossSection` from a density field.

    Args:
        mesh: the mesh to sample on.
        env: the simulation environment.
        density_fn: ``density_fn(X, Y) -> rho`` in ``[0, 1]`` (vectorized over
            the mesh coordinate arrays).
        n_min: cladding refractive index (``rho = 0``).
        n_max: core refractive index (``rho = 1``).

    Returns:
        A :class:`meow.CrossSection` whose index is the interpolated field.
    """
    from meow.cross_section import CrossSection

    nx, ny, nz = index_arrays(mesh, density_fn, n_min, n_max)
    return CrossSection.from_index_arrays(mesh=mesh, env=env, nx=nx, ny=ny, nz=nz)


def eps_jacobian_components(
    mesh: Mesh2D,
    ddensity_fn: DensityFn,
    n_min: float,
    n_max: float,
) -> tuple[FloatArray2D, FloatArray2D, FloatArray2D]:
    """Per-component permittivity Jacobian ``d eps/d param`` from ``d rho/d param``.

    Because ``eps = eps_min + rho (eps_max - eps_min)``, the chain rule gives
    ``d eps/d param = (n_max^2 - n_min^2) * d rho/d param``. Sample the
    density's parameter-derivative on the Yee grids and scale - this is the
    analytic ``eps_jacobian`` to pass to
    :func:`meow.make_differentiable_neffs`.

    Args:
        mesh: the mesh to sample on.
        ddensity_fn: ``d rho/d param`` as a function ``(X, Y) -> array``.
        n_min: cladding index.
        n_max: core index.

    Returns:
        ``(deps_xx, deps_yy, deps_zz)`` on the ``Ex``/``Ey``/``Ez`` positions.
    """
    scale = n_max**2 - n_min**2
    drx, dry, drz = sample_components(mesh, ddensity_fn)
    return (scale * drx, scale * dry, scale * drz)
