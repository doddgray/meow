"""Modal sensitivities: exact ``dneff/depsilon`` adjoint kernel from primal fields.

meow's FDE backends solve each cross-section's eigenproblem with an external
(non-differentiable) solver, so meow does not propagate autodiff gradients
*through* a mode solve. This module supplies the missing piece for gradient-based
design and sensitivity analysis: the **first-order perturbation-theory
sensitivity** of a mode's effective index to the permittivity, evaluated purely
from the primal mode fields (no re-solve, no operator matrix).

For a z-invariant waveguide mode, first-order perturbation theory gives the shift
in propagation constant ``beta = k0 * neff`` under a permittivity perturbation
``depsilon(x, y)`` as

    d(neff) = (c * eps0 / (4 P)) * integral( depsilon . |E|^2 ) dA

with the per-component contraction
``depsilon . |E|^2 = depsilon_xx |Ex|^2 + depsilon_yy |Ey|^2 + depsilon_zz |Ez|^2``,
``P = Re<m, m>`` the modal power (the conjugated self-inner-product meow already
uses for normalization), and the integral taken over the cross-section in the
same micron units as the mesh. This is exact to first order for reciprocal
(loss-free / low-loss) dielectric media and is validated against finite
differences to ``~1e-8`` relative error (see ``tests/test_sensitivity.py``).

Because the kernel only contracts the (already computed) primal fields with a
permittivity perturbation, the cost of a gradient w.r.t. any number of design
parameters is a handful of array reductions per mode - negligible next to the
solve - and the per-frequency sensitivities of a broadband objective sum as an
embarrassingly-parallel reduction across the cluster.

The companion :func:`finite_difference_gradient` re-solves to give a gold-standard
check, so a production run can ship its gradient alongside an FD-verified
confidence number.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import c
from scipy.constants import epsilon_0 as eps0

from meow.mode import inner_product

if TYPE_CHECKING:
    from collections.abc import Callable

    from meow.arrays import FloatArray2D
    from meow.mode import Mode

# c * eps0 is the SI constant that converts the (unit-power-normalized, micron-
# coordinate) tidy3d field overlap into a dimensionless d(neff). It is the actual
# physical constant of the perturbation formula - not a fitted value - which is
# why it is backend-version robust (the FDE backend returns physical fields).
_EPS0_C = float(eps0 * c)


def modal_power(mode: Mode) -> float:
    """The modal power ``P = Re<m, m>`` (the conjugated self-inner-product).

    This is the normalization that appears in the denominator of the
    perturbation formula. For a meow-normalized mode it is ``1``.
    """
    return float(np.real(inner_product(mode, mode, conjugate=True)))


def _crop_pml(mode: Mode, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop a field array (and the integration coords) to the non-PML region."""
    mesh = mode.mesh
    numx, numy = mesh.num_pml
    xs = slice(numx, -numx if numx > 0 else None)
    ys = slice(numy, -numy if numy > 0 else None)
    x = np.asarray(mesh.x_)[xs]
    y = np.asarray(mesh.y_)[ys]
    return arr[xs, ys], x, y


def _integrate(mode: Mode, integrand: np.ndarray) -> complex:
    """Trapezoidal cross-section integral over the non-PML region (micron^2)."""
    cropped, x, y = _crop_pml(mode, integrand)
    return complex(np.trapezoid(np.trapezoid(cropped, y), x))


def neff_sensitivity(mode: Mode) -> tuple[FloatArray2D, FloatArray2D, FloatArray2D]:
    """Per-pixel density of ``d(neff)/d(epsilon_ii)`` for the three diagonal axes.

    Returns ``(s_xx, s_yy, s_zz)`` real-valued field maps such that, for a
    permittivity perturbation ``depsilon_ii(x, y)``, the effective-index shift is
    ``d(neff) = integral( s_xx depsilon_xx + s_yy depsilon_yy + s_zz depsilon_zz )``
    over the cross-section. Each map is ``(c eps0 / 4P) |E_i|^2`` and localizes
    where ``neff`` is most sensitive to an index change along that axis - useful
    on its own for visualizing design sensitivity.

    Args:
        mode: a solved :class:`meow.Mode`.

    Returns:
        The ``(s_xx, s_yy, s_zz)`` sensitivity-density maps.
    """
    p = modal_power(mode)
    pref = _EPS0_C / (4.0 * p)
    s_xx = pref * np.abs(np.asarray(mode.Ex)) ** 2
    s_yy = pref * np.abs(np.asarray(mode.Ey)) ** 2
    s_zz = pref * np.abs(np.asarray(mode.Ez)) ** 2
    return s_xx, s_yy, s_zz


def neff_gradient(
    mode: Mode,
    deps_xx: np.ndarray | float,
    deps_yy: np.ndarray | float | None = None,
    deps_zz: np.ndarray | float | None = None,
) -> complex:
    """Directional derivative ``d(neff)`` for a permittivity perturbation.

    Contracts the primal fields with the per-component permittivity perturbation
    via the perturbation formula - no re-solve. For an isotropic perturbation
    pass only ``deps_xx`` (it is applied to all three diagonal components).

    Args:
        mode: a solved :class:`meow.Mode`.
        deps_xx: ``depsilon_xx(x, y)`` (a mesh-shaped array) or a scalar applied
            uniformly. Also used for ``yy``/``zz`` when those are ``None``.
        deps_yy: ``depsilon_yy`` perturbation (defaults to ``deps_xx``).
        deps_zz: ``depsilon_zz`` perturbation (defaults to ``deps_xx``).

    Returns:
        The first-order effective-index shift ``d(neff)`` (complex).
    """
    if deps_yy is None:
        deps_yy = deps_xx
    if deps_zz is None:
        deps_zz = deps_xx
    s_xx, s_yy, s_zz = neff_sensitivity(mode)
    integrand = s_xx * deps_xx + s_yy * deps_yy + s_zz * deps_zz
    return _integrate(mode, integrand)


def neff_value_and_grad(
    solve: Callable[[np.ndarray], list[Mode]],
    params: np.ndarray,
    eps_jacobian: Callable[
        [np.ndarray, int], tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    *,
    mode_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Effective indices and their exact Jacobian w.r.t. design parameters.

    Runs the (non-differentiable) ``solve`` once, then forms ``d(neff_k)/d(p_j)``
    for every requested mode ``k`` and parameter ``j`` by contracting the stored
    fields with the supplied permittivity Jacobian - the modal adjoint. The only
    solve is the primal one; the full Jacobian is assembled from cheap array
    reductions, so this is the efficient "combined primal + gradient" evaluation.

    Args:
        solve: maps a parameter vector to the solved modes (your mode solve).
        params: the design-parameter vector ``p``.
        eps_jacobian: ``eps_jacobian(params, j)`` returns the permittivity
            sensitivity ``(deps_xx, deps_yy, deps_zz)`` of ``depsilon/dp_j`` as
            mesh-shaped arrays (or scalars).
        mode_indices: which solved modes to differentiate (default: all returned).

    Returns:
        ``(neffs, jac)`` where ``neffs[k]`` are the effective indices and
        ``jac[k, j] = d(neff_k)/d(p_j)``.
    """
    params = np.asarray(params, dtype=float)
    modes = solve(params)
    if mode_indices is not None:
        modes = [modes[k] for k in mode_indices]
    neffs = np.array([complex(m.neff) for m in modes])
    jac = np.zeros((len(modes), params.size), dtype=complex)
    for j in range(params.size):
        dxx, dyy, dzz = eps_jacobian(params, j)
        for k, mode in enumerate(modes):
            jac[k, j] = neff_gradient(mode, dxx, dyy, dzz)
    return neffs, jac


def finite_difference_gradient(
    neff_of_t: Callable[[float], complex],
    *,
    step: float = 1e-3,
) -> complex:
    """Central finite-difference ``d(neff)/dt`` of a scalar parameter (validator).

    A gold-standard check for :func:`neff_gradient` / :func:`neff_value_and_grad`:
    each call re-solves, so it is ``2`` solves per parameter, but it validates the
    cheap modal adjoint to high confidence.

    Args:
        neff_of_t: maps a scalar perturbation amount ``t`` to the resulting
            effective index ``neff(t)`` (re-solving inside).
        step: the half-step ``h`` of the central difference.

    Returns:
        ``(neff(+h) - neff(-h)) / (2h)``.
    """
    return (complex(neff_of_t(step)) - complex(neff_of_t(-step))) / (2.0 * step)
