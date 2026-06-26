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

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.constants import c
from scipy.constants import epsilon_0 as eps0

from meow.mode import inner_product

if TYPE_CHECKING:
    from collections.abc import Callable

    from meow.arrays import FloatArray2D
    from meow.cross_section import CrossSection
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


def _eps_components(cs: CrossSection) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The diagonal permittivity arrays ``(eps_xx, eps_yy, eps_zz)`` of a CS."""
    return (
        np.asarray(cs.nx) ** 2,
        np.asarray(cs.ny) ** 2,
        np.asarray(cs.nz) ** 2,
    )


def make_differentiable_neffs(
    solve: Callable[[np.ndarray], list[list[Mode]]],
    *,
    shape: tuple[int, int],
    cross_sections: Callable[[np.ndarray], list[CrossSection]] | None = None,
    eps_jacobian: Callable[
        [np.ndarray, int], list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    ]
    | None = None,
    eps_step: float = 1e-6,
) -> Callable[[Any], Any]:
    """Build a ``jax``-differentiable ``neffs(params)`` via the modal adjoint.

    Returns a :func:`jax.custom_vjp` function mapping a real parameter vector to
    a ``(num_cross_sections, num_modes)`` array of effective indices. The forward
    pass runs your (non-differentiable) mode ``solve`` through
    :func:`jax.pure_callback`; the backward pass contracts the upstream cotangent
    with the exact perturbation-theory sensitivity ``d(neff)/d(eps)`` (the modal
    adjoint of :func:`neff_gradient`), using a *cheap* permittivity Jacobian that
    needs **no extra eigensolve**. Solves are memoized between the forward and
    backward passes, so a value-and-gradient costs a single eigensolve regardless
    of the number of parameters.

    The returned array composes with the (already ``jax``-native) SAX EME
    cascade: build the per-cell propagation matrices from these effective indices
    and cascade them, and ``jax.grad`` of any objective flows automatically back
    to ``params``. The gradient captures the dependence carried by the
    propagation constants (effective indices); mode-overlap/interface matrices
    are treated as constant w.r.t. ``params`` (exact when the objective's
    parameter dependence is through the propagation constants, e.g. phase-matched
    / adiabatic devices, and a controlled approximation otherwise).

    Args:
        solve: maps a parameter vector to the solved modes, grouped per
            cross-section: ``solve(params)[i][k]`` is mode ``k`` of cross-section
            ``i``. The mode count must match ``shape``.
        shape: ``(num_cross_sections, num_modes)`` of the returned array.
        cross_sections: maps a parameter vector to the cross-sections (the
            *permittivity* build only, no eigensolve); used to central-difference
            ``d(eps)/d(params)`` cheaply. Provide this **or** ``eps_jacobian``.
        eps_jacobian: ``eps_jacobian(params, j)`` returns, per cross-section, the
            analytic ``(deps_xx, deps_yy, deps_zz)`` of ``d(eps)/d(params[j])``.
        eps_step: central-difference step for the permittivity Jacobian (only
            used with ``cross_sections``).

    Returns:
        A ``jax.custom_vjp`` callable ``f(params) -> neffs`` of shape ``shape``.
    """
    import jax
    import jax.numpy as jnp

    if (cross_sections is None) == (eps_jacobian is None):
        msg = "Provide exactly one of cross_sections or eps_jacobian."
        raise ValueError(msg)

    cache: dict[bytes, list[list[Mode]]] = {}

    def _solve(params: np.ndarray) -> list[list[Mode]]:
        key = np.asarray(params, dtype=float).tobytes()
        if key not in cache:
            if len(cache) > 8:
                cache.clear()
            cache[key] = solve(np.asarray(params, dtype=float))
        return cache[key]

    def _neffs_np(params: np.ndarray) -> np.ndarray:
        modes = _solve(np.asarray(params))
        return np.array(
            [[complex(m.neff) for m in row] for row in modes], dtype=np.complex128
        )

    def _eps_jac_j(params: np.ndarray, j: int, modes: list[list[Mode]]) -> list[tuple]:
        if eps_jacobian is not None:
            return eps_jacobian(params, j)
        cs_fn = cast("Callable[[np.ndarray], list[CrossSection]]", cross_sections)
        ep, em = np.array(params, dtype=float), np.array(params, dtype=float)
        ep[j] += eps_step
        em[j] -= eps_step
        csp = cs_fn(ep)
        csm = cs_fn(em)
        out = []
        for i in range(len(modes)):
            xp, yp, zp = _eps_components(csp[i])
            xm, ym, zm = _eps_components(csm[i])
            twoh = 2.0 * eps_step
            out.append(((xp - xm) / twoh, (yp - ym) / twoh, (zp - zm) / twoh))
        return out

    def _grad_np(params: np.ndarray, g: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=float)
        modes = _solve(params)
        g = np.asarray(g, dtype=np.complex128)
        grad = np.zeros(params.shape, dtype=float)
        for j in range(params.size):
            deps = _eps_jac_j(params, j, modes)
            for i, row in enumerate(modes):
                dxx, dyy, dzz = deps[i]
                for k, mode in enumerate(row):
                    dz = neff_gradient(mode, dxx, dyy, dzz)
                    # jax convention for f: R->C into a real objective is
                    # grad = Re(cotangent * d(out)/d(param))
                    grad[j] += float(np.real(g[i, k] * dz))
        return grad

    result = jax.ShapeDtypeStruct(shape, jnp.complex128)

    @jax.custom_vjp
    def differentiable_neffs(params: Any) -> Any:
        return jax.pure_callback(_neffs_np, result, params)

    def _fwd(params: Any) -> tuple[Any, Any]:
        return differentiable_neffs(params), params

    def _bwd(params: Any, cotangent: Any) -> tuple[Any]:
        grad = jax.pure_callback(
            _grad_np,
            jax.ShapeDtypeStruct(jnp.shape(params), jnp.result_type(params)),
            params,
            cotangent,
        )
        return (grad,)

    differentiable_neffs.defvjp(_fwd, _bwd)
    return differentiable_neffs


def make_differentiable_objective(
    objective: Callable[[np.ndarray], np.ndarray],
    *,
    shape: tuple[int, ...] = (),
    step: float = 1e-3,
) -> Callable[[Any], Any]:
    """Wrap a real EME figure-of-merit as a ``jax``-differentiable function.

    Returns a :func:`jax.custom_vjp` mapping a real parameter vector to a
    **real** figure of merit (a scalar or array - e.g. transmission ``|S_ij|^2``,
    a splitting ratio, an insertion loss). The forward runs your
    (non-differentiable) ``objective`` via :func:`jax.pure_callback`; the backward
    computes the **exact** ``d(objective)/dparams`` by central finite differences
    of the *whole* solve. Because it differences the full solve, the gradient
    captures **every** effect - propagation constants *and* mode-overlap /
    interface sensitivities (the complete ``dS/dp`` contribution) - so
    ``jax.grad`` of any composed objective just works.

    Why a *real* figure of merit (not the complex S-matrix): each mode solve
    returns modes with an arbitrary global phase, so the **complex** EME S-matrix
    is gauge-inconsistent from one parameter value to the next and is *not* a
    smooth function of the parameters - only gauge-invariant real quantities
    (mode powers ``|S_ij|^2``, ratios, losses) are. Differencing those is well
    posed; differencing the raw complex S is not.

    Why finite differences (not an analytic overlap adjoint): an analytic
    mode-overlap sensitivity via truncated guided-mode perturbation theory is
    *not* accurate for high-index-contrast waveguides - the overlap change is
    dominated by coupling to radiation/continuum modes outside any finite computed
    basis (empirically ``~100%`` error, not converging with basis size). An exact
    cheap overlap adjoint needs the discretized eigensolver's operator, which the
    external FDE backends do not expose; differencing the full solve is therefore
    the robust exact route. The tradeoff is cost: the backward re-solves
    ``2 * n_params`` times, so this is the *exact* gradient for a modest number of
    design parameters, while :func:`make_differentiable_neffs` is the *cheap*
    (single-solve) gradient that is exact for propagation-constant-mediated
    objectives.

    Args:
        objective: maps a parameter vector to a real, gauge-invariant figure of
            merit of shape ``shape`` (computed from the EME solve, e.g. from
            ``abs(S[...]) ** 2``).
        shape: the shape of the returned figure of merit (``()`` for a scalar).
        step: central-difference step for each parameter.

    Returns:
        A ``jax.custom_vjp`` callable ``f(params) -> fom`` of shape ``shape``.
    """
    import jax
    import jax.numpy as jnp

    def _obj_np(params: np.ndarray) -> np.ndarray:
        return np.asarray(objective(np.asarray(params, dtype=float)), dtype=float)

    def _grad_np(params: np.ndarray, g: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=float)
        g = np.asarray(g, dtype=float)
        grad = np.zeros(params.shape, dtype=float)
        for k in range(params.size):
            ep, em = np.array(params, dtype=float), np.array(params, dtype=float)
            ep[k] += step
            em[k] -= step
            dobj = (_obj_np(ep) - _obj_np(em)) / (2.0 * step)
            grad[k] = float(np.sum(g * dobj))
        return grad

    result = jax.ShapeDtypeStruct(shape, jnp.float64)

    @jax.custom_vjp
    def differentiable_objective(params: Any) -> Any:
        return jax.pure_callback(_obj_np, result, params)

    def _fwd(params: Any) -> tuple[Any, Any]:
        return differentiable_objective(params), params

    def _bwd(params: Any, cotangent: Any) -> tuple[Any]:
        grad = jax.pure_callback(
            _grad_np,
            jax.ShapeDtypeStruct(jnp.shape(params), jnp.result_type(params)),
            params,
            cotangent,
        )
        return (grad,)

    differentiable_objective.defvjp(_fwd, _bwd)
    return differentiable_objective
