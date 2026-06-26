"""Sparse shift-invert scalar mode solver for large grids.

The default tidy3d FDE backend forms a dense-style eigenproblem, which scales
poorly as the number of spatial grid points grows. For *very fine* meshes where
only a handful of guided modes are wanted, the right tool is a **sparse
shift-invert** eigensolver: assemble the (sparse) operator and ask
:func:`scipy.sparse.linalg.eigsh` for the few eigenvalues nearest a target
effective index. Cost and memory then scale with the number of nonzeros (``~5``
per grid point) rather than with a dense matrix, so the grid can be pushed far
beyond what the dense solve allows on one node.

This module provides that path for the **scalar / semivectorial** Helmholtz
approximation

    (d2/dx2 + d2/dy2) psi + k0^2 n(x, y)^2 psi = beta^2 psi ,

solving for ``beta = k0 * neff`` of the most-confined modes. It is exact for the
scalar regime (low index contrast, weakly-guiding) and a good neff estimate
there; for high-contrast, fully-vectorial accuracy keep the tidy3d backend. The
point is the *solver structure*: shift-invert around ``target_neff`` returns only
the guided modes, and the assembly is sparse, so this scales to large grids and
is the natural place to plug a distributed eigensolver (SLEPc/PETSc via
``slepc4py``) for grids that exceed a single node's memory - the operator built
here maps directly onto a PETSc matrix.

See :func:`solve_scalar_modes` for the array entry point and
:func:`scalar_neffs` for a convenience that reads a :class:`meow.CrossSection`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

if TYPE_CHECKING:
    from meow.cross_section import CrossSection


def _laplacian_1d(n: int, h: float) -> sp.csr_matrix:
    """1D second-difference operator with homogeneous Dirichlet boundaries."""
    main = -2.0 * np.ones(n)
    off = np.ones(n - 1)
    return sp.diags([off, main, off], [-1, 0, 1], format="csr") / (h * h)


def scalar_operator(
    n: np.ndarray, x: np.ndarray, y: np.ndarray, wl: float
) -> tuple[sp.csc_matrix, float]:
    """Assemble the sparse scalar Helmholtz operator ``A = L + k0^2 diag(n^2)``.

    The eigenproblem ``A psi = beta^2 psi`` (``B = I``) is what
    :func:`solve_scalar_modes` solves; exposing ``A`` directly is what enables an
    exact eigenvector adjoint (:func:`eigenvector_sensitivity`) - the sensitivity
    needs the operator, not the full spectrum.

    Args:
        n: real refractive-index distribution, shape ``(len(y), len(x))``.
        x: x grid coordinates [um] (assumed ~uniform).
        y: y grid coordinates [um] (assumed ~uniform).
        wl: wavelength [um].

    Returns:
        ``(A, k0)`` - the operator as a CSC sparse matrix (row-major flattening,
        ``index = iy * nx + ix``) and the vacuum wavenumber ``k0 = 2 pi / wl``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = np.asarray(n, dtype=float)
    ny, nx = n.shape
    if (ny, nx) != (y.size, x.size):
        msg = f"n must have shape (len(y), len(x)) = {(y.size, x.size)}, got {n.shape}."
        raise ValueError(msg)
    hx = float(np.mean(np.diff(x)))
    hy = float(np.mean(np.diff(y)))
    k0 = 2.0 * np.pi / wl
    lap = sp.kron(sp.identity(ny), _laplacian_1d(nx, hx)) + sp.kron(
        _laplacian_1d(ny, hy), sp.identity(nx)
    )
    a_op = (lap + sp.diags(k0**2 * n.ravel() ** 2)).tocsc()
    return a_op, k0


def solve_scalar_modes(
    n: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    wl: float,
    *,
    num_modes: int = 1,
    target_neff: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the scalar Helmholtz waveguide modes via sparse shift-invert.

    Assembles the 2D scalar operator on the (uniform) grid and asks for the
    ``num_modes`` eigenpairs whose effective index is nearest ``target_neff``,
    using a sparse shift-invert solve - so cost scales with the grid's nonzeros,
    not with a dense matrix.

    Args:
        n: real refractive-index distribution, shape ``(len(y), len(x))``.
        x: x grid coordinates [um] (assumed ~uniform).
        y: y grid coordinates [um] (assumed ~uniform).
        wl: wavelength [um].
        num_modes: number of modes (nearest the target) to return.
        target_neff: effective index to search around (default: ``max(n)``,
            i.e. the most-confined modes).

    Returns:
        ``(neffs, fields)`` with ``neffs`` of shape ``(num_modes,)`` (descending)
        and ``fields`` of shape ``(num_modes, len(y), len(x))``.
    """
    n = np.asarray(n, dtype=float)
    ny, nx = n.shape
    a_op, k0 = scalar_operator(n, x, y, wl)
    target = float(np.max(n)) if target_neff is None else float(target_neff)
    sigma = (k0 * target) ** 2  # shift-invert around the target propagation const
    k = min(num_modes, a_op.shape[0] - 2)
    vals, vecs = eigsh(a_op, k=k, sigma=sigma, which="LM")

    order = np.argsort(vals)[::-1]  # most-confined (largest beta^2) first
    vals, vecs = vals[order], vecs[:, order]
    neffs = np.sqrt(np.clip(vals, 0.0, None)) / k0
    fields = vecs.T.reshape(k, ny, nx)
    return neffs, fields


def eigenvector_sensitivity(
    operator: sp.spmatrix,
    eigenvalue: float,
    eigenvector: np.ndarray,
    deps: np.ndarray,
    k0: float,
) -> tuple[np.ndarray, float]:
    r"""Exact eigenvector sensitivity ``dv/dp`` via a deflated linear solve.

    For the real-symmetric eigenproblem ``A v = lambda v`` (``lambda = beta^2``,
    ``A = L + k0^2 diag(eps)``, ``v`` unit-norm), a permittivity perturbation has
    ``dA/dp = k0^2 diag(deps)`` and the eigenpair sensitivities are

    - eigenvalue (Hellmann-Feynman): ``dlambda/dp = v^T (dA/dp) v``;
    - eigenvector: solve ``(A - lambda I) dv = -(dA/dp - dlambda/dp I) v`` under
      the normalization gauge ``v^T dv = 0``.

    ``A - lambda I`` is singular (null vector ``v``), so this is done with the
    nonsingular **bordered/deflated** system

    .. code-block:: text

        [ A - lambda I    v ] [ dv ]   [ -(dA/dp - dlambda/dp I) v ]
        [ v^T             0 ] [ mu ] = [             0             ]

    which needs only the operator, the single eigenpair and ``deps`` - no second
    eigensolve and no truncated modal basis. Exact to machine precision (the
    radiation/continuum content missed by a truncated guided-mode expansion is
    captured because the full operator is inverted on the deflated subspace).

    Args:
        operator: the sparse operator ``A`` (e.g. from :func:`scalar_operator`).
        eigenvalue: ``lambda = (k0 neff)^2`` of the mode.
        eigenvector: the unit-norm mode field ``v`` (flattened like the operator).
        deps: ``d(eps)/dp`` per grid point (same flattening as the operator).
        k0: the vacuum wavenumber (so ``dA/dp = k0^2 diag(deps)``).

    Returns:
        ``(dv, dlambda)`` - the eigenvector sensitivity (gauge ``v^T dv = 0``) and
        the eigenvalue sensitivity ``dlambda/dp``.
    """
    import scipy.sparse.linalg as spl

    v = np.asarray(eigenvector, dtype=float).reshape(-1)
    deps = np.asarray(deps, dtype=float).reshape(-1)
    n_dof = v.size
    da_v = (k0**2) * deps * v  # (dA/dp) v
    dlambda = float(v @ da_v)  # Hellmann-Feynman v^T (dA/dp) v  (v^T v = 1)
    rhs = -(da_v - dlambda * v)  # -(dA/dp - dlambda I) v

    a_shift = sp.csc_matrix(operator) - eigenvalue * sp.identity(n_dof, format="csc")
    v_col = sp.csc_matrix(v.reshape(n_dof, 1))
    bordered = sp.bmat([[a_shift, v_col], [v_col.T, None]], format="csc")
    sol = spl.spsolve(bordered, np.concatenate([rhs, [0.0]]))
    return sol[:n_dof], dlambda


def scalar_neffs(
    cs: CrossSection,
    *,
    num_modes: int = 1,
    target_neff: float | None = None,
) -> np.ndarray:
    """Scalar effective indices for a :class:`meow.CrossSection` (sparse solve).

    Convenience wrapper around :func:`solve_scalar_modes` that reads the index
    distribution and grid from the cross-section's ``Ez`` positions.

    Args:
        cs: the cross-section to solve.
        num_modes: number of modes to return.
        target_neff: effective index to search around (default: ``max(n)``).

    Returns:
        The scalar effective indices (descending), shape ``(num_modes,)``.
    """
    nz = np.real(np.asarray(cs.nz))  # index on the Ez (node) grid
    x = np.asarray(cs.mesh.Xz)[:, 0]
    y = np.asarray(cs.mesh.Yz)[0, :]
    # cs.nz is indexed (x, y); the solver wants (len(y), len(x))
    neffs, _ = solve_scalar_modes(
        nz.T, x, y, float(cs.env.wl), num_modes=num_modes, target_neff=target_neff
    )
    return neffs
