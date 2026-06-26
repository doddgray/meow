"""Deflated-linear-solve eigenvector adjoint (meow.fde.sparse).

Validates the exact eigenvector/eigenvalue sensitivities from the bordered solve
against finite differences, on the same gauge-invariant overlap-power harness
(``G_i = sum_j |<A_i|B_j>|^2``) that exposed the truncated-modal failure - and
shows the adjoint is orders of magnitude more accurate than the truncated
guided-mode perturbation prediction.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigsh

from meow.fde import sparse

WL = 1.55
N_CORE, N_CLAD, H = 3.0, 1.444, 0.22
NM = 4
NX, NY = 101, 61
X_, Y_ = np.linspace(-1.5, 1.5, NX), np.linspace(-1.0, 1.0, NY)
XX, YY = np.meshgrid(X_, Y_)  # (NY, NX)


def _n_a(s: float, w: float = 0.9) -> np.ndarray:
    core = (np.abs(XX) < w / 2) & (YY > 0) & (YY < H)
    n = np.where(core, N_CORE, N_CLAD)
    return np.where(core & (XX > 0), N_CORE + s, n)  # right-half bump


def _n_b(w: float = 0.6) -> np.ndarray:
    core = (np.abs(XX) < w / 2) & (YY > 0) & (YY < H)
    return np.where(core, N_CORE, N_CLAD)


def _solve(n: np.ndarray) -> tuple:
    a_op, k0 = sparse.scalar_operator(n, X_, Y_, WL)
    vals, vecs = eigsh(a_op, k=NM, sigma=(k0 * float(n.max())) ** 2, which="LM")
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order], a_op, k0


def _deps() -> np.ndarray:
    bump = (np.abs(XX) < 0.45) & (YY > 0) & (YY < H) & (XX > 0)
    return (2.0 * N_CORE * bump).ravel()


def test_eigenvalue_sensitivity_matches_fd() -> None:
    """Hellmann-Feynman dlambda/ds matches a finite difference of the eigenvalue."""
    vals, vecs, a_op, k0 = _solve(_n_a(0.0))
    deps = _deps()
    h = 1e-3
    vp, _, _, _ = _solve(_n_a(h))
    vm, _, _, _ = _solve(_n_a(-h))
    for i in range(NM):
        _, dlam = sparse.eigenvector_sensitivity(a_op, vals[i], vecs[:, i], deps, k0)
        fd = (vp[i] - vm[i]) / (2 * h)
        np.testing.assert_allclose(dlam, fd, rtol=1e-3)


def test_eigenvector_sensitivity_matches_fd() -> None:
    """dv/ds from the deflated solve matches a (gauge-aligned) finite difference."""
    vals, vecs, a_op, k0 = _solve(_n_a(0.0))
    deps = _deps()
    h = 1e-3
    _, vp, _, _ = _solve(_n_a(h))
    _, vm, _, _ = _solve(_n_a(-h))
    i = 0  # well-separated fundamental
    v0 = vecs[:, i]
    dv_adj, _ = sparse.eigenvector_sensitivity(a_op, vals[i], v0, deps, k0)
    # align FD signs to v0, then central difference and project out the v0 gauge
    vp_i = vp[:, i] * np.sign(vp[:, i] @ v0)
    vm_i = vm[:, i] * np.sign(vm[:, i] @ v0)
    dv_fd = (vp_i - vm_i) / (2 * h)
    dv_fd -= (v0 @ dv_fd) * v0
    assert np.linalg.norm(dv_adj - dv_fd) / np.linalg.norm(dv_fd) < 1e-3


def test_overlap_power_sensitivity_beats_truncated() -> None:
    """dG/ds: the adjoint matches FD; the truncated modal prediction does not."""
    vals, vecs, a_op, k0 = _solve(_n_a(0.0))
    _, wvecs, _, _ = _solve(_n_b())
    deps = _deps()
    omat = vecs.T @ wvecs

    h = 1e-3
    g_p = np.sum((_solve(_n_a(h))[1].T @ wvecs) ** 2, axis=1)
    g_m = np.sum((_solve(_n_a(-h))[1].T @ wvecs) ** 2, axis=1)
    dG_fd = (g_p - g_m) / (2 * h)

    dG_adj = np.zeros(NM)
    dG_trunc = np.zeros(NM)
    for i in range(NM):
        dv, _ = sparse.eigenvector_sensitivity(a_op, vals[i], vecs[:, i], deps, k0)
        dG_adj[i] = float(np.sum(2 * omat[i, :] * (dv @ wvecs)))
        dv_t = np.zeros(vecs.shape[0])
        for m in range(NM):
            if m == i:
                continue
            c = (k0**2) * (vecs[:, m] @ (deps * vecs[:, i])) / (vals[i] - vals[m])
            dv_t += c * vecs[:, m]
        dG_trunc[i] = float(np.sum(2 * omat[i, :] * (dv_t @ wvecs)))

    mask = np.abs(dG_fd) > 1e-4
    adj_err = np.max(np.abs(dG_adj[mask] - dG_fd[mask]) / np.abs(dG_fd[mask]))
    trunc_err = np.max(np.abs(dG_trunc[mask] - dG_fd[mask]) / np.abs(dG_fd[mask]))
    assert adj_err < 1e-2  # exact up to the FD step
    assert trunc_err > 0.3  # truncated basis misses the continuum coupling
    assert adj_err < 0.05 * trunc_err  # orders of magnitude better
