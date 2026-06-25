"""Sparse shift-invert scalar mode solver (meow.fde.sparse).

Validated against the trusted tidy3d vectorial backend in the low-contrast
(scalar) regime, where the scalar Helmholtz approximation is accurate.
"""

from __future__ import annotations

import numpy as np
import pytest

import meow as mw
from meow.fde import sparse

N_CORE, N_CLAD = 1.50, 1.45


def _low_contrast_cs() -> mw.CrossSection:
    core = mw.Structure(
        material=mw.IndexMaterial(name="core", n=N_CORE),
        geometry=mw.Box(
            x_min=-1.0, x_max=1.0, y_min=0.0, y_max=0.7, z_min=-1.0, z_max=2.0
        ),
        mesh_order=5,
    )
    clad = mw.Structure(
        material=mw.IndexMaterial(name="clad", n=N_CLAD),
        geometry=mw.Box(
            x_min=-6.0, x_max=6.0, y_min=-5.0, y_max=5.0, z_min=-1.0, z_max=2.0
        ),
        mesh_order=10,
    )
    mesh = mw.Mesh2D(x=np.linspace(-6.0, 6.0, 201), y=np.linspace(-5.0, 5.0, 161))
    cell = mw.create_cells([core, clad], mesh, [1.0], z_min=0.0)[0]
    return mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=1.55))


def test_scalar_matches_tidy3d_low_contrast() -> None:
    """Sparse scalar neff matches tidy3d's vectorial neff for low contrast."""
    cs = _low_contrast_cs()
    n_vec = float(np.real(mw.compute_modes(cs, num_modes=1)[0].neff))
    n_sca = float(sparse.scalar_neffs(cs, num_modes=1)[0])
    assert N_CLAD < n_sca < N_CORE  # guided
    assert n_sca == pytest.approx(n_vec, abs=2e-3)


def test_shift_invert_returns_descending_guided_modes() -> None:
    """Requesting several modes returns guided neffs in descending order."""
    cs = _low_contrast_cs()
    nz = np.real(np.asarray(cs.nz))  # (nx, ny) on the Ez node grid
    x = np.asarray(cs.mesh.Xz)[:, 0]
    y = np.asarray(cs.mesh.Yz)[0, :]
    neffs, fields = sparse.solve_scalar_modes(nz.T, x, y, 1.55, num_modes=3)
    assert neffs.shape == (3,)
    assert fields.shape == (3, y.size, x.size)
    assert np.all(np.diff(neffs) <= 1e-9)  # descending
    assert neffs[0] < N_CORE


def test_solve_scalar_modes_rejects_bad_shape() -> None:
    """A mismatched index/grid shape is a clear error."""
    with pytest.raises(ValueError, match="shape"):
        sparse.solve_scalar_modes(
            np.ones((4, 5)), np.arange(4), np.arange(4), 1.55
        )
