"""Tests for the expanded (an)isotropic optical material interface."""

import importlib.util

import numpy as np
import pytest
from pydantic import ValidationError

import meow as mw

ENV = mw.Environment(wl=1.55, T=25.0)


def _make_cs(
    material: mw.Material, *, subpixel_smoothing: bool = True
) -> mw.CrossSection:
    """Create a small rectangular-waveguide cross-section for the material."""
    mesh = mw.Mesh2D(
        x=np.linspace(-1.0, 1.0, 41),
        y=np.linspace(-0.6, 0.6, 25),
    )
    box = mw.Box(x_min=-0.25, x_max=0.25, y_min=-0.11, y_max=0.11, z_min=0.0, z_max=1.0)
    struct = mw.Structure(material=material, geometry=box)
    cell = mw.Cell(structures=[struct], mesh=mesh, z_min=0.0, z_max=1.0)
    return mw.CrossSection.from_cell(
        cell=cell, env=ENV, subpixel_smoothing=subpixel_smoothing
    )


def _rotated_tensor(eps_diag: list[float], angle_deg: float) -> np.ndarray:
    """Rotate a diagonal permittivity tensor about the z-axis."""
    th = np.deg2rad(angle_deg)
    rot = np.array(
        [
            [np.cos(th), -np.sin(th), 0.0],
            [np.sin(th), np.cos(th), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return rot @ np.diag(eps_diag) @ rot.T


def test_eps_from_scalar() -> None:
    mat = mw.AnisotropicMaterial(name="am_scalar", eps=4.0)
    assert mat.eps.shape == (3, 3)
    assert np.allclose(mat.eps, 4.0 * np.eye(3))
    assert mat.is_isotropic


def test_eps_from_diagonal() -> None:
    mat = mw.AnisotropicMaterial(name="am_diag", eps=[4.0, 4.41, 4.84])
    assert mat.eps.shape == (3, 3)
    assert np.allclose(mat.eps, np.diag([4.0, 4.41, 4.84]))
    assert not mat.is_isotropic


def test_eps_from_full_tensor() -> None:
    eps = _rotated_tensor([4.0, 4.41, 4.84], 30.0)
    mat = mw.AnisotropicMaterial(name="am_full", eps=eps)
    assert mat.eps.shape == (3, 3)
    assert np.allclose(mat.eps, eps)
    assert not mat.is_isotropic
    assert np.allclose(mat.eps_tensor(ENV), eps)


@pytest.mark.parametrize(
    "eps",
    [
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
        [[1.0, 0.0], [0.0, 2.0]],
        np.ones((3, 2)),
        np.ones((2, 3, 3)),
    ],
)
def test_eps_invalid_spec_raises(eps: object) -> None:
    with pytest.raises(ValidationError):
        mw.AnisotropicMaterial(name="am_invalid", eps=eps)


def test_from_n_scalar() -> None:
    mat = mw.AnisotropicMaterial.from_n("am_from_n_scalar", 2.0)
    assert np.allclose(mat.eps, 4.0 * np.eye(3))
    assert mat.is_isotropic


def test_from_n_diagonal() -> None:
    mat = mw.AnisotropicMaterial.from_n("am_from_n_diag", [2.0, 2.1, 2.2])
    assert np.allclose(mat.eps, np.diag([4.0, 4.41, 4.84]))


def test_effective_index() -> None:
    mat = mw.AnisotropicMaterial(name="am_neff", eps=[4.0, 4.0, 4.0])
    assert np.isclose(complex(mat(ENV)), 2.0)


def test_index_material_eps_tensor() -> None:
    mat = mw.IndexMaterial(name="im_eps", n=2.0)
    assert np.allclose(mat.eps_tensor(ENV), 4.0 * np.eye(3))


def test_sampled_material_eps_tensor() -> None:
    n = complex(mw.silicon(ENV))
    assert np.allclose(mw.silicon.eps_tensor(ENV), n**2 * np.eye(3))


def test_serialization_roundtrip() -> None:
    eps = _rotated_tensor([4.0, 4.41, 4.84], 30.0)
    mat = mw.AnisotropicMaterial(name="am_serde", eps=eps)
    mat2 = mw.AnisotropicMaterial.model_validate(mat.model_dump())
    assert mat2 == mat
    assert np.allclose(mat2.eps, eps)


@pytest.mark.parametrize("subpixel_smoothing", [True, False])
def test_cross_section_diagonal_components(subpixel_smoothing: bool) -> None:  # noqa: FBT001
    """The cross-section samples the matching tensor diagonal per component."""
    eps_diag = [4.0, 9.0, 16.0]
    mat = mw.AnisotropicMaterial(name="am_cs_diag", eps=eps_diag)
    cs = _make_cs(mat, subpixel_smoothing=subpixel_smoothing)
    # interior (non-interface) structure pixels hold the unsmoothed value,
    # which is also the maximum since the background is air (eps=1)
    assert np.isclose(np.max(np.real(cs.nx**2)), eps_diag[0])
    assert np.isclose(np.max(np.real(cs.ny**2)), eps_diag[1])
    assert np.isclose(np.max(np.real(cs.nz**2)), eps_diag[2])


def test_cross_section_offdiag_components() -> None:
    eps = _rotated_tensor([4.0, 4.41, 4.84], 30.0)
    mat = mw.AnisotropicMaterial(name="am_cs_full", eps=eps)
    cs = _make_cs(mat)
    for prop, (i, j) in {
        "eps_xy": (0, 1),
        "eps_xz": (0, 2),
        "eps_yx": (1, 0),
        "eps_yz": (1, 2),
        "eps_zx": (2, 0),
        "eps_zy": (2, 1),
    }.items():
        arr = getattr(cs, prop)
        assert arr.shape == cs.nx.shape
        values = set(np.unique(arr))
        # background pixels are 0, in-structure pixels carry the tensor entry
        assert values == {0.0 + 0.0j, complex(eps[i, j])}


def test_cross_section_offdiag_zero_for_isotropic() -> None:
    mat = mw.IndexMaterial(name="im_cs_iso", n=2.0)
    cs = _make_cs(mat)
    for prop in ["eps_xy", "eps_xz", "eps_yx", "eps_yz", "eps_zx", "eps_zy"]:
        assert not np.any(getattr(cs, prop))


def test_fde_isotropic_equivalence() -> None:
    """All isotropic specs must produce identical modes."""
    n = 3.0
    materials = [
        mw.IndexMaterial(name="fde_iso_n", n=n),
        mw.AnisotropicMaterial(name="fde_iso_scalar", eps=n**2),
        mw.AnisotropicMaterial(name="fde_iso_diag", eps=[n**2, n**2, n**2]),
        mw.AnisotropicMaterial(name="fde_iso_full", eps=n**2 * np.eye(3)),
    ]
    neffs = [
        [m.neff for m in mw.compute_modes(_make_cs(mat), num_modes=2)]
        for mat in materials
    ]
    for other in neffs[1:]:
        np.testing.assert_allclose(neffs[0], other)


def test_fde_diagonal_birefringence() -> None:
    """The TE-like fundamental mode is mostly sensitive to eps_xx."""
    mat_xx_hi = mw.AnisotropicMaterial(name="fde_bir_hi", eps=[3.2**2, 2.8**2, 3.0**2])
    mat_xx_lo = mw.AnisotropicMaterial(name="fde_bir_lo", eps=[2.8**2, 3.2**2, 3.0**2])
    neff_hi = np.real(mw.compute_modes(_make_cs(mat_xx_hi), num_modes=1)[0].neff)
    neff_lo = np.real(mw.compute_modes(_make_cs(mat_xx_lo), num_modes=1)[0].neff)
    assert neff_hi > neff_lo
    assert 1.0 < neff_lo < 2.8
    assert 1.0 < neff_hi < 3.2


def test_fde_full_tensor() -> None:
    """Off-diagonal permittivity reaches the tensorial tidy3d solver.

    The open-source tidy3d package delegates the fully tensorial eigensolver
    to the proprietary `tidy3d-extras` package: without it installed, tidy3d
    raises a NotImplementedError that points the user at it.
    """
    eps = _rotated_tensor([3.2**2, 2.8**2, 3.0**2], 30.0)
    mat = mw.AnisotropicMaterial(name="fde_full_tensor", eps=eps)
    cs = _make_cs(mat)
    if importlib.util.find_spec("tidy3d_extras") is None:
        with pytest.raises(NotImplementedError, match="tensorial"):
            mw.compute_modes(cs, num_modes=1)
    else:
        modes = mw.compute_modes(cs, num_modes=1)
        neff = np.real(modes[0].neff)
        assert np.isfinite(neff)
        assert 1.0 < neff < 3.2
