"""Offline tests for the RefractiveIndex.INFO + Sellmeier utility (meow.ridb).

Everything here is network-free: URL parsing, YAML parsing, formula evaluation
and Sellmeier fitting are exercised with synthetic data and inline YAML
documents. The only networked entry point, :func:`meow.fetch_ri_entry`, is not
tested here.
"""

from __future__ import annotations

import numpy as np
import pytest

import meow as mw
from meow.ridb import _as_index

# a 3-term Sellmeier ground truth (fused-silica-like) to generate synthetic data
_B = np.array([0.6961663, 0.4079426, 0.8974794])
_C = np.array([0.0684043, 0.1162414, 9.896161]) ** 2  # poles in um^2


def _truth(wls: np.ndarray) -> np.ndarray:
    n2 = np.ones_like(wls)
    for b, c in zip(_B, _C, strict=True):
        n2 = n2 + b * wls**2 / (wls**2 - c)
    return np.sqrt(n2)


# ----------------------------------------------------------------------------
# formula evaluation
# ----------------------------------------------------------------------------
def test_eval_formula1_matches_sellmeier_model() -> None:
    """Formula 1 (sqrt poles) agrees with the equivalent SellmeierModel."""
    coeffs = np.array(
        [0.0, _B[0], np.sqrt(_C[0]), _B[1], np.sqrt(_C[1]), _B[2], np.sqrt(_C[2])]
    )
    wls = np.linspace(0.3, 2.0, 25)
    from_formula = mw.eval_ri_formula(1, coeffs, wls)
    model = mw.SellmeierModel(b=_B, c=_C)
    assert np.allclose(from_formula, model.index(wls), atol=1e-12)
    # physically sensible fused-silica index near 1.55 um
    n1550 = float(mw.eval_ri_formula(1, coeffs, np.array([1.55]))[0])
    assert 1.4 < n1550 < 1.5


def test_eval_formula2_and_cauchy_and_gas() -> None:
    """Formulas 2 (Sellmeier-2), 3/5 (Cauchy) and 6 (gas) evaluate sanely."""
    wls = np.array([0.5, 1.0, 1.5])
    # formula 2: poles given directly (um^2)
    f2 = mw.eval_ri_formula(2, np.array([0.0, 1.0, 0.01]), wls)
    assert np.all(f2 > 1.0)
    # formula 3 (n^2 = c0 + sum) and 5 (n = c0 + sum)
    f3 = mw.eval_ri_formula(3, np.array([2.25, 0.01, -2.0]), wls)
    f5 = mw.eval_ri_formula(5, np.array([1.5, 0.01, -2.0]), wls)
    assert np.all(np.isfinite(f3))
    assert np.all(np.isfinite(f5))
    # formula 6 (gases): n very close to 1
    f6 = mw.eval_ri_formula(6, np.array([0.0, 1e-4, 100.0]), wls)
    assert np.all(np.abs(f6 - 1.0) < 1e-2)


def test_eval_formula_unsupported_raises() -> None:
    """An unsupported formula type raises a helpful NotImplementedError."""
    with pytest.raises(NotImplementedError):
        mw.eval_ri_formula(4, np.array([1.0, 2.0]), np.array([1.0]))


# ----------------------------------------------------------------------------
# URL parsing
# ----------------------------------------------------------------------------
def test_ri_data_url_query_form() -> None:
    """The website ?shelf=&book=&page= form maps to the raw mirror paths."""
    cands = mw.ri_data_url(
        "https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson"
    )
    assert cands[0].endswith("data/main/SiO2/nk/Malitson.yml")
    assert cands[1].endswith("data/main/SiO2/n2/Malitson.yml")
    assert all("/main/database/data/" in c for c in cands)


def test_ri_data_url_direct_yml_and_bare_path() -> None:
    """A direct .yml URL passes through; a bare shelf/book/page path is mapped."""
    direct = "https://example.com/foo/bar/Baz.yml"
    assert mw.ri_data_url(direct) == [direct]
    cands = mw.ri_data_url("main/LiNbO3/Zelmon-o")
    assert cands[0].endswith("data/main/LiNbO3/nk/Zelmon-o.yml")


def test_ri_data_url_bad_reference_raises() -> None:
    """A reference that is not parseable raises a ValueError."""
    with pytest.raises(ValueError, match="parse"):
        mw.ri_data_url("notaurl")


# ----------------------------------------------------------------------------
# YAML parsing
# ----------------------------------------------------------------------------
_YAML_FORMULA_PLUS_K = """
REFERENCES: "test"
DATA:
  - type: formula 1
    wavelength_range: 0.4 5.0
    coefficients: 0 2.6734 0.01764 1.2290 0.05914 12.614 474.6
  - type: tabulated k
    data: |
      0.40 0.001
      1.55 0.0002
      5.00 0.01
"""

_YAML_TABULATED_NK = """
DATA:
  - type: tabulated nk
    data: |
      0.5 1.50 0.001
      1.0 1.45 0.0005
      2.0 1.44 0.0002
"""


def test_parse_yaml_formula_plus_k_table() -> None:
    """A formula-1 entry with a separate k table evaluates n from the formula."""
    entry = mw.parse_ri_yaml(_YAML_FORMULA_PLUS_K, name="LN")
    assert entry.has_formula
    assert entry.formula_type == 1
    assert entry.wl_range == (0.4, 5.0)
    n = entry.index(np.array([1.55]))
    assert np.real(n[0]) > 2.0  # LiNbO3-like
    assert np.imag(n[0]) > 0.0  # k carried from the table


def test_parse_yaml_tabulated_nk() -> None:
    """A tabulated nk entry interpolates n and k from the table."""
    entry = mw.parse_ri_yaml(_YAML_TABULATED_NK)
    assert entry.has_table
    assert not entry.has_formula
    assert entry.wl_range == (0.5, 2.0)
    n = entry.index(np.array([1.0]))
    assert np.isclose(np.real(n[0]), 1.45, atol=1e-9)
    assert np.isclose(np.imag(n[0]), 0.0005, atol=1e-9)
    grid = entry.default_wls(50)
    assert grid[0] == 0.5
    assert grid[-1] == 2.0


# ----------------------------------------------------------------------------
# Sellmeier fitting
# ----------------------------------------------------------------------------
def test_fit_sellmeier_recovers_synthetic_data() -> None:
    """A 3-term fit recovers a 3-term ground truth to machine precision."""
    wls = np.linspace(0.35, 1.9, 60)
    n = _truth(wls)
    model = mw.fit_sellmeier(wls, n, num_terms=3, name="fs")
    assert model.num_terms == 3
    assert model.rms_error is not None
    assert model.rms_error < 1e-8
    assert np.allclose(model.index(wls), n, atol=1e-6)
    # no pole inside the transparency window
    lo, hi = wls.min(), wls.max()
    assert not np.any((model.c > lo**2) & (model.c < hi**2))


def test_fit_sellmeier_respects_wl_range() -> None:
    """Restricting wl_range sets the model's stated validity to that window."""
    wls = np.linspace(0.3, 3.0, 80)
    n = _truth(wls)
    model = mw.fit_sellmeier(wls, n, num_terms=2, wl_range=(1.0, 2.0))
    assert model.wl_range == (1.0, 2.0)
    test_wls = np.linspace(1.0, 2.0, 20)
    assert np.allclose(model.index(test_wls), _truth(test_wls), atol=1e-3)


def test_sellmeier_model_coefficients_roundtrip() -> None:
    """coefficients() reports the term count, poles and offset as plain data."""
    model = mw.SellmeierModel(b=_B, c=_C, a0=0.0, wl_range=(0.4, 2.0))
    d = model.coefficients()
    assert d["num_terms"] == 3
    assert len(d["b"]) == len(d["c_um2"]) == 3
    assert d["wl_range_um"] == (0.4, 2.0)


# ----------------------------------------------------------------------------
# temperature-dependent Sellmeier
# ----------------------------------------------------------------------------
def test_fit_temperature_sellmeier_interpolates() -> None:
    """A linear-in-T model recovers the per-T data and interpolates between."""
    wls = np.linspace(0.5, 2.0, 50)
    dn_dt = 1e-4  # per degC, applied as a uniform index shift

    def n_at(t: float) -> np.ndarray:
        return _truth(wls) + dn_dt * (t - 20.0)

    datasets = {20.0: (wls, n_at(20.0)), 80.0: (wls, n_at(80.0))}
    model = mw.fit_temperature_sellmeier(datasets, num_terms=3, t_degree=1)
    assert model.num_terms == 3
    # recovers each measured temperature
    assert np.allclose(model.index(wls, 20.0), n_at(20.0), atol=5e-4)
    assert np.allclose(model.index(wls, 80.0), n_at(80.0), atol=5e-4)
    # interpolates to an unmeasured temperature with the right trend
    n50 = model.index(wls, 50.0)
    assert np.allclose(n50, n_at(50.0), atol=1e-3)


# ----------------------------------------------------------------------------
# meow material builders
# ----------------------------------------------------------------------------
def test_sellmeier_material_is_dispersive() -> None:
    """sellmeier_material builds a SampledMaterial that interpolates n(wl)."""
    grid = np.linspace(0.4, 2.0, 40)
    model = mw.fit_sellmeier(grid, _truth(grid), num_terms=3, wl_range=(0.4, 2.0))
    mat = mw.sellmeier_material("fs", model)
    assert isinstance(mat, mw.SampledMaterial)
    n_lo = mat(mw.Environment(wl=0.5, T=25.0))
    n_hi = mat(mw.Environment(wl=1.8, T=25.0))
    assert np.real(n_lo) > np.real(n_hi)  # normal dispersion
    assert np.isclose(np.real(n_lo), _truth(np.array([0.5]))[0], atol=1e-3)


def test_material_from_ri_with_and_without_fit() -> None:
    """material_from_ri samples an entry directly and via a Sellmeier refit."""
    entry = mw.parse_ri_yaml(_YAML_TABULATED_NK)
    direct = mw.material_from_ri(entry, name="tab")
    assert isinstance(direct, mw.SampledMaterial)
    # refit a smooth source (a SellmeierModel) to a 2-term model
    model = mw.SellmeierModel(b=_B, c=_C, wl_range=(0.4, 2.0))
    refit = mw.material_from_ri(model, name="refit", fit_terms=2, wl_range=(0.6, 1.8))
    n = refit(mw.Environment(wl=1.0, T=25.0))
    assert np.isclose(np.real(n), _truth(np.array([1.0]))[0], atol=2e-3)


def test_anisotropic_material_is_uniaxial() -> None:
    """A uniaxial crystal (n_o != n_e) yields a non-isotropic material."""
    wls = np.linspace(0.5, 2.0, 40)
    ordinary = (wls, _truth(wls))
    extraordinary = (wls, _truth(wls) + 0.07)  # birefringence
    mat = mw.anisotropic_material(
        "uniaxial",
        {"x": ordinary, "y": ordinary, "z": extraordinary},
        wls=wls,
    )
    assert isinstance(mat, mw.SampledAnisotropicMaterial)
    assert not mat.is_isotropic
    eps = mat.eps_tensor(mw.Environment(wl=1.55, T=25.0))
    nxx, nyy, nzz = np.sqrt(eps[0, 0]), np.sqrt(eps[1, 1]), np.sqrt(eps[2, 2])
    assert np.isclose(np.real(nxx), np.real(nyy), atol=1e-6)
    assert np.real(nzz) > np.real(nxx)


def test_temperature_material_interpolates_in_wl_and_t() -> None:
    """temperature_material builds a (wl, T) SampledMaterial matching the model."""
    wls = np.linspace(0.5, 2.0, 40)
    dn_dt = 2e-4

    def n_at(t: float) -> np.ndarray:
        return _truth(wls) + dn_dt * (t - 20.0)

    datasets = {20.0: (wls, n_at(20.0)), 80.0: (wls, n_at(80.0))}
    model = mw.fit_temperature_sellmeier(datasets, num_terms=3, t_degree=1)
    mat = mw.temperature_material("fs_T", model, wls=wls)
    assert isinstance(mat, mw.SampledMaterial)
    assert "T" in mat.params
    n = mat(mw.Environment(wl=1.55, T=50.0))
    expected = float(model.index(np.array([1.55]), 50.0)[0])
    assert np.isclose(np.real(n), expected, atol=2e-3)


def test_as_index_rejects_unknown_source() -> None:
    """_as_index raises TypeError on an unsupported source type."""
    with pytest.raises(TypeError):
        _as_index(object(), np.array([1.0]))
