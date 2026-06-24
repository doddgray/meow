"""RefractiveIndex.INFO interface + Sellmeier model fitting.

This module turns *published* refractive-index data - either from
`refractiveindex.info <https://refractiveindex.info>`_ or from user-supplied
datasets - into frequency-dependent :mod:`meow` material models. It has three
layers:

1. **RefractiveIndex.INFO access** - parse a ``refractiveindex.info`` URL (the
   ``?shelf=&book=&page=`` form or a direct ``.yml`` link), fetch the entry's
   YAML (cached on disk), and evaluate its refractive index whether it is given
   as *tabulated* ``n``/``k`` data or as a *formula* fit (Sellmeier, Sellmeier-2,
   Cauchy, gas formulas - RefractiveIndex.INFO formulas 1, 2, 3, 5, 6).

2. **Sellmeier fitting** (:func:`fit_sellmeier`) - fit an ``N``-term Sellmeier
   model ``n^2 = 1 + a0 + sum_j b_j lam^2 / (lam^2 - c_j)`` to tabular index data
   over a chosen **wavelength range of validity**, parameterized by the number
   of terms. :class:`TemperatureSellmeier` extends this to data measured at
   several temperatures, fitting a polynomial-in-temperature for each
   coefficient so ``n(lambda, T)`` can be evaluated continuously.

3. **meow material builders** - :func:`sellmeier_material` /
   :func:`material_from_ri` produce an isotropic
   :class:`meow.SampledMaterial`; :func:`anisotropic_material` builds a
   :class:`meow.SampledAnisotropicMaterial` from a *different* entry/dataset per
   crystal axis; :func:`temperature_material` builds a ``(wavelength,
   temperature)``-dispersive :class:`meow.SampledMaterial`.

The YAML parsing, formula evaluation, URL parsing and Sellmeier fitting are all
pure/offline (and unit-tested without network); only :func:`fetch_ri_entry`
touches the network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import parse_qs, urlparse

import numpy as np

if TYPE_CHECKING:
    from meow.materials import SampledAnisotropicMaterial, SampledMaterial

# the GitHub mirror of the RefractiveIndex.INFO database (the canonical source
# of the YAML entries); entries live under data/<shelf>/<book>/<nk|n2>/<page>.yml
_GH_RAW = (
    "https://raw.githubusercontent.com/polyanskiy/"
    "refractiveindex.info-database/main/database"
)


# ==========================================================================
# RefractiveIndex.INFO formula evaluation (standard, public-domain physics)
# ==========================================================================
def eval_ri_formula(
    formula_type: int, coeffs: np.ndarray, wls: np.ndarray
) -> np.ndarray:
    """Evaluate a RefractiveIndex.INFO dispersion *formula* at wavelengths ``wls``.

    Implements the common RefractiveIndex.INFO formula types (``wls`` in um):

    - 1 (Sellmeier): ``n^2 - 1 = c0 + sum_i c_{2i-1} lam^2 / (lam^2 - c_{2i}^2)``
    - 2 (Sellmeier-2): ``n^2 - 1 = c0 + sum_i c_{2i-1} lam^2 / (lam^2 - c_{2i})``
    - 3 (Cauchy): ``n^2 = c0 + sum_i c_{2i-1} lam^{c_{2i}}``
    - 5 (Cauchy): ``n = c0 + sum_i c_{2i-1} lam^{c_{2i}}``
    - 6 (gases): ``n - 1 = c0 + sum_i c_{2i-1} / (c_{2i} - lam^{-2})``

    Args:
        formula_type: the RefractiveIndex.INFO formula number.
        coeffs: the formula coefficients (the ``coefficients`` field, in order).
        wls: wavelengths [um].

    Returns:
        The real refractive index at each wavelength.
    """
    wls = np.asarray(wls, dtype=float)
    c = np.asarray(coeffs, dtype=float)
    c0 = c[0]
    rest = c[1:]
    if formula_type == 1:
        n2 = 1.0 + c0
        for i in range(0, len(rest) - 1, 2):
            n2 = n2 + rest[i] * wls**2 / (wls**2 - rest[i + 1] ** 2)
        return np.sqrt(n2)
    if formula_type == 2:
        n2 = 1.0 + c0
        for i in range(0, len(rest) - 1, 2):
            n2 = n2 + rest[i] * wls**2 / (wls**2 - rest[i + 1])
        return np.sqrt(n2)
    if formula_type == 3:
        n2 = c0 + sum(
            rest[i] * wls ** rest[i + 1] for i in range(0, len(rest) - 1, 2)
        )
        return np.sqrt(n2)
    if formula_type == 5:
        return c0 + sum(
            rest[i] * wls ** rest[i + 1] for i in range(0, len(rest) - 1, 2)
        )
    if formula_type == 6:
        n = 1.0 + c0
        for i in range(0, len(rest) - 1, 2):
            n = n + rest[i] / (rest[i + 1] - wls ** (-2))
        return n
    msg = (
        f"RefractiveIndex.INFO formula {formula_type} is not supported "
        "(supported: 1, 2, 3, 5, 6). Use the tabulated data instead."
    )
    raise NotImplementedError(msg)


# ==========================================================================
# a parsed RefractiveIndex.INFO (or user) entry
# ==========================================================================
@dataclass
class RIEntry:
    """A parsed refractive-index entry (tabulated data and/or a formula).

    Attributes:
        name: a human-readable name.
        wl_range: the ``(min, max)`` wavelength validity range [um], if known.
        formula_type: the RefractiveIndex.INFO formula number, or ``None``.
        coeffs: the formula coefficients, if a formula is present.
        wl_tab: tabulated wavelengths [um], if tabulated data is present.
        n_tab: tabulated complex refractive index aligned with ``wl_tab``.
    """

    name: str = "material"
    wl_range: tuple[float, float] | None = None
    formula_type: int | None = None
    coeffs: np.ndarray | None = None
    wl_tab: np.ndarray | None = None
    n_tab: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def has_formula(self) -> bool:
        """Whether the entry carries a dispersion formula."""
        return self.formula_type is not None and self.coeffs is not None

    @property
    def has_table(self) -> bool:
        """Whether the entry carries tabulated data."""
        return self.wl_tab is not None and len(self.wl_tab) > 0

    def index(self, wls: np.ndarray) -> np.ndarray:
        """Evaluate the (complex) refractive index at ``wls`` [um].

        Prefers the analytic formula (real index); falls back to interpolating
        the tabulated data (carrying ``k`` if present).
        """
        wls = np.asarray(wls, dtype=float)
        if self.has_formula:
            ft = cast("int", self.formula_type)
            cf = cast("np.ndarray", self.coeffs)
            n = eval_ri_formula(ft, cf, wls).astype(complex)
            if self.has_table:  # carry k (absorption) from an accompanying table
                wl_tab = cast("np.ndarray", self.wl_tab)
                n_tab = cast("np.ndarray", self.n_tab)
                k = np.interp(wls, wl_tab, np.imag(n_tab))
                n = np.real(n) + 1j * k
            return n
        if self.has_table:
            wl_tab = cast("np.ndarray", self.wl_tab)
            n_tab = cast("np.ndarray", self.n_tab)
            nr = np.interp(wls, wl_tab, np.real(n_tab))
            ni = np.interp(wls, wl_tab, np.imag(n_tab))
            return nr + 1j * ni
        msg = f"RIEntry {self.name!r} has neither a formula nor tabulated data."
        raise ValueError(msg)

    def default_wls(self, npts: int = 200) -> np.ndarray:
        """A dense wavelength grid over the entry's validity range."""
        if self.wl_range is not None:
            lo, hi = self.wl_range
        elif self.has_table:
            wl_tab = cast("np.ndarray", self.wl_tab)
            lo, hi = float(wl_tab.min()), float(wl_tab.max())
        else:
            msg = "no wavelength range available; pass wls explicitly."
            raise ValueError(msg)
        return np.linspace(lo, hi, npts)


# ==========================================================================
# YAML parsing
# ==========================================================================
def _parse_tabulated(data_str: str, *, has_n: bool, has_k: bool) -> tuple:
    rows = np.array(
        [[float(x) for x in line.split()] for line in data_str.strip().splitlines()]
    )
    wl = rows[:, 0]
    if has_n and has_k:
        n = rows[:, 1] + 1j * rows[:, 2]
    elif has_n:
        n = rows[:, 1].astype(complex)
    else:  # k only
        n = 1j * rows[:, 1]
    order = np.argsort(wl)
    return wl[order], n[order]


def parse_ri_yaml(text: str, name: str | None = None) -> RIEntry:
    """Parse a RefractiveIndex.INFO YAML document into an :class:`RIEntry`.

    Handles the ``DATA`` blocks: ``tabulated n`` / ``tabulated k`` /
    ``tabulated nk`` and ``formula N`` (with ``coefficients`` and
    ``wavelength_range`` / ``range``). When both a formula and a ``k`` table are
    present the formula provides ``n`` and the table provides ``k``.
    """
    import yaml

    doc = yaml.safe_load(text)
    blocks = doc.get("DATA", doc.get("data", []))
    entry = RIEntry(name=name or "material")
    wl_tab_k = n_tab_k = None
    for block in blocks:
        btype = str(block.get("type", "")).strip().lower()
        if btype.startswith("formula"):
            entry.formula_type = int(btype.split()[-1])
            entry.coeffs = np.array(
                [float(x) for x in str(block["coefficients"]).split()], dtype=float
            )
            rng = block.get("wavelength_range") or block.get("range")
            if rng is not None:
                lo, hi = (float(x) for x in str(rng).split())
                entry.wl_range = (lo, hi)
        elif btype.startswith("tabulated"):
            kind = btype.split()[-1]
            wl, n = _parse_tabulated(
                str(block["data"]), has_n="n" in kind, has_k="k" in kind
            )
            if kind == "k":
                wl_tab_k, n_tab_k = wl, n
            else:
                entry.wl_tab, entry.n_tab = wl, n
                if entry.wl_range is None:
                    entry.wl_range = (float(wl.min()), float(wl.max()))
    # merge a separate k-table onto the n-table / formula
    if wl_tab_k is not None and entry.wl_tab is None:
        # no n-table: keep the k-table so index() can carry absorption onto the
        # formula (or, with no formula, expose k alone)
        entry.wl_tab, entry.n_tab = wl_tab_k, n_tab_k
    elif wl_tab_k is not None and entry.has_table:
        n_tab = cast("np.ndarray", entry.n_tab)
        n_tab_k = cast("np.ndarray", n_tab_k)
        k = np.interp(cast("np.ndarray", entry.wl_tab), wl_tab_k, np.imag(n_tab_k))
        entry.n_tab = np.real(n_tab) + 1j * k
    return entry


# ==========================================================================
# URL handling + (cached) network fetch
# ==========================================================================
def ri_data_url(url: str) -> list[str]:
    """Candidate raw-YAML URLs for a ``refractiveindex.info`` reference.

    Accepts a direct ``.yml`` URL, a database path, or the website's
    ``?shelf=&book=&page=`` query form, and returns the candidate raw URLs to
    try (the database has used both ``data-nk`` and ``data`` subtrees).
    """
    if url.endswith(".yml"):
        return [url]
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if {"shelf", "book", "page"} <= qs.keys():
        shelf, book, page = qs["shelf"][0], qs["book"][0], qs["page"][0]
    else:
        # a bare "shelf/book/page" (or ".../<nk|n2>/page") database path
        m = re.search(r"([^/]+)/([^/]+)/([^/?#]+?)(?:\.yml)?$", parsed.path or url)
        if not m:
            msg = f"Could not parse a RefractiveIndex.INFO reference from {url!r}."
            raise ValueError(msg)
        shelf, book, page = m.groups()
    # the entry sits in the book's "nk" (real+imag) or "n2" (n^2) subtree; try
    # both, then the legacy flat layout as a fallback
    return [
        f"{_GH_RAW}/data/{shelf}/{book}/nk/{page}.yml",
        f"{_GH_RAW}/data/{shelf}/{book}/n2/{page}.yml",
        f"{_GH_RAW}/data/{shelf}/{book}/{page}.yml",
    ]


def _cache_path(cache_dir: str | Path, url: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", url).strip("_")
    return Path(cache_dir) / f"{safe}"


def fetch_ri_entry(
    url: str, *, cache_dir: str | Path | None = None, timeout: float = 30.0
) -> RIEntry:
    """Fetch + parse a RefractiveIndex.INFO entry from a URL (cached on disk).

    Args:
        url: a ``refractiveindex.info`` URL (``?shelf=&book=&page=`` or a direct
            ``.yml`` link) or a ``shelf/book/page`` database path.
        cache_dir: directory to cache downloaded YAML in (default
            ``~/.cache/meow/ridb``); reused on subsequent calls to avoid network.
        timeout: network timeout in seconds.

    Returns:
        The parsed :class:`RIEntry`.
    """
    import urllib.request

    cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache/meow/ridb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    candidates = ri_data_url(url)
    name = _entry_name(url)
    last_err: Exception | None = None
    for cand in candidates:
        cached = _cache_path(cache_dir, cand)
        if cached.exists():
            return parse_ri_yaml(cached.read_text(), name=name)
        req = urllib.request.Request(  # noqa: S310 - https GitHub raw URLs only
            cand, headers={"User-Agent": "meow-ridb"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                text = resp.read().decode("utf-8")
        except Exception as e:  # noqa: BLE001 - try the next candidate
            last_err = e
            continue
        cached.write_text(text)
        return parse_ri_yaml(text, name=name)
    msg = (
        f"Could not fetch {url!r} (tried {candidates}). Last error: {last_err!r}. "
        "If offline, pass a local YAML via parse_ri_yaml or a (wls, n) dataset."
    )
    raise RuntimeError(msg)


def _entry_name(url: str) -> str:
    qs = parse_qs(urlparse(url).query)
    if {"book", "page"} <= qs.keys():
        return f"{qs['book'][0]}-{qs['page'][0]}"
    return Path(urlparse(url).path or url).stem or "material"


# ==========================================================================
# Sellmeier model + fitting
# ==========================================================================
@dataclass
class SellmeierModel:
    """An ``N``-term Sellmeier dispersion model.

    ``n^2(lambda) = 1 + a0 + sum_j b_j lambda^2 / (lambda^2 - c_j)`` with
    ``lambda`` in um and the poles ``c_j`` in um^2.

    Attributes:
        b: the oscillator strengths ``b_j`` (length = number of terms).
        c: the squared pole wavelengths ``c_j`` [um^2].
        a0: a constant offset (usually 0).
        wl_range: the fitted ``(min, max)`` wavelength range of validity [um].
        name: a human-readable name.
        rms_error: RMS index error of the fit over the fit data, if known.
    """

    b: np.ndarray
    c: np.ndarray
    a0: float = 0.0
    wl_range: tuple[float, float] | None = None
    name: str = "sellmeier"
    rms_error: float | None = None

    @property
    def num_terms(self) -> int:
        """The number of Sellmeier terms."""
        return len(self.b)

    def eps(self, wls: np.ndarray) -> np.ndarray:
        """Relative permittivity ``n^2`` at ``wls`` [um]."""
        wls = np.asarray(wls, dtype=float)
        n2 = np.full_like(wls, 1.0 + self.a0, dtype=float)
        for b_j, c_j in zip(self.b, self.c, strict=True):
            n2 = n2 + b_j * wls**2 / (wls**2 - c_j)
        return n2

    def index(self, wls: np.ndarray) -> np.ndarray:
        """Refractive index ``n`` at ``wls`` [um]."""
        return np.sqrt(self.eps(wls))

    def coefficients(self) -> dict[str, Any]:
        """The fitted coefficients as a plain dict (for serialization/printing)."""
        return {
            "a0": float(self.a0),
            "b": [float(x) for x in self.b],
            "c_um2": [float(x) for x in self.c],
            "wl_range_um": self.wl_range,
            "num_terms": self.num_terms,
            "rms_error": self.rms_error,
        }


def _restrict(wls: np.ndarray, n: np.ndarray, wl_range: tuple | None) -> tuple:
    if wl_range is None:
        return wls, n
    lo, hi = wl_range
    m = (wls >= lo) & (wls <= hi)
    if m.sum() < 2:
        msg = f"fewer than 2 data points in wl_range {wl_range}."
        raise ValueError(msg)
    return wls[m], n[m]


def fit_sellmeier(
    wls: np.ndarray,
    n: np.ndarray,
    *,
    num_terms: int = 3,
    wl_range: tuple[float, float] | None = None,
    fit_a0: bool = False,
    n_restarts: int = 6,
    name: str = "sellmeier",
    seed: int = 0,
) -> SellmeierModel:
    """Fit an ``N``-term Sellmeier model to tabulated index data.

    Args:
        wls: sampled wavelengths [um].
        n: sampled (real) refractive index at ``wls``.
        num_terms: number of Sellmeier terms ``N`` (each adds a ``b_j``, ``c_j``).
        wl_range: restrict the fit (and the model's stated validity) to this
            ``(min, max)`` wavelength window [um]; defaults to the data extent.
        fit_a0: also fit the constant offset ``a0`` (default ``0``).
        n_restarts: number of randomized initial guesses; the best (lowest RMS
            index error with no pole inside the fit window) is returned.
        name: name for the resulting model.
        seed: RNG seed for the restart initial guesses (reproducible).

    Returns:
        The fitted :class:`SellmeierModel`.
    """
    from scipy.optimize import least_squares

    wls = np.asarray(wls, dtype=float)
    n = np.real(np.asarray(n)).astype(float)
    order = np.argsort(wls)
    wls, n = wls[order], n[order]
    fit_wls, fit_n = _restrict(wls, n, wl_range)
    lo, hi = float(fit_wls.min()), float(fit_wls.max())
    # the stated validity is the requested window (if any), else the data extent
    stated_range = wl_range if wl_range is not None else (lo, hi)
    rng = np.random.default_rng(seed)

    def unpack(p: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        a0 = p[0] if fit_a0 else 0.0
        rest = p[1:] if fit_a0 else p
        return a0, rest[:num_terms], rest[num_terms:]

    def residual(p: np.ndarray) -> np.ndarray:
        a0, b, c = unpack(p)
        n2 = 1.0 + a0
        for b_j, c_j in zip(b, c, strict=True):
            n2 = n2 + b_j * fit_wls**2 / (fit_wls**2 - c_j)
        # large penalty if a pole sits inside the fit window (n2 may go <0)
        model = np.sqrt(np.clip(n2, 1e-6, None))
        return model - fit_n

    # poles live outside the transparency window: some UV (c < lo^2), some IR
    lo2, hi2 = lo**2, hi**2
    best: SellmeierModel | None = None
    best_cost = np.inf
    for _ in range(max(1, n_restarts)):
        c0 = []
        for j in range(num_terms):
            if j % 2 == 0:  # UV pole below the window
                c0.append((lo * rng.uniform(0.2, 0.95)) ** 2)
            else:  # IR pole above the window
                c0.append((hi * rng.uniform(1.1, 4.0)) ** 2)
        b0 = rng.uniform(0.2, 1.5, size=num_terms)
        p0 = np.array(([0.0] if fit_a0 else []) + list(b0) + c0)
        try:
            sol = least_squares(residual, p0, method="lm", max_nfev=4000)
        except Exception:  # noqa: BLE001, S112 - bad start, try another
            continue
        a0, b, c = unpack(sol.x)
        if np.any((c > lo2) & (c < hi2)):  # pole inside window -> reject
            continue
        rms = float(np.sqrt(np.mean(residual(sol.x) ** 2)))
        if rms < best_cost:
            best_cost = rms
            best = SellmeierModel(
                b=np.asarray(b), c=np.asarray(c), a0=float(a0),
                wl_range=stated_range, name=name, rms_error=rms,
            )
    if best is None:
        msg = "Sellmeier fit failed for all restarts; try fewer terms or a dataset."
        raise RuntimeError(msg)
    return best


@dataclass
class TemperatureSellmeier:
    """A temperature-dependent Sellmeier model fitted to multi-temperature data.

    Each Sellmeier coefficient is modelled as a polynomial in ``(T - T_ref)``,
    so ``n(lambda, T)`` is continuous in both wavelength and temperature.

    Attributes:
        models: the per-temperature fitted :class:`SellmeierModel` objects.
        temperatures: the temperatures [degC] of ``models``.
        t_ref: the reference temperature [degC] of the polynomial expansion.
        b_poly / c_poly: per-term polynomial coefficients (highest power first)
            for ``b_j(T)`` and ``c_j(T)``.
        a0_poly: polynomial for ``a0(T)``.
        wl_range: the common fitted validity range [um].
    """

    models: list[SellmeierModel]
    temperatures: np.ndarray
    t_ref: float
    b_poly: np.ndarray  # (num_terms, deg+1)
    c_poly: np.ndarray  # (num_terms, deg+1)
    a0_poly: np.ndarray  # (deg+1,)
    wl_range: tuple[float, float] | None = None
    name: str = "sellmeier(T)"

    @property
    def num_terms(self) -> int:
        """The number of Sellmeier terms."""
        return self.b_poly.shape[0]

    def at(self, temperature: float) -> SellmeierModel:
        """The :class:`SellmeierModel` evaluated at ``temperature`` [degC]."""
        dt = float(temperature) - self.t_ref
        b = np.array([np.polyval(p, dt) for p in self.b_poly])
        c = np.array([np.polyval(p, dt) for p in self.c_poly])
        a0 = float(np.polyval(self.a0_poly, dt))
        return SellmeierModel(
            b=b, c=c, a0=a0, wl_range=self.wl_range,
            name=f"{self.name}@{temperature}C",
        )

    def index(self, wls: np.ndarray, temperature: float) -> np.ndarray:
        """Refractive index at ``wls`` [um] and ``temperature`` [degC]."""
        return self.at(temperature).index(wls)


def fit_temperature_sellmeier(
    datasets: dict[float, tuple[np.ndarray, np.ndarray]],
    *,
    num_terms: int = 3,
    wl_range: tuple[float, float] | None = None,
    t_ref: float | None = None,
    t_degree: int = 1,
    name: str = "sellmeier(T)",
    **fit_kwargs: Any,
) -> TemperatureSellmeier:
    """Fit a temperature-dependent Sellmeier model from per-temperature datasets.

    Args:
        datasets: ``{temperature_degC: (wls_um, n)}`` index data at each
            measured temperature.
        num_terms: number of Sellmeier terms.
        wl_range: common fit/validity range [um].
        t_ref: reference temperature for the polynomial expansion (default: the
            mean of the dataset temperatures).
        t_degree: polynomial degree in ``(T - t_ref)`` for each coefficient
            (clamped to ``len(temperatures) - 1``).
        name: model name.
        **fit_kwargs: forwarded to :func:`fit_sellmeier`.

    Returns:
        The fitted :class:`TemperatureSellmeier`.
    """
    temps = np.array(sorted(datasets), dtype=float)
    if t_ref is None:
        t_ref = float(np.mean(temps))
    # fit each temperature to the SAME number of terms; align the poles by
    # sorting so b_j/c_j track the same oscillator across temperatures
    models = []
    for t in temps:
        wls, n = datasets[t]
        m = fit_sellmeier(wls, n, num_terms=num_terms, wl_range=wl_range, **fit_kwargs)
        idx = np.argsort(m.c)
        models.append(
            SellmeierModel(b=m.b[idx], c=m.c[idx], a0=m.a0, wl_range=m.wl_range,
                           name=f"{name}@{t}C", rms_error=m.rms_error)
        )
    deg = int(min(t_degree, len(temps) - 1))
    dt = temps - t_ref
    b_stack = np.array([m.b for m in models])  # (nT, num_terms)
    c_stack = np.array([m.c for m in models])
    a0_stack = np.array([m.a0 for m in models])
    b_poly = np.array([np.polyfit(dt, b_stack[:, j], deg) for j in range(num_terms)])
    c_poly = np.array([np.polyfit(dt, c_stack[:, j], deg) for j in range(num_terms)])
    a0_poly = np.polyfit(dt, a0_stack, deg)
    common_range = models[0].wl_range if wl_range is None else wl_range
    return TemperatureSellmeier(
        models=models, temperatures=temps, t_ref=t_ref,
        b_poly=b_poly, c_poly=c_poly, a0_poly=np.atleast_1d(a0_poly),
        wl_range=common_range, name=name,
    )


# ==========================================================================
# meow material builders
# ==========================================================================
def _as_index(source: Any, wls: np.ndarray) -> np.ndarray:
    """Evaluate a source (RIEntry / SellmeierModel / (wls, n) / callable) at wls."""
    if isinstance(source, RIEntry):
        return source.index(wls)
    if isinstance(source, SellmeierModel):
        return source.index(wls).astype(complex)
    if callable(source):
        return np.asarray(source(wls), dtype=complex)
    if isinstance(source, tuple) and len(source) == 2:
        s_wls, s_n = (np.asarray(a) for a in source)
        nr = np.interp(wls, s_wls, np.real(s_n))
        ni = np.interp(wls, s_wls, np.imag(s_n))
        return nr + 1j * ni
    msg = (
        "source must be an RIEntry, SellmeierModel, (wls, n) tuple or callable; "
        f"got {type(source)}."
    )
    raise TypeError(msg)


def sellmeier_material(
    name: str, model: SellmeierModel, *, wls: np.ndarray | None = None, npts: int = 200
) -> SampledMaterial:
    """Build an isotropic :class:`meow.SampledMaterial` from a Sellmeier model.

    The analytic model is sampled on ``wls`` (default: a dense grid over the
    model's validity range) so meow can interpolate it dispersively.
    """
    from meow.materials import SampledMaterial

    if wls is None:
        if model.wl_range is None:
            msg = "pass wls (the Sellmeier model has no wl_range)."
            raise ValueError(msg)
        wls = np.linspace(*model.wl_range, npts)
    wls = np.asarray(wls, dtype=float)
    n = model.index(wls).astype(complex)
    import pandas as pd

    df = pd.DataFrame({"wl": wls, "nr": np.real(n), "ni": np.imag(n)})
    return SampledMaterial.from_df(name, df)


def material_from_ri(
    source: Any,
    *,
    name: str | None = None,
    wls: np.ndarray | None = None,
    fit_terms: int | None = None,
    wl_range: tuple[float, float] | None = None,
    npts: int = 200,
) -> SampledMaterial:
    """Build an isotropic :class:`meow.SampledMaterial` from an RI source.

    Args:
        source: an :class:`RIEntry` (e.g. from :func:`fetch_ri_entry`), a
            :class:`SellmeierModel`, a ``(wls, n)`` dataset, or a callable
            ``n(wls)``.
        name: material name (defaults to the entry/model name).
        wls: wavelengths [um] to sample on (default: a dense grid over the
            source's range).
        fit_terms: if given, first fit an ``N``-term Sellmeier model to the
            source over ``wl_range`` and build the material from that fit.
        wl_range: validity range [um] for the fit / the sampling grid.
        npts: number of sample points when ``wls`` is not given.

    Returns:
        A dispersive isotropic :class:`meow.SampledMaterial`.
    """
    if wls is None:
        if wl_range is not None:
            wls = np.linspace(*wl_range, npts)
        elif isinstance(source, RIEntry):
            wls = source.default_wls(npts)
        elif isinstance(source, SellmeierModel) and source.wl_range:
            wls = np.linspace(*source.wl_range, npts)
        else:
            msg = "pass wls or wl_range to sample the material."
            raise ValueError(msg)
    wls = np.asarray(wls, dtype=float)
    label = name or getattr(source, "name", "material")
    if fit_terms is not None:
        n_grid = max(npts, 4 * fit_terms)
        grid = wls if wl_range is None else np.linspace(*wl_range, n_grid)
        n = np.real(_as_index(source, grid))
        model = fit_sellmeier(grid, n, num_terms=fit_terms, wl_range=wl_range,
                              name=label)
        return sellmeier_material(label, model, wls=wls)
    from meow.materials import SampledMaterial

    n = _as_index(source, wls)
    import pandas as pd

    df = pd.DataFrame({"wl": wls, "nr": np.real(n), "ni": np.imag(n)})
    return SampledMaterial.from_df(label, df)


def anisotropic_material(
    name: str,
    axes: dict[str, Any],
    *,
    wls: np.ndarray | None = None,
    fit_terms: int | None = None,
    wl_range: tuple[float, float] | None = None,
    npts: int = 200,
) -> SampledAnisotropicMaterial:
    """Build a :class:`meow.SampledAnisotropicMaterial` from per-axis RI sources.

    Args:
        name: material name.
        axes: ``{"x": src_x, "y": src_y, "z": src_z}`` where each source is an
            :class:`RIEntry`, :class:`SellmeierModel`, ``(wls, n)`` dataset or
            callable. A uniaxial crystal can give two axes the same source
            (ordinary) and one a different source (extraordinary).
        wls: common wavelength grid [um] (default: dense grid over ``wl_range``).
        fit_terms: if given, fit each axis to an ``N``-term Sellmeier first.
        wl_range: validity range [um].
        npts: number of sample points when ``wls`` is not given.

    Returns:
        A dispersive :class:`meow.SampledAnisotropicMaterial`.
    """
    from meow.materials import SampledAnisotropicMaterial

    if set(axes) != {"x", "y", "z"}:
        msg = f"axes must have keys x, y, z; got {sorted(axes)}."
        raise ValueError(msg)
    if wls is None:
        if wl_range is not None:
            wls = np.linspace(*wl_range, npts)
        else:
            ranges = [
                s.wl_range or (s.default_wls()[0], s.default_wls()[-1])
                for s in axes.values()
                if isinstance(s, (RIEntry, SellmeierModel)) and
                (s.wl_range or isinstance(s, RIEntry))
            ]
            if not ranges:
                msg = "pass wls or wl_range to sample the anisotropic material."
                raise ValueError(msg)
            lo = max(r[0] for r in ranges)
            hi = min(r[1] for r in ranges)
            wls = np.linspace(lo, hi, npts)
    wls = np.asarray(wls, dtype=float)

    cols = []
    for ax in ("x", "y", "z"):
        src = axes[ax]
        if fit_terms is not None:
            grid = wls if wl_range is None else np.linspace(
                *wl_range, max(npts, 4 * fit_terms)
            )
            n_grid = np.real(_as_index(src, grid))
            src = fit_sellmeier(grid, n_grid, num_terms=fit_terms,
                                wl_range=wl_range, name=f"{name}-{ax}")
        cols.append(_as_index(src, wls))
    n_diag = np.stack(cols, axis=-1)  # (N, 3) = (n_xx, n_yy, n_zz)
    return SampledAnisotropicMaterial.from_n(name, wls, n_diag)


def temperature_material(
    name: str,
    model: TemperatureSellmeier,
    *,
    wls: np.ndarray | None = None,
    temperatures: np.ndarray | None = None,
    npts: int = 80,
    n_temps: int = 11,
) -> SampledMaterial:
    """Build a ``(wavelength, temperature)``-dispersive :class:`meow.SampledMaterial`.

    Samples ``n(lambda, T)`` from a :class:`TemperatureSellmeier` on a
    wavelength x temperature grid; meow then interpolates in both the
    environment wavelength ``wl`` and temperature ``T``.

    Args:
        name: material name.
        model: the fitted :class:`TemperatureSellmeier`.
        wls: wavelength grid [um] (default: dense grid over the model range).
        temperatures: temperature grid [degC] (default: spanning the fitted data).
        npts: number of wavelength samples when ``wls`` is not given.
        n_temps: number of temperature samples when ``temperatures`` is not given.
    """
    from meow.materials import SampledMaterial

    if wls is None:
        if model.wl_range is None:
            msg = "pass wls (the model has no wl_range)."
            raise ValueError(msg)
        wls = np.linspace(*model.wl_range, npts)
    if temperatures is None:
        temperatures = np.linspace(
            float(model.temperatures.min()), float(model.temperatures.max()), n_temps
        )
    wls = np.asarray(wls, dtype=float)
    temperatures = np.asarray(temperatures, dtype=float)
    WL, T = np.meshgrid(wls, temperatures, indexing="ij")
    NR = np.empty_like(WL)
    for k, t in enumerate(temperatures):
        NR[:, k] = np.real(model.index(wls, float(t)))
    import pandas as pd

    df = pd.DataFrame(
        {"wl": WL.ravel(), "T": T.ravel(), "nr": NR.ravel(), "ni": np.zeros(WL.size)}
    )
    return SampledMaterial.from_df(name, df)
