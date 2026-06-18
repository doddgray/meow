"""Label waveguide modes by their Hermite-Gaussian character.

A guided mode of a (weakly guiding) rectangular waveguide closely resembles an
elliptical Hermite-Gaussian (HG) beam: a product of a Hermite polynomial of
order ``m`` along the horizontal axis and order ``n`` along the vertical axis,
multiplied by a Gaussian envelope.  The dominant *transverse electric* field
component is ``Ex`` for quasi-TE modes and ``Ey`` for quasi-TM modes.

This module labels a mode by the ``(polarization, m, n)`` of the HG template
that best fits its field.  For each candidate the HG template is centered at the
field centroid and given widths matching the field's variance along each axis
(order-corrected), and these four parameters ``(x0, y0, wx, wy)`` are optionally
refined to maximize the overlap with the mode.  The label is the candidate with
the smallest normalized squared error

    error = 1 - |<HG, F>|^2 / (||HG||^2 . (||Ex||^2 + ||Ey||^2))

where ``F`` is ``Ex`` for a TE candidate and ``Ey`` for a TM candidate.  Because
the overlap is normalized by the *total* transverse energy, TE and TM candidates
compete fairly: a quasi-TM mode (small ``Ex``) cannot score well against a TE
template.  The complex amplitude is projected out analytically, so the error is
independent of the mode's global phase.

The error itself is a useful confidence metric: a clean low-order guided mode
fits with ``error < ~0.1``, whereas near-cutoff, leaky, or strongly hybrid modes
fit poorly and are easy to flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite

if TYPE_CHECKING:
    from meow.arrays import ComplexArray2D, FloatArray2D
    from meow.mode import Mode, Modes

Polarization = Literal["TE", "TM"]


def _hermite_gaussian_1d(x: np.ndarray, order: int, x0: float, w: float) -> np.ndarray:
    """Evaluate an (unnormalized) 1D Hermite-Gaussian of a given order.

    ``u_m(x) = H_m(sqrt(2) (x - x0) / w) exp(-(x - x0)^2 / w^2)``

    with ``H_m`` the physicists' Hermite polynomial.  ``w`` is the ``1/e`` field
    radius of the fundamental (``m = 0``) Gaussian envelope.
    """
    w = max(float(w), 1e-12)
    xi = np.sqrt(2.0) * (x - x0) / w
    return eval_hermite(order, xi) * np.exp(-(((x - x0) / w) ** 2))


def hermite_gaussian_field(
    x: np.ndarray,
    y: np.ndarray,
    m: int,
    n: int,
    x0: float,
    y0: float,
    wx: float,
    wy: float,
) -> FloatArray2D:
    """Build a 2D elliptical Hermite-Gaussian field on a separable grid.

    Args:
        x: 1D array of x-coordinates (length ``Nx``).
        y: 1D array of y-coordinates (length ``Ny``).
        m: Hermite order along x (number of horizontal nodes).
        n: Hermite order along y (number of vertical nodes).
        x0: beam center along x.
        y0: beam center along y.
        wx: ``1/e`` field radius of the fundamental envelope along x.
        wy: ``1/e`` field radius of the fundamental envelope along y.

    Returns:
        A real ``(Nx, Ny)`` array ``HG_{m,n}(x, y)`` (indexed ``[x, y]``).
    """
    ux = _hermite_gaussian_1d(np.asarray(x, dtype=float), int(m), float(x0), float(wx))
    uy = _hermite_gaussian_1d(np.asarray(y, dtype=float), int(n), float(y0), float(wy))
    return ux[:, None] * uy[None, :]


@dataclass(frozen=True)
class ModeLabel:
    """The Hermite-Gaussian label assigned to a `Mode`.

    Attributes:
        pol: the fitted polarization, ``"TE"`` (Ex-dominant) or ``"TM"``
            (Ey-dominant).
        m: the fitted Hermite order along the horizontal (x) axis.
        n: the fitted Hermite order along the vertical (y) axis.
        error: the normalized squared error of the best fit (0 = perfect,
            1 = no overlap).  Doubles as a confidence metric.
        overlap: the normalized squared overlap ``1 - error`` of the best fit.
        x0: fitted beam center along x [um].
        y0: fitted beam center along y [um].
        wx: fitted ``1/e`` field radius along x [um].
        wy: fitted ``1/e`` field radius along y [um].
    """

    pol: Polarization
    m: int
    n: int
    error: float
    overlap: float
    x0: float
    y0: float
    wx: float
    wy: float

    @property
    def name(self) -> str:
        """A compact label such as ``"TE00"`` or ``"TM21"``."""
        return f"{self.pol}{self.m}{self.n}"

    def __str__(self) -> str:
        return f"{self.name} (error={self.error:.3g})"


def _integrate2d(arr: np.ndarray, x: np.ndarray, y: np.ndarray) -> complex:
    """Integrate a 2D array (indexed ``[x, y]``) over the x and y coordinates."""
    return np.trapezoid(np.trapezoid(arr, y, axis=1), x, axis=0)


def _component_grid(
    mode: Mode, pol: Polarization
) -> tuple[ComplexArray2D, np.ndarray, np.ndarray]:
    """Return ``(field, x, y)`` for the dominant transverse field of a pol."""
    if pol == "TE":
        return np.asarray(mode.Ex), mode.mesh.Xx[:, 0], mode.mesh.Yx[0, :]
    return np.asarray(mode.Ey), mode.mesh.Xy[:, 0], mode.mesh.Yy[0, :]


def _moments(
    field: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, float, float]:
    """Intensity-weighted centroid, std and total energy of a field component.

    Returns ``(x0, y0, sigma_x, sigma_y, energy)`` with ``energy = ||field||^2``.
    """
    intensity = np.abs(field) ** 2
    energy = float(np.real(_integrate2d(intensity, x, y)))
    if energy <= 0:
        return float(np.mean(x)), float(np.mean(y)), 0.0, 0.0, 0.0
    x0 = float(np.real(_integrate2d(intensity * x[:, None], x, y)) / energy)
    y0 = float(np.real(_integrate2d(intensity * y[None, :], x, y)) / energy)
    mx = _integrate2d(intensity * (x[:, None] - x0) ** 2, x, y)
    my = _integrate2d(intensity * (y[None, :] - y0) ** 2, x, y)
    varx = float(np.real(mx) / energy)
    vary = float(np.real(my) / energy)
    return x0, y0, np.sqrt(max(varx, 0.0)), np.sqrt(max(vary, 0.0)), energy


def _fit_error(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    m: int,
    n: int,
    x0: float,
    y0: float,
    wx: float,
    wy: float,
    total_energy: float,
) -> float:
    """Normalized squared error of an HG template against a field component.

    The optimal complex amplitude is projected out analytically, so the result
    is invariant to the global phase of ``field``.
    """
    hg = hermite_gaussian_field(x, y, m, n, x0, y0, wx, wy)
    hg_norm = float(np.real(_integrate2d(hg * hg, x, y)))
    if hg_norm <= 0 or total_energy <= 0:
        return 1.0
    overlap = _integrate2d(hg * field, x, y)
    overlap_sq = float(np.abs(overlap) ** 2 / (hg_norm * total_energy))
    return float(np.clip(1.0 - overlap_sq, 0.0, 1.0))


def _width_guess(sigma: float, order: int) -> float:
    """Variance-matched envelope width for an HG of a given order.

    A pure ``HG_m`` has intensity variance ``(m + 1/2) w^2 / 2``.  Inverting,
    ``w = sigma sqrt(2 / (m + 1/2))`` matches the measured variance ``sigma^2``.
    For ``m = 0`` this reduces to the usual ``w = 2 sigma``.
    """
    return float(sigma) * np.sqrt(2.0 / (order + 0.5)) if sigma > 0 else 1.0


def _optimize_fit(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    m: int,
    n: int,
    guess: tuple[float, float, float, float],
    total_energy: float,
) -> tuple[float, tuple[float, float, float, float]]:
    """Refine ``(x0, y0, wx, wy)`` to minimize the HG fit error (Nelder-Mead).

    Widths are optimized in log-space to stay positive.  Falls back to the
    initial guess if the optimizer does not improve on it.
    """
    x0, y0, wx, wy = guess
    p0 = np.array([x0, y0, np.log(max(wx, 1e-6)), np.log(max(wy, 1e-6))])

    # keep log-widths in a sane range so Nelder-Mead can't overflow exp()
    log_lo, log_hi = np.log(1e-3), np.log(1e3)

    def objective(p: np.ndarray) -> float:
        lwx = float(np.clip(p[2], log_lo, log_hi))
        lwy = float(np.clip(p[3], log_lo, log_hi))
        return _fit_error(
            field, x, y, m, n, p[0], p[1], np.exp(lwx), np.exp(lwy), total_energy
        )

    err0 = objective(p0)
    try:
        res = minimize(
            objective,
            p0,
            method="Nelder-Mead",
            options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-6},
        )
    except Exception:  # noqa: BLE001 - optimization is best-effort
        return err0, guess
    if not np.isfinite(res.fun) or res.fun >= err0:
        return err0, guess
    return float(res.fun), (
        float(res.x[0]),
        float(res.x[1]),
        float(np.exp(np.clip(res.x[2], log_lo, log_hi))),
        float(np.exp(np.clip(res.x[3], log_lo, log_hi))),
    )


def label_mode_candidates(
    mode: Mode,
    *,
    max_order_x: int = 4,
    max_order_y: int = 3,
    optimize: bool = True,
) -> list[ModeLabel]:
    """Score every Hermite-Gaussian candidate against a mode, best first.

    Args:
        mode: the mode to label.
        max_order_x: largest horizontal Hermite order to try (inclusive).
        max_order_y: largest vertical Hermite order to try (inclusive).
        optimize: refine the HG center and widths of each candidate to minimize
            the fit error (recommended).  If ``False`` only the variance-matched
            initial guess is used.

    Returns:
        A list of `ModeLabel`, sorted by increasing ``error``.
    """
    Ex, xx, yx = _component_grid(mode, "TE")
    Ey, xy, yy = _component_grid(mode, "TM")
    _, _, _, _, te_energy = _moments(Ex, xx, yx)
    _, _, _, _, tm_energy = _moments(Ey, xy, yy)
    total_energy = te_energy + tm_energy

    candidates: list[ModeLabel] = []
    for pol, field, x, y in (("TE", Ex, xx, yx), ("TM", Ey, xy, yy)):
        x0, y0, sx, sy, _ = _moments(field, x, y)
        for m in range(max_order_x + 1):
            for n in range(max_order_y + 1):
                guess = (x0, y0, _width_guess(sx, m), _width_guess(sy, n))
                if optimize:
                    err, (fx0, fy0, fwx, fwy) = _optimize_fit(
                        field, x, y, m, n, guess, total_energy
                    )
                else:
                    err = _fit_error(field, x, y, m, n, *guess, total_energy)
                    fx0, fy0, fwx, fwy = guess
                candidates.append(
                    ModeLabel(
                        pol=pol,  # type: ignore[arg-type]
                        m=m,
                        n=n,
                        error=err,
                        overlap=1.0 - err,
                        x0=fx0,
                        y0=fy0,
                        wx=fwx,
                        wy=fwy,
                    )
                )
    candidates.sort(key=lambda c: c.error)
    return candidates


def label_mode(
    mode: Mode,
    *,
    max_order_x: int = 4,
    max_order_y: int = 3,
    optimize: bool = True,
) -> ModeLabel:
    """Label a mode by its best-fitting Hermite-Gaussian character.

    Args:
        mode: the mode to label.
        max_order_x: largest horizontal Hermite order to try (inclusive).
        max_order_y: largest vertical Hermite order to try (inclusive).
        optimize: refine each candidate's center and widths (recommended).

    Returns:
        The best-fitting `ModeLabel`.  Its ``error`` attribute indicates how
        cleanly the mode matches an HG profile.
    """
    return label_mode_candidates(
        mode,
        max_order_x=max_order_x,
        max_order_y=max_order_y,
        optimize=optimize,
    )[0]


def label_modes(
    modes: Modes,
    *,
    max_order_x: int = 4,
    max_order_y: int = 3,
    optimize: bool = True,
) -> list[ModeLabel]:
    """Label every mode in a list. See :func:`label_mode`."""
    return [
        label_mode(
            m,
            max_order_x=max_order_x,
            max_order_y=max_order_y,
            optimize=optimize,
        )
        for m in modes
    ]


def filter_modes_by_label(
    modes: Modes,
    *,
    pol: Polarization | None = None,
    m: int | None = None,
    n: int | None = None,
    max_error: float = 1.0,
    max_order_x: int = 4,
    max_order_y: int = 3,
    optimize: bool = True,
) -> Modes:
    """Select modes whose Hermite-Gaussian label matches the given criteria.

    This makes it possible to pick out, e.g., only the fundamental TE mode
    (``pol="TE", m=0, n=0``) of a multimode waveguide before building an EME
    model, so the simulation can be restricted to specific modes.

    Args:
        modes: the modes to filter.
        pol: required polarization (``"TE"``/``"TM"``), or ``None`` for any.
        m: required horizontal order, or ``None`` for any.
        n: required vertical order, or ``None`` for any.
        max_error: discard modes whose best fit error exceeds this threshold
            (use this to reject poorly-confined / hybrid modes).
        max_order_x: largest horizontal Hermite order to try (inclusive).
        max_order_y: largest vertical Hermite order to try (inclusive).
        optimize: refine each candidate's center and widths (recommended).

    Returns:
        The subset of ``modes`` matching the criteria, in their original order.
    """
    selected: list[Mode] = []
    for mode in modes:
        label = label_mode(
            mode,
            max_order_x=max_order_x,
            max_order_y=max_order_y,
            optimize=optimize,
        )
        if label.error > max_error:
            continue
        if pol is not None and label.pol != pol:
            continue
        if m is not None and label.m != m:
            continue
        if n is not None and label.n != n:
            continue
        selected.append(mode)
    return selected
