"""Generalized adiabatic dichroic beam-splitter designer (after Magden 2018).

This example generalizes the silicon dichroic filter of Magden et al.,
Nat. Commun. 9, 3009 (2018) (reproduced in ``magden2018_dichroic.py``) into a
**platform-parametric designer**: given a single waveguide layer - the core
and cladding materials, the core thickness, the sidewall angle, the partial
etch fraction (if not fully etched), the minimum fabricable tip width and gap,
and a maximum device length - it designs and optimizes adiabatic dichroic
beam splitters with a **targeted cutoff wavelength**.

Design principle (same as the paper):

- A solid strip waveguide ``WGA`` is laterally coupled to a sub-wavelength
  multi-rail waveguide ``WGB`` (``n_rails`` narrow rails on a common pitch).
- Because the multi-rail ``WGB`` is less dispersive than the solid ``WGA``,
  their effective indices ``n_WGA(lambda)`` and ``n_WGB(lambda)`` cross at
  **exactly one** wavelength - the filter cutoff ``lambda_c``.
- The cutoff is set by the ``WGA`` width: wider ``WGA`` -> higher index ->
  longer cutoff. The designer root-finds the width that phase-matches the two
  waveguides at the target cutoff, ``n_WGA(w_a, lambda_c) = n_WGB(lambda_c)``.
- The device then routes ``lambda < lambda_c`` into ``WGA`` (short pass) and
  ``lambda > lambda_c`` into ``WGB`` (long pass) by adiabatic mode evolution.

Optimization: the coupling ``kappa`` (FDE coupled-mode overlap) and the
``d n_WGA / d w`` slope set, through the Landau-Zener criterion, the
phase-matching taper length needed to keep the diabatic jump below a target.
Within the ``max_length`` budget the designer picks the **largest coupling gap**
(sharpest cutoff) whose required taper still fits, and reports the predicted
extinction. The resulting dimensions feed the parametric layout of
``magden2018_dichroic.dichroic_filter``; this module adds the platform-aware
extrusion so the device can be meshed and EME-simulated on any layer stack.

Run with ``python -m examples.papers.dichroic_designer``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import gdsfactory as gf
import numpy as np
from scipy.optimize import brentq

import meow as mw
from examples.papers import _resolution
from examples.papers.magden2018_dichroic import LAYER_WG, dichroic_filter

if TYPE_CHECKING:
    from examples.papers._ad_optimize import OptimizationTrace

FIGDIR = Path(__file__).parent / "figures"
pick = _resolution.pick

SCALAR_KAPPA_CORRECTION = 0.4
"""Empirical correction for the scalar overlap's high-index-contrast
overestimate of the coupling, calibrated so that the silicon Magden 2018
coupling (~5/mm at the 750 nm gap) is recovered. Lower-contrast platforms are
overestimated less, so the corrected ``|kappa|`` (and the extinction estimate
derived from it) is approximate - validate a final design with a full EME."""


# --------------------------------------------------------------------------
# platform + sub-wavelength WGB specification
# --------------------------------------------------------------------------
@dataclass
class Platform:
    """A single etched waveguide layer and its fabrication limits.

    Args:
        core: the (high-index) core material.
        clad: the surrounding cladding material (used above and below the core).
        core_thickness: the core layer thickness [um].
        sidewall_deg: rib sidewall angle, measured from vertical [degrees].
        etch_fraction: fraction of the core etched away outside the ribs
            (1.0 = fully etched, <1.0 leaves a slab of ``(1 - f) * thickness``).
        min_tip: minimum fabricable feature/tip width [um].
        min_gap: minimum fabricable gap [um].
        max_length: maximum allowed device length [um].
        clad_thickness: modeled cladding thickness above/below the core [um].
    """

    core: mw.Material
    clad: mw.Material
    core_thickness: float
    sidewall_deg: float = 0.0
    etch_fraction: float = 1.0
    min_tip: float = 0.05
    min_gap: float = 0.10
    max_length: float = 2000.0
    clad_thickness: float = 1.0

    @property
    def slab_thickness(self) -> float:
        """Remaining core slab thickness (0 if fully etched)."""
        return self.core_thickness * (1.0 - self.etch_fraction)


@dataclass
class WGB:
    """A sub-wavelength multi-rail (segmented) WGB cross-section.

    ``frac_mid``/``frac_out`` scale the central/outer rail widths as fractions
    of ``rail_width`` (only meaningful for ``n_rails == 3``; ``1.0`` recovers
    the uniform-rail-width WGB).
    """

    rail_width: float
    gap: float
    n_rails: int = 3
    frac_mid: float = 1.0
    frac_out: float = 1.0

    @property
    def widths(self) -> list[float]:
        """Per-rail widths (heterogeneous mid/outer widths for 3 rails)."""
        if self.n_rails == 3:
            mid, out = self.frac_mid * self.rail_width, self.frac_out * self.rail_width
            return [out, mid, out]
        return [self.rail_width] * self.n_rails

    @property
    def total_width(self) -> float:
        """Edge-to-edge width of the multi-rail WGB."""
        return float(sum(self.widths) + (self.n_rails - 1) * self.gap)

    def centers(self, x0: float = 0.0) -> list[float]:
        """Lateral rail centres, symmetric about ``x0``."""
        widths = self.widths
        pos = x0 - self.total_width / 2
        centers = []
        for w in widths:
            centers.append(pos + w / 2)
            pos += w + self.gap
        return centers


# --------------------------------------------------------------------------
# cross-sections + FDE effective indices
# --------------------------------------------------------------------------
def _ridge_structures(
    platform: Platform,
    widths: list[float],
    centers: list[float],
    x_span: tuple[float, float],
) -> list[mw.Structure3D]:
    """Rib structures (ridges + optional slab + cladding) on the platform."""
    h = platform.core_thickness
    h_slab = platform.slab_thickness
    run = (h - h_slab) * np.tan(np.deg2rad(platform.sidewall_deg))
    structs: list[mw.Structure3D] = []
    for w, x0 in zip(widths, centers, strict=True):
        base_half = 0.5 * (w + 2 * run)  # drawn poly = base; top width = w
        poly = np.array(
            [
                (0.0, x0 - base_half),
                (1.0, x0 - base_half),
                (1.0, x0 + base_half),
                (0.0, x0 + base_half),
            ]
        )
        structs.append(
            mw.Structure(
                material=platform.core,
                geometry=mw.Prism(
                    poly=poly,
                    h_min=h_slab,
                    h_max=h,
                    axis="y",
                    sidewall_angle=platform.sidewall_deg,
                ),
                mesh_order=1,
            )
        )
    if h_slab > 1e-9:
        structs.append(
            mw.Structure(
                material=platform.core,
                geometry=mw.Box(
                    x_min=x_span[0],
                    x_max=x_span[1],
                    y_min=0.0,
                    y_max=h_slab,
                    z_min=0.0,
                    z_max=1.0,
                ),
                mesh_order=2,
            )
        )
    structs.append(
        mw.Structure(
            material=platform.clad,
            geometry=mw.Box(
                x_min=x_span[0],
                x_max=x_span[1],
                y_min=-platform.clad_thickness,
                y_max=h + platform.clad_thickness,
                z_min=0.0,
                z_max=1.0,
            ),
            mesh_order=10,
        )
    )
    return structs


def _mesh(platform: Platform, x_min: float, x_max: float, res: float) -> mw.Mesh2D:
    h = platform.core_thickness
    tcl = platform.clad_thickness
    return mw.Mesh2D(
        x=np.arange(x_min, x_max + res / 2, res),
        y=np.arange(-tcl, h + tcl + res / 2, res),
    )


def _te_modes(
    structures: list[mw.Structure3D],
    wl: float,
    mesh: mw.Mesh2D,
    num_modes: int = 8,
    compute_modes: Callable | None = None,
) -> list[mw.Mode]:
    compute_modes = compute_modes or mw.compute_modes
    cell = mw.Cell(structures=structures, mesh=mesh, z_min=0.0, z_max=1.0)
    cs = mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=wl, T=25.0))
    modes = compute_modes(cs, num_modes=num_modes)
    return [m for m in modes if m.te_fraction > 0.5] or list(modes)


def solid_neff(
    platform: Platform,
    w_a: float,
    wl: float,
    res: float = 0.03,
    compute_modes: Callable | None = None,
) -> float:
    """Fundamental TE effective index of an isolated solid WGA strip."""
    span = max(2.5, 3 * w_a)
    mesh = _mesh(platform, -span, span, res)
    structs = _ridge_structures(platform, [w_a], [0.0], (-span, span))
    return float(
        np.real(
            _te_modes(structs, wl, mesh, compute_modes=compute_modes)[0].neff
        )
    )


def segmented_neff(
    platform: Platform,
    wgb: WGB,
    wl: float,
    x0: float = 0.0,
    res: float = 0.03,
    compute_modes: Callable | None = None,
) -> float:
    """Fundamental TE effective index of the isolated multi-rail WGB."""
    span = max(2.5, x0 + 1.5 * wgb.total_width)
    mesh = _mesh(platform, -span, span, res)
    structs = _ridge_structures(platform, wgb.widths, wgb.centers(x0), (-span, span))
    return float(
        np.real(
            _te_modes(structs, wl, mesh, compute_modes=compute_modes)[0].neff
        )
    )


# --------------------------------------------------------------------------
# the design: solve the WGA width for a targeted cutoff
# --------------------------------------------------------------------------
def phase_match_width(
    platform: Platform,
    cutoff_wl: float,
    wgb: WGB,
    w_bracket: tuple[float, float] = (0.1, 1.2),
    res: float = 0.04,
    compute_modes: Callable | None = None,
) -> float:
    """Solid WGA width that phase-matches WGB at ``cutoff_wl`` [um].

    Root-finds ``n_WGA(w_a, cutoff_wl) = n_WGB(cutoff_wl)``; ``n_WGA`` increases
    monotonically with width, so the crossing is unique.
    """
    n_b = segmented_neff(platform, wgb, cutoff_wl, res=res, compute_modes=compute_modes)

    def mismatch(w_a: float) -> float:
        return (
            solid_neff(platform, w_a, cutoff_wl, res=res, compute_modes=compute_modes)
            - n_b
        )

    lo, hi = w_bracket
    f_lo, f_hi = mismatch(lo), mismatch(hi)
    # expand the bracket if needed (keep tips fabricable, cap the width)
    for _ in range(6):
        if f_lo < 0 < f_hi:
            break
        if f_lo >= 0:
            lo = max(2 * platform.min_tip, lo / 1.5)
            f_lo = mismatch(lo)
        if f_hi <= 0:
            hi = min(3.0, hi * 1.4)
            f_hi = mismatch(hi)
    if not (f_lo < 0 < f_hi):
        msg = (
            f"cutoff {cutoff_wl * 1e3:.0f} nm not reachable for this WGB on this "
            f"platform within WGA widths [{lo:.3f}, {hi:.3f}] um."
        )
        raise ValueError(msg)
    return float(brentq(mismatch, lo, hi, xtol=2e-4, rtol=1e-4))


def _solid_solve_and_cs(
    platform: Platform, w_a: float, res: float
) -> tuple[list[mw.Structure3D], mw.Mesh2D]:
    """Shared structures/mesh builder for the AD forward and eps-Jacobian paths."""
    span = max(2.5, 3 * w_a)
    mesh = _mesh(platform, -span, span, res)
    structs = _ridge_structures(platform, [w_a], [0.0], (-span, span))
    return structs, mesh


def optimize_phase_match_width(
    platform: Platform,
    cutoff_wl: float,
    wgb: WGB,
    *,
    w0: float = 0.4,
    res: float = 0.04,
    steps: int = 25,
    lr: float = 0.05,
    w_bounds: tuple[float, float] = (0.1, 1.5),
) -> tuple[float, OptimizationTrace]:
    """Gradient-based (AD) alternative to :func:`phase_match_width`.

    Minimizes the phase-mismatch loss ``(n_WGA(w_a, cutoff_wl) - n_WGB(cutoff_wl))^2``
    over the *fixed* layer stack (``platform``, ``wgb``) via ``jax.grad`` +
    projected Adam, using :func:`meow.make_differentiable_neffs` for an exact,
    single-solve-per-iteration gradient (meow's tidy3d cross-section builder
    already applies Kottke subpixel smoothing, so the width -> eps map is smooth
    enough for the default finite-difference eps-Jacobian). Where
    :func:`phase_match_width` root-finds the crossing, this reaches the same
    optimum by descending an explicit loss - useful as a template for objectives
    that are not simple 1D root-finds (e.g. multiple simultaneous targets).

    Returns:
        ``(w_a_opt, trace)`` - the optimized width and the optimization trace
        (see :mod:`examples.papers._ad_optimize`).
    """
    import jax.numpy as jnp

    from examples.papers._ad_optimize import adam_minimize

    n_b = segmented_neff(platform, wgb, cutoff_wl, res=res)

    def solve(params: np.ndarray) -> list[list[mw.Mode]]:
        structs, mesh = _solid_solve_and_cs(platform, float(params[0]), res)
        return [_te_modes(structs, cutoff_wl, mesh)[:1]]

    def cross_sections(params: np.ndarray) -> list[mw.CrossSection]:
        structs, mesh = _solid_solve_and_cs(platform, float(params[0]), res)
        cell = mw.Cell(structures=structs, mesh=mesh, z_min=0.0, z_max=1.0)
        return [mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=cutoff_wl))]

    f = mw.make_differentiable_neffs(solve, shape=(1, 1), cross_sections=cross_sections)

    def loss_fn(params: object) -> object:
        n_a = jnp.real(f(params)[0, 0])
        return (n_a - n_b) ** 2

    w_opt, trace = adam_minimize(
        loss_fn,
        [w0],
        steps=steps,
        lr=lr,
        bounds=[w_bounds],
        param_names=("w_a [um]",),
        objective_name="phase-match loss $(n_a - n_b)^2$",
    )
    return float(w_opt[0]), trace


def coupling_kappa(
    platform: Platform,
    w_a: float,
    wgb: WGB,
    gap: float,
    wl: float,
    res: float = 0.03,
    compute_modes: Callable | None = None,
) -> float:
    """Approximate coupled-mode coupling ``|kappa|`` [1/um] at edge gap ``gap``.

    Scalar coupled-mode overlap integral between the isolated WGA and WGB modes
    placed at their coupled positions. This is self-contained (no calibration)
    but the high-index-contrast scalar overlap is only approximate; it sets the
    *order of magnitude* of the coupling for the length budget below.
    """
    x0_b = w_a / 2 + gap + wgb.total_width / 2
    span = max(2.5, x0_b + 1.5 * wgb.total_width)
    mesh = _mesh(platform, -span, span, res)
    mode_a = _te_modes(
        _ridge_structures(platform, [w_a], [0.0], (-span, span)),
        wl,
        mesh,
        compute_modes=compute_modes,
    )[0]
    mode_b = _te_modes(
        _ridge_structures(platform, wgb.widths, wgb.centers(x0_b), (-span, span)),
        wl,
        mesh,
        compute_modes=compute_modes,
    )[0]
    k0 = 2 * np.pi / wl
    n_clad2 = float(np.real(mode_a.cs.nx[0, 0]) ** 2)
    pert_a = np.clip(np.real(mode_a.cs.nx**2) - n_clad2, 0.0, None)
    pert_b = np.clip(np.real(mode_b.cs.nx**2) - n_clad2, 0.0, None)
    ex_a, ex_b = np.real(mode_a.Ex), np.real(mode_b.Ex)
    kappa_ab = 0.5 * k0 * float(np.sum(pert_b * ex_a * ex_b)) / float(np.sum(ex_a**2))
    kappa_ba = 0.5 * k0 * float(np.sum(pert_a * ex_a * ex_b)) / float(np.sum(ex_b**2))
    return float(SCALAR_KAPPA_CORRECTION * np.sqrt(abs(kappa_ab * kappa_ba)))


# --------------------------------------------------------------------------
# the optimization: lengths + gap within the device-length budget
# --------------------------------------------------------------------------
@dataclass
class DichroicDesign:
    """A designed dichroic beam splitter for one targeted cutoff."""

    platform: Platform
    cutoff_wl: float
    wgb: WGB
    w_a: float
    gap: float
    lengths: tuple[float, float, float, float]
    kappa: float
    dn_dw: float
    extinction_db: float
    component: gf.Component = field(repr=False)
    opt_trace: OptimizationTrace | None = field(default=None, repr=False)
    """The AD optimization trace (see :mod:`examples.papers._ad_optimize`) that
    produced ``w_a``, when built with ``design_dichroic(..., use_gradient=True)``;
    ``None`` for the (default) root-find path."""

    @property
    def total_length(self) -> float:
        """Total device length [um]."""
        return float(sum(self.lengths))


def _taper_extinction(
    kappa: float, dn_dw: float, w_a: float, w_tip: float, wl: float, l2: float
) -> float:
    """Predicted short/long-pass extinction [dB] from the Landau-Zener jump.

    As WGA tapers tip->full over the phase-matching length ``l2`` the detuning
    ``delta = 0.5 k0 (n_WGA - n_WGB)`` sweeps through zero at rate
    ``alpha = 0.5 k0 (dn/dw)(w_a - w_tip)/l2``. The diabatic jump probability is
    ``P = exp(-2 pi kappa^2 / alpha)`` and the extinction is ``-10 log10 P``.
    """
    k0 = 2 * np.pi / wl
    alpha = 0.5 * k0 * dn_dw * max(w_a - w_tip, 1e-6) / max(l2, 1e-6)
    p_jump = float(np.exp(-2 * np.pi * kappa**2 / max(alpha, 1e-30)))
    p_jump = min(max(p_jump, 1e-12), 1.0 - 1e-12)
    return float(-10 * np.log10(p_jump))


def _allocate_lengths(
    platform: Platform, gap: float, gap_out: float
) -> tuple[float, float, float, float]:
    """Split ``max_length`` into the four adiabatic sections.

    Small fixed input/output (segmentation) sections; the lateral separation
    ``L3`` is set by a gentle (~1 degree) bend of the WGA centre to the final
    gap; the rest of the budget goes to the critical phase-matching taper L2.
    """
    l_io = max(20.0, 0.05 * platform.max_length)
    shift = gap_out - gap  # lateral WGA displacement in section 3
    l3 = float(
        np.clip(shift / np.tan(np.deg2rad(1.0)), 50.0, 0.6 * platform.max_length)
    )
    l2 = platform.max_length - 2 * l_io - l3
    if l2 < l_io:  # budget too small: share the remainder
        l2 = l3 = max(l_io, 0.5 * (platform.max_length - 2 * l_io))
    return l_io, l2, l3, l_io


# --------------------------------------------------------------------------
# joint AD optimization over the full practical design-parameter set
# --------------------------------------------------------------------------
JOINT_PARAM_NAMES: tuple[str, ...] = (
    "w_a [um]",
    "w_b [um]",
    "gap [um]",
    "g_b [um]",
    "frac_mid",
    "frac_out",
    "l1 [um]",
    "l2 [um]",
    "l3 [um]",
    "l4 [um]",
)
"""Parameter order of :func:`optimize_dichroic_joint`."""


def _joint_geometric_quantities(
    platform: Platform,
    cutoff_wl: float,
    res: float,
    w_a: float,
    w_b: float,
    gap: float,
    g_b: float,
    frac_mid: float,
    frac_out: float,
    compute_modes: Callable | None = None,
) -> tuple[float, float, float, float]:
    """``(n_a, n_b, kappa, dn_dw)`` at one geometric point of the joint design."""
    wgb = WGB(rail_width=w_b, gap=g_b, n_rails=3, frac_mid=frac_mid, frac_out=frac_out)
    n_b = segmented_neff(platform, wgb, cutoff_wl, res=res, compute_modes=compute_modes)
    n_a = solid_neff(platform, w_a, cutoff_wl, res=res, compute_modes=compute_modes)
    kappa = coupling_kappa(
        platform, w_a, wgb, gap, cutoff_wl, res=res, compute_modes=compute_modes
    )
    dw = 0.02
    n_hi = solid_neff(
        platform, w_a + dw, cutoff_wl, res=res, compute_modes=compute_modes
    )
    n_lo = solid_neff(
        platform, w_a - dw, cutoff_wl, res=res, compute_modes=compute_modes
    )
    dn_dw = (n_hi - n_lo) / (2 * dw)
    return n_a, n_b, kappa, dn_dw


JOINT_X0_DEFAULT: tuple[float, ...] = (
    0.60, 0.25, 0.75, 0.12, 1.0, 1.0, 200.0, 260.0, 900.0, 200.0,
)
"""Default (deliberately off-target) initial guess for
:func:`optimize_dichroic_joint`."""


def optimize_dichroic_joint(
    platform: Platform,
    cutoff_wl: float,
    *,
    x0: tuple[float, ...] = JOINT_X0_DEFAULT,
    bounds: tuple[tuple[float, float], ...] | None = None,
    target_extinction_db: float = 20.0,
    max_total_length: float = 5000.0,
    res: float = 0.05,
    steps: int = 26,
    lr: float = 0.025,
    compute_modes: Callable | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    """Joint AD-gradient optimization over the full practical parameter set.

    Minimizes a composite loss over the 10-parameter vector ``(w_a, w_b, gap,
    g_b, frac_mid, frac_out, l1, l2, l3, l4)`` (see :data:`JOINT_PARAM_NAMES`):
    the WGA/WGB full widths, the WGA-WGB coupling gap, the inter-rail gap of
    the three-ridge WGB, the fractional middle/outer WGB rail widths (as
    fractions of ``w_b``), and the four adiabatic section lengths, allowing up
    to ``max_total_length`` (5 mm by default). The loss combines:

    - the phase-match residual ``(n_WGA - n_WGB)^2`` at ``cutoff_wl``,
    - a soft hinge penalizing a Landau-Zener extinction (:func:`_taper_extinction`)
      below ``target_extinction_db``,
    - a small linear preference for a more compact (shorter total length)
      device - since the required extinction is met by *either* a smaller
      coupling gap (stronger, exponentially larger ``kappa``) or a longer
      phase-matching taper, this preference is what drives the gap towards
      the tightest coupling that still meets the extinction target within a
      short taper, rather than leaving it undetermined, and
    - a soft hinge penalizing a total length beyond ``max_total_length``.

    The loss is wrapped once with :func:`meow.make_differentiable_objective`
    (exact central finite differences of the whole FDE-based objective), and
    optimized by ``jax.grad`` + projected Adam
    (:func:`examples.papers._ad_optimize.adam_minimize`)
    in **bounds-normalized** coordinates, so a single learning rate is
    meaningful across the mixed micron/dimensionless/length-in-microns
    parameter scales. The four length parameters do not affect the FDE
    quantities, so their finite-difference steps are served from a cache keyed
    on the 6 geometric parameters - no extra eigensolves.

    Returns:
        ``(params_opt, trace)`` - the optimized parameter vector, in the
        :data:`JOINT_PARAM_NAMES` order, and the optimization trace (with
        ``trace.params`` de-normalized back to physical units).
    """
    import jax.numpy as jnp

    from examples.papers._ad_optimize import adam_minimize

    if bounds is None:
        w_tip, min_gap = platform.min_tip, platform.min_gap
        bounds = (
            (2 * w_tip, 0.9),  # w_a
            (2 * w_tip, 0.45),  # w_b - stay in the sub-wavelength (weakly-guiding)
            (min_gap, 1.2),  # gap - the coupling weakens sharply beyond ~1.2 um
            (min_gap, 0.30),  # g_b
            (0.6, 1.6),  # frac_mid
            (0.6, 1.6),  # frac_out
            (20.0, 400.0),  # l1
            (20.0, max_total_length),  # l2
            (50.0, 3000.0),  # l3
            (20.0, 400.0),  # l4
        )
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    scale = hi - lo

    cache: dict[tuple[float, ...], tuple[float, float, float, float]] = {}

    def geometric(params6: tuple[float, ...]) -> tuple[float, float, float, float]:
        if params6 not in cache:
            if len(cache) > 64:
                cache.clear()
            cache[params6] = _joint_geometric_quantities(
                platform, cutoff_wl, res, *params6, compute_modes=compute_modes
            )
        return cache[params6]

    def objective(params: np.ndarray) -> np.ndarray:
        w_a, w_b, gap, g_b, frac_mid, frac_out, l1, l2, l3, l4 = (
            float(p) for p in params
        )
        n_a, n_b, kappa, dn_dw = geometric((w_a, w_b, gap, g_b, frac_mid, frac_out))
        phase_loss = (n_a - n_b) ** 2
        extinction_db = _taper_extinction(
            kappa, dn_dw, w_a, platform.min_tip, cutoff_wl, l2
        )
        ext_shortfall = max(target_extinction_db - extinction_db, 0.0)
        total_length = l1 + l2 + l3 + l4
        length_excess = max(total_length - max_total_length, 0.0)
        # the compactness preference excludes l2 (the critical phase-matching
        # taper, whose length is driven by the extinction shortfall above) so
        # it does not fight that gradient; it only trims the non-critical
        # input/bend/output sections (l1, l3, l4).
        loss = (
            2000.0 * phase_loss
            + 0.02 * ext_shortfall**2
            + 3e-4 * (l1 + l3 + l4)
            + 1e-4 * length_excess**2
        )
        return np.asarray(loss, dtype=float)

    differentiable_loss = mw.make_differentiable_objective(objective, shape=())

    def loss_fn(x_norm: object) -> object:
        real = lo + jnp.asarray(x_norm) * scale
        return differentiable_loss(real)

    x0_arr = np.clip(np.asarray(x0, dtype=float), lo, hi)
    x0_norm = (x0_arr - lo) / scale
    x_opt_norm, trace = adam_minimize(
        loss_fn,
        list(x0_norm),
        steps=steps,
        lr=lr,
        bounds=[(0.0, 1.0)] * len(x0),
        param_names=JOINT_PARAM_NAMES,
        objective_name="joint design loss",
    )
    trace.params = [lo + np.asarray(p) * scale for p in trace.params]
    x_opt = lo + x_opt_norm * scale
    return x_opt, trace


def design_dichroic(
    platform: Platform,
    cutoff_wl: float,
    wgb: WGB | None = None,
    gap_out: float = 2.0,
    target_extinction_db: float = 20.0,
    res: float = 0.04,
    compute_modes: Callable | None = None,
    *,
    use_gradient: bool = False,
    gradient_w0: float = 0.4,
    gradient_steps: int = 25,
) -> DichroicDesign:
    """Design + optimize a dichroic beam splitter for a target cutoff.

    1. Choose a default sub-wavelength WGB from the fabrication limits (rails a
       little above ``min_tip``, gaps at ``min_gap``) if none is given.
    2. Set the WGA width that phase-matches WGB at ``cutoff_wl`` - either by
       root-finding (:func:`phase_match_width`, the default) or, when
       ``use_gradient=True``, by **AD gradient-based optimization**
       (:func:`optimize_phase_match_width`, ``jax.grad`` + Adam through
       :func:`meow.make_differentiable_neffs`) from the initial guess
       ``gradient_w0``. Both reach the same phase-matched width; the gradient
       path is the template for objectives that are not simple 1D root-finds.
    3. Pick the **largest coupling gap** (sharpest cutoff) whose Landau-Zener
       phase-matching taper still fits the length budget at ``target_extinction``
       (a larger gap weakens ``kappa`` and lengthens the required taper).
    4. Allocate the four section lengths and build the parametric layout.
    """
    if wgb is None:
        wgb = WGB(
            rail_width=max(2 * platform.min_tip, 0.25),
            gap=platform.min_gap,
            n_rails=3,
        )

    opt_trace = None
    if use_gradient:
        w_a, opt_trace = optimize_phase_match_width(
            platform, cutoff_wl, wgb, w0=gradient_w0, res=res, steps=gradient_steps
        )
    else:
        w_a = phase_match_width(
            platform, cutoff_wl, wgb, res=res, compute_modes=compute_modes
        )
    # dn_WGA/dw at the design width (for the adiabaticity rate)
    dw = 0.02
    n_hi = solid_neff(
        platform, w_a + dw, cutoff_wl, res=res, compute_modes=compute_modes
    )
    n_lo = solid_neff(
        platform, w_a - dw, cutoff_wl, res=res, compute_modes=compute_modes
    )
    dn_dw = (n_hi - n_lo) / (2 * dw)

    # sweep the gap upward; keep the largest gap whose taper fits the budget
    gaps = np.round(np.arange(platform.min_gap, platform.min_gap + 1.2, 0.1), 4)
    best = None
    for gap in gaps:
        kappa = coupling_kappa(
            platform,
            w_a,
            wgb,
            float(gap),
            cutoff_wl,
            res=res,
            compute_modes=compute_modes,
        )
        _, l2, _, _ = _allocate_lengths(platform, float(gap), gap_out)
        er = _taper_extinction(kappa, dn_dw, w_a, platform.min_tip, cutoff_wl, l2)
        if best is None or er >= target_extinction_db:
            best = (float(gap), kappa, er)
        if er < target_extinction_db:
            break
    gap, kappa, extinction_db = best  # type: ignore[misc]

    lengths = _allocate_lengths(platform, gap, gap_out)
    component = dichroic_filter(
        w_a=w_a,
        w_b=wgb.rail_width,
        g_b=wgb.gap,
        gap=gap,
        gap_out=gap_out,
        w_tip=platform.min_tip,
        l1=lengths[0],
        l2=lengths[1],
        l3=lengths[2],
        l4=lengths[3],
        frac_mid=wgb.frac_mid,
        frac_out=wgb.frac_out,
    )
    return DichroicDesign(
        platform=platform,
        cutoff_wl=cutoff_wl,
        wgb=wgb,
        w_a=w_a,
        gap=gap,
        lengths=lengths,
        kappa=kappa,
        dn_dw=dn_dw,
        extinction_db=extinction_db,
        component=component,
        opt_trace=opt_trace,
    )


# --------------------------------------------------------------------------
# platform-aware extrusion of the designed device
# --------------------------------------------------------------------------
def to_params(design: DichroicDesign) -> dict:
    """A picklable parameter dict fully describing ``design`` (no gdsfactory).

    The gdsfactory ``Component`` of a :class:`DichroicDesign` cannot be pickled,
    so this returns only the (picklable) platform, sub-wavelength WGB and scalar
    design outputs. :func:`design_from_params` rebuilds an equivalent design
    (re-creating the layout) from it - e.g. inside a slurm job that received the
    dict over the wire.
    """
    return {
        "platform": design.platform,
        "cutoff_wl": design.cutoff_wl,
        "wgb": design.wgb,
        "w_a": design.w_a,
        "gap": design.gap,
        "lengths": tuple(design.lengths),
        "kappa": design.kappa,
        "dn_dw": design.dn_dw,
        "extinction_db": design.extinction_db,
    }


def design_from_params(
    platform: Platform,
    cutoff_wl: float,
    wgb: WGB,
    w_a: float,
    gap: float,
    lengths: tuple[float, float, float, float],
    *,
    kappa: float = 0.0,
    dn_dw: float = 0.0,
    extinction_db: float = 0.0,
    gap_out: float = 2.0,
) -> DichroicDesign:
    """Rebuild a :class:`DichroicDesign` (incl. its layout) from scalar params.

    Unlike :func:`design_dichroic` this does *no* optimization or FDE solving -
    it just re-creates the parametric layout for an already-chosen design, so it
    is cheap and deterministic (used to reconstruct a design from
    :func:`to_params` on a worker node).
    """
    component = dichroic_filter(
        w_a=w_a,
        w_b=wgb.rail_width,
        g_b=wgb.gap,
        gap=gap,
        gap_out=gap_out,
        w_tip=platform.min_tip,
        l1=lengths[0],
        l2=lengths[1],
        l3=lengths[2],
        l4=lengths[3],
        frac_mid=wgb.frac_mid,
        frac_out=wgb.frac_out,
    )
    return DichroicDesign(
        platform=platform,
        cutoff_wl=cutoff_wl,
        wgb=wgb,
        w_a=w_a,
        gap=gap,
        lengths=tuple(lengths),
        kappa=kappa,
        dn_dw=dn_dw,
        extinction_db=extinction_db,
        component=component,
    )


def device_structures(design: DichroicDesign) -> list[mw.Structure3D]:
    """Extrude a designed device layout onto its platform (rib + slab + clad)."""
    platform = design.platform
    h = platform.core_thickness
    h_slab = platform.slab_thickness
    run = (h - h_slab) * np.tan(np.deg2rad(platform.sidewall_deg))
    rules = {
        LAYER_WG: [
            mw.GdsExtrusionRule(
                material=platform.core,
                h_min=h_slab,
                h_max=h,
                buffer=run,  # drawn width is the rib top; grow the base
                sidewall_angle=platform.sidewall_deg,
            )
        ]
    }
    structs = mw.extrude_gds(design.component, rules)
    z_max = float(design.component.xmax)
    x_lo = float(design.component.ymin) - 1.5
    x_hi = float(design.component.ymax) + 1.5
    extra: list[mw.Structure3D] = []
    if h_slab > 1e-9:
        extra.append(
            mw.Structure(
                material=platform.core,
                geometry=mw.Box(
                    x_min=x_lo,
                    x_max=x_hi,
                    y_min=0.0,
                    y_max=h_slab,
                    z_min=0.0,
                    z_max=z_max,
                ),
                mesh_order=2,
            )
        )
    extra.append(
        mw.Structure(
            material=platform.clad,
            geometry=mw.Box(
                x_min=x_lo,
                x_max=x_hi,
                y_min=-platform.clad_thickness,
                y_max=h + platform.clad_thickness,
                z_min=0.0,
                z_max=z_max,
            ),
            mesh_order=10,
        )
    )
    return structs + extra


# --------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------
def _silicon_platform() -> Platform:
    """A 220 nm SOI platform (matches the Magden 2018 silicon filter)."""
    return Platform(
        core=mw.silicon,
        clad=mw.silicon_oxide,
        core_thickness=0.22,
        sidewall_deg=0.0,
        etch_fraction=1.0,
        min_tip=0.05,
        min_gap=0.10,
        max_length=2000.0,
    )


def tapered_component(
    design: DichroicDesign,
    port_widths: dict[str, float] | None = None,
    taper_lengths: dict[str, float] | float = 20.0,
) -> gf.Component:
    """The designed device with optional linear access tapers on its ports.

    ``port_widths`` maps port names (``in0`` / ``short_pass`` / ``long_pass``)
    to target widths [um]; the default (``None``) adds *no* taper and keeps the
    designed edge widths. Useful to match a routing/measurement width.
    """
    from examples.papers._designer_extras import tapered_ports

    return tapered_ports(
        design.component, port_widths, taper_lengths, layer=LAYER_WG
    )


def ad_optimization_figure(
    platform: Platform,
    wgb: WGB,
    cutoff_wl: float,
    res: float,
    out_path: Path,
    *,
    gradient_w0: float = 0.60,
    gradient_steps: int = 25,
) -> dict[str, object]:
    """AD-optimization demo figure: trace + before/after performance + layout.

    Designs a device by **gradient-based** phase-match-width optimization
    (:func:`design_dichroic` with ``use_gradient=True``) starting from a
    deliberately off-target initial width ``gradient_w0``, at the *fixed*
    platform/WGB layer stack and the target ``cutoff_wl``. Plots:

    - (a) the optimization trace (loss and width vs. iteration);
    - (b) the index-crossing performance before (initial width) and after
      (optimized width) optimization, with the target cutoff marked;
    - (c) the optimized device layout.
    """
    import matplotlib.pyplot as plt

    from examples.papers._ad_optimize import plot_trace
    from examples.papers._plot import plot_component

    design = design_dichroic(
        platform, cutoff_wl, wgb=wgb, res=res,
        use_gradient=True, gradient_w0=gradient_w0, gradient_steps=gradient_steps,
    )
    assert design.opt_trace is not None  # noqa: S101 (use_gradient=True always sets it)
    trace = design.opt_trace
    w_init, w_opt = gradient_w0, design.w_a

    fig = plt.figure(figsize=(13, 8))
    grid = fig.add_gridspec(2, 2)
    ax_loss = fig.add_subplot(grid[0, 0])
    ax_params = fig.add_subplot(grid[0, 1])
    plot_trace(trace, ax_loss, ax_params, loss_ylog=True)

    ax_perf = fig.add_subplot(grid[1, 0])
    wls = np.linspace(cutoff_wl * 0.85, cutoff_wl * 1.15, 25)
    n_b = [segmented_neff(platform, wgb, wl, res=res) for wl in wls]
    n_a_init = [solid_neff(platform, w_init, wl, res=res) for wl in wls]
    n_a_opt = [solid_neff(platform, w_opt, wl, res=res) for wl in wls]
    ax_perf.plot(wls * 1e3, n_b, "k--", lw=2, label="WGB (fixed)")
    ax_perf.plot(
        wls * 1e3, n_a_init, "C3:", label=f"WGA initial ({w_init * 1e3:.0f} nm)"
    )
    ax_perf.plot(
        wls * 1e3, n_a_opt, "C0-", label=f"WGA optimized ({w_opt * 1e3:.0f} nm)"
    )
    ax_perf.axvline(cutoff_wl * 1e3, color="0.5", ls=":", lw=1)
    ax_perf.set_xlabel("wavelength [nm]")
    ax_perf.set_ylabel("effective index")
    ax_perf.set_title(f"phase-match performance at target {cutoff_wl * 1e3:.0f} nm")
    ax_perf.legend(fontsize=8)
    ax_perf.grid(visible=True)

    ax_layout = fig.add_subplot(grid[1, 1])
    plot_component(design.component, ax_layout)
    ax_layout.set_aspect("auto")
    ax_layout.set_title(
        f"AD-optimized device: w_a={w_opt * 1e3:.0f} nm, "
        f"gap={design.gap * 1e3:.0f} nm, L={design.total_length:.0f} um,\n"
        f"ER~{design.extinction_db:.0f} dB",
        fontsize=9,
    )

    fig.suptitle(
        "Dichroic designer: AD gradient-based phase-match optimization "
        f"(jax.grad via make_differentiable_neffs, target "
        f"{cutoff_wl * 1e3:.0f} nm)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "figure": str(out_path),
        "w_init_nm": round(w_init * 1e3, 1),
        "w_opt_nm": round(w_opt * 1e3, 1),
        "final_loss": trace.losses[-1],
        "iterations": len(trace.losses),
    }


def design_dichroic_joint(
    platform: Platform,
    cutoff_wl: float,
    *,
    x0: tuple[float, ...] = JOINT_X0_DEFAULT,
    gap_out: float = 2.0,
    target_extinction_db: float = 20.0,
    max_total_length: float = 5000.0,
    res: float = 0.05,
    steps: int = 26,
    lr: float = 0.025,
    compute_modes: Callable | None = None,
) -> DichroicDesign:
    """Design a dichroic splitter by joint AD optimization of every parameter.

    Runs :func:`optimize_dichroic_joint` - the full 10-parameter (widths, gaps,
    fractional rail widths, section lengths) joint optimization - and builds the
    resulting :class:`DichroicDesign` (its ``opt_trace`` holds the 10-parameter
    trace, unlike :func:`design_dichroic`'s single-parameter ``opt_trace``).
    """
    params, trace = optimize_dichroic_joint(
        platform,
        cutoff_wl,
        x0=x0,
        target_extinction_db=target_extinction_db,
        max_total_length=max_total_length,
        res=res,
        steps=steps,
        lr=lr,
        compute_modes=compute_modes,
    )
    w_a, w_b, gap, g_b, frac_mid, frac_out, l1, l2, l3, l4 = (float(p) for p in params)
    wgb = WGB(rail_width=w_b, gap=g_b, n_rails=3, frac_mid=frac_mid, frac_out=frac_out)
    _, _, kappa, dn_dw = _joint_geometric_quantities(
        platform,
        cutoff_wl,
        res,
        w_a,
        w_b,
        gap,
        g_b,
        frac_mid,
        frac_out,
        compute_modes=compute_modes,
    )
    extinction_db = _taper_extinction(
        kappa, dn_dw, w_a, platform.min_tip, cutoff_wl, l2
    )
    component = dichroic_filter(
        w_a=w_a,
        w_b=w_b,
        g_b=g_b,
        gap=gap,
        gap_out=gap_out,
        w_tip=platform.min_tip,
        l1=l1,
        l2=l2,
        l3=l3,
        l4=l4,
        frac_mid=frac_mid,
        frac_out=frac_out,
    )
    return DichroicDesign(
        platform=platform,
        cutoff_wl=cutoff_wl,
        wgb=wgb,
        w_a=w_a,
        gap=gap,
        lengths=(l1, l2, l3, l4),
        kappa=kappa,
        dn_dw=dn_dw,
        extinction_db=extinction_db,
        component=component,
        opt_trace=trace,
    )


def joint_ad_optimization_figure(
    platform: Platform,
    cutoff_wl: float,
    out_path: Path,
    *,
    x0: tuple[float, ...] = JOINT_X0_DEFAULT,
    res: float = 0.05,
    steps: int = 26,
    lr: float = 0.025,
) -> dict[str, object]:
    """Joint AD-optimization demo: trace over all 10 parameters + performance + layout.

    Designs a device by **joint gradient-based** optimization
    (:func:`design_dichroic_joint`) over the full practical parameter set -
    both waveguide full widths, the WGA-WGB coupling gap, the WGB inter-rail
    gap, the fractional middle/outer WGB rail widths, and all four section
    lengths (up to 5 mm total) - starting from a deliberately off-target
    initial guess, at the target ``cutoff_wl``. Plots:

    - (a) the optimization trace (composite loss vs. iteration);
    - (b) every design parameter's trajectory vs. iteration;
    - (c) the index-crossing performance before (initial parameters) and
      after (optimized parameters) optimization, with the target cutoff
      marked;
    - (d) the optimized device layout.
    """
    import matplotlib.pyplot as plt

    from examples.papers._ad_optimize import plot_trace
    from examples.papers._plot import plot_component

    design = design_dichroic_joint(
        platform, cutoff_wl, x0=x0, res=res, steps=steps, lr=lr
    )
    assert design.opt_trace is not None  # noqa: S101 (always set here)
    trace = design.opt_trace
    p_init = np.asarray(trace.params[0])
    p_opt = np.asarray(trace.params[-1])

    fig = plt.figure(figsize=(15, 10))
    grid = fig.add_gridspec(2, 2)
    ax_loss = fig.add_subplot(grid[0, 0])
    ax_params = fig.add_subplot(grid[0, 1])
    plot_trace(trace, ax_loss, ax_params, loss_ylog=True)
    # the 10 parameters span wildly different scales (sub-um widths/gaps,
    # O(1) fractions, up-to-5-mm lengths); plot_trace's shared linear axis
    # would flatten most of them to the baseline, so re-plot this panel with
    # each parameter's trajectory min-max normalized to its own [0, 1] range
    # (physical values are reported in the layout panel/return dict instead).
    ax_params.clear()
    all_params = np.asarray(trace.params)
    p_lo, p_hi = all_params.min(axis=0), all_params.max(axis=0)
    p_span = np.where(p_hi > p_lo, p_hi - p_lo, 1.0)
    it = np.arange(len(trace.losses))
    for j, name in enumerate(trace.param_names):
        ax_params.plot(
            it, (all_params[:, j] - p_lo[j]) / p_span[j], "o-", ms=3, label=name
        )
    ax_params.set_xlabel("iteration")
    ax_params.set_ylabel("parameter value (min-max normalized)")
    ax_params.legend(fontsize=7, ncol=2)
    ax_params.grid(visible=True)
    ax_params.set_title("design parameters (each normalized to its own range)")

    ax_perf = fig.add_subplot(grid[1, 0])
    wls = np.linspace(cutoff_wl * 0.85, cutoff_wl * 1.15, 25)
    wgb_init = WGB(
        rail_width=float(p_init[1]), gap=float(p_init[3]),
        n_rails=3, frac_mid=float(p_init[4]), frac_out=float(p_init[5]),
    )
    n_b_init = [segmented_neff(platform, wgb_init, wl, res=res) for wl in wls]
    n_a_init = [solid_neff(platform, float(p_init[0]), wl, res=res) for wl in wls]
    n_b_opt = [segmented_neff(platform, design.wgb, wl, res=res) for wl in wls]
    n_a_opt = [solid_neff(platform, design.w_a, wl, res=res) for wl in wls]
    ax_perf.plot(wls * 1e3, n_b_init, "C3:", label="WGB initial")
    ax_perf.plot(wls * 1e3, n_a_init, "C1:", label="WGA initial")
    ax_perf.plot(wls * 1e3, n_b_opt, "k--", lw=2, label="WGB optimized")
    ax_perf.plot(wls * 1e3, n_a_opt, "C0-", lw=2, label="WGA optimized")
    ax_perf.axvline(cutoff_wl * 1e3, color="0.5", ls=":", lw=1)
    ax_perf.set_xlabel("wavelength [nm]")
    ax_perf.set_ylabel("effective index")
    ax_perf.set_title(f"phase-match performance at target {cutoff_wl * 1e3:.0f} nm")
    ax_perf.legend(fontsize=7)
    ax_perf.grid(visible=True)

    ax_layout = fig.add_subplot(grid[1, 1])
    plot_component(design.component, ax_layout)
    ax_layout.set_aspect("auto")
    ax_layout.set_title(
        f"joint AD-optimized device: w_a={design.w_a * 1e3:.0f} nm, "
        f"w_b={design.wgb.rail_width * 1e3:.0f} nm, gap={design.gap * 1e3:.0f} nm,\n"
        f"g_b={design.wgb.gap * 1e3:.0f} nm, frac_mid={design.wgb.frac_mid:.2f}, "
        f"frac_out={design.wgb.frac_out:.2f}, L={design.total_length:.0f} um, "
        f"ER~{design.extinction_db:.2f} dB",
        fontsize=8,
    )

    fig.suptitle(
        "Dichroic designer: joint AD gradient-based optimization over all "
        f"practical design parameters (jax.grad via make_differentiable_objective, "
        f"target {cutoff_wl * 1e3:.0f} nm)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "figure": str(out_path),
        "params_init": dict(
            zip(JOINT_PARAM_NAMES, np.round(p_init, 4).tolist(), strict=True)
        ),
        "params_opt": dict(
            zip(JOINT_PARAM_NAMES, np.round(p_opt, 4).tolist(), strict=True)
        ),
        "final_loss": trace.losses[-1],
        "extinction_db": round(design.extinction_db, 3),
        "total_length_um": round(design.total_length, 1),
        "iterations": len(trace.losses),
    }


# shared broad-band grid axis: covers every demo cutoff (1.30 .. 2.00 um)
GRID_BAND = (1.30, 2.00)


def analyze_dichroic_design(
    design: DichroicDesign,
    out_dir: Path | str,
    *,
    band: tuple[float, float] = GRID_BAND,
    num_cells: int | None = None,
    num_modes: int | None = None,
) -> dict:
    """EME broad-band short-/long-pass spectrum + GDS + field plots for a design.

    Mirrors ``dichroic_designer_slurm`` but in-process (the spectrum is
    distributed over local worker threads). Writes ``*_spectrum.{png,csv,json}``
    (``t_short`` / ``t_long`` over the shared :data:`GRID_BAND`), the GDS and a
    summary into ``out_dir`` and returns the summary dict.
    """
    from concurrent.futures import ThreadPoolExecutor

    from examples.papers import _analysis, _backends

    n = pick(low=9, medium=31, high=61)
    settings = {
        "label": f"{design.cutoff_wl * 1e3:.0f}nm",
        "num_cells": num_cells or _resolution.num_cells(low=16, medium=48),
        "num_modes": num_modes or _resolution.num_modes(low=4, medium=6),
        "device_res": pick(low=0.07, medium=0.05, high=0.035),
        "backend": _backends.backend_name(),
        "spectrum_wls": np.linspace(band[0], band[1], n),
        "prop_wls": np.array([design.cutoff_wl]),
        "n_neff": pick(low=4, medium=9, high=15),
        "num_z": pick(low=200, medium=600, high=1000),
    }
    workers = _backends.max_workers() or 4

    def executor_factory(_name: str) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=workers)

    run = _analysis.submit_dichroic_run(
        to_params(design), settings, executor_factory=executor_factory,
        out_dir=out_dir, save_fields=False,
    )
    return run.gather()


def dichroic_test_structures(
    design: DichroicDesign,
    out_dir: Path | str,
    *,
    counts: tuple[int, ...] = (0, 1, 2, 4),
    chip_width: float = 5000.0,
) -> dict[str, object]:
    """Cut-back coupler array (constant length, 5 mm chip) as GDS + a preview.

    Rows with a varied number of cascaded dichroic couplings but a constant
    total waveguide length between regularly-spaced ports on either side of the
    chip -- the cut-back layout for the per-coupler excess loss.
    """
    import matplotlib.pyplot as plt

    from examples.papers._designer_extras import coupler_cutback_array
    from examples.papers._plot import plot_component

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{design.cutoff_wl * 1e3:.0f}nm"
    w_port = float(design.component.ports["in0"].width)
    array = coupler_cutback_array(
        design.component, counts=counts, in_port="in0", thru_port="long_pass",
        chip_width=chip_width, pitch=8.0 * w_port + 4.0, width=w_port,
        layer=LAYER_WG,
    )
    gpath = out / f"dichroic_{stem}_cutback_array.gds"
    array.write_gds(gpath)

    fig, ax = plt.subplots(figsize=(11, 4))
    plot_component(array, ax)
    ax.set_title(
        f"{stem} cutoff: coupler cut-back array "
        f"({', '.join(str(n) for n in counts)} couplings, {chip_width / 1e3:.0f} mm)"
    )
    ax.set_aspect("auto")
    fpath = out / f"dichroic_{stem}_test_structures.png"
    fig.tight_layout()
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    return {"gds": str(gpath), "figure": str(fpath)}


def dichroic_spectrum_grid(
    designs: list[DichroicDesign],
    analyses_root: Path,
    out_path: Path,
    *,
    band: tuple[float, float] = GRID_BAND,
) -> Path | None:
    """Column grid of every design's broad-band short/long-pass spectrum."""
    import json

    from examples.papers import _analysis
    from examples.papers._designer_extras import spectrum_grid

    rows = []
    for d in designs:
        label = f"{d.cutoff_wl * 1e3:.0f}nm"
        stem = _analysis._file_stem(label)
        jpath = analyses_root / label / f"{stem}_spectrum.json"
        if not jpath.exists():
            continue
        spec = json.loads(jpath.read_text())
        rows.append({
            "label": f"{d.cutoff_wl * 1e3:.0f} nm cutoff",
            "wls": np.asarray(spec["wavelength_um"]),
            "short_pass": np.asarray(spec["t_short"]),
            "long_pass": np.asarray(spec["t_long"]),
            "design_wls": [d.cutoff_wl],
        })
    if not rows:
        return None
    return spectrum_grid(
        rows, out_path, db=True, xlim_nm=(band[0] * 1e3, band[1] * 1e3),
        ports=(("short_pass", "C0"), ("long_pass", "C3")),
        title="Dichroic designer: broad-band short-/long-pass spectra",
    )


def main() -> dict[str, object]:
    """Design dichroic splitters for several cutoffs on an SOI platform."""
    import matplotlib.pyplot as plt

    from examples.papers._plot import plot_component

    FIGDIR.mkdir(exist_ok=True, parents=True)
    gf.gpdk.PDK.activate()
    platform = _silicon_platform()
    wgb = WGB(rail_width=0.25, gap=0.10, n_rails=3)
    res = pick(low=0.05, medium=0.03, high=0.02)
    targets = pick(
        low=[1.40, 1.55],
        medium=[1.40, 1.50, 1.60, 1.70],
        high=[1.40, 1.50, 1.60, 1.70],
    )

    designs = [design_dichroic(platform, wl_c, wgb=wgb, res=res) for wl_c in targets]
    summary = {
        f"{d.cutoff_wl * 1e3:.0f}nm": {
            "w_a_nm": round(d.w_a * 1e3, 1),
            "gap_nm": round(d.gap * 1e3, 0),
            "length_um": round(d.total_length, 0),
            "kappa_per_mm": round(d.kappa * 1e3, 2),
            "extinction_db": round(d.extinction_db, 1),
        }
        for d in designs
    }

    # n_eff crossings + cutoff-vs-width + a designed layout
    fig = plt.figure(figsize=(13, 7))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    wls = np.linspace(1.30, 1.80, pick(low=5, medium=11, high=21))
    ax = fig.add_subplot(grid[0, 0])
    n_b = [segmented_neff(platform, wgb, wl, res=res) for wl in wls]
    ax.plot(wls * 1e3, n_b, "k--", lw=2, label="WGB (sub-wavelength)")
    for d in designs:
        n_a = [solid_neff(platform, d.w_a, wl, res=res) for wl in wls]
        ax.plot(wls * 1e3, n_a, label=f"WGA {d.w_a * 1e3:.0f} nm")
        ax.plot(d.cutoff_wl * 1e3, np.interp(d.cutoff_wl, wls, n_b), "o", color="k")
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("effective index")
    ax.set_title("designed WGA / WGB index crossings (= cutoffs)")
    ax.legend(fontsize=7)
    ax.grid(visible=True)

    ax = fig.add_subplot(grid[0, 1])
    ax.plot(
        [d.w_a * 1e3 for d in designs], [d.cutoff_wl * 1e3 for d in designs], "C0o-"
    )
    ax.set_xlabel("WGA width [nm]")
    ax.set_ylabel("targeted cutoff [nm]")
    ax.set_title("cutoff vs WGA width (design curve)")
    ax.grid(visible=True)

    ax = fig.add_subplot(grid[1, :])
    plot_component(designs[-1].component, ax)
    d = designs[-1]
    ax.set_title(
        f"designed device for {d.cutoff_wl * 1e3:.0f} nm cutoff: "
        f"w_a={d.w_a * 1e3:.0f} nm, gap={d.gap * 1e3:.0f} nm, "
        f"L={d.total_length:.0f} um, ER~{d.extinction_db:.0f} dB"
    )
    ax.set_aspect("auto")

    fig.suptitle("Generalized dichroic beam-splitter designer (SOI demo)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "dichroic_designer.png", dpi=150)
    plt.close(fig)

    # per-design EME broad-band spectra (-> column grid) + cut-back test arrays
    analyses_root = FIGDIR / "dichroic_designer"
    analyses, tests = {}, {}
    for d in designs:
        label = f"{d.cutoff_wl * 1e3:.0f}nm"
        analyses[label] = analyze_dichroic_design(d, analyses_root / label)
        tests[label] = dichroic_test_structures(d, analyses_root / label)
    grid = dichroic_spectrum_grid(
        designs, analyses_root, FIGDIR / "dichroic_designer_spectrum_grid.png"
    )

    # single-parameter AD demo (jax.grad through make_differentiable_neffs)
    ad_demo = ad_optimization_figure(
        platform, wgb, targets[-1], res,
        FIGDIR / "dichroic_designer_ad_optimization.png",
        gradient_w0=pick(low=0.7, medium=0.7, high=0.7),
        gradient_steps=pick(low=15, medium=25, high=25),
    )

    # joint AD optimization demo over every practical design parameter
    # (jax.grad through make_differentiable_objective; a coarser resolution
    # than the discrete-sweep designs above bounds its ~2 * n_params re-solve
    # cost per gradient step)
    joint_ad_demo = joint_ad_optimization_figure(
        platform, targets[-1],
        FIGDIR / "dichroic_designer_joint_ad_optimization.png",
        res=pick(low=0.08, medium=0.05, high=0.04),
        steps=pick(low=12, medium=26, high=30),
    )

    return {
        "designs": summary,
        "analyses": analyses,
        "test_structures": tests,
        "spectrum_grid": str(grid) if grid else None,
        "ad_optimization": ad_demo,
        "joint_ad_optimization": joint_ad_demo,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
