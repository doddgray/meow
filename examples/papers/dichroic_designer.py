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

import gdsfactory as gf
import numpy as np
from scipy.optimize import brentq

import meow as mw
from examples.papers import _resolution
from examples.papers.magden2018_dichroic import LAYER_WG, dichroic_filter

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
    """A sub-wavelength multi-rail (segmented) WGB cross-section."""

    rail_width: float
    gap: float
    n_rails: int = 3

    @property
    def total_width(self) -> float:
        """Edge-to-edge width of the multi-rail WGB."""
        return self.n_rails * self.rail_width + (self.n_rails - 1) * self.gap

    def centers(self, x0: float = 0.0) -> list[float]:
        """Lateral rail centres, symmetric about ``x0``."""
        pitch = self.rail_width + self.gap
        return [x0 + (i - (self.n_rails - 1) / 2) * pitch for i in range(self.n_rails)]


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
    structs = _ridge_structures(
        platform, [wgb.rail_width] * wgb.n_rails, wgb.centers(x0), (-span, span)
    )
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
        _ridge_structures(
            platform, [wgb.rail_width] * wgb.n_rails, wgb.centers(x0_b), (-span, span)
        ),
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


def design_dichroic(
    platform: Platform,
    cutoff_wl: float,
    wgb: WGB | None = None,
    gap_out: float = 2.0,
    target_extinction_db: float = 20.0,
    res: float = 0.04,
    compute_modes: Callable | None = None,
) -> DichroicDesign:
    """Design + optimize a dichroic beam splitter for a target cutoff.

    1. Choose a default sub-wavelength WGB from the fabrication limits (rails a
       little above ``min_tip``, gaps at ``min_gap``) if none is given.
    2. Root-find the WGA width that phase-matches WGB at ``cutoff_wl``.
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


# shared broad-band grid axis: covers every demo cutoff (1.30 .. 2.00 um)
GRID_BAND = (1.30, 2.00)


def analyze_dichroic_design(
    design: DichroicDesign,
    out_dir: Path | str,
    *,
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
        "spectrum_wls": np.linspace(GRID_BAND[0], GRID_BAND[1], n),
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
    designs: list[DichroicDesign], analyses_root: Path, out_path: Path
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
        rows, out_path, db=True, xlim_nm=(GRID_BAND[0] * 1e3, GRID_BAND[1] * 1e3),
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
    return {
        "designs": summary,
        "analyses": analyses,
        "test_structures": tests,
        "spectrum_grid": str(grid) if grid else None,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
