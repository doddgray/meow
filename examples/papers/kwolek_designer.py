"""Generalized FAQUAD wavelength-filter designer for X-cut TFLN / TFLT.

This example generalizes the fast-quasi-adiabatic (FAQUAD) wavelength
combiner/filter of Kwolek et al., arXiv:2603.27034 (2026) (reproduced in
``kwolek2026_faquad.py``) into a **platform-parametric designer**: given a
thin-film X-cut platform - the (anisotropic) core material, the film
thickness, the rib etch depth, the sidewall angle and the fabrication limits -
and a target fundamental/second-harmonic (FH/SH) wavelength pair, it designs
and optimizes a FAQUAD coupler that

- transfers the FH adiabatically to the cross port (the combiner action), and
- keeps the strongly-confined SH in the bar port (the dichroic filter action).

Design principle (same as the paper):

- Two identical rib waveguides are brought to a fabrication-limited minimum gap
  ``g_m`` over a constant-gap interaction length ``l_m`` (Region I) and then
  separated by curvature-continuous **Euler S-bends** (Regions II/III), opening
  the gap smoothly from ``g_m`` to ``g_f``.
- The coupling ``kappa(g) = kappa_0 exp(-g/g_0)`` and the phase-mismatch slope
  ``s = d(delta beta)/d(top-width)`` are extracted from FDE solves at the FH.
- The FAQUAD mixing angle ``chi(z)`` is driven from 0 to ``pi`` at a constant
  adiabaticity ``eta``; the realized top-width difference ``dTW(z)`` follows
  ``kappa cot(chi) / s`` and is tapered to zero at the device ends.

Optimization (within a maximum device length budget):

1. The nominal waveguide top width is chosen as the **largest width whose FAQUAD
   device still meets the FH extinction target within the length budget**. The
   FH/SH coupling contrast ``kappa_FH / kappa_SH`` grows with width (a wider rib
   confines the SH far more than the FH), so this maximizes SH rejection subject
   to keeping a fabricable, budget-fitting FH combiner.
2. The constant-gap length ``l_m`` is then chosen as the shortest length whose
   FAQUAD adiabaticity ``eta`` meets the target FH extinction (``eta`` shrinks
   with ``l_m``); if the target does not fit the budget, the longest ``l_m``
   within ``max_length`` is used.

Because the SH is far more confined than the FH, its coupling ``kappa_SH`` is
orders of magnitude weaker, so the same geometry filters the SH into the bar
port automatically; the designer reports ``kappa_SH`` as a diagnostic.

The predicted extinction is an order-of-magnitude estimate from the FAQUAD
adiabaticity; validate a final design with a full EME (see
``kwolek_designer_slurm.py``).

Run with ``python -m examples.papers.kwolek_designer``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import gdsfactory as gf
import numpy as np

import meow as mw
from examples.papers import _backends, _resolution
from examples.papers.kwolek2026_faquad import (
    LAYER_RIB,
    FaquadDesign,
    adaptive_cell_lengths,
    ln_material,
    sio2_material,
)

FIGDIR = Path(__file__).parent / "figures"
pick = _resolution.pick


# --------------------------------------------------------------------------
# materials
# --------------------------------------------------------------------------
# X-cut lithium tantalate (congruent), ordinary/extraordinary indices from the
# tabulated dispersion of Bond, J. Appl. Phys. 36, 1674 (1965) (via
# refractiveindex.info), valid ~0.45-1.6 um - covering all FH/SH pairs here.
_LT_WL = np.array([0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20, 1.40, 1.60])
_LT_NO = np.array(
    [2.2420, 2.2160, 2.1834, 2.1652, 2.1538, 2.1454, 2.1391, 2.1305, 2.1236, 2.1174]
)
_LT_NE = np.array(
    [2.2468, 2.2205, 2.1878, 2.1696, 2.1578, 2.1493, 2.1432, 2.1341, 2.1273, 2.1213]
)


def lt_material(wl: float) -> mw.AnisotropicMaterial:
    """Uniaxial lithium tantalate for an x-cut film (crystal z in-plane).

    Mode-plane axes follow the LN convention of ``kwolek2026_faquad.ln_material``:
    x (horizontal, in-plane) is the extraordinary axis, so the permittivity
    tensor diagonal is ``(ne^2, no^2, no^2)``.
    """
    no = float(np.interp(wl, _LT_WL, _LT_NO))
    ne = float(np.interp(wl, _LT_WL, _LT_NE))
    return mw.AnisotropicMaterial(
        name="LiTaO3_xcut",
        eps=[ne**2, no**2, no**2],
        meta={"color": (0.2, 0.45, 0.7, 0.9)},
    )


# --------------------------------------------------------------------------
# platform specification
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TFPlatform:
    """A thin-film X-cut ferroelectric rib-waveguide platform.

    Args:
        name: human-readable platform name (e.g. ``"TFLN-300nm"``).
        core: ``wl -> AnisotropicMaterial`` factory for the core (LN or LT).
        clad: ``wl -> Material`` factory for the under-cladding (SiO2).
        core_thickness: film thickness ``H_film`` [um].
        etch_depth: rib etch depth [um] (slab = ``core_thickness - etch_depth``).
        sidewall_deg: rib sidewall angle from vertical [degrees].
        g_m: minimum fabricable (Region I) gap [um].
        g_f: final output gap [um].
        box_thickness: modeled SiO2 under-cladding thickness [um].
        max_length: maximum allowed device length [um].
    """

    name: str
    core: Callable[[float], mw.Material]
    clad: Callable[[float], mw.Material] = sio2_material
    core_thickness: float = 0.30
    etch_depth: float = 0.10
    sidewall_deg: float = 25.0
    g_m: float = 0.80
    g_f: float = 3.0
    box_thickness: float = 1.2
    max_length: float = 2000.0

    @property
    def slab_thickness(self) -> float:
        """Remaining (unetched) slab thickness [um]."""
        return max(self.core_thickness - self.etch_depth, 0.0)

    @property
    def sidewall_run(self) -> float:
        """Lateral run of the angled sidewall over the rib height [um]."""
        return self.etch_depth * np.tan(np.deg2rad(self.sidewall_deg))


def tfln_platform(core_thickness: float, **kwargs: float) -> TFPlatform:
    """An X-cut thin-film lithium niobate platform of a given thickness."""
    return TFPlatform(
        name=f"TFLN-{core_thickness * 1e3:.0f}nm",
        core=ln_material,
        core_thickness=core_thickness,
        **kwargs,
    )


def tflt_platform(core_thickness: float, **kwargs: float) -> TFPlatform:
    """An X-cut thin-film lithium tantalate platform of a given thickness."""
    return TFPlatform(
        name=f"TFLT-{core_thickness * 1e3:.0f}nm",
        core=lt_material,
        core_thickness=core_thickness,
        **kwargs,
    )


# --------------------------------------------------------------------------
# cross-sections + FDE effective indices
# --------------------------------------------------------------------------
def _background(
    platform: TFPlatform, wl: float, z_max: float, x_span: tuple[float, float]
) -> list[mw.Structure3D]:
    """SiO2 under-cladding + LN/LT slab (air top-cladding after etching)."""
    core = platform.core(wl)
    clad = platform.clad(wl)
    structs = [
        mw.Structure(
            material=clad,
            geometry=mw.Box(
                x_min=x_span[0],
                x_max=x_span[1],
                y_min=-platform.box_thickness,
                y_max=0.0,
                z_min=0.0,
                z_max=z_max,
            ),
            mesh_order=10,
        )
    ]
    if platform.slab_thickness > 1e-9:
        structs.append(
            mw.Structure(
                material=core,
                geometry=mw.Box(
                    x_min=x_span[0],
                    x_max=x_span[1],
                    y_min=0.0,
                    y_max=platform.slab_thickness,
                    z_min=0.0,
                    z_max=z_max,
                ),
                mesh_order=8,
            )
        )
    return structs


def rib_structures(
    platform: TFPlatform,
    wl: float,
    widths: list[float],
    centers: list[float],
    x_span: tuple[float, float] = (-4.5, 4.5),
) -> list[mw.Structure3D]:
    """Angled-sidewall rib waveguides on the platform (for FDE calibration).

    The given ``widths`` are the rib *top* widths; the drawn prism base is
    grown by the sidewall run so the extruded trapezoid has the nominal width
    at the rib top.
    """
    core = platform.core(wl)
    h_slab = platform.slab_thickness
    h_film = platform.core_thickness
    run = platform.sidewall_run
    ribs = [
        mw.Structure(
            material=core,
            geometry=mw.Prism(
                poly=np.array(
                    [
                        (0.0, x0 - (w + 2 * run) / 2),
                        (1.0, x0 - (w + 2 * run) / 2),
                        (1.0, x0 + (w + 2 * run) / 2),
                        (0.0, x0 + (w + 2 * run) / 2),
                    ]
                ),
                h_min=h_slab,
                h_max=h_film,
                axis="y",
                sidewall_angle=platform.sidewall_deg,
            ),
            mesh_order=5,
        )
        for w, x0 in zip(widths, centers, strict=True)
    ]
    return ribs + _background(platform, wl, 1.0, x_span)


def _mesh(platform: TFPlatform, x_span: tuple[float, float], res: float) -> mw.Mesh2D:
    h = platform.core_thickness
    return mw.Mesh2D(
        x=np.arange(x_span[0], x_span[1] + res / 2, res),
        y=np.arange(-platform.box_thickness, h + 0.6 + res / 2, res),
    )


def _te_neffs(
    structures: list[mw.Structure3D],
    wl: float,
    mesh: mw.Mesh2D,
    num_modes: int = 8,
    num_te: int = 2,
    compute_modes: Callable | None = None,
) -> list[float]:
    """Real effective indices of the first ``num_te`` TE modes."""
    compute_modes = compute_modes or mw.compute_modes
    cell = mw.Cell(structures=structures, mesh=mesh, z_min=0.0, z_max=1.0)
    cs = mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=wl, T=25.0))
    modes = compute_modes(cs, num_modes=num_modes)
    te = [m for m in modes if m.te_fraction > 0.5]
    # fall back to the lowest-order modes if too few are TE-majority (can happen
    # for thick films at the short SH wavelength, where modes are more hybrid)
    chosen = te if len(te) >= num_te else list(modes)
    return [float(np.real(m.neff)) for m in chosen[:num_te]]


def slab_neff(
    platform: TFPlatform, wl: float, res: float, compute_modes: Callable | None = None
) -> float:
    """Effective index of the planar slab/cladding background (mode floor).

    Rib-guided modes have ``neff`` above this; it is the cutoff reference for
    counting guided modes. If the film is fully etched there is no slab and the
    floor is the SiO2 cladding index.
    """
    if platform.slab_thickness <= 1e-9:
        return float(np.real(platform.clad(wl).n))
    mesh = _mesh(platform, (-4.5, 4.5), res)
    neffs = _te_neffs(
        _background(platform, wl, 1.0, (-4.5, 4.5)),
        wl,
        mesh,
        num_modes=2,
        num_te=1,
        compute_modes=compute_modes,
    )
    return neffs[0] if neffs else float(np.real(platform.clad(wl).n))


def count_guided_te(
    platform: TFPlatform,
    width: float,
    wl: float,
    res: float,
    floor: float,
    compute_modes: Callable | None = None,
) -> int:
    """Number of guided (neff > slab floor) TE modes of an isolated rib."""
    span = max(3.0, 2.0 * width)
    neffs = _te_neffs(
        rib_structures(platform, wl, [width], [0.0], (-span, span)),
        wl,
        _mesh(platform, (-span, span), res),
        num_modes=6,
        num_te=6,
        compute_modes=compute_modes,
    )
    return int(np.sum(np.asarray(neffs) > floor + 2e-3))


# --------------------------------------------------------------------------
# optimization step 1: the nominal top width
# --------------------------------------------------------------------------
def _sep_length(theta_max_deg: float, g_f: float) -> float:
    """Length of one Euler S-bend separation [um] (geometry only)."""
    return FaquadDesign(
        1.0, 0.3, 1.0, l_m=2.0, theta_max_deg=theta_max_deg, g_f=g_f
    ).l_sep


def optimize_width(
    platform: TFPlatform,
    fh_wl: float,
    *,
    theta_max_deg: float = 3.0,
    target_extinction_db: float = 20.0,
    res: float = 0.04,
    w_lo: float = 0.6,
    w_hi: float = 2.0,
    compute_modes: Callable | None = None,
) -> float:
    """Optimized nominal top width [um] for the FH/SH filter.

    The FH/SH coupling contrast ``kappa_FH / kappa_SH`` rises monotonically with
    width (a wider rib confines the SH far more strongly than the FH), so the
    best SH rejection wants the widest waveguide. But a wider rib also weakens
    ``kappa_FH``, lengthening the adiabatic transfer. We therefore pick the
    **largest width whose FAQUAD device still meets the FH extinction target
    within the platform length budget** - the best SH rejection compatible with
    a fabricable FH combiner. Feasibility is monotonic in width, so a bisection
    converges.
    """
    eta_target = 10 ** (-target_extinction_db / 20.0)
    sep = _sep_length(theta_max_deg, platform.g_f)
    l_m_max = max(50.0, platform.max_length - 2 * sep)

    def feasible(w: float) -> bool:
        kappa_0, g_0, dbeta_dtw = calibrate_coupling(
            platform, w, fh_wl, res=res, compute_modes=compute_modes
        )
        eta_min = FaquadDesign(
            kappa_0, g_0, dbeta_dtw, l_m=l_m_max, theta_max_deg=theta_max_deg
        ).eta
        return eta_min <= eta_target

    if feasible(w_hi):
        return float(w_hi)
    if not feasible(w_lo):
        return float(w_lo)
    lo, hi = w_lo, w_hi
    for _ in range(6):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return float(lo)


# --------------------------------------------------------------------------
# optimization step 2: FDE coupling calibration
# --------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _calibrate_cached(
    platform: TFPlatform,
    w_top: float,
    wl: float,
    gaps: tuple[float, ...],
    res: float,
    compute_modes: Callable | None,
) -> tuple[float, float, float]:
    k0 = 2 * np.pi / wl
    span = max(3.6, 1.5 * (w_top + max(gaps)))
    mesh = _mesh(platform, (-span, span), res)
    kappas = []
    for g in gaps:
        x0 = (w_top + g) / 2
        n_p, n_m = _te_neffs(
            rib_structures(platform, wl, [w_top, w_top], [-x0, x0], (-span, span)),
            wl,
            mesh,
            num_modes=4,
            num_te=2,
            compute_modes=compute_modes,
        )
        kappas.append(0.5 * k0 * abs(n_p - n_m))
    slope, intercept = np.polyfit(np.asarray(gaps), np.log(np.asarray(kappas)), 1)
    g_0 = -1.0 / slope
    kappa_0 = float(np.exp(intercept))

    dws = np.array([-0.05, 0.0, 0.05])
    neffs = [
        _te_neffs(
            rib_structures(platform, wl, [w_top + dw], [0.0], (-span, span)),
            wl,
            mesh,
            num_modes=4,
            num_te=1,
            compute_modes=compute_modes,
        )[0]
        for dw in dws
    ]
    dbeta_dtw = float(k0 * np.polyfit(dws, neffs, 1)[0])
    return kappa_0, float(g_0), dbeta_dtw


def calibrate_coupling(
    platform: TFPlatform,
    w_top: float,
    wl: float,
    res: float = 0.04,
    compute_modes: Callable | None = None,
) -> tuple[float, float, float]:
    """Extract ``(kappa_0, g_0, d(beta)/d(TW))`` from FDE solves at ``wl``.

    ``kappa(g) = kappa_0 exp(-g/g_0)`` is fit to the symmetric/antisymmetric
    supermode splitting at three gaps; the phase-mismatch slope is the
    isolated-waveguide ``d(neff)/d(width)`` times ``k0``.
    """
    gaps = (platform.g_m, 0.5 * (platform.g_m + 1.2), 1.2)
    return _calibrate_cached(
        platform, round(w_top, 4), wl, gaps, res, compute_modes
    )


# --------------------------------------------------------------------------
# the design + optimization
# --------------------------------------------------------------------------
@dataclass
class FaquadFilterDesign:
    """A designed FAQUAD wavelength filter for one FH/SH pair."""

    platform: TFPlatform
    fh_wl: float
    sh_wl: float
    w_top: float
    kappa_0: float
    g_0: float
    dbeta_dtw: float
    kappa_sh: float
    design: FaquadDesign = field(repr=False)
    component: gf.Component = field(repr=False)
    extinction_db: float = 0.0

    @property
    def total_length(self) -> float:
        """Total device length [um]."""
        return 2.0 * self.design.half_length

    @property
    def eta(self) -> float:
        """FAQUAD adiabaticity parameter of the design."""
        return self.design.eta


def _eta_for_lm(
    kappa_0: float, g_0: float, dbeta_dtw: float, l_m: float, theta_max_deg: float
) -> float:
    """FAQUAD adiabaticity parameter for a trial constant-gap length."""
    return FaquadDesign(
        kappa_0, g_0, dbeta_dtw, l_m=l_m, theta_max_deg=theta_max_deg
    ).eta


def optimize_lm(
    kappa_0: float,
    g_0: float,
    dbeta_dtw: float,
    theta_max_deg: float,
    sep_length: float,
    target_extinction_db: float,
    max_length: float,
) -> float:
    """Shortest constant-gap length ``l_m`` meeting the FH extinction target.

    The adiabaticity ``eta`` decreases with ``l_m``; the FAQUAD diabatic error
    scales like ``eta^2``, so the predicted FH extinction is
    ``-20 log10(eta)``. The smallest ``l_m`` whose extinction meets the target
    (and whose total length ``l_m + 2*sep_length`` fits ``max_length``) is
    returned, else the longest ``l_m`` within the budget.
    """
    eta_target = 10 ** (-target_extinction_db / 20.0)
    l_m_max = max(50.0, max_length - 2 * sep_length)
    candidates = np.linspace(50.0, l_m_max, 40)
    best = float(l_m_max)
    for l_m in candidates:
        eta = _eta_for_lm(kappa_0, g_0, dbeta_dtw, float(l_m), theta_max_deg)
        if eta <= eta_target:
            best = float(l_m)
            break
    return best


def design_faquad_filter(
    platform: TFPlatform,
    fh_wl: float,
    sh_wl: float,
    *,
    w_top: float | None = None,
    theta_max_deg: float = 3.0,
    target_extinction_db: float = 25.0,
    res: float = 0.04,
    compute_modes: Callable | None = None,
) -> FaquadFilterDesign:
    """Design + optimize a FAQUAD FH/SH filter on a thin-film platform.

    1. Pick the nominal single-mode-at-FH top width (unless given).
    2. FDE-calibrate the FH coupling ``kappa(g)`` and mismatch slope.
    3. Choose the shortest constant-gap length meeting the FH extinction target
       within the platform length budget.
    4. Build the Euler-S-bend FAQUAD layout and report the SH coupling.
    """
    if w_top is None:
        w_top = optimize_width(
            platform,
            fh_wl,
            theta_max_deg=theta_max_deg,
            target_extinction_db=target_extinction_db,
            res=res,
            compute_modes=compute_modes,
        )

    kappa_0, g_0, dbeta_dtw = calibrate_coupling(
        platform, w_top, fh_wl, res=res, compute_modes=compute_modes
    )

    # SH coupling diagnostic: kappa at the minimum gap (should be tiny)
    k0_sh = 2 * np.pi / sh_wl
    span = max(3.6, 1.5 * (w_top + platform.g_m))
    x0 = (w_top + platform.g_m) / 2
    n_p, n_m = _te_neffs(
        rib_structures(platform, sh_wl, [w_top, w_top], [-x0, x0], (-span, span)),
        sh_wl,
        _mesh(platform, (-span, span), res),
        num_modes=8,
        num_te=2,
        compute_modes=compute_modes,
    )
    kappa_sh = float(0.5 * k0_sh * abs(n_p - n_m))

    # length of one Euler S-bend separation for this theta_max (geometry only)
    sep_length = _sep_length(theta_max_deg, platform.g_f)
    l_m = optimize_lm(
        kappa_0,
        g_0,
        dbeta_dtw,
        theta_max_deg,
        sep_length,
        target_extinction_db,
        platform.max_length,
    )

    design = FaquadDesign(
        kappa_0,
        g_0,
        dbeta_dtw,
        l_m=l_m,
        theta_max_deg=theta_max_deg,
        g_f=platform.g_f,
    )
    component = faquad_combiner(design, w_top)
    extinction_db = float(-20.0 * np.log10(max(design.eta, 1e-6)))
    return FaquadFilterDesign(
        platform=platform,
        fh_wl=fh_wl,
        sh_wl=sh_wl,
        w_top=float(w_top),
        kappa_0=kappa_0,
        g_0=g_0,
        dbeta_dtw=dbeta_dtw,
        kappa_sh=kappa_sh,
        design=design,
        component=component,
        extinction_db=extinction_db,
    )


# --------------------------------------------------------------------------
# picklable parameter round-trip (for shipping a design to a worker)
# --------------------------------------------------------------------------
def to_params(design: FaquadFilterDesign) -> dict:
    """A picklable parameter dict fully describing ``design`` (no gdsfactory).

    The gdsfactory ``Component`` of a :class:`FaquadFilterDesign` cannot be
    pickled; this returns the (picklable) platform, the FAQUAD geometry
    dataclass and the scalar design outputs. :func:`filter_from_params` rebuilds
    an equivalent design (re-creating the layout) from it.
    """
    return {
        "platform": design.platform,
        "fh_wl": design.fh_wl,
        "sh_wl": design.sh_wl,
        "w_top": design.w_top,
        "kappa_0": design.kappa_0,
        "g_0": design.g_0,
        "dbeta_dtw": design.dbeta_dtw,
        "kappa_sh": design.kappa_sh,
        "faquad": design.design,
        "extinction_db": design.extinction_db,
    }


def filter_from_params(
    platform: TFPlatform,
    fh_wl: float,
    sh_wl: float,
    w_top: float,
    kappa_0: float,
    g_0: float,
    dbeta_dtw: float,
    kappa_sh: float,
    faquad: FaquadDesign,
    extinction_db: float = 0.0,
) -> FaquadFilterDesign:
    """Rebuild a :class:`FaquadFilterDesign` (incl. layout) from scalar params.

    Does no optimization or FDE calibration - it just re-creates the parametric
    combiner layout for an already-chosen design (used to reconstruct a design
    from :func:`to_params` on a worker node).
    """
    component = faquad_combiner(faquad, w_top)
    return FaquadFilterDesign(
        platform=platform,
        fh_wl=fh_wl,
        sh_wl=sh_wl,
        w_top=w_top,
        kappa_0=kappa_0,
        g_0=g_0,
        dbeta_dtw=dbeta_dtw,
        kappa_sh=kappa_sh,
        design=faquad,
        component=component,
        extinction_db=extinction_db,
    )


# --------------------------------------------------------------------------
# layout + platform-aware extrusion / EME cells
# --------------------------------------------------------------------------
def faquad_combiner(
    design: FaquadDesign, w_top: float, num_points: int = 601
) -> gf.Component:
    """Parametric FAQUAD combiner layout for a designed device (paper Fig. 1a).

    Two rib waveguides whose top-width difference follows the FAQUAD taper and
    whose gap follows the constant-gap + Euler-S-bend separation profile (drawn
    polygons are the rib *top* widths).
    """
    c = gf.Component()
    z = np.linspace(-design.half_length, design.half_length, num_points)
    gap = design.gap(z)
    dtw = design.dtw(z)
    w_a = w_top + dtw / 2
    w_b = w_top - dtw / 2
    y_a_lo, y_a_hi = gap / 2, gap / 2 + w_a
    y_b_hi, y_b_lo = -gap / 2, -gap / 2 - w_b
    zs = z - z[0]
    for lo, hi in [(y_a_lo, y_a_hi), (y_b_lo, y_b_hi)]:
        upper = np.stack([zs, hi], axis=1)
        lower = np.stack([zs, lo], axis=1)[::-1]
        c.add_polygon(np.concatenate([upper, lower]), layer=LAYER_RIB)
    c.add_port(
        "in_bar",
        center=(0.0, float((y_b_lo[0] + y_b_hi[0]) / 2)),
        width=0.002 * round(float(w_b[0]) / 0.002),
        orientation=180,
        layer=LAYER_RIB,
    )
    for name, ys in [("out_bar", (y_b_lo, y_b_hi)), ("out_cross", (y_a_lo, y_a_hi))]:
        c.add_port(
            name,
            center=(float(zs[-1]), float((ys[0][-1] + ys[1][-1]) / 2)),
            width=0.002 * round(float(ys[1][-1] - ys[0][-1]) / 0.002),
            orientation=0,
            layer=LAYER_RIB,
        )
    return c


def device_structures(
    platform: TFPlatform, component: gf.Component, wl: float
) -> list[mw.Structure3D]:
    """Extrude a designed combiner with angled sidewalls onto the platform."""
    rules = {
        LAYER_RIB: [
            mw.GdsExtrusionRule(
                material=platform.core(wl),
                h_min=platform.slab_thickness,
                h_max=platform.core_thickness,
                buffer=platform.sidewall_run,
                sidewall_angle=platform.sidewall_deg,
            )
        ]
    }
    structs = mw.extrude_gds(component, rules)
    z_max = float(component.xmax)
    x_lo = float(component.ymin) - 1.5
    x_hi = float(component.ymax) + 1.5
    return structs + _background(platform, wl, z_max, (x_lo, x_hi))


def device_mesh(platform: TFPlatform, component: gf.Component, res: float) -> mw.Mesh2D:
    x_lo = float(component.ymin) - 1.2
    x_hi = float(component.ymax) + 1.2
    return _mesh(platform, (x_lo, x_hi), res)


def device_cells(
    design: FaquadFilterDesign,
    wl: float,
    num_cells: int = 128,
    res: float = 0.03,
) -> list[mw.Cell]:
    """Slice a designed device into adaptive FAQUAD EME cells."""
    platform = design.platform
    structs = device_structures(platform, design.component, wl)
    length = float(design.component.xmax)
    ls = adaptive_cell_lengths(design.design, num_cells)
    ls = ls * (length / float(np.sum(ls)))
    mesh = device_mesh(platform, design.component, res)
    return mw.create_cells(structs, mesh, ls, z_min=0.0)


def dispersive_core(platform: TFPlatform, wls: Any) -> mw.SampledAnisotropicMaterial:
    """A wavelength-sampled (dispersive) version of the platform core tensor.

    Samples the closed-form LN/LT uniaxial indices over ``wls`` so the resulting
    material is evaluated at the environment wavelength - which lets a *single*
    EME job sweep the whole wavelength range (the slice-group spectrum engine
    overrides ``env.wl`` per sweep point instead of rebuilding the cells).
    """
    wls = np.asarray(wls, dtype=float)
    cores = [platform.core(float(w)) for w in wls]  # AnisotropicMaterial per wl
    n = np.array(
        [np.sqrt(np.real(np.diag(c.eps))) for c in cores]  # ty: ignore[unresolved-attribute]
    )  # (N, 3) = (ne, no, no) per wavelength
    return mw.SampledAnisotropicMaterial.from_n(f"{platform.name}_core_disp", wls, n)


def dispersive_clad(platform: TFPlatform, wls: Any) -> mw.SampledMaterial:
    """A wavelength-sampled (dispersive) SiO2 under-cladding for the platform."""
    wls = np.asarray(wls, dtype=float)
    clads = [platform.clad(float(w)) for w in wls]  # IndexMaterial per wl
    n = np.array([c.n for c in clads], dtype=complex)  # ty: ignore[unresolved-attribute]
    return mw.SampledMaterial(name="SiO2_clad_disp", n=n, params={"wl": wls})


def device_cells_dispersive(
    design: FaquadFilterDesign,
    wls: Any,
    num_cells: int = 128,
    res: float = 0.03,
) -> list[mw.Cell]:
    """EME cells with a *dispersive* core/cladding sampled over ``wls``.

    Unlike :func:`device_cells` (which bakes the material at a single
    wavelength), these cells carry wavelength-sampled materials, so one EME job
    can sweep the full ``wls`` band (FH..SH and beyond) by varying only the
    environment wavelength - the decomposition used by the ``*_slurm`` examples.
    """
    wls = np.asarray(wls, dtype=float)
    platform = design.platform
    core_m = dispersive_core(platform, wls)
    clad_m = dispersive_clad(platform, wls)
    disp = replace(platform, core=lambda _wl: core_m, clad=lambda _wl: clad_m)
    structs = device_structures(disp, design.component, float(wls.mean()))
    length = float(design.component.xmax)
    ls = adaptive_cell_lengths(design.design, num_cells)
    ls = ls * (length / float(np.sum(ls)))
    mesh = device_mesh(disp, design.component, res)
    return mw.create_cells(structs, mesh, ls, z_min=0.0)


# --------------------------------------------------------------------------
# the design matrix + demo
# --------------------------------------------------------------------------
WAVELENGTH_PAIRS = [(1.55, 0.775), (1.35, 0.675), (1.06, 0.530)]
"""Target FH/SH (fundamental/second-harmonic) wavelength pairs [um]."""

CORE_THICKNESSES = [0.30, 0.40, 0.50, 0.60]
"""X-cut film thicknesses [um]."""

MATERIALS = {"TFLN": tfln_platform, "TFLT": tflt_platform}
"""Core-material platform factories (lithium niobate / lithium tantalate)."""


def platform_matrix() -> list[TFPlatform]:
    """Every (material, thickness) platform in the design matrix."""
    return [
        factory(t) for factory in MATERIALS.values() for t in CORE_THICKNESSES
    ]


def design_matrix(
    res: float = 0.04, compute_modes: Callable | None = None
) -> list[FaquadFilterDesign]:
    """Design a FAQUAD filter for every (material, thickness, FH/SH) combination."""
    designs: list[FaquadFilterDesign] = []
    for platform in platform_matrix():
        for fh, sh in WAVELENGTH_PAIRS:
            designs.append(
                design_faquad_filter(
                    platform, fh, sh, res=res, compute_modes=compute_modes
                )
            )
    return designs


def _summary(designs: list[FaquadFilterDesign]) -> dict[str, dict[str, float]]:
    return {
        f"{d.platform.name}/{d.fh_wl * 1e3:.0f}-{d.sh_wl * 1e3:.0f}nm": {
            "w_top_nm": round(d.w_top * 1e3, 0),
            "l_m_um": round(d.design.l_m, 0),
            "length_um": round(d.total_length, 0),
            "kappa0_fh_per_mm": round(d.kappa_0 * 1e3, 2),
            "kappa_sh_per_mm": round(d.kappa_sh * 1e3, 4),
            "eta": round(d.eta, 3),
            "pred_er_db": round(d.extinction_db, 1),
        }
        for d in designs
    }


def faquad_label(design: FaquadFilterDesign) -> str:
    """Human-readable design key (platform + FH/SH pair)."""
    return f"{design.platform.name}/{design.fh_wl * 1e3:.0f}-{design.sh_wl * 1e3:.0f}nm"


def _analysis_settings(design: FaquadFilterDesign) -> dict:
    """Per-design settings for the broad-band transmission + FH/SH field analysis."""
    from examples.papers import _analysis

    return {
        "label": faquad_label(design),
        "num_cells": _resolution.num_cells(low=12, medium=48),
        "num_modes": _resolution.num_modes(low=2, medium=4),
        "device_res": pick(low=0.08, medium=0.05, high=0.035),
        "backend": _backends.backend_name(),
        "spectrum_wls": _analysis.faquad_band(
            design.fh_wl, design.sh_wl, n=pick(low=9, medium=41, high=121)
        ),
        "prop_wls": np.array([design.sh_wl, design.fh_wl]),
        "num_z": pick(low=200, medium=600, high=1000),
    }


def analyze_design(
    design: FaquadFilterDesign,
    out_dir: Path | str,
    *,
    executor_factory: Callable | None = None,
    save_fields: bool = True,
) -> dict:
    """Compute + save a design's broad-band spectrum, GDS and FH/SH field plots.

    Runs the same analysis as ``kwolek_designer_slurm`` but in-process (the EME
    is distributed across local worker threads by default). Writes the
    ``*_spectrum.png``, ``*_propagation.png``, ``*_design.png``, ``*.gds`` and
    ``*_results.npz`` into ``out_dir`` and returns the summary dict.
    """
    from concurrent.futures import ThreadPoolExecutor

    from examples.papers import _analysis

    if executor_factory is None:
        workers = _backends.max_workers() or 4

        def executor_factory(_name: str) -> ThreadPoolExecutor:
            return ThreadPoolExecutor(max_workers=workers)

    run = _analysis.submit_faquad_run(
        to_params(design),
        _analysis_settings(design),
        executor_factory=executor_factory,
        out_dir=out_dir,
        save_fields=save_fields,
    )
    return run.gather()


def main() -> dict[str, object]:
    """Design the full matrix (a coarse subset at low resolution) and plot it.

    Writes the designer summary figure ``figures/kwolek_designer.png`` and, for
    *every* design, a dense broad-band (``0.8*SH .. 1.2*FH``) transmission
    spectrum, the device GDS and FH/SH propagating-field plots into
    ``figures/kwolek_designer/<design>/`` (see :func:`analyze_design`).
    """
    import matplotlib.pyplot as plt

    from examples.papers import _analysis
    from examples.papers._plot import plot_component

    FIGDIR.mkdir(exist_ok=True, parents=True)
    gf.gpdk.PDK.activate()
    res = pick(low=0.06, medium=0.04, high=0.03)

    if _resolution.is_low():
        platforms = [tfln_platform(0.30), tflt_platform(0.50)]
        pairs = WAVELENGTH_PAIRS[:1]
    else:
        platforms = platform_matrix()
        pairs = WAVELENGTH_PAIRS

    designs = [
        design_faquad_filter(p, fh, sh, res=res)
        for p in platforms
        for fh, sh in pairs
    ]
    summary = _summary(designs)

    fig = plt.figure(figsize=(13, 9))
    grid = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.1])

    # (a) optimized single-mode width vs thickness, per material/FH
    ax = fig.add_subplot(grid[0, 0])
    for mat in MATERIALS:
        for fh, _sh in pairs:
            xs = [
                d.platform.core_thickness * 1e3
                for d in designs
                if d.platform.name.startswith(mat) and d.fh_wl == fh
            ]
            ys = [
                d.w_top * 1e3
                for d in designs
                if d.platform.name.startswith(mat) and d.fh_wl == fh
            ]
            if xs:
                ax.plot(xs, ys, "o-", label=f"{mat} FH {fh * 1e3:.0f} nm")
    ax.set_xlabel("core thickness [nm]")
    ax.set_ylabel("optimized top width [nm]")
    ax.set_title("FAQUAD filter designer: optimized top width vs thickness")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(visible=True)

    # (b) FH vs SH coupling (the dichroic contrast) per design
    ax = fig.add_subplot(grid[1, 0])
    labels = list(summary)
    ax.semilogy(
        range(len(designs)),
        [d.kappa_0 * np.exp(-d.platform.g_m / d.g_0) * 1e3 for d in designs],
        "C0o",
        label="kappa_FH(g_m) [1/mm]",
    )
    ax.semilogy(
        range(len(designs)),
        [max(d.kappa_sh * 1e3, 1e-4) for d in designs],
        "C3s",
        label="kappa_SH(g_m) [1/mm]",
    )
    ax.set_xticks(range(len(designs)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_ylabel("coupling [1/mm]")
    ax.set_title("FH (couples) vs SH (decoupled) -> dichroic filtering")
    ax.legend(fontsize=7)
    ax.grid(visible=True, which="both")

    # (c) a representative designed layout
    ax = fig.add_subplot(grid[2, 0])
    d = designs[0]
    plot_component(d.component, ax)
    ax.set_title(
        f"{d.platform.name}, FH/SH {d.fh_wl * 1e3:.0f}/{d.sh_wl * 1e3:.0f} nm: "
        f"w={d.w_top * 1e3:.0f} nm, l_m={d.design.l_m:.0f} um, "
        f"L={d.total_length:.0f} um, eta={d.eta:.2f}"
    )
    ax.set_aspect("auto")

    fig.suptitle(
        "Generalized FAQUAD wavelength-filter designer (X-cut TFLN / TFLT)"
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "kwolek_designer.png", dpi=150)
    plt.close(fig)

    # per-design broad-band transmission spectra + GDS + FH/SH field plots
    analyses_root = FIGDIR / "kwolek_designer"
    analyses = {
        faquad_label(d): analyze_design(
            d, analyses_root / _analysis._file_stem(faquad_label(d))
        )
        for d in designs
    }
    return {"designs": summary, "analyses": analyses}


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
