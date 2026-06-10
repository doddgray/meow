"""Kwolek et al., "Ultra-broadband, Low-loss Wavelength Combiners and
Filters: Novel Designs and Experiments in Thin-film Lithium Niobate",
arXiv:2603.27034 (2026).

This example reproduces the paper's design workflow for fast-quasi-adiabatic
(FAQUAD) wavelength combiners/filters on thin-film lithium niobate with
gdsfactory (parametric layout) and meow (FDE + EME):

Platform (paper Sec. 3): 300 nm x-cut TFLN, ~100 nm etch depth (200 nm
slab), 65 degree sidewall angle (from the substrate plane; 25 degrees from
vertical), ~1.2 um top width, SiO2 under-cladding. The anisotropy of LN is
modeled with meow's ``AnisotropicMaterial`` (uniaxial tensor diagonal), and
the sloped ribs use the angled-sidewall GDS extrusion.

Design workflow (paper Sec. 2, Eqs. 8-12):

1. FDE calibration: the inter-waveguide coupling kappa(g) is extracted from
   the symmetric-supermode splitting at several gaps and fit to
   ``kappa = kappa_0 * exp(-g / g_0)``; the phase-mismatch slope
   ``s = d(delta beta)/d(TW)`` is extracted from isolated-waveguide solves.
2. Closed-form FAQUAD geometry: a constant minimum-gap section of length
   ``l_m`` (gap ``g_m``, Region I) is connected by curvature-limited cubic
   transitions ``g(z) = g_m + (2/3) a^2 (|z| - l_m/2)^3`` (Region II) up to
   the decoupling gap ``g_c``; the FAQUAD mixing angle ``chi(z)`` follows
   from the constant-adiabaticity condition (Eq. 11) with
   ``eta = 1 / (kappa_m (l_m + 2/3 Gamma(1/3) z_0))`` (Eq. 12), and the
   top-width-difference taper is ``dTW(z) = kappa(z) cot(chi(z)) / s``.
3. EME validation: at the fundamental harmonic (FH, 1550 nm) the input
   adiabatically transfers to the cross port (combiner), while at the
   second harmonic (SH, 775 nm) the strongly-confined mode stays in the bar
   port, yielding the dichroic combiner/filter behavior of paper Figs. 1f/2.
"""

from __future__ import annotations

from functools import lru_cache

import gdsfactory as gf
import numpy as np
from scipy.special import gamma as gamma_fn

import meow as mw

LAYER_RIB = (1, 0)

H_FILM = 0.30
H_SLAB = 0.20
SIDEWALL_DEG = 25.0  # 65 deg from the substrate plane = 25 deg from vertical
W_TOP = 1.2
"""Nominal waveguide top width."""

G_M = 0.80
"""Minimum (fabrication-limited) gap in Region I."""

G_C = 1.20
"""Gap at which residual coupling is negligible (end of FAQUAD evolution)."""

G_F = 3.0
"""Final gap between the output ports."""

T_BOX = 1.2
"""Modeled SiO2 under-cladding thickness."""


def ln_material(wl: float) -> mw.AnisotropicMaterial:
    """Uniaxial lithium niobate for an x-cut film, propagation along crystal y.

    Mode-plane axes: x (horizontal, in-plane) is the crystal z (extraordinary)
    axis, y (vertical) and the propagation axis are ordinary axes, so the
    permittivity tensor diagonal is ``(ne^2, no^2, no^2)``. Refractive
    indices follow the congruent-LN Sellmeier equations of Zelmon et al.
    (1997).
    """
    wl2 = wl**2
    no2 = 1 + 2.6734 * wl2 / (wl2 - 0.01764) + 1.2290 * wl2 / (wl2 - 0.05914)
    no2 += 12.614 * wl2 / (wl2 - 474.6)
    ne2 = 1 + 2.9804 * wl2 / (wl2 - 0.02047) + 0.5981 * wl2 / (wl2 - 0.0666)
    ne2 += 8.9543 * wl2 / (wl2 - 416.08)
    return mw.AnisotropicMaterial(
        name="LiNbO3_xcut",
        eps=[ne2, no2, no2],
        meta={"color": (0.6, 0.2, 0.6, 0.9)},
    )


def sio2_material(wl: float) -> mw.IndexMaterial:
    """Fused-silica cladding (Malitson 1965 Sellmeier)."""
    wl2 = wl**2
    n2 = 1 + 0.6961663 * wl2 / (wl2 - 0.0684043**2)
    n2 += 0.4079426 * wl2 / (wl2 - 0.1162414**2)
    n2 += 0.8974794 * wl2 / (wl2 - 9.896161**2)
    return mw.IndexMaterial(
        name="SiO2_clad", n=float(np.sqrt(n2)), meta={"color": (0.9, 0.9, 0.9, 0.9)}
    )


# --- FAQUAD closed-form geometry (paper Eqs. 8-12) ---


class FaquadDesign:
    """Closed-form FAQUAD coupler geometry from calibrated kappa(g) data.

    Args:
        kappa_0: coupling prefactor [1/um] of ``kappa = kappa_0 exp(-g/g_0)``.
        g_0: coupling decay length [um].
        dbeta_dtw: phase-mismatch slope d(delta beta)/d(top width) [1/um^2].
        l_m: length of the constant-gap (Region I) section [um].
        a: curvature parameter of the cubic gap transitions [1/um].
        dtw_max: maximum top-width difference [um].
        l_sep: length of the Region III separation/taper section [um].
    """

    def __init__(
        self,
        kappa_0: float,
        g_0: float,
        dbeta_dtw: float,
        l_m: float = 60.0,
        a: float = 0.012,
        dtw_max: float = 0.08,
        l_sep: float = 80.0,
    ) -> None:
        self.kappa_0 = kappa_0
        self.g_0 = g_0
        self.dbeta_dtw = dbeta_dtw
        self.l_m = l_m
        self.a = a
        self.dtw_max = dtw_max
        self.l_sep = l_sep

        self.kappa_m = kappa_0 * np.exp(-G_M / g_0)
        self.z_0 = (3 * g_0 / (2 * a**2)) ** (1 / 3)
        # paper Eq. 12: constant adiabaticity parameter
        self.eta = 1.0 / (self.kappa_m * (l_m + (2 / 3) * gamma_fn(1 / 3) * self.z_0))
        # half-length of the cubic transition (up to the decoupling gap g_c)
        self.z_c = self.l_m / 2 + (1.5 * (G_C - G_M) / a**2) ** (1 / 3)
        self.half_length = self.z_c + l_sep

    def gap(self, z: float | np.ndarray) -> np.ndarray:
        """Edge-to-edge gap g(z); z=0 is the device center (paper Eq. 9)."""
        z = np.asarray(z, dtype=float)
        az = np.abs(z)
        g = np.full_like(z, G_M)
        cubic = (az > self.l_m / 2) & (az <= self.z_c)
        g = np.where(cubic, G_M + (2 / 3) * self.a**2 * (az - self.l_m / 2) ** 3, g)
        # Region III: separate further to the final gap (cosine-smoothed)
        sep = az > self.z_c
        t = np.clip((az - self.z_c) / self.l_sep, 0.0, 1.0)
        g = np.where(sep, G_C + (G_F - G_C) * 0.5 * (1 - np.cos(np.pi * t)), g)
        return g

    def kappa(self, z: float | np.ndarray) -> np.ndarray:
        """Coupling profile kappa(z) = kappa_0 exp(-g(z)/g_0) (paper Eq. 10)."""
        return self.kappa_0 * np.exp(-self.gap(z) / self.g_0)

    def chi(self, z: float | np.ndarray) -> np.ndarray:
        """FAQUAD mixing angle chi(z) (paper Eq. 11), chi(0) = pi/2.

        The integral of the cubic-tail coupling
        ``int_0^u exp(-(t/z_0)^3) dt`` has the closed form
        ``(z_0/3) Gamma(1/3) P(1/3, (u/z_0)^3)`` with P the regularized lower
        incomplete gamma function, so chi is evaluated exactly.
        """
        from scipy.special import gammainc

        z = np.asarray(z, dtype=float)
        az = np.abs(z)
        inner = -2 * self.eta * self.kappa_m * np.clip(z, -self.l_m / 2, self.l_m / 2)
        u = np.maximum(az - self.l_m / 2, 0.0)
        tail = (self.z_0 / 3) * gamma_fn(1 / 3) * gammainc(1 / 3, (u / self.z_0) ** 3)
        cos_chi = inner - 2 * self.eta * self.kappa_m * np.sign(z) * tail
        return np.arccos(np.clip(cos_chi, -1.0, 1.0))

    def dtw(self, z: float | np.ndarray) -> np.ndarray:
        """Top-width difference dTW(z) = kappa cot(chi) / s, clipped."""
        chi = self.chi(z)
        with np.errstate(divide="ignore", invalid="ignore"):
            dbeta = self.kappa(z) / np.tan(chi)
        dtw = np.nan_to_num(dbeta / self.dbeta_dtw, nan=0.0, posinf=np.inf)
        return np.clip(dtw, -self.dtw_max, self.dtw_max)


# --- meow structures ---


def _background(wl: float, z_max: float, x_span: tuple[float, float]) -> list:
    """SiO2 under-cladding + LN slab (air top cladding after etching)."""
    ln = ln_material(wl)
    sio2 = sio2_material(wl)
    box = mw.Structure(
        material=sio2,
        geometry=mw.Box(
            x_min=x_span[0],
            x_max=x_span[1],
            y_min=-T_BOX,
            y_max=0.0,
            z_min=0.0,
            z_max=z_max,
        ),
        mesh_order=10,
    )
    slab = mw.Structure(
        material=ln,
        geometry=mw.Box(
            x_min=x_span[0],
            x_max=x_span[1],
            y_min=0.0,
            y_max=H_SLAB,
            z_min=0.0,
            z_max=z_max,
        ),
        mesh_order=8,
    )
    return [box, slab]


def rib_structures(
    wl: float, widths: list[float], centers: list[float]
) -> list[mw.Structure3D]:
    """Straight TFLN ribs (for FDE calibration), via angled-sidewall prisms."""
    ln = ln_material(wl)
    ribs = [
        mw.Structure(
            material=ln,
            geometry=mw.Prism(
                poly=np.array(
                    [
                        (0.0, x0 - w / 2),
                        (1.0, x0 - w / 2),
                        (1.0, x0 + w / 2),
                        (0.0, x0 + w / 2),
                    ]
                ),
                h_min=H_SLAB,
                h_max=H_FILM,
                axis="y",
                # the drawn polygon is the (wider) rib base; the *top* width
                # is the nominal w, so draw the base wider by the slope:
                sidewall_angle=SIDEWALL_DEG,
            ),
        )
        for w, x0 in zip(
            [
                w + 2 * (H_FILM - H_SLAB) * np.tan(np.deg2rad(SIDEWALL_DEG))
                for w in widths
            ],
            centers,
            strict=True,
        )
    ]
    return ribs + _background(wl, 1.0, x_span=(-4.5, 4.5))


def calib_mesh(res: float = 0.04) -> mw.Mesh2D:
    return mw.Mesh2D(
        x=np.arange(-3.6, 3.6 + res / 2, res),
        y=np.arange(-0.8, 0.9 + res / 2, res),
    )


def solve_te_neffs(
    structures: list[mw.Structure3D],
    wl: float,
    mesh: mw.Mesh2D,
    num_modes: int = 4,
    num_te: int = 2,
) -> list[float]:
    """Real effective indices of the first TE modes of a cross-section."""
    cell = mw.Cell(structures=structures, mesh=mesh, z_min=0.0, z_max=1.0)
    cs = mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=wl, T=25.0))
    modes = mw.compute_modes(cs, num_modes=num_modes)
    te = [m for m in modes if m.te_fraction > 0.5]
    return [float(np.real(m.neff)) for m in te[:num_te]]


@lru_cache(maxsize=8)
def calibrate(wl: float = 1.55, res: float = 0.04) -> tuple[float, float, float]:
    """Extract (kappa_0, g_0, dbeta_dtw) from FDE solves (workflow step 1)."""
    mesh = calib_mesh(res)
    k0 = 2 * np.pi / wl

    gaps = np.array([G_M, 1.0, G_C])
    kappas = []
    for g in gaps:
        x0 = (W_TOP + g) / 2
        n_p, n_m = solve_te_neffs(
            rib_structures(wl, [W_TOP, W_TOP], [-x0, x0]), wl, mesh
        )
        kappas.append(0.5 * k0 * (n_p - n_m))
    slope, intercept = np.polyfit(gaps, np.log(np.asarray(kappas)), 1)
    g_0 = -1.0 / slope
    kappa_0 = float(np.exp(intercept))

    dws = np.array([-0.05, 0.0, 0.05])
    neffs = [
        solve_te_neffs(rib_structures(wl, [W_TOP + dw], [0.0]), wl, mesh, 4, 1)[0]
        for dw in dws
    ]
    dbeta_dtw = float(k0 * np.polyfit(dws, neffs, 1)[0])
    return kappa_0, float(g_0), dbeta_dtw


# --- parametric layout ---


@gf.cell
def faquad_combiner(
    kappa_0: float,
    g_0: float,
    dbeta_dtw: float,
    l_m: float = 60.0,
    a: float = 0.012,
    dtw_max: float = 0.08,
    l_sep: float = 80.0,
    w_top: float = W_TOP,
    num_points: int = 121,
) -> gf.Component:
    """Parametric FAQUAD wavelength combiner layout (paper Fig. 1a).

    Two rib waveguides whose top-width difference follows the FAQUAD taper
    and whose gap follows the constant + cubic + separation profile. The
    drawn polygons are the rib *top* widths; the angled sidewalls are added
    at extrusion time.
    """
    design = FaquadDesign(kappa_0, g_0, dbeta_dtw, l_m, a, dtw_max, l_sep)
    c = gf.Component()
    z = np.linspace(-design.half_length, design.half_length, num_points)
    gap = design.gap(z)
    dtw = design.dtw(z)
    w_a = w_top + dtw / 2  # waveguide A (top, ends as the cross port)
    w_b = w_top - dtw / 2  # waveguide B (bottom, the input/bar port)

    y_a_lo = gap / 2
    y_a_hi = gap / 2 + w_a
    y_b_hi = -gap / 2
    y_b_lo = -gap / 2 - w_b

    zs = z - z[0]  # layout coordinates start at 0
    for lo, hi in [(y_a_lo, y_a_hi), (y_b_lo, y_b_hi)]:
        upper = np.stack([zs, hi], axis=1)
        lower = np.stack([zs, lo], axis=1)[::-1]
        c.add_polygon(np.concatenate([upper, lower]), layer=LAYER_RIB)

    c.add_port(
        "in_bar",
        center=(0.0, float((y_b_lo[0] + y_b_hi[0]) / 2)),
        width=float(w_b[0]),
        orientation=180,
        layer=LAYER_RIB,
    )
    c.add_port(
        "out_bar",
        center=(float(zs[-1]), float((y_b_lo[-1] + y_b_hi[-1]) / 2)),
        width=float(w_b[-1]),
        orientation=0,
        layer=LAYER_RIB,
    )
    c.add_port(
        "out_cross",
        center=(float(zs[-1]), float((y_a_lo[-1] + y_a_hi[-1]) / 2)),
        width=float(w_a[-1]),
        orientation=0,
        layer=LAYER_RIB,
    )
    return c


def device_structures(component: gf.Component, wl: float) -> list[mw.Structure3D]:
    """Extrude the combiner with 65-degree sidewalls into meow structures.

    The drawn polygon is the rib *top* width: grow it by the sidewall run
    so the extruded trapezoid has the drawn width at the rib top.
    """
    run = (H_FILM - H_SLAB) * np.tan(np.deg2rad(SIDEWALL_DEG))
    extrusion_rules = {
        LAYER_RIB: [
            mw.GdsExtrusionRule(
                material=ln_material(wl),
                h_min=H_SLAB,
                h_max=H_FILM,
                buffer=run,
                sidewall_angle=SIDEWALL_DEG,
            ),
        ],
    }
    structs = mw.extrude_gds(component, extrusion_rules)
    z_max = float(component.xmax)
    return structs + _background(wl, z_max, x_span=(-4.5, 4.5))


def device_mesh(res: float = 0.04) -> mw.Mesh2D:
    return mw.Mesh2D(
        x=np.arange(-3.8, 3.8 + res / 2, res),
        y=np.arange(-0.8, 0.9 + res / 2, res),
    )


def device_cells(
    component: gf.Component,
    wl: float,
    num_cells: int = 24,
    res: float = 0.04,
) -> list[mw.Cell]:
    structs = device_structures(component, wl)
    length = float(component.xmax)
    Ls = np.full(num_cells, length / num_cells)
    return mw.create_cells(structs, device_mesh(res), Ls, z_min=0.0)


def bar_cross_transmission(
    cells: list[mw.Cell],
    wl: float,
    num_modes: int = 4,
) -> tuple[float, float]:
    """(bar, cross) power for the fundamental TE mode injected in port B.

    The input fundamental mode of the asymmetric input cross-section is the
    one localized in waveguide B (negative x); transmission is classified by
    the lateral energy centroid of the output modes.
    """
    env = mw.Environment(wl=wl, T=25.0)
    css = [mw.CrossSection.from_cell(cell=c, env=env) for c in cells]
    modes = [mw.compute_modes(cs, num_modes=num_modes) for cs in css]
    S, pm = mw.compute_s_matrix(modes, cells=cells)
    S = np.asarray(S)

    def centroid(mode: mw.Mode) -> float:
        density = np.abs(mode.Ex) ** 2
        return float(np.sum(mode.cs.mesh.Xx * density) / np.sum(density))

    in_idx = min(
        range(min(2, len(modes[0]))), key=lambda i: centroid(modes[0][i])
    )  # input mode in waveguide B (bottom, drawn at negative x)
    t_bar = t_cross = 0.0
    for i, mode in enumerate(modes[-1]):
        power = float(np.abs(S[pm[f"right@{i}"], pm[f"left@{in_idx}"]]) ** 2)
        if centroid(mode) < 0:
            t_bar += power
        else:
            t_cross += power
    return t_bar, t_cross
