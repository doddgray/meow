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
2. FAQUAD geometry: a constant minimum-gap section of length ``l_m`` (gap
   ``g_m``, Region I) is connected by Euler (clothoid) S-bend separations,
   parameterized by their lateral offset and maximum waveguide-axis angle,
   that smoothly open the gap from ``g_m`` to the final gap ``g_f`` past the
   decoupling gap ``g_c``. The FAQUAD mixing angle ``chi(z)`` follows from
   the constant-adiabaticity condition (Eq. 11) -- with the adiabaticity
   parameter ``eta`` fixed by ``chi(z->end) = pi`` (Eq. 12) -- evaluated for
   the S-bend coupling envelope; the top-width difference is
   ``dTW(z) = kappa(z) cot(chi(z)) / s`` in Regions I-II and is linearly
   tapered to zero beyond ``g_c`` (Region III), so it vanishes at the device
   ends.
3. EME validation: at the fundamental harmonic (FH, 1550 nm) the input
   adiabatically transfers to the cross port (combiner), while at the
   second harmonic (SH, 775 nm) the strongly-confined mode stays in the bar
   port, yielding the dichroic combiner/filter behavior of paper Figs. 1f/2.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import gdsfactory as gf
import numpy as np
from gdsfactory import path as gp
from scipy.integrate import cumulative_trapezoid

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

L_M = 120.0
"""Constant-gap (Region I) length [um].

Set from a converged-EME dichroic sweep. The device is multi-objective: the
FH cross transfer is coupler-like in ``l_m`` (it peaks near ``l_m ~ 150 um``
where the accumulated FH coupling completes the supermode swap), while the SH
extinction *degrades* with length (a longer device gives the weakly-coupled SH
mode more length to leak into the cross port). ``120 um`` trades a little FH
cross (~0.92 vs a ~0.96 peak) for the best SH bar/cross contrast. (The paper's
nominal value is longer; this reproduction optimizes the modeled stack, whose
FDE-calibrated coupling differs from the paper's.)
"""

THETA_MAX_DEG = 1.0
"""Maximum waveguide-axis angle of the Euler S-bend separation [deg].

The lateral separation that opens the gap from ``g_m`` to ``g_f`` is realized
with Euler (clothoid) S-bends whose tangent angle rises smoothly from zero to
this maximum (relative to the propagation/horizontal axis) at the inflection
and back to zero, giving curvature-continuous routing. It is kept small
(``1 deg``): the shallow-etched rib mode sits only ~0.02 in index above the
slab, so a steeper bend angle (the original ``3 deg``) radiates the guided mode
into the slab continuum -- converged EME shows that dropping 3 deg -> 1 deg cuts
the slab radiation loss from ~50% to a few percent.
"""

T_BOX = 1.2
"""Modeled SiO2 under-cladding thickness."""


def _ensure_pdk() -> None:
    """Activate the generic gdsfactory PDK if none is active (for euler paths)."""
    try:
        gf.get_active_pdk()
    except ValueError:
        gf.gpdk.PDK.activate()


def euler_sbend(
    lateral_offset: float, max_axis_angle_deg: float, npoints: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    """Sampled Euler (clothoid) S-bend parameterized by offset and max angle.

    The S-bend is two back-to-back Euler bends, ``euler(+theta)`` then
    ``euler(-theta)``, so the waveguide-axis angle (relative to the horizontal
    propagation axis) rises smoothly from zero to ``+max_axis_angle_deg`` at
    the inflection and back to zero, with continuous curvature throughout. The
    minimum radius is scaled so the net lateral displacement equals
    ``lateral_offset``.

    Args:
        lateral_offset: net transverse displacement of the bend [um].
        max_axis_angle_deg: peak tangent angle magnitude relative to the
            horizontal (propagation) axis [deg].
        npoints: samples per Euler segment.

    Returns:
        ``(z, y)`` arrays of the centerline, starting at ``(0, 0)`` with a
        monotonically increasing ``z`` (propagation) coordinate.
    """
    _ensure_pdk()
    theta = float(max_axis_angle_deg)

    def build(radius: float) -> tuple[np.ndarray, np.ndarray]:
        p1 = gp.euler(radius=radius, angle=+theta, p=1.0, npoints=npoints)
        p2 = gp.euler(radius=radius, angle=-theta, p=1.0, npoints=npoints)
        path = gp.Path()
        path.append(p1)
        path.append(p2)
        pts = np.asarray(path.points, dtype=float)
        return pts[:, 0] - pts[0, 0], pts[:, 1] - pts[0, 1]

    _, y = build(1.0)
    radius = lateral_offset / (y[-1] - y[0])
    return build(radius)


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
    """FAQUAD coupler geometry from calibrated kappa(g) data, with Euler S-bends.

    The constant-gap interaction section (Region I, length ``l_m``, gap
    ``g_m``) is connected on either side to Euler (clothoid) S-bend
    separations that smoothly open the gap from ``g_m`` to the final gap
    ``g_f`` (paper Fig. 1a/1b). Because the S-bend curvature is continuous,
    the gap ``g(z)`` and -- through the constant-adiabaticity condition -- the
    coupling angle ``chi(z)`` and top-width difference ``dTW(z)`` all vary
    smoothly with position. The decoupling gap ``g_c`` (end of the FAQUAD
    evolution) is reached partway along the S-bend; beyond it (Region III)
    ``dTW`` is linearly tapered back to zero at the device ends, as in the
    paper.

    Args:
        kappa_0: coupling prefactor [1/um] of ``kappa = kappa_0 exp(-g/g_0)``.
        g_0: coupling decay length [um].
        dbeta_dtw: phase-mismatch slope d(delta beta)/d(top width) [1/um^2].
        l_m: length of the constant-gap (Region I) section [um].
        theta_max_deg: maximum waveguide-axis angle of the Euler S-bend [deg].
        g_c: decoupling gap ending the FAQUAD evolution [um].
        g_f: final gap between the output ports [um].
        dtw_max: fabrication limit on the top-width difference [um].
    """

    def __init__(
        self,
        kappa_0: float,
        g_0: float,
        dbeta_dtw: float,
        l_m: float = L_M,
        theta_max_deg: float = THETA_MAX_DEG,
        g_c: float = G_C,
        g_f: float = G_F,
        dtw_max: float = 0.5,
    ) -> None:
        self.kappa_0 = kappa_0
        self.g_0 = g_0
        self.dbeta_dtw = dbeta_dtw
        self.l_m = l_m
        self.theta_max_deg = theta_max_deg
        self.g_c = g_c
        self.g_f = g_f
        self.dtw_max = dtw_max
        self.kappa_m = kappa_0 * np.exp(-G_M / g_0)

        # Euler S-bend that opens each waveguide laterally by (g_f - g_m)/2,
        # i.e. opens the edge-to-edge gap from g_m to g_f.
        z_sb, y_sb = euler_sbend((g_f - G_M) / 2.0, theta_max_deg)
        self._z_sb = l_m / 2 + z_sb  # device coordinate (positive half)
        self._g_sb = G_M + 2 * y_sb  # gap along the S-bend
        self.l_sep = float(z_sb[-1])  # S-bend (separation) length per side
        self.half_length = float(self._z_sb[-1])
        # device coordinate where the gap reaches the decoupling gap g_c
        self.z_c = float(np.interp(g_c, self._g_sb, self._z_sb))

        # chi(z) for a constant adiabaticity parameter eta, integrated
        # numerically: cos(chi) = -2 eta int_0^z kappa dz', with chi(0) = pi/2
        # and eta fixed by chi(half_length) = pi (paper Eq. 11-12 generalized
        # to the Euler-S-bend coupling envelope).
        zg = np.linspace(-self.half_length, self.half_length, 8001)
        integral = cumulative_trapezoid(self.kappa(zg), zg, initial=0.0)
        integral -= np.interp(0.0, zg, integral)
        self.eta = 1.0 / (2.0 * integral[-1])
        self._zg = zg
        self._chi = np.arccos(np.clip(-2.0 * self.eta * integral, -1.0, 1.0))
        # signed top-width difference at the decoupling gap (+z_c side, < 0),
        # from which Region III is linearly tapered to zero.
        self._dtw_c = float(self._dtw_faquad(self.z_c))

    def gap(self, z: float | np.ndarray) -> np.ndarray:
        """Edge-to-edge gap g(z); z=0 is the device center.

        Constant ``g_m`` in Region I, then the Euler S-bend profile that opens
        smoothly to ``g_f`` (symmetric about z=0).
        """
        z = np.asarray(z, dtype=float)
        az = np.abs(z)
        return np.where(
            az <= self.l_m / 2,
            G_M,
            np.interp(az, self._z_sb, self._g_sb),
        )

    def kappa(self, z: float | np.ndarray) -> np.ndarray:
        """Coupling profile kappa(z) = kappa_0 exp(-g(z)/g_0) (paper Eq. 10)."""
        return self.kappa_0 * np.exp(-self.gap(z) / self.g_0)

    def chi(self, z: float | np.ndarray) -> np.ndarray:
        """FAQUAD mixing angle chi(z), chi(0) = pi/2, monotonic 0 -> pi."""
        return np.interp(np.asarray(z, dtype=float), self._zg, self._chi)

    def _dtw_faquad(self, z: float | np.ndarray) -> np.ndarray:
        """Raw FAQUAD top-width difference dTW = 2 kappa cot(chi) / s.

        The supermode mixing angle satisfies ``tan(chi) = kappa / delta`` with
        the *half* mismatch ``delta = (beta_A - beta_B)/2``, so the full
        mismatch is ``2 kappa cot(chi)`` and the top-width difference is
        ``dTW = 2 kappa cot(chi) / s`` (clipped to the fabrication limit).
        """
        chi = self.chi(z)
        with np.errstate(divide="ignore", invalid="ignore"):
            dbeta = 2 * self.kappa(z) / np.tan(chi)
        dtw = np.nan_to_num(dbeta / self.dbeta_dtw, nan=0.0)
        return np.clip(dtw, -self.dtw_max, self.dtw_max)

    def dtw(self, z: float | np.ndarray) -> np.ndarray:
        """Top-width difference dTW(z), tapered linearly to zero past g_c.

        In Regions I-II (gap up to the decoupling gap ``g_c``) dTW follows the
        FAQUAD prescription; in Region III it is linearly tapered to zero at
        the device ends, enhancing fabrication tolerance and bandwidth (paper
        Sec. 2, Fig. 1b).
        """
        z = np.asarray(z, dtype=float)
        az = np.abs(z)
        span = self.half_length - self.z_c
        ramp = np.clip((self.half_length - az) / span, 0.0, 1.0)
        return np.where(
            az <= self.z_c,
            self._dtw_faquad(z),
            np.sign(z) * self._dtw_c * ramp,
        )

    def __repr__(self) -> str:
        return (
            f"FaquadDesign(l_m={self.l_m:.1f}, theta_max_deg={self.theta_max_deg:.1f}, "
            f"l_sep={self.l_sep:.1f}, z_c={self.z_c:.1f}, eta={self.eta:.3f})"
        )


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
    num_modes: int = 8,
    num_te: int = 2,
    compute_modes: Callable | None = None,
) -> list[float]:
    """Real effective indices of the first TE modes of a cross-section."""
    compute_modes = compute_modes or mw.compute_modes
    cell = mw.Cell(structures=structures, mesh=mesh, z_min=0.0, z_max=1.0)
    cs = mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=wl, T=25.0))
    modes = compute_modes(cs, num_modes=num_modes)
    te = [m for m in modes if m.te_fraction > 0.5]
    return [float(np.real(m.neff)) for m in te[:num_te]]


@lru_cache(maxsize=8)
def calibrate(
    wl: float = 1.55, res: float = 0.04, compute_modes: Callable | None = None
) -> tuple[float, float, float]:
    """Extract (kappa_0, g_0, dbeta_dtw) from FDE solves (workflow step 1).

    ``compute_modes`` selects the FDE backend (default: tidy3d); it is part of
    the memoization key, so different backends are cached separately.
    """
    mesh = calib_mesh(res)
    k0 = 2 * np.pi / wl

    gaps = np.array([G_M, 1.0, G_C])
    kappas = []
    for g in gaps:
        x0 = (W_TOP + g) / 2
        n_p, n_m = solve_te_neffs(
            rib_structures(wl, [W_TOP, W_TOP], [-x0, x0]),
            wl,
            mesh,
            compute_modes=compute_modes,
        )
        kappas.append(0.5 * k0 * (n_p - n_m))
    slope, intercept = np.polyfit(gaps, np.log(np.asarray(kappas)), 1)
    g_0 = -1.0 / slope
    kappa_0 = float(np.exp(intercept))

    dws = np.array([-0.05, 0.0, 0.05])
    neffs = [
        solve_te_neffs(
            rib_structures(wl, [W_TOP + dw], [0.0]),
            wl,
            mesh,
            4,
            1,
            compute_modes=compute_modes,
        )[0]
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
    l_m: float = L_M,
    theta_max_deg: float = THETA_MAX_DEG,
    w_top: float = W_TOP,
    num_points: int = 601,
) -> gf.Component:
    """Parametric FAQUAD wavelength combiner layout (paper Fig. 1a).

    Two rib waveguides whose top-width difference follows the FAQUAD taper
    and whose gap follows the constant-gap + Euler-S-bend separation profile.
    The drawn polygons are the rib *top* widths; the angled sidewalls are
    added at extrusion time.
    """
    design = FaquadDesign(kappa_0, g_0, dbeta_dtw, l_m, theta_max_deg)
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
        width=0.002 * round(float(w_b[0]) / 0.002),
        orientation=180,
        layer=LAYER_RIB,
    )
    c.add_port(
        "out_bar",
        center=(float(zs[-1]), float((y_b_lo[-1] + y_b_hi[-1]) / 2)),
        width=0.002 * round(float(w_b[-1]) / 0.002),
        orientation=0,
        layer=LAYER_RIB,
    )
    c.add_port(
        "out_cross",
        center=(float(zs[-1]), float((y_a_lo[-1] + y_a_hi[-1]) / 2)),
        width=0.002 * round(float(w_a[-1]) / 0.002),
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


def adaptive_cell_lengths(design: FaquadDesign, num_cells: int) -> np.ndarray:
    """EME cell lengths concentrated where the geometry changes fastest.

    Uniform slicing wastes cells on the slowly-varying constant-gap region
    while starving the separation regions, where the laterally moving
    waveguides need fine slicing to avoid staircase misalignment loss. Cell
    edges are placed at equal quantiles of a density combining the local
    gap slope with a uniform floor.
    """
    z = np.linspace(-design.half_length, design.half_length, 4001)
    dg = np.abs(np.gradient(design.gap(z), z))
    dchi = np.abs(np.gradient(design.chi(z), z))
    # the supermode basis rotates with chi even where the gap is constant;
    # both rates set how finely the EME staircase must sample the device.
    density = dg / max(np.mean(dg), 1e-12) + dchi / max(np.mean(dchi), 1e-12)
    density = density + 0.3 + 1e-12
    cum = np.cumsum(density)
    cum = (cum - cum[0]) / (cum[-1] - cum[0])
    edges = np.interp(np.linspace(0.0, 1.0, num_cells + 1), cum, z)
    return np.diff(edges)


def device_cells(
    component: gf.Component,
    wl: float,
    num_cells: int = 128,
    res: float = 0.04,
    design: FaquadDesign | None = None,
) -> list[mw.Cell]:
    """Discretize the combiner into EME cells (adaptive if a design is given)."""
    structs = device_structures(component, wl)
    length = float(component.xmax)
    if design is None:
        Ls = np.full(num_cells, length / num_cells)
    else:
        Ls = adaptive_cell_lengths(design, num_cells)
        Ls = Ls * (length / float(np.sum(Ls)))  # absorb rounding
    return mw.create_cells(structs, device_mesh(res), Ls, z_min=0.0)


def _mode_centroid(mode: mw.Mode) -> float:
    """Lateral (x) energy centroid of a mode's dominant field component."""
    density = np.abs(mode.Ex) ** 2
    return float(np.sum(mode.cs.mesh.Xx * density) / np.sum(density))


_NEFF_MARGIN = 0.005
"""Index margin above the slab continuum for a mode to count as rib-guided."""


def slab_neff(
    wl: float, cell: mw.Cell, *, compute_modes: Callable | None = None
) -> float:
    """Effective index of the bare TFLN-slab continuum on a cell's mesh.

    Solves the background-only cross-section (SiO2 box + 200 nm LN slab, no
    ribs) on the *same* mesh as ``cell``, so the returned ``neff`` is the top
    of the slab/box-mode ladder. Rib-guided modes are exactly those above it --
    this is the threshold that separates real bar/cross transmission from the
    spurious slab modes the hard-wall lateral window discretizes the slab
    continuum into (the shallow etch leaves the guided index only ~0.02-0.1
    above the slab).
    """
    solver = compute_modes or mw.compute_modes
    x = np.asarray(cell.mesh.x)
    structs = _background(wl, cell.z_max, x_span=(float(x.min()), float(x.max())))
    scell = mw.create_cells(
        structs, cell.mesh, np.array([cell.z_max - cell.z_min]), z_min=cell.z_min
    )[0]
    cs = mw.CrossSection.from_cell(cell=scell, env=mw.Environment(wl=wl, T=25.0))
    return max(float(np.real(m.neff)) for m in solver(cs, num_modes=2))


def rib_guided_indices(modes: list[mw.Mode], slab_index: float) -> list[int]:
    """Indices of the rib-guided modes (``neff`` above the slab continuum).

    Every other solved mode is a slab/box mode of the finite simulation window
    and carries radiation, not bar/cross power. The ribs are single-mode at the
    fundamental (FH) but multimode at the second harmonic (SH), so this returns
    *all* guided modes -- the transmission sums over them per side, which is
    what makes the metric stable in ``num_modes`` and correct at both bands.
    """
    return [
        i
        for i, m in enumerate(modes)
        if float(np.real(m.neff)) > slab_index + _NEFF_MARGIN
    ]


def input_launch_index(modes: list[mw.Mode], slab_index: float) -> int:
    """Index of the guided fundamental of waveguide B (the device input port).

    Waveguide B is drawn at negative x, so the launch mode is the highest-index
    rib-guided mode whose energy centroid is negative (falling back to the
    global fundamental if none qualifies).
    """
    guided = rib_guided_indices(modes, slab_index)
    left = [i for i in guided if _mode_centroid(modes[i]) < 0]
    pool = left or guided or list(range(len(modes)))
    return max(pool, key=lambda i: float(np.real(modes[i].neff)))


def bar_cross_transmission(
    cells: list[mw.Cell],
    wl: float,
    num_modes: int = 8,
    *,
    parallel: bool | None = None,
    compute_modes: Callable | None = None,
) -> tuple[float, float]:
    """(bar, cross) power for the guided TE mode injected in port B.

    The input is the guided fundamental localized in waveguide B (negative x).
    Transmission is the EME power coupled from it into the *rib-guided* output
    modes (:func:`rib_guided_indices`), summed per side: the bar port collects
    the negative-x modes (waveguide B), the cross port the positive-x modes
    (waveguide A). Summing every guided mode -- not just the fundamental --
    matters at the second harmonic, where the ribs are multimode. Power that
    ends up in the spurious slab/box modes of the finite window is radiation
    **loss**, so ``1 - bar - cross`` is the physical loss.

    Counting only the rib-guided modes (by an explicit slab-index threshold) is
    what makes the metric stable: summing *every* output mode by the sign of its
    centroid instead conflates slab radiation with transmission and swings
    wildly as more slab modes are resolved with ``num_modes``.

    The EME S-matrix is cascaded serially or, if ``parallel`` (or the
    ``MEOW_PAPER_PARALLEL`` environment variable) is set, with the parallel
    slice-group engine. ``compute_modes`` selects the FDE backend for the
    serial path and for the input/output mode classification; the parallel
    path always uses the deterministic default (tidy3d) backend.
    """
    from examples.papers._backends import device_s_matrix

    solver = compute_modes or mw.compute_modes
    env = mw.Environment(wl=wl, T=25.0)
    S, pm = device_s_matrix(
        cells,
        env,
        num_modes=num_modes,
        parallel=parallel,
        compute_modes=compute_modes,
    )
    S = np.asarray(S)
    cs_in = mw.CrossSection.from_cell(cell=cells[0], env=env)
    cs_out = mw.CrossSection.from_cell(cell=cells[-1], env=env)
    modes_in = solver(cs_in, num_modes=num_modes)
    modes_out = solver(cs_out, num_modes=num_modes)

    slab_in = slab_neff(wl, cells[0], compute_modes=solver)
    slab_out = slab_neff(wl, cells[-1], compute_modes=solver)
    in_idx = input_launch_index(modes_in, slab_in)

    def power(i: int) -> float:
        return float(np.abs(S[pm[f"right@{i}"], pm[f"left@{in_idx}"]]) ** 2)

    t_bar = t_cross = 0.0
    for i in rib_guided_indices(modes_out, slab_out):
        if _mode_centroid(modes_out[i]) < 0:
            t_bar += power(i)
        else:
            t_cross += power(i)
    return t_bar, t_cross
