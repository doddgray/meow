"""Kwolek et al., "Ultra-broadband, Low-loss Wavelength Combiners and
Filters: Novel Designs and Experiments in Thin-film Lithium Niobate",
arXiv:2603.27034 (2026).

This example reproduces the paper's design workflow for fast-quasi-adiabatic
(FAQUAD) wavelength combiners/filters on thin-film lithium niobate with
gdsfactory (parametric layout) and meow (FDE + EME):

Platform (paper Sec. 3): 300 nm x-cut TFLN, ~100 nm etch depth (200 nm slab),
65 degree sidewall angle (from the substrate plane; 25 degrees from vertical),
~1.2 um top width, SiO2 under-cladding. The anisotropy of LN is modeled with
meow's ``AnisotropicMaterial`` (uniaxial tensor diagonal, extraordinary axis
in-plane and perpendicular to propagation), and the sloped ribs use the
angled-sidewall GDS extrusion.

A note on convergence: the fundamental harmonic transfers cleanly to the cross
port (FH cross ~0.9, low loss). The *second* harmonic is intrinsically harder --
the rib is strongly multimode at 775 nm, so the EME cascade needs many modes to
conserve power and the SH transmission is not converged to ~1% at the resolution
used here (it is a lower bound; the SH stays in the bar port but with appreciable
modal scattering). See the README for the convergence study and discussion.

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
"""Film / slab thickness [um]: the paper's 300 nm film with a ~100 nm etch.

A deeper-etched ridge was tried earlier to suppress the SH "leakage", but that
was an artifact of a launcher bug (the SH input mode was TM, not the TE
fundamental -- see :func:`input_launch_index`). With the correct TE launch the
deep ridge is in fact pathologically multimode at SH (the EME cascade cannot
conserve power at feasible mode counts), so the paper's shallow stack -- much
less multimode at SH and far better-conditioned for the cascade -- is used.
"""
SIDEWALL_DEG = 25.0  # 65 deg from the substrate plane = 25 deg from vertical
W_TOP = 1.2
"""Nominal waveguide top width."""

G_M = 0.80
"""Minimum (fabrication-limited) gap in Region I [um] (paper value)."""

G_C = 1.20
"""Gap at which residual coupling is negligible (end of FAQUAD evolution)."""

G_F = 3.0
"""Final gap between the output ports."""

L_M = 264.0
"""Constant-gap (Region I) length [um] (paper final design value).

The paper's final design uses ``l_m = 264 um`` together with the cubic-bend
curvature ``a = 2 mm^-1`` and decoupling gap ``g_c = 1.2 um``, giving an
adiabaticity parameter ``eta ~ 0.189`` (paper Sec. 2).
"""

A_CURV = 0.002
"""Cubic-bend curvature parameter ``a`` [1/um] (= 2 mm^-1, paper value).

Region II opens the gap as the cubic ``g_e(z) = (2/3) a^2 (z - l_m/2)^3 + g_m``
(paper Eq. 9), the paraxial Euler-bend approximation that keeps the coupling
envelope ``kappa(z) = kappa_m exp(-((|z|-l_m/2)/z_0)^3)`` (Eq. 10) with
``z_0 = (3 g_0 / (2 a^2))^(1/3)``. Smaller ``a`` -> gentler, longer bends.
"""

STRAIGHT_OUT = 10.0
"""Straight outer-waveguide length [um] appended past the Euler bend (Region III)."""

THETA_MAX_DEG = 1.0
"""Backward-compatible Euler-angle cap [deg] (largely vestigial).

The separation geometry is now the paper's cubic bend (Region II, set by ``a``)
joined to a slope-matched Euler bend that flattens to the straight outer
waveguide (Region III); the bend angles that result are sub-degree, so this cap
no longer binds. It is retained only so older call sites keep working.
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


def euler_to_straight(
    theta_c_deg: float, lateral_rise: float, npoints: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    """Euler (clothoid) bend that *starts* at angle ``theta_c`` and flattens.

    Region III of the coupler (paper Fig. 1a): the waveguide leaves the cubic
    bend (Region II) at tangent angle ``theta_c`` and must rejoin a straight
    outer waveguide. A genuine ``gdsfactory`` Euler segment ``euler(+theta_c)``
    turns from 0 to ``theta_c`` with curvature rising linearly along its arc
    length; traversed in reverse it starts at tangent ``theta_c`` (slope-matched
    to the cubic) and relaxes to 0 (straight) with the curvature falling
    smoothly to zero -- the curvature-optimized, low-loss transition the paper
    uses. The clothoid is scale-invariant, so the radius is chosen to give the
    required lateral rise ``(g_f - g_c)/2`` without changing the entry angle.

    Returns ``(z, y)`` of the centerline starting at ``(0, 0)`` with ``z``
    increasing and ``y`` rising by ``lateral_rise``.
    """
    _ensure_pdk()

    def build(radius: float) -> tuple[np.ndarray, np.ndarray]:
        p = gp.euler(radius=radius, angle=float(theta_c_deg), p=1.0, npoints=npoints)
        pts = np.asarray(p.points, dtype=float)
        z = pts[-1, 0] - pts[::-1, 0]  # reverse + mirror: slope theta_c -> 0
        y = pts[-1, 1] - pts[::-1, 1]
        return z - z[0], y - y[0]

    _, y1 = build(1.0)
    rise = float(y1[-1])
    radius = lateral_rise / rise if rise > 1e-9 else 1.0
    return build(radius)


def _ln_eps(wl: float) -> tuple[float, float]:
    """(ne^2, no^2) of congruent LiNbO3 (Zelmon et al. 1997 Sellmeier)."""
    wl2 = wl**2
    no2 = 1 + 2.6734 * wl2 / (wl2 - 0.01764) + 1.2290 * wl2 / (wl2 - 0.05914)
    no2 += 12.614 * wl2 / (wl2 - 474.6)
    ne2 = 1 + 2.9804 * wl2 / (wl2 - 0.02047) + 0.5981 * wl2 / (wl2 - 0.0666)
    ne2 += 8.9543 * wl2 / (wl2 - 416.08)
    return float(ne2), float(no2)


#: Material-model selector for the LiNbO3 core. ``"anisotropic"`` is the real
#: uniaxial crystal (tensor ``(ne^2, no^2, no^2)``); ``"isotropic"`` is the
#: deliberately *fake* isotropic LN that puts the extraordinary index ``ne`` on
#: every axis (tensor ``(ne^2, ne^2, ne^2)``). The two are run side by side to
#: bracket the influence of the LN anisotropy -- in particular the TE/TM mode
#: crossings at the SH band, which only the anisotropic model exhibits.
LN_MODELS = ("anisotropic", "isotropic")


def ln_material(wl: float, model: str = "anisotropic") -> mw.AnisotropicMaterial:
    """Lithium niobate for an x-cut film, propagation along crystal y.

    Mode-plane axes: x (horizontal, in-plane) is the crystal z (extraordinary)
    axis, y (vertical) and the propagation axis are ordinary axes, so the real
    (uniaxial) permittivity tensor diagonal is ``(ne^2, no^2, no^2)``.
    Refractive indices follow the congruent-LN Sellmeier equations of Zelmon
    et al. (1997).

    Args:
        wl: wavelength [um].
        model: ``"anisotropic"`` (the real uniaxial crystal) or ``"isotropic"``
            (a fake isotropic LN with the extraordinary index ``ne`` on all
            three axes, ``(ne^2, ne^2, ne^2)``). The isotropic model removes the
            TE/TM birefringence and so is free of the SH-band mode crossings the
            real crystal shows; comparing the two isolates that effect.
    """
    ne2, no2 = _ln_eps(wl)
    if model == "isotropic":
        eps = [ne2, ne2, ne2]
        name = "LiNbO3_xcut_iso"
    elif model == "anisotropic":
        eps = [ne2, no2, no2]
        name = "LiNbO3_xcut"
    else:
        msg = f"unknown LN material model {model!r}; expected one of {LN_MODELS}"
        raise ValueError(msg)
    return mw.AnisotropicMaterial(
        name=name,
        eps=eps,
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


VARIANTS = ("faquad_bends", "faquad_taper", "linear_taper")
"""Taper/bend variants compared in paper Fig. 2a.

- ``"faquad_bends"``: the proposed design -- the FAQUAD top-width taper
  ``dTW = 2 kappa cot(chi)/s`` is followed through the cubic separation bend
  (Region II) so the adiabaticity is held constant over the *whole* evolution.
- ``"faquad_taper"``: FAQUAD adiabaticity enforced only in the constant-gap
  Region I; ``dTW`` is frozen at its Region-I value through the bend (the
  conventional "optimize the straight section only" approach the paper improves
  on).
- ``"linear_taper"``: a naive linear top-width taper of the same peak ``dTW``.

All three share the *same* physical gap profile and device length; only the
top-width taper differs, exactly as in the paper's Fig. 2a comparison.
"""


class FaquadDesign:
    """FAQUAD coupler geometry from calibrated kappa(g) data (paper Fig. 1a).

    The longitudinal layout follows the paper's three regions:

    - **Region I** (``|z| <= l_m/2``): a constant minimum-gap ``g_m`` straight
      interaction section in which the FAQUAD top-width taper drives the
      supermode mixing angle ``chi``.
    - **Region II** (cubic bend): the gap opens as the cubic
      ``g_e(z) = (2/3) a^2 (z - l_m/2)^3 + g_m`` (paper Eq. 9), the paraxial
      Euler-bend approximation that yields the closed-form coupling envelope
      ``kappa(z) = kappa_m exp(-((|z|-l_m/2)/z_0)^3)`` (Eq. 10), continuing the
      FAQUAD evolution while the waveguides separate, up to the decoupling gap
      ``g_c``.
    - **Region III** (Euler bend -> straight): a genuine clothoid that *matches
      the cubic bend's exit angle* and relaxes the curvature to zero into a
      straight outer waveguide at the final gap ``g_f``; here ``dTW`` is tapered
      linearly to zero (Fig. 1b).

    Because the gap, the coupling ``kappa(z)``, the mixing angle ``chi(z)`` and
    the top-width difference ``dTW(z)`` all vary smoothly, the supermode evolves
    adiabatically; the adiabaticity parameter ``eta`` is fixed by
    ``chi(z -> end) = pi`` (Eq. 12).

    Args:
        kappa_0: coupling prefactor [1/um] of ``kappa = kappa_0 exp(-g/g_0)``.
        g_0: coupling decay length [um].
        dbeta_dtw: phase-mismatch slope d(delta beta)/d(top width) [1/um^2].
        l_m: length of the constant-gap (Region I) section [um].
        theta_max_deg: vestigial Euler-angle cap (see :data:`THETA_MAX_DEG`).
        g_c: decoupling gap ending the FAQUAD/cubic evolution [um].
        g_f: final gap between the straight output ports [um].
        dtw_max: fabrication limit on the top-width difference [um].
        a: cubic-bend curvature parameter [1/um] (paper ``a``; see
            :data:`A_CURV`).
        variant: top-width taper law, one of :data:`VARIANTS`.
        straight_out: straight outer-waveguide length appended past Region III.
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
        a: float = A_CURV,
        variant: str = "faquad_bends",
        straight_out: float = STRAIGHT_OUT,
    ) -> None:
        if variant not in VARIANTS:
            msg = f"unknown variant {variant!r}; expected one of {VARIANTS}"
            raise ValueError(msg)
        self.kappa_0 = kappa_0
        self.g_0 = g_0
        self.dbeta_dtw = dbeta_dtw
        self.l_m = l_m
        self.theta_max_deg = theta_max_deg
        self.g_c = g_c
        self.g_f = g_f
        self.dtw_max = dtw_max
        self.a = a
        self.variant = variant
        self.kappa_m = kappa_0 * np.exp(-G_M / g_0)
        # paper closed-form coupling-envelope length z_0 = (3 g_0 / (2 a^2))^1/3
        self.z_0 = float((3.0 * g_0 / (2.0 * a**2)) ** (1.0 / 3.0))

        self._build_gap_profile(straight_out)

        # chi(z) for a constant adiabaticity parameter eta, integrated
        # numerically: cos(chi) = -2 eta int_0^z kappa dz', with chi(0) = pi/2
        # and eta fixed by chi(half_length) = pi (paper Eqs. 11-12).
        zg = np.linspace(-self.half_length, self.half_length, 8001)
        integral = cumulative_trapezoid(self.kappa(zg), zg, initial=0.0)
        integral -= np.interp(0.0, zg, integral)
        self.eta = 1.0 / (2.0 * integral[-1])
        self._zg = zg
        self._chi = np.arccos(np.clip(-2.0 * self.eta * integral, -1.0, 1.0))
        # FAQUAD dTW at the constant-gap end (Region I/II boundary) and at the
        # decoupling gap, used by the taper variants / Region III linear taper.
        self._dtw_m = float(self._dtw_faquad(self.l_m / 2))
        self._dtw_c = float(self._dtw_faquad(self.z_c))

    def _build_gap_profile(self, straight_out: float) -> None:
        """Assemble the positive-half gap profile g(|z|) over Regions I-III."""
        l_m, a, g_c, g_f = self.l_m, self.a, self.g_c, self.g_f
        # Region II: cubic gap opening from g_m at l_m/2 to g_c at z_II.
        dz_c = ((g_c - G_M) * 3.0 / (2.0 * a**2)) ** (1.0 / 3.0)
        z_ii = l_m / 2 + dz_c
        z2 = np.linspace(l_m / 2, z_ii, 200)
        g2 = (2.0 / 3.0) * a**2 * (z2 - l_m / 2) ** 3 + G_M
        # Region III: Euler bend matched to the cubic exit slope, opening the
        # gap from g_c to g_f (half-gap rise (g_f - g_c)/2) and flattening.
        slope_c = a**2 * dz_c**2  # d(g/2)/dz at z_II (half-gap centerline)
        theta_c = float(np.degrees(np.arctan(slope_c)))
        ze, ye = euler_to_straight(theta_c, (g_f - g_c) / 2.0)
        z3 = z_ii + ze
        g3 = g_c + 2.0 * ye
        # Region I (constant) + straight outer tail.
        z1 = np.array([0.0, l_m / 2])
        g1 = np.array([G_M, G_M])
        z4 = np.array([z3[-1], z3[-1] + straight_out])
        g4 = np.array([g_f, g_f])
        zc = np.concatenate([z1, z2[1:], z3[1:], z4[1:]])
        gc = np.concatenate([g1, g2[1:], g3[1:], g4[1:]])
        # de-duplicate / enforce strictly increasing z for np.interp
        keep = np.concatenate([[True], np.diff(zc) > 1e-9])
        self._z_half = zc[keep]
        self._g_half = gc[keep]
        self.z_ii = float(z_ii)
        self.theta_c_deg = theta_c
        self.l_sep = float(self._z_half[-1] - l_m / 2)  # separation length / side
        self.half_length = float(self._z_half[-1])
        self.z_c = float(z_ii)  # gap reaches g_c exactly at z_II

    def gap(self, z: float | np.ndarray) -> np.ndarray:
        """Edge-to-edge gap g(z); z=0 is the device center (symmetric)."""
        az = np.abs(np.asarray(z, dtype=float))
        return np.interp(az, self._z_half, self._g_half)

    def kappa(self, z: float | np.ndarray) -> np.ndarray:
        """Coupling profile kappa(z) = kappa_0 exp(-g(z)/g_0) (paper Eq. 10)."""
        return self.kappa_0 * np.exp(-self.gap(z) / self.g_0)

    def chi(self, z: float | np.ndarray) -> np.ndarray:
        """FAQUAD mixing angle chi(z), chi(0) = pi/2, monotonic 0 -> pi."""
        return np.interp(np.asarray(z, dtype=float), self._zg, self._chi)

    def dbeta(self, z: float | np.ndarray) -> np.ndarray:
        """Phase-mismatch ``delta beta(z) = kappa(z) cot(chi(z))`` (paper Sec. 2)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nan_to_num(self.kappa(z) / np.tan(self.chi(z)), nan=0.0)

    def _dtw_faquad(self, z: float | np.ndarray) -> np.ndarray:
        """Raw FAQUAD top-width difference dTW = 2 kappa cot(chi) / s.

        The supermode mixing angle satisfies ``tan(chi) = kappa / delta`` with
        the *half* mismatch ``delta = (beta_A - beta_B)/2``, so the full
        mismatch is ``2 kappa cot(chi)`` and the top-width difference is
        ``dTW = 2 kappa cot(chi) / s`` (clipped to the fabrication limit).
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            dbeta = 2 * self.kappa(z) / np.tan(self.chi(z))
        dtw = np.nan_to_num(dbeta / self.dbeta_dtw, nan=0.0)
        return np.clip(dtw, -self.dtw_max, self.dtw_max)

    def dtw(self, z: float | np.ndarray) -> np.ndarray:
        """Top-width difference dTW(z) for the configured :attr:`variant`.

        All variants are antisymmetric in ``z`` and return to zero at the device
        ends. ``faquad_bends`` follows the FAQUAD prescription through the cubic
        bend; ``faquad_taper`` freezes ``dTW`` past Region I; ``linear_taper``
        ramps a linear taper of the same peak amplitude. In every case Region III
        (past the decoupling gap ``g_c``) is linearly tapered to zero.
        """
        z = np.asarray(z, dtype=float)
        az = np.abs(z)
        span = max(self.half_length - self.z_c, 1e-9)
        ramp = np.clip((self.half_length - az) / span, 0.0, 1.0)
        if self.variant == "linear_taper":
            peak = abs(self._dtw_c)
            rise = -peak * np.clip(az / max(self.z_c, 1e-9), 0.0, 1.0)
            inner = np.sign(z) * rise  # antisymmetric, sign matches FAQUAD
            return np.where(az <= self.z_c, inner, np.sign(z) * (-peak) * ramp)
        if self.variant == "faquad_taper":
            # FAQUAD only in Region I; frozen at its end value through the bend.
            inner = np.where(
                az <= self.l_m / 2,
                self._dtw_faquad(z),
                np.sign(z) * self._dtw_m,
            )
            return np.where(az <= self.z_c, inner, np.sign(z) * self._dtw_c * ramp)
        # faquad_bends (default): FAQUAD through Region II, linear in Region III.
        return np.where(
            az <= self.z_c,
            self._dtw_faquad(z),
            np.sign(z) * self._dtw_c * ramp,
        )

    def adiabaticity(
        self, z: float | np.ndarray, *, constant_width: bool = False
    ) -> np.ndarray:
        """Local adiabaticity eta(z) = (dchi/dz) sin(chi) / (2 kappa) (Eq. 7).

        With the FAQUAD ``chi(z)`` this is flat at the design ``eta``. With
        ``constant_width=True`` the phase-mismatch is *frozen* past Region I
        (constant top-width bends), so ``chi`` is recomputed from
        ``tan(chi) = kappa / delta_beta_frozen`` -- giving the strong deviation
        of the constant-width bends in paper Fig. 5c.
        """
        z = np.asarray(z, dtype=float)
        kap = self.kappa(z)
        if not constant_width:
            chi = self.chi(z)
        else:
            # Constant-width bends: the phase-mismatch is frozen at its
            # Region-I-end value, so chi follows tan(chi) = kappa / dbeta_frozen
            # (same 0..pi branch as the FAQUAD chi) instead of the engineered
            # FAQUAD profile -- the deviation the paper's Fig. 5c highlights.
            dbeta_frozen = float(self.dbeta(self.l_m / 2))
            az = np.abs(z)
            chi_bend = np.arctan2(kap, dbeta_frozen)  # in (0, pi/2)
            chi = np.where(
                az <= self.l_m / 2,
                self.chi(z),
                np.where(z >= 0, np.pi - chi_bend, chi_bend),
            )
        dchi = np.gradient(chi, z)
        return np.abs(dchi) * np.sin(chi) / (2.0 * np.maximum(kap, 1e-12))

    def __repr__(self) -> str:
        return (
            f"FaquadDesign(l_m={self.l_m:.1f}, a={self.a:.4f}, "
            f"z_ii={self.z_ii:.1f}, l_sep={self.l_sep:.1f}, "
            f"half_length={self.half_length:.1f}, eta={self.eta:.3f}, "
            f"variant={self.variant!r})"
        )


# --- meow structures ---


def _background(
    wl: float, z_max: float, x_span: tuple[float, float], model: str = "anisotropic"
) -> list:
    """SiO2 under-cladding + LN slab (air top cladding after etching)."""
    ln = ln_material(wl, model)
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
    wl: float,
    widths: list[float],
    centers: list[float],
    model: str = "anisotropic",
) -> list[mw.Structure3D]:
    """Straight TFLN ribs (for FDE calibration), via angled-sidewall prisms."""
    ln = ln_material(wl, model)
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
    return ribs + _background(wl, 1.0, x_span=(-4.5, 4.5), model=model)


def calib_mesh(res: float = 0.04) -> mw.Mesh2D:
    return mw.Mesh2D(
        x=np.arange(-3.6, 3.6 + res / 2, res),
        y=np.arange(-1.0, 1.0 + res / 2, res),
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


@lru_cache(maxsize=16)
def calibrate(
    wl: float = 1.55,
    res: float = 0.04,
    compute_modes: Callable | None = None,
    model: str = "anisotropic",
) -> tuple[float, float, float]:
    """Extract (kappa_0, g_0, dbeta_dtw) from FDE solves (workflow step 1).

    ``compute_modes`` selects the FDE backend (default: tidy3d) and ``model``
    selects the LN material model (:func:`ln_material`); both are part of the
    memoization key, so different backends/models are cached separately.
    """
    mesh = calib_mesh(res)
    k0 = 2 * np.pi / wl

    gaps = np.array([G_M, 1.0, G_C])
    kappas = []
    for g in gaps:
        x0 = (W_TOP + g) / 2
        n_p, n_m = solve_te_neffs(
            rib_structures(wl, [W_TOP, W_TOP], [-x0, x0], model),
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
            rib_structures(wl, [W_TOP + dw], [0.0], model),
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


def combiner_from_design(
    design: FaquadDesign, w_top: float = W_TOP, num_points: int = 601
) -> gf.Component:
    """Build the FAQUAD combiner layout from a :class:`FaquadDesign` via paths.

    The whole device is defined with a single straight ``gdsfactory`` path that
    is extruded with two **parametric-width** sections -- one per rib waveguide.
    Each section's :func:`width_function` traces the rib *top* width ``w(z)``
    and its :func:`offset_function` traces the rib centerline ``g(z)/2 + w/2``
    along the constant-gap interaction region (I), the cubic separation bend
    (II) and the matched Euler bend into the straight outer waveguide (III).
    Sampling the straight base path uniformly in ``z`` makes the path parameter
    ``t`` linear in ``z`` (``t = z/L``), so the profiles map directly. The
    angled rib sidewalls are added at extrusion (:func:`device_structures`).

    The top waveguide A ends as the cross port; the bottom waveguide B carries
    the input/bar port. (``gdsfactory`` measures section ``offset`` to the
    right of the travel direction, i.e. ``y = -offset``, which the offset
    functions account for.)
    """
    from gdsfactory import path as _gp
    from gdsfactory.cross_section import CrossSection, Section

    length = 2.0 * design.half_length
    zc = np.linspace(0.0, length, num_points)  # layout x coordinate
    z = zc - design.half_length  # device coordinate (-half .. +half)
    t = zc / length  # normalized path parameter in [0, 1]
    gap = design.gap(z)
    dtw = design.dtw(z)
    w_a = w_top + dtw / 2  # waveguide A (top, +y, cross port)
    w_b = w_top - dtw / 2  # waveguide B (bottom, -y, input/bar port)
    y_a = gap / 2 + w_a / 2  # rib-A centerline (> 0)
    y_b = -(gap / 2 + w_b / 2)  # rib-B centerline (< 0)

    def _section(y_center: np.ndarray, width: np.ndarray, name: str) -> Section:
        return Section(
            width=float(width[0]),
            offset=float(-y_center[0]),  # gdsfactory sign: y = -offset
            layer=LAYER_RIB,
            name=name,
            width_function=lambda tt: np.interp(np.asarray(tt), t, width),
            offset_function=lambda tt: -np.interp(np.asarray(tt), t, y_center),
        )

    xs = CrossSection(
        sections=(
            _section(y_a, w_a, "cross"),
            _section(y_b, w_b, "bar"),
        )
    )
    extruded = _gp.extrude(_gp.straight(length=length, npoints=num_points), xs)
    c = gf.Component()
    c.add_ref(extruded)
    c.flatten()

    def _q(w: float) -> float:
        return 0.002 * round(w / 0.002)

    c.add_port(
        "in_bar",
        center=(0.0, float(y_b[0])),
        width=_q(float(w_b[0])),
        orientation=180,
        layer=LAYER_RIB,
    )
    c.add_port(
        "in_cross",
        center=(0.0, float(y_a[0])),
        width=_q(float(w_a[0])),
        orientation=180,
        layer=LAYER_RIB,
    )
    c.add_port(
        "out_bar",
        center=(float(zc[-1]), float(y_b[-1])),
        width=_q(float(w_b[-1])),
        orientation=0,
        layer=LAYER_RIB,
    )
    c.add_port(
        "out_cross",
        center=(float(zc[-1]), float(y_a[-1])),
        width=_q(float(w_a[-1])),
        orientation=0,
        layer=LAYER_RIB,
    )
    return c


@gf.cell
def faquad_combiner(
    kappa_0: float,
    g_0: float,
    dbeta_dtw: float,
    l_m: float = L_M,
    theta_max_deg: float = THETA_MAX_DEG,
    w_top: float = W_TOP,
    num_points: int = 601,
    variant: str = "faquad_bends",
) -> gf.Component:
    """Parametric FAQUAD wavelength combiner layout (paper Fig. 1a).

    Builds the three-region cubic-bend FAQUAD layout (see
    :class:`FaquadDesign`) from a single ``gdsfactory`` path extruded with
    parametric rib top widths (:func:`combiner_from_design`). ``variant`` picks
    the top-width taper law compared in paper Fig. 2a.
    """
    design = FaquadDesign(
        kappa_0, g_0, dbeta_dtw, l_m, theta_max_deg, variant=variant
    )
    return combiner_from_design(design, w_top, num_points)


def device_structures(
    component: gf.Component, wl: float, model: str = "anisotropic"
) -> list[mw.Structure3D]:
    """Extrude the combiner with 65-degree sidewalls into meow structures.

    The drawn polygon is the rib *top* width: grow it by the sidewall run
    so the extruded trapezoid has the drawn width at the rib top.
    """
    run = (H_FILM - H_SLAB) * np.tan(np.deg2rad(SIDEWALL_DEG))
    extrusion_rules = {
        LAYER_RIB: [
            mw.GdsExtrusionRule(
                material=ln_material(wl, model),
                h_min=H_SLAB,
                h_max=H_FILM,
                buffer=run,
                sidewall_angle=SIDEWALL_DEG,
            ),
        ],
    }
    structs = mw.extrude_gds(component, extrusion_rules)
    z_max = float(component.xmax)
    return structs + _background(wl, z_max, x_span=(-4.5, 4.5), model=model)


def device_mesh(res: float = 0.04) -> mw.Mesh2D:
    # waveguides sit at x ~ +-2.1; a moderate lateral span keeps the discretized
    # slab continuum sparse (fewer spurious box modes near the guided index) while
    # the deep oxide / tall air margins keep the hard walls clear of the mode.
    return mw.Mesh2D(
        x=np.arange(-3.6, 3.6 + res / 2, res),
        y=np.arange(-1.0, 1.0 + res / 2, res),
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
    model: str = "anisotropic",
) -> list[mw.Cell]:
    """Discretize the combiner into EME cells (adaptive if a design is given)."""
    structs = device_structures(component, wl, model)
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


def output_port_x() -> float:
    """Lateral position [um] of each output rib center at the separated ends.

    At the device ends the gap is ``G_F`` and the ribs have the nominal top
    width, so the rib centers sit at ``+-(G_F/2 + W_TOP/2)``.
    """
    return G_F / 2 + W_TOP / 2


def _port_confinement(mode: mw.Mode, x_center: float, half_width: float) -> float:
    """Fraction of a mode's transverse power within ``half_width`` of ``x_center``."""
    density = np.abs(mode.Ex) ** 2
    x = np.asarray(mode.cs.mesh.Xx)
    inside = np.abs(x - x_center) < half_width
    return float(density[inside].sum() / density.sum())


def port_mode_indices(
    modes: list[mw.Mode],
    *,
    x_port: float | None = None,
    half_width: float = 1.0,
    thresh: float = 0.5,
) -> tuple[list[int], list[int]]:
    """Indices of the modes guided in the ``(bar, cross)`` output ribs.

    A genuine port mode is one whose power is *localized* in a single rib --
    ``> thresh`` of its energy within ``half_width`` of that rib's center. This
    confinement test is what separates real bar/cross transmission from the
    spurious slab/box modes of the finite simulation window, and it is robust
    where an ``neff``-threshold is not: at the second harmonic each rib is
    multimode and a *dense cluster of delocalized slab modes sits at the same
    neff as the higher-order rib modes*, so only the spatial localization tells
    them apart. The bar rib is at negative x (waveguide B), the cross rib at
    positive x (waveguide A). Every non-localized (slab) mode is excluded, so
    the leftover power is honest radiation loss; summing *all* the localized
    modes per side (not just the fundamental) is required at SH.
    """
    xp = output_port_x() if x_port is None else x_port
    bar = [
        i for i, m in enumerate(modes)
        if _port_confinement(m, -xp, half_width) > thresh
    ]
    cross = [
        i for i, m in enumerate(modes)
        if _port_confinement(m, xp, half_width) > thresh
    ]
    return bar, cross


def input_launch_index(
    modes: list[mw.Mode],
    *,
    x_port: float | None = None,
    half_width: float = 1.0,
    thresh: float = 0.5,
) -> int:
    """Index of the guided fundamental **TE** mode of waveguide B (the input).

    Waveguide B is the bar (negative-x) rib; the launch mode is the
    highest-``neff`` *TE-polarized* mode localized there. The TE restriction is
    essential: the combiner is a TE device (it is calibrated and designed for
    the TE supermode coupling), but at the second harmonic the strongly-confined
    ridge's highest-``neff`` mode is actually TM (vertical polarization) -- so
    ranking by ``neff`` alone would launch the wrong polarization at SH (and
    plotting its minor ``Ex`` component looks like a higher-order mode). Falls
    back to the highest-``neff`` localized mode, then the global fundamental, if
    no confined TE mode is found.
    """
    bar, _ = port_mode_indices(
        modes, x_port=x_port, half_width=half_width, thresh=thresh
    )
    pool = bar or list(range(len(modes)))
    te_pool = [i for i in pool if float(modes[i].te_fraction) > 0.5]
    return max(te_pool or pool, key=lambda i: float(np.real(modes[i].neff)))


def bar_cross_transmission(
    cells: list[mw.Cell],
    wl: float,
    num_modes: int = 8,
    *,
    parallel: bool | None = None,
    compute_modes: Callable | None = None,
) -> tuple[float, float]:
    """(bar, cross) power for the guided TE mode injected in port B.

    The input is the guided fundamental localized in waveguide B (the bar rib,
    negative x). Transmission is the EME power coupled from it into the modes
    *localized in each output rib* (:func:`port_mode_indices`), summed per side:
    the bar port collects the negative-x rib modes (waveguide B), the cross port
    the positive-x rib modes (waveguide A). Power that ends up in the delocalized
    slab/box modes of the finite window is radiation **loss**, so
    ``1 - bar - cross`` is the physical loss.

    Selecting port modes by spatial confinement -- not by an ``neff`` threshold
    or the sign of a centroid -- is what makes the metric stable in
    ``num_modes``: at the second harmonic each rib is multimode and a dense
    cluster of delocalized slab modes sits at the same neff as the higher-order
    rib modes, so only localization separates transmission from radiation.
    Summing all the localized modes per side (not just the fundamental) is
    likewise required at SH.

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

    in_idx = input_launch_index(modes_in)
    bar_out, cross_out = port_mode_indices(modes_out)

    def power(i: int) -> float:
        return float(np.abs(S[pm[f"right@{i}"], pm[f"left@{in_idx}"]]) ** 2)

    t_bar = float(sum(power(i) for i in bar_out))
    t_cross = float(sum(power(i) for i in cross_out))
    return t_bar, t_cross
