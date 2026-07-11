"""Shared infrastructure for **bent-waveguide EME** ring-bus coupler examples.

This module is the common engine behind the two literature-reproduction
examples (:mod:`bogaerts2012_point_coupler`, a straight-bus-to-ring *point*
coupler; :mod:`moille2019_pulley`, a wrap-around *pulley* coupler) and their
arbitrary-platform *designer* counterparts. It exercises meow's **bent** FDE
mode solver (``Mesh2D.bend_radius`` -> a conformal-transform curved-waveguide
solve) and the EME S-matrix engine.

Physics recap (all symbols in Optimode-like units: um, wavelengths in um)
------------------------------------------------------------------------
A microring couples to a bus through the evanescent overlap of the ring's
**bent** whispering-gallery mode with the bus mode. Two geometries:

* **Point coupler** -- a *straight* bus tangent to the ring. The interaction is
  lumped: the edge-to-edge gap opens quadratically away from the tangent point,
  ``g(z) = g0 + z**2 / (2 R)``, so the coupling is a single localized event with

      kappa_field = integral kappa0(g(z)) dz  (phase-matched)  ->  |kappa|^2 = sin^2(kappa_field).

  The ring mode is solved **bent** (radius ``R``); the bus mode is solved
  *straight*, so their phase mismatch ``delta = (beta_bus - beta_ring)/2`` is a
  genuine bent-vs-straight quantity.

* **Pulley coupler** -- the bus is itself *bent* to wrap the ring over an
  angular length ``Lc = R_bus * theta``. Bus and ring are concentric arcs about
  one center, so both are solved bent and their even/odd **supermodes** give the
  coupling directly. Over the uniform wrap the transfer follows the
  phase-mismatched directional-coupler law (Moille 2019, Eq. cited in the
  example)

      |kappa|^2(Lc) = kappa0^2 / (kappa0^2 + delta^2) * sin^2(sqrt(kappa0^2 + delta^2) * Lc),

  whose small-``Lc`` limit is the familiar ``(kappa0 Lc)^2 sinc^2(...)``.

The coupling ``kappa0`` and mismatch ``delta`` are extracted self-consistently
from **one** FDE supermode solve (:func:`supermode_coupling`): the ring/bus
supermode split fixes ``sqrt(kappa0^2 + delta^2)`` and the supermodes' rail
localization fixes the mixing angle, so ``kappa0`` and ``delta`` agree with the
field that actually beats. For the point coupler, ``kappa0(g)`` is sampled over a
few gaps and fit to ``A exp(-g / gamma)``. Everything downstream (spectra, ring
Q, extraction efficiency, and the designers that invert for a target coupling)
is built on these.

The module is deliberately platform-agnostic: :class:`StripPlatform` describes
any strip/ridge stack (SOI, Si3N4, TFLN, ...) and every routine takes a platform,
so the same code reproduces the papers and designs couplers on a new platform.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

import meow as mw

FIGDIR = Path(__file__).parent / "figures"
C0 = 299792458.0  # m/s (only ratios/units-cancelling uses)


# ======================================================================
# platform specification (arbitrary strip / ridge waveguide stack)
# ======================================================================
@dataclass(frozen=True)
class StripPlatform:
    """A strip/ridge waveguide platform for ring-bus couplers.

    The core is a rectangle of height ``thickness`` centered on ``y = 0``. The
    surrounding medium is ``clad`` (top + sides); an optional ``substrate`` fills
    below the core (``y < -thickness/2``), modeling e.g. a buried-oxide box under
    an air-clad nitride. For a symmetric oxide-clad stack leave ``substrate``
    ``None`` and set ``clad`` to oxide.

    Args:
        name: human-readable platform name (used in figure titles/filenames).
        core: ``wl -> Material`` factory for the guiding core.
        clad: ``wl -> Material`` factory for the top/side cladding.
        thickness: core (film) thickness [um].
        substrate: optional ``wl -> Material`` factory for the under-cladding.
        box_thickness: modeled substrate/under-cladding thickness [um].
        clad_thickness: modeled top-cladding thickness above the core [um].
        default_width: nominal single-mode core width [um] (design starting point).
    """

    name: str
    core: Callable[[float], mw.Material]
    clad: Callable[[float], mw.Material]
    thickness: float
    substrate: Callable[[float], mw.Material] | None = None
    box_thickness: float = 1.4
    clad_thickness: float = 1.2
    default_width: float = 0.5

    def core_mat(self, wl: float) -> mw.Material:
        return _as_material(self.core, wl)

    def clad_mat(self, wl: float) -> mw.Material:
        return _as_material(self.clad, wl)

    def sub_mat(self, wl: float) -> mw.Material:
        src = self.substrate if self.substrate is not None else self.clad
        return _as_material(src, wl)


def _as_material(src: Any, wl: float) -> mw.Material:
    """Resolve a ``wl -> Material`` factory *or* a bare Material to a Material."""
    if callable(src) and not isinstance(src, mw.MaterialBase):
        return src(wl)
    return src


# ---- platform presets --------------------------------------------------
def soi_platform(thickness: float = 0.22, *, buried: bool = True,
                 default_width: float = 0.45) -> StripPlatform:
    """220 nm silicon-on-insulator strip (Bogaerts 2012 / Xu 2008 platform).

    ``buried=True`` gives a symmetric SiO2-clad strip (the usual ring stack);
    ``buried=False`` gives an air top-clad strip on a buried-oxide box.
    """
    clad = mw.silicon_oxide if buried else mw.IndexMaterial(n=1.0, name="air")
    return StripPlatform(
        name=f"SOI-{thickness * 1e3:.0f}nm{'' if buried else '-airclad'}",
        core=mw.silicon, clad=clad,
        substrate=mw.silicon_oxide if not buried else None,
        thickness=thickness, default_width=default_width,
    )


def sin_platform(thickness: float = 0.78, *, buried: bool = True,
                 default_width: float = 1.5) -> StripPlatform:
    """Silicon-nitride strip on oxide (Moille 2019 / Hosseini 2010 platform).

    ``buried=True`` = symmetric SiO2 cladding (Moille's octave-comb ring);
    ``buried=False`` = air top-clad nitride on an oxide box (visible microdisks).
    """
    clad = mw.silicon_oxide if buried else mw.IndexMaterial(n=1.0, name="air")
    return StripPlatform(
        name=f"SiN-{thickness * 1e3:.0f}nm{'' if buried else '-airclad'}",
        core=mw.silicon_nitride, clad=clad,
        substrate=mw.silicon_oxide if not buried else None,
        thickness=thickness, default_width=default_width,
    )


# ======================================================================
# geometry -> cross-section (straight or bent) -> modes
# ======================================================================
def _rect(x_center: float, width: float, y_center: float, height: float) -> np.ndarray:
    return np.array([
        [x_center - width / 2, y_center - height / 2],
        [x_center + width / 2, y_center - height / 2],
        [x_center + width / 2, y_center + height / 2],
        [x_center - width / 2, y_center + height / 2],
    ])


def cross_section(
    platform: StripPlatform,
    cores: list[tuple[float, float]],
    wl: float,
    *,
    res: float = 0.03,
    bend_radius: float = np.inf,
    plane_center: tuple[float, float] = (0.0, 0.0),
    x_span: tuple[float, float] | None = None,
    num_pml: int = 10,
) -> mw.CrossSection:
    """A meow :class:`CrossSection` of one or more strip cores on the platform.

    ``cores`` is a list of ``(width, x_center)`` in um. A finite ``bend_radius``
    (with the radial axis = meow-x) selects the **bent** FDE solve; the radius is
    measured at ``plane_center`` (default the guide at ``x = 0``).
    """
    t = platform.thickness
    xs = [xc for _, xc in cores]
    ws = [w for w, _ in cores]
    if x_span is None:
        lo = min(x - w for x, w in zip(xs, ws)) - 1.6
        hi = max(x + w for x, w in zip(xs, ws)) + 1.6
        x_span = (lo, hi)
    span = max(abs(x_span[0]), abs(x_span[1])) + 2.0

    structs: list[mw.Structure2D] = [
        mw.Structure2D(  # full-domain cladding background (lowest priority)
            material=platform.clad_mat(wl),
            geometry=mw.Polygon2D(poly=_rect(0.0, 4 * span, 0.0, 4 * span)),
            mesh_order=12,
        )
    ]
    if platform.substrate is not None:  # under-cladding box below the core
        y_lo = -platform.box_thickness - t / 2
        structs.append(
            mw.Structure2D(
                material=platform.sub_mat(wl),
                geometry=mw.Polygon2D(
                    poly=_rect(0.0, 4 * span, (y_lo - t / 2) / 2, (t / 2 - y_lo)),
                ),
                mesh_order=10,
            )
        )
    for w, xc in cores:
        structs.append(
            mw.Structure2D(
                material=platform.core_mat(wl),
                geometry=mw.Polygon2D(poly=_rect(xc, w, 0.0, t)),
                mesh_order=5,
            )
        )

    y_lo = -platform.box_thickness - t / 2
    y_hi = platform.clad_thickness + t / 2
    mesh = mw.Mesh2D(
        x=np.arange(x_span[0], x_span[1] + res / 2, res),
        y=np.arange(y_lo, y_hi + res / 2, res),
        num_pml=(num_pml, num_pml),
        ez_interfaces=True,
        bend_axis=1,
        bend_radius=float(bend_radius),
        plane_center=plane_center,
    )
    return mw.CrossSection(
        structures=structs, mesh=mesh, env=mw.Environment(wl=wl, T=25.0)
    )


def solve(cs: mw.CrossSection, num_modes: int = 2) -> list[mw.Mode]:
    """Solve modes, sorted by descending real effective index."""
    modes = mw.compute_modes(cs, num_modes=num_modes)
    return sorted(modes, key=lambda m: float(np.real(m.neff)), reverse=True)


def _te_sorted(modes: list[mw.Mode]) -> list[mw.Mode]:
    te = [m for m in modes if float(mw.te_fraction(m)) >= 0.5]
    return te if te else modes


def isolated_neff(
    platform: StripPlatform, width: float, wl: float,
    *, bend_radius: float = np.inf, res: float = 0.03, x_center: float = 0.0,
) -> tuple[float, mw.Mode]:
    """Fundamental-TE effective index (and mode) of one isolated core.

    ``x_center`` places the core at a radial offset within a bent frame whose
    reference radius (at ``x = 0``) is ``bend_radius``. This matters for pulley
    phase-matching: the ring (``x = 0``) and the outer bus (``x = sep``) must be
    read out in the **same** frame so their indices share a reference arc length.
    """
    x_span = (x_center - width - 1.6, x_center + width + 1.6)
    cs = cross_section(platform, [(width, x_center)], wl,
                       res=res, bend_radius=bend_radius, x_span=x_span)
    modes = _te_sorted(solve(cs, num_modes=2))
    return float(np.real(modes[0].neff)), modes[0]


# ======================================================================
# coupling extraction: supermode split -> (kappa0, delta) -> A exp(-g/gamma)
# ======================================================================
def _ring_frac(mode: mw.Mode, sep: float) -> float:
    """Fraction of a mode's transverse energy on the ring rail (``x < sep/2``)."""
    xx = np.asarray(mode.cs.mesh.Xx)
    dens = (np.abs(np.asarray(mode.Ex)) ** 2 + np.abs(np.asarray(mode.Ey)) ** 2
            + np.abs(np.asarray(mode.Ez)) ** 2)
    return float(dens[xx < sep / 2].sum() / (dens.sum() + 1e-30))


def _select_ring_bus(modes: list[mw.Mode], sep: float) -> tuple[mw.Mode, mw.Mode]:
    """Pick the ring-branch and bus-branch fundamental supermodes.

    The ring/bus TE0 pair are the highest-``n_eff`` ring-majority and bus-majority
    modes (so a multimode ring's TE1 is skipped). Near phase matching both hybrids
    straddle 50/50 and this reduces to the top two by ``n_eff``.
    """
    ring = [m for m in modes if _ring_frac(m, sep) > 0.5]
    bus = [m for m in modes if _ring_frac(m, sep) <= 0.5]
    if ring and bus:
        m_ring = max(ring, key=lambda m: np.real(m.neff))
        m_bus = max(bus, key=lambda m: np.real(m.neff))
    else:
        m_ring, m_bus = modes[0], modes[1]
    pair = sorted([m_ring, m_bus], key=lambda m: np.real(m.neff), reverse=True)
    return pair[0], pair[1]


def supermode_split(
    platform: StripPlatform, w_ring: float, w_bus: float, gap: float, wl: float,
    *, bend_radius: float = np.inf, res: float = 0.03, num_modes: int = 6,
) -> tuple[float, float, list[mw.Mode]]:
    """Ring/bus fundamental supermode indices of two cores separated by ``gap``.

    The ring core sits at ``x = 0`` (so ``bend_radius`` is the ring centerline
    radius) and the bus at larger radius. Returns ``(n_higher, n_lower, [higher,
    lower])`` for the ring/bus TE0 branch pair (selected by rail localization, so
    a multimode ring's higher-order modes don't masquerade as the bus).
    """
    sep = gap + (w_ring + w_bus) / 2  # center-to-center
    cs = cross_section(platform, [(w_ring, 0.0), (w_bus, sep)], wl,
                       res=res, bend_radius=bend_radius)
    modes = _te_sorted(solve(cs, num_modes=num_modes))
    hi, lo = _select_ring_bus(modes, sep)
    return float(np.real(hi.neff)), float(np.real(lo.neff)), [hi, lo]


def symmetric_kappa0(
    platform: StripPlatform, w_ref: float, gap: float, wl: float,
    *, bend_radius: float = np.inf, res: float = 0.03,
) -> float:
    """Bare evanescent coupling ``kappa0`` [1/um] of a *symmetric* coupler.

    For two identical (phase-matched, ``delta = 0``) cores the supermode split is
    exactly ``2 kappa0``, so ``kappa0 = pi (n_even - n_odd) / lambda``. This
    isolates the coupling strength from any phase mismatch; the asymmetry of the
    real ring/bus enters separately through :func:`mismatch`.
    """
    n_even, n_odd, _ = supermode_split(
        platform, w_ref, w_ref, gap, wl, bend_radius=bend_radius, res=res
    )
    return float(np.pi * (n_even - n_odd) / wl)


def supermode_coupling(
    platform: StripPlatform, w_ring: float, w_bus: float, gap: float, wl: float,
    *, bend_radius: float, res: float = 0.03,
) -> tuple[float, float]:
    """Self-consistent ``(kappa0, |delta|)`` [1/um] from one bent supermode solve.

    A two-guide coupler is a 2-level system: the supermode split
    ``S = k0 (n_even - n_odd) = 2 sqrt(kappa0^2 + delta^2)`` fixes the *radius*, and
    the even supermode's rail localization ``r`` (energy fraction on the ring)
    fixes the *mixing angle*. Solving the 2-level model gives

        kappa0 = S sqrt(r (1 - r)) ,   |delta| = (S/2) |2r - 1| .

    At phase matching ``r -> 1/2`` (delocalized supermodes, ``delta -> 0``); a
    strongly mismatched coupler has rail-localized supermodes (``r -> 0 or 1``).
    This matches exactly what :func:`pulley_propagation` beats, so the coupling
    law and the field agree.
    """
    sep = gap + (w_ring + w_bus) / 2
    n_even, n_odd, modes = supermode_split(
        platform, w_ring, w_bus, gap, wl, bend_radius=bend_radius, res=res
    )
    s = (2 * np.pi / wl) * (n_even - n_odd)
    r = _ring_frac(modes[0], sep)
    kappa0 = float(s * np.sqrt(max(r * (1 - r), 0.0)))
    # even (faster) mode on the ring rail (r>0.5) => ring faster => bus slower => delta<0
    delta = float(0.5 * s * (2 * r - 1)) * (-1.0)
    return kappa0, delta


def mismatch(
    platform: StripPlatform, w_ring: float, w_bus: float, gap: float, wl: float,
    *, radius: float, mode: str, res: float = 0.03,
) -> float:
    """Signed phase mismatch ``delta`` [1/um] (bus faster -> positive).

    For a pulley the magnitude comes from the self-consistent supermode split +
    localization (:func:`supermode_coupling`); for a point coupler the straight
    bus is compared to the bent ring directly.
    """
    if mode == "pulley":
        return supermode_coupling(platform, w_ring, w_bus, gap, wl,
                                  bend_radius=radius, res=res)[1]
    k0 = 2 * np.pi / wl
    n_ring, _ = isolated_neff(platform, w_ring, wl, bend_radius=radius, res=res)
    n_bus, _ = isolated_neff(platform, w_bus, wl, bend_radius=np.inf, res=res)
    return float(0.5 * k0 * (n_bus - n_ring))


@dataclass
class CouplingModel:
    """Fitted evanescent coupling ``kappa0(g) = A exp(-g / gamma)`` [1/um].

    ``delta`` is the phase mismatch [1/um] at the reference geometry; ``gaps`` and
    ``kappa0s`` are the sampled calibration points (for plotting the fit).
    """

    A: float
    gamma: float
    delta: float
    gaps: np.ndarray
    kappa0s: np.ndarray
    wl: float
    platform_name: str = ""

    def kappa0(self, gap: float | np.ndarray) -> np.ndarray:
        return self.A * np.exp(-np.asarray(gap, float) / self.gamma)


def calibrate_coupling(
    platform: StripPlatform, w_ring: float, w_bus: float, gaps: np.ndarray, wl: float,
    *, radius: float, mode: str, res: float = 0.03,
) -> CouplingModel:
    """Calibrate the coupling model on a platform: ``kappa0(g)`` fit + ``delta``.

    Samples the bare coupling ``kappa0`` (symmetric split at the mean width) over
    ``gaps``, fits ``A exp(-g/gamma)``, and evaluates the ring/bus phase mismatch
    ``delta`` at the median gap. ``mode`` is ``'point'`` (straight bus) or
    ``'pulley'`` (concentric bent bus).
    """
    w_ref = 0.5 * (w_ring + w_bus)
    br = radius if mode == "pulley" else np.inf
    gaps = np.asarray(gaps, float)
    k0s = np.asarray([
        symmetric_kappa0(platform, w_ref, float(g), wl, bend_radius=br, res=res)
        for g in gaps
    ])
    good = k0s > 0
    if good.sum() >= 2:
        slope, intercept = np.polyfit(gaps[good], np.log(k0s[good]), 1)
        gamma = float(-1.0 / slope) if slope < 0 else float(np.ptp(gaps) or 1.0)
        A = float(np.exp(intercept))
    else:  # degenerate: fall back to a single point
        A = float(k0s.max() if k0s.size else 0.0)
        gamma = float(np.ptp(gaps) or 1.0)
    delta = mismatch(platform, w_ring, w_bus, float(np.median(gaps)), wl,
                     radius=radius, mode=mode, res=res)
    return CouplingModel(A=A, gamma=gamma, delta=float(delta),
                         gaps=gaps, kappa0s=k0s, wl=wl,
                         platform_name=platform.name)


# ======================================================================
# coupled-mode transfer: point (parabolic gap) and pulley (uniform arc)
# ======================================================================
def _cmt_cross_power(kappa_z: np.ndarray, delta: float, dz: float) -> float:
    """|b(L)|^2 for da/dz = -i(delta a + kappa b), db/dz = -i(-delta b + kappa a).

    Integrates the two-mode CMT with a(0)=1 (bus), b(0)=0 (ring) via RK4, for a
    z-dependent coupling ``kappa_z`` and constant mismatch ``delta``.
    """
    a = 1.0 + 0j
    b = 0.0 + 0j

    def deriv(a: complex, b: complex, k: float) -> tuple[complex, complex]:
        return (-1j * (delta * a + k * b), -1j * (-delta * b + k * a))

    for i in range(len(kappa_z) - 1):
        k0 = kappa_z[i]
        km = 0.5 * (kappa_z[i] + kappa_z[i + 1])
        k1 = kappa_z[i + 1]
        da1, db1 = deriv(a, b, k0)
        da2, db2 = deriv(a + 0.5 * dz * da1, b + 0.5 * dz * db1, km)
        da3, db3 = deriv(a + 0.5 * dz * da2, b + 0.5 * dz * db2, km)
        da4, db4 = deriv(a + dz * da3, b + dz * db3, k1)
        a += dz / 6 * (da1 + 2 * da2 + 2 * da3 + da4)
        b += dz / 6 * (db1 + 2 * db2 + 2 * db3 + db4)
    return float(abs(b) ** 2)


def point_cross_power(
    model: CouplingModel, gap: float, radius: float,
    *, half_window: float | None = None, npts: int = 801,
) -> float:
    """Power coupling ``|kappa|^2`` of a straight-bus point coupler.

    The edge-to-edge gap opens as ``g(z) = gap + z^2 / (2 R)`` about the tangent
    point; the coupling is integrated over ``|z| <= half_window`` (default a few
    ``sqrt(R gamma)`` where the Gaussian envelope has decayed).
    """
    if half_window is None:
        half_window = 4.0 * np.sqrt(max(radius * model.gamma, 1e-6))
    z = np.linspace(-half_window, half_window, npts)
    g = gap + z**2 / (2.0 * radius)
    kz = model.kappa0(g)
    return _cmt_cross_power(kz, model.delta, z[1] - z[0])


def point_cross_power_analytic(model: CouplingModel, gap: float, radius: float) -> float:
    """Phase-matched closed form ``sin^2(A e^{-gap/gamma} sqrt(2 pi R gamma))``.

    The Gaussian integral of ``kappa0(g(z))`` over the parabolic gap gives the
    effective interaction ``kappa_field = A e^{-gap/gamma} sqrt(2 pi R gamma)``;
    ``|kappa|^2 = sin^2(kappa_field)``. Ignores the (usually small) mismatch.
    """
    kappa_field = model.A * np.exp(-gap / model.gamma) * np.sqrt(
        2 * np.pi * radius * model.gamma
    )
    return float(np.sin(kappa_field) ** 2)


def pulley_cross_power(kappa0: float, delta: float, length: float) -> float:
    """Power coupling ``|kappa|^2`` of a uniform pulley over arc length ``Lc``.

    ``|kappa|^2 = kappa0^2/(kappa0^2+delta^2) * sin^2(sqrt(kappa0^2+delta^2) Lc)``
    (phase-mismatched directional coupler; Moille 2019).
    """
    s = np.hypot(kappa0, delta)
    if s <= 0:
        return 0.0
    return float(kappa0**2 / s**2 * np.sin(s * length) ** 2)


def pulley_cross_power_sinc(kappa0: float, delta: float, length: float) -> float:
    """Small-transfer ``(kappa0 Lc)^2 sinc^2(...)`` limit (Moille 2019 Eq. form)."""
    arg = length * np.hypot(kappa0, delta)
    return float((kappa0 * length) ** 2 * np.sinc(arg / np.pi) ** 2)


# ======================================================================
# ring-resonator observables (transfer functions, Q, extraction efficiency)
# ======================================================================
def all_pass_transmission(t: float, a: float, phi: np.ndarray) -> np.ndarray:
    """|Through|^2 of an all-pass ring: ``(a^2-2at cos phi+t^2)/(1-2at cos phi+a^2 t^2)``."""
    num = a**2 - 2 * a * t * np.cos(phi) + t**2
    den = 1 - 2 * a * t * np.cos(phi) + (a * t) ** 2
    return num / den


def add_drop_through_drop(
    t1: float, t2: float, a: float, phi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """|Through|^2 and |Drop|^2 of a symmetric add-drop ring."""
    den = np.abs(1 - t1 * t2 * a * np.exp(1j * phi)) ** 2
    through = np.abs(t2 - t1 * a * np.exp(1j * phi)) ** 2 / den
    drop = ((1 - t1**2) * (1 - t2**2) * a) / den
    return through, drop


def q_coupling(kappa2: float, ng: float, radius: float, wl: float) -> float:
    """Coupling (external) quality factor ``Q_c = pi n_g (2 pi R) / (lambda kappa^2)``."""
    if kappa2 <= 0:
        return np.inf
    return float(np.pi * ng * (2 * np.pi * radius) / (wl * kappa2))


def q_intrinsic(loss_db_per_cm: float, ng: float, wl_um: float) -> float:
    """Intrinsic Q from a propagation loss ``alpha`` [dB/cm]: ``2 pi n_g / (lambda alpha)``."""
    alpha_per_um = loss_db_per_cm / (10 * np.log10(np.e)) / 1e4  # dB/cm -> 1/um (power)
    if alpha_per_um <= 0:
        return np.inf
    return float(2 * np.pi * ng / (wl_um * alpha_per_um))


def extraction_efficiency(qc: float, qi: float) -> float:
    """Bus extraction efficiency ``eta = 1/(1 + Q_c/Q_i)``."""
    return float(1.0 / (1.0 + qc / qi)) if np.isfinite(qc) else 0.0


def round_trip_amplitude(loss_db_per_cm: float, radius: float) -> float:
    """Round-trip field transmission ``a = exp(-alpha_amp * 2 pi R)`` from dB/cm."""
    alpha_amp_per_um = loss_db_per_cm / (20 * np.log10(np.e)) / 1e4  # amplitude 1/um
    return float(np.exp(-alpha_amp_per_um * 2 * np.pi * radius))


# ======================================================================
# bent EME: concentric two-guide cells -> supermode propagation field
# ======================================================================
def pulley_cells(
    platform: StripPlatform, w_ring: float, w_bus: float, gap: float, wl: float,
    length: float, *, num_cells: int, res: float = 0.04,
    bend_radius: float = np.inf,
) -> list[mw.Cell]:
    """A uniform concentric ring+bus coupling section as bent EME cells.

    Two strip cores (ring at ``x = 0``, bus at ``x = sep``) extruded over
    ``z in [0, length]`` and sliced into ``num_cells`` identical bent cells (mesh
    ``bend_radius`` = the ring radius). The uniformity makes this a genuine
    directional-coupler EME whose supermodes carry the coupling.
    """
    t = platform.thickness
    sep = gap + (w_ring + w_bus) / 2
    x_lo, x_hi = -w_ring - 1.6, sep + w_bus + 1.6
    span = 4 * (x_hi - x_lo)
    structs: list[mw.Structure3D] = [
        mw.Structure(
            material=platform.clad_mat(wl),
            geometry=mw.Box(x_min=-span, x_max=span, y_min=-span, y_max=span,
                            z_min=0.0, z_max=length),
            mesh_order=12,
        )
    ]
    if platform.substrate is not None:
        structs.append(
            mw.Structure(
                material=platform.sub_mat(wl),
                geometry=mw.Box(x_min=-span, x_max=span,
                                y_min=-platform.box_thickness - t / 2, y_max=-t / 2,
                                z_min=0.0, z_max=length),
                mesh_order=10,
            )
        )
    for w, xc in ((w_ring, 0.0), (w_bus, sep)):
        structs.append(
            mw.Structure(
                material=platform.core_mat(wl),
                geometry=mw.Box(x_min=xc - w / 2, x_max=xc + w / 2,
                                y_min=-t / 2, y_max=t / 2,
                                z_min=0.0, z_max=length),
                mesh_order=5,
            )
        )
    mesh = mw.Mesh2D(
        x=np.arange(x_lo, x_hi + res / 2, res),
        y=np.arange(-platform.box_thickness - t / 2,
                    platform.clad_thickness + t / 2 + res / 2, res),
        num_pml=(10, 10), ez_interfaces=True, bend_axis=1,
        bend_radius=float(bend_radius),
    )
    lengths = np.full(num_cells, length / num_cells)
    return mw.create_cells(structs, mesh, lengths, z_min=0.0)


def pulley_propagation(
    platform: StripPlatform, w_ring: float, w_bus: float, gap: float, radius: float,
    length: float, wl: float, *, num_cells: int = 12, num_modes: int = 4,
    res: float = 0.05, num_z: int = 400,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Bent-EME |E| field of a bus-excited pulley, from the two bent supermodes.

    A uniform coupler is exactly the coherent sum of its even/odd supermodes:
    ``E(x, z) = c_e psi_e(x) e^{i beta_e z} + c_o psi_o(x) e^{i beta_o z}``. We
    solve the concentric bent supermodes with meow's FDE (``num_cells`` is kept
    for API symmetry with the cell-based EME), inject the **bus** rail
    (``c_e = +-c_o`` chosen to light ``x = sep`` at ``z = 0``), and evaluate the
    field along the arc. Returns ``(|E|[z, x], x_transverse, ring_fraction_out)``.
    """
    sep = gap + (w_ring + w_bus) / 2
    cs = cross_section(platform, [(w_ring, 0.0), (w_bus, sep)], wl,
                       res=res, bend_radius=radius)
    modes = _te_sorted(solve(cs, num_modes=num_modes))[:2]
    k0 = 2 * np.pi / wl
    beta = np.array([k0 * np.real(m.neff) for m in modes])
    mesh = modes[0].cs.mesh
    xx = np.asarray(mesh.x_)
    y_arr = np.asarray(mesh.y_)
    iy = int(np.argmin(np.abs(y_arr)))  # slice at the core center (y ~ 0)
    psi = np.array([np.asarray(m.Ex)[:, iy] for m in modes])  # (2, Nx)
    # choose the even/odd sign that concentrates z=0 energy on the bus rail
    bus_mask = xx > sep / 2
    combos = {1.0: psi[0] + psi[1], -1.0: psi[0] - psi[1]}
    sign = max(combos, key=lambda s:
               np.abs(combos[s][bus_mask]).sum() / (np.abs(combos[s]).sum() + 1e-30))
    c = np.array([1.0, sign]) / np.sqrt(2)
    z = np.linspace(0.0, length, num_z)
    field = np.abs(
        c[0] * psi[0][None, :] * np.exp(1j * beta[0] * z)[:, None]
        + c[1] * psi[1][None, :] * np.exp(1j * beta[1] * z)[:, None]
    )
    ring_mask = xx < sep / 2
    out = field[-1]
    ring_frac = float((out[ring_mask] ** 2).sum() / ((out**2).sum() + 1e-30))
    return field, xx, ring_frac


# ======================================================================
# the standard plot suite
# ======================================================================
def _agg():
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_cross_section(cs: mw.CrossSection, path: Path, *, title: str) -> None:
    """Refractive-index map of the (bent) coupler cross-section."""
    plt = _agg()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    cs._visualize(ax=ax, cbar=True, show=False)
    ax.set_title(title)
    ax.set_xlabel("radial x [um]")
    ax.set_ylabel("y [um]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_modes(
    modes: list[mw.Mode], labels: list[str], path: Path, *, title: str,
    field: str = "Ex",
) -> None:
    """|E| of a set of modes (isolated ring/bus or even/odd supermodes)."""
    plt = _agg()
    n = len(modes)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.4), squeeze=False)
    for ax, m, lbl in zip(axes[0], modes, labels):
        xx = np.asarray(m.cs.mesh.Xx)
        yy = np.asarray(m.cs.mesh.Yy)
        f = np.abs(np.asarray(getattr(m, field)))
        ax.pcolormesh(xx, yy, f / (f.max() or 1.0), shading="auto", cmap="magma")
        ax.set_title(f"{lbl}\n$n_{{eff}}$={np.real(m.neff):.4f}", fontsize=9)
        ax.set_xlabel("radial x [um]")
        ax.set_aspect("equal")
    axes[0][0].set_ylabel("y [um]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_neff_vs_radius(
    radii: np.ndarray, neffs: np.ndarray, straight_neff: float, path: Path,
    *, title: str,
) -> None:
    """Bent-mode effective index vs bend radius (curvature dispersion)."""
    plt = _agg()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(radii, neffs, "o-", ms=4, label="bent mode")
    ax.axhline(straight_neff, color="0.5", ls="--", label="straight ($R\\to\\infty$)")
    ax.set_xlabel("bend radius R [um]")
    ax.set_ylabel(r"$n_{eff}$")
    ax.grid(visible=True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_kappa0_vs_gap(model: CouplingModel, path: Path, *, title: str) -> None:
    """Calibrated ``kappa0(g)`` samples + fitted ``A exp(-g/gamma)``."""
    plt = _agg()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(model.gaps * 1000, model.kappa0s, "o", label="FDE supermode split")
    gg = np.linspace(model.gaps.min(), model.gaps.max(), 100)
    ax.semilogy(gg * 1000, model.kappa0(gg), "-",
                label=fr"fit $A\,e^{{-g/\gamma}}$, $\gamma$={model.gamma * 1e3:.0f} nm")
    ax.set_xlabel("gap g [nm]")
    ax.set_ylabel(r"$\kappa_0$ [1/um]")
    ax.grid(visible=True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_coupling_curve(
    x: np.ndarray, kappa2: np.ndarray, path: Path, *, title: str, xlabel: str,
    target: float | None = None, x_solution: float | None = None,
    extra: dict[str, np.ndarray] | None = None,
) -> None:
    """Power coupling ``|kappa|^2`` vs a swept variable (length/gap/angle)."""
    plt = _agg()
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.plot(x, kappa2, "-", lw=2, label=r"$|\kappa|^2$")
    for lbl, vals in (extra or {}).items():
        ax.plot(x, vals, "--", lw=1.2, label=lbl)
    if target is not None:
        ax.axhline(target, color="C3", ls=":", label=f"target = {target:.3f}")
    if x_solution is not None:
        ax.axvline(x_solution, color="0.4", ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"power coupling $|\kappa|^2$")
    ax.grid(visible=True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spectrum(
    wls: np.ndarray, series: dict[str, np.ndarray], path: Path, *, title: str,
    ylabel: str = r"power coupling $|\kappa|^2$", center_nm: float | None = None,
    logy: bool = False,
) -> None:
    """Coupling / Q spectrum vs wavelength."""
    plt = _agg()
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    for lbl, vals in series.items():
        (ax.semilogy if logy else ax.plot)(wls * 1000, vals, label=lbl)
    if center_nm is not None:
        ax.axvline(center_nm, color="0.6", ls=":", lw=1)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel(ylabel)
    ax.grid(visible=True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ring_response(
    detune: np.ndarray, curves: dict[str, np.ndarray], path: Path, *, title: str,
) -> None:
    """All-pass / add-drop transmission vs round-trip detuning phase."""
    plt = _agg()
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for lbl, vals in curves.items():
        ax.plot(detune / np.pi, 10 * np.log10(np.maximum(vals, 1e-6)), label=lbl)
    ax.set_xlabel(r"round-trip detuning $\phi/\pi$")
    ax.set_ylabel("transmission [dB]")
    ax.grid(visible=True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_propagation(
    field: np.ndarray, x_trans: np.ndarray, length: float, path: Path, *, title: str,
) -> None:
    """|E| along the propagation (arc) direction: z vs radial x."""
    plt = _agg()
    norm = field / (field.max() or 1.0)
    z = np.linspace(0.0, length, field.shape[0])
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.pcolormesh(z, x_trans, norm.T, shading="auto", cmap="magma")
    ax.set_xlabel("propagation (arc) z [um]")
    ax.set_ylabel("radial x [um]")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="|E| (norm.)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_point_layout(
    w_ring: float, w_bus: float, gap: float, radius: float, path: Path, *, title: str,
) -> None:
    """Top-view schematic of a straight-bus point coupler tangent to a ring."""
    plt = _agg()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    th = np.linspace(-0.9, 0.9, 400)
    xr, yr = radius * np.sin(th), radius * (1 - np.cos(th))
    for off in (-w_ring / 2, w_ring / 2):
        ax.plot(xr, yr - off, "C0", lw=1)
    ax.fill_between(xr, yr - w_ring / 2, yr + w_ring / 2, color="C0", alpha=0.3)
    y_bus = -(gap + (w_ring + w_bus) / 2)
    ax.fill_between([xr.min(), xr.max()], y_bus - w_bus / 2, y_bus + w_bus / 2,
                    color="C3", alpha=0.4)
    ax.annotate("", xy=(0, -gap - w_ring / 2), xytext=(0, y_bus + w_bus / 2),
                arrowprops={"arrowstyle": "<->", "color": "k"})
    ax.text(0.3, y_bus + w_bus, f"gap = {gap * 1e3:.0f} nm", fontsize=9)
    ax.text(0, radius * (1 - np.cos(0.6)), f"ring R = {radius:.1f} um",
            ha="center", fontsize=9, color="C0")
    ax.set_aspect("equal")
    ax.set_xlabel("z [um]")
    ax.set_ylabel("x [um]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pulley_layout(
    w_ring: float, w_bus: float, gap: float, radius: float, wrap_deg: float,
    path: Path, *, title: str,
) -> None:
    """Top-view schematic of a wrap-around pulley coupler."""
    plt = _agg()
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    th = np.linspace(0, 2 * np.pi, 500)
    for r, w, c in ((radius, w_ring, "C0"), (radius + gap + (w_ring + w_bus) / 2,
                                             w_bus, "C3")):
        if c == "C0":
            for off in (-w / 2, w / 2):
                ax.plot((r + off) * np.cos(th), (r + off) * np.sin(th), c, lw=1)
    r_bus = radius + gap + (w_ring + w_bus) / 2
    half = np.deg2rad(wrap_deg) / 2
    tb = np.linspace(np.pi / 2 - half, np.pi / 2 + half, 200)
    for off in (-w_bus / 2, w_bus / 2):
        ax.plot((r_bus + off) * np.cos(tb), (r_bus + off) * np.sin(tb), "C3", lw=1)
    # straight access tails leaving the wrap tangentially
    for sgn in (-1, 1):
        a = np.pi / 2 + sgn * half
        x0, y0 = r_bus * np.cos(a), r_bus * np.sin(a)
        tx, ty = -np.sin(a), np.cos(a)
        ax.plot([x0, x0 + sgn * 8 * tx], [y0, y0 + sgn * 8 * ty], "C3", lw=1)
    ax.text(0, 0, f"R = {radius:.1f} um\nwrap = {wrap_deg:.0f}°",
            ha="center", va="center", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ======================================================================
# designer inversions: hit a target coupling on an arbitrary platform
# ======================================================================
def design_point_gap(
    model: CouplingModel, radius: float, target_kappa2: float,
    *, g_lo: float = 0.05, g_hi: float = 0.6,
) -> float:
    """Smallest gap whose point coupler reaches ``target_kappa2`` (bisection)."""
    tgt = float(target_kappa2)

    def excess(g: float) -> float:
        return point_cross_power(model, g, radius) - tgt

    if excess(g_lo) < 0:  # even the tightest gap under-couples
        return float(g_lo)
    if excess(g_hi) > 0:  # even the widest gap over-couples
        return float(g_hi)
    lo, hi = g_lo, g_hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0:
            lo = mid  # too much coupling -> widen the gap
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def design_pulley_length(
    kappa0: float, delta: float, target_kappa2: float, *, l_max: float = 400.0,
) -> float:
    """Shortest wrap length ``Lc`` reaching ``target_kappa2`` (first crossing).

    The maximum attainable coupling is ``kappa0^2/(kappa0^2+delta^2)``; if the
    target exceeds it the phase mismatch is too large (returns the length of the
    first peak).
    """
    s = np.hypot(kappa0, delta)
    fmax = kappa0**2 / s**2 if s > 0 else 0.0
    if fmax <= target_kappa2:  # mismatch-limited: return the first peak
        return float(np.pi / (2 * s)) if s > 0 else float(l_max)
    ls = np.linspace(1e-3, l_max, 4000)
    k2 = kappa0**2 / s**2 * np.sin(s * ls) ** 2
    hit = np.where(k2 >= target_kappa2)[0]
    return float(ls[hit[0]]) if hit.size else float(l_max)


def design_pulley_phasematch_width(
    platform: StripPlatform, w_ring: float, gap: float, radius: float, wl: float,
    *, w_lo: float = 0.3, w_hi: float = 2.0, res: float = 0.05,
) -> float:
    """Pulley bus width that phase-matches the ring (``delta = 0``) by bisection."""
    def d(w: float) -> float:
        return mismatch(platform, w_ring, w, gap, wl, radius=radius,
                        mode="pulley", res=res)

    d_lo, d_hi = d(w_lo), d(w_hi)
    if d_lo * d_hi > 0:  # no sign change in range -> return the closest endpoint
        return float(w_lo if abs(d_lo) < abs(d_hi) else w_hi)
    lo, hi = w_lo, w_hi
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        if d(mid) * d(lo) > 0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def group_index(
    platform: StripPlatform, width: float, wl: float,
    *, bend_radius: float = np.inf, res: float = 0.03, dwl: float = 0.02,
) -> float:
    """Group index ``n_g = n_eff - lambda dn_eff/dlambda`` by central difference."""
    n0, _ = isolated_neff(platform, width, wl, bend_radius=bend_radius, res=res)
    n_p, _ = isolated_neff(platform, width, wl + dwl, bend_radius=bend_radius, res=res)
    n_m, _ = isolated_neff(platform, width, wl - dwl, bend_radius=bend_radius, res=res)
    dn_dwl = (n_p - n_m) / (2 * dwl)
    return float(n0 - wl * dn_dwl)
