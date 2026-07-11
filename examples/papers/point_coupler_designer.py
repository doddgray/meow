"""Designer: straight-bus **point** coupler for a target coupling at a target
wavelength, on an *arbitrary* waveguide platform.

Companion to :mod:`examples.papers.bogaerts2012_point_coupler`. Given a platform
(any :class:`~examples.papers._bent_coupler.StripPlatform`: SOI, Si3N4, TFLN, ...),
a ring radius, an operating wavelength and a **target power coupling**
``|kappa|^2``, this inverts the bent-mode coupled-mode model for the **ring-bus
gap** that hits it, then emits the full design report + plots.

Workflow (all via meow's bent FDE + CMT):

1. calibrate the evanescent coupling ``kappa0(g) = A e^{-g/gamma}`` on the
   platform at the design wavelength (a handful of bent supermode solves);
2. invert ``|kappa|^2(gap) = target`` for the required gap (bisection);
3. report the achieved coupling, its wavelength dependence, and the ring
   loaded/coupled/intrinsic Q at a chosen propagation loss (critical-coupling
   check);
4. draw the calibration, the ``|kappa|^2`` vs gap design curve (target + solution
   marked), the coupling spectrum, the ring transfer at the designed coupling,
   and the layout.

Two demo platforms are designed by default (an SOI O-band coupler and a Si3N4
C-band coupler). Adapt ``DESIGNS`` for your own platform/target.

Run with ``python -m examples.papers.point_coupler_designer``.
Set ``MEOW_EXAMPLE_RES=low|medium|high`` to trade speed for accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from examples.papers import _bent_coupler as bc
from examples.papers import _resolution as res

FIGDIR = bc.FIGDIR / "point_coupler_designer"


@dataclass
class PointDesign:
    """A designed straight-bus point coupler."""

    label: str
    platform: bc.StripPlatform
    width: float
    radius: float
    wl: float
    target_kappa2: float
    gap: float
    achieved_kappa2: float
    model: bc.CouplingModel
    loss_db_cm: float

    @property
    def ng(self) -> float:
        """Ring group index at the design wavelength."""
        return bc.group_index(self.platform, self.width, self.wl,
                              bend_radius=self.radius, res=0.03)


def design_point_coupler(
    platform: bc.StripPlatform, radius: float, wl: float, target_kappa2: float,
    *, width: float | None = None, loss_db_cm: float = 3.0, label: str = "design",
    res_um: float = 0.03,
) -> PointDesign:
    """Solve for the gap giving ``target_kappa2`` on ``platform`` at ``wl``."""
    width = width if width is not None else platform.default_width
    gaps = np.linspace(0.08, 0.35, res.pick(low=4, medium=6, high=8))
    model = bc.calibrate_coupling(platform, width, width, gaps, wl,
                                  radius=radius, mode="point", res=res_um)
    gap = bc.design_point_gap(model, radius, target_kappa2)
    achieved = bc.point_cross_power(model, gap, radius)
    return PointDesign(label, platform, width, radius, wl, target_kappa2, gap,
                       achieved, model, loss_db_cm)


def _report(design: PointDesign) -> dict[str, Any]:
    ng = design.ng
    a = bc.round_trip_amplitude(design.loss_db_cm, design.radius)
    qi = bc.q_intrinsic(design.loss_db_cm, ng, design.wl)
    qc = bc.q_coupling(design.achieved_kappa2, ng, design.radius, design.wl)
    ql = 1.0 / (1.0 / qi + 1.0 / qc)
    kappa2_crit = 1.0 - a**2
    return {
        "label": design.label, "platform": design.platform.name,
        "width_nm": round(design.width * 1e3, 1), "radius_um": design.radius,
        "wl_nm": round(design.wl * 1e3, 1),
        "target_kappa2": round(design.target_kappa2, 4),
        "designed_gap_nm": round(design.gap * 1e3, 1),
        "achieved_kappa2": round(design.achieved_kappa2, 4),
        "group_index": round(ng, 4),
        "kappa2_for_critical_coupling": round(kappa2_crit, 4),
        "Qi": round(qi, -2), "Qc": round(qc, -2), "QL": round(ql, -2),
    }


def _plots(design: PointDesign, out: Any, res_um: float) -> list[str]:
    figs: list[str] = []
    tag = design.label

    cores = [(design.width, 0.0), (design.width, design.gap + design.width)]
    cs = bc.cross_section(design.platform, cores, design.wl, res=res_um,
                          bend_radius=design.radius)
    bc.plot_cross_section(cs, out / f"{tag}_01_cross_section.png",
                          title=f"{tag}: coupler cross-section "
                                f"(gap {design.gap * 1e3:.0f} nm)")
    figs.append(f"{tag}_01_cross_section.png")

    bc.plot_kappa0_vs_gap(design.model, out / f"{tag}_02_calibration.png",
                          title=f"{tag}: coupling calibration ({design.platform.name})")
    figs.append(f"{tag}_02_calibration.png")

    gg = np.linspace(0.08, 0.35, 60)
    k2 = np.array([bc.point_cross_power(design.model, float(g), design.radius)
                   for g in gg])
    bc.plot_coupling_curve(gg * 1000, k2, out / f"{tag}_03_kappa2_vs_gap.png",
                           title=f"{tag}: |kappa|^2 vs gap (R={design.radius:.0f} um)",
                           xlabel="gap g [nm]", target=design.target_kappa2,
                           x_solution=design.gap * 1000)
    figs.append(f"{tag}_03_kappa2_vs_gap.png")

    wls = np.linspace(design.wl - 0.05, design.wl + 0.05,
                      res.pick(low=5, medium=9, high=13))
    spec = []
    for wl in wls:
        m = bc.calibrate_coupling(
            design.platform, design.width, design.width,
            np.linspace(0.08, 0.35, res.pick(low=4, medium=6, high=8)), float(wl),
            radius=design.radius, mode="point", res=res_um)
        spec.append(bc.point_cross_power(m, design.gap, design.radius))
    bc.plot_spectrum(wls, {f"gap {design.gap * 1e3:.0f} nm": np.array(spec)},
                     out / f"{tag}_04_spectrum.png",
                     title=f"{tag}: coupling spectrum", center_nm=design.wl * 1000)
    figs.append(f"{tag}_04_spectrum.png")

    phi = np.linspace(-np.pi, np.pi, 800)
    a = bc.round_trip_amplitude(design.loss_db_cm, design.radius)
    t = float(np.sqrt(max(1.0 - design.achieved_kappa2, 0.0)))
    bc.plot_ring_response(
        phi, {f"designed (t={t:.3f}, a={a:.3f})": bc.all_pass_transmission(t, a, phi)},
        out / f"{tag}_05_all_pass.png",
        title=f"{tag}: all-pass ring at the designed coupling")
    figs.append(f"{tag}_05_all_pass.png")

    bc.plot_point_layout(design.width, design.width, design.gap, design.radius,
                         out / f"{tag}_06_layout.png",
                         title=f"{tag}: point coupler ({design.platform.name})")
    figs.append(f"{tag}_06_layout.png")
    return figs


# platform, radius[um], wl[um], target |kappa|^2, width[um], loss[dB/cm]
DESIGNS = [
    ("SOI_Oband", lambda: bc.soi_platform(0.22, buried=True), 12.0, 1.31, 0.03,
     0.40, 3.0),
    ("SiN_Cband", lambda: bc.sin_platform(0.40, buried=True, default_width=1.0),
     40.0, 1.55, 0.05, 1.0, 0.10),
]


def main() -> dict[str, Any]:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    res_um = res.pick(low=0.04, medium=0.03, high=0.02)
    reports, figs = {}, []
    for label, make, radius, wl, target, width, loss in DESIGNS:
        design = design_point_coupler(make(), radius, wl, target, width=width,
                                      loss_db_cm=loss, label=label, res_um=res_um)
        reports[label] = _report(design)
        figs += _plots(design, FIGDIR, res_um)

    import meow as mw

    flat = {f"{lbl}_{k}": v for lbl, r in reports.items() for k, v in r.items()
            if k != "label"}
    mw.save_summary(FIGDIR / "summary", flat)
    return {"out_dir": str(FIGDIR), "reports": reports, "figures": figs}


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
