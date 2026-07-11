"""Bogaerts et al., *Silicon microring resonators*, Laser Photonics Rev. 6, 47
(2012) -- and Xu, Fattal & Beausoleil, *Silicon microring resonators with
1.5-um radius*, Opt. Express 16, 4309 (2008).

**Straight-bus-to-ring "point" coupler on 220 nm SOI.** A single-mode Si strip
bus runs tangent to a microring; the edge-to-edge gap opens quadratically away
from the tangent point, ``g(z) = g0 + z^2/(2R)``, so the coupling is a localized
(lumped) event described by a self-/cross-coupling pair ``(t, kappa)`` with
``|t|^2 + |kappa|^2 = 1``. This is the *bent-waveguide* coupling problem: the
ring guides a **bent** whispering-gallery mode (solved with meow's
``bend_radius`` FDE), the bus a straight mode, and their evanescent overlap over
the tangent region sets ``|kappa|^2``.

Reproduced here (all via :mod:`examples.papers._bent_coupler`, i.e. meow's bent
FDE + coupled-mode theory):

- the coupler cross-section and the **bent-ring vs straight-bus modes**;
- the ring's **curvature dispersion** ``n_eff(R)`` (the bent mode's index rising
  toward the straight value as ``R`` grows);
- the evanescent coupling calibration ``kappa0(g) = A e^{-g/gamma}``;
- **``|kappa|^2`` vs gap** for several ring radii -- the paper's design curve:
  ``~1-2%`` at a 200 nm gap rising steeply as the gap closes (Bogaerts Fig. of
  gap-vs-coupling), and ``kappa^2`` growing with radius;
- the **coupling spectrum** ``|kappa|^2(lambda)`` across the C-band;
- the ring **all-pass and add-drop transfer functions** at under-/critical-/
  over-coupling (Bogaerts Eqs.);
- the **critical-coupling** design point and loaded/intrinsic Q, checked against
  Xu 2008 (R = 1.5 um, ``kappa^2 ~ 0.8%``, ``Q_i ~ 2x10^4``, ``Q_L ~ 9x10^3``).

Run with ``python -m examples.papers.bogaerts2012_point_coupler``.
Set ``MEOW_EXAMPLE_RES=low|medium|high`` to trade speed for accuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from examples.papers import _bent_coupler as bc
from examples.papers import _resolution as res

FIGDIR = bc.FIGDIR / "bogaerts2012_point_coupler"
WL = 1.55
WIDTH = 0.45  # single-mode 220 nm SOI strip [um]


def main() -> dict[str, Any]:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    platform = bc.soi_platform(0.22, buried=True, default_width=WIDTH)
    device_res = res.pick(low=0.04, medium=0.03, high=0.02)
    figs: list[str] = []

    # -- 1. coupler cross-section (ring + tangent bus at the design gap) --------
    design_gap = 0.15
    cs = bc.cross_section(
        platform, [(WIDTH, 0.0), (WIDTH, design_gap + WIDTH)], WL,
        res=device_res, bend_radius=10.0,
    )
    bc.plot_cross_section(cs, FIGDIR / "01_cross_section.png",
                          title=f"Point coupler cross-section ({platform.name}, "
                                f"gap {design_gap * 1e3:.0f} nm)")
    figs.append("01_cross_section.png")

    # -- 2. bent-ring vs straight-bus isolated modes ---------------------------
    _, ring_mode = bc.isolated_neff(platform, WIDTH, WL, bend_radius=10.0,
                                    res=device_res)
    _, bus_mode = bc.isolated_neff(platform, WIDTH, WL, bend_radius=np.inf,
                                   res=device_res)
    bc.plot_modes([ring_mode, bus_mode],
                  ["bent ring mode (R=10 um)", "straight bus mode"],
                  FIGDIR / "02_modes.png", title="Isolated coupler modes ($|E_x|$)")
    figs.append("02_modes.png")

    # -- 3. ring curvature dispersion n_eff(R) ---------------------------------
    radii = np.linspace(3.0, 25.0, res.pick(low=6, medium=9, high=14))
    neffs = np.array([bc.isolated_neff(platform, WIDTH, WL, bend_radius=float(r),
                                       res=device_res)[0] for r in radii])
    n_straight, _ = bc.isolated_neff(platform, WIDTH, WL, bend_radius=np.inf,
                                     res=device_res)
    bc.plot_neff_vs_radius(radii, neffs, n_straight, FIGDIR / "03_neff_vs_radius.png",
                           title=f"Ring curvature dispersion ({WIDTH * 1e3:.0f} nm "
                                 f"{platform.name})")
    figs.append("03_neff_vs_radius.png")

    # -- 4. coupling calibration kappa0(g) -------------------------------------
    gaps = np.linspace(0.10, 0.30, res.pick(low=4, medium=5, high=7))
    model = bc.calibrate_coupling(platform, WIDTH, WIDTH, gaps, WL,
                                  radius=10.0, mode="point", res=device_res)
    bc.plot_kappa0_vs_gap(model, FIGDIR / "04_kappa0_vs_gap.png",
                          title="Evanescent coupling calibration (point coupler)")
    figs.append("04_kappa0_vs_gap.png")

    # -- 5. |kappa|^2 vs gap for several ring radii ----------------------------
    gg = np.linspace(0.10, 0.30, 40)
    curves = {f"R = {r:.0f} um":
              np.array([bc.point_cross_power(model, float(g), r) for g in gg])
              for r in (5.0, 10.0, 20.0)}
    plt_curve = {k: v for k, v in curves.items() if k != "R = 10 um"}
    bc.plot_coupling_curve(gg * 1000, curves["R = 10 um"],
                           FIGDIR / "05_kappa2_vs_gap.png",
                           title="Point-coupler power coupling vs gap",
                           xlabel="gap g [nm]", extra=plt_curve)
    figs.append("05_kappa2_vs_gap.png")

    # -- 6. coupling spectrum |kappa|^2(lambda) --------------------------------
    wls = np.linspace(1.50, 1.60, res.pick(low=5, medium=9, high=15))
    k2_spec = []
    for wl in wls:
        m = bc.calibrate_coupling(platform, WIDTH, WIDTH, gaps, float(wl),
                                  radius=10.0, mode="point", res=device_res)
        k2_spec.append(bc.point_cross_power(m, design_gap, 10.0))
    bc.plot_spectrum(wls, {"gap 150 nm, R=10 um": np.array(k2_spec)},
                     FIGDIR / "06_spectrum.png",
                     title="Point-coupler coupling spectrum", center_nm=WL * 1000)
    figs.append("06_spectrum.png")

    # -- 7. ring transfer functions (all-pass + add-drop) ----------------------
    phi = np.linspace(-np.pi, np.pi, 800)
    a = 0.98  # round-trip field transmission
    ap = {f"t={t:.3f} ({name})": bc.all_pass_transmission(t, a, phi)
          for t, name in ((a, "critical"), (0.995, "under"), (0.95, "over"))}
    bc.plot_ring_response(phi, ap, FIGDIR / "07_all_pass.png",
                          title="All-pass ring transmission (a=0.98)")
    figs.append("07_all_pass.png")
    thru, drop = bc.add_drop_through_drop(0.985, 0.985, a, phi)
    bc.plot_ring_response(phi, {"through": thru, "drop": drop},
                          FIGDIR / "08_add_drop.png",
                          title="Add-drop ring response ($t_1=t_2=0.985$)")
    figs.append("08_add_drop.png")

    # -- 8. point-coupler layout schematic -------------------------------------
    bc.plot_point_layout(WIDTH, WIDTH, design_gap, 10.0, FIGDIR / "09_layout.png",
                         title="Straight-bus point coupler")
    figs.append("09_layout.png")

    # -- Xu 2008 critical-coupling check (R = 1.5 um) --------------------------
    ng = bc.group_index(platform, WIDTH, WL, bend_radius=10.0, res=device_res)
    xu_R = 1.5
    xu_loss_db_cm = 30.0  # small-ring propagation loss consistent with Q_i ~ 2e4
    qi_xu = bc.q_intrinsic(xu_loss_db_cm, ng, WL)
    a_xu = bc.round_trip_amplitude(xu_loss_db_cm, xu_R)
    kappa2_crit = 1.0 - a_xu**2  # critical coupling: |kappa|^2 = round-trip loss
    qc_crit = bc.q_coupling(kappa2_crit, ng, xu_R, WL)
    ql_crit = 1.0 / (1.0 / qi_xu + 1.0 / qc_crit)

    summary = {
        "platform": platform.name,
        "kappa0_A_per_um": round(model.A, 4),
        "gamma_nm": round(model.gamma * 1e3, 1),
        "kappa2_gap125nm_R10": round(bc.point_cross_power(model, 0.125, 10.0), 4),
        "kappa2_gap200nm_R10": round(bc.point_cross_power(model, 0.200, 10.0), 4),
        "group_index": round(ng, 4),
        "xu2008_R_um": xu_R,
        "xu2008_kappa2_critical": round(kappa2_crit, 4),
        "xu2008_Qi": round(qi_xu, -2),
        "xu2008_Qc_critical": round(qc_crit, -2),
        "xu2008_QL_critical": round(ql_crit, -2),
    }
    bc_save_summary(FIGDIR / "summary", summary)
    bc_save_table(FIGDIR / "kappa2_vs_gap",
                  {"gap_nm": gg * 1000, **dict(curves)})
    return {"out_dir": str(FIGDIR), "summary": summary, "figures": figs}


def bc_save_summary(path: Path, summary: dict[str, Any]) -> None:
    import meow as mw

    mw.save_summary(path, summary)


def bc_save_table(path: Path, table: dict[str, Any]) -> None:
    import meow as mw

    mw.save_table(path, table)


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
