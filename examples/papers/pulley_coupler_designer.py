"""Designer: wrap-around **pulley** coupler for a target coupling at a target
wavelength, on an *arbitrary* waveguide platform.

Companion to :mod:`examples.papers.moille2019_pulley`. Given a platform, a ring
radius and width, an operating wavelength and a **target power coupling**
``|kappa|^2``, this designs the pulley in two steps that mirror real practice:

1. **phase-match the bus width** -- find the pulley bus width ``W`` that nulls the
   ring/bus phase mismatch ``delta`` (so coupling accumulates coherently and the
   full range of ``|kappa|^2`` is reachable by length alone);
2. **solve the wrap length** ``Lc`` (equivalently the wrap angle
   ``theta = Lc / R_bus``) that reaches the target coupling.

It then reports the design and draws: the phase-matching scan, the ``|kappa|^2``
vs wrap-length curve (target + solution marked), the coupling/extraction Q
spectrum, the bent-EME propagation field at the design, and the layout.

Two demo platforms are designed by default (a Si3N4 telecom pulley and a strongly
over-coupled Si3N4 extractor). Adapt ``DESIGNS`` for your own platform/target.

Run with ``python -m examples.papers.pulley_coupler_designer``.
Set ``MEOW_EXAMPLE_RES=low|medium|high`` to trade speed for accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from examples.papers import _bent_coupler as bc
from examples.papers import _resolution as res

FIGDIR = bc.FIGDIR / "pulley_coupler_designer"


@dataclass
class PulleyDesign:
    """A designed wrap-around pulley coupler."""

    label: str
    platform: bc.StripPlatform
    ring_width: float
    bus_width: float
    radius: float
    gap: float
    wl: float
    target_kappa2: float
    length: float
    kappa0: float
    delta: float
    achieved_kappa2: float
    loss_db_cm: float

    @property
    def sep(self) -> float:
        return self.gap + (self.ring_width + self.bus_width) / 2

    @property
    def wrap_deg(self) -> float:
        return float(np.rad2deg(self.length / (self.radius + self.sep)))


def design_pulley_coupler(
    platform: bc.StripPlatform, radius: float, wl: float, target_kappa2: float,
    *, ring_width: float | None = None, gap: float = 0.5, loss_db_cm: float = 0.1,
    label: str = "design", res_um: float = 0.04,
) -> PulleyDesign:
    """Design a pulley (phase-match the bus, then solve the wrap length)."""
    ring_width = ring_width if ring_width is not None else platform.default_width
    w_pm = bc.design_pulley_phasematch_width(platform, ring_width, gap, radius, wl,
                                             res=res_um)
    kappa0, delta = bc.supermode_coupling(platform, ring_width, w_pm, gap, wl,
                                          bend_radius=radius, res=res_um)
    length = bc.design_pulley_length(kappa0, delta, target_kappa2)
    achieved = bc.pulley_cross_power(kappa0, delta, length)
    return PulleyDesign(label, platform, ring_width, w_pm, radius, gap, wl,
                        target_kappa2, length, kappa0, delta, achieved, loss_db_cm)


def _report(design: PulleyDesign) -> dict[str, Any]:
    ng = bc.group_index(design.platform, design.ring_width, design.wl,
                        bend_radius=design.radius, res=0.04)
    qi = bc.q_intrinsic(design.loss_db_cm, ng, design.wl)
    qc = bc.q_coupling(design.achieved_kappa2, ng, design.radius, design.wl)
    return {
        "label": design.label, "platform": design.platform.name,
        "ring_width_nm": round(design.ring_width * 1e3, 1),
        "phasematch_bus_width_nm": round(design.bus_width * 1e3, 1),
        "radius_um": design.radius, "gap_nm": round(design.gap * 1e3, 1),
        "wl_nm": round(design.wl * 1e3, 1),
        "target_kappa2": round(design.target_kappa2, 4),
        "kappa0_per_um": round(design.kappa0, 5),
        "delta_per_um": round(design.delta, 5),
        "wrap_length_um": round(design.length, 2),
        "wrap_angle_deg": round(design.wrap_deg, 1),
        "achieved_kappa2": round(design.achieved_kappa2, 4),
        "Qi": round(qi, -2), "Qc": round(qc, -2),
        "extraction_eta": round(bc.extraction_efficiency(qc, qi), 4),
    }


def _plots(design: PulleyDesign, out: Any, res_um: float) -> list[str]:
    figs: list[str] = []
    tag = design.label
    pf, wl = design.platform, design.wl

    n_even, n_odd, sup = bc.supermode_split(pf, design.ring_width, design.bus_width,
                                            design.gap, wl, bend_radius=design.radius,
                                            res=res_um)
    bc.plot_modes(sup[:2], ["ring-branch supermode", "bus-branch supermode"],
                  out / f"{tag}_01_supermodes.png",
                  title=f"{tag}: phase-matched supermodes")
    figs.append(f"{tag}_01_supermodes.png")

    widths = np.linspace(max(0.3, design.ring_width - 1.0), design.ring_width + 0.3,
                         res.pick(low=5, medium=8, high=11))
    deltas = np.array([bc.mismatch(pf, design.ring_width, float(w), design.gap, wl,
                                   radius=design.radius, mode="pulley", res=res_um)
                       for w in widths])
    _plot_pm(widths, deltas, design.bus_width, out / f"{tag}_02_phase_match.png", tag)
    figs.append(f"{tag}_02_phase_match.png")

    lpi = np.pi / (2 * max(design.kappa0, 1e-6))
    lcs = np.linspace(0.0, min(2.2 * lpi, 500.0), 300)
    k2 = np.array([bc.pulley_cross_power(design.kappa0, design.delta, float(L))
                   for L in lcs])
    bc.plot_coupling_curve(lcs, k2, out / f"{tag}_03_kappa2_vs_length.png",
                           title=f"{tag}: |kappa|^2 vs wrap length",
                           xlabel="wrap length $L_c$ [um]",
                           target=design.target_kappa2, x_solution=design.length)
    figs.append(f"{tag}_03_kappa2_vs_length.png")

    wls = np.linspace(wl - 0.15, wl + 0.15, res.pick(low=5, medium=9, high=13))
    qc_s = []
    for w in wls:
        kk, dd = bc.supermode_coupling(pf, design.ring_width, design.bus_width,
                                       design.gap, float(w), bend_radius=design.radius,
                                       res=res_um)
        k2w = bc.pulley_cross_power(kk, dd, design.length)
        ng = bc.group_index(pf, design.ring_width, float(w), bend_radius=design.radius,
                            res=res_um)
        qc_s.append(bc.q_coupling(k2w, ng, design.radius, float(w)))
    bc.plot_spectrum(wls, {f"$Q_c$ ($L_c$={design.length:.0f} um)": np.array(qc_s)},
                     out / f"{tag}_04_Qc_spectrum.png",
                     title=f"{tag}: coupling-Q spectrum", ylabel="$Q_c$", logy=True,
                     center_nm=wl * 1000)
    figs.append(f"{tag}_04_Qc_spectrum.png")

    field, x_trans, _ = bc.pulley_propagation(
        pf, design.ring_width, design.bus_width, design.gap, design.radius,
        2.0 * lpi, wl, num_modes=res.num_modes(low=3, medium=4, high=4),
        res=res.pick(low=0.06, medium=0.05, high=0.04),
        num_z=res.pick(low=250, medium=400, high=600))
    bc.plot_propagation(field, x_trans, 2.0 * lpi, out / f"{tag}_05_propagation.png",
                        title=f"{tag}: bent-EME beat ($L_\\pi$={lpi:.0f} um)")
    figs.append(f"{tag}_05_propagation.png")

    bc.plot_pulley_layout(design.ring_width, design.bus_width, design.gap,
                          design.radius, design.wrap_deg,
                          out / f"{tag}_06_layout.png",
                          title=f"{tag}: pulley coupler ({pf.name})")
    figs.append(f"{tag}_06_layout.png")
    return figs


def _plot_pm(widths, deltas, w_pm, path, tag):
    plt = bc._agg()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(widths * 1000, deltas, "o-", ms=4)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(w_pm * 1000, color="C2", ls="--",
               label=f"phase-matched W={w_pm * 1e3:.0f} nm")
    ax.set_xlabel("pulley bus width W [nm]")
    ax.set_ylabel(r"phase mismatch $\delta$ [1/um]")
    ax.grid(visible=True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(f"{tag}: pulley phase matching")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# label, platform factory, radius[um], ring_width[um], gap[um], wl[um],
# target |kappa|^2, loss[dB/cm]
DESIGNS = [
    ("SiN_telecom", lambda: bc.sin_platform(0.78, buried=True, default_width=1.5),
     23.3, 1.5, 0.30, 1.55, 0.25, 0.1),
    ("SiN_extractor", lambda: bc.sin_platform(0.60, buried=True, default_width=1.2),
     30.0, 1.2, 0.35, 1.55, 0.40, 0.1),
]


def main() -> dict[str, Any]:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    res_um = res.pick(low=0.05, medium=0.04, high=0.03)
    reports, figs = {}, []
    for label, make, radius, rw, gap, wl, target, loss in DESIGNS:
        design = design_pulley_coupler(make(), radius, wl, target, ring_width=rw,
                                       gap=gap, loss_db_cm=loss, label=label,
                                       res_um=res_um)
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
