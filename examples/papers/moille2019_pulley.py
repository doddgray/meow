"""Moille, Li, Briles, Yu, Drake, Lu, Rao, Westly, Papp & Srinivasan,
*Broadband resonator-waveguide coupling for efficient extraction of
octave-spanning microcombs*, Opt. Lett. 44(19), 4737 (2019).

**Wrap-around "pulley" coupler on 780 nm silicon nitride.** The bus is itself
*bent* to follow the ring over an angular length ``Lc = R_bus * theta``, so the
coupling accumulates over a long curved region and behaves like a
**phase-mismatched directional coupler**. This is the natural home of the
bent-waveguide EME solver: bus and ring are concentric arcs about one center, so
both are solved *bent* (meow ``bend_radius``) and their even/odd **supermodes**
carry the coupling directly.

Over the uniform wrap the transferred power follows (Moille 2019)

    |kappa|^2(Lc) = kappa0^2/(kappa0^2 + delta^2) sin^2(sqrt(kappa0^2+delta^2) Lc),

whose small-``Lc`` limit is ``(kappa0 Lc)^2 sinc^2(...)``. Because the extra
wrap-length knob decouples "how much coupling" from "how small a gap", the pulley
reaches strong, *broadband* coupling at a comfortable ``700 nm`` gap -- the
paper's route to 20 dB better short-wavelength extraction than a straight bus.

Reproduced here (all via :mod:`examples.papers._bent_coupler`):

- the **concentric bent supermodes** (even/odd) of the ring+pulley cross-section;
- the ring/bus **curvature dispersion** and the **phase-matching bus width** that
  nulls ``delta`` (the paper's "narrow the pulley to phase-match" design rule);
- the **``sinc^2`` coupling law** ``|kappa|^2(Lc)`` at the paper's wrap lengths
  ``Lc in {15, 17, 26, 40} um``;
- the **bent-EME propagation** field: a bus-excited pulley section, power
  sloshing between the bus and ring rails (meow ``propagate_modes`` on bent cells);
- the **coupling / extraction spectrum**: ``Q_c(lambda)`` (only ~1 order of
  magnitude pump->short-wavelength, vs many for a straight bus) and the bus
  extraction efficiency ``eta = 1/(1 + Q_c/Q_i)`` with ``Q_i ~ 3x10^6``.

Run with ``python -m examples.papers.moille2019_pulley``.
Set ``MEOW_EXAMPLE_RES=low|medium|high`` to trade speed for accuracy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from examples.papers import _bent_coupler as bc
from examples.papers import _resolution as res

FIGDIR = bc.FIGDIR / "moille2019_pulley"
WL = 1.55
R = 23.3       # ring outer radius [um]
RW = 1.5       # ring width [um]
W_BUS = 0.55   # pulley bus width [um]
GAP = 0.70     # coupling gap [um]
LCS = (15.0, 17.0, 26.0, 40.0)  # wrap lengths [um]
QI = 3.0e6     # measured intrinsic Q


def main() -> dict[str, Any]:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    platform = bc.sin_platform(0.78, buried=True, default_width=RW)
    device_res = res.pick(low=0.06, medium=0.045, high=0.03)
    figs: list[str] = []
    sep = GAP + (RW + W_BUS) / 2

    # -- 1. concentric bent supermodes (even/odd) ------------------------------
    n_even, n_odd, sup = bc.supermode_split(platform, RW, W_BUS, GAP, WL,
                                            bend_radius=R, res=device_res)
    bc.plot_modes(sup[:2], ["even supermode", "odd supermode"],
                  FIGDIR / "01_supermodes.png",
                  title=f"Concentric bent supermodes (R={R} um, gap {GAP * 1e3:.0f} nm)")
    figs.append("01_supermodes.png")

    # -- 2. cross-section ------------------------------------------------------
    cs = bc.cross_section(platform, [(RW, 0.0), (W_BUS, sep)], WL,
                          res=device_res, bend_radius=R)
    bc.plot_cross_section(cs, FIGDIR / "02_cross_section.png",
                          title=f"Pulley cross-section ({platform.name})")
    figs.append("02_cross_section.png")

    # -- 3. ring curvature dispersion ------------------------------------------
    radii = np.linspace(10.0, 40.0, res.pick(low=5, medium=8, high=12))
    neffs = np.array([bc.isolated_neff(platform, RW, WL, bend_radius=float(r),
                                       res=device_res)[0] for r in radii])
    n_straight, _ = bc.isolated_neff(platform, RW, WL, bend_radius=np.inf,
                                     res=device_res)
    bc.plot_neff_vs_radius(radii, neffs, n_straight, FIGDIR / "03_neff_vs_radius.png",
                           title=f"Ring curvature dispersion ({RW * 1e3:.0f} nm SiN)")
    figs.append("03_neff_vs_radius.png")

    # -- 4. phase-matching bus width (delta -> 0) ------------------------------
    widths = np.linspace(0.4, 1.2, res.pick(low=5, medium=8, high=12))
    deltas = np.array([bc.mismatch(platform, RW, float(w), GAP, WL, radius=R,
                                   mode="pulley", res=device_res) for w in widths])
    w_pm = bc.design_pulley_phasematch_width(platform, RW, GAP, R, WL,
                                             res=device_res)
    _plot_phasematch(widths, deltas, w_pm, W_BUS, FIGDIR / "04_phase_match.png")
    figs.append("04_phase_match.png")

    # -- 5. sinc^2 coupling law |kappa|^2(Lc): paper W vs phase-matched W -------
    k0, delta = bc.supermode_coupling(platform, RW, W_BUS, GAP, WL,
                                      bend_radius=R, res=device_res)
    k0_pm, delta_pm = bc.supermode_coupling(platform, RW, w_pm, GAP, WL,
                                            bend_radius=R, res=device_res)
    lpi = np.pi / (2 * max(k0_pm, 1e-6))  # phase-matched beat length
    lcs = np.linspace(0.0, min(2.2 * lpi, 400.0), 400)
    k2 = np.array([bc.pulley_cross_power(k0, delta, float(L)) for L in lcs])
    k2_pm = np.array([bc.pulley_cross_power(k0_pm, delta_pm, float(L)) for L in lcs])
    bc.plot_coupling_curve(
        lcs, k2_pm, FIGDIR / "05_kappa2_vs_length.png",
        title="Pulley coupling vs wrap length (Moille 2019)",
        xlabel="wrap length $L_c$ [um]",
        extra={f"paper W={W_BUS * 1e3:.0f} nm (mismatched, $\\delta$={delta:.3f})": k2},
    )
    figs.append("05_kappa2_vs_length.png")
    kappa2_at = {f"Lc={L:.0f}um": bc.pulley_cross_power(k0, delta, L) for L in LCS}

    # -- 6. bent-EME propagation field (bus-excited supermode beating) ---------
    lc_field = 2.0 * lpi  # ~one full bus->ring->bus oscillation
    field, x_trans, ring_out = bc.pulley_propagation(
        platform, RW, w_pm, GAP, R, lc_field, WL,
        num_modes=res.num_modes(low=3, medium=4, high=4),
        res=res.pick(low=0.06, medium=0.05, high=0.04),
        num_z=res.pick(low=250, medium=400, high=600),
    )
    bc.plot_propagation(field, x_trans, lc_field, FIGDIR / "06_propagation.png",
                        title=f"Bent-EME pulley field (bus input, phase-matched, "
                              f"$L_\\pi$={lpi:.0f} um)")
    figs.append("06_propagation.png")

    # -- 7. coupling / extraction spectrum -------------------------------------
    wls = np.linspace(1.05, 1.60, res.pick(low=6, medium=10, high=16))
    qc_pump, eta_pump = [], []
    for wl in wls:
        kk, dd = bc.supermode_coupling(platform, RW, w_pm, GAP, float(wl),
                                       bend_radius=R, res=device_res)
        k2w = bc.pulley_cross_power(kk, dd, 26.0)
        ng = bc.group_index(platform, RW, float(wl), bend_radius=R, res=device_res)
        qc = bc.q_coupling(k2w, ng, R, float(wl))
        qc_pump.append(qc)
        eta_pump.append(bc.extraction_efficiency(qc, QI))
    bc.plot_spectrum(wls, {"$Q_c$ (pulley, $L_c$=26 um)": np.array(qc_pump)},
                     FIGDIR / "07_Qc_spectrum.png",
                     title="Pulley coupling-Q spectrum", ylabel="$Q_c$", logy=True)
    bc.plot_spectrum(wls, {"extraction $\\eta$": np.array(eta_pump)},
                     FIGDIR / "08_extraction.png",
                     title=f"Bus extraction efficiency ($Q_i$={QI:.0e})",
                     ylabel="$\\eta = 1/(1+Q_c/Q_i)$")
    figs += ["07_Qc_spectrum.png", "08_extraction.png"]

    # -- 9. layout schematic ---------------------------------------------------
    wrap_deg = np.rad2deg(26.0 / (R + sep))
    bc.plot_pulley_layout(RW, w_pm, GAP, R, wrap_deg, FIGDIR / "09_layout.png",
                          title="Wrap-around pulley coupler")
    figs.append("09_layout.png")

    summary = {
        "platform": platform.name, "ring_R_um": R, "ring_W_um": RW,
        "bus_W_um": W_BUS, "gap_um": GAP,
        "kappa0_per_um": round(k0, 5),
        "delta_per_um": round(delta, 5),
        "phase_match_bus_width_um": round(w_pm, 3),
        "supermode_split": round(n_even - n_odd, 6),
        **{k: round(v, 4) for k, v in kappa2_at.items()},
        "Qc_pump_Lc26_phasematched": round(qc_pump[-1], -3),
        "eta_pump_Lc26": round(eta_pump[-1], 4),
        "Qc_flatness_octave": round(max(qc_pump) / min(qc_pump), 2),
    }
    import meow as mw

    mw.save_summary(FIGDIR / "summary", summary)
    mw.save_table(FIGDIR / "kappa2_vs_length", {"Lc_um": lcs, "kappa2": k2})
    return {"out_dir": str(FIGDIR), "summary": summary, "figures": figs}


def _plot_phasematch(widths, deltas, w_pm, w_paper, path):
    plt = bc._agg()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(widths * 1000, deltas, "o-", ms=4)
    ax.axhline(0, color="0.5", ls="-", lw=0.8)
    ax.axvline(w_pm * 1000, color="C2", ls="--",
               label=f"phase-matched W={w_pm * 1e3:.0f} nm")
    ax.axvline(w_paper * 1000, color="C3", ls=":",
               label=f"paper W={w_paper * 1e3:.0f} nm")
    ax.set_xlabel("pulley bus width W [nm]")
    ax.set_ylabel(r"phase mismatch $\delta$ [1/um]")
    ax.grid(visible=True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("Pulley phase matching: bus width that nulls $\\delta$")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
