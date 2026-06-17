"""Dichroic designer thickness sweep: Si3N4 / SiO2 at 200, 100 and 40 nm.

Runs the generalized ``dichroic_designer`` on fully-etched Si3N4 cores of three
thicknesses (200, 100, 40 nm) in SiO2 cladding, designing splitters for cutoffs
from 900 to 1200 nm (50 nm steps, plus 990 nm). Thinner cores are more weakly
confined, so each thickness uses a wider sub-wavelength WGB and the phase-match
WGA widths grow (and the modeled coupling weakens); the 40 nm core is near the
edge of guidance (its WGB index sits just above the SiO2 cladding), needing
micron-scale waveguides.

For each thickness it writes:

- ``dichroic_designer_si3n4_<t>nm.png``: WGA/WGB index crossings (the cutoffs),
  the design + optimization outputs vs cutoff, and one designed layout;
- ``dichroic_designer_si3n4_<t>nm_grid.png``: each optimized device layout next
  to its simulated (coupled-mode) transmission spectrum, one row per cutoff.

Run with ``python -m examples.papers.dichroic_designer_si3n4_thickness``.
"""

from __future__ import annotations

import os
from pathlib import Path

import gdsfactory as gf
import numpy as np

import meow as mw
from examples.papers._plot import plot_component
from examples.papers.dichroic_designer import (
    WGB,
    DichroicDesign,
    Platform,
    design_dichroic,
    segmented_neff,
    solid_neff,
)
from examples.papers.magden2018_dichroic import analytical_transmission

FIGDIR = Path(__file__).parent / "figures"
FAST = bool(int(os.environ.get("MEOW_EXAMPLE_FAST", "0")))

# 900..1200 nm by 50 nm, plus 990 nm.
CUTOFFS = np.array([0.90, 0.95, 0.99, 1.00, 1.05, 1.10, 1.15, 1.20])

# per core thickness: (thickness_um, clad_thickness_um, WGB, mesh res_um)
THICKNESS_CONFIGS: dict[int, tuple[float, float, WGB, float]] = {
    200: (0.20, 1.0, WGB(0.20, 0.05, 3), 0.04),
    100: (0.10, 1.2, WGB(0.30, 0.05, 3), 0.05),
    40: (0.04, 2.0, WGB(0.60, 0.05, 3), 0.07),
}


def platform(thickness_um: float, clad_thickness: float) -> Platform:
    """Fully-etched Si3N4 / SiO2 platform with 50 nm min features, 2 mm max."""
    return Platform(
        core=mw.silicon_nitride,
        clad=mw.silicon_oxide,
        core_thickness=thickness_um,
        sidewall_deg=0.0,
        etch_fraction=1.0,
        min_tip=0.05,
        min_gap=0.05,
        max_length=2000.0,
        clad_thickness=clad_thickness,
    )


def sweep(
    plat: Platform, wgb: WGB, cutoffs: np.ndarray, res: float
) -> list[DichroicDesign | None]:
    """Design a splitter at every cutoff; ``None`` where the design is unreachable."""
    designs: list[DichroicDesign | None] = []
    for wl_c in cutoffs:
        try:
            designs.append(design_dichroic(plat, float(wl_c), wgb=wgb, res=res))
        except ValueError as e:
            print(f"  cutoff {wl_c * 1e3:.0f} nm: {e}")
            designs.append(None)
    return designs


def transmission_spectrum(
    plat: Platform,
    design: DichroicDesign,
    wgb: WGB,
    res: float,
    n_fde: int = 5,
    window: float = 0.08,
    n_fine: int = 121,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coupled-mode short/long-pass transmission spectrum of a designed device.

    The detuning ``delta(lambda) = 0.5 k0 (n_WGA - n_WGB)`` is sampled by FDE in
    a window around the cutoff; the adiabatic short-pass power is
    ``|T_A|^2 = 0.5 (1 + gamma / sqrt(1 + gamma^2))`` with ``gamma = delta /
    kappa`` (paper Eq. 3), floored at the device's predicted Landau-Zener
    extinction so the suppressed port cannot beat the device's adiabaticity.
    """
    wl_c = design.cutoff_wl
    wls_fde = np.linspace(wl_c * (1 - window), wl_c * (1 + window), n_fde)
    deltas = np.array(
        [
            0.5
            * (2 * np.pi / wl)
            * (
                solid_neff(plat, design.w_a, float(wl), res=res)
                - segmented_neff(plat, wgb, float(wl), res=res)
            )
            for wl in wls_fde
        ]
    )
    wls = np.linspace(wls_fde[0], wls_fde[-1], n_fine)
    gamma = np.interp(wls, wls_fde, deltas) / max(design.kappa, 1e-9)
    floor = 10 ** (-design.extinction_db / 10)
    t_short = np.clip(analytical_transmission(gamma), floor, 1.0 - floor)
    return wls, t_short, 1.0 - t_short


def _summary(designs: list[DichroicDesign | None]) -> dict[str, object]:
    return {
        f"{d.cutoff_wl * 1e3:.0f}nm": {
            "w_a_nm": round(d.w_a * 1e3, 1),
            "gap_nm": round(d.gap * 1e3, 0),
            "length_um": round(d.total_length, 0),
            "kappa_per_mm": round(d.kappa * 1e3, 2),
            "extinction_db": round(d.extinction_db, 1),
        }
        for d in designs
        if d is not None
    }


def result_figure(
    name: str,
    plat: Platform,
    wgb: WGB,
    designs: list[DichroicDesign | None],
    res: float,
) -> None:
    """The 3-panel result figure (index crossings, design sweep, one layout)."""
    import matplotlib.pyplot as plt

    ok = [d for d in designs if d is not None]
    fig = plt.figure(figsize=(13, 8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

    ax = fig.add_subplot(grid[0, 0])
    wls = np.linspace(0.85, 1.27, 6 if FAST else 11)
    n_b = [segmented_neff(plat, wgb, wl, res=res) for wl in wls]
    ax.plot(wls * 1e3, n_b, "k--", lw=2, label="WGB (sub-wavelength)")
    for d in ok:
        n_a = [solid_neff(plat, d.w_a, wl, res=res) for wl in wls]
        ax.plot(wls * 1e3, n_a, lw=1, label=f"WGA {d.w_a * 1e3:.0f} nm")
        ax.plot(d.cutoff_wl * 1e3, np.interp(d.cutoff_wl, wls, n_b), "ok", ms=4)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("effective index")
    ax.set_title(f"{name}: WGA / WGB index crossings (= cutoffs)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(visible=True)

    ax = fig.add_subplot(grid[0, 1])
    cuts = np.array([d.cutoff_wl * 1e3 for d in ok])
    ax.plot(cuts, [d.w_a * 1e3 for d in ok], "C0o-", label="WGA width [nm]")
    ax.plot(cuts, [d.gap * 1e3 for d in ok], "C1s-", label="opt. gap [nm]")
    ax.set_xlabel("targeted cutoff [nm]")
    ax.set_ylabel("nm")
    ax2 = ax.twinx()
    ax2.plot(cuts, [d.extinction_db for d in ok], "C2^--")
    ax2.set_ylabel("predicted extinction [dB]", color="C2")
    ax2.tick_params(axis="y", labelcolor="C2")
    ax.set_title("design + optimization vs cutoff (L $\\leq$ 2 mm)")
    ax.legend(fontsize=7, loc="center left")
    ax.grid(visible=True)

    ax = fig.add_subplot(grid[1, :])
    if ok:
        d = ok[len(ok) // 2]
        plot_component(d.component, ax)
        ax.set_title(
            f"designed device for {d.cutoff_wl * 1e3:.0f} nm cutoff: "
            f"w_a={d.w_a * 1e3:.0f} nm, gap={d.gap * 1e3:.0f} nm, "
            f"L={d.total_length:.0f} um, ER~{d.extinction_db:.0f} dB"
        )
        ax.set_aspect("auto")

    fig.suptitle(f"Dichroic beam-splitter designer: {name}, 900-1200 nm cutoffs")
    fig.tight_layout()
    fig.savefig(FIGDIR / f"dichroic_designer_si3n4_{name}.png", dpi=150)
    plt.close(fig)


def grid_figure(
    name: str,
    plat: Platform,
    wgb: WGB,
    designs: list[DichroicDesign | None],
    res: float,
) -> None:
    """Each optimized layout next to its simulated transmission spectrum."""
    import matplotlib.pyplot as plt

    ok = [d for d in designs if d is not None]
    nrows = len(ok)
    fig, axes = plt.subplots(
        nrows, 2, figsize=(13, 1.7 * nrows), squeeze=False, width_ratios=[2.2, 1]
    )
    for row, d in enumerate(ok):
        ax_l, ax_s = axes[row]
        plot_component(d.component, ax_l)
        ax_l.set_aspect("auto")
        ax_l.set_title(
            f"{d.cutoff_wl * 1e3:.0f} nm cutoff: w_a={d.w_a * 1e3:.0f} nm, "
            f"gap={d.gap * 1e3:.0f} nm, L={d.total_length:.0f} um",
            fontsize=8,
        )
        ax_l.set_ylabel("x [um]", fontsize=8)
        wls, t_short, t_long = transmission_spectrum(plat, d, wgb, res)
        ax_s.plot(wls * 1e3, 10 * np.log10(t_short), "C0", label="short (WGA)")
        ax_s.plot(wls * 1e3, 10 * np.log10(t_long), "C3", label="long (WGB)")
        ax_s.axvline(d.cutoff_wl * 1e3, color="0.5", ls=":", lw=0.8)
        ax_s.set_ylim(-max(35, d.extinction_db + 5), 2)
        ax_s.set_ylabel("T [dB]", fontsize=8)
        ax_s.grid(visible=True)
        if row == 0:
            ax_s.legend(fontsize=7)
    axes[-1, 0].set_xlabel("z [um]")
    axes[-1, 1].set_xlabel("wavelength [nm]")
    fig.suptitle(
        f"Dichroic designer {name}: optimized layout + simulated transmission"
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / f"dichroic_designer_si3n4_{name}_grid.png", dpi=150)
    plt.close(fig)


def main() -> dict[str, object]:
    """Design + plot the Si3N4 thickness sweep (200, 100, 40 nm)."""
    FIGDIR.mkdir(exist_ok=True, parents=True)
    gf.gpdk.PDK.activate()
    cutoffs = CUTOFFS[::3] if FAST else CUTOFFS
    thicknesses = [200, 40] if FAST else [200, 100, 40]

    out: dict[str, object] = {}
    for t_nm in thicknesses:
        t_um, clad_t, wgb, res = THICKNESS_CONFIGS[t_nm]
        if FAST:
            res = max(res, 0.06)
        name = f"{t_nm}nm"
        print(f"=== {name} ===", flush=True)
        plat = platform(t_um, clad_t)
        designs = sweep(plat, wgb, cutoffs, res)
        result_figure(name, plat, wgb, designs, res)
        grid_figure(name, plat, wgb, designs, res)
        out[name] = _summary(designs)
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
