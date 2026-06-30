"""Design dichroic beam splitters in 200 nm Si3N4 with the generalized designer.

Uses ``dichroic_designer`` to design and optimize adiabatic dichroic beam
splitters on a fully-etched 200 nm silicon-nitride / silicon-dioxide platform,
for target cutoff wavelengths from 900 to 1200 nm (50 nm spacing). The device
length budget is 2 mm and the minimum fabricable tip width and gap are 50 nm.

A solid Si3N4 strip (WGA) is coupled to a sub-wavelength three-rail Si3N4
waveguide (WGB). Because the multi-rail WGB is less dispersive than the solid
strip, the two effective indices cross once - the cutoff - and the cutoff is
tuned by the WGA width. The WGB here uses 200 nm rails on 50 nm gaps, chosen so
that the index crossing is steep enough that the cutoff is not hypersensitive
to the WGA width (a 200 nm-wide WGB instead has the curves crossing too
shallowly, making the cutoff very fabrication-sensitive).

Run with ``python -m examples.papers.dichroic_designer_si3n4``.
"""

from __future__ import annotations

from pathlib import Path

import gdsfactory as gf
import numpy as np

import meow as mw
from examples.papers import _resolution
from examples.papers._plot import plot_component
from examples.papers.dichroic_designer import (
    WGB,
    DichroicDesign,
    Platform,
    analyze_dichroic_design,
    design_dichroic,
    dichroic_spectrum_grid,
    dichroic_test_structures,
    segmented_neff,
    solid_neff,
)

FIGDIR = Path(__file__).parent / "figures"
pick = _resolution.pick

# shared broad-band grid axis covering the 900-1200 nm Si3N4 cutoffs
SI3N4_BAND = (0.80, 1.35)


def si3n4_platform() -> Platform:
    """Fully-etched 200 nm Si3N4 with SiO2 cladding; 50 nm min features, 2 mm max."""
    return Platform(
        core=mw.silicon_nitride,
        clad=mw.silicon_oxide,
        core_thickness=0.20,
        sidewall_deg=0.0,
        etch_fraction=1.0,
        min_tip=0.05,
        min_gap=0.05,
        max_length=2000.0,
    )


# 200 nm rails on 50 nm gaps: sub-wavelength, with a steep (well-conditioned)
# index crossing across the 900-1200 nm design band.
WGB_DESIGN = WGB(rail_width=0.20, gap=0.05, n_rails=3)
TARGET_CUTOFFS = np.round(np.arange(0.90, 1.201, 0.05), 4)  # 900..1200 nm by 50 nm


def design_all(
    res: float = 0.04,
) -> list[DichroicDesign]:
    """Design a splitter at every target cutoff on the Si3N4 platform."""
    platform = si3n4_platform()
    return [
        design_dichroic(platform, float(wl_c), wgb=WGB_DESIGN, res=res)
        for wl_c in TARGET_CUTOFFS
    ]


def main() -> dict[str, object]:
    """Design Si3N4 dichroics for 900-1200 nm cutoffs and plot the results."""
    import matplotlib.pyplot as plt

    FIGDIR.mkdir(exist_ok=True, parents=True)
    gf.gpdk.PDK.activate()
    platform = si3n4_platform()
    res = pick(low=0.05, medium=0.035, high=0.025)
    cutoffs = pick(low=TARGET_CUTOFFS[::3], medium=TARGET_CUTOFFS, high=TARGET_CUTOFFS)

    designs = [
        design_dichroic(platform, float(wl_c), wgb=WGB_DESIGN, res=res)
        for wl_c in cutoffs
    ]
    summary = {
        f"{d.cutoff_wl * 1e3:.0f}nm": {
            "w_a_nm": round(d.w_a * 1e3, 1),
            "gap_nm": round(d.gap * 1e3, 0),
            "length_um": round(d.total_length, 0),
            "kappa_per_mm": round(d.kappa * 1e3, 2),
            "extinction_db": round(d.extinction_db, 1),
        }
        for d in designs
    }

    fig = plt.figure(figsize=(13, 8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

    # (a) WGA/WGB index crossings = the targeted cutoffs
    ax = fig.add_subplot(grid[0, 0])
    wls = np.linspace(0.85, 1.27, pick(low=6, medium=13, high=21))
    n_b = [segmented_neff(platform, WGB_DESIGN, wl, res=res) for wl in wls]
    ax.plot(wls * 1e3, n_b, "k--", lw=2, label="WGB (3x200 nm rails)")
    for d in designs:
        n_a = [solid_neff(platform, d.w_a, wl, res=res) for wl in wls]
        ax.plot(wls * 1e3, n_a, lw=1, label=f"WGA {d.w_a * 1e3:.0f} nm")
        n_b_at = np.interp(d.cutoff_wl, wls, n_b)
        ax.plot(d.cutoff_wl * 1e3, n_b_at, "o", color="k", ms=4)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("effective index")
    ax.set_title("Si3N4 WGA / WGB index crossings (= cutoffs)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(visible=True)

    # (b) the design + optimization outputs vs targeted cutoff
    ax = fig.add_subplot(grid[0, 1])
    cuts = np.array([d.cutoff_wl * 1e3 for d in designs])
    ax.plot(cuts, [d.w_a * 1e3 for d in designs], "C0o-", label="WGA width [nm]")
    ax.plot(cuts, [d.gap * 1e3 for d in designs], "C1s-", label="opt. gap [nm]")
    ax.set_xlabel("targeted cutoff [nm]")
    ax.set_ylabel("nm")
    ax2 = ax.twinx()
    ax2.plot(cuts, [d.extinction_db for d in designs], "C2^--", label="pred. ER [dB]")
    ax2.set_ylabel("predicted extinction [dB]", color="C2")
    ax2.tick_params(axis="y", labelcolor="C2")
    ax.set_title("design + optimization vs cutoff (L $\\leq$ 2 mm)")
    ax.legend(fontsize=7, loc="center left")
    ax.grid(visible=True)

    # (c) a designed device layout (the mid-band cutoff)
    ax = fig.add_subplot(grid[1, :])
    d = designs[len(designs) // 2]
    plot_component(d.component, ax)
    ax.set_title(
        f"designed Si3N4 device for {d.cutoff_wl * 1e3:.0f} nm cutoff: "
        f"w_a={d.w_a * 1e3:.0f} nm, gap={d.gap * 1e3:.0f} nm, "
        f"L={d.total_length:.0f} um, ER~{d.extinction_db:.0f} dB"
    )
    ax.set_aspect("auto")

    fig.suptitle(
        "Dichroic beam-splitter designer: fully-etched 200 nm Si3N4 / SiO2, "
        "900-1200 nm cutoffs"
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "dichroic_designer_si3n4.png", dpi=150)
    plt.close(fig)

    # per-design EME broad-band spectra (-> column grid) + cut-back test arrays
    analyses_root = FIGDIR / "dichroic_designer_si3n4"
    for d in designs:
        label = f"{d.cutoff_wl * 1e3:.0f}nm"
        analyze_dichroic_design(d, analyses_root / label, band=SI3N4_BAND)
        dichroic_test_structures(d, analyses_root / label)
    grid = dichroic_spectrum_grid(
        designs, analyses_root,
        FIGDIR / "dichroic_designer_si3n4_spectrum_grid.png", band=SI3N4_BAND,
    )
    return {"designs": summary, "spectrum_grid": str(grid) if grid else None}


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
