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
    dichroic_test_structures,
    joint_ad_optimization_figure,
    segmented_neff,
    solid_neff,
)
from examples.papers.magden2018_dichroic import analytical_transmission

FIGDIR = Path(__file__).parent / "figures"
pick = _resolution.pick

# shared broad-band grid axis covering the 900-1200 nm cutoffs
SI3N4_BAND = (0.80, 1.35)

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
    fit_window: float = 0.10,
    max_window: float = 0.20,
    gamma_span: float = 6.0,
    n_fine: int = 121,
    wls: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coupled-mode short/long-pass transmission spectrum of a designed device.

    The detuning ``delta(lambda) = 0.5 k0 (n_WGA - n_WGB)`` is sampled by FDE in
    a window around the cutoff and fit linearly; the plotted wavelength span is
    then chosen so ``gamma = delta / kappa`` reaches +-``gamma_span`` (i.e. the
    full roll-off is shown), capped at ``+-max_window``. The adiabatic short-pass
    power is ``|T_A|^2 = 0.5 (1 + gamma / sqrt(1 + gamma^2))`` (paper Eq. 3),
    floored at the device's predicted Landau-Zener extinction so the suppressed
    port cannot beat the device's adiabaticity.
    """
    wl_c = design.cutoff_wl
    wls_fde = np.linspace(wl_c * (1 - fit_window), wl_c * (1 + fit_window), n_fde)
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
    slope, intercept = np.polyfit(wls_fde, deltas, 1)  # delta ~ slope*wl + intercept
    kappa = max(design.kappa, 1e-9)
    if wls is None:
        half = (
            gamma_span * kappa / abs(slope) if abs(slope) > 1e-12 else max_window * wl_c
        )
        half = min(half, max_window * wl_c)
        wls = np.linspace(wl_c - half, wl_c + half, n_fine)
    gamma = (slope * wls + intercept) / kappa
    floor = 10 ** (-design.extinction_db / 10)
    t_short = np.clip(analytical_transmission(gamma), floor, 1.0 - floor)
    return wls, t_short, 1.0 - t_short


def eme_spectrum(
    design: DichroicDesign,
    out_dir: Path,
    *,
    band: tuple[float, float] = SI3N4_BAND,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense **EME** short-/long-pass spectrum of a design over ``band``.

    Runs the full distributed EME (no field save) via
    :func:`dichroic_designer.analyze_dichroic_design` and reads back the saved
    ``t_short`` / ``t_long`` arrays -- the numerical counterpart of the
    (cheap, coupled-mode) :func:`transmission_spectrum`.
    """
    import json

    from examples.papers import _analysis

    analyze_dichroic_design(design, out_dir, band=band)
    stem = _analysis._file_stem(f"{design.cutoff_wl * 1e3:.0f}nm")
    spec = json.loads((out_dir / f"{stem}_spectrum.json").read_text())
    return (
        np.asarray(spec["wavelength_um"]),
        np.asarray(spec["t_short"]),
        np.asarray(spec["t_long"]),
    )


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
    wls = np.linspace(0.85, 1.27, pick(low=6, medium=11, high=21))
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
    eme: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> None:
    """Each optimized layout next to its simulated transmission spectrum.

    The coupled-mode short-/long-pass spectrum is drawn dashed; when ``eme``
    (cutoff -> ``(wls, t_short, t_long)``) is given, the dense **EME** spectrum is
    overlaid solid for comparison.
    """
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
        ts_db = 10 * np.log10(t_short)
        tl_db = 10 * np.log10(t_long)
        ax_s.plot(wls * 1e3, ts_db, "C0--", label="short (coupled-mode)")
        ax_s.plot(wls * 1e3, tl_db, "C3--", label="long (coupled-mode)")
        ymin = float(min(ts_db.min(), tl_db.min()))
        if eme and d.cutoff_wl in eme:
            e_wls, e_short, e_long = eme[d.cutoff_wl]
            es_db = 10 * np.log10(np.clip(e_short, 1e-6, None))
            el_db = 10 * np.log10(np.clip(e_long, 1e-6, None))
            ax_s.plot(e_wls * 1e3, es_db, "C0", label="short (EME)")
            ax_s.plot(e_wls * 1e3, el_db, "C3", label="long (EME)")
            ymin = min(ymin, float(es_db.min()), float(el_db.min()))
        ax_s.axvline(d.cutoff_wl * 1e3, color="0.5", ls=":", lw=0.8)
        ax_s.set_ylim(max(-42.0, ymin - 3), 2)
        ax_s.set_ylabel("T [dB]", fontsize=8)
        ax_s.grid(visible=True)
        if row == 0:
            ax_s.legend(fontsize=6)
    axes[-1, 0].set_xlabel("z [um]")
    axes[-1, 1].set_xlabel("wavelength [nm]")
    fig.suptitle(f"Dichroic designer {name}: optimized layout + simulated transmission")
    fig.tight_layout()
    fig.savefig(FIGDIR / f"dichroic_designer_si3n4_{name}_grid.png", dpi=150)
    plt.close(fig)


def column_grid_figure(
    name: str,
    plat: Platform,
    wgb: WGB,
    designs: list[DichroicDesign | None],
    res: float,
    *,
    band: tuple[float, float] = SI3N4_BAND,
    n: int = 121,
    eme: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> None:
    """Column of every cutoff's broad-band spectrum on one shared axis.

    Unlike :func:`grid_figure` (layout + per-design narrow spectrum), this stacks
    the dense short-/long-pass spectra in a column over the *same* wavelength
    axis, with each design's targeted cutoff drawn as a dashed line. When ``eme``
    (cutoff -> ``(wls, t_short, t_long)``) is given, the **dense EME** spectra are
    overlaid (solid) on the coupled-mode estimate (dashed).
    """
    from examples.papers._designer_extras import spectrum_grid

    wls = np.linspace(band[0], band[1], n)
    rows = []
    for d in [d for d in designs if d is not None]:
        _wls, t_short, t_long = transmission_spectrum(plat, d, wgb, res, wls=wls)
        row = {
            "label": f"{d.cutoff_wl * 1e3:.0f} nm cutoff",
            "wls": wls,
            "short_pass_cm": t_short,
            "long_pass_cm": t_long,
            "design_wls": [d.cutoff_wl],
        }
        if eme and d.cutoff_wl in eme:
            e_wls, e_short, e_long = eme[d.cutoff_wl]
            row["wls_short_pass"] = e_wls
            row["wls_long_pass"] = e_wls
            row["short_pass"] = e_short
            row["long_pass"] = e_long
        rows.append(row)
    if rows:
        ports = [
            ("short_pass", "C0", "-", "short (EME)"),
            ("long_pass", "C3", "-", "long (EME)"),
            ("short_pass_cm", "C0", "--", "short (coupled-mode)"),
            ("long_pass_cm", "C3", "--", "long (coupled-mode)"),
        ]
        spectrum_grid(
            rows,
            FIGDIR / f"dichroic_designer_si3n4_{name}_spectrum_grid.png",
            db=True,
            xlim_nm=(band[0] * 1e3, band[1] * 1e3),
            ports=ports,
            title=f"Si3N4 {name}: broad-band short-/long-pass (EME vs coupled-mode)",
        )


def thickness_test_structures(name: str, designs: list[DichroicDesign | None]) -> None:
    """Per-cutoff cut-back coupler arrays (constant length, 5 mm chip) + GDS."""
    root = FIGDIR / f"dichroic_designer_si3n4_{name}"
    for d in [d for d in designs if d is not None]:
        dichroic_test_structures(d, root / f"{d.cutoff_wl * 1e3:.0f}nm")


def main() -> dict[str, object]:
    """Design + plot the Si3N4 thickness sweep (200, 100, 40 nm)."""
    FIGDIR.mkdir(exist_ok=True, parents=True)
    gf.gpdk.PDK.activate()
    cutoffs = pick(low=CUTOFFS[::3], medium=CUTOFFS, high=CUTOFFS)
    thicknesses = pick(low=[200, 40], medium=[200, 100, 40], high=[200, 100, 40])

    out: dict[str, object] = {}
    for t_nm in thicknesses:
        t_um, clad_t, wgb, res = THICKNESS_CONFIGS[t_nm]
        if _resolution.is_low():
            res = max(res, 0.06)
        elif _resolution.level() == "high":
            res = min(res, 0.03)
        name = f"{t_nm}nm"
        print(f"=== {name} ===", flush=True)
        plat = platform(t_um, clad_t)
        designs = sweep(plat, wgb, cutoffs, res)
        # dense EME spectrum per design (overlaid on the coupled-mode estimate)
        root = FIGDIR / f"dichroic_designer_si3n4_{name}"
        eme = {
            d.cutoff_wl: eme_spectrum(d, root / f"{d.cutoff_wl * 1e3:.0f}nm")
            for d in designs
            if d is not None
        }
        result_figure(name, plat, wgb, designs, res)
        grid_figure(name, plat, wgb, designs, res, eme=eme)
        column_grid_figure(name, plat, wgb, designs, res, eme=eme)
        thickness_test_structures(name, designs)
        out[name] = _summary(designs)
        if t_nm == 200:
            # joint AD optimization demo (all practical parameters) at the
            # representative (200 nm core) thickness only, to bound run time.
            joint_ad_demo = joint_ad_optimization_figure(
                plat,
                float(cutoffs[len(cutoffs) // 2]),
                FIGDIR / f"dichroic_designer_si3n4_{name}_joint_ad_optimization.png",
                x0_crosssection=(wgb.rail_width, wgb.gap, 1.0, 1.0),
                x0_lengths=(0.50, 150.0, 200.0, 600.0, 150.0),
                res=pick(low=0.09, medium=0.06, high=0.05),
                crosssection_steps=pick(low=10, medium=20, high=24),
                length_steps=pick(low=8, medium=16, high=20),
                analysis_dir=FIGDIR / f"dichroic_designer_si3n4_{name}_joint",
            )
            out[f"{name}_joint_ad_optimization"] = joint_ad_demo
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
