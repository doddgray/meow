"""Reproduce the simulated figures of Kwolek et al., arXiv:2603.27034.

Generates, for **each LiNbO3 material model** (``anisotropic`` -- the real
uniaxial crystal -- and ``isotropic`` -- a fake LN with the extraordinary index
on every axis), into ``examples/papers/figures/`` (filenames suffixed by the
model):

- ``kwolek2026_fig1_<model>.png``: the gdsfactory combiner layout built from a
  parametric-width path (paper Fig. 1a), the gap and top-width-difference
  profiles (Fig. 1b), the FAQUAD mixing angle chi(z) (Fig. 1c), the
  symmetric/antisymmetric supermodes (Fig. 1d) and the EME field propagation at
  FH and SH (Fig. 1e/f).
- ``kwolek2026_fig2_<model>.png``: every subfigure of paper Fig. 2 -- (a) FH
  extinction ratio vs detuning for the FAQUAD-bends / FAQUAD-taper / linear-taper
  variants, (b) SH extinction ratio, (c) total and radiated loss at FH and SH,
  (d) the FH fabrication-tolerance map (etch depth x top width).
- ``kwolek2026_fig5_<model>.png``: the supplemental coupling-model verification
  -- (a) mixing-angle error and (b) coupling-magnitude error over the
  (gap, dTW) plane vs the FDE sweep, and (c) the realized adiabaticity eta(z)
  for the designed / constant-width / FAQUAD bends.
- ``kwolek2026_broadband_<model>.png``: the bar and cross transmission across
  more than an octave (~0.8*SH .. 1.2*FH), the wide-band dichroic response.

The two material models are run so the TE/TM mode crossings at the SH band (a
purely anisotropic effect) can be isolated.

Resolution is selected with ``MEOW_EXAMPLE_RES`` in ``{low, medium, high}``
(default ``medium``): ``low`` is a coarse-but-quick version (used by the test
suite), ``medium`` is the converged full-quality reproduction and ``high``
pushes the mesh / modes / cell count further still.
"""

from __future__ import annotations

from pathlib import Path

import gdsfactory as gf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import meow as mw
import meow.eme.propagation as prop
from examples.papers import _resolution
from examples.papers._backends import parallel_enabled, resolve_backend
from examples.papers._plot import plot_component
from examples.papers.kwolek2026_faquad import (
    G_C,
    G_M,
    H_FILM,
    LN_MODELS,
    W_TOP,
    FaquadDesign,
    bar_cross_transmission,
    calib_mesh,
    calibrate,
    device_cells,
    faquad_combiner,
    input_launch_index,
    rib_structures,
    solve_te_neffs,
)

gf.gpdk.PDK.activate()

pick = _resolution.pick
FIGDIR = Path(__file__).parent / "figures"

WL_FH = 1.55
WL_SH = 0.775
MODELS = LN_MODELS

# Accuracy knobs (low / medium / high via MEOW_EXAMPLE_RES). FH converges well
# at the medium values (cross ~0.9). The second harmonic is intrinsically
# under-converged (the rib is strongly multimode at 775 nm), so cells * modes is
# deliberately capped and the SH numbers are a lower bound (see the README).
RES = pick(low=0.045, medium=0.03, high=0.022)
NUM_CELLS = _resolution.num_cells(low=110, medium=170, high=200)
NUM_MODES = _resolution.num_modes(low=8, medium=12, high=14)
# The Fig 1e field reconstruction holds every cell's modes at once -> lighter.
FIELD_CELLS = min(NUM_CELLS, pick(low=100, medium=130, high=150))
FIELD_MODES = min(NUM_MODES, 8)

# Constant-gap (Region I) length for the reproduction. The paper's final design
# uses l_m = 264 um (see kwolek2026_faquad.L_M) at its measured coupling; meow's
# calibrated (and converged) coupling for this stack is ~2x weaker, so reaching
# the paper's adiabaticity (eta ~ 0.2, clean FH transfer) needs a proportionally
# longer interaction length -- l_m ~ 520 um here. The FAQUAD methodology (design
# at constant adiabaticity, set by l_m) is identical; only the absolute length
# scales with the modeled coupling strength.
FIG_L_M = 520.0

BACKEND = resolve_backend()
PARALLEL = parallel_enabled()


def _show(fig: plt.Figure) -> None:
    """Display a freshly generated figure if an interactive backend is active."""
    if not mpl.get_backend().lower().startswith("agg"):
        fig.show()
        plt.pause(0.1)


def _design(
    model: str, variant: str = "faquad_bends"
) -> tuple[FaquadDesign, gf.Component]:
    """Calibrated FAQUAD design + parametric-width layout for a material model."""
    kappa_0, g_0, dbeta_dtw = calibrate(
        WL_FH, res=RES, compute_modes=BACKEND, model=model
    )
    design = FaquadDesign(kappa_0, g_0, dbeta_dtw, l_m=FIG_L_M, variant=variant)
    component = faquad_combiner(
        kappa_0, g_0, dbeta_dtw, l_m=FIG_L_M, variant=variant
    )
    return design, component


def _transmission(
    component: gf.Component, wl: float, model: str, design: FaquadDesign
) -> tuple[float, float]:
    """(bar, cross) transmission of the launched TE mode at one wavelength."""
    cells = device_cells(
        component, wl, num_cells=NUM_CELLS, res=RES, design=design, model=model
    )
    return bar_cross_transmission(
        cells, wl, num_modes=NUM_MODES, parallel=PARALLEL, compute_modes=BACKEND
    )


def _spectrum(
    component: gf.Component, wls: np.ndarray, model: str, design: FaquadDesign
) -> tuple[np.ndarray, np.ndarray]:
    """Bar/cross transmission spectrum over ``wls`` for a fixed layout."""
    bars, crosses = [], []
    for wl in wls:
        t_bar, t_cross = _transmission(component, float(wl), model, design)
        bars.append(t_bar)
        crosses.append(t_cross)
    return np.asarray(bars), np.asarray(crosses)


# --------------------------------------------------------------------------
# Figure 1: layout, profiles, supermodes, propagating fields
# --------------------------------------------------------------------------
def _supermode_panel(
    ax_pair: list[plt.Axes], gap: float, title: str, model: str
) -> None:
    """Plot the symmetric/antisymmetric supermodes at a given gap."""
    x0 = (W_TOP + gap) / 2
    structures = rib_structures(WL_FH, [W_TOP, W_TOP], [-x0, x0], model)
    cell = mw.Cell(structures=structures, mesh=calib_mesh(RES), z_min=0, z_max=1)
    cs = mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=WL_FH, T=25.0))
    modes = BACKEND(cs, num_modes=2)
    for ax, mode, name in zip(ax_pair, modes[:2], ["sym", "antisym"], strict=False):
        X, Y = mode.cs.mesh.Xx, mode.cs.mesh.Yx
        ax.pcolormesh(X, Y, np.real(mode.Ex), cmap="RdBu")
        ax.set_title(f"{name} (g = {gap * 1e3:.0f} nm)\n{title}", fontsize=8)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-0.3, 0.6)
        ax.set_aspect("equal")


def figure1(model: str) -> dict[str, float]:
    """Layout / profiles / supermodes / FH-SH propagation (paper Fig. 1)."""
    design, component = _design(model)

    fig = plt.figure(figsize=(13, 11))
    grid = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1.2])

    ax = fig.add_subplot(grid[0, :])
    plot_component(component, ax)
    ax.set_aspect("auto")
    ax.set_title("Fig. 1a: FAQUAD combiner layout (cubic + Euler bends)")

    z = np.linspace(-design.half_length, design.half_length, 600)
    ax = fig.add_subplot(grid[1, 0:2])
    ax.plot(z, design.gap(z) * 1e3, "C0", label="gap g(z)")
    ax.axhline(G_M * 1e3, color="C0", ls=":", lw=0.8)
    ax.axhline(G_C * 1e3, color="C0", ls="--", lw=0.8)
    for zb in (design.l_m / 2, design.z_ii):
        ax.axvspan(zb, design.z_ii, color="0.85", alpha=0.0)
    ax.axvline(design.l_m / 2, color="0.7", ls=":", lw=0.8)
    ax.axvline(-design.l_m / 2, color="0.7", ls=":", lw=0.8)
    ax.set_xlabel("z [um]")
    ax.set_ylabel("gap [nm]", color="C0")
    ax2 = ax.twinx()
    ax2.plot(z, design.dtw(z) * 1e3, "C3", label="dTW(z)")
    ax2.set_ylabel("dTW [nm]", color="C3")
    ax.set_title("Fig. 1b: gap and top-width-difference profiles")
    ax.grid(visible=True)

    ax = fig.add_subplot(grid[1, 2:4])
    ax.plot(z, design.chi(z) / np.pi, "C2")
    ax.set_xlabel("z [um]")
    ax.set_ylabel("$\\chi(z) / \\pi$")
    ax.set_title("Fig. 1c: FAQUAD coupling angle")
    ax.grid(visible=True)

    axes_d = [fig.add_subplot(grid[2, i]) for i in range(4)]
    _supermode_panel(axes_d[0:2], G_M, "device midpoint", model)
    _supermode_panel(axes_d[2:4], G_C, "end of FAQUAD evolution", model)

    results: dict[str, float] = {}
    for i, (wl, label) in enumerate([(WL_FH, "FH 1550 nm"), (WL_SH, "SH 775 nm")]):
        env = mw.Environment(wl=wl, T=25.0)
        t_bar, t_cross = _transmission(component, wl, model, design)
        results[f"bar_{label.split()[0]}"] = t_bar
        results[f"cross_{label.split()[0]}"] = t_cross

        cells_f = device_cells(
            component, wl, num_cells=FIELD_CELLS, res=RES, design=design, model=model
        )
        css = [mw.CrossSection.from_cell(cell=c, env=env) for c in cells_f]
        modes = [BACKEND(cs, num_modes=FIELD_MODES) for cs in css]
        in_idx = input_launch_index(modes[0])
        ex_l = np.zeros(len(modes[0]))
        ex_l[in_idx] = 1.0
        ex_r = np.zeros(len(modes[-1]))
        z_pts = np.linspace(
            0.0, sum(c.length for c in cells_f), pick(low=400, medium=800, high=1200)
        )
        Ex, x_pts = prop.propagate_modes(
            modes, cells_f, excitation_l=ex_l, excitation_r=ex_r, y=0.25, z=z_pts
        )
        ax = fig.add_subplot(grid[3, 2 * i : 2 * i + 2])
        ax.imshow(
            np.abs(np.asarray(Ex).T) ** 2,
            aspect="auto",
            origin="lower",
            extent=(
                float(z_pts[0]), float(z_pts[-1]),
                float(x_pts[0]), float(x_pts[-1]),
            ),
            cmap="inferno",
        )
        ax.set_xlabel("z [um]")
        ax.set_ylabel("x [um]")
        ax.set_ylim(-3, 3)
        ax.set_title(f"Fig. 1e: |Ex|$^2$ propagation at {label}")

    fig.suptitle(
        f"Kwolek 2026, Fig. 1 [{model} LN]: FAQUAD TFLN combiner "
        f"(FH cross = {results['cross_FH']:.3f}, SH bar = {results['bar_SH']:.3f}; "
        f"{_resolution.level()}-res)"
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / f"kwolek2026_fig1_{model}.png", dpi=150)
    _show(fig)
    plt.close(fig)
    return results


# --------------------------------------------------------------------------
# Figure 2: extinction ratio, loss, fabrication tolerance
# --------------------------------------------------------------------------
def _fab_tolerance_loss(
    model: str, d_etch_nm: float, d_width_nm: float
) -> float:
    """FH total loss [dB] for an etch-depth / top-width fabrication offset.

    Rebuilds the calibration + layout with the perturbed slab thickness and
    nominal width (monkeypatching the module constants the builders read), then
    returns ``-10 log10(cross)`` at the FH. Restores the constants afterwards.
    """
    import examples.papers.kwolek2026_faquad as kwmod

    h_slab0, w_top0 = kwmod.H_SLAB, kwmod.W_TOP
    try:
        # a deeper etch removes slab (slab = H_FILM - etch); +d_etch deepens etch
        kwmod.H_SLAB = float(np.clip(h_slab0 - d_etch_nm * 1e-3, 0.02, H_FILM - 0.02))
        kwmod.W_TOP = float(w_top0 + d_width_nm * 1e-3)
        kappa_0, g_0, dbeta_dtw = calibrate(
            WL_FH, res=RES, compute_modes=BACKEND, model=model
        )
        design = FaquadDesign(kappa_0, g_0, dbeta_dtw, l_m=FIG_L_M)
        component = faquad_combiner(
            kappa_0, g_0, dbeta_dtw, l_m=FIG_L_M, w_top=kwmod.W_TOP
        )
        _t_bar, t_cross = _transmission(component, WL_FH, model, design)
    finally:
        kwmod.H_SLAB, kwmod.W_TOP = h_slab0, w_top0
        calibrate.cache_clear()
    return float(-10 * np.log10(max(t_cross, 1e-9)))


def figure2(model: str) -> dict[str, float]:
    """Every subfigure of paper Fig. 2 (a-d)."""
    n_fh = pick(low=5, medium=11, high=21)
    n_sh = pick(low=4, medium=7, high=13)
    wls_fh = np.linspace(1.50, 1.60, n_fh)
    wls_sh = np.linspace(0.755, 0.795, n_sh)
    d_lam_fh = (wls_fh - WL_FH) * 1e3  # detuning [nm]

    # (a) FH extinction ratio for the three taper/bend variants
    er_fh: dict[str, np.ndarray] = {}
    bars_fh: dict[str, np.ndarray] = {}
    crosses_fh: dict[str, np.ndarray] = {}
    for variant in ("faquad_bends", "faquad_taper", "linear_taper"):
        design, component = _design(model, variant)
        bars, crosses = _spectrum(component, wls_fh, model, design)
        bars_fh[variant], crosses_fh[variant] = bars, crosses
        er_fh[variant] = 10 * np.log10(
            np.maximum(crosses, 1e-9) / np.maximum(bars, 1e-9)
        )

    # (b) SH extinction ratio of the proposed (faquad_bends) design
    design_b, component_b = _design(model, "faquad_bends")
    bars_sh, crosses_sh = _spectrum(component_b, wls_sh, model, design_b)
    er_sh = 10 * np.log10(np.maximum(bars_sh, 1e-9) / np.maximum(crosses_sh, 1e-9))

    # (d) FH fabrication-tolerance map (etch depth x top width)
    grid_e = pick(low=np.array([-25.0, 0.0, 25.0]),
                  medium=np.array([-25.0, -12.5, 0.0, 12.5, 25.0]),
                  high=np.array([-25.0, -12.5, 0.0, 12.5, 25.0]))
    grid_w = pick(low=np.array([-50.0, 0.0, 50.0]),
                  medium=np.array([-50.0, -25.0, 0.0, 25.0, 50.0]),
                  high=np.array([-50.0, -25.0, 0.0, 25.0, 50.0]))
    d_etch, d_width = grid_e, grid_w
    loss_map = np.array(
        [[_fab_tolerance_loss(model, de, dw) for dw in d_width] for de in d_etch]
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    labels = {
        "faquad_bends": ("FAQUAD bends", "C3"),
        "faquad_taper": ("FAQUAD taper", "C1"),
        "linear_taper": ("Linear taper", "k"),
    }
    for variant, (lbl, col) in labels.items():
        axes[0, 0].plot(d_lam_fh, er_fh[variant], "o-", color=col, label=lbl, ms=3)
    axes[0, 0].set_xlabel("$\\Delta\\lambda$ [nm]")
    axes[0, 0].set_ylabel("extinction ratio [dB]")
    axes[0, 0].set_title("Fig. 2a: FH extinction ratio")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(visible=True)

    axes[0, 1].plot(wls_sh * 1e3, er_sh, "C2o-", ms=3)
    axes[0, 1].set_xlabel("wavelength [nm]")
    axes[0, 1].set_ylabel("extinction ratio [dB]")
    axes[0, 1].set_title("Fig. 2b: SH extinction ratio")
    axes[0, 1].grid(visible=True)

    # (c) total loss (solid) and radiated loss (dashed) at FH and SH
    b_fh, c_fh = bars_fh["faquad_bends"], crosses_fh["faquad_bends"]
    loss_fh_total = -10 * np.log10(np.maximum(c_fh, 1e-9))  # power not in cross
    loss_fh_rad = -10 * np.log10(np.maximum(b_fh + c_fh, 1e-9))  # radiated only
    loss_sh_total = -10 * np.log10(np.maximum(bars_sh, 1e-9))  # power not in bar
    loss_sh_rad = -10 * np.log10(np.maximum(bars_sh + crosses_sh, 1e-9))
    axc = axes[1, 0]
    axc.plot(wls_fh * 1e3, loss_fh_total, "C3-", label="FH total")
    axc.plot(wls_fh * 1e3, loss_fh_rad, "C3--", label="FH radiated")
    axc.set_xlabel("FH wavelength [nm]", color="C3")
    axc.set_ylabel("loss [dB]")
    axt = axc.twiny()
    axt.plot(wls_sh * 1e3, loss_sh_total, "C2-", label="SH total")
    axt.plot(wls_sh * 1e3, loss_sh_rad, "C2--", label="SH radiated")
    axt.set_xlabel("SH wavelength [nm]", color="C2")
    axc.set_title("Fig. 2c: total (solid) and radiated (dashed) loss")
    axc.grid(visible=True)
    lines = axc.get_lines() + axt.get_lines()
    axc.legend(lines, [ln.get_label() for ln in lines], fontsize=7)

    axd = axes[1, 1]
    im = axd.imshow(
        loss_map,
        origin="lower",
        aspect="auto",
        extent=(d_width[0], d_width[-1], d_etch[0], d_etch[-1]),
        cmap="viridis",
    )
    for i, de in enumerate(d_etch):
        for j, dw in enumerate(d_width):
            axd.text(dw, de, f"{loss_map[i, j]:.2f}", ha="center", va="center",
                     fontsize=7, color="w")
    fig.colorbar(im, ax=axd, label="loss [dB]")
    axd.set_xlabel("$\\Delta$ top width [nm]")
    axd.set_ylabel("$\\Delta$ etch depth [nm]")
    axd.set_title("Fig. 2d: FH total loss vs fabrication offset")

    fig.suptitle(
        f"Kwolek 2026, Fig. 2 [{model} LN]: simulated performance (EME); "
        f"{_resolution.level()}-res"
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / f"kwolek2026_fig2_{model}.png", dpi=150)
    _show(fig)
    plt.close(fig)
    return {
        "er_fh_db_max": float(np.max(er_fh["faquad_bends"])),
        "er_sh_db_min": float(np.min(er_sh)),
        "loss_fh_db_min": float(np.min(loss_fh_total)),
        "fab_loss_db_max": float(np.max(loss_map)),
    }


# --------------------------------------------------------------------------
# Figure 5 (supplemental): coupling-model verification
# --------------------------------------------------------------------------
def _coupling_sweep(
    model: str, gaps: np.ndarray, dtws: np.ndarray
) -> tuple[np.ndarray, np.ndarray, FaquadDesign]:
    """FDE-simulated (chi error [deg], Gamma error [%]) over the (gap, dTW) grid.

    For each grid point the two-rib symmetric/antisymmetric splitting gives the
    coupling magnitude ``Gamma_sim``; the isolated single-rib indices give the
    half phase-mismatch ``dbeta_sim``; ``kappa_sim = sqrt(Gamma^2 - dbeta^2)``
    and ``chi_sim = atan2(kappa, dbeta)``. These are compared against the reduced
    model ``kappa = kappa_0 exp(-g/g0)``, ``dbeta = s * dTW / 2``.
    """
    kappa_0, g_0, dbeta_dtw = calibrate(
        WL_FH, res=RES, compute_modes=BACKEND, model=model
    )
    design = FaquadDesign(kappa_0, g_0, dbeta_dtw, l_m=FIG_L_M)
    k0 = 2 * np.pi / WL_FH
    mesh = calib_mesh(RES)
    chi_err = np.zeros((len(gaps), len(dtws)))
    gam_err = np.zeros((len(gaps), len(dtws)))
    for i, g in enumerate(gaps):
        for j, dtw in enumerate(dtws):
            w_a, w_b = W_TOP + dtw / 2, W_TOP - dtw / 2
            x0 = (W_TOP + g) / 2
            n_p, n_m = solve_te_neffs(
                rib_structures(WL_FH, [w_a, w_b], [-x0, x0], model),
                WL_FH, mesh, compute_modes=BACKEND,
            )
            gamma_sim = 0.5 * k0 * (n_p - n_m)
            na = solve_te_neffs(rib_structures(WL_FH, [w_a], [0.0], model),
                                WL_FH, mesh, 4, 1, compute_modes=BACKEND)[0]
            nb = solve_te_neffs(rib_structures(WL_FH, [w_b], [0.0], model),
                                WL_FH, mesh, 4, 1, compute_modes=BACKEND)[0]
            dbeta_sim = 0.5 * k0 * (na - nb)
            kappa_sim = np.sqrt(max(gamma_sim**2 - dbeta_sim**2, 0.0))
            chi_sim = np.arctan2(kappa_sim, dbeta_sim)
            kappa_m = kappa_0 * np.exp(-g / g_0)
            dbeta_m = 0.5 * dbeta_dtw * dtw
            gamma_m = np.hypot(kappa_m, dbeta_m)
            chi_m = np.arctan2(kappa_m, dbeta_m)
            chi_err[i, j] = np.degrees(chi_sim - chi_m)
            gam_err[i, j] = 100.0 * (gamma_sim - gamma_m) / max(gamma_m, 1e-12)
    return chi_err, gam_err, design


def figure5(model: str) -> dict[str, float]:
    """Coupling-model verification + realized adiabaticity (paper Fig. 5)."""
    n_g = pick(low=5, medium=9, high=13)
    n_d = pick(low=5, medium=9, high=13)
    gaps = np.linspace(G_M, 1.20, n_g)
    dtws = np.linspace(0.0, 0.40, n_d)
    chi_err, gam_err, design = _coupling_sweep(model, gaps, dtws)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ext = (dtws[0], dtws[-1], gaps[0], gaps[-1])
    im0 = axes[0].imshow(chi_err, origin="lower", aspect="auto", extent=ext,
                         cmap="PuOr", vmin=-2, vmax=2)
    fig.colorbar(im0, ax=axes[0], label="$\\chi$ error [deg]")
    axes[0].set_title("Fig. 5a: mixing-angle error")

    im1 = axes[1].imshow(gam_err, origin="lower", aspect="auto", extent=ext,
                         cmap="Greens")
    fig.colorbar(im1, ax=axes[1], label="$\\Gamma$ error [%]")
    axes[1].set_title("Fig. 5b: coupling-magnitude error")

    # the trajectory traced by the design (gap vs |dTW|) in Regions I-II
    z_traj = np.linspace(0.0, design.z_c, 200)
    axes[0].plot(np.abs(design.dtw(z_traj)), design.gap(z_traj), "r-", lw=1.5)
    axes[1].plot(np.abs(design.dtw(z_traj)), design.gap(z_traj), "r-", lw=1.5)
    for ax in axes[:2]:
        ax.set_xlabel("$\\Delta$ top width [um]")
        ax.set_ylabel("top gap [um]")

    # (c) realized adiabaticity eta(z): designed vs FAQUAD vs constant-width
    z = np.linspace(-design.half_length, design.half_length, 1200)
    eta_faquad = design.adiabaticity(z)
    eta_cw = design.adiabaticity(z, constant_width=True)
    axes[2].axhline(design.eta, color="k", ls="--", label="designed adiabaticity")
    axes[2].plot(z, np.clip(eta_cw, 0, 1.0), "C2", label="constant-width bends")
    axes[2].plot(z, np.clip(eta_faquad, 0, 1.0), "C1", label="FAQUAD bends")
    for zb in (design.l_m / 2, design.z_ii):
        axes[2].axvspan(zb, design.z_ii, color="0.85", alpha=0.0)
    axes[2].axvspan(design.l_m / 2, design.z_ii, color="0.85")
    axes[2].axvspan(-design.z_ii, -design.l_m / 2, color="0.85")
    axes[2].set_ylim(0, min(0.6, 3 * design.eta))
    axes[2].set_xlim(-design.z_ii - 30, design.z_ii + 30)
    axes[2].set_xlabel("z [um]")
    axes[2].set_ylabel("$\\eta$")
    axes[2].set_title("Fig. 5c: realized adiabaticity")
    axes[2].legend(fontsize=8)
    axes[2].grid(visible=True)

    fig.suptitle(
        f"Kwolek 2026, Fig. 5 [{model} LN]: coupling-model verification; "
        f"{_resolution.level()}-res"
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / f"kwolek2026_fig5_{model}.png", dpi=150)
    _show(fig)
    plt.close(fig)
    return {
        "chi_err_max_deg": float(np.max(np.abs(chi_err))),
        "gamma_err_max_pct": float(np.max(np.abs(gam_err))),
        "eta": float(design.eta),
    }


# --------------------------------------------------------------------------
# Broad-band (> 1 octave) transmission
# --------------------------------------------------------------------------
def figure_broadband(model: str) -> dict[str, float]:
    """Bar and cross transmission across more than an octave (~0.8*SH..1.2*FH)."""
    design, component = _design(model)
    n = pick(low=11, medium=31, high=61)
    wls = np.linspace(0.8 * WL_SH, 1.2 * WL_FH, n)  # 0.62 .. 1.86 um (>1.5 octave)
    bars, crosses = _spectrum(component, wls, model, design)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(wls * 1e3, bars, "C0o-", ms=3, label="bar (input) port")
    ax.plot(wls * 1e3, crosses, "C3s-", ms=3, label="cross port")
    ax.axvline(WL_SH * 1e3, color="C2", ls=":", lw=1, label="SH 775 nm")
    ax.axvline(WL_FH * 1e3, color="C1", ls=":", lw=1, label="FH 1550 nm")
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("transmission")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(
        f"Kwolek 2026 broadband [{model} LN]: bar/cross over >1 octave; "
        f"{_resolution.level()}-res"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(visible=True)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"kwolek2026_broadband_{model}.png", dpi=150)
    _show(fig)
    plt.close(fig)
    return {
        "fh_cross": float(np.interp(WL_FH, wls, crosses)),
        "sh_bar": float(np.interp(WL_SH, wls, bars)),
    }


def main() -> dict[str, object]:
    FIGDIR.mkdir(exist_ok=True, parents=True)
    out: dict[str, object] = {}
    for model in MODELS:
        out[model] = {
            "fig1": figure1(model),
            "fig2": figure2(model),
            "fig5": figure5(model),
            "broadband": figure_broadband(model),
        }
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
