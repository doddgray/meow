"""Reproduce the model/layout figures of Kwolek et al., arXiv:2603.27034.

Generates (into ``examples/papers/figures/``):

- ``kwolek2026_fig1.png``: the gdsfactory combiner layout (paper Fig. 1a),
  the gap and top-width-difference profiles (Fig. 1b), the FAQUAD mixing
  angle chi(z) (Fig. 1c), the symmetric/antisymmetric supermodes at the
  device midpoint and at the decoupling gap (Fig. 1d), and the EME field
  propagation at the fundamental (FH) and second harmonic (SH) (Fig. 1e),
  with the nominal bar/cross performance at both wavelengths (Fig. 1f).
- ``kwolek2026_fig2.png``: extinction ratio and total loss spectra at FH
  and SH (the model counterparts of paper Fig. 2a-c).

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
    FaquadDesign,
    bar_cross_transmission,
    calibrate,
    device_cells,
    faquad_combiner,
    input_launch_index,
    rib_structures,
    slab_neff,
)

gf.gpdk.PDK.activate()

pick = _resolution.pick
FIGDIR = Path(__file__).parent / "figures"

WL_FH = 1.55
WL_SH = 0.775
# Accuracy knobs (low / medium / high via MEOW_EXAMPLE_RES). The medium values
# already give a converged reproduction; high pushes the mode-solver grid
# (RES), the number of EME cells (NUM_CELLS) and the modes per cross-section
# (NUM_MODES) further still.
# The combiner is a weakly-guided shallow-etched rib whose mode sits only
# ~0.02-0.1 in index above the slab continuum, so the EME needs a fairly fine
# transverse mesh and -- crucially -- enough modes per cross-section to both
# resolve the guided fundamental(s) and carry the slab radiation in the cascade
# (too few modes makes the cascade non-unitary and the transmission spurious).
# The laterally-separating waveguides also need many cells to avoid staircase
# misalignment loss. The medium values give a converged, power-consistent
# reproduction; low is a coarse-but-quick look.
RES = pick(low=0.05, medium=0.025, high=0.018)
NUM_CELLS = _resolution.num_cells(low=64, medium=150, high=200)
NUM_MODES = _resolution.num_modes(low=8, medium=12, high=16)


def _show(fig: plt.Figure) -> None:
    """Display a freshly generated figure if an interactive backend is active."""
    if not mpl.get_backend().lower().startswith("agg"):
        fig.show()
        plt.pause(0.1)

# FDE backend ("tidy3d"/"mpb"/"lumerical" or MEOW_PAPER_BACKEND) and whether to
# cascade the EME with the parallel slice-group engine (MEOW_PAPER_PARALLEL).
BACKEND = resolve_backend()
PARALLEL = parallel_enabled()


def _design() -> tuple[FaquadDesign, gf.Component]:
    kappa_0, g_0, dbeta_dtw = calibrate(WL_FH, res=RES, compute_modes=BACKEND)
    design = FaquadDesign(kappa_0, g_0, dbeta_dtw)
    component = faquad_combiner(kappa_0, g_0, dbeta_dtw)
    return design, component


def _supermode_panel(ax_pair: list[plt.Axes], gap: float, title: str) -> None:
    """Plot the symmetric/antisymmetric supermodes at a given gap."""
    from examples.papers.kwolek2026_faquad import W_TOP, calib_mesh

    x0 = (W_TOP + gap) / 2
    structures = rib_structures(WL_FH, [W_TOP, W_TOP], [-x0, x0])
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


def figure1() -> dict[str, float]:
    design, component = _design()

    fig = plt.figure(figsize=(13, 11))
    grid = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1.2])

    # (a) layout
    ax = fig.add_subplot(grid[0, :])
    plot_component(component, ax)
    ax.set_title("Fig. 1a: FAQUAD combiner layout (gdsfactory)")

    # (b) gap and top-width-difference profiles
    z = np.linspace(-design.half_length, design.half_length, 400)
    ax = fig.add_subplot(grid[1, 0:2])
    ax.plot(z, design.gap(z) * 1e3, "C0", label="gap g(z)")
    ax.axhline(G_M * 1e3, color="C0", ls=":", lw=0.8)
    ax.axhline(G_C * 1e3, color="C0", ls="--", lw=0.8)
    ax.set_xlabel("z [um]")
    ax.set_ylabel("gap [nm]", color="C0")
    ax2 = ax.twinx()
    ax2.plot(z, design.dtw(z) * 1e3, "C3", label="dTW(z)")
    ax2.set_ylabel("dTW [nm]", color="C3")
    ax.set_title("Fig. 1b: gap and top-width-difference profiles")
    ax.grid(visible=True)

    # (c) FAQUAD mixing angle
    ax = fig.add_subplot(grid[1, 2:4])
    ax.plot(z, design.chi(z) / np.pi, "C2")
    ax.set_xlabel("z [um]")
    ax.set_ylabel("$\\chi(z) / \\pi$")
    ax.set_title("Fig. 1c: FAQUAD coupling angle")
    ax.grid(visible=True)

    # (d) supermodes at the midpoint gap and at the decoupling gap
    axes_d = [fig.add_subplot(grid[2, i]) for i in range(4)]
    _supermode_panel(axes_d[0:2], G_M, "device midpoint")
    _supermode_panel(axes_d[2:4], G_C, "end of FAQUAD evolution")

    # (e) EME field propagation at FH and SH
    results: dict[str, float] = {}
    for i, (wl, label) in enumerate([(WL_FH, "FH 1550 nm"), (WL_SH, "SH 775 nm")]):
        cells = device_cells(component, wl, num_cells=NUM_CELLS, res=RES, design=design)
        env = mw.Environment(wl=wl, T=25.0)
        css = [mw.CrossSection.from_cell(cell=c, env=env) for c in cells]
        modes = [BACKEND(cs, num_modes=NUM_MODES) for cs in css]

        in_idx = input_launch_index(  # launch the guided B (bar) mode
            modes[0], slab_neff(wl, cells[0], compute_modes=BACKEND)
        )
        ex_l = np.zeros(len(modes[0]))
        ex_l[in_idx] = 1.0
        ex_r = np.zeros(len(modes[-1]))
        z_pts = np.linspace(
            0.0, sum(c.length for c in cells), pick(low=400, medium=800, high=1200)
        )
        Ex, x_pts = prop.propagate_modes(
            modes, cells, excitation_l=ex_l, excitation_r=ex_r, y=0.25, z=z_pts
        )
        ax = fig.add_subplot(grid[3, 2 * i : 2 * i + 2])
        ax.imshow(
            np.abs(np.asarray(Ex).T) ** 2,
            aspect="auto",
            origin="lower",
            extent=(
                float(z_pts[0]),
                float(z_pts[-1]),
                float(x_pts[0]),
                float(x_pts[-1]),
            ),
            cmap="inferno",
        )
        ax.set_xlabel("z [um]")
        ax.set_ylabel("x [um]")
        ax.set_ylim(-3, 3)
        ax.set_title(f"Fig. 1e: |Ex|$^2$ propagation at {label}")

        t_bar, t_cross = bar_cross_transmission(
            cells, wl, num_modes=NUM_MODES, parallel=PARALLEL, compute_modes=BACKEND
        )
        results[f"bar_{label.split()[0]}"] = t_bar
        results[f"cross_{label.split()[0]}"] = t_cross

    fig.suptitle(
        "Kwolek 2026, Fig. 1: FAQUAD-optimized TFLN wavelength combiner "
        f"(FH cross = {results['cross_FH']:.3f}, SH bar = {results['bar_SH']:.3f}; "
        f"{_resolution.level()}-res)"
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "kwolek2026_fig1.png", dpi=150)
    _show(fig)
    plt.close(fig)
    return results


def figure2() -> dict[str, float]:
    """Extinction-ratio and loss spectra at FH and SH (paper Fig. 2)."""
    design, component = _design()

    n_fh = pick(low=3, medium=7, high=13)
    n_sh = pick(low=2, medium=5, high=9)
    wls_fh = np.linspace(1.50, 1.60, n_fh)
    wls_sh = np.linspace(0.755, 0.795, n_sh)

    spectra: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, wls in [("FH", wls_fh), ("SH", wls_sh)]:
        bars, crosses = [], []
        for wl in wls:
            cells = device_cells(
                component, wl, num_cells=NUM_CELLS, res=RES, design=design
            )
            t_bar, t_cross = bar_cross_transmission(
                cells, wl, num_modes=NUM_MODES, parallel=PARALLEL, compute_modes=BACKEND
            )
            bars.append(t_bar)
            crosses.append(t_cross)
        spectra[label] = (wls, np.asarray(bars), np.asarray(crosses))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    wls, bars, crosses = spectra["FH"]
    er_fh = 10 * np.log10(np.maximum(crosses, 1e-9) / np.maximum(bars, 1e-9))
    axes[0].plot(wls * 1e3, er_fh, "C0o-")
    axes[0].set_xlabel("wavelength [nm]")
    axes[0].set_ylabel("extinction ratio [dB]")
    axes[0].set_title("Fig. 2a: ER at FH (cross / bar)")
    axes[0].grid(visible=True)

    wls_s, bars_s, crosses_s = spectra["SH"]
    er_sh = 10 * np.log10(np.maximum(bars_s, 1e-9) / np.maximum(crosses_s, 1e-9))
    axes[1].plot(wls_s * 1e3, er_sh, "C3o-")
    axes[1].set_xlabel("wavelength [nm]")
    axes[1].set_ylabel("extinction ratio [dB]")
    axes[1].set_title("Fig. 2b: ER at SH (bar / cross)")
    axes[1].grid(visible=True)

    loss_fh = -10 * np.log10(np.maximum(bars + crosses, 1e-9))
    loss_sh = -10 * np.log10(np.maximum(bars_s + crosses_s, 1e-9))
    axes[2].plot(wls * 1e3, loss_fh, "C0o-", label="FH")
    ax2 = axes[2].twiny()
    ax2.plot(wls_s * 1e3, loss_sh, "C3s-", label="SH")
    axes[2].set_xlabel("FH wavelength [nm]")
    ax2.set_xlabel("SH wavelength [nm]")
    axes[2].set_ylabel("total loss [dB]")
    axes[2].set_title("Fig. 2c: total loss")
    axes[2].grid(visible=True)

    fig.suptitle(
        "Kwolek 2026, Fig. 2: simulated combiner performance (EME); "
        f"{_resolution.level()}-res (SH extinction is optimistic until converged)"
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "kwolek2026_fig2.png", dpi=150)
    _show(fig)
    plt.close(fig)
    return {
        "er_fh_db_min": float(np.min(er_fh)),
        "er_sh_db_min": float(np.min(er_sh)),
        "loss_fh_db_max": float(np.max(loss_fh)),
    }


def main() -> dict[str, object]:
    FIGDIR.mkdir(exist_ok=True, parents=True)
    fig1 = figure1()
    fig2 = figure2()
    return {**fig1, **fig2}


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
