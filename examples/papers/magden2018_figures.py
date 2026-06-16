"""Reproduce the model/layout figures of Magden et al., Nat. Commun. 9, 3009.

Generates (into ``examples/papers/figures/``):

- ``magden2018_fig1.png``: coupled-mode profiles below/at/above cutoff
  (paper Fig. 1d), isolated WGA/WGB effective indices vs wavelength with the
  phase-matching cutoffs (Fig. 1e), and supermode effective indices (Fig. 1f).
- ``magden2018_fig2.png``: half phase mismatch delta and coupling |kappa| vs
  wavelength (Fig. 2a), analytical vs EME-simulated transmission around the
  cutoff (Fig. 2b), and extinction ratio vs gamma (Fig. 2c).
- ``magden2018_fig3.png``: the gdsfactory filter layout (Fig. 3a) and EME
  transmission of the quasi-even mode vs the length of each adiabatic
  section (Fig. 3b-d).
- ``magden2018_fig4.png``: simulated top-down light propagation through the
  full device with EME, at wavelengths spanning the cutoff (paper Fig. 4a-e):
  short-pass light evolves into WGA, long-pass light stays in WGB.
- ``magden2018_fig5.png``: short-pass and long-pass spectra of the filter
  from the FDE-calibrated mode-evolution model (the model counterpart of
  the measured Fig. 5a) and the cutoff shift with WGA width (Fig. 5d).

The FDE backend ("tidy3d"/"mpb"/"lumerical") and whether the EME is cascaded
with the parallel engine are selected via the ``MEOW_PAPER_BACKEND`` and
``MEOW_PAPER_PARALLEL`` environment variables (see ``examples/papers/
_backends.py``).

Run with ``MEOW_EXAMPLE_FAST=1`` for a coarse-but-quick version (used by the
test suite); the default settings take tens of minutes on a laptop.
"""

from __future__ import annotations

import os
from pathlib import Path

import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np

import meow as mw
from examples.papers._backends import parallel_enabled, resolve_backend
from examples.papers._plot import plot_component
from examples.papers.magden2018_dichroic import (
    H_SI,
    W_A,
    analytical_transmission,
    coupled_structures,
    coupled_supermode_neffs,
    delta_kappa,
    device_cells,
    device_mesh,
    dichroic_filter,
    fundamental_neff,
    lateral_positions,
    mesh2d,
    solve_modes,
    wga_structures,
    wgb_structures,
)

gf.gpdk.PDK.activate()

FAST = bool(int(os.environ.get("MEOW_EXAMPLE_FAST", "0")))
FIGDIR = Path(__file__).parent / "figures"

RES = 0.05 if FAST else 0.02
NUM_MODES = 3 if FAST else 4

# FDE backend ("tidy3d"/"mpb"/"lumerical" or MEOW_PAPER_BACKEND) and whether to
# cascade the EME with the parallel slice-group engine (MEOW_PAPER_PARALLEL).
BACKEND = resolve_backend()
PARALLEL = parallel_enabled()


def _mesh() -> mw.Mesh2D:
    return mesh2d(res=RES)


def figure1() -> dict[str, float]:
    """Mode profiles, isolated-waveguide and supermode effective indices."""
    mesh = _mesh()
    fig = plt.figure(figsize=(13, 8))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])

    # (top row) quasi-even mode below / at / above the cutoff (Fig. 1d)
    wls_probe = [1.50, 1.54, 1.58]
    labels = [
        "$\\lambda < \\lambda_C$",
        "$\\lambda \\approx \\lambda_C$",
        "$\\lambda > \\lambda_C$",
    ]
    for i, (wl, label) in enumerate(zip(wls_probe, labels, strict=True)):
        ax = fig.add_subplot(grid[0, i])
        modes = solve_modes(
            coupled_structures(), wl, mesh=mesh, num_modes=2, compute_modes=BACKEND
        )
        mode = modes[0]
        X, Y = mode.cs.mesh.Xx, mode.cs.mesh.Yx
        ax.pcolormesh(X, Y, np.abs(mode.Ex) ** 2, cmap="inferno")
        ax.set_title(f"{label} ({wl * 1e3:.0f} nm)")
        ax.set_xlim(-0.8, 2.6)
        ax.set_ylim(-0.4, 0.6)
        ax.set_aspect("equal")
        ax.set_xlabel("x [um]")

    # (bottom left/middle) Fig. 1e: isolated waveguide neff vs wavelength
    n_wl = 5 if FAST else 11
    wls = np.linspace(1.49, 1.59, n_wl)
    widths_a = [0.312, 0.318, 0.324]
    ax_e = fig.add_subplot(grid[1, 0:2])
    n_b = np.array(
        [
            fundamental_neff(wgb_structures(), wl, mesh=mesh, compute_modes=BACKEND)
            for wl in wls
        ]
    )
    ax_e.plot(wls * 1e3, n_b, "k--", label="WGB (3 segments)")
    cutoffs: dict[str, float] = {}
    for w_a in widths_a:
        n_a = np.array(
            [
                fundamental_neff(
                    wga_structures(w_a), wl, mesh=mesh, compute_modes=BACKEND
                )
                for wl in wls
            ]
        )
        ax_e.plot(wls * 1e3, n_a, label=f"WGA {w_a * 1e3:.0f} nm")
        # cutoff = phase-matching wavelength (paper Fig. 1e / Fig. 4d):
        # n_a - n_b decreases through zero with wavelength
        diff = n_a - n_b
        wl_c = np.interp(0.0, diff[::-1], wls[::-1])
        cutoffs[f"{w_a * 1e3:.0f}"] = float(wl_c)
    ax_e.set_xlabel("wavelength [nm]")
    ax_e.set_ylabel("effective index")
    ax_e.set_title("Fig. 1e: isolated WGA / WGB effective indices")
    ax_e.legend(fontsize=8)
    ax_e.grid(visible=True)

    # (bottom right) Fig. 1f: supermode effective indices near the cutoff.
    #
    # The isolated WGA/WGB indices (psi_A, psi_B) cross at the phase-matching
    # wavelength, while the coupled supermodes (psi_+, psi_-) anticross with a
    # gap set by the coupling kappa. At the 750 nm design gap the supermode
    # splitting is only ~1e-3 in n_eff, which is at the edge of the FDE
    # solver's consistency with the (separately meshed) isolated solves - a
    # direct supermode solve there can yield a splitting smaller than the
    # isolated detuning 2|delta|, which is unphysical and pushes psi_A/psi_B
    # outside [psi_-, psi_+]. We therefore build the avoided crossing from
    # coupled-mode theory: psi_+- = n_avg +- sqrt(delta^2 + kappa^2), with the
    # detuning from the isolated solves and the coupling measured directly as
    # the half-splitting at the phase-matching wavelength (where delta = 0, so
    # the splitting is purely 2*kappa). This guarantees the textbook ordering
    # psi_- <= psi_A, psi_B <= psi_+ that the paper's Fig. 1f shows.
    ax_f = fig.add_subplot(grid[1, 2])
    n_wl_f = 5 if FAST else 9
    wl_c0 = cutoffs[f"{W_A * 1e3:.0f}"]
    wls_f = np.linspace(wl_c0 - 0.008, wl_c0 + 0.008, n_wl_f)
    # same isolated cross-sections as Fig. 1e, so psi_A and psi_B cross exactly
    # at the cutoff wl_c0.
    n_as = np.array(
        [
            fundamental_neff(wga_structures(W_A), wl, mesh=mesh, compute_modes=BACKEND)
            for wl in wls_f
        ]
    )
    n_bs = np.array(
        [
            fundamental_neff(wgb_structures(), wl, mesh=mesh, compute_modes=BACKEND)
            for wl in wls_f
        ]
    )
    n_p_c, n_m_c = coupled_supermode_neffs(wl_c0, mesh=mesh, compute_modes=BACKEND)
    kappa_neff = 0.5 * (n_p_c - n_m_c)  # coupling (n_eff units) at phase matching
    n_avg = 0.5 * (n_as + n_bs)
    split = np.sqrt((0.5 * (n_as - n_bs)) ** 2 + kappa_neff**2)
    n_plus, n_minus = n_avg + split, n_avg - split
    ax_f.plot(wls_f * 1e3, n_plus, "C0", label="$\\psi_+$")
    ax_f.plot(wls_f * 1e3, n_minus, "C1", label="$\\psi_-$")
    ax_f.plot(wls_f * 1e3, n_as, "k:", label="$\\psi_A$")
    ax_f.plot(wls_f * 1e3, n_bs, "k--", label="$\\psi_B$")
    ax_f.set_xlabel("wavelength [nm]")
    ax_f.set_title("Fig. 1f: supermode anticrossing")
    ax_f.legend(fontsize=8)
    ax_f.grid(visible=True)

    fig.suptitle("Magden 2018, Fig. 1: spectrally selective waveguide modes")
    fig.tight_layout()
    fig.savefig(FIGDIR / "magden2018_fig1.png", dpi=150)
    plt.close(fig)
    return cutoffs


def figure2(wl_c: float) -> dict[str, float]:
    """delta/|kappa| dispersion and the resulting filter roll-off."""
    mesh = _mesh()
    n_wl = 5 if FAST else 11
    wls = np.linspace(wl_c - 0.02, wl_c + 0.02, n_wl)
    deltas, kappas = [], []
    for wl in wls:
        d, k = delta_kappa(wl, mesh=mesh, compute_modes=BACKEND)
        deltas.append(d)
        kappas.append(k)
    deltas = np.asarray(deltas)
    # floor kappa: far from phase matching the supermode-splitting extraction
    # of kappa degenerates (and may clamp to 0 on coarse meshes)
    kappas = np.maximum(np.asarray(kappas), 1e-4)
    gamma = deltas / kappas
    t_a = analytical_transmission(gamma)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    ax = axes[0]
    ax.plot(wls * 1e3, deltas, "C3", label="$\\delta(\\lambda)$")
    ax.plot(wls * 1e3, kappas, "C0", label="$|\\kappa(\\lambda)|$")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("[1/um]")
    ax.set_title("Fig. 2a: $\\delta$, $|\\kappa|$ (g = 750 nm)")
    ax.legend(fontsize=8)
    ax.grid(visible=True)

    ax = axes[1]
    ax.plot(wls * 1e3, t_a, "C3", label="analytical $|T_A|^2$")
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("transmission")
    ax.set_title("Fig. 2b: power in WGA around the cutoff")
    ax.legend(fontsize=8)
    ax.grid(visible=True)

    gammas = np.linspace(-10, 10, 201)
    t = analytical_transmission(gammas)
    er = 10 * np.log10(t / (1 - t))
    ax = axes[2]
    ax.plot(gammas, er, "C0")
    ax.set_xlabel("$\\gamma$")
    ax.set_ylabel("extinction ratio [dB]")
    ax.set_title("Fig. 2c: extinction ratio vs $\\gamma$")
    ax.grid(visible=True)

    fig.suptitle("Magden 2018, Fig. 2: coupled-mode filter response")
    fig.tight_layout()
    fig.savefig(FIGDIR / "magden2018_fig2.png", dpi=150)
    plt.close(fig)

    roll = np.gradient(10 * np.log10(np.maximum(1 - t_a, 1e-9)), wls * 1e3)
    return {"max_rolloff_db_per_nm": float(np.max(np.abs(roll)))}


def _section_sweep(
    cells: list[mw.Cell],
    section_slices: dict[str, slice],
    wls: tuple[float, float],
    lengths: np.ndarray,
) -> dict[str, dict[float, np.ndarray]]:
    """EME transmission of the quasi-even mode vs section length.

    The modes of each section are solved once per wavelength; sweeping the
    section length only rescales the propagation phases (cell_lengths), as
    a linearly stretched taper passes through the same cross-sections.
    """
    results: dict[str, dict[float, np.ndarray]] = {}
    for name, sl in section_slices.items():
        section_cells = cells[sl]
        design_length = sum(c.length for c in section_cells)
        results[name] = {}
        for wl in wls:
            env = mw.Environment(wl=wl, T=25.0)
            css = [mw.CrossSection.from_cell(cell=c, env=env) for c in section_cells]
            modes = [BACKEND(cs, num_modes=NUM_MODES) for cs in css]
            base_lengths = np.asarray([c.length for c in section_cells])
            trans = []
            for length in lengths:
                S, pm = mw.compute_s_matrix(
                    modes,
                    cell_lengths=list(base_lengths * (length / design_length)),
                )
                trans.append(
                    float(np.abs(np.asarray(S)[pm["right@0"], pm["left@0"]]) ** 2)
                )
            results[name][wl] = np.asarray(trans)
    return results


def figure3() -> dict[str, float]:
    """Layout + EME convergence of the four adiabatic sections (Fig. 3)."""
    component = dichroic_filter()
    n1, n2, n3, n4 = (3, 4, 6, 3) if FAST else (6, 10, 48, 8)
    cells = device_cells(component, cells_per_section=(n1, n2, n3, n4), mesh=_mesh())
    section_slices = {
        "(1) develop WGB": slice(0, n1),
        "(2) taper WGA": slice(n1, n1 + n2),
        "(3) separate": slice(n1 + n2, n1 + n2 + n3),
        "(4) merge WGB": slice(n1 + n2 + n3, n1 + n2 + n3 + n4),
    }
    n_lengths = 4 if FAST else 8
    lengths = np.linspace(40, 1100, n_lengths)
    wls = (1.53, 1.55)  # below / above the ~1540 nm cutoff (paper Fig. 3b-d)
    sweeps = _section_sweep(cells, section_slices, wls, lengths)

    fig = plt.figure(figsize=(13, 7))
    grid = fig.add_gridspec(2, 4, height_ratios=[1, 1.3])
    ax_layout = fig.add_subplot(grid[0, :])
    plot_component(component, ax_layout)
    ax_layout.set_title(
        "Fig. 3a: dichroic filter layout (gdsfactory) - x: propagation [um]"
    )

    summary: dict[str, float] = {}
    for i, (name, sweep) in enumerate(sweeps.items()):
        ax = fig.add_subplot(grid[1, i])
        for wl, trans in sweep.items():
            db = 10 * np.log10(np.maximum(trans, 1e-9))
            ax.plot(
                lengths, db, marker="o", ms=3, label=f"$\\lambda$ = {wl * 1e3:.0f} nm"
            )
            summary[f"{name} @ {wl * 1e3:.0f}nm"] = float(db[-1])
        ax.set_xlabel("section length [um]")
        ax.set_ylabel("transmission [dB]")
        ax.set_title(name)
        ax.legend(fontsize=7)
        ax.grid(visible=True)

    fig.suptitle("Magden 2018, Fig. 3: adiabatic transition optimization (EME)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "magden2018_fig3.png", dpi=150)
    plt.close(fig)
    return summary


def _supermode_intensity_slice(mode: mw.Mode) -> tuple[np.ndarray, np.ndarray]:
    """``(x, |Ex(x)|^2)`` of a mode along the y-midplane of the Si core."""
    Xx, Yx = mode.cs.mesh.Xx, mode.cs.mesh.Yx
    x = np.asarray(Xx[:, 0])
    y = np.asarray(Yx[0, :])
    jy = int(np.argmin(np.abs(y - H_SI / 2)))
    return x, np.abs(np.asarray(mode.Ex)[:, jy]) ** 2


def _quasi_even_branch(modes_per_cell: list[list[mw.Mode]]) -> list[mw.Mode]:
    """Follow the quasi-even supermode psi_+ along the device.

    The adiabatic short/long-pass routing is carried by the quasi-even
    (higher-index) supermode. At the example's weak FDE coupling the WGA and
    WGB branches are nearly degenerate at phase matching, so tracking the
    *input* branch by overlap follows the diabatic (WGB) path. Instead we seed
    psi_+ as the highest-index TE mode at the per-slice phase-matching cell
    (the smallest neff gap between the two coupled supermodes) and track it by
    field overlap *outward* in both directions. Below the cutoff this carries
    the field from the input WGB strip across to WGA; above the cutoff it stays
    in WGB.
    """
    te_per_cell = [
        [m for m in modes if m.te_fraction > 0.5] or list(modes)
        for modes in modes_per_cell
    ]
    gaps = [
        float(np.real(te[0].neff) - np.real(te[1].neff)) if len(te) >= 2 else np.inf
        for te in te_per_cell
    ]
    seed = int(np.argmin(gaps))
    chosen: list[mw.Mode | None] = [None] * len(te_per_cell)
    chosen[seed] = te_per_cell[seed][0]  # psi_+ = higher-index supermode

    def best_overlap(ref: mw.Mode, candidates: list[mw.Mode]) -> mw.Mode:
        return max(candidates, key=lambda m: abs(complex(mw.inner_product(ref, m))))

    for i in range(seed + 1, len(te_per_cell)):
        chosen[i] = best_overlap(chosen[i - 1], te_per_cell[i])
    for i in range(seed - 1, -1, -1):
        chosen[i] = best_overlap(chosen[i + 1], te_per_cell[i])
    return [m for m in chosen if m is not None]


def figure4(cutoffs: dict[str, float]) -> dict[str, float]:
    """Simulated top-down light propagation through the device (paper Fig. 4).

    Shows the adiabatic mode-evolution field: at each cross-section along the
    device the highest-index TE supermode - the adiabatic continuation of the
    single input mode - is solved, and its ``|Ex|^2`` along the Si y-midplane
    is stacked into a top-down ``(z, x)`` map at several wavelengths spanning
    the cutoff (paper Fig. 4a-e). The bright band tracks the supermode from
    the input WGB strip (x ~ 0): below the cutoff it crosses into WGA and
    exits the short-pass port (bottom, negative x); above the cutoff it stays
    in WGB (long-pass, x ~ 0).

    This is the field the *designed* (adiabatic) device carries. A brute-force
    full-device EME at the example's FDE-calibrated coupling is instead
    strongly diabatic (see the module note in magden2018_dichroic.py), so the
    quantitative roll-off is taken from the coupled-mode model in figure 5.
    """
    component = dichroic_filter()
    wl_c = cutoffs[f"{W_A * 1e3:.0f}"]
    if FAST:
        wls = [wl_c - 0.03, wl_c, wl_c + 0.03]
        n_sec = (4, 6, 18, 4)
    else:
        wls = [wl_c - 0.04, wl_c - 0.02, wl_c, wl_c + 0.02, wl_c + 0.04]
        n_sec = (8, 12, 48, 8)
    mesh = device_mesh(res=RES)
    _, _, y_a_final = lateral_positions()
    cells = device_cells(component, cells_per_section=n_sec, mesh=mesh)
    z_centers = np.cumsum([c.length for c in cells]) - 0.5 * np.asarray(
        [c.length for c in cells]
    )

    fig, axes = plt.subplots(len(wls), 1, figsize=(11, 1.9 * len(wls)), squeeze=False)
    out: dict[str, float] = {}
    for ax, wl in zip(axes[:, 0], wls, strict=True):
        env = mw.Environment(wl=wl, T=25.0)
        css = [mw.CrossSection.from_cell(cell=c, env=env) for c in cells]
        modes = [BACKEND(cs, num_modes=NUM_MODES) for cs in css]
        branch = _quasi_even_branch(modes)
        columns, x_ref = [], None
        for mode in branch:
            x, col = _supermode_intensity_slice(mode)
            x_ref = x
            columns.append(col)
        img = np.asarray(columns).T  # (x, z)
        ax.imshow(
            img / max(float(img.max()), 1e-30),
            aspect="auto",
            origin="lower",
            extent=(0.0, float(z_centers[-1]), float(x_ref[0]), float(x_ref[-1])),
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
        )
        ax.axhline(0.0, color="w", lw=0.4, ls=":")
        ax.axhline(y_a_final, color="w", lw=0.4, ls=":")
        tag = " (cutoff)" if abs(wl - wl_c) < 1e-9 else ""
        ax.set_ylabel("x [um]")
        ax.set_title(f"$\\lambda$ = {wl * 1e3:.1f} nm{tag}", fontsize=9)
        # lateral centroid of the output slice: < split => WGA (short-pass)
        split = y_a_final / 2
        out_col = img[:, -1]
        centroid = float(np.sum(x_ref * out_col) / max(out_col.sum(), 1e-30))
        out[f"out_centroid_{wl * 1e3:.0f}nm"] = centroid
        out[f"out_is_short_{wl * 1e3:.0f}nm"] = float(centroid < split)
    axes[-1, 0].set_xlabel("z [um]")
    fig.suptitle(
        "Magden 2018, Fig. 4: adiabatic mode-evolution field "
        "(short-pass routes to WGA below the cutoff)"
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "magden2018_fig4.png", dpi=150)
    plt.close(fig)
    return out


def figure5(cutoffs: dict[str, float]) -> dict[str, float]:
    """Filter spectra (model counterpart of the measured Fig. 5).

    The short/long-pass spectra follow from the mode-evolution picture: an
    adiabatic device routes the quasi-even mode, so the port powers are the
    coupled-mode |T_A|^2 and 1 - |T_A|^2 of the spectrally selective
    cross-section (paper Eq. 3) with delta(lambda) and kappa(lambda)
    computed by FDE. (A full-device EME would need impractically long
    transitions at our model's kappa; see the module note in
    magden2018_dichroic.py. The per-section EME convergence - the paper's
    actual EME usage - is reproduced in figure 3, and the simulated light
    propagation through the device is shown in figure 4.)
    """
    mesh = _mesh()
    wl_c = cutoffs[f"{W_A * 1e3:.0f}"]
    n_wl = 5 if FAST else 13
    wls = np.linspace(wl_c - 0.04, wl_c + 0.04, n_wl)
    deltas, kappas = [], []
    for wl in wls:
        d, k = delta_kappa(wl, mesh=mesh, compute_modes=BACKEND)
        deltas.append(d)
        kappas.append(k)
    gamma = np.asarray(deltas) / np.maximum(np.asarray(kappas), 1e-4)
    t_short = analytical_transmission(gamma)  # power staying in WGA
    t_long = 1.0 - t_short

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(
        wls * 1e3,
        10 * np.log10(np.maximum(t_short, 1e-6)),
        "C0o-",
        label="short-pass port (WGA)",
    )
    ax.plot(
        wls * 1e3,
        10 * np.log10(np.maximum(t_long, 1e-6)),
        "C3s-",
        label="long-pass port (WGB)",
    )
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("transmission [dB]")
    ax.set_title("Fig. 5a (model): mode-evolution filter spectra")
    ax.legend(fontsize=8)
    ax.grid(visible=True)

    ax = axes[1]
    widths = np.array([float(k) for k in cutoffs])
    lams = np.array([cutoffs[k] for k in cutoffs]) * 1e3
    ax.plot(widths, lams, "C0o-")
    ax.set_xlabel("$w_A$ [nm]")
    ax.set_ylabel("cutoff wavelength [nm]")
    ax.set_title("Fig. 5d: cutoff shift with WGA width")
    ax.grid(visible=True)

    fig.suptitle("Magden 2018, Fig. 5: filter spectra (FDE + coupled-mode model)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "magden2018_fig5.png", dpi=150)
    plt.close(fig)
    crossover = float(np.interp(0.0, t_short[::-1] - t_long[::-1], wls[::-1]))
    return {"crossover_wl_um": crossover}


def main() -> dict[str, object]:
    """Generate all Magden 2018 figures; returns key validation numbers."""
    FIGDIR.mkdir(exist_ok=True, parents=True)
    cutoffs = figure1()
    fig2 = figure2(cutoffs[f"{W_A * 1e3:.0f}"])
    fig3 = figure3()
    fig4 = figure4(cutoffs)
    fig5 = figure5(cutoffs)
    return {"cutoffs_um": cutoffs, **fig2, **fig3, **fig4, **fig5}


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
