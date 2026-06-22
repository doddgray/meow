"""Analysis + plotting of designed dichroic devices, runnable as slurm jobs.

This module turns a *designed* device (a dichroic beam-splitter coupler or a
FAQUAD wavelength filter) into the figures and data products the paper-based
examples care about:

- a **transmission spectrum** to each output port over a dense set of
  wavelengths (``<label>_spectrum.png`` + the raw arrays in
  ``<label>_results.npz``);
- **propagation plots** of the optical intensity ``|Ex|^2`` along the device at
  a few wavelengths around the cutoff (``<label>_propagation.png``);
- a **design figure** analogous to the paper reproductions (layout + index
  crossings for the dichroic coupler; layout + FAQUAD profiles for the filter);
- the device **GDS** (``<label>.gds``) and a JSON summary.

The two entry points :func:`analyze_dichroic` and :func:`analyze_faquad` are
plain top-level functions that take a *picklable* parameter dict (see
``dichroic_designer.to_params`` / ``kwolek_designer.to_params``) plus a settings
dict, write all their outputs into ``out_dir`` and return a small summary dict.
That makes them shippable to a slurm job: the whole simulation + analysis +
plotting runs on the cluster and only the summary travels back, while the
figures/GDS/data land directly in the job's (timestamped) output folder on the
shared filesystem.

Wavelength controls (sensible defaults, overridable per call or via env vars):

- transmission spectrum: :func:`spectrum_wavelengths` - ``span`` (fractional
  half-width about the cutoff) and ``n`` points
  (``MEOW_SPECTRUM_SPAN`` / ``MEOW_SPECTRUM_NPTS``);
- propagation wavelengths: :func:`propagation_wavelengths` - ``span``/``n``
  (``MEOW_PROP_SPAN`` / ``MEOW_PROP_NPTS``) or an explicit comma-separated list
  ``MEOW_PROP_WLS``.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import meow as mw
import meow.eme.propagation as prop
from examples.papers import dichroic_designer as dd
from examples.papers import kwolek_designer as kd
from examples.papers.magden2018_dichroic import GAP_OUT, lateral_positions

# defaults for the wavelength controls
DEFAULT_SPECTRUM_SPAN = 0.06  # fractional half-width about the cutoff
DEFAULT_SPECTRUM_NPTS = 61
DEFAULT_PROP_SPAN = 0.04
DEFAULT_PROP_NPTS = 5


def _file_stem(label: str) -> str:
    """Filesystem-safe stem for output filenames derived from a run label."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")


# ==========================================================================
# wavelength controls
# ==========================================================================
def spectrum_wavelengths(
    cutoff_wl: float, *, span: float | None = None, n: int | None = None
) -> np.ndarray:
    """Dense transmission-spectrum wavelengths [um] centered on ``cutoff_wl``.

    ``span`` is the fractional half-width (so the band is
    ``cutoff*(1 +- span)``) and ``n`` the number of points; both fall back to
    the ``MEOW_SPECTRUM_SPAN`` / ``MEOW_SPECTRUM_NPTS`` env vars, then to the
    module defaults. ``n`` is forced odd so the cutoff itself is sampled.
    """
    span = float(os.environ.get("MEOW_SPECTRUM_SPAN", span or DEFAULT_SPECTRUM_SPAN))
    n = int(os.environ.get("MEOW_SPECTRUM_NPTS", n or DEFAULT_SPECTRUM_NPTS))
    n = max(3, n | 1)  # odd -> includes the cutoff exactly
    return np.linspace(cutoff_wl * (1 - span), cutoff_wl * (1 + span), n)


def propagation_wavelengths(
    cutoff_wl: float, *, span: float | None = None, n: int | None = None
) -> np.ndarray:
    """Wavelengths [um] for the propagation field plots (around the cutoff).

    An explicit comma-separated ``MEOW_PROP_WLS`` (um) wins; otherwise a
    symmetric set of ``n`` points across ``cutoff*(1 +- span)`` (env
    ``MEOW_PROP_SPAN`` / ``MEOW_PROP_NPTS``), forced odd so the cutoff is one of
    them.
    """
    explicit = os.environ.get("MEOW_PROP_WLS")
    if explicit:
        return np.array([float(x) for x in explicit.split(",") if x.strip()])
    span = float(os.environ.get("MEOW_PROP_SPAN", span or DEFAULT_PROP_SPAN))
    n = int(os.environ.get("MEOW_PROP_NPTS", n or DEFAULT_PROP_NPTS))
    n = max(3, n | 1)
    return np.linspace(cutoff_wl * (1 - span), cutoff_wl * (1 + span), n)


# ==========================================================================
# low-level EME analysis helpers
# ==========================================================================
def _centroid(mode: mw.Mode) -> float:
    d = np.abs(mode.Ex) ** 2
    return float(np.sum(mode.cs.mesh.Xx * d) / np.sum(d))


def _cell_modes(
    cells: list[mw.Cell],
    env: mw.Environment,
    num_modes: int,
    backend: Callable | None = None,
) -> list[list[mw.Mode]]:
    backend = backend or mw.compute_modes
    return [
        backend(mw.CrossSection.from_cell(cell=c, env=env), num_modes=num_modes)
        for c in cells
    ]


def propagate_field(
    cells: list[mw.Cell],
    env: mw.Environment,
    num_modes: int,
    *,
    y: float,
    input_kind: str = "fundamental",
    num_z: int = 400,
    backend: Callable | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate the input excitation and return ``(|Ex|^2(z, x), x, z)``.

    ``input_kind`` selects the launched mode of the first cell: ``"fundamental"``
    excites mode 0; ``"bar"`` excites the mode whose energy centroid is furthest
    to negative x (the FAQUAD bar/input guide).
    """
    modes = _cell_modes(cells, env, num_modes, backend)
    if input_kind == "bar":
        idx = min(range(min(2, len(modes[0]))), key=lambda k: _centroid(modes[0][k]))
    else:
        idx = 0
    z = np.linspace(0.0, float(sum(c.length for c in cells)), num_z)
    Ex, x = prop.propagate_modes(modes, cells, excite_mode_l=idx, y=y, z=z)
    return np.abs(np.asarray(Ex)) ** 2, np.asarray(x), z


def _attribute_split(
    modes_out: list[mw.Mode],
    s_matrix: Any,
    port_map: dict[str, int],
    split: float,
    *,
    in_port: str = "left@0",
) -> tuple[float, float]:
    """(below-split, above-split) output power for the ``in_port`` input.

    Splits the output modes by the lateral position ``split`` of their energy
    centroid (e.g. WGA short-pass below / WGB long-pass above for the dichroic
    coupler, or bar/cross about x = 0 for the FAQUAD filter).
    """
    s = np.asarray(s_matrix)
    below = above = 0.0
    for i, mode in enumerate(modes_out):
        power = float(np.abs(s[port_map[f"right@{i}"], port_map[in_port]]) ** 2)
        if _centroid(mode) < split:
            below += power
        else:
            above += power
    return below, above


# ==========================================================================
# dichroic coupler: cells, ports (re-exported by the slurm examples)
# ==========================================================================
def dichroic_device_mesh(design: dd.DichroicDesign, res: float) -> mw.Mesh2D:
    """Cross-section mesh spanning the full designed dichroic device width."""
    comp = design.component
    plat = design.platform
    x_lo = float(comp.ymin) - 1.0
    x_hi = float(comp.ymax) + 1.0
    h, tcl = plat.core_thickness, plat.clad_thickness
    return mw.Mesh2D(
        x=np.arange(x_lo, x_hi + res / 2, res),
        y=np.arange(-tcl, h + tcl + res / 2, res),
    )


def dichroic_device_cells(
    design: dd.DichroicDesign, num_cells: int = 16, res: float = 0.06
) -> list[mw.Cell]:
    """Slice a designed dichroic device into ``num_cells`` equal-length cells."""
    structs = dd.device_structures(design)
    length = float(design.component.xmax)
    lengths = np.full(num_cells, length / num_cells)
    mesh = dichroic_device_mesh(design, res)
    return mw.create_cells(structs, mesh, lengths, z_min=0.0)


def dichroic_short_pass_split(design: dd.DichroicDesign) -> float:
    """Lateral position [um] separating the WGA short-pass and WGB long-pass
    output ports (output modes with their energy centroid below this go to the
    short-pass port).
    """
    _, _, y_a_final = lateral_positions(
        design.w_a, design.wgb.rail_width, design.wgb.gap, design.gap, GAP_OUT
    )
    return y_a_final / 2


def dichroic_port_powers_from_s(
    cells: list[mw.Cell],
    env: mw.Environment,
    s_matrix: Any,
    port_map: dict[str, int],
    num_modes: int,
    split: float,
    *,
    modes_out: list[mw.Mode] | None = None,
) -> tuple[float, float]:
    """(short-pass, long-pass) power for the fundamental input.

    The output modes are solved once on the final cell (unless ``modes_out`` is
    supplied); the parallel engine returns no field data, so this only needs the
    precomputed ``split`` scalar and can run in a later session.
    """
    if modes_out is None:
        cs_out = mw.CrossSection.from_cell(cell=cells[-1], env=env)
        modes_out = mw.compute_modes(cs_out, num_modes=num_modes)
    return _attribute_split(modes_out, s_matrix, port_map, split)


# ==========================================================================
# plotting
# ==========================================================================
def _use_agg() -> Any:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_transmission_dichroic(
    wls: np.ndarray,
    t_short: np.ndarray,
    t_long: np.ndarray,
    cutoff_wl: float,
    path: Path,
    title: str,
) -> None:
    """Short-/long-pass transmission spectrum [dB] vs wavelength."""
    plt = _use_agg()
    fig, ax = plt.subplots(figsize=(7, 4))
    eps = 1e-6
    ts_db = 10 * np.log10(np.clip(t_short, eps, None))
    tl_db = 10 * np.log10(np.clip(t_long, eps, None))
    ax.plot(wls * 1e3, ts_db, "C0", label="short-pass (WGA)")
    ax.plot(wls * 1e3, tl_db, "C3", label="long-pass (WGB)")
    ax.axvline(cutoff_wl * 1e3, color="0.5", ls=":", lw=0.9, label="cutoff")
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("transmission [dB]")
    ax.set_ylim(max(-42.0, float(min(ts_db.min(), tl_db.min())) - 3), 2)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(visible=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_transmission_faquad(
    wls_fh: np.ndarray,
    bar_fh: np.ndarray,
    cross_fh: np.ndarray,
    wls_sh: np.ndarray,
    bar_sh: np.ndarray,
    cross_sh: np.ndarray,
    path: Path,
    title: str,
) -> None:
    """FH/SH extinction-ratio and loss spectra (model counterpart of Fig. 2)."""
    plt = _use_agg()
    eps = 1e-9
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    er_fh = 10 * np.log10(np.maximum(cross_fh, eps) / np.maximum(bar_fh, eps))
    axes[0].plot(wls_fh * 1e3, er_fh, "C0o-")
    axes[0].set_xlabel("FH wavelength [nm]")
    axes[0].set_ylabel("extinction ratio [dB]")
    axes[0].set_title("ER at FH (cross / bar)")
    axes[0].grid(visible=True)
    er_sh = 10 * np.log10(np.maximum(bar_sh, eps) / np.maximum(cross_sh, eps))
    axes[1].plot(wls_sh * 1e3, er_sh, "C3o-")
    axes[1].set_xlabel("SH wavelength [nm]")
    axes[1].set_ylabel("extinction ratio [dB]")
    axes[1].set_title("ER at SH (bar / cross)")
    axes[1].grid(visible=True)
    loss_fh = -10 * np.log10(np.maximum(bar_fh + cross_fh, eps))
    loss_sh = -10 * np.log10(np.maximum(bar_sh + cross_sh, eps))
    axes[2].plot(wls_fh * 1e3, loss_fh, "C0o-", label="FH")
    ax2 = axes[2].twiny()
    ax2.plot(wls_sh * 1e3, loss_sh, "C3s-", label="SH")
    axes[2].set_xlabel("FH wavelength [nm]")
    ax2.set_xlabel("SH wavelength [nm]")
    axes[2].set_ylabel("total loss [dB]")
    axes[2].set_title("total loss")
    axes[2].grid(visible=True)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_propagation(
    panels: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    path: Path,
    title: str,
    *,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Grid of ``|Ex|^2(z, x)`` maps, one per wavelength."""
    plt = _use_agg()
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.2 * n), squeeze=False)
    for ax, (label, intensity, x, z) in zip(axes[:, 0], panels, strict=True):
        ax.imshow(
            intensity.T,
            aspect="auto",
            origin="lower",
            extent=(float(z[0]), float(z[-1]), float(x[0]), float(x[-1])),
            cmap="inferno",
        )
        ax.set_ylabel("x [um]")
        ax.set_title(f"|Ex|$^2$ at {label}", fontsize=9)
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[-1, 0].set_xlabel("z (propagation) [um]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_dichroic_design(design: dd.DichroicDesign, path: Path, *, n_neff: int) -> None:
    """Layout + WGA/WGB index crossing for one designed dichroic coupler."""
    plt = _use_agg()
    from examples.papers._plot import plot_component

    plat, wgb = design.platform, design.wgb
    fig = plt.figure(figsize=(12, 6))
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 1])

    ax = fig.add_subplot(grid[0, 0])
    wls = np.linspace(design.cutoff_wl * 0.85, design.cutoff_wl * 1.15, n_neff)
    n_b = [dd.segmented_neff(plat, wgb, float(wl), res=0.04) for wl in wls]
    n_a = [dd.solid_neff(plat, design.w_a, float(wl), res=0.04) for wl in wls]
    ax.plot(wls * 1e3, n_b, "k--", lw=2, label="WGB (sub-wavelength)")
    ax.plot(wls * 1e3, n_a, "C0", lw=1.5, label=f"WGA {design.w_a * 1e3:.0f} nm")
    ax.axvline(design.cutoff_wl * 1e3, color="0.5", ls=":", lw=0.9)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("effective index")
    ax.set_title("WGA / WGB index crossing (= cutoff)")
    ax.legend(fontsize=8)
    ax.grid(visible=True)

    ax = fig.add_subplot(grid[1, 0])
    plot_component(design.component, ax)
    ax.set_aspect("auto")
    ax.set_title(
        f"layout: w_a={design.w_a * 1e3:.0f} nm, gap={design.gap * 1e3:.0f} nm, "
        f"L={design.total_length:.0f} um, ER~{design.extinction_db:.0f} dB"
    )
    fig.suptitle(f"Dichroic coupler design ({design.cutoff_wl * 1e3:.0f} nm cutoff)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_faquad_design(design: kd.FaquadFilterDesign, path: Path) -> None:
    """Layout + FAQUAD gap/dTW profiles + mixing angle (Fig. 1a-c counterpart)."""
    plt = _use_agg()
    from examples.papers._plot import plot_component

    fq = design.design
    z = np.linspace(-fq.half_length, fq.half_length, 400)
    fig = plt.figure(figsize=(12, 7))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    ax = fig.add_subplot(grid[0, :])
    plot_component(design.component, ax)
    ax.set_aspect("auto")
    ax.set_title(
        f"FAQUAD combiner layout: w_top={design.w_top * 1e3:.0f} nm, "
        f"L={design.total_length:.0f} um"
    )

    ax = fig.add_subplot(grid[1, 0])
    ax.plot(z, fq.gap(z) * 1e3, "C0", label="gap g(z)")
    ax.set_xlabel("z [um]")
    ax.set_ylabel("gap [nm]", color="C0")
    ax2 = ax.twinx()
    ax2.plot(z, fq.dtw(z) * 1e3, "C3", label="dTW(z)")
    ax2.set_ylabel("dTW [nm]", color="C3")
    ax.set_title("gap and top-width-difference profiles")
    ax.grid(visible=True)

    ax = fig.add_subplot(grid[1, 1])
    ax.plot(z, fq.chi(z) / np.pi, "C2")
    ax.set_xlabel("z [um]")
    ax.set_ylabel(r"$\chi(z) / \pi$")
    ax.set_title("FAQUAD mixing angle")
    ax.grid(visible=True)

    fig.suptitle(
        f"FAQUAD filter {design.platform.name} "
        f"{design.fh_wl * 1e3:.0f}/{design.sh_wl * 1e3:.0f} nm"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_json(path: Path, data: dict) -> None:
    with Path(path).open("w") as f:
        json.dump(data, f, indent=2, default=str)


# ==========================================================================
# top-level analysis jobs (picklable, shippable to a slurm task)
# ==========================================================================
def analyze_dichroic(out_dir: str, spec: dict, settings: dict) -> dict:
    """Full analysis of one dichroic coupler design; returns a summary dict.

    Rebuilds the design from ``spec`` (``dichroic_designer.to_params``), computes
    the short-/long-pass transmission spectrum and the propagation fields, and
    writes the spectrum/propagation/design figures, the GDS and the raw data
    into ``out_dir``. Designed to run as a slurm job.
    """
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    label = settings["label"]
    stem = settings.get("file_stem") or _file_stem(label)
    num_cells = settings["num_cells"]
    num_modes = settings["num_modes"]
    res = settings["device_res"]
    spectrum_wls = np.asarray(settings["spectrum_wls"], dtype=float)
    prop_wls = np.asarray(settings["prop_wls"], dtype=float)
    num_z = int(settings.get("num_z", 400))
    n_neff = int(settings.get("n_neff", 7))

    design = dd.design_from_params(**spec)
    cells = dichroic_device_cells(design, num_cells, res)
    split = dichroic_short_pass_split(design)
    y_core = design.platform.core_thickness / 2.0

    # transmission spectrum (cells are wavelength-independent geometry)
    t_short, t_long = [], []
    for wl in spectrum_wls:
        env = mw.Environment(wl=float(wl), T=25.0)
        modes = _cell_modes(cells, env, num_modes)
        s_matrix, port_map = mw.compute_s_matrix(modes, cells=cells)
        below, above = _attribute_split(modes[-1], s_matrix, port_map, split)
        t_short.append(below)
        t_long.append(above)
    t_short = np.asarray(t_short)
    t_long = np.asarray(t_long)

    # propagation maps around the cutoff
    panels = []
    for wl in prop_wls:
        env = mw.Environment(wl=float(wl), T=25.0)
        intensity, x, z = propagate_field(
            cells, env, num_modes, y=y_core, num_z=num_z
        )
        panels.append((f"{wl * 1e3:.0f} nm", intensity, x, z))

    # figures + GDS + data
    design.component.write_gds(str(out / f"{stem}.gds"))
    plot_dichroic_design(design, out / f"{stem}_design.png", n_neff=n_neff)
    plot_transmission_dichroic(
        spectrum_wls, t_short, t_long, design.cutoff_wl,
        out / f"{stem}_spectrum.png",
        f"{label}: dichroic transmission spectrum",
    )
    plot_propagation(
        panels, out / f"{stem}_propagation.png",
        f"{label}: intensity propagation across the cutoff",
    )
    np.savez(
        out / f"{stem}_results.npz",
        spectrum_wls=spectrum_wls,
        t_short=t_short,
        t_long=t_long,
        prop_wls=prop_wls,
    )

    i_c = int(np.argmin(np.abs(spectrum_wls - design.cutoff_wl)))
    summary = {
        "label": label,
        "kind": "dichroic",
        "cutoff_nm": round(design.cutoff_wl * 1e3, 1),
        "w_a_nm": round(design.w_a * 1e3, 1),
        "gap_nm": round(design.gap * 1e3, 0),
        "length_um": round(design.total_length, 0),
        "short_pass_at_cutoff": round(float(t_short[i_c]), 4),
        "long_pass_at_cutoff": round(float(t_long[i_c]), 4),
        "out_dir": str(out),
        "files": sorted(p.name for p in out.glob(f"{stem}*")),
    }
    save_json(out / f"{stem}_summary.json", summary)
    return summary


def analyze_faquad(out_dir: str, spec: dict, settings: dict) -> dict:
    """Full analysis of one FAQUAD filter design; returns a summary dict.

    Rebuilds the design from ``spec`` (``kwolek_designer.to_params``), computes
    the FH and SH bar/cross transmission spectra and the FH/SH propagation
    fields, and writes the spectrum/propagation/design figures, the GDS and the
    raw data into ``out_dir``. Designed to run as a slurm job.
    """
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    label = settings["label"]
    stem = settings.get("file_stem") or _file_stem(label)
    num_cells = settings["num_cells"]
    num_modes = settings["num_modes"]
    res = settings["device_res"]
    fh_wls = np.asarray(settings["fh_wls"], dtype=float)
    sh_wls = np.asarray(settings["sh_wls"], dtype=float)
    prop_wls = np.asarray(settings["prop_wls"], dtype=float)
    num_z = int(settings.get("num_z", 400))

    design = kd.filter_from_params(**spec)
    y_core = design.platform.core_thickness / 2.0

    def bar_cross_spectrum(wls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bars, crosses = [], []
        for wl in wls:
            cells = kd.device_cells(design, float(wl), num_cells=num_cells, res=res)
            env = mw.Environment(wl=float(wl), T=25.0)
            modes = _cell_modes(cells, env, num_modes)
            s_matrix, port_map = mw.compute_s_matrix(modes, cells=cells)
            # input = bar fundamental (most negative centroid of first cell)
            in_idx = min(
                range(min(2, len(modes[0]))), key=lambda k: _centroid(modes[0][k])
            )
            t_bar, t_cross = _attribute_split(
                modes[-1], s_matrix, port_map, 0.0, in_port=f"left@{in_idx}"
            )
            bars.append(t_bar)
            crosses.append(t_cross)
        return np.asarray(bars), np.asarray(crosses)

    bar_fh, cross_fh = bar_cross_spectrum(fh_wls)
    bar_sh, cross_sh = bar_cross_spectrum(sh_wls)

    # propagation at the FH/SH cutoffs and the requested neighbours
    panels = []
    for wl in prop_wls:
        cells = kd.device_cells(design, float(wl), num_cells=num_cells, res=res)
        env = mw.Environment(wl=float(wl), T=25.0)
        intensity, x, z = propagate_field(
            cells, env, num_modes, y=y_core, input_kind="bar", num_z=num_z
        )
        panels.append((f"{wl * 1e3:.0f} nm", intensity, x, z))

    design.component.write_gds(str(out / f"{stem}.gds"))
    plot_faquad_design(design, out / f"{stem}_design.png")
    plot_transmission_faquad(
        fh_wls, bar_fh, cross_fh, sh_wls, bar_sh, cross_sh,
        out / f"{stem}_spectrum.png",
        f"{label}: FAQUAD filter ER / loss spectra",
    )
    plot_propagation(
        panels, out / f"{stem}_propagation.png",
        f"{label}: intensity propagation (bar input)",
        ylim=(-3, 3),
    )
    np.savez(
        out / f"{stem}_results.npz",
        fh_wls=fh_wls, bar_fh=bar_fh, cross_fh=cross_fh,
        sh_wls=sh_wls, bar_sh=bar_sh, cross_sh=cross_sh,
        prop_wls=prop_wls,
    )

    eps = 1e-9
    i_fh = int(np.argmin(np.abs(fh_wls - design.fh_wl)))
    i_sh = int(np.argmin(np.abs(sh_wls - design.sh_wl)))
    summary = {
        "label": label,
        "kind": "faquad",
        "platform": design.platform.name,
        "fh_nm": round(design.fh_wl * 1e3, 0),
        "sh_nm": round(design.sh_wl * 1e3, 0),
        "length_um": round(design.total_length, 0),
        "fh_cross": round(float(cross_fh[i_fh]), 4),
        "fh_bar": round(float(bar_fh[i_fh]), 4),
        "sh_bar": round(float(bar_sh[i_sh]), 4),
        "sh_cross": round(float(cross_sh[i_sh]), 4),
        "fh_er_db": round(
            float(10 * np.log10(max(cross_fh[i_fh], eps) / max(bar_fh[i_fh], eps))), 2
        ),
        "sh_er_db": round(
            float(10 * np.log10(max(bar_sh[i_sh], eps) / max(cross_sh[i_sh], eps))), 2
        ),
        "out_dir": str(out),
        "files": sorted(p.name for p in out.glob(f"{stem}*")),
    }
    save_json(out / f"{stem}_summary.json", summary)
    return summary
