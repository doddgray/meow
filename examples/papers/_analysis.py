"""Analysis + plotting of designed dichroic devices, runnable as slurm jobs.

This module turns a *designed* device (a dichroic beam-splitter coupler or a
FAQUAD wavelength filter) into the figures and data products the paper-based
examples care about:

- a **transmission spectrum** to each output port over a dense set of
  wavelengths (``<label>_spectrum.png`` + the raw arrays saved redundantly as
  ``<label>_spectrum.csv`` and ``<label>_spectrum.json``);
- **propagation plots** of the optical intensity ``|Ex|^2`` along the device at
  a few wavelengths around the cutoff (``<label>_propagation.png``), with the
  underlying per-cell mode fields saved to a single compressed HDF5 dataset
  (``<label>_fields.h5``) via :func:`meow.save_fields`;
- a **design figure** analogous to the paper reproductions (layout + index
  crossings for the dichroic coupler; layout + FAQUAD profiles for the filter);
- the device **GDS** (``<label>.gds``) and a scalar summary written redundantly
  as ``<label>_summary.csv`` and ``<label>_summary.json``.

The EME itself is **distributed as concurrent jobs**: :func:`submit_dichroic_run`
/ :func:`submit_faquad_run` rebuild the design from a *picklable* parameter dict
(``dichroic_designer.to_params`` / ``kwolek_designer.to_params``) and submit the
dense transmission spectrum as slice-group jobs (no fields) plus - when full
fields are saved - the propagation fields as single-cell jobs, returning a
picklable :class:`DichroicRun` / :class:`FaquadRun`. A later session reloads it
(see :func:`load_run`) and calls :meth:`~_Run.gather` to collect the distributed
results, assemble the spectra/fields and write the figures, GDS and data into
the run's output folder.

Wavelength controls (sensible defaults, overridable per call or via env vars):

- transmission spectrum: :func:`spectrum_wavelengths` - ``span`` (fractional
  half-width about the cutoff) and ``n`` points
  (``MEOW_SPECTRUM_SPAN`` / ``MEOW_SPECTRUM_NPTS``);
- propagation wavelengths: :func:`propagation_wavelengths` - ``span``/``n``
  (``MEOW_PROP_SPAN`` / ``MEOW_PROP_NPTS``) or an explicit comma-separated list
  ``MEOW_PROP_WLS``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import meow as mw
import meow.eme.propagation as prop
from examples.papers import _backends
from examples.papers import dichroic_designer as dd
from examples.papers import kwolek_designer as kd
from examples.papers.magden2018_dichroic import GAP_OUT, lateral_positions

# the generic, distributed analysis-run base class + its loader now live in the
# main library (meow.eme.run); the device-specific runs below subclass it.
AnalysisRun = mw.AnalysisRun
_Run = mw.AnalysisRun  # historical alias

# defaults for the wavelength controls
DEFAULT_SPECTRUM_SPAN = 0.06  # fractional half-width about the cutoff
DEFAULT_SPECTRUM_NPTS = 61
DEFAULT_PROP_SPAN = 0.04
DEFAULT_PROP_NPTS = 5


def _file_stem(label: str) -> str:
    """Filesystem-safe stem for output filenames derived from a run label."""
    return mw.safe_label(label)


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


def faquad_band(fh_wl: float, sh_wl: float, *, n: int | None = None) -> np.ndarray:
    """Dense, broad FAQUAD spectrum band [um]: ``0.8*SH .. 1.2*FH`` (>1 octave).

    ``n`` falls back to ``MEOW_SPECTRUM_NPTS`` then the module default.
    """
    n = int(os.environ.get("MEOW_SPECTRUM_NPTS", n or DEFAULT_SPECTRUM_NPTS))
    return np.linspace(0.8 * sh_wl, 1.2 * fh_wl, max(3, n))


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
    design: dd.DichroicDesign, num_cells: int = 128, res: float = 0.06
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


def plot_transmission_faquad_broad(
    wls: np.ndarray,
    bar: np.ndarray,
    cross: np.ndarray,
    fh_wl: float,
    sh_wl: float,
    path: Path,
    title: str,
) -> None:
    """Dense, broad-band bar/cross transmission spectrum of a FAQUAD filter.

    Spans more than an octave (from ~0.8*SH to ~1.2*FH); the SH should stay in
    the bar port and the FH transfer to the cross port. The FH and SH design
    wavelengths are marked.
    """
    plt = _use_agg()
    eps = 1e-6
    fig, ax = plt.subplots(figsize=(8, 4))
    bar_db = 10 * np.log10(np.clip(bar, eps, None))
    cross_db = 10 * np.log10(np.clip(cross, eps, None))
    ax.plot(wls * 1e3, bar_db, "C0", label="bar (input guide)")
    ax.plot(wls * 1e3, cross_db, "C3", label="cross (coupled guide)")
    ax.axvline(fh_wl * 1e3, color="C2", ls=":", lw=1.0, label=f"FH {fh_wl * 1e3:.0f}nm")
    ax.axvline(sh_wl * 1e3, color="C1", ls=":", lw=1.0, label=f"SH {sh_wl * 1e3:.0f}nm")
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("transmission [dB]")
    ax.set_ylim(max(-42.0, float(min(bar_db.min(), cross_db.min())) - 3), 2)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(visible=True)
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


# ==========================================================================
# field-saving control + port attribution from a precomputed mode mapping
# ==========================================================================
def save_fields_enabled(save_fields: bool | None = None) -> bool:  # noqa: FBT001
    """Whether to save full per-cell mode fields (single-cell job decomposition).

    When enabled, propagation/field analysis is possible and each cell is solved
    as its own job keeping the full fields; when disabled, only the field-free
    slice-group spectrum is computed. Falls back to the ``MEOW_SAVE_FIELDS`` env
    var (default on).
    """
    if save_fields is not None:
        return save_fields
    return bool(int(os.environ.get("MEOW_SAVE_FIELDS", "1")))


def _output_index_split(
    cells: list[mw.Cell],
    env: mw.Environment,
    num_modes: int,
    split: float,
    backend: Callable | None = None,
) -> tuple[list[int], list[int]]:
    """(below, above) output-mode indices, by energy centroid vs ``split``.

    Solved once on the final cell; the deterministic mode ordering lets the same
    index->port mapping be reused for every (nearby) wavelength of a spectrum,
    so the field-free slice-group S-matrices can be attributed to ports without
    re-solving the output modes at each wavelength.
    """
    backend = backend or mw.compute_modes
    out_modes = backend(
        mw.CrossSection.from_cell(cell=cells[-1], env=env), num_modes=num_modes
    )
    below = [i for i, m in enumerate(out_modes) if _centroid(m) < split]
    above = [i for i in range(len(out_modes)) if i not in below]
    return below, above


def _bar_input_index(
    cells: list[mw.Cell],
    env: mw.Environment,
    num_modes: int,
    backend: Callable | None = None,
) -> int:
    """Index of the bar (most negative centroid) input mode of the first cell."""
    backend = backend or mw.compute_modes
    in_modes = backend(
        mw.CrossSection.from_cell(cell=cells[0], env=env), num_modes=num_modes
    )
    return min(range(min(2, len(in_modes))), key=lambda k: _centroid(in_modes[k]))


def _attribute_by_index(
    s_matrix: Any,
    port_map: dict[str, int],
    below: list[int],
    above: list[int],
    in_port: str = "left@0",
) -> tuple[float, float]:
    """(below, above) output power for ``in_port`` using a fixed index mapping."""
    s = np.asarray(s_matrix)

    def power(i: int) -> float:
        return float(np.abs(s[port_map[f"right@{i}"], port_map[in_port]]) ** 2)

    return sum(power(i) for i in below), sum(power(i) for i in above)


# ==========================================================================
# distributed analysis runs: EME as concurrent jobs, assembled + plotted
# at gather time (a different python session). The generic base class
# (:class:`meow.AnalysisRun`) and its loaders now live in the main library;
# the device-specific runs below subclass it and fill in :meth:`gather`.
# ==========================================================================
@dataclass
class DichroicRun(_Run):
    """A distributed dichroic-coupler analysis run.

    The transmission spectrum is a single :class:`meow.ParallelEMESpectrumJobs`
    (slice-group decomposition, no fields); the optional propagation ``fields``
    are :class:`meow.ParallelFieldModeJobs` (single-cell decomposition keeping
    the full fields), one per propagation wavelength.
    """

    spectrum: mw.ParallelEMESpectrumJobs
    fields: dict[int, mw.ParallelFieldModeJobs]
    split: float

    def handles(self) -> list[Any]:
        return [self.spectrum, *self.fields.values()]

    def gather(self) -> dict:
        import gdsfactory as gf

        gf.gpdk.PDK.activate()
        out = Path(self.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        design = dd.design_from_params(**self.spec)
        num_modes = self.settings["num_modes"]
        cells = self.spectrum.cells
        spectrum_wls = np.asarray(self.spectrum.wls, dtype=float)

        backend = _backends.resolve_backend(self.settings.get("backend"))
        env_c = mw.Environment(wl=design.cutoff_wl, T=25.0)
        below, above = _output_index_split(
            cells, env_c, num_modes, self.split, backend
        )
        t_short, t_long = [], []
        for s_matrix, port_map in self.spectrum.result():
            b, a = _attribute_by_index(s_matrix, port_map, below, above)
            t_short.append(b)
            t_long.append(a)
        t_short, t_long = np.asarray(t_short), np.asarray(t_long)

        panels = _propagation_panels(
            self.fields, design.platform.core_thickness / 2.0,
            int(self.settings.get("num_z", 400)), input_kind="fundamental",
        )

        stem = self.stem
        design.component.write_gds(str(out / f"{stem}.gds"))
        plot_dichroic_design(
            design, out / f"{stem}_design.png",
            n_neff=int(self.settings.get("n_neff", 7)),
        )
        plot_transmission_dichroic(
            spectrum_wls, t_short, t_long, design.cutoff_wl,
            out / f"{stem}_spectrum.png",
            f"{self.label}: dichroic transmission spectrum",
        )
        if panels:
            plot_propagation(
                panels, out / f"{stem}_propagation.png",
                f"{self.label}: intensity propagation across the cutoff",
            )
        # dense per-cell mode fields -> one compressed HDF5 dataset per run;
        # the (less dense) transmission spectrum -> redundant CSV + JSON
        _save_field_datasets(self.fields, out / f"{stem}_fields.h5")
        mw.save_table(
            out / f"{stem}_spectrum",
            {"wavelength_um": spectrum_wls, "t_short": t_short, "t_long": t_long},
        )
        i_c = int(np.argmin(np.abs(spectrum_wls - design.cutoff_wl)))
        summary = {
            "label": self.label,
            "kind": "dichroic",
            "cutoff_nm": round(design.cutoff_wl * 1e3, 1),
            "w_a_nm": round(design.w_a * 1e3, 1),
            "gap_nm": round(design.gap * 1e3, 0),
            "length_um": round(design.total_length, 0),
            "short_pass_at_cutoff": round(float(t_short[i_c]), 4),
            "long_pass_at_cutoff": round(float(t_long[i_c]), 4),
            "saved_fields": bool(self.fields),
            "out_dir": str(out),
            "files": sorted(
                p.name
                for p in out.glob(f"{stem}*")
                if not p.name.endswith(("_summary.json", "_summary.csv"))
            ),
        }
        mw.save_summary(out / f"{stem}_summary", summary)
        return summary


@dataclass
class FaquadRun(_Run):
    """A distributed FAQUAD-filter analysis run.

    The dense, broad-band bar/cross transmission spectrum (from ~0.8*SH to
    ~1.2*FH, more than an octave) is a single :class:`meow.ParallelEMESpectrumJobs`
    over *dispersive* cells (so each slice-group job sweeps the whole band in one
    task); the optional propagation ``fields`` are single-cell
    :class:`meow.ParallelFieldModeJobs` at the FH and SH.
    """

    spectrum: mw.ParallelEMESpectrumJobs
    fields: dict[int, mw.ParallelFieldModeJobs]

    def handles(self) -> list[Any]:
        return [self.spectrum, *self.fields.values()]

    def gather(self) -> dict:
        import gdsfactory as gf

        gf.gpdk.PDK.activate()
        out = Path(self.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        design = kd.filter_from_params(**self.spec)
        num_modes = self.settings["num_modes"]
        backend = _backends.resolve_backend(self.settings.get("backend"))
        cells = self.spectrum.cells
        wls = np.asarray(self.spectrum.wls, dtype=float)

        # bar/cross port mapping solved once at the FH (stable across the band)
        env_ref = mw.Environment(wl=design.fh_wl, T=25.0)
        in_idx = _bar_input_index(cells, env_ref, num_modes, backend)
        below, above = _output_index_split(cells, env_ref, num_modes, 0.0, backend)
        bar, cross = [], []
        for s_matrix, port_map in self.spectrum.result():
            t_bar, t_cross = _attribute_by_index(
                s_matrix, port_map, below, above, in_port=f"left@{in_idx}"
            )
            bar.append(t_bar)
            cross.append(t_cross)
        bar, cross = np.asarray(bar), np.asarray(cross)

        panels = _propagation_panels(
            self.fields, design.platform.core_thickness / 2.0,
            int(self.settings.get("num_z", 400)), input_kind="bar",
        )

        stem = self.stem
        design.component.write_gds(str(out / f"{stem}.gds"))
        plot_faquad_design(design, out / f"{stem}_design.png")
        plot_transmission_faquad_broad(
            wls, bar, cross, design.fh_wl, design.sh_wl,
            out / f"{stem}_spectrum.png",
            f"{self.label}: FAQUAD broad-band transmission (bar input)",
        )
        if panels:
            plot_propagation(
                panels, out / f"{stem}_propagation.png",
                f"{self.label}: intensity propagation at FH / SH (bar input)",
                ylim=(-3, 3),
            )
        # dense per-cell mode fields -> one compressed HDF5 dataset per run;
        # the (less dense) broad-band bar/cross spectrum -> redundant CSV + JSON
        _save_field_datasets(self.fields, out / f"{stem}_fields.h5")
        mw.save_table(
            out / f"{stem}_spectrum",
            {"wavelength_um": wls, "bar": bar, "cross": cross},
        )
        eps = 1e-9
        i_fh = int(np.argmin(np.abs(wls - design.fh_wl)))
        i_sh = int(np.argmin(np.abs(wls - design.sh_wl)))
        fh_er = float(10 * np.log10(max(cross[i_fh], eps) / max(bar[i_fh], eps)))
        sh_er = float(10 * np.log10(max(bar[i_sh], eps) / max(cross[i_sh], eps)))
        summary = {
            "label": self.label,
            "kind": "faquad",
            "platform": design.platform.name,
            "fh_nm": round(design.fh_wl * 1e3, 0),
            "sh_nm": round(design.sh_wl * 1e3, 0),
            "length_um": round(design.total_length, 0),
            "band_nm": [round(float(wls[0]) * 1e3, 0), round(float(wls[-1]) * 1e3, 0)],
            "fh_cross": round(float(cross[i_fh]), 4),
            "fh_bar": round(float(bar[i_fh]), 4),
            "sh_bar": round(float(bar[i_sh]), 4),
            "sh_cross": round(float(cross[i_sh]), 4),
            "fh_er_db": round(fh_er, 2),
            "sh_er_db": round(sh_er, 2),
            "saved_fields": bool(self.fields),
            "out_dir": str(out),
            "files": sorted(
                p.name
                for p in out.glob(f"{stem}*")
                if not p.name.endswith(("_summary.json", "_summary.csv"))
            ),
        }
        mw.save_summary(out / f"{stem}_summary", summary)
        return summary


def _save_field_datasets(
    fields: dict[int, mw.ParallelFieldModeJobs], path: Path
) -> Path | None:
    """Save the per-cell mode fields of every propagation wavelength to one HDF5.

    Each wavelength's :class:`meow.ParallelFieldModeJobs` is bundled into an
    xarray dataset (complex ``Ex..Hz`` + ``neff``) and the wavelengths are
    concatenated along a ``wl`` dimension, then written as a single gzip-
    compressed netCDF (HDF5) file. Returns ``None`` when no fields were saved.
    """
    if not fields:
        return None
    import xarray as xr

    per_wl = []
    for wl_nm, handle in sorted(fields.items()):
        ds = handle.to_dataset(attrs={"wl_nm": int(wl_nm)})
        per_wl.append(ds.expand_dims(wl_nm=[int(wl_nm)]))
    combined = xr.concat(per_wl, dim="wl_nm") if len(per_wl) > 1 else per_wl[0]
    return mw.save_fields(combined, path)


def _propagation_panels(
    fields: dict[int, mw.ParallelFieldModeJobs],
    y_core: float,
    num_z: int,
    *,
    input_kind: str,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Collect propagation panels from the per-cell field-mode handles."""
    panels = []
    for wl_nm, handle in sorted(fields.items()):
        modes = handle.result()
        if input_kind == "bar":
            n_in = min(2, len(modes[0]))
            idx = min(range(n_in), key=lambda k: _centroid(modes[0][k]))
        else:
            idx = 0
        z = np.linspace(0.0, float(sum(c.length for c in handle.cells)), num_z)
        field, x = prop.propagate_modes(
            modes, handle.cells, excite_mode_l=idx, y=y_core, z=z
        )
        panels.append((f"{wl_nm} nm", np.abs(np.asarray(field)) ** 2, np.asarray(x), z))
    return panels


def load_run(path: str | Path) -> AnalysisRun:
    """Load an :class:`meow.AnalysisRun` saved by :meth:`~meow.AnalysisRun.save`."""
    obj = mw.load_run(path)
    if not isinstance(obj, AnalysisRun):
        msg = f"{path} is not an analysis run record."
        raise TypeError(msg)
    return obj


# ==========================================================================
# submission: distribute one design's EME as concurrent jobs
# ==========================================================================
def submit_dichroic_run(
    spec: dict,
    settings: dict,
    *,
    executor_factory: Callable[[str], Any],
    out_dir: str | Path,
    save_fields: bool | None = None,
) -> DichroicRun:
    """Submit one dichroic design's distributed EME and return a :class:`DichroicRun`.

    The dense transmission spectrum is submitted as slice-group jobs
    (:func:`meow.submit_s_matrix_spectrum`); when ``save_fields`` is on, the
    propagation fields are additionally submitted as single-cell mode jobs
    (:func:`meow.submit_cell_modes`), one per propagation wavelength.
    ``executor_factory(name)`` builds a :func:`meow.slurm_executor` per job group
    (so submitit logs land in distinct subfolders).
    """
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    save_fields = save_fields_enabled(save_fields)
    backend = _backends.resolve_backend(settings.get("backend"))
    design = dd.design_from_params(**spec)
    cells = dichroic_device_cells(
        design, settings["num_cells"], settings["device_res"]
    )
    env = mw.Environment(wl=design.cutoff_wl, T=25.0)
    spectrum = mw.submit_s_matrix_spectrum(
        cells, env,
        executor=executor_factory("spectrum"),
        wls=np.asarray(settings["spectrum_wls"], dtype=float),
        num_modes=settings["num_modes"],
        compute_modes=backend,
    )
    fields: dict[int, mw.ParallelFieldModeJobs] = {}
    if save_fields:
        for wl in np.asarray(settings["prop_wls"], dtype=float):
            wl_nm = round(float(wl) * 1e3)
            fields[wl_nm] = mw.submit_cell_modes(
                cells, mw.Environment(wl=float(wl), T=25.0),
                executor=executor_factory(f"fields_{wl_nm}nm"),
                num_modes=settings["num_modes"],
                compute_modes=backend,
            )
    return DichroicRun(
        spec=spec, settings=settings, label=settings["label"],
        out_dir=str(out_dir), save_fields=save_fields,
        spectrum=spectrum, fields=fields,
        split=dichroic_short_pass_split(design),
    )


def submit_faquad_run(
    spec: dict,
    settings: dict,
    *,
    executor_factory: Callable[[str], Any],
    out_dir: str | Path,
    save_fields: bool | None = None,
) -> FaquadRun:
    """Submit one FAQUAD design's distributed EME and return a :class:`FaquadRun`.

    The dense, broad-band (``0.8*SH .. 1.2*FH``) bar/cross spectrum is submitted
    as slice-group jobs over *dispersive* cells (:func:`meow.submit_s_matrix_spectrum`),
    so each job sweeps the whole band within a single task; when ``save_fields``
    is on, the FH/SH propagation fields are submitted as single-cell mode jobs.
    """
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    save_fields = save_fields_enabled(save_fields)
    backend = _backends.resolve_backend(settings.get("backend"))
    design = kd.filter_from_params(**spec)
    num_modes = settings["num_modes"]
    spectrum_wls = np.asarray(settings["spectrum_wls"], dtype=float)
    cells = kd.device_cells_dispersive(
        design, spectrum_wls,
        num_cells=settings["num_cells"], res=settings["device_res"],
    )
    spectrum = mw.submit_s_matrix_spectrum(
        cells, mw.Environment(wl=design.fh_wl, T=25.0),
        executor=executor_factory("spectrum"),
        wls=spectrum_wls, num_modes=num_modes, compute_modes=backend,
    )
    fields: dict[int, mw.ParallelFieldModeJobs] = {}
    if save_fields:
        for wl in np.asarray(settings["prop_wls"], dtype=float):
            wl_nm = round(float(wl) * 1e3)
            fields[wl_nm] = mw.submit_cell_modes(
                cells, mw.Environment(wl=float(wl), T=25.0),
                executor=executor_factory(f"fields_{wl_nm}nm"),
                num_modes=num_modes, compute_modes=backend,
            )
    return FaquadRun(
        spec=spec, settings=settings, label=settings["label"],
        out_dir=str(out_dir), save_fields=save_fields,
        spectrum=spectrum, fields=fields,
    )
