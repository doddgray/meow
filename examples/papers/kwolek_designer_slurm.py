"""Slurm-parallelized FAQUAD filter designer: remote EME + concurrent workflows.

This is the slurm-cluster version of :mod:`examples.papers.kwolek_designer`. It
designs the generalized FAQUAD wavelength filters of that module - X-cut TFLN
and TFLT, 300/400/500/600 nm films, for the 1550/775, 1350/675 and 1060/530 nm
FH/SH pairs - and then runs a **full-device EME** of each design at both the FH
and the SH with the parallel slice-group engine
(:func:`meow.compute_s_matrix_parallel`), distributing the per-slice mode solves
as independent jobs. Pass a :func:`meow.slurm_executor` and those jobs run on a
slurm cluster; the design workflows themselves can be run either

- **blocking** (:func:`run_blocking`): each design's two (FH + SH) EMEs are
  parallelized across the cluster, processed one design after another; or
- **concurrent / async** (:func:`run_concurrent`): every design's EMEs are
  submitted at once and awaited together with :func:`asyncio.gather`, so all the
  (material x thickness x wavelength-pair) design workflows are in flight on the
  cluster simultaneously.

**Analysis runs (plots + GDS).** The default workflow (:func:`submit_runs` /
:func:`gather_runs`, used by ``main`` and the ``submit``/``gather``
subcommands) submits *one slurm job per design* that runs the whole analysis -
FH/SH extinction-ratio and loss spectra, intensity-propagation plots at the FH
and SH, a layout + FAQUAD-profile design figure, and the device GDS - writing
every figure, the GDS and the raw data into a fresh **timestamped subfolder**
of ``MEOW_SLURM_FOLDER`` (one per submitted job). Wavelength controls follow
the ``MEOW_SPECTRUM_*`` / ``MEOW_PROP_*`` env vars (see
:mod:`examples.papers._analysis`).

**Reloading results in a later session.** Because submitit persists every job
in its ``folder``, submission and collection can happen in different processes.
:func:`gather_runs` - run in a *different* session pointing at the same
``MEOW_SLURM_FOLDER`` - reloads the persisted :class:`_slurm.SavedRun` records,
reattaches to the still-running jobs and returns their summaries. A lighter
S-matrix-only path (:func:`submit_designs` / :func:`gather_results`, built on
:class:`meow.ParallelEMEJobs`, writing one ``.eme.pkl`` per (design, harmonic))
is kept for just the FH/SH figures of merit.

See ``examples/papers/README.md`` ("Running EME on a slurm cluster") for how to
configure the local and remote environments.

Run the in-session demo (jobs as local subprocesses) with::

    python -m examples.papers.kwolek_designer_slurm

or split submission and collection across two sessions::

    python -m examples.papers.kwolek_designer_slurm submit   # session A
    python -m examples.papers.kwolek_designer_slurm gather    # session B (later)

On a login node of a slurm cluster, set ``MEOW_SLURM_CLUSTER=slurm`` (and
``MEOW_SLURM_PARTITION``) and a shared ``MEOW_SLURM_FOLDER`` first.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import gdsfactory as gf
import numpy as np

import meow as mw
from examples.papers import _analysis, _slurm
from examples.papers.kwolek_designer import (
    WAVELENGTH_PAIRS,
    FaquadFilterDesign,
    design_faquad_filter,
    device_cells,
    platform_matrix,
    to_params,
)

FAST = bool(int(os.environ.get("MEOW_EXAMPLE_FAST", "0")))
JOB_FOLDER = Path(os.environ.get("MEOW_SLURM_FOLDER", "meow_kwolek_jobs"))


# --------------------------------------------------------------------------
# bar/cross power attribution for a FAQUAD device
# --------------------------------------------------------------------------
def _centroid(mode: mw.Mode) -> float:
    d = np.abs(mode.Ex) ** 2
    return float(np.sum(mode.cs.mesh.Xx * d) / np.sum(d))


def _bar_cross(
    cells: list[mw.Cell],
    env: mw.Environment,
    s_matrix: Any,
    port_map: dict[str, int],
    num_modes: int,
) -> tuple[float, float]:
    """(bar, cross) power for the fundamental input launched in waveguide B.

    Waveguide B (the bar/input guide) sits at negative x and waveguide A (the
    cross/output guide) at positive x. The input is the fundamental mode of the
    input (separated) cross-section localized in B; the output power is sorted
    into bar/cross by the lateral energy centroid of each output mode. The
    parallel engine returns no field data, so the input/output modes are solved
    once on the end cells.
    """
    s = np.asarray(s_matrix)
    modes_in = mw.compute_modes(
        mw.CrossSection.from_cell(cell=cells[0], env=env), num_modes=num_modes
    )
    modes_out = mw.compute_modes(
        mw.CrossSection.from_cell(cell=cells[-1], env=env), num_modes=num_modes
    )
    in_idx = min(range(min(2, len(modes_in))), key=lambda i: _centroid(modes_in[i]))
    t_bar = t_cross = 0.0
    for i, mode in enumerate(modes_out):
        amp = s[port_map[f"right@{i}"], port_map[f"left@{in_idx}"]]
        power = float(np.abs(amp) ** 2)
        if _centroid(mode) < 0:
            t_bar += power
        else:
            t_cross += power
    return t_bar, t_cross


# --------------------------------------------------------------------------
# EME (FH + SH) of a designed device through the parallel / slurm engine
# --------------------------------------------------------------------------
def _eme_at(
    design: FaquadFilterDesign,
    wl: float,
    *,
    executor: Any | None,
    num_cells: int,
    num_modes: int,
    res: float,
    max_workers: int | None,
) -> tuple[float, float]:
    cells = device_cells(design, wl, num_cells=num_cells, res=res)
    env = mw.Environment(wl=wl, T=25.0)
    s, pm = mw.compute_s_matrix_parallel(
        cells, env, num_modes=num_modes, executor=executor, max_workers=max_workers
    )
    return _bar_cross(cells, env, s, pm, num_modes)


async def _aeme_at(
    design: FaquadFilterDesign,
    wl: float,
    *,
    executor: Any | None,
    num_cells: int,
    num_modes: int,
    res: float,
    max_workers: int | None,
) -> tuple[float, float]:
    cells = device_cells(design, wl, num_cells=num_cells, res=res)
    env = mw.Environment(wl=wl, T=25.0)
    s, pm = await mw.acompute_s_matrix_parallel(
        cells, env, num_modes=num_modes, executor=executor, max_workers=max_workers
    )
    return _bar_cross(cells, env, s, pm, num_modes)


def eme_filter(
    design: FaquadFilterDesign,
    *,
    executor: Any | None = None,
    num_cells: int = 48,
    num_modes: int = 4,
    res: float = 0.05,
    max_workers: int | None = None,
) -> dict[str, float]:
    """Blocking full-device EME of one design at the FH and the SH.

    Returns the FH cross transmission, the SH bar transmission, and the FH/SH
    extinction ratios - the dichroic figures of merit.
    """
    kw = {
        "executor": executor,
        "num_cells": num_cells,
        "num_modes": num_modes,
        "res": res,
        "max_workers": max_workers,
    }
    fh_bar, fh_cross = _eme_at(design, design.fh_wl, **kw)
    sh_bar, sh_cross = _eme_at(design, design.sh_wl, **kw)
    return _figures_of_merit(fh_bar, fh_cross, sh_bar, sh_cross)


async def aeme_filter(
    design: FaquadFilterDesign,
    *,
    executor: Any | None = None,
    num_cells: int = 48,
    num_modes: int = 4,
    res: float = 0.05,
    max_workers: int | None = None,
) -> dict[str, float]:
    """Async full-device EME of one design at the FH and the SH (both in flight)."""
    kw = {
        "executor": executor,
        "num_cells": num_cells,
        "num_modes": num_modes,
        "res": res,
        "max_workers": max_workers,
    }
    (fh_bar, fh_cross), (sh_bar, sh_cross) = await asyncio.gather(
        _aeme_at(design, design.fh_wl, **kw), _aeme_at(design, design.sh_wl, **kw)
    )
    return _figures_of_merit(fh_bar, fh_cross, sh_bar, sh_cross)


def _figures_of_merit(
    fh_bar: float, fh_cross: float, sh_bar: float, sh_cross: float
) -> dict[str, float]:
    eps = 1e-9
    fh_er = float(10 * np.log10(max(fh_cross, eps) / max(fh_bar, eps)))
    sh_er = float(10 * np.log10(max(sh_bar, eps) / max(sh_cross, eps)))
    return {
        "fh_cross": round(float(fh_cross), 4),
        "fh_bar": round(float(fh_bar), 4),
        "sh_bar": round(float(sh_bar), 4),
        "sh_cross": round(float(sh_cross), 4),
        "fh_er_db": round(fh_er, 2),
        "sh_er_db": round(sh_er, 2),
    }


# --------------------------------------------------------------------------
# workflows over the whole design matrix
# --------------------------------------------------------------------------
def run_blocking(
    designs: list[FaquadFilterDesign], *, executor: Any | None = None, **eme_kwargs: Any
) -> dict[str, dict[str, float]]:
    """Run each design's (cluster-parallelized) FH+SH EME one after another."""
    return {
        _key(d): eme_filter(d, executor=executor, **eme_kwargs) for d in designs
    }


async def run_concurrent(
    designs: list[FaquadFilterDesign], *, executor: Any | None = None, **eme_kwargs: Any
) -> dict[str, dict[str, float]]:
    """Run every design's EME concurrently (all jobs in flight together)."""
    results = await asyncio.gather(
        *(aeme_filter(d, executor=executor, **eme_kwargs) for d in designs)
    )
    return {_key(d): r for d, r in zip(designs, results, strict=True)}


def _key(d: FaquadFilterDesign) -> str:
    return f"{d.platform.name}/{d.fh_wl * 1e3:.0f}-{d.sh_wl * 1e3:.0f}nm"


# --------------------------------------------------------------------------
# multi-session workflow: submit now, gather (reload) in a later session
# --------------------------------------------------------------------------
def submit_designs(
    designs: list[FaquadFilterDesign],
    *,
    executor: Any,
    folder: Path | str = JOB_FOLDER,
    num_cells: int = 48,
    num_modes: int = 4,
    res: float = 0.05,
) -> list[_slurm.SavedEME]:
    """Submit every design's FH *and* SH EME to the cluster *without waiting*.

    Two persisted records (``...|fh`` and ``...|sh``) are written into
    ``folder`` per design through :func:`meow.submit_s_matrix_parallel`. The
    submitting session can then exit; collect the results in any later session
    with :func:`gather_results` pointed at the same ``folder`` (use a
    :func:`meow.slurm_executor` with ``cluster="slurm"`` so the jobs outlive
    this process).
    """
    folder = Path(folder)
    records: list[_slurm.SavedEME] = []
    for design in designs:
        key = _key(design)
        for role, wl in (("fh", design.fh_wl), ("sh", design.sh_wl)):
            cells = device_cells(design, wl, num_cells=num_cells, res=res)
            env = mw.Environment(wl=wl, T=25.0)
            records.append(
                _slurm.submit_eme(
                    f"{key}|{role}",
                    cells,
                    env,
                    executor=executor,
                    num_modes=num_modes,
                    folder=folder,
                    meta={"key": key, "role": role},
                )
            )
    return records


def gather_results(
    folder: Path | str = JOB_FOLDER,
) -> dict[str, dict[str, float]]:
    """Collect EME results submitted by :func:`submit_designs` from ``folder``.

    Runs in a **different python session** from the submitter: it reloads every
    persisted :class:`_slurm.SavedEME` record (reattaching to the running
    submitit jobs), cascades each (design, harmonic) S-matrix, attributes the
    bar/cross powers and recombines the FH and SH halves of each design into
    its dichroic figures of merit.
    """
    bar_cross: dict[str, dict[str, tuple[float, float]]] = {}
    for rec in _slurm.load_records(folder):
        s, pm = rec.jobs.result()
        powers = _bar_cross(rec.jobs.cells, rec.jobs.env, s, pm, rec.num_modes)
        bar_cross.setdefault(rec.meta["key"], {})[rec.meta["role"]] = powers
    out: dict[str, dict[str, float]] = {}
    for key, halves in bar_cross.items():
        if {"fh", "sh"} <= halves.keys():
            (fh_bar, fh_cross), (sh_bar, sh_cross) = halves["fh"], halves["sh"]
            out[key] = _figures_of_merit(fh_bar, fh_cross, sh_bar, sh_cross)
    return out


# --------------------------------------------------------------------------
# rich analysis runs: FH/SH spectra + propagation plots + GDS, one job/design
# --------------------------------------------------------------------------
def analysis_settings(
    design: FaquadFilterDesign,
    *,
    num_cells: int,
    num_modes: int,
    device_res: float,
) -> dict[str, Any]:
    """Per-design analysis settings (FH/SH spectra + propagation wavelengths).

    Bounds/counts default to bands around the FH and SH and honour the
    ``MEOW_SPECTRUM_*`` / ``MEOW_PROP_*`` env vars (see
    :mod:`examples.papers._analysis`).
    """
    fh_wls = _analysis.spectrum_wavelengths(
        design.fh_wl, span=0.03, n=5 if FAST else 11
    )
    sh_wls = _analysis.spectrum_wavelengths(
        design.sh_wl, span=0.03, n=3 if FAST else 7
    )
    prop_fh = _analysis.propagation_wavelengths(design.fh_wl, span=0.02, n=3)
    prop_sh = _analysis.propagation_wavelengths(
        design.sh_wl, span=0.02, n=3 if FAST else 5
    )
    explicit = os.environ.get("MEOW_PROP_WLS")
    prop_wls = (
        np.array([float(x) for x in explicit.split(",") if x.strip()])
        if explicit
        else np.concatenate([prop_fh, prop_sh])
    )
    return {
        "num_cells": num_cells,
        "num_modes": num_modes,
        "device_res": device_res,
        "fh_wls": fh_wls,
        "sh_wls": sh_wls,
        "prop_wls": prop_wls,
        "num_z": 200 if FAST else 600,
    }


def submit_runs(
    designs: list[FaquadFilterDesign],
    *,
    folder: Path | str = JOB_FOLDER,
    executor_factory: Any | None = None,
    num_cells: int = 48,
    num_modes: int = 4,
    device_res: float = 0.05,
) -> list[_slurm.SavedRun]:
    """Submit a full analysis (FH/SH spectra + propagation + plots + GDS)/design.

    Each design becomes one slurm job (:func:`_analysis.analyze_faquad`) writing
    all its figures, GDS and data into a fresh timestamped subfolder of
    ``folder``; returns immediately with persisted :class:`_slurm.SavedRun`
    handles. Collect later with :func:`gather_runs`.
    """
    if executor_factory is None:
        def executor_factory(sub: Path) -> Any:
            return make_executor(folder=sub)

    return [
        _slurm.submit_run(
            _analysis.analyze_faquad,
            to_params(design),
            analysis_settings(
                design,
                num_cells=num_cells,
                num_modes=num_modes,
                device_res=device_res,
            ),
            executor_factory=executor_factory,
            folder=folder,
            label=_key(design),
        )
        for design in designs
    ]


def gather_runs(folder: Path | str = JOB_FOLDER) -> dict[str, dict]:
    """Reload and collect the analysis summaries from ``folder`` (any session)."""
    return {r.label: r.result() for r in _slurm.load_runs(folder)}


async def agather_runs(folder: Path | str = JOB_FOLDER) -> dict[str, dict]:
    """Async :func:`gather_runs` (awaits all the analysis jobs together)."""
    runs = _slurm.load_runs(folder)
    summaries = await asyncio.gather(*(r.aresult() for r in runs))
    return {r.label: s for r, s in zip(runs, summaries, strict=True)}


def make_executor(
    folder: Path | str = JOB_FOLDER,
    cluster: str | None = None,
    *,
    timeout_min: int = 60,
    cpus_per_task: int = 2,
    mem_gb: float | None = None,
    slurm_partition: str | None = None,
) -> Any:
    """A :func:`meow.slurm_executor` for the EME jobs.

    ``cluster`` selects the backend: ``"slurm"`` (require a cluster), ``"local"``
    (local subprocesses, the default here), ``"debug"`` (in-process), or ``None``
    (auto: slurm if available, else local). ``MEOW_SLURM_CLUSTER`` and
    ``MEOW_SLURM_PARTITION`` override the demo defaults.
    """
    cluster = cluster or os.environ.get("MEOW_SLURM_CLUSTER", "local")
    slurm_partition = slurm_partition or os.environ.get("MEOW_SLURM_PARTITION")
    return mw.slurm_executor(
        folder=str(folder),
        cluster=cluster,
        timeout_min=timeout_min,
        cpus_per_task=cpus_per_task,
        mem_gb=mem_gb,
        slurm_partition=slurm_partition,
    )


def design_matrix(
    res: float = 0.04, pairs: list[tuple[float, float]] | None = None
) -> list[FaquadFilterDesign]:
    """Design a FAQUAD filter for every (material, thickness, FH/SH) combination."""
    pairs = pairs or WAVELENGTH_PAIRS
    return [
        design_faquad_filter(platform, fh, sh, res=res)
        for platform in platform_matrix()
        for fh, sh in pairs
    ]


def _demo_designs() -> tuple[list[FaquadFilterDesign], dict[str, Any]]:
    """The designs + EME settings shared by every entry point of this demo."""
    gf.gpdk.PDK.activate()
    res = 0.06 if FAST else 0.04
    if FAST:
        from examples.papers.kwolek_designer import tfln_platform

        designs = [
            design_faquad_filter(tfln_platform(0.30), *WAVELENGTH_PAIRS[0], res=res)
        ]
    else:
        designs = design_matrix(res=res)
    eme_kwargs = {
        "num_cells": 16 if FAST else 48,
        "num_modes": 2 if FAST else 4,
        "res": 0.07 if FAST else 0.05,
    }
    return designs, eme_kwargs


def _submit_runs() -> list[_slurm.SavedRun]:
    designs, eme_kwargs = _demo_designs()
    return submit_runs(
        designs,
        folder=JOB_FOLDER,
        num_cells=eme_kwargs["num_cells"],
        num_modes=eme_kwargs["num_modes"],
        device_res=eme_kwargs["res"],
    )


def main() -> dict[str, object]:
    """Design the matrix and run each design's full analysis on the cluster.

    Submits one analysis job per design (FH/SH spectra + propagation + plots +
    GDS, each into its own timestamped subfolder of ``MEOW_SLURM_FOLDER``) and
    awaits them. Set ``MEOW_SLURM_CLUSTER=slurm`` (and ``MEOW_SLURM_PARTITION``)
    to dispatch to a cluster instead of local subprocesses.
    """
    _submit_runs()
    return {"gathered": asyncio.run(agather_runs(JOB_FOLDER))}


def submit_main() -> dict[str, object]:
    """Session A: submit every design's analysis to the cluster and return."""
    records = _submit_runs()
    return {
        "submitted": {
            r.label: {"job_id": r.job_id, "out_dir": r.out_dir} for r in records
        },
        "folder": str(JOB_FOLDER),
        "next": "run 'gather' in a later session with the same MEOW_SLURM_FOLDER",
    }


def gather_main() -> dict[str, object]:
    """Session B (later): reload and collect the run summaries (any session)."""
    return {"gathered": gather_runs(JOB_FOLDER)}


if __name__ == "__main__":
    _slurm.cli_main(
        "examples.papers.kwolek_designer_slurm",
        {"run": main, "submit": submit_main, "gather": gather_main},
    )
