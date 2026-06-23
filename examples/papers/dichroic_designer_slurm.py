"""Slurm-parallelized dichroic designer: remote EME + concurrent workflows.

This is a slurm-cluster version of the dichroic designer examples. It designs
the Si3N4 splitters of ``dichroic_designer_si3n4`` and then runs a **full-device
EME** of each design with the parallel slice-group engine
(:func:`meow.compute_s_matrix_parallel`), distributing the per-slice mode solves
as independent jobs. Pass a :func:`meow.slurm_executor` and those jobs run on a
slurm cluster; the design workflows themselves can be run either

- **blocking** (:func:`run_blocking`): each device's EME is parallelized across
  the cluster, and the designs are processed one after another; or
- **concurrent / async** (:func:`run_concurrent`): every device's EME is
  submitted at once and awaited together with :func:`asyncio.gather`, so all the
  design workflows' jobs are in flight on the cluster simultaneously.

**Distributed analysis runs (plots + GDS).** The default workflow
(:func:`submit_runs` / :func:`gather_runs`, used by ``main`` and the
``submit``/``gather`` subcommands) breaks each design's EME into **subsets of
cells run concurrently as separate slurm jobs**: the dense short-/long-pass
transmission spectrum is distributed as overlapping slice-group jobs (as in
``examples/parallel_eme_spectrum.py``), and - when full fields are saved (the
default, ``save_fields`` / ``MEOW_SAVE_FIELDS``) - the propagation fields are
distributed as single-cell jobs that keep each cell's full mode fields. The
:func:`gather_runs` step (a later session) reattaches to those jobs, assembles
the spectrum + propagation, and writes the spectrum/propagation/design figures,
the device GDS and the raw data into a fresh **timestamped subfolder** of
``MEOW_SLURM_FOLDER`` (one per design). Wavelength controls: env vars
``MEOW_SPECTRUM_SPAN``/``MEOW_SPECTRUM_NPTS`` and
``MEOW_PROP_SPAN``/``MEOW_PROP_NPTS``/``MEOW_PROP_WLS`` (see
:mod:`examples.papers._analysis`); the resolution preset (mesh / modes / cells)
follows ``MEOW_EXAMPLE_RES`` in ``{low, medium, high}``.

**Reloading results in a later session.** Because submitit persists every job
in its ``folder``, submission and collection do not have to happen in the same
process. :func:`gather_runs` reloads the persisted run records
(``<run_dir>/run.pkl``) and reattaches to the still-running jobs. A lighter
S-matrix-only multi-session path (:func:`submit_designs` / :func:`gather_results`,
built on :class:`meow.ParallelEMEJobs`) is also kept for just the port powers.

See ``examples/papers/README.md`` ("Running EME on a slurm cluster") for how to
configure the local and remote environments for blocking, async and
multi-session execution.

Run the in-session demo (jobs as local subprocesses) with::

    python -m examples.papers.dichroic_designer_slurm

or split submission and collection across two sessions::

    python -m examples.papers.dichroic_designer_slurm submit   # session A
    python -m examples.papers.dichroic_designer_slurm gather    # session B (later)

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
from examples.papers import _analysis, _backends, _resolution, _slurm
from examples.papers.dichroic_designer import (
    DichroicDesign,
    Platform,
    design_dichroic,
    to_params,
)
from examples.papers.dichroic_designer_si3n4 import (
    TARGET_CUTOFFS,
    WGB_DESIGN,
    si3n4_platform,
)

pick = _resolution.pick
JOB_FOLDER = Path(os.environ.get("MEOW_SLURM_FOLDER", "meow_dichroic_jobs"))

# EME cell + port-attribution helpers live in ``_analysis`` (shared with the
# single-design and thickness-sweep examples); re-exported here under their
# historical names.
device_mesh = _analysis.dichroic_device_mesh
device_cells = _analysis.dichroic_device_cells
short_pass_split = _analysis.dichroic_short_pass_split
port_powers_from_s = _analysis.dichroic_port_powers_from_s


def _port_powers(
    design: DichroicDesign,
    cells: list[mw.Cell],
    env: mw.Environment,
    s_matrix: Any,
    port_map: dict[str, int],
    num_modes: int,
) -> tuple[float, float]:
    """(short-pass, long-pass) power for a design (see :func:`port_powers_from_s`)."""
    return port_powers_from_s(
        cells, env, s_matrix, port_map, num_modes, short_pass_split(design)
    )


# --------------------------------------------------------------------------
# EME through the parallel / slurm engine
# --------------------------------------------------------------------------
def eme_ports(
    design: DichroicDesign,
    *,
    executor: Any | None = None,
    num_cells: int = 128,
    num_modes: int = 8,
    res: float = 0.06,
    max_workers: int | None = None,
) -> tuple[float, float]:
    """Blocking full-device EME via the parallel slice-group engine.

    The per-slice mode solves run as independent jobs on ``executor`` (a
    :func:`meow.slurm_executor`); with ``executor=None`` they run as local
    subprocesses.
    """
    cells = device_cells(design, num_cells, res)
    env = mw.Environment(wl=design.cutoff_wl, T=25.0)
    s, pm = mw.compute_s_matrix_parallel(
        cells, env, num_modes=num_modes, executor=executor, max_workers=max_workers
    )
    return _port_powers(design, cells, env, s, pm, num_modes)


async def aeme_ports(
    design: DichroicDesign,
    *,
    executor: Any | None = None,
    num_cells: int = 128,
    num_modes: int = 8,
    res: float = 0.06,
    max_workers: int | None = None,
) -> tuple[float, float]:
    """Async full-device EME (awaits the jobs without blocking the event loop)."""
    cells = device_cells(design, num_cells, res)
    env = mw.Environment(wl=design.cutoff_wl, T=25.0)
    s, pm = await mw.acompute_s_matrix_parallel(
        cells, env, num_modes=num_modes, executor=executor, max_workers=max_workers
    )
    return _port_powers(design, cells, env, s, pm, num_modes)


# --------------------------------------------------------------------------
# workflows
# --------------------------------------------------------------------------
def run_blocking(
    designs: list[DichroicDesign], *, executor: Any | None = None, **eme_kwargs: Any
) -> dict[str, tuple[float, float]]:
    """Run each design's (cluster-parallelized) EME one after another."""
    return {
        f"{d.cutoff_wl * 1e3:.0f}nm": eme_ports(d, executor=executor, **eme_kwargs)
        for d in designs
    }


async def run_concurrent(
    designs: list[DichroicDesign], *, executor: Any | None = None, **eme_kwargs: Any
) -> dict[str, tuple[float, float]]:
    """Run every design's EME concurrently (all jobs in flight together)."""
    results = await asyncio.gather(
        *(aeme_ports(d, executor=executor, **eme_kwargs) for d in designs)
    )
    return {
        f"{d.cutoff_wl * 1e3:.0f}nm": r for d, r in zip(designs, results, strict=True)
    }


# --------------------------------------------------------------------------
# multi-session workflow: submit now, gather (reload) in a later session
# --------------------------------------------------------------------------
def submit_designs(
    designs: list[DichroicDesign],
    *,
    executor: Any,
    folder: Path | str = JOB_FOLDER,
    num_cells: int = 128,
    num_modes: int = 8,
    res: float = 0.06,
) -> list[_slurm.SavedEME]:
    """Submit every design's full-device EME to the cluster *without waiting*.

    Each design's per-slice jobs are submitted through
    :func:`meow.submit_s_matrix_parallel` and a small picklable record
    (``<cutoff>.eme.pkl``) is written into ``folder``. This returns as soon as
    the jobs are queued, so the submitting session can exit; collect the
    results later (in any session) with :func:`gather_results` pointed at the
    same ``folder``. Use a :func:`meow.slurm_executor` (``cluster="slurm"``)
    so the jobs outlive this process.
    """
    folder = Path(folder)
    records = []
    for design in designs:
        cells = device_cells(design, num_cells, res)
        env = mw.Environment(wl=design.cutoff_wl, T=25.0)
        records.append(
            _slurm.submit_eme(
                f"{design.cutoff_wl * 1e3:.0f}nm",
                cells,
                env,
                executor=executor,
                num_modes=num_modes,
                folder=folder,
                meta={"split": short_pass_split(design)},
            )
        )
    return records


def gather_results(
    folder: Path | str = JOB_FOLDER,
) -> dict[str, tuple[float, float]]:
    """Collect EME results submitted by :func:`submit_designs` from ``folder``.

    Designed to run in a **different python session** from the one that
    submitted the jobs: it reloads each persisted :class:`_slurm.SavedEME`
    record (which reattaches to the still-running submitit jobs), blocks until
    each job has finished, cascades the device S-matrix and attributes the
    short-/long-pass powers - without needing the original design objects.
    """
    out: dict[str, tuple[float, float]] = {}
    for rec in _slurm.load_records(folder):
        s, pm = rec.jobs.result()
        out[rec.label] = port_powers_from_s(
            rec.jobs.cells, rec.jobs.env, s, pm, rec.num_modes, rec.meta["split"]
        )
    return out


# --------------------------------------------------------------------------
# rich analysis runs: spectra + propagation plots + GDS, one slurm job/design
# --------------------------------------------------------------------------
def analysis_settings(
    design: DichroicDesign,
    *,
    num_cells: int,
    num_modes: int,
    device_res: float,
) -> dict[str, Any]:
    """Per-design analysis settings (dense spectrum + propagation wavelengths).

    Wavelength bounds/counts default to sensible bands around the design's
    cutoff and honour the ``MEOW_SPECTRUM_*`` / ``MEOW_PROP_*`` env vars (see
    :mod:`examples.papers._analysis`).
    """
    return {
        "num_cells": num_cells,
        "num_modes": num_modes,
        "device_res": device_res,
        "backend": _backends.backend_name(),
        "spectrum_wls": _analysis.spectrum_wavelengths(
            design.cutoff_wl, n=pick(low=5, medium=None, high=121)
        ),
        "prop_wls": _analysis.propagation_wavelengths(
            design.cutoff_wl, n=pick(low=3, medium=5, high=7)
        ),
        "n_neff": pick(low=4, medium=9, high=15),
        "num_z": pick(low=200, medium=600, high=1000),
    }


def submit_runs(
    designs: list[DichroicDesign],
    *,
    folder: Path | str = JOB_FOLDER,
    executor_factory: Any | None = None,
    num_cells: int = 128,
    num_modes: int = 8,
    device_res: float = 0.05,
    save_fields: bool | None = None,
) -> list[Any]:
    """Submit each design's *distributed* EME analysis; return the run records.

    For each design the dense transmission spectrum is submitted as concurrent
    slice-group jobs and (when ``save_fields`` is on, the default) the
    propagation fields as concurrent single-cell jobs - see
    :func:`_analysis.submit_dichroic_run`. Each design's jobs go into a fresh
    timestamped subfolder of ``folder`` with a persisted ``run.pkl``; collect
    the assembled spectra/propagation plots/GDS later (any session) with
    :func:`gather_runs`. ``executor_factory(dir)`` builds the per-job executor
    (default: :func:`make_executor`).
    """
    if executor_factory is None:
        def executor_factory(sub: Path) -> Any:
            return make_executor(folder=sub)

    return [
        _slurm.start_run(
            _analysis.submit_dichroic_run,
            to_params(design),
            analysis_settings(
                design,
                num_cells=num_cells,
                num_modes=num_modes,
                device_res=device_res,
            ),
            folder=folder,
            label=f"{design.cutoff_wl * 1e3:.0f}nm",
            executor_factory=executor_factory,
            save_fields=save_fields,
        )
        for design in designs
    ]


def gather_runs(folder: Path | str = JOB_FOLDER) -> dict[str, dict]:
    """Reload each run, collect its distributed EME jobs and write the outputs.

    Runs in any later session: reattaches to the persisted jobs, assembles the
    transmission spectrum (and propagation, if fields were saved), and writes
    the spectrum/propagation/design figures, the GDS and the data into each
    run's timestamped subfolder, returning the summary dicts.
    """
    return {r.label: r.gather() for r in _slurm.load_runs(folder)}


async def agather_runs(folder: Path | str = JOB_FOLDER) -> dict[str, dict]:
    """Async :func:`gather_runs` (assembles + plots all the runs together)."""
    runs = _slurm.load_runs(folder)
    summaries = await asyncio.gather(*(r.agather() for r in runs))
    return {r.label: s for r, s in zip(runs, summaries, strict=True)}


def make_executor(
    folder: Path | str = JOB_FOLDER,
    cluster: str | None = None,
    *,
    timeout_min: int | None = None,
    cpus_per_task: int | None = None,
    mem_gb: float | None = None,
    slurm_partition: str | None = None,
) -> Any:
    """A :func:`meow.slurm_executor` for the EME jobs.

    ``cluster`` selects the backend: ``"slurm"`` (require a cluster), ``"local"``
    (local subprocesses, the default here), ``"debug"`` (in-process), or ``None``
    (auto: slurm if available, else local). The cluster, per-task cpu count,
    wall-clock timeout and partition default to the ``MEOW_SLURM_CLUSTER``,
    ``MEOW_CPUS_PER_TASK``, ``MEOW_TIMEOUT_MIN`` and ``MEOW_SLURM_PARTITION``
    environment variables (shared by every example's parallel/slurm runs).
    """
    return mw.slurm_executor(
        folder=str(folder),
        cluster=cluster or _backends.slurm_cluster(),
        timeout_min=_backends.timeout_min() if timeout_min is None else timeout_min,
        cpus_per_task=(
            _backends.cpus_per_task() if cpus_per_task is None else cpus_per_task
        ),
        mem_gb=mem_gb,
        slurm_partition=slurm_partition or _backends.slurm_partition(),
    )


def _design_sweep(platform: Platform, cutoffs: np.ndarray, res: float) -> list:
    return [
        design_dichroic(platform, float(wl_c), wgb=WGB_DESIGN, res=res)
        for wl_c in cutoffs
    ]


def _demo_designs() -> tuple[list[DichroicDesign], dict[str, Any]]:
    """The designs + EME settings shared by every entry point of this demo."""
    gf.gpdk.PDK.activate()
    platform = si3n4_platform()
    res = pick(low=0.06, medium=0.045, high=0.03)
    cutoffs = pick(
        low=TARGET_CUTOFFS[::3], medium=TARGET_CUTOFFS[:4], high=TARGET_CUTOFFS
    )
    eme_kwargs = {
        "num_cells": _resolution.num_cells(low=8, medium=16),
        "num_modes": _resolution.num_modes(low=2, medium=4),
        "res": pick(low=0.07, medium=0.05, high=0.035),
    }
    return _design_sweep(platform, cutoffs, res), eme_kwargs


def _submit_runs() -> list[Any]:
    designs, eme_kwargs = _demo_designs()
    return submit_runs(
        designs,
        folder=JOB_FOLDER,
        num_cells=eme_kwargs["num_cells"],
        num_modes=eme_kwargs["num_modes"],
        device_res=eme_kwargs["res"],
    )


def main() -> dict[str, object]:
    """Design a few Si3N4 splitters and run their distributed analysis on slurm.

    Submits each design's EME as concurrent cell-subset jobs (dense spectrum +
    propagation fields), then assembles + plots them, writing the figures, GDS
    and data into per-design timestamped subfolders of ``MEOW_SLURM_FOLDER``. By
    default the jobs run as local subprocesses; set ``MEOW_SLURM_CLUSTER=slurm``
    (and ``MEOW_SLURM_PARTITION``) to use a cluster.
    """
    _submit_runs()
    return {"gathered": asyncio.run(agather_runs(JOB_FOLDER))}


def submit_main() -> dict[str, object]:
    """Session A: submit every design's distributed EME analysis and return.

    Each design's jobs land in a timestamped subfolder of ``MEOW_SLURM_FOLDER``;
    assemble + plot them later with :func:`gather_main`.
    """
    records = _submit_runs()
    return {
        "submitted": {
            r.label: {"job_ids": r.job_ids, "out_dir": r.out_dir} for r in records
        },
        "folder": str(JOB_FOLDER),
        "next": "run 'gather' in a later session with the same MEOW_SLURM_FOLDER",
    }


def gather_main() -> dict[str, object]:
    """Session B (later): reload and collect the run summaries (any session)."""
    return {"gathered": gather_runs(JOB_FOLDER)}


if __name__ == "__main__":
    _slurm.cli_main(
        "examples.papers.dichroic_designer_slurm",
        {"run": main, "submit": submit_main, "gather": gather_main},
    )
