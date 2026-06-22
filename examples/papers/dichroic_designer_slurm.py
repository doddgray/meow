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

**Analysis runs (plots + GDS).** The default workflow (:func:`submit_runs` /
:func:`gather_runs`, used by ``main`` and the ``submit``/``gather``
subcommands) submits *one slurm job per design* that runs the whole analysis -
a dense short-/long-pass transmission spectrum, intensity-propagation plots at
a few wavelengths around the cutoff, a layout + index-crossing design figure,
and the device GDS - writing every figure, the GDS and the raw data into a
fresh **timestamped subfolder** of ``MEOW_SLURM_FOLDER`` (one per submitted
job). See :mod:`examples.papers._analysis` for the wavelength controls (env
vars ``MEOW_SPECTRUM_SPAN``/``MEOW_SPECTRUM_NPTS`` and
``MEOW_PROP_SPAN``/``MEOW_PROP_NPTS``/``MEOW_PROP_WLS``).

**Reloading results in a later session.** Because submitit persists every job
in its ``folder``, submission and collection do not have to happen in the same
process. :func:`gather_runs` - run in a *different* python session pointing at
the same ``MEOW_SLURM_FOLDER`` - reloads the persisted :class:`_slurm.SavedRun`
records, reattaches to the still-running jobs and returns their summaries (the
figures/GDS already on disk). A lighter S-matrix-only multi-session path
(:func:`submit_designs` / :func:`gather_results`, built on
:class:`meow.ParallelEMEJobs`) is also kept for just the port powers.

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
from examples.papers import _analysis, _slurm
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

FAST = bool(int(os.environ.get("MEOW_EXAMPLE_FAST", "0")))
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
    num_cells: int = 16,
    num_modes: int = 4,
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
    num_cells: int = 16,
    num_modes: int = 4,
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
    num_cells: int = 16,
    num_modes: int = 4,
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
        "spectrum_wls": _analysis.spectrum_wavelengths(
            design.cutoff_wl, n=5 if FAST else None
        ),
        "prop_wls": _analysis.propagation_wavelengths(
            design.cutoff_wl, n=3 if FAST else None
        ),
        "n_neff": 4 if FAST else 9,
        "num_z": 200 if FAST else 600,
    }


def submit_runs(
    designs: list[DichroicDesign],
    *,
    folder: Path | str = JOB_FOLDER,
    executor_factory: Any | None = None,
    num_cells: int = 16,
    num_modes: int = 4,
    device_res: float = 0.05,
) -> list[_slurm.SavedRun]:
    """Submit a full analysis (spectrum + propagation + plots + GDS) per design.

    Each design becomes one slurm job (:func:`_analysis.analyze_dichroic`) that
    writes all of its figures, GDS and data into a fresh timestamped subfolder
    of ``folder``; this returns immediately with the persisted
    :class:`_slurm.SavedRun` handles. Collect the summaries later (any session)
    with :func:`gather_runs`. ``executor_factory(submitit_dir)`` builds the
    per-run executor (default: :func:`make_executor`).
    """
    if executor_factory is None:
        def executor_factory(sub: Path) -> Any:
            return make_executor(folder=sub)

    return [
        _slurm.submit_run(
            _analysis.analyze_dichroic,
            to_params(design),
            analysis_settings(
                design,
                num_cells=num_cells,
                num_modes=num_modes,
                device_res=device_res,
            ),
            executor_factory=executor_factory,
            folder=folder,
            label=f"{design.cutoff_wl * 1e3:.0f}nm",
        )
        for design in designs
    ]


def gather_runs(folder: Path | str = JOB_FOLDER) -> dict[str, dict]:
    """Reload and collect the analysis summaries from ``folder`` (any session).

    The figures/GDS/data already live in each run's timestamped subfolder; this
    reattaches to the (persisted) jobs and returns their summary dicts.
    """
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


def _design_sweep(platform: Platform, cutoffs: np.ndarray, res: float) -> list:
    return [
        design_dichroic(platform, float(wl_c), wgb=WGB_DESIGN, res=res)
        for wl_c in cutoffs
    ]


def _demo_designs() -> tuple[list[DichroicDesign], dict[str, Any]]:
    """The designs + EME settings shared by every entry point of this demo."""
    gf.gpdk.PDK.activate()
    platform = si3n4_platform()
    res = 0.06 if FAST else 0.045
    cutoffs = TARGET_CUTOFFS[::3] if FAST else TARGET_CUTOFFS[:4]
    eme_kwargs = {
        "num_cells": 8 if FAST else 16,
        "num_modes": 2 if FAST else 4,
        "res": 0.07 if FAST else 0.05,
    }
    return _design_sweep(platform, cutoffs, res), eme_kwargs


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
    """Design a few Si3N4 splitters and run their full analysis on the cluster.

    Submits one analysis job per design (spectrum + propagation + plots + GDS,
    each into its own timestamped subfolder of ``MEOW_SLURM_FOLDER``) and awaits
    them. By default the jobs run as local subprocesses; set
    ``MEOW_SLURM_CLUSTER=slurm`` (and ``MEOW_SLURM_PARTITION``) to use a cluster.
    """
    _submit_runs()
    return {"gathered": asyncio.run(agather_runs(JOB_FOLDER))}


def submit_main() -> dict[str, object]:
    """Session A: submit every design's analysis to the cluster and return.

    Each design's figures/GDS/data land in a timestamped subfolder of
    ``MEOW_SLURM_FOLDER``; collect the summaries later with :func:`gather_main`.
    """
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
        "examples.papers.dichroic_designer_slurm",
        {"run": main, "submit": submit_main, "gather": gather_main},
    )
