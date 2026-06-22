"""Async slurm EME of a *single* adiabatic dichroic coupler (submit + reload).

Where ``dichroic_designer_slurm`` runs an *array* of designs, this example
focuses on a **single** adiabatic dichroic-coupler design and walks through the
three stages of a slurm EME explicitly:

1. **prepare** (:func:`prepare`): design one Si3N4 dichroic coupler for a target
   cutoff and slice it into EME cells.
2. **asynchronously deploy** (:func:`submit`): submit the per-slice mode solves
   to the cluster with :func:`meow.submit_s_matrix_parallel` and return *at
   once* - the jobs run on the cluster, not in this process. A small picklable
   handle (a :class:`meow.ParallelEMEJobs` via ``examples.papers._slurm``) is
   written into the job folder.
3. **gather** (:func:`agather` / :func:`gather`): collect the finished jobs,
   cascade the device S-matrix and attribute the short-/long-pass powers.

Because step 2 only *submits*, steps 2 and 3 can run in **different python
sessions**: submit from a short-lived session (or a script that exits while the
cluster works), then reload the handle and gather the result later, as long as
both sessions point ``MEOW_SLURM_FOLDER`` at the same shared folder and the jobs
outlive the submitter (true for ``cluster="slurm"``). See
``examples/papers/README.md`` ("Running EME on a slurm cluster").

Run the full async flow in one process (jobs as local subprocesses) with::

    python -m examples.papers.dichroic_coupler_slurm

or split deployment and collection across two sessions::

    python -m examples.papers.dichroic_coupler_slurm submit   # session A
    python -m examples.papers.dichroic_coupler_slurm gather    # session B (later)

On a slurm login node set ``MEOW_SLURM_CLUSTER=slurm`` (and
``MEOW_SLURM_PARTITION``) and a shared ``MEOW_SLURM_FOLDER`` first.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import gdsfactory as gf

import meow as mw
from examples.papers import _slurm
from examples.papers.dichroic_designer import DichroicDesign, design_dichroic
from examples.papers.dichroic_designer_si3n4 import WGB_DESIGN, si3n4_platform
from examples.papers.dichroic_designer_slurm import (
    device_cells,
    make_executor,
    port_powers_from_s,
    short_pass_split,
)

FAST = bool(int(os.environ.get("MEOW_EXAMPLE_FAST", "0")))
JOB_FOLDER = Path(os.environ.get("MEOW_SLURM_FOLDER", "meow_dichroic_coupler_jobs"))

# the single coupler's persisted-record label (its filename stem in JOB_FOLDER)
RECORD_LABEL = "dichroic_coupler"


# --------------------------------------------------------------------------
# 1. prepare: design one coupler and slice it into EME cells
# --------------------------------------------------------------------------
def design_coupler(
    cutoff_wl: float = 1.0, res: float = 0.05
) -> DichroicDesign:
    """Design a single Si3N4 adiabatic dichroic coupler for ``cutoff_wl`` [um]."""
    gf.gpdk.PDK.activate()
    return design_dichroic(si3n4_platform(), cutoff_wl, wgb=WGB_DESIGN, res=res)


def prepare(
    design: DichroicDesign, *, num_cells: int = 16, res: float = 0.05
) -> tuple[list[mw.Cell], mw.Environment]:
    """Slice a designed coupler into EME cells at its cutoff wavelength."""
    cells = device_cells(design, num_cells, res)
    env = mw.Environment(wl=design.cutoff_wl, T=25.0)
    return cells, env


# --------------------------------------------------------------------------
# 2. asynchronously deploy: submit the EME jobs and return immediately
# --------------------------------------------------------------------------
def submit(
    design: DichroicDesign,
    *,
    executor: Any,
    folder: Path | str = JOB_FOLDER,
    num_cells: int = 16,
    num_modes: int = 4,
    res: float = 0.05,
) -> _slurm.SavedEME:
    """Asynchronously deploy the coupler's EME: submit its slice jobs and return.

    Returns as soon as the jobs are queued, persisting a single
    ``dichroic_coupler.eme.pkl`` handle into ``folder``. The submitting session
    may then exit; collect the result later with :func:`gather` / :func:`agather`
    pointed at the same ``folder``. Pass a :func:`meow.slurm_executor` with
    ``cluster="slurm"`` so the jobs outlive this process.
    """
    cells, env = prepare(design, num_cells=num_cells, res=res)
    return _slurm.submit_eme(
        RECORD_LABEL,
        cells,
        env,
        executor=executor,
        num_modes=num_modes,
        folder=folder,
        meta={"split": short_pass_split(design)},
    )


# --------------------------------------------------------------------------
# 3. gather: reload the handle and collect the finished result
# --------------------------------------------------------------------------
def _ports(record: _slurm.SavedEME, s_matrix: Any, port_map: dict) -> dict[str, float]:
    t_short, t_long = port_powers_from_s(
        record.jobs.cells,
        record.jobs.env,
        s_matrix,
        port_map,
        record.num_modes,
        record.meta["split"],
    )
    return {
        "short_pass": round(float(t_short), 4),
        "long_pass": round(float(t_long), 4),
    }


def gather(folder: Path | str = JOB_FOLDER) -> dict[str, float]:
    """Reload the persisted handle and block until the EME result is ready.

    Can run in a **different session** from :func:`submit`: it reattaches to the
    cluster jobs through the persisted :class:`meow.ParallelEMEJobs` handle.
    """
    record = _slurm.load_record(folder, RECORD_LABEL)
    s, pm = record.jobs.result()
    return _ports(record, s, pm)


async def agather(folder: Path | str = JOB_FOLDER) -> dict[str, float]:
    """Async :func:`gather` (awaits the jobs without blocking the event loop)."""
    record = _slurm.load_record(folder, RECORD_LABEL)
    s, pm = await record.jobs.aresult()
    return _ports(record, s, pm)


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------
def _settings() -> dict[str, Any]:
    return {
        "cutoff_wl": 1.0,
        "design_res": 0.06 if FAST else 0.05,
        "num_cells": 8 if FAST else 16,
        "num_modes": 2 if FAST else 4,
        "res": 0.07 if FAST else 0.05,
    }


async def run_async(folder: Path | str = JOB_FOLDER) -> dict[str, object]:
    """Prepare, asynchronously deploy and gather the single coupler's EME.

    Submission returns immediately (the jobs run on the cluster); the result is
    then awaited with :func:`agather`. In a real two-session run the process
    could exit after :func:`submit` and reload the handle later - here both
    halves run in one event loop for the demo.
    """
    settings = _settings()
    design = design_coupler(settings["cutoff_wl"], res=settings["design_res"])
    saved = submit(
        design,
        executor=make_executor(folder=folder),
        folder=folder,
        num_cells=settings["num_cells"],
        num_modes=settings["num_modes"],
        res=settings["res"],
    )
    ports = await agather(folder)
    return {
        "design": {
            "cutoff_nm": round(design.cutoff_wl * 1e3, 0),
            "w_a_nm": round(design.w_a * 1e3, 1),
            "gap_nm": round(design.gap * 1e3, 0),
            "length_um": round(design.total_length, 0),
        },
        "job_ids": saved.jobs.job_ids,
        "ports": ports,
    }


def main() -> dict[str, object]:
    """Run the full prepare -> async deploy -> gather flow in one process."""
    return asyncio.run(run_async(JOB_FOLDER))


def submit_main() -> dict[str, object]:
    """Session A: design + asynchronously deploy the coupler's EME, then return."""
    settings = _settings()
    design = design_coupler(settings["cutoff_wl"], res=settings["design_res"])
    saved = submit(
        design,
        executor=make_executor(folder=JOB_FOLDER),
        folder=JOB_FOLDER,
        num_cells=settings["num_cells"],
        num_modes=settings["num_modes"],
        res=settings["res"],
    )
    return {
        "submitted": saved.jobs.job_ids,
        "folder": str(JOB_FOLDER),
        "next": "run 'gather' in a later session with the same MEOW_SLURM_FOLDER",
    }


def gather_main() -> dict[str, object]:
    """Session B (later): reload the handle and collect the coupler's result."""
    return {"ports": gather(JOB_FOLDER)}


if __name__ == "__main__":
    _slurm.cli_main(
        "examples.papers.dichroic_coupler_slurm",
        {"run": main, "submit": submit_main, "gather": gather_main},
    )
