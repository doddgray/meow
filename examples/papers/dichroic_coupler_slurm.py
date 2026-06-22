"""Async slurm analysis of a *single* adiabatic dichroic coupler (submit + reload).

Where ``dichroic_designer_slurm`` runs an *array* of designs, this example
focuses on a **single** adiabatic dichroic-coupler design and walks through the
three stages of a slurm analysis explicitly:

1. **prepare** (:func:`design_coupler`): design one Si3N4 dichroic coupler for a
   target cutoff wavelength.
2. **asynchronously deploy** (:func:`submit`): submit the whole analysis as a
   single slurm job (:func:`examples.papers._analysis.analyze_dichroic`) and
   return *at once* - the job runs on the cluster, not in this process. A small
   picklable :class:`_slurm.SavedRun` handle is written into the run's
   timestamped output folder.
3. **gather** (:func:`agather` / :func:`gather`): collect the finished job's
   summary; the figures, GDS and data already live in the run folder.

The analysis job computes and saves, into a fresh **timestamped subfolder** of
``MEOW_SLURM_FOLDER``:

- a dense short-/long-pass **transmission spectrum** (``*_spectrum.png`` + the
  raw arrays in ``*_results.npz``);
- **propagation plots** of the intensity ``|Ex|^2`` along the device at a few
  wavelengths on either side of and at the cutoff (``*_propagation.png``);
- a layout + index-crossing **design figure** (``*_design.png``); and
- the device **GDS** (``*.gds``) and a JSON summary.

Because step 2 only *submits*, steps 2 and 3 can run in **different python
sessions**: submit from a short-lived session, then reload the handle and gather
later, as long as both sessions point ``MEOW_SLURM_FOLDER`` at the same shared
folder and the jobs outlive the submitter (true for ``cluster="slurm"``).
Wavelength controls (spectrum/propagation bounds + counts) have sensible
defaults and are overridable via the ``MEOW_SPECTRUM_*`` / ``MEOW_PROP_*`` env
vars (see :mod:`examples.papers._analysis`).

Run the full async flow in one process (job as a local subprocess) with::

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

from examples.papers import _analysis, _slurm
from examples.papers.dichroic_designer import DichroicDesign, design_dichroic, to_params
from examples.papers.dichroic_designer_si3n4 import WGB_DESIGN, si3n4_platform
from examples.papers.dichroic_designer_slurm import analysis_settings, make_executor

FAST = bool(int(os.environ.get("MEOW_EXAMPLE_FAST", "0")))
JOB_FOLDER = Path(os.environ.get("MEOW_SLURM_FOLDER", "meow_dichroic_coupler_jobs"))

# the single coupler's run label (also the timestamped subfolder/file stem)
RECORD_LABEL = "dichroic_coupler"


# --------------------------------------------------------------------------
# 1. prepare: design one coupler
# --------------------------------------------------------------------------
def design_coupler(cutoff_wl: float = 1.0, res: float = 0.05) -> DichroicDesign:
    """Design a single Si3N4 adiabatic dichroic coupler for ``cutoff_wl`` [um]."""
    gf.gpdk.PDK.activate()
    return design_dichroic(si3n4_platform(), cutoff_wl, wgb=WGB_DESIGN, res=res)


# --------------------------------------------------------------------------
# 2. asynchronously deploy: submit the analysis job and return immediately
# --------------------------------------------------------------------------
def submit(
    design: DichroicDesign,
    *,
    folder: Path | str = JOB_FOLDER,
    executor_factory: Any | None = None,
    num_cells: int = 16,
    num_modes: int = 4,
    device_res: float = 0.05,
) -> _slurm.SavedRun:
    """Asynchronously deploy the coupler's full analysis as a single slurm job.

    Returns as soon as the job is queued, persisting a :class:`_slurm.SavedRun`
    handle into a fresh timestamped subfolder of ``folder`` (where the job will
    also write its spectrum/propagation/design figures, GDS and data). The
    submitting session may then exit; collect the result later with
    :func:`gather` / :func:`agather`. ``executor_factory(submitit_dir)`` builds
    the executor (default :func:`make_executor`); use ``cluster="slurm"`` so the
    job outlives this process.
    """
    if executor_factory is None:
        def executor_factory(sub: Path) -> Any:
            return make_executor(folder=sub)

    return _slurm.submit_run(
        _analysis.analyze_dichroic,
        to_params(design),
        analysis_settings(
            design, num_cells=num_cells, num_modes=num_modes, device_res=device_res
        ),
        executor_factory=executor_factory,
        folder=folder,
        label=RECORD_LABEL,
    )


# --------------------------------------------------------------------------
# 3. gather: reload the handle and collect the finished result
# --------------------------------------------------------------------------
def _latest_run(folder: Path | str) -> _slurm.SavedRun:
    runs = [r for r in _slurm.load_runs(folder) if r.label == RECORD_LABEL]
    if not runs:
        msg = f"no '{RECORD_LABEL}' analysis run found under {folder}"
        raise FileNotFoundError(msg)
    return runs[-1]


def gather(folder: Path | str = JOB_FOLDER) -> dict:
    """Reload the persisted handle and block until the analysis is ready.

    Can run in a **different session** from :func:`submit`: it reattaches to the
    cluster job through the persisted :class:`_slurm.SavedRun` handle and returns
    its summary (the figures/GDS/data are already in the run folder).
    """
    return _latest_run(folder).result()


async def agather(folder: Path | str = JOB_FOLDER) -> dict:
    """Async :func:`gather` (awaits the job without blocking the event loop)."""
    return await _latest_run(folder).aresult()


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------
def _settings() -> dict[str, Any]:
    return {
        "cutoff_wl": 1.0,
        "design_res": 0.06 if FAST else 0.05,
        "num_cells": 8 if FAST else 16,
        "num_modes": 2 if FAST else 4,
        "device_res": 0.07 if FAST else 0.05,
    }


def _submit(folder: Path | str) -> _slurm.SavedRun:
    settings = _settings()
    design = design_coupler(settings["cutoff_wl"], res=settings["design_res"])
    return submit(
        design,
        folder=folder,
        num_cells=settings["num_cells"],
        num_modes=settings["num_modes"],
        device_res=settings["device_res"],
    )


async def run_async(folder: Path | str = JOB_FOLDER) -> dict[str, object]:
    """Prepare, asynchronously deploy and gather the single coupler's analysis.

    Submission returns immediately (the job runs on the cluster); the result is
    then awaited with :func:`agather`. In a real two-session run the process
    could exit after :func:`submit` and reload the handle later - here both
    halves run in one event loop for the demo.
    """
    saved = _submit(folder)
    summary = await agather(folder)
    return {"out_dir": saved.out_dir, "summary": summary}


def main() -> dict[str, object]:
    """Run the full prepare -> async deploy -> gather flow in one process."""
    return asyncio.run(run_async(JOB_FOLDER))


def submit_main() -> dict[str, object]:
    """Session A: design + asynchronously deploy the coupler's analysis, then return."""
    saved = _submit(JOB_FOLDER)
    return {
        "submitted": {"job_id": saved.job_id, "out_dir": saved.out_dir},
        "folder": str(JOB_FOLDER),
        "next": "run 'gather' in a later session with the same MEOW_SLURM_FOLDER",
    }


def gather_main() -> dict[str, object]:
    """Session B (later): reload the handle and collect the coupler's summary."""
    return {"summary": gather(JOB_FOLDER)}


if __name__ == "__main__":
    _slurm.cli_main(
        "examples.papers.dichroic_coupler_slurm",
        {"run": main, "submit": submit_main, "gather": gather_main},
    )
