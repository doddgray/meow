"""Async slurm analysis of a *single* adiabatic dichroic coupler (submit + reload).

Where ``dichroic_designer_slurm`` runs an *array* of designs, this example
focuses on a **single** adiabatic dichroic-coupler design and walks through the
three stages of a slurm analysis explicitly:

1. **prepare** (:func:`design_coupler`): design one Si3N4 dichroic coupler for a
   target cutoff wavelength.
2. **asynchronously deploy** (:func:`submit`): break the coupler's EME into
   **subsets of cells run concurrently as separate slurm jobs** - the dense
   transmission spectrum as slice-group jobs and (when full fields are saved)
   the propagation fields as single-cell jobs - returning *at once*. A picklable
   run record (``run.pkl``) is written into the run's timestamped output folder.
3. **gather** (:func:`agather` / :func:`gather`): reattach to the jobs, assemble
   the spectrum + propagation and write the figures, GDS and data into the run
   folder.

The :func:`gather` step computes and saves, into a fresh **timestamped
subfolder** of ``MEOW_SLURM_FOLDER``:

- a dense short-/long-pass **transmission spectrum** (``*_spectrum.png`` + the
  raw arrays saved redundantly as ``*_spectrum.csv`` / ``*_spectrum.json``);
- **propagation plots** of the intensity ``|Ex|^2`` along the device at a few
  wavelengths on either side of and at the cutoff (``*_propagation.png``), with
  the per-cell mode fields in a compressed HDF5 dataset (``*_fields.h5``);
- a layout + index-crossing **design figure** (``*_design.png``); and
- the device **GDS** (``*.gds``) and a ``*_summary.csv`` / ``*_summary.json``.

Because step 2 only *submits*, steps 2 and 3 can run in **different python
sessions**: submit from a short-lived session, then reload the run and gather
later, as long as both sessions point ``MEOW_SLURM_FOLDER`` at the same shared
folder and the jobs outlive the submitter (true for ``cluster="slurm"``).
Wavelength controls (spectrum/propagation bounds + counts) have sensible
defaults and are overridable via the ``MEOW_SPECTRUM_*`` / ``MEOW_PROP_*`` env
vars; full-field saving is toggled by ``save_fields`` / ``MEOW_SAVE_FIELDS`` and
the resolution preset by ``MEOW_EXAMPLE_RES`` (see
:mod:`examples.papers._analysis` / :mod:`examples.papers._resolution`).

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

from examples.papers import _analysis, _resolution, _slurm
from examples.papers.dichroic_designer import DichroicDesign, design_dichroic, to_params
from examples.papers.dichroic_designer_si3n4 import WGB_DESIGN, si3n4_platform
from examples.papers.dichroic_designer_slurm import analysis_settings, make_executor

pick = _resolution.pick
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
# 2. asynchronously deploy: distribute the EME jobs and return immediately
# --------------------------------------------------------------------------
def submit(
    design: DichroicDesign,
    *,
    folder: Path | str = JOB_FOLDER,
    executor_factory: Any | None = None,
    num_cells: int = 128,
    num_modes: int = 8,
    device_res: float = 0.05,
    save_fields: bool | None = None,
) -> Any:
    """Asynchronously deploy the coupler's EME as concurrent cell-subset jobs.

    Submits the dense transmission spectrum as slice-group jobs and (when
    ``save_fields`` is on, the default) the propagation fields as single-cell
    jobs, returning as soon as they are queued. A persisted run record
    (``run.pkl``) is written into a fresh timestamped subfolder of ``folder``;
    the submitting session may then exit and a later session can :func:`gather`
    the assembled spectrum/propagation/design figures, GDS and data.
    ``executor_factory(dir)`` builds the per-job executor (default
    :func:`make_executor`); use ``cluster="slurm"`` so the jobs outlive this
    process.
    """
    if executor_factory is None:
        def executor_factory(sub: Path) -> Any:
            return make_executor(folder=sub)

    return _slurm.start_run(
        _analysis.submit_dichroic_run,
        to_params(design),
        analysis_settings(
            design, num_cells=num_cells, num_modes=num_modes, device_res=device_res
        ),
        folder=folder,
        label=RECORD_LABEL,
        executor_factory=executor_factory,
        save_fields=save_fields,
    )


# --------------------------------------------------------------------------
# 3. gather: reload the run and assemble + plot the results
# --------------------------------------------------------------------------
def _latest_run(folder: Path | str) -> Any:
    runs = [r for r in _slurm.load_runs(folder) if r.label == RECORD_LABEL]
    if not runs:
        msg = f"no '{RECORD_LABEL}' analysis run found under {folder}"
        raise FileNotFoundError(msg)
    return runs[-1]


def gather(folder: Path | str = JOB_FOLDER) -> dict:
    """Reload the run and assemble the distributed EME into figures + data.

    Can run in a **different session** from :func:`submit`: it reattaches to the
    cluster jobs through the persisted run record, assembles the spectrum (and
    propagation, if fields were saved) and writes the figures/GDS/data into the
    run folder, returning the summary.
    """
    return _latest_run(folder).gather()


async def agather(folder: Path | str = JOB_FOLDER) -> dict:
    """Async :func:`gather` (assembles + plots off the event loop)."""
    return await _latest_run(folder).agather()


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------
def _settings() -> dict[str, Any]:
    return {
        "cutoff_wl": 1.0,
        "design_res": pick(low=0.06, medium=0.05, high=0.03),
        "num_cells": _resolution.num_cells(low=8, medium=16),
        "num_modes": _resolution.num_modes(low=2, medium=4),
        "device_res": pick(low=0.07, medium=0.05, high=0.035),
    }


def _submit(folder: Path | str) -> Any:
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
    """Session A: design + asynchronously deploy the coupler's EME, then return."""
    saved = _submit(JOB_FOLDER)
    return {
        "submitted": {"job_ids": saved.job_ids, "out_dir": saved.out_dir},
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
