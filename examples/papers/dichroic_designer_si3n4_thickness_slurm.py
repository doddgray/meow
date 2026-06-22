"""Async slurm version of the Si3N4 dichroic thickness sweep.

This is the cluster version of :mod:`examples.papers.dichroic_designer_si3n4_thickness`:
it designs fully-etched Si3N4 dichroic splitters across three core thicknesses
(200, 100, 40 nm) and the 900-1200 nm cutoffs, and then runs **all the
simulation, analysis and plotting of each design asynchronously as a slurm
job**. Each (thickness, cutoff) design becomes one job
(:func:`examples.papers._analysis.analyze_dichroic`) that computes a dense
short-/long-pass transmission spectrum and the intensity propagation across the
cutoff, and writes its spectrum/propagation/design figures, the device GDS and
the raw data into a fresh **timestamped subfolder** of ``MEOW_SLURM_FOLDER``.

Because submitit persists every job in its folder, the jobs can be submitted
from one (short-lived) python session and their summaries reloaded/gathered in a
*different* session later - see the ``submit`` / ``gather`` subcommands. The
figures/GDS/data are written directly into the shared run folders, so the later
session only collects the small summary dicts.

Wavelength controls (spectrum/propagation bounds + counts) default sensibly and
are overridable via the ``MEOW_SPECTRUM_*`` / ``MEOW_PROP_*`` env vars (see
:mod:`examples.papers._analysis`).

Run the full async flow in one process (jobs as local subprocesses) with::

    python -m examples.papers.dichroic_designer_si3n4_thickness_slurm

or split submission and collection across two sessions::

    python -m examples.papers.dichroic_designer_si3n4_thickness_slurm submit
    python -m examples.papers.dichroic_designer_si3n4_thickness_slurm gather

On a slurm login node set ``MEOW_SLURM_CLUSTER=slurm`` (and
``MEOW_SLURM_PARTITION``) and a shared ``MEOW_SLURM_FOLDER`` first.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import gdsfactory as gf
import numpy as np

from examples.papers import _analysis, _slurm
from examples.papers.dichroic_designer import DichroicDesign, to_params
from examples.papers.dichroic_designer_si3n4_thickness import (
    CUTOFFS,
    THICKNESS_CONFIGS,
    platform,
    sweep,
)
from examples.papers.dichroic_designer_slurm import analysis_settings, make_executor

FAST = bool(int(os.environ.get("MEOW_EXAMPLE_FAST", "0")))
JOB_FOLDER = Path(os.environ.get("MEOW_SLURM_FOLDER", "meow_si3n4_thickness_jobs"))


def _label(t_nm: int, design: DichroicDesign) -> str:
    """Run label: thickness + targeted cutoff (unique across the sweep)."""
    return f"{t_nm}nm-{design.cutoff_wl * 1e3:.0f}nm"


def thickness_designs(
    thicknesses: list[int], cutoffs: np.ndarray
) -> list[tuple[int, DichroicDesign, float]]:
    """Design every (thickness, cutoff) splitter; skip unreachable ones.

    Returns ``(thickness_nm, design, device_res)`` tuples.
    """
    gf.gpdk.PDK.activate()
    out: list[tuple[int, DichroicDesign, float]] = []
    for t_nm in thicknesses:
        t_um, clad_t, wgb, res = THICKNESS_CONFIGS[t_nm]
        if FAST:
            res = max(res, 0.06)
        plat = platform(t_um, clad_t)
        out.extend(
            (t_nm, design, res)
            for design in sweep(plat, wgb, cutoffs, res)
            if design is not None
        )
    return out


def submit_runs(
    designs: list[tuple[int, DichroicDesign, float]],
    *,
    folder: Path | str = JOB_FOLDER,
    executor_factory: Any | None = None,
    num_cells: int = 16,
    num_modes: int = 4,
) -> list[_slurm.SavedRun]:
    """Submit one analysis job per (thickness, cutoff) design *without waiting*.

    Each job writes its figures/GDS/data into a fresh timestamped subfolder of
    ``folder``; returns the persisted :class:`_slurm.SavedRun` handles. Collect
    the summaries later with :func:`gather_runs`.
    """
    if executor_factory is None:
        def executor_factory(sub: Path) -> Any:
            return make_executor(folder=sub)

    records = []
    for t_nm, design, device_res in designs:
        records.append(
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
                label=_label(t_nm, design),
            )
        )
    return records


def gather_runs(folder: Path | str = JOB_FOLDER) -> dict[str, dict]:
    """Reload and collect every analysis summary from ``folder`` (any session)."""
    return {r.label: r.result() for r in _slurm.load_runs(folder)}


async def agather_runs(folder: Path | str = JOB_FOLDER) -> dict[str, dict]:
    """Async :func:`gather_runs` (awaits all the analysis jobs together)."""
    runs = _slurm.load_runs(folder)
    summaries = await asyncio.gather(*(r.aresult() for r in runs))
    return {r.label: s for r, s in zip(runs, summaries, strict=True)}


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------
def _sweep_config() -> tuple[list[int], np.ndarray, dict[str, int]]:
    thicknesses = [200, 40] if FAST else [200, 100, 40]
    cutoffs = CUTOFFS[::3] if FAST else CUTOFFS
    settings = {"num_cells": 8 if FAST else 16, "num_modes": 2 if FAST else 4}
    return thicknesses, cutoffs, settings


def _submit_all() -> list[_slurm.SavedRun]:
    thicknesses, cutoffs, settings = _sweep_config()
    designs = thickness_designs(thicknesses, cutoffs)
    return submit_runs(designs, folder=JOB_FOLDER, **settings)


def main() -> dict[str, object]:
    """Design the thickness sweep and run every design's analysis on the cluster."""
    _submit_all()
    return {"gathered": asyncio.run(agather_runs(JOB_FOLDER))}


def submit_main() -> dict[str, object]:
    """Session A: submit the whole sweep's analysis jobs and return."""
    records = _submit_all()
    return {
        "submitted": {
            r.label: {"job_id": r.job_id, "out_dir": r.out_dir} for r in records
        },
        "folder": str(JOB_FOLDER),
        "next": "run 'gather' in a later session with the same MEOW_SLURM_FOLDER",
    }


def gather_main() -> dict[str, object]:
    """Session B (later): reload and collect the sweep summaries."""
    return {"gathered": gather_runs(JOB_FOLDER)}


if __name__ == "__main__":
    _slurm.cli_main(
        "examples.papers.dichroic_designer_si3n4_thickness_slurm",
        {"run": main, "submit": submit_main, "gather": gather_main},
    )
