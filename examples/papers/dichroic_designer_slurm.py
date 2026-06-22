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

**Reloading results in a later session.** Because submitit persists every job
(payload, logs and result) in its ``folder``, submission and collection do not
have to happen in the same process. :func:`submit_designs` submits every
design's EME *without waiting* and writes one small ``<cutoff>.eme.pkl`` record
(a :class:`meow.ParallelEMEJobs` handle, via ``examples.papers._slurm``) into
the job folder; :func:`gather_results` - run in a *different* python session
pointing at the same ``MEOW_SLURM_FOLDER`` - reloads those records, reattaches
to the still-running cluster jobs and cascades each device's S-matrix. This is
demonstrated by the ``submit`` / ``gather`` subcommands below.

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
from examples.papers import _slurm
from examples.papers.dichroic_designer import (
    DichroicDesign,
    Platform,
    design_dichroic,
    device_structures,
)
from examples.papers.dichroic_designer_si3n4 import (
    TARGET_CUTOFFS,
    WGB_DESIGN,
    si3n4_platform,
)
from examples.papers.magden2018_dichroic import GAP_OUT, lateral_positions

FAST = bool(int(os.environ.get("MEOW_EXAMPLE_FAST", "0")))
JOB_FOLDER = Path(os.environ.get("MEOW_SLURM_FOLDER", "meow_dichroic_jobs"))


# --------------------------------------------------------------------------
# EME cells of a designed device
# --------------------------------------------------------------------------
def device_mesh(design: DichroicDesign, res: float) -> mw.Mesh2D:
    """Cross-section mesh spanning the full designed device width."""
    comp = design.component
    plat = design.platform
    x_lo = float(comp.ymin) - 1.0
    x_hi = float(comp.ymax) + 1.0
    h, tcl = plat.core_thickness, plat.clad_thickness
    return mw.Mesh2D(
        x=np.arange(x_lo, x_hi + res / 2, res),
        y=np.arange(-tcl, h + tcl + res / 2, res),
    )


def device_cells(
    design: DichroicDesign, num_cells: int = 16, res: float = 0.06
) -> list[mw.Cell]:
    """Slice a designed device into ``num_cells`` equal-length EME cells."""
    structs = device_structures(design)
    length = float(design.component.xmax)
    lengths = np.full(num_cells, length / num_cells)
    return mw.create_cells(structs, device_mesh(design, res), lengths, z_min=0.0)


def short_pass_split(design: DichroicDesign) -> float:
    """Lateral position [um] separating the WGA short-pass and WGB long-pass
    output ports of a designed device (output modes with their energy centroid
    below this go to the short-pass port).
    """
    _, _, y_a_final = lateral_positions(
        design.w_a, design.wgb.rail_width, design.wgb.gap, design.gap, GAP_OUT
    )
    return y_a_final / 2


def port_powers_from_s(
    cells: list[mw.Cell],
    env: mw.Environment,
    s_matrix: Any,
    port_map: dict[str, int],
    num_modes: int,
    split: float,
) -> tuple[float, float]:
    """(short-pass, long-pass) power for the fundamental input mode.

    The output modes (needed to attribute power to the WGA short-pass and WGB
    long-pass ports) are solved once on the final cell; the parallel engine
    itself returns no field data. Only the precomputed ``split`` scalar is
    needed besides the S-matrix, so this can run in a later session that
    reloaded the EME result without the original design object.
    """
    s = np.asarray(s_matrix)
    cs_out = mw.CrossSection.from_cell(cell=cells[-1], env=env)
    modes_out = mw.compute_modes(cs_out, num_modes=num_modes)

    def centroid(mode: mw.Mode) -> float:
        d = np.abs(mode.Ex) ** 2
        return float(np.sum(mode.cs.mesh.Xx * d) / np.sum(d))

    t_short = t_long = 0.0
    for i, mode in enumerate(modes_out):
        power = float(np.abs(s[port_map[f"right@{i}"], port_map["left@0"]]) ** 2)
        if centroid(mode) < split:
            t_short += power
        else:
            t_long += power
    return t_short, t_long


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


def make_executor(
    folder: Path | str = JOB_FOLDER,
    cluster: str | None = None,
    *,
    timeout_min: int = 30,
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


def _fmt(d: dict[str, tuple[float, float]]) -> dict[str, dict[str, float]]:
    return {
        k: {"short_pass": round(s, 4), "long_pass": round(lng, 4)}
        for k, (s, lng) in d.items()
    }


def main() -> dict[str, object]:
    """Design a few Si3N4 splitters and run their EME on the cluster.

    Demonstrates both the blocking and the concurrent (async) workflows. By
    default the EME jobs run as local subprocesses; set ``MEOW_SLURM_CLUSTER=slurm``
    (and ``MEOW_SLURM_PARTITION``) on a slurm login node to run them on the cluster.
    """
    designs, eme_kwargs = _demo_designs()
    blocking = run_blocking(designs, executor=make_executor(), **eme_kwargs)
    concurrent = asyncio.run(
        run_concurrent(designs, executor=make_executor(), **eme_kwargs)
    )
    return {"blocking": _fmt(blocking), "concurrent": _fmt(concurrent)}


def submit_main() -> dict[str, object]:
    """Session A: submit every design's EME to the cluster and return.

    Writes one persisted record per design into ``MEOW_SLURM_FOLDER`` and exits
    without waiting. Collect the results later with :func:`gather_main`.
    """
    designs, eme_kwargs = _demo_designs()
    records = submit_designs(
        designs,
        executor=make_executor(),
        folder=JOB_FOLDER,
        num_cells=eme_kwargs["num_cells"],
        num_modes=eme_kwargs["num_modes"],
        res=eme_kwargs["res"],
    )
    return {
        "submitted": {r.label: r.jobs.job_ids for r in records},
        "folder": str(JOB_FOLDER),
        "next": "run 'gather' in a later session with the same MEOW_SLURM_FOLDER",
    }


def gather_main() -> dict[str, object]:
    """Session B (later): reload and collect the results from ``MEOW_SLURM_FOLDER``."""
    return {"gathered": _fmt(gather_results(JOB_FOLDER))}


if __name__ == "__main__":
    _slurm.cli_main(
        "examples.papers.dichroic_designer_slurm",
        {"run": main, "submit": submit_main, "gather": gather_main},
    )
