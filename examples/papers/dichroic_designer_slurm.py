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
  design workflows' jobs are in flight on the cluster simultaneously - and,
  because submitit persists jobs in its ``folder``, they can be collected from a
  *different* python session.

See ``examples/papers/README.md`` ("Running EME on a slurm cluster") for how to
configure the local and remote environments for blocking, async and
multi-session execution.

Run locally (jobs as local subprocesses) with::

    python -m examples.papers.dichroic_designer_slurm

or, on a login node of a slurm cluster, set ``MEOW_SLURM_CLUSTER=slurm`` (and
``MEOW_SLURM_PARTITION``) first.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import gdsfactory as gf
import numpy as np

import meow as mw
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


def _port_powers(
    design: DichroicDesign,
    cells: list[mw.Cell],
    env: mw.Environment,
    s_matrix: Any,
    port_map: dict[str, int],
    num_modes: int,
) -> tuple[float, float]:
    """(short-pass, long-pass) power for the fundamental input mode.

    The output modes (needed to attribute power to the WGA short-pass and WGB
    long-pass ports) are solved once on the final cell; the parallel engine
    itself returns no field data.
    """
    s = np.asarray(s_matrix)
    cs_out = mw.CrossSection.from_cell(cell=cells[-1], env=env)
    modes_out = mw.compute_modes(cs_out, num_modes=num_modes)
    _, _, y_a_final = lateral_positions(
        design.w_a, design.wgb.rail_width, design.wgb.gap, design.gap, GAP_OUT
    )
    split = y_a_final / 2

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


def main() -> dict[str, object]:
    """Design a few Si3N4 splitters and run their EME on the cluster.

    Demonstrates both the blocking and the concurrent (async) workflows. By
    default the EME jobs run as local subprocesses; set ``MEOW_SLURM_CLUSTER=slurm``
    (and ``MEOW_SLURM_PARTITION``) on a slurm login node to run them on the cluster.
    """
    gf.gpdk.PDK.activate()
    platform = si3n4_platform()
    res = 0.06 if FAST else 0.045
    cutoffs = TARGET_CUTOFFS[::3] if FAST else TARGET_CUTOFFS[:4]
    eme_kwargs = {
        "num_cells": 8 if FAST else 16,
        "num_modes": 2 if FAST else 4,
        "res": 0.07 if FAST else 0.05,
    }

    designs = _design_sweep(platform, cutoffs, res)

    blocking = run_blocking(designs, executor=make_executor(), **eme_kwargs)
    concurrent = asyncio.run(
        run_concurrent(designs, executor=make_executor(), **eme_kwargs)
    )

    def fmt(d: dict[str, tuple[float, float]]) -> dict[str, dict[str, float]]:
        return {
            k: {"short_pass": round(s, 4), "long_pass": round(lng, 4)}
            for k, (s, lng) in d.items()
        }

    return {"blocking": fmt(blocking), "concurrent": fmt(concurrent)}


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
