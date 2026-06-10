"""Parallelized EME based on concurrent slice-group jobs.

The chain of cells (2D slices) is split into small overlapping groups of
contiguous cells: triplets in the middle of the chain and pairs at the ends
(by default). Each group is solved as an independent job that computes the
modes of its cells, the overlap-based interface S-matrices between them, and
returns only the effective indices and interface matrices - never the mode
field data. The main process then builds the propagation matrices from the
returned effective indices and cascades everything into the full EME
S-matrix.

Because shared boundary cells are re-solved in two jobs, no field data has to
be transferred between jobs, which drastically reduces the memory and
storage footprint at the cost of roughly 1.5x (triplets) to 2x (pairs)
redundant mode solves. This relies on the mode solver being deterministic:
the same cross section must yield the same mode basis (ordering, sign and
phase) in every job, which holds for the default tidy3d-based solver
(deterministic eigensolver seed + deterministic mode normalization) as long
as all jobs run the same software stack. A consistency check on the
effective indices of shared cells guards against violations.

Jobs can run locally (subprocesses via ``ProcessPoolExecutor``, the default)
or on a slurm cluster by passing a ``submitit`` executor (see
:func:`slurm_executor`). Any object with an
``executor.submit(fn, *args) -> job`` method where ``job.result()`` returns
the function result can be used.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp
import numpy as np
import sax

from meow.cell import Cell
from meow.cross_section import CrossSection
from meow.eme.cascade import cascade_s_matrices
from meow.eme.interface import compute_interface_s_matrix
from meow.environment import Environment


@runtime_checkable
class JobExecutor(Protocol):
    """Anything that can run group jobs: e.g. a ``concurrent.futures.Executor``
    or a ``submitit`` executor.
    """  # noqa: D205

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Schedule ``fn(*args, **kwargs)``; returns a job with ``.result()``."""
        ...


@dataclass
class GroupResult:
    """The result of one slice-group job.

    Contains everything the main process needs to assemble the full EME
    S-matrix - small per-cell/per-interface matrices only, no mode fields.
    """

    start: int
    """Index of the first cell of the group within the full chain."""
    neffs: list[np.ndarray] = field(default_factory=list)
    """Effective indices of the modes of each cell in the group."""
    interfaces: list[np.ndarray] = field(default_factory=list)
    """Dense interface S-matrix between each pair of adjacent cells."""


def chunk_cell_indices(
    num_cells: int, max_interfaces_per_job: int = 2
) -> list[tuple[int, int]]:
    """Split a chain of cells into contiguous overlapping slice groups.

    Each group ``(start, stop)`` covers the cells ``start..stop`` (inclusive)
    and the ``stop - start`` interfaces between them. Adjacent groups share
    one boundary cell, whose modes are redundantly re-solved in both jobs so
    that no mode field data needs to be exchanged between jobs.

    With the default of 2 interfaces per job, the chain is divided into
    triplets of cells, with a final pair when the number of interfaces is
    odd (e.g. 6 cells -> groups (0, 2), (2, 4), (4, 5)).

    Args:
        num_cells: the total number of cells in the EME chain.
        max_interfaces_per_job: how many interfaces each job computes:
            1 yields pairs of cells, 2 yields triplets, etc.

    Returns:
        A list of ``(start, stop)`` inclusive cell-index ranges.
    """
    if num_cells < 2:
        msg = f"At least 2 cells are required for an EME simulation. Got {num_cells}."
        raise ValueError(msg)
    if max_interfaces_per_job < 1:
        msg = f"max_interfaces_per_job should be >= 1. Got {max_interfaces_per_job}."
        raise ValueError(msg)
    num_interfaces = num_cells - 1
    groups = []
    start = 0
    while start < num_interfaces:
        stop = min(start + max_interfaces_per_job, num_interfaces)
        groups.append((start, stop))
        start = stop
    return groups


def compute_group_result(
    cells_data: list[dict[str, Any] | Cell],
    env_data: dict[str, Any] | Environment,
    start: int,
    num_modes: int = 10,
    compute_modes_kwargs: dict[str, Any] | None = None,
    interface_kwargs: dict[str, Any] | None = None,
) -> GroupResult:
    """Solve one slice group: modes, overlaps and interface S-matrices.

    This is the function executed by each parallel job. It accepts the cells
    in serialized (``model_dump``) or model form so it can be shipped to a
    subprocess or a slurm job, and returns only effective indices and
    interface S-matrices: the mode fields never leave the job.

    Args:
        cells_data: the (serialized) cells of this group, in chain order.
        env_data: the (serialized) simulation environment.
        start: index of the first cell of the group within the full chain.
        num_modes: number of modes to compute per cell.
        compute_modes_kwargs: extra kwargs for ``compute_modes``.
        interface_kwargs: extra kwargs for ``compute_interface_s_matrix``.

    Returns:
        The :class:`GroupResult` for this group.
    """
    from meow.fde import compute_modes  # fmt: skip

    cells = [Cell.model_validate(c) for c in cells_data]
    env = Environment.model_validate(env_data)
    css = [CrossSection.from_cell(cell=cell, env=env) for cell in cells]
    modes = [
        compute_modes(cs, num_modes=num_modes, **(compute_modes_kwargs or {}))
        for cs in css
    ]
    neffs = [
        np.asarray([m.neff for m in modes_], dtype=np.complex128) for modes_ in modes
    ]
    interfaces = []
    for modes_l, modes_r in pairwise(modes):
        S, _ = compute_interface_s_matrix(modes_l, modes_r, **(interface_kwargs or {}))
        interfaces.append(np.asarray(S))
    return GroupResult(start=start, neffs=neffs, interfaces=interfaces)


def compute_s_matrix_parallel(
    cells: list[Cell],
    env: Environment,
    *,
    num_modes: int = 10,
    max_interfaces_per_job: int = 2,
    executor: JobExecutor | None = None,
    max_workers: int | None = None,
    sax_backend: sax.Backend = "klu",
    neff_atol: float = 1e-6,
    compute_modes_kwargs: dict[str, Any] | None = None,
    interface_kwargs: dict[str, Any] | None = None,
) -> sax.SDenseMM:
    """Compute the EME S-matrix using concurrent slice-group jobs.

    The cells are divided into overlapping groups (triplets in the middle,
    pairs at the ends by default; see :func:`chunk_cell_indices`) and each
    group is solved as an independent job via :func:`compute_group_result`.
    Only effective indices and interface S-matrices are returned by the jobs;
    the final S-matrix is cascaded in the calling process.

    Args:
        cells: the cells of the EME chain.
        env: the simulation environment.
        num_modes: number of modes to compute per cell.
        max_interfaces_per_job: how many interfaces each job computes
            (1 -> pairs of cells, 2 -> triplets, ...).
        executor: where to run the jobs. Defaults to a local
            ``ProcessPoolExecutor`` (spawned subprocesses). Pass a submitit
            executor (e.g. from :func:`slurm_executor`) to run the jobs on a
            slurm cluster, or any object satisfying :class:`JobExecutor`.
        max_workers: max number of local subprocesses (only used when no
            executor is given).
        sax_backend: SAX backend used to cascade the S-matrices.
        neff_atol: tolerance for the consistency check between the effective
            indices of shared cells solved redundantly in two jobs.
        compute_modes_kwargs: extra kwargs for ``compute_modes``.
        interface_kwargs: extra kwargs for ``compute_interface_s_matrix``.

    Returns:
        A tuple ``(S, port_map)`` in SAX dense multimode format.
    """
    groups = chunk_cell_indices(len(cells), max_interfaces_per_job)
    if executor is None:
        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=mp.get_context("spawn")
        ) as own_executor:
            jobs = _submit_group_jobs(
                own_executor,
                cells,
                env,
                groups,
                num_modes,
                compute_modes_kwargs,
                interface_kwargs,
            )
            results = [job.result() for job in jobs]
    else:
        jobs = _submit_group_jobs(
            executor,
            cells,
            env,
            groups,
            num_modes,
            compute_modes_kwargs,
            interface_kwargs,
        )
        results = [job.result() for job in jobs]
    return _assemble_s_matrix(
        results, cells, env, sax_backend=sax_backend, neff_atol=neff_atol
    )


async def acompute_s_matrix_parallel(
    cells: list[Cell],
    env: Environment,
    *,
    num_modes: int = 10,
    max_interfaces_per_job: int = 2,
    executor: JobExecutor | None = None,
    max_workers: int | None = None,
    sax_backend: sax.Backend = "klu",
    neff_atol: float = 1e-6,
    compute_modes_kwargs: dict[str, Any] | None = None,
    interface_kwargs: dict[str, Any] | None = None,
) -> sax.SDenseMM:
    """Async version of :func:`compute_s_matrix_parallel`.

    Awaits the group jobs concurrently (without blocking the event loop), so
    multiple EME simulations can be in flight at the same time, e.g. when
    sweeping a parameter with jobs running on a slurm cluster.
    """
    groups = chunk_cell_indices(len(cells), max_interfaces_per_job)
    own_executor: ProcessPoolExecutor | None = None
    if executor is None:
        executor = own_executor = ProcessPoolExecutor(
            max_workers=max_workers, mp_context=mp.get_context("spawn")
        )
    try:
        jobs = _submit_group_jobs(
            executor,
            cells,
            env,
            groups,
            num_modes,
            compute_modes_kwargs,
            interface_kwargs,
        )
        results = list(
            await asyncio.gather(*(asyncio.to_thread(job.result) for job in jobs))
        )
    finally:
        if own_executor is not None:
            own_executor.shutdown()
    return _assemble_s_matrix(
        results, cells, env, sax_backend=sax_backend, neff_atol=neff_atol
    )


def slurm_executor(
    folder: str = "meow_eme_jobs",
    *,
    cluster: str | None = None,
    timeout_min: int = 60,
    cpus_per_task: int = 2,
    mem_gb: float | None = None,
    slurm_partition: str | None = None,
    **parameters: Any,
) -> JobExecutor:
    """Create a submitit executor to run EME group jobs on a slurm cluster.

    Requires the optional ``submitit`` package (``pip install submitit``).

    Args:
        folder: directory used by submitit for job logs and pickled payloads.
        cluster: "slurm" to require slurm, "local" to run the jobs as local
            subprocesses with the same machinery, "debug" to run them in
            process. None (default) auto-detects: slurm when available,
            local subprocesses otherwise.
        timeout_min: time limit per job in minutes.
        cpus_per_task: number of cpus per job.
        mem_gb: memory per job in GB.
        slurm_partition: the slurm partition to submit to.
        **parameters: any extra submitit parameters
            (see ``submitit.AutoExecutor.update_parameters``).

    Returns:
        A submitit executor that can be passed as ``executor=`` to
        :func:`compute_s_matrix_parallel` or :func:`acompute_s_matrix_parallel`.
    """
    try:
        import submitit
    except ImportError as e:
        msg = (
            "Running EME jobs through slurm requires the 'submitit' package. "
            "Install it with: pip install submitit"
        )
        raise ImportError(msg) from e

    executor = submitit.AutoExecutor(folder=folder, cluster=cluster)
    params: dict[str, Any] = {
        "timeout_min": timeout_min,
        "cpus_per_task": cpus_per_task,
    }
    if mem_gb is not None:
        params["mem_gb"] = mem_gb
    if slurm_partition is not None:
        params["slurm_partition"] = slurm_partition
    executor.update_parameters(**params, **parameters)
    return executor


def _submit_group_jobs(
    executor: JobExecutor,
    cells: list[Cell],
    env: Environment,
    groups: list[tuple[int, int]],
    num_modes: int,
    compute_modes_kwargs: dict[str, Any] | None,
    interface_kwargs: dict[str, Any] | None,
) -> list[Any]:
    cells_data = [cell.model_dump() for cell in cells]
    env_data = env.model_dump()
    return [
        executor.submit(
            compute_group_result,
            cells_data[start : stop + 1],
            env_data,
            start,
            num_modes,
            compute_modes_kwargs,
            interface_kwargs,
        )
        for start, stop in groups
    ]


def _assemble_s_matrix(
    results: list[GroupResult],
    cells: list[Cell],
    env: Environment,
    *,
    sax_backend: sax.Backend = "klu",
    neff_atol: float = 1e-6,
) -> sax.SDenseMM:
    """Assemble the full EME S-matrix from the group job results."""
    num_cells = len(cells)
    neffs: dict[int, np.ndarray] = {}
    interface_arrays: dict[int, np.ndarray] = {}
    for result in sorted(results, key=lambda r: r.start):
        for offset, n in enumerate(result.neffs):
            i = result.start + offset
            n = np.asarray(n)
            if i in neffs:
                _check_shared_cell_consistency(i, neffs[i], n, neff_atol)
            else:
                neffs[i] = n
        for offset, S in enumerate(result.interfaces):
            interface_arrays[result.start + offset] = np.asarray(S)

    missing_cells = [i for i in range(num_cells) if i not in neffs]
    missing_interfaces = [i for i in range(num_cells - 1) if i not in interface_arrays]
    if missing_cells or missing_interfaces:
        msg = (
            "Incomplete parallel EME results: missing modes for cells "
            f"{missing_cells} and interface matrices for {missing_interfaces}."
        )
        raise ValueError(msg)

    propagations: dict[str, sax.SDictMM] = {
        f"p_{i}": _propagation_s_dict(neffs[i], env.wl, cells[i].length)
        for i in range(num_cells)
    }
    interfaces: dict[str, sax.SDenseMM] = {}
    for k in range(num_cells - 1):
        num_left, num_right = len(neffs[k]), len(neffs[k + 1])
        S = interface_arrays[k]
        if S.shape != (num_left + num_right, num_left + num_right):
            msg = (
                f"Interface {k} S-matrix has shape {S.shape}, expected "
                f"{(num_left + num_right, num_left + num_right)}. The mode "
                "bases of the parallel jobs are inconsistent."
            )
            raise ValueError(msg)
        port_map = {f"left@{i}": i for i in range(num_left)}
        port_map |= {f"right@{i}": num_left + i for i in range(num_right)}
        interfaces[f"i_{k}_{k + 1}"] = (jnp.asarray(S), port_map)
    return cascade_s_matrices(propagations, interfaces, sax_backend=sax_backend)


def _check_shared_cell_consistency(
    cell_index: int,
    neffs1: np.ndarray,
    neffs2: np.ndarray,
    atol: float,
) -> None:
    """Verify that a cell solved redundantly in two jobs gave the same modes.

    The parallel decomposition assumes the mode solver is deterministic: the
    shared boundary cell of two adjacent groups must yield the same mode
    basis in both jobs for the cascaded S-matrix to be meaningful. Differing
    mode counts are fatal; small effective-index deviations (e.g. from
    heterogeneous cluster nodes) only trigger a warning, but large ones hint
    at swapped or missing modes.
    """
    if len(neffs1) != len(neffs2):
        msg = (
            f"Cell {cell_index} was solved in two parallel jobs which found a "
            f"different number of modes ({len(neffs1)} != {len(neffs2)}). "
            "The jobs' mode bases are inconsistent; this usually means the "
            "jobs ran with different settings or software versions."
        )
        raise ValueError(msg)
    max_diff = float(np.max(np.abs(neffs1 - neffs2))) if len(neffs1) else 0.0
    if max_diff > atol:
        msg = (
            f"Cell {cell_index} was solved in two parallel jobs whose "
            f"effective indices differ by up to {max_diff:.3e} (> {atol:.0e}). "
            "The mode bases may be inconsistent (e.g. near-degenerate modes "
            "swapping order); the assembled S-matrix may be inaccurate."
        )
        warnings.warn(msg, stacklevel=2)


def _propagation_s_dict(
    neffs: np.ndarray, wl: float, cell_length: float
) -> sax.SDictMM:
    """Diagonal propagation S-matrix of a cell from its effective indices."""
    s_dict = {
        (f"left@{i}", f"right@{i}"): jnp.exp(2j * jnp.pi * neff / wl * cell_length)
        for i, neff in enumerate(neffs)
    }
    return {**s_dict, **{(p2, p1): v for (p1, p2), v in s_dict.items()}}
