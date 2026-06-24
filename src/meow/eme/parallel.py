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
phase) in every job. This holds for the default tidy3d-based solver
(deterministic eigensolver seed + deterministic mode normalization) and for
the MPB backend (which seeds its randomized initial fields deterministically),
as long as all jobs run the same software stack. A consistency check on the
effective indices of shared cells guards against violations.

Jobs can run locally (subprocesses via ``ProcessPoolExecutor``, the default)
or on a slurm cluster by passing a ``submitit`` executor (see
:func:`slurm_executor`). Any object with an
``executor.submit(fn, *args) -> job`` method where ``job.result()`` returns
the function result can be used.

:func:`compute_s_matrix_parallel` submits the jobs and blocks until they
finish. For long cluster runs, :func:`submit_s_matrix_parallel` instead
returns a picklable :class:`ParallelEMEJobs` handle right after submitting, so
the result can be collected later - even from a *different python session* -
by saving the handle and reloading it once the (persisted) submitit jobs have
finished.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import pickle
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp
import numpy as np
import sax
from scipy.constants import c

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
    compute_modes: Callable | None = None,
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
        compute_modes: the FDE backend to use (default:
            ``meow.fde.compute_modes``, i.e. tidy3d). Must be a picklable
            top-level callable so it can be shipped to the worker; it must be
            *deterministic* (the tidy3d and the seeded MPB backends both are).

    Returns:
        The :class:`GroupResult` for this group.
    """
    if compute_modes is None:
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


@dataclass
class GroupSpectrumResult:
    """The result of one slice-group job swept over wavelength.

    The modes of the group's cells are re-solved at every sweep wavelength
    inside the job, so only the frequency-dependent effective indices and
    interface S-matrices ever leave the job - no mode field data.
    """

    start: int
    """Index of the first cell of the group within the full chain."""
    wls: np.ndarray = field(default_factory=lambda: np.empty(0))
    """The swept wavelengths."""
    per_wl: list[GroupResult] = field(default_factory=list)
    """The :class:`GroupResult` of this group at each wavelength."""


def compute_group_spectrum(
    cells_data: list[dict[str, Any] | Cell],
    env_data: dict[str, Any] | Environment,
    start: int,
    wls: Any,
    num_modes: int = 10,
    compute_modes_kwargs: dict[str, Any] | None = None,
    interface_kwargs: dict[str, Any] | None = None,
    compute_modes: Callable | None = None,
) -> GroupSpectrumResult:
    """Solve one slice group at every sweep wavelength.

    This is the function executed by each parallel job for a spectrum
    calculation: it runs the frequency-dependent mode simulations of its
    cells (dispersive materials are evaluated at each wavelength through
    the environment) and returns the per-wavelength effective indices and
    interface S-matrices.

    Args:
        cells_data: the (serialized) cells of this group, in chain order.
        env_data: the (serialized) simulation environment.
        start: index of the first cell of the group within the full chain.
        wls: the wavelengths to sweep.
        num_modes: number of modes to compute per cell.
        compute_modes_kwargs: extra kwargs for ``compute_modes``.
        interface_kwargs: extra kwargs for ``compute_interface_s_matrix``.
        compute_modes: the (picklable, deterministic) FDE backend to use
            (default: ``meow.fde.compute_modes``, tidy3d).
    """
    cells: list[dict[str, Any] | Cell] = [Cell.model_validate(c) for c in cells_data]
    env_dict = Environment.model_validate(env_data).model_dump()
    wls = np.asarray(wls, dtype=np.float64)
    per_wl = [
        compute_group_result(
            cells,
            {**env_dict, "wl": float(wl)},
            start,
            num_modes,
            compute_modes_kwargs,
            interface_kwargs,
            compute_modes,
        )
        for wl in wls
    ]
    return GroupSpectrumResult(start=start, wls=wls, per_wl=per_wl)


def _resolve_wls(wls: Any | None, freqs: Any | None) -> np.ndarray:
    """Resolve a sweep given as wavelengths [um] or optical frequencies [Hz]."""
    if (wls is None) == (freqs is None):
        msg = "Specify exactly one of wls (wavelengths in um) or freqs (in Hz)."
        raise ValueError(msg)
    if freqs is not None:
        return c / np.asarray(freqs, dtype=np.float64) * 1e6
    return np.asarray(wls, dtype=np.float64)


def compute_s_matrix_spectrum(
    cells: list[Cell],
    env: Environment,
    *,
    wls: Any | None = None,
    freqs: Any | None = None,
    num_modes: int = 10,
    max_interfaces_per_job: int = 2,
    executor: JobExecutor | None = None,
    max_workers: int | None = None,
    sax_backend: sax.Backend = "klu",
    neff_atol: float = 1e-6,
    compute_modes_kwargs: dict[str, Any] | None = None,
    interface_kwargs: dict[str, Any] | None = None,
    compute_modes: Callable | None = None,
) -> list[sax.SDenseMM]:
    """Compute EME S-matrix spectra using concurrent slice-group jobs.

    The sweep can be given either as wavelengths (``wls``, in um) or as
    optical frequencies (``freqs``, in Hz). Each concurrent job (local
    subprocess or slurm task) solves the frequency-dependent modes of its
    slice group at every sweep point and returns only the per-frequency
    effective indices and interface S-matrices; the full S-matrix at each
    sweep point is then cascaded in the calling process.

    Args:
        cells: the cells of the EME chain.
        env: the simulation environment (its ``wl`` is overridden per point).
        wls: the wavelengths to sweep [um].
        freqs: the optical frequencies to sweep [Hz] (alternative to wls).
        num_modes: number of modes to compute per cell.
        max_interfaces_per_job: how many interfaces each job computes.
        executor: where to run the jobs (see :func:`compute_s_matrix_parallel`).
        max_workers: max number of local subprocesses (only used when no
            executor is given).
        sax_backend: SAX backend used to cascade the S-matrices.
        neff_atol: tolerance of the shared-cell consistency check.
        compute_modes_kwargs: extra kwargs for ``compute_modes``.
        interface_kwargs: extra kwargs for ``compute_interface_s_matrix``.
        compute_modes: the (picklable, deterministic) FDE backend to run in
            each job (default: ``meow.fde.compute_modes``, tidy3d).

    Returns:
        A list of ``(S, port_map)`` tuples, one per sweep point, in the
        order of the given ``wls``/``freqs`` array.
    """
    wls_arr = _resolve_wls(wls, freqs)
    groups = chunk_cell_indices(len(cells), max_interfaces_per_job)
    if executor is None:
        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=mp.get_context("spawn")
        ) as own_executor:
            jobs = _submit_spectrum_jobs(
                own_executor,
                cells,
                env,
                groups,
                wls_arr,
                num_modes,
                compute_modes_kwargs,
                interface_kwargs,
                compute_modes,
            )
            results = [job.result() for job in jobs]
    else:
        jobs = _submit_spectrum_jobs(
            executor,
            cells,
            env,
            groups,
            wls_arr,
            num_modes,
            compute_modes_kwargs,
            interface_kwargs,
            compute_modes,
        )
        results = [job.result() for job in jobs]
    return _assemble_spectrum(
        results, cells, env, wls_arr, sax_backend=sax_backend, neff_atol=neff_atol
    )


async def acompute_s_matrix_spectrum(
    cells: list[Cell],
    env: Environment,
    *,
    wls: Any | None = None,
    freqs: Any | None = None,
    num_modes: int = 10,
    max_interfaces_per_job: int = 2,
    executor: JobExecutor | None = None,
    max_workers: int | None = None,
    sax_backend: sax.Backend = "klu",
    neff_atol: float = 1e-6,
    compute_modes_kwargs: dict[str, Any] | None = None,
    interface_kwargs: dict[str, Any] | None = None,
    compute_modes: Callable | None = None,
) -> list[sax.SDenseMM]:
    """Async version of :func:`compute_s_matrix_spectrum`."""
    wls_arr = _resolve_wls(wls, freqs)
    groups = chunk_cell_indices(len(cells), max_interfaces_per_job)
    own_executor: ProcessPoolExecutor | None = None
    if executor is None:
        executor = own_executor = ProcessPoolExecutor(
            max_workers=max_workers, mp_context=mp.get_context("spawn")
        )
    try:
        jobs = _submit_spectrum_jobs(
            executor,
            cells,
            env,
            groups,
            wls_arr,
            num_modes,
            compute_modes_kwargs,
            interface_kwargs,
            compute_modes,
        )
        results = list(
            await asyncio.gather(*(asyncio.to_thread(job.result) for job in jobs))
        )
    finally:
        if own_executor is not None:
            own_executor.shutdown()
    return _assemble_spectrum(
        results, cells, env, wls_arr, sax_backend=sax_backend, neff_atol=neff_atol
    )


def _submit_spectrum_jobs(
    executor: JobExecutor,
    cells: list[Cell],
    env: Environment,
    groups: list[tuple[int, int]],
    wls: np.ndarray,
    num_modes: int,
    compute_modes_kwargs: dict[str, Any] | None,
    interface_kwargs: dict[str, Any] | None,
    compute_modes: Callable | None = None,
) -> list[Any]:
    cells_data = [cell.model_dump() for cell in cells]
    env_data = env.model_dump()
    return [
        executor.submit(
            compute_group_spectrum,
            cells_data[start : stop + 1],
            env_data,
            start,
            wls,
            num_modes,
            compute_modes_kwargs,
            interface_kwargs,
            compute_modes,
        )
        for start, stop in groups
    ]


def _assemble_spectrum(
    results: list[GroupSpectrumResult],
    cells: list[Cell],
    env: Environment,
    wls: np.ndarray,
    *,
    sax_backend: sax.Backend = "klu",
    neff_atol: float = 1e-6,
) -> list[sax.SDenseMM]:
    """Assemble the per-wavelength S-matrices from the group spectra."""
    env_dict = env.model_dump()
    spectra = []
    for i, wl in enumerate(wls):
        env_wl = Environment.model_validate({**env_dict, "wl": float(wl)})
        spectra.append(
            _assemble_s_matrix(
                [r.per_wl[i] for r in results],
                cells,
                env_wl,
                sax_backend=sax_backend,
                neff_atol=neff_atol,
            )
        )
    return spectra


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
    compute_modes: Callable | None = None,
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
        compute_modes: the FDE backend to run in each job (default:
            ``meow.fde.compute_modes``, tidy3d). Must be a picklable top-level
            callable. The slice-group method needs a *deterministic* backend so
            shared cells re-solved in two workers return the same mode basis;
            the tidy3d backend and the (seeded) MPB backend
            (``meow.compute_modes_mpb``) both satisfy this.

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
                compute_modes,
            )
            results = [job.result() for job in jobs]
        return _assemble_s_matrix(
            results, cells, env, sax_backend=sax_backend, neff_atol=neff_atol
        )
    return submit_s_matrix_parallel(
        cells,
        env,
        executor=executor,
        num_modes=num_modes,
        max_interfaces_per_job=max_interfaces_per_job,
        sax_backend=sax_backend,
        neff_atol=neff_atol,
        compute_modes_kwargs=compute_modes_kwargs,
        interface_kwargs=interface_kwargs,
        compute_modes=compute_modes,
    ).result()


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
    compute_modes: Callable | None = None,
) -> sax.SDenseMM:
    """Async version of :func:`compute_s_matrix_parallel`.

    Awaits the group jobs concurrently (without blocking the event loop), so
    multiple EME simulations can be in flight at the same time, e.g. when
    sweeping a parameter with jobs running on a slurm cluster.
    """
    if executor is not None:
        return await submit_s_matrix_parallel(
            cells,
            env,
            executor=executor,
            num_modes=num_modes,
            max_interfaces_per_job=max_interfaces_per_job,
            sax_backend=sax_backend,
            neff_atol=neff_atol,
            compute_modes_kwargs=compute_modes_kwargs,
            interface_kwargs=interface_kwargs,
            compute_modes=compute_modes,
        ).aresult()
    groups = chunk_cell_indices(len(cells), max_interfaces_per_job)
    own_executor = ProcessPoolExecutor(
        max_workers=max_workers, mp_context=mp.get_context("spawn")
    )
    try:
        jobs = _submit_group_jobs(
            own_executor,
            cells,
            env,
            groups,
            num_modes,
            compute_modes_kwargs,
            interface_kwargs,
            compute_modes,
        )
        results = list(
            await asyncio.gather(*(asyncio.to_thread(job.result) for job in jobs))
        )
    finally:
        own_executor.shutdown()
    return _assemble_s_matrix(
        results, cells, env, sax_backend=sax_backend, neff_atol=neff_atol
    )


class _SubmittedJobs:
    """Mixin for a picklable handle over a list of submitted ``jobs``.

    Provides the bits shared by every submit/collect handle: inspecting the
    submitit job ids / folder, polling whether the jobs are done without
    blocking, and pickling the whole handle to disk so a *different python
    session* can reload it and collect the results (the submitit jobs persist
    their results in their ``folder`` and reload them on unpickling).
    """

    jobs: list[Any]

    @property
    def job_ids(self) -> list[str]:
        """The submitit job ids (empty strings for executors without ids)."""
        return [str(getattr(job, "job_id", "")) for job in self.jobs]

    @property
    def folder(self) -> str | None:
        """The submitit job folder the results are persisted in, if any."""
        for job in self.jobs:
            paths = getattr(job, "paths", None)
            folder = getattr(paths, "folder", None)
            if folder is not None:
                return str(folder)
        return None

    def done(self) -> bool:
        """Whether every job has finished (best-effort, never blocks)."""
        for job in self.jobs:
            is_done = getattr(job, "done", None)
            if callable(is_done):
                try:
                    if not is_done():
                        return False
                except Exception:  # noqa: BLE001 - a not-yet-known job isn't done
                    return False
        return True

    def save(self, path: str | Path) -> Path:
        """Pickle this handle to ``path`` so a later session can collect it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: str | Path):  # noqa: ANN206 - returns cls
        """Load a handle saved by :meth:`save` (e.g. in a later session)."""
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            msg = f"{path} does not contain a {cls.__name__}."
            raise TypeError(msg)
        return obj


@dataclass
class ParallelEMEJobs(_SubmittedJobs):
    """Handle for a submitted - but not yet collected - parallel EME run.

    Returned by :func:`submit_s_matrix_parallel`. It bundles the submitted
    slice-group jobs with the (small) metadata the calling process needs to
    cascade their results into the full EME S-matrix - the cells (for their
    lengths) and the environment (for its wavelength), but *not* any mode
    field data.

    The handle is **picklable**, which is what enables collecting the result
    in a *different python session* from the one that submitted the jobs: save
    it with :meth:`save` right after submitting, then in a later session
    :meth:`load` it and call :meth:`result` (or :meth:`aresult`). This works
    because a :func:`slurm_executor` (submitit) job persists its payload, logs
    and result in its ``folder`` and reloads them on unpickling, so the jobs
    keep running on the cluster after the submitting process exits (and
    ``job.result()`` reads the result back from the shared folder).

    Attributes:
        jobs: the submitted slice-group jobs (each has a ``.result()``).
        cells: the cells of the EME chain (used to build the propagation
            matrices when assembling the final S-matrix).
        env: the simulation environment.
        sax_backend: SAX backend used to cascade the S-matrices.
        neff_atol: tolerance of the shared-cell consistency check.
    """

    jobs: list[Any]
    cells: list[Cell]
    env: Environment
    sax_backend: sax.Backend = "klu"
    neff_atol: float = 1e-6

    def result(self) -> sax.SDenseMM:
        """Block until all jobs finish, then cascade the full EME S-matrix.

        Returns:
            A tuple ``(S, port_map)`` in SAX dense multimode format.
        """
        results = [job.result() for job in self.jobs]
        return _assemble_s_matrix(
            results,
            self.cells,
            self.env,
            sax_backend=self.sax_backend,
            neff_atol=self.neff_atol,
        )

    async def aresult(self) -> sax.SDenseMM:
        """Async :meth:`result` (awaits the jobs without blocking the loop)."""
        results = list(
            await asyncio.gather(*(asyncio.to_thread(job.result) for job in self.jobs))
        )
        return _assemble_s_matrix(
            results,
            self.cells,
            self.env,
            sax_backend=self.sax_backend,
            neff_atol=self.neff_atol,
        )



def submit_s_matrix_parallel(
    cells: list[Cell],
    env: Environment,
    *,
    executor: JobExecutor,
    num_modes: int = 10,
    max_interfaces_per_job: int = 2,
    sax_backend: sax.Backend = "klu",
    neff_atol: float = 1e-6,
    compute_modes_kwargs: dict[str, Any] | None = None,
    interface_kwargs: dict[str, Any] | None = None,
    compute_modes: Callable | None = None,
) -> ParallelEMEJobs:
    """Submit the slice-group jobs of a parallel EME *without* awaiting them.

    This is the non-blocking half of :func:`compute_s_matrix_parallel`: it
    chunks the cells, submits the group jobs to ``executor`` and returns a
    :class:`ParallelEMEJobs` handle immediately, instead of blocking until the
    jobs finish. Collect the result later (possibly in another python session)
    with :meth:`ParallelEMEJobs.result` / :meth:`ParallelEMEJobs.aresult`.

    The point is **multi-session execution**: submit a long EME on a slurm
    cluster from one (short-lived) session, :meth:`~ParallelEMEJobs.save` the
    handle, and :meth:`~ParallelEMEJobs.load` it in a later session to gather
    the results once the cluster jobs have finished. This requires an executor
    whose jobs outlive the submitting process and persist their results - i.e.
    a :func:`slurm_executor` (``cluster="slurm"`` on a real cluster, or
    ``"local"``/``"debug"`` for testing the workflow). A bare
    ``concurrent.futures`` executor does not persist its jobs, so pass one of
    those only for in-session collection.

    Args:
        cells: the cells of the EME chain.
        env: the simulation environment.
        executor: where to run the jobs - typically a :func:`slurm_executor`.
        num_modes: number of modes to compute per cell.
        max_interfaces_per_job: how many interfaces each job computes
            (1 -> pairs of cells, 2 -> triplets, ...).
        sax_backend: SAX backend used to cascade the S-matrices.
        neff_atol: tolerance of the shared-cell consistency check.
        compute_modes_kwargs: extra kwargs for ``compute_modes``.
        interface_kwargs: extra kwargs for ``compute_interface_s_matrix``.
        compute_modes: the (picklable, deterministic) FDE backend to run in
            each job (default: ``meow.fde.compute_modes``, tidy3d).

    Returns:
        A :class:`ParallelEMEJobs` handle over the submitted jobs.
    """
    if executor is None:
        msg = (
            "submit_s_matrix_parallel requires an executor whose jobs persist "
            "(e.g. meow.slurm_executor(...)). For an in-process local "
            "ProcessPoolExecutor run, use compute_s_matrix_parallel instead."
        )
        raise ValueError(msg)
    groups = chunk_cell_indices(len(cells), max_interfaces_per_job)
    jobs = _submit_group_jobs(
        executor,
        cells,
        env,
        groups,
        num_modes,
        compute_modes_kwargs,
        interface_kwargs,
        compute_modes,
    )
    return ParallelEMEJobs(
        jobs=jobs,
        cells=cells,
        env=env,
        sax_backend=sax_backend,
        neff_atol=neff_atol,
    )


@dataclass
class ParallelEMESpectrumJobs(_SubmittedJobs):
    """Handle for a submitted - but not yet collected - parallel EME *spectrum*.

    Returned by :func:`submit_s_matrix_spectrum`; the spectrum analogue of
    :class:`ParallelEMEJobs`. Each slice-group job solves the dispersive modes
    of its cells at every sweep wavelength and returns only the per-wavelength
    effective indices and interface S-matrices (no mode fields). Picklable, so
    it can be saved after submission and reloaded in a later session to gather
    the per-wavelength S-matrices.

    Attributes:
        jobs: the submitted slice-group spectrum jobs.
        cells: the cells of the EME chain.
        env: the simulation environment (its ``wl`` is overridden per point).
        wls: the swept wavelengths [um], in result order.
        sax_backend: SAX backend used to cascade the S-matrices.
        neff_atol: tolerance of the shared-cell consistency check.
    """

    jobs: list[Any]
    cells: list[Cell]
    env: Environment
    wls: np.ndarray
    sax_backend: sax.Backend = "klu"
    neff_atol: float = 1e-6

    def result(self) -> list[sax.SDenseMM]:
        """Block until all jobs finish; return one ``(S, port_map)`` per wl."""
        results = [job.result() for job in self.jobs]
        return _assemble_spectrum(
            results,
            self.cells,
            self.env,
            self.wls,
            sax_backend=self.sax_backend,
            neff_atol=self.neff_atol,
        )

    async def aresult(self) -> list[sax.SDenseMM]:
        """Async :meth:`result` (awaits the jobs without blocking the loop)."""
        results = list(
            await asyncio.gather(*(asyncio.to_thread(job.result) for job in self.jobs))
        )
        return _assemble_spectrum(
            results,
            self.cells,
            self.env,
            self.wls,
            sax_backend=self.sax_backend,
            neff_atol=self.neff_atol,
        )


def submit_s_matrix_spectrum(
    cells: list[Cell],
    env: Environment,
    *,
    executor: JobExecutor,
    wls: Any | None = None,
    freqs: Any | None = None,
    num_modes: int = 10,
    max_interfaces_per_job: int = 2,
    sax_backend: sax.Backend = "klu",
    neff_atol: float = 1e-6,
    compute_modes_kwargs: dict[str, Any] | None = None,
    interface_kwargs: dict[str, Any] | None = None,
    compute_modes: Callable | None = None,
) -> ParallelEMESpectrumJobs:
    """Submit the slice-group jobs of an EME *spectrum* without awaiting them.

    The non-blocking, multi-session half of :func:`compute_s_matrix_spectrum`:
    it chunks the cells, submits one spectrum job per slice group (each solving
    its cells at *every* sweep wavelength) and returns a
    :class:`ParallelEMESpectrumJobs` handle immediately. Use it - like
    :func:`submit_s_matrix_parallel` - to distribute a dense spectrum as
    concurrent slurm jobs and collect it later (possibly in another session).
    """
    if executor is None:
        msg = (
            "submit_s_matrix_spectrum requires an executor whose jobs persist "
            "(e.g. meow.slurm_executor(...)). For an in-process run use "
            "compute_s_matrix_spectrum instead."
        )
        raise ValueError(msg)
    wls_arr = _resolve_wls(wls, freqs)
    groups = chunk_cell_indices(len(cells), max_interfaces_per_job)
    jobs = _submit_spectrum_jobs(
        executor,
        cells,
        env,
        groups,
        wls_arr,
        num_modes,
        compute_modes_kwargs,
        interface_kwargs,
        compute_modes,
    )
    return ParallelEMESpectrumJobs(
        jobs=jobs,
        cells=cells,
        env=env,
        wls=wls_arr,
        sax_backend=sax_backend,
        neff_atol=neff_atol,
    )


def compute_cell_modes(
    cell_data: dict[str, Any] | Cell,
    env_data: dict[str, Any] | Environment,
    num_modes: int = 10,
    compute_modes_kwargs: dict[str, Any] | None = None,
    compute_modes: Callable | None = None,
) -> list[Any]:
    """Solve and return the *full* modes (with fields) of a single cell.

    Unlike :func:`compute_group_result` (which returns only effective indices
    and interface matrices), this keeps the complete :class:`~meow.Mode`
    objects, so the returned modes can be saved per cell and reused later for
    field reconstruction / propagation. This is the per-cell job used by
    :func:`submit_cell_modes`.
    """
    if compute_modes is None:
        from meow.fde import compute_modes  # fmt: skip

    cell = Cell.model_validate(cell_data)
    env = Environment.model_validate(env_data)
    cs = CrossSection.from_cell(cell=cell, env=env)
    return list(compute_modes(cs, num_modes=num_modes, **(compute_modes_kwargs or {})))


@dataclass
class ParallelFieldModeJobs(_SubmittedJobs):
    """Handle for submitted *single-cell* mode jobs that keep the full fields.

    Returned by :func:`submit_cell_modes`. Each job solves the modes of one
    cell and returns the complete :class:`~meow.Mode` objects (fields included),
    so :meth:`result` yields the per-cell mode bases needed to reconstruct or
    propagate fields (e.g. with :func:`meow.propagate_modes`) or to cascade the
    S-matrix with :func:`meow.compute_s_matrix`. Picklable for multi-session
    collection; note the field data makes these results much larger than the
    slice-group handles.

    Attributes:
        jobs: the submitted per-cell mode jobs (one per cell, in chain order).
        cells: the cells of the EME chain (aligned with ``jobs``).
        env: the simulation environment.
    """

    jobs: list[Any]
    cells: list[Cell]
    env: Environment

    def result(self) -> list[list[Any]]:
        """Block until all jobs finish; return the modes of each cell."""
        return [job.result() for job in self.jobs]

    async def aresult(self) -> list[list[Any]]:
        """Async :meth:`result` (awaits the jobs without blocking the loop)."""
        return list(
            await asyncio.gather(*(asyncio.to_thread(job.result) for job in self.jobs))
        )

    def to_dataset(self, *, attrs: dict[str, Any] | None = None) -> Any:
        """Collect the modes and bundle their fields into an xarray Dataset.

        Blocks until the per-cell jobs finish, then returns a dataset of the
        complex ``Ex..Hz`` fields and ``neff`` (see
        :func:`meow.eme.io.modes_to_dataset`). Save it to a single compressed
        HDF5 file with :meth:`save_fields` / :func:`meow.eme.io.save_fields`.
        """
        from meow.eme.io import modes_to_dataset

        return modes_to_dataset(self.result(), attrs=attrs)

    def save_fields(
        self,
        path: str | Path,
        *,
        complevel: int = 4,
        attrs: dict[str, Any] | None = None,
    ) -> Path:
        """Save the full per-cell mode fields to a compressed HDF5 file.

        Collects the modes (blocking until the jobs finish) and writes their
        complete ``Ex..Hz`` fields and effective indices to a single gzip-
        compressed netCDF (HDF5) dataset at ``path`` (see
        :func:`meow.eme.io.save_fields`).
        """
        from meow.eme.io import save_fields

        return save_fields(self.to_dataset(attrs=attrs), path, complevel=complevel)


def submit_cell_modes(
    cells: list[Cell],
    env: Environment,
    *,
    executor: JobExecutor,
    num_modes: int = 10,
    compute_modes_kwargs: dict[str, Any] | None = None,
    compute_modes: Callable | None = None,
) -> ParallelFieldModeJobs:
    """Submit one mode-solve job per cell, keeping the full mode fields.

    The single-cell decomposition used when the complete mode fields must be
    saved for later analysis (field reconstruction / propagation), as opposed
    to the field-free slice-group decomposition of
    :func:`submit_s_matrix_parallel`. Each cell is solved as its own concurrent
    job (e.g. a slurm task) and the returned :class:`ParallelFieldModeJobs`
    handle is picklable for multi-session collection.
    """
    if executor is None:
        msg = (
            "submit_cell_modes requires an executor whose jobs persist "
            "(e.g. meow.slurm_executor(...))."
        )
        raise ValueError(msg)
    env_data = env.model_dump()
    jobs = [
        executor.submit(
            compute_cell_modes,
            cell.model_dump(),
            env_data,
            num_modes,
            compute_modes_kwargs,
            compute_modes,
        )
        for cell in cells
    ]
    return ParallelFieldModeJobs(jobs=jobs, cells=cells, env=env)


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
    compute_modes: Callable | None = None,
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
            compute_modes,
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
