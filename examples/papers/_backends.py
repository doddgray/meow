"""Mode-solver backend + parallel/slurm resource settings for the examples.

All the paper examples resolve their FDE backend and their parallel/slurm
resource settings from environment variables; the logic is centralized here so
every example (and the figure scripts) shares one consistent mechanism.

Backend selection (``backend_name`` / ``resolve_backend``):
    - pass a callable (any ``compute_modes``-compatible function), or
    - pass a name: ``"tidy3d"`` (default), ``"mpb"`` or ``"lumerical"``, or
    - leave it unset to read the ``MEOW_PAPER_BACKEND`` environment variable.

  The resolved backend is threaded all the way through to the parallel
  slice-group jobs and the single-cell field jobs (via the ``compute_modes``
  argument of :func:`meow.compute_s_matrix_parallel` /
  :func:`meow.submit_s_matrix_spectrum` / :func:`meow.submit_cell_modes`), so the
  chosen solver is used both for local runs and for slurm jobs. The slice-group
  cascade re-solves shared boundary cells in separate workers and checks they
  agree, which needs a *deterministic* solver - tidy3d or the seeded MPB
  backend (single-cell field jobs have no shared cells, so any backend works).

Parallel / slurm resources (env-overridable, shared by every example):
    - ``MEOW_CPUS_PER_TASK`` (:func:`cpus_per_task`) - cpus per slurm task and
      the local worker count;
    - ``MEOW_TIMEOUT_MIN`` (:func:`timeout_min`) - per-job wall-clock limit;
    - ``MEOW_SLURM_PARTITION`` (:func:`slurm_partition`);
    - ``MEOW_SLURM_CLUSTER`` (:func:`slurm_cluster`) - submitit cluster selector;
    - ``MEOW_MAX_WORKERS`` (:func:`max_workers`) - local worker count override;
    - ``MEOW_PAPER_PARALLEL`` (:func:`parallel_enabled`) - use the parallel
      slice-group engine in the figure scripts instead of the serial path.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import meow as mw

BACKENDS: dict[str, Callable] = {
    "tidy3d": mw.compute_modes_tidy3d,
    "mpb": mw.compute_modes_mpb,
    "lumerical": mw.compute_modes_lumerical,
}


def resolve_backend(backend: str | Callable | None = None) -> Callable:
    """Resolve a mode-solver backend to a ``compute_modes`` callable.

    Args:
        backend: a ``compute_modes``-compatible callable, one of the names in
            :data:`BACKENDS`, or ``None`` to read ``MEOW_PAPER_BACKEND``
            (defaulting to ``"tidy3d"``).

    Returns:
        The selected ``compute_modes`` callable.
    """
    if callable(backend):
        return backend
    name = backend or os.environ.get("MEOW_PAPER_BACKEND", "tidy3d")
    try:
        return BACKENDS[name]
    except KeyError:
        msg = f"Unknown backend {name!r}; choose from {sorted(BACKENDS)}."
        raise ValueError(msg) from None


def backend_name(name: str | None = None) -> str:
    """The active backend name (``MEOW_PAPER_BACKEND``; default ``tidy3d``)."""
    return name or os.environ.get("MEOW_PAPER_BACKEND", "tidy3d")


def parallel_enabled(parallel: bool | None = None) -> bool:  # noqa: FBT001
    """Whether to use the parallel EME engine.

    Args:
        parallel: explicit override; if ``None``, reads the
            ``MEOW_PAPER_PARALLEL`` environment variable (default off).
    """
    if parallel is not None:
        return parallel
    return bool(int(os.environ.get("MEOW_PAPER_PARALLEL", "0")))


# --------------------------------------------------------------------------
# parallel / slurm resource settings (shared by every example, env-overridable)
# --------------------------------------------------------------------------
def cpus_per_task(default: int = 2) -> int:
    """CPUs per parallel task (env ``MEOW_CPUS_PER_TASK``).

    Used for both the slurm jobs (submitit ``cpus_per_task``) and, as the worker
    count, for the local multithreaded / multiprocess parallel runs.
    """
    return int(os.environ.get("MEOW_CPUS_PER_TASK", default))


def timeout_min(default: int = 60) -> int:
    """Per-job wall-clock limit in minutes (env ``MEOW_TIMEOUT_MIN``)."""
    return int(os.environ.get("MEOW_TIMEOUT_MIN", default))


def slurm_partition() -> str | None:
    """The slurm partition to submit to (env ``MEOW_SLURM_PARTITION``)."""
    return os.environ.get("MEOW_SLURM_PARTITION")


def slurm_cluster(default: str = "local") -> str | None:
    """The submitit cluster selector (env ``MEOW_SLURM_CLUSTER``; default local)."""
    return os.environ.get("MEOW_SLURM_CLUSTER", default)


def max_workers() -> int | None:
    """Worker count for local parallel runs (``MEOW_MAX_WORKERS`` or cpus/task).

    Falls back to ``MEOW_CPUS_PER_TASK`` when set, else ``None`` (the engine's
    own default).
    """
    explicit = os.environ.get("MEOW_MAX_WORKERS")
    if explicit:
        return int(explicit)
    cpus = os.environ.get("MEOW_CPUS_PER_TASK")
    return int(cpus) if cpus else None


def device_s_matrix(
    cells: list[mw.Cell],
    env: mw.Environment,
    *,
    num_modes: int = 8,
    parallel: bool | None = None,
    compute_modes: Callable | None = None,
    **parallel_kwargs: Any,
) -> tuple[Any, dict[str, int]]:
    """Compute the EME S-matrix of a cell stack, serially or in parallel.

    Args:
        cells: the EME cells (in chain order).
        env: the simulation environment.
        num_modes: number of modes per cell.
        parallel: use the parallel slice-group engine (see
            :func:`parallel_enabled`).
        compute_modes: the FDE backend to use (default: the resolved
            ``MEOW_PAPER_BACKEND``). It is threaded through to the parallel
            slice-group jobs as well, so the chosen backend (tidy3d / mpb /
            lumerical) is used both serially and in parallel; the slice-group
            method needs a *deterministic* backend (tidy3d or seeded mpb).
        **parallel_kwargs: forwarded to
            :func:`meow.compute_s_matrix_parallel` (e.g. ``executor``).

    Returns:
        ``(S, port_map)`` in SAX dense multimode format.
    """
    compute_modes = resolve_backend(compute_modes)
    if parallel_enabled(parallel):
        parallel_kwargs.setdefault("max_workers", max_workers())
        return mw.compute_s_matrix_parallel(
            cells, env, num_modes=num_modes, compute_modes=compute_modes,
            **parallel_kwargs,
        )
    css = [mw.CrossSection.from_cell(cell=c, env=env) for c in cells]
    modes = [compute_modes(cs, num_modes=num_modes) for cs in css]
    return mw.compute_s_matrix(modes, cells=cells)
