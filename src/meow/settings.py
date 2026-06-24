"""Environment-driven solver backend, parallel/slurm resource and resolution settings.

These knobs were originally defined in the paper examples (``_backends.py`` and
``_resolution.py``); they are part of the main library so any project - not just
the bundled examples - can resolve a mode-solver backend and its parallel/slurm
resources from a single, consistent set of environment variables.

Backend selection (:func:`backend_name` / :func:`resolve_backend`):
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

Parallel / slurm resources (env-overridable):
    - ``MEOW_CPUS_PER_TASK`` (:func:`cpus_per_task`) - cpus per slurm task and
      the local worker count;
    - ``MEOW_TIMEOUT_MIN`` (:func:`timeout_min`) - per-job wall-clock limit;
    - ``MEOW_SLURM_PARTITION`` (:func:`slurm_partition`);
    - ``MEOW_SLURM_CLUSTER`` (:func:`slurm_cluster`) - submitit cluster selector;
    - ``MEOW_MAX_WORKERS`` (:func:`max_workers`) - local worker count override;
    - ``MEOW_PAPER_PARALLEL`` (:func:`parallel_enabled`) - use the parallel
      slice-group engine instead of the serial path.

Resolution presets (:func:`level` / :func:`pick` / :func:`num_cells` /
:func:`num_modes`), a three-level ``MEOW_EXAMPLE_RES`` knob in
``{"low", "medium", "high"}``:

- **low**: a coarse smoke-test resolution (what ``MEOW_EXAMPLE_FAST=1`` selects);
- **medium** (default): the standard full-quality settings;
- **high**: finer mesh, more modes and more EME cells - converged (slow).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

# ==========================================================================
# mode-solver backend selection
# ==========================================================================
_BACKEND_NAMES = ("tidy3d", "mpb", "lumerical")


def _backends() -> dict[str, Callable]:
    """The known ``compute_modes`` backends (imported lazily to avoid cycles)."""
    from meow.fde import (
        compute_modes_lumerical,
        compute_modes_mpb,
        compute_modes_tidy3d,
    )

    return {
        "tidy3d": compute_modes_tidy3d,
        "mpb": compute_modes_mpb,
        "lumerical": compute_modes_lumerical,
    }


def resolve_backend(backend: str | Callable | None = None) -> Callable:
    """Resolve a mode-solver backend to a ``compute_modes`` callable.

    Args:
        backend: a ``compute_modes``-compatible callable, one of ``"tidy3d"``,
            ``"mpb"`` or ``"lumerical"``, or ``None`` to read
            ``MEOW_PAPER_BACKEND`` (defaulting to ``"tidy3d"``).

    Returns:
        The selected ``compute_modes`` callable.
    """
    if callable(backend):
        return backend
    name = backend or os.environ.get("MEOW_PAPER_BACKEND", "tidy3d")
    try:
        return _backends()[name]
    except KeyError:
        msg = f"Unknown backend {name!r}; choose from {sorted(_BACKEND_NAMES)}."
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


# ==========================================================================
# parallel / slurm resource settings (env-overridable)
# ==========================================================================
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
    cells: list[Any],
    env: Any,
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
    import meow as mw

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


# ==========================================================================
# resolution presets (MEOW_EXAMPLE_RES in {low, medium, high})
# ==========================================================================
LEVELS = ("low", "medium", "high")
DEFAULT_LEVEL = "medium"

# The converged ("high") standard values for the two main EME knobs; these are
# also the default ``num_cells`` / ``num_modes`` of the example functions.
HIGH_NUM_CELLS = 128
HIGH_NUM_MODES = 8


def level() -> str:
    """The active resolution level (from ``MEOW_EXAMPLE_RES``; default medium).

    Accepts the full names or a unique prefix (``l``/``m``/``h``). Falls back to
    ``low`` when the legacy ``MEOW_EXAMPLE_FAST`` is set, else to
    :data:`DEFAULT_LEVEL`.
    """
    val = os.environ.get("MEOW_EXAMPLE_RES")
    if val:
        v = val.strip().lower()
        for lvl in LEVELS:
            if lvl == v or lvl.startswith(v):
                return lvl
        msg = f"MEOW_EXAMPLE_RES must be one of {LEVELS!r}, got {val!r}."
        raise ValueError(msg)
    if os.environ.get("MEOW_EXAMPLE_FAST", "0") not in ("0", "", "false", "False"):
        return "low"
    return DEFAULT_LEVEL


def pick(*, low: T, medium: T, high: T) -> T:
    """Return the value for the active resolution level."""
    return {"low": low, "medium": medium, "high": high}[level()]


def is_low() -> bool:
    """Whether the active level is the coarse ``low`` (smoke-test) resolution."""
    return level() == "low"


def num_cells(*, low: int, medium: int, high: int = HIGH_NUM_CELLS) -> int:
    """Number of EME cells for the active level (env ``MEOW_NUM_CELLS`` wins).

    When ``MEOW_NUM_CELLS`` is set it overrides the resolution-derived value
    (including the converged ``high`` standard of :data:`HIGH_NUM_CELLS`).
    """
    override = os.environ.get("MEOW_NUM_CELLS")
    return int(override) if override else pick(low=low, medium=medium, high=high)


def num_modes(*, low: int, medium: int, high: int = HIGH_NUM_MODES) -> int:
    """Number of modes per cross-section for the active level.

    When ``MEOW_NUM_MODES`` is set it overrides the resolution-derived value
    (including the converged ``high`` standard of :data:`HIGH_NUM_MODES`).
    """
    override = os.environ.get("MEOW_NUM_MODES")
    return int(override) if override else pick(low=low, medium=medium, high=high)
