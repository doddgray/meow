"""Mode-solver backend selection and parallel-EME helpers for the examples.

Both paper-reproduction examples can run with any of meow's FDE backends and
can compute their EME S-matrices either serially or with the parallel
slice-group engine. The selection is centralized here so the example modules
and their figure scripts share one consistent mechanism.

Backend selection (``resolve_backend``):
    - pass a callable (any ``compute_modes``-compatible function), or
    - pass a name: ``"tidy3d"`` (default), ``"mpb"`` or ``"lumerical"``, or
    - leave it unset to read the ``MEOW_PAPER_BACKEND`` environment variable.

Parallelism (``parallel_enabled`` / ``device_s_matrix``):
    - the EME S-matrix of a stack of cells can be cascaded with
      :func:`meow.compute_s_matrix_parallel` (concurrent slice-group jobs)
      instead of the serial path, toggled by a flag or the
      ``MEOW_PAPER_PARALLEL`` environment variable.

Note: the parallel slice-group engine re-solves shared boundary cells in
separate worker processes and checks that they return identical effective
indices, which requires a *deterministic* mode solver. Both the tidy3d backend
and the (deterministically seeded) MPB backend satisfy this, and
``meow.compute_s_matrix_parallel`` accepts a ``compute_modes`` backend. For
simplicity this example helper still runs the parallel path with the default
tidy3d backend; pass a backend to :func:`meow.compute_s_matrix_parallel`
directly to parallelize with MPB.
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


def parallel_enabled(parallel: bool | None = None) -> bool:  # noqa: FBT001
    """Whether to use the parallel EME engine.

    Args:
        parallel: explicit override; if ``None``, reads the
            ``MEOW_PAPER_PARALLEL`` environment variable (default off).
    """
    if parallel is not None:
        return parallel
    return bool(int(os.environ.get("MEOW_PAPER_PARALLEL", "0")))


def device_s_matrix(
    cells: list[mw.Cell],
    env: mw.Environment,
    *,
    num_modes: int = 4,
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
            :func:`parallel_enabled`). The parallel path always uses the
            deterministic default (tidy3d) backend and ignores
            ``compute_modes``.
        compute_modes: backend used for the serial path (default: tidy3d).
        **parallel_kwargs: forwarded to
            :func:`meow.compute_s_matrix_parallel`.

    Returns:
        ``(S, port_map)`` in SAX dense multimode format.
    """
    if parallel_enabled(parallel):
        return mw.compute_s_matrix_parallel(
            cells, env, num_modes=num_modes, **parallel_kwargs
        )
    compute_modes = compute_modes or mw.compute_modes
    css = [mw.CrossSection.from_cell(cell=c, env=env) for c in cells]
    modes = [compute_modes(cs, num_modes=num_modes) for cs in css]
    return mw.compute_s_matrix(modes, cells=cells)
