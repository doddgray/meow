"""Backwards-compatible shim - the backend/resource settings now live in meow.

The mode-solver backend selection and the parallel/slurm resource settings the
examples used to define here are now part of the main library, in
:mod:`meow.settings` (so any project can share the same ``MEOW_*`` environment
variables). This module just re-exports them under their historical names.
"""

from __future__ import annotations

import meow as mw
from meow.settings import (
    backend_name,
    cpus_per_task,
    device_s_matrix,
    max_workers,
    parallel_enabled,
    resolve_backend,
    slurm_cluster,
    slurm_partition,
    timeout_min,
)

# the known backends, kept for backwards compatibility (see meow.settings)
BACKENDS = {
    "tidy3d": mw.compute_modes_tidy3d,
    "mpb": mw.compute_modes_mpb,
    "lumerical": mw.compute_modes_lumerical,
}

__all__ = [
    "BACKENDS",
    "backend_name",
    "cpus_per_task",
    "device_s_matrix",
    "max_workers",
    "parallel_enabled",
    "resolve_backend",
    "slurm_cluster",
    "slurm_partition",
    "timeout_min",
]
