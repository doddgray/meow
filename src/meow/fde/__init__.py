"""FDE Implementations & Backends."""

from __future__ import annotations

from meow.fde.default import (
    compute_modes,
)
from meow.fde.dispersion import (
    ModeMetrics,
    dispersion_metrics,
    effective_area,
    solve_mode,
)
from meow.fde.lumerical import (
    Sim,
    compute_modes_lumerical,
    create_lumerical_geometries,
    get_sim,
)
from meow.fde.mpb import (
    compute_modes_mpb,
)
from meow.fde.post_process import (
    filter_modes,
    normalize_modes,
    orthonormalize_modes,
    post_process_modes,
)
from meow.fde.tidy3d import (
    compute_modes_tidy3d,
)

__all__ = [
    "ModeMetrics",
    "Sim",
    "compute_modes",
    "compute_modes_lumerical",
    "compute_modes_mpb",
    "compute_modes_tidy3d",
    "create_lumerical_geometries",
    "dispersion_metrics",
    "effective_area",
    "filter_modes",
    "get_sim",
    "normalize_modes",
    "orthonormalize_modes",
    "post_process_modes",
    "solve_mode",
]
