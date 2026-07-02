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
from meow.fde.sparse import (
    EigenvectorAdjoint,
    ScalarModeSolution,
    eigenvector_sensitivity,
    scalar_neffs,
    scalar_operator,
    solve_scalar_modes,
    solve_scalar_modes_full,
)
from meow.fde.tidy3d import (
    compute_modes_tidy3d,
)

__all__ = [
    "EigenvectorAdjoint",
    "ModeMetrics",
    "ScalarModeSolution",
    "Sim",
    "compute_modes",
    "compute_modes_lumerical",
    "compute_modes_mpb",
    "compute_modes_tidy3d",
    "create_lumerical_geometries",
    "dispersion_metrics",
    "effective_area",
    "eigenvector_sensitivity",
    "filter_modes",
    "get_sim",
    "normalize_modes",
    "orthonormalize_modes",
    "post_process_modes",
    "scalar_neffs",
    "scalar_operator",
    "solve_mode",
    "solve_scalar_modes",
    "solve_scalar_modes_full",
]
