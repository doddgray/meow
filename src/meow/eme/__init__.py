"""SAX EME."""

from meow.eme.cascade import (
    cascade_s_matrices,
    compute_s_matrix_sax,
    downselect_s,
)
from meow.eme.default import (
    compute_s_matrix,
)
from meow.eme.interface import (
    PassivityMethod,
    compute_interface_s_matrices,
    compute_interface_s_matrix,
    enforce_passivity,
    overlap_matrix,
)
from meow.eme.parallel import (
    GroupResult,
    JobExecutor,
    acompute_s_matrix_parallel,
    chunk_cell_indices,
    compute_group_result,
    compute_s_matrix_parallel,
    slurm_executor,
)
from meow.eme.propagation import (
    compute_mode_amplitudes,
    compute_propagation_s_matrices,
    compute_propagation_s_matrix,
    l2r_matrices,
    pi_pairs,
    plot_fields,
    propagate,
    propagate_modes,
    r2l_matrices,
    select_ports,
    split_square_matrix,
    track_modes,
)
from meow.eme.solve import (
    tsvd_solve,
)

__all__ = [
    "GroupResult",
    "JobExecutor",
    "PassivityMethod",
    "acompute_s_matrix_parallel",
    "cascade_s_matrices",
    "chunk_cell_indices",
    "compute_group_result",
    "compute_interface_s_matrices",
    "compute_interface_s_matrix",
    "compute_mode_amplitudes",
    "compute_propagation_s_matrices",
    "compute_propagation_s_matrix",
    "compute_s_matrix",
    "compute_s_matrix_parallel",
    "compute_s_matrix_sax",
    "downselect_s",
    "enforce_passivity",
    "l2r_matrices",
    "overlap_matrix",
    "pi_pairs",
    "plot_fields",
    "propagate",
    "propagate_modes",
    "r2l_matrices",
    "select_ports",
    "slurm_executor",
    "split_square_matrix",
    "track_modes",
    "tsvd_solve",
]
