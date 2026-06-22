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
    GroupSpectrumResult,
    JobExecutor,
    ParallelEMEJobs,
    acompute_s_matrix_parallel,
    acompute_s_matrix_spectrum,
    chunk_cell_indices,
    compute_group_result,
    compute_group_spectrum,
    compute_s_matrix_parallel,
    compute_s_matrix_spectrum,
    slurm_executor,
    submit_s_matrix_parallel,
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
    "GroupSpectrumResult",
    "JobExecutor",
    "ParallelEMEJobs",
    "PassivityMethod",
    "acompute_s_matrix_parallel",
    "acompute_s_matrix_spectrum",
    "cascade_s_matrices",
    "chunk_cell_indices",
    "compute_group_result",
    "compute_group_spectrum",
    "compute_interface_s_matrices",
    "compute_interface_s_matrix",
    "compute_mode_amplitudes",
    "compute_propagation_s_matrices",
    "compute_propagation_s_matrix",
    "compute_s_matrix",
    "compute_s_matrix_parallel",
    "compute_s_matrix_sax",
    "compute_s_matrix_spectrum",
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
    "submit_s_matrix_parallel",
    "track_modes",
    "tsvd_solve",
]
