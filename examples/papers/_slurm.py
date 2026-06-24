"""Backwards-compatible shim - the slurm run orchestration now lives in meow.

The machinery for submitting and *reloading* distributed (slurm) EME runs across
python sessions the examples used to define here is now part of the main library,
in :mod:`meow.eme.run`. This module just re-exports it under its historical
names (including the ``_safe`` private alias the examples referenced).
"""

from __future__ import annotations

from meow.eme.run import (
    RECORD_SUFFIX,
    RUN_RECORD,
    AnalysisRun,
    SavedEME,
    cli_main,
    load_record,
    load_records,
    load_run,
    load_runs,
    make_run_dir,
    record_path,
    safe_label,
    start_run,
    submit_eme,
)

# historical private alias used by the examples
_safe = safe_label

__all__ = [
    "RECORD_SUFFIX",
    "RUN_RECORD",
    "AnalysisRun",
    "SavedEME",
    "cli_main",
    "load_record",
    "load_records",
    "load_run",
    "load_runs",
    "make_run_dir",
    "record_path",
    "safe_label",
    "start_run",
    "submit_eme",
]
