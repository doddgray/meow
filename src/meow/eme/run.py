"""Submitting and *reloading* distributed (slurm) EME runs across sessions.

The distributed EME workflow supports a **two-session pattern**: submit the EME
jobs to the cluster in one python session, then collect the results in a *later*
session - even after the submitting process has exited. This module holds the
machinery that makes that work on top of :func:`meow.submit_s_matrix_parallel` /
:class:`meow.ParallelEMEJobs`.

Two granularities are provided:

- :class:`SavedEME` - a lightweight picklable record bundling a
  :class:`meow.ParallelEMEJobs` handle with a few example-specific scalars
  (``meta``), enough to turn the cascaded S-matrix into a figure of merit. Use
  :func:`submit_eme` / :func:`load_records` for the just-the-port-powers path.
- :class:`AnalysisRun` - a richer base class for a *distributed analysis run*
  that assembles spectra/fields and writes figures, GDS and data into a
  timestamped output folder. Subclass it (e.g. for a specific device) and use
  :func:`start_run` / :func:`load_runs` to submit and reload them.

Only the **job handles** are pickled (their live submitit job objects cannot be
represented otherwise); the computed *result data* is saved with the xarray /
CSV / JSON helpers in :mod:`meow.eme.io`. Because the handles reattach to the
persisted submitit jobs on unpickling, a later session only needs the shared
job folder - no live handle from the submitting process.
"""

from __future__ import annotations

import pickle
import re
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from meow.cell import Cell
    from meow.eme.parallel import ParallelEMEJobs
    from meow.environment import Environment

# the per-record filename suffix written into the job folder
RECORD_SUFFIX = ".eme.pkl"
# the per-run analysis-record filename written into each timestamped run folder
RUN_RECORD = "run.pkl"


def safe_label(label: str) -> str:
    """A filesystem-safe stem for a record label."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")


# kept private alias for backwards compatibility with the examples
_safe = safe_label


# ==========================================================================
# lightweight S-matrix-only records (one ``<label>.eme.pkl`` per device)
# ==========================================================================
@dataclass
class SavedEME:
    """A submitted parallel EME plus the scalars needed to finish it later.

    Picklable, so it can be written to the shared job folder right after
    submission and reloaded in a later python session to collect the result.

    Attributes:
        jobs: the :class:`meow.ParallelEMEJobs` handle over the submitted jobs.
        label: a human-readable key (also the record's filename stem).
        num_modes: number of modes per cell (needed to re-attribute the ports).
        meta: example-specific scalars used to compute the figures of merit
            from the cascaded S-matrix (kept tiny and picklable on purpose).
    """

    jobs: ParallelEMEJobs
    label: str
    num_modes: int
    meta: dict[str, Any] = field(default_factory=dict)

    def path(self, folder: str | Path) -> Path:
        """The record's path inside ``folder``."""
        return Path(folder) / f"{safe_label(self.label)}{RECORD_SUFFIX}"

    def save(self, folder: str | Path) -> Path:
        """Write this record into ``folder`` as ``<label>.eme.pkl``."""
        path = self.path(folder)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: str | Path) -> SavedEME:
        """Load a record written by :meth:`save`."""
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            msg = f"{path} is not a {cls.__name__}."
            raise TypeError(msg)
        return obj


def submit_eme(
    label: str,
    cells: list[Cell],
    env: Environment,
    *,
    executor: Any,
    num_modes: int,
    folder: str | Path,
    meta: dict[str, Any] | None = None,
    **submit_kwargs: Any,
) -> SavedEME:
    """Submit one device EME (non-blocking) and persist it into ``folder``.

    Wraps :func:`meow.submit_s_matrix_parallel` and immediately writes a
    :class:`SavedEME` record next to the submitit jobs, so a later session can
    reload it from the shared folder.
    """
    from meow.eme.parallel import submit_s_matrix_parallel

    jobs = submit_s_matrix_parallel(
        cells, env, executor=executor, num_modes=num_modes, **submit_kwargs
    )
    record = SavedEME(jobs=jobs, label=label, num_modes=num_modes, meta=meta or {})
    record.save(folder)
    return record


def record_path(folder: str | Path, label: str) -> Path:
    """The path a record with ``label`` is saved to inside ``folder``."""
    return Path(folder) / f"{safe_label(label)}{RECORD_SUFFIX}"


def load_record(folder: str | Path, label: str) -> SavedEME:
    """Load the single :class:`SavedEME` record with ``label`` from ``folder``."""
    return SavedEME.load(record_path(folder, label))


def load_records(folder: str | Path) -> list[SavedEME]:
    """Load every :class:`SavedEME` record persisted in ``folder`` (sorted)."""
    return [SavedEME.load(p) for p in sorted(Path(folder).glob(f"*{RECORD_SUFFIX}"))]


# ==========================================================================
# distributed analysis runs (timestamped output folders)
# ==========================================================================
@dataclass
class AnalysisRun:
    """A submitted, distributed analysis run (base class).

    Holds the picklable design ``spec`` and ``settings`` plus the submitted EME
    job handles (added by subclasses). It is pickled into ``<out_dir>/run.pkl``
    by :meth:`save`; a later session reloads it and calls :meth:`gather` to
    collect the distributed EME results, assemble the spectra/fields and write
    the figures, GDS and data into ``out_dir`` (using the xarray / CSV / JSON
    helpers in :mod:`meow.eme.io`).

    Subclasses provide :meth:`handles` (the submitted job handles) and
    :meth:`gather` (collect + assemble + write the outputs).
    """

    spec: dict
    settings: dict
    label: str
    out_dir: str
    save_fields: bool

    @property
    def stem(self) -> str:
        """Filesystem-safe stem for this run's output filenames."""
        return self.settings.get("file_stem") or safe_label(self.label)

    def handles(self) -> list[Any]:
        """All submitted job handles (overridden by subclasses)."""
        raise NotImplementedError

    @property
    def job_ids(self) -> list[str]:
        """The submitit job ids of every handle in this run."""
        return [jid for h in self.handles() for jid in h.job_ids]

    def done(self) -> bool:
        """Whether every submitted job of this run has finished (never blocks)."""
        return all(h.done() for h in self.handles())

    def save(self) -> Path:
        """Pickle this run to ``<out_dir>/run.pkl`` for a later-session gather."""
        path = Path(self.out_dir) / RUN_RECORD
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        return path

    def gather(self) -> dict:
        """Collect the distributed results and write the outputs (subclassed)."""
        raise NotImplementedError

    async def agather(self) -> dict:
        """Async :meth:`gather` (collects + plots off the event loop)."""
        import asyncio

        return await asyncio.to_thread(self.gather)


def make_run_dir(folder: str | Path, label: str) -> Path:
    """Create and return a fresh timestamped subfolder of ``folder`` for a run.

    Each submitted analysis job gets its own ``<timestamp>-<label>`` directory
    (under the jobs folder) into which all of its data, plots, GDS and the
    submitit job logs are written.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(folder) / f"{stamp}-{safe_label(label)}"
    # disambiguate if two runs land in the same second
    n = 1
    base = run_dir
    while run_dir.exists():
        run_dir = base.with_name(f"{base.name}-{n}")
        n += 1
    run_dir.mkdir(parents=True)
    return run_dir


def start_run(
    submit_fn: Any,
    spec: dict,
    settings: dict,
    *,
    folder: str | Path,
    label: str,
    executor_factory: Any,
    save_fields: bool | None = None,
) -> Any:
    """Create a timestamped run folder, submit the distributed EME and persist it.

    ``submit_fn(spec, settings, executor_factory=..., out_dir=run_dir,
    save_fields=...)`` submits the design's EME as concurrent jobs into a fresh
    ``<timestamp>-<label>`` run folder and returns an :class:`AnalysisRun`; this
    saves it to ``<run_dir>/run.pkl`` so a later session can reload and
    :meth:`~AnalysisRun.gather` it. ``executor_factory(dir)`` builds a
    :func:`meow.slurm_executor` rooted at ``dir`` (under the run folder, so the
    submitit logs are co-located); it is wrapped to accept a per-job-group name.
    """
    run_dir = make_run_dir(folder, label)
    settings = {**settings, "label": label, "file_stem": safe_label(label)}

    def named_executor(name: str) -> Any:
        return executor_factory(run_dir / "submitit" / name)

    run = submit_fn(
        spec,
        settings,
        executor_factory=named_executor,
        out_dir=run_dir,
        save_fields=save_fields,
    )
    run.save()
    return run


def load_run(run_path: str | Path) -> Any:
    """Load a single analysis run from a specific run/job directory (or its file).

    ``run_path`` may be the run's timestamped directory (its ``run.pkl`` is
    loaded) or the ``run.pkl`` file itself. Use this to reattach to a *specific*
    run rather than the most-recent / all matching runs of :func:`load_runs`.
    """
    path = Path(run_path)
    if path.is_dir():
        path = path / RUN_RECORD
    with path.open("rb") as f:
        return pickle.load(f)


def load_runs(folder: str | Path) -> list[Any]:
    """Load every analysis run persisted under ``folder`` (recursively).

    Each distributed analysis run pickles itself to ``<run_dir>/run.pkl`` (see
    :class:`AnalysisRun`); this globs all of them under ``folder`` and unpickles
    them, so a later session can reattach to the submitted jobs and
    :meth:`~AnalysisRun.gather` the results. Records that fail to unpickle (e.g.
    corrupt, or written by an incompatible older version) are skipped with a
    warning rather than crashing the whole load.
    """
    runs: list[Any] = []
    for path in sorted(Path(folder).glob(f"**/{RUN_RECORD}")):
        try:
            with path.open("rb") as f:
                runs.append(pickle.load(f))
        except Exception as e:  # noqa: BLE001 - skip corrupt/incompatible records
            warnings.warn(
                f"skipping unreadable run record {path}: {e!r}", stacklevel=2
            )
    return runs


def cli_main(module: str, commands: dict[str, Any]) -> None:
    """Tiny ``run | submit | gather`` dispatcher shared by the slurm examples.

    Reads the subcommand from ``sys.argv`` (default ``run``) and prints the
    selected entry point's result as JSON.
    """
    import json
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd not in commands:
        usage = "|".join(commands)
        msg = f"usage: python -m {module} [{usage}]"
        raise SystemExit(msg)
    print(json.dumps(commands[cmd](), indent=2, default=str))  # noqa: T201
