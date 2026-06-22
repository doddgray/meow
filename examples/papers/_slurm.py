"""Shared helpers for submitting and *reloading* slurm-based EME results.

The slurm examples (``dichroic_designer_slurm``, ``kwolek_designer_slurm`` and
``dichroic_coupler_slurm``) all support a **two-session workflow**: submit the
EME jobs to the cluster in one python session, then collect the results in a
*later* session - even after the submitting process has exited. This module
holds the small amount of machinery they share to make that work, on top of
:func:`meow.submit_s_matrix_parallel` / :class:`meow.ParallelEMEJobs`.

The key object is :class:`SavedEME`: a picklable record bundling

- a :class:`meow.ParallelEMEJobs` handle (the submitted slice-group jobs plus
  the cells/environment needed to cascade their results), and
- a few example-specific scalars (``meta``) needed to turn the cascaded
  S-matrix into the example's figures of merit (e.g. the lateral split that
  separates the short-/long-pass ports).

``submit_eme`` writes one ``<label>.eme.pkl`` record into the job folder per
EME; ``load_records`` reads them all back in a later session. Because
:class:`meow.ParallelEMEJobs` reattaches to the persisted submitit jobs on
unpickling, the later session only needs the shared job folder - no live
handle from the submitting process.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import meow as mw

# the per-record filename suffix written into the job folder
RECORD_SUFFIX = ".eme.pkl"
# the per-run analysis-record filename written into each timestamped run folder
RUN_RECORD = "run.pkl"


def _safe(label: str) -> str:
    """A filesystem-safe stem for a record label."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")


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
            from the cascaded S-matrix (kept tiny and picklable on purpose -
            no gdsfactory ``Component`` or other heavyweight objects).
    """

    jobs: mw.ParallelEMEJobs
    label: str
    num_modes: int
    meta: dict[str, Any] = field(default_factory=dict)

    def path(self, folder: str | Path) -> Path:
        """The record's path inside ``folder``."""
        return Path(folder) / f"{_safe(self.label)}{RECORD_SUFFIX}"

    def save(self, folder: str | Path) -> Path:
        """Write this record into ``folder`` as ``<label>.eme.pkl``."""
        path = self.path(folder)
        path.parent.mkdir(parents=True, exist_ok=True)
        import pickle

        with path.open("wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: str | Path) -> SavedEME:
        """Load a record written by :meth:`save`."""
        import pickle

        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            msg = f"{path} is not a {cls.__name__}."
            raise TypeError(msg)
        return obj


def submit_eme(
    label: str,
    cells: list[mw.Cell],
    env: mw.Environment,
    *,
    executor: Any,
    num_modes: int,
    folder: str | Path,
    meta: dict[str, Any] | None = None,
    **submit_kwargs: Any,
) -> SavedEME:
    """Submit one device EME (non-blocking) and persist it into ``folder``.

    Wraps :func:`meow.submit_s_matrix_parallel` and immediately writes a
    :class:`SavedEME` record next to the submitit jobs, so a later session
    can reload it from the shared folder.
    """
    jobs = mw.submit_s_matrix_parallel(
        cells, env, executor=executor, num_modes=num_modes, **submit_kwargs
    )
    record = SavedEME(jobs=jobs, label=label, num_modes=num_modes, meta=meta or {})
    record.save(folder)
    return record


def record_path(folder: str | Path, label: str) -> Path:
    """The path a record with ``label`` is saved to inside ``folder``."""
    return Path(folder) / f"{_safe(label)}{RECORD_SUFFIX}"


def load_record(folder: str | Path, label: str) -> SavedEME:
    """Load the single :class:`SavedEME` record with ``label`` from ``folder``."""
    return SavedEME.load(record_path(folder, label))


def load_records(folder: str | Path) -> list[SavedEME]:
    """Load every :class:`SavedEME` record persisted in ``folder`` (sorted)."""
    return [SavedEME.load(p) for p in sorted(Path(folder).glob(f"*{RECORD_SUFFIX}"))]


# ==========================================================================
# coarse-grained "analyze one design" runs (timestamped output folders)
# ==========================================================================
def make_run_dir(folder: str | Path, label: str) -> Path:
    """Create and return a fresh timestamped subfolder of ``folder`` for a run.

    Each submitted analysis job gets its own ``<timestamp>-<label>`` directory
    (under the MEOW jobs folder) into which all of its data, plots, GDS and the
    submitit job logs are written.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(folder) / f"{stamp}-{_safe(label)}"
    # disambiguate if two runs land in the same second
    n = 1
    base = run_dir
    while run_dir.exists():
        run_dir = base.with_name(f"{base.name}-{n}")
        n += 1
    run_dir.mkdir(parents=True)
    return run_dir


@dataclass
class SavedRun:
    """A submitted analysis job (returning a summary dict) plus its location.

    Picklable, so the submitting session can persist it into the run's
    timestamped folder and a later session can reload it (reattaching to the
    still-running submitit job) to collect the summary. The figures/GDS/data
    are written by the job directly into ``out_dir`` on the shared filesystem.
    """

    job: Any
    label: str
    out_dir: str

    @property
    def job_id(self) -> str:
        return str(getattr(self.job, "job_id", ""))

    def done(self) -> bool:
        is_done = getattr(self.job, "done", None)
        if callable(is_done):
            try:
                return bool(is_done())
            except Exception:  # noqa: BLE001 - a not-yet-known job isn't done
                return False
        return True

    def result(self) -> dict:
        return self.job.result()

    async def aresult(self) -> dict:
        return await asyncio.to_thread(self.job.result)

    def save(self) -> Path:
        path = Path(self.out_dir) / RUN_RECORD
        path.parent.mkdir(parents=True, exist_ok=True)
        import pickle

        with path.open("wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: str | Path) -> SavedRun:
        import pickle

        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            msg = f"{path} is not a {cls.__name__}."
            raise TypeError(msg)
        return obj


def submit_run(
    analyze_fn: Any,
    spec: dict,
    settings: dict,
    *,
    executor_factory: Any,
    folder: str | Path,
    label: str,
) -> SavedRun:
    """Submit one analysis job into a fresh timestamped run folder and persist it.

    A fresh ``<timestamp>-<label>`` run folder is created under ``folder``;
    ``executor_factory(submitit_dir)`` builds the executor (so submitit's own
    logs/payloads land inside the run folder too), and
    ``analyze_fn(out_dir, spec, settings)`` (e.g.
    :func:`examples.papers._analysis.analyze_dichroic`) runs on the cluster and
    writes all its outputs into the run folder. Returns immediately with a
    persisted :class:`SavedRun` handle.
    """
    run_dir = make_run_dir(folder, label)
    executor = executor_factory(run_dir / "submitit")
    # the human label may contain path-unsafe characters (e.g. a "/"); pass a
    # filesystem-safe stem for the output filenames alongside it.
    settings = {**settings, "label": label, "file_stem": _safe(label)}
    job = executor.submit(analyze_fn, str(run_dir), spec, settings)
    record = SavedRun(job=job, label=label, out_dir=str(run_dir))
    record.save()
    return record


def load_runs(folder: str | Path) -> list[SavedRun]:
    """Load every :class:`SavedRun` persisted under ``folder`` (recursively)."""
    return [
        SavedRun.load(p) for p in sorted(Path(folder).glob(f"**/{RUN_RECORD}"))
    ]


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
    print(json.dumps(commands[cmd](), indent=2, default=str))
