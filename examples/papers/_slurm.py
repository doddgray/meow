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
    save_fields=...)`` (e.g.
    :func:`examples.papers._analysis.submit_dichroic_run`) submits the design's
    EME as concurrent jobs into a fresh ``<timestamp>-<label>`` run folder and
    returns a run object; this saves it to ``<run_dir>/run.pkl`` so a later
    session can reload and :meth:`gather` it. ``executor_factory(dir)`` builds a
    :func:`meow.slurm_executor` rooted at ``dir`` (under the run folder, so the
    submitit logs are co-located); it is wrapped to accept a per-job-group name.
    """
    run_dir = make_run_dir(folder, label)
    settings = {**settings, "label": label, "file_stem": _safe(label)}

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


def load_runs(folder: str | Path) -> list[Any]:
    """Load every analysis run persisted under ``folder`` (recursively).

    Each distributed analysis run pickles itself to ``<run_dir>/run.pkl`` (see
    :class:`examples.papers._analysis._Run`); this globs all of them under
    ``folder`` and unpickles them, so a later session can reattach to the
    submitted jobs and :meth:`gather` the results.
    """
    import pickle

    runs: list[Any] = []
    for path in sorted(Path(folder).glob(f"**/{RUN_RECORD}")):
        with path.open("rb") as f:
            runs.append(pickle.load(f))
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
    print(json.dumps(commands[cmd](), indent=2, default=str))
