"""Slurm-distributed EME verification of the Ramadan 1998 adiabatic coupler.

Slurm-cluster companion to :mod:`examples.papers.ramadan1998`. It runs the same
Region-II-length sweep (the meow EME analogue of the paper's BPM convergence
study), but distributes each length's full-device EME as concurrent slice-group
jobs via :func:`meow.submit_s_matrix_parallel`, persisting a picklable record per
length so submission and collection can happen in *different* python sessions
(as for the other ``*_slurm`` examples).

- :func:`submit_sweep` builds each length's coupler, submits its EME to the
  cluster and writes one ``<length>.eme.pkl`` record into the shared folder;
- :func:`gather_sweep` (a later session) reattaches to the persisted jobs,
  cascades each device S-matrix, attributes the bar/cross split and writes the
  verification figure + the bar/cross data (CSV + JSON).

Run the in-session demo (jobs as local subprocesses) with::

    python -m examples.papers.ramadan1998_slurm

or split submission and collection across sessions::

    python -m examples.papers.ramadan1998_slurm submit   # session A
    python -m examples.papers.ramadan1998_slurm gather    # session B (later)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

import meow as mw
from examples.papers import _backends, _resolution, _slurm
from examples.papers import ramadan1998 as r

JOB_FOLDER = Path(os.environ.get("MEOW_SLURM_FOLDER", "meow_ramadan_jobs"))


def make_executor(
    folder: Path | str = JOB_FOLDER, cluster: str | None = None
) -> Any:
    """A :func:`meow.slurm_executor` honouring the shared ``MEOW_*`` resources."""
    return mw.slurm_executor(
        folder=str(folder),
        cluster=cluster or _backends.slurm_cluster(),
        timeout_min=_backends.timeout_min(),
        cpus_per_task=_backends.cpus_per_task(),
        slurm_partition=_backends.slurm_partition(),
    )


def submit_sweep(
    lengths_ii: np.ndarray,
    *,
    executor: Any,
    folder: Path | str = JOB_FOLDER,
    wl: float = 1.55,
    num_modes: int = 6,
    res: float = 0.035,
    num_cells: int = 120,
    length_i: float = 20.0,
    length_iii: float = 20.0,
) -> list[_slurm.SavedEME]:
    """Submit one distributed EME per Region-II length; persist the records."""
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    folder = Path(folder)
    env = mw.Environment(wl=wl, T=25.0)
    records = []
    for l_ii in lengths_ii:
        comp = r.adiabatic_coupler_component(
            length_i=length_i, length_ii=float(l_ii), length_iii=length_iii
        )
        cells = r.device_cells(comp, num_cells=num_cells, res=res)
        records.append(
            _slurm.submit_eme(
                f"L_II_{float(l_ii):.0f}um",
                cells,
                env,
                executor=executor,
                num_modes=num_modes,
                folder=folder,
                meta={"length_ii": float(l_ii)},
            )
        )
    return records


def gather_sweep(folder: Path | str = JOB_FOLDER) -> dict[str, Any]:
    """Reload the persisted records, attribute bar/cross, write the figure + data."""
    out = r.FIGDIR / "ramadan1998_slurm"
    out.mkdir(parents=True, exist_ok=True)
    lengths, bar, cross = [], [], []
    for rec in _slurm.load_records(folder):
        s, pm = rec.jobs.result()
        b, x = r.attribute_bar_cross(
            s, pm, rec.jobs.cells[-1], rec.jobs.env, num_modes=rec.num_modes
        )
        lengths.append(float(rec.meta["length_ii"]))
        bar.append(b)
        cross.append(x)
    order = np.argsort(lengths)
    lengths = np.asarray(lengths)[order]
    bar = np.asarray(bar)[order]
    cross = np.asarray(cross)[order]
    r.plot_eme_verification(lengths, bar, cross, out / "eme_verification.png")
    mw.save_table(
        out / "eme_verification",
        {"length_ii_um": lengths, "bar": bar, "cross": cross},
    )
    return {
        "out_dir": str(out),
        "lengths_ii_um": lengths.tolist(),
        "bar": bar.tolist(),
        "cross": cross.tolist(),
    }


def _sweep_lengths() -> np.ndarray:
    return np.linspace(30.0, 300.0, _resolution.pick(low=3, medium=5, high=7))


def submit_main() -> dict[str, Any]:
    """Session A: submit the distributed EME sweep and return immediately."""
    records = submit_sweep(
        _sweep_lengths(),
        executor=make_executor(),
        folder=JOB_FOLDER,
        num_modes=_resolution.num_modes(low=4, medium=6, high=8),
        res=_resolution.pick(low=0.06, medium=0.035, high=0.025),
        num_cells=_resolution.num_cells(low=24, medium=120, high=200),
    )
    return {
        "submitted": {r_.label: r_.jobs.job_ids for r_ in records},
        "folder": str(JOB_FOLDER),
        "next": "run 'gather' in a later session with the same MEOW_SLURM_FOLDER",
    }


def gather_main() -> dict[str, Any]:
    """Session B (later): reattach, collect and write the verification figure."""
    return gather_sweep(JOB_FOLDER)


def main() -> dict[str, Any]:
    """In-session demo: submit the sweep (local subprocesses) then gather it."""
    submit_main()
    return gather_main()


if __name__ == "__main__":
    _slurm.cli_main(
        "examples.papers.ramadan1998_slurm",
        {"run": main, "submit": submit_main, "gather": gather_main},
    )
