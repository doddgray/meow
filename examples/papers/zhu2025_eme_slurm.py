"""Slurm-distributed EME model of the Zhu 2025 trident edge coupler.

Slurm companion to :mod:`examples.papers.zhu2025_eme`. The dense facet->output
transmission spectrum is the expensive part (a full-device EME per wavelength on
a large mesh), so each wavelength's EME is submitted as an independent cluster
job; a later session gathers them and writes the same spectrum / propagation /
annotated-layout plots. The per-wavelength job rebuilds the device from the
(picklable) :class:`~examples.papers.zhu2025.TridentFacet` spec, so no live
gdsfactory objects need to cross the process boundary.

Run the in-session demo (local subprocesses) with::

    python -m examples.papers.zhu2025_eme_slurm

or split submission/collection across sessions with the ``submit``/``gather``
subcommands.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

import meow as mw
from examples.papers import _backends, _resolution, _slurm
from examples.papers import zhu2025 as z
from examples.papers import zhu2025_eme as ze

JOB_FOLDER = Path(os.environ.get("MEOW_SLURM_FOLDER", "meow_zhu_eme_jobs"))


def wavelength_transmission_job(
    facet: z.TridentFacet,
    l1: float,
    l2: float,
    w_out: float,
    wl: float,
    num_cells: int,
    num_modes: int,
    fine_res: float,
) -> tuple[float, float]:
    """(te_db, tm_db) facet->output transmission at one wavelength (a cluster job).

    Top-level + picklable so submitit can ship it; rebuilds the device from the
    facet spec inside the worker.
    """
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    comp = ze.trident_taper_component(facet, l1=l1, l2=l2, w_out=w_out)
    db = ze.transmission_db(
        comp, facet, np.array([wl]), num_cells=num_cells,
        num_modes=num_modes, fine_res=fine_res,
    )
    return float(db["TE"][0]), float(db["TM"][0])


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


def _eme_kwargs() -> dict[str, Any]:
    return {
        "num_cells": _resolution.num_cells(low=12, medium=24, high=48),
        "num_modes": _resolution.num_modes(low=6, medium=8, high=12),
        "fine_res": _resolution.pick(low=0.10, medium=0.07, high=0.05),
    }


def submit_spectrum(
    facet: z.TridentFacet,
    *,
    label: str,
    executor: Any,
    folder: Path | str = JOB_FOLDER,
    l1: float = 165.0,
    l2: float = 130.0,
    w_out: float = 0.8,
    spec_npts: int | None = None,
) -> dict[str, Any]:
    """Submit one EME job per spectrum wavelength; persist a small record."""
    import pickle

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    npts = spec_npts or _resolution.pick(low=5, medium=9, high=17)
    wls = np.linspace(1.1, 2.2, npts)
    kw = _eme_kwargs()
    jobs = [
        executor.submit(
            wavelength_transmission_job, facet, l1, l2, w_out, float(wl), **kw
        )
        for wl in wls
    ]
    # pickle the job handles directly: submitit jobs reattach to their persisted
    # results on unpickle, so a later session can gather from the folder alone.
    record = {"label": label, "facet": facet, "l1": l1, "l2": l2, "w_out": w_out,
              "wls": wls, "jobs": jobs}
    with (folder / f"{_slurm.safe_label(label)}.eme_spec.pkl").open("wb") as f:
        pickle.dump(record, f)
    return {"label": label, "jobs": jobs, "wls": wls, "record": record}


def gather_spectrum(handle: dict[str, Any], out: Path) -> dict[str, Any]:
    """Collect a submitted spectrum + write spectrum/propagation/layout plots."""
    label, wls = handle["label"], handle["wls"]
    facet = handle["record"]["facet"]
    rec = handle["record"]
    l1, l2, w_out = rec["l1"], rec["l2"], rec["w_out"]
    results = [j.result() for j in handle["jobs"]]
    db = {"TE": np.array([r[0] for r in results]),
          "TM": np.array([r[1] for r in results])}
    out.mkdir(parents=True, exist_ok=True)
    comp = ze.trident_taper_component(facet, l1=l1, l2=l2, w_out=w_out)
    ze.plot_annotated_layout(comp, facet, out / f"{label}_layout.png",
                             title=f"Zhu 2025 trident edge coupler - {label} (slurm)")
    ze.plot_spectrum(wls, db, out / f"{label}_spectrum.png",
                     title=f"Facet->output transmission - {label} (slurm)")
    mw.save_table(out / f"{label}_spectrum",
                  {"wl_nm": wls * 1000, "te_db": db["TE"], "tm_db": db["TM"]})
    kw = _eme_kwargs()
    for pol in ("TE", "TM"):
        field, zs = ze.propagate(comp, facet, 1.55, pol, **kw)
        ze.plot_propagation(field, zs, out / f"{label}_propagation_{pol}.png",
                            title=f"|E| {pol} input @ 1550 nm - {label} (slurm)")
    return {"label": label,
            "te_db_1550": float(np.interp(1.55, wls, db["TE"])),
            "tm_db_1550": float(np.interp(1.55, wls, db["TM"]))}


def _designs() -> dict[str, z.TridentFacet]:
    from examples.papers import zhu2025_designer as zd

    best, *_ = zd.optimize_facet(target_mfd=8.0, wl=1.31)
    return {"paper": z.TridentFacet(), "designer": best}


def submit_main() -> dict[str, Any]:
    """Session A: submit the per-wavelength EME jobs for both designs."""
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    ex = make_executor()
    out = {}
    for label, facet in _designs().items():
        h = submit_spectrum(facet, label=label, executor=ex, folder=JOB_FOLDER)
        out[label] = h["record"]["job_ids"]
    return {"submitted": out, "folder": str(JOB_FOLDER),
            "next": "run 'gather' in a later session with the same MEOW_SLURM_FOLDER"}


def gather_main() -> dict[str, Any]:
    """Session B: reload persisted records, collect the jobs and write all plots."""
    import pickle

    gf_out = ze.FIGDIR / "zhu2025_eme_slurm"
    summaries = {}
    for path in sorted(JOB_FOLDER.glob("*.eme_spec.pkl")):
        with path.open("rb") as f:
            record = pickle.load(f)  # the submitit job handles reattach on unpickle
        handle = {"label": record["label"], "wls": record["wls"],
                  "jobs": record["jobs"], "record": record}
        summaries[record["label"]] = gather_spectrum(handle, gf_out)
    return {"out_dir": str(gf_out), "summaries": summaries}


def main() -> dict[str, Any]:
    """In-session demo: submit (local subprocesses) then gather both designs."""
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    ex = make_executor(cluster="local")
    out = ze.FIGDIR / "zhu2025_eme_slurm"
    summaries = {}
    for label, facet in _designs().items():
        handle = submit_spectrum(facet, label=label, executor=ex, folder=JOB_FOLDER)
        summaries[label] = gather_spectrum(handle, out)
    return {"out_dir": str(out), "summaries": summaries}


if __name__ == "__main__":
    _slurm.cli_main(
        "examples.papers.zhu2025_eme_slurm",
        {"run": main, "submit": submit_main, "gather": gather_main},
    )
