"""Slurm-distributed **broadband coupling spectrum** of a pulley coupler.

Slurm companion to :mod:`examples.papers.pulley_coupler_designer`. The natural
parallel axis for a bent-coupler study is *wavelength*: each wavelength needs an
independent bent supermode solve to get ``(kappa0, delta, n_g)``. This module
submits one cluster job per wavelength (over a broad, >1-octave band), then a
later session gathers them into the coupling / coupling-Q / extraction spectra.

Usage (mirrors the other ``*_slurm`` examples)::

    python -m examples.papers.pulley_coupler_slurm            # in-session demo
    python -m examples.papers.pulley_coupler_slurm submit     # session A: submit
    python -m examples.papers.pulley_coupler_slurm gather     # session B: gather

Resources honour the shared ``MEOW_*`` env vars (see :mod:`meow.settings`); the
in-session ``run`` uses local subprocess "jobs" so it works without a cluster.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

import meow as mw
from examples.papers import _backends
from examples.papers import _bent_coupler as bc
from examples.papers import _resolution as res
from examples.papers import pulley_coupler_designer as pd

JOB_FOLDER = Path(os.environ.get("MEOW_SLURM_FOLDER", "meow_pulley_jobs"))


def spectrum_job(
    platform: bc.StripPlatform, ring_width: float, bus_width: float, gap: float,
    radius: float, wl: float, res_um: float,
) -> tuple[float, float, float]:
    """(kappa0, delta, n_g) at one wavelength -- a picklable cluster job."""
    kappa0, delta = bc.supermode_coupling(platform, ring_width, bus_width, gap, wl,
                                          bend_radius=radius, res=res_um)
    ng = bc.group_index(platform, ring_width, wl, bend_radius=radius, res=res_um)
    return float(kappa0), float(delta), float(ng)


def make_executor(folder: Path | str = JOB_FOLDER, cluster: str | None = None) -> Any:
    return mw.slurm_executor(
        folder=str(folder), cluster=cluster or _backends.slurm_cluster(),
        timeout_min=_backends.timeout_min(), cpus_per_task=_backends.cpus_per_task(),
        slurm_partition=_backends.slurm_partition(),
    )


def submit_design(
    label: str, make_platform: Any, radius: float, ring_width: float, gap: float,
    wl: float, target: float, loss: float, *, executor: Any,
    folder: Path | str = JOB_FOLDER,
) -> dict[str, Any]:
    """Design a pulley, then submit one bent-solve job per spectrum wavelength."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    res_um = res.pick(low=0.05, medium=0.04, high=0.03)
    design = pd.design_pulley_coupler(make_platform(), radius, wl, target,
                                      ring_width=ring_width, gap=gap,
                                      loss_db_cm=loss, label=label, res_um=res_um)
    wls = np.linspace(0.8 * wl, 1.25 * wl, res.pick(low=6, medium=12, high=24))
    jobs = [executor.submit(spectrum_job, design.platform, design.ring_width,
                            design.bus_width, design.gap, design.radius, float(w),
                            res_um) for w in wls]
    record = {"label": label, "design": design, "wls": wls, "jobs": jobs}
    with (folder / f"{label}.pulley.pkl").open("wb") as f:
        pickle.dump(record, f)
    return record


def gather_design(record: dict[str, Any], out: Path) -> dict[str, Any]:
    """Collect one submitted spectrum and write coupling / Qc / extraction plots."""
    design = record["design"]
    wls = record["wls"]
    ks, ds, ngs = zip(*[j.result() for j in record["jobs"]], strict=True)
    k2 = np.array([bc.pulley_cross_power(k, d, design.length)
                   for k, d in zip(ks, ds, strict=True)])
    qc = np.array([bc.q_coupling(float(k), float(n), design.radius, float(w))
                   for k, n, w in zip(k2, ngs, wls, strict=True)])
    qi = bc.q_intrinsic(design.loss_db_cm, float(np.mean(ngs)), design.wl)
    eta = np.array([bc.extraction_efficiency(float(q), qi) for q in qc])
    out.mkdir(parents=True, exist_ok=True)
    bc.plot_spectrum(wls, {r"$|\kappa|^2$": k2}, out / f"{record['label']}_kappa2.png",
                     title=f"{record['label']}: broadband coupling (slurm)",
                     center_nm=design.wl * 1000)
    bc.plot_spectrum(wls, {"$Q_c$": qc}, out / f"{record['label']}_Qc.png",
                     title=f"{record['label']}: coupling-Q (slurm)", ylabel="$Q_c$",
                     logy=True, center_nm=design.wl * 1000)
    bc.plot_spectrum(wls, {r"$\eta$": eta}, out / f"{record['label']}_extraction.png",
                     title=f"{record['label']}: extraction (slurm)",
                     ylabel=r"$\eta$", center_nm=design.wl * 1000)
    mw.save_table(out / f"{record['label']}_spectrum",
                  {"wl_nm": wls * 1000, "kappa2": k2, "Qc": qc, "eta": eta})
    return {"label": record["label"], "Qc_flatness": float(np.max(qc) / np.min(qc))}


DESIGNS = pd.DESIGNS  # reuse the designer's demo platforms/targets


def submit_main() -> dict[str, Any]:
    ex = make_executor()
    labels = [submit_design(label, make, radius, rw, gap, wl, target, loss,
                            executor=ex)["label"]
              for label, make, radius, rw, gap, wl, target, loss in DESIGNS]
    return {"submitted": labels, "folder": str(JOB_FOLDER),
            "next": "run 'gather' later with the same MEOW_SLURM_FOLDER"}


def gather_main() -> dict[str, Any]:
    out = FIGDIR
    summaries = {}
    for path in sorted(JOB_FOLDER.glob("*.pulley.pkl")):
        with path.open("rb") as f:
            record = pickle.load(f)
        summaries[record["label"]] = gather_design(record, out)
    return {"out_dir": str(out), "summaries": summaries}


FIGDIR = bc.FIGDIR / "pulley_coupler_slurm"


def main() -> dict[str, Any]:
    ex = make_executor(cluster="local")
    summaries = {}
    for label, make, radius, rw, gap, wl, target, loss in DESIGNS:
        record = submit_design(label, make, radius, rw, gap, wl, target, loss,
                               executor=ex)
        summaries[label] = gather_design(record, FIGDIR)
    return {"out_dir": str(FIGDIR), "summaries": summaries}


if __name__ == "__main__":
    from examples.papers import _slurm

    _slurm.cli_main(
        "examples.papers.pulley_coupler_slurm",
        {"run": main, "submit": submit_main, "gather": gather_main},
    )
