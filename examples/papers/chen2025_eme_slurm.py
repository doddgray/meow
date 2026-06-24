"""Slurm-distributed EME model of the Chen 2025 FAQUAD TFLN PSR.

Slurm companion to :mod:`examples.papers.chen2025_eme`: the FAQUAD profile is
computed once, then each spectrum wavelength's full-device EME is submitted as an
independent cluster job; a later session gathers them and writes the routing
spectrum, propagation field and annotated layout. Run
``python -m examples.papers.chen2025_eme_slurm`` (or ``submit``/``gather``).
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

import meow as mw
from examples.papers import _backends, _resolution, _slurm
from examples.papers import _eme_model as em
from examples.papers import chen2025 as c
from examples.papers import chen2025_eme as ce

JOB_FOLDER = Path(os.environ.get("MEOW_SLURM_FOLDER", "meow_chen_eme_jobs"))
CENTER = ce.CENTER


def route_job(
    platform: Any, zs: np.ndarray, w1: np.ndarray, length: float, wl: float,
    num_cells: int, num_modes: int, fine_res: float,
) -> tuple[float, float]:
    """(cross, through) output power at one wavelength (a picklable cluster job)."""
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    return ce.route_powers(platform, zs, w1, length, wl, num_cells=num_cells,
                           num_modes=num_modes, fine_res=fine_res)


def make_executor(folder: Path | str = JOB_FOLDER, cluster: str | None = None) -> Any:
    """A :func:`meow.slurm_executor` honouring the shared ``MEOW_*`` resources."""
    return mw.slurm_executor(
        folder=str(folder), cluster=cluster or _backends.slurm_cluster(),
        timeout_min=_backends.timeout_min(), cpus_per_task=_backends.cpus_per_task(),
        slurm_partition=_backends.slurm_partition(),
    )


def _kw() -> dict[str, Any]:
    return {
        "num_cells": _resolution.num_cells(low=20, medium=40, high=80),
        "num_modes": _resolution.num_modes(low=4, medium=6, high=10),
        "fine_res": _resolution.pick(low=0.05, medium=0.035, high=0.025),
    }


def submit_design(
    platform: Any, length: float, *, label: str, executor: Any,
    folder: Path | str = JOB_FOLDER,
) -> dict[str, Any]:
    """Compute the FAQUAD profile, submit one EME job per wavelength, persist."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    scan_res = _resolution.pick(low=0.05, medium=0.035, high=0.025)
    scan_n = _resolution.pick(low=8, medium=13, high=21)
    w1s = np.linspace(0.1, 0.65, scan_n)
    neffs, _ = c.neff_evolution(platform, w1s, CENTER, res=scan_res)
    w_hyb = c.hybridization_point(w1s, neffs)
    zs, w1 = c.faquad_profile(w1s, neffs, length=length)
    zs = np.asarray(zs)
    kw = _kw()
    wls = em.octave_wls(CENTER, _resolution.pick(low=5, medium=9, high=15))
    jobs = [executor.submit(route_job, platform, zs, w1, length, float(wl), **kw)
            for wl in wls]
    record = {"label": label, "platform": platform, "length": length, "zs": zs,
              "w1": w1, "w_hyb": w_hyb, "wls": wls, "jobs": jobs}
    with (folder / f"{_slurm.safe_label(label)}.chen_eme.pkl").open("wb") as f:
        pickle.dump(record, f)
    return record


def gather_design(record: dict[str, Any], out: Path) -> dict[str, Any]:
    """Collect a submitted spectrum + write spectrum / propagation / layout."""
    r = record
    res = [j.result() for j in r["jobs"]]
    cross = np.array([x[0] for x in res])
    thru = np.array([x[1] for x in res])
    tot = cross + thru
    out.mkdir(parents=True, exist_ok=True)
    zs, w1, length = r["zs"], r["w1"], r["length"]
    em.plot_annotated_layout(
        ce._layout(zs, w1, length), out / f"{r['label']}_layout.png",
        title=f"Chen 2025 FAQUAD PSR - {r['label']} (slurm)",
        layer_styles={ce.LAYER_WG: ("TFLN ridges", "#9467bd")},
        dividers=[(float(np.interp(r["w_hyb"], w1, zs)),
                   f"hyb w1={r['w_hyb']:.2f} um")],
        params=f"w0={ce.W0} um, g={ce.GAP} um, L={length:.0f} um",
        ylim=(-2, 3),
    )
    em.plot_spectrum(
        r["wls"],
        {"TM0 -> cross [dB]": 10 * np.log10(np.maximum(cross / tot, 1e-4)),
         "residual through [dB]": 10 * np.log10(np.maximum(thru / tot, 1e-4))},
        out / f"{r['label']}_spectrum.png",
        title=f"TM0 through-input routing - {r['label']} (slurm)",
        center_nm=CENTER * 1000, ylabel="normalized output [dB]",
    )
    mw.save_table(out / f"{r['label']}_spectrum",
                  {"wl_nm": r["wls"] * 1000, "cross_frac": cross / tot,
                   "through_frac": thru / tot, "total_transmission": tot})
    kw = _kw()
    cells = ce.device_cells(r["platform"], zs, w1, length, wl=CENTER,
                           num_cells=kw["num_cells"], fine_res=kw["fine_res"])
    modes = ce._solve(cells, CENTER, kw["num_modes"])
    idx = ce._tm0_through_index(modes[0])
    field, x_trans = mw.propagate_modes(modes, cells, excite_mode_l=idx,
                                        y=r["platform"].core_thickness / 2, num_z=400)
    em.plot_propagation(np.abs(np.asarray(field)), np.asarray(x_trans), length,
                        out / f"{r['label']}_propagation_TM0.png",
                        title=f"|E| TM0 through-input @ 1550 nm - {r['label']} (slurm)",
                        ylim=(-2, 3))
    return {"label": r["label"],
            "cross_pct_1550": float(100 * np.interp(CENTER, r["wls"], cross / tot))}


def _designs() -> list[tuple[str, float]]:
    return [("paper_300um", 300.0), ("designer_200um", 200.0)]


def submit_main() -> dict[str, Any]:
    """Session A: submit per-wavelength EME jobs for both device lengths."""
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    ex = make_executor()
    platform = c.chen_platform()
    labels = [submit_design(platform, length, label=label, executor=ex)["label"]
              for label, length in _designs()]
    return {"submitted": labels, "folder": str(JOB_FOLDER),
            "next": "run 'gather' later with the same MEOW_SLURM_FOLDER"}


def gather_main() -> dict[str, Any]:
    """Session B: reload persisted records, collect the jobs and write plots."""
    out = c.FIGDIR / "chen2025_eme_slurm"
    summaries = {}
    for path in sorted(JOB_FOLDER.glob("*.chen_eme.pkl")):
        with path.open("rb") as f:
            record = pickle.load(f)
        summaries[record["label"]] = gather_design(record, out)
    return {"out_dir": str(out), "summaries": summaries}


def main() -> dict[str, Any]:
    """In-session demo: submit (local subprocesses) then gather both designs."""
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    ex = make_executor(cluster="local")
    out = c.FIGDIR / "chen2025_eme_slurm"
    platform = c.chen_platform()
    summaries = {}
    for label, length in _designs():
        record = submit_design(platform, length, label=label, executor=ex)
        summaries[label] = gather_design(record, out)
    return {"out_dir": str(out), "summaries": summaries}


if __name__ == "__main__":
    _slurm.cli_main(
        "examples.papers.chen2025_eme_slurm",
        {"run": main, "submit": submit_main, "gather": gather_main},
    )
