"""Slurm-distributed EME model of the Mao 2019 splitting-ratio coupler.

Slurm companion to :mod:`examples.papers.mao2019_eme`: each spectrum wavelength's
full-device EME is submitted as an independent cluster job (rebuilding the SOI
ADC from the picklable stack + ``dw_out`` inside the worker); a later session
gathers them and writes the splitting-ratio spectrum, the propagation field and
the annotated layout.

Run ``python -m examples.papers.mao2019_eme_slurm`` (or ``submit``/``gather``).
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
from examples.papers import mao2019 as m
from examples.papers import mao2019_designer as md
from examples.papers import mao2019_eme as me

JOB_FOLDER = Path(os.environ.get("MEOW_SLURM_FOLDER", "meow_mao_eme_jobs"))
CENTER = me.CENTER


def split_job(
    stack: m.SOIStack, dw_out: float, wl: float,
    num_cells: int, num_modes: int, fine_res: float,
) -> tuple[float, float]:
    """(upper, lower) output power at one wavelength (a picklable cluster job)."""
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    comp = md.adc_component(stack, dw_out)
    return me.split_powers(comp, wl, num_cells=num_cells, num_modes=num_modes,
                           fine_res=fine_res)


def make_executor(folder: Path | str = JOB_FOLDER, cluster: str | None = None) -> Any:
    """A :func:`meow.slurm_executor` honouring the shared ``MEOW_*`` resources."""
    return mw.slurm_executor(
        folder=str(folder), cluster=cluster or _backends.slurm_cluster(),
        timeout_min=_backends.timeout_min(), cpus_per_task=_backends.cpus_per_task(),
        slurm_partition=_backends.slurm_partition(),
    )


def _kw() -> dict[str, Any]:
    return {
        "num_cells": _resolution.num_cells(low=40, medium=120, high=240),
        "num_modes": _resolution.num_modes(low=4, medium=6, high=8),
        "fine_res": _resolution.pick(low=0.04, medium=0.03, high=0.02),
    }


def submit_design(
    stack: m.SOIStack, dw_out: float, *, label: str, executor: Any,
    folder: Path | str = JOB_FOLDER, spec_npts: int | None = None,
) -> dict[str, Any]:
    """Submit one EME job per spectrum wavelength; persist the job handles."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    npts = spec_npts or _resolution.pick(low=5, medium=7, high=13)
    wls = em.octave_wls(CENTER, npts)
    kw = _kw()
    jobs = [executor.submit(split_job, stack, dw_out, float(wl), **kw) for wl in wls]
    record = {"label": label, "stack": stack, "dw_out": dw_out,
              "wls": wls, "jobs": jobs}
    with (folder / f"{_slurm.safe_label(label)}.mao_eme.pkl").open("wb") as f:
        pickle.dump(record, f)
    return record


def gather_design(record: dict[str, Any], out: Path) -> dict[str, Any]:
    """Collect a submitted spectrum + write spectrum / propagation / layout."""
    label, wls, stack, dw = (record["label"], record["wls"],
                             record["stack"], record["dw_out"])
    res = [j.result() for j in record["jobs"]]
    up = np.array([r[0] for r in res])
    dn = np.array([r[1] for r in res])
    tot = up + dn
    sr_up, sr_dn = 100 * up / tot, 100 * dn / tot
    out.mkdir(parents=True, exist_ok=True)
    comp = md.adc_component(stack, dw)
    dvd, prm = me._dividers(stack, dw)
    em.plot_annotated_layout(
        comp, out / f"{label}_layout.png",
        title=f"Mao 2019 SR coupler - {label} (slurm)",
        layer_styles={m.LAYER_WG: ("SOI waveguides", "#1f77b4")},
        dividers=dvd, params=prm, ylim=(-2.5, 2.5),
    )
    em.plot_spectrum(wls, {"upper (bar) [%]": sr_up, "lower (cross) [%]": sr_dn},
                     out / f"{label}_spectrum.png",
                     title=f"Output splitting ratio - {label} (slurm)",
                     center_nm=CENTER * 1000, ylabel="splitting ratio [%]")
    mw.save_table(out / f"{label}_spectrum",
                  {"wl_nm": wls * 1000, "sr_upper_pct": sr_up, "sr_lower_pct": sr_dn,
                   "total_transmission": tot})
    kw = _kw()
    cells = me.device_cells(comp, num_cells=kw["num_cells"], fine_res=kw["fine_res"])
    modes = me._solve(cells, CENTER, kw["num_modes"])
    idx = me._upper_input_index(modes[0])
    field, x_trans = mw.propagate_modes(modes, cells, excite_mode_l=idx,
                                        y=m.H_SI / 2, num_z=400)
    em.plot_propagation(np.abs(np.asarray(field)), np.asarray(x_trans),
                        float(comp.xmax), out / f"{label}_propagation_TE.png",
                        title=f"|E| TE port-1 @ 1310 nm - {label} (slurm)",
                        ylim=(-2.5, 2.5))
    return {"label": label, "sr_upper_pct_1310": float(np.interp(CENTER, wls, sr_up))}


def _designs() -> list[tuple[str, m.SOIStack, float]]:
    stack = m.SOIStack()
    dw, *_ = md.design_dw_out(0.75, stack, res=0.02, npts=7)
    return [("paper_50_50", stack, 0.0), ("designer_75_25", stack, dw)]


def submit_main() -> dict[str, Any]:
    """Session A: submit the per-wavelength EME jobs for both designs."""
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    ex = make_executor()
    out = {label: submit_design(stack, dw, label=label, executor=ex)["label"]
           for label, stack, dw in _designs()}
    return {"submitted": list(out), "folder": str(JOB_FOLDER),
            "next": "run 'gather' later with the same MEOW_SLURM_FOLDER"}


def gather_main() -> dict[str, Any]:
    """Session B: reload persisted records, collect the jobs and write plots."""
    out = m.FIGDIR / "mao2019_eme_slurm"
    summaries = {}
    for path in sorted(JOB_FOLDER.glob("*.mao_eme.pkl")):
        with path.open("rb") as f:
            record = pickle.load(f)
        summaries[record["label"]] = gather_design(record, out)
    return {"out_dir": str(out), "summaries": summaries}


def main() -> dict[str, Any]:
    """In-session demo: submit (local subprocesses) then gather both designs."""
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    ex = make_executor(cluster="local")
    out = m.FIGDIR / "mao2019_eme_slurm"
    summaries = {}
    for label, stack, dw in _designs():
        record = submit_design(stack, dw, label=label, executor=ex)
        summaries[label] = gather_design(record, out)
    return {"out_dir": str(out), "summaries": summaries}


if __name__ == "__main__":
    _slurm.cli_main(
        "examples.papers.mao2019_eme_slurm",
        {"run": main, "submit": submit_main, "gather": gather_main},
    )
