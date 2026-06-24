"""Slurm-distributed EME model of the Song 2023 TFLN rotator taper.

Slurm companion to :mod:`examples.papers.song2023_eme`: each spectrum
wavelength's full-taper EME is submitted as an independent cluster job; a later
session gathers them and writes the conversion spectrum, propagation field and
annotated layout. Run ``python -m examples.papers.song2023_eme_slurm`` (or
``submit``/``gather``).
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
from examples.papers import song2023 as s
from examples.papers import song2023_eme as se

JOB_FOLDER = Path(os.environ.get("MEOW_SLURM_FOLDER", "meow_song_eme_jobs"))
CENTER = se.CENTER


def conversion_job(
    platform: Any, w_start: float, w_end: float, length: float, wl: float,
    num_cells: int, num_modes: int, fine_res: float,
) -> tuple[float, float]:
    """(TE-pol, TM-pol) output power at one wavelength (a picklable cluster job)."""
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    return se.conversion_powers(platform, w_start, w_end, length, wl,
                                num_cells=num_cells, num_modes=num_modes,
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
        "num_cells": _resolution.num_cells(low=20, medium=40, high=80),
        "num_modes": _resolution.num_modes(low=4, medium=6, high=8),
        "fine_res": _resolution.pick(low=0.05, medium=0.035, high=0.025),
        "length": 300.0,
    }


def _w0(platform: Any) -> float:
    scan_res = _resolution.pick(low=0.05, medium=0.035, high=0.025)
    scan_n = _resolution.pick(low=8, medium=13, high=21)
    widths = np.linspace(0.8, 1.9, scan_n)
    _, fracs = s.hybridization_scan(platform, widths, CENTER, res=scan_res)
    return s.hybridization_width(widths, fracs)


def submit_design(
    platform: Any, *, label: str, executor: Any, folder: Path | str = JOB_FOLDER,
) -> dict[str, Any]:
    """Submit one EME job per spectrum wavelength; persist the job handles."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    kw = _kw()
    length = kw.pop("length")
    w0 = _w0(platform)
    w_start, w_end = w0 - 0.3, w0 + 0.3
    wls = em.octave_wls(CENTER, _resolution.pick(low=5, medium=9, high=15))
    jobs = [executor.submit(conversion_job, platform, w_start, w_end, length,
                            float(wl), **kw) for wl in wls]
    record = {"label": label, "platform": platform, "w0": w0, "length": length,
              "w_start": w_start, "w_end": w_end, "wls": wls, "jobs": jobs}
    with (folder / f"{_slurm.safe_label(label)}.song_eme.pkl").open("wb") as f:
        pickle.dump(record, f)
    return record


def gather_design(record: dict[str, Any], out: Path) -> dict[str, Any]:
    """Collect a submitted spectrum + write spectrum / propagation / layout."""
    r = record
    res = [j.result() for j in r["jobs"]]
    te = np.array([x[0] for x in res])
    tm = np.array([x[1] for x in res])
    tot = te + tm
    out.mkdir(parents=True, exist_ok=True)
    em.plot_annotated_layout(
        se._layout(r["w_start"], r["w_end"], r["length"]),
        out / f"{r['label']}_layout.png",
        title=f"Song 2023 TFLN rotator - {r['label']} (slurm)",
        layer_styles={se.LAYER_WG: ("TFLN ridge", "#9467bd")},
        dividers=[(r["length"] / 2, f"hyb w0={r['w0']:.2f} um")],
        params=f"w: {r['w_start']:.2f}->{r['w_end']:.2f} um, L={r['length']:.0f} um",
        ylim=(-3, 3),
    )
    em.plot_spectrum(
        r["wls"],
        {"TM0->TE conversion [dB]": 10 * np.log10(np.maximum(te / tot, 1e-4)),
         "residual TM [dB]": 10 * np.log10(np.maximum(tm / tot, 1e-4))},
        out / f"{r['label']}_spectrum.png",
        title=f"TM0->TE1 conversion - {r['label']} (slurm)",
        center_nm=CENTER * 1000, ylabel="normalized output [dB]",
    )
    mw.save_table(out / f"{r['label']}_spectrum",
                  {"wl_nm": r["wls"] * 1000, "te_frac": te / tot, "tm_frac": tm / tot,
                   "total_transmission": tot})
    kw = _kw()
    length = kw.pop("length")
    cells = se.device_cells(r["platform"], r["w_start"], r["w_end"], length,
                           wl=CENTER, num_cells=kw["num_cells"],
                           fine_res=kw["fine_res"])
    modes = se._solve(cells, CENTER, kw["num_modes"])
    idx = se._tm0_input_index(modes[0])
    field, x_trans = mw.propagate_modes(modes, cells, excite_mode_l=idx,
                                        y=r["platform"].core_thickness / 2, num_z=400)
    em.plot_propagation(np.abs(np.asarray(field)), np.asarray(x_trans), length,
                        out / f"{r['label']}_propagation_TM0.png",
                        title=f"|E| TM0 input @ 1550 nm - {r['label']} (slurm)",
                        ylim=(-3, 3))
    return {"label": r["label"],
            "conversion_pct_1550": float(100 * np.interp(CENTER, r["wls"], te / tot))}


def _designs() -> list[tuple[str, Any]]:
    return [("paper_300nm", s.song_platform()),
            ("designer_500nm",
             s.tfln_platform(0.50, etch_depth=0.20, sidewall_deg=15.0))]


def submit_main() -> dict[str, Any]:
    """Session A: submit per-wavelength EME jobs for both platforms."""
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    ex = make_executor()
    labels = [submit_design(p, label=label, executor=ex)["label"]
              for label, p in _designs()]
    return {"submitted": labels, "folder": str(JOB_FOLDER),
            "next": "run 'gather' later with the same MEOW_SLURM_FOLDER"}


def gather_main() -> dict[str, Any]:
    """Session B: reload persisted records, collect the jobs and write plots."""
    out = s.FIGDIR / "song2023_eme_slurm"
    summaries = {}
    for path in sorted(JOB_FOLDER.glob("*.song_eme.pkl")):
        with path.open("rb") as f:
            record = pickle.load(f)
        summaries[record["label"]] = gather_design(record, out)
    return {"out_dir": str(out), "summaries": summaries}


def main() -> dict[str, Any]:
    """In-session demo: submit (local subprocesses) then gather both designs."""
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    ex = make_executor(cluster="local")
    out = s.FIGDIR / "song2023_eme_slurm"
    summaries = {}
    for label, platform in _designs():
        record = submit_design(platform, label=label, executor=ex)
        summaries[label] = gather_design(record, out)
    return {"out_dir": str(out), "summaries": summaries}


if __name__ == "__main__":
    _slurm.cli_main(
        "examples.papers.song2023_eme_slurm",
        {"run": main, "submit": submit_main, "gather": gather_main},
    )
