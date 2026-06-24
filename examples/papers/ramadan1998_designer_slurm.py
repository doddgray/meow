"""Slurm-distributed adiabatic-coupler designer (Ramadan 1998 design rules).

Slurm-cluster companion to :mod:`examples.papers.ramadan1998_designer`. It
designs adiabatic couplers for a *matrix* of new target specifications
(wavelengths x coupler kinds) on a chosen layer stack and distributes, for each
design, a meow EME verification of a representative coupling section as a
concurrent slice-group job (the full optimized 3 dB lengths are millimetre-scale
and impractical to EME directly, so the distributed check uses a capped,
representative Region-II length to confirm the bar/cross coupling trend).

- :func:`submit_designs` designs each spec (FDE ``kappa_II`` + design rules),
  writes its GDS + design-rule figure, and submits the representative EME,
  persisting one record per design;
- :func:`gather_designs` reattaches, attributes the bar/cross split and writes a
  combined summary (CSV + JSON).

Run with ``python -m examples.papers.ramadan1998_designer_slurm`` (or the
``submit`` / ``gather`` subcommands across two sessions).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import meow as mw
from examples.papers import _backends, _resolution, _slurm
from examples.papers import ramadan1998 as r
from examples.papers import ramadan1998_designer as rd

JOB_FOLDER = Path(os.environ.get("MEOW_SLURM_FOLDER", "meow_ramadan_designer_jobs"))

# representative Region-II length used for the distributed EME check (the full
# optimized lengths are mm-scale; this confirms the coupling trend affordably)
CHECK_LENGTH_II = 200.0


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


def design_matrix(wavelengths: list[float], kinds: list[str]) -> list[tuple]:
    """The (wavelength, kind) design matrix."""
    return [(wl, kind) for wl in wavelengths for kind in kinds]


def submit_designs(
    *,
    wavelengths: list[float] | None = None,
    kinds: list[str] | None = None,
    epsilon: float = 0.02,
    executor: Any,
    folder: Path | str = JOB_FOLDER,
    res: float = 0.035,
    num_modes: int = 6,
    num_cells: int = 120,
) -> list[_slurm.SavedEME]:
    """Design each spec, write its GDS + figure, and submit its EME verification."""
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    folder = Path(folder)
    out = r.FIGDIR / "ramadan1998_designer_slurm"
    out.mkdir(parents=True, exist_ok=True)
    wavelengths = wavelengths or [1.31, 1.55]
    kinds = kinds or ["3dB", "full"]
    records = []
    for wl, kind in design_matrix(wavelengths, kinds):
        stack = rd.soi_stack(wl=wl)
        spec = rd.design(stack, kind=kind, epsilon=epsilon, res=res)
        component = rd.designed_component(spec, stack)
        stem = f"{stack.name}_{kind}_{int(wl * 1000)}nm"
        component.write_gds(str(out / f"{stem}.gds"))
        rd.plot_design_point(spec, out / f"{stem}_design_rule.png",
                             title=f"{stack.name} @ {wl * 1000:.0f} nm")
        # representative (capped) coupler for the distributed EME verification
        check = r.adiabatic_coupler_component(
            length_i=20.0, length_ii=CHECK_LENGTH_II, length_iii=20.0,
            w_a_in=stack.w + 0.10, w_b_in=stack.w, w_mid=stack.w + 0.05,
            gap_ii=stack.gap,
        )
        cells = r.device_cells(check, num_cells=num_cells, res=res)
        records.append(
            _slurm.submit_eme(
                stem, cells, mw.Environment(wl=wl, T=25.0),
                executor=executor, num_modes=num_modes, folder=folder,
                meta={"kind": kind, "wl_nm": wl * 1000, "epsilon": epsilon,
                      "kappa_ii": spec.kappa_ii, "length_um": spec.length_um,
                      "coupling_length_um": spec.coupling_length_um, "stem": stem},
            )
        )
    return records


def gather_designs(folder: Path | str = JOB_FOLDER) -> dict[str, Any]:
    """Reattach to each design's EME check, attribute bar/cross, write summaries."""
    out = r.FIGDIR / "ramadan1998_designer_slurm"
    out.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    for rec in _slurm.load_records(folder):
        s, pm = rec.jobs.result()
        bar, cross = r.attribute_bar_cross(
            s, pm, rec.jobs.cells[-1], rec.jobs.env, num_modes=rec.num_modes
        )
        m = rec.meta
        summary = {
            **m,
            "check_length_ii_um": CHECK_LENGTH_II,
            "bar_at_check": round(bar, 4),
            "cross_at_check": round(cross, 4),
        }
        mw.save_summary(out / f"{m['stem']}_summary", summary)
        summaries[rec.label] = summary
    return {"out_dir": str(out), "designs": summaries}


def submit_main() -> dict[str, Any]:
    """Session A: design the matrix and submit each EME verification."""
    records = submit_designs(
        executor=make_executor(),
        folder=JOB_FOLDER,
        res=_resolution.pick(low=0.06, medium=0.035, high=0.025),
        num_modes=_resolution.num_modes(low=4, medium=6, high=8),
        num_cells=_resolution.num_cells(low=24, medium=120, high=200),
    )
    return {
        "submitted": {r_.label: r_.jobs.job_ids for r_ in records},
        "folder": str(JOB_FOLDER),
        "next": "run 'gather' in a later session with the same MEOW_SLURM_FOLDER",
    }


def gather_main() -> dict[str, Any]:
    """Session B (later): reattach, collect and summarize the designs."""
    return gather_designs(JOB_FOLDER)


def main() -> dict[str, Any]:
    """In-session demo: design the matrix, submit (local) then gather."""
    submit_main()
    return gather_main()


if __name__ == "__main__":
    _slurm.cli_main(
        "examples.papers.ramadan1998_designer_slurm",
        {"run": main, "submit": submit_main, "gather": gather_main},
    )
