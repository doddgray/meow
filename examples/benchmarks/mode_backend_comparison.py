"""Compare the tidy3d and MPB mode-solver backends on benchmark waveguides.

For three nominally equivalent waveguide models (step-index fiber, Si3N4
strip 800 nm x 1.5 um, and an x-cut TFLN rib with anisotropy and 65 degree
sidewalls; see ``structures.py``) this script:

1. compares the modal metrics - effective index, group index, GVD
   (dispersion parameter D), effective area and TE polarization fraction -
   computed with both backends at sample wavelengths;
2. sweeps the fundamental-mode effective index over vacuum wavelengths
   spanning 0.4 - 1.8 um with both backends;
3. measures effective-index convergence and solver runtime against grid
   resolution for both backends.

Figures are written to ``examples/figures/``. The MPB parts require the
``meep``/``mpb`` python bindings (conda-forge ``pymeep``); without them
only the tidy3d results are produced.

Run with: ``python -m examples.benchmarks.mode_backend_comparison``
(optionally set ``MEOW_BENCH_FAST=1`` for a reduced sweep).
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.benchmarks.structures import STRUCTURES
from meow.fde.dispersion import dispersion_metrics, solve_mode
from meow.fde.mpb import compute_modes_mpb
from meow.fde.tidy3d import compute_modes_tidy3d

FIGDIR = Path(__file__).parent.parent / "figures"
FAST = bool(int(os.environ.get("MEOW_BENCH_FAST", "0")))

HAVE_MPB = importlib.util.find_spec("meep") is not None

WLS_SWEEP = {
    "fiber": np.linspace(0.4, 1.8, 5 if FAST else 8),
    "si3n4": np.linspace(0.4, 1.8, 5 if FAST else 8),
    "tfln": np.linspace(0.4, 1.6, 5 if FAST else 7),
}
WL_METRICS = 1.55
RESOLUTIONS = [0.1, 0.07, 0.05] if FAST else [0.1, 0.08, 0.06, 0.04, 0.03]
WLS_CONV = [0.6, 1.0, 1.55]
NUM_MODES = 2


def _backends() -> dict[str, object]:
    backends: dict[str, object] = {"tidy3d": compute_modes_tidy3d}
    if HAVE_MPB:
        backends["mpb"] = compute_modes_mpb
    return backends


def metrics_comparison() -> dict[str, dict[str, dict[str, float]]]:
    """Modal metrics at 1.55 um for each structure and backend."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name, (make_structs, make_mesh) in STRUCTURES.items():
        structs, mesh = make_structs(), make_mesh(0.08 if FAST else 0.04)
        out[name] = {}
        for bname, backend in _backends().items():
            m = dispersion_metrics(
                structs,
                WL_METRICS,
                mesh,
                num_modes=NUM_MODES,
                compute_modes=backend,
            )
            out[name][bname] = {
                "neff": m.neff,
                "group_index": m.group_index,
                "dispersion_D_ps_nm_km": m.dispersion_D,
                "beta2_s2_per_m": m.beta2,
                "effective_area_um2": m.effective_area,
                "te_fraction": m.te_fraction,
            }
    return out


def wavelength_sweep() -> dict[str, dict[str, tuple[list, list]]]:
    """neff(lambda) of the fundamental mode for each structure and backend."""
    out: dict[str, dict[str, tuple[list, list]]] = {}
    for name, (make_structs, make_mesh) in STRUCTURES.items():
        structs, mesh = make_structs(), make_mesh(0.08 if FAST else 0.05)
        out[name] = {}
        for bname, backend in _backends().items():
            wls, neffs = [], []
            for wl in WLS_SWEEP[name]:
                mode = solve_mode(
                    structs,
                    float(wl),
                    mesh,
                    num_modes=NUM_MODES,
                    compute_modes=backend,
                )
                wls.append(float(wl))
                neffs.append(float(np.real(mode.neff)))
            out[name][bname] = (wls, neffs)
    return out


def convergence_and_runtime() -> dict:
    """Neff and runtime vs grid resolution for each structure/backend/wl."""
    out: dict = {}
    for name, (make_structs, make_mesh) in STRUCTURES.items():
        structs = make_structs()
        out[name] = {}
        for bname, backend in _backends().items():
            out[name][bname] = {}
            for wl in WLS_CONV:
                rows = []
                for res in RESOLUTIONS:
                    mesh = make_mesh(res)
                    t0 = time.perf_counter()
                    mode = solve_mode(
                        structs,
                        wl,
                        mesh,
                        num_modes=NUM_MODES,
                        compute_modes=backend,
                    )
                    dt = time.perf_counter() - t0
                    n_grid = (len(mesh.x) - 1) * (len(mesh.y) - 1)
                    rows.append(
                        {
                            "res": res,
                            "n_grid": n_grid,
                            "neff": float(np.real(mode.neff)),
                            "runtime_s": dt,
                        }
                    )
                out[name][bname][wl] = rows
    return out


def plot_all(metrics: dict, sweeps: dict, conv: dict) -> None:
    for name in STRUCTURES:
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

        ax = axes[0]
        for bname, (wls, neffs) in sweeps[name].items():
            ax.plot(wls, neffs, "o-" if bname == "tidy3d" else "s--", label=bname, ms=4)
        ax.set_xlabel("wavelength [um]")
        ax.set_ylabel("$n_{eff}$")
        ax.set_title(f"{name}: fundamental mode dispersion")
        ax.legend(fontsize=8)
        ax.grid(visible=True)

        ax = axes[1]
        for bname, by_wl in conv[name].items():
            for wl, rows in by_wl.items():
                ax.plot(
                    [r["res"] for r in rows],
                    [r["neff"] for r in rows],
                    "o-" if bname == "tidy3d" else "s--",
                    ms=4,
                    label=f"{bname} {wl} um",
                )
        ax.set_xlabel("grid resolution [um]")
        ax.set_ylabel("$n_{eff}$")
        ax.invert_xaxis()
        ax.set_title(f"{name}: convergence vs grid size")
        ax.legend(fontsize=6)
        ax.grid(visible=True)

        ax = axes[2]
        for bname, by_wl in conv[name].items():
            for wl, rows in by_wl.items():
                ax.loglog(
                    [r["n_grid"] for r in rows],
                    [r["runtime_s"] for r in rows],
                    "o-" if bname == "tidy3d" else "s--",
                    ms=4,
                    label=f"{bname} {wl} um",
                )
        ax.set_xlabel("grid points")
        ax.set_ylabel("runtime [s]")
        ax.set_title(f"{name}: solver runtime vs grid size")
        ax.legend(fontsize=6)
        ax.grid(visible=True, which="both")

        table = " | ".join(
            f"{b}: ng={metrics[name][b]['group_index']:.3f}, "
            f"D={metrics[name][b]['dispersion_D_ps_nm_km']:.0f}, "
            f"Aeff={metrics[name][b]['effective_area_um2']:.2f}um2, "
            f"TE={metrics[name][b]['te_fraction']:.2f}"
            for b in metrics[name]
        )
        fig.suptitle(f"mode-solver backends - {name}   [{table}]", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIGDIR / f"mode_backends_{name}.png", dpi=150)
        plt.close(fig)


def main() -> dict:
    FIGDIR.mkdir(exist_ok=True, parents=True)
    metrics = metrics_comparison()
    sweeps = wavelength_sweep()
    conv = convergence_and_runtime()
    plot_all(metrics, sweeps, conv)
    with (FIGDIR / "mode_backends_metrics.json").open("w") as f:
        json.dump({"metrics": metrics, "convergence": conv}, f, indent=2)
    return metrics


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
