"""Speed benchmarks for the HPC / gradient performance features.

Times and plots four things added for large gradient-driven runs:

1. **Cached smoothing** - a wavelength sweep that reuses one cell (the subpixel-
   smoothing geometry is cached) vs. rebuilding the cell each wavelength.
2. **Sparse vs dense solver** - ``meow.fde.sparse.scalar_neffs`` (sparse shift-
   invert) vs. the dense tidy3d solve, swept over grid size (runtime scaling).
3. **Kottke vs axis smoothing** - the per-solve overhead of the normal-projected
   Kottke smoothing relative to the default axis scheme.
4. **Gradient cost** - the cheap modal-adjoint ``neff_value_and_grad`` (one solve)
   vs. a finite-difference gradient (``2 N_params`` solves).

Figures are written to ``examples/figures/``. Set ``MEOW_BENCH_FAST=1`` for a
reduced sweep. Run with ``python -m examples.benchmarks.performance_features``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import meow as mw
from meow.fde import sparse

FAST = os.environ.get("MEOW_BENCH_FAST", "0") not in ("0", "", "false", "False")
FIGDIR = Path(__file__).resolve().parents[1] / "figures"


def _timed(fn: Any, *args: Any, repeat: int = 1, **kwargs: Any) -> float:
    """Best-of-``repeat`` wall time of ``fn`` in seconds."""
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return best


def _strip_cell(width: float = 0.5, npx: int = 121, npy: int = 81) -> mw.Cell:
    core = mw.Structure(
        material=mw.IndexMaterial(name="core", n=3.45),
        geometry=mw.Box(
            x_min=-width / 2, x_max=width / 2, y_min=0.0, y_max=0.22,
            z_min=-1.0, z_max=2.0,
        ),
        mesh_order=5,
    )
    clad = mw.Structure(
        material=mw.IndexMaterial(name="clad", n=1.444),
        geometry=mw.Box(
            x_min=-1.5, x_max=1.5, y_min=-1.0, y_max=1.0, z_min=-1.0, z_max=2.0
        ),
        mesh_order=10,
    )
    mesh = mw.Mesh2D(x=np.linspace(-1.5, 1.5, npx), y=np.linspace(-1.0, 1.0, npy))
    return mw.create_cells([core, clad], mesh, [1.0], z_min=0.0)[0]


def bench_smoothing_cache() -> dict[str, float]:
    """Cross-section build cost across a wavelength sweep: cached vs rebuilt."""
    n_wl = 4 if FAST else 10
    wls = np.linspace(1.5, 1.6, n_wl)

    def cached() -> None:
        cell = _strip_cell()
        for wl in wls:
            cs = mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=float(wl)))
            _ = cs.nx, cs.ny, cs.nz

    def rebuilt() -> None:
        for wl in wls:
            cs = mw.CrossSection.from_cell(
                cell=_strip_cell(), env=mw.Environment(wl=float(wl))
            )
            _ = cs.nx, cs.ny, cs.nz

    t_cached = _timed(cached)
    t_rebuilt = _timed(rebuilt)
    return {"cached_s": t_cached, "rebuilt_s": t_rebuilt,
            "speedup": t_rebuilt / t_cached}


def bench_sparse_vs_dense() -> dict[str, list[float]]:
    """Runtime of sparse-scalar vs dense mode solve over grid size."""
    sizes = [81, 141] if FAST else [81, 141, 201, 261]
    dense, sp, npts = [], [], []
    for n in sizes:
        cell = _strip_cell(width=0.5, npx=n, npy=n)
        cs = mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=1.55))
        _ = cs.nz  # warm the (cached) index
        dense.append(_timed(lambda cs=cs: mw.compute_modes(cs, num_modes=1)))
        sp.append(_timed(lambda cs=cs: sparse.scalar_neffs(cs, num_modes=1)))
        npts.append(n * n)
    return {"npts": npts, "dense_s": dense, "sparse_s": sp}


def bench_kottke_overhead() -> dict[str, float]:
    """Per-solve cost of Kottke vs axis subpixel smoothing."""
    cell = _strip_cell(npx=141, npy=141)

    def solve(method: str) -> None:
        cs = mw.CrossSection.from_cell(
            cell=_strip_cell(npx=141, npy=141), env=mw.Environment(wl=1.55),
            smoothing_method=method,
        )
        mw.compute_modes(cs, num_modes=1)

    _ = cell
    t_axis = _timed(lambda: solve("axis"))
    t_kottke = _timed(lambda: solve("kottke"))
    return {"axis_s": t_axis, "kottke_s": t_kottke, "overhead": t_kottke / t_axis}


def bench_gradient_cost(n_params: int = 4) -> dict[str, float]:
    """Modal-adjoint gradient (1 solve) vs finite differences (2 N_params solves).

    ``n_params`` "design parameters" each scale the core index in one horizontal
    band of the core, so finite differences need ``2 * n_params`` re-solves while
    the modal adjoint needs a single solve (then cheap per-parameter contractions).
    """
    n_params = 2 if FAST else n_params
    mesh = mw.Mesh2D(x=np.linspace(-1.5, 1.5, 121), y=np.linspace(-1.0, 1.0, 81))
    bands = np.linspace(0.0, 0.22, n_params + 1)
    base = np.full(n_params, 3.45)

    def _solve(params: np.ndarray) -> list[mw.Mode]:
        structs = [
            mw.Structure(
                material=mw.IndexMaterial(name=f"core{j}", n=float(params[j])),
                geometry=mw.Box(x_min=-0.25, x_max=0.25, y_min=float(bands[j]),
                                y_max=float(bands[j + 1]), z_min=-1.0, z_max=2.0),
                mesh_order=5 - j * 0.0,  # same priority; bands are disjoint
            )
            for j in range(n_params)
        ]
        clad = mw.Structure(
            material=mw.IndexMaterial(name="clad", n=1.444),
            geometry=mw.Box(x_min=-1.5, x_max=1.5, y_min=-1.0, y_max=1.0,
                            z_min=-1.0, z_max=2.0),
            mesh_order=10,
        )
        cell = mw.create_cells([*structs, clad], mesh, [1.0], z_min=0.0)[0]
        return mw.compute_modes(
            mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=1.55)),
            num_modes=1,
        )

    # band masks on each Yee grid (built from coordinates - no eigensolve)
    grids = {"x": (mesh.Xx, mesh.Yx), "y": (mesh.Xy, mesh.Yy), "z": (mesh.Xz, mesh.Yz)}

    def eps_jac(params: np.ndarray, j: int) -> tuple:
        out = []
        for gx, gy in grids.values():
            gx, gy = np.asarray(gx), np.asarray(gy)
            mask = (np.abs(gx) < 0.25) & (gy >= bands[j]) & (gy < bands[j + 1])
            out.append(2.0 * params[j] * mask)
        return tuple(out)

    def adjoint() -> None:
        mw.neff_value_and_grad(_solve, base, eps_jac)

    def fd() -> None:
        for j in range(n_params):
            for sgn in (+1, -1):
                p = base.copy()
                p[j] += sgn * 1e-3
                _solve(p)

    t_adj = _timed(adjoint)
    t_fd = _timed(fd)
    return {"n_params": float(n_params), "adjoint_s": t_adj, "fd_s": t_fd,
            "speedup": t_fd / t_adj}


def bench_eigvec_adjoint() -> dict[str, float]:
    """Exact eigenvector sensitivity: deflated solve vs finite-difference re-solve.

    Per design parameter, the deflated bordered solve is a single sparse linear
    solve on the assembled operator, whereas a finite-difference eigenvector
    needs two sparse *eigensolves* - so the adjoint is markedly cheaper per
    parameter (and the gap widens with the number of parameters).
    """
    from scipy.sparse.linalg import eigsh

    from meow.fde import sparse

    npx = 81 if FAST else 161
    x = np.linspace(-1.5, 1.5, npx)
    y = np.linspace(-1.0, 1.0, npx)
    xx, yy = np.meshgrid(x, y)
    n = np.where((np.abs(xx) < 0.25) & (yy > 0) & (yy < 0.22), 3.45, 1.444)
    a_op, k0 = sparse.scalar_operator(n, x, y, 1.55)
    vals, vecs = eigsh(a_op, k=1, sigma=(k0 * float(n.max())) ** 2, which="LM")
    deps = (2.0 * 3.45 * ((np.abs(xx) < 0.25) & (yy > 0) & (yy < 0.22))).ravel()

    def adjoint() -> None:
        sparse.eigenvector_sensitivity(a_op, vals[0], vecs[:, 0], deps, k0)

    def fd_resolve() -> None:  # the FD eigenvector cost: two eigensolves
        for _ in range(2):
            eigsh(a_op, k=1, sigma=(k0 * float(n.max())) ** 2, which="LM")

    t_adj = _timed(adjoint, repeat=3)
    t_fd = _timed(fd_resolve, repeat=3)
    return {"adjoint_s": t_adj, "fd_resolve_s": t_fd, "speedup": t_fd / t_adj}


def main() -> dict[str, Any]:
    """Run all benchmarks, print a summary and save a scaling plot."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    cache = bench_smoothing_cache()
    sd = bench_sparse_vs_dense()
    kottke = bench_kottke_overhead()
    grad = bench_gradient_cost()
    adj = bench_eigvec_adjoint()

    print("\n=== smoothing cache (wavelength sweep) ===")
    print(f"  rebuilt {cache['rebuilt_s']:.3f}s  cached {cache['cached_s']:.3f}s  "
          f"-> {cache['speedup']:.1f}x faster")
    print("=== Kottke vs axis smoothing (per solve) ===")
    print(f"  axis {kottke['axis_s']:.3f}s  kottke {kottke['kottke_s']:.3f}s  "
          f"-> {kottke['overhead']:.2f}x")
    print(f"=== gradient cost ({int(grad['n_params'])} params) ===")
    print(f"  adjoint {grad['adjoint_s']:.3f}s  finite-diff {grad['fd_s']:.3f}s  "
          f"-> {grad['speedup']:.1f}x faster")
    print("=== eigenvector adjoint (per parameter) ===")
    print(f"  deflated solve {adj['adjoint_s']:.4f}s  FD re-solve "
          f"{adj['fd_resolve_s']:.4f}s  -> {adj['speedup']:.1f}x faster")
    print("=== sparse vs dense solve (runtime [s] by grid points) ===")
    for n, d, s in zip(sd["npts"], sd["dense_s"], sd["sparse_s"], strict=True):
        print(f"  {n:>7d} pts: dense {d:.3f}s  sparse {s:.3f}s")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(sd["npts"], sd["dense_s"], "o-", label="dense (tidy3d)")
    ax.loglog(sd["npts"], sd["sparse_s"], "s-", label="sparse shift-invert")
    ax.set(xlabel="grid points", ylabel="mode-solve time [s]",
           title="Sparse vs dense mode-solve scaling")
    ax.grid(visible=True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = FIGDIR / "benchmark_performance_features.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return {"cache": cache, "sparse_dense": sd, "kottke": kottke, "gradient": grad,
            "eigvec_adjoint": adj, "figure": str(out)}


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
