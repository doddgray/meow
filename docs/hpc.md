# HPC, gradients & performance

meow ships features aimed at large, gradient-driven runs on clusters: exact
modal gradients, end-to-end autodiff, differentiable inverse design, and several
performance/scaling knobs for many spatial points and frequencies. This page is
a map; the worked example with plots is the
[modal gradients notebook](nbs/07_modal_gradients.md).

## Modal gradients (the adjoint)

meow's FDE backends solve each cross-section with an external eigensolver, so it
does not differentiate *through* a mode solve directly. Instead it provides the
**modal adjoint** — first-order perturbation theory for the effective-index
sensitivity, evaluated from the primal fields:

$$\mathrm{d}(n_\mathrm{eff}) = \frac{c\,\varepsilon_0}{4P}\int \mathrm{d}\varepsilon\;|\mathbf{E}|^2\,\mathrm{d}A .$$

| function | purpose |
|---|---|
| `meow.neff_sensitivity(mode)` | per-pixel `∂neff/∂ε_ii` density maps |
| `meow.neff_gradient(mode, dε…)` | directional `d(neff)` for a permittivity perturbation |
| `meow.neff_value_and_grad(solve, params, eps_jacobian)` | effective indices **and** their Jacobian from one solve |
| `meow.finite_difference_gradient(neff_of_t)` | gold-standard re-solve validator |

The kernel is exact for reciprocal low-loss dielectrics (validated against finite
differences to ~1e-8) and costs only array reductions over the primal fields — so
a gradient w.r.t. any number of parameters is negligible next to the solve, and
per-frequency sensitivities of a broadband objective sum as a parallel reduction.

## End-to-end autodiff

`meow.make_differentiable_neffs(solve, shape=..., cross_sections=...)` returns a
`jax.custom_vjp` mapping design parameters to effective indices. The forward runs
your solve via `jax.pure_callback`; the backward uses the adjoint with a cheap
permittivity Jacobian (no extra eigensolve), and solves are memoized so a
value-and-gradient is a single eigensolve. The returned indices compose with the
already-JAX SAX EME cascade, so `jax.grad` of any objective flows back to the
design parameters automatically.

## Differentiable inverse design (`meow.levelset`)

A staircased polygon boundary is not differentiable in geometry; a **density /
level-set** field is. `meow.levelset` maps a smooth `ρ(x,y) ∈ [0,1]` to
permittivity (`ε = ε_min + ρ(ε_max − ε_min)`, constant `dε/dρ`), samples it on the
Yee grids, builds a solvable cross-section
(`CrossSection.from_index_arrays` / `levelset.density_cross_section`), and
returns the analytic permittivity Jacobian
(`levelset.eps_jacobian_components`) to feed `make_differentiable_neffs`. The full
chain *params → density → ε → modes → S-matrix → objective* is differentiable.

## Threads vs. jobs vs. nodes

There are four independent parallel axes; match each to the right hardware layer.

- **Frequency** — distribute a dense sweep across nodes with `wls_per_job`
  (one job per slice-group × wavelength batch) instead of a few jobs each
  sweeping the whole spectrum; cascade the independent per-wavelength
  S-matrices concurrently with `cascade_workers`:
  ```python
  S = mw.compute_s_matrix_spectrum(
      cells, env, wls=wls, wls_per_job=8, cascade_workers=8,
      executor=mw.slurm_executor(...),
  )
  ```
- **Cell groups** — `compute_s_matrix_parallel` / `*_spectrum` already
  distribute the cell chain via `chunk_cell_indices`.
- **Grid points (one solve)** — threaded BLAS inside the backend; cap it to
  avoid oversubscription when several jobs share a node:
  ```python
  os.environ["MEOW_SOLVER_THREADS"] = "4"     # or, in-process:
  with mw.limit_threads(4):
      modes = mw.compute_modes(cs, num_modes=8)
  ```
  Set `solver_threads × jobs_per_node ≤ cores_per_node`.

## Dense spectra: cached smoothing

The subpixel-smoothing geometry (shapely interface detection + dual-cell area
fractions) is **wavelength-independent**, so it is computed once per cell and
reused across a whole wavelength/temperature sweep automatically — the per-sweep
cost drops to a single material-permittivity assembly per frequency.

## Large grids: sparse shift-invert

For very fine meshes where only a few guided modes are wanted, a sparse
shift-invert solve scales with the operator's nonzeros (~5 per grid point)
rather than a dense matrix:

```python
from meow.fde import sparse
neff = sparse.scalar_neffs(cs, num_modes=1)[0]
```

`meow.fde.sparse` implements the scalar / semivectorial Helmholtz approximation
(validated against the tidy3d vectorial backend in the low-contrast regime). The
assembled operator maps directly onto a PETSc matrix for an MPI-distributed
SLEPc solve on grids that exceed one node — the production extension for the
fully-vectorial problem.
"""
