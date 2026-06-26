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
design parameters automatically. This is exact for objectives whose parameter
dependence is through the propagation constants.

## Full `dS/dp` (mode-overlap sensitivities)

When the objective depends on **mode mismatch / overlaps** (e.g. butt-coupling
transmission), the neff-only gradient is incomplete. `meow.make_differentiable_objective`
wraps a real, gauge-invariant figure of merit (transmission `|S_ij|²`, splitting
ratio, loss) as a `jax.custom_vjp` whose backward differences the *whole* solve —
so the gradient includes every effect (the complete `dS/dp`) and composes with
`jax.grad`, at the cost of `2·N_params` re-solves. Two design notes:

- The **complex** EME S-matrix is gauge-inconsistent across re-solves (each mode
  solve has an arbitrary global phase), so only gauge-invariant *real* quantities
  are smooth in the parameters — differentiate those, not the raw complex S.
- An *analytic* overlap (eigenvector) sensitivity via truncated guided-mode
  perturbation theory is **not** accurate for high-contrast waveguides (the
  overlap change is dominated by radiation modes outside any finite computed
  basis), so differencing the full solve is the robust exact route. Use the cheap
  `make_differentiable_neffs` for propagation-mediated objectives and
  `make_differentiable_objective` when overlap/mismatch effects matter.

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

## Tilted interfaces: Kottke smoothing

The default `"axis"` smoothing picks arithmetic *or* harmonic averaging per field
component from the interface's grid alignment — first-order for tilted / curved
interfaces. `CrossSection.from_cell(..., smoothing_method="kottke")` instead
computes the interface **normal** (from a fill-fraction gradient) and applies the
normal-projected effective permittivity `ε_ii = ε∥ + (ε⊥ − ε∥)·n_i²` — continuous
in the interface orientation, so tilted-interface `neff` error drops markedly
(coarser grids for the same accuracy). It stays diagonal-anisotropic so the
default solver handles it; the full tensor's off-diagonal `eps_xy` needs the
tensorial solver (`tidy3d-extras`) and is omitted. The (geometry-only) Kottke
plan is cached on the cell like the axis plan.

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

### Exact eigenvector adjoint (operator access)

Because `meow.fde.sparse` *owns* the operator, it can also give an **exact,
cheap eigenvector/overlap sensitivity** — the piece the truncated-modal adjoint
could not. `scalar_operator` returns the assembled `A`, and
`eigenvector_sensitivity` solves the deflated/bordered system

```text
[ A − λI   v ] [ dv ]   [ −(dA/dp − dλ/dp I) v ]
[ vᵀ       0 ] [ μ  ] = [           0          ]
```

for `dv/dp` from a single sparse linear solve per parameter (no second
eigensolve, no truncated basis). On the same gauge-invariant overlap-power
harness `G_i = Σ_j |⟨A_i|B_j⟩|²` that exposed the truncated-modal failure, this
matches finite differences to `~1e-5` while the truncated prediction is ~100%
off — and it is faster than a finite-difference eigenvector (one linear solve vs
two eigensolves). This is exactly why "operator access" is the enabling
condition for a cheap exact `dS/dp`; the same construction extends to the
full-vector operator and to SLEPc (which has built-in eigenpair-derivative
routines).
