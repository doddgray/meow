# Paper reproduction examples

Design workflows from two published photonic-filter papers, implemented with
[gdsfactory](https://gdsfactory.github.io/gdsfactory/) parametric cells and
this repository's FDE mode solver + EME engine. Each example defines the
paper's device as a parametric layout, extrudes it into a 3D meow model, and
reproduces the paper's model/layout figures (experimental figures - SEMs and
measurements - can of course only be reproduced as their simulated
counterparts).

## Magden et al., Nat. Commun. 9, 3009 (2018)

*Transmissive silicon photonic dichroic filters with spectrally selective
waveguides.* 220 nm SOI; a solid waveguide WGA (318 nm) couples to a
three-segment sub-wavelength waveguide WGB (250 nm segments, 100 nm gaps)
across a 750 nm gap. The two waveguides are phase matched at exactly one
wavelength (the filter cutoff, ~1540 nm); four adiabatic sections
(L1/L2/L3/L4 = 200/260/900/200 um) route short- and long-pass light to
separate output ports.

- `magden2018_dichroic.py`: the parametric `dichroic_filter()` PCell (paper
  Fig. 3a: a straight three-segment WGB - constant total width and inter-ridge
  gaps, single-ridge ports, finite 50 nm taper tips - with WGA tipping in at a
  constant coupling gap in section 2 and bending away in section 3), the
  extrusion/meshing helpers and the coupled-mode quantities
  (delta, kappa, |T_A|^2).
- `magden2018_figures.py`: reproduces Fig. 1 (modes + effective indices +
  supermode anticrossing, with psi_A/psi_B bracketed by psi_+/psi_-), Fig. 2
  (delta in 1/um and the monotonic coupling |kappa| in 1/mm on twin axes over
  1530-1550 nm, the transmission roll-off, and the extinction ratio vs
  |gamma|), Fig. 3 (layout + EME optimization of the four adiabatic section
  lengths), Fig. 4 (top-down adiabatic mode-evolution field at the paper's
  nine wavelengths 1300-2800 nm in a 3x3 grid) and the model counterpart of
  Fig. 5 (coupled-mode filter spectra + cutoff-vs-width shift). The coupling
  |kappa| is the FDE coupled-mode overlap integral (monotonic) with its
  absolute scale anchored to the ~5/mm design coupling at the 750 nm gap; set
  `MEOW_EXAMPLE_HIFI=1` to also overlay a slow full-device EME spectrum on
  Fig. 5.

Validation anchors (full-quality run): the phase-matching cutoffs land at
1496/1537/1573 nm for the 312/318/324 nm WGA widths - matching the paper's
~1500/1540/1580 nm (Fig. 4d), including the ~6.4 nm-per-nm width
sensitivity; the per-section EME sweeps converge to low-loss transmission
(sections 1/2/4 below 0.2 dB) at the paper's chosen lengths; and the
mode-evolution spectra cross over from short-pass to long-pass at the
cutoff with a multi-dB/nm roll-off. Note: at our model's coupling
(kappa ~ 0.003/um at the 750 nm gap) a full-device end-to-end EME would
need impractically long transitions to be adiabatic, so the Fig. 4 spectra
use the paper's coupled-mode response (Eq. 3) with FDE-computed
delta(lambda) and kappa(lambda); the per-section convergence sweeps are
the paper's actual EME usage.

## Kwolek et al., arXiv:2603.27034 (2026)

*Ultra-broadband, low-loss wavelength combiners and filters in thin-film
lithium niobate.* 300 nm x-cut TFLN, ~100 nm etch (200 nm slab), 65 degree
sidewall angle, ~1.2 um waveguides. A fast-quasi-adiabatic (FAQUAD)
directional coupler combines/filters the fundamental harmonic (1550 nm,
transfers to the cross port) and its second harmonic (775 nm, stays in the
bar port).

- `kwolek2026_faquad.py`: LN anisotropy via `AnisotropicMaterial` (uniaxial
  tensor), angled-sidewall rib extrusion, the FDE calibration step
  (kappa(g) exponential fit + d(beta)/d(width) slope), the FAQUAD geometry
  of the paper's Eqs. 8-12 (`FaquadDesign`) with Euler (clothoid) S-bend
  separations -- parameterized by lateral offset and maximum waveguide-axis
  angle -- giving a smooth gap and a top-width difference that returns to
  zero at the device ends, and the parametric `faquad_combiner()` PCell.
- `kwolek2026_figures.py`: reproduces Fig. 1 (layout, gap/width profiles,
  FAQUAD mixing angle, supermodes, EME field propagation at FH and SH) and
  Fig. 2 (extinction-ratio and loss spectra at FH and SH).

Validation anchors (full-quality run): chi(0) = pi/2 and the Euler-S-bend
gap/dTW vary smoothly with dTW returning to zero at the device ends; the FH
input transfers adiabatically to the cross port (cross/bar = 0.89/0.07) while
the SH stays in the bar port with 22-25 dB extinction across 755-795 nm
(above the paper's > 19 dB); the FH loss is flat (~0.18 dB) across
1500-1600 nm. The remaining gap to the paper's dB-level FH figures
(> 25 dB FH extinction) is set by the example's EME discretization and by
the difference between our FDE-calibrated coupling and the paper's; both
improve with finer meshes,
more cells and more modes.

## Generalized dichroic beam-splitter designer

`dichroic_designer.py` generalizes the Magden 2018 approach into a
**platform-parametric designer** for adiabatic dichroic beam splitters with a
*targeted cutoff wavelength*. A `Platform` captures a single waveguide layer -
core/cladding materials, core thickness, sidewall angle, partial etch fraction,
the minimum fabricable tip width and gap, and a maximum device length - and:

- builds platform-aware rib cross-sections (partial-etch slab + angled
  sidewalls) and their FDE effective indices for a solid `WGA` strip and a
  sub-wavelength multi-rail `WGB`;
- `phase_match_width()` root-finds the `WGA` width that makes
  `n_WGA(lambda_c) = n_WGB(lambda_c)`, i.e. sets the targeted cutoff (on 220 nm
  SOI with the paper's WGB this recovers the paper's ~318 nm width at the
  ~1540 nm cutoff);
- `design_dichroic()` picks the largest coupling gap (sharpest cutoff) whose
  Landau-Zener phase-matching taper fits the length budget, allocates the four
  section lengths, and builds/extrudes the device on the platform.

The coupling `|kappa|` uses a scalar coupled-mode overlap with an empirical
high-contrast correction (calibrated to the silicon ~5/mm at the 750 nm gap),
so the extinction estimate is approximate - validate a final design with a
full EME. `main()` designs a few cutoffs on an SOI platform and writes
`figures/dichroic_designer.png`.

`dichroic_designer_si3n4.py` applies the designer to a **fully-etched 200 nm
Si3N4 / SiO2 platform** (50 nm minimum tip width and gap, 2 mm length budget),
designing splitters for cutoffs from 900 to 1200 nm in 50 nm steps. With a
3x200 nm-rail WGB the phase-match WGA width sweeps ~410-480 nm across the band;
`figures/dichroic_designer_si3n4.png` shows the index crossings, the design and
optimization outputs vs cutoff, and a designed device.

`dichroic_designer_si3n4_thickness.py` sweeps three Si3N4 core thicknesses
(200, 100, 40 nm) over the same cutoffs plus 990 nm. Thinner cores are more
weakly confined, so each uses a wider sub-wavelength WGB and needs wider WGA
strips (the 40 nm core sits near the edge of guidance, with micron-scale
waveguides). For each thickness it writes a result figure and a grid placing
each optimized device layout next to its simulated (coupled-mode) transmission
spectrum.

`dichroic_designer_slurm.py` is a **slurm-cluster version** of the Si3N4
designer. It designs the same splitters and, for each one, breaks the EME into
**subsets of cells run concurrently as separate slurm jobs** (`submit_runs` ->
`_analysis.submit_dichroic_run`): the dense short-/long-pass transmission
spectrum is distributed as overlapping slice-group jobs (the
`examples/parallel_eme_spectrum.py` decomposition), and - when full fields are
saved (`save_fields` keyword / `MEOW_SAVE_FIELDS` env var, default on) - the
propagation fields are distributed as single-cell jobs that keep each cell's
full mode fields. `gather_runs` (a later session) reattaches to those jobs,
assembles the results and writes into a fresh **timestamped subfolder** of the
MEOW jobs folder:

- a dense **short-/long-pass transmission spectrum** (`*_spectrum.png` + the raw
  arrays saved redundantly as `*_spectrum.csv` / `*_spectrum.json`);
- **intensity-propagation plots** `|Ex|^2(z, x)` at a few wavelengths on either
  side of and at the cutoff (`*_propagation.png`, when fields were saved), with
  the per-cell mode fields in a single compressed HDF5 dataset (`*_fields.h5`);
- a layout + WGA/WGB index-crossing **design figure** (`*_design.png`,
  analogous to `dichroic_designer.py`); and
- the device **GDS** (`*.gds`) and a scalar summary written redundantly as
  `*_summary.csv` / `*_summary.json`.

Because submitit persists each job in its `folder`, submit and gather can happen
in *different* python sessions (see "Reloading / gathering results in a later
python session" below). Spectrum/propagation wavelength bounds/counts are set by
`MEOW_SPECTRUM_*` / `MEOW_PROP_*` and the mesh/modes/cells resolution by
`MEOW_EXAMPLE_RES` (see `_analysis.py` / `_resolution.py`).

It also keeps the in-session EME helpers `run_blocking` / `run_concurrent`
(blocking vs. `asyncio.gather` over `meow.compute_s_matrix_parallel` /
`meow.acompute_s_matrix_parallel`) and a lighter S-matrix-only multi-session
path `submit_designs` / `gather_results` (built on `meow.ParallelEMEJobs`,
writing one `<cutoff>.eme.pkl` per design) for just the port powers.

`dichroic_coupler_slurm.py` is a focused **single-design** companion: rather than
an array of designs it prepares, asynchronously deploys and gathers the
distributed analysis of *one* adiabatic dichroic coupler. It walks through the
three stages explicitly - `design_coupler` (design one coupler), `submit`
(distribute the EME cell-subset jobs, returning immediately, into a timestamped
subfolder) and `gather`/`agather` (reload the persisted run record, assemble and
plot) - so the submit and gather steps can run in separate python sessions,
producing the same spectrum/propagation/design figures, GDS and data as above.

## Generalized FAQUAD wavelength-filter designer

`kwolek_designer.py` generalizes the Kwolek 2026 FAQUAD coupler into a
**platform-parametric designer** for X-cut thin-film lithium niobate (TFLN) and
lithium tantalate (TFLT). A `TFPlatform` captures one etched film - the
(anisotropic) core material, the film thickness, the rib etch depth, the
sidewall angle and the fabrication limits - and `design_faquad_filter()` designs
an Euler-S-bend FAQUAD coupler for a target fundamental/second-harmonic (FH/SH)
wavelength pair that transfers the FH to the cross port while the strongly
confined SH stays in the bar port (the dichroic filter action).

The design matrix spans both materials (`tfln_platform`, `tflt_platform`), the
four core thicknesses 300/400/500/600 nm (`CORE_THICKNESSES`) and the three
FH/SH pairs 1550/775, 1350/675 and 1060/530 nm (`WAVELENGTH_PAIRS`). LiTaO3 uses
the tabulated dispersion of Bond (1965); LiNbO3 reuses the Zelmon Sellmeier of
`kwolek2026_faquad.ln_material`. The optimization (within a device-length
budget) picks the largest top width whose FAQUAD device still meets the FH
extinction target - maximizing SH rejection (the FH/SH coupling contrast grows
with width) - then the shortest constant-gap length meeting the target
adiabaticity. `main()` writes `figures/kwolek_designer.png` (optimized widths,
the FH-vs-SH coupling contrast, and a designed layout) and then, for *every*
design, an analysis into `figures/kwolek_designer/<design>/`: a **dense,
broad-band bar/cross transmission spectrum** spanning more than an octave (from
0.8*SH to 1.2*FH; `*_spectrum.png` + the redundant `*_spectrum.csv` /
`*_spectrum.json`), the device **GDS**, and **propagating-field plots at the FH
and SH** (`*_propagation.png`, with the per-cell mode fields in a compressed
HDF5 `*_fields.h5`). The EME is distributed across local worker threads
(`analyze_design`).

`kwolek_designer_slurm.py` is the **slurm-cluster version**: it produces the
same per-design output (broad-band spectrum + GDS + FH/SH field plots) but
distributes each design's EME as concurrent cell-subset jobs (`submit_runs` ->
`_analysis.submit_faquad_run`). Like the dichroic examples, **each job runs the
whole wavelength sweep within a single task**: the broad spectrum is a
slice-group spectrum job set over *dispersive* cells (the LN/LT tensor is
wavelength-sampled, so a single task sweeps the band by varying only the
environment wavelength), plus single-cell field jobs at the FH/SH when fields
are saved. `gather_runs` assembles and writes the figures, GDS and data into a
timestamped subfolder. It also keeps the in-session EME helpers `run_blocking` /
`run_concurrent` and the lighter S-matrix-only `submit_designs` /
`gather_results` multi-session path (one FH and one SH `.eme.pkl` per design)
for just the figures of merit.

`dichroic_designer_si3n4_thickness_slurm.py` is the **slurm-cluster version of
the thickness sweep** (`dichroic_designer_si3n4_thickness.py`). It designs the
fully-etched Si3N4 splitters across the three core thicknesses (200/100/40 nm)
and the 900-1200 nm cutoffs, then runs *all the simulation, analysis and
plotting of each (thickness, cutoff) design asynchronously as slurm jobs* - each
design's EME broken into concurrent cell-subset jobs writing its
spectrum/propagation/design figures, GDS and data into a fresh timestamped
subfolder. As with the others, `submit_runs` returns immediately and
`gather_runs` assembles every run in a later session.

## Running

```sh
uv run python -m examples.papers.magden2018_figures
uv run python -m examples.papers.kwolek2026_figures
uv run python -m examples.papers.dichroic_designer
uv run python -m examples.papers.dichroic_designer_si3n4
uv run python -m examples.papers.dichroic_designer_si3n4_thickness
uv run python -m examples.papers.dichroic_designer_si3n4_thickness_slurm
uv run python -m examples.papers.dichroic_designer_slurm
uv run python -m examples.papers.dichroic_coupler_slurm
uv run python -m examples.papers.kwolek_designer
uv run python -m examples.papers.kwolek_designer_slurm
```

Figures for the non-slurm examples are written to `examples/papers/figures/`.
The slurm examples instead write each design's figures, GDS and data into a
timestamped subfolder of the MEOW jobs folder (`MEOW_SLURM_FOLDER`, default
`meow_*_jobs/`).

**Resolution.** Every example takes a `MEOW_EXAMPLE_RES` resolution level in
`{low, medium, high}` (default `medium`) instead of the old boolean
`MEOW_EXAMPLE_FAST`:

- `low` - a coarse smoke-test resolution (used by `src/tests/test_paper_examples.py`;
  `MEOW_EXAMPLE_FAST=1` is still honoured and maps to `low`);
- `medium` - the previous full-quality settings (tens of minutes);
- `high` - finer mesh resolution, more modes per cross-section and more EME
  cells, increased to the point where the simulated quantities are expected to
  be converged (slow).

`pick(low=..., medium=..., high=...)` (now `meow.settings.pick`, re-exported by
the thin `_resolution.py` shim) chooses each per-knob value for the active level. The two main EME knobs have a fixed converged
standard at `high` - **128 EME cells and 8 modes** per cross-section - which is
also the default `num_cells` / `num_modes` of every example function. Two env
vars override them directly (at any level, including `high`):

| env var | overrides |
| --- | --- |
| `MEOW_NUM_CELLS` | number of EME cells |
| `MEOW_NUM_MODES` | number of modes per cross-section |

```sh
# converged run; or override just the cell/mode counts
MEOW_EXAMPLE_RES=high uv run python -m examples.papers.dichroic_designer_slurm
MEOW_NUM_CELLS=256 MEOW_NUM_MODES=12 uv run python -m examples.papers.dichroic_coupler_slurm
```

### Backends and parallel resources

Every example resolves its FDE mode-solver backend and its parallel/slurm
resource settings from environment variables. These now live in the library, in
`meow.settings` (the `_backends.py` shim just re-exports them):

| env var | meaning | default |
| --- | --- | --- |
| `MEOW_PAPER_BACKEND` | mode solver: `tidy3d`, `mpb` or `lumerical` | `tidy3d` |
| `MEOW_CPUS_PER_TASK` | cpus per parallel task / local worker count | `2` |
| `MEOW_TIMEOUT_MIN` | per-job wall-clock limit [min] | `60` |
| `MEOW_SLURM_PARTITION` | slurm partition to submit to | (unset) |
| `MEOW_SLURM_CLUSTER` | submitit cluster: `slurm`/`local`/`debug` | `local` |
| `MEOW_MAX_WORKERS` | local worker count (else `MEOW_CPUS_PER_TASK`) | (unset) |

The chosen backend is threaded all the way through to the parallel slice-group
jobs and the single-cell field jobs, so `tidy3d`/`mpb`/`lumerical` work both for
the in-session local runs and for the slurm jobs (the slice-group cascade needs
a *deterministic* backend - tidy3d or seeded mpb). `MEOW_CPUS_PER_TASK`,
`MEOW_TIMEOUT_MIN` and `MEOW_SLURM_PARTITION` are applied to every executor the
examples build (`make_executor`) and, as the worker count, to the local
multithreaded/multiprocess runs.

```sh
# pick the mpb backend; 8 cpus and a 2-hour limit per job on the "cpu" partition
MEOW_PAPER_BACKEND=mpb uv run python -m examples.papers.magden2018_figures
MEOW_SLURM_CLUSTER=slurm MEOW_SLURM_PARTITION=cpu \
MEOW_CPUS_PER_TASK=8 MEOW_TIMEOUT_MIN=120 MEOW_PAPER_BACKEND=tidy3d \
  uv run python -m examples.papers.dichroic_designer_slurm submit
```

`MEOW_PAPER_PARALLEL=1` additionally makes the figure scripts
(`magden2018_figures`, `kwolek2026_figures`) cascade their device EME with the
parallel slice-group engine instead of the serial path.

### Running EME on a slurm cluster

`dichroic_designer_slurm.py` distributes the per-slice mode solves of each
device's EME as independent jobs via `meow.slurm_executor`, which wraps a
[submitit](https://github.com/facebookincubator/submitit) `AutoExecutor`. The
same code runs the jobs as local subprocesses, in-process, or on a real slurm
cluster - only the executor's `cluster` selector changes.

**1. Install submitit.** It is an optional dependency, needed on the machine
that *submits* the jobs (a slurm login node, or your laptop for local runs) and
on the cluster *compute* nodes that run them:

```sh
pip install submitit          # or: uv pip install submitit
```

**2. Pick an executor backend.** The example builds its executor through
`make_executor(folder, cluster, *, timeout_min, cpus_per_task, mem_gb,
slurm_partition)`, which forwards to `meow.slurm_executor`. The `cluster`
argument (or the `MEOW_SLURM_CLUSTER` environment variable) selects the
backend:

| `cluster`  | where jobs run                          | use for                          |
| ---------- | --------------------------------------- | -------------------------------- |
| `"debug"`  | in the calling process (serial)         | quick checks / the test suite    |
| `"local"`  | local subprocesses (`submitit` local)   | a multi-core workstation (default here) |
| `"slurm"`  | submitted to slurm via `sbatch`         | a real cluster                   |
| `None`     | slurm if available, else local          | auto-detect                      |

By default the example uses `"local"` so it runs anywhere. To dispatch to slurm,
set the environment variables on the login node and run the module:

```sh
export MEOW_SLURM_CLUSTER=slurm
export MEOW_SLURM_PARTITION=cpu      # your cluster's partition name
export MEOW_SLURM_FOLDER=$HOME/meow_dichroic_jobs   # on a shared filesystem
uv run python -m examples.papers.dichroic_designer_slurm
```

**3. Configure the slurm resources.** `make_executor`/`slurm_executor` expose
the common per-job knobs - `timeout_min` (wall-clock limit), `cpus_per_task`,
`mem_gb`, `slurm_partition` - and pass any extra keyword through to
submitit's `update_parameters` (e.g. `slurm_array_parallelism`, `gpus_per_node`,
`slurm_additional_parameters={"account": "..."}`). Size `cpus_per_task`/`mem_gb`
to a single cross-section mode solve, since each job solves a small group of
slices (a triplet or a pair of cells), not the whole device. The
`MEOW_SLURM_PARTITION` env var overrides the partition without editing code.

**4. Use a shared filesystem.** submitit communicates with each job entirely
through its `folder` (the pickled callable + arguments go in, logs and the
pickled result come out). On a cluster that folder **must live on a filesystem
visible to both the login node and the compute nodes** (e.g. your `$HOME` or a
scratch mount) - set it with `MEOW_SLURM_FOLDER`. Each job there is a
self-contained record, which is what enables multi-session collection (below).

**5. Match the software stack on the compute nodes.** The parallel engine
re-solves shared boundary cells in separate jobs and checks that they return
identical effective indices, so every job must run the *same deterministic*
mode solver and the *same* meow/tidy3d (or MPB) versions as the submitter.
submitit embeds the submitting interpreter (`sys.executable`) in the generated
sbatch script, so the cleanest approach is to **submit from the exact
environment you want the jobs to run in** - activate your meow venv/conda env on
the login node before launching - so the compute nodes import the same `meow`
and backend. If your cluster needs extra job setup (module loads, sourcing a
conda env), pass it through with submitit's `setup` parameter, e.g.
`make_executor(..., setup=["module load anaconda", "conda activate meow"])`
(forwarded to `update_parameters`).

**Blocking vs. async execution.** Both workflows submit the *same* cluster
jobs; they differ only in how the submitting session waits:

- `run_blocking(designs, executor=...)` parallelizes one device's EME across the
  cluster, blocks until it finishes, then moves to the next device. Simplest;
  the cluster is busy with one device at a time.
- `run_concurrent(designs, executor=...)` is `async`: it submits *every*
  device's jobs up front and awaits them together with `asyncio.gather`
  (`meow.acompute_s_matrix_parallel` polls the jobs without blocking the event
  loop), so all the design workflows are in flight at once and the cluster's
  scheduler packs them across all available nodes. Drive it with
  `asyncio.run(run_concurrent(designs, executor=make_executor()))`.

**Reloading / gathering results in a later python session.** Because every job
is persisted in the submitit `folder`, submission and collection do not have to
happen in the same process. A sweep can be launched from one (short-lived)
session and its results gathered later from another, as long as both point
`MEOW_SLURM_FOLDER` at the *same shared folder* and the jobs outlive the
submitting process (true for `cluster="slurm"`, where the work runs under
`sbatch`, not inside your python process).

meow exposes **submit/collect splits** for exactly this. Each one submits the
jobs and returns a *picklable* handle *without blocking*; save it with
`handle.save(path)` right after submitting and `<Handle>.load(path)` it in a
later session to reattach to the still-running jobs (it pickles the submitit
jobs, which reload their results from `folder`). Poll without blocking with
`handle.done()` and inspect `handle.job_ids` / `handle.folder`:

> Pickle is used **only** for these in-flight job handles (their live submitit
> job objects cannot be represented otherwise). All *computed result data* is
> saved with the library's xarray helpers instead: the dense per-cell mode
> fields go into a single compressed HDF5 dataset via `meow.save_fields` /
> `meow.ParallelFieldModeJobs.save_fields`, and the less dense tabular data
> (spectra, summaries) is written redundantly as CSV **and** JSON via
> `meow.save_table` / `meow.save_summary` (see `meow.eme.io`).

- `meow.submit_s_matrix_parallel(cells, env, executor=...)` →
  `meow.ParallelEMEJobs`: the slice-group jobs of a single-wavelength EME;
  `handle.result()` cascades the full S-matrix.
- `meow.submit_s_matrix_spectrum(cells, env, executor=..., wls=...)` →
  `meow.ParallelEMESpectrumJobs`: the slice-group jobs of a dense **spectrum**
  (each job solves its cells at every wavelength); `handle.result()` returns one
  `(S, port_map)` per wavelength. This is the field-free decomposition used when
  full fields are not needed (as in `examples/parallel_eme_spectrum.py`).
- `meow.submit_cell_modes(cells, env, executor=...)` →
  `meow.ParallelFieldModeJobs`: **one job per cell** that keeps each cell's
  *full mode fields*; `handle.result()` returns the per-cell modes, for field
  reconstruction / propagation (e.g. `meow.propagate_modes`) or
  `meow.compute_s_matrix`. This is the single-cell decomposition used when full
  fields are saved for subsequent analysis.

The **default analysis workflow** of every slurm example uses these to break
each design's EME into **subsets of cells run concurrently as separate slurm
jobs**: `submit_runs` (the single-coupler example: `submit`) submits the dense
transmission spectrum as slice-group spectrum jobs and - when full fields are
saved (`save_fields` keyword / `MEOW_SAVE_FIELDS` env var, default on) - the
propagation fields as single-cell jobs, persisting one picklable run record
(`run.pkl`) per design into a fresh **timestamped subfolder** of
`MEOW_SLURM_FOLDER`. `gather_runs` (`gather` / `agather`) walks those subfolders
in a later session, reattaches to the jobs, assembles the spectrum + propagation
and writes the figures, GDS and data. To reattach to *one specific* run instead,
`meow`'s example helper `examples.papers._slurm.load_run(run_dir)` loads a single
run from its timestamped directory (or its `run.pkl` file); `load_runs(folder)`
loads them all and **skips any corrupt or version-incompatible `run.pkl`** with a
warning rather than crashing. Spectrum/propagation wavelength bounds and counts
are set by `MEOW_SPECTRUM_SPAN` / `MEOW_SPECTRUM_NPTS` and `MEOW_PROP_SPAN` /
`MEOW_PROP_NPTS` (or an explicit `MEOW_PROP_WLS` list); the mesh / modes / cell
resolution by `MEOW_EXAMPLE_RES`.

A lighter S-matrix-only path is also available on the array examples
(`submit_designs` / `gather_results`): it writes one small `<label>.eme.pkl`
record (the `ParallelEMEJobs` handle plus the few scalars needed for the port
powers) into the job folder, for when you only want the figures of merit and not
the plots. Every example module exposes `submit` and `gather` subcommands that
drive the analysis workflow:

```sh
# session A (login node): submit the sweep to the cluster and return at once
MEOW_SLURM_CLUSTER=slurm MEOW_SLURM_PARTITION=cpu \
MEOW_SLURM_FOLDER=$HOME/meow_dichroic_jobs \
  uv run python -m examples.papers.dichroic_designer_slurm submit

# session B (later, same $MEOW_SLURM_FOLDER): reload the persisted run handles
# and collect the summaries once the cluster jobs have finished (the figures,
# GDS and data are already in each run's timestamped subfolder)
MEOW_SLURM_CLUSTER=slurm MEOW_SLURM_PARTITION=cpu \
MEOW_SLURM_FOLDER=$HOME/meow_dichroic_jobs \
  uv run python -m examples.papers.dichroic_designer_slurm gather
```

The single-design `dichroic_coupler_slurm.py` works the same way (`submit` then
`gather`), and demonstrates the async path explicitly - `submit` returns the
moment the jobs are queued and `agather` awaits them. The lower-level building
blocks are the meow submit/collect handles - e.g. a dense spectrum distributed
as slice-group jobs, with full fields disabled:

```python
import meow as mw

# session A: distribute the spectrum jobs, then exit
executor = mw.slurm_executor(folder="$HOME/meow_jobs", cluster="slurm")
handle = mw.submit_s_matrix_spectrum(
    cells, env, executor=executor, wls=wls, num_modes=4
)
handle.save("$HOME/meow_jobs/spectrum.pkl")     # nothing else needed

# session B (later): reload and collect one (S, port_map) per wavelength
handle = mw.ParallelEMESpectrumJobs.load("$HOME/meow_jobs/spectrum.pkl")
if handle.done():
    spectra = handle.result()

# to keep the full mode fields instead (single-cell jobs, for propagation):
#   handle = mw.submit_cell_modes(cells, env, executor=executor, num_modes=4)
#   modes_per_cell = mw.ParallelFieldModeJobs.load(path).result()
```

Note on fidelity: quantitative numbers (cutoff wavelength, dB-level losses
and extinction ratios) are mesh- and discretization-sensitive. The
segmented-waveguide effective index of the Magden filter needs <= 20 nm mesh
resolution to converge, and the dB-scale extinction floors of the FAQUAD
combiner keep improving with finer meshes, more EME cells and more modes than
the example defaults.
