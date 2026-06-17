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

Validation anchors (full-quality run): chi(0) = pi/2 and the closed-form
eta of Eq. 12 hold exactly; the FH input transfers adiabatically to the
cross port (cross/bar = 0.63/0.28) while the SH stays in the bar port with
17-19 dB extinction across 755-795 nm, close to the paper's > 19 dB; the
FH loss is flat (~0.4 dB) across 1500-1600 nm. The remaining gap to the
paper's dB-level figures (> 25 dB FH extinction, < 0.1 dB loss) is set by
the example's EME discretization and by the difference between our
FDE-calibrated coupling and the paper's; both improve with finer meshes,
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
designer. It designs the same splitters and then runs a *full-device EME* of
each one with the parallel slice-group engine
(`meow.compute_s_matrix_parallel`), distributing the per-slice mode solves as
independent cluster jobs through a `meow.slurm_executor`. It exposes two
workflows over a list of designs:

- **blocking** (`run_blocking`): each device's EME is parallelized across the
  cluster, and the designs are processed one after another (`eme_ports` ->
  `meow.compute_s_matrix_parallel`); and
- **concurrent / async** (`run_concurrent`): every device's EME is submitted at
  once and awaited together with `asyncio.gather`
  (`aeme_ports` -> `meow.acompute_s_matrix_parallel`), so all the design
  workflows' jobs are in flight on the cluster simultaneously.

Because submitit persists each job (payload, logs and result) in its job
`folder`, the concurrent jobs can also be submitted from one python session and
their results collected from a *different* session - see "Running EME on a
slurm cluster" below.

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
the FH-vs-SH coupling contrast, and a designed layout).

`kwolek_designer_slurm.py` is the **slurm-cluster version**. It designs the full
(material x thickness x FH/SH) matrix and runs a *full-device EME at both the FH
and the SH* of each design with the parallel slice-group engine
(`meow.compute_s_matrix_parallel`), reporting the FH cross / SH bar
transmissions and extinction ratios. Like the dichroic slurm example it exposes
both a **blocking** (`run_blocking` -> `eme_filter`) and a **concurrent / async**
(`run_concurrent` -> `aeme_filter` -> `asyncio.gather`) workflow, so every
design's jobs can be in flight on the cluster at once; pass a
`meow.slurm_executor` (or rely on local subprocesses by default).

## Running

```sh
uv run python -m examples.papers.magden2018_figures
uv run python -m examples.papers.kwolek2026_figures
uv run python -m examples.papers.dichroic_designer
uv run python -m examples.papers.dichroic_designer_si3n4
uv run python -m examples.papers.dichroic_designer_si3n4_thickness
uv run python -m examples.papers.dichroic_designer_slurm
uv run python -m examples.papers.kwolek_designer
uv run python -m examples.papers.kwolek_designer_slurm
```

Figures are written to `examples/papers/figures/`. The default settings take
tens of minutes; set `MEOW_EXAMPLE_FAST=1` for a coarse smoke-test version
(used by `src/tests/test_paper_examples.py`).

### Backends and parallel EME

Both examples are backend- and parallel-aware (see `_backends.py`):

- `MEOW_PAPER_BACKEND=tidy3d|mpb|lumerical` selects the FDE mode solver used
  for all the examples' serial solves (default `tidy3d`). The MPB backend
  needs the `meep`/`mpb` conda-forge bindings.
- `MEOW_PAPER_PARALLEL=1` cascades the device EME with the parallel
  slice-group engine (`meow.compute_s_matrix_parallel`) instead of the serial
  path. The parallel engine re-solves shared cells in separate processes and
  checks them for consistency, so it always uses the deterministic tidy3d
  backend (the two knobs are independent: parallel runs do not use MPB).

```sh
MEOW_PAPER_BACKEND=mpb uv run python -m examples.papers.magden2018_figures
MEOW_PAPER_PARALLEL=1 uv run python -m examples.papers.kwolek2026_figures
```

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

**Running across multiple python sessions.** Because every job is persisted in
the submitit `folder`, submission and collection do not have to happen in the
same process. A long sweep can be launched from one (short-lived) session and
its results gathered later from another, as long as both point
`MEOW_SLURM_FOLDER` at the *same shared folder* and the jobs outlive the
submitting process (true for `cluster="slurm"`, where the work runs under
`sbatch`, not inside your python process). Concretely: call `make_executor()`
with a fixed `folder`, submit with `executor.submit(...)` /
`compute_s_matrix_parallel`, record the returned submitit job ids, and in a
second session reconstruct the jobs from that folder (e.g.
`submitit.AutoExecutor(folder=...)` / `submitit.SlurmJob(folder=..., job_id=...)`)
and read their `.result()`. This lets several design workflows run concurrently
on the cluster while you start, stop and reattach python sessions on the login
node.

```sh
# session A (login node): launch the sweep on the cluster
MEOW_SLURM_CLUSTER=slurm MEOW_SLURM_PARTITION=cpu \
MEOW_SLURM_FOLDER=$HOME/meow_dichroic_jobs \
  uv run python -m examples.papers.dichroic_designer_slurm

# session B (later, same $MEOW_SLURM_FOLDER): collect results from the
# persisted submitit jobs once they have finished
```

Note on fidelity: quantitative numbers (cutoff wavelength, dB-level losses
and extinction ratios) are mesh- and discretization-sensitive. The
segmented-waveguide effective index of the Magden filter needs <= 20 nm mesh
resolution to converge, and the dB-scale extinction floors of the FAQUAD
combiner keep improving with finer meshes, more EME cells and more modes than
the example defaults.
