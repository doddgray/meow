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
  (kappa(g) exponential fit + d(beta)/d(width) slope), and the paper's
  **three-region** FAQUAD geometry (`FaquadDesign`, Eqs. 8-12):

  - **Region I** -- a constant minimum-gap `g_m` straight interaction section;
  - **Region II** -- the **cubic separation bend** `g_e(z) = (2/3) a^2 (z -
    l_m/2)^3 + g_m` (Eq. 9), the paraxial Euler approximation that gives the
    closed-form coupling envelope `kappa(z) = kappa_m exp(-((|z|-l_m/2)/z_0)^3)`
    (Eq. 10), continuing the FAQUAD evolution to the decoupling gap `g_c`;
  - **Region III** -- a genuine **Euler (clothoid) bend matched to the cubic
    exit angle** that relaxes the curvature to zero into the straight outer
    waveguide at the final gap `g_f`.

  The whole device is laid out as a **single `gdsfactory` path extruded with
  two parametric-width sections** (`combiner_from_design` /
  `faquad_combiner()`): each rib's `width_function`/`offset_function` traces its
  top width `w(z)` and centerline `g(z)/2 + w/2` along all three regions, and
  the 65-degree sidewalls are added at extrusion. Three top-width-taper
  `VARIANTS` (`faquad_bends`, `faquad_taper`, `linear_taper`) share the gap
  profile but differ in the taper, for the Fig. 2a comparison.
- `kwolek2026_figures.py`: reproduces, **for each LiNbO3 material model**
  (`anisotropic` -- the real uniaxial crystal -- and `isotropic` -- a fake LN
  with the extraordinary index on every axis; see below), Fig. 1 (layout,
  gap/width profiles, mixing angle, supermodes, FH/SH propagation), **every
  subfigure of Fig. 2** (a: FH extinction ratio for the three taper variants;
  b: SH extinction ratio; c: total and radiated loss at FH and SH; d: the FH
  fabrication-tolerance map over etch depth x top width), **all of supplemental
  Fig. 5** (a/b: mixing-angle and coupling-magnitude error over the (gap, dTW)
  plane vs the FDE sweep, with the design trajectory overlaid; c: realized
  adiabaticity eta(z) for the designed / constant-width / FAQUAD bends), and a
  **broad-band (> 1 octave, ~0.8*SH .. 1.2*FH) bar/cross transmission** plot.
  Outputs are written suffixed by model, e.g. `kwolek2026_fig2_anisotropic.png`.

**Two material models.** Everything is run with both `ln_material(wl,
"anisotropic")` (the real `(ne^2, no^2, no^2)` tensor) and `ln_material(wl,
"isotropic")` (a deliberately fake `(ne^2, ne^2, ne^2)` LN). The TE/TM mode
crossings at the SH band are a purely anisotropic effect; running the isotropic
model alongside isolates their influence on the SH transmission.

**Launch / metric / polarization.** The device operates on the **TE** mode, so
the input is the fundamental TE rib mode (`input_launch_index`, `te_fraction >
0.5`) -- not simply the highest-`neff` mode, which at the second harmonic is
actually **TM**; launching TM was an earlier bug that made the SH field look
like a higher-order mode and made the SH transmission meaningless. Bar/cross is
then the EME output power summed over the modes **localized in each output rib**
(`port_mode_indices`, by spatial confinement), with everything non-localized
counted as loss -- a metric that is stable in `num_modes` where a naive
"classify every mode by centroid sign" is not.

**FH vs SH (honest status).** With the correct TE launch, the **fundamental**
behaves as intended: it transfers to the cross port with **cross ~ 0.9-0.98**
and low loss, converging well in mesh/cells/modes (chi(0)=pi/2, the gap/dTW vary
smoothly through the cubic + Euler bends, dTW->0 at the ends). The **second
harmonic is intrinsically hard**: the rib is strongly multimode at 775 nm, so the
serial EME cascade (which holds every cell's modes at once) needs more modes than
fit in memory to fully conserve power. The SH stays in the bar port (bar/cross
contrast is large) but a non-negligible fraction scatters among the dense SH
modes, and the SH numbers are **not converged to ~1%** at a feasible cost. (An
earlier "deep-etched 500/400 nm" variant appeared to give ~0.9 SH bar at ~20 dB,
but that was the TM-launch artifact; with the correct TE launch the deep ridge is
*more* multimode, so the example uses the paper's shallow 300/100 nm stack.)

**Interaction length and the modeled coupling.** The paper's final design uses
`l_m = 264 um` (`kwolek2026_faquad.L_M`) at its *measured* coupling. meow's
calibrated -- and resolution-converged -- coupling for this stack is about 2x
weaker (`kappa_m ~ 0.0074/um`), so reaching the same FAQUAD adiabaticity
(`eta ~ 0.19-0.22`, the regime of a clean, complete FH transfer) takes a
proportionally longer interaction length: the reproduction figures use
`FIG_L_M ~ 520 um` (`kwolek2026_figures`), at which the FH cross transfer is a
clean `~ 0.98`. The FAQUAD methodology (design at constant adiabaticity, the knob
being `l_m`) is identical; only the absolute length scales with the modeled
coupling strength.

**Convergence (to ~1%).** FH: mesh `Δ ~ 0.02-0.03 µm`, `~150-200` cells,
`~10-12` modes. SH: needs `Δ ~ 0.015 µm` and many more modes (`>= 20-24`) than
the serial cascade can hold in memory alongside the cells, which is why SH is not
1%-converged; a distributed / chunked EME (or the `meow.fde.sparse` operator
path) would be the route to a converged SH.

**Test structures (paper Fig. 3).** `kwolek2026_test_structures.py` emits the
companion passives for measuring excess loss / intrinsic Q after fab: all-pass
and add-drop micro-rings (radius + gap sweeps), and the **Fig. 3 FH
characterization layout** -- a `dut_resonator` (a racetrack whose coupler *is*
the FAQUAD device, closed into a ring with matched Euler bends) above a plain
`control_resonator` of the same footprint (`fh_measurement_layout`) -- all on
the same angled-sidewall rib layer, written to GDS.

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

**Spectrum grid, port tapers and test structures (shared `_designer_extras`).**
The wavelength-varying designers (`kwolek_designer` and the `dichroic_designer`
family) share three helpers in `_designer_extras.py`:

- `spectrum_grid` -- a **column of the dense broad-band spectra**, one row per
  design, all on the **same wavelength axis**, with each design's target
  wavelength(s) drawn as **dashed vertical lines**. `kwolek_designer` writes it
  as `figures/kwolek_designer_spectrum_grid.png` (FH/SH marked), the dichroic
  designer as `figures/dichroic_designer_spectrum_grid.png` (cutoff marked).
- `tapered_ports` -- optional **linear access tapers** from the device edge
  width to a target width at each named port, controlled by the
  `port_widths` / `taper_lengths` keyword arguments of `faquad_combiner`
  (Kwolek) and `tapered_component` (dichroic); the **default adds no taper** and
  keeps the designed edge widths.
- `coupler_cutback_array` -- a **test-structure array** with a varied number of
  cascaded cross/bar couplings but a **constant total waveguide length** between
  regularly-spaced ports on either side of a **5 mm-wide chip** (the cut-back
  layout for the per-coupler excess loss), plus a routed binary `splitter_tree`.

`kwolek_designer.design_test_structures` writes, per design, the cut-back array
**and all the paper's resonant-loss structures** (all-pass + add-drop rings and
the Fig. 3 DUT / control racetracks) as GDS and a labelled preview;
`dichroic_designer.dichroic_test_structures` writes the cut-back array for the
dichroic coupler.

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

## AD gradient-based design optimization

`dichroic_designer.py` and `kwolek_designer.py` can optimize their nominal
design parameter directly by **`jax.grad`** through
`meow.make_differentiable_neffs` (see the library's
[HPC & gradients guide](../../docs/hpc.md)), instead of the default
root-find/bisection, over the *same fixed layer stack* used everywhere else in
the module:

- **`dichroic_designer.optimize_phase_match_width`** minimizes the phase-mismatch
  loss `(n_WGA(w_a, cutoff_wl) - n_WGB(cutoff_wl))^2` over the WGA width `w_a` at
  the target cutoff - the gradient-descent counterpart of `phase_match_width`'s
  `brentq` root-find. Pass `design_dichroic(..., use_gradient=True,
  gradient_w0=...)` to use it in a full design.
- **`kwolek_designer.optimize_width_gradient`** directly **maximizes the FH/SH
  coupling contrast** `log(kappa_FH) - log(kappa_SH)` over the nominal top width
  `w_top` at the fixed minimum gap - the literal Kwolek design goal stated as a
  scalar objective, with a soft feasibility penalty (so the optimizer finds a
  genuine interior optimum instead of saturating at the width bound) - the
  gradient-based counterpart of `optimize_width`'s bisection-on-feasibility. Pass
  `design_faquad_filter(..., use_gradient=True, gradient_w0=...)` to use it.

Both objectives are built from the effective-index **splitting**/**crossing** of
a two-waveguide cross-section, which is exactly what `make_differentiable_neffs`
differentiates exactly from a *single* eigensolve (meow's tidy3d cross-section
builder already applies Kottke subpixel smoothing, so the width -> permittivity
map is smooth enough for its default finite-difference `eps` Jacobian) -
`jax.grad` + a small Adam optimizer (`examples/papers/_ad_optimize.py`) then
walks the width to the optimum, recording an **optimization trace** (objective
and parameter value per iteration).

`ad_optimization_figure()` in each module runs the optimizer from a
deliberately off-target initial width and plots, in one figure: the
optimization trace, the before/after performance (the index-crossing curve for
dichroic; FH/SH coupling vs. width for Kwolek) at the initial vs. optimized
width, and the optimized device layout. `main()` writes these as
`figures/dichroic_designer_ad_optimization.png` and
`figures/kwolek_designer_ad_optimization.png`.

This pattern generalizes to any designer whose target quantity is a neff
crossing or splitting (most of the wavelength-varying designers in this
directory); `mao2019_designer.py`, `ramadan1998_designer.py` and
`song2023_designer.py` are not yet converted and remain on their original
closed-form / bisection designers.

### Two-stage joint optimization over every practical dichroic parameter

`dichroic_designer.design_dichroic_joint` extends the single-parameter
`optimize_phase_match_width` to a **two-stage joint optimization** over every
practical degree of freedom of the dichroic beam splitter, each stage driven
by `jax.grad` through `meow.make_differentiable_objective` (exact central
finite differences of the whole FDE-based design objective):

1. **`optimize_dichroic_crosssection`** picks the WGB shape - its rail-width
   scale `w_b`, its **inter-rail gap** `g_b`, and the **fractional
   middle/outer rail widths** `frac_mid`/`frac_out` (`mid_width = frac_mid *
   w_b`, `out_width = frac_out * w_b`; `1.0` recovers the uniform-rail-width
   WGB) - to **maximize the group-velocity mismatch** with WGA, subject to
   the two being **exactly** phase-matched at the target cutoff. The WGA
   width `w_a` is *not* a free parameter here: for every candidate WGB it is
   root-found by `phase_match_width` (the same `brentq` solve
   `design_dichroic` uses), so every point the optimizer visits - and
   therefore its result - has a genuine mode crossing at the target
   wavelength by construction. (An earlier version of this loss instead
   added the phase-match residual `(n_WGA - n_WGB)^2` as a *soft* penalty
   alongside the group-velocity-mismatch reward; because that reward is
   linear and unbounded while the residual is a bounded quadratic, the
   optimizer could settle at a small nonzero mismatch - trading away an
   exact crossing for a larger mismatch - so the resulting device did not
   actually filter at the target wavelength. Root-finding `w_a` inside the
   loss removes that trade-off entirely.) The loss is
   `-gvm_weight * sign * (ng_WGA - ng_WGB)`, where the group index `ng = n -
   wavelength * dn/dwavelength` is a central finite difference over
   wavelength of the isolated-waveguide effective index, and `sign`
   (`reference_gvm_sign`, computed once from the original Magden 2018 SOI
   design: the solid WGA strip has the higher group index - is more
   dispersive - than the segmented WGB) orients the mismatch to match that
   design's short-pass (WGA) / long-pass (WGB) convention. A sharper
   (higher-group-index-mismatch) crossing gives better spectral selectivity
   away from the cutoff. The WGA-WGB coupling gap is **not** a parameter
   here either - it has no effect on either isolated-waveguide quantity,
   only on the coupling `kappa` - so it moves to stage 2 instead.
2. **`optimize_dichroic_lengths`** then takes that *fixed* cross-section and
   picks the coupling `gap` and the four section lengths `l1..l4` (up to a
   5 mm total budget) to **minimize the predicted insertion loss while
   keeping the phase-matching transition adiabatic**: the loss is `-ER[dB]`
   (the Landau-Zener extinction, unbounded and always improvable with more
   length or a smaller/stronger-coupling gap) plus a small compactness
   preference on the non-critical section lengths (`l1`, `l3`, `l4` - not
   `l2`, the phase-matching taper whose length is what actually controls the
   adiabaticity) plus a hard penalty beyond the length budget. With no
   explicit penalty on the gap itself, this reliably drives it to the
   tightest coupling that still fits the remaining budget into `l2`. Because
   `kappa` is a *difference* of two overlapping-but-separated mode fields (a
   numerically delicate quantity, unlike the isolated-waveguide indices in
   stage 1), this stage's finite-difference step defaults larger than usual
   (`fd_step=1e-2`) - the default `1e-3` is small enough that solver noise
   can flip its sign.

Both stages optimize in **bounds-normalized** coordinates (each parameter
mapped to `[0, 1]` over its box bound) so a single learning rate is
meaningful across the mixed micron/dimensionless/length-in-microns parameter
scales.

`WGB` and `magden2018_dichroic.dichroic_filter` (plus its `lateral_positions`/
`w_b_total` helpers) were generalized with `frac_mid`/`frac_out` parameters to
support the heterogeneous-rail-width layout stage 1 can produce; both default
to `1.0`, so every pre-existing call site (with a uniform-width WGB) is
unaffected.

`design_dichroic_joint()` runs both stages and builds the resulting
`DichroicDesign` (`opt_trace` is stage 1's trace, `opt_trace_lengths` stage
2's); `joint_ad_optimization_figure()` runs it from deliberately off-target
initial guesses and plots both loss traces, both parameter trajectories, the
before/after index-crossing performance, and the optimized layout, written as
`figures/dichroic_designer_joint_ad_optimization.png` (and the
`_si3n4`/`_si3n4_200nm` counterparts in the Si3N4 variants). Because
`make_differentiable_objective` re-solves the whole objective (including,
for stage 1, an inner root-find) per finite-difference step, this
optimization is substantially more expensive per iteration than the
single-parameter path; the demo therefore uses a coarser mesh resolution and
fewer iterations than the discrete-sweep designs in the same module.

Passing `analysis_dir=...` to `joint_ad_optimization_figure()` additionally
runs `analyze_dichroic_design(..., save_fields=True)` on the optimized
design, writing its broadband EME short-/long-pass transmission spectrum
(`*_spectrum.png`) and its propagating-field intensity plot at the cutoff
wavelength (`*_propagation.png`, plus the raw fields as `*_fields.h5`)
alongside the GDS/design/summary files - the same distributed-EME machinery
`dichroic_designer_slurm.py` and `kwolek_designer.analyze_design` use. All
three `main()`s (`dichroic_designer`, `dichroic_designer_si3n4`,
`dichroic_designer_si3n4_thickness`'s 200 nm case) pass this so every
joint-optimized design gets its own `*_joint/` analysis folder.

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
