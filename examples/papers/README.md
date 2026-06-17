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
  (kappa(g) exponential fit + d(beta)/d(width) slope), the closed-form
  FAQUAD geometry of the paper's Eqs. 8-12 (`FaquadDesign`), and the
  parametric `faquad_combiner()` PCell.
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

## Running

```sh
uv run python -m examples.papers.magden2018_figures
uv run python -m examples.papers.kwolek2026_figures
uv run python -m examples.papers.dichroic_designer
uv run python -m examples.papers.dichroic_designer_si3n4
uv run python -m examples.papers.dichroic_designer_si3n4_thickness
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

Note on fidelity: quantitative numbers (cutoff wavelength, dB-level losses
and extinction ratios) are mesh- and discretization-sensitive. The
segmented-waveguide effective index of the Magden filter needs <= 20 nm mesh
resolution to converge, and the dB-scale extinction floors of the FAQUAD
combiner keep improving with finer meshes, more EME cells and more modes than
the example defaults.
