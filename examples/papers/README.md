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

- `magden2018_dichroic.py`: the parametric `dichroic_filter()` PCell, the
  extrusion/meshing helpers and the coupled-mode quantities
  (delta, kappa, |T_A|^2).
- `magden2018_figures.py`: reproduces Fig. 1 (modes + effective indices +
  supermode anticrossing), Fig. 2 (delta/kappa dispersion, transmission
  roll-off, extinction ratio), Fig. 3 (layout + EME optimization of the four
  adiabatic section lengths) and the model counterpart of Fig. 4 (full-device
  EME spectra + cutoff-vs-width shift).

Validation anchors: the phase-matching cutoff lands in the C-band at
converged mesh resolution (<= 20 nm), matching the paper's ~1540 nm design;
the EME length sweeps converge to low-loss transmission at the paper's chosen
section lengths; the full-device spectra cross over from the short-pass to
the long-pass port around the cutoff.

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

Validation anchors: chi(0) = pi/2 and the closed-form eta of Eq. 12; the
FH input transfers adiabatically to the cross port while the SH stays in the
bar port with high extinction; loss/extinction trends across the FH and SH
bands follow the paper's Fig. 2.

## Running

```sh
uv run python -m examples.papers.magden2018_figures
uv run python -m examples.papers.kwolek2026_figures
```

Figures are written to `examples/papers/figures/`. The default settings take
tens of minutes; set `MEOW_EXAMPLE_FAST=1` for a coarse smoke-test version
(used by `src/tests/test_paper_examples.py`).

Note on fidelity: quantitative numbers (cutoff wavelength, dB-level losses
and extinction ratios) are mesh- and discretization-sensitive. The
segmented-waveguide effective index of the Magden filter needs <= 20 nm mesh
resolution to converge, and the dB-scale extinction floors of the FAQUAD
combiner keep improving with finer meshes, more EME cells and more modes than
the example defaults.
