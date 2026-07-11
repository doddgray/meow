# Bent-waveguide EME: ring–bus couplers

MEOW's FDE mode solver can solve **bent** waveguide modes (the
`Mesh2D.bend_radius` field applies the conformal transform that maps a curved
guide to an equivalent straight graded-index guide). Combined with the EME
S-matrix engine and coupled-mode theory, this lets you design the two workhorse
ring-resonator bus couplers directly from the geometry:

- a **point coupler** — a straight bus tangent to the ring, and
- a **pulley (wrap-around) coupler** — a bus bent to follow the ring over an
  angular length.

Both are implemented as platform-agnostic, literature-validated examples under
[`examples/papers`](../examples/papers), sharing one engine,
[`_bent_coupler.py`](../examples/papers/_bent_coupler.py).

## Why the bend matters

A microring guides a **whispering-gallery** mode: the field is pushed radially
outward and its effective index falls below the straight value as the radius
shrinks (curvature dispersion). Coupling to a bus is the evanescent overlap of
this bent mode with the bus mode, so getting the coupling — and, for a pulley,
the phase-matching — right *requires* solving the bent mode, not a straight
approximation.

## The physics, in one page

Both couplers reduce to a two-mode (ring/bus) coupled system. From **one** bent
supermode solve of the ring+bus cross-section, `supermode_coupling` extracts

- the coupling `kappa0` [1/µm] and phase mismatch `delta` [1/µm], via the
  2-level identity `S = k0 (n_even − n_odd) = 2·sqrt(kappa0² + delta²)` (the
  supermode split) together with the supermodes' **rail localization** (the
  mixing angle). At phase matching the supermodes are 50/50 delocalized
  (`delta → 0`); a mismatched coupler has rail-localized supermodes.

**Point coupler.** The edge-to-edge gap opens quadratically about the tangent
point, `g(z) = g0 + z²/(2R)`, so the interaction is localized. The evanescent
coupling is calibrated as `kappa0(g) = A·exp(−g/gamma)` (a few supermode solves)
and integrated over the parabolic gap:

```
kappa_field = ∫ kappa0(g(z)) dz = A·e^(−g0/gamma)·sqrt(2π R gamma)   (phase-matched)
|kappa|² = sin²(kappa_field)
```

Point coupling is weak and set almost entirely by the closest-approach gap; it
grows with radius (√R) but saturates well below unity — which is exactly why
strong or broadband coupling wants a pulley.

**Pulley coupler.** The bus wraps the ring over an arc `Lc = R_bus·θ`; bus and
ring are concentric, so both are solved bent and their supermodes carry the
coupling. Over the uniform wrap the transfer is the phase-mismatched
directional-coupler law (Moille 2019):

```
|kappa|²(Lc) = kappa0²/(kappa0² + delta²) · sin²(sqrt(kappa0² + delta²)·Lc)
```

whose small-`Lc` limit is `(kappa0·Lc)²·sinc²(...)`. The extra wrap-length knob
decouples *how much* coupling from *how small* a gap: at phase matching
(`delta → 0`, set by narrowing the bus width) any coupling is reachable by
length; a mismatched pulley oscillates and is capped at
`kappa0²/(kappa0² + delta²)`.

Downstream, both feed the standard ring observables — all-pass / add-drop
transfer functions, loaded/coupled/intrinsic Q, and the bus extraction
efficiency `η = 1/(1 + Q_c/Q_i)`.

## Examples

| Script | What it does |
|---|---|
| [`bogaerts2012_point_coupler.py`](../examples/papers/bogaerts2012_point_coupler.py) | 220 nm SOI point coupler — reproduces Bogaerts 2012 gap↔coupling and Xu 2008's R = 1.5 µm critical coupling (Q_i ≈ 2×10⁴, Q_L ≈ 9×10³, κ² ≈ 0.8%). |
| [`moille2019_pulley.py`](../examples/papers/moille2019_pulley.py) | 780 nm Si₃N₄ pulley — reproduces Moille 2019's `sinc²`/phase-matching law, the phase-matching bus width, the bent-EME bus→ring beat, and the broadband Q_c / extraction spectrum (Q_i = 3×10⁶). |
| [`point_coupler_designer.py`](../examples/papers/point_coupler_designer.py) | Designs a point coupler on an **arbitrary** platform for a target κ² at a target wavelength (inverts for the gap). |
| [`pulley_coupler_designer.py`](../examples/papers/pulley_coupler_designer.py) | Designs a pulley on an arbitrary platform: phase-matches the bus width, then solves the wrap length/angle for a target κ². |
| [`pulley_coupler_slurm.py`](../examples/papers/pulley_coupler_slurm.py) | Slurm-distributed broadband coupling spectrum (one bent solve per wavelength). |

Every script writes the full **standard plot suite**: cross-section index map,
mode fields (bent ring vs straight bus, or even/odd supermodes), curvature
dispersion `n_eff(R)`, coupling calibration `kappa0(g)`, the `|kappa|²` design
curve vs gap/length (with the target and solution marked), the coupling / Q /
extraction spectrum, the bent-EME propagation field, the ring transfer
functions, and a layout schematic. PNGs land in
`examples/papers/figures/<script>/`.

## Running

```sh
# fast, coarse smoke resolution
MEOW_EXAMPLE_RES=low python -m examples.papers.bogaerts2012_point_coupler
MEOW_EXAMPLE_RES=low python -m examples.papers.moille2019_pulley
MEOW_EXAMPLE_RES=low python -m examples.papers.point_coupler_designer
MEOW_EXAMPLE_RES=low python -m examples.papers.pulley_coupler_designer

# accurate (slower)
MEOW_EXAMPLE_RES=high python -m examples.papers.moille2019_pulley
```

`MEOW_EXAMPLE_RES` (`low`/`medium`/`high`) trades speed for mesh/cell/sweep
resolution. Designing on your own platform is a one-liner — build a
`StripPlatform` (or use `soi_platform` / `sin_platform`) and call
`design_point_coupler` / `design_pulley_coupler`.

Tests: [`src/tests/test_bent_couplers.py`](../src/tests/test_bent_couplers.py)
(`pytest src/tests/test_bent_couplers.py -m "not slow"`).
