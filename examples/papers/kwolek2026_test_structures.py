"""Resonator test structures for the Kwolek 2026 TFLN combiner platform.

Companion to :mod:`examples.papers.kwolek2026_faquad`. To measure the *excess*
properties of the fabricated devices -- propagation loss, group index and the
intrinsic quality factor of the rib waveguide -- the paper accompanies the
functional combiners with passive **micro-ring resonator** test structures on
the same rib layer. This module emits those as gdsfactory cells and writes their
GDS so they can be dropped onto a mask:

- all-pass (single-bus) rings swept over **radius** (separates bend loss from
  straight-waveguide loss via the radius dependence of the loaded Q), and over
  the bus-ring **gap** (brackets critical coupling so the intrinsic Q -- hence
  propagation loss -- can be extracted from the on/under/over-coupled lineshape);
- an add-drop (two-bus) ring for a clean, directly-fit loaded Q.

The waveguide is the same nominal rib as the combiner (``W_TOP`` top width on
``LAYER_RIB``); the angled sidewalls are added at extrusion just like the
combiner, so these layouts are drop-in on the same process. Run
``python -m examples.papers.kwolek2026_test_structures`` to write the GDS and a
preview figure into ``examples/papers/figures/``.
"""

from __future__ import annotations

from pathlib import Path

import gdsfactory as gf

from examples.papers.kwolek2026_faquad import (
    LAYER_RIB,
    W_TOP,
    faquad_combiner,
)

gf.gpdk.PDK.activate()

FIGDIR = Path(__file__).parent / "figures"
GDSDIR = FIGDIR / "gds"

# default test sweeps (radius separates bend/straight loss; gap brackets
# critical coupling so the intrinsic Q / propagation loss can be extracted)
RADII_UM = (30.0, 60.0, 120.0)
GAPS_UM = (0.4, 0.5, 0.6, 0.7)

# representative (layout-only) FAQUAD calibration for the DUT coupler block; the
# resonator test structures are drawn for masks, not simulated, so the exact
# coupling numbers are immaterial here (use the EME examples for performance).
_FAQUAD_PARAMS = (0.05, 0.4, 0.2)


def rib_cross_section(width: float = W_TOP) -> gf.CrossSection:
    """The combiner's rib waveguide as a gdsfactory cross-section."""
    return gf.cross_section.cross_section(width=width, layer=LAYER_RIB)


def faquad_coupler(l_m: float = 120.0) -> gf.Component:
    """The FAQUAD combiner as a drop-in coupler block (layout only).

    A shortened constant-gap section keeps the resonator test layout compact;
    use :mod:`examples.papers.kwolek2026_faquad` / ``kwolek_designer`` for the
    full paper-length, simulated device.
    """
    return faquad_combiner(*_FAQUAD_PARAMS, l_m=l_m)


@gf.cell
def all_pass_ring(
    radius: float = 60.0, gap: float = 0.5, width: float = W_TOP
) -> gf.Component:
    """All-pass (single-bus) micro-ring resonator on the rib layer."""
    return gf.components.ring_single(
        radius=radius, gap=gap, cross_section=rib_cross_section(width)
    )


@gf.cell
def add_drop_ring(
    radius: float = 60.0, gap: float = 0.5, width: float = W_TOP
) -> gf.Component:
    """Add-drop (two-bus) micro-ring resonator for a clean loaded-Q fit."""
    return gf.components.ring_double(
        radius=radius, gap=gap, cross_section=rib_cross_section(width)
    )


@gf.cell
def dut_resonator(
    l_m: float = 120.0,
    radius: float = 40.0,
    bus_ext: float = 30.0,
    width: float = W_TOP,
) -> gf.Component:
    """Racetrack resonator whose coupler *is* the FAQUAD device (paper Fig. 3a).

    The FH characterization (paper Fig. 3) integrates the device-under-test
    (DUT) FAQUAD coupler into a racetrack resonator and compares its loaded Q
    with a control resonator. Here the cross-port waveguide of the FAQUAD
    combiner is closed into a ring with two 180-degree Euler bends and a return
    straight, while the bar-port waveguide is the coupling bus (with input/output
    access straights ``bus_in`` / ``bus_out``).
    """
    comb = faquad_coupler(l_m=l_m)
    c = gf.Component()
    ref = c << comb
    xs = rib_cross_section(width)
    length_x = float(ref.ports["out_cross"].center[0] - ref.ports["in_cross"].center[0])
    b1 = c << gf.components.bend_euler(angle=180, radius=radius, cross_section=xs)
    b1.connect("o1", ref.ports["out_cross"])
    top = c << gf.components.straight(length=length_x, cross_section=xs)
    top.connect("o1", b1.ports["o2"])
    b2 = c << gf.components.bend_euler(angle=180, radius=radius, cross_section=xs)
    b2.connect("o1", top.ports["o2"])
    gf.routing.route_single(
        c, b2.ports["o2"], ref.ports["in_cross"], cross_section=xs, radius=radius
    )
    bus_in = c << gf.components.straight(length=bus_ext, cross_section=xs)
    bus_in.connect("o2", ref.ports["in_bar"])
    bus_out = c << gf.components.straight(length=bus_ext, cross_section=xs)
    bus_out.connect("o1", ref.ports["out_bar"])
    c.add_port("bus_in", port=bus_in.ports["o1"])
    c.add_port("bus_out", port=bus_out.ports["o2"])
    return c


@gf.cell
def control_resonator(
    l_m: float = 120.0,
    radius: float = 40.0,
    gap: float = 0.8,
    width: float = W_TOP,
) -> gf.Component:
    """Plain racetrack control resonator matching the DUT footprint (Fig. 3a).

    Same loop dimensions as :func:`dut_resonator` but with a straight, uniform
    bus coupler instead of the FAQUAD device, so the difference in loaded Q
    isolates the DUT's excess loss.
    """
    length_x = faquad_coupler(l_m=l_m).xsize
    c = gf.components.ring_single(
        radius=radius,
        gap=gap,
        length_x=length_x,
        length_y=0.0,
        cross_section=rib_cross_section(width),
        bend="bend_euler",
    )
    return c


@gf.cell
def fh_measurement_layout(
    l_m: float = 120.0, radius: float = 40.0, pitch: float = 60.0
) -> gf.Component:
    """The FH measurement configuration: stacked DUT and control resonators.

    Reproduces the layout of paper Fig. 3a/3b -- the resonator containing the
    FAQUAD DUT above a control resonator of the same footprint, each fed by a
    bus waveguide.
    """
    c = gf.Component()
    dut = c << dut_resonator(l_m=l_m, radius=radius)
    ctrl = c << control_resonator(l_m=l_m, radius=radius)
    dut.dymin = ctrl.dymax + pitch
    dut.dxmin = ctrl.dxmin
    return c


@gf.cell
def resonator_test_array(
    radii: tuple[float, ...] = RADII_UM,
    gaps: tuple[float, ...] = GAPS_UM,
    pitch: float = 40.0,
    width: float = W_TOP,
) -> gf.Component:
    """A laid-out array of the ring test structures (radius and gap sweeps).

    Top block: all-pass rings of fixed gap swept over ``radii`` (bend/straight
    loss separation). Bottom block: all-pass rings of the middle radius swept
    over ``gaps`` (critical-coupling bracket). Plus one add-drop ring.
    """
    c = gf.Component()
    y = 0.0
    mid_radius = radii[len(radii) // 2]
    for r in radii:
        ref = c << all_pass_ring(radius=r, gap=gaps[len(gaps) // 2], width=width)
        ref.dymin = y
        ref.dxmin = 0.0
        y = ref.dymax + pitch
    y += pitch
    for g in gaps:
        ref = c << all_pass_ring(radius=mid_radius, gap=g, width=width)
        ref.dymin = y
        ref.dxmin = 0.0
        y = ref.dymax + pitch
    ref = c << add_drop_ring(radius=mid_radius, gap=gaps[len(gaps) // 2], width=width)
    ref.dymin = y + pitch
    ref.dxmin = 0.0
    return c


def write_gds(gdsdir: Path = GDSDIR) -> list[Path]:
    """Write each test structure and the combined array to GDS; return paths."""
    gdsdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for r in RADII_UM:
        comp = all_pass_ring(radius=r, gap=GAPS_UM[len(GAPS_UM) // 2])
        path = gdsdir / f"kwolek2026_allpass_r{int(r)}um.gds"
        comp.write_gds(path)
        written.append(path)
    for g in GAPS_UM:
        comp = all_pass_ring(radius=RADII_UM[len(RADII_UM) // 2], gap=g)
        path = gdsdir / f"kwolek2026_allpass_gap{int(g * 1000)}nm.gds"
        comp.write_gds(path)
        written.append(path)
    addrop = add_drop_ring()
    path = gdsdir / "kwolek2026_adddrop_ring.gds"
    addrop.write_gds(path)
    written.append(path)
    array = resonator_test_array()
    path = gdsdir / "kwolek2026_resonator_test_array.gds"
    array.write_gds(path)
    written.append(path)
    # Fig. 3 FH-characterization layout: DUT + control racetrack resonators
    for name, comp in [
        ("kwolek2026_dut_resonator", dut_resonator()),
        ("kwolek2026_control_resonator", control_resonator()),
        ("kwolek2026_fh_measurement_layout", fh_measurement_layout()),
    ]:
        path = gdsdir / f"{name}.gds"
        comp.write_gds(path)
        written.append(path)
    return written


def main() -> dict[str, object]:
    """Write the GDS and a labelled preview figure of the test structures."""
    import matplotlib.pyplot as plt

    from examples.papers._plot import plot_component

    FIGDIR.mkdir(parents=True, exist_ok=True)
    written = write_gds()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    plot_component(all_pass_ring(radius=60.0, gap=0.5), axes[0, 0])
    axes[0, 0].set_title("all-pass ring (R = 60 um, gap = 500 nm)")
    plot_component(resonator_test_array(), axes[0, 1])
    axes[0, 1].set_title("resonator test array (radius + gap sweeps)")
    plot_component(dut_resonator(), axes[1, 0])
    axes[1, 0].set_title("DUT racetrack (FAQUAD coupler in loop) -- Fig. 3a")
    plot_component(fh_measurement_layout(), axes[1, 1])
    axes[1, 1].set_title("FH measurement layout: DUT over control -- Fig. 3a/b")
    for ax in axes.flat:
        ax.set_aspect("equal")
    fig.suptitle("Kwolek 2026: TFLN rib resonator test structures (loss / Q)")
    fig.tight_layout()
    out = FIGDIR / "kwolek2026_test_structures.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return {"gds": [str(p) for p in written], "figure": str(out)}


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
