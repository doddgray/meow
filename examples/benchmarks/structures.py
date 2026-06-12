"""Benchmark waveguide structures for mode-solver backend comparisons.

Three nominally equivalent test cases solved by all backends:

- ``fiber``: a step-index circular-core fiber (core n=1.45, r=1.0 um in a
  n=1.35 cladding), drawn as a many-sided polygon prism along z.
- ``si3n4``: a rectangular Si3N4 strip (800 nm thick x 1.5 um wide,
  n=1.996) in SiO2 (n=1.444).
- ``tfln``: an x-cut thin-film lithium niobate rib: 600 nm film, 400 nm
  etch depth (200 nm slab), 1.0 um top width, 65 degree sidewalls
  (25 degrees from vertical), dispersive uniaxial dielectric tensor
  (Zelmon 1997 Sellmeier) on a SiO2 under-cladding.

All structures use the same coordinate conventions as the rest of meow:
x lateral, y vertical, z propagation.
"""

from __future__ import annotations

import numpy as np

import meow as mw

# --- materials ---

fiber_core = mw.IndexMaterial(name="bench_fiber_core", n=1.45)
fiber_clad = mw.IndexMaterial(name="bench_fiber_clad", n=1.35)
si3n4 = mw.IndexMaterial(name="bench_si3n4", n=1.996)
sio2 = mw.IndexMaterial(name="bench_sio2", n=1.444)


def ln_xcut() -> mw.SampledAnisotropicMaterial:
    """Dispersive x-cut congruent LiNbO3 tensor (Zelmon 1997 Sellmeier).

    Mode-plane axes: x is the crystal z (extraordinary) axis, so the
    tensor diagonal is (ne^2, no^2, no^2).
    """
    wls = np.linspace(0.4, 2.0, 33)
    wl2 = wls**2
    no2 = 1 + 2.6734 * wl2 / (wl2 - 0.01764) + 1.2290 * wl2 / (wl2 - 0.05914)
    no2 += 12.614 * wl2 / (wl2 - 474.6)
    ne2 = 1 + 2.9804 * wl2 / (wl2 - 0.02047) + 0.5981 * wl2 / (wl2 - 0.0666)
    ne2 += 8.9543 * wl2 / (wl2 - 416.08)
    return mw.SampledAnisotropicMaterial(
        name="bench_ln_xcut", wls=wls, eps=np.stack([ne2, no2, no2], axis=1)
    )


# --- geometry parameters ---

FIBER_RADIUS = 1.0
SI3N4_THICKNESS = 0.8
SI3N4_WIDTH = 1.5
TFLN_FILM = 0.6
TFLN_ETCH = 0.4
TFLN_SLAB = TFLN_FILM - TFLN_ETCH
TFLN_TOP_WIDTH = 1.0
TFLN_SIDEWALL_DEG = 25.0  # 65 deg from the substrate plane


def fiber_structures() -> list[mw.Structure3D]:
    """Step-index circular-core fiber (polygonal core approximation)."""
    theta = np.linspace(0, 2 * np.pi, 65)[:-1]
    circle = np.stack(
        [FIBER_RADIUS * np.cos(theta), FIBER_RADIUS * np.sin(theta)], axis=1
    )
    core = mw.Structure(
        material=fiber_core,
        geometry=mw.Prism(poly=circle, h_min=0.0, h_max=1.0, axis="z"),
        mesh_order=1,
    )
    clad = mw.Structure(
        material=fiber_clad,
        geometry=mw.Box(x_min=-4, x_max=4, y_min=-4, y_max=4, z_min=0.0, z_max=1.0),
        mesh_order=9,
    )
    return [core, clad]


def fiber_mesh(res: float = 0.05) -> mw.Mesh2D:
    return mw.Mesh2D(
        x=np.arange(-2.6, 2.6 + res / 2, res),
        y=np.arange(-2.6, 2.6 + res / 2, res),
    )


def si3n4_structures() -> list[mw.Structure3D]:
    """Rectangular Si3N4 strip: 800 nm thick x 1.5 um wide, SiO2 clad."""
    core = mw.Structure(
        material=si3n4,
        geometry=mw.Box(
            x_min=-SI3N4_WIDTH / 2,
            x_max=SI3N4_WIDTH / 2,
            y_min=0.0,
            y_max=SI3N4_THICKNESS,
            z_min=0.0,
            z_max=1.0,
        ),
        mesh_order=1,
    )
    clad = mw.Structure(
        material=sio2,
        geometry=mw.Box(x_min=-4, x_max=4, y_min=-3, y_max=3.8, z_min=0.0, z_max=1.0),
        mesh_order=9,
    )
    return [core, clad]


def si3n4_mesh(res: float = 0.05) -> mw.Mesh2D:
    return mw.Mesh2D(
        x=np.arange(-2.2, 2.2 + res / 2, res),
        y=np.arange(-1.4, 2.2 + res / 2, res),
    )


def tfln_structures() -> list[mw.Structure3D]:
    """X-cut TFLN rib: 600 nm film, 400 nm etch, 65 deg sidewalls.

    The rib is drawn with its *top* width = 1.0 um: the prism polygon is
    the (wider) rib base at the slab level, narrowing with height through
    the angled-sidewall extrusion.
    """
    ln = ln_xcut()
    run = TFLN_ETCH * np.tan(np.deg2rad(TFLN_SIDEWALL_DEG))
    base_half = TFLN_TOP_WIDTH / 2 + run
    rib = mw.Structure(
        material=ln,
        geometry=mw.Prism(
            poly=np.array(
                [
                    (0.0, -base_half),
                    (1.0, -base_half),
                    (1.0, base_half),
                    (0.0, base_half),
                ]
            ),
            h_min=TFLN_SLAB,
            h_max=TFLN_FILM,
            axis="y",
            sidewall_angle=TFLN_SIDEWALL_DEG,
        ),
        mesh_order=1,
    )
    slab = mw.Structure(
        material=ln,
        geometry=mw.Box(
            x_min=-4, x_max=4, y_min=0.0, y_max=TFLN_SLAB, z_min=0.0, z_max=1.0
        ),
        mesh_order=2,
    )
    box = mw.Structure(
        material=sio2,
        geometry=mw.Box(x_min=-4, x_max=4, y_min=-2, y_max=0.0, z_min=0.0, z_max=1.0),
        mesh_order=9,
    )
    return [rib, slab, box]


def tfln_mesh(res: float = 0.05) -> mw.Mesh2D:
    return mw.Mesh2D(
        x=np.arange(-2.4, 2.4 + res / 2, res),
        y=np.arange(-1.2, 1.6 + res / 2, res),
    )


STRUCTURES = {
    "fiber": (fiber_structures, fiber_mesh),
    "si3n4": (si3n4_structures, si3n4_mesh),
    "tfln": (tfln_structures, tfln_mesh),
}
