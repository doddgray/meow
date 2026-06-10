"""Magden et al., "Transmissive silicon photonic dichroic filters with
spectrally selective waveguides", Nat. Commun. 9, 3009 (2018).

This example reproduces the C-band dichroic filter design workflow of the
paper with gdsfactory (parametric layout) and meow (FDE mode solving + EME):

- The spectrally selective cross-section couples a solid silicon waveguide
  WGA (width ``w_a`` = 318 nm) to a sub-wavelength-segmented waveguide WGB
  (3 segments of ``w_b`` = 250 nm separated by ``g_b`` = 100 nm gaps) across
  an edge-to-edge gap of ``gap`` = 750 nm on a 220 nm SOI platform.
- WGA and WGB are phase matched at exactly one wavelength (the filter
  cutoff): below the cutoff the quasi-even supermode lives in WGA, above it
  in WGB, so adiabatic mode evolution separates short- and long-pass light.
- The full filter consists of four adiabatic sections (paper Fig. 3a):
  (1) development of the segmented WGB next to the input waveguide,
  (2) tapering of WGA through the phase-matching condition,
  (3) slow lateral separation of WGA and WGB to a 2 um gap, and
  (4) conversion of the segmented WGB back into a solid output waveguide.
  C-band design lengths: L1 = L4 = 200 um, L2 = 260 um, L3 = 900 um.

Where the paper leaves layout details unspecified (input taper width, the
exact morph in section 4), this example documents its assumptions inline.
"""

from __future__ import annotations

import gdsfactory as gf
import numpy as np

import meow as mw

LAYER_WG = (1, 0)

H_SI = 0.22
"""Silicon layer height (220 nm SOI)."""

T_BOX = 1.0
"""Modeled buried-oxide thickness below the waveguides."""

T_CLAD = 1.0
"""Modeled top-oxide thickness above the waveguides."""

# default C-band design values from the paper
W_A = 0.318
W_A_IN = 0.40  # assumption: input/output solid waveguide width (not specified)
W_B = 0.25
G_B = 0.10
GAP = 0.75
GAP_OUT = 2.0
W_SEG_MIN = 0.10  # minimum width/gap from the fabrication design rules
L1, L2, L3, L4 = 200.0, 260.0, 900.0, 200.0


def w_b_total(w_b: float = W_B, g_b: float = G_B) -> float:
    """Total width of the 3-segment WGB."""
    return 3 * w_b + 2 * g_b


@gf.cell
def dichroic_filter(
    w_a: float = W_A,
    w_a_in: float = W_A_IN,
    w_b: float = W_B,
    g_b: float = G_B,
    gap: float = GAP,
    gap_out: float = GAP_OUT,
    w_seg_min: float = W_SEG_MIN,
    l1: float = L1,
    l2: float = L2,
    l3: float = L3,
    l4: float = L4,
    points_per_section: int = 30,
) -> gf.Component:
    """Parametric 1x2 dichroic filter (paper Fig. 3a).

    The component is drawn with the propagation direction along x (which
    meow maps to its z axis) and the cross-section along y. WGA runs
    straight along y=0; the three WGB segments develop above it, couple,
    and are separated and merged into the solid long-pass output.
    """
    c = gf.Component()
    z1, z2, z3, z4 = l1, l1 + l2, l1 + l2 + l3, l1 + l2 + l3 + l4

    def wga_width(z: float) -> float:
        if z <= z1:  # section 1: input width held
            return w_a_in
        if z <= z2:  # section 2: taper through phase matching
            return w_a_in + (w_a - w_a_in) * (z - z1) / l2
        return w_a  # sections 3 & 4

    def seg_width(z: float) -> float:
        if z <= z1:  # section 1: segments develop from the min feature size
            return w_seg_min + (w_b - w_seg_min) * z / l1
        if z <= z3:
            return w_b
        # section 4: segments widen until they merge into a solid waveguide
        return w_b + (w_b + g_b - w_b) * (z - z3) / l4

    def seg_gap(z: float) -> float:
        if z <= z3:
            return g_b
        return g_b * (1 - (z - z3) / l4)  # close the gaps in section 4

    def wgb_offset(z: float) -> float:
        """Edge-to-edge gap between WGA and WGB."""
        if z <= z2:
            return gap
        if z <= z3:  # section 3: slow separation (sine-smoothed)
            t = (z - z2) / l3
            return gap + (gap_out - gap) * 0.5 * (1 - np.cos(np.pi * t))
        return gap_out

    zs = []
    for z_start, z_stop in [(0, z1), (z1, z2), (z2, z3), (z3, z4)]:
        zs.extend(np.linspace(z_start, z_stop, points_per_section, endpoint=False))
    zs = np.asarray([*zs, z4])

    # WGA: straight strip centered on y=0
    wa = np.asarray([wga_width(z) for z in zs])
    upper = np.stack([zs, wa / 2], axis=1)
    lower = np.stack([zs, -wa / 2], axis=1)[::-1]
    c.add_polygon(np.concatenate([upper, lower]), layer=LAYER_WG)

    # WGB: three segments above WGA
    for i in range(3):
        y_lo, y_hi = [], []
        for z in zs:
            w_seg, g_seg = seg_width(z), seg_gap(z)
            base = wga_width(z) / 2 + wgb_offset(z)
            lo = base + i * (w_seg + g_seg)
            y_lo.append(lo)
            y_hi.append(lo + w_seg)
        upper = np.stack([zs, y_hi], axis=1)
        lower = np.stack([zs, y_lo], axis=1)[::-1]
        c.add_polygon(np.concatenate([upper, lower]), layer=LAYER_WG)

    c.add_port("in0", center=(0, 0), width=w_a_in, orientation=180, layer=LAYER_WG)
    c.add_port("short_pass", center=(z4, 0), width=w_a, orientation=0, layer=LAYER_WG)
    y_lp = wga_width(z4) / 2 + gap_out + w_b_total(w_b, g_b) / 2
    c.add_port("long_pass", center=(z4, y_lp), width=w_b, orientation=0, layer=LAYER_WG)
    return c


def oxide_structures(z_max: float, x_span: tuple[float, float]) -> list[mw.Structure3D]:
    """SiO2 bottom and top cladding boxes for the extruded device."""
    return [
        mw.Structure(
            material=mw.silicon_oxide,
            geometry=mw.Box(
                x_min=x_span[0],
                x_max=x_span[1],
                y_min=-T_BOX,
                y_max=H_SI + T_CLAD,
                z_min=0.0,
                z_max=z_max,
            ),
            mesh_order=10,
        ),
    ]


def extrude_filter(component: gf.Component) -> list[mw.Structure3D]:
    """Extrude the gdsfactory filter layout into meow 3D structures."""
    extrusion_rules = {
        LAYER_WG: [
            mw.GdsExtrusionRule(material=mw.silicon, h_min=0.0, h_max=H_SI),
        ],
    }
    structs = mw.extrude_gds(component, extrusion_rules)
    z_max = float(component.xmax)
    return structs + oxide_structures(z_max, x_span=(-3.0, 6.0))


def mesh2d(
    x_min: float = -1.6,
    x_max: float = 4.2,
    y_min: float = -0.9,
    y_max: float = 1.1,
    res: float = 0.025,
) -> mw.Mesh2D:
    """Cross-section mesh. x: lateral (chip plane), y: vertical."""
    return mw.Mesh2D(
        x=np.arange(x_min, x_max + res / 2, res),
        y=np.arange(y_min, y_max + res / 2, res),
    )


# --- isolated waveguide structures for the Fig. 1e/1f mode analyses ---


def wga_structures(w_a: float = W_A) -> list[mw.Structure3D]:
    """An isolated WGA, centered at x=0."""
    si = mw.Structure(
        material=mw.silicon,
        geometry=mw.Box(
            x_min=-w_a / 2, x_max=w_a / 2, y_min=0, y_max=H_SI, z_min=0, z_max=1
        ),
    )
    return [si, *oxide_structures(1.0, x_span=(-3.0, 3.0))]


def wgb_structures(
    w_b: float = W_B, g_b: float = G_B, x0: float = 0.0
) -> list[mw.Structure3D]:
    """An isolated 3-segment WGB, centered at x=x0."""
    total = w_b_total(w_b, g_b)
    segs = [
        mw.Structure(
            material=mw.silicon,
            geometry=mw.Box(
                x_min=x0 - total / 2 + i * (w_b + g_b),
                x_max=x0 - total / 2 + i * (w_b + g_b) + w_b,
                y_min=0,
                y_max=H_SI,
                z_min=0,
                z_max=1,
            ),
        )
        for i in range(3)
    ]
    return [*segs, *oxide_structures(1.0, x_span=(-3.0, 3.0))]


def coupled_structures(
    w_a: float = W_A, w_b: float = W_B, g_b: float = G_B, gap: float = GAP
) -> list[mw.Structure3D]:
    """The spectrally selective cross-section: WGA + WGB at gap=750 nm."""
    x0_b = w_a / 2 + gap + w_b_total(w_b, g_b) / 2
    si_a = mw.Structure(
        material=mw.silicon,
        geometry=mw.Box(
            x_min=-w_a / 2, x_max=w_a / 2, y_min=0, y_max=H_SI, z_min=0, z_max=1
        ),
    )
    return [si_a, *wgb_structures(w_b, g_b, x0=x0_b)]


def fundamental_neff(
    structures: list[mw.Structure3D],
    wl: float,
    mesh: mw.Mesh2D | None = None,
    num_modes: int = 2,
) -> float:
    """Effective index of the fundamental TE mode of a cross-section."""
    modes = solve_modes(structures, wl, mesh=mesh, num_modes=num_modes)
    te = [m for m in modes if m.te_fraction > 0.5]
    return float(np.real((te or modes)[0].neff))


def solve_modes(
    structures: list[mw.Structure3D],
    wl: float,
    mesh: mw.Mesh2D | None = None,
    num_modes: int = 4,
) -> list[mw.Mode]:
    """Solve the modes of a single cross-section at z=0.5."""
    mesh = mesh or mesh2d()
    cell = mw.Cell(structures=structures, mesh=mesh, z_min=0.0, z_max=1.0)
    env = mw.Environment(wl=wl, T=25.0)
    cs = mw.CrossSection.from_cell(cell=cell, env=env)
    return mw.compute_modes(cs, num_modes=num_modes)


def delta_kappa(
    wl: float,
    w_a: float = W_A,
    gap: float = GAP,
    mesh: mw.Mesh2D | None = None,
) -> tuple[float, float]:
    """Half phase mismatch delta and coupling |kappa| at one wavelength.

    delta = (beta_A - beta_B) / 2 from the isolated waveguides and
    |kappa| = sqrt(((beta_+ - beta_-)/2)**2 - delta**2) from the supermode
    splitting of the coupled cross-section (coupled mode theory).
    """
    k0 = 2 * np.pi / wl
    n_a = fundamental_neff(wga_structures(w_a), wl, mesh=mesh)
    n_b = fundamental_neff(wgb_structures(), wl, mesh=mesh)
    modes = solve_modes(coupled_structures(w_a=w_a, gap=gap), wl, mesh=mesh)
    te = [m for m in modes if m.te_fraction > 0.5][:2]
    n_p, n_m = (float(np.real(m.neff)) for m in te[:2])
    delta = 0.5 * k0 * (n_a - n_b)
    half_splitting = 0.5 * k0 * (n_p - n_m)
    kappa_sq = max(half_splitting**2 - delta**2, 0.0)
    return delta, float(np.sqrt(kappa_sq))


def analytical_transmission(gamma: np.ndarray) -> np.ndarray:
    """|T_A|^2 of the quasi-even mode in WGA (paper Eq. 3): gamma=delta/|kappa|."""
    return 0.5 * (1 + gamma / np.sqrt(1 + gamma**2))


def device_cells(
    component: gf.Component,
    cells_per_section: tuple[int, int, int, int] = (6, 8, 12, 6),
    lengths: tuple[float, float, float, float] = (L1, L2, L3, L4),
    mesh: mw.Mesh2D | None = None,
) -> list[mw.Cell]:
    """Discretize the four filter sections into EME cells."""
    structs = extrude_filter(component)
    mesh = mesh or mesh2d()
    Ls: list[float] = []
    for n, length in zip(cells_per_section, lengths, strict=True):
        Ls.extend([length / n] * n)
    return mw.create_cells(structs, mesh, np.asarray(Ls), z_min=0.0)


def mode_x_centroid(mode: mw.Mode) -> float:
    """Energy-weighted lateral position of a mode (for port attribution)."""
    density = np.abs(mode.Ex) ** 2 + np.abs(mode.Ey) ** 2 + np.abs(mode.Ez) ** 2
    X = mode.cs.mesh.Xx
    return float(np.sum(X * density) / np.sum(density))


def port_transmissions(
    S: np.ndarray,
    port_map: dict[str, int],
    modes_out: list[mw.Mode],
    x_split: float,
) -> tuple[float, float]:
    """(short-pass, long-pass) power transmission for fundamental TE input.

    Output modes with energy centroid below ``x_split`` belong to WGA
    (short-pass port), the others to WGB (long-pass port).
    """
    S = np.asarray(S)
    t_short = t_long = 0.0
    for i, mode in enumerate(modes_out):
        amp = S[port_map[f"right@{i}"], port_map["left@0"]]
        power = float(np.abs(amp) ** 2)
        if mode_x_centroid(mode) < x_split:
            t_short += power
        else:
            t_long += power
    return t_short, t_long
