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
- The full filter consists of four adiabatic sections (paper Fig. 3a),
  with WGB running straight along the axis and WGA tipping in below it:
  (1) a single input strip grows into the straight 3-segment WGB by tapering
      the central and outer ridges simultaneously at constant total width and
      constant inter-ridge gaps,
  (2) WGA tapers up from a tip at a constant edge-to-edge coupling gap,
      through the phase-matching condition,
  (3) WGA bends away from WGB, separating to a 2 um gap, and
  (4) the 3-segment WGB merges back into a single output strip (the reverse of
      section 1).
  WGA is the short-pass output, WGB the long-pass output. C-band design
  lengths: L1 = L4 = 200 um, L2 = 260 um, L3 = 900 um.

Where the paper leaves layout details unspecified (the section-1/4
morphology, the single-ridge port extension length), this example documents
its assumptions inline (matching the reference layout in the paper's
Fig. 3a). All taper tips use a finite minimum fabricable width ``W_TIP``
(default 50 nm), and the central WGB ridge extends ``L_EXT`` past the
outer-ridge tips so the WGB EME ports are single-ridge waveguides.
"""

from __future__ import annotations

from collections.abc import Callable

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
W_B = 0.25
G_B = 0.10
GAP = 0.75
W_TIP = 0.05
"""Minimum fabricable ridge/taper-tip width (all taper tips use this)."""
L_EXT = 10.0
"""Length the central WGB ridge extends beyond the outer-ridge taper tips, so
the EME ports at the WGB edges are single-ridge waveguides."""
# NOTE on end-to-end EME: our discretized FDE model yields a coupling of
# kappa ~ 0.003/um at the 750 nm gap, for which the paper's 260 um
# phase-matching taper is strongly diabatic (Landau-Zener jump
# probability ~93%), so a full-device EME at example-budget resolution
# does not reproduce the short/long-pass separation end to end. The
# paper's EME usage - per-section transmission convergence sweeps - is
# reproduced in the Fig. 3 script, and the filter response is obtained
# from the coupled-mode model (paper Eq. 3) with FDE-computed
# delta(lambda) and kappa(lambda), as in the paper's Fig. 2b.
GAP_OUT = 2.0
L1, L2, L3, L4 = 200.0, 260.0, 900.0, 200.0
KAPPA_DESIGN = 5.0e-3
"""Design coupling |kappa| at the 750 nm gap [1/um] (~5/mm; paper Fig. 2a).

The FDE coupling *overlap integral* gives the wavelength dependence (shape) of
|kappa|, but its absolute scale in meow's normalized field units is
convention-dependent (and the high-index-contrast overlap is only approximate),
so the absolute coupling is anchored to this design value at the cutoff."""


def w_b_total(w_b: float = W_B, g_b: float = G_B) -> float:
    """Total width of the 3-segment WGB."""
    return 3 * w_b + 2 * g_b


def lateral_positions(
    w_a: float = W_A,
    w_b: float = W_B,
    g_b: float = G_B,
    gap: float = GAP,
    gap_out: float = GAP_OUT,
) -> tuple[list[float], float, float]:
    """Lateral (meow-x) centres of the device waveguides (paper Fig. 3a).

    The three WGB segments are centred on the lateral axis (x=0); WGA sits
    *below* them (negative x), at the coupling gap in section 2 and bending to
    the final gap in section 3.

    Returns:
        ``(seg_centers, y_a_couple, y_a_final)``: the WGB segment centres and
        the WGA centre in the coupling region and after separation.
    """
    pitch_b = w_b + g_b
    seg_centers = [(i - 1) * pitch_b for i in range(3)]
    wgb_left_edge = min(seg_centers) - w_b / 2
    y_a_couple = wgb_left_edge - gap - w_a / 2
    y_a_final = wgb_left_edge - gap_out - w_a / 2
    return seg_centers, y_a_couple, y_a_final


@gf.cell
def dichroic_filter(
    w_a: float = W_A,
    w_b: float = W_B,
    g_b: float = G_B,
    gap: float = GAP,
    gap_out: float = GAP_OUT,
    w_tip: float = W_TIP,
    l_ext: float = L_EXT,
    l1: float = L1,
    l2: float = L2,
    l3: float = L3,
    l4: float = L4,
    points_per_section: int = 30,
) -> gf.Component:
    """Parametric 1x2 dichroic filter (paper Fig. 3a).

    Drawn with propagation along the gds x-axis (meow ``z``) and the lateral
    cross-section along the gds y-axis (meow ``x``); the 220 nm Si height is
    added at extrusion time. The topology reproduces paper Fig. 3a:

    - WGB (long-pass, segmented) runs **straight** on the lateral axis with a
      **constant total width** (``3 w_b + 2 g_b``) and constant inter-ridge
      gaps everywhere. Near each end the central and outer ridges taper
      *simultaneously* - the outer ridges shrink to a ``w_tip`` tip while the
      central ridge widens to take up the freed width - so the three-ridge WGB
      smoothly becomes a single ridge. The central ridge then extends a further
      ``l_ext`` past the outer-ridge tips, so the WGB ports (z=0, z=z4) are
      single-ridge waveguides.
    - WGA (short-pass, solid strip) is **absent until section 2**, where it
      tapers up from a ``w_tip`` tip at a **constant edge-to-edge gap** to WGB
      (its centre shifts as it widens); in section 3 it bends away from WGB
      (linear lateral shift to the final gap) and then runs straight to its
      output.

    All taper tips use the minimum fabricable width ``w_tip``.
    """
    c = gf.Component()
    z1, z2, z3, z4 = l1, l1 + l2, l1 + l2 + l3, l1 + l2 + l3 + l4
    _, y_a_couple, y_a_final = lateral_positions(w_a, w_b, g_b, gap, gap_out)
    total = w_b_total(w_b, g_b)  # constant WGB total width
    wgb_left_edge = -total / 2  # constant WGA-facing edge of WGB
    z_ol, z_or = l_ext, z4 - l_ext  # outer-ridge tip locations

    def strip(zs: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
        upper = np.stack([zs, center + half], axis=1)
        lower = np.stack([zs, center - half], axis=1)[::-1]
        return np.concatenate([upper, lower])

    def zsamp(intervals: list[tuple[float, float]]) -> np.ndarray:
        pts: list[float] = []
        for a, b in intervals:
            pts.extend(np.linspace(a, b, points_per_section, endpoint=False))
        pts.append(intervals[-1][1])
        return np.asarray(pts, dtype=float)

    def outer_t(z: float) -> float:
        """Outer-ridge taper fraction: 0 at the tips, 1 in the coupling region."""
        if z <= z_ol:
            return 0.0
        if z < z1:
            return (z - z_ol) / (z1 - z_ol)
        if z <= z3:
            return 1.0
        if z < z_or:
            return (z_or - z) / (z_or - z3)
        return 0.0

    def outer_w(z: float) -> float:
        return w_tip + (w_b - w_tip) * outer_t(z)

    def central_w(z: float) -> float:
        # constant total width: central takes up whatever the outer ridges
        # (w_tip where they are absent) leave; = w_b in the coupling region.
        ow = outer_w(z) if z_ol <= z <= z_or else w_tip
        return total - 2 * g_b - 2 * ow

    # WGB central ridge (extends the full device length, single-ridge at ports)
    zc = zsamp([(0, z_ol), (z_ol, z1), (z1, z3), (z3, z_or), (z_or, z4)])
    c.add_polygon(
        strip(zc, np.zeros_like(zc), np.array([0.5 * central_w(z) for z in zc])),
        layer=LAYER_WG,
    )
    # WGB outer ridges (only between the tips), at constant total width so their
    # outer edge stays at +-total/2 and the gap to the central ridge stays g_b
    zo = zsamp([(z_ol, z1), (z1, z3), (z3, z_or)])
    ow_half = np.array([0.5 * outer_w(z) for z in zo])
    oc = np.array([total / 2 - 0.5 * outer_w(z) for z in zo])  # outer-ridge centre
    for sign in (-1.0, 1.0):
        c.add_polygon(strip(zo, sign * oc, ow_half), layer=LAYER_WG)

    # WGA: present from z1 (tip in section 2 at constant gap, bend in section 3)
    za = zsamp([(z1, z2), (z2, z3), (z3, z4)])

    def wga_w(z: float) -> float:
        if z <= z2:  # section 2: taper up from a w_tip tip
            return w_tip + (w_a - w_tip) * (z - z1) / l2
        return w_a

    def wga_center(z: float) -> float:
        if z <= z2:  # constant gap: WGA near (upper) edge fixed at wgb_left_edge-gap
            return (wgb_left_edge - gap) - 0.5 * wga_w(z)
        if z <= z3:  # section 3: straight linear bend away from WGB
            return y_a_couple + (y_a_final - y_a_couple) * (z - z2) / l3
        return y_a_final

    a_center = np.array([wga_center(z) for z in za])
    a_half = np.array([0.5 * wga_w(z) for z in za])
    c.add_polygon(strip(za, a_center, a_half), layer=LAYER_WG)

    def _snap(w: float) -> float:
        return 0.002 * round(float(w) / 0.002)  # gdsfactory 2 nm port grid

    w_port = _snap(central_w(0.0))  # single-ridge WGB port width
    c.add_port("in0", center=(0.0, 0.0), width=w_port, orientation=180, layer=LAYER_WG)
    c.add_port(
        "long_pass", center=(z4, 0.0), width=w_port, orientation=0, layer=LAYER_WG
    )
    c.add_port(
        "short_pass",
        center=(z4, y_a_final),
        width=_snap(w_a),
        orientation=0,
        layer=LAYER_WG,
    )
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
    return structs + oxide_structures(z_max, x_span=(-3.6, 1.2))


def mesh2d(
    x_min: float = -1.6,
    x_max: float = 4.2,
    y_min: float = -0.9,
    y_max: float = 1.1,
    res: float = 0.025,
) -> mw.Mesh2D:
    """Cross-section mesh for the isolated/coupled mode analyses (Fig. 1/2).

    x: lateral (chip plane), y: vertical. The default span fits the coupled
    cross-section used for the coupling analysis (WGA at x=0, WGB to +x).
    """
    return mw.Mesh2D(
        x=np.arange(x_min, x_max + res / 2, res),
        y=np.arange(y_min, y_max + res / 2, res),
    )


def device_mesh(
    x_min: float = -3.3,
    x_max: float = 0.9,
    y_min: float = -0.9,
    y_max: float = 1.1,
    res: float = 0.025,
) -> mw.Mesh2D:
    """Cross-section mesh spanning the full device width (paper Fig. 3a/4).

    Wide enough laterally (meow x) to contain WGB on x=0 and WGA all the way
    down to its separated position ``y_a_final`` in section 3.
    """
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
    num_modes: int = 8,
    compute_modes: Callable | None = None,
) -> float:
    """Effective index of the fundamental TE mode of a cross-section."""
    modes = solve_modes(
        structures, wl, mesh=mesh, num_modes=num_modes, compute_modes=compute_modes
    )
    te = [m for m in modes if m.te_fraction > 0.5]
    return float(np.real((te or modes)[0].neff))


def solve_modes(
    structures: list[mw.Structure3D],
    wl: float,
    mesh: mw.Mesh2D | None = None,
    num_modes: int = 8,
    compute_modes: Callable | None = None,
) -> list[mw.Mode]:
    """Solve the modes of a single cross-section at z=0.5.

    Args:
        structures: the 3D structures defining the cross-section.
        wl: vacuum wavelength [um].
        mesh: the 2D mesh (default: :func:`mesh2d`).
        num_modes: number of modes to compute.
        compute_modes: FDE backend (default: ``meow.compute_modes``, tidy3d).
    """
    mesh = mesh or mesh2d()
    compute_modes = compute_modes or mw.compute_modes
    cell = mw.Cell(structures=structures, mesh=mesh, z_min=0.0, z_max=1.0)
    env = mw.Environment(wl=wl, T=25.0)
    cs = mw.CrossSection.from_cell(cell=cell, env=env)
    return compute_modes(cs, num_modes=num_modes)


def coupled_supermode_neffs(
    wl: float,
    w_a: float = W_A,
    gap: float = GAP,
    mesh: mw.Mesh2D | None = None,
    compute_modes: Callable | None = None,
) -> tuple[float, float]:
    """The (n_+, n_-) TE supermode effective indices of the coupled section.

    The pair is selected so that ``n_+`` is the higher-index supermode: at
    weak coupling these are the quasi-even/quasi-odd combinations of the WGA
    and WGB modes whose splitting gives the coupling ``kappa``.
    """
    modes = solve_modes(
        coupled_structures(w_a=w_a, gap=gap), wl, mesh=mesh, compute_modes=compute_modes
    )
    te = [m for m in modes if m.te_fraction > 0.5][:2]
    n_p, n_m = sorted((float(np.real(m.neff)) for m in te[:2]), reverse=True)
    return n_p, n_m


def _fundamental_te_mode(
    structures: list[mw.Structure3D],
    wl: float,
    mesh: mw.Mesh2D | None,
    compute_modes: Callable | None,
) -> mw.Mode:
    """The fundamental TE mode of a cross-section (full field data)."""
    modes = solve_modes(
        structures, wl, mesh=mesh, num_modes=4, compute_modes=compute_modes
    )
    te = [m for m in modes if m.te_fraction > 0.5]
    return (te or list(modes))[0]


def delta_kappa(
    wl: float,
    w_a: float = W_A,
    gap: float = GAP,
    mesh: mw.Mesh2D | None = None,
    compute_modes: Callable | None = None,
) -> tuple[float, float]:
    """Half phase mismatch ``delta`` [1/um] and the coupling overlap [a.u.].

    ``delta = (beta_A - beta_B) / 2`` from the isolated WGA and WGB modes, and
    the coupling overlap is the coupled-mode integral

    ``kappa ~ sqrt(kappa_ab kappa_ba)``,
    ``kappa_pq = k0 int (n^2 - n_q^2) E_p . E_q dA / sqrt(P_p P_q)``

    evaluated with the isolated WGA (at x=0) and WGB (at its coupling-region
    position) modes, where ``P`` is the modal power. Unlike a supermode-
    splitting extraction, this overlap is **monotonic** and well-conditioned
    (it never requires subtracting two near-equal effective indices, which for
    the weakly coupled, segmented WGB produced a spurious peak at the cutoff).

    The returned overlap is in arbitrary units (meow's field normalization is
    convention-dependent); :func:`delta_kappa_spectrum` rescales it to the
    physical design coupling :data:`KAPPA_DESIGN`.
    """
    mesh = mesh or mesh2d()
    k0 = 2 * np.pi / wl
    x0_b = w_a / 2 + gap + w_b_total() / 2
    mode_a = _fundamental_te_mode(wga_structures(w_a), wl, mesh, compute_modes)
    mode_b = _fundamental_te_mode(wgb_structures(x0=x0_b), wl, mesh, compute_modes)

    delta = 0.5 * k0 * float(np.real(mode_a.neff) - np.real(mode_b.neff))

    n_ox2 = float(np.real(mode_a.cs.nx[0, 0]) ** 2)  # cladding (corner) index^2
    pert_a = np.clip(np.real(mode_a.cs.nx**2) - n_ox2, 0.0, None)  # WGA silicon
    pert_b = np.clip(np.real(mode_b.cs.nx**2) - n_ox2, 0.0, None)  # WGB silicon
    e_dot_e = np.real(
        mode_a.Ex * mode_b.Ex + mode_a.Ey * mode_b.Ey + mode_a.Ez * mode_b.Ez
    )
    p_a = abs(float(np.real(np.sum(mode_a.Ex * mode_a.Hy - mode_a.Ey * mode_a.Hx))))
    p_b = abs(float(np.real(np.sum(mode_b.Ex * mode_b.Hy - mode_b.Ey * mode_b.Hx))))
    norm = np.sqrt(p_a * p_b) + 1e-30
    kappa_ab = k0 * float(np.sum(pert_b * e_dot_e)) / norm
    kappa_ba = k0 * float(np.sum(pert_a * e_dot_e)) / norm
    overlap = float(np.sqrt(abs(kappa_ab * kappa_ba)))
    return delta, overlap


def delta_kappa_spectrum(
    wls: np.ndarray,
    wl_ref: float | None = None,
    w_a: float = W_A,
    gap: float = GAP,
    mesh: mw.Mesh2D | None = None,
    compute_modes: Callable | None = None,
    kappa_ref: float = KAPPA_DESIGN,
) -> tuple[np.ndarray, np.ndarray]:
    """``delta(lambda)`` [1/um] and calibrated ``|kappa|(lambda)`` [1/um].

    Computes the half phase mismatch and the (monotonic) coupling overlap at
    every wavelength with :func:`delta_kappa`, then rescales the overlap so
    that ``|kappa| = kappa_ref`` at ``wl_ref`` (the phase-matching/cutoff
    wavelength, located where ``delta = 0`` if ``wl_ref`` is not given).
    """
    wls = np.asarray(wls, dtype=float)
    deltas = np.empty_like(wls)
    overlaps = np.empty_like(wls)
    for i, wl in enumerate(wls):
        deltas[i], overlaps[i] = delta_kappa(
            float(wl), w_a=w_a, gap=gap, mesh=mesh, compute_modes=compute_modes
        )
    if wl_ref is None:
        order = np.argsort(deltas)
        wl_ref = float(np.interp(0.0, deltas[order], wls[order]))
    overlap_ref = float(np.interp(wl_ref, wls, overlaps))
    kappas = kappa_ref * overlaps / max(overlap_ref, 1e-30)
    return deltas, kappas


def analytical_transmission(gamma: np.ndarray) -> np.ndarray:
    """|T_A|^2 of the quasi-even mode in WGA (paper Eq. 3): gamma=delta/|kappa|."""
    return 0.5 * (1 + gamma / np.sqrt(1 + gamma**2))


def device_cells(
    component: gf.Component,
    cells_per_section: tuple[int, int, int, int] = (6, 8, 12, 6),
    lengths: tuple[float, float, float, float] = (L1, L2, L3, L4),
    mesh: mw.Mesh2D | None = None,
    l_ext: float = L_EXT,
) -> list[mw.Cell]:
    """Discretize the four filter sections into EME cells.

    When a section has at least two cells, the first cell of section 1 and the
    last cell of section 4 are made exactly ``l_ext`` long, so they coincide
    with the single-ridge WGB extensions and the EME input/output ports are
    single-ridge cross-sections.
    """
    structs = extrude_filter(component)
    mesh = mesh or device_mesh()
    n1, n2, n3, n4 = cells_per_section
    l1, l2, l3, l4 = lengths
    Ls: list[float] = []
    # section 1: single-ridge port cell (l_ext) + uniform taper, if room
    if n1 >= 2 and 0 < l_ext < l1:
        Ls.append(l_ext)
        Ls.extend([(l1 - l_ext) / (n1 - 1)] * (n1 - 1))
    else:
        Ls.extend([l1 / n1] * n1)
    Ls.extend([l2 / n2] * n2)
    Ls.extend([l3 / n3] * n3)
    # section 4: uniform taper + single-ridge port cell (l_ext), if room
    if n4 >= 2 and 0 < l_ext < l4:
        Ls.extend([(l4 - l_ext) / (n4 - 1)] * (n4 - 1))
        Ls.append(l_ext)
    else:
        Ls.extend([l4 / n4] * n4)
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


def device_port_transmission(
    component: gf.Component,
    wl: float,
    *,
    cells_per_section: tuple[int, int, int, int] = (6, 8, 12, 6),
    mesh: mw.Mesh2D | None = None,
    num_modes: int = 8,
    parallel: bool | None = None,
    compute_modes: Callable | None = None,
) -> tuple[float, float]:
    """Full-device EME (short-pass, long-pass) power for fundamental TE input.

    Builds the EME cells of the whole filter and cascades them either serially
    or with the parallel slice-group engine (``parallel=True`` or the
    ``MEOW_PAPER_PARALLEL`` environment variable). The output modes (needed to
    attribute power to the WGA/WGB ports) are solved once on the final cell.

    Note: the parallel engine always uses the deterministic default (tidy3d)
    backend; ``compute_modes`` selects the backend for the serial path and for
    the output-mode classification.
    """
    from examples.papers._backends import device_s_matrix

    cells = device_cells(component, cells_per_section=cells_per_section, mesh=mesh)
    env = mw.Environment(wl=wl, T=25.0)
    S, pm = device_s_matrix(
        cells,
        env,
        num_modes=num_modes,
        parallel=parallel,
        compute_modes=compute_modes,
    )
    solver = compute_modes or mw.compute_modes
    cs_out = mw.CrossSection.from_cell(cell=cells[-1], env=env)
    modes_out = solver(cs_out, num_modes=num_modes)
    # at the output WGB sits on x=0 and WGA at y_a_final (negative x); split
    # the lateral axis halfway between them to attribute the port powers.
    _, _, y_a_final = lateral_positions()
    x_split = y_a_final / 2
    return port_transmissions(np.asarray(S), pm, modes_out, x_split)
