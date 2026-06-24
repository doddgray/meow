"""Ramadan, Scarmozzino & Osgood, *Adiabatic Couplers: Design Rules and
Optimization*, J. Lightwave Technol. 16(2), 277 (1998).

This example reproduces the paper's analytic model and its figures, and adds a
meow EME verification of the adiabatic transfer (the paper validates its
analytic design rules with BPM; meow is an EME tool, so the numerical overlay
here is computed by **EME** on an equivalent strip-waveguide adiabatic coupler -
the meow analogue of the paper's BPM run).

The paper analyses a linearly tapered adiabatic coupler made of three regions:

- **Region I** - two asymmetric waveguides converging at a constant angle
  (separation ``d_I -> d_II``);
- **Region II** - two waveguides tapering in width at constant separation
  ``d_II`` (the width difference goes to zero for a 3 dB coupler);
- **Region III** - the waveguides diverge (symmetric for the 3 dB coupler;
  asymmetric, opposite to the input, for the full coupler).

Power conversion to the *unwanted* local-normal (system) mode is the loss
mechanism. The key analytic results, reproduced here exactly, are the
unwanted-to-wanted system-mode power ratios

- Region I (eq 10, with eq 9 its large-asynchronicity asymptote)::

      q_I = X_Io**2 / (4 * eta**2 * (X_IIo**2 + 1)**2)          (10)
      q_I ~ X_Io**2 / (4 * eta**2 * X_IIo**4)                   (9)

- Region II (eq 15)::

      q_II = (m * X_IIo / (2 * kappa_II * L_II))**2 * sin(kappa_II * L_II)**2

  (``m = 1`` for a 3 dB coupler, ``m = 2`` for a full coupler), whose envelope
  drops as ``(m X_IIo / (2 kappa_II L_II))**2``;

and the overall ratio ``q = 2 q_I + q_II`` (eq 18/19). The isolated-waveguide
power error tolerance ``epsilon`` maps to a system-mode tolerance by
``q = (epsilon/4)**2`` (3 dB) or ``q = epsilon`` (full) (eq 21), giving the
**design rules** for the required device length (eqs 22/23) and, after
optimizing the Region-I/Region-II length ratio, the minimum lengths (eqs
28/29)::

      L_3dB,opt ~ (8 / pi) * (L_c / epsilon)
      L_full,opt ~ (3 / pi) * (L_c / sqrt(epsilon))

with ``L_c = pi / (2 kappa_II)`` the conventional directional-coupler coupling
length in Region II.

Here ``X = delta_beta / (2 kappa)`` is the *asynchronicity* (ratio of the
isolated-waveguide propagation-constant difference to their coupling) and
``eta`` the Region-I *adiabaticity* parameter; the asynchronicity grows
exponentially along Region I, so ``X_Io = X_IIo * exp(gamma)`` with ``gamma``
the Region-I adiabaticity (e.g. ``gamma = 5`` in the paper's Fig. 2).

Run with ``python -m examples.papers.ramadan1998``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FIGDIR = Path(__file__).parent / "figures"
LAYER_WG = (1, 0)

# SOI strip-waveguide stack used for the meow EME verification (the meow
# analogue of the paper's BPM run): a 220 nm silicon core in oxide cladding.
H_SI = 0.22
T_CLAD = 0.8
T_BOX = 0.8


# ==========================================================================
# analytic theory (the paper's eqs, transcribed exactly)
# ==========================================================================
def q_region1(x_io: float | np.ndarray, x_iio: float | np.ndarray, eta: float) -> Any:
    """Region-I unwanted/wanted system-mode power ratio, eq (10)."""
    x_io = np.asarray(x_io, dtype=float)
    x_iio = np.asarray(x_iio, dtype=float)
    return x_io**2 / (4 * eta**2 * (x_iio**2 + 1) ** 2)


def q_region1_asymptote(
    x_io: float | np.ndarray, x_iio: float | np.ndarray, eta: float
) -> Any:
    """Large-asynchronicity asymptote of the Region-I ratio, eq (9)."""
    x_io = np.asarray(x_io, dtype=float)
    x_iio = np.asarray(x_iio, dtype=float)
    return x_io**2 / (4 * eta**2 * x_iio**4)


def q_region2(
    length_ii: float | np.ndarray,
    *,
    kappa_ii: float,
    x_iio: float,
    m: int = 1,
) -> Any:
    """Region-II unwanted/wanted system-mode power ratio, eq (15).

    ``m = 1`` for a 3 dB coupler, ``m = 2`` for a full coupler.
    """
    length_ii = np.asarray(length_ii, dtype=float)
    phase = kappa_ii * length_ii
    envelope = (m * x_iio / (2 * phase)) ** 2
    return envelope * np.sin(phase) ** 2


def q_region2_envelope(
    length_ii: float | np.ndarray, *, kappa_ii: float, x_iio: float, m: int = 1
) -> Any:
    """Envelope of the Region-II ratio (eq 15 without the ``sin**2`` factor)."""
    length_ii = np.asarray(length_ii, dtype=float)
    return (m * x_iio / (2 * kappa_ii * length_ii)) ** 2


def coupling_length(kappa_ii: float) -> float:
    """Conventional directional-coupler coupling length ``L_c = pi/(2 kappa)``."""
    return np.pi / (2 * kappa_ii)


def q_from_tolerance(epsilon: float, *, kind: str = "3dB") -> float:
    """System-mode tolerance ``q`` for an isolated-power tolerance ``epsilon``.

    eq (21): ``q = (epsilon/4)**2`` for a 3 dB coupler, ``q = epsilon`` for a
    full coupler. The 3 dB coupler needs a far smaller ``q`` for the same
    isolated-waveguide power balance, which is why it must be longer.
    """
    return (epsilon / 4.0) ** 2 if kind == "3dB" else float(epsilon)


def optimum_alpha(x_io: float, x_iio: float, *, m: int = 1) -> float:
    """Optimum Region-I/Region-II length ratio ``alpha = L_I/L_II``, eq (25)."""
    return (
        np.log(x_io / x_iio) / (2 * m * x_iio * (x_iio**2 + 1))
    ) ** (2.0 / 3.0)


def length_3db_optimum(epsilon: float, kappa_ii: float) -> float:
    """Minimum 3 dB adiabatic coupler length, eq (28): ``(8/pi)(L_c/epsilon)``."""
    return (8.0 / np.pi) * coupling_length(kappa_ii) / epsilon


def length_full_optimum(epsilon: float, kappa_ii: float) -> float:
    """Minimum full adiabatic coupler length, eq (29): ``(3/pi)(L_c/sqrt e)``."""
    return (3.0 / np.pi) * coupling_length(kappa_ii) / np.sqrt(epsilon)


@dataclass
class AdiabaticCouplerSpec:
    """A designed adiabatic coupler (geometry + the analytic figures of merit).

    Attributes:
        kind: ``"3dB"`` or ``"full"``.
        kappa_ii: Region-II coupling coefficient [1/um].
        x_iio: asynchronicity at the Region-II input.
        gamma: Region-I adiabaticity (sets ``X_Io = X_IIo exp(gamma)``).
        epsilon: target isolated-waveguide power error tolerance.
        length_um: optimized total device length [um].
    """

    kind: str
    kappa_ii: float
    x_iio: float
    gamma: float
    epsilon: float
    length_um: float

    @property
    def x_io(self) -> float:
        """Asynchronicity at the Region-I input, ``X_Io = X_IIo exp(gamma)``."""
        return self.x_iio * np.exp(self.gamma)

    @property
    def coupling_length_um(self) -> float:
        """Conventional directional-coupler coupling length ``L_c`` [um]."""
        return coupling_length(self.kappa_ii)


def design_coupler(
    *,
    kind: str,
    kappa_ii: float,
    epsilon: float,
    x_iio: float = 2.0,
    gamma: float = 2.0,
) -> AdiabaticCouplerSpec:
    """Apply the paper's optimized design rules (eqs 25-29) to size a coupler.

    Returns the minimum length for the requested isolated-power tolerance,
    using ``X_IIo ~ 2`` and ``gamma ~ 2`` (the paper's optimization range where
    the optimal ``X_IIo`` is weakly dependent and ``~ 2``).
    """
    length = (
        length_3db_optimum(epsilon, kappa_ii)
        if kind == "3dB"
        else length_full_optimum(epsilon, kappa_ii)
    )
    return AdiabaticCouplerSpec(
        kind=kind,
        kappa_ii=kappa_ii,
        x_iio=x_iio,
        gamma=gamma,
        epsilon=epsilon,
        length_um=float(length),
    )


# ==========================================================================
# figure reproductions (Figs 1-5) + the design-rule optimization
# ==========================================================================
def _use_agg() -> Any:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_schematic(path: Path) -> None:
    """Reproduce the Fig. 1 / Fig. 5 adiabatic-coupler region schematics."""
    plt = _use_agg()
    fig, axes = plt.subplots(2, 1, figsize=(9, 6))

    def draw(ax: Any, *, full: bool, title: str) -> None:
        l1, l2, l3 = 4.0, 3.0, 4.0
        d_i, d_ii, d_out = 1.6, 0.5, 1.8
        wa_in, wb_in, w_mid = 0.55, 0.30, 0.42
        # region I: converging; region II: tapering at constant gap;
        # region III: diverging. for the full coupler the widths swap at output.
        z1 = np.array([0.0, l1])
        z2 = np.array([l1, l1 + l2])
        z3 = np.array([l1 + l2, l1 + l2 + l3])
        for sign, w_in, w_out, col in (
            (1, wa_in, wb_in if full else w_mid, "#1f77b4"),
            (-1, wb_in, wa_in if full else w_mid, "#d62728"),
        ):
            yc1 = sign * (np.array([d_i, d_ii]) / 2 + 0.4)
            yc2 = sign * (np.array([d_ii, d_ii]) / 2 + 0.4)
            yc3 = sign * (np.array([d_ii, d_out]) / 2 + 0.4)
            ax.fill_between(z1, yc1 - w_in / 2, yc1 + w_in / 2, color=col, alpha=0.7)
            ax.fill_between(
                z2, yc2 - np.array([w_in, w_out]) / 2,
                yc2 + np.array([w_in, w_out]) / 2, color=col, alpha=0.7,
            )
            ax.fill_between(z3, yc3 - w_out / 2, yc3 + w_out / 2, color=col, alpha=0.7)
        for zb in (l1, l1 + l2):
            ax.axvline(zb, color="0.6", ls="--", lw=0.8)
        for z, lbl in ((l1 / 2, "I"), (l1 + l2 / 2, "II"), (l1 + l2 + l3 / 2, "III")):
            ax.text(z, 1.7, lbl, ha="center")
        ax.set_title(title)
        ax.set_xlabel("z (propagation)")
        ax.set_ylabel("x")
        ax.set_ylim(-2, 2)

    draw(axes[0], full=False, title="3 dB adiabatic coupler (Fig. 1a)")
    draw(axes[1], full=True, title="Full adiabatic coupler (Fig. 1b)")
    fig.suptitle("Ramadan 1998: adiabatic-coupler geometry")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_region1(path: Path, *, x_io: float = 119.0, eta: float = 420.0) -> None:
    """Reproduce Fig. 2: Region-I power ratio vs asynchronicity ``X_IIo``.

    Solid: eq (9) (large-asynchronicity asymptote); dashed: eq (10) (empirical,
    valid down to ``X_IIo ~ 1``). ``X_Io`` is fixed by the (fixed) Region-I
    input separation; ``eta`` (Region-I adiabaticity) sets the vertical scale.
    """
    plt = _use_agg()
    x_iio = np.logspace(np.log10(0.3), np.log10(10.0), 400)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(x_iio, q_region1_asymptote(x_io, x_iio, eta), "C0-", label="Equation (9)")
    ax.loglog(x_iio, q_region1(x_io, x_iio, eta), "k--", label="Equation (10)")
    ax.set_xlabel(r"Asynchronicity at the input of Region-II, $X_{IIo}$")
    ax.set_ylabel(r"Unwanted-to-wanted system-mode power ratio, $q_I$")
    ax.set_ylim(1e-4, 3e-2)
    ax.set_xlim(0.3, 10)
    ax.grid(visible=True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title(r"Fig. 2: Region-I power ratio ($\gamma=5$)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_region1_vs_length(path: Path) -> None:
    """Reproduce Fig. 3: Region-I power ratio vs Region-I length ``L_I``.

    In eq (10) the Region-I adiabaticity ``eta = (delta_beta_o/gamma) L_I/(d_I-d_II)``
    grows linearly with ``L_I``, so ``q_I ~ 1/L_I**2``. The two cases
    ``X_IIo = 0.66`` and ``1.86`` (paper's Fig. 3a/3b) at ``X_Io = 119`` are
    normalized to the paper's scale (``q_I ~ 0.1`` at ``L_I = 2000 um`` for the
    ``X_IIo = 0.66`` case).
    """
    plt = _use_agg()
    x_io = 119.0
    l_ref, eta_ref = 2000.0, 131.0  # eta(L_ref); sets the vertical scale
    length = np.linspace(1000.0, 25000.0, 400)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for x_iio, col, lbl in ((0.66, "C0", "0.66"), (1.86, "C3", "1.86")):
        eta = eta_ref * (length / l_ref)
        q = q_region1(x_io, x_iio, eta)
        ax.semilogy(length / 1e4, q, col, label=rf"$X_{{IIo}}={lbl}$ (eq 10)")
    ax.set_xlabel(r"Length of Region-I, $L_I$ [$\times 10^4\,\mu$m]")
    ax.set_ylabel(r"Unwanted-to-wanted system-mode power ratio, $q_I$")
    ax.set_ylim(1e-4, 1e-1)
    ax.grid(visible=True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title("Fig. 3: Region-I power ratio vs length")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_region2(path: Path, *, kappa_ii: float = 0.005, x_iio: float = 0.66) -> None:
    """Reproduce Fig. 4: Region-II power ratio vs Region-II length ``L_II``.

    eq (15) oscillates with nulls at ``kappa_II L_II = n*pi``; its envelope
    ``(m X_IIo / (2 kappa_II L_II))**2`` (dashed) bounds it.
    """
    plt = _use_agg()
    length = np.linspace(100.0, 4000.0, 2000)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.semilogy(
        length, q_region2(length, kappa_ii=kappa_ii, x_iio=x_iio, m=1),
        "C0-", lw=0.8, label="Equation (15)",
    )
    ax.semilogy(
        length, q_region2_envelope(length, kappa_ii=kappa_ii, x_iio=x_iio, m=1),
        "k--", label="Envelope",
    )
    ax.set_xlabel(r"Length of Region-II, $L_{II}$ [$\mu$m]")
    ax.set_ylabel(r"Unwanted-to-wanted system-mode power ratio, $q_{II}$")
    ax.set_ylim(1e-6, 1e-1)
    ax.grid(visible=True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title(r"Fig. 4: Region-II power ratio (3 dB, $X_{IIo}=0.66$)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_design_rules(path: Path, *, kappa_ii: float = 0.005) -> None:
    """Optimized device length vs isolated-power tolerance (eqs 28/29).

    Shows the paper's central result: the 3 dB coupler length scales as
    ``1/epsilon`` while the full coupler scales as ``1/sqrt(epsilon)``, so 3 dB
    adiabatic couplers must be much longer for the same tolerance.
    """
    plt = _use_agg()
    eps = np.logspace(-3, -0.7, 200)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(
        eps, [length_3db_optimum(e, kappa_ii) for e in eps],
        "C0", label=r"3 dB, eq (28): $\frac{8}{\pi}\frac{L_c}{\epsilon}$",
    )
    ax.loglog(
        eps, [length_full_optimum(e, kappa_ii) for e in eps],
        "C3", label=r"full, eq (29): $\frac{3}{\pi}\frac{L_c}{\sqrt{\epsilon}}$",
    )
    ax.set_xlabel(r"Isolated-waveguide power tolerance, $\epsilon$")
    ax.set_ylabel(r"Optimum device length [$\mu$m]")
    ax.grid(visible=True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title(rf"Design rules ($L_c={coupling_length(kappa_ii):.0f}\,\mu$m)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ==========================================================================
# meow EME verification on an SOI strip-waveguide adiabatic 3 dB coupler
# (the meow analogue of the paper's BPM numerical check)
# ==========================================================================
def adiabatic_coupler_component(
    *,
    length_i: float,
    length_ii: float,
    length_iii: float,
    w_a_in: float = 0.50,
    w_b_in: float = 0.40,
    w_mid: float = 0.45,
    gap_i: float = 1.0,
    gap_ii: float = 0.20,
    gap_out: float = 1.5,
    npts: int = 200,
) -> Any:
    """Build a 3-region SOI adiabatic 3 dB coupler as a gdsfactory component.

    Two strip waveguides (upper ``A``, lower ``B``) share a constant-gap
    coupling Region II in which their widths taper to the same ``w_mid`` (the 3
    dB condition). Region I converges the gap from ``gap_i`` to ``gap_ii`` at
    fixed input widths; Region III diverges the gap to ``gap_out`` at fixed
    ``w_mid``. Propagation is along gdsfactory ``x``; lateral is ``y``.
    """
    import gdsfactory as gf

    z1, z2, z3 = length_i, length_i + length_ii, length_i + length_ii + length_iii

    def seg(z: float) -> tuple[float, float, float]:
        """(gap, w_a, w_b) at propagation coordinate ``z``."""
        if z <= z1:
            t = z / length_i if length_i else 1.0
            return gap_i + (gap_ii - gap_i) * t, w_a_in, w_b_in
        if z <= z2:
            t = (z - z1) / length_ii if length_ii else 1.0
            return gap_ii, w_a_in + (w_mid - w_a_in) * t, w_b_in + (w_mid - w_b_in) * t
        t = (z - z2) / length_iii if length_iii else 1.0
        return gap_ii + (gap_out - gap_ii) * t, w_mid, w_mid

    zs = np.linspace(0.0, z3, npts)
    gw = np.array([seg(float(z)) for z in zs])
    gap, w_a, w_b = gw[:, 0], gw[:, 1], gw[:, 2]
    # inner edges at +-gap/2; A above, B below
    a_lo, a_hi = gap / 2, gap / 2 + w_a
    b_hi, b_lo = -gap / 2, -gap / 2 - w_b

    c = gf.Component()

    def strip(top: np.ndarray, bot: np.ndarray) -> np.ndarray:
        upper = np.column_stack([zs, top])
        lower = np.column_stack([zs[::-1], bot[::-1]])
        return np.vstack([upper, lower])

    c.add_polygon(strip(a_hi, a_lo), layer=LAYER_WG)
    c.add_polygon(strip(b_hi, b_lo), layer=LAYER_WG)
    c.add_port(
        "in_a", center=(0.0, float(a_lo[0] + w_a[0] / 2)), width=w_a_in,
        orientation=180, layer=LAYER_WG,
    )
    c.add_port(
        "out_a", center=(z3, float(a_lo[-1] + w_mid / 2)), width=w_mid,
        orientation=0, layer=LAYER_WG,
    )
    return c


def _extrude(component: Any, x_span: tuple[float, float]) -> list[Any]:
    import meow as mw

    rule = mw.GdsExtrusionRule(material=mw.silicon, h_min=0.0, h_max=H_SI)
    structs = mw.extrude_gds(component, {LAYER_WG: [rule]})
    oxide = mw.Structure(
        material=mw.silicon_oxide,
        geometry=mw.Box(
            x_min=x_span[0], x_max=x_span[1], y_min=-T_BOX, y_max=H_SI + T_CLAD,
            z_min=0.0, z_max=float(component.xmax),
        ),
        mesh_order=10,
    )
    return [*structs, oxide]


def device_cells(component: Any, *, num_cells: int, res: float) -> list[Any]:
    """Slice the coupler into ``num_cells`` equal-length EME cells."""
    import meow as mw

    span = 2.0 + 0.5 * 1.5  # lateral half-width margin
    mesh = mw.Mesh2D(
        x=np.arange(-span, span + res / 2, res),
        y=np.arange(-T_BOX, H_SI + T_CLAD + res / 2, res),
    )
    structs = _extrude(component, x_span=(-span, span))
    length = float(component.xmax)
    lengths = np.full(num_cells, length / num_cells)
    return mw.create_cells(structs, mesh, lengths, z_min=0.0)


def bar_cross_transmission(
    cells: list[Any],
    env: Any,
    *,
    num_modes: int,
    compute_modes: Callable | None = None,
) -> tuple[float, float]:
    """(bar, cross) output power for the fundamental input mode.

    ``bar`` is the power that stays in the upper waveguide (``A``, lateral
    centroid > 0), ``cross`` the power coupled to the lower waveguide (``B``).
    For an ideal adiabatic 3 dB coupler both approach 0.5.
    """
    from examples.papers._backends import device_s_matrix

    s, pm = device_s_matrix(
        cells, env, num_modes=num_modes, parallel=False, compute_modes=compute_modes
    )
    return attribute_bar_cross(
        s, pm, cells[-1], env, num_modes=num_modes, compute_modes=compute_modes
    )


def attribute_bar_cross(
    s_matrix: Any,
    port_map: dict[str, int],
    cell_last: Any,
    env: Any,
    *,
    num_modes: int,
    compute_modes: Callable | None = None,
) -> tuple[float, float]:
    """(bar, cross) split for the fundamental input from a cascaded S-matrix.

    Solves the output modes on the final cell and attributes each output port's
    power to the upper (``bar``, lateral centroid > 0) or lower (``cross``)
    waveguide. Shared by the in-process and the distributed (slurm) paths.
    """
    import meow as mw

    solver = compute_modes or mw.compute_modes
    modes_out = solver(mw.CrossSection.from_cell(cell=cell_last, env=env), num_modes)
    s = np.asarray(s_matrix)
    bar = cross = 0.0
    for i, mode in enumerate(modes_out):
        power = float(np.abs(s[port_map[f"right@{i}"], port_map["left@0"]]) ** 2)
        density = np.abs(mode.Ex) ** 2 + np.abs(mode.Ey) ** 2 + np.abs(mode.Ez) ** 2
        centroid = float(np.sum(mode.cs.mesh.Xx * density) / np.sum(density))
        if centroid > 0:
            bar += power
        else:
            cross += power
    return bar, cross


def eme_length_sweep(
    lengths_ii: np.ndarray,
    *,
    wl: float = 1.55,
    num_cells: int,
    num_modes: int,
    res: float,
    length_i: float = 20.0,
    length_iii: float = 20.0,
    compute_modes: Callable | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep the Region-II length and return (bar, cross) transmission arrays.

    As Region II lengthens the transfer becomes adiabatic and the split
    approaches 50/50 - the EME analogue of the paper's BPM convergence study.
    """
    import meow as mw

    env = mw.Environment(wl=wl, T=25.0)
    bar, cross = [], []
    for l_ii in lengths_ii:
        comp = adiabatic_coupler_component(
            length_i=length_i, length_ii=float(l_ii), length_iii=length_iii
        )
        cells = device_cells(comp, num_cells=num_cells, res=res)
        b, x = bar_cross_transmission(
            cells, env, num_modes=num_modes, compute_modes=compute_modes
        )
        bar.append(b)
        cross.append(x)
    return np.asarray(bar), np.asarray(cross)


def plot_eme_verification(
    lengths_ii: np.ndarray, bar: np.ndarray, cross: np.ndarray, path: Path
) -> None:
    """Plot the EME bar/cross split vs Region-II length (coarse cross-check).

    NOTE: this is a *qualitative*, deliberately affordable EME cross-check (the
    meow analogue of the paper's BPM run), not a converged result. On a
    high-index-contrast SOI strip the transmitted power ``bar + cross`` is well
    below 1 and length-dependent because the coarse mesh / limited mode basis
    leaks power at every cell interface over the ~100+ cells of the device; a
    converged check needs a much finer mesh and more modes (far slower). The
    paper's matched *model results* are the analytic Figs. 2-4 and the design
    rules; this panel only shows the coupling trend.
    """
    plt = _use_agg()
    total = np.asarray(bar) + np.asarray(cross)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(lengths_ii, bar, "C0o-", label="bar (input guide A)")
    ax.plot(lengths_ii, cross, "C3s-", label="cross (guide B)")
    ax.plot(lengths_ii, total, "k.--", lw=1.0, label="bar + cross (transmitted)")
    ax.axhline(0.5, color="0.5", ls=":", lw=1.0, label="ideal 3 dB (50/50)")
    ax.set_xlabel(r"Region-II length, $L_{II}$ [$\mu$m]")
    ax.set_ylabel("output power fraction")
    ax.set_ylim(0, 1.05)
    ax.grid(visible=True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("meow EME (coarse cross-check, under-resolved — see docstring)")
    ax.text(
        0.5, 0.02,
        "qualitative only: coarse mesh leaks power; analytic Figs 2-4 are the "
        "matched result",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color="0.4",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> dict[str, Any]:
    """Reproduce the Ramadan 1998 analytic figures + design rules into FIGDIR.

    Always writes the (fast) analytic figures, the geometry schematics, the
    design-rule plot and a representative GDS. When ``run_eme`` (default on; set
    ``MEOW_RAMADAN_EME=0`` to skip) it additionally runs the meow EME
    Region-II-length sweep on the SOI coupler and writes the verification figure
    plus the bar/cross data (CSV + JSON). Resolution follows ``MEOW_EXAMPLE_RES``.
    """
    import os

    import meow as mw
    from examples.papers import _resolution as res

    out = FIGDIR / "ramadan1998"
    out.mkdir(parents=True, exist_ok=True)
    plot_schematic(out / "fig1_schematic.png")
    plot_region1(out / "fig2_region1_vs_asynchronicity.png")
    plot_region1_vs_length(out / "fig3_region1_vs_length.png")
    plot_region2(out / "fig4_region2_vs_length.png")
    plot_design_rules(out / "design_rules_length_vs_tolerance.png")

    # a representative GDS of a 3 dB adiabatic coupler
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    coupler = adiabatic_coupler_component(length_i=20, length_ii=120, length_iii=20)
    coupler.write_gds(str(out / "adiabatic_3db_coupler.gds"))

    summary: dict[str, Any] = {"out_dir": str(out)}
    if os.environ.get("MEOW_RAMADAN_EME", "1") not in ("0", "", "false", "False"):
        npts = res.pick(low=3, medium=5, high=7)
        device_res = res.pick(low=0.06, medium=0.035, high=0.025)
        num_modes = res.num_modes(low=4, medium=6, high=8)
        lengths_ii = np.linspace(30.0, 300.0, npts)
        # ~1 EME cell per 1.5 um of device, plus the fixed I/III regions
        num_cells = res.num_cells(
            low=24, medium=int((np.mean(lengths_ii) + 40) / 1.5), high=200
        )
        bar, cross = eme_length_sweep(
            lengths_ii, num_cells=num_cells, num_modes=num_modes, res=device_res
        )
        plot_eme_verification(lengths_ii, bar, cross, out / "eme_verification.png")
        mw.save_table(
            out / "eme_verification",
            {"length_ii_um": lengths_ii, "bar": bar, "cross": cross},
        )
        summary["eme"] = {
            "lengths_ii_um": lengths_ii.tolist(),
            "bar": bar.tolist(),
            "cross": cross.tolist(),
        }
    summary["files"] = sorted(p.name for p in out.glob("*"))
    return summary


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
