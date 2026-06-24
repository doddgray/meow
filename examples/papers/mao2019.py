"""Mao et al., *Adiabatic Coupler With Design-Intended Splitting Ratio*,
J. Lightwave Technol. 37(24), 6147 (2019).

This example reproduces the paper's semi-analytical design relation between an
adiabatic coupler's power **splitting ratio** (SR) and the output-waveguide
width-difference, and adds a designer that hits a target SR on a chosen stack.

The device is a five-region SOI adiabatic coupler (Fig. 1). The SR is set at the
right-hand side of Region III by the output width-difference
``dw_out = w3 - w4``: a symmetric output (``dw_out = 0``) gives 50/50, and
increasing ``dw_out`` pushes more power into the wider guide. Following the
paper's coupled-local-mode analysis (eqs 7-12), the upper/lower output power
ratio is

    P_up / P_down = 1 / tan(xi/2)**2,     tan(xi) = c / phi,

with ``phi = (pi/lambda) dn_i`` (``dn_i`` = effective-index difference of the
two *isolated* output waveguides ``w3``, ``w4``) and the local-mode coupling
``c = pi dn_{i,i+1} / lambda`` (``dn_{i,i+1}`` = the supermode index splitting of
the *coupled* ``w3``/``w4`` pair at gap ``g2``). Both ``dn_i`` and
``dn_{i,i+1}`` are obtained from meow's FDE solver, so the SR-vs-``dw_out`` curve
(paper Fig. 2) is reproduced cheaply and exactly from the model.

The Region-III adiabatic insertion loss (eq 5) oscillates as
``(1/(kappa L))**2 sin(kappa L)**2`` just like the Ramadan 1998 Region-II loss.

Run with ``python -m examples.papers.mao2019``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FIGDIR = Path(__file__).parent / "figures"
LAYER_WG = (1, 0)

# SOI strip stack (O-band design, the paper's 1260-1360 nm window)
H_SI = 0.22
T_CLAD = 0.6
T_BOX = 0.6


@dataclass
class SOIStack:
    """SOI strip stack for the SR analysis/designer."""

    core_thickness: float = H_SI
    wl: float = 1.31
    w1: float = 0.35  # wider output waveguide width [um]
    g2: float = 0.15  # output coupling gap [um]
    name: str = "SOI-220nm"


def _cross_section(widths: list[tuple[float, float]], wl: float, res: float) -> Any:
    """A meow cross-section of SOI strips at given (center_x, width) positions."""
    import meow as mw

    h = H_SI
    structs = [
        mw.Structure(
            material=mw.silicon,
            geometry=mw.Box(
                x_min=cx - w / 2, x_max=cx + w / 2, y_min=0.0, y_max=h,
                z_min=0.0, z_max=1.0,
            ),
        )
        for cx, w in widths
    ]
    structs.append(
        mw.Structure(
            material=mw.silicon_oxide,
            geometry=mw.Box(
                x_min=-2.0, x_max=2.0, y_min=-T_BOX, y_max=h + T_CLAD,
                z_min=0.0, z_max=1.0,
            ),
            mesh_order=10,
        )
    )
    mesh = mw.Mesh2D(
        x=np.arange(-2.0, 2.0 + res / 2, res),
        y=np.arange(-T_BOX, h + T_CLAD + res / 2, res),
    )
    cells = mw.create_cells(structs, mesh, np.array([1.0]), z_min=0.0)
    return mw.CrossSection.from_cell(cell=cells[0], env=mw.Environment(wl=wl, T=25.0))


def single_neff(width: float, wl: float, *, res: float = 0.02) -> float:
    """Fundamental TE effective index of an isolated SOI strip of ``width``."""
    import meow as mw

    cs = _cross_section([(0.0, width)], wl, res)
    modes = mw.compute_modes(cs, num_modes=2)
    te = max(
        (m for m in modes if float(mw.te_fraction(m)) >= 0.5),
        key=lambda m: np.real(m.neff), default=modes[0],
    )
    return float(np.real(te.neff))


def supermode_split(
    w3: float, w4: float, gap: float, wl: float, *, res: float = 0.02
) -> float:
    """Supermode index splitting ``dn_{i,i+1}`` of a coupled ``w3``/``w4`` pair."""
    import meow as mw

    cx3 = (gap + w3) / 2
    cx4 = -(gap + w4) / 2
    cs = _cross_section([(cx3, w3), (cx4, w4)], wl, res)
    modes = mw.compute_modes(cs, num_modes=2)
    neffs = sorted((float(np.real(m.neff)) for m in modes), reverse=True)
    return abs(neffs[0] - neffs[1])


def splitting_ratio(
    dw_out: float, stack: SOIStack, *, res: float = 0.02
) -> float:
    """Semi-analytical SR for output width-difference ``dw_out`` (eqs 7-12).

    ``w3 = w1`` (fixed wider guide), ``w4 = w1 - dw_out``. Returns the fraction
    of power in the upper (wider) output waveguide.
    """
    w3 = stack.w1
    w4 = stack.w1 - dw_out
    wl = stack.wl
    dn_i = abs(single_neff(w3, wl, res=res) - single_neff(w4, wl, res=res))
    dn_sup = supermode_split(w3, w4, stack.g2, wl, res=res)
    phi = np.pi * dn_i / wl
    c = np.pi * dn_sup / wl
    xi = np.arctan2(c, phi)  # -> pi/2 as phi -> 0 (symmetric -> 50/50)
    ratio_up_down = 1.0 / np.tan(xi / 2) ** 2
    return float(ratio_up_down / (1.0 + ratio_up_down))


def _use_agg() -> Any:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_sr_vs_dwout(
    dw_outs: np.ndarray, srs: np.ndarray, path: Path, *, wl: float = 1.31
) -> None:
    """Reproduce Fig. 2: semi-analytical SR vs output width-difference."""
    plt = _use_agg()
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.plot(dw_outs * 1000, srs * 100, "C0o-")
    ax.set_xlabel(r"$\Delta w_{out}$ [nm]")
    ax.set_ylabel("splitting ratio (upper port) [%]")
    ax.set_ylim(45, 100)
    ax.grid(visible=True, alpha=0.3)
    ax.set_title(f"Fig. 2: SR vs output width-difference @ {wl * 1000:.0f} nm")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_schematic(path: Path) -> None:
    """Reproduce the Fig. 1 five-region ADC schematic."""
    plt = _use_agg()
    fig, ax = plt.subplots(figsize=(10, 3))
    ls = [("I", 3.0), ("II", 2.0), ("III", 3.0), ("IV", 2.0), ("V", 3.0)]
    g1, g2 = 1.4, 0.5
    z = 0.0
    bounds = []
    for _, length in ls:
        bounds.append((z, z + length))
        z += length
    # upper/lower waveguide centre gaps per region (converge in II, hold, diverge IV)
    gaps = [g1, g2, g2, g1, g1]
    w_up = [0.45, 0.42, 0.40, 0.40, 0.45]
    w_dn = [0.35, 0.34, 0.34, 0.34, 0.35]
    rows = zip(bounds, gaps, w_up, w_dn, ls, strict=True)
    for (z0, z1), gap, wu, wd, (lbl, _) in rows:
        yc_u, yc_d = gap / 2 + 0.4, -(gap / 2 + 0.4)
        ax.fill_between([z0, z1], yc_u - wu / 2, yc_u + wu / 2, color="C0", alpha=0.7)
        ax.fill_between([z0, z1], yc_d - wd / 2, yc_d + wd / 2, color="C3", alpha=0.7)
        ax.axvline(z1, color="0.7", ls="--", lw=0.6)
        ax.text((z0 + z1) / 2, 1.6, lbl, ha="center")
    ax.set_xlabel("z (propagation)")
    ax.set_ylabel("x")
    ax.set_ylim(-2, 2)
    ax.set_title("Fig. 1: five-region adiabatic coupler with intended SR")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> dict[str, Any]:
    """Reproduce the Mao 2019 SR-vs-width-difference design relation."""
    import gdsfactory as gf

    import meow as mw
    from examples.papers import _resolution as res

    gf.gpdk.PDK.activate()
    out = FIGDIR / "mao2019"
    out.mkdir(parents=True, exist_ok=True)
    stack = SOIStack()
    npts = res.pick(low=6, medium=11, high=26)
    dw_outs = np.linspace(0.0, 0.05, npts)
    device_res = res.pick(low=0.03, medium=0.02, high=0.015)
    srs = np.array([splitting_ratio(float(d), stack, res=device_res) for d in dw_outs])
    plot_sr_vs_dwout(dw_outs, srs, out / "fig2_sr_vs_dwout.png", wl=stack.wl)
    plot_schematic(out / "fig1_schematic.png")
    mw.save_table(
        out / "sr_vs_dwout", {"dw_out_nm": dw_outs * 1000, "sr_upper": srs}
    )
    summary = {"wl_nm": stack.wl * 1000, "sr_at_0": float(srs[0]),
               "sr_at_max_dwout": float(srs[-1])}
    mw.save_summary(out / "summary", summary)
    return {"out_dir": str(out), "summary": summary,
            "files": sorted(p.name for p in out.glob("*"))}


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
