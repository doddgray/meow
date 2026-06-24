"""Adiabatic-coupler designer applying the Ramadan 1998 design rules to new specs.

Companion to :mod:`examples.papers.ramadan1998`. Where the analysis module
reproduces the paper's figures, this **designer** applies the paper's optimized
design rules (eqs 28/29) to a *new* set of target specifications on a *specified
waveguide layer stack*, producing the analogous figures and the device GDS.

Design workflow (the paper's, made concrete with meow FDE):

1. On the chosen layer stack, the Region-II coupling coefficient ``kappa_II`` of
   two identical coupled waveguides at the design gap is obtained from the FDE
   supermode splitting, ``kappa_II = pi (n_even - n_odd) / lambda``, giving the
   conventional coupling length ``L_c = pi / (2 kappa_II)``.
2. The optimized device length for the target isolated-power error tolerance
   ``epsilon`` follows from the design rules: ``L_3dB ~ (8/pi) L_c/epsilon``
   (eq 28) and ``L_full ~ (3/pi) L_c/sqrt(epsilon)`` (eq 29).
3. The three-region adiabatic coupler is laid out at that length and written to
   GDS; the design-rule curve (with the chosen design point marked), the device
   layout and an optional meow-EME check at the designed length are saved.

Default new specs (sensible per-paper defaults): a 220 nm silicon-on-insulator
strip stack, a 1310 nm (O-band) center wavelength, and a target tolerance of
``epsilon = 0.02`` (about a 0.4 dB / 20:1 isolated-waveguide power balance),
designing both a 3 dB and a full (cross) adiabatic coupler.

Run with ``python -m examples.papers.ramadan1998_designer``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from examples.papers import ramadan1998 as r
from examples.papers.ramadan1998 import (
    AdiabaticCouplerSpec,
    coupling_length,
    length_3db_optimum,
    length_full_optimum,
)

FIGDIR = r.FIGDIR


@dataclass
class Stack:
    """A waveguide layer stack for the designer.

    Attributes:
        core: high-index core material.
        clad: cladding material.
        core_thickness: core layer thickness [um].
        wl: design center wavelength [um].
        w: nominal (Region-II) waveguide width [um].
        gap: Region-II edge-to-edge gap [um].
    """

    core: Any
    clad: Any
    core_thickness: float
    wl: float
    w: float
    gap: float
    name: str = "SOI-220nm"


def soi_stack(wl: float = 1.31) -> Stack:
    """Default 220 nm silicon-on-insulator strip stack at ``wl`` [um]."""
    import meow as mw

    return Stack(
        core=mw.silicon, clad=mw.silicon_oxide, core_thickness=0.22,
        wl=wl, w=0.40, gap=0.20, name="SOI-220nm",
    )


def coupling_coefficient(stack: Stack, *, res: float = 0.02) -> float:
    """Region-II coupling coefficient ``kappa_II`` [1/um] from FDE supermodes.

    Solves the two lowest supermodes of two identical ``stack.w`` waveguides at
    the design gap and returns ``kappa_II = pi (n_even - n_odd) / lambda``.
    """
    import meow as mw

    w, gap, h = stack.w, stack.gap, stack.core_thickness
    centers = (gap / 2 + w / 2, -(gap / 2 + w / 2))
    structs = [
        mw.Structure(
            material=stack.core,
            geometry=mw.Box(
                x_min=cx - w / 2, x_max=cx + w / 2, y_min=0.0, y_max=h,
                z_min=0.0, z_max=1.0,
            ),
        )
        for cx in centers
    ]
    structs.append(
        mw.Structure(
            material=stack.clad,
            geometry=mw.Box(
                x_min=-3.0, x_max=3.0, y_min=-r.T_BOX, y_max=h + r.T_CLAD,
                z_min=0.0, z_max=1.0,
            ),
            mesh_order=10,
        )
    )
    mesh = mw.Mesh2D(
        x=np.arange(-3.0, 3.0 + res / 2, res),
        y=np.arange(-r.T_BOX, h + r.T_CLAD + res / 2, res),
    )
    cells = mw.create_cells(structs, mesh, np.array([1.0]), z_min=0.0)
    env = mw.Environment(wl=stack.wl, T=25.0)
    cs = mw.CrossSection.from_cell(cell=cells[0], env=env)
    modes = mw.compute_modes(cs, num_modes=2)
    n_even, n_odd = float(np.real(modes[0].neff)), float(np.real(modes[1].neff))
    return float(np.pi * abs(n_even - n_odd) / stack.wl)


def design(
    stack: Stack, *, kind: str, epsilon: float, res: float = 0.02
) -> AdiabaticCouplerSpec:
    """Design a ``kind`` adiabatic coupler on ``stack`` for tolerance ``epsilon``."""
    kappa_ii = coupling_coefficient(stack, res=res)
    length = (
        length_3db_optimum(epsilon, kappa_ii)
        if kind == "3dB"
        else length_full_optimum(epsilon, kappa_ii)
    )
    return AdiabaticCouplerSpec(
        kind=kind, kappa_ii=kappa_ii, x_iio=2.0, gamma=2.0,
        epsilon=epsilon, length_um=float(length),
    )


def plot_design_point(
    spec: AdiabaticCouplerSpec, path: Path, *, title: str
) -> None:
    """Design-rule length-vs-tolerance curve with the chosen design point marked."""
    plt = r._use_agg()
    eps = np.logspace(-3, -0.7, 200)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    lc = coupling_length(spec.kappa_ii)
    ax.loglog(eps, [length_3db_optimum(e, spec.kappa_ii) for e in eps], "C0",
              label=r"3 dB (eq 28)")
    ax.loglog(eps, [length_full_optimum(e, spec.kappa_ii) for e in eps], "C3",
              label=r"full (eq 29)")
    ax.plot([spec.epsilon], [spec.length_um], "k*", ms=14,
            label=f"design: {spec.kind}, L={spec.length_um:.0f} um")
    ax.set_xlabel(r"Isolated-waveguide power tolerance, $\epsilon$")
    ax.set_ylabel(r"Optimum device length [$\mu$m]")
    ax.grid(visible=True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title(f"{title}  ($L_c={lc:.0f}\\,\\mu$m)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_layout(spec: AdiabaticCouplerSpec, component: Any, path: Path) -> None:
    """Draw the designed coupler layout."""
    plt = r._use_agg()
    from examples.papers._plot import plot_component

    fig, ax = plt.subplots(figsize=(10, 3))
    plot_component(component, ax)
    ax.set_aspect("auto")
    ax.set_title(
        f"{spec.kind} adiabatic coupler: L={spec.length_um:.0f} um, "
        f"kappa_II={spec.kappa_ii:.3f}/um, eps={spec.epsilon}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def designed_component(spec: AdiabaticCouplerSpec, stack: Stack) -> Any:
    """Lay out the designed coupler (Region II = the optimized length)."""
    # split the optimized length: most into Region II, short converging/diverging
    l_taper = max(15.0, 0.1 * spec.length_um)
    l_ii = max(10.0, spec.length_um - 2 * l_taper)
    w_out = 0.50 if spec.kind == "3dB" else 0.55
    return r.adiabatic_coupler_component(
        length_i=l_taper, length_ii=l_ii, length_iii=l_taper,
        w_a_in=stack.w + 0.10, w_b_in=stack.w, w_mid=stack.w + 0.05,
        gap_ii=stack.gap, gap_out=w_out + 1.0,
    )


def design_and_save(
    stack: Stack, *, kind: str, epsilon: float, out: Path, res: float = 0.02
) -> dict[str, Any]:
    """Design one coupler, write its GDS + figures, return a summary dict."""
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    spec = design(stack, kind=kind, epsilon=epsilon, res=res)
    component = designed_component(spec, stack)
    stem = f"{stack.name}_{kind}_{int(stack.wl * 1000)}nm"
    component.write_gds(str(out / f"{stem}.gds"))
    plot_design_point(spec, out / f"{stem}_design_rule.png",
                      title=f"{stack.name} @ {stack.wl * 1000:.0f} nm")
    plot_layout(spec, component, out / f"{stem}_layout.png")
    return {
        "kind": kind, "stack": stack.name, "wl_nm": stack.wl * 1000,
        "epsilon": epsilon, "kappa_ii_per_um": spec.kappa_ii,
        "coupling_length_um": spec.coupling_length_um, "length_um": spec.length_um,
        "stem": stem,
    }


def main() -> dict[str, Any]:
    """Design a 3 dB and a full adiabatic coupler for new O-band SOI specs."""
    out = FIGDIR / "ramadan1998_designer"
    out.mkdir(parents=True, exist_ok=True)
    stack = soi_stack(wl=1.31)
    summaries = {
        kind: design_and_save(stack, kind=kind, epsilon=0.02, out=out)
        for kind in ("3dB", "full")
    }
    import meow as mw

    for kind, s in summaries.items():
        mw.save_summary(out / f"{s['stem']}_summary", s)
        _ = kind
    return {"out_dir": str(out), "designs": summaries,
            "files": sorted(p.name for p in out.glob("*"))}


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
