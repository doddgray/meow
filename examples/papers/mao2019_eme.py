"""Full EME model of the Mao 2019 splitting-ratio adiabatic coupler.

Builds an eigenmode-expansion model of the five-region SOI ADC laid out by
:func:`examples.papers.mao2019_designer.adc_component`, for the paper's 50/50
coupler and a designer 75/25 coupler, and saves for each:

- the **upper/lower output transmission spectrum** (the key S-matrix elements,
  i.e. the splitting ratio) in dB over an octave about 1.31 um, for the TE input
  at port 1;
- the **propagating |E| field** for the TE port-1 input at 1.31 um;
- a **section-annotated layout** (Regions I-V) with the design parameters.

Run with ``python -m examples.papers.mao2019_eme``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from examples.papers import _eme_model as em
from examples.papers import mao2019 as m
from examples.papers import mao2019_designer as md

FIGDIR = m.FIGDIR
LAYER_WG = m.LAYER_WG
CENTER = 1.31


def _extrude(component: Any) -> list[Any]:
    import meow as mw

    rule = mw.GdsExtrusionRule(material=mw.silicon, h_min=0.0, h_max=m.H_SI)
    structs = mw.extrude_gds(component, {LAYER_WG: [rule]})
    oxide = mw.Structure(
        material=mw.silicon_oxide,
        geometry=mw.Box(
            x_min=-3.5, x_max=3.5, y_min=-m.T_BOX, y_max=m.H_SI + m.T_CLAD,
            z_min=0.0, z_max=float(component.xmax),
        ),
        mesh_order=10,
    )
    return [*structs, oxide]


def _mesh(fine_res: float) -> Any:
    import meow as mw

    x = np.arange(-3.5, 3.5 + fine_res / 2, fine_res)
    y = np.arange(-m.T_BOX, m.H_SI + m.T_CLAD + fine_res / 2, fine_res)
    return mw.Mesh2D(x=x, y=y)


def device_cells(component: Any, *, num_cells: int, fine_res: float) -> list[Any]:
    import meow as mw

    structs = _extrude(component)
    lengths = np.full(num_cells, float(component.xmax) / num_cells)
    return mw.create_cells(structs, _mesh(fine_res), lengths, z_min=0.0)


def _solve(cells: list[Any], wl: float, num_modes: int) -> list[list[Any]]:
    import meow as mw

    env = mw.Environment(wl=wl, T=25.0)
    css = [mw.CrossSection.from_cell(cell=c, env=env) for c in cells]
    return [mw.compute_modes(cs, num_modes=num_modes) for cs in css]


def _upper_input_index(modes_in: list[Any]) -> int:
    """Highest-neff TE input mode localized in the upper waveguide (centroid>0)."""
    import meow as mw

    cands = [
        i for i, mo in enumerate(modes_in)
        if float(mw.te_fraction(mo)) >= 0.5 and em.lateral_centroid(mo) > 0
    ]
    if not cands:
        return 0
    return max(cands, key=lambda i: np.real(modes_in[i].neff))


def split_powers_db(
    component: Any, wl: float, *, num_cells: int, num_modes: int, fine_res: float
) -> tuple[float, float]:
    """(upper, lower) output power transmission [dB] for the TE port-1 input."""
    import meow as mw

    cells = device_cells(component, num_cells=num_cells, fine_res=fine_res)
    modes = _solve(cells, wl, num_modes)
    s, pm = mw.compute_s_matrix(modes, cells=cells)
    s = np.asarray(s)
    in_idx = _upper_input_index(modes[0])
    up = dn = 0.0
    for i, mo in enumerate(modes[-1]):
        p = float(np.abs(s[pm[f"right@{i}"], pm[f"left@{in_idx}"]]) ** 2)
        if em.lateral_centroid(mo) > 0:
            up += p
        else:
            dn += p
    return 10 * np.log10(max(up, 1e-6)), 10 * np.log10(max(dn, 1e-6))


def run_design(
    component: Any,
    label: str,
    out: Path,
    *,
    dividers: list[tuple[float, str]],
    params: str,
    num_cells: int,
    num_modes: int,
    fine_res: float,
    spec_npts: int,
) -> dict[str, Any]:
    """EME-model one ADC layout; save spectrum, propagation, annotated layout."""
    import meow as mw

    em.plot_annotated_layout(
        component, out / f"{label}_layout.png",
        title=f"Mao 2019 SR adiabatic coupler - {label}",
        layer_styles={LAYER_WG: ("SOI waveguides", "#1f77b4")},
        dividers=dividers, params=params, ylim=(-2.5, 2.5),
    )
    wls = em.octave_wls(CENTER, spec_npts)
    up, dn = [], []
    for wl in wls:
        u, d = split_powers_db(component, float(wl), num_cells=num_cells,
                               num_modes=num_modes, fine_res=fine_res)
        up.append(u)
        dn.append(d)
    series = {r"$|S_{up,1}|^2$ (bar)": np.array(up),
              r"$|S_{down,1}|^2$ (cross)": np.array(dn)}
    em.plot_spectrum(wls, series, out / f"{label}_spectrum.png",
                     title=f"Output splitting spectrum - {label}",
                     center_nm=CENTER * 1000)
    mw.save_table(out / f"{label}_spectrum",
                  {"wl_nm": wls * 1000, "upper_db": up, "lower_db": dn})
    # propagation: TE port-1 input at the design wavelength
    cells = device_cells(component, num_cells=num_cells, fine_res=fine_res)
    modes = _solve(cells, CENTER, num_modes)
    idx = _upper_input_index(modes[0])
    field, x_trans = mw.propagate_modes(modes, cells, excite_mode_l=idx,
                                        y=m.H_SI / 2, num_z=400)
    em.plot_propagation(np.abs(np.asarray(field)), np.asarray(x_trans),
                        float(component.xmax), out / f"{label}_propagation_TE.png",
                        title=f"|E| propagation, TE port-1 input @ 1310 nm - {label}",
                        ylim=(-2.5, 2.5))
    return {"label": label, "upper_db_1310": float(np.interp(CENTER, wls, up)),
            "lower_db_1310": float(np.interp(CENTER, wls, dn))}


def _dividers(stack: m.SOIStack, dw_out: float) -> tuple[list[tuple[float, str]], str]:
    lt, ls, l3 = 40.0, 30.0, 60.0
    z = [lt, lt + ls, lt + ls + l3, lt + 2 * ls + l3]
    labels = ["I", "II", "III", "IV", "V"]
    centers = [lt / 2, lt + ls / 2, lt + ls + l3 / 2, lt + 1.5 * ls + l3,
               lt + 2 * ls + l3 + lt / 2]
    dividers = [(zz, "") for zz in z] + list(zip(centers, labels, strict=True))
    params = (f"w1={stack.w1} um, g2={stack.g2} um, dw_out={dw_out * 1000:.1f} nm "
              f"@ {stack.wl * 1000:.0f} nm")
    return dividers, params


def main() -> dict[str, Any]:
    """EME-model the paper 50/50 and the designer 75/25 SOI couplers."""
    import gdsfactory as gf

    import meow as mw
    from examples.papers import _resolution as res

    gf.gpdk.PDK.activate()
    out = FIGDIR / "mao2019_eme"
    out.mkdir(parents=True, exist_ok=True)
    stack = m.SOIStack()
    kw = {
        "num_cells": res.num_cells(low=14, medium=28, high=56),
        "num_modes": res.num_modes(low=4, medium=6, high=10),
        "fine_res": res.pick(low=0.04, medium=0.025, high=0.018),
        "spec_npts": res.pick(low=5, medium=9, high=17),
    }
    device_res = res.pick(low=0.03, medium=0.02, high=0.015)
    summaries = {}
    # paper-optimized 50/50 coupler (dw_out = 0)
    dvd, prm = _dividers(stack, 0.0)
    summaries["paper_50_50"] = run_design(
        md.adc_component(stack, 0.0), "paper_50_50", out,
        dividers=dvd, params=prm, **kw
    )
    # designer 75/25 coupler
    dw, *_ = md.design_dw_out(0.75, stack, res=device_res, npts=kw["spec_npts"])
    dvd, prm = _dividers(stack, dw)
    summaries["designer_75_25"] = run_design(
        md.adc_component(stack, dw), "designer_75_25", out,
        dividers=dvd, params=prm, **kw
    )
    mw.save_summary(out / "summary",
                    {f"{k}_{kk}": vv for k, s in summaries.items()
                     for kk, vv in s.items() if kk != "label"})
    return {"out_dir": str(out), "summaries": summaries,
            "files": sorted(p.name for p in out.glob("*.png"))}


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
