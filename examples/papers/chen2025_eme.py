"""Full EME model of the Chen 2025 FAQUAD TFLN polarization splitter-rotator.

Builds an EME model of the PSR's adiabatic mode-evolution waveguide: a fixed
through ridge (``w0 = 1 um``) beside a cross ridge that widens along the FAQUAD
profile ``w1(z)`` (0.1 -> 0.65 um) at a fixed top gap ``g = 0.4 um`` on the
400 nm TFLN platform. For the paper's 300 um device and a compact 200 um
designer device it saves:

- the **mode-routing spectrum** in dB over an octave about 1.55 um: for a TM0
  through-input, the power routed to the **cross** waveguide (the rotated
  TM0->TE conversion that defines the PSR) vs the power left in the **through**
  waveguide, normalized;
- the **propagating |E| field** for the TM0 through-input at 1.55 um;
- a **section-annotated layout** (FAQUAD-tapered cross guide, gap, hybridization).

The FAQUAD profile (vs a linear taper) is the paper's key result: it slows the
widening through the mode anti-crossing.

Run with ``python -m examples.papers.chen2025_eme``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from examples.papers import _eme_model as em
from examples.papers import chen2025 as c

FIGDIR = c.FIGDIR
LAYER_WG = (1, 0)
CENTER = 1.55
W0 = 1.0  # through-waveguide top width [um]
GAP = 0.4  # top gap between through and cross ridges [um]


def _ridge_prism(
    platform: Any, wl: float, z_x_outline: np.ndarray
) -> Any:
    import meow as mw

    return mw.Structure(
        material=platform.core(wl),
        geometry=mw.Prism(
            poly=z_x_outline, h_min=platform.slab_thickness,
            h_max=platform.core_thickness, axis="y",
            sidewall_angle=platform.sidewall_deg,
        ),
        mesh_order=5,
    )


def psr_structures(
    platform: Any, wl: float, zs: np.ndarray, w1: np.ndarray, length: float,
    *, x_span: tuple[float, float] = (-3.0, 4.0),
) -> list[Any]:
    """Through ridge (fixed ``W0``) + FAQUAD-widening cross ridge + slab/clad."""
    from examples.papers.kwolek_designer import _background

    run = platform.sidewall_run
    # through ridge centred at x = -(W0/2 + GAP/2)/?  -> place through at x=0
    cx_thru = 0.0
    half_t = (W0 + 2 * run) / 2
    thru = _ridge_prism(platform, wl, np.array(
        [(0.0, cx_thru - half_t), (length, cx_thru - half_t),
         (length, cx_thru + half_t), (0.0, cx_thru + half_t)]))
    # cross ridge: inner edge fixed at x = W0/2 + GAP; widens outward by w1(z)
    inner = W0 / 2 + GAP
    cx = inner + w1 / 2
    half_c = (w1 + 2 * run) / 2
    lower = [(float(z), float(cc - hc))
             for z, cc, hc in zip(zs, cx, half_c, strict=True)]
    upper = [(float(z), float(cc + hc))
             for z, cc, hc in zip(zs[::-1], cx[::-1], half_c[::-1], strict=True)]
    cross = _ridge_prism(platform, wl, np.array([*lower, *upper]))
    return [thru, cross, *_background(platform, wl, length, x_span)]


def device_cells(
    platform: Any, zs: np.ndarray, w1: np.ndarray, length: float,
    *, wl: float, num_cells: int, fine_res: float,
) -> list[Any]:
    import meow as mw

    structs = psr_structures(platform, wl, zs, w1, length)
    h = platform.core_thickness
    mesh = mw.Mesh2D(
        x=np.arange(-3.0, 4.0 + fine_res / 2, fine_res),
        y=np.arange(-platform.box_thickness, h + 0.8 + fine_res / 2, fine_res),
    )
    lengths = np.full(num_cells, length / num_cells)
    return mw.create_cells(structs, mesh, lengths, z_min=0.0)


def _solve(cells: list[Any], wl: float, num_modes: int) -> list[list[Any]]:
    import meow as mw

    env = mw.Environment(wl=wl, T=25.0)
    css = [mw.CrossSection.from_cell(cell=c_, env=env) for c_ in cells]
    return [mw.compute_modes(cs, num_modes=num_modes) for cs in css]


def _tm0_through_index(modes_in: list[Any]) -> int:
    """Highest-neff TM input mode localized in the through waveguide (centroid<gap)."""
    import meow as mw

    mid = W0 / 2 + GAP / 2
    cands = [i for i, mo in enumerate(modes_in)
             if float(mw.te_fraction(mo)) < 0.5 and em.lateral_centroid(mo) < mid]
    if not cands:
        cands = [i for i, mo in enumerate(modes_in)
                 if float(mw.te_fraction(mo)) < 0.5]
    return max(cands, key=lambda i: np.real(modes_in[i].neff)) if cands else 0


def route_powers(
    platform: Any, zs: np.ndarray, w1: np.ndarray, length: float, wl: float,
    *, num_cells: int, num_modes: int, fine_res: float,
) -> tuple[float, float]:
    """(cross, through) output power for the TM0 through-input."""
    import meow as mw

    cells = device_cells(platform, zs, w1, length, wl=wl,
                         num_cells=num_cells, fine_res=fine_res)
    modes = _solve(cells, wl, num_modes)
    s_mat, pm = mw.compute_s_matrix(modes, cells=cells)
    s_mat = np.asarray(s_mat)
    in_idx = _tm0_through_index(modes[0])
    mid = W0 / 2 + GAP / 2
    cross = through = 0.0
    for i, mo in enumerate(modes[-1]):
        p = float(np.abs(s_mat[pm[f"right@{i}"], pm[f"left@{in_idx}"]]) ** 2)
        if em.lateral_centroid(mo) > mid:
            cross += p
        else:
            through += p
    return cross, through


def run_design(
    platform: Any, length: float, label: str, out: Path,
    *, num_cells: int, num_modes: int, fine_res: float, spec_npts: int,
    scan_res: float, scan_n: int,
) -> dict[str, Any]:
    """EME-model one FAQUAD PSR; save routing spectrum, propagation, layout."""
    import meow as mw

    w1s = np.linspace(0.1, 0.65, scan_n)
    neffs, _ = c.neff_evolution(platform, w1s, CENTER, res=scan_res)
    w_hyb = c.hybridization_point(w1s, neffs)
    zs, w1 = c.faquad_profile(w1s, neffs, length=length)
    zs = np.asarray(zs)
    em.plot_annotated_layout(
        _layout(zs, w1, length), out / f"{label}_layout.png",
        title=f"Chen 2025 FAQUAD TFLN PSR - {label}",
        layer_styles={LAYER_WG: ("TFLN ridges", "#9467bd")},
        dividers=[(float(np.interp(w_hyb, w1, zs)), f"hyb w1={w_hyb:.2f} um")],
        params=f"w0={W0} um, g={GAP} um, w1:0.1->0.65 um, L={length:.0f} um, "
               f"{platform.name}",
        ylim=(-2, 3),
    )
    wls = em.octave_wls(CENTER, spec_npts)
    cross, thru = [], []
    for wl in wls:
        cr, th = route_powers(platform, zs, w1, length, float(wl),
                              num_cells=num_cells, num_modes=num_modes,
                              fine_res=fine_res)
        cross.append(cr)
        thru.append(th)
    cross, thru = np.array(cross), np.array(thru)
    tot = cross + thru
    series = {
        "TM0 -> cross (rotated) [dB]": 10 * np.log10(np.maximum(cross / tot, 1e-4)),
        "residual through [dB]": 10 * np.log10(np.maximum(thru / tot, 1e-4)),
    }
    em.plot_spectrum(wls, series, out / f"{label}_spectrum.png",
                     title=f"TM0 through-input routing - {label}",
                     center_nm=CENTER * 1000, ylabel="normalized output [dB]")
    mw.save_table(out / f"{label}_spectrum",
                  {"wl_nm": wls * 1000, "cross_frac": cross / tot,
                   "through_frac": thru / tot, "total_transmission": tot})
    cells = device_cells(platform, zs, w1, length, wl=CENTER,
                        num_cells=num_cells, fine_res=fine_res)
    modes = _solve(cells, CENTER, num_modes)
    idx = _tm0_through_index(modes[0])
    field, x_trans = mw.propagate_modes(modes, cells, excite_mode_l=idx,
                                        y=platform.core_thickness / 2, num_z=400)
    em.plot_propagation(np.abs(np.asarray(field)), np.asarray(x_trans), length,
                        out / f"{label}_propagation_TM0.png",
                        title=f"|E| TM0 through-input @ 1550 nm - {label}",
                        ylim=(-2, 3))
    return {"label": label,
            "cross_pct_1550": float(100 * np.interp(CENTER, wls, cross / tot))}


def _layout(zs: np.ndarray, w1: np.ndarray, length: float) -> Any:
    """Top-view gdsfactory layout: straight through + FAQUAD cross ridge."""
    import gdsfactory as gf

    comp = gf.Component()
    comp.add_polygon(np.array(
        [(0.0, -W0 / 2), (length, -W0 / 2), (length, W0 / 2), (0.0, W0 / 2)]),
        layer=LAYER_WG)
    inner = W0 / 2 + GAP
    top = np.column_stack([zs, inner + w1])
    bot = np.column_stack([zs[::-1], np.full_like(zs, inner)[::-1]])
    comp.add_polygon(np.vstack([top, bot]), layer=LAYER_WG)
    return comp


def main() -> dict[str, Any]:
    """EME-model the paper 300 um and a compact 200 um FAQUAD TFLN PSR."""
    import gdsfactory as gf

    import meow as mw
    from examples.papers import _resolution as res

    gf.gpdk.PDK.activate()
    out = FIGDIR / "chen2025_eme"
    out.mkdir(parents=True, exist_ok=True)
    platform = c.chen_platform()
    kw = {
        "num_cells": res.num_cells(low=20, medium=40, high=80),
        "num_modes": res.num_modes(low=4, medium=6, high=10),
        "fine_res": res.pick(low=0.05, medium=0.035, high=0.025),
        "spec_npts": res.pick(low=5, medium=9, high=15),
        "scan_res": res.pick(low=0.05, medium=0.035, high=0.025),
        "scan_n": res.pick(low=8, medium=13, high=21),
    }
    summaries = {
        "paper_300um": run_design(platform, 300.0, "paper_300um", out, **kw),
        "designer_200um": run_design(platform, 200.0, "designer_200um", out, **kw),
    }
    mw.save_summary(out / "summary",
                    {f"{k}_{kk}": vv for k, sm in summaries.items()
                     for kk, vv in sm.items() if kk != "label"})
    return {"out_dir": str(out), "summaries": summaries,
            "files": sorted(p.name for p in out.glob("*.png"))}


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
