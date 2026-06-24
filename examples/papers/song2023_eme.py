"""Full EME model of the Song 2023 TFLN polarization rotator taper.

The PRS's enabling element is the partially-etched x-cut TFLN ridge that widens
through the TM0/TE1 hybridization, adiabatically converting an input TM0 into
TE1. This builds an EME model of that widening ridge (a single tapered-rib Prism,
so the waveguide only *widens* and stays centred - well suited to straight-cell
EME), for the paper's 300 nm platform and a designer 500 nm platform, and saves:

- the **polarization-conversion spectrum** in dB over an octave about 1.55 um:
  the TE-polarized output fraction (TM0->TE1 conversion) and the residual
  TM-polarized output, for the TM0 input;
- the **propagating |E| field** for the TM0 input at 1.55 um (the rotation);
- a **section-annotated layout** (taper start/cutoff widths, hybridization).

Run with ``python -m examples.papers.song2023_eme``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from examples.papers import _eme_model as em
from examples.papers import song2023 as s

FIGDIR = s.FIGDIR
LAYER_WG = (1, 0)
CENTER = 1.55


def tapered_ridge_structures(
    platform: Any, wl: float, w_start: float, w_end: float, length: float,
    *, npts: int = 120, x_span: tuple[float, float] = (-4.0, 4.0),
) -> list[Any]:
    """A single linearly-widening rib Prism (z in [0, length]) + slab/clad."""
    import meow as mw
    from examples.papers.kwolek_designer import _background

    run = platform.sidewall_run
    zs = np.linspace(0.0, length, npts)
    w = w_start + (w_end - w_start) * (zs / length)
    half = (w + 2 * run) / 2
    lower = [(float(z), -float(h)) for z, h in zip(zs, half, strict=True)]
    upper = [(float(z), float(h)) for z, h in zip(zs[::-1], half[::-1], strict=True)]
    rib = mw.Structure(
        material=platform.core(wl),
        geometry=mw.Prism(
            poly=np.array([*lower, *upper]),
            h_min=platform.slab_thickness, h_max=platform.core_thickness,
            axis="y", sidewall_angle=platform.sidewall_deg,
        ),
        mesh_order=5,
    )
    return [rib, *_background(platform, wl, length, x_span)]


def device_cells(
    platform: Any, w_start: float, w_end: float, length: float,
    *, wl: float, num_cells: int, fine_res: float,
) -> list[Any]:
    import meow as mw

    structs = tapered_ridge_structures(platform, wl, w_start, w_end, length)
    h = platform.core_thickness
    mesh = mw.Mesh2D(
        x=np.arange(-4.0, 4.0 + fine_res / 2, fine_res),
        y=np.arange(-platform.box_thickness, h + 0.8 + fine_res / 2, fine_res),
    )
    lengths = np.full(num_cells, length / num_cells)
    return mw.create_cells(structs, mesh, lengths, z_min=0.0)


def _solve(cells: list[Any], wl: float, num_modes: int) -> list[list[Any]]:
    import meow as mw

    env = mw.Environment(wl=wl, T=25.0)
    css = [mw.CrossSection.from_cell(cell=c, env=env) for c in cells]
    return [mw.compute_modes(cs, num_modes=num_modes) for cs in css]


def _tm0_input_index(modes_in: list[Any]) -> int:
    """Highest-neff TM (te_fraction<0.5) input mode = the TM0 to be rotated."""
    import meow as mw

    tms = [i for i, mo in enumerate(modes_in) if float(mw.te_fraction(mo)) < 0.5]
    return max(tms, key=lambda i: np.real(modes_in[i].neff)) if tms else 0


def conversion_powers(
    platform: Any, w_start: float, w_end: float, length: float, wl: float,
    *, num_cells: int, num_modes: int, fine_res: float,
) -> tuple[float, float]:
    """(TE-pol, TM-pol) output power for the TM0 input (TM0->TE conversion)."""
    import meow as mw

    cells = device_cells(platform, w_start, w_end, length, wl=wl,
                         num_cells=num_cells, fine_res=fine_res)
    modes = _solve(cells, wl, num_modes)
    s_mat, pm = mw.compute_s_matrix(modes, cells=cells)
    s_mat = np.asarray(s_mat)
    in_idx = _tm0_input_index(modes[0])
    te = tm = 0.0
    for i, mo in enumerate(modes[-1]):
        p = float(np.abs(s_mat[pm[f"right@{i}"], pm[f"left@{in_idx}"]]) ** 2)
        if float(mw.te_fraction(mo)) >= 0.5:
            te += p
        else:
            tm += p
    return te, tm


def run_design(
    platform: Any, w0: float, label: str, out: Path, *, length: float,
    num_cells: int, num_modes: int, fine_res: float, spec_npts: int,
) -> dict[str, Any]:
    """EME-model one rotator taper; save conversion spectrum, propagation, layout."""
    import meow as mw

    w_start, w_end = w0 - 0.3, w0 + 0.3
    comp = _layout(w_start, w_end, length)
    em.plot_annotated_layout(
        comp, out / f"{label}_layout.png",
        title=f"Song 2023 TFLN rotator taper - {label}",
        layer_styles={LAYER_WG: ("TFLN ridge", "#9467bd")},
        dividers=[(length / 2, f"hyb w0={w0:.2f} um")],
        params=f"w: {w_start:.2f}->{w_end:.2f} um, L={length:.0f} um, "
               f"{platform.name} @ {CENTER * 1000:.0f} nm",
        ylim=(-3, 3),
    )
    wls = em.octave_wls(CENTER, spec_npts)
    te, tm = [], []
    for wl in wls:
        t, u = conversion_powers(platform, w_start, w_end, length, float(wl),
                                 num_cells=num_cells, num_modes=num_modes,
                                 fine_res=fine_res)
        te.append(t)
        tm.append(u)
    te, tm = np.array(te), np.array(tm)
    tot = te + tm
    series = {"TM0->TE conversion [dB]": 10 * np.log10(np.maximum(te / tot, 1e-4)),
              "residual TM [dB]": 10 * np.log10(np.maximum(tm / tot, 1e-4))}
    em.plot_spectrum(wls, series, out / f"{label}_spectrum.png",
                     title=f"TM0->TE1 conversion - {label}", center_nm=CENTER * 1000,
                     ylabel="normalized output [dB]")
    mw.save_table(out / f"{label}_spectrum",
                  {"wl_nm": wls * 1000, "te_frac": te / tot, "tm_frac": tm / tot,
                   "total_transmission": tot})
    cells = device_cells(platform, w_start, w_end, length, wl=CENTER,
                        num_cells=num_cells, fine_res=fine_res)
    modes = _solve(cells, CENTER, num_modes)
    idx = _tm0_input_index(modes[0])
    field, x_trans = mw.propagate_modes(modes, cells, excite_mode_l=idx,
                                        y=platform.core_thickness / 2, num_z=400)
    em.plot_propagation(np.abs(np.asarray(field)), np.asarray(x_trans), length,
                        out / f"{label}_propagation_TM0.png",
                        title=f"|E| TM0 input @ 1550 nm (rotation) - {label}",
                        ylim=(-3, 3))
    return {"label": label,
            "conversion_pct_1550": float(100 * np.interp(CENTER, wls, te / tot))}


def _layout(w_start: float, w_end: float, length: float) -> Any:
    """A top-view gdsfactory ridge taper (for the annotated-layout plot only)."""
    import gdsfactory as gf

    c = gf.Component()
    zs = np.linspace(0.0, length, 80)
    w = w_start + (w_end - w_start) * (zs / length)
    top = np.column_stack([zs, w / 2])
    bot = np.column_stack([zs[::-1], (-w / 2)[::-1]])
    c.add_polygon(np.vstack([top, bot]), layer=LAYER_WG)
    return c


def main() -> dict[str, Any]:
    """EME-model the paper (300 nm) and designer (500 nm) TFLN rotator tapers."""
    import gdsfactory as gf

    import meow as mw
    from examples.papers import _resolution as res

    gf.gpdk.PDK.activate()
    out = FIGDIR / "song2023_eme"
    out.mkdir(parents=True, exist_ok=True)
    kw = {
        "num_cells": res.num_cells(low=20, medium=40, high=80),
        "num_modes": res.num_modes(low=4, medium=6, high=8),
        "fine_res": res.pick(low=0.05, medium=0.035, high=0.025),
        "spec_npts": res.pick(low=5, medium=9, high=15),
        "length": 300.0,
    }
    scan_res = res.pick(low=0.05, medium=0.035, high=0.025)
    scan_n = res.pick(low=8, medium=13, high=21)
    summaries = {}
    for label, platform in (("paper_300nm", s.song_platform()),
                            ("designer_500nm", s.tfln_platform(0.50, etch_depth=0.20,
                                                               sidewall_deg=15.0))):
        widths = np.linspace(0.8, 1.9, scan_n)
        _, fracs = s.hybridization_scan(platform, widths, CENTER, res=scan_res)
        w0 = s.hybridization_width(widths, fracs)
        summaries[label] = run_design(platform, w0, label, out, **kw)
    mw.save_summary(out / "summary",
                    {f"{k}_{kk}": vv for k, sm in summaries.items()
                     for kk, vv in sm.items() if kk != "label"})
    return {"out_dir": str(out), "summaries": summaries,
            "files": sorted(p.name for p in out.glob("*.png"))}


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
