"""Full EME model of the Zhu 2025 3D trident SiN edge coupler.

Builds a genuine eigenmode-expansion model of the tapering trident (the three
SiN levels become three GDS layers extruded to different vertical ranges; the
lateral convergence of the trident arms and the growth of the central output
waveguide are the top-view geometry), then for both the paper-optimized facet
and a designer facet it computes and saves:

- a **dense transmission spectrum** of the key S-matrix elements (facet ->
  output-waveguide) in dB over an octave around 1.55 um, for TE and TM input;
- the **propagating |E| field** along the device for the TE and TM facet inputs
  at 1.55 um;
- a **pretty annotated layout** with section labels (L1 mode-transform, L2
  merge) and parameter values.

This is the meow EME analogue of the paper's 3D-FDTD propagation study; it runs
at a feasible mesh/cell resolution (the collective facet mode is ~10 um, so the
mesh is large) - increase ``MEOW_EXAMPLE_RES`` for convergence.

Run with ``python -m examples.papers.zhu2025_eme``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from examples.papers import zhu2025 as z

FIGDIR = z.FIGDIR
# three SiN levels -> three GDS layers, extruded to three vertical ranges
LAYER_MID = (1, 0)
LAYER_TOP = (2, 0)
LAYER_BOT = (3, 0)


def trident_taper_component(
    facet: z.TridentFacet,
    *,
    l1: float = 165.0,
    l2: float = 130.0,
    w2_start: float = 0.346,
    w_out: float = 0.8,
    tip: float = 0.10,
    npts: int = 160,
) -> Any:
    """Top-view layout of the tapering trident on three SiN-level GDS layers.

    Over ``L1`` the trident arms (mid layer) converge laterally to the centre
    and the assisted top/bottom guides taper down, while the central output
    waveguide grows from ``w2_start``; over ``L2`` the output waveguide widens to
    ``w_out``. gdsfactory ``x`` is propagation, ``y`` is lateral.
    """
    import gdsfactory as gf

    ltot = l1 + l2
    zs = np.linspace(0.0, ltot, npts)

    def lerp(a: float, b: float, t: np.ndarray) -> np.ndarray:
        return a + (b - a) * np.clip(t, 0, 1)

    t1 = zs / l1  # progresses 0->1 over region I
    t_all = zs / ltot

    def strip(lat: np.ndarray, width: np.ndarray) -> np.ndarray:
        top = np.column_stack([zs, lat + width / 2])
        bot = np.column_stack([zs[::-1], (lat - width / 2)[::-1]])
        return np.vstack([top, bot])

    c = gf.Component()
    # middle layer: left + right trident arms converge to centre; centre output
    c.add_polygon(strip(lerp(-facet.g1, 0.0, t1), lerp(facet.w3, tip, t1)), LAYER_MID)
    c.add_polygon(strip(lerp(+facet.g1, 0.0, t1), lerp(facet.w3, tip, t1)), LAYER_MID)
    c.add_polygon(strip(np.zeros_like(zs), lerp(w2_start, w_out, t_all)), LAYER_MID)
    # assisted top/bottom layers: centred, taper down to a tip over region I
    c.add_polygon(strip(np.zeros_like(zs), lerp(facet.w1, tip, t1)), LAYER_TOP)
    c.add_polygon(strip(np.zeros_like(zs), lerp(facet.w1, tip, t1)), LAYER_BOT)
    c.add_port("facet", center=(0.0, 0.0), width=2 * facet.g1, orientation=180,
               layer=LAYER_MID)
    c.add_port("out", center=(ltot, 0.0), width=w_out, orientation=0, layer=LAYER_MID)
    c.info["l1"] = l1
    c.info["l2"] = l2
    return c


def _extrude(component: Any, facet: z.TridentFacet) -> list[Any]:
    """Extrude the three GDS layers to their vertical SiN levels + oxide."""
    import meow as mw

    h, h1 = facet.h, facet.h1
    sin = mw.silicon_nitride
    rules = {
        LAYER_BOT: [mw.GdsExtrusionRule(material=sin, h_min=0.0, h_max=h)],
        LAYER_MID: [
            mw.GdsExtrusionRule(material=mw.silicon_nitride, h_min=h1, h_max=h1 + h)
        ],
        LAYER_TOP: [
            mw.GdsExtrusionRule(
                material=mw.silicon_nitride, h_min=2 * h1, h_max=2 * h1 + h
            )
        ],
    }
    structs = mw.extrude_gds(component, rules)
    oxide = mw.Structure(
        material=mw.silicon_oxide,
        geometry=mw.Box(
            x_min=-9.0, x_max=9.0, y_min=-2.0, y_max=2 * h1 + h + 2.0,
            z_min=0.0, z_max=float(component.xmax),
        ),
        mesh_order=10,
    )
    return [*structs, oxide]


def _mesh(facet: z.TridentFacet, *, fine_res: float) -> Any:
    """Graded mesh: fine over the vertical SiN stack, coarse in the cladding."""
    import meow as mw

    top = 2 * facet.h1 + facet.h
    x = z._graded_axis(3.5, fine_res, 9.0, 0.3)
    yfine = np.arange(-0.4, top + 0.4 + fine_res / 2, fine_res)
    ylo = np.arange(-2.0, -0.4, 0.3)
    yhi = np.arange(top + 0.7, top + 2.0 + 0.3, 0.3)
    y = np.concatenate([ylo, yfine, yhi])
    return mw.Mesh2D(x=x, y=y)


def mid_y(facet: z.TridentFacet) -> float:
    """Vertical centre of the middle SiN layer (the output-waveguide level)."""
    return facet.h1 + facet.h / 2


def device_cells(
    component: Any, facet: z.TridentFacet, *, num_cells: int, fine_res: float
) -> list[Any]:
    """Slice the trident taper into ``num_cells`` EME cells."""
    import meow as mw

    structs = _extrude(component, facet)
    length = float(component.xmax)
    lengths = np.full(num_cells, length / num_cells)
    return mw.create_cells(structs, _mesh(facet, fine_res=fine_res), lengths, z_min=0.0)


# ==========================================================================
# EME: spectrum + field propagation
# ==========================================================================
def _solve_modes(cells: list[Any], wl: float, num_modes: int) -> list[list[Any]]:
    import meow as mw

    env = mw.Environment(wl=wl, T=25.0)
    css = [mw.CrossSection.from_cell(cell=c, env=env) for c in cells]
    return [mw.compute_modes(cs, num_modes=num_modes) for cs in css]


def _pol_indices(modes: list[Any]) -> dict[str, int]:
    """Index of the highest-neff TE and TM mode in a mode set."""
    import meow as mw

    te = max(
        (i for i, m in enumerate(modes) if float(mw.te_fraction(m)) >= 0.5),
        key=lambda i: np.real(modes[i].neff), default=0,
    )
    tm = max(
        (i for i, m in enumerate(modes) if float(mw.te_fraction(m)) < 0.5),
        key=lambda i: np.real(modes[i].neff), default=min(1, len(modes) - 1),
    )
    return {"TE": int(te), "TM": int(tm)}


def _input_mode_indices(modes_in: list[Any]) -> dict[str, int]:
    """Indices of the TE and TM collective facet input modes."""
    return _pol_indices(modes_in)


def transmission_db(
    component: Any,
    facet: z.TridentFacet,
    wls: np.ndarray,
    *,
    num_cells: int,
    num_modes: int,
    fine_res: float,
) -> dict[str, np.ndarray]:
    """Facet->output transmission (dB) vs wavelength for TE and TM facet input."""
    import meow as mw

    structs = _extrude(component, facet)
    length = float(component.xmax)
    lengths = np.full(num_cells, length / num_cells)
    mesh = _mesh(facet, fine_res=fine_res)
    te_db, tm_db = [], []
    for wl in wls:
        env = mw.Environment(wl=float(wl), T=25.0)
        cells = mw.create_cells(structs, mesh, lengths, z_min=0.0)
        modes = [
            mw.compute_modes(mw.CrossSection.from_cell(cell=c, env=env), num_modes)
            for c in cells
        ]
        s, pm = mw.compute_s_matrix(modes, cells=cells)
        s = np.asarray(s)
        in_idx = _pol_indices(modes[0])
        out_idx = _pol_indices(modes[-1])
        # each input polarization couples to the output mode of the same pol
        for key, store in (("TE", te_db), ("TM", tm_db)):
            amp = s[pm[f"right@{out_idx[key]}"], pm[f"left@{in_idx[key]}"]]
            store.append(20 * np.log10(max(abs(complex(amp)), 1e-6)))
    return {"TE": np.asarray(te_db), "TM": np.asarray(tm_db)}


def propagate(
    component: Any,
    facet: z.TridentFacet,
    wl: float,
    pol: str,
    *,
    num_cells: int,
    num_modes: int,
    fine_res: float,
    num_z: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """|E|(x, z) along the device for the TE/TM facet input (mid-layer y-plane)."""
    import meow as mw

    cells = device_cells(component, facet, num_cells=num_cells, fine_res=fine_res)
    modes = _solve_modes(cells, wl, num_modes)
    idx = _input_mode_indices(modes[0])[pol]
    # propagate_modes returns (field[z, x], x_transverse); propagation is axis 0
    field, x_trans = mw.propagate_modes(
        modes, cells, excite_mode_l=idx, y=mid_y(facet), num_z=num_z
    )
    length = float(component.xmax)
    return np.abs(np.asarray(field)), np.asarray(x_trans), length


# ==========================================================================
# plots
# ==========================================================================
def plot_spectrum(
    wls: np.ndarray, db: dict[str, np.ndarray], path: Path, *, title: str
) -> None:
    """Dense facet->output transmission spectrum in dB (TE and TM)."""
    plt = z._use_agg()
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.plot(wls * 1000, db["TE"], "C0-", label=r"TE: $|S_{out,TE}|^2$")
    ax.plot(wls * 1000, db["TM"], "C3-", label=r"TM: $|S_{out,TM}|^2$")
    ax.axvline(1550, color="0.6", ls=":", lw=1)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("transmission [dB]")
    ax.grid(visible=True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_propagation(
    field: np.ndarray,
    x_trans: np.ndarray,
    length: float,
    path: Path,
    *,
    title: str,
) -> None:
    """|E| along the device: propagation z (axis 0) vs transverse x (axis 1)."""
    plt = z._use_agg()
    norm = field / (field.max() or 1.0)
    zprop = np.linspace(0.0, length, field.shape[0])
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.pcolormesh(zprop, x_trans, norm.T, shading="auto", cmap="magma")
    ax.set_xlabel("propagation z [um]")
    ax.set_ylabel("lateral x [um]")
    ax.set_ylim(-6, 6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="|E| (norm.)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_annotated_layout(
    component: Any, facet: z.TridentFacet, path: Path, *, title: str
) -> None:
    """Pretty top-view layout coloured by SiN level, with section + param labels."""
    plt = z._use_agg()
    from matplotlib.collections import PolyCollection

    colors = {LAYER_MID: "#1f77b4", LAYER_TOP: "#2ca02c", LAYER_BOT: "#d62728"}
    names = {LAYER_MID: "mid (trident+output)", LAYER_TOP: "top assist",
             LAYER_BOT: "bottom assist"}
    fig, ax = plt.subplots(figsize=(11, 3.4))
    try:
        dbu = component.layout().dbu
    except AttributeError:
        dbu = None

    def to_verts(p: Any) -> np.ndarray:
        if dbu is not None and hasattr(p, "each_point_hull"):
            return np.asarray([(pt.x * dbu, pt.y * dbu) for pt in p.each_point_hull()])
        return np.asarray(p)

    polys_by_layer = component.get_polygons(by="tuple")
    for layer, polys in polys_by_layer.items():
        key = tuple(layer) if not isinstance(layer, tuple) else layer
        col = colors.get(key, "0.5")
        verts = [to_verts(p) for p in polys]
        ax.add_collection(
            PolyCollection(verts, facecolors=col, edgecolors="k", linewidths=0.2,
                           alpha=0.75, label=names.get(key))
        )
    l1, l2 = component.info["l1"], component.info["l2"]
    ax.axvline(l1, color="0.4", ls="--", lw=1)
    ax.text(l1 / 2, 5.4, f"L1 = {l1:.0f} um\n(mode transform)", ha="center", fontsize=8)
    ax.text(l1 + l2 / 2, 5.4, f"L2 = {l2:.0f} um\n(merge)", ha="center", fontsize=8)
    ax.text(
        2.0, -5.6,
        f"g1={facet.g1} um, h1={facet.h1} um, w1={facet.w1} um, "
        f"w3={facet.w3} um, H={facet.h} um",
        fontsize=8, color="0.3",
    )
    ax.autoscale_view()
    ax.set_ylim(-6.5, 6.5)
    ax.set_xlabel("propagation z [um]")
    ax.set_ylabel("lateral x [um]")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_design(
    facet: z.TridentFacet,
    label: str,
    out: Path,
    *,
    l1: float,
    l2: float,
    w_out: float,
    num_cells: int,
    num_modes: int,
    fine_res: float,
    spec_npts: int,
) -> dict[str, Any]:
    """Build + EME-model one facet design; save spectrum, propagation, layout."""
    import meow as mw

    comp = trident_taper_component(facet, l1=l1, l2=l2, w_out=w_out)
    plot_annotated_layout(comp, facet, out / f"{label}_layout.png",
                          title=f"Zhu 2025 trident edge coupler - {label}")
    wls = np.linspace(1.1, 2.2, spec_npts)  # a full octave (factor 2) around 1.55 um
    db = transmission_db(comp, facet, wls, num_cells=num_cells,
                         num_modes=num_modes, fine_res=fine_res)
    plot_spectrum(wls, db, out / f"{label}_spectrum.png",
                  title=f"Facet->output transmission - {label}")
    mw.save_table(out / f"{label}_spectrum",
                  {"wl_nm": wls * 1000, "te_db": db["TE"], "tm_db": db["TM"]})
    for pol in ("TE", "TM"):
        field, x_trans, length = propagate(comp, facet, 1.55, pol, num_cells=num_cells,
                                           num_modes=num_modes, fine_res=fine_res)
        plot_propagation(field, x_trans, length,
                         out / f"{label}_propagation_{pol}.png",
                         title=f"|E| propagation, {pol} input @ 1550 nm - {label}")
    return {"label": label, "te_db_1550": float(np.interp(1.55, wls, db["TE"])),
            "tm_db_1550": float(np.interp(1.55, wls, db["TM"]))}


def main() -> dict[str, Any]:
    """EME-model the paper-optimized and designer trident edge couplers."""
    import gdsfactory as gf

    import meow as mw
    from examples.papers import _resolution as res
    from examples.papers import zhu2025_designer as zd

    gf.gpdk.PDK.activate()
    out = FIGDIR / "zhu2025_eme"
    out.mkdir(parents=True, exist_ok=True)
    kw = {
        "num_cells": res.num_cells(low=12, medium=24, high=48),
        "num_modes": res.num_modes(low=6, medium=8, high=12),
        "fine_res": res.pick(low=0.10, medium=0.07, high=0.05),
        "spec_npts": res.pick(low=5, medium=9, high=17),
    }
    summaries = {}
    # paper-optimized facet
    summaries["paper"] = run_design(
        z.TridentFacet(), "paper", out, l1=165, l2=130, w_out=0.8, **kw
    )
    # designer facet (the 8 um-target optimum from the designer scan)
    best, _, _, _ = zd.optimize_facet(target_mfd=8.0, wl=1.31)
    summaries["designer"] = run_design(
        best, "designer", out, l1=165, l2=130, w_out=0.8, **kw
    )
    mw.save_summary(out / "summary",
                    {f"{k}_{kk}": vv for k, s in summaries.items()
                     for kk, vv in s.items() if kk != "label"})
    return {"out_dir": str(out), "summaries": summaries,
            "files": sorted(p.name for p in out.glob("*.png"))}


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
