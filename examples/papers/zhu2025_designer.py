"""Trident edge-coupler designer applying the Zhu 2025 workflow to new specs.

Companion to :mod:`examples.papers.zhu2025`. Where the analysis module
reproduces the paper's overlap figures, this **designer** applies the same
workflow to a *new* target specification - a target fiber mode-field diameter,
wavelength and SiN platform - by tuning the facet geometry so the collective
trident mode matches the target fiber, then emits the analogous figures and a
top-view GDS of the tapering trident.

Design workflow (the paper's, made concrete with meow FDE):

1. For a target fiber MFD and wavelength, scan the trident horizontal pitch
   ``G1`` (and vertical spacing ``H1``) and, at each, solve the collective TE
   facet mode and its overlap with the target Gaussian (eq 1).
2. Pick the geometry maximizing the overlap; report the optimized facet, the
   TE/TM overlap and the overlap-vs-wavelength curve.
3. Lay out the three SiN levels of the tapering trident (facet -> single output
   waveguide) on separate GDS layers and write the GDS.

Default new specs: a 1310 nm O-band design to a smaller-MFD (6 um) lensed fiber
on a 0.2 um SiN platform.

Run with ``python -m examples.papers.zhu2025_designer``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from examples.papers import zhu2025 as z

FIGDIR = z.FIGDIR
LAYER_MID = (1, 0)
LAYER_TOP = (2, 0)
LAYER_BOT = (3, 0)


# =====================================================================
# User-defined multi-layer (trident) edge-coupler stack
# =====================================================================


@dataclass
class WaveguideLayer:
    """One core layer of a vertically-stacked edge coupler.

    Args:
        name: label for plots / GDS.
        material: meow material (or refractive index) of the core.
        thickness: core thickness [um].
        facet_width: full lateral width at the chip facet (the wide end) [um].
        min_tip_width: width of the inverse taper tip where the layer begins
            (its narrow end, deeper in the chip) [um].
        gds_layer: GDS layer/datatype for this level.
        color: render colour (muted red / green / blue by default).
    """

    name: str
    material: Any
    thickness: float
    facet_width: float
    min_tip_width: float
    gds_layer: tuple[int, int]
    color: str


@dataclass
class LayerStack:
    """A vertically-stacked multi-layer edge-coupler platform.

    The cores are listed bottom-to-top; ``spacings`` are the oxide gaps between
    adjacent cores (the inter-core cladding thicknesses), and ``bottom_clad`` /
    ``top_clad`` are the buried-oxide and top-cladding thicknesses. ``main_index``
    is the layer that continues, alone, as the single output waveguide; the input
    is a single-layer waveguide of ``standard_width`` that tapers up to the main
    layer's width before the 3D trident (the assist layers) tapers into place.
    """

    layers: list[WaveguideLayer]
    spacings: list[float]
    bottom_clad: float = 0.5
    top_clad: float = 4.5
    clad_material: Any = None
    main_index: int = 1
    standard_width: float = 1.0
    l_input: float = 20.0  # single-layer straight input [um]
    l_intaper: float = 120.0  # standard_width -> main facet width taper [um]
    l_trident: float = 200.0  # 3D trident (assist layers grow to facet) [um]

    def y_centers(self) -> list[float]:
        """Vertical centre of each core [um], stack centred about zero."""
        if not self.spacings:
            return [0.0]
        edges = [0.0]
        for s in self.spacings:
            edges.append(edges[-1] + s)
        centre = sum(edges) / len(edges)
        return [e - centre for e in edges]


def default_si3n4_stack() -> LayerStack:
    """Default Si3N4 triple-layer trident stack (200/100/40 nm cores).

    Facet widths 2 / 5 / 6 um for the 200 / 100 / 40 nm cores; the thick 200 nm
    layer is the central output guide (drawn red), flanked by the wider, thinner
    assist layers (green, blue) that expand the facet mode for the fibre. Layer
    spacings and oxide claddings follow the paper's trident levels.
    """
    import meow as mw

    sin = mw.silicon_nitride
    main = WaveguideLayer("core-200nm", sin, 0.200, 2.0, 0.15, LAYER_MID, "C3")
    a100 = WaveguideLayer("assist-100nm", sin, 0.100, 5.0, 0.15, LAYER_BOT, "C2")
    a040 = WaveguideLayer("assist-40nm", sin, 0.040, 6.0, 0.18, LAYER_TOP, "C0")
    return LayerStack(
        layers=[a100, main, a040],  # bottom -> top; main is central
        spacings=[2.2, 2.2],
        bottom_clad=0.5,
        top_clad=4.5,
        clad_material=mw.silicon_oxide,
        main_index=1,
        standard_width=1.0,
    )


def _layer_profile(
    stack: LayerStack, layer: WaveguideLayer, npts: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """``(z, width, z_start)`` for one layer (main vs assist taper schedule)."""
    z_ts = stack.l_input + stack.l_intaper
    ztot = z_ts + stack.l_trident
    if layer is stack.layers[stack.main_index]:
        z = np.linspace(0.0, ztot, npts)
        ramp = np.clip((z - stack.l_input) / max(stack.l_intaper, 1e-9), 0.0, 1.0)
        w = stack.standard_width + (layer.facet_width - stack.standard_width) * ramp
        return z, w, 0.0
    z = np.linspace(z_ts, ztot, max(npts // 2, 8))
    frac = (z - z_ts) / max(stack.l_trident, 1e-9)
    w = layer.min_tip_width + (layer.facet_width - layer.min_tip_width) * frac
    return z, w, z_ts


def stack_render_layers(stack: LayerStack, *, npts: int = 200) -> list[Any]:
    """Build :class:`_plot3d.StackLayer` ribbons for 2.5D/3D rendering."""
    from examples.papers._plot3d import StackLayer

    ycs = stack.y_centers()
    out = []
    for lay, yc in zip(stack.layers, ycs, strict=True):
        zc, wc, z0 = _layer_profile(stack, lay, npts)
        out.append(
            StackLayer(
                name=f"{lay.name} ({lay.thickness * 1e3:.0f} nm)",
                z=zc, width=wc, y_center=yc, thickness=lay.thickness,
                color=lay.color, z_start=z0,
            )
        )
    return out


def stack_gds(stack: LayerStack, *, npts: int = 200) -> Any:
    """Top-view GDS of the stack (one mask layer per core; single-layer input)."""
    import gdsfactory as gf

    c = gf.Component()
    ztot = stack.l_input + stack.l_intaper + stack.l_trident
    for lay in stack.layers:
        zc, wc, _ = _layer_profile(stack, lay, npts)
        top = np.column_stack([zc, wc / 2])
        bot = np.column_stack([zc[::-1], (-wc / 2)[::-1]])
        c.add_polygon(np.vstack([top, bot]), layer=lay.gds_layer)
    main = stack.layers[stack.main_index]
    c.add_port("input", center=(0.0, 0.0), width=stack.standard_width,
               orientation=180, layer=main.gds_layer)
    c.add_port("facet", center=(ztot, 0.0), width=main.facet_width,
               orientation=0, layer=main.gds_layer)
    return c


def render_stack(stack: LayerStack, name: str, out_dir: Path) -> dict[str, str]:
    """Save the 2.5D and 3D perspective RGB renderings of a stack."""
    import matplotlib.pyplot as plt

    from examples.papers._plot3d import plot_stack_2p5d, plot_stack_3d

    out_dir.mkdir(parents=True, exist_ok=True)
    layers = stack_render_layers(stack)

    fig = plt.figure(figsize=(13, 5))
    ax25 = fig.add_subplot(1, 2, 1)
    plot_stack_2p5d(ax25, layers)
    ax25.set_title("2.5D (top view, layers lifted)")
    ax25.legend(loc="upper left", fontsize=7)
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    plot_stack_3d(ax3d, layers)
    ax3d.set_title("3D perspective (vertical not to scale)")
    fig.suptitle(
        f"{name}: stacked trident edge coupler "
        f"({'/'.join(f'{L.thickness * 1e3:.0f}' for L in stack.layers)} nm cores)"
    )
    fig.tight_layout()
    p25 = out_dir / f"{name}_geometry_2p5d_3d.png"
    fig.savefig(p25, dpi=150)
    plt.close(fig)
    return {"render": str(p25)}


def design_stack(stack: LayerStack, name: str, out_dir: Path) -> dict[str, Any]:
    """Render + write GDS + summary for one stack design version."""
    import meow as mw

    out_dir.mkdir(parents=True, exist_ok=True)
    files = render_stack(stack, name, out_dir)
    gds = stack_gds(stack)
    gds_path = out_dir / f"{name}.gds"
    gds.write_gds(str(gds_path))
    ztot = stack.l_input + stack.l_intaper + stack.l_trident
    summary = {
        "name": name,
        "n_layers": len(stack.layers),
        "core_thicknesses_nm": [L.thickness * 1e3 for L in stack.layers],
        "facet_widths_um": [L.facet_width for L in stack.layers],
        "min_tip_widths_um": [L.min_tip_width for L in stack.layers],
        "layer_spacings_um": stack.spacings,
        "bottom_clad_um": stack.bottom_clad,
        "top_clad_um": stack.top_clad,
        "standard_input_width_um": stack.standard_width,
        "total_length_um": ztot,
        "main_layer": stack.layers[stack.main_index].name,
    }
    mw.save_summary(out_dir / f"{name}_summary", summary)
    return {"gds": str(gds_path), "summary": summary, **files}


def default_stack_versions() -> dict[str, LayerStack]:
    """The design versions emitted by :func:`main_stacks` (extend freely)."""
    si3n4 = default_si3n4_stack()
    # a compact variant: tighter inter-layer spacing and shorter trident
    compact = replace(si3n4, spacings=[1.2, 1.2], l_trident=140.0,
                      l_intaper=90.0)
    return {"si3n4_triple_200_100_40": si3n4, "si3n4_triple_compact": compact}


def main_stacks(out_dir: Path | None = None) -> dict[str, Any]:
    """Design + render + write GDS for every stack version; return the manifest."""
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    out_dir = out_dir or (FIGDIR / "zhu2025_stack_designer")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, stack in default_stack_versions().items():
        results[name] = design_stack(stack, name, out_dir)
    return {"out_dir": str(out_dir), "versions": results}


def optimize_facet(
    *,
    target_mfd: float,
    wl: float,
    base: z.TridentFacet | None = None,
    pitches: np.ndarray | None = None,
) -> tuple[z.TridentFacet, float, np.ndarray, np.ndarray]:
    """Scan the trident pitch ``G1`` to best match a target fiber MFD.

    Returns ``(best_facet, best_eta, pitches, etas)``.
    """
    base = base or z.TridentFacet()
    if pitches is None:
        pitches = np.linspace(1.0, 3.0, 6)
    etas = []
    for g1 in pitches:
        facet = replace(base, g1=float(g1), h1=float(g1) + 0.1)
        modes = z.facet_modes(facet, wl, num_modes=6)
        te = z._pick_polarization(modes, "te")
        etas.append(z.fiber_overlap(te, "te", mfd=target_mfd))
    etas = np.asarray(etas)
    i = int(np.argmax(etas))
    best = replace(base, g1=float(pitches[i]), h1=float(pitches[i]) + 0.1)
    return best, float(etas[i]), pitches, etas


def trident_gds(
    facet: z.TridentFacet,
    *,
    l1: float = 165.0,
    l2: float = 130.0,
    w_out: float = 0.8,
) -> Any:
    """Top-view GDS of the tapering trident on three SiN-level layers.

    The trident arms (mid layer) and the assisted top/bottom waveguides taper
    from the facet over ``L1`` and then merge into a single output waveguide of
    width ``w_out`` over ``L2`` (the paper's two mode-transformation stages).
    """
    import gdsfactory as gf

    c = gf.Component()
    ztot = l1 + l2

    def taper(layer: tuple, y0: float, w_facet: float, y_out: float) -> None:
        # facet bar (z=0) -> converge to centre output over L1+L2
        zs = np.linspace(0.0, ztot, 60)
        yc = y0 + (y_out - y0) * (zs / ztot)
        w = w_facet + (w_out - w_facet) * np.clip((zs - l1) / max(l2, 1e-9), 0, 1)
        top = np.column_stack([zs, yc + w / 2])
        bot = np.column_stack([zs[::-1], (yc - w / 2)[::-1]])
        c.add_polygon(np.vstack([top, bot]), layer=layer)

    # mid-layer trident: left + right arms merge to centre; top/bottom assisted
    taper(LAYER_MID, -facet.g1, facet.w3, 0.0)
    taper(LAYER_MID, +facet.g1, facet.w3, 0.0)
    taper(LAYER_TOP, 0.0, facet.w1, 0.0)
    taper(LAYER_BOT, 0.0, facet.w1, 0.0)
    c.add_port("facet", center=(0.0, 0.0), width=2 * facet.g1, orientation=180,
               layer=LAYER_MID)
    c.add_port("out", center=(ztot, 0.0), width=w_out, orientation=0, layer=LAYER_MID)
    return c


def plot_pitch_scan(
    pitches: np.ndarray, etas: np.ndarray, best_g1: float, path: Path
) -> None:
    """Overlap vs trident pitch, with the chosen design point."""
    plt = z._use_agg()
    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.plot(pitches, etas * 100, "C0o-")
    ax.axvline(best_g1, color="k", ls="--", label=f"design G1={best_g1:.2f} um")
    ax.set_xlabel(r"trident pitch $G_1$ [um]")
    ax.set_ylabel("TE overlap with target fiber [%]")
    ax.grid(visible=True, alpha=0.3)
    ax.legend()
    ax.set_title("Designer: facet-mode / fiber overlap vs pitch")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> dict[str, Any]:
    """Design a trident edge coupler for a new target fiber/platform."""
    import gdsfactory as gf

    import meow as mw

    gf.gpdk.PDK.activate()
    out = FIGDIR / "zhu2025_designer"
    out.mkdir(parents=True, exist_ok=True)
    target_mfd, wl = 8.0, 1.31  # reduced-MFD (e.g. PM/UHNA) fiber, O-band
    best, eta, pitches, etas = optimize_facet(target_mfd=target_mfd, wl=wl)
    plot_pitch_scan(pitches, etas, best.g1, out / "pitch_scan.png")
    z.plot_facet_modes(best, wl, out / "facet_modes.png")
    wls = np.linspace(1.26, 1.36, 5)
    te, tm = z.overlap_vs_wavelength(best, wls, mfd=target_mfd)
    z.plot_overlap_vs_wavelength(wls, te, tm, out / "overlap_vs_wavelength.png")
    comp = trident_gds(best)
    comp.write_gds(str(out / "trident_edge_coupler.gds"))
    summary = {
        "target_mfd_um": target_mfd, "wl_nm": wl * 1000,
        "best_g1_um": best.g1, "best_h1_um": best.h1,
        "eta_te": round(eta, 4), "eta_te_band_min": float(te.min()),
    }
    mw.save_summary(out / "summary", summary)
    return {"out_dir": str(out), "summary": summary,
            "files": sorted(p.name for p in out.glob("*"))}


if __name__ == "__main__":
    import json

    # New: user-defined multi-layer stack designer (geometry + 2.5D/3D + GDS).
    print(json.dumps(main_stacks(), indent=2, default=str))
