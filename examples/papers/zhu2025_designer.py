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

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from examples.papers import zhu2025 as z

FIGDIR = z.FIGDIR
LAYER_MID = (1, 0)
LAYER_TOP = (2, 0)
LAYER_BOT = (3, 0)


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
        pitches = np.linspace(1.4, 3.0, 5)
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
    target_mfd, wl = 6.0, 1.31  # smaller-MFD lensed fiber, O-band
    best, eta, pitches, etas = optimize_facet(target_mfd=target_mfd, wl=wl)
    plot_pitch_scan(pitches, etas, best.g1, out / "pitch_scan.png")
    z.plot_facet_modes(best, wl, out / "facet_modes.png")
    wls = np.linspace(1.26, 1.36, 5)
    te, tm = z.overlap_vs_wavelength(best, wls)
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

    print(json.dumps(main(), indent=2, default=str))
