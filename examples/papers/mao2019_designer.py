"""Adiabatic-coupler splitting-ratio designer (Mao 2019 workflow, new specs).

Companion to :mod:`examples.papers.mao2019`. Given a *target* splitting ratio on
a chosen SOI stack, this designer inverts the paper's semi-analytical
SR-vs-output-width-difference relation to find the required ``dw_out``, lays out
the five-region adiabatic coupler at that width-difference and writes its GDS,
plus the SR curve (with the design point marked) and the device layout.

Default new specs: a 1310 nm O-band 75/25 coupler on a 220 nm SOI strip stack.

Run with ``python -m examples.papers.mao2019_designer``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from examples.papers import mao2019 as m

FIGDIR = m.FIGDIR
LAYER_WG = m.LAYER_WG


def design_dw_out(
    target_sr: float, stack: m.SOIStack, *, res: float = 0.02, npts: int = 9
) -> tuple[float, np.ndarray, np.ndarray]:
    """Find the ``dw_out`` giving ``target_sr`` by interpolating the SR curve.

    Returns ``(dw_out, dw_grid, sr_grid)``.
    """
    dw_grid = np.linspace(0.0, 0.06, npts)
    sr_grid = np.array([m.splitting_ratio(float(d), stack, res=res) for d in dw_grid])
    # SR is monotonically increasing in dw_out; interpolate the inverse
    target = float(np.clip(target_sr, sr_grid.min(), sr_grid.max()))
    dw = float(np.interp(target, sr_grid, dw_grid))
    return dw, dw_grid, sr_grid


def adc_component(
    stack: m.SOIStack,
    dw_out: float,
    *,
    w: float = 0.40,
    g1: float = 1.5,
    lt: float = 40.0,
    ls: float = 30.0,
    length_iii: float = 60.0,
    npts: int = 200,
) -> Any:
    """Lay out the five-region SR-coupler (two waveguides) as a gdsfactory cell."""
    import gdsfactory as gf

    w1, g2 = stack.w1, stack.g2
    w2 = w1 - 0.02  # input width-difference (Region II asymmetry)
    w3, w4 = stack.w1, stack.w1 - dw_out
    z = [0, lt, lt + ls, lt + ls + length_iii, lt + 2 * ls + length_iii,
         2 * lt + 2 * ls + length_iii]

    def gap(zz: float) -> float:
        if zz <= z[1]:
            return g1
        if zz <= z[2]:
            return g1 + (g2 - g1) * (zz - z[1]) / ls
        if zz <= z[3]:
            return g2
        if zz <= z[4]:
            return g2 + (g1 - g2) * (zz - z[3]) / ls
        return g1

    def wid(zz: float, top: bool) -> float:
        a, b, c, d = (w, w1, w3, w) if top else (w, w2, w4, w)
        if zz <= z[1]:
            return a + (b - a) * zz / lt
        if zz <= z[3]:
            return b + (c - b) * (zz - z[1]) / (z[3] - z[1])
        if zz <= z[5]:
            return c + (d - c) * (zz - z[3]) / (z[5] - z[3])
        return d

    zs = np.linspace(0.0, z[5], npts)
    c = gf.Component()
    for top in (True, False):
        sign = 1 if top else -1
        gc = np.array([sign * (gap(float(zz)) / 2 + wid(float(zz), top) / 2) for zz in zs])
        half = np.array([wid(float(zz), top) / 2 for zz in zs])
        upper = np.column_stack([zs, gc + half])
        lower = np.column_stack([zs[::-1], (gc - half)[::-1]])
        c.add_polygon(np.vstack([upper, lower]), layer=LAYER_WG)
    c.add_port("in1", center=(0.0, g1 / 2 + w / 2), width=w, orientation=180,
               layer=LAYER_WG)
    c.add_port("out_up", center=(z[5], g1 / 2 + w / 2), width=w, orientation=0,
               layer=LAYER_WG)
    return c


def plot_design_point(
    dw_grid: np.ndarray, sr_grid: np.ndarray, dw: float, target_sr: float, path: Path
) -> None:
    """SR-vs-dw_out curve with the chosen design point."""
    plt = m._use_agg()
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.plot(dw_grid * 1000, sr_grid * 100, "C0o-")
    ax.plot([dw * 1000], [target_sr * 100], "k*", ms=15,
            label=f"design: SR={target_sr * 100:.0f}%, $\\Delta w$={dw * 1000:.1f} nm")
    ax.set_xlabel(r"$\Delta w_{out}$ [nm]")
    ax.set_ylabel("splitting ratio (upper port) [%]")
    ax.grid(visible=True, alpha=0.3)
    ax.legend()
    ax.set_title("Designer: target SR -> output width-difference")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_layout(component: Any, path: Path, *, title: str) -> None:
    """Draw the designed coupler layout."""
    plt = m._use_agg()
    from examples.papers._plot import plot_component

    fig, ax = plt.subplots(figsize=(11, 3))
    plot_component(component, ax)
    ax.set_aspect("auto")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> dict[str, Any]:
    """Design a 75/25 SOI adiabatic coupler at 1310 nm and emit GDS + figures."""
    import gdsfactory as gf

    import meow as mw
    from examples.papers import _resolution as res

    gf.gpdk.PDK.activate()
    out = FIGDIR / "mao2019_designer"
    out.mkdir(parents=True, exist_ok=True)
    stack = m.SOIStack()
    target_sr = 0.75
    device_res = res.pick(low=0.03, medium=0.02, high=0.015)
    npts = res.pick(low=7, medium=9, high=13)
    dw, dw_grid, sr_grid = design_dw_out(
        target_sr, stack, res=device_res, npts=npts
    )
    plot_design_point(dw_grid, sr_grid, dw, target_sr, out / "design_point.png")
    comp = adc_component(stack, dw)
    comp.write_gds(str(out / "adc_75_25.gds"))
    plot_layout(comp, out / "adc_layout.png",
                title=f"75/25 SOI ADC @ 1310 nm (dw_out={dw * 1000:.1f} nm)")
    summary = {
        "target_sr": target_sr, "wl_nm": stack.wl * 1000,
        "dw_out_nm": round(dw * 1000, 2), "w3_um": stack.w1,
        "w4_um": round(stack.w1 - dw, 4),
    }
    mw.save_summary(out / "summary", summary)
    return {"out_dir": str(out), "summary": summary,
            "files": sorted(p.name for p in out.glob("*"))}


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
