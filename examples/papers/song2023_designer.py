"""TFLN polarization-rotator designer (Song 2023 workflow, new specs).

Companion to :mod:`examples.papers.song2023`. Given a new TFLN platform
(film thickness, etch, wavelength), this designer finds the ridge **TM0/TE1
hybridization width** ``w0`` from a meow vectorial-FDE width scan, sets the
adiabatic rotator taper start/cutoff widths to ``w0 -+ 0.3 um``, and emits the
hybridization figure plus a top-view GDS of the rotator taper feeding the
adiabatic coupler.

Default new specs: a thicker 500 nm x-cut TFLN film at 1550 nm.

Run with ``python -m examples.papers.song2023_designer``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from examples.papers import song2023 as s
from examples.papers.kwolek_designer import tfln_platform

FIGDIR = s.FIGDIR
LAYER_WG = (1, 0)


def design_platform(core_thickness: float = 0.50) -> Any:
    """A new (thicker) x-cut TFLN platform for the designer."""
    return tfln_platform(core_thickness, etch_depth=0.30, sidewall_deg=15.0)


def rotator_taper_gds(
    w_start: float, w_end: float, *, l_taper: float = 150.0, w_out: float = 0.9
) -> Any:
    """Top-view GDS: the rotator ridge widening, then an adiabatic-coupler arm."""
    import gdsfactory as gf

    c = gf.Component()
    zs = np.linspace(0.0, l_taper, 80)
    w = w_start + (w_end - w_start) * (zs / l_taper)
    top = np.column_stack([zs, w / 2])
    bot = np.column_stack([zs[::-1], (-w / 2)[::-1]])
    c.add_polygon(np.vstack([top, bot]), layer=LAYER_WG)
    # adiabatic-coupler through arm (narrows back to a single-mode output)
    zc = np.linspace(l_taper, l_taper + 120.0, 40)
    wc = w_end + (w_out - w_end) * (zc - l_taper) / 120.0
    topc = np.column_stack([zc, wc / 2])
    botc = np.column_stack([zc[::-1], (-wc / 2)[::-1]])
    c.add_polygon(np.vstack([topc, botc]), layer=LAYER_WG)
    c.add_port("in", center=(0.0, 0.0), width=w_start, orientation=180, layer=LAYER_WG)
    c.add_port("out", center=(l_taper + 120.0, 0.0), width=w_out, orientation=0,
               layer=LAYER_WG)
    return c


def main() -> dict[str, Any]:
    """Design a TFLN polarization rotator for a new (500 nm) film."""
    import gdsfactory as gf

    import meow as mw
    from examples.papers import _resolution as res

    gf.gpdk.PDK.activate()
    out = FIGDIR / "song2023_designer"
    out.mkdir(parents=True, exist_ok=True)
    platform = design_platform(0.50)
    wl = 1.55
    npts = res.pick(low=8, medium=15, high=29)
    widths = np.linspace(0.7, 1.8, npts)
    device_res = res.pick(low=0.04, medium=0.03, high=0.02)
    neffs, fracs = s.hybridization_scan(platform, widths, wl, res=device_res)
    w0 = s.hybridization_width(widths, fracs)
    s.plot_hybridization(widths, neffs, fracs, w0, out / "hybridization.png")
    w_start, w_end = w0 - 0.3, w0 + 0.3
    comp = rotator_taper_gds(w_start, w_end)
    comp.write_gds(str(out / "tfln_rotator.gds"))
    summary = {
        "platform": platform.name, "wl_nm": wl * 1000,
        "hybridization_width_um": round(w0, 3),
        "taper_w_start_um": round(w_start, 3), "taper_w_end_um": round(w_end, 3),
    }
    mw.save_summary(out / "summary", summary)
    return {"out_dir": str(out), "summary": summary,
            "files": sorted(p.name for p in out.glob("*"))}


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
