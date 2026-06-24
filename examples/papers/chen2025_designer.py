"""FAQUAD TFLN PSR designer (Chen 2025 workflow, new specs).

Companion to :mod:`examples.papers.chen2025`. Given a new target AMEW length on
a chosen TFLN platform, this designer computes the FAQUAD cross-waveguide width
profile ``w1(z)`` from a meow vectorial-FDE neff scan, then lays out the
two-waveguide PSR (fixed through guide + FAQUAD-tapered cross guide) and writes
its GDS, plus the FAQUAD profile and the neff-evolution figure.

Default new specs: a shorter 200 um AMEW on the 400 nm TFLN platform.

Run with ``python -m examples.papers.chen2025_designer``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from examples.papers import chen2025 as c

FIGDIR = c.FIGDIR
LAYER_WG = c.LAYER_WG


def psr_gds(
    z: np.ndarray, w1: np.ndarray, *, w0: float = 1.0, g: float = 0.4
) -> Any:
    """Lay out the PSR: a straight through guide + the FAQUAD-tapered cross guide."""
    import gdsfactory as gf

    comp = gf.Component()
    # through waveguide (constant w0), centred at y = 0
    comp.add_polygon(
        np.array([(z[0], -w0 / 2), (z[-1], -w0 / 2), (z[-1], w0 / 2), (z[0], w0 / 2)]),
        layer=LAYER_WG,
    )
    # cross waveguide: inner edge at fixed gap g above the through guide
    inner = w0 / 2 + g
    top = np.column_stack([z, inner + w1])
    bot = np.column_stack([z[::-1], np.full_like(z, inner)[::-1]])
    comp.add_polygon(np.vstack([top, bot]), layer=LAYER_WG)
    comp.add_port("thru_in", center=(z[0], 0.0), width=w0, orientation=180,
                  layer=LAYER_WG)
    comp.add_port("thru_out", center=(z[-1], 0.0), width=w0, orientation=0,
                  layer=LAYER_WG)
    comp.add_port("cross_out", center=(z[-1], inner + w1[-1] / 2), width=w1[-1],
                  orientation=0, layer=LAYER_WG)
    return comp


def main() -> dict[str, Any]:
    """Design a compact 200 um FAQUAD PSR on the 400 nm TFLN platform."""
    import gdsfactory as gf

    import meow as mw
    from examples.papers import _resolution as res

    gf.gpdk.PDK.activate()
    out = FIGDIR / "chen2025_designer"
    out.mkdir(parents=True, exist_ok=True)
    platform = c.chen_platform()
    length = 200.0
    npts = res.pick(low=8, medium=15, high=29)
    w1s = np.linspace(0.1, 0.65, npts)
    device_res = res.pick(low=0.04, medium=0.03, high=0.02)
    neffs, _ = c.neff_evolution(platform, w1s, 1.58, res=device_res)
    w_hyb = c.hybridization_point(w1s, neffs)
    c.plot_neff_evolution(w1s, neffs, w_hyb, out / "neff_evolution.png")
    z, w1 = c.faquad_profile(w1s, neffs, length=length)
    c.plot_faquad_profile(z, w1, out / "faquad_profile.png")
    comp = psr_gds(z, w1)
    comp.write_gds(str(out / "tfln_psr.gds"))
    summary = {
        "platform": platform.name, "amew_length_um": length,
        "hybridization_w1_um": round(w_hyb, 3),
        "w1_start_um": float(w1s[0]), "w1_end_um": float(w1s[-1]),
    }
    mw.save_summary(out / "summary", summary)
    return {"out_dir": str(out), "summary": summary,
            "files": sorted(p.name for p in out.glob("*"))}


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
