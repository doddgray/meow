"""Song et al., *Fully adiabatic polarization rotator-splitter based on thin-film
lithium niobate platform*, Opt. Express 31(12), 19604 (2023).

This example reproduces the core mode-evolution physics of the TFLN PRS and adds
a designer. The PRS rotates/splits polarization by adiabatic mode evolution: a
partially-etched x-cut TFLN ridge is linearly widened so that an input TM0 stays
on its local-normal mode and **adiabatically evolves into TE1** (then an adiabatic
coupler separates TE1 from TE0). The enabling physics is the **TM0/TE1
hybridization** of the ridge: near a critical width ``w0`` (~1.2 um in the paper)
the second and third modes swap polarization character through an avoided
crossing.

Reproduced here with meow's full-vectorial FDE on anisotropic LN:

- the ridge effective indices ``neff`` of modes 1-3 vs width (paper Fig. 2a);
- the TE (``Ex``) polarization fraction of modes 2/3 vs width, crossing 50% at
  the hybridization width (paper Fig. 2b).

These set the rotator's taper start/cutoff widths (``w0 -+ 0.3 um``). The taper
and coupler length-dependent conversion efficiencies (Figs 2d/3d) need EME and
are left as coarse cross-checks.

Run with ``python -m examples.papers.song2023``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from examples.papers.kwolek_designer import tfln_platform

FIGDIR = Path(__file__).parent / "figures"


def song_platform() -> Any:
    """The paper's x-cut TFLN platform: 300 nm film, 200 nm etch (100 nm slab)."""
    # the paper's sidewall is 75 deg from horizontal = 15 deg from vertical
    return tfln_platform(0.30, etch_depth=0.20, sidewall_deg=15.0)


def _ridge_cross_section(platform: Any, width: float, wl: float, res: float) -> Any:
    """A symmetric-SiO2-clad ridge cross-section at the given top ``width``."""
    import meow as mw
    from examples.papers.kwolek_designer import rib_structures

    structs = rib_structures(platform, wl, [width], [0.0], x_span=(-4.0, 4.0))
    h = platform.core_thickness
    # symmetric SiO2 top cladding (the paper uses symmetric oxide), below the rib
    structs.append(
        mw.Structure(
            material=platform.clad(wl),
            geometry=mw.Box(
                x_min=-4.0, x_max=4.0, y_min=0.0, y_max=h + 0.8,
                z_min=0.0, z_max=1.0,
            ),
            mesh_order=9,
        )
    )
    mesh = mw.Mesh2D(
        x=np.arange(-4.0, 4.0 + res / 2, res),
        y=np.arange(-platform.box_thickness, h + 0.8 + res / 2, res),
    )
    cell = mw.Cell(structures=structs, mesh=mesh, z_min=0.0, z_max=1.0)
    return mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=wl, T=25.0))


def ridge_modes(
    platform: Any, width: float, wl: float, *, num_modes: int = 4, res: float = 0.03
) -> list[Any]:
    """Solve the lowest vectorial ridge modes, sorted by descending neff."""
    import meow as mw

    cs = _ridge_cross_section(platform, width, wl, res)
    modes = mw.compute_modes(cs, num_modes=num_modes)
    return sorted(modes, key=lambda m: float(np.real(m.neff)), reverse=True)


def hybridization_scan(
    platform: Any, widths: np.ndarray, wl: float, *, res: float = 0.03
) -> tuple[np.ndarray, np.ndarray]:
    """neff[w, mode] and Ex-fraction[w, mode] for the lowest 3 modes vs width."""
    import meow as mw

    neffs, fracs = [], []
    for w in widths:
        modes = ridge_modes(platform, float(w), wl, num_modes=4, res=res)[:3]
        neffs.append([float(np.real(m.neff)) for m in modes])
        fracs.append([float(mw.te_fraction(m)) for m in modes])
    return np.asarray(neffs), np.asarray(fracs)


def hybridization_width(widths: np.ndarray, fracs: np.ndarray) -> float:
    """Width where modes 2 and 3 swap TE/TM character (Ex fractions cross)."""
    diff = fracs[:, 1] - fracs[:, 2]  # mode2 - mode3 TE fraction
    sign = np.sign(diff)
    crossings = np.where(np.diff(sign) != 0)[0]
    if crossings.size:
        i = crossings[0]
        return float(np.interp(0.0, [diff[i], diff[i + 1]], [widths[i], widths[i + 1]]))
    return float(widths[np.argmin(np.abs(diff))])


def _use_agg() -> Any:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_hybridization(
    widths: np.ndarray, neffs: np.ndarray, fracs: np.ndarray, w0: float, path: Path
) -> None:
    """Reproduce Fig. 2(a,b): neff and Ex fraction vs ridge width."""
    plt = _use_agg()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.3))
    for k, lbl in enumerate(("mode 1", "mode 2", "mode 3")):
        a.plot(widths, neffs[:, k], "o-", label=lbl, ms=3)
    a.axvline(w0, color="0.5", ls="--", label=f"hybridization w0={w0:.2f} um")
    a.set_xlabel("ridge width [um]")
    a.set_ylabel(r"$n_{eff}$")
    a.legend(fontsize=8)
    a.grid(visible=True, alpha=0.3)
    a.set_title("Fig. 2(a): ridge effective indices")
    for k, lbl in ((1, "mode 2"), (2, "mode 3")):
        b.plot(widths, fracs[:, k] * 100, "o-", label=lbl, ms=3)
    b.axhline(50, color="0.6", ls=":")
    b.axvline(w0, color="0.5", ls="--")
    b.set_xlabel("ridge width [um]")
    b.set_ylabel("TE (Ex) fraction [%]")
    b.legend(fontsize=8)
    b.grid(visible=True, alpha=0.3)
    b.set_title("Fig. 2(b): TE polarization fraction")
    fig.suptitle("Song 2023 TFLN PRS: TM0/TE1 hybridization")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> dict[str, Any]:
    """Reproduce the Song 2023 hybridization analysis (Fig. 2a/2b)."""
    import gdsfactory as gf

    import meow as mw
    from examples.papers import _resolution as res

    gf.gpdk.PDK.activate()
    out = FIGDIR / "song2023"
    out.mkdir(parents=True, exist_ok=True)
    platform = song_platform()
    npts = res.pick(low=8, medium=15, high=29)
    widths = np.linspace(0.8, 1.9, npts)
    device_res = res.pick(low=0.04, medium=0.03, high=0.02)
    neffs, fracs = hybridization_scan(platform, widths, 1.55, res=device_res)
    w0 = hybridization_width(widths, fracs)
    plot_hybridization(widths, neffs, fracs, w0, out / "fig2_hybridization.png")
    mw.save_table(
        out / "hybridization",
        {"width_um": widths, "neff1": neffs[:, 0], "neff2": neffs[:, 1],
         "neff3": neffs[:, 2], "te2": fracs[:, 1], "te3": fracs[:, 2]},
    )
    summary = {"hybridization_width_um": round(w0, 3),
               "taper_w_start_um": round(w0 - 0.3, 3),
               "taper_w_end_um": round(w0 + 0.3, 3)}
    mw.save_summary(out / "summary", summary)
    return {"out_dir": str(out), "summary": summary,
            "files": sorted(p.name for p in out.glob("*"))}


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
