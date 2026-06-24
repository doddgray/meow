"""Chen et al., *Compact and broadband polarization splitter-rotator on thin-film
lithium niobate minimized with the fast quasi-adiabatic algorithm*, Opt. Lett.
50(3), 710 (2025).

This example reproduces the mode-evolution analysis of the TFLN PSR and the
**fast quasi-adiabatic (FAQUAD)** width profile that makes it compact, and adds a
designer. The PSR is an adiabatic mode-evolution waveguide: a fixed through
waveguide (top width ``w0 = 1 um``) beside a *cross* waveguide whose top width
``w1`` widens from 0.1 to 0.65 um at a fixed top gap ``g = 0.4 um`` on a 400 nm
x-cut TFLN ridge (etch 200 nm, sidewall 65 deg). The through-TE mode keeps the
largest ``neff`` and stays put, while the through-TM mode transfers to TE in the
cross waveguide through a **mode-hybridization point** (~0.37 um) where the
second and third modes' indices anti-cross (paper Fig. 1d).

Reproduced here with meow's full-vectorial FDE on anisotropic LN:

- the ``neff`` evolution of the first four modes vs ``w1`` (paper Fig. 1d),
  locating the hybridization point;
- the **FAQUAD width profile** ``w1(z)``: the FAQUAD criterion (eq 1) keeps the
  local adiabaticity constant, so length is concentrated where the modes are
  closest (the anti-crossing). With a smooth inter-mode coupling the density is
  ``rho(w1) ~ 1 / (beta2**2 - beta3**2)**2``; integrating it gives ``w1(z)``,
  far more efficient than a linear taper.

The length-dependent IL/PER (which the paper finalizes with EME) is left as a
coarse cross-check.

Run with ``python -m examples.papers.chen2025``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from examples.papers.kwolek_designer import tfln_platform

FIGDIR = Path(__file__).parent / "figures"
LAYER_WG = (1, 0)


def chen_platform() -> Any:
    """The paper's x-cut TFLN platform: 400 nm film, 200 nm etch, 65 deg sidewall."""
    # 65 deg from horizontal = 25 deg from vertical
    return tfln_platform(0.40, etch_depth=0.20, sidewall_deg=25.0)


def _two_wg_cross_section(
    platform: Any, w1: float, wl: float, res: float, *, w0: float = 1.0, g: float = 0.4
) -> Any:
    """Through (``w0``) + cross (``w1``) ridges at top gap ``g``."""
    import meow as mw
    from examples.papers.kwolek_designer import rib_structures

    c_thru = -(g + w0 + w1) / 4  # place the pair roughly centred
    c_cross = c_thru + (w0 + w1) / 2 + g
    structs = rib_structures(
        platform, wl, [w0, w1], [c_thru, c_cross], x_span=(-3.5, 3.5)
    )
    h = platform.core_thickness
    mesh = mw.Mesh2D(
        x=np.arange(-3.5, 3.5 + res / 2, res),
        y=np.arange(-platform.box_thickness, h + 0.6 + res / 2, res),
    )
    cell = mw.Cell(structures=structs, mesh=mesh, z_min=0.0, z_max=1.0)
    return mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=wl, T=25.0))


def neff_evolution(
    platform: Any, w1s: np.ndarray, wl: float = 1.58, *, res: float = 0.03
) -> tuple[np.ndarray, np.ndarray]:
    """neff[w1, mode] and Ex-fraction[w1, mode] of the first 4 modes vs ``w1``."""
    import meow as mw

    neffs, fracs = [], []
    for w1 in w1s:
        cs = _two_wg_cross_section(platform, float(w1), wl, res)
        modes = sorted(
            mw.compute_modes(cs, num_modes=4),
            key=lambda m: float(np.real(m.neff)), reverse=True,
        )
        neffs.append([float(np.real(m.neff)) for m in modes])
        fracs.append([float(mw.te_fraction(m)) for m in modes])
    return np.asarray(neffs), np.asarray(fracs)


def hybridization_point(w1s: np.ndarray, neffs: np.ndarray) -> float:
    """The ``w1`` where modes 2 and 3 anti-cross (their index gap is smallest)."""
    gap = np.abs(neffs[:, 1] - neffs[:, 2])
    return float(w1s[int(np.argmin(gap))])


def faquad_profile(
    w1s: np.ndarray, neffs: np.ndarray, *, length: float = 300.0
) -> tuple[np.ndarray, np.ndarray]:
    """FAQUAD width profile ``w1(z)`` (smooth-coupling approximation, eq 1/4).

    The local adiabaticity density ``rho ~ 1/(beta2**2 - beta3**2)**2`` is
    integrated and inverted so the cross-guide widens slowly through the
    anti-crossing. Returns ``(z, w1)`` sampled along the device of ``length`` um.
    """
    wl = 1.58
    beta = 2 * np.pi * neffs / wl
    dbeta2 = beta[:, 1] ** 2 - beta[:, 2] ** 2
    rho = 1.0 / np.maximum(dbeta2**2, 1e-12)
    # cumulative adiabaticity cost -> normalized arclength z(w1)
    cost = np.concatenate([[0.0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * np.diff(w1s))])
    z = length * cost / cost[-1]
    return z, w1s


def _use_agg() -> Any:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_neff_evolution(
    w1s: np.ndarray, neffs: np.ndarray, w_hyb: float, path: Path
) -> None:
    """Reproduce Fig. 1(d): neff evolution of the first four modes vs ``w1``."""
    plt = _use_agg()
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for k in range(neffs.shape[1]):
        ax.plot(w1s, neffs[:, k], "o-", ms=3, label=f"mode {k + 1}")
    ax.axvline(w_hyb, color="0.5", ls="--", label=f"hybridization {w_hyb:.2f} um")
    ax.set_xlabel(r"cross-waveguide width $w_1$ [um]")
    ax.set_ylabel(r"$n_{eff}$")
    ax.legend(fontsize=8)
    ax.grid(visible=True, alpha=0.3)
    ax.set_title("Fig. 1(d): neff evolution along the AMEW (1.58 um)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_faquad_profile(z: np.ndarray, w1: np.ndarray, path: Path) -> None:
    """FAQUAD width profile vs a linear taper of the same length."""
    plt = _use_agg()
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.plot(z, w1, "C0-", lw=2, label="FAQUAD profile")
    ax.plot([z[0], z[-1]], [w1[0], w1[-1]], "k--", label="linear taper")
    ax.set_xlabel(r"position along AMEW $z$ [um]")
    ax.set_ylabel(r"cross width $w_1$ [um]")
    ax.legend()
    ax.grid(visible=True, alpha=0.3)
    ax.set_title("FAQUAD width profile (slows through the anti-crossing)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> dict[str, Any]:
    """Reproduce the Chen 2025 neff evolution + FAQUAD width profile."""
    import gdsfactory as gf

    import meow as mw
    from examples.papers import _resolution as res

    gf.gpdk.PDK.activate()
    out = FIGDIR / "chen2025"
    out.mkdir(parents=True, exist_ok=True)
    platform = chen_platform()
    npts = res.pick(low=8, medium=15, high=29)
    w1s = np.linspace(0.1, 0.65, npts)
    device_res = res.pick(low=0.04, medium=0.03, high=0.02)
    neffs, fracs = neff_evolution(platform, w1s, 1.58, res=device_res)
    w_hyb = hybridization_point(w1s, neffs)
    plot_neff_evolution(w1s, neffs, w_hyb, out / "fig1d_neff_evolution.png")
    z, w1 = faquad_profile(w1s, neffs, length=300.0)
    plot_faquad_profile(z, w1, out / "faquad_profile.png")
    mw.save_table(
        out / "neff_evolution",
        {"w1_um": w1s, **{f"neff{k + 1}": neffs[:, k] for k in range(neffs.shape[1])}},
    )
    mw.save_table(out / "faquad_profile", {"z_um": z, "w1_um": w1})
    summary = {"hybridization_w1_um": round(w_hyb, 3), "amew_length_um": 300.0}
    mw.save_summary(out / "summary", summary)
    return {"out_dir": str(out), "summary": summary,
            "files": sorted(p.name for p in out.glob("*"))}


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
