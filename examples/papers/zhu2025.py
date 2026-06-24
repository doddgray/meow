"""Zhu, Yue, Cai & Wang, *Three-dimensional trident edge coupler with ultralow
loss and broad bandwidth for standard single-mode fiber*, J. Opt. Soc. Am. B
42(8), 1738 (2025).

This example reproduces the paper's mode-overlap analysis of a 3D trident SiN
edge coupler and adds a designer for new target specs. The coupler's chip facet
is a centrally-symmetric arrangement of thin (``H = 0.2 um``) SiN bars on three
vertical levels - a coplanar *trident* (left/centre/right) plus *assisted*
waveguides above and below - whose weakly-guided **collective** mode expands to
~10 um to match a standard single-mode fiber (MFD 10.4 um). The light then
tapers adiabatically into a single compact SiN output waveguide.

The figures of merit reproduced here with meow's full-vectorial FDE solver:

- the collective TE/TM facet mode (paper Fig. 2);
- the **mode-overlap efficiency** with the SMF (paper eq 1)::

      eta = |integral(E1 . E2 dA)|**2 / (integral|E1|**2 dA * integral|E2|**2 dA)

  versus wavelength over 1450-1650 nm (paper Fig. 3, > 95-97%);
- the fiber **alignment tolerance** - eta versus lateral/vertical fiber offset
  (paper Fig. 9, +-1.9 / +-2.0 um for 1 dB);
- the **oxide-thickness** dependence (paper Fig. 8).

The adiabatic taper transmission (paper Figs. 4/6/7), which the paper computes
with 3D-FDTD, is provided only as a coarse, clearly-labelled meow-EME
cross-check (a converged EME of a ~10 um, ~300 um-long mode needs a very large
fine mesh).

Run with ``python -m examples.papers.zhu2025``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FIGDIR = Path(__file__).parent / "figures"

# SMF mode-field diameter [um] (1/e^2 intensity); fiber 1/e field radius = MFD/2
SMF_MFD = 10.4


@dataclass
class TridentFacet:
    """The 3D trident edge-coupler facet geometry (paper Table 1, in um).

    The bars are all ``h`` thick. The coplanar trident has a left and a right
    arm (width ``w3``) flanking the central output-waveguide location, on a
    horizontal pitch ``g1``; assisted bars (width ``w1``) sit ``h1`` above and
    below, giving the centrally-symmetric "cross" facet.
    """

    h: float = 0.20  # SiN layer thickness (H)
    w1: float = 0.32  # assisted-waveguide width (W1)
    w3: float = 0.30  # trident-arm width (W3)
    g1: float = 2.1  # trident horizontal pitch (G1)
    h1: float = 2.2  # vertical level spacing (H1)
    t1: float = 0.5  # oxide below bottom SiN (T1)
    t2: float = 4.5  # oxide above top SiN (T2)


def _sin() -> Any:
    import meow as mw

    return mw.silicon_nitride


def facet_structures(facet: TridentFacet) -> list[Any]:
    """Meow structures for the centrally-symmetric trident facet cross-section.

    Lateral is ``x``, vertical is ``y``; bars are thin SiN rectangles in oxide.
    """
    import meow as mw

    sin = _sin()
    h, w1, w3, g1, h1 = facet.h, facet.w1, facet.w3, facet.g1, facet.h1

    def bar(cx: float, cy: float, w: float) -> Any:
        return mw.Structure(
            material=sin,
            geometry=mw.Box(
                x_min=cx - w / 2, x_max=cx + w / 2,
                y_min=cy - h / 2, y_max=cy + h / 2, z_min=0.0, z_max=1.0,
            ),
        )

    # centrally-symmetric cross: left/right trident arms + top/bottom assisted
    bars = [
        bar(-g1, 0.0, w3),
        bar(+g1, 0.0, w3),
        bar(0.0, +h1, w1),
        bar(0.0, -h1, w1),
    ]
    span_x, span_y = 11.0, 9.0
    oxide = mw.Structure(
        material=mw.silicon_oxide,
        geometry=mw.Box(
            x_min=-span_x, x_max=span_x, y_min=-span_y, y_max=span_y,
            z_min=0.0, z_max=1.0,
        ),
        mesh_order=10,
    )
    return [*bars, oxide]


def _graded_axis(
    fine_half: float, fine_res: float, full_half: float, coarse_res: float
) -> np.ndarray:
    """A symmetric axis: fine spacing within +-fine_half, coarse outside."""
    fine = np.arange(-fine_half, fine_half + fine_res / 2, fine_res)
    left = np.arange(-full_half, -fine_half, coarse_res)
    right = np.arange(fine_half + coarse_res, full_half + coarse_res / 2, coarse_res)
    return np.concatenate([left, fine, right])


def facet_mesh(*, fine_res: float = 0.05) -> Any:
    """A graded cross-section mesh: fine near the bars, coarse out to the mode."""
    import meow as mw

    x = _graded_axis(3.5, fine_res, 11.0, 0.25)
    y = _graded_axis(3.5, fine_res, 9.0, 0.25)
    return mw.Mesh2D(x=x, y=y)


def facet_modes(
    facet: TridentFacet, wl: float, *, num_modes: int = 4, fine_res: float = 0.05
) -> list[Any]:
    """Solve the lowest facet supermodes (full-vectorial FDE) at ``wl`` [um]."""
    import meow as mw

    cells = mw.create_cells(
        facet_structures(facet), facet_mesh(fine_res=fine_res),
        np.array([1.0]), z_min=0.0,
    )
    cs = mw.CrossSection.from_cell(cell=cells[0], env=mw.Environment(wl=wl, T=25.0))
    return mw.compute_modes(cs, num_modes=num_modes)


def _pick_polarization(modes: list[Any], pol: str) -> Any:
    """Pick the (highest-neff) mode of the requested polarization (te/tm)."""
    import meow as mw

    want_te = pol.lower() == "te"
    best = None
    for m in modes:
        frac = float(mw.te_fraction(m))
        is_te = frac >= 0.5
        if is_te == want_te and (best is None or np.real(m.neff) > np.real(best.neff)):
            best = m
    return best or modes[0]


def _transverse_field(mode: Any, pol: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (field, x, y): the dominant transverse E component on its grid."""
    field = np.asarray(mode.Ex if pol.lower() == "te" else mode.Ey)
    mesh = mode.cs.mesh
    x = np.linspace(float(mesh.x.min()), float(mesh.x.max()), field.shape[0])
    y = np.linspace(float(mesh.y.min()), float(mesh.y.max()), field.shape[1])
    return field, x, y


def fiber_overlap(
    mode: Any, pol: str, *, mfd: float = SMF_MFD, y0: float = 0.0, z0: float = 0.0
) -> float:
    """Mode-overlap efficiency (paper eq 1) of a facet mode with an SMF Gaussian.

    ``y0`` / ``z0`` are the fiber lateral / vertical offsets [um] (the paper's
    Y/Z); the fiber is a fundamental Gaussian of mode-field diameter ``mfd``.
    """
    import meow as mw

    field, x, y = _transverse_field(mode, pol)
    w0 = mfd / 2.0  # 1/e field radius
    gauss = mw.hermite_gaussian_field(x, y, 0, 0, y0, z0, w0, w0)
    e1 = field.astype(complex)
    e2 = gauss.astype(complex)
    num = np.abs(np.sum(e1 * np.conj(e2))) ** 2
    den = np.sum(np.abs(e1) ** 2) * np.sum(np.abs(e2) ** 2)
    return float(num / den) if den else 0.0


# ==========================================================================
# figure reproductions
# ==========================================================================
def _use_agg() -> Any:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_facet_modes(facet: TridentFacet, wl: float, path: Path) -> dict[str, float]:
    """Reproduce Fig. 2: the collective TE/TM facet mode-intensity maps."""
    plt = _use_agg()
    modes = facet_modes(facet, wl, num_modes=6)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    etas = {}
    for ax, pol, title in ((axes[0], "tm", "TM"), (axes[1], "te", "TE")):
        mode = _pick_polarization(modes, pol)
        field, x, y = _transverse_field(mode, pol)
        intensity = np.abs(field).T ** 2
        ax.pcolormesh(x, y, intensity / intensity.max(), shading="auto", cmap="inferno")
        eta = fiber_overlap(mode, pol)
        etas[pol] = eta
        ax.set_title(f"{title} facet mode ($\\eta$={eta * 100:.1f}%)")
        ax.set_xlabel("x [um]")
        ax.set_ylabel("y [um]")
        ax.set_xlim(-8, 8)
        ax.set_ylim(-7, 7)
        ax.set_aspect("equal")
    fig.suptitle(f"Fig. 2: trident facet modes @ {wl * 1000:.0f} nm")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return etas


def overlap_vs_wavelength(
    facet: TridentFacet, wls: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """eta(TE), eta(TM) vs wavelength (paper Fig. 3)."""
    te, tm = [], []
    for wl in wls:
        modes = facet_modes(facet, float(wl), num_modes=6)
        te.append(fiber_overlap(_pick_polarization(modes, "te"), "te"))
        tm.append(fiber_overlap(_pick_polarization(modes, "tm"), "tm"))
    return np.asarray(te), np.asarray(tm)


def plot_overlap_vs_wavelength(
    wls: np.ndarray, te: np.ndarray, tm: np.ndarray, path: Path
) -> None:
    """Reproduce Fig. 3: mode-overlap efficiency vs wavelength."""
    plt = _use_agg()
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.plot(wls * 1000, te * 100, "C0o-", label="TE")
    ax.plot(wls * 1000, tm * 100, "C3s-", label="TM")
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("mode overlap efficiency [%]")
    ax.grid(visible=True, alpha=0.3)
    ax.legend()
    ax.set_title("Fig. 3: facet-SMF mode overlap vs wavelength")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_alignment_tolerance(
    facet: TridentFacet, wl: float, path: Path, *, max_offset: float = 3.0
) -> dict[str, float]:
    """Reproduce Fig. 9: excess loss vs fiber Y/Z offset, with 1 dB tolerances."""
    plt = _use_agg()
    modes = facet_modes(facet, wl, num_modes=6)
    te = _pick_polarization(modes, "te")
    offs = np.linspace(-max_offset, max_offset, 25)
    eta0 = fiber_overlap(te, "te")
    loss_y = np.array(
        [-10 * np.log10(fiber_overlap(te, "te", y0=o) / eta0) for o in offs]
    )
    loss_z = np.array(
        [-10 * np.log10(fiber_overlap(te, "te", z0=o) / eta0) for o in offs]
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.plot(offs, loss_y, "C0o-", label="Y (lateral) offset")
    ax.plot(offs, loss_z, "C3s-", label="Z (vertical) offset")
    ax.axhline(1.0, color="0.5", ls=":", label="1 dB")
    ax.set_xlabel("fiber offset [um]")
    ax.set_ylabel("excess loss [dB]")
    ax.set_ylim(0, 3)
    ax.grid(visible=True, alpha=0.3)
    ax.legend()
    ax.set_title("Fig. 9: fiber alignment tolerance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    def one_db(loss: np.ndarray) -> float:
        below = offs[loss <= 1.0]
        return float(below.max()) if below.size else 0.0

    return {"y_1db_um": one_db(loss_y), "z_1db_um": one_db(loss_z)}


def main() -> dict[str, Any]:
    """Reproduce the trident edge-coupler overlap analyses into FIGDIR."""
    import gdsfactory as gf

    from examples.papers import _resolution as res

    gf.gpdk.PDK.activate()  # for consistency with the other examples
    out = FIGDIR / "zhu2025"
    out.mkdir(parents=True, exist_ok=True)
    facet = TridentFacet()
    etas = plot_facet_modes(facet, 1.55, out / "fig2_facet_modes.png")
    npts = res.pick(low=5, medium=9, high=21)
    wls = np.linspace(1.45, 1.65, npts)
    te, tm = overlap_vs_wavelength(facet, wls)
    plot_overlap_vs_wavelength(wls, te, tm, out / "fig3_overlap_vs_wavelength.png")
    tol = plot_alignment_tolerance(facet, 1.55, out / "fig9_alignment_tolerance.png")

    import meow as mw

    mw.save_table(
        out / "overlap_vs_wavelength",
        {"wavelength_nm": wls * 1000, "eta_te": te, "eta_tm": tm},
    )
    summary = {
        "eta_te_1550": etas.get("te"), "eta_tm_1550": etas.get("tm"),
        "eta_te_min": float(te.min()), "eta_tm_min": float(tm.min()),
        **tol,
    }
    mw.save_summary(out / "summary", summary)
    return {"out_dir": str(out), "summary": summary,
            "files": sorted(p.name for p in out.glob("*"))}


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
