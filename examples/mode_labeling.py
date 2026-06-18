"""Label waveguide modes by their Hermite-Gaussian character.

This example demonstrates :mod:`meow.mode_label`: every guided mode of a
multimode waveguide is fitted against a family of elliptical Hermite-Gaussian
(HG) templates -- of various orders and both TE/TM polarizations -- each
initially centered on the mode's field centroid with widths matching the
field's transverse variances and then refined to maximize overlap. The mode is
labeled by the ``(polarization, m, n)`` of the lowest-squared-error fit.

It runs on two horizontally-multimode cores:

- a Si3N4 strip (n=1.996) in SiO2, and
- a thin-film lithium niobate (TFLN) rib with the dispersive uniaxial
  dielectric tensor of congruent x-cut LiNbO3 (Zelmon 1997 Sellmeier).

Finally it shows how the labels let you *filter* for specific modes (e.g. only
the TE modes, or just the fundamental TE00) before building an EME model.

Run with::

    python examples/mode_labeling.py

It prints a label table for each waveguide and writes comparison figures to
``examples/figures/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import meow as mw
from meow.mode_label import (
    filter_modes_by_label,
    hermite_gaussian_field,
    label_mode,
    label_mode_candidates,
    label_modes,
)

WL = 1.55
SIO2_N = 1.444
FIG_DIR = Path(__file__).resolve().parent / "figures"


def ln_xcut() -> mw.SampledAnisotropicMaterial:
    """Dispersive x-cut congruent LiNbO3 tensor (Zelmon 1997 Sellmeier).

    Mode-plane axes: x is the crystal z (extraordinary) axis, so the tensor
    diagonal is ``(ne^2, no^2, no^2)``.
    """
    wls = np.linspace(0.4, 2.0, 33)
    wl2 = wls**2
    no2 = 1 + 2.6734 * wl2 / (wl2 - 0.01764) + 1.2290 * wl2 / (wl2 - 0.05914)
    no2 += 12.614 * wl2 / (wl2 - 474.6)
    ne2 = 1 + 2.9804 * wl2 / (wl2 - 0.02047) + 0.5981 * wl2 / (wl2 - 0.0666)
    ne2 += 8.9543 * wl2 / (wl2 - 416.08)
    return mw.SampledAnisotropicMaterial(
        name="ln_xcut_demo", wls=wls, eps=np.stack([ne2, no2, no2], axis=1)
    )


def solve_waveguide(
    core_material: mw.Material,
    *,
    core_w: float,
    core_h: float,
    num_modes: int,
    res: float = 0.04,
) -> mw.Modes:
    """Solve the modes of a simple rectangular core embedded in SiO2."""
    core = mw.Structure(
        material=core_material,
        geometry=mw.Box(
            x_min=-core_w / 2,
            x_max=core_w / 2,
            y_min=0.0,
            y_max=core_h,
            z_min=0.0,
            z_max=1.0,
        ),
        mesh_order=1,
    )
    clad = mw.Structure(
        material=mw.IndexMaterial(name="demo_sio2", n=SIO2_N),
        geometry=mw.Box(
            x_min=-6, x_max=6, y_min=-3, y_max=3 + core_h, z_min=0.0, z_max=1.0
        ),
        mesh_order=9,
    )
    mesh = mw.Mesh2D(
        x=np.arange(-3.5, 3.5 + res / 2, res),
        y=np.arange(-2.0, 2.0 + core_h + res / 2, res),
    )
    cell = mw.Cell(structures=[core, clad], mesh=mesh, z_min=0.0, z_max=1.0)
    env = mw.Environment(wl=WL, T=25.0)
    cs = mw.CrossSection.from_cell(cell=cell, env=env)
    return mw.compute_modes(cs, num_modes=num_modes)


def print_label_table(name: str, modes: mw.Modes) -> list:
    """Print a neff / te_fraction / HG-label table for a set of modes."""
    labels = label_modes(modes)
    print(f"\n{name}")
    print(f"{'#':>2}  {'neff':>8}  {'te_frac':>7}  {'label':>6}  {'error':>7}")
    print("  " + "-" * 38)
    for i, (mode, label) in enumerate(zip(modes, labels, strict=True)):
        print(
            f"{i:>2}  {mode.neff.real:>8.4f}  {mode.te_fraction:>7.3f}  "
            f"{label.name:>6}  {label.error:>7.4f}"
        )
    return labels


def plot_modes_vs_fits(name: str, modes: mw.Modes, labels: list, path: Path) -> None:
    """Plot each mode's dominant field next to its best-fit HG template."""
    n = len(modes)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.0), squeeze=False)
    for j, (mode, label) in enumerate(zip(modes, labels, strict=True)):
        if label.pol == "TE":
            field = np.real(mode.Ex)
            x, y = mode.mesh.Xx[:, 0], mode.mesh.Yx[0, :]
        else:
            field = np.real(mode.Ey)
            x, y = mode.mesh.Xy[:, 0], mode.mesh.Yy[0, :]
        hg = hermite_gaussian_field(
            x, y, label.m, label.n, label.x0, label.y0, label.wx, label.wy
        )
        # match the HG sign/scale to the field for display
        scale = np.sum(hg * field) / np.sum(hg * hg)
        hg = hg * scale

        vmax = np.abs(field).max() or 1.0
        X, Y = np.meshgrid(x, y, indexing="ij")
        for row, data, tag in ((0, field, "mode"), (1, hg, "HG fit")):
            ax = axes[row][j]
            ax.pcolormesh(X, Y, data, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{label.name}\nerr={label.error:.3f}", fontsize=9)
            if j == 0:
                ax.set_ylabel(tag, fontsize=9)
    fig.suptitle(f"{name}: dominant E-field (top) vs best-fit Hermite-Gaussian")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure -> {path}")


def demo_eme_filtering(modes: mw.Modes) -> None:
    """Filter for specific modes, then build an EME model from them."""
    print("\nFiltering modes for an EME model")
    te00 = filter_modes_by_label(modes, pol="TE", m=0, n=0, max_error=0.2)
    te_modes = filter_modes_by_label(modes, pol="TE", max_error=0.2)
    tm_modes = filter_modes_by_label(modes, pol="TM", max_error=0.2)
    te00_names = [label_mode(m).name for m in te00]
    te_names = [label_mode(m).name for m in te_modes]
    tm_names = [label_mode(m).name for m in tm_modes]
    print(f"  TE00 only : {len(te00)} mode -> {te00_names}")
    print(f"  all TE    : {len(te_modes)} modes -> {te_names}")
    print(f"  all TM    : {len(tm_modes)} modes -> {tm_names}")

    # Build a two-segment straight section using only the TE modes. In a real
    # EME model you would filter the modes of every cell the same way to keep
    # the simulation restricted to the polarization / orders of interest.
    S, port_map = mw.compute_s_matrix([te_modes, te_modes], cell_lengths=[10.0, 10.0])
    S = np.asarray(S)
    n = len(te_modes)
    i_in, i_out = port_map["left@0"], port_map["right@0"]
    t00 = np.abs(S[i_out, i_in]) ** 2
    print(
        f"  EME with {n} TE modes: TE00 transmission through 20 um straight = "
        f"{t00:.4f} (expected ~1.0)"
    )


def main() -> None:
    print("=" * 60)
    print("Hermite-Gaussian mode labeling")
    print("=" * 60)

    si3n4 = mw.IndexMaterial(name="si3n4_demo", n=1.996)
    si3n4_modes = solve_waveguide(si3n4, core_w=3.0, core_h=0.7, num_modes=10)
    si3n4_labels = print_label_table("Si3N4 strip (3.0 x 0.7 um)", si3n4_modes)
    plot_modes_vs_fits(
        "Si3N4", si3n4_modes[:6], si3n4_labels[:6], FIG_DIR / "mode_labeling_si3n4.png"
    )
    demo_eme_filtering(si3n4_modes)

    ln_modes = solve_waveguide(ln_xcut(), core_w=2.2, core_h=0.6, num_modes=10)
    ln_labels = print_label_table("x-cut LiNbO3 rib (2.2 x 0.6 um)", ln_modes)
    plot_modes_vs_fits(
        "LiNbO3", ln_modes[:6], ln_labels[:6], FIG_DIR / "mode_labeling_linbo3.png"
    )
    demo_eme_filtering(ln_modes)

    # Show the full candidate ranking for one mode, illustrating that the
    # lowest-squared-error template wins.
    print("\nCandidate ranking for the Si3N4 TE10 mode (top 5):")
    te10 = next(m for m in si3n4_modes if label_mode(m).name == "TE10")
    for c in label_mode_candidates(te10)[:5]:
        print(f"  {c.name}: error={c.error:.4f}  overlap={c.overlap:.4f}")


if __name__ == "__main__":
    main()
