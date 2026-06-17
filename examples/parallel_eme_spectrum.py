"""Parallelized EME spectra with dispersive (an)isotropic materials.

This example computes scattering-matrix element spectra of a stepped
silicon waveguide taper with ``meow.compute_s_matrix_spectrum``: the chain
of cells is split into overlapping slice groups and each concurrent job
(local thread/subprocess here; pass ``executor=mw.slurm_executor(...)`` to
run each group as a slurm task) runs the frequency-dependent mode
simulations of its slices at every sweep point, returning only the
per-frequency effective indices and interface S-matrices.

Material dispersion enters through the environment wavelength:

- the core uses ``mw.silicon``, a ``SampledMaterial`` with a tabulated
  n(lambda);
- a second device variant uses a ``SampledAnisotropicMaterial`` - a
  wavelength-sampled dielectric *tensor* (here a lithium-niobate-like
  negative uniaxial birefringence with artificial dispersion) - showing
  that dispersive tensors propagate through the same parallel pipeline.

The same sweep is run once as a function of wavelength and once as a
function of optical frequency (``freqs=`` in Hz), which by construction
yields identical S-matrices at corresponding points.

Run with: ``uv run python -m examples.parallel_eme_spectrum``
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

import meow as mw

FIGDIR = Path(__file__).parent / "figures"

WIDTHS = (0.40, 0.55, 0.70, 0.85)
CELL_LENGTH = 2.0
NUM_MODES = 3
WLS = np.linspace(1.45, 1.65, 9)


def dispersive_uniaxial_material() -> mw.SampledAnisotropicMaterial:
    """A lithium-niobate-like dispersive uniaxial dielectric tensor."""
    wls = np.linspace(1.3, 1.8, 11)
    no = 2.21 - 0.10 * (wls - 1.55)  # ordinary index, artificial dispersion
    ne = 2.13 - 0.14 * (wls - 1.55)  # extraordinary index
    return mw.SampledAnisotropicMaterial.from_n(
        "uniaxial_dispersive", wls, np.stack([ne, no, no], axis=1)
    )


def taper_cells(
    material: mw.Material, widths: tuple[float, ...] = WIDTHS
) -> list[mw.Cell]:
    """A small stepped-width taper with the given core material."""
    structs = [
        mw.Structure(
            material=material,
            geometry=mw.Box(
                x_min=-w / 2,
                x_max=w / 2,
                y_min=0.0,
                y_max=0.22,
                z_min=i * CELL_LENGTH,
                z_max=(i + 1) * CELL_LENGTH,
            ),
        )
        for i, w in enumerate(widths)
    ]
    mesh = mw.Mesh2D(
        x=np.linspace(-1.0, 1.0, 41),
        y=np.linspace(-0.4, 0.62, 35),
    )
    return [
        mw.Cell(
            structures=structs,
            mesh=mesh,
            z_min=i * CELL_LENGTH,
            z_max=(i + 1) * CELL_LENGTH,
        )
        for i in range(len(widths))
    ]


def transmission(spectra: list) -> np.ndarray:
    """|S_00|^2: fundamental-mode transmission at each sweep point."""
    return np.array(
        [
            float(np.abs(np.asarray(S)[pm["right@0"], pm["left@0"]]) ** 2)
            for S, pm in spectra
        ]
    )


def main() -> dict[str, float]:
    env = mw.Environment(wl=1.55, T=25.0)
    executor = ThreadPoolExecutor(max_workers=4)

    # device 1: dispersive isotropic silicon core, swept in wavelength
    cells_si = taper_cells(mw.silicon)
    spectra_si = mw.compute_s_matrix_spectrum(
        cells_si, env, wls=WLS, num_modes=NUM_MODES, executor=executor
    )

    # the same sweep as a function of optical frequency (identical physics)
    freqs = c / (WLS * 1e-6)
    spectra_si_f = mw.compute_s_matrix_spectrum(
        cells_si, env, freqs=freqs, num_modes=NUM_MODES, executor=executor
    )

    # device 2: dispersive uniaxial (tensor) core
    # wider waveguides: the lower-index uniaxial core needs more confinement
    cells_ln = taper_cells(
        dispersive_uniaxial_material(), widths=(0.7, 0.85, 1.0, 1.15)
    )
    spectra_ln = mw.compute_s_matrix_spectrum(
        cells_ln, env, wls=WLS, num_modes=NUM_MODES, executor=executor
    )

    t_si = transmission(spectra_si)
    t_si_f = transmission(spectra_si_f)
    t_ln = transmission(spectra_ln)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(WLS * 1e3, 10 * np.log10(t_si), "C0o-", label="silicon (sampled n)")
    axes[0].plot(
        WLS * 1e3,
        10 * np.log10(t_ln),
        "C3s-",
        label="uniaxial (sampled tensor)",
    )
    axes[0].set_xlabel("wavelength [nm]")
    axes[0].set_ylabel("$|S_{00}|^2$ [dB]")
    axes[0].set_title("EME transmission spectra (parallel slice-group jobs)")
    axes[0].legend(fontsize=8)
    axes[0].grid(visible=True)

    axes[1].plot(freqs * 1e-12, 10 * np.log10(t_si_f), "C0o-", label="freqs= sweep")
    axes[1].plot(
        freqs * 1e-12,
        10 * np.log10(t_si),
        "k.",
        ms=3,
        label="wls= sweep (same points)",
    )
    axes[1].set_xlabel("optical frequency [THz]")
    axes[1].set_ylabel("$|S_{00}|^2$ [dB]")
    axes[1].set_title("Same spectrum vs optical frequency")
    axes[1].legend(fontsize=8)
    axes[1].grid(visible=True)

    fig.suptitle("Parallelized EME S-matrix spectra with dispersive materials")
    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True, parents=True)
    fig.savefig(FIGDIR / "parallel_eme_spectrum.png", dpi=150)
    plt.close(fig)

    return {
        "t_si_min_db": float(np.min(10 * np.log10(t_si))),
        "t_ln_min_db": float(np.min(10 * np.log10(t_ln))),
        "wl_freq_max_diff": float(np.max(np.abs(t_si - t_si_f))),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2))
