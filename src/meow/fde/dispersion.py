"""Backend-agnostic modal dispersion metrics (group index, GVD, ...).

Group index and group-velocity dispersion are spectral quantities: they
require the effective index in a small wavelength neighborhood. These
helpers evaluate ``neff(wl)`` with any mode-solver backend (tidy3d, MPB,
Lumerical, ...) by re-solving the cross-section at ``wl`` and ``wl +- dwl``
and central-differencing, which gives every backend the same dispersion
feature set.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.constants import c

from meow.cell import Cell
from meow.cross_section import CrossSection
from meow.environment import Environment
from meow.mode import Mode


def effective_area(mode: Mode) -> float:
    """The (nonlinear) effective mode area of a `Mode` [um^2].

    Uses the standard nonlinear-optics definition

    ``A_eff = (integral |E|^2 dA)^2 / integral |E|^4 dA``

    with ``|E|^2 = |Ex|^2 + |Ey|^2 + |Ez|^2``. The integrals assume the
    (locally) uniform grid spacing of the mode's mesh.
    """
    intensity = np.abs(mode.Ex) ** 2 + np.abs(mode.Ey) ** 2 + np.abs(mode.Ez) ** 2
    dA = float(np.mean(mode.mesh.dx)) * float(np.mean(mode.mesh.dy))
    denom = float(np.sum(intensity**2))
    if denom == 0.0:
        return 0.0
    return float(dA * np.sum(intensity) ** 2 / denom)


@dataclass
class ModeMetrics:
    """Spectral and modal metrics of a waveguide mode at one wavelength."""

    wl: float
    """The vacuum wavelength [um]."""
    neff: float
    """The (real) effective index."""
    group_index: float
    """The group index ``ng = neff - wl * dneff/dwl``."""
    beta2: float
    """Group-velocity dispersion ``beta2 = d^2(beta)/d(omega)^2`` [s^2/m]."""
    dispersion_D: float
    """Dispersion parameter ``D = -(wl/c) d^2(neff)/d(wl)^2`` [ps/(nm km)]."""
    effective_area: float
    """The (nonlinear) effective mode area [um^2]."""
    te_fraction: float
    """The TE polarization fraction."""


def solve_mode(
    structures: list[Any],
    wl: float,
    mesh: Any,
    *,
    num_modes: int = 2,
    mode_index: int = 0,
    compute_modes: Callable | None = None,
    env_kwargs: dict[str, Any] | None = None,
    **compute_kwargs: Any,
) -> Mode:
    """Solve a single mode of a cross-section at one wavelength.

    Args:
        structures: the 3D structures defining the cross-section.
        wl: the vacuum wavelength [um].
        mesh: the 2D mesh to discretize with.
        num_modes: number of modes to compute.
        mode_index: which mode (sorted by decreasing neff) to return.
        compute_modes: the mode-solver backend
            (default: ``meow.fde.compute_modes``, i.e. tidy3d).
        env_kwargs: extra environment parameters (e.g. ``T``).
        **compute_kwargs: extra kwargs for the backend.
    """
    if compute_modes is None:
        from meow.fde import compute_modes  # fmt: skip
    cell = Cell(structures=structures, mesh=mesh, z_min=0.0, z_max=1.0)
    env = Environment(wl=wl, **({"T": 25.0} | (env_kwargs or {})))
    cs = CrossSection.from_cell(cell=cell, env=env)
    modes = compute_modes(cs, num_modes=num_modes, **compute_kwargs)
    return modes[mode_index]


def dispersion_metrics(
    structures: list[Any],
    wl: float,
    mesh: Any,
    *,
    dwl: float = 0.005,
    num_modes: int = 2,
    mode_index: int = 0,
    compute_modes: Callable | None = None,
    env_kwargs: dict[str, Any] | None = None,
    **compute_kwargs: Any,
) -> ModeMetrics:
    """Compute neff, group index, GVD, effective area and TE fraction.

    The effective index is evaluated at ``wl`` and ``wl +- dwl`` (so any
    material dispersion of the structures' materials is included) and
    differentiated centrally:

    - ``ng = neff - wl * dneff/dwl``
    - ``D = -(wl / c) * d^2 neff / dwl^2``  (in ps/(nm km))
    - ``beta2 = wl^3 / (2 pi c^2) * d^2 neff / dwl^2``  (in s^2/m)

    Works with any backend passed as ``compute_modes``; the same mode index
    (sorted by decreasing effective index) is followed at all three
    wavelengths.
    """

    def solve(w: float) -> Mode:
        return solve_mode(
            structures,
            w,
            mesh,
            num_modes=num_modes,
            mode_index=mode_index,
            compute_modes=compute_modes,
            env_kwargs=env_kwargs,
            **compute_kwargs,
        )

    mode_m = solve(wl - dwl)
    mode_0 = solve(wl)
    mode_p = solve(wl + dwl)

    n_m, n_0, n_p = (float(np.real(m.neff)) for m in (mode_m, mode_0, mode_p))
    dn_dwl = (n_p - n_m) / (2 * dwl)  # [1/um]
    d2n_dwl2 = (n_p - 2 * n_0 + n_m) / dwl**2  # [1/um^2]

    ng = n_0 - wl * dn_dwl
    # SI conversions: wl [um] -> [m]
    wl_m = wl * 1e-6
    d2n_dwl2_si = d2n_dwl2 * 1e12  # [1/m^2]
    beta2 = wl_m**3 / (2 * np.pi * c**2) * d2n_dwl2_si  # [s^2/m]
    D_si = -(wl_m / c) * d2n_dwl2_si  # [s/m^2]
    D = D_si * 1e6  # [ps/(nm km)]

    return ModeMetrics(
        wl=wl,
        neff=n_0,
        group_index=float(ng),
        beta2=float(beta2),
        dispersion_D=float(D),
        effective_area=effective_area(mode_0),
        te_fraction=mode_0.te_fraction,
    )
