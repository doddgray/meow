"""FDE Tidy3d backend (default backend for MEOW)."""

from __future__ import annotations

import importlib.util
import warnings
from collections.abc import Callable
from types import SimpleNamespace
from typing import Literal

import numpy as np
from pydantic import PositiveFloat, PositiveInt
from scipy.constants import c
from tidy3d.components.mode.solver import compute_modes as _compute_modes

from meow.cross_section import CrossSection
from meow.fde.post_process import post_process_modes
from meow.mode import Mode, Modes
from meow.settings import limit_threads, solver_threads

HAS_TIDY3D_EXTRAS = importlib.util.find_spec("tidy3d_extras") is not None
"""Whether the optional ``tidy3d-extras`` package is importable. It provides
tidy3d's local (subpixel) dielectric smoothing and the fully-tensorial mode
solver."""


def _enable_local_smoothing(local_smoothing: bool | None) -> None:  # noqa: FBT001
    """Enable tidy3d's local (subpixel) dielectric smoothing when requested.

    Args:
        local_smoothing:
            - ``None`` (default): auto - enable the smoothing iff
              ``tidy3d-extras`` is installed, otherwise do nothing;
            - ``True``: require ``tidy3d-extras`` and enable it (raise if the
              package is missing);
            - ``False``: leave tidy3d's smoothing preference untouched.

    When enabled, this sets ``tidy3d.config.simulation.use_local_subpixel`` so
    tidy3d's mode solver performs local dielectric smoothing. Note that meow
    additionally applies its own subpixel smoothing when building the
    permittivity (see ``CrossSection.subpixel_smoothing``).
    """
    if local_smoothing is False:
        return
    if not HAS_TIDY3D_EXTRAS:
        if local_smoothing is True:
            msg = (
                "local dielectric smoothing requires the optional 'tidy3d-extras' "
                "package; install it with: pip install 'tidy3d[extras]'"
            )
            raise ImportError(msg)
        return  # auto-mode and the package is unavailable: nothing to do
    try:
        import tidy3d as td  # fmt: skip

        td.config.simulation.use_local_subpixel = True
    except (ImportError, AttributeError) as e:  # pragma: no cover - defensive
        warnings.warn(
            f"could not enable tidy3d local dielectric smoothing: {e}", stacklevel=2
        )


def compute_modes_tidy3d(
    cs: CrossSection,
    num_modes: PositiveInt = 10,
    target_neff: PositiveFloat | None = None,
    precision: Literal["single", "double"] = "double",
    post_process: Callable = post_process_modes,
    *,
    local_smoothing: bool | None = None,
) -> Modes:
    """Compute ``Modes`` for a given ``CrossSection``.

    Args:
        cs: the cross-section to solve modes for.
        num_modes: number of modes to compute.
        target_neff: effective index near which to search for modes.
        precision: floating-point precision, ``"single"`` or ``"double"``.
        post_process: callable applied to the raw mode list before returning.
        local_smoothing: enable tidy3d's local (subpixel) dielectric smoothing
            via the optional ``tidy3d-extras`` package. ``None`` (default)
            enables it automatically when ``tidy3d-extras`` is installed;
            ``True`` requires it; ``False`` leaves the tidy3d preference alone.
            See :func:`_enable_local_smoothing`.

    Returns:
        The computed and post-processed collection of modes.
    """
    if num_modes < 1:
        msg = "You need to request at least 1 mode."
        raise ValueError(msg)

    _enable_local_smoothing(local_smoothing)

    eps_cross = [
        cs.nx**2,
        cs.eps_xy,
        cs.eps_xz,
        cs.eps_yx,
        cs.ny**2,
        cs.eps_yz,
        cs.eps_zx,
        cs.eps_zy,
        cs.nz**2,
    ]

    if np.isinf(cs.mesh.bend_radius) or np.isnan(cs.mesh.bend_radius):
        bend_radius = None
        bend_axis = None
    else:
        bend_radius = cs.mesh.bend_radius
        bend_axis = cs.mesh.bend_axis

    mode_spec = SimpleNamespace(  # tidy3d.ModeSpec alternative (prevents type checking)
        num_modes=num_modes,
        target_neff=target_neff,
        num_pml=cs.mesh.num_pml,
        filter_pol=None,
        angle_theta=cs.mesh.angle_theta,
        angle_phi=cs.mesh.angle_phi,
        bend_radius=bend_radius,
        precision=precision,
        bend_axis=bend_axis,
        track_freq="central",
        group_index_step=False,
    )

    with warnings.catch_warnings(), limit_threads(solver_threads()):
        warnings.filterwarnings("ignore", message=".*Input has data type int64.*")
        warnings.filterwarnings("ignore", message=".*divide by zero.*")
        warnings.filterwarnings("ignore", message=".*overflow encountered.*")
        warnings.filterwarnings("ignore", message=".*invalid value.*")
        ((Ex, Ey, Ez), (Hx, Hy, Hz)), neffs = (
            x.squeeze()
            for x in _compute_modes(
                eps_cross=eps_cross,
                coords=[cs.mesh.x, cs.mesh.y],
                freq=c / (cs.env.wl * 1e-6),
                mode_spec=mode_spec,
                precision=precision,
                plane_center=cs.mesh.plane_center,
            )[:2]
        )

    if num_modes == 1:
        modes = [
            Mode(
                cs=cs,
                Ex=Ex,
                Ey=Ey,
                Ez=Ez,
                Hx=Hx,
                Hy=Hy,
                Hz=Hz,
                neff=np.asarray(neffs, dtype=np.complex128).item(),
            )
            for _ in range(num_modes)
        ]
    else:  # num_modes > 1
        modes = [
            Mode(
                cs=cs,
                Ex=Ex[..., i],
                Ey=Ey[..., i],
                Ez=Ez[..., i],
                Hx=Hx[..., i],
                Hy=Hy[..., i],
                Hz=Hz[..., i],
                neff=neffs[i],
            )
            for i in range(num_modes)
        ]

    modes = sorted(modes, key=lambda m: float(np.real(m.neff)), reverse=True)
    return post_process(modes)
