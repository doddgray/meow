"""FDE MPB backend (MIT Photonic Bands, via the ``meep.mpb`` bindings).

This backend solves the waveguide modes of a meow ``CrossSection`` with
MPB's plane-wave eigensolver: at a fixed vacuum wavelength it runs
``find_k`` for ``k`` along the propagation axis (``neff = k / omega``) and
extracts the corresponding E and H fields.

The dielectric tensor (including anisotropy with off-diagonal components,
e.g. from rotated uniaxial materials, and angled-sidewall geometry through
the cross-section's smoothed permittivity arrays) is passed to MPB through
a material function that samples the exact same permittivity arrays used
by the tidy3d backend, so the two backends solve nominally identical
problems. MPB is a lossless (real-epsilon) solver: imaginary permittivity
parts are ignored.

Requires the ``meep``/``mpb`` python bindings, which are available from
conda-forge (``conda install -c conda-forge pymeep``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import PositiveFloat, PositiveInt
from scipy.constants import epsilon_0, mu_0

from meow.cross_section import CrossSection
from meow.fde.post_process import post_process_modes
from meow.mode import Mode, Modes


def compute_modes_mpb(
    cs: CrossSection,
    num_modes: PositiveInt = 10,
    target_neff: PositiveFloat | None = None,
    resolution: PositiveFloat | None = None,
    tolerance: float = 1e-9,
    post_process: Callable = post_process_modes,
) -> Modes:
    """Compute ``Modes`` for a given ``CrossSection`` with MPB.

    Args:
        cs: the cross-section to solve modes for.
        num_modes: number of modes (bands) to compute.
        target_neff: effective index used as the ``find_k`` starting guess.
        resolution: MPB grid resolution in points per um (default: matched
            to the cross-section mesh spacing).
        tolerance: ``find_k`` convergence tolerance.
        post_process: callable applied to the raw mode list before returning.

    Returns:
        The computed and post-processed collection of modes.
    """
    try:
        import meep as mp
        from meep import mpb
    except ImportError as e:
        msg = (
            "The MPB backend requires the meep/mpb python bindings. "
            "Install them with: conda install -c conda-forge pymeep"
        )
        raise ImportError(msg) from e

    if num_modes < 1:
        msg = "You need to request at least 1 mode."
        raise ValueError(msg)

    mesh = cs.mesh
    x_min, x_max = float(mesh.x.min()), float(mesh.x.max())
    y_min, y_max = float(mesh.y.min()), float(mesh.y.max())
    Lx, Ly = x_max - x_min, y_max - y_min
    x0, y0 = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)

    if resolution is None:
        resolution = 1.0 / float(np.mean(mesh.dx))

    material_func, n_max = _make_material_func(cs, x0, y0)

    omega = 1.0 / float(cs.env.wl)  # MPB units: a = 1 um
    n_guess = float(target_neff) if target_neff else 0.95 * n_max

    ms = mpb.ModeSolver(
        geometry_lattice=mp.Lattice(size=mp.Vector3(Lx, Ly)),
        default_material=material_func,
        resolution=resolution,
        num_bands=int(num_modes),
        tolerance=1e-7,
    )

    ks = ms.find_k(
        mp.NO_PARITY,
        omega,
        1,
        int(num_modes),
        mp.Vector3(0, 0, 1),
        tolerance,
        n_guess * omega,
        max(0.1 * omega, 0.05),
        n_max * omega,
    )

    # find_k re-inits the underlying C solver band-by-band and leaves it in a
    # single-band state; re-initialize with all bands for field extraction.
    ms.num_bands = int(num_modes)
    ms.init_params(mp.NO_PARITY, True)  # noqa: FBT003

    # MPB grid coordinates (cell-centered) for interpolation onto our mesh
    eps_grid = ms.get_epsilon()
    nx_mp, ny_mp = eps_grid.shape[0], eps_grid.shape[1]
    xs_mp = np.linspace(x_min, x_max, nx_mp, endpoint=False)
    xs_mp += 0.5 * (xs_mp[1] - xs_mp[0])
    ys_mp = np.linspace(y_min, y_max, ny_mp, endpoint=False)
    ys_mp += 0.5 * (ys_mp[1] - ys_mp[0])

    modes = []
    for band, k in enumerate(ks, start=1):
        neff = float(k) / omega
        ms.solve_kpoint(mp.Vector3(0, 0, float(k)))
        E = np.squeeze(np.asarray(ms.get_efield(band, bloch_phase=False)))
        H = np.squeeze(np.asarray(ms.get_hfield(band, bloch_phase=False)))
        # MPB uses natural units (eps0 = mu0 = 1) where |H| ~ |E| for a mode;
        # meow expects the SI relative scale H ~ E / eta0.
        H = H * np.sqrt(epsilon_0 / mu_0)
        fields = {}
        for name, arr in [
            ("Ex", E[..., 0]),
            ("Ey", E[..., 1]),
            ("Ez", E[..., 2]),
            ("Hx", H[..., 0]),
            ("Hy", H[..., 1]),
            ("Hz", H[..., 2]),
        ]:
            fields[name] = _interp_to(arr, xs_mp, ys_mp, *_positions(mesh, name))
        modes.append(Mode(cs=cs, neff=complex(neff), **fields))

    modes = sorted(modes, key=lambda m: float(np.real(m.neff)), reverse=True)
    return post_process(modes)


def _make_material_func(
    cs: CrossSection, x0: float, y0: float
) -> tuple[Callable, float]:
    """Build an MPB material function sampling the cross-section's eps arrays.

    The permittivity arrays (including off-diagonal anisotropy) are the same
    smoothed arrays the tidy3d backend consumes; each MPB sample point takes
    the value of the nearest mesh cell. Also returns the maximum refractive
    index (for the ``find_k`` bracket).
    """
    import meep as mp  # fmt: skip

    mesh = cs.mesh
    eps_xx = np.real(np.asarray(cs.nx) ** 2)
    eps_yy = np.real(np.asarray(cs.ny) ** 2)
    eps_zz = np.real(np.asarray(cs.nz) ** 2)
    eps_xy = np.real(np.asarray(cs.eps_xy))
    eps_xz = np.real(np.asarray(cs.eps_xz))
    eps_yz = np.real(np.asarray(cs.eps_yz))
    xs = np.asarray(mesh.x_[:], dtype=float)  # cell centers
    ys = np.asarray(mesh.y_[:], dtype=float)

    def _nearest(coords: np.ndarray, v: float) -> int:
        i = int(np.clip(np.searchsorted(coords, v), 0, coords.shape[0] - 1))
        if i > 0 and abs(coords[i - 1] - v) <= abs(coords[i] - v):
            i -= 1
        return i

    def _sample(arr: np.ndarray, x: float, y: float) -> float:
        return float(arr[_nearest(xs, x), _nearest(ys, y)])

    def material_func(p: Any) -> Any:
        x, y = p.x + x0, p.y + y0
        diag = mp.Vector3(
            _sample(eps_xx, x, y), _sample(eps_yy, x, y), _sample(eps_zz, x, y)
        )
        offdiag = mp.Vector3(
            _sample(eps_xy, x, y), _sample(eps_xz, x, y), _sample(eps_yz, x, y)
        )
        return mp.Medium(epsilon_diag=diag, epsilon_offdiag=offdiag)

    n_max = float(np.sqrt(max(eps_xx.max(), eps_yy.max(), eps_zz.max())))
    return material_func, n_max


def _positions(mesh: Any, field: str) -> tuple[np.ndarray, np.ndarray]:
    """The meow Yee-grid positions of a given field component."""
    X, Y = {
        "Ex": (mesh.Xx, mesh.Yx),
        "Ey": (mesh.Xy, mesh.Yy),
        "Ez": (mesh.Xz, mesh.Yz),
        "Hx": (mesh.Xy, mesh.Yy),
        "Hy": (mesh.Xx, mesh.Yx),
        "Hz": (mesh.Xz_, mesh.Yz_),
    }[field]
    return np.asarray(X[:, 0]), np.asarray(Y[0, :])


def _interp_to(
    arr: np.ndarray,
    xs_src: np.ndarray,
    ys_src: np.ndarray,
    xs_dst: np.ndarray,
    ys_dst: np.ndarray,
) -> np.ndarray:
    """Bilinearly interpolate a complex MPB field onto meow grid positions."""
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator(
        (xs_src, ys_src), arr, bounds_error=False, fill_value=0.0
    )
    X, Y = np.meshgrid(xs_dst, ys_dst, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)
    return interp(pts).reshape(X.shape).astype(np.complex128)
