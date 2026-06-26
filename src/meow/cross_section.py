"""A CrossSection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Self

import numpy as np
import shapely
from pydantic import Field, PrivateAttr
from shapely.ops import unary_union

from meow.arrays import ComplexArray2D, IntArray2D
from meow.base_model import BaseModel, cached_property
from meow.cell import (
    Cell,
    _classify_structures_by_mesh_order_and_material,
    _create_full_material_array,
    sort_structures,
)
from meow.environment import Environment
from meow.materials import Material
from meow.mesh import Mesh2D
from meow.structures import Structure2D


class CrossSection(BaseModel):
    """A `CrossSection` is built from a `Cell` with an `Environment`.

    This uniquely defines the refractive index everywhere.
    """

    structures: list[Structure2D] = Field(
        description="the 2D structures in the CrossSection"
    )
    mesh: Mesh2D = Field(description="the mesh to discretize the structures with")
    env: Environment = Field(
        description="the environment for which the cross section was calculated"
    )
    subpixel_smoothing: bool = Field(
        default=True,
        description="use subpixel smoothing at interfaces; if False: winner-takes-all",
    )
    smoothing_method: Literal["axis", "kottke"] = Field(
        default="axis",
        description=(
            "subpixel-smoothing scheme: 'axis' (per-component arithmetic/harmonic, "
            "the default) or 'kottke' (Farjadpour/Kottke tensor smoothing using the "
            "interface normal - second-order accurate for tilted interfaces; "
            "isotropic/diagonal materials only)"
        ),
    )
    _cell: Cell | None = PrivateAttr(default=None)

    @classmethod
    def from_cell(
        cls,
        *,
        cell: Cell,
        env: Environment,
        subpixel_smoothing: bool = True,
        smoothing_method: Literal["axis", "kottke"] = "axis",
    ) -> Self:
        """Create a CrossSection from a Cell and Environment."""
        obj = cls(
            structures=cell.structures_2d,
            mesh=cell.mesh,
            env=env,
            subpixel_smoothing=subpixel_smoothing,
            smoothing_method=smoothing_method,
        )
        # keep a handle on the originating cell so the (geometry-only) subpixel
        # smoothing plan can be cached there and reused across a wavelength sweep
        obj._cell = cell
        return obj

    @classmethod
    def from_index_arrays(
        cls,
        *,
        mesh: Mesh2D,
        env: Environment,
        nx: ComplexArray2D,
        ny: ComplexArray2D | None = None,
        nz: ComplexArray2D | None = None,
    ) -> Self:
        """Build a CrossSection directly from per-component index arrays.

        Bypasses polygon rasterization/smoothing: the (complex) refractive index
        on the ``Ex``/``Ey``/``Ez`` Yee positions is supplied directly. This is
        the entry point for *continuous* permittivity distributions - e.g. a
        differentiable level-set / density field for inverse design (see
        :mod:`meow.levelset`) - which have no polygonal structures.

        Args:
            mesh: the mesh the arrays are defined on. ``nx`` must have the shape
                of ``mesh.Xx`` (``ny`` of ``mesh.Xy``, ``nz`` of ``mesh.Xz``).
            env: the simulation environment.
            nx: refractive index on the ``Ex`` positions.
            ny: index on the ``Ey`` positions (defaults to ``nx``).
            nz: index on the ``Ez`` positions (defaults to ``nx``).

        Returns:
            A CrossSection whose ``nx``/``ny``/``nz`` return the given arrays
            (off-diagonal permittivity components are zero / isotropic).
        """
        obj = cls(structures=[], mesh=mesh, env=env, subpixel_smoothing=False)
        nxa = np.asarray(nx, dtype=np.complex128)
        nya = nxa if ny is None else np.asarray(ny, dtype=np.complex128)
        nza = nxa if nz is None else np.asarray(nz, dtype=np.complex128)
        obj._cache["nx"] = nxa
        obj._cache["ny"] = nya
        obj._cache["nz"] = nza
        return obj

    @cached_property
    def materials(self) -> dict[Material, int]:
        """Return a dictionary mapping materials to their indices."""
        materials: dict[Material, int] = {}
        for i, structure in enumerate(sort_structures(self.structures), start=1):
            if structure.material not in materials:
                materials[structure.material] = i
        return materials

    @cached_property
    def _m_full(self) -> IntArray2D:
        """Return the material index array for the full mesh."""
        return _create_full_material_array(self.mesh, self.structures, self.materials)

    @cached_property
    def n_full(self) -> ComplexArray2D:
        """Return the refractive index array for the full mesh."""
        n_full = np.ones_like(self.mesh.X_full, dtype=np.complex128)
        for material, idx in self.materials.items():
            n_full = np.where(self._m_full == idx, material(self.env), n_full)
        return n_full

    def _smoothing_plan_for(self, component: Literal["x", "y", "z"]) -> _SmoothingPlan:
        """The (cached) wavelength-independent subpixel-smoothing plan.

        The plan depends only on the geometry, so when this cross-section was
        built from a cell it is cached on that cell and reused across a whole
        wavelength/temperature sweep (the expensive shapely work runs once).
        """
        cell = self._cell
        key = f"_smoothing_plan_{component}"
        if cell is not None and key in cell._cache:
            return cell._cache[key]
        plan = _smoothing_plan(
            self.mesh, self._m_full, self.materials, self.structures, component
        )
        if cell is not None:
            cell._cache[key] = plan
        return plan

    def _kottke_plan_for(self, component: Literal["x", "y", "z"]) -> _KottkePlan:
        """The (cell-cached) wavelength-independent Kottke smoothing plan."""
        cell = self._cell
        key = f"_kottke_plan_{component}"
        if cell is not None and key in cell._cache:
            return cell._cache[key]
        plan = _kottke_plan(
            self.mesh, self._m_full, self.materials, self.structures, component
        )
        if cell is not None:
            cell._cache[key] = plan
        return plan

    def _smoothed_n(self, component: Literal["x", "y", "z"]) -> ComplexArray2D:
        """Smoothed index on a component's Yee positions (dispatch on method)."""
        if not self.subpixel_smoothing:
            return _compute_winner_takes_all_n(
                self.mesh, self._m_full, self.materials, self.env, self.structures,
                component,
            )
        if self.smoothing_method == "kottke":
            return _assemble_kottke(
                self._kottke_plan_for(component), self.materials, self.env, component
            )
        return _assemble_smoothed_n(
            self._smoothing_plan_for(component), self.materials, self.env, component
        )

    @cached_property
    def nx(self) -> ComplexArray2D:
        """Return the smoothed refractive index on the Ex positions."""
        return self._smoothed_n("x")

    @cached_property
    def ny(self) -> ComplexArray2D:
        """Return the smoothed refractive index on the Ey positions."""
        return self._smoothed_n("y")

    @cached_property
    def nz(self) -> ComplexArray2D:
        """Return the smoothed refractive index on the Ez positions."""
        return self._smoothed_n("z")

    @cached_property
    def eps_xy(self) -> ComplexArray2D:
        """Return the off-diagonal permittivity eps_xy on the Ex positions."""
        return _compute_offdiag_eps(self._m_full, self.materials, self.env, "x", "y")

    @cached_property
    def eps_xz(self) -> ComplexArray2D:
        """Return the off-diagonal permittivity eps_xz on the Ex positions."""
        return _compute_offdiag_eps(self._m_full, self.materials, self.env, "x", "z")

    @cached_property
    def eps_yx(self) -> ComplexArray2D:
        """Return the off-diagonal permittivity eps_yx on the Ey positions."""
        return _compute_offdiag_eps(self._m_full, self.materials, self.env, "y", "x")

    @cached_property
    def eps_yz(self) -> ComplexArray2D:
        """Return the off-diagonal permittivity eps_yz on the Ey positions."""
        return _compute_offdiag_eps(self._m_full, self.materials, self.env, "y", "z")

    @cached_property
    def eps_zx(self) -> ComplexArray2D:
        """Return the off-diagonal permittivity eps_zx on the Ez positions."""
        return _compute_offdiag_eps(self._m_full, self.materials, self.env, "z", "x")

    @cached_property
    def eps_zy(self) -> ComplexArray2D:
        """Return the off-diagonal permittivity eps_zy on the Ez positions."""
        return _compute_offdiag_eps(self._m_full, self.materials, self.env, "z", "y")

    def _visualize(
        self,
        *,
        ax: Any = None,
        n_cmap: Any = None,
        cbar: bool = True,
        show: bool = True,
        style: Literal["polygons", "pixelated"] = "polygons",
        **ignored: Any,
    ) -> None:
        """Visualize the cross section.

        Args:
            ax: the matplotlib axes to plot on (default: current axes).
            n_cmap: colormap mapping refractive index to color.
            cbar: add a colorbar mapping colors to material indices.
            show: call plt.show() after plotting.
            style: "polygons" draws the exact structure polygons (including
                angled sidewalls) as vector graphics; "pixelated" draws the
                rasterized refractive index on the full mesh grid.
            **ignored: extra ignored kwargs (`debug_grid=True` implies the
                "pixelated" style and draws the half-integer mesh grid).
        """
        if ignored.get("debug_grid", False):
            style = "pixelated"
        if style == "polygons":
            self._visualize_polygons(ax=ax, n_cmap=n_cmap, cbar=cbar, show=show)
        else:
            self._visualize_pixelated(
                ax=ax, n_cmap=n_cmap, cbar=cbar, show=show, **ignored
            )

    def _visualize_polygons(
        self,
        *,
        ax: Any = None,
        n_cmap: Any = None,
        cbar: bool = True,
        show: bool = True,
        edgecolor: str | None = "black",
        linewidth: float = 0.5,
    ) -> None:
        """Visualize the cross section as exact structure polygons.

        Unlike the pixelated visualization, this draws the actual projected
        2D geometries (clipped to the mesh bounds), so features smaller than
        the mesh resolution - such as angled sidewalls - are rendered exactly.
        """
        import matplotlib.pyplot as plt  # fmt: skip
        from matplotlib import colors  # fmt: skip
        from matplotlib.cm import ScalarMappable  # fmt: skip
        from matplotlib.patches import PathPatch, Rectangle  # fmt: skip
        from mpl_toolkits.axes_grid1 import make_axes_locatable  # fmt: skip

        if n_cmap is None:
            n_cmap = colors.LinearSegmentedColormap.from_list(
                name="c_cmap", colors=["#ffffff", "#86b5dc"]
            )
        if ax is not None:
            plt.sca(ax)
        else:
            ax = plt.gca()

        x_min, x_max = float(self.mesh.x.min()), float(self.mesh.x.max())
        y_min, y_max = float(self.mesh.y.min()), float(self.mesh.y.max())
        mesh_bounds = shapely.box(x_min, y_min, x_max, y_max)

        mat_n: dict[int, float] = {0: 1.0}  # background: air
        mat_name: dict[int, str] = {0: "air"}
        for material, idx in self.materials.items():
            mat_n[idx] = float(np.real(material(self.env)))
            mat_name[idx] = material.name

        vmin, vmax = min(mat_n.values()), max(mat_n.values())
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        ax.add_patch(
            Rectangle(
                xy=(x_min, y_min),
                width=x_max - x_min,
                height=y_max - y_min,
                facecolor=n_cmap(norm(mat_n[0])),
                edgecolor="none",
                zorder=0,
            )
        )

        effective_polys = _effective_material_polygons(self.structures, self.materials)
        for idx, poly in effective_polys.items():
            clipped = shapely.intersection(poly, mesh_bounds)
            path = _shapely_to_path(clipped)
            if path is None:
                continue
            ax.add_patch(
                PathPatch(
                    path,
                    facecolor=n_cmap(norm(mat_n[idx])),
                    edgecolor=edgecolor or "none",
                    linewidth=linewidth,
                )
            )

        plt.axis("scaled")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        plt.grid(visible=True)

        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            values = sorted(set(mat_n.values()))
            mappable = ScalarMappable(norm=norm, cmap=n_cmap)
            _cbar = plt.colorbar(mappable, ticks=values, cax=cax)
            names = {v: [] for v in values}
            for idx, v in mat_n.items():
                names[v].append(mat_name[idx])
            labels = [f"{'|'.join(names[v])} {v:.3f}" for v in values]
            _cbar.ax.set_yticklabels(labels, rotation=90, va="center", ha="center")
            plt.sca(ax)

        if show:
            plt.show()

    def _visualize_pixelated(
        self,
        *,
        ax: Any = None,
        n_cmap: Any = None,
        cbar: bool = True,
        show: bool = True,
        **ignored: Any,
    ) -> None:
        import matplotlib.pyplot as plt  # fmt: skip
        from matplotlib import colors  # fmt: skip
        from mpl_toolkits.axes_grid1 import make_axes_locatable  # fmt: skip

        debug_grid = ignored.pop("debug_grid", False)
        if n_cmap is None:
            n_cmap = colors.LinearSegmentedColormap.from_list(
                name="c_cmap", colors=["#ffffff", "#86b5dc"]
            )
        if ax is not None:
            plt.sca(ax)
        else:
            ax = plt.gca()
        n_full = np.real(self.n_full).copy()
        n_full[0, 0] = 1.0
        plt.pcolormesh(self.mesh.X_full, self.mesh.Y_full, n_full, cmap=n_cmap)
        plt.axis("scaled")
        if not debug_grid:
            plt.grid(visible=True)
        else:
            dx = self.mesh.dx
            dy = self.mesh.dy
            x_ticks = np.sort(np.unique(self.mesh.X_full.ravel()))[::2]
            y_ticks = np.sort(np.unique(self.mesh.Y_full.ravel()))[::2]
            plt.xticks(x_ticks - 0.25 * dx, ["" for _ in x_ticks - 0.25 * dx])
            plt.yticks(y_ticks - 0.25 * dy, ["" for _ in y_ticks - 0.25 * dy])
            plt.xticks(
                x_ticks + 0.25 * dx, ["" for _ in x_ticks + 0.25 * dx], minor=True
            )
            plt.yticks(
                y_ticks + 0.25 * dy, ["" for _ in y_ticks + 0.25 * dy], minor=True
            )
            plt.grid(visible=True, which="major", ls="-")
            plt.grid(visible=True, which="minor", ls=":")
        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            values = np.unique(np.real(self.n_full))
            _cbar = plt.colorbar(ticks=values, cax=cax)
            # material_names = ['air'] + [mat.name for mat in self.cell.materials]
            # labels = [f"\n{n}\n{v:.3f}" for n, v in zip(material_names, values)]
            labels = [f"{v:.3f}" for v in values]
            _cbar.ax.set_yticklabels(labels, rotation=90, va="center", ha="center")
            plt.sca(ax)
        if show:
            plt.show()


# --- Subpixel permittivity smoothing ---

_COMPONENT_SLICES = {
    "x": (slice(1, None, 2), slice(None, None, 2)),  # Ex: [1::2, ::2]
    "y": (slice(None, None, 2), slice(1, None, 2)),  # Ey: [::2, 1::2]
    "z": (slice(None, None, 2), slice(None, None, 2)),  # Ez: [::2, ::2]
}

_COMPONENT_INDEX = {"x": 0, "y": 1, "z": 2}


def _material_eps_component(
    material: Material,
    env: Environment,
    row: Literal["x", "y", "z"],
    col: Literal["x", "y", "z"],
) -> np.complex128:
    """Get a single component of the material permittivity tensor."""
    i, j = _COMPONENT_INDEX[row], _COMPONENT_INDEX[col]
    return np.complex128(material.eps_tensor(env)[i, j])


def _compute_offdiag_eps(
    m_full: IntArray2D,
    materials: dict[Material, int],
    env: Environment,
    row: Literal["x", "y", "z"],
    col: Literal["x", "y", "z"],
) -> ComplexArray2D:
    """Compute an off-diagonal permittivity component eps_{row,col}.

    The component is sampled at the E_{row} Yee positions without subpixel
    smoothing (each pixel takes the value of the material that rasterized
    onto it). The background (air) contributes zero off-diagonal permittivity.
    """
    si, sj = _COMPONENT_SLICES[row]
    m_comp = m_full[si, sj]
    eps = np.zeros_like(m_comp, dtype=np.complex128)
    for material, idx in materials.items():
        e = _material_eps_component(material, env, row, col)
        if e != 0:
            eps[m_comp == idx] = e
    return eps


def _dual_cell_bounds(
    mesh: Mesh2D,
    component: Literal["x", "y", "z"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (x_lo, x_hi, y_lo, y_hi) for the dual cells of a field component.

    Each field component lives at a specific position on the Yee grid. Its dual
    cell is the area over which the effective permittivity should be averaged.

    Full-grid indices (in x_full/y_full):
      Ex at (2i+1, 2j):
        dual cell x=[x_full[2i], x_full[2i+2]], y=[y_full[2j-1], y_full[2j+1]]
      Ey at (2i, 2j+1):
        dual cell x=[x_full[2i-1], x_full[2i+1]], y=[y_full[2j], y_full[2j+2]]
      Ez at (2i, 2j):
        dual cell x=[x_full[2i-1], x_full[2i+1]], y=[y_full[2j-1], y_full[2j+1]]
    """
    xf = mesh.x_full
    yf = mesh.y_full

    si, sj = _COMPONENT_SLICES[component]
    # Number of component grid points
    len(range(*si.indices(len(xf))))
    len(range(*sj.indices(len(yf))))

    # Full-grid indices for this component
    fi = np.arange(len(xf))[si]  # shape (ni,)
    fj = np.arange(len(yf))[sj]  # shape (nj,)

    def _bounds(vals: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo = np.empty(len(indices))
        hi = np.empty(len(indices))
        for k, idx in enumerate(indices):
            if idx > 0:
                lo[k] = vals[idx - 1]
            else:
                lo[k] = vals[0]
            if idx < len(vals) - 1:
                hi[k] = vals[idx + 1]
            else:
                hi[k] = vals[-1]
        return lo, hi

    x_lo, x_hi = _bounds(xf, fi)
    y_lo, y_hi = _bounds(yf, fj)

    return x_lo, x_hi, y_lo, y_hi


@dataclass
class _SmoothingPlan:
    """The wavelength-independent part of subpixel smoothing for one component.

    Everything here depends only on the geometry (mesh + structures), not on the
    environment, so it can be computed once (the expensive shapely interface /
    area-fraction work) and reused at every wavelength/temperature. Only the
    cheap material-permittivity assembly in :func:`_assemble_smoothed_n` is
    wavelength-dependent.
    """

    m_comp: IntArray2D
    """Material-index array on this component's Yee positions."""
    has_interface: bool
    """Whether any interface pixels need smoothing."""
    ii: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    """Row indices of the interface pixels."""
    jj: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    """Column indices of the interface pixels."""
    fractions: dict[int, np.ndarray] = field(default_factory=dict)
    """Per-material area fraction at each interface pixel (key 0 is background)."""
    use_harmonic: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    """Whether each interface pixel uses harmonic (vs arithmetic) averaging."""


def _smoothing_plan(
    mesh: Mesh2D,
    m_full: IntArray2D,
    materials: dict[Material, int],
    structures: list[Structure2D],
    component: Literal["x", "y", "z"],
) -> _SmoothingPlan:
    """Compute the wavelength-independent subpixel-smoothing plan for a component.

    This does the geometry-only work - interface detection, dual-cell area
    fractions (the expensive shapely intersections) and the harmonic/arithmetic
    orientation masks - so it can be cached and reused across a wavelength sweep.
    """
    si, sj = _COMPONENT_SLICES[component]
    m_comp = m_full[si, sj]

    # Identify interface pixels (where material differs from any 4-neighbor)
    padded = np.pad(m_comp, 1, mode="edge")
    is_interface = (
        (m_comp != padded[:-2, 1:-1])  # top
        | (m_comp != padded[2:, 1:-1])  # bottom
        | (m_comp != padded[1:-1, :-2])  # left
        | (m_comp != padded[1:-1, 2:])  # right
    )

    if not np.any(is_interface):
        return _SmoothingPlan(m_comp=m_comp, has_interface=False)

    # Compute dual cell bounds for this component
    x_lo, x_hi, y_lo, y_hi = _dual_cell_bounds(mesh, component)

    # Get interface pixel indices
    ii, jj = np.where(is_interface)

    # Build dual cell boxes (vectorized)
    dual_boxes = shapely.box(x_lo[ii], y_lo[jj], x_hi[ii], y_hi[jj])
    dual_areas = shapely.area(dual_boxes)

    # Compute area fractions per material for each interface pixel using
    # effective (non-overlapping) polygons that respect mesh-order overwrite.
    effective_polys = _effective_material_polygons(structures, materials)

    # Area fractions: shape (n_interface_pixels,) per material, guaranteed
    # to be non-overlapping by construction.
    fractions: dict[int, np.ndarray] = {}
    total_struct_fraction = np.zeros(len(ii), dtype=float)
    for idx, poly in effective_polys.items():
        intersections = shapely.intersection(poly, dual_boxes)
        areas = shapely.area(intersections)
        frac = areas / dual_areas
        fractions[idx] = frac
        total_struct_fraction += frac

    # Clip accumulated area to avoid tiny numerical overshoots > 1.
    total_struct_fraction = np.clip(total_struct_fraction, 0.0, 1.0)

    # Background (air) gets the remainder.
    fractions[0] = np.maximum(1.0 - total_struct_fraction, 0.0)

    # Determine interface orientation
    # Use padded array to check material changes in x and y directions
    diff_x = padded[:-2, 1:-1][ii, jj] != padded[2:, 1:-1][ii, jj]
    diff_y = padded[1:-1, :-2][ii, jj] != padded[1:-1, 2:][ii, jj]

    normal_x = diff_x & ~diff_y  # vertical interface, normal along x
    normal_y = ~diff_x & diff_y  # horizontal interface, normal along y
    # ~normal_x & ~normal_y  # corner or ambiguous -> arithmetic avg

    # Determine which averaging to use for this component at each pixel
    # harmonic when E is perpendicular to interface (E component == normal direction)
    # arithmetic when E is parallel to interface
    use_harmonic = np.zeros(len(ii), dtype=bool)
    if component == "x":
        use_harmonic[normal_x] = True  # Ex perpendicular to vertical interface
    elif component == "y":
        use_harmonic[normal_y] = True  # Ey perpendicular to horizontal interface
    # Ez is always parallel to interface -> always arithmetic
    # Corners -> arithmetic (use_harmonic stays False)

    return _SmoothingPlan(
        m_comp=m_comp,
        has_interface=True,
        ii=ii,
        jj=jj,
        fractions=fractions,
        use_harmonic=use_harmonic,
    )


def _assemble_smoothed_n(
    plan: _SmoothingPlan,
    materials: dict[Material, int],
    env: Environment,
    component: Literal["x", "y", "z"],
) -> ComplexArray2D:
    """Assemble the smoothed index from a (cached) plan and the per-env eps.

    Only this step depends on the environment (wavelength/temperature): it looks
    up each material's permittivity and applies the precomputed area-fraction
    averaging.
    """
    env_eps = np.complex128(1.0) ** 2  # background: air (n=1)
    eps = np.full_like(plan.m_comp, env_eps, dtype=np.complex128)
    mat_eps: dict[int, complex] = {0: env_eps}
    for material, idx in materials.items():
        e = _material_eps_component(material, env, component, component)
        eps[plan.m_comp == idx] = e
        mat_eps[idx] = e

    if not plan.has_interface:
        return np.sqrt(eps)

    ii, jj = plan.ii, plan.jj
    use_harmonic = plan.use_harmonic
    eps_eff = np.zeros(len(ii), dtype=np.complex128)

    # Arithmetic average: eps_eff = sum(f_i * eps_i)
    arith_mask = ~use_harmonic
    if np.any(arith_mask):
        eps_arith = np.zeros(int(arith_mask.sum()), dtype=np.complex128)
        for idx, frac in plan.fractions.items():
            eps_arith += frac[arith_mask] * mat_eps[idx]
        eps_eff[arith_mask] = eps_arith

    # Harmonic average: 1/eps_eff = sum(f_i / eps_i)
    if np.any(use_harmonic):
        inv_eps_harm = np.zeros(int(use_harmonic.sum()), dtype=np.complex128)
        for idx, frac in plan.fractions.items():
            e = mat_eps[idx]
            if abs(e) > 0:
                inv_eps_harm += frac[use_harmonic] / e
        # Avoid division by zero
        safe = np.abs(inv_eps_harm) > 1e-30
        eps_harm = np.zeros_like(inv_eps_harm)
        eps_harm[safe] = 1.0 / inv_eps_harm[safe]
        eps_harm[~safe] = eps[ii[use_harmonic], jj[use_harmonic]][~safe]
        eps_eff[use_harmonic] = eps_harm

    # Write back smoothed values
    eps[ii, jj] = eps_eff

    return np.sqrt(eps)


def _compute_smoothed_n(
    mesh: Mesh2D,
    m_full: IntArray2D,
    materials: dict[Material, int],
    env: Environment,
    structures: list[Structure2D],
    component: Literal["x", "y", "z"],
    plan: _SmoothingPlan | None = None,
) -> ComplexArray2D:
    """Compute subpixel-smoothed refractive index for a field component.

    At interface pixels, the effective permittivity is computed using
    area-fraction-weighted averaging:
      - E parallel to interface:      arithmetic avg  eps_eff = sum(f_i * eps_i)
      - E perpendicular to interface: harmonic avg    1/eps_eff = sum(f_i / eps_i)

    The wavelength-independent geometry work is factored into
    :func:`_smoothing_plan`; pass a precomputed ``plan`` to reuse it across a
    wavelength sweep.
    """
    if plan is None:
        plan = _smoothing_plan(mesh, m_full, materials, structures, component)
    return _assemble_smoothed_n(plan, materials, env, component)


# ==========================================================================
# Kottke / Farjadpour tensor subpixel smoothing (opt-in, 2nd-order accurate)
# ==========================================================================
@dataclass
class _KottkePlan:
    """Wavelength-independent plan for Kottke tensor subpixel smoothing.

    Extends :class:`_SmoothingPlan` with the interface *normal* at each interface
    pixel, computed from the gradient of a material fill-fraction field. The
    Kottke (Farjadpour 2006 / Kottke 2008) effective-permittivity tensor averages
    harmonically along the interface normal and arithmetically tangentially, as a
    full tensor - second-order accurate even for tilted interfaces, unlike the
    axis-aligned arithmetic/harmonic choice.
    """

    base: _SmoothingPlan
    n_x: np.ndarray = field(default_factory=lambda: np.empty(0))
    n_y: np.ndarray = field(default_factory=lambda: np.empty(0))
    eligible: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))


def _fraction_fields(
    mesh: Mesh2D,
    structures: list[Structure2D],
    materials: dict[Material, int],
    component: Literal["x", "y", "z"],
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Per-material area-fraction field on the full component grid (+ its coords)."""
    x_lo, x_hi, y_lo, y_hi = _dual_cell_bounds(mesh, component)
    xc = 0.5 * (x_lo + x_hi)
    yc = 0.5 * (y_lo + y_hi)
    xlo, ylo = np.meshgrid(x_lo, y_lo, indexing="ij")
    xhi, yhi = np.meshgrid(x_hi, y_hi, indexing="ij")
    boxes = shapely.box(xlo.ravel(), ylo.ravel(), xhi.ravel(), yhi.ravel())
    areas = shapely.area(boxes)
    polys = _effective_material_polygons(structures, materials)
    fields = {}
    for idx, poly in polys.items():
        frac = shapely.area(shapely.intersection(poly, boxes)) / areas
        fields[idx] = frac.reshape(xlo.shape)
    return fields, xc, yc


def _kottke_plan(
    mesh: Mesh2D,
    m_full: IntArray2D,
    materials: dict[Material, int],
    structures: list[Structure2D],
    component: Literal["x", "y", "z"],
) -> _KottkePlan:
    """Compute the (wavelength-independent) Kottke plan: base plan + normals."""
    base = _smoothing_plan(mesh, m_full, materials, structures, component)
    if not base.has_interface:
        return _KottkePlan(base=base)
    fields, xc, yc = _fraction_fields(mesh, structures, materials, component)
    ii, jj = base.ii, base.jj
    # the interface normal is the gradient of the fill-fraction field of whichever
    # material has the steepest boundary at the pixel (sign is irrelevant: the
    # tensor uses n_i n_j); pick that material per pixel
    best_mag = np.zeros(len(ii))
    n_x = np.zeros(len(ii))
    n_y = np.zeros(len(ii))
    for fld in fields.values():
        gx, gy = np.gradient(fld, xc, yc)
        gmag = np.hypot(gx[ii, jj], gy[ii, jj])
        take = gmag > best_mag
        best_mag[take] = gmag[take]
        n_x[take] = gx[ii, jj][take]
        n_y[take] = gy[ii, jj][take]
    norm = np.hypot(n_x, n_y)
    eligible = norm > 1e-9
    safe_norm = np.where(eligible, norm, 1.0)
    return _KottkePlan(
        base=base, n_x=n_x / safe_norm, n_y=n_y / safe_norm, eligible=eligible
    )


def _assemble_kottke(
    plan: _KottkePlan,
    materials: dict[Material, int],
    env: Environment,
    component: Literal["x", "y", "z"],
) -> ComplexArray2D:
    """Assemble the (diagonal) Kottke-smoothed index for a component.

    Each interface pixel uses the **normal-projected** effective permittivity
    ``eps_ii = eps_p + (eps_n - eps_p) n_i^2``, where ``eps_p`` is the arithmetic
    (tangential) and ``eps_n`` the harmonic (normal) average and ``n`` is the
    interface normal. This is the diagonal of the Kottke/Farjadpour tensor; it
    is continuous in the interface orientation (unlike the axis-aligned
    harmonic/arithmetic switch), giving second-order accuracy for tilted
    interfaces while staying diagonal-anisotropic so the default (base tidy3d)
    solver can use it.

    The full Kottke tensor additionally carries an off-diagonal
    ``eps_xy = (eps_n - eps_p) n_x n_y``; that makes the medium fully anisotropic
    and needs the tensorial solver (``tidy3d-extras``), so it is omitted here.
    """
    base = plan.base
    env_eps = np.complex128(1.0) ** 2
    eps = np.full_like(base.m_comp, env_eps, dtype=np.complex128)
    mat_eps: dict[int, complex] = {0: env_eps}
    for material, idx in materials.items():
        e = _material_eps_component(material, env, component, component)
        eps[base.m_comp == idx] = e
        mat_eps[idx] = e
    if not base.has_interface:
        return np.sqrt(eps)

    ii, jj = base.ii, base.jj
    # arithmetic (tangential) and harmonic (normal) averages at interface pixels
    eps_p = np.zeros(len(ii), dtype=np.complex128)
    inv = np.zeros(len(ii), dtype=np.complex128)
    for idx, frac in base.fractions.items():
        eps_p += frac * mat_eps[idx]
        if abs(mat_eps[idx]) > 0:
            inv += frac / mat_eps[idx]
    safe = np.abs(inv) > 1e-30
    eps_n = np.where(safe, 1.0 / np.where(safe, inv, 1.0), eps_p)

    if component == "z":
        # Ez is parallel to any in-plane interface -> arithmetic (tangential)
        eps[ii, jj] = eps_p
        return np.sqrt(eps)

    n_par = plan.n_x if component == "x" else plan.n_y
    kottke_diag = eps_p + (eps_n - eps_p) * n_par**2
    axis_diag = np.where(base.use_harmonic, eps_n, eps_p)
    eps[ii, jj] = np.where(plan.eligible, kottke_diag, axis_diag)
    return np.sqrt(eps)


def _compute_winner_takes_all_n(
    mesh: Mesh2D,
    m_full: IntArray2D,
    materials: dict[Material, int],
    env: Environment,
    structures: list[Structure2D],
    component: Literal["x", "y", "z"],
) -> ComplexArray2D:
    """Compute refractive index using winner-takes-all (no subpixel smoothing).

    For each dual cell:
      1. Compute overlap fraction of each material polygon with the cell.
      2. If no single material covers >= 50%, assign background index.
      3. Otherwise, pick the material with the highest overlap.
      4. Ties are broken by mesh order (higher wins).
    """
    si, sj = _COMPONENT_SLICES[component]
    m_comp = m_full[si, sj]

    # Build eps array from material indices (start with non-interface values)
    env_eps = np.complex128(1.0) ** 2
    eps = np.full_like(m_comp, env_eps, dtype=np.complex128)
    mat_eps: dict[int, complex] = {0: env_eps}
    for material, idx in materials.items():
        e = _material_eps_component(material, env, component, component)
        eps[m_comp == idx] = e
        mat_eps[idx] = e

    # Identify interface pixels
    padded = np.pad(m_comp, 1, mode="edge")
    is_interface = (
        (m_comp != padded[:-2, 1:-1])
        | (m_comp != padded[2:, 1:-1])
        | (m_comp != padded[1:-1, :-2])
        | (m_comp != padded[1:-1, 2:])
    )

    if not np.any(is_interface):
        return np.sqrt(eps)

    # Compute dual cell bounds and interface pixel indices
    x_lo, x_hi, y_lo, y_hi = _dual_cell_bounds(mesh, component)
    ii, jj = np.where(is_interface)
    dual_boxes = shapely.box(x_lo[ii], y_lo[jj], x_hi[ii], y_hi[jj])
    dual_areas = shapely.area(dual_boxes)

    # Compute area fractions per material
    effective_polys = _effective_material_polygons(structures, materials)

    # Build mesh-order lookup: mat_idx -> max mesh_order among its structures
    mat_mesh_order: dict[int, int] = {0: -1}  # background has lowest priority
    for s in structures:
        idx = materials[s.material]
        mat_mesh_order[idx] = max(mat_mesh_order.get(idx, -1), s.mesh_order)

    fractions: dict[int, np.ndarray] = {}
    total_struct_fraction = np.zeros(len(ii), dtype=float)
    for idx, poly in effective_polys.items():
        intersections = shapely.intersection(poly, dual_boxes)
        areas = shapely.area(intersections)
        frac = areas / dual_areas
        fractions[idx] = frac
        total_struct_fraction += frac

    total_struct_fraction = np.clip(total_struct_fraction, 0.0, 1.0)
    fractions[0] = np.maximum(1.0 - total_struct_fraction, 0.0)

    # Winner-takes-all: pick material with highest overlap per pixel
    n_pixels = len(ii)
    best_idx = np.zeros(n_pixels, dtype=int)  # 0 = background
    best_frac = fractions[0].copy()
    best_mesh_order = np.full(n_pixels, mat_mesh_order[0])

    for idx, frac in fractions.items():
        if idx == 0:
            continue
        mo = mat_mesh_order.get(idx, -1)
        # Win if: higher fraction, or same fraction but higher mesh order
        wins = (frac > best_frac) | ((frac == best_frac) & (mo > best_mesh_order))
        best_idx[wins] = idx
        best_frac[wins] = frac[wins]
        best_mesh_order[wins] = mo

    # If no material covers >= 50%, fall back to background
    below_threshold = best_frac < 0.5
    best_idx[below_threshold] = 0

    # Assign eps from winner
    winner_eps = np.array([mat_eps[idx] for idx in best_idx])
    eps[ii, jj] = winner_eps

    return np.sqrt(eps)


def _shapely_to_path(geom: shapely.Geometry) -> Any:
    """Convert a shapely geometry to a matplotlib compound Path.

    Handles (multi)polygons with holes; non-polygonal parts (e.g. lines from
    degenerate intersections) are skipped. Returns None if no polygonal area
    is left.
    """
    from matplotlib.path import Path  # fmt: skip

    paths = []
    for poly in getattr(geom, "geoms", [geom]):
        if not isinstance(poly, shapely.Polygon) or poly.is_empty:
            continue
        paths.append(Path(np.asarray(poly.exterior.coords)))
        paths.extend(Path(np.asarray(ring.coords)) for ring in poly.interiors)
    if not paths:
        return None
    return Path.make_compound_path(*paths)


def _effective_material_polygons(
    structures: list[Structure2D],
    materials: dict[Material, int],
) -> dict[int, shapely.Geometry]:
    """Create non-overlapping effective polygons per material.

    The effective polygons follow the same overwrite precedence as
    `_create_full_material_array`: later groups overwrite earlier groups.
    """
    grouped = _classify_structures_by_mesh_order_and_material(structures, materials)
    ordered_groups: list[tuple[int, shapely.Geometry]] = []
    for group_structs in grouped.values():
        mat_idx = materials[group_structs[0].material]
        poly: Any = unary_union([s.geometry._shapely_polygon() for s in group_structs])
        ordered_groups.append((mat_idx, poly))

    effective_by_mat: dict[int, shapely.Geometry] = {}
    occupied = shapely.GeometryCollection()
    for mat_idx, poly in reversed(ordered_groups):
        eff = poly.difference(occupied)
        if not eff.is_empty:
            if mat_idx in effective_by_mat:
                effective_by_mat[mat_idx] = unary_union(
                    [effective_by_mat[mat_idx], eff]
                )
            else:
                effective_by_mat[mat_idx] = eff
        occupied = unary_union([occupied, poly])
    return effective_by_mat
