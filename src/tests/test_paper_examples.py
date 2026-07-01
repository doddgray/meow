"""Tests for the paper-reproduction examples (examples/papers)."""

import os
import sys
from pathlib import Path

import gdsfactory as gf
import numpy as np
import pytest

import meow as mw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MEOW_EXAMPLE_RES", "low")  # coarse smoke-test resolution

gf.gpdk.PDK.activate()

from examples.papers import (  # noqa: E402
    kwolek2026_faquad as kw,
)
from examples.papers import (  # noqa: E402
    kwolek2026_test_structures as kts,
)
from examples.papers import (  # noqa: E402
    kwolek_designer as kd,
)
from examples.papers import (  # noqa: E402
    kwolek_designer_slurm as ks,
)
from examples.papers import (  # noqa: E402
    magden2018_dichroic as md,
)
from examples.papers import (  # noqa: E402
    zhu2025_designer as zd,
)

# --- Kwolek 2026: ring-resonator test structures ---


def test_kwolek_test_structures_gds(tmp_path: Path) -> None:
    """All-pass + add-drop rings build with ports and write GDS."""
    ap = kts.all_pass_ring(radius=40.0, gap=0.5)
    assert {"o1", "o2"} <= {p.name for p in ap.ports}
    ad = kts.add_drop_ring(radius=40.0, gap=0.5)
    assert len(ad.ports) >= 4  # two bus waveguides
    written = kts.write_gds(tmp_path)
    assert written
    assert all(p.suffix == ".gds" and p.exists() for p in written)


def test_designer_extras_cutback_and_tapers() -> None:
    """Shared designer helpers: no-op tapers, tapers, constant-length cut-back."""
    from examples.papers import _designer_extras as dx

    comb = kw.faquad_combiner(0.05, 0.4, 0.2, l_m=40.0)
    # default: no taper -> identity
    assert dx.tapered_ports(comb) is comb
    tapered = dx.tapered_ports(comb, {"in_bar": 2.0}, taper_lengths=10.0)
    assert {p.name for p in tapered.ports} == {p.name for p in comb.ports}
    assert tapered.ports["in_bar"].width == pytest.approx(2.0, abs=2e-3)
    # cut-back array: regularly-spaced ports across a 5 mm chip, equal row length
    arr = dx.coupler_cutback_array(
        comb, counts=(0, 1, 2), in_port="in_bar", thru_port="out_bar",
        chip_width=5000.0, pitch=50.0, width=kw.W_TOP,
    )
    assert arr.xmax - arr.xmin == pytest.approx(5000.0, abs=5.0)  # 5 mm wide
    assert {f"in_{n}" for n in (0, 1, 2)} <= {p.name for p in arr.ports}


def test_designer_extras_spectrum_grid(tmp_path: Path) -> None:
    """The spectrum grid stacks one row per design and writes a figure."""
    from examples.papers import _designer_extras as dx

    wls = np.linspace(0.62, 1.86, 20)
    rows = [
        {"label": f"d{i}", "wls": wls, "bar": np.full_like(wls, 0.5),
         "cross": np.full_like(wls, 0.5), "design_wls": [0.775, 1.55]}
        for i in range(2)
    ]
    out = dx.spectrum_grid(rows, tmp_path / "grid.png", db=True)
    assert Path(out).exists()


def test_kwolek_fig3_resonator_layouts() -> None:
    """The Fig. 3 DUT/control racetracks and measurement layout build."""
    dut = kts.dut_resonator(l_m=40.0, radius=30.0)
    assert {"bus_in", "bus_out"} <= {p.name for p in dut.ports}
    # the FAQUAD coupler is closed into a ring -> the loop rises well above 0
    assert dut.ysize > 2 * 30.0  # at least the two 180-degree bends tall
    ctrl = kts.control_resonator(l_m=40.0, radius=30.0)
    assert ctrl.get_polygons()
    layout = kts.fh_measurement_layout(l_m=40.0, radius=30.0)
    # DUT stacked above the control resonator
    assert layout.ysize > dut.ysize


# --- Zhu 2025: user-defined multi-layer stack designer ---


def test_zhu_stack_designer_geometry() -> None:
    """The default Si3N4 triple stack has the requested cores/widths/taper."""
    stack = zd.default_si3n4_stack()
    assert [round(L.thickness * 1e3) for L in stack.layers] == [100, 200, 40]
    assert [L.facet_width for L in stack.layers] == [5.0, 2.0, 6.0]
    # three distinct vertical levels, centred about zero
    ycs = stack.y_centers()
    assert len(ycs) == 3
    assert np.isclose(sum(ycs), 0.0)
    # the main (central) layer spans the whole device; assists start at the trident
    main = stack.layers[stack.main_index]
    _z_main, w_main, z0_main = zd._layer_profile(stack, main, npts=64)
    assert z0_main == 0.0
    assert np.isclose(w_main[0], stack.standard_width)  # single-layer input width
    assert np.isclose(w_main[-1], main.facet_width)  # central width at facet
    assist = stack.layers[0]
    _, _, z0_assist = zd._layer_profile(stack, assist, npts=64)
    assert z0_assist == pytest.approx(stack.l_input + stack.l_intaper)


def test_zhu_stack_designer_writes_gds_and_plots(tmp_path: Path) -> None:
    """design_stack writes a GDS, a 2.5D/3D render and a summary."""
    stack = zd.default_si3n4_stack()
    res = zd.design_stack(stack, "smoke", tmp_path)
    assert Path(res["gds"]).exists()
    assert Path(res["render"]).exists()
    assert res["summary"]["core_thicknesses_nm"] == [100.0, 200.0, 40.0]


# --- Magden 2018: silicon dichroic filter ---


def test_dichroic_filter_layout() -> None:
    c = md.dichroic_filter()
    port_names = {p.name for p in c.ports}
    assert {"in0", "short_pass", "long_pass"} <= port_names
    total_length = md.L1 + md.L2 + md.L3 + md.L4
    assert np.isclose(c.xmax - c.xmin, total_length, atol=1e-3)
    # WGA + 3 WGB segments
    polys = [p for ps in c.get_polygons().values() for p in ps]
    assert len(polys) == 4


def test_dichroic_filter_extrusion() -> None:
    c = md.dichroic_filter()
    structs = md.extrude_filter(c)
    si = [s for s in structs if s.material.name == "silicon"]
    oxide = [s for s in structs if "oxide" in s.material.name]
    assert len(si) == 4
    assert len(oxide) >= 1


def test_dichroic_topology_matches_fig3a() -> None:
    """WGB runs straight on the axis; WGA tips in and bends away below it.

    Reproduces the paper Fig. 3a layout: the long-pass (WGB) output sits on
    the lateral axis (x ~ 0) while the short-pass (WGA) output is displaced to
    negative x by the section-3 bend (the previous layout had these swapped).
    """
    seg_centers, y_a_couple, y_a_final = md.lateral_positions()
    # WGB straight & centred on the axis; WGA below it and bending further away
    assert np.isclose(seg_centers[1], 0.0)
    assert y_a_final < y_a_couple < 0.0
    c = md.dichroic_filter()
    ports = {p.name: p for p in c.ports}
    y_short = ports["short_pass"].center[1]
    y_long = ports["long_pass"].center[1]
    assert y_short < y_long  # WGA (short-pass) below WGB (long-pass)
    assert np.isclose(y_long, 0.0, atol=1e-6)
    assert np.isclose(y_short, y_a_final, atol=1e-6)


def _y_intervals(
    c: gf.Component, z: float, *, wgb_only: bool
) -> list[tuple[float, float]]:
    """Lateral (y) extents of the Si ridges crossing the plane at ``z``."""
    import shapely

    cut = shapely.LineString([(z, -6.0), (z, 6.0)])
    out: list[tuple[float, float]] = []
    for ps in c.get_polygons().values():
        for p in ps:
            try:
                dbu = c.layout().dbu
                pts = np.asarray(
                    [(pt.x * dbu, pt.y * dbu) for pt in p.each_point_hull()]
                )
            except AttributeError:
                pts = np.asarray(p)
            inter = shapely.Polygon(pts).intersection(cut)
            for geom in getattr(inter, "geoms", [inter]):
                if geom.is_empty:
                    continue
                ys = np.asarray(geom.coords)[:, 1]
                lo, hi = float(ys.min()), float(ys.max())
                if wgb_only and 0.5 * (lo + hi) < -0.6:  # skip WGA (well below axis)
                    continue
                out.append((lo, hi))
    return sorted(out)


def test_dichroic_constant_width_gaps_and_single_ridge_ports() -> None:
    """The refined WGB taper keeps constant total width and gaps, single-ridge
    ports and finite (w_tip) taper tips; WGA tapers at constant gap."""
    c = md.dichroic_filter()
    total = md.w_b_total()
    z1 = md.L1
    z4 = md.L1 + md.L2 + md.L3 + md.L4

    # single-ridge WGB ports (only the central ridge in the l_ext extensions)
    assert len(_y_intervals(c, 1.0, wgb_only=True)) == 1
    assert len(_y_intervals(c, z4 - 1.0, wgb_only=True)) == 1
    # three ridges in the coupling region
    coupling = _y_intervals(c, z1 + 200.0, wgb_only=True)
    assert len(coupling) == 3

    # constant total WGB width (envelope) in the taper and the coupling region
    for z in (md.L_EXT + 20.0, z1 + 200.0):
        ivs = _y_intervals(c, z, wgb_only=True)
        envelope = ivs[-1][1] - ivs[0][0]
        assert np.isclose(envelope, total, atol=2e-3)
    # constant inter-ridge gaps (= g_b) in the coupling region
    gaps = [coupling[i + 1][0] - coupling[i][1] for i in range(len(coupling) - 1)]
    assert np.allclose(gaps, md.G_B, atol=2e-3)

    # all taper tips have the finite minimum width w_tip (WGA tip just past z1)
    wga = _y_intervals(c, z1 + 0.5, wgb_only=False)[0]
    assert wga[1] - wga[0] == pytest.approx(md.W_TIP, abs=2e-3)

    # WGA tapers at a constant edge-to-edge gap through section 2: its upper
    # (WGB-facing) edge stays fixed while it widens
    z2 = md.L1 + md.L2
    edge_a = _y_intervals(c, z1 + 20.0, wgb_only=False)[0][1]
    edge_b = _y_intervals(c, z2 - 20.0, wgb_only=False)[0][1]
    assert np.isclose(edge_a, edge_b, atol=2e-3)


def test_delta_kappa_spectrum_monotonic() -> None:
    """The coupling |kappa| is monotonic across the cutoff (no spurious peak).

    The previous supermode-splitting extraction gave a kappa that peaked at the
    cutoff and clamped to zero away from it; the coupled-mode overlap integral
    is monotonic and positive, and is calibrated to the design coupling.
    """
    wls = np.linspace(1.530, 1.545, 4)
    mesh = md.mesh2d(res=0.08)
    deltas, kappas = md.delta_kappa_spectrum(wls, wl_ref=1.537, mesh=mesh)
    assert np.all(np.diff(deltas) < 0)  # detuning decreases through zero
    assert np.all(kappas > 0)
    dk = np.diff(kappas)
    assert np.all(dk >= -1e-12) or np.all(dk <= 1e-12)  # monotonic
    # calibrated to the design coupling at the reference wavelength
    assert np.interp(1.537, wls, kappas) == pytest.approx(md.KAPPA_DESIGN, rel=1e-6)


def test_backend_resolver_and_device_s_matrix() -> None:
    from examples.papers import _backends

    assert _backends.resolve_backend("tidy3d") is mw.compute_modes_tidy3d
    assert _backends.resolve_backend("mpb") is mw.compute_modes_mpb
    assert _backends.resolve_backend(mw.compute_modes) is mw.compute_modes
    with pytest.raises(ValueError, match="Unknown backend"):
        _backends.resolve_backend("nope")
    assert _backends.parallel_enabled(parallel=True) is True
    assert _backends.parallel_enabled(parallel=False) is False

    # serial device S-matrix on a tiny stack returns SAX multimode ports
    c = md.dichroic_filter()
    cells = md.device_cells(
        c, cells_per_section=(1, 1, 1, 1), mesh=md.device_mesh(res=0.08)
    )
    env = mw.Environment(wl=1.55, T=25.0)
    S, pm = _backends.device_s_matrix(cells, env, num_modes=2, parallel=False)
    assert "left@0" in pm
    assert "right@0" in pm
    assert np.asarray(S).shape[0] == len(pm)


def test_phase_matching_cutoff_in_c_band() -> None:
    """The WGA(318nm)/WGB phase-matching point lies in the C-band.

    This is the central design result of the paper (Fig. 1e: cutoff near
    1540 nm). At converged resolution (<= 20 nm) the sign of
    n_WGA - n_WGB must flip between 1500 and 1600 nm.
    """
    mesh = md.mesh2d(res=0.02)
    diffs = []
    for wl in (1.50, 1.60):
        n_a = md.fundamental_neff(md.wga_structures(0.318), wl, mesh=mesh)
        n_b = md.fundamental_neff(md.wgb_structures(), wl, mesh=mesh)
        diffs.append(n_a - n_b)
    assert diffs[0] > 0  # below cutoff: WGA has the higher index
    assert diffs[1] < 0  # above cutoff: WGB has the higher index


def test_analytical_transmission() -> None:
    gammas = np.array([-1e3, 0.0, 1e3])
    t = md.analytical_transmission(gammas)
    assert np.isclose(t[1], 0.5)  # gamma = 0 marks the 3 dB cutoff
    assert t[0] < 1e-6
    assert t[2] > 1 - 1e-6
    assert np.all(np.diff(md.analytical_transmission(np.linspace(-5, 5, 50))) > 0)


def test_device_cells_pipeline() -> None:
    c = md.dichroic_filter()
    cells = md.device_cells(c, cells_per_section=(2, 2, 3, 2), mesh=md.mesh2d(res=0.05))
    assert len(cells) == 9
    assert np.isclose(sum(cell.length for cell in cells), md.L1 + md.L2 + md.L3 + md.L4)


# --- Generalized dichroic beam-splitter designer ---


def test_dichroic_designer_platform_and_wgb() -> None:
    from examples.papers import dichroic_designer as dd

    plat = dd.Platform(
        core=mw.silicon, clad=mw.silicon_oxide, core_thickness=0.30, etch_fraction=0.5
    )
    assert plat.slab_thickness == pytest.approx(0.15)
    wgb = dd.WGB(rail_width=0.25, gap=0.10, n_rails=3)
    assert wgb.total_width == pytest.approx(3 * 0.25 + 2 * 0.10)
    assert wgb.centers(0.0) == pytest.approx([-0.35, 0.0, 0.35])
    # partial etch -> ridge + slab (both core) + cladding
    structs = dd._ridge_structures(plat, [0.30], [0.0], (-2.0, 2.0))
    cores = [s for s in structs if s.material is mw.silicon]
    assert len(cores) == 2  # ridge + slab
    assert any(isinstance(s.geometry, mw.Prism) for s in cores)  # angled-able ridge


def test_dichroic_designer_reproduces_magden_width() -> None:
    """On 220 nm SOI the designer's phase-match width matches the paper's 318 nm.

    Targeting the ~1540 nm cutoff with the paper's sub-wavelength WGB, the WGA
    width that makes n_WGA = n_WGB is ~318 nm (Magden 2018), validating the
    generalized phase-matching design.
    """
    from examples.papers import dichroic_designer as dd

    plat = dd.Platform(core=mw.silicon, clad=mw.silicon_oxide, core_thickness=0.22)
    wgb = dd.WGB(rail_width=0.25, gap=0.10, n_rails=3)
    n_b = dd.segmented_neff(plat, wgb, 1.54, res=0.05)
    w_a = dd.phase_match_width(plat, 1.54, wgb, res=0.05)
    assert 0.28 < w_a < 0.36  # ~318 nm
    # self-consistent phase match (loose: coarse test mesh)
    assert dd.solid_neff(plat, w_a, 1.54, res=0.05) == pytest.approx(n_b, abs=6e-3)
    # wider WGA -> higher index -> would phase-match at a longer cutoff
    assert dd.solid_neff(plat, w_a + 0.03, 1.54, res=0.05) > n_b


def test_optimize_phase_match_width_matches_root_find() -> None:
    """The AD gradient-based optimizer converges near the root-find width.

    Both minimize the same phase-mismatch residual (jax.grad + Adam vs.
    brentq), so from an off-target initial guess the optimizer should land
    close to the root-find's answer, with the loss trace strictly improving
    over the initial value.
    """
    from examples.papers import dichroic_designer as dd

    plat = dd.Platform(core=mw.silicon, clad=mw.silicon_oxide, core_thickness=0.22)
    wgb = dd.WGB(rail_width=0.25, gap=0.10, n_rails=3)
    w_root = dd.phase_match_width(plat, 1.54, wgb, res=0.05)
    w_opt, trace = dd.optimize_phase_match_width(
        plat, 1.54, wgb, w0=0.6, res=0.05, steps=15
    )
    assert abs(w_opt - w_root) < 0.1
    assert trace.losses[-1] < trace.losses[0]
    assert trace.param_names == ("w_a [um]",)


def test_design_dichroic_gradient_path_populates_trace() -> None:
    """design_dichroic(use_gradient=True) runs the AD optimizer and stores it."""
    from examples.papers import dichroic_designer as dd

    plat = dd.Platform(core=mw.silicon, clad=mw.silicon_oxide, core_thickness=0.22)
    wgb = dd.WGB(rail_width=0.25, gap=0.10, n_rails=3)
    d = dd.design_dichroic(
        plat, 1.54, wgb=wgb, res=0.05,
        use_gradient=True, gradient_w0=0.6, gradient_steps=6,
    )
    assert d.opt_trace is not None
    assert len(d.opt_trace.losses) == 7
    assert d.total_length <= plat.max_length + 1.0


def test_wgb_heterogeneous_rail_widths() -> None:
    """WGB.frac_mid/frac_out give heterogeneous mid/outer rail widths.

    ``frac_mid=frac_out=1.0`` (the default) must reproduce the uniform-width
    WGB exactly; scaling them apart changes only the corresponding rails.
    """
    from examples.papers import dichroic_designer as dd

    uniform = dd.WGB(rail_width=0.25, gap=0.10, n_rails=3)
    assert uniform.widths == pytest.approx([0.25, 0.25, 0.25])
    assert uniform.total_width == pytest.approx(3 * 0.25 + 2 * 0.10)
    assert uniform.centers(0.0) == pytest.approx([-0.35, 0.0, 0.35])

    hetero = dd.WGB(rail_width=0.25, gap=0.10, n_rails=3, frac_mid=1.4, frac_out=0.8)
    out_w, mid_w = 0.8 * 0.25, 1.4 * 0.25
    assert hetero.widths == pytest.approx([out_w, mid_w, out_w])
    total = mid_w + 2 * out_w + 2 * 0.10
    assert hetero.total_width == pytest.approx(total)
    centers = hetero.centers(0.0)
    assert centers == pytest.approx(
        [-(mid_w / 2 + 0.10 + out_w / 2), 0.0, mid_w / 2 + 0.10 + out_w / 2]
    )
    # rails are contiguous with the specified gaps, symmetric about x0
    assert centers[1] - mid_w / 2 - (centers[0] + out_w / 2) == pytest.approx(0.10)


def test_dichroic_filter_heterogeneous_rails_matches_uniform_default() -> None:
    """dichroic_filter(frac_mid=frac_out=1.0) matches the pre-existing layout."""
    default = md.dichroic_filter()
    explicit = md.dichroic_filter(frac_mid=1.0, frac_out=1.0)
    assert default.xmax == pytest.approx(explicit.xmax)
    assert default.ymin == pytest.approx(explicit.ymin)
    assert default.ymax == pytest.approx(explicit.ymax)

    # a heterogeneous WGB still builds a valid, wider-than-mid-alone layout
    hetero = md.dichroic_filter(frac_mid=1.3, frac_out=0.7)
    assert hetero.ymax - hetero.ymin > 0


def test_reference_gvm_sign_matches_original_paper() -> None:
    """The Magden 2018 SOI design has ng_WGA > ng_WGB (WGA is more dispersive).

    This fixes the sign every cross-section optimization's group-velocity
    mismatch term is oriented to match.
    """
    from examples.papers import dichroic_designer as dd

    assert dd.reference_gvm_sign() == pytest.approx(1.0)


def test_optimize_dichroic_crosssection_reduces_loss() -> None:
    """Stage 1: phase-match + max-GVM cross-section optimizer runs and improves."""
    from examples.papers import dichroic_designer as dd

    plat = dd.Platform(core=mw.silicon, clad=mw.silicon_oxide, core_thickness=0.22)
    params, trace = dd.optimize_dichroic_crosssection(
        plat, 1.54, res=0.09, steps=2, lr=0.05
    )
    assert params.shape == (5,)
    assert trace.param_names == dd.CROSSSECTION_PARAM_NAMES
    assert len(trace.losses) == 3
    assert trace.losses[-1] < trace.losses[0]


def test_optimize_dichroic_lengths_reduces_loss() -> None:
    """Stage 2: gap + lengths adiabatic-loss optimizer runs and improves."""
    from examples.papers import dichroic_designer as dd

    plat = dd.Platform(core=mw.silicon, clad=mw.silicon_oxide, core_thickness=0.22)
    wgb = dd.WGB(rail_width=0.25, gap=0.10, n_rails=3)
    params, trace = dd.optimize_dichroic_lengths(
        plat, 1.54, 0.35, wgb, res=0.09, steps=2, lr=0.1
    )
    assert params.shape == (5,)
    assert trace.param_names == dd.LENGTHS_PARAM_NAMES
    assert len(trace.losses) == 3
    assert trace.losses[-1] < trace.losses[0]


def test_design_dichroic_joint_populates_trace() -> None:
    """design_dichroic_joint runs both stages and builds a valid design."""
    from examples.papers import dichroic_designer as dd

    plat = dd.Platform(core=mw.silicon, clad=mw.silicon_oxide, core_thickness=0.22)
    d = dd.design_dichroic_joint(
        plat, 1.54, res=0.09, crosssection_steps=2, length_steps=2
    )
    assert d.opt_trace is not None
    assert d.opt_trace_lengths is not None
    assert len(d.opt_trace.losses) == 3
    assert len(d.opt_trace_lengths.losses) == 3
    assert d.total_length <= 5000.0 + 1.0
    assert d.component is not None


def test_dichroic_designer_si3n4_platform_and_width() -> None:
    """The Si3N4 example designs a fabricable splitter in the 900-1200 nm band."""
    from examples.papers import dichroic_designer as dd
    from examples.papers import dichroic_designer_si3n4 as sn

    plat = sn.si3n4_platform()
    assert plat.core is mw.silicon_nitride
    assert plat.clad is mw.silicon_oxide
    assert plat.core_thickness == pytest.approx(0.20)
    assert plat.etch_fraction == pytest.approx(1.0)  # fully etched -> no slab
    assert plat.slab_thickness == pytest.approx(0.0)
    assert plat.min_tip == pytest.approx(0.05)
    assert plat.min_gap == pytest.approx(0.05)
    assert plat.max_length == pytest.approx(2000.0)
    # targeted cutoffs span 900-1200 nm in 50 nm steps
    assert np.isclose(sn.TARGET_CUTOFFS[0], 0.90)
    assert np.isclose(sn.TARGET_CUTOFFS[-1], 1.20)
    assert np.allclose(np.diff(sn.TARGET_CUTOFFS), 0.05)
    # a 1050 nm cutoff phase-matches at a fabricable WGA width
    w_a = dd.phase_match_width(plat, 1.05, sn.WGB_DESIGN, res=0.05)
    assert 0.30 < w_a < 0.65
    assert w_a > plat.min_tip


def test_dichroic_designer_thickness_sweep_setup() -> None:
    """The Si3N4 thickness-sweep config spans 900-1200 nm (with 990 nm)."""
    from examples.papers import dichroic_designer_si3n4_thickness as ts

    # cutoffs cover 900-1200 nm and include 990 nm
    assert ts.CUTOFFS.min() == pytest.approx(0.90)
    assert ts.CUTOFFS.max() == pytest.approx(1.20)
    assert any(np.isclose(ts.CUTOFFS, 0.99))
    # 200/100/40 nm cores, each with a (progressively wider) sub-wavelength WGB
    assert set(ts.THICKNESS_CONFIGS) == {200, 100, 40}
    rails = [ts.THICKNESS_CONFIGS[t][2].rail_width for t in (200, 100, 40)]
    assert rails == sorted(rails)  # thinner core -> wider rails
    for t_nm, (t_um, _clad, wgb, _res) in ts.THICKNESS_CONFIGS.items():
        assert t_um == pytest.approx(t_nm / 1000)
        assert wgb.gap == pytest.approx(0.05)
    plat = ts.platform(*ts.THICKNESS_CONFIGS[100][:2])
    assert plat.core is mw.silicon_nitride
    assert plat.max_length == pytest.approx(2000.0)


def _minimal_slurm_design() -> object:
    """A tiny manually-built design (avoids the expensive design_dichroic)."""
    from examples.papers import dichroic_designer as dd
    from examples.papers import magden2018_dichroic as md

    plat = dd.Platform(
        core=mw.silicon_nitride,
        clad=mw.silicon_oxide,
        core_thickness=0.20,
        min_tip=0.05,
        min_gap=0.05,
        max_length=2000.0,
        clad_thickness=1.0,
    )
    wgb = dd.WGB(0.20, 0.05, 3)
    w_a, gap = 0.5, 0.6
    comp = md.dichroic_filter(
        w_a=w_a,
        w_b=wgb.rail_width,
        g_b=wgb.gap,
        gap=gap,
        gap_out=2.0,
        w_tip=0.05,
        l1=10,
        l2=10,
        l3=10,
        l4=10,
    )
    return dd.DichroicDesign(
        platform=plat,
        cutoff_wl=1.0,
        wgb=wgb,
        w_a=w_a,
        gap=gap,
        lengths=(10, 10, 10, 10),
        kappa=0.005,
        dn_dw=1.0,
        extinction_db=20.0,
        component=comp,
    )


def test_slurm_designer_executor_and_cells(tmp_path: Path) -> None:
    """The slurm example builds an executor and slices a device into cells."""
    from examples.papers import dichroic_designer_slurm as ds

    # make_executor honours the cluster argument (debug -> in-process executor)
    executor = ds.make_executor(folder=tmp_path / "jobs", cluster="debug")
    assert hasattr(executor, "submit")

    design = _minimal_slurm_design()
    cells = ds.device_cells(design, num_cells=4, res=0.1)
    assert len(cells) == 4
    # the cells tile the full device length end-to-end
    assert cells[0].z_min == pytest.approx(0.0)
    assert cells[-1].z_max == pytest.approx(float(design.component.xmax))


def test_slurm_designer_blocking_and_concurrent_agree(tmp_path: Path) -> None:
    """Blocking and async EME paths give the same port powers (debug executor)."""
    import asyncio

    from examples.papers import dichroic_designer_slurm as ds

    design = _minimal_slurm_design()
    eme_kwargs = {"num_cells": 4, "num_modes": 2, "res": 0.1}

    blocking = ds.run_blocking(
        [design],
        executor=ds.make_executor(folder=tmp_path / "blocking", cluster="debug"),
        **eme_kwargs,
    )
    concurrent = asyncio.run(
        ds.run_concurrent(
            [design],
            executor=ds.make_executor(folder=tmp_path / "concurrent", cluster="debug"),
            **eme_kwargs,
        )
    )
    assert set(blocking) == set(concurrent) == {"1000nm"}
    for key in blocking:
        b_short, b_long = blocking[key]
        c_short, c_long = concurrent[key]
        assert b_short == pytest.approx(c_short, abs=1e-9)
        assert b_long == pytest.approx(c_long, abs=1e-9)
        # power is conserved/split between the two output ports
        assert 0.0 <= b_short <= 1.0
        assert 0.0 <= b_long <= 1.0


def test_slurm_designer_submit_then_gather(tmp_path: Path) -> None:
    """submit_designs persists records that gather_results reloads later.

    Submits each design's EME (local submitit cluster) into a shared folder,
    drops the in-memory handles, then reloads + collects from the folder alone -
    matching the blocking workflow's port powers.
    """
    pytest.importorskip("submitit")
    from examples.papers import dichroic_designer_slurm as ds

    design = _minimal_slurm_design()
    eme_kwargs = {"num_cells": 4, "num_modes": 2, "res": 0.1}
    folder = tmp_path / "shared"

    records = ds.submit_designs(
        [design],
        executor=ds.make_executor(folder=folder, cluster="local"),
        folder=folder,
        **eme_kwargs,
    )
    assert [r.label for r in records] == ["1000nm"]
    assert (folder / "1000nm.eme.pkl").exists()
    del records  # only the persisted records remain, as in a later session

    gathered = ds.gather_results(folder)
    blocking = ds.run_blocking(
        [design],
        executor=ds.make_executor(folder=tmp_path / "block", cluster="local"),
        **eme_kwargs,
    )
    assert set(gathered) == set(blocking) == {"1000nm"}
    g_short, g_long = gathered["1000nm"]
    b_short, b_long = blocking["1000nm"]
    assert g_short == pytest.approx(b_short, abs=1e-9)
    assert g_long == pytest.approx(b_long, abs=1e-9)


def _tiny_analysis_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the dense spectrum / propagation grids so analysis runs are fast."""
    monkeypatch.setenv("MEOW_SPECTRUM_NPTS", "3")
    monkeypatch.setenv("MEOW_PROP_NPTS", "3")


# --- example settings: resolution levels, backend + parallel resources ---


def test_resolution_levels_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """low/medium/high + the 128/8 high standard + MEOW_NUM_* overrides + FAST."""
    from examples.papers import _resolution as r

    monkeypatch.delenv("MEOW_NUM_CELLS", raising=False)
    monkeypatch.delenv("MEOW_NUM_MODES", raising=False)
    monkeypatch.delenv("MEOW_EXAMPLE_FAST", raising=False)

    monkeypatch.setenv("MEOW_EXAMPLE_RES", "high")
    assert r.level() == "high"
    assert r.pick(low=1, medium=2, high=3) == 3
    # high is the converged 128-cell / 8-mode standard
    assert r.num_cells(low=8, medium=16) == 128
    assert r.num_modes(low=2, medium=4) == 8

    monkeypatch.setenv("MEOW_EXAMPLE_RES", "low")
    assert r.is_low()
    assert r.num_cells(low=8, medium=16) == 8

    # env vars override the resolution-derived value (incl. the high standard)
    monkeypatch.setenv("MEOW_NUM_CELLS", "55")
    monkeypatch.setenv("MEOW_NUM_MODES", "9")
    assert r.num_cells(low=8, medium=16) == 55
    assert r.num_modes(low=2, medium=4) == 9

    # legacy MEOW_EXAMPLE_FAST maps to low
    monkeypatch.delenv("MEOW_EXAMPLE_RES")
    monkeypatch.setenv("MEOW_EXAMPLE_FAST", "1")
    assert r.level() == "low"


def test_backend_and_resource_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The solver backend and parallel resources are env-configurable."""
    from examples.papers import _backends as b

    monkeypatch.setenv("MEOW_PAPER_BACKEND", "mpb")
    assert b.backend_name() == "mpb"
    assert b.resolve_backend() is mw.compute_modes_mpb
    assert b.resolve_backend("tidy3d") is mw.compute_modes_tidy3d
    assert b.resolve_backend("lumerical") is mw.compute_modes_lumerical

    monkeypatch.setenv("MEOW_CPUS_PER_TASK", "6")
    monkeypatch.setenv("MEOW_TIMEOUT_MIN", "42")
    monkeypatch.setenv("MEOW_SLURM_PARTITION", "cpu")
    monkeypatch.setenv("MEOW_SLURM_CLUSTER", "slurm")
    assert b.cpus_per_task() == 6
    assert b.timeout_min() == 42
    assert b.slurm_partition() == "cpu"
    assert b.slurm_cluster() == "slurm"
    assert b.max_workers() == 6  # falls back to cpus_per_task


def test_make_executor_honours_resource_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """make_executor reads the cpus/timeout/partition env vars."""
    pytest.importorskip("submitit")
    from examples.papers import dichroic_designer_slurm as ds

    monkeypatch.setenv("MEOW_CPUS_PER_TASK", "3")
    monkeypatch.setenv("MEOW_TIMEOUT_MIN", "7")
    executor = ds.make_executor(folder=tmp_path / "e", cluster="local")
    params = executor._executor.parameters
    assert params["cpus_per_task"] == 3
    assert params["timeout_min"] == 7


def test_example_defaults_are_128_cells_and_8_modes() -> None:
    """The example EME functions default to 128 cells and 8 modes."""
    import inspect

    from examples.papers import _analysis
    from examples.papers import dichroic_designer_slurm as ds
    from examples.papers import kwolek_designer_slurm as kslurm

    def default(fn: object, name: str) -> int:
        return inspect.signature(fn).parameters[name].default

    assert default(_analysis.dichroic_device_cells, "num_cells") == 128
    for fn in (ds.submit_runs, kslurm.submit_runs):
        assert default(fn, "num_cells") == 128
        assert default(fn, "num_modes") == 8


def test_slurm_load_run_and_corrupt_records_skipped(tmp_path: Path) -> None:
    """load_run targets a specific run dir; load_runs skips corrupt records."""
    import pickle

    from examples.papers import _slurm

    folder = tmp_path / "runs"
    good = folder / "20990101-000000-good"
    good.mkdir(parents=True)
    (good / "run.pkl").write_bytes(pickle.dumps({"label": "good", "ok": True}))
    bad = folder / "20990101-000001-bad"
    bad.mkdir(parents=True)
    (bad / "run.pkl").write_bytes(b"not a valid pickle \x00\x01\x02")

    # load_run loads a *specific* run by directory or by its run.pkl file
    assert _slurm.load_run(good)["label"] == "good"
    assert _slurm.load_run(good / "run.pkl")["ok"] is True

    # load_runs skips the corrupt/incompatible record (warns) instead of crashing
    with pytest.warns(UserWarning, match="unreadable run record"):
        runs = _slurm.load_runs(folder)
    assert [r["label"] for r in runs] == ["good"]


def test_kwolek_designer_broadband_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The non-slurm kwolek_designer analysis saves a >1-octave spectrum, GDS and
    FH/SH field plots for a design (run in-process via local worker threads)."""
    monkeypatch.setenv("MEOW_NUM_CELLS", "4")
    monkeypatch.setenv("MEOW_NUM_MODES", "2")
    monkeypatch.setenv("MEOW_SPECTRUM_NPTS", "3")

    d = kd.design_faquad_filter(
        kd.tfln_platform(0.30), 1.55, 0.775, w_top=1.2, res=0.08
    )
    out = tmp_path / "design"
    summary = kd.analyze_design(d, out, save_fields=True)

    assert summary["kind"] == "faquad"
    # the band spans more than one octave (0.8*SH .. 1.2*FH)
    lo, hi = summary["band_nm"]
    assert hi / lo > 2.0
    produced = {p.name for p in out.iterdir()}
    assert any(n.endswith("_spectrum.png") for n in produced)
    assert any(n.endswith("_propagation.png") for n in produced)
    assert any(n.endswith(".gds") for n in produced)
    assert 0.0 <= summary["fh_cross"] <= 1.0
    assert 0.0 <= summary["sh_bar"] <= 1.0


def test_dichroic_designer_submit_runs_then_gather(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """submit_runs ships one analysis job/design that writes plots + GDS, and
    gather_runs reloads its summary from a timestamped subfolder in 'session B'."""
    pytest.importorskip("submitit")
    _tiny_analysis_env(monkeypatch)
    monkeypatch.setenv("MEOW_PAPER_BACKEND", "tidy3d")
    from examples.papers import dichroic_designer_slurm as ds

    design = _minimal_slurm_design()
    folder = tmp_path / "runs"
    records = ds.submit_runs(
        [design],
        folder=folder,
        executor_factory=lambda sub: ds.make_executor(folder=sub, cluster="local"),
        num_cells=4,
        num_modes=2,
        device_res=0.1,
    )
    assert [r.label for r in records] == ["1000nm"]
    # the chosen solver backend is recorded for the distributed jobs + gather
    assert records[0].settings["backend"] == "tidy3d"
    run_dir = Path(records[0].out_dir)
    assert run_dir.parent == folder
    assert (run_dir / "run.pkl").exists()
    del records  # only the persisted run record remains, as in a later session

    gathered = ds.gather_runs(folder)
    assert set(gathered) == {"1000nm"}
    summary = gathered["1000nm"]
    # spectrum/propagation/design figures, GDS and data were produced
    produced = {p.name for p in run_dir.iterdir()}
    for suffix in ("_spectrum.png", "_propagation.png", "_design.png"):
        assert any(name.endswith(suffix) for name in produced)
    assert "1000nm.gds" in produced
    # tabular spectrum saved redundantly as CSV + JSON; mode fields as one HDF5
    assert "1000nm_spectrum.csv" in produced
    assert "1000nm_spectrum.json" in produced
    assert "1000nm_fields.h5" in produced
    assert "1000nm_summary.csv" in produced
    assert 0.0 <= summary["short_pass_at_cutoff"] <= 1.0
    assert 0.0 <= summary["long_pass_at_cutoff"] <= 1.0


def test_dichroic_coupler_single_submit_then_gather(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The single-coupler example submits one analysis job and reloads it later."""
    import asyncio

    pytest.importorskip("submitit")
    _tiny_analysis_env(monkeypatch)
    from examples.papers import dichroic_coupler_slurm as dc

    design = _minimal_slurm_design()
    folder = tmp_path / "coupler"

    saved = dc.submit(
        design,
        folder=folder,
        executor_factory=lambda sub: dc.make_executor(folder=sub, cluster="local"),
        num_cells=4,
        num_modes=2,
        device_res=0.1,
    )
    assert saved.label == dc.RECORD_LABEL
    run_dir = Path(saved.out_dir)
    assert (run_dir / "run.pkl").exists()
    del saved  # later session only has the persisted record

    summary = dc.gather(folder)
    asummary = asyncio.run(dc.agather(folder))
    assert summary == asummary
    produced = {p.name for p in run_dir.iterdir()}
    assert f"{dc.RECORD_LABEL}.gds" in produced
    assert any(n.endswith("_propagation.png") for n in produced)
    assert 0.0 <= summary["short_pass_at_cutoff"] <= 1.0
    assert 0.0 <= summary["long_pass_at_cutoff"] <= 1.0


def test_thickness_sweep_slurm_submit_then_gather(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The slurm thickness sweep submits one analysis job per (thickness, cutoff)
    design and reloads their summaries from per-run timestamped folders."""
    pytest.importorskip("submitit")
    _tiny_analysis_env(monkeypatch)
    from examples.papers import dichroic_designer_si3n4_thickness_slurm as ts

    design = _minimal_slurm_design()
    designs = [(200, design, 0.1), (100, design, 0.1)]
    folder = tmp_path / "thickness"
    records = ts.submit_runs(
        designs,
        folder=folder,
        executor_factory=lambda sub: ts.make_executor(folder=sub, cluster="local"),
        num_cells=4,
        num_modes=2,
        save_fields=False,  # spectrum-only keeps the smoke test light
    )
    assert {r.label for r in records} == {"200nm-1000nm", "100nm-1000nm"}
    del records

    gathered = ts.gather_runs(folder)
    assert set(gathered) == {"200nm-1000nm", "100nm-1000nm"}
    for label, summary in gathered.items():
        run_dir = Path(summary["out_dir"])
        produced = {p.name for p in run_dir.iterdir()}
        assert any(n.endswith("_spectrum.png") for n in produced)
        assert f"{label}.gds" in produced
        assert summary["saved_fields"] is False


# --- Kwolek 2026: TFLN FAQUAD combiner ---


@pytest.fixture(scope="module")
def calibration() -> tuple[float, float, float]:
    return kw.calibrate(1.55, res=0.06)


def test_calibration(calibration: tuple[float, float, float]) -> None:
    kappa_0, g_0, dbeta_dtw = calibration
    assert kappa_0 > 0
    assert 0.1 < g_0 < 1.5  # evanescent decay length in a plausible range
    assert dbeta_dtw > 0  # wider waveguide -> higher beta


def test_faquad_design_profiles(calibration: tuple[float, float, float]) -> None:
    design = kw.FaquadDesign(*calibration)
    # gap profile: constant g_m in region I, smoothly opening along the Euler
    # S-bend through the decoupling gap g_c to the final gap g_f
    assert np.isclose(design.gap(0.0), kw.G_M)
    assert np.isclose(design.gap(design.l_m / 2), kw.G_M)
    assert np.isclose(design.gap(design.z_c), kw.G_C, atol=1e-6)
    assert np.isclose(design.gap(design.half_length), kw.G_F)
    # gap is smooth and monotonically non-decreasing away from the center
    zc = np.linspace(0.0, design.half_length, 400)
    assert np.all(np.diff(design.gap(zc)) >= -1e-9)
    # adiabaticity parameter normalizes chi to sweep the full 0 -> pi range
    assert design.eta > 0
    assert np.isclose(design.chi(design.half_length), np.pi, atol=1e-6)
    # mixing angle: chi(0) = pi/2, antisymmetric about the center
    assert np.isclose(design.chi(0.0), np.pi / 2)
    z = np.linspace(-design.half_length, design.half_length, 31)
    assert np.allclose(np.cos(design.chi(z)), -np.cos(design.chi(-z)), atol=1e-6)
    assert np.all(np.diff(design.chi(z)) >= -1e-9)  # monotonic 0 -> pi
    # top-width difference: antisymmetric, returns to zero at the device ends,
    # and stays within the fabrication limit
    dtw = design.dtw(z)
    assert np.allclose(dtw, -dtw[::-1], atol=1e-9)
    assert np.isclose(design.dtw(0.0), 0.0, atol=1e-9)
    assert np.isclose(design.dtw(design.half_length), 0.0, atol=1e-9)
    assert np.isclose(design.dtw(-design.half_length), 0.0, atol=1e-9)
    assert np.max(np.abs(dtw)) <= design.dtw_max + 1e-12


def test_ln_material_anisotropy() -> None:
    ln = kw.ln_material(1.55)
    assert isinstance(ln, mw.AnisotropicMaterial)
    assert not ln.is_isotropic
    eps = np.real(np.diag(ln.eps))
    ne, no_y, no_z = np.sqrt(eps)
    assert np.isclose(no_y, no_z)
    assert ne < no_y  # negative uniaxial crystal
    assert 2.1 < ne < 2.2
    assert 2.2 < no_y < 2.3


def test_ln_material_isotropic_uses_extraordinary_index() -> None:
    """The fake isotropic LN puts ne on every axis (no birefringence)."""
    aniso = kw.ln_material(1.55, "anisotropic")
    iso = kw.ln_material(1.55, "isotropic")
    ne = np.sqrt(np.real(aniso.eps[0, 0]))
    eps_iso = np.real(np.diag(iso.eps))
    assert np.allclose(eps_iso, ne**2)  # ne on all three axes
    assert np.allclose(np.sqrt(eps_iso), np.sqrt(eps_iso[0]))  # truly isotropic


def test_faquad_three_region_cubic_geometry(
    calibration: tuple[float, float, float],
) -> None:
    """Region I constant gap, Region II cubic (Eq. 9), Region III Euler."""
    design = kw.FaquadDesign(*calibration)
    # the cubic Region II reaches the decoupling gap g_c exactly at z_ii
    assert np.isclose(design.gap(design.z_ii), kw.G_C, atol=1e-6)
    assert design.l_m / 2 < design.z_ii < design.half_length
    # the cubic gap matches Eq. 9 in Region II
    z = 0.5 * (design.l_m / 2 + design.z_ii)
    expected = (2.0 / 3.0) * design.a**2 * (z - design.l_m / 2) ** 3 + kw.G_M
    assert np.isclose(design.gap(z), expected, rtol=1e-3)
    # the closed-form coupling-envelope length z_0 = (3 g_0 / (2 a^2))^(1/3)
    assert np.isclose(design.z_0, (3 * design.g_0 / (2 * design.a**2)) ** (1 / 3))


def test_faquad_variants_share_geometry_differ_in_taper(
    calibration: tuple[float, float, float],
) -> None:
    """The Fig. 2a variants share the gap profile but differ in dTW."""
    designs = {v: kw.FaquadDesign(*calibration, variant=v) for v in kw.VARIANTS}
    ref = designs["faquad_bends"]
    zs = np.linspace(-ref.half_length, ref.half_length, 121)  # spans all regions
    for v, d in designs.items():
        assert np.allclose(d.gap(zs), ref.gap(zs))  # same physical gap/length
        assert np.isclose(d.half_length, ref.half_length)
        dtw = d.dtw(zs)
        assert np.allclose(dtw, -dtw[::-1], atol=1e-9)  # antisymmetric
        assert np.isclose(d.dtw(0.0), 0.0, atol=1e-9)
        if v != "faquad_bends":  # a different taper law -> different profile
            assert not np.allclose(d.dtw(zs), ref.dtw(zs))


def test_combiner_has_ring_closure_port(
    calibration: tuple[float, float, float],
) -> None:
    """The parametric-path combiner exposes in_cross (for ring closure)."""
    c = kw.faquad_combiner(*calibration)
    assert "in_cross" in {p.name for p in c.ports}


def test_combiner_layout(calibration: tuple[float, float, float]) -> None:
    c = kw.faquad_combiner(*calibration)
    port_names = {p.name for p in c.ports}
    assert {"in_bar", "out_bar", "out_cross"} <= port_names
    polys = [p for ps in c.get_polygons().values() for p in ps]
    assert len(polys) == 2
    design = kw.FaquadDesign(*calibration)
    assert np.isclose(c.xmax - c.xmin, 2 * design.half_length, atol=0.2)


def test_combiner_extrusion_has_angled_sidewalls(
    calibration: tuple[float, float, float],
) -> None:
    c = kw.faquad_combiner(*calibration)
    structs = kw.device_structures(c, 1.55)
    prisms = [s for s in structs if isinstance(s.geometry, mw.Prism)]
    assert len(prisms) == 2
    for s in prisms:
        assert s.geometry.sidewall_angle == kw.SIDEWALL_DEG
        assert isinstance(s.material, mw.AnisotropicMaterial)


def test_sh_stays_in_bar_port(calibration: tuple[float, float, float]) -> None:
    """At the second harmonic the mode is decoupled and stays in its port.

    This is the dichroic behavior of the combiner (paper Fig. 1f): even a
    coarse model shows orders of magnitude between bar and cross at SH.
    """
    c = kw.faquad_combiner(*calibration)
    cells = kw.device_cells(c, 0.775, num_cells=10, res=0.06)
    t_bar, t_cross = kw.bar_cross_transmission(cells, 0.775, num_modes=6)
    # bar/cross is measured by projecting onto the *rib-guided* modes (the slab
    # continuum is excluded), summed per side. At this very coarse resolution
    # the SH bar/cross contrast is strong; note that the converged SH extinction
    # is more modest (a few-x / ~7 dB), as the SH cross-port leakage is itself
    # resolution-dependent and only fully resolves on a finer mesh.
    assert t_bar > 20 * t_cross


# --- Generalized FAQUAD wavelength-filter designer (Kwolek 2026) ---


def test_lt_material_anisotropy() -> None:
    lt = kd.lt_material(1.55)
    assert isinstance(lt, mw.AnisotropicMaterial)
    assert not lt.is_isotropic
    ne, no_y, no_z = np.sqrt(np.real(np.diag(lt.eps)))
    assert np.isclose(no_y, no_z)
    assert ne > no_y  # LiTaO3 is (weakly) positive uniaxial: ne > no
    assert 2.1 < no_y < 2.25


def test_tfplatform_geometry() -> None:
    p = kd.tfln_platform(0.50, etch_depth=0.2)
    assert p.core is kd.ln_material
    assert p.name == "TFLN-500nm"
    assert p.slab_thickness == pytest.approx(0.30)
    assert p.sidewall_run == pytest.approx(0.2 * np.tan(np.deg2rad(p.sidewall_deg)))
    # fully etched -> no slab
    assert kd.tflt_platform(0.30, etch_depth=0.30).slab_thickness == pytest.approx(0.0)


def test_platform_matrix_covers_materials_and_thicknesses() -> None:
    platforms = kd.platform_matrix()
    assert len(platforms) == len(kd.MATERIALS) * len(kd.CORE_THICKNESSES)  # 2 x 4
    names = {p.name for p in platforms}
    assert "TFLN-300nm" in names
    assert "TFLT-600nm" in names
    # the three FH/SH pairs are octave-spaced (second harmonic)
    for fh, sh in kd.WAVELENGTH_PAIRS:
        assert np.isclose(sh, fh / 2, atol=1e-9)


def test_faquad_combiner_layout() -> None:
    design = kw.FaquadDesign(0.05, 0.4, 0.2)
    c = kd.faquad_combiner(design, w_top=1.2)
    port_names = {p.name for p in c.ports}
    assert {"in_bar", "out_bar", "out_cross"} <= port_names
    polys = [p for ps in c.get_polygons().values() for p in ps]
    assert len(polys) == 2
    assert np.isclose(c.xmax - c.xmin, 2 * design.half_length, atol=0.3)


def test_design_faquad_filter_dichroic() -> None:
    """A designed filter couples the FH far more strongly than the SH.

    Passing ``w_top`` skips the (FDE) width optimization; the calibration still
    runs, so the SH coupling at the minimum gap must be well below the FH one -
    the basis of the dichroic FH(cross)/SH(bar) behavior - and the device must
    fit the platform length budget.
    """
    p = kd.tfln_platform(0.30)
    d = kd.design_faquad_filter(p, 1.55, 0.775, w_top=1.2, res=0.07)
    assert d.platform.name == "TFLN-300nm"
    assert d.total_length <= p.max_length + 1.0
    kappa_fh_gm = d.kappa_0 * np.exp(-p.g_m / d.g_0)
    assert d.kappa_sh < kappa_fh_gm  # SH decoupled relative to the FH
    assert 0.2 < d.g_0 < 0.8  # plausible evanescent decay length [um]
    assert d.eta > 0
    # the layout matches the designed half-length
    assert np.isclose(
        d.component.xmax - d.component.xmin, 2 * d.design.half_length, atol=0.3
    )


def test_optimize_width_within_bounds() -> None:
    p = kd.tfln_platform(0.30, max_length=1500.0)
    w = kd.optimize_width(p, 1.55, target_extinction_db=18.0, res=0.08)
    assert 0.6 <= w <= 2.0


def test_optimize_width_gradient_matches_scale_of_bisection() -> None:
    """The AD gradient-based width optimizer lands in a physically sane range.

    Both optimizers maximize the same FH/SH contrast subject to a length-budget
    feasibility constraint (bisection-on-feasibility vs. jax.grad + a soft
    penalty), so their widths should agree to within a reasonable margin - not
    exactly (different objectives/tolerances), but the same order.
    """
    p = kd.tfln_platform(0.30, max_length=1500.0)
    w_bisect = kd.optimize_width(p, 1.55, target_extinction_db=18.0, res=0.08)
    w_grad, trace = kd.optimize_width_gradient(
        p, 1.55, 0.775, target_extinction_db=18.0, w0=0.8, res=0.08, steps=10
    )
    assert 0.4 <= w_grad <= 2.0
    assert abs(w_grad - w_bisect) < 0.5  # same order of magnitude
    assert len(trace.losses) == 11  # steps + the final converged point
    assert trace.objective_name


def test_design_faquad_filter_gradient_path_populates_trace() -> None:
    """design_faquad_filter(use_gradient=True) runs the AD optimizer and stores it."""
    p = kd.tfln_platform(0.30)
    d = kd.design_faquad_filter(
        p, 1.55, 0.775, res=0.08, use_gradient=True, gradient_w0=0.8, gradient_steps=6
    )
    assert d.opt_trace is not None
    assert len(d.opt_trace.losses) == 7
    assert 0.4 <= d.w_top <= 2.0


def test_kwolek_slurm_executor_and_cells(tmp_path) -> None:  # noqa: ANN001
    executor = ks.make_executor(folder=tmp_path / "jobs", cluster="debug")
    assert hasattr(executor, "submit")

    d = kd.design_faquad_filter(
        kd.tfln_platform(0.30), 1.55, 0.775, w_top=1.2, res=0.08
    )
    cells = ks.device_cells(d, 1.55, num_cells=5, res=0.1)
    assert len(cells) == 5
    assert cells[0].z_min == pytest.approx(0.0)
    assert cells[-1].z_max == pytest.approx(float(d.component.xmax))


def test_kwolek_slurm_blocking_and_concurrent_agree(tmp_path) -> None:  # noqa: ANN001
    """Blocking and async EME paths give identical FoM (debug executor)."""
    import asyncio

    d = kd.design_faquad_filter(
        kd.tfln_platform(0.30), 1.55, 0.775, w_top=1.2, res=0.08
    )
    eme_kwargs = {"num_cells": 5, "num_modes": 2, "res": 0.1}

    blocking = ks.run_blocking(
        [d],
        executor=ks.make_executor(folder=tmp_path / "b", cluster="debug"),
        **eme_kwargs,
    )
    concurrent = asyncio.run(
        ks.run_concurrent(
            [d],
            executor=ks.make_executor(folder=tmp_path / "c", cluster="debug"),
            **eme_kwargs,
        )
    )
    key = "TFLN-300nm/1550-775nm"
    assert set(blocking) == set(concurrent) == {key}
    assert blocking[key]["fh_cross"] == pytest.approx(concurrent[key]["fh_cross"])
    assert blocking[key]["sh_bar"] == pytest.approx(concurrent[key]["sh_bar"])
    # power stays physical
    assert 0.0 <= blocking[key]["fh_cross"] <= 1.0
    assert 0.0 <= blocking[key]["sh_bar"] <= 1.0


def test_kwolek_slurm_submit_then_gather(tmp_path) -> None:  # noqa: ANN001
    """submit_designs writes one FH and one SH record per design; gather_results
    reloads them from the folder and recombines the FH/SH figures of merit."""
    pytest.importorskip("submitit")

    d = kd.design_faquad_filter(
        kd.tfln_platform(0.30), 1.55, 0.775, w_top=1.2, res=0.08
    )
    eme_kwargs = {"num_cells": 5, "num_modes": 2, "res": 0.1}
    folder = tmp_path / "shared"

    records = ks.submit_designs(
        [d],
        executor=ks.make_executor(folder=folder, cluster="local"),
        folder=folder,
        **eme_kwargs,
    )
    key = "TFLN-300nm/1550-775nm"
    assert {r.label for r in records} == {f"{key}|fh", f"{key}|sh"}
    del records

    gathered = ks.gather_results(folder)
    blocking = ks.run_blocking(
        [d],
        executor=ks.make_executor(folder=tmp_path / "block", cluster="local"),
        **eme_kwargs,
    )
    assert set(gathered) == {key}
    assert gathered[key]["fh_cross"] == pytest.approx(blocking[key]["fh_cross"])
    assert gathered[key]["sh_bar"] == pytest.approx(blocking[key]["sh_bar"])


def test_kwolek_slurm_submit_runs_then_gather(
    tmp_path,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """submit_runs distributes each FH/SH sweep point as its own slice-group job
    (spectrum-only here); gather_runs assembles the FH/SH spectra + GDS."""
    pytest.importorskip("submitit")
    monkeypatch.setenv("MEOW_SPECTRUM_NPTS", "2")

    d = kd.design_faquad_filter(
        kd.tfln_platform(0.30), 1.55, 0.775, w_top=1.2, res=0.08
    )
    folder = tmp_path / "runs"
    records = ks.submit_runs(
        [d],
        folder=folder,
        executor_factory=lambda sub: ks.make_executor(folder=sub, cluster="local"),
        num_cells=4,
        num_modes=2,
        device_res=0.1,
        save_fields=False,  # spectrum-only keeps the smoke test light
    )
    key = "TFLN-300nm/1550-775nm"
    assert [r.label for r in records] == [key]
    run_dir = Path(records[0].out_dir)
    del records

    gathered = ks.gather_runs(folder)
    assert set(gathered) == {key}
    summary = gathered[key]
    produced = {p.name for p in run_dir.iterdir()}
    for suffix in ("_spectrum.png", "_design.png"):
        assert any(name.endswith(suffix) for name in produced)
    assert f"{key.replace('/', '_')}.gds" in produced
    assert summary["kind"] == "faquad"
    assert summary["saved_fields"] is False
    assert 0.0 <= summary["fh_cross"] <= 1.0
    assert 0.0 <= summary["sh_bar"] <= 1.0


# --- Ramadan 1998: adiabatic-coupler design rules (analytic) ---


def test_ramadan1998_analytic_design_rules() -> None:
    """The transcribed eqs reproduce the paper's key relations and scalings."""
    from examples.papers import ramadan1998 as ram

    # eq (10) -> its eq (9) asymptote at large asynchronicity
    x_io, eta = 119.0, 420.0
    assert ram.q_region1(x_io, 8.0, eta) == pytest.approx(
        ram.q_region1_asymptote(x_io, 8.0, eta), rel=0.05
    )
    # eq (15): nulls at kappa_II L_II = n*pi, envelope bounds the oscillation
    kappa = 0.005
    null = np.pi / kappa  # first null length
    assert ram.q_region2(null, kappa_ii=kappa, x_iio=0.66) < 1e-9
    peak_len = 1500.0
    assert ram.q_region2(
        peak_len, kappa_ii=kappa, x_iio=0.66
    ) <= ram.q_region2_envelope(peak_len, kappa_ii=kappa, x_iio=0.66)
    # eq (28)/(29): 3 dB scales as 1/eps, full as 1/sqrt(eps); 3 dB is longer
    l_3db = ram.length_3db_optimum(0.01, kappa)
    l_full = ram.length_full_optimum(0.01, kappa)
    assert l_3db > l_full
    assert ram.length_3db_optimum(0.005, kappa) == pytest.approx(2 * l_3db, rel=1e-9)
    assert ram.length_full_optimum(0.0025, kappa) == pytest.approx(
        2 * l_full, rel=1e-9
    )
    assert ram.coupling_length(kappa) == pytest.approx(np.pi / (2 * kappa))


def test_ramadan1998_designer_coupling_and_lengths() -> None:
    """The designer derives kappa_II from FDE and sizes a longer 3 dB coupler."""
    from examples.papers import ramadan1998_designer as rd

    stack = rd.soi_stack(wl=1.31)
    kappa = rd.coupling_coefficient(stack, res=0.05)
    assert kappa > 0
    spec_3db = rd.design(stack, kind="3dB", epsilon=0.02, res=0.05)
    spec_full = rd.design(stack, kind="full", epsilon=0.02, res=0.05)
    assert spec_3db.length_um > spec_full.length_um
    assert spec_3db.coupling_length_um == pytest.approx(np.pi / (2 * kappa), rel=1e-6)
