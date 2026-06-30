"""Shared helpers for the wavelength-varying designer examples.

Three reusable pieces used by the designers that sweep a target wavelength
(``kwolek_designer`` and the ``dichroic_designer`` family):

1. :func:`spectrum_grid` -- a column of dense broad-band transmission spectra,
   one row per design, all sharing the same wavelength axis, with each design's
   target wavelength(s) marked by dashed vertical lines.
2. :func:`tapered_ports` -- add linear access tapers to named ports of a device,
   from the device's own edge width to a target port width over a given taper
   length. Controlled per-port by keyword; the default (no entry) is *no taper*
   -- the device keeps the width at its edge.
3. :func:`coupler_cutback_array` / :func:`splitter_tree` -- test-structure arrays
   of a 2x2 coupler with a *varied number of cross/bar couplings* but a
   *constant total waveguide length* between regularly-spaced ports on either
   side of a (default 5 mm) chip, the standard cut-back layout for extracting the
   per-coupler excess loss.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gdsfactory as gf
import numpy as np


# --------------------------------------------------------------------------
# 1. column grid of broad-band transmission spectra
# --------------------------------------------------------------------------
def spectrum_grid(
    rows: Sequence[dict[str, Any]],
    out_path: Any,
    *,
    title: str = "",
    xlim_nm: tuple[float, float] | None = None,
    ports: Sequence[tuple[str, str]] = (("bar", "C0"), ("cross", "C3")),
    db: bool = False,
) -> Any:
    """Plot a column of broad-band transmission spectra (one row per design).

    Every row shares the same wavelength axis (so designs are directly
    comparable), and each design's target wavelength(s) are drawn as dashed
    vertical lines.

    Args:
        rows: one dict per design with keys ``"label"`` (row title), ``"wls"``
            (wavelengths [um]), one array per entry in ``ports`` (keyed by the
            port name, e.g. ``"bar"``/``"cross"``), and ``"design_wls"`` (the
            target wavelength(s) [um] to mark with dashed lines).
        out_path: file path to save the figure to.
        title: overall figure title.
        xlim_nm: shared x-axis bounds [nm]; inferred from the data if ``None``.
        ports: ``(key, color)`` pairs naming the transmission arrays to plot.
        db: plot transmission in dB (else linear 0..1).

    Returns:
        the saved figure path (``out_path``).
    """
    import matplotlib.pyplot as plt

    n = len(rows)
    fig, axes = plt.subplots(
        n, 1, figsize=(9, max(2.2 * n, 2.5)), squeeze=False, sharex=True
    )
    axes = axes[:, 0]

    if xlim_nm is None:
        lo = min(float(np.min(r["wls"])) for r in rows) * 1e3
        hi = max(float(np.max(r["wls"])) for r in rows) * 1e3
        xlim_nm = (lo, hi)

    for ax, row in zip(axes, rows, strict=True):
        wls_nm = np.asarray(row["wls"]) * 1e3
        for key, color in ports:
            if key not in row:
                continue
            y = np.asarray(row[key], dtype=float)
            if db:
                y = 10 * np.log10(np.clip(y, 1e-6, None))
            ax.plot(wls_nm, y, color=color, label=key)
        for wl in np.atleast_1d(row.get("design_wls", [])):
            ax.axvline(float(wl) * 1e3, color="0.4", ls="--", lw=1)
        ax.set_xlim(*xlim_nm)
        ax.set_ylabel("T [dB]" if db else "T")
        ax.set_title(row.get("label", ""), fontsize=9, loc="left")
        ax.grid(visible=True, alpha=0.4)
        ax.legend(fontsize=7, ncol=len(ports), loc="center right")
    axes[-1].set_xlabel("wavelength [nm]")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# 2. per-port access tapers
# --------------------------------------------------------------------------
def tapered_ports(
    component: gf.Component,
    port_widths: dict[str, float] | None = None,
    taper_lengths: dict[str, float] | float = 20.0,
    *,
    layer: Any = (1, 0),
    default_length: float = 20.0,
) -> gf.Component:
    """Return a copy of ``component`` with linear access tapers on named ports.

    For each port named in ``port_widths`` a straight linear taper is attached,
    widening/narrowing from the device's own edge width at that port to the
    requested target width over the port's taper length, and the taper's far end
    becomes the new port (same name). Ports not listed are left untouched.

    The default ``port_widths=None`` adds *no* tapers -- the device keeps the
    widths it was designed with at its edges (the requested default behavior).

    Args:
        component: the device to wrap.
        port_widths: ``{port_name: target_width_um}``; ``None`` -> no tapers.
        taper_lengths: a single length [um] for all tapered ports, or a
            ``{port_name: length_um}`` mapping (missing ports use
            ``default_length``).
        layer: the waveguide layer for the tapers.
        default_length: taper length [um] for ports absent from a
            ``taper_lengths`` mapping.

    Returns:
        a new component with the tapers added and the named ports moved to the
        taper tips (or the original component if ``port_widths`` is empty).
    """
    if not port_widths:
        return component
    c = gf.Component()
    ref = c << component
    kept = {p.name for p in component.ports} - set(port_widths)
    for name in kept:
        c.add_port(name, port=ref.ports[name])
    for name, w_target in port_widths.items():
        port = ref.ports[name]
        if isinstance(taper_lengths, dict):
            length = float(taper_lengths.get(name, default_length))
        else:
            length = float(taper_lengths)
        w_edge = float(port.width)
        taper = c << gf.components.taper(
            length=length, width1=w_edge, width2=float(w_target), layer=layer
        )
        taper.connect("o1", port)
        c.add_port(name, port=taper.ports["o2"])
    c.flatten()
    return c


# --------------------------------------------------------------------------
# 3. constant-length coupler cut-back / splitter-tree test structures
# --------------------------------------------------------------------------
def coupler_cutback(
    coupler: gf.Component,
    n_couplers: int,
    *,
    in_port: str,
    thru_port: str,
    total_length: float,
    width: float = 1.0,
    layer: Any = (1, 0),
    y: float = 0.0,
) -> gf.Component:
    """One cut-back row: ``n_couplers`` couplers in series at constant length.

    The couplers are spliced in series along a bus (output ``thru_port`` of one
    into the input ``in_port`` of the next); a length-matching straight is added
    so the **total waveguide length is ``total_length`` regardless of
    ``n_couplers``** (the cut-back condition that isolates the per-coupler excess
    loss from propagation loss). The input and output ports sit on the left
    (``x = 0``) and right (``x = total_length``) chip edges at height ``y``.
    """
    c = gf.Component()
    xs = gf.cross_section.cross_section(width=width, layer=layer)
    # measured along-axis length the couplers consume (their thru-port run)
    span = abs(
        float(coupler.ports[thru_port].center[0])
        - float(coupler.ports[in_port].center[0])
    )
    used = n_couplers * span
    # remaining straight, split before/after so ports land on the chip edges
    pad = max(total_length - used, 0.0)
    lead = c << gf.components.straight(length=pad / 2 + 1e-3, cross_section=xs)
    lead.dxmin, lead.dy = 0.0, y
    prev = lead.ports["o2"]
    for _ in range(n_couplers):
        ref = c << coupler
        ref.connect(in_port, prev)
        prev = ref.ports[thru_port]
    tail = c << gf.components.straight(length=pad / 2 + 1e-3, cross_section=xs)
    tail.connect("o1", prev)
    c.add_port("in", port=lead.ports["o1"])
    c.add_port("out", port=tail.ports["o2"])
    return c


def coupler_cutback_array(
    coupler: gf.Component,
    counts: Sequence[int] = (0, 1, 2, 4),
    *,
    in_port: str,
    thru_port: str,
    chip_width: float = 5000.0,
    pitch: float = 60.0,
    width: float = 1.0,
    layer: Any = (1, 0),
    total_length: float | None = None,
) -> gf.Component:
    """A cut-back **array**: one constant-length row per coupler count.

    Rows are stacked at a regular ``pitch`` (regularly-spaced ports on either
    side of the ``chip_width``-wide chip). Every row has the same total
    waveguide length (defaulting to the chip width plus the longest chain's
    coupler run), so a transmission cut-back over ``counts`` gives the
    per-coupler excess loss. ``counts`` mixes cross/bar couplings simply by the
    number of cascaded couplers.
    """
    span = abs(
        float(coupler.ports[thru_port].center[0])
        - float(coupler.ports[in_port].center[0])
    )
    if total_length is None:
        total_length = max(chip_width, max(counts) * span + 50.0)
    c = gf.Component()
    for i, n in enumerate(counts):
        row = c << coupler_cutback(
            coupler, n, in_port=in_port, thru_port=thru_port,
            total_length=total_length, width=width, layer=layer, y=i * pitch,
        )
        c.add_port(f"in_{n}", port=row.ports["in"])
        c.add_port(f"out_{n}", port=row.ports["out"])
    return c


def splitter_tree(
    coupler: gf.Component,
    n_stages: int = 2,
    *,
    in_port: str,
    bar_port: str,
    cross_port: str,
    x_pitch: float | None = None,
    y_leaf: float = 80.0,
    radius: float = 30.0,
    width: float = 1.0,
    layer: Any = (1, 0),
) -> gf.Component:
    """A binary 1 -> 2**``n_stages`` splitter tree built from the coupler.

    Each stage doubles the branch count by routing *both* outputs of every
    coupler (bar + cross) forward to the next stage with Euler bends, giving a
    balanced tree whose root-to-leaf paths all traverse ``n_stages`` couplings --
    the branching counterpart of the cut-back chain. Stages are placed at a
    regular ``x_pitch``; leaves are regularly spaced on the right at ``y_leaf``
    pitch. Routing failures (very deep / tight trees) are surfaced to the caller.
    """
    c = gf.Component()
    xs = gf.cross_section.cross_section(width=width, layer=layer)
    span = abs(
        float(coupler.ports[bar_port].center[0] - coupler.ports[in_port].center[0])
    )
    x_pitch = x_pitch or (span + 4 * radius + 20.0)
    n_leaves = 2**n_stages
    full = y_leaf * (n_leaves - 1)

    def place(stage: int, idx: int) -> gf.ComponentReference:
        ref = c << coupler
        block = full / (2**stage)
        y = -full / 2 + block * (idx + 0.5)
        ref.dx, ref.dy = stage * x_pitch, y
        return ref

    root = place(0, 0)
    c.add_port("in", port=root.ports[in_port])
    parents = [root]
    for stage in range(1, n_stages):
        children = []
        for pi, parent in enumerate(parents):
            for w, pport in ((0, bar_port), (1, cross_port)):
                child = place(stage, 2 * pi + w)
                gf.routing.route_single(
                    c, parent.ports[pport], child.ports[in_port],
                    cross_section=xs, radius=radius,
                )
                children.append(child)
        parents = children
    j = 0
    for parent in parents:
        for pport in (bar_port, cross_port):
            stub = c << gf.components.straight(length=20.0, cross_section=xs)
            stub.connect("o1", parent.ports[pport])
            c.add_port(f"out_{j}", port=stub.ports["o2"])
            j += 1
    return c
