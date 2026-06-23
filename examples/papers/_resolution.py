"""Low / medium / high resolution selection for the paper examples.

Replaces the old boolean ``MEOW_EXAMPLE_FAST`` switch with a three-level
resolution knob, ``MEOW_EXAMPLE_RES`` in ``{"low", "medium", "high"}``:

- **low**: a coarse smoke-test resolution (used by the test suite; what
  ``MEOW_EXAMPLE_FAST=1`` used to select);
- **medium** (default): the previous full-quality settings;
- **high**: finer mesh resolution, more modes per cross-section and more EME
  cells - increased to the point where the simulated quantities are expected to
  be converged (slow).

Use :func:`pick` to choose a per-knob value for the active level, e.g.
``num_modes = pick(low=2, medium=4, high=6)``. ``MEOW_EXAMPLE_FAST=1`` is still
honoured (mapped to ``low``) for backwards compatibility.
"""

from __future__ import annotations

import os
from typing import TypeVar

T = TypeVar("T")

LEVELS = ("low", "medium", "high")
DEFAULT_LEVEL = "medium"


def level() -> str:
    """The active resolution level (from ``MEOW_EXAMPLE_RES``; default medium).

    Accepts the full names or a unique prefix (``l``/``m``/``h``). Falls back to
    ``low`` when the legacy ``MEOW_EXAMPLE_FAST`` is set, else to
    :data:`DEFAULT_LEVEL`.
    """
    val = os.environ.get("MEOW_EXAMPLE_RES")
    if val:
        v = val.strip().lower()
        for lvl in LEVELS:
            if lvl == v or lvl.startswith(v):
                return lvl
        msg = f"MEOW_EXAMPLE_RES must be one of {LEVELS!r}, got {val!r}."
        raise ValueError(msg)
    if os.environ.get("MEOW_EXAMPLE_FAST", "0") not in ("0", "", "false", "False"):
        return "low"
    return DEFAULT_LEVEL


def pick(*, low: T, medium: T, high: T) -> T:
    """Return the value for the active resolution level."""
    return {"low": low, "medium": medium, "high": high}[level()]


def is_low() -> bool:
    """Whether the active level is the coarse ``low`` (smoke-test) resolution."""
    return level() == "low"
