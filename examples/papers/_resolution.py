"""Backwards-compatible shim - the resolution presets now live in meow.

The low / medium / high resolution selection (``MEOW_EXAMPLE_RES``) the examples
used to define here is now part of the main library, in :mod:`meow.settings`.
This module just re-exports it under its historical names.
"""

from __future__ import annotations

from meow.settings import (
    DEFAULT_LEVEL,
    HIGH_NUM_CELLS,
    HIGH_NUM_MODES,
    LEVELS,
    is_low,
    level,
    num_cells,
    num_modes,
    pick,
)

__all__ = [
    "DEFAULT_LEVEL",
    "HIGH_NUM_CELLS",
    "HIGH_NUM_MODES",
    "LEVELS",
    "is_low",
    "level",
    "num_cells",
    "num_modes",
    "pick",
]
