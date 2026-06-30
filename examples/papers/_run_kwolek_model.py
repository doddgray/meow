"""Internal helper: run the Kwolek figure set for one LiNbO3 material model.

Used to fan the two material models out across processes; not part of the public
example API. ``python -m examples.papers._run_kwolek_model <model>``.
"""
from __future__ import annotations

import json
import sys

import matplotlib as mpl

mpl.use("Agg")

from examples.papers import kwolek2026_figures as kwfig


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "anisotropic"
    kwfig.FIGDIR.mkdir(exist_ok=True, parents=True)
    out = {
        "fig1": kwfig.figure1(model),
        "fig5": kwfig.figure5(model),
        "broadband": kwfig.figure_broadband(model),
        "fig2": kwfig.figure2(model),
    }
    print(json.dumps({model: out}, indent=2, default=str))


if __name__ == "__main__":
    main()
