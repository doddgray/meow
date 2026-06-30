"""Internal helper: (re)generate only the Kwolek broad-band + Fig. 2 spectra for
one LiNbO3 material model (the EME-spectrum-heavy figures). Fig. 1 / Fig. 5 are
produced by ``_run_kwolek_model``. Not part of the public example API.
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
    out = {"broadband": kwfig.figure_broadband(model), "fig2": kwfig.figure2(model)}
    print(json.dumps({model: out}, indent=2, default=str))


if __name__ == "__main__":
    main()
