"""Tests for the xarray-based EME result I/O (meow.eme.io)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import meow as mw


def _two_cell_modes() -> list[list[mw.Mode]]:
    """Solve 2 modes on each of 2 tiny cross-sections (a minimal field stack)."""
    structs = [
        mw.Structure(
            material=mw.silicon,
            geometry=mw.Prism(
                poly=np.array([(-0.2, -0.1), (0.2, -0.1), (0.2, 0.1), (-0.2, 0.1)]),
                h_min=0.0,
                h_max=0.22,
                axis="y",
            ),
        ),
    ]
    mesh = mw.Mesh2D(
        x=np.linspace(-1.0, 1.0, 21),
        y=np.linspace(-1.0, 1.0, 21),
    )
    cells = mw.create_cells(structs, mesh, [1.0, 1.0], z_min=0.0)
    env = mw.Environment(wl=1.55, T=25.0)
    return [
        mw.compute_modes(mw.CrossSection.from_cell(cell=c, env=env), num_modes=2)
        for c in cells
    ]


def test_modes_to_dataset_and_hdf5_roundtrip(tmp_path: Path) -> None:
    """Mode fields save to a single compressed HDF5 and reload exactly (complex)."""
    modes = _two_cell_modes()
    ds = mw.modes_to_dataset(modes, attrs={"note": "test"})
    for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert comp in ds.data_vars
        assert np.iscomplexobj(ds[comp].values)
    assert ds.neff.shape == (2, 2)

    path = tmp_path / "fields.h5"
    out = mw.save_fields(ds, path)
    assert out == path
    assert path.exists()

    loaded = mw.load_fields(path)
    assert np.iscomplexobj(loaded.Ex.values)
    assert np.iscomplexobj(loaded.neff.values)
    assert np.allclose(loaded.Ex.values, ds.Ex.values)
    assert np.allclose(loaded.neff.values, ds.neff.values)


def test_save_fields_is_compressed(tmp_path: Path) -> None:
    """gzip compression is applied: a compressible field shrinks vs uncompressed."""
    import xarray as xr

    field = np.zeros((4, 4, 64, 64), dtype=np.complex128)
    ds = xr.Dataset({"Ex": (("cell", "mode", "Ex_y", "Ex_x"), field)})
    raw_bytes = field.size * field.dtype.itemsize  # ~1 MB
    compressed = mw.save_fields(ds, tmp_path / "fields.h5", complevel=9)
    # the highly compressible (all-zero) field shrinks far below its raw size
    assert compressed.stat().st_size < raw_bytes / 4


def test_field_mode_handle_save_fields(tmp_path: Path) -> None:
    """ParallelFieldModeJobs.save_fields collects modes and writes the HDF5."""
    from concurrent.futures import ThreadPoolExecutor

    structs = [
        mw.Structure(
            material=mw.silicon,
            geometry=mw.Prism(
                poly=np.array([(-0.2, -0.1), (0.2, -0.1), (0.2, 0.1), (-0.2, 0.1)]),
                h_min=0.0,
                h_max=0.22,
                axis="y",
            ),
        ),
    ]
    mesh = mw.Mesh2D(x=np.linspace(-1, 1, 21), y=np.linspace(-1, 1, 21))
    cells = mw.create_cells(structs, mesh, [1.0, 1.0], z_min=0.0)
    env = mw.Environment(wl=1.55, T=25.0)
    handle = mw.submit_cell_modes(
        cells, env, executor=ThreadPoolExecutor(max_workers=2), num_modes=2
    )
    path = handle.save_fields(tmp_path / "h.h5")
    assert path.exists()
    loaded = mw.load_fields(path)
    assert loaded.Ex.sizes["cell"] == 2


def test_save_table_writes_csv_and_json(tmp_path: Path) -> None:
    """Tabular data is written redundantly to both CSV and JSON."""
    data = {"wavelength_um": [1.5, 1.55, 1.6], "bar": [0.9, 0.5, 0.1]}
    csv_path, json_path = mw.save_table(tmp_path / "spectrum", data)
    assert csv_path.name == "spectrum.csv"
    assert json_path.name == "spectrum.json"
    assert csv_path.exists()
    assert json_path.exists()

    with json_path.open() as f:
        assert json.load(f) == {
            "wavelength_um": [1.5, 1.55, 1.6],
            "bar": [0.9, 0.5, 0.1],
        }
    import pandas as pd

    frame = pd.read_csv(csv_path)
    assert list(frame.columns) == ["wavelength_um", "bar"]
    assert frame["bar"].tolist() == [0.9, 0.5, 0.1]
    assert mw.load_table(tmp_path / "spectrum")["bar"] == [0.9, 0.5, 0.1]


def test_save_summary_writes_csv_and_json(tmp_path: Path) -> None:
    """Scalar summaries are written redundantly to both CSV and JSON (one row)."""
    summary = {"kind": "faquad", "band_nm": [700, 1900], "fh_cross": 0.83}
    csv_path, json_path = mw.save_summary(tmp_path / "summary", summary)
    assert csv_path.exists()
    assert json_path.exists()

    with json_path.open() as f:
        assert json.load(f) == summary
    import pandas as pd

    frame = pd.read_csv(csv_path)
    assert len(frame) == 1
    assert frame["kind"].iloc[0] == "faquad"
    # the list field is JSON-encoded so the CSV stays a single row
    assert json.loads(frame["band_nm"].iloc[0]) == [700, 1900]


def test_modes_to_dataset_requires_modes() -> None:
    with pytest.raises(ValueError, match="at least one cell"):
        mw.modes_to_dataset([])
