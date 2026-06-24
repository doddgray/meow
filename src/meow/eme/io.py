"""xarray-based saving and loading of EME results.

This replaces ad-hoc pickle / ``np.savez`` persistence of *computed result data*
with self-describing, portable formats:

- **mode field datasets** (the dense per-cell ``Ex/Ey/Ez/Hx/Hy/Hz`` fields and
  effective indices) are saved as a single **compressed HDF5** (netCDF) dataset
  with :func:`save_fields` / :func:`load_fields`. Complex fields are stored as
  split real/imaginary float variables so standard gzip compression applies and
  the files are readable by any netCDF/HDF5 stack;
- **less dense tabular data** (transmission spectra, effective-index tables,
  scalar summaries) is written **redundantly to both CSV and JSON** with
  :func:`save_table` / :func:`save_summary`, so it can be opened either with a
  dataframe tool or as plain JSON.

Pickle is intentionally *not* used here; it is kept only for reattaching to
in-flight submitit job handles (see :mod:`meow.eme.parallel`), whose live job
objects xarray cannot represent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

    from meow.mode import Mode

# the field components stored per mode
_FIELD_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


# ==========================================================================
# mode field datasets -> compressed HDF5
# ==========================================================================
def modes_to_dataset(
    modes_by_cell: list[list[Mode]], *, attrs: dict[str, Any] | None = None
) -> xr.Dataset:
    """Bundle the per-cell modes of an EME chain into an :class:`xarray.Dataset`.

    Args:
        modes_by_cell: the modes of each cell, in chain order (e.g. the result
            of :meth:`meow.ParallelFieldModeJobs.result`).
        attrs: optional dataset-level attributes (stored in the HDF5 header).

    Returns:
        A dataset with the complex field components ``Ex..Hz`` (dims
        ``(cell, mode, <comp>_y, <comp>_x)``) and the complex ``neff`` (dims
        ``(cell, mode)``), plus the cross-section mesh ``x``/``y`` coordinates.
    """
    import xarray as xr

    if not modes_by_cell or not modes_by_cell[0]:
        msg = "modes_by_cell must contain at least one cell with one mode."
        raise ValueError(msg)
    n_modes = min(len(cell) for cell in modes_by_cell)
    data_vars: dict[str, Any] = {}
    for comp in _FIELD_COMPONENTS:
        stack = np.array(
            [
                [np.asarray(getattr(m, comp)) for m in cell[:n_modes]]
                for cell in modes_by_cell
            ]
        )
        data_vars[comp] = (("cell", "mode", f"{comp}_y", f"{comp}_x"), stack)
    neff = np.array(
        [[complex(m.neff) for m in cell[:n_modes]] for cell in modes_by_cell]
    )
    data_vars["neff"] = (("cell", "mode"), neff)

    mesh = modes_by_cell[0][0].cs.mesh
    coords = {
        "mesh_x": ("mesh_x", np.asarray(mesh.x, dtype=float)),
        "mesh_y": ("mesh_y", np.asarray(mesh.y, dtype=float)),
    }
    return xr.Dataset(data_vars, coords=coords, attrs=attrs or {})


def _split_complex(ds: xr.Dataset) -> xr.Dataset:
    """Replace each complex variable with split ``_real``/``_imag`` float vars."""
    out = ds.copy()
    for name in list(out.data_vars):
        if np.iscomplexobj(out[name].values):
            var = out[name]
            out[f"{name}_real"] = var.real
            out[f"{name}_imag"] = var.imag
            out = out.drop_vars([name])
    out.attrs = {**out.attrs, "_complex_split": 1}
    return out


def _join_complex(ds: xr.Dataset) -> xr.Dataset:
    """Recombine ``_real``/``_imag`` float vars back into complex variables."""
    out = ds.copy()
    reals = [n for n in out.data_vars if str(n).endswith("_real")]
    for real_name in reals:
        base = str(real_name)[: -len("_real")]
        imag_name = f"{base}_imag"
        if imag_name in out.data_vars:
            out[base] = out[real_name] + 1j * out[imag_name]
            out = out.drop_vars([real_name, imag_name])
    out.attrs = {k: v for k, v in out.attrs.items() if k != "_complex_split"}
    return out


def save_fields(
    fields: xr.Dataset | list[list[Mode]],
    path: str | Path,
    *,
    complevel: int = 4,
) -> Path:
    """Save mode fields to a single compressed HDF5 (netCDF) file.

    Args:
        fields: an :class:`xarray.Dataset` (e.g. from :func:`modes_to_dataset`)
            or the raw per-cell modes to bundle first.
        path: the output ``.h5`` path.
        complevel: gzip compression level (0-9).

    Returns:
        The written path.
    """
    import xarray as xr

    ds = fields if isinstance(fields, xr.Dataset) else modes_to_dataset(fields)
    ds = _split_complex(ds)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        name: {"zlib": True, "complevel": complevel}
        for name in ds.data_vars
        if ds[name].ndim > 0
    }
    ds.to_netcdf(path, engine="h5netcdf", encoding=encoding)
    return path


def load_fields(path: str | Path) -> xr.Dataset:
    """Load mode fields written by :func:`save_fields` (recombining complex)."""
    import xarray as xr

    ds = xr.load_dataset(path, engine="h5netcdf")
    return _join_complex(ds)


# ==========================================================================
# tabular data -> redundant CSV + JSON
# ==========================================================================
def _table_paths(path_stem: str | Path) -> tuple[Path, Path]:
    stem = Path(path_stem)
    if stem.suffix in (".csv", ".json"):
        stem = stem.with_suffix("")
    return stem.with_suffix(".csv"), stem.with_suffix(".json")


def save_table(
    path_stem: str | Path, data: dict[str, Any]
) -> tuple[Path, Path]:
    """Write column data redundantly as both CSV and JSON.

    Args:
        path_stem: output path stem (any ``.csv``/``.json`` suffix is dropped);
            the two files ``<stem>.csv`` and ``<stem>.json`` are written.
        data: a mapping of column name -> 1-D array/sequence (equal lengths).

    Returns:
        ``(csv_path, json_path)``.
    """
    import pandas as pd

    csv_path, json_path = _table_paths(path_stem)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({k: np.asarray(v).ravel() for k, v in data.items()})
    frame.to_csv(csv_path, index=False)
    with json_path.open("w") as f:
        json.dump(
            {k: np.asarray(v).ravel().tolist() for k, v in data.items()},
            f,
            indent=2,
            default=str,
        )
    return csv_path, json_path


def save_summary(
    path_stem: str | Path, summary: dict[str, Any]
) -> tuple[Path, Path]:
    """Write a scalar-summary dict redundantly as both CSV (one row) and JSON.

    Args:
        path_stem: output path stem; writes ``<stem>.csv`` and ``<stem>.json``.
        summary: a flat mapping of name -> scalar (lists are JSON-encoded in the
            single CSV row so the CSV stays one line per summary).

    Returns:
        ``(csv_path, json_path)``.
    """
    import pandas as pd

    csv_path, json_path = _table_paths(path_stem)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        k: (json.dumps(v, default=str) if isinstance(v, (list, dict)) else v)
        for k, v in summary.items()
    }
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    return csv_path, json_path


def load_table(path_stem: str | Path) -> dict[str, Any]:
    """Load a table written by :func:`save_table` (preferring the JSON copy)."""
    csv_path, json_path = _table_paths(path_stem)
    if json_path.exists():
        with json_path.open() as f:
            return json.load(f)
    import pandas as pd

    return {k: v.tolist() for k, v in pd.read_csv(csv_path).to_dict("series").items()}
