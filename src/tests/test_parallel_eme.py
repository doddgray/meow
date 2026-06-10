"""Tests for parallelized EME via concurrent slice-group jobs."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import meow as mw
from meow.eme.parallel import _check_shared_cell_consistency

NUM_MODES = 3
WIDTHS = (0.4, 0.55, 0.7, 0.85)
CELL_LENGTH = 2.0

silicon = mw.IndexMaterial(name="si_parallel_test", n=3.45)


def _stepped_taper() -> tuple[list[mw.Cell], mw.Environment]:
    """A small stepped-width waveguide taper: 4 cells, 3 interfaces."""
    structs = [
        mw.Structure(
            material=silicon,
            geometry=mw.Box(
                x_min=-w / 2,
                x_max=w / 2,
                y_min=0.0,
                y_max=0.22,
                z_min=i * CELL_LENGTH,
                z_max=(i + 1) * CELL_LENGTH,
            ),
        )
        for i, w in enumerate(WIDTHS)
    ]
    mesh = mw.Mesh2D(
        x=np.linspace(-1.0, 1.0, 41),
        y=np.linspace(-0.4, 0.62, 35),
    )
    cells = [
        mw.Cell(
            structures=structs,
            mesh=mesh,
            z_min=i * CELL_LENGTH,
            z_max=(i + 1) * CELL_LENGTH,
        )
        for i in range(len(WIDTHS))
    ]
    return cells, mw.Environment(wl=1.55, T=25.0)


@pytest.fixture(scope="module")
def taper() -> tuple[list[mw.Cell], mw.Environment]:
    return _stepped_taper()


@pytest.fixture(scope="module")
def serial_s_matrix(taper: tuple[list[mw.Cell], mw.Environment]) -> tuple:
    """The serial reference EME S-matrix."""
    cells, env = taper
    css = [mw.CrossSection.from_cell(cell=cell, env=env) for cell in cells]
    modes = [mw.compute_modes(cs, num_modes=NUM_MODES) for cs in css]
    return mw.compute_s_matrix(modes, cells=cells)


def test_chunking_triplets_and_end_pair() -> None:
    """The default chunking yields triplets plus a pair for odd interfaces."""
    assert mw.chunk_cell_indices(2) == [(0, 1)]  # single pair
    assert mw.chunk_cell_indices(3) == [(0, 2)]  # single triplet
    assert mw.chunk_cell_indices(4) == [(0, 2), (2, 3)]  # triplet + end pair
    assert mw.chunk_cell_indices(6) == [(0, 2), (2, 4), (4, 5)]
    assert mw.chunk_cell_indices(7) == [(0, 2), (2, 4), (4, 6)]  # all triplets


def test_chunking_pairs() -> None:
    assert mw.chunk_cell_indices(4, max_interfaces_per_job=1) == [
        (0, 1),
        (1, 2),
        (2, 3),
    ]


def test_chunking_covers_all_interfaces() -> None:
    for num_cells in range(2, 12):
        for max_if in (1, 2, 3):
            groups = mw.chunk_cell_indices(num_cells, max_if)
            covered = sorted(k for start, stop in groups for k in range(start, stop))
            assert covered == list(range(num_cells - 1))  # exactly once each


def test_chunking_invalid() -> None:
    with pytest.raises(ValueError, match="At least 2 cells"):
        mw.chunk_cell_indices(1)
    with pytest.raises(ValueError, match="max_interfaces_per_job"):
        mw.chunk_cell_indices(4, max_interfaces_per_job=0)


def test_group_result_contains_no_field_data(
    taper: tuple[list[mw.Cell], mw.Environment],
) -> None:
    """A group job returns only neffs and interface matrices, no mode fields."""
    cells, env = taper
    result = mw.compute_group_result(
        [c.model_dump() for c in cells[:3]], env.model_dump(), 0, NUM_MODES
    )
    assert result.start == 0
    assert len(result.neffs) == 3
    assert len(result.interfaces) == 2
    for k, S in enumerate(result.interfaces):
        n_l, n_r = len(result.neffs[k]), len(result.neffs[k + 1])
        assert S.shape == (n_l + n_r, n_l + n_r)


def test_parallel_matches_serial_with_thread_executor(
    taper: tuple[list[mw.Cell], mw.Environment],
    serial_s_matrix: tuple,
) -> None:
    cells, env = taper
    S_ref, pm_ref = serial_s_matrix
    S_par, pm_par = mw.compute_s_matrix_parallel(
        cells, env, num_modes=NUM_MODES, executor=ThreadPoolExecutor(max_workers=2)
    )
    assert pm_par == pm_ref
    np.testing.assert_allclose(np.asarray(S_par), np.asarray(S_ref), atol=1e-9)


def test_parallel_matches_serial_with_pair_jobs(
    taper: tuple[list[mw.Cell], mw.Environment],
    serial_s_matrix: tuple,
) -> None:
    cells, env = taper
    S_ref, pm_ref = serial_s_matrix
    S_par, pm_par = mw.compute_s_matrix_parallel(
        cells,
        env,
        num_modes=NUM_MODES,
        max_interfaces_per_job=1,
        executor=ThreadPoolExecutor(max_workers=3),
    )
    assert pm_par == pm_ref
    np.testing.assert_allclose(np.asarray(S_par), np.asarray(S_ref), atol=1e-9)


def test_parallel_matches_serial_with_subprocesses(
    taper: tuple[list[mw.Cell], mw.Environment],
    serial_s_matrix: tuple,
) -> None:
    """Default backend: concurrent local subprocesses (spawned)."""
    cells, env = taper
    S_ref, pm_ref = serial_s_matrix
    S_par, pm_par = mw.compute_s_matrix_parallel(
        cells, env, num_modes=NUM_MODES, max_workers=2
    )
    assert pm_par == pm_ref
    np.testing.assert_allclose(np.asarray(S_par), np.asarray(S_ref), atol=1e-9)


def test_parallel_matches_serial_with_submitit(
    taper: tuple[list[mw.Cell], mw.Environment],
    serial_s_matrix: tuple,
    tmp_path,  # noqa: ANN001
) -> None:
    """The slurm job machinery (run locally via submitit's local cluster)."""
    pytest.importorskip("submitit")
    cells, env = taper
    S_ref, pm_ref = serial_s_matrix
    executor = mw.slurm_executor(
        folder=str(tmp_path / "jobs"), cluster="local", timeout_min=10
    )
    S_par, pm_par = mw.compute_s_matrix_parallel(
        cells, env, num_modes=NUM_MODES, executor=executor
    )
    assert pm_par == pm_ref
    np.testing.assert_allclose(np.asarray(S_par), np.asarray(S_ref), atol=1e-9)


def test_async_parallel_matches_serial(
    taper: tuple[list[mw.Cell], mw.Environment],
    serial_s_matrix: tuple,
) -> None:
    import asyncio

    cells, env = taper
    S_ref, pm_ref = serial_s_matrix

    async def main() -> tuple:
        return await mw.acompute_s_matrix_parallel(
            cells,
            env,
            num_modes=NUM_MODES,
            executor=ThreadPoolExecutor(max_workers=2),
        )

    S_par, pm_par = asyncio.run(main())
    assert pm_par == pm_ref
    np.testing.assert_allclose(np.asarray(S_par), np.asarray(S_ref), atol=1e-9)


def test_shared_cell_mode_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="different number of modes"):
        _check_shared_cell_consistency(
            1, np.array([2.5 + 0j, 2.0 + 0j]), np.array([2.5 + 0j]), atol=1e-6
        )


def test_shared_cell_neff_deviation_warns() -> None:
    with pytest.warns(UserWarning, match="effective indices differ"):
        _check_shared_cell_consistency(
            1, np.array([2.5 + 0j]), np.array([2.5 + 1e-3 + 0j]), atol=1e-6
        )


def test_shared_cell_consistent_neffs_pass() -> None:
    _check_shared_cell_consistency(
        1, np.array([2.5 + 0j]), np.array([2.5 + 1e-12 + 0j]), atol=1e-6
    )


def test_slurm_executor_requires_submitit(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "submitit":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="pip install submitit"):
        mw.slurm_executor()
