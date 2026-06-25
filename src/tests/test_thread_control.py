"""Tests for solver thread-pinning (meow.settings.limit_threads/solver_threads)."""

from __future__ import annotations

import numpy as np
import pytest

import meow as mw
from meow.settings import _THREAD_ENV_VARS


def test_solver_threads_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """solver_threads honours MEOW_SOLVER_THREADS and the default otherwise."""
    monkeypatch.delenv("MEOW_SOLVER_THREADS", raising=False)
    assert mw.solver_threads() is None
    assert mw.solver_threads(default=4) == 4
    monkeypatch.setenv("MEOW_SOLVER_THREADS", "3")
    assert mw.solver_threads() == 3
    assert mw.solver_threads(default=8) == 3


def test_limit_threads_none_is_noop() -> None:
    """A None / non-positive limit runs the block without touching anything."""
    with mw.limit_threads(None):
        assert float(np.sum(np.ones(8))) == 8.0
    with mw.limit_threads(0):
        assert float(np.sum(np.ones(8))) == 8.0


def test_limit_threads_actually_limits_blas() -> None:
    """Inside the block threadpoolctl reports the requested cap on every pool."""
    threadpoolctl = pytest.importorskip("threadpoolctl")
    with mw.limit_threads(1):
        # force a BLAS call so the native pools exist, then inspect them
        _ = np.linalg.eigh(np.eye(16) + 0.1)
        infos = threadpoolctl.threadpool_info()
        assert infos  # at least one native pool present
        assert all(info["num_threads"] == 1 for info in infos)


def test_limit_threads_env_fallback_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    """The env-var fallback sets the *_NUM_THREADS vars and restores them after."""
    # hide threadpoolctl so limit_threads takes the env-var path
    monkeypatch.setitem(__import__("sys").modules, "threadpoolctl", None)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "7")
    import os

    with mw.limit_threads(2):
        for key in _THREAD_ENV_VARS:
            assert os.environ[key] == "2"
    # previously-unset stays unset; previously-set is restored to its old value
    assert "OMP_NUM_THREADS" not in os.environ
    assert os.environ["MKL_NUM_THREADS"] == "7"
