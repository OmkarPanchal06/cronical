"""Shared pytest fixtures for the Cronical test suite.

Two guarantees are enforced here for every test in the repository:

* No test writes to the real project tree. Anything that touches the filesystem
  is redirected to ``tmp_path`` via :func:`Settings.for_root`.
* The global :func:`get_settings` cache and the ``cronical`` logging handlers are
  reset before and after each test, so ordering never changes the result.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from cronical.config import Settings, reload_settings
from cronical.utils.logging import reset_logging


@pytest.fixture(autouse=True)
def _isolate_global_state() -> Iterator[None]:
    """Reset the settings cache and logging handlers around every test."""
    reload_settings()
    reset_logging()
    yield
    reload_settings()
    reset_logging()


@pytest.fixture
def isolated_project_root(tmp_path: Path) -> Path:
    """Return a throwaway project root that mirrors the real directory layout."""
    root = tmp_path / "project"
    root.mkdir()
    return root


@pytest.fixture
def settings(isolated_project_root: Path) -> Settings:
    """Return :class:`Settings` rooted at :func:`isolated_project_root`."""
    return Settings.for_root(isolated_project_root)
