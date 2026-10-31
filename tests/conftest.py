"""Shared fixtures.

The guiding rule: a test never writes into the real `data/` or `archive/`
directories, and never downloads anything. Anything that needs a layout gets a
`Settings` pointed at `tmp_path`.
"""

from __future__ import annotations

import pytest

from visualtaste.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _never_touch_the_real_corpus(monkeypatch, tmp_path):
    """Make `get_settings()` point at `tmp_path` for the duration of every test.

    This exists because of a real incident: the API's `lifespan` built its own
    settings from `get_settings()` and overwrote the state a test had injected,
    so the test suite read the live 80,000-image corpus and wrote a 50 MB
    database snapshot into the real archive on shutdown. Everything still
    passed, which is what made it dangerous.

    The bug is fixed, but a guard is cheaper than remembering. Any code that
    reaches for the process-wide settings during a test now gets a temporary
    directory instead of the real one.
    """
    get_settings.cache_clear()
    sandbox = Settings(
        project_root=tmp_path / "sandbox",
        data_root=tmp_path / "sandbox" / "data",
        archive_root=tmp_path / "sandbox" / "archive",
        min_free_gb=0,
    )
    monkeypatch.setattr("visualtaste.config.get_settings", lambda: sandbox)
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path):
    """A `Settings` whose every root is inside `tmp_path`.

    `archive_root` is set explicitly so the detection ladder never runs during a
    test -- otherwise results would depend on which drives happen to be plugged
    into the machine running the suite.
    """
    return Settings(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        archive_root=tmp_path / "archive",
    )
