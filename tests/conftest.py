"""Shared fixtures.

The guiding rule: a test never writes into the real `data/` or `archive/`
directories, and never downloads anything. Anything that needs a layout gets a
`Settings` pointed at `tmp_path`.
"""

from __future__ import annotations

import pytest

from visualtaste.config import Settings


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
