"""Shared test fixtures.

``get_config()`` memoizes a single ``Config`` built from the environment, and
several runtime paths mutate that singleton in place (e.g. the benchmark runner
redirects ``environment.workspace_dir`` to its run directory). Resetting the
singleton around every test keeps such mutations from leaking across tests, so
test order never changes behavior.
"""

from __future__ import annotations

import pytest

from src.config import reset_config


@pytest.fixture(autouse=True)
def _isolated_global_config():
    """Reset the process-global configuration before and after each test."""
    reset_config()
    yield
    reset_config()
