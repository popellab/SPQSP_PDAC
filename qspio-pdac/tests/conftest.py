"""Shared pytest fixtures for qspio-pdac tests."""

import pytest
from pathlib import Path
import tempfile


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
