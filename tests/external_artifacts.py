"""Explicit opt-in checks for immutable experiment data excluded from Git."""
from pathlib import Path
import pytest


def requires_artifacts(*paths):
    missing = [path for path in paths if not Path(path).exists()]
    return pytest.mark.skipif(
        bool(missing),
        reason='External experiment artifacts are not bundled: ' + ', '.join(missing))
