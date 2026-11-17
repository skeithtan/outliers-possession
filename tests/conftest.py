"""pytest setup: repo root on sys.path (so `engine`, `tools`, `games` import) and the `slow` mark."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_configure(config):
    """Register markers so `-m "not slow"` works without warnings."""
    config.addinivalue_line(
        "markers", "slow: needs the match data bundle and parses it (seconds, not ms)")
