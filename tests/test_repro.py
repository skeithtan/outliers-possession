"""PLAN.md 3.8: `python -m engine.run` is byte-for-byte reproducible and always validates.

The test runs the CLI twice in fresh subprocesses and compares the md5s. The default mode is the
real engine (engine.possession.run, card P1-C), so the test needs the match data bundle. It is
skipped when the bundle is absent. Phase 5 extends this test to a fresh-clone run.
"""

import os
import subprocess
import sys

import pytest

import games
from engine import submission

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ASSETS = games.asset_paths()
HAVE_DATA = all(os.path.isfile(_ASSETS[k])
                for k in ("atd_meta", "atd_raw", "homography", "smart_tagging"))


def _run_engine(out):
    result = subprocess.run([sys.executable, "-m", "engine.run", "--mode", "engine",
                             "--out", out], cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True)
    assert result.returncode == 0, result.stdout
    assert "OK    " in result.stdout  # the validator's success line
    return result.stdout


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_DATA, reason="match data bundle not present: the engine needs it")
def test_engine_run_is_reproducible_and_valid(tmp_path):
    first, second = str(tmp_path / "a.csv"), str(tmp_path / "b.csv")
    out_first = _run_engine(first)
    _run_engine(second)
    md5 = submission.md5_of(first)
    assert md5 == submission.md5_of(second)
    assert "md5   {}".format(md5) in out_first
    assert "verdict: PASS" in out_first  # the PLAN 5.3 vetoes hold
    submission.validate(first)  # raises on failure
