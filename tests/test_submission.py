"""P0-C: submission writer, summary and baseline50 gate (docs/plan/tasks/P0-C-submission-and-run.md).

Synthetic tests need no data. The `slow` tests need the bundle: the Phase 0 gate (baseline50
byte-identical to the organisers' file) and the PLAN.md 1.7 reference numbers for smooth50
(594 possessions, ~32% D, 133 D segments), cross-checked against possession_metrics.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

import games
from engine import run, submission

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

BASELINE50 = os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")
BASELINE50_MD5 = "23924df8d47236bfacf527d93079fc66"
N_TACTICAL = 199319

_TACTICAL = games.asset_paths()
HAVE_DATA = all(os.path.isfile(_TACTICAL[k]) for k in ("atd_meta", "atd_raw"))
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="match data bundle not present")

TINY_BLOCKS = [(1, 2, 5)]  # one graded block, frames 2..5 inclusive


# -- writer --------------------------------------------------------------------------------------

def test_writer_format(tmp_path):
    path = str(tmp_path / "s.csv")
    labels = np.array(["X", "X", "H", "A", "D", "H", "X"])
    assert submission.write_submission(labels, path, n_frames=7, blocks=TINY_BLOCKS) == path
    with open(path, "rb") as handle:
        content = handle.read()
    assert content == b"frame,label\n0,X\n1,X\n2,H\n3,A\n4,D\n5,H\n6,X\n"


def test_writer_accepts_object_arrays(tmp_path):
    path = str(tmp_path / "s.csv")
    submission.write_submission(np.array(["D", "H"], dtype=object), path, n_frames=2,
                                blocks=[(1, 0, 1)])
    assert open(path).read() == "frame,label\n0,D\n1,H\n"


def test_writer_raises_on_graded_x(tmp_path):
    path = str(tmp_path / "s.csv")
    labels = np.array(["X", "X", "H", "X", "D", "H", "X"])
    with pytest.raises(ValueError, match="graded"):
        submission.write_submission(labels, path, n_frames=7, blocks=TINY_BLOCKS)
    assert not os.path.exists(path)


def test_writer_raises_on_bad_label(tmp_path):
    labels = np.array(["X", "X", "H", "B", "D", "H", "X"])
    with pytest.raises(ValueError, match="outside"):
        submission.write_submission(labels, str(tmp_path / "s.csv"), n_frames=7,
                                    blocks=TINY_BLOCKS)


def test_writer_raises_on_bad_length(tmp_path):
    with pytest.raises(ValueError, match="expected"):
        submission.write_submission(np.array(["D"] * 6), str(tmp_path / "s.csv"), n_frames=7,
                                    blocks=TINY_BLOCKS)


def test_default_length_is_tactical_video():
    assert submission.n_frames_default() == N_TACTICAL


def test_md5_of(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"abc")
    assert submission.md5_of(str(path)) == "900150983cd24fb0d6963f7d28e17f72"


# -- summarize -----------------------------------------------------------------------------------

def _synthetic():
    """One block of 90 frames: H x30, A x20, D x30, H x10, with X outside it."""
    labels = np.array(["X"] * 5 + ["H"] * 30 + ["A"] * 20 + ["D"] * 30 + ["H"] * 10 + ["X"] * 5)
    return labels, [(1, 5, 94)]


def test_summarize_known_counts():
    labels, blocks = _synthetic()
    s = submission.summarize(labels, blocks, fps=10)
    half = s["halves"][1]
    assert half["graded_frames"] == 90
    assert half["possessions"] == 3
    assert half["d_segments"] == 1
    assert half["transitions"] == 3
    assert half["graded_x"] == 0
    assert half["d_share"] == pytest.approx(30 / 90.0)
    assert half["mean_s"] == pytest.approx(2.0)      # (30 + 20 + 10) / 3 frames at 10 fps
    assert half["median_s"] == pytest.approx(2.0)
    assert half["longest_s"] == pytest.approx(3.0)
    assert half["transitions_per_min"] == pytest.approx(3 / (9.0 / 60.0))
    assert s["total"]["possessions"] == 3


def test_summarize_does_not_merge_across_blocks():
    """Same label either side of a gap: two spells, as possession_metrics.segments counts."""
    labels = np.array(["H"] * 10 + ["D"] * 5 + ["H"] * 10)
    s = submission.summarize(labels, [(1, 0, 9), (2, 15, 24)], fps=10)
    assert s["total"]["possessions"] == 2
    assert s["total"]["d_segments"] == 0
    assert s["total"]["transitions"] == 0


def test_summarize_counts_graded_x():
    labels = np.array(["H", "X", "H", "D"])
    s = submission.summarize(labels, [(1, 0, 3)], fps=1)
    assert s["total"]["graded_x"] == 1
    assert s["vetoes"]["graded_x_eq_0"] is False
    assert s["ok"] is False


def test_vetoes():
    # 100 x (H x10, A x10), the first 60 followed by one D frame: 200 spells, 60 D segments,
    # D share 60 / 2060 < 18% -> only the D share veto fails.
    unit = ["H"] * 10 + ["A"] * 10
    seq = []
    for i in range(100):
        seq += unit
        if i < 60:
            seq += ["D"]
    labels = np.array(seq)
    s = submission.summarize(labels, [(1, 0, len(labels) - 1)], fps=25)
    assert s["total"]["possessions"] == 200
    assert s["total"]["d_segments"] == 60
    assert s["vetoes"] == {"possessions_180_450": True, "d_share_18_55": False,
                           "d_segments_ge_55": True, "graded_x_eq_0": True}
    assert s["ok"] is False

    # Pad with D to lift the share into range: everything passes.
    labels = np.concatenate([labels, np.array(["D"] * 1000)])
    s = submission.summarize(labels, [(1, 0, len(labels) - 1)], fps=25)
    assert s["vetoes"]["d_share_18_55"] is True
    assert s["ok"] is True
    assert "PASS" in submission.format_summary(s)


def test_format_summary_has_halves_and_total():
    labels, blocks = _synthetic()
    text = submission.format_summary(submission.summarize(labels, blocks, fps=10))
    assert "half 1" in text and "total" in text and "possessions" in text


# -- baseline reimplementation (synthetic) -------------------------------------------------------

def test_nearest_team_first_slot_wins_ties_and_nobody_without_ball():
    nan = np.nan
    tactical = {
        "px": np.array([[0.5, 0.5, nan], [nan, nan, nan], [0.1, 0.9, 0.2]], dtype=np.float32),
        "py": np.array([[0.4, 0.6, nan], [nan, nan, nan], [0.5, 0.5, 0.5]], dtype=np.float32),
        "bx": np.array([0.5, 0.5, nan], dtype=np.float32),
        "by": np.array([0.5, 0.5, nan], dtype=np.float32),
        "team_of_slot": np.array([1, 0, 0], dtype=np.int8),
    }
    codes = run.nearest_team_codes(tactical, (100.0, 100.0))
    assert codes.tolist() == [1, run.NOBODY, run.NOBODY]  # tie -> slot 0 (away)


def test_smooth_majority_ties_to_home_and_d_when_nobody():
    codes = np.array([0, 1, -1, -1, -1, -1, -1], dtype=np.int8)
    assert run.smooth_majority(codes, 1).tolist() == ["H", "H", "A", "D", "D", "D", "D"]


# -- data-backed (slow) --------------------------------------------------------------------------

@pytest.mark.slow
@needs_data
def test_gate_baseline50_byte_identical(tmp_path):
    """Phase 0 gate: --mode baseline50 reproduces the organisers' file byte-for-byte."""
    out = str(tmp_path / "b50.csv")
    result = subprocess.run([sys.executable, "-m", "engine.run", "--mode", "baseline50",
                             "--out", out], cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True)
    assert result.returncode == 0, result.stdout
    assert submission.md5_of(out) == BASELINE50_MD5
    with open(out, "rb") as mine, open(BASELINE50, "rb") as theirs:
        assert mine.read() == theirs.read()


def test_summarize_reproduces_plan_reference_for_smooth50():
    """PLAN.md 1.7: smooth50 has 594 possessions, 32% D, 133 D segments; agrees with pm."""
    codes = pm.label_codes(pm.read_label_csv(BASELINE50))
    labels = np.array(pm.LABELS)[codes]
    s = submission.summarize(labels)
    assert s["total"]["possessions"] == 594
    assert s["total"]["d_segments"] == 133
    assert round(s["total"]["d_share"], 2) == 0.32
    assert s["vetoes"]["possessions_180_450"] is False  # PLAN 5.3: expected for the baseline

    runs = [(first, last + 1) for _, first, last in submission.data.graded_blocks()]
    assert s["total"]["possessions"] == len(pm.possession_spells(codes, runs))
    assert s["total"]["d_segments"] == sum(1 for _, _, k in pm.segments(codes, runs)
                                           if k == pm.LABELS.index("D"))
