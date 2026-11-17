"""tools/analysis/referee.py: the label-free referees of RESEARCH-5 Section 4.

Every test builds its arrays by hand. No test needs the match data, a label or the video.
"""

import numpy as np
import pytest

from tools.analysis import referee

H, A, D, X = 0, 1, 2, 3


def _codes(spec):
    """[('H', 40), ('D', 20), ...] -> int8 codes."""
    letters = "HADX"
    return np.concatenate([np.full(n, letters.index(c), dtype=np.int8) for c, n in spec])


def test_runs_of_gives_half_open_runs():
    mask = np.array([0, 1, 1, 0, 0, 1], dtype=bool)
    assert referee.runs_of(mask) == [(1, 3), (5, 6)]
    assert referee.runs_of(np.zeros(4, dtype=bool)) == []


def test_mixture_reads_a_cell_between_the_two_agreement_cells():
    assert referee.mixture(0.8, 0.8, 0.1) == pytest.approx(1.0)
    assert referee.mixture(0.1, 0.8, 0.1) == pytest.approx(0.0)
    assert referee.mixture(0.45, 0.8, 0.1) == pytest.approx(0.5)
    # Outside the two agreement shares the estimate is clipped.
    assert referee.mixture(0.95, 0.8, 0.1) == 1.0
    assert referee.mixture(0.0, 0.8, 0.1) == 0.0
    assert referee.mixture(None, 0.8, 0.1) is None
    assert referee.mixture(0.5, 0.3, 0.3) is None


def test_camera_away_needs_a_mapped_frame():
    valid = np.array([1, 0, 1, 1, 0], dtype=bool)
    replay = np.array([0, 0, 1, 0, 0], dtype=bool)
    b_frame = np.array([10, 11, 12, 13, -1])
    away, covered = referee.camera_away(valid, replay, b_frame)
    assert covered.tolist() == [True, True, True, True, False]
    # Invalid homography, replay flag. The last frame has no broadcast frame, so it is not away.
    assert away.tolist() == [False, True, True, False, False]


def test_camera_referee_sides_with_the_system_that_the_camera_supports():
    # X and Y agree on a stoppage of 200 frames and on 400 alive frames. X alone calls a second
    # stoppage of 200 frames. The camera is away in the common stoppage and in the X stoppage.
    x = _codes([("H", 200), ("D", 200), ("A", 200), ("D", 200), ("H", 100)])
    y = _codes([("H", 200), ("D", 200), ("A", 200), ("A", 200), ("H", 100)])
    n = len(x)
    away = np.zeros(n, dtype=bool)
    away[200:400] = True
    away[600:800] = True
    covered = np.ones(n, dtype=bool)
    out = referee.camera_referee(x, y, away, covered, [(0, n)], fps=25)
    assert out["cells"]["both_dead"] == {"frames": 200, "away_share": 1.0}
    assert out["cells"]["both_alive"] == {"frames": 500, "away_share": 0.0}
    cell = out["cells"]["x_dead_y_alive"]
    assert cell["frames"] == 200 and cell["dead_share_estimate"] == pytest.approx(1.0)
    assert out["cells"]["x_alive_y_dead"]["frames"] == 0
    # The X stoppage touches no common stoppage, so it is one isolated run of 8 s.
    assert out["runs"]["x_dead_y_alive"] == [{
        "kind": "isolated", "length": "from_5s", "runs": 1, "frames": 200, "away_share": 1.0,
        "dead_share_estimate": pytest.approx(1.0)}]


def test_camera_referee_sides_against_a_stoppage_that_the_camera_does_not_show():
    x = _codes([("H", 200), ("D", 200), ("A", 200), ("D", 200), ("H", 100)])
    y = _codes([("H", 200), ("D", 200), ("A", 200), ("A", 200), ("H", 100)])
    n = len(x)
    away = np.zeros(n, dtype=bool)
    away[200:400] = True          # the camera leaves for the common stoppage only
    out = referee.camera_referee(x, y, away, np.ones(n, dtype=bool), [(0, n)], fps=25)
    assert out["cells"]["x_dead_y_alive"]["dead_share_estimate"] == pytest.approx(0.0)


def test_camera_referee_marks_a_run_next_to_a_common_stoppage_as_edge():
    # X ends the stoppage 50 frames later than Y.
    x = _codes([("H", 100), ("D", 150), ("A", 100)])
    y = _codes([("H", 100), ("D", 100), ("A", 150)])
    n = len(x)
    out = referee.camera_referee(x, y, np.zeros(n, dtype=bool), np.ones(n, dtype=bool),
                                 [(0, n)], fps=25)
    rows = out["runs"]["x_dead_y_alive"]
    assert len(rows) == 1
    assert rows[0]["kind"] == "edge" and rows[0]["length"] == "2_to_5s" and rows[0]["frames"] == 50


def test_camera_referee_counts_only_graded_frames_that_the_broadcast_covers():
    x = _codes([("X", 50), ("H", 100), ("D", 100), ("X", 50)])
    y = _codes([("X", 50), ("H", 100), ("H", 100), ("X", 50)])
    n = len(x)
    covered = np.ones(n, dtype=bool)
    covered[150:200] = False      # half of the disagreement has no broadcast frame
    away = np.ones(n, dtype=bool)
    out = referee.camera_referee(x, y, away, covered, [(50, 250)], fps=25)
    assert out["graded_frames"] == 200
    assert out["covered_frames"] == 150
    assert out["cells"]["x_dead_y_alive"]["frames"] == 50
    assert out["cells"]["both_alive"]["frames"] == 100


def test_camera_referee_fills_an_abstention_as_the_scorer_does():
    x = _codes([("H", 50), ("X", 50), ("D", 100)])
    y = _codes([("H", 100), ("D", 100)])
    n = len(x)
    out = referee.camera_referee(x, y, np.zeros(n, dtype=bool), np.ones(n, dtype=bool),
                                 [(0, n)], fps=25)
    # The X frames of X repeat H, so the two systems agree on every frame.
    assert out["cells"]["both_alive"]["frames"] == 100
    assert out["cells"]["both_dead"]["frames"] == 100


def test_edge_distance_is_zero_on_the_first_frame_of_a_new_value():
    values = np.array([5, 5, 5, 7, 7, 7, 7])
    assert referee.edge_distance(values).tolist() == [3, 2, 1, 0, 1, 2, 3]
    assert referee.edge_distance(np.array([1, 1, 1])).tolist() == [3, 3, 3]


def test_tag_referee_estimates_the_share_of_frames_where_x_is_right():
    # 1,000 agreement frames: the tag names the common team on 800 (accuracy 0.8).
    # 200 disagreement frames (X says H, Y says A): the tag names H on 160. With accuracy 0.8
    # that is the share of a referee that sides with X on every frame.
    x = _codes([("H", 1000), ("H", 200)])
    y = _codes([("H", 1000), ("A", 200)])
    paint = np.concatenate([np.full(800, H), np.full(200, A), np.full(160, H), np.full(40, A)])
    out = referee.tag_referee(x, y, paint, [(0, 1200)], edge_frames=0)
    assert out["agree_frames"] == 1000 and out["tag_accuracy"] == pytest.approx(0.8)
    assert out["differ_frames"] == 200 and out["tag_share_for_x"] == pytest.approx(0.8)
    assert out["x_right_estimate"] == pytest.approx(1.0)


def test_tag_referee_gives_one_half_when_the_tags_cannot_tell():
    x = _codes([("H", 1000), ("H", 200)])
    y = _codes([("H", 1000), ("A", 200)])
    paint = np.concatenate([np.full(800, H), np.full(200, A), np.full(100, H), np.full(100, A)])
    out = referee.tag_referee(x, y, paint, [(0, 1200)], edge_frames=0)
    assert out["x_right_estimate"] == pytest.approx(0.5)


def test_tag_referee_leaves_out_dead_frames_untagged_frames_and_tag_edges():
    x = _codes([("H", 100), ("D", 100), ("H", 100)])
    y = _codes([("H", 100), ("H", 100), ("A", 100)])
    paint = np.concatenate([np.full(100, H), np.full(100, H), np.full(50, -1), np.full(50, A)])
    out = referee.tag_referee(x, y, paint, [(0, 300)], edge_frames=0)
    # Frames 100..199: X says D. Frames 200..249: no tag. Left: 100 agreement, 50 disagreement.
    assert out["agree_frames"] == 100 and out["differ_frames"] == 50
    assert out["tag_share_for_x"] == pytest.approx(0.0)
    # With an edge of 10 frames the frames next to each change of the tag team drop out.
    edged = referee.tag_referee(x, y, paint, [(0, 300)], edge_frames=10)
    assert edged["agree_frames"] == 100
    assert edged["differ_frames"] == 40


def test_tag_referee_returns_none_without_frames():
    x = _codes([("D", 100)])
    out = referee.tag_referee(x, x, np.full(100, H), [(0, 100)], edge_frames=0)
    assert out["tagged_alive_frames"] == 0
    assert out["tag_accuracy"] is None and out["x_right_estimate"] is None
