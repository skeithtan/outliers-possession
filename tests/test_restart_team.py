"""tools/analysis/restart_team.py on synthetic sequences (P9-C measurements). No match data needed."""

import numpy as np
import pandas as pd

from engine.config import Config
from tools.analysis import restart_team as rt

PITCH = (105.0, 68.0)


def _ball(x, y, visible=None, fps=25.0):
    """A ball dict as engine.tracks.prepare() gives it, from metre positions."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if visible is None:
        visible = np.isfinite(x) & np.isfinite(y)
    visible = np.asarray(visible, dtype=bool)
    x = np.where(visible, x, np.nan)
    y = np.where(visible, y, np.nan)
    vx = np.gradient(np.nan_to_num(x)) * fps
    vy = np.gradient(np.nan_to_num(y)) * fps
    run_id = np.full(x.size, -1, dtype=np.int64)
    run = -1
    for i in range(x.size):
        if visible[i]:
            if i == 0 or not visible[i - 1]:
                run += 1
            run_id[i] = run
    vx = np.where(visible, vx, np.nan)
    vy = np.where(visible, vy, np.nan)
    return {"x": x, "y": y, "visible": visible, "vx": vx, "vy": vy,
            "speed": np.hypot(vx, vy), "run_id": run_id, "fps": fps}


def test_label_runs_and_restarts_keep_only_d_runs_ended_by_a_team():
    labels = np.array(list("XXDDHHHDDDAAADDXX"))
    assert rt.label_runs(labels)[:3] == [(0, 2, "X"), (2, 4, "D"), (4, 7, "H")]
    # D 2-4 ends with H, D 7-10 ends with A, D 13-15 ends with X: no restart.
    assert rt.restarts_of(labels) == [(2, 4, "H"), (7, 10, "A")]
    assert rt.restarts_of(np.array(list("HHDD"))) == []


def test_match_exits_is_one_to_one_and_greedy_by_difference():
    # 1000 and 1010 both want 1008; 1010 is nearer (2), so 1000 takes 990 (10).
    assert rt.match_exits([1000, 1010, 5000], [990, 1008, 5200], tol=125) == [0, 1, -1]
    # A tie goes to the earlier engine exit.
    assert rt.match_exits([1000], [990, 1010], tol=125) == [0]
    assert rt.match_exits([1000], [], tol=125) == [-1]


def test_first_and_last_live_label_stop_at_x():
    labels = np.array(list("HXDDAA"))
    assert rt.first_live_label(labels, 2) == "A"
    assert rt.last_live_label(labels, 4) is None      # X at 1 stops the walk
    assert rt.last_live_label(np.array(list("AHDD")), 3) == "H"
    assert rt.first_live_label(np.array(list("DDX")), 0) is None


def test_line_of_and_heading_line():
    assert rt.line_of(50.0, 70.0, PITCH) == "touch"
    assert rt.line_of(-1.0, 30.0, PITCH) == "goal0"
    assert rt.line_of(106.0, 34.0, PITCH) == "goalL"
    assert rt.line_of(-0.5, -2.0, PITCH) == "touch"   # the larger excess decides at a corner
    assert rt.line_of(50.0, 30.0, PITCH) is None
    assert rt.heading_line(100.0, 34.0, 5.0, 0.0, PITCH) == ("goalL", 5.0)
    assert rt.heading_line(50.0, 60.0, 0.0, 4.0, PITCH) == ("touch", 8.0)
    line, dist = rt.heading_line(10.0, 60.0, -3.0, 3.0, PITCH)
    assert line == "touch" and abs(dist - float(np.hypot(8.0, 8.0))) < 1e-9
    assert rt.heading_line(50.0, 30.0, 0.0, 0.0, PITCH)[0] is None


def test_out_geometry_seen_heading_and_inside():
    cfg = Config.default()
    n = 400
    # Seen: the ball rolls over the touchline at frame 201, the onset is at 200.
    y = 60.0 + 0.04 * np.arange(n)
    x = np.full(n, 50.0)
    line, cross, how = rt.out_geometry(_ball(x, y), 200, PITCH, cfg)
    assert (line, how) == ("touch", "seen") and cross == int(np.argmax(y > 68.0))
    # Heading: last seen at 150 at 2.5 m/s toward the goal line 15 m away, then missing.
    x = 90.0 + 0.1 * (np.arange(n) - 150)
    vis = np.arange(n) <= 150
    ball = _ball(x, np.full(n, 34.0), vis)
    ball["vx"][150], ball["vy"][150], ball["speed"][150] = 2.5, 0.0, 2.5
    line, cross, how = rt.out_geometry(ball, 200, PITCH, cfg)
    assert (line, cross, how) == ("goalL", 150, "heading")
    # Inside: the ball rests in midfield.
    line, cross, how = rt.out_geometry(_ball(np.full(n, 50.0), np.full(n, 30.0)), 200, PITCH,
                                       cfg)
    assert (line, cross, how) == (None, 200, "inside")


def test_contact_rows_and_last_contact():
    events = pd.DataFrame({
        "frame": [10, 20, 30, 40, 50],
        "kind": ["release", "contested", "control", "touch", "control"],
        "team": ["H", None, "A", "H", "A"],
        "notes": ["", "", "receive", "redirect", "keeper_hold;end=60"]})
    frames, teams, kinds = rt.contact_rows(events)
    assert frames.tolist() == [10, 30, 40, 50]
    assert kinds.tolist() == ["release", "receive", "touch", "keeper_hold"]
    assert rt.last_contact(frames, teams, 45, 100, kinds) == ("H", 40, "touch")
    assert rt.last_contact(frames, teams, 40, 100, kinds) == ("A", 30, "receive")  # strictly before
    assert rt.last_contact(frames, teams, 200, 100, kinds) == (None, None, None)   # too old
    assert rt.last_contact(frames, teams, 5, 100) == (None, None, None)


def test_combine_order_then_nearest_then_owner():
    none = dict((r, None) for r in rt.RULES)
    assert rt.combine(dict(none, b="A", a="H", f="A")) == "H"      # a comes before b
    assert rt.combine(dict(none, d="A", f="H", e="H")) == "A"
    assert rt.combine(dict(none, f="H", e="A")) == "H"
    assert rt.combine(dict(none, e="A")) == "A"
    assert rt.combine(none) is None


def test_inward_crossings_mark_a_rest_beyond_the_touchline():
    n = 300
    x = np.full(n, 40.0)
    y = np.full(n, 30.0)
    # Thrower: the ball rests 0.7 m beyond the touchline 100..199, then is thrown in at 200.
    y[100:200] = 68.7
    y[200:230] = 68.7 - 0.2 * np.arange(1, 31)
    vis = np.ones(n, dtype=bool)
    vis[90:100] = False
    rest = np.zeros(n, dtype=bool)
    rest[100:190] = True
    ball = _ball(x, y, vis)
    crossings = rt.inward_crossings(ball, PITCH, rest)
    assert crossings == [(int(np.argmax((np.arange(n) >= 200) & (y <= 68.0))), True)]
    # The same throw without a rest beyond the line is not rested.
    assert rt.inward_crossings(ball, PITCH, np.zeros(n, dtype=bool))[0][1] is False
    # A crossing over a gap longer than CROSS_GAP frames does not count.
    vis2 = vis.copy()
    vis2[190:200 + rt.CROSS_GAP + 10] = False
    assert rt.inward_crossings(_ball(x, y, vis2), PITCH, rest) == []


def test_kind_of_exit_nearest_within_tolerance():
    kinds = {100: ("throw_in", "H"), 300: ("corner", "A")}
    assert rt.kind_of_exit(kinds, 110, 50) == ("throw_in", "H", -10)
    assert rt.kind_of_exit(kinds, 200, 50) == ("none", None, None)
    assert rt.kind_of_exit(kinds, 300, 0) == ("corner", "A", 0)
    assert rt.kind_of_exit({}, 300, 0) == ("none", None, None)


def test_score_rule_counts_applies_and_right():
    records = [
        {"status": "matched", "team": "H", "eng_rules": {"a": "H"}},
        {"status": "matched", "team": "A", "eng_rules": {"a": "H"}},
        {"status": "matched", "team": "A", "eng_rules": {"a": None}},
        {"status": "no_stoppage", "team": "A"}]
    assert rt.score_rule(records, "eng_rules", "a", True) == (2, 1)


def test_buckets_cut_signed_offsets():
    text = rt._buckets([-60, -20, 0, 9, 10, 49, 50, 200])
    assert text == "< -50: 1, -50..-10: 1, -10..+10: 2, +10..+50: 2, >= +50: 2"
