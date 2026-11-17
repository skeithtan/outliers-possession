"""tools/analysis/foul_onset.py on synthetic sequences (RESEARCH-3 R3-C). No match data."""

import numpy as np

from engine import data, deadball, tracks
from engine.config import Config
from tools.analysis import foul_onset as fo

FPS = 25.0
PITCH = (105.0, 68.0)
CFG = Config.default()


def _feats(n, **values):
    """A feats dict with the keys deadball reads (open play by default)."""
    base = {"dH": 10.0, "dA": 10.0, "meanv_all": 2.5, "meanv_near": 2.5, "sep": 8.0}
    base.update(values)
    out = {k: (np.full(n, float(v)) if np.ndim(v) == 0 else np.asarray(v, dtype=np.float64))
           for k, v in base.items()}
    out["dmin"] = np.fmin(out["dH"], out["dA"])
    out["d_second"] = np.fmax(out["dH"], out["dA"])
    team = np.where(out["dH"] <= out["dA"], data.HOME, data.AWAY)
    out["nearest_team"] = np.where(np.isfinite(out["dmin"]), team, -1).astype(np.int8)
    return out


def _ball_from_speed(speed, start, heading=(1.0, 0.0)):
    speed = np.asarray(speed, dtype=np.float64)
    step = np.concatenate(([0.0], np.cumsum(speed[:-1]) / FPS))
    x, y = start[0] + heading[0] * step, start[1] + heading[1] * step
    return tracks.clean_ball(x / PITCH[0], y / PITCH[1], PITCH, CFG, fps=FPS)


def test_onsets_and_exits_of_a_label_vector():
    labels = np.array(list("XDDHHHDDDAAHDX"))
    assert fo.onsets_of(labels).tolist() == [6, 12]   # the D after X is no onset
    assert fo.exit_of(labels, 6) == 9
    assert fo.exit_of(labels, 12) == 13


def test_onset_status_classes_and_tie():
    eng = [100, 200, 260, 500]
    assert fo.onset_status(105, eng, "H") == ("matched", -5)
    assert fo.onset_status(160, eng, "H") == ("late", 40)      # 200 nearer than 100
    assert fo.onset_status(300, eng, "H") == ("early", -40)
    assert fo.onset_status(230, eng, "H") == ("early", -30)    # tie 200 / 260: the earlier
    assert fo.onset_status(800, eng, "D") == ("inside", None)
    assert fo.onset_status(800, eng, "A") == ("missed", None)


def test_signature_fires_on_vstop_or_on_a_fall():
    n = 400
    meanv = np.full(n, 3.0)
    meanv[201:] = 2.0                       # one third lower after c = 200
    sig = fo.signature(meanv, 200, FPS)
    assert np.isclose(sig["pre"], 3.0) and np.isclose(sig["min2"], 2.0)
    assert np.isclose(sig["fall"], 1.0 / 3.0)
    assert not fo.fires(sig, CFG.V_STOP)
    assert fo.fires(sig, CFG.V_STOP, 0.3)
    assert not fo.fires(sig, CFG.V_STOP, 0.4)
    meanv[230] = 1.2                        # one slow frame within 2 s
    assert fo.fires(fo.signature(meanv, 200, FPS), CFG.V_STOP)
    meanv[230] = 2.0
    meanv[260] = 1.2                        # beyond 2 s (frame 250): no V_STOP firing
    assert not fo.fires(fo.signature(meanv, 200, FPS), CFG.V_STOP)


def test_signature_with_nan_never_fires():
    sig = fo.signature(np.full(300, np.nan), 100, FPS)
    assert sig["pre"] != sig["pre"] and not fo.fires(sig, CFG.V_STOP, 0.2)


def test_pair_anchors_need_both_teams_within_the_width():
    frames = [10, 15, 40, 60, 75, 76]
    teams = ["H", "A", "A", "H", "H", None]
    assert fo.pair_anchors(frames, teams, 12) == [15]
    assert fo.pair_anchors(frames, teams, 20) == [15, 60]


def test_metrica_kind_priority_and_detail():
    foul = [{"type": "CHALLENGE", "subtypes": ("GROUND", "FAULT", "WON")},
            {"type": "FAULT RECEIVED", "subtypes": ()}]
    assert fo.metrica_kind(foul) == ("foul", "CHALLENGE GROUND")
    assert fo.metrica_kind(foul[1:]) == ("foul", "FAULT RECEIVED only")
    assert fo.metrica_kind([{"type": "BALL LOST", "subtypes": ("OFFSIDE",)}])[0] == "offside"
    assert fo.metrica_kind([{"type": "SHOT", "subtypes": ("ON TARGET", "GOAL")}])[0] == "goal"
    assert fo.metrica_kind([{"type": "SHOT", "subtypes": ("OFF TARGET", "OUT")}])[0] == "out"
    assert fo.metrica_kind([{"type": "BALL OUT", "subtypes": ()}]) == ("out", "BALL OUT")
    assert fo.metrica_kind([]) == ("no_stop", "-")


def test_dfl_kind_rules():
    assert fo.dfl_kind(1000, 1400, "throw_in", [990], [])[0] == "out"
    assert fo.dfl_kind(1000, 1400, "kick_off", [], [])[0] == "goal"
    assert fo.dfl_kind(1000, 1400, "free_kick", [995], []) == ("foul", "free_kick", -5)
    assert fo.dfl_kind(1000, 1400, "free_kick", [995], [1003]) == ("offside", "free_kick", 3)
    assert fo.dfl_kind(1000, 1400, "free_kick", [700], [])[0] == "free_kick_no_event"
    assert fo.dfl_kind(1000, 1400, "none", [1100], []) == ("foul", "none", 100)
    assert fo.dfl_kind(1000, 1400, "none", [], [])[0] == "no_event"
    assert fo.dfl_kind(1000, 1400, "penalty", [], []) == ("foul", "penalty", None)


def test_primary_cause_order():
    base = {"entry": False, "hint": False, "out": "none", "pursuit": "none", "stop_4s": False,
            "slow_4s": False, "vanished": False}
    assert fo.primary_cause(base) == "no_stop_4s"
    assert fo.primary_cause(dict(base, stop_4s=True)) == "stop_4s_touched"
    assert fo.primary_cause(dict(base, vanished=True)) == "gap_no_stop"
    assert fo.primary_cause(dict(base, pursuit="chased", vanished=True)) == "pursuit_chased"
    assert fo.primary_cause(dict(base, pursuit="chased", hint=True)) == "collective_hint"
    assert fo.primary_cause(dict(base, pursuit="unconfirmed", hint=True)) == \
        "pursuit_unconfirmed"
    assert fo.primary_cause(dict(base, out="unconfirmed", pursuit="unconfirmed")) == \
        "out_unconfirmed"
    assert fo.primary_cause(dict(base, entry=True, out="unconfirmed")) == "entry_no_edge"
    assert fo.furthest(["no_drop", "chased"], fo.PURSUIT_STAGES) == "chased"
    assert fo.furthest([], fo.PURSUIT_STAGES) == "none"


def test_truth_and_score_signature():
    ref_onsets = [300, 1000]
    assert fo.truth_of(290, ref_onsets, "H") == ("true", 300)     # onset 10 frames later
    assert fo.truth_of(320, ref_onsets, "D") == ("true", 300)     # onset 20 frames before
    assert fo.truth_of(340, ref_onsets, "D") == ("dead", None)
    assert fo.truth_of(600, ref_onsets, "A") == ("false", None)
    slow = {"pre": 3.0, "min2": 1.0, "fall": 0.67}
    calm = {"pre": 3.0, "min2": 2.9, "fall": 0.03}
    anchors = [
        {"frame": 290, "sig": slow, "truth": "true", "ref_onset": 300, "ref_kind": "foul",
         "confirmed": True, "engine": "alive", "v_stop": CFG.V_STOP},
        {"frame": 600, "sig": slow, "truth": "false", "ref_onset": None, "ref_kind": None,
         "confirmed": False, "engine": "alive", "v_stop": CFG.V_STOP},
        {"frame": 950, "sig": calm, "truth": "true", "ref_onset": 1000, "ref_kind": "out",
         "confirmed": False, "engine": "late_entry", "v_stop": CFG.V_STOP},
    ]
    onsets = [{"frame": 300, "kind": "foul"}, {"frame": 1000, "kind": "out"}]
    r = fo.score_signature(anchors, onsets, None)
    assert (r["fire"], r["true"], r["false"], r["recall_n"], r["recall_d"]) == (2, 1, 1, 1, 2)
    assert r["offsets"] == [10] and r["offsets_foul"] == [10]
    r = fo.score_signature(anchors, onsets, None, confirmed_only=True, kinds=("foul",))
    assert (r["fire"], r["precision"], r["recall_n"], r["recall_d"]) == (1, 1.0, 1, 1)
    r = fo.score_signature(anchors, onsets, 0.01)                 # the calm one fires too
    assert (r["fire"], r["true"], r["recall_n"]) == (3, 2, 2)


def test_pursuit_diagnose_agrees_with_the_engine_and_names_the_failing_test():
    n = 220
    t = np.arange(n)
    speed = np.where(t < 40, 1.0, np.minimum(15.0, 1.0 + 3.5 * (t - 39)))
    speed = np.where(t > 44, np.maximum(15.0 - 0.15 * (t - 44), 0.0), speed)
    ball = _ball_from_speed(speed, (20.0, 34.0))
    d_kick = np.where(t < 40, 0.5, 0.5 + 0.4 * (t - 40))
    near = np.where(t < 60, 3.5, np.maximum(3.5 - 2.5 * (t - 60) / 40.0, 1.0))
    feats = _feats(n, dA=d_kick, dH=np.where(t < 40, 4.0, 4.0 + 0.3 * (t - 40)),
                   meanv_near=near, meanv_all=np.where(t >= 70, 1.0, 3.0), sep=14.0)
    horizon = deadball._frames(CFG.ONSET_CONFIRM_S, FPS)
    starts, stops = deadball._stop_runs(feats, CFG, FPS)
    rows = fo.pursuit_diagnose(ball, feats, CFG, starts, stops, horizon)
    table = deadball.dead_onsets(ball, feats, CFG, pitch=PITCH)
    engine = sorted(table.loc[table["mechanism"] == "pursuit_stop", "frame"].tolist())
    assert sorted(o for _, o, s in rows if s == "ok") == engine == [60]
    # The kicking team reaches the ball after the onset: "chased".
    feats["dA"] = np.where((t >= 85) & (t < 95), 0.3, feats["dA"])
    rows = fo.pursuit_diagnose(ball, feats, CFG, starts, stops, horizon)
    assert [s for _, _, s in rows] == ["chased"]
    # Players keep running: no collective stop, "unconfirmed".
    feats = _feats(n, dA=d_kick, dH=np.where(t < 40, 4.0, 4.0 + 0.3 * (t - 40)),
                   meanv_near=near, meanv_all=3.0, sep=14.0)
    starts, stops = deadball._stop_runs(feats, CFG, FPS)
    rows = fo.pursuit_diagnose(ball, feats, CFG, starts, stops, horizon)
    assert [s for _, _, s in rows] == ["unconfirmed"]


def test_out_diagnose_names_an_unconfirmed_missing_ball_out():
    n, last = 200, 49
    y = 65.0 - 8.0 * (last - np.arange(n)) / FPS
    x = np.full(n, 50.0)
    x[last + 1:] = np.nan
    y[last + 1:] = np.nan
    ball = tracks.clean_ball(x / PITCH[0], y / PITCH[1], PITCH, CFG, fps=FPS)
    horizon = deadball._frames(CFG.ONSET_CONFIRM_S, FPS)
    for meanv, stage in ((np.where(np.arange(n) >= 60, 1.0, 3.0), "ok"), (3.0, "unconfirmed")):
        feats = _feats(n, meanv_all=meanv, sep=14.0)
        starts, stops = deadball._stop_runs(feats, CFG, FPS)
        rows = fo.out_diagnose(ball, CFG, PITCH, FPS, starts, stops, horizon)
        assert [r[2] for r in rows] == [stage]
        table = deadball.dead_onsets(ball, feats, CFG, pitch=PITCH)
        engine = set(table.loc[table["mechanism"] == "out", "frame"])
        assert set(r[1] for r in rows if r[2] == "ok") <= engine


def test_window_helpers_clip_and_ignore_nan():
    v = np.array([1.0, np.nan, 3.0, 5.0])
    assert fo.window_mean(v, -5, 3) == 2.0
    assert fo.window_min(v, 2, 10) == 3.0
    assert np.isnan(fo.window_mean(v, 1, 2))
    assert fo.near_any(100, [10, 220], 125) and not fo.near_any(100, [226], 125)
