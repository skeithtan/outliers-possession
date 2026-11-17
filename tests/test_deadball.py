"""Synthetic tests for engine/deadball.py and the M1 overlay in engine/possession.py (card P1-B).

The ball dict comes from engine.tracks.clean_ball on hand-built metre positions (consistent
velocity / run_id / dv_k); the feats dict is built directly with only the keys deadball reads.
No match data is needed except games.py constants (kickoff test).
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from engine import data, deadball, possession, tracks
from engine.config import Config

FPS = 25.0
PITCH = (105.0, 68.0)
CFG = Config.default()


def make_ball(x, y):
    """Ball dict from metre positions (NaN = missing) via tracks.clean_ball."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return tracks.clean_ball(x / PITCH[0], y / PITCH[1], PITCH, CFG, fps=FPS)


def make_feats(n, **values):
    """feats dict with every key deadball reads; defaults = ordinary open play.

    Defaults: both teams 10 m from the ball, players moving at 2.5 m/s, centroids 8 m apart.
    Scalars broadcast; arrays are used as given. dmin / nearest_team follow dH / dA unless given.
    """
    feats = {"dH": 10.0, "dA": 10.0, "meanv_all": 2.5, "meanv_near": 2.5, "sep": 8.0}
    feats.update(values)
    out = {}
    for key, value in feats.items():
        arr = np.asarray(value, dtype=np.float64)
        out[key] = np.full(n, float(arr)) if arr.ndim == 0 else arr.astype(np.float64)
    if "dmin" not in out:
        out["dmin"] = np.fmin(out["dH"], out["dA"])
    if "nearest_team" not in values:
        team = np.where(out["dH"] <= out["dA"], data.HOME, data.AWAY)
        out["nearest_team"] = np.where(np.isfinite(out["dmin"]), team, -1).astype(np.int8)
    else:
        out["nearest_team"] = np.asarray(values["nearest_team"], dtype=np.int8)
    if "d_second" not in out:
        out["d_second"] = np.fmax(out["dH"], out["dA"])
    return out


def positions_from_speed(speed, start, heading=(0.0, 1.0)):
    """Metre positions of a ball moving along `heading` with per-frame speeds (m/s)."""
    speed = np.asarray(speed, dtype=np.float64)
    step = np.concatenate(([0.0], np.cumsum(speed[:-1]) / FPS))
    return start[0] + heading[0] * step, start[1] + heading[1] * step


def onsets_of(table, mechanism):
    return table[table["mechanism"] == mechanism]


# -- out ----------------------------------------------------------------------------------------

def test_out_fast_onset_at_extrapolated_crossing():
    """8 m/s toward the touchline, last seen 3 m inside: onset within +-2 of the true crossing."""
    n, last = 200, 49
    y = 65.0 - 8.0 * (last - np.arange(n)) / FPS
    x = np.full(n, 50.0)
    x[last + 1:] = np.nan
    y[last + 1:] = np.nan
    ball = make_ball(x, y)
    stop = np.arange(n) >= 60
    feats = make_feats(n, meanv_all=np.where(stop, 1.0, 3.0), sep=14.0)
    out = onsets_of(deadball.dead_onsets(ball, feats, CFG, pitch=PITCH), "out")
    true_crossing = last + 3.0 / 8.0 * FPS
    assert len(out) == 1
    assert abs(out["frame"].iloc[0] - true_crossing) <= 2
    assert out["confidence"].iloc[0] == CFG.CONF_OUT_EXTRAPOLATED


def test_out_slow_onset_at_last_visible_plus_lag():
    """3 m/s decelerating, 13 m from the line, then a gap and a collective stop (12851)."""
    n, last = 250, 49
    speed = 3.0 + CFG.FRICTION_DECEL * (last - np.arange(last + 1)) / FPS
    _, y = positions_from_speed(speed, (40.0, 0.0))
    y = y - y[-1] + (PITCH[1] - 13.0)
    x = np.full(n, 40.0)
    yy = np.full(n, np.nan)
    yy[:last + 1] = y
    x[last + 1:] = np.nan
    ball = make_ball(x, yy)
    stop = np.arange(n) >= 75
    feats = make_feats(n, meanv_all=np.where(stop, 1.1, 3.0), sep=14.0)
    out = onsets_of(deadball.dead_onsets(ball, feats, CFG, pitch=PITCH), "out")
    assert out["frame"].tolist() == [last + CFG.DELTA_OUT_LAG]
    assert out["confidence"].iloc[0] == CFG.CONF_OUT_LAG
    e_dead = deadball.dead_evidence(ball, feats, None, None, CFG, pitch=PITCH)
    assert (e_dead[80:] > CFG.DEAD_EVIDENCE_HIGH).all()      # gap + collective stop: strong


def test_out_needs_a_collective_stop_when_the_ball_is_missing():
    """Same ball as the fast out, but players keep running: no onset (16625 false out)."""
    n, last = 200, 49
    y = 65.0 - 8.0 * (last - np.arange(n)) / FPS
    x = np.full(n, 50.0)
    x[last + 1:] = np.nan
    y[last + 1:] = np.nan
    feats = make_feats(n, meanv_all=3.0, sep=14.0)
    assert onsets_of(deadball.dead_onsets(make_ball(x, y), feats, CFG, pitch=PITCH), "out").empty


def test_out_observed_beyond_the_line_needs_no_confirmation():
    """A shot tracked over the goal line (18027): onset at the first frame beyond the line."""
    n = 80
    x = 10.0 - 20.0 * np.arange(n) / FPS          # 20 m/s toward x = 0, crosses at frame 12.5
    y = np.full(n, 30.0)
    ball = make_ball(x, y)
    feats = make_feats(n)
    out = onsets_of(deadball.dead_onsets(ball, feats, CFG, pitch=PITCH), "out")
    assert out["frame"].tolist() == [13]
    e_dead = deadball.dead_evidence(ball, feats, None, None, CFG, pitch=PITCH)
    assert (e_dead[15:] > 0.95).all() and (e_dead[:12] < CFG.DEAD_EVIDENCE_HIGH).all()



def _touchline_track():
    """400 frames of ball (C10 brief). Each block starts after a gap except the first.

      0-119    rolls toward y = 68 and crosses it at frame 100 (seen at y = 69): a real out.
      200-210  reappears beyond the touchline at y = 69.5 and moves inward to y = 66: a throw-in.
      300-310  reappears beyond the goal line at x = -1.0 and moves inward to x = 2.
      350-360  reappears beyond the touchline at y = 69 and moves outward to y = 71.
    """
    n = 400
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    x[:120] = 50.0
    y[:100] = np.linspace(60.0, 67.9, 100)
    y[100:120] = 69.0
    x[200:211] = 40.0
    y[200:211] = np.linspace(69.5, 66.0, 11)
    x[300:311] = np.linspace(-1.0, 2.0, 11)
    y[300:311] = 34.0
    x[350:361] = 70.0
    y[350:361] = np.linspace(69.0, 71.0, 11)
    return make_ball(x, y), make_feats(n)


def test_touchline_reappearance_moving_inward_is_not_an_out_onset():
    """A ball first seen beyond the touchline and moving inward is a throw-in (C10: 19772)."""
    ball, feats = _touchline_track()
    out = onsets_of(deadball.dead_onsets(ball, feats, CFG, pitch=PITCH), "out")
    assert out["frame"].tolist() == [100, 300, 350]
    assert (out["confidence"] == CFG.CONF_OUT_OBSERVED).all()


def test_touchline_rule_is_deterministic_and_keeps_the_reappear_restart():
    """The dropped onset leaves the reappear candidate at frame 200 in place."""
    ball, feats = _touchline_track()
    first = deadball.dead_onsets(ball, feats, CFG, pitch=PITCH)
    second = deadball.dead_onsets(ball, feats, CFG, pitch=PITCH)
    pd.testing.assert_frame_equal(first, second)
    table = deadball.restarts(ball, feats, None, None, CFG, onsets=first)
    reappear = table[table["kind"] == "reappear"]
    assert 200 in reappear["frame"].tolist()


def test_touchline_rule_falls_back_to_vy_without_a_later_visible_frame():
    """With no later frame in the window (OUT_INWARD_FRAMES = 0), vy decides the direction."""
    n = 60
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    x[20:23] = 30.0
    y[20:23] = [-1.6, -1.3, -1.0]          # beyond y = 0, moving toward the pitch
    x[40:43] = 30.0
    y[40:43] = [-1.0, -1.3, -1.6]          # beyond y = 0, moving away from the pitch
    cfg = dataclasses.replace(CFG, OUT_INWARD_FRAMES=0)
    ball = make_ball(x, y)
    out = onsets_of(deadball.dead_onsets(ball, make_feats(n), cfg, pitch=PITCH), "out")
    assert out["frame"].tolist() == [40]


# -- pursuit stop -------------------------------------------------------------------------------

def _pursuit_case(d_other):
    """A plays the ball at frame 40 (15 m/s, then slowing); A falls behind; the 6 nearest players
    decelerate from 3.5 m/s at frame 60 to 1.0 m/s at 100; collective stop from 70."""
    n = 220
    t = np.arange(n)
    speed = np.where(t < 40, 1.0, np.minimum(15.0, 1.0 + 3.5 * (t - 39)))
    speed = np.where(t > 44, np.maximum(15.0 - 0.15 * (t - 44), 0.0), speed)
    x, y = positions_from_speed(speed, (20.0, 34.0), heading=(1.0, 0.0))
    ball = make_ball(x, y)
    d_kick = np.where(t < 40, 0.5, 0.5 + 0.4 * (t - 40))
    near = np.where(t < 60, 3.5, np.maximum(3.5 - 2.5 * (t - 60) / 40.0, 1.0))
    feats = make_feats(n, dA=d_kick, dH=d_other, meanv_near=near,
                       meanv_all=np.where(t >= 70, 1.0, 3.0), sep=14.0)
    return ball, feats


def test_pursuit_stop_onset_at_start_of_deceleration():
    t = np.arange(220)
    ball, feats = _pursuit_case(d_other=np.where(t < 40, 4.0, 4.0 + 0.3 * (t - 40)))
    ps = onsets_of(deadball.dead_onsets(ball, feats, CFG, pitch=PITCH), "pursuit_stop")
    assert ps["frame"].tolist() == [60]


def test_pursuit_stop_with_a_defender_near_the_landing():
    """The other team's distance falls (17380: dA 8.6 -> 3.4 m): still a pursuit stop."""
    t = np.arange(220)
    ball, feats = _pursuit_case(d_other=np.maximum(10.0 - 0.15 * np.maximum(t - 40, 0), 2.0))
    ps = onsets_of(deadball.dead_onsets(ball, feats, CFG, pitch=PITCH), "pursuit_stop")
    assert ps["frame"].tolist() == [60]


def test_pursuit_stop_rejected_when_a_teammate_receives():
    """The kicking team reaches the ball 1 s after the onset: a pass, not a stop (18612)."""
    t = np.arange(220)
    ball, feats = _pursuit_case(d_other=12.0)
    feats["dA"] = np.where((t >= 85) & (t < 95), 0.3, feats["dA"])
    feats["dmin"] = np.fmin(feats["dH"], feats["dA"])
    assert onsets_of(deadball.dead_onsets(ball, feats, CFG, pitch=PITCH), "pursuit_stop").empty


# -- evidence: what must stay low ---------------------------------------------------------------

def test_live_gap_with_players_running_keeps_e_dead_low():
    """Ball missing 8 s while players run at 3 m/s: hold, not D (PLAN 1.1, R-F2)."""
    n = 300
    x = 50.0 + 2.0 * np.arange(n) / FPS
    y = np.full(n, 34.0)
    x[50:250] = np.nan
    y[50:250] = np.nan
    ball = make_ball(x, y)
    feats = make_feats(n, meanv_all=3.0, sep=10.0, dH=1.0, dA=4.0)
    e_dead = deadball.dead_evidence(ball, feats, None, None, CFG, pitch=PITCH)
    assert e_dead[50:250].max() < 0.1
    assert deadball.dead_onsets(ball, feats, CFG, pitch=PITCH).empty


def test_ghost_ball_after_a_shot_keeps_e_dead_low():
    """27 m/s shot, then the ball 'stationary' with nobody within 6 m (13755-13800)."""
    n = 100
    t = np.arange(n)
    x = np.where(t < 20, 60.0 - 27.0 * t / FPS, 60.0 - 27.0 * 19 / FPS)
    y = np.where(t < 20, 34.0, 34.0 + 0.03 * np.sin(t))       # jitter on a still ball
    ball = make_ball(x, y)
    far = np.where(t < 5, 0.5, 6.7)
    feats = make_feats(n, dH=far, dA=far + 1.0, meanv_all=1.2, sep=14.0)
    e_dead = deadball.dead_evidence(ball, feats, None, None, CFG, pitch=PITCH)
    ghost = deadball.ghost_mask(ball, feats, CFG)
    assert ghost[25:].all() and not ghost[:19].any()
    assert not deadball.setpiece_mask(ball, feats, CFG).any()
    assert e_dead.max() < CFG.DEAD_EVIDENCE_HIGH        # weak terms (slow + sep) stay below


def test_weak_terms_alone_stay_below_high():
    n = 50
    ball = make_ball(np.full(n, 50.0) + 0.2 * np.arange(n), np.full(n, 34.0))
    feats = make_feats(n, meanv_all=1.0, sep=15.0)
    e_dead = deadball.dead_evidence(ball, feats, None, None, CFG, pitch=PITCH)
    assert e_dead.max() < CFG.DEAD_EVIDENCE_HIGH


def test_set_piece_tag_tail_is_weak_and_only_at_the_end():
    n = 400
    ball = make_ball(np.full(n, np.nan), np.full(n, np.nan))
    feats = make_feats(n, meanv_all=3.0)
    tags = pd.DataFrame([{"code": "SET PIECES", "team": None, "type": "Throw-in",
                          "start_frame": 100, "end_frame": 300},
                         {"code": "GOALS", "team": "H", "type": None,
                          "start_frame": 0, "end_frame": 400}])
    e_dead = deadball.dead_evidence(ball, feats, tags, None, CFG, pitch=PITCH)
    tail = int(CFG.SETPIECE_TAG_TAIL_S * FPS)
    assert e_dead[300 - tail:300].min() > e_dead[100:300 - tail].max()
    assert e_dead.max() < CFG.DEAD_EVIDENCE_HIGH             # weak prior; GOALS is no mask


# -- restarts -----------------------------------------------------------------------------------

def test_kick_restart_from_a_rolling_ball():
    """2.2 m/s roll, then +2 m/s per 3 frames to 7 m/s (16400): kick within 3 frames of the rise,
    although no 3-frame step exceeds 3 m/s."""
    n, rise = 160, 60
    t = np.arange(n)
    speed = np.where(t < rise, 2.2 - 0.005 * t, 0.0)
    speed = np.where(t >= rise, np.minimum((2.2 - 0.005 * rise) + 2.0 / 3.0 * (t - rise + 1), 7.0),
                     speed)
    x, y = positions_from_speed(speed, (30.0, 30.0), heading=(1.0, 0.0))
    ball = make_ball(x, y)
    assert np.nanmax(ball["dv_k"]) <= 2.1
    feats = make_feats(n, dA=0.8, dH=12.0, meanv_all=np.where(t > rise + 20, 2.5, 1.0))
    table = deadball.restarts(ball, feats, None, None, CFG)
    kicks = table[table["kind"] == "kick"]
    assert len(kicks) == 1
    assert abs(kicks["frame"].iloc[0] - rise) <= 3
    assert kicks["team"].iloc[0] == "A" and bool(kicks["confirmed"].iloc[0])


def test_set_piece_e_dead_high_then_kick_restart_with_team():
    """Ball still, one H player at 0.8 m, others far; then a kick at frame 100."""
    n, kick = 200, 100
    t = np.arange(n)
    speed = np.where(t < kick, 0.0, np.minimum(4.0 * (t - kick + 1), 12.0))
    x, y = positions_from_speed(speed, (30.0, 20.0), heading=(1.0, 0.0))
    y = y + np.where(t < kick, 0.02 * np.sin(t), 0.0)
    ball = make_ball(x, y)
    taker = np.where(t < kick + 3, 0.8, 4.0)
    feats = make_feats(n, dH=taker, dA=np.full(n, 9.0), d_second=np.full(n, 8.0),
                       meanv_all=np.where(t < kick, 0.8, 2.5), sep=14.0)
    e_dead = deadball.dead_evidence(ball, feats, None, None, CFG, pitch=PITCH)
    assert (e_dead[:kick - 2] > CFG.DEAD_EVIDENCE_HIGH).all()
    assert deadball.setpiece_mask(ball, feats, CFG)[:kick - 2].all()
    kicks = deadball.restarts(ball, feats, None, None, CFG)
    kicks = kicks[kicks["kind"] == "kick"]
    assert len(kicks) == 1 and abs(kicks["frame"].iloc[0] - kick) <= 3
    assert kicks["team"].iloc[0] == "H"


def test_reappear_restart_team_only_when_close():
    n = 120
    t = np.arange(n)
    x = np.where(t < 60, np.nan, 40.0 + 9.0 * (t - 60) / FPS)
    y = np.where(t < 60, np.nan, 30.0)
    ball = make_ball(x, y)
    near = make_feats(n, dH=np.where(t >= 60, 2.0, np.nan), dA=np.where(t >= 60, 6.0, np.nan),
                      meanv_all=2.5)
    far = make_feats(n, dH=np.where(t >= 60, 5.0, np.nan), dA=np.where(t >= 60, 6.0, np.nan),
                     meanv_all=2.5)
    for feats, team in ((near, "H"), (far, None)):
        table = deadball.restarts(ball, feats, None, None, CFG)
        reappear = table[table["kind"] == "reappear"]
        assert reappear["frame"].tolist() == [60]
        assert reappear["team"].iloc[0] == team


def test_restart_priors_adjust_confidence_only():
    n, kick = 200, 100
    t = np.arange(n)
    speed = np.where(t < kick, 0.0, np.minimum(4.0 * (t - kick + 1), 12.0))
    x, y = positions_from_speed(speed, (30.0, 20.0), heading=(1.0, 0.0))
    ball = make_ball(x, y)
    feats = make_feats(n, dH=np.where(t < kick + 3, 0.8, 4.0), dA=9.0, meanv_all=2.5)
    agree = pd.DataFrame([{"code": "SET PIECES", "team": "H", "type": "Free Kick",
                           "start_frame": 50, "end_frame": 150}])
    clash = agree.assign(team="A")
    base = deadball.restarts(ball, feats, None, None, CFG)
    up = deadball.restarts(ball, feats, None, agree, CFG)
    down = deadball.restarts(ball, feats, None, clash, CFG)
    assert up["team"].tolist() == base["team"].tolist() == down["team"].tolist()
    assert up["confidence"].iloc[0] > base["confidence"].iloc[0] >= down["confidence"].iloc[0]


# -- kickoff ------------------------------------------------------------------------------------

def test_kickoff_frames_from_games_py():
    kick = deadball.kickoff_frames()
    assert kick == data.kickoff_frames()
    assert abs(kick[1] - 11185) <= 1
    assert abs(kick[2] - 113742) <= 1


def test_pre_kickoff_frames_are_certain_dead():
    n = 200
    ball = make_ball(np.full(n, 50.0) + 0.1 * np.arange(n), np.full(n, 34.0))
    feats = make_feats(n)
    e_dead = deadball.dead_evidence(ball, feats, None, [(1, 0, 99), (2, 100, 199)], CFG,
                                    kickoffs={1: 0, 2: 150}, pitch=PITCH)
    assert (e_dead[100:150] == 1.0).all()
    assert e_dead[:100].max() < 0.5 and e_dead[150:].max() < 0.5


# -- M1 overlay (engine/possession.py) ----------------------------------------------------------

def _candidates(rows):
    table = pd.DataFrame(rows, columns=deadball.RESTART_COLUMNS)
    return table.astype({"frame": np.int64, "confirmed": bool})


def _onsets(rows):
    return pd.DataFrame(rows, columns=deadball.ONSET_COLUMNS).astype({"frame": np.int64})


def test_dead_intervals_onset_to_first_armed_confirmed_restart():
    n = 300
    onsets = _onsets([(50, "out", 0.6), (200, "collective_stop", 0.3)])
    cands = _candidates([(60, "H", "kick", 0.6, True),      # too early: not armed (< 1 s)
                         (90, "H", "kick", 0.6, False),     # unconfirmed: a re-play inside D
                         (120, "A", "kick", 0.6, True)])
    dead, restarts = possession.dead_intervals(n, onsets, np.zeros(n), cands, [(1, 0, n - 1)],
                                               {1: 0}, CFG, FPS)
    assert np.flatnonzero(dead).tolist() == list(range(50, 120))   # collective stop: no entry
    assert restarts == [(120, "A")]


def test_dead_intervals_evidence_dwell_and_kickoff():
    n = 400
    e_dead = np.zeros(n)
    e_dead[100:146] = 0.9          # 46 frames < COLLECTIVE_STOP_S: stays alive (17161-17207)
    e_dead[200:260] = 0.9          # 60 frames: D from the start of the run
    cands = _candidates([(300, None, "reappear", 0.6, True)])
    dead, restarts = possession.dead_intervals(n, _onsets([]), e_dead, cands, [(2, 0, n - 1)],
                                               {2: 30}, CFG, FPS)
    expected = list(range(0, 30)) + list(range(200, 300))
    assert np.flatnonzero(dead).tolist() == expected
    assert restarts == [(30, None), (300, None)]


def test_m1_labels_hold_ghost_restart_team_and_x():
    n = 100
    blocks = [(1, 10, 89)]
    base = np.array(["D"] * 15 + ["H"] * 25 + ["D"] * 10 + ["A"] * 20 + ["H"] * 30)
    ghost = np.zeros(n, dtype=bool)
    ghost[48:54] = True            # straddles the baseline's H -> A flip at 50
    dead = np.zeros(n, dtype=bool)
    dead[62:66] = True
    labels = possession.m1_labels(base, dead, [(66, "A")], ghost, blocks, n)
    assert (labels[:10] == "X").all() and (labels[90:] == "X").all()
    assert not (labels[10:90] == "X").any()
    assert (labels[10:40] == "H").all()             # block opens on D: back-filled
    assert (labels[40:50] == "H").all()             # baseline D (no ball): held
    assert (labels[48:54] == "H").all()             # ghost: frozen at the frame before
    assert (labels[54:62] == "A").all()
    assert (labels[62:66] == "D").all()
    assert (labels[66:70] == "A").all()             # restart team until the owner's next change
    assert (labels[70:90] == "H").all()


# -- C11: chain confirmation and the kick offset ------------------------------------------------

def _kick_track(kicks, n=1000, moving_from=480):
    """A stoppage with kicks of a resting ball and players that start to move late.

    Each kick raises the ball speed from 0 to 5 m/s over 9 frames with an A player 0.8 m away,
    holds 5 m/s for 10 frames and stops over 10 frames. meanv_all is 1.0 before `moving_from`
    and 2.0 from it. Returns (ball, feats).
    """
    t = np.arange(n)
    speed = np.zeros(n)
    for kick in kicks:
        rise = np.clip((t - kick + 1) / 9.0, 0.0, 1.0) * 5.0
        fall = np.clip(1.0 - (t - kick - 19) / 10.0, 0.0, 1.0)
        speed = np.maximum(speed, np.where(t >= kick, rise * fall, 0.0))
    x, y = positions_from_speed(speed, (30.0, 30.0), heading=(1.0, 0.0))
    feats = make_feats(n, dA=0.8, dH=12.0, meanv_all=np.where(t >= moving_from, 2.0, 1.0),
                       sep=14.0)
    return make_ball(x, y), feats


def _kick_rows(table):
    kicks = table[table["kind"] == "kick"]
    return kicks["frame"].tolist(), kicks["confirmed"].tolist()


def test_chain_confirms_a_kick_that_the_next_confirmed_kick_follows():
    """Kicks at 300 and 370, players move from 480 (C11: the 18355 goal kick)."""
    ball, feats = _kick_track((300, 370))
    frames, confirmed = _kick_rows(deadball.restarts(ball, feats, None, None, CFG))
    assert len(frames) == 2 and abs(frames[0] - 300) <= 2 and abs(frames[1] - 370) <= 2
    assert confirmed == [True, True]
    players = deadball.restarts(ball, feats, None, None,
                                dataclasses.replace(CFG, RESTART_CHAIN_FRAMES=0))
    assert _kick_rows(players)[1] == [False, True]          # 300 is confirmed by the chain only


def test_chain_rejects_a_kick_when_the_next_one_comes_too_late():
    """The second kick 100 frames later (more than RESTART_CHAIN_FRAMES): the first fails."""
    ball, feats = _kick_track((300, 400))
    frames, confirmed = _kick_rows(deadball.restarts(ball, feats, None, None, CFG))
    assert frames[1] - frames[0] > CFG.RESTART_CHAIN_FRAMES
    assert confirmed == [False, True]


def test_chain_keeps_an_isolated_kick_unconfirmed():
    """One kick, no later candidate, players never move: unconfirmed."""
    ball, feats = _kick_track((300,), moving_from=10 ** 6)
    assert _kick_rows(deadball.restarts(ball, feats, None, None, CFG))[1] == [False]


def test_chain_keeps_an_in_stoppage_replay_rejected():
    """A re-play 335 frames before the real restart stays unconfirmed (16058)."""
    ball, feats = _kick_track((300, 635), n=1200, moving_from=700)
    frames, confirmed = _kick_rows(deadball.restarts(ball, feats, None, None, CFG))
    assert abs(frames[1] - frames[0] - 335) <= 2
    assert confirmed == [False, True]


def test_chain_does_not_cross_a_stoppage_onset():
    """Kicks at 300 and 370 with an out onset at 340: the first stays unconfirmed."""
    ball, feats = _kick_track((300, 370))
    onsets = _onsets([(340, "out", 1.0)])
    assert _kick_rows(deadball.restarts(ball, feats, None, None, CFG, onsets=onsets))[1] == [
        False, True]
    soft = _onsets([(340, "collective_stop", 0.3)])            # a soft hint is no barrier
    assert _kick_rows(deadball.restarts(ball, feats, None, None, CFG, onsets=soft))[1] == [
        True, True]


def test_chain_confirmed_rules_on_plain_arrays():
    """Same-frame rows do not chain each other, a chain never runs through a chained row, and a
    barrier on the later frame breaks the chain."""
    frames = [100, 170, 240, 500, 500]
    players = [False, False, True, False, True]
    out = deadball.chain_confirmed(frames, players, [], 87)
    assert out.tolist() == [False, True, True, False, True]   # 100 does not chain through 170
    assert deadball.chain_confirmed(frames, players, [240], 87).tolist() == [
        False, False, True, False, True]
    assert deadball.chain_confirmed(frames, players, [], 0).tolist() == players
    assert deadball.chain_confirmed([], [], [], 87).tolist() == []


def test_restarts_with_the_chain_are_deterministic():
    ball, feats = _kick_track((300, 370))
    first = deadball.restarts(ball, feats, None, None, CFG)
    second = deadball.restarts(ball, feats, None, None, CFG)
    pd.testing.assert_frame_equal(first, second)


def test_kick_offset_moves_a_kick_exit_and_not_a_reappear_exit():
    """A stoppage 400-499 exited by a kick at 500 ends at 512. A reappear exit moves by
    DELTA_RESTART_REAPPEAR only. The entry at an out onset moves by DELTA_DEAD (0)."""
    assert CFG.DELTA_RESTART_KICK == 12 and CFG.DELTA_DEAD == 0
    n, blocks = 1000, [(1, 0, 999)]
    stoppage = np.zeros(n, dtype=bool)
    stoppage[400:500] = True
    onsets = _onsets([(400, "out", 1.0)])
    for kind, delta in (("kick", CFG.DELTA_RESTART_KICK),
                        ("reappear", CFG.DELTA_RESTART_REAPPEAR)):
        cands = _candidates([(500, "H", kind, 0.6, True)])
        out, restart_list = possession.shift_stoppage(stoppage, [(500, "H")], onsets, cands, CFG,
                                                      blocks, {1: 0})
        assert np.flatnonzero(out).tolist() == list(range(400, 500 + delta)), kind
        assert restart_list == [(500 + delta, "H")]
