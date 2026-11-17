"""Synthetic tests for the gap-dwell onset at the players' deceleration knee (COMPONENT_KNEE.md).

ONSET_KNEE moves a gap-plus-stop dwell entry of possession.dead_intervals() to the knee of the
smoothed mean player speed plus DELTA_KNEE. The new entry stays inside [lo, entry], where lo is
the largest of the previous exit, the look-back cap, the last confident contact and, with
KNEE_GAP_CLAMP, the start of the ball gap. Onset rows and entries with a visible ball never
move. No match data is needed.
"""

import dataclasses

import numpy as np
import pandas as pd

from engine import deadball, possession
from tests.test_deadball import CFG, FPS, _candidates, _onsets

N = 1000
LAST_SEEN = 380        # last visible frame: the gap starts at 381
FALL = 400             # players start to slow down here, 3.0 -> 1.0 m/s over 50 frames
ENTRY = 436            # first frame of meanv_all under V_STOP = 1.6 (1.6 at 400 + 1.4 / 0.04)
KNEE = 417             # smoothed speed last at the level 2.3 (half-way from 3.0 to 1.6)
REAPPEAR = 800
ON = dataclasses.replace(CFG, ONSET_KNEE=True, DELTA_KNEE=0)
OFF = dataclasses.replace(CFG, ONSET_KNEE=False)


def _speed(fall=FALL, top=3.0, bottom=1.0, frames=50):
    """meanv_all: `top` until `fall`, a straight fall to `bottom` over `frames`, then flat."""
    t = np.arange(N, dtype=np.float64)
    ramp = top + (bottom - top) * (t - fall) / frames
    return np.where(t < fall, top, np.maximum(ramp, bottom))


def _ball(last_seen=LAST_SEEN, reappear=REAPPEAR):
    visible = np.ones(N, dtype=bool)
    visible[last_seen + 1:reappear] = False
    return {"visible": visible, "fps": FPS}


def _e_dead(ball, speed, cfg=CFG):
    """e_D high where the ball is missing and the players are under V_STOP (the gap-stop term)."""
    return np.where(~ball["visible"] & (speed < cfg.V_STOP), 0.9, 0.05).astype(np.float32)


def _events(rows=()):
    return pd.DataFrame(list(rows), columns=["frame", "kind", "team", "confidence"]).astype(
        {"frame": np.int64, "confidence": np.float64})


def _stoppage(cfg, ball=None, speed=None, events=None, onsets=None, records=None,
              candidates=None, e_dead=None):
    ball = _ball() if ball is None else ball
    speed = _speed() if speed is None else speed
    e_dead = _e_dead(ball, speed, cfg) if e_dead is None else e_dead
    cands = candidates if candidates is not None else _candidates(
        [(REAPPEAR, "H", "reappear", 0.6, True)])
    knee = None
    if cfg.ONSET_KNEE:
        knee = possession.knee_retimer(ball, {"meanv_all": speed}, events, cfg, FPS, records)
    return possession.dead_intervals(N, _onsets([]) if onsets is None else onsets, e_dead,
                                     cands, [(1, 0, N - 1)], {1: 0}, cfg, FPS, None, knee)


def _first(dead):
    return int(np.flatnonzero(dead)[0])


# -- the helpers --------------------------------------------------------------------------------

def test_smoothed_speed_keeps_a_constant_and_spreads_a_spike_over_13_frames():
    flat = deadball.smoothed_speed(np.full(100, 2.0), CFG, FPS)
    assert np.allclose(flat, 2.0)
    spike = np.zeros(100)
    spike[50] = 13.0
    out = deadball.smoothed_speed(spike, CFG, FPS)
    assert np.flatnonzero(out > 0).tolist() == list(range(44, 57))
    assert np.allclose(out[44:57], 1.0)


def test_smoothed_speed_skips_nan_frames():
    values = np.full(40, 2.0)
    values[10:15] = np.nan
    out = deadball.smoothed_speed(values, CFG, FPS)
    assert np.allclose(out, 2.0)
    assert np.isnan(deadball.smoothed_speed(np.full(20, np.nan), CFG, FPS)).all()


def test_knee_is_the_last_frame_at_the_half_way_level():
    smooth = deadball.smoothed_speed(_speed(), CFG, FPS)
    assert deadball.deceleration_knee(smooth, ENTRY, ENTRY - 100, CFG, FPS) == KNEE


def test_no_knee_when_the_players_were_already_slow():
    smooth = deadball.smoothed_speed(np.full(N, 1.4), CFG, FPS)
    assert deadball.deceleration_knee(smooth, ENTRY, ENTRY - 100, CFG, FPS) is None


def test_no_knee_when_the_window_is_empty():
    smooth = deadball.smoothed_speed(_speed(), CFG, FPS)
    assert deadball.deceleration_knee(smooth, ENTRY, ENTRY, CFG, FPS) is None


def test_knee_is_the_lower_bound_when_the_fall_began_before_it():
    smooth = deadball.smoothed_speed(_speed(), CFG, FPS)
    assert deadball.deceleration_knee(smooth, ENTRY, 425, CFG, FPS) == 425


def test_gap_starts_is_the_frame_after_the_last_sighting():
    visible = np.array([True, True, False, False, True, False])
    assert deadball.gap_starts(visible).tolist() == [1, 2, 2, 2, 5, 5]


# -- dead_intervals with the knee ---------------------------------------------------------------

def test_switch_off_keeps_the_detected_entry():
    dead, restarts = _stoppage(OFF)
    assert _first(dead) == ENTRY and not dead[REAPPEAR]
    assert restarts == [(REAPPEAR, "H")]


def test_gap_dwell_entry_moves_to_the_knee_and_the_exit_stays():
    rows = []
    dead, restarts = _stoppage(ON, records=rows)
    assert _first(dead) == KNEE
    assert dead[KNEE:REAPPEAR].all() and not dead[REAPPEAR:].any()
    assert restarts == [(REAPPEAR, "H")]
    assert rows == [{"entry": ENTRY, "lo": LAST_SEEN + 1, "bound": "gap", "knee": KNEE,
                     "new": KNEE, "floor": None}]


def test_delta_knee_moves_the_entry_and_the_gap_clamp_holds_it():
    cfg = dataclasses.replace(ON, DELTA_KNEE=-50)
    assert _first(_stoppage(cfg)[0]) == LAST_SEEN + 1


def test_without_the_gap_clamp_the_look_back_cap_holds_it():
    cfg = dataclasses.replace(ON, DELTA_KNEE=-200, KNEE_GAP_CLAMP=False)
    rows = []
    assert _first(_stoppage(cfg, records=rows)[0]) == ENTRY - 100
    assert rows[0]["bound"] == "lookback"


def test_a_confident_contact_holds_the_entry():
    cfg = dataclasses.replace(ON, DELTA_KNEE=-200, KNEE_GAP_CLAMP=False)
    sure = _events([(405, "release", "H", 0.8)])
    rows = []
    assert _first(_stoppage(cfg, events=sure, records=rows)[0]) == 405
    assert rows[0]["bound"] == "contact"


def test_a_weak_contact_or_a_later_contact_does_not_hold_it():
    cfg = dataclasses.replace(ON, DELTA_KNEE=-200, KNEE_GAP_CLAMP=False)
    weak = _events([(405, "touch", "H", 0.3), (500, "release", "A", 0.9)])
    assert _first(_stoppage(cfg, events=weak)[0]) == ENTRY - 100


def test_the_entry_never_moves_later():
    cfg = dataclasses.replace(ON, DELTA_KNEE=100)
    assert _first(_stoppage(cfg)[0]) == ENTRY


def test_an_onset_row_never_moves():
    onsets = _onsets([(ENTRY, "out", 0.6)])
    dead, _ = _stoppage(dataclasses.replace(ON, DELTA_KNEE=-50), onsets=onsets)
    assert _first(dead) == ENTRY


def test_a_dwell_entry_with_a_visible_ball_never_moves():
    ball = _ball()
    speed = _speed()
    e_dead = np.where(speed < CFG.V_STOP, 0.9, 0.05).astype(np.float32)   # e.g. a placement
    ball["visible"][:] = True
    dead, _ = _stoppage(dataclasses.replace(ON, DELTA_KNEE=-50), ball=ball, speed=speed,
                        e_dead=e_dead)
    assert _first(dead) == ENTRY


def test_the_entry_never_moves_before_the_previous_exit():
    # A first stoppage 300..420 (exit at 420), then the gap dwell of a second fall. With no
    # contact in between, the knee may reach the exit frame itself, and no earlier.
    speed = np.minimum(_speed(fall=200, frames=10), 3.0)
    speed[420:] = _speed(fall=500)[420:]
    ball = _ball(last_seen=299, reappear=REAPPEAR)
    ball["visible"][420:460] = True                     # the ball is back between the two
    e_dead = _e_dead(ball, speed)
    e_dead[300:420] = 0.9
    cands = _candidates([(420, "A", "kick", 0.6, True), (REAPPEAR, "H", "reappear", 0.6, True)])
    cfg = dataclasses.replace(ON, DELTA_KNEE=-400, KNEE_GAP_CLAMP=False, KNEE_LOOKBACK_S=20.0)
    rows = []
    dead, restarts = _stoppage(cfg, ball=ball, speed=speed, candidates=cands, e_dead=e_dead,
                               records=rows)
    assert restarts == [(420, "A"), (REAPPEAR, "H")]
    assert [(r["bound"], r["lo"], r["knee"], r["new"]) for r in rows] == [
        ("pos", 0, None, 300), ("pos", 420, 517, 420)]
    assert dead[300:REAPPEAR].all() and not dead[:300].any()


def test_knee_fields_are_semantic_and_on_by_default():
    # On since 8 Nov 2026: variant 2 passed the acceptance test (COMPONENT_KNEE.md Section 9).
    assert CFG.ONSET_KNEE is True and CFG.KNEE_GAP_CLAMP is True
    for name in ("ONSET_KNEE", "DELTA_KNEE", "KNEE_LOOKBACK_S", "KNEE_SMOOTH_S",
                 "KNEE_PRE_FROM_S", "KNEE_PRE_TO_S", "KNEE_LEVEL_FRAC", "KNEE_GAP_CLAMP",
                 "KNEE_MIN_PRE_SPELL_S"):
        assert name in possession.SEMANTIC_FIELDS
    assert possession.detection_key(OFF) == possession.detection_key(ON)


def test_knee_is_deterministic():
    cfg = dataclasses.replace(ON, DELTA_KNEE=-7)
    first, _ = _stoppage(cfg)
    second, _ = _stoppage(cfg)
    assert np.array_equal(first, second)


# -- variant 2: the preceding spell keeps at least KNEE_MIN_PRE_SPELL_S (COMPONENT_KNEE.md 8) --

def _labels(text):
    return np.array(list(text))


def _row(entry, new):
    return {"entry": entry, "new": new}


def test_floor_moves_an_entry_that_cuts_a_short_spell_later():
    # An A spell starts at 4 after an H spell; the re-timed entry at 6 leaves it 2 frames.
    labels = _labels("HHHHAADDDDDDDD")
    assert possession.spell_floors(labels, [_row(10, 6)], 3) == {10: 7}


def test_floor_returns_the_entry_when_the_spell_cannot_reach_the_minimum():
    labels = _labels("HHHHAADDDDDDDD")
    assert possession.spell_floors(labels, [_row(8, 6)], 5) == {8: 8}


def test_floor_keeps_a_long_enough_spell_and_entries_that_did_not_move():
    labels = _labels("HHHHAAAADDDDDD")
    assert possession.spell_floors(labels, [_row(10, 8), _row(9, 9)], 3) == {}


def test_floor_ignores_an_entry_after_a_d_frame():
    labels = _labels("HHHDDDDDDDDDDD")
    assert possession.spell_floors(labels, [_row(10, 6)], 3) == {}


def test_floors_replace_the_new_frame_and_are_recorded():
    rows = []
    knee = possession.knee_retimer(_ball(), {"meanv_all": _speed()}, None, ON, FPS, rows,
                                   floors={ENTRY: 430})
    dead, _ = possession.dead_intervals(N, _onsets([]), _e_dead(_ball(), _speed()),
                                        _candidates([(REAPPEAR, "H", "reappear", 0.6, True)]),
                                        [(1, 0, N - 1)], {1: 0}, ON, FPS, None, knee)
    assert _first(dead) == 430
    assert rows[0]["floor"] == KNEE and rows[0]["new"] == 430
