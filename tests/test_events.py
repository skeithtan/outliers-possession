"""Synthetic tests for engine/events.py and the M2 orchestrator in engine/possession.py (card P1-A).

Each scene builds metre positions for the ball and the players. The ball goes through
engine.tracks.clean_ball and the players through engine.tracks.frame_features, so the inputs have
the same shape and units as the match. The slow test at the end needs the match data bundle.
"""

import time

import numpy as np
import pandas as pd
import pytest

import games
from engine import data, deadball, events, possession, tracks
from engine.config import Config

FPS = 25.0
PITCH = (105.0, 68.0)
# The P1-A tests below test the "dv3" detector. The C8 tests at the end use BUMP.
CFG = Config(**dict(Config.default().as_dict(), CONTACT_DETECTOR="dv3"))
BUMP = Config.default()
H, A = data.HOME, data.AWAY


# -- scene builders -----------------------------------------------------------------------------

def integrate(speeds, headings_deg, x0=30.0, y0=30.0):
    """Ball positions (m) when the ball moves at speeds[t] m/s along headings_deg[t] from t to t+1."""
    speeds = np.asarray(speeds, dtype=np.float64)
    rad = np.radians(np.asarray(headings_deg, dtype=np.float64))
    step_x = speeds * np.cos(rad) / FPS
    step_y = speeds * np.sin(rad) / FPS
    x = x0 + np.concatenate(([0.0], np.cumsum(step_x[:-1])))
    y = y0 + np.concatenate(([0.0], np.cumsum(step_y[:-1])))
    return x, y


def static(team, x, y, n):
    """A player who stands at (x, y) for n frames."""
    return team, np.full(n, float(x)), np.full(n, float(y))


def build(bx, by, players):
    """(ball, feats, slot_conf) of a scene, through engine.tracks."""
    px = np.column_stack([p[1] for p in players]).astype(np.float64)
    py = np.column_stack([p[2] for p in players]).astype(np.float64)
    tactical = {"px": px / PITCH[0], "py": py / PITCH[1],
                "team_of_slot": np.array([p[0] for p in players], dtype=np.int8),
                "ps": np.full(px.shape, np.nan), "fps": FPS}
    ball = tracks.clean_ball(np.asarray(bx) / PITCH[0], np.asarray(by) / PITCH[1], PITCH, CFG,
                             fps=FPS)
    feats = tracks.frame_features(tactical, ball, PITCH, CFG, fps=FPS)
    ball["airborne_suspect"] = feats["airborne_suspect"].copy()
    return ball, feats, tracks.slot_confidence(tactical, CFG)


def detect(scene, e_dead=None):
    ball, feats, slot_conf = scene
    return events.detect_events(ball, feats, None, slot_conf, CFG, e_dead=e_dead)


def pass_scene():
    """A1 kicks a resting ball at frame 60 (1 -> 12 m/s), the ball rolls 15 m, A2 stops it."""
    speeds = np.concatenate((np.zeros(60), [3.0, 6.0, 9.0], np.full(28, 12.0),
                             [9.0, 6.0, 3.0, 1.0], np.full(50, 0.5)))
    n = speeds.size
    bx, by = integrate(speeds, np.zeros(n))
    stop = 60 + 3 + 28 + 4 + 10
    players = [static(A, 30.0, 30.3, n), static(A, bx[stop], by[stop] + 0.6, n),
               static(H, 30.0, 10.0, n), static(H, 50.0, 10.0, n)]
    return build(bx, by, players), stop


# -- the card's cases ---------------------------------------------------------------------------

def test_pass_gives_release_and_control_of_the_same_team_and_nothing_in_flight():
    scene, stop = pass_scene()
    table = detect(scene)
    feats = scene[1]
    release = table[(table["kind"] == "release") & (table["team"] == "A")]
    assert len(release) == 1 and abs(int(release["frame"].iloc[0]) - 60) <= 3
    controls = table[(table["kind"] == "control") & (table["team"] == "A")]
    assert (controls["frame"] > 80).any()                       # A2 controls the reception
    assert (table["team"] != "H").all()
    in_flight = np.flatnonzero(feats["dmin"] > 3.0)
    assert not table["frame"].isin(in_flight).any()


def test_interception_mid_flight_is_an_A_touch():
    """H releases at 30, A turns the ball through 60 degrees at 70 (10 -> 7 m/s)."""
    speeds = np.concatenate((np.zeros(30), [3.0, 6.0, 9.0], np.full(37, 10.0), np.full(40, 7.0)))
    headings = np.concatenate((np.zeros(70), np.full(40, 60.0)))
    n = speeds.size
    bx, by = integrate(speeds, headings)
    players = [static(H, 30.0, 29.7, n), static(A, bx[70] + 0.3, by[70] - 0.4, n),
               static(A, 90.0, 60.0, n)]
    table = detect(build(bx, by, players))
    kick = table[(table["team"] == "H") & (table["kind"] == "release")]
    assert len(kick) == 1 and abs(int(kick["frame"].iloc[0]) - 30) <= 3
    touch = table[(table["team"] == "A") & table["kind"].isin(("touch", "release"))
                  & (abs(table["frame"] - 70) <= 4)]
    assert len(touch) == 1
    assert touch["notes"].iloc[0] != events.DEFLECTION_NOTE


def test_deflection_is_labelled_with_half_confidence():
    """12 -> 10.8 m/s (-10%) and a 5 degree turn while passing 0.8 m from an A player."""
    speeds = np.concatenate((np.full(50, 12.0), np.full(50, 10.8)))
    headings = np.concatenate((np.zeros(50), np.full(50, 5.0)))
    n = speeds.size
    bx, by = integrate(speeds, headings)
    players = [static(A, bx[50], by[50] + 0.8, n), static(H, 10.0, 60.0, n)]
    table = detect(build(bx, by, players))
    rows = table[table["team"] == "A"]
    assert len(rows) == 1
    assert rows["notes"].iloc[0] == events.DEFLECTION_NOTE
    assert rows["confidence"].iloc[0] <= 0.5
    assert rows["dist_m"].iloc[0] <= 1.0


def test_contested_touch_has_no_team():
    """H and A both 0.6 m from a resting ball that is kicked at frame 40."""
    speeds = np.concatenate((np.zeros(40), [3.0, 6.0, 9.0], np.full(37, 12.0)))
    n = speeds.size
    bx, by = integrate(speeds, np.zeros(n))
    players = [static(H, 30.0, 30.6, n), static(A, 30.0, 29.4, n)]
    table = detect(build(bx, by, players))
    near = table[abs(table["frame"] - 40) <= 3]
    assert near["kind"].tolist() == ["contested"]
    assert near["team"].iloc[0] is None
    assert (table["kind"] != "control").all()                   # contested frames never control


def test_airborne_ball_gives_no_event_and_the_next_ground_contact_is_attributed():
    """18 -> 16.5 m/s in the air beside a player, landing at 8 m/s, then an H contact (8 -> 5)."""
    speeds = np.concatenate((np.full(40, 18.0), np.full(20, 16.5), np.full(30, 8.0),
                             np.full(40, 5.0)))
    headings = np.concatenate((np.zeros(90), np.full(40, 30.0)))
    n = speeds.size
    bx, by = integrate(speeds, headings, x0=5.0)
    players = [static(A, bx[40], by[40] + 2.4, n), static(A, bx[40], by[40] - 1.0, n),
               static(H, bx[90], by[90] + 1.0, n)]
    scene = build(bx, by, players)
    assert scene[0]["airborne_suspect"][38:42].all()
    table = detect(scene)
    assert not (table["frame"] < 80).any()                      # nothing during the flight
    contact = table[abs(table["frame"] - 90) <= 4]
    assert contact["team"].tolist() == ["H"]


def dribble_scene(first=30, frames=40, n=100, speed=6.0, offset=0.8, opponent=None):
    """An A player runs at `speed` m/s beside the ball, `offset` m away, on frames
    [first, first + frames). He runs 4 m to the side on the other frames. An H player stands
    far away, or runs `opponent` m from the ball on the other side on the same frames."""
    bx, by = integrate(np.full(n, speed), np.zeros(n))
    side = np.full(n, 4.0)
    side[first:first + frames] = offset
    players = [(A, bx.copy(), by + side)]
    if opponent is None:
        players.append(static(H, 10.0, 60.0, n))
    else:
        other = np.full(n, 4.0)
        other[first:first + frames] = float(opponent)
        players.append((H, bx.copy(), by - other))
    return build(bx, by, players)


def test_dribble_with_a_fast_ball_is_a_dribble_and_not_a_control():
    """6 m/s with the ball 0.8 m away for 40 frames: one dribble at the first frame (M10)."""
    table = detect(dribble_scene())
    assert (table["kind"] != "control").all()
    assert table["kind"].tolist() == ["dribble"]
    row = table.iloc[0]
    assert (int(row["frame"]), row["team"], row["notes"]) == (30, "A", "end=69")
    assert 0.5 <= row["confidence"] < 1.0
    assert row["dist_m"] == pytest.approx(0.8)


def test_short_run_beside_the_ball_is_not_a_dribble():
    """13 frames (13674-13698 has a longest run of 13 frames within 1 m, RESEARCH M10)."""
    assert detect(dribble_scene(frames=13)).empty


def test_contested_dribble_has_no_team():
    """An H player runs 0.9 m from the ball on the other side: the margin is 0.1 m."""
    table = detect(dribble_scene(opponent=0.9))
    assert table["kind"].tolist() == ["dribble"]
    assert table["team"].iloc[0] is None
    assert table["notes"].iloc[0] == "contested;end=69"


def test_slow_ball_at_a_standing_player_is_a_control_not_a_dribble():
    """0.5 m/s for 60 frames, within 0.7 m of a player who stands still."""
    n = 60
    bx, by = integrate(np.full(n, 0.5), np.zeros(n))
    players = [static(A, 30.6, 30.3, n), static(H, 10.0, 60.0, n)]
    table = detect(build(bx, by, players))
    assert table["kind"].tolist() == ["control"]
    assert table["team"].tolist() == ["A"]


def test_ball_that_passes_a_player_gives_no_event():
    """6 m/s past a standing player: within 1 m on 5 frames, no speed change."""
    n = 100
    bx, by = integrate(np.full(n, 6.0), np.zeros(n))
    players = [static(A, bx[50], by[50] + 0.8, n), static(H, 10.0, 60.0, n)]
    scene = build(bx, by, players)
    assert (scene[1]["dA"] <= CFG.R_DRIBBLE).sum() == 5
    assert detect(scene).empty


def test_no_dribble_inside_dead_ball():
    scene = dribble_scene()
    n = scene[0]["visible"].shape[0]
    assert detect(scene, e_dead=np.full(n, 0.9)).empty


def test_dribble_detection_is_deterministic():
    first, second = detect(dribble_scene()), detect(dribble_scene())
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == events.COLUMNS


def test_no_event_inside_dead_ball():
    scene, _ = pass_scene()
    n = scene[0]["visible"].shape[0]
    assert detect(scene, e_dead=np.full(n, 0.9)).empty
    partial = np.zeros(n)
    partial[50:70] = 0.9                                        # dead around the kick only
    table = detect(scene, e_dead=partial)
    assert not ((table["frame"] >= 50) & (table["frame"] < 70)).any()
    assert (table["kind"] == "control").any()


def test_detect_events_is_deterministic():
    scene, _ = pass_scene()
    ball, feats, slot_conf = scene
    noise = np.random.RandomState(0).normal(0.0, 0.01, size=ball["x"].shape)
    noisy = build(ball["x"] + noise, ball["y"] - noise,
                  [static(A, 30.0, 30.3, noise.size), static(H, 50.0, 31.0, noise.size)])
    first, second = detect(noisy), detect(noisy)
    pd.testing.assert_frame_equal(first, second)
    assert first["frame"].is_monotonic_increasing
    assert list(first.columns) == events.COLUMNS


# -- helpers and contracts ----------------------------------------------------------------------

def test_contact_is_at_the_start_of_the_speed_rise():
    scene, _ = pass_scene()
    ball = scene[0]
    peak = 60 + int(np.nanargmax(ball["dv_k"][55:70])) - 5
    contact = events.contact_frame(ball["speed"], ball["run_id"], peak, CFG)
    assert contact < peak
    assert ball["speed"][contact] < CFG.CONTROL_MAX_SPEED
    assert peak - contact <= CFG.TOUCH_ONSET_MAX_FRAMES


def test_confidence_bounds_match_the_owner_threshold():
    assert events.confidence(0.0, np.inf, 1.0, CFG) == pytest.approx(CFG.EVENT_CONF_MAX)
    worst = events.confidence(CFG.R_TOUCH, CFG.CONTESTED_MARGIN_M, 1.0, CFG)
    assert worst >= CFG.OWNER_MIN_CONFIDENCE
    halved = events.confidence(0.0, np.inf, CFG.OVERCOUNT_CONF_FACTOR, CFG)
    assert halved < CFG.OWNER_MIN_CONFIDENCE
    assert CFG.EVENT_CONF_MAX * CFG.DEFLECTION_CONF_FACTOR < CFG.OWNER_MIN_CONFIDENCE


def table_of(rows):
    """Event table from (frame, kind, team, confidence, notes) tuples."""
    frame = pd.DataFrame([(f, k, t, c, 0.5, nt) for f, k, t, c, nt in rows],
                         columns=events.COLUMNS)
    frame["team"] = frame["team"].astype(object).where(frame["team"].notna(), None)
    return frame


def test_touch_mask_ignores_control_rows():
    table = table_of([(3, "touch", "H", 0.7, ""), (5, "control", "A", 0.7, "end=9"),
                      (7, "contested", None, 0.7, ""), (8, "release", "A", 0.7, "")])
    assert np.flatnonzero(events.touch_mask(table, 10)).tolist() == [3, 7, 8]


# -- deadball with real events ------------------------------------------------------------------

def _stop_scene(n=200):
    ball = tracks.clean_ball(np.full(n, 0.5), np.full(n, 0.5), PITCH, CFG, fps=FPS)
    feats = {"dH": np.full(n, 3.0), "dA": np.full(n, 4.0), "dmin": np.full(n, 3.0),
             "d_second": np.full(n, 4.0), "nearest_team": np.zeros(n, dtype=np.int8),
             "meanv_all": np.full(n, 1.0), "meanv_near": np.full(n, 1.0), "sep": np.full(n, 14.0)}
    return ball, feats


def test_collective_stop_uses_real_touches_when_given():
    ball, feats = _stop_scene()
    pitch = PITCH
    no_events = deadball.dead_onsets(ball, feats, CFG, pitch=pitch, events=table_of([]))
    assert (no_events["mechanism"] == "collective_stop").sum() == 1
    touches = table_of([(f, "touch", "H", 0.7, "") for f in range(20, 200, 40)])
    quiet = deadball.dead_onsets(ball, feats, CFG, pitch=pitch, events=touches)
    assert (quiet["mechanism"] == "collective_stop").sum() == 0
    proxy = deadball.dead_onsets(ball, feats, CFG, pitch=pitch)            # M1 path
    assert (proxy["mechanism"] == "collective_stop").sum() == 1


def test_last_toucher_comes_from_the_event_table():
    ball, feats = _stop_scene()
    table = table_of([(10, "touch", "H", 0.7, ""), (20, "release", "A", 0.7, ""),
                      (30, "control", "H", 0.7, "end=40"), (35, "contested", None, 0.7, "")])
    assert deadball._last_toucher_before(ball, feats, CFG, 50, table) == "A"
    assert deadball._last_toucher_before(ball, feats, CFG, 15, table) == "H"
    assert deadball._last_toucher_before(ball, feats, CFG, 5, table) is None


# -- the M2 orchestrator ------------------------------------------------------------------------

def test_owner_events_apply_rule_confidence_and_deflection_filter():
    table = table_of([
        (10, "release", "A", 0.7, ""),
        (20, "release", "H", 0.4, ""),                       # below OWNER_MIN_CONFIDENCE
        (30, "touch", "H", 0.8, ""),                         # touches do not set the owner
        (40, "release", "H", 0.35, events.DEFLECTION_NOTE),  # confirmed by the control at 55
        (55, "control", "H", 0.7, "end=60"),
        (70, "release", "A", 0.35, events.DEFLECTION_NOTE),  # not confirmed within 25 frames
        (100, "release", "A", 0.7, ""),
        (110, "contested", None, 0.8, ""),
    ])
    assert possession.owner_events(table, CFG) == [(10, "A"), (40, "H"), (55, "H"), (100, "A")]
    release_only = possession.owner_events(
        table, Config(**dict(CFG.as_dict(), POSSESSION_RULE="release")))
    assert (55, "H") not in release_only and (10, "A") in release_only


def test_m2_owner_holds_backfills_and_takes_the_restart_team():
    blocks = [(1, 0, 99), (2, 150, 199)]
    owner = possession.m2_owner([(30, "A"), (60, "H")], [(80, "A"), (150, None)], blocks, 220)
    assert set(owner[0:60]) == {"A"}                          # back-fill from the first event
    assert set(owner[60:80]) == {"H"}
    assert set(owner[80:100]) == {"A"}                        # the restart team owns
    assert set(owner[150:200]) == {"H"}                       # a block without events
    assert set(owner[100:150]) == {""} and set(owner[200:]) == {""}
    kick = possession.m2_owner([(30, "A")], [(0, "H")], [(1, 0, 99)], 100)
    assert set(kick[0:30]) == {"H"}                           # the kickoff team before any event


# -- C8: bump-centre contacts -------------------------------------------------------------------
# The ATD smooths the ball with a kernel of sigma about 9.5 frames (RESEARCH-2 M13). The scenes
# smooth the ball velocity with that Gaussian and integrate it. A step of size s then gives an
# impulse of about 0.79 s at h = 12, so the scenes use speeds of 1.6 x DV_CONTACT.

SIGMA = 9.5
FAST = 1.6 * BUMP.DV_CONTACT


def gaussian_smooth(values, sigma=SIGMA):
    """`values` convolved with a normalised Gaussian of `sigma` frames, edges held."""
    values = np.asarray(values, dtype=np.float64)
    half = int(4 * sigma)
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate((np.full(half, values[0]), values, np.full(half, values[-1])))
    return np.convolve(padded, kernel, mode="valid")


def smooth_path(speeds, headings_deg, x0=10.0, y0=10.0):
    """Ball positions (m) from per-frame speed and heading, with the velocity smoothed."""
    rad = np.radians(np.asarray(headings_deg, dtype=np.float64))
    speeds = np.asarray(speeds, dtype=np.float64)
    vx = gaussian_smooth(speeds * np.cos(rad))
    vy = gaussian_smooth(speeds * np.sin(rad))
    x = x0 + np.concatenate(([0.0], np.cumsum(vx[:-1]) / FPS))
    y = y0 + np.concatenate(([0.0], np.cumsum(vy[:-1]) / FPS))
    return x, y


def piecewise(n, changes):
    """(N,) array that takes the value of each (frame, value) of `changes` from that frame on."""
    out = np.zeros(n)
    for frame, value in changes:
        out[frame:] = value
    return out


def bump_build(bx, by, players):
    """(ball, feats, slot_conf) of a scene through engine.tracks, with the BUMP config."""
    px = np.column_stack([p[1] for p in players]).astype(np.float64)
    py = np.column_stack([p[2] for p in players]).astype(np.float64)
    tactical = {"px": px / PITCH[0], "py": py / PITCH[1],
                "team_of_slot": np.array([p[0] for p in players], dtype=np.int8),
                "ps": np.full(px.shape, np.nan), "fps": FPS}
    ball = tracks.clean_ball(np.asarray(bx) / PITCH[0], np.asarray(by) / PITCH[1], PITCH, BUMP,
                             fps=FPS)
    feats = tracks.frame_features(tactical, ball, PITCH, BUMP, fps=FPS)
    ball["airborne_suspect"] = feats["airborne_suspect"].copy()
    return ball, feats, tracks.slot_confidence(tactical, BUMP)


def bump_detect(scene, e_dead=None):
    ball, feats, slot_conf = scene
    return events.detect_events(ball, feats, None, slot_conf, BUMP, e_dead=e_dead)


def contact_rows(table):
    """The rows of `table` that are not control runs or dribbles."""
    runs = table["notes"].astype(str).str.startswith("end=") | (table["kind"] == "dribble")
    return table[~runs.to_numpy()]


def far(team, n):
    return static(team, 100.0, 65.0, n)


def kick_scene(n=200, at=100):
    """The ball rests at (10, 10) and leaves at FAST m/s along x from frame `at`."""
    return smooth_path(piecewise(n, [(at, FAST)]), np.zeros(n))


def test_kick_is_a_release_at_the_bump_centre():
    n, at = 200, 100
    bx, by = kick_scene(n, at)
    scene = bump_build(bx, by, [static(H, 10.0, 10.3, n), far(A, n)])
    rows = contact_rows(bump_detect(scene))
    assert rows["kind"].tolist() == ["release"]
    row = rows.iloc[0]
    assert row["team"] == "H" and abs(int(row["frame"]) - at) <= 2
    assert row["confidence"] >= BUMP.OWNER_MIN_CONFIDENCE
    # the dv3 contact walks back to the start of the rise, the bump stamp does not
    dv3 = events.detect_events(*scene[:2], None, scene[2], CFG)
    dv3_rows = dv3[dv3["kind"] == "release"]
    assert len(dv3_rows) == 1 and int(dv3_rows["frame"].iloc[0]) < at - 4


def test_reception_is_a_control_with_the_receive_note():
    n, at = 200, 100
    bx, by = smooth_path(piecewise(n, [(0, FAST), (at, 0.0)]), np.zeros(n))
    scene = bump_build(bx, by, [static(A, bx[at] + 0.4, by[at], n), far(H, n)])
    table = bump_detect(scene)
    rows = contact_rows(table)
    assert rows["kind"].tolist() == ["control"]
    row = rows.iloc[0]
    assert (row["team"], row["notes"]) == ("A", events.RECEIVE_NOTE)
    assert abs(int(row["frame"]) - at) <= 2
    assert (table["kind"] != "touch").all()


def redirect_scene(redirector, next_team=None, n=220):
    """H1 kicks at 40 (to FAST m/s), the ball turns 60 degrees at 100 beside `redirector`,
    and stops at frame 160 beside a player of `next_team` (no player if None)."""
    speeds = piecewise(n, [(40, FAST)] + ([(160, 0.0)] if next_team is not None else []))
    headings = piecewise(n, [(100, 60.0)])
    bx, by = smooth_path(speeds, headings)
    players = [static(H, 10.0, 10.3, n),
               static(redirector, bx[100] + 0.25, by[100] - 0.45, n),
               far(H if redirector == A else A, n)]
    if next_team is not None:
        players.append(static(next_team, bx[160] + 0.3, by[160] + 0.3, n))
    return bump_build(bx, by, players)


def test_redirect_by_the_owner_team_is_a_touch():
    rows = contact_rows(bump_detect(redirect_scene(H)))
    assert rows["kind"].tolist() == ["release", "touch"]
    touch = rows.iloc[1]
    assert (touch["team"], touch["notes"]) == ("H", events.REDIRECT_NOTE)
    assert abs(int(touch["frame"]) - 100) <= 3


def test_redirect_by_the_other_team_confirmed_by_its_next_contact_is_a_control():
    rows = contact_rows(bump_detect(redirect_scene(A, next_team=A)))
    assert rows["kind"].tolist() == ["release", "control", "control"]
    redirect = rows.iloc[1]
    assert (redirect["team"], redirect["notes"]) == ("A", events.REDIRECT_NOTE)
    assert abs(int(redirect["frame"]) - 100) <= 3
    assert rows.iloc[2]["notes"] == events.RECEIVE_NOTE


def test_unconfirmed_redirect_by_the_other_team_has_no_row():
    table = bump_detect(redirect_scene(A, next_team=H))
    rows = contact_rows(table)
    assert rows["kind"].tolist() == ["release", "control"]
    assert rows["team"].tolist() == ["H", "H"]                  # the H receive at 160
    assert (table["team"] != "A").all()                         # no A event, so no A owner event
    assert all(team == "H" for _, team in possession.release_rows(table, BUMP))


def test_dead_frame_between_breaks_the_confirmation():
    scene = redirect_scene(A, next_team=A)
    n = scene[0]["visible"].shape[0]
    e_dead = np.zeros(n)
    e_dead[130:135] = 0.9                                       # a stoppage between 100 and 160
    rows = contact_rows(bump_detect(scene, e_dead=e_dead))
    assert not ((rows["team"] == "A") & (abs(rows["frame"] - 100) <= 3)).any()


def test_ball_that_passes_a_player_at_2_m_gives_no_contact():
    n = 200
    bx, by = smooth_path(np.full(n, FAST), np.zeros(n))
    assert contact_rows(bump_detect(bump_build(bx, by, [static(A, bx[100], by[100] + 2.0, n),
                                                        far(H, n)]))).empty
    turn_x, turn_y = smooth_path(np.full(n, FAST), piecewise(n, [(100, 60.0)]))
    scene = bump_build(turn_x, turn_y, [static(A, turn_x[100], turn_y[100] + 2.0, n), far(H, n)])
    assert np.nanmax(events.impulse(scene[0], BUMP.CONTACT_HALF_WINDOW)) >= BUMP.DV_CONTACT
    assert contact_rows(bump_detect(scene)).empty               # nobody within R_TOUCH


def test_contested_contact_has_no_team():
    n, at = 200, 100
    bx, by = kick_scene(n, at)
    scene = bump_build(bx, by, [static(H, 10.0, 10.4, n), static(A, 10.0, 9.4, n)])
    rows = contact_rows(bump_detect(scene))
    assert rows["kind"].tolist() == ["contested"]
    assert rows["team"].iloc[0] is None


def test_no_contact_inside_dead_ball_and_deterministic():
    n, at = 200, 100
    bx, by = kick_scene(n, at)
    scene = bump_build(bx, by, [static(H, 10.0, 10.3, n), far(A, n)])
    assert bump_detect(scene, e_dead=np.full(n, 0.9)).empty
    partial = np.zeros(n)
    partial[95:105] = 0.9
    assert contact_rows(bump_detect(scene, e_dead=partial)).empty
    first, second = bump_detect(scene), bump_detect(scene)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == events.COLUMNS


def test_impulse_and_window_maxima_helpers():
    ball = {"vx": np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0]), "vy": np.zeros(7),
            "run_id": np.array([0, 0, 0, 0, 0, 1, 1])}
    size = events.impulse(ball, 1)
    assert np.isnan(size[0]) and np.isnan(size[-1])
    assert size.tolist()[1:4] == [0.0, 5.0, 5.0]
    assert np.isnan(size[4]) and np.isnan(size[5])              # the ends lie in two runs
    values = np.array([1.0, 3.0, 3.0, 2.0, np.nan, 4.0])
    assert events.window_maxima(values, 1, np.arange(6)).tolist() == [1, 5]


def test_m2_uses_the_dv3_detector_and_bad_detector_names_raise():
    assert possession.dv3_config(BUMP).CONTACT_DETECTOR == "dv3"
    assert possession.dv3_config(CFG) is CFG
    bad = Config(**dict(BUMP.as_dict(), CONTACT_DETECTOR="other"))
    bx, by = kick_scene()
    scene = bump_build(bx, by, [static(H, 10.0, 10.3, 200), far(A, 200)])
    with pytest.raises(ValueError):
        events.detect_events(*scene[:2], None, scene[2], bad)


# -- the match (slow, needs the data bundle) ----------------------------------------------------

_ASSETS = games.asset_paths()
HAVE_DATA = all(__import__("os").path.isfile(_ASSETS[k])
                for k in ("atd_meta", "atd_raw", "homography", "smart_tagging"))


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_DATA, reason="match data bundle not present")
def test_full_match_events_are_fast_deterministic_and_plausible():
    tactical = data.load_tactical()
    ball, feats, slot_conf = tracks.prepare(tactical, CFG)
    e_dead = events.match_dead_evidence(ball, feats, tactical, CFG)
    start = time.time()
    first = events.detect_events(ball, feats, tactical, slot_conf, CFG, e_dead=e_dead)
    assert time.time() - start < 10.0
    second = events.detect_events(ball, feats, tactical, slot_conf, CFG, e_dead=e_dead)
    pd.testing.assert_frame_equal(first, second)
    graded = np.zeros(ball["visible"].shape[0], dtype=bool)
    for _, lo, hi in data.graded_blocks():
        graded[lo:hi + 1] = True
    minutes = (graded & ball["visible"]).sum() / FPS / 60.0
    rows = first[graded[first["frame"].to_numpy()]
                 & ~first["kind"].isin(("control", "dribble")).to_numpy()]
    assert 15.0 <= len(rows) / minutes <= 50.0                # PLAN 3.3 plausible band
    table = events.event_table(tactical, CFG)
    assert list(table.columns) == events.TABLE_COLUMNS
    assert table["frame"].tolist() == first["frame"].tolist()


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_DATA, reason="match data bundle not present")
def test_full_match_bump_contacts_are_fast_deterministic_and_keep_the_other_rows():
    tactical = data.load_tactical()
    ball, feats, slot_conf = tracks.prepare(tactical, BUMP)
    e_dead = events.match_dead_evidence(ball, feats, tactical, BUMP)
    start = time.time()
    first = events.detect_events(ball, feats, tactical, slot_conf, BUMP, e_dead=e_dead)
    assert time.time() - start < 10.0
    second = events.detect_events(ball, feats, tactical, slot_conf, BUMP, e_dead=e_dead)
    pd.testing.assert_frame_equal(first, second)
    old = events.detect_events(ball, feats, tactical, slot_conf, CFG, e_dead=e_dead)
    for frame in (first, old):                                  # control runs and dribbles stay
        assert frame["kind"].isin(events.KINDS).all()
    def runs(table):
        return table[table["notes"].astype(str).str.contains("end=")].reset_index(drop=True)

    pd.testing.assert_frame_equal(runs(first), runs(old))
    bump = first[~first["notes"].astype(str).str.contains("end=")]
    assert set(bump["notes"]) <= {"", events.RECEIVE_NOTE, events.REDIRECT_NOTE}
