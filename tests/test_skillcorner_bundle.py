"""tools/skillcorner_bundle.py: SkillCorner Open Data match to an ATD-style bundle and a reference.

The synthetic tests write a few frames of SkillCorner-like files into a temporary folder. They need
no match data and no network. The last tests compare games.py with the built bundles and skip
when the bundles are absent.
"""

import json
import os

import numpy as np
import pytest

import games
from engine import data
from tools import skillcorner_bundle as sc

H, A, D, X = sc.H, sc.A, sc.D, sc.X
MATCH_ID = "900001"
HOME_ID, AWAY_ID = 11, 22
# Players: two home (one goalkeeper), two away (one goalkeeper), one home substitute who never
# plays. Player ids are not in team order, so the slot order is tested.
PLAYERS = [(7, HOME_ID, "GK"), (3, HOME_ID, "CM"), (9, AWAY_ID, "GK"), (5, AWAY_ID, "CF"),
           (8, HOME_ID, "CF")]
# Tracking frames 0..24. Period 1 is 2..10 and period 2 is 13..22.
N_SOURCE = 25
PERIODS = {1: (2, 10), 2: (13, 22)}


def _match_json(folder):
    raw = {
        "id": int(MATCH_ID),
        "date_time": "2024-01-01T00:00:00Z",
        "home_team": {"id": HOME_ID, "name": "Home FC"},
        "away_team": {"id": AWAY_ID, "name": "Away & Co"},
        "players": [{"id": pid, "team_id": team, "player_role": {"acronym": role}}
                    for pid, team, role in PLAYERS],
        "match_periods": [{"period": h, "start_frame": a, "end_frame": b}
                          for h, (a, b) in sorted(PERIODS.items())],
        "pitch_length": 100,
        "pitch_width": 60,
        "home_team_side": ["left_to_right", "right_to_left"],
    }
    path = os.path.join(folder, "{}_match.json".format(MATCH_ID))
    with open(path, "w") as handle:
        json.dump(raw, handle)


def _period(f):
    for half, (a, b) in PERIODS.items():
        if a <= f <= b:
            return half
    return None


def _ball(f):
    """Ball at x = f metres, y = 1. Undetected (extrapolated) on frames 5 and 16."""
    if _period(f) is None:
        return {"x": None, "y": None, "z": None, "is_detected": None}
    return {"x": float(f), "y": 1.0, "z": 0.1, "is_detected": f not in (5, 16)}


def _tracking_lines(swap_group=False):
    lines = []
    for f in range(N_SOURCE):
        half = _period(f)
        players = []
        if half is not None:
            # The home goalkeeper stands at x = -40 in half 1 (left_to_right) and +40 in half 2.
            gk_x = -40.0 if half == 1 else 40.0
            players = [{"player_id": 7, "x": gk_x, "y": 0.0, "is_detected": True},
                       {"player_id": 9, "x": -gk_x, "y": 0.0, "is_detected": False},
                       {"player_id": 5, "x": 2.0, "y": -3.0, "is_detected": True}]
            if f != 8:     # player 3 has no position on frame 8
                players.append({"player_id": 3, "x": float(f), "y": 10.0, "is_detected": True})
        group = None
        if half is not None:
            group = "home team" if f < 8 or f >= 17 else "away team"
            if swap_group:
                group = "away team" if group == "home team" else "home team"
        lines.append(json.dumps({
            "frame": f, "timestamp": None, "period": half, "ball_data": _ball(f),
            "possession": {"player_id": None, "group": group}, "player_data": players}))
    return lines


def _write_match(folder, swap_group=False, phases=None):
    os.makedirs(folder, exist_ok=True)
    _match_json(folder)
    with open(os.path.join(folder, "{}_tracking_extrapolated.jsonl".format(MATCH_ID)), "w") as h:
        h.write("\n".join(_tracking_lines(swap_group)) + "\n")
    # Phases: home 3..8, away 8..10 (adjacent), away 17..20 in half 2 (a home team phase in the
    # tracking group, see _tracking_lines), and a half-1 phase cut at the period end.
    rows = phases if phases is not None else [
        (3, 8, 1, HOME_ID), (8, 12, 1, AWAY_ID), (17, 20, 2, HOME_ID)]
    with open(os.path.join(folder, "{}_phases_of_play.csv".format(MATCH_ID)), "w") as h:
        h.write("index,match_id,frame_start,frame_end,period,team_in_possession_id,duration\n")
        for i, (a, b, half, team) in enumerate(rows):
            h.write("{},{},{},{},{},{},{}\n".format(i, MATCH_ID, a, b, half, team,
                                                    (b - a) / 10.0))
    return folder


# -- frame mapping --------------------------------------------------------------------------------

def test_video_block_and_nearest_source():
    # 2.5 * 2 = 5 and 2.5 * 10 = 25. 2.5 * 13 = 32.5 rounds up, 2.5 * 22 = 55.
    assert sc.video_block(2, 10) == (5, 25)
    assert sc.video_block(13, 22) == (33, 55)
    assert sc.video_block(13, 23) == (33, 57)
    # round(0.4 v): 0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4 -> 0, 0, 1, 1, 2, 2, 2.
    assert sc.nearest_source(7).tolist() == [0, 0, 1, 1, 2, 2, 2]


def test_resample_interpolates_and_never_fills_a_gap():
    out = sc.resample(np.array([0.0, 10.0, 20.0]), 6)
    assert np.allclose(out, [0.0, 4.0, 8.0, 12.0, 16.0, 20.0])
    gap = sc.resample(np.array([0.0, np.nan, 20.0, 30.0]), 8)
    # v = 0 is on frame 0. v = 1..4 touch the missing frame 1. v = 5 is on frame 2.
    assert gap[0] == 0.0
    assert np.isnan(gap[1:5]).all()
    assert gap[5] == 20.0
    assert np.allclose(gap[6:8], [24.0, 28.0])
    table = sc.resample(np.array([[0.0, 1.0], [10.0, np.nan]]), 3)
    assert np.allclose(table[:, 0], [0.0, 4.0, 8.0])
    assert table[0, 1] == 1.0 and np.isnan(table[1:, 1]).all()


def test_source_labels_and_phase_cuts():
    spans = {1: (2, 10), 2: (13, 22)}
    phases = [(3, 8, 1, HOME_ID), (8, 12, 1, AWAY_ID), (17, 20, 2, HOME_ID)]
    labels = sc.source_labels(N_SOURCE, spans, phases, HOME_ID, AWAY_ID)
    expected = [X, X, D, H, H, H, H, H, A, A, A, X, X, D, D, D, D, H, H, H, D, D, D, X, X]
    assert labels.tolist() == expected
    assert sc.clip_phases(phases, spans)[1] == (8, 11, 1, AWAY_ID)
    with pytest.raises(ValueError):
        sc.source_labels(N_SOURCE, spans, [(3, 8, 1, 99)], HOME_ID, AWAY_ID)
    with pytest.raises(ValueError):
        sc.source_labels(N_SOURCE, spans, [(3, 8, 1, HOME_ID), (6, 9, 1, AWAY_ID)],
                         HOME_ID, AWAY_ID)
    with pytest.raises(ValueError):
        sc.source_labels(N_SOURCE, spans, [(11, 14, 1, HOME_ID)], HOME_ID, AWAY_ID)


# -- the bundle -----------------------------------------------------------------------------------

def test_build_writes_a_bundle_the_engine_reads(tmp_path):
    folder = _write_match(str(tmp_path / "raw" / MATCH_ID))
    out = str(tmp_path / "out")
    bundle, summary = sc.build_match(folder, out)
    stem = "SC-" + MATCH_ID
    assert bundle["stem"] == stem
    assert bundle["blocks"] == {1: (5, 25), 2: (33, 55)}
    assert bundle["n_frames"] == 56
    fields = summary["games_entry"]
    assert fields["halves"] == {1: (6, 26), 2: (34, 56)}
    assert fields["offsets"] == {1: 1, 2: 1}
    assert fields["kickoff"] == {"minute": 0, "second": 0, "frame": 5}
    assert fields["kickoff_2nd"] == {"minute": 0, "second": 1, "frame": 8}
    assert fields["pitch"] == (100.0, 60.0)
    assert fields["assets"]["atd_raw"] == "skillcorner/{}/{}_atd.txt".format(MATCH_ID, stem)

    # Slots: home first, sorted by player id. Player 8 never plays and has no slot.
    assert bundle["slots"] == [(HOME_ID, 3), (HOME_ID, 7), (AWAY_ID, 5), (AWAY_ID, 9)]
    assert summary["slots"] == {"home": 2, "away": 2}

    # Ball: x = f metres, normalised with length 100. Frame 5 is extrapolated, so the video
    # frames 10..14 (between source frames 4 and 6) are missing. Frame 4 is v = 10.
    bx = bundle["bx"]
    assert np.isnan(bx[:5]).all() and np.isnan(bx[26:33]).all() and np.isnan(bx[56:]).all()
    assert bx[10] == pytest.approx((4.0 + 50.0) / 100.0)
    assert np.isnan(bx[11:15]).all()
    assert bx[15] == pytest.approx((6.0 + 50.0) / 100.0)
    assert bx[16] == pytest.approx((6.4 + 50.0) / 100.0)
    assert bundle["by"][15] == pytest.approx(0.5 - 1.0 / 60.0)
    # Player 3 has no position on frame 8 (v = 20): v = 18..22 are missing, v = 17 is not.
    p3 = bundle["px"][:, 0]
    assert np.isnan(p3[18:23]).all() and np.isfinite(p3[17]) and np.isfinite(p3[23])
    # The extrapolated away goalkeeper is kept.
    assert np.isfinite(bundle["px"][5:26, 3]).all()

    # Reference: nearest source frame. v = 7 -> 2.8 -> 3 (H). v = 20 -> 8 (A). Outside the
    # blocks X. v = 42 -> 16.8 -> 17 (H). v = 50 -> 20 (D).
    labels = bundle["labels"]
    assert (labels[:5] == X).all() and (labels[26:33] == X).all()
    assert labels[5] == D and labels[7] == H and labels[20] == A and labels[25] == A
    assert labels[42] == H and labels[50] == D and labels[55] == D

    stats = summary["stats"]
    assert stats["side_check"]["agrees"]
    assert stats["phases_cut_at_period_end"] == 1
    assert stats["periods_agree"]
    assert stats["team_check"] > 0.5

    # The engine's own parser reads the files.
    meta = data._atd_meta(os.path.join(out, stem + "_atd.xml"), "SC-1874553")
    assert meta["team_names"] == ("Home FC", "Away & Co")
    assert meta["team_of_slot"].tolist() == [0, 0, 1, 1]
    assert meta["player_channels"] == ("x", "y") and meta["ball_channels"] == ("x", "y")
    arrays = data._parse_atd_raw(os.path.join(out, stem + "_atd.txt"), 4, 56,
                                 meta["player_channels"], meta["ball_channels"])
    assert np.allclose(arrays["bx"], bundle["bx"], equal_nan=True, atol=1e-5)
    assert np.isnan(arrays["ps"]).all()
    with open(os.path.join(out, stem + "_reference.csv")) as handle:
        rows = handle.read().splitlines()
    assert rows[0] == "frame,label" and len(rows) == 57 and rows[8] == "7,H"
    assert summary["reference_md5"] == sc.md5(os.path.join(out, stem + "_reference.csv"))


def test_build_is_deterministic(tmp_path):
    folder = _write_match(str(tmp_path / "raw" / MATCH_ID))
    sc.build_match(folder, str(tmp_path / "a"))
    sc.build_match(folder, str(tmp_path / "b"))
    for suffix in ("_atd.xml", "_atd.txt", "_reference.csv", "_build.json"):
        name = "SC-" + MATCH_ID + suffix
        with open(str(tmp_path / "a" / name), "rb") as one, \
                open(str(tmp_path / "b" / name), "rb") as two:
            assert one.read() == two.read()


def test_swapped_teams_raise(tmp_path):
    folder = _write_match(str(tmp_path / "raw" / MATCH_ID), swap_group=True)
    with pytest.raises(ValueError, match="team_check"):
        sc.build_match(folder, str(tmp_path / "out"))


def test_entry_text_is_valid_python(tmp_path):
    folder = _write_match(str(tmp_path / "raw" / MATCH_ID))
    bundle, summary = sc.build_match(folder, str(tmp_path / "out"))
    text = sc.entry_text(bundle, summary["games_entry"])
    assert max(len(line) for line in text.splitlines()) <= 99
    entry = eval("{" + text + "}")["SC-" + MATCH_ID]
    for field in ("n_video_frames", "fps", "offsets", "halves", "kickoff", "kickoff_2nd",
                  "pitch", "assets"):
        assert entry[field] == summary["games_entry"][field]


# -- the games.py entries (no data needed) --------------------------------------------------------

SC_GAMES = sorted(name for name in games.GAMES if name.startswith("SC-"))


def test_twenty_skillcorner_entries():
    assert len(SC_GAMES) == 20


@pytest.mark.parametrize("name", SC_GAMES)
def test_skillcorner_entry_is_consistent(name):
    entry = games.GAMES[name]
    assert entry["fps"] == 25
    assert entry["offsets"] == {1: 1, 2: 1}
    assert entry["n_video_frames"] == entry["halves"][2][1]
    blocks = data.graded_blocks(name)
    assert blocks[-1][2] == entry["n_video_frames"] - 1
    assert blocks[0][2] < blocks[1][1] - 1          # an X gap separates the halves
    kicks = data.kickoff_frames(name)
    assert kicks[1] == blocks[0][1] and kicks[2] == blocks[1][1]
    match_id = name[len("SC-"):]
    assert entry["assets"] == {
        "atd_meta": "skillcorner/{0}/{1}_atd.xml".format(match_id, name),
        "atd_raw": "skillcorner/{0}/{1}_atd.txt".format(match_id, name)}


@pytest.mark.parametrize("name", SC_GAMES)
def test_skillcorner_entry_matches_its_built_bundle(name):
    """Skips unless MTAH_DATA_DIR holds the bundle (see the games.py comment block)."""
    entry = games.GAMES[name]
    path = os.path.join(games.DATA_DIR, entry["assets"]["atd_meta"].replace(
        "_atd.xml", "_build.json"))
    if not os.path.isfile(path):
        pytest.skip("no SkillCorner bundle under MTAH_DATA_DIR")
    with open(path) as handle:
        built = json.load(handle)["games_entry"]
    for field in ("n_video_frames", "fps", "kickoff", "kickoff_2nd", "assets"):
        assert built[field] == entry[field]
    assert {int(h): v for h, v in built["offsets"].items()} == entry["offsets"]
    assert {int(h): tuple(v) for h, v in built["halves"].items()} == entry["halves"]
    assert tuple(built["pitch"]) == entry["pitch"]
