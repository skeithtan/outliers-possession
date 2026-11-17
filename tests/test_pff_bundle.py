"""tools/pff_bundle.py: PFF FC World Cup 2022 match to an ATD-style bundle and a reference.

The synthetic tests write a few frames of PFF-like files into a temporary folder. They need no
match data and no network. The last tests compare games.py with the built bundles and skip when
the bundles are absent.
"""

import bz2
import json
import os

import numpy as np
import pytest

import games
from engine import data
from tools import pff_bundle as pff

H, A, D, X = pff.H, pff.A, pff.D, pff.X
GAME_ID = "900002"
HOME_ID, AWAY_ID = 11, 22
# Roster: (player id, team, jersey, position group). Player 112 never plays.
ROSTER = [(101, HOME_ID, 1, "GK"), (107, HOME_ID, 7, "CM"), (112, HOME_ID, 12, "CF"),
          (201, AWAY_ID, 1, "GK"), (209, AWAY_ID, 9, "CF")]
# Tracking frameNum of each period. Period 3 (extra time) is not in the bundle.
PERIODS = {1: (30, 89), 2: (120, 179), 3: (200, 209)}


def _t(v):
    """The time in seconds of video frame v."""
    return v / 25.0


def _metadata():
    return [{"id": GAME_ID, "date": "2022-12-01T00:00:00",
             "homeTeam": {"id": str(HOME_ID), "name": "Home FC", "shortName": "HOM"},
             "awayTeam": {"id": str(AWAY_ID), "name": "Away & Co", "shortName": "AWA"},
             "stadium": {"id": "1", "name": "S", "pitches": [{"length": 100.0, "width": 60.0}]},
             "startPeriod1": 1.0, "endPeriod1": 2.97, "startPeriod2": 4.0, "endPeriod2": 5.97,
             "homeTeamStartLeft": True, "fps": 29.97}]


def _roster():
    return [{"player": {"id": str(pid), "nickname": "P{}".format(pid)},
             "positionGroupType": group, "shirtNumber": str(jersey), "started": True,
             "team": {"id": str(team), "name": "T"}}
            for pid, team, jersey, group in ROSTER]


def _row(v, kind, half, team=None, seq=None, setpiece=None, ptype=None, player=None,
         out=None, end_v=None, direction=None):
    t = _t(v)
    return {"gameId": int(GAME_ID), "gameEventId": v, "possessionEventId": None,
            "startTime": t, "endTime": _t(end_v) if end_v is not None else t,
            "eventTime": t, "sequence": seq,
            "gameEvents": {"gameEventType": kind, "period": half, "teamId": team,
                           "playerId": player, "setpieceType": setpiece, "outType": out,
                           "initialNonEvent": False},
            "possessionEvents": {"possessionEventType": ptype} if ptype else None,
            "stadiumMetadata": {"teamAttackingDirection": direction}}


def _events():
    # Period 1. The comments give the labels that each row starts.
    rows = [
        _row(25, "FIRSTKICKOFF", 1, HOME_ID, 1, "K", "PA", 107, direction="R"),   # H
        _row(30, "OTB", 1, HOME_ID, 1, "O", "IT", 107, direction="R"),
        _row(35, "OTB", 1, AWAY_ID, 1, "O", "RE", 209),        # a rebound: no change
        _row(40, "OTB", 1, AWAY_ID, 2, "O", "IT", 209, direction="L"),            # A
        _row(45, "OTB", 1, None, 3, "O", "CH"),                # no team: skipped
        _row(47, "OTB", 1, HOME_ID, 3, "O", "PA", 107, direction="R"),            # H
        _row(50, "OUT", 1, out="T"),                           # D
        _row(52, "SUB", 1, HOME_ID),
        _row(60, "OTB", 1, AWAY_ID, 4, "T", "PA", 209, end_v=63, direction="L"),  # A
        _row(65, "OTB", 1, HOME_ID, 5, "F", "PA", 107, direction="R"),  # D at 63, H at 65
        _row(74, "END", 1),
        # Period 2.
        _row(100, "SECONDKICKOFF", 2, AWAY_ID, 6, "K", "PA", 209, direction="R"),  # A
        _row(110, "OTB", 2, AWAY_ID, 6, "O", "IT", 209, direction="R"),
        _row(120, "OUT", 2, out="H"),                          # D
        _row(130, "OTB", 2, AWAY_ID, 7, "K", "PA", 209, direction="R"),            # A
        _row(140, "END", 2),
        _row(170, "THIRDKICKOFF", 3, HOME_ID, 8, "K", "PA", 107),
    ]
    return rows


def _expected_labels():
    labels = [X] * 140
    for a, b, code in ((26, 40, H), (40, 47, A), (47, 50, H), (50, 60, D), (60, 63, A),
                       (63, 65, D), (65, 74, H), (101, 120, A), (120, 130, D), (130, 140, A)):
        labels[a:b] = [code] * (b - a)
    return labels


def _half(f):
    for half, (a, b) in PERIODS.items():
        if a <= f <= b:
            return half
    return None


def _tracking_lines():
    lines = []
    for f in range(PERIODS[1][0], PERIODS[3][1] + 1):
        half = _half(f)
        if half is None:
            continue
        gk_x = -40.0 if half in (1, 3) else 40.0
        mx = (f - 30) * 0.1
        home = [{"jerseyNum": "1", "confidence": "HIGH", "visibility": "VISIBLE",
                 "x": gk_x, "y": 0.0}]
        if f != 50:     # player 7 has no position on frame 50
            home.append({"jerseyNum": "7", "confidence": "HIGH", "visibility": "VISIBLE",
                         "x": mx, "y": 5.0})
        if f == 60:     # a second entry with jersey 7: dropped
            home.append({"jerseyNum": "7", "confidence": "LOW", "visibility": "ESTIMATED",
                         "x": 30.0, "y": 30.0})
        away = [{"jerseyNum": "1", "confidence": "HIGH", "visibility": "ESTIMATED",
                 "x": -gk_x, "y": 0.0},
                {"jerseyNum": "9", "confidence": "MEDIUM", "visibility": "ESTIMATED",
                 "x": mx, "y": 5.0}]
        if f in (31, 32):   # a jersey that is not in the roster: dropped
            away.append({"jerseyNum": "99", "confidence": "LOW", "visibility": "VISIBLE",
                         "x": 0.0, "y": 0.0})
        visible = "ESTIMATED" if 40 <= f <= 44 else "VISIBLE"
        row = {"version": None, "videoTimeMs": f * 1001 / 30.0, "frameNum": f, "period": half,
               "homePlayers": home, "awayPlayers": away,
               "homePlayersSmoothed": [], "awayPlayersSmoothed": [],
               "balls": [{"visibility": visible, "x": mx, "y": 5.0, "z": 0.0}],
               "ballsSmoothed": {"visibility": "VISIBLE", "x": 99.0, "y": 99.0, "z": 0.0},
               "game_event": None, "possession_event": None}
        if f == 31:
            row["game_event"] = {"start_frame": 30, "start_time": 30 * 1001 / 30000.0}
        lines.append(json.dumps(row, separators=(",", ":")))     # compact, as the files
        if f == 33:
            # The files repeat a frame for each further possession event on it.
            row["game_event"] = {"start_frame": 33, "start_time": 33 * 1001 / 30000.0}
            lines.append(json.dumps(row, separators=(",", ":")))
    return lines


def _write_raw(folder):
    for name in ("tracking", "events", "metadata", "rosters"):
        os.makedirs(os.path.join(folder, name), exist_ok=True)
    with open(os.path.join(folder, "metadata", GAME_ID + ".json"), "w") as handle:
        json.dump(_metadata(), handle)
    with open(os.path.join(folder, "rosters", GAME_ID + ".json"), "w") as handle:
        json.dump(_roster(), handle)
    with open(os.path.join(folder, "events", GAME_ID + ".json"), "w") as handle:
        json.dump(_events(), handle)
    with bz2.open(os.path.join(folder, "tracking", GAME_ID + ".jsonl.bz2"), "wt") as handle:
        handle.write("\n".join(_tracking_lines()) + "\n")
    return folder


# -- frame mapping --------------------------------------------------------------------------------

def test_video_block_and_frame_of_time():
    # 30 * 25025 / 30000 = 25.025 -> 26. 89 * 25025 / 30000 = 74.24 -> 74.
    assert pff.video_block(30, 89) == (26, 74)
    assert pff.video_block(120, 179) == (101, 149)
    # 0 and 1001 * 25 / 25025 frames: whole numbers on both sides.
    assert pff.video_block(0, 1200) == (0, 1001)
    assert pff.video_frame_of(1.0) == 25
    assert pff.video_frame_of(0.019) == 0 and pff.video_frame_of(0.021) == 1


def test_resample_interpolates_and_never_fills_a_gap():
    # Tracking frames 0..1200 hold their own frame number. Video frame v lies at 1200 v / 1001.
    values = np.arange(1201, dtype=np.float64)
    out = pff.resample(values, 1002, 0)
    assert out[0] == 0.0 and out[1001] == 1200.0
    assert out[1] == pytest.approx(1200.0 / 1001.0)
    # Row 0 at frameNum 10: video frames before 10 * 1001 / 1200 are missing.
    shifted = pff.resample(values, 20, 10)
    assert np.isnan(shifted[:9]).all()
    assert shifted[9] == pytest.approx(9 * 1200.0 / 1001.0 - 10)
    gap = values.copy()
    gap[5] = np.nan
    out = pff.resample(gap, 10, 0)
    # v = 4 lies at 4.795 (frames 4 and 5), v = 5 at 5.994 (frames 5 and 6): both missing.
    assert np.isnan(out[4]) and np.isnan(out[5])
    assert np.isfinite(out[3]) and np.isfinite(out[6])


def test_period_of_line_reads_compact_and_spaced_json():
    assert pff._period_of_line('{"frameNum":5,"period":2,"periodElapsedTime":0.0}') == 2
    assert pff._period_of_line('{"frameNum": 5, "period": 1, "x": 0}') == 1
    assert pff._period_of_line('{"frameNum":5,"period":null,"periodElapsedTime":null}') is None
    # The first match is the top-level field, before the nested lists.
    assert pff._period_of_line('{"period":3,"homePlayers":[{"period":1}]}') == 3
    with pytest.raises(ValueError):
        pff._period_of_line('{"frameNum":5}')


# -- the reference --------------------------------------------------------------------------------

def test_change_points_follow_the_sequences():
    rows = pff.read_events_from_list(_events(), HOME_ID, AWAY_ID)
    points, counts = pff.change_points([r for r in rows if r["period"] == 1])
    frames = [(int(round(t * 25)), code) for t, code in points]
    assert frames == [(25, H), (40, A), (47, H), (50, D), (60, A), (63, D), (65, H), (74, X)]
    assert counts["restarts_without_out"] == 1
    assert counts["teamless_rows_skipped"] == 1
    assert counts["owner_changes"] == 2
    assert pff.sequence_teams(rows)[1] == H and pff.sequence_teams(rows)[3] == H


def test_reference_labels():
    rows = pff.read_events_from_list(_events(), HOME_ID, AWAY_ID)
    labels, _ = pff.reference_labels(rows, {1: (26, 73), 2: (101, 139)}, 140)
    assert labels.tolist() == _expected_labels()


# -- the bundle -----------------------------------------------------------------------------------

def test_build_writes_a_bundle_the_engine_reads(tmp_path):
    raw = _write_raw(str(tmp_path / "raw"))
    out = str(tmp_path / "out")
    bundle, summary = pff.build_match(raw, GAME_ID, out)
    stem = "PFF-" + GAME_ID
    assert bundle["stem"] == stem
    assert bundle["spans"] == {1: (30, 89), 2: (120, 179)}
    # Period 1: tracking 26..74, cut by the END event at v = 74. Period 2: the kickoff event
    # (v = 100) is before the tracking (v = 101), the END event at v = 140 cuts the end.
    assert bundle["blocks"] == {1: (26, 73), 2: (101, 139)}
    assert bundle["n_frames"] == 140
    fields = summary["games_entry"]
    assert fields["halves"] == {1: (27, 74), 2: (102, 140)}
    assert fields["offsets"] == {1: 1, 2: 1}
    assert fields["kickoff"] == {"minute": 0, "second": 1, "frame": 1}
    assert fields["kickoff_2nd"] == {"minute": 0, "second": 4, "frame": 1}
    assert fields["pitch"] == (100.0, 60.0)
    assert fields["assets"]["atd_raw"] == "pff/{}/{}_atd.txt".format(GAME_ID, stem)

    # Slots: home first, sorted by player id. Player 112 never plays. Jersey 99 is dropped.
    assert bundle["slots"] == [(HOME_ID, 101), (HOME_ID, 107), (AWAY_ID, 201), (AWAY_ID, 209)]
    stats = summary["stats"]
    assert stats["unknown_jerseys"] == {"away:99": 2}
    assert stats["duplicate_jersey_entries"] == 1
    assert stats["repeated_frames"] == 1 and stats["repeated_frames_differ"] == 0
    assert stats["event_frames"]["copies"] == 2

    # Ball: x = 0.1 (f - 30) metres on the raw VISIBLE samples, never the smoothed ball (99 m).
    # Frames 40..44 are ESTIMATED: v = 33..37 touch them and are missing.
    s = 30 * 1200.0 / 1001.0
    bx = bundle["bx"]
    assert np.isnan(bx[:26]).all() and np.isnan(bx[74:101]).all()
    assert bx[30] == pytest.approx(((s - 30) * 0.1 + 50.0) / 100.0)
    assert np.isnan(bx[33:38]).all() and np.isfinite(bx[32]) and np.isfinite(bx[38])
    assert bundle["by"][30] == pytest.approx(0.5 - 5.0 / 60.0)
    # Player 107 is missing on frame 50: v = 41 (49.15) and v = 42 (50.35) are missing.
    p7 = bundle["px"][:, 1]
    assert np.isnan(p7[41:43]).all() and np.isfinite(p7[40]) and np.isfinite(p7[43])
    # The duplicate jersey 7 entry on frame 60 is dropped: no 30 m jump near v = 50.
    assert np.nanmax(np.abs(np.diff(p7[26:74]))) < 0.01
    # The ESTIMATED away players are kept.
    assert np.isfinite(bundle["px"][26:74, 3]).all()

    assert bundle["labels"].tolist() == _expected_labels()
    assert stats["side_check"]["agrees"]
    assert stats["direction_check"] == {"1": 1.0, "2": 1.0}
    assert stats["actor_ball"]["median_m"] == pytest.approx(0.0, abs=1e-6)
    assert stats["actor_ball"]["best_shift"] == 0
    assert stats["event_frames"]["start_frame_minus_round_fps_time"]["nonzero"] == 0
    assert stats["out_types"] == {"H": 1, "T": 1}
    states = stats["ball_state_by_label"]
    assert sum(states["alive"].values()) == pytest.approx(1.0)
    assert states["alive"]["estimated"] > 0 and states["dead"]["none"] == 0.0

    # The engine's own parser reads the files.
    meta = data._atd_meta(os.path.join(out, stem + "_atd.xml"), "METRICA-S3")
    assert meta["team_names"] == ("Home FC", "Away & Co")
    assert meta["team_of_slot"].tolist() == [0, 0, 1, 1]
    assert meta["player_channels"] == ("x", "y") and meta["ball_channels"] == ("x", "y")
    arrays = data._parse_atd_raw(os.path.join(out, stem + "_atd.txt"), 4, 140,
                                 meta["player_channels"], meta["ball_channels"])
    assert np.allclose(arrays["bx"], bundle["bx"], equal_nan=True, atol=1e-5)
    with open(os.path.join(out, stem + "_atd.xml")) as handle:
        assert "PFF FC World Cup 2022" in handle.read()
    with open(os.path.join(out, stem + "_reference.csv")) as handle:
        rows = handle.read().splitlines()
    assert rows[0] == "frame,label" and len(rows) == 141 and rows[31] == "30,H"
    assert summary["reference_md5"] == pff.sc.md5(os.path.join(out, stem + "_reference.csv"))


def test_build_is_deterministic(tmp_path):
    raw = _write_raw(str(tmp_path / "raw"))
    pff.build_match(raw, GAME_ID, str(tmp_path / "a"))
    pff.build_match(raw, GAME_ID, str(tmp_path / "b"))
    for suffix in ("_atd.xml", "_atd.txt", "_reference.csv", "_build.json"):
        name = "PFF-" + GAME_ID + suffix
        with open(str(tmp_path / "a" / name), "rb") as one, \
                open(str(tmp_path / "b" / name), "rb") as two:
            assert one.read() == two.read()


def test_wrong_side_is_reported(tmp_path):
    raw = _write_raw(str(tmp_path / "raw"))
    meta = _metadata()
    meta[0]["homeTeamStartLeft"] = False
    with open(os.path.join(raw, "metadata", GAME_ID + ".json"), "w") as handle:
        json.dump(meta, handle)
    _, summary = pff.build_match(raw, GAME_ID, str(tmp_path / "out"))
    assert not summary["stats"]["side_check"]["agrees"]
    assert summary["stats"]["direction_check"] == {"1": 0.0, "2": 0.0}


def test_entry_text_is_valid_python(tmp_path):
    raw = _write_raw(str(tmp_path / "raw"))
    bundle, summary = pff.build_match(raw, GAME_ID, str(tmp_path / "out"))
    text = pff.entry_text(bundle, summary["games_entry"])
    assert max(len(line) for line in text.splitlines()) <= 99
    entry = eval("{" + text + "}")["PFF-" + GAME_ID]
    for field in ("n_video_frames", "fps", "offsets", "halves", "kickoff", "kickoff_2nd",
                  "pitch", "assets"):
        assert entry[field] == summary["games_entry"][field]
    # The entries command gives the same text from the build file.
    with open(str(tmp_path / "out" / ("PFF-" + GAME_ID + "_build.json"))) as handle:
        assert pff.entry_from_build(json.load(handle)) == text


# -- the games.py entries (no data needed) --------------------------------------------------------

PFF_GAMES = sorted(name for name in games.GAMES if name.startswith("PFF-"))
REFS = os.path.expanduser(os.environ.get("MTAH_DATA_DIR", "~/mtah2027/refs"))


@pytest.mark.parametrize("name", PFF_GAMES)
def test_pff_entry_is_consistent(name):
    entry = games.GAMES[name]
    game_id = name[len("PFF-"):]
    assert entry["fps"] == 25 and entry["offsets"] == {1: 1, 2: 1}
    (a1, b1), (a2, b2) = entry["halves"][1], entry["halves"][2]
    assert 1 <= a1 < b1 < a2 < b2 == entry["n_video_frames"]
    assert games.anchor_video_frame(entry["kickoff"]) == a1 - 1
    assert games.anchor_video_frame(entry["kickoff_2nd"]) == a2 - 1
    assert entry["assets"]["atd_meta"] == "pff/{0}/PFF-{0}_atd.xml".format(game_id)
    assert entry["assets"]["atd_raw"] == "pff/{0}/PFF-{0}_atd.txt".format(game_id)


@pytest.mark.parametrize("name", PFF_GAMES)
def test_pff_entry_matches_its_built_bundle(name):
    game_id = name[len("PFF-"):]
    path = os.path.join(REFS, "pff", game_id, name + "_build.json")
    if not os.path.isfile(path):
        pytest.skip("no PFF bundle at {}".format(path))
    with open(path) as handle:
        built = json.load(handle)["games_entry"]
    entry = games.GAMES[name]
    for field in ("n_video_frames", "fps", "kickoff", "kickoff_2nd", "assets"):
        assert entry[field] == built[field], field
    assert entry["pitch"] == tuple(built["pitch"])
    assert entry["halves"] == {int(h): tuple(v) for h, v in built["halves"].items()}
    assert entry["offsets"] == {int(h): v for h, v in built["offsets"].items()}
