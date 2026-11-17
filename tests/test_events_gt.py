"""tools/events_gt.py: the frame-level reference from Metrica events (Phase 5, V1).

Synthetic event lists only. The tests need no match data. Event frames are frame ids. With an
offset of 1 the video frame is the frame id minus 1.
"""

import json

import numpy as np
import pytest

from tools import events_gt as eg

H, A, D, X = eg.H, eg.A, eg.D, eg.X
FPS = 25.0
OFFSETS = {1: 1, 2: 1}


def ev(team, kind, start, end=None, subtypes=(), period=1):
    """One raw Metrica event."""
    subs = [{"name": s} for s in subtypes]
    return {"team": {"name": team}, "type": {"name": kind},
            "subtypes": subs[0] if len(subs) == 1 else (subs or None),
            "start": {"frame": start}, "end": {"frame": start if end is None else end},
            "period": period}


def first_half():
    """Kick-off, pass chain, interception, ball out and throw-in, foul and free kick, end."""
    return [
        ev("Team A", "SET PIECE", 101, subtypes=["KICK OFF"]),
        ev("Team A", "PASS", 101, 120),
        ev("Team A", "CARRY", 120, 150),
        ev("Team A", "PASS", 150, 170),
        ev("Team A", "BALL LOST", 170, 190, ["INTERCEPTION"]),
        ev("Team B", "RECOVERY", 190, subtypes=["INTERCEPTION"]),
        ev("Team B", "CARRY", 190, 220),
        ev("Team B", "BALL OUT", 220, 240),
        ev("Team A", "SET PIECE", 300, subtypes=["THROW IN"]),
        ev("Team A", "PASS", 300, 320),
        ev("Team B", "CHALLENGE", 330, subtypes=["GROUND", "WON"]),
        ev("Team A", "CHALLENGE", 330, subtypes=["GROUND", "LOST"]),
        ev("Team A", "CARRY", 335, 350),
        ev("Team B", "CHALLENGE", 360, subtypes=["TACKLE", "FAULT", "LOST"]),
        ev("Team A", "FAULT RECEIVED", 360),
        ev("Team A", "SET PIECE", 420, subtypes=["FREE KICK", "INDIRECT"]),
        ev("Team A", "PASS", 420, 440),
        ev("Team B", "RECOVERY", 440),
        ev("Team B", "PASS", 440, 460),
        ev("Team B", "BALL LOST", 470, 480, ["END HALF"]),
    ]


def derive(raw, n_frames=600, home="Team A"):
    return eg.derive(eg.normalise(raw), home, n_frames, OFFSETS, FPS)


def expected_first_half(n_frames=600):
    want = np.full(n_frames, X, dtype=np.int8)
    want[100:189] = H     # kick-off and pass chain (video = frame id - 1)
    want[189:239] = A     # the recovery of Team B, held until the ball goes out
    want[239:299] = D     # BALL OUT end frame to the throw-in
    want[299:359] = H     # the challenge at 330 does not change the owner
    want[359:419] = D     # the foul to the free kick
    want[419:439] = H
    want[439:479] = A     # the block ends at the END HALF end frame, half-open
    return want


def test_first_half_labels():
    codes, info = derive(first_half())
    np.testing.assert_array_equal(codes, expected_first_half())
    assert info["blocks"] == [(1, 100, 479)]
    assert info["dead"] == [(239, 299), (359, 419)]
    assert info["onset_kinds"] == {"stop": 3, "no_stop": 0}   # BALL OUT, CHALLENGE, FAULT
    assert info["unknown_owner_frames"] == 0


def test_team_mapping_follows_home_team():
    codes, _ = derive(first_half(), home="Team B")
    want = expected_first_half()
    swapped = want.copy()
    swapped[want == H] = A
    swapped[want == A] = H
    np.testing.assert_array_equal(codes, swapped)


def test_second_period_has_its_own_block():
    raw = first_half() + [
        ev("Team B", "SET PIECE", 701, subtypes=["KICK OFF"], period=2),
        ev("Team B", "PASS", 701, 720, period=2),
        ev("Team B", "BALL LOST", 760, 770, ["END HALF"], period=2),
    ]
    codes, info = derive(raw, n_frames=900)
    assert info["blocks"] == [(1, 100, 479), (2, 700, 769)]
    assert (codes[479:700] == X).all()
    assert (codes[700:769] == A).all()
    assert (codes[769:] == X).all()


def test_goal_stops_until_kick_off():
    raw = [
        ev("Team A", "SET PIECE", 101, subtypes=["KICK OFF"]),
        ev("Team A", "PASS", 101, 130),
        ev("Team A", "SHOT", 130, 150, ["ON TARGET", "GOAL"]),
        ev("Team B", "SET PIECE", 400, subtypes=["KICK OFF"]),
        ev("Team B", "PASS", 400, 420),
        ev("Team B", "BALL LOST", 450, 460, ["END HALF"]),
    ]
    codes, info = derive(raw)
    assert info["blocks"] == [(1, 100, 459)]   # the first KICK OFF opens the block
    assert (codes[100:149] == H).all()
    assert (codes[149:399] == D).all()
    assert (codes[399:459] == A).all()


def test_goal_kick_subtype_is_a_restart():
    raw = [
        ev("Team A", "SET PIECE", 101, subtypes=["KICK OFF"]),
        ev("Team A", "SHOT", 120, 140, ["OFF TARGET", "OUT"]),
        ev("Team B", "BALL LOST", 300, 340, ["GOAL KICK"]),
        ev("Team A", "RECOVERY", 340),
        ev("Team A", "BALL LOST", 380, 390, ["END HALF"]),
    ]
    codes, _ = derive(raw)
    assert (codes[139:299] == D).all()
    assert (codes[299:339] == A).all()       # the restart names the owner
    assert (codes[339:389] == H).all()


def test_pause_after_a_stop_is_a_restart():
    # A goal kick without the GOAL KICK subtype: the goalkeeper's RECOVERY 6 s after the out.
    raw = [
        ev("Team A", "SET PIECE", 101, subtypes=["KICK OFF"]),
        ev("Team A", "SHOT", 120, 140, ["OFF TARGET", "OUT"]),
        ev("Team B", "RECOVERY", 290),
        ev("Team B", "PASS", 290, 320),
        ev("Team B", "CHALLENGE", 400, subtypes=["GROUND", "FAULT", "WON"]),
        ev("Team B", "SET PIECE", 500, subtypes=["FREE KICK", "INDIRECT"]),
        ev("Team B", "BALL LOST", 550, 560, ["END HALF"]),
    ]
    codes, info = derive(raw)
    assert info["pause_restarts"] == 1
    assert (codes[139:289] == D).all()
    assert (codes[289:399] == A).all()
    assert (codes[399:499] == D).all()


def test_play_on_after_a_stop_keeps_dead_until_the_set_piece():
    # An offside flagged late: play goes on for a moment, the free kick is the restart.
    raw = [
        ev("Team A", "SET PIECE", 101, subtypes=["KICK OFF"]),
        ev("Team A", "PASS", 110, 140, ["OFFSIDE"]),
        ev("Team A", "CARRY", 140, 160),
        ev("Team B", "SET PIECE", 300, subtypes=["FREE KICK", "INDIRECT"]),
        ev("Team B", "BALL LOST", 350, 360, ["END HALF"]),
    ]
    codes, info = derive(raw)
    assert info["pause_restarts"] == 0
    assert (codes[100:139] == H).all()
    assert (codes[139:299] == D).all()
    assert (codes[299:359] == A).all()


def test_set_piece_without_a_stop_opens_dead_at_the_previous_event():
    # A throw-in that must be retaken: no stopping event between the two throw-ins.
    raw = [
        ev("Team A", "SET PIECE", 101, subtypes=["KICK OFF"]),
        ev("Team A", "BALL OUT", 110, 130),
        ev("Team B", "SET PIECE", 200, subtypes=["THROW IN", "RETAKEN"]),
        ev("Team B", "BALL LOST", 200, 215, ["FORCED"]),
        ev("Team B", "SET PIECE", 300, subtypes=["THROW IN"]),
        ev("Team B", "BALL LOST", 350, 360, ["END HALF"]),
    ]
    codes, info = derive(raw)
    assert info["onset_kinds"] == {"stop": 1, "no_stop": 1}
    assert (codes[129:199] == D).all()
    assert (codes[199:214] == A).all()
    assert (codes[214:299] == D).all()
    assert (codes[299:359] == A).all()


@pytest.mark.parametrize("event", [
    ev("Team A", "RECOVERY", 150, subtypes=["GOAL"]),
    ev("Team A", "BALL LOST", 150, 160, ["REFEREE HIT"]),
    ev("Team A", "CHALLENGE", 150, subtypes=["GROUND", "ADVANTAGE", "WON"]),
    ev("Team A", "CARD", 150, subtypes=["YELLOW"]),
    ev("Team A", "SHOT", 150, 160, ["WOODWORK", "OFF TARGET"]),
])
def test_events_that_do_not_stop_play(event):
    assert not eg.is_stop(eg.normalise([event])[0])


@pytest.mark.parametrize("event", [
    ev("Team A", "BALL OUT", 150, 160),
    ev("Team A", "FAULT RECEIVED", 150),
    ev("Team A", "CHALLENGE", 150, subtypes=["AERIAL", "FAULT", "WON"]),
    ev("Team A", "BALL LOST", 150, subtypes=["OFFSIDE"]),
    ev("Team A", "BALL LOST", 150, subtypes=["HAND BALL"]),
    ev("Team A", "PASS", 150, 170, ["DEEP BALL", "OFFSIDE"]),
    ev("Team A", "SHOT", 150, 160, ["OFF TARGET", "OUT"]),
    ev("Team A", "SHOT", 150, 160, ["ON TARGET", "GOAL"]),
])
def test_events_that_stop_play(event):
    assert eg.is_stop(eg.normalise([event])[0])


def test_period_without_kick_off_raises():
    with pytest.raises(ValueError):
        derive([ev("Team A", "PASS", 101, 120)])


def test_reference_stats():
    codes, _ = derive(first_half())
    stats = eg.reference_stats(codes, FPS)
    assert stats["graded_frames"] == 379
    assert stats["possessions"] == 5          # H A | H | H A
    assert stats["d_segments"] == 2
    assert stats["d_share"] == pytest.approx(120 / 379.0)


def test_main_writes_one_row_per_video_frame(tmp_path):
    path = tmp_path / "events.json"
    path.write_text(json.dumps({"data": first_half()}))
    out = tmp_path / "ref.csv"
    assert eg.main(["--game", "METRICA-S3", "--events", str(path), "--out", str(out),
                    "--home-team", "Team A"]) == 0
    lines = out.read_text().splitlines()
    assert lines[0] == "frame,label"
    assert len(lines) == 1 + 143761
    assert lines[1 + 100] == "100,H" and lines[1 + 239] == "239,D" and lines[1 + 479] == "479,X"


# -- the CSV event format of Sample Games 1 and 2 (P9-A) ------------------------------------------

CSV_HEADER = ("Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],"
              "From,To,Start X,Start Y,End X,End Y")


def csv_text(raw, rename=None):
    """The raw JSON-style events above as a Metrica events CSV (Sample Games 1 and 2)."""
    rename = rename or {}
    lines = [CSV_HEADER]
    for e in raw:
        subs = e["subtypes"]
        if subs is None:
            names = []
        elif isinstance(subs, dict):
            names = [subs["name"]]
        else:
            names = [s["name"] for s in subs]
        start, end = e["start"]["frame"], e["end"]["frame"]
        lines.append(",".join([rename.get(e["team"]["name"], e["team"]["name"]), e["type"]["name"],
                               "-".join(names), str(e["period"]), str(start),
                               "{:.2f}".format(start * 0.04), str(end), "{:.2f}".format(end * 0.04),
                               "Player1", "", "0.5", "0.5", "NaN", "NaN"]))
    return "\n".join(lines) + "\n"


def test_csv_subtypes_split_at_the_hyphen():
    assert eg.csv_subtypes("HEAD-ON TARGET-GOAL") == ("HEAD", "ON TARGET", "GOAL")
    assert eg.csv_subtypes("OFF TARGET") == ("OFF TARGET",)
    assert eg.csv_subtypes("") == ()
    assert eg.csv_subtypes(None) == ()


def test_csv_and_json_readers_give_the_same_reference(tmp_path):
    raw = first_half() + [
        ev("Team B", "SET PIECE", 701, subtypes=["KICK OFF"], period=2),
        ev("Team B", "PASS", 701, 720, period=2),
        ev("Team A", "SHOT", 730, 740, ["HEAD", "ON TARGET", "GOAL"], period=2),
        ev("Team B", "SET PIECE", 800, subtypes=["KICK OFF"], period=2),
        ev("Team B", "BALL OUT", 820, 830, period=2),
    ]
    path = tmp_path / "events.csv"
    path.write_text(csv_text(raw))
    from_csv = eg.load_events_csv(str(path))
    from_json = eg.normalise(raw)
    assert from_csv == from_json
    codes, info = eg.derive(from_csv, "Team A", 900, OFFSETS, FPS)
    np.testing.assert_array_equal(codes[:700], expected_first_half(700))
    # Period 2 has no END HALF: the block ends at the end frame of its last event.
    assert info["blocks"] == [(1, 100, 479), (2, 700, 829)]
    assert (codes[739:799] == D).all() and (codes[799:829] == A).all()


def test_csv_end_frame_before_start_takes_the_start(tmp_path):
    raw = [ev("Away", "SET PIECE", 1, 0, ["KICK OFF"]), ev("Away", "PASS", 1, 3)]
    path = tmp_path / "events.csv"
    path.write_text(csv_text(raw))
    events = eg.load_events_csv(str(path))
    assert [(e["start"], e["end"]) for e in events] == [(1, 1), (1, 3)]


def test_csv_without_a_needed_column_raises(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text("Team,Type,Subtype,Period,Start Frame\nHome,PASS,,1,10\n")
    with pytest.raises(ValueError, match="End Frame"):
        eg.load_events_csv(str(path))


def test_csv_row_without_frames_raises():
    with pytest.raises(ValueError):
        eg.normalise_csv([{"Team": "Home", "Type": "PASS", "Subtype": "", "Period": "1",
                           "Start Frame": "", "End Frame": "4"}])


def test_load_any_picks_the_reader_from_the_extension(tmp_path):
    raw = first_half()
    (tmp_path / "e.csv").write_text(csv_text(raw))
    (tmp_path / "e.json").write_text(json.dumps({"data": raw}))
    assert eg.load_any(str(tmp_path / "e.csv")) == eg.load_any(str(tmp_path / "e.json"))
    assert eg.load_any(str(tmp_path / "e.csv"), "csv") == eg.normalise(raw)
    with pytest.raises(ValueError):
        eg.load_any(str(tmp_path / "e.csv"), "xml")


def test_main_reads_a_csv_for_a_sample_game_1_entry(tmp_path):
    path = tmp_path / "Sample_Game_1_RawEventsData.csv"
    path.write_text(csv_text(first_half(), rename={"Team A": "Home", "Team B": "Away"}))
    out = tmp_path / "ref.csv"
    assert eg.main(["--game", "METRICA-S1", "--events", str(path), "--out", str(out),
                    "--home-team", "Home"]) == 0
    lines = out.read_text().splitlines()
    assert len(lines) == 1 + 145006
    assert lines[1 + 100] == "100,H" and lines[1 + 189] == "189,A" and lines[1 + 239] == "239,D"
