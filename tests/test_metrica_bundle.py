"""tools/metrica_bundle.py: Metrica Sample Game 1 and 2 tracking CSVs to an ATD bundle (P9-A).

The tests write a small synthetic CSV pair in a temporary folder. They need no match data. One
test reads the real sample files. It runs only when METRICA_SAMPLE_DIR names the data/ folder of
the Metrica sample-data clone.
"""

import os

import numpy as np
import pytest

import games
from engine import data
from tools import metrica_bundle as mb

NAN = float("nan")
# Six frames, two periods. Home has two players: H11 on every frame, H12 from frame 4 on (a
# substitute). Away has one player. The ball is missing on frame 3 and beyond the line on 5.
HOME_PLAYERS = [
    [(0.1, 0.5), (NAN, NAN)],
    [(0.11, 0.51), (NAN, NAN)],
    [(0.12, 0.52), (NAN, NAN)],
    [(0.13, 0.53), (0.4, 0.3)],
    [(0.14, 0.54), (0.41, 0.31)],
    [(0.15, 0.55), (0.42, 0.32)],
]
AWAY_PLAYERS = [[(0.9, 0.5)], [(0.89, 0.49)], [(0.88, 0.48)], [(0.87, 0.47)], [(0.86, 0.46)],
                [(0.85, 0.45)]]
BALL = [(0.5, 0.5), (0.52, 0.5), (NAN, NAN), (0.6, 0.4), (1.02, -0.01), (0.7, 0.3)]
PERIODS = [1, 1, 1, 2, 2, 2]


def _cell(value):
    return "NaN" if value != value else repr(value)


def write_team(path, team, numbers, players, ball=BALL, periods=PERIODS, frames=None,
               times=None):
    """Write one per-team tracking CSV in the Metrica Sample Game 1/2 layout."""
    n_players = len(numbers)
    head = [["", "", ""], ["", "", ""], ["Period", "Frame", "Time [s]"]]
    for n in numbers:
        head[0] += [team, ""]
        head[1] += [n, ""]
        head[2] += ["Player" + n, ""]
    head[0] += ["", ""]
    head[1] += ["", ""]
    head[2] += ["Ball", ""]
    rows = [",".join(cells) for cells in head]
    frames = frames or list(range(1, len(players) + 1))
    times = times or [f * 0.04 for f in frames]
    for k, frame in enumerate(frames):
        cells = [str(periods[k]), str(frame), "{:.2f}".format(times[k])]
        for j in range(n_players):
            cells += [_cell(players[k][j][0]), _cell(players[k][j][1])]
        cells += [_cell(ball[k][0]), _cell(ball[k][1])]
        rows.append(",".join(cells))
    path.write_text("\n".join(rows) + "\n")
    return str(path)


def pair(tmp_path, **away_kwargs):
    home = write_team(tmp_path / "g_Home.csv", "Home", ["11", "12"], HOME_PLAYERS)
    away = write_team(tmp_path / "g_Away.csv", "Away", ["25"], AWAY_PLAYERS, **away_kwargs)
    return home, away


def test_read_team_csv(tmp_path):
    home, _ = pair(tmp_path)
    table = mb.read_team_csv(home)
    assert table["team"] == "Home"
    assert table["numbers"] == ["11", "12"]
    assert table["names"] == ["Player11", "Player12"]
    assert table["frame"].tolist() == [1, 2, 3, 4, 5, 6]
    assert table["period"].tolist() == PERIODS
    assert table["players"].shape == (6, 2, 2)
    assert np.isnan(table["players"][:3, 1]).all()
    np.testing.assert_array_equal(table["ball"][4], [1.02, -0.01])


def test_bundle_loads_through_the_engine_data_layer(tmp_path, monkeypatch):
    home, away = pair(tmp_path)
    summary = mb.convert(home, away, str(tmp_path / "out"), "g")
    assert summary["n_frames"] == 6 and summary["fps"] == 25
    assert summary["periods"] == {1: (1, 3), 2: (4, 6)}
    assert summary["slots"] == {"home": 2, "away": 1}
    assert summary["on_pitch_per_frame"] == {"home": [1, 2], "away": [1]}
    entry = dict(games.GAMES["METRICA-S1"], n_video_frames=6, halves={1: (1, 3), 2: (4, 6)},
                 assets={"atd_meta": "out/g_atd.xml", "atd_raw": "out/g_atd.txt"})
    monkeypatch.setitem(games.GAMES, "FAKE-METRICA", entry)
    out = data.load_tactical("FAKE-METRICA", data_dir=str(tmp_path), use_cache=False)
    assert out["team_names"] == ("Home", "Away")
    assert out["team_of_slot"].tolist() == [0, 0, 1]
    assert out["n_frames"] == 6 and out["n_slots"] == 3
    players = np.concatenate([np.array(HOME_PLAYERS), np.array(AWAY_PLAYERS)], axis=1)
    np.testing.assert_array_equal(out["px"], players[:, :, 0].astype(np.float32))
    np.testing.assert_array_equal(out["py"], players[:, :, 1].astype(np.float32))
    np.testing.assert_array_equal(out["bx"], np.array(BALL)[:, 0].astype(np.float32))
    np.testing.assert_array_equal(out["by"], np.array(BALL)[:, 1].astype(np.float32))
    assert np.isnan(out["ps"]).all()     # no speed channel: engine.tracks derives the speed


def test_raw_line_layout(tmp_path):
    home, away = pair(tmp_path)
    mb.convert(home, away, str(tmp_path), "g")
    lines = (tmp_path / "g_atd.txt").read_text().splitlines()
    assert len(lines) == 6
    assert lines[0] == "1:0.10000,0.50000;NaN,NaN;0.90000,0.50000:0.50000,0.50000"
    assert lines[2].endswith(":NaN,NaN")
    assert lines[4].endswith(":1.02000,-0.01000")


def test_conversion_is_deterministic(tmp_path):
    home, away = pair(tmp_path)
    mb.convert(home, away, str(tmp_path / "a"), "g")
    mb.convert(home, away, str(tmp_path / "b"), "g")
    for suffix in ("_atd.txt", "_atd.xml"):
        assert (tmp_path / "a" / ("g" + suffix)).read_bytes() == (
            tmp_path / "b" / ("g" + suffix)).read_bytes()


def test_metadata_has_the_period_blocks_and_the_field(tmp_path):
    home, away = pair(tmp_path)
    mb.convert(home, away, str(tmp_path), "g")
    xml = (tmp_path / "g_atd.xml").read_text()
    assert "<Name>first_half_start</Name><Value>1</Value>" in xml
    assert "<Name>second_half_end</Name><Value>6</Value>" in xml
    assert "<FrameRate>25</FrameRate>" in xml
    assert "<Width>105</Width><Height>68</Height>" in xml
    assert '<Team id="HOME"><Name>Home</Name></Team>' in xml


def test_metre_round_trip_is_exact_at_the_written_precision():
    values = np.array([[0.00082, 0.48238], [1.05996, -0.03991], [NAN, NAN]])
    back = mb.to_normalised(mb.to_metres(values, mb.DEFAULT_PITCH), mb.DEFAULT_PITCH)
    mb.check_round_trip(values, back)
    np.testing.assert_allclose(mb.to_metres(values, (105.0, 68.0))[0], [0.0861, 32.80184])


def test_different_ball_columns_are_refused(tmp_path):
    other = [list(b) for b in BALL]
    other[1] = [0.53, 0.5]
    home, away = pair(tmp_path, ball=[tuple(b) for b in other])
    with pytest.raises(ValueError, match="ball"):
        mb.convert(home, away, str(tmp_path), "g")


def test_a_frame_gap_is_refused(tmp_path):
    home = write_team(tmp_path / "h.csv", "Home", ["11", "12"], HOME_PLAYERS,
                      frames=[1, 2, 3, 5, 6, 7])
    away = write_team(tmp_path / "a.csv", "Away", ["25"], AWAY_PLAYERS,
                      frames=[1, 2, 3, 5, 6, 7])
    with pytest.raises(ValueError, match="Frame"):
        mb.convert(home, away, str(tmp_path), "g")


def test_periods_must_be_contiguous_blocks():
    with pytest.raises(ValueError, match="periods"):
        mb.period_blocks(np.array([1, 1, 2, 1]), np.arange(1, 5))
    assert mb.period_blocks(np.array([1, 2, 2]), np.arange(1, 4)) == {1: (1, 1), 2: (2, 3)}


def test_time_column_must_match_the_frame_rate():
    frames = np.arange(1, 7)
    assert mb.frame_rate(frames, frames * 0.04) == 25
    with pytest.raises(ValueError, match="Time"):
        mb.frame_rate(frames, np.array([0.04, 0.08, 0.12, 0.2, 0.24, 0.28]))


def test_team_labels_must_be_home_then_away(tmp_path):
    home = write_team(tmp_path / "h.csv", "Away", ["11", "12"], HOME_PLAYERS)
    away = write_team(tmp_path / "a.csv", "Away", ["25"], AWAY_PLAYERS)
    with pytest.raises(ValueError, match="team labels"):
        mb.convert(home, away, str(tmp_path), "g")


def test_bad_header_is_refused(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b\nc,d\nPeriod,Frame,Seconds,Ball,\n1,1,0.04,0.5,0.5\n")
    with pytest.raises(ValueError, match="row 3"):
        mb.read_team_csv(str(path))


# -- the games.py entries -------------------------------------------------------------------------

@pytest.mark.parametrize("game", ["METRICA-S1", "METRICA-S2"])
def test_games_entries_are_consistent(game):
    entry = games.GAMES[game]
    assert entry["offsets"] == {1: 1, 2: 1}
    (a1, b1), (a2, b2) = entry["halves"][1], entry["halves"][2]
    assert a1 == 1 and a2 == b1 + 1 and b2 == entry["n_video_frames"]
    blocks = data.graded_blocks(game)
    kick = data.kickoff_frames(game)
    assert blocks[0][1] <= kick[1] <= blocks[0][2]
    assert blocks[1][1] <= kick[2] <= blocks[1][2]
    assert sorted(entry["assets"]) == ["atd_meta", "atd_raw"]
    assert all(not os.path.isabs(p) for p in entry["assets"].values())
    assert entry["pitch"] == mb.DEFAULT_PITCH


SAMPLE_DIR = os.environ.get("METRICA_SAMPLE_DIR")


@pytest.mark.slow
@pytest.mark.skipif(not SAMPLE_DIR or not os.path.isdir(SAMPLE_DIR),
                    reason="METRICA_SAMPLE_DIR does not name the sample-data data/ folder")
@pytest.mark.parametrize("number", [1, 2])
def test_games_entries_match_the_sample_files(number):
    stem = "Sample_Game_{}".format(number)
    path = os.path.join(SAMPLE_DIR, stem, stem + "_RawTrackingData_Home_Team.csv")
    if not os.path.isfile(path):
        pytest.skip("{} not present".format(path))
    table = mb.read_team_csv(path)
    entry = games.GAMES["METRICA-S{}".format(number)]
    assert len(table["frame"]) == entry["n_video_frames"]
    assert mb.period_blocks(table["period"], table["frame"]) == entry["halves"]
    assert mb.frame_rate(table["frame"], table["time"]) == entry["fps"]
