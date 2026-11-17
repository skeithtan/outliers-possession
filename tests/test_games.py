"""G1: the engine reads one match of games.py, chosen at run time (PLAN.md 5.5).

These tests need no match data. They read games.py only, or they build small synthetic files in
a temporary folder.
"""

import os

import numpy as np
import pytest

import games
from engine import data, run, submission


def test_resolve_game_default_is_games_game():
    assert data.resolve_game() == games.GAME
    assert data.resolve_game(None) == games.GAME
    assert data.resolve_game("METRICA-S3") == "METRICA-S3"
    with pytest.raises(KeyError):
        data.resolve_game("NO-SUCH-GAME")


def test_game_none_reads_the_same_entry_as_games_game():
    """Without --game the engine reads exactly what it read before G1."""
    assert data.graded_blocks() == data.graded_blocks(games.GAME)
    assert data.graded_blocks() == [(h, a, b) for h, a, b in games.half_video_blocks(games.GAME)]
    assert data.pitch_m() == data.pitch_m(games.GAME)
    assert data.kickoff_frames() == data.kickoff_frames(games.GAME)
    assert data.fps() == games.GAMES[games.GAME]["fps"]
    assert data.frame_count() == games.source_frames(games.DEFAULT_SOURCE, games.GAME)
    assert submission.n_frames_default() == data.frame_count(games.GAME)


def test_another_game_reads_its_own_entry():
    entry = games.GAMES["METRICA-S3"]
    assert data.pitch_m("METRICA-S3") == tuple(float(v) for v in entry["pitch"])
    assert data.graded_blocks("METRICA-S3") == [
        (h, a, b) for h, a, b in games.half_video_blocks("METRICA-S3")]
    assert data.fps("METRICA-S3") == entry["fps"]


def test_run_cli_rejects_an_unknown_game(tmp_path):
    with pytest.raises(SystemExit):
        run.main(["--out", str(tmp_path / "x.csv"), "--game", "NO-SUCH-GAME"])


# -- per-game assets (G1 step 2) ------------------------------------------------------------------

def _fake_game(monkeypatch, assets, name="FAKE"):
    """Add a games.py entry with its own "assets" field, for this test only."""
    entry = dict(games.GAMES["METRICA-S3"], assets=assets, n_video_frames=3)
    monkeypatch.setitem(games.GAMES, name, entry)
    return name


def test_tactical_frame_count_is_the_entry_count():
    for name, entry in games.GAMES.items():
        assert data.frame_count(name) == entry["n_video_frames"]
    # For games.GAME the entry and the organisers' SOURCES table agree.
    assert data.frame_count(games.GAME) == games.source_frames(games.DEFAULT_SOURCE, games.GAME)


def test_entry_without_assets_uses_the_organisers_layout():
    assert data.asset_paths(games.GAME) == games.asset_paths(games.GAME)
    assert data.asset_paths(games.GAME, source="broadcast") == games.asset_paths(
        games.GAME, source="broadcast")
    assert data.has_asset("smart_tagging", games.GAME)


def test_entry_with_assets_reads_its_own_files(tmp_path, monkeypatch):
    game = _fake_game(monkeypatch, {"atd_meta": "g/meta.xml", "atd_raw": "g/raw.txt"})
    paths = data.asset_paths(game, data_dir=str(tmp_path))
    assert paths == {"atd_meta": str(tmp_path / "g" / "meta.xml"),
                     "atd_raw": str(tmp_path / "g" / "raw.txt")}
    assert not data.has_asset("smart_tagging", game)
    assert data.frame_count(game) == 3
    with pytest.raises(KeyError):
        data.asset_paths(game, source="broadcast")
    with pytest.raises(IOError):
        data.require(["atd_meta"], game, data_dir=str(tmp_path))
    with pytest.raises(KeyError):
        data.require(["smart_tagging"], game, data_dir=str(tmp_path))
    (tmp_path / "g").mkdir()
    (tmp_path / "g" / "meta.xml").write_text("<main/>")
    assert data.require(["atd_meta"], game, data_dir=str(tmp_path)) == [paths["atd_meta"]]


def test_cache_name_keeps_the_default_game_file():
    assert data._cache_path("c", games.GAME, "tactical") == os.path.join("c", "tactical.npz")
    assert data._cache_path("c", "METRICA-S3", "tactical") == os.path.join(
        "c", "METRICA-S3_tactical.npz")


# -- synthetic ATD files (G1 steps 3 and 4) -------------------------------------------------------

def _write_lines(path, lines):
    path.write_text("".join(line + "\n" for line in lines))
    return str(path)


def _metadata(path, players, specs, channels=("x", "y"), ball=("x", "y")):
    """Write a small FIFA/EPTS ATD XML and return its path.

    `players` is [(player id, team id)] in <Player> order. `specs` is one list of player ids per
    DataFormatSpecification, in the order of the raw line.
    """
    teams = "".join('<Team id="{0}"><Name>Team {0}</Name></Team>'.format(t) for t in ("TA", "TB"))
    plist = "".join('<Player id="{}" teamId="{}"/>'.format(p, t) for p, t in players)
    pchan = "".join('<PlayerChannel channelId="{1}" id="{0}_{1}" playerId="{0}"/>'.format(p, c)
                    for p, _ in players for c in channels)
    blocks = ""
    for order in specs:
        refs = "".join('<PlayerChannelRef playerChannelId="{}_{}"/>'.format(p, c)
                       for p in order for c in channels)
        brefs = "".join('<BallChannelRef channelId="{}"/>'.format(c) for c in ball)
        blocks += "<DataFormatSpecification>{}{}</DataFormatSpecification>".format(refs, brefs)
    xml = ("<main><Metadata><GlobalConfig><FrameRate>25</FrameRate></GlobalConfig>"
           "<Teams>{}</Teams><Players>{}</Players><Devices>{}</Devices></Metadata>"
           "<DataFormatSpecifications>{}</DataFormatSpecifications></main>").format(
               teams, plist, pchan, blocks)
    path.write_text(xml)
    return str(path)


def test_parse_atd_raw_with_two_values_per_object(tmp_path):
    lines = ["1:0.1,0.2;NaN,NaN:0.5,0.5",
             "2:0.11,0.21;0.9,0.8:NaN,NaN",
             "3:NaN,NaN;0.91,0.81:0.52,0.48"]
    out = data._parse_atd_raw(_write_lines(tmp_path / "atd.txt", lines), n_slots=2, n_frames=3,
                              player_channels=("x", "y"), ball_channels=("x", "y"))
    assert out["px"].shape == (3, 2) and out["px"].dtype == np.float32
    assert out["px"][0, 0] == np.float32(0.1) and out["py"][1, 1] == np.float32(0.8)
    assert out["ps"].shape == (3, 2) and np.isnan(out["ps"]).all()
    assert out["bx"][2] == np.float32(0.52) and np.isnan(out["by"][1])
    # Three values on a line that declares two per object is a layout error.
    bad = ["1:0.1,0.2,1.0;NaN,NaN,NaN:0.5,0.5,NaN"]
    with pytest.raises(ValueError):
        data._parse_atd_raw(_write_lines(tmp_path / "bad.txt", bad), n_slots=2, n_frames=1,
                            player_channels=("x", "y"), ball_channels=("x", "y"))


def test_parse_atd_raw_follows_the_channel_order(tmp_path):
    lines = ["1:2.5,0.1,0.2;NaN,NaN,NaN:0.5,0.6"]
    out = data._parse_atd_raw(_write_lines(tmp_path / "atd.txt", lines), n_slots=2, n_frames=1,
                              player_channels=("s", "x", "y"), ball_channels=("y", "x"))
    assert out["ps"][0, 0] == np.float32(2.5) and out["px"][0, 0] == np.float32(0.1)
    assert out["bx"][0] == np.float32(0.6) and out["by"][0] == np.float32(0.5)


def test_atd_meta_reads_the_channels_of_the_specification(tmp_path, monkeypatch):
    players = [("P1", "TA"), ("P2", "TB")]
    path = _metadata(tmp_path / "meta.xml", players, [["P1", "P2"]])
    meta = data._atd_meta(path, "METRICA-S3")
    assert meta["player_channels"] == ("x", "y") and meta["ball_channels"] == ("x", "y")
    assert meta["team_names"] == ("Team TA", "Team TB")
    assert meta["team_of_slot"].tolist() == [data.HOME, data.AWAY]
    path3 = _metadata(tmp_path / "meta3.xml", players, [["P1", "P2"]], channels=("x", "y", "s"),
                      ball=("x", "y", "s"))
    assert data._atd_meta(path3, "METRICA-S3")["player_channels"] == ("x", "y", "s")
    nox = _metadata(tmp_path / "nox.xml", players, [["P1", "P2"]], channels=("y", "s"))
    with pytest.raises(ValueError):
        data._atd_meta(nox, "METRICA-S3")


def test_slot_table_from_the_first_of_two_specifications(tmp_path):
    """Five players, four slots: a substitute (P5 for P2) enters in the second block."""
    players = [("P1", "TA"), ("P2", "TA"), ("P3", "TB"), ("P4", "TB"), ("P5", "TA")]
    first, second = ["P3", "P1", "P4", "P2"], ["P3", "P1", "P4", "P5"]
    meta = data._atd_meta(_metadata(tmp_path / "meta.xml", players, [first, second]),
                          "METRICA-S3")
    # The slot order is the order of the first block, not the <Player> order.
    assert meta["team_of_slot"].tolist() == [data.AWAY, data.HOME, data.AWAY, data.HOME]
    assert meta["player_channels"] == ("x", "y")


def test_specifications_must_keep_each_team_in_its_slots(tmp_path):
    players = [("P1", "TA"), ("P2", "TA"), ("P3", "TB"), ("P4", "TB"), ("P5", "TB")]
    swapped = _metadata(tmp_path / "meta.xml", players,
                        [["P1", "P2", "P3", "P4"], ["P1", "P5", "P3", "P4"]])
    with pytest.raises(ValueError):
        data._atd_meta(swapped, "METRICA-S3")
    other_count = _metadata(tmp_path / "count.xml", players,
                            [["P1", "P2", "P3", "P4"], ["P1", "P2", "P3"]])
    with pytest.raises(ValueError):
        data._atd_meta(other_count, "METRICA-S3")


# -- optional Smart Tagging (G1 step 5) -----------------------------------------------------------

TAG_COLUMNS = ["id", "code", "team", "type", "half", "start_s", "end_s", "start_frame",
               "end_frame"]


def test_load_tagging_without_a_tagging_asset_is_empty(tmp_path, monkeypatch):
    # The fake entry names files that do not exist: load_tagging must not open any of them.
    game = _fake_game(monkeypatch, {"atd_meta": "g/meta.xml", "atd_raw": "g/raw.txt"})
    tags = data.load_tagging(game, data_dir=str(tmp_path))
    assert len(tags) == 0
    assert list(tags.columns) == TAG_COLUMNS
    for column in ("id", "half", "start_frame", "end_frame"):
        assert tags[column].dtype == np.int64
    for column in ("start_s", "end_s"):
        assert tags[column].dtype == np.float64
    assert tags["code"].str.upper().tolist() == []


def test_load_tagging_with_a_missing_tagging_file_fails(tmp_path, monkeypatch):
    game = _fake_game(monkeypatch, {"atd_meta": "g/meta.xml", "smart_tagging": "g/tags.xml"})
    with pytest.raises(IOError):
        data.load_tagging(game, data_dir=str(tmp_path))


# -- optional half-1 kickoff anchor (G1 finding 6) ------------------------------------------------

def test_kickoff_anchor_field_is_opt_in(monkeypatch):
    assert data.KICKOFF_ANCHOR_FIELD not in games.GAMES[games.GAME]
    # Without the field, half 1 starts at the first graded frame, as before G1.
    assert data.kickoff_frames(games.GAME)[1] == data.graded_blocks(games.GAME)[0][1]
    entry = dict(games.GAMES["METRICA-S3"], **{data.KICKOFF_ANCHOR_FIELD: False})
    monkeypatch.setitem(games.GAMES, "FAKE", entry)
    assert data.kickoff_frames("FAKE")[1] == data.graded_blocks("FAKE")[0][1]
    entry = dict(entry, kickoff={"minute": 0, "second": 14, "frame": 10},
                 **{data.KICKOFF_ANCHOR_FIELD: True})
    monkeypatch.setitem(games.GAMES, "FAKE", entry)
    kick = data.kickoff_frames("FAKE")
    assert kick[1] == 14 * 25 + 10
    assert kick[2] == games.anchor_video_frame(entry["kickoff_2nd"])


# -- player speed for an ATD without a speed channel (G1 finding 7) -------------------------------

def test_speed_from_positions_fills_only_an_atd_without_speed():
    from engine import tracks

    pitch = (100.0, 50.0)
    n = 6
    # Slot 0 runs along x at 0.05 pitch lengths per frame: 5 m per frame, 125 m/s at 25 fps.
    px = np.stack([0.1 + 0.05 * np.arange(n), np.full(n, 0.5)], axis=1).astype(np.float32)
    py = np.full((n, 2), 0.5, dtype=np.float32)
    px[3, 1] = np.nan
    py[3, 1] = np.nan
    tactical = {"px": px, "py": py, "ps": np.full((n, 2), np.nan, dtype=np.float32),
                "fps": 25, "game": "METRICA-S3"}
    out = tracks.speed_from_positions(tactical, pitch)
    assert out is not tactical and out["ps"].dtype == np.float32
    assert np.allclose(out["ps"][:, 0], 125.0)
    assert np.isnan(out["ps"][3, 1]) and out["ps"][0, 1] == 0.0
    assert np.isnan(tactical["ps"]).all()      # the input is not changed
    with_speed = dict(tactical, ps=np.where(np.arange(2) == 0, 1.0, np.nan) * np.ones((n, 1)))
    assert tracks.speed_from_positions(with_speed, pitch) is with_speed
