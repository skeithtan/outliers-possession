"""tools/dfl_bundle.py: Sportec Open DFL match to an ATD-style bundle and a reference (P9-A).

The synthetic tests write a few frames of DFL-like XML into a temporary folder. They need no match
data. The last tests compare games.py with the built bundles and skip when the bundles are absent.
"""

import json
import os

import numpy as np
import pytest

import games
from engine import data
from tools import dfl_bundle as dfl

H, A, D, X = dfl.H, dfl.A, dfl.D, dfl.X
GAME = "DFL-T00001"


def _matchinfo(path, first_half_ms=200, second_half_ms=120):
    """A match information XML. The guest team comes first, as in some DFL files."""
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?><PutDataRequest><MatchInformation>'
        '<General MatchId="DFL-MAT-T00001" MatchTitle="Home FC:Guest FC" Result="1:0"/>'
        '<Environment PitchX="105.00" PitchY="68.00"/>'
        '<Teams>'
        '<Team TeamId="CLU-G" TeamName="Guest FC" Role="guest"><Players>'
        '<Player PersonId="Q1"/></Players></Team>'
        '<Team TeamId="CLU-H" TeamName="Home FC" Role="home"><Players>'
        '<Player PersonId="P1"/><Player PersonId="P2"/></Players></Team>'
        '</Teams>'
        '<OtherGameInformation TotalTimeFirstHalf="{}" TotalTimeSecondHalf="{}" '
        'PlayingTimeFirstHalf="120" PlayingTimeSecondHalf="120"/>'
        '</MatchInformation></PutDataRequest>').format(first_half_ms, second_half_ms)
    path.write_text(xml, encoding="utf-8")
    return str(path)


def _frames(first, values, ball=False):
    """<Frame> elements from N = first on, one for each tuple of `values`."""
    out = []
    for i, value in enumerate(values):
        n = first + i
        t = "2023-01-01T12:00:{:06.3f}+00:00".format(n * 0.04)
        if ball:
            x, y, status, possession = value
            extra = ' BallStatus="{}"'.format(status)
            if possession is not None:
                extra += ' BallPossession="{}"'.format(possession)
            out.append('<Frame N="{}" T="{}" X="{}" Y="{}" Z="0.00" S="0.00"{}/>'.format(
                n, t, x, y, extra))
        else:
            x, y, s = value
            out.append('<Frame N="{}" T="{}" X="{}" Y="{}" S="{}"/>'.format(n, t, x, y, s))
    return "".join(out)


def _frameset(section, team, person, frames):
    return '<FrameSet GameSection="{}" TeamId="{}" PersonId="{}">{}</FrameSet>'.format(
        section, team, person, frames)


# Half 1 is N 10..15, half 2 is N 30..33, so the video frame is N - 10.
BALL_1 = [(0.0, 0.0, 1, 2), (1.0, 0.0, 1, 2), (2.0, 0.0, 1, 1), (60.0, 10.0, 0, 1),
          (60.0, 10.0, 0, 2), (3.0, 0.0, 1, None)]
BALL_2 = [(9.0, 9.0, 0, 1), (0.0, 0.0, 1, 1), (1.0, 0.0, 1, 2), (2.0, 0.0, 1, 2)]
WANT_LABELS = [A, A, H, D, D, X] + [X] * 14 + [D, H, A, A]


def _positions(path, extra=""):
    """A positions XML: one referee, two home players, one guest player and the ball."""
    still = [(0.0, 0.0, 0.0)] * 6
    body = "".join([
        _frameset("firstHalf", "referee", "R1", _frames(10, still)),
        _frameset("firstHalf", "CLU-H", "P1", _frames(10, [(52.5, 34.0, 36.0)] * 6)),
        _frameset("firstHalf", "CLU-G", "Q1", _frames(10, [(-52.5, -34.0, 7.2)] * 6)),
        _frameset("firstHalf", "BALL", "B", _frames(10, BALL_1, ball=True)),
        _frameset("secondHalf", "CLU-H", "P1", _frames(30, [(1.0, 1.0, 3.6)] * 2)),
        _frameset("secondHalf", "CLU-H", "P2", _frames(32, [(2.0, 2.0, 3.6)] * 2)),
        _frameset("secondHalf", "CLU-G", "Q1", _frames(30, [(3.0, 3.0, 3.6)] * 4)),
        _frameset("secondHalf", "BALL", "B", _frames(30, BALL_2, ball=True)),
        extra,
    ])
    path.write_text('<?xml version="1.0" encoding="UTF-8"?><PutDataRequest><Positions>'
                    '{}</Positions></PutDataRequest>'.format(body), encoding="utf-8")
    return str(path)


def _bundle(tmp_path, **kwargs):
    info = dfl.read_matchinfo(_matchinfo(tmp_path / "info.xml", **kwargs))
    positions = dfl.read_positions(_positions(tmp_path / "pos.xml"),
                                   {t["team_id"] for t in info["teams"]})
    return dfl.build(info, positions)


def test_stem_and_sealed_ids():
    assert dfl.stem_of("DFL-MAT-J03WMX") == "DFL-J03WMX"
    assert dfl.SEALED == ("DFL-MAT-J03WQQ", "DFL-MAT-J03WR9")


def test_frames_halves_and_offset(tmp_path):
    bundle = _bundle(tmp_path)
    assert bundle["fps"] == 25
    assert bundle["halves"] == {1: (10, 15), 2: (30, 33)}
    assert bundle["offset"] == 10
    assert bundle["n_frames"] == 24


def test_labels_follow_ball_status_and_possession(tmp_path):
    bundle = _bundle(tmp_path)
    assert bundle["labels"].tolist() == WANT_LABELS
    # BallStatus 1 without BallPossession stays X. It never becomes D.
    assert bundle["unknown_label_frames"] == {1: 1, 2: 0}
    assert dfl.coverage(bundle) == {1: 5 / 6.0, 2: 1.0}


def test_home_team_is_first_whatever_the_file_order(tmp_path):
    bundle = _bundle(tmp_path)
    assert [t["name"] for t in bundle["teams"]] == ["Home FC", "Guest FC"]
    assert bundle["slots"] == [("CLU-H", "P1"), ("CLU-H", "P2"), ("CLU-G", "Q1")]


def test_coordinates_are_normalised_with_y_down_and_speed_in_ms(tmp_path):
    bundle = _bundle(tmp_path)
    assert bundle["px"][0].tolist()[0] == 1.0 and bundle["py"][0, 0] == 0.0
    assert bundle["px"][0, 2] == 0.0 and bundle["py"][0, 2] == 1.0
    assert bundle["ps"][0, 0] == pytest.approx(10.0)
    assert bundle["ps"][0, 2] == pytest.approx(2.0)
    assert (bundle["bx"][0], bundle["by"][0]) == (0.5, 0.5)
    assert np.isnan(bundle["px"][0, 1])          # P2 is not on the pitch in half 1
    assert np.isnan(bundle["px"][10]).all()      # halftime frames have no data
    assert np.isnan(bundle["bx"][10])
    assert np.isfinite(bundle["px"][22, 1])      # P2 comes on at N 32


def test_half_end_must_match_the_match_information(tmp_path):
    with pytest.raises(ValueError, match="duration"):
        _bundle(tmp_path, first_half_ms=240)
    with pytest.raises(ValueError, match="whole number"):
        _bundle(tmp_path, first_half_ms=210)


def test_unknown_team_id_is_an_error(tmp_path):
    info = dfl.read_matchinfo(_matchinfo(tmp_path / "info.xml"))
    extra = _frameset("firstHalf", "CLU-X", "Z1", _frames(10, [(0.0, 0.0, 0.0)]))
    with pytest.raises(ValueError, match="unknown TeamId"):
        dfl.read_positions(_positions(tmp_path / "pos.xml", extra), {"CLU-H", "CLU-G"})
    assert info["match_id"] == "DFL-MAT-T00001"


def test_frame_rate_from_times():
    times = ["2023-01-01T12:00:00.000+00:00", "2023-01-01T12:00:00.040+00:00",
             "2023-01-01T12:00:00.080+00:00"]
    assert dfl.frame_rate(times) == 25
    with pytest.raises(ValueError):
        dfl.frame_rate(times[:1])


def test_games_fields_and_entry_text_agree(tmp_path):
    bundle = _bundle(tmp_path)
    fields = dfl.games_fields(bundle, "dfl/T00001")
    assert fields["offsets"] == {1: 10, 2: 10}
    assert fields["kickoff"] == {"minute": 0, "second": 0, "frame": 0}
    assert fields["kickoff_2nd"] == {"minute": 0, "second": 0, "frame": 21}
    assert "kickoff_from_anchor" not in fields
    assert fields["assets"]["atd_raw"] == "dfl/T00001/DFL-T00001_atd.txt"
    parsed = eval("{" + dfl.entry_text(bundle, fields) + "}")
    assert parsed == {"DFL-T00001": fields}


def test_kickoff_from_anchor_when_half_1_starts_dead(tmp_path):
    bundle = _bundle(tmp_path)
    bundle["kickoff_video"] = {1: 2, 2: 21}
    fields = dfl.games_fields(bundle, "dfl/T00001")
    assert fields["kickoff_from_anchor"] is True
    assert fields["kickoff"] == {"minute": 0, "second": 0, "frame": 2}


def _write(tmp_path):
    bundle = _bundle(tmp_path)
    out = tmp_path / "refs" / "dfl" / "T00001"
    summary = dfl.write_bundle(bundle, str(out), "dfl/T00001",
                               {"matchinfo": str(tmp_path / "info.xml")})
    return bundle, summary, out


def test_engine_data_layer_reads_the_bundle(tmp_path, monkeypatch):
    bundle, summary, out = _write(tmp_path)
    monkeypatch.setitem(games.GAMES, GAME, summary["games_entry"])
    root = str(tmp_path / "refs")
    tactical = data.load_tactical(GAME, data_dir=root, use_cache=False)
    assert tactical["team_names"] == ("Home FC", "Guest FC")
    assert tactical["team_of_slot"].tolist() == [0, 0, 1]
    assert tactical["n_frames"] == 24
    np.testing.assert_allclose(tactical["px"], bundle["px"], atol=1e-5)
    np.testing.assert_allclose(tactical["ps"], bundle["ps"], atol=1e-3)
    np.testing.assert_allclose(tactical["by"], bundle["by"], atol=1e-5)
    assert data.graded_blocks(GAME) == [(1, 0, 5), (2, 20, 23)]
    assert data.kickoff_frames(GAME) == {1: 0, 2: 21}


def test_reference_csv_and_summary(tmp_path):
    _, summary, out = _write(tmp_path)
    rows = (out / "DFL-T00001_reference.csv").read_text().splitlines()
    assert rows[0] == "frame,label"
    assert "".join(r.split(",")[1] for r in rows[1:]) == "".join(
        dfl.events_gt.LETTER[c] for c in WANT_LABELS)
    assert summary["slots"] == {"home": 2, "guest": 1}
    assert summary["inputs"]["matchinfo"]["sha256"] == dfl.sha256(str(tmp_path / "info.xml"))
    with open(str(out / "DFL-T00001_build.json")) as handle:
        assert json.load(handle)["match_id"] == "DFL-MAT-T00001"


def test_build_is_deterministic(tmp_path):
    _, first, _ = _write(tmp_path)
    _, second, _ = _write(tmp_path)
    assert first["outputs"] == second["outputs"]


def test_audit_refuses_a_sealed_match(tmp_path, monkeypatch):
    _, summary, out = _write(tmp_path)
    built = out / "DFL-T00001_build.json"
    record = json.loads(built.read_text())
    record["match_id"] = dfl.SEALED[0]
    built.write_text(json.dumps(record))
    monkeypatch.setitem(games.GAMES, GAME, summary["games_entry"])
    monkeypatch.setattr(games, "DATA_DIR", str(tmp_path / "refs"))
    with pytest.raises(SystemExit, match="sealed"):
        dfl.main(["audit", "--game", GAME])


def test_reference_stats_count_short_and_flicker_spells():
    fps = 25.0
    codes = np.array([X] + [H] * 100 + [A] * 10 + [H] * 100 + [D] * 50 + [A] * 60 + [X],
                     dtype=np.int8)
    runs = dfl.pm.valid_runs(codes)
    stats = dfl._reference_stats(codes, runs, fps)
    assert stats["possessions"] == 4
    assert stats["spells_under_1s"] == 1
    assert stats["flicker_spells_under_1s"] == 1
    assert stats["transitions"] == {"H<->A": 2, "alive->D": 1, "D->alive": 1}
    assert stats["d_segments"] == 1


# -- the games.py entries (no data needed) --------------------------------------------------------

DFL_GAMES = sorted(name for name in games.GAMES if name.startswith("DFL-"))


def test_seven_dfl_entries_include_the_sealed_ones():
    assert len(DFL_GAMES) == 7
    for match_id in dfl.SEALED:
        assert dfl.stem_of(match_id) in DFL_GAMES


@pytest.mark.parametrize("name", DFL_GAMES)
def test_dfl_entry_is_consistent(name):
    entry = games.GAMES[name]
    assert entry["fps"] == 25
    assert entry["offsets"][1] == entry["offsets"][2] == entry["halves"][1][0]
    assert entry["n_video_frames"] == entry["halves"][2][1] - entry["offsets"][1] + 1
    blocks = data.graded_blocks(name)
    assert blocks[0][1] == 0 and blocks[-1][2] == entry["n_video_frames"] - 1
    assert blocks[0][2] < blocks[1][1] - 1          # an X gap separates the halves
    kicks = data.kickoff_frames(name)
    for half, first, last in blocks:
        assert first <= kicks[half] <= last
    short = name.split("-")[1]
    assert entry["assets"] == {"atd_meta": "dfl/{0}/{1}_atd.xml".format(short, name),
                               "atd_raw": "dfl/{0}/{1}_atd.txt".format(short, name)}


@pytest.mark.parametrize("name", DFL_GAMES)
def test_dfl_entry_matches_its_built_bundle(name):
    """Skips unless MTAH_DATA_DIR holds the bundle (see the games.py comment block)."""
    entry = games.GAMES[name]
    path = os.path.join(games.DATA_DIR, entry["assets"]["atd_meta"].replace(
        "_atd.xml", "_build.json"))
    if not os.path.isfile(path):
        pytest.skip("no DFL bundle under MTAH_DATA_DIR")
    with open(path) as handle:
        built = json.load(handle)["games_entry"]
    for field in ("n_video_frames", "fps", "kickoff", "kickoff_2nd", "assets"):
        assert built[field] == entry[field]
    assert {int(h): v for h, v in built["offsets"].items()} == entry["offsets"]
    assert {int(h): tuple(v) for h, v in built["halves"].items()} == entry["halves"]
    assert tuple(built["pitch"]) == entry["pitch"]
    assert built.get("kickoff_from_anchor", False) == entry.get("kickoff_from_anchor", False)
