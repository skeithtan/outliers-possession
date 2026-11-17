"""P0-A: data layer alignment assertions (docs/plan/tasks/P0-A-data-layer.md).

The match-specific numbers below (199319, 11185, 113742, 638, ...) live here as assertions and
nowhere in engine/ (PLAN.md 5.5). Tests marked `slow` need the data bundle and parse it once per
session (~5-10 s per source when the cache is cold).
"""

import glob
import os
import re

import numpy as np
import pytest

import games
from engine import data
from tools.gt import public_gt_codes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_TACTICAL = 199319
N_BROADCAST = 164488
PUBLIC_WINDOW = (11185, 18684)          # inclusive, PLAN.md Section 1
KICKOFF_1, KICKOFF_2 = 11185, 113742    # DATA.md / starter/load_smart_tagging.py

# Exactly the assets the loaders read (not the videos), so a missing input skips, never passes.
_TACTICAL = games.asset_paths()
_BROADCAST = games.asset_paths(source="broadcast")
HAVE_DATA = all(os.path.isfile(_TACTICAL[k]) for k in
                ("atd_meta", "atd_raw", "homography", "smart_tagging")) and all(
    os.path.isfile(_BROADCAST[k]) for k in ("atd_meta", "atd_raw", "homography"))
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="match data bundle not present")


@pytest.fixture(scope="session")
def tactical():
    return data.load_tactical()


@pytest.fixture(scope="session")
def broadcast():
    return data.load_broadcast()


@pytest.fixture(scope="session")
def tagging():
    return data.load_tagging()


# -- tactical ATD --------------------------------------------------------------------------------

@pytest.mark.slow
@needs_data
def test_tactical_raw_file_contiguous_frame_ids():
    """Independent of the loader: the raw file has 199,319 lines, frame_id 1..199319 in order."""
    expected = 1
    with open(games.asset_paths()["atd_raw"]) as handle:
        for line in handle:
            assert int(line[:line.index(":")]) == expected
            expected += 1
    assert expected - 1 == N_TACTICAL


@pytest.mark.slow
@needs_data
def test_tactical_arrays(tactical):
    assert tactical["n_frames"] == N_TACTICAL
    assert tactical["fps"] == 25
    n_slots = tactical["n_slots"]
    assert n_slots == 46
    for key in ("px", "py", "ps"):
        assert tactical[key].shape == (N_TACTICAL, n_slots)
        assert tactical[key].dtype == np.float32
    for key in ("bx", "by"):
        assert tactical[key].shape == (N_TACTICAL,)
        assert tactical[key].dtype == np.float32
    # Normalised coordinates: finite values sit around [0, 1] (players can stand off the pitch).
    finite = tactical["bx"][np.isfinite(tactical["bx"])]
    assert finite.size > 0.4 * N_TACTICAL
    assert -0.2 < finite.min() and finite.max() < 1.2


@pytest.mark.slow
@needs_data
def test_team_of_slot(tactical):
    team = tactical["team_of_slot"]
    assert team.dtype == np.int8
    assert int((team == 0).sum()) == 24
    assert int((team == 1).sum()) == 22
    assert tactical["team_names"][0] == "Barcelona"


# -- broadcast -----------------------------------------------------------------------------------

@pytest.mark.slow
@needs_data
def test_broadcast_arrays(broadcast):
    assert broadcast["n_frames"] == N_BROADCAST
    assert broadcast["px"].shape == (N_BROADCAST, broadcast["n_slots"])
    assert broadcast["bx"].shape == (N_BROADCAST,)
    assert broadcast["valid"].dtype == np.bool_
    assert broadcast["valid"].shape == (N_BROADCAST,)
    assert abs(broadcast["valid"].mean() - 0.658) < 0.01  # PLAN.md Appendix B


# -- homography ----------------------------------------------------------------------------------

@pytest.mark.slow
@needs_data
def test_tactical_homography_all_valid():
    matrices, error, valid = data.load_homography("tactical")
    assert matrices.shape == (N_TACTICAL, 3, 3)
    assert error.shape == valid.shape == (N_TACTICAL,)
    assert valid.dtype == np.bool_
    assert valid.all()


# -- Smart Tagging -------------------------------------------------------------------------------

@needs_data
def test_tagging_counts(tagging):
    assert len(tagging) == 638
    assert int((tagging["code"] == "SET PIECES").sum()) == 59
    assert set(tagging["team"].dropna().unique()) == {"H", "A"}
    assert int(tagging["team"].isna().sum()) == 36          # "N/A" -> None
    throw_ins = tagging[tagging["type"] == "Throw-in"]
    assert len(throw_ins) == 26 and throw_ins["team"].isna().all()  # PLAN.md 1.6
    assert set(tagging["half"]) == {1, 2}
    assert (tagging["end_frame"] >= tagging["start_frame"]).all()
    for column in ("max_players_in_the_box", "side", "direction_of_ball_entry"):
        assert column in tagging.columns


@needs_data
def test_tagged_kickoffs_contain_real_kickoffs(tagging):
    """Time-base check (starter/load_smart_tagging.py::check_time_base), half-open windows."""
    kickoffs = tagging[(tagging["code"] == "SET PIECES")
                       & (tagging["type"].str.lower() == "kick off (start)")]
    assert len(kickoffs) == 2
    for half, frame in ((1, KICKOFF_1), (2, KICKOFF_2)):
        rows = kickoffs[kickoffs["half"] == half]
        assert len(rows) == 1
        assert int(rows["start_frame"].iloc[0]) <= frame < int(rows["end_frame"].iloc[0])


# -- games.py derived constants ------------------------------------------------------------------

def test_kickoff_frames():
    kick = data.kickoff_frames()
    assert abs(kick[1] - KICKOFF_1) <= 1
    assert abs(kick[2] - KICKOFF_2) <= 1
    assert kick == {1: 11185, 2: 113741}


def test_graded_blocks_and_pitch():
    assert data.graded_blocks() == [(1, 11185, 90651), (2, 110808, 190197)]
    assert data.pitch_m() == (105.3, 68.0)


# -- public GT (tools/gt.py) ---------------------------------------------------------------------

def test_public_gt_window_and_counts():
    codes = public_gt_codes()
    assert codes.dtype == np.int8 and codes.shape == (N_TACTICAL,)
    index = np.arange(N_TACTICAL)
    in_window = (index >= PUBLIC_WINDOW[0]) & (index <= PUBLIC_WINDOW[1])
    assert np.array_equal(codes != 3, in_window)
    counts = np.bincount(codes, minlength=4)
    assert counts[0] == 3454 and counts[1] == 2577 and counts[2] == 1469


# -- cache ---------------------------------------------------------------------------------------

ARRAY_KEYS = ("px", "py", "ps", "bx", "by", "team_of_slot")


def _assert_same(a, b, keys=ARRAY_KEYS):
    for key in keys:
        assert a[key].dtype == b[key].dtype, key
        assert np.array_equal(a[key], b[key], equal_nan=True), key


@pytest.mark.slow
@needs_data
def test_cache_roundtrip_and_invalidation(tmp_path, tactical):
    cache_dir = str(tmp_path)
    first = data.load_tactical(cache_dir=cache_dir)            # parses, writes cache
    cache_file = os.path.join(cache_dir, "tactical.npz")
    assert os.path.isfile(cache_file)
    second = data.load_tactical(cache_dir=cache_dir)           # reads cache
    _assert_same(first, second)
    _assert_same(first, tactical)                              # repo cache agrees too
    assert first["team_names"] == second["team_names"]

    # A cache with the wrong key must be ignored and rebuilt.
    tampered = {k: second[k] for k in ARRAY_KEYS}
    tampered["team_names"] = np.array(second["team_names"])
    tampered["bx"] = np.zeros_like(tampered["bx"])
    data._write_cache(cache_file, "stale-key", tampered)
    third = data.load_tactical(cache_dir=cache_dir)
    _assert_same(first, third)


@pytest.mark.slow
@needs_data
def test_broadcast_cache_matches_fresh_parse(broadcast):
    fresh = data.load_broadcast(use_cache=False)
    _assert_same(broadcast, fresh, ARRAY_KEYS + ("valid",))
    assert broadcast["team_names"] == fresh["team_names"]


def test_cache_key_tracks_file_size_and_mtime(tmp_path):
    """A changed bundle (new mtime or new size) must change the cache key (card: Cache)."""
    paths = {"atd_meta": _write(tmp_path / "a.xml", ["<x/>"]),
             "atd_raw": _write(tmp_path / "a.txt", ["1:0,0,0:0,0,0"])}
    key = data._cache_key("G", "tactical", paths)
    assert data._cache_key("G", "tactical", paths) == key
    assert data._cache_key("G", "broadcast", paths) != key
    stat = os.stat(paths["atd_raw"])
    os.utime(paths["atd_raw"], ns=(stat.st_atime_ns, stat.st_mtime_ns + 10 ** 9))
    key_touched = data._cache_key("G", "tactical", paths)
    assert key_touched != key
    with open(paths["atd_raw"], "a") as handle:
        handle.write("x")
    os.utime(paths["atd_raw"], ns=(stat.st_atime_ns, stat.st_mtime_ns + 10 ** 9))
    assert data._cache_key("G", "tactical", paths) != key_touched


# -- fast synthetic tests ------------------------------------------------------------------------

def _write(path, lines):
    with open(path, "w") as handle:
        handle.write("".join(line + "\n" for line in lines))
    return str(path)


def test_parse_atd_raw_synthetic(tmp_path):
    lines = ["1:0.1,0.2,1.5;NaN,NaN,NaN:0.5,0.5,NaN",
             "2:0.11,0.21,1.6;0.9,0.8,2.0:NaN,NaN,NaN",
             "3:NaN,NaN,NaN;0.91,0.81,2.1:0.52,0.48,7.0"]
    out = data._parse_atd_raw(_write(tmp_path / "atd.txt", lines), n_slots=2, n_frames=3)
    assert out["px"].shape == (3, 2) and out["px"].dtype == np.float32
    assert out["px"][0, 0] == np.float32(0.1) and np.isnan(out["px"][0, 1])
    assert out["ps"][1, 1] == np.float32(2.0)
    assert np.isnan(out["bx"][1]) and out["by"][2] == np.float32(0.48)


@pytest.mark.parametrize("lines", [
    ["1:0.1,0.2,1.5;NaN,NaN,NaN:0.5,0.5,NaN", "3:0.1,0.2,1.5;NaN,NaN,NaN:0.5,0.5,NaN"],  # gap
    ["2:0.1,0.2,1.5;NaN,NaN,NaN:0.5,0.5,NaN", "1:0.1,0.2,1.5;NaN,NaN,NaN:0.5,0.5,NaN"],  # order
    ["1:0.1,0.2,1.5:0.5,0.5,NaN", "2:0.1,0.2,1.5;NaN,NaN,NaN:0.5,0.5,NaN"],              # short
    ["1:0.1,0.2,1.5;NaN,NaN,NaN:0.5,0.5,NaN"],                                           # count
    # Separator totals balance (one line 1 group, the other 3): must still be rejected.
    ["1:0.1,0.2,1.5:0.5,0.5,NaN", "2:0.1,0.2,1.5;1,1,1;2,2,2:0.5,0.5,NaN"],
    ["1:0.1,0.2,1.5;1,1,1;2,2,2:0.5,0.5,NaN", "2:0.1,0.2,1.5:0.5,0.5,NaN"],
])
def test_parse_atd_raw_rejects_malformed(tmp_path, lines):
    with pytest.raises(ValueError):
        data._parse_atd_raw(_write(tmp_path / "atd.txt", lines), n_slots=2, n_frames=2)


def test_team_side_mapping():
    names = ("Barcelona", "Atletic Club")
    assert data._team_side("Barcelona", names) == "H"
    assert data._team_side("BARCELONA ", names) == "H"
    assert data._team_side("FC Barcelona", names) == "H"
    assert data._team_side("Atletic Club", names) == "A"
    assert data._team_side("athletic club", names) == "A"
    assert data._team_side("N/A", names) is None
    assert data._team_side(None, names) is None
    with pytest.raises(ValueError):
        data._team_side("Real Madrid", names)


def test_half_number():
    assert data._half_number("1st Half") == 1
    assert data._half_number("2nd Half") == 2
    with pytest.raises(ValueError):
        data._half_number("Extra time")


def test_engine_never_touches_ground_truth():
    """Team hard rule: engine/ never reads the GT and never imports tools.gt."""
    forbidden = re.compile(r"possession_gt|evaluation/gt|['\"]gt['\"]|tools\.gt|from tools|"
                           r"import tools|_gt_public")
    sources = glob.glob(os.path.join(ROOT, "engine", "**", "*.py"), recursive=True)
    assert sources
    for path in sources:
        with open(path) as handle:
            text = handle.read()
        assert not forbidden.search(text), path
