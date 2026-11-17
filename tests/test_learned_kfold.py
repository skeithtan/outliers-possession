"""The folds of the learned stage match LEARNED_STAGE_PROTOCOL.md Section 3 and
LEARNED_STAGE_PROTOCOL_R2.md Section 1 (no torch needed)."""

import os
import re

import pytest

from tools.learned_spike import decoders
from tools.learned_spike import folds as fo
from tools.learned_spike import pff

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R2 = os.path.join(ROOT, "docs", "plan", "calibration", "LEARNED_STAGE_PROTOCOL_R2.md")

# The exact lists of the protocol, written out by hand from Section 3.
EXPECTED = {
    0: (["METRICA-S1", "SC-1874553", "SC-1996435", "SC-2013725"],
        ["DFL-J03WMX", "SC-1886347"]),
    1: (["METRICA-S2", "SC-1886347", "SC-1996436", "SC-2015213"],
        ["DFL-J03WMX", "SC-1899585"]),
    2: (["METRICA-S3", "SC-1899585", "SC-2006229", "SC-2016236"],
        ["DFL-J03WMX", "SC-1925299"]),
    3: (["DFL-J03WMX", "SC-1925299", "SC-2006363", "SC-2017461"],
        ["METRICA-S1", "SC-1927964"]),
    4: (["DFL-J03WN1", "SC-1927964", "SC-2007448"],
        ["METRICA-S1", "SC-1953632"]),
    5: (["DFL-J03WOH", "SC-1953632", "SC-2007721"],
        ["METRICA-S1", "SC-1959846"]),
    6: (["DFL-J03WOY", "SC-1959846", "SC-2010085"],
        ["METRICA-S1", "SC-1986691"]),
    7: (["DFL-J03WPY", "SC-1986691", "SC-2011166"],
        ["METRICA-S1", "SC-1874553"]),
}


def test_match_lists():
    assert len(fo.OPEN) == 8 and len(fo.SC) == 20 and len(set(fo.ALL)) == 28
    assert list(fo.SC) == sorted(fo.SC)
    assert not set(fo.ALL) & set(fo.SEALED)
    assert fo.COMPETITION not in fo.ALL


def test_sc_list_matches_the_recorded_references():
    from tools.analysis import external_sweep as xs
    assert list(fo.SC) == sorted(xs.SKILLCORNER_REFERENCE_MD5)
    assert list(fo.OPEN) == list(xs.OPEN_GAMES)
    assert set(fo.SEALED) == set(xs.SEALED_GAMES)


def test_competition_name_matches_games_py():
    import games
    assert fo.COMPETITION == games.GAME
    for g in fo.ALL:
        assert g in games.GAMES


@pytest.mark.parametrize("k", range(8))
def test_exact_fold_lists(k):
    f = fo.fold(k)
    test, val = EXPECTED[k]
    assert f["test"] == test
    assert f["val"] == val
    assert f["train"] == [g for g in fo.ALL if g not in test + val] + list(fo.PFF_TRAIN)
    assert len(f["train"]) == 28 - len(test) - len(val) + 43


def test_every_match_tested_exactly_once():
    tested = [g for f in fo.folds() for g in f["test"]]
    assert sorted(tested) == sorted(fo.ALL)
    assert [len(f["test"]) for f in fo.folds()] == [4, 4, 4, 4, 3, 3, 3, 3]


def test_no_test_or_validation_match_in_training():
    for f in fo.folds():
        assert not set(f["train"]) & set(f["test"])
        assert not set(f["train"]) & set(f["val"])
        assert not set(f["test"]) & set(f["val"])
        assert set(f["train"]) | set(f["test"]) | set(f["val"]) == set(fo.TRAIN_ALL)


def test_validation_pair_has_one_open_match_of_another_provider():
    for f in fo.folds():
        open_test, open_val = f["test"][0], f["val"][0]
        assert open_val in fo.OPEN
        assert fo.provider(open_val) != fo.provider(open_test)
        assert fo.provider(f["val"][1]) == "skillcorner"


@pytest.mark.parametrize("game", ["DFL-J03WQQ", "DFL-J03WR9", "BAR-ATH", "SC-0000000", "X"])
def test_forbidden_and_unknown_matches_refused(game):
    with pytest.raises(SystemExit):
        fo.check(game)
    with pytest.raises(SystemExit):
        fo.check_all(["METRICA-S1", game])
    with pytest.raises(SystemExit):
        fo.views(game)


def test_views_per_provider():
    assert fo.views("SC-1874553") == ("raw", "degraded")
    assert fo.views("METRICA-S3") == ("raw", "matched", "sigma95", "degraded")
    assert fo.views("DFL-J03WPY") == ("raw", "matched", "sigma95", "degraded")
    assert fo.views("PFF-3814") == ("raw",)
    assert fo.provider("PFF-10516") == "pff"


# -- round 2: PFF as training data only (R2 Section 1) -------------------------------------------

def r2_md5_table():
    """{match: (set, md5)} parsed from R2 Section 8."""
    text = open(R2).read().split("## 8.")[1]
    rows = re.findall(r"^\| (PFF-\d+) \| (training|sealed) \| `([0-9a-f]{32})` \|$", text,
                      re.MULTILINE)
    return {g: (kind, md5) for g, kind, md5 in rows}


def test_pff_lists_follow_the_protocol_rule():
    import games
    u = sorted((g for g in games.GAMES if g.startswith("PFF-")), key=lambda g: int(g[4:]))
    assert len(u) == 62 and not set(u) & set(fo.PFF_EXCLUDED)
    rest = [g for g in u if g not in fo.PFF_PILOT]
    assert len(rest) == 58
    assert tuple(rest[0::4]) == fo.PFF_SEALED
    assert tuple(g for i, g in enumerate(rest) if i % 4) == fo.PFF_TRAIN
    assert len(fo.PFF_TRAIN) == 43 and len(fo.PFF_SEALED) == 15
    assert len(fo.TRAIN_ALL) == 71 and len(set(fo.TRAIN_ALL)) == 71


def test_pff_md5_table_is_the_frozen_section_8():
    if not os.path.isfile(R2):
        pytest.skip("the public copy leaves out the records under docs/ (tools/export_public.py)")
    table = r2_md5_table()
    assert len(table) == 58
    assert {g: m for g, (_, m) in table.items()} == pff.REFERENCE_MD5
    assert sorted(g for g, (k, _) in table.items() if k == "training") == sorted(fo.PFF_TRAIN)
    assert sorted(g for g, (k, _) in table.items() if k == "sealed") == sorted(fo.PFF_SEALED)


def test_pff_training_matches_are_in_every_fold_train_and_never_held_out():
    for f in fo.folds():
        assert set(fo.PFF_TRAIN) <= set(f["train"])
        assert not set(fo.PFF_TRAIN) & set(f["val"] + f["test"])
        assert all(fo.provider(g) != "pff" for g in f["val"] + f["test"])
    assert fo.train_games() == list(fo.ALL) + list(fo.PFF_TRAIN)


@pytest.mark.parametrize("game", list(fo.PFF_SEALED) + list(fo.PFF_PILOT))
def test_sealed_and_pilot_pff_never_enter_training(game):
    with pytest.raises(AssertionError):
        fo.assert_not_sealed(["SC-1874553", game])
    with pytest.raises(AssertionError):
        fo.check_train(game)
    with pytest.raises(AssertionError):
        fo.views(game)
    with pytest.raises(AssertionError):
        fo.crop_shares(list(fo.ALL) + [game])
    with pytest.raises(SystemExit):
        fo.check(game)
    with pytest.raises((AssertionError, SystemExit)):
        pff.reference(game)


def test_only_the_v1_gate_accepts_a_sealed_pff_match():
    for game in fo.PFF_SEALED:
        assert fo.check_sealed_v1(game) == game
    for game in fo.PFF_TRAIN[:3] + fo.PFF_PILOT + ("SC-1874553", "BAR-ATH"):
        with pytest.raises(SystemExit):
            fo.check_sealed_v1(game)


def test_pff_training_match_is_not_a_scored_match():
    with pytest.raises(SystemExit):
        fo.check("PFF-3814")              # the 28 of criteria 1-5 only
    assert fo.check_train("PFF-3814") == "PFF-3814"


def test_crop_shares_uniform_over_matches():
    shares = fo.crop_shares(fo.train_games())
    assert shares["pff"] == pytest.approx(43 / 71.0)
    assert shares["skillcorner"] == pytest.approx(20 / 71.0)
    assert shares["dfl"] == pytest.approx(5 / 71.0)
    assert shares["metrica"] == pytest.approx(3 / 71.0)
    assert fo.crop_shares(fo.fold(0)["train"])["pff"] == pytest.approx(43 / 65.0)
    assert fo.crop_shares(fo.fold(7)["train"])["pff"] == pytest.approx(43 / 66.0)


def test_pff_md5_check_refuses_a_changed_reference(tmp_path):
    game = fo.PFF_TRAIN[0]
    path = tmp_path / "pff" / game[4:] / "{}_reference.csv".format(game)
    path.parent.mkdir(parents=True)
    path.write_text("frame,label\n0,X\n")
    assert pff.differences([game], str(tmp_path))
    with pytest.raises(SystemExit):
        pff.check_references(str(tmp_path), [game])
    with pytest.raises(SystemExit):
        pff.reference(game, str(tmp_path))
    missing = pff.differences([fo.PFF_TRAIN[1]], str(tmp_path))
    assert missing and "missing" in missing[0]


# -- round 2 change B: M's cost grid --------------------------------------------------------------

def test_model_grid_keeps_ha_at_most_twice_ad():
    grid = decoders.model_grid()
    full = decoders.full_model_grid()
    assert len(full) == 42 and len(grid) == 32
    assert all(ha <= 2 * ad for ha, ad in grid)
    assert [p for p in full if p[0] <= 2 * p[1]] == grid       # order kept, H-A major
    assert (24.0, 8.0) not in grid and (32.0, 8.0) not in grid  # round 1 median-like pairs gone
    assert (16.0, 8.0) in grid and (64.0, 32.0) in grid and (1.0, 2.0) in grid
    assert decoders.GRID_HA == (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)   # H grid untouched


def test_model_grid_direct_switch_never_dearer_than_through_d():
    trans = decoders.model_trans(decoders.model_grid())
    for t in trans:
        assert t[0, 1] <= t[0, 2] + t[2, 1]
        assert t[1, 0] <= t[1, 2] + t[2, 0]


def test_fold_index_out_of_range():
    with pytest.raises(ValueError):
        fo.fold(8)
    with pytest.raises(ValueError):
        fo.fold(-1)


def test_final_steps_median_rounded_to_200():
    assert fo.final_steps([1400] * 16 + [1800] * 16) == 1600
    assert fo.final_steps([1000, 1200]) == 1200          # median 1100, a half goes up
    assert fo.final_steps([200, 400, 400, 2000]) == 400
    assert fo.final_steps([50, 50]) == 200               # at least 200
    assert fo.final_steps([1300]) == 1400
    with pytest.raises(ValueError):
        fo.final_steps([])
