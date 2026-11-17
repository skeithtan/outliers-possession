"""tools/gapmask.py on synthetic ball visibility and labels (P9-E).

The synthetic tests need no match data. The last test builds the mask of this match and needs the
match data bundle. It is skipped without it.
"""

import os

import numpy as np
import pytest

import games
from tools import gapmask as gm

BLOCKS = [(1, 100, 4099), (2, 4500, 8499)]
N = 9000


def synthetic_missing():
    """A ball-missing mask with gaps inside the blocks and two gaps on block edges."""
    missing = np.zeros(N, dtype=bool)
    missing[:100] = True                      # before the first block: not graded
    missing[100:130] = True                   # touches the first block start: cut
    for start, length in ((400, 10), (900, 150), (1600, 300), (2500, 60), (3300, 40),
                          (5000, 200), (6100, 25), (7000, 400)):
        missing[start:start + length] = True
    missing[8450:8500] = True                 # touches the second block end: cut
    return missing


def visible_graded(missing):
    return gm.graded_mask(N, BLOCKS) & ~missing


def test_observed_gaps_leave_out_runs_on_a_block_edge():
    gaps, cut = gm.observed_gaps(synthetic_missing(), BLOCKS)
    assert cut == 2
    assert [b - a for a, b in gaps] == [10, 150, 300, 60, 40, 200, 25, 400]


def test_drawn_gaps_reach_the_fraction_within_one_gap_length():
    missing = synthetic_missing()
    visible = visible_graded(missing)
    lengths = [b - a for a, b in gm.observed_gaps(missing, BLOCKS)[0]]
    for seed in range(5):
        drawn = gm.draw_gaps(visible, lengths, 0.15, seed)
        assert drawn["reached"]
        assert drawn["target"] == int(np.ceil(0.15 * visible.sum()))
        assert drawn["hidden"] == int(drawn["lengths"].sum())
        assert drawn["hidden"] >= drawn["target"]
        assert drawn["hidden"] - int(drawn["lengths"][-1]) < drawn["target"]
        assert set(drawn["lengths"].tolist()) <= set(lengths)


def test_drawn_gaps_never_overlap_and_keep_the_margin():
    missing = synthetic_missing()
    visible = visible_graded(missing)
    lengths = [b - a for a, b in gm.observed_gaps(missing, BLOCKS)[0]]
    drawn = gm.draw_gaps(visible, lengths, 0.3, 7)
    hidden = gm.hidden_mask(N, drawn["starts"], drawn["lengths"])
    assert int(hidden.sum()) == drawn["hidden"]          # no two inserted gaps share a frame
    assert not (hidden & ~visible).any()                 # never on a gap or off the blocks
    for start, length in zip(drawn["starts"].tolist(), drawn["lengths"].tolist()):
        others = hidden.copy()
        others[start:start + length] = False
        around = slice(start - gm.MARGIN, start + length + gm.MARGIN)
        assert visible[around].all() and not others[around].any()


def test_drawn_gaps_are_deterministic_for_a_seed():
    missing = synthetic_missing()
    visible = visible_graded(missing)
    lengths = [b - a for a, b in gm.observed_gaps(missing, BLOCKS)[0]]
    first = gm.draw_gaps(visible, lengths, 0.15, 3)
    second = gm.draw_gaps(visible, lengths, 0.15, 3)
    other = gm.draw_gaps(visible, lengths, 0.15, 4)
    assert np.array_equal(first["starts"], second["starts"])
    assert np.array_equal(first["lengths"], second["lengths"])
    assert gm.mask_md5("G", 3, 0.15, first["starts"], first["lengths"]) == \
        gm.mask_md5("G", 3, 0.15, second["starts"], second["lengths"])
    assert not np.array_equal(first["starts"], other["starts"])


def test_a_draw_that_fits_nowhere_is_rejected():
    visible = np.zeros(200, dtype=bool)
    visible[10:60] = True                                # 50 visible frames
    drawn = gm.draw_gaps(visible, [100, 5], 0.2, 0)
    assert drawn["reached"] and set(drawn["lengths"].tolist()) == {5}
    assert drawn["rejected"] == drawn["draws"] - len(drawn["lengths"])


def test_label_change_accounting():
    """Ten graded frames, four hidden. Two hidden frames and one outside frame change."""
    H, A, D, X = gm.H, gm.A, gm.D, gm.X
    ref = np.array([X, H, H, H, H, H, A, A, D, D, D, X])
    new = np.array([X, H, H, D, D, H, A, H, D, D, A, X])
    graded = np.zeros(12, dtype=bool)
    graded[1:11] = True
    hidden = np.zeros(12, dtype=bool)
    hidden[2:6] = True
    hidden[0] = True                                     # not graded: never counted
    out = gm.label_changes(ref, new, hidden, graded)
    assert out["hidden_frames"] == 4 and out["hidden_changed"] == 2
    assert out["hidden_changed_share"] == 0.5
    assert out["graded_frames"] == 10 and out["graded_changed"] == 4
    assert out["graded_changed_share"] == 0.4
    assert out["outside_changed"] == 2
    assert out["hidden_pairs"] == {"HD": 2}
    assert out["outside_pairs"] == {"AH": 1, "DA": 1}


def test_alive_holds_count_gaps_inside_one_spell():
    H, A, D = gm.H, gm.A, gm.D
    ref = np.array([H] * 40 + [A] * 20 + [D] * 20)
    new = ref.copy()
    new[12:16] = D                                       # the first gap does not hold
    mask = {"starts": np.array([10, 30, 55]), "lengths": np.array([10, 5, 10])}
    out = gm.alive_holds(ref, new, mask)
    assert out["gaps"] == 2                              # the third gap crosses A to D
    assert out["frames"] == 15 and out["frames_kept"] == 11 and out["gaps_all_kept"] == 1


def test_bucket_of_and_names():
    assert [gm.bucket_of(n, 25) for n in (1, 49, 50, 124, 125, 249, 250, 749, 750, 9000)] == \
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert [gm.bucket_name(*b) for b in gm.BUCKETS_S] == \
        ["< 2 s", "2-5 s", "5-10 s", "10-30 s", "> 30 s"]


def test_apply_mask_hides_the_ball_only():
    tactical = {"bx": np.linspace(0.1, 0.9, 6, dtype=np.float32),
                "by": np.full(6, 0.5, dtype=np.float32),
                "px": np.ones((6, 2), dtype=np.float32)}
    hidden = np.array([False, True, True, False, False, False])
    out = gm.apply_mask(tactical, hidden)
    assert np.isnan(out["bx"][1:3]).all() and np.isnan(out["by"][1:3]).all()
    assert np.isfinite(out["bx"][[0, 3, 4, 5]]).all()
    assert out["px"] is tactical["px"]
    assert np.isfinite(tactical["bx"]).all()             # the input is not changed


def test_frozen_mask_is_reused_and_checked(tmp_path):
    missing = synthetic_missing()
    path = gm.mask_path(str(tmp_path), "G", 0, 0.15)
    first, reused = gm.frozen_mask(path, missing, BLOCKS, "G", 0, 0.15)
    assert not reused and os.path.isfile(path)
    second, reused = gm.frozen_mask(path, missing, BLOCKS, "G", 0, 0.15)
    assert reused and second["md5"] == first["md5"]
    assert np.array_equal(second["starts"], first["starts"])
    with pytest.raises(ValueError):
        gm.frozen_mask(path, missing, BLOCKS, "G", 1, 0.15)
    other = missing.copy()
    other[2000:2100] = True
    with pytest.raises(ValueError):
        gm.frozen_mask(path, other, BLOCKS, "G", 0, 0.15)


_ASSETS = games.asset_paths()
HAVE_DATA = all(os.path.isfile(_ASSETS[k]) for k in ("atd_meta", "atd_raw"))


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_DATA, reason="match data bundle not present")
def test_mask_of_this_match():
    """The frozen P9-E mask of BAR-ATH: seed 0, fraction 0.15."""
    from engine import data

    missing = np.isnan(data.load_tactical()["bx"])
    blocks = data.graded_blocks()
    mask = gm.build_mask(missing, blocks, games.GAME, 0, 0.15)
    assert mask["md5"] == "8a38290527d559eea06e4a8ade41343f"
    assert len(mask["observed_lengths"]) == 147 and mask["observed_cut"] == 3
    assert mask["reached"] and mask["hidden"] == 15169 and len(mask["lengths"]) == 48
    hidden = gm.hidden_mask(len(missing), mask["starts"], mask["lengths"])
    assert not (hidden & (missing | ~gm.graded_mask(len(missing), blocks))).any()
