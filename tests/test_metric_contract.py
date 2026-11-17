"""Contract tests: the scorer semantics that our plan relies on (PLAN.md 1.3 and 1.7, P9-D).

Every test builds label arrays by hand and calls evaluation/possession_metrics.py directly. The
scorer file is the definition of the metric. If one of these tests fails, the organisers changed
the scorer, or our reading of it is wrong. Then read the scorer again before you change the
engine. The tests need no match data.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALUATION = os.path.join(ROOT, "evaluation")
if EVALUATION not in sys.path:
    sys.path.insert(0, EVALUATION)
import possession_metrics as pm  # noqa: E402

H, A, D, X = pm.H, pm.A, pm.D, pm.X


def _codes(*runs):
    """int8 codes from (label, length) pairs: _codes((H, 3), (A, 2)) is H H H A A."""
    return np.concatenate([np.full(n, k, dtype=np.int8) for k, n in runs])


# -- (a) a predicted X inside a reference possession ------------------------------------------

def test_predicted_x_inside_a_possession_is_free_on_spell_f1_and_costs_macro_f1():
    """PLAN 1.7: X on a graded frame forward-fills under spellF1 and is a miss on macroF1.

    The X frames sit inside an H possession. The forward fill gives them H, so the spells are
    those of the held label. On macroF1 the X frames are false negatives of H.
    """
    gt = _codes((X, 5), (H, 60), (A, 60), (D, 40), (H, 60), (X, 5))
    held = gt.copy()
    abstain = gt.copy()
    abstain[20:40] = X
    wrong = gt.copy()
    wrong[20:40] = A

    res_held = pm.evaluate(gt, held)
    res_x = pm.evaluate(gt, abstain)
    res_wrong = pm.evaluate(gt, wrong)
    assert res_held["spell_f1_mean"] == res_x["spell_f1_mean"] == 1.0
    assert res_held["macro_f1"] == 1.0
    assert res_x["macro_f1"] < res_held["macro_f1"]
    assert res_x["n_abstain_on_graded"] == 20
    # The X frames count as misses of H, never as false positives of a class.
    assert res_x["per_class"]["H"]["recall"] == pytest.approx(100 / 120.0)
    assert res_x["per_class"]["H"]["precision"] == 1.0
    # A wrong team call on the same frames splits the possession, an X does not.
    assert res_wrong["spell_f1_mean"] < res_x["spell_f1_mean"]
    assert res_wrong["n_pred_spells"] == res_x["n_pred_spells"] + 2


def test_predicted_x_after_a_hand_over_carries_the_old_team_across_it():
    """EVALUATION.md: the fill repeats the previous answer, so X at a hand-over makes it late."""
    gt = _codes((H, 100), (A, 100))
    pred = gt.copy()
    pred[100:115] = X
    result = pm.evaluate(gt, pred)
    loose = result["transitions"][max(pm.TAUS)]
    assert loose["n_matched"] == 1
    assert loose["mean_latency"] == 15.0


# -- (b) a reference X separates spells --------------------------------------------------------

def test_reference_x_separates_two_spells_of_the_same_team():
    """PLAN 1.7: a spell is a maximal H or A run inside one graded block of the reference.

    Two H runs around a reference X are two reference spells. A prediction that writes H on
    the X frames too still has two spells, because the scorer encodes each block on its own.
    No transition is counted at a block edge.
    """
    gt = _codes((H, 50), (X, 30), (H, 50))
    pred = np.full(gt.shape, H, dtype=np.int8)
    runs = pm.valid_runs(gt)
    assert runs == [(0, 50), (80, 130)]
    assert pm.possession_spells(gt, runs) == [(0, 50, H), (80, 130, H)]
    assert pm.possession_spells(pred, runs) == [(0, 50, H), (80, 130, H)]
    result = pm.evaluate(gt, pred)
    assert result["n_gt_spells"] == result["n_pred_spells"] == 2
    assert result["spell_f1_mean"] == 1.0
    assert result["transitions"][max(pm.TAUS)]["n_gt"] == 0


def test_reference_x_stops_the_fill_of_a_predicted_x():
    """The fill of a predicted X runs inside one block. It never carries a label across X."""
    gt = _codes((H, 50), (X, 30), (A, 50))
    pred = gt.copy()
    pred[50:80] = H            # the prediction on the reference X frames
    pred[80:90] = X            # head of the second block
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    assert np.all(filled[80:90] == A)          # back-filled from inside the block, not H
    result = pm.evaluate(gt, pred)
    assert result["spell_f1_mean"] == 1.0
    assert result["macro_f1"] < 1.0


# -- (c) the scorer averages a fixed three classes ----------------------------------------------

def test_perfect_prediction_on_an_h_a_only_reference_gives_macro_f1_two_thirds():
    """IDEAS-REVIEW 3 item 2: macroF1 averages H, A and D, present in the reference or not.

    A class with no support and no prediction has F1 0, so a perfect prediction on a reference
    without D scores 2/3 on macroF1 and 1 on spellF1.
    """
    gt = _codes((X, 10), (H, 80), (A, 70), (H, 40), (X, 10))
    result = pm.evaluate(gt, gt.copy())
    assert result["per_class"]["D"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                                        "support": 0}
    assert result["macro_f1"] == pytest.approx(2.0 / 3.0)
    assert result["spell_f1_mean"] == 1.0


# -- (d) the IoU of a shifted short spell ------------------------------------------------------

SHORT = 19      # the 0.8 s H possession 15346-15365 of PLAN 1.3, in frames
FLANK = 120     # length of each flanking A spell


def _shifted(k):
    """(reference, prediction): A H A with the 19-frame H spell moved k frames later."""
    gt = _codes((A, FLANK), (H, SHORT), (A, FLANK))
    pred = gt.copy()
    pred[FLANK:FLANK + SHORT] = A
    pred[FLANK + k:FLANK + k + SHORT] = H
    return gt, pred


@pytest.mark.parametrize("k", list(range(SHORT)))
def test_shift_of_k_frames_gives_the_iou_of_plan_1_3(k):
    """PLAN 1.3: a 6-frame shift of the 19-frame spell gives IoU 0.52, an 11-frame shift < 0.3.

    A shift of k frames keeps SHORT - k frames in common and spans SHORT + k frames, so
    IoU = (SHORT - k) / (SHORT + k). The spell is matched at threshold t if IoU >= t.
    """
    gt, pred = _shifted(k)
    expected = (SHORT - k) / float(SHORT + k)
    runs = pm.valid_runs(gt)
    overlaps = pm.spell_overlaps(pm.possession_spells(gt, runs), pm.possession_spells(pred, runs))
    short_iou = [iou for iou, gi, _ in overlaps if gi == 1]
    assert short_iou == [pytest.approx(expected)]
    flags = {t: bool(pm.matched_gt_flags(overlaps, 3, t)[1]) for t in pm.SPELL_IOUS}
    assert flags == {t: expected >= t for t in pm.SPELL_IOUS}


def test_the_two_shifts_that_plan_1_3_names():
    """PLAN 1.3: 6 frames gives 0.52 (matched at 0.5, not at 0.6). 11 frames fails even 0.3."""
    assert (SHORT - 6) / float(SHORT + 6) == pytest.approx(0.52)
    gt, pred = _shifted(6)
    per_iou = pm.evaluate(gt, pred)["spell_f1"]
    assert per_iou[0.5]["n_matched"] == 3 and per_iou[0.6]["n_matched"] == 2
    gt, pred = _shifted(11)
    assert pm.evaluate(gt, pred)["spell_f1"][0.3]["n_matched"] == 2


def test_a_missed_short_spell_merges_its_neighbours():
    """PLAN 1.3 and 1.7: a missed short spell costs three spells, not one.

    The prediction has one A spell over A H A. Each flanking reference spell has
    IoU FLANK / (2 FLANK + SHORT) against the merge, and the scorer gives the merge to one of
    them only.
    """
    gt = _codes((A, FLANK), (H, SHORT), (A, FLANK))
    pred = np.full(gt.shape, A, dtype=np.int8)
    runs = pm.valid_runs(gt)
    overlaps = pm.spell_overlaps(pm.possession_spells(gt, runs), pm.possession_spells(pred, runs))
    merge = FLANK / float(2 * FLANK + SHORT)
    assert [iou for iou, _, _ in overlaps] == [pytest.approx(merge)] * 2
    assert pm.matched_gt_flags(overlaps, 3, 0.3).tolist() == [True, False, False]
    assert pm.matched_gt_flags(overlaps, 3, 0.5).tolist() == [False, False, False]


# -- (e) frames where the reference is X at the block edges ------------------------------------

def test_prediction_on_reference_x_frames_changes_no_score():
    """PLAN 1.7: X in the reference is never graded. A prediction there is not scored.

    Two predictions that differ only on the reference X frames before, between and after the
    blocks give the same result on every key. The confusion matrix differs only in its X row,
    which the scorer drops (scored_confusion).
    """
    gt = _codes((X, 20), (H, 60), (A, 40), (D, 30), (X, 25), (D, 15), (A, 50), (X, 20))
    first = gt.copy()
    second = gt.copy()
    outside = gt == X
    first[outside] = H
    second[outside] = np.where(np.arange(gt.size)[outside] % 2, A, D)
    one, two = pm.evaluate(gt, first), pm.evaluate(gt, second)
    assert sorted(one) == sorted(two)
    for key in sorted(one):
        if key == "confusion":
            assert one[key][:3] == two[key][:3]
            assert one[key][3] != two[key][3]
        else:
            assert one[key] == two[key], key
    assert one["spell_f1_mean"] == one["macro_f1"] == 1.0


def test_block_edges_give_no_transition_and_no_spell_across_them():
    """The first and last frames of a block start and end spells. They are not transitions.

    The reference opens the second block on D and the prediction opens it on H. The edge
    frame 125 is the first frame of a block, so the scorer counts a D -> A transition at
    140 in the reference and an H -> A one at 140 in the prediction, and nothing at 125.
    """
    gt = _codes((X, 20), (H, 80), (X, 25), (D, 15), (A, 50), (X, 20))
    pred = gt.copy()
    pred[125:140] = H
    runs = pm.valid_runs(gt)
    assert runs == [(20, 100), (125, 190)]
    assert pm.transitions(gt, runs) == [(140, D, A)]
    assert pm.transitions(pred, runs) == [(140, H, A)]
    assert pm.possession_spells(pred, runs) == [(20, 100, H), (125, 140, H), (140, 190, A)]
    result = pm.evaluate(gt, pred)
    assert result["transitions"][max(pm.TAUS)]["n_matched"] == 1
    assert result["n_graded"] == 145
