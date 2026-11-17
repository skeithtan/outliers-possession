"""tools/failure_classes.py: the failure-class tables of tools/score_game.py (P9-D).

Every test builds label arrays by hand. The tests need no match data.
"""

import numpy as np
import pytest

from tools import failure_classes as fc
from tools import score_game as sg
from tools.analysis import r2_metric_decomp as r2

pm = fc.pm
H, A, D, X = pm.H, pm.A, pm.D, pm.X


def _seq(*runs):
    """int8 codes from (label, length) pairs: _seq((H, 3), (A, 2)) is H H H A A."""
    return np.concatenate([np.full(n, k, dtype=np.int8) for k, n in runs])


# A reference with a 1 s, a 3 s and an 8 s spell at 25 fps, then a stoppage.
REF = _seq((X, 5), (H, 25), (A, 75), (H, 200), (D, 50), (X, 5))
RUNS = pm.valid_runs(REF)


def test_short_spell_recall_uses_the_scorer_matching_at_iou_half():
    """The 1 s spell is missed, the 3 s spell has IoU 0.75, the 8 s spell has IoU 0.6."""
    pred = REF.copy()
    pred[5:30] = A                       # the A spell grows to 5..105: IoU 75 / 100
    pred[225:305] = A                    # the H spell keeps 105..225: IoU 120 / 200
    rows = fc.spell_recall_by_duration(REF, pred, RUNS)
    assert [r["bucket"] for r in rows] == ["< 2 s", "2-5 s", "> 5 s"]
    assert [(r["n"], r["recalled"], r["recall"]) for r in rows] == [
        (1, 0, 0.0), (1, 1, 1.0), (1, 1, 1.0)]
    # At IoU 0.7 the 8 s spell fails. score_game writes the same flags into the JSON report.
    assert fc.spell_recall_by_duration(REF, pred, RUNS, iou=0.7)[2]["recalled"] == 0
    assert sg.match_flags(REF, pred)["0.5"] == [0, 1, 1]
    assert sg.match_flags(REF, pred)["0.7"] == [0, 1, 0]


def test_short_spell_recall_bucket_edges_and_empty_buckets():
    """A spell of exactly 2 s or 5 s is in the middle bucket. An empty bucket has no recall."""
    ref = _seq((H, 50), (A, 125), (D, 10))
    rows = fc.spell_recall_by_duration(ref, ref, pm.valid_runs(ref))
    assert [(r["n"], r["recalled"]) for r in rows] == [(0, 0), (2, 2), (0, 0)]
    assert rows[0]["recall"] is None and rows[2]["recall"] is None


def test_predicted_x_is_filled_before_the_recall():
    pred = REF.copy()
    pred[50:60] = X                      # inside the A spell: the fill gives A
    rows = fc.spell_recall_by_duration(REF, pred, RUNS)
    assert [r["recalled"] for r in rows] == [1, 1, 1]


def test_possession_ratio():
    assert fc.possession_ratio(17, 15) == pytest.approx(15 / 17.0)
    assert fc.possession_ratio(0, 4) is None


def test_dead_edges_collapse_the_teams_and_match_one_to_one():
    """H and A are one live state. A team change is no D edge. Latency is results - reference."""
    ref = _seq((H, 100), (D, 100), (A, 100), (H, 100), (D, 100), (H, 100))
    pred = ref.copy()
    pred[100:105] = H                    # the first entry is 5 frames late
    pred[195:200] = A                    # the first exit is 5 frames early
    pred[400:500] = H                    # the second stoppage is missed
    pred[300:350] = D                    # a spurious stoppage inside the reference H spell
    edges = fc.dead_edges(ref, pred, pm.valid_runs(ref))
    assert edges["tol"] == max(pm.TAUS)
    entry, leave = edges["entry"], edges["exit"]
    assert (entry["n_ref"], entry["n_pred"], entry["matched"]) == (2, 2, 1)
    assert (leave["n_ref"], leave["n_pred"], leave["matched"]) == (2, 2, 1)
    assert entry["precision"] == entry["recall"] == 0.5
    assert entry["mean_latency"] == 5.0 and leave["mean_latency"] == -5.0
    assert entry["median_latency"] == 5.0 and leave["mean_abs_latency"] == 5.0


def test_dead_edges_outside_the_tolerance_do_not_match():
    ref = _seq((H, 100), (D, 100), (H, 100))
    pred = ref.copy()
    pred[100:126] = H                    # 26 frames late: beyond tau 25
    edges = fc.dead_edges(ref, pred, pm.valid_runs(ref))
    assert edges["entry"]["matched"] == 0 and edges["entry"]["mean_latency"] is None
    assert edges["exit"]["matched"] == 1 and edges["exit"]["mean_latency"] == 0.0


def _decomp_case():
    """A reference and a prediction with a late hand-over, a missed spell and a spurious one."""
    ref = _seq((X, 10), (H, 600), (A, 40), (H, 260), (D, 80), (A, 300), (X, 10))
    pred = ref.copy()
    pred[610:650] = H                    # the 40-frame A spell is missed: the H spells merge
    pred[1040:1060] = H                  # a 20-frame spurious H spell inside A
    pred[990:1000] = D                   # the restart 10 frames late
    pred[60:130] = A                     # a 70-frame A excursion, 480 frames before 610
    return ref, pred


def test_decompose_reproduces_the_m19_script():
    """fc.spell_table and fc.group_edits give the classes and gains of r2_metric_decomp.py."""
    ref, pred = _decomp_case()
    runs = pm.valid_runs(ref)
    filled = pm.fill_abstentions(pred, runs)
    rows, unmatched = r2.spell_table(ref, filled, runs)
    mine, mine_unmatched = fc.spell_table(ref, filled, runs)
    assert [r["class"] for r in mine] == [r["class"] for r in rows]
    assert [r["passes"] for r in mine] == [r["passes"] for r in rows]
    assert [p["class"] for p in mine_unmatched] == [p["class"] for p in unmatched]
    edits = r2.edits(ref, filled, runs, [p for p in unmatched if p["class"].startswith("spurious")])
    dec = fc.decompose(ref, pred, runs)
    assert dec["base"] == pytest.approx(edits["base"])
    old = list(edits["groups"].values())
    new = list(dec["groups"].values())
    assert new == pytest.approx(old)
    assert sum(v["paired"] for v in dec["alignment"].values()) == len(edits["pairs"])
    assert sum(v["missed"] for v in dec["alignment"].values()) == len(edits["missed"])


def test_decompose_classes_of_the_constructed_case():
    ref, pred = _decomp_case()
    dec = fc.decompose(ref, pred, pm.valid_runs(ref))
    assert sum(dec["spell_classes"].values()) == 4
    assert dec["spell_classes"].get("missed: covered by other team") == 1
    assert dec["alignment"]["restart"] == {"paired": 1, "missed": 0, "spurious": 0}
    assert dec["groups"]["all missed insertions"] > 0.0
    assert dec["groups"]["boundary moves restart only"] == pytest.approx(0.0)


def test_decompose_of_a_perfect_prediction():
    dec = fc.decompose(REF, REF, RUNS)
    assert dec["base"] == 1.0
    assert dec["spell_classes"] == {"ok": 3}
    assert dec["unmatched_classes"] == {}
    assert all(v == 0.0 for v in dec["groups"].values())


def test_format_lines_prints_every_table():
    ref, pred = _decomp_case()
    result, gt_stats, pred_stats = sg.score_codes(ref, pred)
    report = sg.failure_report(ref, pred, gt_stats, pred_stats)
    text = "\n".join(fc.format_lines(report))
    for word in ("short-spell recall", "possession ratio", "D entry", "D exit",
                 "reference spells by class", "unmatched results spells", "alignment",
                 "group edit gains"):
        assert word in text
