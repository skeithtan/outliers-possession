"""tools/paired_summary.py: the cross-match paired summary of score_game reports (P9-D).

The tests write tiny references and results, score them with tools/score_game.py into a
temporary folder, and read the reports back. The tests need no match data.
"""

import json
import os

import numpy as np
import pytest

from tools import paired_summary as ps
from tools import score_game as sg

H, A, D, X = 0, 1, 2, 3


def _seq(*runs):
    return np.concatenate([np.full(n, k, dtype=np.int8) for k, n in runs])


def _write(path, codes):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        for i, c in enumerate(codes):
            handle.write("{},{}\n".format(i, "HADX"[c]))
    return path


def _match(tmp_path, name, ref, good, bad):
    """Score `good` and `bad` against `ref`. Return the two report paths."""
    ref_path = _write(str(tmp_path / (name + "_ref.csv")), ref)
    out = []
    for version, codes in (("good", good), ("bad", bad)):
        res = _write(str(tmp_path / "{}_{}.csv".format(name, version)), codes)
        out_dir = str(tmp_path / version)
        assert sg.main([res, "--gt", ref_path, "--out-dir", out_dir, "--name", name]) == 0
        out.append(os.path.join(out_dir, name + ".json"))
    return out


REF_1 = _seq((X, 5), (H, 60), (A, 30), (H, 20), (D, 40), (A, 80), (H, 50), (X, 5))
REF_2 = _seq((H, 90), (A, 15), (H, 70), (D, 30), (H, 40), (A, 60))


def _late(ref, k):
    """`ref` with every label change k frames late."""
    out = ref.copy()
    out[k:] = ref[:-k]
    out[:k] = ref[0]
    return out


def test_report_holds_the_flags_and_the_failure_tables(tmp_path):
    good, _ = _match(tmp_path, "one", REF_1, REF_1, _late(REF_1, 12))
    report = json.load(open(good))
    assert sorted(report["spell_match_flags"]) == ["0.3", "0.4", "0.5", "0.6", "0.7"]
    assert all(v == [1] * 5 for v in report["spell_match_flags"].values())
    tables = report["summary"]["failure_classes"]
    assert sorted(tables) == ["d_edges", "decomposition", "possession_ratio", "recall_iou",
                              "short_spell_recall"]
    assert tables["possession_ratio"] == 1.0


def test_one_match_gives_the_paired_delta_of_score_game(tmp_path):
    """With one match, the draws are those of score_game.paired_delta (same seed, same order)."""
    good, bad = _match(tmp_path, "one", REF_1, REF_1, _late(REF_1, 12))
    summary = ps.paired_summary(ps.by_reference([good]), ps.by_reference([bad]), 300)
    pair = sg.paired_delta(REF_1, REF_1, _late(REF_1, 12), n_draws=300)
    (row,) = summary["matches"]
    assert row["delta"] == pytest.approx(pair["delta"])
    assert summary["mean_delta"] == pytest.approx(pair["delta"])
    assert (summary["low"], summary["high"]) == pytest.approx((pair["low"], pair["high"]))
    assert (row["low"], row["high"]) == pytest.approx((pair["low"], pair["high"]))
    assert summary["share_positive"] == pytest.approx(pair["share_positive"])


def test_two_matches_average_the_per_match_differences(tmp_path):
    one = _match(tmp_path, "one", REF_1, REF_1, _late(REF_1, 12))
    two = _match(tmp_path, "two", REF_2, _late(REF_2, 3), _late(REF_2, 20))
    side_a = ps.by_reference([str(tmp_path / "good")])
    side_b = ps.by_reference([one[1], two[1]])
    summary = ps.paired_summary(side_a, side_b, 200)
    deltas = [m["delta"] for m in summary["matches"]]
    assert [m["reference"] for m in summary["matches"]] == ["one_ref.csv", "two_ref.csv"]
    assert summary["mean_delta"] == pytest.approx(np.mean(deltas))
    assert all(d > 0 for d in deltas)
    assert summary["low"] <= summary["mean_delta"] <= summary["high"]
    assert ps.paired_summary(side_a, side_b, 200) == summary          # deterministic
    lines = ps.format_lines(summary)
    assert len(lines) == 4 and "mean of 2 per-match differences" in lines[-1]


def test_main_prints_the_summary(tmp_path, capsys):
    _match(tmp_path, "one", REF_1, REF_1, _late(REF_1, 12))
    assert ps.main(["--a", str(tmp_path / "good"), "--b", str(tmp_path / "bad"),
                    "--draws", "50"]) == 0
    assert "one_ref.csv" in capsys.readouterr().out


def test_mismatched_references_are_refused(tmp_path):
    one = _match(tmp_path, "one", REF_1, REF_1, REF_1)
    two = _match(tmp_path, "two", REF_2, REF_2, REF_2)
    with pytest.raises(ValueError):
        ps.paired_summary(ps.by_reference([one[0], two[0]]), ps.by_reference([one[1]]), 10)
    with pytest.raises(ValueError):
        ps.by_reference([one[0], one[0]])


def test_old_report_without_flags_is_refused(tmp_path):
    path = str(tmp_path / "old.json")
    with open(path, "w") as handle:
        json.dump({"ground_truth": "r.csv", "n_pred_spells": 3, "submission": "x"}, handle)
    with pytest.raises(ValueError):
        ps.load(path)
