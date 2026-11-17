"""tools/score_game.py: scoring a results CSV against a reference CSV of the same length.

Two tiny label arrays. The tests need no match data.
"""

import csv
import json
import os

import numpy as np
import pytest

from tools import score_game as sg

H, A, D, X = 0, 1, 2, 3
#            X  H  H  H  H  A  A  A  D  D  H  H  X
REF = np.array([X, H, H, H, H, A, A, A, D, D, H, H, X], dtype=np.int8)


def test_perfect_prediction():
    result, gt_stats, pred_stats = sg.score_codes(REF, REF)
    assert result["spell_f1_mean"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert gt_stats == pred_stats
    assert gt_stats["possessions"] == 3
    assert gt_stats["d_segments"] == 1
    assert gt_stats["d_share"] == pytest.approx(2 / 11.0)
    assert gt_stats["mean_duration_s"] == pytest.approx(3 / 25.0)


def test_late_switch_and_missing_dead_ball():
    pred = np.array([D, H, H, H, H, H, A, A, A, A, H, H, D], dtype=np.int8)
    result, gt_stats, pred_stats = sg.score_codes(REF, pred)
    head = sg.summary(result, gt_stats, pred_stats)
    assert pred_stats["possessions"] == 3
    assert pred_stats["d_segments"] == 0       # frames 0 and 12 are not graded
    assert pred_stats["d_share"] == 0.0
    assert head["matched"] + head["missed"] == 3          # H>A, A>D, D>H in the reference
    assert set(head["spell_f1"]) == {"0.3", "0.4", "0.5", "0.6", "0.7"}
    assert 0.0 < head["spell_f1_mean"] < 1.0


def test_x_is_forward_filled_for_the_statistics():
    pred = REF.copy()
    pred[3] = X
    _, _, pred_stats = sg.score_codes(REF, pred)
    assert pred_stats["possessions"] == 3


def test_paired_delta_identical_is_zero():
    pair = sg.paired_delta(REF, REF, REF, n_draws=50)
    assert pair["delta"] == 0.0 and pair["low"] == 0.0 and pair["high"] == 0.0
    assert pair["share_positive"] == 0.0


def test_paired_delta_sign_and_determinism():
    worse = np.array([X, H, H, H, H, H, H, H, D, D, H, H, X], dtype=np.int8)
    first = sg.paired_delta(REF, REF, worse, n_draws=200)
    again = sg.paired_delta(REF, REF, worse, n_draws=200)
    assert first == again
    assert first["delta"] > 0.0
    assert first["low"] <= first["delta"] <= first["high"]


def test_write_report(tmp_path):
    pred = np.array([D, H, H, H, A, A, A, A, D, D, H, H, D], dtype=np.int8)
    result, gt_stats, pred_stats = sg.score_codes(REF, pred)
    head = sg.summary(result, gt_stats, pred_stats)
    json_path, trans_path = sg.write_report(str(tmp_path), "x", "ref.csv", result, head)
    report = json.load(open(json_path))
    assert report["submission"] == "x" and report["ground_truth"] == "ref.csv"
    assert report["summary"]["reference"]["possessions"] == 3
    with open(trans_path) as handle:
        rows = list(csv.DictReader(handle))
    assert {r["status"] for r in rows} <= {"matched", "missed", "spurious"}
    assert any(r["status"] == "matched" and r["latency"] == "-1" for r in rows)


def _write(path, codes):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        for i, c in enumerate(codes):
            handle.write("{},{}\n".format(i, "HADX"[c]))


def test_main_and_length_check(tmp_path, capsys):
    ref, res, short = (str(tmp_path / n) for n in ("ref.csv", "res.csv", "short.csv"))
    _write(ref, REF)
    _write(res, REF)
    _write(short, REF[:-1])
    out_dir = str(tmp_path / "out")
    assert sg.main([res, "--gt", ref, "--out-dir", out_dir, "--versus", res, "--draws", "20"]) == 0
    assert "spellF1 1.0000" in capsys.readouterr().out
    assert os.path.isfile(os.path.join(out_dir, "res.json"))
    with pytest.raises(ValueError):
        sg.main([short, "--gt", ref, "--out-dir", out_dir])


def test_main_refuses_evaluation_reports(tmp_path):
    ref = str(tmp_path / "ref.csv")
    _write(ref, REF)
    with pytest.raises(SystemExit):
        sg.main([ref, "--gt", ref, "--out-dir",
                 os.path.join(sg.EVALUATION, "reports", "s3")])
