"""Round 5 final scoring pieces on synthetic data: r3_eval variant R5 (feature version on the
model records, G7's capped limits, the ratio record), r5_final.sh (dry run), r5_step0's
breakdown, and the feature version of the probability dumps (evaluate.member_probs)."""

import json
import os
import subprocess

import numpy as np
import pytest

from tools.learned_spike import evaluate as ev
from tools.learned_spike import r3_eval as r3
from tools.learned_spike import r5_step0

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def r5_switches(**r4over):
    r4 = {"views": ["raw"], "dfl_still": "removed", "val_cache": "/v", "sc_alive_dead": True,
          "feature_version": "stage-v3"}
    r4.update(r4over)
    return {"target": "raw", "short_weight": "off", "mix": "sc_quarter",
            "short_median_bins": None,
            "aug": {"cache": "/x", "k": 16, "share": 0.5, "manifest_md5": "abc"}, "r4": r4}


def test_r5_model_records_need_the_feature_version():
    assert r3.check_r4_switches(r5_switches(), "R5") == []
    assert any("feature_version" in p
               for p in r3.check_r4_switches(r5_switches(feature_version=None), "R5"))
    assert any("feature_version" in p for p in r3.check_r4_switches(
        r5_switches(feature_version="stage50-v2"), "R5"))
    # a round 4 model is not a round 5 model, and R4 rejects a feature-version record
    sw = r5_switches()
    del sw["r4"]["feature_version"]
    assert r3.check_r4_switches(sw, "R4") == []
    assert any("feature_version" in p for p in r3.check_r4_switches(sw, "R5"))
    assert any("unknown round 4" in p for p in r3.check_r4_switches(r5_switches(), "R4"))
    rec = {"switches": r5_switches(), "steps": 2000, "train": []}
    assert r3.check_model_record(rec, None, "R5") == []
    assert "R5" in r3.VARIANTS and r3.result_name("R5") == "LEARNED_STAGE_RESULT_R5"


def test_r5_limits_are_the_lower_of_the_two():
    assert r3.limit8("a", 0.04, "R5") == pytest.approx(0.06)          # 1.5 x 0.04 < 0.0717
    assert r3.limit8("a", 0.06, "R5") == pytest.approx(0.0717)        # capped
    assert r3.limit8("b", 0.01, "R5") == pytest.approx(0.015)
    assert r3.limit8("b", 0.03, "R5") == pytest.approx(0.0289)
    assert r3.limit8("a", 0.06, "R4") == pytest.approx(0.09)          # R4 unchanged
    assert r3.limit8("a", 0.06) == pytest.approx(0.09)


def _bar(seed_bar, seed_sc, copy_bar, copy_sc):
    return {"disagreement": {"seed": {"bar": seed_bar, "sc_mean_fold": seed_sc},
                             "copy": {"bar": copy_bar, "sc_mean_fold": copy_sc}},
            "features": {"outside": []}, "envelope_provided": {},
            "agreement_final_vs_fold": 1.0,
            "systems": {"M": {}, "c23": {}}}


def test_bounds8_r5_caps_and_records_the_ratio(monkeypatch):
    monkeypatch.setattr(ev, "ENVELOPE_KEYS", {"M": []})
    # round 4's competition-match numbers against a SkillCorner mean of 0.0478 / 0.0193
    b = r3.bounds8(_bar(0.0754, 0.0478, 0.0303, 0.0193), variant="R5")
    assert b["a"]["limit"] == pytest.approx(0.0717) and not b["a"]["pass"]
    assert b["a"]["limit_uncapped"] == pytest.approx(1.5 * 0.0478)
    assert b["a"]["ratio_to_sc_fold_mean"] == pytest.approx(0.0754 / 0.0478)
    assert b["b"]["limit"] == pytest.approx(0.0289) and not b["b"]["pass"]
    b = r3.bounds8(_bar(0.050, 0.040, 0.020, 0.015), variant="R5")
    assert b["a"]["limit"] == pytest.approx(0.060) and b["a"]["pass"]
    assert b["b"]["limit"] == pytest.approx(0.0225) and b["b"]["pass"]
    b4 = r3.bounds8(_bar(0.0754, 0.0478, 0.0303, 0.0193))
    assert "ratio_to_sc_fold_mean" not in b4["a"] and b4["a"]["limit"] == pytest.approx(0.0717)


def test_r5_final_dry_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    ca = tmp_path / "ca"
    ca.mkdir()
    env = dict(os.environ, CA=str(ca), MTAH_DATA_DIR=str(tmp_path / "d"),
               C23_DEV=str(tmp_path / "c"), PAIR="12,12", PRIOR="0", OOF_RATIO="/x.json",
               DRY_RUN="1", REHEARSAL="1", AS_MATCH="DFL-J03WMX")
    env.pop("VARIANT", None)          # the default variant R5 (final_R6.sh exports VARIANT=R6)
    for k in ("STOP_BEFORE_LABEL_FREE", "R3_EVAL_REHEARSAL_GAMES", "REPRO_GAMES"):
        env.pop(k, None)
    sh = os.path.join(ROOT, "tools", "learned_spike", "r5_final.sh")
    res = subprocess.run(["bash", sh, str(run), str(tmp_path / "out")], env=env, cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    text = res.stdout.decode()
    assert res.returncode == 0, text
    assert "DRY RUN" in text and "--identity" not in text and "--variant R5" in text
    assert "learned_v5.npz" in text and "learned_v4" not in text
    assert "r5_step0" in text and "--as-match DFL-J03WMX" in text and "stage-v3" in text
    assert not (tmp_path / "out" / "work" / "preflight.json").exists()
    env.pop("REHEARSAL")
    env.pop("AS_MATCH")
    env["STOP_BEFORE_LABEL_FREE"] = "1"
    res = subprocess.run(["bash", sh, str(run), str(tmp_path / "out2")], env=env, cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    text = res.stdout.decode()
    assert res.returncode == 0 and "--identity" not in text and "r5_step0" not in text  # identity: label-free call only
    assert "STOP_BEFORE_LABEL_FREE" in text


def _stats(graded, rows):
    """A match_stats-like dict: rows {name: (frames, seed_dis, copy_dis)}."""
    full = {k: (0, 0, 0) for k in r5_step0.ROWS}
    full.update(rows)
    return {"graded": graded,
            "seed_total": full["all"][1] / graded, "copy_total": full["all"][2] / graded,
            "stats": {k: {"frames": f, "seed_dis": s, "copy_dis": c}
                      for k, (f, s, c) in full.items()}}


def test_step0_breakdown_contributions():
    bar = _stats(100, {"all": (100, 10, 4), "track_only": (60, 8, 3), "neither": (40, 2, 1)})
    sc = [_stats(100, {"all": (100, 5, 2), "track_only": (30, 2, 1), "neither": (70, 3, 1)}),
          _stats(100, {"all": (100, 5, 2), "track_only": (30, 2, 1), "neither": (70, 3, 1)})]
    tables, contrib = r5_step0.breakdown(bar, sc)
    c = contrib["seed"]
    assert c["excess"] == pytest.approx(0.05) and c["ratio_to_sc_mean"] == pytest.approx(2.0)
    assert c["rows"]["track_only"]["contribution"] == pytest.approx(0.06)
    assert c["rows"]["neither"]["contribution"] == pytest.approx(-0.01)
    assert c["track_cells_share_of_excess"] == pytest.approx(1.2)
    assert tables["match"]["track_only"]["seed_rate_pooled"] == pytest.approx(8 / 60)
    md = r5_step0.part1_md({"match": "X", "rehearsal": True, "feature_version": "stage-v3",
                            "band_games": [], "feature_cache": "/c", "tables": tables,
                            "contributions": contrib})
    assert "REHEARSAL" in md and "track_only" in md
    assert r5_step0.SPREAD["stage-v3"] == ("H_iqrx", "H_iqry", "A_iqrx", "A_iqry")


def test_member_probs_read_the_weights_version(monkeypatch, tmp_path):
    from engine import data, learned
    from engine import learned_features as lf
    from tests.test_learned_v3 import _match, _random_state
    from tools.learned_spike import export
    monkeypatch.setattr(data, "graded_blocks", lambda game=None: [(1, 0, 1179), (2, 1220, 2399)])
    monkeypatch.setattr(data, "pitch_m", lambda game=None: (105.0, 68.0))
    tac = _match()
    path = str(tmp_path / "w.npz")
    export.write(path, [_random_state(55, seed=s) for s in range(2)])
    nets = learned.load_weights(path)
    probs = ev.member_probs(nets, tac, "METRICA-S3")
    want = lf.downsample(lf.frame_features(tac, "METRICA-S3", version="stage-v3"))
    assert probs.shape == (2, 4, want.shape[0], 3)
    assert np.array_equal(probs[0], learned.copy_probabilities(nets[0],
                                                                learned.symmetric_copies(want)))


# -- round 6: variant R6 (stage50-v2s; G7 recorded, not blocking) --------------------------------

def test_r6_records_and_limits():
    sw = r5_switches(feature_version="stage50-v2s")
    assert r3.check_r4_switches(sw, "R6") == []
    assert any("feature_version" in p for p in r3.check_r4_switches(r5_switches(), "R6"))
    assert r3.limit8("a", 0.06, "R6") == pytest.approx(0.0717)
    assert r3.limit8("b", 0.01, "R6") == pytest.approx(0.015)
    assert r3.result_name("R6") == "LEARNED_STAGE_RESULT_R6"
    assert r3.blocking_of("R6") == ("5", "8") and r3.blocking_of("R5") == r3.R4_BLOCKING


def test_r6_verdict_records_g7():
    ok, bad = {"pass": True}, {"pass": False}
    crits = {"5": ok, "6": ok, "7": bad, "7x": ok, "8": ok}
    choice, reason = r3.verdict(crits, True, blocking=r3.blocking_of("R6"))
    assert choice == "M" and "G7: fail (recorded; the owner decides)" in reason
    assert r3.verdict(crits, True, blocking=r3.blocking_of("R5"))[0] == r3.BASE
    choice, reason = r3.verdict(dict(crits, **{"8": bad}), True, blocking=r3.blocking_of("R6"))
    assert choice == r3.BASE
    choice, reason = r3.verdict(dict(crits, **{"7": ok}), True, blocking=r3.blocking_of("R6"))
    assert "G7: pass" in reason
    md = r3.result_markdown("/w", crits, "M", "x", {}, True, None, "R6")
    assert "G7: fail (recorded; the owner decides)" in md and "LEARNED_STAGE_RESULT_R6" in md


def test_r6_final_dry_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    ca = tmp_path / "ca"
    ca.mkdir()
    env = dict(os.environ, CA=str(ca), MTAH_DATA_DIR=str(tmp_path / "d"),
               C23_DEV=str(tmp_path / "c"), PAIR="12,12", PRIOR="0", OOF_RATIO="/x.json",
               DRY_RUN="1", REHEARSAL="1", AS_MATCH="DFL-J03WMX", VARIANT="R6")
    env.pop("STOP_BEFORE_LABEL_FREE", None)
    sh = os.path.join(ROOT, "tools", "learned_spike", "r5_final.sh")
    res = subprocess.run(["bash", sh, str(run), str(tmp_path / "out")], env=env, cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    text = res.stdout.decode()
    assert res.returncode == 0, text
    assert "--variant R6" in text and "learned_v6.npz" in text and "learned_v5" not in text
    assert "stage50-v2s" in text and "goes on to G8" in text and "repro_check" in text
