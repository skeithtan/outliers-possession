"""Unit tests of the round 3 scoring driver (tools/learned_spike/r3_eval.py). No match data."""

import argparse
import json
import os

import pytest

from tools.learned_spike import evaluate as ev
from tools.learned_spike import folds
from tools.learned_spike import r3_eval as r3


def test_nested_index_leaves_the_match_out_and_breaks_ties_low():
    games = list(r3.DEV)
    # Two grid points; point 1 is better everywhere except on METRICA-S1.
    scores = {g: [0.5, 0.6] for g in games}
    scores["METRICA-S1"] = [0.9, 0.1]
    i_all, _ = r3.nested_index(scores, games, 2)
    i_out, v_out = r3.nested_index(scores, games, 2, test="METRICA-S1")
    assert i_out == 1 and v_out == pytest.approx(0.6)
    # With S1 in: point 0 scores (0.633 + 0.5) / 2, point 1 (0.433 + 0.6) / 2.
    assert i_all == 0
    ties = {g: [0.5, 0.5] for g in games}
    assert r3.nested_index(ties, games, 2)[0] == 0


def test_nested_index_weighs_the_two_providers_equally():
    games = list(r3.DEV)
    scores = {g: ([0.7, 0.6] if folds.provider(g) == "metrica" else [0.5, 0.65]) for g in games}
    # point 0: (0.7 + 0.5) / 2 = 0.6; point 1: (0.6 + 0.65) / 2 = 0.625
    assert r3.nested_index(scores, games, 2)[0] == 1


def test_match_costs_nested_for_open_matches_final_for_skillcorner():
    costs = {"M_nested": {"METRICA-S1": [8.0, 4.0]}, "M_final": [2.0, 4.0],
             "H_nested": {"METRICA-S1": 4.0}, "H_final": 8.0}
    c = r3.match_costs(costs, "METRICA-S1")
    assert c["M"] == (8.0, 4.0) and c["H"] == 4.0 and c["source"] == "nested"
    sc = folds.SC[0]
    c = r3.match_costs(costs, sc)
    assert c["M"] == (2.0, 4.0) and c["H"] == 8.0 and c["source"] == "final"
    assert r3.final_costs(costs)["M"] == (2.0, 4.0)
    over = ev.overrides("M", "/w.npz", c)
    assert over["LEARNED_COST_HA"] == 2.0 and over["LEARNED_COST_AD"] == 4.0


def test_summary_names_the_baseline_c23():
    x = {"METRICA-S1": 0.8, "DFL-J03WMX": 0.6, folds.SC[0]: 0.7}
    b = {"METRICA-S1": 0.7, "DFL-J03WMX": 0.5, folds.SC[0]: 0.7}
    s = r3.summary(x, b)
    assert "c23_mean" in s and "c22_mean" not in s
    assert s["per_match"]["METRICA-S1"]["c23"] == 0.7
    assert all("c23_mean" in r for r in s["per_provider"].values())
    assert s["delta"] == pytest.approx((0.1 + 0.1 + 0.0) / 3)


def test_criteria_1_and_2_record_only_3_blocks():
    good = {"delta": 0.1, "low": 0.05, "high": 0.15,
            "per_provider": {p: {"delta": 0.1, "wins": 3, "n": 3} for p in ev.PROVIDERS}}
    bad = dict(good, low=-0.01)
    assert r3.crit1(bad, good)["blocking"] is False and r3.crit1(bad, good)["pass"] is False
    assert r3.crit2(good)["blocking"] is False
    c3 = r3.crit3(bad, good)
    assert c3["blocking"] is True and c3["pass"] is False
    assert c3["recorded_metrica_dfl_only"] == [0.1, 0.05, 0.15]


def crits(**over):
    out = {str(i): {"pass": True} for i in range(1, 11)}
    out.update(over)
    return out


def test_verdict_two_way_choice():
    assert r3.verdict(crits(), True)[0] == "M"
    assert r3.verdict(crits(), False)[0] == "c23"
    # 1 and 2 are recorded only: their failure does not stop M.
    assert r3.verdict(crits(**{"1": {"pass": False}, "2": {"pass": False}}), True)[0] == "M"
    choice, reason = r3.verdict(crits(**{"4": {"pass": False}}), True)
    assert choice == "c23" and "4" in reason
    choice, reason = r3.verdict(crits(**{"8": None}), True, stop="fresh clone pending")
    assert choice is None and "8" in reason
    # A failure decides even when a later criterion is missing.
    assert r3.verdict(crits(**{"3": {"pass": False}, "8": None}), True)[0] == "c23"


def bar_record(seed_bar=0.04, sc_fold=0.03, p50=8.0, agree=0.95):
    env = {"possession_s.p50": {"bar": p50, "ext_min": 3.0, "ext_max": 11.2,
                                "inside": 3.0 <= p50 <= 11.2},
           "possession_s.p90": {"bar": 30.0, "ext_min": 12.08, "ext_max": 44.96, "inside": True},
           "d_segment_s.p50": {"bar": 20.0, "ext_min": 12.32, "ext_max": 23.84, "inside": True},
           "d_segment_s.p90": {"bar": 50.0, "ext_min": 40.8, "ext_max": 66.34, "inside": True}}
    return {"disagreement": {"seed": {"bar": seed_bar, "sc_mean_fold": sc_fold},
                             "copy": {"bar": 0.01, "sc_mean_fold": 0.01}},
            "features": {"outside": []}, "envelope_provided": env,
            "agreement_final_vs_fold": agree,
            "systems": {"M": {"d_share_ball_missing": 0.8}, "c23": {"d_share_ball_missing": 0.9}}}


def test_bounds8_limits():
    assert r3.bounds8(bar_record())["pass"] is True
    assert r3.bounds8(bar_record(seed_bar=0.046))["a"]["pass"] is False      # > 1.5 x 0.03
    assert r3.bounds8(bar_record(p50=12.5))["e"]["pass"] is False
    assert r3.bounds8(bar_record(agree=0.89))["f"]["pass"] is False
    b = bar_record()
    b["features"]["outside"] = ["H_nvis"]
    assert r3.bounds8(b)["c"]["pass"] is False
    ok = r3.bounds8(b, {"H_nvis": {"explanation": "x", "covered_by_stress": True}})
    assert ok["c"]["pass"] is True


def test_v2_v3_bounds_against_c23():
    v = {"v2": {"systems": {"c23": {"flip": 0.50, "alive_raw": 10},
                            "M": {"flip": 0.61, "alive_raw": 10}}, "k_frames": 20},
         "v3": {"c23": 0.90, "M": 0.93}}
    assert r3.crit9(v)["pass"] is False and r3.crit10(v)["pass"] is False
    v["v2"]["systems"]["M"]["flip"] = 0.60
    v["v3"]["M"] = 0.92
    assert r3.crit9(v)["pass"] is True and r3.crit10(v)["pass"] is True


def model_record(k, **over):
    spec = folds.fold(k, pff=False)
    rec = {"seed": 0, "steps": 2000, "train": list(spec["train"]),
           "switches": {"target": "raw", "short_weight": "off", "mix": "sc_quarter"}}
    rec.update(over)
    return rec


def test_model_record_checks_r3b_rules():
    assert r3.check_model_record(model_record(0), 0) == []
    bad = model_record(0, train=list(folds.fold(0, pff=False)["train"]) + ["PFF-3814"])
    assert any("PFF" in p for p in r3.check_model_record(bad, 0))
    leak = model_record(0, train=list(folds.fold(0, pff=False)["train"]) + ["METRICA-S1"])
    assert any("held-out" in p for p in r3.check_model_record(leak, 0))
    wrong = model_record(0, switches={"target": "ctl2", "short_weight": "off", "mix": "match"})
    assert len(r3.check_model_record(wrong, 0)) == 2


def test_label_free_match_refuses_sealed_and_competition_as_rehearsal(tmp_path):
    ns = argparse.Namespace(work=str(tmp_path), run=str(tmp_path), refs_root="~",
                            sample_root="~", mtah_root="~")
    ctx = r3.make_ctx(ns)
    for game in ("DFL-J03WQQ", "DFL-J03WR9", "BAR-ATH", folds.SC[0]):
        with pytest.raises(SystemExit):
            r3.Match(ctx, game)
    m = r3.Match(ctx, "METRICA-S2")
    assert m.rehearsal and m.game == "METRICA-S2" and m.folder.endswith("bar_as_METRICA-S2")
    assert ctx.args.c22_ext is None          # a c22 run is never copied as the baseline


def test_matched_view_job_has_its_own_cache_and_data_root(tmp_path):
    ns = argparse.Namespace(work=str(tmp_path), run=str(tmp_path), refs_root="/refs",
                            sample_root="/sample", mtah_root="/mtah")
    ctx = r3.make_ctx(ns)
    job = ev.make_job(ctx, "METRICA-S1", "c22", "c23", ev.VIEW_MATCHED, None,
                      {"M": (1.0, 1.0), "H": 1.0, "source": "x"})
    raw = ev.make_job(ctx, "METRICA-S1", "c22", "c23", ev.VIEW_RAW, None,
                      {"M": (1.0, 1.0), "H": 1.0, "source": "x"})
    assert job["hidden"] is None and job["engine_cache"].endswith(
        os.path.join("views", "matched", "engine_data"))
    assert job["data_dir"].startswith("/mtah") and raw["data_dir"] == "/refs"
    assert job["key"] != raw["key"] and raw["engine_cache"] is None


def test_gate_reads_the_files_of_the_work_folder(tmp_path):
    good = {"delta": 0.1, "low": 0.05, "high": 0.15,
            "per_provider": {p: {"delta": 0.1, "wins": 3, "n": 3} for p in ev.PROVIDERS}}
    sm = {"M|nested|provided|0": dict(good, low=-0.001), "M|nested|ctl2|0": good,
          "M|final|ctl2|0": good}
    (tmp_path / "scores.json").write_text(json.dumps({"summaries": sm}))
    assert r3.main(["gate", "--work", str(tmp_path), "--criteria", "3"]) == 3
    assert r3.main(["gate", "--work", str(tmp_path), "--criteria", "4"]) == 4
    sm["M|nested|provided|0"] = good
    (tmp_path / "scores.json").write_text(json.dumps({"summaries": sm}))
    assert r3.main(["gate", "--work", str(tmp_path), "--criteria", "3"]) == 0


def test_label_free_costs_use_rule_b_and_record_the_frozen_pair(tmp_path):
    costs = {"M_nested": {"METRICA-S1": [8.0, 4.0]}, "M_final": [8.0, 4.0],
             "H_nested": {}, "H_final": 8.0}
    (tmp_path / "costs_r3.json").write_text(json.dumps(costs))
    with pytest.raises(SystemExit):
        r3.label_free_costs(str(tmp_path), "b", True)          # views.json missing
    from tools.learned_spike import r3_views
    i = r3_views.GRID.index((32.0, 16.0))
    views = {"grid": r3_views.GRID, "rules": {"b": {"final": i, "final_score": 0.05},
                                              "c": {"final": 0, "final_score": 0.1}}}
    (tmp_path / "views.json").write_text(json.dumps(views))
    c = r3.label_free_costs(str(tmp_path), "b", True)
    assert c["M"] == (32.0, 16.0) and c["frozen_rule_pair"] == [8.0, 4.0]
    over = ev.overrides("M", "/w.npz", c)
    assert over["LEARNED_STRUCTURE"] is True and over["LEARNED_COST_HA"] == 32.0
    a = r3.label_free_costs(str(tmp_path), "a", False)
    assert a["M"] == (8.0, 4.0) and "LEARNED_STRUCTURE" not in ev.overrides("M", "/w", a)


def test_summaries_skip_shift_rows_that_were_not_scored():
    row = lambda v: {"ctl2": {"0": {"spell_f1": v}}, "provided": {"0": {"spell_f1": v}}}  # noqa
    scores = {g: {"c23": row(0.5), "M": row(0.6)} for g in ("METRICA-S1", "DFL-J03WMX")}
    sm = r3.summaries(scores, list(scores), ["M"])
    assert set(sm) == {"M|nested|ctl2|0", "M|nested|provided|0"}
