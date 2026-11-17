"""Unit tests of the round 3 tools added on learned-r3-tools: the R3c variant of the preflight,
the Section 12a walk (r3_eval.py pair12a, --pair) and the sealed-match judge (r3_sealed.py).
No match data: tiny fake run folders and synthetic arrays."""

import argparse
import json
import os

import numpy as np
import pytest

from tools.learned_spike import folds
from tools.learned_spike import r3_eval as r3

AUG = {"cache": "/x/r3aug_cache", "k": 16, "share": 0.5, "manifest_md5": "866095b3"}


# -- Task 1: the variant of the preflight ---------------------------------------------------------

def switches(aug=None):
    out = {"target": "raw", "short_weight": "off", "mix": "sc_quarter",
           "short_median_bins": None}
    if aug is not None:
        out["aug"] = dict(aug)
    return out


def fake_run(root, aug=None, final_aug="same", best_step=1200):
    """A finished r3_launch.sh folder in miniature: 32 fold models, 4 final models, interim check,
    nested costs and 4-seed out-of-fold files (the nets are a few distinct bytes)."""
    for k in range(folds.N_FOLDS):
        d = os.path.join(root, "folds", "fold{}".format(k))
        os.makedirs(d)
        for s in r3.SEEDS:
            with open(os.path.join(d, "net_s{}.pt".format(s)), "wb") as handle:
                handle.write("fold{} seed{}".format(k, s).encode())
            rec = {"seed": s, "steps": r3.STEPS, "best_step": best_step,
                   "train": list(folds.fold(k, pff=False)["train"]), "switches": switches(aug),
                   "environment": {"git_commit": None}}
            with open(os.path.join(d, "model_s{}.json".format(s)), "w") as handle:
                json.dump(rec, handle)
    d = os.path.join(root, "final")
    os.makedirs(d)
    faug = aug if final_aug == "same" else final_aug
    for s in r3.SEEDS:
        with open(os.path.join(d, "net_s{}.pt".format(s)), "wb") as handle:
            handle.write("final seed{}".format(s).encode())
        rec = {"seed": s, "steps": r3.STEPS, "stop_at": folds.final_steps([best_step]),
               "train": list(folds.train_games(pff=False)), "switches": switches(faug),
               "environment": {"git_commit": None}}
        with open(os.path.join(d, "final_s{}.json".format(s)), "w") as handle:
            json.dump(rec, handle)
    with open(os.path.join(d, "nested_costs.json"), "w") as handle:
        json.dump({"final_costs": [8.0, 0.5]}, handle)
    os.makedirs(os.path.join(root, "interim"))
    with open(os.path.join(root, "interim", "check.json"), "w") as handle:
        json.dump({"kill_check": {"passed": True}}, handle)
    os.makedirs(os.path.join(root, "oof_p5"))
    for game in r3.DEV:
        np.savez(os.path.join(root, "oof_p5", game + "_p5.npz"), p5=np.zeros((2, 3)),
                 seeds=np.array(r3.SEEDS))
    return root


def preflight(run, work, variant):
    return r3.main(["preflight", "--run", str(run), "--work", str(work), "--variant", variant])


def test_preflight_r3b_accepts_an_r3b_run_and_rejects_an_r3c_run(tmp_path):
    b = fake_run(str(tmp_path / "r3b"))
    assert preflight(b, tmp_path / "wb", "R3B") == 0
    rec = json.load(open(str(tmp_path / "wb" / "preflight.json")))
    assert rec["variant"] == "R3B" and rec["problems"] == [] and rec["pass"] is True
    c = fake_run(str(tmp_path / "r3c"), aug=AUG)
    assert preflight(c, tmp_path / "wc", "R3B") == 2
    rec = json.load(open(str(tmp_path / "wc" / "preflight.json")))
    assert len(rec["problems"]) == 36 and all("aug" in p for p in rec["problems"])


def test_preflight_r3c_accepts_an_r3c_run_and_rejects_an_r3b_run(tmp_path):
    c = fake_run(str(tmp_path / "r3c"), aug=AUG)
    assert preflight(c, tmp_path / "wc", "R3C") == 0
    rec = json.load(open(str(tmp_path / "wc" / "preflight.json")))
    assert rec["variant"] == "R3C" and rec["aug_manifest_md5"] == ["866095b3"]
    b = fake_run(str(tmp_path / "r3b"))
    assert preflight(b, tmp_path / "wb", "R3C") == 2
    rec = json.load(open(str(tmp_path / "wb" / "preflight.json")))
    assert all("aug is missing" in p for p in rec["problems"])


def test_preflight_r3c_rejects_mixed_aug_caches_and_wrong_aug_values(tmp_path):
    other = dict(AUG, manifest_md5="ffff")
    c = fake_run(str(tmp_path / "mixed"), aug=AUG, final_aug=other)
    assert preflight(c, tmp_path / "w1", "R3C") == 2
    rec = json.load(open(str(tmp_path / "w1" / "preflight.json")))
    assert any("different aug caches" in p for p in rec["problems"])
    k8 = fake_run(str(tmp_path / "k8"), aug=dict(AUG, k=8))
    assert preflight(k8, tmp_path / "w2", "R3C") == 2
    rec = json.load(open(str(tmp_path / "w2" / "preflight.json")))
    assert all("aug.k = 8" in p for p in rec["problems"])


def test_model_record_rejects_unknown_switches_in_both_variants():
    rec = {"seed": 0, "steps": r3.STEPS, "train": list(folds.fold(0, pff=False)["train"]),
           "switches": dict(switches(AUG), extra=1)}
    assert any("unknown switches" in p for p in r3.check_model_record(rec, 0, "R3C"))
    rec["switches"] = dict(switches(), extra=1)
    assert any("unknown switches" in p for p in r3.check_model_record(rec, 0, "R3B"))
    with pytest.raises(ValueError):
        r3.check_model_record(rec, 0, "R3D")


def test_decide_names_the_result_after_the_variant(tmp_path):
    work, out = tmp_path / "w", tmp_path / "o"
    work.mkdir()
    (work / "preflight.json").write_text(json.dumps({
        "variant": "R3C", "run": "/r", "problems": [], "commits": [], "aug_manifest_md5": ["m"],
        "kill_check_interim": {"passed": True}}))
    assert r3.main(["decide", "--work", str(work), "--out-dir", str(out)]) == 0
    md = (out / "LEARNED_STAGE_RESULT_R3C.md").read_text()
    assert md.startswith("# LEARNED_STAGE_RESULT_R3C: round 3, variant R3c")
    assert "R3_DEVIATION_AUG.md Section 13" in md
    rec = json.load(open(str(out / "LEARNED_STAGE_RESULT_R3C.json")))
    assert rec["variant"] == "R3C" and "R3c" in rec["disclosure"]
    assert not (out / "LEARNED_STAGE_RESULT_R3B.md").exists()
    with pytest.raises(SystemExit):
        r3.main(["decide", "--work", str(work), "--out-dir", str(out), "--variant", "R3B"])


# -- Task 2: Section 12a --------------------------------------------------------------------------

def views_with(scores):
    """A views.json whose rule b Score of grid pair i is scores[i] (Metrica gain on one cell)."""
    from tools.learned_spike import r3_views as rv
    n = len(rv.GRID)
    tab = {v: {i: {} for i in rv.INPUTS} for v in rv.VIEWS}
    base = {v: {i: {} for i in rv.INPUTS} for v in rv.VIEWS}
    for v in rv.VIEWS:
        for inp in rv.INPUTS:
            for g in r3.DEV:
                base[v][inp][g] = 0.5
                tab[v][inp][g] = [0.5 + 1.0 for _ in range(n)]
    for i, sc in enumerate(scores):
        for g in rv.METRICA:
            tab["loose"]["raw"][g][i] = 0.5 + sc          # the minimum cell is this one
    return {"grid": [list(p) for p in rv.GRID], "spell_f1": tab, "c23": base}


def test_score_ranking_breaks_ties_by_lower_ha_then_lower_ad():
    from tools.learned_spike import r3_views as rv
    scores = [0.0] * len(rv.GRID)
    i_hi = rv.GRID.index((64.0, 16.0))
    i_tie_a, i_tie_b = rv.GRID.index((32.0, 16.0)), rv.GRID.index((16.0, 32.0))
    scores[i_hi] = 0.05
    scores[i_tie_a] = scores[i_tie_b] = 0.03
    ranking = r3.score_ranking(views_with(scores))
    assert [i for i, _ in ranking[:3]] == [i_hi, i_tie_b, i_tie_a]
    assert abs(ranking[0][1] - 0.05) < 1e-12
    # the first of the ranking is r3_views' rule b pick
    v = views_with(scores)
    assert rv.apply_rule(v["spell_f1"], v["c23"], "b")["final"] == ranking[0][0]


def test_walk_takes_the_first_positive_pair_inside_8e_and_never_a_nonpositive_one():
    grid = [(1.0, 1.0), (2.0, 1.0), (4.0, 1.0), (8.0, 1.0)]
    ranking = [(3, 0.05), (2, 0.04), (1, 0.02), (0, -0.01)]
    inside = {(8.0, 1.0): False, (4.0, 1.0): False, (2.0, 1.0): True, (1.0, 1.0): True}
    seen = []

    def decode_stats(pair):
        seen.append(pair)
        return {"possession_s.p50": {"bar": pair[0]}}, inside[pair]
    rows, chosen = r3.walk_12a(ranking, grid, decode_stats)
    assert chosen["pair"] == [2.0, 1.0] and chosen["rank"] == 3
    assert seen == [(8.0, 1.0), (4.0, 1.0), (2.0, 1.0)]       # stops at the chosen pair
    assert rows[3]["decoded"] is False and "never used" in rows[3]["note"]
    inside[(2.0, 1.0)] = False
    rows, chosen = r3.walk_12a(ranking, grid, decode_stats)
    assert chosen is None                                      # (1, 1) has Score <= 0
    rows, chosen = r3.walk_12a(ranking, grid, decode_stats, full=True)
    assert chosen is None and sum(r["decoded"] for r in rows) == 3


def bar_record(p50=12.5, seed_bar=0.04, agree=0.95, possessions=150):
    env = {"possession_s.p50": {"bar": p50, "ext_min": 3.0, "ext_max": 11.2,
                                "inside": 3.0 <= p50 <= 11.2},
           "possession_s.p90": {"bar": 30.0, "ext_min": 12.0, "ext_max": 45.0, "inside": True},
           "d_segment_s.p50": {"bar": 20.0, "ext_min": 12.0, "ext_max": 24.0, "inside": True},
           "d_segment_s.p90": {"bar": 50.0, "ext_min": 40.0, "ext_max": 66.0, "inside": True}}
    stats = {"possessions": possessions, "d_share": 0.45, "graded_x": 0,
             "possession_s": {"p50": p50}, "d_segment_s": {"p50": 20.0},
             "d_share_ball_missing": 0.8}
    return {"disagreement": {"seed": {"bar": seed_bar, "sc_mean_fold": 0.03},
                             "copy": {"bar": 0.01, "sc_mean_fold": 0.01}},
            "features": {"outside": []}, "envelope_provided": env,
            "agreement_final_vs_fold": agree,
            "systems": {"M": stats, "c23": dict(stats, d_share_ball_missing=0.9)}}


def test_12a_applies_only_when_8e_alone_fails(monkeypatch):
    monkeypatch.setattr(r3, "crit6", lambda s, limits=None: {"pass": s["possessions"] >= 100})
    assert r3.applies_12a(bar_record())[0] is True
    assert r3.applies_12a(bar_record(p50=8.0))[0] is False              # 8e passes
    ok, why = r3.applies_12a(bar_record(seed_bar=0.05))                  # 8a fails too
    assert not ok and "8a" in why
    ok, why = r3.applies_12a(bar_record(agree=0.8))                      # 8f fails too
    assert not ok and "8f" in why
    ok, why = r3.applies_12a(bar_record(possessions=50))                 # criterion 6 fails
    assert not ok and "criterion 6" in why


def test_ensemble_from_dumps_is_the_rounded_mean_of_the_seeds(tmp_path):
    from engine import learned
    rng = np.random.RandomState(0)
    p = rng.dirichlet(np.ones(3), size=(4, 4, 6))      # seeds x copies x bins x 3
    paths = []
    for i in range(4):
        paths.append(str(tmp_path / "seed_{}.npz".format(i)))
        np.savez(paths[-1], p5=p[i].mean(axis=0))
    out = r3.ensemble_from_dumps(paths, 30)
    engine = learned.upsample(np.round(p.reshape(16, 6, 3).sum(axis=0) / 16.0, 4), 30)
    assert np.allclose(out, engine, atol=1e-12)


def test_pair_override_is_recorded_as_section_12a_and_checked_against_the_walk(tmp_path):
    from tools.learned_spike import r3_views
    costs = {"M_nested": {}, "M_final": [8.0, 0.5], "H_nested": {}, "H_final": 8.0}
    (tmp_path / "costs_r3.json").write_text(json.dumps(costs))
    views = {"grid": r3_views.GRID, "rules": {"b": {"final": r3_views.GRID.index((64.0, 16.0)),
                                                     "final_score": 0.05}}}
    (tmp_path / "views.json").write_text(json.dumps(views))
    c = r3.label_free_costs(str(tmp_path), "b", True, (32.0, 16.0))
    assert c["M"] == (32.0, 16.0) and c["rule_pair"] == [64.0, 16.0]
    assert c["section12a"] is True and c["rule"].startswith("Section 12a")
    assert c["frozen_rule_pair"] == [8.0, 0.5]
    ns = argparse.Namespace(work=str(tmp_path), run=str(tmp_path), refs_root="~",
                            sample_root="~", mtah_root="~")
    match = r3.Match(r3.make_ctx(ns), "METRICA-S2", section12a=True)
    assert match.folder.endswith("bar_as_METRICA-S2_12a")
    assert match.rule_folder.endswith("bar_as_METRICA-S2")
    with pytest.raises(SystemExit):
        r3.check_pair12a(match, (32.0, 16.0))                 # no walk yet
    os.makedirs(match.rule_folder)
    rec = {"applies": True, "chosen": {"pair": [32.0, 16.0]}}
    with open(os.path.join(match.rule_folder, "pair12a.json"), "w") as handle:
        json.dump(rec, handle)
    r3.check_pair12a(match, (32.0, 16.0))
    with pytest.raises(SystemExit):
        r3.check_pair12a(match, (16.0, 16.0))
    rec["applies"] = False
    with open(os.path.join(match.rule_folder, "pair12a.json"), "w") as handle:
        json.dump(rec, handle)
    with pytest.raises(SystemExit):
        r3.check_pair12a(match, (32.0, 16.0))
    assert r3.parse_pair("32,16") == (32.0, 16.0)


def test_label_free_folder_prefers_the_12a_rerun(tmp_path):
    for d in ("bar_as_METRICA-S2", "bar_as_METRICA-S2_12a"):
        os.makedirs(str(tmp_path / d))
    (tmp_path / "bar_as_METRICA-S2" / "bar.json").write_text("{}")
    assert r3.label_free_folder(str(tmp_path)).endswith("bar_as_METRICA-S2")
    (tmp_path / "bar_as_METRICA-S2_12a" / "bar.json").write_text("{}")
    assert r3.label_free_folder(str(tmp_path)).endswith("bar_as_METRICA-S2_12a")


def test_walk_decodes_nothing_when_12a_does_not_apply():
    rows, chosen = r3.walk_12a([(1, 0.05), (0, 0.01)], [(1.0, 1.0), (2.0, 1.0)], None)
    assert chosen is None and not any(r["decoded"] for r in rows)


# -- Task 3: the sealed-match judge ---------------------------------------------------------------

def test_sealed_gate_needs_the_owner_value_and_an_existing_prereg(tmp_path):
    from tools.learned_spike import r3_sealed as rs
    prereg = tmp_path / "prereg.md"
    with pytest.raises(SystemExit):
        rs.gate({}, str(prereg))
    with pytest.raises(SystemExit):
        rs.gate({"UNSEAL_DFL": "owner-approved-12nov"}, str(prereg))      # file missing
    prereg.write_text("expectations")
    with pytest.raises(SystemExit):
        rs.gate({"UNSEAL_DFL": "yes"}, str(prereg))
    with pytest.raises(SystemExit):
        rs.gate({"UNSEAL_DFL": "rehearsal"}, str(prereg))                 # not the go
    rs.gate({"UNSEAL_DFL": "owner-approved-12nov"}, str(prereg))
    with pytest.raises(SystemExit):                                       # never the go here
        rs.gate({"UNSEAL_DFL": "owner-approved-12nov"}, str(prereg), rehearsal=True)
    rs.gate({"UNSEAL_DFL": "rehearsal"}, str(prereg), rehearsal=True)


def test_sealed_run_refuses_before_writing_anything(tmp_path, monkeypatch):
    from tools.learned_spike import r3_sealed as rs
    monkeypatch.delenv("UNSEAL_DFL", raising=False)
    out = tmp_path / "out"
    argv = ["run", "--out-dir", str(out), "--variant", "R3B", "--weights", "/nope.npz",
            "--pair", "64,16", "--prereg", str(tmp_path / "missing.md")]
    with pytest.raises(SystemExit):
        rs.main(argv)
    monkeypatch.setenv("UNSEAL_DFL", "owner-approved-12nov")
    with pytest.raises(SystemExit):
        rs.main(argv)                                                     # prereg missing
    assert not out.exists()


def test_sealed_run_refuses_a_second_real_start(tmp_path, monkeypatch):
    from tools.learned_spike import r3_sealed as rs
    monkeypatch.setenv("UNSEAL_DFL", "owner-approved-12nov")
    prereg = tmp_path / "prereg.md"
    prereg.write_text("x")
    out = tmp_path / "out"
    out.mkdir()
    (out / "SEALED_RESULT_R3B.json").write_text(json.dumps(
        {"rehearsal": False, "started": "t", "status": "running"}))
    with pytest.raises(SystemExit) as exc:
        rs.main(["run", "--out-dir", str(out), "--variant", "R3B", "--weights", "/nope.npz",
                 "--pair", "64,16", "--prereg", str(prereg)])
    assert "once" in str(exc.value)


def test_sealed_mapping_and_the_judge_rule():
    from tools.learned_spike import r3_sealed as rs
    assert rs.mapping() == {"DFL-J03WQQ": "DFL-J03WQQ", "DFL-J03WR9": "DFL-J03WR9"}
    assert rs.mapping("DFL-J03WMX,DFL-J03WN1") == {"DFL-J03WQQ": "DFL-J03WMX",
                                                   "DFL-J03WR9": "DFL-J03WN1"}
    assert rs.mapping("DFL-J03WMX")["DFL-J03WR9"] == "DFL-J03WMX"
    for bad in ("DFL-J03WQQ", "METRICA-S1", "BAR-ATH", "SC-1874553", "A,B,C"):
        with pytest.raises(SystemExit):
            rs.mapping(bad)
    assert rs.judge({"DFL-J03WQQ": 0.02, "DFL-J03WR9": -0.01}, True)["pass"] is True
    assert rs.judge({"DFL-J03WQQ": 0.01, "DFL-J03WR9": -0.02}, True)["pass"] is False
    assert rs.judge({"DFL-J03WQQ": 0.0, "DFL-J03WR9": 0.0}, True)["pass"] is True
    assert rs.judge({"DFL-J03WQQ": 0.05}, True)["pass"] is False          # one match missing
    assert rs.judge({"DFL-J03WQQ": 0.05, "DFL-J03WR9": 0.05}, False)["pass"] is False


def test_sealed_stress_child_refuses_a_sealed_match_without_the_go(monkeypatch):
    from tools.learned_spike import corrupt
    from tools.learned_spike import r3_sealed as rs
    monkeypatch.delenv("UNSEAL_DFL", raising=False)
    with pytest.raises(SystemExit):
        rs.main(["stress-child", "--game", "DFL-J03WQQ", "--out", "/x.csv", "--config", "/c",
                 "--hidden-out", "/h.npz", "--engine-cache", "/e"])
    with pytest.raises(SystemExit):
        rs.main(["stress-child", "--game", "BAR-ATH", "--out", "/x.csv", "--config", "/c",
                 "--hidden-out", "/h.npz", "--engine-cache", "/e"])
    assert "DFL-J03WQQ" in corrupt.FORBIDDEN        # other tools keep refusing


def test_sealed_prereg_comes_first_in_the_result(tmp_path):
    from tools.learned_spike import r3_sealed as rs
    rec = {"variant": "R3C", "rehearsal": True, "mapping": {"DFL-J03WQQ": "DFL-J03WMX",
                                                            "DFL-J03WR9": "DFL-J03WN1"},
           "started": "now", "prereg": {"path": "/p.md", "md5": "abc", "text": "I expect X"},
           "expectations": ["WQQ raw: M >= c23"], "weights": "/w.npz", "weights_md5": "w",
           "pair": [64.0, 16.0], "engine_source_hash": "e", "commit": "c", "dirty": False}
    md = rs.write_result(str(tmp_path), rec)
    text = open(md).read()
    assert md.endswith("SEALED_RESULT_R3C.md") and text.startswith("# REHEARSAL")
    assert text.index("I expect X") < text.index("## Result")
    assert "WQQ raw: M >= c23" in text


def test_short_d_segments_counts_one_and_two_frame_dead_runs():
    codes = np.array([0] * 10 + [2] + [1] * 10 + [2, 2] + [0] * 10 + [2] * 5 + [1] * 5,
                     dtype=np.int8)
    assert r3.short_d_segments(codes, [(0, len(codes))]) == 2


def test_rule_attempt_is_kept_in_the_result(monkeypatch):
    monkeypatch.setattr(r3, "crit6", lambda s, limits=None: {"pass": True})
    bar = bar_record()
    bar["final_costs"] = {"M": [64.0, 16.0], "rule": "rule b"}
    bar["md5"] = {"c23": "a", "M": "b", "Mfold": "c"}
    for k in ("c23", "M"):
        bar["systems"][k]["d_segments"] = 205
    rec = r3.rule_attempt(bar)
    assert rec["pair"] == [64.0, 16.0] and rec["criterion7"]["pass"] is False
    md = r3.rule_attempt_markdown(bar)
    assert "failed, kept" in md and "possession_s.p50 | 12.5" in md
