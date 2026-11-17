"""Round 4 final scoring (r3_eval variant R4, r4_oof_ratio, r4_final.sh) on synthetic data."""

import argparse
import json
import os
import subprocess

import pytest

from tools.learned_spike import evaluate as ev
from tools.learned_spike import r3_eval as r3
from tools.learned_spike import r4_oof_ratio as oof

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def r4_switches(**over):
    sw = {"target": "raw", "short_weight": "off", "mix": "sc_quarter", "short_median_bins": None,
          "aug": {"cache": "/x", "k": 16, "share": 0.5, "manifest_md5": "abc"},
          "r4": {"views": ["raw"], "dfl_still": "removed", "val_cache": "/v",
                 "sc_alive_dead": True, "ballfree": {"cache": "/b", "share": 0.15}}}
    sw.update(over)
    return sw


def test_r4_switch_check():
    assert r3.check_r4_switches(r4_switches()) == []
    assert r3.check_r4_switches(r4_switches(short_weight="sqrt")) == []
    bad = r4_switches()
    bad["r4"] = dict(bad["r4"], views=["raw", "degraded"])
    assert any("views" in p for p in r3.check_r4_switches(bad))
    bad = r4_switches()
    bad["r4"] = dict(bad["r4"], sc_labels="/pg")                 # B and B' together
    assert any("both B" in p for p in r3.check_r4_switches(bad))
    assert any("round 4 record" in p for p in r3.check_r4_switches(r4_switches(r4=None)))
    assert any("aug" in p for p in r3.check_r4_switches(r4_switches(aug=None)))
    rec = {"switches": r4_switches(), "steps": 2000, "train": []}
    assert r3.check_model_record(rec, None, "R4") == []
    assert r3.check_model_record(dict(rec, steps=60), None, "R4") == ["steps 60 != 2000"]
    # an R3b model is not an R4 model
    assert r3.check_model_record({"switches": dict(r3.R3B_SWITCHES), "steps": 2000}, None, "R4")


def _work(tmp_path, variant="R4"):
    w = tmp_path / "work"
    w.mkdir()
    (w / "preflight.json").write_text(json.dumps({"variant": variant, "rehearsal": True}))
    return str(w)


def test_costs_fixed_and_label_free_costs(tmp_path):
    w = _work(tmp_path)
    args = argparse.Namespace(work=w, run="/r", pair=(16.0, 8.0),
                                                      prior_bonus=1.0, no_structure=False)
    assert r3.cmd_costs_fixed(args) == 0
    costs = r3.load_costs(w)
    assert costs["fixed"] and costs["M_final"] == [16.0, 8.0] and costs["prior_bonus"] == 1.0
    mc = r3.match_costs(costs, "METRICA-S1")
    assert mc["M"] == (16.0, 8.0) and mc["prior_bonus"] == 1.0 and mc["structure"] is True
    over = ev.overrides("M", "/w.npz", mc)
    assert over["LEARNED_PRIOR_BONUS"] == 1.0 and over["LEARNED_STRUCTURE"] is True
    lf = r3.label_free_costs(w, "fixed", True)
    assert lf["M"] == (16.0, 8.0) and lf["prior_bonus"] == 1.0 and "fixed" in lf["rule"]
    with pytest.raises(SystemExit):
        r3.label_free_costs(w, "b", True)                         # no rule b in round 4
    with pytest.raises(SystemExit):
        r3.label_free_costs(w, "fixed", True, (32.0, 16.0))       # no walk
    args.pair = (8.0, 4.0)
    with pytest.raises(SystemExit):
        r3.cmd_costs_fixed(args)                                  # other costs already there
    (tmp_path / "x").mkdir()
    w2 = _work(tmp_path / "x")
    args2 = argparse.Namespace(work=w2, run="/r", pair=(32.0, 8.0),
                                                       prior_bonus=0.0, no_structure=False)
    with pytest.raises(SystemExit):
        r3.cmd_costs_fixed(args2)                                 # H-A > 2 x alive-dead
    args2.pair, args2.prior_bonus = (16.0, 8.0), 0.5
    with pytest.raises(SystemExit):
        r3.cmd_costs_fixed(args2)                                 # bonus not 0, 1 or 2


def test_costs_fixed_refuses_round3_work(tmp_path):
    w = _work(tmp_path, "R3B")
    args = argparse.Namespace(work=w, run="/r", pair=(16.0, 8.0),
                                                      prior_bonus=0.0, no_structure=False)
    with pytest.raises(SystemExit):
        r3.cmd_costs_fixed(args)


def test_round3_overrides_unchanged():
    c = {"M": (32.0, 16.0), "H": 32.0, "structure": True}
    assert "LEARNED_PRIOR_BONUS" not in ev.overrides("M", "/w.npz", c)
    assert "LEARNED_PRIOR_BONUS" not in ev.overrides("M", "/w.npz", dict(c, prior_bonus=0.0))
    costs = {"M_nested": {"METRICA-S1": [8.0, 4.0]}, "M_final": [16.0, 8.0], "H_nested": {},
             "H_final": 32.0}
    mc = r3.match_costs(costs, "METRICA-S1")
    assert "prior_bonus" not in mc and "structure" not in mc and mc["source"] == "nested"


def _stats(n, per90, p10, p25):
    return {"possessions": n, "possessions_per_90min": per90,
            "possession_s": {"p10": p10, "p25": p25, "p50": 5.0, "p90": 20.0}}


def test_r4_bounds():
    refs = [_stats(300, 200.0, 1.0, 2.0), _stats(400, 260.0, 1.6, 3.0)]
    ok = r3.r4_bounds(_stats(210, 220.0, 1.2, 2.5), _stats(200, 0, 0, 0), refs, (0.9, 1.2))
    assert ok["pass"] and ok["bounds"]["possessions_over_c23"]["value"] == pytest.approx(1.05)
    import numpy as np
    js = r3.r4_bounds({"possessions": np.int64(210), "possessions_per_90min": np.float64(220.0),
                       "possession_s": {"p10": np.float64(1.2), "p25": np.float64(2.5)}},
                      _stats(200, 0, 0, 0), refs, (np.float64(0.9), np.float64(1.2)))
    json.dumps(js)                                       # numpy values serialise (rehearsal bug)
    for m, rng, key in ((_stats(210, 270.0, 1.2, 2.5), (0.9, 1.2), "possessions_per_90min"),
                        (_stats(210, 220.0, 0.9, 2.5), (0.9, 1.2), "possession_s.p10"),
                        (_stats(210, 220.0, 1.2, 3.5), (0.9, 1.2), "possession_s.p25"),
                        (_stats(260, 220.0, 1.2, 2.5), (0.9, 1.2), "possessions_over_c23")):
        r = r3.r4_bounds(m, _stats(200, 0, 0, 0), refs, rng)
        assert not r["pass"] and not r["bounds"][key]["inside"]


def test_verdict_r4_blocking():
    crit = {k: {"pass": True} for k in r3.R4_BLOCKING}
    choice, reason = r3.verdict(crit, True, None, r3.R4_BLOCKING)
    assert choice == "M" and "G7" in reason
    crit["7x"] = {"pass": False}
    assert r3.verdict(crit, True, None, r3.R4_BLOCKING)[0] == "c23"
    crit["7x"] = None
    assert r3.verdict(crit, True, None, r3.R4_BLOCKING)[0] is None
    # criteria 3, 4, 9, 10 do not block in round 4
    crit = {k: {"pass": True} for k in r3.R4_BLOCKING}
    crit.update({"3": {"pass": False}, "4": {"pass": False}})
    assert r3.verdict(crit, True, None, r3.R4_BLOCKING)[0] == "M"


def test_criteria_reads_r4bounds(tmp_path):
    w = _work(tmp_path)
    folder = os.path.join(w, "bar_as_DFL-J03WMX")
    os.makedirs(folder)
    with open(os.path.join(folder, "r4bounds.json"), "w") as h:
        json.dump({"pass": False, "bounds": {"possessions_over_c23": {"inside": False}}}, h)
    crits = r3.criteria_so_far(w)
    assert crits["7x"]["pass"] is False and crits["7x"]["blocking"] is True


def test_oof_ratio_helpers():
    from tools.learned_spike import folds
    for g in folds.ALL:
        assert g in folds.fold(oof.fold_of(g), pff=False)["test"]
    rows = {"a": {"ratio": 0.9}, "b": {"ratio": 1.1}, "c": {"ratio": 1.0}}
    r = oof.ratio_range(rows)
    assert (r["ratio_min"], r["ratio_max"], r["ratio_median"], r["n_matches"]) == (0.9, 1.1, 1.0, 3)
    with pytest.raises(SystemExit):
        oof.fold_of("DFL-J03WQQ")


def test_repro_check_has_prior_option():
    from tools.learned_spike import repro_check
    with pytest.raises(SystemExit) as e:
        repro_check.main(["run", "--help"])
    assert e.value.code == 0


def test_r4_final_dry_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    out = tmp_path / "out"
    env = dict(os.environ, CA=str(tmp_path / "ca"), MTAH_DATA_DIR=str(tmp_path / "d"),
               C23_DEV=str(tmp_path / "c"), PAIR="16,8", PRIOR="1", OOF_RATIO="/x.json",
               DRY_RUN="1", REHEARSAL="1", AS_MATCH="DFL-J03WMX")
    env.pop("STOP_BEFORE_LABEL_FREE", None)    # the caller's run settings must not leak in
    sh = os.path.join(ROOT, "tools", "learned_spike", "r4_final.sh")
    res = subprocess.run(["bash", sh, str(run), str(out)], env=env, cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    text = res.stdout.decode()
    assert res.returncode == 0, text
    assert "DRY RUN" in text and "--identity" not in text          # rehearsal: no identity
    assert "costs-fixed" in text and "--pair 16,8 --prior-bonus 1" in text
    assert "--as-match DFL-J03WMX" in text and "r4bounds" in text and "--prior-bonus 1" in text
    assert not (out / "work" / "preflight.json").exists()           # nothing ran
    env.pop("REHEARSAL")
    env.pop("AS_MATCH")
    env["STOP_BEFORE_LABEL_FREE"] = "1"
    res = subprocess.run(["bash", sh, str(run), str(tmp_path / "out2")], env=env, cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    text = res.stdout.decode()
    assert res.returncode == 0 and "--identity" in text and "bar " not in text
    assert "STOP_BEFORE_LABEL_FREE" in text


def test_stress_rehearsal_subset_only_in_rehearsal(tmp_path, monkeypatch):
    w = _work(tmp_path)                                     # preflight says rehearsal
    monkeypatch.delenv("R3_EVAL_REHEARSAL_GAMES", raising=False)
    assert r3.stress_games(w) == r3.ALL
    monkeypatch.setenv("R3_EVAL_REHEARSAL_GAMES", "METRICA-S1 SC-1874553")
    assert r3.stress_games(w) == ("METRICA-S1", "SC-1874553")
    real = tmp_path / "real"
    real.mkdir()
    (real / "preflight.json").write_text(json.dumps({"variant": "R4", "rehearsal": False}))
    with pytest.raises(SystemExit):
        r3.stress_games(str(real))
    monkeypatch.setenv("R3_EVAL_REHEARSAL_GAMES", "DFL-J03WQQ")
    with pytest.raises(SystemExit):
        r3.stress_games(w)


def test_repro_check_round4_costs_have_no_h(tmp_path):
    from tools.learned_spike import repro_check
    path = tmp_path / "costs.json"
    path.write_text(json.dumps({"M": {"cost_ha": 16.0, "cost_ad": 8.0}, "H": {"cost_ha": None},
                                "prior_bonus": 1.0}))
    c = ev.costs_record(str(path), None)
    assert c["H"] is None and repro_check.candidates_of(c) == ("M",)
    assert repro_check.candidates_of({"M": (1, 1), "H": 32.0}) == ("M", "H")
    rec = {"n_graded": 10, "M": {"torch_vs_numpy": {"graded_frames": 0, "listed": [],
                                                     "frames": 0}, "end_to_end_equal": True}}
    assert repro_check.verdict({"G": rec}, ("M",))["pass"] is True
