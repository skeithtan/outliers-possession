"""The decision logic of tools/learned_spike/evaluate.py on synthetic tables (no torch, no data)."""

import json
import os

import numpy as np
import pytest

from tools.learned_spike import evaluate as ev
from tools.learned_spike import folds as fo

GAMES = list(fo.ALL)


def table(fn):
    return {g: float(fn(i, g)) for i, g in enumerate(GAMES)}


def summary(delta, low, per=None):
    """A delta_summary()-shaped dict with given numbers."""
    per = per or {p: {"n": n, "delta": delta, "wins": n, "x_mean": 0.6, "c22_mean": 0.5}
                  for p, n in (("metrica", 3), ("dfl", 5), ("skillcorner", 20))}
    return {"x_mean": 0.5 + delta, "c22_mean": 0.5, "delta": delta, "low": low,
            "high": delta + 0.02, "per_provider": per, "per_match": {}}


# -- weights and bootstrap ------------------------------------------------------------------------

def test_provider_weights_one_third_each():
    w = ev.provider_weights(GAMES)
    assert w["METRICA-S1"] == pytest.approx(1 / 9)
    assert w["DFL-J03WMX"] == pytest.approx(1 / 15)
    assert w["SC-1874553"] == pytest.approx(1 / 60)
    assert sum(w.values()) == pytest.approx(1.0)


def test_balanced_mean_is_mean_of_provider_means():
    values = {g: (1.0 if g.startswith("METRICA") else 0.0 if g.startswith("DFL") else 0.5)
              for g in GAMES}
    assert ev.balanced_mean(values) == pytest.approx(0.5)


def test_bootstrap_deterministic_and_degenerate():
    deltas = table(lambda i, g: 0.01 * (i % 5))
    assert ev.stratified_bootstrap(deltas) == ev.stratified_bootstrap(deltas)
    const = {g: 0.03 for g in GAMES}
    low, high = ev.stratified_bootstrap(const)
    assert low == pytest.approx(0.03) and high == pytest.approx(0.03)


def test_bootstrap_matches_hand_written_draws():
    deltas = table(lambda i, g: np.sin(i))
    groups = ev.by_provider(deltas)
    rng = np.random.RandomState(0)
    draws = []
    for _ in range(ev.N_DRAWS):
        means = []
        for p in ev.PROVIDERS:
            a = np.array([deltas[g] for g in groups[p]])
            means.append(a[rng.randint(0, a.size, a.size)].mean())
        draws.append(np.mean(means))
    low, high = np.percentile(draws, (5, 95))
    assert ev.stratified_bootstrap(deltas) == (pytest.approx(low), pytest.approx(high))


def test_delta_summary_wins_and_per_provider():
    x = table(lambda i, g: 0.6 if i % 2 == 0 else 0.4)
    base = {g: 0.5 for g in GAMES}
    s = ev.delta_summary(x, base)
    assert s["per_provider"]["metrica"]["wins"] == 2      # indices 0 and 2
    assert s["per_provider"]["metrica"]["n"] == 3
    assert s["delta"] == pytest.approx(ev.balanced_mean(x) - 0.5)
    with pytest.raises(ValueError):
        ev.delta_summary(x, {"METRICA-S1": 0.5})


def test_shift_codes_edges_are_x():
    codes = np.array([0, 0, 1, 1, 2], dtype=np.int8)
    assert ev.shift_codes(codes, 2).tolist() == [3, 3, 0, 0, 1]
    assert ev.shift_codes(codes, -2).tolist() == [1, 1, 2, 3, 3]
    assert ev.shift_codes(codes, 0).tolist() == codes.tolist()


# -- criteria -------------------------------------------------------------------------------------

def test_two_thirds_rule():
    assert ev.two_thirds(2, 3) and not ev.two_thirds(1, 3)
    assert ev.two_thirds(4, 5) and not ev.two_thirds(3, 5)
    assert ev.two_thirds(14, 20) and not ev.two_thirds(13, 20)


def test_crit2_needs_mean_and_two_thirds_per_provider():
    per = {"metrica": {"n": 3, "delta": 0.01, "wins": 2},
           "dfl": {"n": 5, "delta": 0.02, "wins": 4},
           "skillcorner": {"n": 20, "delta": 0.0, "wins": 14}}
    assert ev.crit2(summary(0.01, 0.005, per))["pass"]
    per["skillcorner"]["wins"] = 13
    assert not ev.crit2(summary(0.01, 0.005, per))["pass"]
    per["skillcorner"]["wins"] = 14
    per["dfl"]["delta"] = -0.001
    assert not ev.crit2(summary(0.01, 0.005, per))["pass"]


def test_crit1_needs_both_cost_sets():
    assert ev.crit1(summary(0.02, 0.01), summary(0.02, 0.001))["pass"]
    assert not ev.crit1(summary(0.02, 0.01), summary(0.02, 0.0))["pass"]
    assert not ev.crit3(summary(0.02, -0.001))["pass"]


def test_crit4_three_parts():
    un = summary(0.04, 0.02)
    assert ev.crit4(summary(0.02, 0.001), un, 0.06, 0.02)["pass"]          # half kept, +0.04
    assert not ev.crit4(summary(0.019, 0.001), un, 0.06, 0.02)["pass"]     # under half
    assert not ev.crit4(summary(0.03, -0.001), un, 0.06, 0.02)["pass"]     # bound
    assert not ev.crit4(summary(0.03, 0.01), un, 0.0701, 0.02)["pass"]     # extra drop
    assert ev.crit4(summary(0.03, 0.01), un, 0.07, 0.02)["pass"]


def test_crit5_ten_points_per_provider():
    c22 = {"metrica": 0.20, "dfl": 0.30, "skillcorner": 0.25}
    assert ev.crit5({"metrica": 0.30, "dfl": 0.30, "skillcorner": 0.10}, c22)["pass"]
    assert not ev.crit5({"metrica": 0.31, "dfl": 0.30, "skillcorner": 0.10}, c22)["pass"]
    assert not ev.crit5({"metrica": 0.2, "dfl": 0.3}, c22)["pass"]


def test_pooled_share_and_hidden_count():
    hidden = np.array([1, 1, 1, 0, 1], dtype=bool)
    ref = np.array([0, 1, 2, 0, 1])
    codes = np.array([2, 0, 2, 2, 2])
    assert ev.d_share_hidden(hidden, ref, codes) == (3, 2)
    share = ev.pooled_share({"METRICA-S1": (3, 2), "METRICA-S2": (1, 0)})
    assert share["metrica"] == pytest.approx(0.5)


def test_crit6_vetoes():
    lim = {"poss_min": 180, "poss_max": 450, "d_min": 0.18, "d_max": 0.55, "dseg_min": 55,
           "x_max": 0}
    good = {"possessions": 300, "d_share": 0.4, "d_segments": 60, "graded_x": 0}
    assert ev.crit6(good, lim)["pass"]
    for key, value in (("possessions", 451), ("d_share", 0.17), ("d_segments", 54),
                       ("graded_x", 1)):
        assert not ev.crit6(dict(good, **{key: value}), lim)["pass"]


def bar_record(seed=0.01, sc_seed=0.01, outside=(), agree=0.95, inside=True):
    env = {k: {"bar": 1.0, "ext_min": 0.0, "ext_max": 2.0, "inside": inside}
           for k in ev.ENVELOPE_KEYS["M"]}
    stats = {"possessions": 300, "d_share": 0.4, "d_segments": 60, "graded_x": 0,
             "d_share_ball_missing": 0.5}
    return {"disagreement": {"seed": {"bar": seed, "sc_mean_fold": sc_seed,
                                      "sc_mean_final": sc_seed / 10},
                             "copy": {"bar": 0.01, "sc_mean_fold": 0.01,
                                      "sc_mean_final": 0.001}},
            "features": {"outside": list(outside), "train_games": list(fo.TRAIN_ALL)},
            "final_npz": "/x/learned_v2.npz", "final_npz_md5": "f" * 32,
            "median_costs": {"M": [16.0, 8.0], "H": 16.0, "source": "s", "smoke": False},
            "envelope_provided": {"M": env, "H": dict(env)},
            "agreement_final_vs_fold": {"M": agree, "H": agree},
            "systems": {"M": stats, "H": stats, "c22": stats}}


def v23_record(flip_m=0.31, flip_c22=0.21, dm=0.52, dc=0.50):
    """A v23.json-shaped record: V2 flips and V3 D shares."""
    systems = {"c22": {"k_frames": 1000, "alive_raw": 800, "flipped_to_d": 168, "flip": flip_c22},
               "M": {"k_frames": 1000, "alive_raw": 790, "flipped_to_d": 245, "flip": flip_m},
               "H": {"k_frames": 1000, "alive_raw": 800, "flipped_to_d": 168, "flip": flip_c22}}
    return {"v2": {"systems": systems, "k_frames": 1000, "view": list(ev.VIEW_ARGS[ev.VIEW_V2]),
                   "ball_random_state": 19},
            "v3": {"c22": dc, "M": dm, "H": dc},
            "final_npz": "/x/learned_v2.npz", "final_npz_md5": "f" * 32,
            "median_costs": {"M": [16.0, 8.0], "H": 16.0, "source": "s", "smoke": False}}


def test_bounds8():
    assert ev.bounds8("M", bar_record())["pass"]
    assert ev.bounds8("M", bar_record(seed=0.015, sc_seed=0.01))["pass"]
    assert not ev.bounds8("M", bar_record(seed=0.0151, sc_seed=0.01))["pass"]
    assert not ev.bounds8("M", bar_record(agree=0.899))["pass"]
    assert not ev.bounds8("M", bar_record(inside=False))["pass"]
    # A feature outside the band needs an explanation; a slot count also needs stress cover.
    assert not ev.bounds8("M", bar_record(outside=["bspeed"]))["pass"]
    assert ev.bounds8("M", bar_record(outside=["bspeed"]),
                      {"bspeed": {"explanation": "x"}})["pass"]
    assert not ev.bounds8("M", bar_record(outside=["H_nvis"]),
                          {"H_nvis": {"explanation": "x"}})["pass"]
    assert ev.bounds8("M", bar_record(outside=["H_nvis"]),
                      {"H_nvis": {"explanation": "x", "covered_by_stress": True}})["pass"]


def test_bounds8_h_exempt_from_d_statistics():
    rec = bar_record()
    rec["envelope_provided"]["H"]["d_segment_s.p50"]["inside"] = False
    assert ev.bounds8("H", rec)["pass"]
    assert not ev.bounds8("M", rec)["pass"]


# -- choice ---------------------------------------------------------------------------------------

def result(mean, fails=()):
    return {"mean": mean, "criteria": {str(i): {"pass": str(i) not in fails}
                                       for i in range(1, 11)}}


def test_choice_neither_one_both():
    assert ev.choose({"M": result(0.7, "1"), "H": result(0.7, "2")})["choice"] == "c22"
    assert ev.choose({"M": result(0.7), "H": result(0.7, "3")})["choice"] == "M"
    assert ev.choose({"M": result(0.7, "8"), "H": result(0.6)})["choice"] == "H"
    assert ev.choose({"M": result(0.72), "H": result(0.70)})["choice"] == "M"
    assert ev.choose({"M": result(0.70), "H": result(0.72)})["choice"] == "H"


def test_choice_h_tie_break():
    assert ev.choose({"M": result(0.7099), "H": result(0.70)})["choice"] == "H"
    assert ev.choose({"M": result(0.7101), "H": result(0.70)})["choice"] == "M"


def test_choice_transfer_rule():
    out = ev.choose({"M": result(0.80, "4"), "H": result(0.60)})
    assert out["choice"] == "H" and out["transfer_rule"]
    # H fails a criterion other than 4, 5, 7: the condition holds, H does not qualify, c22.
    out = ev.choose({"M": result(0.80, "7"), "H": result(0.60, "1")})
    assert out["choice"] == "c22" and "note" in out
    # H fails 5 too: no transfer.
    out = ev.choose({"M": result(0.80, "5"), "H": result(0.60, "5")})
    assert out["choice"] == "c22" and not out["transfer_rule"]


# -- decide refuses without inputs ---------------------------------------------------------------

def test_decide_lists_missing_inputs(tmp_path):
    inputs, missing = ev.check_inputs(str(tmp_path), str(tmp_path / "repro.json"))
    assert len(missing) == 5
    assert any("v23.json" in m for m in missing)
    for name in ("scores", "stress"):
        with open(os.path.join(str(tmp_path), name + ".json"), "w") as handle:
            json.dump({"smoke": True, "complete": False, "games": GAMES[:1]}, handle)
    _, missing = ev.check_inputs(str(tmp_path), None)
    assert any("smoke" in m for m in missing)
    with pytest.raises(SystemExit):
        ev.main(["decide", "--work", str(tmp_path), "--repro", str(tmp_path / "none.json"),
                 "--out-md", str(tmp_path / "r.md")])
    assert not (tmp_path / "r.md").exists()


def test_evaluate_all_end_to_end_on_synthetic_inputs(tmp_path):
    base = {g: 0.6 for g in GAMES}
    up = {g: 0.65 + 0.001 * i for i, g in enumerate(GAMES)}
    sums = {}
    for ref in ("ctl2", "provided"):
        for kind in ("fold", "median"):
            for cand in ev.CANDIDATES:
                sums["{}|{}|{}|0".format(cand, kind, ref)] = ev.delta_summary(up, base)
    shares = {"metrica": 0.3, "dfl": 0.3, "skillcorner": 0.3}
    stress = {"summaries": {c: {"stressed": ev.delta_summary(up, base),
                                "unstressed": ev.delta_summary(up, base)}
                            for c in ev.CANDIDATES},
              "drops": {s: {"balanced": 0.01} for s in ev.STRESS_SYSTEMS},
              "d_hidden": {ev.VIEW_U30: {s: {"per_provider": shares}
                                         for s in ev.STRESS_SYSTEMS}}}
    inputs = {"scores": {"summaries": sums}, "stress": stress, "bar": bar_record(),
              "repro": {"pass": True}, "v23": v23_record()}
    results, decision = ev.evaluate_all(inputs)
    assert decision["qualifies"] == {"M": True, "H": True}
    assert decision["choice"] == "H"        # equal means: tie-break
    assert set(results["M"]["criteria"]) == {str(i) for i in range(1, 11)}
    # M fails V2 (criterion 9): only H qualifies, so H; not the transfer rule (4, 5, 7 only).
    inputs["v23"] = v23_record(flip_m=0.3101)
    results, decision = ev.evaluate_all(inputs)
    assert not results["M"]["criteria"]["9"]["pass"] and results["H"]["criteria"]["9"]["pass"]
    assert decision["choice"] == "H" and not decision["transfer_rule"]
    inputs["v23"] = v23_record()
    inputs["repro"] = {"pass": False}
    assert ev.evaluate_all(inputs)[1]["choice"] == "c22"


def full_inputs(tmp_path):
    """Synthetic scores.json, stress.json, bar.json and repro.json in tmp_path."""
    rng = np.random.RandomState(1)
    scores = {g: {s: {ref: {"0": {"spell_f1": 0.6 + 0.05 * (s != "c22") + 0.01 * rng.rand()}}
                      for ref in ("ctl2", "provided")} for s in ev.RUN_SYSTEMS} for g in GAMES}
    for g in GAMES:
        for s in ("c22", "M", "H"):
            for ref in ("ctl2", "provided"):
                for shift in ev.SHIFTS:
                    scores[g][s][ref][str(shift)] = scores[g][s][ref]["0"]
    sc = {"games": GAMES, "complete": True, "smoke": False, "scores": scores,
          "summaries": ev.summaries(scores, GAMES)}
    shares = {"metrica": 0.3, "dfl": 0.3, "skillcorner": 0.3}
    per = {p: 0.01 for p in ev.PROVIDERS}
    up = {g: scores[g]["M"]["ctl2"]["0"]["spell_f1"] for g in GAMES}
    base = {g: scores[g]["c22"]["ctl2"]["0"]["spell_f1"] for g in GAMES}
    st = {"games": GAMES, "complete": True, "smoke": False,
          "summaries": {c: {"stressed": ev.delta_summary(up, base),
                            "unstressed": ev.delta_summary(up, base)} for c in ev.CANDIDATES},
          "drops": {s: {"balanced": 0.01, "per_provider": per, "relative_per_provider": per}
                    for s in ev.STRESS_SYSTEMS},
          "d_hidden": {v: {s: {"per_provider": shares} for s in ev.STRESS_SYSTEMS}
                       for v in (ev.VIEW_U30, ev.VIEW_STRESS)}}
    bar = bar_record()
    bar.update(complete=True, smoke=False)
    for s in ev.BAR_SYSTEMS:
        bar["systems"][s] = bar["systems"]["M"]
    v23 = v23_record()
    v23.update(complete=True, smoke=False)
    for name, obj in (("scores", sc), ("stress", st), ("bar", bar), ("v23", v23)):
        ev.write_json(os.path.join(str(tmp_path), name + ".json"), obj)
    ev.write_json(os.path.join(str(tmp_path), "repro.json"), {"pass": True})


def test_decide_writes_the_result_files(tmp_path):
    full_inputs(tmp_path)
    out = tmp_path / "LEARNED_STAGE_RESULT_R2.md"
    assert ev.main(["decide", "--work", str(tmp_path), "--repro", str(tmp_path / "repro.json"),
                    "--out-md", str(out)]) == 0
    text = out.read_text()
    assert "## Verdict" in text and "LEARNED_STAGE_RESULT_R2" in text
    assert "Honesty note" in text and "V2 and V3" in text and "| 10 |" in text
    rec = json.loads((tmp_path / "LEARNED_STAGE_RESULT_R2.json").read_text())
    assert rec["decision"]["choice"] in ("M", "H", "c22")
    assert set(rec["results"]) == {"M", "H"}
    assert rec["final_npz_md5"] == "f" * 32


def test_decide_defaults_to_the_round_2_files_and_refuses_round_1(tmp_path):
    assert os.path.basename(ev.RESULT_MD) == "LEARNED_STAGE_RESULT_R2.md"
    assert os.path.basename(ev.FINAL_NPZ) == "learned_v2.npz"
    full_inputs(tmp_path)
    with pytest.raises(SystemExit):
        ev.main(["decide", "--work", str(tmp_path), "--repro", str(tmp_path / "repro.json"),
                 "--out-md", ev.ROUND1_RESULT_MD])
    with pytest.raises(SystemExit):
        ev.check_final_out(ev.ROUND1_NPZ)
    assert ev.check_final_out(ev.FINAL_NPZ) == ev.FINAL_NPZ


def test_decide_needs_the_round_2_band_and_matching_v23(tmp_path):
    full_inputs(tmp_path)
    bar = ev.read_json(str(tmp_path / "bar.json"))
    bar["features"]["train_games"] = list(fo.ALL)              # round 1 band: refused
    ev.write_json(str(tmp_path / "bar.json"), bar)
    _, missing = ev.check_inputs(str(tmp_path), str(tmp_path / "repro.json"))
    assert any("8c band" in m for m in missing)
    full_inputs(tmp_path)
    v23 = ev.read_json(str(tmp_path / "v23.json"))
    v23["final_npz_md5"] = "0" * 32
    ev.write_json(str(tmp_path / "v23.json"), v23)
    _, missing = ev.check_inputs(str(tmp_path), str(tmp_path / "repro.json"))
    assert any("final_npz_md5" in m for m in missing)


def test_bounds8_uses_the_fold_ensemble_baseline():
    rec = bar_record(seed=0.015, sc_seed=0.01)      # in-sample final rate 0.001 is recorded only
    out = ev.bounds8("M", rec)
    assert out["a"]["sc_mean"] == pytest.approx(0.01) and out["a"]["pass"]
    assert out["a"]["sc_mean_final_recorded"] == pytest.approx(0.001)


def test_criterion5_view_is_ball_gaps_only():
    args = ev.VIEW_ARGS[ev.VIEW_U30]
    assert "--no-frag" in args and "--no-swap" in args
    assert args[args.index("--ball") + 1] == "uniform30"
    assert args[args.index("--ghost-per-10min") + 1] == "0"
    assert ev.VIEW_ARGS[ev.VIEW_STRESS] == ("--ball", "matched")


# -- round 2: V2 and V3 (criteria 9 and 10) -------------------------------------------------------

def test_v2_view_is_the_criterion5_view_with_seed_0():
    args = ev.VIEW_ARGS[ev.VIEW_V2]
    assert args[:len(ev.VIEW_ARGS[ev.VIEW_U30])] == ev.VIEW_ARGS[ev.VIEW_U30]
    assert args[args.index("--seed") + 1] == "0"
    assert args[args.index("--ghost-per-10min") + 1] == "0"
    assert "--no-frag" in args and "--no-swap" in args
    assert ev.ball_seed() == 19


def test_v2_counts_denominator_and_flips():
    H, A, D = 0, 1, 2
    raw = np.array([H, H, A, D, H, A, H, D, H, A])
    hid = np.array([H, D, D, D, D, A, D, H, H, D])
    hidden = np.array([1, 1, 1, 1, 1, 1, 0, 1, 0, 0], dtype=bool)
    graded = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=bool)
    r = ev.v2_counts(raw, hid, hidden, graded)
    # K = frames 0-4 and 7; alive on raw: 0, 1, 2, 4; D on the hidden view: 1, 2, 4.
    assert r == {"k_frames": 6, "alive_raw": 4, "flipped_to_d": 3, "flip": 0.75}
    assert ev.v2_counts(raw, hid, np.zeros(10, bool), graded)["flip"] is None


def test_crit9_bound_and_h_exempt():
    assert ev.crit9(v23_record(flip_m=0.31, flip_c22=0.21), "M")["pass"]
    assert not ev.crit9(v23_record(flip_m=0.3101, flip_c22=0.21), "M")["pass"]
    out = ev.crit9(v23_record(flip_m=0.9, flip_c22=0.1), "H")
    assert out["pass"] and out["exempt"]
    rec = v23_record()
    rec["v2"]["systems"]["M"]["flip"] = None
    assert not ev.crit9(rec, "M")["pass"]


def test_crit10_bound_and_h_exempt():
    assert ev.crit10(v23_record(dm=0.52, dc=0.50), "M")["pass"]
    assert not ev.crit10(v23_record(dm=0.5201, dc=0.50), "M")["pass"]
    assert ev.crit10(v23_record(dm=0.52, dc=0.50), "M")["numbers"]["limit"] == pytest.approx(0.52)
    out = ev.crit10(v23_record(dm=0.9, dc=0.1), "H")
    assert out["pass"] and out["exempt"]


def test_d_share_missing_matches_bar_report_unrounded():
    from tools.learned_spike import bar_report as br
    rng = np.random.RandomState(3)
    codes = rng.randint(0, 3, 400).astype(np.int8)
    codes[:20] = 3
    runs = [(20, 200), (220, 400)]
    missing = rng.rand(400) < 0.4
    got = ev.d_share_missing(codes, runs, missing)
    import possession_metrics as pm
    filled = pm.fill_abstentions(codes, runs)
    sel = br.graded_mask(runs, 400) & missing
    assert got == pytest.approx(float(np.mean(filled[sel] == 2)))
    assert round(got, 4) == br.system_stats(codes, runs, missing)["d_share_ball_missing"]


def test_qualify_needs_ten_criteria_and_transfer_rule_unchanged():
    assert ev.qualifies(result(0.7)["criteria"])
    assert not ev.qualifies(result(0.7, "10")["criteria"])
    # M fails 9 or 10: no transfer rule (Section 7 verbatim: 4, 5, 7), H qualifies alone.
    out = ev.choose({"M": result(0.80, "10"), "H": result(0.60)})
    assert out["choice"] == "H" and not out["transfer_rule"]
    out = ev.choose({"M": result(0.80, "9"), "H": result(0.60, "1")})
    assert out["choice"] == "c22"


# -- round 2: V1 on the sealed PFF set ------------------------------------------------------------

S = list(fo.PFF_SEALED)


def v1_tables(deltas, raw_deltas=None, base=0.4):
    raw_deltas = deltas if raw_deltas is None else raw_deltas
    c = {"ctl2": {g: base for g in S}, "provided": {g: base for g in S}}
    x = {"ctl2": {g: base + d for g, d in zip(S, deltas)},
         "provided": {g: base + d for g, d in zip(S, raw_deltas)}}
    return x, c


def test_match_bootstrap_equals_single_provider_stratified():
    d = {g: 0.01 * ((i * 7) % 5 - 1) for i, g in enumerate(fo.SC)}
    assert ev.match_bootstrap(d) == ev.stratified_bootstrap(d)
    d = {g: 0.01 * i for i, g in enumerate(S)}
    assert ev.match_bootstrap(d) == ev.match_bootstrap(d)


def test_v1_rule_c_needs_nine_of_fifteen():
    assert ev.win_share_ok(9, 15) and not ev.win_share_ok(8, 15)
    x, c = v1_tables([0.05] * 9 + [-0.001] * 6)
    r = ev.v1_rules(x, c, 15)
    assert r["wins"] == 9 and r["c"]
    x, c = v1_tables([0.05] * 8 + [-0.001] * 7)
    assert not ev.v1_rules(x, c, 15)["c"]


def test_v1_rules_a_b_c():
    x, c = v1_tables([0.03] * 15)
    r = ev.v1_rules(x, c, 15)
    assert r["a"] and r["b"] and r["c"] and r["pass"]
    # rule b: provided mean must be > 0 and its 5 % bound > -0.010
    x, c = v1_tables([0.03] * 15, [-0.001] * 15)
    r = ev.v1_rules(x, c, 15)
    assert r["a"] and not r["b"] and not r["pass"]
    x, c = v1_tables([0.03] * 15, [0.06] * 7 + [-0.05] * 8)
    r = ev.v1_rules(x, c, 15)
    assert r["provided"]["delta"] > 0 and r["provided"]["low"] < -0.010 and not r["b"]
    # rule a: a negative bound fails even with a positive mean
    x, c = v1_tables([0.2] + [-0.005] * 14)
    r = ev.v1_rules(x, c, 15)
    assert r["ctl2"]["delta"] > 0 and not r["a"] and not r["pass"]


def test_v1_unfinished_run_fails():
    x, c = v1_tables([0.03] * 15)
    del x["ctl2"][S[0]]
    r = ev.v1_rules(x, c, 15)
    assert not r["complete"] and not r["pass"]


def test_v1_verdict_sequential_fallback():
    assert ev.v1_verdict("M", True, {"M": True})[0] == "M"
    ships, _, run_h = ev.v1_verdict("M", True, {"M": False})
    assert ships is None and run_h
    assert ev.v1_verdict("M", True, {"M": False, "H": True})[0] == "H"
    assert ev.v1_verdict("M", True, {"M": False, "H": False})[0] == "c22"
    assert ev.v1_verdict("M", False, {"M": False})[0] == "c22"
    assert ev.v1_verdict("H", True, {"H": True})[0] == "H"
    assert ev.v1_verdict("H", True, {"H": False})[0] == "c22"       # even if M would pass
    with pytest.raises(SystemExit):
        ev.v1_verdict("c22", True, {})


def test_sealed_refuses_without_a_chosen_candidate(tmp_path):
    md = tmp_path / "LEARNED_STAGE_RESULT_R2.md"
    common = ["--work", str(tmp_path / "w"), "--stage", str(tmp_path / "stage"),
              "--result-md", str(md), "--final-npz", str(tmp_path / "learned_v2.npz")]
    for cmd in ("sealed", "sealed-record"):
        with pytest.raises(SystemExit) as exc:
            ev.main([cmd] + common)
        assert "missing" in str(exc.value)
    ev.write_json(str(tmp_path / "LEARNED_STAGE_RESULT_R2.json"),
                  {"decision": {"choice": "c22", "qualifies": {"M": False, "H": False}}})
    for cmd in ("sealed", "sealed-record"):
        with pytest.raises(SystemExit) as exc:
            ev.main([cmd] + common)
        assert "c22" in str(exc.value)
    assert not (tmp_path / "w" / "sealed.json").exists()


def test_sealed_refuses_without_or_against_the_v1_record():
    rec = {"decision": {"choice": "M"}}
    now = {"final_npz_md5": "a" * 32, "final_costs_md5": "b" * 32, "engine_source_hash": "c",
           "engine_dirty": False}
    assert "sealed-record" in ev.check_v1_record(rec, now, True)[0]
    rec["v1_record"] = {"final_npz_md5": "a" * 32, "final_costs_md5": "b" * 32,
                        "engine_source_hash": "c", "engine_commit": "abc"}
    assert ev.check_v1_record(rec, now, True) == []
    assert ev.check_v1_record(rec, dict(now, final_npz_md5="d" * 32), True)
    assert ev.check_v1_record(rec, dict(now, engine_source_hash="x"), True)
    assert ev.check_v1_record(rec, dict(now, engine_dirty=True), True)
    assert ev.check_v1_record(rec, now, False)
    assert ev.check_v1_record(rec, dict(now, final_costs_md5="e" * 32), True)


def test_sealed_matches_only_through_the_v1_gate():
    for g in S:
        with pytest.raises(SystemExit):
            ev.selected_games(type("A", (), {"games": [g]})())


def test_feature_band_pools_the_named_training_set_only(tmp_path):
    from tools.learned_spike import bar_report as br
    (tmp_path / "feats" / "raw").mkdir(parents=True)
    feats = np.zeros((10, 50))
    with pytest.raises(SystemExit):
        br.feature_band(str(tmp_path), feats, fo.COMPETITION, ["PFF-3814"])   # file missing
    with pytest.raises(AssertionError):
        br.feature_band(str(tmp_path), feats, fo.COMPETITION, ["PFF-3813"])   # sealed
    with pytest.raises(AssertionError):
        br.feature_band(str(tmp_path), feats, fo.COMPETITION, ["PFF-3812"])   # pilot


def test_decide_does_not_run_again_after_the_v1_record(tmp_path):
    full_inputs(tmp_path)
    out = tmp_path / "LEARNED_STAGE_RESULT_R2.md"
    args = ["decide", "--work", str(tmp_path), "--repro", str(tmp_path / "repro.json"),
            "--out-md", str(out)]
    assert ev.main(args) == 0
    rec = ev.read_json(str(tmp_path / "LEARNED_STAGE_RESULT_R2.json"))
    rec["v1_record"] = {"final_npz_md5": "f" * 32}
    ev.write_json(str(tmp_path / "LEARNED_STAGE_RESULT_R2.json"), rec)
    with pytest.raises(SystemExit):
        ev.main(args)
