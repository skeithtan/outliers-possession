"""Round 5 measures (r5_measures.py): disagreement, dead-ball recall, summaries."""

import numpy as np

from tools.learned_spike import bar_report as br, r5_measures as rm


def _probs(n_bins, rng, shift=0.0):
    p = rng.dirichlet([1, 1, 1], size=n_bins) + shift
    return p / p.sum(axis=1, keepdims=True)


def test_group_matches_prob_group(tmp_path):
    rng = np.random.RandomState(0)
    graded = np.zeros(500, bool)
    graded[20:480] = True
    members = [_probs(100, rng) for _ in range(3)]
    paths = []
    for i, m in enumerate(members):
        path = tmp_path / "m{}.npz".format(i)
        np.savez(path, p5=m)
        paths.append(str(path))
    ref = br.prob_group(paths, graded)
    got = rm.group(members, graded)
    assert round(got[0], 4) == ref["share_any_disagree"]
    assert round(got[1], 4) == ref["mean_pairwise_disagree"]


def test_disagree_match_one_and_two_seeds():
    rng = np.random.RandomState(1)
    graded = np.ones(250, bool)
    same = _probs(50, rng)
    one = {0: np.stack([same] * 4)}
    d = rm.disagree_match(one, graded)
    assert d["copy"] == {"0": 0.0} and d["seed_pair"] == {} and "seeds_any" not in d
    two = {0: np.stack([same] * 4), 1: np.stack([_probs(50, rng) for _ in range(4)])}
    d = rm.disagree_match(two, graded)
    assert d["copy"]["0"] == 0.0 and d["copy"]["1"] > 0 and d["seed_pair"]["0-1"] > 0
    assert d["seeds_any"] == d["seed_pair"]["0-1"]


def test_dead_recall_split_and_summary():
    codes = np.array([2] * 10 + [0] * 10 + [0] * 10)
    segs = [(0, 10), (10, 20), (20, 30)]
    ball = np.array([True] * 10 + [False] * 10 + [True] * 4 + [False] * 6)
    rows = rm.dead_recall(codes, segs, ball)
    assert rows == [(1.0, True), (0.0, False), (0.0, False)]
    s = rm.summarise_recall(rows)
    assert s["ball_in_view"] == {"segments": 1, "share_D": 1.0, "share_segments_no_D": 0.0}
    assert s["ball_hidden"]["segments"] == 2 and s["ball_hidden"]["share_segments_no_D"] == 1.0


def test_mean_and_ratio_dicts():
    m = rm.mean_dicts([{"a": {"x": 1.0}, "b": None}, {"a": {"x": 3.0}, "b": 2.0}])
    assert m == {"a": {"x": 2.0}, "b": 2.0}
    assert rm.ratio_dicts({"a": 1.0, "b": None}, {"a": 4.0, "b": 1.0}) == {"a": 0.25, "b": None}


def test_section2_score_eligibility():
    tm = [{"pair": [8, 4], "gain": 0.05, "merge_net": 0.3, "merge_c23": 0.2},
          {"pair": [12, 12], "gain": 0.02, "merge_net": 0.1, "merge_c23": 0.2}]
    td = [{"pair": [8, 4], "gain": 0.09, "merge_net": 0.1, "merge_c23": 0.2},
          {"pair": [12, 12], "gain": 0.04, "merge_net": 0.1, "merge_c23": 0.2}]
    sc = rm.section2_score(tm, td)
    assert sc["pair"] == [12, 12] and abs(sc["score"] - 0.03) < 1e-12
    assert [r["eligible"] for r in sc["per_pair"]] == [False, True]


def _summary(seed_ratios, copy_ratios, score, f1, sc=None):
    import numpy as _np
    dev = {"ratio_defect_raw": {"seed_pair": dict(enumerate(seed_ratios)),
                                "copy": dict(enumerate(copy_ratios))},
           "median_seed_ratio": float(_np.median(seed_ratios)),
           "median_copy_ratio": float(_np.median(copy_ratios)),
           "score": {"score": score, "pair": [12, 12]},
           "defect": {"spellF1_ensemble": f1}}
    out = {"summary": {"dev": dev}}
    if sc:
        din, live, split = sc
        out["summary"]["skillcorner"] = {
            "dead_recall": {"input": {"model": {"ball_in_view": {"share_D": din}},
                                      "c23": {"ball_in_view": {"share_D": 0.668}}}},
            "live_slow_D": {"model": {"share_D": live}},
            "merge_split": {"model": {"split": split}}, "split_reference": "possession groups"}
    return out


def test_rules_pass_and_fail_lines():
    r0 = _summary([1.437, 1.447, 1.396, 1.448, 1.413, 1.578], [1.479, 1.497, 1.443, 1.504],
                  0.05, 0.63)
    r1 = _summary([1.30] * 6, [1.50] * 4, 0.045, 0.625, (0.30, 0.10, 0.20))
    out, lines = rm.rules(r0, r1)
    assert out["section2"] and all(l.startswith("PASS") for l in lines if l[:4] in ("PASS", "FAIL"))
    r2 = _summary([1.30] * 6, [1.50] * 4, 0.04, 0.62, (0.40, 0.105, 0.205))
    out, lines = rm.rules(r0, r1, r2)
    assert out["section9_3"]                     # 0.40 >= 0.30 + 0.25 * 0.368 = 0.392
    r2b = _summary([1.30] * 6, [1.50] * 4, 0.04, 0.62, (0.38, 0.105, 0.205))
    out, lines = rm.rules(r0, r1, r2b)
    assert not out["section9_3"] and any(l.startswith("FAIL rule 1") for l in lines)
    r1bad = _summary([1.60] * 6, [1.50] * 4, 0.045, 0.625, (0.30, 0.10, 0.20))
    out, lines = rm.rules(r0, r1bad)
    assert not out["section2"] and lines[1].startswith("FAIL rule 1")


def test_live_slow_counts(monkeypatch):
    refc = np.array([0, 0, 1, 2, 0], np.int8)
    codes = np.array([2, 0, 2, 2, 2], np.int8)
    speed = np.array([np.nan, 0.5, 0.2, 0.1, 3.0])
    monkeypatch.setattr(rm, "graded_frames", lambda game, n: np.ones(n, bool))
    assert rm.live_slow(codes, refc, "METRICA-S1", speed) == (1, 2)
