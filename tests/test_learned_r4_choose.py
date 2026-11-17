"""Round 4 choice rules on synthetic inputs (r4_choose.py: protocol R4 Sections 2 and 4)."""

import copy

import pytest

from tools.learned_spike import r4_choose as rc

MET = ["METRICA-S1", "METRICA-S2", "METRICA-S3"]
DFL = ["DFL-J03WMX", "DFL-J03WN1", "DFL-J03WOH", "DFL-J03WOY", "DFL-J03WPY"]
SC = ["SC-1874553", "SC-1886347"]
REFLEN = {"pos50": 5.0, "pos90": 20.0, "d50": 15.0, "d90": 50.0}


def _pairs():
    from tools.learned_spike import r4_screen
    return [list(p) for p in r4_screen.grid_pairs()]


def make_grid(gain=lambda prov, ref, pair, g: 0.02, merge=lambda prov, pair: 0.3,
              lengths=lambda pair: dict(REFLEN), bonus=0.0, c23_merge=0.4):
    pairs = _pairs()
    matches = {}
    for prov, games, refs in (("metrica", MET, ("provided", "loose")), ("dfl", DFL, ("provided",)),
                              ("skillcorner", SC, ("possgroup", "phases"))):
        for g in games:
            ref_lengths = {}
            for k, r in enumerate(refs):
                L = dict(REFLEN)
                if g == games[0] and r == ("phases" if prov == "skillcorner" else "provided"):
                    L = {"pos50": 3.0, "pos90": 12.0, "d50": 12.0, "d90": 40.0}
                if g == games[-1] and r == ("phases" if prov == "skillcorner" else "provided"):
                    L = {"pos50": 11.0, "pos90": 45.0, "d50": 24.0, "d90": 66.0}
                ref_lengths[r] = L
            row = {"provider": prov, "ref_lengths": ref_lengths,
                   "c23": {r: {"merge": c23_merge, "spellF1": 0.5} for r in refs}, "pairs": []}
            for p in pairs:
                cell = {"pair": p, "lengths": lengths(tuple(p))}
                for r in refs:
                    cell[r] = {"delta": gain(prov, r, tuple(p), g), "merge": merge(prov, tuple(p))}
                row["pairs"].append(cell)
            matches[g] = row
    return {"pairs": pairs, "prior_bonus": bonus, "matches": matches}


def test_highest_score_and_tie_break():
    def gain(prov, ref, pair, g):
        return 0.05 if pair in ((16.0, 8.0), (8.0, 8.0)) else 0.02
    rec = rc.cost_choice(make_grid(gain=gain))
    assert rec["pair"] == [8.0, 8.0]                     # tie: lower H-A wins
    assert rec["decision"] == "pair [8.0, 8.0]"


def test_score_is_min_of_five_cells():
    def gain(prov, ref, pair, g):
        if pair == (16.0, 8.0):
            return -0.01 if ref == "phases" else 0.2      # one bad cell sinks the pair
        return 0.02
    rec = rc.cost_choice(make_grid(gain=gain))
    row = [r for r in rec["rows"] if r["pair"] == [16.0, 8.0]][0]
    assert row["score"] == pytest.approx(-0.01)
    assert rec["pair"] != [16.0, 8.0]


def test_eligibility_8e_and_merge():
    def lengths(pair):
        L = dict(REFLEN)
        if pair[0] >= 16.0:
            L["pos50"] = 12.0                             # median outside [3, 11]
        return L

    def gain(prov, ref, pair, g):
        return 0.1 if pair[0] >= 16.0 else 0.02
    rec = rc.cost_choice(make_grid(gain=gain, lengths=lengths))
    assert rec["pair"][0] < 16.0
    rec = rc.cost_choice(make_grid(merge=lambda prov, pair: 0.5))   # merges more than c23
    assert rec["decision"] == "c23 ships" and "no pair" in rec["reason"]


def test_floor():
    def gain(prov, ref, pair, g):
        return 0.005 if prov == "dfl" else 0.05
    rec = rc.cost_choice(make_grid(gain=gain))
    assert rec["decision"] == "c23 ships" and "floor" in rec["reason"]


def test_prior_rules():
    base = make_grid()
    pair = rc.cost_choice(base)["pair"]

    def better(prov, ref, pair_, g):                       # every match up a little
        return 0.025
    g1 = make_grid(gain=better, bonus=1.0)
    g2 = make_grid(gain=lambda *a: 0.01, bonus=2.0)        # mean lower by 0.01: refused
    rec = rc.cost_choice(base, g1, g2)
    assert rec["prior"]["candidates"]["1"]["acceptable"] is True
    assert rec["prior"]["bonus"] == 1.0
    cand2 = rec["prior"]["candidates"]["2"]
    assert cand2["acceptable"] is False
    # worst match up on only one test: refused
    def one(prov, ref, pair_, g):
        return 0.04 if prov == "metrica" else 0.02
    rec = rc.cost_choice(base, make_grid(gain=one, bonus=1.0))
    assert rec["prior"]["bonus"] == 0.0
    # both acceptable: higher Score wins, then 1
    hi = make_grid(gain=lambda prov, ref, p, g: 0.03, bonus=2.0)
    rec = rc.cost_choice(base, g1, hi)
    assert rec["prior"]["bonus"] == 2.0
    rec = rc.cost_choice(base, g1, make_grid(gain=better, bonus=2.0))
    assert rec["prior"]["bonus"] == 1.0
    assert pair == rec["pair"]


def test_grid_checks():
    g = make_grid()
    bad = copy.deepcopy(g)
    bad["prior_bonus"] = 1.0
    with pytest.raises(SystemExit):
        rc.cost_choice(bad)
    with pytest.raises(SystemExit):
        rc.cost_choice(g, make_grid(bonus=2.0))            # grid1 decoded with the wrong bonus


# -- Section 2 ----------------------------------------------------------------------------------

def S(score, merge_m=0.3, merge_d=0.4, split_s=0.3, pair=(16.0, 8.0)):
    return {"score": score, "pair": list(pair) if score is not None else None,
            "merge": {"T-M": merge_m, "T-D": merge_d}, "split": {"T-S": split_s}}


def test_screen_rule1_and_2():
    assert rc.screen_choice(S(0.03), S(0.02))["chosen"] == "S1'"          # +0.01 < margin
    assert rc.screen_choice(S(0.036), S(0.02))["chosen"] == "S1"
    assert rc.screen_choice(S(0.035), S(0.02))["chosen"] == "S1"          # exactly the margin
    assert rc.screen_choice(S(0.03), S(None))["chosen"] == "S1"           # S1' has no score
    assert rc.screen_choice(S(None), S(None))["chosen"] is None
    assert rc.screen_choice(S(None), S(0.0), S(0.02))["chosen"] == "S2"
    assert rc.screen_choice(S(None), S(0.0), S(0.01))["chosen"] == "S1'"
    assert rc.screen_choice(S(None), S(0.0), S(None))["chosen"] == "S1'"


def test_screen_rule3():
    cur = S(0.02, 0.3, 0.4, 0.30)
    assert rc.screen_choice(S(None), cur, None, S(0.04, 0.29, 0.39, 0.31))["chosen"] == "S3"
    assert rc.screen_choice(S(None), cur, None, S(0.04, 0.31, 0.39, 0.30))["chosen"] == "S1'"
    assert rc.screen_choice(S(None), cur, None, S(0.04, 0.29, 0.40, 0.30))["chosen"] == "S1'"
    assert rc.screen_choice(S(None), cur, None, S(0.04, 0.29, 0.39, 0.32))["chosen"] == "S1'"
    assert rc.screen_choice(S(None), cur, None, S(0.03, 0.2, 0.2, 0.2))["chosen"] == "S1'"


def test_recipe_summary_reads_score_pair():
    res = {"score": {"score": 0.02, "pair": [16.0, 8.0]},
           "summary": {t: [{"pair": [8.0, 4.0], "merge_net": 0.9, "split_net": 0.9},
                           {"pair": [16.0, 8.0], "merge_net": 0.3, "split_net": 0.2}]
                       for t in ("T-M", "T-D", "T-S")}}
    s = rc.recipe_summary(res)
    assert s["merge"] == {"T-M": 0.3, "T-D": 0.3, "T-S": 0.3} and s["split"]["T-S"] == 0.2


def test_screen_rule3_s2b_plain_margin():
    cur = S(0.02, 0.3, 0.4, 0.30)
    # S2b: score margin only (no merge or split condition)
    assert rc.screen_choice(S(None), cur, None, None, S(0.035, 0.9, 0.9, 0.9))["chosen"] == "S2b"
    assert rc.screen_choice(S(None), cur, None, None, S(0.03))["chosen"] == "S1'"
    # S3 is then judged against S2b
    out = rc.screen_choice(S(None), cur, None, S(0.06, 0.2, 0.2, 0.2), S(0.04, 0.3, 0.4, 0.3))
    assert out["chosen"] == "S3" and any(l.startswith("rule 3: S2b") for l in out["log"])
    assert rc.screen_choice(S(None), cur, None, S(0.05, 0.2, 0.2, 0.2),
                            S(0.04, 0.3, 0.4, 0.3))["chosen"] == "S2b"


def test_deciding_cell_and_drop_one():
    def gain(prov, ref, pair, g):
        if pair == (16.0, 8.0):
            return 0.01 if ref == "phases" else 0.08       # phases decides (16, 8)
        if pair == (6.0, 3.0):
            return 0.03
        return 0.02
    rec = rc.cost_choice(make_grid(gain=gain))
    assert rec["pair"] == [6.0, 3.0] and rec["deciding_cell"] in [
        "{} {}".format(t, r) for t, r in rc.CELLS]
    assert rec["drop_one"]["T-S phases"]["pair"] == [16.0, 8.0]
    assert rec["drop_one"]["T-M provided"]["pair"] == [6.0, 3.0]
    text = rc.costs_md(rec, "t")
    assert "decided the pair" in text and "without T-S phases: [16.0, 8.0]" in text
