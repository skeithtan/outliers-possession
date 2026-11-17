"""Round 5 choice rules on synthetic inputs: r4_choose costs --step3 mean2 (deviation 8.4 of
LEARNED_STAGE_PROTOCOL_R4.md, round 5 Section 3) and the round 5 screening score (screen5)."""

import pytest

from tests.test_learned_r4_choose import make_grid
from tools.learned_spike import r4_choose as rc


def test_mean2_scores_the_mean_of_t_m_and_t_d_only():
    def gain(prov, ref, pair, g):
        if pair == (16.0, 8.0):
            # strong on T-M and T-D, bad on the three cells that leave the objective
            return {("metrica", "provided"): 0.20, ("dfl", "provided"): 0.10}.get((prov, ref),
                                                                                 -0.05)
        if pair == (12.0, 12.0):
            return {("metrica", "provided"): 0.12, ("dfl", "provided"): 0.06}.get((prov, ref),
                                                                                 0.03)
        return 0.02
    grid = make_grid(gain=gain)
    five = rc.cost_choice(grid)
    mean2 = rc.cost_choice(grid, step3="mean2")
    assert five["pair"] == [12.0, 12.0]                 # the five-cell minimum sinks (16, 8)
    assert mean2["pair"] == [16.0, 8.0] and mean2["score"] == pytest.approx(0.15)
    assert mean2["step3"] == "mean2" and "deciding_cell" not in mean2
    row = [r for r in mean2["rows"] if r["pair"] == [12.0, 12.0]][0]
    assert row["score"] == pytest.approx(0.09)
    assert five["step3"] == "five" and "deciding_cell" in five   # the default is unchanged
    with pytest.raises(ValueError):
        rc.cost_choice(grid, step3="mean3")


def test_mean2_keeps_eligibility_tie_floor_and_prior():
    def gain(prov, ref, pair, g):
        return 0.05 if pair in ((16.0, 8.0), (8.0, 8.0)) else 0.02
    assert rc.cost_choice(make_grid(gain=gain), step3="mean2")["pair"] == [8.0, 8.0]
    rec = rc.cost_choice(make_grid(merge=lambda prov, pair: 0.5), step3="mean2")
    assert rec["decision"] == "c23 ships"
    rec = rc.cost_choice(make_grid(gain=lambda prov, ref, p, g: 0.005 if prov == "dfl"
                                   else 0.05), step3="mean2")
    assert rec["decision"] == "c23 ships" and "floor" in rec["reason"]
    base = make_grid()
    # bonus 1: T-D lower by 0.006 (limit 0.005): refused, as in round 4's 8.4 record
    g1 = make_grid(gain=lambda prov, ref, p, g: 0.014 if prov == "dfl" else 0.03, bonus=1.0)
    rec = rc.cost_choice(base, g1, step3="mean2")
    assert rec["prior"]["bonus"] == 0.0 and not rec["prior"]["candidates"]["1"]["acceptable"]
    # both acceptable: the higher mean2 Score wins
    g1 = make_grid(gain=lambda prov, ref, p, g: 0.03, bonus=1.0)
    g2 = make_grid(gain=lambda prov, ref, p, g: 0.04 if ref == "provided" else 0.021,
                   bonus=2.0)
    rec = rc.cost_choice(base, g1, g2, step3="mean2")
    assert rec["prior"]["bonus"] == 2.0
    assert rc.cost_choice(base, g1, g2)["prior"]["bonus"] == 1.0   # five: min cell decides


def _score_json(rows):
    """A score.json-like dict: {test: [summary rows]} for T-M and T-D."""
    out = {"summary": {"T-M": [], "T-D": []}, "networks": {}, "rehearsal": False}
    for pair, (gm, gd, mm, md) in rows.items():
        out["summary"]["T-M"].append({"pair": list(pair), "gain": gm, "merge_net": mm,
                                      "merge_c23": 0.40})
        out["summary"]["T-D"].append({"pair": list(pair), "gain": gd, "merge_net": md,
                                      "merge_c23": 0.60})
    return out


def test_screen5_score():
    rows = {p: (0.10, 0.08, 0.2, 0.4) for p in rc.R5_SCREEN_PAIRS}
    rows[(8.0, 4.0)] = (0.20, 0.20, 0.45, 0.4)          # best gains but merges more on T-M
    rows[(16.0, 16.0)] = (0.13, 0.09, 0.2, 0.6)         # T-D merge equal to c23: eligible
    rec = rc.screen5_score(_score_json(rows))
    assert rec["pair"] == [16.0, 16.0] and rec["score"] == pytest.approx(0.11)
    elig = {tuple(r["pair"]): r["eligible"] for r in rec["rows"]}
    assert elig[(8.0, 4.0)] is False and elig[(16.0, 16.0)] is True
    assert "(12, 12)" in rc.screen5_md(rec, "t")
    none = rc.screen5_score(_score_json({p: (0.1, 0.1, 0.9, 0.9) for p in rc.R5_SCREEN_PAIRS}))
    assert none["score"] is None and none["pair"] is None
    missing = _score_json({p: (0.1, 0.1, 0.2, 0.2) for p in rc.R5_SCREEN_PAIRS[:-1]})
    with pytest.raises(SystemExit):
        rc.screen5_score(missing)
    assert rc.R5_SCREEN_PAIRS == ((8.0, 4.0), (8.0, 8.0), (12.0, 12.0), (16.0, 8.0),
                                  (16.0, 16.0), (32.0, 16.0))
