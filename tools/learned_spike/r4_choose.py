"""Round 4 choices: the screening choice (protocol Section 2) and the costs and prior (Section 4). Not part of the submission.

    # Section 2: which recipe is trained in full (score.json of r4_screen score, per recipe)
    python -m tools.learned_spike.r4_choose screen --s1 DIR/ABC/score.json \
        --s1p DIR/AQC/score.json [--s2 DIR/.../score.json] [--s2b DIR/.../score.json] \
        [--s3 DIR/.../score.json] [--out F]
    # Section 4: the pair and the prior bonus (grid.json of r4_screen grid on the left-out
    # ensembles, four seeds, at bonus 0, 1 and 2)
    python -m tools.learned_spike.r4_choose costs --grid0 G0 [--grid1 G1 --grid2 G2] [--out F]
        [--step3 five|mean2]
    # Round 5 screening score (LEARNED_STAGE_PROTOCOL_R5.md Section 2, "Score"): from the
    # score.json of r4_screen score --r5 (the six pairs of R5_SCREEN_PAIRS, seeds 0 1 2 3)
    python -m tools.learned_spike.r4_choose screen5 --score DIR/score.json [--out F]

LEARNED_STAGE_PROTOCOL_R4.md as frozen (commit bb32624, 13 November 2026).

Section 2 (screen_choice()). A recipe's score is the one r4_screen wrote (None if no pair is
eligible: such a recipe is never chosen). Margin MARGIN = 0.015.
1. S1' is the starting point; S1 replaces it only if its score is higher by at least the margin.
2. S2 replaces the chosen one only if its score is higher by at least the margin.
3. S2b (S2's outcome + F; screened only if F was ready in time) replaces the current recipe only
   if its score is higher by at least the margin.
4. S3 replaces the current recipe only if its score is higher by at least the margin, its merge
   share is lower on T-M and on T-D, and its split share on T-S is not higher by more than
   SPLIT_TOL = 0.01. Merge and split shares are the means over matches at each recipe's own score
   pair (the pair that gave its score).
If no recipe clears a margin, the simpler one stays. A candidate replaces a current recipe
without a score whenever it has one (no margin is defined against no score). If neither S1 nor
S1' has a score, no recipe is chosen.

Section 4 (cost_choice()). Step 0 of the frozen text (the decoder family, decided by a separate
study; if it wins, extra decoder features are tuned after steps 1 to 5 fixed the costs) is not
part of this tool: it implements steps 1 to 6 for the two-cost decoder. Inputs are grid.json files: every pair of the 80-pair grid with H-A
<= 2 x alive-dead, per match, network (left-out ensemble of the provider) and c23 on the same
input, spellF1 per reference, possessions, merge share, and the length statistics.
1. Grid: the pairs of the file (checked against r4_screen.grid_pairs(): the finer grid of
   deviation 8.1, 146 pairs).
2. Eligible if (a) for each of the four 8e statistics (median and 90th percentile of possession
   length and of dead-ball segment length) the median over the eight development matches of the
   network's output lies inside [min, max] of the 28 external references as provided (Metrica
   as provided, DFL as provided, SkillCorner phases), and (b) the mean merge share of the network
   is at most c23's on T-M (as provided) and on T-D.
3. Score = min of five mean gains over c23: T-M provided, T-M loose, T-D, T-S possession groups,
   T-S phases.
4. The highest Score among eligible pairs; ties: lower H-A, then lower alive-dead.
5. Floor: T-M provided and T-D at that pair at least FLOOR = +0.01, else c23 ships; c23 ships if
   no pair is eligible.
6. Prior: bonus 1 or 2 replaces 0 only if, at the pair, on each of T-M provided, T-D and T-S
   possession groups the mean gain is not lower than bonus 0's by more than PRIOR_TOL = 0.005;
   the worst match's gain is higher than bonus 0's on at least two of those three; and 2(b)
   still holds. If both 1 and 2 are acceptable: the higher Score, then 1.

Step 3 variants (--step3): "five" (default, as frozen above: the smallest of the five cells)
and "mean2" (deviation 8.4 of LEARNED_STAGE_PROTOCOL_R4.md, the rule of round 5's Section 3:
Score = the mean of the two cells T-M as provided and T-D; the other three cells are recorded).
Everything else is the same under both: eligibility 2(a) and 2(b), the tie rule, the floor of
step 5, and step 6 (whose "higher Score" uses the same step 3).

Round 5 screening score (screen5_score()): over the six pairs (8, 4), (8, 8), (12, 12), (16, 8),
(16, 16), (32, 16), among the pairs whose mean merge share is at most c23's on T-M and on T-D,
the highest mean of the two mean gains (T-M as provided, T-D). None if no pair is eligible.

Reads only the given files. Python 3.9.
"""

import argparse
import json
import sys

import numpy as np

MARGIN = 0.015
SPLIT_TOL = 0.01
FLOOR = 0.01
PRIOR_TOL = 0.005
E8 = ("pos50", "pos90", "d50", "d90")
CELLS = (("T-M", "provided"), ("T-M", "loose"), ("T-D", "provided"), ("T-S", "possgroup"),
         ("T-S", "phases"))
PROVIDER_TEST = {"metrica": "T-M", "dfl": "T-D", "skillcorner": "T-S"}
REF_AS_PROVIDED = {"metrica": "provided", "dfl": "provided", "skillcorner": "phases"}
STEP3 = ("five", "mean2")
MEAN2_CELLS = ("T-M provided", "T-D provided")
R5_SCREEN_PAIRS = ((8.0, 4.0), (8.0, 8.0), (12.0, 12.0), (16.0, 8.0), (16.0, 16.0),
                   (32.0, 16.0))


def step3_score(cells, step3="five"):
    """Step 3: "five" the smallest of the five cells; "mean2" the mean of T-M provided and T-D."""
    if step3 == "five":
        return min(cells.values())
    if step3 == "mean2":
        return float(np.mean([cells[k] for k in MEAN2_CELLS]))
    raise ValueError("step3 must be one of {}, not {!r}".format(STEP3, step3))


def load(path):
    with open(path) as h:
        return json.load(h)


# -- Section 2 ----------------------------------------------------------------------------------

def recipe_summary(res):
    """{"score", "pair", "merge": {test: net mean}, "split": {test: net mean}} of a score.json."""
    sc = (res.get("score") or {}).get("score")
    pair = (res.get("score") or {}).get("pair")
    out = {"score": sc, "pair": pair, "merge": {}, "split": {}}
    if pair is None:
        return out
    for test in ("T-M", "T-D", "T-S"):
        for row in res["summary"][test]:
            if [float(x) for x in row["pair"]] == [float(x) for x in pair]:
                out["merge"][test] = row["merge_net"]
                out["split"][test] = row["split_net"]
    return out


def beats(cand, cur, margin=MARGIN):
    """(True/False, reason) for the score margin."""
    if cand["score"] is None:
        return False, "no score"
    if cur["score"] is None:
        return True, "the current recipe has no score"
    ok = cand["score"] >= cur["score"] + margin - 1e-12
    return ok, "score {:+.4f} vs {:+.4f} (margin {})".format(cand["score"], cur["score"], margin)


def screen_choice(s1, s1p, s2=None, s3=None, s2b=None):
    """The Section 2 decision. Inputs are recipe_summary() dicts (or None if not run)."""
    log = []
    if s1p["score"] is None and s1["score"] is None:
        return {"chosen": None, "log": ["neither S1 nor S1' has a score: no recipe is chosen"]}
    cur, name = s1p, "S1'"
    ok, why = beats(s1, s1p)
    log.append("rule 1: S1 against S1': {} -> {}".format(why, "S1" if ok else "S1'"))
    if ok:
        cur, name = s1, "S1"
    if s2 is not None:
        ok, why = beats(s2, cur)
        log.append("rule 2: S2 against {}: {} -> {}".format(name, why, "S2" if ok else name))
        if ok:
            cur, name = s2, "S2"
    if s2b is not None:
        ok, why = beats(s2b, cur)
        log.append("rule 3: S2b against {}: {} -> {}".format(name, why, "S2b" if ok else name))
        if ok:
            cur, name = s2b, "S2b"
    if s3 is not None:
        ok, why = beats(s3, cur)
        extra = []
        if ok and cur["score"] is not None:
            for t in ("T-M", "T-D"):
                lower = s3["merge"].get(t, np.inf) < cur["merge"].get(t, -np.inf)
                extra.append("merge {} {:.3f} vs {:.3f}: {}".format(
                    t, s3["merge"].get(t, np.nan), cur["merge"].get(t, np.nan),
                    "lower" if lower else "not lower"))
                ok = ok and lower
            sp = s3["split"].get("T-S", np.inf) <= cur["split"].get("T-S", -np.inf) + SPLIT_TOL
            extra.append("split T-S {:.3f} vs {:.3f}: {}".format(
                s3["split"].get("T-S", np.nan), cur["split"].get("T-S", np.nan),
                "ok" if sp else "higher by more than {}".format(SPLIT_TOL)))
            ok = ok and sp
        log.append("rule 4: S3 against {}: {}{} -> {}".format(
            name, why, "; " + "; ".join(extra) if extra else "", "S3" if ok else name))
        if ok:
            cur, name = s3, "S3"
    return {"chosen": name, "score": cur["score"], "pair": cur["pair"], "log": log}


# -- Section 4 ----------------------------------------------------------------------------------

def _games(grid, provider):
    return sorted(g for g, r in grid["matches"].items() if r["provider"] == provider)


def ref_ranges(grid):
    """{stat: (min, max)} of the four 8e statistics over the 28 references as provided."""
    vals = {k: [] for k in E8}
    for g, r in grid["matches"].items():
        ref = r["ref_lengths"][REF_AS_PROVIDED[r["provider"]]]
        for k in E8:
            if ref[k] is not None:
                vals[k].append(ref[k])
    return {k: (min(v), max(v)) for k, v in vals.items()}


def pair_index(grid, pair):
    pairs = [tuple(float(x) for x in p) for p in grid["pairs"]]
    return pairs.index(tuple(float(x) for x in pair))


def cell_gain(grid, i, test, ref):
    prov = {v: k for k, v in PROVIDER_TEST.items()}[test]
    return float(np.mean([grid["matches"][g]["pairs"][i][ref]["delta"]
                          for g in _games(grid, prov)]))


def worst_gain(grid, i, test, ref):
    prov = {v: k for k, v in PROVIDER_TEST.items()}[test]
    return float(min(grid["matches"][g]["pairs"][i][ref]["delta"] for g in _games(grid, prov)))


def merge_ok(grid, i):
    """Step 2(b): mean merge share network <= c23 on T-M (as provided) and on T-D."""
    out = {}
    for prov in ("metrica", "dfl"):
        gs = _games(grid, prov)
        net = float(np.mean([grid["matches"][g]["pairs"][i]["provided"]["merge"] for g in gs]))
        c23 = float(np.mean([grid["matches"][g]["c23"]["provided"]["merge"] for g in gs]))
        out[PROVIDER_TEST[prov]] = {"net": net, "c23": c23, "ok": net <= c23}
    return all(v["ok"] for v in out.values()), out


def pair_rows(grid, step3="five"):
    """Steps 2 and 3 for every pair: [{"pair", "eligible", "e8", "merge", "cells", "score"}]."""
    ranges = ref_ranges(grid)
    dev = _games(grid, "metrica") + _games(grid, "dfl")
    if len(dev) != 8:
        raise SystemExit("the grid holds {} development matches, not 8".format(len(dev)))
    rows = []
    for i, pair in enumerate(grid["pairs"]):
        e8 = {}
        for k in E8:
            med = float(np.median([grid["matches"][g]["pairs"][i]["lengths"][k] for g in dev]))
            lo, hi = ranges[k]
            e8[k] = {"median": med, "range": [lo, hi], "inside": lo <= med <= hi}
        mok, merge = merge_ok(grid, i)
        cells = {"{} {}".format(t, r): cell_gain(grid, i, t, r) for t, r in CELLS}
        rows.append({"pair": [float(x) for x in pair], "index": i,
                     "eligible": all(v["inside"] for v in e8.values()) and mok,
                     "e8": e8, "merge": merge, "cells": cells,
                     "score": step3_score(cells, step3)})
    return rows


def rank(rows):
    """Eligible rows by Score descending; ties to the lower H-A, then the lower alive-dead."""
    ok = [r for r in rows if r["eligible"]]
    return sorted(ok, key=lambda r: (-r["score"], r["pair"][0], r["pair"][1]))


def prior_choice(grids, pair, base_rows, step3="five"):
    """Step 6 at `pair`. grids: {bonus: grid}; base_rows: pair_rows(grid0)."""
    g0 = grids[0]
    i0 = pair_index(g0, pair)
    base = {(t, r): cell_gain(g0, i0, t, r) for t, r in (("T-M", "provided"), ("T-D", "provided"),
                                                         ("T-S", "possgroup"))}
    worst0 = {(t, r): worst_gain(g0, i0, t, r) for (t, r) in base}
    out = {"bonus": 0.0, "candidates": {}}
    acceptable = []
    for b in sorted(k for k in grids if k):
        g = grids[b]
        i = pair_index(g, pair)
        means = {k: cell_gain(g, i, *k) for k in base}
        worst = {k: worst_gain(g, i, *k) for k in base}
        not_lower = all(means[k] >= base[k] - PRIOR_TOL - 1e-12 for k in base)
        worst_up = sum(worst[k] > worst0[k] for k in base)
        mok, merge = merge_ok(g, i)
        cells = {"{} {}".format(t, r): cell_gain(g, i, t, r) for t, r in CELLS}
        rec = {"means": {"{} {}".format(*k): v for k, v in means.items()},
               "worst": {"{} {}".format(*k): v for k, v in worst.items()},
               "mean_not_lower_by_more_than_0.005": not_lower,
               "worst_higher_on": worst_up, "merge_ok": mok, "merge": merge,
               "score": step3_score(cells, step3)}
        rec["acceptable"] = bool(not_lower and worst_up >= 2 and mok)
        out["candidates"][str(b)] = rec
        if rec["acceptable"]:
            acceptable.append((b, rec["score"]))
    if acceptable:
        out["bonus"] = float(sorted(acceptable, key=lambda x: (-x[1], x[0]))[0][0])
    out["base"] = {"means": {"{} {}".format(*k): v for k, v in base.items()},
                   "worst": {"{} {}".format(*k): v for k, v in worst0.items()}}
    return out


def drop_one(rows):
    """For information: the best eligible pair if each of the five cells were dropped in turn."""
    out = {}
    names = ["{} {}".format(t, r) for t, r in CELLS]
    for drop in names:
        ok = [r for r in rows if r["eligible"]]
        if not ok:
            out[drop] = None
            continue
        sc = lambda r: min(v for k, v in r["cells"].items() if k != drop)  # noqa: E731
        best = sorted(ok, key=lambda r: (-sc(r), r["pair"][0], r["pair"][1]))[0]
        out[drop] = {"pair": best["pair"], "score": sc(best)}
    return out


def cost_choice(grid0, grid1=None, grid2=None, step3="five"):
    """Steps 1 to 6. Returns the record (ranking, chosen pair or c23 ships, prior).

    `step3`: "five" (as frozen) or "mean2" (deviation 8.4; round 5 Section 3).
    """
    step3_score({k: 0.0 for k in ["{} {}".format(t, r) for t, r in CELLS]}, step3)
    from tools.learned_spike import r4_screen
    want = [list(p) for p in r4_screen.grid_pairs()]
    if [[float(x) for x in p] for p in grid0["pairs"]] != [[float(x) for x in p] for p in want]:
        raise SystemExit("grid0 does not hold the pairs of r4_screen.grid_pairs()")
    if float(grid0.get("prior_bonus", 0.0)) != 0.0:
        raise SystemExit("grid0 must be decoded with prior bonus 0")
    rows = pair_rows(grid0, step3)
    ranking = rank(rows)
    rec = {"rows": rows, "ranking": [r["pair"] for r in ranking], "step3": step3}
    if not ranking:
        rec.update(decision="c23 ships", reason="no pair is eligible (step 2)")
        return rec
    best = ranking[0]
    rec["pair"] = best["pair"]
    rec["score"] = best["score"]
    floor = {k: best["cells"][k] for k in ("T-M provided", "T-D provided")}
    rec["floor"] = {k: {"gain": v, "ok": v >= FLOOR - 1e-12} for k, v in floor.items()}
    if not all(v["ok"] for v in rec["floor"].values()):
        rec.update(decision="c23 ships", reason="floor (step 5) failed at {}".format(best["pair"]))
        return rec
    rec["decision"] = "pair {}".format(best["pair"])
    if step3 == "five":
        rec["deciding_cell"] = min(best["cells"], key=lambda k: best["cells"][k])
        rec["drop_one"] = drop_one(rows)
    grids = {0: grid0}
    for b, g in ((1, grid1), (2, grid2)):
        if g is not None:
            if float(g.get("prior_bonus", -1)) != float(b):
                raise SystemExit("grid{} was not decoded with prior bonus {}".format(b, b))
            grids[b] = g
    if len(grids) > 1:
        rec["prior"] = prior_choice(grids, best["pair"], rows, step3)
        rec["decision"] += ", prior bonus {:g}".format(rec["prior"]["bonus"])
    return rec


def costs_md(rec, title):
    w = ["# " + title, ""]
    w.append("Decision: **{}**{}".format(rec["decision"], " ({})".format(rec["reason"])
                                         if rec.get("reason") else ""))
    w.append("")
    w.append("Step 3: {}".format(
        "mean of T-M provided and T-D (deviation 8.4)" if rec.get("step3") == "mean2"
        else "smallest of the five cells (as frozen)"))
    if rec.get("deciding_cell"):
        w.append("")
        w.append("Cell that decided the pair (the smallest at {}): **{}**.".format(
            rec["pair"], rec["deciding_cell"]))
        w.append("For information, the best pair if one cell were dropped: " + "; ".join(
            "without {}: {}".format(k, "none" if v is None else "{} (Score {:+.4f})".format(
                v["pair"], v["score"])) for k, v in rec["drop_one"].items()) + ".")
    w += ["", "| rank | pair | Score | T-M prov | T-M loose | T-D | T-S pg | T-S phases | "
          "8e medians (pos50 pos90 d50 d90) | merge T-M / T-D net vs c23 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    order = {tuple(p): k + 1 for k, p in enumerate(rec["ranking"])}
    for r in sorted(rec["rows"], key=lambda r: (order.get(tuple(r["pair"]), 999), -r["score"])):
        c = r["cells"]
        w.append("| {} | ({:g}, {:g}) | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} "
                 "| {} | {:.3f}/{:.3f} vs {:.3f}/{:.3f} |".format(
                     order.get(tuple(r["pair"]), "not eligible"), r["pair"][0], r["pair"][1],
                     r["score"], c["T-M provided"], c["T-M loose"], c["T-D provided"],
                     c["T-S possgroup"], c["T-S phases"],
                     " ".join("{:.1f}{}".format(r["e8"][k]["median"],
                                                "" if r["e8"][k]["inside"] else "!")
                              for k in E8),
                     r["merge"]["T-M"]["net"], r["merge"]["T-D"]["net"],
                     r["merge"]["T-M"]["c23"], r["merge"]["T-D"]["c23"]))
    if "prior" in rec:
        w += ["", "Prior at the pair: bonus {:g}".format(rec["prior"]["bonus"]), "",
              "```", json.dumps(rec["prior"], indent=1, sort_keys=True), "```"]
    w.append("")
    w.append("(`!` = median outside the range of the 28 references as provided.)")
    return "\n".join(w) + "\n"


# -- Round 5 screening score (LEARNED_STAGE_PROTOCOL_R5.md Section 2) -------------------------

def screen5_score(res):
    """The round 5 screening score of a score.json of r4_screen score (T-M and T-D summaries
    holding the six R5_SCREEN_PAIRS). Returns {"rows", "score", "pair"}."""
    rows = []
    for pair in R5_SCREEN_PAIRS:
        cells = {}
        for test in ("T-M", "T-D"):
            if test not in res.get("summary", {}):
                raise SystemExit("score.json has no {} summary".format(test))
            hit = [r for r in res["summary"][test]
                   if [float(x) for x in r["pair"]] == list(pair)]
            if not hit:
                raise SystemExit("score.json has no pair {} on {} (run r4_screen score "
                                 "--r5)".format(pair, test))
            cells[test] = hit[0]
        eligible = all(cells[t]["merge_net"] <= cells[t]["merge_c23"] for t in cells)
        rows.append({"pair": list(pair), "gain_T-M": cells["T-M"]["gain"],
                     "gain_T-D": cells["T-D"]["gain"],
                     "mean2": float(np.mean([cells["T-M"]["gain"], cells["T-D"]["gain"]])),
                     "merge_T-M": [cells["T-M"]["merge_net"], cells["T-M"]["merge_c23"]],
                     "merge_T-D": [cells["T-D"]["merge_net"], cells["T-D"]["merge_c23"]],
                     "eligible": bool(eligible)})
    ok = [r for r in rows if r["eligible"]]
    # Ties (not expected with float gains): the lower H-A, then the lower alive-dead cost.
    best = sorted(ok, key=lambda r: (-r["mean2"], r["pair"][0], r["pair"][1]))[0] if ok else None
    return {"rows": rows, "score": best["mean2"] if best else None,
            "pair": best["pair"] if best else None,
            "networks": res.get("networks"), "rehearsal": bool(res.get("rehearsal")),
            "rule": "max over the six pairs with merge share net <= c23 on T-M and T-D of the "
                    "mean of the T-M (as provided) and T-D mean gains"}


def screen5_md(rec, title):
    w = ["# " + title, ""]
    if rec.get("rehearsal"):
        w += ["**REHEARSAL: not a result.**", ""]
    w.append("Score: **{}** at {}".format(
        "none (no eligible pair)" if rec["score"] is None else "{:+.4f}".format(rec["score"]),
        rec["pair"]))
    w += ["", "| pair | T-M gain | T-D gain | mean | merge T-M net / c23 | merge T-D net / c23 "
          "| eligible |", "|---|---|---|---|---|---|---|"]
    for r in rec["rows"]:
        w.append("| ({:g}, {:g}) | {:+.4f} | {:+.4f} | {:+.4f} | {:.3f} / {:.3f} | {:.3f} / "
                 "{:.3f} | {} |".format(r["pair"][0], r["pair"][1], r["gain_T-M"],
                                         r["gain_T-D"], r["mean2"], r["merge_T-M"][0],
                                         r["merge_T-M"][1], r["merge_T-D"][0],
                                         r["merge_T-D"][1], "yes" if r["eligible"] else "no"))
    return "\n".join(w) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen")
    s.add_argument("--s1", required=True)
    s.add_argument("--s1p", required=True)
    s.add_argument("--s2")
    s.add_argument("--s2b")
    s.add_argument("--s3")
    s.add_argument("--out")
    c = sub.add_parser("costs")
    c.add_argument("--grid0", required=True)
    c.add_argument("--grid1")
    c.add_argument("--grid2")
    c.add_argument("--out", help="output stem (writes .json and .md)")
    c.add_argument("--title", default="Round 4 decoder costs and prior (Section 4)")
    c.add_argument("--step3", choices=STEP3, default="five",
                   help="step 3: five (as frozen) or mean2 (deviation 8.4, round 5 Section 3)")
    s5 = sub.add_parser("screen5", help="round 5 screening score of one score.json")
    s5.add_argument("--score", required=True)
    s5.add_argument("--out", help="output stem (writes .json and .md)")
    s5.add_argument("--title", default="Round 5 screening score (Section 2)")
    a = p.parse_args(argv)
    if a.cmd == "screen5":
        rec = screen5_score(load(a.score))
        rec["input"] = a.score
        text = screen5_md(rec, a.title)
        print(text)
        if a.out:
            with open(a.out + ".json", "w") as h:
                json.dump(rec, h, indent=1, sort_keys=True)
            with open(a.out + ".md", "w") as h:
                h.write(text)
        return 0
    if a.cmd == "screen":
        summ = {k: recipe_summary(load(v)) if v else None
                for k, v in (("s1", a.s1), ("s1p", a.s1p), ("s2", a.s2), ("s2b", a.s2b),
                             ("s3", a.s3))}
        rec = screen_choice(summ["s1"], summ["s1p"], summ["s2"], summ["s3"], summ["s2b"])
        rec["inputs"] = {k: v for k, v in vars(a).items() if k.startswith("s")}
        rec["summaries"] = summ
        print("\n".join(rec["log"]))
        print("chosen: {}".format(rec["chosen"]))
        if a.out:
            with open(a.out, "w") as h:
                json.dump(rec, h, indent=1, sort_keys=True)
        return 0
    rec = cost_choice(load(a.grid0), load(a.grid1) if a.grid1 else None,
                      load(a.grid2) if a.grid2 else None, a.step3)
    text = costs_md(rec, a.title)
    print(text)
    if a.out:
        with open(a.out + ".json", "w") as h:
            json.dump(rec, h, indent=1, sort_keys=True)
        with open(a.out + ".md", "w") as h:
            h.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
