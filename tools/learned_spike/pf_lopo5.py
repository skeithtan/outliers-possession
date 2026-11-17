"""F5 per-frame member + arm A network: left-out-provider and out-of-fold evaluation (round 6). Not part of the submission.

    # pf venv
    python -m tools.learned_spike.pf_lopo5 lopo --out OUT/lopo_F5      # step 3
    python -m tools.learned_spike.pf_lopo5 oof --out OUT/oof_eval --pair HA AD   # step 4

Network (arm A): recipe AQCSJ, seeds 0-3, left-out-provider networks; per-network 5 fps
probabilities (mean of the four symmetric copies) from r4_screen's cache
(r6_work/score_AQCSJ/p5_cache, key <game>__<md5[:12] of net.pt>), averaged over the seeds and
rounded once (decoders.ensemble_p5), engine.learned.upsample. Out of fold: r6_train/oof_p5
(Metrica, DFL; computed on the still-ball-removed DFL input) and, for SkillCorner, the fold
models r6_train/folds/fold<k>/net_s<s>.pt on r5_work/ca2s features (cached by oof-net-sc in the
torch venv, the rounding of r4_screen.ensemble).
Per-frame member F5: pf_train fit-all --variant F5 (left-out provider, or out of fold with the
fold's test matches left out), predicted on pf_features eval views (feats_F5/eval).
Ensemble: 0.5 x F5 + 0.5 x network (frame probabilities). Decoder: engine.learned.decode_model,
kickoff rule; prior bonus b: engine.learned.apply_prior with c23's output on the same input.

Cost rule (coordinator, 13 Nov, round 4's rule reduced): pairs with team-change cost in
HA_GRID and alive-dead cost in AD_GRID, team change at most twice alive-dead; eligible if the
mean merge share is at most c23's on T-M and on T-D; Score = mean of the T-M and T-D gains;
highest wins, ties to the lower team change (then lower alive-dead). Prior at that pair:
r4_choose step 6 (bonus 1 or 2 replaces 0 only if on each of T-M, T-D, T-S possession groups the
mean gain is not lower than bonus 0's by more than 0.005, the worst match's gain is higher on at
least two of the three, and the merge eligibility still holds; if both, the higher Score, then 1).
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from engine import data, learned  # noqa: E402
from tools.learned_spike import decoders, folds  # noqa: E402
from tools.learned_spike import pf_eval as pe  # noqa: E402
from tools.learned_spike import pf_features as pff  # noqa: E402
from tools.learned_spike import pf_train  # noqa: E402
from tools.learned_spike import r4_screen as rs  # noqa: E402
from tools.learned_spike import r5_measures as r5m  # noqa: E402

SP = pe.SP
R6 = SP + "/r6_pf"
ARMA_RECIPE = SP + "/r6_work/screen/AQCSJ"
ARMA_CACHE = SP + "/r6_work/score_AQCSJ/p5_cache"
ARMA_OOF = SP + "/r6_train/oof_p5"
ARMA_FOLDS = SP + "/r6_train/folds"
FEATS = R6 + "/feats_F5"
HA_GRID = (4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)
AD_GRID = (4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0)
GRID = [(a, b) for a in HA_GRID for b in AD_GRID if a <= 2 * b]
TESTS = ("T-M", "T-D", "T-S", "T-Sp")
PRIOR_TOL = 0.005


def log(text):
    print(time.strftime("%H:%M:%S"), text, file=sys.stderr, flush=True)


_MD5 = {}


def arma_lopo_p5(game):
    prov = folds.provider(pff.check(game))
    ps = []
    for s in range(4):
        path = os.path.join(ARMA_RECIPE, prov, "seed{}".format(s), "net.pt")
        if path not in _MD5:
            _MD5[path] = pe.md5_12(path)
        ps.append(np.load(os.path.join(ARMA_CACHE, "{}__{}.npy".format(game, _MD5[path]))))
    return decoders.ensemble_p5(ps)


def arma_oof_p5(game):
    if game.startswith("SC-"):
        return np.load(os.path.join(R6, "oof_armA_sc", game + ".npy"))
    return np.load(os.path.join(ARMA_OOF, game + "_p5.npz"))["p5"]


def f5_probs(models_dir, game, cache_dir):
    path = os.path.join(cache_dir, game + ".npy")
    if os.path.isfile(path):
        return np.load(path)
    with open(os.path.join(models_dir, "models.pkl"), "rb") as h:
        models = pickle.load(h)
    F = np.load(os.path.join(FEATS, "eval", pff.check(game) + ".npy"))
    P = pf_train.predict(models, F, "full")
    os.makedirs(cache_dir, exist_ok=True)
    np.save(path, P)
    return P


def c23_of(test, game, n):
    d = {"T-M": pe.ARGS.c23_dev, "T-D": pe.ARGS.c23_dfl, "T-S": pe.ARGS.c23_sc,
         "T-Sp": pe.ARGS.c23_sc}[test]
    return rs.read_c23(d, game, n)


def decode(fp, game, pair, bonus=0.0, rule=None):
    n, blocks, forced = pe.structure(game)
    if bonus:
        return learned.decode_model(fp, blocks, pair[0], pair[1], forced, rule_codes=rule,
                                    prior_bonus=float(bonus))
    return learned.decode_model(fp, blocks, pair[0], pair[1], forced)


def score_set(systems, tests, pairs, bonus=0.0, recall=False):
    """{system: {test: {game: [ {spellF1, merge, split, n, delta, (recall, d90)} per pair ]}}},
    plus c23 rows. systems: {name: fn(game) -> frame probs}."""
    out = {s: {t: {} for t in tests} for s in systems}
    c23rows = {t: {} for t in tests}
    for test in tests:
        for game in pe.games_of(test):
            ref, c23 = pe.refs(test, game)
            base = rs.measures(ref, c23)
            refc = rs.read_labels(pe.ARGS.cache, game)
            if recall:
                base.update(pe.extra(c23, game, refc))
            c23rows[test][game] = base
            rule = c23 if bonus else None
            for s, fn in systems.items():
                fp = fn(game)
                rows = []
                for pair in pairs:
                    codes = decode(fp, game, pair, bonus, rule)
                    m = rs.measures(ref, codes)
                    if recall:
                        m.update(pe.extra(codes, game, refc))
                    m["delta"] = m["spellF1"] - base["spellF1"]
                    rows.append(m)
                out[s][test][game] = rows
        log("scored {} ({} systems, {} pairs, bonus {:g})".format(test, len(systems), len(pairs),
                                                                  bonus))
    return out, c23rows


def summary(res, c23rows, i):
    """Per test at pair index i: gain, merge, split (system / c23), worst gain."""
    out = {}
    for t, d in res.items():
        gs = list(d)
        out[t] = {"gain": float(np.mean([d[g][i]["delta"] for g in gs])),
                  "worst": float(min(d[g][i]["delta"] for g in gs)),
                  "up": int(sum(d[g][i]["delta"] > 0 for g in gs)), "n": len(gs),
                  "merge": float(np.mean([d[g][i]["merge"] for g in gs])),
                  "split": float(np.mean([d[g][i]["split"] for g in gs])),
                  "merge_c23": float(np.mean([c23rows[t][g]["merge"] for g in gs])),
                  "split_c23": float(np.mean([c23rows[t][g]["split"] for g in gs])),
                  "poss": float(np.mean([d[g][i]["n"] for g in gs])),
                  "poss_c23": float(np.mean([c23rows[t][g]["n"] for g in gs]))}
        if "recall" in d[gs[0]][i]:
            for who, rows in (("sys", [d[g][i] for g in gs]), ("c23", [c23rows[t][g]
                                                                      for g in gs])):
                rec = r5m.summarise_recall([x for r in rows for x in r["recall"]])
                out[t]["recall_view_" + who] = rec["ball_in_view"]["share_D"]
                out[t]["recall_hidden_" + who] = rec["ball_hidden"]["share_D"]
                out[t]["segs_view"] = rec["ball_in_view"]["segments"]
                out[t]["segs_hidden"] = rec["ball_hidden"]["segments"]
                out[t]["d90_" + who] = float(np.mean([r["d90"] for r in rows]))
            out[t]["d90_ref"] = float(np.mean([r5m.d_per_90(rs.read_labels(pe.ARGS.cache, g),
                                                            g) for g in gs]))
    return out


def cost_rule(res, c23rows, pairs):
    rows = []
    for i, p in enumerate(pairs):
        s = summary(res, c23rows, i)
        ok = all(s[t]["merge"] <= s[t]["merge_c23"] for t in ("T-M", "T-D"))
        rows.append({"pair": list(p), "eligible": bool(ok),
                     "score": 0.5 * (s["T-M"]["gain"] + s["T-D"]["gain"]),
                     "T-M": s["T-M"]["gain"], "T-D": s["T-D"]["gain"],
                     "merge_TM": [s["T-M"]["merge"], s["T-M"]["merge_c23"]],
                     "merge_TD": [s["T-D"]["merge"], s["T-D"]["merge_c23"]]})
    ok = sorted([r for r in rows if r["eligible"]],
                key=lambda r: (-r["score"], r["pair"][0], r["pair"][1]))
    return rows, (ok[0] if ok else None)


def prior_rule(base, cands):
    """base, cands[b]: summary dicts at the pair (tests T-M, T-D, T-S)."""
    keys = ("T-M", "T-D", "T-S")
    out = {"bonus": 0.0, "candidates": {}}
    acc = []
    for b, s in sorted(cands.items()):
        not_lower = all(s[t]["gain"] >= base[t]["gain"] - PRIOR_TOL - 1e-12 for t in keys)
        worst_up = sum(s[t]["worst"] > base[t]["worst"] for t in keys)
        mok = all(s[t]["merge"] <= s[t]["merge_c23"] for t in ("T-M", "T-D"))
        score = 0.5 * (s["T-M"]["gain"] + s["T-D"]["gain"])
        ok = bool(not_lower and worst_up >= 2 and mok)
        out["candidates"][str(b)] = {"gains": {t: s[t]["gain"] for t in keys},
                                     "worst": {t: s[t]["worst"] for t in keys},
                                     "not_lower": not_lower, "worst_higher_on": worst_up,
                                     "merge_ok": mok, "score": score, "acceptable": ok}
        if ok:
            acc.append((b, score))
    if acc:
        out["bonus"] = float(sorted(acc, key=lambda x: (-x[1], x[0]))[0][0])
    return out


def cmd_lopo(args):
    pcache = os.path.join(args.out, "probs_F5")
    models = {p: os.path.join(R6, "models_all", "F5", p) for p in ("metrica", "dfl",
                                                                   "skillcorner")}

    def net(game):
        return learned.upsample(arma_lopo_p5(game), data.frame_count(game))

    def f5(game):
        return f5_probs(models[folds.provider(game)], game, pcache)

    def ens(game):
        return 0.5 * f5(game) + 0.5 * net(game)
    systems = {"armA": net, "F5": f5, "ens0.5": ens}
    rec = {"generated": time.strftime("%Y-%m-%d %H:%M"), "grid": [list(p) for p in GRID]}
    res, c23rows = score_set(systems, TESTS, [(12.0, 12.0)], recall=True)
    rec["at_12_12"] = {s: summary(res[s], c23rows, 0) for s in systems}
    pe.write_json(os.path.join(args.out, "lopo_F5.json"), rec)
    grid, c23g = score_set({"armA": net, "ens0.5": ens}, ("T-M", "T-D"), GRID)
    rec["cost_rule"] = {}
    for s in ("armA", "ens0.5"):
        rows, best = cost_rule(grid[s], c23g, GRID)
        rec["cost_rule"][s] = {"rows": rows, "chosen": best}
        pe.write_json(os.path.join(args.out, "lopo_F5.json"), rec)
    rec["prior"] = {}
    for s, fn in (("ens0.5", ens), ("armA", net)):
        best = rec["cost_rule"][s]["chosen"]
        if best is None:
            continue
        pair = tuple(best["pair"])
        sums = {}
        for b in (0.0, 1.0, 2.0):
            r, c = score_set({s: fn}, ("T-M", "T-D", "T-S", "T-Sp"), [pair], bonus=b)
            sums[b] = summary(r[s], c, 0)
        rec["prior"][s] = {"pair": list(pair), "summaries": {str(b): v for b, v in sums.items()},
                           "rule": prior_rule(sums[0.0], {1.0: sums[1.0], 2.0: sums[2.0]})}
        pe.write_json(os.path.join(args.out, "lopo_F5.json"), rec)
    return 0


def cmd_oof(args):
    pcache = os.path.join(args.out, "probs_F5_oof")
    pair = tuple(args.pair)

    def fold_of(game):
        for k in range(folds.N_FOLDS):
            if game in folds.fold(k, pff=False)["test"]:
                return k
        raise ValueError(game)

    def net(game):
        return learned.upsample(arma_oof_p5(game), data.frame_count(game))

    def f5(game):
        k = fold_of(game)
        P = f5_probs(os.path.join(R6, "models_oof", "fold{}".format(k), "F5", "all"), game,
                     pcache)
        return P

    def ens(game):
        return 0.5 * f5(game) + 0.5 * net(game)
    systems = {"armA": net, "F5": f5, "ens0.5": ens}
    pairs = [(12.0, 12.0)] + ([pair] if pair != (12.0, 12.0) else [])
    rec = {"generated": time.strftime("%Y-%m-%d %H:%M"), "pairs": [list(p) for p in pairs],
           "bonus": args.bonus}
    for b in sorted({0.0, float(args.bonus)}):
        res, c23rows = score_set(systems, ("T-M", "T-D", "T-S"), pairs, bonus=b)
        rec["bonus_{:g}".format(b)] = {
            "summary": {s: [summary(res[s], c23rows, i) for i in range(len(pairs))]
                        for s in systems},
            "per_match": {s: {t: {g: [r["delta"] for r in rows] for g, rows in d.items()}
                              for t, d in res[s].items()} for s in systems}}
        pe.write_json(os.path.join(args.out, "oof_F5.json"), rec)
    # save the out-of-fold per-frame probabilities (25 fps) as asked
    os.makedirs(os.path.join(R6, "oof_F5"), exist_ok=True)
    for g in folds.ALL:
        np.save(os.path.join(R6, "oof_F5", g + ".npy"), f5(g).astype(np.float32))
    return 0


def f4(v):
    return "-" if v is None else "{:+.4f}".format(v)


def f3(v):
    return "-" if v is None else "{:.3f}".format(v)


def cmd_report_lopo(args):
    r = __import__("json").load(open(os.path.join(args.out, "lopo_F5.json")))
    A = r["at_12_12"]
    L = ["## Left-out-provider gains at (12,12), prior 0", "",
         "| system | T-M | T-D | T-S | T-Sp | mean T-M,T-D |", "|---|---|---|---|---|---|"]
    for s in ("armA", "F5", "ens0.5"):
        L.append("| {} | {} | {} | {} | {} | {} |".format(s, *[f4(A[s][t]["gain"]) for t in TESTS],
                                                       f4(0.5 * (A[s]["T-M"]["gain"] +
                                                                 A[s]["T-D"]["gain"]))))
    L += ["", "Matches up (gain > 0) / worst match gain:", "", "| system | T-M | T-D | T-S | T-Sp |",
          "|---|---|---|---|---|"]
    for s in ("armA", "F5", "ens0.5"):
        L.append("| {} | ".format(s) + " | ".join("{}/{} ({})".format(
            A[s][t]["up"], A[s][t]["n"], f4(A[s][t]["worst"])) for t in TESTS) + " |")
    L += ["", "Merge / split shares (system; c23) and possessions per match (system / c23):", "",
          "| system | test | merge | split | possessions |", "|---|---|---|---|---|"]
    for s in ("armA", "F5", "ens0.5"):
        for t in TESTS:
            x = A[s][t]
            L.append("| {} | {} | {} ({}) | {} ({}) | {:.0f} / {:.0f} |".format(
                s, t, f3(x["merge"]), f3(x["merge_c23"]), f3(x["split"]), f3(x["split_c23"]),
                x["poss"], x["poss_c23"]))
    t = "T-S"
    L += ["", "## Dead-ball recall on SkillCorner (stoppages >= 2 s; share labelled D) and "
          "dead-ball segments per 90 min", "",
          "Stoppages: {} with the ball in view, {} hidden. c23: in view {}, hidden {}, "
          "segments/90 {:.1f}; reference {:.1f}.".format(
              A["armA"][t]["segs_view"], A["armA"][t]["segs_hidden"],
              f3(A["armA"][t]["recall_view_c23"]), f3(A["armA"][t]["recall_hidden_c23"]),
              A["armA"][t]["d90_c23"], A["armA"][t]["d90_ref"]), "",
          "| system | in view | hidden | segments/90 |", "|---|---|---|---|"]
    for s in ("armA", "F5", "ens0.5"):
        x = A[s][t]
        L.append("| {} | {} | {} | {:.1f} |".format(s, f3(x["recall_view_sys"]),
                                                    f3(x["recall_hidden_sys"]), x["d90_sys"]))
    for s, cr in r.get("cost_rule", {}).items():
        ch = cr["chosen"]
        L += ["", "## Cost rule: {}".format(s), "",
              "Eligible pairs: {} of {}. Chosen: {}.".format(
                  sum(x["eligible"] for x in cr["rows"]), len(cr["rows"]),
                  "none (c23 would ship)" if ch is None else "({:g},{:g}) Score {} (T-M {}, T-D {})"
                  .format(ch["pair"][0], ch["pair"][1], f4(ch["score"]), f4(ch["T-M"]),
                          f4(ch["T-D"]))), "",
              "Top eligible pairs:", "", "| pair | Score | T-M | T-D | merge T-M (c23) | "
              "merge T-D (c23) |", "|---|---|---|---|---|---|"]
        ok = sorted([x for x in cr["rows"] if x["eligible"]],
                    key=lambda x: (-x["score"], x["pair"][0], x["pair"][1]))[:8]
        for x in ok:
            L.append("| ({:g},{:g}) | {} | {} | {} | {} ({}) | {} ({}) |".format(
                x["pair"][0], x["pair"][1], f4(x["score"]), f4(x["T-M"]), f4(x["T-D"]),
                f3(x["merge_TM"][0]), f3(x["merge_TM"][1]), f3(x["merge_TD"][0]),
                f3(x["merge_TD"][1])))
        best_any = max(cr["rows"], key=lambda x: x["score"])
        L.append("")
        L.append("Best pair ignoring eligibility: ({:g},{:g}) Score {} (eligible: {}).".format(
            best_any["pair"][0], best_any["pair"][1], f4(best_any["score"]),
            best_any["eligible"]))
    for s, pr in r.get("prior", {}).items():
        L += ["", "## Prior at the chosen pair ({:g},{:g}): {}".format(pr["pair"][0], pr["pair"][1],
                                                                      s), "",
              "| bonus | T-M | T-D | T-S | T-Sp | worst T-M / T-D / T-S | merge T-M / T-D "
              "(c23) |", "|---|---|---|---|---|---|---|"]
        for b, sm in sorted(pr["summaries"].items()):
            L.append("| {} | {} | {} | {} | {} | {} / {} / {} | {} / {} ({} / {}) |".format(
                b, *[f4(sm[t]["gain"]) for t in TESTS], *[f4(sm[t]["worst"]) for t in
                                                          ("T-M", "T-D", "T-S")],
                f3(sm["T-M"]["merge"]), f3(sm["T-D"]["merge"]), f3(sm["T-M"]["merge_c23"]),
                f3(sm["T-D"]["merge_c23"])))
        L += ["", "Step 6 decision: bonus {:g}. Candidates: {}".format(
            pr["rule"]["bonus"], __import__("json").dumps(
                {b: {k: v for k, v in c.items() if k in ("not_lower", "worst_higher_on",
                                                         "merge_ok", "acceptable")}
                 for b, c in pr["rule"]["candidates"].items()}))]
    print("\n".join(L))
    return 0


def cmd_report_oof(args):
    r = __import__("json").load(open(os.path.join(args.out, "oof_F5.json")))
    L = []
    for key in sorted(k for k in r if k.startswith("bonus_")):
        R = r[key]
        for i, pair in enumerate(r["pairs"]):
            L += ["## Out of fold, pair ({:g},{:g}), prior {}".format(pair[0], pair[1],
                                                                      key[6:]), "",
                  "| system | T-M (3) | T-D (5) | dev mean (8) | dev matches up | T-S (20) | "
                  "merge T-S (c23) | possessions T-S (c23) |", "|---|---|---|---|---|---|---|---|"]
            for s in ("armA", "F5", "ens0.5"):
                S = R["summary"][s][i]
                pm = R["per_match"][s]
                dev = [pm["T-M"][g][i] for g in pm["T-M"]] + [pm["T-D"][g][i] for g in pm["T-D"]]
                L.append("| {} | {} | {} | {} | {}/8 (min {}) | {} | {} ({}) | {:.0f} ({:.0f}) |"
                         .format(s, f4(S["T-M"]["gain"]), f4(S["T-D"]["gain"]),
                                 f4(float(np.mean(dev))), sum(d > 0 for d in dev),
                                 f4(min(dev)), f4(S["T-S"]["gain"]), f3(S["T-S"]["merge"]),
                                 f3(S["T-S"]["merge_c23"]), S["T-S"]["poss"],
                                 S["T-S"]["poss_c23"]))
            L += ["", "Per development match (gain over c23):", "",
                  "| match | armA | F5 | ens0.5 |", "|---|---|---|---|"]
            pmA = R["per_match"]
            for t in ("T-M", "T-D"):
                for g in pmA["armA"][t]:
                    L.append("| {} | {} | {} | {} |".format(g, *[f4(pmA[s][t][g][i])
                                                                 for s in ("armA", "F5",
                                                                           "ens0.5")]))
            L.append("")
    print("\n".join(L))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("lopo")
    a.add_argument("--out", required=True)
    b = sub.add_parser("oof")
    b.add_argument("--out", required=True)
    b.add_argument("--pair", type=float, nargs=2, default=[12.0, 12.0])
    b.add_argument("--bonus", type=float, default=0.0)
    c = sub.add_parser("report-lopo")
    c.add_argument("--out", required=True)
    d = sub.add_parser("report-oof")
    d.add_argument("--out", required=True)
    args = p.parse_args(argv)
    return {"lopo": cmd_lopo, "oof": cmd_oof, "report-lopo": cmd_report_lopo,
            "report-oof": cmd_report_oof}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
