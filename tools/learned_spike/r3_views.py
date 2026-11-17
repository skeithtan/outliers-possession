"""Round 3, recorded only: the development matches under three reference views, three inputs and
the whole cost grid, and three candidate cost rules. Not part of the submission.

    $E -m tools.learned_spike.r3_views refs  --work W                     # the loose-touch variant
    $E -m tools.learned_spike.r3_views grid  --work W --run RUN --p5 DIR --npz-dir W/npz \
        --c23-dev DIR [--engine-cache DIR] [--out W/views.json]

Nothing here is a criterion of LEARNED_STAGE_PROTOCOL_R3.md; the frozen criteria use what the
protocol says (r3_eval.py). This tool measures what the choice of reference convention costs.

Reference views (development matches only):
- "provided": the reference as provided (the build cache `labels/<game>.npz` key `raw`; for
  Metrica it is rebuilt here from the events with tools/events_gt.py and must equal the cache).
- "loose": the loose-touch variant of LEARNINGS F2 (2 Nov 2026), Metrica only: a RECOVERY that a
  BALL LOST or BALL OUT of the same team and period follows within 2 frames is no owner event,
  nor its CARRY of at most 2 frames (tools/analysis/redirect_forensics.loose_touch_events, the
  team's code, unchanged), then tools/events_gt.derive(). DFL has no event-derived variant: its
  "loose" view is its reference as provided (stated in every table).
- "ctl2": the 2 s controlled view, control_labels(provided, 50).

Inputs: "raw" (the out-of-fold probabilities of RUN, --p5), "matched" (the matched tracking view)
and "stress" (corrupt.py's stress view, the criterion 4 view: `--ball matched`, default
fragmentation, swaps and ghosts, seed from the match name). On matched and stress the
probabilities are computed here with the numpy engine (engine.learned) from the fold weights of
the match (W/npz/fold{k}.npz), on the guarded tracking that engine M reads; the M labels of the
stress view at the scored costs are checked against the engine's own stress run when it exists.
c23 on each input comes from its engine run (raw: --c23-dev; matched and stress: W/runs).

Cost grid: H-A {0.25 .. 32, 64, 128} x alive-dead {0.5 .. 32, 64} (80 pairs); rule B's 56 pairs
are the subset with H-A <= 32 and alive-dead <= 32.

Candidate cost rules (each written so that it can be frozen before any run on the competition
match; each reads the 8 development matches only, never SkillCorner, PFF or BAR-ATH):
- a. rule B as frozen: on raw input, the pair of rule B's 56 with the best mean of the Metrica
  mean and the DFL mean spellF1 against the references as provided.
- b. best worst case (R3_DEVIATION_COSTS.md): for a pair p, provider q in {Metrica, DFL}, view v
  in {provided, loose} and input i in {raw, matched, stress}, G(p, q, v, i) = [mean spellF1 of M
  at p over the matches of q] minus [the same for c23]; the score of p is the minimum over the
  twelve (q, v, i); the pick is the pair of the 80 with the highest score. (DFL's loose view is
  its reference as provided, so DFL enters with its own worst input.)
- bp (recorded only): as b, but the gain is first averaged over the two providers (the
  provider-balanced mean), then the minimum is taken over the six (v, i). With DFL's views
  identical this dilutes the Metrica worst case; it picked a lower pair on seed 0.
- c. equal weight: the score of p is the mean over the two views and the three inputs of the
  provider-balanced mean spellF1 of M at p (equivalently of G, since c23's part is constant);
  the pick is the pair of the 80 with the highest score.
Ties go to the lower index (lower H-A, then lower alive-dead). Each rule is also applied leaving
each development match out, which gives the per-match pair with which that match is scored
(as rule B does), and the final pair is the pick on all eight.

Section 8e: the envelope of the 28 references as provided, and the union with the three Metrica
loose-touch references, recorded (the criterion is unchanged). Python 3.9.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation"),
           os.path.join(ROOT, "tools", "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from tools.learned_spike import folds  # noqa: E402

DEV = tuple(folds.OPEN)
METRICA = tuple(g for g in DEV if g.startswith("METRICA"))
GRID_HA = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
GRID_AD = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
GRID = [(a, b) for a in GRID_HA for b in GRID_AD]
RULE_B = [i for i, (a, b) in enumerate(GRID) if a <= 32.0 and b <= 32.0]
VIEWS = ("provided", "loose", "ctl2")
RULES = ("a", "b", "bp", "c")
INPUTS = ("raw", "matched", "stress")
EVENTS = {"METRICA-S1": ("Sample_Game_1/Sample_Game_1_RawEventsData.csv", "refs"),
          "METRICA-S2": ("Sample_Game_2/Sample_Game_2_RawEventsData.csv", "refs"),
          "METRICA-S3": ("Sample_Game_3/Sample_Game_3_events.json", "sample")}
REFS_ROOT = os.path.expanduser(os.path.join("~", "mtah2027", "refs"))
SAMPLE_ROOT = os.path.expanduser(os.path.join("~", "mtah2027", "sample-data", "data"))
MTAH_ROOT = os.path.expanduser(os.path.join("~", "mtah2027"))


def log(text):
    print(text, file=sys.stderr, flush=True)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(obj, handle, indent=1, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


# -- pure rules (tested) --------------------------------------------------------------------------

def provider_mean(per_match, games):
    """Mean of the Metrica mean and the DFL mean of {game: value} over `games`."""
    groups = {}
    for g in games:
        groups.setdefault(folds.provider(g), []).append(per_match[g])
    return float(np.mean([np.mean(groups[p]) for p in ("metrica", "dfl") if p in groups]))


def pick(score_of, candidates):
    """The index of `candidates` with the highest score_of(i); ties go to the first."""
    best, best_v = None, None
    for i in candidates:
        v = score_of(i)
        if best_v is None or v > best_v + 1e-12:
            best, best_v = i, v
    return best, best_v


def rule_scores(tab, base, games, rule):
    """score_of(i) of one rule over `games`.

    tab[view][input][game][i]: spellF1 of M at pair i; base[view][input][game]: c23's spellF1.
    """
    def g(i, v, inp):
        return (provider_mean({x: tab[v][inp][x][i] for x in games}, games)
                - provider_mean({x: base[v][inp][x] for x in games}, games))

    if rule == "a":
        return lambda i: provider_mean({x: tab["provided"]["raw"][x][i] for x in games}, games)
    def gp(i, v, inp, prov):
        sub = [x for x in games if folds.provider(x) == prov]
        return float(np.mean([tab[v][inp][x][i] - base[v][inp][x] for x in sub]))

    if rule == "b":
        return lambda i: min(gp(i, v, inp, prov) for prov in ("metrica", "dfl")
                             for v in ("provided", "loose") for inp in INPUTS)
    if rule == "bp":
        return lambda i: min(g(i, v, inp) for v in ("provided", "loose") for inp in INPUTS)
    if rule == "c":  # noqa: E501
        return lambda i: float(np.mean([g(i, v, inp) for v in ("provided", "loose")
                                        for inp in INPUTS]))
    raise ValueError(rule)


def apply_rule(tab, base, rule, inputs_present=INPUTS):
    """{"final": index, "nested": {game: index}} of a rule (a: rule B's 56 pairs; b, c: all)."""
    cands = RULE_B if rule == "a" else list(range(len(GRID)))
    out = {"nested": {}}
    out["final"], out["final_score"] = pick(rule_scores(tab, base, list(DEV), rule), cands)
    for game in DEV:
        rest = [g for g in DEV if g != game]
        out["nested"][game] = pick(rule_scores(tab, base, rest, rule), cands)[0]
    return out


# -- references -----------------------------------------------------------------------------------

def metrica_refs(game, cache):
    """(provided rebuilt, loose-touch variant, info) of one Metrica match from its events."""
    import events_gt
    import games
    import redirect_forensics as rf
    path, root = EVENTS[game]
    entry = games.GAMES[game]
    events = events_gt.load_any(os.path.join(SAMPLE_ROOT, path))
    home = events_gt.home_team_name(game, data_dir=REFS_ROOT if root == "refs" else SAMPLE_ROOT)
    args = (int(entry["n_video_frames"]), entry["offsets"], float(entry["fps"]))
    provided, _ = events_gt.derive(events, home, *args)
    kept, dropped = rf.loose_touch_events(events)
    loose, _ = events_gt.derive(kept, home, *args)
    cached = np.load(os.path.join(cache, "labels", game + ".npz"))["raw"].astype(np.int8)
    info = {"home": home, "recovery_dropped": sum(1 for e in dropped if e["type"] == "RECOVERY"),
            "recovery_total": sum(1 for e in events if e["type"] == "RECOVERY"),
            "carry_dropped": sum(1 for e in dropped if e["type"] == "CARRY"),
            "rebuilt_equals_cache": bool(np.array_equal(np.asarray(provided, np.int8), cached))}
    return np.asarray(provided, np.int8), np.asarray(loose, np.int8), info


def load_refs(work, cache):
    """{view: {game: codes}} of the 8 development matches (W/views/refs/ as written by refs)."""
    import possession_metrics as pm
    from analysis import control_reference as cref
    out = {v: {} for v in VIEWS}
    for game in DEV:
        provided = np.load(os.path.join(cache, "labels", game + ".npz"))["raw"].astype(np.int8)
        out["provided"][game] = provided
        path = os.path.join(work, "views", "refs", game + "_loose.npy")
        out["loose"][game] = np.load(path).astype(np.int8) if game in METRICA else provided
        labels = np.asarray(pm.LABELS)[provided]
        out["ctl2"][game] = np.array([{"H": 0, "A": 1, "D": 2, "X": 3}[str(v)] for v in
                                      cref.control_labels(labels, 50)], dtype=np.int8)
    return out


def cmd_refs(args):
    folder = os.path.join(args.work, "views", "refs")
    os.makedirs(folder, exist_ok=True)
    rec = {}
    for game in METRICA:
        provided, loose, info = metrica_refs(game, args.cache)
        if not info["rebuilt_equals_cache"]:
            raise SystemExit("{}: the rebuilt reference differs from the cache".format(game))
        np.save(os.path.join(folder, game + "_loose.npy"), loose)
        rec[game] = dict(info, possessions=ref_count(provided), possessions_loose=ref_count(loose),
                         loose_md5=__import__("hashlib").md5(loose.tobytes()).hexdigest())
        log("{}: {}".format(game, rec[game]))
    write_json(os.path.join(folder, "refs.json"), rec)
    print(json.dumps(rec, sort_keys=True))
    return 0


def ref_count(codes):
    from tools.learned_spike import r3_scope as r
    runs = r.pm.valid_runs(codes)
    return len(r.spells(codes, runs))


# -- probabilities on the three inputs ------------------------------------------------------------

def input_probs(game, inp, p5_dir, npz_dir, engine_cache, work):
    """Frame probabilities (N, 3) of M's fold ensemble for one development match and input."""
    from engine import data, learned
    from tools.learned_spike import decoders
    from tools.learned_spike import r3_eval
    n = data.frame_count(game)
    if inp == "raw":
        return decoders.upsample(np.load(os.path.join(p5_dir, game + "_p5.npz"))["p5"], n)
    from analysis import external_sweep as xs
    k = folds.OPEN.index(game)
    nets = learned.load_weights(os.path.join(npz_dir, "fold{}.npz".format(k)))
    if inp == "matched":
        root = xs.data_root(game, "matched", REFS_ROOT, SAMPLE_ROOT, MTAH_ROOT)
        cache = os.path.join(work, "views", "matched", "engine_data")
        os.makedirs(cache, exist_ok=True)
        tac = data.load_tactical(game, data_dir=root, cache_dir=cache)
    else:
        from tools.learned_spike import corrupt
        root = xs.data_root(game, "raw", REFS_ROOT, SAMPLE_ROOT, MTAH_ROOT)
        raw = data.load_tactical(game, data_dir=root, cache_dir=engine_cache,
                                 use_cache=engine_cache is not None)
        tac, _ = corrupt.corrupt(raw, game, ball="matched")
    _, frames = learned.frame_probabilities(nets, r3_eval.guard(tac, game), game)
    return frames


def c23_codes(args, game, inp, n):
    """c23's labels on one input: --c23-dev (raw), the engine runs of W (matched, stress)."""
    import score_game as sg
    from tools.learned_spike import evaluate as ev
    from tools.learned_spike import r3_eval
    if inp == "raw":
        return sg.read_codes(r3_eval.c23_dev_csv(args, game), n)
    ns = argparse.Namespace(work=args.work, run=args.run, refs_root=ev.DEFAULT_REFS_ROOT,
                            sample_root=ev.DEFAULT_SAMPLE_ROOT, mtah_root=ev.DEFAULT_MTAH_ROOT,
                            engine_cache=args.engine_cache, sc_engine_cache=args.engine_cache)
    ctx = r3_eval.make_ctx(ns)
    view = ev.VIEW_MATCHED if inp == "matched" else ev.VIEW_STRESS
    job = ev.make_job(ctx, game, "c22", "c23", view, None, {"M": (1.0, 1.0), "H": 1.0,
                                                            "source": "-"})
    if not os.path.isfile(job["csv"]):
        return None
    return sg.read_codes(job["csv"], n)


def engine_m_codes(args, game, inp, n):
    """(pair, codes) of engine M at the nested pair of costs_r3.json on one input, if run."""
    import score_game as sg
    from tools.learned_spike import evaluate as ev
    from tools.learned_spike import r3_eval
    path = os.path.join(args.work, "costs_r3.json")
    if not os.path.isfile(path):
        return None
    costs = r3_eval.match_costs(json.load(open(path)), game)
    ns = argparse.Namespace(work=args.work, run=args.run, refs_root=ev.DEFAULT_REFS_ROOT,
                            sample_root=ev.DEFAULT_SAMPLE_ROOT, mtah_root=ev.DEFAULT_MTAH_ROOT,
                            engine_cache=args.engine_cache, sc_engine_cache=args.engine_cache)
    ctx = r3_eval.make_ctx(ns)
    view = {"raw": ev.VIEW_RAW, "matched": ev.VIEW_MATCHED, "stress": ev.VIEW_STRESS}[inp]
    weights = os.path.join(args.npz_dir, "fold{}.npz".format(folds.OPEN.index(game)))
    job = ev.make_job(ctx, game, "M", "M", view, weights, costs)
    if not os.path.isfile(job["csv"]):
        return None
    return tuple(costs["M"]), sg.read_codes(job["csv"], n)


def cmd_grid(args):
    from engine import data
    from tools.learned_spike import decoders
    from tools.learned_spike import r3_scope as r
    refs = load_refs(args.work, args.cache)
    tab = {v: {i: {} for i in INPUTS} for v in VIEWS}
    base = {v: {i: {} for i in INPUTS} for v in VIEWS}
    counts = {i: {} for i in INPUTS}
    missing, checks = [], {}
    for game in DEV:
        folds.check(game)
        n = data.frame_count(game)
        blocks = data.graded_blocks(game)
        for inp in INPUTS:
            frames = input_probs(game, inp, args.p5, args.npz_dir, args.engine_cache, args.work)
            paths = decoders.decode_model(frames, blocks, GRID)
            c23 = c23_codes(args, game, inp, n)
            if c23 is None:
                missing.append("{} {} c23".format(game, inp))
            runs = r.pm.valid_runs(refs["provided"][game])
            counts[inp][game] = [len(r.spells(r.pm.fill_abstentions(p, runs), runs))
                                 for p in paths]
            engine_m = engine_m_codes(args, game, inp, n)
            if engine_m is not None:
                pair, codes = engine_m
                checks.setdefault(inp, {})[game] = bool(np.array_equal(
                    paths[GRID.index(pair)], codes))
            for v in VIEWS:
                ref = refs[v][game]
                tab[v][inp][game] = [r.spell_f1(ref, p) for p in paths]
                base[v][inp][game] = r.spell_f1(ref, c23) if c23 is not None else None
            log("grid: {} {}".format(game, inp))
    if checks and not all(all(v.values()) for v in checks.values()):
        raise SystemExit("the decode here differs from engine M: {}".format(checks))
    if missing:
        raise SystemExit("c23 runs missing (run r3_eval run and stress first): {}".format(missing))
    rules = {k: apply_rule(tab, base, k) for k in RULES}
    rec = {"grid": GRID, "rule_b_indices": RULE_B, "views": VIEWS, "inputs": INPUTS,
           "dfl_loose_is_provided": True, "spell_f1": tab, "c23": base, "possessions": counts,
           "reference_possessions": {v: {g: ref_count(refs[v][g]) for g in DEV} for v in VIEWS},
           "rules": rules, "gains": gains_of(tab, base, rules),
           "engine_equals_this_decode": checks,
           "envelope_8e": envelope_8e(args, refs), "p5": os.path.abspath(args.p5),
           "npz_dir": os.path.abspath(args.npz_dir)}
    write_json(args.out or os.path.join(args.work, "views.json"), rec)
    for k in RULES:
        print("rule {}: final pair {} (score {:.4f}); nested {}".format(
            k, GRID[rules[k]["final"]], rules[k]["final_score"],
            {g: GRID[i] for g, i in rules[k]["nested"].items()}))
    return 0


def gains_of(tab, base, rules):
    """{rule: {view: {input: {game: gain of M at the nested pair over c23}}}}."""
    out = {}
    for k, rule in rules.items():
        out[k] = {v: {inp: {g: tab[v][inp][g][rule["nested"][g]] - base[v][inp][g] for g in DEV}
                      for inp in INPUTS} for v in VIEWS}
        out[k]["final_pair_gain"] = {v: {inp: {g: tab[v][inp][g][rule["final"]] - base[v][inp][g]
                                               for g in DEV} for inp in INPUTS} for v in VIEWS}
    return out


def envelope_8e(args, refs):
    """The 8e statistics of the 28 references as provided, and with the Metrica variants."""
    from tools.learned_spike import bar_report as br
    from tools.learned_spike import evaluate as ev
    import possession_metrics as pm
    ns = argparse.Namespace(work=args.work, run=args.run, refs_root=ev.DEFAULT_REFS_ROOT,
                            sample_root=ev.DEFAULT_SAMPLE_ROOT, mtah_root=ev.DEFAULT_MTAH_ROOT,
                            ref_cache=args.ref_cache, engine_cache=None, sc_engine_cache=None)
    from tools.learned_spike import r3_eval
    ctx = r3_eval.make_ctx(ns)
    stats = {}
    for game in folds.ALL:
        codes = ev.references(ctx, game)["provided"]
        stats[game] = br.system_stats(codes, br.graded_runs(game), None)
    loose = {g: br.system_stats(refs["loose"][g], br.graded_runs(g), None) for g in METRICA}
    keys = (("possession_s", "p50"), ("possession_s", "p90"), ("d_segment_s", "p50"),
            ("d_segment_s", "p90"))
    out = {}
    for a, q in keys:
        prov = [s[a][q] for s in stats.values()]
        both = prov + [s[a][q] for s in loose.values()]
        out["{}.{}".format(a, q)] = {"provided": [min(prov), max(prov)],
                                     "union_with_loose": [min(both), max(both)],
                                     "metrica_loose": {g: s[a][q] for g, s in loose.items()}}
    del pm
    return out


# -- markdown -------------------------------------------------------------------------------------

def markdown(rec):
    """The tables of the result file (recorded next to the frozen criteria)."""
    tab, base, rules = rec["spell_f1"], rec["c23"], rec["rules"]
    grid = [tuple(p) for p in rec["grid"]]
    w = ["## Reference views, inputs and the whole cost grid (recorded; decides nothing)\n",
         "Development matches, M = the run's out-of-fold fold ensembles. Views: as provided; "
         "loose-touch variant (LEARNINGS F2, Metrica only; DFL's loose view is its reference as "
         "provided); 2 s view. Inputs: raw, matched, stress. Cells: Metrica mean / DFL mean "
         "spellF1. See `tools/learned_spike/r3_views.py` for the rules.\n"]
    w.append("Reference possessions: {}\n".format(json.dumps(rec["reference_possessions"],
                                                             sort_keys=True)))
    cols = [(v, i) for v in VIEWS for i in INPUTS]
    w.append("| system | " + " | ".join("{} {}".format(v, i) for v, i in cols) + " |")
    w.append("|---|" + "---|" * len(cols))

    def cell(get):
        met = np.mean([get(g) for g in METRICA])
        dfl = np.mean([get(g) for g in DEV if g not in METRICA])
        return "{:.3f} / {:.3f}".format(met, dfl)
    w.append("| c23 | " + " | ".join(cell(lambda g, v=v, i=i: base[v][i][g])
                                     for v, i in cols) + " |")
    for k, (a, b) in enumerate(grid):
        mark = "" if k in rec["rule_b_indices"] else " (outside rule B)"
        w.append("| M {:g}/{:g}{} | ".format(a, b, mark) + " | ".join(
            cell(lambda g, v=v, i=i, k=k: tab[v][i][g][k]) for v, i in cols) + " |")
    w.append("\n### Candidate cost rules\n")
    for name, text in (("a", "rule B as frozen (raw, as provided, 56 pairs)"),
                       ("b", "best worst case over {Metrica, DFL} x {as provided, loose} x "
                             "{raw, matched, stress} of the provider mean gain over c23 "
                             "(80 pairs; R3_DEVIATION_COSTS.md)"),
                       ("bp", "recorded: best worst case over {as provided, loose} x {raw, "
                              "matched, stress} of the provider-balanced gain (DFL pooled in)"),
                       ("c", "equal weight of the two views, mean over the three inputs "
                             "(80 pairs)")):
        r = rules[name]
        w.append("**{}. {}**: final pair {}, score {:+.4f}; nested pairs {}.\n".format(
            name, text, grid[r["final"]], r["final_score"],
            ", ".join("{} {}".format(g, grid[i]) for g, i in sorted(r["nested"].items()))))
        w.append("| match | " + " | ".join("{} {}".format(v, i) for v, i in cols) + " |")
        w.append("|---|" + "---|" * len(cols))
        gains = rec["gains"][name]
        for g in DEV:
            w.append("| {} | ".format(g) + " | ".join("{:+.4f}".format(gains[v][i][g])
                                                      for v, i in cols) + " |")
        w.append("")
    w.append("### Section 8e bounds as the union over both views (recorded; 8e unchanged)\n")
    w.append("| statistic | as provided (28) | union with the 3 Metrica loose-touch references |")
    w.append("|---|---|---|")
    for key, r in sorted(rec["envelope_8e"].items()):
        w.append("| {} | {} to {} s | {} to {} s |".format(key, r["provided"][0], r["provided"][1],
                                                          r["union_with_loose"][0],
                                                          r["union_with_loose"][1]))
    return "\n".join(w) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("refs")
    p.add_argument("--work", required=True)
    p.add_argument("--cache", required=True, help="build.py cache (labels/<game>.npz)")
    p = sub.add_parser("grid")
    p.add_argument("--work", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--p5", required=True, help="out-of-fold <game>_p5.npz of the raw view")
    p.add_argument("--npz-dir", required=True, help="fold{k}.npz of evaluate.py export")
    p.add_argument("--c23-dev", required=True)
    p.add_argument("--engine-cache")
    p.add_argument("--ref-cache")
    p.add_argument("--out")
    p = sub.add_parser("rules", help="recompute the rules of an existing views.json in place")
    p.add_argument("--views", required=True)
    p = sub.add_parser("markdown")
    p.add_argument("--views", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "rules":
        rec = json.load(open(args.views))
        rec["rules"] = {k: apply_rule(rec["spell_f1"], rec["c23"], k) for k in RULES}
        rec["gains"] = gains_of(rec["spell_f1"], rec["c23"], rec["rules"])
        write_json(args.views, rec)
        for k in RULES:
            print("rule {}: final pair {}".format(k, tuple(GRID[rec["rules"][k]["final"]])))
        return 0
    if args.cmd == "markdown":
        print(markdown(json.load(open(args.views))))
        return 0
    return {"refs": cmd_refs, "grid": cmd_grid}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
