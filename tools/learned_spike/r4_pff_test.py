"""Round 4 test T-P: the final ensemble against c23 on the 43 open PFF matches. Not part of the submission.

    # 1. engine venv: c23 on the 43 open PFF matches (the rule engine only; the c23 checkout)
    python tools/learned_spike/r3_c23_dev.py --engine-repo C23_CHECKOUT --ext W/c23_ext \
        --out W/c23 --games-set pff --jobs 4
    # 2. engine venv: the final ensemble (engine, LEARNED_MODE model) at one cost pair
    python -m tools.learned_spike.r4_pff_test run --games-set pff --weights NPZ --pair 16 8 \
        --work W --jobs 4
    # 3. engine venv: the scores and the rule
    python -m tools.learned_spike.r4_pff_test score --games-set pff --weights NPZ --pair 16 8 \
        --c23 W/c23 --work W
    # rehearsal of the same mode on the 20 SkillCorner matches: --games-set sc (references as
    # provided, the SkillCorner phases; c23 outputs of the SkillCorner matches)

The owner's decision of 13 November 2026 ("PFF can be used as test"): the 43 open PFF matches
(folds.PFF_TRAIN) may be scored, never trained or validated on. The 4 pilot and 15 sealed PFF
matches stay refused (check_games(): folds.assert_not_sealed, then membership of PFF_TRAIN).

Measures (PFF_TRANSFER_TEST.md Sections 4 and 5 with c23 in place of c22), own tracking (raw),
no degraded or stress view:

- primary spellF1 against the 2 s controlled reference (control_labels(ref, 50)), secondary
  against the reference as provided; per-match delta (network minus c23), mean, bootstrap over
  matches (1,000 draws, RandomState(0), 5-95 %), win share: evaluate.v1_rules() unchanged;
- rule a: 2 s view, 5 % bound of the mean delta above 0; rule b: as provided, mean above 0 and
  5 % bound above -0.010; rule c: delta > 0 in the 2 s view on at least 60 % of the matches;
  a match without both outputs fails the set (evaluate.v1_rules n_expected);
- recorded: deltas at reference shifts -10 and +10 frames; per match and system the possession
  count against the reference's, merge share, split share and time-weighted merge share
  (r4_screen.merge_share), D share; pooled D shares.

The network runs inside the engine (engine.run with LEARNED_MODE model, the weights file, the
pair, LEARNED_STRUCTURE on unless --structure off), as the submission would run it. Engine data
caches and outputs go under --work (outside the repository). The tool never reads evaluation/gt
and never runs the competition match or a sealed DFL match. Python 3.9.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from tools.learned_spike import folds  # noqa: E402

CONTROL_FRAMES = 50
SHIFTS = (-10, 0, 10)
REFS_ROOT = os.path.join("~", "mtah2027", "refs")
SAMPLE_ROOT = os.path.join("~", "mtah2027", "sample-data", "data")
GAME_SETS = ("pff", "sc")


def log(text):
    print(text, file=sys.stderr, flush=True)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(obj, handle, indent=1, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def md5_file(path):
    d = hashlib.md5()
    with open(path, "rb") as h:
        for block in iter(lambda: h.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


def check_games(games):
    """The open PFF matches or the SkillCorner matches only; pilot and sealed PFF refused."""
    games = folds.assert_not_sealed(games)
    for g in games:
        if g.startswith("PFF-"):
            if g not in folds.PFF_TRAIN:
                raise SystemExit("{}: not one of the 43 open PFF matches".format(g))
        elif folds.provider(folds.check(g)) != "skillcorner":
            raise SystemExit("{}: T-P runs on the open PFF matches (rehearsal: SkillCorner)"
                             .format(g))
    return games


def games_of(game_set):
    if game_set == "pff":
        return check_games(list(folds.PFF_TRAIN))
    if game_set == "sc":
        return check_games(list(folds.SC))
    raise ValueError(game_set)


# Round 6: the per-frame member (Config.LEARNED_PF), set by main() from --pf-weights and
# --pf-weight: None (the network alone) or {"weights", "md5", "weight"}.
PF = None


def tag_of(weights_md5, pair, structure, bonus=0.0):
    return "m_{}_{:g}_{:g}{}{}{}".format(weights_md5[:12], pair[0], pair[1],
                                         "" if structure else "_nostruct",
                                         "_prior{:g}".format(bonus) if bonus else "",
                                         "_pf{}w{:g}".format(PF["md5"][:8], PF["weight"])
                                         if PF else "")


def overrides(weights, pair, structure, bonus=0.0):
    """Engine config of the network run. A non-zero bonus sets Config.LEARNED_PRIOR_BONUS (the
    rule-engine prior of engine.learned); 0 leaves the field out (its default is 0.0)."""
    over = {"LEARNED_MODE": "model", "LEARNED_WEIGHTS": os.path.abspath(weights),
            "LEARNED_COST_HA": float(pair[0]), "LEARNED_COST_AD": float(pair[1])}
    if structure:
        over["LEARNED_STRUCTURE"] = True
    if bonus:
        if bonus < 0:
            raise SystemExit("the prior bonus must be >= 0")
        over["LEARNED_PRIOR_BONUS"] = float(bonus)
    if PF:
        over.update(LEARNED_PF="on", LEARNED_PF_WEIGHTS=PF["weights"],
                    LEARNED_PF_WEIGHT=float(PF["weight"]))
    return over


def m_job(work, game, weights, weights_md5, pair, structure, bonus=0.0):
    from analysis import external_sweep as xs
    root = os.path.abspath(os.path.expanduser(REFS_ROOT))
    over = overrides(weights, pair, structure, bonus)
    resolved, _ = xs.resolve_config(over)
    key_obj = {"game": game, "view": "raw", "config": resolved, "data_dir": root,
               "engine": xs.engine_source_hash(), "weights_md5": weights_md5}
    if PF:
        key_obj["pf_md5"] = PF["md5"]
    key = hashlib.sha1(json.dumps(key_obj, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    where = os.path.join(work, tag_of(weights_md5, pair, structure, bonus))
    return {"game": game, "overrides": over, "data_dir": root, "key": key,
            "csv": os.path.join(where, "runs", "{}_{}.csv".format(game, key)),
            "cache": os.path.join(where, "engine_data")}


def m_run(job):
    """external_sweep --engine-child on one match, unless the CSV exists."""
    from analysis import external_sweep as xs
    if os.path.isfile(job["csv"]):
        return job["game"], job["csv"]
    base = job["csv"][:-4]
    os.makedirs(os.path.dirname(base), exist_ok=True)
    os.makedirs(job["cache"], exist_ok=True)
    write_json(base + ".config.json", job["overrides"])
    cmd = [sys.executable, os.path.abspath(xs.__file__), "--engine-child", job["cache"],
           "--game", job["game"], "--config", base + ".config.json", "--out", base + ".tmp.csv"]
    with open(base + ".log", "w") as out:
        code = subprocess.call(cmd, cwd=ROOT, env=dict(os.environ, MTAH_DATA_DIR=job["data_dir"]),
                               stdout=out, stderr=subprocess.STDOUT)
    if code != 0:
        raise RuntimeError("engine.run failed on {}, see {}.log".format(job["game"], base))
    write_json(base + ".meta.json", dict(job, output_md5=md5_file(base + ".tmp.csv")))
    os.replace(base + ".tmp.csv", job["csv"])
    return job["game"], job["csv"]


def cmd_run(args):
    games = games_of(args.games_set)
    wmd5 = md5_file(args.weights)
    jobs = [m_job(os.path.abspath(args.work), g, args.weights, wmd5, args.pair,
                  args.structure == "on", args.prior_bonus) for g in games]
    with concurrent.futures.ProcessPoolExecutor(args.jobs) as pool:
        for game, csv in pool.map(m_run, jobs):
            log("{} {}".format(game, csv))
    return 0


def references(game):
    """{"provided", "ctl2", "md5"}: PFF via pff.reference (md5 of R2 Section 8 checked),
    SkillCorner via external_sweep.prepare_reference (rehearsal)."""
    import possession_metrics as pm
    from analysis import control_reference as cref
    check_games([game])
    refs = os.path.abspath(os.path.expanduser(REFS_ROOT))
    if game.startswith("PFF-"):
        from tools.learned_spike import pff
        path, digest = pff.reference(game, refs)
    else:
        from analysis import external_sweep as xs
        path, digest = xs.prepare_reference(game, refs, os.path.abspath(
            os.path.expanduser(SAMPLE_ROOT)), None)
    provided = pm.label_codes(pm.read_label_csv(path)).astype(np.int8)
    labels = np.asarray(pm.LABELS)[provided]
    ctl2 = pm.label_codes(cref.control_labels(labels, CONTROL_FRAMES)).astype(np.int8)
    return {"provided": provided, "ctl2": ctl2, "md5": digest}


def cmd_score(args):
    import score_game as sg
    from tools.learned_spike import evaluate as ev
    from tools.learned_spike import r4_screen as rs
    games = games_of(args.games_set)
    wmd5 = md5_file(args.weights)
    structure = args.structure == "on"
    work = os.path.abspath(args.work)
    scores = {s: {str(sh): {"ctl2": {}, "provided": {}} for sh in SHIFTS} for s in ("M", "c23")}
    per_match, missing = {}, []
    for game in games:
        job = m_job(work, game, args.weights, wmd5, args.pair, structure, args.prior_bonus)
        c23_path = os.path.join(args.c23, game + ".csv")
        if not (os.path.isfile(job["csv"]) and os.path.isfile(c23_path)):
            missing.append(game)
            continue
        ref = references(game)
        n = len(ref["provided"])
        out = {"M": np.asarray(sg.read_codes(job["csv"], n), np.int8),
               "c23": np.asarray(sg.read_codes(c23_path, n), np.int8)}
        row = {"reference_md5": ref["md5"],
               "csv_md5": {"M": md5_file(job["csv"]), "c23": md5_file(c23_path)}}
        for kind in ("ctl2", "provided"):
            r0 = ref[kind]
            row["n_ref_" + kind] = len(rs.spells(r0, __import__("possession_metrics")
                                                 .valid_runs(r0)))
            for s in ("M", "c23"):
                for sh in SHIFTS:
                    r = ev.shift_codes(r0, sh) if sh else r0
                    scores[s][str(sh)][kind][game] = float(
                        sg.score_codes(r, out[s])[0]["spell_f1_mean"])
                row.setdefault(s, {})[kind] = rs.measures(r0, out[s])
        per_match[game] = row
        log("{} 2 s {:+.4f} provided {:+.4f}".format(
            game, scores["M"]["0"]["ctl2"][game] - scores["c23"]["0"]["ctl2"][game],
            scores["M"]["0"]["provided"][game] - scores["c23"]["0"]["provided"][game]))
    rules = ev.v1_rules(scores["M"]["0"], scores["c23"]["0"], n_expected=len(games))
    shifted = {}
    done = sorted(per_match)
    for sh in SHIFTS:
        shifted[str(sh)] = {k: float(np.mean([scores["M"][str(sh)][k][g]
                                              - scores["c23"][str(sh)][k][g] for g in done]))
                            if done else None for k in ("ctl2", "provided")}

    def med(f):
        return float(np.median([f(per_match[g]) for g in done])) if done else None
    summary = {s: {"count_ratio_provided_median": med(lambda r: r[s]["provided"]["n"]
                                                       / r["n_ref_provided"]),
                   "merge_provided_mean": float(np.mean([per_match[g][s]["provided"]["merge"]
                                                         for g in done])) if done else None,
                   "split_provided_mean": float(np.mean([per_match[g][s]["provided"]["split"]
                                                         for g in done])) if done else None,
                   "merge_time_provided_mean": float(np.mean(
                       [per_match[g][s]["provided"]["merge_time"] for g in done]))
                   if done else None,
                   "d_share_mean": float(np.mean([per_match[g][s]["provided"]["d_share"]
                                                  for g in done])) if done else None}
               for s in ("M", "c23")}
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "games_set": args.games_set,
           "rehearsal": args.games_set != "pff", "weights": os.path.abspath(args.weights),
           "pf": PF,
           "weights_md5": wmd5, "pair": list(args.pair), "structure": structure,
           "prior_bonus": args.prior_bonus,
           "missing": missing, "rules": rules, "mean_delta_by_shift": shifted,
           "summary": summary, "per_match": per_match, "scores": scores,
           "conventions": {"bootstrap": "over matches, sorted, RandomState(0), 1000 draws, "
                                        "5-95 % (evaluate.match_bootstrap)",
                           "rules": "a: 2 s low > 0; b: provided mean > 0 and low > -0.010; "
                                    "c: 5 x wins >= 3 x n", "base": "c23"}}
    os.makedirs(args.out, exist_ok=True)
    name = "tp_{}".format(tag_of(wmd5, args.pair, structure, args.prior_bonus))
    write_json(os.path.join(args.out, name + ".json"), res)
    with open(os.path.join(args.out, name + ".md"), "w") as h:
        h.write(result_md(res))
    log("rules: {}".format({k: rules.get(k) for k in ("complete", "a", "b", "c", "pass")}))
    return 0


def result_md(res):
    r = res["rules"]
    lines = ["# T-P: final ensemble against c23, {} matches ({})".format(
        len(res["per_match"]), res["games_set"]), ""]
    if res["rehearsal"]:
        lines += ["**REHEARSAL on SkillCorner (references as provided): not a PFF result.**",
                  ""]
    lines += ["- weights {} (md5 {}), pair ({:g}, {:g}), kickoff rule {}, prior bonus {:g}".format(
        res["weights"], res["weights_md5"], res["pair"][0], res["pair"][1],
        "on" if res["structure"] else "off", res.get("prior_bonus", 0.0)),
        "- missing outputs: {}".format(
        res["missing"] or "none"), ""]
    if r.get("complete"):
        lines += ["| view | mean delta | 5 % | 95 % |", "|---|---|---|---|",
                  "| 2 s | {:+.4f} | {:+.4f} | {:+.4f} |".format(
                      r["ctl2"]["delta"], r["ctl2"]["low"], r["ctl2"]["high"]),
                  "| as provided | {:+.4f} | {:+.4f} | {:+.4f} |".format(
                      r["provided"]["delta"], r["provided"]["low"], r["provided"]["high"]), "",
                  "- wins (2 s): {}/{} ({:.0%})".format(r["wins"], r["n"], r["win_share"]),
                  "- rule a {} / b {} / c {}: **{}**".format(
                      r["a"], r["b"], r["c"], "pass" if r["pass"] else "fail"), ""]
    else:
        lines += ["- incomplete: the set fails (a match without both outputs)", ""]
    s = res["summary"]
    lines += ["| system | count / ref (median) | merge | split | time merge | D share |",
              "|---|---|---|---|---|---|"]
    for k in ("M", "c23"):
        if s[k]["merge_provided_mean"] is None:
            continue
        lines.append("| {} | {:.2f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            k, s[k]["count_ratio_provided_median"], s[k]["merge_provided_mean"],
            s[k]["split_provided_mean"], s[k]["merge_time_provided_mean"], s[k]["d_share_mean"]))
    return "\n".join(lines) + "\n"


# -- G9: the sealed PFF matches, once, behind the owner's gate -------------------------------

SEALED_GATE = "owner-approved-13nov"       # UNSEAL_PFF value of the real run
REHEARSAL_GATE = "rehearsal-sc"            # UNSEAL_PFF value of the SkillCorner rehearsal
GATE_ENV = "UNSEAL_PFF"


def sealed_gate(rehearse, prereg, environ=None):
    """Checked before anything else is read: the environment value and the pre-registration."""
    environ = os.environ if environ is None else environ
    want = REHEARSAL_GATE if rehearse else SEALED_GATE
    if environ.get(GATE_ENV) != want:
        raise SystemExit("sealed refuses: {}={} is required{}".format(
            GATE_ENV, "<the owner's value>" if not rehearse else want,
            " (rehearsal)" if rehearse else ""))
    if not prereg or not os.path.isfile(prereg):
        raise SystemExit("sealed refuses: the pre-registration file --prereg is missing")
    return want


def sealed_games(rehearse, rehearse_games=None):
    """The 15 sealed PFF matches, or (rehearsal) SkillCorner matches only."""
    if rehearse:
        games = list(rehearse_games or folds.SC[:len(folds.PFF_SEALED)])
        for g in games:
            if folds.provider(folds.check(g)) != "skillcorner":
                raise SystemExit("{}: the rehearsal runs on SkillCorner matches only".format(g))
        return games
    return [folds.check_sealed_v1(g) for g in folds.PFF_SEALED]


def sealed_paths(out, rehearse):
    tag = "REHEARSAL_SC" if rehearse else "PFF"
    return (os.path.join(out, "SEALED_{}_STARTED".format(tag)),
            os.path.join(out, "SEALED_{}_RESULT.json".format(tag)),
            os.path.join(out, "SEALED_{}_RESULT.md".format(tag)))


def judge(games, scores_m, scores_c):
    """Rules a, b, c on the set (evaluate.v1_rules, n_expected = the set size): a match without
    both outputs makes the set incomplete, and an incomplete set fails."""
    from tools.learned_spike import evaluate as ev
    return ev.v1_rules(scores_m, scores_c, n_expected=len(games))


def run_sealed_set(games, run_fn, ref_fn, log_fn=log):
    """Run and score every match. run_fn(game, system) -> codes or raises; ref_fn(game) ->
    {"provided", "ctl2", "md5"}. Return (scores_m, scores_c, per_match, error)."""
    import score_game as sg
    scores = {"M": {"ctl2": {}, "provided": {}}, "c23": {"ctl2": {}, "provided": {}}}
    per_match, error = {}, None
    for game in games:
        try:
            out = {sys_: run_fn(game, sys_) for sys_ in ("c23", "M")}
        except Exception as exc:     # an unfinished run fails the set
            error = "{}: {}".format(game, exc)
            log_fn(error)
            break
        ref = ref_fn(game)
        row = {"reference_md5": ref["md5"]}
        for kind in ("ctl2", "provided"):
            for sys_ in ("M", "c23"):
                scores[sys_][kind][game] = float(
                    sg.score_codes(ref[kind], out[sys_])[0]["spell_f1_mean"])
        for sys_ in ("M", "c23"):
            try:
                from tools.learned_spike import r4_screen as rs
                row[sys_] = rs.measures(ref["provided"], out[sys_])
            except Exception:        # recorded measures only; never decides
                row[sys_] = None
        per_match[game] = row
    return scores["M"], scores["c23"], per_match, error


def cmd_sealed(args):
    rehearse = bool(args.rehearse)
    sealed_gate(rehearse, args.prereg)                      # before anything else
    out = os.path.abspath(args.out)
    if os.path.commonpath([out, ROOT]) == ROOT:
        raise SystemExit("--out must be outside the repository")
    started, res_json, res_md = sealed_paths(out, rehearse)
    if os.path.exists(started) or os.path.exists(res_json):
        raise SystemExit("{} already holds a start of this judge: the sealed matches are opened "
                         "once; an unfinished run counts as failed".format(out))
    if args.prior_bonus < 0:
        raise SystemExit("the prior bonus must be >= 0")
    if args.pair[0] > 2 * args.pair[1]:
        raise SystemExit("team-change cost above twice the alive-dead cost")
    wmd5 = md5_file(args.weights)
    if wmd5 != args.weights_md5:
        raise SystemExit("weights md5 {} differs from --weights-md5 {}".format(
            wmd5, args.weights_md5))
    from analysis import external_sweep as xs
    os.makedirs(out, exist_ok=True)
    with open(args.prereg, "rb") as h:
        prereg = h.read()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    rec = {"status": "started", "rehearsal": rehearse,
           "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "prereg": {"path": os.path.abspath(args.prereg), "md5": hashlib.md5(prereg).hexdigest(),
                      "text": prereg.decode("utf-8", "replace")},
           "weights": os.path.abspath(args.weights), "weights_md5": wmd5,
           "pair": list(args.pair), "prior_bonus": args.prior_bonus,
           "structure": args.structure == "on", "engine_source_hash": xs.engine_source_hash(),
           "commit": commit,
           "rule": "PFF_TRANSFER_TEST.md Section 5 with c23 (this checkout, LEARNED_MODE off): "
                   "a 2 s low > 0; b provided mean > 0 and low > -0.010; c 5 x wins >= 3 x n"}
    with open(started, "w") as h:
        h.write(rec["started"] + "\n")
    write_json(res_json, rec)                               # pre-registration first
    games = sealed_games(rehearse, args.rehearse_games)
    work = os.path.join(out, "runs")
    root = os.path.abspath(os.path.expanduser(REFS_ROOT))

    def run_fn(game, system):
        if system == "M":
            job = m_job(work, game, args.weights, wmd5, args.pair, args.structure == "on",
                        args.prior_bonus)
        else:
            job = c23_job(work, game, root)
        _, csv = m_run(job)
        import score_game as sg
        return np.asarray(sg.read_codes(csv, __import__("engine.data", fromlist=["x"])
                                        .frame_count(game)), np.int8)

    def ref_fn(game):
        if rehearse:
            return references(game)
        return sealed_reference(game, root)

    sm, sc, per_match, error = run_sealed_set(games, run_fn, ref_fn)
    rules = judge(games, sm, sc) if error is None else {"complete": False, "pass": False}
    rec.update(status="finished" if error is None else "unfinished (fails)", error=error,
               finished=time.strftime("%Y-%m-%d %H:%M:%S"), matches=games, rules=rules,
               scores={"M": sm, "c23": sc}, per_match=per_match,
               verdict="pass" if rules.get("pass") else "fail")
    write_json(res_json, rec)
    with open(res_md, "w") as h:
        h.write("# {} sealed judge: {}\n\n{}".format(
            "REHEARSAL (SkillCorner)" if rehearse else "PFF", rec["verdict"],
            result_md({"per_match": per_match, "games_set": "sealed", "rehearsal": rehearse,
                       "weights": rec["weights"], "weights_md5": wmd5, "pair": rec["pair"],
                       "structure": rec["structure"], "missing": [error] if error else [],
                       "rules": rules, "summary": {k: {"merge_provided_mean": None}
                                                   for k in ("M", "c23")}})))
    log("sealed judge: {}".format(rec["verdict"]))
    return 0


def c23_job(work, game, root):
    """c23 = this checkout's engine with the learned stage off (no override)."""
    from analysis import external_sweep as xs
    resolved, _ = xs.resolve_config({})
    key_obj = {"game": game, "view": "raw", "config": resolved, "data_dir": root,
               "engine": xs.engine_source_hash()}
    key = hashlib.sha1(json.dumps(key_obj, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    where = os.path.join(work, "c23")
    return {"game": game, "overrides": {}, "data_dir": root, "key": key,
            "csv": os.path.join(where, "runs", "{}_{}.csv".format(game, key)),
            "cache": os.path.join(where, "engine_data")}


def sealed_reference(game, root):
    """Reference of a sealed PFF match (pff.reference with sealed=True: md5 of R2 Section 8)."""
    import possession_metrics as pm
    from analysis import control_reference as cref
    from tools.learned_spike import pff
    path, digest = pff.reference(folds.check_sealed_v1(game), root, sealed=True)
    provided = pm.label_codes(pm.read_label_csv(path)).astype(np.int8)
    labels = np.asarray(pm.LABELS)[provided]
    return {"provided": provided, "md5": digest,
            "ctl2": pm.label_codes(cref.control_labels(labels, CONTROL_FRAMES)).astype(np.int8)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "score"):
        s = sub.add_parser(name)
        s.add_argument("--games-set", choices=GAME_SETS, required=True)
        s.add_argument("--weights", required=True, help="engine weights npz (export.py)")
        s.add_argument("--pair", type=float, nargs=2, required=True)
        s.add_argument("--structure", choices=("on", "off"), default="on")
        s.add_argument("--work", required=True, help="engine runs (outside the repository)")
    sub.choices["run"].add_argument("--jobs", type=int, default=4)
    sub.choices["score"].add_argument("--c23", required=True, help="folder of c23 <game>.csv")
    sub.choices["score"].add_argument("--out", required=True)
    z = sub.add_parser("sealed", help="G9: the 15 sealed PFF matches once (owner's gate)")
    z.add_argument("--prereg", required=True)
    z.add_argument("--weights", required=True)
    z.add_argument("--weights-md5", required=True)
    z.add_argument("--pair", type=float, nargs=2, required=True)
    z.add_argument("--prior-bonus", type=float, default=0.0)
    z.add_argument("--structure", choices=("on", "off"), default="on")
    z.add_argument("--out", required=True)
    z.add_argument("--rehearse", action="store_true", help="SkillCorner matches, rehearsal gate")
    z.add_argument("--rehearse-games", nargs="+", default=None)
    for name in ("run", "score"):
        sub.choices[name].add_argument("--prior-bonus", type=float, default=0.0)
    for name in ("run", "score", "sealed"):
        sub.choices[name].add_argument("--pf-weights", default=None,
                                       help="round 6: the per-frame member's weights file "
                                            "(Config.LEARNED_PF on in the engine runs)")
        sub.choices[name].add_argument("--pf-weight", type=float, default=0.5)
    args = p.parse_args(argv)
    global PF
    PF = ({"weights": os.path.abspath(args.pf_weights), "md5": md5_file(args.pf_weights),
           "weight": float(args.pf_weight)} if args.pf_weights else None)
    if args.cmd == "sealed":
        return cmd_sealed(args)
    if args.prior_bonus < 0:
        raise SystemExit("the prior bonus must be >= 0")
    if os.path.commonpath([os.path.abspath(args.work), ROOT]) == ROOT:
        raise SystemExit("--work must be outside the repository")
    if args.pair[0] > 2 * args.pair[1]:
        raise SystemExit("team-change cost above twice the alive-dead cost")
    return {"run": cmd_run, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
