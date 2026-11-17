"""Round 3 sealed DFL judge: c23 against M on DFL-J03WQQ and DFL-J03WR9, once. Not part of the
submission.

    UNSEAL_DFL=owner-approved-12nov tools/learned_spike/r3_sealed.sh run --out-dir OUT \\
        --variant R3B --weights OUT_SCORE/learned_v3.npz --weights-md5 MD5 --pair 64,16 \\
        --prereg docs/plan/research/<committed pre-registration>.md \\
        --expect "J03WQQ raw as provided: ..." [--expect ...] [--result-json RESULT.json]
    tools/learned_spike/r3_sealed.sh run ... --rehearse-as DFL-J03WMX[,DFL-J03WN1]

What it does (LEARNED_STAGE_PROTOCOL_R3.md Sections 4 and 9, adoption runbook Section 2):

0. Refuses to start unless the environment holds UNSEAL_DFL=owner-approved-12nov AND the file
   given by --prereg exists (both checked before anything else is read), and, for the real
   run, unless OUT holds no earlier start of this judge (the sealed matches are opened once; a
   run that does not finish counts as failed; --resume records an owner-approved resume).
1. Writes OUT/SEALED_RESULT_<variant>.md and .json with the pre-registration first: the text
   and md5 of --prereg, every --expect line, the weights md5, the cost pair, the engine source
   hash and commit, the time. Nothing has been built or run on a sealed match at that point.
2. Builds the inputs of each match, as the development scoring does (r3_eval.py, evaluate.py):
   raw (the DFL bundle under --refs-root, tools/dfl_bundle.py, built 2 Nov 2026), matched (the
   sigma 3 view of ATD_VIEW.md Section 9, tools/atd_view.py --sigma 3 --players off, written
   under OUT/view-s3; --allow-sealed only for a sealed name) and stress (criterion 4's view,
   corrupt.py `--ball matched` with fragmentation, swaps and ghosts, seed from the match name).
   The engine reads the tracking files only; the engine.data cache goes under OUT/cache.
3. Runs c23 (no override: LEARNED_MODE off) and M (LEARNED_MODE model, --weights, the --pair
   costs, LEARNED_STRUCTURE on) on the 2 matches x 3 inputs: 12 engine runs, one at a time.
   Each output is checked for length.
4. Only then opens the references: the reference as provided (<refs-root>/dfl/<ID>/
   DFL-<ID>_reference.csv, the dfl_bundle reference) and its 2 s view
   (control_reference.control_labels(..., 50)). Scores spellF1 and macroF1 of both systems per
   match, input and reference, with the possession counts and D share of each output and of the
   references.
5. The judge (R3 Section 4): the mean over the two matches of delta(M - c23) of spellF1, raw
   input, reference as provided, >= 0. Every other cell is recorded. Writes the result files.

Rehearsal: --rehearse-as G1[,G2] maps DFL-J03WQQ to G1 and DFL-J03WR9 to G2 (G1 if G2 is not
given), two development DFL matches. Every step runs on them, the c23 raw output must equal its
R3 Section 8 md5, the rebuilt matched view must equal the development view under --mtah-root,
and every output says REHEARSAL. A rehearsal never touches a sealed match.

The tool never reads evaluation/gt, never runs the competition match, and never reads a PFF
file. Every other tool keeps refusing the sealed matches (folds.check, corrupt.FORBIDDEN,
bar_report.FORBIDDEN_EXT): only this tool's stress child lifts corrupt.FORBIDDEN, for the one
sealed match it runs, in its own process, after the same environment check. Python 3.9.
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np  # noqa: E402

from tools.learned_spike import folds  # noqa: E402

UNSEAL_VAR = "UNSEAL_DFL"
UNSEAL_VALUE = "owner-approved-12nov"
REHEARSAL_VALUE = "rehearsal"          # a rehearsal runs with UNSEAL_DFL=rehearsal, never the go
JUDGE_RULE = ("mean over J03WQQ and J03WR9 of delta(M - c23), spellF1, raw input, reference as "
              "provided, >= 0 (R3 Sections 4 and 9); a run that does not finish counts as failed")
SEALED = tuple(folds.SEALED)                    # ("DFL-J03WQQ", "DFL-J03WR9")
INPUTS = ("raw", "matched", "stress")
SYSTEMS = ("c23", "M")
REFS = ("provided", "ctl2")
CONTROL_FRAMES = 50
SIGMA = 3.0                                     # ATD_VIEW.md Section 9.1: the DFL matched view
VARIANTS = ("R3B", "R3C")
DEFAULT_REFS_ROOT = os.path.join("~", "mtah2027", "refs")
DEFAULT_SAMPLE_ROOT = os.path.join("~", "mtah2027", "sample-data", "data")
DEFAULT_MTAH_ROOT = os.path.join("~", "mtah2027")


def log(text):
    print(text, file=sys.stderr, flush=True)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(obj, handle, indent=1, sort_keys=True, default=float)
        handle.write("\n")
    os.replace(tmp, path)


def md5_file(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# -- pure rules (tested) --------------------------------------------------------------------------

def gate(environ, prereg, rehearsal=False):
    """Raise SystemExit unless UNSEAL_DFL=owner-approved-12nov is set and `prereg` is a file.
    A rehearsal needs UNSEAL_DFL=rehearsal instead, so that the owner's value is never set while
    rehearsing (and a rehearsal command without --rehearse-as is refused)."""
    want = REHEARSAL_VALUE if rehearsal else UNSEAL_VALUE
    if environ.get(UNSEAL_VAR) != want:
        raise SystemExit("refused: set {}={} in the environment ({})".format(
            UNSEAL_VAR, want, "rehearsal" if rehearsal else "the owner's go"))
    if not prereg or not os.path.isfile(prereg):
        raise SystemExit("refused: --prereg {!r} is not a file (commit the pre-registration "
                         "first)".format(prereg))


def mapping(rehearse_as=None):
    """{sealed name: data match}: the identity, or the rehearsal's development DFL matches."""
    if not rehearse_as:
        return {g: g for g in SEALED}
    names = [x.strip() for x in rehearse_as.split(",") if x.strip()]
    if not 1 <= len(names) <= 2:
        raise SystemExit("--rehearse-as takes one or two development matches")
    for g in names:
        if g not in folds.OPEN or not g.startswith("DFL-"):
            raise SystemExit("--rehearse-as takes development DFL matches {}, not {}".format(
                [x for x in folds.OPEN if x.startswith("DFL-")], g))
        folds.check(g)
    return {SEALED[0]: names[0], SEALED[1]: names[-1]}


def judge(deltas, complete):
    """The sealed judge of R3 Section 4: {"pass", "mean", "rule"}. deltas: {match: delta(M - c23)
    of spellF1, raw input, as provided}. Incomplete (a run did not finish) counts as failed."""
    vals = [deltas.get(g) for g in SEALED]
    if not complete or any(v is None for v in vals):
        return {"pass": False, "mean": None,
                "rule": "incomplete: counts as failed (R3 Section 4); " + JUDGE_RULE}
    mean = float(np.mean(vals))
    return {"pass": bool(mean >= 0.0), "mean": mean, "rule": JUDGE_RULE}


def run_key(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()[:16]


# -- inputs ---------------------------------------------------------------------------------------

def dfl_files(root, game):
    mid = game[len("DFL-"):]
    folder = os.path.join(root, "dfl", mid)
    return {"atd_raw": os.path.join(folder, "DFL-{}_atd.txt".format(mid)),
            "atd_meta": os.path.join(folder, "DFL-{}_atd.xml".format(mid))}


def build_matched(out, game, raw_root, mtah_root, rehearsal):
    """Root of the sigma 3 view of one match under OUT/view-s3 (tools/atd_view.py), built once.
    In a rehearsal the result is compared with the development view under --mtah-root."""
    root = os.path.join(out, "view-s3")
    files = dfl_files(root, game)
    manifest = files["atd_raw"] + ".view.json"
    rec = {"root": root}
    if not os.path.isfile(manifest):
        command = [sys.executable, os.path.join(ROOT, "tools", "atd_view.py"), "--game", game,
                   "--in-root", raw_root, "--out-root", root, "--sigma", str(SIGMA),
                   "--players", "off"]
        if game in SEALED:
            command.append("--allow-sealed")
        log("build matched view {}".format(game))
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(out, "logs", "atd_view_{}.log".format(game)), "w") as handle:
            code = subprocess.call(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        if code != 0:
            raise RuntimeError("atd_view failed on {} (see OUT/logs)".format(game))
    rec["atd_raw_md5"] = md5_file(files["atd_raw"])
    if rehearsal:
        dev = dfl_files(os.path.join(mtah_root, "refs-smooth-s3"), game)["atd_raw"]
        rec["development_view_md5"] = md5_file(dev) if os.path.isfile(dev) else None
        rec["equals_development_view"] = rec["development_view_md5"] == rec["atd_raw_md5"]
    return rec


def engine_hash():
    from analysis import external_sweep as xs
    return xs.engine_source_hash()


def corrupt_hash():
    from tools.learned_spike import evaluate as ev
    return ev.corrupt_hash()


def overrides(system, weights, pair):
    if system == "c23":
        return {}
    return {"LEARNED_MODE": "model", "LEARNED_WEIGHTS": weights, "LEARNED_COST_HA": pair[0],
            "LEARNED_COST_AD": pair[1], "LEARNED_STRUCTURE": True}


def run_engine(ctx, sealed_name, game, inp, system):
    """CSV of one engine run (cached under OUT/runs by a key of everything it depends on)."""
    from analysis import external_sweep as xs
    over = overrides(system, ctx["weights"], ctx["pair"])
    root = ctx["roots"][game][inp]
    key = run_key({"game": game, "input": inp, "config": xs.resolve_config(over)[0],
                   "data_dir": root, "engine": ctx["engine_hash"],
                   "weights_md5": ctx["weights_md5"] if system == "M" else None,
                   "corrupt": ctx["corrupt_hash"] if inp == "stress" else None})
    where = os.path.join(ctx["out"], "runs", inp, sealed_name)
    csv = os.path.join(where, "{}_{}.csv".format(system, key))
    if os.path.isfile(csv):
        return csv
    os.makedirs(where, exist_ok=True)
    base = csv[:-4]
    write_json(base + ".config.json", over)
    env = dict(os.environ, MTAH_DATA_DIR=root)
    cache_raw = os.path.join(ctx["out"], "cache", "raw", "engine_data")
    if inp == "stress":
        command = [sys.executable, "-m", "tools.learned_spike.r3_sealed", "stress-child",
                   "--game", game, "--out", base + ".tmp.csv", "--config", base + ".config.json",
                   "--hidden-out", base + ".hidden.npz", "--engine-cache", cache_raw]
    else:
        cache = cache_raw if inp == "raw" else os.path.join(ctx["out"], "cache", inp,
                                                            "engine_data")
        os.makedirs(cache, exist_ok=True)
        command = [sys.executable, os.path.abspath(xs.__file__), "--engine-child", cache,
                   "--game", game, "--config", base + ".config.json", "--out", base + ".tmp.csv"]
    os.makedirs(cache_raw, exist_ok=True)
    log("run {} {} {} ({})".format(inp, game, system, key))
    t0 = time.time()
    with open(base + ".log", "w") as handle:
        code = subprocess.call(command, cwd=ROOT, env=env, stdout=handle,
                               stderr=subprocess.STDOUT)
    if code != 0:
        raise RuntimeError("engine run failed: {} {} {} (see {}.log)".format(inp, game, system,
                                                                             base))
    write_json(base + ".meta.json", {"game": game, "sealed_name": sealed_name, "input": inp,
                                     "system": system, "key": key, "overrides": over,
                                     "data_dir": root, "seconds": round(time.time() - t0, 1),
                                     "output_md5": md5_file(base + ".tmp.csv")})
    os.replace(base + ".tmp.csv", csv)
    return csv


# -- scoring --------------------------------------------------------------------------------------

def references(game, refs_root, sample_root, cache):
    """{"provided", "ctl2", "provided_md5", "path"} (evaluate.references, with the reference as
    provided of the dfl_bundle; a sealed match has no recorded md5, its md5 is recorded here)."""
    from analysis import control_reference as cref
    from analysis import external_sweep as xs
    import possession_metrics as pm
    path, digest = xs.prepare_reference(game, refs_root, sample_root, cache)
    provided = pm.label_codes(pm.read_label_csv(path)).astype(np.int8)
    lut = {"H": 0, "A": 1, "D": 2, "X": 3}
    ctl2 = np.array([lut[str(v)] for v in cref.control_labels(np.asarray(pm.LABELS)[provided],
                                                              CONTROL_FRAMES)], dtype=np.int8)
    return {"provided": provided, "ctl2": ctl2, "provided_md5": digest, "path": path}


def counts(codes, game):
    from tools.learned_spike import bar_report as br
    s = br.system_stats(codes, br.graded_runs(game), None)
    return {"possessions": s["possessions"], "d_share": s["d_share"],
            "possession_s_p50": s["possession_s"]["p50"]}


def score(ref, pred):
    import score_game as sg
    r = sg.score_codes(ref, pred)[0]
    return {"spell_f1": float(r["spell_f1_mean"]), "macro_f1": float(r["macro_f1"])}


# -- result files ---------------------------------------------------------------------------------

def result_paths(out, variant):
    name = "SEALED_RESULT_{}".format(variant)
    return os.path.join(out, name + ".md"), os.path.join(out, name + ".json")


def prereg_markdown(rec):
    w = []
    if rec["rehearsal"]:
        w.append("# REHEARSAL, not the sealed judge: SEALED_RESULT_{}\n".format(rec["variant"]))
        w.append("**This is a rehearsal of the tool on development matches** ({}). No sealed "
                 "match was opened, built or run.\n".format(", ".join(
                     "{} -> {}".format(k, v) for k, v in sorted(rec["mapping"].items()))))
    else:
        w.append("# SEALED_RESULT_{}: sealed DFL judge, round 3\n".format(rec["variant"]))
    w.append("Written by `tools/learned_spike/r3_sealed.py` (R3 Sections 4 and 9; adoption "
             "runbook Section 2). The pre-registration below was written into this file before "
             "any input of the two matches was built and before any engine run.\n")
    w.append("## Pre-registration (written {})\n".format(rec["started"]))
    w.append("- File: `{}`, md5 `{}`".format(rec["prereg"]["path"], rec["prereg"]["md5"]))
    w.append("- Variant {}, weights `{}` md5 `{}`, costs H-A {:g} / alive-dead {:g}, "
             "LEARNED_STRUCTURE on".format(rec["variant"], rec["weights"], rec["weights_md5"],
                                           rec["pair"][0], rec["pair"][1]))
    w.append("- Engine source hash `{}`, commit `{}`{}".format(
        rec["engine_source_hash"], rec["commit"], " (uncommitted changes in engine/, games.py "
        "or tools/learned_spike)" if rec["dirty"] else ""))
    w.append("- Judge: {}\n".format(JUDGE_RULE))
    w.append("Expectations given on the command line:\n")
    for line in rec["expectations"] or ["(none given)"]:
        w.append("- " + line)
    w.append("\nPre-registration text, verbatim:\n")
    w.append("```text\n{}\n```\n".format(rec["prereg"]["text"].rstrip("\n")))
    return "\n".join(w) + "\n"


def result_markdown(rec):
    w = [prereg_markdown(rec)]
    w.append("## Result{}\n".format(" (REHEARSAL)" if rec["rehearsal"] else ""))
    if rec.get("error"):
        w.append("**The run did not finish**: {}. It counts as failed (R3 Section 4).\n".format(
            rec["error"]))
    j = rec.get("judge")
    if j:
        w.append("**Judge: {}**. {}{}\n".format(
            "pass" if j["pass"] else "FAIL", j["rule"],
            "; mean delta {:+.4f}".format(j["mean"]) if j["mean"] is not None else ""))
    rows = rec.get("scores") or {}
    if rows:
        w.append("| match | input | reference | c23 spellF1 / macroF1 | M spellF1 / macroF1 | "
                 "delta spellF1 | possessions c23 / M / reference | D share c23 / M |")
        w.append("|---|---|---|---|---|---|---|---|")
        for name in SEALED:
            for inp in INPUTS:
                for ref in REFS:
                    cell = rows.get(name, {}).get(inp, {}).get(ref)
                    if not cell:
                        continue
                    label = name if not rec["rehearsal"] else "{} (= {})".format(
                        name, rec["mapping"][name])
                    w.append("| {} | {} | {} | {:.4f} / {:.4f} | {:.4f} / {:.4f} | {:+.4f} | "
                             "{} / {} / {} | {:.3f} / {:.3f} |".format(
                                 label, inp, ref, cell["c23"]["spell_f1"],
                                 cell["c23"]["macro_f1"], cell["M"]["spell_f1"],
                                 cell["M"]["macro_f1"], cell["delta_spell_f1"],
                                 cell["counts"]["c23"]["possessions"],
                                 cell["counts"]["M"]["possessions"],
                                 cell["counts"]["reference"]["possessions"],
                                 cell["counts"]["c23"]["d_share"],
                                 cell["counts"]["M"]["d_share"]))
        w.append("")
    if rec.get("outputs"):
        w.append("Output md5s: `{}`\n".format(json.dumps(rec["outputs"], sort_keys=True)))
    if rec.get("references"):
        w.append("References: `{}`\n".format(json.dumps(rec["references"], sort_keys=True)))
    if rec.get("checks"):
        w.append("Checks: `{}`\n".format(json.dumps(rec["checks"], sort_keys=True)))
    if rec.get("timings"):
        w.append("Wall time: {} s in all (`{}`).\n".format(
            rec["timings"].get("total_s"), json.dumps(rec["timings"], sort_keys=True)))
    return "\n".join(w) + "\n"


def write_result(out, rec):
    md, js = result_paths(out, rec["variant"])
    write_json(js, rec)
    with open(md, "w") as handle:
        handle.write(result_markdown(rec))
    return md


# -- the run --------------------------------------------------------------------------------------

def git_out(*args):
    try:
        return subprocess.check_output(("git",) + args, cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def check_result_json(path, weights_md5):
    """The R3 result must say M and hold the same weights md5 (when --result-json is given)."""
    rec = json.load(open(path))
    if rec.get("choice") != "M":
        raise SystemExit("refused: {} says {!r}, not M (the sealed judge runs only after M passed "
                         "criteria 3 to 10)".format(path, rec.get("choice")))
    if rec.get("final_npz_md5") and rec["final_npz_md5"] != weights_md5:
        raise SystemExit("refused: weights md5 {} differs from {} in {}".format(
            weights_md5, rec["final_npz_md5"], path))
    return {"path": os.path.abspath(path), "choice": rec["choice"],
            "final_costs": rec.get("final_costs"), "final_npz_md5": rec.get("final_npz_md5")}


def cmd_run(args):
    rehearsal = bool(args.rehearse_as)
    gate(os.environ, args.prereg, rehearsal)            # before anything else is read
    names = mapping(args.rehearse_as)
    out = os.path.abspath(os.path.expanduser(args.out_dir))
    if os.path.commonpath([out, ROOT]) == ROOT:
        raise SystemExit("--out-dir must be outside the repository")
    if args.variant not in VARIANTS:
        raise SystemExit("--variant: {}".format(VARIANTS))
    md_path, js_path = result_paths(out, args.variant)
    if os.path.isfile(js_path) and not rehearsal:
        prev = json.load(open(js_path))
        if not prev.get("rehearsal") and not args.resume:
            raise SystemExit("refused: {} records a start of the sealed judge at {} (status {}). "
                             "The sealed matches are run once; a run that did not finish counts "
                             "as failed. --resume only with the owner's decision, recorded."
                             .format(js_path, prev.get("started"), prev.get("status")))
    weights = os.path.abspath(args.weights)
    weights_md5 = md5_file(weights)
    if args.weights_md5 and args.weights_md5 != weights_md5:
        raise SystemExit("refused: {} md5 {} != --weights-md5 {}".format(weights, weights_md5,
                                                                         args.weights_md5))
    result_check = check_result_json(args.result_json, weights_md5) if args.result_json else None
    pair = tuple(float(x) for x in args.pair.split(","))
    if len(pair) != 2:
        raise SystemExit("--pair takes H-A,alive-dead")
    prereg_text = open(args.prereg).read()
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    dirty = bool(git_out("status", "--porcelain", "--untracked-files=no", "--", "engine",
                         "games.py", "tools/learned_spike"))
    t_start = time.time()
    rec = {"variant": args.variant, "rehearsal": rehearsal, "mapping": names,
           "status": "pre-registered, nothing built or run yet", "started": now(),
           "prereg": {"path": os.path.abspath(args.prereg), "md5": md5_file(args.prereg),
                      "text": prereg_text},
           "expectations": list(args.expect or []), "weights": weights,
           "weights_md5": weights_md5, "pair": list(pair), "structure": True,
           "engine_source_hash": engine_hash(), "commit": git_out("rev-parse", "HEAD"),
           "dirty": dirty, "result_json": result_check,
           "resumed": bool(args.resume and os.path.isfile(js_path)),
           "unseal": "{}={}".format(UNSEAL_VAR, os.environ.get(UNSEAL_VAR))}
    if os.path.isfile(js_path) and args.resume and not rehearsal:
        rec["resumed_from"] = {k: v for k, v in json.load(open(js_path)).items()
                               if k in ("started", "status", "error")}
    write_result(out, rec)                               # step 1: pre-registration first
    log("pre-registration written to {}".format(md_path))
    refs_root = os.path.abspath(os.path.expanduser(args.refs_root))
    sample_root = os.path.abspath(os.path.expanduser(args.sample_root))
    mtah_root = os.path.abspath(os.path.expanduser(args.mtah_root))
    ctx = {"out": out, "weights": weights, "weights_md5": weights_md5, "pair": pair,
           "engine_hash": rec["engine_source_hash"], "corrupt_hash": corrupt_hash(), "roots": {}}
    timings, checks, outputs = {}, {}, {}
    rec["status"] = "running"
    write_result(out, rec)
    try:
        from engine import data
        import score_game as sg
        csvs = {}
        for name in SEALED:                              # step 2: inputs
            game = names[name]
            files = dfl_files(refs_root, game)
            missing = [p for p in files.values() if not os.path.isfile(p)]
            if missing:
                raise RuntimeError("tracking files missing: {}".format(missing))
            t0 = time.time()
            matched = build_matched(out, game, refs_root, mtah_root, rehearsal)
            timings["matched_view_" + name] = round(time.time() - t0, 1)
            checks["matched_view_" + name] = matched
            if rehearsal and not matched.get("equals_development_view"):
                raise RuntimeError("rehearsal: the rebuilt matched view of {} differs from the "
                                   "development view".format(game))
            ctx["roots"][game] = {"raw": refs_root, "matched": matched["root"],
                                  "stress": refs_root}
        for name in SEALED:                              # step 3: 12 engine runs
            game = names[name]
            n = data.frame_count(game)
            for inp in INPUTS:
                for system in SYSTEMS:
                    t0 = time.time()
                    csv = run_engine(ctx, name, game, inp, system)
                    timings["{}|{}|{}".format(name, inp, system)] = round(time.time() - t0, 1)
                    sg.read_codes(csv, n)                # length and labels
                    csvs[(name, inp, system)] = csv
                    outputs["{}|{}|{}".format(name, inp, system)] = md5_file(csv)
            if rehearsal:
                from tools.learned_spike import r3_eval
                want = r3_eval.C23_DEV_MD5.get(game)
                got = outputs["{}|raw|c23".format(name)]
                checks["c23_raw_md5_" + name] = {"got": got, "R3_section_8": want,
                                                 "equal": got == want}
                if got != want:
                    raise RuntimeError("rehearsal: c23 raw md5 on {} {} != R3 Section 8 {}"
                                       .format(game, got, want))
        rec["outputs"] = outputs
        rec["status"] = "engine runs complete, references not yet read"
        write_result(out, rec)
        scores, ref_rec, deltas = {}, {}, {}             # step 4: references and scores
        for name in SEALED:
            game = names[name]
            n = data.frame_count(game)
            refs = references(game, refs_root, sample_root, os.path.join(out, "refs"))
            ref_rec[name] = {"path": refs["path"], "md5": refs["provided_md5"], "frames": n}
            ref_counts = {r: counts(refs[r], game) for r in REFS}
            scores[name] = {}
            for inp in INPUTS:
                codes = {s: sg.read_codes(csvs[(name, inp, s)], n) for s in SYSTEMS}
                sys_counts = {s: counts(codes[s], game) for s in SYSTEMS}
                scores[name][inp] = {}
                for ref in REFS:
                    cell = {s: score(refs[ref], codes[s]) for s in SYSTEMS}
                    cell["delta_spell_f1"] = cell["M"]["spell_f1"] - cell["c23"]["spell_f1"]
                    cell["delta_macro_f1"] = cell["M"]["macro_f1"] - cell["c23"]["macro_f1"]
                    cell["counts"] = dict(sys_counts, reference=ref_counts[ref])
                    scores[name][inp][ref] = cell
            deltas[name] = scores[name]["raw"]["provided"]["delta_spell_f1"]
        rec.update(scores=scores, references=ref_rec, deltas_raw_provided=deltas,
                   judge=judge(deltas, True), status="complete")
    except (Exception, SystemExit) as exc:                # noqa: B902 (recorded, then reported)
        rec.update(status="failed (incomplete)", error="{}: {}".format(type(exc).__name__, exc),
                   judge=judge({}, False))
    timings["total_s"] = round(time.time() - t_start, 1)
    rec.update(timings=timings, checks=checks, outputs=outputs, finished=now())
    md = write_result(out, rec)
    j = rec["judge"]
    print("{}sealed judge {}: {}{}; wrote {}".format(
        "REHEARSAL (not the sealed judge) " if rehearsal else "", args.variant,
        "pass" if j["pass"] else "FAIL", "" if j["mean"] is None else
        " (mean delta {:+.4f})".format(j["mean"]), md))
    return 0 if rec["status"] == "complete" else 4


def cmd_stress_child(args):
    """corrupt.py engine on one match (criterion 4's stress view). For a sealed match the
    environment check of `run` is repeated and corrupt.FORBIDDEN is lifted for that match only,
    in this process only."""
    from tools.learned_spike import corrupt
    from tools.learned_spike import evaluate as ev
    if args.game in SEALED:
        if os.environ.get(UNSEAL_VAR) != UNSEAL_VALUE:
            raise SystemExit("refused: {} is sealed and {} is not set".format(args.game,
                                                                              UNSEAL_VAR))
        corrupt.FORBIDDEN = set(corrupt.FORBIDDEN) - {args.game}
    elif args.game not in folds.OPEN:
        raise SystemExit("{}: a development or sealed DFL match only".format(args.game))
    argv = ["engine", "--game", args.game, "--out", args.out, "--config", args.config,
            "--hidden-out", args.hidden_out, "--engine-cache", args.engine_cache]
    return corrupt.main(argv + list(ev.VIEW_ARGS[ev.VIEW_STRESS]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out-dir", required=True, help="new folder outside the repository")
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--weights", required=True,
                   help="final weights npz (r3_score OUT/learned_v3.npz)")
    p.add_argument("--weights-md5", help="refuse if the npz md5 differs")
    p.add_argument("--pair", required=True, help="H-A,alive-dead of the label-free step")
    p.add_argument("--prereg", required=True, help="the committed pre-registration text")
    p.add_argument("--expect", action="append", help="a pre-registered expectation (repeat)")
    p.add_argument("--result-json", help="LEARNED_STAGE_RESULT_<variant>.json: must say M")
    p.add_argument("--rehearse-as", help="G1[,G2]: development DFL matches in place of the "
                                         "sealed matches (rehearsal)")
    p.add_argument("--resume", action="store_true",
                   help="resume an interrupted real run (only with the owner's decision; "
                        "recorded in the result)")
    p.add_argument("--refs-root", default=DEFAULT_REFS_ROOT)
    p.add_argument("--sample-root", default=DEFAULT_SAMPLE_ROOT)
    p.add_argument("--mtah-root", default=DEFAULT_MTAH_ROOT)
    p = sub.add_parser("stress-child")
    for name in ("--game", "--out", "--config", "--hidden-out", "--engine-cache"):
        p.add_argument(name, required=True)
    args = parser.parse_args(argv)
    return {"run": cmd_run, "stress-child": cmd_stress_child}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
