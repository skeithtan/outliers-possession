"""K2 checks that external_sweep.py does not report (calibration/COMPONENT_BACKDATE.md).

    python tools/analysis/backdate_checks.py cap --out CAP.json [--jobs N] [--cache DIR]
    python tools/analysis/backdate_checks.py checks --view raw|matched|sigma95 --cap-s S
        --out CHECKS.json [--games G ...] [--variant-ref S3_VARIANT.csv] [--jobs N] [--cache DIR]
    python tools/analysis/backdate_checks.py diff OFF.csv ON.csv [--game NAME] [--out DIFF.json]
    python tools/analysis/backdate_checks.py verdict --dir WORK [--out VERDICT.json]

Not part of the submission. Deterministic. Python 3.9.

Mode `cap` (COMPONENT_BACKDATE.md Section 3) runs the engine with BACKDATE_CHANGES false on the
three Metrica matches, raw and matched views, and the five open DFL matches on the same views
(recorded only). For every reference change from one team V to the other team U at frame r, it
follows the engine labels from r. If the engine label at r is V, the walk goes on while the
reference stays U and the engine stays V. If it stops on a frame where the reference is U and
the engine is U, the change is late by the number of frames walked (the T_late class of
tools/analysis/error_anatomy). The cap is the 90th percentile of the late lags pooled over the
three Metrica matches and the raw and matched views (numpy linear interpolation), rounded up to
a whole frame and divided by the frame rate.

Mode `checks` (Sections 4 and 6) runs, for each match of one view, the engine with the switch
off and on (one prepare(), two decodes; the BACKDATE_* fields are semantic). It records the
paired spellF1 difference on minus off, the moved changes and their shifts, the frames changed,
the possession spells under 1 s that exist only with the switch on (criterion v), the recall at
IoU 0.5 of reference spells under 2 s and of longer spells, and the possession count and D
share. With `--variant-ref`, METRICA-S3 is also scored against the F2 variant reference of
tools/analysis/redirect_forensics.py (row "METRICA-S3-F2").

Mode `diff` compares two engine outputs of the same match, label-free: frames that differ on
the graded frames, the stretches of changed frames and their lengths, and whether each stretch
ends at a direct H <-> A change of the first file. It reads no reference. Use it for the
BAR-ATH report of Section 6.

Mode `verdict` reads the outputs that `run_k2_eval.sh` writes into one work folder and applies
the acceptance test of Section 4 as written: `sweep_<view>_<provider>/BACKDATE_CHANGES.json` of
external_sweep.py, `checks_<view>.json` of mode `checks` and `barath_on.log` of engine.run. It
prints one line per criterion and KEEP or KILL. It runs no engine.

The tool never runs on the competition match in modes `cap` and `checks` (external_sweep
check_games) and never on a sealed match.
"""

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, "tools")
ANALYSIS = os.path.join(TOOLS, "analysis")
for _path in (ROOT, TOOLS, ANALYSIS, os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

H, A, D = 0, 1, 2
CAP_QUANTILE = 90.0                       # Section 3: the 90th percentile of the late lags
CAP_POOL_VIEWS = ("raw", "matched")       # Section 3: the pooled views
SHORT_SPELL_FRAMES = 25                   # criterion (v): spells under 1 s at 25 fps
SHORT_RECALL_S = 2.0                      # RESEARCH-4 0.2 (4): spells under 2 s
RECALL_IOU = 0.5
VARIANT_GAME = "METRICA-S3"


# -- pure functions (unit tested) -------------------------------------------------------------

def late_lags(ref, ours, runs):
    """[lag, ...]: frames by which `ours` follows each reference change between teams late.

    `ref` and `ours` are int codes (0 H, 1 A, 2 D, 3 X), `ours` with abstentions filled.
    `runs` are the graded runs [(start, stop), ...] of the reference. A reference change V -> U
    at r (both teams, inside one run) is late if ours[r] is V and, walking on while ref stays U
    and ours stays V, the walk ends on a frame where ref is U and ours is U. The lag is the
    number of frames walked (at least 1).
    """
    import numpy as np

    ref = np.asarray(ref)
    ours = np.asarray(ours)
    lags = []
    for start, stop in runs:
        for r in range(start + 1, stop):
            v, u = ref[r - 1], ref[r]
            if v == u or v not in (H, A) or u not in (H, A) or ours[r] != v:
                continue
            k = r
            while k < stop and ref[k] == u and ours[k] == v:
                k += 1
            if k < stop and ref[k] == u and ours[k] == u:
                lags.append(int(k - r))
    return lags


def cap_from_lags(lags, fps, quantile=CAP_QUANTILE):
    """(cap frames, cap seconds): the quantile of the lags, rounded up to a frame. None if empty."""
    import numpy as np

    if not lags:
        return None, None
    frames = int(math.ceil(float(np.percentile(np.asarray(lags, dtype=np.float64), quantile))
                           - 1e-9))
    return frames, frames / float(fps)


def spells(codes, runs):
    """[(start, stop, team), ...]: the possession spells of codes on the runs (pm definition)."""
    import possession_metrics as pm

    return [(int(a), int(b), int(t)) for a, b, t in pm.possession_spells(codes, runs)]


def new_short_spells(off, on, runs, short=SHORT_SPELL_FRAMES):
    """Spells under `short` frames of `on` that are not spells of `off` (same start, end, team)."""
    before = set(spells(off, runs))
    return [s for s in spells(on, runs) if s[1] - s[0] < short and s not in before]


def new_short_possessions(off, on, runs, short=SHORT_SPELL_FRAMES):
    """K2b criterion (v), COMPONENT_BACKDATE_B.md: spells under `short` frames of `on` that
    contain no spell of `off` of the same team. A short spell that only grew is not new."""
    before = spells(off, runs)
    out = []
    for a, b, t in spells(on, runs):
        if b - a >= short:
            continue
        if not any(t == t0 and a <= a0 and b0 <= b for a0, b0, t0 in before):
            out.append((a, b, t))
    return out


def edge_exempt(short, off_engine, runs, short_frames=SHORT_SPELL_FRAMES):
    """K2b criterion (v), COMPONENT_BACKDATE_B.md 4.1: the spells of `short` (verbatim-new short
    spells) that are exempt. A spell is exempt only if (a) its start is the start or its end is
    the end of a graded run of the reference, and (b) the switch-off engine spell of the same team
    that overlaps it, counted on the engine labels before any cut to the reference runs
    (`off_engine`, codes, whole match), has at least `short_frames` frames."""
    import numpy as np

    starts = {int(a) for a, _ in runs}
    stops = {int(b) for _, b in runs}
    codes = np.asarray(off_engine)
    out = []
    for a, b, t in short:
        if a not in starts and b not in stops:
            continue
        hits = np.flatnonzero(codes[a:b] == t)
        longest = 0
        for k in hits:
            i = j = a + int(k)
            while i > 0 and codes[i - 1] == t:
                i -= 1
            while j < len(codes) and codes[j] == t:
                j += 1
            longest = max(longest, j - i)
        if longest >= short_frames:
            out.append((a, b, t))
    return out


def recall_split(ref, pred, runs, fps, split_s=SHORT_RECALL_S, iou=RECALL_IOU):
    """{"short": [recalled, n], "long": [recalled, n]}: reference spells under / from split_s."""
    import numpy as np
    import possession_metrics as pm

    gt_spells = pm.possession_spells(ref, runs)
    pred_spells = pm.possession_spells(pred, runs)
    flags = pm.matched_gt_flags(pm.spell_overlaps(gt_spells, pred_spells), len(gt_spells), iou)
    seconds = np.array([(b - a) / float(fps) for a, b, _ in gt_spells])
    out = {}
    for name, mask in (("short", seconds < split_s), ("long", seconds >= split_s)):
        out[name] = [int(np.asarray(flags)[mask].sum()) if len(flags) else 0, int(mask.sum())]
    return out


def label_diff(off, on, graded):
    """Label-free comparison of two code vectors on the graded mask (mode `diff`)."""
    import numpy as np

    off = np.asarray(off)
    on = np.asarray(on)
    changed = (off != on) & np.asarray(graded, dtype=bool)
    idx = np.flatnonzero(changed)
    stretches = []
    if idx.size:
        breaks = np.flatnonzero(np.diff(idx) != 1)
        starts = np.concatenate([[idx[0]], idx[breaks + 1]])
        stops = np.concatenate([idx[breaks], [idx[-1]]]) + 1
        stretches = [(int(a), int(b)) for a, b in zip(starts, stops)]
    lengths = [b - a for a, b in stretches]
    at_change = 0
    for a, b in stretches:
        if (b < len(off) and off[b - 1] in (H, A) and off[b] in (H, A) and off[b] != off[b - 1]
                and np.all(on[a:b] == off[b])):
            at_change += 1
    return {"frames_changed": int(changed.sum()), "stretches": len(stretches),
            "stretches_ending_at_a_change_of_off": at_change,
            "median_shift": float(np.median(lengths)) if lengths else None,
            "max_shift": int(max(lengths)) if lengths else None,
            "shift_quantiles": ([float(q) for q in np.percentile(lengths, [10, 25, 50, 75, 90])]
                                if lengths else None)}


# -- engine children ---------------------------------------------------------------------------

def _config(overrides):
    import dataclasses
    from engine import config

    cfg = config.Config.default()
    return dataclasses.replace(cfg, **overrides) if overrides else cfg


def child_cap(game, data_cache, ref_path, overrides):
    """Late lags of one match and view with the switch off."""
    from engine import data, possession
    import possession_metrics as pm

    data.DEFAULT_CACHE_DIR = data_cache
    cfg = _config(dict(overrides, BACKDATE_CHANGES=False))
    labels = possession.run(cfg, game=game)
    ref = pm.label_codes(pm.read_label_csv(ref_path))
    runs = pm.valid_runs(ref)
    ours = pm.fill_abstentions(pm.label_codes(labels), runs)
    return {"game": game, "lags": late_lags(ref, ours, runs), "fps": float(pm.FPS)}


def child_checks(game, data_cache, ref_path, overrides, cap_s, draws, variant_ref):
    """Off and on runs of one match and view, and the Section 4 numbers."""
    import numpy as np
    from engine import data, possession
    import possession_metrics as pm
    import score_game as sg

    data.DEFAULT_CACHE_DIR = data_cache
    off_cfg = _config(dict(overrides, BACKDATE_CHANGES=False, BACKDATE_CAP_S=float(cap_s)))
    on_cfg = _config(dict(overrides, BACKDATE_CHANGES=True, BACKDATE_CAP_S=float(cap_s)))
    inputs = possession.prepare(off_cfg, game)
    off_labels = possession.run(off_cfg, inputs=inputs)
    on_labels, parts = possession.run(on_cfg, return_parts=True, inputs=inputs)
    off_codes, on_codes = pm.label_codes(off_labels), pm.label_codes(on_labels)
    shifts = [row["t1"] - row["t0"] for row in parts["backdate"]]
    common = {"moved": len(shifts),
              "median_shift": float(np.median(shifts)) if shifts else None,
              "shift_quantiles": ([float(q) for q in np.percentile(shifts, [10, 25, 50, 75, 90])]
                                  if shifts else None),
              "frames_changed_all": int(np.sum(off_codes != on_codes))}
    out = []
    refs = [(game, ref_path)]
    if variant_ref and game == VARIANT_GAME:
        refs.append((game + "-F2", variant_ref))
    for name, path in refs:
        ref = pm.label_codes(pm.read_label_csv(path))
        runs = pm.valid_runs(ref)
        off = pm.fill_abstentions(off_codes, runs)
        on = pm.fill_abstentions(on_codes, runs)
        stats_off = sg.label_stats(off_codes, runs)
        stats_on = sg.label_stats(on_codes, runs)
        short = new_short_spells(off, on, runs)
        fresh = new_short_possessions(off, on, runs)
        exempt = edge_exempt(short, off_codes, runs)
        row = dict(common, game=name, reference=os.path.basename(path),
                   spell_f1_off=float(pm.spell_scores(ref, off, runs)["mean_f1"]),
                   spell_f1_on=float(pm.spell_scores(ref, on, runs)["mean_f1"]),
                   delta=sg.paired_delta(ref, on_codes, off_codes, n_draws=draws),
                   new_short_spells=len(short), new_short_spell_list=short,
                   new_short_possessions=len(fresh), new_short_possession_list=fresh,
                   new_short_spells_exempt=len(exempt), new_short_spell_exempt_list=exempt,
                   new_short_spells_binding=len(short) - len(exempt),
                   short_spells_off=sum(1 for s in spells(off, runs)
                                        if s[1] - s[0] < SHORT_SPELL_FRAMES),
                   short_spells_on=sum(1 for s in spells(on, runs)
                                       if s[1] - s[0] < SHORT_SPELL_FRAMES),
                   recall_off=recall_split(ref, off, runs, pm.FPS),
                   recall_on=recall_split(ref, on, runs, pm.FPS),
                   possessions_off=stats_off["possessions"],
                   possessions_on=stats_on["possessions"],
                   d_share_off=stats_off["d_share"], d_share_on=stats_on["d_share"])
        out.append(row)
    return out


def _run_child(job):
    env = dict(os.environ, MTAH_DATA_DIR=job["data_dir"])
    command = [sys.executable, os.path.abspath(__file__), "child", "--payload",
               json.dumps(job, sort_keys=True)]
    done = subprocess.run(command, cwd=ROOT, env=env, check=True, stdout=subprocess.PIPE)
    return json.loads(done.stdout.decode("utf-8").strip().splitlines()[-1])


def _jobs(games, view, args, extra):
    import external_sweep as es

    refs_root = os.path.expanduser(args.refs_root or es.DEFAULT_REFS_ROOT)
    sample_root = os.path.expanduser(args.sample_root or es.DEFAULT_SAMPLE_ROOT)
    mtah_root = os.path.expanduser(args.mtah_root or es.DEFAULT_MTAH_ROOT)
    cache = os.path.abspath(os.path.expanduser(args.cache or es.DEFAULT_CACHE))
    jobs = []
    for game in es.check_games(games):
        ref, _ = es.prepare_reference(game, refs_root, sample_root, cache)
        job = {"game": game, "view": view, "ref": ref,
               "data_dir": es.data_root(game, view, refs_root, sample_root, mtah_root),
               "data_cache": os.path.join(es.view_dir(cache, view), "engine_data")}
        job.update(extra)
        jobs.append(job)
    return jobs


def _pool(jobs, n):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, n)) as pool:
        return list(pool.map(_run_child, jobs))


def _write(report, path):
    text = json.dumps(report, sort_keys=True, indent=1)
    if path:
        with open(path, "w") as handle:
            handle.write(text + "\n")
    return text


def main_cap(args):
    import numpy as np
    import external_sweep as es

    overrides = _load_overrides(args.config)
    games = args.games or list(es.OPEN_GAMES)
    results = []
    for view in args.views:
        for res in _pool(_jobs(games, view, args, {"mode": "cap", "overrides": overrides}),
                         args.jobs):
            res["view"] = view
            results.append(res)
    pooled = [lag for r in results if r["game"].startswith("METRICA-")
              and r["view"] in CAP_POOL_VIEWS for lag in r["lags"]]
    fps = results[0]["fps"] if results else 25.0
    cap_frames, cap_s = cap_from_lags(pooled, fps)
    rows = []
    for r in results:
        lags = np.asarray(r["lags"], dtype=np.float64)
        rows.append({"game": r["game"], "view": r["view"], "late": int(lags.size),
                     "quantiles_10_25_50_75_90": ([float(q) for q in np.percentile(
                         lags, [10, 25, 50, 75, 90])] if lags.size else None),
                     "in_pool": r["game"].startswith("METRICA-") and r["view"] in CAP_POOL_VIEWS})
    report = {"rule": "p{:g} of the late lags, METRICA-S1..S3, views {}, ceil to a frame".format(
                  CAP_QUANTILE, "+".join(CAP_POOL_VIEWS)),
              "overrides": overrides, "pooled_n": len(pooled),
              "pooled_quantiles_10_25_50_75_90": ([float(q) for q in np.percentile(
                  pooled, [10, 25, 50, 75, 90])] if pooled else None),
              "cap_frames": cap_frames, "cap_s": cap_s, "matches": rows}
    _write(report, args.out)
    for row in rows:
        print("{:8s} {:12s} late {:4d}  q10/25/50/75/90 {}{}".format(
            row["view"], row["game"], row["late"], row["quantiles_10_25_50_75_90"],
            "" if row["in_pool"] else "  (recorded only)"))
    print("pooled n {}  cap {} frames = {} s".format(len(pooled), cap_frames, cap_s))
    return 0 if cap_s is not None else 2


def main_checks(args):
    import numpy as np
    import external_sweep as es

    overrides = _load_overrides(args.config)
    games = args.games or list(es.OPEN_GAMES)
    variant = os.path.abspath(os.path.expanduser(args.variant_ref)) if args.variant_ref else None
    jobs = _jobs(games, args.view, args, {"mode": "checks", "overrides": overrides,
                                          "cap_s": args.cap_s, "draws": args.draws,
                                          "variant_ref": variant})
    rows = [row for res in _pool(jobs, args.jobs) for row in res]
    report = {"view": args.view, "cap_s": args.cap_s, "overrides": overrides, "matches": rows}
    for group, prefix in (("metrica", "METRICA-"), ("dfl", "DFL-")):
        picked = [r for r in rows if r["game"].startswith(prefix) and not r["game"].endswith("-F2")]
        report[group + "_mean_delta"] = (float(np.mean([r["delta"]["delta"] for r in picked]))
                                         if picked else None)
        report[group + "_new_short_spells"] = sum(r["new_short_spells"] for r in picked)
        report[group + "_new_short_possessions"] = sum(r["new_short_possessions"]
                                                      for r in picked)
        report[group + "_new_short_spells_exempt"] = sum(r["new_short_spells_exempt"]
                                                        for r in picked)
        report[group + "_new_short_spells_binding"] = sum(r["new_short_spells_binding"]
                                                         for r in picked)
        for state in ("off", "on"):
            for part in ("short", "long"):
                hit = sum(r["recall_" + state][part][0] for r in picked)
                n = sum(r["recall_" + state][part][1] for r in picked)
                report["{}_recall_{}_{}".format(group, part, state)] = hit / float(n) if n else None
    _write(report, args.out)
    for r in rows:
        d = r["delta"]
        print("{:8s} {:15s} {:.4f} -> {:.4f}  delta {:+.4f} [{:+.4f}, {:+.4f}]  moved {:4d}"
              "  median shift {}  new <1 s {}".format(args.view, r["game"], r["spell_f1_off"],
                                                     r["spell_f1_on"], d["delta"], d["low"],
                                                     d["high"], r["moved"], r["median_shift"],
                                                     r["new_short_spells"]))
    print("{} means: Metrica {}, DFL {}".format(args.view, report["metrica_mean_delta"],
                                                report["dfl_mean_delta"]))
    return 0


def main_diff(args):
    import numpy as np
    from engine import data
    import possession_metrics as pm

    off = pm.label_codes(pm.read_label_csv(args.off))
    on = pm.label_codes(pm.read_label_csv(args.on))
    if len(off) != len(on):
        raise SystemExit("the two files have different lengths")
    from engine import possession

    graded = np.zeros(len(off), dtype=bool)
    game = data.resolve_game(args.game)
    _, _, blocks, _, _ = possession.match_inputs(game)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    report = dict(label_diff(off, on, graded), game=game, off=os.path.basename(args.off),
                  on=os.path.basename(args.on))
    print(_write(report, args.out))
    return 0


def sweep_row(path):
    """The BACKDATE_CHANGES true row of an external_sweep.py JSON."""
    with open(path) as handle:
        table = json.load(handle)
    rows = [row for row in table["rows"] if row["value"] is True]
    if len(rows) != 1:
        raise SystemExit("{}: expected one row with the value true".format(path))
    return rows[0]


def verdict(sweeps, checks, bararth_log, criterion_v="verbatim"):
    """[(criterion, passed, detail), ...] and the overall result of the Section 4 test.

    `sweeps` is {(view, provider): sweep_row()}, provider "metrica" or "dfl", views "raw",
    "matched" and "sigma95". `checks` is {view: mode `checks` report}. `bararth_log` is the
    text that engine.run printed with the switch on. `criterion_v` "verbatim" is K2's (v)
    (COMPONENT_BACKDATE.md 4.1); "edge" is K2b's (v) (COMPONENT_BACKDATE_B.md 4.1): the verbatim
    test with the reference-edge exemption of edge_exempt(); the containment count is recorded.
    """
    out = []

    def add(name, passed, detail):
        out.append((name, bool(passed), detail))

    judged = ("raw", "matched")
    for view in judged:
        row = sweeps[(view, "metrica")]
        cm = row["versus_base"]
        add("(i) Metrica mean >= 0, 5 % end >= 0, " + view,
            cm["mean_delta"] >= 0 and cm["low"] >= 0,
            "{:+.4f} [{:+.4f}, {:+.4f}]".format(cm["mean_delta"], cm["low"], cm["high"]))
        dfl = sweeps[(view, "dfl")]["versus_base"]
        add("(ii) DFL mean >= -0.010, " + view, dfl["mean_delta"] >= -0.010,
            "{:+.4f} [{:+.4f}, {:+.4f}]".format(dfl["mean_delta"], dfl["low"], dfl["high"]))
        deltas = [g["versus_base"]["delta"] for g in row["games"].values()]
        add("(iii) Metrica delta >= 0 on at least 2 of 3, " + view,
            sum(d >= 0 for d in deltas) >= 2 and len(deltas) == 3,
            " ".join("{:+.4f}".format(d) for d in deltas))
    add("(iv) BAR-ATH vetoes PASS", "verdict: PASS" in bararth_log,
        "engine.run verdict line")
    for view in judged:
        n = checks[view]["metrica_new_short_spells"]
        if criterion_v == "verbatim":
            add("(v) no new Metrica spell under 1 s, " + view, n == 0, "{} new".format(n))
        else:
            m = checks[view]["metrica_new_short_spells_binding"]
            e = checks[view]["metrica_new_short_spells_exempt"]
            c = checks[view]["metrica_new_short_possessions"]
            add("(v, K2b) no new Metrica spell under 1 s, reference-edge exemption, " + view,
                m == 0, "{} new, {} exempt; containment count {} recorded".format(m + e, e, c))
    raw = sweeps[("raw", "metrica")]["games"]
    add("extra: S1, S2, S3 raw, no interval wholly below zero",
        all(g["versus_base"]["high"] >= 0 for g in raw.values()),
        " ".join("{}<{:+.4f}>".format(k[-2:], g["versus_base"]["high"])
                 for k, g in sorted(raw.items())))
    means = {"Metrica sigma95": sweeps[("sigma95", "metrica")]["versus_base"],
             "Metrica matched": sweeps[("matched", "metrica")]["versus_base"],
             "DFL matched": sweeps[("matched", "dfl")]["versus_base"]}
    add("extra: non-negative means (Metrica sigma95, Metrica matched, DFL matched)",
        all(m["mean_delta"] >= 0 for m in means.values()),
        ", ".join("{} {:+.4f}".format(k, m["mean_delta"]) for k, m in means.items()))
    add("extra: supported (5 % end > 0) on at least one of them",
        any(m["low"] > 0 for m in means.values()),
        ", ".join("{} {:+.4f}".format(k, m["low"]) for k, m in means.items()))
    for view in judged:
        f2 = [r for r in checks[view]["matches"] if r["game"] == VARIANT_GAME + "-F2"]
        add("extra: F2 variant of S3 delta >= 0, " + view,
            len(f2) == 1 and f2[0]["delta"]["delta"] >= 0,
            "{:+.4f}".format(f2[0]["delta"]["delta"]) if f2 else "missing")
    every = [g["versus_base"]["delta"] for (view, _), row in sweeps.items() if view in judged
             for g in row["games"].values()]
    add("not inert: some delta differs from zero", any(d != 0 for d in every),
        "{} match deltas".format(len(every)))
    return out, all(passed for _, passed, _ in out)


def main_verdict(args):
    sweeps = {}
    for view in ("raw", "matched", "sigma95"):
        for provider in ("metrica", "dfl"):
            sweeps[(view, provider)] = sweep_row(os.path.join(
                args.dir, "sweep_{}_{}".format(view, provider), "BACKDATE_CHANGES.json"))
    checks = {}
    for view in ("raw", "matched", "sigma95"):
        with open(os.path.join(args.dir, "checks_{}.json".format(view))) as handle:
            checks[view] = json.load(handle)
    with open(os.path.join(args.dir, "barath_on.log")) as handle:
        log = handle.read()
    lines, keep = verdict(sweeps, checks, log, args.criterion_v)
    for name, passed, detail in lines:
        print("{:4s}  {}  ({})".format("pass" if passed else "FAIL", name, detail))
    recorded = []
    for view in ("raw", "matched", "sigma95"):
        c = checks[view]
        recorded.append({"view": view,
                         "metrica_recall_short_off_on": [c["metrica_recall_short_off"],
                                                         c["metrica_recall_short_on"]],
                         "metrica_recall_long_off_on": [c["metrica_recall_long_off"],
                                                        c["metrica_recall_long_on"]],
                         "dfl_mean": sweeps[(view, "dfl")]["versus_base"],
                         "moved": {r["game"]: r["moved"] for r in c["matches"]}})
    print("recorded (decides nothing): " + json.dumps(recorded, sort_keys=True))
    print("RESULT: {}".format("KEEP" if keep else "KILL"))
    _write({"criteria": [{"name": n, "pass": p, "detail": d} for n, p, d in lines],
            "keep": keep, "recorded": recorded}, args.out)
    return 0


def _load_overrides(path):
    if not path:
        return {}
    with open(path) as handle:
        return json.load(handle)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("cap", "checks"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--games", nargs="*", default=None)
        p.add_argument("--config", default=None, help="JSON overrides of engine/config.py")
        p.add_argument("--jobs", type=int, default=1)
        p.add_argument("--cache", default=None)
        p.add_argument("--refs-root", default=None)
        p.add_argument("--sample-root", default=None)
        p.add_argument("--mtah-root", default=None)
    sub.choices["cap"].add_argument("--views", nargs="+", default=list(CAP_POOL_VIEWS),
                                    choices=("raw", "matched", "sigma95"))
    checks = sub.choices["checks"]
    checks.add_argument("--view", required=True, choices=("raw", "matched", "sigma95"))
    checks.add_argument("--cap-s", type=float, required=True)
    checks.add_argument("--draws", type=int, default=1000)
    checks.add_argument("--variant-ref", default=None)
    diff = sub.add_parser("diff")
    diff.add_argument("off")
    diff.add_argument("on")
    diff.add_argument("--game", default=None)
    diff.add_argument("--out", default=None)
    verdict_parser = sub.add_parser("verdict")
    verdict_parser.add_argument("--dir", required=True)
    verdict_parser.add_argument("--criterion-v", choices=("verbatim", "edge"), default="verbatim",
                                help="K2 (v) as written, or K2b (v) with the edge exemption")
    verdict_parser.add_argument("--out", default=None)
    child = sub.add_parser("child")
    child.add_argument("--payload", required=True)
    args = parser.parse_args(argv)
    if args.mode == "child":
        job = json.loads(args.payload)
        if job["mode"] == "cap":
            result = child_cap(job["game"], job["data_cache"], job["ref"], job["overrides"])
        else:
            result = child_checks(job["game"], job["data_cache"], job["ref"], job["overrides"],
                                  job["cap_s"], job["draws"], job["variant_ref"])
        print(json.dumps(result, sort_keys=True))
        return 0
    return {"cap": main_cap, "checks": main_checks, "diff": main_diff,
            "verdict": main_verdict}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
