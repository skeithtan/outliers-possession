"""Per-transition forensics on the public window (PLAN.md 5.2, card P0-D).

    python tools/forensics.py evaluation/results/<name>.csv [--window 50] [--only missed|spurious|all]
                              [--json] [--reappear-lookback 30] [--game name]

Needs the scorer's reports (`python evaluation/run_evaluation.py` writes
evaluation/reports/<name>.json and <name>_transitions.csv); exits 2 with that hint if they are
absent. Reads the public ground truth ONLY through tools/gt.py (tools/ and tests/ may; engine/ may
not). Debugging aid only: it explains misses, it never produces labels.

For every GT transition in the public window: frame, from->to, matched / missed (tolerance = the
loosest tau of the report), our nearest same-destination transition and its latency, a +-window
trace every 3 frames (`dH dA v dv_3 meanv ours gt`), engine events if `engine.events` exists, and
the Smart Tagging phases overlapping the trace. For each spurious transition of ours: the same
trace centred on our frame. Then the public summary, latencies grouped by kind, and the sandwich
check (GT `X D X` with the same team either side: does our output keep the two spells apart?).
Segments are recomputed from the GT CSV, never hardcoded (Appendix A is only a cross-check).
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as cm  # noqa: E402
from _common import A, D, H, LETTER, X, pm  # noqa: E402
from gt import public_gt_codes  # noqa: E402  (the only permitted GT reader)

TRACE_STEP = 3             # frames between trace rows (card P0-D)
DEFAULT_WINDOW = 50        # +-2 s at 25 fps (PLAN 5.2)
# Kick vs reappear (PLAN 1.4). The card's literal rule is "ball missing on restart frame - 1",
# which is 0 of 4 public restarts: 12989 and 17838 reappear 10-13 frames BEFORE the GT restart
# (PLAN 1.4 table). "Reappear" therefore looks back this many frames for a missing ball; the
# literal rule is --reappear-lookback 1. Value 30 = 1.2 s, longer than the 10-13 frame offsets
# and shorter than any rest period before a kick (16400, 18389: ball visible for 60+ frames).
REAPPEAR_LOOKBACK = 30
KINDS = ("H<->A", "alive->D", "D->alive (kick)", "D->alive (reappear)")


# -- GT / transition bookkeeping -------------------------------------------------------------------

def kind_of(gt_from, gt_to, ball_missing, frame, lookback):
    """Transition kind label for latency grouping (PLAN 1.4)."""
    if gt_from in (H, A) and gt_to in (H, A):
        return KINDS[0]
    if gt_to == D:
        return KINDS[1]
    lo = max(0, frame - lookback)
    reappear = bool(ball_missing[lo:frame].any())
    return KINDS[3] if reappear else KINDS[2]


def nearest_same_destination(gt_tr, pred_trans):
    """Our transition into the same label as `gt_tr` closest in time (any distance), or None."""
    cands = [p for p in pred_trans if p[2] == gt_tr[2]]
    if not cands:
        return None
    return min(cands, key=lambda p: (abs(p[0] - gt_tr[0]), p[0]))


def sandwich_check(gt_codes, pred_filled, runs):
    """For each GT `T D T` (same alive team either side), is our output separated there?

    `separate` is True when our filled output has a label other than T somewhere inside the GT
    `D` stretch (a non-T label between the two spells); False means the two spells were merged.
    """
    segs = pm.segments(gt_codes, runs)
    out = []
    for i in range(1, len(segs) - 1):
        (_, _, before), (start, stop, mid), (_, _, after) = segs[i - 1], segs[i], segs[i + 1]
        if mid != D or before != after or before not in (H, A):
            continue
        ours = pred_filled[start:stop]
        counts = {LETTER[k]: int((ours == k).sum()) for k in (H, A, D)}
        out.append({"team": LETTER[before], "gt_d_start": int(start), "gt_d_stop": int(stop),
                    "separate": bool((ours != before).any()), "our_frames": counts,
                    "our_d_share_of_gt_d": counts["D"] / float(stop - start)})
    return out


def read_transitions_csv(path):
    """Rows of the scorer's <name>_transitions.csv as dicts (ints where present)."""
    rows = []
    with open(path) as handle:
        for rec in csv.DictReader(handle):
            row = dict(rec)
            for key in ("gt_frame", "pred_frame", "latency"):
                row[key] = int(rec[key]) if rec.get(key) not in (None, "") else None
            rows.append(row)
    return rows


# -- traces ----------------------------------------------------------------------------------------

def trace_offsets(window, step=TRACE_STEP):
    """Offsets -window..+window every `step` frames, always including 0."""
    return sorted(set(range(0, -window - 1, -step)) | set(range(0, window + 1, step)))


def _finite(value):
    """float(value) or None when NaN/inf (keeps the JSON strictly valid)."""
    return float(value) if np.isfinite(value) else None


def build_trace(tactical, pitch, gt_codes, pred_filled, centre, window):
    """List of row dicts for the trace around `centre` (frames clipped to the video)."""
    n = len(gt_codes)
    offs = [o for o in trace_offsets(window) if 0 <= centre + o < n]
    lo, hi = centre + min(offs), centre + max(offs) + 1
    kin = cm.kinematics(tactical, lo, hi, pitch)
    rows = []
    for o in offs:
        i = centre + o - lo
        rows.append({"off": o, "frame": centre + o,
                     "dH": _finite(kin["dH"][i]), "dA": _finite(kin["dA"][i]),
                     "v": _finite(kin["v"][i]), "dv_3": _finite(kin["dv_3"][i]),
                     "meanv": _finite(kin["meanv"][i]),
                     "ours": LETTER[int(pred_filled[centre + o])],
                     "gt": LETTER[int(gt_codes[centre + o])]})
    return rows


def _num(value):
    return "    ." if value is None or not np.isfinite(value) else "{:5.1f}".format(value)


def format_trace(rows, mark_frame):
    lines = ["      off  frame     dH    dA     v  dv_3 meanv  ours gt"]
    for r in rows:
        lines.append("   {} {:+4d} {:>6} {} {} {} {} {}    {}    {}".format(
            "->" if r["frame"] == mark_frame else "  ", r["off"], r["frame"], _num(r["dH"]),
            _num(r["dA"]), _num(r["v"]), _num(r["dv_3"]), _num(r["meanv"]), r["ours"], r["gt"]))
    return lines


def tag_overlaps(tags, lo, hi):
    """Smart Tagging rows overlapping [lo, hi) as short strings relative to `lo + window`."""
    hit = tags[(tags["start_frame"] < hi) & (tags["end_frame"] > lo)]
    return [{"code": r.code, "team": r.team, "type": r.type, "start_frame": int(r.start_frame),
             "end_frame": int(r.end_frame)} for r in hit.itertuples(index=False)]


def format_tags(items, centre):
    if not items:
        return ["   tags: none overlap"]
    lines = ["   Smart Tagging phases overlapping the window (offsets from the transition):"]
    for it in items:
        lines.append("     {:<30} {:<4} {:+6d} .. {:+6d}{}".format(
            it["code"], it["team"] or "-", it["start_frame"] - centre, it["end_frame"] - centre,
            "  [{}]".format(it["type"]) if it["type"] else ""))
    return lines


def events_in_window(events_module, tactical, lo, hi):
    """Engine events in [lo, hi) as dicts, or [] if the module has no recognised entry point.

    Duck-typed against PLAN 3.3's table `frame, kind, team, confidence`: the module may expose
    `event_table(tactical)` returning a DataFrame (or list of dicts) with those columns. Any
    other shape is ignored; forensics must keep working while the engine is being written.
    """
    fn = getattr(events_module, "event_table", None)
    if fn is None:
        return []
    try:
        table = fn(tactical)
        records = table.to_dict("records") if hasattr(table, "to_dict") else list(table)
        return [r for r in records if lo <= int(r["frame"]) < hi]
    except Exception:
        return []


def kickoff_check(pred, tactical, tags, blocks, kickoffs):
    """The kickoffs of our output (full match) with the kickoff rules of the engine (C6).

    Uses engine.possession.kickoff_decisions() and goal_kickoffs() with tracks.mark_nearest().
    The CSV gives the stoppages (our `D`) and the restarts (our first alive frame after a `D`),
    so the restart candidates of the engine are not rebuilt. Rule "none" means the centre mark
    and Law 8 give no team: the engine then used the restart candidate team or the owner, and
    `ours` shows the result. Returns [] if the engine modules are missing.
    One row per kickoff: kind, frame, rule, team, mark distance (m), our label at the frame,
    and for an after-goal kickoff the GOALS window end and its team.
    """
    import pandas as pd
    from engine import data
    from engine.config import Config
    possession = cm.optional_module("possession")
    tracks = cm.optional_module("tracks")
    if possession is None or tracks is None or not hasattr(possession, "goal_kickoffs"):
        return []
    cfg = Config.default()
    mark = tracks.mark_nearest(tactical, data.pitch_m(tactical.get("game")))
    none = {"frame": np.zeros(0, dtype=np.int64), "team": np.zeros(0, dtype=object),
            "confirmed": np.zeros(0, dtype=bool)}
    rules =possession.kickoff_decisions(kickoffs, pd.DataFrame(none), cfg, mark)
    rows = []
    for half in sorted(rules):
        team, rule, dist = rules[half]
        frame = int(kickoffs[half])
        rows.append({"kind": "half {} kickoff".format(half), "frame": frame, "rule": rule,
                     "team": team, "mark_dist_m": _finite(dist), "ours": LETTER[int(pred[frame])],
                     "goals_end": None, "goals_team": None})
    stoppage = np.asarray(pred) == D
    edges = np.flatnonzero(stoppage[:-1] & np.isin(np.asarray(pred)[1:], (H, A))) + 1
    restart_list = [(int(f), LETTER[int(pred[f])]) for f in edges]
    goals = {}
    if tags is not None and len(tags):
        rows_g = tags[tags["code"].astype(str).str.upper() == possession.GOALS_CODE]
        goals = dict((int(r.end_frame), r.team) for r in rows_g.itertuples(index=False))
    for frame, team, rule, dist in possession.goal_kickoffs(tags, stoppage, restart_list, blocks,
                                                            cfg, mark):
        end = max(e for e in goals if e <= frame)
        rows.append({"kind": "after goal", "frame": int(frame), "rule": rule, "team": team,
                     "mark_dist_m": _finite(dist), "ours": LETTER[int(pred[frame])],
                     "goals_end": end, "goals_team": goals[end]})
    return rows


def format_kickoffs(res):
    lines = ["", "KICKOFFS (full match; engine rules: centre mark within KICKOFF_TAKER_M over "
                 "KICKOFF_SEARCH_FRAMES, then Law 8 for the other half)"]
    if not res.get("kickoffs"):
        lines.append("  engine kickoff rules not available")
    for row in res.get("kickoffs", []):
        goal = "" if row["goals_end"] is None else "   GOALS window ends {} (team {})".format(
            row["goals_end"], row["goals_team"])
        dist = "-" if row["mark_dist_m"] is None else "{:.2f}".format(row["mark_dist_m"])
        lines.append("  {:<16} frame {:>6}   rule {:<7} team {}   nearest to the mark {} m   "
                     "ours {}{}".format(row["kind"], row["frame"], row["rule"], row["team"] or "-",
                                         dist, row["ours"], goal))
    return lines


# -- main analysis ---------------------------------------------------------------------------------

def analyse(results_path, window=DEFAULT_WINDOW, lookback=REAPPEAR_LOOKBACK, tactical=None,
            tags=None, pitch=None, reports_dir=cm.REPORTS_DIR, game=None):
    """Everything forensics prints, as a dict (also what --json writes).

    Raises FileNotFoundError(hint) if the scorer's reports for this results file do not exist.
    `game` is the match of games.py. None means games.GAME. Only a match with a public ground
    truth in evaluation/gt/ can have scorer reports.
    """
    from engine import data
    game = cm.game_name(game)
    name = cm.result_name(results_path)
    json_path = os.path.join(reports_dir, name + ".json")
    csv_path = os.path.join(reports_dir, name + "_transitions.csv")
    if not (os.path.isfile(json_path) and os.path.isfile(csv_path)):
        raise FileNotFoundError(
            "no scorer reports for {0}: expected\n  {1}\n  {2}\nProduce them with\n"
            "  python evaluation/run_evaluation.py\n(the file must be in evaluation/results/)"
            .format(name, json_path, csv_path))
    with open(json_path) as handle:
        report = json.load(handle)
    tol = max(int(t) for t in report["taus"])
    rows = read_transitions_csv(csv_path)

    # The engine reads the ATD with the C9 goalkeeper slot-swap guard applied. The traces and
    # the events here use the same data, so they show what the engine saw.
    if tactical is None:
        from engine import possession
        tactical = possession.engine_tactical(game=game)
    tags = tags if tags is not None else data.load_tagging(game)
    pitch = pitch or data.pitch_m(game)
    ball_missing = np.isnan(tactical["bx"])

    gt = public_gt_codes(game)
    runs = pm.valid_runs(gt)
    pred = cm.read_results(results_path, game)
    filled = pm.fill_abstentions(pred, runs)
    gt_trans = pm.transitions(gt, runs)
    pred_trans = pm.transitions(filled, runs)
    pairs, missed, spurious = pm.match_transitions(gt_trans, pred_trans, tol)

    stale = (len(pairs), len(missed), len(spurious)) != (
        report["transitions"][str(tol)]["n_matched"], report["transitions"][str(tol)]["n_missed"],
        report["transitions"][str(tol)]["n_spurious"])
    csv_status = {r["gt_frame"]: r for r in rows if r["gt_frame"] is not None}
    mismatch = [g[0] for g in gt_trans if csv_status.get(g[0], {}).get("status") !=
                ("matched" if g[0] in {p[0][0] for p in pairs} else "missed")]
    pair_of = {g[0]: p for g, p in pairs}

    events_mod = cm.optional_module("events")
    out = {"name": name, "tol": tol, "window": window, "reports_stale": bool(stale or mismatch),
           "events_module": events_mod is not None, "reappear_lookback": lookback,
           "gt_segments": [(int(a), int(b), LETTER[k]) for a, b, k in pm.segments(gt, runs)],
           "gt_transitions": [], "spurious": []}

    for g in gt_trans:
        frame = g[0]
        near = nearest_same_destination(g, pred_trans)
        matched = frame in pair_of
        pred_tr = pair_of[frame] if matched else near
        kind = kind_of(g[1], g[2], ball_missing, frame, lookback)
        rec = {"gt_frame": frame, "from": LETTER[g[1]], "to": LETTER[g[2]], "kind": kind,
               "status": "matched" if matched else "missed",
               "ball_missing_at_f_minus_1": bool(ball_missing[frame - 1]),
               "our_frame": None if pred_tr is None else pred_tr[0],
               "our_from_to": None if pred_tr is None else
               "{}->{}".format(LETTER[pred_tr[1]], LETTER[pred_tr[2]]),
               "latency": None if pred_tr is None else pred_tr[0] - frame,
               "trace": build_trace(tactical, pitch, gt, filled, frame, window),
               "tags": tag_overlaps(tags, frame - window, frame + window + 1),
               "events": events_in_window(events_mod, tactical, frame - window, frame + window + 1)
               if events_mod is not None else []}
        out["gt_transitions"].append(rec)

    for p in spurious:
        gaps = [(abs(g[0] - p[0]), g) for g in gt_trans]
        dist, g_near = min(gaps, key=lambda item: item[0]) if gaps else (None, None)
        out["spurious"].append({
            "our_frame": p[0], "from": LETTER[p[1]], "to": LETTER[p[2]],
            "gt_label_at_frame": LETTER[int(gt[p[0]])],
            "nearest_gt_transition": None if g_near is None else
            {"frame": g_near[0], "from_to": "{}->{}".format(LETTER[g_near[1]], LETTER[g_near[2]]),
             "distance": int(dist)},
            "trace": build_trace(tactical, pitch, gt, filled, p[0], window),
            "tags": tag_overlaps(tags, p[0] - window, p[0] + window + 1),
            "events": events_in_window(events_mod, tactical, p[0] - window, p[0] + window + 1)
            if events_mod is not None else [],
            "evidence": None})  # filled by the engine's deadball/events once they expose terms

    out["sandwiches"] = sandwich_check(gt, filled, runs)
    out["kickoffs"] = kickoff_check(pred, tactical, tags, data.graded_blocks(game),
                                    data.kickoff_frames(game))
    out["latency_by_kind"] = _latency_by_kind(out["gt_transitions"])
    out["summary"] = _summary(report)
    return out


def _latency_by_kind(gt_records):
    """Per kind: n GT, n matched, matched latencies' mean/median, and nearest-candidate latency."""
    result = {}
    for kind in KINDS:
        recs = [r for r in gt_records if r["kind"] == kind]
        matched = [r["latency"] for r in recs if r["status"] == "matched"]
        nearest = [r["latency"] for r in recs if r["latency"] is not None]
        result[kind] = {
            "n_gt": len(recs), "n_matched": len(matched),
            "matched_mean": float(np.mean(matched)) if matched else None,
            "matched_median": float(np.median(matched)) if matched else None,
            "nearest_n": len(nearest),
            "nearest_mean": float(np.mean(nearest)) if nearest else None,
            "nearest_median": float(np.median(nearest)) if nearest else None,
            "nearest_latencies": nearest}
    return result


def _summary(report):
    trans = report["transitions"]
    return {
        "spell_f1_mean": report["spell_f1_mean"],
        "macro_f1": report["macro_f1"], "f1_D": report["per_class"]["D"]["f1"],
        "fragmentation": report["fragmentation"],
        "n_gt_spells": report["n_gt_spells"], "n_pred_spells": report["n_pred_spells"],
        "per_iou": {k: {kk: v[kk] for kk in ("n_matched", "n_missed", "n_spurious", "f1")}
                    for k, v in report["spell_f1"].items()},
        "transitions": {k: {kk: v[kk] for kk in ("n_gt", "n_pred", "n_matched", "n_missed",
                                                 "n_spurious", "mean_latency", "median_latency",
                                                 "mean_abs_latency", "f1")}
                        for k, v in trans.items()}}


# -- printing --------------------------------------------------------------------------------------

def format_summary(res):
    s = res["summary"]
    lines = ["FORENSICS  {}   (public window; scorer tolerance tau={}; trace window +-{})".format(
        res["name"], res["tol"], res["window"])]
    if res["reports_stale"]:
        lines.append("!! the scorer reports disagree with a recomputation: re-run "
                     "python evaluation/run_evaluation.py")
    lines.append("events: " + ("engine.events present" if res["events_module"]
                               else "module not present"))
    lines.append("spellF1 {:.4f}   macroF1 {:.4f}   f1_D {:.4f} (a macroF1 quantity, not a "
                 "target: PLAN 5.1)   fragmentation {:.2f}   spells {} (GT {})".format(
                     s["spell_f1_mean"], s["macro_f1"], s["f1_D"], s["fragmentation"],
                     s["n_pred_spells"], s["n_gt_spells"]))
    lines.append("per-IoU spells:  " + "   ".join(
        "{}: {} matched / {} missed / {} spurious".format(k, v["n_matched"], v["n_missed"],
                                                          v["n_spurious"])
        for k, v in sorted(s["per_iou"].items())))
    for k, v in sorted(s["transitions"].items(), key=lambda item: int(item[0])):
        lines.append("transitions tol {:>2}: matched {} / missed {} / spurious {} (GT {}, ours {})"
                     "   mean_latency {}  median {}".format(
                         k, v["n_matched"], v["n_missed"], v["n_spurious"], v["n_gt"],
                         v["n_pred"], _lat(v["mean_latency"]), _lat(v["median_latency"])))
    return lines


def _lat(value):
    return "-" if value is None else "{:+.1f}".format(value)


def format_kinds(res):
    lines = ["", "LATENCY BY KIND (frames, ours - GT; D->alive split by the ball: reappear = ball "
                 "missing in the {} frames before the GT restart)".format(res["reappear_lookback"]),
             "  {:<22}{:>5}{:>9}{:>18}{:>28}".format(
                 "kind", "GT", "matched", "matched mean/med", "nearest (any dist) mean/med")]
    for kind, v in res["latency_by_kind"].items():
        lines.append("  {:<22}{:>5}{:>9}{:>18}{:>28}   {}".format(
            kind, v["n_gt"], v["n_matched"],
            "{}/{}".format(_lat(v["matched_mean"]), _lat(v["matched_median"])),
            "{}/{}".format(_lat(v["nearest_mean"]), _lat(v["nearest_median"])),
            "n=%d" % v["nearest_n"]))
    return lines


def format_sandwich(res):
    lines = ["", "SANDWICH CHECK (GT `T D T`: does our output keep the two spells apart?)"]
    if not res["sandwiches"]:
        lines.append("  none in the GT")
    for sw in res["sandwiches"]:
        lines.append("  {t} D {t} at {a}-{b}: {verdict}   our labels inside the GT D: H {H} / A {A} "
                     "/ D {D} frames".format(
                         t=sw["team"], a=sw["gt_d_start"], b=sw["gt_d_stop"],
                         verdict="SEPARATE" if sw["separate"] else "MERGED",
                         **sw["our_frames"]))
    return lines


def format_gt_record(rec, window):
    lines = ["", "=" * 96,
             "GT {}->{} at {}  [{}]  {}   ball missing at f-1: {}".format(
                 rec["from"], rec["to"], rec["gt_frame"], rec["kind"], rec["status"].upper(),
                 "yes" if rec["ball_missing_at_f_minus_1"] else "no")]
    if rec["our_frame"] is None:
        lines.append("   our nearest same-destination transition: none")
    else:
        lines.append("   our nearest same-destination transition: {} at {} (latency {:+d} "
                     "frames)".format(rec["our_from_to"], rec["our_frame"], rec["latency"]))
    lines += format_trace(rec["trace"], rec["gt_frame"])
    lines += format_tags(rec["tags"], rec["gt_frame"])
    if rec["events"]:
        lines.append("   events: " + "; ".join("{} {} {}".format(
            e.get("frame"), e.get("kind"), e.get("team")) for e in rec["events"]))
    return lines


def format_spurious_record(rec):
    near = rec["nearest_gt_transition"]
    lines = ["", "=" * 96,
             "SPURIOUS {}->{} at {}   GT label there: {}   nearest GT transition: {}".format(
                 rec["from"], rec["to"], rec["our_frame"], rec["gt_label_at_frame"],
                 "none" if near is None else "{} at {} ({:+d} frames away)".format(
                     near["from_to"], near["frame"], near["frame"] - rec["our_frame"]))]
    lines += format_trace(rec["trace"], rec["our_frame"])
    lines += format_tags(rec["tags"], rec["our_frame"])
    if rec["events"]:
        lines.append("   events: " + "; ".join("{} {} {}".format(
            e.get("frame"), e.get("kind"), e.get("team")) for e in rec["events"]))
    lines.append("   evidence term: n/a (engine evidence terms not exposed)")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results", help="results CSV (frame,label), scored by run_evaluation.py")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help="trace half-width in frames (default %(default)s)")
    parser.add_argument("--only", choices=("missed", "spurious", "all"), default="all")
    parser.add_argument("--json", action="store_true",
                        help="also write evaluation/reports/<name>_forensics.json")
    parser.add_argument("--reappear-lookback", type=int, default=REAPPEAR_LOOKBACK,
                        help="frames before a GT restart searched for a missing ball "
                             "(1 = the card's literal rule); default %(default)s")
    parser.add_argument("--game", default=None, choices=sorted(cm.games.GAMES),
                        help="match of games.py (default: games.GAME)")
    args = parser.parse_args(argv)

    try:
        res = analyse(args.results, args.window, args.reappear_lookback, game=args.game)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    lines = format_summary(res) + format_kinds(res) + format_sandwich(res) + format_kickoffs(res)
    if args.only in ("missed", "all"):
        shown = [r for r in res["gt_transitions"] if args.only == "all" or r["status"] == "missed"]
        lines.append("\n#### GT transitions ({} shown of {})".format(
            len(shown), len(res["gt_transitions"])))
        for rec in shown:
            lines += format_gt_record(rec, args.window)
    if args.only in ("spurious", "all"):
        lines.append("\n#### Spurious transitions of ours ({})".format(len(res["spurious"])))
        for rec in res["spurious"]:
            lines += format_spurious_record(rec)
    print("\n".join(lines))

    if args.json:
        os.makedirs(cm.REPORTS_DIR, exist_ok=True)
        path = os.path.join(cm.REPORTS_DIR, res["name"] + "_forensics.json")
        with open(path, "w") as handle:
            json.dump(res, handle, indent=1, default=_jsonable)
        print("\nWrote {}".format(path))
    return 0


def _jsonable(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value))


if __name__ == "__main__":
    sys.exit(main())
