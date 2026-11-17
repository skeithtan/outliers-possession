"""R2-f: exact spellF1 decomposition on the public window (research direction F).

    python tools/analysis/r2_metric_decomp.py [--out DIR] [names ...]

For each result file in evaluation/results/ (default c5_dribble m2_events m3_decoder
baseline_smooth50):

1. Spell table. Every public GT spell with its best same-team predicted spell, the IoU, the
   thresholds it passes under the scorer's own greedy matching, and a failure class:
   ok / boundary / merged / split / consumed / missed-by-other-team / missed-by-D / wrong-team.
   Frame cost: boundary frames to correct (deficit first, the cheapest order) to reach IoU 0.5
   and 0.7 with the same predicted spell.
2. Transition alignment. GT and predicted transitions aligned by an order-preserving DP:
   a pair needs the same destination label and |latency| <= LMAX, costs |latency|; an unpaired
   transition costs PENALTY. Paired = boundary error, unpaired GT = missed, unpaired pred =
   spurious. This is a convention, stated in FINDINGS.md.
3. Marginal value. Exact spellF1 gain of each single edit of the predicted array (move one
   boundary to the GT frame, insert one missed GT segment, remove one spurious predicted
   segment, repair one GT segment), and of each edit class applied together.

Reads only the public GT, through tools/gt.py. Never opens a full-match GT file.
"""
import argparse
import csv
import json
import os
import tempfile
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from gt import public_gt_codes  # noqa: E402
import possession_metrics as pm  # noqa: E402

LET = "HADX"
LMAX = 300       # frames: largest latency still read as "the same transition, late"
PENALTY = 100    # frames: cost of leaving a transition unpaired (pairs beyond 2*PENALTY split)
DEFAULT_NAMES = ("c5_dribble", "m2_events", "m3_decoder", "baseline_smooth50")
DEFAULT_OUT = os.path.join(tempfile.gettempdir(), "r2_research", "f_metric")


def load_pred(name, runs):
    path = os.path.join(ROOT, "evaluation", "results", name + ".csv")
    codes = pm.label_codes(pm.read_label_csv(path))
    return pm.fill_abstentions(codes, runs)


def greedy_pairs(overlaps, threshold):
    """{gt_index: pred_index} exactly as pm.matched_gt_flags matches them."""
    out, used = {}, set()
    for iou, gi, pj in overlaps:
        if iou < threshold:
            break
        if gi in out or pj in used:
            continue
        out[gi] = pj
        used.add(pj)
    return out


def iou(a, b):
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    return inter / float(max(a[1], b[1]) - min(a[0], b[0]))


def frames_to_pass(g, p, threshold):
    """Fewest boundary frames to correct so that IoU(g, p') >= threshold. Deficit frames first
    (each adds 1 to the intersection, which is never worse than removing 1 from the union)."""
    inter = min(g[1], p[1]) - max(g[0], p[0])
    union = max(g[1], p[1]) - min(g[0], p[0])
    if inter <= 0:
        return None
    deficit = (g[1] - g[0]) - inter
    excess = (p[1] - p[0]) - inter
    n = 0
    while inter / float(union) < threshold:
        if deficit > 0:
            deficit -= 1
            inter += 1
        elif excess > 0:
            excess -= 1
            union -= 1
        else:
            break
        n += 1
    return n


def spell_table(gt, pred, runs):
    gsp = pm.possession_spells(gt, runs)
    psp = pm.possession_spells(pred, runs)
    psegs = pm.segments(pred, runs)
    ov = pm.spell_overlaps(gsp, psp)
    pairs = {t: greedy_pairs(ov, t) for t in pm.SPELL_IOUS}
    matched_pred = {t: set(pairs[t].values()) for t in pm.SPELL_IOUS}
    rows = []
    for gi, g in enumerate(gsp):
        same = [(iou(g, p), pj) for pj, p in enumerate(psp) if p[2] == g[2] and iou(g, p) > 0]
        same.sort(key=lambda x: (-x[0], x[1]))
        passes = [t for t in pm.SPELL_IOUS if gi in pairs[t]]
        row = {"gt_start": g[0], "gt_stop": g[1], "team": LET[g[2]], "len": g[1] - g[0],
               "passes": passes, "n_pass": len(passes)}
        cover = {0: 0, 1: 0, 2: 0}
        for s in psegs:
            inter = min(g[1], s[1]) - max(g[0], s[0])
            if inter > 0:
                cover[s[2]] = cover.get(s[2], 0) + inter
        row["cover_same"] = cover[g[2]] / float(row["len"])
        row["cover_other"] = cover[1 - g[2]] / float(row["len"])
        row["cover_D"] = cover[2] / float(row["len"])
        if not same:
            row.update(best_iou=0.0, pred=None, d_start=None, d_stop=None,
                       need_05=None, need_07=None)
            # team-blind: an other-team spell aligned with g is a wrong-team call
            other = [(iou(g, p), p) for p in psp if p[2] != g[2] and iou(g, p) > 0]
            other.sort(key=lambda x: -x[0])
            if other and other[0][0] >= 0.5:
                cls = "wrong-team"
            elif row["cover_D"] >= row["cover_other"]:
                cls = "missed: covered by D"
            else:
                cls = "missed: covered by other team"
            row["class"] = cls if len(passes) < 5 else "ok"
            rows.append(row)
            continue
        best, pj = same[0]
        p = psp[pj]
        row.update(best_iou=round(best, 3), pred=(p[0], p[1]), d_start=p[0] - g[0],
                   d_stop=p[1] - g[1], need_05=frames_to_pass(g, p, 0.5),
                   need_07=frames_to_pass(g, p, 0.7))
        merged_with = [gk for gk, h in enumerate(gsp) if gk != gi and h[2] == g[2]
                       and (min(h[1], p[1]) - max(h[0], p[0])) >= max(25, 0.3 * (h[1] - h[0]))]
        split = [q for b, q in same if (min(g[1], psp[q][1]) - max(g[0], psp[q][0]))
                 >= max(25, 0.15 * row["len"])]
        consumed = [t for t in pm.SPELL_IOUS if gi not in pairs[t] and best >= t
                    and pj in matched_pred[t]]
        row["merged_with"] = [(gsp[k][0], gsp[k][1]) for k in merged_with]
        row["n_split"] = len(split)
        row["consumed_at"] = consumed
        if len(passes) == 5:
            cls = "ok"
        elif consumed:
            cls = "consumed (best pred matched to a neighbour)"
        elif merged_with:
            cls = "merged"
        elif len(split) >= 2:
            cls = "split"
        else:
            a, b = row["d_start"], row["d_stop"]
            if abs(a) > 25 and abs(b) > 25 and a * b > 0:
                cls = "boundary (whole spell shifted {})".format("late" if a > 0 else "early")
            elif abs(a) > 25 and abs(b) > 25:
                cls = "boundary (both ends)"
            else:
                cls = "boundary ({} end)".format("start" if abs(a) >= abs(b) else "stop")
        row["class"] = cls
        rows.append(row)

    prow = []
    for pj, p in enumerate(psp):
        hit = [t for t in pm.SPELL_IOUS if pj in matched_pred[t]]
        if hit:
            continue
        gl = gt[p[0]:p[1]]
        frac = {LET[k]: float(np.mean(gl == k)) for k in (0, 1, 2)}
        same_frac = frac[LET[p[2]]]
        if same_frac < 0.2:
            cls = "spurious (inside GT {})".format(max(frac, key=frac.get))
        else:
            cls = "unmatched fragment ({:.0%} on same-team GT)".format(same_frac)
        prow.append({"start": p[0], "stop": p[1], "team": LET[p[2]], "len": p[1] - p[0],
                     "gt_frac": frac, "class": cls})
    return rows, prow


def align(gt_tr, pr_tr):
    """Order-preserving DP alignment. Returns (pairs, missed, spurious)."""
    n, m = len(gt_tr), len(pr_tr)
    inf = 10 ** 9
    cost = np.full((n + 1, m + 1), inf, dtype=np.int64)
    move = np.zeros((n + 1, m + 1), dtype=np.int8)
    cost[0, 0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            c = cost[i, j]
            if c >= inf:
                continue
            if i < n and c + PENALTY < cost[i + 1, j]:
                cost[i + 1, j] = c + PENALTY
                move[i + 1, j] = 1
            if j < m and c + PENALTY < cost[i, j + 1]:
                cost[i, j + 1] = c + PENALTY
                move[i, j + 1] = 2
            if i < n and j < m and gt_tr[i][2] == pr_tr[j][2]:
                d = abs(pr_tr[j][0] - gt_tr[i][0])
                if d <= LMAX and c + d < cost[i + 1, j + 1]:
                    cost[i + 1, j + 1] = c + d
                    move[i + 1, j + 1] = 3
    pairs, missed, spurious = [], [], []
    i, j = n, m
    while i or j:
        mv = move[i, j]
        if mv == 3:
            pairs.append((gt_tr[i - 1], pr_tr[j - 1]))
            i, j = i - 1, j - 1
        elif mv == 1:
            missed.append(gt_tr[i - 1])
            i -= 1
        else:
            spurious.append(pr_tr[j - 1])
            j -= 1
    return pairs[::-1], missed[::-1], spurious[::-1]


def kind(tr):
    if tr[2] == pm.D:
        return "onset"
    if tr[1] == pm.D:
        return "restart"
    return "H<->A"


def score(gt, pred):
    return pm.evaluate(gt, pred)["spell_f1_mean"]


def move_boundary(pred, g, q):
    out = pred.copy()
    if q > g:
        out[g:q] = pred[q]
    elif q < g:
        out[q:g] = pred[q - 1]
    return out


def next_frame(trs, frame, stop):
    later = [t[0] for t in trs if t[0] > frame]
    return min(later) if later else stop


def edits(gt, pred, runs, spurious_spells):
    gt_tr = pm.transitions(gt, runs)
    pr_tr = pm.transitions(pred, runs)
    pairs, missed, spurious = align(gt_tr, pr_tr)
    stop = runs[-1][1]
    base = score(gt, pred)
    items = []
    all_b, all_ins, all_rem = pred.copy(), pred.copy(), pred.copy()
    for g, q in pairs:
        if g[0] == q[0]:
            continue
        new = move_boundary(pred, g[0], q[0])
        items.append({"edit": "move {} boundary {}{}@{} by {:+d} to GT".format(
            kind(g), LET[g[1]], LET[g[2]], g[0], g[0] - q[0]), "class": "boundary-" + kind(g),
            "latency": q[0] - g[0], "gain": score(gt, new) - base})
        all_b = move_boundary(all_b, g[0], q[0])
    for g in missed:
        e = next_frame(gt_tr, g[0], stop)
        new = pred.copy()
        new[g[0]:e] = gt[g[0]:e]
        items.append({"edit": "insert missed GT {} segment {}-{} ({}f, {})".format(
            LET[g[2]], g[0], e, e - g[0], kind(g)), "class": "missed-" + kind(g),
            "gain": score(gt, new) - base})
        all_ins[g[0]:e] = gt[g[0]:e]
    # Spurious predicted spells are spell-centric: unmatched at every IoU and < 20 % of their
    # frames on same-team GT. Removing one sets it to the majority GT label it sits in.
    for p in spurious_spells:
        host = int(np.bincount(gt[p["start"]:p["stop"]], minlength=4)[:3].argmax())
        new = pred.copy()
        new[p["start"]:p["stop"]] = host
        items.append({"edit": "remove spurious pred {} spell {}-{} ({}f, inside GT {})".format(
            p["team"], p["start"], p["stop"], p["len"], LET[host]), "class": "spurious-spell",
            "gain": score(gt, new) - base})
        all_rem[p["start"]:p["stop"]] = host
    items.sort(key=lambda d: -d["gain"])
    seg_items = []
    for s in pm.segments(gt, runs):
        new = pred.copy()
        new[s[0]:s[1]] = gt[s[0]:s[1]]
        seg_items.append({"segment": "{}-{} {} ({}f)".format(s[0], s[1], LET[s[2]], s[1] - s[0]),
                          "gain": score(gt, new) - base})
    seg_items.sort(key=lambda d: -d["gain"])
    both = all_b.copy()
    for g in missed:
        e = next_frame(gt_tr, g[0], stop)
        both[g[0]:e] = gt[g[0]:e]
    groups = {
        "all boundary moves (no insertions, no removals)": score(gt, all_b) - base,
        "all missed-transition insertions (no boundary moves)": score(gt, all_ins) - base,
        "all spurious-spell removals": score(gt, all_rem) - base,
        "boundaries + insertions": score(gt, both) - base,
    }
    for lo, hi in ((1, 25), (26, 50), (51, LMAX)):
        arr = pred.copy()
        for g, q in pairs:
            if lo <= abs(q[0] - g[0]) <= hi:
                arr = move_boundary(arr, g[0], q[0])
        groups["boundary moves, |latency| {}-{}".format(lo, hi)] = score(gt, arr) - base
    for k in ("H<->A", "onset", "restart"):
        arr = pred.copy()
        for g, q in pairs:
            if kind(g) == k and g[0] != q[0]:
                arr = move_boundary(arr, g[0], q[0])
        groups["boundary moves, {} only".format(k)] = score(gt, arr) - base
    return {"base": base, "pairs": [(g, q, q[0] - g[0], kind(g)) for g, q in pairs],
            "missed": missed, "spurious": spurious, "items": items, "segments": seg_items,
            "groups": groups}


def fmt_spell_rows(rows):
    out = ["| GT spell | team | len | best pred | IoU | Δstart | Δstop | passes | class |"
           " frames to 0.5 | frames to 0.7 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append("| {}-{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["gt_start"], r["gt_stop"], r["team"], r["len"],
            "{}-{}".format(*r["pred"]) if r["pred"] else "-", r["best_iou"],
            "" if r["d_start"] is None else "{:+d}".format(r["d_start"]),
            "" if r["d_stop"] is None else "{:+d}".format(r["d_stop"]),
            ",".join(str(t) for t in r["passes"]) or "none", r["class"],
            "" if r["need_05"] is None else r["need_05"],
            "" if r["need_07"] is None else r["need_07"]))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(DEFAULT_NAMES))
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    gt = public_gt_codes()
    runs = pm.valid_runs(gt)
    report = {}
    md = []
    for name in args.names:
        pred = load_pred(name, runs)
        res = pm.evaluate(gt, pred)
        rows, prow = spell_table(gt, pred, runs)
        ed = edits(gt, pred, runs, [p for p in prow if p["class"].startswith("spurious")])
        counts = {}
        for r in rows:
            c = r["class"].split(" (")[0].split(":")[0]
            counts[c] = counts.get(c, 0) + 1
        report[name] = {"spellF1": res["spell_f1_mean"],
                        "matched": [res["spell_f1"][t]["n_matched"] for t in pm.SPELL_IOUS],
                        "n_pred": res["n_pred_spells"], "rows": rows, "pred_unmatched": prow,
                        "class_counts": counts, "edits": ed}
        md.append("## {}  spellF1 {:.4f}  matched {} of 17  n_pred {}".format(
            name, res["spell_f1_mean"], report[name]["matched"], res["n_pred_spells"]))
        md.append(fmt_spell_rows(rows))
        md.append("\nclass counts: " + json.dumps(counts))
        md.append("\npredicted spells unmatched at every IoU:")
        for p in prow:
            md.append("- {}-{} {} {}f: {}  GT frac {}".format(
                p["start"], p["stop"], p["team"], p["len"], p["class"],
                {k: round(v, 2) for k, v in p["gt_frac"].items()}))
        md.append("\ntransition alignment (LMAX {}, PENALTY {}):".format(LMAX, PENALTY))
        for g, q, lat, k in ed["pairs"]:
            md.append("- paired {} {}{}@{} -> pred @{} latency {:+d}".format(
                k, LET[g[1]], LET[g[2]], g[0], q[0], lat))
        for g in ed["missed"]:
            md.append("- MISSED {} {}{}@{}".format(kind(g), LET[g[1]], LET[g[2]], g[0]))
        for q in ed["spurious"]:
            md.append("- SPURIOUS {} {}{}@{}".format(kind(q), LET[q[1]], LET[q[2]], q[0]))
        md.append("\nmarginal value, single edits (sorted):\n\n| edit | class | gain |\n|---|---|---|")
        for it in ed["items"]:
            md.append("| {} | {} | {:+.4f} |".format(it["edit"], it["class"], it["gain"]))
        md.append("\ngroup edits:\n\n| group | gain |\n|---|---|")
        for k, v in ed["groups"].items():
            md.append("| {} | {:+.4f} |".format(k, v))
        md.append("\nrepair one GT segment (pred[seg] = GT), top 10:\n\n| segment | gain |\n|---|---|")
        for it in ed["segments"][:10]:
            md.append("| {} | {:+.4f} |".format(it["segment"], it["gain"]))
        md.append("")
    with open(os.path.join(args.out, "decomp.md"), "w") as fh:
        fh.write("\n".join(md))
    with open(os.path.join(args.out, "decomp.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=lambda o: o if not isinstance(o, np.integer)
                  else int(o))
    print("\n".join(md))


if __name__ == "__main__":
    main()
