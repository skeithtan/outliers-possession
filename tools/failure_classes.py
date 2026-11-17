"""Failure-class reports for one results file against one reference (P9-D). Not part of the submission.

tools/score_game.py calls these functions and prints their tables. Each function takes code
vectors (H=0 A=1 D=2 X=3) of equal length and the graded runs of the reference, as
possession_metrics.valid_runs() gives them. A predicted X is forward-filled inside each run
first, as the scorer does for its temporal metrics.

    spell_recall_by_duration()   recall of reference spells by duration bucket
    possession_ratio()           predicted possessions over reference possessions
    dead_edges()                 D entry and exit precision, recall and signed latency
    decompose()                  the per-class decomposition of RESEARCH-2 M19

Definitions, the same in every report:

- A reference spell is recalled when the scorer's own one-to-one greedy spell matching
  (possession_metrics.matched_gt_flags) matches it at IoU RECALL_IOU or more. PLAN 1.3: success
  on a short spell means matched at IoU 0.5, not merely present.
- A D entry is a change from a live label to D. A D exit is a change from D to a live label.
  H and A are one live state for this table. Edges match one to one within the loosest
  tolerance of possession_metrics.TAUS, with possession_metrics.match_transitions(). The
  latency is the predicted frame minus the reference frame.

Deterministic. numpy only. Python 3.9. The module never opens a file.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALUATION = os.path.join(ROOT, "evaluation")
if EVALUATION not in sys.path:
    sys.path.insert(0, EVALUATION)
import possession_metrics as pm  # noqa: E402

LETTER = "HADX"
# IoU at which a reference spell counts as recalled (PLAN 1.3, R-F10).
RECALL_IOU = 0.5
# Duration bucket edges in seconds: < 2 s, 2 to 5 s, > 5 s (IDEAS-REVIEW 3 item 4).
DURATION_EDGES_S = (2.0, 5.0)
# The live state of the D edge table: H and A collapsed into one code.
LIVE = pm.H
# M19 alignment (tools/analysis/r2_metric_decomp.py): the largest latency still read as the
# same transition late, and the cost of a transition left unpaired, both in frames.
ALIGN_MAX_LATENCY = 300
ALIGN_PENALTY = 100
# M19 latency bands of the group edits, in frames.
LATENCY_BANDS = ((1, 25), (26, 50), (51, ALIGN_MAX_LATENCY))
KINDS = ("H<->A", "onset", "restart")


def _filled(pred, runs):
    return pm.fill_abstentions(np.asarray(pred, dtype=np.int8), runs)


# -- short-spell recall ------------------------------------------------------------------------

def bucket_names(edges=DURATION_EDGES_S):
    """The bucket names for these edges: ['< 2 s', '2-5 s', '> 5 s'] for the default."""
    lo, hi = edges
    return ["< {:g} s".format(lo), "{:g}-{:g} s".format(lo, hi), "> {:g} s".format(hi)]


def spell_recall_by_duration(gt, pred, runs, fps=pm.FPS, iou=RECALL_IOU,
                             edges=DURATION_EDGES_S):
    """[{bucket, n, recalled, recall}] over the reference spells, by duration.

    A spell of d seconds is in the first bucket if d < edges[0], in the second if
    edges[0] <= d <= edges[1], else in the third. "recall" is None for an empty bucket.
    """
    gt = np.asarray(gt, dtype=np.int8)
    gt_spells = pm.possession_spells(gt, runs)
    pred_spells = pm.possession_spells(_filled(pred, runs), runs)
    flags = pm.matched_gt_flags(pm.spell_overlaps(gt_spells, pred_spells), len(gt_spells), iou)
    seconds = np.array([(b - a) / float(fps) for a, b, _ in gt_spells])
    lo, hi = edges
    masks = [seconds < lo, (seconds >= lo) & (seconds <= hi), seconds > hi]
    out = []
    for name, mask in zip(bucket_names(edges), masks):
        n, hit = int(mask.sum()), int(flags[mask].sum()) if len(flags) else 0
        out.append({"bucket": name, "n": n, "recalled": hit,
                    "recall": hit / float(n) if n else None})
    return out


def possession_ratio(gt_possessions, pred_possessions):
    """Predicted possessions over reference possessions, None without a reference spell."""
    return pred_possessions / float(gt_possessions) if gt_possessions else None


# -- D entry and exit --------------------------------------------------------------------------

def live_dead(codes):
    """Copy of codes with H and A collapsed to LIVE. D and X stay as they are."""
    codes = np.array(codes, dtype=np.int8)
    codes[codes == pm.A] = LIVE
    return codes


def dead_edges(gt, pred, runs, tol=max(pm.TAUS)):
    """{"entry": {...}, "exit": {...}}: one-to-one matching of the D edges within tol frames.

    Each value holds n_ref, n_pred, matched, precision, recall and the signed latency of the
    matched pairs (mean, median and mean absolute, None without a pair).
    """
    gt_trans = pm.transitions(live_dead(gt), runs)
    pred_trans = pm.transitions(live_dead(_filled(pred, runs)), runs)
    out = {}
    for name, dest in (("entry", pm.D), ("exit", LIVE)):
        ref = [t for t in gt_trans if t[2] == dest]
        got = [t for t in pred_trans if t[2] == dest]
        score = pm.transition_scores(ref, got, tol)
        out[name] = {"n_ref": score["n_gt"], "n_pred": score["n_pred"],
                     "matched": score["n_matched"], "precision": score["precision"],
                     "recall": score["recall"], "mean_latency": score["mean_latency"],
                     "median_latency": score["median_latency"],
                     "mean_abs_latency": score["mean_abs_latency"]}
    out["tol"] = int(tol)
    return out


# -- M19 decomposition -------------------------------------------------------------------------

def _greedy_pairs(overlaps, threshold):
    """{gt_index: pred_index}, matched as possession_metrics.matched_gt_flags matches them."""
    out, used = {}, set()
    for iou, gi, pj in overlaps:
        if iou < threshold:
            break
        if gi in out or pj in used:
            continue
        out[gi] = pj
        used.add(pj)
    return out


def _iou(a, b):
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    return inter / float(max(a[1], b[1]) - min(a[0], b[0]))


def frames_to_pass(g, p, threshold):
    """Fewest boundary frames to correct so that IoU(g, p) reaches threshold.

    The function corrects the missing frames first, then the excess frames. A missing frame
    adds one to the intersection, which is never worse than one frame less of union.
    None if g and p do not overlap.
    """
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
    """(rows, unmatched): the M19 spell table and the predicted spells unmatched at every IoU.

    `pred` is already filled. Each row is one reference spell with its best same-team
    predicted spell, the IoU, the edge differences, the thresholds it passes and a class:
    ok, consumed, merged, split, boundary, wrong-team, "missed: covered by D" or
    "missed: covered by other team" (RESEARCH-2 M19).
    """
    gsp = pm.possession_spells(gt, runs)
    psp = pm.possession_spells(pred, runs)
    psegs = pm.segments(pred, runs)
    ov = pm.spell_overlaps(gsp, psp)
    pairs = {t: _greedy_pairs(ov, t) for t in pm.SPELL_IOUS}
    matched_pred = {t: set(pairs[t].values()) for t in pm.SPELL_IOUS}
    rows = []
    for gi, g in enumerate(gsp):
        same = [(_iou(g, p), pj) for pj, p in enumerate(psp) if p[2] == g[2] and _iou(g, p) > 0]
        same.sort(key=lambda x: (-x[0], x[1]))
        passes = [t for t in pm.SPELL_IOUS if gi in pairs[t]]
        row = {"gt_start": int(g[0]), "gt_stop": int(g[1]), "team": LETTER[g[2]],
               "len": int(g[1] - g[0]), "passes": passes, "n_pass": len(passes)}
        cover = {pm.H: 0, pm.A: 0, pm.D: 0}
        for s in psegs:
            inter = min(g[1], s[1]) - max(g[0], s[0])
            if inter > 0 and s[2] in cover:
                cover[s[2]] += inter
        other_team = pm.A if g[2] == pm.H else pm.H
        row["cover_same"] = cover[g[2]] / float(row["len"])
        row["cover_other"] = cover[other_team] / float(row["len"])
        row["cover_D"] = cover[pm.D] / float(row["len"])
        if not same:
            row.update(best_iou=0.0, pred=None, d_start=None, d_stop=None,
                       need_05=None, need_07=None)
            other = sorted((_iou(g, p) for p in psp if p[2] != g[2] and _iou(g, p) > 0),
                           reverse=True)
            if other and other[0] >= 0.5:
                cls = "wrong-team"
            elif row["cover_D"] >= row["cover_other"]:
                cls = "missed: covered by D"
            else:
                cls = "missed: covered by other team"
            row["class"] = cls if len(passes) < len(pm.SPELL_IOUS) else "ok"
            rows.append(row)
            continue
        best, pj = same[0]
        p = psp[pj]
        row.update(best_iou=round(best, 3), pred=[int(p[0]), int(p[1])],
                   d_start=int(p[0] - g[0]), d_stop=int(p[1] - g[1]),
                   need_05=frames_to_pass(g, p, 0.5), need_07=frames_to_pass(g, p, 0.7))
        merged_with = [gk for gk, h in enumerate(gsp) if gk != gi and h[2] == g[2]
                       and (min(h[1], p[1]) - max(h[0], p[0])) >= max(25, 0.3 * (h[1] - h[0]))]
        split = [q for _, q in same if (min(g[1], psp[q][1]) - max(g[0], psp[q][0]))
                 >= max(25, 0.15 * row["len"])]
        consumed = [t for t in pm.SPELL_IOUS if gi not in pairs[t] and best >= t
                    and pj in matched_pred[t]]
        if len(passes) == len(pm.SPELL_IOUS):
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

    unmatched = []
    for pj, p in enumerate(psp):
        if any(pj in matched_pred[t] for t in pm.SPELL_IOUS):
            continue
        window = gt[p[0]:p[1]]
        frac = {LETTER[k]: float(np.mean(window == k)) for k in (pm.H, pm.A, pm.D)}
        same_frac = frac[LETTER[p[2]]]
        if same_frac < 0.2:
            cls = "spurious (inside GT {})".format(max(frac, key=frac.get))
        else:
            cls = "unmatched fragment ({:.0%} on same-team GT)".format(same_frac)
        unmatched.append({"start": int(p[0]), "stop": int(p[1]), "team": LETTER[p[2]],
                          "len": int(p[1] - p[0]), "gt_frac": frac, "class": cls})
    return rows, unmatched


def align(gt_tr, pr_tr, max_latency=ALIGN_MAX_LATENCY, penalty=ALIGN_PENALTY):
    """(pairs, missed, spurious): an order-preserving alignment of two transition lists.

    A pair needs the same destination label and |latency| <= max_latency, and costs
    |latency|. An unpaired transition costs `penalty`. The dynamic programme takes the
    alignment of least total cost, with fixed tie breaks (RESEARCH-2 M19).
    """
    n, m = len(gt_tr), len(pr_tr)
    inf = 10 ** 9
    cost = np.full((n + 1, m + 1), inf, dtype=np.int64)
    move = np.zeros((n + 1, m + 1), dtype=np.int8)
    cost[0, 0] = 0
    gt_frames = [int(t[0]) for t in gt_tr]
    pr_frames = [int(t[0]) for t in pr_tr]
    for i in range(n + 1):
        for j in range(m + 1):
            c = int(cost[i, j])
            if c >= inf:
                continue
            if i < n and c + penalty < cost[i + 1, j]:
                cost[i + 1, j] = c + penalty
                move[i + 1, j] = 1
            if j < m and c + penalty < cost[i, j + 1]:
                cost[i, j + 1] = c + penalty
                move[i, j + 1] = 2
            if i < n and j < m and gt_tr[i][2] == pr_tr[j][2]:
                d = abs(pr_frames[j] - gt_frames[i])
                if d <= max_latency and c + d < cost[i + 1, j + 1]:
                    cost[i + 1, j + 1] = c + d
                    move[i + 1, j + 1] = 3
    pairs, missed, spurious = [], [], []
    i, j = n, m
    while i or j:
        step = move[i, j]
        if step == 3:
            pairs.append((gt_tr[i - 1], pr_tr[j - 1]))
            i, j = i - 1, j - 1
        elif step == 1:
            missed.append(gt_tr[i - 1])
            i -= 1
        else:
            spurious.append(pr_tr[j - 1])
            j -= 1
    return pairs[::-1], missed[::-1], spurious[::-1]


def kind(transition):
    """'onset' for an entry into D, 'restart' for an exit from D, else 'H<->A'."""
    if transition[2] == pm.D:
        return "onset"
    if transition[1] == pm.D:
        return "restart"
    return "H<->A"


def _spell_f1(gt, pred, runs):
    return pm.spell_scores(gt, pred, runs)["mean_f1"]


def _move_boundary(pred, g, q):
    """Copy of pred with the predicted transition at q moved to the reference frame g."""
    out = pred.copy()
    if q > g:
        out[g:q] = pred[q]
    elif q < g:
        out[q:g] = pred[q - 1]
    return out


def _segment_end(gt_tr, frame, runs):
    """First reference transition after `frame` in the same run, else the end of that run."""
    stop = next((b for a, b in runs if a <= frame < b), runs[-1][1])
    later = [t[0] for t in gt_tr if frame < t[0] < stop]
    return min(later) if later else stop


def group_edits(gt, pred, runs, pairs, missed, spurious_spells):
    """{group: spellF1 gain} of the M19 edit classes, each applied to the whole prediction.

    The groups: all boundary moves, all missed-transition insertions, all spurious-spell
    removals, boundaries and insertions together, boundary moves by |latency| band and by
    transition kind. A boundary move sets the predicted transition to the reference frame.
    An insertion copies the reference from a missed transition to the next reference
    transition. A removal gives a spurious spell the majority reference label inside it.
    """
    gt_tr = pm.transitions(gt, runs)
    base = _spell_f1(gt, pred, runs)
    all_b, all_ins, all_rem = pred.copy(), pred.copy(), pred.copy()
    for g, q in pairs:
        all_b = _move_boundary(all_b, g[0], q[0])
    for g in missed:
        end = _segment_end(gt_tr, g[0], runs)
        all_ins[g[0]:end] = gt[g[0]:end]
    for p in spurious_spells:
        host = int(np.bincount(gt[p["start"]:p["stop"]], minlength=4)[:3].argmax())
        all_rem[p["start"]:p["stop"]] = host
    both = all_b.copy()
    for g in missed:
        end = _segment_end(gt_tr, g[0], runs)
        both[g[0]:end] = gt[g[0]:end]
    groups = {
        "all boundary moves": _spell_f1(gt, all_b, runs) - base,
        "all missed insertions": _spell_f1(gt, all_ins, runs) - base,
        "all spurious removals": _spell_f1(gt, all_rem, runs) - base,
        "boundaries + insertions": _spell_f1(gt, both, runs) - base,
    }
    for lo, hi in LATENCY_BANDS:
        arr = pred.copy()
        for g, q in pairs:
            if lo <= abs(q[0] - g[0]) <= hi:
                arr = _move_boundary(arr, g[0], q[0])
        groups["boundary moves |latency| {}-{}".format(lo, hi)] = _spell_f1(gt, arr, runs) - base
    for name in KINDS:
        arr = pred.copy()
        for g, q in pairs:
            if kind(g) == name and g[0] != q[0]:
                arr = _move_boundary(arr, g[0], q[0])
        groups["boundary moves {} only".format(name)] = _spell_f1(gt, arr, runs) - base
    return groups


def class_counts(rows):
    """{class: count} with the detail in brackets dropped, sorted by class name."""
    counts = {}
    for row in rows:
        name = row["class"].split(" (")[0]
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def decompose(gt, pred, runs):
    """The M19 decomposition of one prediction against one reference, as a dict.

    Keys: spellF1 ("base"), the reference spell classes ("spell_classes") and rows ("rows"),
    the unmatched predicted spells ("unmatched_classes", "unmatched"), the transition
    alignment counts by kind ("alignment"), and the group edit gains ("groups").
    """
    gt = np.asarray(gt, dtype=np.int8)
    pred = _filled(pred, runs)
    rows, unmatched = spell_table(gt, pred, runs)
    pairs, missed, spurious = align(pm.transitions(gt, runs), pm.transitions(pred, runs))
    alignment = {}
    for name in KINDS:
        alignment[name] = {
            "paired": sum(1 for g, _ in pairs if kind(g) == name),
            "missed": sum(1 for g in missed if kind(g) == name),
            "spurious": sum(1 for q in spurious if kind(q) == name),
        }
    spurious_spells = [p for p in unmatched if p["class"].startswith("spurious")]
    return {"base": _spell_f1(gt, pred, runs), "spell_classes": class_counts(rows),
            "rows": rows, "unmatched_classes": class_counts(unmatched), "unmatched": unmatched,
            "alignment": alignment,
            "groups": group_edits(gt, pred, runs, pairs, missed, spurious_spells)}


# -- printing ----------------------------------------------------------------------------------

def _pct(value):
    return "-" if value is None else "{:.1%}".format(value)


def _lat(value):
    return "-" if value is None else "{:+.1f}".format(value)


def format_lines(report):
    """Printable lines for the failure-class keys that score_game adds to its summary."""
    lines = ["  short-spell recall @IoU {:g}: {}".format(
        report["recall_iou"], "  ".join("{} {}/{} ({})".format(
            b["bucket"], b["recalled"], b["n"], _pct(b["recall"]))
            for b in report["short_spell_recall"]))]
    ratio = report["possession_ratio"]
    lines.append("  possession ratio results/reference {}".format(
        "-" if ratio is None else "{:.3f}".format(ratio)))
    edges = report["d_edges"]
    for name in ("entry", "exit"):
        e = edges[name]
        lines.append("  D {:<5} @{}: precision {} recall {}  matched {} of reference {}, "
                     "results {}  latency mean {} median {}".format(
                         name, edges["tol"], _pct(e["precision"]), _pct(e["recall"]),
                         e["matched"], e["n_ref"], e["n_pred"], _lat(e["mean_latency"]),
                         _lat(e["median_latency"])))
    dec = report["decomposition"]
    lines.append("  reference spells by class: " + ", ".join(
        "{} {}".format(k, v) for k, v in dec["spell_classes"].items()))
    lines.append("  unmatched results spells: " + (", ".join(
        "{} {}".format(k, v) for k, v in dec["unmatched_classes"].items()) or "none"))
    lines.append("  alignment: " + "; ".join(
        "{} paired {} missed {} spurious {}".format(k, v["paired"], v["missed"], v["spurious"])
        for k, v in dec["alignment"].items()))
    lines.append("  group edit gains: " + ", ".join(
        "{} {:+.3f}".format(k, v) for k, v in dec["groups"].items()))
    return lines
