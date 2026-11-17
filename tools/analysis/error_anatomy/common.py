"""Shared helpers for the error anatomy (OPEN external references only). Deterministic, numpy only.

Loading: load(view, game) -> dict with gt, ours (X forward-filled inside the reference runs, as
the scorer does), runs, the engine parts saved by run_parts.py, and the reference graded hours.

Owner changes: owner_changes(codes, runs) lists every change of team between consecutive alive
runs of one graded run (direct H<->A, or across a D stretch), and every same-team pair across D.

Matching: match_changes() is a one-to-one, order-preserving matching per destination team with
maximum cardinality first and then the least total |latency| (a DP), within +-tol frames.

Scoring: fast_spell_flags() / fast_spell_f1() reproduce possession_metrics.spell_scores exactly
(checked by check_fast_scoring()).
"""
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("ERROR_ANATOMY_WORK", os.path.join("/tmp", "error_anatomy"))  # outputs, never the repo
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (REPO, os.path.join(REPO, "tools"), os.path.join(REPO, "tools", "analysis"),
          os.path.join(REPO, "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)
import possession_metrics as pm  # noqa: E402

H, A, D, X = 0, 1, 2, 3
OPEN_GAMES = ("METRICA-S1", "METRICA-S2", "METRICA-S3",
              "DFL-J03WMX", "DFL-J03WN1", "DFL-J03WOH", "DFL-J03WOY", "DFL-J03WPY")
FORBIDDEN = ("BAR-ATH", "DFL-J03WQQ", "DFL-J03WR9")
IOUS = pm.SPELL_IOUS
FPS = 25.0
BIG = 10 ** 6


def group_of(game):
    return "Metrica" if game.startswith("METRICA-") else "DFL"


_REF_CACHE = {}


def ref_path(game):
    if game in FORBIDDEN or game not in OPEN_GAMES:
        raise SystemExit("refused: {}".format(game))
    if game.startswith("DFL-"):
        mid = game[len("DFL-"):]
        return os.path.expanduser("~/mtah2027/refs/dfl/{}/DFL-{}_reference.csv".format(mid, mid))
    return os.path.join(WORK, "refs", "{}_ref.csv".format(game))


def load_gt(game):
    if game not in _REF_CACHE:
        import score_game as sg
        _REF_CACHE[game] = np.asarray(sg.read_codes(ref_path(game)), dtype=np.int8)
    return _REF_CACHE[game]


def load(view, game):
    if game in FORBIDDEN or game not in OPEN_GAMES:
        raise SystemExit("refused: {}".format(game))
    gt = load_gt(game)
    z = np.load(os.path.join(WORK, "runs", view, game + ".npz"))
    arrays = {k: z[k] for k in z.files}
    with open(os.path.join(WORK, "runs", view, game + "_tables.pkl"), "rb") as handle:
        tables = pickle.load(handle)
    runs = pm.valid_runs(gt)
    raw = arrays["labels"].astype(np.int8)
    if raw.shape != gt.shape:
        raise SystemExit("{} {}: length mismatch".format(view, game))
    ours = pm.fill_abstentions(raw, runs)
    graded = int(sum(b - a for a, b in runs))
    out = {"view": view, "game": game, "group": group_of(game), "gt": gt, "ours": ours,
           "raw_labels": raw, "runs": runs, "graded": graded, "hours": graded / FPS / 3600.0}
    out.update(arrays)
    out.update(tables)
    return out


# -- runs and spells ---------------------------------------------------------------------------

def run_segments(codes, runs):
    """(starts, stops, labels, run_index) of the constant segments inside each graded run."""
    s_all, e_all, l_all, r_all = [], [], [], []
    for ri, (a, b) in enumerate(runs):
        w = codes[a:b]
        if not len(w):
            continue
        br = np.flatnonzero(w[1:] != w[:-1]) + 1
        st = np.concatenate(([0], br)) + a
        en = np.concatenate((br, [len(w)])) + a
        s_all.append(st)
        e_all.append(en)
        l_all.append(w[st - a])
        r_all.append(np.full(len(st), ri))
    if not s_all:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z.astype(np.int8), z
    return (np.concatenate(s_all).astype(np.int64), np.concatenate(e_all).astype(np.int64),
            np.concatenate(l_all).astype(np.int8), np.concatenate(r_all).astype(np.int64))


def spells(codes, runs):
    """(starts, stops, teams) of the possessions (maximal H or A runs) inside the graded runs."""
    s, e, l, _ = run_segments(codes, runs)
    keep = (l == H) | (l == A)
    return s[keep], e[keep], l[keep]


def fast_overlaps(gs, ge, gl, ps, pe, pl):
    """(iou, gi, pj) arrays of every same-team overlapping pair, sorted as pm.spell_overlaps."""
    if len(gs) == 0 or len(ps) == 0:
        return np.zeros(0), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    lo = np.searchsorted(pe, gs, side="right")     # first pred with stop > g.start
    hi = np.searchsorted(ps, ge, side="left")      # pred with start < g.stop
    gi_list, pj_list = [], []
    for i in range(len(gs)):
        if hi[i] > lo[i]:
            js = np.arange(lo[i], hi[i])
            gi_list.append(np.full(len(js), i))
            pj_list.append(js)
    if not gi_list:
        return np.zeros(0), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    gi = np.concatenate(gi_list)
    pj = np.concatenate(pj_list)
    same = gl[gi] == pl[pj]
    gi, pj = gi[same], pj[same]
    inter = np.minimum(ge[gi], pe[pj]) - np.maximum(gs[gi], ps[pj])
    union = np.maximum(ge[gi], pe[pj]) - np.minimum(gs[gi], ps[pj])
    ok = inter > 0
    gi, pj, inter, union = gi[ok], pj[ok], inter[ok], union[ok]
    iou = inter / union.astype(np.float64)
    order = np.lexsort((pj, gi, -iou))
    return iou[order], gi[order], pj[order]


def greedy_flags(iou, gi, pj, n_gt, n_pred, threshold):
    """(gt matched bool, pred matched bool, pairs dict gi->pj) as pm.matched_gt_flags."""
    gflag = np.zeros(n_gt, dtype=bool)
    pflag = np.zeros(n_pred, dtype=bool)
    pairs = {}
    for v, i, j in zip(iou, gi, pj):
        if v < threshold:
            break
        if gflag[i] or pflag[j]:
            continue
        gflag[i] = True
        pflag[j] = True
        pairs[int(i)] = int(j)
    return gflag, pflag, pairs


def spell_eval(gt_sp, pred_codes, runs, ious=IOUS):
    """Dict: spellF1 mean, F1 per IoU, gt flags (len(ious), n_gt), pred flags, pred spells."""
    gs, ge, gl = gt_sp
    ps, pe, pl = spells(pred_codes, runs)
    iou, gi, pj = fast_overlaps(gs, ge, gl, ps, pe, pl)
    f1s, gflags, pflags = [], [], []
    for t in ious:
        gf, pf, _ = greedy_flags(iou, gi, pj, len(gs), len(ps), t)
        tp = int(gf.sum())
        fp, fn = len(ps) - tp, len(gs) - tp
        f1s.append(2.0 * tp / (2 * tp + fp + fn) if tp else 0.0)
        gflags.append(gf)
        pflags.append(pf)
    return {"mean": float(np.mean(f1s)), "f1": f1s, "gflags": np.array(gflags),
            "pflags": np.array(pflags), "pred": (ps, pe, pl)}


def check_fast_scoring(gt, pred, runs):
    """Assert fast scoring == possession_metrics.spell_scores on these codes."""
    ref = pm.spell_scores(gt, pred, runs)
    got = spell_eval(spells(gt, runs), pred, runs)
    assert abs(ref["mean_f1"] - got["mean"]) < 1e-12, (ref["mean_f1"], got["mean"])
    for k, t in enumerate(IOUS):
        assert abs(ref["per_iou"][t]["f1"] - got["f1"][k]) < 1e-12
    return got["mean"]


# -- owner changes -----------------------------------------------------------------------------

def owner_changes(codes, runs):
    """List of dicts, one per pair of consecutive alive runs inside one graded run.

    frame: first frame of the later alive run. src, dst: the teams. kind: "direct" (adjacent),
    "across_D" (a D stretch between, teams differ) or "same_D" (a D stretch, same team).
    gap: frames of D between. d_start: first D frame (= end of the earlier alive run).
    prev_start: start of the earlier alive run. next_stop: end of the later alive run.
    """
    s, e, l, r = run_segments(codes, runs)
    alive = (l == H) | (l == A)
    out = []
    idx = np.flatnonzero(alive)
    for k in range(1, len(idx)):
        i0, i1 = idx[k - 1], idx[k]
        if r[i0] != r[i1]:
            continue
        gap = int(s[i1] - e[i0])
        if l[i0] != l[i1]:
            kind = "direct" if gap == 0 else "across_D"
        else:
            kind = "same_D"
        out.append({"frame": int(s[i1]), "src": int(l[i0]), "dst": int(l[i1]), "kind": kind,
                    "gap": gap, "d_start": int(e[i0]), "prev_start": int(s[i0]),
                    "next_stop": int(e[i1]), "run": int(r[i1])})
    return out


def match_changes(ref_list, our_list, tol):
    """One-to-one order-preserving matching per destination team within +-tol frames.

    Maximum cardinality first, then the least total |latency|. Returns (pairs, ref_unmatched,
    our_unmatched) with pairs as (ref index, our index) into the given lists.
    """
    pairs = []
    ref_un, our_un = [], []
    for key in (H, A):
        ri = [i for i, c in enumerate(ref_list) if c["dst"] == key]
        oi = [j for j, c in enumerate(our_list) if c["dst"] == key]
        rf = np.array([ref_list[i]["frame"] for i in ri], dtype=np.int64)
        of = np.array([our_list[j]["frame"] for j in oi], dtype=np.int64)
        n, m = len(ri), len(oi)
        if n == 0 or m == 0:
            ref_un += ri
            our_un += oi
            continue
        # DP over (i, j): cost[i][j] = best cost of aligning rf[:i] and of[:j].
        cost = np.full((n + 1, m + 1), np.inf)
        move = np.zeros((n + 1, m + 1), dtype=np.int8)
        cost[0, :] = BIG * np.arange(m + 1)
        cost[:, 0] = BIG * np.arange(n + 1)
        move[0, 1:] = 2
        move[1:, 0] = 1
        # Banded: a pred j can only match ref i if |of[j]-rf[i]| <= tol.
        for i in range(1, n + 1):
            row_prev = cost[i - 1]
            row = cost[i]
            for j in range(1, m + 1):
                best = row_prev[j] + BIG
                mv = 1
                c2 = row[j - 1] + BIG
                if c2 < best:
                    best, mv = c2, 2
                d = abs(int(of[j - 1]) - int(rf[i - 1]))
                if d <= tol:
                    c3 = row_prev[j - 1] + d
                    if c3 < best:
                        best, mv = c3, 3
                row[j] = best
                move[i, j] = mv
        i, j = n, m
        while i > 0 or j > 0:
            mv = move[i, j]
            if mv == 3:
                pairs.append((ri[i - 1], oi[j - 1]))
                i, j = i - 1, j - 1
            elif mv == 1:
                ref_un.append(ri[i - 1])
                i -= 1
            else:
                our_un.append(oi[j - 1])
                j -= 1
    pairs.sort()
    return pairs, sorted(ref_un), sorted(our_un)


def quantiles(values, qs=(10, 25, 50, 75, 90)):
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {str(q): None for q in qs}
    return {str(q): float(np.percentile(v, q)) for q in qs}


def frac(num, den):
    return float(num) / float(den) if den else None
