"""Q4: possession-level accounting by error class, with exact Shapley values of spellF1.

    python q4.py        -> out/q4.json, out/episodes_<view>.csv

Every graded frame where our (filled) label differs from the reference belongs to exactly one
error episode, and every episode to exactly one class:

Team episodes (we alive, reference alive, teams differ; a maximal run with a constant pair):
  T_late      the reference changed team (live) and we followed later (start: reference change,
              end: our change to the reference team)
  T_missed_absorbed  the reference changed team (live) and changed back before we followed: the
              whole reference possession of the other team sits inside our spell
  T_missed_to_dead   the reference changed team (live) and we never followed before a D (ours
              or the reference's) or the run end
  T_early     we changed team before the reference did (start: our change, end: reference change)
  T_spurious  we changed team and the reference did not (end: we return, a D, or the run end)
  T_restart_after_missedD  the reference restarts with the other team while we stayed alive
              through its stoppage (downstream of a missed or short D)
  T_restart_wrong   the episode starts at our restart or at both restarts or a run start: wrong
              team after a (re)start
  T_swap      both changed at the same frame (rare)
D-mask episodes:
  D_onset     timing of a stoppage entry (we enter early: false D touching the reference D on its
              right; or late: missed D touching our D on its right)
  D_restart   timing of a stoppage exit (late exit: false D touching the reference D on its left;
              early exit: missed D touching our D on its left)
  D_spurious  a stoppage we invent inside reference play (false D touching no reference D), or
              our D covering a whole reference alive stretch between two reference stoppages
  D_missed    a reference stoppage we miss entirely (missed D touching none of our D), or an alive
              island of ours inside a reference stoppage

Fixing a set S of classes = setting their frames to the reference. v(S) = spellF1 of that output.
v(all) = 1. The exact Shapley value of each class (2^12 subsets) splits 1 - v(empty) additively.
Per possession (IoU 0.5): which classes, fixed alone, recover an unmatched reference spell or
remove an unmatched predicted spell; the minimal fixing subsets otherwise.
"""
import itertools
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
WORK = C.WORK

CLASSES = ("T_late", "T_missed_absorbed", "T_missed_to_dead", "T_early", "T_spurious",
           "T_restart_wrong", "T_restart_after_missedD", "T_swap",
           "D_onset", "D_restart", "D_spurious", "D_missed")
K = len(CLASSES)
CI = {c: i for i, c in enumerate(CLASSES)}
IOU_HEAD = 0.5
IHEAD = list(C.IOUS).index(IOU_HEAD)


def episodes(gt, ours, runs):
    """List of episode dicts (start, stop, cls, sub, our, ref) and an int8 class map (-1 none)."""
    cls_map = np.full(len(gt), -1, dtype=np.int8)
    out = []
    alive = lambda v: v == C.H or v == C.A  # noqa: E731
    for a0, b0 in runs:
        g = gt[a0:b0]
        o = ours[a0:b0]
        ga = (g == C.H) | (g == C.A)
        oa = (o == C.H) | (o == C.A)
        etype = np.zeros(b0 - a0, dtype=np.int16)          # 0 = no error
        team_err = ga & oa & (g != o)
        etype[team_err] = 10 + 2 * o[team_err] + g[team_err]  # pair code (W, R)
        etype[(o == C.D) & ga] = 1                           # false D
        etype[oa & (g == C.D)] = 2                           # missed D
        br = np.flatnonzero(etype[1:] != etype[:-1]) + 1
        starts = np.concatenate(([0], br))
        stops = np.concatenate((br, [len(etype)]))
        for s, e in zip(starts, stops):
            t = int(etype[s])
            if t == 0:
                continue
            a, b = a0 + int(s), a0 + int(e)
            prev = a - 1 if a > a0 else None
            nxt = b if b < b0 else None
            rec = {"start": a, "stop": b, "len": b - a}
            if t >= 10:
                W, R = int(ours[a]), int(gt[a])
                rec.update(our="HA"[W], ref="HA"[R])
                if prev is None:
                    st = "run_start"
                elif gt[prev] == C.D and ours[prev] == C.D:
                    st = "both_restart"
                elif gt[prev] == C.D:
                    st = "ref_restart_we_alive"
                elif ours[prev] == C.D:
                    st = "our_restart_ref_alive"
                elif gt[prev] == W and ours[prev] == W:
                    st = "ref_change"
                elif gt[prev] == R and ours[prev] == R:
                    st = "our_change"
                else:
                    st = "swap"
                if nxt is None:
                    en = "run_end"
                elif gt[nxt] == C.D or ours[nxt] == C.D:
                    en = "dead"
                elif ours[nxt] == R and gt[nxt] == R:
                    en = "our_change"
                elif ours[nxt] == W and gt[nxt] == W:
                    en = "ref_change"
                else:
                    en = "swap"
                if st == "ref_change":
                    if en == "our_change":
                        cls = "T_late"
                    elif en in ("ref_change", "swap"):
                        cls = "T_missed_absorbed"
                    else:
                        cls = "T_missed_to_dead"
                elif st == "our_change":
                    cls = "T_early" if en == "ref_change" else "T_spurious"
                elif st == "swap":
                    cls = "T_swap"
                elif st == "ref_restart_we_alive":
                    cls = "T_restart_after_missedD"
                else:
                    cls = "T_restart_wrong"
                rec.update(cls=cls, sub="{}>{}".format(st, en))
            elif t == 1:
                left = prev is None or gt[prev] == C.D
                right = nxt is None or gt[nxt] == C.D
                rec.update(our="D", ref="HA"[int(gt[a])])
                if left and right:
                    cls, sub = "D_spurious", "false_D_bridge"
                elif left:
                    cls, sub = "D_restart", "late_restart"
                elif right:
                    cls, sub = "D_onset", "early_onset"
                else:
                    cls, sub = "D_spurious", "spurious_stoppage"
                rec.update(cls=cls, sub=sub)
            else:
                left = prev is not None and ours[prev] == C.D
                right = nxt is not None and ours[nxt] == C.D
                rec.update(our="HA"[int(ours[a])], ref="D")
                if left and right:
                    cls, sub = "D_missed", "alive_island"
                elif left:
                    cls, sub = "D_restart", "early_restart"
                elif right:
                    cls, sub = "D_onset", "late_onset"
                else:
                    cls, sub = "D_missed", "missed_stoppage"
                rec.update(cls=cls, sub=sub)
            cls_map[a:b] = CI[rec["cls"]]
            out.append(rec)
    return out, cls_map


def subsets_eval(gt, ours, runs, cls_map):
    """For every subset (bitmask) of CLASSES: spellF1 mean, F1@0.5, gt flags@0.5, pred info."""
    gsp = C.spells(gt, runs)
    n_sub = 1 << K
    v_mean = np.zeros(n_sub)
    v_head = np.zeros(n_sub)
    gflag = np.zeros((n_sub, len(gsp[0])), dtype=bool)
    pred_info = []
    idx = cls_map.astype(np.int64) + 1           # 0 = no error, k + 1 = class k
    for S in range(n_sub):
        fixed = ours.copy()
        if S:
            lut = np.zeros(K + 1, dtype=bool)
            for k in range(K):
                lut[k + 1] = bool(S >> k & 1)
            m = lut[idx]
            fixed[m] = gt[m]
        ev = C.spell_eval(gsp, fixed, runs)
        v_mean[S] = ev["mean"]
        v_head[S] = ev["f1"][IHEAD]
        gflag[S] = ev["gflags"][IHEAD]
        pred_info.append((ev["pred"], ev["pflags"][IHEAD]))
    return v_mean, v_head, gflag, pred_info, gsp


def shapley(v):
    """Exact Shapley values of a set function v over K players (v indexed by bitmask)."""
    phi = np.zeros(K)
    fact = [math.factorial(i) for i in range(K + 1)]
    for k in range(K):
        bit = 1 << k
        for S in range(1 << K):
            if S & bit:
                continue
            s = bin(S).count("1")
            w = fact[s] * fact[K - s - 1] / fact[K]
            phi[k] += w * (v[S | bit] - v[S])
    return phi


def minimal_fixes(flag_rows):
    """Minimal subsets (as tuples of class names) whose flag is True; flag_rows: bool (2^K,)."""
    hits = np.flatnonzero(flag_rows)
    if len(hits) == 0:
        return None, []
    sizes = np.array([bin(int(S)).count("1") for S in hits])
    smin = sizes.min()
    mins = []
    for S in hits[sizes == smin]:
        mins.append(tuple(CLASSES[k] for k in range(K) if int(S) >> k & 1))
    return int(smin), mins


def analyse(view, game):
    M = C.load(view, game)
    gt, ours, runs = M["gt"], M["ours"], M["runs"]
    eps, cls_map = episodes(gt, ours, runs)
    # every differing graded frame must have a class
    graded = np.zeros(len(gt), dtype=bool)
    for a, b in runs:
        graded[a:b] = True
    diff = graded & (gt != ours)
    assert np.array_equal(diff, cls_map >= 0), "class map does not cover the errors"
    v_mean, v_head, gflag, pred_info, gsp = subsets_eval(gt, ours, runs, cls_map)
    full = (1 << K) - 1
    assert abs(v_mean[full] - 1.0) < 1e-12, v_mean[full]
    res = {"view": view, "game": game, "group": M["group"], "base": v_mean[0],
           "base_head": v_head[0], "hours": M["hours"],
           "n_ref_spells": int(len(gsp[0])), "n_pred_spells": int(len(pred_info[0][0][0]))}
    res["shapley_mean"] = dict(zip(CLASSES, shapley(v_mean).tolist()))
    res["shapley_head"] = dict(zip(CLASSES, shapley(v_head).tolist()))
    res["alone_mean"] = {c: v_mean[1 << k] - v_mean[0] for k, c in enumerate(CLASSES)}
    res["last_mean"] = {c: v_mean[full] - v_mean[full ^ (1 << k)] for k, c in enumerate(CLASSES)}
    res["alone_head"] = {c: v_head[1 << k] - v_head[0] for k, c in enumerate(CLASSES)}
    team_bits = sum(1 << CI[c] for c in CLASSES if c.startswith("T_"))
    dead_bits = sum(1 << CI[c] for c in CLASSES if c.startswith("D_"))
    res["team_all"] = v_mean[team_bits] - v_mean[0]        # = D2 oracle_team gain
    res["dead_all"] = v_mean[dead_bits] - v_mean[0]
    # episode counts and frames per class
    epdf = pd.DataFrame(eps)
    res["episodes"] = {c: {"n": int((epdf.cls == c).sum()),
                           "frames": int(epdf[epdf.cls == c]["len"].sum()),
                           "len_median": float(epdf[epdf.cls == c]["len"].median())
                           if (epdf.cls == c).any() else None} for c in CLASSES}
    res["episode_subs"] = epdf.groupby(["cls", "sub"]).size().to_dict()
    res["episode_subs"] = {"{}|{}".format(k[0], k[1]): int(v) for k, v in res["episode_subs"].items()}
    # possession attribution at IoU 0.5: reference spells
    unmatched = np.flatnonzero(~gflag[0])
    ref_attr = []
    for i in unmatched:
        size, mins = minimal_fixes(gflag[:, i])
        singles = [m[0] for m in mins] if size == 1 else []
        ref_attr.append({"i": int(i), "start": int(gsp[0][i]), "stop": int(gsp[1][i]),
                         "dur_s": (gsp[1][i] - gsp[0][i]) / C.FPS, "size": size,
                         "singles": singles, "mins": ["+".join(m) for m in mins]})
    res["ref_unmatched"] = len(unmatched)
    res["ref_attr"] = ref_attr
    # predicted spells: unmatched in the base output
    (ps, pe, pl), pflag0 = pred_info[0]
    un_pred = np.flatnonzero(~pflag0)
    resolved = np.zeros((1 << K, len(un_pred)), dtype=bool)
    for S in range(1 << K):
        (qs, qe, ql), qflag = pred_info[S]
        lo = np.searchsorted(qe, ps[un_pred], side="right")
        hi = np.searchsorted(qs, pe[un_pred], side="left")
        for n, j in enumerate(un_pred):
            span = qflag[lo[n]:hi[n]]
            resolved[S, n] = bool(span.all()) if len(span) else True
    pred_attr = []
    for n, j in enumerate(un_pred):
        size, mins = minimal_fixes(resolved[:, n])
        singles = [m[0] for m in mins] if size == 1 else []
        seg = cls_map[ps[j]:pe[j]]
        inside = np.bincount(seg[seg >= 0], minlength=K)
        pred_attr.append({"j": int(j), "start": int(ps[j]), "stop": int(pe[j]),
                          "dur_s": (pe[j] - ps[j]) / C.FPS, "size": size, "singles": singles,
                          "mins": ["+".join(m) for m in mins],
                          "dominant_inside": CLASSES[int(np.argmax(inside))] if inside.sum() else "none",
                          "err_frac_inside": float((seg >= 0).mean())})
    res["pred_unmatched"] = len(un_pred)
    res["pred_attr"] = pred_attr
    for e in eps:
        e.update(view=view, game=game, group=M["group"])
    return res, eps


def attr_counts(items):
    """Counts: single-class fixes (exclusive), multi single-class fixes (ambiguous), needs 2+."""
    single = {c: 0 for c in CLASSES}
    share = {c: 0.0 for c in CLASSES}
    amb = 0
    compound = {}
    unfixable = 0
    for it in items:
        if it["size"] is None:
            unfixable += 1
            continue
        if it["size"] == 1:
            if len(it["singles"]) == 1:
                single[it["singles"][0]] += 1
            else:
                amb += 1
            for c in it["singles"]:
                share[c] += 1.0 / len(it["singles"])
        else:
            key = it["mins"][0] if len(it["mins"]) == 1 else "|".join(sorted(it["mins"]))[:120]
            compound[key] = compound.get(key, 0) + 1
            # fractional credit across the classes of the minimal subsets
            cs = set(c for m in it["mins"] for c in m.split("+"))
            for c in cs:
                share[c] += 1.0 / len(cs)
    top = dict(sorted(compound.items(), key=lambda kv: -kv[1])[:12])
    return {"single_exclusive": single, "fractional_share": share, "ambiguous_single": amb,
            "compound_n": int(sum(compound.values())), "compound_top": top,
            "unfixable": unfixable, "n": len(items)}


def main():
    os.makedirs(os.path.join(WORK, "out"), exist_ok=True)
    report = {}
    for view in ("matched", "raw"):
        per, all_eps = [], []
        for game in C.OPEN_GAMES:
            res, eps = analyse(view, game)
            per.append(res)
            all_eps += eps
            print(view, game, "base %.4f" % res["base"], "team_all %+.3f" % res["team_all"],
                  "dead_all %+.3f" % res["dead_all"], flush=True)
        pd.DataFrame(all_eps).to_csv(os.path.join(WORK, "out", "episodes_%s.csv" % view), index=False)
        rep = {"per_match": per}
        for name, pref in (("Metrica", "METRICA-"), ("DFL", "DFL-"), ("pooled", "")):
            sel = [r for r in per if r["game"].startswith(pref)]
            agg = {"games": [r["game"] for r in sel], "base": float(np.mean([r["base"] for r in sel]))}
            for key in ("shapley_mean", "shapley_head", "alone_mean", "last_mean", "alone_head"):
                agg[key] = {c: float(np.mean([r[key][c] for r in sel])) for c in CLASSES}
            agg["team_all"] = float(np.mean([r["team_all"] for r in sel]))
            agg["dead_all"] = float(np.mean([r["dead_all"] for r in sel]))
            agg["episodes"] = {c: {"n": int(sum(r["episodes"][c]["n"] for r in sel)),
                                   "frames": int(sum(r["episodes"][c]["frames"] for r in sel))}
                               for c in CLASSES}
            subs = {}
            for r in sel:
                for k, v in r["episode_subs"].items():
                    subs[k] = subs.get(k, 0) + v
            agg["episode_subs"] = dict(sorted(subs.items()))
            agg["ref_attr"] = attr_counts([a for r in sel for a in r["ref_attr"]])
            agg["pred_attr"] = attr_counts([a for r in sel for a in r["pred_attr"]])
            agg["n_ref_spells"] = int(sum(r["n_ref_spells"] for r in sel))
            agg["n_pred_spells"] = int(sum(r["n_pred_spells"] for r in sel))
            agg["ref_unmatched"] = int(sum(r["ref_unmatched"] for r in sel))
            agg["pred_unmatched"] = int(sum(r["pred_unmatched"] for r in sel))
            agg["pred_dominant_inside"] = pd.Series(
                [a["dominant_inside"] for r in sel for a in r["pred_attr"]]).value_counts().to_dict()
            rep[name] = agg
        # balanced = mean of the Metrica and DFL means
        rep["balanced"] = {key: {c: 0.5 * (rep["Metrica"][key][c] + rep["DFL"][key][c]) for c in CLASSES}
                           for key in ("shapley_mean", "shapley_head", "alone_mean", "last_mean")}
        rep["balanced"]["base"] = 0.5 * (rep["Metrica"]["base"] + rep["DFL"]["base"])
        for r in per:
            r.pop("ref_attr")
            r.pop("pred_attr")
        report[view] = rep
    with open(os.path.join(WORK, "out", "q4.json"), "w") as handle:
        json.dump(report, handle, indent=1, sort_keys=True,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote out/q4.json")


if __name__ == "__main__":
    main()
