"""Round 3 scoping (stage 1): decoder diagnosis on the round 2 out-of-fold probabilities.

Not part of the submission. Reads the round 2 fold test probabilities (<game>_p5.npz), the
build.py label cache (raw and ctl2 reference codes) and the c22 runs recorded in fold.json. It
never reads evaluation/gt, never opens a sealed match and never decodes BAR-ATH against labels.
Results go to OUT. See docs/plan/research/LEARNED-R3-SCOPING.md. Python 3.9, numpy only.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import possession_metrics as pm  # noqa: E402
import score_game as sg  # noqa: E402
from engine import data, decoder  # noqa: E402
from engine.config import Config  # noqa: E402
from tools.learned_spike import decoders  # noqa: E402

H, A, D, X = 0, 1, 2, 3
FPS = 25
DEV = ["METRICA-S1", "METRICA-S2", "METRICA-S3", "DFL-J03WMX", "DFL-J03WN1", "DFL-J03WOH",
       "DFL-J03WOY", "DFL-J03WPY"]
SEALED = {"DFL-J03WQQ", "DFL-J03WR9", "BAR-ATH"}


def provider(game):
    return game.split("-")[0]


# -- spellF1, fast and equal to possession_metrics.spell_scores ---------------------------------

def spells(codes, runs):
    out = []
    for a, b in runs:
        w = codes[a:b]
        if b <= a:
            continue
        br = np.flatnonzero(w[1:] != w[:-1]) + 1
        st = np.concatenate(([0], br)) + a
        sp = np.concatenate((br, [b - a])) + a
        lab = w[st - a]
        for s, e, k in zip(st, sp, lab):
            if k == H or k == A:
                out.append((int(s), int(e), int(k)))
    return out


def overlaps(gs, ps):
    out = []
    j0 = 0
    for i, (g0, g1, gk) in enumerate(gs):
        while j0 < len(ps) and ps[j0][1] <= g0:
            j0 += 1
        j = j0
        while j < len(ps) and ps[j][0] < g1:
            p0, p1, pk = ps[j]
            if pk == gk:
                inter = min(g1, p1) - max(g0, p0)
                uni = max(g1, p1) - min(g0, p0)
                out.append((inter / float(uni), i, j))
            j += 1
    out.sort(key=lambda t: (-t[0], t[1], t[2]))
    return out


def spell_f1_from(gs, ps):
    ov = overlaps(gs, ps)
    f1s = []
    for thr in pm.SPELL_IOUS:
        used_g, used_p, tp = set(), set(), 0
        for iou, i, j in ov:
            if iou < thr:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            tp += 1
        f1s.append(2.0 * tp / (len(gs) + len(ps)) if tp else 0.0)
    return float(np.mean(f1s))


def spell_f1(ref, pred, runs=None):
    ref = np.asarray(ref, dtype=np.int8)
    if runs is None:
        runs = pm.valid_runs(ref)
    pred = pm.fill_abstentions(np.asarray(pred, dtype=np.int8), runs)
    return spell_f1_from(spells(ref, runs), spells(pred, runs))


def stats(codes, runs):
    sp = spells(np.asarray(codes, dtype=np.int8), runs)
    lens = np.array([b - a for a, b, _ in sp], dtype=float) / FPS
    s = sg.label_stats(codes, runs)
    return {"n": len(sp), "median_s": float(np.median(lens)) if len(lens) else 0.0,
            "p90_s": float(np.percentile(lens, 90)) if len(lens) else 0.0,
            "mean_s": float(lens.mean()) if len(lens) else 0.0,
            "d_share": s["d_share"], "d_segments": s["d_segments"]}


def d_segment_lengths(codes, runs):
    out = []
    for a, b in runs:
        w = codes[a:b]
        br = np.flatnonzero(w[1:] != w[:-1]) + 1
        st = np.concatenate(([0], br))
        sp = np.concatenate((br, [b - a]))
        out += [(e - s) / FPS for s, e in zip(st, sp) if w[s] == D]
    return np.array(out)


# -- data -------------------------------------------------------------------------------------

def fold_of(stage, game):
    for k in range(8):
        if os.path.isfile(os.path.join(stage, "folds", "fold%d" % k, "test", game + "_p5.npz")):
            return k
    raise KeyError(game)


def load(game, stage, cache):
    assert game not in SEALED
    k = fold_of(stage, game)
    fdir = os.path.join(stage, "folds", "fold%d" % k)
    p5 = np.load(os.path.join(fdir, "test", game + "_p5.npz"))["p5"]
    lab = np.load(os.path.join(cache, "labels", game + ".npz"))
    raw, ctl2 = lab["raw"].astype(np.int8), lab["ctl2"].astype(np.int8)
    n = len(raw)
    with open(os.path.join(fdir, "costs.json")) as h:
        costs = json.load(h)
    with open(os.path.join(fdir, "fold.json")) as h:
        fold = json.load(h)
    c22_path = fold["c22_test_runs"][game][0]
    c22 = sg.read_codes(c22_path, n)
    m_csv = sg.read_codes(os.path.join(fdir, "test", game + "_M.csv"), n)
    probs = decoders.upsample(p5, n)
    blocks = data.graded_blocks(game)
    return {"game": game, "fold": k, "p5": p5, "probs": probs, "raw": raw, "ctl2": ctl2,
            "c22": c22, "m_csv": m_csv, "blocks": blocks,
            "cost": (costs["M"]["cost_ha"], costs["M"]["cost_ad"]), "costs_json": costs}


# -- decoders ---------------------------------------------------------------------------------

def viterbi(probs, blocks, cha, cad):
    return decoders.decode_model(probs, blocks, [(cha, cad)])[0]


def viterbi_many(probs, blocks, pairs):
    return decoders.decode_model(probs, blocks, pairs)


_CFG = Config()


def fold_min(codes, probs, blocks, min_poss=None, min_dead=None, cha=0.0, cad=0.0):
    """The engine's minimum-duration post-pass (decoder.min_durations) on each graded block."""
    cfg = _CFG
    if min_poss is not None or min_dead is not None:
        import dataclasses
        cfg = dataclasses.replace(_CFG, MIN_POSS=int(min_poss if min_poss is not None
                                                      else _CFG.MIN_POSS),
                                  MIN_DEAD=int(min_dead if min_dead is not None
                                               else _CFG.MIN_DEAD))
    out = codes.copy()
    logp = np.log(np.clip(probs, decoders.LOG_FLOOR, 1.0))
    for _, a, b in blocks:
        b = b + 1
        n = b - a
        c = np.zeros((n, 3, 3))
        c[:, H, A] = c[:, A, H] = cha
        c[:, :2, D] = cad
        c[:, D, :2] = cad
        out[a:b] = decoder.min_durations(codes[a:b].astype(np.int8), logp[a:b], c, cfg)
    return out


def argmax_codes(probs, blocks):
    out = np.full(len(probs), X, dtype=np.int8)
    am = probs.argmax(axis=1).astype(np.int8)
    for _, a, b in blocks:
        out[a:b + 1] = am[a:b + 1]
    return out


# -- truth samples: forward filtering, backward sampling --------------------------------------

def ffbs(probs, blocks, trans, n_samples, seed, step=1):
    """int8 (S, N) label paths drawn from the chain with transition matrix `trans` (per step)
    and the frame posteriors as emissions (flat prior), on the graded blocks; X elsewhere.

    step > 1 samples on every step-th frame and repeats each sample over the step.
    """
    rng = np.random.RandomState(seed)
    n = len(probs)
    out = np.full((n_samples, n), X, dtype=np.int8)
    for _, a, b in blocks:
        b = b + 1
        idx = np.arange(a, b, step)
        e = np.clip(probs[idx], 1e-6, 1.0)
        m = len(idx)
        alpha = np.empty((m, 3))
        al = e[0] / e[0].sum()
        alpha[0] = al
        for t in range(1, m):
            al = (al @ trans) * e[t]
            al /= al.sum()
            alpha[t] = al
        z = np.empty((n_samples, m), dtype=np.int8)
        u = rng.random_sample((n_samples, m))
        w = alpha[-1]
        z[:, -1] = np.searchsorted(np.cumsum(w), u[:, -1] * w.sum())
        for t in range(m - 2, -1, -1):
            w = alpha[t][None, :] * trans[:, z[:, t + 1]].T          # (S, 3)
            c = np.cumsum(w, axis=1)
            z[:, t] = (u[:, t:t + 1] * c[:, -1:] > c).sum(axis=1)
        z = np.minimum(z, 2)
        full = np.repeat(z, step, axis=1)[:, :b - a]
        out[:, a:b] = full
    return out


def transition_matrix(ref_list, step=1):
    """Per-step transition matrix of H, A, D estimated from reference code arrays (graded)."""
    cnt = np.zeros((3, 3))
    for ref in ref_list:
        for a, b in pm.valid_runs(ref):
            w = ref[a:b:step]
            w = w[w != X]
            np.add.at(cnt, (w[:-1], w[1:]), 1)
    cnt += 1.0
    return cnt / cnt.sum(axis=1, keepdims=True)


def mbr_pick(candidates, samples, blocks):
    runs = [(a, b + 1) for _, a, b in blocks]
    sample_spells = [spells(s, runs) for s in samples]
    scores = []
    for c in candidates:
        cs = spells(np.asarray(c, dtype=np.int8), runs)
        scores.append(float(np.mean([spell_f1_from(ss, cs) for ss in sample_spells])))
    return int(np.argmax(scores)), scores
