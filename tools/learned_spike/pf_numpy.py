"""Pure-numpy inference of the per-frame boosted models (round 6 spike feasibility). Not part of the submission.

    # pf venv (exports from scikit-learn, then predicts with numpy only)
    python -m tools.learned_spike.pf_numpy check --models OUT/models/full/metrica/models.pkl \
        --feats OUT/feats/eval/METRICA-S1.npy

export() turns a fitted HistGradientBoostingClassifier into flat arrays (per node: feature,
threshold, left, right, is_leaf, value; per tree: root offset) plus the baseline. predict_raw()
walks all trees level by level for all frames at once (numpy gathers only): x <= threshold
goes left (the features hold no NaN; the missing-value branch is not needed). The check
compares with scikit-learn's predict_proba and times both.
"""
import argparse
import pickle
import sys
import time

import numpy as np


def export(clf):
    feats, thr, left, right, leaf, value, roots, depth = [], [], [], [], [], [], [], 0
    off = 0
    for (pred,) in clf._predictors:
        nd = pred.nodes
        roots.append(off)
        feats.append(nd["feature_idx"].astype(np.int32))
        thr.append(nd["num_threshold"].astype(np.float64))
        left.append(nd["left"].astype(np.int32) + off)
        right.append(nd["right"].astype(np.int32) + off)
        leaf.append(nd["is_leaf"].astype(bool))
        value.append(nd["value"].astype(np.float64))
        depth = max(depth, int(pred.get_max_depth()))
        off += len(nd)
    base = float(np.ravel(clf._baseline_prediction)[0])
    return {"feature": np.concatenate(feats), "threshold": np.concatenate(thr),
            "left": np.concatenate(left), "right": np.concatenate(right),
            "leaf": np.concatenate(leaf), "value": np.concatenate(value),
            "roots": np.array(roots, np.int32), "depth": depth, "baseline": base}


def predict_raw(m, X):
    n = X.shape[0]
    raw = np.full(n, m["baseline"])
    rows = np.arange(n)
    for root in m["roots"]:
        node = np.full(n, root, np.int32)
        for _ in range(m["depth"]):
            lf = m["leaf"][node]
            if lf.all():
                break
            go_left = X[rows, m["feature"][node]] <= m["threshold"][node]
            nxt = np.where(go_left, m["left"][node], m["right"][node])
            node = np.where(lf, node, nxt)
        raw += m["value"][node]
    return raw


def predict_proba1(m, X):
    return 1.0 / (1.0 + np.exp(-predict_raw(m, X)))


KEYS = ("feature", "threshold", "left", "right", "leaf", "value", "roots")


def save_npz(path, models):
    """pf_model.npz: per model m in (dead, team) the arrays m_<key> of export(), m_depth and
    m_baseline; plus the column permutation / signs of the team swap (swap_perm, swap_sign)."""
    out = {}
    for name in ("dead", "team"):
        m = export(models[name])
        for k in KEYS:
            out["{}_{}".format(name, k)] = m[k]
        out[name + "_depth"] = np.int64(m["depth"])
        out[name + "_baseline"] = np.float64(m["baseline"])
    return out


def load_npz(path):
    z = np.load(path)
    models = {}
    for name in ("dead", "team"):
        m = {k: z["{}_{}".format(name, k)] for k in KEYS}
        m["depth"] = int(z[name + "_depth"])
        m["baseline"] = float(z[name + "_baseline"])
        models[name] = m
    return models, z["swap_perm"], z["swap_sign"]


def predict_frames(models, perm, sign, F):
    """(N, 3) H, A, D probabilities, numpy only, exactly as pf_train.predict: p(D) the mean over
    F and its team swap; p(home | alive) the mean of TEAM(F) and 1 - TEAM(swap F)."""
    Fs = F[:, perm] * sign
    pd_ = 0.5 * (predict_proba1(models["dead"], F) + predict_proba1(models["dead"], Fs))
    ph = 0.5 * (predict_proba1(models["team"], F) + 1.0 - predict_proba1(models["team"], Fs))
    return np.stack([(1 - pd_) * ph, (1 - pd_) * (1 - ph), pd_], axis=1)


def cmd_check(args):
    with open(args.models, "rb") as h:
        models = pickle.load(h)
    F = np.load(args.feats)[:, models["cols"]]
    if args.rows:
        F = F[:args.rows]
    for name in ("dead", "team"):
        clf = models[name]
        m = export(clf)
        t0 = time.time()
        ref = clf.predict_proba(F)[:, 1]
        t1 = time.time()
        mine = predict_proba1(m, F)
        t2 = time.time()
        print("{}: trees {} nodes {} depth {}; max |diff| {:.2e}; sklearn {:.2f}s numpy {:.2f}s "
              "({} frames)".format(name, len(m["roots"]), len(m["feature"]), m["depth"],
                                   float(np.abs(ref - mine).max()), t1 - t0, t2 - t1, len(F)))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--models", required=True)
    c.add_argument("--feats", required=True)
    c.add_argument("--rows", type=int, default=0)
    args = p.parse_args(argv)
    return {"check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
