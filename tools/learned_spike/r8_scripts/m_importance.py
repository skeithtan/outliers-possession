"""Arm M (round 8): split-gain importance of a member's tree models by column group. Not part of the submission.

    # pf venv
    python -m tools.learned_spike.r8_scripts.m_importance --spec FEATS_F6/spec.json \
        --models DIR [DIR ...] --out FILE.json

Importance of a column = the sum of the split gains (the "gain" field of scikit-learn's
HistGradientBoosting predictor nodes, inner nodes only) of the splits on it, over all trees;
reported as shares of the model's total per group: base (the 50 columns of
engine.learned_features), proximity (the 8 proximity-touch columns), context (their windows),
and the four groups of engine.learned_pf_groups (G-a ball against the blocks, G-b block motion,
G-c block area and overlap, G-d sharpness), plus the top columns. Python 3.9.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

GROUPS = {
    "G-a": ("gbehind", "glevel", "gahead", "gdahead_p1", "gdahead_n1", "gbc", "gbdeep",
            "gbfront"),
    "G-b": ("gvc", "gvdeep", "gvfront"),
    "G-c": ("garea", "g_ovx", "g_ovy", "g_gapx", "g_cdist"),
    "G-d": ("gnacc", "g_bacc", "g_bturn", "g_bacc_m04", "g_bacc_m1", "g_bturn_m04",
            "g_bturn_m1"),
}


def group_of(i, name):
    if i < 50:
        return "base"
    for g, keys in GROUPS.items():
        stem = name[2:] if name[:2] in ("H_", "A_") else name
        if stem in keys:
            return g
    return "proximity" if "__" not in name else "context"


def importance(clf, names):
    gain = np.zeros(clf.n_features_in_)
    for pr in clf._predictors:
        nd = pr[0].nodes
        inner = ~nd["is_leaf"].astype(bool)
        np.add.at(gain, nd["feature_idx"][inner], nd["gain"][inner])
    tot = gain.sum()
    groups = {}
    for i, g in enumerate(gain):
        k = group_of(i, names[i])
        groups[k] = groups.get(k, 0.0) + float(g / tot)
    top = [(names[i], float(gain[i] / tot)) for i in np.argsort(-gain)[:12]]
    return {"groups": groups, "top": top}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--spec", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    names = json.load(open(a.spec))["names"]
    res = {}
    for d in a.models:
        with open(os.path.join(d, "models.pkl"), "rb") as h:
            m = pickle.load(h)
        res[d] = {k: importance(m[k], names) for k in ("dead", "team")}
    json.dump(res, open(a.out, "w"), indent=1)
    for d, r in res.items():
        for k in ("dead", "team"):
            print(d[-30:], k, {g: round(v, 3) for g, v in sorted(r[k]["groups"].items())})
    return 0


if __name__ == "__main__":
    sys.exit(main())
