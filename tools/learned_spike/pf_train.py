"""Per-frame gradient-boosted models, left-out-provider fits and inference (round 6 spike). Not part of the submission.

    # pf venv, OMP_NUM_THREADS=4, nice 10
    python -m tools.learned_spike.pf_train fit --feats OUT/feats --labels CA --left-out metrica \
        --arm full --out OUT/models
    python -m tools.learned_spike.pf_train predict --feats OUT/feats --models OUT/models \
        --left-out metrica --arm full --out OUT/probs

Two binary scikit-learn HistGradientBoostingClassifier models per fit (generic settings, not
tuned: learning_rate 0.1, max_leaf_nodes 63, min_samples_leaf 200, up to MAX_ITER 400
iterations, random_state 0):

- DEAD: D against alive, on the training rows of every provider but the left-out one. Each
  Metrica / DFL row weighs 1; SkillCorner rows weigh f so that SkillCorner as a whole has a
  quarter of the weight its match count would give: f = 0.25 x (mean rows per non-SkillCorner
  training match) / (mean rows per SkillCorner training match).
- TEAM: home against away on alive rows (label H or A) of Metrica and DFL only (SkillCorner's
  team labels follow another convention). Row weight 1; arm "spellw" (coordinator's arm e):
  sqrt(median alive spell length of the match / length of the row's own reference spell),
  clipped to [1/3, 3] (a spell = a run of one alive label inside a graded block).

Early stopping: a held-out tenth (rounded up, at least one) of the model's training matches,
chosen by sorting the matches by crc32 of their name; trained to MAX_ITER without sklearn's
own early stopping, then cut at the iteration with the smallest weighted log loss on the
held-out matches (staged_predict_proba; the model's predictor list is truncated there, which
equals training that many iterations). The held-out matches never train the model.

Inference on a match view F: p(D) = mean of DEAD on F and on swap(F); p(home | alive) = mean of
TEAM on F and of 1 - TEAM on swap(F); p(H) = (1 - p(D)) p(home | alive), p(A) = (1 - p(D))
(1 - p(home | alive)). So the output is exactly team symmetric.

Arms: full (all columns), no_touch (b and its context removed), no_context (c removed),
no_prox (proximity touches and their context removed), spellw (all columns, TEAM weighted).
Python 3.9.
"""

import argparse
import json
import math
import os
import pickle
import sys
import time
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from engine import data  # noqa: E402
from tools.learned_spike import folds  # noqa: E402
from tools.learned_spike import pf_features as pff  # noqa: E402

MAX_ITER = 400
PARAMS = dict(learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200, early_stopping=False,
              random_state=0)
SC_SHARE = 0.25
ARMS = ("full", "no_touch", "no_context", "no_prox", "spellw")


def log(text):
    print(time.strftime("%H:%M:%S"), text, file=sys.stderr, flush=True)


def col_kind(arm):
    return "full" if arm == "spellw" else arm


def training_games(left_out):
    return [g for g in folds.ALL if folds.provider(g) != left_out]


def val_split(games):
    k = max(1, int(math.ceil(len(games) / 10.0)))
    order = sorted(games, key=lambda g: zlib.crc32(g.encode()))
    return sorted(order[k:]), sorted(order[:k])


def spell_weights(game, codes, idx):
    n = len(codes)
    block = np.full(n, -1)
    for k, (_, a, b) in enumerate(data.graded_blocks(game)):
        block[a:b + 1] = k
    key = np.where((block >= 0) & (codes <= 1), block * 4 + codes, -1)
    change = np.flatnonzero(np.diff(key) != 0) + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [n]])
    length = np.zeros(n)
    lens = []
    for s, e in zip(starts, stops):
        if key[s] >= 0:
            length[s:e] = e - s
            lens.append(e - s)
    med = float(np.median(lens))
    w = np.sqrt(med / np.maximum(length[idx], 1))
    return np.clip(w, 1 / 3.0, 3.0)


def load_rows(feats, labels, games, cols, team_only=False, spellw=False):
    Xs, ys, ws, gs = [], [], [], []
    for g in games:
        with np.load(os.path.join(feats, "train", pff.check(g) + ".npz")) as z:
            X, y, idx = z["X"][:, cols], z["y"].astype(np.int8), z["idx"]
        if team_only:
            keep = y <= 1
            X, y, idx = X[keep], y[keep], idx[keep]
        w = np.ones(len(y))
        if spellw:
            w = spell_weights(g, pff.labels(labels, g), idx)
        Xs.append(X)
        ys.append(y)
        ws.append(w)
        gs.append(np.full(len(y), g, dtype=object))
    return Xs, ys, ws, gs


def sc_factor(games, ys):
    sc = [len(y) for g, y in zip(games, ys) if g.startswith("SC-")]
    other = [len(y) for g, y in zip(games, ys) if not g.startswith("SC-")]
    if not sc:
        return 1.0
    return SC_SHARE * float(np.mean(other)) / float(np.mean(sc))


def fit_one(X, y, w, Xv, yv, wv, name):
    from sklearn.ensemble import HistGradientBoostingClassifier
    t0 = time.time()
    clf = HistGradientBoostingClassifier(max_iter=MAX_ITER, **PARAMS)
    clf.fit(X, y, sample_weight=w)
    fit_s = time.time() - t0
    losses = []
    eps = 1e-12
    for p in clf.staged_predict_proba(Xv):
        p1 = np.clip(p[:, 1], eps, 1 - eps)
        ll = -(wv * (yv * np.log(p1) + (1 - yv) * np.log(1 - p1))).sum() / wv.sum()
        losses.append(float(ll))
    best = int(np.argmin(losses)) + 1
    full_p = clf.predict_proba(Xv[:200])[:, 1]
    clf._predictors = clf._predictors[:best]          # n_iter_ follows (a property)
    assert clf.n_iter_ == best
    if hasattr(clf, "train_score_"):
        clf.train_score_ = clf.train_score_[:best + 1]
    staged = None
    for i, p in enumerate(clf.staged_predict_proba(Xv[:200])):
        staged = p[:, 1]
    assert np.allclose(clf.predict_proba(Xv[:200])[:, 1], staged)
    rec = {"model": name, "best_iter": best, "trained_iter": len(losses),
           "val_loss_best": losses[best - 1], "val_loss_last": losses[-1], "fit_s": fit_s,
           "rows": int(len(y)), "val_rows": int(len(yv)), "pos_share": float(np.mean(y)),
           "n_leaves_total": int(sum(len(pr[0].nodes) for pr in clf._predictors)),
           "max_depth": int(max(pr[0].get_max_depth() for pr in clf._predictors)),
           "changed_by_cut": bool(not np.allclose(full_p, staged))}
    log("{}: best iter {} / {} val loss {:.4f} (last {:.4f}) {:.0f}s rows {}".format(
        name, best, len(losses), losses[best - 1], losses[-1], fit_s, len(y)))
    return clf, rec


def cmd_fit(args):
    arm = args.arm
    cols = pff.ablation_columns(col_kind(arm))
    out = os.path.join(args.out, arm, args.left_out)
    os.makedirs(out, exist_ok=True)
    games = training_games(args.left_out)
    assert not any(folds.provider(g) == args.left_out for g in games)
    record = {"left_out": args.left_out, "arm": arm, "n_features": int(len(cols)),
              "params": dict(PARAMS, max_iter=MAX_ITER)}
    # DEAD
    tr, va = val_split(games)
    Xs, ys, ws, _ = load_rows(args.feats, args.labels, tr + va, cols)
    f = sc_factor(tr + va, ys)
    for i, g in enumerate(tr + va):
        if g.startswith("SC-"):
            ws[i] = ws[i] * f
    ntr = len(tr)
    X = np.concatenate(Xs[:ntr])
    y = (np.concatenate(ys[:ntr]) == 2).astype(np.int8)
    w = np.concatenate(ws[:ntr])
    Xv = np.concatenate(Xs[ntr:])
    yv = (np.concatenate(ys[ntr:]) == 2).astype(np.int8)
    wv = np.concatenate(ws[ntr:])
    del Xs
    dead, rd = fit_one(X, y, w, Xv, yv, wv, "DEAD")
    rd.update(train=tr, val=va, sc_row_factor=f)
    del X, Xv
    # TEAM
    tgames = [g for g in games if not g.startswith("SC-")]
    tr, va = val_split(tgames)
    Xs, ys, ws, _ = load_rows(args.feats, args.labels, tr + va, cols, team_only=True,
                              spellw=(arm == "spellw"))
    ntr = len(tr)
    X = np.concatenate(Xs[:ntr])
    y = (np.concatenate(ys[:ntr]) == 0).astype(np.int8)
    w = np.concatenate(ws[:ntr])
    Xv = np.concatenate(Xs[ntr:])
    yv = (np.concatenate(ys[ntr:]) == 0).astype(np.int8)
    wv = np.concatenate(ws[ntr:])
    team, rt = fit_one(X, y, w, Xv, yv, wv, "TEAM")
    rt.update(train=tr, val=va)
    record.update(DEAD=rd, TEAM=rt)
    with open(os.path.join(out, "models.pkl"), "wb") as h:
        pickle.dump({"dead": dead, "team": team, "cols": cols}, h)
    with open(os.path.join(out, "record.json"), "w") as h:
        json.dump(record, h, indent=1, default=str)
    return 0


def predict(models, F, kind, flag=None):
    """Symmetric probabilities (N, 3). Models fitted with the source column (variants F2-F4)
    need `flag` (0 or 1): it is appended as the last column (kind "inv")."""
    cols = models["cols"]
    if models.get("fset", "full") != "full":
        perm, sign = pff.fset_swap(models["fset"])
        perm, sign = perm[cols], sign[cols]
        assert np.array_equal(cols, np.arange(len(perm)))
    else:
        perm, sign = pff.swap_table(kind)
    X = F[:, cols]
    if models.get("flag_column"):
        if flag is None:
            raise SystemExit("this model reads the source column: give the flag")
        X = np.concatenate([X, np.full((len(X), 1), float(flag), X.dtype)], axis=1)
        perm = np.concatenate([perm, [len(perm)]])
        sign = np.concatenate([sign, [1.0]]).astype(sign.dtype)
    Xs = X[:, perm] * sign
    nd = models["dead"].n_features_in_        # F6a: DEAD reads the leading F5T columns only
    pd_ = 0.5 * (models["dead"].predict_proba(X[:, :nd])[:, 1] +
                 models["dead"].predict_proba(Xs[:, :nd])[:, 1])
    ph = 0.5 * (models["team"].predict_proba(X)[:, 1] +
                1.0 - models["team"].predict_proba(Xs)[:, 1])
    return np.stack([(1 - pd_) * ph, (1 - pd_) * (1 - ph), pd_], axis=1)


def cmd_predict(args):
    arm = args.arm
    with open(os.path.join(args.models, arm, args.left_out, "models.pkl"), "rb") as h:
        models = pickle.load(h)
    out = os.path.join(args.out, arm)
    os.makedirs(out, exist_ok=True)
    games = [g for g in folds.ALL if folds.provider(g) == args.left_out]
    times = {}
    for g in games:
        F = np.load(os.path.join(args.feats, "eval", pff.check(g) + ".npy"))
        t0 = time.time()
        P = predict(models, F, col_kind(arm))
        times[g] = time.time() - t0
        np.save(os.path.join(out, g + ".npy"), P.astype(np.float64))
        log("{} {} predicted {:.1f}s".format(arm, g, times[g]))
    with open(os.path.join(out, "predict_time_{}.json".format(args.left_out)), "w") as h:
        json.dump(times, h, indent=1)
    return 0


# -- all-28 fits for the open-PFF test (coordinator, 13 Nov) ------------------------------------

VARIANTS = {
    # name: (SkillCorner in TEAM with possession-group labels, source column, short-spell
    #        TEAM weights, defect copies 0 and 1 only)
    "F1": (False, False, False, False),
    "F2": (True, True, False, False),
    "F3": (True, True, True, False),
    "F4": (True, True, False, True),
    "F5": (False, False, False, False),     # feature set F5 (pf_features.FSETS), F1's recipe
    # F5S (coordinator, de-starved check): F5 with SkillCorner in TEAM (possession groups,
    # quarter weight), no source column; used with --dead-from the F5 fit of the same split
    "F5S": (True, False, False, False),
    "F5T": (False, False, False, False),    # F5 on feature set F5T (base stage50-v2t)
    # Round 8, arm M (protocol R8 Section 3 row M): F5T's recipe on feature set F6 (F5T's
    # columns + engine.learned_pf_groups). F6a: the group columns in TEAM only (DEAD reads
    # F5T's columns, so it is F5T's DEAD: --dead-from the F5T fit of the same split gives the
    # same model); F6b: in TEAM and DEAD.
    "F6a": (False, False, False, False),
    "F6b": (False, False, False, False),
    # F6c (protocol R8 note 6.3, follow-up M2): F6a with the group columns in robust units per
    # match (rows built by m_groups --scale robust into their own feature folder)
    "F6c": (False, False, False, False),
}
VARIANT_FSET = {"F5": "F5", "F5S": "F5", "F5T": "F5T", "F6a": "F6", "F6b": "F6", "F6c": "F6"}
# Variants whose DEAD model reads only the first n columns of the feature set.
DEAD_NCOLS = {"F6a": "F5T", "F6c": "F5T"}


def load_rows_all(feats, labels, pg, games, cols, team=False, team_sc=False, flag=False,
                  spellw=False, defect_only=False):
    """Training rows of the all-28 fits. DFL rows come from the c_train_j input (build).
    team: alive rows only, y = 1 for home; SkillCorner team labels from the possession groups
    `pg` (its D frames are the same as the phases'). defect_only: the rows of defect copies 0
    (train/<g>.npz view 1) and 1 (train_aug1/<g>.npz) instead of clean view + copy 0."""
    Xs, ys, ws = [], [], []
    for g in games:
        assert not g.startswith("PFF-") and g in folds.ALL, g
        with np.load(os.path.join(feats, "train", pff.check(g) + ".npz")) as z:
            X, y, idx, view = z["X"][:, cols], z["y"].astype(np.int8), z["idx"], z["view"]
        if defect_only:
            keep = view == 1
            X, y, idx = X[keep], y[keep], idx[keep]
            with np.load(os.path.join(feats, "train_aug1", g + ".npz")) as z:
                X = np.concatenate([X, z["X"][:, cols]])
                y = np.concatenate([y, z["y"].astype(np.int8)])
                idx = np.concatenate([idx, z["idx"]])
        codes = None
        if team and g.startswith("SC-"):
            codes = pff.labels(pg, g)
            assert np.array_equal(codes[idx] == 2, y == 2), g
            y = codes[idx].astype(np.int8)
        if team:
            keep = y <= 1
            X, y, idx = X[keep], y[keep], idx[keep]
        if flag:
            X = np.concatenate([X, np.full((len(X), 1), 1.0 if g.startswith("SC-") else 0.0,
                                           X.dtype)], axis=1)
        w = np.ones(len(y))
        if team and spellw:
            w = spell_weights(g, codes if codes is not None else pff.labels(labels, g), idx)
        Xs.append(X)
        ys.append((y == 0) if team else (y == 2))
        ws.append(w)
    f = sc_factor(games, ys)
    for i, g in enumerate(games):
        if g.startswith("SC-"):
            ws[i] = ws[i] * f
    return Xs, [np.asarray(y, np.int8) for y in ys], ws, f


def fit_from_rows(Xs, ys, ws, ntr, name):
    X = np.concatenate(Xs[:ntr])
    y = np.concatenate(ys[:ntr])
    w = np.concatenate(ws[:ntr])
    Xv = np.concatenate(Xs[ntr:])
    yv = np.concatenate(ys[ntr:])
    wv = np.concatenate(ws[ntr:])
    Xs.clear()                     # the per-match arrays are no longer needed (memory)
    return fit_one(X, y, w, Xv, yv, wv, name)


def cmd_fit_all(args):
    team_sc, flag, spellw, defect = VARIANTS[args.variant]
    fset = VARIANT_FSET.get(args.variant, "full")
    cols = pff.ablation_columns("full") if fset == "full" else np.arange(len(pff.fset_spec(fset)))
    left = args.left_out
    out = os.path.join(args.out, args.variant, left)
    os.makedirs(out, exist_ok=True)
    games = [g for g in folds.ALL if left == "all" or folds.provider(g) != left]
    if args.games_out:
        games = [g for g in games if g not in args.games_out]
    record = {"variant": args.variant, "team_sc": team_sc, "flag": flag, "spellw": spellw,
              "defect_only": defect, "n_features": int(len(cols)) + int(flag),
              "params": dict(PARAMS, max_iter=MAX_ITER), "fset": fset, "left_out": left,
              "games_out": args.games_out or [], "training_games": games}
    dead_cols = cols
    if args.variant in DEAD_NCOLS:
        dead_cols = np.arange(len(pff.fset_spec(DEAD_NCOLS[args.variant])))
        record["dead_n_features"] = int(len(dead_cols))
    if args.dead_from:
        with open(os.path.join(args.dead_from, "models.pkl"), "rb") as h:
            dead = pickle.load(h)["dead"]
        assert dead.n_features_in_ == len(dead_cols), (dead.n_features_in_, len(dead_cols))
        rd = json.load(open(os.path.join(args.dead_from, "record.json")))["DEAD"]
        rd["copied_from"] = args.dead_from
    else:
        tr, va = val_split(games)
        Xs, ys, ws, f = load_rows_all(args.feats, args.labels, args.pg, tr + va, dead_cols,
                                      flag=flag, defect_only=defect)
        dead, rd = fit_from_rows(Xs, ys, ws, len(tr), "DEAD")
        rd.update(train=tr, val=va, sc_row_factor=f)
        del Xs
    tgames = games if team_sc else [g for g in games if not g.startswith("SC-")]
    tr, va = val_split(tgames)
    Xs, ys, ws, f = load_rows_all(args.feats, args.labels, args.pg, tr + va, cols, team=True,
                                  team_sc=team_sc, flag=flag, spellw=spellw, defect_only=defect)
    team, rt = fit_from_rows(Xs, ys, ws, len(tr), "TEAM")
    rt.update(train=tr, val=va, sc_row_factor=f)
    record.update(DEAD=rd, TEAM=rt)
    with open(os.path.join(out, "models.pkl"), "wb") as h:
        pickle.dump({"dead": dead, "team": team, "cols": cols, "kind": "full",
                     "flag_column": flag, "fset": fset, "dead_cols": dead_cols}, h)
    with open(os.path.join(out, "record.json"), "w") as h:
        json.dump(record, h, indent=1, default=str)
    open(os.path.join(out, "DONE"), "w").close()
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "predict"):
        s = sub.add_parser(name)
        s.add_argument("--feats", required=True)
        s.add_argument("--left-out", choices=("metrica", "dfl", "skillcorner"), required=True)
        s.add_argument("--arm", choices=ARMS, default="full")
        s.add_argument("--out", required=True)
        if name == "fit":
            s.add_argument("--labels", required=True)
        else:
            s.add_argument("--models", required=True)
    a = sub.add_parser("fit-all")
    a.add_argument("--feats", required=True)
    a.add_argument("--labels", required=True)
    a.add_argument("--pg", required=True, help="SkillCorner possession-group label folder")
    a.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    a.add_argument("--dead-from", default=None, help="reuse the DEAD model of another fit "
                   "(same rows and weights, e.g. F3 from F2)")
    a.add_argument("--out", required=True)
    a.add_argument("--left-out", default="all", choices=("all", "metrica", "dfl", "skillcorner"))
    a.add_argument("--games-out", nargs="+", default=None,
                   help="matches left out (out-of-fold fits: the fold's test matches)")
    args = p.parse_args(argv)
    return {"fit": cmd_fit, "predict": cmd_predict, "fit-all": cmd_fit_all}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
