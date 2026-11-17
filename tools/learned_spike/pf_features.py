"""Per-frame (25 fps) features for the gradient-boosted spike (round 6). Not part of the submission.

    # pf venv (numpy, pandas, scikit-learn), OMP_NUM_THREADS small, nice 10
    python -m tools.learned_spike.pf_features build --labels CA --engine-cache DIR --out OUT [--games G ...]
    python -m tools.learned_spike.pf_features symtest --engine-cache DIR --game METRICA-S1

Columns (feature_spec()), all at 25 fps, computed per match view from the parsed tracking:

a. the 50 columns of engine.learned_features.frame_features (stage50-v2) before any binning;
b. touch columns from the rule engine's contact detector, engine.events.event_table(tactical)
   (tracks.prepare + detect_events, default Config, label-free). A contact is any row of the
   table (touch, release, contested, control, dribble) at its frame; its team is +1 home, -1
   away, 0 contested (several rows on one frame: their common team, else 0). Contacts are
   counted inside the graded block of the frame only (reset at each block start):
     tl_team   team of the last contact at or before t (0 if none)
     tl_since  log1p(seconds since it), capped at 60 s (cap if none)
     tn_team   team of the next contact after t (0 if none)
     tn_until  log1p(seconds until it), capped at 60 s
     t_differ  1 if both exist and their teams differ
     t_n2, t_n5  contacts within +-2 s and +-5 s
     t_streak  length of the run of consecutive contacts of the last contact's team ending at
               the last contact (0 if the last is contested or none), capped at 10
c. context: for CTX_BASE (ball visibility both forms, ball speed, nearest distance per team and
   its difference, counts within 3 m per team, mean player speed, still share) and every touch
   column, centred means over +-0.5 s, +-1.5 s, +-4.5 s (12, 37, 112 frames each side) and
   one-sided means over the past [t-w, t-1] and the next [t+1, t+w] for w = 1 s, 3 s; the window
   is cut at the graded block edges (never across a half or block boundary); an empty one-sided
   window (block edge) takes the value at t. Frames outside the blocks are 0.
d. ball missing (ball_raw_vis) and t_since / t_until of the ball are kept as in a.
e. proximity touches (coordinator, 13 Nov): a frame where the ball is visible (raw and cleaned)
   and the nearest player of a team is within 1.8 m (H_d, A_d of a); team = the nearer team.
   The same eight columns as b, prefixed "p" (ptl_team ... pt_streak): last / next team and
   time use every proximity frame (so ptl_since is 0 inside a run); the counts and the streak
   count runs (the first frame of each run of consecutive proximity frames of one team). Their
   context columns are built as in c. Order: a, b, e, then the context of CTX_BASE, b, e.

Symmetry: every column has a kind as in learned_features ("inv", "neg", "x", "y", or a team
prefix H/A plus an axis). A context column inherits the kind of its source; touch team columns
are "neg". swap() = the column permutation and signs of learned_features._symmetry_tables on
this spec (H <-> A, pitch turned 180 degrees). symtest checks it on real tracking: features of
the tracking with home and away exchanged against swap(features).

Views: eval_view (Metrica, SkillCorner raw; DFL input "c", still ball removed) and the training
views: the tracking as provided (DFL: r5_inputs input "c_train_j") and defect copy 0 of
augment.augment(version "r3") made from that tracking. Training rows: every STRIDE-th graded
frame (label H, A or D inside a graded block) from a per-match offset crc32(game) % STRIDE.

Never reads a competition, sealed or PFF match (folds.check, FORBID). Python 3.9.
"""

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from engine import data, events  # noqa: E402
from engine import learned_features as lf  # noqa: E402
from tools.learned_spike import folds  # noqa: E402

FPS = 25
STRIDE = 8
CAP_S = 60.0
FORBID = ("BAR", "ATH", "J03WQQ", "J03WR9", "PFF")
TOUCH = [("tl_team", "neg"), ("tl_since", "inv"), ("tn_team", "neg"), ("tn_until", "inv"),
         ("t_differ", "inv"), ("t_n2", "inv"), ("t_n5", "inv"), ("t_streak", "inv")]
PROX = [("p" + n, k) for n, k in TOUCH]
PROX_M = 1.8
CTX_BASE = ["ball_vis", "ball_raw_vis", "bspeed", "H_d", "A_d", "near_diff", "H_n30", "A_n30",
            "meanv_all", "still_all"]
CENTRED = (("c05", 12), ("c15", 37), ("c45", 112))
ONESIDED = (("p1", 25), ("p3", 75), ("n1", 25), ("n3", 75))


def check(game):
    """External matches; and the 43 open PFF matches as a TEST only (owner's decision of
    13 Nov: never the 15 sealed or the 4 pilot matches; never trained on, see pf_train)."""
    if game.startswith("PFF-"):
        folds.assert_not_sealed([game])
        if game not in folds.PFF_TRAIN:
            raise SystemExit("{}: not one of the 43 open PFF matches".format(game))
        return game
    assert not any(f in game.upper() for f in FORBID), game
    return folds.check(game)


def feature_spec(touch=True, context=True, prox=True):
    spec = list(lf.SPEC)
    if touch:
        spec += TOUCH
    if prox:
        spec += PROX
    if context:
        kinds = dict(spec)
        src = CTX_BASE + ([n for n, _ in TOUCH] if touch else []) + (
            [n for n, _ in PROX] if prox else [])
        for name in src:
            for suf, _ in CENTRED + ONESIDED:
                spec.append(("{}__{}".format(name, suf), kinds[name]))
    return spec


SPEC = feature_spec()
NAMES = [n for n, _ in SPEC]
NF = len(SPEC)
PERM, SWAP_SIGN, _MIRROR = lf._symmetry_tables(SPEC)


def swap(F):
    """The team-swapped copy of a (..., NF) matrix (H <-> A, signed columns negated)."""
    return F[..., PERM] * SWAP_SIGN


def columns(names_subset):
    return np.array([NAMES.index(n) for n in names_subset], np.int64)


def ablation_columns(kind):
    """Column indices kept by an ablation: 'full', 'no_touch' (b and their context gone),
    'no_context' (c gone)."""
    if kind == "full":
        return np.arange(NF)
    sub = feature_spec(touch=(kind != "no_touch"), context=(kind != "no_context"),
                       prox=(kind != "no_prox"))
    return columns([n for n, _ in sub])


def swap_table(kind):
    """(perm, sign) of the swap inside the columns of an ablation (indices into the subset)."""
    cols = ablation_columns(kind)
    pos = {c: i for i, c in enumerate(cols)}
    return np.array([pos[PERM[c]] for c in cols]), SWAP_SIGN[cols]


# -- touch columns -----------------------------------------------------------------------------

def contacts(tac):
    """(frames int64 sorted unique, team float +1/-1/0) of the contact detector's rows."""
    table = events.event_table(tac)
    if len(table) == 0:
        return np.zeros(0, np.int64), np.zeros(0)
    fr = table["frame"].to_numpy(np.int64)
    tm = np.array([1.0 if t == "H" else (-1.0 if t == "A" else 0.0) for t in table["team"]])
    uf = np.unique(fr)
    team = np.zeros(len(uf))
    for i, f in enumerate(uf):
        vals = set(tm[fr == f].tolist())
        team[i] = vals.pop() if len(vals) == 1 else 0.0
    return uf, team


def touch_block(n, cf, ct, ef=None, et=None):
    """(n, 8) touch columns for one block of n frames, contacts at local frames cf (sorted).
    ef, et (default cf, ct): the events counted by t_n2, t_n5 and t_streak."""
    if ef is None:
        ef, et = cf, ct
    out = np.zeros((n, len(TOUCH)), np.float32)
    t = np.arange(n)
    cap = np.log1p(CAP_S)
    if len(cf) == 0:
        out[:, 1] = cap
        out[:, 3] = cap
        return out
    il = np.searchsorted(cf, t, side="right") - 1          # last contact <= t
    inn = np.searchsorted(cf, t, side="right")             # next contact > t
    has_l = il >= 0
    has_n = inn < len(cf)
    lt = np.where(has_l, ct[np.clip(il, 0, None)], 0.0)
    nt = np.where(has_n, ct[np.clip(inn, None, len(cf) - 1)], 0.0)
    since = np.where(has_l, (t - cf[np.clip(il, 0, None)]) / FPS, CAP_S)
    until = np.where(has_n, (cf[np.clip(inn, None, len(cf) - 1)] - t) / FPS, CAP_S)
    out[:, 0] = lt
    out[:, 1] = np.log1p(np.minimum(since, CAP_S))
    out[:, 2] = nt
    out[:, 3] = np.log1p(np.minimum(until, CAP_S))
    out[:, 4] = (has_l & has_n & (lt != nt)).astype(np.float32)
    for col, w in ((5, 2 * FPS), (6, 5 * FPS)):
        out[:, col] = np.searchsorted(ef, t + w, side="right") - np.searchsorted(ef, t - w,
                                                                                 side="left")
    if len(ef) == 0:
        return out
    streak = np.zeros(len(ef))
    for i in range(len(ef)):
        if et[i] == 0:
            streak[i] = 0
        elif i > 0 and et[i - 1] == et[i]:
            streak[i] = streak[i - 1] + 1
        else:
            streak[i] = 1
    ie = np.searchsorted(ef, t, side="right") - 1
    out[:, 7] = np.where(ie >= 0, np.minimum(streak[np.clip(ie, 0, None)], 10), 0)
    return out


def proximity(base):
    """(frames, team, run-start frames, run-start teams) of the proximity touches: ball visible
    (raw and cleaned) and the nearest player of a team within PROX_M metres; the team is the
    nearer one (+1 home, -1 away, 0 on an exact tie). Events for counts and streak: the first
    frame of each run of consecutive proximity frames of one team."""
    names = lf.feature_names()
    col = {n: base[:, names.index(n)] for n in ("ball_vis", "ball_raw_vis", "H_d", "A_d")}
    hd, ad = col["H_d"] * 10.0, col["A_d"] * 10.0
    vis = (col["ball_vis"] > 0.5) & (col["ball_raw_vis"] > 0.5)
    on = vis & (np.minimum(hd, ad) <= PROX_M)
    team = np.where(hd < ad, 1.0, np.where(ad < hd, -1.0, 0.0))
    fr = np.flatnonzero(on)
    tm = team[fr]
    start = np.ones(len(fr), bool)
    start[1:] = (np.diff(fr) > 1) | (tm[1:] != tm[:-1])
    return fr, tm, fr[start], tm[start]


# -- context -----------------------------------------------------------------------------------

def window_means(x, lo, hi):
    """Mean of x over [t+lo, t+hi] cut to [0, n); x (n, k). Empty window -> x[t]."""
    n = x.shape[0]
    cs = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0, dtype=np.float64)])
    t = np.arange(n)
    a = np.clip(t + lo, 0, n)
    b = np.clip(t + hi + 1, 0, n)
    cnt = (b - a)[:, None]
    s = cs[b] - cs[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(cnt > 0, s / np.maximum(cnt, 1), x)
    return m.astype(np.float32)


def context_block(x):
    res = []
    for _, w in CENTRED:
        res.append(window_means(x, -w, w))
    for suf, w in ONESIDED:
        res.append(window_means(x, -w, -1) if suf[0] == "p" else window_means(x, 1, w))
    # order: per source column, the 7 windows (feature_spec order)
    stack = np.stack(res, axis=2)            # (n, k, 7)
    return stack.reshape(x.shape[0], -1)


# Feature sets: name -> (base feature version, engine-contact columns b on/off).
# "full": the round 6 spike (248 columns). "F5" (coordinator, 13 Nov): base columns of
# "stage50-v2s" (exactly team-symmetric, as the network it is averaged with), NO engine-contact
# columns (b) nor their context (the engine must not depend on the contact detector); the
# proximity touches (e) and the other context columns are kept (240 - 8 - 56 = 184 columns).
FSETS = {"full": (lf.FEATURE_VERSION, True), "F5": ("stage50-v2s", False),
         # F5T (coordinator, 14 Nov): F5 with the base columns of stage50-v2t (attack
         # direction from the home median against the away median)
         "F5T": ("stage50-v2t", False),
         # F6 (round 8, arm M): F5T's columns, then the group columns of
         # engine.learned_pf_groups (rows built by r8_scripts/m_groups.py, not by compute())
         "F6": ("stage50-v2t", False)}


def fset_spec(fset):
    spec = feature_spec(touch=FSETS[fset][1])
    if fset == "F6":
        from engine import learned_pf_groups as lpg
        spec = spec + list(lpg.GROUP_SPEC)
    return spec


def fset_names(fset):
    return [n for n, _ in fset_spec(fset)]


def fset_swap(fset):
    """(perm, sign) of the team swap on the columns of a feature set."""
    perm, sign, _ = lf._symmetry_tables(fset_spec(fset))
    return perm, sign


def compute(tac, game, fset="full"):
    """float32 (N, columns of the feature set) per-frame features of one match view."""
    check(game)
    version, with_touch = FSETS[fset]
    if version == lf.FEATURE_VERSION:
        base = lf.frame_features(tac, game)
    else:
        base = lf.frame_features(tac, game, version=version)
    n = base.shape[0]
    blocks = data.graded_blocks(game)
    touch = np.zeros((n, len(TOUCH)), np.float32)
    if with_touch:
        cf_all, ct_all = contacts(tac)
        for _, first, last in blocks:
            sel = (cf_all >= first) & (cf_all <= last)
            touch[first:last + 1] = touch_block(last - first + 1, cf_all[sel] - first,
                                                ct_all[sel])
    pf, pt, pef, pet = proximity(base)
    prox = np.zeros((n, len(PROX)), np.float32)
    for _, first, last in blocks:
        s1 = (pf >= first) & (pf <= last)
        s2 = (pef >= first) & (pef <= last)
        prox[first:last + 1] = touch_block(last - first + 1, pf[s1] - first, pt[s1],
                                           pef[s2] - first, pet[s2])
    if with_touch:
        head = np.concatenate([base, touch, prox], axis=1)
        src = columns(CTX_BASE + [nm for nm, _ in TOUCH] + [nm for nm, _ in PROX])
    else:
        head = np.concatenate([base, prox], axis=1)
        hn = lf.feature_names() + [nm for nm, _ in PROX]
        src = np.array([hn.index(c) for c in CTX_BASE + [nm for nm, _ in PROX]])
    ctx = np.zeros((n, len(src) * (len(CENTRED) + len(ONESIDED))), np.float32)
    for _, first, last in blocks:
        ctx[first:last + 1] = context_block(head[first:last + 1][:, src].astype(np.float64))
    F = np.concatenate([head, ctx], axis=1).astype(np.float32)
    assert F.shape[1] == len(fset_spec(fset)), (F.shape, len(fset_spec(fset)))
    return F


# -- views -------------------------------------------------------------------------------------

def raw_tac(game, engine_cache):
    from tools.learned_spike import build
    return build.load_view(check(game), "raw", engine_cache)


def eval_view(game, engine_cache):
    from tools.learned_spike import r4_inputs
    tac = raw_tac(game, engine_cache)
    if game.startswith("DFL-"):
        tac, _ = r4_inputs.load_dfl(tac, game, "c")
    return tac


def train_base(game, engine_cache):
    from tools.learned_spike import r5_inputs
    tac = raw_tac(game, engine_cache)
    if game.startswith("DFL-"):
        tac, _ = r5_inputs.load_dfl(tac, game, "c_train_j")
    return tac


def aug_copy(tac, codes, game, k=0):
    from tools.learned_spike import augment
    out, _ = augment.augment(tac, codes, game, k, data.graded_blocks(game), data.pitch_m(game),
                             version="r3")
    return out


def cmd_build_aug(args):
    """Training rows of defect copy k (same rows as build) into OUT/train_aug<k>/<game>.npz."""
    k = int(args.k)
    for game in args.games or list(folds.ALL):
        assert not game.startswith("PFF-")
        check(game)
        path = os.path.join(args.out, "train_aug{}".format(k), game + ".npz")
        if os.path.isfile(path) and not args.force:
            continue
        t0 = time.time()
        codes = labels(args.labels, game)
        idx = train_rows(game, codes)
        Fa = compute(aug_copy(train_base(game, args.engine_cache), codes, game, k), game)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path + ".tmp.npz", X=Fa[idx], y=codes[idx], idx=idx)
        os.replace(path + ".tmp.npz", path)
        print("{} aug k{} {:.1f}s".format(game, k, time.time() - t0), file=sys.stderr,
              flush=True)
    return 0


def labels(folder, game):
    with np.load(os.path.join(folder, "labels", check(game) + ".npz")) as z:
        return z["raw"].astype(np.int8)


def train_rows(game, codes):
    graded = np.zeros(len(codes), bool)
    for _, first, last in data.graded_blocks(game):
        graded[first:last + 1] = True
    graded &= codes <= 2
    off = zlib.crc32(game.encode()) % STRIDE
    idx = np.flatnonzero(graded)
    return idx[off::STRIDE]


def build_game(game, args):
    out = args.out
    epath = os.path.join(out, "eval", game + ".npy")
    tpath = os.path.join(out, "train", game + ".npz")
    if os.path.isfile(epath) and os.path.isfile(tpath) and not args.force:
        return None
    codes = labels(args.labels, game)
    tm = {}
    t0 = time.time()
    tac_e = eval_view(game, args.engine_cache)
    Fe = compute(tac_e, game, args.fset)
    tm["eval_view_s"] = time.time() - t0
    assert Fe.shape[0] == len(codes), (game, Fe.shape, len(codes))
    idx = train_rows(game, codes)
    if game.startswith("DFL-"):
        t0 = time.time()
        tac_t = train_base(game, args.engine_cache)
        Ft = compute(tac_t, game, args.fset)
        tm["train_view_s"] = time.time() - t0
        X0 = Ft[idx]
        del Ft
    else:
        tac_t = tac_e
        X0 = Fe[idx]
    t0 = time.time()
    Fa = compute(aug_copy(tac_t, codes, game), game, args.fset)
    tm["aug_view_s"] = time.time() - t0
    X1 = Fa[idx]
    del Fa
    os.makedirs(os.path.dirname(epath), exist_ok=True)
    os.makedirs(os.path.dirname(tpath), exist_ok=True)
    np.save(epath + ".tmp.npy", Fe)
    os.replace(epath + ".tmp.npy", epath)
    np.savez(tpath + ".tmp.npz", X=np.concatenate([X0, X1]), y=np.concatenate(
        [codes[idx], codes[idx]]), idx=np.concatenate([idx, idx]),
             view=np.concatenate([np.zeros(len(idx), np.int8), np.ones(len(idx), np.int8)]))
    os.replace(tpath + ".tmp.npz", tpath)
    tm.update(frames=int(len(codes)), train_rows=int(2 * len(idx)))
    with open(os.path.join(out, "timing", game + ".json"), "w") as h:
        json.dump(tm, h, indent=1)
    return tm


def cmd_build(args):
    os.makedirs(os.path.join(args.out, "timing"), exist_ok=True)
    with open(os.path.join(args.out, "spec.json"), "w") as h:
        json.dump({"fset": args.fset, "base_version": FSETS[args.fset][0],
                   "names": fset_names(args.fset),
                   "kinds": [k for _, k in fset_spec(args.fset)], "stride": STRIDE}, h)
    games = args.games or list(folds.ALL)
    for game in games:
        check(game)
        t0 = time.time()
        tm = build_game(game, args)
        print("{} {} {:.1f}s".format(game, "skipped" if tm is None else tm, time.time() - t0),
              file=sys.stderr, flush=True)
    return 0


def swapped_tracking(tac):
    """The tracking with home and away exchanged (slots relabelled; coordinates as they are:
    the orientation step turns the pitch so the new home team attacks +x)."""
    team = np.asarray(tac["team_of_slot"])
    names = tac.get("team_names")
    out = dict(tac, team_of_slot=np.where(team == 0, 1, np.where(team == 1, 0, team)).astype(
        team.dtype))
    if names is not None and len(names) == 2:
        out["team_names"] = type(names)([names[1], names[0]]) if not isinstance(
            names, dict) else names
    return out


def cmd_symtest(args):
    game = check(args.game)
    tac = eval_view(game, args.engine_cache)
    F = compute(tac, game, args.fset)
    G = compute(swapped_tracking(tac), game, args.fset)
    perm_f, sign_f = fset_swap(args.fset)

    def swap(M):
        return M[..., perm_f] * sign_f
    S = swap(F)
    # involution and table checks
    assert np.array_equal(swap(swap(F)), F), "swap is not an involution"
    blocks = data.graded_blocks(game)
    graded = np.zeros(len(F), bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    d = np.abs(np.nan_to_num(G[graded]) - np.nan_to_num(S[graded]))
    per = d.max(axis=0)
    share = (d > 1e-4).mean(axis=0)
    nf = F.shape[1]
    names = fset_names(args.fset)
    groups = ({"a_base50": slice(0, 50), "b_touch": slice(50, 58), "d_prox": slice(58, 66),
               "c_context": slice(66, nf)} if FSETS[args.fset][1] else
              {"a_base50": slice(0, 50), "e_prox": slice(50, 58), "c_context": slice(58, nf)})
    rep = {"game": game, "fset": args.fset, "nf": nf, "involution": True,
           "max_abs_diff": {k: float(per[s].max()) for k, s in groups.items()},
           "share_frames_diff_gt_1e-4": {k: float(share[s].max()) for k, s in groups.items()},
           "worst_columns": [(names[i], float(per[i]), float(share[i]))
                             for i in np.argsort(-share)[:10]],
           "pairs_checked": [(names[i], names[perm_f[i]], float(sign_f[i]))
                             for i in (0, 4, 14, 18, 50, 52, names.index("H_d__c05"))]}
    print(json.dumps(rep, indent=1))
    if args.out:
        with open(args.out, "w") as h:
            json.dump(rep, h, indent=1)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--labels", required=True)
    b.add_argument("--engine-cache", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--games", nargs="+", default=None)
    b.add_argument("--force", action="store_true")
    b.add_argument("--fset", choices=sorted(FSETS), default="full")
    a = sub.add_parser("build-aug")
    a.add_argument("--labels", required=True)
    a.add_argument("--engine-cache", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--k", type=int, required=True)
    a.add_argument("--games", nargs="+", default=None)
    a.add_argument("--force", action="store_true")
    s = sub.add_parser("symtest")
    s.add_argument("--engine-cache", required=True)
    s.add_argument("--game", default="METRICA-S1")
    s.add_argument("--out", default=None)
    s.add_argument("--fset", choices=sorted(FSETS), default="full")
    args = p.parse_args(argv)
    return {"build": cmd_build, "symtest": cmd_symtest, "build-aug": cmd_build_aug}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
