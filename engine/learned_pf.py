"""The per-frame member of the learned stage: gradient-boosted trees at 25 fps, numpy only.

Config.LEARNED_PF "on" averages its frame probabilities with the network's (engine.learned):
p = w p_pf + (1 - w) p_network, w = Config.LEARNED_PF_WEIGHT. This module computes the
member's features and its probabilities; it reads the tracking only, never a label.

Features (pf_features(); the definitions of the round 6 per-frame spike,
tools/learned_spike/pf_features.py, rewritten here because the engine imports nothing from
tools/). At 25 fps, float32 as the models saw them in training:

a. the 50 columns of engine.learned_features.frame_features() in the base version the weights
   file names (stage50-v2s for the shipped spec; stage50-v2 for the spike's first fits);
e. proximity touches: a frame where the ball is visible (ball_vis and ball_raw_vis of a) and the
   nearest player of a team is within PROX_M metres (H_d, A_d of a, times 10); the team is the
   nearer one (+1 home, -1 away, 0 on an exact tie). Eight columns per graded block (reset at
   each block start, 0 outside the blocks except the "since"/"until" caps):
     ptl_team   team of the last proximity frame at or before t (0 if none)
     ptl_since  log1p(seconds since it), capped at CAP_S
     ptn_team   team of the next proximity frame after t (0 if none)
     ptn_until  log1p(seconds until it), capped at CAP_S
     pt_differ  1 if both exist and their teams differ
     pt_n2, pt_n5  number of proximity runs starting within +-2 s and +-5 s (a run is a stretch
                of consecutive proximity frames of one team; it counts at its first frame)
     pt_streak  the number of consecutive runs of the last run's team up to the last run
                (0 if that run is a tie or there is none), capped at 10
c. context: for CTX_BASE and the eight proximity columns, centred means over +-12, +-37 and
   +-112 frames and one-sided means over the past [t-w, t-1] and the next [t+1, t+w] for
   w = 25, 75 frames, cut at the graded block edges (never across a block); an empty one-sided
   window takes the value at t; frames outside the blocks are 0. Means: float64 cumulative sums
   of the float32 columns (np.cumsum along time is a sequential sum, the same bits on every
   IEEE platform), divided by the window length, stored as float32.

g. member spec F6 (round 8): the group columns of engine.learned_pf_groups (ball against each
   team's block, block motion, block area, sharpness), after all of the above in a file's
   column list (member_spec()). They are computed only when a weights file lists one of them,
   so a file without them (learned_pf_v7.npz) gets exactly the features it got before.

The spike's contact columns (tl_team ... t_streak, from the rule engine's event table) and their
context are NOT computed: a weights file that lists one is refused (load_pf()).

Models: two binary tree ensembles, DEAD (dead ball) and TEAM (home given alive), in flat arrays
(per node: feature, threshold, left, right, missing goes left, is leaf, value; per tree: root;
per model: baseline). predict_raw() is the prediction of scikit-learn's
HistGradientBoostingClassifier: start at the baseline and add, tree after tree in file order,
the value of the leaf the row reaches (x <= threshold goes left, a NaN goes where
missing_go_to_left says); the same float64 additions in the same order, so the same bits.
The tree walk is vectorised over frames: per tree, only the frames that have not reached a
leaf move on, level by level.

Team symmetry. Each model is evaluated on the features F of the tracking as given and on a
team-swapped copy G. The weights file names how G is made ("swap_mode"):

- "columns" (the default, and how the spike measured the member): G = swap(F), the column
  permutation and sign changes of swap_columns(). Exactly as evaluated; the output is
  team-swap symmetric only as far as the features are (see the caveat below).
- "tracking": G = the features of the tracking with home and away exchanged
  (swapped_tactical(): the slots relabelled, the coordinates as they are; the orientation step
  then turns the pitch). The output is exactly symmetric, bit for bit:
  p(D) = (DEAD(F) + DEAD(G)) / 2,  q = (TEAM(F) - TEAM(G)) / 2,
  p(H) = (1 - p(D)) (1/2 + q),  p(A) = (1 - p(D)) (1/2 - q).
Under a team swap of the input, F and G exchange exactly, so p(D) is unchanged bit for bit and
p(H), p(A) exchange bit for bit (a - b = -(b - a) and x + y = y + x hold exactly in IEEE
arithmetic). The spike evaluated swap(F), the column-permuted and sign-changed copy of F
(swap_columns()), in place of G; the two agree up to the float rounding of the features
(swap_mode "columns"). Caveat, measured on the 28 external matches: the base features'
orientation rule (engine.learned_features.home_attacks_positive: home attacks +x when the home
team's median x over the half is below the centre) is not team-swap equivariant when both
teams' medians fall on the same side of the centre (13 of the 28 matches have such a half,
mostly SkillCorner); there the two modes differ by much more than rounding (SC-1874553: 47 % of
frames above 1e-4). Engine.learned uses the same mode for the network's swapped copies.

The probabilities are rounded to PROB_DECIMALS (4) once, as engine.learned rounds the network's
(math: the sigmoid's exp may differ in its last bit between C libraries; rounding keeps that from
reaching the decoder unless a value lies on a rounding boundary).

Python 3.9, numpy only. No randomness, no clock, no threads.
"""

import hashlib

import numpy as np

from engine import data
from engine import learned_features as lf
from engine import learned_pf_groups as lpg

PF_FORMAT = "learned-pf-trees-v1"
FPS = 25
CAP_S = 60.0
PROX_M = 1.8
STREAK_CAP = 10
PROB_DECIMALS = 4
CONTACT = [("tl_team", "neg"), ("tl_since", "inv"), ("tn_team", "neg"), ("tn_until", "inv"),
           ("t_differ", "inv"), ("t_n2", "inv"), ("t_n5", "inv"), ("t_streak", "inv")]
PROX = [("p" + n, k) for n, k in CONTACT]
CTX_BASE = ["ball_vis", "ball_raw_vis", "bspeed", "H_d", "A_d", "near_diff", "H_n30", "A_n30",
            "meanv_all", "still_all"]
CENTRED = (("c05", 12), ("c15", 37), ("c45", 112))
ONESIDED = (("p1", 25), ("p3", 75), ("n1", 25), ("n3", 75))
MODELS = ("dead", "team")
SWAP_MODES = ("columns", "tracking")
NODE_ARRAYS = ("feature", "threshold", "left", "right", "missing_left", "leaf", "value")


# -- the column spec ----------------------------------------------------------------------------

def full_spec(base_version=lf.FEATURE_VERSION):
    """[(name, kind)] of every column of the spike (base, contact, proximity, context), in the
    spike's order; kinds as engine.learned_features (context columns inherit their source's)."""
    spec = list(lf.feature_spec(base_version)) + CONTACT + PROX
    kinds = dict(spec)
    for name in CTX_BASE + [n for n, _ in CONTACT] + [n for n, _ in PROX]:
        for suf, _ in CENTRED + ONESIDED:
            spec.append(("{}__{}".format(name, suf), kinds[name]))
    return spec


def member_spec(base_version=lf.FEATURE_VERSION):
    """full_spec() and then the group columns of member spec F6 (engine.learned_pf_groups,
    round 8): every column a weights file may list."""
    return full_spec(base_version) + list(lpg.GROUP_SPEC)


CONTACT_NAMES = set(n for n, _ in CONTACT)


def computable(name):
    """True for a column this module computes (not a contact column or its context)."""
    return name.split("__")[0] not in CONTACT_NAMES


def swap_columns(columns, base_version=lf.FEATURE_VERSION):
    """(perm, sign) inside `columns`: column i of the swapped copy is sign[i] x column perm[i].
    Raise ValueError if the columns are not closed under the team swap."""
    spec = member_spec(base_version)
    names = [n for n, _ in spec]
    perm, sign, _ = lf._symmetry_tables(spec)
    pos = {n: i for i, n in enumerate(columns)}
    out_perm, out_sign = [], []
    for n in columns:
        j = names.index(n)
        partner = names[perm[j]]
        if partner not in pos:
            raise ValueError("column {} needs its swap partner {}".format(n, partner))
        out_perm.append(pos[partner])
        out_sign.append(sign[j])
    return np.array(out_perm, np.int64), np.array(out_sign, np.float32)


# -- features -----------------------------------------------------------------------------------

def touch_block(n, cf, ct, ef, et):
    """(n, 8) float32 columns of one block of n frames: events at local frames cf (sorted, team
    ct) for last/next, run starts ef (team et) for the counts and the streak."""
    out = np.zeros((n, len(PROX)), np.float32)
    t = np.arange(n)
    cap = np.log1p(CAP_S)
    if len(cf) == 0:
        out[:, 1] = cap
        out[:, 3] = cap
        return out
    il = np.searchsorted(cf, t, side="right") - 1
    inn = np.searchsorted(cf, t, side="right")
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
        out[:, col] = (np.searchsorted(ef, t + w, side="right")
                       - np.searchsorted(ef, t - w, side="left"))
    if len(ef) == 0:
        return out
    # Streak of each run start: consecutive starts of the same non-zero team.
    m = len(ef)
    new = np.ones(m, bool)
    new[1:] = et[1:] != et[:-1]
    start = np.maximum.accumulate(np.where(new, np.arange(m), 0))
    streak = (np.arange(m) - start + 1).astype(np.float64)
    streak[et == 0] = 0.0
    ie = np.searchsorted(ef, t, side="right") - 1
    out[:, 7] = np.where(ie >= 0, np.minimum(streak[np.clip(ie, 0, None)], STREAK_CAP), 0)
    return out


def proximity(base, base_version):
    """(frames, team, run-start frames, run-start teams) of the proximity touches."""
    names = lf.feature_names(base_version)
    col = {n: base[:, names.index(n)] for n in ("ball_vis", "ball_raw_vis", "H_d", "A_d")}
    hd, ad = col["H_d"] * 10.0, col["A_d"] * 10.0       # float32, as the spike computed them
    vis = (col["ball_vis"] > 0.5) & (col["ball_raw_vis"] > 0.5)
    on = vis & (np.minimum(hd, ad) <= PROX_M)
    team = np.where(hd < ad, 1.0, np.where(ad < hd, -1.0, 0.0))
    fr = np.flatnonzero(on)
    tm = team[fr]
    start = np.ones(len(fr), bool)
    start[1:] = (np.diff(fr) > 1) | (tm[1:] != tm[:-1])
    return fr, tm, fr[start], tm[start]


def window_means(x, lo, hi):
    """float32 mean of x (n, k) over [t + lo, t + hi] cut to [0, n); an empty window -> x[t]."""
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
    """(n, 7 k) context of the block's source columns x (n, k) float64: per source column the
    windows c05, c15, c45, p1, p3, n1, n3."""
    res = [window_means(x, -w, w) for _, w in CENTRED]
    for suf, w in ONESIDED:
        res.append(window_means(x, -w, -1) if suf[0] == "p" else window_means(x, 1, w))
    return np.stack(res, axis=2).reshape(x.shape[0], -1)


def pf_features(tac, game, columns, base_version=lf.FEATURE_VERSION, cfg=None,
                group_scale=lpg.SCALE_RAW):
    """float32 (N, len(columns)) per-frame features of one match, in the order of `columns`.

    Raise ValueError for a column this module does not compute (contact columns).
    `cfg` is passed to frame_features() (ball cleaning fields; None = Config.default()).
    """
    bad = [c for c in columns if not computable(c)]
    if bad:
        raise ValueError("columns the engine does not compute (contact detector): {}".format(
            bad[:5]))
    lf.check_version(base_version)
    base = lf.frame_features(tac, game, cfg=cfg, version=base_version)
    n = base.shape[0]
    blocks = data.graded_blocks(game)
    pf, pt, pef, pet = proximity(base, base_version)
    prox = np.zeros((n, len(PROX)), np.float32)
    for _, first, last in blocks:
        s1 = (pf >= first) & (pf <= last)
        s2 = (pef >= first) & (pef <= last)
        prox[first:last + 1] = touch_block(last - first + 1, pf[s1] - first, pt[s1],
                                           pef[s2] - first, pet[s2])
    base_names = lf.feature_names(base_version)
    head_names = base_names + [nm for nm, _ in PROX]
    head = np.concatenate([base, prox], axis=1)
    src_names = CTX_BASE + [nm for nm, _ in PROX]
    src = np.array([head_names.index(s) for s in src_names], np.int64)
    ctx = np.zeros((n, len(src) * (len(CENTRED) + len(ONESIDED))), np.float32)
    for _, first, last in blocks:
        ctx[first:last + 1] = context_block(head[first:last + 1][:, src].astype(np.float64))
    ctx_names = ["{}__{}".format(s, suf) for s in src_names for suf, _ in CENTRED + ONESIDED]
    allnames = head_names + ctx_names
    index = {nm: i for i, nm in enumerate(allnames)}
    missing = [c for c in columns if c not in index and c not in lpg.GROUP_SET]
    if missing:
        raise ValueError("unknown columns {}".format(missing[:5]))
    full = np.concatenate([head, ctx], axis=1)
    if any(c in lpg.GROUP_SET for c in columns):
        # member spec F6 (round 8): the group columns, computed only when the file lists one
        for i, nm in enumerate(lpg.GROUP_NAMES):
            index[nm] = full.shape[1] + i
        full = np.concatenate([full, lpg.group_features(tac, game, base_version, cfg,
                                                        group_scale)], axis=1)
    return np.ascontiguousarray(full[:, [index[c] for c in columns]], dtype=np.float32)


def swapped_tactical(tac):
    """The tracking with home and away exchanged: slots relabelled (team 0 <-> 1), coordinates
    unchanged (frame_features turns the pitch so the new home team attacks +x)."""
    team = np.asarray(tac["team_of_slot"])
    return dict(tac, team_of_slot=np.where(team == 0, 1, np.where(team == 1, 0, team)).astype(
        team.dtype))


# -- models -------------------------------------------------------------------------------------

def load_pf(path):
    """The weights file of the per-frame member (tools/learned_spike/pf_export.py) as a dict:
    "columns", "base_version", "format", "md5", and per model in MODELS a dict of the node arrays,
    "roots" and "baseline". Raise ValueError on another format, an inconsistent array or a column
    this module cannot compute."""
    with open(path, "rb") as handle:
        md5 = hashlib.md5(handle.read()).hexdigest()
    with np.load(path, allow_pickle=False) as z:
        stored = {k: z[k] for k in z.files}
    fmt = str(stored.get("format", ""))
    if fmt != PF_FORMAT:
        raise ValueError("{}: format {!r}, the engine reads {!r}".format(path, fmt, PF_FORMAT))
    columns = [str(c) for c in stored["columns"]]
    base_version = lf.check_version(str(stored["base_version"]))
    bad = [c for c in columns if not computable(c)]
    if bad:
        raise ValueError("{}: columns the engine does not compute: {}".format(path, bad[:5]))
    swap_mode = str(stored.get("swap_mode", "columns"))
    if swap_mode not in SWAP_MODES:
        raise ValueError("{}: swap_mode {!r}, not one of {}".format(path, swap_mode, SWAP_MODES))
    group_scale = str(stored.get("group_scale", lpg.SCALE_RAW))
    if group_scale not in lpg.SCALES:
        raise ValueError("{}: group_scale {!r}, not one of {}".format(path, group_scale,
                                                                      lpg.SCALES))
    model = {"columns": columns, "base_version": base_version, "format": fmt, "md5": md5,
             "version": str(stored.get("version", "")), "swap_mode": swap_mode,
             "group_scale": group_scale}
    for name in MODELS:
        m = {k: stored["{}/{}".format(name, k)] for k in NODE_ARRAYS + ("roots",)}
        m["baseline"] = float(stored["{}/baseline".format(name)])
        n = len(m["feature"])
        if any(len(m[k]) != n for k in NODE_ARRAYS):
            raise ValueError("{}: {} node arrays of different lengths".format(path, name))
        inner = ~m["leaf"].astype(bool)
        if (m["feature"][inner] < 0).any() or (m["feature"][inner] >= len(columns)).any():
            raise ValueError("{}: {} feature index outside the columns".format(path, name))
        for k in ("left", "right"):
            if ((m[k][inner] < 0) | (m[k][inner] >= n)).any():
                raise ValueError("{}: {} child index outside the nodes".format(path, name))
        m["feature"] = m["feature"].astype(np.int64)
        m["left"] = m["left"].astype(np.int64)
        m["right"] = m["right"].astype(np.int64)
        m["leaf"] = m["leaf"].astype(bool)
        m["missing_left"] = m["missing_left"].astype(bool)
        m["threshold"] = m["threshold"].astype(np.float64)
        m["value"] = m["value"].astype(np.float64)
        m["roots"] = m["roots"].astype(np.int64)
        model[name] = m
    swap_columns(columns, base_version)          # the columns must be closed under the swap
    return model


def predict_raw(m, X):
    """float64 (N,) raw score: baseline + the leaf value of each tree, trees in file order."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    raw = np.full(n, m["baseline"])
    feature, threshold, left, right = m["feature"], m["threshold"], m["left"], m["right"]
    leaf, missing_left, value = m["leaf"], m["missing_left"], m["value"]
    all_rows = np.arange(n)
    for root in m["roots"]:
        if leaf[root]:
            raw += value[root]
            continue
        node = np.full(n, root, np.int64)
        rows = all_rows
        cur = node
        while rows.size:
            f = feature[cur]
            x = X[rows, f]
            go_left = np.where(np.isnan(x), missing_left[cur], x <= threshold[cur])
            cur = np.where(go_left, left[cur], right[cur])
            done = leaf[cur]
            node[rows[done]] = cur[done]
            keep = ~done
            rows = rows[keep]
            cur = cur[keep]
        raw += value[node]
    return raw


def sigmoid(raw):
    """1 / (1 + exp(-raw)), elementwise (large |raw| saturates without a warning)."""
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-raw))


def predict_proba(m, X):
    return sigmoid(predict_raw(m, X))


def combine(pd_f, pd_g, pt_f, pt_g):
    """(N, 3) H, A, D from the two models on the features F and the swapped features G."""
    p_dead = 0.5 * (pd_f + pd_g)
    q = 0.5 * (pt_f - pt_g)
    alive = 1.0 - p_dead
    return np.stack([alive * (0.5 + q), alive * (0.5 - q), p_dead], axis=1)


def member_probabilities(model, tac, game, cfg=None, swap_mode=None):
    """float64 (N, 3) unrounded H, A, D of the per-frame member.

    swap_mode (None: the weights file's): "columns", G = the column-permuted, sign-changed copy
    of F; "tracking", G = the features of swapped_tactical(tac).
    """
    swap_mode = model.get("swap_mode", "columns") if swap_mode is None else swap_mode
    cols, base = model["columns"], model["base_version"]
    gs = model.get("group_scale", lpg.SCALE_RAW)
    F = pf_features(tac, game, cols, base, cfg, gs)
    if swap_mode == "tracking":
        G = pf_features(swapped_tactical(tac), game, cols, base, cfg, gs)
    elif swap_mode == "columns":
        perm, sign = swap_columns(cols, base)
        G = F[:, perm] * sign
    else:
        raise ValueError(swap_mode)
    return combine(predict_proba(model["dead"], F), predict_proba(model["dead"], G),
                   predict_proba(model["team"], F), predict_proba(model["team"], G))


def frame_probabilities_pf(tac, game, model, cfg=None):
    """float64 (N, 3) H, A, D of the per-frame member at 25 fps, rounded to PROB_DECIMALS."""
    return np.round(member_probabilities(model, tac, game, cfg), PROB_DECIMALS)
