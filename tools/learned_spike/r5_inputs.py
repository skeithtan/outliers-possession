"""Round 5 inputs: DFL still ball kept in view with jitter (step J) and the widened defect copies (X). Not part of the submission.

    # measurements on SkillCorner (external broadcast tracking), summary statistics only
    python -m tools.learned_spike.r5_inputs measure-jitter --labels CACHE --engine-cache DIR --out J.json
    python -m tools.learned_spike.r5_inputs measure-tracklets --engine-cache DIR --out T.json
    # training cache: DFL raw and degraded from input dfl_c_train_j, other matches linked to BASE
    python -m tools.learned_spike.r5_inputs cache --kind c_train_j --base CACHE --engine-cache DIR \
        --out NEW [--feature-version V]
    # evaluation view: every DFL still-ball run kept in view and jittered, no gaps (dfl_j_eval)
    python -m tools.learned_spike.r5_inputs cache --kind j_eval --base CACHE --engine-cache DIR --out NEW
    # aug copies with switch X: all 28 matches rebuilt (DFL from dfl_c_train) ...
    python -m tools.learned_spike.r5_inputs aug --dfl-input c_train --labels CACHE --engine-cache DIR \
        --out AUG5X [--games G ...] [--feature-version V]
    # ... or DFL rebuilt from dfl_c_train_j and the other matches linked to AUG5X
    python -m tools.learned_spike.r5_inputs aug --dfl-input c_train_j --link-from AUG5X ... --out AUG5XJ
    # after all builders of a folder: completeness check, aug manifest, marker file
    python -m tools.learned_spike.r5_inputs mark --kind c_train_j|j_eval|aug --dir NEW [--dfl-input I]

Step J (docs/plan/calibration/R5_AUG.md), input `dfl_c_train_j` (load_dfl(mode="c_train_j")):
as dfl_c_train (r4_inputs.py), except that each still-ball run of tools/still_ball.py (exactly
zero displacement for 13 frames or more) is, with probability P_KEEP, kept in view with jitter
instead of hidden, and a kept run gets no extension. The same jitter, with probability P_SLOW, is
put on the slow runs of a moving ball (label-free: runs of at least SLOW_MIN_FRAMES frames where
the ball is visible, not in a still-ball run, and moves less than SLOW_MPS between consecutive
frames). One numpy RandomState per run, seeded by jitter_seed() (the match, the kind of run and
its first frame), draws the decision first, then the run's rates, then its path. The jitter of
a run: per axis, white noise filtered with a Gaussian of JITTER_TAU_S / 2 (so that its
autocorrelation falls to 1/e at JITTER_TAU_S, the statistic measured), scaled to a per-axis
RMS of JITTER_SIGMA_M; plus a drift of the run's centre at JITTER_DRIFT_MPS in a uniform
direction, from the run's first frame, its length capped at JITTER_EXCURSION_M. Then the steps
(a) and (b) of dfl_c_train with the same seed (r4_inputs.train_seed): label-free ball gaps on
visible frames until 10 %, and the U(0, 1.5 s) extension of every HIDDEN still-ball run (the
extension is drawn for every run, so the stream does not depend on the decisions).

Input `dfl_j_eval` (mode "j_eval", evaluation only, fixed seeds of purpose "eval", never used in
training): every still-ball run kept in view and jittered; no gaps, no slow-run jitter.

Switch X: augment.augment(..., version="x") (component 6 widened, component 9 flicker).

Feature version: every builder computes features through frame_features() below, which passes
`--feature-version` (default: the environment variable FEATURE_VERSION; empty = the current
features) to the feature code. See frame_features() for what the coordinator wires at merge.

The ranges of the jitter are measured on the 20 SkillCorner matches (measure-jitter), never on the
competition match. The two sealed DFL matches, the competition match and every PFF match are
refused (still_ball.check_game, build.check, aug_build.check_games). Python 3.9.
"""

import argparse
import inspect
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
import still_ball  # noqa: E402
from tools.learned_spike import augment, degrade, folds, r4_inputs  # noqa: E402

FPS = 25
D_CODE = 2
# -- step J ------------------------------------------------------------------------------------
P_KEEP = 0.5                 # share of still-ball runs kept in view (brief)
P_SLOW = 0.5                 # share of slow runs jittered (brief: "the same probability")
SLOW_MPS = 1.0               # slow run: frame-to-frame speed under 1 m/s throughout (brief)
SLOW_MIN_FRAMES = still_ball.MIN_FRAMES   # 13 frames, as the still-ball rule (brief)
# Ranges: the 10th to 90th percentile over the 20 SkillCorner matches of the per-match process
# rates that reproduce SkillCorner under the estimators of measure-jitter (calibrate();
# $SP/r5_work/measure/jitter_sc.json, R5_AUG.md Section 4). Step J was DROPPED from round 5
# (coordinator, 13 Nov 2026): no cache uses these modes; the code stays, off by default.
JITTER_SIGMA_M = (0.0725, 0.1018)     # per-axis RMS of the jitter, m (uniform)
JITTER_TAU_S = (0.547, 0.741)         # autocorrelation time (1/e), s (log-uniform)
JITTER_DRIFT_MPS = (0.141, 0.180)     # speed of the centre's drift, m/s (uniform)
JITTER_EXCURSION_M = (0.644, 0.910)   # cap on the drift's length, m (uniform)
# Training J inputs and their keep probability of a still-ball run (round 6 arm A2: keep all).
# The per-run draws do not depend on it (decision drawn first, then the rates, then the path),
# so the runs kept at 0.5 are kept, with the same jitter, at 1.0.
J_KEEP = {"c_train_j": P_KEEP, "c_train_j100": 1.0}
TRAIN_J = tuple(J_KEEP)
MODES = TRAIN_J + ("j_eval",)
PURPOSES = ("train", "eval")
MARKER = r4_inputs.MARKER
# -- defect evaluation view ---------------------------------------------------------------------

DEFECT_EVAL_K = (100, 101)   # copy indices training never uses (training: k = 0 .. 15)
DEFECT_EVAL_GAMES = tuple(g for g in folds.OPEN)   # the 8 development matches
DEFECT_EVAL_MARKER = "defect_eval.json"


def check_defect_eval_seeds(games=DEFECT_EVAL_GAMES, ks=DEFECT_EVAL_K, train_k=16):
    """Raise AssertionError if a seed of an evaluation copy equals a stress-view seed or any seed
    of a training copy (k < train_k, every component of every switch). Return the count."""
    train_names = augment.COMPONENTS + ("plan",) + augment.BALL_SEEDS + augment.FLICKER_SEEDS
    n = 0
    for g in games:
        bad = augment.stress_seeds(g)
        train = {augment.seed_of(g, k, c) for k in range(train_k) for c in train_names}
        for k in ks:
            assert k >= train_k, k
            seen = set()
            for c in augment.COMPONENTS + ("plan",):
                s = augment.seed_of(g, k, c)
                assert s not in bad and s not in train and s not in seen, (g, k, c)
                seen.add(s)
                n += 1
    augment.check_seeds(games, max(ks) + 1)
    return n


def defect_eval_path(out, game, k):
    return os.path.join(out, "feats", "defect", game, "k{}.npy".format(int(k)))


def build_defect_eval(labels, engine_cache, out, version=None, games=DEFECT_EVAL_GAMES,
                      force=False):
    """The defect evaluation view: for each development match (Metrica raw; DFL input dfl_c,
    the still-ball rule alone), copies DEFECT_EVAL_K of augment.augment at the round 3 ranges
    (version "r3", no ball component), features at `version` (5 fps .npy)."""
    from engine import data
    check_defect_eval_seeds(games)
    _claim(out, {"kind": "defect_eval", "feature_version": version,
                 "ks": list(DEFECT_EVAL_K), "augment": "r3 ranges, ball off"})
    rec = {}
    for game in games:
        if game not in DEFECT_EVAL_GAMES:
            raise SystemExit("{}: development matches only".format(game))
        t0 = time.time()
        codes = r4_inputs._codes(labels, game)
        todo = [k for k in DEFECT_EVAL_K
                if force or not os.path.isfile(defect_eval_path(out, game, k))]
        if not todo:
            continue
        tac, _ = r4_inputs.load_dfl(r4_inputs._raw_tac(game, engine_cache), game, "c")
        for k in todo:
            aug, rep = augment.augment(tac, codes, game, k, data.graded_blocks(game),
                                       data.pitch_m(game))
            _, f5 = _features(aug, game, len(codes), version)
            r4_inputs._save_npy(defect_eval_path(out, game, k), f5)
            rec["{}/k{}".format(game, k)] = {c: rep.get(c) for c in augment.COMPONENTS
                                             if c in rep}
        print(game, "defect eval {:.1f}s".format(time.time() - t0), file=sys.stderr,
              flush=True)
    _merge_reports(out, {"defect_eval_" + k.replace("/", "_"): v for k, v in rec.items()})
    missing = [defect_eval_path(out, g, k) for g in DEFECT_EVAL_GAMES for k in DEFECT_EVAL_K
               if not os.path.isfile(defect_eval_path(out, g, k))]
    if not missing:
        with open(os.path.join(out, DEFECT_EVAL_MARKER), "w") as h:
            json.dump({"feature_version": version, "ks": list(DEFECT_EVAL_K),
                       "games": list(DEFECT_EVAL_GAMES), "dfl_input": "c",
                       "augment_md5": r4_inputs._md5(augment.__file__.replace(".pyc", ".py"))},
                      h, indent=1, sort_keys=True)
    return rec


# -- measurement on SkillCorner ----------------------------------------------------------------
MIN_DEAD_S = 2.0             # reference dead-ball segments of at least 2 s (brief)
NEAR_STILL_M = 0.5           # nearly still: net displacement over the centred 1 s below 0.5 m
NEAR_STILL_SENS = (0.25, 0.5, 1.0)   # thresholds reported for sensitivity
NEAR_MIN_S = 1.0             # a nearly-still run counts if it lasts 1 s or more
TAU_MIN_S = 2.0              # tau is estimated on runs of 2 s or more
TAU_MAX_LAG_S = 2.0
CAL_ITERS = 6                # fixed-point iterations of calibrate()
CAL_REPS = 10                # each match's run lengths simulated 10 times
QUANT = (0.10, 0.90)


def _runs(mask):
    padded = np.concatenate(([False], np.asarray(mask, bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[0::2], edges[1::2]


# -- the jitter process -------------------------------------------------------------------------

def jitter_seed(game, kind, start, purpose="train"):
    """Fixed seed of one run's draws (kind "still" or "slow", start its first frame)."""
    if purpose not in PURPOSES or kind not in ("still", "slow"):
        raise ValueError((purpose, kind))
    key = "r5j|{}|{}|{}|{}".format(purpose, game, kind, int(start))
    return zlib.crc32(key.encode("utf-8")) & 0x7FFFFFFF


def draw_rates(rng):
    """The rates of one run (always the same number of draws)."""
    return {"sigma_m": float(rng.uniform(*JITTER_SIGMA_M)),
            "tau_s": float(augment._loguniform(rng, *JITTER_TAU_S)),
            "drift_mps": float(rng.uniform(*JITTER_DRIFT_MPS)),
            "excursion_m": float(rng.uniform(*JITTER_EXCURSION_M))}


def smoothed_noise(n, rng, sigma_m, tau_s, fps=FPS):
    """(N,) white noise filtered with a Gaussian of tau_s * fps / 2 frames, RMS sigma_m.

    The autocorrelation of the result is exp(-lag^2 / (4 s^2)) with s the kernel's sigma, so it
    falls to 1/e at lag 2 s = tau_s * fps frames."""
    s = max(float(tau_s) * fps / 2.0, 1e-6)
    k = augment.gaussian_kernel(s)
    k = k / np.sqrt((k ** 2).sum())
    r = (k.size - 1) // 2
    return np.convolve(rng.standard_normal(n + 2 * r), k, mode="valid")[:n] * sigma_m


def jitter_path(n, rng, rates, fps=FPS):
    """(dx, dy) metres, (n,) each: smoothed noise per axis plus the capped drift."""
    theta = rng.uniform(0.0, 2.0 * np.pi)
    nx = smoothed_noise(n, rng, rates["sigma_m"], rates["tau_s"], fps)
    ny = smoothed_noise(n, rng, rates["sigma_m"], rates["tau_s"], fps)
    d = np.minimum(rates["drift_mps"] * np.arange(n) / float(fps), rates["excursion_m"])
    return nx + d * np.cos(theta), ny + d * np.sin(theta)


def slow_mask(bx, by, pitch, still, fps=FPS):
    """bool (N,): frames of the slow runs (label-free, module docstring)."""
    xm = np.asarray(bx, np.float64) * pitch[0]
    ym = np.asarray(by, np.float64) * pitch[1]
    vis = np.isfinite(xm) & np.isfinite(ym)
    step = np.full(len(xm), np.inf)
    with np.errstate(invalid="ignore"):
        step[1:] = np.hypot(np.diff(xm), np.diff(ym)) * fps
    step[~np.isfinite(step)] = np.inf
    slow = vis & ~np.asarray(still, bool) & (step < SLOW_MPS)
    # the first frame of a run has no step from a visible frame; a run counts from its 2nd frame
    out = np.zeros(len(xm), bool)
    for a, b in zip(*_runs(slow)):
        if b - a >= SLOW_MIN_FRAMES:
            out[a:b] = True
    return out


def apply_j(tac, game, pitch, purpose="train", p_keep=P_KEEP):
    """(bx, by, still, hidden, report): step J on a raw DFL tactical dict (copies).

    still: the still-ball frames of the rule; hidden: the frames of the still-ball runs that
    are hidden (missing in the output). purpose "train": each still run kept and jittered with
    P_KEEP, else hidden; each slow run jittered with P_SLOW. purpose "eval": every still run
    kept and jittered, slow runs untouched."""
    still_ball.check_game(game)
    bx = np.array(tac["bx"], dtype=np.float64) * pitch[0]
    by = np.array(tac["by"], dtype=np.float64) * pitch[1]
    still = still_ball.still_mask(tac["bx"], tac["by"])
    slow = slow_mask(tac["bx"], tac["by"], pitch, still) if purpose == "train" \
        else np.zeros(len(bx), bool)
    hidden = np.zeros(len(bx), bool)
    rep = {"still_runs": 0, "still_kept": 0, "still_kept_frames": 0, "slow_runs": 0,
           "slow_jittered": 0, "slow_jittered_frames": 0, "purpose": purpose}
    bad = augment.stress_seeds(game)
    for kind, mask in (("still", still), ("slow", slow)):
        for a, b in zip(*_runs(mask)):
            seed = jitter_seed(game, kind, a, purpose)
            assert seed not in bad, (game, kind, a)
            rng = np.random.RandomState(seed)
            go = rng.uniform() < (p_keep if kind == "still" else P_SLOW)
            rates = draw_rates(rng)
            if purpose == "eval":
                go = kind == "still"
            rep[kind + "_runs"] += 1
            if not go:
                if kind == "still":
                    hidden[a:b] = True
                continue
            dx, dy = jitter_path(b - a, rng, rates)
            bx[a:b] += dx
            by[a:b] += dy
            key = "still_kept" if kind == "still" else "slow_jittered"
            rep[key] += 1
            rep[key + "_frames"] += int(b - a)
    bx[hidden] = np.nan
    by[hidden] = np.nan
    rep["still_frames"] = int(still.sum())
    return ((bx / pitch[0]).astype(np.float32), (by / pitch[1]).astype(np.float32), still,
            hidden, rep)


def train_gaps_j(bx, by, still, hidden, rng):
    """Steps (a) and (b) of dfl_c_train after step J: (a) exactly as r4_inputs.train_gaps, then
    (b) an extension drawn for every still run, applied only to the hidden ones."""
    bx, by, rep = r4_inputs.train_gaps(bx, by, np.zeros(len(bx), bool), rng)
    n = len(bx)
    extended = 0
    for a, b in zip(*_runs(still)):
        ext = int(rng.uniform(0, r4_inputs.EXTEND_MAX_S * FPS))
        if not hidden[a]:
            continue
        e = min(n, b + ext)
        extended += int(np.isfinite(bx[b:e]).sum())
        bx[b:e] = np.nan
        by[b:e] = np.nan
    rep["extended"] = extended
    return bx, by, rep


def load_dfl(tac, game, mode, pitch=None):
    """(tactical dict, report) under an input mode; r4_inputs.MODES are passed through.

    "c_train_j": step J (purpose train), then (a) and (b). "j_eval": step J (purpose eval).
    Matches of other providers are returned unchanged."""
    if mode in r4_inputs.MODES:
        return r4_inputs.load_dfl(tac, game, mode)
    if mode not in MODES:
        raise ValueError("input mode {!r} not in {}".format(mode, MODES + r4_inputs.MODES))
    if not game.startswith("DFL-"):
        return tac, {"mode": "raw"}
    if pitch is None:
        from engine import data
        pitch = data.pitch_m(game)
    purpose = "train" if mode in TRAIN_J else "eval"
    bx, by, still, hidden, rep = apply_j(tac, game, pitch, purpose,
                                         J_KEEP.get(mode, P_KEEP))
    rep["mode"] = mode
    if mode in TRAIN_J:
        rep["p_keep"] = J_KEEP[mode]
        bx, by, r = train_gaps_j(bx, by, still, hidden,
                                 np.random.RandomState(r4_inputs.train_seed(game)))
        rep.update(r)
    rep["ball_missing"] = int((~np.isfinite(bx)).sum())
    return dict(tac, bx=bx, by=by), rep


# -- features -----------------------------------------------------------------------------------

def feature_version(value=None):
    """The feature version of a build: the option, else $FEATURE_VERSION, else None (current)."""
    v = value if value is not None else os.environ.get("FEATURE_VERSION", "")
    return v or None


def frame_features(tac, game, version=None):
    """features.frame_features at a feature version.

    None: the current features (no argument passed). Otherwise the version is passed as a
    keyword `feature_version` or `version` if features.frame_features takes one, or through
    features.set_feature_version() if that exists; else SystemExit. MERGE NOTE: if the feature
    code of the other branch names its option differently, wire it here (one place: every
    builder of this module and aug_build.py calls this function)."""
    from tools.learned_spike import features
    if version is None:
        return features.frame_features(tac, game)
    params = inspect.signature(features.frame_features).parameters
    for name in ("feature_version", "version"):
        if name in params:
            return features.frame_features(tac, game, **{name: version})
    setter = getattr(features, "set_feature_version", None)
    if setter is not None:
        setter(version)
        return features.frame_features(tac, game)
    raise SystemExit("feature version {!r}: features.frame_features takes no version option "
                     "(wire r5_inputs.frame_features at merge)".format(version))


def _features(tac, game, n_codes, version):
    from tools.learned_spike import features
    f25 = frame_features(tac, game, version)
    if f25.shape[0] != n_codes:
        raise SystemExit("{}: {} feature frames, {} reference frames".format(
            game, f25.shape[0], n_codes))
    return f25, features.downsample(f25)


# -- builders -----------------------------------------------------------------------------------

def _link(src, dst):
    """r4_inputs._link, safe when parallel builders write the same link."""
    try:
        r4_inputs._link(src, dst)
    except FileExistsError:
        if os.path.realpath(dst) != os.path.realpath(src):
            raise


def _claim(out, rec):
    """Write out/build_r5.json on the first call; refuse a folder built with other settings."""
    path = os.path.join(out, "build_r5.json")
    os.makedirs(out, exist_ok=True)
    if os.path.isfile(path):
        with open(path) as h:
            old = json.load(h)
        if old != rec:
            raise SystemExit("{} was started with {}, not {}".format(out, old, rec))
        return
    tmp = path + ".tmp-{}".format(os.getpid())
    with open(tmp, "w") as h:
        json.dump(rec, h, indent=1, sort_keys=True)
    os.replace(tmp, path)     # atomic: a parallel builder reads the whole record or none


def build_cache(kind, base, engine_cache, out, version=None, force=False, games=None):
    """Cache folder `out` of input dfl_c_train_j (views raw, degraded) or dfl_j_eval (raw):
    links to `base` for labels and non-DFL features, DFL rebuilt."""
    if os.path.abspath(out) == os.path.abspath(base):
        raise SystemExit("--out must be a new folder")
    if kind not in MODES:
        raise SystemExit("kind {!r} not in {}".format(kind, MODES))
    _claim(out, {"kind": kind, "feature_version": version, "base": os.path.abspath(base)})
    views = ("raw",) if kind == "j_eval" else ("raw", "degraded")
    for game in folds.ALL:
        _link(os.path.join(base, "labels", game + ".npz"),
                        os.path.join(out, "labels", game + ".npz"))
        for v in views:
            if not game.startswith("DFL-"):
                _link(os.path.join(base, "feats", v, game + ".npz"),
                                os.path.join(out, "feats", v, game + ".npz"))
    reports = {}
    for game in folds.OPEN:
        if not game.startswith("DFL-") or (games is not None and game not in games):
            continue
        still_ball.check_game(game)
        if not force and all(os.path.isfile(os.path.join(out, "feats", v, game + ".npz"))
                             for v in views):
            continue
        t0 = time.time()
        codes = r4_inputs._codes(base, game)
        tac, rep = load_dfl(r4_inputs._raw_tac(game, engine_cache), game, kind)
        for v in views:
            t = tac if v == "raw" else degrade.degrade(tac, codes, game)
            f25, f5 = _features(t, game, len(codes), version)
            r4_inputs._save_npz(os.path.join(out, "feats", v, game + ".npz"), f25, f5)
        rep["seconds"] = round(time.time() - t0, 1)
        reports[game] = rep
        print(game, json.dumps(rep), file=sys.stderr, flush=True)
    _merge_reports(out, reports)
    return reports


def build_aug(dfl_input, labels, engine_cache, out, version=None, link_from=None,
              k_views=16, force=False, games=None, aug_version="x"):
    """Aug folder `out` with augment version `aug_version`: "x" (switch X, round 5) or "r3"
    (round 4's copies, rebuilt at another feature version; round 6 recipe AQCSJ).

    dfl_input "c_train": every one of the 28 matches is rebuilt (Metrica and SkillCorner from
    the raw input, DFL from dfl_c_train). dfl_input "c_train_j": the DFL matches are rebuilt
    from dfl_c_train_j and the others are linked to `link_from` (an X folder built with
    "c_train"). `games` limits the rebuilt matches (parallel builders); links always written.
    """
    from tools.learned_spike import aug_build
    from engine import data
    if dfl_input not in ("c_train",) + TRAIN_J:
        raise SystemExit("dfl_input {!r}".format(dfl_input))
    if dfl_input in TRAIN_J and link_from is None:
        raise SystemExit("--dfl-input c_train_j needs --link-from (the c_train X folder)")
    if aug_version not in augment.PLAN_VERSIONS:
        raise SystemExit("aug version {!r} not in {}".format(aug_version, augment.PLAN_VERSIONS))
    augment.check_seeds(folds.ALL, k_views, flicker=aug_version == "x")
    _claim(out, {"kind": "aug", "aug_version": aug_version, "dfl_input": dfl_input,
                 "feature_version": version,
                 "link_from": os.path.abspath(link_from) if link_from else None})
    if link_from is not None:
        mark_link = (still_marker(link_from) or {})
        if mark_link.get("feature_version", None) != version or \
                mark_link.get("aug_version") != aug_version or \
                mark_link.get("dfl_input") != "c_train":
            raise SystemExit("{} is not a finished c_train aug folder of version {} at feature "
                             "version {} (marker {})".format(link_from, aug_version, version,
                                                            mark_link))
    rebuilt = [g for g in folds.ALL if dfl_input == "c_train" or g.startswith("DFL-")]
    for game in folds.ALL:
        if game not in rebuilt:
            _link(os.path.join(link_from, "feats", "aug", game),
                            os.path.join(out, "feats", "aug", game))
    reports = {}
    for game in rebuilt:
        if games is not None and game not in games:
            continue
        aug_build.check_games([game])
        if game.startswith("DFL-"):
            still_ball.check_game(game)
        codes = r4_inputs._codes(labels, game)
        tac = None
        t0 = time.time()
        for k in range(k_views):
            path = aug_build.feat_path(out, game, k)
            if os.path.isfile(path) and not force:
                continue
            if tac is None:
                tac, rep = load_dfl(r4_inputs._raw_tac(game, engine_cache), game, dfl_input)
                reports[game] = {"input": rep, "copies": {}}
            aug, arep = augment.augment(tac, codes, game, k, data.graded_blocks(game),
                                        data.pitch_m(game), version=aug_version)
            reports[game]["copies"]["k{}".format(k)] = {
                "plan": arep["plan"], **{c: arep.get(c) for c in
                                         augment.COMPONENTS + ("flicker",) if c in arep}}
            _, f5 = _features(aug, game, len(codes), version)
            r4_inputs._save_npy(path, f5)
        if game in reports:
            reports[game]["seconds"] = round(time.time() - t0, 1)
        print(game, "aug done {:.1f}s".format(time.time() - t0), file=sys.stderr, flush=True)
    _merge_reports(out, reports)
    return reports


def _merge_reports(out, reports):
    """One file per match (out/reports_r5/<game>.json): parallel builders never share one."""
    for game, rep in reports.items():
        path = os.path.join(out, "reports_r5", game + ".json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        old = {}
        if os.path.isfile(path):
            with open(path) as h:
                old = json.load(h)
        if "copies" in rep and "copies" in old:
            rep = dict(rep, copies=dict(old["copies"], **rep["copies"]))
        tmp = path + ".tmp-{}".format(os.getpid())
        with open(tmp, "w") as h:
            json.dump(rep, h, indent=1, sort_keys=True, default=float)
        os.replace(tmp, path)


def still_marker(folder):
    path = os.path.join(folder, MARKER)
    if not os.path.isfile(path):
        return None
    with open(path) as h:
        return json.load(h)


def mark(kind, folder, dfl_input=None, k_views=16):
    """Check that a folder is complete, then write the marker (and, for aug, the manifest).

    kind c_train_j / j_eval: marker dfl_input "c_train_j" / "j_eval". kind aug: dfl_input
    "c_train" or "c_train_j" as built (read from build_r5.json if not given)."""
    from tools.learned_spike import aug_build
    path = os.path.join(folder, "build_r5.json")
    if not os.path.isfile(path):
        raise SystemExit("{}: no build_r5.json (not a round 5 folder)".format(folder))
    with open(path) as h:
        build = json.load(h)
    if build["kind"] != kind:
        raise SystemExit("{} holds kind {}, not {}".format(folder, build["kind"], kind))
    missing = []
    for game in folds.ALL:
        if kind in MODES:
            views = ("raw",) if kind == "j_eval" else ("raw", "degraded")
            paths = [os.path.join(folder, "labels", game + ".npz")] + [
                os.path.join(folder, "feats", v, game + ".npz") for v in views]
        else:
            paths = [aug_build.feat_path(folder, game, k) for k in range(k_views)]
        missing += [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit("{} incomplete: {} files missing, e.g. {}".format(
            folder, len(missing), missing[:3]))
    inp = kind if kind in MODES else (dfl_input or build["dfl_input"])
    rec = {"dfl_still": "removed", "dfl_input": inp, "kind": kind,
           "games": [g for g in folds.ALL if g.startswith("DFL-")],
           "feature_version": build["feature_version"],
           "min_frames": still_ball.MIN_FRAMES, "gap_s": r4_inputs.GAP_S,
           "gap_share": r4_inputs.GAP_SHARE, "extend_max_s": r4_inputs.EXTEND_MAX_S,
           "still_ball_md5": r4_inputs._md5(os.path.join(ROOT, "tools", "still_ball.py")),
           "r4_inputs_md5": r4_inputs._md5(r4_inputs.__file__.replace(".pyc", ".py")),
           "r5_inputs_md5": r4_inputs._md5(os.path.abspath(__file__).replace(".pyc", ".py")),
           "doc": "docs/plan/calibration/R5_AUG.md"}
    if inp in MODES:
        rec["jitter"] = {"p_keep": J_KEEP.get(inp, 1.0), "p_slow": P_SLOW, "slow_mps": SLOW_MPS,
                         "slow_min_frames": SLOW_MIN_FRAMES, "sigma_m": JITTER_SIGMA_M,
                         "tau_s": JITTER_TAU_S, "drift_mps": JITTER_DRIFT_MPS,
                         "excursion_m": JITTER_EXCURSION_M}
    if kind == "aug":
        rec["aug_version"] = build.get("aug_version", "x")
    if kind == "aug" and rec["aug_version"] == "x":
        rec["x"] = {"ghost_per_10min": augment.GHOST_PER_10MIN_X, "ghost_s": augment.GHOST_S_X,
                    "ghost_offset_m": augment.GHOST_OFFSET_M_X,
                    "flicker_per_slot_min": augment.FLICKER_PER_SLOT_MIN,
                    "flicker_gap_s": augment.FLICKER_GAP_S}
    if kind == "aug":
        rec["manifest_files"] = len(aug_build.write_manifest(folder, list(folds.ALL),
                                                             k_views)["files"])
    with open(os.path.join(folder, MARKER), "w") as h:
        json.dump(rec, h, indent=1, sort_keys=True)
    if build["feature_version"]:
        # train.py reads the folder's feature version from build.py's marker: check every
        # feature file's width and write it (feature_version.json)
        from tools.learned_spike import build as build_mod
        build_mod.mark(folder, build["feature_version"])
    return rec


# -- defect evaluation view ---------------------------------------------------------------------

DEFECT_EVAL_K = (100, 101)   # copy indices training never uses (training: k = 0 .. 15)
DEFECT_EVAL_GAMES = tuple(g for g in folds.OPEN)   # the 8 development matches
DEFECT_EVAL_MARKER = "defect_eval.json"


def check_defect_eval_seeds(games=DEFECT_EVAL_GAMES, ks=DEFECT_EVAL_K, train_k=16):
    """Raise AssertionError if a seed of an evaluation copy equals a stress-view seed or any seed
    of a training copy (k < train_k, every component of every switch). Return the count."""
    train_names = augment.COMPONENTS + ("plan",) + augment.BALL_SEEDS + augment.FLICKER_SEEDS
    n = 0
    for g in games:
        bad = augment.stress_seeds(g)
        train = {augment.seed_of(g, k, c) for k in range(train_k) for c in train_names}
        for k in ks:
            assert k >= train_k, k
            seen = set()
            for c in augment.COMPONENTS + ("plan",):
                s = augment.seed_of(g, k, c)
                assert s not in bad and s not in train and s not in seen, (g, k, c)
                seen.add(s)
                n += 1
    augment.check_seeds(games, max(ks) + 1)
    return n


def defect_eval_path(out, game, k):
    return os.path.join(out, "feats", "defect", game, "k{}.npy".format(int(k)))


def build_defect_eval(labels, engine_cache, out, version=None, games=DEFECT_EVAL_GAMES,
                      force=False):
    """The defect evaluation view: for each development match (Metrica raw; DFL input dfl_c,
    the still-ball rule alone), copies DEFECT_EVAL_K of augment.augment at the round 3 ranges
    (version "r3", no ball component), features at `version` (5 fps .npy)."""
    from engine import data
    check_defect_eval_seeds(games)
    _claim(out, {"kind": "defect_eval", "feature_version": version,
                 "ks": list(DEFECT_EVAL_K), "augment": "r3 ranges, ball off"})
    rec = {}
    for game in games:
        if game not in DEFECT_EVAL_GAMES:
            raise SystemExit("{}: development matches only".format(game))
        t0 = time.time()
        codes = r4_inputs._codes(labels, game)
        todo = [k for k in DEFECT_EVAL_K
                if force or not os.path.isfile(defect_eval_path(out, game, k))]
        if not todo:
            continue
        tac, _ = r4_inputs.load_dfl(r4_inputs._raw_tac(game, engine_cache), game, "c")
        for k in todo:
            aug, rep = augment.augment(tac, codes, game, k, data.graded_blocks(game),
                                       data.pitch_m(game))
            _, f5 = _features(aug, game, len(codes), version)
            r4_inputs._save_npy(defect_eval_path(out, game, k), f5)
            rec["{}/k{}".format(game, k)] = {c: rep.get(c) for c in augment.COMPONENTS
                                             if c in rep}
        print(game, "defect eval {:.1f}s".format(time.time() - t0), file=sys.stderr,
              flush=True)
    _merge_reports(out, {"defect_eval_" + k.replace("/", "_"): v for k, v in rec.items()})
    missing = [defect_eval_path(out, g, k) for g in DEFECT_EVAL_GAMES for k in DEFECT_EVAL_K
               if not os.path.isfile(defect_eval_path(out, g, k))]
    if not missing:
        with open(os.path.join(out, DEFECT_EVAL_MARKER), "w") as h:
            json.dump({"feature_version": version, "ks": list(DEFECT_EVAL_K),
                       "games": list(DEFECT_EVAL_GAMES), "dfl_input": "c",
                       "augment_md5": r4_inputs._md5(augment.__file__.replace(".pyc", ".py"))},
                      h, indent=1, sort_keys=True)
    return rec


# -- measurement on SkillCorner -----------------------------------------------------------------

def dead_mask(codes, blocks, min_s=MIN_DEAD_S, fps=FPS):
    """bool (N,): frames of the reference D segments of at least min_s inside graded blocks."""
    n = len(codes)
    graded = np.zeros(n, bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    out = np.zeros(n, bool)
    for a, b in zip(*_runs((np.asarray(codes) == D_CODE) & graded)):
        if b - a >= min_s * fps:
            out[a:b] = True
    return out


def near_still_runs(xm, ym, dead, thr=NEAR_STILL_M, fps=FPS, min_s=NEAR_MIN_S):
    """[(a, b)]: maximal runs of dead frames with a visible ball whose net displacement over the
    centred 1 s (frames t - fps//2 .. t + fps//2) is below `thr` metres, of min_s or longer."""
    n = len(xm)
    w = fps // 2
    nd = np.full(n, np.inf)
    if n > 2 * w:
        with np.errstate(invalid="ignore"):
            d = np.hypot(xm[2 * w:] - xm[:n - 2 * w], ym[2 * w:] - ym[:n - 2 * w])
        nd[w:n - w] = np.where(np.isfinite(d), d, np.inf)
    ok = dead & np.isfinite(xm) & np.isfinite(ym) & (nd < thr)
    return [(int(a), int(b)) for a, b in zip(*_runs(ok)) if b - a >= min_s * fps]


def run_stats(x, y, fps=FPS):
    """{sigma_m, tau_s, drift_mps, excursion_m, steps} of one run (metres, contiguous)."""
    n = len(x)
    t = np.arange(n) / float(fps)
    res, slope = [], []
    for v in (x, y):
        c = np.polyfit(t, v, 1)
        slope.append(c[0])
        res.append(v - np.polyval(c, t))
    r = np.array(res)
    sigma = float(np.sqrt((r ** 2).mean()))
    tau = None
    if n >= TAU_MIN_S * fps:
        z = r - r.mean(axis=1, keepdims=True)
        den = float((z ** 2).sum())
        prev = 1.0
        for lag in range(1, min(n - 1, int(TAU_MAX_LAG_S * fps)) + 1):
            ac = float((z[:, lag:] * z[:, :-lag]).sum()) / den if den > 0 else 0.0
            if ac < np.exp(-1.0):
                frac = (prev - np.exp(-1.0)) / (prev - ac) if prev != ac else 0.0
                tau = (lag - 1 + frac) / float(fps)
                break
            prev = ac
    drift = float(np.hypot(*slope))
    return {"sigma_m": sigma, "tau_s": tau, "drift_mps": drift,
            "excursion_m": drift * n / float(fps),
            "steps": np.hypot(np.diff(x), np.diff(y))}


def match_jitter(xm, ym, dead, thr=NEAR_STILL_M, fps=FPS):
    """Per-match summary of the nearly-still runs (the per-match values of R5_AUG.md)."""
    runs = near_still_runs(xm, ym, dead, thr, fps)
    stats = [run_stats(xm[a:b], ym[a:b], fps) for a, b in runs]
    if not stats:
        return {"runs": 0}
    steps = np.concatenate([s["steps"] for s in stats])
    taus = [s["tau_s"] for s in stats if s["tau_s"] is not None]
    lengths = np.array([b - a for a, b in runs]) / float(fps)
    return {"runs": len(runs), "frames": int(sum(b - a for a, b in runs)),
            "run_s_p50": float(np.median(lengths)), "run_s_p90": float(np.quantile(lengths, .9)),
            "step_m_p50": float(np.median(steps)), "step_m_p90": float(np.quantile(steps, .9)),
            "sigma_m": float(np.median([s["sigma_m"] for s in stats])),
            "tau_s": float(np.median(taus)) if taus else None,
            "tau_runs": len(taus),
            "tau_censored": int(sum(1 for s in stats if s["tau_s"] is None
                                    and len(s["steps"]) + 1 >= TAU_MIN_S * fps)),
            "drift_mps": float(np.median([s["drift_mps"] for s in stats])),
            "excursion_m_p90": float(np.quantile([s["excursion_m"] for s in stats], .9)),
            "lengths_frames": [int(b - a) for a, b in runs]}


def simulate_check(lengths, rates, seed, fps=FPS):
    """The estimators of match_jitter() applied to the jitter process itself (a still ball plus
    jitter_path at `rates`) on runs of the given lengths: does the process reproduce what was
    measured? Returns the same keys as match_jitter (without lengths)."""
    rng = np.random.RandomState(seed)
    stats = []
    for n in lengths:
        dx, dy = jitter_path(int(n), rng, rates, fps)
        stats.append(run_stats(dx, dy, fps))
    steps = np.concatenate([s["steps"] for s in stats])
    taus = [s["tau_s"] for s in stats if s["tau_s"] is not None]
    return {"step_m_p50": float(np.median(steps)), "step_m_p90": float(np.quantile(steps, .9)),
            "sigma_m": float(np.median([s["sigma_m"] for s in stats])),
            "tau_s": float(np.median(taus)) if taus else None,
            "drift_mps": float(np.median([s["drift_mps"] for s in stats])),
            "excursion_m_p90": float(np.quantile([s["excursion_m"] for s in stats], .9))}


def calibrate(lengths, meas, seed, iters=CAL_ITERS, reps=CAL_REPS, fps=FPS):
    """Process rates whose estimates (simulate_check, the estimators of match_jitter on the
    process itself, same run lengths repeated `reps` times) equal the measured sigma and tau.

    The estimators read short detrended runs, so they see less than the process holds; this
    inverts them (fixed point, common random numbers). The drift and the excursion cap are
    taken as measured. Returns (rates, the estimates at those rates)."""
    rates = {"sigma_m": meas["sigma_m"], "tau_s": meas["tau_s"],
             "drift_mps": meas["drift_mps"], "excursion_m": meas["excursion_m_p90"]}
    lengths = list(lengths) * int(reps)
    for _ in range(int(iters)):
        est = simulate_check(lengths, rates, seed, fps)
        rates = dict(rates, sigma_m=rates["sigma_m"] * meas["sigma_m"] / est["sigma_m"],
                     tau_s=rates["tau_s"] * meas["tau_s"] / est["tau_s"])
    return rates, simulate_check(lengths, rates, seed, fps)


def measure_jitter(labels, engine_cache, games=None):
    """measure-jitter: per SkillCorner match and the ranges (10th to 90th percentile)."""
    from engine import data
    games = list(games or folds.SC)
    out = {"definition": {
        "dead": "reference (as provided) D segments of >= {} s inside graded blocks".format(
            MIN_DEAD_S),
        "nearly_still": "dead frames with a visible ball whose net displacement over the "
                        "centred 1 s is below {} m; runs of >= {} s".format(NEAR_STILL_M,
                                                                            NEAR_MIN_S),
        "sigma_m": "per run: per-axis RMS of the position minus a per-axis linear fit; match: "
                   "median over runs",
        "tau_s": "per run (>= {} s): first lag where the autocorrelation of the detrended "
                 "position (axes pooled) falls below 1/e, linear between lags; match: median"
                 .format(TAU_MIN_S),
        "drift_mps": "per run: speed of the linear fit; match: median over runs",
        "excursion_m_p90": "per run: drift x run length; match: 90th percentile over runs",
        "step_m": "frame-to-frame displacement of the ball on the nearly-still frames (25 fps, "
                  "SkillCorner's 10 Hz ball interpolated by the bundle)"},
        "matches": {}, "sensitivity": {}}
    for game in games:
        if not game.startswith("SC-"):
            raise SystemExit("{}: measure-jitter reads SkillCorner matches only".format(game))
        tac = r4_inputs._raw_tac(game, engine_cache)
        pitch = data.pitch_m(game)
        codes = r4_inputs._codes(labels, game)
        xm = np.asarray(tac["bx"], np.float64) * pitch[0]
        ym = np.asarray(tac["by"], np.float64) * pitch[1]
        dead = dead_mask(codes, data.graded_blocks(game))
        rec = match_jitter(xm, ym, dead)
        rec["dead_frames"] = int(dead.sum())
        rec["dead_visible_frames"] = int((dead & np.isfinite(xm)).sum())
        out["matches"][game] = rec
        out["sensitivity"][game] = {}
        for thr in NEAR_STILL_SENS:
            r = match_jitter(xm, ym, dead, thr)
            r.pop("lengths_frames", None)
            out["sensitivity"][game][str(thr)] = r
        print(game, json.dumps({k: v for k, v in rec.items() if k != "lengths_frames"}),
              file=sys.stderr, flush=True)
    keys = ("sigma_m", "tau_s", "drift_mps", "excursion_m_p90", "step_m_p50", "step_m_p90")
    out["ranges"] = {}
    for k in keys:
        vals = [m[k] for m in out["matches"].values() if m.get(k) is not None]
        out["ranges"][k] = {"p10": float(np.quantile(vals, QUANT[0])),
                            "p50": float(np.median(vals)),
                            "p90": float(np.quantile(vals, QUANT[1])),
                            "min": float(min(vals)), "max": float(max(vals)), "n": len(vals)}
    out["sensitivity_ranges"] = {}
    for thr in NEAR_STILL_SENS:
        out["sensitivity_ranges"][str(thr)] = {}
        for k in ("sigma_m", "tau_s", "drift_mps", "excursion_m_p90", "step_m_p50"):
            vals = [s[str(thr)][k] for s in out["sensitivity"].values()
                    if s[str(thr)].get(k) is not None]
            out["sensitivity_ranges"][str(thr)][k] = [
                float(np.quantile(vals, QUANT[0])), float(np.quantile(vals, QUANT[1]))]
    # The process at the measured values, through the same estimators (simulate_check), then
    # the process rates that reproduce each match (calibrate): the ranges of step J.
    out["simulate_check"], out["calibrated"] = {}, {}
    for j, (game, m) in enumerate(out["matches"].items()):
        if not m.get("runs") or m.get("tau_s") is None:
            continue
        rates = {"sigma_m": m["sigma_m"], "tau_s": m["tau_s"], "drift_mps": m["drift_mps"],
                 "excursion_m": m["excursion_m_p90"]}
        out["simulate_check"][game] = simulate_check(m["lengths_frames"] * CAL_REPS, rates,
                                                     1000 + j)
        cal, est = calibrate(m["lengths_frames"], m, 2000 + j)
        out["calibrated"][game] = {"rates": cal, "estimates": est}
    out["calibrated_ranges"] = {}
    for k in ("sigma_m", "tau_s", "drift_mps", "excursion_m"):
        vals = [c["rates"][k] for c in out["calibrated"].values()]
        out["calibrated_ranges"][k] = {"p10": float(np.quantile(vals, QUANT[0])),
                                       "p50": float(np.median(vals)),
                                       "p90": float(np.quantile(vals, QUANT[1])),
                                       "min": float(min(vals)), "max": float(max(vals))}
    return out


def measure_tracklets(engine_cache, games=None):
    """measure-tracklets: tracklet births per slot-minute (slot_stats.match_stats)."""
    from tools.learned_spike import build, slot_stats
    games = list(games or folds.SC)
    out = {"definition": "slot_stats.match_stats on the raw tracking, graded frames; births per "
                         "slot-minute = births per 10 min / 10 / mean number of visible slots "
                         "(both teams); per half, then the frame-weighted mean of the halves",
           "matches": {}}
    for game in games:
        build.check(game)
        if game not in folds.ALL:
            raise SystemExit("{}: external matches only".format(game))
        st = slot_stats.match_stats(r4_inputs._raw_tac(game, engine_cache), game)
        num = den = 0.0
        halves = {}
        for h, hs in st["halves"].items():
            slots = hs["concurrent_home"]["mean"] + hs["concurrent_away"]["mean"]
            births = hs["tracklets"]
            minutes = hs["minutes"]
            halves[h] = {"births_per_10min": hs["births_per_10min"], "visible_slots": slots,
                         "births_per_slot_min": births / minutes / slots if slots else None,
                         "lifetime_s_p50": hs["lifetime_s"]["p50"],
                         "share_tracklets_under_2s": hs["share_tracklets_under_2s"],
                         "slot_gap_s_p50": hs["slot_gap_s"]["p50"],
                         "ghosts_per_10min": hs["ghosts"]["per_10min"]}
            num += births
            den += minutes * slots
        out["matches"][game] = {"halves": halves,
                                "births_per_slot_min": num / den if den else None}
        print(game, json.dumps(out["matches"][game]), file=sys.stderr, flush=True)
    for key, prov in (("range", ("skillcorner",)), ("range_metrica_dfl", ("metrica", "dfl"))):
        vals = [m["births_per_slot_min"] for g, m in out["matches"].items()
                if folds.provider(g) in prov]
        if vals:
            out[key] = {"p10": float(np.quantile(vals, .1)), "p50": float(np.median(vals)),
                        "p90": float(np.quantile(vals, .9)), "min": float(min(vals)),
                        "max": float(max(vals)), "n": len(vals)}
    return out


# -- command line -------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    mj = sub.add_parser("measure-jitter")
    mj.add_argument("--labels", required=True)
    mj.add_argument("--engine-cache", required=True)
    mj.add_argument("--out", required=True)
    mj.add_argument("--games", nargs="+", default=None)
    mt = sub.add_parser("measure-tracklets")
    mt.add_argument("--engine-cache", required=True)
    mt.add_argument("--out", required=True)
    mt.add_argument("--games", nargs="+", default=None)
    c = sub.add_parser("cache")
    c.add_argument("--kind", choices=MODES, required=True)
    c.add_argument("--base", required=True)
    c.add_argument("--engine-cache", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--games", nargs="+", default=None)
    a = sub.add_parser("aug")
    a.add_argument("--dfl-input", choices=("c_train",) + TRAIN_J, required=True)
    a.add_argument("--labels", required=True)
    a.add_argument("--engine-cache", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--link-from", default=None)
    a.add_argument("--aug-version", choices=augment.PLAN_VERSIONS, default="x",
                   help="x: switch X (round 5); r3: round 4's copies (round 6)")
    a.add_argument("--games", nargs="+", default=None)
    de = sub.add_parser("defect-eval")
    de.add_argument("--labels", required=True)
    de.add_argument("--engine-cache", required=True)
    de.add_argument("--out", required=True)
    de.add_argument("--games", nargs="+", default=list(DEFECT_EVAL_GAMES))
    m = sub.add_parser("mark")
    m.add_argument("--kind", choices=MODES + ("aug",), required=True)
    m.add_argument("--dir", required=True)
    m.add_argument("--dfl-input", default=None)
    for s in (c, a, de):
        s.add_argument("--force", action="store_true")
        s.add_argument("--feature-version", default=None,
                       help="feature version passed to the feature code (default: "
                            "$FEATURE_VERSION; empty = the current features)")
    args = p.parse_args(argv)
    if args.cmd in ("measure-jitter", "measure-tracklets"):
        res = measure_jitter(args.labels, args.engine_cache, args.games) \
            if args.cmd == "measure-jitter" else measure_tracklets(args.engine_cache, args.games)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as h:
            json.dump(res, h, indent=1, sort_keys=True)
        print(json.dumps(res.get("ranges", res.get("range")), sort_keys=True))
    elif args.cmd == "defect-eval":
        build_defect_eval(args.labels, args.engine_cache, args.out,
                          feature_version(args.feature_version), args.games, args.force)
    elif args.cmd == "cache":
        build_cache(args.kind, args.base, args.engine_cache, args.out,
                    feature_version(args.feature_version), args.force, args.games)
    elif args.cmd == "aug":
        build_aug(args.dfl_input, args.labels, args.engine_cache, args.out,
                  feature_version(args.feature_version), args.link_from, force=args.force,
                  games=args.games, aug_version=args.aug_version)
    else:
        print(json.dumps(mark(args.kind, args.dir, args.dfl_input), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
