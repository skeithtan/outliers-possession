"""The training recipe of the learned stage, shared by kfold.py and final.py. Not part of the submission.

This module implements Section 2 of LEARNED_STAGE_PROTOCOL.md with the advisor changes of
9 November 2026:

- Data: the feature and label caches of build.py. Each match uses only the views it has
  (folds.views()): raw, matched, sigma95 and degraded for DFL and Metrica, raw and degraded for
  SkillCorner, raw only for the 43 PFF training matches (LEARNED_STAGE_PROTOCOL_R2.md Section 1).
  load_games() refuses the sealed and pilot PFF matches (folds.assert_not_sealed()).
- Batches: BATCH random crops of CROP_BINS bins (2 minutes at 5 fps). For each crop the tool
  draws a training match (uniform over matches), a view of that match (uniform over its views),
  a start, and one of the four symmetric copies. Input noise INPUT_NOISE.
- Loss: cross entropy on the 2 s controlled labels (round 3 switch TARGET "raw": the labels as
  provided; configure()), X masked, bins within BOUNDARY_BINS of a
  label change weighted BOUNDARY_WEIGHT times (lomo.boundary_weights).
- Optimiser: Adam, one-cycle schedule (peak LR, 10 % warm-up), gradient clip 1.0.
- Checkpoint (folds): every EVAL_EVERY steps the tool computes the loss on the raw view of each
  validation match. The checkpoint with the lowest mean over the validation matches is kept. The
  first such step wins a tie. Without validation matches (final models) the last step is kept.

Training is deterministic for a given seed, thread count and torch build: torch.manual_seed,
numpy RandomState and torch.use_deterministic_algorithms(True).

predict_p5() gives the symmetric mean probabilities of one model in float64, the precision of
the numpy inference in engine/. decoders.ensemble_p5() averages the models and rounds.

Python 3.9.
"""

import hashlib
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from tools.learned_spike import features, folds, model, pff  # noqa: E402
from tools.learned_spike.lomo import boundary_weights  # noqa: E402

STEPS = 2000
CROP_BINS = 600            # 2 minutes at 5 fps
BATCH = 16
LR = 2e-3
PCT_START = 0.1
CLIP = 1.0
INPUT_NOISE = 0.02
BOUNDARY_BINS = 5          # +-1 s at 5 fps (lomo.BOUNDARY_BINS)
BOUNDARY_WEIGHT = 3.0      # lomo.BOUNDARY_WEIGHT
EVAL_EVERY = 200
FOLD_SEEDS = (0, 1, 2, 3)

# Round 3 switches (LEARNED_STAGE_PROTOCOL_R3.md, DRAFT). The defaults are round 2 behaviour.
TARGETS = ("ctl2", "raw")            # A: the label view the loss reads
SHORT_WEIGHTS = ("off", "sqrt")      # A2: sqrt(median / length) on possession bins, clip [1/3, 3]
MIXES = ("match", "provider", "sc_quarter")  # A3: uniform over matches, provider first, or
                                             # R3b: a SkillCorner match at weight 1/4
MATCH_WEIGHTS = {"skillcorner": 0.25}         # R3b, taken from outside, not tuned
SHORT_CLIP = (1.0 / 3.0, 3.0)
TARGET = "ctl2"
SHORT_WEIGHT = "off"
MIX = "match"
# Round 3 augmentation switch (docs/plan/calibration/R3_AUG.md). Off by default: with AUG_CACHE
# None nothing below changes. On: each crop of a match that has aug views (augment.py,
# aug_build.py) takes, with probability AUG_SHARE, one of its K aug views (uniform), else a view
# drawn as before. Validation stays on the raw view.
AUG_CACHE = None
AUG_SHARE = 0.5          # half the crops keep the existing views (judgement, not tuned)
AUG_K = 16
# Round 4 switches (LEARNED_STAGE_PROTOCOL_R4.md Section 1). Off by default: with the defaults
# nothing below changes, the JSON records included (r4_record() is {}).
# A: the training views kept (None = folds.views(), all of them). Validation stays on raw.
TRAIN_VIEWS = None
# B: a build.py-style folder whose labels/ hold the SkillCorner labels (the possession groups of
# tools/skillcorner_fine.py, cache_possgroup). Training and validation read SkillCorner labels
# there; every other provider keeps the labels of --cache.
SC_LABELS = None
# C: the DFL still ball (tools/still_ball.py, r4_inputs.py). Two inputs: dfl_c_train (the rule,
# then label-free gaps and gap extensions) for every training view, dfl_c (the rule alone) for
# validation and scoring. VAL_CACHE: the folder the validation (raw-only) loads read their
# features from (None = --cache). "removed" asserts the markers (STILL_MARKER) of r4_inputs.py:
# --cache, the aug and ball-free folders hold dfl_c_train, VAL_CACHE holds dfl_c; "kept"
# asserts that no folder read carries a marker.
DFL_STILL = "kept"
DFL_STILLS = ("kept", "removed")
STILL_MARKER = "dfl_still.json"
VAL_CACHE = None
# D': the ball-free twins (r4_inputs.py ballfree): with BALLFREE_CACHE set, in BALLFREE_SHARE of
# the crops one window of BALLFREE_MIN_BINS (20 s) to the whole crop, placed uniformly, takes its
# features from the ball-free twin of the same match and view family (raw, degraded or aug copy
# k). Labels unchanged. The crop is drawn first exactly as without the switch.
BALLFREE_CACHE = None
BALLFREE_SHARE = 0.15    # judgement, not tuned (protocol R4 step D)
BALLFREE_MIN_BINS = 100  # 20 s at 5 fps
# B': SkillCorner supervises alive against dead only. On SkillCorner crops (training) and
# SkillCorner validation matches the per-bin loss is -log(pH + pA) on bins labelled H or A and
# -log pD on bins labelled D (sc_alive_dead_ce()); other providers keep the cross entropy.
SC_ALIVE_DEAD = False


# Round 5 feature versions (features.VERSIONS). FEATURE_VERSION is the version the model reads;
# the network's input width follows it. Each folder read (--cache, --val-cache, the aug and
# ball-free folders) must hold that version (marker VERSION_MARKER of build.py; a folder without
# a marker holds "stage50-v2"), or a version whose columns include it: the columns are then
# selected by name (generic; no version holds another's columns today). Default: round 4.
FEATURE_VERSION = features.FEATURE_VERSION
VERSION_MARKER = "feature_version.json"


def configure_features(version=None):
    """Set FEATURE_VERSION for this process (None: "stage50-v2"). Return it."""
    global FEATURE_VERSION
    FEATURE_VERSION = features.check_version(version or features.FEATURE_VERSION)
    return FEATURE_VERSION


def add_feature_args(parser):
    parser.add_argument("--feature-version", choices=features.VERSIONS, default=None,
                        help="R5: feature version the model reads (default stage50-v2); the "
                             "cache folders must hold it or a version that contains its columns")


def n_inputs():
    """The input width of the network: the number of columns of FEATURE_VERSION."""
    return features.n_features(FEATURE_VERSION)


def folder_version(folder):
    """The feature version of a cache folder: its build.py marker, "stage50-v2" without one."""
    path = os.path.join(folder, VERSION_MARKER)
    if not os.path.isfile(path):
        return features.FEATURE_VERSION
    import json
    with open(path) as handle:
        mark = json.load(handle)
    version = features.check_version(mark.get("feature_version"))
    if mark.get("columns") != features.feature_names(version):
        raise SystemExit("{}: the columns of the marker are not those of {}".format(folder,
                                                                                   version))
    return version


def folder_columns(folder):
    """None if `folder` holds FEATURE_VERSION, else the int columns of FEATURE_VERSION in it.

    SystemExit when the folder's version lacks a column of FEATURE_VERSION, or when it is
    another version with the same column names (stage50-v2 against stage50-v2s: the same names
    computed differently).
    """
    have = folder_version(folder)
    if have == FEATURE_VERSION:
        return None
    if features.feature_names(have) == features.feature_names(FEATURE_VERSION):
        raise SystemExit("{} holds feature version {}; the model reads {} (same columns, "
                         "computed differently: pass --feature-version {} or use another "
                         "folder)".format(folder, have, FEATURE_VERSION, have))
    try:
        return features.column_indices(features.feature_names(FEATURE_VERSION), have)
    except ValueError:
        raise SystemExit("{} holds feature version {}; the model reads {} (pass "
                         "--feature-version {} or use another folder)".format(
                             folder, have, FEATURE_VERSION, have))


class ColumnView(object):
    """Columns of a (bins, NF) array, taken on access: a crop reads only its rows."""

    def __init__(self, array, columns):
        self.array, self.columns = array, np.asarray(columns, dtype=np.int64)

    @property
    def shape(self):
        return (self.array.shape[0], int(self.columns.size))

    def __len__(self):
        return self.array.shape[0]

    def __getitem__(self, key):
        return np.asarray(self.array[key])[..., self.columns]

    def __array__(self, dtype=None, copy=None):
        out = np.asarray(self.array)[..., self.columns]
        return out if dtype is None else out.astype(dtype)


def select_columns(array, columns):
    """`array` itself (columns None), its columns now (an in-memory array) or a ColumnView (a
    memory-mapped array)."""
    if columns is None:
        return array
    if isinstance(array, np.memmap):
        return ColumnView(array, columns)
    return np.ascontiguousarray(np.asarray(array)[..., columns])


def crop_shares(train_games):
    """{provider: expected share of the crops} under the current MIX."""
    if MIX == "sc_quarter":
        w = {g: MATCH_WEIGHTS.get(folds.provider(g), 1.0) for g in train_games}
    elif MIX == "provider":
        provs = sorted(set(folds.provider(g) for g in train_games))
        w = {g: 1.0 / sum(folds.provider(h) == folds.provider(g) for h in train_games) / len(provs)
             for g in train_games}
    else:
        w = {g: 1.0 for g in train_games}
    tot = sum(w.values())
    out = {}
    for g, v in w.items():
        out[folds.provider(g)] = out.get(folds.provider(g), 0.0) + v / tot
    return {p: out[p] for p in sorted(out)}


def configure(target="ctl2", short_weight="off", mix="match", aug_cache=None, aug_k=None,
              views=None, sc_labels=None, dfl_still="kept", ballfree_cache=None,
              ballfree_share=None, val_cache=None, sc_alive_dead=False):
    """Set the round 3 and round 4 switches for this process. Defaults reproduce round 2 exactly.

    aug_cache: the aug_build.py folder (switch AUG on), or None (off, the default).
    Round 4 (all off by default): views (A), sc_labels (B), dfl_still (C), ballfree_cache and
    ballfree_share (D); see the constants above.
    """
    global TARGET, SHORT_WEIGHT, MIX, AUG_CACHE, AUG_K
    global TRAIN_VIEWS, SC_LABELS, DFL_STILL, BALLFREE_CACHE, BALLFREE_SHARE, VAL_CACHE
    global SC_ALIVE_DEAD
    if target not in TARGETS or short_weight not in SHORT_WEIGHTS or mix not in MIXES:
        raise ValueError("bad switch: target={} short_weight={} mix={}".format(
            target, short_weight, mix))
    if views is not None:
        views = tuple(views)
        if not views or "raw" not in views or not set(views) <= set(folds.FULL_VIEWS) \
                or len(set(views)) != len(views):
            raise ValueError("bad switch: views={} (a subset of {} that holds raw)".format(
                views, folds.FULL_VIEWS))
    if dfl_still not in DFL_STILLS:
        raise ValueError("bad switch: dfl_still={}".format(dfl_still))
    share = 0.15 if ballfree_share is None else float(ballfree_share)
    if not 0.0 < share < 1.0:
        raise ValueError("bad switch: ballfree_share={}".format(share))
    if ballfree_share is not None and ballfree_cache is None:
        raise ValueError("ballfree_share needs ballfree_cache")
    TARGET, SHORT_WEIGHT, MIX = target, short_weight, mix
    AUG_CACHE = aug_cache
    AUG_K = 16 if aug_k is None else int(aug_k)
    if dfl_still == "removed" and val_cache is None:
        raise ValueError("dfl_still removed needs val_cache (the dfl_c folder)")
    if ballfree_cache is not None and views is not None and set(views) - {"raw", "degraded"}:
        raise ValueError("the ball-free twins exist for raw, degraded and the aug copies only")
    TRAIN_VIEWS, SC_LABELS, DFL_STILL, VAL_CACHE = views, sc_labels, dfl_still, val_cache
    SC_ALIVE_DEAD = bool(sc_alive_dead)
    BALLFREE_CACHE, BALLFREE_SHARE = ballfree_cache, share
    out = {"target": TARGET, "short_weight": SHORT_WEIGHT, "mix": MIX}
    if AUG_CACHE is not None:
        out.update(aug_cache=AUG_CACHE, aug_k=AUG_K, aug_share=AUG_SHARE)
    out.update(r4_record().get("r4", {}))
    return out


def add_switch_args(parser):
    """The round 3 and round 4 training switches on an argparse parser (defaults: round 2)."""
    parser.add_argument("--target", choices=TARGETS, default="ctl2")
    parser.add_argument("--short-weight", choices=SHORT_WEIGHTS, default="off")
    parser.add_argument("--mix", choices=MIXES, default="match")
    parser.add_argument("--aug-cache", default=None,
                        help="R3 augmentation switch: the aug_build.py folder (default off)")
    parser.add_argument("--aug-k", type=int, default=None, help="aug views per match (16)")
    parser.add_argument("--views", nargs="+", choices=folds.FULL_VIEWS, default=None,
                        help="R4 step A: training views kept (default: all of folds.views())")
    parser.add_argument("--sc-labels", default=None,
                        help="R4 step B: folder whose labels/ hold the SkillCorner possession "
                             "groups (default: the labels of --cache)")
    parser.add_argument("--dfl-still", choices=DFL_STILLS, default="kept",
                        help="R4 step C: 'removed' requires the r4_inputs.py folders (markers "
                             "dfl_still.json: dfl_c_train for training, dfl_c for --val-cache)")
    parser.add_argument("--val-cache", default=None,
                        help="R4 step C: features of the validation matches (dfl_c folder)")
    parser.add_argument("--sc-alive-dead", action="store_true",
                        help="R4 arm B': SkillCorner supervises alive against dead only")
    parser.add_argument("--ballfree-cache", default=None,
                        help="R4 step D': r4_inputs.py ballfree folder (default off)")
    parser.add_argument("--ballfree-share", type=float, default=None,
                        help="R4 step D': share of crops with a ball-free window (0.15)")
    add_feature_args(parser)


def configure_args(args):
    """configure() from the arguments of add_switch_args()."""
    configure_features(getattr(args, "feature_version", None))
    return configure(args.target, args.short_weight, args.mix, args.aug_cache, args.aug_k,
                     views=args.views, sc_labels=args.sc_labels, dfl_still=args.dfl_still,
                     ballfree_cache=args.ballfree_cache, ballfree_share=args.ballfree_share,
                     val_cache=args.val_cache, sc_alive_dead=args.sc_alive_dead)


def r4_record():
    """The round 4 switches for the JSON records ({} when all are off)."""
    out = {}
    if TRAIN_VIEWS is not None:
        out["views"] = list(TRAIN_VIEWS)
    if SC_LABELS is not None:
        out["sc_labels"] = os.path.abspath(SC_LABELS)
    if DFL_STILL != "kept":
        out["dfl_still"] = DFL_STILL
    if VAL_CACHE is not None:
        out["val_cache"] = os.path.abspath(VAL_CACHE)
    if SC_ALIVE_DEAD:
        out["sc_alive_dead"] = True
    if BALLFREE_CACHE is not None:
        out["ballfree"] = {"cache": os.path.abspath(BALLFREE_CACHE), "share": BALLFREE_SHARE}
    if FEATURE_VERSION != features.FEATURE_VERSION:
        out["feature_version"] = FEATURE_VERSION
    return {"r4": out} if out else {}


def train_views(game):
    """The training views of a match under switch A (folds.views() when off)."""
    have = folds.views(game)
    if TRAIN_VIEWS is None:
        return have
    kept = tuple(v for v in have if v in TRAIN_VIEWS)
    if not kept:
        raise SystemExit("{}: no training view left by --views {}".format(game, TRAIN_VIEWS))
    return kept


def label_path(cache, game):
    """The label file of a match: SkillCorner from SC_LABELS when switch B is on."""
    base = SC_LABELS if (SC_LABELS is not None and folds.provider(game) == "skillcorner") \
        else cache
    return os.path.join(base, "labels", game + ".npz")


def still_marker(folder):
    """The step C marker dict of a cache folder, or None."""
    path = os.path.join(folder, STILL_MARKER)
    if not os.path.isfile(path):
        return None
    import json
    with open(path) as handle:
        return json.load(handle)


# Round 5 step J (R5_AUG.md, r5_inputs.py): a training folder may hold dfl_c_train_j instead of
# dfl_c_train; the training cache and the aug folder must then both hold it (never mixed).
TRAIN_INPUTS = ("c_train", "c_train_j", "c_train_j100")   # c_train_j100: J keeping every run (r6 A2)


def check_still(cache):
    """Switch C: the folders read must match DFL_STILL (markers of r4_inputs.py iff removed)."""
    want = [(cache, TRAIN_INPUTS)] + [(f, TRAIN_INPUTS) for f in (AUG_CACHE, BALLFREE_CACHE)
                                      if f is not None]
    if VAL_CACHE is not None:
        want.append((VAL_CACHE, ("c",)))
    seen = set()
    for f, dfl_input in want:
        mark = still_marker(f)
        if DFL_STILL == "removed" and (mark is None or mark.get("dfl_input") not in dfl_input):
            raise SystemExit("--dfl-still removed: {} is not a {} folder (marker {})".format(
                f, " or ".join(dfl_input), STILL_MARKER))
        if DFL_STILL == "removed" and dfl_input == TRAIN_INPUTS:
            seen.add(mark.get("dfl_input"))
        if DFL_STILL == "kept" and mark is not None:
            raise SystemExit("{} holds a still-ball input ({}); pass --dfl-still removed".format(
                f, mark.get("dfl_input")))
    if len(seen) > 1:
        raise SystemExit("training folders mix DFL inputs {} (step J needs the J cache and the "
                         "J aug folder together)".format(sorted(seen)))
    return DFL_STILL


def ballfree_path(folder, game, family):
    """r4_inputs.ballfree_path: family "raw", "degraded" or "k<k>"."""
    return os.path.join(folder, "feats", "ballfree", game, family + ".npy")


def load_ballfree(game, views, n_aug, shape):
    """{family: memory-mapped ball-free twin} of a training match (switch D')."""
    fams = list(views) + ["k{}".format(k) for k in range(n_aug)]
    out = {}
    for fam in fams:
        path = ballfree_path(BALLFREE_CACHE, game, fam)
        if not os.path.isfile(path):
            raise SystemExit("{}: no ball-free twin {} in {}".format(game, fam, BALLFREE_CACHE))
        out[fam] = select_columns(np.load(path, mmap_mode="r"), folder_columns(BALLFREE_CACHE))
        if out[fam].shape != shape:
            raise SystemExit("{}: ball-free twin {} and features differ in shape".format(
                game, fam))
    return out


def splice_ballfree(rng, x, twin, start):
    """x with one window of BALLFREE_MIN_BINS to len(x) bins taken from twin[start:...]."""
    n = x.shape[0]
    lo = min(BALLFREE_MIN_BINS, n)
    length = int(rng.randint(lo, n + 1))
    off = int(rng.randint(0, n - length + 1))
    x = np.array(x, copy=True)
    x[off:off + length] = twin[start + off:start + off + length]
    return x


def aug_record():
    """The augmentation switch for the JSON records ({} when off, so off records are unchanged)."""
    if AUG_CACHE is None:
        return {}
    man = os.path.join(AUG_CACHE, "manifest.json")
    return {"aug": {"cache": os.path.abspath(AUG_CACHE), "k": AUG_K, "share": AUG_SHARE,
                    "manifest_md5": md5_of(man) if os.path.isfile(man) else None}}


def load_aug(game):
    """[K memory-mapped (bins, NF) float32 arrays] of a match's aug views, [] if it has none."""
    from tools.learned_spike import aug_build
    paths = [aug_build.feat_path(AUG_CACHE, game, k) for k in range(AUG_K)]
    if not all(os.path.isfile(p) for p in paths):
        if any(os.path.isfile(p) for p in paths) or folds.provider(game) != "pff":
            raise SystemExit("{}: incomplete aug views in {}".format(game, AUG_CACHE))
        return []
    cols = folder_columns(AUG_CACHE)
    return [select_columns(np.load(p, mmap_mode="r"), cols) for p in paths]


def spell_lengths(bins):
    """[(start, stop)] of the H or A runs of a 5 fps label array (codes 0 H, 1 A, 2 D, 3 X)."""
    b = np.asarray(bins)
    if len(b) == 0:
        return []
    br = np.flatnonzero(b[1:] != b[:-1]) + 1
    st = np.concatenate(([0], br))
    sp = np.concatenate((br, [len(b)]))
    return [(int(a), int(e)) for a, e in zip(st, sp) if b[a] in (0, 1)]


def short_weights(bins, median):
    """float32 per bin: clip(sqrt(median / length), 1/3, 3) on possession bins, 1 elsewhere."""
    w = np.ones(len(bins), dtype=np.float32)
    for a, e in spell_lengths(bins):
        w[a:e] = np.clip(np.sqrt(median / float(e - a)), SHORT_CLIP[0], SHORT_CLIP[1])
    return w


def prepare_short_weights(data, train_games):
    """Attach item["short_w"] to every match of `data`; the median runs over the training
    matches only (all providers, target labels at 5 fps). Return the median (bins)."""
    lengths = [e - a for g in train_games for a, e in spell_lengths(data[g]["bins"])]
    median = float(np.median(lengths))
    for item in data.values():
        item["short_w"] = short_weights(item["bins"], median)
    return median


def recipe(steps, eval_every):
    """The recipe constants as a dict, for the JSON records."""
    return {"steps": steps, "crop_bins": CROP_BINS, "batch": BATCH, "lr": LR,
            "pct_start": PCT_START, "clip": CLIP, "input_noise": INPUT_NOISE,
            "boundary_bins": BOUNDARY_BINS, "boundary_weight": BOUNDARY_WEIGHT,
            "eval_every": eval_every, "params": count_params(),
            "target": TARGET, "short_weight": SHORT_WEIGHT, "mix": MIX, **aug_record(),
            **r4_record()}


def count_params():
    return int(sum(p.numel() for p in model.Net(n_in=n_inputs()).parameters()))


def setup(threads):
    """Fix the thread count and ask torch for deterministic kernels."""
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def load_games(cache, games, raw_only=False):
    """{game: {"feats": {view: F}, "views", "raw", "ctl2", "bins_ctl2"}} from the build.py cache.

    The matches must be training matches (folds.check_train_all()): the sealed and pilot PFF
    matches raise AssertionError. Each feature matrix must have one bin per label bin.
    """
    out = {}
    check_still(cache)
    for game in folds.check_train_all(folds.assert_not_sealed(games)):
        views = ("raw",) if raw_only else train_views(game)
        lab = np.load(label_path(cache, game))
        bins = features.bin_labels(lab["ctl2"])
        target_bins = bins if TARGET == "ctl2" else features.bin_labels(lab[TARGET])
        feats = {}
        fcache = VAL_CACHE if (raw_only and VAL_CACHE is not None) else cache
        cols = folder_columns(fcache)
        for v in views:
            with np.load(os.path.join(fcache, "feats", v, game + ".npz")) as npz:
                if "feature_version" in npz.files and \
                        str(npz["feature_version"]) != folder_version(fcache):
                    raise SystemExit("{} {}: feature file of version {} in a {} folder".format(
                        game, v, npz["feature_version"], folder_version(fcache)))
                feats[v] = select_columns(npz["F"], cols)
                if int(npz["n_frames"]) != len(lab["ctl2"]) or feats[v].shape[0] != len(bins):
                    raise SystemExit("{} {}: features and labels differ in length".format(game, v))
        out[game] = {"feats": feats, "views": tuple(views), "raw": lab["raw"],
                     "provider": folds.provider(game),
                     "ctl2": lab["ctl2"], "bins_ctl2": bins, "bins": target_bins}
        if AUG_CACHE is not None and not raw_only:
            aug = load_aug(game)
            if any(a.shape != feats["raw"].shape for a in aug):
                raise SystemExit("{}: aug views and features differ in shape".format(game))
            out[game]["aug"] = aug
        if BALLFREE_CACHE is not None and not raw_only:
            out[game]["ballfree"] = load_ballfree(game, views, len(out[game].get("aug", [])),
                                                  feats["raw"].shape)
    return out


def cache_files(cache, games, views_of=None):
    """The cache files that training on `games` reads, as paths (labels under switch B)."""
    views_of = train_views if views_of is None else views_of
    paths = []
    for game in games:
        paths.append(label_path(cache, game))
        paths += [os.path.join(cache, "feats", v, game + ".npz") for v in views_of(game)]
    return paths


def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_batch(rng, data, train_games, forced_pool=None, forced_n=0):
    """(x, y, w) tensors of one batch (see the module docstring).

    Round 8 fine-tune (finetune_model()): with forced_pool, the first forced_n crops of the
    batch draw their match uniformly from forced_pool instead of the MIX; every other step of a
    crop (view or aug copy, start, symmetric copy, noise, weights) is unchanged. The default
    (None, 0) is the sampling of every earlier round, random draws included.
    """
    xs, ys, ws, scs = [], [], [], []
    for i in range(BATCH):
        if forced_pool and i < forced_n:
            game = forced_pool[rng.randint(len(forced_pool))]
        elif MIX == "provider":
            provs = sorted(set(folds.provider(g) for g in train_games))
            prov = provs[rng.randint(len(provs))]
            pool = [g for g in train_games if folds.provider(g) == prov]
            game = pool[rng.randint(len(pool))]
        elif MIX == "sc_quarter":
            w = np.array([MATCH_WEIGHTS.get(folds.provider(g), 1.0) for g in train_games])
            c = np.cumsum(w)
            game = train_games[min(int(np.searchsorted(c, rng.random_sample() * c[-1],
                                                        side="right")), len(train_games) - 1)]
        else:
            game = train_games[rng.randint(len(train_games))]
        item = data[game]
        if AUG_CACHE is not None and item.get("aug") and rng.random_sample() < AUG_SHARE:
            k = rng.randint(len(item["aug"]))
            feats, family = item["aug"][k], "k{}".format(k)
        else:
            view = item["views"][rng.randint(len(item["views"]))]
            feats, family = item["feats"][view], view
        n = feats.shape[0]
        start = rng.randint(0, max(1, n - CROP_BINS))
        x = np.asarray(feats[start:start + CROP_BINS])
        if BALLFREE_CACHE is not None and rng.random_sample() < BALLFREE_SHARE:
            x = splice_ballfree(rng, x, item["ballfree"][family], start)
        y = item["bins"][start:start + CROP_BINS].astype(np.int64)
        swap, mirror = bool(rng.randint(2)), bool(rng.randint(2))
        x = features.transform(x, swap, mirror)
        if swap:
            y = model.swap_labels(y)
        w = boundary_weights(y)
        if SHORT_WEIGHT == "sqrt":
            w = w * item["short_w"][start:start + CROP_BINS]
        y = np.where(y == 3, 0, y)
        xs.append(x)
        ys.append(y)
        ws.append(w)
        scs.append(folds.provider(game) == "skillcorner" if SC_ALIVE_DEAD else False)
    x = torch.from_numpy(np.stack(xs)).float()
    x = x + INPUT_NOISE * torch.from_numpy(rng.standard_normal(x.shape).astype(np.float32))
    y, w = torch.from_numpy(np.stack(ys)), torch.from_numpy(np.stack(ws))
    if SC_ALIVE_DEAD:
        return x, y, w, torch.from_numpy(np.array(scs, dtype=bool))
    return x, y, w


def sc_alive_dead_ce(logits, y, sc):
    """Per-bin loss (N, T): cross entropy, but -log(pH + pA) on alive bins and -log pD on dead
    bins of the rows where sc (N,) is True (arm B'). y: targets with X already mapped to 0."""
    logp = torch.log_softmax(logits, dim=-1)
    ce = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    alive = -torch.logsumexp(logp[..., :2], dim=-1)
    dead = -logp[..., 2]
    ad = torch.where(y == 2, dead, alive)
    return torch.where(sc.view(-1, *([1] * (y.dim() - 1))).expand_as(y), ad, ce)


@torch.no_grad()
def val_loss(net, item):
    """Weighted cross entropy of one match, raw view, one copy (as lomo.val_loss)."""
    net.eval()
    x = torch.from_numpy(item["feats"]["raw"])[None].float()
    y = item["bins"].astype(np.int64)
    w = torch.from_numpy(boundary_weights(y))
    logits = net(x)[0]
    yt = torch.from_numpy(np.where(y == 3, 0, y))
    if SC_ALIVE_DEAD and item.get("provider") == "skillcorner":
        ce = sc_alive_dead_ce(logits[None], yt[None], torch.tensor([True]))[0]
    else:
        ce = torch.nn.functional.cross_entropy(logits, yt, reduction="none")
    return float((ce * w).sum() / w.sum())


def train_model(data, train_games, steps, seed, val_games=(), eval_every=EVAL_EVERY, log=None,
                stop_at=None):
    """Train one model. Return (state_dict, record).

    The one-cycle schedule always spans `steps`. With stop_at, training stops after step stop_at
    of that schedule and keeps the weights there (the final models, LEARNED_STAGE_PROTOCOL §2).

    With validation matches the state is the checkpoint of the lowest mean validation loss
    (checked every eval_every steps and at the last step). Without them it is the last step.
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    short_median = prepare_short_weights(data, train_games) if SHORT_WEIGHT == "sqrt" else None
    net = model.Net(n_in=n_inputs())
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps,
                                                pct_start=PCT_START)
    best = (float("inf"), None, -1)
    history = []
    t0 = time.time()
    t_eval = 0.0
    last = steps if stop_at is None else int(stop_at)
    for step in range(1, last + 1):
        net.train()
        batch = sample_batch(rng, data, train_games)
        x, y, w = batch[:3]
        logits = net(x)
        if SC_ALIVE_DEAD:
            ce = sc_alive_dead_ce(logits, y, batch[3]).reshape(-1)
        else:
            ce = torch.nn.functional.cross_entropy(logits.reshape(-1, model.N_CLASSES),
                                                   y.reshape(-1), reduction="none")
        loss = (ce * w.reshape(-1)).sum() / w.sum().clamp_min(1.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        opt.step()
        sched.step()
        check = bool(val_games) and (step % eval_every == 0 or step == last)
        if check or step % 50 == 0 or step == last:
            entry = {"step": step, "train": round(float(loss.detach()), 6),
                     "elapsed": round(time.time() - t0, 2)}
            if check:
                te = time.time()
                losses = [val_loss(net, data[g]) for g in val_games]
                t_eval += time.time() - te
                entry["val"] = [round(v, 6) for v in losses]
                entry["val_mean"] = float(np.mean(losses))
                if entry["val_mean"] < best[0]:
                    best = (entry["val_mean"], {k: v.clone() for k, v in net.state_dict().items()},
                            step)
            history.append(entry)
            if log is not None:
                log("seed {} step {} train {:.4f}{} {:.2f}s".format(
                    seed, step, entry["train"],
                    " val {:.4f}".format(entry["val_mean"]) if check else "", entry["elapsed"]))
    seconds = time.time() - t0
    if val_games:
        state, best_step, best_val = best[1], best[2], best[0]
    else:
        state, best_step, best_val = net.state_dict(), last, None
    record = {"seed": seed, "steps": steps, "stop_at": stop_at, "best_step": best_step, "best_val_mean": best_val,
              "train_seconds": round(seconds, 2), "eval_seconds": round(t_eval, 2),
              "seconds_per_step": round((seconds - t_eval) / last, 4), "history": history,
              "switches": {"target": TARGET, "short_weight": SHORT_WEIGHT, "mix": MIX,
                           "short_median_bins": short_median, **aug_record(), **r4_record()}}
    return state, record


# Round 8 arm N (LEARNED_STAGE_PROTOCOL_R8.md, Section 3 row N): a short fine-tune of trained
# networks with a fixed share of each batch drawn from one provider's matches. Settings fixed
# before any result: share 0.5 of Metrica, peak learning rate FT_LR (a quarter of LR) with the
# one-cycle shape of the recipe, FT_STEPS in {200, 400}. The weights of the last step are kept.
FT_LR = 5e-4
FT_SHARE = 0.5
FT_STEPS = (200, 400)
FT_PROVIDER = "metrica"


FT_LOSSES = ("full", "team")   # the loss on the forced crops (N: full; N2, note 6.3: team)


def team_only_ce(logits, y):
    """Per-bin loss (N, T) of the team call alone: -log softmax over the H and A logits at the
    label's team (log pH - log(pH + pA)). Defined on every bin; finetune_model() gives it weight
    only on bins labelled H or A. y: targets with X already mapped to 0."""
    logp2 = torch.log_softmax(logits[..., :2], dim=-1)
    return -logp2.gather(-1, y.clamp(max=1).unsqueeze(-1)).squeeze(-1)


def finetune_model(data, train_games, steps, seed, init_state, lr=FT_LR, share=FT_SHARE,
                   provider=FT_PROVIDER, log=None, forced_loss="full"):
    """Fine-tune a trained network. Return (state_dict, record).

    The network starts from init_state (a model.Net state dictionary; no optimiser state).
    Adam is new, with the one-cycle schedule of the recipe (PCT_START, gradient clip CLIP) over
    `steps` at peak `lr`. In every batch round(share x BATCH) crops come from the matches of
    `provider` in train_games (uniform over those matches), the rest from the recipe's MIX over
    all of train_games (sample_batch(forced_pool=...)). The loss, the switches of configure()
    and the crop steps are those of train_model(). No validation: the last step is kept.

    forced_loss "team" (round 8 N2, protocol R8 note 6.3): on the forced crops (the first
    round(share x BATCH) rows of each batch) the per-bin loss is team_only_ce() and the weight
    of every bin labelled D is set to 0 (X already has 0), so those crops teach home against
    away on live bins only and nothing about dead against live; their live bins keep the
    recipe's boundary weights. The other rows keep the recipe's loss. The batch loss is, as
    for "full", the weighted mean over every weighted bin of the batch: sum(loss x w) / sum(w).
    The same seed draws the same crops under both settings (only the loss differs).
    """
    if forced_loss not in FT_LOSSES:
        raise ValueError("forced_loss must be one of {}".format(FT_LOSSES))
    pool = [g for g in train_games if folds.provider(g) == provider]
    if not pool:
        raise SystemExit("no {} match in the training games".format(provider))
    n_forced = int(round(share * BATCH))
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    if SHORT_WEIGHT == "sqrt":
        prepare_short_weights(data, train_games)
    net = model.net_for_state(init_state)
    if net.n_in != n_inputs():
        raise SystemExit("the initial network reads {} columns, the feature version {} has "
                         "{}".format(net.n_in, FEATURE_VERSION, n_inputs()))
    net.load_state_dict(init_state)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=PCT_START)
    history = []
    t0 = time.time()
    for step in range(1, steps + 1):
        net.train()
        batch = sample_batch(rng, data, train_games, forced_pool=pool, forced_n=n_forced)
        x, y, w = batch[:3]
        logits = net(x)
        if SC_ALIVE_DEAD:
            ce = sc_alive_dead_ce(logits, y, batch[3])
        else:
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, model.N_CLASSES), y.reshape(-1),
                reduction="none").reshape(y.shape)
        if forced_loss == "team":
            ce = ce.clone()
            ce[:n_forced] = team_only_ce(logits[:n_forced], y[:n_forced])
            w = w.clone()
            w[:n_forced] = torch.where(y[:n_forced] == 2, torch.zeros_like(w[:n_forced]),
                                       w[:n_forced])
        loss = (ce.reshape(-1) * w.reshape(-1)).sum() / w.sum().clamp_min(1.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == steps:
            entry = {"step": step, "train": round(float(loss.detach()), 6),
                     "elapsed": round(time.time() - t0, 2)}
            history.append(entry)
            if log is not None:
                log("finetune seed {} step {} train {:.4f} {:.2f}s".format(
                    seed, step, entry["train"], entry["elapsed"]))
    seconds = time.time() - t0
    record = {"seed": seed, "steps": steps, "stop_at": None, "best_step": steps,
              "best_val_mean": None, "train_seconds": round(seconds, 2),
              "seconds_per_step": round(seconds / steps, 4), "history": history,
              "finetune": {"lr": lr, "share": share, "provider": provider, "pool": pool,
                           "forced_loss": forced_loss,
                           "forced_crops_per_batch": n_forced, "pct_start": PCT_START},
              "switches": {"target": TARGET, "short_weight": SHORT_WEIGHT, "mix": MIX,
                           **aug_record(), **r4_record()}}
    return net.state_dict(), record


def net_version(path):
    """The feature version a saved network was trained on, from the record beside it
    (r4_screen: model.json; kfold: model_s<s>.json; final: final_s<s>.json; the round 4
    record's feature_version, absent = stage50-v2). None when there is no record."""
    import json
    folder, name = os.path.split(os.path.abspath(path))
    cands = [os.path.join(folder, "model.json")]
    if name.startswith("net_s") and name.endswith(".pt"):
        s = name[len("net_s"):-len(".pt")]
        cands = [os.path.join(folder, "model_s{}.json".format(s)),
                 os.path.join(folder, "final_s{}.json".format(s))] + cands
    for rec in cands:
        if os.path.isfile(rec):
            with open(rec) as handle:
                r = json.load(handle)
            sw = r.get("switches") or (r.get("recipe") or {})
            r4 = sw.get("r4") or {}
            return features.check_version(r4.get("feature_version") or features.FEATURE_VERSION)
    return None


def load_net(path):
    state = torch.load(path, weights_only=True)
    net = model.net_for_state(state)
    net.load_state_dict(state)
    net.eval()
    return net


@torch.no_grad()
def predict_p5(net, feats):
    """float64 (bins, 3) mean probabilities of one model over the four symmetric copies."""
    net64 = model.Net(n_in=net.inp[0].in_features, hidden=net.inp[0].out_features).double()
    net64.load_state_dict({k: v.double() for k, v in net.state_dict().items()})
    net64.eval()
    total = np.zeros((feats.shape[0], model.N_CLASSES))
    for swap, mirror in model.SYMMETRIES:
        x = torch.from_numpy(np.ascontiguousarray(features.transform(feats, swap, mirror)))
        p = torch.softmax(net64(x[None].double())[0], dim=-1).numpy()
        if swap:
            p = p[:, [1, 0, 2]]
        total += p
    return total / len(model.SYMMETRIES)


def environment(threads):
    """Provenance: git commit of the training code, versions, thread count."""
    def git(*args):
        try:
            return subprocess.check_output(("git",) + args, cwd=ROOT,
                                           stderr=subprocess.DEVNULL).decode().strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    dirty = git("status", "--porcelain", "--", "tools/learned_spike", "engine", "games.py")
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(dirty),
            "torch": torch.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "machine": platform.machine(),
            "threads": threads}


def reference_md5(games, ext):
    """{game: md5} of the reference CSV of each match (external_sweep.prepare_reference).

    A PFF training match uses pff.reference(), which checks the md5 of R2 Section 8.
    """
    from analysis import external_sweep as xs
    refs = os.path.abspath(os.path.expanduser(xs.DEFAULT_REFS_ROOT))
    sample = os.path.abspath(os.path.expanduser(xs.DEFAULT_SAMPLE_ROOT))
    out = {}
    for g in folds.check_train_all(folds.assert_not_sealed(games)):
        if g.startswith("PFF-"):
            out[g] = pff.reference(g, refs)[1]
        else:
            out[g] = xs.prepare_reference(folds.check(g), refs, sample, ext)[1]
    return out


def check_cache(cache):
    """Raise AssertionError if the build.py cache holds a sealed or pilot PFF match (R2 §1)."""
    names = []
    for sub in [os.path.join(cache, "labels")] + [
            os.path.join(cache, "feats", v) for v in folds.FULL_VIEWS]:
        if os.path.isdir(sub):
            names += [n[:-len(".npz")] for n in os.listdir(sub) if n.endswith(".npz")]
    folds.assert_not_sealed(sorted(set(names)))
    return len(set(names))


def check_pff_references():
    """pff.check_references() on the default refs root: no training if one md5 differs."""
    from analysis import external_sweep as xs
    return pff.check_references(os.path.abspath(os.path.expanduser(xs.DEFAULT_REFS_ROOT)))
