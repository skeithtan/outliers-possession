"""Round 4 inputs: DFL still ball removed (step C) and the ball-free twins (step D'). Not part of the submission.

    # scoring / validation cache: DFL with the still-ball rule alone (input dfl_c)
    python -m tools.learned_spike.r4_inputs cache --kind c --base CACHE --engine-cache DIR --out NEW
    # training cache: DFL raw and degraded from input dfl_c_train
    python -m tools.learned_spike.r4_inputs cache --kind c_train --base CACHE --engine-cache DIR --out NEW
    # aug copies: DFL rebuilt from dfl_c_train, the other matches linked from the round 3 folder
    python -m tools.learned_spike.r4_inputs aug --base AUG --labels CACHE --engine-cache DIR --out NEW
    # the same with the ball component (augment.py 8): Metrica rebuilt too, SkillCorner linked
    python -m tools.learned_spike.r4_inputs aug --ball ... [--games G ...]; mark --kind aug --ball
    # ball-free twins of every training view (raw, degraded, the 16 aug copies) of the 28 matches
    python -m tools.learned_spike.r4_inputs ballfree --labels CACHE --engine-cache DIR --out NEW \
        [--games G ...]
    # after all builders: completeness check, aug manifest, marker file
    python -m tools.learned_spike.r4_inputs mark --kind c|c_train|aug|ballfree --dir NEW

LEARNED_STAGE_PROTOCOL_R4.md Section 1 with the coordinator's changes of 13 November 2026:

- Input `dfl_c` (load_dfl(mode="c")): the still-ball rule of tools/still_ball.py alone. Used for
  validation, scoring (test T-D) and the rule engine's runs (still_ball.py writes the bundle).
- Input `dfl_c_train` (mode "c_train"), every DFL TRAINING view, in this order:
  1. the still-ball rule (dfl_c);
  2. (a) label-free ball gaps: gaps of log-uniform length GAP_S (1 to 12 s) placed uniformly on
     frames where the ball is visible, until GAP_SHARE (10 %) of the visible-ball frames are
     hidden (degrade.py step 2 with the visible-ball mask in place of the alive mask);
  3. (b) each gap of the still-ball rule is extended by U(0, EXTEND_MAX_S) s (1.5 s) after the
     ball moves again, so the ball does not come back on the exact restart frame.
  One numpy RandomState per match, fixed seed train_seed() (crc32 of the name and a suffix, so
  the draws differ from degrade.py's for the same match). The `degraded` view (degrade.py) and
  the 16 aug copies (augment.py) of a DFL match are then made from this input.
- Ball-free twins (step D'): for each training view of each of the 28 matches (raw, degraded,
  aug copy k), the same tracking with the ball missing on every frame, through the normal
  feature code (features.frame_features), never by zeroing columns. DFL twins come from
  dfl_c_train. train.py splices a window of a twin into a crop (switch --ballfree-cache).

The caches are new folders: links to the base cache for everything that does not change, real
files for what does. Existing caches are never written. Markers (train.STILL_MARKER) tell
train.py which DFL input a folder holds. The two sealed DFL matches, the competition match and
every PFF match are refused (still_ball.check_game, build.check, aug_build.check_games).
Python 3.9.
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
import still_ball  # noqa: E402
from tools.learned_spike import degrade, folds  # noqa: E402

GAP_S = (1.0, 12.0)          # (a): as degrade.LIVE_GAP_S
GAP_SHARE = 0.10             # (a): as degrade.LIVE_GAP_SHARE, of the visible-ball frames
EXTEND_MAX_S = 1.5           # (b)
FPS = 25
MODES = ("raw", "c", "c_train")
KINDS = ("c", "c_train", "aug", "ballfree")
MARKER = "dfl_still.json"    # train.STILL_MARKER


def train_seed(game):
    """Fixed seed of the dfl_c_train draws of a match."""
    return zlib.crc32((game + "/r4-dfl-c-train").encode("utf-8")) & 0x7FFFFFFF


def _runs(mask):
    padded = np.concatenate(([False], np.asarray(mask, bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[0::2], edges[1::2]


def train_gaps(bx, by, still, rng):
    """(bx, by, report) after steps (a) and (b) of dfl_c_train on arrays where the still frames
    (bool mask `still`) are already missing. Copies; the inputs are not changed."""
    bx = np.array(bx, dtype=np.float32)
    by = np.array(by, dtype=np.float32)
    n = len(bx)
    vis_idx = np.flatnonzero(np.isfinite(bx) & np.isfinite(by))
    target = int(GAP_SHARE * vis_idx.size)
    hidden, tries = 0, 0
    while hidden < target and tries < 100000 and vis_idx.size:
        tries += 1
        length = int(degrade._loguniform(rng, *GAP_S) * FPS)
        s = int(vis_idx[rng.randint(0, vis_idx.size)])
        e = min(n, s + length)
        newly = int(np.isfinite(bx[s:e]).sum())
        bx[s:e] = np.nan
        by[s:e] = np.nan
        hidden += newly
    extended = 0
    for a, b in zip(*_runs(still)):
        ext = int(rng.uniform(0, EXTEND_MAX_S * FPS))
        e = min(n, b + ext)
        extended += int(np.isfinite(bx[b:e]).sum())
        bx[b:e] = np.nan
        by[b:e] = np.nan
    return bx, by, {"gap_target": target, "gap_hidden": hidden, "extended": extended}


def load_dfl(tac, game, mode):
    """(tactical dict, report) of a match under an input mode.

    "raw": unchanged. "c": the still-ball rule (DFL only). "c_train": the rule, then (a) and (b)
    (DFL only). Matches of other providers are returned unchanged in every mode.
    """
    if mode not in MODES:
        raise ValueError("input mode {!r} not in {}".format(mode, MODES))
    if mode == "raw" or not game.startswith("DFL-"):
        return tac, {"mode": "raw"}
    out, still = still_ball.remove_still(tac, game, "removed")
    rep = {"mode": mode, "still_frames": int(still.sum())}
    if mode == "c_train":
        bx, by, r = train_gaps(out["bx"], out["by"], still,
                               np.random.RandomState(train_seed(game)))
        out = dict(out, bx=bx, by=by)
        rep.update(r)
    rep["ball_missing"] = int((~np.isfinite(out["bx"])).sum())
    return out, rep


def ball_free(tac):
    """A copy of a tactical dict with the ball missing on every frame."""
    return dict(tac, bx=np.full(len(tac["bx"]), np.nan, dtype=np.float32),
                by=np.full(len(tac["by"]), np.nan, dtype=np.float32))


# -- builders ------------------------------------------------------------------------------

def _link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.lexists(dst):
        os.symlink(os.path.abspath(src), dst)


def _save_npz(path, f25, f5):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, F=f5, n_frames=np.int64(f25.shape[0]))
    os.replace(tmp, path)


def _save_npy(path, f5):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path + ".tmp.npy", f5)
    os.replace(path + ".tmp.npy", path)


def _raw_tac(game, engine_cache):
    from tools.learned_spike import build
    build.check(game)
    return build.load_view(game, "raw", engine_cache)


def _codes(labels, game):
    return np.load(os.path.join(labels, "labels", game + ".npz"))["raw"]


def _features(tac, game, n_codes):
    from tools.learned_spike import features
    f25 = features.frame_features(tac, game)
    if f25.shape[0] != n_codes:
        raise SystemExit("{}: {} feature frames, {} reference frames".format(
            game, f25.shape[0], n_codes))
    return f25, features.downsample(f25)


def build_cache(kind, base, engine_cache, out, force=False):
    """Cache folder `out`: links to `base` for labels and non-DFL features, DFL rebuilt."""
    if os.path.abspath(out) == os.path.abspath(base):
        raise SystemExit("--out must be a new folder")
    views = ("raw",) if kind == "c" else ("raw", "degraded")
    reports = {}
    for game in folds.ALL:
        _link(os.path.join(base, "labels", game + ".npz"),
              os.path.join(out, "labels", game + ".npz"))
        for v in views:
            dst = os.path.join(out, "feats", v, game + ".npz")
            if not game.startswith("DFL-"):
                _link(os.path.join(base, "feats", v, game + ".npz"), dst)
    for game in folds.OPEN:
        if not game.startswith("DFL-"):
            continue
        still_ball.check_game(game)
        if not force and all(os.path.isfile(os.path.join(out, "feats", v, game + ".npz"))
                             for v in views):
            continue
        t0 = time.time()
        codes = _codes(base, game)
        tac, rep = load_dfl(_raw_tac(game, engine_cache), game, kind)
        for v in views:
            t = tac if v == "raw" else degrade.degrade(tac, codes, game)
            f25, f5 = _features(t, game, len(codes))
            _save_npz(os.path.join(out, "feats", v, game + ".npz"), f25, f5)
        rep["seconds"] = round(time.time() - t0, 1)
        reports[game] = rep
        print(game, json.dumps(rep), file=sys.stderr, flush=True)
    _merge_reports(out, reports)
    return reports


def build_aug(base, labels, engine_cache, out, k_views=16, force=False, ball=False,
              games=None):
    """Aug folder `out`: DFL copies from dfl_c_train, other matches linked to `base`.

    ball=True (switch BALL, augment.py component 8, R4_AUG_BALL.md): the Metrica copies are
    rebuilt too (from the raw input, as aug_build.py builds them) with the component on, and
    the DFL copies get it as well; SkillCorner stays linked (the component leaves it as it is).
    `games` limits the rebuilt matches (parallel builders); links are always written.
    """
    from tools.learned_spike import aug_build, augment
    from engine import data
    reports = {}
    rebuilt = [g for g in folds.OPEN if g.startswith("DFL-")
               or (ball and augment.ball_applies(g))]
    for game in folds.ALL:
        if game not in rebuilt:
            _link(os.path.join(base, "feats", "aug", game),
                  os.path.join(out, "feats", "aug", game))
    for game in rebuilt:
        if games is not None and game not in games:
            continue
        if game.startswith("DFL-"):
            still_ball.check_game(game)
        codes = _codes(labels, game)
        tac = None
        for k in range(k_views):
            path = aug_build.feat_path(out, game, k)
            if os.path.isfile(path) and not force:
                continue
            if tac is None:
                tac, rep = load_dfl(_raw_tac(game, engine_cache), game, "c_train")
                reports[game] = rep
            aug, arep = augment.augment(tac, codes, game, k, data.graded_blocks(game),
                                        data.pitch_m(game), ball=ball)
            if ball:
                reports[game].setdefault("ball", {})["k{}".format(k)] = {
                    "plan": arep["plan"].get("ball"), "result": arep.get("ball")}
            _, f5 = _features(aug, game, len(codes))
            _save_npy(path, f5)
        print(game, "aug done", file=sys.stderr, flush=True)
    _merge_reports(out, reports)
    return reports


def ballfree_path(out, game, family):
    """The twin of family "raw", "degraded" or "k<k>" (train.ballfree_path)."""
    return os.path.join(out, "feats", "ballfree", game, family + ".npy")


def build_ballfree(games, labels, engine_cache, out, k_views=16, force=False):
    """Ball-free twins of raw, degraded and the aug copies; DFL from dfl_c_train."""
    from tools.learned_spike import augment
    from engine import data
    for game in games:
        if game not in folds.ALL:
            raise SystemExit("{}: ball-free twins for the 28 external matches only".format(game))
        t0 = time.time()
        codes = _codes(labels, game)
        fams = ["raw", "degraded"] + ["k{}".format(k) for k in range(k_views)]
        todo = [f for f in fams if force or not os.path.isfile(ballfree_path(out, game, f))]
        if not todo:
            continue
        tac, _ = load_dfl(_raw_tac(game, engine_cache), game, "c_train")
        blocks, pitch = data.graded_blocks(game), data.pitch_m(game)
        for fam in todo:
            if fam == "raw":
                t = tac
            elif fam == "degraded":
                t = degrade.degrade(tac, codes, game)
            else:
                t = augment.augment(tac, codes, game, int(fam[1:]), blocks, pitch)[0]
            _, f5 = _features(ball_free(t), game, len(codes))
            _save_npy(ballfree_path(out, game, fam), f5)
        print(game, "ballfree done {:.1f}s".format(time.time() - t0), file=sys.stderr,
              flush=True)


def _merge_reports(out, reports):
    if not reports:
        return
    path = os.path.join(out, "reports_r4.json")
    old = {}
    if os.path.isfile(path):
        with open(path) as h:
            old = json.load(h)
    old.update(reports)
    os.makedirs(out, exist_ok=True)
    with open(path, "w") as h:
        json.dump(old, h, indent=1, sort_keys=True)


def mark(kind, folder, k_views=16, ball=False):
    """Check that a folder is complete for its kind, then write the marker (and aug manifest)."""
    from tools.learned_spike import aug_build
    missing = []
    for game in folds.ALL:
        if kind in ("c", "c_train"):
            views = ("raw",) if kind == "c" else ("raw", "degraded")
            paths = [os.path.join(folder, "labels", game + ".npz")] + [
                os.path.join(folder, "feats", v, game + ".npz") for v in views]
        elif kind == "aug":
            paths = [aug_build.feat_path(folder, game, k) for k in range(k_views)]
        else:
            paths = [ballfree_path(folder, game, f) for f in
                     ["raw", "degraded"] + ["k{}".format(k) for k in range(k_views)]]
        missing += [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit("{} incomplete: {} files missing, e.g. {}".format(
            folder, len(missing), missing[:3]))
    rec = {"dfl_still": "removed", "dfl_input": "c" if kind == "c" else "c_train",
           "kind": kind, "games": [g for g in folds.ALL if g.startswith("DFL-")],
           "min_frames": still_ball.MIN_FRAMES, "gap_s": GAP_S, "gap_share": GAP_SHARE,
           "extend_max_s": EXTEND_MAX_S,
           "still_ball_md5": _md5(os.path.join(ROOT, "tools", "still_ball.py")),
           "r4_inputs_md5": _md5(os.path.abspath(__file__).replace(".pyc", ".py"))}
    if kind == "aug" and ball:
        from tools.learned_spike import augment
        rec["ball"] = {"sigma": augment.BALL_SIGMA, "offset_m": augment.BALL_OFFSET_M,
                       "tau_s": augment.BALL_TAU_S, "cap": augment.BALL_OFFSET_CAP,
                       "p_on": augment.P_ON, "providers": list(augment.BALL_PROVIDERS),
                       "doc": "docs/plan/calibration/R4_AUG_BALL.md"}
    if kind == "aug":
        rec["manifest_files"] = len(aug_build.write_manifest(folder, list(folds.ALL),
                                                             k_views)["files"])
    with open(os.path.join(folder, MARKER), "w") as h:
        json.dump(rec, h, indent=1, sort_keys=True)
    return rec


def _md5(path):
    import hashlib
    d = hashlib.md5()
    with open(path, "rb") as h:
        for block in iter(lambda: h.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--kind", choices=("c", "c_train"), required=True)
    c.add_argument("--base", required=True)
    c.add_argument("--engine-cache", required=True)
    c.add_argument("--out", required=True)
    a = sub.add_parser("aug")
    a.add_argument("--base", required=True, help="round 3 aug_build.py folder (read only)")
    a.add_argument("--labels", required=True)
    a.add_argument("--engine-cache", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--ball", action="store_true",
                   help="switch BALL: augment.py component 8 on (R4_AUG_BALL.md)")
    a.add_argument("--games", nargs="+", default=None,
                   help="rebuild only these matches (links are always written)")
    b = sub.add_parser("ballfree")
    b.add_argument("--labels", required=True)
    b.add_argument("--engine-cache", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--games", nargs="+", default=list(folds.ALL))
    m = sub.add_parser("mark")
    m.add_argument("--kind", choices=KINDS, required=True)
    m.add_argument("--dir", required=True)
    m.add_argument("--ball", action="store_true", help="aug folder built with --ball")
    for s in (c, a, b):
        s.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "cache":
        build_cache(args.kind, args.base, args.engine_cache, args.out, args.force)
    elif args.cmd == "aug":
        build_aug(args.base, args.labels, args.engine_cache, args.out, force=args.force,
                  ball=args.ball, games=args.games)
    elif args.cmd == "ballfree":
        build_ballfree(args.games, args.labels, args.engine_cache, args.out, force=args.force)
    else:
        print(json.dumps(mark(args.kind, args.dir, ball=args.ball), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
