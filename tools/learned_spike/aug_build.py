"""Build the aug views of the round 3 augmentation switch (augment.py). Not part of the submission.

    python -m tools.learned_spike.aug_build --out DIR --labels-cache DIR --engine-cache DIR
        [--games G ...] [--k 16] [--manifest]

For each training match and copy k it writes DIR/feats/aug/<game>/k<k>.npy (float32 5 fps
feature matrix, one bin per label bin, the features of features.py on augment.augment()) and
DIR/reports/<game>.json (the defect plan and event counts of every copy). The reference codes
come from LABELS/labels/<game>.npz (`raw`, the build.py label cache, read only). With
--manifest it writes DIR/manifest.json: {relative path: md5} of every feature file, after all
builders have finished.

The default --games is the 28 external training matches (folds.ALL): no PFF match, no sealed
match, never the competition match (build.check()). The tool checks before it starts that no
seed of a copy equals a seed of corrupt.py's stress view (augment.check_seeds()). Python 3.9.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from engine import data  # noqa: E402
from tools.learned_spike import augment, build, features, folds  # noqa: E402

DEFAULT_K = 16


def feat_path(out, game, k):
    return os.path.join(out, "feats", "aug", game, "k{}.npy".format(int(k)))


def check_games(games):
    for g in games:
        build.check(g)
        if folds.provider(g) == "pff" or g not in folds.ALL:
            raise SystemExit("{}: the aug views are for the 28 external matches only".format(g))
    return games


def build_game(game, out, labels_cache, engine_cache, k_views, force=False):
    check_games([game])
    codes = np.load(os.path.join(labels_cache, "labels", game + ".npz"))["raw"]
    n_bins = len(features.bin_labels(codes))
    tac = None
    reports = []
    for k in range(k_views):
        path = feat_path(out, game, k)
        if os.path.isfile(path) and not force:
            continue
        if tac is None:
            tac = build.load_view(game, "raw", engine_cache)
            blocks, pitch = data.graded_blocks(game), data.pitch_m(game)
        t0 = time.time()
        aug, rep = augment.augment(tac, codes, game, k, blocks, pitch)
        f25 = features.frame_features(aug, game)
        if f25.shape[0] != len(codes):
            raise SystemExit("{} k{}: {} frames, {} reference frames".format(
                game, k, f25.shape[0], len(codes)))
        f5 = features.downsample(f25)
        assert f5.shape[0] == n_bins
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path + ".tmp.npy", f5)
        os.replace(path + ".tmp.npy", path)
        rep["seconds"] = round(time.time() - t0, 2)
        reports.append(rep)
        print("{} k{} {:.1f}s {}".format(game, k, rep["seconds"],
                                          {c: rep.get(c) for c in augment.COMPONENTS}),
              file=sys.stderr, flush=True)
    if reports:
        rpath = os.path.join(out, "reports", game + ".json")
        os.makedirs(os.path.dirname(rpath), exist_ok=True)
        old = []
        if os.path.isfile(rpath):
            with open(rpath) as h:
                old = [r for r in json.load(h) if r["k"] not in {x["k"] for x in reports}]
        with open(rpath, "w") as h:
            json.dump(sorted(old + reports, key=lambda r: r["k"]), h, indent=1, sort_keys=True,
                      default=float)


def md5_of(path):
    d = hashlib.md5()
    with open(path, "rb") as h:
        for block in iter(lambda: h.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


def write_manifest(out, games, k_views):
    man = {}
    for g in games:
        for k in range(k_views):
            p = feat_path(out, g, k)
            if not os.path.isfile(p):
                raise SystemExit("missing " + p)
            man[os.path.relpath(p, out)] = md5_of(p)
    rec = {"k": k_views, "games": list(games), "files": man,
           "ranges_module": md5_of(augment.__file__.replace(".pyc", ".py"))}
    with open(os.path.join(out, "manifest.json"), "w") as h:
        json.dump(rec, h, indent=1, sort_keys=True)
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--labels-cache", required=True, help="build.py cache (labels/, read only)")
    p.add_argument("--engine-cache", required=True, help="engine.data npz cache, raw view")
    p.add_argument("--games", nargs="+", default=list(folds.ALL))
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--force", action="store_true")
    p.add_argument("--manifest", action="store_true", help="only write manifest.json")
    args = p.parse_args(argv)
    games = check_games(list(args.games))
    augment.check_seeds(games, args.k)
    if args.manifest:
        rec = write_manifest(args.out, games, args.k)
        print("manifest: {} files".format(len(rec["files"])))
        return 0
    for g in games:
        t0 = time.time()
        build_game(g, args.out, args.labels_cache, args.engine_cache, args.k, args.force)
        print("{} done {:.1f}s".format(g, time.time() - t0), file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
