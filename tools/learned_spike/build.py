"""Build the feature caches of the learned spike. Not part of the submission.

    python -m tools.learned_spike.build --cache DIR --ref-cache DIR --engine-cache DIR
        [--games G ...] [--views raw matched ...] [--feature-version V] [--labels-from DIR]
    python -m tools.learned_spike.build --mark DIR --feature-version V

For each match and view the tool writes DIR/feats/<view>/<game>.npz with the 5 fps feature
matrix `F` (float32, bins x NF). For each match it writes DIR/labels/<game>.npz with the
reference codes at 25 fps: `raw` (the reference as it is) and `ctl2` (control_labels with
50 frames, the 2 s controlled view). Codes are those of evaluation/possession_metrics.

Views: raw, matched and sigma95 come from tools/analysis/external_sweep.data_root. The view
`degraded` is the raw view after tools/learned_spike/degrade.py. A match gets only the views
that exist for it (folds.views()): a SkillCorner match has raw and degraded, no smoothed views.
The tool skips a requested view that does not exist for a match.

The default --games is the round 2 training set (folds.TRAIN_ALL): the 28 external matches
of the learned stage and the 43 PFF training matches of LEARNED_STAGE_PROTOCOL_R2.md Section 1.
A PFF match has the raw view only (no degraded view) and its reference is
~/mtah2027/refs/pff/<id>/PFF-<id>_reference.csv. The tool checks every PFF reference md5
against R2 Section 8 before it builds anything (pff.check_references()). The engine cache
(--engine-cache) must hold the raw npz cache of each match, or the tool parses the bundle and
writes it there. --no-pff-engine-cache keeps the PFF tactical npz out of the engine cache (the
PFF training matches need only their features).

Feature versions (round 5): --feature-version picks the version of features.py
(features.VERSIONS). The default is features.default_version(): $LEARNED_FEATURE_VERSION when
set, else "stage50-v2", the version of every cache before round 5. Each feature file then also
holds `feature_version` (a string) and `columns` (the column names), and DIR/feature_version.json
records the version and the columns of the whole folder (VERSION_MARKER). train.py reads the
marker and refuses a folder of another version, or selects the columns of its version by name
from a folder of a version that holds them all (none does today; the rule is generic). A folder
without a marker is a "stage50-v2" folder. The tool refuses to add files of one version to a
folder marked with another. --labels-from DIR links DIR/labels/<game>.npz instead of reading
the reference (no --ref-cache needed). --mark DIR checks the width of every feature file of a
folder built by another tool (r4_inputs.py, aug_build.py with $LEARNED_FEATURE_VERSION) and
writes the marker.

To add a match, add its games.py entry and teach reference_path() where its reference CSV is.

The tool refuses the competition match, the sealed DFL matches, and the sealed and pilot PFF
matches (folds.assert_not_sealed()). Python 3.9.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import games  # noqa: E402
import possession_metrics as pm  # noqa: E402
from analysis import control_reference as cref  # noqa: E402
from analysis import external_sweep as xs  # noqa: E402
from engine import data  # noqa: E402
from tools.learned_spike import degrade, features, folds, pff  # noqa: E402

OPEN_GAMES = xs.OPEN_GAMES
FORBIDDEN = set(xs.SEALED_GAMES) | {games.GAME}
VIEWS = ("raw", "matched", "sigma95", "degraded")
CONTROL_FRAMES = 50
HOME = os.path.expanduser("~")
REFS_ROOT = os.path.join(HOME, "mtah2027", "refs")
SAMPLE_ROOT = os.path.join(HOME, "mtah2027", "sample-data", "data")
MTAH_ROOT = os.path.join(HOME, "mtah2027")
VERSION_MARKER = "feature_version.json"    # train.VERSION_MARKER


def marker_path(folder):
    return os.path.join(folder, VERSION_MARKER)


def read_marker(folder):
    """The feature version marker dict of a cache folder, or None for an unmarked folder."""
    path = marker_path(folder)
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def folder_version(folder):
    """The feature version of a cache folder: its marker, "stage50-v2" without one."""
    mark = read_marker(folder)
    if mark is None:
        return features.FEATURE_VERSION
    version = features.check_version(mark.get("feature_version"))
    if mark.get("columns") != features.feature_names(version):
        raise SystemExit("{}: the marker's columns are not those of {}".format(folder, version))
    return version


def marker_record(version):
    return {"feature_version": version, "columns": features.feature_names(version),
            "n_features": features.n_features(version)}


def write_marker(folder, version):
    """Write DIR/feature_version.json, or check it. A folder marked with another version, or
    an unmarked folder that already holds feature files, raises SystemExit."""
    rec = marker_record(version)
    old = read_marker(folder)
    if old is not None:
        if old.get("feature_version") != version or old.get("columns") != rec["columns"]:
            raise SystemExit("{} is marked {}, not {}".format(
                folder, old.get("feature_version"), version))
        return rec
    if feature_files(folder):
        raise SystemExit("{} already holds unmarked feature files (stage50-v2); use a new "
                         "folder".format(folder))
    os.makedirs(folder, exist_ok=True)
    with open(marker_path(folder), "w") as handle:
        json.dump(rec, handle, indent=1)
    return rec


def feature_files(folder):
    """Every feature file of a cache folder: feats/<view>/*.npz, feats/aug/*/*.npy and
    feats/ballfree/*/*.npy (links followed)."""
    pats = [os.path.join(folder, "feats", "*", "*.npz"),
            os.path.join(folder, "feats", "aug", "*", "*.npy"),
            os.path.join(folder, "feats", "ballfree", "*", "*.npy")]
    return sorted(p for pat in pats for p in glob.glob(pat))


def mark(folder, version):
    """Check that every feature file of `folder` has the width of `version` (and, if it carries
    one, its version), then write the marker. Return (marker, number of files)."""
    width = features.n_features(version)
    files = feature_files(folder)
    if not files:
        raise SystemExit("{}: no feature files".format(folder))
    bad = []
    for path in files:
        if path.endswith(".npz"):
            with np.load(path) as npz:
                shape = npz["F"].shape
                stored = str(npz["feature_version"]) if "feature_version" in npz.files else None
        else:
            shape, stored = np.load(path, mmap_mode="r").shape, None
        if shape[-1] != width or (stored is not None and stored != version):
            bad.append((path, shape, stored))
    if bad:
        raise SystemExit("{}: {} files are not {} ({} columns), e.g. {}".format(
            folder, len(bad), version, width, bad[:2]))
    old = read_marker(folder)
    if old is not None and old.get("feature_version") != version:
        raise SystemExit("{} is marked {}".format(folder, old.get("feature_version")))
    rec = marker_record(version)
    with open(marker_path(folder), "w") as handle:
        json.dump(rec, handle, indent=1)
    return rec, len(files)


def check(game):
    """Raise SystemExit for the competition match and the sealed DFL matches.

    Raise AssertionError for a sealed or pilot PFF match (R2 Section 1).
    """
    folds.assert_not_sealed([game])
    if game in FORBIDDEN:
        raise SystemExit("{} is forbidden for the spike (competition or sealed match)".format(game))


def available_views(game, views):
    """The views of `views` that exist for `game` (folds.views()), in the order given."""
    check(game)
    if game.startswith("PFF-"):
        have = folds.views(game)
    else:
        have = folds.views(game) if game in folds.ALL else VIEWS
    return [v for v in views if v in have]


def reference_path(game, ref_cache):
    """The external reference CSV of a match (external_sweep.prepare_reference, md5 checked)."""
    check(game)
    if game.startswith("PFF-"):
        return pff.reference(game, REFS_ROOT)[0]
    return xs.prepare_reference(game, REFS_ROOT, SAMPLE_ROOT, ref_cache)[0]


def load_codes(game, ref_cache):
    path = reference_path(game, ref_cache)
    table = pm.read_label_csv(path)
    raw = pm.label_codes(table)
    labels = np.asarray(pm.LABELS, dtype=object)[raw]
    ctl = pm.label_codes(cref.control_labels(labels, CONTROL_FRAMES))
    return raw, ctl


def load_view(game, view, engine_cache):
    """The tactical dict of a match in a view (raw, matched, sigma95)."""
    check(game)
    if game.startswith("PFF-"):
        if view != "raw":
            raise SystemExit("{}: PFF has the raw view only".format(game))
        root = REFS_ROOT
    else:
        root = xs.data_root(game, view, REFS_ROOT, SAMPLE_ROOT, MTAH_ROOT)
    cache_dir = engine_cache if view == "raw" else None
    return data.load_tactical(game, data_dir=root, cache_dir=cache_dir,
                              use_cache=cache_dir is not None)


def build_game(game, views, cache, ref_cache, engine_cache, force=False, version=None,
               labels_from=None):
    check(game)
    version = features.default_version() if version is None else version
    lab_path = os.path.join(cache, "labels", game + ".npz")
    if labels_from is not None:
        src = os.path.join(labels_from, "labels", game + ".npz")
        if not os.path.isfile(src):
            raise SystemExit("{}: no labels in {}".format(game, labels_from))
        if not os.path.lexists(lab_path):
            os.makedirs(os.path.dirname(lab_path), exist_ok=True)
            os.symlink(os.path.abspath(src), lab_path)
    elif force or not os.path.isfile(lab_path):
        raw, ctl = load_codes(game, ref_cache)
        os.makedirs(os.path.dirname(lab_path), exist_ok=True)
        np.savez(lab_path, raw=raw, ctl2=ctl)
    labels = np.load(lab_path)
    raw_tac = None
    for view in available_views(game, views):
        path = os.path.join(cache, "feats", view, game + ".npz")
        if os.path.isfile(path) and not force:
            continue
        t0 = time.time()
        if view in ("raw", "degraded"):
            if raw_tac is None:
                raw_tac = load_view(game, "raw", engine_cache)
            tac = raw_tac if view == "raw" else degrade.degrade(raw_tac, labels["raw"], game)
        else:
            tac = load_view(game, view, engine_cache)
        f25 = features.frame_features(tac, game, version=version)
        if f25.shape[0] != len(labels["raw"]):
            raise SystemExit("{} {}: {} feature frames, {} reference frames".format(
                game, view, f25.shape[0], len(labels["raw"])))
        f5 = features.downsample(f25)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez(tmp, F=f5, n_frames=np.int64(f25.shape[0]), feature_version=np.str_(version),
                 columns=np.array(features.feature_names(version)))
        os.replace(tmp, path)
        print("{} {} bins={} {:.1f}s".format(game, view, f5.shape[0], time.time() - t0),
              file=sys.stderr, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache")
    parser.add_argument("--ref-cache", help="external_sweep cache (Metrica refs)")
    parser.add_argument("--engine-cache", help="engine.data npz cache, raw view")
    parser.add_argument("--feature-version", choices=features.VERSIONS, default=None,
                        help="features.py version (default: $LEARNED_FEATURE_VERSION, else "
                             "stage50-v2)")
    parser.add_argument("--labels-from", default=None,
                        help="link the labels of this build.py folder (no --ref-cache needed)")
    parser.add_argument("--mark", default=None, metavar="DIR",
                        help="only check the widths of DIR and write its version marker")
    parser.add_argument("--games", nargs="+", default=list(folds.TRAIN_ALL))
    parser.add_argument("--views", nargs="+", default=list(VIEWS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-pff-engine-cache", action="store_true",
                        help="parse the PFF bundles without writing their engine npz cache")
    args = parser.parse_args(argv)
    version = args.feature_version or features.default_version()
    if args.mark:
        rec, count = mark(args.mark, version)
        print("{}: {} files, marked {}".format(args.mark, count, rec["feature_version"]))
        return 0
    if not args.cache or not args.engine_cache or not (args.ref_cache or args.labels_from):
        parser.error("--cache, --engine-cache and --ref-cache (or --labels-from) are required")
    write_marker(args.cache, version)
    folds.assert_not_sealed(args.games)
    for game in args.games:
        check(game)
    if any(g.startswith("PFF-") for g in args.games):
        pff.check_references(REFS_ROOT)
    for game in args.games:
        t0 = time.time()
        engine_cache = args.engine_cache
        if game.startswith("PFF-") and args.no_pff_engine_cache:
            engine_cache = None
        build_game(game, args.views, args.cache, args.ref_cache, engine_cache, args.force,
                   version, args.labels_from)
        print("{} done {:.1f}s".format(game, time.time() - t0), file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
