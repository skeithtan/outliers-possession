"""Label-free report on the competition match for two or more systems. Not part of the submission.

    python -m tools.learned_spike.bar_report --pred c22=C22.csv --pred M=M.csv [--pred H=H.csv]
        [--restart-ref c22] [--ext NAME=GAME:PRED.csv ...] [--seed-probs NAME=F.npz ...]
        [--copy-probs NAME=F.npz ...] [--feature-cache DIR] [--out REPORT.json]

The tool reads prediction CSVs of the competition match (frame,label, one per system), the
ball of the tactical ATD and, if given, predictions of external matches, probability dumps and
a feature cache. It reads no label, never reads evaluation/gt, never uses the public window and
never scores. It prints and writes only aggregates. Every number counts graded frames of
games.py. These are the parts of the report:

- Per system: possessions, D share, D segments, graded X, mean possession length, the
  quantiles of the possession and D segment lengths, and the share of the missing-ball graded
  frames that the system labels D.
- Per pair of systems: the 4 x 4 agreement matrix (shares of graded frames), the agreement, the
  team agreement where both say alive, and the anatomy of the disagreement: its kind (team,
  alive or dead, X), its rate with the ball visible and missing, and its rate by the distance
  to the nearest restart of the reference system (--restart-ref, default the first --pred).
  A restart is a change from D to H or A in the reference system. The pair also gets a
  histogram of the lengths of the disagreement runs.
- Envelope: the possession count per 90 graded minutes, the D share and the length quantiles
  of each system on the competition match against the range of the same system on the
  external matches (--ext NAME=GAME:PATH, NAME the --pred name of the system).
- Probability groups (--seed-probs, --copy-probs; NAME=npz, repeated, each npz with `p5`, the
  5 fps H A D probabilities of one member, see bar_probs.py): the share of graded frames where
  the members' arg max differs, and the mean pairwise disagreement. Seeds and symmetric copies
  are two separate groups.
- Features (--feature-cache DIR with DIR/feats/raw/<game>.npz): the median of each feature on
  the graded bins of the competition match against the 1 to 99 % band of the graded bins of
  the training matches in the cache, and the share of the competition bins outside the band
  (2 % if the match looks like the training matches).

Python 3.9, numpy and pandas only.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import games  # noqa: E402
import possession_metrics as pm  # noqa: E402
import score_game as sg  # noqa: E402
from engine import data  # noqa: E402

FORBIDDEN_EXT = {games.GAME, "DFL-J03WQQ", "DFL-J03WR9"}
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
RESTART_BUCKETS_S = (2.0, 5.0, 10.0)        # <2 s, 2-5 s, 5-10 s, >10 s from a restart
RUN_BUCKETS_S = (0.2, 1.0, 2.0, 5.0, 10.0, 30.0)
BIN = 5                                      # features.BIN, 25 fps to 5 fps
TOP_FEATURES = 6                             # features listed by their share of bins outside


def parse_pairs(items):
    """[(name, value)] from NAME=VALUE strings, in the given order."""
    out = []
    for item in items or []:
        if "=" not in item:
            raise SystemExit("expected NAME=VALUE, got {!r}".format(item))
        out.append(tuple(item.split("=", 1)))
    return out


def graded_runs(game):
    """[(first, stop)] of the graded blocks, stop exclusive."""
    return [(first, last + 1) for _, first, last in data.graded_blocks(game)]


def graded_mask(runs, n):
    mask = np.zeros(n, dtype=bool)
    for a, b in runs:
        mask[a:b] = True
    return mask


def qdict(values, fps=pm.FPS):
    """{"p10": seconds, ...} of lengths in frames; None values when empty."""
    values = np.asarray(values, dtype=np.float64) / fps
    if values.size == 0:
        return {"p{:g}".format(100 * q): None for q in QUANTILES}
    return {"p{:g}".format(100 * q): round(float(v), 2)
            for q, v in zip(QUANTILES, np.quantile(values, QUANTILES))}


def system_stats(codes, runs, ball_missing, fps=pm.FPS):
    """Label-free numbers of one system on one match."""
    n = len(codes)
    graded = graded_mask(runs, n)
    stats = sg.label_stats(codes, runs, fps)
    filled = pm.fill_abstentions(np.asarray(codes, dtype=np.int8), runs)
    spells = pm.possession_spells(filled, runs)
    dsegs = [b - a for a, b, code in pm.segments(filled, runs) if code == pm.D]
    minutes = graded.sum() / float(fps) / 60.0
    missing = graded & ball_missing if ball_missing is not None else None
    return {
        "possessions": int(stats["possessions"]),
        "possessions_per_90min": round(stats["possessions"] / minutes * 90.0, 1),
        "d_share": round(stats["d_share"], 4),
        "d_segments": int(stats["d_segments"]),
        "graded_x": int(np.sum(np.asarray(codes)[graded] == pm.X)),
        "mean_possession_s": round(stats["mean_duration_s"], 2),
        "possession_s": qdict([b - a for a, b, _ in spells], fps),
        "d_segment_s": qdict(dsegs, fps),
        "d_share_ball_missing": round(float(np.mean(filled[missing] == pm.D)), 4)
        if missing is not None and missing.any() else None,
        "d_share_ball_visible": round(float(np.mean(filled[graded & ~ball_missing] == pm.D)), 4)
        if ball_missing is not None else None,
    }


def restart_distance(ref_codes, runs, fps=pm.FPS):
    """float (N,): seconds from each frame to the nearest D -> alive change of `ref_codes`."""
    filled = pm.fill_abstentions(np.asarray(ref_codes, dtype=np.int8), runs)
    starts = []
    for a, b in runs:
        seg = filled[a:b]
        change = np.flatnonzero((seg[:-1] == pm.D) & np.isin(seg[1:], (pm.H, pm.A))) + 1
        starts.extend((change + a).tolist())
    n = len(ref_codes)
    if not starts:
        return np.full(n, np.inf)
    starts = np.asarray(sorted(starts))
    frames = np.arange(n)
    pos = np.searchsorted(starts, frames)
    left = np.where(pos > 0, frames - starts[np.maximum(pos - 1, 0)], np.inf)
    right = np.where(pos < starts.size, starts[np.minimum(pos, starts.size - 1)] - frames, np.inf)
    return np.minimum(left, right) / float(fps)


def bucket_names(edges, unit="s"):
    names = ["<{:g}{}".format(edges[0], unit)]
    names += ["{:g}-{:g}{}".format(lo, hi, unit) for lo, hi in zip(edges[:-1], edges[1:])]
    names.append(">{:g}{}".format(edges[-1], unit))
    return names


def pair_report(a, b, runs, ball_missing, restart_s, fps=pm.FPS):
    """Agreement matrix, disagreement anatomy and disagreement run lengths of two systems."""
    n = len(a)
    graded = graded_mask(runs, n)
    ga, gb = np.asarray(a)[graded], np.asarray(b)[graded]
    total = float(graded.sum())
    matrix = [[round(float(np.sum((ga == i) & (gb == j))) / total, 4) for j in range(4)]
              for i in range(4)]
    differ = np.asarray(a) != np.asarray(b)
    alive_a, alive_b = np.isin(a, (pm.H, pm.A)), np.isin(b, (pm.H, pm.A))
    both_alive = graded & alive_a & alive_b
    team = graded & differ & alive_a & alive_b
    dead = graded & differ & ((alive_a & (np.asarray(b) == pm.D)) |
                              (alive_b & (np.asarray(a) == pm.D)))
    xdiff = graded & differ & ((np.asarray(a) == pm.X) | (np.asarray(b) == pm.X))
    out = {
        "matrix_rows_first_cols_second_HADX": matrix,
        "agreement": round(1.0 - float(np.sum(graded & differ)) / total, 4),
        "team_agreement_both_alive": round(1.0 - float(team.sum()) / both_alive.sum(), 4)
        if both_alive.any() else None,
        "disagreement_kind_share_of_graded": {
            "team": round(team.sum() / total, 4), "alive_dead": round(dead.sum() / total, 4),
            "x": round(xdiff.sum() / total, 4)},
    }
    if ball_missing is not None:
        out["disagreement_rate_by_ball"] = {
            "visible": round(float(differ[graded & ~ball_missing].mean()), 4),
            "missing": round(float(differ[graded & ball_missing].mean()), 4)
            if (graded & ball_missing).any() else None,
            "missing_share_of_disagreement": round(
                float((graded & differ & ball_missing).sum()) / max(1, (graded & differ).sum()), 4)}
    if restart_s is not None:
        edges = (0.0,) + RESTART_BUCKETS_S + (np.inf,)
        names = bucket_names(RESTART_BUCKETS_S)
        rates = {}
        for name, lo, hi in zip(names, edges[:-1], edges[1:]):
            sel = graded & (restart_s >= lo) & (restart_s < hi)
            rates[name] = {"frames_share": round(sel.sum() / total, 4),
                           "disagreement_rate": round(float(differ[sel].mean()), 4)
                           if sel.any() else None}
        out["disagreement_by_restart_distance"] = rates
    lengths = []
    for lo, hi in runs:
        starts, stops = _runs(differ[lo:hi])
        lengths.extend((stops - starts).tolist())
    lengths = np.asarray(lengths, dtype=np.float64) / fps
    edges = (0.0,) + RUN_BUCKETS_S + (np.inf,)
    hist = {}
    for name, lo, hi in zip(bucket_names(RUN_BUCKETS_S), edges[:-1], edges[1:]):
        sel = (lengths >= lo) & (lengths < hi)
        hist[name] = {"runs": int(sel.sum()),
                      "frames_share_of_disagreement": round(
                          float(lengths[sel].sum()) / max(lengths.sum(), 1e-9), 4)}
    out["disagreement_runs"] = {"count": int(lengths.size), "hist": hist}
    return out


def _runs(mask):
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[0::2], edges[1::2]


ENVELOPE_KEYS = ("possessions_per_90min", "d_share", "mean_possession_s")
ENVELOPE_QUANTILE_KEYS = ("possession_s", "d_segment_s")


def envelope(bar, externals):
    """{key: {"bar", "ext_min", "ext_max", "inside"}} for one system."""
    out = {}

    def one(key, value, values):
        values = [v for v in values if v is not None]
        if value is None or not values:
            return
        lo, hi = min(values), max(values)
        out[key] = {"bar": value, "ext_min": lo, "ext_max": hi, "inside": bool(lo <= value <= hi)}
    for key in ENVELOPE_KEYS:
        one(key, bar[key], [e[key] for e in externals])
    for key in ENVELOPE_QUANTILE_KEYS:
        for q in bar[key]:
            one("{}.{}".format(key, q), bar[key][q], [e[key][q] for e in externals])
    return out


def upsample(p5, n_frames, bin_size=BIN):
    """Linear interpolation of bin probabilities onto every frame (model.upsample, numpy)."""
    centres = np.arange(p5.shape[0]) * bin_size + bin_size // 2
    frames = np.arange(n_frames)
    return np.stack([np.interp(frames, centres, p5[:, k]) for k in range(p5.shape[1])], axis=1)


def prob_group(paths, graded):
    """Arg-max disagreement of a group of probability dumps on graded frames."""
    tops = []
    for path in paths:
        with np.load(path) as npz:
            if "p25" in npz.files:       # round 6: frame probabilities (evaluate.dump_probs pf)
                frames = np.asarray(npz["p25"], dtype=np.float64)
            else:
                frames = upsample(np.asarray(npz["p5"], dtype=np.float64), graded.size)
            tops.append(frames[graded].argmax(axis=1))
    tops = np.vstack(tops)
    any_diff = (tops != tops[:1]).any(axis=0)
    pairs = [float(np.mean(tops[i] != tops[j]))
             for i, j in itertools.combinations(range(len(tops)), 2)]
    return {"members": len(paths), "share_any_disagree": round(float(any_diff.mean()), 4),
            "mean_pairwise_disagree": round(float(np.mean(pairs)), 4) if pairs else None}


def graded_bins(game, n_bins):
    """bool (n_bins,): the 5 fps bins whose centre frame is graded."""
    centres = np.minimum(np.arange(n_bins) * BIN + BIN // 2, data.frame_count(game) - 1)
    return graded_mask(graded_runs(game), data.frame_count(game))[centres]


def feature_band(cache, bar_feats, bar_game, train_games=None):
    """Median of each feature on the competition match against the training 1 to 99 % band.

    With train_games (LEARNED_STAGE_PROTOCOL_R2.md Section 2: the 28 external matches and the
    43 PFF training matches) the band pools exactly those matches, and a missing file raises
    SystemExit. Without it the band pools every match of the cache (round 1).
    """
    from tools.learned_spike import build, features
    version = build.folder_version(cache)
    if bar_feats.shape[1] != features.n_features(version):
        raise SystemExit("feature band: {} holds {}, the match features have {} columns".format(
            cache, version, bar_feats.shape[1]))
    lo, hi, used = training_band(cache, train_games)
    bins = bar_feats[graded_bins(bar_game, bar_feats.shape[0])]
    bar = np.median(bins, axis=0)
    share_out = ((bins < lo) | (bins > hi)).mean(axis=0)
    rows = {}
    for k, (name, _) in enumerate(features.feature_spec(version)):
        rows[name] = {"bar_median": round(float(bar[k]), 4), "train_p1": round(float(lo[k]), 4),
                      "train_p99": round(float(hi[k]), 4),
                      "inside": bool(lo[k] <= bar[k] <= hi[k]),
                      "bar_share_outside_band": round(float(share_out[k]), 4)}
    top = sorted(rows, key=lambda n: (-rows[n]["bar_share_outside_band"], n))[:TOP_FEATURES]
    return {"train_games": used, "feature_version": version,
            "outside": sorted(n for n, r in rows.items() if not r["inside"]),
            "most_bins_outside": top, "features": rows}


def training_band(cache, train_games=None):
    """(p1, p99, games): the 1 and 99 % quantiles of each feature over the graded bins of the
    training matches in DIR/feats/raw (see feature_band())."""
    from tools.learned_spike import folds
    root = os.path.join(cache, "feats", "raw")
    pooled = []
    used = []
    if train_games is None:
        names = sorted(os.listdir(root))
    else:
        folds.check_train_all(folds.assert_not_sealed(train_games))
        names = [g + ".npz" for g in train_games]
        missing = [n for n in names if not os.path.isfile(os.path.join(root, n))]
        if missing:
            raise SystemExit("feature band: {} missing in {}".format(" ".join(missing), root))
    for name in names:
        game = name[:-len(".npz")]
        if not name.endswith(".npz") or game in FORBIDDEN_EXT or game not in games.GAMES:
            continue
        with np.load(os.path.join(root, name)) as npz:
            f = np.asarray(npz["F"], dtype=np.float64)
        pooled.append(f[graded_bins(game, f.shape[0])])
        used.append(game)
    pooled = np.vstack(pooled)
    lo = np.quantile(pooled, 0.01, axis=0)
    hi = np.quantile(pooled, 0.99, axis=0)
    return lo, hi, used


def bar_features(game, load=None, version=None):
    """float64 (bins, NF): the 5 fps features of the competition match for the 8c band.

    `version` is the feature version (None: "stage50-v2", as before round 5); build_report()
    passes the version of the feature cache, which must be the model's.

    They come from the ATD that the engine reads (engine.possession.engine_tactical(), with the
    C9 goalkeeper guard), so the band measures the features the network sees (R2 Section 9, D1).
    `load` (game -> tactical dict) replaces engine_tactical(): the round 3 rehearsal reads a
    development match in place of the competition match (r3_eval.py --as-match).
    """
    from engine.possession import engine_tactical
    from tools.learned_spike import features
    tac = load(game) if load is not None else engine_tactical(None, game)
    version = version or features.FEATURE_VERSION
    return features.downsample(features.frame_features(tac, game, version=version)).astype(
        np.float64)


def build_report(args):
    # args.game and args.load_tactical are set only by the round 3 rehearsal (r3_eval.py
    # --as-match), which reads a development match in place of the competition match.
    game = getattr(args, "game", None) or games.GAME
    load = getattr(args, "load_tactical", None)
    n = data.frame_count(game)
    runs = graded_runs(game)
    graded = graded_mask(runs, n)
    preds = parse_pairs(args.pred)
    if len(preds) < 2:
        raise SystemExit("give at least two --pred NAME=PATH")
    codes = {name: sg.read_codes(path, n) for name, path in preds}
    tac = load(game) if load is not None else data.load_tactical(game)
    ball_missing = ~np.isfinite(np.asarray(tac["bx"], dtype=np.float64))
    report = {"game": game, "graded_frames": int(graded.sum()),
              "ball_missing_share_graded": round(float(ball_missing[graded].mean()), 4),
              "systems": {}, "pairs": {}}
    for name, _ in preds:
        report["systems"][name] = system_stats(codes[name], runs, ball_missing)
    ref = args.restart_ref or preds[0][0]
    restart_s = restart_distance(codes[ref], runs)
    report["restart_ref"] = ref
    for (na, _), (nb, _) in itertools.combinations(preds, 2):
        report["pairs"]["{} vs {}".format(na, nb)] = pair_report(
            codes[na], codes[nb], runs, ball_missing, restart_s)

    ext = {}
    for name, value in parse_pairs(args.ext):
        ext_game, path = value.split(":", 1)
        if ext_game in FORBIDDEN_EXT:
            raise SystemExit("{} cannot be an external match".format(ext_game))
        ext_codes = sg.read_codes(path, data.frame_count(ext_game))
        ext.setdefault(name, []).append(
            dict(system_stats(ext_codes, graded_runs(ext_game), None), game=ext_game))
    if ext:
        report["envelope"] = {name: envelope(report["systems"][name], ext[name])
                              for name in sorted(ext) if name in report["systems"]}
        report["external_stats"] = ext

    for flag, key in ((args.seed_probs, "seed_disagreement"),
                      (args.copy_probs, "copy_disagreement")):
        pairs = parse_pairs(flag)
        if pairs:
            report[key] = dict(prob_group([p for _, p in pairs], graded),
                               names=[k for k, _ in pairs])

    if args.feature_cache:
        from tools.learned_spike import build
        bar_feats = bar_features(game, load, build.folder_version(args.feature_cache))
        report["features"] = feature_band(args.feature_cache, bar_feats, game,
                                          getattr(args, "feature_games", None))
    return report


def print_report(report, out=sys.stdout):
    w = out.write
    w("{}: {} graded frames, ball missing on {:.1%}\n".format(
        report["game"], report["graded_frames"], report["ball_missing_share_graded"]))
    w("\nsystem   poss  /90min  D share  D segs  graded X  mean poss s  D|ball missing  D|ball visible\n")
    def na(v):
        # None when the match has no graded frame with the ball missing (e.g. a DFL match used
        # as the rehearsal match: its raw ball is never missing).
        return "-" if v is None else v
    for name, s in report["systems"].items():
        w("{:<8} {:>4}  {:>6}  {:>7.3f}  {:>6}  {:>8}  {:>11}  {:>14}  {:>14}\n".format(
            name, s["possessions"], s["possessions_per_90min"], s["d_share"], s["d_segments"],
            s["graded_x"], s["mean_possession_s"], na(s["d_share_ball_missing"]),
            na(s["d_share_ball_visible"])))
    w("\npossession length s (p10 p25 p50 p75 p90) / D segment length s\n")
    for name, s in report["systems"].items():
        w("{:<8} {}  /  {}\n".format(name, " ".join(str(v) for v in s["possession_s"].values()),
                                    " ".join(str(v) for v in s["d_segment_s"].values())))
    w("\nrestart reference: {}\n".format(report["restart_ref"]))
    for pair, p in report["pairs"].items():
        kinds = p["disagreement_kind_share_of_graded"]
        w("\n{}: agreement {:.3f}, team agreement where both alive {}\n".format(
            pair, p["agreement"], p["team_agreement_both_alive"]))
        w("  disagreement kinds (share of graded): team {} alive/dead {} X {}\n".format(
            kinds["team"], kinds["alive_dead"], kinds["x"]))
        w("  matrix (rows {} H A D X, cols second):\n".format(pair.split(" vs ")[0]))
        for row in p["matrix_rows_first_cols_second_HADX"]:
            w("    " + " ".join("{:.4f}".format(v) for v in row) + "\n")
        if "disagreement_rate_by_ball" in p:
            b = p["disagreement_rate_by_ball"]
            w("  disagreement rate: ball visible {} / missing {}; missing share of disagreement {}\n"
              .format(b["visible"], b["missing"], b["missing_share_of_disagreement"]))
        if "disagreement_by_restart_distance" in p:
            w("  by distance to a restart: " + ", ".join(
                "{} {} (frames {})".format(k, v["disagreement_rate"], v["frames_share"])
                for k, v in p["disagreement_by_restart_distance"].items()) + "\n")
        h = p["disagreement_runs"]
        w("  disagreement runs: {}; ".format(h["count"]) + ", ".join(
            "{} {} ({:.0%})".format(k, v["runs"], v["frames_share_of_disagreement"])
            for k, v in h["hist"].items()) + "\n")
    for name, env in report.get("envelope", {}).items():
        outside = [k for k, v in env.items() if not v["inside"]]
        w("\nenvelope {} ({} externals): outside {}\n".format(
            name, len(report["external_stats"][name]), outside or "none"))
        for k, v in env.items():
            w("  {:<22} BAR {:>8}  ext [{}, {}]{}\n".format(
                k, v["bar"], v["ext_min"], v["ext_max"], "" if v["inside"] else "  OUTSIDE"))
    for key in ("seed_disagreement", "copy_disagreement"):
        if key in report:
            r = report[key]
            w("\n{}: {} members, any disagree {}, mean pairwise {}\n".format(
                key, r["members"], r["share_any_disagree"], r["mean_pairwise_disagree"]))
    if "features" in report:
        f = report["features"]
        w("\nfeatures: {} training matches; BAR median outside the 1-99 % band: {}\n".format(
            len(f["train_games"]), f["outside"] or "none"))
        for name in f["outside"]:
            r = f["features"][name]
            w("  {:<14} BAR {:>8}  band [{}, {}]\n".format(
                name, r["bar_median"], r["train_p1"], r["train_p99"]))
        w("  share of BAR graded bins outside the band (2 % expected): " + ", ".join(
            "{} {}".format(n, f["features"][n]["bar_share_outside_band"])
            for n in f["most_bins_outside"]) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pred", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--restart-ref")
    parser.add_argument("--ext", action="append", help="NAME=GAME:PATH, NAME a --pred name")
    parser.add_argument("--seed-probs", action="append", help="NAME=npz with p5")
    parser.add_argument("--copy-probs", action="append", help="NAME=npz with p5")
    parser.add_argument("--feature-cache")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    report = build_report(args)
    print_report(report)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
