"""Measure DELTA_KNEE on the external references (COMPONENT_KNEE.md Section 3). Not part of the submission.

    python tools/analysis/knee_lag.py --out DIR [--views raw matched] [--games G ...]
        [--cache DIR] [--refs-root DIR] [--sample-root DIR] [--mtah-root DIR]

For each match and view the tool runs possession.run() with ONSET_KNEE true, DELTA_KNEE 0,
KNEE_GAP_CLAMP false and KNEE_MIN_PRE_SPELL_S 0 (variant 1, the Section 3 measurement), and
reads the knee rows of its parts. Each eligible gap-plus-stop dwell
entry e with a knee k is paired with a reference onset r (the first frame of a reference `D` run
after an `H` or `A` frame): the start of the reference `D` run that holds e if that start is an
onset, else the first reference onset in (e, leave), where leave is the end of the engine
stoppage that starts at the entry. The lag is r - k in frames.

DELTA_KNEE is the median of the lags pooled over the three Metrica matches and both views,
rounded with numpy.round. DFL is measured and recorded only. The tool refuses BAR-ATH and the
sealed matches (the checks of external_sweep.py). It writes one JSON file per match and view
and a summary JSON. It prints summary numbers only, never rows.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from tools.analysis import external_sweep as es  # noqa: E402

LIVE = ("H", "A")


def reference_onsets(labels):
    """int64 array of the frames f with labels[f] == 'D' and labels[f - 1] in H, A."""
    labels = np.asarray(labels)
    prev_live = np.isin(labels[:-1], LIVE)
    return np.flatnonzero((labels[1:] == "D") & prev_live).astype(np.int64) + 1


def pair_onset(labels, onsets, entry, leave):
    """The reference onset paired with an entry (module docstring), or None."""
    if labels[entry] == "D":
        start = entry
        while start > 0 and labels[start - 1] == "D":
            start -= 1
        return int(start) if start > 0 and labels[start - 1] in LIVE else None
    later = onsets[(onsets > entry) & (onsets < leave)]
    return int(later[0]) if later.size else None


def stoppage_end(stoppage, start):
    """The first frame after the stoppage run that holds `start`."""
    rest = np.flatnonzero(~np.asarray(stoppage[start:], dtype=bool))
    return int(start + rest[0]) if rest.size else int(len(stoppage))


def _quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0}
    q = np.percentile(values, [10, 25, 50, 75, 90])
    return {"n": int(values.size), "p10": float(q[0]), "p25": float(q[1]),
            "median": float(q[2]), "p75": float(q[3]), "p90": float(q[4])}


def measure(game, labels):
    """{counts and lags} of one match, in the current MTAH_DATA_DIR view."""
    from engine import possession
    from engine.config import Config

    cfg = dataclasses.replace(Config.default(), ONSET_KNEE=True, DELTA_KNEE=0,
                              KNEE_GAP_CLAMP=False, KNEE_MIN_PRE_SPELL_S=0.0)
    _, parts = possession.run(cfg, return_parts=True, game=game)
    stoppage = parts["stoppage"]
    onsets = reference_onsets(labels)
    lags, late, bounds = [], [], {}
    no_knee = unpaired = 0
    for row in parts["knee"]:
        bounds[row["bound"]] = bounds.get(row["bound"], 0) + 1
        if row["knee"] is None:
            no_knee += 1
            continue
        leave = stoppage_end(stoppage, row["new"])
        ref = pair_onset(labels, onsets, row["entry"], leave)
        if ref is None:
            unpaired += 1
            continue
        lags.append(int(ref - row["knee"]))
        late.append(int(row["entry"] - ref))
    return {"game": game, "gap_entries": len(parts["knee"]), "no_knee": no_knee,
            "unpaired": unpaired, "bounds": bounds, "lags": lags,
            "lag_quantiles": _quantiles(lags), "entry_minus_onset": _quantiles(late)}


def child(args):
    from engine import data
    from tools.analysis import restart_team as rt
    import games

    data.DEFAULT_CACHE_DIR = args.data_cache
    labels = rt.read_labels(args.ref, int(games.GAMES[args.game]["n_video_frames"]))
    out = measure(args.game, labels)
    out["view"] = args.view
    with open(args.out, "w") as handle:
        json.dump(out, handle, sort_keys=True)


def run_all(args):
    names = es.check_games(args.games)
    refs_root = os.path.abspath(os.path.expanduser(args.refs_root))
    sample_root = os.path.abspath(os.path.expanduser(args.sample_root))
    mtah_root = os.path.abspath(os.path.expanduser(args.mtah_root))
    cache = os.path.abspath(args.cache)
    os.makedirs(args.out, exist_ok=True)
    records = []
    for view in args.views:
        for game in names:
            ref, _ = es.prepare_reference(game, refs_root, sample_root, cache)
            path = os.path.join(args.out, "{}_{}.json".format(game, view))
            env = dict(os.environ, MTAH_DATA_DIR=es.data_root(game, view, refs_root,
                                                                 sample_root, mtah_root))
            subprocess.run([sys.executable, os.path.abspath(__file__), "--child", "--game",
                            game, "--view", view, "--ref", ref, "--out", path, "--data-cache",
                            os.path.join(es.view_dir(cache, view), "engine_data")],
                           cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
            with open(path) as handle:
                records.append(json.load(handle))
    summary = summarise(records)
    with open(os.path.join(args.out, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=1, sort_keys=True))


def summarise(records):
    """Per match and view counts, per-view and pooled medians, and DELTA_KNEE (Metrica)."""
    out = {"per_match": {}, "pooled": {}}
    for rec in records:
        out["per_match"]["{}|{}".format(rec["game"], rec["view"])] = {
            k: rec[k] for k in ("gap_entries", "no_knee", "unpaired", "bounds",
                                "lag_quantiles", "entry_minus_onset")}
    for provider, prefix in (("metrica", "METRICA-"), ("dfl", "DFL-")):
        views = sorted(set(r["view"] for r in records))
        for view in views + ["all"]:
            lags = [x for r in records if r["game"].startswith(prefix)
                    and (view == "all" or r["view"] == view) for x in r["lags"]]
            out["pooled"]["{}|{}".format(provider, view)] = _quantiles(lags)
    metrica = out["pooled"]["metrica|all"]
    out["DELTA_KNEE"] = int(np.round(metrica["median"])) if metrica["n"] else None
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out")
    parser.add_argument("--views", nargs="+", default=["raw", "matched"])
    parser.add_argument("--games", nargs="+", default=list(es.OPEN_GAMES))
    parser.add_argument("--cache", default=es.DEFAULT_CACHE)
    parser.add_argument("--refs-root", default=es.DEFAULT_REFS_ROOT)
    parser.add_argument("--sample-root", default=es.DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--mtah-root", default=es.DEFAULT_MTAH_ROOT)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--game", help=argparse.SUPPRESS)
    parser.add_argument("--view", help=argparse.SUPPRESS)
    parser.add_argument("--ref", help=argparse.SUPPRESS)
    parser.add_argument("--data-cache", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        child(args)
    else:
        if not args.out:
            parser.error("--out is required")
        run_all(args)


if __name__ == "__main__":
    main()
