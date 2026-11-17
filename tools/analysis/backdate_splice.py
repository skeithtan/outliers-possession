"""K2 feasibility: move each owner change back to where the new team took the ball (a splice).

    python tools/analysis/backdate_splice.py --view matched --out /tmp/k2_matched.json
        [--radius 1.5] [--cap-frames 75] [--min-frames 5] [--config OVERRIDES.json]
        [--games G1 G2 ...] [--jobs N] [--cache DIR] [--draws 1000]

Not part of the submission. This is a splice of the engine output, not an engine run.

The engine changes the owner at an owner event of the new team: a kick, a control run, a dribble,
a receive. The error anatomy of RESEARCH-4 Section 2 found that the late changes (more than 25
frames after the reference) are mostly fired by kicks, control runs and dribbles: the team got the
ball earlier and the first event that the engine sees comes later. The splice tests the simplest
repair. For each direct change H <-> A of the engine labels at frame t1 to team U, it walks back
from t1 - 1 while all of these hold on the frame:

- the ball is visible (cleaned ball),
- the nearest player of U is within `--radius` of the ball (default: the engine's R_TOUCH),
- U is the nearer team (or the other team has no visible player),
- the engine label is not D,

and stops at the previous change or `--cap-frames` before t1. If the stretch is at least
`--min-frames` long, the frames from its first frame t0 to t1 take the team U. Nothing else
changes.

For each match the tool scores the engine labels and the spliced labels against the reference
(spellF1, `tools/score_game.py` paired bootstrap of the reference possessions, seed 0) and prints
the per-match difference, its 5 to 95 % interval, the provider means and the number of moved
changes with the median shift. It never runs on the competition match or on a sealed match.
Deterministic. Python 3.9.
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, "tools")
ANALYSIS = os.path.join(TOOLS, "analysis")
for _path in (ROOT, TOOLS, ANALYSIS, os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

H, A, D = 0, 1, 2


def backdate(labels, d_home, d_away, visible, radius, cap, min_frames):
    """(spliced codes, [(t1, t0), ...]): every direct H <-> A change moved back (module docstring).

    `labels` are int codes (0 H, 1 A, 2 D, 3 X). `d_home`, `d_away` are the nearest distances of
    each team to the ball (NaN or inf without a ball or a player). `visible` is the cleaned ball.
    """
    import numpy as np

    labels = np.asarray(labels, dtype=np.int8)
    out = labels.copy()
    alive = (labels == H) | (labels == A)
    changes = np.flatnonzero(alive[1:] & alive[:-1] & (labels[1:] != labels[:-1])) + 1
    d_home = np.asarray(d_home, dtype=np.float64)
    d_away = np.asarray(d_away, dtype=np.float64)
    visible = np.asarray(visible, dtype=bool)
    previous = -1
    moved = []
    for t1 in changes:
        t1 = int(t1)
        team = int(labels[t1])
        own = d_home if team == H else d_away
        other = d_away if team == H else d_home
        low = max(previous + 1, t1 - int(cap))
        k = t1 - 1
        while (k >= low and visible[k] and np.isfinite(own[k]) and own[k] <= radius
               and (not np.isfinite(other[k]) or own[k] <= other[k]) and labels[k] != D):
            k -= 1
        start = k + 1
        if t1 - start >= int(min_frames):
            out[start:t1] = team
            moved.append((t1, start))
        previous = t1
    return out, moved


def child(game, view, data_cache, ref_path, overrides, radius, cap, min_frames, draws):
    """Scores of one match and view, as a dict. Runs inside the subprocess of the match."""
    import numpy as np
    from engine import config, data, possession
    import possession_metrics as pm
    import score_game as sg

    data.DEFAULT_CACHE_DIR = data_cache
    cfg = config.Config.default()
    if overrides:
        import dataclasses
        cfg = dataclasses.replace(cfg, **overrides)
    inputs = possession.prepare(cfg, game)
    labels, _ = possession.run(cfg, return_parts=True, inputs=inputs)
    codes = pm.label_codes(labels)
    feats, ball = inputs["feats"], inputs["ball"]
    reach = float(cfg.R_TOUCH) if radius is None else float(radius)
    spliced, moved = backdate(codes, feats["dH"], feats["dA"], ball["visible"], reach, cap,
                              min_frames)
    ref = pm.label_codes(pm.read_label_csv(ref_path))
    runs = pm.valid_runs(ref)
    base = pm.spell_scores(ref, pm.fill_abstentions(codes, runs), runs)["mean_f1"]
    after = pm.spell_scores(ref, pm.fill_abstentions(spliced, runs), runs)["mean_f1"]
    paired = sg.paired_delta(ref, spliced, codes, n_draws=draws)
    shifts = [t1 - t0 for t1, t0 in moved]
    return {"game": game, "view": view, "radius": reach, "cap_frames": int(cap),
            "min_frames": int(min_frames), "spell_f1": float(base), "spell_f1_spliced": float(after),
            "delta": paired, "moved": len(moved),
            "median_shift": float(np.median(shifts)) if shifts else None}


def run_child(job):
    """Run one match in a subprocess with its MTAH_DATA_DIR. Return the parsed JSON."""
    env = dict(os.environ, MTAH_DATA_DIR=job["data_dir"])
    command = [sys.executable, os.path.abspath(__file__), "--child", "--game", job["game"],
               "--view", job["view"], "--data-cache", job["data_cache"], "--ref", job["ref"],
               "--cap-frames", str(job["cap"]), "--min-frames", str(job["min_frames"]),
               "--draws", str(job["draws"]), "--overrides", json.dumps(job["overrides"])]
    if job["radius"] is not None:
        command += ["--radius", str(job["radius"])]
    done = subprocess.run(command, cwd=ROOT, env=env, check=True, stdout=subprocess.PIPE)
    return json.loads(done.stdout.decode("utf-8").strip().splitlines()[-1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--view", default="matched", choices=("raw", "matched", "sigma95"))
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--cap-frames", type=int, default=75)
    parser.add_argument("--min-frames", type=int, default=5)
    parser.add_argument("--config", default=None, help="JSON overrides of engine/config.py")
    parser.add_argument("--games", nargs="*", default=None)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--out", default=None)
    parser.add_argument("--refs-root", default=None)
    parser.add_argument("--sample-root", default=None)
    parser.add_argument("--mtah-root", default=None)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--game", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--data-cache", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ref", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--overrides", default="{}", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        result = child(args.game, args.view, args.data_cache, args.ref, json.loads(args.overrides),
                       args.radius, args.cap_frames, args.min_frames, args.draws)
        print(json.dumps(result, sort_keys=True))
        return 0

    import numpy as np
    import external_sweep as es
    overrides = {}
    if args.config:
        with open(args.config) as handle:
            overrides = json.load(handle)
    games = es.check_games(args.games or list(es.OPEN_GAMES))
    refs_root = os.path.expanduser(args.refs_root or es.DEFAULT_REFS_ROOT)
    sample_root = os.path.expanduser(args.sample_root or es.DEFAULT_SAMPLE_ROOT)
    mtah_root = os.path.expanduser(args.mtah_root or es.DEFAULT_MTAH_ROOT)
    cache = os.path.abspath(os.path.expanduser(args.cache or es.DEFAULT_CACHE))
    jobs = []
    for game in games:
        ref, _ = es.prepare_reference(game, refs_root, sample_root, cache)
        jobs.append({"game": game, "view": args.view, "ref": ref, "overrides": overrides,
                     "data_dir": es.data_root(game, args.view, refs_root, sample_root, mtah_root),
                     "data_cache": os.path.join(es.view_dir(cache, args.view), "engine_data"),
                     "radius": args.radius, "cap": args.cap_frames, "min_frames": args.min_frames,
                     "draws": args.draws})
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(run_child, jobs))
    report = {"view": args.view, "overrides": overrides, "matches": results}
    for group, prefix in (("metrica", "METRICA-"), ("dfl", "DFL-")):
        deltas = [r["delta"]["delta"] for r in results if r["game"].startswith(prefix)]
        report[group + "_mean_delta"] = float(np.mean(deltas)) if deltas else None
    text = json.dumps(report, sort_keys=True, indent=1)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    for r in results:
        d = r["delta"]
        print("{:10s} {:12s} spellF1 {:.4f} -> {:.4f}  delta {:+.4f} [{:+.4f}, {:+.4f}]  moved {:4d}"
              "  median shift {}".format(args.view, r["game"], r["spell_f1"], r["spell_f1_spliced"],
                                         d["delta"], d["low"], d["high"], r["moved"],
                                         r["median_shift"]))
    print("{} means: Metrica {:+.4f}, DFL {:+.4f}".format(args.view, report["metrica_mean_delta"],
                                                          report["dfl_mean_delta"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
