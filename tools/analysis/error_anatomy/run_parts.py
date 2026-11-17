"""Run the default engine on the eight OPEN external references and keep its intermediate parts.

    python run_parts.py [--views matched raw] [--jobs 4]

One child process per (view, match). The child sets the npz cache of engine.data to
cache/<view>/engine_data (this folder), MTAH_DATA_DIR comes from external_sweep.data_root, and it
wraps engine.decoder.decode in memory (as tools/analysis/diagnostics_8nov.py does) to keep the
Viterbi path before min_durations. Nothing in the repository is written.

Outputs, runs/<view>/<game>.npz and runs/<view>/<game>_tables.pkl (events, onsets, restarts,
restart lists, owner events, kick teams). Refuses BAR-ATH and the sealed DFL matches.
"""
import argparse
import concurrent.futures
import os
import pickle
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("ERROR_ANATOMY_WORK", os.path.join("/tmp", "error_anatomy"))  # outputs, never the repo
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (REPO, os.path.join(REPO, "tools"), os.path.join(REPO, "tools", "analysis"),
          os.path.join(REPO, "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)

OPEN_GAMES = ("METRICA-S1", "METRICA-S2", "METRICA-S3",
              "DFL-J03WMX", "DFL-J03WN1", "DFL-J03WOH", "DFL-J03WOY", "DFL-J03WPY")
FORBIDDEN = ("BAR-ATH", "DFL-J03WQQ", "DFL-J03WR9")
H, A, D, X = 0, 1, 2, 3


def check(game):
    if game in FORBIDDEN or game not in OPEN_GAMES:
        raise SystemExit("refused: {}".format(game))


def child(game, view, out_prefix, data_cache):
    check(game)
    from engine import data
    data.DEFAULT_CACHE_DIR = data_cache
    from engine import decoder, possession
    from engine.config import Config

    cfg = Config.default()
    seen = []
    original = decoder.decode

    def traced(log_e, costs, cfg_, forced=None):
        vit = decoder.viterbi(log_e, costs, forced)
        final = decoder.min_durations(vit, log_e, costs, cfg_, forced)
        seen.append(vit)
        return final

    decoder.decode = traced
    try:
        inputs = possession.prepare(cfg, game)
        labels, parts = possession.run(cfg, return_parts=True, inputs=inputs)
    finally:
        decoder.decode = original
    n = int(inputs["n_frames"])
    codes = np.full(n, X, dtype=np.int8)
    for idx, letter in enumerate("HADX"):
        codes[labels == letter] = idx
    # NB: run() may decode twice (knee floors); the last len(blocks) traced paths are the final.
    blocks = inputs["blocks"]
    vit_full = np.full(n, -1, dtype=np.int8)
    for (half, first, last), vit in zip(blocks, seen[-len(blocks):]):
        vit_full[first:last + 1] = vit
    ball, feats = inputs["ball"], inputs["feats"]
    np.savez_compressed(
        out_prefix + ".npz", labels=codes, vit=vit_full,
        owner=np.asarray(parts["owner"], dtype=np.int8),
        stoppage=np.asarray(parts["stoppage"], dtype=bool),
        p_dead=np.asarray(parts["p_dead"], dtype=np.float32),
        e_dead=np.asarray(parts["e_dead"], dtype=np.float32),
        near_lo=np.asarray(parts["near_lo"], dtype=np.float32),
        unknown_lo=np.asarray(parts["unknown_lo"], dtype=np.float32),
        tag_lo=np.asarray(parts["tag_lo"], dtype=np.float32),
        log_e=np.asarray(parts["log_e"], dtype=np.float32),
        visible=np.asarray(ball["visible"], dtype=bool),
        teleport=np.asarray(ball["teleport"], dtype=bool),
        speed=np.asarray(ball["speed"], dtype=np.float32),
        vx=np.asarray(ball["vx"], dtype=np.float32), vy=np.asarray(ball["vy"], dtype=np.float32),
        run_id=np.asarray(ball["run_id"], dtype=np.int64),
        dmin=np.asarray(feats["dmin"], dtype=np.float32),
        bx=np.asarray(ball["x"], dtype=np.float32), by=np.asarray(ball["y"], dtype=np.float32),
        dH=np.asarray(feats["dH"], dtype=np.float32), dA=np.asarray(feats["dA"], dtype=np.float32),
        nearest_team=np.asarray(feats["nearest_team"], dtype=np.int8),
        ghost=np.asarray(inputs["ghost"], dtype=bool),
        blocks=np.asarray([(h, f, l) for h, f, l in blocks], dtype=np.int64),
        fps=np.float64(inputs["fps"]))
    tables = {"events": parts["events"], "onsets": parts["onsets"], "restarts": parts["restarts"],
              "restart_list": parts["restart_list"], "restart_events": parts["restart_events"],
              "owner_events": parts["owner_events"], "kick_teams": parts["kick_teams"],
              "kick_rules": parts["kick_rules"], "goal_kickoffs": parts["goal_kickoffs"],
              "kickoffs": inputs["kickoffs"], "knee": parts.get("knee"),
              "config": cfg.as_dict(), "config_hash": cfg.config_hash()}
    with open(out_prefix + "_tables.pkl", "wb") as handle:
        pickle.dump(tables, handle, protocol=4)
    return 0


def run_child(job):
    game, view = job["game"], job["view"]
    check(game)
    prefix = os.path.join(WORK, "runs", view, game)
    if os.path.isfile(prefix + ".npz") and os.path.isfile(prefix + "_tables.pkl"):
        return prefix
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    cache = os.path.join(WORK, "cache", view, "engine_data")
    os.makedirs(cache, exist_ok=True)
    env = dict(os.environ, MTAH_DATA_DIR=job["root"])
    cmd = [sys.executable, os.path.abspath(__file__), "--child", game, view, prefix, cache]
    with open(prefix + ".log", "w") as log:
        code = subprocess.call(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    if code != 0:
        raise RuntimeError("child failed {} {}".format(game, view))
    return prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", nargs=4)
    parser.add_argument("--views", nargs="+", default=["matched", "raw"])
    parser.add_argument("--games", nargs="+", default=list(OPEN_GAMES))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.child:
        game, view, prefix, cache = args.child
        return child(game, view, prefix, cache)
    import external_sweep as es
    refs_root = os.path.expanduser(es.DEFAULT_REFS_ROOT)
    sample_root = os.path.expanduser(es.DEFAULT_SAMPLE_ROOT)
    mtah_root = os.path.expanduser(es.DEFAULT_MTAH_ROOT)
    jobs = []
    for view in args.views:
        for game in args.games:
            check(game)
            jobs.append({"game": game, "view": view,
                         "root": es.data_root(game, view, refs_root, sample_root, mtah_root)})
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for prefix in pool.map(run_child, jobs):
            print("done", prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
