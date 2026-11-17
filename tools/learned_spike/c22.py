"""The c22 labels of external matches, from engine.run with the default config. Not part of the submission.

    python -m tools.learned_spike.c22 --ext DIR --games G ... [--jobs 4]

The hybrid decoder of the learned stage (LEARNED_STAGE_PROTOCOL.md, Section 1) keeps the alive
and dead frames of c22. Cost selection on a validation match needs those frames. This tool
runs `engine.run` with the default config (c22) on the raw view of each match and caches the
CSV.

The runs use tools/analysis/external_sweep.py: the run key covers the match, the full config,
the data root and the source of engine/ and games.py. So a change of the engine gives a new
key and a new run. The CSV is <ext>/views/raw/runs/<game>/<key>.csv, and the engine npz cache
is <ext>/views/raw/engine_data. path_of() gives the path without running anything, so the
torch venv can find a run that the engine venv wrote.

Run the tool with the engine venv (~/.venvs/barca). It refuses every match outside the 28 of
the protocol (folds.check). Python 3.9.
"""

import argparse
import concurrent.futures
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from analysis import external_sweep as xs  # noqa: E402
from tools.learned_spike import folds  # noqa: E402

OVERRIDES = {}     # the default config: c22


def _roots():
    refs = os.path.abspath(os.path.expanduser(xs.DEFAULT_REFS_ROOT))
    sample = os.path.abspath(os.path.expanduser(xs.DEFAULT_SAMPLE_ROOT))
    mtah = os.path.abspath(os.path.expanduser(xs.DEFAULT_MTAH_ROOT))
    return refs, sample, mtah


def job_of(game, ext):
    """The external_sweep job dict of the c22 run of `game` on the raw view."""
    folds.check(game)
    root = xs.data_root(game, "raw", *_roots())
    resolved, _ = xs.resolve_config(dict(OVERRIDES))
    key = xs.run_key(game, resolved, root, xs.engine_source_hash(), "raw")
    return {"game": game, "key": key, "overrides": dict(OVERRIDES), "data_dir": root,
            "where": xs.view_dir(os.path.abspath(ext), "raw")}


def path_of(game, ext):
    """The CSV path of the c22 run of `game`. The file can be missing."""
    job = job_of(game, ext)
    return xs.run_path(job["where"], game, job["key"])


def run(games, ext, jobs=1):
    """Run c22 on each match unless its CSV is cached. Return {game: CSV path}."""
    todo = [job_of(g, ext) for g in games]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        paths = list(pool.map(xs.run_engine, todo))
    return dict(zip(games, paths))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ext", required=True, help="cache folder (views/raw/engine_data, runs)")
    parser.add_argument("--games", nargs="+", required=True)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)
    for game, path in sorted(run(args.games, args.ext, args.jobs).items()):
        print("{} {}".format(game, path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
