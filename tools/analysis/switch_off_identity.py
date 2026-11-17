"""Check that the engine with LEARNED_MODE off writes the c22 outputs byte for byte.

    python -m tools.analysis.switch_off_identity --check [--jobs 4] [--cache DIR]
    python -m tools.analysis.switch_off_identity --record PATH --engine-root DIR

LEARNED_STAGE_PROTOCOL.md makes the switch-off identity a gate. With LEARNED_MODE off the
engine must write the same CSV as c22 on 16 external outputs: the eight
open references of external_sweep.OPEN_GAMES, in the raw and the matched views. The tool runs
engine.run on each match and view in a subprocess, through `external_sweep.py --engine-child`,
with the MTAH_DATA_DIR of external_sweep.data_root(). It never runs BAR-ATH or a sealed match.
It reads no reference and scores nothing.

- `--record PATH` writes the 16 md5s as JSON. `--engine-root` names the source tree that runs.
  We recorded RECORD from an export of commit 7ae7984, the engine before the learned stage.
- `--check` compares the md5s of this tree, run with an explicit LEARNED_MODE "off" config
  (OFF_CONFIG; the default is the learned candidate since branch r6e-adopt), with RECORD. The
  exit code is 1 on a difference.

tests/test_learned.py checks the BAR-ATH md5 of the same gate. Python 3.9.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from tools.analysis import external_sweep as xs  # noqa: E402

RECORD = os.path.join(ROOT, "docs", "plan", "calibration", "learned_switch_off_md5.json")
# Round 3 (LEARNED_STAGE_PROTOCOL_R3.md Section 0.3): the baseline is c23. Recorded with
# --engine-root on a checkout of tag `submission` (engine tree 241e70d6...), whose BAR-ATH md5 is
# 7844cad862e38d55cf9a3fc29b18ae85. Compare with `--check --against RECORD_C23`.
RECORD_C23 = os.path.join(ROOT, "docs", "plan", "calibration", "learned_switch_off_md5_c23.json")
VIEWS = ("raw", "matched")
DEFAULT_CACHE = os.path.join(ROOT, "cache", "switch_off_identity")
# The configuration --check runs: the rule engine, whatever the default of engine/config.py.
OFF_CONFIG = {"LEARNED_MODE": "off"}


def md5_of(path):
    """The hex md5 of a file."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def jobs(games=xs.OPEN_GAMES, views=VIEWS):
    """[(game, view, MTAH_DATA_DIR)] of the gate, sorted by game and view."""
    refs = os.path.expanduser(xs.DEFAULT_REFS_ROOT)
    sample = os.path.expanduser(xs.DEFAULT_SAMPLE_ROOT)
    mtah = os.path.expanduser(xs.DEFAULT_MTAH_ROOT)
    out = []
    for game in sorted(games):
        xs.check_games([game])
        for view in views:
            out.append((game, view, xs.data_root(game, view, refs, sample, mtah)))
    return out


def run_one(job, engine_root, cache, config=None):
    """The md5 of the engine output of one (game, view, data root) job: the default config, or
    the JSON config file `config`."""
    game, view, root = job
    data_cache = os.path.join(cache, view, "engine_data")
    os.makedirs(data_cache, exist_ok=True)
    handle, out = tempfile.mkstemp(suffix=".csv")
    os.close(handle)
    try:
        command = [sys.executable, os.path.join(engine_root, "tools", "analysis",
                                                "external_sweep.py"),
                   "--engine-child", data_cache, "--game", game, "--out", out]
        if config is not None:
            command += ["--config", config]
        result = subprocess.run(command, cwd=engine_root, env=dict(os.environ, MTAH_DATA_DIR=root),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True)
        if result.returncode != 0:
            raise RuntimeError("engine.run failed on {} {}:\n{}".format(
                game, view, result.stdout[-2000:]))
        return md5_of(out)
    finally:
        os.remove(out)


def measure(engine_root=ROOT, cache=DEFAULT_CACHE, n_jobs=1, config=None):
    """{"<game>/<view>": md5} of the engine output on every job of jobs(), with the default
    config or the JSON config file `config`."""
    todo = jobs()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as pool:
        md5s = list(pool.map(lambda job: run_one(job, engine_root, cache, config), todo))
    return {"{}/{}".format(game, view): md5 for (game, view, _), md5 in zip(todo, md5s)}


def compare(recorded, measured):
    """[key] of the outputs whose md5 differs or is missing on one side, sorted."""
    keys = set(recorded) | set(measured)
    return sorted(k for k in keys if recorded.get(k) != measured.get(k))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="compare with RECORD")
    action.add_argument("--record", help="write the md5s to this JSON file")
    parser.add_argument("--engine-root", default=ROOT, help="the source tree that runs")
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="engine.data npz cache root")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--against", default=RECORD,
                        help="--check: the record to compare with (round 3: RECORD_C23)")
    args = parser.parse_args(argv)
    if args.record:
        # The default of the engine root: the record trees predate the learned stage.
        measured = measure(os.path.abspath(args.engine_root), args.cache, args.jobs)
        with open(args.record, "w") as handle:
            json.dump(measured, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print("wrote {} md5s to {}".format(len(measured), args.record))
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        config = os.path.join(tmp, "off.json")
        with open(config, "w") as handle:
            json.dump(OFF_CONFIG, handle)
        measured = measure(os.path.abspath(args.engine_root), args.cache, args.jobs, config)
    with open(args.against) as handle:
        recorded = json.load(handle)
    bad = compare(recorded, measured)
    for key in sorted(measured):
        print("{:<22} {}  {}".format(key, measured[key], "DIFF" if key in bad else "same"))
    print("{} of {} outputs identical".format(len(measured) - len(bad), len(recorded)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
