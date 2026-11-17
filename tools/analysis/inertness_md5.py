"""Check whether a config field is inert: run the engine at several values and compare md5 only.

    python tools/analysis/inertness_md5.py --field RELEASE_K --values 5 9 15 \
        --out-dir /tmp/bucketb/inert [--views raw matched] [--games NAME ...] [--jobs 4]

The script writes one override file {FIELD: value} for each value. It runs
`python -m engine.run --game NAME --config FILE --out CSV` once for each match, view and value.
It reads the md5 of each output file and prints a table of md5 prefixes. It prints no label and
no tracking row. It also writes `<out-dir>/<FIELD>_md5.json` with the full md5 of every run.

A field is inert on a match and a view when every value gives the same md5. The field is inert
when it is inert on every match and view that ran (COMPLIANCE_PROTOCOL.md, Section 1, "Inert
field", and grids.json, bucket b).

Views. `raw` runs each match on its own export. `matched` runs the external matches on the
smoothed view of tools/atd_view.py: sigma 3 frames for DFL and sigma 5 frames for Metrica
(COMPLIANCE_PROTOCOL.md, Section 1). BAR-ATH has the raw view only, because the competition ball
is already smoothed by its provider. The data roots are under `--mtah-root` (default
~/mtah2027):

    BAR-ATH              raw  data
    DFL-*, METRICA-S1/2  raw  refs                matched  refs-smooth-s3 (DFL), refs-smooth-s5
    METRICA-S3           raw  sample-data/data    matched  refs-smooth-s5

The engine caches its parsed arrays in cache/<game>_tactical.npz, keyed on the input paths. Two
views of one match share that file, so the script runs the views one after the other. Inside a
view it runs the matches in parallel and the values of one match in sequence.

The script refuses the sealed matches DFL-J03WQQ and DFL-J03WR9 (SPLITS.md). It reads no ground
truth file. It sets no seed because it uses none.
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEALED = ("DFL-J03WQQ", "DFL-J03WR9")
OPEN_REFERENCES = ("DFL-J03WMX", "DFL-J03WN1", "DFL-J03WOH", "DFL-J03WOY", "DFL-J03WPY",
                   "METRICA-S1", "METRICA-S2", "METRICA-S3")
DEFAULT_GAMES = ("BAR-ATH",) + OPEN_REFERENCES
VIEWS = ("raw", "matched")
MD5_LINE = re.compile(r"^md5\s+([0-9a-f]{32})\s*$", re.MULTILINE)


def data_root(game, view, base):
    """MTAH_DATA_DIR of one match in one view, or None when the view does not exist."""
    if game in SEALED:
        raise ValueError("{} is sealed".format(game))
    if view == "raw":
        if game == "BAR-ATH":
            return os.path.join(base, "data")
        if game == "METRICA-S3":
            return os.path.join(base, "sample-data", "data")
        return os.path.join(base, "refs")
    if view == "matched":
        if game == "BAR-ATH":
            return None
        if game.startswith("DFL-"):
            return os.path.join(base, "refs-smooth-s3")
        return os.path.join(base, "refs-smooth-s5")
    raise ValueError("unknown view {}".format(view))


def is_inert(md5_by_value):
    """True when every value of one match and view gave the same md5 (and none failed)."""
    hashes = list(md5_by_value.values())
    return bool(hashes) and None not in hashes and len(set(hashes)) == 1


def _value_text(value):
    """File-name text of a value: 5 -> '5', 2.5 -> '2.5'."""
    return str(value).replace("/", "_")


def _run_one(game, view, root, field, value, out_dir):
    """md5 of one engine run, or None when the run failed."""
    tag = "{}_{}_{}_{}".format(field, _value_text(value), game, view)
    config = os.path.join(out_dir, "{}_{}.json".format(field, _value_text(value)))
    csv = os.path.join(out_dir, tag + ".csv")
    env = dict(os.environ, MTAH_DATA_DIR=root)
    proc = subprocess.run([sys.executable, "-m", "engine.run", "--game", game, "--config", config,
                           "--out", csv], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True)
    with open(os.path.join(out_dir, tag + ".log"), "w") as handle:
        handle.write(proc.stdout)
    found = MD5_LINE.search(proc.stdout)
    return found.group(1) if proc.returncode == 0 and found else None


def _parse_value(text):
    """int when the text is an integer, else float."""
    try:
        return int(text)
    except ValueError:
        return float(text)


def run(field, values, games, views, out_dir, base, jobs):
    """{view: {game: {value: md5}}} for every match that has the view."""
    os.makedirs(out_dir, exist_ok=True)
    for value in values:
        with open(os.path.join(out_dir, "{}_{}.json".format(field, _value_text(value))),
                  "w") as handle:
            json.dump({field: value}, handle)
    result = {}
    for view in views:
        todo = [(game, data_root(game, view, base)) for game in games]
        todo = [(game, root) for game, root in todo if root is not None]

        def one_game(item):
            game, root = item
            return game, dict((str(v), _run_one(game, view, root, field, v, out_dir))
                              for v in values)

        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            result[view] = dict(pool.map(one_game, todo))
    return result


def report(field, result):
    """Markdown table: one row per match and view, the md5 prefix of each value."""
    values = None
    lines = []
    for view in sorted(result):
        for game in sorted(result[view]):
            row = result[view][game]
            if values is None:
                values = list(row)
                lines.append("| match | view | " + " | ".join(
                    "{} {}".format(field, v) for v in values) + " | inert |")
                lines.append("|---|---|" + "---|" * len(values) + "---|")
            cells = [(row[v] or "FAILED")[:8] for v in values]
            lines.append("| {} | {} | {} | {} |".format(game, view, " | ".join(cells),
                                                         "yes" if is_inert(row) else "NO"))
    every = all(is_inert(row) for by_game in result.values() for row in by_game.values())
    lines.append("")
    lines.append("{} inert on every match and view: {}".format(field, "yes" if every else "NO"))
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--field", required=True)
    parser.add_argument("--values", nargs="+", required=True)
    parser.add_argument("--games", nargs="+", default=list(DEFAULT_GAMES))
    parser.add_argument("--views", nargs="+", default=list(VIEWS), choices=VIEWS)
    parser.add_argument("--out-dir", required=True, help="outside the repository")
    parser.add_argument("--mtah-root", default="~/mtah2027")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    refused = sorted(set(args.games) & set(SEALED))
    if refused:
        raise SystemExit("refused: {} are sealed".format(", ".join(refused)))
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    values = [_parse_value(v) for v in args.values]
    result = run(args.field, values, args.games, args.views, out_dir,
                 os.path.expanduser(args.mtah_root), args.jobs)
    with open(os.path.join(out_dir, "{}_md5.json".format(args.field)), "w") as handle:
        json.dump(result, handle, indent=1, sort_keys=True)
    sys.stdout.write(report(args.field, result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
