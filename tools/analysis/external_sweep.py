"""Sweep engine/config.py values on the external open references only. Not part of the submission.

    python tools/analysis/external_sweep.py --field NAME --values V1 V2 ... --out DIR
    python tools/analysis/external_sweep.py --grid GRID.json --out DIR
        [--base OVERRIDES.json] [--games G1 G2 ...] [--jobs N] [--cache DIR]
        [--view raw|matched|sigma95] [--weights equal|provider] [--match-tolerance 0.002]
        [--draws 1000] [--seed 0] [--refs-root DIR] [--sample-root DIR] [--mtah-root DIR]
        [--allow-sealed]

The tool selects config values on external matches. It never runs on the competition match.

Inputs:

- `--field` and `--values` sweep one field. Each value is a JSON literal, so `10`, `1.5`,
  `true`, `"text"` and `[1, 2]` keep their type. A value that is not valid JSON is a string.
- `--grid` takes a JSON object {FIELD: [value, ...]}. The tool sweeps each field on its own,
  one field at a time, with the other fields at the base. It does not build the product.
- `--base` takes a JSON object of {FIELD: value} overrides, the `--config` file of engine.run.
  Every grid point applies its own field on top of the base. The base run is the base alone.
  With a base, you can do coordinate passes: sweep one field, fix it in the base, sweep the next.
- `--games` names the matches. The default is the eight open references of
  docs/plan/calibration/SPLITS.md and P9A_component_table.md. The 20 SkillCorner matches `SC-<id>`
  of docs/plan/calibration/SKILLCORNER_REFERENCES.md run only when you name them.

The tool refuses these matches:

- games.GAME (BAR-ATH), always. There is no flag for it. Selection on the competition match is
  forbidden (team hard rules).
- The sealed matches DFL-J03WQQ and DFL-J03WR9, unless `--allow-sealed` is given. Use that flag
  only at the final selection (docs/plan/calibration/SPLITS.md).

References. A DFL match uses the provider reference
`<refs-root>/dfl/<ID>/DFL-<ID>_reference.csv`. A Metrica match uses the event-derived reference
of tools/events_gt.py. The tool writes it once into `<cache>/refs/` with the command of
S1_S2_validation.md (S1, S2) and S3_validation.md (S3). A SkillCorner match uses the reference of
tools/skillcorner_bundle.py, `<refs-root>/skillcorner/<ID>/SC-<ID>_reference.csv`, made from the
SkillCorner phases of play. The tool checks the md5 of every reference against REFERENCE_MD5 or
SKILLCORNER_REFERENCE_MD5 and stops on a mismatch.

Views. `--view` selects the input view of the tracking. The references do not change with the
view. The roots come from docs/plan/calibration/ATD_VIEW.md, Sections 7 to 9, and
OUT_LAG_FALLBACK.md:

- `raw` (default): `MTAH_DATA_DIR` is `<refs-root>` (default ~/mtah2027/refs) for METRICA-S1,
  METRICA-S2 and DFL, and `<sample-root>` (default ~/mtah2027/sample-data/data) for METRICA-S3.
- `sigma95`: the frozen sigma 9.5 view, `<mtah-root>/refs-smooth` for all matches.
- `matched`: the matched-smoothing view of ATD_VIEW.md Section 9. DFL uses the sigma 3 view
  `<mtah-root>/refs-smooth-s3`. Metrica uses the sigma 5 view `<mtah-root>/refs-smooth-s5`.

A SkillCorner match has the raw view only, with `MTAH_DATA_DIR` `<refs-root>`. The tool refuses
the other views for it.

Runs. For each match and each config, the tool runs engine.run with `--game G --config C.json
--out R.csv` in a subprocess. The subprocess sets the npz cache folder of engine.data to
`<cache>/views/<view>/engine_data` in memory and then calls engine.run.main. So each view has its
own data cache, and the tool never edits engine/. `--jobs` runs that many engine processes at the
same time. The tool keeps each engine output in `<cache>/views/<view>/runs/<game>/<key>.csv`, and
each score in `<cache>/views/<view>/scores/`. The key
is a sha1 of the match, the view, the full resolved config, the data root and the source of
engine/ and games.py. A second sweep with the same key reads the file and does not run the
engine. Two values that resolve to the same config share one run.

Scores. The tool scores each run against the reference with tools/score_game.py. For each value
and each match it records spellF1 (`spell_f1_mean`), macroF1, the possession count and the D
share of the run, and the paired spellF1 difference against the base run. The paired interval
is the 5 to 95 percentile of the bootstrap of score_game.paired_delta (`--draws`, `--seed`).

Controlled references. `--ref-control S` scores every run against the controlled view of its
reference (tools/analysis/control_reference.py): a team run shorter than S seconds inside an
alive stretch goes to the team before it. DFL gives the ball to a team at each touch, so this
view lets DFL judge owner-side changes. The engine runs do not change, only the scores. The
default 0 keeps the reference as it is.

Outputs. For each field the tool writes `<out>/<FIELD>.json` and `<out>/<FIELD>.md`:

- spellF1 per match for each value.
- The mean over all matches, each match counted once. The mean over the Metrica matches and
  the mean over the DFL matches. When a SkillCorner match runs, also the mean over the
  SkillCorner matches (JSON key "mean_skillcorner", column "SkillCorner"). Without one, the files
  have no such key or column, as before.
- The cross-match paired difference against the base run, from tools/paired_summary.py.
- The value with the highest spellF1 on each match (the argmax), and the median of these.
- The plateau. A value is on the plateau when the paired 5 to 95 % interval of its mean
  difference to the best value contains zero. The best value is the value with the highest mean.
  Ties go to the value listed first. The JSON key "no_effect" is true when no value changes
  any score of the base run on any match.
- The plateau of each match, P_m. It holds the values v with S(m, v) >= max S(m, .) - tolerance
  (`--match-tolerance`, default 0.002). The midpoint of P_m is the mean of its first and last
  grid index. The table also gives the median of these midpoints over the matches.
- With `--weights provider`, also the provider-balanced mean. Each provider gets half the weight,
  shared equally by its matches: 1/6 for each of the three Metrica matches, 1/10 for each of the
  five DFL matches. With SkillCorner matches each of the three providers gets a third. The tool
  gives this mean its own paired interval against the base run and against the best value of
  this mean, and its own plateau. The draws are those of
  tools/paired_summary.py, so equal weights give the same interval as that tool.

Deterministic: matches in sorted order, values in the order given, a fixed bootstrap seed,
sorted JSON, no clock in the outputs. The time of a sweep goes to stderr only. Python 3.9.
"""

import argparse
import concurrent.futures
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, "tools")
for _path in (ROOT, TOOLS, os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import games  # noqa: E402
import paired_summary as ps  # noqa: E402
import score_game as sg  # noqa: E402
from analysis import control_reference as cref  # noqa: E402

# The eight open references (docs/plan/calibration/SPLITS.md, P9A_component_table.md).
OPEN_GAMES = ("METRICA-S1", "METRICA-S2", "METRICA-S3",
              "DFL-J03WMX", "DFL-J03WN1", "DFL-J03WOH", "DFL-J03WOY", "DFL-J03WPY")
# The two sealed DFL matches of docs/plan/calibration/SPLITS.md, as games.py names them.
SEALED_GAMES = ("DFL-J03WQQ", "DFL-J03WR9")
# md5 of each open reference CSV (docs/plan/calibration/OUT_LAG_FALLBACK.md, reference table).
# The SkillCorner md5s are in SKILLCORNER_REFERENCE_MD5 below.
REFERENCE_MD5 = {
    "METRICA-S1": "312d0b2f2827d7780e7e30937b7d356d",
    "METRICA-S2": "776b3a6051a757631493a9fa47d02a43",
    "METRICA-S3": "fcbfa2e124698e1d0066c4aa747885bf",
    "DFL-J03WMX": "b69baa62b53dfc824d10de5d76051418",
    "DFL-J03WN1": "a12b4a205160e4f685afb123d39dc533",
    "DFL-J03WOH": "123ee4dcc7da66813caf41dabfd75311",
    "DFL-J03WOY": "654532a8bb696122101cca7b4e60b7db",
    "DFL-J03WPY": "cb9f1d685c1603d0c9f67c255477eb10",
}
# md5 of each SkillCorner reference CSV (tools/skillcorner_bundle.py, SKILLCORNER_REFERENCES.md).
SKILLCORNER_REFERENCE_MD5 = {
    "SC-1874553": "44aecea70dc4ea8b3ff0e45ce3d7eb0d",
    "SC-1886347": "caf1a4b035f818a21f021ef10ff48c45",
    "SC-1899585": "a09a9fdadb80f7e715f6628e4b845566",
    "SC-1925299": "6cada3f17ced5020b4986991f51eab3b",
    "SC-1927964": "5580cfd11bccc2f375f99acfe18119f8",
    "SC-1953632": "9ba97c6361b32c87f29b7fc18b697bfd",
    "SC-1959846": "58ab845ab54e9b010ad7cd91461d8f8f",
    "SC-1986691": "006f7af73949912c9cf925c092b9b359",
    "SC-1996435": "3bcf71a0b4dc75790ac549edabb557f1",
    "SC-1996436": "8f3d013055d7987f81f12c58f039e030",
    "SC-2006229": "01a32e7d07ce7ab124f27e59f620c35a",
    "SC-2006363": "50b0438442037cc1e1e672154842f845",
    "SC-2007448": "4770ab54b18a95ff5c4795d2a8bc883f",
    "SC-2007721": "73809932179b76f0ea62646251e5cc08",
    "SC-2010085": "4f1563d70eaf481bca4dfe37da51b453",
    "SC-2011166": "8a40a74dfc24027654f85ec86a5f42f0",
    "SC-2013725": "ffb4998982607a4e7b847300537efcfb",
    "SC-2015213": "ca0794cf0f86b7e5e9ead4c664fdc6e9",
    "SC-2016236": "6e9582052fdc8eb0bd05e8acb89887dd",
    "SC-2017461": "0cc220679f159d49f8c7700057cc4558",
}
# The Metrica event files under <sample-root>, and the data root that tools/events_gt.py reads
# (S1_S2_validation.md 1.4 and the reproduce block, S3_validation.md 1.1).
METRICA_EVENTS = {
    "METRICA-S1": ("Sample_Game_1/Sample_Game_1_RawEventsData.csv", "refs"),
    "METRICA-S2": ("Sample_Game_2/Sample_Game_2_RawEventsData.csv", "refs"),
    "METRICA-S3": ("Sample_Game_3/Sample_Game_3_events.json", "sample"),
}
DEFAULT_REFS_ROOT = os.path.join("~", "mtah2027", "refs")
DEFAULT_SAMPLE_ROOT = os.path.join("~", "mtah2027", "sample-data", "data")
DEFAULT_MTAH_ROOT = os.path.join("~", "mtah2027")
# The smoothed views under <mtah-root>, per provider (ATD_VIEW.md Sections 7 and 9). "raw" uses
# <refs-root> and <sample-root>. Matched: the sigma closest to BAR-ATH, 3 on DFL and 5 on Metrica.
VIEWS = {
    "raw": None,
    "sigma95": {"dfl": "refs-smooth", "metrica": "refs-smooth"},
    "matched": {"dfl": "refs-smooth-s3", "metrica": "refs-smooth-s5"},
}
# Per-match plateau: values within this spellF1 of the best value on that match (coordinator's
# selection protocol, 8 Nov 2026).
DEFAULT_MATCH_TOLERANCE = 0.002
# cache/ is gitignored (.gitignore). Engine outputs never get committed.
DEFAULT_CACHE = os.path.join(ROOT, "cache", "external_sweep")
DEFAULT_DRAWS = sg.DEFAULT_DRAWS
DEFAULT_SEED = ps.DEFAULT_SEED
KEY_LENGTH = 16   # hex characters of a run key
GROUPS = (("metrica", "METRICA-"), ("dfl", "DFL-"))
# The third provider, SkillCorner Open Data (docs/plan/calibration/SKILLCORNER_REFERENCES.md).
# Its mean and its column appear only when a sweep runs a SkillCorner match, so that a sweep of
# the eight open references writes the same files as before.
SKILLCORNER_PREFIX = "SC-"
SKILLCORNER_GROUP = ("skillcorner", SKILLCORNER_PREFIX)
GROUP_TITLES = {"metrica": "Metrica", "dfl": "DFL", "skillcorner": "SkillCorner"}


# -- inputs ----------------------------------------------------------------------------------

def parse_value(text):
    """A CLI value as a JSON literal. Text that is not valid JSON stays a string."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def value_label(value):
    """The printed form of a value: its compact JSON."""
    return json.dumps(value, sort_keys=True)


def check_games(names, allow_sealed=False):
    """The sorted match names. Raise SystemExit for a forbidden or unknown match."""
    out = sorted(set(names))
    for game in out:
        if game == games.GAME:
            raise SystemExit("{} is the competition match. The sweep never runs on it "
                             "(team hard rules).".format(game))
        if game in SEALED_GAMES and not allow_sealed:
            raise SystemExit("{} is sealed (docs/plan/calibration/SPLITS.md). Pass "
                             "--allow-sealed at the final selection only.".format(game))
        if game not in games.GAMES:
            raise SystemExit("{} is not a games.py entry".format(game))
        if not (game.startswith("DFL-") or game in METRICA_EVENTS
                or game.startswith(SKILLCORNER_PREFIX)):
            raise SystemExit("{} has no external reference for this tool".format(game))
    if not out:
        raise SystemExit("no match to run")
    return out


def load_sweeps(field=None, values=None, grid=None):
    """[(field, [value, ...]), ...] sorted by field, from --field/--values or a --grid file."""
    if grid is not None:
        with open(grid) as handle:
            spec = json.load(handle)
        if not isinstance(spec, dict) or not spec:
            raise SystemExit("{}: expected a JSON object {{FIELD: [values]}}".format(grid))
        sweeps = []
        for name in sorted(spec):
            if not isinstance(spec[name], list) or not spec[name]:
                raise SystemExit("{}: {} needs a non-empty list of values".format(grid, name))
            sweeps.append((name, list(spec[name])))
        return sweeps
    if not field or not values:
        raise SystemExit("give --field and --values, or --grid")
    return [(field, [parse_value(v) for v in values])]


def md5_of(path):
    """md5 hex digest of a file."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def data_root(game, view, refs_root, sample_root, mtah_root):
    """The MTAH_DATA_DIR of a match in a view (P9A_component_table.md, ATD_VIEW.md 7 to 9).

    A SkillCorner match has the raw view only. Raise SystemExit for any other view.
    """
    if game.startswith(SKILLCORNER_PREFIX):
        if VIEWS[view] is not None:
            raise SystemExit("{}: view {} is not defined for SkillCorner. Use --view raw "
                             "(SKILLCORNER_REFERENCES.md).".format(game, view))
        return refs_root
    if VIEWS[view] is None:
        return sample_root if game == "METRICA-S3" else refs_root
    provider = "dfl" if game.startswith("DFL-") else "metrica"
    return os.path.join(mtah_root, VIEWS[view][provider])


def prepare_reference(game, refs_root, sample_root, cache):
    """(path, md5) of the reference of a match. Write a Metrica reference once into the cache.

    Raise SystemExit if the md5 differs from REFERENCE_MD5. A sealed match has no recorded md5.
    """
    if game.startswith("DFL-"):
        match_id = game[len("DFL-"):]
        path = os.path.join(refs_root, "dfl", match_id, "DFL-{}_reference.csv".format(match_id))
    elif game.startswith(SKILLCORNER_PREFIX):
        match_id = game[len(SKILLCORNER_PREFIX):]
        path = os.path.join(refs_root, "skillcorner", match_id,
                            "{}{}_reference.csv".format(SKILLCORNER_PREFIX, match_id))
    else:
        path = os.path.join(cache, "refs", "{}_ref.csv".format(game))
        if not os.path.isfile(path):
            events, root = METRICA_EVENTS[game]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            env = dict(os.environ, MTAH_DATA_DIR=refs_root if root == "refs" else sample_root)
            tmp = path + ".tmp"
            with open(os.devnull, "w") as quiet:
                subprocess.run([sys.executable, os.path.join(TOOLS, "events_gt.py"), "--game",
                                game, "--events", os.path.join(sample_root, events),
                                "--out", tmp], cwd=ROOT, env=env, stdout=quiet, check=True)
            os.replace(tmp, path)
    if os.path.abspath(path).startswith(os.path.join(ROOT, "evaluation")):
        raise SystemExit("{}: a reference under evaluation/ is not external".format(path))
    digest = md5_of(path)
    expected = REFERENCE_MD5.get(game, SKILLCORNER_REFERENCE_MD5.get(game))
    if expected is not None and digest != expected:
        raise SystemExit("{}: reference md5 {} differs from the recorded {}".format(
            game, digest, expected))
    return path, digest


def control_view(game, reference, seconds, cache):
    """(path, md5) of the controlled view of a reference (tools/analysis/control_reference.py).

    The view goes to <cache>/refs-control-<seconds>s/. Its md5 differs from the md5 of the
    reference, so its scores get their own cache entries.
    """
    src, digest = reference
    folder = os.path.join(cache, "refs-control-{:g}s".format(seconds))
    dst = os.path.join(folder, "{}_{}.csv".format(game, digest[:12]))
    if not os.path.isfile(dst):
        cref.write_control_reference(src, dst, seconds)
    return dst, md5_of(dst)


def engine_source_hash():
    """sha1 of the source of engine/*.py and games.py. A code change gives new run keys."""
    digest = hashlib.sha1()
    paths = sorted(glob.glob(os.path.join(ROOT, "engine", "*.py")))
    for path in paths + [os.path.join(ROOT, "games.py")]:
        digest.update(os.path.relpath(path, ROOT).encode("utf-8"))
        with open(path, "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()


def resolve_config(overrides):
    """(full config dict, Config.config_hash()) of a {FIELD: value} overrides dict.

    The tool loads the overrides with engine.config.load(), as engine.run does. So an unknown
    field raises, and each value takes the type of its default.
    """
    from engine import config
    handle, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(handle, "w") as out:
            json.dump(overrides, out, sort_keys=True)
        cfg = config.load(path)
    finally:
        os.remove(path)
    return cfg.as_dict(), cfg.config_hash()


def run_key(game, resolved, root, source_hash, view="raw"):
    """The cache key of one engine run: match, view, full config, data root, engine source."""
    text = json.dumps({"game": game, "view": view, "config": resolved, "data_dir": root,
                       "engine": source_hash}, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:KEY_LENGTH]


# -- engine runs and scores ------------------------------------------------------------------

def view_dir(cache, view):
    """<cache>/views/<view>: the runs, scores and engine data cache of one view."""
    return os.path.join(cache, "views", view)


def run_path(where, game, key, suffix=".csv"):
    """<where>/runs/<game>/<key><suffix>. `where` is a view_dir()."""
    return os.path.join(where, "runs", game, key + suffix)


def engine_child(data_cache, argv):
    """Run engine.run.main(argv) with the npz cache of engine.data in `data_cache`.

    The change lives in this process only. engine/ stays as it is.
    """
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from engine import data
    from engine import run as engine_run
    data.DEFAULT_CACHE_DIR = data_cache
    return engine_run.main(argv)


def run_engine(job):
    """Run engine.run for one job dict, unless its CSV is in the cache. Return the CSV path.

    The engine prints only its full-match summary. The tool writes it to the .log file next to
    the CSV and does not print it.
    """
    where = job["where"]
    csv_path = run_path(where, job["game"], job["key"])
    if os.path.isfile(csv_path):
        return csv_path
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    config_path = run_path(where, job["game"], job["key"], ".config.json")
    with open(config_path, "w") as handle:
        json.dump(job["overrides"], handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp = run_path(where, job["game"], job["key"], ".tmp.csv")
    log_path = run_path(where, job["game"], job["key"], ".log")
    env = dict(os.environ, MTAH_DATA_DIR=job["data_dir"])
    command = [sys.executable, os.path.abspath(__file__), "--engine-child",
               os.path.join(where, "engine_data"), "--game", job["game"], "--config",
               config_path, "--out", tmp]
    with open(log_path, "w") as log:
        code = subprocess.call(command, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT)
    if code != 0:
        raise RuntimeError("engine.run failed on {} (key {}), see {}".format(
            job["game"], job["key"], log_path))
    os.replace(tmp, csv_path)
    return csv_path


def score_record(gt_codes, pred_codes):
    """The numbers the sweep keeps for one run, from tools/score_game.py.

    spellF1, macroF1, the possession count and D share of the run on the graded frames of the
    reference, the predicted spell count and the per-spell match flags of paired_summary.
    """
    result, _, pred_stats = sg.score_codes(gt_codes, pred_codes)
    return {"spell_f1": float(result["spell_f1_mean"]), "macro_f1": float(result["macro_f1"]),
            "possessions": int(pred_stats["possessions"]),
            "d_share": float(pred_stats["d_share"]),
            "n_pred_spells": int(result["n_pred_spells"]),
            "flags": sg.match_flags(gt_codes, pred_codes)}


def cached_score(where, game, key, ref_md5, gt_codes):
    """score_record() of one run, cached in <where>/scores/<game>/<key>_<ref md5>.json."""
    path = os.path.join(where, "scores", game, "{}_{}.json".format(key, ref_md5[:12]))
    if os.path.isfile(path):
        with open(path) as handle:
            return json.load(handle)
    pred = sg.read_codes(run_path(where, game, key), len(gt_codes))
    record = score_record(gt_codes, pred)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as handle:
        json.dump(record, handle, sort_keys=True)
    os.replace(path + ".tmp", path)
    return record


# -- the table -------------------------------------------------------------------------------

def _flags(record):
    rows = [record["flags"]["{:.1f}".format(iou)] for iou in ps.pm.SPELL_IOUS]
    return np.asarray(rows, dtype=np.int8).reshape(len(ps.pm.SPELL_IOUS), -1)


def paired_game(record_a, record_b, n_draws=DEFAULT_DRAWS, seed=DEFAULT_SEED):
    """Paired spellF1 difference a - b on one match from two score records.

    The same draws as score_game.paired_delta(): one RandomState(seed), and in each draw one
    resample of the reference possessions for both runs. Returns {"delta", "low", "high"}.
    """
    flags_a, flags_b = _flags(record_a), _flags(record_b)
    n_a, n_b = record_a["n_pred_spells"], record_b["n_pred_spells"]
    n_gt = flags_a.shape[1]
    every = np.arange(n_gt)
    delta = ps.mean_f1(flags_a, n_a, every) - ps.mean_f1(flags_b, n_b, every)
    rng = np.random.RandomState(seed)
    draws = np.empty(n_draws, dtype=np.float64)
    for k in range(n_draws):
        picks = rng.randint(0, n_gt, n_gt)
        draws[k] = ps.mean_f1(flags_a, n_a, picks) - ps.mean_f1(flags_b, n_b, picks)
    if not n_draws:
        return {"delta": delta, "low": None, "high": None}
    low, high = np.percentile(draws, ps.pm.BOOTSTRAP_PERCENTILES)
    return {"delta": delta, "low": float(low), "high": float(high)}


def _side(records, label):
    """The by_reference() dict of tools/paired_summary.py from {game: score record}."""
    return {game: (game, _flags(rec), rec["n_pred_spells"], label)
            for game, rec in records.items()}


def cross_match(records_a, records_b, n_draws=DEFAULT_DRAWS, seed=DEFAULT_SEED):
    """paired_summary() of a - b over the matches: {"mean_delta", "low", "high"}."""
    out = ps.paired_summary(_side(records_a, "a"), _side(records_b, "b"), n_draws, seed)
    return {"mean_delta": out["mean_delta"], "low": out["low"], "high": out["high"]}


def provider_weights(names):
    """{game: weight}: each provider present gets an equal share, split equally by its matches.

    With the three Metrica and the five DFL matches: 1/6 for each Metrica match and 1/10 for
    each DFL match.
    """
    counts = {}
    for game in names:
        counts[_provider(game)] = counts.get(_provider(game), 0) + 1
    return {game: 1.0 / (len(counts) * counts[_provider(game)]) for game in names}


def _groups(names):
    """GROUPS, plus SKILLCORNER_GROUP if a SkillCorner match is in `names`."""
    if any(g.startswith(SKILLCORNER_PREFIX) for g in names):
        return GROUPS + (SKILLCORNER_GROUP,)
    return GROUPS


def _provider(game):
    for group, prefix in GROUPS + (SKILLCORNER_GROUP,):
        if game.startswith(prefix):
            return group
    raise ValueError("{}: unknown provider".format(game))


def weighted_cross_match(records_a, records_b, weights, n_draws=DEFAULT_DRAWS,
                         seed=DEFAULT_SEED):
    """Weighted mean of the per-match paired differences a - b, with its 5 to 95 % interval.

    The draws are those of paired_summary.paired_summary(): one RandomState(seed), and in each
    draw one resample per match in sorted order. Equal weights give that tool's result.
    Returns {"mean_delta", "low", "high"}.
    """
    names = sorted(records_a)
    if sorted(records_b) != names or sorted(weights) != names:
        raise ValueError("the two versions and the weights must name the same matches")
    w = np.array([weights[g] for g in names], dtype=np.float64)
    w = w / w.sum()
    sides = [(_flags(records_a[g]), records_a[g]["n_pred_spells"], _flags(records_b[g]),
              records_b[g]["n_pred_spells"]) for g in names]
    rng = np.random.RandomState(seed)
    per_match = np.empty((n_draws, len(names)), dtype=np.float64)
    for draw in range(n_draws):
        for col, (flags_a, n_a, flags_b, n_b) in enumerate(sides):
            n_gt = flags_a.shape[1]
            picks = rng.randint(0, n_gt, n_gt)
            per_match[draw, col] = ps.mean_f1(flags_a, n_a, picks) - ps.mean_f1(flags_b, n_b,
                                                                                picks)
    deltas = np.array([ps.mean_f1(fa, na, np.arange(fa.shape[1]))
                       - ps.mean_f1(fb, nb, np.arange(fb.shape[1]))
                       for fa, na, fb, nb in sides])
    if not n_draws:
        return {"mean_delta": float(np.dot(w, deltas)), "low": None, "high": None}
    low, high = np.percentile(per_match.dot(w), ps.pm.BOOTSTRAP_PERCENTILES)
    return {"mean_delta": float(np.dot(w, deltas)), "low": float(low), "high": float(high)}


def match_plateaus(values, scores, names, tolerance=DEFAULT_MATCH_TOLERANCE):
    """{game: P_m}: the values within `tolerance` of the best spellF1 on each match.

    P_m is {"values", "indices", "best", "midpoint_index", "midpoint_value"}. The midpoint is the
    mean of the first and the last grid index of P_m. Its value is the grid value at that index,
    or the mean of the two neighbouring numeric values on a half index, else None.
    """
    out = {}
    for g in names:
        spells = [per_game[g]["spell_f1"] for per_game in scores]
        top = max(spells)
        indices = [i for i, s in enumerate(spells) if s >= top - tolerance]
        mid = 0.5 * (indices[0] + indices[-1])
        if mid == int(mid):
            mid_value = values[int(mid)]
        else:
            pair = values[int(mid)], values[int(mid) + 1]
            mid_value = (0.5 * (pair[0] + pair[1]) if all(_numeric(v) for v in pair)
                         else None)
        out[g] = {"values": [values[i] for i in indices], "indices": indices, "best": top,
                  "midpoint_index": mid, "midpoint_value": mid_value}
    return out


def _numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _mean(values):
    return float(np.mean(values)) if values else None


def field_table(field, values, scores, base_scores, n_draws=DEFAULT_DRAWS, seed=DEFAULT_SEED,
                weights="equal", tolerance=DEFAULT_MATCH_TOLERANCE):
    """The sweep table of one field as a dict.

    `values` is the list of grid values. `scores[i]` is {game: score record} of value i.
    `base_scores` is {game: score record} of the base run. Every dict has the same matches.
    `weights` "provider" adds the provider-balanced mean, its intervals and its plateau.
    `tolerance` sets the per-match plateau.
    """
    names = sorted(base_scores)
    for i, per_game in enumerate(scores):
        if sorted(per_game) != names:
            raise ValueError("value {} has matches {}, the base has {}".format(
                value_label(values[i]), sorted(per_game), names))
    balance = provider_weights(names) if weights == "provider" else None
    rows = []
    for value, per_game in zip(values, scores):
        spells = [per_game[g]["spell_f1"] for g in names]
        row = {"value": value, "label": value_label(value), "games": {},
               "mean": _mean(spells)}
        for group, prefix in _groups(names):
            row["mean_" + group] = _mean([per_game[g]["spell_f1"] for g in names
                                          if g.startswith(prefix)])
        for g in names:
            rec = per_game[g]
            row["games"][g] = dict(
                {k: rec[k] for k in ("spell_f1", "macro_f1", "possessions", "d_share")},
                versus_base=paired_game(rec, base_scores[g], n_draws, seed))
        row["versus_base"] = cross_match(per_game, base_scores, n_draws, seed)
        if balance is not None:
            row["mean_provider"] = float(sum(balance[g] * per_game[g]["spell_f1"]
                                             for g in names))
            row["versus_base_provider"] = weighted_cross_match(per_game, base_scores, balance,
                                                               n_draws, seed)
        rows.append(row)

    def plateau(best, suffix):
        for row, per_game in zip(rows, scores):
            if suffix:
                pair = weighted_cross_match(per_game, scores[best], balance, n_draws, seed)
            else:
                pair = cross_match(per_game, scores[best], n_draws, seed)
            row["versus_best" + suffix] = pair
            high = pair["high"]
            row["plateau" + suffix] = bool(row is rows[best] or (high is not None
                                                                 and high >= 0.0))
        return [row["value"] for row in rows if row["plateau" + suffix]]

    best = int(np.argmax([row["mean"] for row in rows]))   # argmax: the first on a tie
    table = {"plateau": plateau(best, "")}
    if balance is not None:
        best_provider = int(np.argmax([row["mean_provider"] for row in rows]))
        table.update(plateau_provider=plateau(best_provider, "_provider"),
                     best_provider=values[best_provider],
                     weights={g: balance[g] for g in names})
    argmax = {}
    for g in names:
        spells = [per_game[g]["spell_f1"] for per_game in scores]
        argmax[g] = values[int(np.argmax(spells))]
    numeric = all(_numeric(v) for v in argmax.values())
    per_match = match_plateaus(values, scores, names, tolerance)
    same = ("spell_f1", "macro_f1", "possessions", "d_share", "n_pred_spells", "flags")
    no_effect = all(per_game[g][k] == base_scores[g][k]
                    for per_game in scores for g in names for k in same)
    table.update({
        "no_effect": bool(no_effect),
        "field": field,
        "values": list(values),
        "games": names,
        "rows": rows,
        "base": {g: {k: base_scores[g][k] for k in ("spell_f1", "macro_f1", "possessions",
                                                    "d_share")} for g in names},
        "base_mean": _mean([base_scores[g]["spell_f1"] for g in names]),
        "best": values[best],
        "argmax": argmax,
        "median_argmax": float(np.median([argmax[g] for g in names])) if numeric else None,
        "match_plateaus": per_match,
        "match_tolerance": float(tolerance),
        "median_match_midpoint_index": float(np.median(
            [per_match[g]["midpoint_index"] for g in names])),
        "weighting": weights,
        "draws": int(n_draws),
        "seed": int(seed),
    })
    return table


def _num(value, fmt="{:.4f}"):
    return "-" if value is None else fmt.format(value)


def _interval(pair, key="delta"):
    if pair["low"] is None:
        return "{:+.4f}".format(pair[key])
    return "{:+.4f} [{:+.4f}, {:+.4f}]".format(pair[key], pair["low"], pair["high"])


def _labels(values):
    return ", ".join("`{}`".format(value_label(v)) for v in values)


def format_markdown(table, base_overrides=None, view="raw"):
    """The Markdown page of one field_table()."""
    names = table["games"]
    provider = table.get("weighting") == "provider"
    short = [g.replace("METRICA-", "M-").replace("DFL-", "") for g in names]
    lines = ["# External sweep: {}".format(table["field"]), ""]
    lines.append("View: `{}`. Base overrides: `{}`. Matches: {}. Paired bootstrap: {} draws, "
                 "seed {}.".format(view, value_label(base_overrides or {}), ", ".join(names),
                                   table["draws"], table["seed"]))
    lines += ["", "## spellF1 per match", ""]
    groups = [group for group, _ in _groups(names)]
    head = ["value"] + short + ["mean"] + [GROUP_TITLES[group] for group in groups] + [
        "Δ vs base [5–95 %]", "Δ vs best [5–95 %]", "plateau"]
    if provider:
        head += ["provider mean", "Δ vs base, provider [5–95 %]",
                 "Δ vs best, provider [5–95 %]", "plateau, provider"]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    for row in table["rows"]:
        cells = [row["label"]] + [_num(row["games"][g]["spell_f1"]) for g in names]
        cells += [_num(row["mean"])] + [_num(row["mean_" + group]) for group in groups]
        cells += [_interval(row["versus_base"], "mean_delta"),
                  _interval(row["versus_best"], "mean_delta"),
                  "yes" if row["plateau"] else ""]
        if provider:
            cells += [_num(row["mean_provider"]),
                      _interval(row["versus_base_provider"], "mean_delta"),
                      _interval(row["versus_best_provider"], "mean_delta"),
                      "yes" if row["plateau_provider"] else ""]
        lines.append("| " + " | ".join(cells) + " |")
    pad = len(head) - len(names) - 1
    base_row = ["base"] + [_num(table["base"][g]["spell_f1"]) for g in names]
    base_row += [_num(table["base_mean"])] + [""] * (pad - 1)
    lines.append("| " + " | ".join(base_row) + " |")
    arg_row = ["argmax"] + [value_label(table["argmax"][g]) for g in names] + [""] * pad
    lines.append("| " + " | ".join(arg_row) + " |")
    lines += ["", "Best mean: `{}`. Median of the per-match argmax: {}. Plateau: {}.".format(
        value_label(table["best"]), _num(table["median_argmax"], "{:g}"),
        _labels(table["plateau"]))]
    if table.get("no_effect"):
        lines += ["", "No value changes any score on any match. The best value and the argmax "
                      "are the first value only because ties go to the value listed first."]
    if provider:
        lines += ["", "Best provider-balanced mean: `{}`. Provider plateau: {}.".format(
            value_label(table["best_provider"]), _labels(table["plateau_provider"]))]
    lines += ["", "## Plateau of each match", "",
              "P_m holds the values within {:g} of the best spellF1 on the match. The midpoint "
              "is the mean of the first and the last grid index of P_m.".format(
                  table["match_tolerance"]), "",
              "| match | best spellF1 | P_m | midpoint index | midpoint value |",
              "|---|---|---|---|---|"]
    for g in names:
        item = table["match_plateaus"][g]
        lines.append("| {} | {:.4f} | {} | {:g} | {} |".format(
            g, item["best"], _labels(item["values"]), item["midpoint_index"],
            "-" if item["midpoint_value"] is None else value_label(item["midpoint_value"])))
    lines += ["", "Median midpoint index over the matches: {:g}.".format(
        table["median_match_midpoint_index"])]
    lines += ["", "## Each run against the base run", "",
              "| value | match | spellF1 | macroF1 | possessions | D share | "
              "Δ spellF1 vs base [5–95 %] |",
              "|---|---|---|---|---|---|---|"]
    for row in table["rows"]:
        for g in names:
            cell = row["games"][g]
            lines.append("| {} | {} | {:.4f} | {:.4f} | {} | {:.1f} % | {} |".format(
                row["label"], g, cell["spell_f1"], cell["macro_f1"], cell["possessions"],
                100.0 * cell["d_share"], _interval(cell["versus_base"])))
    return "\n".join(lines) + "\n"


# -- the sweep -------------------------------------------------------------------------------

def sweep(sweeps, game_names, base_overrides, out_dir, cache, jobs=1, n_draws=DEFAULT_DRAWS,
          seed=DEFAULT_SEED, refs_root=None, sample_root=None, view="raw", mtah_root=None,
          weights="equal", tolerance=DEFAULT_MATCH_TOLERANCE, ref_control=0.0):
    """Run, score and tabulate every sweep. Write <out>/<FIELD>.json and .md. Return the paths."""
    refs_root = os.path.abspath(os.path.expanduser(refs_root or DEFAULT_REFS_ROOT))
    sample_root = os.path.abspath(os.path.expanduser(sample_root or DEFAULT_SAMPLE_ROOT))
    mtah_root = os.path.abspath(os.path.expanduser(mtah_root or DEFAULT_MTAH_ROOT))
    cache = os.path.abspath(cache)
    where = view_dir(cache, view)
    roots = {g: data_root(g, view, refs_root, sample_root, mtah_root) for g in game_names}
    for g in game_names:
        if not os.path.isdir(roots[g]):
            raise SystemExit("{}: data root {} of view {} does not exist".format(
                g, roots[g], view))
    # The references do not depend on the view.
    references = {g: prepare_reference(g, refs_root, sample_root, cache) for g in game_names}
    if ref_control:
        references = {g: control_view(g, references[g], ref_control, cache) for g in game_names}
        view = "{}, references controlled {:g} s".format(view, ref_control)
    source_hash = engine_source_hash()

    def plan(overrides):
        resolved, config_hash = resolve_config(overrides)
        keys = {g: run_key(g, resolved, roots[g], source_hash, view) for g in game_names}
        return {"overrides": overrides, "config_hash": config_hash, "keys": keys}

    base = plan(dict(base_overrides))
    points = [(field, [plan(dict(base_overrides, **{field: v})) for v in values])
              for field, values in sweeps]
    jobs_by_key = {}
    for item in [base] + [p for _, plans in points for p in plans]:
        for g in game_names:
            jobs_by_key[(g, item["keys"][g])] = {
                "game": g, "key": item["keys"][g], "overrides": item["overrides"],
                "data_dir": roots[g], "where": where}
    ordered = [jobs_by_key[k] for k in sorted(jobs_by_key)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        list(pool.map(run_engine, ordered))

    gt = {g: sg.read_codes(references[g][0]) for g in game_names}

    def scores_of(item):
        return {g: cached_score(where, g, item["keys"][g], references[g][1], gt[g])
                for g in game_names}

    base_scores = scores_of(base)
    written = []
    os.makedirs(out_dir, exist_ok=True)
    for (field, values), (_, plans) in zip(sweeps, points):
        table = field_table(field, values, [scores_of(p) for p in plans], base_scores,
                            n_draws, seed, weights, tolerance)
        table.update(
            view=view, base_overrides=dict(base_overrides), base_config_hash=base["config_hash"],
            config_hashes=[p["config_hash"] for p in plans], engine_source=source_hash,
            references={g: references[g][1] for g in game_names},
            run_keys={"base": base["keys"], "values": [p["keys"] for p in plans]})
        json_path = os.path.join(out_dir, "{}.json".format(field))
        with open(json_path, "w") as handle:
            json.dump(table, handle, indent=2, sort_keys=True)
            handle.write("\n")
        md_path = os.path.join(out_dir, "{}.md".format(field))
        with open(md_path, "w") as handle:
            handle.write(format_markdown(table, base_overrides, view))
        written += [json_path, md_path]
    return written


def main(argv=None):
    """CLI entry point. Return the process exit code."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["--engine-child"]:
        # Internal: one engine run with the npz cache of the view (run_engine).
        return engine_child(argv[1], argv[2:])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--field", help="config field to sweep")
    parser.add_argument("--values", nargs="+", help="values of --field, as JSON literals")
    parser.add_argument("--grid", help="JSON object {FIELD: [values]}, one sweep per field")
    parser.add_argument("--base", help="JSON object of {FIELD: value} under every grid point")
    parser.add_argument("--games", nargs="+", default=list(OPEN_GAMES),
                        help="matches (default: the eight open references)")
    parser.add_argument("--out", required=True, help="output folder of the tables")
    parser.add_argument("--view", choices=sorted(VIEWS), default="raw",
                        help="input view of the tracking (default: raw)")
    parser.add_argument("--weights", choices=("equal", "provider"), default="equal",
                        help="provider: add the provider-balanced mean (default: equal only)")
    parser.add_argument("--match-tolerance", type=float, default=DEFAULT_MATCH_TOLERANCE,
                        help="spellF1 tolerance of the per-match plateau (default: {:g})".format(
                            DEFAULT_MATCH_TOLERANCE))
    parser.add_argument("--jobs", type=int, default=1, help="engine runs at the same time")
    parser.add_argument("--cache", default=DEFAULT_CACHE,
                        help="engine outputs, references and scores (default: {})".format(
                            os.path.relpath(DEFAULT_CACHE, ROOT)))
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS, help="bootstrap draws")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="bootstrap seed")
    parser.add_argument("--refs-root", default=DEFAULT_REFS_ROOT,
                        help="raw MTAH_DATA_DIR of METRICA-S1, S2 and DFL, and the DFL references")
    parser.add_argument("--sample-root", default=DEFAULT_SAMPLE_ROOT,
                        help="raw MTAH_DATA_DIR of METRICA-S3 and the Metrica event files")
    parser.add_argument("--mtah-root", default=DEFAULT_MTAH_ROOT,
                        help="the folder of the smoothed views (default: ~/mtah2027)")
    parser.add_argument("--ref-control", type=float, default=0.0,
                        help="score against the controlled view of each reference, with this "
                             "minimum control time in seconds (default: 0, the reference as is)")
    parser.add_argument("--allow-sealed", action="store_true",
                        help="allow the sealed matches (the final selection only)")
    args = parser.parse_args(argv)
    if args.grid and (args.field or args.values):
        parser.error("give --grid or --field/--values, not both")
    game_names = check_games(args.games, args.allow_sealed)
    sweeps = load_sweeps(args.field, args.values, args.grid)
    base_overrides = {}
    if args.base:
        with open(args.base) as handle:
            base_overrides = json.load(handle)
        if not isinstance(base_overrides, dict):
            parser.error("{}: expected a JSON object of overrides".format(args.base))
    out_dir = os.path.abspath(args.out)
    if out_dir.startswith(os.path.join(ROOT, "evaluation")):
        parser.error("never write the sweep under evaluation/")
    start = time.time()
    written = sweep(sweeps, game_names, base_overrides, out_dir, args.cache, args.jobs,
                    args.draws, args.seed, args.refs_root, args.sample_root, args.view,
                    args.mtah_root, args.weights, args.match_tolerance, args.ref_control)
    for path in written:
        print("wrote {}".format(path))
    # The time of a sweep goes to stderr only, never into an output file.
    sys.stderr.write("sweep time {:.1f} s\n".format(time.time() - start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
