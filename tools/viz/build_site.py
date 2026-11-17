"""Build the public bundle of the visualizer (site/data/SCHEMA.md Section 2).

    python -m tools.viz.build_site --version <name> --results evaluation/results/<name>.csv
                                   [--game BAR-ATH] [--out site/data/public]
                                   [--allow-md5-mismatch]

Run the scorer first (`python evaluation/run_evaluation.py`). The exporter refuses a version
that has no report in `evaluation/reports/`. It runs the engine once, reads the scorer reports,
the forensics, the sanity statistics, the config and its provenance classes, the external
validation numbers of the technical document, the ledger and the git identity, and writes
one JSON file per section of the site.

The public bundle holds frame-level detail only inside the public window. The rest of the match
appears as per-minute aggregates. write_bundle() is the only writer, and it runs the guard on
every object before it writes a file.
"""

import argparse
import copy
import glob
import json
import os
import re
import sys

import numpy as np

from tools.viz import guard, rle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(ROOT, "site", "data", "public")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
REPORTS_DIR = os.path.join(ROOT, "evaluation", "reports")
README_PATH = os.path.join(ROOT, "README.md")
# External validation numbers of the technical document (tools/docs/external_results.py). They
# come from other matches only. external_summary.json keeps the summary numbers the page shows.
EXTERNAL_PATH = os.path.join(ROOT, "docs", "technical", "data", "external.json")
# Origin class of each engine constant (docs/plan/calibration/PROVENANCE.md).
PROVENANCE_PATH = os.path.join(ROOT, "docs", "plan", "calibration", "provenance_current.json")
TESTS_DIR = os.path.join(ROOT, "tests")

# Schema version of both bundles (SCHEMA.md 2.1).
SCHEMA_VERSION = 1
# Label codes (SCHEMA.md Section 1).
H, A, D, X = 0, 1, 2, 3
TEAMS = ("H", "A")
# One minute bin is 60 s at 25 fps (SCHEMA.md Section 4).
BIN_FRAMES = 1500
# The organiser baselines whose full-match labels are public (SCHEMA.md 2.3).
BASELINES = ("baseline_smooth50", "baseline_smooth125")
# The baseline that the minute bins and the sanity file compare with (SCHEMA.md 2.4, 2.8).
REFERENCE_BASELINE = "baseline_smooth50"
# The organiser baselines that scores.json carries when they have a report (SCHEMA.md 2.6).
SCORE_BASELINES = ("baseline_raw", "baseline_smooth25", "baseline_smooth50", "baseline_smooth125")
# Report keys that scores.json copies unchanged (SCHEMA.md 2.6).
SCORE_KEYS = ("spell_f1_mean", "macro_f1", "spell_f1", "spell_f1_at_headline",
              "spell_iou_headline", "transitions", "taus", "per_class", "confusion",
              "n_gt_spells", "n_pred_spells", "n_gt_segments", "n_pred_segments",
              "spell_count_error", "mean_spell_duration_s", "gt_mean_spell_duration_s",
              "bootstrap_low", "bootstrap_high", "bootstrap_n", "spell_bootstrap_low",
              "spell_bootstrap_high", "accuracy", "kappa", "alive_dead_f1", "fragmentation",
              "share_err", "pred_home_share", "gt_home_share")
# Decimals of a confidence (SCHEMA.md Section 1) and of a bin share (SCHEMA.md 2.4).
CONFIDENCE_DECIMALS = 3
SHARE_DECIMALS = 4
# Python version of the reproduction card (team rule: "Keep the code compatible with Python 3.9").
REPRO_PYTHON = "3.9"
REPRO_COMMAND = "python -m engine.run --out evaluation/results/{}.csv"
# Requirement files whose exact pins the reproduction card lists (plan "Exporters").
REQUIREMENT_FILES = ("requirements.txt", "requirements-dev.txt", "requirements-starter.txt")
PIN_PATTERN = re.compile(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)")
TEST_PATTERN = re.compile(r"^\s*def test_\w+\(", re.MULTILINE)
TEAM_HEADING = "## Team"
# Keys of a forensics trace row that the public bundle keeps (SCHEMA.md 2.7).
TRACE_KEYS = ("off", "ours", "gt")
# Smart Tagging fields of a forensics tag row that the public bundle keeps (SCHEMA.md 2.7).
TAG_KEYS = ("code", "team", "type")
# Sanity lists of frame-level rows over the full match. The public bundle keeps their counts only
# (SCHEMA.md 2.8, evaluation/results/README.md).
SANITY_DROP = (("gaps", "first"), ("set_pieces", "without_d"), ("d_block_end", "segments"))
# Decimals of an external score in external_summary.json (SCHEMA.md 2.11).
EXTERNAL_DECIMALS = 6
# The two views of an external match: the provider's own tracking, and the same tracking with
# the ball smoothed like this match's ball ("ATD-like"). SCHEMA.md 2.11.
EXTERNAL_VIEWS = (("own", "raw"), ("atd", "matched"))
# Match groups of the summary rows (SCHEMA.md 2.11).
EXTERNAL_GROUPS = ("metrica", "dfl", "all")
# Origin classes of the constants, in the order of the technical document (SCHEMA.md 2.9).
CONFIG_CLASSES = ("E", "F", "U", "D", "P")


class BuildError(RuntimeError):
    """The inputs cannot make a valid public bundle."""


# -- JSON --------------------------------------------------------------------------------------

def jsonable(value):
    """A copy of `value` with numpy types, tuples and non-string keys turned into JSON types.

    A float that is NaN or infinite becomes None.
    """
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else None
    return value


def serialize(obj):
    """UTF-8 bytes of `obj`: sorted keys, compact separators, one trailing newline."""
    text = json.dumps(jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)
    return (text + "\n").encode("utf-8")


# -- small helpers -------------------------------------------------------------------------------

def _team(value):
    """'H', 'A' or None."""
    return value if value in TEAMS else None


def _confidence(value):
    """A confidence rounded to CONFIDENCE_DECIMALS, or None."""
    if value is None:
        return None
    value = float(value)
    return round(value, CONFIDENCE_DECIMALS) if np.isfinite(value) else None


def graded_runs(blocks):
    """Half-open `[(start, stop)]` of the graded blocks `[(half, first, last_inclusive)]`."""
    return [(int(first), int(last) + 1) for _, first, last in blocks]


def public_window(gt_codes):
    """`(start, stop)` of the public window: the one stretch where the public GT is not X."""
    graded = (np.asarray(gt_codes) != X).astype(np.int8)
    edges = np.diff(np.concatenate(([0], graded, [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if len(starts) != 1:
        raise BuildError("the public GT has {} graded stretches, expected 1".format(len(starts)))
    return int(starts[0]), int(stops[0])


def fill_abstentions(codes, runs):
    """`codes` with X held from the previous label inside each run, as the scorer does."""
    out = np.asarray(codes).copy()
    for start, stop in runs:
        window = out[start:stop]
        known = np.flatnonzero(window != X)
        if not len(known):
            continue
        carry = np.maximum.accumulate(np.where(window != X, np.arange(len(window)), -1))
        carry[carry < 0] = known[0]
        out[start:stop] = window[carry]
    return out


def segment_starts(codes, runs, wanted):
    """Sorted start frames of the maximal runs of a code in `wanted`, inside each run."""
    starts = []
    for start, stop in runs:
        for first, _, letter in rle.encode(codes, start, stop):
            if rle.LETTERS.index(letter) in wanted:
                starts.append(first)
    return np.array(sorted(starts), dtype=np.int64)


def mask_runs(mask, lo=0, hi=None):
    """Half-open `[start, stop]` runs of True in `mask[lo:hi]`, in absolute frames."""
    mask = np.asarray(mask, dtype=bool)
    hi = len(mask) if hi is None else hi
    part = mask[lo:hi].astype(np.int8)
    edges = np.diff(np.concatenate(([0], part, [0])))
    return [[lo + int(a), lo + int(b)] for a, b in zip(np.flatnonzero(edges == 1),
                                                        np.flatnonzero(edges == -1))]


def _inside(frame, window):
    lo, hi = window
    return lo <= int(frame) < hi


# -- the bundle files ----------------------------------------------------------------------------

def labels_window(inputs, window):
    """labels_window.json (SCHEMA.md 2.2)."""
    lo, hi = window
    out = {"window": {"start": lo, "stop": hi},
           "ours": rle.encode(inputs["ours"], lo, hi),
           "gt_public": rle.encode(inputs["gt"], lo, hi)}
    for name in BASELINES:
        out[name] = rle.encode(inputs["baselines"][name], lo, hi)
    return out


def labels_baseline(name, codes):
    """labels_baseline_<name>.json (SCHEMA.md 2.3)."""
    return {"name": name, "n_frames": int(len(codes)), "runs": rle.encode(codes)}


def _count_by(rows, key_index, frames, start, stop):
    """{key: count} of the rows whose frame lies in [start, stop). `frames` is sorted."""
    lo, hi = np.searchsorted(frames, [start, stop], side="left")
    counts = {}
    for row in rows[lo:hi]:
        key = str(row[key_index])
        counts[key] = counts.get(key, 0) + 1
    return counts


def minute_bins(inputs):
    """minute_bins.json (SCHEMA.md 2.4 and Section 4)."""
    blocks = inputs["blocks"]
    runs = graded_runs(blocks)
    bins = []
    for half, first, last in blocks:
        stop = int(last) + 1
        for start in range(int(first), stop, BIN_FRAMES):
            bins.append((int(half), start, min(start + BIN_FRAMES, stop)))

    events = sorted(inputs["events"], key=lambda row: row[0])
    onsets = sorted(inputs["onsets"], key=lambda row: row[0])
    restarts = sorted([row for row in inputs["restarts"] if row[4]], key=lambda row: row[0])
    frames = {name: np.array([row[0] for row in rows], dtype=np.int64)
              for name, rows in (("events", events), ("onsets", onsets),
                                 ("restarts", restarts))}

    def rows_for(codes, with_detail):
        filled = fill_abstentions(codes, runs)
        spell_starts = segment_starts(filled, runs, (H, A))
        d_starts = segment_starts(filled, runs, (D,))
        out = []
        for half, start, stop in bins:
            part = filled[start:stop]
            n = float(stop - start)
            row = {"start": start, "stop": stop, "half": half,
                   "share_h": round(np.sum(part == H) / n, SHARE_DECIMALS),
                   "share_a": round(np.sum(part == A) / n, SHARE_DECIMALS),
                   "share_d": round(np.sum(part == D) / n, SHARE_DECIMALS),
                   "share_x": round(np.sum(np.asarray(codes)[start:stop] == X) / n,
                                    SHARE_DECIMALS),
                   "n_possession_starts": int(np.sum((spell_starts >= start) &
                                                     (spell_starts < stop))),
                   "n_d_segments": int(np.sum((d_starts >= start) & (d_starts < stop))),
                   "events_by_kind": {}, "onsets_by_mechanism": {}, "restarts_by_kind": {}}
            if with_detail:
                row["events_by_kind"] = _count_by(events, 1, frames["events"], start, stop)
                row["onsets_by_mechanism"] = _count_by(onsets, 1, frames["onsets"], start, stop)
                row["restarts_by_kind"] = _count_by(restarts, 2, frames["restarts"], start, stop)
            out.append(row)
        return out

    return {"bin_frames": BIN_FRAMES, "n_bins": len(bins),
            "ours": rows_for(inputs["ours"], True),
            REFERENCE_BASELINE: rows_for(inputs["baselines"][REFERENCE_BASELINE], False)}


def window_detail(inputs, window):
    """window_detail.json (SCHEMA.md 2.5)."""
    lo, hi = window
    return {
        "events": [[int(f), str(k), _team(t), _confidence(c)]
                   for f, k, t, c in inputs["events"] if _inside(f, window)],
        "owner_events": [[int(f), _team(t), None if k is None else str(k), _confidence(c)]
                         for f, t, k, c in inputs["owner_events"] if _inside(f, window)],
        "onsets": [[int(f), str(m), _confidence(c)]
                   for f, m, c in inputs["onsets"] if _inside(f, window)],
        "restarts": [[int(f), _team(t), str(k), _confidence(c), bool(ok)]
                     for f, t, k, c, ok in inputs["restarts"] if _inside(f, window)],
        "stoppages": mask_runs(inputs["stoppage"], lo, hi)}


def scores(reports):
    """scores.json (SCHEMA.md 2.6): whitelisted report keys, plus three derived keys."""
    out = {}
    for name in sorted(reports):
        report = reports[name]
        entry = {key: copy.deepcopy(report[key]) for key in SCORE_KEYS if key in report}
        entry["bootstrap"] = {"lo": report.get("spell_bootstrap_low"),
                              "hi": report.get("spell_bootstrap_high")}
        entry["n_spells"] = report.get("n_pred_spells")
        entry["duration_s"] = {"mean": report.get("mean_spell_duration_s"),
                               "gt_mean": report.get("gt_mean_spell_duration_s")}
        out[name] = entry
    return out


def transitions(forensics_result, events, window):
    """transitions.json (SCHEMA.md 2.7): forensics without kinematics, clipped to the window."""
    lo, hi = window
    half_width = int(forensics_result["window"])

    def trace(rows):
        return [{key: row[key] for key in TRACE_KEYS} for row in rows
                if _inside(row["frame"], window)]

    def tags(items):
        out = []
        for item in items:
            start, stop = max(lo, int(item["start_frame"])), min(hi, int(item["end_frame"]))
            if stop > start:
                row = {key: item.get(key) for key in TAG_KEYS}
                row.update(start_frame=start, end_frame=stop)
                out.append(row)
        return out

    def near(centre):
        return [[int(f), str(k), _team(t), _confidence(c)] for f, k, t, c in events
                if _inside(f, window) and abs(int(f) - centre) <= half_width]

    gt_rows = []
    for rec in forensics_result["gt_transitions"]:
        row = {key: rec[key] for key in ("gt_frame", "from", "to", "kind", "status", "our_frame",
                                         "our_from_to", "latency", "ball_missing_at_f_minus_1")}
        row.update(labels_trace=trace(rec["trace"]), tags=tags(rec["tags"]),
                   events=near(rec["gt_frame"]))
        gt_rows.append(row)
    spurious = []
    for rec in forensics_result["spurious"]:
        row = {key: rec[key] for key in ("our_frame", "from", "to", "gt_label_at_frame",
                                         "nearest_gt_transition")}
        row.update(labels_trace=trace(rec["trace"]), tags=tags(rec["tags"]),
                   events=near(rec["our_frame"]))
        spurious.append(row)
    kickoffs = [{key: value for key, value in row.items() if key != "mark_dist_m"}
                for row in forensics_result.get("kickoffs", []) if _inside(row["frame"], window)]
    return {"name": forensics_result["name"], "tol": forensics_result["tol"],
            "trace_half_width": half_width,
            "reappear_lookback": forensics_result["reappear_lookback"],
            "reports_stale": forensics_result["reports_stale"],
            "gt_transitions": gt_rows, "spurious": spurious,
            "latency_by_kind": copy.deepcopy(forensics_result["latency_by_kind"]),
            "kickoffs": kickoffs,
            "sandwiches": copy.deepcopy(forensics_result["sandwiches"]),
            "summary": copy.deepcopy(forensics_result["summary"])}


def sanity_public(stats):
    """A sanity_stats() dict without its lists of full-match frame rows (SANITY_DROP)."""
    out = copy.deepcopy(stats)
    for section, key in SANITY_DROP:
        if section in out and key in out[section]:
            del out[section][key]
    return out


def sanity_file(inputs):
    """sanity.json (SCHEMA.md 2.8)."""
    return {"ours": sanity_public(inputs["sanity"]["ours"]),
            REFERENCE_BASELINE: sanity_public(inputs["sanity"][REFERENCE_BASELINE]),
            "limits": inputs["limits"], "watched": inputs["watched"]}


def _r(value):
    """An external score rounded to EXTERNAL_DECIMALS, or None."""
    if value is None:
        return None
    return round(float(value), EXTERNAL_DECIMALS)


def _interval(record, mean_key):
    """`{mean, low, high}` of a paired difference record."""
    return {"mean": _r(record[mean_key]), "low": _r(record["low"]), "high": _r(record["high"])}


def external_summary(ext):
    """external_summary.json (SCHEMA.md 2.11): summary numbers of the external matches.

    `ext` is docs/technical/data/external.json. The result keeps the scores per match and per
    provider, the paired differences with their intervals, the component switch-offs, the oracle
    split and the sealed matches. It holds no frame, position or label. A match of the
    competition is refused.
    """
    games = set(ext["sealed"])
    for _view, key in EXTERNAL_VIEWS:
        games |= set(ext["open"][key]["matches"])
    bad = sorted(g for g in games if not g.startswith(("METRICA-", "DFL-")))
    if bad:
        raise BuildError("external numbers name a match that is not external: {}".format(bad))

    summary = {}
    for group in EXTERNAL_GROUPS:
        row = {}
        for view, key in EXTERNAL_VIEWS:
            src = ext["open"][key]["summary"][group]
            row[view] = {
                "engine_spell_f1": _r(src["engine_spell_f1"]),
                "engine_macro_f1": _r(src["engine_macro_f1"]),
                "baseline_spell_f1": _r(src["baseline50_spell_f1"]),
                "baseline_macro_f1": _r(src["baseline50_macro_f1"]),
                "engine_minus_baseline": _interval(src["engine_minus_baseline"], "mean_delta"),
                "recall_by_duration": [
                    {"bucket": b["bucket"], "n": int(b["n"]), "recalled": int(b["recalled"]),
                     "recall": _r(b["recall"])} for b in src["engine_recall_by_duration"]]}
        summary[group] = row

    matches = []
    for game in sorted(ext["open"]["raw"]["matches"]):
        row = {"game": game, "provider": "metrica" if game.startswith("METRICA-") else "dfl"}
        for view, key in EXTERNAL_VIEWS:
            src = ext["open"][key]["matches"][game]
            row[view] = {"engine_spell_f1": _r(src["engine"]["spell_f1"]),
                         "engine_macro_f1": _r(src["engine"]["macro_f1"]),
                         "baseline_spell_f1": _r(src["baseline50"]["spell_f1"]),
                         "engine_minus_baseline": _interval(src["engine_minus_baseline"],
                                                            "delta"),
                         "reference_possessions": int(src["reference_possessions"])}
        matches.append(row)

    sealed = []
    for game in sorted(ext["sealed"]):
        row = {"game": game}
        for view, key in EXTERNAL_VIEWS:
            src = ext["sealed"][game][key]
            row[view] = {"engine_spell_f1": _r(src["spell_f1"]),
                         "engine_macro_f1": _r(src["macro_f1"])}
        sealed.append(row)

    ablations = []
    for item in ext["ablations"]:
        row = {"field": item["field"], "group": item["group"], "label": item["label"]}
        for view, key in EXTERNAL_VIEWS:
            src = item[key]
            row[view] = {"mean": _r(src["mean_delta"]), "low": _r(src["low"]),
                         "high": _r(src["high"]), "metrica": _r(src["metrica"]),
                         "dfl": _r(src["dfl"])}
        ablations.append(row)

    oracle = {}
    for view, key in EXTERNAL_VIEWS:
        oracle[view] = {group: {"engine": _r(ext["oracle"][key][group]["ours"]),
                                "reference_teams": _r(ext["oracle"][key][group]["oracle_team"]),
                                "reference_dead_ball": _r(ext["oracle"][key][group]["oracle_dead"])}
                        for group in EXTERNAL_GROUPS}

    return {"summary": summary, "matches": matches, "sealed": sealed, "ablations": ablations,
            "oracle": oracle}


def config_with_classes(config, provenance):
    """config.json with the origin class of each field (SCHEMA.md 2.9). A pure function.

    `provenance` is docs/plan/calibration/provenance_current.json. A field that the inventory
    does not list gets the class None, and `classes.counts` does not count it.
    """
    out = copy.deepcopy(config)
    by_name = {row["name"]: row["class"] for row in provenance.get("fields", [])}
    counts = {name: 0 for name in CONFIG_CLASSES}
    for section in out["sections"]:
        for field in section["fields"]:
            cls = by_name.get(field["name"])
            field["class"] = cls
            if cls in counts:
                counts[cls] += 1
    out["classes"] = {"counts": counts,
                      "definitions": dict(provenance.get("class_definitions", {}))}
    return out


def manifest(inputs, window, files):
    """manifest.json (SCHEMA.md 2.1). `files` maps a file name to its size in bytes."""
    lo, hi = window
    return {"schema": SCHEMA_VERSION, "version": inputs["version"], "md5": inputs["md5"],
            "config_hash": inputs["config"]["config_hash"], "git": inputs["git"],
            "game": inputs["game"], "n_frames": int(inputs["n_frames"]),
            "fps": int(inputs["fps"]), "pitch_m": [float(v) for v in inputs["pitch_m"]],
            "team_names": dict(inputs["team_names"]),
            "graded_blocks": [[a, b] for a, b in graded_runs(inputs["blocks"])],
            "kickoffs": [{"half": int(h), "frame": int(f)}
                         for h, f in sorted(inputs["kickoffs"].items())],
            "public_window": {"start": lo, "stop": hi},
            "exempt": list(guard.DEFAULT_EXEMPT),
            "team": list(inputs["team"]), "repro": inputs["repro"], "files": files}


def assemble(inputs):
    """{file name: object} of the public bundle. A pure function of `inputs`.

    `inputs` is the dict of load_inputs(). Tests call this function on synthetic inputs.
    """
    window = public_window(inputs["gt"])
    if len(inputs["ours"]) != inputs["n_frames"]:
        raise BuildError("ours has {} frames, expected {}".format(len(inputs["ours"]),
                                                                  inputs["n_frames"]))
    bundle = {"labels_window.json": labels_window(inputs, window),
              "minute_bins.json": minute_bins(inputs),
              "window_detail.json": window_detail(inputs, window),
              "scores.json": scores(inputs["reports"]),
              "transitions.json": transitions(inputs["forensics"], inputs["events"], window),
              "sanity.json": sanity_file(inputs),
              "config.json": inputs["config"],
              "external_summary.json": external_summary(inputs["external"]),
              "ladder.json": inputs["ladder"],
              "cycles.json": inputs["cycles"]}
    for name in BASELINES:
        bundle["labels_{}.json".format(name)] = labels_baseline(name, inputs["baselines"][name])
    files = {name: len(serialize(obj)) for name, obj in sorted(bundle.items())}
    bundle["manifest.json"] = manifest(inputs, window, files)
    return bundle


# -- the writer ----------------------------------------------------------------------------------

def write_bundle(out_dir, bundle):
    """Guard every object of `bundle`, then write it to `out_dir`. Return {name: bytes}.

    Nothing is written if one object fails the guard. A stale JSON file in `out_dir` that the
    bundle does not hold is removed. The guard then checks the written folder once more.
    """
    if guard.MANIFEST not in bundle:
        raise BuildError("the bundle has no manifest.json")
    payloads = {name: serialize(obj) for name, obj in bundle.items()}
    manifest_obj = json.loads(payloads[guard.MANIFEST].decode("utf-8"))
    window, exempt, n_frames = guard.manifest_limits(manifest_obj)
    listed = set(manifest_obj["files"])
    if listed != set(bundle) - {guard.MANIFEST}:
        raise BuildError("manifest files {} differ from the bundle".format(sorted(listed)))
    for name in sorted(payloads):
        obj = json.loads(payloads[name].decode("utf-8"))
        guard.assert_no_coordinates(obj, name, window, exempt, n_frames)
        if name != guard.MANIFEST and len(payloads[name]) != manifest_obj["files"][name]:
            raise BuildError("{}: size differs from the manifest".format(name))

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for name in sorted(os.listdir(out_dir)):
        if name.endswith(".json") and name not in payloads:
            os.remove(os.path.join(out_dir, name))
    for name in sorted(payloads):
        with open(os.path.join(out_dir, name), "wb") as handle:
            handle.write(payloads[name])
    found = guard.check_dir(out_dir)
    if found:
        raise guard.LeakError("written bundle fails the guard:\n  " + "\n  ".join(found[:20]))
    return {name: len(payload) for name, payload in sorted(payloads.items())}


# -- loading -------------------------------------------------------------------------------------

def read_team(path=README_PATH):
    """Names of the "## Team" list of README.md, in order."""
    names, inside = [], False
    with open(path, "rb") as handle:
        lines = handle.read().decode("utf-8").splitlines()
    for line in lines:
        if line.strip() == TEAM_HEADING:
            inside = True
            continue
        if inside and line.startswith("## "):
            break
        if inside and line.startswith("- "):
            names.append(line[2:].strip())
    return names


def read_pins(root=ROOT):
    """{package: version} of every exact pin in REQUIREMENT_FILES."""
    pins = {}
    for name in REQUIREMENT_FILES:
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        with open(path) as handle:
            for line in handle:
                match = PIN_PATTERN.match(line.strip())
                if match:
                    pins[match.group(1)] = match.group(2)
    return pins


def count_tests(tests_dir=TESTS_DIR):
    """Number of test functions in tests/test_*.py. A parametrised test counts once."""
    total = 0
    for path in sorted(glob.glob(os.path.join(tests_dir, "test_*.py"))):
        with open(path, "rb") as handle:
            total += len(TEST_PATTERN.findall(handle.read().decode("utf-8")))
    return total


def read_json(path):
    """The JSON object of a repository file."""
    with open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def read_reports(names, reports_dir=REPORTS_DIR, warn=None):
    """{name: report dict} of the names that have a report. Warn about the others."""
    out = {}
    for name in names:
        path = os.path.join(reports_dir, name + ".json")
        if not os.path.isfile(path):
            if warn:
                warn("warning: no report for {}, skipped in scores.json".format(name))
            continue
        with open(path) as handle:
            out[name] = json.load(handle)
    return out


def _owner_event_rows(parts, cfg):
    """[(frame, team, kind, confidence)] of the owner events of the engine run."""
    from engine import possession

    table = parts["events"]
    rows = table[possession.owner_mask(table, cfg)]
    detail = {}
    for frame, kind, conf in zip(rows["frame"], rows["kind"], rows["confidence"]):
        detail.setdefault(int(frame), (str(kind), float(conf)))
    out = []
    for frame, team in parts["owner_events"]:
        kind, conf = detail.get(int(frame), (None, None))
        out.append((int(frame), team, kind, conf))
    return out


def table_rows(table, columns):
    """Rows of a DataFrame as tuples of plain Python values, in table order."""
    return [tuple(row) for row in zip(*[table[column].tolist() for column in columns])]


def warn_stderr(text):
    """Print a warning to stderr."""
    print(text, file=sys.stderr)


def load_inputs(version, results_path, game=None, allow_mismatch=False, reports_dir=REPORTS_DIR,
                warn=warn_stderr):
    """Everything assemble() needs, loaded from the repository, the data and one engine run."""
    from engine import data, possession, submission
    from engine.config import Config
    from tools import _common as cm, forensics, gt as gt_module, sanity
    from tools.viz import config_doc, gitinfo, ledger
    from tools.viz._engine_state import engine_state

    if cm.result_name(results_path) != version:
        raise BuildError("--results {} does not name version {}".format(results_path, version))
    if not os.path.isfile(os.path.join(reports_dir, version + ".json")):
        raise BuildError("no scorer report for {} in {}. Run python evaluation/run_evaluation.py"
                         .format(version, reports_dir))
    cfg = Config.default()
    game = data.resolve_game(game)
    codes, prepared, parts, md5 = engine_state(cfg, game, results_path, allow_mismatch)
    tactical = possession.engine_tactical(cfg, game)
    home, away = tactical["team_names"]

    baselines, baseline_md5 = {}, {}
    for name in SCORE_BASELINES:
        path = os.path.join(RESULTS_DIR, name + ".csv")
        if name in BASELINES:
            baselines[name] = cm.read_results(path, game)
        if os.path.isfile(path):
            baseline_md5[name] = submission.md5_of(path)

    experiments = ledger.parse_experiments(ledger.read(ledger.EXPERIMENTS_PATH))
    best = ledger.parse_current_best(ledger.read(ledger.PROGRESS_PATH))
    ladder = ledger.ladder(experiments, best, baseline_md5)
    names = [row["version"] for row in ladder["rows"] + ladder["rejected"]]
    names += [name for name in SCORE_BASELINES if name not in names]
    if version not in names:
        names.append(version)
    reports = read_reports(names, reports_dir, warn)

    tags = prepared["tagging"]
    ball_missing = np.isnan(data.load_tactical(game)["bx"])
    stats = {"ours": sanity.sanity_stats(codes, tags=tags, ball_missing=ball_missing, game=game),
             REFERENCE_BASELINE: sanity.sanity_stats(baselines[REFERENCE_BASELINE], tags=tags,
                                                     ball_missing=ball_missing, game=game)}
    result = forensics.analyse(results_path, tactical=tactical, tags=tags, game=game,
                               reports_dir=reports_dir)
    return {
        "version": version, "md5": md5, "game": game, "n_frames": int(prepared["n_frames"]),
        "fps": int(round(prepared["fps"])), "pitch_m": list(data.pitch_m(game)),
        "team_names": {"H": str(home), "A": str(away)},
        "blocks": prepared["blocks"], "kickoffs": prepared["kickoffs"],
        "ours": codes, "gt": gt_module.public_gt_codes(game), "baselines": baselines,
        "events": table_rows(parts["events"], ("frame", "kind", "team", "confidence")),
        "owner_events": _owner_event_rows(parts, cfg),
        "onsets": table_rows(parts["onsets"], ("frame", "mechanism", "confidence")),
        "restarts": table_rows(parts["restarts"],
                               ("frame", "team", "kind", "confidence", "confirmed")),
        "stoppage": np.asarray(parts["stoppage"], dtype=bool),
        "reports": reports, "forensics": result, "sanity": stats,
        "limits": {"possessions": [cfg.VETO_POSSESSIONS_MIN, cfg.VETO_POSSESSIONS_MAX],
                   "d_share": [cfg.VETO_D_SHARE_MIN, cfg.VETO_D_SHARE_MAX],
                   "min_d_segments": cfg.VETO_MIN_D_SEGMENTS,
                   "max_graded_x": cfg.VETO_MAX_GRADED_X},
        "watched": {"possessions": list(sanity.WATCH_POSSESSIONS),
                    "d_segments": list(sanity.WATCH_D_SEGMENTS),
                    "min_tag_agreement": sanity.WATCH_MIN_TAG_AGREEMENT},
        "config": config_with_classes(config_doc.document(cfg), read_json(PROVENANCE_PATH)),
        "external": read_json(EXTERNAL_PATH), "ladder": ladder,
        "cycles": ledger.cycles(ledger.read(ledger.LEARNINGS_PATH)),
        "git": gitinfo.head(), "team": read_team(),
        "repro": {"command": REPRO_COMMAND.format(version), "python": REPRO_PYTHON,
                  "pins": read_pins(), "n_tests": count_tests()}}


def build(args):
    """Load, assemble and write the public bundle. Return the process exit code."""
    try:
        inputs = load_inputs(args.version, args.results, args.game, args.allow_md5_mismatch)
        bundle = assemble(inputs)
        sizes = write_bundle(args.out, bundle)
    except (BuildError, guard.LeakError, ValueError, FileNotFoundError) as exc:
        print("build_site: {}".format(exc), file=sys.stderr)
        return 1
    total = sum(sizes.values())
    print("wrote {} files, {} bytes, to {}".format(len(sizes), total, args.out))
    print("version {}  md5 {}".format(inputs["version"], inputs["md5"]))
    return 0


def main(argv=None):
    """CLI entry point. Return 0 on success, 1 on a refused build, 2 on a usage error."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True, help="version name, for example c22_kink_contact")
    parser.add_argument("--results", required=True, help="results CSV of that version")
    parser.add_argument("--game", default=None, help="match of games.py (default: games.GAME)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output folder (default: %(default)s)")
    parser.add_argument("--allow-md5-mismatch", action="store_true",
                        help="accept a results file that the engine does not reproduce")
    return build(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
