"""Ablation runner for the Phase 5 freeze (card P5, step 2). Not part of the submission.

    python tools/ablate.py --out DIR [--spec FILE.json] [--names a,b] [--reference-only]
    python tools/ablate.py --out DIR --game METRICA-S3 --gt REFERENCE.csv
    python tools/ablate.py --list

The tool runs the engine once for the reference configuration and once for each ablation.
Then it scores the result directory with the official scorer and runs the full-match sanity
statistics of tools/sanity.py on each file. It prints one table row for each ablation and
writes DIR/ablations.md and DIR/ablations.json.

An ablation is a JSON object with these keys:

    name     lower-case letters, digits and "_", unique, not "reference"
    config   {FIELD: value} overrides of engine/config.py (the `--config` file of engine.run)
    patches  optional list of tool-level patches, for a component that has no config switch
    note     optional text for the table

`--spec` takes a JSON list of ablations. Without `--spec` the tool uses P5_ABLATIONS below.
An ablation without patches runs `python -m engine.run --config`, the official command. An
ablation with patches runs this file with `--child`. The child process applies the patches to
the engine modules in memory and then calls engine.run.main. The tool never edits engine/.

Patches:

    drop_onsets:<mechanism>   deadball.dead_onsets() returns no row of this mechanism
                              (out, pursuit_stop or collective_stop)
    no_throw_in_rule          deadball._is_throw_in_reappearance() is always False (C10 off)

`--game` runs the engine on another match of games.py (`engine.run --game`). `--gt` scores
against a reference CSV of the same length instead of the public ground truth, for example the
output of tools/events_gt.py on Metrica Sample Game 3. Never pass a reference made from the
competition match. Without the two options the tool runs BAR-ATH against the public ground
truth, as before.

DIR holds results/<name>.csv, configs/<name>.json, logs/<name>.log and score/, the scorer's
output with reports/<name>.json and reports/<name>_transitions.csv. Keep DIR outside the
repository or in a gitignored directory. Never commit the CSV files.

The table compares each ablation with the reference: the frames that change, the public
(or `--gt`) transitions that change at the loosest scorer tolerance, and the distance of each full-match
statistic to its PLAN.md 5.3 veto. The tool does not decide anything. A reviewer reads
the table and writes the ledger rows.

Deterministic: fixed order, no clock, no network, sorted JSON. Python 3.9.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

REFERENCE = "reference"
NAME_PATTERN = re.compile(r"^[a-z0-9_]+$")
ONSET_MECHANISMS = ("out", "pursuit_stop", "collective_stop")   # engine.deadball.MECHANISMS
PATCHES = tuple("drop_onsets:" + m for m in ONSET_MECHANISMS) + ("no_throw_in_rule",)
SCORER = os.path.join(ROOT, "evaluation", "run_evaluation.py")
TOLERANCE = 25          # frames, the loosest tolerance of evaluation/possession_metrics.TAUS

# PLAN.md 5.3 vetoes, the same values as tools/sanity.py. The half lines are watched only.
VETO_POSSESSIONS = (180, 450)
VETO_D_SHARE = (0.18, 0.55)
VETO_MIN_D_SEGMENTS = 55

# The P5 ablation list on the current best (c8_bump). Each entry turns one component off.
P5_ABLATIONS = [
    {"name": "c8_off_dv3", "config": {"CONTACT_DETECTOR": "dv3"},
     "note": "C8 off: dv_3 touch rows in place of the bump-centre contacts"},
    {"name": "c9_off", "config": {"GK_SWAP_R": 0.0},
     "note": "C9 off: no goalkeeper slot-swap guard"},
    {"name": "c11h_off", "config": {"RESTART_CHAIN_FRAMES": 0},
     "note": "C11 H off: no restart chain confirmation"},
    {"name": "c11k_off", "config": {"DELTA_RESTART_KICK": 0},
     "note": "C11 K off: no kick exit offset"},
    {"name": "c11_off", "config": {"RESTART_CHAIN_FRAMES": 0, "DELTA_RESTART_KICK": 0},
     "note": "C11 H and K off"},
    {"name": "c10_off", "config": {}, "patches": ["no_throw_in_rule"],
     "note": "C10 off (tool patch: OUT_INWARD_FRAMES has no off value)"},
    {"name": "p2_reappear_off", "config": {"DELTA_RESTART_REAPPEAR": 0},
     "note": "P2 off: no reappear exit offset"},
    {"name": "c5_dribble_off", "config": {"DRIBBLE_FRAMES": 1000000000},
     "note": "C5 off: no dribble rows (run length never reached)"},
    {"name": "c6_mark_off", "config": {"KICKOFF_TAKER_M": -1.0},
     "note": "C6 off: the centre mark rule never gives a team"},
    {"name": "c7_off", "config": {"OWNER_RELEASE_KINDS": ["release", "control", "dribble",
                                                          "touch"]},
     "note": "C7 off: a bare touch releases the owner again"},
    {"name": "onset_out_off", "config": {}, "patches": ["drop_onsets:out"],
     "note": "no out onsets, observed or missing-ball (tool patch)"},
    {"name": "onset_out_missing_off", "config": {"OUT_MAX_RANGE_M": -1.0},
     "note": "no missing-ball out onsets (config), observed outs stay"},
    {"name": "onset_pursuit_off", "config": {"PURSUIT_MIN_BALL_SPEED": 1000000.0},
     "note": "no pursuit_stop onsets (no ball is fast enough)"},
    {"name": "onset_collective_off", "config": {}, "patches": ["drop_onsets:collective_stop"],
     "note": "no collective_stop onsets (tool patch)"},
    {"name": "gap_stop_term_off", "config": {"W_GAP_STOP": 0.0},
     "note": "gap plus collective-stop evidence term off"},
    {"name": "tag_team_prior_off", "config": {"TAG_TEAM_PRIOR_RATIO": 1.0},
     "note": "Smart Tagging team phase prior off"},
    {"name": "tag_tail_off", "config": {"W_TAG_TAIL": 0.0},
     "note": "Smart Tagging set-piece tail evidence off"},
    {"name": "restart_prior_off", "config": {"RESTART_PRIOR_STEP": 0.0},
     "note": "restart confidence priors off (tags and out prior)"},
    {"name": "tags_all_off", "config": {"TAG_TEAM_PRIOR_RATIO": 1.0, "W_TAG_TAIL": 0.0,
                                        "RESTART_PRIOR_STEP": 0.0},
     "note": "every Smart Tagging prior off (GOALS window ends stay)"},
    {"name": "deflection_off", "config": {"DEFLECTION_MAX_HEADING_DEG": 0.0},
     "note": "deflection filter off (dv3 rows only)"},
    {"name": "redirect_off", "config": {"REDIRECT_FRACTION": 0.0},
     "note": "bump redirect rule off: every contact is a kick or a receive"},
    {"name": "flight_on", "config": {"FLIGHT_ASSIGNMENT": True},
     "note": "P2 flight assignment on"},
    {"name": "cheap_switch_off", "config": {"C_SWITCH_EVENT": 8.0},
     "note": "H<->A cheap window off (C_SWITCH_EVENT = C_SWITCH)"},
    {"name": "cheap_dead_off", "config": {"C_DEAD_EVENT": 6.0},
     "note": "alive->D cheap window off (C_DEAD_EVENT = C_DEAD)"},
    {"name": "cheap_alive_off", "config": {"C_ALIVE_EVENT": 6.0},
     "note": "D->alive cheap window off (C_ALIVE_EVENT = C_ALIVE, team bonus stays)"},
    {"name": "cheap_all_off", "config": {"C_SWITCH_EVENT": 8.0, "C_DEAD_EVENT": 6.0,
                                         "C_ALIVE_EVENT": 6.0},
     "note": "all three cheap windows off"},
    {"name": "min_poss_x2", "config": {"MIN_POSS": 20},
     "note": "MIN_POSS doubled (10 to 20 frames)"},
    {"name": "min_dead_x2", "config": {"MIN_DEAD": 50},
     "note": "MIN_DEAD doubled (25 to 50 frames)"},
    {"name": "min_both_x2", "config": {"MIN_POSS": 20, "MIN_DEAD": 50},
     "note": "MIN_POSS and MIN_DEAD doubled"},
]


# -- specs ----------------------------------------------------------------------------------------

def check_specs(specs):
    """Return the specs as a list of dicts with every key. Raise ValueError on a bad spec."""
    if not isinstance(specs, list):
        raise ValueError("the spec must be a JSON list of ablations")
    out, seen = [], set()
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("each ablation must be a JSON object: {!r}".format(spec))
        unknown = sorted(set(spec) - {"name", "config", "patches", "note"})
        if unknown:
            raise ValueError("unknown ablation keys {}".format(unknown))
        name = spec.get("name")
        if not isinstance(name, str) or not NAME_PATTERN.match(name) or name == REFERENCE:
            raise ValueError("bad ablation name {!r}".format(name))
        if name in seen:
            raise ValueError("duplicate ablation name {!r}".format(name))
        seen.add(name)
        config = spec.get("config", {})
        if not isinstance(config, dict):
            raise ValueError("{}: config must be a JSON object".format(name))
        patches = list(spec.get("patches", []))
        bad = [p for p in patches if p not in PATCHES]
        if bad:
            raise ValueError("{}: unknown patches {} (known: {})".format(name, bad, PATCHES))
        out.append({"name": name, "config": dict(config), "patches": patches,
                    "note": str(spec.get("note", ""))})
    return out


def check_config_fields(specs):
    """Raise ValueError if an override names a field that engine/config.py does not have."""
    from engine.config import Config

    known = set(Config.default().as_dict())
    for spec in specs:
        unknown = sorted(set(spec["config"]) - known)
        if unknown:
            raise ValueError("{}: unknown config fields {}".format(spec["name"], unknown))


# -- patches (child process only) -----------------------------------------------------------------

def drop_mechanism(dead_onsets, mechanism):
    """A wrapper of dead_onsets() that removes the rows of one onset mechanism."""
    def wrapped(*args, **kwargs):
        table = dead_onsets(*args, **kwargs)
        return table[table["mechanism"] != mechanism].reset_index(drop=True)
    return wrapped


def never_throw_in(*args, **kwargs):
    """Stand-in for deadball._is_throw_in_reappearance(): no reappearance is a throw-in."""
    return False


def apply_patches(patches, deadball):
    """Apply the named patches to the module `deadball`, in list order."""
    for patch in patches:
        if patch == "no_throw_in_rule":
            deadball._is_throw_in_reappearance = never_throw_in
        elif patch.startswith("drop_onsets:"):
            mechanism = patch.split(":", 1)[1]
            deadball.dead_onsets = drop_mechanism(deadball.dead_onsets, mechanism)
        else:
            raise ValueError("unknown patch {!r}".format(patch))


def game_args(game):
    """The `--game` arguments of engine.run: none for the default match (the P5 commands)."""
    return [] if game is None else ["--game", game]


def child(spec_path, out_csv, config_path, game=None):
    """Child process: apply the patches of one ablation, then run engine.run.main."""
    with open(spec_path) as handle:
        spec = check_specs([json.load(handle)])[0]
    from engine import deadball
    from engine import run as engine_run

    apply_patches(spec["patches"], deadball)
    return engine_run.main(["--out", out_csv, "--config", config_path] + game_args(game))


# -- running and scoring --------------------------------------------------------------------------

def _write_json(path, value):
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_one(spec, out_dir, python, game=None):
    """Run the engine for one spec on `game` (None: games.GAME). Return the path of its CSV."""
    for sub in ("results", "configs", "logs"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    name = spec["name"]
    out_csv = os.path.join(out_dir, "results", name + ".csv")
    config_path = os.path.join(out_dir, "configs", name + ".json")
    _write_json(config_path, spec["config"])
    if name == REFERENCE:
        command = [python, "-m", "engine.run", "--out", out_csv] + game_args(game)
    elif spec["patches"]:
        spec_path = os.path.join(out_dir, "configs", name + ".spec.json")
        _write_json(spec_path, spec)
        command = [python, os.path.abspath(__file__), "--child", spec_path, out_csv,
                   config_path] + game_args(game)
    else:
        command = ([python, "-m", "engine.run", "--out", out_csv, "--config", config_path]
                   + game_args(game))
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True)
    with open(os.path.join(out_dir, "logs", name + ".log"), "w") as handle:
        handle.write(" ".join(command) + "\n" + result.stdout)
    if result.returncode != 0:
        raise RuntimeError("{} failed:\n{}".format(name, result.stdout[-2000:]))
    return out_csv


def score(out_dir, python, gt=None):
    """Run the official scorer on DIR/results. Return the reports directory.

    `gt` None scores against the public ground truth. A path scores against that CSV
    (`run_evaluation.py --gt PATH`).
    """
    score_dir = os.path.join(out_dir, "score")
    os.makedirs(score_dir, exist_ok=True)
    command = [python, SCORER, "--results", os.path.join(out_dir, "results"),
               "--out", score_dir, "--no-detail"]
    if gt is not None:
        command += ["--gt", os.path.abspath(gt)]
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True)
    with open(os.path.join(out_dir, "logs", "score.log"), "w") as handle:
        handle.write(result.stdout)
    if result.returncode != 0:
        raise RuntimeError("the scorer skipped a file:\n{}".format(result.stdout[-2000:]))
    return os.path.join(score_dir, "reports")


# -- comparison -----------------------------------------------------------------------------------

def changed_stretches(codes, ref_codes):
    """(frames, stretches): the frames where two label vectors differ and their maximal runs."""
    diff = np.asarray(codes) != np.asarray(ref_codes)
    frames = int(diff.sum())
    if frames == 0:
        return 0, 0
    edges = np.diff(np.concatenate(([0], diff.astype(np.int8), [0])))
    return frames, int((edges == 1).sum())


def read_transitions(path):
    """{"matched": {gt_frame: latency}, "missed": set, "spurious": set of (frame, from, to)}."""
    out = {"matched": {}, "missed": set(), "spurious": set()}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            if row["status"] == "matched":
                out["matched"][int(row["gt_frame"])] = int(row["latency"])
            elif row["status"] == "missed":
                out["missed"].add(int(row["gt_frame"]))
            else:
                out["spurious"].add((int(row["pred_frame"]), row["pred_from"], row["pred_to"]))
    return out


def transition_changes(trans, ref):
    """The public transitions that differ from the reference, as lists of short strings."""
    lost = sorted(f for f in ref["matched"] if f not in trans["matched"])
    gained = sorted(f for f in trans["matched"] if f not in ref["matched"])
    moved = sorted(f for f in trans["matched"]
                   if f in ref["matched"] and trans["matched"][f] != ref["matched"][f])
    added = sorted(trans["spurious"] - ref["spurious"])
    removed = sorted(ref["spurious"] - trans["spurious"])
    return {
        "lost": ["{} ({:+d})".format(f, ref["matched"][f]) for f in lost],
        "gained": ["{} ({:+d})".format(f, trans["matched"][f]) for f in gained],
        "latency": ["{} {:+d}>{:+d}".format(f, ref["matched"][f], trans["matched"][f])
                    for f in moved],
        "spurious_added": ["{} {}>{}".format(*s) for s in added],
        "spurious_removed": ["{} {}>{}".format(*s) for s in removed],
    }


def veto_margins(row):
    """{statistic: distance to its nearest PLAN.md 5.3 veto edge}. Negative means violated."""
    p, d = row["possessions"][2], row["d_share"][2]
    return {"possessions": min(p - VETO_POSSESSIONS[0], VETO_POSSESSIONS[1] - p),
            "d_share": min(d - VETO_D_SHARE[0], VETO_D_SHARE[1] - d),
            "d_share_h1_line": VETO_D_SHARE[1] - row["d_share"][0],
            "d_share_h2_line": VETO_D_SHARE[1] - row["d_share"][1],
            "d_segments": row["d_segments"] - VETO_MIN_D_SEGMENTS}


def p5_flag(row, ref):
    """The statistics that the component pushes toward a veto, if it helps the public score.

    The row is the ablation, with the component off. The reference has the component on. The
    result is empty unless the reference has the higher public spellF1. Then it names each
    full-match statistic that sits nearer its veto edge in the reference than in the row. This
    is the literal drop test of card P5, step 2. It does not weigh the size of the move or the
    forensic evidence for the component. A reviewer does that.
    """
    if not ref["spell_f1"] > row["spell_f1"]:
        return []
    mine, theirs = veto_margins(row), veto_margins(ref)
    return sorted(key for key in mine if theirs[key] < mine[key])


def summarise(name, path, codes, ref_codes, report, trans, ref_trans, stats, spec):
    """One table row as a dict."""
    tol = report["transitions"][str(TOLERANCE)]
    halves, total = stats["halves"], stats["total"]
    order = [halves[1], halves[2], total]
    short = int(round((total["poss_share_lt_1s"] or 0.0) * total["possessions"]))
    frames, stretches = changed_stretches(codes, ref_codes)
    row = {
        "name": name,
        "note": spec.get("note", ""),
        "config": spec.get("config", {}),
        "patches": spec.get("patches", []),
        "md5": md5(path),
        "spell_f1": float(report["spell_f1_mean"]),
        "macro_f1": float(report["macro_f1"]),
        "matched": int(tol["n_matched"]),
        "missed": int(tol["n_missed"]),
        "spurious": int(tol["n_spurious"]),
        "possessions": [int(s["possessions"]) for s in order],
        "d_share": [float(s["d_share"]) for s in order],
        "d_segments": int(total["d_segments"]),
        "short_spells": short,
        "tag_agreement": stats["tags"]["total"]["agreement"] if "tags" in stats else None,
        "vetoes": "fail: " + ",".join(stats["violations"]) if stats["violations"] else "pass",
        "frames_changed": frames,
        "stretches_changed": stretches,
        "public": transition_changes(trans, ref_trans) if ref_trans is not None else None,
    }
    return row


def md5(path):
    """Hex md5 of a file (engine.submission.md5_of, without the engine import)."""
    import hashlib

    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# -- output ---------------------------------------------------------------------------------------

TABLE_HEADER = ("| name | md5 | spellF1 | macroF1 | m/mi/sp @25 | poss H1/H2/T | D % H1/H2/T "
                "| D segs | <1 s | tags % | vetoes | frames (stretches) changed |")


def table_line(row):
    """One markdown table line for a row."""
    return ("| {name} | {md5} | {sf:.3f} | {mf:.4f} | {m}/{mi}/{sp} | {p[0]}/{p[1]}/{p[2]} "
            "| {d[0]:.1f}/{d[1]:.1f}/{d[2]:.1f} | {segs} | {short} | {tags} | {veto} "
            "| {fr} ({st}) |").format(
                name=row["name"], md5=row["md5"][:8], sf=row["spell_f1"], mf=row["macro_f1"],
                m=row["matched"], mi=row["missed"], sp=row["spurious"], p=row["possessions"],
                d=[100.0 * v for v in row["d_share"]], segs=row["d_segments"],
                short=row["short_spells"],
                tags="-" if row["tag_agreement"] is None
                else "{:.1f}".format(100.0 * row["tag_agreement"]),
                veto=row["vetoes"], fr=row["frames_changed"], st=row["stretches_changed"])


def change_lines(row, ref, scope="public"):
    """Text lines: the scored transitions that change and the veto margins that move.

    `scope` names the ground truth in the text: "public", or "reference" for a `--gt` file.
    """
    lines = []
    public = row["public"] or {}
    parts = ["{} {}".format(key, ", ".join(values)) for key, values in sorted(public.items())
             if values]
    lines.append("- **{}**: {} {}".format(row["name"], scope, "; ".join(parts) if parts
                                          else "no transition change"))
    mine, theirs = veto_margins(row), veto_margins(ref)
    moves = []
    for key in sorted(mine):
        delta = mine[key] - theirs[key]
        if delta == 0:
            continue
        scale = 100.0 if key.startswith("d_share") else 1.0
        moves.append("{} margin {:+.{p}f} ({})".format(
            key, scale * delta, "nearer veto" if delta < 0 else "farther",
            p=1 if scale != 1.0 else 0))
    lines.append("  full match: {}".format("; ".join(moves) if moves else "no veto margin moves"))
    flagged = p5_flag(row, ref)
    if flagged:
        lines.append("  P5 check: the component raises the {} spellF1 by {:.3f} and moves "
                     "{} toward a veto".format(scope, ref["spell_f1"] - row["spell_f1"],
                                               ", ".join(flagged)))
    return lines


def render(rows, scope="public"):
    """The markdown report as a list of lines. rows[0] is the reference.

    `scope` names the ground truth: "public", or "reference" for a `--gt` file.
    """
    ref = rows[0]
    lines = [TABLE_HEADER, "|" + "---|" * 12]
    lines += [table_line(row) for row in rows]
    lines += ["", "{} transitions and veto margins against the reference "
                  "(lost / gained matches, latency changes, spurious added / removed):".format(
                      scope.capitalize()), ""]
    for row in rows[1:]:
        lines += change_lines(row, ref, scope)
    return lines


# -- main -----------------------------------------------------------------------------------------

def load_inputs(game=None):
    """(blocks, tags, ball_missing) of `game` for tools/sanity.py, loaded once."""
    from engine import data

    return (data.graded_blocks(game), data.load_tagging(game),
            np.isnan(data.load_tactical(game)["bx"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", help="output directory (outside git, or gitignored)")
    parser.add_argument("--spec", help="JSON list of ablations (default: P5_ABLATIONS)")
    parser.add_argument("--names", help="comma-separated subset of the ablation names")
    parser.add_argument("--python", default=sys.executable, help="interpreter for the runs")
    parser.add_argument("--list", action="store_true", help="print the ablations and stop")
    parser.add_argument("--game", default=None,
                        help="match of games.py for engine.run --game (default: games.GAME)")
    parser.add_argument("--gt", default=None,
                        help="reference CSV to score against (default: the public ground truth)")
    parser.add_argument("--child", nargs=3, metavar=("SPEC", "OUT_CSV", "CONFIG"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.child:
        return child(*args.child, game=args.game)

    if args.spec:
        with open(args.spec) as handle:
            specs = check_specs(json.load(handle))
    else:
        specs = check_specs(P5_ABLATIONS)
    if args.names:
        wanted = [n for n in args.names.split(",") if n]
        missing = sorted(set(wanted) - set(s["name"] for s in specs))
        if missing:
            raise ValueError("unknown ablation names {}".format(missing))
        specs = [s for s in specs if s["name"] in wanted]
    check_config_fields(specs)
    if args.list:
        for spec in specs:
            print("{:<24} {} {} {}".format(spec["name"], json.dumps(spec["config"],
                                                                     sort_keys=True),
                                           spec["patches"] or "", spec["note"]))
        return 0
    if not args.out:
        parser.error("--out is required")
    if args.game is not None:
        import games

        if args.game not in games.GAMES:
            parser.error("no such game {!r}; have {}".format(args.game, sorted(games.GAMES)))
    if args.gt is not None and not os.path.isfile(args.gt):
        parser.error("no such reference CSV {}".format(args.gt))

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    reference = {"name": REFERENCE, "config": {}, "patches": [], "note": "default config"}
    every = [reference] + specs
    paths = {}
    for spec in every:
        print("run {}".format(spec["name"]), flush=True)
        paths[spec["name"]] = run_one(spec, out_dir, args.python, args.game)
    reports_dir = score(out_dir, args.python, args.gt)

    from tools import sanity
    import _common as cm

    blocks, tags, ball_missing = load_inputs(args.game)
    ref_codes = cm.read_results(paths[REFERENCE], args.game)
    ref_trans = read_transitions(os.path.join(reports_dir, REFERENCE + "_transitions.csv"))
    rows = []
    for spec in every:
        name = spec["name"]
        codes = cm.read_results(paths[name], args.game)
        with open(os.path.join(reports_dir, name + ".json")) as handle:
            report = json.load(handle)
        trans = read_transitions(os.path.join(reports_dir, name + "_transitions.csv"))
        stats = sanity.sanity_stats(codes, tags=tags, ball_missing=ball_missing, blocks=blocks,
                                    game=args.game)
        rows.append(summarise(name, paths[name], codes, ref_codes, report, trans,
                              None if name == REFERENCE else ref_trans, stats, spec))

    lines = render(rows, "public" if args.gt is None else "reference")
    with open(os.path.join(out_dir, "ablations.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    _write_json(os.path.join(out_dir, "ablations.json"), rows)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
