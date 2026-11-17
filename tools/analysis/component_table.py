"""Per-component table of the P5 ablations across several references (P9-A3). Not part of the submission.

    python tools/analysis/component_table.py --paired DIR --games G1 G2 ... [--public GAME]
                                             [--ablate DIR] [--draws 1000] [--seed 0]
                                             [--out-md FILE] [--out-json FILE]

Inputs. For each game, DIR/<game>/ holds the reports of tools/score_game.py: `reference.json`
(the reference run scored alone) and `<ablation>.json` (the ablation scored with
`--versus reference.csv`). The reports of one game must use one reference CSV. `--ablate`
is the folder of tools/ablate.py runs. When it holds `<game>/ablations.json`, the table adds the
frames that each ablation changes.

For each ablation and each game the tool reads the paired spellF1 difference (ablation minus
reference) and its 5 to 95 % interval from the `versus` field of the report. It adds the macroF1
difference and the change of the matched, missed and spurious transitions at the loosest
tolerance. It then calls tools/paired_summary.paired_summary() twice for each ablation: once
over the games of `--games`, and once over those games plus `--public`. The summary is the mean
of the per-match differences with a paired bootstrap interval.

Sign convention: delta = ablation minus reference. A negative delta means that the component the
ablation removes helps on that reference. `flight_on` adds a component, so a negative delta there
means that the shipped choice (off) helps.

The tool prints a Markdown report or writes it to `--out-md`, and writes the numbers to
`--out-json`. It reads only score_game reports and ablate.py summaries. It runs no engine and
reads no ground truth file. Deterministic: sorted inputs, a fixed seed. Python 3.9, numpy only.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import paired_summary as ps  # noqa: E402

REFERENCE = "reference"
DEFAULT_DRAWS = ps.DEFAULT_DRAWS
DEFAULT_SEED = ps.DEFAULT_SEED


def ablation_order(names):
    """The names in the order of tools/ablate.P5_ABLATIONS, unknown names last and sorted."""
    try:
        import ablate
        known = [spec["name"] for spec in ablate.P5_ABLATIONS]
    except ImportError:          # pragma: no cover - ablate.py ships with the repository
        known = []
    first = [n for n in known if n in names]
    return first + sorted(n for n in names if n not in known)


def report_paths(paired_dir, game):
    """{name: path} of the score_game reports of one game (sanity files left out)."""
    folder = os.path.join(paired_dir, game)
    out = {}
    for entry in sorted(os.listdir(folder)):
        if entry.endswith(".json") and not entry.endswith("_sanity.json"):
            out[entry[:-len(".json")]] = os.path.join(folder, entry)
    if REFERENCE not in out:
        raise ValueError("{}: no {}.json".format(folder, REFERENCE))
    return out


def _read(path):
    with open(path) as handle:
        return json.load(handle)


def frames_changed(ablate_dir, game):
    """{name: (frames, stretches)} from <ablate_dir>/<game>/ablations.json, or {}."""
    if not ablate_dir:
        return {}
    path = os.path.join(ablate_dir, game, "ablations.json")
    if not os.path.exists(path):
        return {}
    return {row["name"]: (int(row["frames_changed"]), int(row["stretches_changed"]))
            for row in _read(path)}


def match_row(report, ref_report):
    """The per-match numbers of one ablation against the reference, as a dict.

    Raises ValueError when the report has no paired `versus` field or the two reports use
    different reference files.
    """
    if report["ground_truth"] != ref_report["ground_truth"]:
        raise ValueError("{} and {} use different references".format(
            report["submission"], ref_report["submission"]))
    head, ref = report["summary"], ref_report["summary"]
    if "versus" not in head:
        raise ValueError("{}: no paired versus field".format(report["submission"]))
    pair = head["versus"]
    return {"spell_f1": float(head["spell_f1_mean"]), "ref_spell_f1": float(ref["spell_f1_mean"]),
            "delta": float(pair["delta"]), "low": float(pair["low"]),
            "high": float(pair["high"]), "share_positive": float(pair["share_positive"]),
            "d_macro": float(head["macro_f1"]) - float(ref["macro_f1"]),
            "d_matched": int(head["matched"]) - int(ref["matched"]),
            "d_missed": int(head["missed"]) - int(ref["missed"]),
            "d_spurious": int(head["spurious"]) - int(ref["spurious"])}


def cross(paths_by_game, games, name, n_draws, seed):
    """paired_summary() of ablation `name` against the reference over `games`."""
    side_a = {}
    side_b = {}
    for game in games:
        item_a = ps.load(paths_by_game[game][name])
        item_b = ps.load(paths_by_game[game][REFERENCE])
        side_a[game + "/" + item_a[0]] = item_a
        side_b[game + "/" + item_b[0]] = item_b
    return ps.paired_summary(side_a, side_b, n_draws, seed)


def build(paired_dir, games, public=None, ablate_dir=None, n_draws=DEFAULT_DRAWS,
          seed=DEFAULT_SEED):
    """The whole table as a dict: {"games", "public", "ablations": [...]}."""
    every = list(games) + ([public] if public else [])
    paths = {g: report_paths(paired_dir, g) for g in every}
    names = set.intersection(*[set(p) - {REFERENCE} for p in paths.values()])
    rows = []
    for name in ablation_order(names):
        per = {}
        for game in every:
            row = match_row(_read(paths[game][name]), _read(paths[game][REFERENCE]))
            changed = frames_changed(ablate_dir, game).get(name)
            row["frames_changed"] = None if changed is None else changed[0]
            row["stretches_changed"] = None if changed is None else changed[1]
            per[game] = row
        external = cross(paths, games, name, n_draws, seed)
        entry = {"name": name, "per_match": per,
                 "external": _strip(external),
                 "n_better": sum(1 for g in games if per[g]["low"] > 0),
                 "n_worse": sum(1 for g in games if per[g]["high"] < 0),
                 "n_positive": sum(1 for g in games if per[g]["delta"] > 0),
                 "n_negative": sum(1 for g in games if per[g]["delta"] < 0)}
        if public:
            entry["with_public"] = _strip(cross(paths, every, name, n_draws, seed))
        rows.append(entry)
    refs = {g: _read(paths[g][REFERENCE])["summary"]["spell_f1_mean"] for g in every}
    return {"games": list(games), "public": public, "reference_spell_f1": refs,
            "draws": int(n_draws), "seed": int(seed), "ablations": rows}


def _strip(summary):
    return {k: summary[k] for k in ("mean_delta", "low", "high", "share_positive")}


def _iv(low, high):
    return "[{:+.3f}, {:+.3f}]".format(low, high)


def short(game):
    """A short column name: DFL-J03WMX -> WMX, METRICA-S1 -> S1, BAR-ATH -> public."""
    if game.startswith("DFL-J03"):
        return game[len("DFL-J03"):]
    if game.startswith("METRICA-"):
        return game[len("METRICA-"):]
    if game == "BAR-ATH":
        return "public"
    return game


def render(table):
    """Markdown lines of a build() result."""
    games, public = table["games"], table["public"]
    every = games + ([public] if public else [])
    lines = []
    lines.append("### Cross-match summary (delta = ablation minus reference, spellF1)")
    lines.append("")
    head = ["ablation", "mean Δ, {} external [5–95 %]".format(len(games)),
            "share > 0", "n Δ>0 / Δ<0", "n interval > 0 / < 0"]
    if public:
        head += ["public Δ [5–95 %]", "mean Δ, {} + public [5–95 %]".format(len(games))]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    for row in table["ablations"]:
        ext = row["external"]
        cells = [row["name"], "{:+.3f} {}".format(ext["mean_delta"], _iv(ext["low"], ext["high"])),
                 ("—" if ext["low"] == ext["high"] == 0.0
                  else "{:.0%}".format(ext["share_positive"])),
                 "{} / {}".format(row["n_positive"], row["n_negative"]),
                 "{} / {}".format(row["n_better"], row["n_worse"])]
        if public:
            pub = row["per_match"][public]
            wp = row["with_public"]
            cells += ["{:+.3f} {}".format(pub["delta"], _iv(pub["low"], pub["high"])),
                      "{:+.3f} {}".format(wp["mean_delta"], _iv(wp["low"], wp["high"]))]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("### spellF1 delta per reference (ablation minus reference)")
    lines.append("")
    head = ["ablation"] + [short(g) for g in every]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    refs = table["reference_spell_f1"]
    lines.append("| reference spellF1 | " + " | ".join("{:.4f}".format(refs[g]) for g in every)
                 + " |")
    for row in table["ablations"]:
        cells = [row["name"]]
        for g in every:
            m = row["per_match"][g]
            mark = ""
            if m["low"] > 0:
                mark = " ▲"
            elif m["high"] < 0:
                mark = " ▼"
            cells.append("{:+.3f}{}".format(m["delta"], mark))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("▲: the 5–95 % interval lies above zero (removing the component helps). "
                 "▼: below zero (the component helps).")
    lines.append("")
    lines.append("### Per reference: interval, macroF1 delta, transitions, frames changed")
    lines.append("")
    for g in every:
        lines.append("#### {} (reference spellF1 {:.4f})".format(g, refs[g]))
        lines.append("")
        lines.append("| ablation | spellF1 | Δ [5–95 %] | draws Δ > 0 | ΔmacroF1 | "
                     "Δ matched / missed / spurious @25 | frames (stretches) changed |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in table["ablations"]:
            m = row["per_match"][g]
            frames = ("—" if m["frames_changed"] is None
                      else "{:,} ({})".format(m["frames_changed"], m["stretches_changed"]))
            lines.append("| {} | {:.4f} | {:+.4f} {} | {:.0%} | {:+.4f} | {:+d} / {:+d} / {:+d} "
                         "| {} |".format(row["name"], m["spell_f1"], m["delta"],
                                         _iv(m["low"], m["high"]), m["share_positive"],
                                         m["d_macro"], m["d_matched"], m["d_missed"],
                                         m["d_spurious"], frames))
        lines.append("")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paired", required=True, help="folder of <game>/<name>.json reports")
    parser.add_argument("--games", nargs="+", required=True, help="the external references")
    parser.add_argument("--public", default=None, help="one more game, shown apart")
    parser.add_argument("--ablate", default=None, help="folder of tools/ablate.py runs")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args(argv)
    table = build(args.paired, args.games, args.public, args.ablate, args.draws, args.seed)
    text = "\n".join(render(table)) + "\n"
    if args.out_md:
        with open(args.out_md, "w") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)
    if args.out_json:
        with open(args.out_json, "w") as handle:
            json.dump(table, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
