"""Cross-match paired summary of two versions from tools/score_game.py reports. Not part of the submission.

    python tools/paired_summary.py --a A.json [A.json ...] --b B.json [B.json ...]
                                   [--draws 1000] [--seed 0]

Each argument is a JSON report of tools/score_game.py or a folder of them. A folder gives every
`*.json` file in it, in sorted order. The tool pairs the reports of version A and version B by
their "ground_truth" field, the file name of the reference. Each reference must appear once on
each side, and both reports of a pair must have the same number of reference spells.

For each match the tool prints spellF1 of A and B and the difference A - B. It then prints the
mean of the per-match differences, so that each match counts once whatever its possession count.
The interval comes from a paired bootstrap with `--draws` draws and seed `--seed`. In each draw
the tool resamples the reference possessions of each match once, scores both versions on the
same resample with the method of possession_metrics.bootstrap_spell_f1(), and takes the mean of
the per-match differences. The matches are drawn in sorted order of their reference names. The
intervals are the 5 and 95 percentiles of the draws.

The tool reads the "spell_match_flags" and "n_pred_spells" fields of each report. A report
without "spell_match_flags" comes from an older score_game. Score that file again.

Deterministic: sorted inputs, a fixed seed. Python 3.9, numpy only.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALUATION = os.path.join(ROOT, "evaluation")
if EVALUATION not in sys.path:
    sys.path.insert(0, EVALUATION)
import possession_metrics as pm  # noqa: E402

DEFAULT_DRAWS = 1000
DEFAULT_SEED = pm.BOOTSTRAP_SEED


def expand(paths):
    """The JSON report paths: a file as given, a folder as its sorted `*.json` files.

    The `_sanity.json` files of tools/sanity.py are left out.
    """
    out = []
    for path in paths:
        if os.path.isdir(path):
            out += [p for p in sorted(glob.glob(os.path.join(path, "*.json")))
                    if not p.endswith("_sanity.json")]
        else:
            out.append(path)
    return out


def load(path):
    """(reference name, flags (5, n_gt) int8, n_pred, report name) of one report."""
    with open(path) as handle:
        report = json.load(handle)
    if "spell_match_flags" not in report:
        raise ValueError("{}: no spell_match_flags. Score the file again with "
                         "tools/score_game.py".format(path))
    rows = [report["spell_match_flags"]["{:.1f}".format(iou)] for iou in pm.SPELL_IOUS]
    flags = np.asarray(rows, dtype=np.int8).reshape(len(pm.SPELL_IOUS), -1)
    return report["ground_truth"], flags, int(report["n_pred_spells"]), report["submission"]


def by_reference(paths):
    """{reference name: load() result} for one version. A reference must appear once."""
    out = {}
    for path in expand(paths):
        item = load(path)
        if item[0] in out:
            raise ValueError("reference {} appears twice in one version".format(item[0]))
        out[item[0]] = item
    return out


def mean_f1(flags, n_pred, picks):
    """spellF1 of one resample of the reference spells, as possession_metrics scores it.

    Identical to score_game._mean_f1 and to the draw of possession_metrics.bootstrap_spell_f1.
    """
    n_gt = len(picks)
    f1s = []
    for row in flags:
        true_positive = int(row[picks].sum())
        false_positive = max(0, n_pred - true_positive)
        false_negative = n_gt - true_positive
        f1s.append(2.0 * true_positive / (2 * true_positive + false_positive + false_negative)
                   if true_positive else 0.0)
    return float(np.mean(f1s))


def paired_summary(side_a, side_b, n_draws=DEFAULT_DRAWS, seed=DEFAULT_SEED):
    """The cross-match summary of two by_reference() dicts, as a dict.

    Returns {"matches": [{reference, a, b, spell_a, spell_b, delta, low, high}], "mean_delta",
    "low", "high", "share_positive", "draws", "seed"}. A match's low and high are the 5 and 95
    percentiles of its own paired draws, taken from the same draws as the mean.
    """
    if sorted(side_a) != sorted(side_b):
        raise ValueError("the two versions have different references: {} and {}".format(
            sorted(side_a), sorted(side_b)))
    names = sorted(side_a)
    if not names:
        raise ValueError("no report to compare")
    for name in names:
        if side_a[name][1].shape != side_b[name][1].shape:
            raise ValueError("{}: the two reports have different reference spell counts"
                             .format(name))
    rng = np.random.RandomState(seed)
    per_match = np.empty((n_draws, len(names)), dtype=np.float64)
    for draw in range(n_draws):
        for col, name in enumerate(names):
            _, flags_a, n_a, _ = side_a[name]
            _, flags_b, n_b, _ = side_b[name]
            n_gt = flags_a.shape[1]
            picks = rng.randint(0, n_gt, n_gt)
            per_match[draw, col] = mean_f1(flags_a, n_a, picks) - mean_f1(flags_b, n_b, picks)
    matches = []
    for col, name in enumerate(names):
        _, flags_a, n_a, sub_a = side_a[name]
        _, flags_b, n_b, sub_b = side_b[name]
        every = np.arange(flags_a.shape[1])
        spell_a, spell_b = mean_f1(flags_a, n_a, every), mean_f1(flags_b, n_b, every)
        low, high = (np.percentile(per_match[:, col], pm.BOOTSTRAP_PERCENTILES)
                     if n_draws else (None, None))
        matches.append({"reference": name, "a": sub_a, "b": sub_b, "spell_a": spell_a,
                        "spell_b": spell_b, "delta": spell_a - spell_b,
                        "low": None if low is None else float(low),
                        "high": None if high is None else float(high)})
    means = per_match.mean(axis=1)
    low, high = (np.percentile(means, pm.BOOTSTRAP_PERCENTILES) if n_draws else (None, None))
    return {"matches": matches,
            "mean_delta": float(np.mean([m["delta"] for m in matches])),
            "low": None if low is None else float(low),
            "high": None if high is None else float(high),
            "share_positive": float(np.mean(means > 0)) if n_draws else None,
            "draws": int(n_draws), "seed": int(seed)}


def _interval(low, high):
    if low is None:
        return "no draws"
    return "paired 5-95% {:+.4f} .. {:+.4f}".format(low, high)


def format_lines(summary):
    """Printable lines of a paired_summary() result."""
    lines = ["{:<32} {:>8} {:>8} {:>8}  {}".format("reference", "A", "B", "A - B",
                                                  "interval")]
    for m in summary["matches"]:
        lines.append("{:<32} {:>8.4f} {:>8.4f} {:>+8.4f}  {}".format(
            m["reference"], m["spell_a"], m["spell_b"], m["delta"],
            _interval(m["low"], m["high"])))
    lines.append("mean of {} per-match differences: {:+.4f}, {}, A higher in {} of {} draws "
                 "(seed {})".format(
                     len(summary["matches"]), summary["mean_delta"],
                     _interval(summary["low"], summary["high"]),
                     "-" if summary["share_positive"] is None
                     else "{:.1%}".format(summary["share_positive"]),
                     summary["draws"], summary["seed"]))
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", nargs="+", required=True, help="reports or folders of version A")
    parser.add_argument("--b", nargs="+", required=True, help="reports or folders of version B")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS,
                        help="paired bootstrap draws (default: {})".format(DEFAULT_DRAWS))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="bootstrap seed (default: {})".format(DEFAULT_SEED))
    args = parser.parse_args(argv)
    summary = paired_summary(by_reference(args.a), by_reference(args.b), args.draws, args.seed)
    print("\n".join(format_lines(summary)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
