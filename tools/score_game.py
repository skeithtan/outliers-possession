"""Score one results CSV against one reference CSV of the same length. Not part of the submission.

    python tools/score_game.py RESULTS.csv --gt REFERENCE.csv [--out-dir DIR] [--name NAME]

The tool scores with the official metric, evaluation/possession_metrics.evaluate(). The graded
frames are the frames where the reference is not X. The tool prints spellF1 for each IoU and the
mean, macroF1, the mean latency and the matched, missed and spurious transitions at the loosest
tolerance. It also prints the possession count, the mean possession duration, the D share and
the D segment count for both files, on the graded frames of the reference.

`--versus OTHER.csv` adds a paired bootstrap of the spellF1 difference, results minus OTHER.
Each draw resamples the possessions of the reference once and scores both files on the same
draw, with the method of possession_metrics.bootstrap_spell_f1(). The seed is 0.

The tool also prints the failure-class tables of tools/failure_classes.py and writes them under
"failure_classes" in the summary of the JSON report:

- The recall of the reference spells by duration (< 2 s, 2 to 5 s, > 5 s). A reference spell
  is recalled when the scorer's one-to-one spell matching matches it at IoU 0.5 or more.
- The predicted possessions over the reference possessions.
- D entry and exit precision, recall and signed latency. H and A are one live state, and the
  edges match one to one within the loosest tolerance.
- The per-class decomposition of RESEARCH-2 M19: the class of each reference spell, the
  unmatched predicted spells, the transition alignment by kind and the group edit gains.

The JSON report also holds "spell_match_flags": for each IoU threshold, 1 where the reference
spell is matched. tools/paired_summary.py reads these flags for its cross-match summary.

The tool writes DIR/<name>.json and DIR/<name>_transitions.csv with the helpers of
evaluation/run_evaluation.py. The default DIR is a folder under /tmp. Never write to
evaluation/reports/. That folder is for the public ground truth only.

Use the tool for a reference from another match, for example tools/events_gt.py on Metrica
Sample Game 3. Never use it with a reference made from the competition match.

Deterministic: a fixed bootstrap seed, sorted JSON. Python 3.9.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALUATION = os.path.join(ROOT, "evaluation")
for _path in (ROOT, EVALUATION):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import possession_metrics as pm  # noqa: E402
import run_evaluation as rev  # noqa: E402

TOOLS = os.path.dirname(os.path.abspath(__file__))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
import failure_classes as fc  # noqa: E402

D = pm.LABELS.index("D")
# The default number of paired bootstrap draws (run_evaluation.py suggests 1000).
DEFAULT_DRAWS = 1000
# Outside the repository, so that no report lands in evaluation/reports/ by accident.
DEFAULT_OUT_DIR = os.path.join("/tmp", "score_game")


def label_stats(codes, runs, fps=pm.FPS):
    """Possessions, mean duration (s), D share and D segments of codes on the graded runs.

    X is forward-filled inside each run first, as the scorer does for its temporal metrics.
    """
    filled = pm.fill_abstentions(np.asarray(codes, dtype=np.int8), runs)
    spells = pm.possession_spells(filled, runs)
    lengths = [b - a for a, b, _ in spells]
    graded = sum(b - a for a, b in runs)
    dead = sum(int(np.sum(filled[a:b] == D)) for a, b in runs)
    return {
        "possessions": len(spells),
        "mean_duration_s": float(np.mean(lengths)) / fps if lengths else 0.0,
        "d_share": dead / float(graded) if graded else 0.0,
        "d_segments": sum(1 for s in pm.segments(filled, runs) if s[2] == D),
    }


def _spell_flags(gt_codes, pred_codes, runs):
    """(flags, n_pred): matched-or-not per reference possession per IoU, and the spell count."""
    gt_spells = pm.possession_spells(gt_codes, runs)
    pred_spells = pm.possession_spells(pm.fill_abstentions(pred_codes, runs), runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    flags = np.vstack([pm.matched_gt_flags(overlaps, len(gt_spells), threshold)
                       for threshold in pm.SPELL_IOUS])
    return flags, len(pred_spells)


def _mean_f1(flags, n_pred, picks):
    """spellF1 (mean over IoU) of one resample, as in possession_metrics.bootstrap_spell_f1."""
    n_gt = len(picks)
    f1s = []
    for row in flags:
        true_positive = int(row[picks].sum())
        false_positive = max(0, n_pred - true_positive)
        false_negative = n_gt - true_positive
        f1s.append(2.0 * true_positive / (2 * true_positive + false_positive + false_negative)
                   if true_positive else 0.0)
    return float(np.mean(f1s))


def paired_delta(gt_codes, codes_a, codes_b, n_draws=DEFAULT_DRAWS, seed=pm.BOOTSTRAP_SEED):
    """Paired bootstrap of spellF1(a) - spellF1(b) over the reference possessions.

    Returns {"delta", "low", "high", "share_positive", "draws"}. "delta" is the difference on the
    full reference. "low" and "high" are the 5 and 95 percentiles of the resampled difference.
    "share_positive" is the share of draws where a scores higher than b.
    """
    gt_codes = np.asarray(gt_codes, dtype=np.int8)
    runs = pm.valid_runs(gt_codes)
    flags_a, n_a = _spell_flags(gt_codes, np.asarray(codes_a, dtype=np.int8), runs)
    flags_b, n_b = _spell_flags(gt_codes, np.asarray(codes_b, dtype=np.int8), runs)
    n_gt = flags_a.shape[1]
    every = np.arange(n_gt)
    delta = _mean_f1(flags_a, n_a, every) - _mean_f1(flags_b, n_b, every)
    rng = np.random.RandomState(seed)
    draws = np.empty(n_draws, dtype=np.float64)
    for k in range(n_draws):
        picks = rng.randint(0, n_gt, n_gt)
        draws[k] = _mean_f1(flags_a, n_a, picks) - _mean_f1(flags_b, n_b, picks)
    low, high = np.percentile(draws, pm.BOOTSTRAP_PERCENTILES) if n_draws else (None, None)
    return {"delta": delta, "low": None if low is None else float(low),
            "high": None if high is None else float(high),
            "share_positive": float(np.mean(draws > 0)) if n_draws else None,
            "draws": int(n_draws)}


def score_codes(gt_codes, pred_codes, taus=pm.TAUS):
    """(result, gt_stats, pred_stats) for two code vectors of equal length."""
    gt_codes = np.asarray(gt_codes, dtype=np.int8)
    pred_codes = np.asarray(pred_codes, dtype=np.int8)
    result = pm.evaluate(gt_codes, pred_codes, taus=taus, bootstrap_n=0)
    runs = pm.valid_runs(gt_codes)
    return result, label_stats(gt_codes, runs), label_stats(pred_codes, runs)


def summary(result, gt_stats, pred_stats, taus=pm.TAUS):
    """The headline numbers as a flat dict (the JSON report keeps everything else)."""
    loose = result["transitions"][max(taus)]
    return {
        "spell_f1_mean": result["spell_f1_mean"],
        "spell_f1": {"{:.1f}".format(iou): result["spell_f1"][iou]["f1"]
                     for iou in pm.SPELL_IOUS},
        "macro_f1": result["macro_f1"],
        "tau": max(taus),
        "mean_latency": loose["mean_latency"],
        "matched": loose["n_matched"],
        "missed": loose["n_missed"],
        "spurious": loose["n_spurious"],
        "reference": gt_stats,
        "results": pred_stats,
    }


def failure_report(gt_codes, pred_codes, gt_stats, pred_stats, fps=pm.FPS, taus=pm.TAUS):
    """The failure-class tables of tools/failure_classes.py for one scored file, as a dict."""
    gt_codes = np.asarray(gt_codes, dtype=np.int8)
    pred_codes = np.asarray(pred_codes, dtype=np.int8)
    runs = pm.valid_runs(gt_codes)
    return {
        "recall_iou": fc.RECALL_IOU,
        "short_spell_recall": fc.spell_recall_by_duration(gt_codes, pred_codes, runs, fps),
        "possession_ratio": fc.possession_ratio(gt_stats["possessions"],
                                                pred_stats["possessions"]),
        "d_edges": fc.dead_edges(gt_codes, pred_codes, runs, max(taus)),
        "decomposition": fc.decompose(gt_codes, pred_codes, runs),
    }


def match_flags(gt_codes, pred_codes):
    """{"0.3": [0 or 1, ...], ...}: per reference spell, 1 where the scorer matches it."""
    gt_codes = np.asarray(gt_codes, dtype=np.int8)
    flags, _ = _spell_flags(gt_codes, np.asarray(pred_codes, dtype=np.int8),
                            pm.valid_runs(gt_codes))
    return {"{:.1f}".format(iou): [int(v) for v in row]
            for iou, row in zip(pm.SPELL_IOUS, flags)}


def format_lines(name, head):
    """Printable lines for one scored file."""
    pct = "{:.1f}%".format
    lines = [
        "{}: spellF1 {:.4f} ({})  macroF1 {:.4f}".format(
            name, head["spell_f1_mean"],
            " ".join("@{} {:.3f}".format(k, v) for k, v in sorted(head["spell_f1"].items())),
            head["macro_f1"]),
        "  transitions @{}: matched {} missed {} spurious {}  mean latency {}".format(
            head["tau"], head["matched"], head["missed"], head["spurious"],
            rev.signed(head["mean_latency"])),
    ]
    for label, stats in (("reference", head["reference"]), ("results", head["results"])):
        lines.append("  {:<9} possessions {:>4}  mean {:5.1f} s  D share {:>6}  D segments {}"
                     .format(label, stats["possessions"], stats["mean_duration_s"],
                             pct(100.0 * stats["d_share"]), stats["d_segments"]))
    return lines


def write_report(out_dir, name, gt_path, result, head, taus=pm.TAUS, flags=None):
    """Write <out_dir>/<name>.json and <out_dir>/<name>_transitions.csv. Return both paths.

    `flags` is the match_flags() dict. The report holds it as "spell_match_flags" if given.
    """
    os.makedirs(out_dir, exist_ok=True)
    report = rev.json_safe(result)
    report["spell_f1"] = {"{:.1f}".format(k): v for k, v in report["spell_f1"].items()}
    report.update(submission=name, ground_truth=os.path.basename(gt_path), taus=list(taus),
                  summary=head)
    if flags is not None:
        report["spell_match_flags"] = flags
    json_path = os.path.join(out_dir, name + ".json")
    with open(json_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    trans_path = os.path.join(out_dir, name + "_transitions.csv")
    rev.write_transitions_csv(trans_path, result, max(taus))
    return json_path, trans_path


def read_codes(path, n_expected=None):
    """A frame,label CSV as codes. Raises ValueError if it is invalid or has the wrong length."""
    table = pm.read_label_csv(path)
    problem = pm.validate_submission(table, n_expected if n_expected is not None else len(table))
    if problem:
        raise ValueError("{}: {}".format(path, problem))
    return pm.label_codes(table)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="results CSV (frame,label)")
    parser.add_argument("--gt", required=True, help="reference CSV (frame,label), same length")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="report folder (default: {})".format(DEFAULT_OUT_DIR))
    parser.add_argument("--name", default=None, help="report name (default: results file stem)")
    parser.add_argument("--versus", default=None,
                        help="second results CSV: paired bootstrap of the spellF1 difference")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS,
                        help="paired bootstrap draws (default: {})".format(DEFAULT_DRAWS))
    args = parser.parse_args(argv)

    out_dir = os.path.abspath(args.out_dir)
    if out_dir.startswith(os.path.join(EVALUATION, "reports")):
        parser.error("never write to evaluation/reports/ (public ground truth only)")
    gt_codes = read_codes(args.gt)
    pred_codes = read_codes(args.results, len(gt_codes))
    name = args.name or os.path.splitext(os.path.basename(args.results))[0]
    result, gt_stats, pred_stats = score_codes(gt_codes, pred_codes)
    head = summary(result, gt_stats, pred_stats)
    if args.versus:
        other = read_codes(args.versus, len(gt_codes))
        head["versus"] = dict(paired_delta(gt_codes, pred_codes, other, args.draws),
                              other=os.path.basename(args.versus))
    head["failure_classes"] = failure_report(gt_codes, pred_codes, gt_stats, pred_stats)
    print("\n".join(format_lines(name, head)))
    if args.versus:
        pair = head["versus"]
        print("  versus {other}: spellF1 delta {delta:+.4f}, paired 5-95% {low:+.4f} .. "
              "{high:+.4f}, higher in {share_positive:.1%} of {draws} draws".format(**pair))
    print("\n".join(fc.format_lines(head["failure_classes"])))
    json_path, trans_path = write_report(out_dir, name, args.gt, result, head,
                                         flags=match_flags(gt_codes, pred_codes))
    print("wrote {} and {}".format(json_path, trans_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
