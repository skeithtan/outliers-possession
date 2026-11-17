"""Submission writer, validator hook and full-match summary (PLAN.md Sections 0, 1.7, 3.8, 5.3).

    write_submission(labels, path)   frame,label CSV, byte-deterministic, refuses graded X
    validate(path)                   runs evaluation/validate_submission.py in a subprocess
    summarize(labels, blocks)        per-half + total possession statistics and the 5.3 vetoes
    format_summary(summary)          the summary as a fixed-width text table
    md5_of(path)                     hex md5 of a file (the reproducibility fingerprint)

Possessions, D segments and transitions are counted inside each graded block separately, never
across the halftime hole, exactly as evaluation/possession_metrics.py::segments() does. This
module never reads the ground truth (team rule).
"""

import hashlib
import os
import subprocess
import sys

import numpy as np

from engine import data
from engine.config import Config

games = data.games
ROOT = data.ROOT

# The submission alphabet and header (docs/challenge SUBMISSION.md; PLAN.md Section 0).
LABELS = ("H", "A", "D", "X")
POSSESSION_LABELS = ("H", "A")
HEADER = "frame,label\n"
VALIDATOR = os.path.join(ROOT, "evaluation", "validate_submission.py")

PERCENT = 100.0
SECONDS_PER_MINUTE = 60.0


def n_frames_default(game=None):
    """Submission length: the tactical video's frame count, from games.py."""
    return data.frame_count(game)


def _graded_mask(n_frames, blocks):
    """bool (n_frames,): True on frames inside any graded block (blocks are inclusive)."""
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        mask[first:last + 1] = True
    return mask


def write_submission(labels, path, n_frames=None, blocks=None, game=None):
    """Write `labels` as a frame,label CSV and return `path`.

    Format: header exactly "frame,label", then "i,L" for i = 0..n_frames-1, "\\n" line endings
    on every platform (binary write), one buffered pass.

    Raises ValueError (and writes nothing) unless len(labels) == n_frames (default: the tactical
    video length of `game` from games.py), every label is one of H/A/D/X, and no X falls inside a
    graded block (default blocks: engine.data.graded_blocks(game)). Graded X is never fixed
    silently (team rule; PLAN.md 1.7). `game` None means games.GAME.
    """
    if n_frames is None:
        n_frames = n_frames_default(game)
    if blocks is None:
        blocks = data.graded_blocks(game)
    labels = np.asarray(labels).astype(str)
    if labels.ndim != 1 or len(labels) != n_frames:
        raise ValueError("expected {} labels, got shape {}".format(n_frames, labels.shape))
    bad = sorted(set(np.unique(labels).tolist()) - set(LABELS))
    if bad:
        raise ValueError("labels outside {}: {}".format(list(LABELS), bad[:5]))
    graded_x = np.flatnonzero((labels == "X") & _graded_mask(n_frames, blocks))
    if graded_x.size:
        raise ValueError("X on {} graded frames (first {}); hold the last owner instead".format(
            graded_x.size, int(graded_x[0])))

    body = "".join(["{},{}\n".format(i, label) for i, label in enumerate(labels.tolist())])
    folder = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "wb") as handle:
        handle.write((HEADER + body).encode("ascii"))
    return path


def validate(path, rows=None):
    """Run the organisers' validator on `path` with this interpreter; return its stdout.

    Raises RuntimeError with the validator's output if it exits non-zero.
    """
    command = [sys.executable, VALIDATOR, os.path.abspath(path)]
    if rows is not None:
        command += ["--rows", str(int(rows))]
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True)
    if result.returncode != 0:
        raise RuntimeError("validator failed on {}:\n{}".format(path, result.stdout))
    return result.stdout


def md5_of(path):
    """Hex md5 of the file at `path`."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runs(window):
    """[(start, stop, label)] maximal constant runs of a 1-D label array (half-open, local)."""
    if not len(window):
        return []
    breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(window)]))
    return [(int(bounds[i]), int(bounds[i + 1]), str(window[bounds[i]]))
            for i in range(len(bounds) - 1)]


def _stats(spell_lengths, n_graded, n_d, n_d_segments, n_transitions, n_graded_x, fps):
    """One row of the summary from raw counts (frames); durations in seconds."""
    lengths = np.asarray(spell_lengths, dtype=np.float64) / fps
    minutes = n_graded / float(fps) / SECONDS_PER_MINUTE
    return {
        "graded_frames": int(n_graded),
        "possessions": int(len(lengths)),
        "mean_s": float(lengths.mean()) if len(lengths) else 0.0,
        "median_s": float(np.median(lengths)) if len(lengths) else 0.0,
        "longest_s": float(lengths.max()) if len(lengths) else 0.0,
        "d_share": n_d / float(n_graded) if n_graded else 0.0,
        "d_segments": int(n_d_segments),
        "transitions": int(n_transitions),
        "transitions_per_min": n_transitions / minutes if minutes else 0.0,
        "graded_x": int(n_graded_x),
    }


def summarize(labels, blocks=None, fps=None, cfg=None, game=None):
    """Full-match plausibility statistics (PLAN.md 5.3), per graded block and in total.

    Returns {"halves": {half: row}, "total": row, "vetoes": {name: bool passed}, "ok": bool},
    where a row has graded_frames, possessions (maximal H/A runs inside the block), mean_s,
    median_s, longest_s (possession durations in seconds), d_share (of graded frames),
    d_segments (maximal D runs inside the block), transitions (label changes inside the block),
    transitions_per_min (per graded minute) and graded_x (must be 0).
    Vetoes are True when the version passes: possessions in [180, 450], D share in [18%, 55%],
    D segments >= 55, no graded X. The limits are the VETO_* fields of `cfg` (engine.config,
    PLAN 5.3). The default is Config.default(). The default blocks and fps come from `game` in
    games.py, and `game` None means games.GAME.
    """
    if blocks is None:
        blocks = data.graded_blocks(game)
    if fps is None:
        fps = data.fps(game)
    if cfg is None:
        cfg = Config.default()
    labels = np.asarray(labels).astype(str)

    halves = {}
    all_spells = []
    totals = dict(n_graded=0, n_d=0, n_d_segments=0, n_transitions=0, n_graded_x=0)
    for half, first, last in blocks:
        window = labels[first:last + 1]
        runs = _runs(window)
        spells = [stop - start for start, stop, label in runs if label in POSSESSION_LABELS]
        counts = dict(n_graded=len(window), n_d=int(np.sum(window == "D")),
                      n_d_segments=sum(1 for _, _, label in runs if label == "D"),
                      n_transitions=max(len(runs) - 1, 0),
                      n_graded_x=int(np.sum(window == "X")))
        halves[half] = _stats(spells, fps=fps, **counts)
        all_spells += spells
        for key in totals:
            totals[key] += counts[key]
    total = _stats(all_spells, fps=fps, **totals)

    lo_share = int(round(cfg.VETO_D_SHARE_MIN * PERCENT))
    hi_share = int(round(cfg.VETO_D_SHARE_MAX * PERCENT))
    vetoes = {
        "possessions_{}_{}".format(cfg.VETO_POSSESSIONS_MIN, cfg.VETO_POSSESSIONS_MAX):
            cfg.VETO_POSSESSIONS_MIN <= total["possessions"] <= cfg.VETO_POSSESSIONS_MAX,
        "d_share_{}_{}".format(lo_share, hi_share):
            cfg.VETO_D_SHARE_MIN <= total["d_share"] <= cfg.VETO_D_SHARE_MAX,
        "d_segments_ge_{}".format(cfg.VETO_MIN_D_SEGMENTS):
            total["d_segments"] >= cfg.VETO_MIN_D_SEGMENTS,
        "graded_x_eq_{}".format(cfg.VETO_MAX_GRADED_X):
            total["graded_x"] <= cfg.VETO_MAX_GRADED_X,
    }
    return {"halves": halves, "total": total, "vetoes": vetoes, "ok": all(vetoes.values())}


def format_summary(summary):
    """Fixed-width text table of a summarize() result, one column per half plus total."""
    columns = [("half {}".format(h), row) for h, row in sorted(summary["halves"].items())]
    columns.append(("total", summary["total"]))
    fields = [
        ("graded frames", "graded_frames", "{:d}"),
        ("possessions", "possessions", "{:d}"),
        ("mean dur (s)", "mean_s", "{:.1f}"),
        ("median dur (s)", "median_s", "{:.1f}"),
        ("longest spell (s)", "longest_s", "{:.1f}"),
        ("D share", "d_share", "{:.1%}"),
        ("D segments", "d_segments", "{:d}"),
        ("transitions/min", "transitions_per_min", "{:.2f}"),
        ("X on graded", "graded_x", "{:d}"),
    ]
    lines = ["{:<18}".format("") + "".join("{:>10}".format(name) for name, _ in columns)]
    for title, key, fmt in fields:
        lines.append("{:<18}".format(title) +
                     "".join("{:>10}".format(fmt.format(row[key])) for _, row in columns))
    lines.append("vetoes (PLAN 5.3): " + "  ".join(
        "{} {}".format(name, "ok" if passed else "FAIL")
        for name, passed in summary["vetoes"].items()))
    lines.append("verdict: {}".format("PASS" if summary["ok"] else "VETOED"))
    return "\n".join(lines)
