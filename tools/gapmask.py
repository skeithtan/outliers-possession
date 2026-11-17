"""Label-free robustness check: hide more of the ball and compare the engine with itself (P9-E).

    python tools/gapmask.py [--game NAME] [--mode MODE] [--config PATH] [--seed 0]
                            [--fraction 0.15] [--out-dir DIR] [--name NAME]

The tool hides an extra share of the ball on the graded frames of one match. It runs the engine on
the masked ball and on the unmasked ball, and compares the two results. The unmasked result of
the same version is the reference. The check measures stability, not accuracy. It reads no
ground truth and no label of the match. It exercises the hold of the last owner through a gap and
the gap-plus-stop term of the dead-ball evidence (PLAN.md 3.6) under more missing ball than the
ATD has.

The mask

The mask depends only on the game, the seed, the fraction and the observed gap runs of the ATD.
It never depends on the engine output, so every version runs on the same mask. These are the
steps:

1. Find the observed gap runs. A gap run is a maximal run of graded frames where the ATD ball x
   is NaN, inside one graded block. A run that touches the first or the last frame of a block is
   left out, because the block edge cuts its length.
2. Count the visible frames: graded frames where the ATD ball x is not NaN. The target is
   ceil(fraction x visible frames).
3. Start numpy's default_rng(seed). Repeat these steps until the hidden frames reach the target:
   a. Draw one observed gap run uniformly, with rng.integers, and take its length L.
   b. List every start s where frames s - MARGIN to s + L + MARGIN - 1 are all visible, graded,
      inside one block and not hidden yet. MARGIN visible frames on each side keep the new gap
      apart from an observed gap and from another new gap.
   c. If the list is empty, reject the draw and go back to step a. Otherwise draw one start
      uniformly from the list with rng.integers and hide frames s to s + L - 1.
4. Stop after MAX_DRAWS draws in total, even if the target is not reached. The report says so.

The hidden frames reach the target within one gap length: the last gap can pass the target by
less than its own length. A long draw that fits nowhere is rejected, so the inserted gaps can be
shorter on average than the observed gaps. The report gives both distributions by bucket.

The tool writes the mask to DIR/mask_<game>_<seed>_<fraction>.npz and reuses the file when it
exists. The npz holds the starts and the lengths of the new gaps. The tool prints the md5 of the
mask content (game, seed, fraction, starts and lengths), because the npz bytes carry a zip time
stamp.

The run

A hidden frame gets NaN in the ATD ball x and y, which is how the ATD marks a missing ball
(engine/data.py). The player arrays do not change. The unmasked run is `python -m engine.run`,
the official command. The masked run is this file with `--child`. The child replaces
engine.data.load_tactical in memory with a function that applies the mask, then calls
engine.run.main. The tool never edits engine/ (the pattern of tools/ablate.py).

The report

The tool scores the masked result against the unmasked result with the official metric,
evaluation/possession_metrics.evaluate(), through tools/score_game.score_codes(). It writes
DIR/gapmask_<game>_<name>_<seed>_<fraction>.json with sorted keys and prints a short table:

- Self-spellF1 for each IoU and the mean, macroF1 and the transitions at the loosest tolerance.
- The share of hidden frames whose label changed, and the share of all graded frames whose label
  changed. The frames that changed are split by the pair (unmasked label, masked label).
- The D inflation: the D share and the D segment count, masked minus unmasked. Both count
  inside each graded block, as tools/sanity.py does.
- The X inflation: the X frames on graded frames of the masked result. The count must stay 0.
- The same numbers for each bucket of inserted gap length (BUCKETS_S). For each bucket also the
  number and the total length of the inserted gaps and the recall of the unmasked spells that
  touch a gap of the bucket.
- The full-match sanity statistics and vetoes of tools/sanity.py for both results. The masked
  result uses the masked ball for the gap checks.

The tool prints summary statistics only. It never writes the masked arrays to the repository and
never prints a tracking row. Keep DIR outside the repository. Deterministic: a fixed seed, fixed
order, no clock, sorted JSON. Python 3.9, numpy and pandas only.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.dirname(os.path.abspath(__file__))
for _path in (ROOT, TOOLS, os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Visible frames kept on each side of an inserted gap. The engine drops a visible run shorter
# than BALL_MIN_RUN = 3 frames (engine/config.py), so 3 frames keep an inserted gap apart from its
# neighbours for the engine too. A tool constant, not read from the config, so that the mask does
# not change with the version.
MARGIN = 3
# Upper limit on the length draws of one mask. A fraction of 0.15 on BAR-ATH needs a few hundred.
MAX_DRAWS = 10000
DEFAULT_SEED = 0
DEFAULT_FRACTION = 0.15
DEFAULT_OUT_DIR = os.path.join("/tmp", "gapmask")
MODES = ("engine", "baseline50", "m1", "m2")     # the --mode choices of engine.run
# Inserted gap length buckets in seconds, half-open [low, high). PLAN.md 1.1 calls a gap longer
# than 2 s long. The other edges are those of tools/analysis/handover_after_gap.py.
BUCKETS_S = ((0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 30.0), (30.0, float("inf")))
H, A, D, X = 0, 1, 2, 3
LETTER = "HADX"


# -- the mask -------------------------------------------------------------------------------------

def runs_of(mask):
    """(starts, stops) of the maximal True runs of a bool array, half-open."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def graded_mask(n_frames, blocks):
    """bool (n_frames,): True on the graded frames. `blocks` is [(half, first, last_inclusive)]."""
    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    return graded


def observed_gaps(ball_missing, blocks):
    """[(start, stop)] half-open gap runs inside the blocks that do not touch a block edge.

    Also returns the number of runs left out because they touch a block edge.
    """
    gaps, cut = [], 0
    for _, first, last in blocks:
        starts, stops = runs_of(ball_missing[first:last + 1])
        for start, stop in zip(starts.tolist(), stops.tolist()):
            if start == 0 or stop == last + 1 - first:
                cut += 1
                continue
            gaps.append((first + start, first + stop))
    return gaps, cut


def draw_gaps(visible, lengths, fraction, seed, margin=MARGIN, max_draws=MAX_DRAWS):
    """New gaps on the visible graded frames, by the procedure of the module docstring.

    `visible` is bool (N,), True on the graded frames with a ball. `lengths` holds the observed
    gap lengths in frames. Returns {"starts", "lengths" (int64 arrays, in draw order), "target",
    "hidden", "draws", "rejected", "reached"}.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.size == 0 or (lengths <= 0).any():
        raise ValueError("need at least one observed gap length, all positive")
    free = np.asarray(visible, dtype=bool).copy()
    n = free.size
    target = int(np.ceil(fraction * int(free.sum())))
    rng = np.random.default_rng(seed)
    starts, picked = [], []
    hidden = draws = rejected = 0
    while hidden < target and draws < max_draws:
        draws += 1
        length = int(lengths[rng.integers(lengths.size)])
        width = length + 2 * margin
        if width > n:
            rejected += 1
            continue
        cumulative = np.concatenate(([0], np.cumsum(free, dtype=np.int64)))
        windows = np.flatnonzero(cumulative[width:] - cumulative[:-width] == width)
        if windows.size == 0:
            rejected += 1
            continue
        start = int(windows[rng.integers(windows.size)]) + margin
        free[start:start + length] = False
        starts.append(start)
        picked.append(length)
        hidden += length
    return {"starts": np.array(starts, dtype=np.int64), "lengths": np.array(picked, dtype=np.int64),
            "target": target, "hidden": int(hidden), "draws": int(draws),
            "rejected": int(rejected), "reached": bool(hidden >= target)}


def hidden_mask(n_frames, starts, lengths):
    """bool (n_frames,): True on the frames of the inserted gaps."""
    mask = np.zeros(n_frames, dtype=bool)
    for start, length in zip(np.asarray(starts).tolist(), np.asarray(lengths).tolist()):
        mask[start:start + length] = True
    return mask


def fraction_text(fraction):
    """The fraction as it appears in a file name: 0.15 -> '0.15'."""
    return "{:g}".format(fraction)


def mask_path(out_dir, game, seed, fraction):
    """DIR/mask_<game>_<seed>_<fraction>.npz."""
    return os.path.join(out_dir, "mask_{}_{}_{}.npz".format(game, seed, fraction_text(fraction)))


def mask_md5(game, seed, fraction, starts, lengths):
    """md5 of the mask content. The npz file bytes carry a zip time stamp, so do not hash them."""
    digest = hashlib.md5()
    digest.update("{}|{}|{}|".format(game, seed, fraction_text(fraction)).encode("ascii"))
    digest.update(np.asarray(starts, dtype="<i8").tobytes())
    digest.update(np.asarray(lengths, dtype="<i8").tobytes())
    return digest.hexdigest()


def lengths_md5(lengths):
    """md5 of the observed gap lengths, stored in the mask file to detect another ATD."""
    return hashlib.md5(np.asarray(lengths, dtype="<i8").tobytes()).hexdigest()


def build_mask(ball_missing, blocks, game, seed, fraction):
    """The mask of one (game, seed, fraction) as a dict. See draw_gaps() for the keys."""
    graded = graded_mask(len(ball_missing), blocks)
    gaps, cut = observed_gaps(ball_missing, blocks)
    lengths = np.array([b - a for a, b in gaps], dtype=np.int64)
    drawn = draw_gaps(graded & ~ball_missing, lengths, fraction, seed)
    drawn.update(game=game, seed=int(seed), fraction=float(fraction),
                 observed_lengths=lengths, observed_cut=int(cut),
                 observed_md5=lengths_md5(lengths),
                 md5=mask_md5(game, seed, fraction, drawn["starts"], drawn["lengths"]))
    return drawn


def save_mask(path, mask):
    """Write the mask dict to an npz (no tracking values, only frame numbers and lengths)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, starts=mask["starts"], lengths=mask["lengths"],
             observed_lengths=mask["observed_lengths"],
             meta=np.array(json.dumps({key: mask[key] for key in sorted(mask)
                                       if key not in ("starts", "lengths", "observed_lengths")},
                                      sort_keys=True)))


def read_mask(path):
    """The mask dict of an npz written by save_mask()."""
    with np.load(path, allow_pickle=False) as npz:
        mask = json.loads(str(npz["meta"]))
        for key in ("starts", "lengths", "observed_lengths"):
            mask[key] = npz[key].astype(np.int64)
    return mask


def frozen_mask(path, ball_missing, blocks, game, seed, fraction):
    """The mask at `path` if it exists, else a new one written there. Returns (mask, reused).

    A file that exists must match the game, the seed, the fraction and the observed gap lengths,
    and must hide visible graded frames only. Otherwise the function raises ValueError.
    """
    if not os.path.isfile(path):
        mask = build_mask(ball_missing, blocks, game, seed, fraction)
        save_mask(path, mask)
        return mask, False
    mask = read_mask(path)
    gaps, _ = observed_gaps(ball_missing, blocks)
    lengths = np.array([b - a for a, b in gaps], dtype=np.int64)
    expect = {"game": game, "seed": int(seed), "fraction": float(fraction),
              "observed_md5": lengths_md5(lengths),
              "md5": mask_md5(game, seed, fraction, mask["starts"], mask["lengths"])}
    wrong = sorted(key for key in expect if mask.get(key) != expect[key])
    if wrong:
        raise ValueError("{}: does not match this mask ({}). Delete it or use another "
                         "--out-dir.".format(path, ", ".join(wrong)))
    hidden = hidden_mask(len(ball_missing), mask["starts"], mask["lengths"])
    if (hidden & (ball_missing | ~graded_mask(len(ball_missing), blocks))).any():
        raise ValueError("{}: hides a frame that is not a visible graded frame".format(path))
    return mask, True


def apply_mask(tactical, hidden):
    """A copy of the load_tactical() dict with NaN ball x and y on the hidden frames."""
    out = dict(tactical)
    for key in ("bx", "by"):
        values = np.array(tactical[key], copy=True)
        values[hidden] = np.nan
        out[key] = values
    return out


# -- the comparison -------------------------------------------------------------------------------

def bucket_name(low, high):
    """'2-5 s', or '> 30 s' for the open bucket, or '< 2 s' for the first."""
    if high == float("inf"):
        return "> {:g} s".format(low)
    if low == 0.0:
        return "< {:g} s".format(high)
    return "{:g}-{:g} s".format(low, high)


def bucket_of(length, fps, buckets=BUCKETS_S):
    """Index of the bucket that holds a gap of `length` frames."""
    seconds = length / float(fps)
    for index, (low, high) in enumerate(buckets):
        if low <= seconds < high:
            return index
    raise ValueError("no bucket for {} s".format(seconds))


def change_pairs(ref, new, where):
    """{"HA": n, ...}: the frames of `where` whose label goes from ref to new, by pair."""
    out = {}
    for a in range(len(LETTER)):
        for b in range(len(LETTER)):
            if a == b:
                continue
            n = int(np.sum(where & (ref == a) & (new == b)))
            if n:
                out[LETTER[a] + LETTER[b]] = n
    return out


def label_changes(ref, new, hidden, graded):
    """Label-change accounting of two code vectors (H=0 A=1 D=2 X=3), on the raw codes.

    Returns {"hidden_frames", "hidden_changed", "hidden_changed_share", "graded_frames",
    "graded_changed", "graded_changed_share", "outside_changed", "hidden_pairs",
    "outside_pairs"}. "outside" means graded and not hidden.
    """
    ref, new = np.asarray(ref), np.asarray(new)
    hidden = np.asarray(hidden, dtype=bool) & graded
    changed = (ref != new) & graded
    outside = graded & ~hidden
    n_hidden, n_graded = int(hidden.sum()), int(graded.sum())
    return {
        "hidden_frames": n_hidden,
        "hidden_changed": int((changed & hidden).sum()),
        "hidden_changed_share": float((changed & hidden).sum()) / n_hidden if n_hidden else None,
        "graded_frames": n_graded,
        "graded_changed": int(changed.sum()),
        "graded_changed_share": float(changed.sum()) / n_graded if n_graded else None,
        "outside_changed": int((changed & outside).sum()),
        "hidden_pairs": change_pairs(ref, new, hidden),
        "outside_pairs": change_pairs(ref, new, outside),
    }


def _overlaps(segments, start, stop):
    """True if any half-open (a, b) of `segments` overlaps [start, stop)."""
    return any(a < stop and start < b for a, b in segments)


def bucket_rows(ref, new, ref_filled, new_filled, mask, runs, flags, fps):
    """One dict of numbers per bucket of inserted gap length (module docstring)."""
    import possession_metrics as pm

    n = len(ref)
    ref_d = [(a, b) for a, b, k in pm.segments(ref_filled, runs) if k == D]
    new_d = [(a, b) for a, b, k in pm.segments(new_filled, runs) if k == D]
    spells = pm.possession_spells(ref, runs)
    gaps = list(zip(mask["starts"].tolist(), mask["lengths"].tolist()))
    observed = [int(v) for v in np.asarray(mask["observed_lengths"]).tolist()]
    rows = []
    for index, (low, high) in enumerate(BUCKETS_S):
        mine = [(s, s + length) for s, length in gaps if bucket_of(length, fps) == index]
        hidden = hidden_mask(n, [a for a, _ in mine], [b - a for a, b in mine])
        n_hidden = int(hidden.sum())
        touched = [i for i, (a, b, _) in enumerate(spells)
                   if any(a < stop and start < b for start, stop in mine)]
        recall = {iou: (float(np.mean([row[i] for i in touched])) if touched else None)
                  for iou, row in sorted(flags.items())}
        rows.append({
            "bucket": bucket_name(low, high),
            "observed_gaps": sum(1 for v in observed if bucket_of(v, fps) == index),
            "gaps": len(mine),
            "frames": n_hidden,
            "seconds": n_hidden / float(fps),
            "hidden_changed": int(((ref != new) & hidden).sum()),
            "hidden_changed_share": (float(((ref != new) & hidden).sum()) / n_hidden
                                     if n_hidden else None),
            "hidden_pairs": change_pairs(ref, new, hidden),
            "d_share_hidden_unmasked": (float((ref_filled[hidden] == D).sum()) / n_hidden
                                        if n_hidden else None),
            "d_share_hidden_masked": (float((new_filled[hidden] == D).sum()) / n_hidden
                                      if n_hidden else None),
            "d_segments_touching_unmasked": sum(1 for a, b in ref_d if _overlaps(mine, a, b)),
            "d_segments_touching_masked": sum(1 for a, b in new_d if _overlaps(mine, a, b)),
            "x_hidden_masked": int((new[hidden] == X).sum()),
            "spells_touching": len(touched),
            "spell_recall_touching": recall,
        })
    return rows


def alive_holds(ref, new, mask):
    """The inserted gaps that lie inside one H or A spell of the unmasked result.

    Returns {"gaps", "frames", "frames_kept", "frames_kept_share", "gaps_all_kept"}: a hidden
    frame is kept when the masked label equals the unmasked label. Hold-through-gap gives a
    share of 1.
    """
    gaps = frames = kept = whole = 0
    for start, length in zip(mask["starts"].tolist(), mask["lengths"].tolist()):
        stretch = ref[start - 1:start + length + 1]
        if stretch.size == 0 or stretch[0] not in (H, A) or not (stretch == stretch[0]).all():
            continue
        same = int((new[start:start + length] == ref[start:start + length]).sum())
        gaps += 1
        frames += length
        kept += same
        whole += int(same == length)
    return {"gaps": gaps, "frames": frames, "frames_kept": kept,
            "frames_kept_share": float(kept) / frames if frames else None,
            "gaps_all_kept": whole}


def sanity_numbers(stats):
    """The tools/sanity.py numbers that the report keeps for one result."""
    total = stats["total"]
    return {
        "possessions": int(total["possessions"]),
        "d_share": float(total["d_share"]),
        "d_segments": int(total["d_segments"]),
        "x_graded": int(total["x_graded"]),
        "poss_share_lt_1s": total["poss_share_lt_1s"],
        "tag_agreement": stats["tags"]["total"]["agreement"] if "tags" in stats else None,
        "set_pieces": {key: value for key, value in stats.get("set_pieces", {}).items()
                       if not isinstance(value, list)},
        "gap_label_changes": stats["gaps"]["n_changes"] if "gaps" in stats else None,
        "long_gaps": stats["gaps"]["long_gaps"] if "gaps" in stats else None,
        "d_block_end_not_half_end": stats["d_block_end"]["n_not_half_end"],
        "vetoes": {v["name"]: {"value": v["value"], "rule": v["rule"],
                               "violated": bool(v["violated"])} for v in stats["vetoes"]},
        "violations": list(stats["violations"]),
    }


def inflation(ref_total, new_total):
    """D, possession and X numbers of two tools/sanity.py totals, masked minus unmasked.

    The totals count inside each graded block, so a D segment never joins two blocks. The
    self-score uses the runs of the scorer instead: the frames where the reference is not X.
    Two blocks without a frame between them are one run there.
    """
    out = {}
    for key in ("d_share", "d_segments", "possessions", "x_graded"):
        out[key + "_unmasked"] = ref_total[key]
        out[key + "_masked"] = new_total[key]
        out[key + "_delta"] = new_total[key] - ref_total[key]
    return out


def compare(ref, new, mask, blocks, fps, ball_missing, tags, game):
    """Every number of the report for one pair of results, as a dict."""
    import possession_metrics as pm
    import sanity
    import score_game

    ref = np.asarray(ref, dtype=np.int8)
    new = np.asarray(new, dtype=np.int8)
    n = len(ref)
    graded = graded_mask(n, blocks)
    runs = [(first, last + 1) for _, first, last in blocks]
    hidden = hidden_mask(n, mask["starts"], mask["lengths"])
    result, ref_stats, new_stats = score_game.score_codes(ref, new)
    head = score_game.summary(result, ref_stats, new_stats)
    flags = score_game.match_flags(ref, new)
    ref_filled = pm.fill_abstentions(ref, runs)
    new_filled = pm.fill_abstentions(new, runs)
    masked_missing = ball_missing | hidden
    ref_sanity = sanity.sanity_stats(ref, tags=tags, ball_missing=ball_missing, blocks=blocks,
                                     game=game)
    new_sanity = sanity.sanity_stats(new, tags=tags, ball_missing=masked_missing, blocks=blocks,
                                     game=game)
    n_visible = int((graded & ~ball_missing).sum())
    return {
        "self_score": {
            "spell_f1": head["spell_f1"], "spell_f1_mean": head["spell_f1_mean"],
            "macro_f1": head["macro_f1"], "tau": head["tau"], "matched": head["matched"],
            "missed": head["missed"], "spurious": head["spurious"],
            "mean_latency": head["mean_latency"],
            "scorer": "evaluation/possession_metrics.evaluate via tools/score_game.score_codes, "
                      "the unmasked result as the reference",
        },
        "changes": label_changes(ref, new, hidden, graded),
        "inflation": inflation(ref_sanity["total"], new_sanity["total"]),
        "buckets": bucket_rows(ref, new, ref_filled, new_filled, mask, runs, flags, fps),
        "alive_holds": alive_holds(ref, new, mask),
        "ball": {
            "graded_frames": int(graded.sum()),
            "visible_unmasked": n_visible,
            "visible_masked": int((graded & ~masked_missing).sum()),
            "visible_share_unmasked": n_visible / float(graded.sum()),
            "visible_share_masked": int((graded & ~masked_missing).sum()) / float(graded.sum()),
            "hidden_share_of_visible": int(hidden.sum()) / float(n_visible) if n_visible else None,
        },
        "sanity": {"unmasked": sanity_numbers(ref_sanity), "masked": sanity_numbers(new_sanity)},
    }


# -- printing -------------------------------------------------------------------------------------

def _pct(value):
    return "-" if value is None else "{:.1f}%".format(100.0 * value)


def format_lines(report):
    """The short printed table of one report."""
    mask, score = report["mask"], report["comparison"]["self_score"]
    changes, infl = report["comparison"]["changes"], report["comparison"]["inflation"]
    ball, holds = report["comparison"]["ball"], report["comparison"]["alive_holds"]
    lines = [
        "{} {} seed {} fraction {}: mask md5 {} ({})".format(
            report["game"], report["name"], mask["seed"], fraction_text(mask["fraction"]),
            mask["md5"], "reused" if report["mask_reused"] else "new"),
        "  inserted {} gaps, {} frames (target {}, {}), {} draws, {} rejected".format(
            len(mask["lengths"]), mask["hidden"], mask["target"],
            "reached" if mask["reached"] else "NOT reached", mask["draws"], mask["rejected"]),
        "  ball visible on graded frames: {} unmasked, {} masked".format(
            _pct(ball["visible_share_unmasked"]), _pct(ball["visible_share_masked"])),
        "  self-spellF1 {:.4f} ({})  macroF1 {:.4f}  @{}: matched {} missed {} spurious {}".format(
            score["spell_f1_mean"],
            " ".join("@{} {:.3f}".format(k, v) for k, v in sorted(score["spell_f1"].items())),
            score["macro_f1"], score["tau"], score["matched"], score["missed"],
            score["spurious"]),
        "  changed: {} of hidden frames ({} of {}), {} of graded frames ({}), {} outside".format(
            _pct(changes["hidden_changed_share"]), changes["hidden_changed"],
            changes["hidden_frames"], _pct(changes["graded_changed_share"]),
            changes["graded_changed"], changes["outside_changed"]),
        "  D share {} -> {} ({:+.2f} pt)  D segments {} -> {} ({:+d})  possessions {} -> {}  "
        "X on graded {}".format(
            _pct(infl["d_share_unmasked"]), _pct(infl["d_share_masked"]),
            100.0 * infl["d_share_delta"], infl["d_segments_unmasked"],
            infl["d_segments_masked"], infl["d_segments_delta"], infl["possessions_unmasked"],
            infl["possessions_masked"], infl["x_graded_masked"]),
        "  gaps inside one alive spell: {} gaps, {} frames, {} kept, {} gaps kept whole".format(
            holds["gaps"], holds["frames"], _pct(holds["frames_kept_share"]),
            holds["gaps_all_kept"]),
        "",
        "  | bucket | observed | gaps | frames | changed | D hidden un/masked | D segs un/masked "
        "| X | spells | recall @0.5 |",
        "  |---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in report["comparison"]["buckets"]:
        recall = row["spell_recall_touching"].get("0.5")
        lines.append("  | {} | {} | {} | {} | {} | {} / {} | {} / {} | {} | {} | {} |".format(
            row["bucket"], row["observed_gaps"], row["gaps"], row["frames"],
            _pct(row["hidden_changed_share"]), _pct(row["d_share_hidden_unmasked"]),
            _pct(row["d_share_hidden_masked"]), row["d_segments_touching_unmasked"],
            row["d_segments_touching_masked"], row["x_hidden_masked"], row["spells_touching"],
            "-" if recall is None else "{:.3f}".format(recall)))
    lines.append("")
    for which in ("unmasked", "masked"):
        sane = report["comparison"]["sanity"][which]
        lines.append("  sanity {:<8} possessions {}  D share {}  D segments {}  X graded {}  "
                     "vetoes {}".format(which, sane["possessions"], _pct(sane["d_share"]),
                                        sane["d_segments"], sane["x_graded"],
                                        "fail: " + ",".join(sane["violations"])
                                        if sane["violations"] else "pass"))
    return lines


# -- running --------------------------------------------------------------------------------------

def engine_args(mode, config, game):
    """The engine.run arguments after --out."""
    args = ["--mode", mode]
    if config is not None:
        args += ["--config", config]
    if game is not None:
        args += ["--game", game]
    return args


def child(mask_file, out_csv, mode, config, game):
    """Child process: apply the mask to engine.data.load_tactical in memory, run engine.run."""
    from engine import data
    from engine import run as engine_run

    mask = read_mask(mask_file)
    name = data.resolve_game(game)
    if mask["game"] != name:
        raise ValueError("{}: mask of {} used on {}".format(mask_file, mask["game"], name))
    original = data.load_tactical

    def masked_tactical(game=None, *args, **kwargs):
        tactical = original(game, *args, **kwargs)
        if data.resolve_game(game) != name:
            return tactical
        hidden = hidden_mask(tactical["n_frames"], mask["starts"], mask["lengths"])
        return apply_mask(tactical, hidden)

    data.load_tactical = masked_tactical
    return engine_run.main(["--out", out_csv] + engine_args(mode, config, game))


def run_engine(command, log_path):
    """Run one engine command from the repository root, write its log, raise on failure."""
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True)
    with open(log_path, "w") as handle:
        handle.write(" ".join(command) + "\n" + result.stdout)
    if result.returncode != 0:
        raise RuntimeError("{} failed:\n{}".format(" ".join(command), result.stdout[-2000:]))


def file_md5(path):
    """Hex md5 of a file."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_name(mode, config):
    """The version name: the mode, plus the config file stem when there is one."""
    if config is None:
        return mode
    return "{}_{}".format(mode, os.path.splitext(os.path.basename(config))[0])


def _jsonable(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default=None, help="match of games.py (default: games.GAME)")
    parser.add_argument("--mode", choices=MODES, default="engine", help="engine.run --mode")
    parser.add_argument("--config", default=None, help="engine.run --config file")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fraction", type=float, default=DEFAULT_FRACTION,
                        help="share of the visible graded frames to hide (default 0.15)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="folder outside the repository (default: {})".format(DEFAULT_OUT_DIR))
    parser.add_argument("--name", default=None,
                        help="version name in the file names (default: the mode and config stem)")
    parser.add_argument("--python", default=sys.executable, help="interpreter for the runs")
    parser.add_argument("--child", nargs=2, metavar=("MASK", "OUT_CSV"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config = os.path.abspath(args.config) if args.config else None

    if args.child:
        return child(args.child[0], args.child[1], args.mode, config, args.game)

    from engine import data

    if not 0.0 < args.fraction < 1.0:
        parser.error("--fraction must be between 0 and 1")
    game = data.resolve_game(args.game)
    out_dir = os.path.abspath(args.out_dir)
    if out_dir == ROOT or out_dir.startswith(ROOT + os.sep):
        parser.error("--out-dir must be outside the repository")
    name = args.name or default_name(args.mode, args.config)
    tag = "{}_{}_{}_{}".format(game, name, args.seed, fraction_text(args.fraction))
    for sub in ("results", "logs"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    blocks = data.graded_blocks(game)
    ball_missing = np.isnan(data.load_tactical(game)["bx"])
    path = mask_path(out_dir, game, args.seed, args.fraction)
    mask, reused = frozen_mask(path, ball_missing, blocks, game, args.seed, args.fraction)

    plain_csv = os.path.join(out_dir, "results", tag + "_unmasked.csv")
    masked_csv = os.path.join(out_dir, "results", tag + "_masked.csv")
    run_engine([args.python, "-m", "engine.run", "--out", plain_csv]
               + engine_args(args.mode, config, args.game),
               os.path.join(out_dir, "logs", tag + "_unmasked.log"))
    run_engine([args.python, os.path.abspath(__file__), "--child", path, masked_csv]
               + engine_args(args.mode, config, args.game),
               os.path.join(out_dir, "logs", tag + "_masked.log"))

    import _common as cm

    ref = cm.read_results(plain_csv, game)
    new = cm.read_results(masked_csv, game)
    comparison = compare(ref, new, mask, blocks, data.fps(game), ball_missing,
                         data.load_tagging(game), game)
    report = {
        "game": game, "name": name, "mode": args.mode,
        "config": os.path.basename(config) if config else None,
        "mask_file": os.path.basename(path), "mask_reused": reused,
        "mask": {key: mask[key] for key in ("game", "seed", "fraction", "md5", "target",
                                            "hidden", "draws", "rejected", "reached",
                                            "observed_cut", "observed_md5")},
        "observed_gaps": int(len(mask["observed_lengths"])),
        "unmasked_md5": file_md5(plain_csv), "masked_md5": file_md5(masked_csv),
        "comparison": comparison,
    }
    report["mask"]["lengths"] = [int(v) for v in mask["lengths"].tolist()]
    json_path = os.path.join(out_dir, "gapmask_{}.json".format(tag))
    with open(json_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=_jsonable)
        handle.write("\n")
    print("\n".join(format_lines(report)))
    print("")
    print("unmasked md5 {}  masked md5 {}".format(report["unmasked_md5"], report["masked_md5"]))
    print("wrote {}".format(json_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
