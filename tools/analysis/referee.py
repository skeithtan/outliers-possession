"""Label-free referees between two label files of one match. Not part of the submission.

    python tools/analysis/referee.py camera --x X.csv --y Y.csv --aligned DIR/bb_on_tactical.npz
    python tools/analysis/referee.py tags --x X.csv --y Y.csv

Two systems label the same match, for example c22 and a learned candidate. No label of the match
exists to say which system is right. This tool compares each system with a signal that neither
system reads (docs/plan/research/RESEARCH-5.md, Section 4):

- `camera`: the state of the broadcast camera. `away` means that the broadcast frame has no valid
  homography or is a replay. A director leaves the wide shot when play stops. The referee tests
  the frames where the two systems disagree on alive or dead.
- `tags`: the team of the Smart Tagging phases. The referee tests the frames where both systems
  say alive and name different teams.

Each referee measures the signal on the frames where the two systems agree. It then reads the
frames where they disagree as a mixture of the two agreement cells. The result is an estimate of
the share of the disagreement frames where each system is right.

The tool reads two submission CSVs, the Smart Tagging file, and the file that
tools/analysis/r2_broadcast_align.py writes. It reads no ground truth and no video. It prints
counts and shares only. A person does not look at a frame. The estimates rest on two assumptions:
the agreement frames are mostly right, and the signal behaves in the disagreement frames as it
does in the agreement frames. The run table of the `camera` referee shows where the second
assumption is weak. The numbers are evidence for a decision. They are not a score.

Python 3.9, numpy only.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import _common as cm  # noqa: E402
from _common import A, D, H, pm  # noqa: E402

# Run lengths of the camera run table, in seconds. A broadcast cut follows a stoppage with a
# delay, so runs under 2 s say little and runs of 5 s and more say most. The values only split a
# report. They select nothing and are not fitted to any label.
RUN_BUCKETS_S = (2.0, 5.0)
# Frames this close to a change of the tag team are left out of the `tags` referee. PLAN.md 1.6
# measured that phase boundaries are soft by about 4 s and lean early.
TAG_EDGE_S = 4.0


def runs_of(mask):
    """Half-open [start, stop) runs of True in a boolean array, as a list of int pairs."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def graded_mask(n_frames, graded_runs):
    """bool (n_frames,): True inside the half-open graded runs."""
    mask = np.zeros(int(n_frames), dtype=bool)
    for start, stop in graded_runs:
        mask[int(start):int(stop)] = True
    return mask


def share(mask, within):
    """(count of `within`, share of `mask` in `within`). The share is None on an empty set."""
    count = int(np.count_nonzero(within))
    if not count:
        return 0, None
    return count, float(np.count_nonzero(mask & within)) / count


def mixture(cell, high, low):
    """The share of a cell that behaves like the `high` cell, from three signal shares.

    The cell is a mixture of frames like the `high` cell and frames like the `low` cell. Its
    signal share is then high * w + low * (1 - w). The function returns w, clipped to 0..1. It
    returns None when a share is missing or when the two agreement cells do not differ.
    """
    if cell is None or high is None or low is None or high == low:
        return None
    return float(min(1.0, max(0.0, (cell - low) / (high - low))))


def camera_away(valid, replay, b_frame):
    """(away, covered) from the arrays of tools/analysis/r2_broadcast_align.py.

    `covered` is True where a broadcast frame maps to the tactical frame (b_frame >= 0). `away`
    is True on a covered frame without a valid homography, or with the replay flag.
    """
    covered = np.asarray(b_frame) >= 0
    away = covered & (~np.asarray(valid, dtype=bool) | np.asarray(replay, dtype=bool))
    return away, covered


def _bucket(length_s):
    low, high = RUN_BUCKETS_S
    if length_s < low:
        return "under_{:g}s".format(low)
    if length_s < high:
        return "{:g}_to_{:g}s".format(low, high)
    return "from_{:g}s".format(high)


def _run_table(cell, both_dead, away, keep, fps, s_dead, s_alive):
    """Away share of the runs of one disagreement cell, by kind and length.

    A run is `edge` when a frame next to it is dead for both systems: the two systems then see
    the same stoppage and differ on where it starts or ends. Else the run is `isolated`: only one
    system sees a stoppage there. The camera follows a stoppage with a delay, so the edge runs
    are the weak place of the estimate and the long isolated runs are the strong place.
    """
    n = len(cell)
    rows = {}
    for start, stop in runs_of(cell):
        before = start > 0 and bool(both_dead[start - 1])
        after = stop < n and bool(both_dead[stop])
        key = ("edge" if before or after else "isolated", _bucket((stop - start) / float(fps)))
        row = rows.setdefault(key, {"runs": 0, "frames": 0, "away": 0})
        seen = keep[start:stop]
        row["runs"] += 1
        row["frames"] += int(np.count_nonzero(seen))
        row["away"] += int(np.count_nonzero(away[start:stop] & seen))
    out = []
    for (kind, bucket), row in sorted(rows.items()):
        rate = row["away"] / float(row["frames"]) if row["frames"] else None
        out.append({"kind": kind, "length": bucket, "runs": row["runs"],
                    "frames": row["frames"], "away_share": rate,
                    "dead_share_estimate": mixture(rate, s_dead, s_alive)})
    return out


def camera_referee(x_codes, y_codes, away, covered, graded_runs, fps):
    """The camera referee for two code arrays (H 0, A 1, D 2, X 3) of one match.

    Abstentions are filled as the scorer fills them. The four cells hold the graded frames that
    the broadcast covers. `dead_share_estimate` of a disagreement cell is the estimated share of
    its frames where the ball is dead: mixture() of the away shares of the two agreement cells.
    """
    x = pm.fill_abstentions(np.asarray(x_codes), graded_runs)
    y = pm.fill_abstentions(np.asarray(y_codes), graded_runs)
    away = np.asarray(away, dtype=bool)
    graded = graded_mask(len(x), graded_runs)
    keep = graded & np.asarray(covered, dtype=bool)
    x_dead, y_dead = x == D, y == D
    x_alive, y_alive = (x == H) | (x == A), (y == H) | (y == A)
    cells = {"both_dead": x_dead & y_dead, "both_alive": x_alive & y_alive,
             "x_dead_y_alive": x_dead & y_alive, "x_alive_y_dead": x_alive & y_dead}
    out = {"graded_frames": int(np.count_nonzero(graded)),
           "covered_frames": int(np.count_nonzero(keep)),
           "away_share_covered": share(away, keep)[1], "cells": {}, "runs": {}}
    rates = {}
    for name, cell in cells.items():
        count, rate = share(away, cell & keep)
        rates[name] = rate
        out["cells"][name] = {"frames": count, "away_share": rate}
    s_dead, s_alive = rates["both_dead"], rates["both_alive"]
    for name in ("x_dead_y_alive", "x_alive_y_dead"):
        out["cells"][name]["dead_share_estimate"] = mixture(rates[name], s_dead, s_alive)
        out["runs"][name] = _run_table(cells[name] & graded, cells["both_dead"], away, keep,
                                       fps, s_dead, s_alive)
    return out


def edge_distance(values):
    """int (N,): frames from each frame to the nearest change of `values` (N when none)."""
    values = np.asarray(values)
    n = len(values)
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    if not len(changes):
        return np.full(n, n, dtype=np.int64)
    index = np.arange(n)
    # `right` is the first change at or after the frame. The first frame of a new value is a
    # change, so its distance is 0.
    right = np.searchsorted(changes, index, side="left")
    after = np.where(right < len(changes), changes[np.minimum(right, len(changes) - 1)] - index, n)
    before = np.where(right > 0, index - changes[np.maximum(right - 1, 0)], n)
    return np.minimum(after, before).astype(np.int64)


def tag_referee(x_codes, y_codes, paint, graded_runs, edge_frames):
    """The tag referee for two code arrays and the per-frame tag team (0 H, 1 A, else none).

    Frames count when both systems say alive, a single team is tagged and the nearest change of
    the tag team is at least `edge_frames` away. `tag_accuracy` is the share of tag frames that
    name the common team where the systems agree. `x_right_estimate` is the estimated share of
    the disagreement frames where X names the right team, under the assumption that a tag error
    does not depend on which system is right.
    """
    x = pm.fill_abstentions(np.asarray(x_codes), graded_runs)
    y = pm.fill_abstentions(np.asarray(y_codes), graded_runs)
    paint = np.asarray(paint)
    graded = graded_mask(len(x), graded_runs)
    alive = ((x == H) | (x == A)) & ((y == H) | (y == A))
    tagged = (paint == H) | (paint == A)
    keep = graded & alive & tagged & (edge_distance(paint) >= int(edge_frames))
    agree, differ = keep & (x == y), keep & (x != y)
    n_agree, accuracy = share(paint == x, agree)
    n_differ, for_x = share(paint == x, differ)
    estimate = None
    if accuracy is not None and for_x is not None and accuracy != 0.5:
        # Where X is right the tag names X with probability q, else with 1 - q. So the share of
        # tag frames for X is q * w + (1 - q) * (1 - w). Solve for w.
        estimate = float(min(1.0, max(0.0, (for_x - (1.0 - accuracy)) / (2.0 * accuracy - 1.0))))
    return {"tagged_alive_frames": int(np.count_nonzero(keep)),
            "agree_frames": n_agree, "tag_accuracy": accuracy,
            "differ_frames": n_differ, "tag_share_for_x": for_x,
            "tag_share_for_y": None if for_x is None else 1.0 - for_x,
            "x_right_estimate": estimate}


def _graded_runs(game):
    from engine import data
    return [(int(first), int(last) + 1) for _, first, last in data.graded_blocks(game)]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Label-free referees between two label files.")
    parser.add_argument("referee", choices=("camera", "tags"))
    parser.add_argument("--x", required=True, help="submission CSV of system X")
    parser.add_argument("--y", required=True, help="submission CSV of system Y")
    parser.add_argument("--aligned", help="bb_on_tactical.npz of r2_broadcast_align.py (camera)")
    parser.add_argument("--game", default=None)
    parser.add_argument("--out", default=None, help="write the JSON here, else print it")
    args = parser.parse_args(argv)

    game = cm.game_name(args.game)
    runs = _graded_runs(game)
    x, y = cm.read_results(args.x, game), cm.read_results(args.y, game)
    if args.referee == "camera":
        if not args.aligned:
            parser.error("the camera referee needs --aligned")
        with np.load(args.aligned) as npz:
            away, covered = camera_away(npz["valid"], npz["replay"], npz["b_frame"])
        result = camera_referee(x, y, away, covered, runs, cm.fps(game))
    else:
        from engine import data
        from tools import sanity
        # The phases whose team is the team in possession: the list of tools/sanity.py.
        paint = cm.tag_team_codes(data.load_tagging(game), len(x), sanity.TAG_PHASE_CODES, runs)
        result = tag_referee(x, y, paint, runs, int(round(TAG_EDGE_S * cm.fps(game))))
    result.update(referee=args.referee, game=game, x=cm.result_name(args.x),
                  y=cm.result_name(args.y))
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
