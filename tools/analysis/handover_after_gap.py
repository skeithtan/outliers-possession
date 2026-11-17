"""How many engine hand-overs land on the first contact after a long ball gap (P9-D).

    python tools/analysis/handover_after_gap.py [--game NAME] [--config PATH]

CODE-REVIEW-RESPONSE finding 3 asks how much the owner memory is exposed: the engine holds the
last owner through a ball gap, so a turnover that happens unseen inside the gap shows up at the
first contact after it. This script sizes that exposure. It designs nothing.

Definitions:

- A hand-over is a change from H to A or from A to H in the engine output, inside a graded
  block. A team change across a D segment is a restart, not a hand-over.
- A long gap is a run of frames without a ball in the ATD that is longer than LONG_GAP_S and
  lies inside one graded block (PLAN 1.1, the gap of tools/sanity.py).
- A contact is a row of the engine's event table, as the decoder reads it: every kind.
- The first contact after a gap is the first event row at or after the first frame with a ball.
- A hand-over lands on a contact when the contact is the nearest event row to the hand-over and
  lies within SWITCH_EVENT_WINDOW frames of it (the cheap switch of the decoder).

The script runs the engine and prints summary statistics only: counts, shares, and the gap
lengths in seconds (median, p90, max), in total and by gap length. It reads no ground truth.
For METRICA-S3, set MTAH_DATA_DIR to the data folder of the sample.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from engine import data, possession  # noqa: E402
from engine.config import Config, load  # noqa: E402

LONG_GAP_S = 2.0                         # PLAN 1.1 and tools/sanity.py
# Gap length buckets in seconds, the edges of PLAN 1.1 (median 11 s, p90 34 s).
BUCKETS_S = ((2.0, 5.0), (5.0, 10.0), (10.0, 30.0), (30.0, float("inf")))


def long_gaps(ball_missing, blocks, min_frames):
    """[(start, stop)] half-open runs without a ball, longer than min_frames, inside one block."""
    out = []
    for _, first, last in blocks:
        miss = np.asarray(ball_missing[first:last + 1], dtype=np.int8)
        edges = np.diff(np.concatenate(([0], miss, [0])))
        starts = np.flatnonzero(edges == 1) + first
        stops = np.flatnonzero(edges == -1) + first
        out += [(int(a), int(b)) for a, b in zip(starts, stops) if b - a > min_frames]
    return out


def hand_overs(labels, blocks):
    """Frames of the H <-> A changes of the labels inside each block (the first frame after)."""
    labels = np.asarray(labels)
    out = []
    for _, first, last in blocks:
        seg = labels[first:last + 1]
        idx = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        out += [first + int(i) for i in idx if {seg[i - 1], seg[i]} == {"H", "A"}]
    return out


def measure(labels, ball_missing, event_frames, blocks, fps, window):
    """The exposure table as a dict (module docstring)."""
    gaps = long_gaps(ball_missing, blocks, int(round(LONG_GAP_S * fps)))
    events = np.unique(np.asarray(event_frames, dtype=np.int64))
    labels = np.asarray(labels)
    first_contact = {}                    # event frame -> (gap start, gap stop), sorted gaps
    for start, stop in gaps:
        block_last = next(last for _, first, last in blocks if first <= start <= last)
        later = events[(events >= stop) & (events <= block_last)]
        if later.size:
            first_contact[int(later[0])] = (start, stop)   # the gap just before wins
    switches = hand_overs(labels, blocks)
    landed = []
    for frame in switches:
        if not events.size:
            break
        pos = int(np.argmin(np.abs(events - frame)))
        nearest = int(events[pos])
        if abs(nearest - frame) <= window and nearest in first_contact:
            landed.append((frame, nearest, first_contact[nearest]))
    alive_gaps = [(a, b) for a, b in gaps if not np.any(labels[a:b] == "D")]

    def lengths(rows):
        return np.array([(b - a) / float(fps) for a, b in rows], dtype=np.float64)

    def stats(values):
        if not values.size:
            return {"n": 0, "median_s": None, "p90_s": None, "max_s": None}
        return {"n": int(values.size), "median_s": float(np.median(values)),
                "p90_s": float(np.percentile(values, 90)), "max_s": float(values.max())}

    landed_gaps = [gap for _, _, gap in landed]
    alive_set = set(alive_gaps)
    landed_alive = [gap for gap in landed_gaps if gap in alive_set]
    delays = np.array([(c - gap[1]) / float(fps) for _, c, gap in landed], dtype=np.float64)
    buckets = []
    for lo, hi in BUCKETS_S:
        def inside(rows):
            return [r for r in rows if lo < (r[1] - r[0]) / float(fps) <= hi]
        buckets.append({"bucket": "{:g}-{:g} s".format(lo, hi) if np.isfinite(hi)
                        else "> {:g} s".format(lo),
                        "gaps": len(inside(gaps)), "alive_gaps": len(inside(alive_gaps)),
                        "hand_overs": len(inside(landed_gaps)),
                        "hand_overs_alive": len(inside(landed_alive)),
                        "frames": int(sum(b - a for a, b in inside(landed_gaps)))})
    return {
        "hand_overs": len(switches),
        "long_gaps": len(gaps), "long_gap_lengths": stats(lengths(gaps)),
        "alive_gaps": len(alive_gaps), "alive_gap_lengths": stats(lengths(alive_gaps)),
        "landed": len(landed), "landed_alive": len(landed_alive),
        "landed_share": len(landed) / float(len(switches)) if switches else None,
        "landed_gap_lengths": stats(lengths(landed_gaps)),
        "landed_gap_frames": int(sum(b - a for a, b in landed_gaps)),
        "contact_delay_median_s": float(np.median(delays)) if delays.size else None,
        "buckets": buckets,
    }


def _f(value, fmt="{:.1f}"):
    return "-" if value is None else fmt.format(value)


def format_lines(game, table):
    """Printable summary lines."""
    def stat_line(name, s):
        return "  {:<34} n {:>4}  median {:>6} s  p90 {:>6} s  max {:>6} s".format(
            name, s["n"], _f(s["median_s"]), _f(s["p90_s"]), _f(s["max_s"]))

    lines = ["{}: hand-overs (H <-> A) {}  long ball gaps (> {:g} s) {}".format(
        game, table["hand_overs"], LONG_GAP_S, table["long_gaps"]),
        stat_line("long gaps", table["long_gap_lengths"]),
        stat_line("gaps labelled alive throughout", table["alive_gap_lengths"]),
        stat_line("gaps whose first contact switches", table["landed_gap_lengths"]),
        "  hand-overs on the first contact after a long gap: {} ({} of all hand-overs), "
        "{} gap frames, median delay from the ball's return to the contact {} s".format(
            table["landed"], _f(None if table["landed_share"] is None
                                else 100.0 * table["landed_share"], "{:.1f}%"),
            table["landed_gap_frames"], _f(table["contact_delay_median_s"])),
        "  of these, after a gap labelled alive throughout: {}, after a gap that holds D: {}"
        .format(table["landed_alive"], table["landed"] - table["landed_alive"]),
        "  {:<10} {:>6} {:>12} {:>12} {:>18} {:>12}".format(
            "gap", "gaps", "alive gaps", "hand-overs", "hand-overs alive", "gap frames")]
    for b in table["buckets"]:
        lines.append("  {:<10} {:>6} {:>12} {:>12} {:>18} {:>12}".format(
            b["bucket"], b["gaps"], b["alive_gaps"], b["hand_overs"], b["hand_overs_alive"],
            b["frames"]))
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default=None, choices=sorted(data.games.GAMES),
                        help="match of games.py (default: games.GAME)")
    parser.add_argument("--config", default=None, help="engine config file (default: Config)")
    args = parser.parse_args(argv)
    game = data.resolve_game(args.game)
    cfg = load(args.config) if args.config else Config.default()
    labels, parts = possession.run(cfg, return_parts=True, game=game)
    tactical = data.load_tactical(game)
    table = measure(labels, np.isnan(tactical["bx"]), parts["events"]["frame"].to_numpy(),
                    data.graded_blocks(game), data.fps(game), int(cfg.SWITCH_EVENT_WINDOW))
    print("\n".join(format_lines(game, table)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
