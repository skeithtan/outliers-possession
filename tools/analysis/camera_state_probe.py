"""R-cam (RESEARCH-6 3.5): one submission's labels against the broadcast camera state.

    python tools/analysis/camera_state_probe.py c22.csv --aligned DIR/bb_on_tactical.npz

The camera state comes from tools/analysis/r2_broadcast_align.py (broadcast ATD tracking and
homography only; no video): a covered tactical frame is `away` when the broadcast frame has no
valid homography or is replay-flagged, else `live wide`. Prints the contingency of the labels
(D vs H/A) against the camera state on graded, covered frames, and the runs where the labels say
D while the camera stays on a live wide shot, by length. Reads no ground truth. Python 3.9.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (ROOT, os.path.join(ROOT, "tools", "analysis")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from engine import data  # noqa: E402
import referee  # noqa: E402

LABELS = "HADX"


def read_labels(path, n_frames):
    codes = np.full(n_frames, 3, dtype=np.int8)
    with open(path) as handle:
        next(handle)
        for line in handle:
            frame, label = line.strip().split(",")
            codes[int(frame)] = LABELS.index(label)
    return codes


def run_table(mask, fps, label, min_list_s=10.0):
    runs = referee.runs_of(mask)
    lens = np.array([e - s for s, e in runs], dtype=float) / fps
    print("{}: runs {} total {:.1f} s".format(label, len(runs), lens.sum() if len(runs) else 0.0))
    for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)):
        sel = (lens >= lo) & (lens < hi)
        print("  {:>4g}-{:<4g} s: runs {:4d} total {:7.1f} s".format(lo, min(hi, 999), sel.sum(),
                                                                   lens[sel].sum() if sel.any() else 0.0))
    long_runs = [(s, e) for (s, e), l in zip(runs, lens) if l >= min_list_s]
    return runs, long_runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels")
    parser.add_argument("--aligned", required=True)
    parser.add_argument("--game", default=None)
    parser.add_argument("--list", type=int, default=40)
    args = parser.parse_args(argv)
    game = data.resolve_game(args.game)
    n_frames, fps = data.frame_count(game), data.fps(game)
    codes = read_labels(args.labels, n_frames)
    with np.load(args.aligned) as npz:
        away, covered = referee.camera_away(npz["valid"], npz["replay"], npz["b_frame"])
        replay = np.asarray(npz["replay"], dtype=bool)
        valid = np.asarray(npz["valid"], dtype=bool)
    graded = np.zeros(n_frames, dtype=bool)
    half_of = np.zeros(n_frames, dtype=np.int8)
    for h, lo, hi in data.graded_blocks(game):
        graded[lo:hi + 1] = True
        half_of[lo:hi + 1] = h
    keep = graded & covered
    live = keep & ~away
    print("labels {} | graded {} | covered {} | live wide {} ({:.1f}%) | away {} (no homography {}, replay-flagged {})".format(
        os.path.basename(args.labels), graded.sum(), keep.sum(), live.sum(), 100.0 * live.sum() / keep.sum(),
        (keep & away).sum(), (keep & ~valid).sum(), (keep & valid & replay).sum()))
    dead = codes == 2
    alive = (codes == 0) | (codes == 1)
    print("\n== contingency on graded covered frames ==")
    print("{:>14s} {:>10s} {:>10s} {:>8s}".format("", "live wide", "away", "row %live"))
    for name, m in (("label D", dead), ("label H/A", alive), ("label X", codes == 3)):
        a, b = (keep & m & ~away).sum(), (keep & m & away).sum()
        print("{:>14s} {:10d} {:10d} {:8.1f}".format(name, a, b, 100.0 * a / max(a + b, 1)))
    print("P(D | live wide) {:.3f} | P(D | away) {:.3f} | P(away | D) {:.3f} | P(away | H/A) {:.3f}".format(
        (dead & live).sum() / live.sum(), (dead & keep & away).sum() / (keep & away).sum(),
        (dead & keep & away).sum() / (dead & keep).sum(), (alive & keep & away).sum() / (alive & keep).sum()))

    print("\n== runs: label D while the camera is on a live wide shot (consecutive frames) ==")
    _, long_d = run_table(dead & live, fps, "D & live wide", 10.0)
    print("runs >= 10 s: {} | total {:.1f} s".format(len(long_d), sum(e - s for s, e in long_d) / fps))
    # Context of the long runs: is the run the whole D segment, or part of one whose other frames
    # are away (the camera cut later / returned earlier)?
    d_runs = referee.runs_of(dead & graded)
    starts = np.array([s for s, _ in d_runs])
    for s, e in long_d[:args.list]:
        k = np.searchsorted(starts, s, side="right") - 1
        ds, de = d_runs[k]
        seg_away = (away & keep)[ds:de].sum()
        print("  D&live {:6d}-{:6d} ({:5.1f} s, half {}) inside D segment {:6d}-{:6d} ({:5.1f} s) whose away frames {} ({:.0f}%)".format(
            s, e - 1, (e - s) / fps, half_of[s], ds, de - 1, (de - ds) / fps, seg_away,
            100.0 * seg_away / (de - ds)))
    # D segments with no away frame at all.
    whole = [(ds, de) for ds, de in d_runs if (away & keep)[ds:de].sum() == 0 and keep[ds:de].all()]
    print("D segments {} | fully covered and never away: {} (total {:.1f} s; >= 10 s: {})".format(
        len(d_runs), len(whole), sum(de - ds for ds, de in whole) / fps,
        sum(1 for ds, de in whole if de - ds >= 10 * fps)))

    print("\n== runs: label H/A while the camera is away (the reverse cell) ==")
    _, long_a = run_table(alive & keep & away, fps, "H/A & away", 10.0)
    print("runs >= 10 s: {} | total {:.1f} s".format(len(long_a), sum(e - s for s, e in long_a) / fps))
    for s, e in long_a[:args.list]:
        print("  H/A&away {:6d}-{:6d} ({:5.1f} s, half {}) replay-flagged {:.0f}%".format(
            s, e - 1, (e - s) / fps, half_of[s], 100.0 * replay[s:e].mean()))

    print("\n== timing: camera cut after a D segment start, and return before its end ==")
    lags, leads = [], []
    for ds, de in d_runs:
        if not keep[ds:de].all():
            continue
        aw = np.flatnonzero((away & keep)[ds:de])
        if len(aw):
            lags.append(aw[0] / fps)
            leads.append((de - ds - 1 - aw[-1]) / fps)
    if lags:
        print("D segments fully covered {} | first away frame after D start (s): p25 {:.1f} median {:.1f} p75 {:.1f}".format(
            len(lags), *np.quantile(lags, (0.25, 0.5, 0.75))))
        print("  last away frame before D end (s): p25 {:.1f} median {:.1f} p75 {:.1f}".format(
            *np.quantile(leads, (0.25, 0.5, 0.75))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
