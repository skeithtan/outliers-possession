"""P1.1 (RESEARCH-6 3.4): the ball speed channel of the tactical ATD, read directly.

    python tools/analysis/ball_speed_probe.py [--labels c22.csv] [--game BAR-ATH]

engine.data._parse_atd_raw drops the third ball value. This tool reads it from the raw .txt and
reports where it is finite, where those frames sit relative to the ball-position gaps, its unit
(ratio to the position-derived speed where both exist), and how it agrees with the labels of
one submission inside the gaps. Summary statistics only: no per-frame rows are printed. Reads no
ground truth and no video. Python 3.9, numpy only.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from engine import data, tracks  # noqa: E402

LABEL_CODE = {"H": 0, "A": 1, "D": 2, "X": 3}
Q = (0.05, 0.25, 0.5, 0.75, 0.95)


def read_ball_columns(raw_path, n_frames):
    """(bx, by, bs) float64 (n_frames,) from the raw ATD: the three ball values of each line."""
    out = np.full((n_frames, 3), np.nan)
    with open(raw_path, "r") as handle:
        for i, line in enumerate(handle):
            ball = line.rstrip("\n").rsplit(":", 1)[1].split(",")
            for j, value in enumerate(ball[:3]):
                value = value.strip()
                if value:
                    out[i, j] = float(value)
    return out[:, 0], out[:, 1], out[:, 2]


def runs_of(mask):
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def quant(values, qs=Q):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return "n=0"
    return "n={} ".format(len(values)) + " ".join(
        "p{:02d}={:.2f}".format(int(q * 100), v) for q, v in zip(qs, np.quantile(values, qs)))


def read_labels(path, n_frames):
    codes = np.full(n_frames, 3, dtype=np.int8)
    with open(path) as handle:
        next(handle)
        for line in handle:
            frame, label = line.strip().split(",")
            codes[int(frame)] = LABEL_CODE[label]
    return codes


def speed_bins(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if not n:
        return "n=0"
    edges = [0.5, 1.0, 2.0, 5.0, 10.0]
    parts = ["<{:g}: {}".format(edges[0], int((values < edges[0]).sum()))]
    for lo, hi in zip(edges[:-1], edges[1:]):
        parts.append("{:g}-{:g}: {}".format(lo, hi, int(((values >= lo) & (values < hi)).sum())))
    parts.append(">={:g}: {}".format(edges[-1], int((values >= edges[-1]).sum())))
    return "n={} ".format(n) + " | ".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", default=None, help="submission CSV to compare inside gaps")
    parser.add_argument("--game", default=None)
    args = parser.parse_args(argv)
    game = data.resolve_game(args.game)

    meta_path, raw_path = data.require(["atd_meta", "atd_raw"], game)
    meta = data._atd_meta(meta_path, game)
    n_frames = data.frame_count(game)
    fps = data.fps(game)
    L, W = data.pitch_m(game)
    blocks = data.graded_blocks(game)
    print("game {} frames {} fps {} ball channels {}".format(game, n_frames, fps,
                                                             meta["ball_channels"]))
    bx, by, bs = read_ball_columns(raw_path, n_frames)
    T = data.load_tactical(game)
    assert np.array_equal(np.isfinite(bx), np.isfinite(T["bx"])), "raw bx differs from loader"

    has_pos = np.isfinite(bx) & np.isfinite(by)
    fin = np.isfinite(bs)
    half_of = np.zeros(n_frames, dtype=np.int8)
    for h, lo, hi in blocks:
        half_of[lo:hi + 1] = h
    print("\n== 1. where the channel is finite ==")
    print("finite speed frames {} | with position {} | without position {}".format(
        fin.sum(), (fin & has_pos).sum(), (fin & ~has_pos).sum()))
    for h in (1, 2, 0):
        sel = half_of == h
        name = "half {}".format(h) if h else "outside halves"
        print("  {:15s} frames {:6d} | finite {:6d} | finite+pos {:6d} | finite no-pos {:6d} | "
              "no-pos total {:6d}".format(name, sel.sum(), (fin & sel).sum(),
                                           (fin & has_pos & sel).sum(),
                                           (fin & ~has_pos & sel).sum(), (~has_pos & sel).sum()))
    print("ball position present on {} frames ({:.1f}%); finite speed on {:.1f}% of those".format(
        has_pos.sum(), 100.0 * has_pos.mean(), 100.0 * (fin & has_pos).sum() / max(has_pos.sum(), 1)))
    print("value is exactly 0 on {} finite frames ({} with pos, {} without)".format(
        (fin & (bs == 0)).sum(), (fin & has_pos & (bs == 0)).sum(), (fin & ~has_pos & (bs == 0)).sum()))

    print("\n== 2. finite-speed frames without position, relative to raw position gaps ==")
    gaps = runs_of(~has_pos)
    classes = {"whole": [0, 0], "start": [0, 0], "end": [0, 0], "both_ends": [0, 0],
               "interior": [0, 0], "none": [0, 0]}
    len_with, len_without = [], []
    gap_rows = []  # (start, stop, class, n_finite)
    for start, stop in gaps:
        f = fin[start:stop]
        n = int(f.sum())
        if n == 0:
            cls = "none"
            len_without.append(stop - start)
        else:
            len_with.append(stop - start)
            sub = runs_of(f)
            if n == stop - start:
                cls = "whole"
            elif len(sub) == 1 and sub[0][0] == 0:
                cls = "start"
            elif len(sub) == 1 and sub[0][1] == stop - start:
                cls = "end"
            elif len(sub) == 2 and sub[0][0] == 0 and sub[1][1] == stop - start:
                cls = "both_ends"
            else:
                cls = "interior"
        classes[cls][0] += 1
        classes[cls][1] += n
        gap_rows.append((start, stop, cls, n))
    print("raw position gaps: {} (frames {}); gaps with any finite speed: {}".format(
        len(gaps), (~has_pos).sum(), len(len_with)))
    for cls, (g, f) in classes.items():
        print("  {:10s} gaps {:5d} finite-speed frames {:6d}".format(cls, g, f))
    print("gap length (frames) with finite speed:    " + quant(len_with))
    print("gap length (frames) without finite speed: " + quant(len_without))
    # Where in a gap does a finite value sit: the run lengths of finite frames at the gap start
    # and end.
    lead, trail = [], []
    for start, stop, cls, n in gap_rows:
        if cls in ("start", "both_ends", "whole"):
            lead.append(runs_of(fin[start:stop])[0][1])
        if cls in ("end", "both_ends"):
            r = runs_of(fin[start:stop])[-1]
            trail.append(r[1] - r[0])
    print("finite run at a gap start (frames): " + quant(lead))
    print("finite run at a gap end (frames):   " + quant(trail))
    # Short gaps (the loader's BALL_MIN_RUN / clean_ball do not change raw gaps, but gaps under
    # 1 s are tracker flicker): share of finite-speed no-pos frames in gaps under 25 frames.
    short = sum(n for s, e, c, n in gap_rows if e - s < fps)
    print("finite no-pos frames in gaps shorter than 1 s: {} of {}".format(short, (fin & ~has_pos).sum()))

    print("\n== 3. value quantiles (channel units as written) ==")
    print("with position:    " + quant(bs[fin & has_pos]))
    print("without position: " + quant(bs[fin & ~has_pos]))
    print("with position, bins:    " + speed_bins(bs[fin & has_pos]))
    print("without position, bins: " + speed_bins(bs[fin & ~has_pos]))

    print("\n== 4. unit: ratio to the position-derived speed (central difference, m/s) ==")
    xm, ym = bx * L, by * W
    derived = np.full(n_frames, np.nan)
    both = has_pos[2:] & has_pos[:-2]
    d = np.hypot(xm[2:] - xm[:-2], ym[2:] - ym[:-2]) * fps / 2.0
    derived[1:-1] = np.where(both, d, np.nan)
    ok = fin & np.isfinite(derived) & (derived > 1.0) & (bs > 0)
    ratio = bs[ok] / derived[ok]
    print("frames with both: {} (derived > 1 m/s, channel > 0)".format(ok.sum()))
    print("ratio channel/derived: " + quant(ratio))
    print("  km/h hypothesis predicts 3.6; m/s predicts 1.0")
    for lo in (1.0, 3.0, 8.0):
        sel = ok & (derived > lo)
        print("  derived > {:g} m/s: n={} median ratio {:.3f}".format(
            lo, sel.sum(), np.median(bs[sel] / derived[sel]) if sel.any() else np.nan))
    # Lag: is the channel a forward or backward difference? Compare with one-sided speeds.
    fwd = np.full(n_frames, np.nan)
    fwd[:-1] = np.where(has_pos[1:] & has_pos[:-1], np.hypot(xm[1:] - xm[:-1], ym[1:] - ym[:-1]) * fps, np.nan)
    bwd = np.full(n_frames, np.nan)
    bwd[1:] = fwd[:-1]
    for name, ref in (("central", derived), ("forward", fwd), ("backward", bwd)):
        sel = fin & np.isfinite(ref) & (ref > 1.0) & (bs > 0)
        r = bs[sel] / ref[sel]
        for scale, unit in ((1.0, "m/s"), (3.6, "km/h")):
            within = np.abs(r / scale - 1.0) < 0.1
            print("  {:8s} ref, {:4s}: n={} share within 10% {:.3f}".format(name, unit, sel.sum(), within.mean()))

    print("\n== 4b. is the channel a speed at all: rank correlation with the derived speed ==")

    def spearman(u, v):
        ru = np.argsort(np.argsort(u)).astype(float)
        rv = np.argsort(np.argsort(v)).astype(float)
        return float(np.corrcoef(ru, rv)[0, 1])

    sel = fin & np.isfinite(derived)
    print("frames with both finite: {} | spearman(channel, derived) {:.3f} | pearson {:.3f}".format(
        sel.sum(), spearman(bs[sel], derived[sel]), float(np.corrcoef(bs[sel], derived[sel])[0, 1])))
    best = []
    for lag in range(-100, 101, 5):
        shifted = np.roll(derived, lag)
        ok2 = fin & np.isfinite(shifted)
        if ok2.sum() > 100:
            best.append((spearman(bs[ok2], shifted[ok2]), lag))
    best.sort(reverse=True)
    print("best lags (spearman, lag frames; positive = channel leads positions): {}".format(
        ", ".join("{:.3f}@{:+d}".format(r, lag) for r, lag in best[:5])))
    # Channel value by derived-speed band, with position: a speed would rise with the band.
    for lo, hi in ((0, 1), (1, 3), (3, 8), (8, 15), (15, 60)):
        band = sel & (derived >= lo) & (derived < hi)
        if band.any():
            print("  derived {:2d}-{:2d} m/s: n={:5d} channel median {:.2f} p90 {:.2f}".format(
                lo, hi, band.sum(), np.median(bs[band]), np.quantile(bs[band], 0.9)))
    # Smoothness: a tracked quantity changes little frame to frame.
    step = np.abs(np.diff(bs))
    both_fin = fin[1:] & fin[:-1]
    print("frame-to-frame |delta| on consecutive finite frames: " + quant(step[both_fin]))
    print("channel finite runs (frames): " + quant([e - s for s, e in runs_of(fin)]))

    cleaned = tracks.clean_ball(T["bx"], T["by"], (L, W), fps=fps)
    vis = np.asarray(cleaned["visible"], dtype=bool)
    print("\n== 5. engine (cleaned) ball gaps vs raw gaps ==")
    print("cleaned invisible frames {} | raw no-pos {} | cleaned-invisible with raw position {}".format(
        (~vis).sum(), (~has_pos).sum(), (~vis & has_pos).sum()))
    print("finite speed on cleaned-invisible frames: {}".format((fin & ~vis).sum()))

    if args.labels:
        codes = read_labels(args.labels, n_frames)
        graded = half_of > 0
        print("\n== 6. agreement with the labels of {} inside ball gaps (graded frames) ==".format(
            os.path.basename(args.labels)))
        for gap_name, gap_mask in (("raw no-position", ~has_pos), ("cleaned invisible", ~vis)):
            print("-- {} gaps --".format(gap_name))
            inside = gap_mask & graded
            for lab, code in (("D", 2), ("H/A", None)):
                sel = inside & ((codes == 2) if code == 2 else ((codes == 0) | (codes == 1)))
                f = sel & fin
                print("  {:3s} frames in gaps {:6d} | finite speed {:6d} ({:.1f}%)".format(
                    lab, sel.sum(), f.sum(), 100.0 * f.sum() / max(sel.sum(), 1)))
                print("      values: " + quant(bs[f]))
                print("      bins:   " + speed_bins(bs[f]))
        # Per gap: gaps (raw) longer than 2 s that hold finite-speed frames, with the label
        # majority and the median channel value on the finite frames.
        print("-- raw gaps >= 2 s with finite speed frames, by label majority in the gap --")
        table = {}
        for start, stop, cls, n in gap_rows:
            if n == 0 or stop - start < 2 * fps or not graded[start:stop].all():
                continue
            seg = codes[start:stop]
            d_share = float((seg == 2).mean())
            maj = "D" if d_share >= 0.5 else "H/A"
            med = float(np.nanmedian(bs[start:stop]))
            moving = med >= 2.0
            key = (maj, "moving(median>=2)" if moving else "still(median<2)")
            row = table.setdefault(key, [0, 0, []])
            row[0] += 1
            row[1] += n
            row[2].append(stop - start)
        for key, (g, f, lens) in sorted(table.items()):
            print("  label {:3s} channel {:18s}: gaps {:4d} finite frames {:6d} gap len {}".format(
                key[0], key[1], g, f, quant(lens, (0.25, 0.5, 0.75))))
        # The first finite value inside gaps that the labels call D: at the gap start, is the
        # ball still or moving (a stoppage with a moving ball is a flight the tracker lost).
        first_vals = []
        for start, stop, cls, n in gap_rows:
            if n == 0 or not graded[start:stop].all():
                continue
            if (codes[start:stop] == 2).mean() >= 0.5:
                idx = np.flatnonzero(fin[start:stop])
                first_vals.append(bs[start + idx[0]])
        print("first finite value in D-majority graded gaps: " + speed_bins(first_vals))
    return 0


if __name__ == "__main__":
    sys.exit(main())
