"""r2 task 4 support: the kick-exit timing that the chain confirmation (variant H) needs.

    python tools/analysis/r2_restart_timing.py

(1) DELTA_RESTART_KICK scan with confirmation H: public restart latencies, matched transitions,
    per-half D share, D segments, possessions. (2) H with each kick exit at the end of its rise
    (peak = first local maximum of the speed after the rise start; possession.shift_stoppage is
    monkeypatched in this process only). (3) rise length (peak - candidate) over the kick exits.
(4) LOO: the CHAIN interval that leaves each public restart / re-play decision unchanged.
(5) exits whose stoppage entry is an `out` onset on the same frame as a reappear candidate.
Deterministic. Reads the public GT only.
"""

import dataclasses
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_restart_common as c  # noqa: E402
import r2_restart_variants as rv  # noqa: E402
from engine import possession  # noqa: E402
from engine.config import Config  # noqa: E402
from engine.tracks import _runs  # noqa: E402


def summary(inp, cfg, conf, name):
    lab, parts = c.run_with_confirmed(inp, cfg, conf)
    pairs, missed, spur, res = c.public_report(lab)
    pub = {g[0]: p[0] - g[0] for g, p in pairs}
    import sanity
    st = sanity.sanity_stats(lab)
    h = st["halves"]
    print("%-14s lat 12989 %s 16400 %s 17838 %s 18389 %s | matched %d spellF1 %.3f macroF1 %.3f "
          "| D1 %.2f D2 %.2f Dseg %d poss %d %s" % (
              name, pub.get(12989), pub.get(16400), pub.get(17838), pub.get(18389), len(pairs),
              res["spell_f1_mean"], res["macro_f1"], 100 * h[1]["d_share"],
              100 * h[2]["d_share"], st["total"]["d_segments"], st["total"]["possessions"],
              st["violations"]))
    return lab


def peak_shift_stoppage(peak_delta):
    """shift_stoppage replacement: kick exits move by peak_delta[frame], reappear as configured."""
    orig = possession.shift_stoppage

    def patched(stoppage, restart_list, onsets, candidates, cfg, blocks, kickoffs):
        confirmed = candidates[candidates["confirmed"]]
        exit_delta = {}
        for frame, kind in zip(confirmed["frame"], confirmed["kind"]):
            d = peak_delta.get(int(frame), 0) if kind == "kick" else int(
                cfg.DELTA_RESTART_REAPPEAR)
            exit_delta.setdefault(int(frame), d)
        kick_frames = set(int(f) for f in kickoffs.values())
        out = np.zeros_like(np.asarray(stoppage, dtype=bool))
        moved = {}
        for _, first, last in blocks:
            starts, stops = _runs(np.asarray(stoppage[first:last + 1], dtype=bool))
            for start, stop in zip(starts + first, stops + first):
                start, stop = int(start), int(stop)
                if stop in exit_delta and stop not in kick_frames and stop <= last:
                    new = int(np.clip(stop + exit_delta[stop], first, last + 1))
                    moved[stop] = new
                    stop = new
                if stop > start:
                    out[start:stop] = True
        return out, [(moved.get(int(f), int(f)), t) for f, t in restart_list]
    return orig, patched


def main():
    cfg = Config.default()
    inp = c.load(cfg)
    vs = rv.variants(inp, cfg)
    print("== (1) DELTA_RESTART_KICK scan")
    for name in ("cur", "H"):
        for dk in (0, 9, 12, 16, 20, 25, 32):
            summary(inp, dataclasses.replace(cfg, DELTA_RESTART_KICK=dk), vs[name],
                    "%s k%+d" % (name, dk))
    frames = inp["restarts"]["frame"].to_numpy()
    speed = inp["ball"]["speed"]
    peak_delta = {int(f): int(c.peak_end(speed, int(f)) - f) for f, k in
                  zip(frames, inp["restarts"]["kind"]) if k == "kick"}
    print("\n== (2) kick exit at the end of the rise")
    orig, patched = peak_shift_stoppage(peak_delta)
    possession.shift_stoppage = patched
    try:
        for name in ("cur", "H"):
            summary(inp, cfg, vs[name], "%s peak" % name)
    finally:
        possession.shift_stoppage = orig
    feat = pd.read_csv(os.path.join(c.SCRATCH, "features.csv"))
    kex = feat[(feat["cls"] == "exit") & (feat["kind"] == "kick")]["frame"].astype(int)
    rise = np.array([peak_delta[f] for f in kex])
    print("\n== (3) rise length (peak - candidate) over %d kick exits: p10/p25/p50/p75/p90 %s; "
          "16393 -> %d, 18355 -> %d" % (len(rise), np.percentile(rise, [10, 25, 50, 75, 90]),
                                         peak_delta[16393], peak_delta[18355]))
    print("\n== (4) CHAIN bounds from public cases")
    conf_frames = frames[vs["cur"]]
    for f in (18355, 16058, 17425):
        nxt = conf_frames[conf_frames > f][0]
        print("  %d: next players-confirmed candidate %d (+%d frames)" % (f, nxt, nxt - f))
    rej = feat[feat["cls"] == "confirm_rej"]["frame"].astype(int).to_numpy()
    gaps = sorted(int(conf_frames[conf_frames > f][0] - f) for f in rej
                  if (conf_frames > f).any())
    print("  all %d confirmation rejects, gap to the next confirmed candidate (sorted): %s" % (
        len(rej), gaps))
    print("  alive-play medians (r2_restart_features / this study): kick-candidate gap 87, "
          "release gap 93.5 frames")
    print("\n== (5) exits whose entry is an out onset on a reappear-candidate frame")
    cand, dead, rl, iv = c.classify(inp, cfg)
    on = inp["onsets"]
    outs = set(on[on["mechanism"] == "out"]["frame"].astype(int))
    reap = set(inp["restarts"][inp["restarts"]["kind"] == "reappear"]["frame"].astype(int))
    hits = [(e, l) for e, l, r in iv if r >= 0 and e in outs and (e in reap or e - 1 in reap
                                                                    or e + 1 in reap)]
    print("  n=%d; D lengths (frames): %s" % (len(hits), sorted(l - e for e, l in hits)))
    print("  %s" % hits)


if __name__ == "__main__":
    main()
