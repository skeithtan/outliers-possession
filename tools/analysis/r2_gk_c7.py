"""C7 measurement: slot confidence and attribution margin of the engine's team events.

For every touch / release / control / dribble row with a team on graded alive frames (engine
stoppage mask off) of the current engine: slot confidence of the row team's nearest slot at the
row frame, and the margin d_other - d_own (m) at the row frame (feats dH / dA; for a touch the
engine measures on the frame of the team inside [c, t + k], so this is the contact-frame value).
Then the same for the public switches: the causing row = the last team row of the new team in
[f - 50, f]. Also the number of new-team touch/release rows within 2 s before the switch.
Usage: python tools/analysis/r2_gk_c7.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ENGINE_ROOT = os.environ.get("R2_ENGINE_ROOT", ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, ENGINE_ROOT)
from engine import data, possession, tracks  # noqa: E402
print("engine from", os.path.dirname(possession.__file__))
from engine.config import Config  # noqa: E402

cfg = Config.default()
inputs = possession.prepare(cfg)
labels, parts = possession.run(cfg, return_parts=True, inputs=inputs)
labels = np.asarray(labels)
tac = data.load_tactical()
conf = tracks.slot_confidence(tac, cfg)
feats = inputs["feats"]
ev = inputs["events"]
graded = np.zeros(len(labels), bool)
for _, lo, hi in data.graded_blocks():
    graded[lo:hi + 1] = True
alive = graded & ~parts["stoppage"]
KINDS = ("touch", "release", "control", "dribble")
rows = ev[ev["kind"].isin(KINDS) & ev["team"].isin(["H", "A"])].copy()
rows = rows[alive[rows["frame"].to_numpy()]]


def values(f, t):
    own = feats["d" + t][f]
    oth = feats["d" + ("A" if t == "H" else "H")][f]
    s = feats["nearest_slot_" + t][f]
    return (float(conf[f, s]) if s >= 0 else np.nan), (oth - own if np.isfinite(oth) else np.inf), own


vals = np.array([values(int(f), t) for f, t in zip(rows["frame"], rows["team"])])
rows["slot_conf"], rows["margin"], rows["d_own"] = vals[:, 0], vals[:, 1], vals[:, 2]
print("config hash", cfg.config_hash())
print("rows on graded alive frames: %d" % len(rows))
for k in KINDS + ("all",):
    r = rows if k == "all" else rows[rows["kind"] == k]
    c = r["slot_conf"].to_numpy()
    m = r["margin"].to_numpy()
    mf = m[np.isfinite(m)]
    print("  %-8s n=%4d | slot conf: 1.0 %.1f%%, 0.5 %.1f%%, 0.25 %.1f%%, < 0.8 %.1f%% | margin p10 %.2f p25 %.2f p50 %.2f p90 %.2f m, inf %d, < 0.4 m %.1f%%"
          % (k, len(r), 100 * np.mean(c == 1.0), 100 * np.mean(np.isclose(c, 0.5)), 100 * np.mean(np.isclose(c, 0.25)),
             100 * np.mean(c < 0.8), np.percentile(mf, 10), np.percentile(mf, 25), np.median(mf), np.percentile(mf, 90),
             np.sum(~np.isfinite(m)), 100 * np.mean(m < 0.4)))
gate = (rows["slot_conf"] >= 0.8) & (rows["margin"] >= 0.4)
print("  rows passing slot conf >= 0.8 and margin >= 0.4 m: %d (%.1f%%)" % (gate.sum(), 100 * gate.mean()))

print("\npublic switches (engine label changes between H and A, 11185-18685):")
sw = [f for f in range(11186, 18686) if labels[f] in ("H", "A") and labels[f - 1] in ("H", "A") and labels[f] != labels[f - 1]]
sw += [f for f in range(11186, 18686) if labels[f] in ("H", "A") and labels[f - 1] == "D"]
sys.path.insert(0, os.path.join(ROOT, "tools"))
from gt import public_gt_codes  # noqa: E402
import possession_metrics as pm  # noqa: E402
gt = public_gt_codes()
code = np.array([{"H": 0, "A": 1, "D": 2}.get(str(x), 3) for x in labels], np.int8)
runs = pm.valid_runs(gt)
_, _, spurious = pm.match_transitions(pm.transitions(gt, runs), pm.transitions(code, runs), 25)
SPUR = set(p[0] for p in spurious)
for f in sorted(sw):
    t = str(labels[f])
    cand = ev[(ev["frame"] >= f - 50) & (ev["frame"] <= f) & (ev["team"] == t) & ev["kind"].isin(KINDS)]
    n2s = int(((ev["frame"] >= f - 50) & (ev["frame"] <= f) & (ev["team"] == t) & ev["kind"].isin(("touch", "release"))).sum())
    tag = "spurious" if f in SPUR else "matched"
    if len(cand):
        r = cand.iloc[-1]
        c, m, d = values(int(r["frame"]), t)
        print("  %6d %s->%s %-8s cause %6d %-8s conf %.2f margin %5.2f m d_own %.2f | new-team touch/release rows in 2 s: %d | passes gate: %s"
              % (f, str(labels[f - 1]), t, tag, int(r["frame"]), r["kind"], c, m, d, n2s, c >= 0.8 and m >= 0.4))
    else:
        print("  %6d %s->%s %-8s no team row of %s in [f-50, f] (restart / decoder)" % (f, str(labels[f - 1]), t, tag, t))
