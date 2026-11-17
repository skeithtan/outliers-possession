"""Signal check: team attribution of strong bump contacts (|J| >= DV_CONTACT) by several windows.

    python q_attr.py   -> out/q_attr.json

Bumps are rebuilt as in q3_detector.maxima_table. Two populations on reference-alive frames:
  handover bumps: a reference direct hand-over W -> R within +-12 frames of the bump; target R.
  inside bumps:   no reference change within +-25 frames; target = reference team at the bump.
Rules (each gives the nearer team and a contested flag with the engine's R_TOUCH / margin test):
  engine   min distance over [c-12, c+12] (engine/events._bump_contacts)
  at_c     distance at c
  post     min distance over [c, c+12]
  post6    min distance over [c+3, c+9]
  pre      min distance over [c-12, c]
Diagnostic only; nothing here is tuned or fed back into the engine.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
WORK = C.WORK
from q3_detector import maxima_table, THR, R_TOUCH, MARGIN  # noqa: E402

RULES = {"engine": (-12, 12), "at_c": (0, 0), "post": (0, 12), "post6": (3, 9), "pre": (-12, 0)}


def attribute(dH, dA, c, lo, hi):
    n = len(dH)
    a, b = max(0, c + lo), min(n, c + hi + 1)
    dh, da = np.min(dH[a:b]), np.min(dA[a:b])
    team = 0 if dh <= da else 1
    d1, d2 = min(dh, da), max(dh, da)
    contested = d2 <= R_TOUCH and (d2 - d1) < MARGIN
    near = d1 <= R_TOUCH
    return team, contested, near


def main():
    report = {}
    for view in ("matched", "raw"):
        acc = {}
        for game in C.OPEN_GAMES:
            M = C.load(view, game)
            grp = M["group"]
            mx, _ = maxima_table(M)
            mx = mx[mx.J >= THR]
            gt, runs = M["gt"], M["runs"]
            changes = C.owner_changes(gt, runs)
            ref_dir = np.array([c["frame"] for c in changes if c["kind"] == "direct"], dtype=np.int64)
            ref_dst = np.array([c["dst"] for c in changes if c["kind"] == "direct"], dtype=np.int64)
            all_ch = np.array([c["frame"] for c in changes if c["kind"] != "same_D"], dtype=np.int64)
            dH = np.where(np.isfinite(M["dH"]), M["dH"], np.inf).astype(np.float64)
            dA = np.where(np.isfinite(M["dA"]), M["dA"], np.inf).astype(np.float64)
            for c in mx.frame.to_numpy():
                if gt[c] not in (C.H, C.A):
                    continue
                k = np.flatnonzero(np.abs(ref_dir - c) <= 12)
                if len(k):
                    pop, target = "handover", int(ref_dst[k[np.argmin(np.abs(ref_dir[k] - c))]])
                elif len(all_ch) == 0 or np.min(np.abs(all_ch - c)) > 25:
                    pop, target = "inside", int(gt[c])
                else:
                    continue
                for rule, (lo, hi) in RULES.items():
                    team, contested, near = attribute(dH, dA, int(c), lo, hi)
                    for g in (grp, "pooled"):
                        key = (g, pop, rule)
                        d = acc.setdefault(key, {"n": 0, "correct": 0, "wrong": 0, "contested": 0,
                                                 "far": 0})
                        d["n"] += 1
                        if not near:
                            d["far"] += 1
                        elif contested:
                            d["contested"] += 1
                        elif team == target:
                            d["correct"] += 1
                        else:
                            d["wrong"] += 1
        out = {}
        for (g, pop, rule), d in sorted(acc.items()):
            n = float(d["n"])
            out.setdefault(g, {}).setdefault(pop, {})[rule] = {
                "n": d["n"], "correct": d["correct"] / n, "wrong": d["wrong"] / n,
                "contested": d["contested"] / n, "far": d["far"] / n}
        report[view] = out
    with open(os.path.join(WORK, "out", "q_attr.json"), "w") as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
    for view in report:
        for g in ("Metrica", "DFL"):
            for pop in ("handover", "inside"):
                print(view, g, pop, {r: ("n%d c%.2f w%.2f k%.2f f%.2f" % (v["n"], v["correct"], v["wrong"], v["contested"], v["far"]))
                                     for r, v in report[view][g][pop].items()})


if __name__ == "__main__":
    main()
