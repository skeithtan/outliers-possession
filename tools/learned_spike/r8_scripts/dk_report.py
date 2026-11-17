"""Round 8 arm DK: the four-test table, guards and verdicts from dk_lopo.py's records. Not part of the submission.

    python -m tools.learned_spike.r8_scripts.dk_report --lopo ~/mtah2027/r8/dk/lopo \
        --pff ~/mtah2027/r8/dk/pff --out ~/mtah2027/r8/dk/report.md

Rule (protocol R8 Section 4, arm without training): four-test mean above E2's by >= 0.005, no
test below E2's by more than 0.003, the share of D inside stoppages with the ball in view on
unseen SkillCorner not below E2's by more than 0.01. Arm K: labels change only inside the
kickoff windows, four-test mean not lower than E2's by more than 0.001.
Python 3.9.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GAIN_BAR = 0.005
TEST_TOL = 0.003
RECALL_TOL = 0.01
K_TOL = 0.001
TESTS = ("T-M", "T-D", "T-S", "T-P")


def _pool_recall(rows, view):
    vals = [s for r in rows for s, v in r["recall"] if v == view]
    return float(np.mean(vals)) if vals else None


def _pool_hidden(rows):
    fr = sum(r["live_hidden_frames"] for r in rows)
    d = sum((r["live_hidden_D"] or 0.0) * r["live_hidden_frames"] for r in rows)
    return d / fr if fr else None


def per_test(lopo, pff):
    """{test: {"games": {game: row}}} with T-P from the 2 s view."""
    out = {t: lopo["tests"][t] for t in ("T-M", "T-D", "T-S")}
    out["T-P"] = {g: r["ctl2"] for g, r in pff["games"].items()} if pff else {}
    return out


def summarise(tests, systems):
    S = {}
    for s in systems + ["c23"]:
        S[s] = {}
        for t, games in tests.items():
            if not games or s not in next(iter(games.values())):
                continue
            rows = [games[g][s] for g in games]
            e2 = [games[g]["E2"] for g in games]
            x = {"n": len(rows),
                 "merge": float(np.mean([r["merge"] for r in rows])),
                 "split": float(np.mean([r["split"] for r in rows])),
                 "poss": float(np.mean([r["n"] for r in rows])),
                 "poss_ref": float(np.mean([games[g]["n_ref"] for g in games])),
                 "live_hidden_D": _pool_hidden(rows)}
            if s != "c23":
                x["gain"] = float(np.mean([r["delta"] for r in rows]))
                d = [r["delta"] - q["delta"] for r, q in zip(rows, e2)]
                x["vs_E2"] = float(np.mean(d))
                x["better"] = int(sum(v > 1e-12 for v in d))
                x["worse"] = int(sum(v < -1e-12 for v in d))
                x["changed"] = int(sum(r["changed_frames_vs_E2"] for r in rows))
            if "recall" in rows[0]:
                x["recall_view"] = _pool_recall(rows, True)
                x["recall_hidden"] = _pool_recall(rows, False)
            S[s][t] = x
        if s != "c23" and all(t in S[s] for t in TESTS):
            S[s]["mean4"] = float(np.mean([S[s][t]["gain"] for t in TESTS]))
    return S


def verdict(S, s):
    if "mean4" not in S[s]:
        return "incomplete", {}
    e = S["E2"]
    up = S[s]["mean4"] - e["mean4"]
    worst = min(S[s][t]["gain"] - e[t]["gain"] for t in TESTS)
    rv = S[s]["T-S"]["recall_view"] - e["T-S"]["recall_view"]
    if s == "K":
        ok = up >= -K_TOL
        return ("mean ok" if ok else "FAIL"), {"up": up, "worst": worst, "recall": rv}
    ok = up >= GAIN_BAR - 1e-12 and worst >= -TEST_TOL - 1e-12 and rv >= -RECALL_TOL - 1e-12
    return ("PASS" if ok else "fail"), {"up": up, "worst": worst, "recall": rv}


def report(S, systems):
    L = ["## Four tests: gain over c23 (difference to E2; matches better / worse than E2)", "",
         "| system | T-M | T-D | T-S | T-P (2 s) | mean | mean - E2 | verdict |",
         "|---|---|---|---|---|---|---|---|"]
    for s in systems:
        cells = []
        for t in TESTS:
            x = S[s].get(t)
            if x is None:
                cells.append("-")
            elif s == "E2":
                cells.append("{:+.4f}".format(x["gain"]))
            else:
                cells.append("{:+.4f} ({:+.4f}; {}/{})".format(x["gain"], x["vs_E2"],
                                                              x["better"], x["worse"]))
        v, info = verdict(S, s) if s != "E2" else ("", {})
        m = S[s].get("mean4")
        L.append("| {} | {} | {} | {} | {} |".format(
            s, " | ".join(cells), "{:+.4f}".format(m) if m is not None else "-",
            "{:+.4f}".format(info["up"]) if info else "-", v if s != "E2" else "reference"))
    L += ["", "## Guards", "",
          "| system | T-S D in stoppages, ball in view / hidden | live frames ball hidden: D share "
          "(T-M / T-D / T-S / T-P) | merge T-M / T-D / T-S / T-P | split T-M / T-D / T-S / T-P | "
          "possessions / reference (T-M / T-D / T-S / T-P) |",
          "|---|---|---|---|---|---|"]
    for s in ["c23"] + systems:
        x = S[s]
        ts = [t for t in TESTS if t in x]
        if "T-S" not in x:
            continue
        L.append("| {} | {:.3f} / {:.3f} | {} | {} | {} | {} |".format(
            s, x["T-S"]["recall_view"], x["T-S"]["recall_hidden"],
            " / ".join("{:.3f}".format(x[t]["live_hidden_D"]) for t in ts),
            " / ".join("{:.3f}".format(x[t]["merge"]) for t in ts),
            " / ".join("{:.3f}".format(x[t]["split"]) for t in ts),
            " / ".join("{:.0f}/{:.0f}".format(x[t]["poss"], x[t]["poss_ref"]) for t in ts)))
    L += ["", "## Frames changed against E2 (sum over the test's matches)", "",
          "| system | " + " | ".join(TESTS) + " |", "|---|" + "---|" * len(TESTS)]
    for s in systems[1:]:
        L.append("| {} | {} |".format(s, " | ".join(str(S[s][t]["changed"]) if t in S[s]
                                                     else "-" for t in TESTS)))
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lopo", required=True)
    p.add_argument("--pff", nargs="*", default=None, help="one or more dk_lopo pff --out folders")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    lopo = json.load(open(os.path.join(args.lopo, "lopo.json")))
    pff = None
    for folder in args.pff or ():
        part = json.load(open(os.path.join(folder, "pff.json")))
        if pff is None:
            pff = part
        else:
            pff["games"].update(part["games"])
    tests = per_test(lopo, pff)
    g0 = next(iter(tests["T-M"].values()))
    systems = [s for s in g0 if s not in ("n_ref", "c23")]
    S = summarise(tests, systems)
    out = report(S, systems)
    with open(args.out, "w") as h:
        h.write(out)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as h:
        json.dump(S, h, indent=1)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
