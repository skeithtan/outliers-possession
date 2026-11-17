"""Plausibility of reference possessions from tracking: share of frames where the nearest player
within R_TOUCH (1.5 m) belongs to the possessing team, to the other team, or nobody is that near.

    python q_plaus.py   -> out/q_plaus.json
Possessions split into: recalled at IoU 0.5 / absorbed (start of a T_missed_absorbed episode) /
other unmatched. Duration bins <1, 1-2, 2-5, >=5 s.
"""
import json, os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
WORK = C.WORK

IH = list(C.IOUS).index(0.5)


def main():
    report = {}
    for view in ("matched", "raw"):
        eps = pd.read_csv(os.path.join(WORK, "out", "episodes_%s.csv" % view))
        rows = []
        for game in C.OPEN_GAMES:
            M = C.load(view, game)
            gt, ours, runs = M["gt"], M["ours"], M["runs"]
            gsp = C.spells(gt, runs)
            ev = C.spell_eval(gsp, ours, runs)
            gflag = ev["gflags"][IH]
            ab = set(int(s) for s in eps[(eps.game == game) & (eps.cls == "T_missed_absorbed")].start)
            nt = M["nearest_team"]
            dmin = M["dmin"]
            near = np.isfinite(dmin) & (dmin <= 1.5)
            for i, (s, e, t) in enumerate(zip(*gsp)):
                s, e, t = int(s), int(e), int(t)
                own = np.mean(near[s:e] & (nt[s:e] == t))
                oth = np.mean(near[s:e] & (nt[s:e] == 1 - t))
                cls = "recalled" if gflag[i] else ("absorbed" if s in ab else "other_unmatched")
                rows.append({"group": M["group"], "cls": cls, "dur": (e - s) / C.FPS,
                             "own": own, "oth": oth, "none": 1 - own - oth,
                             "own_any": bool(np.any(near[s:e] & (nt[s:e] == t)))})
        df = pd.DataFrame(rows)
        df["bin"] = pd.cut(df.dur, [0, 1, 2, 5, 1e9], right=False, labels=["<1s", "1-2s", "2-5s", ">=5s"])
        rep = {}
        for grp in ("Metrica", "DFL"):
            g = df[df.group == grp]
            t = g.groupby(["cls", "bin"], observed=True).agg(n=("own", "size"), own=("own", "mean"),
                                                             oth=("oth", "mean"), none=("none", "mean"),
                                                             own_any=("own_any", "mean"))
            rep[grp] = {"{}|{}".format(a, b): {k: float(v) for k, v in r.items()} for (a, b), r in t.iterrows()}
        report[view] = rep
    with open(os.path.join(WORK, "out", "q_plaus.json"), "w") as h:
        json.dump(report, h, indent=1, sort_keys=True)
    for view in report:
        for grp in report[view]:
            print("==", view, grp)
            for k, v in report[view][grp].items():
                print("  %-28s n%4d own %.2f oth %.2f none %.2f own_any %.2f" % (k, v["n"], v["own"], v["oth"], v["none"], v["own_any"]))


if __name__ == "__main__":
    main()
