"""Side checks on the missed-possession episodes (matched and raw views).

    python q_absorbed.py   -> out/q_absorbed.json
1. Strong bumps (|J| >= 9) inside each T_missed_absorbed episode [start-12, stop+12]: none / one /
   two or more; any attributable to the new team (uncontested, within 1.5 m); any contested.
2. Sign of the summed delayed nearest-player log-odds (engine near_lo) over each T_late,
   T_missed_absorbed and T_missed_to_dead episode: favours the new (reference) team or the old.
3. Ball gaps near reference direct hand-overs: share with a missing ball frame within +-24
   frames, and with a teleport-rejected frame, for missed and detected hand-overs; and for the
   ladder-J cases (no bump >= 4 m/s, new-team player within 1.5 m).
"""
import json, os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
WORK = C.WORK
from q3_detector import maxima_table  # noqa: E402


def main():
    report = {}
    for view in ("matched", "raw"):
        eps = pd.read_csv(os.path.join(WORK, "out", "episodes_%s.csv" % view))
        hd = pd.read_csv(os.path.join(WORK, "out", "handover_detector_%s.csv" % view))
        r1, r2, r3 = [], [], []
        for game in C.OPEN_GAMES:
            M = C.load(view, game)
            mx, _ = maxima_table(M)
            g = M["group"]
            for _, e in eps[(eps.game == game) & (eps.cls == "T_missed_absorbed")].iterrows():
                a, b = int(e.start), int(e.stop)
                w = mx[(mx.frame >= a - 12) & (mx.frame <= b + 12)]
                s = w[w.J >= 9]
                r1.append((g, (b - a) / C.FPS, len(s),
                           int(((s.team == e.ref) & ~s.contested & (s.d1 <= 1.5)).sum()),
                           int(s.contested.sum())))
            nl = M["near_lo"]
            for cls in ("T_late", "T_missed_absorbed", "T_missed_to_dead"):
                for _, e in eps[(eps.game == game) & (eps.cls == cls)].iterrows():
                    a, b = int(e.start), int(e.stop)
                    s = float(np.sum(nl[a:b])) * (1 if e.ref == "H" else -1)
                    r2.append((g, cls, s > 0, s < 0))
            vis, tel = M["visible"], M["teleport"]
            for _, h in hd[hd.game == game].iterrows():
                f = int(h.frame)
                r3.append((g, bool(h.det25), str(h.ladder)[0] == "J",
                           bool((~vis[max(0, f - 24):f + 25]).any()),
                           bool(tel[max(0, f - 24):f + 25].any())))
        d1 = pd.DataFrame(r1, columns=["group", "dur", "n_strong", "n_new", "n_cont"])
        d1["bin"] = pd.cut(d1.dur, [0, 1, 2, 5, 1e9], right=False, labels=["<1s", "1-2s", "2-5s", ">=5s"])
        d2 = pd.DataFrame(r2, columns=["group", "cls", "favors_new", "favors_old"])
        d3 = pd.DataFrame(r3, columns=["group", "det25", "ladder_J", "gap24", "tele24"])
        rep = {"absorbed_contacts": {}, "near_lo_sign": {}, "gaps": {}}
        for (grp, b), x in d1.groupby(["group", "bin"], observed=True):
            rep["absorbed_contacts"]["%s|%s" % (grp, b)] = {
                "n": int(len(x)), "no_strong": float((x.n_strong == 0).mean()),
                "one_strong": float((x.n_strong == 1).mean()), "two_plus": float((x.n_strong >= 2).mean()),
                "any_new_team_strong": float((x.n_new > 0).mean()), "any_contested": float((x.n_cont > 0).mean())}
        for (grp, cls), x in d2.groupby(["group", "cls"]):
            rep["near_lo_sign"]["%s|%s" % (grp, cls)] = {"n": int(len(x)), "favors_new": float(x.favors_new.mean()),
                                                         "favors_old": float(x.favors_old.mean())}
        for grp, x in d3.groupby("group"):
            rep["gaps"][grp] = {"missed_gap24": float(x[~x.det25].gap24.mean()),
                                "detected_gap24": float(x[x.det25].gap24.mean()),
                                "missed_teleport24": float(x[~x.det25].tele24.mean()),
                                "detected_teleport24": float(x[x.det25].tele24.mean()),
                                "ladderJ_gap24": float(x[x.ladder_J].gap24.mean()) if x.ladder_J.any() else None}
        report[view] = rep
    with open(os.path.join(WORK, "out", "q_absorbed.json"), "w") as h:
        json.dump(report, h, indent=1, sort_keys=True)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
