"""Q5 (possession durations, recall by duration at IoU 0.5) and Q6 (ball missing at hand-overs).

    python q56.py    -> out/q56.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
WORK = C.WORK

BINS = ((0, 1), (1, 2), (2, 5), (5, 10), (10, 1e9))
BIN_NAMES = ("<1s", "1-2s", "2-5s", "5-10s", ">=10s")
IH = list(C.IOUS).index(0.5)


def main():
    report = {}
    for view in ("matched", "raw"):
        acc = {}
        for game in C.OPEN_GAMES:
            M = C.load(view, game)
            gt, ours, runs = M["gt"], M["ours"], M["runs"]
            gsp = C.spells(gt, runs)
            ev = C.spell_eval(gsp, ours, runs)
            ps, pe, pl = ev["pred"]
            gd = (gsp[1] - gsp[0]) / C.FPS
            pd_ = (pe - ps) / C.FPS
            gflag = ev["gflags"][IH]
            pflag = ev["pflags"][IH]
            vis = M["visible"]
            changes = C.owner_changes(gt, runs)
            direct = [c["frame"] for c in changes if c["kind"] == "direct"]
            across = [c["frame"] for c in changes if c["kind"] == "across_D"]
            alive = np.zeros(len(gt), dtype=bool)
            for a, b in runs:
                alive[a:b] = np.isin(gt[a:b], (C.H, C.A))
            for g in (M["group"], "pooled"):
                d = acc.setdefault(g, {"ref_dur": [], "our_dur": [], "gflag": [], "pflag": [],
                                       "direct_missing": [], "direct_missing_pm12": [],
                                       "across_missing": [], "alive_missing": [0, 0]})
                d["ref_dur"] += gd.tolist()
                d["our_dur"] += pd_.tolist()
                d["gflag"] += gflag.tolist()
                d["pflag"] += pflag.tolist()
                d["direct_missing"] += [not vis[f] for f in direct]
                d["direct_missing_pm12"] += [not vis[max(0, f - 12):f + 13].all() for f in direct]
                d["across_missing"] += [not vis[f] for f in across]
                d["alive_missing"][0] += int(np.sum(alive & ~vis))
                d["alive_missing"][1] += int(np.sum(alive))
        out = {}
        for g, d in acc.items():
            rd, od = np.array(d["ref_dur"]), np.array(d["our_dur"])
            gf, pf = np.array(d["gflag"]), np.array(d["pflag"])
            rec = {}
            prec = {}
            for name, (lo, hi) in zip(BIN_NAMES, BINS):
                m = (rd >= lo) & (rd < hi)
                rec[name] = {"n": int(m.sum()), "recall": C.frac(gf[m].sum(), m.sum())}
                m2 = (od >= lo) & (od < hi)
                prec[name] = {"n": int(m2.sum()), "precision": C.frac(pf[m2].sum(), m2.sum())}
            out[g] = {
                "ref_n": int(len(rd)), "our_n": int(len(od)),
                "ref_q": C.quantiles(rd), "our_q": C.quantiles(od),
                "ref_share": {"<1s": float(np.mean(rd < 1)), "<2s": float(np.mean(rd < 2)),
                              "<5s": float(np.mean(rd < 5))},
                "our_share": {"<1s": float(np.mean(od < 1)), "<2s": float(np.mean(od < 2)),
                              "<5s": float(np.mean(od < 5))},
                "ref_mean_s": float(rd.mean()), "our_mean_s": float(od.mean()),
                "recall_by_ref_dur": rec, "precision_by_our_dur": prec,
                "q6_direct_ball_missing_at": float(np.mean(d["direct_missing"])),
                "q6_direct_ball_missing_any_pm12": float(np.mean(d["direct_missing_pm12"])),
                "q6_across_ball_missing_at_restart": float(np.mean(d["across_missing"])),
                "q6_alive_frames_ball_missing": d["alive_missing"][0] / float(d["alive_missing"][1]),
                "n_direct": len(d["direct_missing"]), "n_across": len(d["across_missing"]),
            }
        report[view] = out
    with open(os.path.join(WORK, "out", "q56.json"), "w") as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
    for view in report:
        for g in ("Metrica", "DFL", "pooled"):
            print(view, g, json.dumps(report[view][g])[:1800])


if __name__ == "__main__":
    main()
