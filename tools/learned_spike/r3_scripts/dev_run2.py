"""MBR with tempered emissions (alpha) + missed-possession anatomy on Metrica. argv: REPO OUT_JSON"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, sys.argv[1])
from tools.learned_spike import r3_scope as r
ST, CA = os.environ["ST"], os.environ["CA"]
G = {g: r.load(g, ST, CA) for g in r.DEV}
CANDS = [(16, 8), (8, 16), (4, 8), (2, 4), (1, 2), (1, 4), (0.5, 1), (0.25, 0.5)]
ALPHAS = (1.0, 0.3, 0.1)
S, STEP = 32, 5
out = {"mbr": {}, "missed": {}}
t0 = time.time()
true_tab, exp_tab = {}, {}
for g, d in G.items():
    blocks = d["blocks"]
    cands = [r.viterbi(d["probs"], blocks, *c) for c in CANDS]
    am = r.argmax_codes(d["probs"], blocks)
    cands.append(r.fold_min(am, d["probs"], blocks))
    names = ["v_%g_%g" % c for c in CANDS] + ["argmax_minfold"]
    others = [G[o]["raw"] for o in r.DEV if o != g]
    trans = r.transition_matrix(others, step=STEP)
    true = [r.spell_f1(d["raw"], c) for c in cands]
    rec = {"names": names, "true": true, "n": [len(r.spells(c, r.pm.valid_runs(d["raw"]))) for c in cands]}
    for al in ALPHAS:
        pr = np.clip(d["probs"], 1e-6, 1) ** al
        pr = pr / pr.sum(1, keepdims=True)
        samp = r.ffbs(pr, blocks, trans, S, seed=0, step=STEP)
        k, sc = r.mbr_pick(cands, samp, blocks)
        rec["a%g" % al] = {"pick": names[k], "exp": sc, "true_pick": true[k],
                           "sample_n": float(np.mean([len(r.spells(s, [(a, b + 1) for _, a, b in blocks])) for s in samp]))}
    out["mbr"][g] = rec
    print(g, round(time.time() - t0), file=sys.stderr, flush=True)

# missed possessions on Metrica: frozen vs c22, IoU 0.5 matching
BINS = [(0, 1), (1, 2), (2, 5), (5, 1e9)]
for g in [x for x in r.DEV if x.startswith("METRICA")] + [x for x in r.DEV if x.startswith("DFL")]:
    d = G[g]
    runs = r.pm.valid_runs(d["raw"])
    gs = r.spells(d["raw"], runs)
    def matched(pred, thr=0.5):
        ps = r.spells(r.pm.fill_abstentions(pred, runs), runs)
        ov = r.overlaps(gs, ps)
        ug, up = set(), set()
        for iou, i, j in ov:
            if iou < thr: break
            if i in ug or j in up: continue
            ug.add(i); up.add(j)
        return ug
    mf, mc = matched(d["m_csv"]), matched(d["c22"])
    rows = []
    for i, (a, b, k) in enumerate(gs):
        if i in mc and i not in mf:
            L = (b - a) / 25.0
            p = d["probs"][a:b]
            rows.append({"len": L, "p_team_mean": float(p[:, k].mean()),
                         "p_team_max": float(p[:, k].max()),
                         "argmax_share": float((p.argmax(1) == k).mean()),
                         "frozen_label_share_team": float((d["m_csv"][a:b] == k).mean())})
    tab = {}
    for lo, hi in BINS:
        sel = [x for x in rows if lo <= x["len"] < hi]
        allref = [1 for (a, b, k) in gs if lo <= (b - a) / 25 < hi]
        tab["%g-%g" % (lo, hi if hi < 1e9 else 99)] = {
            "n_ref": len(allref), "missed_by_frozen_found_by_c22": len(sel),
            "seen_argmax_ge50": sum(x["argmax_share"] >= 0.5 for x in sel),
            "seen_pmax_ge50": sum(x["p_team_max"] >= 0.5 for x in sel),
            "median_p_team_mean": float(np.median([x["p_team_mean"] for x in sel])) if sel else None}
    out["missed"][g] = {"n_ref": len(gs), "frozen_matched": len(mf), "c22_matched": len(mc),
                        "both": len(mf & mc), "table": tab}
json.dump(out, open(sys.argv[2], "w"), indent=1, default=float)
