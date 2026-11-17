"""Label-free BAR-ATH numbers per decoder variant. argv: REPO BAR_P5_NPZ C23_CSV M_CSV OUT_JSON"""
import json, os, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from tools.learned_spike import r3_scope as r
from tools.learned_spike import decoders
import score_game as sg
from engine import data
ST, CA = os.environ["ST"], os.environ["CA"]
z = np.load(sys.argv[2]); n = int(z["n_frames"]); p5 = z["p5"]
probs = decoders.upsample(p5, n)
blocks = data.graded_blocks("BAR-ATH")
runs = [(a, b + 1) for _, a, b in blocks]
c23 = sg.read_codes(sys.argv[3], n)
m_rec = sg.read_codes(sys.argv[4], n)
frozen = r.viterbi(probs, blocks, 16.0, 12.0)
print("frozen reproduces recorded M:", bool((frozen == m_rec).all()), "and engine run:", bool((frozen == z["m_codes"]).all()), file=sys.stderr)
# nested-raw pair from all 8 development matches (OOF), mean of provider means
HA = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0); AD = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
GRID = [(a, b) for a in HA for b in AD]
G = {g: r.load(g, ST, CA) for g in r.DEV}
sc = {g: [r.spell_f1(d["raw"], p) for p in r.viterbi_many(d["probs"], d["blocks"], GRID)] for g, d in G.items()}
def mpm(i):
    return np.mean([np.mean([sc[g][i] for g in r.DEV if r.provider(g) == p]) for p in ("METRICA", "DFL")])
best = max(range(len(GRID)), key=lambda i: (mpm(i), -i))
pair = GRID[best]
var = {"c23": c23, "a_frozen (16,12)": frozen}
var["b_argmax_minfold"] = r.fold_min(r.argmax_codes(probs, blocks), probs, blocks)
for f in (0.5, 0.25, 0.125):
    var["c_x%g (%g,%g)" % (f, 16 * f, 12 * f)] = r.viterbi(probs, blocks, 16 * f, 12 * f)
var["e_nested_raw (%g,%g)" % pair] = r.viterbi(probs, blocks, *pair)
# MBR, alpha given in argv[6] (nested choice from dev2.json), candidates as in dev_run2
alpha = float(sys.argv[6])
CANDS = [(16, 8), (8, 16), (4, 8), (2, 4), (1, 2), (1, 4), (0.5, 1), (0.25, 0.5)]
cands = [r.viterbi(probs, blocks, *c) for c in CANDS] + [var["b_argmax_minfold"]]
names = ["v_%g_%g" % c for c in CANDS] + ["argmax_minfold"]
trans = r.transition_matrix([G[g]["raw"] for g in r.DEV], step=5)
pr = np.clip(probs, 1e-6, 1) ** alpha; pr /= pr.sum(1, keepdims=True)
k, _ = r.mbr_pick(cands, r.ffbs(pr, blocks, trans, 32, seed=0, step=5), blocks)
var["d_mbr a=%g -> %s" % (alpha, names[k])] = cands[k]
ENV = {"poss_p50": (3.0, 11.2), "poss_p90": (12.08, 44.96), "d_p50": (12.32, 23.84), "d_p90": (40.8, 66.34)}
out = {"nested_pair": pair, "rows": {}}
graded = np.zeros(n, bool)
for a, b in runs: graded[a:b] = True
for name, c in var.items():
    st = r.stats(c, runs)
    dl = r.d_segment_lengths(c, runs)
    v = {"poss_p50": st["median_s"], "poss_p90": st["p90_s"], "d_p50": float(np.median(dl)), "d_p90": float(np.percentile(dl, 90))}
    ok = {k2: ENV[k2][0] <= v[k2] <= ENV[k2][1] for k2 in ENV}
    out["rows"][name] = {"possessions": st["n"], "median_s": round(st["median_s"], 2), "mean_s": round(st["mean_s"], 2),
                         "d_share": round(st["d_share"], 4), "d_segments": st["d_segments"],
                         "agree_c23": round(float((c[graded] == c23[graded]).mean()), 4),
                         "8e": {k2: [round(v[k2], 2), ok[k2]] for k2 in ENV}, "8e_pass": all(ok.values())}
    print(name, json.dumps(out["rows"][name]))
json.dump(out, open(sys.argv[5], "w"), indent=1, default=float)
