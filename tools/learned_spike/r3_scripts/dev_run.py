"""Stage 1 decoder variants on the 8 development matches (out-of-fold probabilities).

argv: REPO OUT_JSON
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, sys.argv[1])
from tools.learned_spike import r3_scope as r  # noqa: E402

ST, CA = os.environ["ST"], os.environ["CA"]
HA = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
AD = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
GRID = [(a, b) for a in HA for b in AD]
LADDER = (0.5, 0.25, 0.125)
MBR_FIXED = [(1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0)]
S, STEP = 32, 5

t0 = time.time()
G = {g: r.load(g, ST, CA) for g in r.DEV}
out = {"games": {}, "grid": GRID}

# grid decodes (for the nested retune) and raw/ctl2 scores
grid_scores = {}
for g, d in G.items():
    paths = r.viterbi_many(d["probs"], d["blocks"], GRID)
    grid_scores[g] = [r.spell_f1(d["raw"], p) for p in paths]
    d["grid_paths"] = paths
print("grid", round(time.time() - t0), file=sys.stderr, flush=True)


def nested_pick(test):
    """Grid pair with the best mean of provider means on the other development matches."""
    others = [g for g in r.DEV if g != test]
    provs = sorted(set(r.provider(g) for g in others))
    best, best_v = None, -1
    for i, pair in enumerate(GRID):
        v = np.mean([np.mean([grid_scores[g][i] for g in others if r.provider(g) == p])
                     for p in provs])
        if v > best_v + 1e-12:
            best, best_v = i, v
    return best, best_v


for g, d in G.items():
    runs_raw = r.pm.valid_runs(d["raw"])
    blocks = d["blocks"]
    rec = {"fold": d["fold"], "fold_cost": d["cost"]}
    rec["ref_raw"] = r.stats(d["raw"], runs_raw)
    rec["ref_ctl2"] = r.stats(d["ctl2"], runs_raw)
    variants = {}
    variants["c22"] = d["c22"]
    variants["a_frozen"] = d["m_csv"]
    am = r.argmax_codes(d["probs"], blocks)
    variants["b0_argmax_raw"] = am
    variants["b_argmax_minfold"] = r.fold_min(am, d["probs"], blocks)
    cha, cad = d["cost"]
    for f in LADDER:
        variants["c_x%g" % f] = r.viterbi(d["probs"], blocks, cha * f, cad * f)
    i, v = nested_pick(g)
    rec["nested_pair"] = GRID[i]
    rec["nested_val"] = v
    variants["e_nested_raw"] = d["grid_paths"][i]
    variants["e_nested_raw_minfold"] = r.fold_min(d["grid_paths"][i], d["probs"], blocks,
                                                  cha=GRID[i][0], cad=GRID[i][1])
    # MBR: candidates and samples (transition matrix from the other dev references, raw)
    others = [G[o]["raw"] for o in r.DEV if o != g]
    trans = r.transition_matrix(others, step=STEP)
    cand_names = (["a_frozen", "b_argmax_minfold", "e_nested_raw"]
                  + ["c_x%g" % f for f in LADDER])
    cands = [variants[k] for k in cand_names]
    for pair in MBR_FIXED:
        cand_names.append("fix_%g_%g" % pair)
        cands.append(r.viterbi(d["probs"], blocks, *pair))
    samples = r.ffbs(d["probs"], blocks, trans, S, seed=0, step=STEP)
    k, sc = r.mbr_pick(cands, samples, blocks)
    variants["d_mbr"] = cands[k]
    rec["mbr_pick"] = cand_names[k]
    rec["mbr_expected"] = dict(zip(cand_names, [round(x, 4) for x in sc]))
    rec["mbr_true"] = {n: round(r.spell_f1(d["raw"], c), 4) for n, c in zip(cand_names, cands)}
    rec["sample_stats"] = r.stats(samples[0], [(a, b + 1) for _, a, b in blocks])
    rec["sample_n_mean"] = float(np.mean([len(r.spells(s, [(a, b + 1) for _, a, b in blocks]))
                                          for s in samples]))
    rec["variants"] = {}
    for name, codes in variants.items():
        st = r.stats(codes, runs_raw)
        rec["variants"][name] = {"spellF1_raw": round(r.spell_f1(d["raw"], codes), 4),
                                 "spellF1_ctl2": round(r.spell_f1(d["ctl2"], codes), 4),
                                 "count_ratio": round(st["n"] / rec["ref_raw"]["n"], 3),
                                 **{k2: round(v2, 3) for k2, v2 in st.items()}}
    out["games"][g] = rec
    print(g, rec["mbr_pick"], round(time.time() - t0), file=sys.stderr, flush=True)

with open(sys.argv[2], "w") as h:
    json.dump(out, h, indent=1, default=float)
