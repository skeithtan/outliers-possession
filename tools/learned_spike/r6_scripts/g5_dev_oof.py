"""Gate G5 (LEARNED_STAGE_PROTOCOL_R4.md Section 5): the eight development matches, out of fold,
own tracking (DFL with the still ball removed), labels as provided, at one pair and prior bonus.
Reuses r4_screen / r4_oof_ratio. Usage: g5_dev_oof.py FOLDS OUT.json HA AD BONUS"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
from tools.learned_spike import folds, r4_screen as rs, r4_oof_ratio as ro, train
fdir, out, ha, ad, bonus = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
CA, CVAL = os.environ["CA"], os.environ["C_VAL"]
C23 = {"metrica": os.environ["C23_DEV"], "dfl": os.environ["C23_DFL"]}
train.setup(2)
res = {"pair": [ha, ad], "prior_bonus": bonus, "matches": {},
       "pf": ({"weights": os.environ["PF_WEIGHTS"], "weight": float(os.environ.get("PF_WEIGHT", "0.5"))}
              if os.environ.get("PF_WEIGHTS") else None)}
for game in folds.OPEN:
    p = folds.provider(game); k = ro.fold_of(game)
    nets = [os.path.join(fdir, "fold{}".format(k), "net_s{}.pt".format(s)) for s in (0, 1, 2, 3)]
    ref = rs.read_labels(CA, game)
    feats = rs.read_feats(CVAL if p == "dfl" else CA, game)
    c23 = rs.read_c23(C23[p], game, len(ref))
    if os.environ.get("PF_WEIGHTS"):
        # Round 6: the candidate network + per-frame member as engine.learned decodes it
        # (pf_gate; PF_WEIGHTS, PF_WEIGHT (0.5) and ENGINE_CACHE from the environment).
        from tools.learned_spike import pf_gate
        p5 = pf_gate.blended_frame_probs(game, "c" if p == "dfl" else "raw", nets,
                                         pf_gate.pf_path(os.environ["PF_WEIGHTS"], fold=k),
                                         float(os.environ.get("PF_WEIGHT", "0.5")), os.environ["ENGINE_CACHE"],
                                         os.path.join(os.path.dirname(os.path.abspath(out)), "g5_pf_cache"))
    else:
        p5 = rs.ensemble(nets, feats, os.path.join(os.path.dirname(os.path.abspath(out)), "g5_p5_cache"), game)
    row = rs.match_rows(game, ref, p5, c23, [(ha, ad)], True, bonus)
    r = row["pairs"][0]
    res["matches"][game] = {"fold": k, "n_ref": row["n_ref"], "c23": row["c23"], "net": r["net"], "delta": r["delta"]}
    print("{} fold {}: net {:.4f} c23 {:.4f} delta {:+.4f}".format(game, k, r["net"]["spellF1"], row["c23"]["spellF1"], r["delta"]), flush=True)
d = [m["delta"] for m in res["matches"].values()]
res["summary"] = {"above_c23": int(sum(x > 0 for x in d)), "n": len(d), "mean": float(np.mean(d)), "min": float(min(d)), "pass": bool(all(x > 0 for x in d))}
json.dump(res, open(out, "w"), indent=1, sort_keys=True)
print("G5:", res["summary"])
