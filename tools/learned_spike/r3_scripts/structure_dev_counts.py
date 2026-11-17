"""What the match-structure step changes on the 8 development matches.

argv: REPO R3B_INTERIM_DIR OUT_JSON; env ST, CA. Development matches only (r3_scope.DEV); the
assert in r3_scope.load refuses BAR-ATH and the sealed matches. Python 3.9, numpy.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
from engine import data, learned  # noqa: E402
from tools.learned_spike import r3_scope as r  # noqa: E402

ST, CA = os.environ["ST"], os.environ["CA"]
check = json.load(open(os.path.join(sys.argv[2], "check.json")))
LAB = "HADX"
out = {}
for g in r.DEV:
    assert g not in r.SEALED
    d = r.load(g, ST, CA)              # round 2 OOF probabilities + references
    n, blocks = len(d["raw"]), d["blocks"]
    runs = [(a, b + 1) for _, a, b in blocks]
    kick = data.kickoff_frames(g)
    mask = learned.pre_kickoff_mask(n, blocks, kick)
    rec = {"kickoffs": kick, "blocks": blocks, "mask_frames": int(mask.sum()),
           "ref_raw_on_mask": {LAB[k]: int((d["raw"][mask] == k).sum()) for k in range(4)}}
    p5b = np.load(os.path.join(sys.argv[2], "p5", g + "_p5.npz"))["p5"]
    for name, probs, costs in (("r2_oof", d["probs"], d["cost"]),
                               ("r3b_seed0_oof", learned.upsample(p5b, n),
                                check["matches"][g]["costs"])):
        cha, cad = float(costs[0]), float(costs[1])
        base = learned.decode_model(probs, blocks, cha, cad)
        new = learned.decode_model(probs, blocks, cha, cad, forced_dead=mask)
        sb, sn = set(r.spells(base, runs)), set(r.spells(new, runs))
        rec[name] = {"costs": [cha, cad], "frames_changed": int((base != new).sum()),
                     "changed_outside_mask": int((base != new)[~mask].sum()),
                     "model_live_on_mask": int(np.isin(base[mask], (0, 1)).sum()),
                     "possessions_removed": len(sb - sn), "possessions_added": len(sn - sb),
                     "n_before": len(sb), "n_after": len(sn),
                     "spellF1_raw_before": round(r.spell_f1(d["raw"], base), 5),
                     "spellF1_raw_after": round(r.spell_f1(d["raw"], new), 5)}
    out[g] = rec
    print(g, json.dumps(rec))
json.dump(out, open(sys.argv[3], "w"), indent=1, default=int)
