"""Label-free counts of the match-structure step on the stored round 2 M output of BAR-ATH.

argv: REPO NPZ OUT_JSON. Reads the stored 5 fps probabilities and codes, games.py and the
Smart Tagging file (input). Reads no label of any match. Python 3.9, numpy.
"""
import json
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
from engine import data, learned  # noqa: E402
from tools.learned_spike import r3_scope as r  # noqa: E402

G = "BAR-ATH"
z = np.load(sys.argv[2])
n = int(z["n_frames"])
probs = learned.upsample(z["p5"], n)
blocks = data.graded_blocks(G)
runs = [(a, b + 1) for _, a, b in blocks]
kick = data.kickoff_frames(G)
mask = learned.pre_kickoff_mask(n, blocks, kick)


def spellset(c):
    return set(r.spells(np.asarray(c, dtype=np.int8), runs))


def row(c):
    st = r.stats(c, runs)
    dl = r.d_segment_lengths(c, runs)
    return {"possessions": st["n"], "median_s": round(st["median_s"], 3),
            "p90_s": round(st["p90_s"], 3), "mean_s": round(st["mean_s"], 3),
            "d_share": round(st["d_share"], 5), "d_segments": st["d_segments"],
            "d_p50_s": round(float(np.median(dl)), 3), "d_p90_s": round(float(np.percentile(dl, 90)), 3)}


def diff(a, b):
    sa, sb = spellset(a), spellset(b)
    return {"frames_changed": int((a != b).sum()),
            "frames_changed_in_mask": int((a != b)[mask].sum()),
            "frames_changed_outside_mask": int((a != b)[~mask].sum()),
            "possessions_removed": len(sa - sb), "possessions_added": len(sb - sa),
            "removed": sorted([list(map(int, s)) for s in sa - sb]),
            "added": sorted([list(map(int, s)) for s in sb - sa])}


out = {"kickoffs": kick, "blocks": blocks, "mask_frames": int(mask.sum()),
       "mask_runs": [[int(a), int(b)] for a, b in learned.alive_runs(np.where(mask, 0, 2).astype(np.int8))]}
stored = z["m_codes"].astype(np.int8)
out["stored_reproduced_16_12"] = bool((learned.decode_model(probs, blocks, 16.0, 12.0) == stored).all())
lab = {0: "H", 1: "A", 2: "D", 3: "X"}
w = stored[mask]
br = np.flatnonzero(w[1:] != w[:-1]) + 1
st = np.concatenate(([0], br)); sp = np.concatenate((br, [len(w)]))
out["stored_in_mask_runs"] = [[lab[int(w[s])], int(e - s)] for s, e in zip(st, sp)]
out["stored_in_mask_counts"] = {lab[k]: int((w == k).sum()) for k in range(4)}

for cha, cad in ((16.0, 12.0), (2.0, 4.0)):
    key = "costs_%g_%g" % (cha, cad)
    base = learned.decode_model(probs, blocks, cha, cad)
    forced = learned.decode_model(probs, blocks, cha, cad, forced_dead=mask)
    over = base.copy(); over[mask] = learned.D
    out[key] = {"before": row(base), "after_forced": row(forced), "after_overwrite": row(over),
                "change_forced": diff(base, forced), "change_overwrite": diff(base, over),
                "forced_vs_overwrite_frames": int((forced != over).sum()),
                "graded_X_after": int(sum((forced[a:b] == learned.X).sum() for a, b in runs))}

# GOALS diagnostic (W3's label-free test): the label at each GOALS end and the D run around it.
tag = data.load_tagging(G)
codes = tag["code"].astype(str).str.upper()
goals = []
for _, t in tag[codes == "GOALS"].iterrows():
    e = int(t["end_frame"])
    rec = {"start": int(t["start_frame"]), "end": e}
    for key, c in (("16_12", stored), ("2_4", learned.decode_model(probs, blocks, 2.0, 4.0))):
        rec["label_at_end_" + key] = lab[int(c[e])]
        # nearest D frame at or after the GOALS end, and the D run that holds the end if any
        if c[e] == learned.D:
            a = e
            while a > 0 and c[a - 1] == learned.D: a -= 1
            b = e
            while b + 1 < n and c[b + 1] == learned.D: b += 1
            rec["d_run_" + key] = [a, b + 1]
        else:
            prev = np.flatnonzero(c[:e] == learned.D)
            rec["last_D_before_end_" + key] = int(prev[-1]) if prev.size else None
            nxt = np.flatnonzero(c[e:] == learned.D)
            rec["next_D_after_end_" + key] = int(e + nxt[0]) if nxt.size else None
    goals.append(rec)
out["goals"] = goals
json.dump(out, open(sys.argv[3], "w"), indent=1, default=int)
print(json.dumps({k: v for k, v in out.items() if k not in ("goals",)}, indent=1, default=int)[:6000])
print(json.dumps(goals, indent=1))
