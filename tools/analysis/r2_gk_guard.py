"""GK-swap follow-up, part 2: prototype guards G1 / G2 by patching the tactical data the engine loads.

The guard moves the live GK-like slot's position into the frozen other-team GK slot on the frames it
fires (a per-frame team override that keeps the team track counts), then runs the unchanged engine
(possession.prepare + run, current config). Writes evaluation/results/rx_gk_<name>.csv.
Usage: python tools/analysis/r2_gk_guard.py <scratch_dir> [R]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
# R2_ENGINE_ROOT: a directory holding another engine/ (a read-only copy of an earlier commit).
# R2_TAG: prefix of the output names, R2_REF: reference CSV the unpatched run must equal.
ENGINE_ROOT = os.environ.get("R2_ENGINE_ROOT", ROOT)
TAG = os.environ.get("R2_TAG", "")
REF = os.environ.get("R2_REF", "c6_kickoff.csv")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)
sys.path.insert(0, ENGINE_ROOT)
import r2_gk_common as G  # noqa: E402
from engine import data, possession, submission  # noqa: E402
print("engine from", os.path.dirname(possession.__file__))
from engine.config import Config  # noqa: E402
import _common as cm  # noqa: E402
from _common import pm  # noqa: E402
from gt import public_gt_codes  # noqa: E402
import sanity  # noqa: E402

OUT = sys.argv[1]
R = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
L, W = data.pitch_m()
T0 = data.load_tactical()
blocks = data.graded_blocks()
gks = G.gk_slots(T0, blocks, L, W)
frozen = G.frozen_mask(T0, L, W)
cfg = Config.default()
gt = public_gt_codes()
runs_pub = pm.valid_runs(gt)
tags = data.load_tagging()
N = T0["n_frames"]
paint = cm.tag_team_codes(tags, N, sanity.TAG_PHASE_CODES, cm.runs_from_blocks(blocks))
print("config hash", cfg.config_hash(), "| R = %.1f m" % R)

print("\n=== frames where the guard fires, by R ===")
for need in (True, False):
    line = []
    for r in (3, 5, 8, 10, 15, 25, 1e9):
        sw = G.guard_frames(T0, blocks, L, W, r, need, gks, frozen)
        line.append("R=%s: %d" % ("inf" if r > 1e8 else "%g" % r, len(sw)))
    print("%s  %s" % ("G1 (beyond half-way)" if need else "G2 (no half-way)    ", " | ".join(line)))
g1 = set(f for f, _, _ in G.guard_frames(T0, blocks, L, W, R, True, gks, frozen))
g2 = set(f for f, _, _ in G.guard_frames(T0, blocks, L, W, R, False, gks, frozen))
print("at R=%.0f: G1 %d frames, G2 %d frames, G2 only %d frames" % (R, len(g1), len(g2), len(g2 - g1)))
px, py, _ = G.metres(T0, L, W)
only = sorted(g2 - g1)
if only:
    # where are the G2-only frames: which box?
    halves = np.array([1 if f < blocks[1][1] else 2 for f in only])
    print("  G2-only frames by half: H1 %d, H2 %d" % ((halves == 1).sum(), (halves == 2).sum()))


def run_variant(tactical):
    orig = data.load_tactical
    data.load_tactical = lambda *a, **k: tactical
    try:
        inputs = possession.prepare(cfg)
        labels, parts = possession.run(cfg, return_parts=True, inputs=inputs)
    finally:
        data.load_tactical = orig
    return np.asarray(labels), inputs["events"]


def public_report(codes):
    sp = pm.spell_scores(gt, codes, runs_pub)
    gtt, prt = pm.transitions(gt, runs_pub), pm.transitions(codes, runs_pub)
    pairs, missed, spurious = pm.match_transitions(gtt, prt, 25)
    sc = pm.transition_scores(gtt, prt, 25)
    lat = {g[0]: p[0] - g[0] for g, p in pairs}
    sp_frames = [p[0] for p in spurious]
    # IoU of the GT spell that starts at the GT A spell containing 15365..15766 (found from the GT)
    gts = pm.possession_spells(gt, runs_pub)
    prs = pm.possession_spells(codes, runs_pub)
    tgt = [s for s in gts if s[0] <= 15500 < s[1]][0]
    best = 0.0
    for a, b, k in prs:
        if k == tgt[2] and a < tgt[1] and tgt[0] < b:
            best = max(best, (min(b, tgt[1]) - max(a, tgt[0])) / float(max(b, tgt[1]) - min(a, tgt[0])))
    return sp, sc, lat, sp_frames, tgt, best


variants = {"base": T0}
variants["G1"] = G.patched_tactical(T0, G.guard_frames(T0, blocks, L, W, R, True, gks, frozen))
variants["G2"] = G.patched_tactical(T0, G.guard_frames(T0, blocks, L, W, R, False, gks, frozen))
res = {}
for name, tac in variants.items():
    labels, ev = run_variant(tac)
    path = os.path.join(ROOT, "evaluation", "results", "rx_gk_%s%s.csv" % (TAG, name)) if name != "base" else os.path.join(OUT, "rx_gk_%sbase.csv" % TAG)
    submission.write_submission(labels, path)
    print("\n##### %s -> %s | md5 %s | %s" % (name, path, submission.md5_of(path), submission.validate(path).strip()))
    codes = cm.read_results(path)
    res[name] = (codes, ev)
    sp, sc, lat, spf, tgt, iou = public_report(codes)
    print("public spellF1 %.4f | per IoU %s" % (sp["mean_f1"], " ".join("%.1f:%.3f" % (k, v["f1"]) for k, v in sorted(sp["per_iou"].items()))))
    print("transitions @25: F1 %.3f (matched %d, missed %d, spurious %d), mean |lat| %.1f"
          % (sc["f1"], sc["n_matched"], sc["n_missed"], sc["n_spurious"], sc["mean_abs_latency"] or np.nan))
    for gf in (11624, 15766):
        print("   GT %d: %s" % (gf, ("matched, latency %+d" % lat[gf]) if gf in lat else "missed"))
    for sf in (11708, 11960, 15654):
        print("   spurious %d: %s" % (sf, "present" if sf in spf else "absent"))
    print("   spurious list:", spf)
    print("   GT spell %d-%d (%s): best same-team IoU %.3f" % (tgt[0], tgt[1], "HADX"[tgt[2]], iou))
    st = sanity.sanity_stats(codes, tags=tags)
    print("full match: possessions %d (H1 %d, H2 %d) | D share H1 %.1f%% H2 %.1f%% total %.1f%% | D segments %d | tag agreement %.1f%% (H1 %.1f, H2 %.1f) | X graded %d | violations %s"
          % (st["total"]["possessions"], st["halves"][1]["possessions"], st["halves"][2]["possessions"],
             100 * st["halves"][1]["d_share"], 100 * st["halves"][2]["d_share"], 100 * st["total"]["d_share"],
             st["total"]["d_segments"], 100 * st["tags"]["total"]["agreement"], 100 * st["tags"]["halves"][1]["agreement"],
             100 * st["tags"]["halves"][2]["agreement"], st["total"]["x_graded"], st["violations"]))

base_codes, base_ev = res["base"]
ref = cm.read_results(os.path.join(ROOT, "evaluation", "results", REF))
print("\nbase == %s on every frame: %s" % (REF, bool(np.array_equal(base_codes, ref))))
key = lambda e: list(zip(e["frame"], e["kind"], e["team"].astype(str)))  # noqa: E731
for name in ("G1", "G2"):
    codes, ev = res[name]
    b = set(key(base_ev))
    v = set(key(ev))
    bf = {(f, k): t for f, k, t in b}
    vf = {(f, k): t for f, k, t in v}
    reassigned = sorted((f, k, bf[(f, k)], vf[(f, k)]) for (f, k) in bf if (f, k) in vf and bf[(f, k)] != vf[(f, k)])
    removed = sorted((f, k, t) for (f, k), t in bf.items() if (f, k) not in vf)
    added = sorted((f, k, t) for (f, k), t in vf.items() if (f, k) not in bf)
    print("\n=== %s: event changes vs base: reassigned %d, removed %d, added %d ===" % (name, len(reassigned), len(removed), len(added)))
    print("  reassigned:", reassigned)
    print("  removed:", removed)
    print("  added:", added)
    diff = codes != base_codes
    a, bnd = [], []
    p = np.concatenate(([0], diff.astype(int), [0]))
    e = np.flatnonzero(np.diff(p))
    print("  label changes: %d frames in %d stretches" % (diff.sum(), len(e) // 2))
    for s0, e0 in zip(e[::2], e[1::2]):
        w = slice(s0, e0)
        pt = paint[w]
        tagtxt = "H %d / A %d / both %d / none %d" % ((pt == 0).sum(), (pt == 1).sum(), (pt == 2).sum(), (pt == -1).sum())
        bl = "".join(sorted(set(cm.LETTER[c] for c in base_codes[w])))
        nl = "".join(sorted(set(cm.LETTER[c] for c in codes[w])))
        g = "".join(sorted(set(cm.LETTER[c] for c in gt[w])))
        print("   %6d-%6d (%4d fr, %5.1f s): %s -> %s | tag phase frames %s | public GT %s"
              % (s0, e0, e0 - s0, (e0 - s0) / 25.0, bl, nl, tagtxt, g))
np.savez_compressed(os.path.join(OUT, "gk_guard_labels_%s.npz" % (TAG or "c6")), base=base_codes, G1=res["G1"][0], G2=res["G2"][0])
