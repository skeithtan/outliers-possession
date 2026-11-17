"""R2-f: structural facts of spellF1, checked exactly with evaluation/possession_metrics.py.

    python tools/analysis/r2_metric_struct.py [--out DIR]

(i)   A spurious switch of k frames inside a host spell of length L at relative position x:
      change in TP/FP per IoU threshold and in spellF1, on a synthetic 20-spell sequence that is
      otherwise perfect, and on the public window starting from c5_dribble.
(ii)  GT A D A sandwich: predict A D A with the D late at both ends versus A A A (merge);
      also A D H with the D missed.
(iii) Value of one more correct short spell versus one spurious one (F1 asymmetry), exact on the
      public window and analytic at full-match counts.
(iv)  Public bootstrap interval (bootstrap_n=1000) for c5_dribble and for variants that match one
      or two more spells.

Synthetic sequences only, plus the public GT via tools/gt.py. Never opens a full-match GT file.
"""
import argparse
import os
import tempfile
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from gt import public_gt_codes  # noqa: E402
import possession_metrics as pm  # noqa: E402

H, A, D, X = 0, 1, 2, 3
DEFAULT_OUT = os.path.join(tempfile.gettempdir(), "r2_research", "f_metric")


def build(segs, pad=10):
    """[(label, length)] -> codes with X padding either side (one valid run)."""
    out = [np.full(pad, X, np.int8)]
    for lab, n in segs:
        out.append(np.full(n, lab, np.int8))
    out.append(np.full(pad, X, np.int8))
    return np.concatenate(out)


def summary(gt, pred):
    r = pm.evaluate(gt, pred)
    m = [r["spell_f1"][t]["n_matched"] for t in pm.SPELL_IOUS]
    return r["spell_f1_mean"], m, r["n_pred_spells"]


def part_i(lines):
    lines.append("## (i) spurious switch of k frames inside a host spell\n")
    lines.append("Synthetic: 20 alternating H/A spells of 300 frames, spell 10 (H) has length L; the"
                 " prediction is perfect except an A run of k frames starting at x*L inside the host."
                 " Baseline spellF1 1.0 with 20 matched.\n")
    lines.append("| L | k | x | host best IoU | matched per IoU 0.3..0.7 | n_pred | spellF1 | ΔF1 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for L in (250, 500, 1000):
        for k in (10, 25, 50, 100):
            for x in (0.0, 0.05, 0.15, 0.25, 0.35, 0.5):
                segs = []
                for i in range(20):
                    segs.append((H if i % 2 == 0 else A, L if i == 10 else 300))
                gt = build(segs)
                pred = gt.copy()
                start = 10 + sum(n for _, n in segs[:10]) + int(round(x * L))
                pred[start:start + k] = A
                f1, m, npred = summary(gt, pred)
                left = int(round(x * L))
                best = max(left, L - left - k) / float(L)
                lines.append("| {} | {} | {} | {:.3f} | {} | {} | {:.4f} | {:+.4f} |".format(
                    L, k, x, best, m, npred, f1, f1 - 1.0))
    lines.append("")
    lines.append("x = 0 puts the run at the host start, where it merges with the preceding A spell"
                 " (a boundary error on two spells, no new spell).\n")
    # on the public window: insert a spurious A run in the middle of the matched 16400-17293 A? no:
    # needs the other team. Use H host 12989-13704 (matched, 715 f) and A host 16400-17293 (893 f).
    gt = public_gt_codes()
    runs = pm.valid_runs(gt)
    c5 = pm.fill_abstentions(pm.label_codes(pm.read_label_csv(
        os.path.join(ROOT, "evaluation", "results", "c5_dribble.csv"))), runs)
    base = summary(gt, c5)
    lines.append("Public window, c5_dribble base {:.4f} {} n_pred {}. Insert a spurious run of k"
                 " frames of the other team at relative position x of a matched host:\n".format(
                     base[0], base[1], base[2]))
    lines.append("| host | k | x | matched | n_pred | spellF1 | Δ |")
    lines.append("|---|---|---|---|---|---|---|")
    for (s, e, lab) in ((12989, 13691, A), (16393, 17304, H)):
        for k in (25, 50, 100):
            for x in (0.1, 0.25, 0.5):
                p = c5.copy()
                st = s + int(round(x * (e - s)))
                p[st:st + k] = lab
                f1, m, npred = summary(gt, p)
                lines.append("| {}-{} | {} | {} | {} | {} | {:.4f} | {:+.4f} |".format(
                    s, e, k, x, m, npred, f1, f1 - base[0]))
    lines.append("")


def part_ii(lines):
    lines.append("## (ii) GT A D A sandwich and A D H\n")
    lines.append("Context: 10 alternating 300-frame spells either side, all predicted perfectly;"
                 " ΔF1 is against a perfect prediction of the whole sequence (22 GT spells).\n")
    lines.append("| GT core | prediction | matched 0.3..0.7 | n_pred | spellF1 | ΔF1 vs perfect |")
    lines.append("|---|---|---|---|---|---|")
    for la, ld in ((300, 200), (100, 100), (300, 500), (150, 250)):
        for second in (A, H):
            ctx_l = [(H if i % 2 == 0 else A, 300) for i in range(10)]   # ends on A
            ctx_l = [(A if lab == H else H, n) for lab, n in ctx_l]       # ends on H
            ctx_r_first = H if second == A else A
            ctx_r = [((ctx_r_first if i % 2 == 0 else (A if ctx_r_first == H else H)), 300)
                     for i in range(10)]
            core = [(A, la), (D, ld), (second, la)]
            gt = build(ctx_l + core + ctx_r)
            off = 10 + 300 * 10
            name = "A{} D{} {}{}".format(la, ld, "HA"[second], la)
            variants = []
            for lag in (25, 50):
                p = gt.copy()
                p[off + la: off + la + lag] = A                    # onset late
                p[off + la + ld: off + la + ld + lag] = D          # restart late
                variants.append(("D late {} at both ends".format(lag), p))
                p = gt.copy()
                p[off + la - lag: off + la] = D                    # onset early
                p[off + la + ld: off + la + ld + lag] = D          # restart late
                variants.append(("D wider by {} each end".format(lag), p))
            p = gt.copy()
            p[off + la: off + la + ld] = A                          # D missed, held by A
            variants.append(("D missed: A A A merge" if second == A
                             else "D missed: A held to the restart", p))
            base = summary(gt, gt)
            for vname, p in variants:
                f1, m, npred = summary(gt, p)
                lines.append("| {} | {} | {} | {} | {:.4f} | {:+.4f} |".format(
                    name, vname, m, npred, f1, f1 - base[0]))
    lines.append("")


def f1(tp, ngt, npred):
    return 2.0 * tp / (ngt + npred) if tp else 0.0


def part_iii(lines):
    lines.append("## (iii) one more correct short spell versus one spurious spell\n")
    lines.append("F1 = 2TP/(n_gt + n_pred) at each threshold. Adding a correct spell: TP+1, n_pred+1;"
                 " adding a spurious one: n_pred+1. A switch pair inside a host (A run inside H)"
                 " adds 2 predicted spells (the run and the second host piece).\n")
    lines.append("| setting | n_gt | n_pred | TP | F1 | +1 correct spell | +1 spurious spell |"
                 " +correct pair (TP+2, P+2) | +spurious pair at edge (P+2) |"
                 " +spurious pair central (TP-1, P+2) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, g, p, tp in (("public c5 @0.5", 17, 17, 10), ("public c5 @0.7", 17, 17, 7),
                           ("full-match guess F1~0.45", 250, 241, 110),
                           ("full-match guess F1~0.60", 250, 241, 147)):
        b = f1(tp, g, p)
        lines.append("| {} | {} | {} | {} | {:.4f} | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} |"
                     " {:+.4f} |".format(name, g, p, tp, b, f1(tp + 1, g, p + 1) - b,
                                         f1(tp, g, p + 1) - b, f1(tp + 2, g, p + 2) - b,
                                         f1(tp, g, p + 2) - b, f1(tp - 1, g, p + 2) - b))
    lines.append("")
    # exact on public: insert the missed 15799-15889 A spell versus a 90-frame spurious H run in
    # the matched 16393-17304 A spell (centre and edge)
    gt = public_gt_codes()
    runs = pm.valid_runs(gt)
    c5 = pm.fill_abstentions(pm.label_codes(pm.read_label_csv(
        os.path.join(ROOT, "evaluation", "results", "c5_dribble.csv"))), runs)
    base = summary(gt, c5)
    p1 = c5.copy()
    p1[15799:15879] = A
    r1 = summary(gt, p1)
    p2 = c5.copy()
    p2[16800:16890] = H
    r2 = summary(gt, p2)
    p3 = c5.copy()
    p3[16450:16540] = H
    r3 = summary(gt, p3)
    lines.append("Exact, public window, from c5_dribble ({:.4f}):\n".format(base[0]))
    lines.append("| edit | matched | n_pred | spellF1 | Δ |\n|---|---|---|---|---|")
    for nm, r in (("insert missed A 15799-15879 (correct short spell)", r1),
                  ("spurious 90 f H run at 16800 (centre of 16393-17304 A)", r2),
                  ("spurious 90 f H run at 16450 (edge of 16393-17304 A)", r3)):
        lines.append("| {} | {} | {} | {:.4f} | {:+.4f} |".format(nm, r[1], r[2], r[0],
                                                                 r[0] - base[0]))
    lines.append("")


def part_iv(lines):
    lines.append("## (iv) public bootstrap resolution (bootstrap_n = 1000, seed 0)\n")
    gt = public_gt_codes()
    runs = pm.valid_runs(gt)
    c5 = pm.fill_abstentions(pm.label_codes(pm.read_label_csv(
        os.path.join(ROOT, "evaluation", "results", "c5_dribble.csv"))), runs)
    v1 = c5.copy()
    v1[15471 - 106:15471] = A          # 15365 boundary snapped (best single edit, +1 spell)
    v2 = v1.copy()
    v2[11351:11501] = H                # plus 11351 boundary snapped
    v3 = c5.copy()
    v3[15799:15879] = A                # insert the missed 15799 spell
    lines.append("| variant | spellF1 | 5% | 95% | width | matched |\n|---|---|---|---|---|---|")
    for nm, p in (("c5_dribble", c5), ("c5 + 15365 boundary snapped", v1),
                  ("c5 + 15365 + 11351 snapped", v2), ("c5 + missed 15799 A inserted", v3)):
        r = pm.evaluate(gt, p, bootstrap_n=1000)
        lo, hi = r["spell_bootstrap_low"], r["spell_bootstrap_high"]
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} |".format(
            nm, r["spell_f1_mean"], lo, hi, hi - lo,
            [r["spell_f1"][t]["n_matched"] for t in pm.SPELL_IOUS]))
    for name in ("baseline_smooth50", "baseline_smooth125", "m2_events"):
        p = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "evaluation", "results",
                                                          name + ".csv")))
        r = pm.evaluate(gt, p, bootstrap_n=1000)
        lo, hi = r["spell_bootstrap_low"], r["spell_bootstrap_high"]
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} |".format(
            name, r["spell_f1_mean"], lo, hi, hi - lo,
            [r["spell_f1"][t]["n_matched"] for t in pm.SPELL_IOUS]))
    lines.append("")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    lines = []
    part_i(lines)
    part_ii(lines)
    part_iii(lines)
    part_iv(lines)
    text = "\n".join(lines)
    with open(os.path.join(args.out, "struct.md"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
