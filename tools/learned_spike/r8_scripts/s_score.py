"""Round 8 arm S: score the set network against E2 on left-out Metrica and left-out DFL. Not part of the submission.

Runs on the round 6 per-frame checkout (origin/learned-r6-pf, commit 6a0af10: pf_eval,
pf_lopo5, the decoder of E2) with the per-frame venv; this file is only a driver:

    SP=<S971> CA=<round 2 cache> PYTHONDONTWRITEBYTECODE=1 ~/.venvs/barca-pf/bin/python \
        tools/learned_spike/r8_scripts/s_score.py --r6pf $SP/r6pf --set-p5 P5DIR --out OUTDIR

Systems at (12, 12), prior 0, kickoff rule (pf_lopo5.score_set, as pf_e2.py):
  E2 (stored pair)        0.75 x net2 (AQCTJ left-out, seeds 0-3) + 0.25 x left-out F5T
  E2 seed 0 pair          0.75 x net2 seed 0 + 0.25 x F5T
  S + member              0.75 x set network + 0.25 x F5T
  avg(S, net2 s0) pair    0.75 x mean(set, net2 seed 0) + 0.25 x F5T
  avg(S, net2 4s) pair    0.75 x mean(set, net2 four-seed mean) + 0.25 x F5T
Network means are taken at 5 fps (decoders.ensemble_p5, rounded once) and upsampled.
Also: where the set network and net2 (four seeds) disagree at 5 fps (argmax), by ball in view
(r5_measures.ball_visible at the bin centre) and near a reference change (within 5 bins).
"""
import argparse
import json
import os
import sys


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--r6pf", required=True)
    p.add_argument("--set-p5", required=True, help="folder of <game>.npy set-network p5")
    p.add_argument("--recipe", default=None)
    p.add_argument("--p5-cache", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--tests", nargs="+", default=["T-M", "T-D"], choices=["T-M", "T-D"])
    args = p.parse_args(argv)
    sys.path.insert(0, args.r6pf)
    import numpy as np
    from engine import data, learned
    from tools.learned_spike import decoders, folds
    from tools.learned_spike import pf_eval as pe
    from tools.learned_spike import pf_lopo5 as L
    sp = pe.SP
    recipe = args.recipe or sp + "/r7_work/screen/AQCTJ"
    cache = args.p5_cache or sp + "/r7_work/score_AQCTJ/p5_cache"
    md5 = {}

    def net_seed(game, s):
        path = os.path.join(recipe, folds.provider(game), "seed{}".format(s), "net.pt")
        if path not in md5:
            md5[path] = pe.md5_12(path)
        return np.load(os.path.join(cache, "{}__{}.npy".format(game, md5[path])))

    def net4(g):
        return decoders.ensemble_p5([net_seed(g, s) for s in range(4)])

    def net0(g):
        return decoders.ensemble_p5([net_seed(g, 0)])

    def setp(g):
        return np.load(os.path.join(args.set_p5, g + ".npy"))

    m5t = {q: os.path.join(L.R6, "models_all", "F5T", q) for q in ("metrica", "dfl",
                                                                    "skillcorner")}
    L.FEATS = os.path.join(L.R6, "feats_F5T")
    f5cache = os.path.join(L.R6, "lopo_E2", "probs_F5T")

    def f5t(g):
        return L.f5_probs(m5t[folds.provider(g)], g, f5cache)

    def up(p5, g):
        return learned.upsample(p5, data.frame_count(g))

    def pair(p5fn):
        return lambda g: 0.25 * f5t(g) + 0.75 * up(p5fn(g), g)

    systems = {
        "E2 (stored pair, net2 4 seeds)": pair(net4),
        "E2 seed 0 pair": pair(net0),
        "net2 4 seeds alone": lambda g: up(net4(g), g),
        "S alone": lambda g: up(setp(g), g),
        "S + member": pair(setp),
        "avg(S, net2 s0) + member": pair(lambda g: decoders.ensemble_p5([setp(g),
                                                                        net_seed(g, 0)])),
        "avg(S, net2 4s) + member": pair(lambda g: decoders.ensemble_p5([setp(g), net4(g)])),
    }
    tests = tuple(args.tests)
    os.makedirs(args.out, exist_ok=True)
    res, c23 = L.score_set(systems, tests, [(12.0, 12.0)], recall=False)
    S = {s: L.summary(res[s], c23, 0) for s in systems}
    per = {s: {t: {g: {k: res[s][t][g][0][k] for k in ("spellF1", "delta", "merge", "split")}
                   for g in res[s][t]} for t in tests} for s in systems}
    # disagreement at 5 fps
    dis = {}
    for t in tests:
        for g in pe.games_of(t):
            ref, _ = pe.refs(t, g)
            n = len(ref)
            nb = -(-n // 5)
            centres = np.minimum(np.arange(nb) * 5 + 2, n - 1)
            rb = np.asarray(ref)[centres]
            a, b = setp(g).argmax(1), net4(g).argmax(1)
            vis = np.asarray(pe.ball_visible(g))[centres].astype(bool)
            ch = np.flatnonzero(rb[1:] != rb[:-1])
            near = np.zeros(nb, bool)
            for c in ch:
                near[max(0, c - 4):c + 6] = True
            graded = rb != 3
            d = (a != b) & graded
            row = {"bins": int(graded.sum()), "disagree": int(d.sum())}
            for name, m in (("in_view", vis), ("hidden", ~vis), ("near_change", near),
                            ("far_from_change", ~near)):
                mm = m & graded
                row[name] = {"bins": int(mm.sum()), "disagree": int((d & mm).sum()),
                             "set_right": int((d & mm & (a == rb)).sum()),
                             "net_right": int((d & mm & (b == rb)).sum())}
            row["set_acc"] = float((a == rb)[graded].mean())
            row["net_acc"] = float((b == rb)[graded].mean())
            dis[g] = row
    json.dump({"summary": S, "per_match": per, "disagreement": dis},
              open(os.path.join(args.out, "s_score.json"), "w"), indent=1)
    L_ = ["## Left-out providers at (12,12), prior 0 (gain over c23; matches up; worst match)",
          "", "| system | T-M | T-D |", "|---|---|---|"]
    for s in systems:
        x = S[s]
        L_.append("| {} | {} |".format(s, " | ".join(
            "{:+.4f} ({}/{}, {:+.4f})".format(x[t]["gain"], x[t]["up"], x[t]["n"], x[t]["worst"])
            for t in tests)))
    L_ += ["", "| system | test | merge (c23) | split (c23) |", "|---|---|---|---|"]
    for s in systems:
        for t in tests:
            x = S[s][t]
            L_.append("| {} | {} | {:.3f} ({:.3f}) | {:.3f} ({:.3f}) |".format(
                s, t, x["merge"], x["merge_c23"], x["split"], x["split_c23"]))
    L_ += ["", "Per match gain over c23:", "", "| match | " + " | ".join(systems) + " |",
           "|---|" + "---|" * len(systems)]
    for t in tests:
        for g in sorted(per[next(iter(systems))][t]):
            L_.append("| {} | {} |".format(g, " | ".join("{:+.4f}".format(per[s][t][g]["delta"])
                                                         for s in systems)))
    L_ += ["", "Set network vs net2 (4 seeds), argmax at 5 fps on graded bins: share of bins "
           "where they disagree, and of those, how many the set / net2 had right:", "",
           "| match | disagree | in view | hidden | near change | far | acc set / net2 |",
           "|---|---|---|---|---|---|---|"]
    for g, r in dis.items():
        def f(k):
            x = r[k]
            return "{:.3f} ({}/{})".format(x["disagree"] / max(x["bins"], 1), x["set_right"],
                                           x["net_right"])
        L_.append("| {} | {:.3f} | {} | {} | {} | {} | {:.3f} / {:.3f} |".format(
            g, r["disagree"] / r["bins"], f("in_view"), f("hidden"), f("near_change"),
            f("far_from_change"), r["set_acc"], r["net_acc"]))
    out = "\n".join(L_) + "\n"
    open(os.path.join(args.out, "s_score.md"), "w").write(out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
