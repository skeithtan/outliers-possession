"""Arm M (round 8): candidate members with the left-out networks on tests (1) to (3). Not part of the submission.

    # pf venv
    python -m tools.learned_spike.r8_scripts.m_score --out ~/mtah2027/r8/m/score \
        --member E2=SP/r6_pf/models_all/F5T/{provider} \
        --member F6a=~/mtah2027/r8/m/models/F6a/{provider} [...]

The harness of LOPO_E2.md (tools/learned_spike/pf_e2.py), with the member as a parameter:
- network: the retrained left-out networks (AQCTJ, seeds 0-3), their per-network 5 fps
  probabilities from r4_screen's cache (r7_work/score_AQCTJ/p5_cache, key <game>__<md5[:12] of
  net.pt>; fallback r7_work/measures_AQCTJ_probs, the copy mean), averaged over the seeds and
  rounded once (decoders.ensemble_p5), engine.learned.upsample;
- member: a pf_train fit-all models.pkl of the match's left-out provider ("{provider}" in the
  path), predicted (pf_train.predict) on the evaluation view's features: feats_F5T/eval/<g>.npy
  (F5T's 184 columns), with feats_F6/eval/<g>.npy (the 36 group columns) appended for an F6
  member;
- blend: w x member + (1 - w) x network (w = 0.25), no rounding, as pf_e2.
Systems: each member blended and alone, and the network alone. Scoring: pf_lopo5.score_set /
summary at (12, 12), prior 0, kickoff rule, dead-ball recall inside SkillCorner stoppages
(pf_eval.extra). Python 3.9.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__)))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from engine import data, learned  # noqa: E402
from tools.learned_spike import decoders, folds  # noqa: E402
from tools.learned_spike import pf_eval as pe  # noqa: E402
from tools.learned_spike import pf_lopo5 as L  # noqa: E402
from tools.learned_spike import pf_train  # noqa: E402

SP = L.SP
RECIPE = SP + "/r7_work/screen/AQCTJ"
P5_CACHE = SP + "/r7_work/score_AQCTJ/p5_cache"
MEASURES = SP + "/r7_work/measures_AQCTJ_probs"
F5T_EVAL = SP + "/r6_pf/feats_F5T/eval"
PAIR = (12.0, 12.0)
_MD5 = {}


def net_p5(game):
    prov = folds.provider(game)
    ps = []
    for s in range(4):
        path = os.path.join(RECIPE, prov, "seed{}".format(s), "net.pt")
        if path not in _MD5:
            _MD5[path] = pe.md5_12(path)
        f = os.path.join(P5_CACHE, "{}__{}.npy".format(game, _MD5[path]))
        if os.path.isfile(f):
            ps.append(np.load(f))
        else:
            ps.append(np.load(os.path.join(MEASURES, "{}__raw__{}.npy".format(
                game, _MD5[path]))).mean(axis=0))
    return decoders.ensemble_p5(ps)


def net(game):
    return learned.upsample(net_p5(game), data.frame_count(game))


def member(template, game, cache, groups_eval):
    if "@" in template:                  # MODELS_TEMPLATE@GROUP_EVAL_DIR (F6c: robust columns)
        template, groups_eval = template.split("@", 1)
    mdir = template.replace("{provider}", folds.provider(game))
    tag = mdir.strip("/").replace("/", "_")[-60:]
    path = os.path.join(cache, tag, game + ".npy")
    if os.path.isfile(path):
        return np.load(path)
    with open(os.path.join(mdir, "models.pkl"), "rb") as h:
        models = pickle.load(h)
    F = np.load(os.path.join(F5T_EVAL, game + ".npy"))
    if models.get("fset") == "F6":
        F = np.concatenate([F, np.load(os.path.join(groups_eval, game + ".npy"))], axis=1)
    P = pf_train.predict(models, F, "full")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, P)
    return P


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--member", nargs="+", required=True, help="NAME=MODELS_DIR_TEMPLATE[@GROUP_EVAL_DIR]")
    p.add_argument("--groups-eval", default=os.path.expanduser("~/mtah2027/r8/m/feats_F6/eval"))
    p.add_argument("--weight", type=float, default=0.25)
    args = p.parse_args(argv)
    members = dict(m.split("=", 1) for m in args.member)
    cache = os.path.join(args.out, "probs")
    systems = {"network alone": net}
    for name, t in members.items():
        systems[name] = (lambda t: lambda g: args.weight * member(t, g, cache, args.groups_eval)
                         + (1.0 - args.weight) * net(g))(t)
        systems[name + " member alone"] = (lambda t: lambda g: member(
            t, g, cache, args.groups_eval))(t)
    res, c23 = L.score_set(systems, L.TESTS, [PAIR], recall=True)
    S = {s: L.summary(res[s], c23, 0) for s in systems}
    per_match = {s: {t: {g: res[s][t][g][0]["delta"] for g in res[s][t]} for t in L.TESTS}
                 for s in systems}
    os.makedirs(args.out, exist_ok=True)
    json.dump({"summary": S, "per_match_gain": per_match, "members": members,
               "weight": args.weight, "pair": PAIR}, open(os.path.join(args.out, "score.json"),
                                                          "w"), indent=1)
    lines = ["| system | T-M | T-D | T-S | T-Sp | D in view (SC) | D hidden (SC) |",
             "|---|---|---|---|---|---|---|"]
    for s in systems:
        x = S[s]
        lines.append("| {} | {} | {:.3f} | {:.3f} |".format(s, " | ".join(
            "{:+.4f} ({}/{}, {:+.4f})".format(x[t]["gain"], x[t]["up"], x[t]["n"], x[t]["worst"])
            for t in L.TESTS), x["T-S"]["recall_view_sys"], x["T-S"]["recall_hidden_sys"]))
    lines += ["", "| system | test | merge (c23) | split (c23) |", "|---|---|---|---|"]
    for s in systems:
        for t in L.TESTS:
            x = S[s][t]
            lines.append("| {} | {} | {:.3f} ({:.3f}) | {:.3f} ({:.3f}) |".format(
                s, t, x["merge"], x["merge_c23"], x["split"], x["split_c23"]))
    out = "\n".join(lines) + "\n"
    open(os.path.join(args.out, "SCORE.md"), "w").write(out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
