"""Out-of-fold scores of round 3 fold models on stressed views of the 8 development matches.

    # 1. features of the stressed views (corrupt.py, its own seeds: crc32 of the match name)
    python -m tools.learned_spike.r3aug_stress views --out DIR --engine-cache DIR
    # 2. out-of-fold 5 fps probabilities of one seed's fold models (torch venv)
    python -m tools.learned_spike.r3aug_stress predict --views DIR --folds DIR --seed 0 --out DIR
    # 3. spellF1 against the references as provided
    python -m tools.learned_spike.r3aug_stress score --pred DIR --cache DIR --check FILE --out FILE

Not part of the submission (docs/plan/calibration/R3_AUG.md Section 5). Views:

- stress        corrupt.corrupt() with its defaults: fragmentation, same- and cross-team identity
                swaps, matched ball gaps, no ghosts, no team flips. The view of criterion 4
                (evaluate.py VIEW_STRESS = `corrupt.py --ball matched`), same seeds.
- stress_noswap the same without the swap component (the defect left out of training).
- swap_only     identity swaps only (no fragmentation, ball untouched).

`score` decodes each match with two cost choices: the per-match nested costs of the raw-view
check (`r3_nested check` output, --check: costs chosen on clean input, never on the stressed
view), and costs chosen nested on the stressed view itself (r3_nested.nested_pick). It reads only
the 8 development matches and their external labels: never BAR-ATH, a sealed match or
evaluation/gt. Python 3.9.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from tools.learned_spike.r3_nested import DEV, FOLD_OF, GRID, nested_pick, provider  # noqa: E402

VIEWS = {"stress": {},
         "stress_noswap": {"swap": False},
         "swap_only": {"frag": False, "ball": "off"}}


def cmd_views(args):
    from tools.learned_spike import build, corrupt, features
    for game in DEV:
        tac = None
        for view, kw in VIEWS.items():
            path = os.path.join(args.out, view, game + ".npy")
            if os.path.isfile(path):
                continue
            if tac is None:
                tac = build.load_view(game, "raw", args.engine_cache)
            bad, rep = corrupt.corrupt(tac, game, **kw)
            f5 = features.downsample(features.frame_features(bad, game))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, f5)
            with open(os.path.join(args.out, view, game + ".json"), "w") as h:
                json.dump(rep, h, indent=1, sort_keys=True, default=float)
            print(game, view, rep["swaps_same"], rep["swaps_cross"],
                  rep["ball_hidden_graded_share"], file=sys.stderr, flush=True)
    return 0


def cmd_predict(args):
    from tools.learned_spike import decoders, train
    train.setup(args.threads)
    for game in DEV:
        net = train.load_net(os.path.join(args.folds, "fold%d" % FOLD_OF[game],
                                          "net_s%d.pt" % args.seed))
        for view in VIEWS:
            f5 = np.load(os.path.join(args.views, view, game + ".npy"))
            p5 = decoders.ensemble_p5([train.predict_p5(net, f5)])
            os.makedirs(os.path.join(args.out, view), exist_ok=True)
            np.savez_compressed(os.path.join(args.out, view, game + "_p5.npz"), p5=p5,
                                seed=args.seed, fold=FOLD_OF[game])
        print(game, file=sys.stderr, flush=True)
    return 0


def provider_means(per_game):
    return {p: float(np.mean([per_game[g] for g in DEV if provider(g) == p]))
            for p in ("METRICA", "DFL")}


def cmd_score(args):
    from engine import data
    from tools.learned_spike import decoders
    from tools.learned_spike import r3_scope as r
    with open(args.check) as h:
        check = json.load(h)
    out = {"check": os.path.abspath(args.check), "views": {},
           "raw": {"spellF1": {g: check["matches"][g]["spellF1"] for g in DEV}}}
    out["raw"]["provider_means"] = provider_means(out["raw"]["spellF1"])
    out["raw"]["balanced"] = float(np.mean(list(out["raw"]["provider_means"].values())))
    for view in VIEWS:
        scores, fixed = {}, {}
        for game in DEV:
            ref = np.load(os.path.join(args.cache, "labels", game + ".npz"))["raw"]
            p5 = np.load(os.path.join(args.pred, view, game + "_p5.npz"))["p5"]
            probs = decoders.upsample(p5, len(ref))
            paths = decoders.decode_model(probs, data.graded_blocks(game), GRID)
            scores[game] = [r.spell_f1(ref, p) for p in paths]
            fixed[game] = scores[game][GRID.index(tuple(check["matches"][game]["costs"]))]
        nested = {g: scores[g][nested_pick(scores, g)[0]] for g in DEV}
        rec = {"raw_costs": fixed, "nested_costs": nested}
        for key in ("raw_costs", "nested_costs"):
            pm = provider_means(rec[key])
            rec[key + "_provider_means"] = pm
            rec[key + "_balanced"] = float(np.mean(list(pm.values())))
        out["views"][view] = rec
    with open(args.out, "w") as h:
        json.dump(out, h, indent=1, sort_keys=True)
    summary = {v: round(out["views"][v]["raw_costs_balanced"], 4) for v in VIEWS}
    summary["raw"] = round(out["raw"]["balanced"], 4)
    print(json.dumps(summary))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("views")
    v.add_argument("--out", required=True)
    v.add_argument("--engine-cache", required=True)
    q = sub.add_parser("predict")
    q.add_argument("--views", required=True)
    q.add_argument("--folds", required=True)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--out", required=True)
    q.add_argument("--threads", type=int, default=2)
    s = sub.add_parser("score")
    s.add_argument("--pred", required=True)
    s.add_argument("--cache", required=True)
    s.add_argument("--check", required=True, help="r3_nested check JSON of the raw view")
    s.add_argument("--out", required=True)
    args = p.parse_args(argv)
    return {"views": cmd_views, "predict": cmd_predict, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
