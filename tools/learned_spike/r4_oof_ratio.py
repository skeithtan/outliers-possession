"""Round 4, G7's count-ratio bound: possessions written / c23's on the 28 external matches, out of fold. Not part of the submission.

    (torch venv) python -m tools.learned_spike.r4_oof_ratio --folds RUN/folds --seeds 0 1 2 3 \
        --cache CACHE --dfl-cache CACHE_C --c23-dev DIR --c23-dfl DIR --c23-sc DIR \
        --pair 16 8 --prior-bonus 1 --out W/oof_ratio.json

LEARNED_STAGE_PROTOCOL_R4.md Section 5, G7: "possessions written / c23's possessions inside the
range of that ratio over the 28 external out-of-fold outputs at the same pair and bonus". This
tool writes that range; `r3_eval r4bounds --oof-ratio` applies it to the label-free step.

Each of the 28 external matches is scored by the fold models that did not train on it (the fold
that tests it, folds.fold(k, pff=False)["test"]: OPEN[k] and the SkillCorner matches SC[j] with
j mod 8 = k), the mean of the given seeds as the engine averages (r4_screen.ensemble), own
tracking (raw; DFL with the still ball removed: --dfl-cache, the dfl_c input), decoded with
engine.learned.decode_model at the pair, the kickoff rule (default on) and the prior bonus on
c23's labels of the same input (--c23-dev for Metrica, --c23-dfl for DFL, --c23-sc for
SkillCorner); with --pf-weights the candidate network + per-frame member as engine.learned
decodes it (pf_gate.py). Possessions are counted as the label-free step counts them
(bar_report.system_stats). No reference is read. Python 3.9.
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
from tools.learned_spike import folds  # noqa: E402


def fold_of(game):
    """The fold whose test set holds `game` (no PFF in round 4)."""
    for k in range(folds.N_FOLDS):
        if game in folds.fold(k, pff=False)["test"]:
            return k
    raise SystemExit("{}: in no fold's test set".format(game))


def inputs_of(game, args):
    """(features folder, c23 folder) of a match on the round 4 input."""
    p = folds.provider(folds.check(game))
    if p == "metrica":
        return args.cache, args.c23_dev
    if p == "dfl":
        return args.dfl_cache, args.c23_dfl
    return args.cache, args.c23_sc


def possessions(codes, game):
    from tools.learned_spike import bar_report as br
    return int(br.system_stats(np.asarray(codes, np.int8), br.graded_runs(game), None)
               ["possessions"])


def ratio_range(rows):
    """{"ratio_min", "ratio_max", "ratio_median"} over the per-match rows."""
    r = [row["ratio"] for row in rows.values()]
    return {"ratio_min": float(min(r)), "ratio_max": float(max(r)),
            "ratio_median": float(np.median(r)), "n_matches": len(r)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--folds", required=True, help="kfold.py output (fold<k>/net_s<s>.pt)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--cache", required=True)
    p.add_argument("--dfl-cache", required=True, help="dfl_c folder (still ball removed)")
    p.add_argument("--c23-dev", required=True)
    p.add_argument("--c23-dfl", required=True)
    p.add_argument("--c23-sc", required=True)
    p.add_argument("--pair", type=float, nargs=2, required=True)
    p.add_argument("--prior-bonus", type=float, required=True)
    p.add_argument("--structure", choices=("on", "off"), default="on")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", required=True)
    p.add_argument("--engine-cache", default=None,
                   help="engine.data cache of the raw tracking (needed with --pf-weights)")
    from tools.learned_spike import pf_gate
    pf_gate.add_pf_args(p)
    a = p.parse_args(argv)
    if a.pf_weights and not a.engine_cache:
        p.error("--pf-weights needs --engine-cache")
    from tools.learned_spike import r4_screen as rs
    from tools.learned_spike import train
    train.setup(a.threads)
    rows = {}
    for game in folds.ALL:
        k = fold_of(game)
        nets = [os.path.join(a.folds, "fold{}".format(k), "net_s{}.pt".format(s))
                for s in a.seeds]
        fdir, cdir = inputs_of(game, a)
        feats = rs.read_feats(fdir, game)
        from engine import data
        n = data.frame_count(game)      # no reference is read
        c23 = rs.read_c23(cdir, game, n)
        if a.pf_weights:
            # Round 6: the candidate network + per-frame member as engine.learned decodes it.
            p5 = pf_gate.blended_frame_probs(
                game, "c" if folds.provider(game) == "dfl" else "raw", nets,
                pf_gate.pf_path(a.pf_weights, fold=k),
                a.pf_weight, a.engine_cache,
                os.path.join(os.path.dirname(os.path.abspath(a.out)), "oof_pf_cache"))
        else:
            p5 = rs.ensemble(nets, feats, os.path.join(os.path.dirname(os.path.abspath(a.out)),
                                                       "oof_p5_cache"), game)
        codes = rs.decode_pairs(p5, game, [tuple(a.pair)], a.structure == "on", c23,
                                a.prior_bonus)[0]
        n_m, n_c = possessions(codes, game), possessions(c23, game)
        rows[game] = {"fold": k, "possessions": n_m, "possessions_c23": n_c,
                      "ratio": n_m / float(n_c)}
        print("{} fold {}: {} / {} = {:.3f}".format(game, k, n_m, n_c, rows[game]["ratio"]),
              file=sys.stderr, flush=True)
    rec = dict(ratio_range(rows), pair=list(a.pair), prior_bonus=a.prior_bonus,
               structure=a.structure == "on", seeds=a.seeds, folds=os.path.abspath(a.folds),
               matches=rows, **pf_gate.pf_record(a))
    with open(a.out, "w") as h:
        json.dump(rec, h, indent=1, sort_keys=True)
    print("ratio range [{:.3f}, {:.3f}] over {} matches".format(rec["ratio_min"], rec["ratio_max"],
                                                                 rec["n_matches"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
