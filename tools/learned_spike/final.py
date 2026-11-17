"""The final ensemble of the learned stage: four models on the round 2 training set. Not part of the submission.

    python -m tools.learned_spike.final --cache DIR --ext DIR --out DIR --seeds 0 1 2 3
        (--steps N | --folds DIR) [--threads 2]
    python -m tools.learned_spike.final --costs --folds DIR --out DIR

This tool implements LEARNED_STAGE_PROTOCOL.md Section 6 with the advisor changes of
9 November 2026.

- Training: one model per seed on all 28 external matches and the 43 PFF training matches
  (folds.TRAIN_ALL, LEARNED_STAGE_PROTOCOL_R2.md Section 1, PFF raw view only), with the recipe
  of train.py. No validation set exists, so the tool keeps the weights of step S. It writes
  OUT/net_s{seed}.pt and OUT/final_s{seed}.json. It checks every PFF reference md5 against R2
  Section 8 first.
- Step count: with --folds, S is the median of the best steps of the fold models (32 values),
  rounded to the nearest multiple of 200 (a half goes up), at least 200. --steps N sets S by
  hand, for a smoke test.
- Costs (--costs): the median of the eight fold costs, for each candidate. The tool writes
  OUT/costs.json. The median of an even count is the mean of the two middle values, so it can
  fall between two grid points.

Python 3.9.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from tools.learned_spike import folds, train  # noqa: E402
from tools.learned_spike.kfold import log, write_json  # noqa: E402



def fold_records(folds_dir):
    """The fold.json records under folds_dir, in fold order. Raise SystemExit if one is missing."""
    records = []
    for k in range(folds.N_FOLDS):
        path = os.path.join(folds_dir, "fold{}".format(k), "fold.json")
        if not os.path.isfile(path):
            raise SystemExit("{} is missing: run every fold first".format(path))
        with open(path) as handle:
            records.append(json.load(handle))
    return records


def best_steps(folds_dir):
    """The best step of every fold model (model_s*.json of each fold)."""
    steps = []
    for k in range(folds.N_FOLDS):
        paths = sorted(glob.glob(os.path.join(folds_dir, "fold{}".format(k), "model_s*.json")))
        if not paths:
            raise SystemExit("fold {} has no model records in {}".format(k, folds_dir))
        for path in paths:
            with open(path) as handle:
                steps.append(int(json.load(handle)["best_step"]))
    return steps


def median_costs(folds_dir):
    """The record of costs.json: the median of the fold costs, per candidate."""
    records = fold_records(folds_dir)
    m = [r["costs"]["M"] for r in records]
    h = [r["costs"]["H"] for r in records]
    return {
        "M": {"cost_ha": float(np.median([c["cost_ha"] for c in m])),
              "cost_ad": float(np.median([c["cost_ad"] for c in m]))},
        "H": {"cost_ha": float(np.median([c["cost_ha"] for c in h]))},
        "fold_costs": {"M": [[c["cost_ha"], c["cost_ad"]] for c in m],
                       "H": [c["cost_ha"] for c in h]},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", help="build.py cache")
    parser.add_argument("--ext", help="c22.py cache (Metrica references), for the md5 record")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(train.FOLD_SEEDS))
    parser.add_argument("--steps", type=int, help="step count (default: from --folds)")
    parser.add_argument("--folds", help="kfold.py output folder")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--costs", action="store_true", help="write costs.json only")
    # Round 3 switches (LEARNED_STAGE_PROTOCOL_R3.md, DRAFT); the defaults are round 2.
    # Round 4 switches (LEARNED_STAGE_PROTOCOL_R4.md Section 1): train.add_switch_args().
    train.add_switch_args(parser)
    parser.add_argument("--no-pff", dest="pff", action="store_false",
                        help="R3b: no PFF match in training, no PFF file read")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    if args.costs:
        if not args.folds:
            raise SystemExit("--costs needs --folds")
        costs = median_costs(args.folds)
        write_json(os.path.join(args.out, "costs.json"), costs)
        print(json.dumps({k: costs[k] for k in ("M", "H")}))
        return 0
    if not args.cache or not args.ext:
        raise SystemExit("training needs --cache and --ext")
    source = {"steps_from": "--steps"}
    if args.steps is None:
        if not args.folds:
            raise SystemExit("give --steps or --folds")
        steps_list = best_steps(args.folds)
        args.steps = folds.final_steps(steps_list)
        source = {"steps_from": "median of fold best steps", "fold_best_steps": steps_list,
                  "median": float(np.median(steps_list))}
    train.setup(args.threads)
    train.configure_args(args)
    if args.pff:
        train.check_pff_references()
    train.check_cache(args.cache)
    games = folds.train_games(pff=args.pff)
    data_ = train.load_games(args.cache, games)
    for seed in args.seeds:
        rec_path = os.path.join(args.out, "final_s{}.json".format(seed))
        if os.path.isfile(rec_path) and os.path.isfile(os.path.join(args.out, "net_s{}.pt".format(seed))):
            with open(rec_path) as handle:
                if json.load(handle).get("stop_at") == args.steps:
                    log("final seed {}: done, skipped (resume)".format(seed))
                    continue
        # The same 2,000-step one-cycle schedule as the folds, stopped at step S (protocol §2).
        state, record = train.train_model(data_, games, train.STEPS, seed, log=log,
                                          stop_at=args.steps)
        record.update(source)
        record.update(train=games, crop_shares=train.crop_shares(games),
                      recipe=train.recipe(args.steps, None),
                      environment=train.environment(args.threads),
                      reference_md5=train.reference_md5(games, args.ext),
                      cache_md5={os.path.relpath(p, args.cache): train.md5_of(p)
                                 for p in train.cache_files(args.cache, games)})
        torch.save(state, os.path.join(args.out, "net_s{}.pt".format(seed)))
        write_json(os.path.join(args.out, "final_s{}.json".format(seed)), record)
        log("final seed {}: {} steps, {:.1f} s, {:.3f} s/step".format(
            seed, args.steps, record["train_seconds"], record["seconds_per_step"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
