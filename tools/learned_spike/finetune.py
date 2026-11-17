"""Round 8 arm N: a short fine-tune of trained networks with a share of Metrica crops. Not part of the submission.

    python -m tools.learned_spike.finetune --init FOLDS/fold0/net_s0.pt --init FOLDS/fold0/net_s1.pt \
        --steps 200 400 --out DIR/fold0 --cache C_TRAIN2TJ [train switches of the recipe, as kfold.py]

LEARNED_STAGE_PROTOCOL_R8.md Section 3 row N and the "Arm N" rule of Section 4. Each --init is
a trained network (r4_screen: <dir>/net.pt beside model.json; kfold: net_s<s>.pt beside
model_s<s>.json; final: net_s<s>.pt beside final_s<s>.json). Its training record gives its
seed and its training matches ("train"). The fine-tune (train.finetune_model) starts from its
weights, runs K steps of the recipe's one-cycle schedule at peak --lr (default train.FT_LR =
5e-4, a quarter of the recipe's 2e-3), and draws round(--share x BATCH) crops of every batch
(default 0.5, 8 of 16) from the Metrica matches of the network's own training matches, the
other crops from the recipe's mix over the same matches. So the fine-tune reads no match the
network did not train on; for a fold network the Metrica crops come from the two Metrica
matches its fold trains on. Validation matches are not read; the last step is kept.

Output: OUT/K<steps>/net_s<seed>.pt, OUT/K<steps>/model_s<seed>.json (the record, with the
switches and the feature version, so export.py and pf_gate read the version) and
OUT/K<steps>/DONE_s<seed>. A job with DONE is skipped. --forced-loss team (N2) puts the team
call alone on the Metrica crops (train.finetune_model). All --init of one call must share the
same training matches (they are loaded once).

The random draws use seed 800 + the network's seed (--seed-offset), so the crops are not those
of the network's first steps. Deterministic for a given seed, thread count and torch build.
Python 3.9.
"""

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from tools.learned_spike import folds, train  # noqa: E402
from tools.learned_spike.kfold import log, write_json  # noqa: E402


def record_of(path):
    """(record path, record) of a trained network (the names of train.net_version)."""
    folder, name = os.path.split(os.path.abspath(path))
    cands = [os.path.join(folder, "model.json")]
    if name.startswith("net_s") and name.endswith(".pt"):
        s = name[len("net_s"):-len(".pt")]
        cands = [os.path.join(folder, "model_s{}.json".format(s)),
                 os.path.join(folder, "final_s{}.json".format(s))] + cands
    for rec in cands:
        if os.path.isfile(rec):
            with open(rec) as handle:
                return rec, json.load(handle)
    raise SystemExit("{}: no training record beside it".format(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--init", action="append", required=True,
                        help="a trained network (repeat for several seeds)")
    parser.add_argument("--steps", type=int, nargs="+", default=list(train.FT_STEPS))
    parser.add_argument("--lr", type=float, default=train.FT_LR)
    parser.add_argument("--share", type=float, default=train.FT_SHARE)
    parser.add_argument("--provider", default=train.FT_PROVIDER,
                        choices=("metrica", "dfl", "skillcorner"))
    parser.add_argument("--seed-offset", type=int, default=800)
    parser.add_argument("--forced-loss", choices=train.FT_LOSSES, default="full",
                        help="loss on the forced (Metrica) crops: full (arm N) or team (N2: the "
                             "team call on live bins only, protocol R8 note 6.3)")
    parser.add_argument("--cache", required=True, help="build.py training cache")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=2)
    train.add_switch_args(parser)
    args = parser.parse_args(argv)
    if not 0.0 < args.share < 1.0:
        parser.error("--share must lie in (0, 1)")
    jobs = []
    games = None
    for path in args.init:
        rec_path, rec = record_of(path)
        g = list(rec["train"])
        if games is None:
            games = g
        elif g != games:
            raise SystemExit("{}: other training matches than the first --init".format(path))
        jobs.append((path, rec_path, int(rec["seed"])))
    folds.check_train_all(folds.assert_not_sealed(games))
    if any(folds.provider(x) == "pff" for x in games):
        raise SystemExit("a PFF match in the training matches: PFF is a test only")
    train.setup(args.threads)
    train.configure_args(args)
    train.check_cache(args.cache)
    todo = [(p, r, s, k) for p, r, s in jobs for k in args.steps
            if not os.path.isfile(os.path.join(args.out, "K{}".format(k), "DONE_s{}".format(s)))]
    if not todo:
        log("all fine-tunes done")
        return 0
    data_ = train.load_games(args.cache, games)
    for path, rec_path, seed, k in todo:
        out = os.path.join(args.out, "K{}".format(k))
        os.makedirs(out, exist_ok=True)
        init = torch.load(path, map_location="cpu", weights_only=True)
        state, record = train.finetune_model(data_, games, k, args.seed_offset + seed, init,
                                             lr=args.lr, share=args.share,
                                             provider=args.provider, log=log,
                                             forced_loss=args.forced_loss)
        record.update(seed=seed, rng_seed=args.seed_offset + seed, init=os.path.abspath(path),
                      init_md5=train.md5_of(path), init_record=os.path.abspath(rec_path),
                      train=games, crop_shares_mix=train.crop_shares(games),
                      recipe=train.recipe(k, None), environment=train.environment(args.threads),
                      cache_md5={os.path.relpath(p, args.cache): train.md5_of(p)
                                 for p in train.cache_files(args.cache, games)})
        torch.save(state, os.path.join(out, "net_s{}.pt".format(seed)))
        write_json(os.path.join(out, "model_s{}.json".format(seed)), record)
        write_json(os.path.join(out, "DONE_s{}".format(seed)),
                   {"steps": k, "seconds": record["train_seconds"],
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        log("fine-tune seed {} K {}: {:.1f} s ({:.3f} s/step) -> {}".format(
            seed, k, record["train_seconds"], record["seconds_per_step"], out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
