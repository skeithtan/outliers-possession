"""The eight folds of the learned stage: train, choose the costs, predict. Not part of the submission.

    python -m tools.learned_spike.kfold --fold K --cache DIR --ext DIR --out DIR
        [--seeds 0 1 2 3] [--steps 2000] [--eval-every 200] [--threads 2]
        [--train-only | --select-only] [--engine-python PY]

This tool implements LEARNED_STAGE_PROTOCOL.md Sections 2 and 3 with the advisor changes of
9 November 2026, and LEARNED_STAGE_PROTOCOL_R2.md Section 1: the training set of every fold also
holds the 43 PFF training matches (raw view only), and M's cost grid keeps only the pairs with
H-A <= 2 x alive-dead (decoders.model_grid()). The folds come from folds.py, the recipe from
train.py, the decoders from decoders.py. Before it trains, the tool checks every PFF reference
md5 against R2 Section 8 (pff.check_references()).

One fold has three steps:

1. Train. For each seed the tool trains one model on the training matches of the fold. The
   checkpoint is the lowest mean loss over the two validation matches, checked every 200 steps
   out of 2,000. It writes OUT/fold{K}/net_s{seed}.pt and OUT/fold{K}/model_s{seed}.json. A
   model whose files exist with the same step count is not trained again.
2. Choose the costs. The fold prediction is the mean of the class probabilities over the four
   models and the four symmetric copies, rounded once to 4 decimals at 5 fps, then upsampled to
   25 fps. On the validation pair the tool decodes the whole grid for candidate M (three-state
   Viterbi, H-A x alive-dead) and for candidate H (hybrid decoder on the c22 labels, H-A only).
   It keeps the costs with the highest mean 2 s spellF1 over the pair. On a tie it keeps the
   lowest costs (H-A first). It writes OUT/fold{K}/costs.json and OUT/fold{K}/fold.json.
3. Predict. Only after step 2 has written its files, the tool predicts the test matches of the
   fold (raw view) and writes OUT/fold{K}/test/<game>_p5.npz (rounded 5 fps probabilities),
   <game>_M.csv and <game>_H.csv. It scores nothing: the scored labels come from engine.run
   with the fold weights (protocol Section 4).

The c22 labels of the validation and test matches come from c22.py (engine.run, default
config). If a run is missing, the tool starts c22.py with --engine-python.

The tool never loads a test match before step 3. It refuses the sealed matches and the
competition match (folds.check). Python 3.9.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import possession_metrics as pm  # noqa: E402
import score_game as sg  # noqa: E402
from engine import data  # noqa: E402
from tools.learned_spike import c22, decoders, folds, train  # noqa: E402

DEFAULT_ENGINE_PYTHON = os.path.join(os.path.expanduser("~"), ".venvs", "barca", "bin", "python")


def log(text):
    print(text, file=sys.stderr, flush=True)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(obj, handle, indent=1, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def write_csv(path, codes):
    labels = np.asarray(pm.LABELS)[codes]
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        for i, lab in enumerate(labels):
            handle.write("{},{}\n".format(i, lab))


def spell_f1(ref_codes, pred_codes):
    return float(sg.score_codes(ref_codes, pred_codes)[0]["spell_f1_mean"])


def c22_codes(games, ext, engine_python, n_frames):
    """{game: c22 codes}, running c22.py for the missing runs. Also {game: (path, md5)}."""
    paths = {g: c22.path_of(g, ext) for g in games}
    missing = [g for g in games if not os.path.isfile(paths[g])]
    if missing:
        log("c22 runs missing for {}, running engine.run".format(" ".join(missing)))
        subprocess.check_call([engine_python, "-m", "tools.learned_spike.c22", "--ext", ext,
                               "--jobs", "4", "--games"] + missing, cwd=ROOT)
    codes = {g: sg.read_codes(paths[g], n_frames[g]) for g in games}
    return codes, {g: [paths[g], train.md5_of(paths[g])] for g in games}


def model_paths(out, seed):
    return os.path.join(out, "net_s{}.pt".format(seed)), os.path.join(out, "model_s{}.json".format(seed))


def train_fold(spec, args, out):
    folds.assert_not_sealed(spec["train"] + spec["val"] + spec["test"])
    if args.pff:
        train.check_pff_references()
    train.check_cache(args.cache)
    data_ = None
    for seed in args.seeds:
        net_path, rec_path = model_paths(out, seed)
        if os.path.isfile(net_path) and os.path.isfile(rec_path):
            with open(rec_path) as handle:
                if json.load(handle).get("steps") == args.steps:
                    log("fold {} seed {}: done, skipped".format(spec["k"], seed))
                    continue
        if data_ is None:
            data_ = train.load_games(args.cache, spec["train"])
            data_.update(train.load_games(args.cache, spec["val"], raw_only=True))
        state, record = train.train_model(data_, spec["train"], args.steps, seed,
                                          val_games=spec["val"], eval_every=args.eval_every,
                                          log=log)
        record.update(fold=spec["k"], train=spec["train"], val=spec["val"],
                      crop_shares=train.crop_shares(spec["train"]),
                      recipe=train.recipe(args.steps, args.eval_every),
                      environment=train.environment(args.threads))
        torch.save(state, net_path)
        write_json(rec_path, record)
        log("fold {} seed {}: best step {} ({:.1f} s, {:.3f} s/step)".format(
            spec["k"], seed, record["best_step"], record["train_seconds"],
            record["seconds_per_step"]))


def ensemble_probs(nets, item):
    """(p5, frame probabilities) of the fold ensemble on the raw view of one match."""
    p5 = decoders.ensemble_p5([train.predict_p5(net, item["feats"]["raw"]) for net in nets])
    return p5, decoders.upsample(p5, len(item["raw"]))


def select_fold(spec, args, out):
    t0 = time.time()
    records, nets = [], []
    for seed in args.seeds:
        net_path, rec_path = model_paths(out, seed)
        with open(rec_path) as handle:
            rec = json.load(handle)
        rec.pop("history", None)
        records.append(rec)
        nets.append(train.load_net(net_path))
    val = train.load_games(args.cache, spec["val"], raw_only=True)
    n_frames = {g: len(val[g]["raw"]) for g in spec["val"]}
    base, c22_files = c22_codes(spec["val"], args.ext, args.engine_python, n_frames)

    grid_m = decoders.model_grid()
    rows_m = [{"cost_ha": cha, "cost_ad": cad, "scores": []} for cha, cad in grid_m]
    rows_h = [{"cost_ha": cha, "scores": []} for cha in decoders.GRID_HA]
    for game in spec["val"]:
        item = val[game]
        _, probs = ensemble_probs(nets, item)
        blocks = data.graded_blocks(game)
        for row, codes in zip(rows_m, decoders.decode_model(probs, blocks, grid_m)):
            row["scores"].append(spell_f1(item[args.select_ref], codes))
        for row, codes in zip(rows_h, decoders.decode_hybrid(probs, base[game],
                                                             decoders.GRID_HA)):
            row["scores"].append(spell_f1(item[args.select_ref], codes))
    for row in rows_m + rows_h:
        row["mean"] = float(np.mean(row["scores"]))
    best_m, best_h = decoders.choose(rows_m), decoders.choose(rows_h)
    costs = {
        "fold": spec["k"], "val": spec["val"],
        "M": {"cost_ha": best_m["cost_ha"], "cost_ad": best_m["cost_ad"],
              "val_mean_spell_f1_ctl2": best_m["mean"]},
        "H": {"cost_ha": best_h["cost_ha"], "val_mean_spell_f1_ctl2": best_h["mean"]},
        "table_M": rows_m, "table_H": rows_h,
        "grid_M_rule": "H-A <= {:g} x alive-dead (R2 change B), {} of {} pairs".format(
            decoders.MAX_HA_PER_AD, len(grid_m), len(decoders.full_model_grid())),
        "round_decimals": decoders.ROUND_DECIMALS, "log_floor": decoders.LOG_FLOOR,
        "select_ref": args.select_ref,
    }
    write_json(os.path.join(out, "costs.json"), costs)
    used = spec["train"] + spec["val"]
    record = {
        "fold": spec["k"], "test": spec["test"], "val": spec["val"], "train": spec["train"],
        "seeds": list(args.seeds), "models": records,
        "crop_shares": train.crop_shares(spec["train"]),
        "best_steps": [r["best_step"] for r in records],
        "costs": costs, "c22_val_runs": c22_files,
        "reference_md5": train.reference_md5(used, args.ext),
        "cache_md5": {os.path.relpath(p, args.cache): train.md5_of(p)
                      for p in train.cache_files(args.cache, spec["train"])
                      + train.cache_files(args.cache, spec["val"], lambda g: ("raw",))},
        "environment": train.environment(args.threads),
        "select_seconds": round(time.time() - t0, 2),
        "test_written": [],
    }
    fold_path = os.path.join(out, "fold.json")
    write_json(fold_path, record)
    log("fold {}: M costs ({}, {}) val {:.4f}; H cost {} val {:.4f} ({:.1f} s)".format(
        spec["k"], best_m["cost_ha"], best_m["cost_ad"], best_m["mean"], best_h["cost_ha"],
        best_h["mean"], record["select_seconds"]))

    # Step 3: the test matches, only after costs.json and fold.json are on disk.
    t1 = time.time()
    test = train.load_games(args.cache, spec["test"], raw_only=True)
    n_test = {g: len(test[g]["raw"]) for g in spec["test"]}
    test_base, test_c22 = c22_codes(spec["test"], args.ext, args.engine_python, n_test)
    tdir = os.path.join(out, "test")
    os.makedirs(tdir, exist_ok=True)
    for game in spec["test"]:
        p5, probs = ensemble_probs(nets, test[game])
        np.savez_compressed(os.path.join(tdir, game + "_p5.npz"), p5=p5)
        m = decoders.decode_model(probs, data.graded_blocks(game),
                                  [(best_m["cost_ha"], best_m["cost_ad"])])[0]
        h = decoders.decode_hybrid(probs, test_base[game], [best_h["cost_ha"]])[0]
        write_csv(os.path.join(tdir, game + "_M.csv"), m)
        write_csv(os.path.join(tdir, game + "_H.csv"), h)
        record["test_written"].append(game)
    record["c22_test_runs"] = test_c22
    record["predict_seconds"] = round(time.time() - t1, 2)
    write_json(fold_path, record)
    log("fold {}: wrote test predictions for {} ({:.1f} s)".format(
        spec["k"], " ".join(spec["test"]), record["predict_seconds"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--cache", required=True, help="build.py cache")
    parser.add_argument("--ext", required=True, help="c22.py cache (engine npz, runs, refs)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(train.FOLD_SEEDS))
    parser.add_argument("--steps", type=int, default=train.STEPS)
    parser.add_argument("--eval-every", type=int, default=train.EVAL_EVERY)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--engine-python", default=DEFAULT_ENGINE_PYTHON)
    # Round 3 switches (LEARNED_STAGE_PROTOCOL_R3.md, DRAFT); the defaults are round 2.
    # Round 4 switches (LEARNED_STAGE_PROTOCOL_R4.md Section 1): train.add_switch_args().
    train.add_switch_args(parser)
    parser.add_argument("--no-pff", dest="pff", action="store_false",
                        help="R3b: no PFF match in any training set, no PFF file read")
    parser.add_argument("--select-ref", choices=("ctl2", "raw"), default="ctl2",
                        help="reference view of the per-fold cost choice (recorded only in R3)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true")
    mode.add_argument("--select-only", action="store_true")
    args = parser.parse_args(argv)
    train.setup(args.threads)
    train.configure_args(args)
    spec = folds.fold(args.fold, pff=args.pff)
    out = os.path.join(args.out, "fold{}".format(args.fold))
    os.makedirs(out, exist_ok=True)
    if not args.select_only:
        train_fold(spec, args, out)
    if not args.train_only:
        select_fold(spec, args, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
