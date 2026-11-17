"""Round 3, R3b: the unseen-provider diagnostic (LEARNED_STAGE_PROTOCOL_R3.md §10, informational).

    # torch venv: train one diagnostic model (seed 0, the R3b recipe, one provider removed)
    python -m tools.learned_spike.r3_diag train --name no-metrica --cache DIR --out DIR
    # torch venv: its 5 fps probabilities on the scored matches and on the tuning matches
    python -m tools.learned_spike.r3_diag predict --name no-metrica --cache DIR --out DIR
    # numpy + engine: scores, cost choices, RESULT.md and result.json
    python -m tools.learned_spike.r3_diag score --cache DIR --out DIR --c23 DIR --r3b DIR
    # third pool (no-skillcorner): train and predict as above with --name no-skillcorner, then
    python -m tools.learned_spike.r3_diag predict-folds --cache DIR --out DIR --r3b DIR
    python -m tools.learned_spike.r3_diag score --names no-skillcorner --cache DIR --out DIR \
        --c23 DIR --r3b DIR --sc-fine DIR --c23-sc DIR

Not part of the submission and outside the frozen criteria: it decides nothing by itself. It asks
how much of R3b's development gain survives when the scored provider is absent from training AND
from the choice of the decoder costs.

Two models, seed 0, the R3b recipe unchanged (labels as provided, no short weight, SkillCorner
match at weight 1/4, no PFF, 2,000 steps, checkpoint at the lowest mean validation loss every 200
steps) except the pool (pool()):

- no-metrica: no Metrica match in training or validation; scored on METRICA-S1, S2, S3.
- no-dfl: no open DFL match in training or validation; scored on the 5 open DFL matches.

The validation pair is the one the recipe gives the first fold that tests the held-out provider
(folds.fold(REF_FOLD).val): it holds one development match of the other provider and one
SkillCorner match, so the checkpoint rule is unchanged and sees no held-out-provider match. The
training set is every other match of the 28 that is not of the held-out provider (the SkillCorner
test matches of that fold included: nothing of them is scored here).

Cost choices reported side by side for each model on its held-out provider:

- honest: the grid pair with the highest mean spellF1 on the references of the other development
  provider only, decoded from R3b seed 0's out-of-fold probabilities of those matches
  (honest_choice()). It never reads a reference of the held-out provider. Weakness: the costs are
  tuned on another model's probabilities (R3b's fold models, which saw the held-out provider) and
  on the other provider's labelling convention; they transfer only if the two models are
  calibrated alike.
- honest-own (secondary): the same choice on the diagnostic model's own probabilities of those
  matches. In-sample for all but the validation match, so over-confident.
- R3b final costs (check.json "final_costs"): chosen with the held-out references; reference only.
- the whole grid; its best pair is the oracle, not usable.

Third pool, no-skillcorner (the unseen-provider probe on real automatic tracking, review
item W2): the 3 Metrica and 5 open DFL matches only, no SkillCorner match in training or validation.
The recipe's validation pairs all hold a SkillCorner match, so the pair keeps the recipe's logic
(one match of each other provider, picked by folds.open_validation) with the SkillCorner slot
given to the second development provider: the development members of the two pools above,
open_validation(0) = DFL-J03WMX and open_validation(3) = METRICA-S1. Training: the other six
development matches. Scored on the 20 SkillCorner matches against BOTH SkillCorner references:
`phases` (the reference as provided, cache labels `raw`) and `possgroup` (the fine reference of
tools/skillcorner_fine.py, --sc-fine labels `raw`), beside c23 (--c23-sc). Costs:

- honest: rule B of r3_nested.py (best mean of the Metrica and DFL means) on the 8 development
  references, decoded from R3b seed 0's out-of-fold probabilities; no SkillCorner reference is
  read. By construction it is R3b's interim final_costs (checked and recorded).
- honest_own: the same rule on this model's own probabilities of the 8 development matches
  (in-sample for six of them).
- oracle per reference, and the whole grid per reference.

Also (predict-folds, no training): R3b's seed-0 fold models on their held-out SkillCorner matches
(fold k tests SC[j] for j mod 8 = k, folds.fold), at the fold's own nested costs (check.json
matches[OPEN[k]]["costs"], chosen without any SkillCorner reference) and at R3b's final costs,
against both references.

The decoder, grid and scorer are those of r3_nested.py. The sealed DFL matches, BAR-ATH and every
PFF match are never read (folds.check(), pool() assertions). Python 3.9.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from tools.learned_spike import folds, r3_nested  # noqa: E402

SEED = 0
GRID = r3_nested.GRID
# R3b switches, unchanged (LEARNED_STAGE_PROTOCOL_R3.md §9).
SWITCHES = {"target": "raw", "short_weight": "off", "mix": "sc_quarter"}
DIAGS = {
    "no-metrica": {"held": "metrica", "tune": "dfl", "ref_fold": 0},   # fold 0 tests METRICA-S1
    "no-dfl": {"held": "dfl", "tune": "metrica", "ref_fold": 3},       # fold 3 tests DFL-J03WMX
}
SC_DIAG = "no-skillcorner"
DIAGS[SC_DIAG] = {"held": "skillcorner", "tune": "dev", "ref_fold": None}
NAMES = ("no-metrica", "no-dfl", SC_DIAG)
SC_REFS = ("phases", "possgroup")      # the two SkillCorner references
FORBIDDEN = frozenset(folds.SEALED + (folds.COMPETITION,)) | folds.NEVER_TRAIN


def log(text):
    print(text, file=sys.stderr, flush=True)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(obj, handle, indent=1, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def pool(name):
    """{"name", "held", "train", "val", "scored", "tune"} of one diagnostic model.

    scored: the development matches of the held-out provider. tune: those of the other
    development provider (the only references the honest cost choice reads).
    """
    d = DIAGS[name]
    held, tune = d["held"], d["tune"]
    if name == SC_DIAG:
        return sc_pool()
    val = list(folds.fold(d["ref_fold"], pff=False)["val"])
    train = [g for g in folds.ALL if folds.provider(g) != held and g not in val]
    scored = [g for g in folds.OPEN if folds.provider(g) == held]
    tuning = [g for g in folds.OPEN if folds.provider(g) == tune]
    every = train + val + scored + tuning
    assert not set(every) & FORBIDDEN, "sealed, competition or PFF match in the pool"
    assert not any(g.startswith("PFF-") for g in every), "PFF match in the pool"
    assert not any(folds.provider(g) == held for g in train + val), "held-out provider in pool"
    assert not set(train) & set(val)
    folds.check_all(every)
    return {"name": name, "held": held, "train": train, "val": val, "scored": scored,
            "tune": tuning, "ref_fold": d["ref_fold"]}


def sc_pool():
    """The no-skillcorner pool (module docstring)."""
    val = [folds.open_validation(0), folds.open_validation(3)]
    train = [g for g in folds.OPEN if g not in val]
    scored = list(folds.SC)
    tuning = list(folds.OPEN)
    every = train + val + scored + tuning
    assert not set(every) & FORBIDDEN, "sealed, competition or PFF match in the pool"
    assert not any(folds.provider(g) in ("skillcorner", "pff") for g in train + val), \
        "SkillCorner or PFF match in training or validation"
    assert sorted(folds.provider(g) for g in val) == ["dfl", "metrica"]
    assert not set(train) & set(val)
    folds.check_all(every)
    return {"name": SC_DIAG, "held": "skillcorner", "train": train, "val": val,
            "scored": scored, "tune": tuning, "ref_fold": None}


def read_ref(cache, game):
    """int8 reference as provided (labels/<game>.npz key raw). The only reference reader."""
    folds.check(game)
    with np.load(os.path.join(cache, "labels", game + ".npz")) as lab:
        return lab["raw"].astype(np.int8)


def read_sc_ref(game, ref, cache, fine):
    """int8 SkillCorner reference: `phases` (cache labels raw) or `possgroup` (fine labels raw).

    Both share the frame grid, the dead-ball frames and the graded mask (skillcorner_fine.py).
    """
    if folds.provider(folds.check(game)) != "skillcorner":
        raise SystemExit("{} is not a SkillCorner match".format(game))
    if ref == "phases":
        return read_ref(cache, game)
    if ref != "possgroup":
        raise ValueError(ref)
    with np.load(os.path.join(fine, "labels", game + ".npz")) as lab:
        return lab["raw"].astype(np.int8)


def read_p5(pred_dir, game):
    with np.load(os.path.join(pred_dir, game + "_p5.npz")) as npz:
        return npz["p5"]


def default_blocks(game):
    from engine import data
    return data.graded_blocks(game)


def grid_scores(games, pred_dir, cache, blocks_fn=default_blocks):
    """{game: [spellF1 of each GRID pair]}, {game: frame probs}, {game: ref} (r3_nested check)."""
    from tools.learned_spike import decoders
    from tools.learned_spike import r3_scope as r
    scores, probs, refs = {}, {}, {}
    for game in games:
        refs[game] = read_ref(cache, game)
        probs[game] = decoders.upsample(read_p5(pred_dir, game), len(refs[game]))
        paths = decoders.decode_model(probs[game], blocks_fn(game), GRID)
        scores[game] = [r.spell_f1(refs[game], p) for p in paths]
    return scores, probs, refs


def pick(scores, games):
    """Index into GRID: highest mean spellF1 over `games`; ties -> lower index (r3_nested)."""
    best, best_v = None, -1.0
    for i in range(len(GRID)):
        v = float(np.mean([scores[g][i] for g in games]))
        if v > best_v + 1e-12:
            best, best_v = i, v
    return best, best_v


def honest_choice(spec, pred_dir, cache, blocks_fn=default_blocks):
    """(index, tuning mean, {game: scores}) chosen on the tuning provider only.

    Reads references and probabilities of spec["tune"] only; refuses any match of the held-out
    provider.
    """
    games = list(spec["tune"])
    assert games and not any(folds.provider(g) == spec["held"] for g in games), \
        "honest choice would read a held-out-provider reference"
    scores, _, _ = grid_scores(games, pred_dir, cache, blocks_fn)
    i, v = pick(scores, games)
    return i, v, scores


def cmd_train(args):
    import torch
    from tools.learned_spike import train
    spec = pool(args.name)
    train.setup(args.threads)
    train.configure(**SWITCHES)
    train.check_cache(args.cache)
    out = os.path.join(args.out, args.name)
    os.makedirs(out, exist_ok=True)
    net_path = os.path.join(out, "net_s%d.pt" % SEED)
    rec_path = os.path.join(out, "model_s%d.json" % SEED)
    if os.path.isfile(net_path) and os.path.isfile(rec_path):
        with open(rec_path) as handle:
            if json.load(handle).get("steps") == args.steps:
                log("{}: done, skipped".format(args.name))
                return 0
    data_ = train.load_games(args.cache, spec["train"])
    data_.update(train.load_games(args.cache, spec["val"], raw_only=True))
    state, record = train.train_model(data_, spec["train"], args.steps, SEED,
                                      val_games=spec["val"], eval_every=args.eval_every, log=log)
    record.update(diagnostic=args.name, held_out=spec["held"], train=spec["train"],
                  val=spec["val"], ref_fold=spec["ref_fold"],
                  crop_shares=train.crop_shares(spec["train"]),
                  recipe=train.recipe(args.steps, args.eval_every),
                  environment=train.environment(args.threads),
                  cache_md5={os.path.relpath(p, args.cache): train.md5_of(p)
                             for p in train.cache_files(args.cache, spec["train"])
                             + train.cache_files(args.cache, spec["val"], lambda g: ("raw",))})
    torch.save(state, net_path)
    write_json(rec_path, record)
    log("{}: best step {} ({:.1f} s, {:.3f} s/step)".format(
        args.name, record["best_step"], record["train_seconds"], record["seconds_per_step"]))
    return 0


def cmd_predict(args):
    from tools.learned_spike import decoders, train
    spec = pool(args.name)
    train.setup(args.threads)
    net = train.load_net(os.path.join(args.out, args.name, "net_s%d.pt" % SEED))
    pdir = os.path.join(args.out, args.name, "p5")
    os.makedirs(pdir, exist_ok=True)
    for game in spec["scored"] + spec["tune"]:
        item = train.load_games(args.cache, [game], raw_only=True)[game]
        p5 = decoders.ensemble_p5([train.predict_p5(net, item["feats"]["raw"])])
        np.savez_compressed(os.path.join(pdir, game + "_p5.npz"), p5=p5, seeds=np.asarray([SEED]))
        log("{}: {}".format(args.name, game))
    return 0


def match_row(game, i, scores, probs, refs, c23_dir, r3b, blocks_fn):
    """Per-match line of one cost choice: spellF1, c23, delta, counts, R3b out-of-fold."""
    import possession_metrics as pm
    import score_game as sg
    from tools.learned_spike import decoders
    from tools.learned_spike import r3_scope as r
    ref = refs[game]
    runs = pm.valid_runs(ref)
    codes = decoders.decode_model(probs[game], blocks_fn(game), [GRID[i]])[0]
    c23 = sg.read_codes(os.path.join(c23_dir, game + ".csv"), len(ref))
    f, f23 = scores[game][i], r.spell_f1(ref, c23)
    m = r3b["matches"][game]
    return {"costs": list(GRID[i]), "spellF1": f, "c23": f23, "delta": f - f23,
            "n": len(r.spells(pm.fill_abstentions(codes, runs), runs)),
            "n_c23": len(r.spells(pm.fill_abstentions(c23, runs), runs)),
            "n_ref": len(r.spells(ref, runs)),
            "r3b_oof_spellF1": m["spellF1"], "r3b_oof_delta": m["delta"],
            "r3b_oof_costs": list(m["costs"]), "r3b_oof_n": m["n"]}


def score_one(name, args, r3b, blocks_fn=default_blocks):
    spec = pool(name)
    r3b_p5 = os.path.join(args.r3b, "interim", "p5")
    own_p5 = os.path.join(args.out, name, "p5")
    # 1. honest choices: tuning provider only (never a held-out reference)
    hi, hv, h_scores = honest_choice(spec, r3b_p5, args.cache, blocks_fn)
    oi, ov, o_scores = honest_choice(spec, own_p5, args.cache, blocks_fn)
    # 2. the held-out provider, scored
    scores, probs, refs = grid_scores(spec["scored"], own_p5, args.cache, blocks_fn)
    fi = GRID.index(tuple(r3b["final_costs"]))
    best, best_v = pick(scores, spec["scored"])
    choices = {
        "honest": {"index": hi, "tune_mean": hv,
                   "how": "tuned on R3b seed-0 out-of-fold probabilities of " + spec["tune"][0]
                          .split("-")[0] + " references only"},
        "honest_own": {"index": oi, "tune_mean": ov,
                       "how": "tuned on this model's own (in-sample) probabilities of the "
                              "tuning provider"},
        "r3b_final": {"index": fi, "how": "R3b final costs (chosen with the held-out "
                                          "references): reference only"},
        "oracle": {"index": best, "how": "best pair on the scored matches: oracle, not usable"},
    }
    for key, ch in choices.items():
        rows = {g: match_row(g, ch["index"], scores, probs, refs, args.c23, r3b, blocks_fn)
                for g in spec["scored"]}
        ch["costs"] = list(GRID[ch["index"]])
        ch["matches"] = rows
        ch["mean_spellF1"] = float(np.mean([rows[g]["spellF1"] for g in spec["scored"]]))
        ch["mean_delta"] = float(np.mean([rows[g]["delta"] for g in spec["scored"]]))
    r3b_mean = float(np.mean([r3b["matches"][g]["delta"] for g in spec["scored"]]))
    rec_path = os.path.join(args.out, name, "model_s%d.json" % SEED)
    rec = {}
    if os.path.isfile(rec_path):
        with open(rec_path) as handle:
            rec = json.load(handle)
    return {"pool": spec, "choices": choices, "r3b_oof_mean_delta": r3b_mean,
            "grid": [{"costs": list(GRID[i]), "scores": {g: scores[g][i] for g in spec["scored"]},
                      "mean": float(np.mean([scores[g][i] for g in spec["scored"]])),
                      "tune_r3b_mean": float(np.mean([h_scores[g][i] for g in spec["tune"]])),
                      "tune_own_mean": float(np.mean([o_scores[g][i] for g in spec["tune"]]))}
                     for i in range(len(GRID))],
            "model": {k: rec.get(k) for k in ("steps", "best_step", "best_val_mean",
                                              "train_seconds", "crop_shares", "recipe",
                                              "environment")}}


def fmt(x):
    return "{:+.3f}".format(x)


def result_md(res):
    lines = ["# R3b unseen-provider diagnostic (informational, decides nothing)", "",
             "LEARNED_STAGE_PROTOCOL_R3.md §10. Seed 0, the R3b recipe with one development "
             "provider removed from training and validation. Scores against the references as "
             "provided; decoder, grid and scorer of r3_nested.py; c23 outputs on the same "
             "matches. R3b out-of-fold: seed 0 fold models, nested costs (interim check.json), "
             "i.e. the provider seen in training.", ""]
    for name in NAMES:
        if name not in res["models"] or name == SC_DIAG:
            continue
        m = res["models"][name]
        p = m["pool"]
        lines += ["## {} (held out: {})".format(name, p["held"]), "",
                  "- training ({}): {}".format(len(p["train"]), " ".join(p["train"])),
                  "- validation: {} (fold {} of the recipe)".format(" ".join(p["val"]),
                                                                    p["ref_fold"]),
                  "- crop shares: {}; best step {} of {}".format(
                      m["model"].get("crop_shares"), m["model"].get("best_step"),
                      m["model"].get("steps")), ""]
        for key in ("honest", "honest_own", "r3b_final", "oracle"):
            ch = m["choices"][key]
            label = {"honest": "(a) honest", "honest_own": "(a2) honest, own probabilities",
                     "r3b_final": "(b) R3b final costs", "oracle": "(c) oracle, not usable"}[key]
            lines += ["### {}: costs ({:g}, {:g}); {}".format(label, ch["costs"][0], ch["costs"][1],
                                                              ch["how"]), "",
                      "| match | spellF1 | c23 | delta | possessions (model / c23 / ref) "
                      "| R3b OOF spellF1 | R3b OOF delta |", "|---|---|---|---|---|---|---|"]
            for g in p["scored"]:
                r = ch["matches"][g]
                lines.append("| {} | {:.4f} | {:.4f} | {} | {} / {} / {} | {:.4f} | {} |".format(
                    g, r["spellF1"], r["c23"], fmt(r["delta"]), r["n"], r["n_c23"], r["n_ref"],
                    r["r3b_oof_spellF1"], fmt(r["r3b_oof_delta"])))
            lines += ["| mean | {:.4f} | | {} | | | {} |".format(
                ch["mean_spellF1"], fmt(ch["mean_delta"]), fmt(m["r3b_oof_mean_delta"])), ""]
        lines += ["### (c) whole grid (mean spellF1 on the scored matches)", "",
                  "| H-A \\ alive-dead | " + " | ".join("{:g}".format(b)
                                                       for b in r3_nested.GRID_AD) + " |",
                  "|---" * (len(r3_nested.GRID_AD) + 1) + "|"]
        best = m["choices"]["oracle"]["index"]
        hon = m["choices"]["honest"]["index"]
        for a in r3_nested.GRID_HA:
            cells = []
            for b in r3_nested.GRID_AD:
                i = GRID.index((a, b))
                mark = (" **oracle, not usable**" if i == best else "") + \
                       (" (honest)" if i == hon else "")
                cells.append("{:.4f}{}".format(m["grid"][i]["mean"], mark))
            lines.append("| {:g} | ".format(a) + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


# -- the no-skillcorner pool: scoring on SkillCorner against both references ------------------

def dev_choice(pred_dir, cache, blocks_fn=default_blocks):
    """(index, objective, {game: scores}): rule B (r3_nested.nested_pick, all 8) on the
    development references only. Never reads a SkillCorner reference."""
    games = list(folds.OPEN)
    assert not any(folds.provider(g) == "skillcorner" for g in games)
    scores, _, _ = grid_scores(games, pred_dir, cache, blocks_fn)
    i, v = r3_nested.nested_pick(scores, None)
    return i, v, scores


def sc_grid(games, pred_dir, cache, fine, blocks_fn=default_blocks):
    """{ref: {game: [spellF1 per GRID pair]}}, {game: paths}, {ref: {game: reference}}.

    One decode per match and grid pair, scored against both SkillCorner references.
    """
    from tools.learned_spike import decoders
    from tools.learned_spike import r3_scope as r
    scores = {ref: {} for ref in SC_REFS}
    refs = {ref: {} for ref in SC_REFS}
    paths = {}
    for game in games:
        for ref in SC_REFS:
            refs[ref][game] = read_sc_ref(game, ref, cache, fine)
        n = len(refs["phases"][game])
        assert len(refs["possgroup"][game]) == n
        assert np.array_equal(refs["phases"][game] == pm_X(), refs["possgroup"][game] == pm_X())
        probs = decoders.upsample(read_p5(pred_dir, game), n)
        paths[game] = decoders.decode_model(probs, blocks_fn(game), GRID)
        for ref in SC_REFS:
            scores[ref][game] = [r.spell_f1(refs[ref][game], q) for q in paths[game]]
    return scores, paths, refs


def pm_X():
    import possession_metrics as pm
    return pm.X


def sc_rows(games, idx_of, scores, paths, refs, c23_sc):
    """{ref: {game: row}} for a cost index per game (idx_of(game))."""
    import possession_metrics as pm
    import score_game as sg
    from tools.learned_spike import r3_scope as r
    out = {}
    for ref in SC_REFS:
        rows = {}
        for game in games:
            i = idx_of(game)
            gt = refs[ref][game]
            runs = pm.valid_runs(gt)
            codes = np.asarray(paths[game][i], np.int8)
            c23 = sg.read_codes(os.path.join(c23_sc, game + ".csv"), len(gt))
            f, f23 = scores[ref][game][i], r.spell_f1(gt, c23)
            rows[game] = {"costs": list(GRID[i]), "spellF1": f, "c23": f23, "delta": f - f23,
                          "n": len(r.spells(pm.fill_abstentions(codes, runs), runs)),
                          "n_c23": len(r.spells(pm.fill_abstentions(c23, runs), runs)),
                          "n_ref": len(r.spells(gt, runs))}
        out[ref] = {"matches": rows,
                    "mean_spellF1": float(np.mean([rows[g]["spellF1"] for g in games])),
                    "mean_c23": float(np.mean([rows[g]["c23"] for g in games])),
                    "mean_delta": float(np.mean([rows[g]["delta"] for g in games])),
                    "up": int(sum(rows[g]["delta"] > 0 for g in games))}
    return out


def fold_of_sc(game):
    """The R3b fold that tests a SkillCorner match (folds.fold: SC[j] with j mod 8 = k)."""
    k = folds.SC.index(game) % folds.N_FOLDS
    assert game in folds.fold(k, pff=False)["test"]
    return k


def cmd_predict_folds(args):
    """R3b seed-0 fold models on their held-out SkillCorner matches (no training)."""
    from tools.learned_spike import decoders, train
    train.setup(args.threads)
    pdir = os.path.join(args.out, "r3b_folds", "p5")
    os.makedirs(pdir, exist_ok=True)
    for game in folds.SC:
        k = fold_of_sc(game)
        net = train.load_net(os.path.join(args.r3b, "folds", "fold%d" % k, "net_s%d.pt" % SEED))
        item = train.load_games(args.cache, [game], raw_only=True)[game]
        p5 = decoders.ensemble_p5([train.predict_p5(net, item["feats"]["raw"])])
        np.savez_compressed(os.path.join(pdir, game + "_p5.npz"), p5=p5,
                            seeds=np.asarray([SEED]), fold=k)
        log("r3b fold {}: {}".format(k, game))
    return 0


def grid_table(scores, games):
    return [{"costs": list(GRID[i]),
             "mean": {ref: float(np.mean([scores[ref][g][i] for g in games])) for ref in SC_REFS}}
            for i in range(len(GRID))]


def score_sc(args, r3b, blocks_fn=default_blocks):
    spec = pool(SC_DIAG)
    games = spec["scored"]
    r3b_p5 = os.path.join(args.r3b, "interim", "p5")
    own_p5 = os.path.join(args.out, SC_DIAG, "p5")
    fi = GRID.index(tuple(r3b["final_costs"]))
    res = {"pool": spec, "refs": list(SC_REFS)}
    if os.path.isdir(own_p5):
        hi, hv, _ = dev_choice(r3b_p5, args.cache, blocks_fn)
        oi, ov, _ = dev_choice(own_p5, args.cache, blocks_fn)
        scores, paths, refs = sc_grid(games, own_p5, args.cache, args.sc_fine, blocks_fn)
        choices = {
            "honest": {"index": hi, "objective": hv, "equals_r3b_final": hi == fi,
                       "how": "rule B on the 8 development references, R3b seed-0 out-of-fold "
                              "probabilities; no SkillCorner reference"},
            "honest_own": {"index": oi, "objective": ov,
                           "how": "rule B on this model's own probabilities of the 8 "
                                  "development matches (in-sample for six)"}}
        for ref in SC_REFS:
            b, _ = pick(scores[ref], games)
            choices["oracle_" + ref] = {"index": b, "how": "best pair on the SkillCorner "
                                                           "matches vs " + ref + ": oracle"}
        for ch in choices.values():
            ch["costs"] = list(GRID[ch["index"]])
            ch["scores"] = sc_rows(games, lambda g, i=ch["index"]: i, scores, paths, refs,
                                   args.c23_sc)
        rec_path = os.path.join(args.out, SC_DIAG, "model_s%d.json" % SEED)
        rec = {}
        if os.path.isfile(rec_path):
            with open(rec_path) as handle:
                rec = json.load(handle)
        res["model"] = {"choices": choices, "grid": grid_table(scores, games),
                        "record": {k: rec.get(k) for k in (
                            "steps", "best_step", "best_val_mean", "train_seconds",
                            "crop_shares", "recipe", "environment")}}
    fold_p5 = os.path.join(args.out, "r3b_folds", "p5")
    if os.path.isdir(fold_p5):
        scores, paths, refs = sc_grid(games, fold_p5, args.cache, args.sc_fine, blocks_fn)
        nested = {g: GRID.index(tuple(r3b["matches"][folds.OPEN[fold_of_sc(g)]]["costs"]))
                  for g in games}
        res["r3b_folds"] = {
            "nested": {"how": "each fold's own nested costs (rule B without its development "
                              "test match; no SkillCorner reference)",
                       "scores": sc_rows(games, lambda g: nested[g], scores, paths, refs,
                                         args.c23_sc)},
            "final": {"how": "R3b final costs", "costs": list(GRID[fi]),
                      "scores": sc_rows(games, lambda g: fi, scores, paths, refs, args.c23_sc)},
            "grid": grid_table(scores, games),
            "fold_of": {g: fold_of_sc(g) for g in games}}
    return res


def sc_table(lines, title, block, games):
    lines += ["### " + title, "",
              "| match | " + " | ".join("{0} spellF1 | {0} c23 | {0} delta".format(r)
                                       for r in SC_REFS)
              + " | possessions model / c23 / phases / possgroup |",
              "|---" * (1 + 3 * len(SC_REFS) + 1) + "|"]
    for g in games:
        cells = []
        for ref in SC_REFS:
            row = block[ref]["matches"][g]
            cells += ["{:.4f}".format(row["spellF1"]), "{:.4f}".format(row["c23"]),
                      fmt(row["delta"])]
        a, b = block["phases"]["matches"][g], block["possgroup"]["matches"][g]
        lines.append("| {} | {} | {} / {} / {} / {} |".format(
            g, " | ".join(cells), a["n"], a["n_c23"], a["n_ref"], b["n_ref"]))
    cells = []
    for ref in SC_REFS:
        cells += ["{:.4f}".format(block[ref]["mean_spellF1"]),
                  "{:.4f}".format(block[ref]["mean_c23"]),
                  "{} ({}/{} up)".format(fmt(block[ref]["mean_delta"]), block[ref]["up"],
                                         len(games))]
    lines += ["| mean | " + " | ".join(cells) + " | |", ""]


def grid_md(lines, title, grid):
    for ref in SC_REFS:
        lines += ["### {} vs {} (mean spellF1 over the 20)".format(title, ref), "",
                  "| H-A \\ alive-dead | " + " | ".join("{:g}".format(b)
                                                       for b in r3_nested.GRID_AD) + " |",
                  "|---" * (len(r3_nested.GRID_AD) + 1) + "|"]
        for a in r3_nested.GRID_HA:
            lines.append("| {:g} | ".format(a) + " | ".join(
                "{:.4f}".format(grid[GRID.index((a, b))]["mean"][ref])
                for b in r3_nested.GRID_AD) + " |")
        lines.append("")


def result_sc_md(res):
    p = res["pool"]
    games = p["scored"]
    lines = ["## no-skillcorner: unseen provider on real automatic tracking (informational)", "",
             "- training ({}): {}".format(len(p["train"]), " ".join(p["train"])),
             "- validation: {} (no SkillCorner match)".format(" ".join(p["val"])),
             "- scored: the 20 SkillCorner matches against `phases` (reference as provided, "
             "coarse) and `possgroup` (possession.group, fine); c23 on the same matches", ""]
    if "model" in res:
        m = res["model"]
        lines.append("- crop shares {}; best step {} of {}".format(
            m["record"].get("crop_shares"), m["record"].get("best_step"),
            m["record"].get("steps")))
        lines.append("")
        for key, ch in m["choices"].items():
            extra = " (= R3b final costs: {})".format(ch["equals_r3b_final"]) \
                if "equals_r3b_final" in ch else ""
            sc_table(lines, "{}: costs ({:g}, {:g}); {}{}".format(
                key, ch["costs"][0], ch["costs"][1], ch["how"], extra), ch["scores"], games)
        grid_md(lines, "no-skillcorner grid", m["grid"])
    if "r3b_folds" in res:
        f = res["r3b_folds"]
        lines += ["## R3b seed-0 fold models on their held-out SkillCorner matches "
                  "(no new training)", ""]
        sc_table(lines, "nested costs per fold; " + f["nested"]["how"], f["nested"]["scores"],
                 games)
        sc_table(lines, "R3b final costs ({:g}, {:g})".format(*f["final"]["costs"]),
                 f["final"]["scores"], games)
        grid_md(lines, "R3b folds grid", f["grid"])
    return "\n".join(lines) + "\n"


def cmd_score(args):
    with open(os.path.join(args.r3b, "interim", "check.json")) as handle:
        r3b = json.load(handle)
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "grid": GRID,
           "switches": SWITCHES, "seed": SEED, "models": {}}
    for name in args.names:
        if name == SC_DIAG:
            if not (args.sc_fine and args.c23_sc):
                raise SystemExit("no-skillcorner needs --sc-fine and --c23-sc")
            res["models"][name] = score_sc(args, r3b)
            for key in ("model", "r3b_folds"):
                blk = res["models"][name].get(key)
                if blk:
                    sc = blk["choices"]["honest"]["scores"] if key == "model" else \
                        blk["nested"]["scores"]
                    log("{} {}: ".format(name, key) + "; ".join(
                        "{} {:.4f} (c23 {:.4f}, {:+.4f})".format(
                            r, sc[r]["mean_spellF1"], sc[r]["mean_c23"], sc[r]["mean_delta"])
                        for r in SC_REFS))
            continue
        res["models"][name] = score_one(name, args, r3b)
        h = res["models"][name]["choices"]["honest"]
        log("{}: honest costs {} mean delta {:+.4f} (R3b OOF {:+.4f})".format(
            name, h["costs"], h["mean_delta"], res["models"][name]["r3b_oof_mean_delta"]))
    tag = "_sc" if args.names == [SC_DIAG] else ""
    write_json(os.path.join(args.out, "result%s.json" % tag), res)
    text = result_md(res)
    if SC_DIAG in res["models"]:
        text += "\n" + result_sc_md(res["models"][SC_DIAG])
    with open(os.path.join(args.out, "RESULT%s.md" % tag), "w") as handle:
        handle.write(text)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--name", choices=NAMES, required=True)
    t.add_argument("--cache", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--eval-every", type=int, default=200)
    t.add_argument("--threads", type=int, default=2)
    p = sub.add_parser("predict")
    p.add_argument("--name", choices=NAMES, required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=2)
    f = sub.add_parser("predict-folds")
    f.add_argument("--cache", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--r3b", required=True, help="R3b run folder (reads folds/fold*/net_s0.pt)")
    f.add_argument("--threads", type=int, default=2)
    s = sub.add_parser("score")
    s.add_argument("--names", choices=NAMES, nargs="+", default=["no-metrica", "no-dfl"])
    s.add_argument("--sc-fine", default=None, help="cache_possgroup of skillcorner_fine.py")
    s.add_argument("--c23-sc", default=None, help="c23 CSVs of the 20 SkillCorner matches")
    s.add_argument("--cache", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--c23", required=True, help="folder of <game>.csv c23 outputs (raw view)")
    s.add_argument("--r3b", required=True, help="R3b run folder (interim/p5, interim/check.json)")
    args = parser.parse_args(argv)
    return {"train": cmd_train, "predict": cmd_predict, "predict-folds": cmd_predict_folds,
            "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
