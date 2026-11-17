"""Round 4 screening: left-out-provider networks and the three tests. Not part of the submission.

    # torch venv: one network (provider left out of training AND validation), one seed
    python -m tools.learned_spike.r4_screen train --provider dfl --seed 0 --out ROOT/screen/ABC \
        --cache C_TRAIN --val-cache C [train switches, as kfold.py]
    # torch venv: the three tests T-M, T-D, T-S (and T-D0, DFL with its still ball, recorded)
    python -m tools.learned_spike.r4_screen score --recipe-dir ROOT/screen/ABC --seeds 0 \
        --cache CACHE --dfl-cache C --sc-labels POSSGROUP --c23-dev DIR --c23-dfl DIR \
        --c23-sc DIR --out DIR
    # torch venv: SkillCorner seen, out of fold (after the full training of Section 3)
    python -m tools.learned_spike.r4_screen score-sc-oof --folds KFOLD_OUT --seeds 0 1 2 3 \
        --cache CACHE --sc-labels POSSGROUP --c23-sc DIR --out DIR

LEARNED_STAGE_PROTOCOL_R4.md Section 2 with the coordinator's changes of 13 November 2026.

Networks. The pools are those of r3_diag.pool(): no-metrica (Metrica in neither training nor
validation; validation DFL-J03WMX and SC-1886347), no-dfl (validation METRICA-S1 and SC-1886347
... as folds.fold(3).val), no-skillcorner (the 6 other development matches; validation
DFL-J03WMX and METRICA-S1). Any round 4 switch of train.py applies (add_switch_args). Output
ROOT/<provider>/seed<k>/{net.pt, model.json, DONE}; a finished network (DONE with the same step
count) is skipped, so a launcher can be rerun.

Tests (the provider tested is always the one left out; own tracking; c23 on the same input):

- T-M: METRICA-S1..S3, labels as provided (--cache labels raw), features --cache raw, c23
  --c23-dev.
- T-D: the five DFL development matches, still ball removed (input dfl_c: features from
  --dfl-cache raw, c23 run on the still_ball.py bundles, --c23-dfl), labels as provided.
- T-S: the 20 SkillCorner matches against the possession groups (--sc-labels labels raw),
  features --cache raw, c23 --c23-sc.
- T-D0 (recorded only, never in the score): DFL with its still ball (--cache raw, --c23-dev).

Probabilities: train.predict_p5 per network (float64, mean of the four symmetric copies), the
mean over the seeds, rounded once to 4 decimals (decoders.ensemble_p5): the engine's mean of the
4 N rows, rounded as engine.learned.bin_probabilities. Frames: engine.learned.upsample. Decode:
engine.learned.decode_model at each cost pair, with the kickoff rule (LEARNED_STRUCTURE,
pre_kickoff_mask) unless --structure off.

Per match and pair, for the network and for c23: spellF1 (possession_metrics.spell_scores, the
mean over the five IoU thresholds), possessions written and in the reference, merge share,
split share (merge_share(), the definition of the round 4 audit script m01_merge.py) and the
time-weighted merge share (swallowed reference possessions weighted by their length).

Recipe score (coordinator's rule, replacing the draft's formula): for each pair, the smallest of
the three test gains (each the mean over the test's matches of spellF1 network minus c23); the
pair is eligible only if, on each of the three tests, the network's mean merge share is at most
c23's; the score is the highest value among eligible pairs (None if no pair is eligible).

Round 5 (LEARNED_STAGE_PROTOCOL_R5.md Section 2): `score --r5` decodes at the six pairs of
r4_choose.R5_SCREEN_PAIRS, scores T-M and T-D by default (T-S only if asked: the
SkillCorner-left-out network has seed 0 only before the decision) and writes "score_r5"
(r4_choose.screen5_score). Feature versions: a network reads the version of its input width
(model.Net.feature_version); the feature folders must hold that version (build.py marker
feature_version.json, none = stage50-v2). read_feats() checks the marker and ensemble() the
width, so a stage-v3 network never reads stage50-v2 features or the other way round.

Round 6, the candidate network + per-frame member: with --pf-weights (score, score-sc-oof,
grid) the probabilities of each match are those engine.learned.apply would decode with
LEARNED_PF "on" (tools/learned_spike/pf_gate.py: the networks as numpy models on the test's
guarded tracking, the member, the blend at --pf-weight); --engine-cache is then required.

Every match name passes folds.check() (no sealed DFL match, no competition match, no PFF).
Python 3.9.
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
from tools.learned_spike import folds, r3_diag  # noqa: E402

PROVIDERS = {"metrica": "no-metrica", "dfl": "no-dfl", "skillcorner": "no-skillcorner"}
PAIRS = ((8.0, 4.0), (8.0, 8.0), (16.0, 8.0), (16.0, 16.0), (32.0, 16.0))
TESTS = {"T-M": "metrica", "T-D": "dfl", "T-S": "skillcorner", "T-Sp": "skillcorner",
         "T-D0": "dfl"}
# T-S: SkillCorner against the possession groups (in the score); T-Sp: the same matches against
# the phases (the reference as provided, --cache labels), recorded beside it.
SCORED_TESTS = ("T-M", "T-D", "T-S")
H, A, D, X = 0, 1, 2, 3


def log(text):
    print(text, file=sys.stderr, flush=True)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(obj, handle, indent=1, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def net_dir(root, provider, seed):
    return os.path.join(root, provider, "seed{}".format(int(seed)))


def test_games(test):
    """The matches of a test (folds names, checked)."""
    prov = TESTS[test]
    games = list(folds.SC) if prov == "skillcorner" else [
        g for g in folds.OPEN if folds.provider(g) == prov]
    return folds.check_all(games)


# -- training ---------------------------------------------------------------------------------

def cmd_train(args):
    import torch
    from tools.learned_spike import train
    spec = r3_diag.pool(PROVIDERS[args.provider])
    assert not any(folds.provider(g) == args.provider for g in spec["train"] + spec["val"])
    train.setup(args.threads)
    train.configure_args(args)
    train.check_cache(args.cache)
    out = net_dir(args.out, args.provider, args.seed)
    os.makedirs(out, exist_ok=True)
    done = os.path.join(out, "DONE")
    if os.path.isfile(done):
        with open(done) as handle:
            if json.load(handle).get("steps") == args.steps:
                log("{} seed {}: done, skipped".format(args.provider, args.seed))
                return 0
    data_ = train.load_games(args.cache, spec["train"])
    data_.update(train.load_games(args.cache, spec["val"], raw_only=True))
    state, record = train.train_model(data_, spec["train"], args.steps, args.seed,
                                      val_games=spec["val"], eval_every=args.eval_every, log=log)
    vcache = args.val_cache or args.cache
    record.update(left_out=args.provider, pool=spec["name"], train=spec["train"],
                  val=spec["val"], crop_shares=train.crop_shares(spec["train"]),
                  recipe=train.recipe(args.steps, args.eval_every),
                  environment=train.environment(args.threads),
                  cache_md5={os.path.relpath(p, args.cache): train.md5_of(p)
                             for p in train.cache_files(args.cache, spec["train"])},
                  val_cache_md5={os.path.relpath(p, vcache): train.md5_of(p)
                                 for p in train.cache_files(vcache, spec["val"],
                                                            lambda g: ("raw",))})
    torch.save(state, os.path.join(out, "net.pt"))
    write_json(os.path.join(out, "model.json"), record)
    write_json(done, {"steps": args.steps, "best_step": record["best_step"],
                      "train_seconds": record["train_seconds"],
                      "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    log("{} seed {}: best step {} ({:.1f} s, {:.3f} s/step)".format(
        args.provider, args.seed, record["best_step"], record["train_seconds"],
        record["seconds_per_step"]))
    return 0


# -- measures ---------------------------------------------------------------------------------

def spells(codes, runs):
    import possession_metrics as pm
    return pm.possession_spells(codes, runs)


def spell_f1(refc, pred):
    """spellF1 as the audit scripts (u.py): mean F1 over the five IoU thresholds."""
    import possession_metrics as pm
    runs = pm.valid_runs(refc)
    pred = pm.fill_abstentions(np.asarray(pred, np.int8), runs)
    return float(pm.spell_scores(refc, pred, runs)["mean_f1"])


def merge_share(refc, pred):
    """(merge share, split share, time-weighted merge share).

    The first two are m01_merge.py's merge_share() unchanged: a reference possession is
    swallowed if one predicted possession (either team) covers at least half of it and at least
    half of another reference possession; the split share is the mirror, over the predicted
    possessions. The third weights the swallowed reference possessions by their length.
    """
    import possession_metrics as pm
    runs = pm.valid_runs(refc)
    pred = pm.fill_abstentions(np.asarray(pred, np.int8), runs)
    R = spells(refc, runs)
    P = spells(pred, runs)
    if not R or not P:
        return 0.0, 0.0, 0.0
    ra = np.array([r[0] for r in R])
    rb = np.array([r[1] for r in R])
    pa = np.array([p[0] for p in P])
    pb = np.array([p[1] for p in P])
    ov = np.clip(np.minimum(rb[:, None], pb[None, :]) - np.maximum(ra[:, None], pa[None, :]),
                 0, None)
    half = ov >= 0.5 * (rb - ra)[:, None]
    k = half.sum(0)
    merged = (half & (k[None, :] >= 2)).any(1)
    halfp = ov >= 0.5 * (pb - pa)[None, :]
    kp = halfp.sum(1)
    split = (halfp & (kp[:, None] >= 2)).any(0)
    w = (rb - ra).astype(float)
    return float(merged.mean()), float(split.mean()), float((w * merged).sum() / w.sum())


def d_share(codes, refc):
    import possession_metrics as pm
    runs = pm.valid_runs(refc)
    pred = pm.fill_abstentions(np.asarray(codes, np.int8), runs)
    graded = np.zeros(len(refc), bool)
    for a, b in runs:
        graded[a:b] = True
    return float((pred[graded] == D).mean()) if graded.any() else 0.0


def measures(refc, codes):
    import possession_metrics as pm
    runs = pm.valid_runs(refc)
    pred = pm.fill_abstentions(np.asarray(codes, np.int8), runs)
    m, s, tw = merge_share(refc, codes)
    return {"spellF1": spell_f1(refc, codes), "n": len(spells(pred, runs)), "merge": m,
            "split": s, "merge_time": tw, "d_share": d_share(codes, refc)}


def decode_prior(frame_probs, blocks, cost_ha, cost_ad, forced_dead=None, prior_codes=None,
                 bonus=0.0):
    """The engine's decode with the rule-engine prior: engine.learned.decode_model with
    rule_codes = c23's output on the same input and prior_bonus = Config.LEARNED_PRIOR_BONUS
    (apply_prior(): the bonus is added to the log-probability of c23's class on its H, A and D
    frames, before the kickoff mask and the Viterbi). bonus 0 is the plain decode."""
    from engine import learned
    if not bonus:
        return learned.decode_model(frame_probs, blocks, cost_ha, cost_ad, forced_dead)
    return learned.decode_model(frame_probs, blocks, cost_ha, cost_ad, forced_dead,
                                rule_codes=prior_codes, prior_bonus=float(bonus))


def decode_pairs(p5, game, pairs, structure=True, prior_codes=None, bonus=0.0):
    """engine.learned decode of 5 fps probabilities at each pair: [int8 codes]."""
    from engine import data, learned
    n = data.frame_count(game)
    # 5 fps bin probabilities are upsampled; frame probabilities (pf_gate: one row per frame,
    # already as the engine's decoder reads them) pass as they are.
    probs = p5 if len(p5) == n else learned.upsample(p5, n)
    blocks = data.graded_blocks(game)
    forced = learned.pre_kickoff_mask(n, blocks, data.kickoff_frames(game)) if structure \
        else None
    return [decode_prior(probs, blocks, float(a), float(b), forced, prior_codes, bonus)
            for a, b in pairs]


def hidden_d_shares(codes, refc, ball_missing):
    """Share of frames labelled D by `codes` among (a) reference-live frames with the ball
    hidden and (b) reference-dead frames with the ball hidden (graded frames only)."""
    import possession_metrics as pm
    runs = pm.valid_runs(refc)
    pred = pm.fill_abstentions(np.asarray(codes, np.int8), runs)
    graded = np.zeros(len(refc), bool)
    for a, b in runs:
        graded[a:b] = True
    live = graded & ball_missing & (refc < 2)
    dead = graded & ball_missing & (refc == D)
    return {"live_hidden_D": float((pred[live] == D).mean()) if live.any() else None,
            "dead_hidden_D": float((pred[dead] == D).mean()) if dead.any() else None,
            "live_hidden_frames": int(live.sum()), "dead_hidden_frames": int(dead.sum())}


def match_rows(game, refc, p5, c23, pairs, structure=True, bonus=0.0, ball_missing=None):
    """{"n_ref", "c23": measures, "pairs": [{"pair", "net": measures, "delta"}]}."""
    import possession_metrics as pm
    if len(c23) != len(refc):
        raise SystemExit("{}: c23 output and reference differ in length".format(game))
    base = measures(refc, c23)
    if ball_missing is not None:
        base["hidden"] = hidden_d_shares(c23, refc, ball_missing)
    rows = []
    for pair, codes in zip(pairs, decode_pairs(p5, game, pairs, structure, c23, bonus)):
        net = measures(refc, codes)
        if ball_missing is not None:
            net["hidden"] = hidden_d_shares(codes, refc, ball_missing)
        rows.append({"pair": list(pair), "net": net, "delta": net["spellF1"] - base["spellF1"]})
    return {"n_ref": len(spells(refc, pm.valid_runs(refc))), "c23": base, "pairs": rows}


def ball_missing_of(game, engine_cache):
    """bool (N,): ball missing in the raw tracking of a match (own tracking, raw view)."""
    from engine import data
    root = os.path.abspath(os.path.expanduser(os.path.join("~", "mtah2027", "refs")))
    tac = data.load_tactical(folds.check(game), data_dir=root, cache_dir=engine_cache,
                             use_cache=engine_cache is not None)
    return ~(np.isfinite(tac["bx"]) & np.isfinite(tac["by"]))


# -- inputs -----------------------------------------------------------------------------------

def read_labels(folder, game):
    folds.check(game)
    with np.load(os.path.join(folder, "labels", game + ".npz")) as lab:
        return lab["raw"].astype(np.int8)


def folder_version(folder):
    """The feature version of a build.py folder (build.folder_version())."""
    from tools.learned_spike import build
    return build.folder_version(folder)


def read_feats(folder, game):
    """Raw-view features of a match. The width must be that of the folder's version."""
    from tools.learned_spike import features
    folds.check(game)
    version = folder_version(folder)
    with np.load(os.path.join(folder, "feats", "raw", game + ".npz")) as npz:
        feats = npz["F"]
        stored = str(npz["feature_version"]) if "feature_version" in npz.files else None
    if feats.shape[-1] != features.n_features(version) or stored not in (None, version):
        raise SystemExit("{} {}: features of width {} (stored {}) in a {} folder".format(
            folder, game, feats.shape[-1], stored, version))
    return VersionedFeats.of(feats, version)


class VersionedFeats(np.ndarray):
    """A feature matrix that carries its folder's feature version (read_feats()): ensemble()
    compares it with the networks' recorded version (stage50-v2 and stage50-v2s share a
    width). Operations that make a new array (np.ascontiguousarray, torch) drop it."""

    @classmethod
    def of(cls, array, version):
        out = np.asarray(array).view(cls)
        out.feature_version = version
        return out

    def __array_finalize__(self, obj):
        self.feature_version = getattr(obj, "feature_version", None)


def read_c23(folder, game, n):
    import score_game as sg
    return np.asarray(sg.read_codes(os.path.join(folder, game + ".csv"), n), np.int8)


def test_inputs(test, game, args):
    """(reference codes, raw features, c23 codes) of one match in one test."""
    if test == "T-M":
        ref, feats_dir, c23_dir = read_labels(args.cache, game), args.cache, args.c23_dev
    elif test == "T-D":
        ref, feats_dir, c23_dir = read_labels(args.cache, game), args.dfl_cache, args.c23_dfl
    elif test == "T-D0":
        ref, feats_dir, c23_dir = read_labels(args.cache, game), args.cache, args.c23_dev
    elif test == "T-S":
        ref, feats_dir, c23_dir = read_labels(args.sc_labels, game), args.cache, args.c23_sc
    elif test == "T-Sp":
        ref, feats_dir, c23_dir = read_labels(args.cache, game), args.cache, args.c23_sc
    else:
        raise ValueError(test)
    return ref, read_feats(feats_dir, game), read_c23(c23_dir, game, len(ref))


def ensemble(net_paths, feats, cache_dir=None, game=None):
    """Rounded 5 fps probabilities of the mean over networks (and their four copies)."""
    from tools.learned_spike import decoders, train
    probs = []
    for path in net_paths:
        cached = None
        if cache_dir is not None:
            key = "{}__{}".format(game, train.md5_of(path)[:12])
            cached = os.path.join(cache_dir, key + ".npy")
            if os.path.isfile(cached):
                probs.append(np.load(cached))
                continue
        net = train.load_net(path)
        recorded = train.net_version(path)
        have = getattr(feats, "feature_version", None)
        if recorded is not None and have is not None and recorded != have:
            raise SystemExit("{}: the network was trained on {}, the features are {}: give the "
                             "feature folders of its version".format(path, recorded, have))
        if net.n_in != feats.shape[-1]:
            raise SystemExit("{}: the network reads {} ({} columns), the features have {} "
                             "columns: give the feature folders of its version".format(
                                 path, net.feature_version, net.n_in, feats.shape[-1]))
        p = train.predict_p5(net, feats)
        if cached is not None:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cached, p)
        probs.append(p)
    return decoders.ensemble_p5(probs)


def pf_record(args):
    from tools.learned_spike import pf_gate
    return pf_gate.pf_record(args)


def candidate_probs(args, game, kind, paths, provider=None, fold=None):
    """Frame probabilities of the candidate network + per-frame member (pf_gate); the member
    of the same left-out provider or fold (pf_gate.pf_path)."""
    from tools.learned_spike import pf_gate
    if not args.engine_cache:
        raise SystemExit("--pf-weights needs --engine-cache (the tracking of the test inputs)")
    return pf_gate.blended_frame_probs(
        game, kind, paths, pf_gate.pf_path(args.pf_weights, provider, fold), args.pf_weight,
        args.engine_cache, os.path.join(args.out, "pf_cache"))


def nets_of(args, provider):
    """Network files of a left-out provider: --nets override, else <recipe>/<provider>/seed<k>."""
    for item in args.nets or []:
        prov, paths = item.split("=", 1)
        if prov == provider:
            return paths.split(",")
    paths = [os.path.join(net_dir(args.recipe_dir, provider, s), "net.pt") for s in args.seeds]
    for s, p in zip(args.seeds, paths):
        if not os.path.isfile(os.path.join(os.path.dirname(p), "DONE")):
            raise SystemExit("{} seed {}: no finished network ({})".format(provider, s, p))
    return paths


# -- summaries --------------------------------------------------------------------------------

def summarise(test_res, pairs):
    """Per pair: mean gain, matches up, mean merge (net, c23), median count ratios."""
    out = []
    games = list(test_res)
    for i, pair in enumerate(pairs):
        d = [test_res[g]["pairs"][i]["delta"] for g in games]
        net = [test_res[g]["pairs"][i]["net"] for g in games]
        c23 = [test_res[g]["c23"] for g in games]
        nref = [test_res[g]["n_ref"] for g in games]
        out.append({
            "pair": list(pair), "gain": float(np.mean(d)), "up": int(sum(x > 0 for x in d)),
            "n_matches": len(games), "min_delta": float(np.min(d)),
            "spellF1_net": float(np.mean([x["spellF1"] for x in net])),
            "spellF1_c23": float(np.mean([x["spellF1"] for x in c23])),
            "merge_net": float(np.mean([x["merge"] for x in net])),
            "merge_c23": float(np.mean([x["merge"] for x in c23])),
            "split_net": float(np.mean([x["split"] for x in net])),
            "split_c23": float(np.mean([x["split"] for x in c23])),
            "merge_time_net": float(np.mean([x["merge_time"] for x in net])),
            "merge_time_c23": float(np.mean([x["merge_time"] for x in c23])),
            "ratio_ref_net": float(np.median([x["n"] / r for x, r in zip(net, nref)])),
            "ratio_ref_c23": float(np.median([x["n"] / r for x, r in zip(c23, nref)])),
            "ratio_c23_net": float(np.median([x["n"] / y["n"] for x, y in zip(net, c23)])),
        })
        if "hidden" in net[0]:
            for who, rows in (("net", net), ("c23", c23)):
                for k in ("live_hidden_D", "dead_hidden_D"):
                    v = [r["hidden"][k] for r in rows if r["hidden"][k] is not None]
                    out[-1]["{}_{}".format(k, who)] = float(np.mean(v)) if v else None
    return out


def recipe_score(summaries, pairs):
    """The coordinator's screening score (module docstring)."""
    per_pair = []
    for i, pair in enumerate(pairs):
        gains = {t: summaries[t][i]["gain"] for t in SCORED_TESTS}
        eligible = all(summaries[t][i]["merge_net"] <= summaries[t][i]["merge_c23"]
                       for t in SCORED_TESTS)
        per_pair.append({"pair": list(pair), "gains": gains, "min_gain": min(gains.values()),
                         "eligible": bool(eligible)})
    ok = [r for r in per_pair if r["eligible"]]
    best = max(ok, key=lambda r: r["min_gain"]) if ok else None
    return {"per_pair": per_pair, "score": best["min_gain"] if best else None,
            "pair": best["pair"] if best else None,
            "rule": "max over eligible pairs of min over T-M, T-D, T-S of mean(spellF1 net - "
                    "c23); eligible: mean merge share net <= c23 on each of the three tests"}


def table_md(title, res, pairs):
    lines = ["# " + title, ""]
    if res.get("rehearsal"):
        lines += ["**REHEARSAL (short training): these numbers are not results.**", ""]
    lines += ["- networks: {}".format(res["networks"]), "- structure (kickoff rule): {}".format(
        res["structure"]), "- prior bonus on c23's label: {}".format(res.get("prior_bonus", 0.0)),
        ""]
    if "score" in res:
        sc = res["score"]
        lines += ["## Recipe score: {} at {}".format(
            "none (no eligible pair)" if sc["score"] is None else "{:+.4f}".format(sc["score"]),
            sc["pair"]), "", "| pair | T-M gain | T-D gain | T-S gain | min | eligible |",
            "|---|---|---|---|---|---|"]
        for r in sc["per_pair"]:
            g = r["gains"]
            lines.append("| ({:g}, {:g}) | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} | {} |".format(
                r["pair"][0], r["pair"][1], g["T-M"], g["T-D"], g["T-S"], r["min_gain"],
                "yes" if r["eligible"] else "no"))
        lines.append("")
    for test, summ in res["summary"].items():
        lines += ["## {} ({} matches)".format(test, summ[0]["n_matches"]), "",
                  "| pair | gain | up | min | merge net / c23 | split net / c23 | "
                  "time merge net / c23 | count/ref net / c23 |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in summ:
            lines.append("| ({:g}, {:g}) | {:+.4f} | {}/{} | {:+.3f} | {:.3f} / {:.3f} | "
                         "{:.3f} / {:.3f} | {:.3f} / {:.3f} | {:.2f} / {:.2f} |".format(
                             r["pair"][0], r["pair"][1], r["gain"], r["up"], r["n_matches"],
                             r["min_delta"], r["merge_net"], r["merge_c23"], r["split_net"],
                             r["split_c23"], r["merge_time_net"], r["merge_time_c23"],
                             r["ratio_ref_net"], r["ratio_ref_c23"]))
        lines.append("")
        if "live_hidden_D_net" in summ[0]:
            lines += ["Ball hidden (raw input): share labelled D among reference-live / "
                      "reference-dead frames with the ball hidden", "",
                      "| pair | live net / c23 | dead net / c23 |", "|---|---|---|"]
            f = lambda v: "-" if v is None else "{:.3f}".format(v)  # noqa: E731
            for r in summ:
                lines.append("| ({:g}, {:g}) | {} / {} | {} / {} |".format(
                    r["pair"][0], r["pair"][1], f(r["live_hidden_D_net"]),
                    f(r["live_hidden_D_c23"]), f(r["dead_hidden_D_net"]),
                    f(r["dead_hidden_D_c23"])))
            lines.append("")
    return "\n".join(lines) + "\n"


# -- scoring commands -------------------------------------------------------------------------

def cmd_score(args):
    from tools.learned_spike import train
    train.setup(args.threads)
    pairs = [tuple(p) for p in args.pairs]
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "pairs": [list(p) for p in pairs],
           "structure": args.structure == "on", "prior_bonus": args.prior_bonus,
           "tests": {}, "summary": {}, "networks": {},
           "rehearsal": bool(args.rehearsal), **pf_record(args), "inputs": {
               k: getattr(args, k) for k in ("cache", "dfl_cache", "sc_labels", "c23_dev",
                                             "c23_dfl", "c23_sc", "recipe_dir")}}
    pcache = os.path.join(args.out, "p5_cache")
    for test in args.tests:
        prov = TESTS[test]
        paths = nets_of(args, prov)
        res["networks"][prov] = paths
        res["tests"][test] = {}
        for game in test_games(test):
            ref, feats, c23 = test_inputs(test, game, args)
            if args.pf_weights:
                p5 = candidate_probs(args, game, "c" if test == "T-D" else "raw", paths,
                                     provider=prov)
            else:
                p5 = ensemble(paths, feats, pcache, game + ("_D0" if test == "T-D0" else ""))
            bm = ball_missing_of(game, args.engine_cache) if prov == "skillcorner" else None
            res["tests"][test][game] = match_rows(game, ref, p5, c23, pairs,
                                                  args.structure == "on", args.prior_bonus, bm)
            log("{} {}: ".format(test, game) + " ".join(
                "{:+.3f}".format(r["delta"]) for r in res["tests"][test][game]["pairs"]))
        res["summary"][test] = summarise(res["tests"][test], pairs)
    if all(t in res["summary"] for t in SCORED_TESTS):
        res["score"] = recipe_score(res["summary"], pairs)
    if getattr(args, "r5", False) and all(t in res["summary"] for t in ("T-M", "T-D")):
        from tools.learned_spike import r4_choose
        res["score_r5"] = r4_choose.screen5_score(res)
    os.makedirs(args.out, exist_ok=True)
    write_json(os.path.join(args.out, "score.json"), res)
    with open(os.path.join(args.out, "SCORE.md"), "w") as handle:
        handle.write(table_md(args.title or "Round 4 screening: " + str(args.recipe_dir), res,
                              pairs))
    if "score" in res:
        log("recipe score {}".format(res["score"]["score"]))
    if "score_r5" in res:
        from tools.learned_spike import r4_choose
        with open(os.path.join(args.out, "SCORE_R5.md"), "w") as handle:
            handle.write(r4_choose.screen5_md(res["score_r5"], (args.title or str(
                args.recipe_dir)) + ": round 5 screening score"))
        log("round 5 screening score {} at {}".format(res["score_r5"]["score"],
                                                      res["score_r5"]["pair"]))
    return 0


def cmd_score_sc_oof(args):
    """SkillCorner seen, out of fold: SC[j] by the fold models of fold j mod 8 (folds.fold)."""
    from tools.learned_spike import train
    train.setup(args.threads)
    pairs = [tuple(p) for p in args.pairs]
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "pairs": [list(p) for p in pairs],
           "structure": args.structure == "on", "prior_bonus": args.prior_bonus,
           "tests": {"T-S-seen": {}, "T-Sp-seen": {}}, "networks": {},
           "rehearsal": bool(args.rehearsal), **pf_record(args)}
    for j, game in enumerate(folds.SC):
        k = j % folds.N_FOLDS
        assert game in folds.fold(k, pff=False)["test"]
        paths = [os.path.join(args.folds, "fold{}".format(k), "net_s{}.pt".format(s))
                 for s in args.seeds]
        res["networks"][game] = paths
        ref = read_labels(args.sc_labels, game)
        ref_p = read_labels(args.cache, game)
        c23 = read_c23(args.c23_sc, game, len(ref))
        if args.pf_weights:
            p5 = candidate_probs(args, game, "raw", paths, fold=k)
        else:
            p5 = ensemble(paths, read_feats(args.cache, game),
                          os.path.join(args.out, "p5_cache"), game)
        bm = ball_missing_of(game, args.engine_cache)
        res["tests"]["T-S-seen"][game] = match_rows(game, ref, p5, c23, pairs,
                                                    args.structure == "on", args.prior_bonus, bm)
        res["tests"]["T-Sp-seen"][game] = match_rows(game, ref_p, p5, c23, pairs,
                                                     args.structure == "on", args.prior_bonus,
                                                     bm)
        log("{} (fold {}) ".format(game, k) + " ".join(
            "{:+.3f}".format(r["delta"]) for r in res["tests"]["T-S-seen"][game]["pairs"]))
    res["summary"] = {t: summarise(res["tests"][t], pairs) for t in ("T-S-seen", "T-Sp-seen")}
    os.makedirs(args.out, exist_ok=True)
    write_json(os.path.join(args.out, "score_sc_oof.json"), res)
    with open(os.path.join(args.out, "SCORE_SC_OOF.md"), "w") as handle:
        handle.write(table_md("SkillCorner seen, out of fold: " + args.folds, res, pairs))
    return 0


# -- cost-rule inputs (Section 4): the whole grid on the left-out-provider ensembles -----------

FPS = 25.0


# Round 4 cost grid of Section 4 step 1 with deviation 8.1 (LEARNED_STAGE_PROTOCOL_R4.md, 13 Nov
# 2026): the values between the powers of two. Round 3's r3_views.GRID is not changed.
R4_GRID_HA = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0,
              96.0, 128.0)
R4_GRID_AD = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0)
R4_GRID = tuple((a, b) for a in R4_GRID_HA for b in R4_GRID_AD if a <= 2 * b)


def grid_pairs():
    """The pairs of Section 4 (R4_GRID: 146 pairs with H-A <= 2 x alive-dead, deviation 8.1)."""
    return [tuple(p) for p in R4_GRID]


def decode_cached(p5, game, pairs, structure, c23, bonus, cache_dir):
    """decode_pairs() with the decoded codes kept on disk, one npz per match and setting (keyed
    by the probabilities, c23's output, the kickoff rule and the bonus): a second grid run on the
    same ensemble decodes nothing again. Returns [int8 codes] in the order of `pairs`."""
    import hashlib
    key = hashlib.md5()
    key.update(np.ascontiguousarray(p5, dtype=np.float64).tobytes())
    key.update(np.ascontiguousarray(c23, dtype=np.int8).tobytes())
    key.update(json.dumps([bool(structure), float(bonus)]).encode())
    path = os.path.join(cache_dir, "{}__{}.npz".format(game, key.hexdigest()[:16]))
    have = {}
    if os.path.isfile(path):
        with np.load(path) as z:
            have = {tuple(float(x) for x in pr): z["codes"][i] for i, pr in enumerate(z["pairs"])}
    todo = [p for p in pairs if tuple(float(x) for x in p) not in have]
    if todo:
        for p, codes in zip(todo, decode_pairs(p5, game, todo, structure, c23, bonus)):
            have[tuple(float(x) for x in p)] = np.asarray(codes, np.int8)
        os.makedirs(cache_dir, exist_ok=True)
        keys = sorted(have)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, pairs=np.array(keys), codes=np.stack([have[k] for k in keys]))
        os.replace(tmp, path)
    return [have[tuple(float(x) for x in p)] for p in pairs]


def lengths_8e(codes, game):
    """Label-free length statistics of one output, through bar_report.system_stats (the code of
    round 3's bound 8e and of the label-free step): the four 8e statistics pos50, pos90 (median
    and 90th percentile of possession length, s) and d50, d90 (of dead-ball segment length),
    plus pos10, pos25, possessions per 90 minutes and the possession count."""
    from tools.learned_spike import bar_report as br
    st = br.system_stats(np.asarray(codes, np.int8), br.graded_runs(game), None)
    p, d = st["possession_s"], st["d_segment_s"]
    return {"n": st["possessions"], "per90": st["possessions_per_90min"],
            "pos10": p.get("p10"), "pos25": p.get("p25"), "pos50": p.get("p50"),
            "pos90": p.get("p90"), "d50": d.get("p50"), "d90": d.get("p90")}


def grid_refs(provider, game, args):
    """{reference name: codes} of one match for the cost rule."""
    if provider == "metrica":
        refs = {"provided": read_labels(args.cache, game)}
        refs["loose"] = np.load(os.path.join(args.loose_dir, game + "_loose.npy")).astype(np.int8)
        return refs
    if provider == "dfl":
        return {"provided": read_labels(args.cache, game)}
    return {"possgroup": read_labels(args.sc_labels, game), "phases": read_labels(args.cache, game)}


def grid_inputs(provider, game, args):
    """(raw features, c23 codes) on the round 4 input (DFL: still ball removed)."""
    n = len(read_labels(args.cache, game))
    if provider == "metrica":
        return read_feats(args.cache, game), read_c23(args.c23_dev, game, n)
    if provider == "dfl":
        return read_feats(args.dfl_cache, game), read_c23(args.c23_dfl, game, n)
    return read_feats(args.cache, game), read_c23(args.c23_sc, game, n)


def cmd_grid(args):
    """One JSON like round 3's views.json: per match, per pair, network and c23."""
    from tools.learned_spike import train
    train.setup(args.threads)
    pairs = grid_pairs()
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "pairs": [list(p) for p in pairs],
           "structure": args.structure == "on", "prior_bonus": args.prior_bonus,
           "rehearsal": bool(args.rehearsal), **pf_record(args), "networks": {}, "matches": {},
           "measures": "spellF1 (net, c23) per reference; n, merge, split per reference; "
                       "8e lengths (pos50, pos90, d50, d90 in s) of the output"}
    pcache = os.path.join(args.out, "p5_cache")
    for prov in args.providers:
        paths = nets_of(args, prov)
        res["networks"][prov] = paths
        test = {"metrica": "T-M", "dfl": "T-D", "skillcorner": "T-S"}[prov]
        for game in test_games(test):
            feats, c23 = grid_inputs(prov, game, args)
            refs = grid_refs(prov, game, args)
            if args.pf_weights:
                p5 = candidate_probs(args, game, "c" if prov == "dfl" else "raw", paths,
                                     provider=prov)
            else:
                p5 = ensemble(paths, feats, pcache, game)
            row = {"provider": prov, "c23": {"lengths": lengths_8e(c23, game)}, "pairs": []}
            row["ref_lengths"] = {name: lengths_8e(rc, game) for name, rc in refs.items()}
            for name, rc in refs.items():
                row["c23"][name] = measures(rc, c23)
                row["n_ref_" + name] = len(spells(rc, __import__("possession_metrics")
                                                  .valid_runs(rc)))
            t0 = time.time()
            for pair, codes in zip(pairs, decode_cached(p5, game, pairs, args.structure == "on",
                                                        c23, args.prior_bonus,
                                                        os.path.join(args.out, "codes_cache"))):
                cell = {"pair": list(pair), "lengths": lengths_8e(codes, game)}
                for name, rc in refs.items():
                    m = measures(rc, codes)
                    cell[name] = {"spellF1": m["spellF1"], "c23": row["c23"][name]["spellF1"],
                                  "delta": m["spellF1"] - row["c23"][name]["spellF1"],
                                  "n": m["n"], "merge": m["merge"], "split": m["split"],
                                  "merge_time": m["merge_time"]}
                row["pairs"].append(cell)
            res["matches"][game] = row
            row["seconds"] = round(time.time() - t0, 1)
            log("grid {} done ({:.0f} s)".format(game, row["seconds"]))
    os.makedirs(args.out, exist_ok=True)
    write_json(os.path.join(args.out, "grid.json"), res)
    return 0


def add_score_args(p):
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--pairs", type=float, nargs=2, action="append", default=None,
                   metavar=("COST_HA", "COST_AD"))
    p.add_argument("--structure", choices=("on", "off"), default="on",
                   help="kickoff rule of the engine decoder (LEARNED_STRUCTURE), default on")
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--cache", required=True, help="main build.py cache (labels, raw features)")
    p.add_argument("--sc-labels", required=True, help="possession-group label folder")
    p.add_argument("--c23-sc", required=True)
    p.add_argument("--rehearsal", action="store_true", help="mark the outputs as a rehearsal")
    p.add_argument("--prior-bonus", type=float, default=0.0,
                   help="bonus on the log-probability of c23's label (decode_prior); 0 = plain")
    p.add_argument("--engine-cache", default=None,
                   help="engine.data npz cache of the raw tracking (ball-hidden shares)")
    p.add_argument("--title", default=None)
    from tools.learned_spike import pf_gate
    pf_gate.add_pf_args(p)


def main(argv=None):
    from tools.learned_spike import train
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--cache", required=True)
    t.add_argument("--out", required=True, help="the recipe folder")
    t.add_argument("--steps", type=int, default=train.STEPS)
    t.add_argument("--eval-every", type=int, default=train.EVAL_EVERY)
    t.add_argument("--threads", type=int, default=2)
    train.add_switch_args(t)
    s = sub.add_parser("score")
    add_score_args(s)
    s.add_argument("--recipe-dir", default=None)
    s.add_argument("--nets", nargs="+", default=None,
                   help="provider=net1.pt,net2.pt overrides (e.g. round 3 networks)")
    s.add_argument("--tests", nargs="+", choices=sorted(TESTS), default=None,
                   help="default T-M T-D T-S T-Sp (with --r5: T-M T-D)")
    s.add_argument("--r5", action="store_true",
                   help="round 5 screening: the six pairs of r4_choose.R5_SCREEN_PAIRS and "
                        "score_r5 (LEARNED_STAGE_PROTOCOL_R5.md Section 2)")
    s.add_argument("--dfl-cache", required=True, help="input dfl_c folder (r4_inputs cache c)")
    s.add_argument("--c23-dev", required=True)
    s.add_argument("--c23-dfl", required=True, help="c23 on the still_ball.py bundles")
    g = sub.add_parser("grid", help="cost-rule inputs: every pair of the round 4 grid "
                                     "(R4_GRID, deviation 8.1)")
    add_score_args(g)
    g.add_argument("--recipe-dir", default=None)
    g.add_argument("--nets", nargs="+", default=None)
    g.add_argument("--providers", nargs="+", choices=sorted(PROVIDERS),
                   default=["metrica", "dfl", "skillcorner"])
    g.add_argument("--dfl-cache", required=True)
    g.add_argument("--c23-dev", required=True)
    g.add_argument("--c23-dfl", required=True)
    g.add_argument("--loose-dir", required=True, help="Metrica loose-touch references "
                                                     "(r3_views refs: <game>_loose.npy)")
    o = sub.add_parser("score-sc-oof")
    add_score_args(o)
    o.add_argument("--folds", required=True, help="kfold.py output folder (fold<k>/net_s<s>.pt)")
    args = parser.parse_args(argv)
    if args.cmd == "score":
        if args.tests is None:
            args.tests = ["T-M", "T-D"] if args.r5 else list(SCORED_TESTS) + ["T-Sp"]
        if args.r5:
            from tools.learned_spike import r4_choose
            if args.pairs:
                raise SystemExit("--r5 fixes the pairs (r4_choose.R5_SCREEN_PAIRS)")
            args.pairs = [tuple(p) for p in r4_choose.R5_SCREEN_PAIRS]
    if args.cmd not in ("train", "grid"):
        args.pairs = [tuple(p) for p in (args.pairs or PAIRS)]
        for a, b in args.pairs:
            if a > 2 * b:
                raise SystemExit("pair ({}, {}): team-change cost above twice the alive-dead "
                                 "cost".format(a, b))
    return {"train": cmd_train, "score": cmd_score, "score-sc-oof": cmd_score_sc_oof,
            "grid": cmd_grid}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
