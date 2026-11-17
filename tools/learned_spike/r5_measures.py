"""Round 5 measures on external matches: disagreement, dead-ball recall, dead-ball counts. Not part of the submission.

    # torch venv. Left-out-provider networks ROOT/<provider>/seed<k>/net.pt (r4_screen.py).
    python -m tools.learned_spike.r5_measures run --recipe-dir ROOT --seeds 0 1 2 3 \
        --cache CACHE --dfl-cache DFL_C --engine-cache DIR --c23-dev DIR --c23-dfl DIR \
        --c23-sc DIR [--defect-dir DEFECT_EVAL] [--providers metrica dfl skillcorner] \
        [--pair 12 12] --out OUT_PREFIX      # writes OUT_PREFIX.json and OUT_PREFIX.md

The provider of a match is always the one left out of the network's training and validation.
Features: Metrica and SkillCorner from CACHE/feats/raw, DFL from DFL_C/feats/raw (input dfl_c,
still ball removed). The defect evaluation view (r5_inputs.py defect-eval: copies k = 100, 101 of
the 8 development matches at round 3's ranges) is read from DEFECT_EVAL/feats/defect.

Probabilities: for each network and each of the four symmetric copies separately, float64 as
train.predict_p5 (a copy's swapped output has H and A put back); stored once per network and
input under OUT_PREFIX_probs/ (r4_screen.py and r3_eval.py keep only the copy MEAN of each
network, so the per-copy probabilities are computed here). A seed's probabilities are the mean
of its four copies (= train.predict_p5).

1. disagree (bar_report.prob_group's definition: arg-max of probabilities upsampled to frames,
   on graded frames; no decoder): per seed, the copy disagreement (share of graded frames where
   the four copies' arg-max is not the same); per pair of seeds, the seed disagreement (share of
   frames where the two seeds' arg-max differs). Also the 8a/8b groups: all seeds together
   (share any differs, mean pairwise) and the four copies averaged over the seeds. Means over a
   provider's matches; on the defect view the two copies of a match are averaged first. For
   Metrica + DFL ("dev", 8 matches, each with its own left-out network) also the ratio defect /
   raw. And spellF1 (pair --pair, prior 0, kickoff rule on) of each seed's decoded output
   against the reference (labels as provided), raw and defect.
2. dead-recall: the decoded output of the seed ensemble (the rounded mean of the seeds, as
   r4_screen.ensemble) and c23's output on the same input. For every reference dead-ball
   segment (code D, inside a graded block) of at least 2 s: the share of its frames labelled D,
   split by whether the ball is in view on at least half of the segment; and the share of
   segments without any D frame. Ball from the input the system ran on (Metrica raw, DFL
   dfl_c, SkillCorner raw); for DFL also from the provider's ball as provided (raw).
3. counts: dead-ball segments per 90 graded minutes (bar_report.system_stats) written by the
   model, by c23 and in the reference, per provider (mean over matches).

4. With the same run: the four-seed ENSEMBLE spellF1 (the seeds' copy means averaged and
   rounded once, decoders.ensemble_p5, as the engine) raw and on the defect view; the medians of
   the seed and copy ratios; Section 2's score (SCORE_PAIRS, merge-eligible on T-M and T-D,
   highest mean of the two gains; r4_screen.measures); per provider the share of D among
   graded reference-live frames (H or A) whose ball is in view and slower than 1 m/s, the speed
   being |ball(t) - ball(t-1)| x 25 on the input the system ran on (frames t and t-1 both with a
   ball; others left out), pooled over frames; and the merge and split shares of r4_screen.
   merge_share (SkillCorner against the possession groups with --sc-labels).
   --sc-seeds gives the SkillCorner-left-out networks their own seed list (9.3: seeds 0 1).
5. rules: Section 2 (R1 against R0) and Section 9.3 (R2 against R1) of
   LEARNED_STAGE_PROTOCOL_R5.md as PASS/FAIL lines, from the JSONs of three runs (R0: round 4's
   networks, v2 caches, defect_eval_v2; R1, R2: stage-v3 caches and defect view).

Networks of any feature version: the input width comes from the state dictionary
(model.net_for_state); features of another width are refused. A provider folder may be a link
(the shared networks of r5_screen_launch.sh).

Every match passes folds.check (no competition match, no sealed DFL match, no PFF). Python 3.9.
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from tools.learned_spike import folds  # noqa: E402

PROVIDERS = ("metrica", "dfl", "skillcorner")
SCORE_PAIRS = ((8.0, 4.0), (8.0, 8.0), (12.0, 12.0), (16.0, 8.0), (16.0, 16.0), (32.0, 16.0))
SLOW_MPS = 1.0               # "slower than 1 m/s": frame-to-frame ball speed (see live_slow)
DEFECT_K = (100, 101)
MIN_DEAD_S = 2.0
FPS = 25
D = 2


def log(text):
    print(text, file=sys.stderr, flush=True)


def provider_games(prov):
    if prov == "skillcorner":
        return folds.check_all(list(folds.SC))
    return folds.check_all([g for g in folds.OPEN if folds.provider(g) == prov])


# -- probabilities ------------------------------------------------------------------------------

def folder_feature_version(folder):
    """The feature version of a feature folder: build.py's marker (train.folder_version), or for
    a defect evaluation folder its defect_eval.json; no record = "stage50-v2". Versions of equal
    width (stage50-v2 and stage50-v2s) cannot be told from the weights, so the inputs decide."""
    from tools.learned_spike import features, train
    path = os.path.join(folder, "defect_eval.json")
    if os.path.isfile(path):
        with open(path) as h:
            v = json.load(h).get("feature_version")
        return features.check_version(v or features.FEATURE_VERSION)
    return train.folder_version(folder)


def copy_probs(net_path, feats, version=None):
    """float64 (4, bins, 3): the network's probabilities on each symmetric copy."""
    import torch
    from tools.learned_spike import features, model
    state = torch.load(net_path, weights_only=True)
    net = model.net_for_state(state).double()      # input width follows the feature version
    net.load_state_dict({k: v.double() for k, v in state.items()})
    net.eval()
    if feats.shape[-1] != net.n_in or (version is not None and
                                       features.n_features(version) != net.n_in):
        raise SystemExit("{}: features have {} columns ({}), the network reads {}; pass the "
                         "caches of its feature version".format(net_path, feats.shape[-1],
                                                                 version, net.n_in))
    out = []
    with torch.no_grad():
        for swap, mirror in model.SYMMETRIES:
            x = torch.from_numpy(np.ascontiguousarray(features.transform(feats, swap, mirror,
                                                                         version=version)))
            p = torch.softmax(net(x[None].double())[0], dim=-1).numpy()
            out.append(p[:, [1, 0, 2]] if swap else p)
    return np.stack(out)


def cached_copy_probs(net_path, feats, cache_dir, key, version=None):
    from tools.learned_spike import train
    trained = train.net_version(net_path)
    if version is not None and trained is not None and trained != version:
        raise SystemExit("{} was trained on feature version {}, the inputs hold {}".format(
            net_path, trained, version))
    path = os.path.join(cache_dir, "{}__{}.npy".format(key, train.md5_of(net_path)[:12]))
    if os.path.isfile(path):
        return np.load(path)
    p = copy_probs(net_path, feats, version)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(path + ".tmp.npy", p)
    os.replace(path + ".tmp.npy", path)
    return p


# -- disagreement (bar_report.prob_group) -------------------------------------------------------

def graded_frames(game, n_frames):
    from tools.learned_spike import bar_report as br
    return br.graded_mask(br.graded_runs(game), n_frames)


def tops(p5, graded):
    from tools.learned_spike import bar_report as br
    return br.upsample(np.asarray(p5, np.float64), graded.size)[graded].argmax(axis=1)


def group(p5_list, graded):
    """prob_group on in-memory members: (share any differs, mean pairwise)."""
    t = np.vstack([tops(p, graded) for p in p5_list])
    any_diff = float((t != t[:1]).any(axis=0).mean())
    pairs = [float(np.mean(t[i] != t[j])) for i, j in itertools.combinations(range(len(t)), 2)]
    return any_diff, (float(np.mean(pairs)) if pairs else None)


def disagree_match(probs, graded):
    """probs: {seed: (4, bins, 3)}. Per-seed copy, per-pair seed, and the 8a/8b groups."""
    seeds = sorted(probs)
    mean = {s: probs[s].mean(axis=0) for s in seeds}
    out = {"copy": {str(s): group(list(probs[s]), graded)[0] for s in seeds},
           "seed_pair": {"{}-{}".format(a, b): group([mean[a], mean[b]], graded)[0]
                         for a, b in itertools.combinations(seeds, 2)}}
    if len(seeds) > 1:
        out["seeds_any"], out["seeds_pairwise"] = group([mean[s] for s in seeds], graded)
    out["copies_over_seeds_any"] = group(
        [np.mean([probs[s][c] for s in seeds], axis=0) for c in range(4)], graded)[0]
    return out


def mean_dicts(rows):
    """Mean of a list of nested dicts of floats (same keys)."""
    if isinstance(rows[0], dict):
        return {k: mean_dicts([r[k] for r in rows]) for k in rows[0]}
    vals = [r for r in rows if r is not None]
    return float(np.mean(vals)) if vals else None


def ratio_dicts(num, den):
    if isinstance(num, dict):
        return {k: ratio_dicts(num[k], den[k]) for k in num}
    return float(num / den) if (num is not None and den) else None


# -- decoding and segments ----------------------------------------------------------------------

def decoders_mean(p5_mean_list):
    """The engine's ensemble: the seeds' mean, rounded once (decoders.ensemble_p5)."""
    from tools.learned_spike import decoders
    return decoders.ensemble_p5(p5_mean_list)


def decode(p5_mean_list, game, pair):
    from tools.learned_spike import decoders, r4_screen
    return r4_screen.decode_pairs(decoders.ensemble_p5(p5_mean_list), game, [pair])[0]


def dead_segments(refc, game, min_s=MIN_DEAD_S):
    from engine import data
    n = len(refc)
    graded = np.zeros(n, bool)
    for _, first, last in data.graded_blocks(game):
        graded[first:last + 1] = True
    m = (np.asarray(refc) == D) & graded
    padded = np.concatenate(([False], m, [False]))
    e = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(e[0::2], e[1::2]) if b - a >= min_s * FPS]


def dead_recall(codes, segs, ball_visible):
    """Per segment (share D, ball in view on >= half)."""
    codes = np.asarray(codes)
    return [(float(np.mean(codes[a:b] == D)), bool(np.mean(ball_visible[a:b]) >= 0.5))
            for a, b in segs]


def summarise_recall(rows):
    out = {}
    for key, sel in (("all", lambda v: True), ("ball_in_view", lambda v: v),
                     ("ball_hidden", lambda v: not v)):
        r = [s for s, v in rows if sel(v)]
        out[key] = {"segments": len(r),
                    "share_D": float(np.mean(r)) if r else None,
                    "share_segments_no_D": float(np.mean([x == 0 for x in r])) if r else None}
    return out


def ball_speed(game, engine_cache, mode):
    """float (N,) m/s: |ball(t) - ball(t-1)| x 25 on the input, NaN unless frames t and t-1
    both have a ball (the frame-to-frame speed of r5_inputs.slow_mask)."""
    from engine import data
    from tools.learned_spike import build, r4_inputs
    tac = build.load_view(folds.check(game), "raw", engine_cache)
    if mode == "c":
        tac, _ = r4_inputs.load_dfl(tac, game, "c")
    pitch = data.pitch_m(game)
    x = np.asarray(tac["bx"], np.float64) * pitch[0]
    y = np.asarray(tac["by"], np.float64) * pitch[1]
    v = np.full(len(x), np.nan)
    v[1:] = np.hypot(np.diff(x), np.diff(y)) * FPS
    return v


def live_slow(codes, refc, game, speed):
    """(frames labelled D, frames): graded reference-live (H or A) frames whose ball is in view
    and slower than SLOW_MPS (ball_speed; frames without a speed are left out)."""
    graded = graded_frames(game, len(refc))
    with np.errstate(invalid="ignore"):
        sel = graded & np.isin(refc, (0, 1)) & np.isfinite(speed) & (speed < SLOW_MPS)
    return int(np.sum(np.asarray(codes)[sel] == D)), int(sel.sum())


def d_per_90(codes, game):
    from tools.learned_spike import bar_report as br
    st = br.system_stats(np.asarray(codes, np.int8), br.graded_runs(game), None)
    minutes = sum(b - a for a, b in br.graded_runs(game)) / float(FPS) / 60.0
    return st["d_segments"] / minutes * 90.0


# -- inputs -------------------------------------------------------------------------------------

def read_npz(folder, sub, game, key):
    folds.check(game)
    with np.load(os.path.join(folder, sub, game + ".npz")) as z:
        return z[key]


def ball_visible(game, engine_cache, mode):
    from tools.learned_spike import build, r4_inputs
    tac = build.load_view(folds.check(game), "raw", engine_cache)
    if mode == "c":
        tac, _ = r4_inputs.load_dfl(tac, game, "c")
    return np.isfinite(tac["bx"]) & np.isfinite(tac["by"])


# -- run ----------------------------------------------------------------------------------------

def run(args):
    import score_game as sg
    from tools.learned_spike import r4_screen, train
    train.setup(args.threads)
    pair = tuple(float(x) for x in args.pair)
    pcache = args.out + "_probs"
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "recipe_dir": args.recipe_dir,
           "seeds": args.seeds, "pair": list(pair), "prior": 0.0,
           "defect_dir": args.defect_dir, "providers": {}, "matches": {}}
    for prov in args.providers:
        seeds = (args.sc_seeds or args.seeds) if prov == "skillcorner" else args.seeds
        nets = {s: os.path.join(args.recipe_dir, prov, "seed{}".format(s), "net.pt")
                for s in seeds}
        for s, p in nets.items():
            if not os.path.isfile(os.path.join(os.path.dirname(p), "DONE")):
                raise SystemExit("{} seed {}: no finished network ({})".format(prov, s, p))
        res["providers"][prov] = {"networks": nets, "seeds": seeds}
        c23_dir = {"metrica": args.c23_dev, "dfl": args.c23_dfl, "skillcorner": args.c23_sc}[prov]
        for game in provider_games(prov):
            t0 = time.time()
            refc = read_npz(args.cache, "labels", game, "raw").astype(np.int8)
            n = len(refc)
            graded = graded_frames(game, n)
            fdir = args.dfl_cache if prov == "dfl" else args.cache
            feats = read_npz(fdir, "feats/raw", game, "F")
            fv = folder_feature_version(fdir)
            probs = {s: cached_copy_probs(nets[s], feats, pcache, game + "__raw", fv)
                     for s in seeds}
            row = {"provider": prov, "raw": disagree_match(probs, graded)}
            means = {s: probs[s].mean(axis=0) for s in seeds}
            row["raw"]["spellF1"] = {str(s): r4_screen.spell_f1(refc, decode([means[s]], game,
                                                                          pair))
                                     for s in seeds}
            codes = decode([means[s] for s in seeds], game, pair)
            row["raw"]["spellF1_ensemble"] = r4_screen.spell_f1(refc, codes)
            if args.defect_dir and prov != "skillcorner":
                per_k = []
                for k in DEFECT_K:
                    f = np.load(os.path.join(args.defect_dir, "feats", "defect", game,
                                             "k{}.npy".format(k)))
                    pk = {s: cached_copy_probs(nets[s], f, pcache, "{}__defect_k{}".format(
                        game, k), folder_feature_version(args.defect_dir)) for s in seeds}
                    d = disagree_match(pk, graded)
                    d["spellF1"] = {str(s): r4_screen.spell_f1(
                        refc, decode([pk[s].mean(axis=0)], game, pair)) for s in seeds}
                    d["spellF1_ensemble"] = r4_screen.spell_f1(
                        refc, decode([pk[s].mean(axis=0) for s in seeds], game, pair))
                    per_k.append(d)
                row["defect"] = mean_dicts(per_k)
            # dead-ball recall and counts: the seed ensemble and c23 on the same input
            c23 = np.asarray(sg.read_codes(os.path.join(c23_dir, game + ".csv"), n), np.int8)
            segs = dead_segments(refc, game)
            balls = {"input": ball_visible(game, args.engine_cache,
                                           "c" if prov == "dfl" else "raw")}
            if prov == "dfl":
                balls["provider"] = ball_visible(game, args.engine_cache, "raw")
            row["dead_recall"] = {who: {b: dead_recall(c, segs, v) for b, v in balls.items()}
                                  for who, c in (("model", codes), ("c23", c23))}
            speed = ball_speed(game, args.engine_cache, "c" if prov == "dfl" else "raw")
            row["live_slow"] = {"model": live_slow(codes, refc, game, speed),
                                "c23": live_slow(c23, refc, game, speed)}
            sref = (read_npz(args.sc_labels, "labels", game, "raw").astype(np.int8)
                    if prov == "skillcorner" and args.sc_labels else refc)
            row["split_reference"] = "possession groups" if sref is not refc else "as provided"
            row["merge_split"] = {who: dict(zip(("merge", "split", "merge_time"),
                                                r4_screen.merge_share(sref, c)))
                                  for who, c in (("model", codes), ("c23", c23))}
            if prov in ("metrica", "dfl"):
                ens = decoders_mean([means[s] for s in seeds])
                base = r4_screen.measures(refc, c23)
                row["score_pairs"] = []
                for pr, cp in zip(SCORE_PAIRS, r4_screen.decode_pairs(ens, game, SCORE_PAIRS)):
                    m = r4_screen.measures(refc, cp)
                    row["score_pairs"].append({"pair": list(pr), "delta": m["spellF1"] -
                                               base["spellF1"], "merge_net": m["merge"],
                                               "merge_c23": base["merge"]})
            row["counts_per_90"] = {"model": d_per_90(codes, game), "c23": d_per_90(c23, game),
                                    "reference": d_per_90(refc, game)}
            res["matches"][game] = row
            log("{} {:.0f}s copy {} seeds {}".format(
                game, time.time() - t0, row["raw"]["copy"], row["raw"]["seed_pair"]))
    summarise(res)
    write(res, args.out)
    return res


def summarise(res):
    s = {}
    groups = {p: [g for g, r in res["matches"].items() if r["provider"] == p]
              for p in res["providers"]}
    dev = [g for g in res["matches"] if res["matches"][g]["provider"] in ("metrica", "dfl")]
    if dev and all(p in groups for p in ("metrica", "dfl")):
        groups["dev"] = dev
    for name, games in groups.items():
        if not games:
            continue
        rows = [res["matches"][g] for g in games]
        out = {"matches": len(games), "raw": mean_dicts([r["raw"] for r in rows])}
        if all("defect" in r for r in rows):
            out["defect"] = mean_dicts([r["defect"] for r in rows])
            out["ratio_defect_raw"] = ratio_dicts(out["defect"], out["raw"])
        rec = {}
        for who in ("model", "c23"):
            for b in rows[0]["dead_recall"][who]:
                pooled = [x for r in rows for x in r["dead_recall"][who][b]]
                rec.setdefault(b, {})[who] = summarise_recall(pooled)
        out["dead_recall"] = rec
        out["counts_per_90"] = mean_dicts([r["counts_per_90"] for r in rows])
        if "ratio_defect_raw" in out:
            rr = out["ratio_defect_raw"]
            out["median_seed_ratio"] = (float(np.median(list(rr["seed_pair"].values())))
                                        if rr["seed_pair"] else None)
            out["median_copy_ratio"] = float(np.median(list(rr["copy"].values())))
        if all("live_slow" in r for r in rows):
            out["live_slow_D"] = {}
            for who in ("model", "c23"):
                d = sum(r["live_slow"][who][0] for r in rows)
                f = sum(r["live_slow"][who][1] for r in rows)
                out["live_slow_D"][who] = {"share_D": d / float(f) if f else None, "frames": f}
            out["merge_split"] = mean_dicts([r["merge_split"] for r in rows])
            out["split_reference"] = rows[0]["split_reference"]
        if all("score_pairs" in r for r in rows):
            out["score_pairs"] = [
                {"pair": rows[0]["score_pairs"][i]["pair"],
                 "gain": float(np.mean([r["score_pairs"][i]["delta"] for r in rows])),
                 "merge_net": float(np.mean([r["score_pairs"][i]["merge_net"] for r in rows])),
                 "merge_c23": float(np.mean([r["score_pairs"][i]["merge_c23"] for r in rows]))}
                for i in range(len(SCORE_PAIRS))]
        s[name] = out
    if "metrica" in s and "dfl" in s and "score_pairs" in s["metrica"]:
        s.setdefault("dev", {})["score"] = section2_score(s["metrica"]["score_pairs"],
                                                          s["dfl"]["score_pairs"])
    res["summary"] = s


def section2_score(tm, td):
    """Section 2's score (round 4's 8.4 objective): over SCORE_PAIRS, among the pairs whose mean
    merge share is at most c23's on T-M and on T-D, the highest mean of the two gains."""
    rows = []
    for a, b in zip(tm, td):
        ok = a["merge_net"] <= a["merge_c23"] and b["merge_net"] <= b["merge_c23"]
        rows.append({"pair": a["pair"], "gain_TM": a["gain"], "gain_TD": b["gain"],
                     "mean_gain": 0.5 * (a["gain"] + b["gain"]), "eligible": bool(ok)})
    ok = [r for r in rows if r["eligible"]]
    best = max(ok, key=lambda r: r["mean_gain"]) if ok else None
    return {"per_pair": rows, "score": best["mean_gain"] if best else None,
            "pair": best["pair"] if best else None}


def fmt(v, nd=4):
    return "-" if v is None else "{:.{}f}".format(v, nd)


def report_md(res):
    w = ["# Round 5 measures: {}".format(res["recipe_dir"]), "",
         "Seeds {}; decoder pair {} prior 0 (spellF1, dead-recall and counts only; the "
         "disagreement is on arg-maxes of probabilities, no decoder). External matches only; "
         "each provider scored by the network that left it out. Generated {}.".format(
             res["seeds"], tuple(res["pair"]), res["generated"]), ""]
    for name, s in res["summary"].items():
        w += ["## {} ({} matches)".format(name, s.get("matches", "-")), ""]
        if "raw" not in s:
            continue
        views = [v for v in ("raw", "defect") if v in s]
        hdr = "| measure | " + " | ".join(views) + (" | defect / raw |" if "defect" in s else " |")
        w += [hdr, "|" + "---|" * (len(views) + 1 + ("defect" in s))]
        keys = [("copy", k) for k in s["raw"]["copy"]] + \
            [("seed_pair", k) for k in s["raw"]["seed_pair"]] + \
            [("spellF1", k) for k in s["raw"]["spellF1"]]
        extra = [k for k in ("seeds_any", "seeds_pairwise", "copies_over_seeds_any")
                 if k in s["raw"]]
        for a, b in keys:
            label = {"copy": "copy disagreement, seed {}", "seed_pair":
                     "seed disagreement, seeds {}", "spellF1": "spellF1, seed {}"}[a].format(b)
            cells = [fmt(s[v][a][b]) for v in views]
            if "defect" in s:
                cells.append(fmt(s["ratio_defect_raw"][a][b], 3))
            w.append("| {} | {} |".format(label, " | ".join(cells)))
        for k in extra:
            cells = [fmt(s[v][k]) for v in views]
            if "defect" in s:
                cells.append(fmt(s["ratio_defect_raw"][k], 3))
            w.append("| {} | {} |".format(k.replace("_", " "), " | ".join(cells)))
        w += ["", "Dead-ball segments of at least 2 s: mean share of frames labelled D "
              "(segments; share of segments without any D)", "",
              "| ball from | segments | model | c23 |", "|---|---|---|---|"]
        for b, rec in s["dead_recall"].items():
            for key in ("all", "ball_in_view", "ball_hidden"):
                m, c = rec["model"][key], rec["c23"][key]
                w.append("| {} ({}) | {} | {} ({}) | {} ({}) |".format(
                    b, key.replace("_", " "), m["segments"], fmt(m["share_D"], 3),
                    fmt(m["share_segments_no_D"], 3), fmt(c["share_D"], 3),
                    fmt(c["share_segments_no_D"], 3)))
        if "median_seed_ratio" in s:
            w.append("")
            w.append("Medians of the ratios defect / raw: seed {}, copy {}; four-seed ensemble "
                     "spellF1 raw {}, defect {}".format(
                         fmt(s["median_seed_ratio"], 3), fmt(s["median_copy_ratio"], 3),
                         fmt(s["raw"].get("spellF1_ensemble")),
                         fmt(s["defect"].get("spellF1_ensemble"))))
        elif "raw" in s and "spellF1_ensemble" in s["raw"]:
            w += ["", "Ensemble spellF1 (raw): {}".format(fmt(s["raw"]["spellF1_ensemble"]))]
        if "live_slow_D" in s:
            ls, ms = s["live_slow_D"], s["merge_split"]
            w += ["", "Reference-live frames with the ball in view and slower than 1 m/s "
                  "(frame-to-frame speed): share labelled D model {} / c23 {} ({} frames). "
                  "Split share (reference {}): model {} / c23 {}; merge share model {} / c23 "
                  "{}.".format(fmt(ls["model"]["share_D"]), fmt(ls["c23"]["share_D"]),
                               ls["model"]["frames"], s["split_reference"],
                               fmt(ms["model"]["split"]), fmt(ms["c23"]["split"]),
                               fmt(ms["model"]["merge"]), fmt(ms["c23"]["merge"]))]
        if "score" in s:
            sc = s["score"]
            w += ["", "Section 2 score: {} at {}".format(fmt(sc["score"]), sc["pair"]), "",
                  "| pair | T-M gain | T-D gain | mean | eligible |", "|---|---|---|---|---|"]
            for r in sc["per_pair"]:
                w.append("| {} | {:+.4f} | {:+.4f} | {:+.4f} | {} |".format(
                    tuple(r["pair"]), r["gain_TM"], r["gain_TD"], r["mean_gain"],
                    "yes" if r["eligible"] else "no"))
        if "counts_per_90" not in s:
            continue
        cnt = s["counts_per_90"]
        w += ["", "Dead-ball segments per 90 graded minutes: model {:.1f}, c23 {:.1f}, "
              "reference {:.1f}".format(cnt["model"], cnt["c23"], cnt["reference"]), ""]
    return "\n".join(w) + "\n"


def write(res, prefix):
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    with open(prefix + ".json.tmp", "w") as h:
        json.dump(res, h, indent=1, sort_keys=True)
    os.replace(prefix + ".json.tmp", prefix + ".json")
    with open(prefix + ".md", "w") as h:
        h.write(report_md(res))


def defect_table(res):
    """The screening baseline on the defect view (dev = 3 Metrica + 5 DFL matches): per seed
    pair the seed-disagreement ratio defect / raw, per seed the copy ratio and the spellF1 on the
    defect view; SkillCorner's raw copy and seed disagreement as extras."""
    dev = res["summary"]["dev"]
    out = {"recipe_dir": res["recipe_dir"], "seeds": res["seeds"], "pair": res["pair"],
           "matches": dev["matches"], "seed_pair": {}, "copy": {}, "spellF1_defect": {},
           "spellF1_raw": {}}
    for k in dev["raw"]["seed_pair"]:
        out["seed_pair"][k] = {"raw": dev["raw"]["seed_pair"][k],
                               "defect": dev["defect"]["seed_pair"][k],
                               "ratio": dev["ratio_defect_raw"]["seed_pair"][k]}
    for k in dev["raw"]["copy"]:
        out["copy"][k] = {"raw": dev["raw"]["copy"][k], "defect": dev["defect"]["copy"][k],
                          "ratio": dev["ratio_defect_raw"]["copy"][k]}
        out["spellF1_defect"][k] = dev["defect"]["spellF1"][k]
        out["spellF1_raw"][k] = dev["raw"]["spellF1"][k]
    out["spellF1_ensemble"] = {"raw": dev["raw"].get("spellF1_ensemble"),
                               "defect": dev["defect"].get("spellF1_ensemble")}
    out["median_seed_ratio"] = dev.get("median_seed_ratio")
    out["median_copy_ratio"] = dev.get("median_copy_ratio")
    out["score"] = dev.get("score")
    if "skillcorner" in res["summary"]:
        sc = res["summary"]["skillcorner"]["raw"]
        out["skillcorner_extra"] = {"copy": sc["copy"], "seed_pair": sc["seed_pair"],
                                    "matches": res["summary"]["skillcorner"]["matches"]}
    w = ["# Round 4 baseline on the defect evaluation view", "",
         "Networks: {} (left-out provider; Metrica matches by the Metrica-left-out network, DFL "
         "matches, input dfl_c, by the DFL-left-out network). {} development matches; defect view "
         "= copies k 100 and 101 (round 3 ranges), averaged per match. Disagreement on arg-maxes "
         "of probabilities (bar_report.prob_group), no decoder; spellF1 at {} prior 0.".format(
             res["recipe_dir"], dev["matches"], tuple(res["pair"])), "",
         "| seeds | seed disagreement raw | defect | defect / raw |", "|---|---|---|---|"]
    for k, v in out["seed_pair"].items():
        w.append("| {} | {:.4f} | {:.4f} | {:.3f} |".format(k, v["raw"], v["defect"], v["ratio"]))
    w += ["", "| seed | copy disagreement raw | defect | defect / raw | spellF1 raw | spellF1 "
          "defect |", "|---|---|---|---|---|---|"]
    for k, v in out["copy"].items():
        w.append("| {} | {:.4f} | {:.4f} | {:.3f} | {:.4f} | {:.4f} |".format(
            k, v["raw"], v["defect"], v["ratio"], out["spellF1_raw"][k],
            out["spellF1_defect"][k]))
    w += ["", "Medians of the ratios: seed {}, copy {}. Four-seed ensemble (seeds' probabilities "
          "averaged and rounded as the engine does) spellF1 at {}: raw {}, defect {}.".format(
              fmt(out["median_seed_ratio"], 3), fmt(out["median_copy_ratio"], 3),
              tuple(res["pair"]), fmt(out["spellF1_ensemble"]["raw"]),
              fmt(out["spellF1_ensemble"]["defect"]))]
    if out["score"]:
        w += ["", "Section 2 score (four-seed ensembles, six pairs, merge-eligible): {} at {}."
              .format(fmt(out["score"]["score"]), out["score"]["pair"])]
    if "skillcorner_extra" in out:
        e = out["skillcorner_extra"]
        w += ["", "Recorded extra: SkillCorner ({} matches, SkillCorner-left-out networks, raw): "
              "copy disagreement {}; seed disagreement {}.".format(
                  e["matches"], ", ".join("seed {} {:.4f}".format(k, v)
                                          for k, v in e["copy"].items()),
                  ", ".join("{} {:.4f}".format(k, v) for k, v in e["seed_pair"].items()))]
    return out, "\n".join(w) + "\n"


R2_MARGIN_RATIO = 0.25       # 9.3 rule 1: a quarter of R1's gap to c23
TOL = 0.010


def _line(ok, text):
    return "{} {}".format("PASS" if ok else "FAIL", text)


def rules(r0, r1, r2=None):
    """Section 2 (R1 against R0) and 9.3 (R2 against R1) as PASS/FAIL lines with every number.

    r0, r1, r2: summaries of `run` JSONs (R0 and R1/R2 with --defect-dir; R1/R2 with the
    SkillCorner-left-out two-seed ensembles, --sc-seeds 0 1)."""
    lines = []
    d0, d1 = r0["summary"]["dev"], r1["summary"]["dev"]
    rr0, rr1 = d0["ratio_defect_raw"], d1["ratio_defect_raw"]
    s0 = list(rr0["seed_pair"].values())
    c0 = list(rr0["copy"].values())
    m1s, m1c = d1["median_seed_ratio"], d1["median_copy_ratio"]
    lo_s, hi_s, lo_c, hi_c = min(s0), max(s0), min(c0), max(c0)
    a = (m1s is not None and m1s < lo_s) or m1c < lo_c
    b = (m1s is None or m1s <= hi_s) and m1c <= hi_c
    lines.append("Section 2 (R1 against R0):")
    lines.append(_line(a and b, "rule 1: R1 median seed ratio {} (R0 lowest {:.3f}, highest {:.3f}); "
                       "R1 median copy ratio {:.3f} (R0 lowest {:.3f}, highest {:.3f}); below a "
                       "lowest: {}; neither above a highest: {}".format(
                           fmt(m1s, 3), lo_s, hi_s, m1c, lo_c, hi_c, a, b)))
    sc0, sc1 = d0["score"]["score"], d1["score"]["score"]
    ok2 = sc0 is not None and sc1 is not None and sc1 >= sc0 - TOL
    lines.append(_line(ok2, "rule 2: score R1 {} at {} against R0 {} at {} (R1 >= R0 - {})".format(
        fmt(sc1), d1["score"]["pair"], fmt(sc0), d0["score"]["pair"], TOL)))
    f0, f1 = d0["defect"]["spellF1_ensemble"], d1["defect"]["spellF1_ensemble"]
    ok3 = f1 >= f0 - TOL
    lines.append(_line(ok3, "rule 3: defect-view four-seed spellF1 R1 {:.4f} against R0 {:.4f} "
                       "(R1 >= R0 - {})".format(f1, f0, TOL)))
    r1_pass = a and b and ok2 and ok3
    lines.append("Section 2 overall: {}".format("PASS (R1 trained in full)" if r1_pass
                                                  else "FAIL (the round ends, c23 ships)"))
    out = {"section2": r1_pass}
    if r2 is not None:
        lines.append("")
        lines.append("Section 9.3 (R2 against R1){}:".format(
            "" if r1_pass else " -- R1 did not pass Section 2, so R2 is not considered"))
        k1, k2 = r1["summary"]["skillcorner"], r2["summary"]["skillcorner"]
        v1 = k1["dead_recall"]["input"]["model"]["ball_in_view"]["share_D"]
        v2 = k2["dead_recall"]["input"]["model"]["ball_in_view"]["share_D"]
        c23v = k1["dead_recall"]["input"]["c23"]["ball_in_view"]["share_D"]
        need = v1 + R2_MARGIN_RATIO * (c23v - v1)
        q1 = v2 >= need
        lines.append(_line(q1, "rule 1: D share in stoppages >= 2 s with the ball in view R2 "
                           "{:.4f}, R1 {:.4f}, c23 {:.4f}; needed R1 + 0.25 x (c23 - R1) = {:.4f}"
                           .format(v2, v1, c23v, need)))
        l1 = k1["live_slow_D"]["model"]["share_D"]
        l2 = k2["live_slow_D"]["model"]["share_D"]
        q2 = l2 <= l1 + 0.01
        lines.append(_line(q2, "rule 2: D share among live frames, ball in view, under 1 m/s R2 "
                           "{:.4f}, R1 {:.4f} (R2 <= R1 + 0.01)".format(l2, l1)))
        p1, p2 = k1["merge_split"]["model"]["split"], k2["merge_split"]["model"]["split"]
        q3 = p2 <= p1 + 0.01
        lines.append(_line(q3, "rule 3: split share ({}) R2 {:.4f}, R1 {:.4f} (R2 <= R1 + 0.01)"
                           .format(k1["split_reference"], p2, p1)))
        d2 = r2["summary"]["dev"]
        sc2 = d2["score"]["score"]
        q4 = sc2 is not None and sc1 is not None and sc2 >= sc1 - TOL
        lines.append(_line(q4, "rule 4: score R2 {} at {} against R1 {} (R2 >= R1 - {})".format(
            fmt(sc2), d2["score"]["pair"], fmt(sc1), TOL)))
        m2s, m2c = d2["median_seed_ratio"], d2["median_copy_ratio"]
        q5 = (m2s is not None and m2s <= hi_s) and m2c <= hi_c
        lines.append(_line(q5, "rule 5: R2 median seed ratio {} (<= R0 highest {:.3f}), median "
                           "copy ratio {:.3f} (<= R0 highest {:.3f})".format(
                               fmt(m2s, 3), hi_s, m2c, hi_c)))
        r2_pass = r1_pass and q1 and q2 and q3 and q4 and q5
        lines.append("Section 9.3 overall: {}".format(
            "PASS (R2 replaces R1)" if r2_pass else "FAIL (R1 is trained in full)"
            if r1_pass else "not considered"))
        out["section9_3"] = r2_pass
    return out, lines


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="disagree, dead-recall and counts in one pass")
    r.add_argument("--recipe-dir", required=True)
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    r.add_argument("--sc-seeds", type=int, nargs="+", default=None,
                   help="seeds of the SkillCorner-left-out networks (default --seeds)")
    r.add_argument("--sc-labels", default=None,
                   help="possession-group labels folder: SkillCorner split share against them")
    r.add_argument("--providers", nargs="+", choices=PROVIDERS, default=list(PROVIDERS))
    r.add_argument("--cache", required=True)
    r.add_argument("--dfl-cache", required=True)
    r.add_argument("--engine-cache", required=True)
    r.add_argument("--c23-dev", required=True)
    r.add_argument("--c23-dfl", required=True)
    r.add_argument("--c23-sc", required=True)
    r.add_argument("--defect-dir", default=None)
    r.add_argument("--pair", type=float, nargs=2, default=[12.0, 12.0])
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--out", required=True, help="output prefix (.json, .md, _probs/)")
    t = sub.add_parser("defect-table", help="the defect-view baseline from a run's JSON")
    t.add_argument("--run", required=True, help="the .json of a run with --defect-dir")
    t.add_argument("--out", required=True, help="output prefix (.json, .md)")
    u = sub.add_parser("rules", help="Section 2 and 9.3 as PASS/FAIL lines")
    u.add_argument("--r0", required=True, help="run JSON of R0 (round 4 networks, v2 inputs)")
    u.add_argument("--r1", required=True, help="run JSON of R1 (stage-v3 inputs)")
    u.add_argument("--r2", default=None, help="run JSON of R2")
    args = p.parse_args(argv)
    if args.cmd == "run":
        run(args)
    elif args.cmd == "rules":
        load = lambda path: json.load(open(path)) if path else None  # noqa: E731
        _, lines = rules(load(args.r0), load(args.r1), load(args.r2))
        print("\n".join(lines))
    else:
        with open(args.run) as h:
            out, md = defect_table(json.load(h))
        with open(args.out + ".json", "w") as h:
            json.dump(out, h, indent=1, sort_keys=True)
        with open(args.out + ".md", "w") as h:
            h.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
