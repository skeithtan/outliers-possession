"""Scoring of the per-frame spike against c23, with r4_screen's measures (round 6). Not part of the submission.

    # pf venv (numpy, pandas)
    python -m tools.learned_spike.pf_eval baseline --out OUT/eval/baseline.json
    python -m tools.learned_spike.pf_eval score --probs OUT/probs/full --out OUT/eval/full.json
    python -m tools.learned_spike.pf_eval stage1 --root OUT

Tests as r4_screen: T-M (Metrica, labels as provided, c23 --c23-dev), T-D (DFL development,
still ball removed: features from the "c" input, c23 on the still_ball bundles, labels as
provided), T-S (SkillCorner against the possession groups), T-Sp (the same against the phases).
Each match is scored only with models that did not train on its provider.

Network (baseline): the AQC left-out-provider networks, seeds 0-3; each seed's probabilities are
the mean of its four symmetric copies as stored by r5_measures (baseline_r4_measures_probs,
key <game>__raw__<md5[:12]>), the seeds averaged and rounded once (decoders.ensemble_p5), then
engine.learned.upsample to 25 fps. Per-frame model: pf_train probabilities (N, 3).
Ensemble w: w x per-frame + (1 - w) x network (frame probabilities, no rounding).
Decode: engine.learned.decode_model with the kickoff rule (pre_kickoff_mask), prior 0.
Measures: r4_screen.measures (spellF1, merge, split...) and summarise (gain = mean over matches
of spellF1 minus c23's). Dead-recall (r5_measures): share of D inside reference stoppages of
at least 2 s (graded, code D), by ball in view on at least half of the stoppage (raw input for
SkillCorner and Metrica, input "c" for DFL); dead-ball segments per 90 graded minutes
(r5_measures.d_per_90) of the output and of the reference.
Python 3.9.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from engine import data, learned  # noqa: E402
from tools.learned_spike import decoders, folds  # noqa: E402
from tools.learned_spike import pf_features as pff  # noqa: E402
from tools.learned_spike import r4_screen as rs  # noqa: E402
from tools.learned_spike import r5_measures as r5m  # noqa: E402

SP = os.environ.get("SP", "/path/to/scratch")  # a local scratch folder of the run
CA = os.environ.get("CA", "/path/to/torch-scratch/train2/cache")  # a local cache of the run
PAIRS = ((4.0, 4.0), (8.0, 8.0), (12.0, 12.0), (16.0, 16.0), (24.0, 24.0), (32.0, 32.0))
WEIGHTS = (0.5, 0.25, 0.75)
TESTS = ("T-M", "T-D", "T-S", "T-Sp")
PROV_TEST = {"metrica": "T-M", "dfl": "T-D", "skillcorner": "T-S"}


class Args:
    cache = CA
    dfl_cache = SP + "/r4_work/cache_c"
    sc_labels = SP + "/sc_fine/cache_possgroup"
    c23_dev = SP + "/r3_out/c23_dev"
    c23_dfl = SP + "/r4_work/c23_dfl_c"
    c23_sc = SP + "/sc_fine/c23_sc"
    engine_cache = SP + "/r4_work/engine_data_raw"
    recipe = SP + "/r4_work/screen/AQC"
    probs5 = SP + "/r5_work/baseline_r4_measures_probs"


ARGS = Args()


def log(text):
    print(time.strftime("%H:%M:%S"), text, file=sys.stderr, flush=True)


def md5_12(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


_MD5 = {}


def net_p5(game):
    """Rounded four-seed 5 fps probabilities of the network that left the game's provider out."""
    prov = folds.provider(pff.check(game))
    ps = []
    for s in range(4):
        path = os.path.join(ARGS.recipe, prov, "seed{}".format(s), "net.pt")
        if path not in _MD5:
            _MD5[path] = md5_12(path)
        p = np.load(os.path.join(ARGS.probs5, "{}__raw__{}.npy".format(game, _MD5[path])))
        ps.append(p.mean(axis=0))
    return decoders.ensemble_p5(ps)


def refs(test, game):
    """(reference codes, c23 codes) of a match in a test (r4_screen's folders)."""
    if test in ("T-M", "T-D", "T-Sp"):
        ref = rs.read_labels(ARGS.cache, game)
    else:
        ref = rs.read_labels(ARGS.sc_labels, game)
    c23_dir = {"T-M": ARGS.c23_dev, "T-D": ARGS.c23_dfl, "T-S": ARGS.c23_sc,
               "T-Sp": ARGS.c23_sc}[test]
    return ref, rs.read_c23(c23_dir, game, len(ref))


def structure(game):
    n = data.frame_count(game)
    blocks = data.graded_blocks(game)
    return n, blocks, learned.pre_kickoff_mask(n, blocks, data.kickoff_frames(game))


def decode(frame_probs, game, pairs):
    n, blocks, forced = structure(game)
    assert frame_probs.shape[0] == n
    return [learned.decode_model(frame_probs, blocks, float(a), float(b), forced)
            for a, b in pairs]


_BALL = {}


def ball_visible(game):
    if game not in _BALL:
        _BALL[game] = r5m.ball_visible(game, ARGS.engine_cache,
                                       "c" if game.startswith("DFL-") else "raw")
    return _BALL[game]


def extra(codes, game, refc):
    segs = r5m.dead_segments(refc, game)
    return {"recall": r5m.dead_recall(codes, segs, ball_visible(game)),
            "d90": r5m.d_per_90(codes, game)}


def games_of(test):
    return rs.test_games("T-S" if test == "T-Sp" else test)


def score_system(probs_of, systems, pairs, tests=TESTS):
    """{system: {test: {game: match row}}}; probs_of(system, game) -> frame probs (N, 3)."""
    res = {s: {t: {} for t in tests} for s in systems}
    for test in tests:
        for game in games_of(test):
            ref, c23 = refs(test, game)
            n_ref = len(rs.spells(ref, __import__("possession_metrics").valid_runs(ref)))
            base = rs.measures(ref, c23)
            base.update(extra(c23, game, rs.read_labels(ARGS.cache, game)))
            for s in systems:
                fp = probs_of(s, game)
                rows = []
                for pair, codes in zip(pairs, decode(fp, game, pairs)):
                    m = rs.measures(ref, codes)
                    m.update(extra(codes, game, rs.read_labels(ARGS.cache, game)))
                    rows.append({"pair": list(pair), "net": m,
                                 "delta": m["spellF1"] - base["spellF1"]})
                res[s][test][game] = {"n_ref": n_ref, "c23": base, "pairs": rows}
            log("{} {} done".format(test, game))
    return res


def summary(test_res, pairs):
    out = rs.summarise(test_res, pairs)
    games = list(test_res)
    for i, row in enumerate(out):
        for who in ("net", "c23"):
            rows = [test_res[g]["pairs"][i]["net"] if who == "net" else test_res[g]["c23"]
                    for g in games]
            pooled = [x for r in rows for x in r["recall"]]
            rec = r5m.summarise_recall(pooled)
            row["recall_view_" + who] = rec["ball_in_view"]["share_D"]
            row["recall_hidden_" + who] = rec["ball_hidden"]["share_D"]
            row["segs_view"] = rec["ball_in_view"]["segments"]
            row["segs_hidden"] = rec["ball_hidden"]["segments"]
            row["d90_" + who] = float(np.mean([r["d90"] for r in rows]))
        row["d90_ref"] = float(np.mean([r5m.d_per_90(rs.read_labels(ARGS.cache, g), g)
                                        for g in games]))
    return out


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as h:
        json.dump(obj, h, indent=1, default=float)
    os.replace(tmp, path)


def cmd_baseline(args):
    pairs = PAIRS

    def probs_of(_s, game):
        return learned.upsample(net_p5(game), data.frame_count(game))
    res = score_system(probs_of, ["net"], pairs)
    summ = {t: summary(res["net"][t], pairs) for t in TESTS}
    write_json(args.out, {"pairs": [list(p) for p in pairs], "matches": res, "summary": summ})
    for t in TESTS:
        r = [x for x in summ[t] if x["pair"] == [12.0, 12.0]][0]
        log("baseline {} (12,12) gain {:+.4f}".format(t, r["gain"]))
    return 0


def pf_probs(folder, game):
    return np.load(os.path.join(folder, pff.check(game) + ".npy"))


def cmd_score(args):
    pairs = PAIRS
    systems = ["pf"] + ["ens{:g}".format(w) for w in WEIGHTS]
    if args.no_ens:
        systems = ["pf"]
    cache = {}

    def probs_of(s, game):
        if game not in cache:
            cache.clear()
            cache[game] = (pf_probs(args.probs, game),
                           learned.upsample(net_p5(game), data.frame_count(game)))
        pf, net = cache[game]
        if s == "pf":
            return pf
        w = float(s[3:])
        return w * pf + (1 - w) * net
    tests = args.tests or TESTS
    res = score_system(probs_of, systems, pairs, tests)
    summ = {s: {t: summary(res[s][t], pairs) for t in tests} for s in systems}
    write_json(args.out, {"pairs": [list(p) for p in pairs], "systems": systems,
                          "probs": args.probs, "matches": res, "summary": summ})
    return 0


def fmt(v):
    return "-" if v is None else "{:+.4f}".format(v)


def stage_table(base, sc, systems, tests=("T-M", "T-D", "T-S", "T-Sp")):
    lines = []
    bsum = base["summary"]
    for test in tests:
        b12 = [x for x in bsum[test] if x["pair"] == [12.0, 12.0]][0]["gain"]
        lines += ["### {} (gain over c23; baseline network alone at (12,12): {:+.4f})".format(
            test, b12), "",
            "| pair | network | " + " | ".join(systems) + " |",
            "|---|---|" + "---|" * len(systems)]
        for i, pr in enumerate(base["pairs"]):
            cells = [fmt(bsum[test][i]["gain"])]
            for s in systems:
                cells.append(fmt(sc["summary"][s][test][i]["gain"]) if test in sc["summary"][s]
                             else "-")
            lines.append("| ({:g},{:g}) | ".format(*pr) + " | ".join(cells) + " |")
        lines.append("")
    return lines


def cmd_stage1(args):
    base = json.load(open(os.path.join(args.root, "eval", "baseline.json")))
    sc = json.load(open(os.path.join(args.root, "eval", "full.json")))
    lines = ["# PF_STAGE1: per-frame model (pf) and ensembles (ens w = w x pf + (1-w) x "
             "network), engine decoder", "",
             "Left-out-provider everywhere. Gain = mean over the test's matches of spellF1 "
             "minus c23's. T-S: SkillCorner vs possession groups; T-Sp: vs phases. "
             "Generated {}.".format(time.strftime("%Y-%m-%d %H:%M")), ""]
    lines += stage_table(base, sc, sc["systems"])
    with open(os.path.join(args.root, "PF_STAGE1"), "w") as h:
        h.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def cmd_merge(args):
    """Merge score JSONs of the same systems (different tests) into one."""
    out = None
    for path in args.inputs:
        r = json.load(open(path))
        if out is None:
            out = r
            continue
        assert r["systems"] == out["systems"] and r["pairs"] == out["pairs"]
        for s in r["systems"]:
            out["matches"][s].update(r["matches"][s])
            out["summary"][s].update(r["summary"][s])
    write_json(args.out, out)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline")
    b.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--probs", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--tests", nargs="+", default=None)
    s.add_argument("--no-ens", action="store_true")
    t = sub.add_parser("stage1")
    t.add_argument("--root", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--inputs", nargs="+", required=True)
    m.add_argument("--out", required=True)
    args = p.parse_args(argv)
    return {"baseline": cmd_baseline, "score": cmd_score, "stage1": cmd_stage1,
            "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
