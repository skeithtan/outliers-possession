"""Round 3 (DRAFT protocol): out-of-fold predictions, nested costs, interim kill check.

    # 1. torch venv: out-of-fold 5 fps probabilities of the 8 development matches
    python -m tools.learned_spike.r3_nested predict --folds DIR --cache DIR --seeds 0 --out DIR
    # 2. any venv with numpy: nested costs (rule B) and the interim kill check (R3 Section 3)
    python -m tools.learned_spike.r3_nested check --pred DIR --cache DIR --c23 DIR --out FILE

Not part of the submission. `predict` loads, for each development match, the models of the fold
that holds it out (`DIR/fold{K}/net_s{seed}.pt`), averages their symmetric probabilities and
rounds once (decoders.ensemble_p5, as kfold.py). `check` applies rule B of
LEARNED_STAGE_PROTOCOL_R3.md: for each development match the Viterbi costs are the grid pair with
the highest mean of the Metrica and DFL mean spellF1 against the references as provided, over the
other seven development matches; the final costs are the same choice over all eight. It then
scores each match against its reference as provided and applies the frozen interim kill check
(KILL_* below). It reads only the 8 development matches: never BAR-ATH, a sealed match,
evaluation/gt or a blind panel. Python 3.9.
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

DEV = ["METRICA-S1", "METRICA-S2", "METRICA-S3", "DFL-J03WMX", "DFL-J03WN1", "DFL-J03WOH",
       "DFL-J03WOY", "DFL-J03WPY"]
FOLD_OF = {g: k for k, g in enumerate(DEV)}          # folds.OPEN order: fold k tests DEV[k]
GRID_HA = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
GRID_AD = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
GRID = [(a, b) for a in GRID_HA for b in GRID_AD]

# Interim kill check, R3 Section 3 (DRAFT values, frozen with the protocol).
KILL_METRICA_EACH = 0.010     # Delta(candidate - c23) >= this on each of the 3 Metrica matches
KILL_METRICA_MEAN = 0.020     # and the Metrica mean Delta >= this
KILL_DFL_MEAN = 0.0           # DFL mean Delta >= this
KILL_DFL_EACH = -0.010        # and Delta >= this on every DFL match
# count ratio: median over Metrica of n_pred / n_ref >= the same median for c23


def provider(game):
    return game.split("-")[0]


def nested_pick(scores, test=None):
    """Index into GRID: best mean of provider means over DEV without `test`; ties -> lower index."""
    games = [g for g in DEV if g != test]
    best, best_v = None, -1.0
    for i in range(len(GRID)):
        v = np.mean([np.mean([scores[g][i] for g in games if provider(g) == p])
                     for p in ("METRICA", "DFL")])
        if v > best_v + 1e-12:
            best, best_v = i, v
    return best, float(best_v)


def kill_check(rows, c23_ratio_metrica):
    """(passed, reasons) of the interim kill check on per-match rows
    {game: {"delta": float, "ratio": float}}."""
    met = [g for g in DEV if provider(g) == "METRICA"]
    dfl = [g for g in DEV if provider(g) == "DFL"]
    reasons = []
    if not all(rows[g]["delta"] >= KILL_METRICA_EACH for g in met):
        reasons.append("Metrica: a match below +{:.3f}".format(KILL_METRICA_EACH))
    if np.mean([rows[g]["delta"] for g in met]) < KILL_METRICA_MEAN:
        reasons.append("Metrica mean below +{:.3f}".format(KILL_METRICA_MEAN))
    if np.mean([rows[g]["delta"] for g in dfl]) < KILL_DFL_MEAN:
        reasons.append("DFL mean below {:.3f}".format(KILL_DFL_MEAN))
    if not all(rows[g]["delta"] >= KILL_DFL_EACH for g in dfl):
        reasons.append("DFL: a match below {:.3f}".format(KILL_DFL_EACH))
    if np.median([rows[g]["ratio"] for g in met]) < c23_ratio_metrica:
        reasons.append("Metrica count ratio below c23's")
    return not reasons, reasons


def cmd_predict(args):
    import torch  # noqa: F401
    from tools.learned_spike import decoders, train
    train.setup(args.threads)
    for game in DEV:
        k = FOLD_OF[game]
        paths = [os.path.join(args.folds, "fold%d" % k, "net_s%d.pt" % s) for s in args.seeds]
        nets = [train.load_net(p) for p in paths]
        # The feature version of the networks (their training records; else their width):
        # --cache must hold it (train.py checks the folder's marker).
        versions = sorted(set(train.net_version(p) or net.feature_version
                              for p, net in zip(paths, nets)))
        if len(versions) != 1:
            raise SystemExit("{}: networks of different feature versions {}".format(game,
                                                                                    versions))
        train.configure_features(versions[0])
        item = train.load_games(args.cache, [game], raw_only=True)[game]
        p5 = decoders.ensemble_p5([train.predict_p5(n, item["feats"]["raw"]) for n in nets])
        os.makedirs(args.out, exist_ok=True)
        np.savez_compressed(os.path.join(args.out, game + "_p5.npz"), p5=p5,
                            seeds=np.asarray(args.seeds), fold=k)
        print(game, k, file=sys.stderr, flush=True)
    return 0


def cmd_check(args):
    import score_game as sg
    from engine import data
    from tools.learned_spike import decoders
    from tools.learned_spike import r3_scope as r
    pm = r.pm
    scores, probs, refs = {}, {}, {}
    for game in DEV:
        lab = np.load(os.path.join(args.cache, "labels", game + ".npz"))
        refs[game] = lab["raw"].astype(np.int8)
        p5 = np.load(os.path.join(args.pred, game + "_p5.npz"))["p5"]
        probs[game] = decoders.upsample(p5, len(refs[game]))
        paths = decoders.decode_model(probs[game], data.graded_blocks(game), GRID)
        scores[game] = [r.spell_f1(refs[game], p) for p in paths]
    out = {"grid": GRID, "matches": {}}
    rows = {}
    c23_ratios = []
    for game in DEV:
        i, v = nested_pick(scores, game)
        runs = pm.valid_runs(refs[game])
        codes = decoders.decode_model(probs[game], data.graded_blocks(game), [GRID[i]])[0]
        c23 = sg.read_codes(os.path.join(args.c23, game + ".csv"), len(refs[game]))
        n_ref = len(r.spells(refs[game], runs))
        n_c = len(r.spells(pm.fill_abstentions(codes, runs), runs))
        n_23 = len(r.spells(pm.fill_abstentions(c23, runs), runs))
        f_c, f_23 = scores[game][i], r.spell_f1(refs[game], c23)
        rows[game] = {"delta": f_c - f_23, "ratio": n_c / float(n_ref)}
        if provider(game) == "METRICA":
            c23_ratios.append(n_23 / float(n_ref))
        out["matches"][game] = {"costs": GRID[i], "nested_val": v, "spellF1": f_c,
                                "c23": f_23, "delta": f_c - f_23, "n": n_c, "n_c23": n_23,
                                "n_ref": n_ref}
    fi, fv = nested_pick(scores, None)
    out["final_costs"] = GRID[fi]
    out["final_val"] = fv
    c23_ratio = float(np.median(c23_ratios))
    passed, reasons = kill_check(rows, c23_ratio)
    out["c23_ratio_metrica"] = c23_ratio
    out["kill_check"] = {"passed": passed, "reasons": reasons}
    with open(args.out, "w") as h:
        json.dump(out, h, indent=1)
    print(json.dumps({"passed": passed, "reasons": reasons,
                      "deltas": {g: round(rows[g]["delta"], 4) for g in DEV}}))
    return 0 if passed else 3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--folds", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=2)
    c = sub.add_parser("check")
    c.add_argument("--pred", required=True)
    c.add_argument("--cache", required=True)
    c.add_argument("--c23", required=True, help="folder of <game>.csv c23 outputs (raw view)")
    c.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return cmd_predict(args) if args.cmd == "predict" else cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
