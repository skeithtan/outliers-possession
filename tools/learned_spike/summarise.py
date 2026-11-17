"""Compare LOMO predictions of the learned spike with c22 on the same references. Not part of the submission.

    python -m tools.learned_spike.summarise --lomo DIR --cache DIR --c22-runs DIR
        --c22-keys SWEEP.json [--pred-name pred.csv] [--c22-name NAME] [--out OUT.json]

For each test match the tool reads DIR/<game>/<pred-name> and the c22 engine output of the raw
view (external_sweep run key from SWEEP.json, "run_keys" -> "base"). It scores both against
the raw reference and the 2 s controlled reference with possession_metrics through
tools/score_game.py. It prints spellF1 per match, the means per provider and overall, and the
paired difference (model - c22) with the 5 to 95 % bootstrap interval of
tools/paired_summary.paired_summary (1000 draws, seed 0). It also prints macroF1, the
possession count and the D share. Python 3.9.
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
import paired_summary as ps  # noqa: E402
import score_game as sg  # noqa: E402

GROUPS = (("Metrica", "METRICA-"), ("DFL", "DFL-"), ("all", ""))


def flags_item(name, gt, pred, label):
    flags, n_pred = sg._spell_flags(gt, pred, sg.pm.valid_runs(gt))
    return (name, flags.astype(np.int8), int(n_pred), label)


def summarise(lomo, cache, c22_runs, keys, pred_name, games=None, c22_name=None):
    games = games or sorted(g for g in os.listdir(lomo)
                            if os.path.isfile(os.path.join(lomo, g, pred_name)))
    out = {"refs": {}}
    for ref in ("ctl2", "raw"):
        side_model, side_c22, rows = {}, {}, []
        for game in games:
            lab = np.load(os.path.join(cache, "labels", game + ".npz"))
            gt = lab[ref].astype(np.int8)
            pred = sg.read_codes(os.path.join(lomo, game, pred_name), len(gt))
            base_path = (os.path.join(lomo, game, c22_name) if c22_name
                         else os.path.join(c22_runs, game, keys[game] + ".csv"))
            base = sg.read_codes(base_path, len(gt))
            rm, _, sm = sg.score_codes(gt, pred)
            rc, _, sc = sg.score_codes(gt, base)
            side_model[game] = flags_item(game, gt, pred, "model")
            side_c22[game] = flags_item(game, gt, base, "c22")
            rows.append({"game": game, "model": rm["spell_f1_mean"], "c22": rc["spell_f1_mean"],
                         "model_macro": rm["macro_f1"], "c22_macro": rc["macro_f1"],
                         "model_poss": sm["possessions"], "c22_poss": sc["possessions"],
                         "model_dshare": sm["d_share"], "c22_dshare": sc["d_share"]})
        groups = {}
        for label, prefix in GROUPS:
            names = [g for g in games if g.startswith(prefix)]
            if not names:
                continue
            summ = ps.paired_summary({g: side_model[g] for g in names},
                                     {g: side_c22[g] for g in names})
            groups[label] = {"n": len(names),
                             "model": float(np.mean([r["model"] for r in rows
                                                     if r["game"] in names])),
                             "c22": float(np.mean([r["c22"] for r in rows
                                                   if r["game"] in names])),
                             "delta": summ["mean_delta"], "low": summ["low"],
                             "high": summ["high"], "share_positive": summ["share_positive"],
                             "per_match": {m["reference"]: [m["delta"], m["low"], m["high"]]
                                           for m in summ["matches"]}}
        out["refs"][ref] = {"rows": rows, "groups": groups}
    return out


def print_summary(out):
    for ref, block in out["refs"].items():
        print("\nreference: {}".format("2 s controlled" if ref == "ctl2" else "raw"))
        print("{:<12} {:>7} {:>7} {:>8}  {:<20} {:>6} {:>6} {:>5} {:>5} {:>6} {:>6}".format(
            "match", "model", "c22", "delta", "5-95 %", "mMac", "cMac", "mPos", "cPos",
            "mD", "cD"))
        per = block["groups"]["all"]["per_match"]
        for r in block["rows"]:
            d, lo, hi = per[r["game"]]
            print("{:<12} {:>7.4f} {:>7.4f} {:>+8.4f}  [{:+.4f}, {:+.4f}] {:>6.3f} {:>6.3f} "
                  "{:>5d} {:>5d} {:>6.3f} {:>6.3f}".format(
                      r["game"], r["model"], r["c22"], d, lo, hi, r["model_macro"],
                      r["c22_macro"], r["model_poss"], r["c22_poss"], r["model_dshare"],
                      r["c22_dshare"]))
        for label, g in block["groups"].items():
            print("{:<12} {:>7.4f} {:>7.4f} {:>+8.4f}  [{:+.4f}, {:+.4f}]  model higher in "
                  "{:.1%} of draws (n={})".format(label + " mean", g["model"], g["c22"],
                                                  g["delta"], g["low"], g["high"],
                                                  g["share_positive"], g["n"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lomo", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--c22-runs")
    parser.add_argument("--c22-keys")
    parser.add_argument("--c22-name", help="read the c22 file DIR/<game>/<name> instead")
    parser.add_argument("--pred-name", default="pred.csv")
    parser.add_argument("--games", nargs="+")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    keys = None
    if args.c22_keys:
        with open(args.c22_keys) as handle:
            keys = json.load(handle)["run_keys"]["base"]
    out = summarise(args.lomo, args.cache, args.c22_runs, keys, args.pred_name, args.games,
                    args.c22_name)
    print_summary(out)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(out, handle, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
