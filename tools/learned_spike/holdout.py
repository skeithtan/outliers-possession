"""Score the eight LOMO fold models on held-out external matches. Not part of the submission.

    python -m tools.learned_spike.holdout --lomo DIR --games G ... --c22-runs DIR --out DIR
        [--ref-cache DIR] [--engine-cache DIR] [--draws 1000] [--seed 0]

The fold models of a LOMO run (DIR/<game>/net.pt) never saw the matches given here. The tool
averages their symmetric class probabilities, decodes them with the median of the switch costs
that the folds chose on their own validation matches, and scores the result against the raw
reference and the 2 s controlled reference. Nothing is chosen on these matches.

It scores c22 on the same references. The c22 run of a match is the base run of a sweep of
tools/analysis/external_sweep.py: <c22-runs>/<game>/<key>.csv, with the key from
run_keys["base"] of the sweep JSON given with --sweep-json. The tool prints a table
per match and the mean difference, model minus c22, with a 5 to 95 % bootstrap interval over
matches. It writes OUT/<game>.csv and OUT/summary.json.

The tool refuses the competition match and the sealed DFL matches (build.check). Python 3.9.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import possession_metrics as pm  # noqa: E402
import score_game as sg  # noqa: E402
from engine import data  # noqa: E402
from tools.learned_spike import build, features, model  # noqa: E402
from tools.learned_spike.lomo import write_csv  # noqa: E402


def load_folds(lomo):
    """(nets, cost_ha, cost_ad): the fold models and the median fold costs."""
    nets, costs = [], []
    for g in sorted(os.listdir(lomo)):
        path = os.path.join(lomo, g, "net.pt")
        if not os.path.isfile(path):
            continue
        with open(os.path.join(lomo, g, "fold.json")) as handle:
            rec = json.load(handle)
        costs.append((rec["cost_ha"], rec["cost_ad"]))
        net = model.Net()
        net.load_state_dict(torch.load(path, weights_only=True))
        net.eval()
        nets.append(net)
    cha = float(np.median([c[0] for c in costs]))
    cad = float(np.median([c[1] for c in costs]))
    return nets, cha, cad


def spell(ref, pred):
    return float(sg.score_codes(ref, pred)[0]["spell_f1_mean"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lomo", required=True)
    parser.add_argument("--games", nargs="+", required=True)
    parser.add_argument("--c22-runs", required=True)
    parser.add_argument("--sweep-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ref-cache", default=os.path.join("/tmp", "spike_refs"))
    parser.add_argument("--engine-cache", default=os.path.join("/tmp", "spike_engine"))
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    nets, cha, cad = load_folds(args.lomo)
    with open(args.sweep_json) as handle:
        base_keys = json.load(handle)["run_keys"]["base"]
    rows = []
    for game in sorted(args.games):
        build.check(game)
        tac = build.load_view(game, "raw", args.engine_cache)
        f5 = features.downsample(features.frame_features(tac, game))
        with torch.no_grad():
            p5 = np.mean([model.predict_symmetric(net, f5) for net in nets], axis=0)
        probs = model.upsample(p5, tac["n_frames"])
        pred = model.decode(probs, data.graded_blocks(game), cha, cad)
        write_csv(os.path.join(args.out, game + ".csv"), pred)
        raw, ctl = build.load_codes(game, args.ref_cache)
        c22_path = os.path.join(args.c22_runs, game, base_keys[game] + ".csv")
        c22 = pm.label_codes(pm.read_label_csv(c22_path))
        row = {"game": game, "model_ctl2": spell(ctl, pred), "c22_ctl2": spell(ctl, c22),
               "model_raw": spell(raw, pred), "c22_raw": spell(raw, c22)}
        rows.append(row)
        print("{game}  2 s: model {model_ctl2:.4f}  c22 {c22_ctl2:.4f}   raw: model "
              "{model_raw:.4f}  c22 {c22_raw:.4f}".format(**row), flush=True)
    table = pd.DataFrame(rows)
    rng = np.random.RandomState(args.seed)
    summary = {"games": len(rows), "cost_ha": cha, "cost_ad": cad, "folds": len(nets)}
    for ref in ("ctl2", "raw"):
        d = (table["model_" + ref] - table["c22_" + ref]).to_numpy()
        boots = [d[rng.randint(len(d), size=len(d))].mean() for _ in range(args.draws)]
        summary[ref] = {"model": round(float(table["model_" + ref].mean()), 4),
                        "c22": round(float(table["c22_" + ref].mean()), 4),
                        "delta": round(float(d.mean()), 4),
                        "low": round(float(np.percentile(boots, 5)), 4),
                        "high": round(float(np.percentile(boots, 95)), 4),
                        "wins": int((d > 0).sum())}
    summary["rows"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                       for r in rows]
    with open(os.path.join(args.out, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({k: summary[k] for k in ("ctl2", "raw", "cost_ha", "cost_ad")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
