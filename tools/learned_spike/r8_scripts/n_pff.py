"""Round 8 arm N, test (4): network sets + member F5T (final) on the 43 open PFF matches, from cached engine inputs. Not part of the submission.

    # torch venv, after source n_env.sh; PFF is a test only (folds.PFF_TRAIN; sealed refused)
    MTAH_DATA_DIR=~/mtah2027/refs python -m tools.learned_spike.r8_scripts.n_pff inputs \
        --engine-cache $NOUT/pff_engine_data --out $NOUT/pff_inputs [--shard K N]
    python -m tools.learned_spike.r8_scripts.n_pff score --inputs $NOUT/pff_inputs \
        --root E2=$NOUT/sets/E2 --seeds 0 1 2 3 --out $NOUT/pff_score \
        [--check-csv-dir R7_GATES/E2/tp/m_31e88976ea8d_12_12_pf0b5e81aaw0.25/runs]

Round 7 scored E2 on PFF through engine.run (r4_pff_test). Running the engine once per match
and per network set is too slow for this round's machine load, so `inputs` runs the engine's
own steps once per match and keeps what does not depend on the networks: the tracking the
engine reads (possession.engine_tactical, default Config), its 5 fps features at stage50-v2t
(learned.features_for) and the member's frame probabilities (learned.pf_frame_probabilities,
member pf_F5T_ffinal.npz = engine/models/learned_pf_v7.npz). `score` then computes each
network's symmetric-copy mean with train.predict_p5 (torch, float64), the seed mean rounded
to 4 decimals, upsample, learned.blend_pf at 0.25, and engine.learned.decode_model at (12, 12)
with the kickoff rule, as learned.apply does; c23 from round 4's c23_pff outputs; references and
scores as r4_pff_test score (2 s controlled reference, and as provided; spell_f1_mean of
score_game). With --check-csv-dir the codes of E2's four final seeds are compared with the
label files round 7's engine runs wrote (the check that this path is the engine's).
Python 3.9.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__)))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

VERSION = "stage50-v2t"
PAIR = (12.0, 12.0)
WEIGHT = 0.25


def log(text):
    print(time.strftime("%H:%M:%S"), text, file=sys.stderr, flush=True)


def games():
    from tools.learned_spike import folds
    return list(folds.assert_not_sealed(folds.PFF_TRAIN))


def cmd_inputs(args):
    from engine import data, learned, learned_features as lf, learned_pf, possession
    from engine.config import Config
    data.DEFAULT_CACHE_DIR = os.path.abspath(os.path.expanduser(args.engine_cache))
    cfg = Config.default()
    member = learned_pf.load_pf(args.member)
    if member.get("swap_mode", "columns") != "columns":
        raise SystemExit("member swap mode is not columns")
    os.makedirs(args.out, exist_ok=True)
    for game in games()[args.shard[0]::args.shard[1]]:
        path = os.path.join(args.out, game + ".npz")
        if os.path.isfile(path):
            continue
        tac = possession.engine_tactical(cfg, game)
        f = lf.downsample(lf.frame_features(tac, game, cfg=cfg, version=VERSION))
        pf = learned.pf_frame_probabilities(cfg, tac, game, member)
        np.savez(path + ".tmp.npz", feats=f, pf=pf, n_frames=int(tac["n_frames"]))
        os.replace(path + ".tmp.npz", path)
        log("{} inputs {}".format(game, f.shape))
    return 0


def cmd_score(args):
    import possession_metrics as pm
    import score_game as sg
    from engine import data, learned
    from tools.learned_spike import decoders, r4_pff_test as tp, r4_screen as rs, train
    train.setup(args.threads)
    systems = dict(x.split("=", 1) for x in args.root)
    path = os.path.join(args.out, "n_pff.json")
    res = json.load(open(path)) if os.path.isfile(path) else {"systems": {}}
    pcache = os.path.join(os.environ.get("NOUT", args.out), "pff_p5_cache")
    os.makedirs(pcache, exist_ok=True)
    refs, c23s = {}, {}
    nets = {}
    for name, root in systems.items():
        paths = [os.path.join(root, "final", "net_s{}.pt".format(s)) for s in args.seeds]
        for p in paths:
            if train.net_version(p) != VERSION:
                raise SystemExit("{}: not a {} network".format(p, VERSION))
        rows = {}
        todo = [g for g in games() if os.path.isfile(os.path.join(args.inputs, g + ".npz"))]
        if len(todo) < len(games()) and not args.partial:
            raise SystemExit("inputs of {} matches missing (--partial: smoke only)".format(
                len(games()) - len(todo)))
        for game in todo:
            z = np.load(os.path.join(args.inputs, game + ".npz"))
            n = int(z["n_frames"])
            per = []
            for p in paths:
                c = os.path.join(pcache, "{}__{}.npy".format(game, train.md5_of(p)[:12]))
                if not os.path.isfile(c):
                    if p not in nets:
                        nets[p] = train.load_net(p)
                    np.save(c, train.predict_p5(nets[p], z["feats"]))
                per.append(np.load(c))
            fp = learned.blend_pf(learned.upsample(decoders.ensemble_p5(per), n), z["pf"], WEIGHT)
            blocks = data.graded_blocks(game)
            forced = learned.pre_kickoff_mask(n, blocks, data.kickoff_frames(game))
            codes = learned.decode_model(fp, blocks, PAIR[0], PAIR[1], forced)
            if game not in refs:
                refs[game] = tp.references(game)
                c23s[game] = np.asarray(sg.read_codes(os.path.join(args.c23, game + ".csv"), n),
                                        np.int8)
            ref, c23 = refs[game], c23s[game]
            row = {}
            for kind in ("ctl2", "provided"):
                f_m = float(sg.score_codes(ref[kind], codes)[0]["spell_f1_mean"])
                f_c = float(sg.score_codes(ref[kind], c23)[0]["spell_f1_mean"])
                row[kind] = {"M": f_m, "c23": f_c, "delta": f_m - f_c,
                             "M_measures": rs.measures(ref[kind], codes),
                             "n_ref": len(rs.spells(ref[kind], pm.valid_runs(ref[kind])))}
            if args.check_csv_dir:
                hits = glob.glob(os.path.join(args.check_csv_dir, game + "_*.csv"))
                if len(hits) == 1:
                    old = np.asarray(sg.read_codes(hits[0], n), np.int8)
                    row["same_as_engine_csv"] = bool(np.array_equal(old, codes))
                    row["frames_differ"] = int((old != codes).sum())
            if len(per) > 1:
                from tools.learned_spike import r5_measures as r5m
                graded = r5m.graded_frames(game, n)
                row["seeds_any"], row["seeds_pairwise"] = r5m.group(per, graded)
            rows[game] = row
            log("{} {} 2 s {:+.4f} provided {:+.4f}{}".format(
                name, game, row["ctl2"]["delta"], row["provided"]["delta"],
                "" if "same_as_engine_csv" not in row else " same as engine csv: {} ({})".format(
                    row["same_as_engine_csv"], row["frames_differ"])))
        d2 = [rows[g]["ctl2"]["delta"] for g in rows]
        dp = [rows[g]["provided"]["delta"] for g in rows]
        summ = {"gain_2s": float(np.mean(d2)), "gain_provided": float(np.mean(dp)),
                "up_2s": int(sum(x > 0 for x in d2)), "n": len(d2), "worst_2s": float(min(d2)),
                "poss_2s": float(np.mean([rows[g]["ctl2"]["M_measures"]["n"] for g in rows])),
                "n_ref_2s": float(np.mean([rows[g]["ctl2"]["n_ref"] for g in rows])),
                "merge_2s": float(np.mean([rows[g]["ctl2"]["M_measures"]["merge"]
                                           for g in rows])),
                "split_2s": float(np.mean([rows[g]["ctl2"]["M_measures"]["split"]
                                           for g in rows]))}
        if all("seeds_pairwise" in rows[g] for g in rows):
            summ["seed_pairwise"] = float(np.mean([rows[g]["seeds_pairwise"] for g in rows]))
            summ["seed_any"] = float(np.mean([rows[g]["seeds_any"] for g in rows]))
        if args.check_csv_dir:
            summ["same_as_engine_csv"] = int(sum(rows[g].get("same_as_engine_csv", False)
                                                 for g in rows))
        res["systems"]["{}_s{}".format(name, "".join(map(str, args.seeds)))] = {
            "root": os.path.abspath(root), "seeds": args.seeds, "summary": summ,
            "per_match": rows}
        log("{}: {}".format(name, summ))
        os.makedirs(args.out, exist_ok=True)
        with open(path + ".tmp", "w") as h:
            json.dump(res, h, indent=1, default=float)
        os.replace(path + ".tmp", path)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("inputs")
    a.add_argument("--engine-cache", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--member", default=os.path.join(ROOT, "engine", "models",
                                                    "learned_pf_v7.npz"))
    a.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("K", "N"))
    s = sub.add_parser("score")
    s.add_argument("--inputs", required=True)
    s.add_argument("--root", action="append", required=True)
    s.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    s.add_argument("--c23", required=True)
    s.add_argument("--check-csv-dir", default=None)
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--partial", action="store_true", help="score the matches with inputs only")
    s.add_argument("--out", required=True)
    args = p.parse_args(argv)
    return {"inputs": cmd_inputs, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
