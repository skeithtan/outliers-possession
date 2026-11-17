"""Round 8 arm DK: E2's frame probabilities on the 43 open PFF matches (inference only). Not part of the submission.

    MTAH_DATA_DIR=~/mtah2027/refs python -m tools.learned_spike.r8_scripts.dk_pff_probs \
        --engine-cache ~/mtah2027/r8/dk/pff_engine_data --out ~/mtah2027/r8/dk/pff_probs \
        [--check-csv-dir R7_GATES/E2/tp/m_31e88976ea8d_12_12_pf0b5e81aaw0.25/runs]

Round 7 scored E2 on PFF through engine.run and kept only the label files, not the
probabilities. This tool runs the same engine path (possession.run in mode "model" with the
default Config: E2's network, member, costs and kickoff rule) on each open PFF match, through
the calls of engine.learned.apply (members, blend_pf, decode_model), and saves per match <out>/<game>.npz with the frame
probabilities `probs` (N, 3), the network's `net` and the member's `pf` frame
probabilities (N, 3) apart, the stock codes `codes` and the ball-in-tracking mask `ball`.
With --check-csv-dir the stock codes are compared with the label file round 7 wrote.
PFF is a test only (folds.PFF_TRAIN, the 43 open matches); the sealed matches are refused.
Nothing is trained. Python 3.9, engine venv.
"""
import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--engine-cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--check-csv-dir", default=None)
    p.add_argument("--games", nargs="*", default=None)
    p.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("K", "N"))
    args = p.parse_args(argv)
    from engine import data, learned, possession
    from engine.config import Config
    from tools.learned_spike import folds
    import score_game as sg
    data.DEFAULT_CACHE_DIR = os.path.abspath(os.path.expanduser(args.engine_cache))
    games = list(args.games or folds.PFF_TRAIN)
    folds.assert_not_sealed(games)
    for g in games:
        if g not in folds.PFF_TRAIN:
            raise SystemExit("{}: not one of the 43 open PFF matches".format(g))
    games = games[args.shard[0]::args.shard[1]]
    cfg = Config.default()
    os.makedirs(args.out, exist_ok=True)
    for game in games:
        path = os.path.join(args.out, game + ".npz")
        if os.path.isfile(path):
            with np.load(path) as z:
                if "net" in z.files:
                    continue
        blocks = data.graded_blocks(game)
        tac = possession.engine_tactical(cfg, game)
        # The two members apart (protocol R8 note 6.1, D4), by the calls of
        # learned.candidate_frame_probabilities; blended and decoded as learned.apply() does.
        from engine import learned_pf
        nets = learned.load_weights(learned.weights_path(cfg))
        _, _, pf_path = learned.pf_settings(cfg)
        pf_model = learned_pf.load_pf(pf_path)
        if pf_model.get("swap_mode", "columns") == "tracking":
            _, p_net = learned.frame_probabilities_paired(nets, tac, game, cfg)
        else:
            _, p_net = learned.frame_probabilities(nets, tac, game, cfg)
        p_pf = learned.pf_frame_probabilities(cfg, tac, game, pf_model)
        probs = learned.blend_pf(p_net, p_pf, cfg.LEARNED_PF_WEIGHT)
        forced = learned.pre_kickoff_mask(len(probs), blocks, data.kickoff_frames(game))
        codes = learned.decode_model(probs, blocks, float(cfg.LEARNED_COST_HA),
                                     float(cfg.LEARNED_COST_AD), forced)
        ball = np.isfinite(tac["bx"]) & np.isfinite(tac["by"])
        same = None
        if args.check_csv_dir:
            hits = glob.glob(os.path.join(args.check_csv_dir, game + "_*.csv"))
            if len(hits) == 1:
                old = np.asarray(sg.read_codes(hits[0], len(codes)), np.int8)
                same = bool(np.array_equal(old, codes))
        np.savez_compressed(path + ".tmp.npz", probs=probs, codes=codes, ball=ball, net=p_net,
                            pf=p_pf)
        os.replace(path + ".tmp.npz", path)
        print(game, "frames", len(codes), "same as round 7 csv:", same, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
