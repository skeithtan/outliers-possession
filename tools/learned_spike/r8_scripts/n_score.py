"""Round 8 arm N: the network sets of E2 and of the fine-tunes, blended with the member F5T, on the external tests. Not part of the submission.

    # torch venv, after source tools/learned_spike/r8_scripts/n_env.sh
    python -m tools.learned_spike.r8_scripts.n_score --root E2=$NOUT/sets/E2 \
        --root K200=$NOUT/ft_sets/K200 --tests T-Moof T-D T-S T-Sp --seeds 0 1 2 3 --out DIR

A set ROOT holds the networks in this layout (symbolic links are fine; each net_s<s>.pt has its
training record beside it as model_s<s>.json):
    ROOT/fold<k>/net_s<s>.pt          fold networks (folds.fold(k, pff=False))
    ROOT/lo_dfl/net_s<s>.pt           networks trained without DFL
    ROOT/lo_skillcorner/net_s<s>.pt   networks trained without SkillCorner

Tests (LEARNED_STAGE_PROTOCOL_R8.md Section 4, the Arm N rule), each at (12, 12), prior 0, the
kickoff rule, member F5T at weight 0.25 (the same member files as E2's gates, $R7M):
- T-Moof: METRICA-S1..S3 out of fold: METRICA-S<k+1> by the fold-k networks with the fold-k
  member pf_F5T_f<k>.npz (as gate G5, g5_dev_oof.py); labels as provided, c23 $C23_DEV.
- T-D: the five DFL matches, still ball removed (input "c"), networks lo_dfl, member
  pf_F5T_lo_dfl.npz, labels as provided, c23 on the still-ball bundles.
- T-S: the 20 SkillCorner matches against the possession groups, networks lo_skillcorner,
  member pf_F5T_lo_skillcorner.npz; T-Sp: the same probabilities against the phases.
Frame probabilities: the engine's candidate path (engine.learned.candidate_frame_probabilities
with the member's swap mode "columns", as pf_gate / r4_screen score --pf-weights), computed in
parts so that a match's tracking is read once for all systems: the engine's 5 fps features of
the guarded test input (pf_gate.input_tactical, learned.features_for at stage50-v2t) and the
member's frame probabilities are cached per match; each network's symmetric-copy mean is
train.predict_p5 (torch, float64) on those features, cached per network; the ensemble is the
mean over the seeds rounded to 4 decimals (decoders.ensemble_p5, as engine bin_probabilities),
upsampled, then learned.blend_pf at weight 0.25. Decode and measures: r4_screen (decode_pairs,
measures). Also per match:
the share of D inside reference stoppages of at least 2 s, split by ball in view
(r5_measures.dead_recall, reference as provided, raw input; "c" for DFL), dead-ball segments per
90 minutes, and the seed disagreement of the networks alone (r5_measures.group on each seed's
5 fps probabilities from the cached features: share of graded frames where the seeds' top
classes differ, any pair and mean pairwise).
Writes OUT/n_score.json (per match and summaries) and OUT/codes/<system>/<test>__<game>.npy.
Every match passes folds.check(); no competition match, no PFF. Python 3.9.
"""

import argparse
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
from tools.learned_spike import folds, pf_gate, r4_screen as rs  # noqa: E402
from tools.learned_spike import r5_measures as r5m  # noqa: E402

PAIR = (12.0, 12.0)
WEIGHT = 0.25
VERSION = "stage50-v2t"
TESTS = ("T-Moof", "T-D", "T-S", "T-Sp")


def log(text):
    print(time.strftime("%H:%M:%S"), text, file=sys.stderr, flush=True)


def env(name):
    v = os.environ.get(name)
    if not v:
        raise SystemExit("set {} (source n_env.sh)".format(name))
    return v


def test_games(test):
    if test == "T-Moof":
        return [g for g in folds.OPEN if folds.provider(g) == "metrica"]
    return rs.test_games("T-S" if test == "T-Sp" else test)


def fold_of(game):
    for k in range(folds.N_FOLDS):
        if game in folds.fold(k, pff=False)["test"]:
            return k
    raise ValueError(game)


def where(test, game):
    """(set subfolder, member file, input kind, feature folder for the seed check)."""
    m = env("R7M")
    if test == "T-Moof":
        k = fold_of(game)
        return "fold{}".format(k), os.path.join(m, "pf_F5T_f{}.npz".format(k)), "raw", env("CA2T")
    if test == "T-D":
        return "lo_dfl", os.path.join(m, "pf_F5T_lo_dfl.npz"), "c", env("C_VAL2T")
    return ("lo_skillcorner", os.path.join(m, "pf_F5T_lo_skillcorner.npz"), "raw",
            env("CA2T"))


def reference(test, game):
    """(reference codes, c23 codes, provided labels for the stoppage measure)."""
    ca = env("CA2T")
    provided = rs.read_labels(ca, game)
    ref = rs.read_labels(env("PG"), game) if test == "T-S" else provided
    c23_dir = {"T-Moof": env("C23_DEV_DIR"), "T-D": env("C23_DFL_DIR"),
               "T-S": env("C23_SC_DIR"), "T-Sp": env("C23_SC_DIR")}[test]
    return ref, rs.read_c23(c23_dir, game, len(ref)), provided


_BALL = {}


def ball_visible(game, kind):
    key = (game, kind)
    if key not in _BALL:
        _BALL[key] = r5m.ball_visible(game, env("ENGINE_CACHE"), kind)
    return _BALL[key]


def extras(codes, game, provided, kind):
    segs = r5m.dead_segments(provided, game)
    return {"recall": r5m.dead_recall(codes, segs, ball_visible(game, kind)),
            "d90": r5m.d_per_90(codes, game)}


_INPUTS = {}


def game_inputs(game, kind, member, cache_dir):
    """(engine 5 fps features, member frame probabilities) of one test input, cached."""
    key = (game, kind, member)
    if key in _INPUTS:
        return _INPUTS[key]
    from engine import learned, learned_features as lf, learned_pf
    os.makedirs(cache_dir, exist_ok=True)
    fpath = os.path.join(cache_dir, "{}__{}__feats.npy".format(game, kind))
    mpath = os.path.join(cache_dir, "{}__{}__{}.npy".format(
        game, kind, pf_gate.md5_file(member)[:12]))
    tac = None
    if not os.path.isfile(fpath):
        tac = pf_gate.input_tactical(game, kind, env("ENGINE_CACHE"))
        f = lf.downsample(lf.frame_features(tac, game, cfg=None, version=VERSION))
        np.save(fpath, f)
    if not os.path.isfile(mpath):
        model = learned_pf.load_pf(member)
        if model.get("swap_mode", "columns") != "columns":
            raise SystemExit("{}: swap mode {} (this tool assumes columns)".format(
                member, model.get("swap_mode")))
        if tac is None:
            tac = pf_gate.input_tactical(game, kind, env("ENGINE_CACHE"))
        np.save(mpath, learned.pf_frame_probabilities(None, tac, game, model))
    _INPUTS.clear()
    _INPUTS[key] = (np.load(fpath), np.load(mpath))
    return _INPUTS[key]


def net_p5(path, game, kind, feats, cache_dir):
    """train.predict_p5 of one network on the engine features (cached by the file's md5)."""
    from tools.learned_spike import train
    c = os.path.join(cache_dir, "{}__{}__{}.npy".format(game, kind, pf_gate.md5_file(path)[:12]))
    if os.path.isfile(c):
        return np.load(c)
    if train.net_version(path) != VERSION:
        raise SystemExit("{}: not a {} network".format(path, VERSION))
    p = train.predict_p5(train.load_net(path), feats)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(c + ".tmp.npy", p)
    os.replace(c + ".tmp.npy", c)
    return p


def frame_probs(paths, game, kind, member, out):
    """The candidate's frame probabilities (module docstring) and the per-seed 5 fps ones."""
    from engine import data, learned
    from tools.learned_spike import decoders
    shared = os.environ.get("NOUT", out)   # caches shared by every output folder
    feats, pf = game_inputs(game, kind, member, os.path.join(shared, "input_cache"))
    per_seed = [net_p5(p, game, kind, feats, os.path.join(shared, "p5_cache")) for p in paths]
    p_net = learned.upsample(decoders.ensemble_p5(per_seed), data.frame_count(game))
    return learned.blend_pf(p_net, pf, WEIGHT), per_seed


def seed_split(per_seed, game):
    from engine import data
    graded = r5m.graded_frames(game, data.frame_count(game))
    any_diff, pairwise = r5m.group(per_seed, graded)
    return {"any": any_diff, "pairwise": pairwise}


def summarise(rows):
    games = list(rows)
    d = [rows[g]["delta"] for g in games]
    out = {"gain": float(np.mean(d)), "up": int(sum(x > 0 for x in d)), "n": len(games),
           "worst": float(min(d)),
           "spellF1": float(np.mean([rows[g]["net"]["spellF1"] for g in games])),
           "spellF1_c23": float(np.mean([rows[g]["c23"]["spellF1"] for g in games]))}
    for k, name in (("merge", "merge"), ("split", "split"), ("n", "poss"),
                    ("d_share", "d_share")):
        out[name] = float(np.mean([rows[g]["net"][k] for g in games]))
        out[name + "_c23"] = float(np.mean([rows[g]["c23"][k] for g in games]))
    out["n_ref"] = float(np.mean([rows[g]["n_ref"] for g in games]))
    for who in ("net", "c23"):
        rec = r5m.summarise_recall([x for g in games for x in rows[g][who]["recall"]])
        out["D_in_view_" + who] = rec["ball_in_view"]["share_D"]
        out["D_hidden_" + who] = rec["ball_hidden"]["share_D"]
        out["segs_view"] = rec["ball_in_view"]["segments"]
        out["segs_hidden"] = rec["ball_hidden"]["segments"]
        out["d90_" + who] = float(np.mean([rows[g][who]["d90"] for g in games]))
    if all("seeds" in rows[g] for g in games):
        out["seed_any"] = float(np.mean([rows[g]["seeds"]["any"] for g in games]))
        out["seed_pairwise"] = float(np.mean([rows[g]["seeds"]["pairwise"] for g in games]))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", action="append", required=True, help="NAME=SET_ROOT")
    p.add_argument("--tests", nargs="+", choices=TESTS, default=list(TESTS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--games", nargs="+", default=None, help="restrict (smoke test)")
    p.add_argument("--no-seed-split", action="store_true")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    from tools.learned_spike import train
    train.setup(args.threads)
    systems = dict(x.split("=", 1) for x in args.root)
    path = os.path.join(args.out, "n_score.json")
    res = json.load(open(path)) if os.path.isfile(path) else {}
    res.setdefault("systems", {})
    res["pair"], res["member_weight"] = list(PAIR), WEIGHT
    for name, root in systems.items():
        sysres = res["systems"].setdefault(name, {"root": os.path.abspath(root),
                                                  "seeds": args.seeds, "tests": {},
                                                  "summary": {}})
        for test in args.tests:
            rows = {}
            for game in test_games(test):
                if args.games and game not in args.games:
                    continue
                folds.check(game)
                sub, member, kind, _ = where(test, game)
                paths = [os.path.join(root, sub, "net_s{}.pt".format(s)) for s in args.seeds]
                for q in paths:
                    if not os.path.isfile(q):
                        raise SystemExit("missing network {}".format(q))
                fp, per_seed = frame_probs(paths, game, kind, member, args.out)
                ref, c23, provided = reference(test, game)
                row = rs.match_rows(game, ref, fp, c23, [PAIR], True, 0.0)
                codes = rs.decode_pairs(fp, game, [PAIR], True, c23, 0.0)[0]
                cdir = os.path.join(args.out, "codes", name)
                os.makedirs(cdir, exist_ok=True)
                np.save(os.path.join(cdir, "{}__{}.npy".format(test, game)), codes)
                r = {"n_ref": row["n_ref"], "c23": row["c23"], "net": row["pairs"][0]["net"],
                     "delta": row["pairs"][0]["delta"], "networks": paths, "member": member}
                r["net"].update(extras(codes, game, provided, kind))
                r["c23"].update(extras(c23, game, provided, kind))
                if not args.no_seed_split and len(paths) > 1 and test != "T-Sp":
                    r["seeds"] = seed_split(per_seed, game)
                rows[game] = r
                log("{} {} {}: spellF1 {:.4f} c23 {:.4f} delta {:+.4f}".format(
                    name, test, game, r["net"]["spellF1"], r["c23"]["spellF1"], r["delta"]))
            sysres["tests"][test] = rows
            sysres["summary"][test] = summarise(rows)
            log("{} {}: gain {:+.4f} ({}/{} up)".format(name, test,
                                                        sysres["summary"][test]["gain"],
                                                        sysres["summary"][test]["up"],
                                                        sysres["summary"][test]["n"]))
            os.makedirs(args.out, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w") as h:
                json.dump(res, h, indent=1, default=float)
            os.replace(tmp, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
