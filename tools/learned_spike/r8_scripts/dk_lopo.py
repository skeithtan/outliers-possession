"""Round 8 arm DK: E2's left-out tests (T-M, T-D, T-S) and open PFF under the study decoders. Not part of the submission.

    # engine venv, numpy only; reads the stored E2 probabilities, writes under --out
    python -m tools.learned_spike.r8_scripts.dk_lopo lopo --out ~/mtah2027/r8/dk/lopo [--systems ...]
    python -m tools.learned_spike.r8_scripts.dk_lopo pff  --out ~/mtah2027/r8/dk/pff \
        --pff-probs ~/mtah2027/r8/dk/pff_probs --c23-pff SP/r4_work/tp/c23_pff

Probabilities (left-out tests): exactly those of round 6's E2 scorer (pf_e2.py on branch
learned-r6-pf): 0.75 x the retrained left-out networks (AQCTJ, seeds 0-3, r4_screen's per-net
5 fps cache, averaged and rounded once by decoders.ensemble_p5, engine.learned.upsample) +
0.25 x the left-out member F5T (r6_pf/lopo_E2/probs_F5T). Tests as pf_eval: T-M Metrica as
provided, T-D DFL with the still ball removed (input "c"), T-S SkillCorner against the
possession groups; c23 outputs of the same inputs. Open PFF: the engine's own frame
probabilities (dk_pff_probs.py), scored as r4_pff_test (2 s controlled reference and as
provided) against c23's outputs of round 4.

Systems: E2 (engine.learned.decode_model at (12, 12) with the kickoff rule), D1 f, D2 (t, s),
D3 c, D4 (geometric pool, note 6.1; rounded to PROB_DECIMALS on PFF as the engine's blend
is, unrounded on the left-out tests as round 6's scorer), K (kickoff rule K(a)). Guards per test: merge / split shares, possessions written and
the reference's, share of D on graded live reference frames with the ball hidden, and on T-S
the share of D inside reference stoppages of >= 2 s with the ball in view / hidden
(r5_measures.dead_recall; c23 0.668 / 0.840, E2 0.573 / 0.881).
Reads no label of the competition match and no sealed match. Python 3.9.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from engine import data, learned  # noqa: E402
from tools.learned_spike import decoders, folds  # noqa: E402
from tools.learned_spike import r4_screen as rs  # noqa: E402
from tools.learned_spike import r5_measures as r5m  # noqa: E402
from tools.learned_spike.r8_scripts import dk_decoders as dd  # noqa: E402

SP = os.environ.get("SP", "/path/to/scratch")  # a local scratch folder of the run
CA = os.environ.get("CA", "/path/to/torch-scratch/train2/cache")  # a local cache of the run
RECIPE = SP + "/r7_work/screen/AQCTJ"
P5_CACHE = SP + "/r7_work/score_AQCTJ/p5_cache"
F5T_PROBS = SP + "/r6_pf/lopo_E2/probs_F5T"
PG = SP + "/sc_fine/cache_possgroup"
C23 = {"T-M": SP + "/r3_out/c23_dev", "T-D": SP + "/r4_work/c23_dfl_c", "T-S": SP + "/sc_fine/c23_sc"}
ENGINE_CACHE = SP + "/r4_work/engine_data_raw"
C23_PFF = SP + "/r4_work/tp/c23_pff"
PAIR = (12.0, 12.0)
W_PF = 0.25
TESTS = ("T-M", "T-D", "T-S")


def log(text):
    print(time.strftime("%H:%M:%S"), text, file=sys.stderr, flush=True)


_MD5 = {}


def e2_probs(game):
    prov = folds.provider(folds.check(game))
    ps = []
    for s in range(4):
        path = os.path.join(RECIPE, prov, "seed{}".format(s), "net.pt")
        if path not in _MD5:
            _MD5[path] = _md5_12(path)
        ps.append(np.load(os.path.join(P5_CACHE, "{}__{}.npy".format(game, _MD5[path]))))
    net = learned.upsample(decoders.ensemble_p5(ps), data.frame_count(game))
    f5t = np.load(os.path.join(F5T_PROBS, game + ".npy"))
    return W_PF * f5t + (1.0 - W_PF) * net, net, f5t


def _md5_12(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


def system_names(which=None):
    names = ["E2"] + ["D1 f={:g}".format(f) for f in dd.GRID_D1] + \
        ["D2 t={:g} s={:g}".format(t, s) for t, s in dd.GRID_D2] + \
        ["D3 c={:g}".format(c) for c in dd.GRID_D3] + ["D4"] + ["K"]
    if which:
        names = [n for n in names if n == "E2" or any(n.startswith(w) for w in which)]
    return names


def decode_all(names, fp, game, ball_in_view, kickoffs=None, pieces=None, pool_decimals=None):
    """{system: codes} for one match; E2 first (D3 and K reuse its path)."""
    n = fp.shape[0]
    blocks = data.graded_blocks(game)
    kick = data.kickoff_frames(game) if kickoffs is None else kickoffs
    forced = learned.pre_kickoff_mask(n, blocks, kick)
    out = {"E2": dd.decode_stock(fp, blocks, PAIR[0], PAIR[1], forced)}
    for name in names:
        if name == "E2":
            continue
        kind, rest = name.split(" ", 1) if " " in name else (name, "")
        par = dict(kv.split("=") for kv in rest.split()) if rest else {}
        if kind == "D1":
            out[name] = dd.decode_d1(fp, blocks, PAIR[0], PAIR[1], forced, ball_in_view,
                                     float(par["f"]))
        elif kind == "D2":
            out[name] = dd.decode_d2(fp, blocks, PAIR[0], PAIR[1], forced, float(par["t"]),
                                     float(par["s"]))
        elif kind == "D3":
            out[name] = dd.decode_d3(fp, blocks, PAIR[0], PAIR[1], forced, float(par["c"]),
                                     pass1=out["E2"])
        elif kind == "D4":
            pooled = dd.geometric_pool(pieces[0], pieces[1], decimals=pool_decimals)
            out[name] = dd.decode_stock(pooled, blocks, PAIR[0], PAIR[1], forced)
        elif kind == "K":
            out[name], _ = dd.decode_k(fp, blocks, PAIR[0], PAIR[1], forced, kick,
                                       stock_codes=out["E2"])
        elif kind == "D0":                                  # neutral settings (self-check)
            out["D1 f=1"] = dd.decode_d1(fp, blocks, PAIR[0], PAIR[1], forced, ball_in_view, 1.0)
            out["D2 t=2 s=0"] = dd.decode_d2(fp, blocks, PAIR[0], PAIR[1], forced, 2.0, 0.0)
        else:
            raise ValueError(name)
    return out


def guards(codes, refc, ball_in_view, game):
    """Share of D on graded reference-live frames with the ball hidden (and frames)."""
    h = rs.hidden_d_shares(codes, refc, ~np.asarray(ball_in_view, bool))
    return {"live_hidden_D": h["live_hidden_D"], "live_hidden_frames": h["live_hidden_frames"]}


def ball_view(game):
    return r5m.ball_visible(game, ENGINE_CACHE, "c" if game.startswith("DFL-") else "raw")


def changed_ranges(a, b):
    diff = np.asarray(a) != np.asarray(b)
    return decoders.runs_of(diff)


def cmd_lopo(args):
    names = system_names(args.systems)
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "systems": names, "tests": {}}
    os.makedirs(args.out, exist_ok=True)
    codes_dir = os.path.join(args.out, "codes")
    os.makedirs(codes_dir, exist_ok=True)
    for test in TESTS:
        res["tests"][test] = {}
        for game in rs.test_games(test):
            refc_ca = rs.read_labels(CA, game)
            ref = refc_ca if test != "T-S" else rs.read_labels(PG, game)
            c23 = rs.read_c23(C23[test], game, len(ref))
            fp, p_net, p_pf = e2_probs(game)
            ball = ball_view(game)
            sysc = decode_all(names, fp, game, ball, pieces=(p_net, p_pf))
            if args.self_check:
                sysc.update(decode_all(["D0"], fp, game, ball))
            np.savez_compressed(os.path.join(codes_dir, game + ".npz"),
                                **{k.replace(" ", "_"): v for k, v in sysc.items()})
            segs = r5m.dead_segments(refc_ca, game)
            row = {"n_ref": len(rs.spells(ref, __import__("possession_metrics").valid_runs(ref)))}
            base = rs.measures(ref, c23)
            base.update(guards(c23, refc_ca, ball, game))
            base["recall"] = r5m.dead_recall(c23, segs, ball)
            row["c23"] = base
            for s, codes in sysc.items():
                m = rs.measures(ref, codes)
                m.update(guards(codes, refc_ca, ball, game))
                m["recall"] = r5m.dead_recall(codes, segs, ball)
                m["delta"] = m["spellF1"] - base["spellF1"]
                m["changed_frames_vs_E2"] = int(np.sum(codes != sysc["E2"]))
                row[s] = m
            res["tests"][test][game] = row
            log("{} {} E2 {:+.4f} {}".format(test, game, row["E2"]["delta"], " ".join(
                "{}:{:+.4f}".format(s, row[s]["delta"] - row["E2"]["delta"])
                for s in sysc if s != "E2")))
        _write(os.path.join(args.out, "lopo.json"), res)
    return 0


def cmd_pff(args):
    import possession_metrics as pm
    import score_game as sg
    from tools.learned_spike import r4_pff_test as tp
    names = system_names(args.systems)
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "systems": names, "games": {}}
    games = tp.games_of("pff")
    if args.games:
        games = tp.check_games([g for g in games if g in args.games])
    games = games[args.shard[0]::args.shard[1]]
    os.makedirs(args.out, exist_ok=True)
    codes_dir = os.path.join(args.out, "codes")
    os.makedirs(codes_dir, exist_ok=True)
    for game in games:
        with np.load(os.path.join(args.pff_probs, game + ".npz")) as z:
            fp, stock, ball = z["probs"], z["codes"], z["ball"]
            pieces = (z["net"], z["pf"])
        refs = tp.references(game)
        n = len(refs["provided"])
        c23 = np.asarray(sg.read_codes(os.path.join(args.c23_pff, game + ".csv"), n), np.int8)
        sysc = decode_all(names, fp, game, ball, pieces=pieces,
                          pool_decimals=learned.PROB_DECIMALS)
        if not np.array_equal(sysc["E2"], stock):
            raise SystemExit("{}: E2 decode differs from the engine's codes".format(game))
        np.savez_compressed(os.path.join(codes_dir, game + ".npz"),
                            **{k.replace(" ", "_"): v for k, v in sysc.items()})
        row = {}
        for kind in ("ctl2", "provided"):
            r0 = refs[kind]
            base = rs.measures(r0, c23)
            base.update(guards(c23, refs["provided"], ball, game))
            r = {"n_ref": len(rs.spells(r0, pm.valid_runs(r0))), "c23": base}
            for s, codes in sysc.items():
                m = rs.measures(r0, codes)
                m.update(guards(codes, refs["provided"], ball, game))
                m["delta"] = m["spellF1"] - base["spellF1"]
                m["changed_frames_vs_E2"] = int(np.sum(codes != sysc["E2"]))
                r[s] = m
            row[kind] = r
        res["games"][game] = row
        log("PFF {} E2 {:+.4f} {}".format(game, row["ctl2"]["E2"]["delta"], " ".join(
            "{}:{:+.4f}".format(s, row["ctl2"][s]["delta"] - row["ctl2"]["E2"]["delta"])
            for s in sysc if s != "E2")))
        _write(os.path.join(args.out, "pff.json"), res)
    return 0


def _write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as h:
        json.dump(obj, h, indent=1, default=float)
    os.replace(tmp, path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("lopo")
    a.add_argument("--out", required=True)
    a.add_argument("--systems", nargs="*", default=None)
    a.add_argument("--self-check", action="store_true")
    b = sub.add_parser("pff")
    b.add_argument("--out", required=True)
    b.add_argument("--pff-probs", required=True)
    b.add_argument("--c23-pff", default=C23_PFF)
    b.add_argument("--systems", nargs="*", default=None)
    b.add_argument("--games", nargs="*", default=None)
    b.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("K", "N"))
    args = p.parse_args(argv)
    return {"lopo": cmd_lopo, "pff": cmd_pff}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
