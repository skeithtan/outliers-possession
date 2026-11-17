"""Criterion 8, part c: the numpy engine against torch on the eight open matches.

    python -m tools.learned_spike.repro_check run --nets NET.pt [NET.pt ...] --weights NPZ
        --costs COSTS.json --engine-cache DIR --work DIR --out JSON
        [--engine-python PY] [--battery-log FILE] [--games G ...]

LEARNED_STAGE_PROTOCOL.md Section 6 asks that the numpy engine and torch give the same decoded
labels with the final ensemble on every frame of the eight open matches, or that every
disagreement is logged and the total is below 0.0001 of the frames. This tool checks that. For
each match it does these steps:

1. It runs the real engine three times, as evaluate.run_one() does for the raw view: c22 (no
   override), M and H with the final weights and the final costs (evaluate.overrides()). The
   runs use --engine-python, the venv of the engine. The c22 md5 is compared with the switch-off
   record (tools/analysis/switch_off_identity.py).
2. It computes the features of the engine (engine.learned_features) on the same tracking.
3. The numpy path: engine.learned.predict_ensemble(), rounded to PROB_DECIMALS, upsampled, and
   decoded with engine.learned.decode_model() (M) and decode_hybrid() (H, on the c22 labels of
   step 1).
4. The torch path: the torch state dictionaries of the final models in float64, one symmetric
   copy at a time, as train.predict_p5() does. The sum runs over the models in file order and
   the copies in SYMMETRIES order, then divides by 4 N and rounds, as the engine does. The same
   decoders decode it.
5. It counts the graded frames where the labels of the two paths differ and records the max
   |p_numpy - p_torch| before and after the rounding. It checks that the labels of step 1 equal
   the labels of the numpy path.

The torch float32 path of the spike (model.predict_symmetric()) is recorded as information only.
The protocol computes in float64.

--tactical guarded (round 2, LEARNED_STAGE_PROTOCOL_R2.md Section 9, D1): step 2 reads the ATD
that the engine reads, possession.engine_tactical(): tracks.speed_from_positions() and the C9
goalkeeper guard on the same load_tactical() result. The record then also gives the guard rows
and the feature md5 of the raw ATD. The default --tactical raw is the round 1 behaviour.

The tool reads no reference and no ground truth, never runs BAR-ATH or a sealed match
(folds.check()), and writes only under --work and --out. torch loads only in the parent
process. The engine-child subcommand runs the engine with numpy only. Python 3.9.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import engine  # noqa: E402,F401  (pins the BLAS threads before numpy loads, as engine.run)
import numpy as np  # noqa: E402

MAX_SHARE = 1e-4            # protocol Section 6: disagreements below 0.0001 of the frames
BATTERY_MIN = 26            # start a heavy step at this charge or more
BATTERY_WAIT_S = 120
MAX_LOGGED = 10000          # frames listed per match and candidate (a pass needs far fewer)
CANDIDATES = ("M", "H")


def log(text):
    print(text, file=sys.stderr, flush=True)


def md5_file(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def codes_md5(codes):
    return hashlib.md5(np.ascontiguousarray(codes, dtype=np.int8).tobytes()).hexdigest()


# -- battery --------------------------------------------------------------------------------------

def battery_ok(line, minimum=BATTERY_MIN):
    """True if a battery guard line shows AC power, no paused job and at least `minimum` %.

    Example line: "11-09 16:24:33 28% charging 'AC Power' paused=0".
    """
    parts = line.split()
    pct = [p for p in parts if p.endswith("%") and p[:-1].isdigit()]
    if not pct or "'AC Power'" not in line or "paused=0" not in parts:
        return False
    return int(pct[0][:-1]) >= minimum


def wait_battery(path, minimum=BATTERY_MIN, wait_s=BATTERY_WAIT_S):
    """Block until the guard file `path` shows battery_ok(). None skips the check."""
    if not path:
        return
    while True:
        try:
            with open(path) as handle:
                line = handle.read().strip()
        except IOError:
            line = ""
        if battery_ok(line, minimum):
            return
        log("battery: {!r}, wait {} s".format(line, wait_s))
        time.sleep(wait_s)


# -- comparison -----------------------------------------------------------------------------------

def graded_mask(blocks, n_frames):
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        mask[first:last + 1] = True
    return mask


def runs_of(mask):
    """[(first, end)] runs of True, end exclusive."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def compare_codes(a, b, graded):
    """{"frames", "graded_frames", "runs", "listed"}: where code arrays a and b differ."""
    diff = np.asarray(a) != np.asarray(b)
    where = np.flatnonzero(diff)
    return {"frames": int(diff.sum()), "graded_frames": int((diff & graded).sum()),
            "runs": [list(r) for r in runs_of(diff)[:MAX_LOGGED]],
            "listed": [[int(f), int(a[f]), int(b[f])] for f in where[:MAX_LOGGED]]}


def candidates_of(costs):
    """("M", "H"), or ("M",) when the costs name no H cost (round 4: no candidate H)."""
    return CANDIDATES if costs.get("H") is not None else ("M",)


def verdict(per_match, candidates=CANDIDATES, max_share=MAX_SHARE):
    """{"pass", per candidate totals}: Section 6 rule over the per-match records.

    A candidate passes if every end-to-end run equals the numpy path and the torch-numpy
    disagreements are 0, or all logged and below max_share of the graded frames.
    """
    out = {"pass": True}
    graded = sum(rec["n_graded"] for rec in per_match.values())
    for cand in candidates:
        disagree = sum(rec[cand]["torch_vs_numpy"]["graded_frames"] for rec in per_match.values())
        logged = all(len(rec[cand]["torch_vs_numpy"]["listed"]) ==
                     rec[cand]["torch_vs_numpy"]["frames"] for rec in per_match.values())
        e2e = all(rec[cand]["end_to_end_equal"] for rec in per_match.values())
        share = disagree / float(graded) if graded else 0.0
        ok = bool(e2e and (disagree == 0 or (logged and share < max_share)))
        out[cand] = {"pass": ok, "disagree_graded_frames": int(disagree),
                     "graded_frames": int(graded), "share": share, "all_logged": bool(logged),
                     "end_to_end_equal_all": bool(e2e)}
        out["pass"] = out["pass"] and ok
    return out


# -- torch ----------------------------------------------------------------------------------------

def torch_copy_probabilities(state, copies, dtype="float64"):
    """(4, bins, 3) class probabilities of one torch model on the symmetric copies.

    float64: the model in double precision, one copy at a time, as train.predict_p5().
    float32: the float32 path of model.predict_symmetric(). Swapped copies get H and A back.
    """
    import torch
    from engine import learned
    from tools.learned_spike import model
    net = model.net_for_state(state)    # input width of its feature version, hidden size
    if dtype == "float64":
        net = net.double()
        net.load_state_dict({k: v.double() for k, v in state.items()})
    else:
        net.load_state_dict({k: v.float() for k, v in state.items()})
    net.eval()
    out = np.empty(copies.shape[:2] + (learned.N_CLASSES,))
    with torch.no_grad():
        for k, (swap, _) in enumerate(learned.SYMMETRIES):
            x = torch.from_numpy(np.ascontiguousarray(copies[k]))[None]
            x = x.double() if dtype == "float64" else x.float()
            p = torch.softmax(net(x)[0], dim=-1).double().numpy()
            out[k] = p[:, [1, 0, 2]] if swap else p
    return out


def torch_ensemble(states, feats, dtype="float64"):
    """float64 (bins, 3): the unrounded torch mean in the summation order of the engine."""
    from engine import learned
    copies = learned.symmetric_copies(feats)
    total = np.zeros((copies.shape[1], learned.N_CLASSES))
    for state in states:
        for p in torch_copy_probabilities(state, copies, dtype):
            total += p
    return total / (len(learned.SYMMETRIES) * len(states))


def torch_ensemble_paired(states, feats, feats_swapped, dtype="float64"):
    """The torch counterpart of engine.learned.predict_ensemble_paired (LEARNED_PF on): copies
    as given, mirror, swapped tracking, swapped and mirror; each copy added to its swapped
    partner (H and A exchanged back) first, in the engine's order."""
    import torch
    from engine import learned
    from engine import learned_features as lf
    from tools.learned_spike import model
    f = np.asarray(feats, np.float64)
    g = np.asarray(feats_swapped, np.float64)
    copies = [f, lf.transform(f, mirror=True), g, lf.transform(g, mirror=True)]
    total = np.zeros((f.shape[0], learned.N_CLASSES))
    for state in states:
        net = model.net_for_state(state)
        if dtype == "float64":
            net = net.double()
            net.load_state_dict({k: v.double() for k, v in state.items()})
        else:
            net.load_state_dict({k: v.float() for k, v in state.items()})
        net.eval()
        p = []
        with torch.no_grad():
            for x in copies:
                t = torch.from_numpy(np.ascontiguousarray(x))[None]
                t = t.double() if dtype == "float64" else t.float()
                p.append(torch.softmax(net(t)[0], dim=-1).double().numpy())
        for k in (0, 1):
            total += p[k] + p[k + 2][:, [1, 0, 2]]
    return total / (len(learned.SYMMETRIES) * len(states))


def weights_equal(states, nets):
    """True if each torch state, cast to float64, equals the numpy model of the same index."""
    if len(states) != len(nets):
        return False
    for state, net in zip(states, nets):
        for name, array in net.items():
            if not np.array_equal(state[name].detach().double().numpy(), array):
                return False
    return True


def load_states(paths):
    import torch
    return [torch.load(p, map_location="cpu", weights_only=True) for p in paths]


# -- the engine runs ------------------------------------------------------------------------------

def read_codes(path, n_frames):
    import possession_metrics as pm
    from engine import learned
    frame = pm.read_label_csv(path)
    if not np.array_equal(frame["frame"].to_numpy(), np.arange(n_frames)):
        raise ValueError("{}: frames are not 0..{}".format(path, n_frames - 1))
    return learned.labels_to_codes(frame["label"].to_numpy().astype(str))


def engine_run(args, game, system, over, root, folder):
    """Run the engine in a child process (venv of --engine-python). Return the CSV path."""
    os.makedirs(folder, exist_ok=True)
    base = os.path.join(folder, "{}_{}".format(game, system))
    with open(base + ".config.json", "w") as handle:
        json.dump(over, handle, indent=1, sort_keys=True)
    command = [args.engine_python, "-m", "tools.learned_spike.repro_check", "engine-child",
               os.path.abspath(args.engine_cache), "--game", game, "--config",
               base + ".config.json", "--out", base + ".tmp.csv"]
    env = dict(os.environ, MTAH_DATA_DIR=root)
    wait_battery(args.battery_log)
    with open(base + ".log", "w") as out:
        code = subprocess.call(command, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
    if code != 0:
        raise RuntimeError("engine run failed: {} {}, see {}.log".format(game, system, base))
    os.replace(base + ".tmp.csv", base + ".csv")
    return base + ".csv"


def check_weights_md5(path, expected):
    got = md5_file(path)
    if expected and got != expected:
        raise SystemExit("weights md5 changed: {} is {}, expected {}".format(path, got, expected))
    return got


def guarded_tactical(tactical, game, blocks, cfg):
    """(tactical, guard rows) as possession.engine_tactical() makes it from load_tactical()."""
    from engine import data, tracks
    pitch = data.pitch_m(game)
    tactical = tracks.speed_from_positions(tactical, pitch)
    patched, swaps, _ = tracks.guard_goalkeeper_swaps(tactical, blocks, cfg, pitch)
    return patched, int(len(swaps))


def check_match(args, game, nets, states, costs, roots):
    """The record of one match (module docstring, steps 1 to 5)."""
    from tools.analysis import external_sweep as xs
    from tools.analysis import switch_off_identity as soi
    from engine import data, learned
    from engine import learned_features as lf
    from engine.config import Config
    from tools.learned_spike import evaluate, folds

    folds.check(game)
    t0 = time.time()
    root = xs.data_root(game, "raw", *roots)
    folder = os.path.join(args.work, "runs")
    weights = os.path.abspath(args.weights)
    rec = {"game": game, "data_dir": root}

    # 1. End-to-end engine runs.
    csv = {"c22": engine_run(args, game, "c22", evaluate.overrides("c22", None, costs), root,
                             folder)}
    cands = candidates_of(costs)
    for cand in cands:
        over = evaluate.overrides(cand, weights, costs)
        if cand == "M" and getattr(args, "structure", False):
            over["LEARNED_STRUCTURE"] = True        # R3_DEVIATION_STRUCTURE.md
        if cand == "M" and getattr(args, "prior_bonus", 0.0):
            over["LEARNED_PRIOR_BONUS"] = float(args.prior_bonus)   # round 4 prior
        csv[cand] = engine_run(args, game, cand, over, root, folder)
    with open(getattr(args, "switch_off_record", None) or soi.RECORD) as handle:
        record = json.load(handle)
    rec["c22_md5"] = md5_file(csv["c22"])
    rec["c22_md5_recorded"] = record.get("{}/raw".format(game))
    rec["c22_equals_switch_off_record"] = rec["c22_md5"] == rec["c22_md5_recorded"]
    rec["engine_md5"] = {cand: md5_file(csv[cand]) for cand in cands}

    # 2. Features of the engine.
    wait_battery(args.battery_log)
    n_frames = data.frame_count(game)
    blocks = data.graded_blocks(game)
    graded = graded_mask(blocks, n_frames)
    tactical = data.load_tactical(game, data_dir=root,
                                  cache_dir=os.path.abspath(args.engine_cache))
    if int(tactical["n_frames"]) != n_frames:
        raise ValueError("{}: {} tracking frames, games.py says {}".format(
            game, tactical["n_frames"], n_frames))
    cfg = Config.default()
    rec["tactical"] = getattr(args, "tactical", "raw")
    if rec["tactical"] == "guarded":
        raw_feats = learned.features_for(nets, tactical, game, cfg)
        rec["raw_feature_md5"] = lf.matrix_md5(raw_feats)
        del raw_feats
        tactical, rec["guard_rows"] = guarded_tactical(tactical, game, blocks, cfg)
    # The features of the weights' version (engine.learned.features_for, as engine.learned.apply).
    rec["feature_version"] = learned.model_version(nets)
    feats = learned.features_for(nets, tactical, game, cfg)
    rec.update(n_frames=int(n_frames), n_graded=int(graded.sum()), bins=int(feats.shape[0]),
               feature_md5=lf.matrix_md5(feats))

    # 3. and 4. Probabilities of the two paths. With the per-frame member (round 6, costs "pf",
    # Config.LEARNED_PF on in the engine runs) the network's swapped copies come from the
    # swapped tracking, paired (engine.learned.predict_ensemble_paired), and the frame
    # probabilities are blended with the member's (numpy only: it has no torch path; its
    # scikit-learn parity is tests/test_learned_pf.py).
    pf = costs.get("pf")
    if pf:
        from engine import learned_pf
        from tools.learned_spike import pf_gate
        pf_file = pf_gate.pf_path(pf["weights"])      # "{fold}" -> the final member
        pf_model = learned_pf.load_pf(pf_file)
        if pf_model.get("swap_mode", "columns") == "tracking":
            g_feats = learned.features_for(nets, learned_pf.swapped_tactical(tactical), game,
                                           cfg)
            p_np = learned.predict_ensemble_paired(nets, feats, g_feats)
            p_t64 = torch_ensemble_paired(states, feats, g_feats, "float64")
            p_t32 = torch_ensemble_paired(states, feats, g_feats, "float32")
        else:
            p_np = learned.predict_ensemble(nets, feats)
            p_t64 = torch_ensemble(states, feats, "float64")
            p_t32 = torch_ensemble(states, feats, "float32")
        p_pf = learned_pf.frame_probabilities_pf(tactical, game, pf_model, cfg)
        rec["pf"] = {"weights": pf_file, "weights_template": pf["weights"],
                     "weight": float(pf["weight"]), "md5": md5_file(pf_file)}
    else:
        p_np = learned.predict_ensemble(nets, feats)
        p_t64 = torch_ensemble(states, feats, "float64")
        p_t32 = torch_ensemble(states, feats, "float32")
    r_np = np.round(p_np, learned.PROB_DECIMALS)
    r_t64 = np.round(p_t64, learned.PROB_DECIMALS)
    r_t32 = np.round(p_t32, learned.PROB_DECIMALS)
    rec["max_dp"] = float(np.abs(p_np - p_t64).max())
    rec["max_dp_rounded"] = float(np.abs(r_np - r_t64).max())
    rec["rounded_values_differ"] = int((r_np != r_t64).sum())
    f_np = learned.upsample(r_np, n_frames)
    f_t64 = learned.upsample(r_t64, n_frames)
    f_t32 = learned.upsample(r_t32, n_frames)
    if pf:
        f_np, f_t64, f_t32 = (learned.blend_pf(f, p_pf, float(pf["weight"]))
                              for f in (f_np, f_t64, f_t32))

    # 5. Decoded labels.
    c22 = read_codes(csv["c22"], n_frames)
    cost_ha, cost_ad = costs["M"]

    forced = (learned.pre_kickoff_mask(n_frames, blocks, data.kickoff_frames(game))
              if getattr(args, "structure", False) else None)
    rec["structure"] = bool(getattr(args, "structure", False))
    bonus = float(getattr(args, "prior_bonus", 0.0) or 0.0)
    rec["prior_bonus"] = bonus

    def decode(cand, frames):
        if cand == "M":
            if bonus:
                # the engine's prior reads the labels of LEARNED_MODE off (c23) of the same run
                return learned.decode_model(frames, blocks, float(cost_ha), float(cost_ad),
                                            forced, rule_codes=c22, prior_bonus=bonus)
            return learned.decode_model(frames, blocks, float(cost_ha), float(cost_ad), forced)
        return learned.decode_hybrid(frames, c22, float(costs["H"]))

    for cand in cands:
        lab_np, lab_t64, lab_t32 = (decode(cand, f) for f in (f_np, f_t64, f_t32))
        lab_e2e = read_codes(csv[cand], n_frames)
        rec[cand] = {"torch_vs_numpy": compare_codes(lab_t64, lab_np, graded),
                     "end_to_end_equal": bool(np.array_equal(lab_e2e, lab_np)),
                     "end_to_end_vs_numpy_frames": int((lab_e2e != lab_np).sum()),
                     "numpy_codes_md5": codes_md5(lab_np), "torch_codes_md5": codes_md5(lab_t64),
                     "end_to_end_codes_md5": codes_md5(lab_e2e),
                     "float32_info": {"graded_frames_differ": int(((lab_t32 != lab_np)
                                                                   & graded).sum())}}
    rec["float32_info"] = {"max_dp": float(np.abs(p_np - p_t32).max()),
                           "rounded_values_differ": int((r_np != r_t32).sum())}
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def cmd_run(args):
    from engine import learned
    from tools.learned_spike import evaluate, folds

    games = folds.check_all(list(args.games) if args.games else list(folds.OPEN))
    for game in games:
        if game not in folds.OPEN:
            raise SystemExit("{}: only the eight open matches are checked".format(game))
    args.work = os.path.abspath(args.work)
    os.makedirs(args.work, exist_ok=True)
    weights_md5 = check_weights_md5(args.weights, args.weights_md5)
    nets = learned.load_weights(os.path.abspath(args.weights))
    states = load_states(args.nets)
    if not weights_equal(states, nets):
        raise SystemExit("the torch states differ from {}".format(args.weights))
    import torch
    torch.set_num_threads(args.threads)
    costs = evaluate.costs_record(args.costs, None)
    roots = (os.path.expanduser(args.refs_root), os.path.expanduser(args.sample_root),
             os.path.expanduser(args.mtah_root))
    per_match = {}
    for game in games:
        log("check {}".format(game))
        check_weights_md5(args.weights, weights_md5)
        per_match[game] = check_match(args, game, nets, states, costs, roots)
        r = per_match[game]
        h = r.get("H") or {"torch_vs_numpy": {"graded_frames": None}, "end_to_end_equal": None}
        log("{}: max_dp {:.3g}, M {} / H {} graded frames differ, end-to-end M {} H {}, "
            "c22 record {}, {} s".format(game, r["max_dp"],
                                         r["M"]["torch_vs_numpy"]["graded_frames"],
                                         h["torch_vs_numpy"]["graded_frames"],
                                         r["M"]["end_to_end_equal"], h["end_to_end_equal"],
                                         r["c22_equals_switch_off_record"], r["seconds"]))
    check_weights_md5(args.weights, weights_md5)
    result = verdict(per_match, candidates_of(costs))
    result["c22_equals_switch_off_record_all"] = all(
        r["c22_equals_switch_off_record"] for r in per_match.values())
    out = {"pass": bool(result["pass"]), "summary": result,
           "max_dp": max(r["max_dp"] for r in per_match.values()),
           "weights": os.path.abspath(args.weights), "weights_md5": weights_md5,
           "nets": {os.path.abspath(p): md5_file(p) for p in args.nets},
           "costs": {"M": list(costs["M"]), "H": costs["H"], "source": costs["source"]},
           "tactical": args.tactical,
           "torch": torch.__version__, "numpy": np.__version__,
           "engine_python": args.engine_python, "per_match": per_match}
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=1, sort_keys=True)
        handle.write("\n")
    log("pass {} (M {}, H {}); wrote {}".format(out["pass"], result["M"]["pass"],
                                               (result.get("H") or {}).get("pass"), args.out))
    return 0 if out["pass"] else 1


def cmd_engine_child(argv):
    """engine.run.main(argv) with the engine.data cache in argv[0] (external_sweep)."""
    from analysis import external_sweep as xs
    return xs.engine_child(argv[0], argv[1:])   # the module evaluate.run_one() runs


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["engine-child"]:
        return cmd_engine_child(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="check the eight open matches")
    p.add_argument("--nets", nargs="+", required=True, help="torch state files, in npz order")
    p.add_argument("--weights", required=True, help="the final .npz of engine/models/")
    p.add_argument("--weights-md5", help="stop if the .npz md5 differs from this")
    p.add_argument("--costs", required=True, help="final costs.json of run_stage.sh")
    p.add_argument("--engine-cache", required=True, help="engine.data cache of the matches")
    p.add_argument("--engine-python", default=sys.executable, help="python of the engine venv")
    p.add_argument("--work", required=True, help="folder for the engine runs")
    p.add_argument("--out", required=True, help="the JSON record")
    p.add_argument("--battery-log", help="battery guard file; wait before each heavy step")
    p.add_argument("--games", nargs="+", help="a subset of the eight open matches")
    p.add_argument("--threads", type=int, default=1, help="torch threads")
    p.add_argument("--tactical", choices=("raw", "guarded"), default="raw",
                   help="ATD of step 2: raw load_tactical() (round 1) or the guarded ATD of "
                        "possession.engine_tactical() (round 2, R2 Section 9, D1)")
    p.add_argument("--structure", action="store_true",
                   help="M with Config.LEARNED_STRUCTURE on (R3_DEVIATION_STRUCTURE.md), in the "
                        "engine runs and in both decoded paths")
    p.add_argument("--prior-bonus", type=float, default=0.0,
                   help="round 4: M with Config.LEARNED_PRIOR_BONUS, in the engine runs and in "
                        "both decoded paths (rule codes: the c22/c23 run of the same match)")
    p.add_argument("--switch-off-record", help="switch-off md5 record of the baseline (round 3: "
                   "docs/plan/calibration/learned_switch_off_md5_c23.json; default the c22 "
                   "record of round 1 and 2)")
    p.add_argument("--refs-root", default=os.path.join("~", "mtah2027", "refs"),
                   help="as evaluate.py --refs-root")
    p.add_argument("--sample-root", default=os.path.join("~", "mtah2027", "sample-data", "data"),
                   help="as evaluate.py --sample-root")
    p.add_argument("--mtah-root", default=os.path.join("~", "mtah2027"),
                   help="as evaluate.py --mtah-root")
    args = parser.parse_args(argv)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
