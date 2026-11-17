"""The candidate "network + per-frame member" on external matches, through engine.learned. Not part of the submission.

The gate tools (r4_screen score / score-sc-oof / grid, g5_dev_oof.py, r4_oof_ratio) score the
network alone from cached 5 fps features and torch. With --pf-weights they score the candidate
as the engine computes it instead (engine.learned.apply with LEARNED_PF "on"):

- the tracking of the test's input (raw view; DFL with the still ball removed, input "c" of
  r4_inputs.load_dfl), with the engine's goalkeeper guard (possession.engine_tactical: speed from
  positions where the ATD has none, tracks.guard_goalkeeper_swaps);
- the network: the given torch state dictionaries as numpy models (engine.learned, the feature
  version of their training records);
- network, member and blend: engine.learned.candidate_frame_probabilities (the member's swap
  mode decides how the network's swapped copies are made).
The result is the (N, 3) frame probabilities that engine.learned.apply hands to the decoder.
They are cached under a key of the network files' md5, the weights file's md5, the weight, the
input kind and the engine source hash.

Out-of-sample members. A left-out or out-of-fold network must be blended with a per-frame member
that did not see the match either. --pf-weights is a path template: "{provider}" is replaced by
the left-out provider (metrica, dfl, skillcorner) for the left-out networks of r4_screen score
and grid, "{fold}" by the fold number for out-of-fold networks (score-sc-oof, G5, the oof ratio,
r3_eval's fold dumps) and by "final" for the final models. A path without a placeholder is used
for every network, and the records say "pf_shared": true (the member then saw the matches it is
scored on). Python 3.9.
"""

import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from tools.learned_spike import folds  # noqa: E402

KINDS = ("raw", "c")


def md5_file(path):
    d = hashlib.md5()
    with open(path, "rb") as h:
        for block in iter(lambda: h.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


def nets_from_paths(paths):
    """engine.learned.Models of torch state dictionaries (the version of their training
    records, export.recorded_version; else of their width)."""
    import torch
    from engine import learned
    from tools.learned_spike import export
    states = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    version = export.recorded_version(paths)
    arrays = export.arrays_of(states, version)
    nets = learned.Models(str(arrays["feature_version"]))
    for k in range(len(states)):
        nets.append({name: np.asarray(arrays["m{}/{}".format(k, name)], dtype=np.float64)
                     for name in learned.LAYER_NAMES})
    return nets


def guarded(tac, game):
    """possession.engine_tactical() on a loaded tactical dict."""
    from engine import data, tracks
    from engine.config import Config
    pitch = data.pitch_m(game)
    tac = tracks.speed_from_positions(dict(tac, game=game), pitch)
    patched, _, _ = tracks.guard_goalkeeper_swaps(tac, data.graded_blocks(game),
                                                  Config.default(), pitch)
    return patched


def input_tactical(game, kind, engine_cache):
    """The guarded tactical dict of a test input: "raw", or "c" (DFL still ball removed)."""
    from tools.learned_spike import build, r4_inputs
    if kind not in KINDS:
        raise ValueError(kind)
    folds.check(game)
    tac = build.load_view(game, "raw", engine_cache)
    if kind == "c" and game.startswith("DFL-"):
        tac, _ = r4_inputs.load_dfl(tac, game, "c")
    return guarded(tac, game)


def blended_frame_probs(game, kind, net_paths, pf_path, weight, engine_cache, cache_dir=None):
    """float64 (N, 3): the engine's frame probabilities of the candidate on one test input."""
    from engine import learned, learned_pf
    from analysis import external_sweep as xs
    key = None
    if cache_dir:
        key = hashlib.sha1(json.dumps({
            "game": game, "kind": kind, "nets": [md5_file(p) for p in net_paths],
            "pf": md5_file(pf_path), "weight": float(weight),
            "engine": xs.engine_source_hash()}, sort_keys=True).encode()).hexdigest()[:16]
        path = os.path.join(cache_dir, "{}__{}.npy".format(game, key))
        if os.path.isfile(path):
            return np.load(path)
    tac = input_tactical(game, kind, engine_cache)
    nets = nets_from_paths(net_paths)
    p = learned.candidate_frame_probabilities(nets, None, tac, game,
                                              learned_pf.load_pf(pf_path), float(weight))
    if key is not None:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(path + ".tmp.npy", p)
        os.replace(path + ".tmp.npy", path)
    return p


def pf_path(template, provider=None, fold=None):
    """The member's weights file for a network: "{provider}" and "{fold}" filled in (fold None:
    "final")."""
    return template.replace("{provider}", str(provider)).replace(
        "{fold}", "final" if fold is None else str(fold))


def pf_shared(template):
    """True when the template names one file for every network (no placeholder)."""
    return "{provider}" not in template and "{fold}" not in template


def add_pf_args(parser):
    parser.add_argument("--pf-weights", default=None,
                        help="the per-frame member's weights file: score the candidate network + "
                             "member through engine.learned (pf_gate)")
    parser.add_argument("--pf-weight", type=float, default=0.5,
                        help="the member's weight in the blend (Config.LEARNED_PF_WEIGHT)")


def pf_record(args):
    """The per-frame member settings for a JSON record ({} without --pf-weights)."""
    if not getattr(args, "pf_weights", None):
        return {}
    t = args.pf_weights
    return {"pf": {"weights_template": os.path.abspath(t), "weight": float(args.pf_weight),
                   "pf_shared": pf_shared(t),
                   "md5": md5_file(t) if pf_shared(t) else None}}
