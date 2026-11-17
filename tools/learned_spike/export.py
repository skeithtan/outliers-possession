"""Export torch state dictionaries of model.Net to one weights file of engine/learned.py.

    python -m tools.learned_spike.export --out FILE.npz NET.pt [NET.pt ...]
    python -m tools.learned_spike.export --out FILE.npz --lomo DIR

Each NET.pt is a state dictionary that torch.save() wrote (lomo.py writes DIR/<game>/net.pt).
With --lomo the tool takes every DIR/*/net.pt in sorted order. The order of the files is the
order of the models in the ensemble.

The weights file holds these arrays:

- `n_models`: the number of models, int64.
- `feature_version`: the feature version of the models (engine.learned_features.VERSIONS), a
  string. Without --feature-version it is the version of the input width of the models
  (learned_features.version_of_width()); "stage50-v2" for the round 2 to 4 models.
- `m{k}/<name>`: each array of model k under its state dictionary name, as float64.

The tool loads the file back with engine.learned.load_weights() and stops if the check fails.
np.savez writes no time stamp into the arrays, so the same state dictionaries give the same
arrays. Python 3.9. torch for the input only.
"""

import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from engine import learned  # noqa: E402
from engine import learned_features  # noqa: E402


def recorded_version(paths):
    """The feature version of saved networks from their training records (train.net_version);
    None if no record names one. Several networks must agree."""
    import json
    out = set()
    for path in paths:
        folder, name = os.path.split(os.path.abspath(path))
        cands = [os.path.join(folder, "model.json")]
        if name.startswith("net_s") and name.endswith(".pt"):
            s = name[len("net_s"):-len(".pt")]
            cands = [os.path.join(folder, "model_s{}.json".format(s)),
                     os.path.join(folder, "final_s{}.json".format(s))] + cands
        for rec in cands:
            if os.path.isfile(rec):
                with open(rec) as handle:
                    r = json.load(handle)
                r4 = (r.get("switches") or {}).get("r4") or {}
                out.add(r4.get("feature_version") or learned_features.FEATURE_VERSION)
                break
    if len(out) > 1:
        raise ValueError("networks trained on different feature versions {}".format(sorted(out)))
    return out.pop() if out else None


def state_version(state_dicts):
    """The feature version of the input width of the models (all must have the same width)."""
    widths = sorted(set(int(tuple(s["inp.0.weight"].shape)[1]) for s in state_dicts))
    if len(widths) != 1:
        raise ValueError("models of different input widths {}".format(widths))
    return learned_features.version_of_width(widths[0])


def arrays_of(state_dicts, version=None):
    """{key: array} of a weights file for a list of {name: tensor or array} dictionaries.

    `version` None takes the version of the input width (state_version(); 50 columns are
    stage50-v2): a stage50-v2s model must be given its version (main() and evaluate's export
    read it from the training records, recorded_version()). A version whose width is not that
    of the models raises ValueError.
    """
    found = state_version(state_dicts)
    version = found if version is None else learned_features.check_version(version)
    if learned_features.n_features(version) != learned_features.n_features(found):
        raise ValueError("feature version {} has {} columns, the models read {}".format(
            version, learned_features.n_features(version), learned_features.n_features(found)))
    out = {"n_models": np.int64(len(state_dicts)),
           "feature_version": np.str_(version)}
    for k, state in enumerate(state_dicts):
        names = sorted(state)
        if sorted(learned.LAYER_NAMES) != names:
            missing = sorted(set(learned.LAYER_NAMES) - set(names))
            extra = sorted(set(names) - set(learned.LAYER_NAMES))
            raise ValueError("model {}: missing {}, extra {}".format(k, missing, extra))
        for name in learned.LAYER_NAMES:
            value = state[name]
            if hasattr(value, "detach"):
                value = value.detach().cpu().double().numpy()
            out["m{}/{}".format(k, name)] = np.asarray(value, dtype=np.float64)
    return out


def write(path, state_dicts, version=None):
    """Write the weights file and check it with engine.learned.load_weights()."""
    np.savez(path, **arrays_of(state_dicts, version))
    nets = learned.load_weights(path)
    if len(nets) != len(state_dicts):
        raise RuntimeError("{}: wrote {} models, read {}".format(path, len(state_dicts),
                                                                 len(nets)))
    return path


def main(argv=None):
    import torch

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="the .npz file to write")
    parser.add_argument("--lomo", help="take every DIR/*/net.pt, sorted")
    parser.add_argument("nets", nargs="*", help="state dictionary files, in ensemble order")
    parser.add_argument("--feature-version", choices=learned_features.VERSIONS, default=None,
                        help="checked against the input width of the models (default: taken "
                             "from it)")
    args = parser.parse_args(argv)
    paths = list(args.nets)
    if args.lomo:
        paths += sorted(glob.glob(os.path.join(args.lomo, "*", "net.pt")))
    if not paths:
        parser.error("give NET.pt files or --lomo DIR")
    states = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    write(args.out, states, args.feature_version or recorded_version(paths))
    for k, path in enumerate(paths):
        print("m{} {}".format(k, path))
    print("wrote {} models to {}".format(len(paths), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
