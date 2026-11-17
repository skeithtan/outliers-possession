"""Export the per-frame member's tree models to the engine's weights file. Not part of the submission.

    # pf venv (scikit-learn): the spike's pickle {"dead", "team", "cols"} (pf_train.py)
    python -m tools.learned_spike.pf_export --pkl MODELS.pkl --base-version stage50-v2s \
        --version F5 --out engine/models/learned_pf_v1.npz
    # numpy only: flat arrays already exported (pf_model.npz with a JSON column list)
    python -m tools.learned_spike.pf_export --npz PF_MODEL.npz --columns-json COLS.json \
        --base-version stage50-v2s --version F5 --out FILE.npz

The weights file (engine.learned_pf.load_pf reads it):

- `format`: engine.learned_pf.PF_FORMAT; `version`: a free version string (e.g. "F5");
  `base_version`: the feature version of the 50 base columns (engine.learned_features);
- `columns`: the column names in the models' feature order (engine.learned_pf.full_spec names);
- per model m in ("dead", "team"): `m/feature`, `m/threshold`, `m/left`, `m/right` (absolute
  node indices), `m/missing_left`, `m/leaf`, `m/value`, `m/roots` (the root node of each tree, in
  the order the trees are summed) and `m/baseline`;
- `source_md5`: the md5 of the input file;
- `group_scale` (only when not "raw"): "robust" for the group columns of member spec F6c
  (engine/learned_pf_groups.py robust_units());
- `swap_mode`: "columns" (default: the team-swapped copy is the column-permuted features, as the
  spike measured it) or "tracking" (the features of the swapped tracking: exactly symmetric;
  see engine/learned_pf.py for the difference).

The column names: from --columns-json (the model's JSON, e.g. final_F5/pf_model.json, whose
"base_feature_version" must equal --base-version); without it, a pickle of the full feature set
maps its "cols" indices onto engine.learned_pf.full_spec() (the order of the spike's
pf_features.py). A pickle of a named feature set ("fset", e.g. F5) needs --columns-json. The
tool loads the result back with engine.learned_pf.load_pf() (which refuses the contact
columns) unless --no-check, and prints the file's md5. Python 3.9.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from engine import learned_pf  # noqa: E402


def md5_file(path):
    d = hashlib.md5()
    with open(path, "rb") as h:
        for block in iter(lambda: h.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


def arrays_from_sklearn(clf):
    """Flat node arrays of a fitted HistGradientBoostingClassifier (binary, numerical splits)."""
    feats, thr, left, right, miss, leaf, value, roots = [], [], [], [], [], [], [], []
    off = 0
    for predictors in clf._predictors:
        if len(predictors) != 1:
            raise SystemExit("binary classifiers only (one predictor per iteration)")
        nd = predictors[0].nodes
        if nd["is_categorical"].any():
            raise SystemExit("categorical splits are not supported")
        roots.append(off)
        feats.append(nd["feature_idx"].astype(np.int32))
        thr.append(nd["num_threshold"].astype(np.float64))
        left.append(np.where(nd["is_leaf"], -1, nd["left"].astype(np.int64) + off))
        right.append(np.where(nd["is_leaf"], -1, nd["right"].astype(np.int64) + off))
        miss.append(nd["missing_go_to_left"].astype(bool))
        leaf.append(nd["is_leaf"].astype(bool))
        value.append(nd["value"].astype(np.float64))
        off += len(nd)
    return {"feature": np.concatenate(feats), "threshold": np.concatenate(thr),
            "left": np.concatenate(left).astype(np.int32),
            "right": np.concatenate(right).astype(np.int32),
            "missing_left": np.concatenate(miss), "leaf": np.concatenate(leaf),
            "value": np.concatenate(value), "roots": np.array(roots, np.int32),
            "baseline": float(np.ravel(clf._baseline_prediction)[0])}


def column_names(cols, base_version):
    names = [n for n, _ in learned_pf.full_spec(base_version)]
    return [names[int(c)] for c in cols]


def read_columns(path, base_version):
    """Column names from a JSON: a list, {"columns": [...]} (with "base_feature_version", which
    must equal base_version when present) or {"cols": [indices into full_spec()]}."""
    with open(path) as h:
        spec = json.load(h)
    if isinstance(spec, dict):
        base = spec.get("base_feature_version")
        if base is not None and base != base_version:
            raise SystemExit("{}: base version {}, --base-version {}".format(path, base,
                                                                          base_version))
        if "columns" in spec:
            return [str(c) for c in spec["columns"]]
        if "cols" in spec:
            return column_names(spec["cols"], base_version)
    return [str(c) for c in spec]


def arrays_of(models, columns, base_version, version, source_md5, swap_mode="columns",
              group_scale="raw"):
    """{key: array} of the weights file from {"dead": arrays, "team": arrays}."""
    if swap_mode not in learned_pf.SWAP_MODES:
        raise SystemExit("swap mode must be one of {}".format(learned_pf.SWAP_MODES))
    out = {"format": np.str_(learned_pf.PF_FORMAT), "version": np.str_(version),
           "base_version": np.str_(base_version), "columns": np.array(columns),
           "source_md5": np.str_(source_md5), "swap_mode": np.str_(swap_mode)}
    if group_scale != "raw":      # member spec F6c (round 8); a raw file keeps its old keys
        if group_scale not in learned_pf.lpg.SCALES:
            raise SystemExit("group scale must be one of {}".format(learned_pf.lpg.SCALES))
        out["group_scale"] = np.str_(group_scale)
    for name in learned_pf.MODELS:
        m = models[name]
        for k in learned_pf.NODE_ARRAYS + ("roots",):
            out["{}/{}".format(name, k)] = np.asarray(m[k])
        out["{}/baseline".format(name)] = np.float64(m["baseline"])
    return out


def from_flat_npz(path, prefix_styles=("{m}/{k}", "{m}_{k}")):
    """{"dead", "team"} arrays from an npz of flat arrays (keys "<model>/<array>" or
    "<model>_<array>"; arrays named as here, or as tools/learned_spike/pf_numpy.export)."""
    alias = {"missing_left": ("missing_left", "missing_go_to_left"), "leaf": ("leaf", "is_leaf")}
    with np.load(path, allow_pickle=False) as z:
        stored = {k: z[k] for k in z.files}
    out = {}
    for m in learned_pf.MODELS:
        arrays = {}
        for k in learned_pf.NODE_ARRAYS + ("roots", "baseline"):
            found = None
            for style in prefix_styles:
                for name in alias.get(k, (k,)):
                    key = style.format(m=m, k=name)
                    if key in stored:
                        found = stored[key]
                        break
                if found is not None:
                    break
            if found is None and k == "missing_left":
                n = len(arrays["feature"])
                found = np.zeros(n, bool)       # no missing value in the spike's features
            if found is None:
                raise SystemExit("{}: no array for {} {}".format(path, m, k))
            arrays[k] = found
        arrays["baseline"] = float(np.ravel(arrays["baseline"])[0])
        out[m] = arrays
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pkl", help="the spike's models.pkl (needs scikit-learn)")
    src.add_argument("--npz", help="flat arrays (with --columns-json)")
    p.add_argument("--columns-json", help="JSON: a list of column names, or {'columns': [...]} "
                                          "or {'cols': [indices]}; required with --npz and for "
                                          "a pickle of a named feature set (F5)")
    p.add_argument("--base-version", required=True, help="feature version of the base columns")
    p.add_argument("--version", required=True, help="version string recorded in the file")
    p.add_argument("--out", required=True)
    p.add_argument("--swap-mode", choices=learned_pf.SWAP_MODES, default="columns",
                   help="how the engine makes the team-swapped copy (engine/learned_pf.py): "
                        "columns (default, as the spike measured it) or tracking (exactly "
                        "symmetric)")
    p.add_argument("--group-scale", choices=("raw", "robust"), default="raw",
                   help="units of the group columns (engine/learned_pf_groups.py): raw, or "
                        "robust units per match (member spec F6c, round 8)")
    p.add_argument("--no-check", action="store_true",
                   help="do not load the result back (files with contact columns, tests)")
    a = p.parse_args(argv)
    if a.pkl:
        import pickle
        with open(a.pkl, "rb") as h:
            obj = pickle.load(h)
        if obj.get("flag_column"):
            raise SystemExit("models with the source flag column cannot run in the engine")
        models = {m: arrays_from_sklearn(obj[m]) for m in learned_pf.MODELS}
        if a.columns_json:
            columns = read_columns(a.columns_json, a.base_version)
        elif obj.get("fset") not in (None, "full"):
            # its "cols" index the feature set's own spec, not full_spec(): the names must
            # come from the model's JSON
            raise SystemExit("feature set {}: give --columns-json".format(obj.get("fset")))
        else:
            columns = column_names(obj["cols"], a.base_version)
        n_feat = obj["team"].n_features_in_
        if obj["dead"].n_features_in_ > n_feat:
            raise SystemExit("DEAD reads more columns than TEAM")
        # (a DEAD model may read only the leading columns: member spec F6a, round 8)
        if len(columns) != n_feat:
            raise SystemExit("{} columns for models of {} features".format(len(columns), n_feat))
        source = a.pkl
    else:
        if not a.columns_json:
            p.error("--npz needs --columns-json")
        columns = read_columns(a.columns_json, a.base_version)
        models = from_flat_npz(a.npz)
        source = a.npz
    np.savez(a.out, **arrays_of(models, columns, a.base_version, a.version, md5_file(source),
                                a.swap_mode, a.group_scale))
    if not a.no_check:
        learned_pf.load_pf(a.out)
    print("wrote {} ({} columns, dead {} trees, team {} trees), md5 {}".format(
        a.out, len(columns), len(models["dead"]["roots"]), len(models["team"]["roots"]),
        md5_file(a.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
