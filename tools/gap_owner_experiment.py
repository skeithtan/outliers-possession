"""Test player-only ownership correction with whole-match nested validation.

Only the three open Metrica matches train an owner model. The five open DFL
matches test transfer. Every copy of a match stays outside its training fold.
All generated arrays, predictions, reports and models stay outside this repository.
The frozen P10-B masks are diagnostic stress tests, not a target-data simulator.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import gap_benchmark as gb
from tools import gapmask as gm
from tools import gap_live_masks as glm
from tools import score_game as sg
from tools.gap_models import gap_owner as go

pm = gb.pm
METRICA = tuple(game for game in gb.OPEN_GAMES if game.startswith("METRICA"))
INPUT_VIEWS = ("raw", "matched")
PRIMARY_VIEWS = ("U", "S")
SEEDS = (0, 1, 2, 3)
# P10-C protocol: mean logistic loss, three regularizers and four local decoders.
L2_GRID = (0.01, 0.1, 1.0)
SWITCH_GRID = (2.0, 8.0)
BASELINE_GRID = (0.5, 1.5)
ENDPOINT_LOG_ODDS = 2.0
TRAIN_STRIDE = 5
CLEAN_LIMIT = 0.01
MAX_ITERATIONS = 50
GRADIENT_TOLERANCE = 1e-8
SCHEMA = 1
SENSITIVITY_VIEWS = tuple(glm.live_view(value) for value in glm.TARGETS)
ALL_VIEWS = gb.VIEWS + SENSITIVITY_VIEWS


def json_hash(value):
    """Hash finite JSON with a stable representation."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    """Hash one local input without printing its content."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array):
    """Include the type and shape in an array checksum."""
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.dtype.str, array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def engine_fingerprint():
    """Hash the modules used by this experiment, excluding unrelated later work."""
    names = ("possession", "tracks", "config", "deadball", "events", "decoder", "data",
             "submission")
    paths = {name: ROOT / "engine" / (name + ".py") for name in names}
    paths["gap_owner"] = ROOT / "tools" / "gap_models" / "gap_owner.py"
    return json_hash({name: file_sha256(path) for name, path in paths.items()})


def write_json(path, value):
    """Write finite JSON outside the source tree."""
    path = gb.outside_repo(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def recipes():
    """Return the fixed grid with conservative choices first for exact ties."""
    result = [{"identity": True, "l2": 1.0, "switch_cost": 8.0,
               "baseline_log_odds": 1.5, "endpoint_log_odds": ENDPOINT_LOG_ODDS}]
    result += [{"identity": False, "l2": ridge, "switch_cost": switch,
                "baseline_log_odds": prior, "endpoint_log_odds": ENDPOINT_LOG_ODDS}
               for ridge, switch, prior in itertools.product(
                   reversed(L2_GRID), reversed(SWITCH_GRID), reversed(BASELINE_GRID))]
    return result


def grouped_folds(games):
    """Keep a complete Metrica match out of each fitting fold."""
    games = tuple(sorted(games))
    if len(games) != len(set(games)) or not set(games).issubset(METRICA) or len(games) < 2:
        raise ValueError("folds require at least two distinct open Metrica matches")
    return [(tuple(game for game in games if game != held), held) for held in games]


def check_split(train_games, test_game):
    """Reject a target match or another provider in the owner training data."""
    if not train_games or len(set(train_games)) != len(train_games):
        raise ValueError("training games must be nonempty and unique")
    if not set(train_games).issubset(METRICA):
        raise ValueError("owner training accepts only open Metrica matches")
    if test_game in train_games:
        raise ValueError("a held-out match cannot appear in training")
    if test_game not in gb.OPEN_GAMES:
        raise ValueError("only open reference matches may be evaluated")


def fit_logistic(features, labels, weights, l2, max_iterations=MAX_ITERATIONS):
    """Fit a zero-intercept logistic model with deterministic Newton steps.

    Mirroring each row and its target leaves this objective unchanged. Zero means
    and RMS scales therefore implement exact symmetric augmentation without copies.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or weight.shape != y.shape or not len(x):
        raise ValueError("invalid training array shapes")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(weight).all():
        raise ValueError("training arrays must be finite")
    if not np.isin(y, (0.0, 1.0)).all() or (weight <= 0).any():
        raise ValueError("training targets must be binary and weights positive")
    if not np.isfinite(l2) or l2 <= 0 or max_iterations < 1:
        raise ValueError("regularization and iteration count must be positive")
    weight = weight / weight.sum()
    scales = np.sqrt(np.sum(weight[:, None] * x * x, axis=0))
    scales[scales == 0] = 1.0
    z = x / scales
    coef = np.zeros(x.shape[1], dtype=np.float64)

    def objective(candidate):
        margin = z @ candidate
        return float(weight @ (np.logaddexp(0.0, margin) - y * margin)
                     + 0.5 * l2 * (candidate @ candidate))

    converged = False
    for iteration in range(max_iterations):
        margin = z @ coef
        probability = np.exp(-np.logaddexp(0.0, -margin))
        gradient = z.T @ (weight * (probability - y)) + l2 * coef
        if float(np.max(np.abs(gradient))) < GRADIENT_TOLERANCE:
            converged = True
            break
        curvature = weight * probability * (1.0 - probability)
        hessian = z.T @ (curvature[:, None] * z) + l2 * np.eye(z.shape[1])
        direction = np.linalg.solve(hessian, gradient)
        before = objective(coef)
        step = 1.0
        while step > 2.0 ** -20:
            candidate = coef - step * direction
            if objective(candidate) <= before:
                coef = candidate
                break
            step *= 0.5
        else:
            raise ValueError("logistic solver could not reduce its objective")
    if not converged:
        margin = z @ coef
        probability = np.exp(-np.logaddexp(0.0, -margin))
        gradient = z.T @ (weight * (probability - y)) + l2 * coef
        converged = float(np.max(np.abs(gradient))) < GRADIENT_TOLERANCE
    if not converged or not np.isfinite(coef).all():
        raise ValueError("logistic solver did not converge")
    return {"means": np.zeros(x.shape[1]).tolist(), "scales": scales.tolist(),
            "coefficients": coef.tolist(), "iterations": iteration + 1,
            "objective": objective(coef), "converged": True}


def fit_model(datasets, train_games, recipe):
    """Build one frozen artifact from the declared training matches only."""
    games = tuple(sorted(train_games))
    if not games or not set(games).issubset(METRICA) or len(set(games)) != len(games):
        raise ValueError("invalid training matches")
    features, targets, weights, hashes, references = [], [], [], [], {}
    for game in games:
        dataset = datasets[game]
        x = np.asarray(dataset["features"], dtype=np.float64)
        reference = np.asarray(dataset["reference"], dtype=np.int8)
        indices = np.flatnonzero(reference < pm.D)[::TRAIN_STRIDE]
        if not len(indices):
            raise ValueError("training match has no live frames")
        features.append(x[indices])
        targets.append((reference[indices] == pm.H).astype(float))
        weights.append(np.full(len(indices), 1.0 / (len(games) * len(indices))))
        references[game] = dataset["reference_sha256"]
        hashes.append({"game": game, "features": array_sha256(x[indices]),
                       "targets": array_sha256(reference[indices]), "stride": TRAIN_STRIDE})
    fitted = fit_logistic(np.concatenate(features), np.concatenate(targets),
                          np.concatenate(weights), recipe["l2"])
    if recipe["identity"]:
        fitted["coefficients"] = [0.0] * len(fitted["coefficients"])
    return {"format": go.FORMAT, "feature_names": list(go.FEATURE_NAMES),
            "means": fitted["means"], "scales": fitted["scales"],
            "coefficients": fitted["coefficients"],
            "feature_spec": {"min_players": int(datasets[games[0]]["min_players"])},
            "decoder": {key: recipe[key] for key in
                        ("switch_cost", "baseline_log_odds", "endpoint_log_odds")},
            "training": {"games": list(games), "reference_sha256": references,
                         "dataset_sha256": json_hash(hashes)},
            "selection": {"recipe": dict(recipe), "training_stride": TRAIN_STRIDE,
                          "equal_match_weight": True, "mirrored_augmentation": "algebraic",
                          "solver": {key: fitted[key] for key in
                                     ("iterations", "objective", "converged")}}}


def validate_codes(codes, blocks):
    """Apply the official table validator and the engine coverage contract."""
    codes = np.asarray(codes)
    if codes.ndim != 1 or not np.isin(codes, (pm.H, pm.A, pm.D, pm.X)).all():
        raise ValueError("invalid prediction codes")
    table = pd.DataFrame({"frame": np.arange(len(codes)), "label": np.asarray(pm.LABELS)[codes]})
    error = pm.validate_submission(table, len(codes))
    if error:
        raise ValueError(error)
    graded = gm.graded_mask(len(codes), blocks)
    if (codes[graded] == pm.X).any() or (codes[~graded] != pm.X).any():
        raise ValueError("prediction violates graded X contract")


def correction_audit(baseline, prediction, eligible):
    """Refuse every edit outside the ownership-only mask."""
    baseline, prediction = np.asarray(baseline), np.asarray(prediction)
    eligible = np.asarray(eligible, bool)
    if baseline.shape != prediction.shape or baseline.shape != eligible.shape:
        raise ValueError("correction array shape mismatch")
    changed = baseline != prediction
    outside = int(np.sum(changed & ~eligible))
    dead = int(np.sum((baseline == pm.D) != (prediction == pm.D)))
    unknown = int(np.sum((baseline == pm.X) != (prediction == pm.X)))
    if outside or dead or unknown:
        raise ValueError("ownership correction changed a protected frame")
    return {"changed_frames": int(changed.sum()), "eligible_frames": int(eligible.sum()),
            "outside_mask_changes": outside, "dead_changes": dead, "x_changes": unknown}


def apply_cell(cell, features, artifact, log_odds=None):
    """Apply a frozen model to cached baseline labels and an audited mask."""
    labels = np.asarray(pm.LABELS)[cell["baseline"]]
    odds = go.predict_log_odds(features, artifact) if log_odds is None else log_odds
    predicted = go.correct(labels, odds, cell["eligible"], cell["engine_blocks"], artifact["decoder"])
    codes = pm.label_codes(predicted)
    correction_audit(cell["baseline"], codes, cell["eligible"])
    validate_codes(codes, cell["engine_blocks"])
    return codes


def choose_recipe(rows):
    """Choose only a positive inner-fold gain, with identity first on exact ties."""
    best = {"recipe": recipes()[0], "gain": 0.0, "clean_min": 0.0, "accepted": True}
    for row in rows:
        if row["clean_min"] < -CLEAN_LIMIT or not np.isfinite(row["gain"]):
            continue
        if row["gain"] > best["gain"]:
            best = row
    return deepcopy(best)


class Trainer:
    """Cache fits and inner estimates by complete training-match identity."""

    def __init__(self, datasets, cells):
        self.datasets, self.cells = datasets, cells
        self.fits, self.evaluations, self.logits = {}, {}, {}
        self.calibration_rows = []

    def model(self, games, recipe):
        key = (tuple(sorted(games)), recipe["l2"])
        if key not in self.fits:
            base_recipe = dict(recipe, identity=False)
            self.fits[key] = fit_model(self.datasets, key[0], base_recipe)
        model = deepcopy(self.fits[key])
        model["decoder"] = {name: recipe[name] for name in
                            ("switch_cost", "baseline_log_odds", "endpoint_log_odds")}
        model["selection"]["recipe"] = dict(recipe)
        if recipe["identity"]:
            model["coefficients"] = [0.0] * len(model["coefficients"])
        return model

    def estimate(self, training, held, recipe):
        check_split(training, held)
        key = (tuple(sorted(training)), held, json_hash(recipe))
        if key in self.evaluations:
            return self.evaluations[key]
        model = self.model(training, recipe)
        logit_key = (tuple(sorted(training)), recipe["l2"], recipe["identity"], held)
        if logit_key not in self.logits:
            self.logits[logit_key] = go.predict_log_odds(self.datasets[held]["features"], model)
        entries = []
        for cell in self.cells:
            if cell["game"] != held or cell["view"] not in ("clean",) + PRIMARY_VIEWS:
                continue
            codes = apply_cell(cell, self.datasets[held]["features"], model, self.logits[logit_key])
            truth = self.datasets[held]["reference"]
            runs = pm.valid_runs(truth)
            score = pm.spell_scores(truth, codes, runs)["mean_f1"]
            base = cell["baseline_spell_f1"]
            row = {"training_games": list(training), "test_game": held,
                   "recipe": recipe, "model_sha256": go.artifact_hash(model),
                   "input_view": cell["input_view"], "view": cell["view"],
                   "seed": cell["seed"], "spell_f1": score, "baseline_spell_f1": base,
                   "delta": score - base, "prediction_sha256": array_sha256(codes),
                   "mask_md5": cell["mask"]["md5"]}
            entries.append(row)
            self.calibration_rows.append(row)
        by_view = {}
        for row in entries:
            by_view.setdefault((row["input_view"], row["view"]), []).append(row["delta"])
        means = {key: float(np.mean(values)) for key, values in by_view.items()}
        if set(means) != set(itertools.product(INPUT_VIEWS, ("clean",) + PRIMARY_VIEWS)):
            raise ValueError("inner validation matrix is incomplete")
        value = {"gain": float(np.mean([delta for (_, view), delta in means.items() if view != "clean"])),
                 "clean_min": min(delta for (_, view), delta in means.items() if view == "clean")}
        self.evaluations[key] = value
        return value

    def select(self, games):
        rows = []
        for recipe in recipes()[1:]:
            estimates = [self.estimate(train, held, recipe) for train, held in grouped_folds(games)]
            rows.append({"recipe": recipe, "gain": float(np.mean([r["gain"] for r in estimates])),
                         "clean_min": min(r["clean_min"] for r in estimates)})
        choice = choose_recipe(rows)
        return choice["recipe"], {"training_games": list(sorted(games)), "choice": choice, "grid": rows}


@contextmanager
def data_root(root, cache):
    """Select an input view without changing repository files."""
    import games
    from engine import data
    previous_root, previous_cache = games.DATA_DIR, data.DEFAULT_CACHE_DIR
    games.DATA_DIR, data.DEFAULT_CACHE_DIR = str(root), str(cache)
    try:
        yield
    finally:
        games.DATA_DIR, data.DEFAULT_CACHE_DIR = previous_root, previous_cache


def prepare_inputs(args):
    """Cache B0 predictions and eligible masks before fitting any model."""
    from engine import data, possession
    from tools.gap_models import gap_dead
    from engine.config import Config
    from engine.submission import write_submission, validate
    out = gb.outside_repo(args.out_dir)
    cfg = Config.default()
    if cfg.DELTA_OUT_LAG != 0 or cfg.DELTA_RESTART_REAPPEAR != 0:
        raise ValueError("this experiment requires the B0 baseline")
    profile = json.loads(Path(args.profile).read_text())
    gb.check_profile(profile)
    generation = engine_fingerprint()
    dead_support_hash = json_hash({name: inspect.getsource(getattr(gap_dead, name))
                                  for name in ("gap_only_mask", "_vector", "_stamp_mask")})
    datasets, cells = {}, []
    for input_view in INPUT_VIEWS:
        root = Path(getattr(args, input_view + "_root")).resolve()
        with data_root(root, out / "parse-cache" / input_view):
            for game in gb.OPEN_GAMES:
                folder = out / "cache" / input_view / game
                folder.mkdir(parents=True, exist_ok=True)
                tactical = data.load_tactical(game)
                n, engine_blocks = tactical["n_frames"], data.graded_blocks(game)
                path = Path(args.references) / (game + ".csv")
                truth = sg.read_codes(str(path), n)
                reference_sha = file_sha256(path)
                blocks = gb.reference_blocks(truth, engine_blocks)
                fps = float(tactical["fps"])
                missing = ~np.isfinite(tactical["bx"]) | ~np.isfinite(tactical["by"])
                tactical_hash = json_hash({key: array_sha256(tactical[key])
                                           for key in ("bx", "by", "px", "py", "ps", "team_of_slot")})
                guarded = possession.engine_tactical(cfg, game)
                features, names = go.feature_matrix(guarded, engine_blocks, data.pitch_m(game), fps,
                                                   min_players=cfg.SHAPE_MIN_PLAYERS)
                if list(names) != list(go.FEATURE_NAMES):
                    raise ValueError("unexpected player feature schema")
                feature_hash = array_sha256(features)
                if game in datasets:
                    if feature_hash != datasets[game]["feature_sha256"]:
                        raise ValueError("ball smoothing changed the player feature matrix")
                else:
                    team_features, team_names = go.team_feature_matrix(
                        guarded, engine_blocks, data.pitch_m(game), fps, min_players=cfg.SHAPE_MIN_PLAYERS)
                    datasets[game] = {"features": features, "reference": truth,
                                      "reference_sha256": reference_sha,
                                      "feature_sha256": feature_hash,
                                      "team_features": team_features, "team_feature_names": team_names,
                                      "min_players": cfg.SHAPE_MIN_PLAYERS}
                    feature_path = out / "cache" / "features" / (game + ".npz")
                    feature_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(feature_path, features=features, team_features=team_features,
                                        reference=truth, missing=missing)
                    write_json(feature_path.with_suffix(".json"), {
                        "game": game, "feature_names": names, "team_feature_names": team_names,
                        "reference_sha256": reference_sha, "feature_sha256": feature_hash,
                        "team_feature_sha256": array_sha256(team_features),
                        "min_players": cfg.SHAPE_MIN_PLAYERS, "fps": fps,
                        "blocks": blocks, "engine_blocks": engine_blocks, "pitch": list(data.pitch_m(game))})
                masks = [("clean", None, {"starts": [], "lengths": [], "md5": None})]
                for view, seed in itertools.product(gb.VIEWS, SEEDS):
                    path = Path(args.mask_root) / game / "masks" / (game + "_" + view + "_" + str(seed) + ".json")
                    mask = gb.frozen_mask(path, missing, blocks, fps, truth, profile, game, view, seed,
                                          require_existing=True)
                    masks.append((view, seed, mask))
                for target in glm.TARGETS:
                    view = glm.live_view(target)
                    path = out / "live_masks" / game / (view + ".json")
                    mask = glm.frozen_live_mask(path, missing, blocks, fps, truth, profile,
                                                game, target, seed=0)
                    masks.append((view, 0, mask))
                for view, seed, mask in masks:
                    name = view + "_" + ("clean" if seed is None else str(seed))
                    cache_path = folder / (name + ".npz")
                    meta_path = folder / (name + ".json")
                    metadata = {"schema": SCHEMA, "engine_md5": generation,
                                "config_hash": cfg.config_hash(), "reference_sha256": reference_sha,
                                "profile_md5": profile["md5"],
                                "feature_sha256": feature_hash, "mask_md5": mask["md5"],
                                "native_missing_sha256": array_sha256(missing), "input_view": input_view,
                                "source_root": str(root), "game": game,
                                "tactical_sha256": tactical_hash, "dead_support_sha256": dead_support_hash,
                                "n_frames": n, "fps": fps, "pitch": list(data.pitch_m(game)),
                                "engine_blocks": [list(block) for block in engine_blocks],
                                "kickoffs": {str(k): int(v) for k, v in possession.match_inputs(game)[3].items()}}
                    if cache_path.exists() and meta_path.exists():
                        saved_meta = json.loads(meta_path.read_text())
                        dead_audit = saved_meta.pop("dead_audit")
                        payload = saved_meta.pop("payload_sha256")
                        if saved_meta != metadata:
                            raise ValueError("prepared input cache provenance changed: " + str(cache_path))
                        with np.load(cache_path, allow_pickle=False) as saved:
                            codes, eligible = saved["baseline"], saved["eligible"]
                            dead_eligible = saved["dead_eligible"]
                        actual = {name: array_sha256(value) for name, value in
                                  (("baseline", codes), ("eligible", eligible), ("dead_eligible", dead_eligible))}
                        if actual != payload:
                            raise ValueError("prepared cache arrays differ from their hashes")
                    else:
                        hidden = gm.hidden_mask(n, mask["starts"], mask["lengths"])
                        with gb.masked_tactical(tactical, hidden, game):
                            inputs = possession.prepare(cfg, game)
                            labels, parts = possession.run(cfg, inputs=inputs, return_parts=True, game=game)
                        combined = dict(inputs, **parts)
                        combined.update(tactical=guarded, pitch=data.pitch_m(game), raw_events=inputs["events"])
                        eligible = go.eligible_mask(labels, combined, cfg, game)
                        dead_eligible, dead_audit = gap_dead.gap_only_mask(
                            inputs, dict(parts, labels=labels), cfg, game=game)
                        codes = pm.label_codes(labels)
                        np.savez_compressed(cache_path, baseline=codes, eligible=eligible,
                                            dead_eligible=dead_eligible)
                        payload = {name: array_sha256(value) for name, value in
                                   (("baseline", codes), ("eligible", eligible), ("dead_eligible", dead_eligible))}
                        write_json(meta_path, dict(metadata, dead_audit=dead_audit, payload_sha256=payload))
                    validate_codes(codes, engine_blocks)
                    csv_path = out / "predictions" / input_view / game / ("B0_" + name + ".csv")
                    csv_path.parent.mkdir(parents=True, exist_ok=True)
                    write_submission(np.asarray(pm.LABELS)[codes], str(csv_path), n_frames=n, blocks=engine_blocks)
                    validate(str(csv_path), rows=n)
                    cells.append({"game": game, "input_view": input_view, "view": view, "seed": seed,
                                  "baseline": codes, "eligible": eligible, "blocks": blocks,
                                  "dead_eligible": dead_eligible, "dead_audit": dead_audit,
                                  "engine_blocks": engine_blocks, "mask": mask, "missing": missing,
                                  "fps": fps, "metadata": metadata, "baseline_csv": str(csv_path),
                                  "baseline_spell_f1": pm.spell_scores(truth, codes, pm.valid_runs(truth))["mean_f1"]})
                print("prepared {} {}: {} paired inputs".format(input_view, game, len(masks)), flush=True)
    return datasets, cells


def aggregate_reports(reports):
    """Report held-out Metrica and DFL transfer separately, with complete pairing."""
    for field in ("engine_md5", "config_hash", "profile_md5"):
        if len({row[field] for row in reports}) != 1:
            raise ValueError("mixed experiment " + field)
    for game in gb.OPEN_GAMES:
        rows = [row for row in reports if row["game"] == game]
        if len({row["reference_sha256"] for row in rows}) != 1:
            raise ValueError("mixed reference versions for " + game)
        candidates = [row for row in rows if row["system"] == "gap_owner"]
        if not candidates or any(not row.get("model_sha256") for row in candidates) or \
                len({row["model_sha256"] for row in candidates}) != 1:
            raise ValueError("mixed model versions for " + game)
        expected_train = sorted(set(METRICA) - {game})
        if any(row["training_games"] != expected_train for row in candidates):
            raise ValueError("outer training split differs for " + game)
    transfer_models = {row["model_sha256"] for row in reports
                       if row["game"].startswith("DFL") and row["system"] == "gap_owner"}
    if len(transfer_models) != 1:
        raise ValueError("DFL transfer must use one final model")
    mask_versions = {}
    for row in reports:
        key = (row["game"], row["view"], row["seed"])
        mask_versions.setdefault(key, set()).add(row["mask_md5"])
    if any(len(values) != 1 for values in mask_versions.values()):
        raise ValueError("raw and matched inputs must share each frozen mask")
    pairs = {}
    for row in reports:
        key = tuple(row[name] for name in ("input_view", "game", "view", "seed"))
        pair = pairs.setdefault(key, {})
        if row["system"] in pair:
            raise ValueError("duplicate experiment report")
        pair[row["system"]] = row
    expected = {(input_view, game, view, seed) for input_view in INPUT_VIEWS for game in gb.OPEN_GAMES
                for view, seeds in (("clean", (None,)), ("U", SEEDS), ("S", SEEDS), ("S_alt", SEEDS))
                for seed in seeds}
    expected.update((input_view, game, view, 0) for input_view in INPUT_VIEWS
                    for game in gb.OPEN_GAMES for view in SENSITIVITY_VIEWS)
    if set(pairs) != expected:
        raise ValueError("experiment matrix is incomplete")
    groups, absolute_groups = {}, {}
    for key, pair in pairs.items():
        if set(pair) != {"B0", "gap_owner"}:
            raise ValueError("missing baseline or candidate")
        left, right = pair["B0"], pair["gap_owner"]
        for field in ("mask_md5", "reference_sha256", "config_hash", "engine_md5", "profile_md5"):
            if left[field] != right[field]:
                raise ValueError("paired experiment provenance differs")
        check_split(right["training_games"], right["game"])
        audit = right["audit"]
        if any(audit[name] for name in ("outside_mask_changes", "dead_changes", "x_changes")):
            raise ValueError("candidate violates its correction contract")
        groups.setdefault(key[:3], []).append(right["metrics"]["score"]["spell_f1_mean"]
                                              - left["metrics"]["score"]["spell_f1_mean"])
        absolute_groups.setdefault(key[:3], []).append((left["metrics"]["score"]["spell_f1_mean"],
                                                        right["metrics"]["score"]["spell_f1_mean"]))
    means = [{"input_view": key[0], "game": key[1], "view": key[2],
              "delta": float(np.mean(values)), "replicates": len(values),
              "baseline_spell_f1": float(np.mean(absolute_groups[key], axis=0)[0]),
              "candidate_spell_f1": float(np.mean(absolute_groups[key], axis=0)[1])}
             for key, values in sorted(groups.items())]
    intervals = []
    for input_view, view, provider in itertools.product(INPUT_VIEWS, ("clean",) + ALL_VIEWS,
                                                        ("METRICA", "DFL", "all")):
        chosen = [r for r in means if r["input_view"] == input_view and r["view"] == view
                  and (provider == "all" or r["game"].startswith(provider))]
        values = [r["delta"] for r in chosen]
        intervals.append(dict(gb.match_interval(values), input_view=input_view, view=view, provider=provider,
                              baseline_spell_f1=float(np.mean([r["baseline_spell_f1"] for r in chosen])),
                              candidate_spell_f1=float(np.mean([r["candidate_spell_f1"] for r in chosen])),
                              interpretation={"METRICA": "held-out matches", "DFL": "provider transfer",
                                              "all": "mixed held-out and provider transfer"}[provider]))
    return {"schema": SCHEMA, "cells": len(reports), "complete": True, "match_means": means,
            "paired_match_intervals": intervals,
            "owner_gate_nonnegative_metrica": all(r["delta"] >= 0 for r in means
                                                   if r["game"].startswith("METRICA") and r["view"] in PRIMARY_VIEWS),
            "positive_primary_metrica_intervals": all(r["low"] > 0 for r in intervals
                                                       if r["provider"] == "METRICA" and r["view"] in PRIMARY_VIEWS),
            "nonnegative_metrica_sensitivity_means": all(r["mean"] >= 0 for r in intervals
                if r["provider"] == "METRICA" and r["view"] in ("S_alt",) + SENSITIVITY_VIEWS),
            "historical_eight_match_primary_gate": all(r["low"] > 0 for r in intervals
                if r["provider"] == "all" and r["view"] in PRIMARY_VIEWS),
            "clean_provider_nonregression": all(r["mean"] >= -CLEAN_LIMIT for r in intervals
                if r["view"] == "clean" and r["provider"] != "all"),
            "warning": "Three Metrica matches and synthetic gaps do not establish competition-match accuracy."}


def ledger_row(report, revision, date):
    """Keep external results outside the public-score columns."""
    score = report["metrics"]["score"]
    name = "p10c_{}_{}_{}_{}_{}".format(report["input_view"], report["game"], report["system"],
                                        report["view"], report["seed"] if report["seed"] is not None else "clean")
    note = "{} external reference; spellF1 {:.6f}; macroF1 {:.6f}; model {}; mask {}; prediction md5 {}".format(
        report["role"], score["spell_f1_mean"], score["macro_f1"], report["model_sha256"] or "none",
        report["mask_md5"] or "clean", report["prediction_md5"])
    return "| {} | {} | {} | {} | — | — | — | — | — | — | — | info (P10-C) | {} |".format(
        date, name, revision, report["config_hash"], note)


def run_experiment(args):
    """Run the predeclared nested experiment and write reviewable local artifacts."""
    from engine.submission import write_submission, validate
    out = gb.outside_repo(args.out_dir)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    source = {"revision": revision, "runner_sha256": file_sha256(__file__),
              "engine_fingerprint": engine_fingerprint(), "profile_sha256": file_sha256(args.profile)}
    write_json(out / "execution.json", source)
    datasets, cells = prepare_inputs(args)
    if args.prepare_only:
        return
    trainer = Trainer(datasets, cells)
    models, choices = {}, []
    for training, held in grouped_folds(METRICA):
        recipe, selection = trainer.select(training)
        model = trainer.model(training, recipe)
        model["selection"]["held_out_game"] = held
        models[held] = model
        choices.append(dict(selection, held_out_game=held))
        write_json(out / "models" / (held + ".json"), model)
        print("selected held-out {}: {}".format(held, recipe), flush=True)
    recipe, selection = trainer.select(METRICA)
    final = trainer.model(METRICA, recipe)
    final["selection"]["role"] = "deployment candidate; not measured on its training matches"
    write_json(out / "models" / "final.json", final)
    choices.append(dict(selection, held_out_game="DFL transfer and target deployment"))
    write_json(out / "selection.json", {"folds": choices, "calibration_cells": len(trainer.calibration_rows)})
    write_json(out / "calibration_rows.json", trainer.calibration_rows)
    for game in gb.OPEN_GAMES:
        if game not in models:
            models[game] = final
    reports, clean = [], {}
    tuning_ledger = []
    for index, row in enumerate(trainer.calibration_rows):
        note = ("inner calibration only; train {}; test {} {} {} seed {}; spellF1 {:.6f}; "
                "delta {:+.6f}; model {}; prediction array sha256 {}").format(
            ",".join(row["training_games"]), row["test_game"], row["input_view"], row["view"],
            row["seed"], row["spell_f1"], row["delta"], row["model_sha256"], row["prediction_sha256"])
        tuning_ledger.append("| {} | p10c_inner_{:04d} | {} | B0 | — | — | — | — | — | — | — | info (P10-C calibration) | {} |".format(
            args.date, index, revision[:7], note))
    (out / "calibration_ledger_rows.md").write_text("\n".join(tuning_ledger) + "\n")
    for cell in cells:
        game = cell["game"]
        model = models[game]
        check_split(model["training"]["games"], game)
        candidate = apply_cell(cell, datasets[game]["features"], model)
        for system, prediction in (("B0", cell["baseline"]), ("gap_owner", candidate)):
            key = (cell["input_view"], game, system)
            if cell["view"] == "clean":
                clean[key] = prediction
            name = "{}_{}_{}".format(system, cell["view"], cell["seed"] if cell["seed"] is not None else "clean")
            csv_path = out / "predictions" / cell["input_view"] / game / (name + ".csv")
            write_submission(np.asarray(pm.LABELS)[prediction], str(csv_path), n_frames=len(prediction),
                             blocks=cell["engine_blocks"])
            validate(str(csv_path), rows=len(prediction))
            metrics = gb.score_cell(datasets[game]["reference"], prediction, clean[key], cell["mask"],
                                    cell["missing"], cell["blocks"], cell["fps"], args.draws)
            report = {"schema": SCHEMA, "game": game, "input_view": cell["input_view"],
                      "view": cell["view"], "seed": cell["seed"], "system": system,
                      "role": "held-out" if game in METRICA else "transfer",
                      "mask_md5": cell["mask"]["md5"], "reference_sha256": datasets[game]["reference_sha256"],
                      "profile_md5": cell["metadata"]["profile_md5"],
                      "config_hash": cell["metadata"]["config_hash"], "engine_md5": cell["metadata"]["engine_md5"],
                      "model_sha256": go.artifact_hash(model) if system == "gap_owner" else None,
                      "training_games": model["training"]["games"] if system == "gap_owner" else [],
                      "prediction_md5": gm.file_md5(str(csv_path)), "prediction_sha256": file_sha256(csv_path),
                      "audit": correction_audit(cell["baseline"], prediction, cell["eligible"]), "metrics": metrics}
            reports.append(report)
            write_json(out / "reports" / cell["input_view"] / game / (name + ".json"), report)
        print("scored {} {} {} {}".format(cell["input_view"], game, cell["view"], cell["seed"]), flush=True)
    summary = aggregate_reports(reports)
    summary["revision"] = revision
    summary["execution"] = source
    summary["selection"] = choices
    write_json(out / "summary.json", summary)
    (out / "ledger_rows.md").write_text("\n".join(ledger_row(row, revision[:7], args.date) for row in reports) + "\n")
    print("complete: {} validated reports".format(len(reports)), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for name in ("raw-root", "matched-root", "references", "mask-root", "profile", "out-dir"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--draws", type=int, default=1000)
    run.add_argument("--date", default="2026-11-02")
    run.add_argument("--prepare-only", action="store_true")
    summarize = sub.add_parser("summarize")
    summarize.add_argument("out_dir")
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.draws < 1:
            raise ValueError("bootstrap draws must be positive")
        run_experiment(args)
    else:
        out = gb.outside_repo(args.out_dir)
        reports = [json.loads(path.read_text()) for path in sorted((out / "reports").rglob("*.json"))]
        summary = aggregate_reports(reports)
        if (out / "execution.json").exists():
            summary["execution"] = json.loads((out / "execution.json").read_text())
            summary["revision"] = summary["execution"]["revision"]
        if (out / "selection.json").exists():
            summary["selection"] = json.loads((out / "selection.json").read_text())["folds"]
        write_json(out / "summary.json", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
