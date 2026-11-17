"""Evaluate selective false-dead correction with eight whole-match outer folds.

Read the frozen P10-C preparation cache; do not recompute baseline detections or
alter masks. Every outer match remains outside training and inner selection.
Models, predictions, interval coordinates and reports stay outside this checkout.
"""

import argparse
from copy import deepcopy
import inspect
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import gap_owner_experiment as shared
from tools import gap_benchmark as gb
from tools import gap_live_masks as glm
from tools import gapmask as gm
from tools import score_game as sg
from tools.gap_models import gap_dead as gd
from tools.gap_models import gap_owner as go

pm = gb.pm
INPUT_VIEWS = shared.INPUT_VIEWS
PRIMARY_VIEWS = shared.PRIMARY_VIEWS
SENSITIVITY_VIEWS = shared.SENSITIVITY_VIEWS
ALL_VIEWS = shared.ALL_VIEWS
SEEDS = shared.SEEDS
TRAIN_STRIDE = 5
CLEAN_LIMIT = 0.01
SCHEMA = 1
L2_GRID = (0.1, 1.0)
THRESHOLD_GRID = (0.8, 0.95)


def recipes():
    """The predeclared grid, with conservative settings first for exact ties."""
    return [{"identity": True, "l2": 1.0, "threshold": 0.95}] + [
        {"identity": False, "l2": ridge, "threshold": threshold}
        for ridge, threshold in itertools.product(reversed(L2_GRID), reversed(THRESHOLD_GRID))]


def grouped_folds(games):
    """Leave every copy of a complete open match outside training."""
    games = tuple(sorted(games))
    if len(games) < 2 or len(games) != len(set(games)) or not set(games).issubset(gb.OPEN_GAMES):
        raise ValueError("folds require distinct open reference matches")
    return [(tuple(g for g in games if g != held), held) for held in games]


def check_split(training, held):
    if not training or len(training) != len(set(training)) or not set(training).issubset(gb.OPEN_GAMES):
        raise ValueError("training accepts only distinct open reference matches")
    if held not in gb.OPEN_GAMES or held in training:
        raise ValueError("the complete held-out match must remain outside training")


def fit_logistic(features, labels, weights, l2, max_iterations=50):
    """Fit weighted logistic loss with a free intercept and training-only scaling."""
    x, y, weight = [np.asarray(v, dtype=np.float64) for v in (features, labels, weights)]
    if x.ndim != 2 or not len(x) or y.shape != (len(x),) or weight.shape != y.shape:
        raise ValueError("invalid training shapes")
    if not all(np.isfinite(v).all() for v in (x, y, weight)) or not np.isin(y, (0, 1)).all():
        raise ValueError("training features and binary targets must be finite")
    if (weight <= 0).any() or not np.isfinite(l2) or l2 <= 0 or max_iterations < 1:
        raise ValueError("weights, regularization and iteration count must be positive")
    if len(np.unique(y)) != 2:
        raise ValueError("state training requires both live and dead targets")
    weight = weight / weight.sum()
    means = weight @ x
    centered = x - means
    scales = np.sqrt(weight @ (centered * centered))
    active = scales > 1e-12
    scales[~active] = 1.0
    # A training-constant column has zero centered value and a ridge optimum of zero.
    z = np.column_stack((centered[:, active] / scales[active], np.ones(len(x))))
    penalty = np.full(z.shape[1], float(l2))
    penalty[-1] = 0.0
    coef = np.zeros(z.shape[1])
    prior = float(weight @ y)
    coef[-1] = math.log(prior / (1.0 - prior))

    def objective(candidate):
        margin = z @ candidate
        return float(weight @ (np.logaddexp(0, margin) - y * margin)
                     + 0.5 * ((penalty * candidate) @ candidate))

    converged = False
    for iteration in range(max_iterations):
        margin = z @ coef
        probability = np.exp(-np.logaddexp(0, -margin))
        gradient = z.T @ (weight * (probability - y)) + penalty * coef
        if np.max(np.abs(gradient)) < 1e-8:
            converged = True
            break
        curvature = weight * probability * (1.0 - probability)
        hessian = z.T @ (curvature[:, None] * z) + np.diag(penalty)
        direction = np.linalg.solve(hessian, gradient)
        before, step = objective(coef), 1.0
        while step > 2.0 ** -20:
            candidate = coef - step * direction
            if objective(candidate) <= before:
                coef = candidate
                break
            step *= 0.5
        else:
            raise ValueError("state logistic objective did not decrease")
    if not converged:
        probability = np.exp(-np.logaddexp(0, -(z @ coef)))
        gradient = z.T @ (weight * (probability - y)) + penalty * coef
        converged = np.max(np.abs(gradient)) < 1e-8
    if not converged or not np.isfinite(coef).all():
        raise ValueError("state logistic fit did not converge")
    full = np.zeros(x.shape[1])
    full[active] = coef[:-1]
    return {"means": means.tolist(), "scales": scales.tolist(), "coefficients": full.tolist(),
            "intercept": float(coef[-1]), "iterations": iteration + 1,
            "objective": objective(coef), "converged": True}


def training_data(datasets, games):
    """Sample clean graded rows and give each training match equal total weight."""
    games = tuple(sorted(games))
    if not games or len(set(games)) != len(games) or not set(games).issubset(gb.OPEN_GAMES):
        raise ValueError("invalid state training matches")
    if len({datasets[g]["min_players"] for g in games}) != 1:
        raise ValueError("training player-feature settings differ")
    features, targets, weights, hashes, references = [], [], [], [], {}
    for game in games:
        dataset = datasets[game]
        truth = np.asarray(dataset["reference"], dtype=np.int8)
        if not np.isin(truth, (pm.H, pm.A, pm.D, pm.X)).all():
            raise ValueError("invalid reference target")
        index = np.flatnonzero(truth != pm.X)[::TRAIN_STRIDE]
        if not len(index):
            raise ValueError("training match has no graded rows")
        selected = np.asarray(dataset["features"][index], dtype=np.float64)
        features.append(selected)
        targets.append((truth[index] < pm.D).astype(float))
        weights.append(np.full(len(index), 1.0 / (len(games) * len(index))))
        references[game] = dataset["reference_sha256"]
        hashes.append({"game": game, "features": shared.array_sha256(selected),
                       "targets": shared.array_sha256(truth[index]), "stride": TRAIN_STRIDE})
    return (np.concatenate(features), np.concatenate(targets), np.concatenate(weights),
            {"games": list(games), "reference_sha256": references,
             "dataset_sha256": shared.json_hash(hashes)})


def fit_model(datasets, training, recipe):
    x, y, weights, provenance = training_data(datasets, training)
    fitted = fit_logistic(x, y, weights, recipe["l2"])
    model = {"format": gd.FORMAT, "feature_names": list(gd.FEATURE_NAMES),
             "feature_spec": {"min_players": int(datasets[training[0]]["min_players"])},
             "means": fitted["means"], "scales": fitted["scales"],
             "coefficients": fitted["coefficients"], "intercept": fitted["intercept"],
             "threshold": recipe["threshold"], "enabled": not recipe["identity"],
             "training": provenance,
             "selection": {"recipe": dict(recipe), "training_stride": TRAIN_STRIDE,
                           "equal_match_weight": True, "solver": {key: fitted[key] for key in
                           ("iterations", "objective", "converged")}}}
    return gd.validate_artifact(model)


def correction_audit(baseline, prediction, eligible, blocks):
    """Verify complete agreed-owner restoration and every protected original label."""
    baseline, prediction, eligible = np.asarray(baseline), np.asarray(prediction), np.asarray(eligible, bool)
    if baseline.ndim != 1 or baseline.shape != prediction.shape or baseline.shape != eligible.shape:
        raise ValueError("correction shapes differ")
    if not np.isin(prediction, (pm.H, pm.A, pm.D, pm.X)).all():
        raise ValueError("invalid correction labels")
    changed = baseline != prediction
    outside = int((changed & ~eligible).sum())
    original_live = int((changed & (baseline < pm.D)).sum())
    unknown = int((changed & (baseline == pm.X)).sum())
    permitted = np.zeros(len(baseline), bool)
    removed = 0
    runs = gd.actionable_runs(np.asarray(pm.LABELS)[baseline], eligible, blocks)
    for start, stop, team in runs:
        permitted[start:stop] = True
        if changed[start:stop].any():
            if not changed[start:stop].all() or not (prediction[start:stop] == pm.LABELS.index(team)).all():
                raise ValueError("state correction fragmented a run or changed its agreed owner")
            removed += 1
    if outside or original_live or unknown or (changed & ~permitted).any():
        raise ValueError("state correction changed a protected frame")
    return {"changed_frames": int(changed.sum()), "d_to_h_frames": int((changed & (prediction == pm.H)).sum()),
            "d_to_a_frames": int((changed & (prediction == pm.A)).sum()), "removed_runs": removed,
            "eligible_frames": int(eligible.sum()), "actionable_runs": len(runs),
            "actionable_frames": int(permitted.sum()), "outside_mask_changes": outside,
            "original_live_changes": original_live, "x_changes": unknown,
            "protected_frames_unchanged": int((~eligible & ~changed).sum())}


def apply_cell(cell, features, model, logits=None):
    labels = np.asarray(pm.LABELS)[cell["baseline"]]
    if not model["enabled"]:
        prediction = cell["baseline"].copy()
    else:
        logits = gd.predict_log_odds(features, model) if logits is None else logits
        prediction = pm.label_codes(gd.correct(labels, logits, cell["dead_eligible"],
                                               cell["engine_blocks"], model["threshold"]))
    correction_audit(cell["baseline"], prediction, cell["dead_eligible"], cell["engine_blocks"])
    shared.validate_codes(prediction, cell["engine_blocks"])
    return prediction


def choose_recipe(rows):
    """Select positive equal-match gain subject to each clean provider/input mean."""
    best = {"recipe": recipes()[0], "gain": 0.0, "clean_provider_means": {}}
    for recipe in recipes()[1:]:
        matching = [row for row in rows if row["recipe"] == recipe]
        if len(matching) != 1:
            raise ValueError("selection requires the complete predeclared grid")
        row = matching[0]
        clean = row["clean_provider_means"]
        if not clean or not np.isfinite(row["gain"]) or any(
                not np.isfinite(value) or value < -CLEAN_LIMIT for value in clean.values()):
            continue
        if row["gain"] > best["gain"]:
            best = row
    return deepcopy(best)


class Trainer:
    """Cache training-set fits and matched inner estimates without outer leakage."""

    def __init__(self, datasets, cells, out=None):
        self.datasets, self.cells = datasets, cells
        self.fits, self.estimates, self.logits, self.scores = {}, {}, {}, {}
        self.calibration_rows = []
        self.out = gb.outside_repo(out) if out is not None else None
        self.fit_source = shared.json_hash({"runner": shared.file_sha256(__file__),
                                           "model": shared.file_sha256(ROOT / "tools/gap_models/gap_dead.py")})

    def model(self, training, recipe):
        training = tuple(sorted(training))
        key = (training, recipe["l2"])
        if key not in self.fits:
            cache_key = shared.json_hash({"games": training, "l2": recipe["l2"], "source": self.fit_source,
                "features": {g: self.datasets[g]["feature_sha256"] for g in training},
                "references": {g: self.datasets[g]["reference_sha256"] for g in training}})
            path = None if self.out is None else self.out / "fits" / (cache_key + ".json")
            if path is not None and path.exists():
                model = gd.load_artifact(path)
                if model["training"]["games"] != list(training) or any(
                        model["training"]["reference_sha256"][g] != self.datasets[g]["reference_sha256"]
                        for g in training):
                    raise ValueError("saved fit has different training provenance")
            else:
                model = fit_model(self.datasets, training, dict(recipe, identity=False))
                if path is not None:
                    shared.write_json(path, model)
            self.fits[key] = model
            print("fitted {} matches, L2 {}".format(len(training), recipe["l2"]), flush=True)
        model = deepcopy(self.fits[key])
        model["threshold"], model["enabled"] = recipe["threshold"], not recipe["identity"]
        model["selection"]["recipe"] = dict(recipe)
        return model

    def estimate(self, training, held, recipe):
        check_split(training, held)
        key = (tuple(sorted(training)), held, shared.json_hash(recipe))
        if key in self.estimates:
            return self.estimates[key]
        model = self.model(training, recipe)
        logit_key = (tuple(sorted(training)), recipe["l2"], held)
        if logit_key not in self.logits:
            self.logits[logit_key] = gd.predict_log_odds(self.datasets[held]["features"], model)
        entries = []
        for cell in self.cells:
            if cell["game"] != held or cell["view"] not in ("clean",) + PRIMARY_VIEWS:
                continue
            codes = apply_cell(cell, self.datasets[held]["features"], model, self.logits[logit_key])
            prediction_hash = shared.array_sha256(codes)
            score_key = held, prediction_hash
            base = cell["baseline_spell_f1"]
            if np.array_equal(codes, cell["baseline"]):
                score = base
            else:
                if score_key not in self.scores:
                    truth = self.datasets[held]["reference"]
                    self.scores[score_key] = pm.spell_scores(truth, codes, pm.valid_runs(truth))["mean_f1"]
                score = self.scores[score_key]
            row = {"training_games": list(training), "test_game": held, "recipe": recipe,
                   "model_sha256": gd.artifact_hash(model), "input_view": cell["input_view"],
                   "view": cell["view"], "seed": cell["seed"], "spell_f1": score,
                   "baseline_spell_f1": base, "delta": score - base,
                   "prediction_sha256": prediction_hash, "mask_md5": cell["mask"]["md5"]}
            entries.append(row)
            self.calibration_rows.append(row)
        groups = {}
        for row in entries:
            groups.setdefault((row["input_view"], row["view"]), []).append(row["delta"])
        if set(groups) != set(itertools.product(INPUT_VIEWS, ("clean",) + PRIMARY_VIEWS)):
            raise ValueError("inner validation matrix is incomplete")
        means = {key: float(np.mean(values)) for key, values in groups.items()}
        estimate = {"game": held, "gain": float(np.mean([v for (_, view), v in means.items() if view != "clean"])),
                    "clean": {iv: means[(iv, "clean")] for iv in INPUT_VIEWS}}
        self.estimates[key] = estimate
        return estimate

    def select(self, games):
        rows = []
        for recipe in recipes()[1:]:
            estimates = [self.estimate(training, held, recipe) for training, held in grouped_folds(games)]
            clean = {}
            for provider, input_view in itertools.product(("METRICA", "DFL"), INPUT_VIEWS):
                values = [r["clean"][input_view] for r in estimates if r["game"].startswith(provider)]
                if values:
                    clean[provider + "/" + input_view] = float(np.mean(values))
            rows.append({"recipe": recipe, "gain": float(np.mean([e["gain"] for e in estimates])),
                         "clean_provider_means": clean})
        choice = choose_recipe(rows)
        return choice["recipe"], {"training_games": list(sorted(games)), "choice": choice, "grid": rows}


def dead_support_hash():
    return shared.json_hash({name: inspect.getsource(getattr(gd, name))
                             for name in ("gap_only_mask", "_vector", "_stamp_mask")})


def load_prepared(args):
    """Verify and read the completed shared cache without changing any of its files."""
    from engine.config import Config
    prepared, out = gb.outside_repo(args.prepared_root), gb.outside_repo(args.out_dir)
    profile = json.loads(Path(args.profile).read_text())
    gb.check_profile(profile)
    generation, support = shared.engine_fingerprint(), dead_support_hash()
    cfg = Config.default()
    datasets, cells = {}, []
    for game in gb.OPEN_GAMES:
        feature_path = prepared / "cache/features" / (game + ".npz")
        meta = json.loads(feature_path.with_suffix(".json").read_text())
        with np.load(feature_path, allow_pickle=False) as source:
            teams, truth, missing = source["team_features"], source["reference"], source["missing"]
            if shared.array_sha256(source["features"]) != meta["feature_sha256"]:
                raise ValueError("owner feature cache checksum mismatch")
        reference_sha = shared.file_sha256(Path(args.references) / (game + ".csv"))
        actual_truth = sg.read_codes(str(Path(args.references) / (game + ".csv")), len(truth))
        if reference_sha != meta["reference_sha256"] or not np.array_equal(actual_truth, truth):
            raise ValueError("prepared reference has changed")
        if list(meta["team_feature_names"]) != list(go.BASE_FEATURES) or meta["game"] != game:
            raise ValueError("prepared team schema differs")
        team_sha = shared.array_sha256(teams)
        if team_sha != meta["team_feature_sha256"]:
            raise ValueError("team feature cache checksum mismatch")
        features, names = gd.features_from_teams(teams, meta["engine_blocks"], meta["fps"])
        feature_sha = shared.array_sha256(features)
        feature_out = out / "features" / (game + ".npy")
        feature_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(feature_out, features, allow_pickle=False)
        del features, teams
        datasets[game] = {"features": np.load(feature_out, mmap_mode="r", allow_pickle=False),
                          "reference": truth, "reference_sha256": reference_sha,
                          "feature_sha256": feature_sha, "min_players": meta["min_players"]}
        shared.write_json(feature_out.with_suffix(".json"), dict(meta, state_feature_names=names,
            state_feature_sha256=feature_sha, team_feature_sha256=team_sha,
            source_npz_sha256=shared.file_sha256(feature_path)))
        for input_view in INPUT_VIEWS:
            specs = [("clean", None)] + list(itertools.product(gb.VIEWS, SEEDS)) + [(v, 0) for v in SENSITIVITY_VIEWS]
            for view, seed in specs:
                name = view + "_" + ("clean" if seed is None else str(seed))
                path = prepared / "cache" / input_view / game / (name + ".npz")
                metadata = json.loads(path.with_suffix(".json").read_text())
                expected = {"engine_md5": generation, "dead_support_sha256": support,
                            "config_hash": cfg.config_hash(), "reference_sha256": reference_sha,
                            "profile_md5": profile["md5"], "feature_sha256": meta["feature_sha256"],
                            "native_missing_sha256": shared.array_sha256(missing), "game": game,
                            "input_view": input_view, "n_frames": len(truth), "fps": meta["fps"],
                            "engine_blocks": meta["engine_blocks"], "pitch": meta["pitch"]}
                if any(metadata.get(k) != v for k, v in expected.items()):
                    raise ValueError("prepared source provenance changed: " + str(path))
                with np.load(path, allow_pickle=False) as saved:
                    payload = {key: saved[key] for key in ("baseline", "eligible", "dead_eligible")}
                if {k: shared.array_sha256(v) for k, v in payload.items()} != metadata["payload_sha256"]:
                    raise ValueError("prepared prediction or eligibility checksum mismatch")
                if view == "clean":
                    mask = {"starts": [], "lengths": [], "md5": None}
                elif view in gb.VIEWS:
                    mask_path = Path(args.mask_root) / game / "masks" / (game + "_" + view + "_" + str(seed) + ".json")
                    mask = gb.frozen_mask(mask_path, missing, meta["blocks"], meta["fps"], truth, profile,
                                          game, view, seed, require_existing=True)
                else:
                    mask_path = prepared / "live_masks" / game / (view + ".json")
                    if not mask_path.exists():
                        raise ValueError("shared live mask has not been prepared")
                    target = next(t for t in glm.TARGETS if glm.live_view(t) == view)
                    mask = glm.frozen_live_mask(mask_path, missing, meta["blocks"], meta["fps"], truth,
                                               profile, game, target, seed=0)
                if metadata["mask_md5"] != mask["md5"]:
                    raise ValueError("prepared mask version differs")
                baseline, eligible = payload["baseline"], payload["dead_eligible"]
                shared.validate_codes(baseline, meta["engine_blocks"])
                if eligible.dtype != bool or eligible.shape != baseline.shape or (eligible & (baseline != pm.D)).any():
                    raise ValueError("prepared dead mask is not a subset of baseline D")
                audit = correction_audit(baseline, baseline, eligible, meta["engine_blocks"])
                if audit["eligible_frames"] != metadata["dead_audit"]["eligible_frames"]:
                    raise ValueError("prepared support audit differs from its mask")
                cells.append({"game": game, "input_view": input_view, "view": view, "seed": seed,
                    "baseline": baseline, "dead_eligible": eligible, "blocks": meta["blocks"],
                    "engine_blocks": meta["engine_blocks"], "mask": mask, "missing": missing,
                    "fps": meta["fps"], "metadata": metadata, "exposure": audit,
                    "baseline_spell_f1": pm.spell_scores(truth, baseline, pm.valid_runs(truth))["mean_f1"]})
        print("loaded state inputs {}".format(game), flush=True)
    return datasets, cells


def aggregate_reports(reports, draws=1000):
    """Require a complete, wholly held-out paired matrix before acceptance checks."""
    fields = ("engine_md5", "config_hash", "profile_md5", "dead_support_sha256",
              "state_source_sha256", "runner_sha256")
    if not reports or any(len({r[k] for r in reports}) != 1 for k in fields):
        raise ValueError("mixed experiment provenance")
    pairs = {}
    masks = {}
    for row in reports:
        key = tuple(row[k] for k in ("input_view", "game", "view", "seed"))
        pair = pairs.setdefault(key, {})
        if row["system"] in pair:
            raise ValueError("duplicate report")
        pair[row["system"]] = row
        masks.setdefault(key[1:], set()).add(row["mask_md5"])
    expected = {(iv, g, v, s) for iv in INPUT_VIEWS for g in gb.OPEN_GAMES
                for v, seeds in (("clean", (None,)), ("U", SEEDS), ("S", SEEDS), ("S_alt", SEEDS)) for s in seeds}
    expected.update((iv, g, v, 0) for iv in INPUT_VIEWS for g in gb.OPEN_GAMES for v in SENSITIVITY_VIEWS)
    if set(pairs) != expected or any(len(v) != 1 for v in masks.values()):
        raise ValueError("incomplete matrix or differing raw/matched masks")
    for game in gb.OPEN_GAMES:
        rows = [r for r in reports if r["game"] == game]
        candidates = [r for r in rows if r["system"] == "gap_dead"]
        if len({r["reference_sha256"] for r in rows}) != 1 or len({r["feature_sha256"] for r in rows}) != 1:
            raise ValueError("mixed game reference or features")
        if not candidates or len({r["model_sha256"] for r in candidates}) != 1 or not candidates[0]["model_sha256"]:
            raise ValueError("mixed outer model")
        if any(r["training_games"] != sorted(set(gb.OPEN_GAMES) - {game}) for r in candidates):
            raise ValueError("outer training differs from leave-one-match-out")
    groups, false_d, totals = {}, {}, {"changed_frames": 0, "removed_runs": 0}
    for key, pair in pairs.items():
        if set(pair) != {"B0", "gap_dead"}:
            raise ValueError("unpaired report")
        base, candidate = pair["B0"], pair["gap_dead"]
        if any(base[k] != candidate[k] for k in fields + ("mask_md5", "reference_sha256", "feature_sha256")):
            raise ValueError("paired provenance differs")
        check_split(candidate["training_games"], candidate["game"])
        audit = candidate["audit"]
        if any(audit[k] for k in ("outside_mask_changes", "original_live_changes", "x_changes")):
            raise ValueError("candidate changed protected labels")
        groups.setdefault(key[:3], []).append(candidate["metrics"]["score"]["spell_f1_mean"] - base["metrics"]["score"]["spell_f1_mean"])
        a, b = base["metrics"]["masked"]["false_d_rate"], candidate["metrics"]["masked"]["false_d_rate"]
        if a is not None and b is not None:
            false_d.setdefault(key[:3], []).append(b - a)
        for field in totals:
            totals[field] += audit[field]
    means = [{"input_view": k[0], "game": k[1], "view": k[2], "delta": float(np.mean(v)), "replicates": len(v)}
             for k, v in sorted(groups.items())]
    false_means = [{"input_view": k[0], "game": k[1], "view": k[2], "delta": float(np.mean(v)), "replicates": len(v)}
                   for k, v in sorted(false_d.items())]
    intervals = []
    for iv, view, provider in itertools.product(INPUT_VIEWS, ("clean",) + ALL_VIEWS, ("METRICA", "DFL", "all")):
        values = [r["delta"] for r in means if r["input_view"] == iv and r["view"] == view
                  and (provider == "all" or r["game"].startswith(provider))]
        intervals.append(dict(gb.match_interval(values, draws), input_view=iv, view=view, provider=provider))
    reduction = []
    for iv, view in itertools.product(INPUT_VIEWS, PRIMARY_VIEWS):
        values = [r["delta"] for r in false_means if r["input_view"] == iv and r["view"] == view]
        reduction.append({"input_view": iv, "view": view, "matches": len(values),
                          "mean_false_d_rate_delta": float(np.mean(values)) if values else None})
    gates = {
        "positive_primary_intervals": all(r["low"] > 0 for r in intervals if r["provider"] == "all" and r["view"] in PRIMARY_VIEWS),
        "nonnegative_metrica_primary_match_means": all(r["delta"] >= 0 for r in means if r["game"].startswith("METRICA") and r["view"] in PRIMARY_VIEWS),
        "clean_provider_nonregression": all(r["mean"] >= -CLEAN_LIMIT for r in intervals if r["view"] == "clean" and r["provider"] != "all"),
        "nonnegative_provider_sensitivity_means": all(r["mean"] >= 0 for r in intervals if r["view"] in ("S_alt",) + SENSITIVITY_VIEWS and r["provider"] != "all"),
        "false_d_decreases_primary": all(r["matches"] == len(gb.OPEN_GAMES) and r["mean_false_d_rate_delta"] < 0 for r in reduction)}
    return {"schema": SCHEMA, "cells": len(reports), "complete": True, "all_outer_matches_held_out": True,
            "match_means": means, "paired_match_intervals": intervals, "false_d_match_means": false_means,
            "primary_false_d_reduction": reduction, "correction_totals": totals, "gates": gates,
            "accepted": all(gates.values()), "method": "Average seeds within matches, then paired whole-match 5th/95th percentiles.",
            "warning": "Synthetic missingness and two external providers do not establish competition-match accuracy."}


def run_experiment(args):
    from engine.submission import write_submission, validate
    out = gb.outside_repo(args.out_dir)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    source = {"revision": revision, "runner_sha256": shared.file_sha256(__file__),
              "model_source_sha256": shared.file_sha256(ROOT / "tools/gap_models/gap_dead.py"),
              "engine_fingerprint": shared.engine_fingerprint(), "dead_support_sha256": dead_support_hash(),
              "profile_sha256": shared.file_sha256(args.profile), "prepared_root": str(Path(args.prepared_root).resolve())}
    shared.write_json(out / "execution.json", source)
    datasets, cells = load_prepared(args)
    shared.write_json(out / "exposure.json", [{key: cell[key] for key in
        ("game", "input_view", "view", "seed", "exposure")} for cell in cells])
    if args.exposure_only:
        return
    trainer = Trainer(datasets, cells, out)
    models, choices = {}, []
    for training, held in grouped_folds(gb.OPEN_GAMES):
        recipe, selection = trainer.select(training)
        model = trainer.model(training, recipe)
        model["selection"]["held_out_game"] = held
        models[held] = model
        choices.append(dict(selection, held_out_game=held))
        shared.write_json(out / "models" / (held + ".json"), model)
        shared.write_json(out / "selection_progress.json", choices)
        print("selected held-out {}: {}".format(held, recipe), flush=True)
    recipe, selection = trainer.select(gb.OPEN_GAMES)
    final = trainer.model(gb.OPEN_GAMES, recipe)
    final["selection"]["role"] = "deployment candidate; never evaluated on its training matches"
    shared.write_json(out / "models/final.json", final)
    choices.append(dict(selection, held_out_game="target deployment only"))
    shared.write_json(out / "selection.json", {"folds": choices, "calibration_cells": len(trainer.calibration_rows)})
    shared.write_json(out / "calibration_rows.json", trainer.calibration_rows)
    tuning = []
    for index, row in enumerate(trainer.calibration_rows):
        note = ("inner calibration only; train {}; test {} {} {} seed {}; spellF1 {:.6f}; delta {:+.6f}; "
                "model {}; prediction array sha256 {}").format(
            ",".join(row["training_games"]), row["test_game"], row["input_view"], row["view"], row["seed"],
            row["spell_f1"], row["delta"], row["model_sha256"], row["prediction_sha256"])
        tuning.append("| {} | p10d_inner_{:04d} | {} | B0 | — | — | — | — | — | — | — | info (P10-D calibration) | {} |".format(
            args.date, index, revision[:7], note))
    (out / "calibration_ledger_rows.md").write_text("\n".join(tuning) + "\n")
    reports, clean, ledger = [], {}, []
    for cell in cells:
        game, model = cell["game"], models[cell["game"]]
        check_split(model["training"]["games"], game)
        candidate = apply_cell(cell, datasets[game]["features"], model)
        for system, prediction in (("B0", cell["baseline"]), ("gap_dead", candidate)):
            key = cell["input_view"], game, system
            if cell["view"] == "clean":
                clean[key] = prediction
            name = "{}_{}_{}".format(system, cell["view"], cell["seed"] if cell["seed"] is not None else "clean")
            path = out / "predictions" / cell["input_view"] / game / (name + ".csv")
            write_submission(np.asarray(pm.LABELS)[prediction], str(path), n_frames=len(prediction), blocks=cell["engine_blocks"])
            validate(str(path), rows=len(prediction))
            metrics = gb.score_cell(datasets[game]["reference"], prediction, clean[key], cell["mask"],
                                    cell["missing"], cell["blocks"], cell["fps"], args.draws)
            # Separate the correction's added/removed false possessions from masking effects.
            metrics["correction_spurious_spell_changes"] = gb.spurious_spell_changes(
                datasets[game]["reference"], prediction, cell["baseline"], pm.valid_runs(datasets[game]["reference"]))
            report = {"schema": SCHEMA, "game": game, "input_view": cell["input_view"], "view": cell["view"],
                      "seed": cell["seed"], "system": system, "role": "held-out", "mask_md5": cell["mask"]["md5"],
                      "reference_sha256": datasets[game]["reference_sha256"],
                      "feature_sha256": datasets[game]["feature_sha256"],
                      **{k: cell["metadata"][k] for k in ("profile_md5", "config_hash", "engine_md5", "dead_support_sha256")},
                      "state_source_sha256": source["model_source_sha256"], "runner_sha256": source["runner_sha256"],
                      "model_sha256": gd.artifact_hash(model) if system == "gap_dead" else None,
                      "training_games": model["training"]["games"] if system == "gap_dead" else [],
                      "prediction_md5": gm.file_md5(str(path)), "prediction_sha256": shared.file_sha256(path),
                      "audit": correction_audit(cell["baseline"], prediction, cell["dead_eligible"], cell["engine_blocks"]),
                      "metrics": metrics}
            reports.append(report)
            shared.write_json(out / "reports" / cell["input_view"] / game / (name + ".json"), report)
            score = metrics["score"]
            note = "held-out external reference; spellF1 {:.6f}; macroF1 {:.6f}; model {}; mask {}; prediction md5 {}".format(
                score["spell_f1_mean"], score["macro_f1"], report["model_sha256"] or "none", report["mask_md5"] or "clean", report["prediction_md5"])
            ledger.append("| {} | p10d_{}_{}_{} | {} | {} | — | — | — | — | — | — | — | info (P10-D) | {} |".format(
                args.date, cell["input_view"], game, name, revision[:7], report["config_hash"], note))
        print("scored {} {} {} {}".format(cell["input_view"], game, cell["view"], cell["seed"]), flush=True)
    summary = aggregate_reports(reports, args.draws)
    summary.update(revision=revision, execution=source, selection=choices)
    shared.write_json(out / "summary.json", summary)
    (out / "ledger_rows.md").write_text("\n".join(ledger) + "\n")
    print("complete: {} validated reports; accepted {}".format(len(reports), summary["accepted"]), flush=True)


def regenerate_summary(out, reports):
    """Preserve execution provenance when a later verifier rebuilds the aggregate."""
    out = gb.outside_repo(out)
    previous_path = out / "summary.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    execution = previous.get("execution")
    if execution is None:
        execution = json.loads((out / "execution.json").read_text())
    selection = previous.get("selection")
    if selection is None:
        selection = json.loads((out / "selection.json").read_text())["folds"]
    fields = {"engine_md5": "engine_fingerprint", "dead_support_sha256": "dead_support_sha256",
              "state_source_sha256": "model_source_sha256", "runner_sha256": "runner_sha256"}
    if any(row[report_key] != execution[source_key]
           for row in reports for report_key, source_key in fields.items()):
        raise ValueError("reports differ from the saved execution provenance")
    summary = aggregate_reports(reports)
    summary.update(revision=execution["revision"], execution=execution, selection=selection)
    summary["verification"] = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip(),
        "runner_sha256": shared.file_sha256(__file__), "execution_metadata_preserved": True}
    shared.write_json(previous_path, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for name in ("prepared-root", "references", "mask-root", "profile", "out-dir"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--draws", type=int, default=1000)
    run.add_argument("--date", default="2026-11-02")
    run.add_argument("--exposure-only", action="store_true")
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
        regenerate_summary(out, reports)
    return 0


if __name__ == "__main__":
    sys.exit(main())
