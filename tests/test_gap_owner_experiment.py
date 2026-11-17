"""Pin training isolation, identity selection and experiment reporting."""

from copy import deepcopy
import itertools

import numpy as np
import pytest

from tools import gap_owner_experiment as experiment


def test_grid_is_fixed_and_identity_is_first():
    grid = experiment.recipes()
    assert len(grid) == 13
    assert grid[0]["identity"]
    assert not any(row["identity"] for row in grid[1:])
    assert {(row["l2"], row["switch_cost"], row["baseline_log_odds"])
            for row in grid[1:]} == set(itertools.product((0.01, 0.1, 1.0), (2.0, 8.0), (0.5, 1.5)))
    assert {row["endpoint_log_odds"] for row in grid} == {2.0}


def test_nested_folds_never_contain_the_outer_test_match():
    outer = experiment.grouped_folds(experiment.METRICA)
    assert len(outer) == 3
    for training, held in outer:
        assert held not in training
        inner = experiment.grouped_folds(training)
        assert len(inner) == 2
        for fit, validation in inner:
            assert len(fit) == 1
            assert held not in fit and held != validation
            assert validation not in fit


@pytest.mark.parametrize("games", [("METRICA-S1",), ("METRICA-S1", "METRICA-S1"),
                                    ("METRICA-S1", "DFL-J03WMX"), ("METRICA-S1", "BAR-ATH")])
def test_grouped_folds_reject_invalid_training_games(games):
    with pytest.raises(ValueError):
        experiment.grouped_folds(games)


@pytest.mark.parametrize("train,test", [(("METRICA-S1",), "METRICA-S1"),
                                         (("DFL-J03WMX",), "METRICA-S1"),
                                         (("METRICA-S1",), "DFL-J03WQQ"),
                                         (("METRICA-S1",), "BAR-ATH"), ((), "METRICA-S1")])
def test_split_rejects_leakage_sealed_and_target_games(train, test):
    with pytest.raises(ValueError):
        experiment.check_split(train, test)


def test_logistic_fit_is_deterministic_and_exactly_antisymmetric():
    x = np.array([[-2.0, 1.0], [-1.0, 0.0], [1.0, 0.0], [2.0, -1.0]])
    y = np.array([0, 0, 1, 1])
    weights = np.ones(4)
    first = experiment.fit_logistic(x, y, weights, 0.1)
    second = experiment.fit_logistic(x, y, weights, 0.1)
    assert first == second
    assert first["means"] == [0.0, 0.0]
    assert first["converged"]
    score = (x / first["scales"]) @ first["coefficients"]
    mirrored = (-x / first["scales"]) @ first["coefficients"]
    assert np.array_equal(score, -mirrored)
    assert np.array_equal(score > 0, y == 1)


def test_implicit_mirroring_matches_explicit_augmentation():
    x = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 4.0], [-1.0, 2.0]])
    y = np.array([0, 1, 1, 0])
    weight = np.array([1.0, 3.0, 2.0, 4.0])
    implicit = experiment.fit_logistic(x, y, weight, 0.01)
    explicit = experiment.fit_logistic(np.concatenate((x, -x)), np.concatenate((y, 1 - y)),
                                        np.concatenate((weight, weight)), 0.01)
    np.testing.assert_allclose(implicit["scales"], explicit["scales"], atol=1e-14)
    np.testing.assert_allclose(implicit["coefficients"], explicit["coefficients"], atol=1e-12)


def test_zero_feature_column_has_finite_unit_scale():
    result = experiment.fit_logistic(np.array([[-1, 0], [1, 0]]), np.array([0, 1]), np.ones(2), 1)
    assert result["scales"][1] == 1
    assert result["coefficients"][1] == 0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_training_values_fail(bad):
    with pytest.raises(ValueError, match="finite"):
        experiment.fit_logistic(np.array([[1, bad], [0, 1]]), np.array([0, 1]), np.ones(2), 0.1)


@pytest.mark.parametrize("ridge", [0, -1, np.nan, np.inf])
def test_bad_regularization_fails(ridge):
    with pytest.raises(ValueError):
        experiment.fit_logistic(np.array([[-1], [1]]), np.array([0, 1]), np.ones(2), ridge)


def datasets():
    result = {}
    size = len(experiment.go.FEATURE_NAMES)
    for index, game in enumerate(experiment.METRICA):
        x = np.zeros((40, size))
        x[:, 0] = np.linspace(-2, 2, 40) + index * 0.1
        result[game] = {"features": x, "reference": np.where(x[:, 0] > 0, 0, 1).astype(np.int8),
                        "reference_sha256": str(index) * 64, "min_players": 6}
    return result


def test_fit_does_not_read_held_out_features_or_labels():
    data = datasets()
    recipe = experiment.recipes()[1]
    train = experiment.METRICA[:2]
    before = experiment.fit_model(data, train, recipe)
    data[experiment.METRICA[2]]["features"][:] = np.nan
    data[experiment.METRICA[2]]["reference"][:] = 3
    after = experiment.fit_model(data, train, recipe)
    assert before == after
    assert before["training"]["games"] == list(train)
    experiment.go.validate_artifact(before)


def test_training_reference_hash_is_in_artifact_identity():
    data = datasets()
    recipe = experiment.recipes()[1]
    before = experiment.fit_model(data, experiment.METRICA[:1], recipe)
    data[experiment.METRICA[0]]["reference_sha256"] = "a" * 64
    after = experiment.fit_model(data, experiment.METRICA[:1], recipe)
    assert experiment.go.artifact_hash(before) != experiment.go.artifact_hash(after)


def test_identity_artifact_has_zero_coefficients():
    model = experiment.fit_model(datasets(), experiment.METRICA[:2], experiment.recipes()[0])
    assert not any(model["coefficients"])
    experiment.go.validate_artifact(model)


def test_choose_recipe_prefers_identity_at_zero_and_rejects_clean_loss():
    grid = experiment.recipes()
    result = experiment.choose_recipe([
        {"recipe": grid[1], "gain": 0.0, "clean_min": 0.0},
        {"recipe": grid[2], "gain": 0.1, "clean_min": -0.01001},
    ])
    assert result["recipe"]["identity"]


def test_positive_tie_keeps_first_conservative_recipe():
    grid = experiment.recipes()
    rows = [{"recipe": row, "gain": 0.01, "clean_min": 0.0} for row in grid[1:]]
    assert experiment.choose_recipe(rows)["recipe"] == grid[1]


def test_audit_rejects_outside_dead_and_unknown_edits():
    baseline = np.array([0, 1, 2, 3], dtype=np.int8)
    for index in (0, 2, 3):
        changed = baseline.copy()
        changed[index] = 1 - changed[index] if index == 0 else 0
        with pytest.raises(ValueError, match="protected"):
            experiment.correction_audit(baseline, changed, np.array([False, True, False, False]))


def test_validation_rejects_graded_unknown_and_ungraded_team():
    blocks = [(1, 1, 2)]
    experiment.validate_codes(np.array([3, 0, 1, 3]), blocks)
    for bad in (np.array([0, 0, 1, 3]), np.array([3, 3, 1, 3])):
        with pytest.raises(ValueError):
            experiment.validate_codes(bad, blocks)


def full_reports():
    reports = []
    view_seeds = [("clean", (None,))] + [(v, experiment.SEEDS) for v in experiment.gb.VIEWS]
    view_seeds += [(v, (0,)) for v in experiment.SENSITIVITY_VIEWS]
    for input_view, game in itertools.product(experiment.INPUT_VIEWS, experiment.gb.OPEN_GAMES):
        train = [g for g in experiment.METRICA if g != game]
        for view, seeds in view_seeds:
            for seed, system in itertools.product(seeds, ("B0", "gap_owner")):
                reports.append({"input_view": input_view, "game": game, "view": view, "seed": seed,
                                "system": system, "mask_md5": view + str(seed), "reference_sha256": game,
                                "config_hash": "b0", "engine_md5": "generation", "training_games": train,
                                "profile_md5": "profile", "model_sha256": "model-" + (game if game in experiment.METRICA else "final"),
                                "audit": {"outside_mask_changes": 0, "dead_changes": 0, "x_changes": 0},
                                "metrics": {"score": {"spell_f1_mean": 0.5 + (0.01 if system == "gap_owner" else 0)}}})
    return reports


def test_aggregation_counts_matches_not_seeds_and_separates_transfer():
    result = experiment.aggregate_reports(full_reports())
    assert result["cells"] == 512
    assert result["complete"]
    for row in result["paired_match_intervals"]:
        assert row["matches"] == {"METRICA": 3, "DFL": 5, "all": 8}[row["provider"]]
        assert row["mean"] == pytest.approx(0.01)
        assert row["baseline_spell_f1"] == pytest.approx(0.5)
        assert row["candidate_spell_f1"] == pytest.approx(0.51)
    assert result["owner_gate_nonnegative_metrica"]
    assert result["positive_primary_metrica_intervals"]


def test_aggregation_rejects_missing_or_duplicate_cells():
    reports = full_reports()
    with pytest.raises(ValueError):
        experiment.aggregate_reports(reports[:-1])
    with pytest.raises(ValueError):
        experiment.aggregate_reports(reports + [reports[0]])


@pytest.mark.parametrize("field", ["mask_md5", "reference_sha256", "config_hash", "engine_md5"])
def test_aggregation_rejects_pair_provenance_drift(field):
    reports = full_reports()
    reports[1][field] = "different"
    with pytest.raises(ValueError):
        experiment.aggregate_reports(reports)


@pytest.mark.parametrize("field", ["engine_md5", "config_hash", "reference_sha256", "profile_md5"])
def test_aggregation_rejects_paired_but_stale_generation(field):
    reports = full_reports()
    reports[0][field] = reports[1][field] = "stale"
    with pytest.raises(ValueError):
        experiment.aggregate_reports(reports)


def test_aggregation_rejects_model_drift_across_seeds():
    reports = full_reports()
    reports[1]["model_sha256"] = "different-model"
    with pytest.raises(ValueError, match="model"):
        experiment.aggregate_reports(reports)


def test_aggregation_requires_one_final_dfl_model():
    reports = full_reports()
    for row in reports:
        if row["game"] == "DFL-J03WMX" and row["system"] == "gap_owner":
            row["model_sha256"] = "different-final-model"
    with pytest.raises(ValueError, match="one final model"):
        experiment.aggregate_reports(reports)


def test_aggregation_requires_identical_raw_and_matched_masks():
    reports = full_reports()
    for row in reports:
        if row["input_view"] == "matched" and row["game"] == "METRICA-S1" and row["view"] == "U" and row["seed"] == 0:
            row["mask_md5"] = "different-paired-mask"
    with pytest.raises(ValueError, match="frozen mask"):
        experiment.aggregate_reports(reports)


def test_inner_selection_never_scores_sensitivity_cells(monkeypatch):
    data = datasets()
    held = experiment.METRICA[0]
    cells = []
    for input_view in experiment.INPUT_VIEWS:
        for view in ("clean",) + experiment.ALL_VIEWS:
            seeds = (None,) if view == "clean" else experiment.SEEDS
            for seed in seeds:
                cells.append({"game": held, "input_view": input_view, "view": view, "seed": seed,
                              "baseline_spell_f1": 0.5, "mask": {"md5": "mask"}})
    trainer = experiment.Trainer(data, cells)
    monkeypatch.setattr(trainer, "model", lambda games, recipe: {})
    monkeypatch.setattr(experiment.go, "predict_log_odds", lambda features, model: np.zeros(len(features)))
    monkeypatch.setattr(experiment.go, "artifact_hash", lambda model: "hash")
    visited = []

    def apply(cell, features, model, log_odds=None):
        visited.append(cell["view"])
        return data[held]["reference"]

    monkeypatch.setattr(experiment, "apply_cell", apply)
    trainer.estimate((experiment.METRICA[1],), held, experiment.recipes()[1])
    assert set(visited) == {"clean", "U", "S"}
    assert len(visited) == 18


def test_aggregation_rejects_held_out_training_leak():
    reports = full_reports()
    row = next(r for r in reports if r["game"] in experiment.METRICA and r["system"] == "gap_owner")
    row["training_games"].append(row["game"])
    with pytest.raises(ValueError):
        experiment.aggregate_reports(reports)


def test_metrica_harm_cannot_be_hidden_by_dfl_gains():
    reports = full_reports()
    for row in reports:
        if row["game"] == "METRICA-S1" and row["system"] == "gap_owner" and row["view"] == "U":
            row["metrics"]["score"]["spell_f1_mean"] = 0.49
    result = experiment.aggregate_reports(reports)
    assert not result["owner_gate_nonnegative_metrica"]


def test_json_hash_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        experiment.json_hash({"value": float("nan")})
