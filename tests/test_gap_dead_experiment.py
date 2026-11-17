"""Independent-match fitting, selection and acceptance checks for P10-D."""

from copy import deepcopy
import itertools

import numpy as np
import pytest

from tools import gap_dead_experiment as experiment
from tools.gap_models import gap_dead as gd

pm = experiment.pm


def dataset(game, offset=0.0, length=100):
    features = np.zeros((length, len(gd.FEATURE_NAMES)), dtype=np.float32)
    features[:, 0] = np.arange(length) / 10.0 + offset
    features[:, 1] = (np.arange(length) % 7) - 3
    reference = np.where(np.arange(length) % 10 < 5, pm.H, pm.D).astype(np.int8)
    return {"features": features, "reference": reference, "min_players": 1,
            "feature_sha256": experiment.shared.array_sha256(features),
            "reference_sha256": "a" * 64}


def test_all_eight_outer_matches_and_every_inner_copy_are_held_out():
    outer = experiment.grouped_folds(experiment.gb.OPEN_GAMES)
    assert len(outer) == 8
    for training, held in outer:
        assert len(training) == 7 and held not in training
        experiment.check_split(training, held)
        for inner_training, inner_held in experiment.grouped_folds(training):
            assert len(inner_training) == 6
            assert held not in inner_training and held != inner_held
            experiment.check_split(inner_training, inner_held)


@pytest.mark.parametrize("training,held", [(("BAR-ATH",), "METRICA-S1"),
    (("DFL-J03WQQ",), "METRICA-S1"), (("METRICA-S1",), "METRICA-S1"),
    (("METRICA-S1",), "DFL-J03WR9"), (("METRICA-S1", "METRICA-S1"), "METRICA-S2")])
def test_target_sealed_duplicate_and_leaking_splits_fail(training, held):
    with pytest.raises(ValueError):
        experiment.check_split(training, held)


def test_fit_intercept_scaling_and_weighted_objective_are_deterministic():
    x = np.column_stack((np.arange(60, dtype=float), np.ones(60)))
    y = (np.arange(60) >= 15).astype(float)
    weights = np.linspace(1.0, 2.0, 60)
    first = experiment.fit_logistic(x, y, weights, 0.1)
    second = experiment.fit_logistic(x, y, weights, 0.1)
    assert first == second
    np.testing.assert_allclose(first["means"], np.average(x, axis=0, weights=weights))
    assert first["coefficients"][0] > 0 and first["coefficients"][1] == 0
    assert first["scales"][1] == 1.0 and first["intercept"] > 0
    probability = 1 / (1 + np.exp(-((x - first["means"]) / first["scales"] @ np.array(first["coefficients"]) + first["intercept"])))
    assert abs(np.average(probability - y, weights=weights)) < 1e-8


def test_equal_match_weight_and_every_fifth_graded_training_row():
    games = ("METRICA-S1", "DFL-J03WMX")
    datasets = {games[0]: dataset(games[0], length=100), games[1]: dataset(games[1], offset=50, length=50)}
    x, y, weights, provenance = experiment.training_data(datasets, games)
    # Alphabetic DFL first has 10 samples, Metrica second has 20.
    assert len(x) == 30
    assert weights[:10].sum() == pytest.approx(0.5)
    assert weights[10:].sum() == pytest.approx(0.5)
    np.testing.assert_array_equal(x[:10], datasets[games[1]]["features"][::5])
    assert set(np.unique(y)) == {0, 1}
    assert provenance["games"] == sorted(games)


def test_held_out_features_and_targets_cannot_change_fit_or_scaling():
    games = ("DFL-J03WMX", "METRICA-S1")
    datasets = {game: dataset(game) for game in games}
    first = experiment.fit_model(datasets, (games[0],), experiment.recipes()[1])
    datasets[games[1]]["features"][:] = 1e10
    datasets[games[1]]["reference"][:] = pm.D
    second = experiment.fit_model(datasets, (games[0],), experiment.recipes()[1])
    assert first == second
    assert first["training"]["games"] == [games[0]]
    gd.validate_artifact(first)


def selection_rows(gains):
    return [{"recipe": recipe, "gain": gain, "clean_provider_means": {
        provider + "/" + view: 0.0 for provider, view in itertools.product(("DFL", "METRICA"), ("raw", "matched"))}}
        for recipe, gain in zip(experiment.recipes()[1:], gains)]


def test_identity_nonpositive_gain_and_conservative_tie_order():
    assert experiment.choose_recipe(selection_rows([0, -0.1, 0, -0.2]))["recipe"]["identity"]
    chosen = experiment.choose_recipe(list(reversed(selection_rows([0.02] * 4))))
    assert chosen["recipe"] == {"identity": False, "l2": 1.0, "threshold": 0.95}
    rows = selection_rows([0.1, 0.08, 0.02, 0.03])
    rows[0]["clean_provider_means"]["METRICA/raw"] = -0.010001
    assert experiment.choose_recipe(rows)["recipe"] == rows[1]["recipe"]
    with pytest.raises(ValueError, match="complete"):
        experiment.choose_recipe(rows[:-1])


def test_selection_uses_provider_clean_mean_and_never_sensitivities(monkeypatch):
    trainer = experiment.Trainer({}, [])
    called = []

    def estimate(training, held, recipe):
        called.append((training, held))
        return {"game": held, "gain": 0.01,
                "clean": {view: (-0.02 if held == "METRICA-S1" else 0.02)
                          for view in experiment.INPUT_VIEWS}}

    monkeypatch.setattr(trainer, "estimate", estimate)
    recipe, result = trainer.select(experiment.gb.OPEN_GAMES)
    assert not recipe["identity"]
    assert result["choice"]["clean_provider_means"]["METRICA/raw"] == pytest.approx(0.02 / 3)
    assert len(called) == 32
    assert all(len(training) == 7 and held not in training for training, held in called)


def test_apply_and_audit_require_whole_agreed_run_and_preserve_every_live_label():
    labels = np.array(list("HDDDHADDDHHDDDAX"))
    codes = pm.label_codes(labels)
    cell = {"baseline": codes, "dead_eligible": labels == "D", "engine_blocks": [(1, 0, 14)]}
    model = experiment.fit_model({"METRICA-S1": dataset("METRICA-S1")}, ("METRICA-S1",), experiment.recipes()[1])
    result = experiment.apply_cell(cell, None, model, logits=np.full(len(labels), 10.0))
    assert "".join(np.asarray(pm.LABELS)[result]) == "HHHHHADDDHHDDDAX"
    audit = experiment.correction_audit(codes, result, cell["dead_eligible"], cell["engine_blocks"])
    assert audit["changed_frames"] == 3 and audit["removed_runs"] == 1
    fragmented = codes.copy()
    fragmented[1] = pm.H
    with pytest.raises(ValueError, match="fragmented"):
        experiment.correction_audit(codes, fragmented, cell["dead_eligible"], cell["engine_blocks"])
    protected = result.copy()
    protected[5] = pm.H
    with pytest.raises(ValueError, match="protected"):
        experiment.correction_audit(codes, protected, cell["dead_eligible"], cell["engine_blocks"])


def reports(delta=0.01, false_delta=-0.1):
    result = []
    views = [("clean", (None,)), ("U", experiment.SEEDS), ("S", experiment.SEEDS), ("S_alt", experiment.SEEDS)]
    views += [(view, (0,)) for view in experiment.SENSITIVITY_VIEWS]
    for iv, game in itertools.product(experiment.INPUT_VIEWS, experiment.gb.OPEN_GAMES):
        for view, seeds in views:
            for seed, system in itertools.product(seeds, ("B0", "gap_dead")):
                candidate = system == "gap_dead"
                result.append({"input_view": iv, "game": game, "view": view, "seed": seed, "system": system,
                    "role": "held-out", "engine_md5": "engine", "config_hash": "B0", "profile_md5": "profile",
                    "dead_support_sha256": "support", "state_source_sha256": "source", "runner_sha256": "runner",
                    "mask_md5": None if view == "clean" else game + view + str(seed),
                    "reference_sha256": game, "feature_sha256": game + "features", "model_sha256": game if candidate else None,
                    "training_games": sorted(set(experiment.gb.OPEN_GAMES) - {game}) if candidate else [],
                    "audit": {"outside_mask_changes": 0, "original_live_changes": 0, "x_changes": 0,
                              "changed_frames": int(candidate), "removed_runs": int(candidate)},
                    "metrics": {"score": {"spell_f1_mean": 0.5 + (delta if candidate else 0)},
                                "masked": {"false_d_rate": 0.5 + (false_delta if candidate else 0)}}})
    return result


def test_acceptance_requires_every_pair_and_whole_match_improvement():
    result = experiment.aggregate_reports(reports(), draws=50)
    assert result["accepted"] and result["cells"] == 512
    assert all(row["matches"] == 8 for row in result["primary_false_d_reduction"])
    assert not experiment.aggregate_reports(reports(delta=0), draws=50)["accepted"]
    assert not experiment.aggregate_reports(reports(false_delta=0), draws=50)["accepted"]
    with pytest.raises(ValueError, match="unpaired"):
        experiment.aggregate_reports(reports()[:-1])


@pytest.mark.parametrize("field", ("engine_md5", "dead_support_sha256", "state_source_sha256", "runner_sha256", "mask_md5", "reference_sha256", "feature_sha256", "model_sha256"))
def test_aggregate_rejects_mixed_provenance(field):
    rows = reports()
    rows[-1][field] = "changed"
    with pytest.raises(ValueError):
        experiment.aggregate_reports(rows)


def test_provider_and_individual_metrica_regressions_cannot_hide_in_pool():
    rows = reports()
    for row in rows:
        if row["system"] == "gap_dead" and row["game"] == "METRICA-S1" and row["view"] == "U":
            row["metrics"]["score"]["spell_f1_mean"] = 0.499
    assert not experiment.aggregate_reports(rows)["gates"]["nonnegative_metrica_primary_match_means"]
    rows = reports()
    for row in rows:
        if row["system"] == "gap_dead" and row["game"].startswith("DFL") and row["view"] == "L090":
            row["metrics"]["score"]["spell_f1_mean"] = 0.499
    assert not experiment.aggregate_reports(rows)["gates"]["nonnegative_provider_sensitivity_means"]


def test_outer_model_never_reuses_the_final_all_match_training_identity():
    rows = reports()
    rows[-1]["training_games"] = sorted(experiment.gb.OPEN_GAMES)
    with pytest.raises(ValueError, match="leave-one-match-out"):
        experiment.aggregate_reports(rows)


def test_inner_estimate_never_visits_sensitivity_cells(monkeypatch):
    game, training = "METRICA-S1", ("DFL-J03WMX",)
    truth = np.full(5, pm.H, dtype=np.int8)
    baseline = pm.label_codes(np.array(list("HDDDH")))
    cells = [{"game": game, "input_view": iv, "view": view, "seed": None if view == "clean" else 0,
              "baseline": baseline, "dead_eligible": baseline == pm.D, "engine_blocks": [(1, 0, 4)],
              "baseline_spell_f1": pm.spell_scores(truth, baseline, pm.valid_runs(truth))["mean_f1"],
              "mask": {"md5": None}}
             for iv, view in itertools.product(experiment.INPUT_VIEWS, ("clean", "U", "S", "S_alt", "L045", "L090", "L180"))]
    datasets = {game: {"features": np.zeros((5, len(gd.FEATURE_NAMES))), "reference": truth}}
    trainer = experiment.Trainer(datasets, cells)
    model = experiment.fit_model({training[0]: dataset(training[0])}, training, experiment.recipes()[1])
    monkeypatch.setattr(trainer, "model", lambda *args: model)
    visited = []
    original = experiment.apply_cell

    def tracked(cell, features, artifact, logits=None):
        visited.append(cell["view"])
        return original(cell, features, artifact, logits)

    monkeypatch.setattr(experiment, "apply_cell", tracked)
    trainer.estimate(training, game, experiment.recipes()[1])
    assert visited == ["clean", "U", "S", "clean", "U", "S"]
    assert len(trainer.calibration_rows) == 6


def test_resummarize_preserves_execution_and_selection_and_records_later_verifier(tmp_path):
    """The summary records the git revision of the verifier, so the test needs a git checkout."""
    if not (experiment.ROOT / ".git").exists():
        pytest.skip("this folder is not a git checkout, so it has no revision to record")
    execution = {"revision": "executed-commit", "runner_sha256": "runner", "model_source_sha256": "source",
                 "engine_fingerprint": "engine", "dead_support_sha256": "support"}
    selection = [{"held_out_game": "METRICA-S1", "chosen_recipe": "frozen"}]
    previous = {"revision": "executed-commit", "execution": execution, "selection": selection}
    experiment.shared.write_json(tmp_path / "summary.json", previous)
    summary = experiment.regenerate_summary(tmp_path, reports())
    assert summary["revision"] == "executed-commit"
    assert summary["execution"] == execution and summary["selection"] == selection
    assert summary["verification"]["execution_metadata_preserved"]
    assert summary["verification"]["runner_sha256"] != "runner"
    changed = reports()
    changed[-1]["runner_sha256"] = "different execution"
    with pytest.raises(ValueError, match="saved execution"):
        experiment.regenerate_summary(tmp_path, changed)
