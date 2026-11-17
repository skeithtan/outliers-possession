"""Synthetic support checks for the selective dead-ball experiment."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from engine import deadball, possession, tracks
from engine.config import Config
from tools.gap_models import gap_dead


def scene(kind="reappear", blocks=None):
    """Build a missing-ball stop with a single explicit restart."""
    cfg = Config.default()
    n, fps, pitch = 240, 25.0, (105.0, 68.0)
    x, y = np.full(n, 50.0), np.full(n, 34.0)
    x[20:160], y[20:160] = np.nan, np.nan
    ball = tracks.clean_ball(x / pitch[0], y / pitch[1], pitch, cfg, fps=fps)
    feats = {key: np.full(n, value) for key, value in {
        "meanv_all": 1.0, "sep": 14.0, "dmin": 10.0, "d_second": 12.0}.items()}
    onsets = pd.DataFrame(columns=deadball.ONSET_COLUMNS)
    restarts = pd.DataFrame([(160, "H", kind, 0.9, True)],
                            columns=deadball.RESTART_COLUMNS)
    blocks = blocks or [(1, 0, n - 1)]
    kickoffs = {half: first for half, first, _ in blocks}
    inputs = {"n_frames": n, "blocks": blocks, "kickoffs": kickoffs, "pitch": pitch,
              "fps": fps, "ball": ball, "feats": feats, "tagging": None,
              "onsets": onsets, "restarts": restarts,
              "ghost": np.zeros(n, dtype=bool), "holds": None, "hold_rows": None,
              "events": pd.DataFrame(columns=["frame"])}
    return cfg, inputs, result(inputs, cfg)


def result(inputs, cfg):
    """Construct matching state-machine and final labels for the scene."""
    inputs["e_dead"] = deadball.dead_evidence(
        inputs["ball"], inputs["feats"], None, inputs["blocks"], cfg,
        kickoffs=inputs["kickoffs"], pitch=inputs["pitch"])
    mask, exits = possession.dead_intervals(
        inputs["n_frames"], inputs["onsets"], inputs["e_dead"], inputs["restarts"],
        inputs["blocks"], inputs["kickoffs"], cfg, inputs["fps"], inputs.get("holds"))
    mask, _ = possession.shift_stoppage(mask, exits, inputs["onsets"], inputs["restarts"],
                                       cfg, inputs["blocks"], inputs["kickoffs"])
    labels = np.full(inputs["n_frames"], "X", dtype="<U1")
    for _, first, last in inputs["blocks"]:
        labels[first:last + 1] = "H"
    labels[mask] = "D"
    return {"stoppage": mask, "labels": labels}


def test_gap_only_stop_is_eligible_without_mutating_inputs():
    cfg, inputs, parts = scene()
    snapshot = deepcopy(inputs)
    mask, audit = gap_dead.gap_only_mask(inputs, parts, cfg)
    assert np.array_equal(np.flatnonzero(mask), np.arange(20, 160))
    assert audit["gap_only_segments"] == 1
    assert audit["segments"][0]["support"] == []
    assert audit["counterfactual_stoppage_frames"] == 0
    for key in ("e_dead", "ghost"):
        np.testing.assert_array_equal(inputs[key], snapshot[key])
    for key in ("onsets", "restarts"):
        pd.testing.assert_frame_equal(inputs[key], snapshot[key])


@pytest.mark.parametrize("mechanism", ["out", "pursuit_stop"])
def test_physical_onset_protects_the_whole_dead_interval(mechanism):
    cfg, inputs, parts = scene()
    inputs["onsets"] = pd.DataFrame([(20, mechanism, 0.9)], columns=deadball.ONSET_COLUMNS)
    parts = result(inputs, cfg)
    mask, audit = gap_dead.gap_only_mask(inputs, parts, cfg)
    assert not mask.any()
    expected = "out_onset" if mechanism == "out" else "pursuit_stop"
    assert expected in audit["segments"][0]["support"]
    assert "counterfactual_support" in audit["segments"][0]["support"]


def test_later_physical_onset_also_protects_the_interval():
    cfg, inputs, parts = scene()
    inputs["onsets"] = pd.DataFrame([(100, "out", 0.9)], columns=deadball.ONSET_COLUMNS)
    parts = result(inputs, cfg)
    mask, audit = gap_dead.gap_only_mask(inputs, parts, cfg)
    assert not mask.any()
    assert audit["protected_by_reason"]["counterfactual_support"] == 1


def test_chosen_kick_restart_protects_even_after_timing_shift():
    cfg, inputs, parts = scene("kick")
    mask, audit = gap_dead.gap_only_mask(inputs, parts, cfg)
    assert not mask.any()
    assert audit["segments"][0]["stop"] == 160 + cfg.DELTA_RESTART_KICK
    assert "kick_restart" in audit["segments"][0]["support"]


def test_unconfirmed_kick_does_not_protect_a_reappearance_exit():
    cfg, inputs, parts = scene()
    inputs["restarts"] = pd.concat([pd.DataFrame([(80, "H", "kick", 0.9, False)],
                                                columns=deadball.RESTART_COLUMNS),
                                     inputs["restarts"]], ignore_index=True)
    parts = result(inputs, cfg)
    mask, audit = gap_dead.gap_only_mask(inputs, parts, cfg)
    assert mask.sum() == 140
    assert audit["protected_by_reason"]["kick_restart"] == 0


@pytest.mark.parametrize("reason", ["ghost", "keeper_hold"])
def test_special_hold_protects_the_whole_interval(reason):
    cfg, inputs, parts = scene()
    if reason == "ghost":
        inputs["ghost"][100] = True
    else:
        hold = np.zeros(inputs["n_frames"], dtype=bool)
        hold[100] = True
        inputs["holds"] = (hold, np.zeros_like(hold))
    mask, audit = gap_dead.gap_only_mask(inputs, parts, cfg)
    assert not mask.any()
    assert reason in audit["segments"][0]["support"]


def test_hold_owner_rows_protect_when_hold_is_play_is_disabled():
    cfg, inputs, parts = scene()
    inputs["hold_rows"] = pd.DataFrame([(20, "keeper_hold;end=159")],
                                       columns=["frame", "notes"])
    mask, audit = gap_dead.gap_only_mask(inputs, parts, cfg)
    assert not mask.any()
    assert audit["protected_by_reason"]["keeper_hold"] == 1


def test_only_missing_frames_of_a_gap_dependent_interval_are_eligible():
    cfg, inputs, parts = scene()
    inputs["ball"]["visible"][100] = True
    inputs["ball"]["x"][100], inputs["ball"]["y"][100] = 50.0, 34.0
    inputs["e_dead"][100] = 0.1
    parts = result(inputs, cfg)
    mask, _ = gap_dead.gap_only_mask(inputs, parts, cfg)
    assert mask.sum() == 139
    assert not mask[100]
    assert not mask[parts["labels"] != "D"].any()


def test_observed_contact_protects_entire_interval():
    cfg, inputs, parts = scene()
    inputs["ball"]["visible"][100] = True
    inputs["ball"]["x"][100], inputs["ball"]["y"][100] = 50.0, 34.0
    inputs["events"] = pd.DataFrame([100], columns=["frame"])
    parts = result(inputs, cfg)
    mask, audit = gap_dead.gap_only_mask(inputs, parts, cfg)
    assert not mask.any()
    assert "observed_contact" in audit["segments"][0]["support"]


def test_prekickoff_and_touching_blocks_remain_separate():
    cfg, inputs, parts = scene(blocks=[(1, 0, 79), (2, 80, 239)])
    inputs["kickoffs"][2] = 120
    parts = result(inputs, cfg)
    mask, audit = gap_dead.gap_only_mask(inputs, parts, cfg)
    assert not mask[80:120].any()
    assert audit["segment_count"] == 2
    assert audit["segments"][0]["stop"] == 80
    assert audit["segments"][1]["start"] == 80
    assert "pre_kickoff" in audit["segments"][1]["support"]


def test_mismatched_run_parts_are_rejected():
    cfg, inputs, parts = scene()
    parts["stoppage"][40] = False
    with pytest.raises(ValueError, match="do not match"):
        gap_dead.gap_only_mask(inputs, parts, cfg)


def test_kick_protection_cannot_cross_a_touching_half_boundary():
    cfg, inputs, parts = scene(blocks=[(1, 0, 99), (2, 100, 239)])
    inputs["restarts"] = pd.DataFrame([(95, "H", "kick", 0.9, True),
                                       (160, "H", "reappear", 0.9, True)],
                                      columns=deadball.RESTART_COLUMNS)
    # A new evidence run starts in the second block after a one-frame break.
    inputs["feats"]["meanv_all"][99] = 3.0
    parts = result(inputs, cfg)
    mask, audit = gap_dead.gap_only_mask(inputs, parts, cfg)
    second = [row for row in audit["segments"] if row["half"] == 2]
    assert len(second) == 1
    assert "kick_restart" not in second[0]["support"]
    assert mask[100:160].all()


def test_counterfactual_uses_frozen_detections(monkeypatch):
    cfg, inputs, parts = scene()
    def forbidden(*args, **kwargs):
        raise AssertionError("the provenance audit must not redetect events")
    monkeypatch.setattr(deadball, "restarts", forbidden)
    monkeypatch.setattr(deadball, "dead_onsets", forbidden)
    monkeypatch.setattr(possession, "prepare", forbidden)
    mask, _ = gap_dead.gap_only_mask(inputs, parts, cfg)
    assert mask.sum() == 140


def test_decoder_dead_outside_state_machine_stays_protected():
    cfg, inputs, parts = scene()
    parts["labels"][10:20] = "D"
    mask, audit = gap_dead.gap_only_mask(inputs, parts, cfg)
    assert not mask.any()
    assert "outside_stoppage" in audit["segments"][0]["support"]


def test_zero_gap_weight_cannot_claim_an_eligible_interval():
    cfg, inputs, parts = scene()
    cfg = replace(cfg, W_GAP_STOP=0.0)
    parts = result(inputs, cfg)
    mask, audit = gap_dead.gap_only_mask(inputs, parts, cfg)
    assert not mask.any()
    assert audit["segment_count"] == 0


@pytest.mark.parametrize("failure", ["labels", "evidence", "blocks", "holds"])
def test_invalid_inputs_fail_closed(failure):
    cfg, inputs, parts = scene()
    if failure == "labels":
        parts["labels"] = parts["labels"][:-1]
    elif failure == "evidence":
        inputs["e_dead"][10] = np.nan
    elif failure == "blocks":
        inputs["blocks"] = [(1, 0, 100), (2, 100, 239)]
    else:
        inputs["hold_rows"] = pd.DataFrame([(20, "keeper_hold;end=999")],
                                           columns=["frame", "notes"])
    with pytest.raises(ValueError):
        gap_dead.gap_only_mask(inputs, parts, cfg)


def artifact():
    """A valid synthetic model with no fitted tracking information."""
    size = len(gap_dead.FEATURE_NAMES)
    return {"format": gap_dead.FORMAT, "feature_names": list(gap_dead.FEATURE_NAMES),
            "feature_spec": {"min_players": 1}, "means": [0.0] * size,
            "scales": [1.0] * size, "coefficients": [0.0] * size, "intercept": 0.0,
            "threshold": 0.8, "enabled": True,
            "training": {"games": ["METRICA-S1"],
                         "reference_sha256": {"METRICA-S1": "a" * 64},
                         "dataset_sha256": "b" * 64}}


def test_features_are_team_symmetric_and_blocked():
    base = np.arange(120 * 2 * 10, dtype=float).reshape(120, 2, 10)
    base[10, 0, 4] = np.nan
    blocks = [(1, 0, 49), (2, 50, 119)]
    first, names = gap_dead.features_from_teams(base, blocks, 25)
    swapped, _ = gap_dead.features_from_teams(base[:, ::-1], blocks, 25)
    np.testing.assert_array_equal(first, swapped)
    assert names == list(gap_dead.FEATURE_NAMES)
    assert np.isfinite(first).all()
    altered = base.copy()
    altered[50:] *= 9
    second, _ = gap_dead.features_from_teams(altered, blocks, 25)
    np.testing.assert_array_equal(first[:50], second[:50])


def test_complete_run_requires_agreeing_neighbors_and_never_changes_live_labels():
    labels = np.array(list("HDDDHADDDHHDDDAX"))
    eligible = labels == "D"
    logits = np.full(len(labels), 10.0)
    result = gap_dead.correct(labels, logits, eligible, [(1, 0, len(labels) - 1)], 0.8)
    assert "".join(result) == "HHHHHADDDHHDDDAX"
    assert np.array_equal(result[labels != "D"], labels[labels != "D"])


def test_partial_islands_and_block_edges_stay_dead():
    labels = np.array(list("HDDDDH"))
    mask = np.array([False, True, True, False, True, False])
    result = gap_dead.correct(labels, np.full(6, 10.0), mask, [(1, 0, 5)], 0.8)
    np.testing.assert_array_equal(result, labels)
    result = gap_dead.correct(labels, np.full(6, 10.0), labels == "D",
                              [(1, 0, 2), (2, 3, 5)], 0.8)
    np.testing.assert_array_equal(result, labels)


def test_mean_logit_controls_whole_run_and_tie_retains_dead():
    labels = np.array(list("HDDDH"))
    threshold = 0.8
    cutoff = np.log(threshold / (1 - threshold))
    mask = labels == "D"
    for logits in (np.full(5, cutoff), np.array([0.0, 10.0, -10.0, 0.0, 0.0])):
        np.testing.assert_array_equal(gap_dead.correct(labels, logits, mask, [(1, 0, 4)], threshold), labels)
    np.testing.assert_array_equal(gap_dead.correct(labels, np.full(5, 10.0), mask,
                                                   [(1, 0, 4)], threshold, enabled=False), labels)


def test_frozen_model_scaling_and_hash_round_trip(tmp_path):
    model = artifact()
    model["means"][0], model["scales"][0], model["coefficients"][0] = 3.0, 2.0, 4.0
    model["intercept"] = 1.0
    features = np.zeros((2, len(gap_dead.FEATURE_NAMES)))
    features[:, 0] = [3.0, 5.0]
    np.testing.assert_allclose(gap_dead.predict_log_odds(features, model), [1.0, 5.0])
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model))
    loaded = gap_dead.load_artifact(path)
    assert gap_dead.artifact_hash(model) == gap_dead.artifact_hash(loaded)


@pytest.mark.parametrize("field,value", [("threshold", 1.0), ("threshold", float("nan")),
                                         ("enabled", 1), ("selection", []),
                                         ("selection", {"bad": float("inf")}),
                                         ("intercept", True)])
def test_model_contract_rejects_invalid_fields(field, value):
    model = artifact()
    model[field] = value
    with pytest.raises(ValueError):
        gap_dead.validate_artifact(model)


def test_model_rejects_duplicate_keys_and_nonfinite_metadata(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"format": "one", "format": "two"}')
    with pytest.raises(ValueError, match="duplicate"):
        gap_dead.load_artifact(path)
    model = artifact()
    model["selection"] = {"bad": 1e300}
    text = json.dumps(model).replace("1e+300", "1e1000")
    path.write_text(text)
    with pytest.raises(ValueError, match="finite"):
        gap_dead.load_artifact(path)


def test_disabled_apply_never_needs_player_features():
    cfg, inputs, parts = scene()
    model = artifact()
    model["enabled"] = False
    output, audit = gap_dead.apply(parts["labels"], inputs, parts, model, cfg)
    np.testing.assert_array_equal(output, parts["labels"])
    assert audit["actionable_runs"] == 1
    assert audit["changed_frames"] == 0


@pytest.mark.parametrize("threshold", (0.8, 0.95))
@pytest.mark.parametrize("length", (3, 25, 1001))
def test_exact_cutoff_remains_dead_at_different_run_lengths(threshold, length):
    import math
    labels = np.array(list("H" + "D" * length + "H"))
    cutoff = math.log(threshold / (1.0 - threshold))
    logits = np.full(len(labels), cutoff)
    predicted = gap_dead.correct(labels, logits, labels == "D", [(1, 0, len(labels) - 1)], threshold)
    np.testing.assert_array_equal(predicted, labels)


def test_enabled_apply_restores_a_whole_synthetic_run_and_preserves_sources(monkeypatch):
    cfg, inputs, parts = scene()
    model = artifact()
    model["intercept"] = 10.0
    inputs["tactical"] = {"synthetic": True}
    monkeypatch.setattr(gap_dead, "feature_matrix", lambda *args: (
        np.zeros((inputs["n_frames"], len(gap_dead.FEATURE_NAMES))), list(gap_dead.FEATURE_NAMES)))
    baseline = parts["labels"].copy()
    predicted, audit = gap_dead.apply(baseline, inputs, parts, model, cfg)
    assert (predicted == "H").all()
    assert audit["removed_runs"] == 1 and audit["changed_frames"] == 140
    np.testing.assert_array_equal(parts["labels"], baseline)


def test_nonfinite_predictions_are_rejected():
    model = artifact()
    model["coefficients"][0] = 1e308
    features = np.full((2, len(gap_dead.FEATURE_NAMES)), 1e308)
    with pytest.raises(ValueError, match="nonfinite"):
        gap_dead.predict_log_odds(features, model)


def test_artifact_provenance_is_generic_but_requires_exact_source_hashes():
    model = artifact()
    model["training"]["games"] = ["EXTERNAL-SYNTHETIC"]
    model["training"]["reference_sha256"] = {"EXTERNAL-SYNTHETIC": "a" * 64}
    gap_dead.validate_artifact(model)
    model["training"]["reference_sha256"]["EXTERNAL-SYNTHETIC"] = "a" * 63
    with pytest.raises(ValueError, match="SHA-256"):
        gap_dead.validate_artifact(model)


def test_enabled_identity_model_consumes_real_engine_parts(tmp_path):
    import os
    import games
    from engine import submission
    assets = games.asset_paths()
    if not all(os.path.isfile(assets[k]) for k in ("atd_meta", "atd_raw", "homography", "smart_tagging")):
        pytest.skip("match data bundle not present")
    # The rejected gap model (P10-C) predates ONSET_KNEE and rebuilds the stoppage without it. It
    # acts on the rule engine's labels, so the learned stage is off (the default since r6e-adopt).
    cfg = replace(Config.default(), ONSET_KNEE=False, LEARNED_MODE="off")
    inputs = possession.prepare(cfg)
    baseline, parts = possession.run(cfg, inputs=inputs, return_parts=True)
    model = artifact()
    model["feature_spec"]["min_players"] = cfg.SHAPE_MIN_PLAYERS
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(model))
    predicted, audit = gap_dead.apply(baseline, inputs, parts, gap_dead.load_artifact(model_path), cfg)
    np.testing.assert_array_equal(predicted, baseline)
    assert audit["enabled"] and audit["actionable_runs"] > 0
    assert audit["changed_frames"] == 0
    from tools.gap_models import apply as gap_apply
    routed, routed_audit = gap_apply.correct("dead", gap_dead.load_artifact(model_path), cfg,
                                             inputs=inputs)
    np.testing.assert_array_equal(routed, baseline)
    assert routed_audit == audit
    destination = tmp_path / "identity.csv"
    submission.write_submission(predicted, str(destination), n_frames=len(predicted), blocks=inputs["blocks"])
    assert "OK" in submission.validate(str(destination), rows=len(predicted))
