"""Synthetic contracts for the P10-C owner correction in `tools/gap_models/`."""

import copy
import itertools
import json
import os

import numpy as np
import pandas as pd
import pytest

import games
from engine import events, possession
from engine.config import Config
from tools.gap_models import apply as gap_apply
from tools.gap_models import gap_owner as go


def tactical(n=180):
    """Two six-player teams with distinct geometry and smooth movement."""
    frame = np.arange(n)[:, None]
    x = np.tile(np.r_[np.linspace(.2, .4, 6), np.linspace(.6, .8, 6)], (n, 1))
    x[:, :6] += frame * .0002
    x[:, 6:] += frame * .0001
    y = np.tile(np.r_[np.linspace(.1, .8, 6), np.linspace(.2, .9, 6)], (n, 1))
    return {"px": x, "py": y, "team_of_slot": np.repeat([0, 1], 6),
            "bx": np.zeros(n), "by": np.zeros(n), "ps": np.zeros_like(x)}


def artifact():
    """A zero-weight model with a complete synthetic training identity."""
    return {"format": go.FORMAT, "feature_names": list(go.FEATURE_NAMES),
            "feature_spec": {"min_players": 6}, "means": [0.0] * len(go.FEATURE_NAMES),
            "scales": [1.0] * len(go.FEATURE_NAMES),
            "coefficients": [0.0] * len(go.FEATURE_NAMES),
            "decoder": {"switch_cost": 2.0, "baseline_log_odds": .5, "endpoint_log_odds": 2.0},
            "training": {"games": ["SYNTH"], "reference_sha256": {"SYNTH": "a" * 64},
                         "dataset_sha256": "b" * 64}}


def parts(n=180):
    return {"tactical": tactical(n), "pitch": (105.0, 68.0), "fps": 25.0,
            "blocks": [(1, 0, n - 1)], "kickoffs": {1: 0},
            "ball": {"visible": np.ones(n, dtype=bool)},
            "ghost": np.zeros(n, dtype=bool), "holds": None,
            "events": pd.DataFrame({"frame": []})}


def swap(labels):
    labels = np.asarray(labels)
    return np.where(labels == "H", "A", np.where(labels == "A", "H", labels))


def test_features_are_player_only_and_team_antisymmetric():
    track = tactical()
    blocks = [(1, 0, 79), (2, 100, 179)]
    expected, names = go.feature_matrix(track, blocks, (105, 68), 25)
    assert expected.shape == (180, 100)
    assert names == list(go.FEATURE_NAMES)
    assert np.isfinite(expected).all()
    unrelated = dict(track, bx=np.full(180, np.nan), by=np.full(180, 1000),
                     ps=np.full(track["px"].shape, 1000))
    same, _ = go.feature_matrix(unrelated, blocks, (105, 68), 25)
    np.testing.assert_array_equal(same, expected)
    flipped = dict(track, team_of_slot=1 - track["team_of_slot"])
    reversed_features, _ = go.feature_matrix(flipped, blocks, (105, 68), 25)
    np.testing.assert_array_equal(reversed_features, -expected)
    assert not expected[80:100].any()


def test_public_team_features_preserve_owner_feature_construction():
    track = tactical()
    track["px"][60, 0] = np.nan
    blocks = [(1, 0, 79), (2, 100, 179)]
    base, names = go.team_feature_matrix(track, blocks, (105, 68), 25)
    assert base.shape == (180, 2, 10)
    assert names == list(go.BASE_FEATURES)
    assert np.isnan(base[80:100]).all()
    swapped, _ = go.team_feature_matrix(dict(track, team_of_slot=1 - track["team_of_slot"]),
                                        blocks, (105, 68), 25)
    np.testing.assert_array_equal(swapped, base[:, ::-1])
    features, _ = go.feature_matrix(track, blocks, (105, 68), 25)
    expected = np.zeros_like(features)
    for frame in range(180):
        containing = [row for row in blocks if row[1] <= frame <= row[2]]
        if not containing:
            continue
        _, first, last = containing[0]
        for index, context in enumerate(go.CONTEXT_SECONDS):
            source = frame + context * 25
            if not first <= source <= last:
                continue
            finite = np.isfinite(base[source])
            values = np.where(finite, base[source], 0)
            expected[frame, index * 20:index * 20 + 20:2] = values[0] - values[1]
            expected[frame, index * 20 + 1:index * 20 + 20:2] = (
                finite[0].astype(float) - finite[1].astype(float))
    np.testing.assert_array_equal(features, expected)


def test_context_and_derivatives_do_not_cross_halves_or_ungraded_frames():
    track = tactical()
    blocks = [(1, 0, 79), (2, 80, 179)]
    expected, names = go.feature_matrix(track, blocks, (105, 68), 25)
    changed = copy.deepcopy(track)
    changed["px"][80:] *= .4
    changed["py"][80:] *= .4
    actual, _ = go.feature_matrix(changed, blocks, (105, 68), 25)
    np.testing.assert_array_equal(actual[:80], expected[:80])
    assert all(expected[79, i] == 0 for i, name in enumerate(names) if name.endswith("t+1s"))
    assert all(expected[80, i] == 0 for i, name in enumerate(names) if name.endswith("t-1s"))


def test_missing_players_have_explicit_indicators_without_filling_geometry():
    track = tactical()
    track["px"][90, 0] = np.nan
    features, names = go.feature_matrix(track, [(1, 0, 179)], (105, 68), 25)
    assert features[90, names.index("centroid_forward_m_observed_t+0s")] == -1
    assert features[90, names.index("visible_count_value_t+0s")] == -1
    assert features[90, names.index("visible_count_observed_t+0s")] == 0
    assert features[89, names.index("centroid_forward_m_observed_t+0s")] == 0
    assert np.isfinite(features).all()
    flipped = dict(track, team_of_slot=1 - track["team_of_slot"])
    other, _ = go.feature_matrix(flipped, [(1, 0, 179)], (105, 68), 25)
    np.testing.assert_array_equal(other, -features)


def test_no_players_or_unresolved_direction_is_finite():
    track = tactical()
    track["px"][:] = np.nan
    features, _ = go.feature_matrix(track, [(1, 0, 179)], (105, 68), 25)
    assert not features.any()
    track = tactical()
    track["px"][:, 6:] = track["px"][:, :6]
    features, names = go.feature_matrix(track, [(1, 0, 179)], (105, 68), 25)
    assert np.isfinite(features).all()
    assert features[90, names.index("centroid_forward_m_value_t+0s")] == 0


@pytest.mark.parametrize("blocks", [[(1, -1, 20)], [(1, 0, 180)], [(1, 20, 10)],
                                   [(1, 0, 100), (2, 100, 179)], [(1, 0., 179)],
                                   [(1, 0, 79), (1, 80, 179)]])
def test_bad_blocks_rejected(blocks):
    with pytest.raises(ValueError, match="blocks"):
        go.feature_matrix(tactical(), blocks, (105, 68), 25)


@pytest.mark.parametrize("minimum", [0, -1, True, 1.5])
def test_bad_minimum_rejected(minimum):
    with pytest.raises(ValueError, match="min_players"):
        go.feature_matrix(tactical(), [(1, 0, 179)], (105, 68), 25, minimum)


def test_eligibility_protects_all_physical_masks_and_half_edges():
    context = parts()
    context["blocks"] = [(1, 10, 79), (2, 100, 179)]
    context["kickoffs"] = {1: 20, 2: 110}
    context["ball"]["visible"][:] = False
    context["ball"]["visible"][25] = True
    context["ghost"][30:35] = True
    hold = np.zeros(180, dtype=bool)
    hold[40:45] = True
    context["holds"] = (hold, np.zeros(180, dtype=bool))
    context["hold_rows"] = pd.DataFrame(
        [(120, "control", "H", .9, 1., "keeper_hold;end=125")], columns=events.COLUMNS)
    context["events"] = pd.DataFrame({"frame": [60]})
    context["raw_events"] = pd.DataFrame({"frame": [150]})
    labels = np.full(180, "H")
    labels[115:118] = "D"
    labels[130] = "X"
    mask = go.eligible_mask(labels, context)
    expected = np.zeros(180, dtype=bool)
    expected[20:80] = True
    expected[110:180] = True
    expected[25] = False
    expected[30:35] = False
    expected[40:45] = False
    expected[48:73] = False
    expected[115:118] = False
    expected[120:126] = False
    expected[130] = False
    expected[138:163] = False
    np.testing.assert_array_equal(mask, expected)


def test_decoder_preserves_masks_dead_unknown_and_exact_zero_frames():
    labels = np.array(list("XHHHHDDHHAAAX"))
    mask = np.ones(len(labels), dtype=bool)
    mask[3] = False
    odds = np.full(len(labels), -10.0)
    odds[8] = 0
    decoder = {"switch_cost": 0., "baseline_log_odds": 0., "endpoint_log_odds": 20.}
    result = go.correct(labels, odds, mask, [(1, 1, 11)], decoder)
    assert result[3] == labels[3]
    assert result[8] == labels[8]
    np.testing.assert_array_equal(result[np.isin(labels, ("D", "X"))], labels[np.isin(labels, ("D", "X"))])
    assert np.any(result != labels)
    flipped = go.correct(swap(labels), -odds, mask, [(1, 1, 11)], decoder)
    np.testing.assert_array_equal(flipped, swap(result))


def test_zero_model_is_identity_even_with_disagreeing_endpoints():
    labels = np.array(list("HHHAAAHHHAAA"))
    result = go.correct(labels, np.zeros(len(labels)), np.ones(len(labels), dtype=bool),
                        [(1, 0, 11)], artifact()["decoder"])
    np.testing.assert_array_equal(result, labels)


def test_decoder_matches_brute_force_minimum_and_tie_symmetry():
    labels = np.array(list("HHAHAA"))
    mask = np.array([False, True, True, True, True, False])
    odds = np.array([0, -.7, .4, 1.5, -.4, 0])
    decoder = {"switch_cost": .6, "baseline_log_odds": .2, "endpoint_log_odds": .8}
    actual = go.correct(labels, odds, mask, [(1, 0, 5)], decoder)
    adjusted = odds[1:5] + np.where(labels[1:5] == "H", .2, -.2)
    adjusted[0] += .8
    adjusted[-1] -= .8
    def cost(path):
        signs = np.where(np.array(path) == "H", 1, -1)
        return float(np.maximum(-adjusted * signs, 0).sum() +
                     .6 * sum(a != b for a, b in zip(path[:-1], path[1:])))
    best = min(cost(path) for path in itertools.product(("H", "A"), repeat=4))
    assert cost(actual[1:5]) == pytest.approx(best)
    np.testing.assert_array_equal(go.correct(swap(labels), -odds, mask, [(1, 0, 5)], decoder),
                                  swap(actual))
    # Model evidence and baseline prior cancel exactly, so baseline decides every tie.
    tie = go.correct(labels, -np.where(labels == "H", .2, -.2), np.ones(6, bool),
                     [(1, 0, 5)], dict(decoder, switch_cost=0., endpoint_log_odds=0.))
    np.testing.assert_array_equal(tie, labels)


def test_disabled_apply_needs_no_features_or_files():
    labels = np.array(list("HADXX"))
    result, audit = go.apply(labels, {}, None)
    np.testing.assert_array_equal(result, labels)
    assert result is not labels
    assert audit == {"enabled": False, "eligible_frames": 0, "changed_frames": 0,
                     "changed_intervals": 0}


def test_apply_uses_supplied_inputs_and_protects_anchors():
    context = parts()
    context["ball"]["visible"][60:120] = False
    context["events"] = pd.DataFrame({"frame": [90]})
    labels = np.full(180, "A")
    model = artifact()
    # Distinct longitudinal spreads provide constant player evidence in this synthetic scene.
    context["tactical"]["px"][:, :6] *= .5
    model["coefficients"][go.FEATURE_NAMES.index("longitudinal_spread_m_value_t+0s")] = -10.
    result, audit = go.apply(labels, context, model)
    mask = go.eligible_mask(labels, context)
    assert audit["enabled"] and audit["changed_frames"] > 0
    assert audit["eligible_frames"] == int(mask.sum())
    np.testing.assert_array_equal(result[~mask], labels[~mask])
    np.testing.assert_array_equal(result[78:103], labels[78:103])
    assert audit["artifact_sha256"] == go.artifact_hash(model)
    flipped_context = copy.deepcopy(context)
    flipped_context["tactical"]["team_of_slot"] = 1 - context["tactical"]["team_of_slot"]
    flipped, other_audit = go.apply(swap(labels), flipped_context, model)
    np.testing.assert_array_equal(flipped, swap(result))
    assert audit["changed_frames"] == other_audit["changed_frames"]


@pytest.mark.parametrize("field,value", [
    ("format", "other"), ("feature_names", []), ("means", [1.] * 100),
    ("scales", [0.] * 100), ("coefficients", [float("nan")] * 100),
    ("coefficients", [True] * 100), ("coefficients", [0.] * 99),
    ("feature_spec", {"min_players": True}), ("selection", []),
    ("decoder", {"switch_cost": -1., "baseline_log_odds": .5, "endpoint_log_odds": 2.}),
    ("training", {"games": ["SYNTH"], "reference_sha256": {}, "dataset_sha256": "b" * 64}),
])
def test_malformed_artifacts_rejected(field, value):
    model = artifact()
    model[field] = value
    with pytest.raises(ValueError):
        go.validate_artifact(model)


def test_artifact_load_hash_and_unknown_fields(tmp_path):
    model = artifact()
    model["selection"] = {"identity": True, "fold": "SYNTH2"}
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model))
    assert go.load_artifact(path) == model
    digest = go.artifact_hash(model)
    assert len(digest) == 64
    assert go.artifact_hash(dict(reversed(list(model.items())))) == digest
    model["unexpected"] = 1
    with pytest.raises(ValueError, match="unknown"):
        go.validate_artifact(model)
    path.write_text('{"format": "one", "format": "two"}')
    with pytest.raises(ValueError, match="duplicate"):
        go.load_artifact(path)
    path.write_text('{"format": NaN}')
    with pytest.raises(ValueError, match="non-finite"):
        go.load_artifact(path)


def test_artifact_rejects_unsorted_games_and_nonfinite_metadata():
    model = artifact()
    model["training"]["games"] = ["Z", "A"]
    with pytest.raises(ValueError, match="sorted"):
        go.validate_artifact(model)
    model = artifact()
    model["selection"] = {"loss": float("inf")}
    with pytest.raises(ValueError, match="finite JSON"):
        go.validate_artifact(model)


def test_model_prediction_has_exact_team_symmetry_and_rejects_bad_features():
    model = artifact()
    model["coefficients"] = np.linspace(-1, 1, 100).tolist()
    model["scales"] = np.linspace(.5, 2, 100).tolist()
    features, _ = go.feature_matrix(tactical(), [(1, 0, 179)], (105, 68), 25)
    odds = go.predict_log_odds(features, model)
    np.testing.assert_array_equal(go.predict_log_odds(-features, model), -odds)
    with pytest.raises(ValueError, match="finite matrix"):
        go.predict_log_odds(np.full((5, 100), np.nan), model)
    with pytest.raises(ValueError, match="finite matrix"):
        go.predict_log_odds(np.ones((5, 99)), model)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), True, -1])
def test_bad_decoder_costs_rejected(value):
    decoder = artifact()["decoder"]
    decoder["switch_cost"] = value
    with pytest.raises(ValueError, match="finite and nonnegative"):
        go.correct(np.array(list("HHHH")), np.ones(4), np.ones(4, bool), [(1, 0, 3)], decoder)


_ASSETS = games.asset_paths()
HAVE_DATA = all(os.path.isfile(_ASSETS[key])
                for key in ("atd_meta", "atd_raw", "homography", "smart_tagging"))


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_DATA, reason="match data bundle not present")
def test_enabled_identity_model_consumes_real_engine_parts_without_changing_b0():
    cfg = Config.default()
    inputs = possession.prepare(cfg)
    baseline, decoded = possession.run(cfg, inputs=inputs, return_parts=True)
    assert inputs["hold_rows"] is not None and len(inputs["hold_rows"]) > 0
    result, audit = go.apply(baseline, dict(inputs, **decoded), artifact(), cfg)
    np.testing.assert_array_equal(result, baseline)
    assert audit["enabled"] and audit["eligible_frames"] > 0
    assert audit["changed_frames"] == 0
    integrated, integrated_audit = gap_apply.correct("owner", artifact(), cfg, inputs=inputs)
    np.testing.assert_array_equal(integrated, baseline)
    assert integrated_audit == audit
