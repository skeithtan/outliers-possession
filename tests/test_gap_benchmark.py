"""Synthetic contracts for the label-scored gap benchmark."""

import json

import numpy as np
import pytest

from tools import gap_benchmark as gb
from tools import gapmask as gm

H, A, D, X = gb.pm.H, gb.pm.A, gb.pm.D, gb.pm.X


def scene():
    blocks = [(1, 10, 399), (2, 450, 849)]
    codes = np.full(900, X, dtype=np.int8)
    for _, a, b in blocks:
        codes[a:b + 1] = H
        codes[a + 40:a + 120] = D
        codes[a + 200:a + 240] = D
        codes[a + 280:b + 1] = A
    missing = np.zeros(900, bool)
    for a, b in ((60, 90), (115, 145), (170, 190), (205, 230), (520, 540)):
        missing[a:b] = True
    profile = gb.build_profile(missing, blocks, 10, codes)
    return blocks, codes, missing, profile


def test_profile_is_frozen_from_observations_and_engine_labels():
    blocks, codes, missing, profile = scene()
    gb.check_profile(profile)
    assert set(profile["class_counts"]) == set(gb.CLASSES)
    assert profile["observed"]["missing_frames"] == missing.sum()
    assert profile["observed"]["missing_fraction"] == pytest.approx(missing.sum() / 790)
    assert {e["class"] for e in profile["entries"]} == set(gb.CLASSES)
    assert gb.build_profile(missing, blocks, 10, codes) == profile
    profile["entries"][0]["duration_s"] = 0.1
    with pytest.raises(ValueError, match="checksum"):
        gb.check_profile(profile)


@pytest.mark.parametrize("view", gb.VIEWS)
def test_masks_are_deterministic_and_use_graded_denominator(view):
    blocks, codes, missing, profile = scene()
    first = gb.draw_mask(missing, blocks, 10, codes, profile, "synthetic", view, 0, 0.15)
    assert first == gb.draw_mask(missing, blocks, 10, codes, profile, "synthetic", view, 0, 0.15)
    assert first["target_frames"] == 119
    assert first["reached"]
    requested = gm.hidden_mask(len(codes), first["starts"], first["lengths"])
    assert first["hidden_frames"] == int((requested & ~missing).sum())
    assert not requested[codes == X].any()
    for start, length in zip(first["starts"], first["lengths"]):
        assert any(a <= start - gm.MARGIN and start + length + gm.MARGIN - 1 <= b
                   for _, a, b in blocks)
    if view != "U":
        assert first["classes"] == [gb.gap_class(codes, s, s + n)
                                    for s, n in zip(first["starts"], first["lengths"])]


def test_uniform_view_does_not_use_teams_or_dead_states():
    blocks, codes, missing, profile = scene()
    other = codes.copy()
    other[codes != X] = A
    a = gb.draw_mask(missing, blocks, 10, codes, profile, "synthetic", "U", 1)
    b = gb.draw_mask(missing, blocks, 10, other, profile, "synthetic", "U", 1)
    assert a["starts"] == b["starts"] and a["lengths"] == b["lengths"]


def test_masks_work_without_native_gaps_and_reject_unplaceable_classes():
    blocks, codes, missing, profile = scene()
    missing[:] = False
    mask = gb.draw_mask(missing, blocks, 10, codes, profile, "synthetic", "S", 2)
    assert mask["reached"] and mask["hidden_frames"] == sum(mask["lengths"])
    codes[codes != X] = H
    mask = gb.draw_mask(missing, blocks, 10, codes, profile, "synthetic", "S", 2,
                        fraction=0.95, max_draws=50)
    assert mask["rejected"] > 0 and mask["draws"] == 50 and not mask["reached"]


def test_saved_mask_rejects_changed_inputs_and_tampering(tmp_path):
    blocks, codes, missing, profile = scene()
    path = tmp_path / "mask.json"
    first = gb.frozen_mask(path, missing, blocks, 10, codes, profile, "synthetic", "U", 0)
    assert first == gb.frozen_mask(path, missing, blocks, 10, codes, profile, "synthetic", "U", 0)
    with pytest.raises(ValueError, match="placement"):
        gb.frozen_mask(path, missing, blocks, 10, codes, profile, "synthetic", "U", 1)
    saved = json.loads(path.read_text())
    saved["starts"][0] += 1
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="content"):
        gb.frozen_mask(path, missing, blocks, 10, codes, profile, "synthetic", "U", 0)


def test_touching_blocks_never_merge_masks():
    blocks, codes, missing, profile = scene()
    blocks = [(1, 0, 49), (2, 50, 99)]
    codes, missing = np.zeros(100, np.int8), np.zeros(100, bool)
    mask = gb.draw_mask(missing, blocks, 10, codes, profile, "synthetic", "U", 3)
    assert all(s + n <= 50 or s >= 50 for s, n in zip(mask["starts"], mask["lengths"]))


def test_d_and_x_are_owner_errors_not_excluded_cases():
    gt = np.array([H, H, A, A, D, X])
    pred = np.array([H, D, X, H, D, H])
    metrics = gb.frame_metrics(gt, pred, gt != X)
    assert metrics["reference_live_frames"] == 4
    assert metrics["false_d_rate"] == 0.25
    assert metrics["ownership_accuracy"] == 0.25
    assert metrics["ownership_accuracy_given_alive"] == 0.5


def test_score_uses_reference_not_clean_prediction_and_counts_outside_changes():
    gt = np.array([X] + [H] * 40 + [A] * 40 + [D] * 40 + [H] * 40 + [X])
    clean = gt.copy()
    clean[15:25] = D
    masked = gt.copy()
    masked[35:48] = D
    masked[155:158] = A
    missing = np.zeros(len(gt), bool)
    mask = {"starts": [30], "lengths": [25]}
    result = gb.score_cell(gt, masked, clean, mask, missing, [(1, 1, 160)], 25, draws=10)
    assert result["score"]["spell_f1_mean"] == gb.pm.evaluate(gt, masked)["spell_f1_mean"]
    assert result["true_hidden_turnovers"]["n"] == 1
    assert result["masked"]["reference_live_frames"] == 25
    assert result["changes"]["outside_changed"] == 13
    assert result["masked"]["false_d_rate"] == 13 / 25
    assert result["missing_after"]["missing_fraction"] == 25 / 160


def test_recovery_does_not_cross_a_reference_boundary():
    gt = np.array([H] * 50 + [A] * 50)
    pred = np.array([D] * 50 + [A] * 50)
    missing = np.zeros(100, bool)
    missing[20:40] = True
    result = gb.recovery_metrics(gt, pred, missing, missing, [(1, 0, 99)], 5)
    assert result["censored"] == 1 and result["recovered"] == 0
    pred[43:50] = H
    result = gb.recovery_metrics(gt, pred, missing, missing, [(1, 0, 99)], 5)
    assert result["mean_latency_frames"] == 3


def test_mask_enters_before_feature_computation_and_loader_is_restored(monkeypatch):
    from engine import data, possession
    original = data.load_tactical
    tactical = {"bx": np.arange(6, dtype=float), "by": np.arange(6, dtype=float),
                "px": np.ones((6, 2))}
    hidden = np.array([False, False, True, True, False, False])

    def run(cfg, game):
        seen = data.load_tactical(game)
        assert np.isnan(seen["bx"][hidden]).all()
        assert np.isnan(seen["by"][hidden]).all()
        assert seen["px"] is tactical["px"]
        assert not np.isnan(tactical["bx"]).any()
        return np.array(["H"] * 6)

    monkeypatch.setattr(possession, "run", run)
    assert (gb.predict(tactical, hidden, "BAR-ATH", None) == H).all()
    assert data.load_tactical is original
    with pytest.raises(RuntimeError):
        with gb.masked_tactical(tactical, hidden, "BAR-ATH"):
            raise RuntimeError("test")
    assert data.load_tactical is original


def test_aggregation_averages_seeds_before_match_pairing():
    reports = []
    for game, current, off in (("DFL-J03WMX", [0.5] * 4, [0.7] * 4),
                               ("METRICA-S1", [0.5], [0.4])):
        for system, scores in (("current", current), ("gap_stop_off", off)):
            for seed, score in enumerate(scores):
                reports.append({"game": game, "system": system, "input_view": "raw",
                                "view": "U", "seed": seed, "profile_md5": "profile",
                                "engine_md5": "engine", "config_hash": system, "mask_md5": str(seed),
                                "reference_md5": game, "metrics": {
                                    "score": {"spell_f1_mean": score},
                                    "paired_vs_clean": {"delta": score - 0.5},
                                    "masked": {"false_d_rate": 0.1, "ownership_accuracy": 0.8},
                                    "missing_after": {"missing_fraction": 0.4}}})
    result = gb.aggregate(reports)
    total = next(r for r in result["paired_match_intervals"] if r["provider"] == "all")
    assert total["matches"] == 2 and total["mean"] == pytest.approx(0.05)
    assert result["cells"] == 10
    with pytest.raises(ValueError, match="duplicate"):
        gb.aggregate(reports + [reports[0]])


def test_cli_refuses_sealed_matches():
    with pytest.raises(SystemExit):
        gb.main(["run", "--game", "DFL-J03WQQ", "--gt", "unused", "--profile", "unused",
                 "--out-dir", "/tmp/unused"])


def _paired_reports():
    reports = []
    for system in gb.SYSTEMS:
        reports.append({"game": "DFL-J03WMX", "system": system, "input_view": "raw",
                        "view": "U", "seed": 0, "profile_md5": "profile",
                        "engine_md5": "engine", "config_hash": system, "mask_md5": "mask", "reference_md5": "reference",
                        "metrics": {"score": {"spell_f1_mean": 0.5},
                                    "paired_vs_clean": {"delta": -0.1},
                                    "masked": {"false_d_rate": 0.1, "ownership_accuracy": 0.8},
                                    "missing_after": {"missing_fraction": 0.4}}})
    return reports


@pytest.mark.parametrize("key", ["mask_md5", "profile_md5", "engine_md5", "reference_md5"])
def test_aggregation_rejects_mismatched_provenance(key):
    reports = _paired_reports()
    reports[1][key] = "different"
    with pytest.raises(ValueError):
        gb.aggregate(reports)


def test_aggregation_rejects_missing_system_or_unequal_seeds():
    reports = _paired_reports()
    with pytest.raises(ValueError, match="incomplete"):
        gb.aggregate(reports[:1])
    reports[1]["seed"] = 1
    with pytest.raises(ValueError, match="unequal seed"):
        gb.aggregate(reports)


def test_match_interval_rejects_zero_draws():
    with pytest.raises(ValueError, match="positive"):
        gb.match_interval([1, 2], draws=0)


def test_completion_checks_expected_cells_not_only_count():
    reports = []
    for game in gb.OPEN_GAMES:
        for view, seeds in (("clean", [None]), ("U", range(4)), ("S", range(4))):
            for seed in seeds:
                pair = _paired_reports()
                for report in pair:
                    report.update(game=game, view=view, seed=seed)
                reports.extend(pair)
    result = gb.aggregate(reports)
    assert result["cells"] == 144 and result["complete_primary_matrix"]
    for report in reports:
        if report["game"] == gb.OPEN_GAMES[-1] and report["view"] == "S":
            report["seed"] += 4
    result = gb.aggregate(reports)
    assert result["cells"] == 144 and not result["complete_primary_matrix"]
    assert result["completion_by_input_view"]["raw"]["missing_cells"] == 8


def test_aggregation_rejects_config_changes_within_system():
    reports = _paired_reports() + _paired_reports()
    for report in reports[2:]:
        report["seed"] = 1
    reports[-1]["config_hash"] = "changed"
    with pytest.raises(ValueError, match="configurations"):
        gb.aggregate(reports)


def test_new_spurious_spell_is_not_hidden_by_removing_an_old_error():
    gt = np.array([H] * 200, dtype=np.int8)
    clean, masked = gt.copy(), gt.copy()
    clean[20:30] = A
    masked[100:110] = A
    changes = gb.spurious_spell_changes(gt, masked, clean, [(0, 200)])
    assert changes["delta"] == 0
    assert changes["new"] == 2 and changes["removed"] == 2
    assert changes["persisted"] == 0
    same = gb.spurious_spell_changes(gt, clean, clean, [(0, 200)])
    assert same["new"] == 0 and same["removed"] == 0


def test_aggregation_rejects_changed_reference_between_seed_pairs():
    reports = _paired_reports() + _paired_reports()
    for report in reports[2:]:
        report["seed"] = 1
        report["reference_md5"] = "changed"
    with pytest.raises(ValueError, match="mixed references"):
        gb.aggregate(reports)


def test_required_existing_masks_never_draw_or_write(tmp_path, monkeypatch):
    blocks, codes, missing, profile = scene()
    path = tmp_path / "mask.json"
    with pytest.raises(ValueError, match="does not exist"):
        gb.frozen_mask(path, missing, blocks, 10, codes, profile, "synthetic", "U", 0,
                       require_existing=True)
    assert not path.exists()
    expected = gb.frozen_mask(path, missing, blocks, 10, codes, profile, "synthetic", "U", 0)
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise AssertionError("frozen reuse must never draw")

    monkeypatch.setattr(gb, "draw_mask", fail)
    actual = gb.frozen_mask(path, missing, blocks, 10, codes, profile, "synthetic", "U", 0,
                           require_existing=True)
    assert actual == expected and path.read_bytes() == before


def test_conditional_audit_excludes_reference_x_and_preserves_state_denominators():
    codes = np.array([X, H, H, D, D, X])
    missing = np.array([True, False, True, True, True, True])
    result = gb.conditioned_missingness(missing, codes, [(1, 0, 5)])
    assert result['graded_frames'] == 4
    assert result['missing_fraction'] == 0.75
    assert result['states']['alive']['missing_fraction'] == 0.5
    assert result['states']['dead']['missing_fraction'] == 1.0


def test_hide_only_audit_detects_irreducible_native_dead_excess():
    codes = np.array([H] * 100 + [D] * 100)
    source_missing = np.zeros(200, bool)
    source_missing[:10] = True
    source_missing[100:174] = True
    native_missing = np.zeros(200, bool)
    native_missing[100:195] = True
    source = gb.conditioned_missingness(source_missing, codes, [(1, 0, 199)])
    native = gb.conditioned_missingness(native_missing, codes, [(1, 0, 199)])
    result = gb.hide_only_feasibility(source, native)
    assert result['impossible_states'] == ['dead']
    assert result['states']['dead']['irreducible_excess_frames'] == 21
    assert result['states']['alive']['additional_frames'] == 10
    assert not result['hide_only_marginals_feasible']
    assert not result['accepted_as_source_simulation']


def test_marginal_feasibility_does_not_assert_observation_equivalence():
    codes = np.array([H] * 100 + [D] * 100)
    missing = np.zeros(200, bool)
    missing[:10] = True
    missing[100:174] = True
    source = gb.conditioned_missingness(missing, codes, [(1, 0, 199)])
    native = gb.conditioned_missingness(np.zeros(200, bool), codes, [(1, 0, 199)])
    result = gb.hide_only_feasibility(source, native)
    assert result['hide_only_marginals_feasible']
    assert result['states']['alive']['target_missing_frames'] == 10
    assert result['states']['dead']['target_missing_frames'] == 74
    assert result['conditional_targets_imply_fraction'] == 0.42
    assert result['status'] == 'marginally_feasible_not_validated'
    assert not result['accepted_as_source_simulation']


def test_missing_source_state_cannot_define_a_target():
    codes = np.array([H] * 10)
    source = gb.conditioned_missingness(np.zeros(10, bool), codes, [(1, 0, 9)])
    result = gb.hide_only_feasibility(source, source)
    assert result['states']['dead']['status'] == 'undefined_state_support'
    assert not result['hide_only_marginals_feasible']


def test_distribution_audit_reports_distances_without_fitted_tolerances():
    source = {'buckets': {'short': 1, 'long': 1}, 'class_counts': {'alive': 1, 'dead': 1},
              'missing_fraction': 0.4, 'median_s': 10, 'p90_s': 30}
    observed = {'buckets': {'short': 0, 'long': 2}, 'class_counts': {'alive': 2, 'dead': 0},
                'missing_fraction': 0.5, 'median_s': 20, 'p90_s': 40}
    result = gb.distribution_distance(source, observed)
    assert result['buckets_total_variation'] == 0.5
    assert result['class_counts_total_variation'] == 0.5
    assert result['missing_fraction_difference'] == pytest.approx(0.1)
    assert result['median_s_difference'] == 10


def test_observation_audit_uses_reference_coverage_for_frozen_masks():
    blocks = [(1, 0, 9), (2, 10, 19)]
    codes = np.array([X, X, H, H, X, H, H, H, X, X, X, A, A, D, D, X, X, X, X, X])
    assert gb.reference_blocks(codes, blocks) == [(1, 2, 3), (1, 5, 7), (2, 11, 14)]
    with pytest.raises(ValueError, match='no graded'):
        gb.reference_blocks(np.full(20, X), blocks)
