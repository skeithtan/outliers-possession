"""tools/analysis/external_settings.py on synthetic sequences. No match data."""

import numpy as np
import pytest

from tools.analysis import external_settings as es


def _labels(spec):
    """'<U1' labels from a list of (label, count) pairs."""
    return np.array(list("".join(label * count for label, count in spec)))


def test_best_on_grid_takes_the_smaller_value_on_a_tie():
    assert es.best_on_grid((1.0, 2.0, 3.0), [0.2, 0.5, 0.5]) == (2.0, 0.5)


def test_round_to_rounds_half_up():
    assert es.round_to(1.65, 0.1) == pytest.approx(1.7)
    assert es.round_to(4.25, 0.5) == pytest.approx(4.5)
    assert es.round_to(16.5, 1) == 17


def test_speed_thresholds_separates_a_slow_dead_ball():
    labels = _labels([("H", 50), ("D", 50)])
    meanv = np.r_[np.full(50, 3.0), np.full(50, 1.0)]
    dead, alive = es.speed_thresholds(meanv, labels, grid=(0.5, 2.0, 3.5))
    assert dead == [0.0, 1.0, 1.0]
    assert alive == [0.0, 0.0, 1.0]
    assert es.best_on_grid((0.5, 2.0, 3.5), es.youden_curve(dead, alive)) == (2.0, 1.0)


def test_speed_thresholds_ignores_x_and_nan():
    labels = _labels([("X", 10), ("D", 10), ("A", 10)])
    meanv = np.r_[np.zeros(10), np.full(5, np.nan), np.ones(5), np.full(10, 3.0)]
    dead, alive = es.speed_thresholds(meanv, labels, grid=(2.0,))
    assert dead == [1.0]
    assert alive == [0.0]


def test_reference_restarts_needs_d_before_alive():
    labels = _labels([("X", 2), ("H", 3), ("D", 2), ("A", 2), ("X", 1), ("H", 1)])
    assert es.reference_restarts(labels).tolist() == [7]


def test_control_frames_stay_clear_of_the_end_of_the_run():
    labels = _labels([("H", 5), ("D", 100), ("A", 5)])
    frames = es.control_frames(labels, 25.0, gap_s=2.0, step=10)
    assert frames.tolist() == [5, 15, 25, 35, 45, 55]


def test_moving_within_measures_the_wait():
    meanv = np.r_[np.zeros(10), np.full(5, 3.0), np.zeros(5)]
    assert es.moving_within([8, 16], meanv, 1.6, (1, 2)) == [0.0, 0.5]
    assert es.moving_within([], meanv, 1.6, (1,)) == [0.0]


def test_lookback_shares_counts_missing_offsets_in_the_total():
    assert es.lookback_shares([0, 2], 4, grid=(0, 1, 2)) == [0.25, 0.25, 0.5]
    assert es.lookback_shares([], 0, grid=(0,)) == [0.0]


def test_select_grid_flags_weak_rows():
    rows = [("a", (1.0, 2.0), [0.01, 0.02]), ("b", (1.0, 2.0), [0.03, 0.01]),
            ("c", (1.0, 2.0), [0.0, 0.5])]
    out = es.select_grid(rows, 0.5)
    assert out["weak_matches"] == 2
    assert out["inconclusive"] is True
    assert out["value"] == 2.0


def test_measure_refuses_the_competition_match_and_the_sealed_matches():
    for game in ("BAR-ATH", "DFL-J03WQQ", "DFL-J03WR9"):
        with pytest.raises(SystemExit):
            es.measure("stop", game, "ref.csv", "dir", {})
