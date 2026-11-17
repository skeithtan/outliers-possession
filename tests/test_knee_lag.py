"""Synthetic tests for tools/analysis/knee_lag.py (COMPONENT_KNEE.md Section 3). No match data."""

import numpy as np

from tools.analysis import knee_lag


def _labels(text):
    return np.array(list(text))


def test_reference_onsets_are_d_frames_after_a_live_frame():
    labels = _labels("XXDDHHHDDAADDXDD")
    assert knee_lag.reference_onsets(labels).tolist() == [7, 11]


def test_an_entry_inside_a_reference_stoppage_pairs_with_its_start():
    labels = _labels("HHHHDDDDDDAA")
    onsets = knee_lag.reference_onsets(labels)
    assert knee_lag.pair_onset(labels, onsets, 7, 10) == 4


def test_a_stoppage_that_follows_x_frames_is_no_onset():
    labels = _labels("XXDDDDHH")
    onsets = knee_lag.reference_onsets(labels)
    assert knee_lag.pair_onset(labels, onsets, 3, 6) is None


def test_an_entry_in_live_play_pairs_with_the_next_onset_before_leave():
    labels = _labels("HHHHHHDDDDAA")
    onsets = knee_lag.reference_onsets(labels)
    assert knee_lag.pair_onset(labels, onsets, 2, 9) == 6
    assert knee_lag.pair_onset(labels, onsets, 2, 6) is None


def test_stoppage_end_is_the_first_frame_after_the_run():
    stoppage = np.array([False, True, True, True, False, True])
    assert knee_lag.stoppage_end(stoppage, 1) == 4
    assert knee_lag.stoppage_end(stoppage, 5) == 6


def test_summary_pools_metrica_over_views_and_rounds_half_to_even():
    records = [
        {"game": "METRICA-S1", "view": "raw", "lags": [-20, -10], "gap_entries": 2,
         "no_knee": 0, "unpaired": 0, "bounds": {}, "lag_quantiles": {}, "entry_minus_onset": {}},
        {"game": "METRICA-S2", "view": "matched", "lags": [-15, -14], "gap_entries": 2,
         "no_knee": 0, "unpaired": 0, "bounds": {}, "lag_quantiles": {}, "entry_minus_onset": {}},
        {"game": "DFL-J03WMX", "view": "raw", "lags": [90], "gap_entries": 1, "no_knee": 0,
         "unpaired": 0, "bounds": {}, "lag_quantiles": {}, "entry_minus_onset": {}},
    ]
    out = knee_lag.summarise(records)
    assert out["pooled"]["metrica|all"]["median"] == -14.5
    assert out["DELTA_KNEE"] == -14
    assert out["pooled"]["dfl|all"]["n"] == 1
