"""Pure functions of the bucket b tools on synthetic inputs. No match data."""

import math

import numpy as np
import pytest

from tools.analysis import ghost_valley as gv
from tools.analysis import inertness_md5 as im
from tools.analysis import restart_confirm_modes as rc


# -- ghost_valley -------------------------------------------------------------------------------

def test_histogram_bins_and_overflow():
    counts, overflow = gv.histogram([0.0, 0.49, 0.5, 1.2, 29.99, 30.0, 45.0, np.nan, -1.0])
    assert counts.shape == (60,)
    assert counts[0] == 2 and counts[1] == 1 and counts[2] == 1 and counts[59] == 1
    assert overflow == 2


def test_smooth3_averages_two_bins_at_the_ends():
    assert gv.smooth3([3, 0, 3, 6]).tolist() == [1.5, 2.0, 3.0, 4.5]


def test_first_valley_on_a_bimodal_histogram():
    counts = [10, 8, 6, 3, 2, 4, 9, 9, 3, 1]
    mode, valley = gv.first_valley(counts)
    assert (mode, valley) == (0, 4)
    assert gv.valley_value(valley) == 2.25


def test_first_valley_skips_a_rising_start_and_reports_no_valley():
    assert gv.first_valley([1, 5, 9, 7, 4, 2]) == (2, None)
    assert gv.first_valley([1, 2, 3]) == (None, None)
    assert gv.valley_value(None) is None


def test_first_valley_takes_the_right_end_of_a_flat_floor():
    # A flat floor: the valley is the last floor bin before the rise (<= left, < right).
    assert gv.first_valley([9, 5, 2, 2, 2, 6]) == (0, 4)


# -- restart_confirm_modes ------------------------------------------------------------------------

def test_censored_quantile_is_numpy_linear_on_finite_values():
    values = [5.0, 1.0, 3.0, 9.0, 7.0]
    for q in (0.0, 0.1, 0.5, 0.9, 1.0):
        assert rc.censored_quantile(values, q) == pytest.approx(np.percentile(values, 100 * q))
    assert rc.censored_quantile([], 0.5) is None


def test_censored_quantile_with_infinite_values():
    values = [1.0, 2.0, 3.0, math.inf, math.inf]
    assert rc.censored_quantile(values, 0.1) == pytest.approx(1.4)
    assert rc.censored_quantile(values, 0.6) == math.inf     # between 3 and inf
    assert rc.censored_quantile(values, 0.5) == 3.0          # exactly on an order statistic


def test_log_midpoint_and_rounding():
    assert rc.log_midpoint_seconds(25.0, 100.0, 25.0) == pytest.approx(2.0)
    assert rc.log_midpoint_seconds(25.0, math.inf, 25.0) is None
    assert rc.log_midpoint_seconds(None, 10.0, 25.0) is None
    assert rc.round_half_up(2.5) == 3 and rc.round_half_up(2.49) == 2
    assert rc.round_half_up(None) is None


def test_fixed_points():
    assert rc.fixed_points({2: 4, 3: 4, 4: 4, 5: 4, 6: None}) == [4]
    assert rc.fixed_points({2: 3, 3: 4}) == []


def test_players_times():
    moving = np.array([False, False, True, False, False, False])
    assert rc.players_times(moving, [0, 2, 3], horizon=2).tolist() == [2.0, 0.0, math.inf]


# -- inertness_md5 ---------------------------------------------------------------------------------

def test_data_roots_of_the_views():
    base = "/m"
    assert im.data_root("BAR-ATH", "raw", base) == "/m/data"
    assert im.data_root("BAR-ATH", "matched", base) is None
    assert im.data_root("DFL-J03WMX", "raw", base) == "/m/refs"
    assert im.data_root("DFL-J03WMX", "matched", base) == "/m/refs-smooth-s3"
    assert im.data_root("METRICA-S1", "matched", base) == "/m/refs-smooth-s5"
    assert im.data_root("METRICA-S3", "raw", base) == "/m/sample-data/data"
    with pytest.raises(ValueError):
        im.data_root("DFL-J03WQQ", "raw", base)


def test_is_inert():
    assert im.is_inert({"5": "a", "9": "a", "15": "a"})
    assert not im.is_inert({"5": "a", "9": "b"})
    assert not im.is_inert({"5": "a", "9": None})
    assert not im.is_inert({})
