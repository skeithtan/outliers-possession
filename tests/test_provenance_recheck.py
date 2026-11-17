"""tools/analysis/provenance_recheck.py on synthetic sequences (P9-F follow-up). No match data."""

import numpy as np
import pytest

from engine import tracks
from engine.config import Config
from tools.analysis import provenance_recheck as pr

PITCH = (105.0, 68.0)
CFG = Config.default()
FPS = 25.0


def _labels(spec):
    """'<U1' labels from a list of (label, count) pairs."""
    return np.array(list("".join(label * count for label, count in spec)))


def test_owner_codes_and_reference_frames():
    labels = _labels([("X", 2), ("H", 5), ("A", 4), ("D", 3), ("A", 3), ("H", 2)])
    assert pr.owner_codes(labels).tolist() == [-1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1,
                                               1, 1, 1, 0, 0]
    assert pr.reference_handovers(labels).tolist() == [7, 17]
    assert pr.reference_onsets(labels).tolist() == [11]
    assert pr.reference_possession_starts(labels) == [
        (7, "A", "handover"), (14, "A", "restart"), (17, "H", "handover")]


def test_nearest_agreement_finds_a_planted_lag_and_bands():
    rng = np.random.RandomState(3)
    n = 400
    owner = np.repeat(rng.randint(0, 2, size=20), 20).astype(np.int8)
    shift = 7
    team = np.full(n, -1, dtype=np.int8)
    team[:n - shift] = owner[shift:]            # the nearest team at s equals the owner at s + 7
    dmin = np.where(np.arange(n) % 2 == 0, 0.8, 2.5)
    usable = np.ones(n, dtype=bool)
    curve = pr.nearest_agreement(owner, team, dmin, usable, range(-3, 15))
    assert curve["7"]["all"] == [n - shift, n - shift]
    assert pr.best_lag(curve, "all")[0] == 7
    # The bands split the same pairs by the source distance.
    close, mid = curve["7"]["le1.0"], curve["7"]["1-3"]
    assert close[1] + mid[1] == curve["7"]["all"][1] and close[0] == close[1]
    assert curve["0"]["all"][0] < curve["7"]["all"][0]
    # A negative lag reads a source frame after the owner frame.
    assert curve["-3"]["all"][1] == n - 3
    first, last, top, rate, at20 = pr.plateau(
        pr.nearest_agreement(owner, team, dmin, usable, range(0, 46)), "all")
    assert top == 7 and rate == 1.0 and first <= 7 <= last and at20 is not None


def test_nearest_agreement_skips_unusable_sources():
    owner = np.zeros(10, dtype=np.int8)
    team = np.zeros(10, dtype=np.int8)
    usable = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 1], dtype=bool)
    out = pr.nearest_agreement(owner, team, np.ones(10), usable, [0, 2])
    assert out["0"]["all"] == [8, 8]
    assert out["2"]["all"] == [6, 6]         # sources 0..7, minus the unusable 3 and 4


def test_flip_leads_walks_back_to_the_start_of_the_run():
    labels = _labels([("H", 60), ("A", 60)])
    n = labels.size
    team = np.zeros(n, dtype=np.int8)
    team[43:] = 1                             # the nearest team flips 17 frames before the label
    usable = np.ones(n, dtype=bool)
    leads = pr.flip_leads(labels, team, usable)
    assert leads["handover"] == [17] and leads["restart"] == []
    summary = pr.lead_summary(leads["handover"])
    assert summary["share_14_25"] == 1.0 and summary["share_early"] == 1.0
    # A nearest team that never flips gives no lead.
    assert pr.flip_leads(labels, np.zeros(n, dtype=np.int8), usable)["handover"] == [None]


def test_stop_delays_and_false_rate():
    starts, stops = np.array([100, 300]), np.array([160, 400])
    delays = pr.stop_delays([50, 120, 170, 450], starts, stops)
    assert delays[0] == 50 and delays[1] == 0 and delays[2] == 130 and np.isinf(delays[3])
    labels = _labels([("H", 400), ("D", 50)])
    confirmed, eligible = pr.false_confirm_rate(labels, starts, stops, 25)
    # Frames 0..374 qualify (25 alive frames must follow). A stop run starting by t + 25 and
    # ending after t passes: t in 75..159 for the first run and 275..374 for the second.
    assert eligible == 375
    assert confirmed == 85 + 100


def _synthetic_pursuit(fast_frames=60):
    """A ball that is kicked at 10 m/s away from nobody while the nearest players stop."""
    n = 400
    x = np.full(n, 30.0)
    y = np.full(n, 34.0)
    start = 100
    x[start:start + fast_frames] = 30.0 + 0.4 * np.arange(fast_frames)
    x[start + fast_frames:] = x[start + fast_frames - 1]
    ball = tracks.clean_ball(x / PITCH[0], y / PITCH[1], PITCH, CFG, fps=FPS)
    d_h = np.full(n, 8.0)
    d_a = np.full(n, 1.0)
    d_a[start - 3:] = np.linspace(2.0, 30.0, n - start + 3)[:n - start + 3]
    near = np.full(n, 3.0)
    near[start + 6:] = 0.5
    feats = {"dH": d_h, "dA": d_a, "meanv_near": near}
    return ball, feats, start


def test_pursuit_candidates_equal_the_engine_and_give_the_kicking_team():
    ball, feats, start = _synthetic_pursuit()
    horizon, chase = 100, 50
    rows = pr.pursuit_candidates(ball, feats, CFG, horizon, chase)
    assert rows, "the synthetic kick must give a candidate"
    assert sorted(r["onset"] for r in rows) == pr.engine_pursuit_onsets(ball, feats, CFG, horizon)
    assert rows[0]["kick"] == "A" and abs(rows[0]["start"] - start) <= 1
    labels = _labels([("A", 150), ("D", 250)])
    summary = pr.kick_summary(rows, labels, feats, ball, CFG, pr.reference_onsets(labels))
    assert summary["all"]["n"] == len(rows)
    assert summary["all"]["engine_team_is_owner"] == len(rows)
    assert summary["all"]["offset_q"]["p50"] == 0.0
    assert pr.is_owned(rows[0], [rows[0]["onset"] + 10])
    assert not pr.is_owned(rows[0], [rows[0]["onset"] + 200])


def test_lookback_offset_counts_back_to_the_clear_margin():
    n = 50
    d_h = np.full(n, 5.0)
    d_a = np.full(n, 5.0)
    d_a[:30] = 1.0                    # A is clearly nearer up to frame 29 only
    visible = np.ones(n, dtype=bool)
    assert pr.lookback_offset(35, "A", d_h, d_a, visible, 0.4) == 6
    assert pr.lookback_offset(20, "A", d_h, d_a, visible, 0.4) == 0
    assert pr.lookback_offset(35, "H", d_h, d_a, visible, 0.4) is None


def test_handover_rises_reads_the_walk_back_of_a_smooth_kick():
    n = 200
    ramp = np.clip((np.arange(n) - 80.0) / 20.0, 0.0, 1.0)
    speed = 1.0 + 13.0 * (3.0 * ramp ** 2 - 2.0 * ramp ** 3)   # a smooth step over 20 frames
    run_id = np.zeros(n, dtype=np.int64)
    dv_signed = tracks.velocity_step(speed, 3, run_id)
    ball = {"speed": speed, "run_id": run_id, "dv_k": np.abs(dv_signed)}
    feats = {"dmin": np.full(n, 0.5)}
    out = pr.handover_rises(ball, feats, [90], CFG)
    assert out["with_peak"] == 1 and out["with_player"] == 1 and out["kicks"] == 1
    # The dv_3 peak is mid-ramp. The walk goes back until the one-frame step falls below
    # DV_TOUCH / DV_K, which is a few frames before the peak, and more than the cap allows only
    # if the ramp is wider than the cap.
    assert out["walk_q"]["n"] == 1 and 3 <= out["walk_q"]["p50"] <= 12
    assert out["rise_mag_q"]["p50"] > 3.0
    assert out["gain_len_q"]["p50"] <= out["rise_len_q"]["p50"]
    assert out["release_share"]["30"] == 1.0
    assert out["release_share"]["5"] <= out["release_share"]["30"]
    # A player far away from the ball: the hand-over is not counted.
    far = pr.handover_rises(ball, {"dmin": np.full(n, 9.0)}, [90], CFG)
    assert far["with_player"] == 0 and far["walk_q"]["n"] == 0


def test_out_lag_rows_on_a_ball_that_vanishes_after_the_onset():
    n = 300
    x = np.full(n, 50.0)
    y = np.full(n, 20.0) + 0.2 * np.arange(n)       # 5 m/s towards y = 68 (touchline)
    visible = np.ones(n, dtype=bool)
    visible[150:] = False
    x = np.where(visible, x, np.nan)
    y = np.where(visible, y, np.nan)
    ball = tracks.clean_ball(x / PITCH[0], y / PITCH[1], PITCH, CFG, fps=FPS)
    summary, near = pr.out_lag_rows(ball, PITCH, CFG, [140, 152])
    # Onset 140: the ball is still visible, 9 frames of track remain. Onset 152: lost 2 frames ago.
    assert near.tolist() == [140, 152]
    assert summary["lag_q"]["min"] == -9.0 and summary["lag_q"]["max"] == 3.0
    assert summary["visible_after_onset"] == 1
    assert summary["moving_inside"] == 2 and summary["heading_q"]["n"] == 2
    # 5 m/s with a friction of 0.5 m/s^2 rolls 25 m, so the ball reaches the line in range:
    # these are extrapolated onsets, not lag-rule onsets.
    assert summary["serving_n"] == 0 and summary["extrap_n"] == 2
    # An onset more than LOSS_NEAR frames before the loss is not a near loss.
    n = 800
    x = np.where(np.arange(n) < 500, 50.0, np.nan)
    y = np.where(np.arange(n) < 500, 30.0, np.nan)
    ball = tracks.clean_ball(x / PITCH[0], y / PITCH[1], PITCH, CFG, fps=FPS)
    summary, near = pr.out_lag_rows(ball, PITCH, CFG, [0])
    assert near.size == 0 and summary["far"] == 1


def test_refused_matches_and_report_runs_on_a_minimal_payload():
    with pytest.raises(SystemExit):
        pr.measure("DFL-J03WQQ", "x.csv")
    with pytest.raises(SystemExit):
        pr.measure("BAR-ATH", "x.csv")
    q = pr.quantiles([1, 2, 3, 4])
    assert q["n"] == 4 and q["p50"] == 2.5
    assert pr.quantiles([np.nan])["p50"] is None
    assert pr.log_odds(0.5) == 0.0 and pr.log_odds(1.0) is None
    assert pr.share(1, 4) == 0.25 and pr.share(1, 0) is None


# -- bucket b rules ---------------------------------------------------------------------------------

def _fine_row(rates_by_band, n=100):
    """One lag row of the fine curve with `n` pairs per band and the given rates."""
    row = {}
    for name, _, _ in pr.FINE_BANDS:
        rate = rates_by_band.get(name, 0.5)
        row[name] = [int(round(rate * n)), n]
    return row


def test_band_pair_sums_the_fine_bands():
    row = _fine_row({"0.0-0.5": 0.9, "0.5-1.0": 0.8, "1.0-1.5": 0.6})
    assert pr.band_pair(row, 0.0, 1.0) == [170, 200]
    assert pr.band_pair(row, 1.0, 1.5) == [60, 100]
    assert pr.band_pair(row, 5.0, 8.0)[1] == 600
    assert pr.band_pair(row, 8.0, np.inf)[1] == 100
    assert pr.band_pair(row, 0.0, np.inf)[1] == 100 * len(pr.FINE_BANDS)


def test_argmax_lag_and_median_half_up():
    fine = {str(lag): _fine_row({"0.0-0.5": 0.5 + 0.01 * (lag if lag <= 12 else 24 - lag)})
            for lag in pr.FINE_LAGS}
    assert pr.argmax_lag(fine, 0.0, 0.5) == 12
    assert pr.median_half_up([3, 4, 5, 6]) == 5          # 4.5 rounds up
    assert pr.median_half_up([3, 4, 4, 6]) == 4


def test_largest_drop_edge():
    assert pr.largest_drop_edge([0.97, 0.95, 0.80, 0.75, 0.70]) == (pytest.approx(1.0),
                                                                     pytest.approx(0.15))
    assert pr.largest_drop_edge([0.9, None, 0.5, 0.4])[0] == pytest.approx(1.5)


def _fine_curve(rate_fn):
    """A fine curve whose band rate is rate_fn(lag, band upper edge), 100 pairs per band."""
    curve = {}
    for lag in pr.FINE_LAGS:
        curve[str(lag)] = _fine_row(dict((name, rate_fn(lag, high))
                                         for name, _, high in pr.FINE_BANDS), n=1000)
    return curve


def test_far_edge_uses_the_pooled_band():
    # Flat at 0.60 beyond 3 m except one noisy 0.5 m band at 7.5-8 m (0.66): a per-band test
    # fails there, the pooled (E, 8] test does not.
    def rate(lag, high):
        if high <= 3.0:
            return 0.9
        return 0.66 if high == 8.0 else 0.60
    fines = {"a": _fine_curve(rate)}
    assert pr.far_edge(fines, 0) == 3
    assert pr.far_edge(fines, 0, tol=0.001) == 5     # the reference band itself always passes


def test_iterate_lag_far_reaches_a_fixed_point_and_records_the_path():
    # The best lag within the band depends on the band: within 3 m it is 6, wider it is 12.
    # The far edge is 3 m at every lag.
    def rate(lag, high):
        if high <= 3.0:
            return 0.9 - 0.001 * abs(lag - 6)
        return 0.6 - 0.001 * abs(lag - 12)
    fines = {"a": _fine_curve(rate), "b": _fine_curve(rate)}
    lag, far, path, cycle = pr.iterate_lag_far(fines)
    assert (lag, far, cycle) == (6, 3, None)
    assert [(s["lag"], s["far"]) for s in path] == [(0, 3), (6, 3), (6, 3)]


def test_iterate_lag_far_takes_the_smaller_lag_of_a_cycle(monkeypatch):
    far_by_lag = {0: 4, 10: 3, 20: 4}
    lag_by_far = {4: 10, 3: 20}
    monkeypatch.setattr(pr, "far_edge", lambda fines, lag: far_by_lag[lag])
    monkeypatch.setattr(pr, "median_best_lag", lambda fines, far: (lag_by_far[int(far)], {}))
    lag, far, path, cycle = pr.iterate_lag_far({})
    assert cycle == [(10, 3), (20, 4)]
    assert (lag, far) == (10, 3)


def test_cover_multiple():
    cover = {"5": 3, "10": 6, "20": 9, "25": 12, "40": 13}
    assert pr.cover_multiple(cover, 13) == 25
    assert pr.cover_multiple(cover, 14) == 40
    assert pr.cover_multiple(cover, 20) is None


def _banded_fine(rate_of, lags=(19,), n=1000):
    """A fine curve whose band rate is rate_of(band low edge), the same at every lag."""
    curve = {}
    for lag in lags:
        row = {}
        for name, low, _high in pr.FINE_BANDS:
            row[name] = [int(round(rate_of(max(low, 0.0)) * n)), n]
        curve[str(lag)] = row
    return curve


def test_select_fixed_lag_keeps_the_lag_and_the_close_edge():
    # Bands below 3 m agree 90 %, bands from 3 m on agree 60 %: the pooled edge is 3 m.
    fine = _banded_fine(lambda low: 0.9 if low < 3.0 else 0.6)
    payloads = [{"game": g, "m1": {"fine": fine}} for g in pr.OPEN_REFERENCES]
    sel = pr.select_fixed_lag(payloads, 19, 1.0)
    assert sel["NEAREST_LAG"] == 19 and sel["NEAR_CLOSE_M"] == 1.0
    assert sel["NEAR_FAR_M"] == 3.0
    assert sel["P_NEAR_FAR"] == 0.9
    assert "fixed lag 19" in pr.fixed_lag_report(sel)
