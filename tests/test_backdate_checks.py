"""Synthetic tests for tools/analysis/backdate_checks.py (COMPONENT_BACKDATE.md). No match data."""

import numpy as np

from tools.analysis import backdate_checks as bc

CODES = {"H": 0, "A": 1, "D": 2, "X": 3}


def _codes(text):
    return np.array([CODES[c] for c in text], dtype=np.int8)


def test_a_late_change_gives_its_lag():
    ref = _codes("HHHHAAAAAA")
    ours = _codes("HHHHHHHAAA")
    assert bc.late_lags(ref, ours, [(0, 10)]) == [3]


def test_exact_early_absorbed_and_dead_changes_are_not_late():
    runs = [(0, 10)]
    assert bc.late_lags(_codes("HHHHAAAAAA"), _codes("HHHHAAAAAA"), runs) == []   # exact
    assert bc.late_lags(_codes("HHHHAAAAAA"), _codes("HHAAAAAAAA"), runs) == []   # early
    assert bc.late_lags(_codes("HHHHAAHHHH"), _codes("HHHHHHHHHH"), runs) == []   # absorbed
    assert bc.late_lags(_codes("HHHHAAADDD"), _codes("HHHHHHHDDD"), runs) == []   # to D
    assert bc.late_lags(_codes("HHHHAAAAAA"), _codes("HHHHHHHHHH"), runs) == []   # run end
    assert bc.late_lags(_codes("HHDDAAAAAA"), _codes("HHDDDDAAAA"), runs) == []   # via D


def test_a_change_at_a_run_edge_is_not_counted():
    ref = _codes("HHHHAAAAAA")
    ours = _codes("HHHHHHHAAA")
    assert bc.late_lags(ref, ours, [(0, 4), (4, 10)]) == []


def test_cap_is_the_90th_percentile_rounded_up_to_a_frame():
    assert bc.cap_from_lags(list(range(1, 11)), 25.0) == (10, 0.4)   # p90 9.1 -> 10
    assert bc.cap_from_lags([50] * 9, 25.0) == (50, 2.0)
    assert bc.cap_from_lags([], 25.0) == (None, None)


def test_new_short_spells_are_only_the_spells_that_did_not_exist():
    runs = [(0, 70)]
    off = _codes("H" * 30 + "A" * 10 + "H" * 30)
    assert bc.new_short_spells(off, off, runs) == []
    on = _codes("H" * 30 + "A" * 5 + "H" * 35)
    assert bc.new_short_spells(off, on, runs) == [(30, 35, 1)]
    assert bc.spells(on, runs) == [(0, 30, 0), (30, 35, 1), (35, 70, 0)]


def test_k2b_a_short_spell_that_only_grew_is_not_a_new_possession():
    runs = [(0, 70)]
    off = _codes("H" * 30 + "A" * 10 + "D" * 30)
    on = _codes("H" * 25 + "A" * 15 + "D" * 30)        # the A spell starts 5 frames earlier
    assert bc.new_short_spells(off, on, runs) == [(25, 40, 1)]
    assert bc.new_short_possessions(off, on, runs) == []
    cut = _codes("H" * 30 + "D" * 40)                  # no A spell at all with the switch off
    assert bc.new_short_possessions(cut, on, runs) == [(25, 40, 1)]
    other = _codes("H" * 25 + "A" * 3 + "H" * 12 + "D" * 30)   # an A spell not contained
    assert bc.new_short_possessions(off, other, runs) == [(25, 28, 1), (28, 40, 0)]


def test_recall_split_counts_short_and_long_reference_spells():
    runs = [(0, 100)]
    ref = _codes("H" * 25 + "A" * 75)          # 1 s and 3 s
    pred = _codes("H" * 100)
    assert bc.recall_split(ref, pred, runs, 25.0) == {"short": [0, 1], "long": [0, 1]}
    assert bc.recall_split(ref, ref, runs, 25.0) == {"short": [1, 1], "long": [1, 1]}


def test_label_diff_finds_moved_changes_on_graded_frames():
    off = _codes("XXHHHHHHAAAAHHHH")
    on = _codes("XXHHHHAAAAAAHHHH")
    graded = np.array([False, False] + [True] * 14)
    out = bc.label_diff(off, on, graded)
    assert out["frames_changed"] == 2
    assert out["stretches"] == 1
    assert out["stretches_ending_at_a_change_of_off"] == 1
    assert out["median_shift"] == 2.0
    on2 = on.copy()
    on2[0] = 2                                  # a change outside the graded frames
    assert bc.label_diff(off, on2, graded)["frames_changed"] == 2


def _row(deltas, mean, low, high=0.02, highs=None):
    highs = highs or [0.02] * len(deltas)
    games = {"G{}".format(i): {"versus_base": {"delta": d, "low": d - 0.005, "high": h}}
             for i, (d, h) in enumerate(zip(deltas, highs))}
    return {"value": True, "games": games,
            "versus_base": {"mean_delta": mean, "low": low, "high": high}}


def _checks(new_short=0, f2=0.003, new_possessions=0, exempt=0):
    return {"metrica_new_short_spells": new_short,
            "metrica_new_short_possessions": new_possessions,
            "metrica_new_short_spells_exempt": exempt,
            "metrica_new_short_spells_binding": new_short - exempt,
            "matches": [{"game": "METRICA-S3-F2", "delta": {"delta": f2}}]}


def _inputs(**changes):
    sweeps = {}
    for view in ("raw", "matched", "sigma95"):
        sweeps[(view, "metrica")] = _row([0.007, 0.006, 0.003], 0.0053, 0.002)
        sweeps[(view, "dfl")] = _row([0.001, 0.002, 0.0, 0.004, 0.003], 0.002, -0.001)
    sweeps.update(changes.get("sweeps", {}))
    checks = {v: _checks() for v in ("raw", "matched", "sigma95")}
    checks.update(changes.get("checks", {}))
    return sweeps, checks, changes.get("log", "verdict: PASS")


def test_verdict_keeps_when_every_criterion_passes():
    lines, keep = bc.verdict(*_inputs())
    assert keep, [line for line in lines if not line[1]]
    assert len(lines) == 15


def test_verdict_kills_on_any_failed_criterion():
    cases = [
        {"sweeps": {("raw", "metrica"): _row([0.007, 0.006, -0.003], 0.003, -0.001)}},
        {"sweeps": {("matched", "dfl"): _row([-0.02] * 5, -0.011, -0.02)}},
        {"sweeps": {("matched", "metrica"): _row([0.01, -0.001, -0.001], 0.0027, 0.0)}},
        {"log": "verdict: FAIL"},
        {"checks": {"raw": _checks(new_short=1)}},
        {"sweeps": {("raw", "metrica"): _row([0.007, 0.006, 0.003], 0.005, 0.001,
                                             highs=[0.02, 0.02, -0.001])}},
        {"sweeps": {("sigma95", "metrica"): _row([0.0, 0.0, -0.003], -0.001, -0.004)}},
        {"checks": {"matched": _checks(f2=-0.001)}},
    ]
    for case in cases:
        _, keep = bc.verdict(*_inputs(**case))
        assert not keep, case


def test_k2b_edge_exemption_needs_a_reference_edge_and_a_long_engine_spell():
    runs = [(0, 50)]
    # S1 matched: the engine A spell has 27 frames, the reference run ends 16 frames into it.
    off_engine = _codes("H" * 34 + "A" * 27 + "D" * 9)
    short = [(29, 50, 1)]
    assert bc.edge_exempt(short, off_engine, runs) == short
    # (b) fails: the engine spell itself is short.
    off_short = _codes("H" * 34 + "A" * 18 + "D" * 18)
    assert bc.edge_exempt(short, off_short, runs) == []
    # (a) fails: no reference edge at the spell.
    assert bc.edge_exempt([(29, 45, 1)], off_engine, [(0, 70)]) == []


def test_k2b_verdict_judges_the_verbatim_count_less_the_exempt_spells():
    sweeps, checks, log = _inputs(checks={"matched": _checks(new_short=1, exempt=1)})
    lines, keep = bc.verdict(sweeps, checks, log, "edge")
    assert keep
    assert any("1 new, 1 exempt" in d for _, _, d in lines)
    sweeps, checks, log = _inputs(checks={"raw": _checks(new_short=1)})
    _, keep = bc.verdict(sweeps, checks, log, "edge")
    assert not keep
    _, keep = bc.verdict(*_inputs(checks={"matched": _checks(new_short=1, exempt=1)}))
    assert not keep


def test_verdict_kills_an_inert_component():
    zero = _row([0.0, 0.0, 0.0], 0.0, 0.0)
    zero_dfl = _row([0.0] * 5, 0.0, 0.0)
    sweeps = {(v, p): (zero if p == "metrica" else zero_dfl)
              for v in ("raw", "matched", "sigma95") for p in ("metrica", "dfl")}
    lines, keep = bc.verdict(*_inputs(sweeps=sweeps))
    assert not keep
    assert [name for name, passed, _ in lines if not passed] == [
        "extra: supported (5 % end > 0) on at least one of them",
        "not inert: some delta differs from zero"]
