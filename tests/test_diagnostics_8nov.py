"""tools/analysis/diagnostics_8nov.py: the pure functions of the three diagnostics.

The tests use short synthetic paths and the real engine.decoder on synthetic evidence. They run
no engine on match data and read no reference.
"""

import importlib.util
import os

import numpy as np
import pytest

from engine import decoder
from engine.config import Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "diagnostics_8nov", os.path.join(ROOT, "tools", "analysis", "diagnostics_8nov.py"))
dg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dg)

H, A, D, X = 0, 1, 2, 3


def _seq(*parts):
    return np.concatenate([np.full(n, k, dtype=np.int8) for k, n in parts])


def _known_owner_evidence(n_h, n_a):
    """log_e of a held owner (P_OWNER 0.9) outside a stoppage (P(D) 0.1): H then A."""
    alive, dead = np.log(0.9), np.log(0.1)
    row_h = [alive + np.log(0.9), alive + np.log(0.1), dead]
    row_a = [alive + np.log(0.1), alive + np.log(0.9), dead]
    return np.array([row_h] * n_h + [row_a] * n_a, dtype=np.float64)


def test_runs_and_owner_changes():
    path = _seq((H, 5), (D, 3), (A, 4), (A, 0), (H, 2), (D, 4), (H, 3))
    starts, stops, labels = dg.runs(path)
    assert list(labels) == [H, D, A, H, D, H]
    assert list(starts) == [0, 5, 8, 12, 14, 18]
    # Changes of owner: H->(D)->A, A->H. H->(D)->H is no change. Direct: A->H only.
    assert dg.owner_changes(path) == (2, 1)
    assert dg.owner_changes(np.array([], dtype=np.int8)) == (0, 0)
    assert dg.owner_changes(_seq((D, 4))) == (0, 0)


def test_bypass_records_survival_classes():
    n = 40
    log_e = np.zeros((n, 3))
    costs = np.zeros((n, 3, 3))
    vit = _seq((H, 10), (D, 3), (A, 10), (D, 2), (A, 5), (D, 4), (H, 6))
    # The post-pass absorbs the first D into A, keeps the third D, and the second D has A on both
    # sides, so it is no change of owner.
    final = _seq((H, 10), (A, 15), (A, 5), (D, 4), (H, 6))
    recs = dg.bypass_records(vit, final, log_e, costs, min_dead=5)
    assert [(r["start"], r["stop"], r["left"], r["right"]) for r in recs] == \
        [(10, 13, H, A), (30, 34, A, H)]
    assert [r["survival"] for r in recs] == ["direct", "with_d"]
    removed = _seq((H, 34), (H, 6))
    assert [r["survival"] for r in dg.bypass_records(vit, removed, log_e, costs, 5)] == \
        ["removed", "removed"]
    # A long D run is not a bypass.
    assert dg.bypass_records(vit, final, log_e, costs, min_dead=3) == \
        [r for r in recs if r["length"] < 3]


def test_best_direct_picks_the_evidence_flip():
    log_e = _known_owner_evidence(6, 6)
    costs = np.zeros((12, 3, 3))
    costs[:, H, A] = 8.0
    score, k = dg.best_direct(log_e, costs, 3, 9, H, A)
    assert k == 6
    expected = log_e[3:6, H].sum() + log_e[6:9, A].sum() - 8.0
    assert score == pytest.approx(expected)


def test_viterbi_bypass_near_onset_and_restart_survives_post_pass():
    """The surveyed mechanism on the real decoder: onset and restart windows overlap."""
    cfg = Config.default()
    log_e = _known_owner_evidence(30, 30)
    costs = decoder.switch_costs(60, cfg, {}, onsets=[30], restarts=[(31, "A")])
    vit = decoder.viterbi(log_e, costs)
    final = decoder.min_durations(vit, log_e, costs, cfg)
    recs = dg.bypass_records(vit, final, log_e, costs, int(cfg.MIN_DEAD))
    assert len(recs) == 1
    rec = recs[0]
    assert (rec["left"], rec["right"], rec["length"]) == (H, A, 1)
    assert rec["entry"] == pytest.approx(cfg.C_DEAD_EVENT)
    assert rec["exit"] == pytest.approx(cfg.C_ALIVE_EVENT - cfg.RESTART_TEAM_BONUS)
    assert rec["advantage"] > 0
    assert rec["survival"] == "direct"
    assert not np.any(final == D)


def test_viterbi_no_bypass_near_onset_only():
    """Near an onset alone, 0.5 + 6 plus the D emission of one frame exceeds C_SWITCH 8."""
    cfg = Config.default()
    log_e = _known_owner_evidence(30, 30)
    costs = decoder.switch_costs(60, cfg, {}, onsets=[30], restarts=[])
    vit = decoder.viterbi(log_e, costs)
    assert not np.any(vit == D)
    assert dg.owner_changes(vit) == (1, 1)


def test_d1_counts_windows_and_shares():
    recs = [{"survival": "direct", "entry": 0.5, "exit": -0.5, "length": 1, "advantage": 5.0},
            {"survival": "removed", "entry": 6.0, "exit": 0.5, "length": 3, "advantage": 1.0}]
    out = dg.d1_counts(recs, (100, 90), (80, 80), {"C_DEAD_EVENT": 0.5, "C_ALIVE": 6.0})
    assert out["a_bypass"] == 2 and out["b_survive"] == 1 and out["b_removed"] == 1
    assert out["d_share"] == pytest.approx(0.02)
    assert out["b_share_final"] == pytest.approx(1 / 80.0)
    assert out["windows"] == {"onset_and_restart": 1, "onset_only": 0, "restart_only": 1,
                              "neither": 0}
    empty = dg.d1_counts([], (0, 0), (0, 0), {"C_DEAD_EVENT": 0.5, "C_ALIVE": 6.0})
    assert empty["d_share"] == 0.0 and empty["advantage_median"] is None


def test_oracle_team():
    ours = np.array([H, H, A, D, D, H, X], dtype=np.int8)
    ref = np.array([A, D, A, H, D, A, A], dtype=np.int8)
    # Alive with a live reference -> reference team. Reference D -> our team. Our D/X unchanged.
    assert list(dg.oracle_team(ours, ref)) == [A, H, A, D, D, A, X]


def test_nearest_alive_fill_ties_to_earlier_and_stays_in_segment():
    codes = np.array([D, H, D, D, A, D, D, H, D], dtype=np.int8)
    out = dg.nearest_alive_fill(codes, [(0, 6), (6, 9)])
    # Frame 0 -> H (only right). Frames 2, 3: H at 1 vs A at 4: 2 -> H, 3 -> A. Frame 5 -> A.
    # Segment (6, 9): frame 6 -> H at 7, frame 8 -> H.
    assert list(out) == [H, H, H, A, A, A, H, H, H]
    assert list(dg.nearest_alive_fill(np.array([D, D], dtype=np.int8), [(0, 2)])) == [D, D]
    tie = np.array([H, D, A], dtype=np.int8)
    assert list(dg.nearest_alive_fill(tie, [(0, 3)])) == [H, H, A]


def test_oracle_dead():
    ours = np.array([H, D, D, A, A, H, X], dtype=np.int8)
    ref = np.array([H, H, D, D, A, A, X], dtype=np.int8)
    out = dg.oracle_dead(ours, ref, [(0, 6)])
    # Frame 1: reference alive, we say D -> nearest alive of ours: H at 0 (tie broken earlier).
    # Frames 2, 3: reference D. Frame 5: our team H. Frame 6: reference X, ours kept.
    assert list(out) == [H, H, D, D, A, H, X]


def test_share_and_dwell_kind():
    assert dg.share([True, False, True], [True, True, False]) == 0.5
    assert dg.share([True], [False]) == 0.0
    assert dg.dwell_kind({"gap_stop": True, "placed": True}) == "dwell_placed"
    assert dg.dwell_kind({"tag_tail": True}) == "dwell_tag_tail"
    assert dg.dwell_kind({}) == "dwell_other"


def test_check_game_refuses_sealed_and_unknown():
    with pytest.raises(SystemExit):
        dg.check_game("DFL-J03WQQ")
    with pytest.raises(SystemExit):
        dg.check_game("DFL-J03WR9")
    with pytest.raises(SystemExit):
        dg.check_game("NOPE")
    dg.check_game("BAR-ATH")
    dg.check_game("DFL-J03WMX")


def test_d_run_profile_and_merge():
    path = _seq((D, 3), (H, 5), (D, 4), (A, 5), (D, 30), (A, 4), (D, 2), (A, 3), (D, 6))
    prof = dg.d_run_profile(path, min_dead=10)
    assert prof["d_runs"] == 5
    assert prof["change"] == 1 and prof["change_lengths"] == [4]
    assert (prof["short"], prof["short_change"], prof["short_same"], prof["short_edge"]) == \
        (4, 1, 1, 2)
    merged = dg.merge_profiles([prof, dg.d_run_profile(_seq((H, 3), (D, 12), (A, 3)), 10)])
    assert merged["d_runs"] == 6 and merged["change"] == 2 and merged["short"] == 4
    assert merged["change_length_min"] == 4 and merged["change_length_median"] == 8.0
    assert dg.merge_profiles([])["change_length_min"] is None
