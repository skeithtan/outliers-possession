"""P0-D: tools/sanity.py and tools/forensics.py (docs/plan/tasks/P0-D-forensics-and-sanity.md).

Synthetic tests need no data. The `slow` tests run the tools on the organisers' baselines
(evaluation/results/*.csv, evaluation/reports/*, the data bundle) and check the PLAN.md 1.7
reference table. Tests may read the public GT (via tools/gt.py); sanity itself never does.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

import games
from tools import forensics, sanity
import _common as cm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "evaluation", "results")
REPORTS = os.path.join(ROOT, "evaluation", "reports")
SMOOTH50 = os.path.join(RESULTS, "baseline_smooth50.csv")
SMOOTH125 = os.path.join(RESULTS, "baseline_smooth125.csv")
H, A, D, X = 0, 1, 2, 3

_TACTICAL = games.asset_paths()
HAVE_DATA = all(os.path.isfile(_TACTICAL[k]) for k in
                ("atd_meta", "atd_raw", "smart_tagging")) and os.path.isfile(SMOOTH50)
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="match data bundle / baselines not present")
HAVE_REPORTS = all(os.path.isfile(os.path.join(REPORTS, "baseline_smooth50" + s))
                   for s in (".json", "_transitions.csv"))
needs_reports = pytest.mark.skipif(
    not HAVE_REPORTS, reason="run python evaluation/run_evaluation.py first")


def _codes(spec):
    """[('H', 40), ('D', 20), ...] -> int8 codes."""
    letters = "HADX"
    return np.concatenate([np.full(n, letters.index(c), dtype=np.int8) for c, n in spec])


# -- sanity: synthetic -------------------------------------------------------------------------

def test_sanity_counts_per_block_and_never_merges_across_halftime():
    # block 1 = frames 10..109: H40 D20 H40 ; block 2 = frames 150..249: H100 ; X elsewhere
    labels = np.concatenate([_codes([("X", 10), ("H", 40), ("D", 20), ("H", 40), ("X", 40),
                                     ("H", 100), ("X", 10)])])
    s = sanity.sanity_stats(labels, blocks=[(1, 10, 109), (2, 150, 249)], use_tags=False,
                            use_ball=False)
    assert s["total"]["possessions"] == 3          # H, H | H -- the two blocks stay apart
    assert s["halves"][1]["possessions"] == 2
    assert s["halves"][2]["possessions"] == 1
    assert s["total"]["d_segments"] == 1
    assert s["total"]["d_share"] == pytest.approx(20 / 200.0)
    assert s["total"]["x_graded"] == 0
    assert s["total"]["poss_longest_s"] == pytest.approx(100 / 25.0)
    assert s["total"]["transitions"] == 2
    assert set(s["violations"]) == {"possessions", "d_share", "d_segments"}


def test_sanity_counts_d_segments_that_end_at_a_block_end():
    """A D segment that runs to the end of a block is a stoppage without a restart.

    Block 1 ends on D at its half's end. Block 2 ends on D at frame 249, but its half ends at
    frame 299 (blocks shorter than the half, as a reference with X can give). Block 3 ends on H.
    """
    labels = _codes([("X", 10), ("H", 80), ("D", 20), ("X", 40), ("A", 70), ("D", 30),
                     ("X", 50), ("H", 50), ("D", 10), ("H", 40)])
    blocks = [(1, 10, 109), (2, 150, 249), (3, 300, 399)]
    out = sanity._d_at_block_end(labels, blocks, {1: 109, 2: 299, 3: 399})
    assert out["n_at_block_end"] == 2 and out["n_not_half_end"] == 1
    assert out["segments"] == [
        {"half": 1, "start": 90, "end": 109, "half_end": True},
        {"half": 2, "start": 220, "end": 249, "half_end": False}]
    stats = sanity.sanity_stats(labels, blocks=blocks, use_tags=False, use_ball=False)
    assert stats["d_block_end"]["n_at_block_end"] == 2
    text = "\n".join(sanity.format_report(stats))
    assert "D segments that end at a block end other than the half's end:" in text


def test_sanity_accepts_strings_and_flags_graded_x():
    labels = np.array(["X", "H", "X", "H", "A", "X"], dtype=object)
    s = sanity.sanity_stats(labels, blocks=[(1, 1, 4)], use_tags=False, use_ball=False)
    assert s["total"]["x_graded"] == 1
    assert "x_graded" in s["violations"]
    # X is forward-filled like the scorer: H X H A -> H H H A -> 2 spells
    assert s["total"]["possessions"] == 2


def test_sanity_veto_table_passes_a_plausible_vector():
    unit = [("H", 100), ("A", 100), ("D", 100)]
    labels = _codes(unit * 100)
    n = len(labels)
    s = sanity.sanity_stats(labels, blocks=[(1, 0, n // 2 - 1), (2, n // 2, n - 1)],
                            use_tags=False, use_ball=False)
    assert s["total"]["possessions"] == 200
    assert s["total"]["d_segments"] == 100
    assert s["violations"] == []


def test_gap_label_changes_counts_only_long_gaps_inside_a_run():
    n = 400
    labels = _codes([("H", 100), ("D", 100), ("A", 200)])
    miss = np.zeros(n, dtype=bool)
    miss[90:210] = True          # 120-frame gap containing the H->D and D->A changes
    miss[300:320] = True         # 20-frame gap: ignored
    labels[310:] = D             # a change inside the short gap is not counted either
    out = sanity._gap_label_changes(labels, miss, [(0, n)])
    assert out["long_gaps"] == 1
    assert out["n_changes"] == 2
    assert [(c["frame"], c["from"], c["to"]) for c in out["first"]] == [(100, "H", "D"),
                                                                        (200, "D", "A")]


def _tags(rows):
    return pd.DataFrame(rows, columns=["id", "code", "team", "type", "half", "start_frame",
                                       "end_frame"])


def test_set_piece_check_lists_tags_without_d_within_margin():
    labels = _codes([("H", 1000), ("D", 50), ("A", 2000)])
    tags = _tags([(1, "SET PIECES", "H", "Corner Kick", 1, 1200, 1400),   # D ends 1050; margin 250 f
                  (2, "SET PIECES", None, "Throw-in", 1, 2500, 2600),     # no D near
                  (3, "BUILD UP", "H", None, 1, 0, 100)])
    out = sanity._set_piece_check(labels, tags, [(0, len(labels))])
    assert out["n_set_pieces"] == 2 and out["n_checked"] == 2
    assert [m["id"] for m in out["without_d"]] == [2]
    assert out["without_d"][0]["type"] == "Throw-in"


def test_tag_agreement_uses_alive_labels_and_drops_two_team_frames():
    n = 400
    labels = _codes([("H", 100), ("A", 100), ("D", 100), ("H", 100)])
    tags = _tags([(1, "BUILD UP", "H", None, 1, 0, 200),          # H on 0..100, A frames wrong
                  (2, "LONG BALL", "A", None, 1, 150, 250),       # overlaps tag 1 on 150..200
                  (3, "DEFENDING IN MIDDLE THIRD", "A", None, 1, 300, 400)])  # not a phase code
    out = sanity._tag_agreement(labels, tags, [(0, n)], n)
    assert out["conflict_frames_dropped"] == 50
    assert out["tagged_frames"] == 200                  # 0..150 H-tag, 200..250 A-tag
    assert out["tagged_and_our_alive"] == 150           # 0..150 alive; 200..250 is our D
    assert out["agree_frames"] == 100                   # 0..100 H==H; 100..150 A vs H tag
    assert out["agreement"] == pytest.approx(100 / 150.0)
    assert out["agreement_all_tagged"] == pytest.approx(100 / 200.0)


# -- forensics: synthetic -----------------------------------------------------------------------

def test_sandwich_check_detects_merged_and_separate():
    runs = [(0, 300)]
    gt = _codes([("A", 100), ("D", 100), ("A", 100)])
    merged = _codes([("A", 300)])
    split = _codes([("A", 100), ("H", 20), ("A", 180)])
    [m] = forensics.sandwich_check(gt, merged, runs)
    assert m["team"] == "A" and m["separate"] is False
    [s] = forensics.sandwich_check(gt, split, runs)
    assert s["separate"] is True
    assert s["our_frames"] == {"H": 20, "A": 80, "D": 0}
    # H D A and A D H are not sandwiches
    assert forensics.sandwich_check(_codes([("H", 50), ("D", 50), ("A", 50)]), merged[:150],
                                    [(0, 150)]) == []


def test_trace_offsets_include_zero_and_step_three():
    offs = forensics.trace_offsets(50)
    assert 0 in offs and offs[0] == -48 and offs[-1] == 48
    assert set(np.diff(offs)) == {3}


def test_kind_of_splits_restarts_by_ball_gap():
    missing = np.zeros(200, dtype=bool)
    missing[100:150] = True
    assert forensics.kind_of(H, A, missing, 120, 30) == "H<->A"
    assert forensics.kind_of(A, D, missing, 120, 30) == "alive->D"
    # restart at 160: ball back at 150, missing within the previous 30 frames -> reappear
    assert forensics.kind_of(D, H, missing, 160, 30) == "D->alive (reappear)"
    assert forensics.kind_of(D, H, missing, 190, 30) == "D->alive (kick)"
    # the card's literal rule (missing on frame - 1) is lookback 1
    assert forensics.kind_of(D, H, missing, 160, 1) == "D->alive (kick)"
    assert forensics.kind_of(D, H, missing, 150, 1) == "D->alive (reappear)"


def test_nearest_same_destination_ignores_other_destinations():
    preds = [(100, A, H), (140, D, A), (300, A, H)]
    assert forensics.nearest_same_destination((120, A, H), preds) == (100, A, H)
    assert forensics.nearest_same_destination((120, H, D), preds) is None


def test_kinematics_on_synthetic_ball():
    n, slots = 20, 4
    px = np.full((n, slots), 0.5, dtype=np.float32)
    py = np.full((n, slots), 0.5, dtype=np.float32)
    px[:, 0] = 0.5 + 0.01   # home, 1.053 m east of the ball track origin (length 105.3)
    px[:, 2] = 0.5 - 0.02   # away, 2.106 m west
    px[:, 1] = np.nan       # unobserved slots
    px[:, 3] = np.nan
    bx = (0.5 + 0.01 * np.arange(n)).astype(np.float32)   # moves 1.053 m / frame = 26.325 m/s
    bx[10:12] = np.nan
    tactical = {"px": px, "py": py, "ps": np.full((n, slots), 2.0, dtype=np.float32),
                "bx": bx, "by": np.full(n, 0.5, dtype=np.float32),
                "team_of_slot": np.array([0, 0, 1, 1], dtype=np.int8)}
    k = cm.kinematics(tactical, 0, n, (105.3, 68.0))
    assert k["v"][3] == pytest.approx(26.325, rel=1e-3)
    assert np.isnan(k["v"][0]) and np.isnan(k["v"][10]) and np.isnan(k["v"][12])
    assert k["dv_3"][2] == pytest.approx(0.0, abs=1e-3)       # constant speed
    assert np.isnan(k["dv_3"][8])                             # speed[11] missing
    assert k["dH"][0] == pytest.approx(1.053, rel=1e-3)
    assert k["dH"][1] == pytest.approx(0.0, abs=1e-3)
    assert k["dA"][0] == pytest.approx(2.106, rel=1e-3)
    assert np.isnan(k["dH"][10])                              # ball missing
    assert k["meanv"][5] == pytest.approx(2.0)
    assert bool(k["ball_missing"][10]) and not bool(k["ball_missing"][9])


def test_forensics_without_reports_exits_2(tmp_path, capsys):
    path = tmp_path / "nothing_scored_yet.csv"
    path.write_text("frame,label\n")
    assert forensics.main([str(path)]) == 2
    assert "run_evaluation.py" in capsys.readouterr().err


# -- baselines (data-backed) --------------------------------------------------------------------

@pytest.fixture(scope="module")
def smooth50_stats():
    return sanity.sanity_stats(cm.read_results(SMOOTH50))


@pytest.mark.slow
@needs_data
def test_sanity_reproduces_smooth50_reference(smooth50_stats):
    t = smooth50_stats["total"]
    assert abs(t["possessions"] - 594) <= 1
    assert abs(t["d_share"] * 100 - 32.3) <= 0.2
    assert abs(t["d_segments"] - 133) <= 1
    assert t["x_graded"] == 0
    assert smooth50_stats["violations"] == ["possessions"]      # PLAN 5.3: expected for smooth50


@pytest.mark.slow
@needs_data
@pytest.mark.skipif(not os.path.isfile(SMOOTH125), reason="baseline_smooth125 missing")
def test_sanity_reproduces_smooth125_reference():
    t = sanity.sanity_stats(cm.read_results(SMOOTH125))["total"]
    assert abs(t["possessions"] - 398) <= 1
    assert abs(t["d_share"] * 100 - 22.3) <= 0.2
    assert abs(t["d_segments"] - 79) <= 1


@pytest.mark.slow
@needs_data
def test_sanity_tag_codes_exist_and_agreement_is_a_fraction(smooth50_stats):
    assert smooth50_stats["tag_codes_missing"] == []
    ag = smooth50_stats["tags"]["total"]["agreement"]
    assert 0.0 < ag < 1.0
    sp = smooth50_stats["set_pieces"]
    assert sp["n_checked"] == 59 and 0 <= sp["n_without_d"] <= 59


@pytest.mark.slow
@needs_data
@needs_reports
def test_forensics_matches_the_scorer_report_for_smooth50():
    with open(os.path.join(REPORTS, "baseline_smooth50.json")) as handle:
        report = json.load(handle)
    res = forensics.analyse(SMOOTH50)
    loose = report["transitions"][str(max(report["taus"]))]
    assert res["reports_stale"] is False
    assert sum(r["status"] == "missed" for r in res["gt_transitions"]) == loose["n_missed"] == 18
    assert sum(r["status"] == "matched" for r in res["gt_transitions"]) == loose["n_matched"]
    assert len(res["spurious"]) == loose["n_spurious"] == 27
    assert res["summary"]["spell_f1_mean"] == pytest.approx(report["spell_f1_mean"])
    # every GT transition got a trace centred on it
    assert all(any(row["off"] == 0 for row in r["trace"]) for r in res["gt_transitions"])
    # kinds partition the 20 public transitions: 12 H<->A, 4 alive->D, 4 D->alive
    kinds = res["latency_by_kind"]
    assert [kinds[k]["n_gt"] for k in forensics.KINDS[:2]] == [12, 4]
    assert kinds[forensics.KINDS[2]]["n_gt"] + kinds[forensics.KINDS[3]]["n_gt"] == 4


@pytest.mark.slow
@needs_data
@needs_reports
def test_sandwich_15889_is_a_boolean_for_smooth50():
    res = forensics.analyse(SMOOTH50)
    [sw] = [s for s in res["sandwiches"] if s["gt_d_start"] == 15889]
    assert sw["team"] == "A" and sw["gt_d_stop"] == 16400
    assert isinstance(sw["separate"], bool)
