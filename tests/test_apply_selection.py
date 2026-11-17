"""tools/analysis/apply_selection.py: the pre-registered selection rule on synthetic tables.

The tests build small spellF1 tables by hand. They run no engine and read no match data.
"""

import importlib.util
import json
import os
from fractions import Fraction

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "apply_selection", os.path.join(ROOT, "tools", "analysis", "apply_selection.py"))
sel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sel)

METRICA = ["METRICA-S1", "METRICA-S2", "METRICA-S3"]
DFL = ["DFL-J03WMX", "DFL-J03WN1", "DFL-J03WOH", "DFL-J03WOY", "DFL-J03WPY"]
GAMES = METRICA + DFL
GRID5 = [1, 2, 3, 4, 5]


def peaked(peak, n=5, top=0.70, step=0.01):
    """Scores that fall by `step` per grid index away from `peak`."""
    return [top - step * abs(i - peak) for i in range(n)]


def flat(n=5, value=0.60):
    return [value] * n


def table(metrica, dfl):
    """{game: scores}: one score list for every Metrica match and one for every DFL match."""
    out = {g: list(metrica[i]) for i, g in enumerate(METRICA)}
    out.update({g: list(dfl[i]) for i, g in enumerate(DFL)})
    return out


def spec(judge, view="raw", grid=GRID5, **extra):
    return dict({"bucket": "a", "judge": judge, "view": view, "grid": list(grid)}, **extra)


# -- the parts -------------------------------------------------------------------------------

def test_judge_weights():
    w = sel.judge_weights("all", GAMES)
    assert w["METRICA-S1"] == Fraction(1, 6) and w["DFL-J03WMX"] == Fraction(1, 10)
    assert sum(w.values()) == 1
    assert sel.judge_weights("metrica", GAMES) == {g: Fraction(1, 3) for g in METRICA}
    assert sel.judge_weights("dfl", GAMES) == {g: Fraction(1, 5) for g in DFL}


def test_plateau_has_gaps_and_midpoint_spans_them():
    spells = [0.700, 0.690, 0.6995, 0.650, 0.6985]
    assert sel.plateau(spells) == [0, 2, 4]
    assert sel.midpoint([0, 2, 4]) == 2
    assert sel.midpoint([1, 2]) == Fraction(3, 2)


def test_plateau_edge_counts_with_rounding():
    assert sel.plateau([0.700, 0.698, 0.6979]) == [0, 1]


def test_weighted_median_plain_and_exact_half():
    w = {g: Fraction(1, 3) for g in METRICA}
    assert sel.weighted_median(dict(zip(METRICA, [0, 4, 1])), w) == (1, False)
    # All three Metrica matches at 0 and all five DFL matches at 4 split the weight 1/2 to 1/2.
    w = sel.judge_weights("all", GAMES)
    points = dict({g: Fraction(0) for g in METRICA}, **{g: Fraction(4) for g in DFL})
    assert sel.weighted_median(points, w) == (2, True)
    # One DFL match moves to 3: the cumulative weight is still exactly 1/2 at 0.
    points["DFL-J03WMX"] = Fraction(3)
    assert sel.weighted_median(points, w) == (Fraction(3, 2), True)
    # One DFL match moves to 0: the cumulative weight at 0 is 6/10, above 1/2.
    points["DFL-J03WMX"] = Fraction(0)
    assert sel.weighted_median(points, w) == (0, False)


def test_snap_nearest_and_ties_toward_centre():
    assert sel.snap(Fraction(5, 4), 5) == (1, False)
    assert sel.snap(Fraction(7, 4), 5) == (2, False)
    assert sel.snap(Fraction(1, 2), 5) == (1, True)       # 0 or 1: 1 is nearer to centre 2
    assert sel.snap(Fraction(7, 2), 5) == (3, True)       # 3 or 4: 3 is nearer to centre 2
    assert sel.snap(Fraction(5, 2), 6) == (2, True)       # 2 or 3 straddle centre 2.5: lower
    assert sel.snap(Fraction(3, 2), 6) == (2, True)       # 1 or 2: 2 is nearer to centre 2.5


# -- bucket a decisions ----------------------------------------------------------------------

def test_metrica_judge_median_of_midpoints():
    raw = table([peaked(1), peaked(3), peaked(4)], [flat()] * 5)
    out = sel.select_grid("F", spec("metrica"), GRID5, {"raw": raw, "matched": raw}, "raw")
    assert out["decision"] == "value"
    assert out["o_m"] == {"METRICA-S1": 1.0, "METRICA-S2": 3.0, "METRICA-S3": 4.0}
    assert out["selected"] == 4                       # index 3
    assert out["details"]["guard_passed"] is True     # DFL is flat


def test_half_index_midpoint_and_snap_tie_toward_centre():
    # S1 plateau {0, 1} gives o = 0.5. S2 and S3 give 0.5 too. The median 0.5 snaps to 1.
    two = [0.70, 0.70, 0.60, 0.55, 0.50]
    raw = table([two, two, two], [flat()] * 5)
    out = sel.select_grid("F", spec("metrica"), GRID5, {"raw": raw, "matched": raw}, "raw")
    assert out["o_m"]["METRICA-S1"] == 0.5
    assert out["details"]["snap_tie"] is True
    assert out["selected"] == 2


GRID7 = [10, 20, 30, 40, 50, 60, 70]


def test_all_judge_exact_half_takes_midpoint_and_has_no_guard():
    raw = table([peaked(1, 7)] * 3, [peaked(5, 7)] * 5)
    out = sel.select_grid("F", spec("all", grid=GRID7), GRID7, {"raw": raw, "matched": raw},
                          "raw")
    assert out["details"]["median_exact_half"] is True
    assert out["details"]["weighted_median_index"] == 3.0
    assert out["selected"] == 40
    assert out["details"]["guard"] == "none: judge all"


def test_guard_moves_to_nearest_passing_value():
    # Metrica selects index 1. The DFL mean peaks at index 5 and falls 0.004 per index, so
    # index 1 is 0.016 below and index 4 (0.004 below) is the nearest that passes.
    raw = table([peaked(1, 7)] * 3, [peaked(5, 7, step=0.004)] * 5)
    out = sel.select_grid("F", spec("metrica", grid=GRID7), GRID7,
                          {"raw": raw, "matched": raw}, "raw")
    assert out["details"]["snapped_index"] == 1
    assert out["details"]["guard_passed"] is False
    assert out["selected"] == 50                       # index 4
    # A loose DFL curve lets index 1 pass.
    raw = table([peaked(1, 7)] * 3, [peaked(5, 7, step=0.001)] * 5)
    out = sel.select_grid("F", spec("metrica", grid=GRID7), GRID7,
                          {"raw": raw, "matched": raw}, "raw")
    assert out["details"]["guard_passed"] is True and out["selected"] == 20


def test_guard_tie_between_two_passing_values_goes_toward_centre():
    # DFL judges and selects index 2. The Metrica mean fails at 2 and passes at 1 and 3 only.
    metrica = [[0.70, 0.70, 0.60, 0.70, 0.60]] * 3
    raw = table(metrica, [peaked(2)] * 5)
    out = sel.select_grid("F", spec("dfl"), GRID5, {"raw": raw, "matched": raw}, "raw")
    assert out["details"]["snapped_index"] == 2 and out["details"]["guard_passed"] is False
    # Both 1 and 3 are one step away and equally near to centre 2: the lower index wins.
    assert out["details"]["guard_index"] == 1 and out["selected"] == 2


def test_guard_uses_the_primary_view_only():
    raw = table([peaked(1, 7)] * 3, [peaked(5, 7, step=0.004)] * 5)
    matched = table([peaked(1, 7)] * 3, [flat(7)] * 5)
    out = sel.select_grid("F", spec("metrica", view="matched", grid=GRID7), GRID7,
                          {"raw": raw, "matched": matched}, "matched")
    assert out["details"]["guard_passed"] is True and out["selected"] == 20


def test_grid_edge_on_most_weight_is_inconclusive_and_takes_the_centre():
    # Metrica judges. S1 and S2 peak at the top edge: 2/3 of the weight. The centre is index 2.
    raw = table([peaked(4), peaked(4), peaked(1)], [flat()] * 5)
    out = sel.select_grid("F", spec("metrica"), GRID5, {"raw": raw, "matched": raw}, "raw")
    assert out["decision"] == "inconclusive" and out["details"]["edge"] is True
    assert out["selected"] == 3 and out["selected_index"] == 2
    # A six-value grid has two centre indices, 2 and 3: the lower one.
    grid6 = [1, 2, 3, 4, 5, 6]
    raw6 = table([peaked(0, 6)] * 3, [flat(6)] * 5)
    out = sel.select_grid("F", spec("metrica", grid=grid6), grid6,
                          {"raw": raw6, "matched": raw6}, "raw")
    assert out["decision"] == "inconclusive" and out["selected"] == 3
    # A pre-registered fallback wins over the centre.
    out = sel.select_grid("F", spec("metrica", inert_fallback="U: Otsu"), GRID5,
                          {"raw": raw, "matched": raw}, "raw")
    assert out["decision"] == "inconclusive" and out["selected"] is None
    assert out["fallback"] == "U: Otsu"


def test_grid_edge_needs_more_than_half_of_the_weight():
    # Judge all: the three Metrica matches at an edge carry exactly 1/2. Not inconclusive.
    raw = table([peaked(0)] * 3, [peaked(2)] * 5)
    out = sel.select_grid("F", spec("all"), GRID5, {"raw": raw, "matched": raw}, "raw")
    assert out["details"]["edge_weight"] == 0.5
    assert out["decision"] == "value"
    # Both edges count together: two DFL matches at 0 and two at 4 plus Metrica at 0 is 9/10.
    raw = table([peaked(0)] * 3, [peaked(0), peaked(0), peaked(4), peaked(4), peaked(2)])
    out = sel.select_grid("F", spec("all"), GRID5, {"raw": raw, "matched": raw}, "raw")
    assert out["decision"] == "inconclusive"


def test_grid_edge_comes_before_tie_to():
    grid = [1.0, 1.25, 1.5, 2.0, 2.5]
    raw = table([flat()] * 3, [peaked(0)] * 5)
    s = spec("dfl", grid=grid, tie_to="V_STOP")
    out = sel.select_grid("F", s, grid, {"raw": raw, "matched": raw}, "raw", tie_value=1.0)
    assert out["decision"] == "inconclusive" and out["selected"] == 1.5


def test_mixed_view_reads_each_provider_from_its_view():
    # DFL matched peaks at index 1, DFL raw at 3, Metrica raw at 2, Metrica matched at 3.
    raw = table([peaked(2)] * 3, [peaked(3)] * 5)
    matched = table([peaked(3)] * 3, [peaked(1)] * 5)
    s = spec("all", view="mixed: DFL matched + Metrica raw", recorded="DFL raw")
    primary = sel.parse_view(s["view"])
    assert primary == {"dfl": "matched", "metrica": "raw"}
    out = sel.select_grid("F", s, GRID5, {"raw": raw, "matched": matched}, primary)
    assert out["o_m"]["DFL-J03WMX"] == 1.0 and out["o_m"]["METRICA-S1"] == 2.0
    # Cumulative weight at 1 is exactly 1/2, so the median is 1.5 and snaps toward centre: 2.
    assert out["selected"] == 3
    rec = sel.recorded_view(s, {"raw": raw, "matched": matched}, GRID5)
    assert rec["view"] == "raw" and rec["o_m"]["DFL-J03WPY"] == 3.0


def recorded_grids():
    """Return the committed grids.json, or skip: the public copy leaves out docs/."""
    path = os.path.join(ROOT, "docs", "plan", "calibration", "grids.json")
    if not os.path.isfile(path):
        pytest.skip("the public copy leaves out the records under docs/ (tools/export_public.py)")
    with open(path) as handle:
        return json.load(handle)


def test_effective_spec_reads_the_appendix_of_grids_json():
    grids = recorded_grids()
    assert sel.effective_spec("EVENT_CONF_MAX", grids) is None
    assert sel.effective_spec("RESTART_K", grids) is None
    # The judges after D3.
    s = sel.effective_spec("REAPPEAR_SPEED", grids)
    assert s["judge"] == "metrica" and s["view"] == "raw" and s["registered_judge"] == "dfl"
    assert s["recorded"] == "DFL raw"
    s = sel.effective_spec("DV_RESTART", grids)
    assert s["judge"] == "metrica" and sel.parse_view(s["view"]) == ["raw", "matched"]
    assert s["recorded"] == "DFL matched"
    # The superseded restart-side judges stay readable.
    old = grids["appendix_8nov"]["superseded"]["judge_changes"]
    s = sel.effective_spec("DV_RESTART", grids, superseded=old)
    assert sel.parse_view(s["view"]) == {"dfl": "matched", "metrica": "raw"}
    s = sel.effective_spec("REAPPEAR_TEAM_M", grids, superseded=old)
    assert s["judge"] == "all" and s["view"] == "raw"
    for field in ("PURSUIT_N_NEAREST", "PURSUIT_KICK_LOOKBACK", "ONSET_CONFIRM_S"):
        assert sel.effective_spec(field, grids) == grids["fields"][field]
    assert sel.effective_spec("V_STOP", grids) == grids["fields"]["V_STOP"]
    assert sel.apply("RESTART_K", None, {})["decision"] == "removed_by_precedence"


def test_inert_needs_every_match_in_both_views():
    almost = [0.600, 0.601, 0.6015, 0.600, 0.600]
    quiet = table([almost] * 3, [almost] * 5)
    out = sel.select_grid("F", spec("all", inert_fallback="U: fallback"), GRID5,
                          {"raw": quiet, "matched": quiet}, "raw")
    assert out["decision"] == "inert" and out["selected"] is None
    assert out["fallback"] == "U: fallback"
    # The other view moves one match: not inert, and the primary view selects.
    loud = dict(quiet, **{"DFL-J03WPY": peaked(1)})
    out = sel.select_grid("F", spec("all"), GRID5, {"raw": quiet, "matched": loud}, "raw")
    assert out["decision"] == "value"
    # One view alone cannot show inertness.
    out = sel.select_grid("F", spec("all"), GRID5, {"raw": quiet}, "raw")
    assert out["decision"] == "value" and out["details"]["inert_needs_both_views"] is True


def test_tie_to_when_the_judged_plateau_holds_the_current_value():
    grid = [1.0, 1.25, 1.5, 2.0, 2.5]
    wide = [0.70, 0.70, 0.70, 0.60, 0.50]             # P_m = {1.0, 1.25, 1.5}
    raw = table([flat()] * 3, [wide] * 5)
    s = spec("dfl", grid=grid, tie_to="V_STOP")
    out = sel.select_grid("F", s, grid, {"raw": raw, "matched": raw}, "raw", tie_value=1.5)
    assert out["decision"] == "tie_to" and out["selected"] == 1.5
    assert out["tie_to"] == "V_STOP"
    assert out["details"]["judged_plateau"] == [1.0, 1.25, 1.5]
    # A current value outside the judged plateau: the median rule selects.
    out = sel.select_grid("F", s, grid, {"raw": raw, "matched": raw}, "raw", tie_value=2.0)
    assert out["decision"] == "value" and out["selected"] == 1.25
    # A current value that is not a grid value is never in the judged plateau.
    out = sel.select_grid("F", s, grid, {"raw": raw, "matched": raw}, "raw", tie_value=1.4)
    assert out["decision"] == "value"
    assert out["details"]["tie_to"]["in_grid"] is False


def test_judged_plateau_needs_a_strict_weighted_majority():
    # Judge "all": the three Metrica matches (weight 1/2) have 1.5 on their plateau, DFL not.
    grid = [1.0, 1.25, 1.5, 2.0, 2.5]
    raw = table([peaked(2)] * 3, [peaked(0)] * 5)
    s = spec("all", grid=grid, tie_to="V_STOP")
    out = sel.select_grid("F", s, grid, {"raw": raw, "matched": raw}, "raw", tie_value=1.5)
    assert 1.5 not in out["details"]["judged_plateau"]
    assert out["details"]["plateau_share"][2] == 0.5
    assert out["decision"] == "value"


# -- bucket c --------------------------------------------------------------------------------

def mech(spells, low, high):
    return {g: {"spells": list(spells[i]), "low": low[i], "high": high[i]}
            for i, g in enumerate(METRICA)}


C_SPEC = {"bucket": "c", "judge": "metrica", "grid": ["release", "release_or_control"]}
CANDS = ["release", "release_or_control"]


def test_mechanism_winner_must_win_both_views_and_be_non_negative():
    # Base is release_or_control (index 1). The intervals are release minus the base.
    better = mech([[0.70, 0.65]] * 3, [0.01] * 3, [0.08] * 3)
    out = sel.select_mechanism("P", C_SPEC, CANDS, {"raw": better, "matched": better}, 1,
                               "release_or_control")
    assert out["decision"] == "value" and out["selected"] == "release"
    # The views disagree: the tie candidate.
    worse = mech([[0.60, 0.65]] * 3, [-0.08] * 3, [-0.01] * 3)
    out = sel.select_mechanism("P", C_SPEC, CANDS, {"raw": better, "matched": worse}, 1,
                               "release_or_control")
    assert out["decision"] == "tie" and out["selected"] == "release_or_control"
    # The winner of both medians is negative on one match: the tie candidate.
    mixed = mech([[0.70, 0.65], [0.70, 0.65], [0.60, 0.65]],
                 [0.01, 0.01, -0.08], [0.08, 0.08, -0.01])
    out = sel.select_mechanism("P", C_SPEC, CANDS, {"raw": mixed, "matched": mixed}, 1,
                               "release_or_control")
    assert out["details"]["views"]["raw"]["winner"] == "release"
    assert out["decision"] == "tie"


def test_mechanism_plateau_tie_goes_to_the_tie_candidate():
    level = mech([[0.650, 0.651]] * 3, [-0.003] * 3, [0.001] * 3)
    out = sel.select_mechanism("P", C_SPEC, CANDS, {"raw": level, "matched": level}, 1,
                               "release_or_control")
    assert out["details"]["views"]["raw"]["winner"] is None
    assert out["decision"] == "tie" and out["selected"] == "release_or_control"


K3 = [["release", "control"], ["release", "control", "dribble"],
      ["release", "control", "dribble", "touch"]]
K_SPEC = {"bucket": "c", "judge": "metrica", "grid": K3, "tie": "fewer kinds"}


def mech3(spells, versus):
    """Three-candidate entries: `versus` is [(low, high) or None per candidate] on every match."""
    return {g: {"spells": list(spells[i]), "versus": versus} for i, g in enumerate(METRICA)}


def test_three_candidates_base_wins_when_it_wins_both_views_and_is_non_negative():
    # Base is index 1. The two others are below it with intervals below zero.
    level = mech3([[0.60, 0.70, 0.60]] * 3, [(-0.12, -0.08), None, (-0.12, -0.08)])
    out = sel.select_mechanism("K", K_SPEC, K3, {"raw": level, "matched": level}, 1, K3[0])
    assert out["decision"] == "value" and out["selected"] == K3[1]
    # A challenger whose interval against the base reaches above zero: the base is not
    # non-negative against it, so the tie candidate (fewer kinds) takes the field.
    close = mech3([[0.60, 0.70, 0.60]] * 3, [(-0.12, -0.08), None, (0.001, 0.02)])
    out = sel.select_mechanism("K", K_SPEC, K3, {"raw": close, "matched": close}, 1, K3[0])
    assert out["decision"] == "tie" and out["selected"] == K3[0]


def test_three_candidates_half_index_median_has_no_winner():
    # The plateau holds the outer two candidates: o_m is the middle index on every match, and
    # a plateau of indices 1 and 2 gives o_m 1.5, a snapping tie.
    split = mech3([[0.60, 0.70, 0.70]] * 3, [(-0.12, -0.08), None, (-0.01, 0.01)])
    out = sel.select_mechanism("K", K_SPEC, K3, {"raw": split, "matched": split}, 1, K3[0])
    assert out["details"]["views"]["raw"]["winner"] is None
    assert out["decision"] == "tie" and out["selected"] == K3[0]


def test_three_candidates_non_base_winner_needs_pairwise_intervals():
    top = mech3([[0.80, 0.70, 0.60]] * 3, [(0.05, 0.12), None, (-0.12, -0.08)])
    with pytest.raises(SystemExit):
        sel.select_mechanism("K", K_SPEC, K3, {"raw": top, "matched": top}, 1, K3[0])


def test_effective_spec_reads_the_bucket_c_record():
    grids = recorded_grids()
    out = sel.effective_spec("OWNER_RELEASE_KINDS", grids)
    assert out["bucket"] == "c" and out["judge"] == "metrica" and out["grid"] == K3
    out = sel.effective_spec("CONTACT_DETECTOR", grids)
    assert out["grid"] == ["bump", "dv3"]


# -- tables on disk --------------------------------------------------------------------------

def sweep_json(path, field, values, spells, view):
    rows = [{"value": v, "games": {g: {"spell_f1": spells[g][i],
                                       "versus_base": {"delta": 0.0, "low": None, "high": None}}
                                   for g in spells}} for i, v in enumerate(values)]
    with open(path, "w") as handle:
        json.dump({"field": field, "values": values, "games": sorted(spells), "rows": rows,
                   "view": view, "base_overrides": {}, "config_hashes": ["h"] * len(values),
                   "base_config_hash": "b", "no_effect": False}, handle)


def test_apply_reads_tables_by_view_and_checks_the_grid(tmp_path):
    raw = table([peaked(1), peaked(3), peaked(4)], [flat()] * 5)
    sweep_json(str(tmp_path / "r.json"), "F", GRID5, raw, "raw")
    sweep_json(str(tmp_path / "m.json"), "F", GRID5, raw, "matched")
    tables = sel.load_tables([str(tmp_path / "r.json"), str(tmp_path / "m.json")])
    out = sel.apply("F", spec("metrica", view="matched"), tables)
    assert out["decision"] == "value" and out["selected"] == 4
    assert out["details"]["primary_view"] == "matched"
    with pytest.raises(SystemExit):
        sel.apply("F", spec("metrica", grid=[1, 2, 3, 4, 6]), tables)
    with pytest.raises(SystemExit):
        sel.apply("G", spec("metrica"), tables)


def test_pooled_views_make_one_row_per_match_and_view():
    # Metrica raw peaks at index 1 and Metrica matched at index 3: six rows of weight 1/6.
    raw = table([peaked(1)] * 3, [peaked(4, step=0.004)] * 5)
    matched = table([peaked(3)] * 3, [peaked(4, step=0.004)] * 5)
    s = spec("metrica", view="pooled: raw + matched", recorded="DFL matched")
    primary = sel.parse_view(s["view"])
    out = sel.select_grid("F", s, GRID5, {"raw": raw, "matched": matched}, primary)
    assert out["o_m"]["METRICA-S1@raw"] == 1.0 and out["o_m"]["METRICA-S1@matched"] == 3.0
    assert out["details"]["weights"]["METRICA-S2@matched"] == "1/6"
    # Cumulative weight is exactly 1/2 at index 1: the median is 2.
    assert out["details"]["weighted_median_index"] == 2.0 and out["selected"] == 3
    # DFL is recorded only, so its guard does not move the value. The details keep the test.
    assert out["details"]["guard"] == "none: the other provider is recorded only"
    assert out["details"]["guard_info"]["passed"] is False


def test_recorded_only_other_provider_skips_the_guard_in_one_view():
    raw = table([peaked(1, 7)] * 3, [peaked(5, 7, step=0.004)] * 5)
    s = spec("metrica", grid=GRID7, recorded="DFL raw")
    out = sel.select_grid("F", s, GRID7, {"raw": raw, "matched": raw}, "raw")
    assert out["selected"] == 20 and out["details"]["guard_info"]["would_move_to"] == 4
