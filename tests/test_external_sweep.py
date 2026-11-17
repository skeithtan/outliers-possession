"""tools/analysis/external_sweep.py: the table, plateau and argmax logic of the external sweep.

The tests score tiny synthetic references and runs in memory with tools/score_game.py. They run
no engine and need no match data.
"""

import importlib.util
import json
import os

import numpy as np
import pytest

from tools import score_game as sg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "external_sweep", os.path.join(ROOT, "tools", "analysis", "external_sweep.py"))
es = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(es)

H, A, D, X = 0, 1, 2, 3


def _seq(*runs):
    return np.concatenate([np.full(n, k, dtype=np.int8) for k, n in runs])


# Two references with several possessions each. "perfect" copies the reference, "merge" joins
# two possessions, "shift" moves every edge a little.
REF = {
    "DFL-T1": _seq((X, 5), (H, 60), (A, 30), (H, 20), (D, 40), (A, 80), (H, 50), (X, 5)),
    "METRICA-T2": _seq((H, 90), (A, 15), (H, 70), (D, 30), (H, 40), (A, 60)),
}
RUNS = {
    "DFL-T1": {
        "perfect": REF["DFL-T1"],
        "merge": _seq((X, 5), (H, 110), (H, 0), (D, 40), (A, 80), (H, 50), (X, 5)),
        "shift": _seq((X, 5), (H, 70), (A, 30), (H, 15), (D, 40), (A, 75), (H, 50), (X, 5)),
    },
    "METRICA-T2": {
        "perfect": REF["METRICA-T2"],
        "merge": _seq((H, 175), (D, 30), (H, 40), (A, 60)),
        "shift": _seq((H, 95), (A, 15), (H, 60), (D, 35), (H, 45), (A, 55)),
    },
}


def _records(name):
    return {g: es.score_record(REF[g], RUNS[g][name]) for g in sorted(REF)}


def test_score_record_matches_score_game():
    for game in sorted(REF):
        rec = es.score_record(REF[game], RUNS[game]["shift"])
        result, _, pred = sg.score_codes(REF[game], RUNS[game]["shift"])
        assert rec["spell_f1"] == pytest.approx(result["spell_f1_mean"])
        assert rec["possessions"] == pred["possessions"]
        flags = es._flags(rec)
        every = np.arange(flags.shape[1])
        assert es.ps.mean_f1(flags, rec["n_pred_spells"], every) == pytest.approx(
            rec["spell_f1"])


def test_paired_game_equals_score_game_paired_delta():
    game = "DFL-T1"
    a, b = RUNS[game]["shift"], RUNS[game]["merge"]
    ours = es.paired_game(es.score_record(REF[game], a), es.score_record(REF[game], b), 200, 0)
    theirs = sg.paired_delta(REF[game], a, b, 200, 0)
    for key in ("delta", "low", "high"):
        assert ours[key] == pytest.approx(theirs[key])


def test_field_table_means_argmax_and_plateau():
    values = [10, 20, 30]
    scores = [_records("merge"), _records("perfect"), _records("shift")]
    table = es.field_table("FIELD", values, scores, _records("shift"), n_draws=200, seed=0)
    assert table["games"] == ["DFL-T1", "METRICA-T2"]
    perfect = table["rows"][1]
    assert perfect["mean"] == pytest.approx(1.0)
    assert perfect["mean_dfl"] == pytest.approx(1.0)
    assert perfect["mean_metrica"] == pytest.approx(1.0)
    merge = table["rows"][0]
    expected = np.mean([scores[0][g]["spell_f1"] for g in table["games"]])
    assert merge["mean"] == pytest.approx(expected)
    assert table["best"] == 20
    assert table["argmax"] == {"DFL-T1": 20, "METRICA-T2": 20}
    assert table["median_argmax"] == 20.0
    # The best value is on the plateau. Its difference to itself is zero.
    assert perfect["plateau"]
    assert perfect["versus_best"] == {"mean_delta": 0.0, "low": 0.0, "high": 0.0}
    # The plateau holds exactly the values whose interval to the best reaches zero.
    for row in table["rows"]:
        assert row["plateau"] == (row["versus_best"]["high"] >= 0.0)
    assert table["plateau"] == [r["value"] for r in table["rows"] if r["plateau"]]
    # The base run is "shift": the third value has a zero difference to the base everywhere.
    assert table["rows"][2]["versus_base"]["mean_delta"] == 0.0
    for game in table["games"]:
        assert table["rows"][2]["games"][game]["versus_base"]["delta"] == 0.0


def test_field_table_ties_and_median_of_split_argmax():
    # Value 1 wins on DFL-T1 only, value 2 on METRICA-T2 only: the median is between them.
    one = {"DFL-T1": es.score_record(REF["DFL-T1"], RUNS["DFL-T1"]["perfect"]),
           "METRICA-T2": es.score_record(REF["METRICA-T2"], RUNS["METRICA-T2"]["merge"])}
    two = {"DFL-T1": es.score_record(REF["DFL-T1"], RUNS["DFL-T1"]["merge"]),
           "METRICA-T2": es.score_record(REF["METRICA-T2"], RUNS["METRICA-T2"]["perfect"])}
    table = es.field_table("FIELD", [1, 2, 3], [one, two, one], one, n_draws=50, seed=0)
    assert table["argmax"] == {"DFL-T1": 1, "METRICA-T2": 2}
    assert table["median_argmax"] == 1.5
    # Equal means go to the value listed first.
    assert table["rows"][0]["mean"] == table["rows"][2]["mean"]
    if table["rows"][0]["mean"] >= table["rows"][1]["mean"]:
        assert table["best"] == 1
    # A non-numeric argmax has no median.
    words = es.field_table("FIELD", ["a", "b"], [one, two], one, n_draws=10, seed=0)
    assert words["median_argmax"] is None
    assert not words["no_effect"]
    inert = es.field_table("FIELD", [1, 2], [one, one], one, n_draws=10, seed=0)
    assert inert["no_effect"]
    assert "No value changes any score" in es.format_markdown(inert)


def test_field_table_is_deterministic_and_markdown_renders():
    scores = [_records("merge"), _records("perfect")]
    first = es.field_table("F", [True, False], scores, _records("merge"), n_draws=100, seed=0)
    second = es.field_table("F", [True, False], scores, _records("merge"), n_draws=100, seed=0)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    page = es.format_markdown(first, {"OTHER": 1})
    assert "| argmax |" in page
    assert "`{\"OTHER\": 1}`" in page
    # One row in the spellF1 table, one row per match in the table against the base run.
    assert page.count("\n| true | ") == 1 + len(first["games"])


def test_field_table_rejects_mismatched_matches():
    scores = [_records("perfect")]
    base = {"DFL-T1": scores[0]["DFL-T1"]}
    with pytest.raises(ValueError):
        es.field_table("F", [1], scores, base, n_draws=10, seed=0)


def test_check_games_refuses_competition_and_sealed_matches():
    with pytest.raises(SystemExit):
        es.check_games(["BAR-ATH"])
    with pytest.raises(SystemExit):
        es.check_games(["BAR-ATH"], allow_sealed=True)
    for game in es.SEALED_GAMES:
        with pytest.raises(SystemExit):
            es.check_games(["DFL-J03WMX", game])
        assert game in es.check_games([game], allow_sealed=True)
    assert es.check_games(list(es.OPEN_GAMES)) == sorted(es.OPEN_GAMES)


def test_values_parse_as_json_literals(tmp_path):
    assert es.load_sweeps("F", ["10", "1.5", "true", "[1, 2]", "\"s\"", "word"]) == [
        ("F", [10, 1.5, True, [1, 2], "s", "word"])]
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"B": [1, 2], "A": [0.5]}))
    assert es.load_sweeps(grid=str(grid)) == [("A", [0.5]), ("B", [1, 2])]


def test_run_key_depends_on_the_full_config():
    resolved, _ = es.resolve_config({})
    same, _ = es.resolve_config({"NEAREST_LAG": resolved["NEAREST_LAG"]})
    other, _ = es.resolve_config({"NEAREST_LAG": resolved["NEAREST_LAG"] + 1})
    key = es.run_key("DFL-J03WMX", resolved, "/r", "src")
    assert key == es.run_key("DFL-J03WMX", same, "/r", "src")
    assert key != es.run_key("DFL-J03WMX", other, "/r", "src")
    assert key != es.run_key("DFL-J03WN1", resolved, "/r", "src")
    assert key != es.run_key("DFL-J03WMX", resolved, "/r", "src2")


def test_provider_weights_balance_the_two_providers():
    weights = es.provider_weights(list(es.OPEN_GAMES))
    for game in es.OPEN_GAMES:
        expected = 1.0 / 6 if game.startswith("METRICA-") else 1.0 / 10
        assert weights[game] == pytest.approx(expected)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert es.provider_weights(["DFL-T1", "METRICA-T2"]) == {"DFL-T1": 0.5, "METRICA-T2": 0.5}


def test_weighted_cross_match_with_equal_weights_equals_paired_summary():
    a, b = _records("shift"), _records("merge")
    equal = {g: 1.0 for g in a}
    ours = es.weighted_cross_match(a, b, equal, 200, 0)
    theirs = es.cross_match(a, b, 200, 0)
    for key in ("mean_delta", "low", "high"):
        assert ours[key] == pytest.approx(theirs[key])
    # All the weight on one match gives that match's own difference.
    lopsided = es.weighted_cross_match(a, b, {"DFL-T1": 1.0, "METRICA-T2": 0.0}, 200, 0)
    assert lopsided["mean_delta"] == pytest.approx(
        a["DFL-T1"]["spell_f1"] - b["DFL-T1"]["spell_f1"])


def test_provider_mode_adds_the_balanced_mean_and_its_plateau():
    scores = [_records("merge"), _records("perfect"), _records("shift")]
    table = es.field_table("F", [1, 2, 3], scores, _records("shift"), n_draws=100, seed=0,
                           weights="provider")
    for row, per_game in zip(table["rows"], scores):
        expected = 0.5 * per_game["DFL-T1"]["spell_f1"] + 0.5 * per_game["METRICA-T2"]["spell_f1"]
        assert row["mean_provider"] == pytest.approx(expected)
        assert row["plateau_provider"] == (row["versus_best_provider"]["high"] >= 0.0)
    assert table["best_provider"] == 2
    assert 2 in table["plateau_provider"]
    assert "provider mean" in es.format_markdown(table)
    plain = es.field_table("F", [1, 2, 3], scores, _records("shift"), n_draws=10, seed=0)
    assert "mean_provider" not in plain["rows"][0]


def test_match_plateaus_and_midpoints():
    def rec(spell):
        return {"spell_f1": spell}
    values = [10, 14, 17, 20]
    scores = [{"G1": rec(0.500), "G2": rec(0.400)},
              {"G1": rec(0.5015), "G2": rec(0.410)},
              {"G1": rec(0.502), "G2": rec(0.4085)},
              {"G1": rec(0.490), "G2": rec(0.420)}]
    out = es.match_plateaus(values, scores, ["G1", "G2"], tolerance=0.002)
    # G1: best 0.502 at index 2. 0.5015 (index 1) and 0.500 (index 0) lie within 0.002.
    assert out["G1"]["indices"] == [0, 1, 2]
    assert out["G1"]["values"] == [10, 14, 17]
    assert out["G1"]["midpoint_index"] == 1.0
    assert out["G1"]["midpoint_value"] == 14
    # G2: a single best value.
    assert out["G2"]["indices"] == [3]
    assert out["G2"]["midpoint_value"] == 20
    half = es.match_plateaus(values, scores, ["G1"], tolerance=0.0016)
    assert half["G1"]["indices"] == [1, 2]
    assert half["G1"]["midpoint_index"] == 1.5
    assert half["G1"]["midpoint_value"] == 15.5


def test_view_roots_and_keys():
    roots = ("/refs", "/sample", "/m")
    assert es.data_root("DFL-J03WMX", "raw", *roots) == "/refs"
    assert es.data_root("METRICA-S1", "raw", *roots) == "/refs"
    assert es.data_root("METRICA-S3", "raw", *roots) == "/sample"
    for game in es.OPEN_GAMES:
        assert es.data_root(game, "sigma95", *roots) == os.path.join("/m", "refs-smooth")
        expected = "refs-smooth-s3" if game.startswith("DFL-") else "refs-smooth-s5"
        assert es.data_root(game, "matched", *roots) == os.path.join("/m", expected)
    resolved, _ = es.resolve_config({})
    assert es.run_key("DFL-J03WMX", resolved, "/r", "src", "raw") != es.run_key(
        "DFL-J03WMX", resolved, "/r", "src", "matched")
    assert es.view_dir("/c", "raw") != es.view_dir("/c", "matched")


# -- SkillCorner, the third provider --------------------------------------------------------------

def test_skillcorner_matches_are_accepted_on_the_raw_view_only():
    names = sorted(es.SKILLCORNER_REFERENCE_MD5)
    assert len(names) == 20
    assert es.check_games(names[:2] + ["DFL-J03WMX"]) == sorted(names[:2] + ["DFL-J03WMX"])
    roots = ("/refs", "/sample", "/m")
    assert es.data_root(names[0], "raw", *roots) == "/refs"
    for view in ("matched", "sigma95"):
        with pytest.raises(SystemExit):
            es.data_root(names[0], view, *roots)
    # The default stays the eight open references.
    assert not any(g.startswith(es.SKILLCORNER_PREFIX) for g in es.OPEN_GAMES)


def test_skillcorner_reference_path_and_md5(tmp_path):
    game = "SC-1874553"
    folder = tmp_path / "skillcorner" / "1874553"
    folder.mkdir(parents=True)
    path = folder / "SC-1874553_reference.csv"
    path.write_text("frame,label\n0,X\n")
    with pytest.raises(SystemExit, match="differs from the recorded"):
        es.prepare_reference(game, str(tmp_path), "/sample", str(tmp_path / "cache"))
    es.SKILLCORNER_REFERENCE_MD5[game], saved = es.md5_of(str(path)), \
        es.SKILLCORNER_REFERENCE_MD5[game]
    try:
        out, digest = es.prepare_reference(game, str(tmp_path), "/sample",
                                           str(tmp_path / "cache"))
    finally:
        es.SKILLCORNER_REFERENCE_MD5[game] = saved
    assert out == str(path) and digest == es.md5_of(str(path))


def test_skillcorner_mean_and_column_appear_only_with_a_skillcorner_match():
    scores = [_records("merge"), _records("perfect")]
    plain = es.field_table("F", [1, 2], scores, _records("merge"), n_draws=10, seed=0,
                           weights="provider")
    assert "mean_skillcorner" not in plain["rows"][0]
    assert "SkillCorner" not in es.format_markdown(plain)
    with_sc = [dict(s, **{"SC-T3": s["DFL-T1"]}) for s in scores]
    base = dict(_records("merge"), **{"SC-T3": _records("merge")["DFL-T1"]})
    table = es.field_table("F", [1, 2], with_sc, base, n_draws=10, seed=0, weights="provider")
    row = table["rows"][1]
    assert row["mean_skillcorner"] == pytest.approx(with_sc[1]["SC-T3"]["spell_f1"])
    assert table["weights"] == {"DFL-T1": pytest.approx(1.0 / 3),
                                "METRICA-T2": pytest.approx(1.0 / 3),
                                "SC-T3": pytest.approx(1.0 / 3)}
    page = es.format_markdown(table)
    assert "| mean | Metrica | DFL | SkillCorner |" in page
