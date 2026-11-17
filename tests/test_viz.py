"""P6-V: the visualizer exporters in tools/viz (plan "Tests", site/data/SCHEMA.md).

Synthetic tests need no data. The `slow` tests run both exporters on the organisers' baseline
`baseline_smooth50` and need the data bundle and the scorer reports. They write to `tmp_path`
only. Tests may read the public GT (through tools/gt.py).
"""

import ast
import copy
import glob
import json
import os

import numpy as np
import pytest

import games
from tools._common import pm
from tools.viz import build_replay, build_site, config_doc, guard, ledger, quant, rle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIZ_DIR = os.path.join(ROOT, "tools", "viz")
RESULTS = os.path.join(ROOT, "evaluation", "results")
REPORTS = os.path.join(ROOT, "evaluation", "reports")
SMOOTH50 = os.path.join(RESULTS, "baseline_smooth50.csv")
SMOOTH50_MD5 = "23924df8d47236bfacf527d93079fc66"   # experiments.md, p0c_baseline50_repro
SMOOTH50_SPELL_F1 = 0.3256                          # experiments.md, baseline_smooth50
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
    """[('H', 40), ('D', 20), ...] to int8 codes."""
    return np.concatenate([np.full(n, "HADX".index(c), dtype=np.int8) for c, n in spec])


# -- rle ----------------------------------------------------------------------------------------

def test_rle_round_trip_and_equal_to_scorer_segments():
    codes = _codes([("X", 10), ("H", 40), ("D", 20), ("A", 5), ("H", 25), ("X", 10)])
    runs = rle.encode(codes)
    assert rle.decode(runs, len(codes)).tolist() == codes.tolist()
    assert all(a[2] != b[2] for a, b in zip(runs, runs[1:]))
    graded = [(10, 100)]
    ours = [(a, b, "HADX".index(k)) for a, b, k in rle.encode(codes, 10, 100)]
    assert ours == pm.segments(codes, graded)
    assert rle.encode(codes, 30, 30) == []


# -- quant --------------------------------------------------------------------------------------

def test_i16_within_half_a_centimetre_with_sentinel_and_overflow():
    values = np.array([0.0, 1.234567, -52.65, 105.3, np.nan, 327.66])
    steps = quant.encode_i16(values)
    assert steps.dtype == np.int16
    assert steps[4] == quant.I16_SENTINEL
    back = quant.decode(steps)
    assert np.isnan(back[4])
    finite = ~np.isnan(values)
    assert np.all(np.abs(back[finite] - values[finite]) <= 0.005 + 1e-12)
    with pytest.raises(OverflowError):
        quant.encode_i16(np.array([327.68]))
    with pytest.raises(OverflowError):
        quant.encode_u16(np.array([-0.5]))
    with pytest.raises(ValueError):
        quant.encode_u8(np.array([0.5, np.nan]))


def test_pack_gives_even_offsets_and_a_complete_header():
    arrays = [("a", np.array([1, -1, 3], dtype=np.int8)),
              ("b", np.array([[1, 2], [3, 4]], dtype=np.int16)),
              ("c", np.array([7], dtype=np.uint8)),
              ("d", np.array([65535, 1], dtype=np.uint16))]
    blob, header = quant.pack(arrays)
    assert list(header) == ["a", "b", "c", "d"]
    for name, array in arrays:
        entry = header[name]
        assert entry["offset"] % 2 == 0
        assert entry["shape"] == list(array.shape)
        assert entry["dtype"] == array.dtype.name
    views = quant.unpack(blob, header)
    for name, array in arrays:
        assert views[name].tolist() == array.tolist()
    with pytest.raises(ValueError):
        quant.pack([("e", np.array([1.0]))])


# -- guard --------------------------------------------------------------------------------------

WINDOW = (11185, 18685)


def test_guard_refuses_coordinate_keys_in_any_case():
    for key in ("x", "Px", "BY", "speed", "dH", "Dmin", "positions"):
        with pytest.raises(guard.LeakError):
            guard.assert_no_coordinates({"rows": [{key: 1.0}]}, "scores.json", WINDOW)
    guard.assert_no_coordinates({"xg": 1.0, "speed_rule": "text"}, "scores.json", WINDOW)


def test_guard_refuses_long_series_frame_length_lists_and_blobs():
    with pytest.raises(guard.LeakError):
        guard.assert_no_coordinates({"a": list(range(guard.MAX_NUMERIC_LEAVES + 1))},
                                    "scores.json", WINDOW)
    with pytest.raises(guard.LeakError):
        guard.assert_no_coordinates({"a": [[1, 2]] * 10001}, "scores.json", WINDOW)
    with pytest.raises(guard.LeakError):
        guard.assert_no_coordinates({"a": ["H"] * 50}, "scores.json", WINDOW, n_frames=50)
    with pytest.raises(guard.LeakError):
        guard.assert_no_coordinates({"a": "QUJD" * 1100}, "scores.json", WINDOW)
    guard.assert_no_coordinates({"note": "a long note, with words. " * 300}, "ladder.json",
                                WINDOW)


def test_guard_window_rule_applies_to_non_exempt_files_only():
    inside = {"events": [[11185, "touch", "H", 1.0]], "stoppages": [[12000, 18685]],
              "ours": [[11185, 18685, "H"]], "rec": {"gt_frame": 18684}}
    guard.assert_no_coordinates(inside, "window_detail.json", WINDOW)
    for bad in ({"events": [[20000, "touch", "H", 1.0]]},
                {"stoppages": [[11000, 11300]]},
                {"ours": [[11185, 18686, "H"]]},
                {"rec": {"our_frame": 18685}},
                {"tags": [{"start_frame": 11000, "end_frame": 11300}]}):
        with pytest.raises(guard.LeakError):
            guard.assert_no_coordinates(bad, "transitions.json", WINDOW)
    rows = {"ours": [{"start": 0, "stop": 1500}], "runs": [[0, 11185, "X"]]}
    guard.assert_no_coordinates(rows, "minute_bins.json", WINDOW)
    guard.assert_no_coordinates(rows, "labels_baseline_smooth50.json", WINDOW)
    with pytest.raises(guard.LeakError):
        guard.assert_no_coordinates(rows, "labels_window.json", WINDOW)
    with pytest.raises(guard.LeakError):  # a manifest cannot exempt a never-exempt file
        guard.assert_no_coordinates(rows, "labels_window.json", WINDOW,
                                    exempt=["labels_window.json"])


def test_guard_manifest_limits_refuse_a_wide_window_and_unknown_exemptions():
    good = {"public_window": {"start": 100, "stop": 400}, "exempt": ["scores.json"],
            "n_frames": 3000}
    assert guard.manifest_limits(good) == ((100, 400), ["scores.json"], 3000)
    wide = dict(good, public_window={"start": 0, "stop": 7501})
    with pytest.raises(guard.LeakError):
        guard.manifest_limits(wide)
    with pytest.raises(guard.LeakError):
        guard.manifest_limits(dict(good, exempt=["*.json"]))


def test_guard_imports_only_the_standard_library():
    with open(os.path.join(VIZ_DIR, "guard.py")) as handle:
        tree = ast.parse(handle.read())
    allowed = {"argparse", "fnmatch", "json", "os", "re", "sys"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert {alias.name.split(".")[0] for alias in node.names} <= allowed
        if isinstance(node, ast.ImportFrom):
            assert node.module.split(".")[0] in allowed


def test_viz_sources_never_import_the_wall_clock():
    for path in sorted(glob.glob(os.path.join(VIZ_DIR, "*.py"))):
        with open(path) as handle:
            tree = ast.parse(handle.read())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            assert not {"datetime", "time"} & set(names), path


# -- assemble and write_bundle on synthetic inputs -----------------------------------------------

def _external_view(delta):
    """One view of a synthetic docs/technical/data/external.json: summary and per match."""
    interval = {"mean_delta": delta, "low": delta - 0.01, "high": delta + 0.01}
    recall = [{"bucket": "< 2 s", "n": 10, "recalled": 2, "recall": 0.2}]
    group = {"engine_spell_f1": 0.6, "engine_macro_f1": 0.8, "baseline50_spell_f1": 0.4,
             "baseline50_macro_f1": 0.7, "engine_minus_baseline": interval,
             "engine_recall_by_duration": recall}
    match = {"engine": {"spell_f1": 0.61, "macro_f1": 0.81},
             "baseline50": {"spell_f1": 0.41, "macro_f1": 0.71},
             "engine_minus_baseline": {"delta": delta, "low": delta - 0.02, "high": delta + 0.02},
             "reference_possessions": 300}
    return {"summary": {"metrica": group, "dfl": group, "all": group},
            "matches": {"METRICA-S1": match, "DFL-J03WMX": match}}


def _synthetic_external():
    """A synthetic docs/technical/data/external.json with two open and one sealed match."""
    side = {"mean_delta": -0.02, "low": -0.03, "high": -0.01, "metrica": -0.04, "dfl": 0.0,
            "value": False}
    oracle = {g: {"ours": 0.5, "oracle_team": 0.9, "oracle_dead": 0.6}
              for g in ("metrica", "dfl", "all")}
    return {"open": {"raw": _external_view(0.2), "matched": _external_view(0.15)},
            "sealed": {"DFL-J03WQQ": {"raw": {"spell_f1": 0.49, "macro_f1": 0.63},
                                      "matched": {"spell_f1": 0.45, "macro_f1": 0.61}}},
            "ablations": [{"field": "DRIBBLE_FRAMES", "group": "owner", "label": "no dribbles",
                           "raw": side, "matched": side}],
            "oracle": {"raw": oracle, "matched": oracle},
            "sensitivity": [], "bar_ath_baseline50_full_match": 0.2433}


def _synthetic_inputs():
    """Inputs of build_site.assemble() for a 3000-frame match with a window [200, 500)."""
    from tools import sanity

    n = 3000
    blocks = [(1, 100, 1399), (2, 1600, 2899)]
    ours = np.full(n, X, dtype=np.int8)
    ours[100:1400] = _codes([("H", 300), ("D", 200), ("A", 800)])
    ours[1600:2900] = _codes([("A", 1000), ("H", 300)])
    gt = np.full(n, X, dtype=np.int8)
    gt[200:500] = _codes([("H", 150), ("A", 150)])
    base = ours.copy()
    base[700:710] = H
    stoppage = ours == D
    trace = [{"off": o, "frame": 350 + o, "dH": 1.0, "dA": 2.0, "v": 3.0, "dv_3": 0.5,
              "meanv": 1.1, "ours": "H", "gt": "A"} for o in (-3, 0, 3)]
    gt_record = {"gt_frame": 350, "from": "H", "to": "A", "kind": "H<->A", "status": "missed",
                 "ball_missing_at_f_minus_1": False, "our_frame": None, "our_from_to": None,
                 "latency": None, "trace": trace,
                 "tags": [{"code": "BUILD UP", "team": "H", "type": None, "start_frame": 50,
                           "end_frame": 360}], "events": []}
    spurious = {"our_frame": 400, "from": "H", "to": "D", "gt_label_at_frame": "A",
                "nearest_gt_transition": {"frame": 350, "from_to": "H->A", "distance": 50},
                "trace": trace, "tags": [], "events": [], "evidence": None}
    forensics_result = {
        "name": "synth", "tol": 25, "window": 50, "reappear_lookback": 30,
        "reports_stale": False, "events_module": True, "gt_segments": [],
        "gt_transitions": [gt_record], "spurious": [spurious],
        "latency_by_kind": {"H<->A": {"n_gt": 1, "n_matched": 0}},
        "kickoffs": [{"kind": "half 1 kickoff", "frame": 200, "rule": "mark", "team": "H",
                      "mark_dist_m": 0.3, "ours": "H", "goals_end": None, "goals_team": None},
                     {"kind": "half 2 kickoff", "frame": 1700, "rule": "law8", "team": "A",
                      "mark_dist_m": None, "ours": "A", "goals_end": None, "goals_team": None}],
        "sandwiches": [], "summary": {"spell_f1_mean": 0.5}}
    stats = sanity.sanity_stats(ours, blocks=blocks, use_tags=False, use_ball=False)
    base_stats = sanity.sanity_stats(base, blocks=blocks, use_tags=False, use_ball=False)
    return {
        "version": "synth", "md5": "0" * 32, "game": "SYNTH", "n_frames": n, "fps": 25,
        "pitch_m": [105.3, 68.0], "team_names": {"H": "Home", "A": "Away"},
        "blocks": blocks, "kickoffs": {1: 100, 2: 1700},
        "ours": ours, "gt": gt,
        "baselines": {"baseline_smooth50": base, "baseline_smooth125": base.copy()},
        "events": [(250, "release", "H", 0.91234), (700, "touch", "A", 1.0),
                   (1650, "control", None, 0.5)],
        "owner_events": [(250, "H", "release", 0.91234), (1650, "A", None, None)],
        "onsets": [(400, "out", 0.8), (1000, "pursuit_stop", 0.7)],
        "restarts": [(450, "H", "kick", 0.9, True), (1100, "A", "reappear", 0.6, False)],
        "stoppage": stoppage,
        "reports": {"synth": {"spell_f1_mean": 0.5, "macro_f1": 0.6, "n_pred_spells": 2,
                              "spell_bootstrap_low": None, "spell_bootstrap_high": None,
                              "mean_spell_duration_s": 6.0, "gt_mean_spell_duration_s": 6.0}},
        "forensics": forensics_result,
        "sanity": {"ours": stats, "baseline_smooth50": base_stats},
        "limits": {"possessions": [180, 450], "d_share": [0.18, 0.55], "min_d_segments": 55,
                   "max_graded_x": 0},
        "watched": {"possessions": [230, 330], "d_segments": [85, 150],
                    "min_tag_agreement": 0.85},
        "config": {"config_hash": "abcdef0123", "n_fields": 0, "sections": []},
        "external": _synthetic_external(),
        "ladder": {"rows": [], "rejected": []}, "cycles": [],
        "git": {"commit": "f" * 40, "commit_date": "2026-11-01T00:00:00+00:00", "tag": None,
                "dirty": False},
        "team": ["One", "Two"],
        "repro": {"command": "python -m engine.run --out x.csv", "python": "3.9", "pins": {},
                  "n_tests": 1}}


def test_assemble_is_pure_and_gives_every_required_key():
    inputs = _synthetic_inputs()
    before = copy.deepcopy(inputs)
    first = build_site.assemble(inputs)
    second = build_site.assemble(inputs)
    assert {k: build_site.serialize(v) for k, v in first.items()} == \
        {k: build_site.serialize(v) for k, v in second.items()}
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, inputs[key])
        else:
            assert build_site.serialize(value) == build_site.serialize(inputs[key]), key

    assert set(first) == {"manifest.json", "labels_window.json", "labels_baseline_smooth50.json",
                          "labels_baseline_smooth125.json", "minute_bins.json",
                          "window_detail.json", "scores.json", "transitions.json", "sanity.json",
                          "config.json", "external_summary.json", "ladder.json",
                          "cycles.json"}
    manifest = first["manifest.json"]
    for key in ("schema", "version", "md5", "config_hash", "git", "game", "n_frames", "fps",
                "pitch_m", "team_names", "graded_blocks", "kickoffs", "public_window", "exempt",
                "team", "repro", "files"):
        assert key in manifest
    assert manifest["public_window"] == {"start": 200, "stop": 500}
    assert manifest["graded_blocks"] == [[100, 1400], [1600, 2900]]
    assert set(manifest["files"]) == set(first) - {"manifest.json"}

    window = first["labels_window.json"]
    assert set(window) == {"window", "ours", "gt_public", "baseline_smooth50",
                           "baseline_smooth125"}
    assert all(200 <= a < b <= 500 for a, b, _ in window["ours"])
    assert rle.decode(window["gt_public"], 3000)[200:500].tolist() == inputs["gt"][200:500].tolist()

    bins = first["minute_bins.json"]
    assert bins["n_bins"] == 2 and len(bins["ours"]) == 2
    row = bins["ours"][0]
    assert (row["start"], row["stop"], row["half"]) == (100, 1400, 1)
    assert row["n_possession_starts"] == 2 and row["n_d_segments"] == 1
    assert row["restarts_by_kind"] == {"kick": 1}
    assert abs(row["share_h"] + row["share_a"] + row["share_d"] + row["share_x"] - 1) < 1e-3
    assert bins["baseline_smooth50"][0]["events_by_kind"] == {}

    detail = first["window_detail.json"]
    assert detail["events"] == [[250, "release", "H", 0.912]]
    assert detail["onsets"] == [[400, "out", 0.8]]
    assert detail["restarts"] == [[450, "H", "kick", 0.9, True]]
    assert detail["stoppages"] == [[400, 500]]

    trans = first["transitions.json"]
    for record in trans["gt_transitions"] + trans["spurious"]:
        assert all(set(r) == {"off", "ours", "gt"} for r in record["labels_trace"])
    assert trans["gt_transitions"][0]["tags"][0]["start_frame"] == 200
    assert [k["frame"] for k in trans["kickoffs"]] == [200]
    assert "mark_dist_m" not in trans["kickoffs"][0]
    assert set(first["scores.json"]["synth"]) >= {"spell_f1_mean", "bootstrap", "n_spells",
                                                  "duration_s"}
    assert set(first["sanity.json"]) == {"ours", "baseline_smooth50", "limits", "watched"}


def test_external_summary_keeps_summary_numbers_of_other_matches_only():
    ext = _synthetic_external()
    out = build_site.external_summary(ext)
    assert set(out) == {"summary", "matches", "sealed", "ablations", "oracle"}
    assert [m["game"] for m in out["matches"]] == ["DFL-J03WMX", "METRICA-S1"]
    assert out["matches"][1]["provider"] == "metrica"
    assert out["summary"]["all"]["own"]["engine_minus_baseline"] == \
        {"mean": 0.2, "low": 0.19, "high": 0.21}
    assert out["sealed"][0]["atd"]["engine_spell_f1"] == 0.45
    assert out["ablations"][0]["own"]["mean"] == -0.02
    assert out["oracle"]["own"]["metrica"]["reference_teams"] == 0.9
    text = build_site.serialize(out).decode("utf-8")
    assert "BAR-ATH" not in text and "bar_ath" not in text
    assert guard.problems(out, "external_summary.json", (200, 500), n_frames=3000) == []

    ext["sealed"]["BAR-ATH"] = ext["sealed"]["DFL-J03WQQ"]
    with pytest.raises(build_site.BuildError):
        build_site.external_summary(ext)


def test_config_with_classes_counts_each_class_and_leaves_the_input_alone():
    config = {"config_hash": "abc", "n_fields": 3, "sections": [
        {"title": "one", "fields": [{"name": "A_M"}, {"name": "B_S"}]},
        {"title": "two", "fields": [{"name": "C_NEW"}]}]}
    provenance = {"class_definitions": {"E": "external"},
                  "fields": [{"name": "A_M", "class": "E"}, {"name": "B_S", "class": "D"}]}
    out = build_site.config_with_classes(config, provenance)
    assert [f["class"] for s in out["sections"] for f in s["fields"]] == ["E", "D", None]
    assert out["classes"]["counts"] == {"E": 1, "F": 0, "U": 0, "D": 1, "P": 0}
    assert out["classes"]["definitions"] == {"E": "external"}
    assert "class" not in config["sections"][0]["fields"][0]


def test_write_bundle_guards_and_writes_nothing_on_a_leak(tmp_path):
    bundle = build_site.assemble(_synthetic_inputs())
    good = tmp_path / "good"
    sizes = build_site.write_bundle(str(good), bundle)
    assert guard.check_dir(str(good)) == []
    assert sizes["manifest.json"] == os.path.getsize(str(good / "manifest.json"))
    assert guard.main([str(good)]) == 0

    leaky = copy.deepcopy(bundle)
    leaky["window_detail.json"]["events"].append([2000, "touch", "H", 1.0])
    bad = tmp_path / "bad"
    with pytest.raises(guard.LeakError):
        build_site.write_bundle(str(bad), leaky)
    assert not bad.exists()

    leaky = copy.deepcopy(bundle)
    leaky["scores.json"]["synth"]["px"] = [1.0]
    with pytest.raises(guard.LeakError):
        build_site.write_bundle(str(bad), leaky)
    assert not bad.exists()

    (good / "planted.json").write_text('{"x": 1}\n')
    assert guard.main([str(good)]) == 1


# -- config_doc ---------------------------------------------------------------------------------

SYNTHETIC_CONFIG = '''
import dataclasses

@dataclasses.dataclass(frozen=True)
class Config:
    # -- first section (PLAN.md 1.1) -----------
    # Touch radius from the 13704 case (P1-A, C8).
    R: float = 1.5
    S: float = 2.0   # inline note, M13

    # -- second section (PLAN.md 5.3) -------
    # Shared block for two fields.
    LO: int = 1
    HI: int = 2
    NAMES: tuple = ("a",
                    "b")   # a list over two lines

    def method(self):
        return 1
'''


def test_config_doc_parses_a_synthetic_class():
    values = {"R": 1.5, "S": 2.0, "LO": 1, "HI": 2, "NAMES": ("a", "b")}
    sections = config_doc.parse(SYNTHETIC_CONFIG, values)
    assert [s["title"] for s in sections] == ["first section (PLAN.md 1.1)",
                                              "second section (PLAN.md 5.3)"]
    fields = {f["name"]: f for s in sections for f in s["fields"]}
    assert fields["R"]["sources"] == ["13704", "P1-A", "C8"]
    # S sits directly below R, so it shares the block of R and adds its inline comment.
    assert fields["S"]["comment"].endswith("(P1-A, C8). inline note, M13")
    assert fields["S"]["sources"][-1] == "M13"
    assert fields["HI"]["comment"] == "Shared block for two fields."
    assert fields["HI"]["sources"] == ["PLAN 5.3"]
    assert fields["NAMES"]["value"] == ["a", "b"] and fields["NAMES"]["type"] == "tuple"
    with pytest.raises(config_doc.ConfigDocError):
        config_doc.parse(SYNTHETIC_CONFIG, dict(values, MISSING=1))
    with pytest.raises(config_doc.ConfigDocError):
        config_doc.parse(SYNTHETIC_CONFIG, {k: v for k, v in values.items() if k != "S"})


def test_config_doc_covers_every_field_of_the_real_config():
    from engine.config import Config

    doc = config_doc.document()
    fields = [f for s in doc["sections"] for f in s["fields"]]
    assert doc["n_fields"] == len(Config.default().as_dict()) == len(fields)
    assert doc["config_hash"] == Config.default().config_hash()
    assert all(f["comment"] for f in fields)
    assert all(f["sources"] for f in fields)
    assert [f["name"] for f in fields] == list(Config.default().as_dict())


# -- ledger -------------------------------------------------------------------------------------

EXPERIMENTS = """# Experiments ledger

| date | name | git | config hash | public spellF1 | public macroF1 | mean_latency@25 | n_pred spells (gt 17) | full possessions | full D share | mean dur (s) | verdict | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-30 | baseline_x | upstream | — | 0.3256 | 0.6207 | +4.0 | 26 | — | — | 10.8 | info | floor |
| 2026-11-01 | c1_new | abc1234 | 0123456789 | 0.750 | 0.948 | −2.6 | 15 | 250 | 49.8% | — | keep (C1) | md5 39562b94a10e23e24c838a1c6c8198ef; impulse \\|J\\| at the bump |
| 2026-11-01 | c2_bad (variant) | abc1234 | 9876543210 | 0.565 | 0.850 | +0.3 | 17 | 228 | 51.3% | — | reject | md5 5caf707c… lost 18389 |
| 2026-11-01 | p5 bootstrap (1000) | abc1234 | — | 0.750 | — | — | — | — | — | — | info | c1_new 0.750 [0.5625, 0.9008] / 0.9483 [0.8747, 0.9714] |
"""

PROGRESS = """# Progress

## Current best

| version | public spellF1 | public macroF1 | mean_latency@25 | full-match possessions | D share | D segments | date |
|---|---|---|---|---|---|---|---|
| baseline_x (organisers) | 0.3256 | 0.6207 | +4.0 | 594 | 32.3% | 133 | 30 Oct |
| **c1_new** (default mode) — current best | 0.750 | 0.948 | −2.6 | 250 | 49.8% | 99 | 1 Nov |

Text after the table.
"""

LEARNINGS = """# LEARNINGS

## C<n> — <short name> (<date>, <kept | rejected | info>)

**Hypothesis.** template.

## C1 — the bump detector (1 Nov, kept)

**Hypothesis.** The ball is smooth (PLAN 1.1). A second
sentence here.

**Result.** fine.
"""


def test_ledger_parses_signs_pipes_verdicts_md5_and_order():
    rows = ledger.parse_experiments(EXPERIMENTS)
    assert [r["name"] for r in rows][:3] == ["baseline_x", "c1_new", "c2_bad (variant)"]
    new = rows[1]
    assert ledger.number(new["mean_latency"]) == -2.6
    assert ledger.number(new["d_share"]) == pytest.approx(0.498)
    assert (new["status"], new["cycle"]) == ("keep", "C1")
    assert new["md5"] == "39562b94a10e23e24c838a1c6c8198ef"
    assert "|J|" in new["note"]
    assert rows[2]["md5"] == "5caf707c"

    best = ledger.parse_current_best(PROGRESS)
    assert [b["version"] for b in best] == ["baseline_x", "c1_new"]
    result = ledger.ladder(rows, best, {"baseline_x": "a" * 32})
    first, second = result["rows"]
    assert (first["status"], first["cycle"], first["md5"]) == ("baseline", "baseline", "a" * 32)
    assert first["possessions"] == 594 and first["config_hash"] is None
    assert (second["status"], second["spell_f1"], second["d_segments"]) == ("kept", 0.75, 99)
    assert second["bootstrap"] == {"lo": 0.5625, "hi": 0.9008}
    assert [r["version"] for r in result["rejected"]] == ["c2_bad"]
    assert result["rejected"][0]["cycle"] == "C2"


def test_ledger_cycles_skip_the_template_and_keep_the_first_sentence():
    items = ledger.cycles(LEARNINGS)
    assert items == [{"id": "C1", "title": "the bump detector", "date": "1 Nov",
                      "status": "kept", "hypothesis": "The ball is smooth (PLAN 1.1)."}]


def test_ledger_ladder_omits_external_only_rejection_without_erasing_history():
    external = ("| 2026-11-02 | p10c_verdict | abc1234 | B0 | — | — | — | — | — | — | — | "
                "reject (P10-C) | external spellF1 0.87; public score not run |\n")
    zero = ("| 2026-11-02 | c3_zero | abc1234 | B0 | 0.0 | — | — | — | — | — | — | "
            "reject (C3) | measured public score |\n")
    rows = ledger.parse_experiments(EXPERIMENTS + external + zero)
    before = copy.deepcopy(rows)
    result = ledger.ladder(rows, ledger.parse_current_best(PROGRESS))
    assert [r["version"] for r in result["rejected"]] == ["c2_bad", "c3_zero"]
    assert result["rejected"][-1]["spell_f1"] == 0.0
    assert rows == before
    assert (rows[-2]["status"], rows[-2]["cycle"]) == ("reject", "P10-C")
    learning = (LEARNINGS + "\n## P10-C — gap ownership (2 November 2026, rejected)\n\n"
                "**Hypothesis.** Player shape can recover gap ownership.\n")
    cycle = ledger.cycles(learning)[-1]
    assert (cycle["id"], cycle["status"]) == ("P10-C", "rejected")


def test_ledger_reads_the_real_files():
    if not os.path.isfile(ledger.PROGRESS_PATH):
        pytest.skip("the public copy leaves out the working notes (tools/export_public.py)")
    rows = ledger.parse_experiments(ledger.read(ledger.EXPERIMENTS_PATH))
    best = ledger.parse_current_best(ledger.read(ledger.PROGRESS_PATH))
    result = ledger.ladder(rows, best)
    assert result["rows"][0]["version"] == "baseline_smooth50"
    assert all(r["status"] == "kept" for r in result["rows"][1:])
    assert "c4_restart_anchor" in [r["version"] for r in result["rejected"]]
    assert all(c["hypothesis"] for c in ledger.cycles(ledger.read(ledger.LEARNINGS_PATH)))


# -- CLIs and the output folder ------------------------------------------------------------------

@pytest.mark.parametrize("main", [build_site.main, build_replay.main, guard.main])
def test_cli_usage_exits_2_and_help_exits_0(main, capsys):
    with pytest.raises(SystemExit) as usage:
        main([])
    assert usage.value.code == 2
    with pytest.raises(SystemExit) as helped:
        main(["--help"])
    assert helped.value.code == 0
    capsys.readouterr()


def test_guard_cli_exits_2_on_a_missing_folder(tmp_path, capsys):
    assert guard.main([str(tmp_path / "absent")]) == 2
    capsys.readouterr()


def test_check_out_dir_refuses_a_committed_path_and_accepts_ignored_ones(tmp_path):
    """git decides what it would commit, so the test needs a git checkout."""
    if not os.path.exists(os.path.join(ROOT, ".git")):
        pytest.skip("this folder is not a git checkout, so git cannot say what it would commit")
    with pytest.raises(build_replay.RefusedPath):
        build_replay.check_out_dir(os.path.join(ROOT, "tools", "viz", "replay_out"))
    with pytest.raises(build_replay.RefusedPath):
        build_replay.check_out_dir(os.path.join(ROOT, "site", "data", "public"))
    local = os.path.join(ROOT, "site", "data", "local")
    assert build_replay.check_out_dir(local) == os.path.realpath(local)
    assert build_replay.check_out_dir(str(tmp_path)) == os.path.realpath(str(tmp_path))


def test_replay_flags_set_their_bits():
    n = 20
    inputs = {"ball": {"visible": np.arange(n) % 2 == 0, "airborne_suspect": np.zeros(n, bool),
                       "teleport": np.zeros(n, bool)},
              "ghost": np.zeros(n, bool)}
    parts = {"stoppage": np.arange(n) >= 15}
    out = build_replay.flags(inputs, parts, [(1, 5, 9)], n)
    graded = 1 << build_replay.FLAG_BITS.index("graded")
    assert out[5] & graded and out[9] & graded and not out[10] & graded
    assert out[0] & 2 and not out[1] & 2
    assert out[16] & 1 and not out[14] & 1


# -- data-gated smoke tests ------------------------------------------------------------------------

@pytest.mark.slow
@needs_data
@needs_reports
def test_build_site_smoke_on_baseline_smooth50(tmp_path, capsys):
    inputs = build_site.load_inputs("baseline_smooth50", SMOOTH50, allow_mismatch=True,
                                    warn=lambda text: None)
    bundle = build_site.assemble(inputs)
    out = tmp_path / "public"
    build_site.write_bundle(str(out), bundle)
    assert guard.check_dir(str(out)) == []
    with open(str(out / "manifest.json")) as handle:
        manifest = json.load(handle)
    assert manifest["md5"] == SMOOTH50_MD5
    assert manifest["pitch_m"] == [105.3, 68.0]
    lo, hi = manifest["public_window"]["start"], manifest["public_window"]["stop"]
    with open(str(out / "labels_window.json")) as handle:
        window = json.load(handle)
    decoded = rle.decode(window["ours"], manifest["n_frames"])
    assert decoded[lo:hi].tolist() == inputs["ours"][lo:hi].tolist()
    with open(str(out / "scores.json")) as handle:
        scores = json.load(handle)
    assert scores["baseline_smooth50"]["spell_f1_mean"] == pytest.approx(SMOOTH50_SPELL_F1,
                                                                         abs=1e-4)
    with open(str(out / "minute_bins.json")) as handle:
        bins = json.load(handle)
    assert bins["n_bins"] == 106
    assert sum(r["n_possession_starts"] for r in bins["baseline_smooth50"]) == 594
    capsys.readouterr()


@pytest.mark.slow
@needs_data
@needs_reports
def test_build_replay_smoke_to_a_tmp_path(tmp_path):
    out = build_replay.check_out_dir(str(tmp_path / "local"))
    build_replay.write_replay("baseline_smooth50", SMOOTH50, out, allow_mismatch=True)
    with open(os.path.join(out, "manifest.json")) as handle:
        manifest = json.load(handle)
    assert [h["frame_start"] for h in manifest["halves"]] == [0, 110808]
    for half in manifest["halves"]:
        n = half["frame_stop"] - half["frame_start"]
        with open(os.path.join(out, half["files"]["positions"])) as handle:
            header = json.load(handle)
        with open(os.path.join(out, half["files"]["positions_bin"]), "rb") as handle:
            blob = handle.read()
        entries = {name: dict(e, dtype="int16") for name, e in header["arrays"].items()}
        views = quant.unpack(blob, entries)
        assert views["px"].shape == (n, 46) and views["py"].shape == (n, 46)
        assert views["bx"].shape == (n, 1)
        decoded = quant.decode(views["px"])
        assert np.isfinite(decoded).any()
        assert np.nanmax(np.abs(decoded)) < 200
        with open(os.path.join(out, half["files"]["frames"])) as handle:
            frames_header = json.load(handle)
        with open(os.path.join(out, half["files"]["frames_bin"]), "rb") as handle:
            frames = quant.unpack(handle.read(), frames_header["arrays"])
        assert all(e["offset"] % 2 == 0 for e in frames_header["arrays"].values())
        assert frames["log_e"].shape == (n, 3)
        graded = 1 << build_replay.FLAG_BITS.index("graded")
        if half["half"] == 1:
            assert frames["flags"][11185] & graded
            assert not frames["flags"][0] & graded
