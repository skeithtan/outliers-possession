"""tools/ablate.py: spec checks, patches, comparisons and the report (card P5, step 2).

Synthetic tests only. They need no match data and run no engine.
"""

import csv
import os

import numpy as np
import pandas as pd
import pytest

from tools import ablate


def _spec(name="a", **extra):
    spec = {"name": name, "config": {}}
    spec.update(extra)
    return spec


# -- specs ----------------------------------------------------------------------------------------

def test_check_specs_fills_defaults():
    out = ablate.check_specs([_spec("x", config={"MIN_POSS": 20})])
    assert out == [{"name": "x", "config": {"MIN_POSS": 20}, "patches": [], "note": ""}]


@pytest.mark.parametrize("specs", [
    {"name": "a"},                                 # not a list
    [_spec("Bad-Name")],                           # bad characters
    [_spec("reference")],                          # reserved name
    [_spec("a"), _spec("a")],                      # duplicate
    [_spec("a", patches=["drop_onsets:goal"])],    # unknown patch
    [_spec("a", colour="red")],                    # unknown key
    [_spec("a", config=[1, 2])],                   # config is not an object
])
def test_check_specs_rejects(specs):
    with pytest.raises(ValueError):
        ablate.check_specs(specs)


def test_check_config_fields_rejects_unknown_field():
    with pytest.raises(ValueError):
        ablate.check_config_fields(ablate.check_specs([_spec(config={"NO_SUCH_FIELD": 1})]))


def test_p5_list_is_valid():
    specs = ablate.check_specs(ablate.P5_ABLATIONS)
    ablate.check_config_fields(specs)
    assert len(set(s["name"] for s in specs)) == len(specs)


def test_onset_mechanisms_match_engine():
    from engine import deadball

    assert ablate.ONSET_MECHANISMS == deadball.MECHANISMS


# -- patches --------------------------------------------------------------------------------------

def test_drop_mechanism_removes_only_that_mechanism():
    table = pd.DataFrame({"frame": [10, 20, 30, 40],
                          "mechanism": ["out", "collective_stop", "pursuit_stop", "out"],
                          "confidence": [1.0, 0.3, 0.6, 0.8]})
    wrapped = ablate.drop_mechanism(lambda: table.copy(), "out")
    out = wrapped()
    assert out["frame"].tolist() == [20, 30]
    assert out.index.tolist() == [0, 1]


def test_apply_patches_on_a_stub_module():
    class Stub(object):
        @staticmethod
        def dead_onsets():
            return pd.DataFrame({"frame": [1, 2], "mechanism": ["out", "pursuit_stop"],
                                 "confidence": [1.0, 0.6]})

        @staticmethod
        def _is_throw_in_reappearance(*args):
            return True

    stub = Stub()
    ablate.apply_patches(["drop_onsets:pursuit_stop", "no_throw_in_rule"], stub)
    assert stub.dead_onsets()["mechanism"].tolist() == ["out"]
    assert stub._is_throw_in_reappearance(None, 0, None, None) is False
    with pytest.raises(ValueError):
        ablate.apply_patches(["unknown"], stub)


# -- comparisons ----------------------------------------------------------------------------------

def test_changed_stretches():
    ref = np.array([0, 0, 1, 1, 2, 2, 0, 0], dtype=np.int8)
    new = np.array([0, 1, 1, 1, 2, 0, 1, 0], dtype=np.int8)
    assert ablate.changed_stretches(new, ref) == (3, 2)
    assert ablate.changed_stretches(ref, ref) == (0, 0)


def _write_transitions(path, rows):
    header = ["status", "gt_frame", "pred_frame", "latency", "gt_from", "gt_to", "pred_from",
              "pred_to"]
    with open(path, "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_transition_changes(tmp_path):
    ref_path, new_path = os.path.join(str(tmp_path), "r.csv"), os.path.join(str(tmp_path), "n.csv")
    _write_transitions(ref_path, [
        {"status": "matched", "gt_frame": 100, "pred_frame": 95, "latency": -5},
        {"status": "matched", "gt_frame": 200, "pred_frame": 210, "latency": 10},
        {"status": "missed", "gt_frame": 300},
        {"status": "spurious", "pred_frame": 400, "pred_from": "H", "pred_to": "A"},
    ])
    _write_transitions(new_path, [
        {"status": "matched", "gt_frame": 100, "pred_frame": 97, "latency": -3},
        {"status": "missed", "gt_frame": 200},
        {"status": "matched", "gt_frame": 300, "pred_frame": 300, "latency": 0},
        {"status": "spurious", "pred_frame": 500, "pred_from": "A", "pred_to": "D"},
    ])
    out = ablate.transition_changes(ablate.read_transitions(new_path),
                                    ablate.read_transitions(ref_path))
    assert out == {"lost": ["200 (+10)"], "gained": ["300 (+0)"], "latency": ["100 -5>-3"],
                   "spurious_added": ["500 A>D"], "spurious_removed": ["400 H>A"]}


def _row(name, spell, poss, d_share, segs):
    return {"name": name, "md5": "0" * 32, "spell_f1": spell, "macro_f1": 0.9, "matched": 1,
            "missed": 2, "spurious": 3, "possessions": poss, "d_share": d_share,
            "d_segments": segs, "short_spells": 4, "tag_agreement": 0.8, "vetoes": "pass",
            "frames_changed": 0, "stretches_changed": 0, "public": {}}


def test_veto_margins():
    margins = ablate.veto_margins(_row("a", 0.5, [100, 100, 200], [0.4, 0.5, 0.45], 60))
    assert margins["possessions"] == 20
    assert margins["d_share"] == pytest.approx(0.10)
    assert margins["d_share_h2_line"] == pytest.approx(0.05)
    assert margins["d_segments"] == 5


def test_p5_flag():
    ref = _row("reference", 0.7, [120, 120, 240], [0.46, 0.54, 0.50], 99)
    # Component off: lower public score, D share farther from 55 %: the component is flagged.
    off = _row("x", 0.6, [120, 120, 240], [0.45, 0.52, 0.48], 99)
    assert ablate.p5_flag(off, ref) == ["d_share", "d_share_h1_line", "d_share_h2_line"]
    # Component off with a higher public score: never flagged.
    better = _row("y", 0.8, [120, 120, 240], [0.45, 0.52, 0.48], 99)
    assert ablate.p5_flag(better, ref) == []
    # Lower public score, but no statistic nearer a veto with the component on.
    same = _row("z", 0.6, [120, 120, 240], [0.46, 0.54, 0.50], 99)
    assert ablate.p5_flag(same, ref) == []


def test_render_is_deterministic():
    ref = _row("reference", 0.7, [120, 120, 240], [0.46, 0.54, 0.50], 99)
    off = _row("x", 0.6, [120, 121, 241], [0.45, 0.52, 0.48], 98)
    off["public"] = {"lost": ["100 (+1)"], "gained": [], "latency": [], "spurious_added": [],
                     "spurious_removed": []}
    lines = ablate.render([ref, off])
    assert lines == ablate.render([ref, off])
    assert lines[0] == ablate.TABLE_HEADER
    assert lines[3].startswith("| x | 00000000 | 0.600 |")
    assert any(line.startswith("- **x**: public lost 100 (+1)") for line in lines)
    assert any("P5 check" in line for line in lines)


def test_render_names_the_reference_scope():
    ref = _row("reference", 0.7, [120, 120, 240], [0.46, 0.54, 0.50], 99)
    off = _row("x", 0.6, [120, 121, 241], [0.45, 0.52, 0.48], 98)
    off["public"] = {"lost": ["100 (+1)"], "gained": [], "latency": [], "spurious_added": [],
                     "spurious_removed": []}
    public = ablate.render([ref, off])
    reference = ablate.render([ref, off], "reference")
    assert public == ablate.render([ref, off], "public")
    assert any(line.startswith("Public transitions") for line in public)
    assert any(line.startswith("Reference transitions") for line in reference)
    assert any(line.startswith("- **x**: reference lost 100 (+1)") for line in reference)
    assert any("raises the reference spellF1" in line for line in reference)


class _Done(object):
    """Stand-in for a finished subprocess."""
    returncode = 0
    stdout = ""


def _recorder(calls):
    def run(command, **kwargs):
        calls.append(command)
        return _Done()
    return run


@pytest.mark.parametrize("game, extra", [(None, []), ("METRICA-S3", ["--game", "METRICA-S3"])])
def test_run_one_passes_the_game(tmp_path, monkeypatch, game, extra):
    calls = []
    monkeypatch.setattr(ablate.subprocess, "run", _recorder(calls))
    out = str(tmp_path)
    ablate.run_one({"name": "reference", "config": {}, "patches": [], "note": ""}, out, "py",
                   game)
    ablate.run_one(ablate.check_specs([_spec("a", config={"MIN_POSS": 20})])[0], out, "py", game)
    ablate.run_one(ablate.check_specs([_spec("b", patches=["no_throw_in_rule"])])[0], out, "py",
                   game)
    results = os.path.join(out, "results")
    configs = os.path.join(out, "configs")
    assert calls[0] == ["py", "-m", "engine.run", "--out",
                        os.path.join(results, "reference.csv")] + extra
    assert calls[1] == ["py", "-m", "engine.run", "--out", os.path.join(results, "a.csv"),
                        "--config", os.path.join(configs, "a.json")] + extra
    assert calls[2][2:] == ["--child", os.path.join(configs, "b.spec.json"),
                            os.path.join(results, "b.csv"),
                            os.path.join(configs, "b.json")] + extra


@pytest.mark.parametrize("gt", [None, "ref.csv"])
def test_score_passes_the_reference(tmp_path, monkeypatch, gt):
    calls = []
    monkeypatch.setattr(ablate.subprocess, "run", _recorder(calls))
    os.makedirs(str(tmp_path / "logs"))
    ablate.score(str(tmp_path), "py", gt)
    tail = [] if gt is None else ["--gt", os.path.abspath(gt)]
    assert calls[0] == ["py", ablate.SCORER, "--results", str(tmp_path / "results"),
                        "--out", str(tmp_path / "score"), "--no-detail"] + tail
