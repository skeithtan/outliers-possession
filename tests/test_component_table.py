"""tools/analysis/component_table.py: the per-component table of the P9-A3 ablations.

The tests write tiny references and results, score them with tools/score_game.py into a
temporary folder laid out as <paired>/<game>/<name>.json, and build the table from them. The
tests need no match data.
"""

import importlib.util
import json
import os

import numpy as np
import pytest

from tools import paired_summary as ps
from tools import score_game as sg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "component_table", os.path.join(ROOT, "tools", "analysis", "component_table.py"))
ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ct)

H, A, D, X = 0, 1, 2, 3

REF_1 = np.concatenate([np.full(n, k, dtype=np.int8) for k, n in
                        ((X, 5), (H, 60), (A, 30), (H, 20), (D, 40), (A, 80), (H, 50), (X, 5))])
REF_2 = np.concatenate([np.full(n, k, dtype=np.int8) for k, n in
                        ((H, 90), (A, 15), (H, 70), (D, 30), (H, 40), (A, 60))])


def _write(path, codes):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        for i, c in enumerate(codes):
            handle.write("{},{}\n".format(i, "HADX"[c]))
    return path


def _late(ref, k):
    out = ref.copy()
    out[k:] = ref[:-k]
    out[:k] = ref[0]
    return out


def _game(tmp_path, game, ref, reference, ablations):
    """Score `reference` alone and each ablation versus it, as the P9-A3 loop does."""
    src = tmp_path / "src" / game
    src.mkdir(parents=True)
    out = str(tmp_path / "paired" / game)
    gt = _write(str(src / (game + "_ref.csv")), ref)
    ref_csv = _write(str(src / "reference.csv"), reference)
    assert sg.main([ref_csv, "--gt", gt, "--out-dir", out, "--name", "reference"]) == 0
    for name, codes in ablations.items():
        csv = _write(str(src / (name + ".csv")), codes)
        assert sg.main([csv, "--gt", gt, "--out-dir", out, "--name", name,
                        "--versus", ref_csv, "--draws", "200"]) == 0
    return out


def _setup(tmp_path):
    _game(tmp_path, "G-ONE", REF_1, _late(REF_1, 12),
          {"redirect_off": REF_1, "c5_dribble_off": _late(REF_1, 20)})
    _game(tmp_path, "G-TWO", REF_2, _late(REF_2, 3),
          {"redirect_off": _late(REF_2, 2), "c5_dribble_off": _late(REF_2, 25)})
    _game(tmp_path, "PUB", REF_1, REF_1,
          {"redirect_off": _late(REF_1, 15), "c5_dribble_off": REF_1})
    abl = tmp_path / "ablate" / "G-ONE"
    abl.mkdir(parents=True)
    with open(str(abl / "ablations.json"), "w") as handle:
        json.dump([{"name": "redirect_off", "frames_changed": 40, "stretches_changed": 3}],
                  handle)
    return str(tmp_path / "paired")


def test_rows_follow_the_ablate_order_and_read_the_paired_delta(tmp_path):
    paired = _setup(tmp_path)
    table = ct.build(paired, ["G-ONE", "G-TWO"], public="PUB",
                     ablate_dir=str(tmp_path / "ablate"), n_draws=200)
    names = [row["name"] for row in table["ablations"]]
    assert names == ["c5_dribble_off", "redirect_off"]          # P5_ABLATIONS order
    red = table["ablations"][1]
    one = red["per_match"]["G-ONE"]
    pair = sg.paired_delta(REF_1, REF_1, _late(REF_1, 12), n_draws=200)
    assert one["delta"] == pytest.approx(pair["delta"])
    assert (one["low"], one["high"]) == pytest.approx((pair["low"], pair["high"]))
    assert one["delta"] == pytest.approx(one["spell_f1"] - one["ref_spell_f1"])
    assert one["frames_changed"] == 40 and one["stretches_changed"] == 3
    assert red["per_match"]["G-TWO"]["frames_changed"] is None
    # the public game takes no part in the external summary
    assert red["n_positive"] == sum(1 for g in ("G-ONE", "G-TWO")
                                    if red["per_match"][g]["delta"] > 0)
    assert red["per_match"]["PUB"]["delta"] < 0
    assert red["n_better"] + red["n_worse"] <= 2


def test_cross_summary_equals_paired_summary(tmp_path):
    paired = _setup(tmp_path)
    table = ct.build(paired, ["G-ONE", "G-TWO"], public="PUB", n_draws=200)
    row = table["ablations"][1]
    side_a = {}
    side_b = {}
    for game in ("G-ONE", "G-TWO"):
        a = ps.load(os.path.join(paired, game, "redirect_off.json"))
        b = ps.load(os.path.join(paired, game, "reference.json"))
        side_a[game + "/" + a[0]] = a
        side_b[game + "/" + b[0]] = b
    expected = ps.paired_summary(side_a, side_b, 200, 0)
    assert row["external"]["mean_delta"] == pytest.approx(expected["mean_delta"])
    assert row["external"]["low"] == pytest.approx(expected["low"])
    assert row["with_public"]["mean_delta"] == pytest.approx(np.mean(
        [row["per_match"][g]["delta"] for g in ("G-ONE", "G-TWO", "PUB")]))
    assert ct.build(paired, ["G-ONE", "G-TWO"], public="PUB", n_draws=200) == table


def test_render_and_main(tmp_path, capsys):
    paired = _setup(tmp_path)
    out_json = str(tmp_path / "t.json")
    assert ct.main(["--paired", paired, "--games", "G-ONE", "G-TWO", "--public", "PUB",
                    "--draws", "50", "--out-json", out_json]) == 0
    text = capsys.readouterr().out
    assert "### Cross-match summary" in text and "| redirect_off |" in text
    assert "#### PUB" in text
    assert json.load(open(out_json))["public"] == "PUB"


def test_a_report_without_versus_or_with_another_reference_is_refused(tmp_path):
    paired = _setup(tmp_path)
    ref = json.load(open(os.path.join(paired, "G-ONE", "reference.json")))
    with pytest.raises(ValueError):
        ct.match_row(ref, ref)                                  # no versus field
    other = json.load(open(os.path.join(paired, "G-TWO", "redirect_off.json")))
    with pytest.raises(ValueError):
        ct.match_row(other, ref)                                # another reference file
    os.remove(os.path.join(paired, "G-TWO", "reference.json"))
    with pytest.raises(ValueError):
        ct.build(paired, ["G-ONE", "G-TWO"], n_draws=10)


def test_short_names():
    assert [ct.short(g) for g in ("DFL-J03WMX", "METRICA-S3", "BAR-ATH", "X")] == \
        ["WMX", "S3", "public", "X"]
