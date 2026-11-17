"""R3b unseen-provider diagnostic (r3_diag.py): pools and the honest cost choice."""

import os

import numpy as np
import pytest

from tools.learned_spike import folds, r3_diag


@pytest.mark.parametrize("name", ["no-metrica", "no-dfl"])
def test_pool_has_no_held_out_provider(name):
    spec = r3_diag.pool(name)
    held = spec["held"]
    assert not any(folds.provider(g) == held for g in spec["train"] + spec["val"])
    assert spec["scored"] == [g for g in folds.OPEN if folds.provider(g) == held]
    assert spec["tune"] and all(folds.provider(g) not in (held, "skillcorner")
                                for g in spec["tune"])
    assert not set(spec["train"]) & set(spec["val"])
    # validation pair of the recipe: one development match of the other provider + one SC
    assert len(spec["val"]) == 2
    assert folds.provider(spec["val"][0]) == spec["tune"][0].split("-")[0].lower()
    assert folds.provider(spec["val"][1]) == "skillcorner"
    assert spec["val"] == folds.fold(spec["ref_fold"], pff=False)["val"]
    assert folds.provider(folds.OPEN[spec["ref_fold"]]) == held
    # every other match of the 28 trains
    assert sorted(spec["train"] + spec["val"]) == sorted(
        g for g in folds.ALL if folds.provider(g) != held)


@pytest.mark.parametrize("name", r3_diag.NAMES)
def test_pool_never_sealed_competition_or_pff(name):
    spec = r3_diag.pool(name)
    every = spec["train"] + spec["val"] + spec["scored"] + spec["tune"]
    for g in every:
        assert g not in folds.SEALED and g != folds.COMPETITION
        assert not g.startswith("PFF-")
        assert g in folds.ALL


def test_pools_exact():
    nm, nd = r3_diag.pool("no-metrica"), r3_diag.pool("no-dfl")
    assert nm["val"] == ["DFL-J03WMX", folds.SC[1]]
    assert nd["val"] == ["METRICA-S1", folds.SC[4]]
    assert len(nm["train"]) == 4 + 19 and len(nd["train"]) == 2 + 19


def test_read_ref_refuses_sealed(tmp_path):
    for g in folds.SEALED + (folds.COMPETITION,):
        with pytest.raises(SystemExit):
            r3_diag.read_ref(str(tmp_path), g)


def _fake(tmp_path, games, n=500):
    """Cache labels and p5 files for `games` only; references alternate H/A every 100 frames."""
    cache, pred = tmp_path / "cache", tmp_path / "pred"
    (cache / "labels").mkdir(parents=True)
    pred.mkdir()
    ref = np.repeat(np.arange(n // 100) % 2, 100).astype(np.int8)
    p5 = np.zeros((n // 5, 3))
    p5[np.arange(n // 5), ref[2::5]] = 0.9
    p5[:, 2] += 0.05
    p5[p5 == 0] = 0.05
    for g in games:
        np.savez(str(cache / "labels" / (g + ".npz")), raw=ref)
        np.savez(str(pred / (g + "_p5.npz")), p5=p5)
    return str(cache), str(pred), lambda g: [(1, 0, n - 1)]


@pytest.mark.parametrize("name", r3_diag.NAMES)
def test_honest_choice_never_reads_held_out_reference(tmp_path, monkeypatch, name):
    pytest.importorskip("score_game")
    spec = r3_diag.pool(name)
    cache, pred, blocks = _fake(tmp_path, spec["tune"])      # held-out refs absent on disk
    seen = []
    real = r3_diag.read_ref

    def spy(c, g):
        seen.append(g)
        assert folds.provider(g) != spec["held"], "read a held-out-provider reference"
        return real(c, g)

    monkeypatch.setattr(r3_diag, "read_ref", spy)
    i, v, scores = r3_diag.honest_choice(spec, pred, cache, blocks)
    assert sorted(seen) == sorted(spec["tune"])
    assert set(scores) == set(spec["tune"])
    assert 0 <= i < len(r3_diag.GRID) and 0.0 <= v <= 1.0


def test_honest_choice_refuses_held_out_tuning():
    spec = dict(r3_diag.pool("no-metrica"))
    spec["tune"] = spec["tune"] + ["METRICA-S1"]
    with pytest.raises(AssertionError):
        r3_diag.honest_choice(spec, "/nonexistent", "/nonexistent")


def test_pick_ties_lowest_index():
    n = len(r3_diag.GRID)
    scores = {"A": [0.5] * n, "B": [0.5] * n}
    assert r3_diag.pick(scores, ["A", "B"])[0] == 0
    scores["A"][9] = 0.9
    scores["B"][9] = 0.1          # mean 0.5: still a tie, index 0 wins
    assert r3_diag.pick(scores, ["A", "B"])[0] == 0
    scores["B"][9] = 0.2
    assert r3_diag.pick(scores, ["A", "B"])[0] == 9


def test_launcher_present():
    path = os.path.join(r3_diag.ROOT, "tools", "learned_spike", "r3_diag_launch.sh")
    text = open(path).read()
    assert "wait_guard" in text and "battery guard not running" in text and "done" in text


# -- the no-skillcorner pool -------------------------------------------------------------------

def test_sc_pool_exact():
    spec = r3_diag.pool(r3_diag.SC_DIAG)
    assert spec["val"] == ["DFL-J03WMX", "METRICA-S1"]
    # the development members of the two other pools' validation pairs
    assert spec["val"] == [r3_diag.pool("no-metrica")["val"][0], r3_diag.pool("no-dfl")["val"][0]]
    assert spec["train"] == ["METRICA-S2", "METRICA-S3", "DFL-J03WN1", "DFL-J03WOH",
                             "DFL-J03WOY", "DFL-J03WPY"]
    assert spec["scored"] == list(folds.SC)
    assert spec["tune"] == list(folds.OPEN)
    assert not any(folds.provider(g) == "skillcorner" for g in spec["train"] + spec["val"])


def test_fold_of_sc_matches_the_recipe():
    for g in folds.SC:
        k = r3_diag.fold_of_sc(g)
        assert g in folds.fold(k, pff=False)["test"]
        assert g not in folds.fold(k, pff=False)["train"]


def test_read_sc_ref(tmp_path):
    cache, fine = tmp_path / "c", tmp_path / "f"
    (cache / "labels").mkdir(parents=True)
    (fine / "labels").mkdir(parents=True)
    g = folds.SC[0]
    np.savez(str(cache / "labels" / (g + ".npz")), raw=np.array([0, 0, 2], np.int8))
    np.savez(str(fine / "labels" / (g + ".npz")), raw=np.array([0, 1, 2], np.int8))
    assert r3_diag.read_sc_ref(g, "phases", str(cache), str(fine)).tolist() == [0, 0, 2]
    assert r3_diag.read_sc_ref(g, "possgroup", str(cache), str(fine)).tolist() == [0, 1, 2]
    with pytest.raises(SystemExit):
        r3_diag.read_sc_ref("METRICA-S1", "phases", str(cache), str(fine))
    with pytest.raises(SystemExit):
        r3_diag.read_sc_ref(folds.SEALED[0], "phases", str(cache), str(fine))


def test_dev_choice_reads_development_references_only(tmp_path, monkeypatch):
    pytest.importorskip("score_game")
    cache, pred, blocks = _fake(tmp_path, folds.OPEN)
    seen = []
    real = r3_diag.read_ref

    def spy(c, g):
        seen.append(g)
        assert folds.provider(g) != "skillcorner"
        return real(c, g)

    monkeypatch.setattr(r3_diag, "read_ref", spy)
    i, v, scores = r3_diag.dev_choice(pred, cache, blocks)
    assert sorted(seen) == sorted(folds.OPEN)
    assert 0 <= i < len(r3_diag.GRID)


def test_sc_launcher_present():
    path = os.path.join(r3_diag.ROOT, "tools", "learned_spike", "r3_diag_sc_launch.sh")
    text = open(path).read()
    assert "wait_guard" in text and "no-skillcorner" in text and "predict-folds" in text
