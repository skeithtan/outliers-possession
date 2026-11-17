"""Round 4 switches and tools (LEARNED_STAGE_PROTOCOL_R4.md Sections 1 and 2)."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import still_ball  # noqa: E402
from tools.learned_spike import folds, r4_inputs  # noqa: E402


# -- step C: the still-ball rule ------------------------------------------------------------

def _ball(n=100):
    bx = np.linspace(0.1, 0.9, n).astype(np.float32)
    by = np.linspace(0.2, 0.8, n).astype(np.float32)
    return bx, by


def test_still_mask_threshold_and_first_frame():
    bx, by = _ball()
    bx[10:23] = bx[10]          # 13 equal positions: frames 10..22
    by[10:23] = by[10]
    bx[50:62] = bx[50]          # 12 equal positions: kept
    by[50:62] = by[50]
    m = still_ball.still_mask(bx, by)
    assert m[10:23].all() and m.sum() == 13
    assert not m[50:62].any()


def test_still_mask_needs_both_axes_and_breaks_on_missing():
    bx, by = _ball()
    bx[0:30] = 0.5              # x still, y moving: not still
    assert not still_ball.still_mask(bx, by).any()
    bx, by = _ball()
    bx[20:40] = 0.5
    by[20:40] = 0.5
    bx[30] = np.nan             # a missing frame splits the run into 10 + 9
    assert not still_ball.still_mask(bx, by).any()
    bx[30] = 0.5
    assert still_ball.still_mask(bx, by).sum() == 20
    assert not still_ball.still_mask(np.full(5, np.nan), np.full(5, np.nan)).any()


def test_still_switch_scope_and_sealed_refused():
    tac = {"bx": np.full(30, 0.5, np.float32), "by": np.full(30, 0.5, np.float32)}
    same, m = still_ball.remove_still(tac, "DFL-J03WMX", "kept")
    assert same is tac and not m.any()
    out, m = still_ball.remove_still(tac, "DFL-J03WMX", "removed")
    assert m.all() and np.isnan(out["bx"]).all() and np.isfinite(tac["bx"]).all()
    same, m = still_ball.remove_still(tac, "METRICA-S1", "removed")
    assert same is tac and not m.any()
    for g in ("DFL-J03WQQ", "DFL-J03WR9"):
        with pytest.raises(SystemExit):
            still_ball.remove_still(tac, g, "removed")
        with pytest.raises(SystemExit):
            still_ball.build_bundle(g, "/nonexistent", "/nonexistent2")
    with pytest.raises(ValueError):
        still_ball.applies("DFL-J03WMX", "maybe")


def test_rewrite_lines_only_ball_xy_of_masked_frames():
    meta = {"ball_channels": ("x", "y"), "player_channels": ("x", "y")}
    import engine.data as data
    meta = {"ball_channels": (data.X_CHANNEL, data.Y_CHANNEL),
            "player_channels": (data.X_CHANNEL, data.Y_CHANNEL)}
    lines = ["1:0.1,0.2;0.3,0.4:0.5,0.6", "2:0.1,0.2;0.3,0.4:0.5,0.6"]
    out = still_ball.rewrite_lines(lines, meta, np.array([False, True]))
    assert out[0] == lines[0] and out[1] == "2:0.1,0.2;0.3,0.4:NaN,NaN"


# -- step C: the training input dfl_c_train ---------------------------------------------------

def _tac(n=20000, seed=0):
    rng = np.random.RandomState(seed)
    bx = (0.5 + np.cumsum(rng.normal(0, 1e-3, n))).astype(np.float32)
    by = (0.5 + np.cumsum(rng.normal(0, 1e-3, n))).astype(np.float32)
    for a in (1000, 5000, 9000):            # three still stretches of 200 frames
        if a + 200 > n:
            continue
        bx[a:a + 200] = bx[a]
        by[a:a + 200] = by[a]
    return {"bx": bx, "by": by, "px": np.zeros((n, 2), np.float32)}


def test_train_gaps_share_extension_and_determinism():
    tac = _tac()
    c, rep_c = r4_inputs.load_dfl(tac, "DFL-J03WMX", "c")
    t1, rep = r4_inputs.load_dfl(tac, "DFL-J03WMX", "c_train")
    t2, _ = r4_inputs.load_dfl(tac, "DFL-J03WMX", "c_train")
    assert np.array_equal(t1["bx"], t2["bx"], equal_nan=True)          # fixed seed per match
    assert rep_c["still_frames"] == 600 and np.isnan(c["bx"]).sum() == 600
    vis_c = np.isfinite(c["bx"]).sum()
    assert rep["gap_hidden"] >= int(0.10 * vis_c) and rep["gap_target"] == int(0.10 * vis_c)
    assert np.isnan(t1["bx"][np.isnan(c["bx"])]).all()                  # still frames stay gone
    ext = int(r4_inputs.EXTEND_MAX_S * r4_inputs.FPS)
    for a in (1000, 5000, 9000):                                        # extension after the run
        after = np.isnan(t1["bx"][a + 200:a + 200 + ext + 1])
        assert after.sum() <= ext + 1
    assert np.isfinite(tac["bx"]).sum() == len(tac["bx"]) - 0           # input untouched
    other, _ = r4_inputs.load_dfl(tac, "METRICA-S1", "c_train")
    assert other is tac
    with pytest.raises(SystemExit):
        r4_inputs.load_dfl(tac, "DFL-J03WQQ", "c")


def test_extension_lengths_within_bound():
    tac = _tac()
    c, _ = r4_inputs.load_dfl(tac, "DFL-J03WMX", "c")
    still = np.isnan(c["bx"])
    rng = np.random.RandomState(1)
    # Without the gaps of (a): set GAP_SHARE to 0 through a visible mask of nothing.
    bx, by, rep = r4_inputs.train_gaps(np.full_like(c["bx"], np.nan), c["by"], still, rng)
    assert rep["gap_target"] == 0
    bx, by, rep = r4_inputs.train_gaps(c["bx"], c["by"], still, np.random.RandomState(2))
    assert 0 <= rep["extended"] <= 3 * int(r4_inputs.EXTEND_MAX_S * r4_inputs.FPS)


def test_ball_free_copy():
    tac = _tac(100)
    bf = r4_inputs.ball_free(tac)
    assert np.isnan(bf["bx"]).all() and np.isnan(bf["by"]).all()
    assert np.isfinite(tac["bx"]).all() and bf["px"] is tac["px"]


# -- the training switches --------------------------------------------------------------------

torch = pytest.importorskip("torch")
from tools.learned_spike import train  # noqa: E402

GAMES = ["METRICA-S1", "DFL-J03WMX", "SC-1874553"]
NF = 50
N_FR = 1500          # frames -> 300 bins


@pytest.fixture
def _reset():
    yield
    train.configure()


def _codes(seed):
    rng = np.random.RandomState(seed)
    out = []
    while len(out) < N_FR:
        out += [int(rng.randint(0, 3))] * int(rng.randint(20, 200))
    return np.array(out[:N_FR], dtype=np.int8)


def _cache(tmp, name, val=0.0, label_shift=0, marker=None, views=("raw", "degraded")):
    root = os.path.join(str(tmp), name)
    for i, g in enumerate(GAMES):
        os.makedirs(os.path.join(root, "labels"), exist_ok=True)
        c = _codes(i + label_shift)
        np.savez(os.path.join(root, "labels", g + ".npz"), raw=c, ctl2=c)
        for v in views:
            os.makedirs(os.path.join(root, "feats", v), exist_ok=True)
            np.savez(os.path.join(root, "feats", v, g + ".npz"),
                     F=np.full((N_FR // 5, NF), val + (v == "degraded"), np.float32),
                     n_frames=np.int64(N_FR))
    if marker:
        with open(os.path.join(root, train.STILL_MARKER), "w") as h:
            json.dump({"dfl_still": "removed", "dfl_input": marker}, h)
    return root


def _aug(tmp, marker=None, k=2):
    from tools.learned_spike import aug_build
    root = os.path.join(str(tmp), "aug")
    for g in GAMES:
        for j in range(k):
            p = aug_build.feat_path(root, g, j)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.save(p, np.full((N_FR // 5, NF), 20.0 + j, np.float32))
    if marker:
        with open(os.path.join(root, train.STILL_MARKER), "w") as h:
            json.dump({"dfl_still": "removed", "dfl_input": marker}, h)
    return root


def _ballfree(tmp, marker="c_train", k=2):
    root = os.path.join(str(tmp), "bf")
    for g in GAMES:
        for fam in ["raw", "degraded"] + ["k{}".format(j) for j in range(k)]:
            p = train.ballfree_path(root, g, fam)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.save(p, np.full((N_FR // 5, NF), -100.0, np.float32))
    with open(os.path.join(root, train.STILL_MARKER), "w") as h:
        json.dump({"dfl_still": "removed", "dfl_input": marker}, h)
    return root


def test_defaults_unchanged(_reset):
    assert train.configure() == {"target": "ctl2", "short_weight": "off", "mix": "match"}
    assert train.r4_record() == {} and "r4" not in train.recipe(2000, 200)
    assert train.TRAIN_VIEWS is None and train.SC_LABELS is None and train.DFL_STILL == "kept"
    assert train.BALLFREE_CACHE is None and train.VAL_CACHE is None
    assert train.train_views("METRICA-S1") == folds.FULL_VIEWS


def test_default_batch_identical_to_round3_code(_reset):
    """With the round 4 switches off, sample_batch draws exactly as the round 3 code did."""
    bins = np.array(([0] * 30 + [1] * 30 + [2] * 20) * 20)[:1000]
    data = {g: {"feats": {"raw": np.random.RandomState(i).rand(1000, NF).astype(np.float32),
                          "degraded": np.zeros((1000, NF), np.float32)},
                "views": ("raw", "degraded"), "bins": bins, "bins_ctl2": bins,
                "aug": [np.ones((1000, NF), np.float32)]} for i, g in enumerate("AB")}
    train.configure(aug_cache="x")
    x, y, w = train.sample_batch(np.random.RandomState(7), data, ["A", "B"])
    # The round 3 sampler, written out (git show f8b4038:tools/learned_spike/train.py).
    rng = np.random.RandomState(7)
    from tools.learned_spike import features, model
    xs = []
    for _ in range(train.BATCH):
        item = data[["A", "B"][rng.randint(2)]]
        if rng.random_sample() < train.AUG_SHARE:
            feats = item["aug"][rng.randint(1)]
        else:
            feats = item["feats"][item["views"][rng.randint(2)]]
        start = rng.randint(0, max(1, feats.shape[0] - train.CROP_BINS))
        xx = np.asarray(feats[start:start + train.CROP_BINS])
        swap, mirror = bool(rng.randint(2)), bool(rng.randint(2))
        xs.append(features.transform(xx, swap, mirror))
    xr = torch.from_numpy(np.stack(xs)).float()
    xr = xr + train.INPUT_NOISE * torch.from_numpy(rng.standard_normal(xr.shape).astype(np.float32))
    assert torch.equal(x, xr)


def test_switch_a_views(tmp_path, _reset):
    with pytest.raises(ValueError):
        train.configure(views=("degraded",))            # raw must stay
    with pytest.raises(ValueError):
        train.configure(views=("raw", "blurred"))
    train.configure(views=("raw", "degraded"))
    assert train.train_views("DFL-J03WMX") == ("raw", "degraded")
    assert train.train_views("SC-1874553") == ("raw", "degraded")
    assert train.r4_record()["r4"]["views"] == ["raw", "degraded"]
    cache = _cache(tmp_path, "c")
    data = train.load_games(cache, ["DFL-J03WMX"])
    assert data["DFL-J03WMX"]["views"] == ("raw", "degraded")
    assert all("matched" not in p for p in train.cache_files(cache, ["DFL-J03WMX"]))


def test_switch_b_sc_labels(tmp_path, _reset):
    cache = _cache(tmp_path, "c")
    pg = _cache(tmp_path, "pg", label_shift=10, views=())
    train.configure(sc_labels=pg, views=("raw", "degraded"))
    data = train.load_games(cache, GAMES)
    assert np.array_equal(data["SC-1874553"]["raw"], _codes(12))         # possession groups
    assert np.array_equal(data["METRICA-S1"]["raw"], _codes(0))          # others unchanged
    val = train.load_games(cache, ["SC-1874553"], raw_only=True)
    assert np.array_equal(val["SC-1874553"]["raw"], _codes(12))          # validation too
    assert train.label_path(cache, "SC-1874553").startswith(pg)


def test_switch_c_markers_and_val_cache(tmp_path, _reset):
    plain = _cache(tmp_path, "plain")
    ctrain = _cache(tmp_path, "ct", val=5.0, marker="c_train")
    cval = _cache(tmp_path, "cv", val=9.0, marker="c", views=("raw",))
    with pytest.raises(ValueError):
        train.configure(dfl_still="removed")                             # needs val_cache
    train.configure(dfl_still="removed", val_cache=cval, views=("raw", "degraded"))
    with pytest.raises(SystemExit):
        train.load_games(plain, GAMES)                                   # no marker
    train.configure(dfl_still="removed", val_cache=ctrain, views=("raw", "degraded"))
    with pytest.raises(SystemExit):
        train.load_games(ctrain, GAMES)                                  # val folder not dfl_c
    train.configure(dfl_still="removed", val_cache=cval, views=("raw", "degraded"))
    tr = train.load_games(ctrain, ["DFL-J03WMX"])
    va = train.load_games(ctrain, ["DFL-J03WMX"], raw_only=True)
    assert float(tr["DFL-J03WMX"]["feats"]["raw"][0, 0]) == 5.0          # dfl_c_train
    assert float(va["DFL-J03WMX"]["feats"]["raw"][0, 0]) == 9.0          # dfl_c
    train.configure(views=("raw", "degraded"))
    with pytest.raises(SystemExit):
        train.load_games(ctrain, GAMES)                                  # marker but kept
    assert train.load_games(plain, ["METRICA-S1"])


def test_switch_d_splice(tmp_path, _reset):
    cache = _cache(tmp_path, "ct", marker="c_train")
    cval = _cache(tmp_path, "cv", marker="c", views=("raw",))
    aug = _aug(tmp_path, marker="c_train")
    bf = _ballfree(tmp_path)
    with pytest.raises(ValueError):
        train.configure(ballfree_share=0.2)                              # share without folder
    with pytest.raises(ValueError):
        train.configure(ballfree_cache=bf, views=("raw", "matched"))
    train.configure(aug_cache=aug, aug_k=2, views=("raw", "degraded"), dfl_still="removed",
                    val_cache=cval, ballfree_cache=bf)
    assert train.BALLFREE_SHARE == 0.15
    data = train.load_games(cache, GAMES)
    assert set(data["SC-1874553"]["ballfree"]) == {"raw", "degraded", "k0", "k1"}
    rng = np.random.RandomState(0)
    spliced, lengths = 0, []
    old = train.CROP_BINS
    try:
        train.CROP_BINS = 250
        for _ in range(40):
            x, y, _ = train.sample_batch(rng, data, GAMES)
            for row in x.numpy():
                low = np.abs(np.abs(row) - 100.0).max(axis=1) < 1.0      # ball-free bins
                if low.any():
                    spliced += 1
                    idx = np.flatnonzero(low)
                    assert idx[-1] - idx[0] + 1 == idx.size              # one window
                    lengths.append(idx.size)
    finally:
        train.CROP_BINS = old
    share = spliced / (40.0 * train.BATCH)
    assert 0.08 < share < 0.23
    assert min(lengths) >= train.BALLFREE_MIN_BINS and max(lengths) <= 250
    win = train.splice_ballfree(np.random.RandomState(3), np.zeros((300, 2)),
                                np.ones((400, 2)), 50)
    assert 100 <= win[:, 0].sum() <= 300


def test_all_switches_train_end_to_end(tmp_path, _reset):
    """A + B + C + D' + E together: a few steps of train_model on a synthetic cache."""
    cache = _cache(tmp_path, "ct", marker="c_train")
    cval = _cache(tmp_path, "cv", marker="c", views=("raw",))
    pg = _cache(tmp_path, "pg", label_shift=10, views=())
    aug = _aug(tmp_path, marker="c_train")
    bf = _ballfree(tmp_path)
    train.configure(target="raw", short_weight="sqrt", mix="sc_quarter", aug_cache=aug, aug_k=2,
                    views=("raw", "degraded"), sc_labels=pg, dfl_still="removed",
                    val_cache=cval, ballfree_cache=bf)
    data = train.load_games(cache, GAMES[1:])
    data.update(train.load_games(cache, ["METRICA-S1"], raw_only=True))
    state, rec = train.train_model(data, GAMES[1:], 4, 0, val_games=["METRICA-S1"],
                                   eval_every=2)
    assert rec["best_step"] in (2, 4) and rec["switches"]["short_weight"] == "sqrt"
    r4 = rec["switches"]["r4"]
    assert r4["views"] == ["raw", "degraded"] and r4["dfl_still"] == "removed"
    assert r4["ballfree"]["share"] == 0.15 and r4["sc_labels"] == os.path.abspath(pg)
    assert train.recipe(4, 2)["r4"] == r4


def test_switch_args_roundtrip(_reset):
    import argparse
    p = argparse.ArgumentParser()
    train.add_switch_args(p)
    a = p.parse_args(["--views", "raw", "degraded", "--sc-labels", "/x", "--dfl-still",
                      "removed", "--val-cache", "/v", "--ballfree-cache", "/b",
                      "--short-weight", "sqrt"])
    out = train.configure_args(a)
    assert out["views"] == ["raw", "degraded"] and out["dfl_still"] == "removed"
    assert train.SHORT_WEIGHT == "sqrt" and train.VAL_CACHE == "/v"


# -- the scoring tool -------------------------------------------------------------------------

from tools.learned_spike import r4_screen  # noqa: E402


def test_merge_share_definition():
    # reference: H 0-100, A 100-200, H 200-300 (all graded); prediction: H 0-300 swallows all
    ref = np.array([0] * 100 + [1] * 100 + [0] * 100, np.int8)
    pred = np.zeros(300, np.int8)
    m, s, tw = r4_screen.merge_share(ref, pred)
    assert m == 1.0 and s == 0.0 and tw == 1.0
    m, s, tw = r4_screen.merge_share(ref, ref)
    assert (m, s, tw) == (0.0, 0.0, 0.0)
    ref1 = np.array([0] * 100 + [1] * 200, np.int8)
    split = np.array([0] * 100 + [1] * 100 + [0] * 50 + [1] * 50, np.int8)  # A run cut in 3
    m, s, tw = r4_screen.merge_share(ref1, split)
    assert s == 0.75 and m == 0.0
    # time weighting: two short refs swallowed out of one long one
    ref2 = np.array([0] * 10 + [1] * 10 + [0] * 280, np.int8)
    pred2 = np.array([0] * 20 + [0] * 280, np.int8)
    m, _, tw = r4_screen.merge_share(ref2, pred2)
    assert m == 1.0 and tw == 1.0
    pred3 = np.array([1] * 20 + [0] * 280, np.int8)
    m, _, tw = r4_screen.merge_share(ref2, pred3)
    assert abs(m - 2 / 3.0) < 1e-9 and abs(tw - 20 / 300.0) < 1e-9


def test_recipe_score_rule():
    pairs = [(8.0, 4.0), (16.0, 8.0)]

    def s(g, mn, mc):
        return {"gain": g, "merge_net": mn, "merge_c23": mc}
    summ = {"T-M": [s(0.05, 0.3, 0.4), s(0.09, 0.5, 0.4)],
            "T-D": [s(0.02, 0.3, 0.6), s(0.10, 0.3, 0.6)],
            "T-S": [s(0.04, 0.2, 0.3), s(0.08, 0.2, 0.3)]}
    r = r4_screen.recipe_score(summ, pairs)
    assert [x["eligible"] for x in r["per_pair"]] == [True, False]
    assert r["score"] == 0.02 and r["pair"] == [8.0, 4.0]
    summ["T-M"][0]["merge_net"] = 0.9
    assert r4_screen.recipe_score(summ, pairs)["score"] is None


def test_score_refuses_bad_pair_and_tests_are_open():
    with pytest.raises(SystemExit):
        r4_screen.main(["score", "--out", "/tmp/x", "--cache", "c", "--sc-labels", "p",
                        "--c23-sc", "s", "--dfl-cache", "d", "--c23-dev", "e", "--c23-dfl", "f",
                        "--pairs", "32", "8"])
    assert r4_screen.test_games("T-D") == [g for g in folds.OPEN if g.startswith("DFL-")]
    assert len(r4_screen.test_games("T-S")) == 20
    for t in r4_screen.TESTS:
        assert not set(r4_screen.test_games(t)) & set(folds.SEALED)


def test_screen_pools_leave_provider_out():
    from tools.learned_spike import r3_diag
    for prov, name in r4_screen.PROVIDERS.items():
        spec = r3_diag.pool(name)
        assert not any(folds.provider(g) == prov for g in spec["train"] + spec["val"])
        assert not any(g.startswith("PFF-") for g in spec["train"] + spec["val"])


def test_launcher_refuses_bad_recipe():
    sh = os.path.join(ROOT, "tools", "learned_spike", "r4_screen_launch.sh")
    assert subprocess.call(["bash", "-n", sh]) == 0
    assert subprocess.call(["bash", sh, "ABX", "0"], stdout=subprocess.DEVNULL) == 2
    assert subprocess.call(["bash", sh, "ABC"], stdout=subprocess.DEVNULL) == 2


def test_c23_dev_data_root_dfl_only():
    sys.path.insert(0, os.path.join(ROOT, "tools", "learned_spike"))
    import r3_c23_dev
    with pytest.raises(SystemExit):
        r3_c23_dev.main(["--engine-repo", ".", "--ext", "e", "--out", "o", "--games",
                         "METRICA-S1", "--data-root", "/x", "--view", "dfl_c"])
    with pytest.raises(SystemExit):
        r3_c23_dev.main(["--engine-repo", ".", "--ext", "e", "--out", "o", "--games",
                         "DFL-J03WMX", "--data-root", "/x"])


# -- test T-P (PFF as a test only) ------------------------------------------------------------

def test_tp_refuses_pilot_and_sealed_pff():
    from tools.learned_spike import r4_pff_test as tp
    assert tp.games_of("pff") == list(folds.PFF_TRAIN) and len(tp.games_of("pff")) == 43
    for g in folds.PFF_SEALED[:3] + folds.PFF_PILOT[:2]:
        with pytest.raises(AssertionError):
            tp.check_games([g])
    for g in ("DFL-J03WQQ", "BAR-ATH", "METRICA-S1", "PFF-3845"):
        with pytest.raises((SystemExit, AssertionError)):
            tp.check_games([g])
    with pytest.raises(SystemExit):
        tp.main(["run", "--games-set", "sc", "--weights", "w.npz", "--pair", "32", "8",
                 "--work", "/tmp/r4_tp_test"])                    # pair outside the rule
    with pytest.raises(SystemExit):
        tp.main(["run", "--games-set", "sc", "--weights", "w.npz", "--pair", "16", "8",
                 "--work", ROOT])                                 # work inside the repository


def test_tp_overrides_and_c23_lists():
    from tools.learned_spike import r4_pff_test as tp
    o = tp.overrides("/x/w.npz", (16.0, 8.0), True)
    assert o == {"LEARNED_MODE": "model", "LEARNED_WEIGHTS": "/x/w.npz", "LEARNED_COST_HA": 16.0,
                 "LEARNED_COST_AD": 8.0, "LEARNED_STRUCTURE": True}
    assert "LEARNED_STRUCTURE" not in tp.overrides("/x/w.npz", (16.0, 8.0), False)
    assert tp.overrides("/x/w.npz", (16.0, 8.0), True, 1.0)["LEARNED_PRIOR_BONUS"] == 1.0
    assert "LEARNED_PRIOR_BONUS" not in tp.overrides("/x/w.npz", (16.0, 8.0), True, 0.0)
    assert tp.tag_of("abcdef0123456789", (16.0, 8.0), True, 1.0).endswith("_prior1")
    from analysis import external_sweep as xs
    resolved, _ = xs.resolve_config(tp.overrides("/x/w.npz", (16.0, 8.0), True, 2.0))
    assert resolved["LEARNED_PRIOR_BONUS"] == 2.0          # a real Config field
    sys.path.insert(0, os.path.join(ROOT, "tools", "learned_spike"))
    import r3_c23_dev
    assert r3_c23_dev.PFF_OPEN == list(folds.PFF_TRAIN)
    assert r3_c23_dev.SC == list(folds.SC)
    assert not set(r3_c23_dev.PFF_OPEN) & (set(folds.PFF_SEALED) | set(folds.PFF_PILOT))


def test_tp_rules_are_v1_rules():
    from tools.learned_spike import evaluate as ev
    games = ["G{}".format(i) for i in range(10)]
    base = {"ctl2": {g: 0.5 for g in games}, "provided": {g: 0.5 for g in games}}
    x = {"ctl2": {g: 0.55 for g in games}, "provided": {g: 0.52 for g in games}}
    r = ev.v1_rules(x, base, n_expected=10)
    assert r["pass"] and r["wins"] == 10
    x["ctl2"]["G0"] = 0.4
    x["ctl2"]["G1"] = 0.4
    x["ctl2"]["G2"] = 0.4
    x["ctl2"]["G3"] = 0.4
    x["ctl2"]["G4"] = 0.4
    assert not ev.v1_rules(x, base, n_expected=10)["c"]          # 5 of 10 < 60 %
    assert not ev.v1_rules(x, base, n_expected=11)["complete"]


# -- second round of mechanics: B', views raw, both SkillCorner references, grid, prior --------

def test_sc_alive_dead_loss_ignores_team_split():
    """B': on SkillCorner alive bins the loss depends on pH + pA only."""
    p = torch.tensor([[[0.5, 0.2, 0.3], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]]], requires_grad=True)
    logits = torch.log(p)                       # softmax(log p) = p
    y = torch.tensor([[0, 1, 2]])
    ce = train.sc_alive_dead_ce(logits, y, torch.tensor([True]))
    ce.sum().backward()
    g = p.grad[0]
    assert torch.allclose(g[0, 0], g[0, 1]) and torch.allclose(g[1, 0], g[1, 1])   # d/d(pH-pA)=0
    assert abs(float(ce[0, 0]) + np.log(0.7)) < 1e-6 and abs(float(ce[0, 2]) + np.log(0.6)) < 1e-6
    other = train.sc_alive_dead_ce(torch.log(p.detach()), y, torch.tensor([False]))
    ref = torch.nn.functional.cross_entropy(torch.log(p.detach())[0], y[0], reduction="none")
    assert torch.allclose(other[0], ref)        # other providers: plain cross entropy


def test_sc_alive_dead_switch_in_training(tmp_path, _reset):
    cache = _cache(tmp_path, "c")
    aug = _aug(tmp_path)
    train.configure(target="raw", mix="sc_quarter", aug_cache=aug, aug_k=2, views=("raw",),
                    sc_alive_dead=True)
    assert train.r4_record()["r4"]["sc_alive_dead"] is True
    data = train.load_games(cache, GAMES[1:])
    data.update(train.load_games(cache, ["METRICA-S1", "SC-1874553"][:1], raw_only=True))
    batch = train.sample_batch(np.random.RandomState(0), data, GAMES[1:])
    assert len(batch) == 4 and batch[3].dtype == torch.bool
    _, rec = train.train_model(data, GAMES[1:], 2, 0, val_games=["METRICA-S1"], eval_every=1)
    assert rec["switches"]["r4"]["sc_alive_dead"] is True
    train.configure(views=("raw",))
    assert len(train.sample_batch(np.random.RandomState(0), data, GAMES[1:])) == 3


def test_views_raw_only_with_ballfree(tmp_path, _reset):
    train.configure(views=("raw",), ballfree_cache=str(tmp_path))
    assert train.train_views("DFL-J03WMX") == ("raw",)


def test_decode_prior_calls_engine_prior():
    from engine import learned
    rng = np.random.RandomState(1)
    probs = rng.dirichlet([1, 1, 1], 300)
    blocks = [(1, 0, 299)]
    prior = rng.randint(0, 4, 300).astype(np.int8)
    for b in (0.5, 1.0, 2.0):
        assert np.array_equal(
            r4_screen.decode_prior(probs, blocks, 8.0, 4.0, None, prior, b),
            learned.decode_model(probs, blocks, 8.0, 4.0, None, rule_codes=prior, prior_bonus=b))


def test_decode_prior_zero_bonus_is_engine_decode():
    from engine import learned
    rng = np.random.RandomState(0)
    probs = rng.dirichlet([1, 1, 1], 400)
    blocks = [(1, 10, 199), (2, 220, 389)]
    prior = rng.randint(0, 4, 400).astype(np.int8)
    plain = learned.decode_model(probs, blocks, 8.0, 4.0)
    assert np.array_equal(r4_screen.decode_prior(probs, blocks, 8.0, 4.0, None, prior, 0.0),
                          plain)
    big = r4_screen.decode_prior(probs, blocks, 0.01, 0.01, None, prior, 50.0)
    inside = np.zeros(400, bool)
    inside[10:200] = True
    inside[220:390] = True
    sel = inside & (prior < 3)
    assert np.array_equal(big[sel], prior[sel])     # a huge bonus follows c23's labels
    assert (big[~inside] == 3).all()


def test_hidden_d_shares():
    ref = np.array([0] * 10 + [2] * 10, np.int8)
    codes = np.array([0] * 8 + [2] * 2 + [2] * 6 + [0] * 4, np.int8)
    miss = np.zeros(20, bool)
    miss[6:14] = True
    h = r4_screen.hidden_d_shares(codes, ref, miss)
    assert h["live_hidden_D"] == 0.5 and h["dead_hidden_D"] == 1.0
    assert h["live_hidden_frames"] == 4 and h["dead_hidden_frames"] == 4


def test_grid_pairs_rule():
    """Section 4 step 1 with deviation 8.1: the finer grid; round 3's GRID unchanged."""
    from tools.learned_spike import r3_views
    pairs = r4_screen.grid_pairs()
    assert len(r3_views.GRID) == 80
    assert len(pairs) == 146 and len(set(pairs)) == 146
    assert all(a <= 2 * b for a, b in pairs)
    assert {a for a, _ in pairs} == {0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64,
                                     96, 128}
    assert {b for _, b in pairs} == {0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64}
    assert set(p for p in r3_views.GRID if p[0] <= 2 * p[1]) <= set(pairs)
    assert set(r4_screen.PAIRS) <= set(pairs)
    assert r4_screen.PAIRS == ((8.0, 4.0), (8.0, 8.0), (16.0, 8.0), (16.0, 16.0), (32.0, 16.0))


def test_decode_cached_equals_plain(tmp_path, monkeypatch):
    from engine import data
    rng = np.random.RandomState(2)
    n = 400
    monkeypatch.setattr(data, "frame_count", lambda g: n)
    monkeypatch.setattr(data, "graded_blocks", lambda g: [(1, 0, n - 1)])
    monkeypatch.setattr(data, "kickoff_frames", lambda g: {1: 0})
    p5 = np.round(rng.dirichlet([1, 1, 1], n // 5), 4)
    c23 = rng.randint(0, 3, n).astype(np.int8)
    pairs = [(8.0, 4.0), (6.0, 3.0), (1.5, 1.0)]
    plain = r4_screen.decode_pairs(p5, "METRICA-S1", pairs, True, c23, 1.0)
    first = r4_screen.decode_cached(p5, "METRICA-S1", pairs, True, c23, 1.0, str(tmp_path))
    assert all(np.array_equal(a, b) for a, b in zip(plain, first))
    calls = []
    monkeypatch.setattr(r4_screen, "decode_pairs", lambda *a, **k: calls.append(a) or [])
    again = r4_screen.decode_cached(p5, "METRICA-S1", pairs[::-1], True, c23, 1.0, str(tmp_path))
    assert not calls and all(np.array_equal(a, b) for a, b in zip(plain[::-1], again))


def test_launcher_b_and_q_exclusive():
    sh = os.path.join(ROOT, "tools", "learned_spike", "r4_screen_launch.sh")
    assert subprocess.call(["bash", sh, "ABQC", "0"], stdout=subprocess.DEVNULL) == 2


# -- G9: sealed PFF judge (synthetic only; no sealed file is read) ----------------------------

def test_sealed_gate_and_prereg(tmp_path):
    from tools.learned_spike import r4_pff_test as tp
    pre = tmp_path / "prereg.md"
    with pytest.raises(SystemExit):
        tp.sealed_gate(False, str(pre), {})                        # no gate value
    with pytest.raises(SystemExit):
        tp.sealed_gate(False, str(pre), {"UNSEAL_PFF": "rehearsal-sc"})
    with pytest.raises(SystemExit):
        tp.sealed_gate(False, str(pre), {"UNSEAL_PFF": tp.SEALED_GATE})   # prereg missing
    pre.write_text("x")
    assert tp.sealed_gate(False, str(pre), {"UNSEAL_PFF": tp.SEALED_GATE}) == tp.SEALED_GATE
    with pytest.raises(SystemExit):
        tp.sealed_gate(True, str(pre), {"UNSEAL_PFF": tp.SEALED_GATE})    # rehearsal value
    assert tp.sealed_games(True)[:2] == list(folds.SC[:2])
    with pytest.raises(SystemExit):
        tp.sealed_games(True, ["METRICA-S1"])
    assert tp.sealed_games(False) == list(folds.PFF_SEALED)           # names only, no file


def test_sealed_refuses_second_run(tmp_path, monkeypatch):
    from tools.learned_spike import r4_pff_test as tp
    pre = tmp_path / "prereg.md"
    pre.write_text("x")
    out = tmp_path / "out"
    out.mkdir()
    started, _, _ = tp.sealed_paths(str(out), True)
    open(started, "w").close()
    monkeypatch.setenv("UNSEAL_PFF", tp.REHEARSAL_GATE)
    with pytest.raises(SystemExit):
        tp.main(["sealed", "--prereg", str(pre), "--weights", "w", "--weights-md5", "m",
                 "--pair", "16", "8", "--out", str(out), "--rehearse"])
    monkeypatch.delenv("UNSEAL_PFF")
    with pytest.raises(SystemExit):                                  # gate first, nothing read
        tp.main(["sealed", "--prereg", str(pre), "--weights", "w", "--weights-md5", "m",
                 "--pair", "16", "8", "--out", str(tmp_path / "o2")])
    assert not (tmp_path / "o2").exists()


def test_sealed_set_scoring_synthetic():
    from tools.learned_spike import r4_pff_test as tp
    games = ["S{}".format(i) for i in range(5)]
    ref = np.array(([0] * 40 + [2] * 20 + [1] * 40) * 5, np.int8)
    noisy = ref.copy()
    noisy[::7] = 1 - np.clip(noisy[::7], 0, 1)
    refs = {"provided": ref, "ctl2": ref, "md5": "x"}
    sm, sc, per, err = tp.run_sealed_set(
        games, lambda g, s: ref if s == "M" else noisy, lambda g: refs, log_fn=lambda t: None)
    assert err is None
    r = tp.judge(games, sm, sc)
    assert r["complete"] and r["pass"] and r["wins"] == 5

    def failing(g, s):
        if g == "S3":
            raise RuntimeError("engine failed")
        return ref
    sm, sc, per, err = tp.run_sealed_set(games, failing, lambda g: refs, log_fn=lambda t: None)
    assert err and not tp.judge(games, sm, sc)["complete"]          # unfinished: fails
    assert not tp.judge(games, sm, sc)["pass"]


def test_step_f_aug_marker_accepted(tmp_path, _reset):
    """Step F: an aug folder whose marker carries an extra "ball" record (dfl_c_train input)."""
    cache = _cache(tmp_path, "ct", marker="c_train")
    cval = _cache(tmp_path, "cv", marker="c", views=("raw",))
    aug = _aug(tmp_path, marker="c_train")
    mark = os.path.join(aug, train.STILL_MARKER)
    rec = json.load(open(mark))
    rec["ball"] = {"sigma": [0.0, 4.0], "offset_m": [0.0, 1.5], "p_on": 0.75}
    json.dump(rec, open(mark, "w"))
    train.configure(aug_cache=aug, aug_k=2, views=("raw",), dfl_still="removed", val_cache=cval)
    assert train.check_still(cache) == "removed"
    assert len(train.load_games(cache, ["DFL-J03WMX"])["DFL-J03WMX"]["aug"]) == 2


def test_launcher_letter_f_needs_c():
    sh = os.path.join(ROOT, "tools", "learned_spike", "r4_screen_launch.sh")
    assert subprocess.call(["bash", sh, "ABF", "0"], stdout=subprocess.DEVNULL) == 2
