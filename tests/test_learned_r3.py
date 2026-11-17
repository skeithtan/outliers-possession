"""Round 3 switches (LEARNED_STAGE_PROTOCOL_R3.md, DRAFT): defaults reproduce round 2."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.learned_spike import r3_nested, train  # noqa: E402


def _item(bins_ctl2, bins_raw, n_feat=50):
    n = len(bins_raw)
    return {"feats": {"raw": np.zeros((n, n_feat), dtype=np.float32)}, "views": ("raw",),
            "bins_ctl2": np.asarray(bins_ctl2), "bins": np.asarray(bins_raw)}


@pytest.fixture(autouse=True)
def _reset():
    yield
    train.configure()


def test_defaults_are_round2():
    assert train.configure() == {"target": "ctl2", "short_weight": "off", "mix": "match"}
    rec = train.recipe(2000, 200)
    assert rec["target"] == "ctl2" and rec["short_weight"] == "off" and rec["mix"] == "match"


def test_bad_switch_raises():
    with pytest.raises(ValueError):
        train.configure(target="2s")


def test_short_weights_formula():
    bins = np.array([0] * 10 + [1] * 2 + [2] * 3 + [0] * 40)
    w = train.short_weights(bins, median=10.0)
    assert np.allclose(w[:10], 1.0)                      # sqrt(10/10)
    assert np.allclose(w[10:12], np.sqrt(5.0))           # sqrt(10/2) = 2.236 < 3
    assert np.allclose(w[12:15], 1.0)                    # D bins
    assert np.allclose(w[15:], 1.0 / 2.0)                # sqrt(10/40) = 0.5 > 1/3
    w2 = train.short_weights(np.array([0] + [1] * 400), median=100.0)
    assert np.isclose(w2[0], 3.0) and np.isclose(w2[1], 0.5)


def test_batch_reads_target_view():
    ctl = [0] * 700
    raw = [0] * 300 + [1] * 400
    data = {"G": _item(ctl, raw)}
    train.configure(target="raw")
    _, y, _ = train.sample_batch(np.random.RandomState(0), data, ["G"])
    assert all(len(np.unique(row)) == 2 for row in y.numpy())     # every crop holds a change
    train.configure(target="ctl2")
    data["G"]["bins"] = data["G"]["bins_ctl2"]       # what load_games stores under ctl2
    _, y, _ = train.sample_batch(np.random.RandomState(0), data, ["G"])
    assert all(len(np.unique(row)) == 1 for row in y.numpy())     # one team per crop (swap aside)


def test_default_batch_unchanged_by_switch_code():
    """Under the defaults the batch equals the round 2 draw (same RNG consumption)."""
    rng_a, rng_b = np.random.RandomState(3), np.random.RandomState(3)
    bins = np.array(([0] * 30 + [1] * 30 + [2] * 20) * 10)
    data = {g: _item(bins, bins) for g in ("A", "B", "C")}
    xa, ya, wa = train.sample_batch(rng_a, data, ["A", "B", "C"])
    train.configure(mix="match", short_weight="off")
    xb, yb, wb = train.sample_batch(rng_b, data, ["A", "B", "C"])
    assert torch.equal(xa, xb) and torch.equal(ya, yb) and torch.equal(wa, wb)


def test_kill_check():
    ok = {g: {"delta": 0.03, "ratio": 0.8} for g in r3_nested.DEV}
    assert r3_nested.kill_check(ok, 0.73)[0]
    bad = dict(ok)
    bad["METRICA-S3"] = {"delta": 0.005, "ratio": 0.8}
    assert not r3_nested.kill_check(bad, 0.73)[0]
    bad = dict(ok)
    bad["DFL-J03WOH"] = {"delta": -0.02, "ratio": 0.8}
    assert not r3_nested.kill_check(bad, 0.73)[0]
    low = {g: {"delta": 0.03, "ratio": 0.7} for g in r3_nested.DEV}
    assert not r3_nested.kill_check(low, 0.73)[0]


def test_nested_pick_leaves_test_out():
    scores = {g: [0.0] * len(r3_nested.GRID) for g in r3_nested.DEV}
    for g in r3_nested.DEV:
        scores[g][5] = 1.0
    scores["METRICA-S1"] = [0.0] * len(r3_nested.GRID)
    scores["METRICA-S1"][7] = 100.0
    assert r3_nested.nested_pick(scores, "METRICA-S1")[0] == 5


def test_r3b_folds_without_pff():
    from tools.learned_spike import folds
    for k in range(folds.N_FOLDS):
        with_pff, without = folds.fold(k), folds.fold(k, pff=False)
        assert not any(g.startswith("PFF-") for g in without["train"])
        assert without["train"] == [g for g in with_pff["train"] if not g.startswith("PFF-")]
        assert without["test"] == with_pff["test"] and without["val"] == with_pff["val"]
        assert not set(without["test"]) & set(without["train"])
    assert not any(g.startswith("PFF-") for g in folds.train_games(pff=False))


def test_sc_quarter_mix():
    from tools.learned_spike import folds
    games = folds.fold(0, pff=False)["train"]
    train.configure(mix="sc_quarter")
    shares = train.crop_shares(games)
    n_sc = sum(folds.provider(g) == "skillcorner" for g in games)
    n_other = len(games) - n_sc
    assert np.isclose(shares["skillcorner"], 0.25 * n_sc / (0.25 * n_sc + n_other))
    bins = np.array([0] * 700)
    data = {}
    for g in games:
        data[g] = _item(bins, bins)
        data[g]["feats"]["raw"][:, 0] = 1.0 if folds.provider(g) == "skillcorner" else 0.0
    rng = np.random.RandomState(0)
    sc = []
    for _ in range(200):
        x, _, _ = train.sample_batch(rng, data, games)
        sc += list((x[:, 0, 0].numpy() > 0.5))
    assert abs(np.mean(sc) - shares["skillcorner"]) < 0.04
    train.configure()
    ref = folds.crop_shares(games)
    got = train.crop_shares(games)
    assert set(ref) == set(got) and all(np.isclose(ref[p], got[p]) for p in ref)
