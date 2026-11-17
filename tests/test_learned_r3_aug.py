"""Round 3 augmentation switch (augment.py, train.AUG_CACHE; docs/plan/calibration/R3_AUG.md)."""

import os

import numpy as np
import pytest

from tools.learned_spike import augment as au

PITCH = (105.0, 68.0)
N = 3000
D_SEG = (1000, 1250)          # one dead-ball segment


def _tac(seed=0, n=N):
    """A synthetic match: 11 + 11 players on random walks, the ball near player 0."""
    rng = np.random.RandomState(seed)
    px = np.clip(0.5 + np.cumsum(rng.normal(0, 0.002, (n, 22)), axis=0)
                 + rng.uniform(-0.3, 0.3, 22), 0.01, 0.99).astype(np.float32)
    py = np.clip(0.5 + np.cumsum(rng.normal(0, 0.002, (n, 22)), axis=0)
                 + rng.uniform(-0.3, 0.3, 22), 0.01, 0.99).astype(np.float32)
    # Three players crowd the ball on the first half of the frames.
    bx = px[:, 0] + 0.003
    by = py[:, 0].copy()
    for j, dx in ((1, 0.01), (12, -0.01)):
        px[: n // 2, j] = px[: n // 2, 0] + dx
        py[: n // 2, j] = py[: n // 2, 0]
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    return {"px": px, "py": py, "ps": np.ones_like(px), "bx": bx.astype(np.float32),
            "by": by.astype(np.float32), "team_of_slot": team, "n_slots": 22, "n_frames": n}


def _codes(n=N):
    c = np.zeros(n, dtype=np.int64)
    c[n // 2:] = 1
    c[D_SEG[0]:D_SEG[1]] = 2
    return c


BLOCKS = [(1, 0, N - 1)]


def _team_counts(tac):
    vis = np.isfinite(tac["px"])
    team = np.asarray(tac["team_of_slot"])
    return vis[:, team == 0].sum(axis=1), vis[:, team == 1].sum(axis=1)


def _only(name, rates):
    return au.augment(_tac(), _codes(), "SYN", 0, BLOCKS, PITCH, only={name: rates})


# -- seeds and plan ---------------------------------------------------------------------------

def test_seeds_differ_from_stress_view():
    from tools.learned_spike import corrupt, folds
    assert au.STRESS_OFFSETS == (corrupt.SEED_FRAG, corrupt.SEED_FLIP, corrupt.SEED_SWAP,
                                 corrupt.SEED_BALL, corrupt.SEED_GHOST, corrupt.SEED_SWAP_CROSS)
    assert au.check_seeds(folds.ALL, 16) == len(folds.ALL) * 16 * (len(au.COMPONENTS) + 1)
    for g in folds.OPEN:
        assert corrupt.seed_of(g, corrupt.SEED_BALL) in au.stress_seeds(g)


def test_plan_deterministic_and_varied():
    assert au.plan("G", 3) == au.plan("G", 3)
    plans = [au.plan("G", k) for k in range(16)]
    assert len({str(p) for p in plans}) == 16
    for p in plans:
        assert set(p) == set(au.COMPONENTS) and "swap" not in p
        if p["live"]:
            assert au.LIVE_SHARE[0] <= p["live"]["share"] <= au.LIVE_SHARE[1]


def test_augment_deterministic_and_input_untouched():
    tac = _tac()
    keep = {k: np.array(v, copy=True) for k, v in tac.items() if isinstance(v, np.ndarray)}
    a, ra = au.augment(tac, _codes(), "G", 5, BLOCKS, PITCH)
    b, rb = au.augment(tac, _codes(), "G", 5, BLOCKS, PITCH)
    for k, v in keep.items():
        assert np.array_equal(v, tac[k], equal_nan=True)
    for k in ("px", "py", "bx", "by"):
        assert np.array_equal(a[k], b[k], equal_nan=True)
    assert a["px"].shape == (N, 22 + 2 * au.EXTRA_SLOTS) and a["n_slots"] == 44
    assert list(a["team_of_slot"][22:]) == [0] * 11 + [1] * 11
    c, _ = au.augment(tac, _codes(), "G", 6, BLOCKS, PITCH)
    assert not np.array_equal(a["px"], c["px"], equal_nan=True)


def test_all_off_is_identity_on_values():
    a, _ = au.augment(_tac(), _codes(), "G", 0, BLOCKS, PITCH, only={})
    t = _tac()
    assert np.array_equal(a["px"][:, :22], t["px"]) and np.isnan(a["px"][:, 22:]).all()
    assert np.array_equal(a["bx"], t["bx"])


# -- each defect ------------------------------------------------------------------------------

def test_dead_blanking_and_return_jitter():
    jit = int(round(au.RETURN_JITTER_S * au.FPS))
    ends = set()
    for k in range(30):
        a, rep = au.augment(_tac(), _codes(), "G", k, BLOCKS, PITCH, only={"dead": {"p": 1.0}})
        hid = np.flatnonzero(np.isnan(a["bx"]))
        assert hid.size and rep["dead"] == hid.size
        assert D_SEG[0] <= hid[0] <= D_SEG[0] + au.DEAD_LEAD_MAX_S * au.FPS
        assert D_SEG[1] - jit <= hid[-1] + 1 <= D_SEG[1] + jit
        assert hid.size == hid[-1] - hid[0] + 1           # one contiguous gap
        ends.add(hid[-1] + 1 - D_SEG[1])
    assert min(ends) < 0 < max(ends)                      # early and late returns both occur
    a, _ = _only("dead", {"p": 0.0})
    assert np.isfinite(a["bx"]).all()


def test_live_gaps_share_and_crowd_placement():
    a, rep = _only("live", {"share": 0.2, "p_crowd": 1.0})
    alive = np.isin(_codes(), (0, 1))
    hid = np.isnan(a["bx"]) & alive
    assert hid.sum() >= 0.2 * alive.sum() and rep["live"] == np.isnan(a["bx"]).sum()
    crowd = au.crowd_frames(_tac()["px"], _tac()["py"], _tac()["bx"], _tac()["by"], PITCH)
    assert crowd[: N // 2].mean() > 0.9 and not crowd[N // 2:].any()
    starts = np.flatnonzero(np.diff(np.concatenate(([0], np.isnan(a["bx"]).astype(int)))) == 1)
    assert crowd[starts].all()                            # every gap starts on a crowded frame


def test_single_dropouts_share():
    a, rep = _only("drop", {"share": 0.1})
    vis = np.isfinite(a["px"][:, :22])
    assert 0.1 <= 1 - vis.mean() < 0.3 and rep["drop"] == (~vis).sum()
    assert np.array_equal(np.isfinite(a["py"]), np.isfinite(a["px"]))


def test_group_dropout_hides_far_players_of_one_team():
    a, rep = _only("group", {"per_10min": 60.0})
    assert rep["group"] > 0
    h, aw = _team_counts(a)
    low = (h <= 11 - au.GROUP_M[0]) | (aw <= 11 - au.GROUP_M[0])
    assert low.any()                                      # several players of a team at once
    t = int(np.flatnonzero(low)[0])
    side = 0 if h[t] < 11 else 1
    tac = _tac()
    cols = np.flatnonzero(tac["team_of_slot"] == side)
    hidden = cols[~np.isfinite(a["px"][t, cols])]
    shown = cols[np.isfinite(a["px"][t, cols])]
    d = np.hypot((tac["px"][t] - tac["bx"][t]) * PITCH[0], (tac["py"][t] - tac["by"][t]) * PITCH[1])
    if shown.size:
        assert d[hidden].min() >= d[shown].max() - 1e-6   # the farthest ones go


def test_wrong_team_makes_overcount():
    a, rep = _only("wrong", {"per_10min": 30.0})
    assert rep["wrong"] > 0
    h, aw = _team_counts(a)
    assert ((h > 11) | (aw > 11)).any()
    assert np.array_equal(h + aw, np.full(N, 22))         # a player moves, none is added
    # The moved positions are real positions of the other team's players at that frame.
    t = int(np.flatnonzero((h > 11) | (aw > 11))[0])
    tac = _tac()
    got = {(round(float(x), 5), round(float(y), 5)) for x, y in zip(a["px"][t], a["py"][t])
           if np.isfinite(x)}
    assert got == {(round(float(x), 5), round(float(y), 5)) for x, y in zip(tac["px"][t], tac["py"][t])}


def test_wrong_team_prefers_player_near_ball():
    a, _ = _only("wrong", {"per_10min": 60.0})
    team = a["team_of_slot"]
    moved_src = ~np.isfinite(a["px"][:, :22])
    near = moved_src[:, 0].sum()                          # player 0 carries the ball
    assert near > moved_src[:, 5].sum()


def test_ghosts_duplicate_within_offset():
    a, rep = _only("ghost", {"per_10min": 30.0})
    assert rep["ghost"] > 0
    h, aw = _team_counts(a)
    assert ((h > 11) | (aw > 11)).any()
    extra = np.flatnonzero(np.isfinite(a["px"][:, 22:]).any(axis=0)) + 22
    t = int(np.flatnonzero(np.isfinite(a["px"][:, extra[0]]))[0])
    d = np.hypot((a["px"][t, :22] - a["px"][t, extra[0]]) * PITCH[0],
                 (a["py"][t, :22] - a["py"][t, extra[0]]) * PITCH[1])
    src = int(np.argmin(d))
    assert d[src] <= au.GHOST_OFFSET_M[1] + 1e-3
    assert a["team_of_slot"][src] == a["team_of_slot"][extra[0]]


def test_reslot_keeps_positions_and_teams():
    tac = _tac()
    tac["px"][500:600, 3] = np.nan                        # a gap: the slot has two runs
    tac["py"][500:600, 3] = np.nan
    a, rep = au.augment(tac, _codes(), "G", 0, BLOCKS, PITCH, only={"reslot": {"p": 1.0}})
    assert rep["reslot"] == 1
    assert np.isnan(a["px"][600:, 3]).all()
    moved = np.flatnonzero(np.isfinite(a["px"][700]) & (np.arange(44) != 3)
                           & (a["team_of_slot"] == 0))
    for t in (100, 700):
        for side in (0, 1):
            m = a["team_of_slot"] == side
            got = np.sort(a["px"][t, m][np.isfinite(a["px"][t, m])])
            want = np.sort(tac["px"][t, tac["team_of_slot"] == side])
            assert np.array_equal(got, want[np.isfinite(want)])
    assert moved.size == 11


def test_no_identity_swap_component():
    """The left-out defect: no component exchanges two players' tracks."""
    assert "swap" not in au.COMPONENTS and au.LEFT_OUT == ("swap",)
    a, _ = au.augment(_tac(), _codes(), "G", 0, BLOCKS, PITCH,
                      only={"drop": {"share": 0.1}, "reslot": {"p": 1.0}})
    tac = _tac()
    # A column may carry another player after a gap (fragmentation), never within a run.
    for col in range(a["px"].shape[1]):
        starts, stops = au._runs(np.isfinite(a["px"][:, col]))
        for s, e in zip(starts, stops):
            src = np.flatnonzero(tac["px"][s] == a["px"][s, col])
            assert src.size >= 1
            assert np.array_equal(a["px"][s:e, col], tac["px"][s:e, src[0]])


# -- the training switch ----------------------------------------------------------------------

torch = pytest.importorskip("torch")
from tools.learned_spike import aug_build, train  # noqa: E402


@pytest.fixture
def _reset():
    yield
    train.configure()


def _item(n=1000, nf=50, val=0.0):
    bins = np.array(([0] * 30 + [1] * 30 + [2] * 20) * (n // 80 + 1))[:n]
    return {"feats": {"raw": np.full((n, nf), val, dtype=np.float32)}, "views": ("raw",),
            "bins_ctl2": bins, "bins": bins}


def test_switch_off_is_default_and_unchanged(_reset):
    assert train.configure() == {"target": "ctl2", "short_weight": "off", "mix": "match"}
    assert train.AUG_CACHE is None and train.aug_record() == {}
    assert "aug" not in train.recipe(2000, 200)
    data = {g: _item() for g in "ABC"}
    xa, ya, wa = train.sample_batch(np.random.RandomState(3), data, list("ABC"))
    train.configure(aug_cache=None)
    data["A"]["aug"] = [np.ones((1000, 50), dtype=np.float32)]   # ignored while off
    ra = np.random.RandomState(3)
    xb, yb, wb = train.sample_batch(ra, data, list("ABC"))
    assert torch.equal(xa, xb) and torch.equal(ya, yb) and torch.equal(wa, wb)


def test_switch_on_draws_aug_views(tmp_path, _reset):
    data = {"A": _item(val=0.0)}
    for k in range(4):
        p = aug_build.feat_path(str(tmp_path), "A", k)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.save(p, np.full((1000, 50), 10.0 + k, dtype=np.float32))
    train.configure(aug_cache=str(tmp_path), aug_k=4)
    data["A"]["aug"] = train.load_aug("A")
    assert len(data["A"]["aug"]) == 4 and isinstance(data["A"]["aug"][0], np.memmap)
    rng = np.random.RandomState(0)
    means = []
    for _ in range(40):
        x, _, _ = train.sample_batch(rng, data, ["A"])
        means += [float(np.abs(v).mean()) for v in x.numpy()]
    from_aug = np.mean([m > 5 for m in means])
    assert 0.35 < from_aug < 0.65                        # AUG_SHARE = 0.5
    assert train.aug_record()["aug"]["k"] == 4


def test_load_aug_refuses_incomplete(tmp_path, _reset):
    p = aug_build.feat_path(str(tmp_path), "METRICA-S1", 0)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p, np.zeros((10, 50), dtype=np.float32))
    train.configure(aug_cache=str(tmp_path), aug_k=2)
    with pytest.raises(SystemExit):
        train.load_aug("METRICA-S1")
    with pytest.raises(SystemExit):
        train.load_aug("SC-1874553")                      # an external match without views
