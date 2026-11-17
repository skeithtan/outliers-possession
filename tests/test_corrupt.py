"""Unit tests of the corruption view, tools/learned_spike/corrupt.py, on a synthetic match.

The synthetic match has 22 players on random walks at 25 fps and a ball that is always seen.
The tests check that the view is deterministic, seeded per match, label-free, that it keeps the
team of each player except on cross-team swaps, that it keeps the slot cap, and that it hides
the ball on graded frames only.
"""

import inspect

import numpy as np
import pytest

from tools.learned_spike import corrupt as C

FPS = 25
PITCH = (105.0, 68.0)
N = 20000
BLOCKS = [(1, 500, 9499), (2, 10500, 19499)]
GAME = "METRICA-S1"   # any games.py name that is not forbidden; it only seeds the draws


def synthetic(seed=0, close_pair=False):
    """A tactical dict: 11 home and 11 away players on random walks, ball always seen."""
    rng = np.random.RandomState(seed)
    n_slots = 22
    start = rng.uniform(0.1, 0.9, size=(1, n_slots, 2))
    steps = rng.normal(0.0, 0.0008, size=(N, n_slots, 2))
    pos = np.clip(start + np.cumsum(steps, axis=0), 0.02, 0.98)
    if close_pair:
        # Home slot 0 and away slot 11 run side by side, 0.3 m apart, for the whole match.
        pos[:, 11, 0] = pos[:, 0, 0] + 0.3 / PITCH[0]
        pos[:, 11, 1] = pos[:, 0, 1]
        # Home slots 1 and 2 also run side by side.
        pos[:, 2, 0] = pos[:, 1, 0] + 0.3 / PITCH[0]
        pos[:, 2, 1] = pos[:, 1, 1]
    px = pos[:, :, 0].astype(np.float32)
    py = pos[:, :, 1].astype(np.float32)
    ball = np.clip(0.5 + np.cumsum(rng.normal(0, 0.002, size=(N, 2)), axis=0), 0, 1)
    return {"px": px, "py": py, "ps": np.full((N, n_slots), 2.0, dtype=np.float32),
            "bx": ball[:, 0].astype(np.float32), "by": ball[:, 1].astype(np.float32),
            "team_of_slot": np.array([0] * 11 + [1] * 11, dtype=np.int8),
            "n_frames": N, "n_slots": n_slots, "fps": FPS, "game": GAME}


def run(tac, game=GAME, **kwargs):
    return C.corrupt(tac, game, blocks=BLOCKS, fps=FPS, pitch=PITCH, **kwargs)


def owners(tac, out):
    """int (N, out slots): the input column whose position each output slot shows, -1 if none."""
    found = np.full(out["px"].shape, -1, dtype=np.int64)
    for k in range(tac["px"].shape[1]):
        same = (out["px"] == tac["px"][:, [k]]) & (out["py"] == tac["py"][:, [k]])
        found[same] = k
    return found


def test_deterministic():
    tac = synthetic()
    a, ra = run(tac)
    b, rb = run(tac)
    for key in ("px", "py", "ps", "bx", "by", "team_of_slot"):
        np.testing.assert_array_equal(a[key], b[key])
    assert ra == rb


def test_seeded_per_match():
    tac = synthetic()
    a, _ = run(tac, game="METRICA-S1")
    b, _ = run(tac, game="METRICA-S2")
    assert not np.array_equal(np.isfinite(a["px"]), np.isfinite(b["px"]))


def test_input_not_changed():
    tac = synthetic()
    copy = {k: np.array(v, copy=True) for k, v in tac.items() if isinstance(v, np.ndarray)}
    run(tac)
    for key, value in copy.items():
        np.testing.assert_array_equal(tac[key], value)


def test_label_free():
    params = inspect.signature(C.corrupt).parameters
    assert not any(name in params for name in ("codes", "labels", "reference", "gt"))
    source = inspect.getsource(C)
    for word in ("evaluation/gt", "read_label_csv", "possession_metrics", "control_labels"):
        assert word not in source


def test_forbidden_matches():
    tac = synthetic()
    for game in ("BAR-ATH", "DFL-J03WQQ", "DFL-J03WR9"):
        with pytest.raises(SystemExit):
            run(tac, game=game)


def test_slot_cap_and_layout():
    out, report = run(synthetic(close_pair=True))
    assert out["n_slots"] == sum(C.POOL) == out["px"].shape[1]
    np.testing.assert_array_equal(out["team_of_slot"],
                                  [0] * C.POOL[0] + [1] * C.POOL[1])
    visible = np.isfinite(out["px"])
    for team, cap in enumerate(C.POOL):
        assert visible[:, out["team_of_slot"] == team].sum(axis=1).max() <= cap
    assert np.array_equal(visible, np.isfinite(out["py"]))
    assert report["chunks"] > 22  # fragmentation cut the tracks


def test_each_player_once_per_frame():
    tac = synthetic(close_pair=True)
    out, _ = run(tac)
    own = owners(tac, out)
    visible = np.isfinite(out["px"])
    assert (own[visible] >= 0).all()
    for k in range(22):
        assert (own == k).sum(axis=1).max() <= 1


def test_team_preserved_without_cross_swaps(monkeypatch):
    monkeypatch.setattr(C, "SWAP_CROSS_S", 0.0)
    tac = synthetic(close_pair=True)
    out, report = run(tac)
    assert report["swaps_cross"] == 0
    assert report["swaps_same"] > 0
    own = owners(tac, out)
    visible = own >= 0
    slot_team = np.broadcast_to(out["team_of_slot"], own.shape)
    assert np.array_equal(tac["team_of_slot"][own[visible]], slot_team[visible])


def test_cross_swap_moves_team(monkeypatch):
    monkeypatch.setattr(C, "SWAP_CROSS_S", 30.0)
    tac = synthetic(close_pair=True)
    out, report = run(tac)
    assert report["swaps_cross"] > 0
    own = owners(tac, out)
    visible = own >= 0
    slot_team = np.broadcast_to(out["team_of_slot"], own.shape)
    assert (tac["team_of_slot"][own[visible]] != slot_team[visible]).any()


def test_no_fragmentation_keeps_every_frame():
    tac = synthetic()
    out, report = run(tac, frag=False, swap=False, ball="off")
    assert report["chunks"] == 22
    assert np.isfinite(out["px"]).sum() == np.isfinite(tac["px"]).sum()
    np.testing.assert_array_equal(out["bx"], tac["bx"])


def test_ball_hidden_on_graded_frames_only():
    tac = synthetic()
    out, _ = run(tac)
    graded = C.graded_mask(BLOCKS, N)
    hidden = np.isfinite(tac["bx"]) & ~np.isfinite(out["bx"])
    assert not hidden[~graded].any()
    share = np.isfinite(out["bx"][graded]).mean()
    # The last gap can pass the target by its own length, which is large on a short match.
    assert abs(share - C.V_BAR) < 0.08


def test_sample_table_inside_range():
    rng = np.random.RandomState(0)
    draws = [C.sample_table(rng, C.LIFE_Q) for _ in range(2000)]
    assert min(draws) >= C.LIFE_Q[0] and max(draws) <= C.LIFE_Q[-1]
    assert abs(np.median(draws) - C.LIFE_Q[10]) < 0.3 * C.LIFE_Q[10]


# -- round 2: --seed and the V2 view on the competition match (R2 Section 3) ---------------------

V2 = dict(frag=False, flip_p=0.0, swap=False, ball="uniform30", ghost_per_10min=0.0)


def test_seed_replaces_the_match_name():
    tac = synthetic()
    a, ra = run(tac, game="METRICA-S1", seed=0, **V2)
    b, rb = run(tac, game="METRICA-S2", seed=0, **V2)
    np.testing.assert_array_equal(a["bx"], b["bx"])          # same seed: the same gaps
    assert ra["ball_seed"] == C.SEED_BALL == C.seed_of("anything", C.SEED_BALL, 0)
    c, _ = run(tac, game="METRICA-S1", **V2)                   # round 1 seeding by name
    assert not np.array_equal(np.isfinite(a["bx"]), np.isfinite(c["bx"]))
    assert C.seed_of("METRICA-S1", 19) == C.seed_of("METRICA-S1", 19, None)


def test_v2_view_hides_the_ball_only():
    tac = synthetic()
    out, report = run(tac, game="BAR-ATH", seed=0, allow_competition=True, **V2)
    # no fragmentation, no swaps: every player frame survives, only the ball goes
    assert np.isfinite(out["px"]).sum() == np.isfinite(tac["px"]).sum()
    hidden = C.hidden_ball(tac, out)
    graded = C.graded_mask(BLOCKS, N)
    assert hidden.any() and not hidden[~graded].any()
    assert abs(hidden[graded].mean() - C.UNIFORM30[0]) < 0.05
    assert report["swaps_same"] == report["swaps_cross"] == 0


def test_competition_match_only_with_the_v2_view_and_a_seed():
    tac = synthetic()
    with pytest.raises(SystemExit):
        run(tac, game="BAR-ATH", **V2)                                    # no allow flag
    with pytest.raises(SystemExit):
        run(tac, game="BAR-ATH", allow_competition=True, **V2)            # no seed
    for change in ({"frag": True}, {"swap": True}, {"ball": "matched"},
                   {"ghost_per_10min": 1.0}, {"flip_p": 0.1}):
        view = dict(V2, **change)
        with pytest.raises(SystemExit):
            run(tac, game="BAR-ATH", seed=0, allow_competition=True, **view)
    for game in ("DFL-J03WQQ", "DFL-J03WR9"):
        with pytest.raises(SystemExit):
            run(tac, game=game, seed=0, allow_competition=True, **V2)


def test_cli_refuses_the_competition_match_outside_v2(tmp_path):
    out = str(tmp_path / "x.csv")
    base = ["--game", "BAR-ATH", "--out", out]
    with pytest.raises(SystemExit):
        C.main(["engine"] + base)
    with pytest.raises(SystemExit):
        C.main(["stats"] + base + ["--allow-competition", "--seed", "0", "--ball", "uniform30",
                                   "--no-frag", "--no-swap", "--ghost-per-10min", "0"])
    with pytest.raises(SystemExit):
        C.main(["engine"] + base + ["--allow-competition", "--seed", "0", "--ball", "uniform30",
                                    "--no-frag", "--ghost-per-10min", "0"])
