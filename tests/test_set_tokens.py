"""Round 8 arm S: the token builder and the set network (tools only, torch needed for the net).

The token test runs on a synthetic tactical dict of an external match's shape (no match data
is read beyond the pitch size and graded blocks of a games.py entry when available).
"""

import importlib.util
import os

import numpy as np
import pytest

HAVE_TORCH = importlib.util.find_spec("torch") is not None


def _tac(n=1000, seed=0):
    rng = np.random.RandomState(seed)
    slots = 28
    team = np.array([0] * 14 + [1] * 14)
    base_x = np.where(team == 0, 0.3, 0.7)
    px = base_x[None, :] + 0.05 * np.cumsum(rng.standard_normal((n, slots)) * 0.01, axis=0)
    py = rng.uniform(0.1, 0.9, slots)[None, :] + 0.05 * np.cumsum(
        rng.standard_normal((n, slots)) * 0.01, axis=0)
    px[:, [12, 13, 26, 27]] = np.nan               # substitutes, never on
    px[200:260, 3] = np.nan                        # a gap
    py = np.where(np.isfinite(px), py, np.nan)
    bx = 0.5 + 0.2 * np.sin(np.arange(n) / 50.0)
    by = 0.5 + 0.2 * np.cos(np.arange(n) / 70.0)
    bx[400:480] = np.nan
    by[400:480] = np.nan
    return {"px": px.astype(np.float32), "py": py.astype(np.float32),
            "bx": bx.astype(np.float32), "by": by.astype(np.float32), "team_of_slot": team}


@pytest.fixture
def fake_game(monkeypatch):
    from engine import data
    monkeypatch.setattr(data, "pitch_m", lambda game: (105.0, 68.0))
    monkeypatch.setattr(data, "graded_blocks", lambda game: [(1, 0, 499), (2, 500, 999)])
    return "SYNTH"


def _sorted_tokens(tok):
    """Player tokens of each team block sorted by (x, y), for comparisons up to order."""
    from tools.learned_spike import set_tokens as st
    out = tok.copy()
    for lo in (0, st.PER_TEAM):
        blk = out[:, lo:lo + st.PER_TEAM]
        key = np.round(blk[..., 0], 4) * 1000 + np.round(blk[..., 1], 4)
        order = np.argsort(key, axis=1, kind="stable")
        out[:, lo:lo + st.PER_TEAM] = np.take_along_axis(blk, order[..., None], axis=1)
    return out


def test_swap_and_mirror_are_relabellings(fake_game):
    from tools.learned_spike import set_tokens as st
    tac = _tac()
    tok = st.tokens(tac, fake_game)
    assert tok.shape == (200, st.N_TOKENS, st.N_COLS)
    assert np.isfinite(tok).all()
    swapped = dict(tac, team_of_slot=1 - tac["team_of_slot"])
    mirrored = dict(tac, py=(1.0 - tac["py"]).astype(np.float32),
                    by=(1.0 - tac["by"]).astype(np.float32))
    for other, (sw, mi) in ((swapped, (True, False)), (mirrored, (False, True))):
        got = _sorted_tokens(st.tokens(other, fake_game))
        want = _sorted_tokens(st.transform(tok, sw, mi))
        np.testing.assert_allclose(got, want, atol=2e-5)


def test_transform_is_an_involution(fake_game):
    from tools.learned_spike import set_tokens as st
    tok = st.tokens(_tac(seed=1), fake_game)
    for sw in (False, True):
        for mi in (False, True):
            np.testing.assert_array_equal(st.transform(st.transform(tok, sw, mi), sw, mi), tok)


def test_ball_missing_zeroes_offsets(fake_game):
    from tools.learned_spike import set_tokens as st
    tok = st.tokens(_tac(), fake_game)
    hidden = tok[:, st.BALL, 9] == 1
    assert hidden.any()
    assert (tok[hidden][:, :2 * st.PER_TEAM, 6:9] == 0).all()


@pytest.mark.skipif(not HAVE_TORCH, reason="torch not installed")
def test_masking_and_permutation():
    import torch
    from tools.learned_spike import set_model as sm
    from tools.learned_spike import set_tokens as st
    torch.manual_seed(0)
    net = sm.SetNet().eval()
    x = torch.randn(1, 30, st.N_TOKENS, st.N_COLS)
    x[..., st.PRESENT_COL] = (torch.rand(1, 30, st.N_TOKENS) > 0.3).float()
    x[..., :st.PER_TEAM, st.TEAM_COL] = 1.0
    x[..., st.PER_TEAM:2 * st.PER_TEAM, st.TEAM_COL] = -1.0
    x[..., st.BALL, st.TEAM_COL] = 0.0
    with torch.no_grad():
        base = net(x)
        # absent tokens carry no information
        y = x.clone()
        absent = y[..., st.PRESENT_COL] == 0
        absent[..., st.BALL] = False
        noise = torch.randn_like(y) * 5
        noise[..., st.PRESENT_COL] = 0
        noise[..., st.TEAM_COL] = 0
        y = torch.where(absent[..., None], y + noise, y)
        torch.testing.assert_close(net(y), base, atol=1e-5, rtol=1e-5)
        # the order of the tokens of a step does not matter
        perm = torch.randperm(2 * st.PER_TEAM).tolist() + [st.BALL]
        torch.testing.assert_close(net(x[:, :, perm]), base, atol=1e-5, rtol=1e-5)
        # chunked encoding gives the same logits
        torch.testing.assert_close(net(x, chunk=7), base, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not HAVE_TORCH, reason="torch not installed")
def test_symmetric_prediction_is_equivariant(fake_game):
    import torch
    from tools.learned_spike import set_model as sm
    from tools.learned_spike import set_tokens as st
    torch.manual_seed(1)
    net = sm.SetNet().eval()
    tok = st.tokens(_tac(seed=2), fake_game)
    p = sm.predict_symmetric(net, tok)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)
    ps = sm.predict_symmetric(net, st.transform(tok, True, False))
    np.testing.assert_allclose(ps, p[:, [1, 0, 2]], atol=1e-5)
    pm = sm.predict_symmetric(net, st.transform(tok, False, True))
    np.testing.assert_allclose(pm, p, atol=1e-5)
