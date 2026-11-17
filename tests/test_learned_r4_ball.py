"""Round 4 ball smoothing and offset (augment.py component 8; docs/plan/calibration/R4_AUG_BALL.md)."""

import importlib.util
import os
import subprocess

import numpy as np
import pytest

from tools.learned_spike import augment as au, folds

PITCH = (105.0, 68.0)
N = 3000
BLOCKS = [(1, 0, N - 1)]


def _tac(seed=0, n=N):
    """A synthetic match (as tests/test_learned_r3_aug.py, which needs torch at import)."""
    rng = np.random.RandomState(seed)
    px = np.clip(0.5 + np.cumsum(rng.normal(0, 0.002, (n, 22)), axis=0)
                 + rng.uniform(-0.3, 0.3, 22), 0.01, 0.99).astype(np.float32)
    py = np.clip(0.5 + np.cumsum(rng.normal(0, 0.002, (n, 22)), axis=0)
                 + rng.uniform(-0.3, 0.3, 22), 0.01, 0.99).astype(np.float32)
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
    c[1000:1250] = 2
    return c

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUND3_COMMIT = "2e85b69"      # augment.py before component 8
ON = {"sigma": 3.0, "offset_m": 1.0, "tau_s": 2.0}


def _round3_module(tmp_path):
    """augment.py as it was before component 8 (git show), or skip outside a checkout."""
    try:
        src = subprocess.run(["git", "show", ROUND3_COMMIT + ":tools/learned_spike/augment.py"],
                             cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history not available")
    path = tmp_path / "augment_r3.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location("augment_r3", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _same(a, b):
    assert set(a) == set(b)
    for key in a:
        x, y = a[key], b[key]
        if isinstance(x, np.ndarray):
            assert x.dtype == y.dtype and x.shape == y.shape
            assert x.tobytes() == y.tobytes(), key
        else:
            assert x == y, key


@pytest.mark.parametrize("game", ["METRICA-S1", "DFL-J03WMX", "SC-1874553"])
def test_switch_off_is_byte_identical_to_round3(tmp_path, game):
    r3 = _round3_module(tmp_path)
    tac, codes = _tac(3), _codes()
    for k in range(16):
        new, rep_new = au.augment(tac, codes, game, k, BLOCKS, PITCH)
        old, rep_old = r3.augment(tac, codes, game, k, BLOCKS, PITCH)
        _same(new, old)
        assert rep_new == rep_old
        off, _ = au.augment(tac, codes, game, k, BLOCKS, PITCH, ball=False)
        _same(off, old)


def test_switch_on_changes_ball_only_and_only_metrica_dfl():
    tac, codes = _tac(4), _codes()
    changed = 0
    for game in ("METRICA-S2", "DFL-J03WOH", "SC-1886347"):
        for k in range(16):
            off, _ = au.augment(tac, codes, game, k, BLOCKS, PITCH)
            on, rep = au.augment(tac, codes, game, k, BLOCKS, PITCH, ball=True)
            for key in ("px", "py", "ps", "team_of_slot"):
                assert off[key].tobytes() == on[key].tobytes()
            # the visibility of the ball is unchanged
            assert np.array_equal(np.isfinite(off["bx"]), np.isfinite(on["bx"]))
            if game.startswith("SC-"):
                assert off["bx"].tobytes() == on["bx"].tobytes()
                assert "ball" not in rep
            elif rep["plan"]["ball"] is not None:
                assert not np.array_equal(off["bx"], on["bx"], equal_nan=True)
                changed += 1
            else:
                assert off["bx"].tobytes() == on["bx"].tobytes()
    assert changed > 0


def test_plan_ball_ranges_rate_and_determinism():
    plans = [au.plan_ball(g, k) for g in folds.ALL for k in range(16)]
    on = [p for p in plans if p is not None]
    assert 0.65 < len(on) / len(plans) < 0.85            # P_ON = 0.75
    for p in on:
        assert au.BALL_SIGMA[0] <= p["sigma"] <= au.BALL_SIGMA[1]
        assert au.BALL_OFFSET_M[0] <= p["offset_m"] <= au.BALL_OFFSET_M[1]
        assert au.BALL_TAU_S[0] <= p["tau_s"] <= au.BALL_TAU_S[1]
    assert plans == [au.plan_ball(g, k) for g in folds.ALL for k in range(16)]
    assert au.plan("DFL-J03WMX", 3) == au.plan("DFL-J03WMX", 3)


def test_ball_seeds_differ_from_stress_view():
    n = au.check_seeds(folds.ALL, 16, ball=True)
    assert n == len(folds.ALL) * 16 * (len(au.COMPONENTS) + 1 + len(au.BALL_SEEDS))


def test_smoothing_preserves_visibility_gaps_and_block_edges():
    n = 400
    x = np.zeros(n)
    x[100:] = 10.0                       # a step inside a run
    vis = np.ones(n, dtype=bool)
    vis[200:230] = False                 # a gap
    x[~vis] = np.nan
    br = au.run_breaks([(1, 0, 299), (2, 300, n - 1)], n)
    x[300:] = 50.0                       # a jump across the block edge
    s = au.smooth_runs(x, vis, br, 4.0)
    assert np.array_equal(np.isfinite(s), vis)
    assert np.allclose(s[300:], 50.0)    # the block edge is not bridged
    assert np.allclose(s[230:300], 10.0)  # the gap is not bridged
    assert 0 < s[100] < 10 and 0 < s[99] < 10   # the step is spread
    # agrees with the round 3 view code (tools/atd_view.smooth_runs)
    import importlib
    atd_view = importlib.import_module("tools.atd_view")
    assert np.allclose(s, atd_view.smooth_runs(x, vis, br, 4.0), equal_nan=True)


def test_component_keeps_gaps_and_bounds_offset():
    tac, codes = _tac(5), _codes()
    for rates in (ON, {"sigma": 0.0, "offset_m": 1.5, "tau_s": 1.0},
                  {"sigma": 4.0, "offset_m": 0.0, "tau_s": 5.0}):
        base, _ = au.augment(tac, codes, "DFL-J03WMX", 0, BLOCKS, PITCH,
                             only={"live": {"share": 0.2, "p_crowd": 0.5}})
        out, rep = au.augment(tac, codes, "DFL-J03WMX", 0, BLOCKS, PITCH,
                              only={"live": {"share": 0.2, "p_crowd": 0.5}, "ball": rates},
                              ball=True)
        vis = np.isfinite(base["bx"])
        assert not vis.all() and vis.any()
        assert np.array_equal(np.isfinite(out["bx"]), vis)
        assert np.array_equal(np.isfinite(out["by"]), vis)
        assert rep["ball"]["offset_max_m"] <= au.BALL_OFFSET_CAP * rates["offset_m"] + 1e-6
    # offset alone: bounded, slowly varying, RMS near the target
    rng = np.random.RandomState(0)
    ox, oy = au.slow_offset(200000, rng, 1.0, 2.0 * au.FPS)
    norm = np.hypot(ox, oy)
    assert norm.max() <= au.BALL_OFFSET_CAP * 1.0 + 1e-9
    assert 0.85 < np.sqrt((norm ** 2).mean()) < 1.1
    assert np.abs(np.diff(ox)).max() < 0.1          # metres per frame: slow
    assert np.array_equal(au.slow_offset(10, rng, 0.0, 50.0)[0], np.zeros(10))


def test_component_deterministic_by_seed_and_input_untouched():
    tac, codes = _tac(6), _codes()
    keep = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in tac.items()}
    a, _ = au.augment(tac, codes, "METRICA-S3", 0, BLOCKS, PITCH, only={"ball": ON}, ball=True)
    b, _ = au.augment(tac, codes, "METRICA-S3", 0, BLOCKS, PITCH, only={"ball": ON}, ball=True)
    c, _ = au.augment(tac, codes, "METRICA-S3", 1, BLOCKS, PITCH, only={"ball": ON}, ball=True)
    assert a["bx"].tobytes() == b["bx"].tobytes() and a["by"].tobytes() == b["by"].tobytes()
    assert a["bx"].tobytes() != c["bx"].tobytes()     # another copy, another offset draw
    for k, v in keep.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(tac[k], v, equal_nan=True)
    # sigma only (no offset): independent of the seed
    s0, _ = au.augment(tac, codes, "METRICA-S3", 0, BLOCKS, PITCH,
                       only={"ball": dict(ON, offset_m=0.0)}, ball=True)
    s1, _ = au.augment(tac, codes, "METRICA-S3", 1, BLOCKS, PITCH,
                       only={"ball": dict(ON, offset_m=0.0)}, ball=True)
    assert s0["bx"].tobytes() == s1["bx"].tobytes()


def test_smoothing_lowers_kick_sharpness():
    """A velocity step: the 3-frame speed rise over the 9-frame one drops with sigma."""
    n = 600
    x = np.where(np.arange(n) < 300, 0.0, (np.arange(n) - 300) * 0.8)   # 0 -> 20 m/s
    vis = np.ones(n, dtype=bool)
    br = au.run_breaks([(1, 0, n - 1)], n)

    def ratio(sig):
        v = np.gradient(au.smooth_runs(x, vis, br, sig)) * au.FPS
        dv3 = (v[3:] - v[:-3]).max()
        dv9 = (v[9:] - v[:-9]).max()
        return dv3 / dv9
    r = [ratio(s) for s in (0.0, 2.0, 4.0)]
    assert r[0] > 0.95 and r[0] > r[1] > r[2]
