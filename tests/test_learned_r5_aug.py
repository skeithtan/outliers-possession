"""Round 5 inputs: switch X (augment.py version "x") and step J (r5_inputs.py, dropped, kept off)."""

import importlib.util
import os
import subprocess
import types

import numpy as np
import pytest

from tools.learned_spike import augment as au, folds, r5_inputs as r5

PITCH = (105.0, 68.0)
N = 6000
BLOCKS = [(1, 0, N - 1)]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUND4_COMMIT = "66154c8"      # augment.py before switch X


def _tac(seed=0, n=N):
    rng = np.random.RandomState(seed)
    px = np.clip(0.5 + np.cumsum(rng.normal(0, 0.002, (n, 22)), axis=0)
                 + rng.uniform(-0.3, 0.3, 22), 0.01, 0.99).astype(np.float32)
    py = np.clip(0.5 + np.cumsum(rng.normal(0, 0.002, (n, 22)), axis=0)
                 + rng.uniform(-0.3, 0.3, 22), 0.01, 0.99).astype(np.float32)
    bx = px[:, 0] + 0.003
    by = py[:, 0].copy()
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    return {"px": px, "py": py, "ps": np.ones_like(px), "bx": bx.astype(np.float32),
            "by": by.astype(np.float32), "team_of_slot": team, "n_slots": 22, "n_frames": n}


def _codes(n=N):
    c = np.zeros(n, dtype=np.int64)
    c[n // 2:] = 1
    c[1000:1250] = 2
    return c


def _old_module(tmp_path):
    try:
        src = subprocess.run(["git", "show", ROUND4_COMMIT + ":tools/learned_spike/augment.py"],
                             cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history not available")
    path = tmp_path / "augment_r4.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location("augment_r4", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _same(a, b):
    assert set(a) == set(b)
    for key in a:
        x, y = a[key], b[key]
        if isinstance(x, np.ndarray):
            assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), key
        else:
            assert x == y, key


@pytest.mark.parametrize("game", ["METRICA-S1", "DFL-J03WMX", "SC-1874553"])
def test_version_r3_is_byte_identical_to_round4(tmp_path, game):
    old = _old_module(tmp_path)
    tac, codes = _tac(3), _codes()
    for k in range(16):
        for ball in (False, True):
            new, rn = au.augment(tac, codes, game, k, BLOCKS, PITCH, ball=ball)
            ref, ro = old.augment(tac, codes, game, k, BLOCKS, PITCH, ball=ball)
            _same(new, ref)
            assert rn == ro


def test_plan_x_keeps_every_draw():
    for g in folds.ALL[:6]:
        for k in range(16):
            assert au.plan(g, k) == au.plan(g, k, "x")
    assert au.GHOST_PER_10MIN_X == au.GHOST_PER_10MIN and au.GHOST_S_X == au.GHOST_S
    assert au.GHOST_OFFSET_M_X == (0.1, 1.5)
    with pytest.raises(ValueError):
        au.plan("METRICA-S1", 0, "y")


def test_check_seeds_with_flicker():
    n = au.check_seeds(folds.ALL, 16, ball=True, flicker=True)
    assert n == len(folds.ALL) * 16 * (len(au.COMPONENTS) + 1 + 2 + 2)
    for g in folds.ALL:
        bad = au.stress_seeds(g)
        for k in range(16):
            old = {au.seed_of(g, k, c) for c in au.COMPONENTS + ("plan",) + au.BALL_SEEDS}
            for c in au.FLICKER_SEEDS:
                s = au.seed_of(g, k, c)
                assert s not in bad and s not in old


def test_ghosts_x_offsets():
    tac, codes = _tac(1), _codes()
    out, rep = au.augment(tac, codes, "METRICA-S1", 0, BLOCKS, PITCH,
                          only={"ghost": {"per_10min": 20.0}}, version="x")
    assert rep["ghost"] > 3
    px, py, team = out["px"], out["py"], out["team_of_slot"]
    far = 0.0
    for col in range(22, px.shape[1]):
        rows = np.flatnonzero(np.isfinite(px[:, col]))
        if rows.size == 0:
            continue
        mates = np.flatnonzero(team[:22] == team[col])
        d = np.hypot((px[rows][:, mates] - px[rows, col][:, None]) * PITCH[0],
                     (py[rows][:, mates] - py[rows, col][:, None]) * PITCH[1])
        near = np.nanmin(d, axis=1)
        assert near.max() <= 1.5 + 1e-3
        far = max(far, near.max())
    assert far > 0.1


def test_flicker_breaks_and_reslots():
    tac, codes = _tac(2), _codes()
    rate = 6.0
    out, rep = au.augment(tac, codes, "DFL-J03WMX", 0, BLOCKS, PITCH,
                          only={"flicker": {"per_slot_min": rate}}, version="x")
    f = rep["flicker"]
    minutes = N / 25.0 / 60.0
    assert 0.5 * rate * minutes * 22 < f["gaps"] < 1.5 * rate * minutes * 22
    assert f["pieces"] > 0 and f["moved"] > 0
    px = out["px"]
    # positions are kept: each frame's set of visible positions per team is a subset of the input
    team = out["team_of_slot"]
    for t in (0, 3000, 5999):
        for side in (0, 1):
            got = np.sort(px[t, (team == side) & np.isfinite(px[t])])
            src = np.sort(tac["px"][t, tac["team_of_slot"] == side])
            assert np.all(np.isin(got, src))
    # gap lengths: hidden frames per gap between 0.2 s and 2 s (a gap may overlap another)
    hidden = np.isfinite(tac["px"]).sum() - np.isfinite(px).sum()
    assert 0.2 * 25 * f["gaps"] * 0.5 < hidden < 2.0 * 25 * f["gaps"]
    # off by default in version x when the plan says so, and absent in version r3
    _, r3 = au.augment(tac, codes, "DFL-J03WMX", 0, BLOCKS, PITCH,
                       only={"flicker": {"per_slot_min": rate}})
    assert "flicker" not in r3


def test_version_x_deterministic_and_input_untouched():
    tac, codes = _tac(4), _codes()
    keep = {k: np.array(v, copy=True) for k, v in tac.items() if isinstance(v, np.ndarray)}
    a, ra = au.augment(tac, codes, "SC-1874553", 5, BLOCKS, PITCH, version="x")
    b, rb = au.augment(tac, codes, "SC-1874553", 5, BLOCKS, PITCH, version="x")
    _same(a, b)
    assert ra == rb and ra["plan"]["version"] == "x" and "flicker" in ra["plan"]
    for k, v in keep.items():
        assert v.tobytes() == np.asarray(tac[k]).tobytes()


# -- step J (dropped from round 5; code kept, no cache uses it) --------------------------------

def _dfl_tac(n=4000):
    tac = _tac(5, n)
    bx = tac["bx"].copy()
    by = tac["by"].copy()
    bx[500:900] = bx[500]
    by[500:900] = by[500]           # still run of 400 frames
    bx[2000:2005] = bx[2000]
    by[2000:2005] = by[2000]        # too short to be still
    bx[3000:3030] = bx[3000] + np.arange(30) * 0.5 / 25 / PITCH[0]   # 0.5 m/s: slow
    by[3000:3030] = by[3000]
    return dict(tac, bx=bx, by=by)


def test_jitter_eval_keeps_every_still_run_and_is_measured_scale():
    tac = _dfl_tac()
    bx, by, still, hidden, rep = r5.apply_j(tac, "DFL-J03WMX", PITCH, "eval")
    assert rep["still_runs"] >= 1 and rep["still_kept"] == rep["still_runs"]
    assert not hidden.any() and np.isfinite(bx[still]).all()
    d = np.hypot((bx[500:900] - tac["bx"][500:900]) * PITCH[0],
                 (by[500:900] - tac["by"][500:900]) * PITCH[1])
    assert d.max() < r5.JITTER_EXCURSION_M[1] + 6 * r5.JITTER_SIGMA_M[1] * np.sqrt(2)
    assert np.all(np.diff(bx[500:900]) != 0)
    assert (bx[:500] == tac["bx"][:500]).all()


def test_jitter_train_deterministic_and_hidden_runs_extended():
    tac = _dfl_tac()
    a, ra = r5.load_dfl(tac, "DFL-J03WMX", "c_train_j", PITCH)
    b, rb = r5.load_dfl(tac, "DFL-J03WMX", "c_train_j", PITCH)
    assert a["bx"].tobytes() == b["bx"].tobytes() and ra == rb
    assert ra["still_runs"] == ra["still_kept"] + (ra["still_runs"] - ra["still_kept"])
    other, _ = r5.load_dfl(tac, "METRICA-S1", "c_train_j", PITCH)
    assert other is tac
    with pytest.raises(SystemExit):
        r5.load_dfl(tac, "DFL-J03WQQ", "c_train_j", PITCH)


def test_slow_mask_label_free_definition():
    tac = _dfl_tac()
    still = r5.still_ball.still_mask(tac["bx"], tac["by"])
    slow = r5.slow_mask(tac["bx"], tac["by"], PITCH, still)
    assert slow[3001:3030].all() and not slow[still].any()


def test_frame_features_passes_version(monkeypatch):
    calls = []
    fake = types.SimpleNamespace(frame_features=lambda tac, game, feature_version=None:
                                 calls.append(feature_version) or np.zeros((3, 2)))
    import tools.learned_spike as pkg
    monkeypatch.setattr(pkg, "features", fake, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "tools.learned_spike.features", fake)
    r5.frame_features({}, "METRICA-S1", "stage-v3t")
    r5.frame_features({}, "METRICA-S1", None)
    assert calls == ["stage-v3t", None]
    monkeypatch.setenv("FEATURE_VERSION", "stage-v3")
    assert r5.feature_version(None) == "stage-v3"
    monkeypatch.setenv("FEATURE_VERSION", "")
    assert r5.feature_version(None) is None


def test_defect_eval_seeds_disjoint():
    assert r5.check_defect_eval_seeds() == 8 * 2 * (len(au.COMPONENTS) + 1)
    assert set(r5.DEFECT_EVAL_GAMES) == set(folds.OPEN) and min(r5.DEFECT_EVAL_K) >= 16
    with pytest.raises(AssertionError):
        r5.check_defect_eval_seeds(ks=(3,))


def test_launcher_letters():
    path = os.path.join(ROOT, "tools", "learned_spike", "r5_screen_launch.sh")
    for recipe in ("AQX", "AQCFX", "ABQC", "AQCZ", "AQCJ", "AQCXJ", "AQCL", "AQCSLJ", "AQT", "AQCST", "AQCTX"):
        out = subprocess.run(["bash", path, recipe, "0"], capture_output=True, text=True,
                             env={"PATH": os.environ["PATH"]})
        assert out.returncode == 2, (recipe, out.stdout, out.stderr)


def test_train_accepts_j_folders_together_only(tmp_path):
    import json as _json
    pytest.importorskip("torch")
    from tools.learned_spike import train

    def folder(name, inp):
        d = tmp_path / name
        d.mkdir()
        (d / train.STILL_MARKER).write_text(_json.dumps({"dfl_still": "removed",
                                                         "dfl_input": inp}))
        return str(d)
    saved = (train.DFL_STILL, train.AUG_CACHE, train.VAL_CACHE, train.BALLFREE_CACHE)
    try:
        val = folder("v", "c")
        for cache_in, aug_in, ok in (("c_train", "c_train", True),
                                     ("c_train_j", "c_train_j", True),
                                     ("c_train_j", "c_train", False),
                                     ("c_train", "j_eval", False)):
            train.DFL_STILL, train.VAL_CACHE, train.BALLFREE_CACHE = "removed", val, None
            train.AUG_CACHE = folder("a" + cache_in + aug_in + str(ok), aug_in)
            cache = folder("c" + cache_in + aug_in + str(ok), cache_in)
            if ok:
                assert train.check_still(cache) == "removed"
            else:
                with pytest.raises(SystemExit):
                    train.check_still(cache)
    finally:
        train.DFL_STILL, train.AUG_CACHE, train.VAL_CACHE, train.BALLFREE_CACHE = saved


def test_keep_all_is_superset_and_half_unchanged(tmp_path):
    tac = _dfl_tac()
    a, ra = r5.load_dfl(tac, "DFL-J03WMX", "c_train_j", PITCH)
    b, rb = r5.load_dfl(tac, "DFL-J03WMX", "c_train_j100", PITCH)
    assert rb["still_kept"] == rb["still_runs"] and ra["p_keep"] == 0.5 and rb["p_keep"] == 1.0
    # before the gap steps: the runs kept at 0.5 carry the same jitter at 1.0
    xa, ya, still, hid_a, _ = r5.apply_j(tac, "DFL-J03WMX", PITCH, "train", 0.5)
    xb, yb, _, hid_b, _ = r5.apply_j(tac, "DFL-J03WMX", PITCH, "train", 1.0)
    kept_a = still & ~hid_a
    assert not hid_b.any() and kept_a.any()
    assert xa[kept_a].tobytes() == xb[kept_a].tobytes()
    # c_train_j at 0.5 is byte-identical to the module before the keep option (343f714)
    try:
        src = subprocess.run(["git", "show", "343f714:tools/learned_spike/r5_inputs.py"],
                             cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history not available")
    path = tmp_path / "r5_inputs_old.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location("r5_inputs_old", str(path))
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    o, _ = old.load_dfl(tac, "DFL-J03WMX", "c_train_j", PITCH)
    assert o["bx"].tobytes() == a["bx"].tobytes() and o["by"].tobytes() == a["by"].tobytes()
