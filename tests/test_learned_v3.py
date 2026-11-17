"""Tests of the round 5 feature version "stage-v3" (engine/learned_features.py, the training twin
tools/learned_spike/features.py, engine/learned.py, export.py, build.py and train.py plumbing).

numpy only. The real-match symmetry test runs when LEARNED_TEST_ENGINE_CACHE names an engine.data
npz cache folder that holds <game>_tactical.npz for METRICA-S1 (an external match, never the
competition match).
"""

import dataclasses
import json
import os

import numpy as np
import pytest

from engine import data, learned
from engine import learned_features as lf
from engine.config import Config
from tools.learned_spike import build, export
from tools.learned_spike import features as spike_features

V2, V3 = lf.FEATURE_VERSION, lf.VERSION_V3
PITCH = (105.0, 68.0)


def _dyadic(a, bits=12):
    """Values on a grid of 2^-bits, so that 1 - x and x * pitch are exact in float64."""
    return (np.round(np.asarray(a, dtype=np.float64) * 2 ** bits) / 2 ** bits).astype(np.float32)


def _match(n=2400, seed=11):
    """A synthetic match on a dyadic grid: two blocks with a gap between them where nobody is
    tracked, a stretch where the away team is not tracked, tracklet gaps, a ghost duplicate."""
    rng = np.random.RandomState(seed)
    px = np.clip(0.5 + np.cumsum(rng.normal(0, 0.002, (n, 24)), axis=0)
                 + np.where(np.arange(24) < 12, -0.15, 0.15), 0.01, 0.99)
    py = np.clip(np.linspace(0.05, 0.95, 24) + np.cumsum(rng.normal(0, 0.002, (n, 24)), axis=0),
                 0.01, 0.99)
    px[1180:1220] = np.nan                     # between the blocks
    py[1180:1220] = np.nan
    px[600:700, 12:] = np.nan                  # the away team untracked for 4 s
    py[600:700, 12:] = np.nan
    px[300:310, 3] = np.nan                    # a 10-frame tracklet gap
    py[300:310, 3] = np.nan
    px[1500:1530, 11] = px[1500:1530, 2] + 0.001   # a ghost on top of slot 2 for 30 frames
    py[1500:1530, 11] = py[1500:1530, 2]
    px[:1500, 11] = np.nan
    py[:1500, 11] = np.nan
    px[1530:, 11] = np.nan
    py[1530:, 11] = np.nan
    px[1960:2001, 5] = np.nan                  # slot 5 visible alone at frame 2001: no velocity
    py[1960:2001, 5] = np.nan
    px[2002:2040, 5] = np.nan
    py[2002:2040, 5] = np.nan
    bx = np.clip(0.5 + np.cumsum(rng.normal(0, 0.003, n)), 0.0, 1.0)
    by = np.clip(0.5 + np.cumsum(rng.normal(0, 0.003, n)), 0.0, 1.0)
    bx[700:760] = np.nan
    by[700:760] = np.nan
    team = np.array([0] * 12 + [1] * 12, dtype=np.int8)
    return {"px": _dyadic(px), "py": _dyadic(py), "bx": _dyadic(bx), "by": _dyadic(by),
            "team_of_slot": team, "n_frames": n}


@pytest.fixture
def synth(monkeypatch):
    monkeypatch.setattr(data, "graded_blocks", lambda game=None: [(1, 0, 1179), (2, 1220, 2399)])
    monkeypatch.setattr(data, "pitch_m", lambda game=None: PITCH)
    return _match()


def _swapped(tac):
    team = np.asarray(tac["team_of_slot"])
    return dict(tac, team_of_slot=np.where(team == 0, 1, np.where(team == 1, 0, team)).astype(
        team.dtype))


def _mirrored(tac):
    py, by = np.asarray(tac["py"]), np.asarray(tac["by"])
    return dict(tac, py=(1.0 - py.astype(np.float64)).astype(py.dtype),
                by=(1.0 - by.astype(np.float64)).astype(by.dtype))


# -- versions and tables ---------------------------------------------------------------------

def test_versions_widths_and_tables():
    assert lf.VERSIONS == ("stage50-v2", "stage-v3", "stage50-v2s", "stage50-v2t")
    assert lf.n_features(V2) == lf.NF == 50 and lf.n_features(V3) == 55
    assert lf.SPEC == lf.feature_spec(V2) and lf.feature_spec() == lf.feature_spec(V2)
    for v in lf.VERSIONS:
        # stage50-v2s and -v2t share stage50-v2's width: the width alone answers stage50-v2
        same = v in (lf.VERSION_V2S, lf.VERSION_V2T)
        assert lf.version_of_width(lf.n_features(v)) == (V2 if same else v)
        perm, swap, mirror = lf.symmetry_tables(v)
        assert np.array_equal(perm[perm], np.arange(lf.n_features(v)))
        assert np.array_equal(swap[perm], swap)
        x = np.random.RandomState(0).normal(size=(7, lf.n_features(v))).astype(np.float32)
        for s, m in ((True, False), (False, True), (True, True)):
            assert np.array_equal(lf.transform(lf.transform(x, s, m), s, m), x)
        assert np.array_equal(lf.transform(x, True), lf.transform(x, True, version=v))
    with pytest.raises(ValueError):
        lf.version_of_width(51)
    with pytest.raises(ValueError):
        lf.check_version("stage-v3t")
    names = lf.feature_names(V3)
    for col in ("walk_all", "H_medx", "A_medy", "H_iqrx", "A_iqry", "H_deep2x", "A_walk",
                "H_none"):
        assert col in names
    for col in ("H_cx", "A_sy", "H_lastx"):
        assert col not in names
    assert spike_features.feature_spec(V3) == lf.feature_spec(V3)
    # The v2 tables are those of the round 2 to 4 weights.
    assert np.array_equal(lf.PERM, lf.symmetry_tables(V2)[0])


def test_v2_default_is_unchanged_and_twins_agree(synth, monkeypatch):
    monkeypatch.delenv(spike_features.VERSION_ENV, raising=False)
    eng2 = lf.frame_features(synth, "METRICA-S3")
    assert eng2.shape == (2400, 50)
    assert eng2.tobytes() == lf.frame_features(synth, "METRICA-S3", version=V2).tobytes()
    assert eng2.tobytes() == spike_features.frame_features(synth, "METRICA-S3").tobytes()
    eng3 = lf.frame_features(synth, "METRICA-S3", version=V3)
    spk3 = spike_features.frame_features(synth, "METRICA-S3", version=V3)
    assert eng3.shape == (2400, 55) and eng3.dtype == np.float32
    assert eng3.tobytes() == spk3.tobytes()
    # The training twin takes its default version from the environment, the engine never does.
    monkeypatch.setenv(spike_features.VERSION_ENV, V3)
    assert spike_features.frame_features(synth, "METRICA-S3").tobytes() == eng3.tobytes()
    assert lf.frame_features(synth, "METRICA-S3").tobytes() == eng2.tobytes()
    monkeypatch.setenv(spike_features.VERSION_ENV, "nonsense")
    with pytest.raises(ValueError):
        spike_features.frame_features(synth, "METRICA-S3")


# -- cleaning --------------------------------------------------------------------------------

def _line(n, x0, vx, y0=30.0):
    t = np.arange(n, dtype=np.float64)
    return x0 + vx * t, np.full(n, y0)


def test_bridge_joins_a_slot_jump_after_a_short_gap():
    n = 80
    x, y = _line(n, 10.0, 0.2)
    px = np.full((n, 3), np.nan)
    py = np.full((n, 3), np.nan)
    px[:30, 0], py[:30, 0] = x[:30], y[:30]          # ends at frame 29
    px[40:, 1], py[40:, 1] = x[40:], y[40:]          # starts at frame 40 in another slot
    px[:, 2], py[:, 2] = 80.0, 10.0                  # the other team, far away
    team = np.array([0, 0, 1])
    bx, by, filled = lf.bridge_gaps(px, py, team)
    assert filled[30:40, 0].all() and filled.sum() == 10
    assert np.allclose(bx[30:40, 0], x[30:40]) and np.allclose(by[30:40, 0], 30.0)
    assert np.isnan(bx[30:40, 1]).all()              # nothing moves between slots
    assert np.array_equal(bx[:30], px[:30], equal_nan=True)
    cx, cy = lf.clean_tracks(px, py, team)
    assert np.isfinite(cx[:40, 0]).all()


def test_bridge_refuses_far_starts_other_teams_long_gaps_and_block_edges():
    n = 80
    x, y = _line(n, 10.0, 0.2)
    base_x = np.full((n, 3), np.nan)
    base_y = np.full((n, 3), np.nan)
    base_x[:30, 0], base_y[:30, 0] = x[:30], y[:30]
    # A start 3 m from the prediction is not linked.
    px, py = base_x.copy(), base_y.copy()
    px[40:, 1], py[40:, 1] = x[40:] + 3.0, y[40:]
    assert not lf.bridge_gaps(px, py, np.array([0, 0, 1]))[2].any()
    # A start of the other team is not linked.
    px, py = base_x.copy(), base_y.copy()
    px[40:, 2], py[40:, 2] = x[40:], y[40:]
    assert not lf.bridge_gaps(px, py, np.array([0, 0, 1]))[2].any()
    # A gap longer than 1 s is not bridged.
    px, py = base_x.copy(), base_y.copy()
    px[56:, 1], py[56:, 1] = x[56:], y[56:]
    assert not lf.bridge_gaps(px, py, np.array([0, 0, 1]))[2].any()
    # Never across a block boundary.
    px, py = base_x.copy(), base_y.copy()
    px[40:, 1], py[40:, 1] = x[40:], y[40:]
    assert lf.bridge_gaps(px, py, np.array([0, 0, 1]), [(1, 0, 79)])[2].any()
    assert not lf.bridge_gaps(px, py, np.array([0, 0, 1]), [(1, 0, 35), (2, 36, 79)])[2].any()


def test_bridge_mutual_best_and_constant_velocity():
    n = 80
    px = np.full((n, 4), np.nan)
    py = np.full((n, 4), np.nan)
    t = np.arange(n, dtype=np.float64)
    px[:30, 0], py[:30, 0] = 10.0 + 0.3 * t[:30], 20.0     # moving: prediction 10 + 0.3 * 40
    px[40:, 1], py[40:, 1] = 10.0 + 0.3 * t[40:], 20.0     # on the prediction
    px[40:, 2], py[40:, 2] = 10.0 + 0.3 * t[40:] + 1.0, 20.0   # 1 m off: a worse candidate
    px[:, 3], py[:, 3] = 90.0, 60.0
    bx, by, filled = lf.bridge_gaps(px, py, np.array([0, 0, 0, 1]))
    assert filled[30:40, 0].all() and filled.sum() == 10
    assert np.allclose(bx[30:40, 0], 10.0 + 0.3 * t[30:40])
    # A short tracklet (fewer than 5 frames) predicts with zero velocity: the moving start at
    # 10 + 0.3 * 40 = 22 m is 2.1 m from the end at 19.9 m, so no link.
    px2 = np.full((n, 2), np.nan)
    py2 = np.full((n, 2), np.nan)
    px2[26:30, 0], py2[26:30, 0] = 10.0 + 0.3 * t[26:30], 20.0
    px2[40:, 1], py2[40:, 1] = 10.0 + 0.3 * t[40:], 20.0
    assert not lf.bridge_gaps(px2, py2, np.array([0, 0]))[2].any()


def test_persistent_duplicates_only_and_no_cap():
    n = 100
    px = np.tile(np.linspace(5.0, 95.0, 14), (n, 1))
    py = np.full((n, 14), 30.0)
    team = np.array([0] * 14)                   # 14 visible slots: no cap, all stay
    keep = lf.clean_mask(px, py, team)
    assert keep.all()
    # Slot 9 on top of slot 2 for 30 frames: dropped over that run only.
    px[10:40, 9], py[10:40, 9] = px[10:40, 2] + 0.2, 30.1
    # Slot 7 on top of slot 4 for 24 frames (a duel): left alone.
    px[50:74, 7], py[50:74, 7] = px[50:74, 4] + 0.2, 30.0
    keep = lf.clean_mask(px, py, team)
    assert not keep[10:40, 9].any() and keep[:10, 9].all() and keep[40:, 9].all()
    assert keep[:, 2].all() and keep[:, 4].all() and keep[:, 7].all()
    assert (~keep).sum() == 30
    # The same pair in two teams is not a duplicate.
    keep = lf.clean_mask(px, py, np.array([0] * 9 + [1] * 5))
    assert keep[10:40, 9].all()


def test_undefined_speed_is_left_out_of_the_speed_columns(synth):
    """Slot 5 is visible alone at frame 2001 (isolated): tracks._difference_velocity gives it no
    velocity, so it counts in nvis but not in meanv, still or walk."""
    f = lf.frame_features(synth, "METRICA-S3", version=V3)
    names = lf.feature_names(V3)
    tac = lf._clean_tactical(synth, "METRICA-S3")
    vis = np.isfinite(tac["px"][2001]) & np.isfinite(tac["py"][2001])
    home = vis & (synth["team_of_slot"] == 0)
    assert home[5]
    n_speed = home.sum() - 1
    assert f[2001, names.index("H_nvis")] == np.float32(min(home.sum(), 11) / 11.0)
    still = f[2001, names.index("H_still")] * n_speed
    walk = f[2001, names.index("H_walk")] * n_speed
    assert abs(still - round(still)) < 1e-4 and abs(walk - round(walk)) < 1e-4


# -- equivariance ----------------------------------------------------------------------------

def _check_equivariance(f, fs, fm, version, tol, flip_share=0.0):
    names = lf.feature_names(version)
    for other, swap, mirror in ((fs, True, False), (fm, False, True)):
        want = lf.transform(f, swap, mirror)
        diff = np.abs(other - want)
        for k, name in enumerate(names):
            bad = diff[:, k] > tol
            assert bad.mean() <= flip_share, (name, swap, mirror, int(bad.sum()),
                                              float(diff[:, k].max()))


def test_v3_is_equivariant_on_a_synthetic_match(synth):
    f = lf.frame_features(synth, "METRICA-S3", version=V3)
    fs = lf.frame_features(_swapped(synth), "METRICA-S3", version=V3)
    fm = lf.frame_features(_mirrored(synth), "METRICA-S3", version=V3)
    _check_equivariance(f, fs, fm, V3, 1e-5)
    # The away team untracked for 4 s: none is 1, the median sits at the centre.
    names = lf.feature_names(V3)
    assert (f[620:690, names.index("A_none")] == 1).all()
    assert (f[620:690, names.index("A_medx")] == 0).all()
    # At 5 fps, after the averaging, the same holds.
    _check_equivariance(lf.downsample(f), lf.downsample(fs), lf.downsample(fm), V3, 1e-5)


def test_v2_fallback_of_cx_is_not_swap_equivariant(synth):
    """The root cause of the r4-adopt symmetry failure (SWAP_DIAG.md): v2 puts cx at -1 for a
    team with no visible player, where a swap needs +1. v3 replaces cx by medx at the centre."""
    f = lf.frame_features(synth, "METRICA-S3")
    fs = lf.frame_features(_swapped(synth), "METRICA-S3")
    cols = [lf.feature_names(V2).index(c) for c in ("H_cx", "A_cx")]
    diff = np.abs(fs - lf.transform(f, True))[:, cols]
    # 40 frames between the blocks (both teams), 100 frames with the away team untracked.
    assert diff.max() == pytest.approx(2.0) and (diff > 1).sum() == 180


REAL_CACHE = os.environ.get("LEARNED_TEST_ENGINE_CACHE")
REAL_GAME = "METRICA-S1"


@pytest.mark.skipif(not REAL_CACHE or not os.path.isfile(
    os.path.join(REAL_CACHE or "", REAL_GAME + "_tactical.npz")),
    reason="LEARNED_TEST_ENGINE_CACHE with the METRICA-S1 engine cache is not set")
def test_v3_is_equivariant_on_a_slice_of_a_real_match(monkeypatch):
    with np.load(os.path.join(REAL_CACHE, REAL_GAME + "_tactical.npz")) as npz:
        full = {k: npz[k] for k in ("px", "py", "bx", "by", "team_of_slot")}
    blocks = data.graded_blocks(REAL_GAME)
    pitch = data.pitch_m(REAL_GAME)
    first = blocks[0][1] + 3000
    lo, hi = first, first + 15000
    tac = {k: (v[lo:hi] if k != "team_of_slot" else v) for k, v in full.items()}
    tac["n_frames"] = hi - lo
    monkeypatch.setattr(data, "graded_blocks", lambda game=None: [(1, 0, hi - lo - 1)])
    monkeypatch.setattr(data, "pitch_m", lambda game=None: pitch)
    f = lf.frame_features(tac, REAL_GAME, version=V3)
    fs = lf.frame_features(_swapped(tac), REAL_GAME, version=V3)
    fm = lf.frame_features(_mirrored(tac), REAL_GAME, version=V3)
    # Real coordinates are not dyadic: 1 - x and the flip of the pitch round differently, so
    # values agree to float32 noise and a threshold (still, walk, radius counts) can flip on a
    # rare frame.
    _check_equivariance(f, fs, fm, V3, 1e-3, flip_share=0.005)


# -- weights and inference -------------------------------------------------------------------

def _random_state(n_in, hidden=8, seed=0):
    rng = np.random.RandomState(seed)
    shapes = {"inp.0.weight": (hidden, n_in), "inp.0.bias": (hidden,),
              "conv.weight": (hidden, hidden, 5), "conv.bias": (hidden,),
              "head.weight": (3, 2 * hidden), "head.bias": (3,)}
    for layer in range(2):
        width = hidden if layer == 0 else 2 * hidden
        for suffix in ("", "_reverse"):
            tail = "_l{}{}".format(layer, suffix)
            shapes["gru.weight_ih" + tail] = (3 * hidden, width)
            shapes["gru.weight_hh" + tail] = (3 * hidden, hidden)
            shapes["gru.bias_ih" + tail] = (3 * hidden,)
            shapes["gru.bias_hh" + tail] = (3 * hidden,)
    return {k: rng.normal(0, 0.3, s) for k, s in shapes.items()}


def test_weights_record_the_version_and_the_engine_follows_it(synth, tmp_path, monkeypatch):
    path = str(tmp_path / "w3.npz")
    states = [_random_state(55, seed=1), _random_state(55, seed=2)]
    export.write(path, states)
    with np.load(path) as npz:
        assert str(npz["feature_version"]) == V3
    nets = learned.load_weights(path)
    assert learned.model_version(nets) == V3
    with pytest.raises(ValueError):
        export.arrays_of(states, V2)
    with pytest.raises(ValueError):
        export.arrays_of([_random_state(55), _random_state(50)])
    # A file that claims v2 with v3 shapes is refused.
    arrays = export.arrays_of(states)
    arrays["feature_version"] = np.str_(V2)
    bad = str(tmp_path / "bad.npz")
    np.savez(bad, **arrays)
    with pytest.raises(ValueError):
        learned.load_weights(bad)
    arrays["feature_version"] = np.str_("stage-v3t")
    np.savez(bad, **arrays)
    with pytest.raises(ValueError):
        learned.load_weights(bad)
    # frame_probabilities computes the v3 features.
    p5, frames = learned.frame_probabilities(nets, synth, "METRICA-S3")
    want = learned.bin_probabilities(
        nets, lf.downsample(lf.frame_features(synth, "METRICA-S3", version=V3)))
    assert np.array_equal(p5, want) and frames.shape == (2400, 3)
    # Symmetric prediction: a swapped input swaps H and A.
    x = lf.downsample(lf.frame_features(synth, "METRICA-S3", version=V3))
    p = learned.predict_ensemble(nets, x)
    assert np.abs(learned.predict_ensemble(nets, lf.transform(x, True)) - p[:, [1, 0, 2]]).max() \
        < 1e-12
    # The stage end to end.
    monkeypatch.setattr(data, "load_tactical", lambda game=None: synth)
    cfg = dataclasses.replace(Config.default(), LEARNED_MODE="model", LEARNED_WEIGHTS=path,
                              LEARNED_STRUCTURE=False, LEARNED_PRIOR_BONUS=0.0)
    monkeypatch.setattr(learned, "weights_path", lambda c: c.LEARNED_WEIGHTS)
    labels = learned.apply(cfg, "METRICA-S3", data.graded_blocks())
    assert labels.shape == (2400,) and set(labels[1180:1220]) == {"X"}


# -- caches: build.py markers and train.py ---------------------------------------------------

def _write_cache(folder, version, game="METRICA-S1", bins=20):
    os.makedirs(os.path.join(folder, "feats", "raw"), exist_ok=True)
    os.makedirs(os.path.join(folder, "labels"), exist_ok=True)
    nf = lf.n_features(version)
    np.savez(os.path.join(folder, "feats", "raw", game + ".npz"),
             F=np.arange(bins * nf, dtype=np.float32).reshape(bins, nf),
             n_frames=np.int64(bins * 5), feature_version=np.str_(version))
    np.savez(os.path.join(folder, "labels", game + ".npz"), raw=np.zeros(bins * 5, np.int64),
             ctl2=np.zeros(bins * 5, np.int64))


def test_build_marker(tmp_path):
    folder = str(tmp_path / "c3")
    rec = build.write_marker(folder, V3)
    assert rec["columns"] == lf.feature_names(V3)
    assert build.write_marker(folder, V3) == rec
    with pytest.raises(SystemExit):
        build.write_marker(folder, V2)
    _write_cache(str(tmp_path / "old"), V2)
    os.remove(str(tmp_path / "old" / "feats" / "raw" / "METRICA-S1.npz"))
    old = str(tmp_path / "old")
    np.savez(os.path.join(old, "feats", "raw", "METRICA-S1.npz"), F=np.zeros((4, 50), np.float32),
             n_frames=np.int64(20))
    with pytest.raises(SystemExit):
        build.write_marker(old, V3)       # unmarked files: a stage50-v2 folder
    with pytest.raises(SystemExit):
        build.mark(old, V3)               # wrong width
    assert build.mark(old, V2)[1] == 1
    with open(build.marker_path(old)) as handle:
        assert json.load(handle)["feature_version"] == V2


def test_train_reads_only_its_version(tmp_path):
    train = pytest.importorskip("tools.learned_spike.train")
    old_version = train.FEATURE_VERSION
    try:
        c3 = str(tmp_path / "c3")
        build.write_marker(c3, V3)
        _write_cache(c3, V3)
        c2 = str(tmp_path / "c2")
        _write_cache(c2, V2)               # unmarked: stage50-v2
        assert train.folder_version(c2) == V2 and train.folder_version(c3) == V3
        train.configure_features(V3)
        assert train.n_inputs() == 55 and train.folder_columns(c3) is None
        with pytest.raises(SystemExit):
            train.folder_columns(c2)
        got = train.load_games(c3, ["METRICA-S1"], raw_only=True)
        assert got["METRICA-S1"]["feats"]["raw"].shape == (20, 55)
        with pytest.raises(SystemExit):
            train.load_games(c2, ["METRICA-S1"], raw_only=True)
        train.configure_features(None)
        with pytest.raises(SystemExit):
            train.load_games(c3, ["METRICA-S1"], raw_only=True)
        assert train.load_games(c2, ["METRICA-S1"], raw_only=True)["METRICA-S1"]["feats"][
            "raw"].shape == (20, 50)
        # Column selection by name (generic: a folder whose marker lists more columns).
        view = train.ColumnView(np.arange(12.0).reshape(3, 4), [3, 1])
        assert view.shape == (3, 2) and np.array_equal(view[1:3], [[7.0, 5.0], [11.0, 9.0]])
        assert np.array_equal(np.asarray(view), np.arange(12.0).reshape(3, 4)[:, [3, 1]])
    finally:
        train.configure_features(old_version)
