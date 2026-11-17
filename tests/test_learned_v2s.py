"""Round 6 feature version "stage50-v2s": stage50-v2 with the team-swap faults removed.

numpy only. The SkillCorner test runs when LEARNED_TEST_ENGINE_CACHE names an engine.data npz
cache folder holding SC-1874553_tactical.npz (an external match).
"""

import json
import os

import numpy as np
import pytest

from engine import data, learned
from engine import learned_features as lf
from tests.test_learned_v3 import (_check_equivariance, _match, _mirrored, _random_state,
                                   _swapped, PITCH)
from tools.learned_spike import build, export
from tools.learned_spike import features as spike_features

V2, V2S = lf.FEATURE_VERSION, lf.VERSION_V2S


@pytest.fixture
def synth(monkeypatch):
    monkeypatch.setattr(data, "graded_blocks", lambda game=None: [(1, 0, 1179), (2, 1220, 2399)])
    monkeypatch.setattr(data, "pitch_m", lambda game=None: PITCH)
    return _match()


def test_v2s_has_the_columns_and_tables_of_v2():
    assert V2S in lf.VERSIONS and lf.feature_spec(V2S) == lf.feature_spec(V2)
    for a, b in zip(lf.symmetry_tables(V2S), lf.symmetry_tables(V2)):
        assert np.array_equal(a, b)
    assert lf.version_of_width(50) == V2          # the width alone means stage50-v2


def test_v2s_is_equivariant_where_v2_is_not(synth):
    f = lf.frame_features(synth, "METRICA-S3", version=V2S)
    fs = lf.frame_features(_swapped(synth), "METRICA-S3", version=V2S)
    fm = lf.frame_features(_mirrored(synth), "METRICA-S3", version=V2S)
    _check_equivariance(f, fs, fm, V2S, 1e-5)
    _check_equivariance(lf.downsample(f), lf.downsample(fs), lf.downsample(fm), V2S, 1e-5)
    # the same synthetic match breaks stage50-v2 (frames with a team out of view)
    g = lf.frame_features(synth, "METRICA-S3")
    gs = lf.frame_features(_swapped(synth), "METRICA-S3")
    assert np.abs(gs - lf.transform(g, True)).max() == pytest.approx(2.0)


def test_v2s_equals_v2_wherever_both_teams_are_in_view(synth):
    f2 = lf.frame_features(synth, "METRICA-S3")
    f2s = lf.frame_features(synth, "METRICA-S3", version=V2S)
    vis = np.isfinite(synth["px"]) & np.isfinite(synth["py"])
    team = synth["team_of_slot"]
    none = (vis[:, team == 0].sum(1) == 0) | (vis[:, team == 1].sum(1) == 0)
    diff = (f2 != f2s).any(axis=1)
    assert diff.any() and not (diff & ~none).any()
    names = lf.feature_names()
    changed = sorted(set(np.array(names)[(f2 != f2s).any(axis=0)]))
    assert changed == ["A_cx", "A_cy", "H_cx", "H_cy"]
    a_none = vis[:, team == 1].sum(1) == 0
    assert (f2s[a_none, names.index("A_cx")] == 0).all()
    assert (f2[a_none, names.index("A_cx")] == -1).all()
    # the twins agree
    assert f2s.tobytes() == spike_features.frame_features(synth, "METRICA-S3",
                                                           version=V2S).tobytes()


def test_direction_of_a_block_without_home_players(monkeypatch):
    monkeypatch.setattr(data, "graded_blocks", lambda game=None: [(1, 0, 99)])
    px = np.full((100, 4), np.nan)
    px[:, 2:] = 0.8                                  # only the away team, in the right half
    tac = {"px": px, "team_of_slot": np.array([0, 0, 1, 1])}
    blocks = data.graded_blocks()
    assert lf.home_attacks_positive(tac, blocks, 105.0) == {1: False}           # v2: 0.5
    assert lf.home_attacks_positive(tac, blocks, 105.0, away_fallback=True) == {1: True}
    sw = dict(tac, team_of_slot=np.array([1, 1, 0, 0]))
    assert lf.home_attacks_positive(sw, blocks, 105.0, away_fallback=True) == {1: False}


REAL_CACHE = os.environ.get("LEARNED_TEST_ENGINE_CACHE")
SC_GAME = "SC-1874553"


@pytest.mark.skipif(not REAL_CACHE or not os.path.isfile(
    os.path.join(REAL_CACHE or "", SC_GAME + "_tactical.npz")),
    reason="LEARNED_TEST_ENGINE_CACHE with the SC-1874553 engine cache is not set")
def test_v2s_is_equivariant_on_a_skillcorner_slice_with_a_team_out_of_view(monkeypatch):
    with np.load(os.path.join(REAL_CACHE, SC_GAME + "_tactical.npz")) as npz:
        full = {k: npz[k] for k in ("px", "py", "bx", "by", "team_of_slot")}
    pitch = data.pitch_m(SC_GAME)
    first = data.graded_blocks(SC_GAME)[0][1] + 3000
    lo, hi = first, first + 15000
    tac = {k: (v[lo:hi] if k != "team_of_slot" else v) for k, v in full.items()}
    tac["n_frames"] = hi - lo
    vis = np.isfinite(tac["px"]) & np.isfinite(tac["py"])
    team = tac["team_of_slot"]
    none = (vis[:, team == 0].sum(1) == 0) | (vis[:, team == 1].sum(1) == 0)
    assert none.mean() > 0.05                        # one team out of view on many frames
    monkeypatch.setattr(data, "graded_blocks", lambda game=None: [(1, 0, hi - lo - 1)])
    monkeypatch.setattr(data, "pitch_m", lambda game=None: pitch)
    f = lf.frame_features(tac, SC_GAME, version=V2S)
    fs = lf.frame_features(_swapped(tac), SC_GAME, version=V2S)
    fm = lf.frame_features(_mirrored(tac), SC_GAME, version=V2S)
    _check_equivariance(f, fs, fm, V2S, 1e-3, flip_share=0.005)
    # stage50-v2 fails the same check on this slice (cx, cy at -1 for the team out of view)
    g = lf.frame_features(tac, SC_GAME)
    gs = lf.frame_features(_swapped(tac), SC_GAME)
    k = lf.feature_names().index("A_cx")
    assert (np.abs(gs - lf.transform(g, True))[:, k] > 1).mean() > 0.01


def test_weights_and_markers_name_v2s(synth, tmp_path):
    states = [_random_state(50, seed=1), _random_state(50, seed=2)]
    p2, p2s = str(tmp_path / "w2.npz"), str(tmp_path / "w2s.npz")
    export.write(p2, states)                         # no version: the width says stage50-v2
    export.write(p2s, states, V2S)
    assert learned.model_version(learned.load_weights(p2)) == V2
    nets = learned.load_weights(p2s)
    assert learned.model_version(nets) == V2S
    with np.load(p2s) as z:
        assert str(z["feature_version"]) == V2S
    want = learned.bin_probabilities(
        nets, lf.downsample(lf.frame_features(synth, "METRICA-S3", version=V2S)))
    assert np.array_equal(learned.frame_probabilities(nets, synth, "METRICA-S3")[0], want)
    assert not np.array_equal(learned.frame_probabilities(learned.load_weights(p2), synth,
                                                          "METRICA-S3")[0], want)
    # a plain list of models has no recorded version: the width decides (stage50-v2)
    assert learned.model_version(list(nets)) == V2
    # the training records name the version for export (recorded_version)
    d = tmp_path / "fold0"
    d.mkdir()
    (d / "model_s0.json").write_text(json.dumps({"switches": {"r4": {"feature_version": V2S}}}))
    (d / "model_s1.json").write_text(json.dumps({"switches": {"r4": {}}}))
    assert export.recorded_version([str(d / "net_s0.pt")]) == V2S
    assert export.recorded_version([str(d / "net_s1.pt")]) == V2
    assert export.recorded_version([str(tmp_path / "x" / "net.pt")]) is None
    with pytest.raises(ValueError):
        export.recorded_version([str(d / "net_s0.pt"), str(d / "net_s1.pt")])
    # cache markers
    c = str(tmp_path / "c2s")
    build.write_marker(c, V2S)
    assert build.folder_version(c) == V2S
    assert build.folder_version(str(tmp_path / "unmarked")) == V2


def test_train_refuses_v2_against_v2s(tmp_path):
    train = pytest.importorskip("tools.learned_spike.train")
    old = train.FEATURE_VERSION
    try:
        c = str(tmp_path / "c2s")
        build.write_marker(c, V2S)
        train.configure_features(None)
        with pytest.raises(SystemExit):
            train.folder_columns(c)
        train.configure_features(V2S)
        assert train.folder_columns(c) is None and train.n_inputs() == 50
        with pytest.raises(SystemExit):
            train.folder_columns(str(tmp_path / "unmarked"))
    finally:
        train.configure_features(old)
