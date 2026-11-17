"""Round 6 feature version "stage50-v2t": stage50-v2s with an attack direction that compares
the two teams (home attacks +x when its median x is below the away median x).

numpy only. The SkillCorner test runs when LEARNED_TEST_ENGINE_CACHE names an engine.data npz
cache folder holding SC-1886347_tactical.npz (an external match whose second half stage50-v2s
turns the wrong way).
"""

import os

import numpy as np
import pytest

from engine import data, learned
from engine import learned_features as lf
from engine import learned_pf
from tests.test_learned_v3 import (_check_equivariance, _dyadic, _match, _mirrored,
                                   _random_state, _swapped, PITCH)
from tools.learned_spike import build, export
from tools.learned_spike import features as spike_features

V2, V2S, V2T = lf.FEATURE_VERSION, lf.VERSION_V2S, lf.VERSION_V2T


def _same_side(tac, shift=0.2):
    """Both teams in the right half (home 0.55, away 0.85): the v2s rule is not swap-equivariant
    there, the team comparison is."""
    out = dict(tac)
    out["px"] = _dyadic(np.clip(tac["px"] + shift, 0.01, 0.99))
    return out


@pytest.fixture
def synth(monkeypatch):
    monkeypatch.setattr(data, "graded_blocks", lambda game=None: [(1, 0, 1179), (2, 1220, 2399)])
    monkeypatch.setattr(data, "pitch_m", lambda game=None: PITCH)
    return _match()


def test_v2t_has_the_columns_and_tables_of_v2():
    assert V2T in lf.VERSIONS and lf.feature_spec(V2T) == lf.feature_spec(V2)
    for a, b in zip(lf.symmetry_tables(V2T), lf.symmetry_tables(V2)):
        assert np.array_equal(a, b)
    assert lf.version_of_width(50) == V2
    assert learned_pf.full_spec(V2T) == learned_pf.full_spec(V2S)


def test_direction_rule():
    blocks = [(1, 0, 99)]
    team = np.array([0, 0, 1, 1])
    sw = np.array([1, 1, 0, 0])

    def direction(px, t=team, **kw):
        return lf.home_attacks_positive({"px": px, "team_of_slot": t}, blocks, 105.0, **kw)[1]

    px = np.full((100, 4), np.nan)
    px[:, :2], px[:, 2:] = 0.6, 0.8                   # both teams right of the centre
    assert direction(px, away_fallback=True) is False          # v2s: home median above 0.5
    assert direction(px, sw, away_fallback=True) is False      # ... and after a swap too
    assert direction(px, compare_teams=True) is True           # v2t: home left of away
    assert direction(px, sw, compare_teams=True) is False      # the swap turns it
    px[:, 2:] = np.nan                                          # home alone: the v2s rule
    assert direction(px, compare_teams=True) is False
    assert direction(px, sw, compare_teams=True) is True
    px[:, 2:] = 0.6                                             # equal medians: +x
    assert direction(px, compare_teams=True) is True
    assert direction(np.full((100, 4), np.nan), compare_teams=True) is True
    # y plays no part: the rule reads px only
    assert direction(px, compare_teams=True) is direction(px.copy(), compare_teams=True)


def test_v2t_is_equivariant_where_v2s_is_not(synth):
    tac = _same_side(synth)
    f = lf.frame_features(tac, "METRICA-S3", version=V2T)
    fs = lf.frame_features(_swapped(tac), "METRICA-S3", version=V2T)
    fm = lf.frame_features(_mirrored(tac), "METRICA-S3", version=V2T)
    _check_equivariance(f, fs, fm, V2T, 1e-5)
    _check_equivariance(lf.downsample(f), lf.downsample(fs), lf.downsample(fm), V2T, 1e-5)
    # stage50-v2s turns the pitch the same way for both orders: the swap check fails
    g = lf.frame_features(tac, "METRICA-S3", version=V2S)
    gs = lf.frame_features(_swapped(tac), "METRICA-S3", version=V2S)
    assert np.abs(gs - lf.transform(g, True)).max() > 1.0
    # the mirror alone does not touch the direction: v2s passes it
    gm = lf.frame_features(_mirrored(tac), "METRICA-S3", version=V2S)
    assert np.abs(gm - lf.transform(g, False, True)).max() <= 1e-5
    # the original synthetic match (home left, away right): v2t is equivariant too
    f = lf.frame_features(synth, "METRICA-S3", version=V2T)
    fs = lf.frame_features(_swapped(synth), "METRICA-S3", version=V2T)
    fm = lf.frame_features(_mirrored(synth), "METRICA-S3", version=V2T)
    _check_equivariance(f, fs, fm, V2T, 1e-5)


def test_v2t_equals_v2s_where_the_direction_agrees(synth):
    a = lf.frame_features(synth, "METRICA-S3", version=V2S)
    b = lf.frame_features(synth, "METRICA-S3", version=V2T)
    assert a.tobytes() == b.tobytes()
    tac = _same_side(synth)
    a = lf.frame_features(tac, "METRICA-S3", version=V2S)
    b = lf.frame_features(tac, "METRICA-S3", version=V2T)
    assert not np.array_equal(a, b)
    # the twins agree
    assert b.tobytes() == spike_features.frame_features(tac, "METRICA-S3",
                                                        version=V2T).tobytes()
    # stage50-v2 and stage50-v2s are unchanged by the new version (the twins agree on both)
    for v in (V2, V2S):
        assert lf.frame_features(tac, "METRICA-S3", version=v).tobytes() == \
            spike_features.frame_features(tac, "METRICA-S3", version=v).tobytes()


REAL_CACHE = os.environ.get("LEARNED_TEST_ENGINE_CACHE")
SC_GAME = "SC-1886347"


@pytest.mark.skipif(not REAL_CACHE or not os.path.isfile(
    os.path.join(REAL_CACHE or "", SC_GAME + "_tactical.npz")),
    reason="LEARNED_TEST_ENGINE_CACHE with the SC-1886347 engine cache is not set")
def test_v2t_is_equivariant_on_a_skillcorner_half_v2s_turns_wrong(monkeypatch):
    with np.load(os.path.join(REAL_CACHE, SC_GAME + "_tactical.npz")) as npz:
        full = {k: npz[k] for k in ("px", "py", "bx", "by", "team_of_slot")}
    pitch = data.pitch_m(SC_GAME)
    lo = data.graded_blocks(SC_GAME)[1][1] + 3000    # second half: home 0.507, away 0.663
    hi = lo + 15000
    tac = {k: (v[lo:hi] if k != "team_of_slot" else v) for k, v in full.items()}
    tac["n_frames"] = hi - lo
    monkeypatch.setattr(data, "graded_blocks", lambda game=None: [(2, 0, hi - lo - 1)])
    monkeypatch.setattr(data, "pitch_m", lambda game=None: pitch)
    blocks = data.graded_blocks()
    assert lf.home_attacks_positive(tac, blocks, 105.0, compare_teams=True) == {2: True}
    assert lf.home_attacks_positive(tac, blocks, 105.0, away_fallback=True) == {2: False}
    f = lf.frame_features(tac, SC_GAME, version=V2T)
    fs = lf.frame_features(_swapped(tac), SC_GAME, version=V2T)
    fm = lf.frame_features(_mirrored(tac), SC_GAME, version=V2T)
    _check_equivariance(f, fs, fm, V2T, 1e-3, flip_share=0.005)
    g = lf.frame_features(tac, SC_GAME, version=V2S)
    gs = lf.frame_features(_swapped(tac), SC_GAME, version=V2S)
    assert (np.abs(gs - lf.transform(g, True)).max(axis=1) > 0.1).mean() > 0.5


def test_weights_and_markers_name_v2t(synth, tmp_path):
    states = [_random_state(50, seed=1), _random_state(50, seed=2)]
    p = str(tmp_path / "w2t.npz")
    export.write(p, states, V2T)
    nets = learned.load_weights(p)
    assert learned.model_version(nets) == V2T
    tac = _same_side(synth)
    want = learned.bin_probabilities(
        nets, lf.downsample(lf.frame_features(tac, "METRICA-S3", version=V2T)))
    assert np.array_equal(learned.frame_probabilities(nets, tac, "METRICA-S3")[0], want)
    c = str(tmp_path / "c2t")
    build.write_marker(c, V2T)
    assert build.folder_version(c) == V2T


def test_train_takes_v2t_and_refuses_v2s_against_it(tmp_path):
    train = pytest.importorskip("tools.learned_spike.train")
    old = train.FEATURE_VERSION
    try:
        c = str(tmp_path / "c2t")
        build.write_marker(c, V2T)
        train.configure_features(V2S)
        with pytest.raises(SystemExit):
            train.folder_columns(c)
        train.configure_features(V2T)
        assert train.folder_columns(c) is None and train.n_inputs() == 50
    finally:
        train.configure_features(old)
