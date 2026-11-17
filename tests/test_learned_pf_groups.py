"""The group columns of member spec F6 (engine/learned_pf_groups.py) and their wiring into the
per-frame member (engine/learned_pf.py). numpy only.

- the columns are closed under the team swap and keep its symmetry and the mirror's on a
  synthetic match;
- the shipped member (learned_pf_v7.npz, spec F5T) never computes them, and its features are
  unchanged;
- a weights file that lists them loads and runs (a synthetic tree model; and
  engine/models/learned_pf_v8.npz when it exists).
"""

import os

import numpy as np
import pytest

from engine import data
from engine import learned_features as lf
from engine import learned_pf as P
from engine import learned_pf_groups as G
from tests.test_learned_pf import make_synthetic_model, synth_with_touches
from tests.test_learned_v3 import _mirrored, _swapped
from tools.learned_spike import pf_export

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V7 = os.path.join(ROOT, "engine", "models", "learned_pf_v7.npz")
V8 = os.path.join(ROOT, "engine", "models", "learned_pf_v8.npz")
BASE = "stage50-v2t"
GAME = "METRICA-S3"


@pytest.fixture
def blocks(monkeypatch):
    monkeypatch.setattr(data, "graded_blocks", lambda game=None: [(1, 0, 1179), (2, 1220, 2399)])
    monkeypatch.setattr(data, "pitch_m", lambda game=None: (105.0, 68.0))


def test_spec_is_closed_under_the_swap():
    names = G.GROUP_NAMES
    assert len(names) == len(set(names)) == 36
    assert not set(names) & set(n for n, _ in P.full_spec(BASE))
    assert all(P.computable(n) for n in names)
    perm, sign = P.swap_columns(names, BASE)
    assert np.array_equal(perm[perm], np.arange(len(names)))
    assert np.all(sign == 1.0)                       # no signed column: pairs and invariants
    assert names[perm[names.index("H_gahead")]] == "A_gahead"
    assert names[perm[names.index("g_ovx")]] == "g_ovx"
    with pytest.raises(ValueError):
        P.swap_columns(["H_gahead"], BASE)          # its partner is missing


def test_team_swap_and_mirror_on_a_synthetic_match(blocks):
    tac = synth_with_touches()
    F = G.group_features(tac, GAME, BASE)
    assert F.dtype == np.float32 and F.shape == (2400, 36) and np.isfinite(F).all()
    perm, sign, mirror = lf._symmetry_tables(G.GROUP_SPEC)
    S = G.group_features(_swapped(tac), GAME, BASE)
    assert np.abs(S - F[:, perm] * sign).max() < 1e-4
    M = G.group_features(_mirrored(tac), GAME, BASE)
    assert np.abs(M - F * mirror).max() < 1e-4
    # every group is used on this match
    for name in ("H_gahead", "A_gbehind", "H_gvc", "A_garea", "g_ovx", "g_cdist", "H_gnacc"):
        assert np.abs(F[:, G.GROUP_NAMES.index(name)]).max() > 0, name


def test_missing_ball_and_the_block_edges(blocks):
    tac = synth_with_touches()
    F = G.group_features(tac, GAME, BASE)
    col = {n: F[:, i] for i, n in enumerate(G.GROUP_NAMES)}
    # the ball is not in the tracking on frames 700-759 (tests.test_learned_v3._match)
    hidden = slice(705, 755)
    for name in ("g_bacc", "g_bturn"):
        assert np.all(col[name][hidden] == 0), name
    # the block columns stay defined there, and the counts follow the filled ball
    assert np.all(col["H_garea"][hidden] > 0)
    n_h = col["H_gbehind"] + col["H_glevel"] + col["H_gahead"]
    assert np.allclose(n_h[hidden], n_h[704])
    # windowed columns are 0 between the blocks
    gap = slice(1180, 1220)
    for name in ("H_gvc", "H_gdahead_p1", "g_bacc_m1"):
        assert np.all(col[name][gap] == 0), name


def test_the_shipped_member_never_computes_them(blocks, monkeypatch):
    model = P.load_pf(V7)
    assert model["md5"] == "0b5e81aa00c0486057aa7eb790c98109"
    assert not set(model["columns"]) & G.GROUP_SET
    before = P.pf_features(synth_with_touches(), GAME, model["columns"], BASE)

    def refuse(*a, **k):
        raise AssertionError("group columns computed for the v7 member")
    monkeypatch.setattr(G, "group_features", refuse)
    after = P.pf_features(synth_with_touches(), GAME, model["columns"], BASE)
    assert np.array_equal(before, after)
    P.frame_probabilities_pf(synth_with_touches(), GAME, model)


def test_a_member_with_group_columns_loads_and_runs(blocks, tmp_path):
    arrays = make_synthetic_model()
    cols = [str(c) for c in arrays["columns"]] + G.GROUP_NAMES
    arrays["columns"] = np.array(cols)
    arrays["base_version"] = np.str_(BASE)
    # point some splits of the team model at the group columns
    feat = arrays["team/feature"].copy()
    inner = ~arrays["team/leaf"]
    idx = np.flatnonzero(inner)[::3]
    feat[idx] = len(cols) - 36 + (np.arange(len(idx)) % 36)
    arrays["team/feature"] = feat
    path = str(tmp_path / "f6.npz")
    np.savez(path, **arrays)
    model = P.load_pf(path)
    tac = synth_with_touches()
    F = P.pf_features(tac, GAME, model["columns"], BASE)
    assert np.array_equal(F[:, -36:], G.group_features(tac, GAME, BASE))
    p = P.member_probabilities(model, tac, GAME, swap_mode="tracking")
    q = P.member_probabilities(model, _swapped(tac), GAME, swap_mode="tracking")
    assert np.array_equal(q, p[:, [1, 0, 2]])
    c = P.member_probabilities(model, tac, GAME)
    d = P.member_probabilities(model, _swapped(tac), GAME)
    assert np.abs(d - c[:, [1, 0, 2]]).max() < 1e-6
    r = P.frame_probabilities_pf(tac, GAME, model)
    assert np.allclose(r.sum(axis=1), 1.0, atol=2e-4)
    # the exporter writes such a file (a DEAD model may read only the leading columns)
    out = pf_export.arrays_of({m: {k: arrays["{}/{}".format(m, k)] for k in P.NODE_ARRAYS +
                                   ("roots", "baseline")} for m in P.MODELS},
                              cols, BASE, "F6-test", "0" * 32)
    assert list(out["columns"]) == cols


@pytest.mark.skipif(not os.path.isfile(V8), reason="engine/models/learned_pf_v8.npz not present")
def test_v8_loads_and_runs(blocks):
    model = P.load_pf(V8)
    assert set(model["columns"]) & G.GROUP_SET
    assert model["base_version"] == BASE
    r = P.frame_probabilities_pf(synth_with_touches(), GAME, model)
    assert r.shape == (2400, 3) and np.allclose(r.sum(axis=1), 1.0, atol=2e-4)


# -- member spec F6c: robust units per match (protocol R8 note 6.3) -------------------------------

# md5 (learned_features.matrix_md5) of the raw group columns on synth_with_touches(), base
# stage50-v2t, as computed by the F6a code (commit 1068a0c): the raw scale must not change.
RAW_SYNTH_MD5 = "76d2150142b8dbe57216d80c03020177"


def test_raw_scale_is_the_f6a_code(blocks):
    F = G.group_features(synth_with_touches(), GAME, BASE)
    assert lf.matrix_md5(F) == RAW_SYNTH_MD5
    assert lf.matrix_md5(G.group_features(synth_with_touches(), GAME, BASE, scale="raw")) == \
        RAW_SYNTH_MD5
    with pytest.raises(ValueError):
        G.group_features(synth_with_touches(), GAME, BASE, scale="other")


def test_robust_units_symmetry_counts_and_centring(blocks):
    tac = synth_with_touches()
    R = G.group_features(tac, GAME, BASE)
    F = G.group_features(tac, GAME, BASE, scale="robust")
    assert F.shape == R.shape and np.isfinite(F).all()
    perm, sign, mirror = lf._symmetry_tables(G.GROUP_SPEC)
    S = G.group_features(_swapped(tac), GAME, BASE, scale="robust")
    assert np.abs(S - F[:, perm] * sign).max() < 1e-3
    M = G.group_features(_mirrored(tac), GAME, BASE, scale="robust")
    assert np.abs(M - F * mirror).max() < 1e-3
    names = G.GROUP_NAMES
    blk = np.zeros(len(F), bool)
    blk[0:1180] = True
    blk[1220:2400] = True
    for i, n in enumerate(names):
        stem = n[2:] if n[:2] in ("H_", "A_") else n
        if stem in G.COUNT_COLUMNS:
            assert np.array_equal(F[:, i], R[:, i]), n     # counts stay raw
    # a pooled team column: the median over both teams' defined frames is 0
    for stem in ("gvc", "garea"):
        pooled = np.concatenate([F[blk, names.index("H_" + stem)],
                                 F[blk, names.index("A_" + stem)]])
        assert abs(np.median(pooled)) < 1e-6, stem
    # an exact relabelling: robust units of the swapped columns equal the swapped robust units
    cols = {n: R[:, i].astype(np.float64) for i, n in enumerate(names)}
    defd = {n: blk.copy() for n in names}
    swapped = {n: cols[names[perm[i]]] for i, n in enumerate(names)}
    a = G.robust_units(cols, defd)
    b = G.robust_units(swapped, defd)
    for i, n in enumerate(names):
        assert np.array_equal(b[n], a[names[perm[i]]]), n


def test_robust_ball_columns_follow_the_missing_ball_convention(blocks):
    F = G.group_features(synth_with_touches(), GAME, BASE, scale="robust")
    col = {n: F[:, i] for i, n in enumerate(G.GROUP_NAMES)}
    hidden = slice(705, 755)
    for name in ("g_bacc", "g_bturn", "g_bacc_m04", "H_gnacc"):
        assert np.all(col[name][hidden] == 0), name
    assert np.all(col["H_gbc"][hidden] != 0)          # positions keep the filled ball


def test_weights_file_flag_selects_the_scale(blocks, tmp_path):
    arrays = make_synthetic_model()
    cols = [str(c) for c in arrays["columns"]] + G.GROUP_NAMES
    arrays["columns"] = np.array(cols)
    arrays["base_version"] = np.str_(BASE)
    raw_path = str(tmp_path / "raw.npz")
    np.savez(raw_path, **arrays)
    assert P.load_pf(raw_path)["group_scale"] == "raw"
    arrays["group_scale"] = np.str_("robust")
    rob_path = str(tmp_path / "robust.npz")
    np.savez(rob_path, **arrays)
    model = P.load_pf(rob_path)
    assert model["group_scale"] == "robust"
    tac = synth_with_touches()
    F = P.pf_features(tac, GAME, model["columns"], BASE, group_scale=model["group_scale"])
    assert np.array_equal(F[:, -36:], G.group_features(tac, GAME, BASE, scale="robust"))
    P.frame_probabilities_pf(tac, GAME, model)
    arrays["group_scale"] = np.str_("other")
    bad = str(tmp_path / "bad.npz")
    np.savez(bad, **arrays)
    with pytest.raises(ValueError):
        P.load_pf(bad)
    # the exporter writes the flag only when it is not raw
    m = {k: {a: arrays["{}/{}".format(k, a)] for a in P.NODE_ARRAYS + ("roots", "baseline")}
         for k in P.MODELS}
    assert "group_scale" not in pf_export.arrays_of(m, cols, BASE, "t", "0" * 32)
    assert str(pf_export.arrays_of(m, cols, BASE, "t", "0" * 32,
                                   group_scale="robust")["group_scale"]) == "robust"
