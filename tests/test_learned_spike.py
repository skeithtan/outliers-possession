"""Synthetic checks of the feature symmetries of tools/learned_spike (numpy only)."""

import numpy as np

from tools.learned_spike import features as fe


def test_symmetries_are_involutions():
    rng = np.random.RandomState(0)
    x = rng.standard_normal((7, fe.NF)).astype(np.float32)
    for swap, mirror in ((True, False), (False, True), (True, True)):
        twice = fe.transform(fe.transform(x, swap, mirror), swap, mirror)
        assert np.allclose(twice, x)


def test_swap_exchanges_team_columns_and_negates_x():
    names = [n for n, _ in fe.SPEC]
    x = np.arange(fe.NF, dtype=np.float32)[None] + 1.0
    s = fe.transform(x, swap=True)[0]
    assert s[names.index("H_d")] == x[0, names.index("A_d")]
    assert s[names.index("H_cx")] == -x[0, names.index("A_cx")]
    assert s[names.index("bx")] == -x[0, names.index("bx")]
    assert s[names.index("near_diff")] == -x[0, names.index("near_diff")]
    assert s[names.index("bspeed")] == x[0, names.index("bspeed")]
    m = fe.transform(x, mirror=True)[0]
    assert m[names.index("by")] == -x[0, names.index("by")]
    assert m[names.index("bx")] == x[0, names.index("bx")]


def test_downsample_and_bin_labels():
    f = np.arange(12, dtype=np.float32)[:, None]
    d = fe.downsample(f, 5)
    assert d.shape == (3, 1) and d[0, 0] == 2.0 and np.isclose(d[2, 0], 10.8)
    codes = np.array([0] * 5 + [1] * 5 + [2] * 2)
    assert list(fe.bin_labels(codes, 5)) == [0, 1, 2]
