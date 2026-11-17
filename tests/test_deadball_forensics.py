"""Tests for tools/analysis/deadball_forensics.py on synthetic label vectors (no match data)."""

import importlib.util
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "tools", "analysis", "deadball_forensics.py")


def _module():
    spec = importlib.util.spec_from_file_location("deadball_forensics", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H, A, D = 0, 1, 2


def _labels(pattern):
    """int8 codes from a string such as 'HHAADD'."""
    return np.array([{"H": H, "A": A, "D": D}[c] for c in pattern], dtype=np.int8)


def test_change_stretches_split_by_label_pair():
    fx = _module()
    old = _labels("HHHHDDDDAAAA")
    new = _labels("HHAADDHHAAHH")
    paint = np.array([0, 0, 1, 1, -1, -1, 0, 0, 1, 1, 0, 0], dtype=np.int8)
    rows = fx.change_stretches(old, new, paint, [(1, 0, 11)])
    assert [(r["start"], r["end"], r["old"], r["new"]) for r in rows] == [
        (2, 4, "H", "A"), (6, 8, "D", "H"), (10, 12, "A", "H")]
    assert [(r["d_agree"], r["d_alive"]) for r in rows] == [(2, 0), (2, 2), (2, 0)]


def test_agreement_split_is_exact():
    """The per-stretch contributions add up to the change of the agreement ratio."""
    fx = _module()
    rng = np.random.RandomState(0)
    old = rng.randint(0, 3, size=400).astype(np.int8)
    new = old.copy()
    flip = rng.rand(400) < 0.2
    new[flip] = rng.randint(0, 3, size=int(flip.sum())).astype(np.int8)
    paint = rng.randint(-1, 2, size=400).astype(np.int8)
    a0, n0 = fx.agreement(old, paint)
    a1, n1 = fx.agreement(new, paint)
    r0 = a0 / float(n0)
    rows = fx.change_stretches(old, new, paint, [(1, 0, 399)])
    total = sum((r["d_agree"] - r0 * r["d_alive"]) / n1 for r in rows)
    assert abs(total - (a1 / float(n1) - r0)) < 1e-12


def test_c8_class():
    fx = _module()
    assert fx.c8_class("control/receive@100", "") == "receive"
    assert fx.c8_class("control/redirect@100", "touch/deflection@95") == "receive"
    assert fx.c8_class("release/@100", "release/deflection@96") == "deflection"
    assert fx.c8_class("release/@100", "release/@96") == "other"
    assert fx.c8_class("", "") == "other"
