"""Tests of tools/learned_spike/r3_views.py (recorded reference views and cost rules)."""

import os

import numpy as np
import pytest

from tools.learned_spike import r3_views as V

CACHE = os.environ.get("CA")
SAMPLE = os.path.join(V.SAMPLE_ROOT, "Sample_Game_1")


def test_grid_contains_rule_b_and_the_wider_pairs():
    assert len(V.GRID) == 80 and len(V.RULE_B) == 56
    from tools.learned_spike import r3_nested
    assert [V.GRID[i] for i in V.RULE_B] == list(r3_nested.GRID)
    assert (64.0, 32.0) in V.GRID and V.GRID.index((64.0, 32.0)) not in V.RULE_B


def table(fn):
    """tab[view][input][game][i] from fn(view, input, game, i); c23 = 0.5 everywhere."""
    tab = {v: {i: {g: [fn(v, i, g, k) for k in range(len(V.GRID))] for g in V.DEV}
               for i in V.INPUTS} for v in V.VIEWS}
    base = {v: {i: {g: 0.5 for g in V.DEV} for i in V.INPUTS} for v in V.VIEWS}
    return tab, base


def test_rules_a_b_c_pick_what_they_say():
    a_best, b_best = V.RULE_B[3], V.GRID.index((64.0, 32.0))

    def fn(v, inp, g, k):
        if k == a_best:     # best as provided on raw, poor on the loose view of matched
            return 0.9 if (v, inp) != ("loose", "matched") else 0.4
        if k == b_best:     # never best, never poor
            return 0.7
        return 0.6
    tab, base = table(fn)
    rules = {k: V.apply_rule(tab, base, k) for k in "abc"}
    assert rules["a"]["final"] == a_best
    assert rules["b"]["final"] == b_best
    assert rules["b"]["final_score"] == pytest.approx(0.2)
    # c: a_best mean gain = (5 x 0.4 - 0.1) / 6 = 0.317 > 0.2
    assert rules["c"]["final"] == a_best
    assert set(rules["a"]["nested"]) == set(V.DEV)


def test_rule_a_never_leaves_rule_b_and_ties_go_low():
    tab, base = table(lambda v, inp, g, k: 1.0 if k == V.GRID.index((128.0, 64.0)) else 0.6)
    assert V.apply_rule(tab, base, "a")["final"] == V.RULE_B[0]
    assert V.apply_rule(tab, base, "b")["final"] == V.GRID.index((128.0, 64.0))


def test_provider_mean_weighs_providers_equally():
    vals = {g: (1.0 if g.startswith("METRICA") else 0.0) for g in V.DEV}
    assert V.provider_mean(vals, list(V.DEV)) == pytest.approx(0.5)


@pytest.mark.skipif(not (CACHE and os.path.isdir(SAMPLE)), reason="Metrica sample data and CA")
def test_loose_variant_rebuilds_the_cached_reference_and_drops_recoveries():
    provided, loose, info = V.metrica_refs("METRICA-S1", CACHE)
    assert info["rebuilt_equals_cache"]
    assert info["recovery_dropped"] == 76 and info["recovery_total"] == 278
    assert V.ref_count(provided) == 351 and V.ref_count(loose) == 255
    # graded frames and dead frames do not move; only the team of some alive frames
    assert np.array_equal(provided == 3, loose == 3)


def test_rule_b_takes_the_worst_provider_bp_pools_them():
    lo, hi = V.GRID.index((8.0, 4.0)), V.GRID.index((64.0, 32.0))

    def fn(v, inp, g, k):
        metrica = g.startswith("METRICA")
        if k == lo:     # Metrica loses under the loose view, DFL gains a lot everywhere
            return 0.45 if (metrica and v == "loose") else (0.9 if not metrica else 0.8)
        if k == hi:     # modest gain everywhere
            return 0.6
        return 0.5
    tab, base = table(fn)
    assert V.apply_rule(tab, base, "b")["final"] == hi       # min over providers: 0.1 > -0.05
    assert V.apply_rule(tab, base, "bp")["final"] == lo      # pooled: (-0.05 + 0.4) / 2 > 0.1
