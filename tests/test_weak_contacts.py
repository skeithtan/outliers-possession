"""tools/analysis/weak_contacts.py on synthetic sequences (RESEARCH-3 R3-A). No match data."""

import numpy as np

from engine import events
from tests.test_events import (A, BUMP, FAST, H, bump_build, far, piecewise, smooth_path,
                               static)
from tools.analysis import weak_contacts as wc

INF = np.inf


def test_switch_stats_counts_direct_team_changes_inside_each_block():
    labels = np.array(list("HHAADDAAHHXXHHAA"))
    blocks = [(1, 0, 9), (2, 12, 15)]
    stats = wc.switch_stats(labels, blocks, fps=1.0)
    # Block 1: H A D A H, switches H->A and A->H (the D breaks A->A). Block 2: H A, one switch.
    assert stats["switches"] == 3
    assert stats["possessions"] == 6
    assert stats["alive_frames"] == 12
    assert np.isclose(stats["alive_min"], 12 / 60.0)
    assert np.isclose(stats["rate"], 3 / (12 / 60.0))
    assert wc.switch_stats(np.array(list("DDDD")), [(1, 0, 3)], 25.0)["rate"] is None


def test_j_band_includes_the_lower_and_excludes_the_upper_bound():
    assert wc.j_band_of(4.99) is None
    assert wc.j_band_of(5.0) == 0
    assert wc.j_band_of(5.99) == 0
    assert wc.j_band_of(6.0) == 1
    assert wc.j_band_of(8.99) == 3
    assert wc.j_band_of(9.0) is None
    assert wc.j_band_of(float("nan")) is None and wc.j_band_of(None) is None
    assert wc.J_BANDS[-1][1] == BUMP.DV_CONTACT


def test_weak_maxima_keeps_window_maxima_under_the_contact_threshold():
    n, h = 300, 5
    size = np.zeros(n)
    size[50] = 7.0      # a weak maximum
    size[100] = 12.0    # a strong maximum (a contact today)
    size[104] = 6.0     # weak, but the strong maximum lies within h
    size[150] = 4.0     # under the floor
    size[200] = 8.5     # weak, but on a dead frame
    size[250] = 5.0     # exactly the floor
    size[252] = 5.0     # a plateau inside the window: the first frame wins
    alive = np.ones(n, dtype=bool)
    alive[200] = False
    got = wc.weak_maxima(size, alive, h, 5.0, 9.0)
    assert got.tolist() == [50, 250]
    assert wc.weak_maxima(np.full(n, np.nan), alive, h, 5.0, 9.0).size == 0


def distances(n, values):
    """(N,) array of +inf with `values` {frame: distance} set."""
    out = np.full(n, INF)
    for frame, value in values.items():
        out[frame] = value
    return out


def test_find_weak_sorts_maxima_into_classes():
    n, h = 300, 2
    owner = np.array(["H"] * 250 + ["D"] * 50)
    graded = np.ones(n, dtype=bool)
    graded[:20] = False
    size = np.zeros(n)
    for c, j in ((10, 6.0), (40, 6.5), (80, 8.2), (120, 7.0), (260, 5.5)):
        size[c] = j
    speed = np.full(n, 5.0)
    speed[40 + h] = 1.0      # 40: the speed drops by 4 >= 0.5 x 6.5, a receive
    speed[80 + h] = 9.5      # 80: it rises by 4.5 >= 0.5 x 8.2, a kick
    # 10: not graded. 40: A (new) at 1.0 m, H far: uncontested. 80: A at 1.2 m, H at 0.8 m:
    # contested. 120: A at 2.0 m: new team far. 260: engine label D.
    d_h = distances(n, {10: 1.0, 40: 4.0, 80: 0.8, 120: 1.0, 260: 1.0})
    d_a = distances(n, {10: 1.0, 41: 1.0, 80: 1.2, 120: 2.0, 260: 1.0})
    cands, counts = wc.find_weak([10, 40, 80, 120, 260], size, speed, d_h, d_a, owner, graded,
                                 h, 1.5, 0.5)
    assert counts == {"maxima": 4, "no_owner": 1, "new_far": 1}
    got = [(c["c"], c["new"], c["contested"], c["band"], c["kind"]) for c in cands]
    assert got == [(40, "A", False, 1, "receive"), (80, "A", True, 3, "kick")]
    assert np.isclose(cands[0]["d_new"], 1.0) and cands[0]["d_own"] == 4.0
    assert np.isclose(cands[1]["d_own"], 0.8)


def test_a_soft_reception_by_the_other_team_is_a_weak_uncontested_candidate():
    # H kicks at 40 from beside its player (a strong contact). At 100 an A player standing on
    # the line takes 8 m/s out of the ball: |J| about 0.79 x 8, under DV_CONTACT, so
    # _bump_contacts gives no row there.
    n = 220
    speeds = piecewise(n, [(40, FAST), (100, FAST - 8.0)])
    bx, by = smooth_path(speeds, piecewise(n, []))
    players = [static(H, 10.0, 10.3, n), static(A, bx[100], by[100] + 0.5, n), far(H, n)]
    ball, feats, _ = bump_build(bx, by, players)
    alive = np.ones(n, dtype=bool)
    h = int(BUMP.CONTACT_HALF_WINDOW)
    rows = events._bump_contacts(ball, feats, None, alive, BUMP)
    assert [(r[1], r[2]) for r in rows] == [("kick", "H")]
    size = events.impulse(ball, h)
    maxima = wc.weak_maxima(size, alive, h, wc.J_LO, BUMP.DV_CONTACT)
    assert [int(c) for c in maxima if abs(int(c) - 100) <= 2]
    owner = np.array(["H"] * n)
    cands, _ = wc.find_weak(maxima, size, ball["speed"], feats["dH"], feats["dA"], owner,
                            np.ones(n, dtype=bool), h, BUMP.R_TOUCH, BUMP.REDIRECT_FRACTION)
    near = [c for c in cands if abs(c["c"] - 100) <= 2]
    assert len(near) == 1
    cand = near[0]
    assert cand["new"] == "A" and not cand["contested"] and cand["kind"] == "receive"
    assert wc.J_LO <= cand["j"] < BUMP.DV_CONTACT


def test_gate_needs_a_defined_clean_shape_and_the_guard():
    base = {"shape": 1.6, "shape_clean": True, "contested": False}
    assert wc.gate(base, 1.45) and not wc.gate(base, 2.0)
    assert not wc.gate(dict(base, shape=None), 1.45)
    assert not wc.gate(dict(base, shape_clean=False), 1.45)
    assert wc.gate(base, 1.45, "uncontested")
    assert wc.gate(dict(base, contested=True), 1.45, "all")
    assert not wc.gate(dict(base, contested=True), 1.45, "uncontested")


def test_nearest_in_and_spell_frames():
    assert wc.nearest_in([10, 30, 50], 28, 5) == 30
    assert wc.nearest_in([10, 30, 50], 40, 10) == 30      # a tie: the earlier one
    assert wc.nearest_in([10, 30, 50], 40, 5) is None
    assert wc.nearest_in([], 40, 5) is None
    labels = np.array(list("HHHAAAADHH"))
    assert wc.spell_frames(labels, 3, 9) == 4             # AAAA up to the D
    assert wc.spell_frames(labels, 8, 9) == 2             # up to the block end
    assert wc.spell_frames(labels, 3, 4) == 2             # capped by a block that ends earlier


def test_flags_text_round_trips_and_pick_filters():
    flags = np.array([[1, 0, 1], [0, 0, 1]], dtype=np.int8)
    assert wc.flags_text(flags) == ["101", "001"]
    assert (wc.flags_array(wc.flags_text(flags)) == flags).all()
    cands = [{"j": 5.5, "band": 0, "contested": True}, {"j": 7.2, "band": 2, "contested": False},
             {"j": 8.9, "band": 3, "contested": True}]
    assert len(wc.pick(cands, cls="contested")) == 2
    assert [c["j"] for c in wc.pick(cands, floor=7.0)] == [7.2, 8.9]
    assert [c["j"] for c in wc.pick(cands, band=2, cls="uncontested")] == [7.2]
