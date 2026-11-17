"""tools/analysis/wider_radius.py on synthetic sequences (P9-B clause 3). No match data."""

import numpy as np

from engine import events
from tests.test_events import (A, BUMP, FAST, H, bump_build, far, piecewise, smooth_path,
                               static)
from tools.analysis import wider_radius as wr

INF = np.inf


def test_band_of_excludes_the_lower_and_includes_the_upper_bound():
    assert wr.band_of(1.5) is None
    assert wr.band_of(1.51) == 0
    assert wr.band_of(2.0) == 0
    assert wr.band_of(2.01) == 1
    assert wr.band_of(3.0) == 2
    assert wr.band_of(4.0) == 3
    assert wr.band_of(4.01) is None
    assert wr.band_of(float("nan")) is None and wr.band_of(None) is None


def distances(n, values):
    """(N,) array of +inf with `values` {frame: distance} set."""
    out = np.full(n, INF)
    for frame, value in values.items():
        out[frame] = value
    return out


def test_find_candidates_sorts_maxima_into_classes_and_bands():
    n, h = 300, 2
    owner = np.array(["H"] * 250 + ["D"] * 50)
    graded = np.ones(n, dtype=bool)
    graded[:20] = False
    # 10: not graded. 40: nobody within 1.5 m, A (new) at 2.2 m at 41. 80: H (owner) at 1.0 m,
    # A at 2.8 m. 120: A within 1.5 m (a contact of the new team today). 160: A at 5 m, outside
    # every band. 200: nobody near, H at 1.8 m and A at 3.5 m. 260: engine label D.
    d_h = distances(n, {10: 2.0, 40: 3.0, 80: 1.0, 120: 2.0, 160: 1.0, 200: 1.8, 260: 3.0})
    d_a = distances(n, {10: 2.0, 41: 2.2, 80: 2.8, 120: 1.2, 160: 5.0, 200: 3.5, 260: 2.0})
    maxima = [10, 40, 80, 120, 160, 200, 260]
    cands, counts = wr.find_candidates(maxima, d_h, d_a, owner, graded, h, 1.5)
    assert counts == {"maxima": 6, "no_owner": 1, "new_near": 1, "out_of_band": 1}
    got = [(c["c"], c["cls"], c["owner"], c["new"], c["band"]) for c in cands]
    assert got == [(40, "nobody", "H", "A", 1), (80, "owner_near", "H", "A", 2),
                   (200, "nobody", "H", "A", 3)]
    assert np.isclose(cands[0]["d_new"], 2.2) and np.isclose(cands[1]["d_own"], 1.0)


def test_a_redirect_beside_a_player_beyond_the_touch_radius_is_a_nobody_candidate():
    # H kicks at 40 from beside its player. At 100 the ball turns 60 degrees with the nearest A
    # player about 2.2 m away, outside the turn. _bump_contacts gives no row there.
    n = 220
    bx, by = smooth_path(piecewise(n, [(40, FAST)]), piecewise(n, [(100, 60.0)]))
    players = [static(H, 10.0, 10.3, n), static(A, bx[100] + 0.8, by[100] - 2.0, n), far(H, n)]
    ball, feats, slot_conf = bump_build(bx, by, players)
    alive = np.ones(n, dtype=bool)
    h = int(BUMP.CONTACT_HALF_WINDOW)
    rows = events._bump_contacts(ball, feats, None, alive, BUMP)
    assert [(r[1], r[2]) for r in rows] == [("kick", "H")]
    maxima = wr.impulse_maxima(events.impulse(ball, h), alive, BUMP)
    assert any(abs(int(c) - 100) <= 2 for c in maxima)
    owner = np.array(["H"] * n)
    cands, counts = wr.find_candidates(maxima, feats["dH"], feats["dA"], owner,
                                       np.ones(n, dtype=bool), h, BUMP.R_TOUCH)
    assert len(cands) == 1 and abs(cands[0]["c"] - 100) <= 2
    cand = cands[0]
    lo, hi = cand["c"] - h, cand["c"] + h + 1
    expected = float(np.min(feats["dA"][lo:hi]))
    assert cand["cls"] == "nobody" and cand["new"] == "A"
    assert 1.5 < cand["d_new"] <= 2.5 and np.isclose(cand["d_new"], expected)
    assert cand["band"] == wr.band_of(expected)
    assert counts["out_of_band"] == 1   # the kick at 40: A is far away


def test_gate_needs_a_defined_clean_shape_and_the_extra_condition():
    base = {"shape": 1.6, "shape_clean": True, "airborne": False, "speed": 6.0}
    assert wr.gate(base, 1.45) and not wr.gate(base, 2.0)
    assert not wr.gate(dict(base, shape=None), 0.5)
    assert not wr.gate(dict(base, shape_clean=False), 0.5)
    assert wr.gate(base, 1.45, "not_airborne")
    assert not wr.gate(dict(base, airborne=True), 1.45, "not_airborne")
    assert wr.gate(base, 1.45, "slow") and not wr.gate(dict(base, speed=9.0), 1.45, "slow")
    assert not wr.gate(dict(base, speed=None), 1.45, "slow")


def test_cost_class_prefers_a_shift_then_reads_the_reference_at_c():
    ref = np.array(list("HHHHHAAAAADDDDD"))
    found = {"A": [5], "H": []}
    cand = {"c": 2, "owner": "H", "new": "A"}
    assert wr.cost_class(cand, ref, found, shift_tol=3) == "shift"
    assert wr.cost_class(cand, ref, found, shift_tol=2) == "spurious"
    assert wr.cost_class({"c": 8, "owner": "H", "new": "A"}, ref, {}, 3) == "ref_new"
    assert wr.cost_class({"c": 12, "owner": "H", "new": "A"}, ref, {}, 3) == "ref_dead"
    # a found hand-over to the other team is no shift
    assert wr.cost_class(cand, ref, {"H": [3]}, 3) == "spurious"


def test_splice_stops_at_the_next_owner_event_a_dead_frame_or_the_block_end():
    labels = np.array(list("HHHHHHHHHHDDHHHHHHHHXXHHHH"))
    blocks = [(1, 0, 19), (2, 22, 25)]
    out = wr.splice_labels(labels, [(2, "A")], [6, 15], blocks)
    assert "".join(out[:12]) == "HHAAAAHHHHDD"            # up to the owner event at 6
    out = wr.splice_labels(labels, [(8, "A")], [6, 15], blocks)
    assert "".join(out[:14]) == "HHHHHHHHAADDHH"          # up to the dead frame at 10
    out = wr.splice_labels(labels, [(17, "A")], [6, 15], blocks)
    assert "".join(out[15:22]) == "HHAAAXX"               # up to the block end at 19
    out = wr.splice_labels(labels, [(11, "A"), (23, "A")], [], blocks)
    assert "".join(out[10:12]) == "DD"                    # a candidate on D changes nothing
    assert "".join(out[22:]) == "HAAA"                    # no owner event: the block end
    assert (wr.splice_labels(labels, [], [6], blocks) == labels).all()
