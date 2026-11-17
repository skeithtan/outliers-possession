"""tools/analysis/shape_evidence.py on synthetic sequences (P9-B). No match data."""

import numpy as np

from engine import events
from tools.analysis import shape_evidence as se


def test_blind_walk_marks_frames_after_a_dead_frame_or_an_undecided_contact():
    n = 100
    blocks = [(1, 0, 99)]
    stoppage = np.zeros(n, dtype=bool)
    stoppage[40:50] = True
    # 0..4: block start before the first anchor. 5: H owns. 20: an A touch (other team). 30: H
    # owns again. 40..49: dead. 60: A owns. 70: a contested row. 80: an A row (the owner's own
    # touch) inside the blind stretch cuts it. 90: H owns.
    anchors = [(5, "H"), (30, "H"), (60, "A"), (90, "H")]
    contacts = [(10, "H"), (20, "A"), (70, None), (80, "A")]
    blind, cause, cut = se.blind_walk(n, blocks, stoppage, anchors, contacts)
    expected = np.zeros(n, dtype=bool)
    expected[0:5] = True
    expected[20:30] = True
    expected[50:60] = True
    expected[70:90] = True
    assert np.array_equal(blind, expected)
    assert set(cause[0:5]) == {"start"}
    assert set(cause[20:30]) == {"contact"}
    assert set(cause[50:60]) == {"dead"}
    assert set(cause[70:90]) == {"contested"}
    assert not stoppage[blind].any()
    assert se.stretches(blind, cut) == [(0, 5), (20, 30), (50, 60), (70, 80), (80, 90)]


def test_an_anchor_on_the_frame_of_a_contact_wins():
    n = 30
    blind, _, _ = se.blind_walk(n, [(1, 0, 29)], np.zeros(n, dtype=bool),
                                [(0, "H"), (10, "A")], [(10, "A"), (20, None)])
    assert not blind[:20].any() and blind[20:].all()


def test_shape_signal_is_the_backward_rise_difference_and_matches_shape_response():
    n, lag = 200, 25
    t = np.arange(n, dtype=np.float64)
    vf_h = 0.02 * t
    vf_a = np.where(t >= 100, -1.0, 0.0)
    stoppage = np.zeros(n, dtype=bool)
    stoppage[150] = True
    d = se.shape_signal(vf_h, vf_a, lag, stoppage)
    assert np.isnan(d[:lag]).all()
    assert np.isclose(d[60], 0.5)                       # H rises 0.5, A flat
    assert np.isclose(d[110], 1.5)                      # and A falls 1.0 inside the lag
    assert np.isnan(d[150:176]).all() and np.isfinite(d[176])
    shape = {"H": vf_h, "A": vf_a}
    for frame in (60, 110, 130):
        assert np.isclose(events.shape_response(shape, frame - lag, "H", lag), d[frame])


def test_agreement_counts_votes_buckets_and_the_combined_matrix():
    ref = np.array(["H", "H", "A", "A", "D", "H"])
    d = np.array([0.3, 2.0, 1.6, -3.0, 1.0, np.nan])
    shape_v = se.vote(d)
    near_v = np.array(["H", "A", "A", "H", "H", ""])
    near_s = np.array(["H", "H", "H", "H", "H", "H"])
    engine = np.array(["H", "A", "A", "A", "H", "H"])
    mask = np.ones(6, dtype=bool)
    ag = se.agreement(mask, ref, shape_v, np.abs(d), near_v, near_s, engine)
    assert ag["ref_ha"] == 5 and ag["engine_right"] == 4
    assert ag["shape"] == {"vote": 4, "right": 3}       # frame 2: shape H against ref A
    assert ag["near"] == {"vote": 4, "right": 2}
    assert ag["shape_bucket"][0] == {"vote": 1, "right": 1}
    assert ag["shape_bucket"][3] == {"vote": 2, "right": 1}
    assert ag["shape_bucket"][4] == {"vote": 1, "right": 1}
    cb = ag["combined"]
    assert cb["agree"] == 1 and cb["agree_right"] == 1           # frame 0
    assert cb["disagree"] == 3                                   # frames 1, 2, 3
    assert cb["disagree_shape"] == 2 and cb["disagree_near"] == 1
    assert cb["strong_disagree"] == 3 and cb["strong_shape"] == 2
    assert cb["none"] == 1                                       # frame 5


def test_splice_flips_only_strong_stretches_that_disagree_with_the_engine():
    labels = np.array(list("HHHHHDAAAA"))
    rows = [{"start": 0, "stop": 6, "d_bar": -2.2, "engine": "H"},
            {"start": 6, "stop": 10, "d_bar": -3.0, "engine": "A"},
            {"start": 6, "stop": 8, "d_bar": None, "engine": "A"}]
    out, changed = se.splice_stretches(labels, rows, 2.0)
    assert "".join(out) == "AAAAADAAAA" and changed == 1
    out, changed = se.splice_stretches(labels, rows, 2.5)
    assert "".join(out) == "HHHHHDAAAA" and changed == 0


def test_majority_and_length_histogram():
    assert se.majority(np.array(list("HAAD"))) == ("A", 0.5)
    assert se.majority(np.array([])) == ("", 0.0)
    assert se.length_histogram([5, 10, 30, 300]) == [1, 1, 1, 0, 0, 1]


def test_stretch_agreement_counts_by_d_bar_bucket():
    rows = [{"ref": "H", "d_bar": 2.0, "near_stretch": "A", "engine": "H", "len": 30},
            {"ref": "A", "d_bar": 2.2, "near_stretch": "A", "engine": "H", "len": 20},
            {"ref": "A", "d_bar": None, "near_stretch": "", "engine": "A", "len": 5},
            {"ref": "D", "d_bar": 3.0, "near_stretch": "H", "engine": "H", "len": 9}]
    out = se.stretch_agreement(rows)
    assert out[3] == {"n": 2, "shape": 1, "near_vote": 2, "near": 1, "engine": 1, "frames": 50}
    assert out[-1] == {"n": 1, "shape": 0, "near_vote": 0, "near": 0, "engine": 1, "frames": 5}
    assert out[4]["n"] == 0
