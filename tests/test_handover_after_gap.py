"""tools/analysis/handover_after_gap.py on a synthetic sequence (P9-D). No match data needed."""

import numpy as np
import pytest

from tools.analysis import handover_after_gap as hg

FPS = 25
BLOCKS = [(1, 0, 999)]


def test_measure_counts_hand_overs_on_the_first_contact_after_a_long_gap():
    """Two long gaps and one short one. Only the switch at the first contact after a gap counts.

    Gap 100-300 (8 s) is labelled H throughout and the first contact after it, at 320, switches
    to A. Gap 500-700 (8 s) holds D, and its first contact at 710 switches back to H after a
    restart. The 1 s gap 800-825 is not long. The switch at 900 sits on an ordinary contact.
    """
    labels = np.array(["H"] * 1000)
    labels[321:] = "A"
    labels[550:705] = "D"
    labels[711:] = "H"
    labels[901:] = "A"
    missing = np.zeros(1000, dtype=bool)
    missing[100:300] = True
    missing[500:700] = True
    missing[800:825] = True
    events = [50, 320, 330, 710, 840, 899]
    table = hg.measure(labels, missing, events, BLOCKS, FPS, window=3)
    assert table["hand_overs"] == 3                  # 321, 711 and 901. 705 is a restart.
    assert table["long_gaps"] == 2 and table["alive_gaps"] == 1
    assert table["landed"] == 2 and table["landed_alive"] == 1
    assert table["landed_gap_frames"] == 400
    assert table["landed_gap_lengths"]["median_s"] == 8.0
    assert table["contact_delay_median_s"] == pytest.approx((20 + 10) / 2.0 / FPS)
    rows = {b["bucket"]: b for b in table["buckets"]}
    assert rows["5-10 s"]["hand_overs"] == 2 and rows["5-10 s"]["hand_overs_alive"] == 1
    assert "hand-overs on the first contact" in "\n".join(hg.format_lines("SYNTH", table))


def test_hand_overs_skip_team_changes_across_d():
    labels = np.array(list("HHHDDDAAAHHH"))
    assert hg.hand_overs(labels, [(1, 0, 11)]) == [9]
