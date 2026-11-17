"""tools/analysis/handover_confirm.py on synthetic sequences (P9-B measurements). No match data."""

import numpy as np

from engine import events
from tools.analysis import handover_confirm as hc


def test_reference_handovers_keep_direct_changes_inside_one_block():
    labels = np.array(list("HHHAAADDHHAAXXHH"))
    blocks = [(1, 0, 11), (2, 12, 15)]
    # H>A at 3 and at 10. D>H at 8 is no hand-over. A>X>H crosses X and a block.
    assert hc.reference_handovers(labels, blocks) == [(3, "H", "A"), (10, "H", "A")]


def test_reference_handovers_ignore_a_change_at_a_block_start():
    labels = np.array(list("HHHHAAAA"))
    assert hc.reference_handovers(labels, [(1, 0, 3), (2, 4, 7)]) == []


def test_match_handovers_is_one_to_one_and_greedy_by_distance():
    # Hand-overs at 100 and 110, contacts at 104 and 108. The pair 110-108 (2 frames) goes
    # first, then 100-104 (4 frames).
    assert hc.match_handovers([100, 110], [104, 108]) == [0, 1]
    # One contact, two hand-overs at equal distance: the earlier hand-over wins the tie.
    assert hc.match_handovers([100, 120], [110]) == [0, -1]


def test_match_handovers_ties_go_to_the_earlier_contact_and_respect_the_window():
    assert hc.match_handovers([100], [95, 105]) == [0]
    assert hc.match_handovers([100], [126], tol=25) == [-1]
    assert hc.match_handovers([100], [125], tol=25) == [0]
    # Contacts need not be sorted. The index refers to the input order.
    assert hc.match_handovers([100], [300, 101]) == [1]


def test_near_handover_window():
    assert hc.near_handover([50, 75, 76, 200], [100], tol=25) == [False, True, True, False]


def test_centred_nanmean_ignores_nan_and_shrinks_at_the_edges():
    values = np.array([1.0, np.nan, 3.0, 5.0, np.nan])
    out = hc.centred_nanmean(values, 1)
    assert np.allclose(out[:4], [1.0, 2.0, 4.0, 4.0])
    assert out[4] == 5.0
    assert np.isnan(hc.centred_nanmean(np.array([np.nan, np.nan]), 1)).all()


def test_team_forward_velocity_signs_by_the_own_goal_and_blanks_outside_blocks():
    n = 40
    team_of_slot = np.array([0, 0, 1])
    pvx = np.zeros((n, 3))
    pvx[:, 0], pvx[:, 1] = 2.0, 4.0      # home moves +x at 3 m/s on average
    pvx[:, 2] = -1.0                     # away moves -x at 1 m/s
    pvx[5, 1] = np.nan                   # a missing player does not count
    blocks = [(1, 0, 19), (2, 20, 39)]
    # Block 0: home defends x = 0, away x = 105. Block 1: the sides swap.
    own_goal = {(0, 0): 0.0, (0, 1): 105.0, (1, 0): 105.0, (1, 1): 0.0}
    vf = hc.team_forward_velocity(pvx, team_of_slot, own_goal, [b for b in blocks], half=0)
    assert np.isclose(vf["H"][10], 3.0) and np.isclose(vf["A"][10], 1.0)
    assert np.isclose(vf["H"][5], 2.0)
    assert np.isclose(vf["H"][30], -3.0) and np.isclose(vf["A"][30], -1.0)
    vf2 = hc.team_forward_velocity(pvx, team_of_slot, own_goal, [(1, 0, 9)], half=0)
    assert np.isnan(vf2["H"][15])


def test_shape_response_and_feature_on_a_step():
    n = 100
    vf = {"H": np.zeros(n), "A": np.zeros(n)}
    vf["H"][50:] = 1.5     # the new team accelerates forward from frame 50
    vf["A"][50:] = -0.5    # the old team drops back
    resp = hc.shape_response(vf, 40, [5, 10, 20, 70])
    assert resp["H"][:3] == [0.0, 1.5, 1.5]
    assert np.isnan(resp["H"][3])        # beyond the array
    assert hc.shape_feature(resp, "H", "A", 1) == 2.0
    assert hc.shape_feature(resp, "A", "H", 1) == -2.0


def test_auc_and_share_above():
    assert hc.auc([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert hc.auc([1.0], [1.0]) == 0.5
    assert hc.auc([np.nan], [1.0]) is None
    assert hc.share_above([1.0, 3.0, 5.0], [0.0, 2.0, 4.0]) == 2.0 / 3.0


def test_category_follows_the_bump_rule():
    from engine.config import Config

    cfg = Config.default()
    assert hc.category(0.5, 3.0, cfg) == "H"
    assert hc.category(3.0, 0.5, cfg) == "A"
    assert hc.category(0.5, 0.7, cfg) == "contested"
    assert hc.category(1.6, 4.0, cfg) == "none"
    assert hc.category(1.0, 1.6, cfg) == "H"       # the other team is beyond R_TOUCH


def test_walk_rule_matches_events_redirect_kinds():
    alive = np.ones(400, dtype=bool)
    alive[300] = False
    contacts = [(10, "kick", "H", 0.5, 0.5), (40, "redirect", "A", 0.5, 0.5),
                (60, "receive", "A", 0.5, 0.5), (90, "redirect", "H", 0.5, 0.5),
                (120, "kick", "A", 0.5, 0.5), (150, "contested", None, 0.5, 0.5),
                (200, "redirect", "A", 0.5, 0.5), (290, "redirect", "H", 0.5, 0.5),
                (320, "kick", "H", 0.5, 0.5)]
    controls = [(100, "control", "H", 0.5, 0.5, "end=105")]
    info, stream = hc.walk_rule(contacts, controls, alive)
    official = events._redirect_kinds(contacts, controls, alive)
    rewalk = {}
    for i, c in enumerate(contacts):
        decision = info.get(i, {}).get("decision")
        if decision == "redirect_own":
            rewalk[c[0]] = "touch"
        elif decision == "redirect_confirmed":
            rewalk[c[0]] = "control"
    assert rewalk == official
    assert info[1]["decision"] == "redirect_confirmed"      # A redirect, next item A receive
    assert info[3]["decision"] == "redirect_confirmed"      # H redirect, next item H control
    assert info[6]["decision"] == "redirect_own"            # A owns after the A kick at 120
    assert info[7]["decision"] == "redirect_dropped"        # a dead frame before the H kick
    assert info[7]["clause"] == "dead_between"
    assert info[0]["decision"] == "kick_none" and info[2]["decision"] == "receive_own"
    assert 5 not in info                                     # contested is not in the stream
    nxt = hc.next_items(stream, info[0]["pos"], np.concatenate(([0], np.cumsum(~alive))))
    assert nxt[0] == (40, "A", True)


def test_walk_and_rows_follow_the_engine_shape_clause():
    from engine.config import Config

    cfg = Config.default()
    alive = np.ones(400, dtype=bool)
    contacts = [(10, "kick", "H", 0.5, 0.5), (40, "redirect", "A", 0.5, 0.5),
                (200, "kick", "H", 0.5, 0.5), (230, "redirect", "A", 0.5, 0.5)]
    shape = {"H": np.zeros(400), "A": np.where(np.arange(400) > 45, 2.0, 0.0)}
    shape["A"][200:] = 0.0                      # no response after the redirect at 230
    info, _ = hc.walk_rule(contacts, [], alive, shape, cfg)
    assert [info[i]["clause"] for i in (1, 3)] == ["shape", "no_next"]
    assert events._redirect_kinds(contacts, [], alive, shape, cfg) == {40: "control"}
    assert hc.rows_after_rule(contacts, [], alive, shape, cfg)[40] == ("control", "A")
    assert hc.rows_after_rule(contacts, [], alive)[40] == ("none", "A")


def test_engine_contacts_rebuild_the_c18_rows_of_the_engine_table():
    import dataclasses

    from engine.config import Config
    from tests.test_events_reach import detect, near_redirect, scene
    from tests.test_events_shape import shape_of

    cfg = Config.default()
    sc = scene(reach_m=2.4, response=True)
    ball, feats, slot_conf, tactical = sc
    alive = np.ones(ball["visible"].shape[0], dtype=bool)
    shape_all = shape_of(tactical, cfg)
    contacts, shape, reach = hc.engine_contacts(ball, feats, alive, cfg, shape_all)
    controls = events._control_rows(ball, feats, None, alive, cfg)
    rows = {f: r for f, r in hc.rows_after_rule(contacts, controls, alive, shape, cfg).items()
            if r[0] != "none"}
    table = detect(sc)
    got = {int(r.frame): (r.kind, r.team) for r in table.itertuples() if r.notes in
           (events.REDIRECT_NOTE, "receive") or r.kind in ("release", "touch", "contested")}
    assert rows == got
    (frame,) = reach                                    # one reach contact, the turn of the scene
    assert rows[frame] == ("control", "A") and frame == int(near_redirect(table)["frame"].iloc[0])
    # Switched off, the contacts and the rows equal the C8 and C15 rules, with no reach frame.
    off = dataclasses.replace(cfg, REACH_SHAPE_CONFIRM=False)
    contacts_off, shape_off, reach_off = hc.engine_contacts(ball, feats, alive, off, shape_all)
    assert reach_off == set() and frame not in {c[0] for c in contacts_off}
    assert contacts_off == events._bump_contacts(ball, feats, None, alive, off)
    assert shape_off is shape_all
