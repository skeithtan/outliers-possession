"""tools/analysis/redirect_forensics.py (F2): the walk that explains the redirect confirmation.

Synthetic contact lists only. The tests need no match data. The contacts are tuples
(frame, kind, team, confidence, distance) as events._bump_contacts() returns them. The control
rows are tuples (frame, "control", team, confidence, distance, note).
"""

import importlib.util
import os

import numpy as np
import pytest

from engine import events as ev
from engine.config import Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "redirect_forensics", os.path.join(ROOT, "tools", "analysis", "redirect_forensics.py"))
rf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rf)

FPS = 25.0


def contact(frame, kind, team, conf=0.8):
    return (frame, kind, team, conf, 0.3)


def control(frame, team, conf=0.8):
    return (frame, "control", team, conf, 0.5, "end={}".format(frame + 10))


def alive_all(n=2000):
    return np.ones(n, dtype=bool)


def by_frame(records):
    return {r["frame"]: r for r in records}


def test_walk_statuses_and_reasons():
    contacts = [
        contact(100, "kick", "H"),
        contact(200, "redirect", "H"),      # the owner's redirect: touch
        contact(300, "redirect", "A"),      # A redirect, A kick next: confirmed
        contact(350, "kick", "A"),
        contact(500, "redirect", "H"),      # H redirect, A kick next: next_other_team
        contact(600, "kick", "A"),
    ]
    records, stream = rf.explain_redirects(contacts, [], alive_all())
    got = by_frame(records)
    assert got[200]["status"] == "touch"
    assert got[300]["status"] == "confirmed" and got[300]["owner_before"] == "H"
    assert got[500]["status"] == "dropped" and got[500]["reason"] == "next_other_team"
    assert got[500]["next"]["team"] == "A" and got[500]["next"]["gap"] == 100
    assert [s[0] for s in stream] == [100, 200, 300, 350, 500, 600]


def test_walk_dead_between_and_no_next():
    alive = alive_all()
    alive[420:440] = False
    contacts = [contact(100, "kick", "H"), contact(400, "redirect", "A"),
                contact(500, "kick", "A"), contact(900, "redirect", "H")]
    records, _ = rf.explain_redirects(contacts, [], alive)
    got = by_frame(records)
    assert got[400]["status"] == "dropped" and got[400]["reason"] == "dead_between"
    assert got[400]["next"]["dead_between"] is True
    assert got[900]["status"] == "dropped" and got[900]["reason"] == "no_next"
    assert got[900]["next"] is None


@pytest.mark.parametrize("dead_at", [None, 450])
def test_walk_matches_the_engine(dead_at):
    """The explanation and events._redirect_kinds() agree on a mixed stream."""
    alive = alive_all()
    if dead_at:
        alive[dead_at:dead_at + 5] = False
    contacts = [
        contact(50, "receive", "A"), contact(120, "redirect", "H"), contact(160, "kick", "H"),
        contact(300, "redirect", "A"), contact(330, "contested", None),
        contact(420, "redirect", "H"), contact(700, "redirect", "A"),
        contact(780, "receive", "A"), contact(900, "redirect", "H"),
    ]
    controls = [control(500, "H"), control(800, "A")]
    records, _ = rf.explain_redirects(contacts, controls, alive)
    kinds = ev._redirect_kinds(contacts, controls, alive)
    rf.check_explanation(records, kinds)
    kinds[300] = "touch"
    with pytest.raises(AssertionError):
        rf.check_explanation(records, kinds)


def test_candidate_c1_control_run_after_an_intervener():
    """C1: a same-team control run within 2 s after the redirect, with another team between."""
    cfg = Config.default()
    contacts = [contact(100, "kick", "H"), contact(300, "redirect", "A"),
                contact(320, "kick", "H")]
    controls = [control(340, "A")]            # 40 frames after the redirect, 1.6 s
    records, stream = rf.explain_redirects(contacts, controls, alive_all())
    rec = by_frame(records)[300]
    assert rec["status"] == "dropped" and rec["reason"] == "next_other_team"
    flags = rf.candidate_flags(rec, stream, alive_all(), FPS, cfg)
    assert flags["C1_control_run"] is True
    assert flags["control_run_gap"] == 40
    # A run 3 s later is out of the horizon.
    records, stream = rf.explain_redirects(contacts, [control(400, "A")], alive_all())
    flags = rf.candidate_flags(by_frame(records)[300], stream, alive_all(), FPS, cfg)
    assert flags["C1_control_run"] is False and flags["control_run_gap"] is None


def test_candidate_c3_merge_within_the_kernel():
    """C3: the other team's item lies within CONTACT_HALF_WINDOW of the redirect."""
    cfg = Config.default()
    h = int(cfg.CONTACT_HALF_WINDOW)
    near = [contact(100, "kick", "H"), contact(300, "redirect", "A"),
            contact(300 + h - 2, "kick", "H"), contact(400, "kick", "A")]
    far = [contact(100, "kick", "H"), contact(300, "redirect", "A"),
           contact(300 + h + 5, "kick", "H"), contact(400, "kick", "A")]
    for contacts, expect in ((near, True), (far, False)):
        records, stream = rf.explain_redirects(contacts, [], alive_all())
        rec = by_frame(records)[300]
        assert rec["status"] == "dropped"
        assert rf.candidate_flags(rec, stream, alive_all(), FPS, cfg)["C3_bump_merge"] is expect


def test_classify_uses_the_challenge_proxies():
    assert rf.classify(control_s=3.0, passes=0) == "a"
    assert rf.classify(control_s=0.5, passes=3) == "a"
    assert rf.classify(control_s=1.9, passes=0) == "b"
    assert rf.classify(control_s=1.9, passes=1) == "c"
    assert rf.classify(control_s=2.5, passes=2) == "c"
    assert rf.classify_run(run_s=2.5, passes=0) == "c"


def test_loose_touch_events_drops_the_recovery_and_its_carry():
    def event(team, kind, start, end=None, period=1):
        return {"order": 0, "team": team, "type": kind, "subtypes": (), "start": start,
                "end": start if end is None else end, "period": period}

    events = [
        event("B", "RECOVERY", 100), event("B", "BALL LOST", 100, 160),
        event("B", "CARRY", 100, 101),                    # the one-frame carry of the touch
        event("A", "RECOVERY", 160), event("A", "PASS", 160, 180),
        event("A", "CARRY", 160, 161),                    # a real recovery stays
        event("B", "RECOVERY", 300), event("B", "BALL OUT", 301, 320),
        event("B", "CARRY", 300, 330),                    # a long carry stays
    ]
    kept, dropped = rf.loose_touch_events(events)
    assert sorted((e["type"], e["start"]) for e in dropped) == [
        ("CARRY", 100), ("RECOVERY", 100), ("RECOVERY", 300)]
    assert len(kept) == len(events) - 3
    assert ("CARRY", 160) in [(e["type"], e["start"]) for e in kept]
    assert ("CARRY", 300) in [(e["type"], e["start"]) for e in kept]


def test_diff_stretches_and_runs_text():
    a = np.array(list("HHHHAAAAHH"))
    b = np.array(list("HHAAAAAAAH"))
    assert rf.diff_stretches(a, b) == [(2, 4), (8, 9)]
    assert rf.runs_text(a, 0, 10) == "H4 A4 H2"
    assert rf.runs_text(a, 3, 3) == ""
