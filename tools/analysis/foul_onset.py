"""Foul onsets and the contested-contact slowdown signature (RESEARCH-3 R3-C). It measures.

    python tools/analysis/foul_onset.py collect --game NAME [--ref REFERENCE.csv]
        [--metrica-events EVENTS] [--dfl-events EV.xml --dfl-positions POS.xml]
        [--public-gt PUBLIC_GT.csv] [--f1-doc DEADBALL-FORENSICS.md] --out OUT.json
    python tools/analysis/foul_onset.py report OUT.json [OUT.json ...] [--out REPORT.md]

`collect` runs the engine (possession.prepare and possession.run with the default Config) on one
match and stores per-onset and per-contact summary numbers in a JSON file. `report` prints the
tables of docs/plan/calibration/R3C_foul_onset.md. Set MTAH_DATA_DIR for each match as
tools/score_game.py needs it. Write the JSON files outside the repository. The script refuses
the sealed matches of calibration/SPLITS.md. It designs no rule and changes no engine file.

Definitions.

- Onset. A frame t with label D whose previous frame t - 1 has label H or A. A reference onset
  comes from the reference CSV. An engine onset comes from the engine labels (after every
  offset). A D run that starts after an X frame (a block start) has no onset.
- Exit of a reference onset: the first frame after its D run.
- Status of a reference onset o. Take the engine onset e nearest o with e - o in
  [-LATE_TOL, +LATE_TOL]. |e - o| <= MATCH_TOL is "matched", e - o in [MATCH_TOL + 1, LATE_TOL]
  is "late", e - o in [-LATE_TOL, -MATCH_TOL - 1] is "early". With no such e the onset is
  "inside" when the engine label at o is D (the engine is still in an earlier stoppage), else
  "missed". Ties go to the earlier engine onset.
- Entry mechanism of an engine onset e: the first frame s of the engine stoppage mask
  (possession.run parts["stoppage"]) nearest e within MATCH_TOL. "out" or "pursuit_stop" when an
  onset row of deadball.dead_onsets() of that mechanism lies at s, "evidence" when s starts a run
  of e_D > DEAD_EVIDENCE_HIGH that lasts COLLECTIVE_STOP_S (the dwell entry of
  possession.dead_intervals), "decoder" when no stoppage starts near e.
- Kind of a reference onset, Metrica. The stopping events of tools/events_gt.py (is_stop) whose
  end frame is the onset (the nearest end within MATCH_TOL when none is exact). The kind is
  "foul" when one of them is FAULT RECEIVED or CHALLENGE with the FAULT subtype, else "offside"
  (OFFSIDE subtype), else "hand_ball" (HAND BALL subtype), else "goal" (SHOT with GOAL), else
  "out" (BALL OUT, or SHOT with OUT), else "no_stop" (the reference opened D before a set piece
  without a stopping event). The detail names the types and subtypes, for example
  "CHALLENGE GROUND" or "FAULT RECEIVED".
- Kind of a reference onset, DFL. The restart event is the set-piece event of the DFL event file
  nearest the exit within DFL_EXIT_TOL frames (restart_team.dfl_kinds). Foul and Offside events
  are read with the same time mapping (the first ball frame N of the half plus the EventTime
  after the time T of that frame, times the frame rate). ThrowIn, CornerKick and GoalKick give
  "out", KickOff "goal", Penalty "foul", RefereeBall "drop_ball". A FreeKick, or no restart
  event, gives "offside" when an Offside event lies in [o - DFL_STOP_BEFORE, exit], else "foul"
  when a Foul event lies there, else "free_kick_no_event" or "no_event". The geometry of the
  ball at the onset (restart_team.out_geometry: "seen" beyond a line, "heading" for a line,
  "inside") is stored beside the kind.
- Causes of a missed onset o (the engine label at o is H or A). Window W = [o - MATCH_TOL,
  o + LATE_TOL]. The flags:
    entry       an out or pursuit_stop onset row, or a dwell entry, lies in W (a candidate
                entered the state machine, but no label edge came near o);
    hint        a collective_stop onset row lies in W (a soft hint: it never enters D);
    out         the furthest stage of a missing-ball out extrapolation whose last visible
                frame lies in [o - LATE_TOL, o + MATCH_TOL] (out_diagnose), or "seen" when the
                visible ball is beyond a line plus PITCH_MARGIN_M in W, or "none";
    pursuit     the furthest stage of a fast run (pursuit_diagnose) that starts in
                [o - LATE_TOL, o + MATCH_TOL], or "none";
    stop_4s     a collective-stop run (meanv_all < V_STOP and sep > SEP_DEAD for
                COLLECTIVE_STOP_S) overlaps [o, o + ONSET_CONFIRM_S] (deadball._confirmed);
    slow_4s     meanv_all < V_STOP on a frame of [o, o + ONSET_CONFIRM_S];
    vanished    the ball is missing on a frame of W.
  The primary cause is the first that holds of: entry ("entry_no_edge"), out unconfirmed
  ("out_unconfirmed"), pursuit unconfirmed ("pursuit_unconfirmed"), hint ("collective_hint"),
  a pursuit stage below unconfirmed ("pursuit_<stage>"), vanished without stop_4s
  ("gap_no_stop"), no stop_4s ("no_stop_4s"), else "stop_4s_touched" (players stopped but a
  touch broke every 2 s quiet run, or the ball is missing then).
- Stages of out_diagnose (furthest first): ok, unconfirmed, reappears (the ball is seen again
  before the crossing), too_far (heading line beyond OUT_MAX_RANGE_M), slow (below
  HEADING_MIN_SPEED), outside (last seen beyond a line already). The script checks that the
  "ok" frames are out onset rows of the engine.
- Stages of pursuit_diagnose (furthest first): ok, unconfirmed (every test but the collective
  stop), chased (a kicking-team player reaches the ball within COLLECTIVE_STOP_S),
  dist_not_rising, no_drop (the nearest players never fall below PURSUIT_STOP_SPEED within
  ONSET_CONFIRM_S), contested_kick (mean team distances closer than CONTESTED_MARGIN_M, or
  no team distance at all),
  already_slow, kick_unseen. The script checks that the "ok" frames equal the pursuit_stop rows
  of the engine.
- Anchors. "contested": the contested rows of the engine event table on graded frames, and the
  dribble rows with no team (a contested dribble, the "contested" cause of
  shape_evidence.py). "pair": a contact row with a team (touch, release, control, dribble)
  whose previous contact row with a team is of the other team at most PAIR_S before it (both
  teams at the ball within 0.5 s); the anchor is the later row. "contested_or_pair" is the
  union of the two. "both_near" is no contact: the first frame of each run of frames with the
  ball visible and the nearest H and the nearest A both within R_TOUCH of it (feats dH, dA),
  on graded frames. It asks whether the physics of a challenge is there when the detector
  gives no contested row.
- Signature at an anchor c, from meanv_all (feats): pre = mean over [c - PRE_S, c), min2 = the
  minimum over (c, c + SIG_S], fall = 1 - min2 / pre. It fires at X when min2 < V_STOP or
  fall > X, X in FALL_GRID ("vstop" alone: min2 < V_STOP). Also stored: the means over
  (c, c + 2 s] and (c, c + 4 s], the ball speed at c and c + 1 s (NaN when missing), sep at c.
  "Confirmed": a collective-stop run overlaps [c, c + ONSET_CONFIRM_S] (deadball._confirmed),
  the test the engine applies to a missing-ball out and a pursuit stop.
- Truth of an anchor against a reference: "true" when a reference onset lies in
  [c - MATCH_TOL, c + LATE_TOL] (the onset nearest c is stored with its kind and o - c), "dead"
  when not true and the reference label at c is D, else "false". The negative class of the
  foul-signature table is every anchor with no reference onset in [c - LATE_TOL, c + LATE_TOL].
- Engine state at an anchor: "dead" when the engine label at c is D, "late_entry" when an engine
  onset lies in (c, c + LATE_TOL], else "alive".
- Precision of a signature: true firings over all firings. Recall: reference onsets (of a kind)
  with a firing anchor in [o - LATE_TOL, o + MATCH_TOL], over the onsets of that kind. "New"
  precision counts only the firings where the engine state is alive. False firings per alive
  minute: false firings over the reference alive frames (H or A) in minutes.
- Onset records (every reference onset, the foul tables read the kind foul). The nearest
  contested anchor at or before o within FOUL_LOOKBACK (c - o, negative), the nearest after
  (within LATE_TOL), the signature at that anchor, the nearest both_near frame at or before o
  within FOUL_LOOKBACK, and at o itself: the means of meanv_all over [o - PRE_S, o),
  (o, o + 2 s] and (o, o + 4 s], the ball speed at o and o + 1 s, sep at o, and the delays
  (frames after o, up to FOUL_LOOKBACK) to the first frame with meanv_all < V_STOP and to the
  first frame with meanv_all below half the mean over [o - PRE_S, o).
- BAR-ATH. Label-free on the full match. With `--public-gt` the script reads the public file at
  the anchors inside the public window PUBLIC_WINDOW only: the public label at c and the public
  onsets in [c - MATCH_TOL, c + LATE_TOL]. With `--f1-doc` it reads the appendix table of
  docs/plan/research/DEADBALL-FORENSICS.md (the D segments of c8_bump) and marks each alive
  anchor that lies in a segment or within LATE_TOL of its start, and each anchor within
  LATE_TOL of a named F1 case (F1_CASES).

The script stores counts and per-onset or per-anchor summary numbers (frames, means), never a
tracking row. numpy and pandas only. Deterministic.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "tools", "analysis")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

SEALED = ("DFL-J03WQQ", "DFL-J03WR9")       # calibration/SPLITS.md
PUBLIC_WINDOW = (11185, 18684)              # team rules, video frames, inclusive
MATCH_TOL = 25        # frames: "matched" (the scorer's tolerance)
LATE_TOL = 125        # frames: "late" up to 5 s, and the negative-class distance of R3-C
FOUL_LOOKBACK = 250   # frames: nearest contested contact before a foul onset (10 s)
PRE_S = 4.0           # s: the "4 s before" of R3-C item 2
SIG_S = 2.0           # s: "within 2 s" of the signature
POST_S = (2.0, 4.0)   # s: the "next 2 s and 4 s" of R3-C item 2
BALL_LATER_S = 1.0    # s: ball speed "1 s later"
PAIR_S = 0.5          # s: "contacts of both teams within 0.5 s" (RESEARCH-3 R3-C)
FALL_GRID = (0.2, 0.3, 0.4, 0.5, 0.6)       # fractions of the pre level
DFL_EXIT_TOL = 100    # frames: restart_team.DFL_EVENT_TOL
DFL_STOP_BEFORE = 250  # frames: a Foul or Offside event up to 10 s before the onset
TEAMS = ("H", "A")
TEAM_KINDS = ("touch", "release", "control", "dribble")
ANCHOR_SETS = ("contested", "pair", "contested_or_pair", "both_near")
STATUSES = ("matched", "late", "early", "inside", "missed")
OUT_STAGES = ("ok", "unconfirmed", "reappears", "too_far", "slow", "outside")
PURSUIT_STAGES = ("ok", "unconfirmed", "chased", "dist_not_rising", "no_drop",
                  "contested_kick", "already_slow", "kick_unseen")
CAUSES = ("entry_no_edge", "out_unconfirmed", "pursuit_unconfirmed", "collective_hint",
          "pursuit_chased", "pursuit_dist_not_rising", "pursuit_no_drop",
          "pursuit_contested_kick", "pursuit_already_slow", "pursuit_kick_unseen",
          "gap_no_stop", "no_stop_4s", "stop_4s_touched")
# Named cases of docs/plan/research/DEADBALL-FORENSICS.md (F1) outside its appendix table:
# sections 0, 3, 4 and 6 (frames of c8_bump on BAR-ATH).
F1_CASES = (24044, 77859, 82254, 86635, 88699, 119728, 129388, 133932, 142371, 143837,
            148633, 175384, 187146)
PUBLIC_CONTESTED = (11253, 13112, 15290)    # RESEARCH-3 R3-C item 4: alive in the public GT


# -- pure parts (tested on synthetic arrays) --------------------------------------------------

def onsets_of(labels):
    """int64 array of the frames t with labels[t] == D and labels[t - 1] in H, A."""
    labels = np.asarray(labels).astype("<U1")
    if labels.size < 2:
        return np.zeros(0, dtype=np.int64)
    prev_alive = (labels[:-1] == "H") | (labels[:-1] == "A")
    return (np.flatnonzero(prev_alive & (labels[1:] == "D")) + 1).astype(np.int64)


def exit_of(labels, onset):
    """The first frame after the D run that starts at `onset`."""
    labels = np.asarray(labels).astype("<U1")
    t = int(onset)
    while t < labels.size and labels[t] == "D":
        t += 1
    return t


def onset_status(onset, eng_onsets, eng_label):
    """(status, offset or None) of one reference onset (module docstring).

    `eng_onsets` is the sorted engine onsets, `eng_label` the engine label at the onset.
    """
    eng = np.asarray(eng_onsets, dtype=np.int64)
    lo = int(np.searchsorted(eng, int(onset) - LATE_TOL, side="left"))
    hi = int(np.searchsorted(eng, int(onset) + LATE_TOL, side="right"))
    if hi > lo:
        cand = eng[lo:hi]
        off = cand - int(onset)
        j = int(np.argmin(np.abs(off)))      # argmin takes the first: the earlier onset on a tie
        d = int(off[j])
        if abs(d) <= MATCH_TOL:
            return "matched", d
        return ("late" if d > 0 else "early"), d
    return ("inside" if str(eng_label) == "D" else "missed"), None


def window_mean(values, lo, hi):
    """nanmean of values[lo:hi] clipped to the array, NaN when no finite value."""
    values = np.asarray(values, dtype=np.float64)
    lo, hi = max(0, int(lo)), min(values.size, int(hi))
    if hi <= lo:
        return float("nan")
    seg = values[lo:hi]
    seg = seg[np.isfinite(seg)]
    return float(seg.mean()) if seg.size else float("nan")


def window_min(values, lo, hi):
    """nanmin of values[lo:hi] clipped to the array, NaN when no finite value."""
    values = np.asarray(values, dtype=np.float64)
    lo, hi = max(0, int(lo)), min(values.size, int(hi))
    if hi <= lo:
        return float("nan")
    seg = values[lo:hi]
    seg = seg[np.isfinite(seg)]
    return float(seg.min()) if seg.size else float("nan")


def signature(meanv, c, fps):
    """{pre, min2, fall, post2, post4}: the slowdown numbers at anchor c (module docstring)."""
    pre_n = int(round(PRE_S * fps))
    sig_n = int(round(SIG_S * fps))
    pre = window_mean(meanv, c - pre_n, c)
    min2 = window_min(meanv, c + 1, c + sig_n + 1)
    fall = 1.0 - min2 / pre if np.isfinite(pre) and pre > 0 and np.isfinite(min2) else float("nan")
    out = {"pre": pre, "min2": min2, "fall": fall}
    for s in POST_S:
        out["post{:d}".format(int(s))] = window_mean(meanv, c + 1, c + int(round(s * fps)) + 1)
    return out


def fires(sig, v_stop, x=None):
    """True when the signature fires: min2 < v_stop, or (x given) fall > x."""
    min2, fall = sig.get("min2"), sig.get("fall")
    hit = min2 is not None and np.isfinite(min2) and min2 < v_stop
    if x is not None and fall is not None and np.isfinite(fall) and fall > x:
        hit = True
    return bool(hit)


def pair_anchors(frames, teams, width):
    """Sorted int list: frames of a team contact whose previous team contact is of the other team
    at most `width` frames before (both teams at the ball)."""
    rows = sorted((int(f), str(t)) for f, t in zip(frames, teams) if t in TEAMS)
    out = set()
    for (f0, t0), (f1, t1) in zip(rows[:-1], rows[1:]):
        if t1 != t0 and f1 - f0 <= int(width):
            out.add(f1)
    return sorted(out)


def metrica_kind(stops):
    """(kind, detail) from the stopping events at one onset (dicts of tools/events_gt)."""
    if not stops:
        return "no_stop", "-"
    names = []
    for e in stops:
        subs = [s for s in e["subtypes"] if s not in ("WON", "LOST")]
        names.append(" ".join([e["type"]] + subs))
    detail = " + ".join(sorted(set(names)))

    def has(test):
        return any(test(e) for e in stops)

    if has(lambda e: e["type"] == "FAULT RECEIVED"
           or (e["type"] == "CHALLENGE" and "FAULT" in e["subtypes"])):
        kind = "foul"
        style = [s for e in stops if e["type"] == "CHALLENGE"
                 for s in e["subtypes"] if s in ("TACKLE", "GROUND", "AERIAL")]
        detail = "CHALLENGE " + "/".join(sorted(set(style))) if style else "FAULT RECEIVED only"
    elif has(lambda e: "OFFSIDE" in e["subtypes"]):
        kind = "offside"
    elif has(lambda e: "HAND BALL" in e["subtypes"]):
        kind = "hand_ball"
    elif has(lambda e: e["type"] == "SHOT" and "GOAL" in e["subtypes"]):
        kind = "goal"
    elif has(lambda e: e["type"] == "BALL OUT" or (e["type"] == "SHOT" and "OUT" in e["subtypes"])):
        kind = "out"
    else:
        kind = "other"
    return kind, detail


def dfl_kind(onset, exit_frame, restart, fouls, offsides):
    """(kind, detail, foul offset or None) of one DFL onset (module docstring).

    `restart` is the restart kind of restart_team.DFL_KINDS values or "none"; `fouls` and
    `offsides` are sorted frame lists of the Foul and Offside events.
    """
    lo, hi = int(onset) - DFL_STOP_BEFORE, int(exit_frame)

    def first_in(frames):
        inside = [int(f) for f in frames if lo <= int(f) <= hi]
        if not inside:
            return None
        return min(inside, key=lambda f: (abs(f - int(onset)), f))

    if restart in ("throw_in", "corner", "goal_kick"):
        return "out", restart, None
    if restart == "kick_off":
        return "goal", restart, None
    if restart == "drop_ball":
        return "drop_ball", restart, None
    foul = first_in(fouls)
    if restart == "penalty":
        return "foul", "penalty", (foul - int(onset)) if foul is not None else None
    off = first_in(offsides)
    if off is not None:
        return "offside", restart, off - int(onset)
    if foul is not None:
        return "foul", restart, foul - int(onset)
    return ("free_kick_no_event" if restart == "free_kick" else "no_event"), restart, None


def primary_cause(flags):
    """The primary cause of a missed onset from its flags (module docstring)."""
    if flags["entry"]:
        return "entry_no_edge"
    if flags["out"] == "unconfirmed":
        return "out_unconfirmed"
    if flags["pursuit"] == "unconfirmed":
        return "pursuit_unconfirmed"
    if flags["hint"]:
        return "collective_hint"
    if flags["pursuit"] not in ("none", "ok"):
        return "pursuit_" + flags["pursuit"]
    if flags["vanished"] and not flags["stop_4s"]:
        return "gap_no_stop"
    if not flags["stop_4s"]:
        return "no_stop_4s"
    return "stop_4s_touched"


def furthest(stages, order):
    """The stage of `stages` that comes first in `order`, or "none"."""
    found = [s for s in order if s in set(stages)]
    return found[0] if found else "none"


def truth_of(c, ref_onsets, ref_label_at_c):
    """(truth, nearest onset or None): true / dead / false of one anchor (module docstring)."""
    ref = np.asarray(ref_onsets, dtype=np.int64)
    lo = int(np.searchsorted(ref, int(c) - MATCH_TOL, side="left"))
    hi = int(np.searchsorted(ref, int(c) + LATE_TOL, side="right"))
    if hi > lo:
        return "true", int(ref[lo])
    return ("dead" if str(ref_label_at_c) == "D" else "false"), None


def near_any(c, frames, tol):
    """True when a frame of the sorted `frames` lies within tol of c."""
    frames = np.asarray(frames, dtype=np.int64)
    lo = int(np.searchsorted(frames, int(c) - tol, side="left"))
    hi = int(np.searchsorted(frames, int(c) + tol, side="right"))
    return hi > lo


def quantiles(values, qs=(10, 50, 90)):
    """{n, p10, p50, p90} of the finite values (n only when empty)."""
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    values = values[np.isfinite(values)]
    out = {"n": int(values.size)}
    for q in qs:
        out["p{}".format(q)] = float(np.percentile(values, q)) if values.size else None
    return out


def score_signature(anchors, onsets, x, confirmed_only=False, kinds=None, new_only=False):
    """{fire, true, false, dead, precision, recall_n, recall_d, offsets} of one signature.

    `anchors` are anchor records (dicts with sig, truth, confirmed, engine, frame); `onsets` are
    reference onset records (frame, kind). `x` None means the V_STOP clause alone. `kinds`
    restricts the recall to these onset kinds (None: all). `new_only` counts only the firings
    where the engine state is alive (precision and recall alike).
    """
    v_stop = None
    firing = []
    for a in anchors:
        v_stop = a["v_stop"]
        if not fires(a["sig"], v_stop, x):
            continue
        if confirmed_only and not a["confirmed"]:
            continue
        if new_only and a["engine"] != "alive":
            continue
        firing.append(a)
    true = [a for a in firing if a["truth"] == "true"]
    false = [a for a in firing if a["truth"] == "false"]
    dead = [a for a in firing if a["truth"] == "dead"]
    frames = np.array(sorted(a["frame"] for a in firing), dtype=np.int64)
    wanted = [o for o in onsets if kinds is None or o["kind"] in kinds]
    hit = 0
    for o in wanted:
        lo = int(np.searchsorted(frames, o["frame"] - LATE_TOL, side="left"))
        hi = int(np.searchsorted(frames, o["frame"] + MATCH_TOL, side="right"))
        hit += int(hi > lo)
    return {"fire": len(firing), "true": len(true), "false": len(false), "dead": len(dead),
            "precision": (len(true) / len(firing)) if firing else None,
            "recall_n": hit, "recall_d": len(wanted),
            "offsets": [a["ref_onset"] - a["frame"] for a in true],
            "offsets_foul": [a["ref_onset"] - a["frame"] for a in true
                             if a.get("ref_kind") == "foul"]}


# -- engine replicas with reasons (checked against the engine on every run) -----------------

def out_diagnose(ball, cfg, pitch, fps, stop_starts, stop_stops, horizon):
    """[(last_visible, onset or None, stage)]: deadball._out_onsets part (b) with its reasons."""
    from engine.deadball import _boundary_distance, _confirmed
    from engine.tracks import _runs

    rows = []
    length, width = pitch
    x, y, visible = ball["x"], ball["y"], ball["visible"]
    n = visible.shape[0]
    run_starts, run_stops = _runs(visible)
    for idx, stop in enumerate(run_stops):
        last = int(stop) - 1
        if stop >= n:
            continue
        next_visible = int(run_starts[idx + 1]) if idx + 1 < len(run_starts) else n
        if not (0.0 <= x[last] <= length and 0.0 <= y[last] <= width):
            rows.append((last, None, "outside"))
            continue
        speed = ball["speed"][last]
        if not np.isfinite(speed) or speed < cfg.HEADING_MIN_SPEED:
            rows.append((last, None, "slow"))
            continue
        ux, uy = ball["vx"][last] / speed, ball["vy"][last] / speed
        dist = _boundary_distance(x[last], y[last], ux, uy, pitch)
        if dist > cfg.OUT_MAX_RANGE_M:
            rows.append((last, None, "too_far"))
            continue
        decel = cfg.FRICTION_DECEL
        if speed * speed >= 2.0 * decel * dist:
            seconds = (speed - np.sqrt(speed * speed - 2.0 * decel * dist)) / decel
            onset = last + int(round(seconds * fps))
        else:
            onset = last + int(cfg.DELTA_OUT_LAG)
        if onset >= next_visible:
            rows.append((last, int(onset), "reappears"))
            continue
        ok = _confirmed(onset, stop_starts, stop_stops, horizon)
        rows.append((last, int(onset), "ok" if ok else "unconfirmed"))
    return rows


def pursuit_diagnose(ball, feats, cfg, stop_starts, stop_stops, horizon):
    """[(run_start, onset or None, stage)]: deadball._pursuit_onsets with the failing test."""
    from engine.deadball import _above, _below, _confirmed, _first_peak, _fps, _frames
    from engine.tracks import _runs

    rows = []
    speed = ball["speed"]
    near = feats["meanv_near"]
    n = speed.shape[0]
    chase = _frames(cfg.COLLECTIVE_STOP_S, _fps(ball))
    fast = ball["visible"] & _above(speed, cfg.PURSUIT_MIN_BALL_SPEED)
    starts, stops = _runs(fast)
    for start, stop in zip(starts, stops):
        start, stop = int(start), int(stop)
        lo = max(0, start - int(cfg.PURSUIT_KICK_LOOKBACK))
        if not ball["visible"][lo:start].all() or start == 0:
            rows.append((start, None, "kick_unseen"))
            continue
        peak = _first_peak(speed, start, stop)
        before = near[lo:start + 1]
        if not np.isfinite(before).any() or np.nanmax(before) < cfg.PURSUIT_STOP_SPEED:
            rows.append((start, None, "already_slow"))
            continue
        with np.errstate(invalid="ignore"):
            mean_h = np.nanmean(feats["dH"][lo:start + 1]) if np.isfinite(
                feats["dH"][lo:start + 1]).any() else np.inf
            mean_a = np.nanmean(feats["dA"][lo:start + 1]) if np.isfinite(
                feats["dA"][lo:start + 1]).any() else np.inf
        if not np.isfinite(min(mean_h, mean_a)) or abs(mean_h - mean_a) < cfg.CONTESTED_MARGIN_M:
            rows.append((start, None, "contested_kick"))   # or no team distance at all
            continue
        d_kick = feats["dH"] if mean_h <= mean_a else feats["dA"]
        end = min(n, start + horizon)
        slow = np.flatnonzero(_below(near[start:end], cfg.PURSUIT_STOP_SPEED))
        if slow.size == 0:
            rows.append((start, None, "no_drop"))
            continue
        drop = start + int(slow[0])
        descent = drop
        while descent - 1 >= 0 and np.isfinite(near[descent - 1]) and \
                near[descent - 1] > near[descent]:
            descent -= 1
        onset = max(peak, descent)
        if not (np.isfinite(d_kick[onset]) and np.isfinite(d_kick[start])
                and d_kick[onset] > d_kick[start]):
            rows.append((start, int(onset), "dist_not_rising"))
            continue
        after = d_kick[onset:min(n, onset + chase)]
        if np.isfinite(after).any() and np.nanmin(after) <= cfg.R_TOUCH:
            rows.append((start, int(onset), "chased"))
            continue
        ok = _confirmed(onset, stop_starts, stop_stops, horizon)
        rows.append((start, int(onset), "ok" if ok else "unconfirmed"))
    return rows


# -- collect --------------------------------------------------------------------------------

def _f(value):
    """A float for JSON: None for NaN."""
    value = float(value)
    return value if np.isfinite(value) else None


def _at(values, t):
    """values[t] as a JSON float, None outside the array or NaN."""
    values = np.asarray(values, dtype=np.float64)
    return _f(values[t]) if 0 <= int(t) < values.size else None


def labels_md5(labels):
    """md5 of the labels written as engine.submission writes them (frame,label rows)."""
    text = "frame,label\n" + "".join("{},{}\n".format(i, v) for i, v in enumerate(labels))
    return hashlib.md5(text.encode("ascii")).hexdigest()


def read_labels(path, n_frames=None):
    """The `label` column of a `frame,label` CSV as an array of one-letter strings."""
    import pandas as pd

    table = pd.read_csv(path, dtype={"frame": np.int64, "label": str})
    labels = table.sort_values("frame")["label"].to_numpy().astype("<U1")
    if n_frames is not None and labels.size != n_frames:
        raise ValueError("{}: {} rows, expected {}".format(path, labels.size, n_frames))
    return labels


def metrica_stop_events(path, game):
    """[(end video frame, event dict)] of the stopping events (tools/events_gt.is_stop)."""
    import events_gt
    import games

    events = events_gt.to_video(events_gt.load_any(path), games.GAMES[game]["offsets"])
    return [(int(e["end"]), e) for e in events
            if events_gt.is_stop(e) and events_gt.END_HALF not in e["subtypes"]]


def dfl_event_frames(events_path, positions_path, game, tags=("Foul", "Offside")):
    """{tag: sorted video frames} of the DFL events with these child tags (restart_team time)."""
    import games
    from tools.analysis import restart_team as rt

    entry = games.GAMES[game]
    fps = float(entry["fps"])
    starts = rt.dfl_half_starts(positions_path)
    out = {tag: [] for tag in tags}
    for event in ET.parse(events_path).getroot().iter("Event"):
        child = next((c for c in event if c.tag in tags), None)
        if child is None:
            continue
        when = rt._parse_time(event.get("EventTime"))
        half = 2 if 2 in starts and when >= starts[2][1] else 1
        first_n, first_t = starts[half]
        frame_n = first_n + int(round((when - first_t).total_seconds() * fps))
        out[child.tag].append(int(frame_n - int(entry["offsets"][half])))
    return {tag: sorted(v) for tag, v in out.items()}


def f1_segments(path):
    """[(start, end)] of the appendix table of DEADBALL-FORENSICS.md (rows "| start | end |")."""
    out = []
    with open(path) as handle:
        text = handle.read()
    part = text.split("## Appendix", 1)[1] if "## Appendix" in text else ""
    for match in re.finditer(r"^\| (\d+) \| (\d+) \|", part, flags=re.M):
        out.append((int(match.group(1)), int(match.group(2))))
    return out


def collect(args):
    """The `collect` command: onset and anchor records for one match."""
    import games
    from engine import deadball, possession
    from engine.config import Config
    from engine.tracks import _runs
    from tools.analysis import restart_team as rt

    game = args.game
    if game in SEALED:
        raise SystemExit("{} is sealed (calibration/SPLITS.md)".format(game))
    if args.public_gt and game != "BAR-ATH":
        raise SystemExit("--public-gt is for BAR-ATH only")
    if game == "BAR-ATH" and args.ref:
        raise SystemExit("BAR-ATH has no full reference: use --public-gt")
    cfg = Config.default()
    inputs = possession.prepare(cfg, game)
    labels, parts = possession.run(cfg, return_parts=True, inputs=inputs)
    labels = np.asarray(labels).astype("<U1")
    n = labels.size
    blocks = inputs["blocks"]
    fps = float(inputs["fps"])
    ball, feats = inputs["ball"], inputs["feats"]
    pitch = possession.match_inputs(game)[4]
    meanv = np.asarray(feats["meanv_all"], dtype=np.float64)
    sep = np.asarray(feats["sep"], dtype=np.float64)
    bspeed = np.where(ball["visible"], np.asarray(ball["speed"], dtype=np.float64), np.nan)
    graded = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        graded[int(first):int(last) + 1] = True
    later = int(round(BALL_LATER_S * fps))
    horizon = deadball._frames(cfg.ONSET_CONFIRM_S, fps)
    stop_starts, stop_stops = deadball._stop_runs(feats, cfg, fps)

    # Engine onsets and their entry mechanism.
    eng_onsets = onsets_of(labels)
    stoppage = np.asarray(parts["stoppage"], dtype=bool)
    st_starts, _ = _runs(stoppage)
    table = inputs["onsets"]
    rows_by = {}
    for f, mech in zip(table["frame"], table["mechanism"]):
        rows_by.setdefault(int(f), set()).add(str(mech))
    high = np.asarray(inputs["e_dead"]) > cfg.DEAD_EVIDENCE_HIGH
    h_starts, h_stops = _runs(high)
    dwell = max(1, int(round(cfg.COLLECTIVE_STOP_S * fps)))
    dwell_entries = set(int(s) for s, e in zip(h_starts, h_stops) if e - s >= dwell)

    def entry_mechanism(e):
        near = st_starts[np.abs(st_starts - e) <= MATCH_TOL] if st_starts.size else st_starts
        if near.size == 0:
            return "decoder"
        s = int(near[np.argmin(np.abs(near - e))])
        mechs = rows_by.get(s, set())
        for m in ("out", "pursuit_stop"):
            if m in mechs:
                return m
        return "evidence" if s in dwell_entries else "other"

    eng_mech = {int(e): entry_mechanism(int(e)) for e in eng_onsets}

    # Replicas with reasons, checked against the engine.
    outs = out_diagnose(ball, cfg, pitch, fps, stop_starts, stop_stops, horizon)
    purs = pursuit_diagnose(ball, feats, cfg, stop_starts, stop_stops, horizon)
    eng_out = set(int(f) for f, m in zip(table["frame"], table["mechanism"]) if m == "out")
    eng_pur = set(int(f) for f, m in zip(table["frame"], table["mechanism"])
                  if m == "pursuit_stop")
    ok_out = set(o for _, o, s in outs if s == "ok")
    ok_pur = set(o for _, o, s in purs if s == "ok")
    if not ok_out <= eng_out:
        raise RuntimeError("out_diagnose finds an out onset the engine does not have")
    if ok_pur != eng_pur:
        raise RuntimeError("pursuit_diagnose differs from the engine's pursuit_stop rows")
    hint_frames = np.array(sorted(int(f) for f, m in zip(table["frame"], table["mechanism"])
                                  if m == "collective_stop"), dtype=np.int64)
    entry_frames = np.array(sorted(eng_out | eng_pur | dwell_entries), dtype=np.int64)
    out_last = np.array([r[0] for r in outs], dtype=np.int64)
    pur_start = np.array([r[0] for r in purs], dtype=np.int64)
    beyond = deadball.outside_pitch(ball, pitch, cfg.PITCH_MARGIN_M)
    visible = np.asarray(ball["visible"], dtype=bool)
    collective = deadball._collective(feats, cfg)

    def in_range(frames, lo, hi):
        a = int(np.searchsorted(frames, lo, side="left"))
        b = int(np.searchsorted(frames, hi, side="right"))
        return a, b

    def miss_flags(o):
        lo, hi = o - MATCH_TOL, o + LATE_TOL
        a, b = in_range(entry_frames, lo, hi)
        entry = b > a
        a, b = in_range(hint_frames, lo, hi)
        hint = b > a
        a, b = in_range(out_last, o - LATE_TOL, o + MATCH_TOL)
        out_stage = furthest([outs[i][2] for i in range(a, b)], OUT_STAGES)
        if out_stage == "none" and beyond[max(0, lo):min(n, hi + 1)].any():
            out_stage = "seen"
        a, b = in_range(pur_start, o - LATE_TOL, o + MATCH_TOL)
        pur_stage = furthest([purs[i][2] for i in range(a, b)], PURSUIT_STAGES)
        stop_4s = deadball._confirmed(o, stop_starts, stop_stops, horizon)
        slow_4s = bool(np.isfinite(meanv[o:o + horizon + 1]).any()
                       and np.nanmin(meanv[o:o + horizon + 1]) < cfg.V_STOP)
        vanished = bool((~visible[max(0, lo):min(n, hi + 1)]).any())
        flags = {"entry": bool(entry), "hint": bool(hint), "out": out_stage,
                 "pursuit": pur_stage, "stop_4s": bool(stop_4s), "slow_4s": slow_4s,
                 "vanished": vanished,
                 "collective_frames_4s": int(collective[o:o + horizon + 1].sum())}
        flags["cause"] = primary_cause(flags)
        return flags

    ref = None
    if args.ref:
        ref = read_labels(args.ref, int(games.GAMES[game]["n_video_frames"]))
    public = read_labels(args.public_gt, n) if args.public_gt else None

    # Reference onsets with kinds and engine status.
    onset_rows = []
    ref_onsets = np.zeros(0, dtype=np.int64)
    if ref is not None:
        ref_onsets = onsets_of(ref)
        stops = metrica_stop_events(args.metrica_events, game) if args.metrica_events else []
        stop_ends = np.array([s[0] for s in stops], dtype=np.int64)
        dfl_restarts, dfl_ev = {}, None
        if args.dfl_events:
            dfl_restarts = rt.dfl_kinds(args.dfl_events, args.dfl_positions, game)
            dfl_ev = dfl_event_frames(args.dfl_events, args.dfl_positions, game)
        for o in ref_onsets:
            o = int(o)
            ex = exit_of(ref, o)
            rec = {"frame": o, "exit": ex, "len": ex - o}
            if args.metrica_events:
                at = [s[1] for s in stops if s[0] == o]
                if not at and stop_ends.size:
                    d = np.abs(stop_ends - o)
                    if d.min() <= MATCH_TOL:
                        at = [s[1] for s in stops if abs(s[0] - o) == d.min()]
                kind, detail = metrica_kind(at)
                rec.update({"kind": kind, "detail": detail, "foul_offset": None})
            elif dfl_ev is not None:
                restart, _, _ = rt.kind_of_exit(dfl_restarts, ex, DFL_EXIT_TOL)
                kind, detail, foff = dfl_kind(o, ex, restart, dfl_ev["Foul"],
                                              dfl_ev["Offside"])
                rec.update({"kind": kind, "detail": detail, "foul_offset": foff})
            else:
                rec.update({"kind": "unknown", "detail": "-", "foul_offset": None})
            rec["geometry"] = rt.out_geometry(ball, o, pitch, cfg)[2]
            status, off = onset_status(o, eng_onsets, labels[o])
            rec.update({"status": status, "offset": off})
            if status in ("matched", "late", "early"):
                rec["engine_mechanism"] = eng_mech[o + off]
            if status == "missed":
                rec["flags"] = miss_flags(o)
            onset_rows.append(rec)

    # Anchors.
    ev = parts["events"]
    kinds = ev["kind"].to_numpy(dtype=object)
    frames = ev["frame"].to_numpy(dtype=np.int64)
    teams = ev["team"].to_numpy(dtype=object)
    contested = sorted(set(int(f) for f, k, t in zip(frames, kinds, teams)
                           if (k == "contested" or (k == "dribble" and t not in TEAMS))
                           and graded[int(f)]))
    team_rows = [(int(f), t) for f, k, t in zip(frames, kinds, teams)
                 if k in TEAM_KINDS and t in TEAMS and graded[int(f)]]
    pairs = pair_anchors([f for f, _ in team_rows], [t for _, t in team_rows],
                         int(round(PAIR_S * fps)))
    both = (visible & ~deadball._above(feats["dH"], cfg.R_TOUCH) & np.isfinite(feats["dH"])
            & ~deadball._above(feats["dA"], cfg.R_TOUCH) & np.isfinite(feats["dA"]) & graded)
    both_starts = [int(s) for s in _runs(both)[0]]
    anchor_sets = {"contested": contested, "pair": pairs,
                   "contested_or_pair": sorted(set(contested) | set(pairs)),
                   "both_near": both_starts}
    ref_kind = {r["frame"]: r["kind"] for r in onset_rows}
    pub_on = np.zeros(0, dtype=np.int64)
    if public is not None:
        pub_on = onsets_of(public)
        pub_on = pub_on[(pub_on >= PUBLIC_WINDOW[0]) & (pub_on <= PUBLIC_WINDOW[1])]
    eng_sorted = np.asarray(eng_onsets, dtype=np.int64)

    def anchor_record(c):
        sig = signature(meanv, c, fps)
        a, b = in_range(eng_sorted, c + 1, c + LATE_TOL)
        engine_state = "dead" if labels[c] == "D" else ("late_entry" if b > a else "alive")
        rec = {"frame": int(c), "half": int(next(h for h, f, l in blocks if f <= c <= l)),
               "sig": {k: _f(v) for k, v in sig.items()},
               "confirmed": bool(deadball._confirmed(c, stop_starts, stop_stops, horizon)),
               "engine": engine_state,
               "engine_onset_after": (int(eng_sorted[a]) - int(c)) if b > a else None,
               "ball_c": _at(bspeed, c), "ball_1s": _at(bspeed, c + later),
               "sep_c": _at(sep, c), "v_stop": float(cfg.V_STOP)}
        if ref is not None:
            truth, near = truth_of(c, ref_onsets, ref[c])
            rec.update({"truth": truth, "ref_onset": near,
                        "ref_kind": ref_kind.get(near) if near is not None else None,
                        "negative": not near_any(c, ref_onsets, LATE_TOL)})
        if public is not None and PUBLIC_WINDOW[0] <= c <= PUBLIC_WINDOW[1]:
            rec["public_label"] = str(public[c])
            rec["public_onsets"] = [int(p) - int(c) for p in pub_on
                                    if c - MATCH_TOL <= p <= c + LATE_TOL]
        return rec

    anchors = {name: [anchor_record(c) for c in frames_] for name, frames_ in anchor_sets.items()}

    # Foul records.
    cont = np.array(contested, dtype=np.int64)
    both_arr = np.array(both_starts, dtype=np.int64)
    pre_n = int(round(PRE_S * fps))
    fouls = []
    for rec in onset_rows:
        o = rec["frame"]
        a, b = in_range(cont, o - FOUL_LOOKBACK, o)
        before = int(cont[b - 1]) if b > a else None
        a2, b2 = in_range(cont, o + 1, o + LATE_TOL)
        after = int(cont[a2]) if b2 > a2 else None
        row = {"frame": o, "kind": rec["kind"], "detail": rec["detail"],
               "status": rec["status"], "offset": rec["offset"],
               "contested_before": (before - o) if before is not None else None,
               "contested_after": (after - o) if after is not None else None,
               "pre_o": _f(window_mean(meanv, o - pre_n, o)),
               "post2_o": _f(window_mean(meanv, o + 1, o + int(round(2 * fps)) + 1)),
               "post4_o": _f(window_mean(meanv, o + 1, o + int(round(4 * fps)) + 1)),
               "ball_o": _at(bspeed, o), "ball_1s": _at(bspeed, o + later),
               "sep_o": _at(sep, o)}
        if before is not None:
            sig = signature(meanv, before, fps)
            row["sig_before"] = {k: _f(v) for k, v in sig.items()}
            row["confirmed_before"] = bool(deadball._confirmed(before, stop_starts, stop_stops,
                                                               horizon))
        a3, b3 = in_range(both_arr, o - FOUL_LOOKBACK, o)
        row["both_near_before"] = (int(both_arr[b3 - 1]) - o) if b3 > a3 else None
        seg = meanv[o:min(n, o + FOUL_LOOKBACK + 1)]
        slow = np.flatnonzero(np.isfinite(seg) & (seg < cfg.V_STOP))
        row["delay_vstop"] = int(slow[0]) if slow.size else None
        half_level = row["pre_o"] / 2.0 if row["pre_o"] is not None else None
        hl = np.flatnonzero(np.isfinite(seg) & (seg < half_level)) if half_level else []
        row["delay_half"] = int(hl[0]) if len(hl) else None
        fouls.append(row)

    alive_ref = int((graded & np.isin(ref, list(TEAMS))).sum()) if ref is not None else None
    alive_eng = int((graded & np.isin(labels, list(TEAMS))).sum())
    alive_eng_halves = {}
    for half, first, last in blocks:
        seg = labels[int(first):int(last) + 1]
        alive_eng_halves[str(half)] = int(np.isin(seg, list(TEAMS)).sum())
    out = {"game": game, "engine_md5": labels_md5(labels), "fps": fps,
           "n_engine_onsets": int(eng_onsets.size),
           "engine_mechanisms": {m: int(sum(1 for v in eng_mech.values() if v == m))
                                 for m in sorted(set(eng_mech.values()))},
           "alive_ref_frames": alive_ref, "alive_engine_frames": alive_eng,
           "alive_engine_halves": alive_eng_halves,
           "n_ref_onsets": int(ref_onsets.size), "onsets": onset_rows, "fouls": fouls,
           "anchors": anchors, "v_stop": float(cfg.V_STOP),
           "diagnose_check": {"out_ok": len(ok_out), "pursuit_ok": len(ok_pur),
                              "engine_out": len(eng_out), "engine_pursuit": len(eng_pur)}}
    if public is not None:
        out["public_onsets"] = [int(p) for p in pub_on]
    if args.f1_doc:
        segs = f1_segments(args.f1_doc)
        out["f1_segments"] = len(segs)
        for name in ANCHOR_SETS:
            for rec in anchors[name]:
                c = rec["frame"]
                hit = [s for s, e in segs if s - LATE_TOL <= c < e]
                rec["f1_segment"] = hit[0] if hit else None
                named = [k for k in F1_CASES if abs(k - c) <= LATE_TOL]
                rec["f1_case"] = named[0] if named else None
    write_json(args.out, out)
    return out


def write_json(path, payload):
    """Write `payload` as sorted, indented JSON (deterministic)."""
    folder = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.write("\n")


# -- report ---------------------------------------------------------------------------------

def _pct(num, den):
    """'12.3 %' or '-'."""
    return "-" if not den else "{:.1f} %".format(100.0 * num / den)


def _pc(num, den):
    """'12.3 % (num/den)' or '-'."""
    return "-" if not den else "{:.1f} % ({}/{})".format(100.0 * num / den, num, den)


def _fmt(value, digits=2):
    """A number for a table cell, `-` for None."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return "{:.{d}f}".format(value, d=digits)
    return str(value)


def _q(values, digits=1):
    """'p50 [p10, p90] (n)' of a list."""
    q = quantiles(values)
    if not q["n"]:
        return "- (0)"
    return "{} [{}, {}] ({})".format(_fmt(q["p50"], digits), _fmt(q["p10"], digits),
                                     _fmt(q["p90"], digits), q["n"])


def short(game):
    """A short column name of a match."""
    return game.replace("DFL-J03", "").replace("METRICA-", "")


def report_deficit(refs, lines):
    """Tables 1a to 1c: the onset deficit by kind and by cause."""
    lines.append("### 1a. Reference onsets by kind and engine status, per match")
    lines.append("")
    lines.append("Cells: matched / late / early / inside / missed of n.")
    lines.append("")
    kinds = sorted(set(r["kind"] for p in refs for r in p["onsets"]))
    lines.append("| kind | " + " | ".join(short(p["game"]) for p in refs) + " | pooled |")
    lines.append("|---|" + "---|" * (len(refs) + 1))
    for kind in kinds:
        cells, pool = [], {s: 0 for s in STATUSES}
        for p in refs:
            rows = [r for r in p["onsets"] if r["kind"] == kind]
            c = {s: sum(1 for r in rows if r["status"] == s) for s in STATUSES}
            for s in STATUSES:
                pool[s] += c[s]
            cells.append("{}/{}/{}/{}/{} of {}".format(*[c[s] for s in STATUSES], len(rows))
                         if rows else "-")
        total = sum(pool.values())
        cells.append("{}/{}/{}/{}/{} of {}".format(*[pool[s] for s in STATUSES], total))
        lines.append("| {} | ".format(kind) + " | ".join(cells) + " |")
    lines.append("")
    lines.append("### 1b. Pooled by provider and kind, with timing")
    lines.append("")
    lines.append("| provider | kind | n | matched | late | early | inside | missed | matched offset "
                 "p50 [p10, p90] | late offset p50 [p10, p90] |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for prov, test in (("Metrica", lambda g: g.startswith("METRICA")),
                       ("DFL", lambda g: g.startswith("DFL"))):
        group = [p for p in refs if test(p["game"])]
        for kind in kinds:
            rows = [r for p in group for r in p["onsets"] if r["kind"] == kind]
            if not rows:
                continue
            c = {s: sum(1 for r in rows if r["status"] == s) for s in STATUSES}
            mo = [r["offset"] for r in rows if r["status"] == "matched"]
            lo = [r["offset"] for r in rows if r["status"] == "late"]
            lines.append("| {} | {} | {} | ".format(prov, kind, len(rows))
                         + " | ".join("{} ({})".format(c[s], _pct(c[s], len(rows)))
                                      for s in STATUSES)
                         + " | {} | {} |".format(_q(mo, 0), _q(lo, 0)))
    lines.append("")
    lines.append("### 1c. Detail of the foul onsets (Metrica subtype; DFL restart kind)")
    lines.append("")
    lines.append("| provider | detail | n | matched | late | early | inside | missed |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for prov, test in (("Metrica", lambda g: g.startswith("METRICA")),
                       ("DFL", lambda g: g.startswith("DFL"))):
        rows = [r for p in refs if test(p["game"]) for r in p["onsets"] if r["kind"] == "foul"]
        for detail in sorted(set(r["detail"] for r in rows)):
            sub = [r for r in rows if r["detail"] == detail]
            lines.append("| {} | {} | {} | ".format(prov, detail, len(sub)) + " | ".join(
                str(sum(1 for r in sub if r["status"] == s)) for s in STATUSES) + " |")
    lines.append("")
    dfl_foul = [r["foul_offset"] for p in refs if p["game"].startswith("DFL")
                for r in p["onsets"] if r["kind"] == "foul"]
    lines.append("DFL Foul event frame minus the reference onset (fouls with an event): "
                 "p50 [p10, p90] (n) = {}.".format(_q(dfl_foul, 0)))
    lines.append("")
    lines.append("### 1d. Engine entry mechanism of the matched and late onsets")
    lines.append("")
    lines.append("| provider | kind | status | out | pursuit_stop | evidence | other | decoder |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for prov, test in (("Metrica", lambda g: g.startswith("METRICA")),
                       ("DFL", lambda g: g.startswith("DFL"))):
        for kind in kinds:
            for status in ("matched", "late"):
                rows = [r for p in refs if test(p["game"]) for r in p["onsets"]
                        if r["kind"] == kind and r["status"] == status]
                if not rows:
                    continue
                lines.append("| {} | {} | {} | ".format(prov, kind, status) + " | ".join(
                    str(sum(1 for r in rows if r.get("engine_mechanism") == m))
                    for m in ("out", "pursuit_stop", "evidence", "other", "decoder")) + " |")
    lines.append("")
    lines.append("### 1e. Primary cause of the missed onsets")
    lines.append("")
    lines.append("| provider | kind | missed | " + " | ".join(CAUSES) + " |")
    lines.append("|---|---|---|" + "---|" * len(CAUSES))
    for prov, test in (("Metrica", lambda g: g.startswith("METRICA")),
                       ("DFL", lambda g: g.startswith("DFL"))):
        for kind in kinds:
            rows = [r for p in refs if test(p["game"]) for r in p["onsets"]
                    if r["kind"] == kind and r["status"] == "missed"]
            if not rows:
                continue
            lines.append("| {} | {} | {} | ".format(prov, kind, len(rows)) + " | ".join(
                str(sum(1 for r in rows if r["flags"]["cause"] == c)) for c in CAUSES) + " |")
    lines.append("")
    lines.append("### 1f. Flags of the missed onsets (counts; a missed onset can carry several)")
    lines.append("")
    lines.append("| provider | kind | missed | out seen | out unconfirmed | out other stage | "
                 "fast run | pursuit unconfirmed | collective hint | stop within 4 s | "
                 "slow within 4 s | ball vanished |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for prov, test in (("Metrica", lambda g: g.startswith("METRICA")),
                       ("DFL", lambda g: g.startswith("DFL"))):
        for kind in kinds:
            rows = [r["flags"] for p in refs if test(p["game"]) for r in p["onsets"]
                    if r["kind"] == kind and r["status"] == "missed"]
            if not rows:
                continue
            cells = [sum(1 for f in rows if f["out"] == "seen"),
                     sum(1 for f in rows if f["out"] == "unconfirmed"),
                     sum(1 for f in rows if f["out"] not in ("none", "seen", "unconfirmed")),
                     sum(1 for f in rows if f["pursuit"] != "none"),
                     sum(1 for f in rows if f["pursuit"] == "unconfirmed"),
                     sum(1 for f in rows if f["hint"]),
                     sum(1 for f in rows if f["stop_4s"]),
                     sum(1 for f in rows if f["slow_4s"]),
                     sum(1 for f in rows if f["vanished"])]
            lines.append("| {} | {} | {} | ".format(prov, kind, len(rows))
                         + " | ".join(str(c) for c in cells) + " |")
    lines.append("")
    lines.append("### 1g. Pursuit stage of the missed onsets with a fast run")
    lines.append("")
    lines.append("| provider | kind | " + " | ".join(PURSUIT_STAGES) + " |")
    lines.append("|---|---|" + "---|" * len(PURSUIT_STAGES))
    for prov, test in (("Metrica", lambda g: g.startswith("METRICA")),
                       ("DFL", lambda g: g.startswith("DFL"))):
        for kind in kinds:
            rows = [r["flags"] for p in refs if test(p["game"]) for r in p["onsets"]
                    if r["kind"] == kind and r["status"] == "missed"
                    and r["flags"]["pursuit"] != "none"]
            if not rows:
                continue
            lines.append("| {} | {} | ".format(prov, kind) + " | ".join(
                str(sum(1 for f in rows if f["pursuit"] == s)) for s in PURSUIT_STAGES) + " |")
    lines.append("")


def report_foul_signature(refs, lines):
    """Tables 2a to 2d: the foul signature."""
    lines.append("### 2a. Foul onsets: the nearest contested contact")
    lines.append("")
    lines.append("| match | foul onsets | contested before within 10 s | before within 2 s | "
                 "c - o p50 [p10, p90] | contested after within 5 s | none either side | "
                 "both teams within 1.5 m in the 2 s before | both_near runs per alive minute "
                 "(all frames) | delay to meanv < V_STOP | delay to half the level before |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    pool, pool_runs, pool_min = [], 0, 0.0
    for p in refs + [None]:
        if p is None:
            rows, runs, minutes = pool, pool_runs, pool_min
        else:
            rows = [r for r in p["fouls"] if r["kind"] == "foul"]
            runs = len(p["anchors"]["both_near"])
            minutes = p["alive_ref_frames"] / (p["fps"] * 60.0)
            pool.extend(rows)
            pool_runs += runs
            pool_min += minutes
        before = [r["contested_before"] for r in rows if r["contested_before"] is not None]
        within2 = sum(1 for b in before if b >= -50)
        after = sum(1 for r in rows if r["contested_after"] is not None)
        none = sum(1 for r in rows if r["contested_before"] is None
                   and r["contested_after"] is None)
        both2 = sum(1 for r in rows if r["both_near_before"] is not None
                    and r["both_near_before"] >= -50)
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {:.2f} | {} | {} |".format(
            "pooled" if p is None else short(p["game"]), len(rows), len(before), within2,
            _q(before, 0), after, none, both2, runs / minutes if minutes else 0.0,
            _q([r["delay_vstop"] for r in rows], 0), _q([r["delay_half"] for r in rows], 0)))
    lines.append("")
    lines.append("Delays in frames after the onset, up to 250 (none within 10 s: not counted). "
                 "The same delays at the other onsets, pooled: to V_STOP {}, to half {}.".format(
                     _q([r["delay_vstop"] for p in refs for r in p["fouls"]
                         if r["kind"] != "foul"], 0),
                     _q([r["delay_half"] for p in refs for r in p["fouls"]
                         if r["kind"] != "foul"], 0)))
    lines.append("")
    lines.append("### 2b. Physics at foul onsets, at other onsets and at negative contested "
                 "contacts")
    lines.append("")
    lines.append("p50 [p10, p90] (n). For onsets the anchor is the onset frame o; for the "
                 "contested contacts it is c. meanv_all in m/s, ball speed in m/s, sep in m.")
    lines.append("")
    lines.append("| provider | population | meanv 4 s before | meanv next 2 s | meanv next 4 s | "
                 "ball at anchor | ball 1 s later | sep at anchor |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for prov, test in (("Metrica", lambda g: g.startswith("METRICA")),
                       ("DFL", lambda g: g.startswith("DFL"))):
        group = [p for p in refs if test(p["game"])]
        fo = [r for p in group for r in p["fouls"] if r["kind"] == "foul"]
        ot = [r for p in group for r in p["fouls"] if r["kind"] != "foul"]
        for name, rows in (("foul onsets", fo), ("other onsets", ot)):
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                prov, name, _q([r["pre_o"] for r in rows], 2),
                _q([r["post2_o"] for r in rows], 2), _q([r["post4_o"] for r in rows], 2),
                _q([r["ball_o"] for r in rows], 1), _q([r["ball_1s"] for r in rows], 1),
                _q([r["sep_o"] for r in rows], 1)))
        sigs = [r for r in fo if r.get("sig_before") and r["contested_before"] >= -LATE_TOL]
        lines.append("| {} | contested before a foul (within 5 s), at c | {} | {} | {} | - | - "
                     "| - |".format(prov, _q([r["sig_before"]["pre"] for r in sigs], 2),
                                    _q([r["sig_before"]["post2"] for r in sigs], 2),
                                    _q([r["sig_before"]["post4"] for r in sigs], 2)))
        neg = [a for p in group for a in p["anchors"]["contested"] if a["negative"]]
        lines.append("| {} | negative contested contacts, at c | {} | {} | {} | {} | {} | {} |"
                     .format(prov, _q([a["sig"]["pre"] for a in neg], 2),
                             _q([a["sig"]["post2"] for a in neg], 2),
                             _q([a["sig"]["post4"] for a in neg], 2),
                             _q([a["ball_c"] for a in neg], 1),
                             _q([a["ball_1s"] for a in neg], 1),
                             _q([a["sep_c"] for a in neg], 1)))
        pos = [a for p in group for a in p["anchors"]["contested"] if a["truth"] == "true"]
        lines.append("| {} | true contested contacts (onset in [c - 25, c + 125]), at c | {} | "
                     "{} | {} | {} | {} | {} |".format(
                         prov, _q([a["sig"]["pre"] for a in pos], 2),
                         _q([a["sig"]["post2"] for a in pos], 2),
                         _q([a["sig"]["post4"] for a in pos], 2),
                         _q([a["ball_c"] for a in pos], 1), _q([a["ball_1s"] for a in pos], 1),
                         _q([a["sep_c"] for a in pos], 1)))
        lines.append("| {} | fall fraction, true contested / negative | {} | {} | - | - | - | - |"
                     .format(prov, _q([a["sig"]["fall"] for a in pos], 2),
                             _q([a["sig"]["fall"] for a in neg], 2)))
    lines.append("")
    for anchor_set in ANCHOR_SETS:
        lines.append("### 2c. Signature grid, anchors `{}`".format(anchor_set))
        lines.append("")
        lines.append("Pooled over the eight references. P = precision (true firings over "
                     "firings), R foul / R all = recall of foul onsets / of all onsets, "
                     "\"conf\" = with the collective-stop confirmation, \"new\" = firings where "
                     "the engine is alive and enters no D within 5 s. FP/min = false firings per "
                     "reference alive minute. Offset = o - c of the true firings.")
        lines.append("")
        lines.append("| signature | firings | P | P new | R foul | R all | FP/min | P conf | "
                     "R foul conf | FP/min conf | P conf new | offset p50 [p10, p90] | "
                     "offset, foul onsets |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        alive_min = sum(p["alive_ref_frames"] for p in refs) / (refs[0]["fps"] * 60.0)
        for x in (None,) + FALL_GRID:
            res = {}
            for conf in (False, True):
                for new in (False, True):
                    for kinds in (("foul",), None):
                        agg = None
                        for p in refs:
                            r = score_signature(p["anchors"][anchor_set], p["onsets"], x,
                                                confirmed_only=conf, kinds=kinds, new_only=new)
                            if agg is None:
                                agg = r
                            else:
                                agg = {k: (agg[k] + r[k]) if k != "precision" else None
                                       for k in r}
                        agg["precision"] = (agg["true"] / agg["fire"]) if agg["fire"] else None
                        res[(conf, new, kinds is None)] = agg
            base = res[(False, False, False)]
            name = "min2 < V_STOP" if x is None else "V_STOP or fall > {:.1f}".format(x)
            lines.append("| {} | {} | {} | {} | {}/{} | {}/{} | {:.2f} | {} | {}/{} | {:.2f} | {} "
                         "| {} | {} |".format(
                             name, base["fire"], _pc(base["true"], base["fire"]),
                             _pc(res[(False, True, False)]["true"],
                                 res[(False, True, False)]["fire"]),
                             base["recall_n"], base["recall_d"],
                             res[(False, False, True)]["recall_n"],
                             res[(False, False, True)]["recall_d"],
                             base["false"] / alive_min,
                             _pc(res[(True, False, False)]["true"],
                                 res[(True, False, False)]["fire"]),
                             res[(True, False, False)]["recall_n"],
                             res[(True, False, False)]["recall_d"],
                             res[(True, False, False)]["false"] / alive_min,
                             _pc(res[(True, True, False)]["true"],
                                 res[(True, True, False)]["fire"]),
                             _q(base["offsets"], 0), _q(base["offsets_foul"], 0)))
        lines.append("")
    lines.append("### 2d. Signature grid per match, anchors `contested`")
    lines.append("")
    lines.append("Cells: P / R foul / FP per alive minute, without confirmation; then the same "
                 "with confirmation.")
    lines.append("")
    lines.append("| signature | " + " | ".join(short(p["game"]) for p in refs) + " |")
    lines.append("|---|" + "---|" * len(refs))
    for x in (None,) + FALL_GRID:
        name = "min2 < V_STOP" if x is None else "V_STOP or fall > {:.1f}".format(x)
        for conf in (False, True):
            cells = []
            for p in refs:
                r = score_signature(p["anchors"]["contested"], p["onsets"], x,
                                    confirmed_only=conf, kinds=("foul",))
                alive = p["alive_ref_frames"] / (p["fps"] * 60.0)
                cells.append("{} / {}/{} / {:.2f}".format(_pct(r["true"], r["fire"]),
                                                          r["recall_n"], r["recall_d"],
                                                          r["false"] / alive))
            lines.append("| {}{} | ".format(name, " conf" if conf else "") + " | ".join(cells)
                         + " |")
    lines.append("")
    lines.append("### 2e. Contested contacts by truth class and engine state (anchors "
                 "`contested`, all firing or not)")
    lines.append("")
    lines.append("| match | contested | true | of which foul onset | dead | false | engine "
                 "dead at c | engine late entry | engine alive |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for p in refs:
        a = p["anchors"]["contested"]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            short(p["game"]), len(a), sum(1 for r in a if r["truth"] == "true"),
            sum(1 for r in a if r["truth"] == "true" and r["ref_kind"] == "foul"),
            sum(1 for r in a if r["truth"] == "dead"), sum(1 for r in a if r["truth"] == "false"),
            sum(1 for r in a if r["engine"] == "dead"),
            sum(1 for r in a if r["engine"] == "late_entry"),
            sum(1 for r in a if r["engine"] == "alive")))
    lines.append("")


def report_cost(refs, bar, lines):
    """Tables 3a and 3b: the cost side."""
    lines.append("### 3a. False firings and the existing confirmation (anchors `contested`, "
                 "pooled)")
    lines.append("")
    lines.append("| signature | false firings | rejected by the confirmation | kept | false "
                 "with the engine alive | of those kept |")
    lines.append("|---|---|---|---|---|---|")
    for x in (None,) + FALL_GRID:
        name = "min2 < V_STOP" if x is None else "V_STOP or fall > {:.1f}".format(x)
        false = [a for p in refs for a in p["anchors"]["contested"]
                 if a["truth"] == "false" and fires(a["sig"], a["v_stop"], x)]
        kept = [a for a in false if a["confirmed"]]
        alive = [a for a in false if a["engine"] == "alive"]
        alive_kept = [a for a in alive if a["confirmed"]]
        lines.append("| {} | {} | {} ({}) | {} | {} | {} |".format(
            name, len(false), len(false) - len(kept), _pct(len(false) - len(kept), len(false)),
            len(kept), len(alive), len(alive_kept)))
    lines.append("")
    if bar is None:
        return
    lines.append("### 3b. BAR-ATH public window: firing contested contacts")
    lines.append("")
    pub = [a for a in bar["anchors"]["contested"] if "public_label" in a]
    lines.append("Contested contacts in {}..{}: {}. Public onsets in the window: {}.".format(
        PUBLIC_WINDOW[0], PUBLIC_WINDOW[1], len(pub), bar.get("public_onsets")))
    lines.append("")
    lines.append("| frame | min2 | fall | fires at | confirmed | engine | public label at c | "
                 "public onset o - c in [-25, +125] | near 11253/13112/15290 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for a in pub:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            a["frame"], _fmt(a["sig"]["min2"]), _fmt(a["sig"]["fall"]), fire_label(a),
            "yes" if a["confirmed"] else "no", a["engine"], a["public_label"],
            ",".join(str(v) for v in a["public_onsets"]) or "-",
            ",".join(str(k) for k in PUBLIC_CONTESTED if abs(k - a["frame"]) <= LATE_TOL) or "-"))
    lines.append("")


def fire_label(a):
    """The loosest signature row that fires at this anchor: 'vstop', 'x>0.6' ... or 'no'."""
    if fires(a["sig"], a["v_stop"], None):
        return "vstop"
    for x in sorted(FALL_GRID, reverse=True):
        if fires(a["sig"], a["v_stop"], x):
            return "fall>{:.1f}".format(x)
    return "no"


def report_barath(bar, lines):
    """Tables 4a and 4b: BAR-ATH label-free."""
    lines.append("### 4a. BAR-ATH contested contacts per half (label-free)")
    lines.append("")
    alive_min = {h: v / (bar["fps"] * 60.0) for h, v in bar["alive_engine_halves"].items()}
    lines.append("| signature | half | contested | firing | confirmed | inside engine D at c | "
                 "engine late entry (onset in (c, c + 5 s]) | engine alive | alive firings per "
                 "engine-alive minute | alive and confirmed |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    anchors = bar["anchors"]["contested"]
    for x in (None,) + FALL_GRID:
        name = "min2 < V_STOP" if x is None else "V_STOP or fall > {:.1f}".format(x)
        for half in sorted(alive_min):
            a = [r for r in anchors if str(r["half"]) == half]
            f = [r for r in a if fires(r["sig"], r["v_stop"], x)]
            alive = [r for r in f if r["engine"] == "alive"]
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {:.2f} | {} |".format(
                name, half, len(a), len(f), sum(1 for r in f if r["confirmed"]),
                sum(1 for r in f if r["engine"] == "dead"),
                sum(1 for r in f if r["engine"] == "late_entry"), len(alive),
                len(alive) / alive_min[half] if alive_min[half] else 0.0,
                sum(1 for r in alive if r["confirmed"])))
    lines.append("")
    lines.append("### 4b. BAR-ATH firing contested contacts with the engine alive (no labels)")
    lines.append("")
    lines.append("Every contested contact where the engine is alive at c and enters no D within "
                 "5 s, and the signature fires at the loosest row (fall > 0.2 or min2 < V_STOP). "
                 "\"fires at\" is the strictest row that still fires. F1 segment: the start of "
                 "the c8_bump D segment of DEADBALL-FORENSICS that holds c (or starts within 5 s "
                 "after it); F1 case: a named F1 frame within 5 s.")
    lines.append("")
    lines.append("| frame | half | pre | min2 | fall | fires at | confirmed | F1 segment | "
                 "F1 case |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in anchors:
        if r["engine"] != "alive" or not fires(r["sig"], r["v_stop"], min(FALL_GRID)):
            continue
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["frame"], r["half"], _fmt(r["sig"]["pre"]), _fmt(r["sig"]["min2"]),
            _fmt(r["sig"]["fall"]), fire_label(r), "yes" if r["confirmed"] else "no",
            _fmt(r.get("f1_segment")), _fmt(r.get("f1_case"))))
    lines.append("")


def report(args):
    """The `report` command: the tables of calibration/R3C_foul_onset.md."""
    payloads = []
    for path in args.files:
        with open(path) as handle:
            payloads.append(json.load(handle))
    refs = [p for p in payloads if p["game"] != "BAR-ATH"]
    refs.sort(key=lambda p: (not p["game"].startswith("METRICA"), p["game"]))
    bar = next((p for p in payloads if p["game"] == "BAR-ATH"), None)
    lines = ["## Runs", "", "| match | engine md5 | reference onsets | engine onsets | contested "
             "| pair | out ok / engine out | pursuit ok / engine pursuit |",
             "|---|---|---|---|---|---|---|---|"]
    for p in refs + ([bar] if bar else []):
        d = p["diagnose_check"]
        lines.append("| {} | `{}` | {} | {} | {} | {} | {}/{} | {}/{} |".format(
            p["game"], p["engine_md5"][:8], p["n_ref_onsets"], p["n_engine_onsets"],
            len(p["anchors"]["contested"]), len(p["anchors"]["pair"]), d["out_ok"],
            d["engine_out"], d["pursuit_ok"], d["engine_pursuit"]))
    lines.append("")
    if refs:
        lines.append("## 1. The onset deficit by kind and by engine cause")
        lines.append("")
        report_deficit(refs, lines)
        lines.append("## 2. The foul signature")
        lines.append("")
        report_foul_signature(refs, lines)
        lines.append("## 3. The cost side")
        lines.append("")
        report_cost(refs, bar, lines)
    if bar is not None:
        lines.append("## 4. BAR-ATH, label-free")
        lines.append("")
        report_barath(bar, lines)
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--game", required=True)
    c.add_argument("--ref")
    c.add_argument("--metrica-events")
    c.add_argument("--dfl-events")
    c.add_argument("--dfl-positions")
    c.add_argument("--public-gt")
    c.add_argument("--f1-doc")
    c.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("files", nargs="+")
    r.add_argument("--out")
    args = parser.parse_args(argv)
    if args.command == "collect":
        if args.dfl_events and not args.dfl_positions:
            parser.error("--dfl-events needs --dfl-positions")
        collect(args)
    else:
        report(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
