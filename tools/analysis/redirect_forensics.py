"""Forensics of the C8 redirect rule (F2). Not part of the submission.

    python tools/analysis/redirect_forensics.py s3 --events EVENTS.json --ref REFERENCE.csv \
        --out DIR [--game METRICA-S3]
    python tools/analysis/redirect_forensics.py drops --game BAR-ATH --out DIR
    python tools/analysis/redirect_forensics.py variant --events EVENTS.json --out REFERENCE.csv

The C8 contact detector (engine/events.py, _contact_rows) gives a contact the kind `redirect`
when its signed speed change is small against its impulse. A redirect by the owning team is a
`touch`. A redirect by the other team becomes a `control` only if the next item of the stream
(uncontested contacts and slow-ball control runs, in frame order) is by the same team with no
dead frame between. Otherwise the redirect has no row. The `redirect_off` ablation of
tools/ablate.py sets REDIRECT_FRACTION to 0, so every contact is a kick or a receive.

Mode `s3` (another match with hand-made events, public data). The tool runs the shipped engine
and `redirect_off`, scores both against the event-derived reference of tools/events_gt.py with
the official evaluate(), and finds the H<->A transitions that `redirect_off` gains and the
spurious transitions it adds. For each case it writes a record: the reference events in +-4 s,
how long the new team keeps the ball in the reference, the ball trace in +-2 s, the nearest H
and A distances, the engine contact rows and, for each redirect, why the confirmation failed,
and the labels of both engine runs. The cases are grouped by stretch (the frames where the two
outputs differ). The first transition of `redirect_off` in a stretch is the flip, the later
ones are returns. A gained flip is classified with the CHALLENGE.md proxies for controlled
possession (classify()), and a return takes the class of its flip.

Mode `drops` (any match, no labels). The tool lists every dropped redirect on graded frames
with the same engine features and the Smart Tagging team phase there. It never reads a
ground truth. Use it on BAR-ATH.

Mode `variant` writes a second reference CSV for the same match. It drops the RECOVERY events
that the same team's BALL LOST or BALL OUT follows at once, with their one-frame CARRY
(loose_touch_events). It is a
diagnostic for the question "does the reference over-segment", not a replacement reference.

Both modes flag the redirects that four candidate confirmation rules would confirm (C1 to C4,
candidate_flags). The tool does not change the engine and does not score a candidate. Mode s3
gives a label-splice estimate only (splice_estimate).

The tool calls the engine's own functions (possession.prepare, possession.run,
events._bump_contacts, events._control_rows, events._dribble_rows, events._redirect_kinds).
explain_redirects() repeats the stream walk of _redirect_kinds() to give a reason for each
redirect, and the tool checks it against _redirect_kinds() on every run.

Output: DIR/cases.json (records with traces) and DIR/cases.md (the tables). Keep DIR outside
the repository. Deterministic. numpy and pandas only. Python 3.9.
"""

import argparse
import dataclasses
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from engine import data, events as ev, possession, tracks  # noqa: E402
from engine.config import Config  # noqa: E402

TOLERANCE = 25          # frames, the loosest tolerance of possession_metrics.TAUS
EVENT_WINDOW_S = 4.0    # reference events listed around a case (the brief)
TRACE_WINDOW_S = 2.0    # ball trace around a case (the brief, PLAN 5.2 forensics window)
TRACE_STEP = 4          # frames between trace rows
CASE_WINDOW_S = 2.0     # engine rows and labels listed around a case
FOLLOW_S = 2.0          # a same-team control run or contact "follows" within this (the brief)
# CHALLENGE.md proxies for controlled possession: "a player keeps the ball for a few seconds"
# or "a team completes more than two consecutive passes". The brief: class (b) is a touch
# "lost again within about 2 s". Control ends at the new team's own BALL LOST or BALL OUT, at
# the other team's next owner event, or at the reference D (classify()).
HELD_S = 3.0
BRIEF_S = 2.0
MIN_PASSES = 3
LOSS_TYPES = ("BALL LOST", "BALL OUT")
CANDIDATES = ("C1_control_run", "C2_low_conf", "C3_bump_merge", "C4_dribble_stream")
# Reference variant (mode `variant`). A Metrica BALL LOST or BALL OUT event of a team that starts
# at the frame of its own RECOVERY, or up to this many frames later, says the team touched the
# ball and lost it at once (the BALL LOST event lasts until the next owner event). CHALLENGE.md:
# a loose ball stays with the last team that had it. Source: F2 cases 51908, 113376, 45461.
LOOSE_LAG = 2


# -- engine scene ---------------------------------------------------------------------------------

def scene(game, cfg):
    """Engine inputs, labels and the redirect internals of one match under `cfg`."""
    inputs = possession.prepare(cfg, game)
    tactical = possession.engine_tactical(cfg, game)
    slot_conf = tracks.slot_confidence(tactical, cfg)
    ball, feats, e_dead = inputs["ball"], inputs["feats"], inputs["e_dead"]
    alive = ~(np.isfinite(e_dead) & (e_dead > cfg.EVENT_SUPPRESS_DEAD))
    contacts = ev._bump_contacts(ball, feats, slot_conf, alive, cfg)
    controls = ev._control_rows(ball, feats, slot_conf, alive, cfg)
    kinds = ev._redirect_kinds(contacts, controls, alive)
    # C4: the uncontested dribble rows join the stream as owner items (the engine's own walk).
    dribbles = [r for r in ev._dribble_rows(ball, feats, slot_conf, alive, cfg)
                if r[2] in ("H", "A")]
    kinds_c4 = ev._redirect_kinds(contacts, sorted(controls + dribbles), alive)
    labels = np.asarray(possession.run(cfg, inputs=inputs)).astype("<U1")
    return {"cfg": cfg, "inputs": inputs, "alive": alive, "contacts": contacts,
            "controls": controls, "dribbles": dribbles, "redirect_kinds": kinds,
            "redirect_kinds_c4": kinds_c4, "labels": labels,
            "impulse": ev.impulse(ball, int(cfg.CONTACT_HALF_WINDOW))}


def _clean_fn(alive):
    dead = np.concatenate(([0], np.cumsum(~np.asarray(alive, dtype=bool))))
    return lambda a, b: dead[b + 1] - dead[a] == 0


def explain_redirects(contacts, controls, alive):
    """([dict] for each redirect contact, the stream): status, owner before, reason of a drop.

    The walk is the one of events._redirect_kinds(). status is `touch` (the owner's redirect),
    `confirmed` (a control) or `dropped`. reason of a drop: `next_other_team`, `dead_between`
    or `no_next`. The record also holds the next stream item and the contested contacts that
    the stream skips before it.
    """
    clean = _clean_fn(alive)
    stream = [(c[0], 0, c[2], c[1], c[3]) for c in contacts if c[1] != "contested"]
    stream += [(r[0], 1, r[2], "control_run", r[3]) for r in controls]
    stream.sort(key=lambda item: (item[0], item[1]))
    contested = sorted(c[0] for c in contacts if c[1] == "contested")
    out = []
    owner, last = None, None
    for i, (frame, _, team, kind, conf) in enumerate(stream):
        if last is not None and not clean(last, frame):
            owner = None
        last = frame
        if kind != "redirect":
            owner = team
            continue
        record = {"frame": int(frame), "team": team, "confidence": float(conf),
                  "owner_before": owner, "stream_index": i}
        nxt = stream[i + 1] if i + 1 < len(stream) else None
        if nxt is not None:
            record["next"] = {"frame": int(nxt[0]), "team": nxt[2], "kind": nxt[3],
                              "confidence": float(nxt[4]), "gap": int(nxt[0] - frame),
                              "dead_between": bool(not clean(frame, nxt[0]))}
            record["contested_between"] = [int(f) for f in contested if frame < f < nxt[0]]
        else:
            record["next"] = None
            record["contested_between"] = []
        if owner == team:
            record["status"], record["reason"] = "touch", ""
        elif nxt is not None and nxt[2] == team and clean(frame, nxt[0]):
            record["status"], record["reason"] = "confirmed", ""
            owner = team
        else:
            record["status"] = "dropped"
            if nxt is None:
                record["reason"] = "no_next"
            elif not clean(frame, nxt[0]):
                record["reason"] = "dead_between"
            else:
                record["reason"] = "next_other_team"
        out.append(record)
    return out, stream


def check_explanation(records, kinds):
    """Raise if the walk disagrees with events._redirect_kinds()."""
    mine = {r["frame"]: ("touch" if r["status"] == "touch" else "control")
            for r in records if r["status"] != "dropped"}
    if mine != {int(k): v for k, v in kinds.items()}:
        raise AssertionError("explain_redirects() disagrees with events._redirect_kinds()")


def candidate_flags(record, stream, alive, fps, cfg):
    """Flags of candidate confirmation rules for one dropped redirect (no engine change).

    C1 control run: a slow-ball control run of the same team starts within FOLLOW_S after the
       redirect, with no dead frame between, whatever stream items lie between (contested
       contacts are never in the stream, so C1 widens the rule to other-team items).
    C2 low-confidence intervener: the next item is the other team's with a confidence below
       OWNER_MIN_CONFIDENCE, and the item after it is the same team's, all alive.
    C3 bump merge (a window in ball-travel terms): the same team's next stream item comes
       first after skipping other-team items that lie within CONTACT_HALF_WINDOW of the
       redirect, with no dead frame between. One smoothed bump cannot separate two contacts
       closer than the kernel half width (M13), so such an item may be the same contact.
    C4 (set by redirect_index, not here): the uncontested dribble rows join the stream of
       events._redirect_kinds() as owner items. The flag is True when the redirect is a
       `control` under that stream.
    """
    clean = _clean_fn(alive)
    frame, team, i = record["frame"], record["team"], record["stream_index"]
    horizon = frame + int(round(FOLLOW_S * fps))
    later = [s for s in stream[i + 1:] if s[0] <= horizon]
    c1 = any(s[3] == "control_run" and s[2] == team and clean(frame, s[0]) for s in later)
    c2 = False
    if (record["reason"] == "next_other_team" and i + 2 < len(stream)
            and record["next"]["confidence"] < cfg.OWNER_MIN_CONFIDENCE):
        after = stream[i + 2]
        c2 = after[2] == team and clean(frame, after[0])
    c3 = False
    h = int(cfg.CONTACT_HALF_WINDOW)
    for s in stream[i + 1:]:
        if not clean(frame, s[0]):
            break
        if s[2] == team:
            c3 = True
            break
        if s[0] - frame > h:
            break
    same_next = next((s for s in stream[i + 1:] if s[2] == team), None)
    run_next = next((s for s in later if s[2] == team and s[3] == "control_run"), None)
    return {"C1_control_run": bool(c1), "C2_low_conf": bool(c2), "C3_bump_merge": bool(c3),
            "same_team_next_gap": int(same_next[0] - frame) if same_next else None,
            "same_team_next_kind": same_next[3] if same_next else None,
            "control_run_gap": int(run_next[0] - frame) if run_next else None}


def redirect_index(sc):
    """{frame: record} of the redirect explanations of a scene, with features and flags."""
    records, stream = explain_redirects(sc["contacts"], sc["controls"], sc["alive"])
    check_explanation(records, sc["redirect_kinds"])
    fps = float(sc["inputs"]["fps"])
    for r in records:
        r.update(contact_features(r["frame"], r["team"], sc))
        r["status_c4"] = sc["redirect_kinds_c4"].get(r["frame"], "dropped")
        dribble = next((d for d in sc["dribbles"] if d[0] > r["frame"] and d[2] == r["team"]),
                       None)
        r["same_team_dribble_gap"] = int(dribble[0] - r["frame"]) if dribble else None
        if r["status"] == "dropped":
            flags = candidate_flags(r, stream, sc["alive"], fps, sc["cfg"])
            flags["C4_dribble_stream"] = r["status_c4"] == "control"
            r["candidates"] = flags
    return {r["frame"]: r for r in records}


# -- features ---------------------------------------------------------------------------------------

def _f(value, digits=2):
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


def contact_features(frame, team, sc):
    """Ball and player features of one contact.

    speed at c - h and c + h, impulse |J| at c, turn angle between v(c - h) and v(c + h), the
    smallest H and A distances over [c - h, c + h], and the share of the next FOLLOW_S seconds
    (frames with a nearest player) on which `team` has the nearest player.
    """
    ball, feats = sc["inputs"]["ball"], sc["inputs"]["feats"]
    h = int(sc["cfg"].CONTACT_HALF_WINDOW)
    n = ball["speed"].shape[0]
    lo, hi = max(0, frame - h), min(n - 1, frame + h)
    v0 = np.array([ball["vx"][lo], ball["vy"][lo]])
    v1 = np.array([ball["vx"][hi], ball["vy"][hi]])
    norm = np.linalg.norm(v0) * np.linalg.norm(v1)
    angle = (np.degrees(np.arccos(np.clip(np.dot(v0, v1) / norm, -1.0, 1.0)))
             if np.isfinite(norm) and norm > 0 else np.nan)
    d_h = ev._finite_or_inf(feats["dH"])[lo:hi + 1]
    d_a = ev._finite_or_inf(feats["dA"])[lo:hi + 1]
    fps = float(sc["inputs"]["fps"])
    near = np.asarray(feats["nearest_team"])[frame + 1:min(n, frame + 1 + int(round(FOLLOW_S
                                                                                  * fps)))]
    code = data.HOME if team == "H" else data.AWAY
    seen = near >= 0
    share = float(np.mean(near[seen] == code)) if seen.any() else np.nan
    return {"speed_before": _f(ball["speed"][lo]), "speed_after": _f(ball["speed"][hi]),
            "impulse": _f(sc["impulse"][frame]), "turn_deg": _f(angle, 0),
            "dH_min": _f(d_h.min()), "dA_min": _f(d_a.min()), "nearest_share_after": _f(share)}


def ball_trace(center, sc, fps):
    """Rows (frame, visible, speed, impulse, dH, dA) every TRACE_STEP frames in +-TRACE_WINDOW_S."""
    ball, feats = sc["inputs"]["ball"], sc["inputs"]["feats"]
    n = ball["speed"].shape[0]
    w = int(round(TRACE_WINDOW_S * fps))
    rows = []
    for f in range(max(0, center - w), min(n, center + w + 1), TRACE_STEP):
        rows.append({"frame": f, "visible": bool(ball["visible"][f]),
                     "speed": _f(ball["speed"][f]), "impulse": _f(sc["impulse"][f]),
                     "dH": _f(feats["dH"][f]), "dA": _f(feats["dA"][f])})
    return rows


def visible_share(center, sc, fps):
    """Share of frames with a visible ball in +-TRACE_WINDOW_S."""
    vis = sc["inputs"]["ball"]["visible"]
    w = int(round(TRACE_WINDOW_S * fps))
    lo, hi = max(0, center - w), min(vis.shape[0], center + w + 1)
    return _f(np.mean(vis[lo:hi]))


def runs_text(labels, lo, hi):
    """Run-length text of labels[lo:hi], for example `H18 A33`."""
    seg = np.asarray(labels)[max(0, lo):hi]
    if seg.size == 0:
        return ""
    edges = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [seg.size]))
    return " ".join("{}{}".format(seg[a], b - a) for a, b in zip(starts, stops))


def engine_rows(table, lo, hi):
    """Event rows of every kind in [lo, hi]."""
    part = table[(table["frame"] >= lo) & (table["frame"] <= hi)]
    return [{"frame": int(r.frame), "kind": r.kind, "team": r.team,
             "confidence": _f(r.confidence), "notes": r.notes}
            for r in part.itertuples(index=False)]


def tag_phase(sc, frame):
    """Smart Tagging team phase at `frame`: H, A, or `-` (none or both), with the codes there."""
    tagging = sc["inputs"]["tagging"]
    if tagging is None or len(tagging) == 0:
        return "-", ""
    lo = possession.tag_log_odds(tagging, sc["inputs"]["n_frames"], sc["cfg"])[frame]
    side = "H" if lo > 0 else ("A" if lo < 0 else "-")
    hit = tagging[(tagging["start_frame"] <= frame) & (tagging["end_frame"] > frame)]
    codes = sorted(set("{}:{}".format(c, t or "-") for c, t in zip(hit["code"], hit["team"])))
    return side, " ".join(codes)


def graded_mask(sc):
    """bool (N,): graded frames of the match (games.py blocks)."""
    mask = np.zeros(sc["inputs"]["n_frames"], dtype=bool)
    for _, first, last in sc["inputs"]["blocks"]:
        mask[first:last + 1] = True
    return mask


# -- reference (mode s3) ---------------------------------------------------------------------------

def reference_events(path, game):
    """Video-frame events of a Metrica events JSON, with team letters H and A."""
    import events_gt

    entry = data.games.GAMES[game]
    events = events_gt.to_video(events_gt.load_events(path), entry["offsets"])
    home = events_gt.home_team_name(game)
    for e in events:
        e["side"] = "H" if e["team"] == home else "A"
        e["owner"] = events_gt.is_owner(e)
    return events


def hold_after(frame, team, ref_codes, events, fps):
    """How `team` keeps the ball in the reference from `frame`, a dict.

    run_s: the reference run of `team` from `frame` (it ends at the other team's owner event,
    at D or at the block end: run_ender). control_s: the run, cut at the start of the first
    BALL LOST or BALL OUT event of `team` at or after `frame` - 2. own_events and passes:
    owner events and PASS events of `team` from `frame` - 2 to the end of control.
    """
    import possession_metrics as pm

    code = pm.LABELS.index(team)
    n = ref_codes.shape[0]
    end = frame
    while end < n and ref_codes[end] == code:
        end += 1
    after = pm.LABELS[ref_codes[end]] if end < n else "X"
    ender = "other" if after in ("H", "A") else ("D" if after == "D" else "end")
    losses = [e["start"] for e in events
              if e["side"] == team and e["type"] in LOSS_TYPES and frame - 2 <= e["start"] < end]
    stop = min([end] + losses)
    own = [e for e in events
           if e["side"] == team and e["owner"] and frame - 2 <= e["start"] <= max(stop, frame)]
    return {"run_s": round((end - frame) / fps, 2), "run_ender": ender,
            "control_s": round(max(0, stop - frame) / fps, 2),
            "control_ender": "own loss" if stop < end else ender,
            "own_events": len(own), "passes": sum(1 for e in own if e["type"] == "PASS")}


def classify(control_s, passes):
    """Class of a gained flip from the CHALLENGE.md proxies.

    a: held (control_s >= HELD_S or passes >= MIN_PASSES). b: lost again (control_s < BRIEF_S
    and no completed pass). c: the rest.
    """
    if control_s >= HELD_S or passes >= MIN_PASSES:
        return "a"
    if control_s < BRIEF_S and passes == 0:
        return "b"
    return "c"


def classify_run(run_s, passes):
    """The first, simpler rule: the whole reference run counts as held (kept for comparison)."""
    if run_s >= HELD_S or passes >= MIN_PASSES:
        return "a"
    if run_s < BRIEF_S:
        return "b"
    return "c"


def events_near(frame, events, fps, until=None):
    """Reference events with a start in EVENT_WINDOW_S before `frame` and up to `until`.

    `until` defaults to EVENT_WINDOW_S after `frame`. A caller that knows where the reference
    spell of the new team ends passes that end plus a margin, so the event that ends it shows.
    """
    w = int(round(EVENT_WINDOW_S * fps))
    last = frame + w if until is None else max(frame + w, int(until))
    return [{"frame": e["start"], "end": e["end"], "type": e["type"],
             "subtypes": "/".join(e["subtypes"]), "team": e["side"]}
            for e in events if frame - w <= e["start"] <= last]


def loose_touch_events(events):
    """(kept events, dropped events) for the reference variant.

    `events` are the normalised events of tools/events_gt.py (frame ids). A RECOVERY is dropped
    when a BALL LOST or BALL OUT of the same team and period starts in [start, start + LOOSE_LAG].
    The CARRY events of the same team that start in the same interval and last at most
    LOOSE_LAG frames go with it, because the taggers add a one-frame CARRY to each RECOVERY.
    The variant is a diagnostic. It removes the brief spells that the reference derives from a
    touch that the taggers themselves mark as lost at once.
    """
    losses = {}
    for e in events:
        if e["type"] in LOSS_TYPES:
            losses.setdefault((e["team"], e["period"]), []).append(e["start"])
    touches = {}
    for e in events:
        starts = losses.get((e["team"], e["period"]), [])
        if e["type"] == "RECOVERY" and any(e["start"] <= f <= e["start"] + LOOSE_LAG
                                           for f in starts):
            touches.setdefault((e["team"], e["period"]), []).append(e["start"])
    kept, dropped = [], []
    for e in events:
        firsts = touches.get((e["team"], e["period"]), [])
        gone = (e["type"] == "RECOVERY" and e["start"] in firsts) or (
            e["type"] == "CARRY" and e["end"] - e["start"] <= LOOSE_LAG
            and any(f <= e["start"] <= f + LOOSE_LAG for f in firsts))
        (dropped if gone else kept).append(e)
    return kept, dropped


def score(ref_codes, labels):
    """Official evaluate() of `labels` against the reference, and its summary."""
    import possession_metrics as pm
    import score_game

    result, gt_stats, pred_stats = score_game.score_codes(ref_codes, pm.label_codes(labels))
    return result, score_game.summary(result, gt_stats, pred_stats)


def transition_diff(res_a, res_b):
    """(gained matched pairs, added spurious, removed spurious) of b against a."""
    ta, tb = res_a["transitions"][TOLERANCE], res_b["transitions"][TOLERANCE]
    matched_a = set(tuple(g) for g, _ in ta["pairs"])
    gained = sorted((tuple(g), tuple(p)) for g, p in tb["pairs"] if tuple(g) not in matched_a)
    spur_a = set(tuple(p) for p in ta["spurious"])
    spur_b = set(tuple(p) for p in tb["spurious"])
    return gained, sorted(spur_b - spur_a), sorted(spur_a - spur_b)


def changed_rows(table_a, table_b):
    """Rows (frame, kind, team) of table_b that table_a does not have."""
    def key(t):
        return set(zip(t["frame"].astype(int), t["kind"], t["team"].astype(str)))
    return sorted(key(table_b) - key(table_a))


def diff_stretches(labels_a, labels_b):
    """[(start, stop)] half-open runs of frames where two label vectors differ."""
    starts, stops = tracks._runs(np.asarray(labels_a) != np.asarray(labels_b))
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def origin_redirect(stretch, new_rows, red):
    """The roff-only owner row at the start of a stretch and its redirect record."""
    start = stretch[0]
    near = [(f, k, t) for f, k, t in new_rows
            if k in ("release", "control") and start - 15 <= f <= start + 2]
    if not near:
        return None, None
    f, k, t = near[-1]
    return {"frame": int(f), "kind": k, "team": t}, red.get(int(f))


def ref_spell_at(frame, team, ref_codes, fps):
    """(start, stop, seconds) of the reference run of `team` nearest `frame` within 4 s."""
    import possession_metrics as pm

    code = pm.LABELS.index(team)
    w = int(round(EVENT_WINDOW_S * fps))
    lo, hi = max(0, frame - w), min(ref_codes.shape[0], frame + w + 1)
    hits = np.flatnonzero(ref_codes[lo:hi] == code)
    if hits.size == 0:
        return None
    f = lo + int(hits[np.argmin(np.abs(lo + hits - frame))])
    a = f
    while a > 0 and ref_codes[a - 1] == code:
        a -= 1
    b = f
    while b < ref_codes.shape[0] and ref_codes[b] == code:
        b += 1
    return int(a), int(b), round((b - a) / fps, 2)


def stretch_records(ship, off, red, ref_codes, events, gained, added, new_rows):
    """One record per stretch where shipped and redirect_off differ."""
    import possession_metrics as pm

    fps = float(ship["inputs"]["fps"])
    w = int(round(CASE_WINDOW_S * fps))
    out = []
    for start, stop in diff_stretches(ship["labels"], off["labels"]):
        inside = lambda f: start - 1 <= f <= stop + 1  # noqa: E731
        trans = [("gained", g, p) for g, p in gained if inside(p[0])]
        trans += [("spurious", None, p) for p in added if inside(p[0])]
        trans.sort(key=lambda item: item[2][0])
        origin, rec = origin_redirect((start, stop), new_rows, red)
        new_team = str(off["labels"][start])
        spell = (ref_spell_at(start, new_team, ref_codes, fps)
                 if new_team in ("H", "A") else None)
        reach = max(stop, spell[1] if spell else stop) + w
        record = {
            "start": start, "stop": stop, "frames": stop - start,
            "shipped": str(ship["labels"][start]), "off": new_team,
            "origin": origin, "redirect": rec,
            "transitions": [], "ref_events": events_near(start, events, fps, until=reach),
            "trace": ball_trace(start, ship, fps), "visible_share": visible_share(start, ship, fps),
            "rows_shipped": engine_rows(ship["inputs"]["events"], start - w, stop + w),
            "labels_shipped": runs_text(ship["labels"], start - w, stop + w),
            "labels_off": runs_text(off["labels"], start - w, stop + w),
            "labels_ref": runs_text(np.array([pm.LABELS[c] for c in ref_codes]), start - w,
                                    stop + w),
            "ref_spell_of_new_team": spell,
        }
        for n, (kind, g, p) in enumerate(trans):
            item = {"kind": kind, "role": "flip" if n == 0 else "return",
                    "pred": [int(p[0]), pm.LABELS[p[1]], pm.LABELS[p[2]]]}
            if g is not None:
                item["gt"] = [int(g[0]), pm.LABELS[g[1]], pm.LABELS[g[2]]]
                item.update(hold_after(int(g[0]), pm.LABELS[g[2]], ref_codes, events, fps))
                item["cls_run_rule"] = classify_run(item["run_s"], item["passes"])
                item["cls"] = classify(item["control_s"], item["passes"])
            record["transitions"].append(item)
        flip = next((t for t in record["transitions"] if t["role"] == "flip"), None)
        if flip is not None and flip["kind"] == "gained":
            record["cls"] = flip["cls"]
        elif flip is not None:
            record["cls"] = "late"   # the flip itself is spurious: see ref_spell_of_new_team
        else:
            record["cls"] = "-"
        for t in record["transitions"]:
            if t["kind"] == "gained":
                t["stretch_cls"] = record["cls"]
        out.append(record)
    return out


def splice_estimate(ship, off, stretches, confirm, ref_codes):
    """spellF1 summary of shipped labels with the redirect_off labels spliced into the
    stretches whose origin redirect `confirm` accepts. An estimate, not an engine run."""
    labels = ship["labels"].copy()
    used = []
    for s in stretches:
        rec = s["redirect"]
        if rec is not None and confirm(rec):
            labels[s["start"]:s["stop"]] = off["labels"][s["start"]:s["stop"]]
            used.append(s["start"])
    _, head = score(ref_codes, labels)
    return {"stretches": used, "spell_f1_mean": round(head["spell_f1_mean"], 4),
            "matched": head["matched"], "spurious": head["spurious"],
            "possessions": head["results"]["possessions"]}


# -- tables -------------------------------------------------------------------------------------------

def _ref_summary(events, lo, hi):
    keep = [e for e in events if lo <= e["frame"] <= hi and e["type"] != "CARRY"]
    return "; ".join("{} {} {}{}".format(e["frame"], e["team"], e["type"].lower(),
                                         " " + e["subtypes"].lower() if e["subtypes"] else "")
                     for e in keep)


def _next_text(r):
    nxt = r["next"]
    if not nxt:
        return "none"
    return "{} {} {:.2f} +{}{}".format(nxt["team"], nxt["kind"], nxt["confidence"], nxt["gap"],
                                       " (dead between)" if nxt["dead_between"] else "")


def _cand_text(r):
    cand = r.get("candidates")
    if not cand:
        return "-"
    return " ".join("x" if cand[k] else "." for k in CANDIDATES)


def markdown_stretches(stretches):
    """Markdown table of the stretches of mode s3."""
    lines = ["| stretch | ship/off | transitions (role: ref frame or roff frame, class) | "
             "ref control s / run s, own events, passes | redirect (owner before; reason) | "
             "next stream item | contested between | v before→after m/s, \\|J\\|, turn | "
             "dH/dA min m | new team nearest share 2 s | C1 C2 C3 C4 | labels ref / ship / off "
             "| ref events (no CARRY) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in stretches:
        r = s["redirect"]
        trans = []
        hold = ""
        for t in s["transitions"]:
            if t["kind"] == "gained":
                trans.append("{} gained {} {}→{} ({})".format(t["role"], t["gt"][0], t["gt"][1],
                                                            t["gt"][2], t["stretch_cls"]))
                if t["role"] == "flip":
                    hold = "{} / {} ({}), {}, {}".format(t["control_s"], t["run_s"],
                                                         t["control_ender"], t["own_events"],
                                                         t["passes"])
            else:
                trans.append("{} spurious {} {}→{}".format(t["role"], t["pred"][0], t["pred"][1],
                                                         t["pred"][2]))
        if not hold and s["ref_spell_of_new_team"]:
            a, b, sec = s["ref_spell_of_new_team"]
            hold = "ref {} spell {}–{} ({} s)".format(s["off"], a, b - 1, sec)
        lines.append("| {}–{} | {}/{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} / {} / {} "
                     "| {} |".format(
                         s["start"], s["stop"] - 1, s["shipped"], s["off"], "; ".join(trans),
                         hold or "-",
                         "{} {} {:.2f} (owner {}; {})".format(r["frame"], r["team"],
                                                              r["confidence"], r["owner_before"],
                                                              r["reason"] or r["status"])
                         if r else "-",
                         _next_text(r) if r else "-",
                         ",".join(str(f) for f in r["contested_between"]) or "-" if r else "-",
                         "{}→{}, {}, {}°".format(r["speed_before"], r["speed_after"],
                                                 r["impulse"], r["turn_deg"]) if r else "-",
                         "{}/{}".format(r["dH_min"], r["dA_min"]) if r else "-",
                         r["nearest_share_after"] if r else "-",
                         _cand_text(r) if r else "-",
                         s["labels_ref"], s["labels_shipped"], s["labels_off"],
                         _ref_summary(s["ref_events"], 0, 10 ** 9)))
    return lines


def markdown_drops(drops):
    """Markdown table of dropped redirects (mode drops)."""
    lines = ["| frame | team | conf | owner before | reason | next stream item | contested "
             "between | v before→after m/s, \\|J\\|, turn | dH/dA min m | nearest share 2 s | "
             "same-team next stream item | same-team dribble | control run ≤ 2 s | C1 C2 C3 C4 "
             "| tag phase | shipped ±2 s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in drops:
        cand = r["candidates"]
        lines.append("| {} | {} | {:.2f} | {} | {} | {} | {} | {}→{}, {}, {}° | {}/{} | {} | {} "
                     "| {} | {} | {} | {} {} | {} |".format(
                         r["frame"], r["team"], r["confidence"], r["owner_before"], r["reason"],
                         _next_text(r), ",".join(str(f) for f in r["contested_between"]) or "-",
                         r["speed_before"], r["speed_after"], r["impulse"], r["turn_deg"],
                         r["dH_min"], r["dA_min"], r["nearest_share_after"],
                         "{} +{}".format(cand["same_team_next_kind"], cand["same_team_next_gap"])
                         if cand["same_team_next_gap"] is not None else "none",
                         "+{}".format(r["same_team_dribble_gap"])
                         if r["same_team_dribble_gap"] is not None else "none",
                         "+{}".format(cand["control_run_gap"])
                         if cand["control_run_gap"] is not None else "none",
                         _cand_text(r), r["tag_side"], r["tag_codes"] or "", r["labels"]))
    return lines


# -- modes --------------------------------------------------------------------------------------------

def off_config(cfg):
    """`cfg` with the redirect rule off, as the redirect_off ablation of tools/ablate.py."""
    return dataclasses.replace(cfg, REDIRECT_FRACTION=0.0)


def status_counts(red, graded):
    """Counts of redirect status and drop reason on graded frames."""
    out = {}
    for r in red.values():
        if not graded[r["frame"]]:
            continue
        key = r["status"] + (":" + r["reason"] if r["reason"] else "")
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def c4_changes(red, graded):
    """Status changes of the graded redirects when dribble rows join the stream (C4)."""
    out = {}
    for r in red.values():
        if not graded[r["frame"]]:
            continue
        before = "control" if r["status"] == "confirmed" else r["status"]
        if before != r["status_c4"]:
            key = "{}->{}".format(before, r["status_c4"])
            out.setdefault(key, []).append(r["frame"])
    return dict(sorted(out.items()))


def candidate_counts(dropped):
    """How many dropped redirects each candidate rule would confirm."""
    return {k: [r["frame"] for r in dropped if r["candidates"][k]] for k in CANDIDATES}


def class_counts(stretches):
    """Counts, frames and reference seconds of the gained transitions per class.

    `seconds` sums the reference control time of the new team over the gained flips.
    """
    out = {}
    for cls in ("a", "b", "c", "late"):
        trans = [t for s in stretches for t in s["transitions"]
                 if t["kind"] == "gained" and t["stretch_cls"] == cls]
        flips = [t for t in trans if t["role"] == "flip"]
        out[cls] = {"n": len(trans), "frames": [t["gt"][0] for t in trans],
                    "flips": len(flips),
                    "control_s": round(sum(t["control_s"] for t in flips), 2),
                    "run_s": round(sum(t["run_s"] for t in flips), 2)}
    return out


def run_s3(args):
    import possession_metrics as pm

    cfg = Config.default()
    ship = scene(args.game, cfg)
    off = scene(args.game, off_config(cfg))
    ref_codes = pm.label_codes(pm.read_label_csv(args.ref))
    events = reference_events(args.events, args.game)
    res_ship, head_ship = score(ref_codes, ship["labels"])
    res_off, head_off = score(ref_codes, off["labels"])
    gained, added, removed = transition_diff(res_ship, res_off)
    red = redirect_index(ship)
    graded = graded_mask(ship)
    new_rows = changed_rows(ship["inputs"]["events"], off["inputs"]["events"])
    stretches = stretch_records(ship, off, red, ref_codes, events, gained, added, new_rows)
    dropped = [r for r in red.values() if r["status"] == "dropped" and graded[r["frame"]]]
    origins = set(s["redirect"]["frame"] for s in stretches if s["redirect"])
    per_flip = [t for s in stretches for t in s["transitions"]
                if t["kind"] == "gained" and t["role"] == "flip"]
    summary = {
        "shipped": head_ship, "redirect_off": head_off,
        "gained": [int(g[0]) for g, _ in gained],
        "spurious_added": [[int(p[0]), pm.LABELS[p[1]], pm.LABELS[p[2]]] for p in added],
        "spurious_removed": [[int(p[0]), pm.LABELS[p[1]], pm.LABELS[p[2]]] for p in removed],
        "stretches": len(stretches),
        "redirects": status_counts(red, graded),
        "dropped_graded": [r["frame"] for r in sorted(dropped, key=lambda r: r["frame"])],
        "dropped_without_effect": sorted(r["frame"] for r in dropped
                                         if r["frame"] not in origins),
        "candidates_on_dropped": candidate_counts(dropped),
        "c4_status_changes": c4_changes(red, graded),
        "classes": class_counts(stretches),
        "flip_classes_run_rule": {t["gt"][0]: t["cls_run_rule"] for t in per_flip},
        "estimates": {k: splice_estimate(ship, off, stretches,
                                         lambda rec, k=k: rec.get("candidates", {}).get(k, False),
                                         ref_codes) for k in CANDIDATES},
    }
    write_outputs(args.out, summary, stretches, markdown_stretches(stretches))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("shipped", "redirect_off")},
                     indent=1, sort_keys=True))
    for name, head in (("shipped", head_ship), ("redirect_off", head_off)):
        print(name, round(head["spell_f1_mean"], 4), head["matched"], head["missed"],
              head["spurious"], head["results"]["possessions"])


def run_drops(args):
    cfg = Config.default()
    ship = scene(args.game, cfg)
    red = redirect_index(ship)
    graded = graded_mask(ship)
    fps = float(ship["inputs"]["fps"])
    w = int(round(CASE_WINDOW_S * fps))
    drops = []
    for f in sorted(red):
        r = red[f]
        if r["status"] != "dropped" or not graded[f]:
            continue
        r = dict(r)
        r["tag_side"], r["tag_codes"] = tag_phase(ship, f)
        r["labels"] = runs_text(ship["labels"], f - w, f + w + 1)
        r["visible_share"] = visible_share(f, ship, fps)
        r["trace"] = ball_trace(f, ship, fps)
        drops.append(r)
    summary = {"game": data.resolve_game(args.game), "redirects": status_counts(red, graded),
               "dropped": len(drops), "candidates": candidate_counts(drops),
               "c4_status_changes": c4_changes(red, graded),
               "reasons": {k: [r["frame"] for r in drops if r["reason"] == k]
                           for k in ("next_other_team", "dead_between", "no_next")}}
    write_outputs(args.out, summary, drops, markdown_drops(drops))
    print(json.dumps(summary, indent=1, sort_keys=True))


def run_variant(args):
    import events_gt

    entry = data.games.GAMES[data.resolve_game(args.game)]
    events = events_gt.load_events(args.events)
    kept, dropped = loose_touch_events(events)
    home = events_gt.home_team_name(args.game)
    codes, info = events_gt.derive(kept, home, int(entry["n_video_frames"]), entry["offsets"],
                                   float(entry["fps"]))
    events_gt.write_csv(args.out, codes)
    stats = events_gt.reference_stats(codes, float(entry["fps"]))
    print("dropped events: {} RECOVERY of {}, {} CARRY".format(
        sum(1 for e in dropped if e["type"] == "RECOVERY"),
        sum(1 for e in events if e["type"] == "RECOVERY"),
        sum(1 for e in dropped if e["type"] == "CARRY")))
    print("possessions {possessions}  mean {mean_duration_s:.1f} s  D share {d_share:.3f}  "
          "D segments {d_segments}".format(**stats))
    print("wrote {}".format(args.out))


def write_outputs(out, summary, records, lines):
    """Write DIR/cases.json and DIR/cases.md."""
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cases.json"), "w") as handle:
        json.dump({"summary": summary, "records": records}, handle, indent=1, sort_keys=True,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
        handle.write("\n")
    with open(os.path.join(out, "cases.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("wrote {}/cases.json and cases.md".format(out))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    s3 = sub.add_parser("s3", help="cases of redirect_off against an events reference")
    s3.add_argument("--game", default="METRICA-S3")
    s3.add_argument("--events", required=True, help="Metrica events JSON of the match")
    s3.add_argument("--ref", required=True, help="reference CSV of tools/events_gt.py")
    s3.add_argument("--out", required=True, help="output directory, outside the repository")
    drops = sub.add_parser("drops", help="dropped redirects on graded frames, no labels")
    drops.add_argument("--game", default=None)
    drops.add_argument("--out", required=True, help="output directory, outside the repository")
    variant = sub.add_parser("variant", help="write the loose-touch reference variant (CSV)")
    variant.add_argument("--game", default="METRICA-S3")
    variant.add_argument("--events", required=True, help="Metrica events JSON of the match")
    variant.add_argument("--out", required=True, help="reference CSV to write, outside the repo")
    args = parser.parse_args(argv)
    if args.mode == "variant":
        run_variant(args)
        return 0
    if args.mode == "s3":
        if data.resolve_game(args.game) == data.games.GAME:
            raise SystemExit("mode s3 needs another match: never a reference of {}".format(
                data.games.GAME))
        run_s3(args)
    else:
        run_drops(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
