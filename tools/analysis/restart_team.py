"""Restart-team measurements for card P9-C. The script measures. It designs no rule.

    python tools/analysis/restart_team.py measure --game NAME --ref REFERENCE.csv
        [--metrica-events EVENTS] [--dfl-events EV.xml --dfl-positions POS.xml]
        [--engine-csv ENGINE.csv] --out OUT.json
    python tools/analysis/restart_team.py exposure --game BAR-ATH [--engine-csv ENGINE.csv]
        --out OUT.json
    python tools/analysis/restart_team.py report OUT.json [OUT.json ...]

`measure` runs the engine on an external reference match and compares its restarts with the
reference. `exposure` runs the engine on the competition match and reads no label. `report`
prints the tables of docs/plan/calibration/P9C_measurements.md from the JSON files. Set
MTAH_DATA_DIR for each match as tools/score_game.py needs it. Write the JSON files outside the
repository. Never run `measure` on a sealed match of calibration/SPLITS.md.

Definitions.

- Label runs. A run is a maximal stretch of frames with one label.
- Restart. A `D` run that the next frame ends with `H` or `A`. The onset is the first `D` frame.
  The exit is the first live frame. The restarting team is the label at the exit. A `D` run
  that ends at `X` or at the end of the file is no restart.
- Reference restart. A restart of the reference labels.
- Engine restart. A restart of the engine labels, after every offset of the engine.
- Match. Each reference exit takes the engine exit within MATCH_TOL frames with the smallest
  absolute difference. The matching is one to one and greedy by that difference. Ties go to the
  earlier engine exit. An unmatched reference restart is "no stoppage" when the engine has no
  `D` frame between the reference onset and the reference exit. It is "exit far" otherwise.
  For an unmatched restart the engine team is the engine label at the reference exit, or the
  first live engine label after it when the engine is still `D` there.
- Restart kind. Metrica: the restart event that starts at the reference exit
  (tools/events_gt.py rules): the SET PIECE subtype, or GOAL KICK on any event type. No tagged
  restart at the exit gives "pause". DFL: the set-piece event of the DFL event file nearest to
  the reference exit within DFL_EVENT_TOL frames (ThrowIn, CornerKick, FreeKick, GoalKick,
  KickOff, Penalty, RefereeBall). The event frame is the first ball frame N of the half plus
  the EventTime after the time T of that frame, times the frame rate. A restart with no event
  takes the out geometry at the reference onset: "geom_touchline", "geom_goal_line" or
  "geom_inside" (the card's fallback).
- Stopping event (Metrica only). The stopping event of tools/events_gt.py that ends at the
  reference onset. Its team is the team that put the ball out or committed the stop.
- Out geometry at an onset. The first visible frame in [onset - GEOM_PRE, onset + GEOM_POST]
  with the ball beyond a pitch line gives that line ("seen"). The largest excess decides at a
  corner. Else the last visible frame in [onset - GEOM_BACK, onset], when the ball moves at
  HEADING_MIN_SPEED or more and heads for a line within OUT_MAX_RANGE_M, gives the line it
  heads for ("heading"). Else the onset is "inside". A goal line is the line x = 0 or
  x = length. A touchline is y = 0 or y = width.
- Last contact. The last row of the engine event table with a team (kinds release, control,
  touch, dribble, and the goalkeeper hold rows of C13b) before the crossing frame of the out
  geometry (the onset when the onset is inside), within LAST_CONTACT_S.

Rules. Each rule gives a team or nothing. An anchor is an (onset, exit, restart frame). For an
engine anchor the restart frame is the detected candidate frame (the exit minus the offset of
the candidate kind). For a reference anchor it is the exit.

- a: out over a touchline: the team that did not make the last contact (Law 15).
- b: out over a goal line: the team that did not make the last contact (Laws 16 and 17). The
  record says "corner" when the last contact is by the team that defends that goal line and
  "goal_kick" otherwise. The defending team comes from tracks.goalkeeper_slots (own goal x).
- c: a goalkeeper hold gap (deadball.keeper_hold_gaps, C13b) that starts in
  [onset - GEOM_PRE, exit] names the team of that goalkeeper. The last such gap decides.
- c2: the ball is visible inside a team's own penalty area within R_TOUCH of that team's
  goalkeeper-like slot on a frame of [restart frame - RESTART_K, restart frame + RESTART_K].
  The nearest goalkeeper decides.
- d: a kickoff (C6). The exit is a games.py kickoff: the engine kickoff team. The exit is a
  kickoff after a GOALS window of the engine: that team. Else the ball is visible within
  KICKOFF_TAKER_M of the centre mark on a frame of [restart frame - RESTART_K, restart frame +
  RESTART_K]: possession.mark_team at the restart frame.
- e: the engine owner before the stoppage: the last `H` or `A` engine label before the onset.
- f: the nearest player within REAPPEAR_TEAM_M over [restart frame - RESTART_K, restart frame +
  RESTART_K], the window of deadball.restarts.
- k: the team of the engine restart in restart_list (engine anchor only). This is what the
  engine uses today. None means that the owner before the stoppage carries over.
- combined: the first team of a, b, c, c2, d in this order. Else f. Else e.

Throw-in at the throw (C14 avenue a). An inward crossing is a visible frame t with the ball
inside the pitch whose previous visible frame p lies at most CROSS_GAP frames before t, with the
ball beyond a touchline at p and 0 <= x <= length at p and at t. The crossing is "rested" when a
frame of deadball.stationary_mask with the ball beyond the touchline lies in the visible run of
p, at or before p.

The script prints and stores summary statistics only. `exposure` reads no ground truth and no
reference. numpy and pandas only. Deterministic.
"""

import argparse
import datetime
import json
import mmap
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (ROOT, os.path.join(ROOT, "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

TEAMS = ("H", "A")
OTHER = {"H": "A", "A": "H"}
MATCH_TOL = 125            # frames, the window of the card (5 s at 25 fps)
GEOM_PRE = 25              # frames before an onset where a seen crossing counts (1 s)
GEOM_POST = 50             # frames after an onset where a seen crossing counts (2 s)
GEOM_BACK = 125            # frames back for the last visible ball of the heading test (5 s)
LAST_CONTACT_S = 10.0      # seconds back for the last contact before the crossing
CROSS_GAP = 25             # frames: the largest gap between p and t of an inward crossing
DFL_EVENT_TOL = 100        # frames between a DFL set-piece event and a reference exit (4 s)
TAG_MARGIN = 250           # frames around a Smart Tagging window to look for a restart (10 s)
CONTACT_KINDS = ("release", "control", "touch", "dribble")
RULES = ("a", "b", "c", "c2", "d", "e", "f", "k", "combined")
LAW_ORDER = ("a", "b", "c", "c2", "d")
# The team-flip stretches of LEARNINGS C12 avenue (a) and C10 result (4), by their first frame.
NAMED_STRETCHES = (("C12", 54264), ("C12", 79091), ("C12", 85173), ("C12", 186800),
                   ("C10", 27924), ("C10", 44826), ("C10", 66124), ("C10", 85888),
                   ("C10", 163975))
METRICA_KINDS = (("KICK OFF", "kick_off"), ("PENALTY", "penalty"), ("CORNER KICK", "corner"),
                 ("GOAL KICK", "goal_kick"), ("THROW IN", "throw_in"),
                 ("FREE KICK", "free_kick"))
DFL_KINDS = {"KickOff": "kick_off", "ThrowIn": "throw_in", "CornerKick": "corner",
             "FreeKick": "free_kick", "GoalKick": "goal_kick", "Penalty": "penalty",
             "RefereeBall": "drop_ball"}


# -- label runs and matching ----------------------------------------------------------------------

def label_runs(labels):
    """[(start, stop, label)]: the maximal runs of one label, half-open, in frame order."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return []
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [labels.size]))
    return [(int(a), int(b), str(labels[a])) for a, b in zip(starts, stops)]


def restarts_of(labels):
    """[(onset, exit, team)]: every `D` run that the next frame ends with `H` or `A`."""
    runs = label_runs(labels)
    out = []
    for i, (start, stop, label) in enumerate(runs):
        if label == "D" and i + 1 < len(runs) and runs[i + 1][2] in TEAMS:
            out.append((start, stop, runs[i + 1][2]))
    return out


def match_exits(ref_exits, eng_exits, tol=MATCH_TOL):
    """[engine index or -1] for each reference exit: one to one, greedy by the difference.

    Each pair has an absolute difference of at most `tol` frames. The pair with the smallest
    difference goes first. Ties go to the earlier engine exit, then to the earlier reference exit.
    """
    eng = np.asarray(eng_exits, dtype=np.int64)
    pairs = []
    for i, ref in enumerate(ref_exits):
        lo = int(np.searchsorted(eng, int(ref) - tol, side="left"))
        hi = int(np.searchsorted(eng, int(ref) + tol, side="right"))
        pairs += [(abs(int(eng[j]) - int(ref)), int(eng[j]), i, j) for j in range(lo, hi)]
    out, used = [-1] * len(ref_exits), set()
    for _, _, i, j in sorted(pairs):
        if out[i] == -1 and j not in used:
            out[i] = j
            used.add(j)
    return out


def first_live_label(labels, frame):
    """The first `H` or `A` label at or after `frame` before an `X`, else None."""
    labels = np.asarray(labels)
    for f in range(int(frame), labels.size):
        if labels[f] in TEAMS:
            return str(labels[f])
        if labels[f] == "X":
            return None
    return None


def last_live_label(labels, frame):
    """The last `H` or `A` label before `frame` after an `X`, else None."""
    labels = np.asarray(labels)
    for f in range(int(frame) - 1, -1, -1):
        if labels[f] in TEAMS:
            return str(labels[f])
        if labels[f] == "X":
            return None
    return None


# -- geometry, contacts and rules ----------------------------------------------------------------

def line_of(x, y, pitch):
    """"goal0", "goalL", "touch" or None: the pitch line the point (m) lies beyond."""
    length, width = pitch
    excess = {"goal0": -x, "goalL": x - length, "touch": max(-y, y - width)}
    best = max(sorted(excess), key=lambda k: excess[k])
    return best if excess[best] > 0.0 else None


def heading_line(x, y, vx, vy, pitch):
    """(line, distance m) of the first pitch line on the heading from (x, y), or (None, inf)."""
    length, width = pitch
    speed = float(np.hypot(vx, vy))
    if not np.isfinite(speed) or speed <= 0.0:
        return None, float("inf")
    ux, uy = vx / speed, vy / speed
    tx = (length - x) / ux if ux > 0 else ((0.0 - x) / ux if ux < 0 else np.inf)
    ty = (width - y) / uy if uy > 0 else ((0.0 - y) / uy if uy < 0 else np.inf)
    if tx <= ty:
        return ("goalL" if ux > 0 else "goal0"), float(tx)
    return "touch", float(ty)


def out_geometry(ball, onset, pitch, cfg):
    """(line or None, crossing frame, how): the out geometry of an onset (module docstring)."""
    x, y, visible = ball["x"], ball["y"], ball["visible"]
    n = visible.shape[0]
    for f in range(max(0, onset - GEOM_PRE), min(n, onset + GEOM_POST + 1)):
        if visible[f]:
            line = line_of(float(x[f]), float(y[f]), pitch)
            if line is not None:
                return line, int(f), "seen"
    lo = max(0, onset - GEOM_BACK)
    seen = np.flatnonzero(visible[lo:min(n, onset + 1)])
    if seen.size:
        last = lo + int(seen[-1])
        speed = ball["speed"][last]
        if np.isfinite(speed) and speed >= cfg.HEADING_MIN_SPEED:
            line, dist = heading_line(float(x[last]), float(y[last]), float(ball["vx"][last]),
                                      float(ball["vy"][last]), pitch)
            if line is not None and dist <= cfg.OUT_MAX_RANGE_M:
                return line, int(last), "heading"
    return None, int(onset), "inside"


def contact_rows(events):
    """(frames int64, teams, kinds): the event rows with a team that count as a contact.

    The kind is the row kind, with "receive" for a control with the note `receive` and
    "keeper_hold" for a C13b goalkeeper row.
    """
    if events is None or len(events) == 0:
        empty = np.zeros(0, dtype=object)
        return np.zeros(0, dtype=np.int64), empty, empty
    rows = events[events["kind"].isin(CONTACT_KINDS) & events["team"].isin(TEAMS)]
    notes = rows["notes"].astype(str).to_numpy() if "notes" in rows else \
        np.full(len(rows), "", dtype=object)
    kinds = []
    for kind, note in zip(rows["kind"].to_numpy(dtype=object), notes):
        if note.startswith("keeper_hold"):
            kinds.append("keeper_hold")
        elif kind == "control" and note.startswith("receive"):
            kinds.append("receive")
        elif kind == "control" and note.startswith("redirect"):
            kinds.append("redirect")
        else:
            kinds.append(str(kind))
    return (rows["frame"].to_numpy(dtype=np.int64), rows["team"].to_numpy(dtype=object),
            np.asarray(kinds, dtype=object))


def last_contact(frames, teams, before, lookback, kinds=None):
    """(team, frame, kind) of the last contact strictly before `before` within `lookback`."""
    idx = int(np.searchsorted(frames, int(before), side="left")) - 1
    if idx >= 0 and frames[idx] >= int(before) - int(lookback):
        kind = str(kinds[idx]) if kinds is not None else None
        return str(teams[idx]), int(frames[idx]), kind
    return None, None, None


def combine(answers, order=LAW_ORDER):
    """The combined team: the first team of `order`, else rule f, else rule e (or None)."""
    for rule in tuple(order) + ("f", "e"):
        if answers.get(rule) in TEAMS:
            return answers[rule]
    return None


def inward_crossings(ball, pitch, rest):
    """[(frame, rested)]: the inward touchline crossings of the module docstring."""
    length, width = pitch
    x, y, visible, run_id = ball["x"], ball["y"], ball["visible"], ball["run_id"]
    with np.errstate(invalid="ignore"):
        on_x = visible & (x >= 0.0) & (x <= length)
        beyond = on_x & ((y < 0.0) | (y > width))
        inside = on_x & (y >= 0.0) & (y <= width)
    rest_beyond = np.asarray(rest, dtype=bool) & beyond
    seen = np.flatnonzero(visible)
    out = []
    for i in range(1, seen.size):
        t, p = int(seen[i]), int(seen[i - 1])
        if t - p > CROSS_GAP or not (beyond[p] and inside[t]):
            continue
        run = run_id[p]
        lo = p
        while lo - 1 >= 0 and run_id[lo - 1] == run:
            lo -= 1
        out.append((t, bool(rest_beyond[lo:p + 1].any())))
    return out


class Context(object):
    """The engine arrays that the rules read for one match."""

    def __init__(self, game, cfg):
        from engine import data, deadball, possession, tracks

        self.cfg = cfg
        self.game = game
        self.inputs = possession.prepare(cfg, game)
        self.labels, self.parts = possession.run(cfg, return_parts=True, inputs=self.inputs)
        self.labels = np.asarray(self.labels)
        self.pitch = data.pitch_m(game)
        self.blocks = self.inputs["blocks"]
        self.ball, self.feats = self.inputs["ball"], self.inputs["feats"]
        self.mark = self.inputs["mark"]
        tactical = possession.engine_tactical(cfg, game)
        slots = tracks.goalkeeper_slots(tactical, self.blocks, self.pitch, cfg)
        self.own_goal = {key: float(info["own_goal_x"]) for key, info in slots.items()}
        self.keepers = tracks.goalkeeper_positions(tactical, slots, self.blocks, self.pitch)
        self.hold_gaps = deadball.keeper_hold_gaps(self.ball, self.keepers, self.pitch, cfg)
        self.contacts = contact_rows(self.parts["events"])
        self.stationary = deadball.stationary_mask(self.ball, cfg)
        self.kickoffs = {int(f): self.parts["kick_teams"].get(h)
                         for h, f in self.inputs["kickoffs"].items()}
        self.goal_kickoffs = {int(row[0]): row[1] for row in self.parts["goal_kickoffs"]}
        self.restart_team = {int(f): t for f, t in self.parts["restart_list"]}
        cands = self.parts["restarts"]
        cands = cands[cands["confirmed"]]
        self.exit_kind = {}
        for frame, kind in zip(cands["frame"], cands["kind"]):
            self.exit_kind.setdefault(int(frame), str(kind))
        self.deltas = {"kick": int(cfg.DELTA_RESTART_KICK),
                       "reappear": int(cfg.DELTA_RESTART_REAPPEAR)}
        self.fps = float(self.inputs["fps"])

    def half_of(self, frame):
        """The half of the graded block that holds `frame`, else None."""
        for half, first, last in self.blocks:
            if first <= frame <= last:
                return int(half)
        return None

    def detected(self, exit_frame):
        """(restart frame, kind) of an engine exit: the exit minus the offset of its kind."""
        if exit_frame in self.kickoffs:
            return exit_frame, "kickoff"
        kind = self.exit_kind.get(int(exit_frame))
        return int(exit_frame) - self.deltas.get(kind, 0), kind

    def nearest(self, frame):
        """(distance, team) of the nearest player over the window of deadball.restarts."""
        from engine import deadball

        k = int(self.cfg.RESTART_K)
        n = self.labels.size
        return deadball._nearest_in_window(self.feats, max(0, frame - k), min(n, frame + k + 1))

    def keeper_at(self, frame):
        """Rule c2: the team whose goalkeeper is within R_TOUCH of the ball in his own box."""
        from engine.tracks import penalty_area_mask

        k = int(self.cfg.RESTART_K)
        x, y, visible = self.ball["x"], self.ball["y"], self.ball["visible"]
        best = (np.inf, None)
        for f in range(max(0, frame - k), min(self.labels.size, frame + k + 1)):
            if not visible[f]:
                continue
            for team in (0, 1):
                goal = self.keepers["own_goal_x"][f, team]
                if not np.isfinite(goal):
                    continue
                dist = float(np.hypot(self.keepers["x"][f, team] - x[f],
                                      self.keepers["y"][f, team] - y[f]))
                if dist <= self.cfg.R_TOUCH and dist < best[0] and \
                        bool(penalty_area_mask(x[f], y[f], goal, self.pitch, self.cfg)):
                    best = (dist, TEAMS[team])
        return best[1]

    def kickoff_team(self, exit_frame, frame):
        """Rule d (module docstring)."""
        from engine import possession

        if exit_frame in self.kickoffs:
            return self.kickoffs[exit_frame]
        if exit_frame in self.goal_kickoffs:
            return self.goal_kickoffs[exit_frame]
        k = int(self.cfg.RESTART_K)
        length, width = self.pitch
        lo, hi = max(0, frame - k), min(self.labels.size, frame + k + 1)
        x, y, visible = self.ball["x"][lo:hi], self.ball["y"][lo:hi], self.ball["visible"][lo:hi]
        with np.errstate(invalid="ignore"):
            near = visible & (np.hypot(x - length / 2.0, y - width / 2.0)
                              <= self.cfg.KICKOFF_TAKER_M)
        if not near.any():
            return None
        team, _ = possession.mark_team(self.mark, frame, self.cfg)
        return team

    def rules(self, onset, exit_frame, frame, engine_anchor):
        """{rule: team or None} plus the geometry fields for one anchor."""
        line, cross, how = out_geometry(self.ball, onset, self.pitch, self.cfg)
        lookback = int(round(LAST_CONTACT_S * self.fps))
        toucher, touch_frame, touch_kind = last_contact(self.contacts[0], self.contacts[1],
                                                        cross, lookback, self.contacts[2])
        answers = dict((rule, None) for rule in RULES)
        sub = None
        if line == "touch" and toucher is not None:
            answers["a"] = OTHER[toucher]
        if line in ("goal0", "goalL") and toucher is not None:
            answers["b"] = OTHER[toucher]
            half = self.half_of(onset)
            goal_x = 0.0 if line == "goal0" else float(self.pitch[0])
            defend = None
            for team in (0, 1):
                own = self.own_goal.get((half, team))
                if own is not None and ((own == 0.0) == (goal_x == 0.0)):
                    defend = TEAMS[team]
            sub = "corner" if toucher == defend else "goal_kick"
        gaps = [g for g in self.hold_gaps if onset - GEOM_PRE <= g[0] <= exit_frame]
        if gaps:
            answers["c"] = TEAMS[gaps[-1][2]]
        answers["c2"] = self.keeper_at(frame)
        answers["d"] = self.kickoff_team(exit_frame, frame)
        answers["e"] = last_live_label(self.labels, onset)
        dist, team = self.nearest(frame)
        answers["f"] = team if dist <= self.cfg.REAPPEAR_TEAM_M else None
        if engine_anchor:
            answers["k"] = self.restart_team.get(int(exit_frame))
        answers["combined"] = combine(answers)
        return answers, {"line": line, "how": how, "cross": cross, "toucher": toucher,
                         "touch_frame": touch_frame, "touch_kind": touch_kind,
                         "goal_sub": sub}


# -- restart kinds of the references --------------------------------------------------------------

def metrica_kinds(path, game):
    """{video start frame: (kind, team letter)} of the tagged restarts of a Metrica event file."""
    import events_gt
    import games

    events = events_gt.to_video(events_gt.load_any(path), games.GAMES[game]["offsets"])
    home = events_gt.home_team_name(game)
    out = {}
    for event in events:
        if not events_gt.is_tagged_restart(event):
            continue
        kind = "set_piece_other"
        for name, label in METRICA_KINDS:
            if name in event["subtypes"]:
                kind = label
                break
        out.setdefault(int(event["start"]), (kind, "H" if event["team"] == home else "A"))
    return out


def metrica_out_consistency(path):
    """{"<restart subtype> <stop type> same|other": count}: the team of the last stopping event
    before each throw-in and corner of a Metrica event file against the restart team."""
    import events_gt

    events = events_gt.load_any(path)
    out = {}
    for i, event in enumerate(events):
        if event["type"] != "SET PIECE" or not ({"THROW IN", "CORNER KICK"}
                                                & set(event["subtypes"])):
            continue
        stops = [e for e in events[:i] if events_gt.is_stop(e)]
        if not stops:
            continue
        key = "{} {} {}".format(event["subtypes"][0], stops[-1]["type"],
                                "same" if stops[-1]["team"] == event["team"] else "other")
        out[key] = out.get(key, 0) + 1
    return out


def metrica_stops(path, game):
    """{video end frame: (stop type, team letter, video start frame)} of the stopping events."""
    import events_gt
    import games

    events = events_gt.to_video(events_gt.load_any(path), games.GAMES[game]["offsets"])
    home = events_gt.home_team_name(game)
    out = {}
    for event in events:
        if events_gt.is_stop(event) and events_gt.END_HALF not in event["subtypes"]:
            out.setdefault(int(event["end"]), (event["type"], "H" if event["team"] == home
                                               else "A", int(event["start"])))
    return out


def _parse_time(text):
    """A DFL ISO time as an aware datetime."""
    return datetime.datetime.fromisoformat(text)


def dfl_half_starts(positions_path):
    """{half: (first N, time T)} of the ball FrameSet of each half of a DFL positions file."""
    pattern = re.compile(rb'<Frame N="(\d+)" T="([^"]+)"')
    out = {}
    with open(positions_path, "rb") as handle:
        view = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            pos = 0
            while True:
                pos = view.find(b'TeamId="BALL"', pos)
                if pos < 0:
                    break
                head = view.rfind(b"<FrameSet", 0, pos)
                section = re.search(rb'GameSection="(\w+)"', view[head:pos + 200]).group(1)
                found = pattern.search(view[pos:pos + 2000])
                half = 1 if section == b"firstHalf" else 2
                out[half] = (int(found.group(1)), _parse_time(found.group(2).decode()))
                pos += 1
        finally:
            view.close()
    return out


def dfl_home_id(game):
    """The DFL team id of the home team (`H`), from the _build.json beside the bundle, or None."""
    import games

    meta = games.GAMES[game].get("assets", {}).get("atd_meta")
    if not meta:
        return None
    path = os.path.join(games.DATA_DIR, meta.replace("_atd.xml", "_build.json"))
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)["home"]["team_id"]


def dfl_kinds(events_path, positions_path, game):
    """{video frame: (kind, team letter or id)} of the set-piece events of a DFL event file.

    The team is `H` or `A` when the _build.json of the bundle names the home team id.
    """
    import games

    entry = games.GAMES[game]
    home = dfl_home_id(game)
    fps = float(entry["fps"])
    starts = dfl_half_starts(positions_path)
    out = {}
    for event in ET.parse(events_path).getroot().iter("Event"):
        child = next((c for c in event if c.tag in DFL_KINDS), None)
        if child is None:
            continue
        when = _parse_time(event.get("EventTime"))
        half = 2 if 2 in starts and when >= starts[2][1] else 1
        first_n, first_t = starts[half]
        frame_n = first_n + int(round((when - first_t).total_seconds() * fps))
        video = frame_n - int(entry["offsets"][half])
        team = child.get("Team")
        if team is None:
            team = next((d.get("Team") for d in child.iter() if d.get("Team")), None)
        if home is not None and team is not None:
            team = "H" if team == home else "A"
        out.setdefault(int(video), (DFL_KINDS[child.tag], team))
    return out


def kind_of_exit(kinds, exit_frame, tol):
    """(kind, team, offset) of the restart event nearest `exit_frame` within `tol` frames."""
    if not kinds:
        return "none", None, None
    frames = np.array(sorted(kinds), dtype=np.int64)
    idx = int(np.argmin(np.abs(frames - int(exit_frame))))
    offset = int(frames[idx]) - int(exit_frame)
    if abs(offset) > tol:
        return "none", None, None
    kind, team = kinds[int(frames[idx])]
    return kind, team, offset


# -- the commands ---------------------------------------------------------------------------------

def read_labels(path, n_frames=None):
    """The `label` column of a `frame,label` CSV as an array of one-letter strings."""
    table = pd.read_csv(path, dtype={"frame": np.int64, "label": str})
    labels = table.sort_values("frame")["label"].to_numpy().astype("<U1")
    if n_frames is not None and labels.size != n_frames:
        raise ValueError("{}: {} rows, expected {}".format(path, labels.size, n_frames))
    return labels


def _quantiles(values):
    """{n, p10, p25, p50, p75, p90, min, max} of a list of numbers (empty: n only)."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0}
    out = {"n": int(values.size), "min": float(values.min()), "max": float(values.max())}
    for q in (10, 25, 50, 75, 90):
        out["p{}".format(q)] = float(np.percentile(values, q))
    return out


def measure(args):
    """The `measure` command: restart-team measurements on one external reference match."""
    import games
    from engine.config import Config

    entry = games.GAMES[args.game]
    cfg = Config.default()
    ctx = Context(args.game, cfg)
    ref = read_labels(args.ref, int(entry["n_video_frames"]))
    engine_check = None
    if args.engine_csv:
        engine_check = bool(np.array_equal(read_labels(args.engine_csv), ctx.labels))
    kinds, tol, consistency, stops = {}, 0, None, {}
    if args.metrica_events:
        kinds, tol = metrica_kinds(args.metrica_events, args.game), 0
        consistency = metrica_out_consistency(args.metrica_events)
        stops = metrica_stops(args.metrica_events, args.game)
    elif args.dfl_events:
        kinds, tol = dfl_kinds(args.dfl_events, args.dfl_positions, args.game), DFL_EVENT_TOL
    ref_restarts = restarts_of(ref)
    eng_restarts = restarts_of(ctx.labels)
    match = match_exits([r[1] for r in ref_restarts], [e[1] for e in eng_restarts])
    records = []
    for i, (onset, exit_frame, team) in enumerate(ref_restarts):
        kind, kind_team, kind_offset = kind_of_exit(kinds, exit_frame, tol)
        rec = {"onset": onset, "exit": exit_frame, "team": team, "kind": kind,
               "kind_team": kind_team, "kind_offset": kind_offset}
        j = match[i]
        if j >= 0:
            e_onset, e_exit, e_team = eng_restarts[j]
            frame, e_kind = ctx.detected(e_exit)
            answers, geo = ctx.rules(e_onset, e_exit, frame, True)
            rec.update({"status": "matched", "engine_exit": e_exit, "engine_onset": e_onset,
                        "engine_team": e_team, "engine_kind": e_kind,
                        "exit_offset": e_exit - exit_frame, "eng_rules": answers,
                        "in_restart_list": e_exit in ctx.restart_team,
                        "eng_geo": geo})
        else:
            dead = bool((ctx.labels[onset:exit_frame] == "D").any())
            rec.update({"status": "exit_far" if dead else "no_stoppage",
                        "engine_team": first_live_label(ctx.labels, exit_frame)})
        answers, geo = ctx.rules(onset, exit_frame, exit_frame, False)
        rec.update({"ref_rules": answers, "ref_geo": geo, "kind_source": "event",
                    "stop": stops.get(int(onset))})
        if kind in ("none", "pause"):
            rec["kind"] = "geom_" + ("inside" if geo["line"] is None else
                                     "touchline" if geo["line"] == "touch" else "goal_line")
            rec["kind_source"] = "geometry"
        records.append(rec)
    # Throw-in at the throw: inward crossings against the reference throw-ins.
    crossings = inward_crossings(ctx.ball, ctx.pitch, ctx.stationary)
    cross_frames = np.array([c[0] for c in crossings], dtype=np.int64)
    throw = []
    for rec in records:
        if rec["kind"] != "throw_in":
            continue
        lo, hi = rec["exit"] - MATCH_TOL, rec["exit"] + MATCH_TOL
        near = [c for c in crossings if lo <= c[0] <= hi]
        before = ctx.ball["visible"][max(0, rec["exit"] - GEOM_PRE):rec["exit"]].any()
        best = min(near, key=lambda c: (abs(c[0] - rec["exit"]), c[0])) if near else None
        throw.append({"exit": rec["exit"], "visible_before": bool(before),
                      "crossing_offset": (best[0] - rec["exit"]) if best else None,
                      "rested": best[1] if best else None,
                      "engine_offset": rec.get("exit_offset")})
    graded = np.zeros(ctx.labels.size, dtype=bool)
    for _, first, last in ctx.blocks:
        graded[first:last + 1] = True
    out = {"game": args.game, "mode": "measure", "engine_equal_csv": engine_check,
           "n_ref_restarts": len(ref_restarts), "n_engine_restarts": len(eng_restarts),
           "n_kind_events": len(kinds), "records": records, "throw_ins": throw,
           "out_consistency": consistency,
           "n_crossings_graded": int(graded[cross_frames].sum()) if cross_frames.size else 0,
           "n_crossings_rested_graded": int(sum(1 for f, r in crossings if r and graded[f]))}
    write_json(args.out, out)
    return out


def exposure(args):
    """The `exposure` command: BAR-ATH restarts without labels (measurements 3 and 4)."""
    from engine import data
    from engine.config import Config

    cfg = Config.default()
    ctx = Context(args.game, cfg)
    engine_check = None
    if args.engine_csv:
        engine_check = bool(np.array_equal(read_labels(args.engine_csv), ctx.labels))
    runs = label_runs(ctx.labels)
    run_at = {}
    for start, stop, label in runs:
        run_at[start] = (stop, label)
    records = []
    for onset, exit_frame, written in restarts_of(ctx.labels):
        frame, kind = ctx.detected(exit_frame)
        answers, geo = ctx.rules(onset, exit_frame, frame, True)
        stop, _ = run_at[exit_frame]
        records.append({"onset": onset, "exit": exit_frame, "written": written,
                        "kind": kind, "kickoff": exit_frame in ctx.kickoffs
                        or exit_frame in ctx.goal_kickoffs, "in_restart_list":
                        exit_frame in ctx.restart_team, "rules": answers, "geo": geo,
                        "run_frames": stop - exit_frame})
    named = []
    exits = np.array([r["exit"] for r in records], dtype=np.int64)
    for source, frame in NAMED_STRETCHES:
        idx = int(np.argmin(np.abs(exits - frame)))
        rec = records[idx]
        named.append({"source": source, "frame": frame, "label_now": str(ctx.labels[frame]),
                      "exit": rec["exit"], "offset": frame - rec["exit"], "record": rec})
    # Throw-in tags of Smart Tagging (measurement 4).
    tagging = data.load_tagging(args.game)
    tags = tagging[(tagging["code"] == "SET PIECES")
                   & (tagging["type"].astype(str).str.lower() == "throw-in")]
    crossings = inward_crossings(ctx.ball, ctx.pitch, ctx.stationary)
    graded = np.zeros(ctx.labels.size, dtype=bool)
    for _, first, last in ctx.blocks:
        graded[first:last + 1] = True
    throw = []
    used = set()
    for start, end in zip(tags["start_frame"].astype(int), tags["end_frame"].astype(int)):
        lo, hi = start - TAG_MARGIN, end + TAG_MARGIN
        inside = [e for e in exits if lo <= e <= hi]
        exit_frame = min(inside, key=lambda e: (abs(e - start), e)) if inside else None
        near = [c for c in crossings if lo <= c[0] <= hi]
        used.update(c[0] for c in near)
        best = min(near, key=lambda c: (abs(c[0] - start), c[0])) if near else None
        rested = [c for c in near if c[1]]
        best_rest = min(rested, key=lambda c: (abs(c[0] - start), c[0])) if rested else None
        to_exit = rest_exit = None
        if exit_frame is not None:
            # Against the engine exit, the crossing nearest to that exit.
            to_exit = min(near, key=lambda c: (abs(c[0] - exit_frame), c[0])) if near else None
            rest_exit = min(rested, key=lambda c: (abs(c[0] - exit_frame), c[0])) \
                if rested else None
        throw.append({
            "tag_start": start, "tag_end": end,
            "exit_minus_start": None if exit_frame is None else int(exit_frame - start),
            "exit_in_window": None if exit_frame is None else bool(start <= exit_frame < end),
            "n_crossings": len(near), "n_rested": len(rested),
            "cross_minus_start": None if best is None else int(best[0] - start),
            "rest_cross_minus_start": None if best_rest is None else int(best_rest[0] - start),
            "rest_cross_minus_exit": None if rest_exit is None
            else int(rest_exit[0] - exit_frame),
            "cross_minus_exit": None if to_exit is None else int(to_exit[0] - exit_frame)})
    away = [c for c in crossings if graded[c[0]] and c[0] not in used]
    out = {"game": args.game, "mode": "exposure", "engine_equal_csv": engine_check,
           "records": records, "named": named, "throw_tags": throw,
           "n_crossings_graded": int(sum(1 for c in crossings if graded[c[0]])),
           "n_rested_graded": int(sum(1 for c in crossings if graded[c[0]] and c[1])),
           "n_crossings_away_from_tags": len(away),
           "n_rested_away_from_tags": int(sum(1 for c in away if c[1]))}
    write_json(args.out, out)
    return out


def write_json(path, payload):
    """Write `payload` as sorted, indented JSON (numpy scalars as Python numbers)."""
    def convert(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        raise TypeError(type(value))
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, default=convert)
        handle.write("\n")


# -- report ---------------------------------------------------------------------------------------

def _pct(right, total):
    """'right/total (p %)' or '0/0'."""
    return "{}/{} ({:.1f} %)".format(right, total, 100.0 * right / total) if total else "0/0"


def score_rule(records, key, rule, only_matched):
    """(applies, right) of one rule over the records."""
    applies = right = 0
    for rec in records:
        if only_matched and rec["status"] != "matched":
            continue
        answers = rec.get(key)
        if answers is None:
            continue
        team = answers.get(rule)
        if team in TEAMS:
            applies += 1
            right += int(team == rec["team"])
    return applies, right


def report_measure(payloads):
    """Markdown tables of measurements 1, 2 and 4 for the `measure` payloads."""
    lines = []
    lines.append("### Measurement 1: the engine's first live team after a restart\n")
    lines.append("| match | restarts | matched | team right | team wrong | exit far | "
                 "no stoppage | unmatched: engine label right | exit offset p50 [p10, p90] |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    total = dict.fromkeys(("n", "m", "r", "w", "far", "none", "ur", "un"), 0)
    offsets_all = []
    for p in payloads:
        recs = p["records"]
        m = [r for r in recs if r["status"] == "matched"]
        right = sum(1 for r in m if r["engine_team"] == r["team"])
        far = sum(1 for r in recs if r["status"] == "exit_far")
        none = sum(1 for r in recs if r["status"] == "no_stoppage")
        un = [r for r in recs if r["status"] != "matched"]
        ur = sum(1 for r in un if r["engine_team"] == r["team"])
        offs = [r["exit_offset"] for r in m]
        offsets_all += offs
        q = _quantiles(offs)
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            p["game"], len(recs), len(m), _pct(right, len(m)), len(m) - right, far, none,
            _pct(ur, len(un)), _fmt_q(q)))
        for key, val in (("n", len(recs)), ("m", len(m)), ("r", right), ("w", len(m) - right),
                         ("far", far), ("none", none), ("ur", ur), ("un", len(un))):
            total[key] += val
    lines.append("| **total** | {n} | {m} | {rr} | {w} | {far} | {none} | {uu} | {q} |".format(
        rr=_pct(total["r"], total["m"]), uu=_pct(total["ur"], total["un"]),
        q=_fmt_q(_quantiles(offsets_all)), **total))
    lines.append("")
    lines.append("By restart kind (all matches; matched restarts; engine team right / matched; "
                 "restarts of the kind that are unmatched):\n")
    kinds = sorted(set(r["kind"] for p in payloads for r in p["records"]))
    lines.append("| kind | " + " | ".join(p["game"] for p in payloads) + " | total |")
    lines.append("|---|" + "---|" * (len(payloads) + 1))
    for kind in kinds:
        cells, tm, tr, tu = [], 0, 0, 0
        for p in payloads:
            recs = [r for r in p["records"] if r["kind"] == kind]
            m = [r for r in recs if r["status"] == "matched"]
            right = sum(1 for r in m if r["engine_team"] == r["team"])
            cells.append("{}/{} u{}".format(right, len(m), len(recs) - len(m)) if recs else "")
            tm, tr, tu = tm + len(m), tr + right, tu + len(recs) - len(m)
        lines.append("| {} | {} | {} |".format(kind, " | ".join(cells),
                                                 "{} u{}".format(_pct(tr, tm), tu)))
    lines.append("")
    for key, title, only in (("eng_rules", "engine anchor, matched restarts", True),
                             ("ref_rules", "reference anchor, all restarts", False)):
        lines.append("### Measurement 2: the rules on the {}\n".format(title))
        lines.append("Each cell: right/applies (accuracy). Rule k is the engine restart team.\n")
        rules = [r for r in RULES if not (r == "k" and key == "ref_rules")]
        lines.append("| match | n | " + " | ".join(rules) + " |")
        lines.append("|---|---|" + "---|" * len(rules))
        tot = dict((r, [0, 0]) for r in rules)
        tot_n = 0
        for p in payloads:
            recs = [r for r in p["records"] if not only or r["status"] == "matched"]
            tot_n += len(recs)
            cells = []
            for rule in rules:
                a, r = score_rule(recs, key, rule, only)
                tot[rule][0] += a
                tot[rule][1] += r
                cells.append(_pct(r, a))
            lines.append("| {} | {} | {} |".format(p["game"], len(recs), " | ".join(cells)))
        lines.append("| **total** | {} | {} |".format(
            tot_n, " | ".join(_pct(tot[r][1], tot[r][0]) for r in rules)))
        lines.append("")
        lines.append("By restart kind, all matches ({}): right/applies.\n".format(title))
        lines.append("| kind | n | " + " | ".join(rules) + " |")
        lines.append("|---|---|" + "---|" * len(rules))
        for kind in kinds:
            recs = [r for p in payloads for r in p["records"]
                    if r["kind"] == kind and (not only or r["status"] == "matched")]
            cells = []
            for rule in rules:
                a, r = score_rule(recs, key, rule, only)
                cells.append(_pct(r, a))
            lines.append("| {} | {} | {} |".format(kind, len(recs), " | ".join(cells)))
        lines.append("")
        geo_key = "eng_geo" if key == "eng_rules" else "ref_geo"
        lines.append("Out geometry at the onset ({}): count by line and how it was found; "
                     "goal-line restarts by the corner / goal-kick reading of rule b.\n".format(
                         title))
        lines.append("| match | touch seen | touch heading | goal seen | goal heading | inside | "
                     "b corner | b goal_kick | no last contact (a or b line) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for p in payloads:
            recs = [r for r in p["records"] if not only or r["status"] == "matched"]
            geos = [r[geo_key] for r in recs]
            cnt = dict()
            for g in geos:
                line = "inside" if g["line"] is None else ("touch" if g["line"] == "touch"
                                                           else "goal")
                cnt[(line, g["how"])] = cnt.get((line, g["how"]), 0) + 1
            corner = sum(1 for g in geos if g["goal_sub"] == "corner")
            gk = sum(1 for g in geos if g["goal_sub"] == "goal_kick")
            nolc = sum(1 for g in geos if g["line"] is not None and g["toucher"] is None)
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                p["game"], cnt.get(("touch", "seen"), 0), cnt.get(("touch", "heading"), 0),
                cnt.get(("goal", "seen"), 0), cnt.get(("goal", "heading"), 0),
                cnt.get(("inside", "inside"), 0), corner, gk, nolc))
        lines.append("")
    lines.append("### Measurements 1 and 2: supporting tables\n")
    for key, geo_key, title, only in (("eng_rules", "eng_geo", "engine anchor", True),
                                      ("ref_rules", "ref_geo", "reference anchor", False)):
        lines.append("Rules a and b by the kind of the last contact and its age ({}, all "
                     "matches): right/applies.\n".format(title))
        lines.append("| last contact | a | b | age under 1 s (a+b) | age 1 to 3 s (a+b) | "
                     "age over 3 s (a+b) |")
        lines.append("|---|---|---|---|---|---|")
        recs = [r for p in payloads for r in p["records"]
                if (not only or r["status"] == "matched")]
        fps = 25.0
        for kind in sorted(set(r[geo_key]["touch_kind"] for r in recs
                               if r[geo_key]["touch_kind"])):
            sub = [r for r in recs if r[geo_key]["touch_kind"] == kind]
            cells = []
            for rule in ("a", "b"):
                a, r = score_rule(sub, key, rule, only)
                cells.append(_pct(r, a))
            for lo, hi in ((0.0, 1.0), (1.0, 3.0), (3.0, 1e9)):
                inb = [r for r in sub if lo * fps <= r[geo_key]["cross"]
                       - r[geo_key]["touch_frame"] < hi * fps]
                a1, r1 = score_rule(inb, key, "a", only)
                a2, r2 = score_rule(inb, key, "b", only)
                cells.append(_pct(r1 + r2, a1 + a2))
            lines.append("| {} | {} |".format(kind, " | ".join(cells)))
        lines.append("")
    lines.append("Metrica only: the engine's last contact (reference anchor) against the team of "
                 "the stopping event that ends at the reference onset, for out restarts (rule a "
                 "or b line). Gap: stopping event start minus last contact frame.\n")
    lines.append("| match | out restarts with a stopping event | last contact = stopping team | "
                 "gap <= 12: same / n | gap > 12: same / n | no last contact |")
    lines.append("|---|---|---|---|---|---|")
    for p in payloads:
        recs = [r for r in p["records"] if r.get("stop") and r["ref_geo"]["line"] is not None]
        if not recs:
            continue
        with_lc = [r for r in recs if r["ref_geo"]["toucher"] is not None]
        same = [r for r in with_lc if r["ref_geo"]["toucher"] == r["stop"][1]]
        near = [r for r in with_lc if r["stop"][2] - r["ref_geo"]["touch_frame"] <= 12]
        far = [r for r in with_lc if r["stop"][2] - r["ref_geo"]["touch_frame"] > 12]
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            p["game"], len(recs), _pct(len(same), len(with_lc)),
            _pct(sum(1 for r in near if r["ref_geo"]["toucher"] == r["stop"][1]), len(near)),
            _pct(sum(1 for r in far if r["ref_geo"]["toucher"] == r["stop"][1]), len(far)),
            len(recs) - len(with_lc)))
    lines.append("")
    cons = [p for p in payloads if p.get("out_consistency")]
    if cons:
        lines.append("Reference self-consistency (Metrica): the team of the last stopping event "
                     "before each throw-in and corner against the restart team.\n")
        lines.append("| match | counts |")
        lines.append("|---|---|")
        for p in cons:
            lines.append("| {} | {} |".format(p["game"], ", ".join(
                "{}: {}".format(k, p["out_consistency"][k])
                for k in sorted(p["out_consistency"]))))
        lines.append("")
    lines.append("Matched restarts by the engine restart team (rule k): where the engine "
                 "team is wrong, and what the rules say where k is None (engine anchor).\n")
    lines.append("| match | matched | k right | k wrong | k None | engine wrong: k right / "
                 "k wrong / k None | k None: engine right | k None: laws a..d right/applies | "
                 "k None: combined right | k None: e right |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    agg = [0] * 12
    for p in payloads:
        m = [r for r in p["records"] if r["status"] == "matched"]
        k = [r["eng_rules"]["k"] for r in m]
        right = [r for r, t in zip(m, k) if t == r["team"]]
        wrong = [r for r, t in zip(m, k) if t in TEAMS and t != r["team"]]
        none = [r for r, t in zip(m, k) if t not in TEAMS]
        bad = [r for r in m if r["engine_team"] != r["team"]]
        bad_split = [sum(1 for r in bad if r["eng_rules"]["k"] == r["team"]),
                     sum(1 for r in bad if r["eng_rules"]["k"] in TEAMS
                         and r["eng_rules"]["k"] != r["team"]),
                     sum(1 for r in bad if r["eng_rules"]["k"] not in TEAMS)]
        laws = [next((r["eng_rules"][x] for x in LAW_ORDER if r["eng_rules"][x] in TEAMS), None)
                for r in none]
        laws_app = [(r, t) for r, t in zip(none, laws) if t in TEAMS]
        row = [len(m), len(right), len(wrong), len(none)] + bad_split + [
            sum(1 for r in none if r["engine_team"] == r["team"]),
            sum(1 for r, t in laws_app if t == r["team"]), len(laws_app),
            sum(1 for r in none if r["eng_rules"]["combined"] == r["team"]),
            sum(1 for r in none if r["eng_rules"]["e"] == r["team"])]
        agg = [a + b for a, b in zip(agg, row)]
        lines.append("| {} | {} | {} | {} | {} | {} / {} / {} | {} | {}/{} | {} | {} |".format(
            p["game"], *row))
    lines.append("| **total** | {} | {} | {} | {} | {} / {} / {} | {} | {}/{} | {} | {} |".format(
        *agg))
    lines.append("")
    lines.append("Combined against the engine on matched restarts (engine anchor):\n")
    lines.append("| match | both right | engine only right | combined only right | both wrong |")
    lines.append("|---|---|---|---|---|")
    agg = [0, 0, 0, 0]
    for p in payloads:
        cells = [0, 0, 0, 0]
        for r in p["records"]:
            if r["status"] != "matched":
                continue
            e_ok = r["engine_team"] == r["team"]
            c_ok = r["eng_rules"]["combined"] == r["team"]
            cells[0 if e_ok and c_ok else 1 if e_ok else 2 if c_ok else 3] += 1
        agg = [a + c for a, c in zip(agg, cells)]
        lines.append("| {} | {} | {} | {} | {} |".format(p["game"], *cells))
    lines.append("| **total** | {} | {} | {} | {} |".format(*agg))
    lines.append("")
    lines.append("Kind-event alignment (event frame minus reference exit, frames; event team "
                 "against the reference restarting team):\n")
    lines.append("| match | set-piece events | restarts with a kind | offset p50 [p10, p90] | "
                 "event team = reference team |")
    lines.append("|---|---|---|---|---|")
    for p in payloads:
        recs = [r for r in p["records"] if r.get("kind_source", "event") == "event"
                and r["kind"] not in ("none", "pause")]
        offs = [r["kind_offset"] for r in recs if r["kind_offset"] is not None]
        agree = _event_team_agreement(p, recs)
        lines.append("| {} | {} | {} | {} | {} |".format(
            p["game"], p["n_kind_events"], len(recs), _fmt_q(_quantiles(offs)), agree))
    lines.append("")
    lines.append("### Measurement 4: the inward touchline crossing at the reference throw-ins\n")
    lines.append("| match | throw-ins | ball visible in the 25 frames before | crossing within "
                 "+-125 | rested | crossing minus exit p50 [p10, p90] | engine exit minus "
                 "exit p50 [p10, p90] | crossings on graded frames (rested) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    buckets = []
    for p in payloads:
        t = p["throw_ins"]
        buckets.append((p["game"], _buckets([x["crossing_offset"] for x in t
                                             if x["crossing_offset"] is not None])))
        cross = [x["crossing_offset"] for x in t if x["crossing_offset"] is not None]
        rested = sum(1 for x in t if x["rested"])
        eng = [x["engine_offset"] for x in t if x["engine_offset"] is not None]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} ({}) |".format(
            p["game"], len(t), sum(1 for x in t if x["visible_before"]), len(cross), rested,
            _fmt_q(_quantiles(cross)), _fmt_q(_quantiles(eng)), p["n_crossings_graded"],
            p["n_crossings_rested_graded"]))
    lines.append("")
    lines.append("Crossing minus reference exit by bucket:\n")
    lines.append("| match | buckets |")
    lines.append("|---|---|")
    for game, text in buckets:
        lines.append("| {} | {} |".format(game, text))
    lines.append("")
    return "\n".join(lines)


def _event_team_agreement(payload, recs):
    """'agree/n' of the restart event team against the reference team (Metrica letters only)."""
    letters = [r for r in recs if r["kind_team"] in TEAMS]
    if not letters:
        return "n/a (team ids)" if recs else "0/0"
    return _pct(sum(1 for r in letters if r["kind_team"] == r["team"]), len(letters))


def _fmt_q(q):
    """'p50 [p10, p90]' of a _quantiles() dict, or '-'."""
    if not q.get("n"):
        return "-"
    return "{:+.0f} [{:+.0f}, {:+.0f}] (n {})".format(q["p50"], q["p10"], q["p90"], q["n"])


def report_exposure(p):
    """Markdown tables of measurements 3 and 4 for the `exposure` payload."""
    lines = ["### Measurement 3: BAR-ATH restarts without labels\n"]
    recs = p["records"]
    graded = [r for r in recs]
    kick = [r for r in graded if r["kickoff"]]
    other = [r for r in graded if not r["kickoff"]]
    no_team = [r for r in other if r["rules"]["k"] not in TEAMS]
    with_team = [r for r in other if r["rules"]["k"] in TEAMS]
    carried = [r for r in no_team if r["written"] == r["rules"]["e"]]
    lines.append("| quantity | count |")
    lines.append("|---|---|")
    lines.append("| engine restarts (D runs ended by H or A) | {} |".format(len(graded)))
    lines.append("| of which kickoffs (games.py or after a GOALS window) | {} |".format(len(kick)))
    lines.append("| other restarts | {} |".format(len(other)))
    lines.append("| with a restart team (rule k) | {} |".format(len(with_team)))
    lines.append("| without a restart team | {} |".format(len(no_team)))
    lines.append("| without a team, written label = owner before the stoppage | {} |".format(
        len(carried)))
    lines.append("")
    lines.append("Rules on the restarts without a team (applies; differs from the written "
                 "label at the exit; frames of the written run from the exit where it "
                 "differs):\n")
    lines.append("| rule | applies | differs | frames of the run that would change |")
    lines.append("|---|---|---|---|")
    for rule in ("a", "b", "c", "c2", "d", "f", "e", "combined", "laws (a..d)"):
        if rule == "laws (a..d)":
            teams = [next((r["rules"][x] for x in LAW_ORDER if r["rules"][x] in TEAMS), None)
                     for r in no_team]
        else:
            teams = [r["rules"][rule] for r in no_team]
        app = [(r, t) for r, t in zip(no_team, teams) if t in TEAMS]
        diff = [(r, t) for r, t in app if t != r["written"]]
        lines.append("| {} | {} | {} | {} |".format(rule, len(app), len(diff),
                                                   sum(r["run_frames"] for r, _ in diff)))
    lines.append("")
    lines.append("Rules on the restarts with a team (rule k): agreement with k.\n")
    lines.append("| rule | applies | agrees with k |")
    lines.append("|---|---|---|")
    for rule in ("a", "b", "c", "c2", "d", "f", "combined"):
        app = [r for r in with_team if r["rules"][rule] in TEAMS]
        agree = sum(1 for r in app if r["rules"][rule] == r["rules"]["k"])
        lines.append("| {} | {} | {} |".format(rule, len(app), _pct(agree, len(app))))
    lines.append("")
    lines.append("Out geometry of the restarts without a team: {}.\n".format(
        _geo_counts([r["geo"] for r in no_team])))
    lines.append("The named team-flip stretches (nearest engine exit; label at the named frame "
                 "in c13b; each rule):\n")
    lines.append("| source | frame | label now | exit | frame - exit | kind | k | written | "
                 "line | last contact | a | b | c | c2 | d | f | e | combined |")
    lines.append("|" + "---|" * 18)
    for item in p["named"]:
        r = item["record"]
        ru = r["rules"]
        g = r["geo"]
        lines.append("| {} | {} | {} | {} | {:+d} | {} | {} | {} | {} {} | {} | {} |".format(
            item["source"], item["frame"], item["label_now"], item["exit"], item["offset"],
            r["kind"], ru["k"] or "-", r["written"], g["line"] or "inside", g["how"],
            g["toucher"] or "-",
            " | ".join(ru[x] or "-" for x in ("a", "b", "c", "c2", "d", "f", "e", "combined"))))
    lines.append("")
    lines.append("### Measurement 4: BAR-ATH throw-in tags\n")
    t = p["throw_tags"]
    ex = [x["exit_minus_start"] for x in t if x["exit_minus_start"] is not None]
    lines.append("| quantity | value |")
    lines.append("|---|---|")
    lines.append("| throw-in tags | {} |".format(len(t)))
    lines.append("| with an engine exit within the window +- {} frames | {} |".format(
        TAG_MARGIN, len(ex)))
    lines.append("| of those, exit inside the window | {} |".format(
        sum(1 for x in t if x["exit_in_window"])))
    lines.append("| engine exit minus tag start, p50 [p10, p90] | {} |".format(
        _fmt_q(_quantiles(ex))))
    ends = [x["exit_minus_start"] - (x["tag_end"] - x["tag_start"]) for x in t
            if x["exit_minus_start"] is not None]
    lines.append("| engine exit minus tag end, p50 [p10, p90] | {} |".format(
        _fmt_q(_quantiles(ends))))
    lines.append("| tags with an inward crossing near the window | {} |".format(
        sum(1 for x in t if x["n_crossings"])))
    lines.append("| tags with a rested inward crossing near the window | {} |".format(
        sum(1 for x in t if x["n_rested"])))
    lines.append("| crossing minus tag start, p50 [p10, p90] | {} |".format(
        _fmt_q(_quantiles([x["cross_minus_start"] for x in t
                           if x["cross_minus_start"] is not None]))))
    lines.append("| rested crossing minus tag start, p50 [p10, p90] | {} |".format(
        _fmt_q(_quantiles([x["rest_cross_minus_start"] for x in t
                           if x["rest_cross_minus_start"] is not None]))))
    lines.append("| rested crossing minus engine exit, p50 [p10, p90] | {} |".format(
        _fmt_q(_quantiles([x["rest_cross_minus_exit"] for x in t
                           if x["rest_cross_minus_exit"] is not None]))))
    lines.append("| any crossing minus engine exit, p50 [p10, p90] | {} |".format(
        _fmt_q(_quantiles([x["cross_minus_exit"] for x in t
                           if x["cross_minus_exit"] is not None]))))
    for key, name in (("cross_minus_exit", "any crossing"), ("rest_cross_minus_exit",
                                                             "rested crossing")):
        values = [x[key] for x in t if x[key] is not None]
        lines.append("| {} minus engine exit by bucket | {} |".format(name, _buckets(values)))
    lines.append("| inward crossings on graded frames (rested) | {} ({}) |".format(
        p["n_crossings_graded"], p["n_rested_graded"]))
    lines.append("| of those, away from every throw-in tag window (rested) | {} ({}) |".format(
        p["n_crossings_away_from_tags"], p["n_rested_away_from_tags"]))
    lines.append("")
    return "\n".join(lines)


def _buckets(values, edges=(-50, -10, 10, 50)):
    """Counts of signed frame values in the buckets cut at `edges` (for example -10..+10)."""
    values = list(values)
    names, lo = [], None
    for edge in tuple(edges) + (None,):
        if lo is None:
            n = sum(1 for v in values if v < edge)
            names.append("< {:+d}: {}".format(edge, n))
        elif edge is None:
            n = sum(1 for v in values if v >= lo)
            names.append(">= {:+d}: {}".format(lo, n))
        else:
            n = sum(1 for v in values if lo <= v < edge)
            names.append("{:+d}..{:+d}: {}".format(lo, edge, n))
        lo = edge
    return ", ".join(names)


def _geo_counts(geos):
    """'line how: count' pairs of a list of geometry dicts, sorted."""
    counts = {}
    for g in geos:
        key = "{} {}".format(g["line"] or "inside", g["how"])
        counts[key] = counts.get(key, 0) + 1
    return ", ".join("{} {}".format(k, counts[k]) for k in sorted(counts))


def report(args):
    """The `report` command: print the markdown tables."""
    payloads = []
    for path in args.inputs:
        with open(path) as handle:
            payloads.append(json.load(handle))
    measured = [p for p in payloads if p["mode"] == "measure"]
    for p in payloads:
        print("{} engine output equal to --engine-csv: {}".format(p["game"],
                                                                  p["engine_equal_csv"]))
    print("")
    if measured:
        print(report_measure(measured))
    for p in payloads:
        if p["mode"] == "exposure":
            print(report_exposure(p))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")
    m = sub.add_parser("measure", help="measurements 1, 2 and 4 on an external reference")
    m.add_argument("--game", required=True)
    m.add_argument("--ref", required=True, help="the reference `frame,label` CSV")
    m.add_argument("--metrica-events", default=None, help="Metrica events JSON or CSV")
    m.add_argument("--dfl-events", default=None, help="DFL events_raw XML")
    m.add_argument("--dfl-positions", default=None, help="DFL positions_raw_observed XML")
    m.add_argument("--engine-csv", default=None, help="a stored engine output to compare")
    m.add_argument("--out", required=True)
    e = sub.add_parser("exposure", help="measurements 3 and 4 on the competition match")
    e.add_argument("--game", default="BAR-ATH")
    e.add_argument("--engine-csv", default=None)
    e.add_argument("--out", required=True)
    r = sub.add_parser("report", help="print the markdown tables")
    r.add_argument("inputs", nargs="+")
    args = parser.parse_args(argv)
    if args.command == "measure":
        if args.dfl_events and not args.dfl_positions:
            parser.error("--dfl-events needs --dfl-positions")
        measure(args)
    elif args.command == "exposure":
        exposure(args)
    elif args.command == "report":
        return report(args)
    else:
        parser.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
