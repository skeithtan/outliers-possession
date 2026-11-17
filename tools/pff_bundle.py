"""PFF FC World Cup 2022 match to an ATD bundle and a reference CSV. Not part of the submission.

    python tools/pff_bundle.py build --raw-dir RAW --game-id ID --out-dir DIR [--rel-dir REL]
    python tools/pff_bundle.py entries DIR/PFF-<id>_build.json ...   # games.py entries again

The tool reads one match of the PFF FC FIFA World Cup 2022 data (64 matches, broadcast tracking
and event data), from https://drive.google.com/drive/folders/1_a_q1e9CXeEPJ3GdCv_3-rNO3gPqacfa
(tools/pff_download.py fetches it). PFF FC gives the data free after a form. We found no
licence text. The team approved its use on 9 November 2026 on one condition: credit PFF FC in
the technical document. docs/plan/calibration/PFF_REFERENCES.md gives the source, the download
and the decisions. The tool writes four files into DIR, the same files as
tools/skillcorner_bundle.py:

    PFF-<id>_atd.xml        FIFA/EPTS-style metadata that engine/data.py reads unchanged
    PFF-<id>_atd.txt        raw tracking lines "<frame_id>:<x,y>;...;<x,y>:<bx,by>"
    PFF-<id>_reference.csv  `frame,label` with H, A, D and X, one row for each video frame
    PFF-<id>_build.json     the games.py entry fields, checks, statistics and input checksums

Keep DIR outside the repository. Our machines use ~/mtah2027/refs/pff/<id>/.

Input files. RAW holds four folders. The tool reads one file from each:

- `tracking/<id>.jsonl.bz2`: one JSON line for each tracking frame. A line has `frameNum`,
  `videoTimeMs`, `period`, `homePlayers` and `awayPlayers` (lists of {jerseyNum, visibility,
  x, y}), `balls` (a list of {visibility, x, y, z}) and the smoothed copies of these lists. The
  tool streams the file line by line. It never writes the decompressed file to disk.
- `events/<id>.json`: a list of event rows. A row has `gameEventId`, `eventTime`, `startTime`,
  `endTime`, `sequence`, `gameEvents` {gameEventType, period, teamId, playerId, setpieceType,
  outType} and `possessionEvents` {possessionEventType}.
- `metadata/<id>.json`: the teams, the pitch, `fps` and `homeTeamStartLeft`.
- `rosters/<id>.json`: the players of the match with `shirtNumber`, team and position group.

Frames. The tracking runs at 29.97 frames per second (30000 / 1001). frameNum f is at time
f * 1001 / 30000 s from the start of the video. The event times are seconds from the start of
the same video. The tool writes 25 Hz, the rate of BAR-ATH. Video frame v is at time v / 25 s.
So v lies at the tracking position s = v * 1200 / 1001. The ATD `frame_id` of a raw line is
v + 1, as in engine/data.py. The games.py `offsets` are 1 for both halves.

Halves. The tool uses periods 1 and 2 only. engine/data.py knows two halves, so the extra time
(periods 3 and 4) and the penalty shoot-out are not in the bundle. The block of a half is the
set of video frames inside the tracking frames of that period. It is cut to the event block:
from the kickoff event (FIRSTKICKOFF or SECONDKICKOFF) to the frame before the END event.

Positions. The tool writes the positions only inside the blocks. Every other frame is NaN. A
video frame on a tracking frame takes that frame. Any other video frame lies between tracking
frames i and i + 1. It takes (1 - w) * p(i) + w * p(i + 1), with w = s - i. If either value is
missing, the video frame is missing. So the tool never fills a gap and never extends a run.

Raw positions, not the smoothed copies. PFF smooths the players with a Kalman filter. PFF puts
the smoothed ball on the ball carrier of the events, so the smoothed ball holds the event data.
The tool reads the raw lists `homePlayers`, `awayPlayers` and `balls` only.

Ball. The tool keeps the raw ball samples with visibility VISIBLE. It treats every ESTIMATED
sample as missing. PFF estimates the ball from context, mostly while it is in the air or off
the screen. The BAR-ATH ball is missing on about 41 % of the graded frames. The height z is not
used.

Players. The tracking names a player by team and jersey number. The roster gives the player id
of each jersey number. A jersey number that is not in the roster of its team is dropped. Each
player with at least one position inside a half gets one slot. The home team comes first, and
each team is sorted by player id. The tool keeps the ESTIMATED positions of the players outside
the camera view, as tools/skillcorner_bundle.py does, so that the engine sees 22 players. If a
team lists one jersey number twice on one frame, the tool keeps the first entry.

Channels and coordinates. Players and ball carry x and y only. The origin is the centre spot, x
runs along the touchlines and y points up. The tool writes `x = (X + length / 2) / length` and
`y = 0.5 - Y / width` with the pitch of the metadata file.

Teams. `H` is `homeTeam` and `A` is `awayTeam` of the metadata file. The home team is the first
`<Team>` of the ATD XML, because engine.data._atd_meta maps the first `<Team>` to `H`.

Labels. The labels come from the event rows of periods 1 and 2, in file order (the file is in
time order). An event row takes effect on video frame round(25 * eventTime). The rules:

- Graded block: the half block above. Every frame outside the blocks is `X`.
- Dead ball: an OUT game event (ball out of play, goal or whistle) starts `D`. `D` lasts until
  the restart. The restart is the next on-the-ball row (OTB or a KICKOFF game event). The END
  game event ends the block.
- Restart without an OUT: a set piece (setpieceType other than O) while the ball is alive starts
  `D` at the end of the game event before it. The build file counts these rows.
- Owner while alive: PFF gives each row a `sequence`, "an uninterrupted possession by one team".
  The team of a sequence is the team of its first row that names a team. The owner changes at
  the first row of a new sequence that names a team. Rebounds, clearances and challenges of the
  other team stay inside the sequence, so they do not change the owner.

Kickoffs. The block of each half starts at the kickoff event. So the games.py `kickoff` and
`kickoff_2nd` anchors are the first video frame of each block.

Checks in the build file. `actor_ball` gives the distance from the ball to the player of each
event row at the event frame, with a scan of frame shifts. `side_check` compares the mean x of
the home goalkeeper in each half with `homeTeamStartLeft`. `direction_check` compares the
`teamAttackingDirection` of the event rows with the same sides. `event_frames` compares the
event times with the frameNum that the tracking file gives to its own copy of each event.

Deterministic: sorted iteration, fixed number formats, no clock, no network. Python 3.9, numpy
and the standard library.
"""

import argparse
import bz2
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import games  # noqa: E402
from tools import events_gt  # noqa: E402
from tools import skillcorner_bundle as sc  # noqa: E402

H, A, D, X = events_gt.H, events_gt.A, events_gt.D, events_gt.X

SOURCE = "drive.google.com/drive/folders/1_a_q1e9CXeEPJ3GdCv_3-rNO3gPqacfa"
PROVIDER = "PFF FC World Cup 2022 (credit PFF FC), converted by tools/pff_bundle.py"
STEM_PREFIX = "PFF-"
# The tracking rate is 29.97 = 30000 / 1001 frames per second. The bundle rate is 25.
SOURCE_NUM, SOURCE_DEN = 30000, 1001
VIDEO_FPS = 25
# Video frame v lies at tracking position v * NUM / DEN = v * 30000 / 25025 = v * 1200 / 1001.
NUM, DEN = SOURCE_NUM, SOURCE_DEN * VIDEO_FPS
HALVES = (1, 2)
VISIBLE = "VISIBLE"
KEEPER_GROUP = "GK"
# Game event types that put a player on the ball and can restart play.
KICKOFF_TYPES = ("FIRSTKICKOFF", "SECONDKICKOFF")
KICKOFF_OF_HALF = {1: "FIRSTKICKOFF", 2: "SECONDKICKOFF"}
ON_BALL_TYPES = ("OTB",) + KICKOFF_TYPES
STOP_TYPE = "OUT"
END_TYPE = "END"
# setpieceType of open play. Every other value (K, C, F, G, T, P, D) is a set piece.
OPEN_PLAY = ("O", None)
# Possession event types where the player of the row plays the ball (actor_ball check).
ACTOR_TYPES = ("PA", "CR", "SH", "CL", "IT", "BC", "TC", "RE")
# The actor_ball check scans frame shifts of the bundle against the event frames, +/- this many.
LAG_SCAN = 15
# actor_ball reports a possession type on its own when it has at least this many rows.
BY_TYPE_MIN = 50
# Distances for the actor_ball shares, in metres.
NEAR_M = (1.0, 2.0, 5.0)
# The position fields of a tracking line. A repeated frame must repeat them.
REPEAT_FIELDS = ("homePlayers", "awayPlayers", "balls")
# The top-level period field of a raw tracking line. The files use compact JSON ("period":1).
PERIOD_FIELD = re.compile(r'"period":\s*(null|\d+)')
# Channel ids of the ATD files that engine/data.py reads.
CHANNELS = sc.CHANNELS
ANCHOR_FPS = games.ANCHOR_FPS


# -- reading the files ---------------------------------------------------------------------------

def file_names(game_id):
    """{name: path relative to RAW} of the four input files of a match."""
    return {"tracking": os.path.join("tracking", "{}.jsonl.bz2".format(game_id)),
            "events": os.path.join("events", "{}.json".format(game_id)),
            "metadata": os.path.join("metadata", "{}.json".format(game_id)),
            "roster": os.path.join("rosters", "{}.json".format(game_id))}


def read_metadata(path):
    """The fields of a PFF metadata file that the bundle needs, as a dict.

    Keys: game_id, title, date, home and away {team_id, name}, pitch (length, width) in metres,
    fps, home_start_left (bool) and periods {half: (start s, end s)} of the video.
    """
    with open(path) as handle:
        raw = json.load(handle)
    if isinstance(raw, list):
        if len(raw) != 1:
            raise ValueError("{}: {} metadata records".format(path, len(raw)))
        raw = raw[0]
    home = {"team_id": int(raw["homeTeam"]["id"]), "name": raw["homeTeam"]["name"]}
    away = {"team_id": int(raw["awayTeam"]["id"]), "name": raw["awayTeam"]["name"]}
    pitches = raw["stadium"]["pitches"]
    if len(pitches) != 1:
        raise ValueError("{}: {} pitches".format(path, len(pitches)))
    periods = {}
    for half in HALVES:
        start, end = raw.get("startPeriod{}".format(half)), raw.get("endPeriod{}".format(half))
        if start is not None and end is not None:
            periods[half] = (float(start), float(end))
    return {"game_id": str(raw["id"]), "title": "{} v {}".format(home["name"], away["name"]),
            "date": str(raw.get("date", "")), "home": home, "away": away,
            "pitch": (float(pitches[0]["length"]), float(pitches[0]["width"])),
            "fps": float(raw["fps"]), "home_start_left": bool(raw["homeTeamStartLeft"]),
            "periods": periods}


def read_roster(path, home_id, away_id):
    """The roster of a match as a dict.

    Keys: player_of {(team id, jersey): player id}, team_of_player {player id: team id},
    jersey_of_player {player id: jersey}, keepers [player ids of the GK position group].
    Raises ValueError for a team other than home and away, or a jersey used twice in a team.
    """
    with open(path) as handle:
        raw = json.load(handle)
    player_of, team_of_player, jersey_of_player, keepers = {}, {}, {}, []
    for item in raw:
        team = int(item["team"]["id"])
        if team not in (home_id, away_id):
            raise ValueError("{}: team {}".format(path, team))
        pid, jersey = int(item["player"]["id"]), int(item["shirtNumber"])
        if (team, jersey) in player_of:
            raise ValueError("{}: jersey {} twice in team {}".format(path, jersey, team))
        player_of[(team, jersey)] = pid
        team_of_player[pid] = team
        jersey_of_player[pid] = jersey
        if item.get("positionGroupType") == KEEPER_GROUP:
            keepers.append(pid)
    return {"player_of": player_of, "team_of_player": team_of_player,
            "jersey_of_player": jersey_of_player, "keepers": sorted(keepers)}


def _number(value):
    """A float, or None for a missing or non-finite value."""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def read_events(path, home_id, away_id):
    """The event rows of an events file, in file order. See read_events_from_list."""
    with open(path) as handle:
        raw = json.load(handle)
    return read_events_from_list(raw, home_id, away_id, path)


def read_events_from_list(raw, home_id, away_id, path="events"):
    """The event rows of a parsed events file as a list of dicts, in file order.

    Keys: index (file order), game_event_id, type, period, team (H, A or None), player_id,
    setpiece, out_type, possession_type, sequence, time (eventTime, else startTime), start,
    end (seconds of the video), direction (teamAttackingDirection), non_event (bool).
    """
    code = {home_id: H, away_id: A}
    rows = []
    for index, item in enumerate(raw):
        game = item.get("gameEvents") or {}
        poss = item.get("possessionEvents") or {}
        team = game.get("teamId")
        team = None if team is None else code.get(int(team))
        if game.get("teamId") is not None and team is None:
            raise ValueError("{}: row {} has team {}".format(path, index, game.get("teamId")))
        player = game.get("playerId")
        time = _number(item.get("eventTime"))
        start = _number(item.get("startTime"))
        period = game.get("period")
        rows.append({
            "index": index, "game_event_id": item.get("gameEventId"),
            "type": game.get("gameEventType"),
            "period": None if period is None else int(period),
            "team": team, "player_id": None if player is None else int(player),
            "setpiece": game.get("setpieceType"), "out_type": game.get("outType"),
            "possession_type": poss.get("possessionEventType"),
            "sequence": None if item.get("sequence") is None else int(item["sequence"]),
            "time": time if time is not None else start, "start": start,
            "end": _number(item.get("endTime")),
            "direction": (item.get("stadiumMetadata") or {}).get("teamAttackingDirection"),
            "non_event": bool(game.get("initialNonEvent")) or bool(poss.get("nonEvent")),
        })
    return rows


def _period_of_line(line):
    """The period of a raw tracking line without a JSON parse, or None if it is null."""
    match = PERIOD_FIELD.search(line)
    if match is None:
        raise ValueError("a tracking line has no period field")
    return None if match.group(1) == "null" else int(match.group(1))


def read_tracking(path, player_of, halves=HALVES):
    """The raw tracking of periods `halves` as arrays, streamed from the bz2 file line by line.

    Returns a dict: first (frameNum of row 0), period int8 (F,) with 0 for no frame, ball_x,
    ball_y float64 (F,) of the VISIBLE ball only, ball_seen bool (F,) where the file has any
    ball, players {player id: (x, y, visible)} float64, float64, bool (F,), unknown {(team side,
    jersey): frames} for jersey numbers not in the roster, duplicates (count of repeated jersey
    entries), multi_ball (frames with more than one ball), event_frames [(frameNum, game event
    start_frame, start_time)] of the event copies in the file, lines (all lines read),
    repeated_frames (lines that repeat the frameNum before them: the file repeats a frame for
    each further possession event on it; the tool keeps the first line) and
    repeated_frames_differ (repeats whose positions differ from the first line).
    Rows are frameNum - first. A frameNum with no line stays empty (period 0).
    """
    team_code = {"homePlayers": "home", "awayPlayers": "away"}
    capacity = 1 << 18
    first = None
    period = np.zeros(capacity, dtype=np.int8)
    ball = np.full((capacity, 2), np.nan)
    ball_seen = np.zeros(capacity, dtype=bool)
    tracks = {}
    unknown, event_frames = {}, []
    duplicates = multi_ball = lines = repeated = repeated_differ = 0
    last, previous = -1, None

    def grow(size):
        nonlocal capacity, period, ball, ball_seen
        while size > capacity:
            capacity *= 2
        period = np.concatenate([period, np.zeros(capacity - period.size, dtype=np.int8)])
        ball = np.concatenate([ball, np.full((capacity - ball.shape[0], 2), np.nan)])
        ball_seen = np.concatenate([ball_seen, np.zeros(capacity - ball_seen.size, bool)])
        for key, (values, seen) in tracks.items():
            tracks[key] = (np.concatenate([values, np.full((capacity - values.shape[0], 2),
                                                           np.nan)]),
                           np.concatenate([seen, np.zeros(capacity - seen.size, bool)]))

    with bz2.open(path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            lines += 1
            half = _period_of_line(line)
            if half not in halves:
                continue
            row = json.loads(line)
            frame = int(row["frameNum"])
            if first is None:
                first = frame
            i = frame - first
            event = row.get("game_event")
            if event and event.get("start_frame") is not None:
                event_frames.append((frame, int(event["start_frame"]),
                                     _number(event.get("start_time"))))
            content = [row.get(name) for name in REPEAT_FIELDS]
            if i == last:
                # The file repeats a frame for each further possession event on it.
                repeated += 1
                repeated_differ += content != previous
                continue
            if i < last:
                raise ValueError("{}: frameNum {} goes back".format(path, frame))
            last, previous = i, content
            if i >= capacity:
                grow(i + 1)
            period[i] = half
            balls = row.get("balls") or []
            if len(balls) > 1:
                multi_ball += 1
            if balls and balls[0].get("x") is not None and balls[0].get("y") is not None:
                ball_seen[i] = True
                if balls[0].get("visibility") == VISIBLE:
                    ball[i] = (balls[0]["x"], balls[0]["y"])
            for field, side in sorted(team_code.items()):
                seen_here = set()
                for item in row.get(field) or ():
                    if item.get("x") is None or item.get("y") is None:
                        continue
                    try:
                        jersey = int(item["jerseyNum"])
                    except (KeyError, TypeError, ValueError):
                        jersey = -1        # no readable jersey number: an unknown player
                    if jersey in seen_here:
                        duplicates += 1
                        continue
                    seen_here.add(jersey)
                    pid = player_of.get((side, jersey))
                    if pid is None:
                        unknown[(side, jersey)] = unknown.get((side, jersey), 0) + 1
                        continue
                    if pid not in tracks:
                        tracks[pid] = (np.full((capacity, 2), np.nan),
                                       np.zeros(capacity, dtype=bool))
                    values, seen = tracks[pid]
                    values[i] = (item["x"], item["y"])
                    seen[i] = item.get("visibility") == VISIBLE
    if first is None:
        raise ValueError("{}: no tracking frame in periods {}".format(path, halves))
    n = last + 1
    return {"first": int(first), "period": period[:n].copy(), "ball_x": ball[:n, 0].copy(),
            "ball_y": ball[:n, 1].copy(), "ball_seen": ball_seen[:n].copy(),
            "players": {pid: (v[:n, 0].copy(), v[:n, 1].copy(), s[:n].copy())
                        for pid, (v, s) in sorted(tracks.items())},
            "unknown": unknown, "duplicates": duplicates, "multi_ball": multi_ball,
            "event_frames": event_frames, "lines": lines, "repeated_frames": repeated,
            "repeated_frames_differ": repeated_differ}


# -- frame mapping -------------------------------------------------------------------------------

def half_spans(period, first):
    """{half: (first, last)} frameNum of each period value 1 and 2. ValueError if one is absent."""
    spans = {}
    for half in HALVES:
        rows = np.flatnonzero(period == half)
        if rows.size == 0:
            raise ValueError("no tracking frame of period {}".format(half))
        spans[half] = (int(rows[0]) + first, int(rows[-1]) + first)
    if spans[2][0] <= spans[1][1]:
        raise ValueError("the periods overlap")
    return spans


def video_block(first, last):
    """(first, last) video frames inside the frameNum span first..last (ceil and floor)."""
    return (-((-first * DEN) // NUM), (last * DEN) // NUM)


def video_frame_of(seconds):
    """The video frame of a time in seconds, round(25 * t), as int."""
    return int(np.floor(seconds * VIDEO_FPS + 0.5))


def resample(values, n_video, first):
    """A tracking array (F,) with row 0 at frameNum `first`, at 25 Hz (n_video rows).

    A video frame on a tracking frame takes its value. A video frame between two tracking
    frames takes their linear interpolation, or NaN if either is NaN or absent.
    """
    values = np.asarray(values, dtype=np.float64)
    v = np.arange(n_video, dtype=np.int64)
    low, rem = np.divmod(v * NUM, DEN)
    low = low - first
    out = np.full(n_video, np.nan)
    n = values.shape[0]
    exact = (low >= 0) & (low < n) & (rem == 0)
    out[exact] = values[low[exact]]
    between = (low >= 0) & (rem > 0) & (low + 1 < n)
    w = rem[between] / float(DEN)
    out[between] = (1.0 - w) * values[low[between]] + w * values[low[between] + 1]
    return out


# -- the reference -------------------------------------------------------------------------------

def sequence_teams(rows):
    """{sequence: team} from the first row of each sequence that names a team."""
    out = {}
    for row in rows:
        if row["sequence"] is not None and row["team"] is not None:
            out.setdefault(row["sequence"], row["team"])
    return out


def change_points(rows):
    """The label changes of the event rows of one half, as (list, counts).

    The list holds (time s, code) in event order: code H or A from the time on, D for a dead
    ball, X for the end of the half. counts holds restarts_without_out, teamless_rows_skipped,
    owner_changes, stops and restarts. The module docstring, Labels, gives the rules.
    """
    teams = sequence_teams(rows)
    points = []
    counts = {"restarts_without_out": 0, "teamless_rows_skipped": 0, "owner_changes": 0,
              "stops": 0, "restarts": 0}
    alive, owner, current, previous_end = False, None, None, None
    for row in rows:
        kind, time = row["type"], row["time"]
        if time is None:
            continue
        if kind == STOP_TYPE:
            if alive:
                points.append((time, D))
                counts["stops"] += 1
            alive = False
            continue
        if kind == END_TYPE:
            points.append((time, X))
            alive, current = False, None
            continue
        if kind not in ON_BALL_TYPES:
            continue
        team = teams.get(row["sequence"], row["team"])
        ends = [t for t in (row["end"], time) if t is not None]
        if not alive:
            if team is None:
                counts["teamless_rows_skipped"] += 1
                continue
            points.append((time, team))
            counts["restarts"] += 1
            alive, owner, current = True, team, row["sequence"]
        elif row["team"] is None:
            counts["teamless_rows_skipped"] += 1
        elif (row["setpiece"] not in OPEN_PLAY and kind == "OTB"
              and row["sequence"] != current and previous_end is not None):
            points.append((min(previous_end, time), D))
            points.append((time, team))
            counts["restarts_without_out"] += 1
            owner, current = team, row["sequence"]
        elif row["sequence"] is not None and row["sequence"] != current:
            current = row["sequence"]
            if team != owner:
                points.append((time, team))
                counts["owner_changes"] += 1
                owner = team
        previous_end = max(ends)
    return points, counts


def event_blocks(rows):
    """{half: (kickoff time, END time)} in seconds from the event rows. ValueError if absent."""
    out = {}
    for half in HALVES:
        kick = [r["time"] for r in rows if r["type"] == KICKOFF_OF_HALF[half]
                and r["period"] == half]
        end = [r["time"] for r in rows if r["type"] == END_TYPE and r["period"] == half]
        if len(kick) != 1 or len(end) != 1:
            raise ValueError("half {}: {} kickoff and {} END events".format(
                half, len(kick), len(end)))
        out[half] = (kick[0], end[0])
    return out


def reference_labels(rows, blocks, n_frames):
    """The 25 Hz labels (int8, n_frames) and the change-point counts of each half."""
    labels = np.full(n_frames, X, dtype=np.int8)
    counts = {}
    for half, (first, last) in sorted(blocks.items()):
        labels[first:last + 1] = D
        points, counts[half] = change_points([r for r in rows if r["period"] == half])
        frames = [(video_frame_of(t), code) for t, code in points]
        for k, (frame, code) in enumerate(frames):
            stop = frames[k + 1][0] if k + 1 < len(frames) else last + 1
            a, b = max(frame, first), min(stop, last + 1)
            if b > a:
                labels[a:b] = D if code == X else code
    return labels, counts


# -- the bundle ----------------------------------------------------------------------------------

def build(meta, roster, rows, tracking):
    """The bundle arrays, the games.py fields and the checks of one match, as a dict.

    Keys: match_id, stem, title, pitch, spans {half: frameNum (first, last)}, blocks {half: video
    (first, last)}, n_frames, teams [home, away], slots [(team id, player id)], px, py
    (n_frames, n_slots) and bx, by (n_frames,) normalised, labels int8 (n_frames,), and stats.
    """
    home, away = meta["home"], meta["away"]
    length, width = meta["pitch"]
    first = tracking["first"]
    spans = half_spans(tracking["period"], first)
    times = event_blocks(rows)
    blocks = {}
    for half, (a, b) in sorted(spans.items()):
        lo, hi = video_block(a, b)
        kick, end = times[half]
        blocks[half] = (max(lo, video_frame_of(kick)), min(hi, video_frame_of(end) - 1))
        if blocks[half][1] <= blocks[half][0]:
            raise ValueError("half {} has an empty block".format(half))
    n_frames = blocks[max(blocks)][1] + 1
    in_block = np.zeros(n_frames, dtype=bool)
    for a, b in blocks.values():
        in_block[a:b + 1] = True

    def to_video(values, norm):
        out = norm(resample(values, n_frames, first))
        out[~in_block] = np.nan
        return out

    def norm_x(values):
        return (values + length / 2.0) / length

    def norm_y(values):
        return 0.5 - values / width

    people = {}
    for pid, (x, _, _) in sorted(tracking["players"].items()):
        if np.isfinite(x).any():
            people.setdefault(roster["team_of_player"][pid], []).append(pid)
    slots = [(team["team_id"], pid) for team in (home, away)
             for pid in sorted(people.get(team["team_id"], []))]
    px = np.full((n_frames, len(slots)), np.nan)
    py = np.full((n_frames, len(slots)), np.nan)
    for column, (_, pid) in enumerate(slots):
        x, y, _ = tracking["players"][pid]
        px[:, column] = to_video(x, norm_x)
        py[:, column] = to_video(y, norm_y)
    bx = to_video(tracking["ball_x"], norm_x)
    by = to_video(tracking["ball_y"], norm_y)

    labels, counts = reference_labels(rows, blocks, n_frames)
    bundle = {"match_id": meta["game_id"], "stem": STEM_PREFIX + meta["game_id"],
              "title": meta["title"], "pitch": (length, width), "spans": spans,
              "blocks": blocks, "n_frames": int(n_frames), "teams": [home, away],
              "slots": slots, "px": px, "py": py, "bx": bx, "by": by, "labels": labels}
    bundle["stats"] = statistics(bundle, meta, roster, rows, tracking, counts)
    return bundle


def _actor_distances(bundle, frames, cols, shift):
    """Ball-to-player distances in metres at frames + shift, where both are present."""
    length, width = bundle["pitch"]
    f = frames + shift
    ok = (f >= 0) & (f < bundle["n_frames"])
    f, c = f[ok], cols[ok]
    dx = (bundle["bx"][f] - bundle["px"][f, c]) * length
    dy = (bundle["by"][f] - bundle["py"][f, c]) * width
    dist = np.hypot(dx, dy)
    return dist[np.isfinite(dist)]


def _actor_scan(bundle, frames, cols):
    """(distances at shift 0, {shift: median or None}, best shift or None)."""
    scan = {}
    for shift in range(-LAG_SCAN, LAG_SCAN + 1):
        dist = _actor_distances(bundle, frames, cols, shift)
        scan[shift] = round(float(np.median(dist)), 4) if dist.size else None
    valid = {s: m for s, m in scan.items() if m is not None}
    best = min(valid, key=lambda s: (valid[s], abs(s))) if valid else None
    return _actor_distances(bundle, frames, cols, 0), scan, best


def actor_ball(bundle, rows, roster):
    """The distance from the ball to the player of each event row, as a dict.

    The rows are the on-the-ball rows of periods 1 and 2 with a player and a possession type in
    ACTOR_TYPES. At the event frame plus a shift, the distance in metres between the ball and
    the slot of the player, where both are present. Keys: rows (with a distance at shift 0),
    median_m, share_within_<d>m, scan {shift: median}, best_shift, and by_type {type: {rows,
    median_m, best_shift, best_median_m}} for the types with at least BY_TYPE_MIN rows.
    """
    column = {pid: k for k, (_, pid) in enumerate(bundle["slots"])}
    frames, cols, types = [], [], []
    for row in rows:
        if (row["type"] in ON_BALL_TYPES and row["period"] in HALVES and row["time"] is not None
                and row["possession_type"] in ACTOR_TYPES and row["player_id"] in column):
            frames.append(video_frame_of(row["time"]))
            cols.append(column[row["player_id"]])
            types.append(row["possession_type"])
    frames, cols = np.array(frames, dtype=np.int64), np.array(cols, dtype=np.int64)
    types = np.array(types, dtype=object)
    at_zero, scan, best = _actor_scan(bundle, frames, cols)
    out = {"rows": int(at_zero.size),
           "median_m": round(float(np.median(at_zero)), 4) if at_zero.size else None,
           "scan": {str(s): m for s, m in sorted(scan.items())}, "best_shift": best,
           "by_type": {}}
    for d in NEAR_M:
        out["share_within_{:g}m".format(d)] = (round(float(np.mean(at_zero <= d)), 4)
                                               if at_zero.size else None)
    for kind in ACTOR_TYPES:
        pick = types == kind
        if int(pick.sum()) < BY_TYPE_MIN:
            continue
        zero, kscan, kbest = _actor_scan(bundle, frames[pick], cols[pick])
        out["by_type"][kind] = {"rows": int(zero.size),
                                "median_m": round(float(np.median(zero)), 4) if zero.size
                                else None,
                                "best_shift": kbest,
                                "best_median_m": None if kbest is None else kscan[kbest]}
    return out


def side_check(bundle, roster, meta):
    """The home goalkeeper side in each half against homeTeamStartLeft, as a dict.

    The goalkeeper is the home GK of the roster with the most frames in the block. With
    homeTeamStartLeft, the home goalkeeper stands at x < 0.5 (normalised) in half 1 and
    x > 0.5 in half 2.
    """
    home_id = meta["home"]["team_id"]
    out = {"home_start_left": meta["home_start_left"], "halves": {}}
    agree = True
    for half, (a, b) in sorted(bundle["blocks"].items()):
        best = None
        for k, (team, pid) in enumerate(bundle["slots"]):
            if team != home_id or pid not in roster["keepers"]:
                continue
            x = bundle["px"][a:b + 1, k]
            count = int(np.isfinite(x).sum())
            if count and (best is None or count > best[0]):
                best = (count, float(np.nanmean(x)), pid)
        if best is None:
            out["halves"][str(half)] = None
            agree = False
            continue
        left = best[1] < 0.5
        expected_left = meta["home_start_left"] if half == 1 else not meta["home_start_left"]
        out["halves"][str(half)] = {"keeper": best[2], "mean_x": round(best[1], 4),
                                    "agrees": left == expected_left}
        agree = agree and left == expected_left
    out["agrees"] = agree
    return out


def direction_check(rows, meta):
    """Share of event rows whose teamAttackingDirection agrees with the team sides, per half."""
    out = {}
    for half in HALVES:
        home_right = meta["home_start_left"] if half == 1 else not meta["home_start_left"]
        good = total = 0
        for row in rows:
            if row["period"] != half or row["team"] is None or row["direction"] not in ("L", "R"):
                continue
            expected = "R" if (row["team"] == H) == home_right else "L"
            total += 1
            good += row["direction"] == expected
        out[str(half)] = round(good / float(total), 4) if total else None
    return out


def segment_lengths(labels):
    """Median and short counts of the H/A and D runs of the graded frames, in seconds."""
    out = {}
    runs = []
    start = 0
    for i in range(1, labels.size + 1):
        if i == labels.size or labels[i] != labels[start]:
            runs.append((int(labels[start]), (i - start) / float(VIDEO_FPS)))
            start = i
    for name, codes in (("alive", (H, A)), ("dead", (D,))):
        lengths = sorted(s for c, s in runs if c in codes)
        out[name] = {"runs": len(lengths),
                     "median_s": round(float(np.median(lengths)), 3) if lengths else None,
                     "under_1s": int(sum(s < 1.0 for s in lengths)),
                     "under_2s": int(sum(s < 2.0 for s in lengths))}
    return out


def ball_state_by_label(bundle, tracking):
    """Shares of VISIBLE, ESTIMATED and no ball on the alive (H, A) and the D frames, as a dict.

    Each graded video frame takes the ball state of its nearest tracking frame. A diagnostic
    only: the bundle keeps the VISIBLE ball alone.
    """
    v = np.arange(bundle["n_frames"], dtype=np.int64)
    nearest = (2 * v * NUM + DEN) // (2 * DEN) - tracking["first"]
    ok = (nearest >= 0) & (nearest < tracking["period"].size)
    state = np.full(bundle["n_frames"], 2, dtype=np.int8)       # 2 = no ball
    idx = nearest[ok]
    state[ok] = np.where(np.isfinite(tracking["ball_x"][idx]), 0,
                         np.where(tracking["ball_seen"][idx], 1, 2))
    out = {}
    for name, codes in (("alive", (H, A)), ("dead", (D,))):
        pick = np.isin(bundle["labels"], codes)
        total = int(pick.sum())
        out[name] = {key: (round(float(np.mean(state[pick] == k)), 4) if total else None)
                     for k, key in enumerate(("visible", "estimated", "none"))}
    return out


def statistics(bundle, meta, roster, rows, tracking, counts):
    """The checks and summary statistics of the build file, as a dict."""
    spans, blocks = bundle["spans"], bundle["blocks"]
    period = tracking["period"]
    graded = bundle["labels"] != X
    in_block = np.zeros(bundle["n_frames"], dtype=bool)
    for a, b in blocks.values():
        in_block[a:b + 1] = True
    frames_in_span = {}
    for half, (a, b) in sorted(spans.items()):
        frames_in_span[str(half)] = {"frames": int(np.sum(period == half)),
                                     "span": b - a + 1}
    in_half = period > 0
    tracks = [tracking["players"][pid] for pid in sorted(tracking["players"])]
    present = np.concatenate([np.isfinite(x[in_half]) for x, _, _ in tracks] or
                             [np.zeros(0, dtype=bool)])
    detected = np.concatenate([d[in_half] for _, _, d in tracks] or [np.zeros(0, dtype=bool)])
    # The tracking repeats the game event copy on every frame of the game event: one per event.
    copies = sorted({(start, t) for _, start, t in tracking["event_frames"] if t is not None})
    offsets = [start - round(t * SOURCE_NUM / float(SOURCE_DEN)) for start, t in copies]
    first_line = {}
    for frame, start, _ in tracking["event_frames"]:
        first_line.setdefault(start, frame)
    stats = events_gt.reference_stats(bundle["labels"], VIDEO_FPS)
    stats = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in sorted(stats.items())}
    halves = [r for r in rows if r["period"] in HALVES]
    out_types = {}
    for r in halves:
        if r["type"] == STOP_TYPE:
            out_types[str(r["out_type"])] = out_types.get(str(r["out_type"]), 0) + 1
    graded_seconds = {str(h): round((b - a + 1) / float(VIDEO_FPS), 2)
                      for h, (a, b) in sorted(blocks.items())}
    return {
        "periods_tracking": {str(h): list(v) for h, v in sorted(spans.items())},
        "tracking_frames": frames_in_span,
        "tracking_lines": tracking["lines"],
        "periods_metadata_s": {str(h): list(v) for h, v in sorted(meta["periods"].items())},
        "graded_seconds": graded_seconds,
        "event_frames": {"copies": len(offsets),
                         "start_frame_minus_round_fps_time": {
                             "min": min(offsets) if offsets else None,
                             "max": max(offsets) if offsets else None,
                             "nonzero": int(sum(o != 0 for o in offsets))},
                         "first_line_after_start_frame": int(sum(
                             line > start for start, line in first_line.items()))},
        "actor_ball": actor_ball(bundle, rows, roster),
        "side_check": side_check(bundle, roster, meta),
        "direction_check": direction_check(rows, meta),
        "change_points": {str(h): c for h, c in sorted(counts.items())},
        "out_types": dict(sorted(out_types.items())),
        "sequences": len({r["sequence"] for r in halves if r["sequence"] is not None}),
        "non_event_rows": int(sum(r["non_event"] for r in halves)),
        "unknown_jerseys": {"{}:{}".format(*k): v for k, v in sorted(tracking["unknown"].items())},
        "duplicate_jersey_entries": tracking["duplicates"],
        "repeated_frames": tracking["repeated_frames"],
        "repeated_frames_differ": tracking["repeated_frames_differ"],
        "multi_ball_frames": tracking["multi_ball"],
        "source_ball_visible_share": round(float(np.mean(
            np.isfinite(tracking["ball_x"][in_half]))), 6),
        "source_ball_estimated_share": round(float(np.mean(
            tracking["ball_seen"][in_half] & ~np.isfinite(tracking["ball_x"][in_half]))), 6),
        "source_player_visible_share": round(float(np.mean(detected[present])), 6)
        if present.any() else None,
        "video_ball_visible_share_graded": round(float(np.mean(
            np.isfinite(bundle["bx"][graded]))), 6),
        "ball_state_by_label": ball_state_by_label(bundle, tracking),
        "video_slots_present_median": float(np.median(np.sum(
            np.isfinite(bundle["px"][in_block]), axis=1))),
        "segments": segment_lengths(bundle["labels"][graded]),
        "reference": stats,
    }


def games_fields(bundle, rel_dir):
    """The games.py entry of the bundle as a dict (the field names of games.GAMES)."""
    return sc.games_fields(bundle, rel_dir)


def entry_text(bundle, fields):
    """The games.py entry as source text, with a comment that names the source of each value."""
    home, away = bundle["teams"]
    halves, spans = fields["halves"], bundle["spans"]
    lines = ['    "{}": {{'.format(bundle["stem"])]
    teams = ["        # Home (H): {}.".format(home["name"]),
             "        # Away (A): {}.".format(away["name"])]
    joined = teams[0] + " " + teams[1].lstrip(" #")
    if len(joined) <= 99:
        teams = [joined]
    lines += teams + [
        "        # PFF FC game {}. Tracking frameNum {}..{} and {}..{} (periods 1 and 2).".format(
            bundle["match_id"], spans[1][0], spans[1][1], spans[2][0], spans[2][1]),
        '        "n_video_frames": {},  # last video frame of half 2 + 1'.format(
            fields["n_video_frames"]),
        '        "fps": 25,  # resampled from 29.97 Hz',
        '        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1',
        "        # Video frames inside the tracking periods, cut to kickoff..END of the events.",
        '        "halves": {{1: ({}, {}), 2: ({}, {})}},'.format(
            halves[1][0], halves[1][1], halves[2][0], halves[2][1]),
        "        # First video frame of each half (the kickoff event).",
        '        "kickoff": {{"minute": {minute}, "second": {second}, "frame": {frame}}},'.format(
            **fields["kickoff"]),
        '        "kickoff_2nd": {{"minute": {minute}, "second": {second}, "frame": {frame}}},'
        .format(**fields["kickoff_2nd"]),
        '        "pitch": ({}, {}),  # stadium pitch length, width of the metadata file'.format(
            *fields["pitch"]),
        '        "assets": {',
        '            "atd_meta": "{}",'.format(fields["assets"]["atd_meta"]),
        '            "atd_raw": "{}",'.format(fields["assets"]["atd_raw"]),
        "        },",
        "    },",
    ]
    return "\n".join(lines)


def entry_from_build(summary):
    """The games.py entry text of a build file (the dict of PFF-<id>_build.json)."""
    fields = dict(summary["games_entry"])
    fields["halves"] = {int(h): tuple(v) for h, v in fields["halves"].items()}
    fields["pitch"] = tuple(fields["pitch"])
    spans = {int(h): tuple(v) for h, v in summary["stats"]["periods_tracking"].items()}
    bundle = {"stem": summary["stem"], "match_id": summary["match_id"],
              "teams": [summary["home"], summary["away"]], "spans": spans}
    return entry_text(bundle, fields)


# -- writing the bundle --------------------------------------------------------------------------

def atd_xml(bundle):
    """The metadata XML of the bundle (skillcorner_bundle.atd_xml with this provider name)."""
    text = sc.atd_xml(bundle)
    old = "<ProviderName>{}</ProviderName>".format(sc._xml_text(sc.PROVIDER))
    new = "<ProviderName>{}</ProviderName>".format(sc._xml_text(PROVIDER))
    if text.count(old) != 1:
        raise ValueError("the provider name of skillcorner_bundle.atd_xml changed")
    return text.replace(old, new)


def write_bundle(bundle, out_dir, rel_dir, inputs=None):
    """Write the four files of the bundle into out_dir. Return the build summary dict.

    `inputs` is {name: path} of the PFF files, for the checksums in the summary.
    """
    os.makedirs(out_dir, exist_ok=True)
    stem = bundle["stem"]
    paths = {name: os.path.join(out_dir, stem + suffix) for name, suffix in (
        ("atd_meta", "_atd.xml"), ("atd_raw", "_atd.txt"), ("reference", "_reference.csv"),
        ("build", "_build.json"))}
    with open(paths["atd_meta"], "w", encoding="utf-8") as handle:
        handle.write(atd_xml(bundle))
    with open(paths["atd_raw"], "w") as handle:
        handle.writelines(sc.atd_lines(bundle))
    events_gt.write_csv(paths["reference"], bundle["labels"])
    home, away = bundle["teams"]
    summary = {
        "match_id": bundle["match_id"], "stem": stem, "title": bundle["title"],
        "source": {"folder": SOURCE, "event_version": "June 3, 2025 (top-level Event Data)",
                   "licence": "none stated; team approval 9 Nov 2026, credit PFF FC"},
        "home": home, "away": away,
        "games_entry": games_fields(bundle, rel_dir),
        "slots": {"home": sum(1 for t, _ in bundle["slots"] if t == home["team_id"]),
                  "away": sum(1 for t, _ in bundle["slots"] if t == away["team_id"])},
        "stats": bundle["stats"],
        "inputs": {name: {"file": os.path.basename(path), "bytes": os.path.getsize(path),
                          "sha256": sc.sha256(path)}
                   for name, path in sorted((inputs or {}).items())},
        "outputs": {name: {"file": os.path.basename(path), "sha256": sc.sha256(path)}
                    for name, path in sorted(paths.items()) if name != "build"},
        "reference_md5": sc.md5(paths["reference"]),
    }
    with open(paths["build"], "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


# -- command line --------------------------------------------------------------------------------

def build_match(raw_dir, game_id, out_dir, rel_dir=None):
    """Read, convert and write one match. Return (bundle, summary)."""
    inputs = {name: os.path.join(raw_dir, rel) for name, rel in file_names(game_id).items()}
    meta = read_metadata(inputs["metadata"])
    if meta["game_id"] != str(game_id):
        raise ValueError("{}: game id {} != {}".format(inputs["metadata"], meta["game_id"],
                                                       game_id))
    home_id, away_id = meta["home"]["team_id"], meta["away"]["team_id"]
    roster = read_roster(inputs["roster"], home_id, away_id)
    sides = {"home": home_id, "away": away_id}
    player_of = {(side, jersey): pid for side, team in sides.items()
                 for (t, jersey), pid in roster["player_of"].items() if t == team}
    rows = read_events(inputs["events"], home_id, away_id)
    tracking = read_tracking(inputs["tracking"], player_of)
    bundle = build(meta, roster, rows, tracking)
    summary = write_bundle(bundle, out_dir, rel_dir or "pff/{}".format(game_id), inputs)
    return bundle, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="convert one PFF FC match")
    b.add_argument("--raw-dir", required=True,
                   help="folder with tracking/, events/, metadata/ and rosters/")
    b.add_argument("--game-id", required=True, help="PFF game id, for example 10502")
    b.add_argument("--out-dir", required=True, help="bundle folder (outside the repository)")
    b.add_argument("--rel-dir", default=None,
                   help="bundle folder relative to MTAH_DATA_DIR, for the games.py assets "
                        "(default: pff/<id>)")
    e = sub.add_parser("entries", help="print the games.py entries of built bundles")
    e.add_argument("builds", nargs="+", help="PFF-<id>_build.json files, in the wanted order")
    args = parser.parse_args(argv)
    if args.command == "entries":
        for path in args.builds:
            with open(path) as handle:
                print(entry_from_build(json.load(handle)))
        return 0
    bundle, summary = build_match(args.raw_dir, args.game_id, args.out_dir, args.rel_dir)
    stats = summary["stats"]
    print("{}: H {!r}, A {!r}".format(bundle["stem"], summary["home"]["name"],
                                      summary["away"]["name"]))
    print("frames {}  halves {}  slots {}".format(
        bundle["n_frames"], summary["games_entry"]["halves"], summary["slots"]))
    print("actor_ball median {} m best shift {}  side_check {}  ball visible {}".format(
        stats["actor_ball"]["median_m"], stats["actor_ball"]["best_shift"],
        stats["side_check"]["agrees"], stats["video_ball_visible_share_graded"]))
    print("reference {}".format(stats["reference"]))
    print(entry_text(bundle, summary["games_entry"]))
    print("wrote {}".format(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
