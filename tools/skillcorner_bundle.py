"""SkillCorner Open Data match to an ATD bundle and a reference CSV. Not part of the submission.

    python tools/skillcorner_bundle.py build --match-dir RAW/<id> --out-dir DIR [--rel-dir REL]

The tool reads one match of SkillCorner Open Data (github.com/SkillCorner/opendata, MIT licence,
copyright SkillCorner). Credit SkillCorner when you use the data. The tool writes four files into
DIR:

    SC-<id>_atd.xml        FIFA/EPTS-style metadata that engine/data.py reads unchanged
    SC-<id>_atd.txt        raw tracking lines "<frame_id>:<x,y>;...;<x,y>:<bx,by>"
    SC-<id>_reference.csv  `frame,label` with H, A, D and X, one row for each video frame
    SC-<id>_build.json     the games.py entry fields, checks, statistics and input checksums

Keep DIR outside the repository. Our machines use ~/mtah2027/refs/skillcorner/<id>/.
docs/plan/calibration/SKILLCORNER_REFERENCES.md gives the download and the rebuild commands.

Input files. The folder data/matches/<id>/ of the repository holds three files that the tool reads:

- `<id>_match.json`: the teams (`home_team`, `away_team`), the players (`id`, `team_id`), the
  periods (`match_periods`), the pitch (`pitch_length`, `pitch_width`) and `home_team_side`.
- `<id>_tracking_extrapolated.jsonl`: one JSON line for each tracking frame at 10 Hz. A line has
  `frame`, `period`, `ball_data` {x, y, z, is_detected}, `possession` {player_id, group} and
  `player_data`, a list of {player_id, x, y, is_detected}. The positions are in metres. The
  origin is the centre spot, x runs along the pitch and y points up.
- `<id>_phases_of_play.csv`: one row for each phase of play. The tool reads `frame_start`,
  `frame_end`, `period` and `team_in_possession_id`.

The files are stored with git LFS. Download them from media.githubusercontent.com, not from
raw.githubusercontent.com, which gives the LFS pointer only.

Frames. The tracking frames are 10 Hz and run 0..F without a gap. The tool writes 25 Hz, the rate
of BAR-ATH. Video frame v is at time v / 25 s, and tracking frame f is at time f / 10 s. So
video frame v lies at the tracking position s = 0.4 * v. The ATD `frame_id` of a raw line is
v + 1, as in engine/data.py. The games.py `offsets` are 1 for both halves, as for Metrica.

Halves. A half is the span of tracking frames with that `period` value. The tool requires each
span to be contiguous. The video block of a half runs from ceil(2.5 * first) to
floor(2.5 * last). The build file compares the spans with `match_periods`.

Positions. The tool writes the tracking positions only inside the video blocks of the halves. Every
other frame is NaN. Inside a block, a video frame on a tracking frame (s is a whole number) takes
that frame. Any other video frame lies between tracking frames i and i + 1. It takes the linear
interpolation (1 - w) * p(i) + w * p(i + 1), with w = s - i. If either value is missing, the
video frame is missing. So the tool never fills a gap and never extends a run.

Ball. The tool keeps the ball samples with `is_detected` true. It treats every other sample as
missing. SkillCorner extrapolates the ball while the camera does not see it. The BAR-ATH ball is
missing instead, on about 41 % of the graded frames. The ball height z is not used.

Players. Each player with at least one position inside a half gets one slot. The home team comes
first, and each team is sorted by player id. A substitute has a slot of his own. A slot is NaN
where its player has no position. The tool keeps the positions that SkillCorner extrapolates
(`is_detected` false). SkillCorner tracks from the broadcast video, so the players outside the
camera view are extrapolated. The tool keeps them so that the engine sees 22 players, as on the
BAR-ATH tactical tracking. The build file gives the share of detected player positions.

Channels. Players and ball carry an x and a y channel only, as for Metrica. The files have no
speed. engine.tracks then takes the player speed from position differences.

Coordinates. The ATD normalises to 0..1 with the origin at the top left and the y axis down. The
tool writes `x = (X + length / 2) / length` and `y = 0.5 - Y / width`. The pitch comes from the
match file. The y axis of the files points up: a phase that starts in channel `wide_left` of a
team that attacks towards +x has y > 0.

Teams. `H` is `home_team` and `A` is `away_team` of the match file. The tool writes the home team
as the first `<Team>` of the ATD XML, because engine.data._atd_meta maps the first `<Team>` to `H`.

Labels. SkillCorner phases of play are a third label convention, coarser than Metrica events and
DFL ball status. A phase covers tracking frames frame_start <= f < frame_end. The column
`duration` is (frame_end - frame_start) / 10 on every row, and a phase often starts on the
frame_end of the phase before it. SkillCorner defines phases only while the ball is in play and
one team has it. So the labels at 10 Hz are:

- Inside a phase: `H` if team_in_possession_id is the home team, `A` if it is the away team.
- Inside a half and outside every phase: `D`.
- Outside the halves: `X`.

In 16 of the 20 matches the last phase of period 1 ends after the last frame of the period, by
0.6 to 19.4 s. The tool cuts each phase at the end of its period (clip_phases).

Video frame v takes the label of the nearest tracking frame, round(0.4 * v). No video frame is a
tie. Every video frame outside the blocks of the halves is `X`. The scorer grades only the
non-X frames (possession_metrics.valid_runs). A gap between two phases also becomes `D` when the
ball is in play but no team has it. The build file counts the gaps shorter than SHORT_GAP_S.

Kickoffs. SkillCorner starts the clock of each period at its first frame. The games.py `kickoff`
and `kickoff_2nd` anchors are the first video frame of each half block. The build file gives the
distance of the first detected ball of each half to the centre spot.

Checks in the build file. `team_check` is the share of tracking frames inside a phase where the
tracking `possession` group names the same team as the phase. `side_check` compares the mean x of
the home goalkeeper in half 1 with `home_team_side`. The goalkeeper is the player with the role GK
and the most frames. The tool raises an error if a check shows a swap of the teams.

Deterministic: sorted iteration, fixed number formats, no clock, no network. Python 3.9, numpy
and the standard library.
"""

import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import games  # noqa: E402
from tools import events_gt  # noqa: E402

H, A, D, X = events_gt.H, events_gt.A, events_gt.D, events_gt.X

# The commit of github.com/SkillCorner/opendata that SKILLCORNER_REFERENCES.md downloads (master
# on 8 Nov 2026).
SOURCE_COMMIT = "4340d274572876239c154c90bc507a9b3250a656"
PROVIDER = "SkillCorner Open Data (MIT, SkillCorner), converted by tools/skillcorner_bundle.py"
STEM_PREFIX = "SC-"
# The tracking rate of the files and the rate of the bundle.
SOURCE_FPS, VIDEO_FPS = 10, 25
# Video frame v lies at tracking position v * NUM / DEN (0.4 * v). Integer arithmetic only.
NUM, DEN = SOURCE_FPS, VIDEO_FPS
HALVES = (1, 2)
HOME_SIDE_VALUES = ("left_to_right", "right_to_left")
# Tracking `possession` group values of the two teams.
GROUP_CODE = {"home team": H, "away team": A}
# The player_role acronym of a goalkeeper in the match file.
KEEPER_ROLE = "GK"
# Channel ids of the ATD files that engine/data.py reads (X_CHANNEL, Y_CHANNEL).
CHANNELS = ("x", "y")
# 5 decimals of a normalised position is 1 mm at 105 m. The input has 1 cm steps.
POSITION_FORMAT = "{:.5f}"
# The build file counts the D gaps between two phases shorter than this, in seconds.
SHORT_GAP_S = 2.0
# The first detected ball within this time of a half start gives the kickoff distance, in seconds.
KICKOFF_WINDOW_S = 2.0
# team_check below this share means the teams are swapped (a correct mapping gives about 0.9).
MIN_TEAM_AGREEMENT = 0.5
ANCHOR_FPS = games.ANCHOR_FPS


# -- reading the files ---------------------------------------------------------------------------

def file_names(match_id):
    """{name: file name} of the three input files of a match."""
    return {"match": "{}_match.json".format(match_id),
            "tracking": "{}_tracking_extrapolated.jsonl".format(match_id),
            "phases": "{}_phases_of_play.csv".format(match_id)}


def read_match(path):
    """The fields of a SkillCorner match file that the bundle needs, as a dict.

    Keys: match_id, title, pitch (length, width) in metres, home and away {team_id, name},
    team_of_player {player id: team id}, keepers [player ids with the role GK], periods
    {half: (start_frame, end_frame)} and home_team_side (a list with one value for each half).
    """
    with open(path) as handle:
        raw = json.load(handle)
    home = {"team_id": int(raw["home_team"]["id"]), "name": raw["home_team"]["name"]}
    away = {"team_id": int(raw["away_team"]["id"]), "name": raw["away_team"]["name"]}
    team_of_player, keepers = {}, []
    for player in raw["players"]:
        team = int(player["team_id"])
        if team not in (home["team_id"], away["team_id"]):
            raise ValueError("{}: player {} has team {}".format(path, player["id"], team))
        team_of_player[int(player["id"])] = team
        if (player.get("player_role") or {}).get("acronym") == KEEPER_ROLE:
            keepers.append(int(player["id"]))
    periods = {int(p["period"]): (int(p["start_frame"]), int(p["end_frame"]))
               for p in raw["match_periods"]}
    sides = list(raw.get("home_team_side") or [])
    for side in sides:
        if side not in HOME_SIDE_VALUES:
            raise ValueError("{}: home_team_side {!r}".format(path, side))
    return {"match_id": str(raw["id"]),
            "title": "{} v {}".format(home["name"], away["name"]),
            "date": str(raw.get("date_time", "")),
            "pitch": (float(raw["pitch_length"]), float(raw["pitch_width"])),
            "home": home, "away": away, "team_of_player": team_of_player,
            "keepers": sorted(keepers), "periods": periods, "home_team_side": sides}


def read_tracking(path, team_of_player):
    """The 10 Hz tracking of a match as arrays, streamed line by line.

    Returns a dict: frames int64 (F,), period int8 (F,) with 0 for none, ball_x, ball_y float64
    (F,) of the detected ball only (NaN elsewhere), ball_seen bool (F,) where the file has a ball
    position at all, group int8 (F,) of the tracking possession group (H, A, or X for none),
    players {player id: (x, y, detected)} with float64 and bool arrays (F,).
    Raises ValueError for a gap in the frame numbers or an unknown player.
    """
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    n = len(rows)
    frames = np.array([r["frame"] for r in rows], dtype=np.int64)
    if n == 0 or not np.array_equal(frames, np.arange(frames[0], frames[0] + n)):
        raise ValueError("{}: the frame numbers are not contiguous".format(path))
    if frames[0] != 0:
        raise ValueError("{}: the first frame is {}, not 0".format(path, frames[0]))
    period = np.array([r["period"] or 0 for r in rows], dtype=np.int8)
    ball_x = np.full(n, np.nan)
    ball_y = np.full(n, np.nan)
    ball_seen = np.zeros(n, dtype=bool)
    group = np.full(n, X, dtype=np.int8)
    players = {}
    for i, row in enumerate(rows):
        ball = row.get("ball_data") or {}
        if ball.get("x") is not None and ball.get("y") is not None:
            ball_seen[i] = True
            if ball.get("is_detected"):
                ball_x[i], ball_y[i] = float(ball["x"]), float(ball["y"])
        owner = (row.get("possession") or {}).get("group")
        if owner is not None:
            group[i] = GROUP_CODE[owner]
        for item in row.get("player_data") or ():
            if item.get("x") is None or item.get("y") is None:
                continue
            pid = int(item["player_id"])
            if pid not in team_of_player:
                raise ValueError("{}: player {} is not in the match file".format(path, pid))
            if pid not in players:
                players[pid] = (np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype=bool))
            px, py, det = players[pid]
            px[i], py[i], det[i] = float(item["x"]), float(item["y"]), bool(item["is_detected"])
    return {"frames": frames, "period": period, "ball_x": ball_x, "ball_y": ball_y,
            "ball_seen": ball_seen, "group": group, "players": players}


def read_phases(path):
    """The phases of play as a list of (frame_start, frame_end, period, team id), sorted."""
    out = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            out.append((int(row["frame_start"]), int(row["frame_end"]), int(row["period"]),
                        int(row["team_in_possession_id"])))
    return sorted(out)


# -- frame mapping -------------------------------------------------------------------------------

def half_spans(period):
    """{half: (first, last)} tracking frames of each `period` value. Contiguous, or ValueError."""
    spans = {}
    for half in HALVES:
        rows = np.flatnonzero(period == half)
        if rows.size == 0:
            raise ValueError("no tracking frame of period {}".format(half))
        first, last = int(rows[0]), int(rows[-1])
        if last - first + 1 != rows.size:
            raise ValueError("period {} is not contiguous".format(half))
        spans[half] = (first, last)
    if spans[2][0] <= spans[1][1]:
        raise ValueError("the periods overlap")
    unknown = sorted(set(np.unique(period).tolist()) - {0, 1, 2})
    if unknown:
        raise ValueError("unknown period values {}".format(unknown))
    return spans


def video_block(first, last):
    """(first, last) video frames of a span of tracking frames: ceil(2.5 a)..floor(2.5 b)."""
    return (-((-first * DEN) // NUM), (last * DEN) // NUM)


def resample(values, n_video):
    """A 10 Hz array (F,) or (F, k) at 25 Hz (n_video rows), by linear interpolation.

    A video frame on a tracking frame takes its value. A video frame between two tracking frames
    takes their linear interpolation, or NaN if either is NaN (module docstring, Positions).
    """
    values = np.asarray(values, dtype=np.float64)
    v = np.arange(n_video, dtype=np.int64)
    low, rem = np.divmod(v * NUM, DEN)
    out = np.full((n_video,) + values.shape[1:], np.nan)
    inside = low < values.shape[0]
    exact = inside & (rem == 0)
    out[exact] = values[low[exact]]
    between = inside & (rem > 0) & (low + 1 < values.shape[0])
    w = (rem[between] / float(DEN)).reshape((-1,) + (1,) * (values.ndim - 1))
    out[between] = (1.0 - w) * values[low[between]] + w * values[low[between] + 1]
    return out


def nearest_source(n_video):
    """The nearest tracking frame of each video frame, round(0.4 * v), as int64 (n_video,)."""
    v = np.arange(n_video, dtype=np.int64)
    return (2 * v * NUM + DEN) // (2 * DEN)


# -- the bundle ----------------------------------------------------------------------------------

def source_labels(n_source, spans, phases, home_id, away_id):
    """Labels of the 10 Hz frames (module docstring, Labels), as int8 (n_source,)."""
    labels = np.full(n_source, X, dtype=np.int8)
    for half, (first, last) in sorted(spans.items()):
        labels[first:last + 1] = D
    code = {home_id: H, away_id: A}
    for start, end, half, team in clip_phases(phases, spans):
        if team not in code:
            raise ValueError("phase {}..{} has team {}".format(start, end, team))
        if np.isin(labels[start:end], (H, A)).any():
            raise ValueError("phase {}..{} overlaps another phase".format(start, end))
        labels[start:end] = code[team]
    return labels


def clip_phases(phases, spans):
    """The phases cut to the tracking frames of their period, [(start, end, period, team)].

    In most matches the last phase of period 1 ends a few frames after the last frame of the
    period. The tool cuts it at the period end. A phase that starts outside its period is an
    error.
    """
    out = []
    for start, end, half, team in phases:
        if half not in spans or not spans[half][0] <= start <= spans[half][1] or end <= start:
            raise ValueError("phase {}..{} lies outside period {}".format(start, end, half))
        out.append((start, min(end, spans[half][1] + 1), half, team))
    return out


def build(match, tracking, phases):
    """The bundle arrays, the games.py fields and the checks of one match, as a dict.

    Keys: match_id, stem, title, pitch, spans {half: tracking (first, last)}, blocks {half: video
    (first, last)}, n_frames, teams [home, away], slots [(team id, player id)], px, py
    (n_frames, n_slots) and bx, by (n_frames,) normalised, labels int8 (n_frames,), source_labels
    int8 (F,), and stats (the checks and statistics of the build file).
    """
    home, away = match["home"], match["away"]
    length, width = match["pitch"]
    spans = half_spans(tracking["period"])
    blocks = {h: video_block(a, b) for h, (a, b) in spans.items()}
    n_frames = blocks[max(blocks)][1] + 1
    n_source = tracking["frames"].size

    in_half = np.zeros(n_source, dtype=bool)
    for first, last in spans.values():
        in_half[first:last + 1] = True
    in_block = np.zeros(n_frames, dtype=bool)
    for first, last in blocks.values():
        in_block[first:last + 1] = True

    def norm_x(values):
        return (values + length / 2.0) / length

    def norm_y(values):
        return 0.5 - values / width

    def to_video(values):
        out = resample(np.where(in_half, values, np.nan), n_frames)
        out[~in_block] = np.nan
        return out

    people = {}
    for pid, (x, _, _) in sorted(tracking["players"].items()):
        if np.isfinite(x[in_half]).any():
            people.setdefault(match["team_of_player"][pid], []).append(pid)
    slots = [(team["team_id"], pid) for team in (home, away)
             for pid in sorted(people.get(team["team_id"], []))]
    px = np.full((n_frames, len(slots)), np.nan)
    py = np.full((n_frames, len(slots)), np.nan)
    for column, (_, pid) in enumerate(slots):
        x, y, _ = tracking["players"][pid]
        px[:, column] = norm_x(to_video(x))
        py[:, column] = norm_y(to_video(y))
    bx = norm_x(to_video(tracking["ball_x"]))
    by = norm_y(to_video(tracking["ball_y"]))

    labels10 = source_labels(n_source, spans, phases, home["team_id"], away["team_id"])
    nearest = np.minimum(nearest_source(n_frames), n_source - 1)
    labels = labels10[nearest]
    labels[~in_block] = X

    bundle = {"match_id": match["match_id"], "stem": STEM_PREFIX + match["match_id"],
              "title": match["title"], "pitch": (length, width), "spans": spans,
              "blocks": blocks, "n_frames": int(n_frames), "teams": [home, away],
              "slots": slots, "px": px, "py": py, "bx": bx, "by": by, "labels": labels,
              "source_labels": labels10}
    bundle["stats"] = statistics(bundle, match, tracking, phases, in_half, in_block)
    return bundle


def statistics(bundle, match, tracking, phases, in_half, in_block):
    """The checks and summary statistics of the build file, as a dict.

    Raises ValueError if team_check or side_check shows a swap of the teams.
    """
    spans, labels10 = bundle["spans"], bundle["source_labels"]
    home_id = match["home"]["team_id"]
    in_phase = np.isin(labels10, (H, A))
    owner = tracking["group"]
    known = in_phase & (owner != X)
    agree = float(np.mean(owner[known] == labels10[known])) if known.any() else None
    if agree is not None and agree < MIN_TEAM_AGREEMENT:
        raise ValueError("team_check {:.3f}: the phases and the tracking possession disagree"
                         .format(agree))

    side = None
    first, last = spans[1]
    keeper_x = {}
    for pid in match["keepers"]:
        x = tracking["players"].get(pid, (np.full(1, np.nan),))[0][first:last + 1]
        if np.isfinite(x).any():
            team = match["team_of_player"][pid]
            count = int(np.isfinite(x).sum())
            if count > keeper_x.get(team, (0, 0.0))[0]:
                keeper_x[team] = (count, float(np.nanmean(x)))
    if match["home_team_side"] and home_id in keeper_x:
        mean_x = keeper_x[home_id][1]
        expected = "left_to_right" if mean_x < 0 else "right_to_left"
        side = {"home_keeper_mean_x_half1_m": round(mean_x, 3),
                "home_team_side_half1": match["home_team_side"][0],
                "agrees": expected == match["home_team_side"][0]}
        if not side["agrees"]:
            raise ValueError("side_check: home keeper mean x {:.2f} m in half 1, "
                             "home_team_side {}".format(mean_x, match["home_team_side"][0]))

    kickoff = {}
    window = int(KICKOFF_WINDOW_S * SOURCE_FPS)
    for half, (first, _) in sorted(spans.items()):
        x = tracking["ball_x"][first:first + window + 1]
        y = tracking["ball_y"][first:first + window + 1]
        seen = np.flatnonzero(np.isfinite(x))
        kickoff[half] = ({"frames_after_start": int(seen[0]),
                          "centre_distance_m": round(float(np.hypot(x[seen[0]], y[seen[0]])), 3)}
                         if seen.size else None)

    gaps = []
    for half, (first, last) in sorted(spans.items()):
        inner = [(a, b) for a, b, h, _ in clip_phases(phases, spans) if h == half]
        for (_, end), (start, _) in zip(inner, inner[1:]):
            if start > end:
                gaps.append((start - end) / float(SOURCE_FPS))
    tracks = [tracking["players"][pid] for pid in sorted(tracking["players"])]
    present = np.concatenate([np.isfinite(x[in_half]) for x, _, _ in tracks] or
                             [np.zeros(0, dtype=bool)])
    detected = np.concatenate([d[in_half] for _, _, d in tracks] or [np.zeros(0, dtype=bool)])
    graded = bundle["labels"] != X
    stats = events_gt.reference_stats(bundle["labels"], VIDEO_FPS)
    stats = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in sorted(stats.items())}
    periods_json = {str(h): list(v) for h, v in sorted(match["periods"].items())}
    return {
        "periods_match_json": periods_json,
        "periods_tracking": {str(h): list(v) for h, v in sorted(spans.items())},
        "periods_agree": all(match["periods"].get(h) == spans[h] for h in spans),
        "team_check": None if agree is None else round(agree, 6),
        "team_check_frames": int(known.sum()),
        "side_check": side,
        "kickoff_ball": {str(h): v for h, v in kickoff.items()},
        "phases": len(phases),
        "phases_cut_at_period_end": sum(1 for a, b in zip(phases, clip_phases(phases, spans))
                                        if a != b),
        "phase_gaps": len(gaps),
        "phase_gaps_under_{:g}s".format(SHORT_GAP_S): int(sum(g < SHORT_GAP_S for g in gaps)),
        "source_in_phase_share": round(float(np.mean(in_phase[in_half])), 6),
        "source_ball_detected_share": round(float(np.mean(
            np.isfinite(tracking["ball_x"][in_half]))), 6),
        "source_ball_extrapolated_share": round(float(np.mean(
            tracking["ball_seen"][in_half] & ~np.isfinite(tracking["ball_x"][in_half]))), 6),
        "source_player_detected_share": round(float(np.mean(detected[present])), 6)
        if present.any() else None,
        "video_ball_visible_share_graded": round(float(np.mean(
            np.isfinite(bundle["bx"][graded]))), 6),
        "video_slots_present_median": float(np.median(np.sum(
            np.isfinite(bundle["px"][in_block]), axis=1))),
        "reference": stats,
    }


def anchor(video_frame):
    """games.py anchor {minute, second, frame} of a video frame at ANCHOR_FPS."""
    seconds, frame = divmod(int(video_frame), ANCHOR_FPS)
    minute, second = divmod(seconds, 60)
    return {"minute": minute, "second": second, "frame": frame}


def games_fields(bundle, rel_dir):
    """The games.py entry of the bundle as a dict (the field names of games.GAMES).

    The offsets are 1: the ATD frame_id is the video frame + 1. The halves are frame_id spans.
    """
    blocks = bundle["blocks"]
    return {
        "n_video_frames": int(bundle["n_frames"]),
        "fps": VIDEO_FPS,
        "offsets": {half: 1 for half in sorted(blocks)},
        "halves": {half: (int(a) + 1, int(b) + 1) for half, (a, b) in sorted(blocks.items())},
        "kickoff": anchor(blocks[1][0]),
        "kickoff_2nd": anchor(blocks[2][0]),
        "pitch": (float(bundle["pitch"][0]), float(bundle["pitch"][1])),
        "assets": {
            "atd_meta": "{}/{}_atd.xml".format(rel_dir, bundle["stem"]),
            "atd_raw": "{}/{}_atd.txt".format(rel_dir, bundle["stem"]),
        },
    }


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
        "        # SkillCorner match {}.".format(bundle["match_id"]),
        '        "n_video_frames": {},  # floor(2.5 * last tracking frame of period 2) + 1'
        .format(fields["n_video_frames"]),
        '        "fps": 25,  # resampled from 10 Hz',
        '        "offsets": {1: 1, 2: 1},  # frame_id = video frame + 1',
        "        # Tracking frames {}..{} and {}..{} (`period` 1 and 2), times 2.5.".format(
            spans[1][0], spans[1][1], spans[2][0], spans[2][1]),
        '        "halves": {{1: ({}, {}), 2: ({}, {})}},'.format(
            halves[1][0], halves[1][1], halves[2][0], halves[2][1]),
        "        # First video frame of each half (SkillCorner starts each period clock there).",
        '        "kickoff": {{"minute": {minute}, "second": {second}, "frame": {frame}}},'.format(
            **fields["kickoff"]),
        '        "kickoff_2nd": {{"minute": {minute}, "second": {second}, "frame": {frame}}},'
        .format(**fields["kickoff_2nd"]),
        '        "pitch": ({}, {}),  # pitch_length, pitch_width of the match file'.format(
            *fields["pitch"]),
        '        "assets": {',
        '            "atd_meta": "{}",'.format(fields["assets"]["atd_meta"]),
        '            "atd_raw": "{}",'.format(fields["assets"]["atd_raw"]),
        "        },",
        "    },",
    ]
    return "\n".join(lines)


# -- writing the bundle --------------------------------------------------------------------------

def _xml_text(value):
    """XML-escaped text."""
    return (str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def atd_xml(bundle):
    """The FIFA/EPTS-style metadata XML of the bundle, as a string.

    It has the elements that engine.data._atd_meta reads: two <Team> (home first), one <Player>
    per slot, the <PlayerChannel> table, one DataFormatSpecification and the FrameRate.
    """
    home, away = bundle["teams"]
    length, width = bundle["pitch"]
    out = ['<?xml version="1.0" encoding="utf-8"?>', "<main>", "  <Metadata>",
           "    <GlobalConfig>",
           "      <ProviderName>{}</ProviderName>".format(_xml_text(PROVIDER)),
           "      <FrameRate>{}</FrameRate>".format(VIDEO_FPS),
           "      <ProviderGlobalParameters>"]
    for half, (first, last) in sorted(bundle["blocks"].items()):
        for name, value in (("half{}_first_frame_id".format(half), first + 1),
                            ("half{}_last_frame_id".format(half), last + 1)):
            out.append("        <ProviderParameter><Name>{}</Name><Value>{}</Value>"
                       "</ProviderParameter>".format(name, value))
    out += ["      </ProviderGlobalParameters>", "    </GlobalConfig>",
            "    <Sessions>",
            '      <Session id="{}">'.format(_xml_text(bundle["match_id"])),
            "        <FieldSize><Width>{}</Width><Height>{}</Height></FieldSize>".format(
                length, width),
            "      </Session>", "    </Sessions>", "    <Teams>"]
    for team in (home, away):
        out.append('      <Team id="{}"><Name>{}</Name></Team>'.format(
            team["team_id"], _xml_text(team["name"])))
    out += ["    </Teams>", "    <Players>"]
    for team_id, pid in bundle["slots"]:
        out.append('      <Player id="{}" teamId="{}"/>'.format(pid, team_id))
    out += ["    </Players>", "    <Devices>", '      <Device id="device1">', "        <Sensors>",
            '          <Sensor id="position">', "            <Channels>"]
    for channel in CHANNELS:
        out.append('              <Channel id="{}"><Unit>normalized</Unit></Channel>'.format(
            channel))
    out += ["            </Channels>", "          </Sensor>", "        </Sensors>",
            "      </Device>", "    </Devices>", "    <PlayerChannels>"]
    for slot, (_, pid) in enumerate(bundle["slots"]):
        for channel in CHANNELS:
            out.append('      <PlayerChannel channelId="{}" id="p{}_{}" playerId="{}"/>'.format(
                channel, slot + 1, channel, pid))
    out += ["    </PlayerChannels>", "  </Metadata>", "  <DataFormatSpecifications>",
            '    <DataFormatSpecification endFrame="{}" separator=":" startFrame="1">'.format(
                bundle["n_frames"])]
    for slot in range(len(bundle["slots"])):
        refs = "".join('<PlayerChannelRef playerChannelId="p{}_{}"/>'.format(slot + 1, channel)
                       for channel in CHANNELS)
        out.append("      <SplitRegister>{}</SplitRegister>".format(refs))
    out.append("      <BallChannelRefs>{}</BallChannelRefs>".format(
        "".join('<BallChannelRef channelId="{}"/>'.format(c) for c in CHANNELS)))
    out += ["    </DataFormatSpecification>", "  </DataFormatSpecifications>", "</main>", ""]
    return "\n".join(out)


def atd_lines(bundle, chunk=10000):
    """The raw ATD lines of the bundle, one string for each video frame, in frame order."""
    n_slots = len(bundle["slots"])
    pair = ",".join([POSITION_FORMAT] * len(CHANNELS))
    template = "{}:" + ";".join([pair] * n_slots) + ":" + pair + "\n"
    for start in range(0, bundle["n_frames"], chunk):
        stop = min(start + chunk, bundle["n_frames"])
        players = np.stack([bundle["px"][start:stop], bundle["py"][start:stop]],
                           axis=2).reshape(stop - start, 2 * n_slots)
        ball = np.stack([bundle["bx"][start:stop], bundle["by"][start:stop]], axis=1)
        table = np.concatenate([players, ball], axis=1).tolist()
        for i, row in enumerate(table):
            yield template.format(start + i + 1, *row).replace("nan", "NaN")


def sha256(path):
    """Hex sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def md5(path):
    """Hex md5 of a file (external_sweep.py records the reference md5)."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bundle(bundle, out_dir, rel_dir, inputs=None):
    """Write the four files of the bundle into out_dir. Return the build summary dict.

    `inputs` is {name: path} of the SkillCorner files, for the checksums in the summary.
    """
    os.makedirs(out_dir, exist_ok=True)
    stem = bundle["stem"]
    paths = {name: os.path.join(out_dir, stem + suffix) for name, suffix in (
        ("atd_meta", "_atd.xml"), ("atd_raw", "_atd.txt"), ("reference", "_reference.csv"),
        ("build", "_build.json"))}
    with open(paths["atd_meta"], "w", encoding="utf-8") as handle:
        handle.write(atd_xml(bundle))
    with open(paths["atd_raw"], "w") as handle:
        handle.writelines(atd_lines(bundle))
    events_gt.write_csv(paths["reference"], bundle["labels"])
    home, away = bundle["teams"]
    summary = {
        "match_id": bundle["match_id"], "stem": stem, "title": bundle["title"],
        "source": {"repository": "github.com/SkillCorner/opendata", "commit": SOURCE_COMMIT,
                   "licence": "MIT, copyright SkillCorner"},
        "home": home, "away": away,
        "games_entry": games_fields(bundle, rel_dir),
        "slots": {"home": sum(1 for t, _ in bundle["slots"] if t == home["team_id"]),
                  "away": sum(1 for t, _ in bundle["slots"] if t == away["team_id"])},
        "stats": bundle["stats"],
        "inputs": {name: {"file": os.path.basename(path), "bytes": os.path.getsize(path),
                          "sha256": sha256(path)}
                   for name, path in sorted((inputs or {}).items())},
        "outputs": {name: {"file": os.path.basename(path), "sha256": sha256(path)}
                    for name, path in sorted(paths.items()) if name != "build"},
        "reference_md5": md5(paths["reference"]),
    }
    with open(paths["build"], "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


# -- command line --------------------------------------------------------------------------------

def build_match(match_dir, out_dir, rel_dir=None):
    """Read, convert and write one match folder. Return (bundle, summary)."""
    match_id = os.path.basename(os.path.normpath(match_dir))
    names = file_names(match_id)
    inputs = {name: os.path.join(match_dir, file) for name, file in names.items()}
    match = read_match(inputs["match"])
    if match["match_id"] != match_id:
        raise ValueError("{}: match id {} != folder {}".format(
            inputs["match"], match["match_id"], match_id))
    tracking = read_tracking(inputs["tracking"], match["team_of_player"])
    phases = read_phases(inputs["phases"])
    bundle = build(match, tracking, phases)
    summary = write_bundle(bundle, out_dir, rel_dir or "skillcorner/{}".format(match_id), inputs)
    return bundle, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="convert one SkillCorner match")
    b.add_argument("--match-dir", required=True,
                   help="folder data/matches/<id> with the three input files")
    b.add_argument("--out-dir", required=True, help="bundle folder (outside the repository)")
    b.add_argument("--rel-dir", default=None,
                   help="bundle folder relative to MTAH_DATA_DIR, for the games.py assets "
                        "(default: skillcorner/<id>)")
    args = parser.parse_args(argv)
    bundle, summary = build_match(args.match_dir, args.out_dir, args.rel_dir)
    stats = summary["stats"]
    print("{}: H {!r}, A {!r}".format(bundle["stem"], summary["home"]["name"],
                                      summary["away"]["name"]))
    print("frames {}  halves {}  slots {}".format(
        bundle["n_frames"], summary["games_entry"]["halves"], summary["slots"]))
    print("team_check {}  side_check {}  ball visible {}  reference {}".format(
        stats["team_check"], (stats["side_check"] or {}).get("agrees"),
        stats["video_ball_visible_share_graded"], stats["reference"]))
    print(entry_text(bundle, summary["games_entry"]))
    print("wrote {}".format(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
