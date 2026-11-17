"""Sportec Open DFL match to an ATD-style bundle and a reference CSV. Not part of the submission.

    python tools/dfl_bundle.py build --positions P.xml --matchinfo M.xml --out-dir DIR
    python tools/dfl_bundle.py audit --game DFL-J03WMX [--json OUT.json] [--allow-sealed]

The tool reads one match of the Sportec Open DFL dataset (Bassek et al. 2025, Sci Data 12, 195,
DOI 10.1038/s41597-025-04505-y). The files are on figshare (DOI 10.6084/m9.figshare.28196177,
CC-BY 4.0). It writes three files and a summary into DIR:

    <stem>_atd.xml        FIFA/EPTS-style metadata that engine/data.py reads unchanged
    <stem>_atd.txt        raw tracking lines "<frame_id>:<x,y,s>;...;<x,y,s>:<bx,by>"
    <stem>_reference.csv  `frame,label` with H, A, D and X, one row for each video frame
    <stem>_build.json     the games.py entry fields, label coverage and input checksums

The stem is `DFL-<id>`, with the last part of the DFL match id (`DFL-MAT-J03WMX` gives
`DFL-J03WMX`). Keep DIR outside the repository. RULES.md allows labelled data from other matches
for validation. The data stays on this machine like every other match asset.

Download. Get the file list with `https://api.figshare.com/v2/articles/28196177`. Download each
file from `https://ndownloader.figshare.com/files/<id>` into a new, empty folder outside the
repository. Check each file against the `computed_md5` of the list. On 2 Nov 2026 the host that
kloppy 3.18 uses (`springernature.figshare.com/ndownloader`) answered HTTP 202 with an empty body.
The tool reads the downloaded XML with the standard library. It does not need kloppy. On
DFL-MAT-J03WMX, kloppy 3.18 `sportec.load_tracking` on the same local files gives the same `H`,
`A` and `D` on all 145,967 frames of the halves.

Input fields. The match information file gives the teams (`TeamId`, `TeamName`, `Role`), the
players (`PersonId`), the pitch (`PitchX`, `PitchY` of `Environment`) and the half durations
(`TotalTimeFirstHalf`, `TotalTimeSecondHalf` of `OtherGameInformation`, in milliseconds). The
positions file has one `FrameSet` for each object and half (`GameSection` `firstHalf` or
`secondHalf`). A `Frame` has the frame counter `N`, the time `T`, the position `X` and `Y` in metres
and the speed `S` in km/h. The ball `FrameSet` has `TeamId="BALL"`. Its frames also carry
`BallStatus` and `BallPossession`. The tool skips the `FrameSet` of each referee
(`TeamId="referee"`). Any other `TeamId` is an error.

Frames. The frame counter `N` of the DFL files starts at 10000 for the first half and at 100000
for the second half. The tool keeps the native 25 Hz. It maps both halves with one offset, the
first `N` of the first half: `video_frame = N - offset`. The halftime jump of the counter becomes
a block of frames with no data. The ATD `frame_id` of a raw line is `video_frame + 1`, as in
engine/data.py. The games.py `halves` are spans of `N`, and the games.py `offsets` are this offset
for both halves.

Halves. A half starts at the first `N` of its ball `FrameSet`. It ends at that frame plus the half
duration of the match information file, converted to frames (`ms * fps / 1000`). The tool checks
that the last ball frame of the half is this frame and that every player frame lies inside the
half. It raises an error if not. The frame rate comes from the `T` times of the ball frames.

Teams. `H` is the team with `Role="home"` and `A` is the team with `Role="guest"`. The match
information file does not always list the home team first. The tool writes the home team as the
first `<Team>` of `<stem>_atd.xml`, because engine.data._atd_meta maps the first `<Team>` to `H`.
`BallPossession` 1 is the home team and 2 is the guest team (kloppy 3.18,
kloppy.sportec.load_open_tracking_data reads it the same way). The audit command checks this
against the nearest player.

Slots. The tool gives one slot to each player with at least one frame, home team first, each team
sorted by `PersonId`. A slot is NaN on frames where its player is not on the pitch. A substitute
therefore has a slot of his own. The ATD XML has one DataFormatSpecification for the whole match.

Coordinates. DFL positions have the origin at the centre spot and the y axis up (kloppy
SportecTrackingDataCoordinateSystem). The ATD normalises to 0..1 with the origin at the top left
and the y axis down (kloppy MetricaCoordinateSystem). The tool writes
`x = (X + length / 2) / length` and `y = 0.5 - Y / width` with the pitch of the match information
file. Inside the tool the values are metres and seconds. The player speed channel `s` is
`S / 3.6`, in m/s, the unit that engine/data.py expects. On DFL-MAT-J03WMX the median ratio of
`S` to the speed from position differences is 3.59. The tool writes no ball speed, because
engine/data.py ignores it. The ball keeps the position of the file on every frame, also while the
ball is dead. The tool blanks nothing.

Labels. On each frame inside a half with a ball frame:

- `BallStatus` 0 is `D`.
- `BallStatus` 1 with `BallPossession` 1 is `H`. With `BallPossession` 2 it is `A`.
- `BallStatus` 1 with any other `BallPossession` is `X`.
- Any other `BallStatus` is `X`.

The seven matches have no frame of the last two kinds. Their label coverage is 100 % of the
frames of the halves.

Every frame outside the two halves is `X`, and so is every frame of a half without a ball frame.
The tool never converts a reference `X` to `D`. The scorer grades only the non-X frames
(possession_metrics.valid_runs), so the halftime block separates the halves, as it does for
BAR-ATH.

Kickoffs. The games.py `kickoff` and `kickoff_2nd` anchors are the first frame of each half with
`BallStatus` 1. If the first half does not start alive, the entry also sets `kickoff_from_anchor`.

Audit. The audit command reads a built bundle through engine.data.load_tactical(), the engine's
own data layer, and its reference CSV. It prints summary statistics only: the ball at each
kickoff, the ball at the `D` onsets and the restarts, the side of each team in each half, the
nearest player's team against the reference team, how the export shows the dead ball and the
reference's own possession and `D` counts. docs/plan/calibration/DFL_validation.md reads them.

Sealed matches. `SEALED` names the two matches of docs/plan/calibration/SPLITS.md. The build
command prints the frame count, the halves, the label coverage and the games.py entry. It prints
no label statistic. The audit command refuses a sealed match without `--allow-sealed`. Use that
option once, at the final selection.

Deterministic: sorted iteration, fixed number formats, no clock, no network. Python 3.9, numpy
and the standard library.
"""

import argparse
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from array import array
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import games  # noqa: E402
import possession_metrics as pm  # noqa: E402
from tools import events_gt  # noqa: E402

H, A, D, X = events_gt.H, events_gt.A, events_gt.D, events_gt.X

# The two sealed matches of docs/plan/calibration/SPLITS.md (the two largest ids in ASCII order).
SEALED = ("DFL-MAT-J03WQQ", "DFL-MAT-J03WR9")

# Names of the DFL files (the figshare record of the dataset).
SECTIONS = (("firstHalf", 1), ("secondHalf", 2))
DURATION_ATTRS = {1: "TotalTimeFirstHalf", 2: "TotalTimeSecondHalf"}
PLAYING_ATTRS = {1: "PlayingTimeFirstHalf", 2: "PlayingTimeSecondHalf"}
BALL_TEAM_ID = "BALL"
REFEREE_TEAM_ID = "referee"
STATUS_ATTR, POSSESSION_ATTR = "BallStatus", "BallPossession"
STATUS_DEAD, STATUS_ALIVE = "0", "1"
HOME_ROLE, GUEST_ROLE = "home", "guest"
# BallPossession value -> team role (kloppy 3.18 sportec tracking deserializer: 1 is home).
POSSESSION_ROLE = {"1": HOME_ROLE, "2": GUEST_ROLE}
ROLE_CODE = {HOME_ROLE: H, GUEST_ROLE: A}
# DFL speed S is km/h. 1 m/s = 3.6 km/h.
KMH_PER_MS = 3.6
# The anchors of games.py count (minute, second, frame) at this rate (games.ANCHOR_FPS).
ANCHOR_FPS = games.ANCHOR_FPS

# Channel ids of the ATD files that engine/data.py reads (X_CHANNEL, Y_CHANNEL, SPEED_CHANNEL).
PLAYER_CHANNELS = ("x", "y", "s")
BALL_CHANNELS = ("x", "y")
# Number formats of the raw ATD lines: 5 decimals of a normalised position is 1 mm at 105 m.
POSITION_FORMAT, SPEED_FORMAT = "{:.5f}", "{:.3f}"


# -- reading the DFL files -----------------------------------------------------------------------

def stem_of(match_id):
    """'DFL-MAT-J03WMX' -> 'DFL-J03WMX', the bundle stem and the games.py name."""
    return "DFL-" + match_id.split("-")[-1]


def read_matchinfo(path):
    """The fields of a DFL match information XML that the bundle needs, as a dict.

    Keys: match_id, title, competition, season, match_day, result, pitch (length, width) in
    metres, teams [{team_id, name, role, players [person ids]}] in file order, durations_ms and
    playing_ms {half: int}.
    """
    root = ET.parse(path).getroot()
    info = root.find("MatchInformation")
    if info is None:
        raise ValueError("{}: no MatchInformation".format(path))
    general = info.find("General").attrib
    environment = info.find("Environment").attrib
    other = info.find("OtherGameInformation").attrib
    teams = []
    for team in info.find("Teams").findall("Team"):
        players = [p.get("PersonId") for p in team.iter("Player")]
        teams.append({"team_id": team.get("TeamId"), "name": team.get("TeamName"),
                      "role": team.get("Role"), "players": players})
    roles = sorted(t["role"] for t in teams)
    if roles != sorted([HOME_ROLE, GUEST_ROLE]):
        raise ValueError("{}: team roles {} are not one home and one guest".format(path, roles))
    return {
        "match_id": general["MatchId"],
        "title": general.get("MatchTitle", ""),
        "competition": general.get("CompetitionName", ""),
        "season": general.get("Season", ""),
        "match_day": general.get("MatchDay", ""),
        "result": general.get("Result", ""),
        "pitch": (float(environment["PitchX"]), float(environment["PitchY"])),
        "teams": teams,
        "durations_ms": {half: int(other[attr]) for half, attr in DURATION_ATTRS.items()},
        "playing_ms": {half: int(other[attr]) for half, attr in PLAYING_ATTRS.items()
                       if attr in other},
    }


def team_by_role(matchinfo, role):
    """The team dict of matchinfo with this role."""
    return next(t for t in matchinfo["teams"] if t["role"] == role)


def _new_track(is_ball):
    """Empty column arrays for one FrameSet."""
    track = {"n": array("q"), "x": array("d"), "y": array("d"), "s": array("d")}
    if is_ball:
        track.update(status=[], possession=[], t=[])
    return track


def read_positions(path, team_ids):
    """The FrameSets of a DFL positions XML as column arrays, streamed.

    `team_ids` is the set of the two club ids. Returns {"players": {(half, team_id, person_id):
    track}, "ball": {half: track}}. A track has the arrays n, x, y, s (metres, km/h as in the
    file). A ball track also has the lists status, possession and t (the time strings of its
    first frames, for the frame rate). Referee FrameSets are skipped.
    """
    section_half = dict(SECTIONS)
    players, ball = {}, {}
    track, frameset, keep_times = None, None, 0
    for event, elem in ET.iterparse(path, events=("start", "end")):
        if event == "start" and elem.tag == "FrameSet":
            frameset = elem
            section = elem.get("GameSection")
            team = elem.get("TeamId")
            if section not in section_half:
                raise ValueError("{}: unknown GameSection {!r}".format(path, section))
            half = section_half[section]
            if team == REFEREE_TEAM_ID:
                track = None
            elif team == BALL_TEAM_ID:
                if half in ball:
                    raise ValueError("{}: two ball FrameSets in {}".format(path, section))
                track = ball[half] = _new_track(True)
                keep_times = 100
            elif team in team_ids:
                key = (half, team, elem.get("PersonId"))
                if key in players:
                    raise ValueError("{}: two FrameSets for {}".format(path, key))
                track = players[key] = _new_track(False)
            else:
                raise ValueError("{}: unknown TeamId {!r}".format(path, team))
        elif event == "end" and elem.tag == "Frame":
            if track is not None:
                track["n"].append(int(elem.get("N")))
                track["x"].append(float(elem.get("X")))
                track["y"].append(float(elem.get("Y")))
                track["s"].append(float(elem.get("S", "nan")))
                if "status" in track:
                    track["status"].append(elem.get(STATUS_ATTR))
                    track["possession"].append(elem.get(POSSESSION_ATTR))
                    if keep_times:
                        track["t"].append(elem.get("T"))
                        keep_times -= 1
            # Drop the finished Frame so that memory stays flat. The FrameSet attributes were
            # read at its start event.
            frameset.clear()
        elif event == "end" and elem.tag == "FrameSet":
            elem.clear()
            track = None
    return {"players": players, "ball": ball}


def frame_rate(times):
    """Frames per second from the ISO time strings of consecutive frames, as an int."""
    stamps = [datetime.fromisoformat(t) for t in times]
    steps = np.diff([s.timestamp() for s in stamps])
    if steps.size == 0:
        raise ValueError("need two frame times to get the frame rate")
    step = float(np.median(steps))
    rate = int(round(1.0 / step))
    if abs(rate * step - 1.0) > 1e-6:
        raise ValueError("frame step {} s is not a whole frame rate".format(step))
    return rate


# -- the bundle ----------------------------------------------------------------------------------

def half_spans(matchinfo, positions, fps):
    """{half: (first N, last N)} from the ball FrameSets and the match information durations.

    Raises ValueError if the last ball frame of a half is not first N + duration in frames, or
    if a player frame lies outside its half.
    """
    spans = {}
    for _, half in SECTIONS:
        if half not in positions["ball"]:
            raise ValueError("no ball FrameSet for half {}".format(half))
        n = np.frombuffer(positions["ball"][half]["n"], dtype=np.int64)
        duration = matchinfo["durations_ms"][half] * fps
        if duration % 1000:
            raise ValueError("half {} duration {} ms is not a whole number of frames".format(
                half, matchinfo["durations_ms"][half]))
        first, last = int(n.min()), int(n.min()) + duration // 1000
        if int(n.max()) != last:
            raise ValueError("half {}: last ball frame {} != first {} + duration {} frames".format(
                half, int(n.max()), first, duration // 1000))
        spans[half] = (first, last)
    for (half, team, person), track in sorted(positions["players"].items()):
        n = np.frombuffer(track["n"], dtype=np.int64)
        if n.size and (n.min() < spans[half][0] or n.max() > spans[half][1]):
            raise ValueError("player {} has frames outside half {}".format(person, half))
    halves = sorted(spans)
    for before, after in zip(halves, halves[1:]):
        if spans[after][0] <= spans[before][1]:
            raise ValueError("halves {} and {} overlap".format(before, after))
    return spans


def anchor(video_frame):
    """games.py anchor {minute, second, frame} of a video frame at ANCHOR_FPS."""
    seconds, frame = divmod(int(video_frame), ANCHOR_FPS)
    minute, second = divmod(seconds, 60)
    return {"minute": minute, "second": second, "frame": frame}


def build(matchinfo, positions):
    """The bundle arrays and the games.py fields of one match, as a dict.

    Keys: match_id, stem, title, fps, pitch, halves {half: (first N, last N)}, offset,
    n_frames, kickoff_video {half: video frame}, teams [home, guest] (dicts of read_matchinfo),
    slots [(team_id, person_id)], px, py, ps (n_frames, n_slots) normalised and m/s, bx, by
    (n_frames,) normalised, labels int8 (n_frames,) and unknown_label_frames {half: count}.
    """
    home, guest = team_by_role(matchinfo, HOME_ROLE), team_by_role(matchinfo, GUEST_ROLE)
    fps = frame_rate(positions["ball"][1]["t"])
    spans = half_spans(matchinfo, positions, fps)
    offset = spans[1][0]
    n_frames = spans[max(spans)][1] - offset + 1
    length, width = matchinfo["pitch"]

    def norm_x(values):
        return (np.asarray(values, dtype=np.float64) + length / 2.0) / length

    def norm_y(values):
        return 0.5 - np.asarray(values, dtype=np.float64) / width

    people = {}
    for (half, team, person) in sorted(positions["players"]):
        people.setdefault(team, set()).add(person)
    slots = [(team["team_id"], person) for team in (home, guest)
             for person in sorted(people.get(team["team_id"], ()))]
    slot_of = {key: i for i, key in enumerate(slots)}

    px = np.full((n_frames, len(slots)), np.nan)
    py = np.full((n_frames, len(slots)), np.nan)
    ps = np.full((n_frames, len(slots)), np.nan)
    for (half, team, person), track in sorted(positions["players"].items()):
        rows = np.frombuffer(track["n"], dtype=np.int64) - offset
        column = slot_of[(team, person)]
        if np.isfinite(px[rows, column]).any():
            raise ValueError("player {} has two frames at one N".format(person))
        px[rows, column] = norm_x(track["x"])
        py[rows, column] = norm_y(track["y"])
        ps[rows, column] = np.asarray(track["s"], dtype=np.float64) / KMH_PER_MS

    bx = np.full(n_frames, np.nan)
    by = np.full(n_frames, np.nan)
    labels = np.full(n_frames, X, dtype=np.int8)
    unknown, kickoff_video = {}, {}
    possession_code = {value: ROLE_CODE[role] for value, role in POSSESSION_ROLE.items()}
    for half in sorted(spans):
        track = positions["ball"][half]
        rows = np.frombuffer(track["n"], dtype=np.int64) - offset
        if np.unique(rows).size != rows.size:
            raise ValueError("half {}: two ball frames at one N".format(half))
        bx[rows] = norm_x(track["x"])
        by[rows] = norm_y(track["y"])
        codes = np.array([_label(status, possession, possession_code)
                          for status, possession in zip(track["status"], track["possession"])],
                         dtype=np.int8)
        labels[rows] = codes
        unknown[half] = int(np.sum(codes == X))
        alive = rows[np.isin(codes, (H, A))]
        kickoff_video[half] = int(alive.min()) if alive.size else int(rows.min())
    return {
        "match_id": matchinfo["match_id"], "stem": stem_of(matchinfo["match_id"]),
        "title": matchinfo["title"], "fps": fps, "pitch": (length, width), "halves": spans,
        "offset": offset, "n_frames": n_frames, "kickoff_video": kickoff_video,
        "teams": [home, guest], "slots": slots, "px": px, "py": py, "ps": ps, "bx": bx,
        "by": by, "labels": labels, "unknown_label_frames": unknown,
    }


def _label(status, possession, possession_code):
    """The reference code of one ball frame (module docstring, Labels)."""
    if status == STATUS_DEAD:
        return D
    if status == STATUS_ALIVE:
        return possession_code.get(possession, X)
    return X


def coverage(bundle):
    """{half: share of the half's frames with a label other than X} and the graded frame count."""
    out = {}
    for half, (first, last) in sorted(bundle["halves"].items()):
        window = bundle["labels"][first - bundle["offset"]:last - bundle["offset"] + 1]
        out[half] = float(np.mean(window != X))
    return out


def games_fields(bundle, rel_dir):
    """The games.py entry of the bundle as a dict (the field names of games.GAMES)."""
    kick = bundle["kickoff_video"]
    first_block = bundle["halves"][1][0] - bundle["offset"]
    fields = {
        "n_video_frames": int(bundle["n_frames"]),
        "fps": int(bundle["fps"]),
        "offsets": {half: int(bundle["offset"]) for half in sorted(bundle["halves"])},
        "halves": {half: (int(a), int(b)) for half, (a, b) in sorted(bundle["halves"].items())},
        "kickoff": anchor(kick[1]),
        "kickoff_2nd": anchor(kick[2]),
        "pitch": (float(bundle["pitch"][0]), float(bundle["pitch"][1])),
        "assets": {
            "atd_meta": "{}/{}_atd.xml".format(rel_dir, bundle["stem"]),
            "atd_raw": "{}/{}_atd.txt".format(rel_dir, bundle["stem"]),
        },
    }
    if kick[1] > first_block:
        fields["kickoff_from_anchor"] = True
    return fields


def entry_text(bundle, fields):
    """The games.py entry as source text, with a comment that names the source of each value."""
    home, guest = bundle["teams"]
    halves = fields["halves"]
    lines = [
        '    "{}": {{'.format(bundle["stem"]),
        "        # {} (home, H) v {} (guest, A). {}.".format(home["name"], guest["name"],
                                                            bundle["match_id"]),
        '        "n_video_frames": {},  # last N of half 2 - offset + 1'.format(
            fields["n_video_frames"]),
        '        "fps": {},  # step of the T times of the ball frames'.format(fields["fps"]),
        '        "offsets": {{1: {0}, 2: {0}}},  # first N of half 1'.format(
            fields["offsets"][1]),
        '        # First N of each ball FrameSet, plus TotalTimeFirstHalf / TotalTimeSecondHalf.',
        '        "halves": {{1: ({}, {}), 2: ({}, {})}},'.format(
            halves[1][0], halves[1][1], halves[2][0], halves[2][1]),
        '        # First frame of each half with BallStatus 1.',
        '        "kickoff": {{"minute": {minute}, "second": {second}, "frame": {frame}}},'.format(
            **fields["kickoff"]),
        '        "kickoff_2nd": {{"minute": {minute}, "second": {second}, "frame": {frame}}},'
        .format(**fields["kickoff_2nd"]),
    ]
    if fields.get("kickoff_from_anchor"):
        lines.append('        "kickoff_from_anchor": True,  # half 1 does not start alive')
    lines += [
        '        "pitch": ({}, {}),  # PitchX, PitchY of the match information'.format(
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
    home, guest = bundle["teams"]
    length, width = bundle["pitch"]
    out = ['<?xml version="1.0" encoding="utf-8"?>', "<main>", "  <Metadata>",
           "    <GlobalConfig>",
           "      <ProviderName>Sportec Solutions (converted by tools/dfl_bundle.py)"
           "</ProviderName>",
           "      <FrameRate>{}</FrameRate>".format(bundle["fps"]),
           "      <ProviderGlobalParameters>"]
    for half, (first, last) in sorted(bundle["halves"].items()):
        for name, value in (("half{}_first_frame_id".format(half), first - bundle["offset"] + 1),
                            ("half{}_last_frame_id".format(half), last - bundle["offset"] + 1)):
            out.append("        <ProviderParameter><Name>{}</Name><Value>{}</Value>"
                       "</ProviderParameter>".format(name, value))
    out += ["      </ProviderGlobalParameters>", "    </GlobalConfig>",
            "    <Sessions>",
            '      <Session id="{}">'.format(_xml_text(bundle["match_id"])),
            "        <FieldSize><Width>{}</Width><Height>{}</Height></FieldSize>".format(
                length, width),
            "      </Session>", "    </Sessions>", "    <Teams>"]
    for team in (home, guest):
        out.append('      <Team id="{}"><Name>{}</Name></Team>'.format(
            _xml_text(team["team_id"]), _xml_text(team["name"])))
    out += ["    </Teams>", "    <Players>"]
    for team_id, person in bundle["slots"]:
        out.append('      <Player id="{}" teamId="{}"/>'.format(_xml_text(person),
                                                               _xml_text(team_id)))
    out += ["    </Players>", "    <Devices>", '      <Device id="device1">', "        <Sensors>",
            '          <Sensor id="position">', "            <Channels>"]
    units = {"x": "normalized", "y": "normalized", "s": "meters_per_second"}
    for channel in PLAYER_CHANNELS:
        out.append('              <Channel id="{0}"><Unit>{1}</Unit></Channel>'.format(
            channel, units[channel]))
    out += ["            </Channels>", "          </Sensor>", "        </Sensors>",
            "      </Device>", "    </Devices>", "    <PlayerChannels>"]
    for slot, (_, person) in enumerate(bundle["slots"]):
        for channel in PLAYER_CHANNELS:
            out.append('      <PlayerChannel channelId="{}" id="p{}_{}" playerId="{}"/>'.format(
                channel, slot + 1, channel, _xml_text(person)))
    out += ["    </PlayerChannels>", "  </Metadata>", "  <DataFormatSpecifications>",
            '    <DataFormatSpecification endFrame="{}" separator=":" startFrame="1">'.format(
                bundle["n_frames"])]
    for slot in range(len(bundle["slots"])):
        refs = "".join('<PlayerChannelRef playerChannelId="p{}_{}"/>'.format(slot + 1, channel)
                       for channel in PLAYER_CHANNELS)
        out.append("      <SplitRegister>{}</SplitRegister>".format(refs))
    out.append("      <BallChannelRefs>{}</BallChannelRefs>".format(
        "".join('<BallChannelRef channelId="{}"/>'.format(c) for c in BALL_CHANNELS)))
    out += ["    </DataFormatSpecification>", "  </DataFormatSpecifications>", "</main>", ""]
    return "\n".join(out)


def atd_lines(bundle, chunk=10000):
    """The raw ATD lines of the bundle, one string for each video frame, in frame order.

    A generator. It formats `chunk` frames at a time, so that memory stays small.
    """
    n_slots = len(bundle["slots"])
    player = ",".join([POSITION_FORMAT, POSITION_FORMAT, SPEED_FORMAT])
    template = "{}:" + ";".join([player] * n_slots) + ":" + ",".join(
        [POSITION_FORMAT] * len(BALL_CHANNELS)) + "\n"
    for start in range(0, bundle["n_frames"], chunk):
        stop = min(start + chunk, bundle["n_frames"])
        players = np.stack([bundle["px"][start:stop], bundle["py"][start:stop],
                            bundle["ps"][start:stop]], axis=2).reshape(stop - start, 3 * n_slots)
        ball = np.stack([bundle["bx"][start:stop], bundle["by"][start:stop]], axis=1)
        table = np.concatenate([players, ball], axis=1).tolist()
        for i, row in enumerate(table):
            yield template.format(start + i + 1, *row).replace("nan", "NaN")


def sha256(path):
    """Hex sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bundle(bundle, out_dir, rel_dir, inputs=None):
    """Write the four files of the bundle into out_dir. Return the build summary dict.

    `inputs` is {name: path} of the DFL files, for the checksums in the summary.
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
    fields = games_fields(bundle, rel_dir)
    home, guest = bundle["teams"]
    summary = {
        "match_id": bundle["match_id"], "stem": stem, "title": bundle["title"],
        "home": {"team_id": home["team_id"], "name": home["name"]},
        "guest": {"team_id": guest["team_id"], "name": guest["name"]},
        "games_entry": fields,
        "slots": {"home": sum(1 for t, _ in bundle["slots"] if t == home["team_id"]),
                  "guest": sum(1 for t, _ in bundle["slots"] if t == guest["team_id"])},
        "label_coverage": coverage(bundle),
        "unknown_label_frames": bundle["unknown_label_frames"],
        "inputs": {name: {"file": os.path.basename(path), "bytes": os.path.getsize(path),
                          "sha256": sha256(path)}
                   for name, path in sorted((inputs or {}).items())},
        "outputs": {name: {"file": os.path.basename(path), "sha256": sha256(path)}
                    for name, path in sorted(paths.items()) if name != "build"},
    }
    with open(paths["build"], "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


# -- audit (open matches only) -------------------------------------------------------------------

def _speed(x, y, fps):
    """Per-frame speed in m/s from positions in metres (NaN at the first frame and next to NaN)."""
    speed = np.full(x.shape, np.nan)
    speed[1:] = np.hypot(np.diff(x), np.diff(y)) * fps
    return speed


def _runs(mask):
    """[(start, stop)] half-open runs of True in a bool array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _pct(values, q):
    """Percentiles of a list as floats, or None for an empty list."""
    return [float(v) for v in np.percentile(values, q)] if len(values) else None


def audit(game, reference_path, matchinfo_path=None):
    """Summary statistics of one DFL bundle and its reference, as a dict (no per-frame rows).

    The bundle is read through engine.data.load_tactical(game), the engine's own data layer.
    Sections: kickoff, transitions, team_side, dead_ball, reference, metadata.
    """
    from engine import data
    entry = games.GAMES[game]
    tactical = data.load_tactical(game)
    length, width = data.pitch_m(game)
    fps = float(entry["fps"])
    codes = read_reference(reference_path, tactical["n_frames"])
    bx, by = tactical["bx"] * length, tactical["by"] * width
    px, py = tactical["px"] * length, tactical["py"] * width
    team_of_slot = tactical["team_of_slot"]
    speed = _speed(bx.astype(np.float64), by.astype(np.float64), fps)
    blocks = data.graded_blocks(game)
    out = {"game": game, "team_names": list(tactical["team_names"]),
           "n_frames": int(tactical["n_frames"]), "blocks": blocks}

    # Kickoffs: the first alive frame of each block against the ball near the centre spot. The
    # search window is the first 10 s of the block.
    kick = []
    for half, first, last in blocks:
        stop = min(first + int(10 * fps), last + 1)
        centre = np.hypot(bx[first:stop] - length / 2, by[first:stop] - width / 2)
        alive_at = np.flatnonzero(np.isin(codes[first:stop], (H, A)))
        k = int(alive_at[0]) if alive_at.size else None
        nearest = int(np.nanargmin(centre))
        kick.append({"half": half, "label_first_frame": events_gt.LETTER[int(codes[first])],
                     "first_alive_frame": k,
                     "first_alive_team": events_gt.LETTER[int(codes[first + k])]
                     if k is not None else None,
                     "centre_distance_at_alive_m": float(centre[k]) if k is not None else None,
                     "nearest_centre_frame": nearest,
                     "nearest_centre_m": float(centre[nearest]),
                     "speed_at_alive_ms": float(speed[first + k])
                     if k is not None and first + k > 0 else None})
    out["kickoff"] = kick

    # Ball-status transitions: where the ball is and what it does around them.
    runs = pm.valid_runs(codes)
    onsets, restarts = [], []
    for frame, before, after in pm.transitions(codes, runs):
        if after == D and before in (H, A):
            onsets.append(frame)
        elif before == D and after in (H, A):
            restarts.append(frame)
    out["transitions"] = _transition_stats(onsets, restarts, bx, by, speed, length, width, fps)

    # Team side: mean x of each team in each half, and the nearest player at alive frames.
    side = []
    for half, first, last in blocks:
        row = {"half": half}
        for team, letter in ((0, "H"), (1, "A")):
            xs = px[first:last + 1, team_of_slot == team]
            row["mean_x_{}".format(letter)] = float(np.nanmean(xs))
        row["H_defends"] = "x=0" if row["mean_x_H"] < row["mean_x_A"] else "x=length"
        side.append(row)
    out["team_side"] = {"halves": side, "nearest": _nearest_agreement(codes, bx, by, px, py,
                                                                      team_of_slot)}

    # Dead and missing ball.
    visible = np.isfinite(bx)
    graded = codes != X
    dead, alive = codes == D, np.isin(codes, (H, A))
    still = np.zeros_like(visible)
    still[1:] = (np.hypot(np.diff(bx), np.diff(by)) == 0) & visible[1:] & visible[:-1]
    dead_segments = [(a, b) for a, b, k in pm.segments(codes, runs) if k == D]
    seg_still = [float(np.mean(speed[a + 1:b] < 0.1)) for a, b in dead_segments if b - a > 1]
    out["dead_ball"] = {
        "visible_share_graded": float(np.mean(visible[graded])),
        "visible_share_dead": float(np.mean(visible[dead])),
        "visible_share_alive": float(np.mean(visible[alive])),
        "dead_speed_below_0_1": float(np.mean(speed[dead & visible] < 0.1)),
        "dead_speed_below_0_5": float(np.mean(speed[dead & visible] < 0.5)),
        "alive_speed_below_0_1": float(np.mean(speed[alive & visible] < 0.1)),
        "dead_exactly_frozen": float(np.mean(still[dead])),
        "alive_exactly_frozen": float(np.mean(still[alive])),
        "dead_segments_mostly_still": float(np.mean(np.array(seg_still) >= 0.9))
        if seg_still else None,
        "dead_outside_pitch": float(np.mean(_outside(bx, by, length, width)[dead & visible])),
        "missing_gaps_over_2s": sum(1 for a, b in _runs(graded & ~visible) if b - a > 2 * fps),
    }

    out["reference"] = _reference_stats(codes, runs, fps)
    if matchinfo_path:
        info = read_matchinfo(matchinfo_path)
        playing = sum(info["playing_ms"].values()) * fps / 1000.0
        total = sum(info["durations_ms"].values()) * fps / 1000.0
        out["metadata"] = {"playing_share": playing / total if total else None,
                           "alive_share_reference": float(np.mean(alive[graded]))}
    return out


def read_reference(path, n_frames):
    """A reference CSV as label codes of length n_frames (ValueError otherwise)."""
    table = pm.read_label_csv(path)
    problem = pm.validate_submission(table, n_frames)
    if problem:
        raise ValueError("{}: {}".format(path, problem))
    return pm.label_codes(table)


def _outside(bx, by, length, width):
    """True where the ball is beyond a pitch line (finite positions only)."""
    with np.errstate(invalid="ignore"):
        return (bx < 0) | (bx > length) | (by < 0) | (by > width)


def _transition_stats(onsets, restarts, bx, by, speed, length, width, fps):
    """Ball facts at the D onsets and the restarts of the reference."""
    outside = _outside(bx, by, length, width)
    window = int(fps)  # one second

    def mean_speed(a, b):
        part = speed[max(a, 0):max(b, 0)]
        part = part[np.isfinite(part)]
        return float(np.mean(part)) if part.size else np.nan

    onset_outside = [bool(outside[f]) for f in onsets]
    cross = []
    for f in onsets:
        if not outside[f]:
            continue
        lo = max(f - 10 * window, 0)
        inside = np.flatnonzero(~outside[lo:f] & np.isfinite(bx[lo:f]))
        if inside.size:
            cross.append(f - (lo + int(inside[-1]) + 1))
    before = [mean_speed(f - window, f) for f in onsets]
    after = [mean_speed(f, f + window) for f in onsets]
    # At a restart the frozen dead ball jumps to the live ball. Measure the jump at the restart
    # frame apart from the movement after it.
    r_before = [mean_speed(f - window, f) for f in restarts]
    r_after = [mean_speed(f + 1, f + 11) for f in restarts]
    jump = [float(np.hypot(bx[f] - bx[f - 1], by[f] - by[f - 1])) for f in restarts]
    first_fast = []
    for f in restarts:
        fast = np.flatnonzero(speed[f + 1:f + 1 + window] > 3.0)
        if fast.size:
            first_fast.append(int(fast[0]) + 1)
    return {
        "onsets": len(onsets), "restarts": len(restarts),
        "onset_ball_outside_share": float(np.mean(onset_outside)) if onsets else None,
        "onset_frames_after_crossing_p25_50_75": _pct(cross, [25, 50, 75]),
        "onset_speed_before_after_median": [float(np.nanmedian(before)), float(np.nanmedian(after))]
        if onsets else None,
        "restart_speed_before_median": float(np.nanmedian(r_before)) if restarts else None,
        "restart_jump_m_p25_50_75": _pct(jump, [25, 50, 75]),
        "restart_jump_over_1m_share": float(np.mean(np.array(jump) > 1.0)) if restarts else None,
        "restart_ball_inside_share": float(np.mean([not outside[f] for f in restarts]))
        if restarts else None,
        "restart_speed_next_10_median": float(np.nanmedian(r_after)) if restarts else None,
        "restart_first_over_3ms_p25_50_75": _pct(first_fast, [25, 50, 75]),
        "restart_with_fast_ball_within_1s": len(first_fast),
    }


def _nearest_agreement(codes, bx, by, px, py, team_of_slot):
    """Share of alive frames where the nearest player's team is the reference team, by distance."""
    alive = np.flatnonzero(np.isin(codes, (H, A)) & np.isfinite(bx))
    dist = np.hypot(px[alive] - bx[alive, None], py[alive] - by[alive, None])
    dist = np.where(np.isfinite(dist), dist, np.inf)
    nearest = np.argmin(dist, axis=1)
    best = dist[np.arange(alive.size), nearest]
    agree = team_of_slot[nearest] == codes[alive]
    out = {}
    for radius in (0.5, 1.0, 1.5):
        mask = best <= radius
        out["within_{}m".format(radius)] = {"frames": int(mask.sum()),
                                            "agree": float(np.mean(agree[mask]))
                                            if mask.any() else None}
    return out


def _reference_stats(codes, runs, fps):
    """Possessions, durations, short spells, D segments and transitions of a reference."""
    stats = events_gt.reference_stats(codes, fps)
    spells = pm.possession_spells(codes, runs)
    seconds = np.array([(b - a) / fps for a, b, _ in spells])
    segs = pm.segments(codes, runs)
    flicker = 0
    for i in range(1, len(segs) - 1):
        a, b, k = segs[i]
        if k in (H, A) and (b - a) < fps and segs[i - 1][2] == segs[i + 1][2] != k \
                and segs[i - 1][2] in (H, A):
            flicker += 1
    d_lengths = np.array([(b - a) / fps for a, b, k in segs if k == D])
    kinds = {"H<->A": 0, "alive->D": 0, "D->alive": 0}
    for _, before, after in pm.transitions(codes, runs):
        if before in (H, A) and after in (H, A):
            kinds["H<->A"] += 1
        elif after == D:
            kinds["alive->D"] += 1
        elif before == D:
            kinds["D->alive"] += 1
    stats.update({
        "median_duration_s": float(np.median(seconds)) if seconds.size else None,
        "spells_under_0_4s": int(np.sum(seconds < 0.4)),
        "spells_under_1s": int(np.sum(seconds < 1.0)),
        "spells_under_2s": int(np.sum(seconds < 2.0)),
        "share_under_1s": float(np.mean(seconds < 1.0)) if seconds.size else None,
        "share_under_2s": float(np.mean(seconds < 2.0)) if seconds.size else None,
        "spells_over_60s": int(np.sum(seconds > 60.0)),
        "flicker_spells_under_1s": flicker,
        "d_segment_median_s": float(np.median(d_lengths)) if d_lengths.size else None,
        "d_segment_p90_s": float(np.percentile(d_lengths, 90)) if d_lengths.size else None,
        "d_segments_under_1s": int(np.sum(d_lengths < 1.0)),
        "transitions": kinds,
    })
    return stats


# -- command line --------------------------------------------------------------------------------

def _cmd_build(args):
    matchinfo = read_matchinfo(args.matchinfo)
    team_ids = {t["team_id"] for t in matchinfo["teams"]}
    positions = read_positions(args.positions, team_ids)
    bundle = build(matchinfo, positions)
    rel_dir = args.rel_dir or "dfl/{}".format(matchinfo["match_id"].split("-")[-1])
    summary = write_bundle(bundle, args.out_dir, rel_dir,
                           {"matchinfo": args.matchinfo, "positions": args.positions})
    fields = summary["games_entry"]
    print("{} {}: H {!r}, A {!r}{}".format(
        bundle["match_id"], bundle["stem"], summary["home"]["name"], summary["guest"]["name"],
        "  (SEALED)" if bundle["match_id"] in SEALED else ""))
    print("frames {}  fps {}  halves {}  offset {}".format(
        fields["n_video_frames"], fields["fps"], fields["halves"], fields["offsets"][1]))
    print("label coverage {}  slots {}".format(
        {h: round(v, 6) for h, v in summary["label_coverage"].items()}, summary["slots"]))
    print(entry_text(bundle, fields))
    print("wrote {}".format(args.out_dir))
    return 0


def _cmd_audit(args):
    entry = games.GAMES[args.game]
    meta = os.path.join(games.DATA_DIR, entry["assets"]["atd_meta"])
    build_path = meta.replace("_atd.xml", "_build.json")
    with open(build_path) as handle:
        built = json.load(handle)
    if built["match_id"] in SEALED and not args.allow_sealed:
        raise SystemExit("{} is sealed (docs/plan/calibration/SPLITS.md). Pass --allow-sealed "
                         "only at the final selection.".format(built["match_id"]))
    reference = args.reference or meta.replace("_atd.xml", "_reference.csv")
    report = audit(args.game, reference, args.matchinfo)
    report["match_id"] = built["match_id"]
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json:
        with open(args.json, "w") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="convert one DFL match")
    b.add_argument("--positions", required=True, help="DFL positions XML")
    b.add_argument("--matchinfo", required=True, help="DFL match information XML")
    b.add_argument("--out-dir", required=True, help="bundle folder (outside the repository)")
    b.add_argument("--rel-dir", default=None,
                   help="bundle folder relative to MTAH_DATA_DIR, for the games.py assets "
                        "(default: dfl/<id>)")
    a = sub.add_parser("audit", help="summary statistics of one built, open match")
    a.add_argument("--game", required=True, choices=sorted(games.GAMES))
    a.add_argument("--reference", default=None, help="reference CSV (default: next to the bundle)")
    a.add_argument("--matchinfo", default=None,
                   help="DFL match information XML, for the playing-time cross-check")
    a.add_argument("--json", default=None, help="also write the report to this file")
    a.add_argument("--allow-sealed", action="store_true",
                   help="audit a sealed match (the final selection only)")
    args = parser.parse_args(argv)
    return _cmd_build(args) if args.command == "build" else _cmd_audit(args)


if __name__ == "__main__":
    sys.exit(main())
