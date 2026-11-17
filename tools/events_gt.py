"""Frame-level H/A/D/X reference from Metrica events (JSON or CSV). Not part of the submission.

    python tools/events_gt.py --game METRICA-S3 --events <events.json> --out <reference.csv>
    python tools/events_gt.py --game METRICA-S1 --events <RawEventsData.csv> --out <reference.csv>

The tool reads the hand-made events of another match and writes a `frame,label` CSV with one
row for each tactical video frame of the games.py entry. RULES.md allows hand-labelled data
from other matches for validation. Never use this tool on the competition match.

Two event formats. Sample Game 3 ships a JSON file (load_events). Sample Games 1 and 2 ship a
CSV file (load_events_csv). `--format auto` picks the reader from the file extension. Both
readers give the same normalised events, and the rules below apply to both without a change.

The reference is a proxy. The organisers derive their ground truth from Metrica's corrected
tracking "owner" and "dead" fields (CHALLENGE.md). This tool derives labels from the event
data instead, with the rules below. Where the two differ, the reference is wrong, not the
organisers' definition.

Frames. An event frame is an ATD `frame_id`. The tool maps it to a video frame with the offset
of its period in games.py: `video_frame = frame_id - offset`. For METRICA-S3 the offset is 1,
the same mapping as engine/data.py. The event x, y of a PASS sits 0.05 m (median) from the
tracking ball at that video frame and 0.2 m one frame either side.

Graded block. Each period has one block. The block starts at the start frame of the first
SET PIECE with the KICK OFF subtype of the period. The block ends at the end frame of the last
event of the period, the event with the END HALF subtype. The end frame is the whistle, so the
block is half-open: the end frame itself is X. Every frame outside the blocks is X.

Dead ball. A stopping event opens `D` at its end frame. `D` closes at the start frame of the
next restart. These events stop play:

- BALL OUT.
- SHOT with the OUT or the GOAL subtype.
- FAULT RECEIVED, and CHALLENGE with the FAULT subtype.
- Any event with the OFFSIDE or the HAND BALL subtype (BALL LOST, PASS).

These events do not stop play, because the event stream shows play that continues after them:
RECOVERY with the GOAL subtype, BALL LOST with the REFEREE HIT subtype, CHALLENGE with the
ADVANTAGE subtype, SHOT that hits the WOODWORK without OUT, and CARD. A CARD always sits inside
a stoppage that a foul already opened.

A restart is one of these events:

- A SET PIECE of any subtype.
- An event with the GOAL KICK subtype. The sample tags goal kicks as PASS or BALL LOST with
  this subtype, never as SET PIECE.
- The first owner event after a stopping event, if it starts PAUSE_S seconds or more after the
  `D` onset. This catches a goal kick without the GOAL KICK subtype.

A restart that follows no stopping event since the previous restart opens `D` at the end frame
of the event before it. A set piece always follows a stoppage. In the sample this happens three
times: two set pieces with the RETAKEN subtype and one foul tagged only as BALL LOST.

Owner while alive. The owner is the team of the last owner event: PASS, CARRY, RECOVERY,
SET PIECE, SHOT, or a restart. The owner changes at the start frame of an owner event of the
other team. BALL LOST and CHALLENGE do not change the owner, because a loose or contested ball
stays with the last team (CHALLENGE.md). The owner is a proxy for the corrected tracking
"owner" field. It is not that field.

Teams. The first <Team> of the ATD metadata is `H` (engine.data._atd_meta). The tool matches the
event team name to the team names that engine.data.load_tactical() returns. For METRICA-S3,
"Team A" is `H` and "Team B" is `A`. For METRICA-S1 and METRICA-S2, tools/metrica_bundle.py
names the teams "Home" and "Away", the Team values of the events CSV. So "Home" is `H`.

The CSV format, against the JSON format of Sample Game 3:

- The CSV holds one row per event with the columns Team, Type, Subtype, Period, Start Frame,
  End Frame, and more that the tool does not read. The JSON holds the same fields as objects.
- The Subtype column joins the subtype names with "-", for example "HEAD-ON TARGET-GOAL".
  The reader splits it at "-". No subtype name of the three samples contains a "-".
- The CSV has no event id and no team id. The team is the Team value only.
- Sample Games 1 and 2 have no CARRY event. Sample Game 3 has them. In the CSV samples the
  owner therefore changes only at a PASS, RECOVERY, SET PIECE or SHOT of the other team, or
  at a restart.
- The CSV gives the frames of the event only as Start Frame and End Frame. One event of
  Sample Game 1 has End Frame 0 before Start Frame 1, the opening kick-off. The reader takes
  the larger value as the end, as the JSON reader does.
- In Sample Games 1 and 2 the second period has no END HALF event. The block then ends at the
  end frame of the last event of the period, the fallback of period_block. In both samples that
  event is a BALL OUT.

Deterministic: events in file order, ties at one frame broken by file order. Python 3.9.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import games  # noqa: E402
import possession_metrics as pm  # noqa: E402

H, A, D, X = 0, 1, 2, 3
LETTER = {H: "H", A: "A", D: "D", X: "X"}

# Event types that name the team on the ball (the brief, from CHALLENGE.md: possession changes
# when a player of the other team plays the ball). BALL LOST and CHALLENGE are not here.
OWNER_TYPES = ("PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT")
# Restarts. Sample Game 3 tags 11 goal kicks with the GOAL KICK subtype on PASS or BALL LOST.
RESTART_TYPE = "SET PIECE"
RESTART_SUBTYPES = ("GOAL KICK",)
KICK_OFF = "KICK OFF"
END_HALF = "END HALF"
# Stopping events (Laws of the Game: ball out, goal, foul, offside, handball).
STOP_TYPES = ("BALL OUT", "FAULT RECEIVED")
STOP_SHOT_SUBTYPES = ("OUT", "GOAL")            # on a SHOT only
STOP_CHALLENGE_SUBTYPES = ("FAULT",)            # on a CHALLENGE only
STOP_ANY_SUBTYPES = ("OFFSIDE", "HAND BALL")    # on any event type
# A restart needs the ball placed, so it comes seconds after the stop. Sample Game 3: after a
# stopping event the next owner event starts within 0.92 s when play goes on (advantage, a late
# offside flag, ties at one frame) and 5.3 s or more later at a goal kick. 2 s sits between.
PAUSE_S = 2.0
# The columns of a Metrica events CSV (Sample Games 1 and 2) that the CSV reader needs.
CSV_COLUMNS = ("Team", "Type", "Subtype", "Period", "Start Frame", "End Frame")
# The separator of the subtype names in the Subtype column of the events CSV.
CSV_SUBTYPE_SEPARATOR = "-"
FORMATS = ("auto", "json", "csv")


# -- events ---------------------------------------------------------------------------------------

def subtype_names(event):
    """The subtype names of one raw Metrica event, as a tuple (empty when there is none)."""
    value = event.get("subtypes")
    if value is None:
        return ()
    if isinstance(value, dict):
        value = [value]
    return tuple(str(item["name"]) for item in value)


def normalise(raw_events):
    """Raw Metrica events -> list of dicts with the keys this module reads.

    Keys: order (position in the file), team (name), type, subtypes (tuple), start, end
    (frame ids), period. The list keeps the file order. Raises ValueError on an event without
    a team, a period or frames.
    """
    out = []
    for order, event in enumerate(raw_events):
        team = (event.get("team") or {}).get("name")
        start = (event.get("start") or {}).get("frame")
        end = (event.get("end") or {}).get("frame")
        period = event.get("period")
        if team is None or start is None or end is None or period is None:
            raise ValueError("event {} has no team, period or frames".format(
                event.get("index", order)))
        out.append({"order": order, "team": str(team), "type": str(event["type"]["name"]),
                    "subtypes": subtype_names(event), "start": int(start),
                    "end": int(max(start, end)), "period": int(period)})
    return out


def load_events(path):
    """The events of a Metrica events JSON ({"data": [...]} or a bare list), normalised."""
    with open(path) as handle:
        payload = json.load(handle)
    raw = payload["data"] if isinstance(payload, dict) else payload
    return normalise(raw)


def csv_subtypes(value):
    """The subtype names of the Subtype column of an events CSV, as a tuple."""
    parts = (part.strip() for part in (value or "").split(CSV_SUBTYPE_SEPARATOR))
    return tuple(part for part in parts if part)


def normalise_csv(rows):
    """Rows of a Metrica events CSV (dicts by column name) -> the dicts of normalise().

    Keys: order (row position), team (the Team value), type, subtypes (tuple), start, end
    (frame ids), period. The end is the larger of Start Frame and End Frame. Raises ValueError
    on a row without a team, a period or frames.
    """
    out = []
    for order, row in enumerate(rows):
        team = (row.get("Team") or "").strip()
        values = [(row.get(name) or "").strip() for name in ("Start Frame", "End Frame",
                                                             "Period")]
        if not team or not all(values):
            raise ValueError("event row {} has no team, period or frames".format(order))
        start, end, period = (int(float(v)) for v in values)
        out.append({"order": order, "team": team, "type": (row.get("Type") or "").strip(),
                    "subtypes": csv_subtypes(row.get("Subtype")), "start": start,
                    "end": max(start, end), "period": period})
    return out


def load_events_csv(path):
    """The events of a Metrica events CSV (Sample Games 1 and 2), normalised."""
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in CSV_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("{}: no column {}".format(path, missing))
        return normalise_csv(list(reader))


def load_any(path, fmt="auto"):
    """The events of `path` with the reader of `fmt`. "auto" reads a .csv file as CSV."""
    if fmt not in FORMATS:
        raise ValueError("unknown event format {!r}; have {}".format(fmt, list(FORMATS)))
    if fmt == "auto":
        fmt = "csv" if path.lower().endswith(".csv") else "json"
    return load_events_csv(path) if fmt == "csv" else load_events(path)


def is_owner(event):
    """True if the event names the team on the ball."""
    return event["type"] in OWNER_TYPES or is_tagged_restart(event)


def is_tagged_restart(event):
    """True for a SET PIECE or an event with a restart subtype (GOAL KICK)."""
    return (event["type"] == RESTART_TYPE
            or any(s in RESTART_SUBTYPES for s in event["subtypes"]))


def is_stop(event):
    """True if the event stops play (see the module docstring for the list)."""
    kind, subs = event["type"], event["subtypes"]
    if kind in STOP_TYPES:
        return True
    if kind == "SHOT" and any(s in STOP_SHOT_SUBTYPES for s in subs):
        return True
    if kind == "CHALLENGE" and any(s in STOP_CHALLENGE_SUBTYPES for s in subs):
        return True
    return any(s in STOP_ANY_SUBTYPES for s in subs)


# -- derivation -----------------------------------------------------------------------------------

def to_video(events, offsets):
    """Copy of the events with start and end as video frames: frame_id - offsets[period]."""
    out = []
    for event in events:
        shift = int(offsets[event["period"]])
        copy = dict(event)
        copy["start"] = event["start"] - shift
        copy["end"] = event["end"] - shift
        out.append(copy)
    return out


def period_block(events):
    """(start, stop) half-open block of one period's events (video frames).

    Start: the first KICK OFF set piece. Stop: the end frame of the last END HALF event, or of
    the last event when there is no END HALF. Raises ValueError without a kick-off.
    """
    kicks = [e["start"] for e in events
             if e["type"] == RESTART_TYPE and KICK_OFF in e["subtypes"]]
    if not kicks:
        raise ValueError("period {} has no KICK OFF set piece".format(events[0]["period"]))
    ends = [e["end"] for e in events if END_HALF in e["subtypes"]]
    stop = max(ends) if ends else max(e["end"] for e in events)
    return min(kicks), stop


def restarts_and_stops(events, fps):
    """(restarts, onsets, kinds) of one period: lists of video frames and onset kinds.

    `restarts` holds the start frames of the restarts. `onsets` holds the frames where `D`
    opens, with `kinds` naming the rule for each ("stop" or "no_stop"). The opening kick-off
    of the period is neither.
    """
    pause = int(round(PAUSE_S * fps))
    block_start = period_block(events)[0]
    restarts, onsets, kinds = [], [], []
    for i, event in enumerate(events):
        if is_stop(event) and END_HALF not in event["subtypes"]:
            onset = event["end"]
            onsets.append(onset)
            kinds.append("stop")
            following = [e for e in events[i + 1:] if is_owner(e) and e["start"] >= onset]
            if following and not is_tagged_restart(following[0]) \
                    and following[0]["start"] - onset >= pause:
                restarts.append(following[0]["start"])
    tagged = [(i, e) for i, e in enumerate(events)
              if is_tagged_restart(e) and e["start"] > block_start]
    restarts += [e["start"] for _, e in tagged]
    restarts = sorted(set(restarts))
    for i, event in tagged:
        previous = [r for r in restarts if r < event["start"]]
        since = previous[-1] if previous else block_start
        if any(since < o <= event["start"] for o in onsets):
            continue
        before = [e["end"] for e in events[:i] if e["start"] < event["start"]]
        if before and max(before) < event["start"]:
            onsets.append(max(before))
            kinds.append("no_stop")
    return restarts, onsets, kinds


def owner_codes(events, home_team, n_frames):
    """int8 codes H/A from the owner events, X before the first one (video frames)."""
    codes = np.full(n_frames, X, dtype=np.int8)
    changes = [(e["start"], e["order"], H if e["team"] == home_team else A)
               for e in events if is_owner(e)]
    changes.sort()
    for k, (frame, _, code) in enumerate(changes):
        stop = changes[k + 1][0] if k + 1 < len(changes) else n_frames
        lo, hi = max(0, frame), min(n_frames, stop)
        if lo < hi:
            codes[lo:hi] = code
    return codes


def derive(events, home_team, n_frames, offsets, fps):
    """(codes, info): the reference as int8 codes (H=0 A=1 D=2 X=3) and a summary dict.

    `events` are normalised events (load_events) with frame ids. `home_team` is the event team
    name that maps to `H`. Every other team name maps to `A`. `offsets` is {period: offset}
    from games.py. The summary holds the blocks, the D intervals and the rule counts.
    """
    events = to_video(events, offsets)
    codes = np.full(n_frames, X, dtype=np.int8)
    info = {"blocks": [], "dead": [], "onset_kinds": {"stop": 0, "no_stop": 0},
            "pause_restarts": 0, "unknown_owner_frames": 0}
    for period in sorted(set(e["period"] for e in events)):
        part = [e for e in events if e["period"] == period]
        start, stop = period_block(part)
        start, stop = max(0, start), min(n_frames, stop)
        info["blocks"].append((period, start, stop))
        owner = owner_codes(part, home_team, n_frames)
        restarts, onsets, kinds = restarts_and_stops(part, fps)
        tagged = set(e["start"] for e in part if is_tagged_restart(e))
        info["pause_restarts"] += sum(1 for r in restarts if r not in tagged)
        for kind in kinds:
            info["onset_kinds"][kind] += 1
        dead = np.zeros(n_frames, dtype=bool)
        for onset in onsets:
            later = [r for r in restarts if r > onset]
            close = later[0] if later else stop
            lo, hi = max(start, onset), min(stop, close)
            if lo < hi:
                dead[lo:hi] = True
        window = owner[start:stop].copy()
        unknown = window == X
        info["unknown_owner_frames"] += int(unknown.sum())
        window[unknown] = D
        window[dead[start:stop]] = D
        codes[start:stop] = window
        edges = np.diff(np.concatenate(([0], dead[start:stop].astype(np.int8), [0])))
        info["dead"] += [(start + int(a), start + int(b)) for a, b in
                         zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
    return codes, info


def reference_stats(codes, fps):
    """Possession count, mean duration, D share and D segments of a label vector.

    The graded runs are the non-X runs of the codes themselves (possession_metrics.valid_runs).
    """
    codes = np.asarray(codes, dtype=np.int8)
    runs = pm.valid_runs(codes)
    spells = pm.possession_spells(codes, runs)
    graded = sum(b - a for a, b in runs)
    d_segments = [s for s in pm.segments(codes, runs) if s[2] == D]
    lengths = [b - a for a, b, _ in spells]
    return {
        "graded_frames": int(graded),
        "possessions": len(spells),
        "mean_duration_s": float(np.mean(lengths)) / fps if lengths else 0.0,
        "d_share": float(np.sum(codes == D)) / graded if graded else 0.0,
        "d_segments": len(d_segments),
        "home_share": pm.possession_stats(codes, runs)["home_share"],
    }


# -- I/O ------------------------------------------------------------------------------------------

def home_team_name(game, data_dir=None):
    """The team name that engine/data.py maps to H for this match (first <Team> of the XML)."""
    from engine import data
    return data.load_tactical(game, data_dir=data_dir)["team_names"][0]


def write_csv(path, codes):
    """Write `frame,label` rows, one per video frame, in frame order."""
    lines = ["frame,label\n"]
    lines += ["{},{}\n".format(i, LETTER[int(c)]) for i, c in enumerate(codes)]
    with open(path, "w") as handle:
        handle.writelines(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", required=True, choices=sorted(games.GAMES),
                        help="the games.py entry of the match")
    parser.add_argument("--events", required=True, help="Metrica events JSON or CSV")
    parser.add_argument("--format", default="auto", choices=FORMATS,
                        help="event file format (default: from the file extension)")
    parser.add_argument("--out", required=True, help="reference CSV to write")
    parser.add_argument("--home-team", default=None,
                        help="event team name for H (default: the first <Team> of the ATD XML)")
    parser.add_argument("--json", default=None, help="also write the summary to this JSON file")
    args = parser.parse_args(argv)

    entry = games.GAMES[args.game]
    events = load_any(args.events, args.format)
    home = args.home_team or home_team_name(args.game)
    names = sorted(set(e["team"] for e in events))
    if home not in names:
        raise ValueError("home team {!r} is not an event team {}".format(home, names))
    fps = float(entry["fps"])
    codes, info = derive(events, home, int(entry["n_video_frames"]), entry["offsets"], fps)
    write_csv(args.out, codes)
    stats = reference_stats(codes, fps)
    summary = {"game": args.game, "home_team": home, "teams": names, "stats": stats,
               "blocks": info["blocks"], "onset_kinds": info["onset_kinds"],
               "pause_restarts": info["pause_restarts"],
               "unknown_owner_frames": info["unknown_owner_frames"],
               "n_dead_intervals": len(info["dead"])}
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print("home team {!r} -> H, {} -> A".format(home, [n for n in names if n != home]))
    print("blocks {}".format(info["blocks"]))
    print("possessions {possessions}  mean {mean_duration_s:.1f} s  D share {d_share:.3f}  "
          "D segments {d_segments}  home share {home_share:.3f}".format(**stats))
    print("D onsets {}  pause restarts {}  unknown owner frames {}".format(
        info["onset_kinds"], info["pause_restarts"], info["unknown_owner_frames"]))
    print("wrote {} ({} rows)".format(args.out, len(codes)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
