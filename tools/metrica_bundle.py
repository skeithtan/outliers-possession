"""Convert a Metrica Sample Game 1 or 2 tracking pair to an ATD bundle. Not part of the submission.

    python tools/metrica_bundle.py --home <stem>_RawTrackingData_Home_Team.csv \
        --away <stem>_RawTrackingData_Away_Team.csv --stem Sample_Game_1 --out-dir DIR

The tool reads the two per-team tracking CSVs of a Metrica sample match and writes
DIR/<stem>_atd.txt and DIR/<stem>_atd.xml. The two files have the layout of the FIFA/EPTS ATD
bundle that engine/data.py reads for BAR-ATH and METRICA-S3. A games.py entry with an "assets"
field then loads the match through the data layer of the engine, unchanged (LEARNINGS G1).
Write the bundle outside the repository. The P9-A references use ~/mtah2027/refs/metrica/<stem>/.

Input. Each CSV has three header rows and one row per tracking frame:

- Row 1 names the team ("Home" or "Away") above the x column of each player.
- Row 2 gives the shirt number above the x column of each player.
- Row 3 names the columns: Period, Frame, Time [s], then an x and a y column for each player,
  then an x and a y column for the ball.
- The data rows hold normalised coordinates from 0 to 1. The origin is the top left corner and
  (0.5, 0.5) is the kick-off point (Metrica sample-data README). NaN marks a missing value.

Decisions. The tool takes each of these from the files and fails loudly if a file disagrees:

- Frame id. The ATD `frame_id` is the Frame column. The tool requires Frame to run 1..N without
  a gap in both files, and the two files to have the same Frame and Period columns. The games.py
  entry then declares the offset 1 for both periods: `video_frame = frame_id - 1`, as METRICA-S3.
- Periods. The tool requires each period to be one contiguous block of frames, in period order.
  The metadata records the first and last frame of each period as first_half_start,
  first_half_end, second_half_start and second_half_end, as the METRICA-S3 metadata does.
- Frame rate. The tool computes it from the Time [s] column and requires every row to have
  Time = Frame / fps.
- Teams. The home file gives the first <Team> of the metadata. engine.data._atd_meta maps the
  first <Team> to `H`, so the Metrica Home team is `H`. The team names are the row 1 labels,
  "Home" and "Away". These are the Team values of the events CSV, so tools/events_gt.py maps the
  events to the same sides.
- Slots. Each player column becomes one slot, home columns first, then away columns, in file
  order. A substitute keeps a slot of his own. A slot is NaN while its player is not on the
  pitch. One DataFormatSpecification covers every frame. The slot table therefore holds one
  team per slot by construction. The player id is the side letter and the shirt number, for
  example H11 and A25. The <Name> is the column name of row 3.
- Channels. Players and ball carry an x and a y channel only. The CSVs have no speed column, so
  the bundle has none. engine.tracks then takes the player speed from position differences
  (LEARNINGS G1, fix 7). The ATD of METRICA-S3 has the same two channels.
- Ball. Both files carry the ball. The tool requires the two ball columns to be identical,
  NaN included, and writes them once.
- Missing values. NaN in the CSV stays NaN on the raw line, the token "NaN" as in METRICA-S3.
  Values outside 0..1 (the ball or a player beyond a line) are kept as they are.
- Units. The tool converts the coordinates to metres when it reads them and back to normalised
  values when it writes them, with the pitch size of --pitch. The values are written with
  DECIMALS decimals. The input has at most DECIMALS decimals, so the round trip writes the input
  values back exactly. The tool checks this.
- Orientation. The tool does not flip or rotate an axis. The sample-data README and kloppy's
  Metrica readers give the same coordinate system (origin top left, y down) for the CSV format
  and for the EPTS format of METRICA-S3.

Deterministic: file order only, no clock, no randomness. numpy, pandas and the standard library.
Python 3.9.
"""

import argparse
import csv
import os
import sys
from xml.sax.saxutils import escape, quoteattr

import numpy as np
import pandas as pd

# The size of both sample pitches in metres (Metrica sample-data README: 105 x 68 m).
DEFAULT_PITCH = (105.0, 68.0)
# Decimals of the written coordinates. The sample CSVs have at most 5. The ATD of METRICA-S3
# also writes 5.
DECIMALS = 5
# The column names of row 3 that are not player columns.
META_COLUMNS = ("Period", "Frame", "Time [s]")
BALL_COLUMN = "Ball"
SIDES = (("Home", "H", "HOME"), ("Away", "A", "AWAY"))
NAN_TOKEN = "NaN"
PROVIDER = "Metrica Sports sample data, converted by tools/metrica_bundle.py"
HALF_PARAMETERS = {1: ("first_half_start", "first_half_end"),
                   2: ("second_half_start", "second_half_end")}


# -- reading --------------------------------------------------------------------------------------

def read_team_csv(path):
    """One per-team tracking CSV as a dict of arrays.

    Returns {"team": label of row 1, "numbers": shirt numbers, "names": column names of row 3,
    "period": int64 (N,), "frame": int64 (N,), "time": float64 (N,), "players": float64
    (N, P, 2) normalised x and y, "ball": float64 (N, 2)}. Raises ValueError when the header
    does not have the layout of the module docstring.
    """
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        rows = [next(reader) for _ in range(3)]
    teams, numbers, names = rows
    width = len(names)
    if tuple(names[:3]) != META_COLUMNS:
        raise ValueError("{}: row 3 starts with {}, expected {}".format(
            path, names[:3], list(META_COLUMNS)))
    if width < 5 or names[-2] != BALL_COLUMN or (width - 3) % 2:
        raise ValueError("{}: row 3 does not end with a Ball x and y column pair".format(path))
    n_players = (width - 5) // 2
    x_columns = [3 + 2 * k for k in range(n_players)]
    labels = sorted(set(teams[c] for c in x_columns))
    if len(labels) != 1 or not labels[0]:
        raise ValueError("{}: row 1 names teams {} above the player columns".format(path, labels))
    player_numbers = [numbers[c].strip() for c in x_columns]
    if any(not n for n in player_numbers) or len(set(player_numbers)) != n_players:
        raise ValueError("{}: the shirt numbers of row 2 are missing or repeated".format(path))
    values = pd.read_csv(path, skiprows=3, header=None, dtype=np.float64,
                         names=list(range(width))).to_numpy()
    return {
        "team": labels[0],
        "numbers": player_numbers,
        "names": [names[c].strip() for c in x_columns],
        "period": values[:, 0].astype(np.int64),
        "frame": values[:, 1].astype(np.int64),
        "time": values[:, 2],
        "players": values[:, 3:3 + 2 * n_players].reshape(len(values), n_players, 2),
        "ball": values[:, width - 2:],
    }


def to_metres(norm, pitch):
    """Normalised (..., 2) coordinates to metres: x times the length, y times the width."""
    return np.asarray(norm, dtype=np.float64) * np.array(pitch, dtype=np.float64)


def to_normalised(metres, pitch):
    """Metres (..., 2) to normalised coordinates, the inverse of to_metres()."""
    return np.asarray(metres, dtype=np.float64) / np.array(pitch, dtype=np.float64)


def _same(a, b):
    """True when two float arrays are equal, NaN at the same places included."""
    return a.shape == b.shape and bool(np.array_equal(np.isnan(a), np.isnan(b))) and bool(
        np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)]))


def check_pair(home, away):
    """Raise ValueError unless the two team files describe the same frames and the same ball."""
    if home["team"] != SIDES[0][0] or away["team"] != SIDES[1][0]:
        raise ValueError("team labels {!r} and {!r}, expected {!r} and {!r}".format(
            home["team"], away["team"], SIDES[0][0], SIDES[1][0]))
    n_frames = len(home["frame"])
    if not np.array_equal(home["frame"], np.arange(1, n_frames + 1)):
        raise ValueError("the Frame column is not 1..{} without a gap".format(n_frames))
    if not np.array_equal(home["frame"], away["frame"]):
        raise ValueError("the two files have different Frame columns")
    if not np.array_equal(home["period"], away["period"]):
        raise ValueError("the two files have different Period columns")
    if not _same(home["ball"], away["ball"]):
        raise ValueError("the two files have different ball columns")


def frame_rate(frame, time):
    """Frames per second from the Frame and Time [s] columns. Raises if they disagree."""
    steps = np.diff(time)
    fps = int(round(1.0 / float(np.median(steps))))
    if not np.allclose(time, frame / float(fps), atol=1e-6):
        raise ValueError("Time [s] is not Frame / {} on every row".format(fps))
    return fps


def period_blocks(period, frame):
    """{period: (first frame id, last frame id)}. Raises unless each period is one block."""
    changes = np.flatnonzero(np.diff(period)) + 1
    starts = np.concatenate(([0], changes))
    stops = np.concatenate((changes, [len(period)]))
    order = [int(period[s]) for s in starts]
    if order != sorted(set(order)):
        raise ValueError("the periods {} are not contiguous blocks in order".format(order))
    return {int(period[a]): (int(frame[a]), int(frame[b - 1])) for a, b in zip(starts, stops)}


# -- writing --------------------------------------------------------------------------------------

def player_ids(home, away):
    """[(player id, team id, name, shirt number)] in slot order: home columns, then away."""
    out = []
    for team, (_, letter, team_id) in ((home, SIDES[0]), (away, SIDES[1])):
        for number, name in zip(team["numbers"], team["names"]):
            out.append((letter + number, team_id, name, number))
    return out


def metadata_xml(players, fps, blocks, n_frames, pitch, team_names):
    """The ATD metadata XML as a string (FIFA/EPTS layout, the elements engine/data.py reads)."""
    parameters = []
    for period in sorted(blocks):
        for name, value in zip(HALF_PARAMETERS.get(period, ("period_{}_start".format(period),
                                                            "period_{}_end".format(period))),
                               blocks[period]):
            parameters.append("        <ProviderParameter><Name>{}</Name><Value>{}</Value>"
                              "</ProviderParameter>".format(name, value))
    teams = ["      <Team id={}><Name>{}</Name></Team>".format(quoteattr(team_id), escape(name))
             for (_, _, team_id), name in zip(SIDES, team_names)]
    plist = ["      <Player id={} teamId={}><Name>{}</Name><ShirtNumber>{}</ShirtNumber>"
             "</Player>".format(quoteattr(pid), quoteattr(tid), escape(name), escape(number))
             for pid, tid, name, number in players]
    channels = ["      <PlayerChannel channelId=\"{1}\" id={0} playerId={2}/>".format(
        quoteattr("{}_{}".format(pid, c)), c, quoteattr(pid))
        for pid, _, _, _ in players for c in ("x", "y")]
    refs = []
    for pid, _, _, _ in players:
        refs.append("        <SplitRegister separator=\",\">")
        refs += ["          <PlayerChannelRef playerChannelId={}/>".format(
            quoteattr("{}_{}".format(pid, c))) for c in ("x", "y")]
        refs.append("        </SplitRegister>")
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<main>",
        "  <Metadata>",
        "    <GlobalConfig>",
        "      <ProviderName>{}</ProviderName>".format(escape(PROVIDER)),
        "      <FrameRate>{}</FrameRate>".format(fps),
        "      <ProviderGlobalParameters>",
    ] + parameters + [
        "      </ProviderGlobalParameters>",
        "    </GlobalConfig>",
        "    <Sessions>",
        "      <Session id=\"session1\">",
        "        <FieldSize><Width>{:g}</Width><Height>{:g}</Height></FieldSize>".format(*pitch),
        "      </Session>",
        "    </Sessions>",
        "    <Teams>",
    ] + teams + [
        "    </Teams>",
        "    <Players>",
    ] + plist + [
        "    </Players>",
        "    <PlayerChannels>",
    ] + channels + [
        "    </PlayerChannels>",
        "  </Metadata>",
        "  <DataFormatSpecifications>",
        "    <DataFormatSpecification endFrame=\"{}\" separator=\":\" startFrame=\"1\">".format(
            n_frames),
        "      <StringRegister name=\"frameCount\"/>",
        "      <SplitRegister separator=\";\">",
    ] + refs + [
        "      </SplitRegister>",
        "      <SplitRegister separator=\",\">",
        "        <BallChannelRef channelId=\"x\"/>",
        "        <BallChannelRef channelId=\"y\"/>",
        "      </SplitRegister>",
        "    </DataFormatSpecification>",
        "  </DataFormatSpecifications>",
        "</main>",
    ]
    return "\n".join(lines) + "\n"


def format_values(values):
    """Strings of a float array with DECIMALS decimals, NAN_TOKEN where the value is NaN."""
    text = np.char.mod("%.{}f".format(DECIMALS), values)
    text[np.isnan(values)] = NAN_TOKEN
    return text


def raw_lines(frame, players_norm, ball_norm):
    """The raw ATD lines "<frame_id>:<x,y>;<x,y>...:<bx,by>", one per frame, with newlines."""
    n_frames, n_slots, _ = players_norm.shape
    cells = format_values(players_norm.reshape(n_frames, 2 * n_slots))
    ball = format_values(ball_norm)
    pattern = "{}:" + ";".join(["{},{}"] * n_slots) + ":{},{}\n"
    return [pattern.format(int(f), *row, *b) for f, row, b in zip(frame, cells.tolist(),
                                                                    ball.tolist())]


def check_round_trip(original, written):
    """Raise unless the written values equal the input values (input has at most DECIMALS)."""
    if not _same(np.round(original, DECIMALS), np.round(written, DECIMALS)):
        raise ValueError("the metre round trip changed a value at {} decimals".format(DECIMALS))
    if not _same(np.round(original, DECIMALS), original):
        raise ValueError("the input has more than {} decimals".format(DECIMALS))


def convert(home_path, away_path, out_dir, stem, pitch=DEFAULT_PITCH):
    """Write <out_dir>/<stem>_atd.txt and <stem>_atd.xml. Return a summary dict."""
    home, away = read_team_csv(home_path), read_team_csv(away_path)
    check_pair(home, away)
    frame = home["frame"]
    n_frames = len(frame)
    fps = frame_rate(frame, home["time"])
    blocks = period_blocks(home["period"], frame)

    # Metres inside the tool (PLAN.md 6): players of both teams in slot order, then the ball.
    players_m = to_metres(np.concatenate([home["players"], away["players"]], axis=1), pitch)
    ball_m = to_metres(home["ball"], pitch)
    players_norm = to_normalised(players_m, pitch)
    ball_norm = to_normalised(ball_m, pitch)
    check_round_trip(np.concatenate([home["players"], away["players"]], axis=1), players_norm)
    check_round_trip(home["ball"], ball_norm)

    players = player_ids(home, away)
    if len(set(p[0] for p in players)) != len(players):
        raise ValueError("two players share an id")
    os.makedirs(out_dir, exist_ok=True)
    xml_path = os.path.join(out_dir, stem + "_atd.xml")
    raw_path = os.path.join(out_dir, stem + "_atd.txt")
    with open(xml_path, "w") as handle:
        handle.write(metadata_xml(players, fps, blocks, n_frames, pitch,
                                  (home["team"], away["team"])))
    with open(raw_path, "w") as handle:
        handle.writelines(raw_lines(frame, players_norm, ball_norm))

    on_pitch = [np.sum(~np.isnan(t["players"][:, :, 0]), axis=1) for t in (home, away)]
    length, width = pitch
    bx, by = ball_m[:, 0], ball_m[:, 1]
    outside = (bx < 0) | (bx > length) | (by < 0) | (by > width)
    return {
        "atd_meta": xml_path,
        "atd_raw": raw_path,
        "n_frames": n_frames,
        "fps": fps,
        "periods": blocks,
        "slots": {"home": len(home["numbers"]), "away": len(away["numbers"])},
        "on_pitch_per_frame": {side: sorted(set(int(v) for v in counts))
                               for side, counts in zip(("home", "away"), on_pitch)},
        "ball_missing_share": float(np.mean(np.isnan(bx))),
        "ball_outside_pitch_share": float(np.mean(outside[~np.isnan(bx)])),
        "goalkeeper_median_x_m": goalkeeper_sides(home, away, blocks, frame, pitch),
    }


def goalkeeper_sides(home, away, blocks, frame, pitch):
    """{side: {period: median x in metres of the first player column}}.

    In both samples the first column of each file is the goalkeeper (shirt 11 and 25, tracked
    on every frame). The value shows which goal each team defends in each period.
    """
    out = {}
    for side, team in (("home", home), ("away", away)):
        x_m = to_metres(team["players"][:, 0, :], pitch)[:, 0]
        out[side] = {period: round(float(np.nanmedian(x_m[(frame >= a) & (frame <= b)])), 1)
                     for period, (a, b) in sorted(blocks.items())}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--home", required=True, help="<stem>_RawTrackingData_Home_Team.csv")
    parser.add_argument("--away", required=True, help="<stem>_RawTrackingData_Away_Team.csv")
    parser.add_argument("--stem", required=True, help="file stem of the bundle")
    parser.add_argument("--out-dir", required=True, help="folder for the bundle")
    parser.add_argument("--pitch", nargs=2, type=float, default=DEFAULT_PITCH,
                        metavar=("LENGTH", "WIDTH"), help="pitch size in metres")
    args = parser.parse_args(argv)
    summary = convert(args.home, args.away, args.out_dir, args.stem, tuple(args.pitch))
    print("frames {n_frames}  fps {fps}  periods {periods}".format(**summary))
    print("slots {slots}  players on the pitch per frame {on_pitch_per_frame}".format(**summary))
    print("ball missing {:.1%}  ball outside the pitch {:.1%} of visible frames".format(
        summary["ball_missing_share"], summary["ball_outside_pitch_share"]))
    print("goalkeeper median x (m) {}".format(summary["goalkeeper_median_x_m"]))
    print("wrote {} and {}".format(summary["atd_meta"], summary["atd_raw"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
