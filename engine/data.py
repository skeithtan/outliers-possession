"""Data layer: every provided asset loaded onto the video-frame index (PLAN.md Section 3.1).

Everything downstream imports from here; nothing else in `engine/` opens `data/`.

    load_tactical()      ATD of the tactical recording -> dict of arrays, cached in cache/tactical.npz
    load_broadcast()     ATD of the broadcast recording (+ homography `valid`), cache/broadcast.npz
    load_homography()    per-frame homography of either source
    load_tagging()       Smart Tagging XML -> DataFrame, frames on the tactical video index
    graded_blocks()      graded video-frame blocks per half, from games.py
    pitch_m()            pitch size in metres, from games.py
    kickoff_frames()     first graded frame of half 1, second-half kickoff anchor
    resolve_game()       the match name that every function above reads
    fps()                frame rate of a match, from games.py
    frame_count()        number of video frames of one source of a match

Every function above takes a `game` argument. None means games.GAME. Each function calls
resolve_game() when it runs, not when Python imports this module. That is why
`python -m engine.run --game <name>` can read another match of games.py.

Conventions (PLAN.md Section 0):
  * ATD `frame_id` is 1-based: row i of every array is video frame i = frame_id - 1.
  * Positions are returned normalised 0-1 exactly as the ATD stores them (normalised coordinates
    only at I/O; callers convert to metres with pitch_m()).
  * Smart Tagging seconds x fps = tactical video frame; intervals are half-open [start, end).

Nothing match-specific is written here (PLAN.md 5.5): every match constant comes from games.py or
from the bundle's own metadata. Never reads the ground truth (team rule). numpy + pandas + stdlib.
"""

import io
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import games  # noqa: E402

# Bump when the parser or the cached layout changes, so stale caches are rebuilt.
CACHE_VERSION = 1
DEFAULT_CACHE_DIR = os.path.join(ROOT, "cache")
CACHE_KEY_FIELD = "_cache_key"

# Optional field of a games.py entry: {asset: path relative to the data folder} for the tactical
# source of that match. games.SOURCES and games.asset_paths() give one file stem per source for
# every game, and the team rules let us add a games.py entry but not change those functions (G1).
ASSETS_FIELD = "assets"
# Optional field of a games.py entry: True takes the half-1 kickoff from the "kickoff" anchor,
# not from the first graded frame (kickoff_frames, G1 finding 6). The graded block of a match
# can open before its kickoff.
KICKOFF_ANCHOR_FIELD = "kickoff_from_anchor"

HOME, AWAY = 0, 1
SIDE_LETTER = {HOME: "H", AWAY: "A"}

# Raw ATD line: "<frame_id>:<x,y,s>;<x,y,s>;...:<bx,by,bs>" (starter/baseline.py). Every slot is
# always present (NaN when unobserved). The values of one object are the channels that the
# DataFormatSpecification of the ATD XML lists for it, in that order, so a line carries
# 1 + len(player channels) * n_slots + len(ball channels) numbers. The ids are the FIFA/EPTS
# channel ids of the ATD XML files we have: position x, position y and speed (G1).
X_CHANNEL, Y_CHANNEL, SPEED_CHANNEL = "x", "y", "s"
# The layout of starter/baseline.py, used when the XML has no DataFormatSpecification.
DEFAULT_CHANNELS = (X_CHANNEL, Y_CHANNEL, SPEED_CHANNEL)

HOMOGRAPHY_COLUMNS = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]

# Smart Tagging label groups with a dedicated column; every other group becomes an extra column.
TAG_TEAM, TAG_TYPE, TAG_HALF = "Team", "Type", "Half"
# Team tag values meaning "no team" (the export writes "N/A", PLAN.md 1.6), compared lower-case.
NO_TEAM_VALUES = frozenset(["", "n/a", "na", "none", "null", "-"])
# Half tag values look like "1st Half" / "2nd Half": the leading integer is the half.
HALF_PATTERN = re.compile(r"^\s*(\d+)")


# -- ATD -----------------------------------------------------------------------------------------

def resolve_game(game=None):
    """The match name to read: `game`, or games.GAME when `game` is None.

    Raises KeyError if games.GAMES has no entry of that name.
    """
    name = games.GAME if game is None else game
    if name not in games.GAMES:
        raise KeyError("no such game {!r}; have {}".format(name, sorted(games.GAMES)))
    return name


def fps(game=None):
    """Frame rate of the match, an int from games.py."""
    return int(games.GAMES[resolve_game(game)]["fps"])


def frame_count(game=None, source=games.DEFAULT_SOURCE):
    """Number of video frames of one source of the match.

    The tactical count is the submission length. It is the "n_video_frames" of the games.py
    entry, which describes the tactical source (games.py). Other sources use games.source_frames().
    """
    game = resolve_game(game)
    if source == games.DEFAULT_SOURCE:
        return int(games.GAMES[game]["n_video_frames"])
    return int(games.source_frames(source, game))


def asset_paths(game=None, data_dir=None, source=games.DEFAULT_SOURCE):
    """{asset: path} of one source of the match. The paths are not checked for existence.

    A games.py entry with an "assets" field (ASSETS_FIELD) names its own tactical files, relative
    to `data_dir` or games.DATA_DIR (MTAH_DATA_DIR). Such a match has no other source. An entry
    without the field uses games.asset_paths(), the organisers' layout.
    """
    game = resolve_game(game)
    own = games.GAMES[game].get(ASSETS_FIELD)
    if own is None:
        return games.asset_paths(game, data_dir, source)
    if source != games.DEFAULT_SOURCE:
        raise KeyError("game {!r} has no source {!r}: its games.py entry names {} files only"
                       .format(game, source, games.DEFAULT_SOURCE))
    root = data_dir or games.DATA_DIR
    return {name: os.path.join(root, own[name]) for name in sorted(own)}


def has_asset(name, game=None, source=games.DEFAULT_SOURCE):
    """True if the match provides the asset `name` for that source (the file can be absent)."""
    return name in asset_paths(game, source=source)


def require(names, game=None, data_dir=None, source=games.DEFAULT_SOURCE):
    """Paths of the named assets, or one error that names every missing file.

    An entry without an "assets" field uses games.require(), the organisers' function.
    """
    game = resolve_game(game)
    if ASSETS_FIELD not in games.GAMES[game]:
        return games.require(names, game, data_dir, source)
    paths = asset_paths(game, data_dir, source)
    unknown = [name for name in names if name not in paths]
    if unknown:
        raise KeyError("game {!r} has no asset {}; it provides {}".format(
            game, unknown, sorted(paths)))
    missing = [paths[name] for name in names if not os.path.isfile(paths[name])]
    if missing:
        raise IOError("match data not found:\n  {}\n\nSet MTAH_DATA_DIR to the folder that the "
                      "\"assets\" paths of {} in games.py start from.".format(
                          "\n  ".join(missing), game))
    return [paths[name] for name in names]


def _cache_path(cache_dir, game, source):
    """npz cache of one source. games.GAME keeps the name <source>.npz of the first caches."""
    name = source if game == games.GAME else "{}_{}".format(game, source)
    return os.path.join(cache_dir or DEFAULT_CACHE_DIR, "{}.npz".format(name))


def _atd_meta(meta_path, game=None):
    """Team names and slot -> team table from an ATD XML (FIFA/EPTS metadata).

    The first <Team> is home (FIFA/EPTS convention; starter/baseline.py::team_of_each_track).
    A slot is one player position on the raw line. The slot order is the player order of the
    first DataFormatSpecification, the order the raw line is written in. A file can have one
    DataFormatSpecification per frame range, for example after a substitution, with other players
    in the slots. Each block must then put the same team in each slot, because the slot table
    holds one team per slot. Without a DataFormatSpecification the slot order is the <Player>
    order.

    The values of each object on a raw line are the channels of the first
    DataFormatSpecification (_spec_layout). Every DataFormatSpecification must list the same
    channels. Without a DataFormatSpecification the layout is DEFAULT_CHANNELS.

    Returns {"team_names": (home, away), "team_of_slot": int8 (n_slots,), "fps": float,
    "player_channels": tuple, "ball_channels": tuple}.
    """
    game = resolve_game(game)
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    team_ids = [team.get("id") for team in teams]
    team_names = tuple((team.findtext("Name") or "").strip() for team in teams)

    players = list(root.iter("Player"))
    if not players:
        raise ValueError("{}: no Player entries".format(meta_path))
    team_of_player = {player.get("id"): player.get("teamId") for player in players}
    unknown = sorted(set(team_of_player.values()) - set(team_ids))
    if unknown:
        raise ValueError("{}: players on unknown teams {}".format(meta_path, unknown))

    channels = _player_channel_table(root)
    layouts = [_spec_layout(spec, channels, meta_path)
               for spec in root.iter("DataFormatSpecification")]
    slot_teams = [player.get("teamId") for player in players]
    player_channels, ball_channels = DEFAULT_CHANNELS, DEFAULT_CHANNELS
    if layouts:
        _, player_channels, ball_channels = layouts[0]
        patterns = []
        for order, _, _ in layouts:
            absent = [pid for pid in order if pid not in team_of_player]
            if absent:
                raise ValueError("{}: DataFormatSpecification names players {} that have no "
                                 "<Player> entry".format(meta_path, absent))
            patterns.append([team_of_player[pid] for pid in order])
        slot_teams = patterns[0]
        if any(pattern != slot_teams for pattern in patterns[1:]):
            raise ValueError("{}: the DataFormatSpecification blocks put different teams in "
                             "the same slot".format(meta_path))
    for _, other_players, other_ball in layouts[1:]:
        if (other_players, other_ball) != (player_channels, ball_channels):
            raise ValueError("{}: the DataFormatSpecification blocks list different channels"
                             .format(meta_path))

    fps_text = root.findtext(".//FrameRate")
    fps = float(fps_text) if fps_text else float(games.GAMES[game]["fps"])
    if fps != float(games.GAMES[game]["fps"]):
        raise ValueError("{}: FrameRate {} != games.py fps {}".format(
            meta_path, fps, games.GAMES[game]["fps"]))

    team_of_slot = np.array([HOME if t == team_ids[0] else AWAY for t in slot_teams],
                            dtype=np.int8)
    return {"team_names": team_names, "team_of_slot": team_of_slot, "fps": fps,
            "player_channels": player_channels, "ball_channels": ball_channels}


def _player_channel_table(root):
    """{PlayerChannel id: (player id, channel id)} of an ATD XML."""
    return {ch.get("id"): (ch.get("playerId"), ch.get("channelId"))
            for ch in root.iter("PlayerChannel")}


def _spec_layout(spec, channels, meta_path):
    """(player ids, player channel ids, ball channel ids) of one DataFormatSpecification.

    The player ids are in the order of the raw line. `channels` is _player_channel_table().
    Every player of the block must carry the same channels in the same order. The players and
    the ball must each carry an x and a y channel.
    """
    players, per_player = [], {}
    for ref in spec.iter("PlayerChannelRef"):
        key = ref.get("playerChannelId")
        if key not in channels:
            raise ValueError("{}: PlayerChannelRef {!r} names no PlayerChannel".format(
                meta_path, key))
        player, channel = channels[key]
        if player not in per_player:
            players.append(player)
            per_player[player] = []
        per_player[player].append(channel)
    kinds = sorted(set(tuple(chans) for chans in per_player.values()))
    if len(kinds) != 1:
        raise ValueError("{}: the players of a DataFormatSpecification carry channels {}".format(
            meta_path, kinds))
    ball = tuple(ref.get("channelId") for ref in spec.iter("BallChannelRef"))
    for name, chans in (("player", kinds[0]), ("ball", ball)):
        if X_CHANNEL not in chans or Y_CHANNEL not in chans:
            raise ValueError("{}: the {} channels {} have no {!r} or no {!r}".format(
                meta_path, name, list(chans), X_CHANNEL, Y_CHANNEL))
    return players, kinds[0], ball


def _check_atd_layout(raw, raw_path, n_slots, n_frames, player_values=len(DEFAULT_CHANNELS),
                      ball_values=len(DEFAULT_CHANNELS)):
    """Raise unless `raw` has n_frames lines, each with the exact separators of the format.

    Per line: 2 ':' (frame | players | ball), n_slots - 1 ';' between player groups,
    player_values - 1 ',' inside each player group and ball_values - 1 ',' in the ball group.
    Checked per line, not in total, so a short line cannot be balanced by a long one and
    silently padded / shifted by the columnar parser.
    """
    buf = np.frombuffer(raw, dtype=np.uint8)
    ends = np.flatnonzero(buf == ord("\n"))
    if buf.size and buf[-1] != ord("\n"):
        ends = np.append(ends, buf.size)
    if ends.size != n_frames:
        raise ValueError("{}: {} lines, games.py expects {}".format(raw_path, ends.size, n_frames))
    commas = (player_values - 1) * n_slots + (ball_values - 1)
    expected = ((":", 2), (";", n_slots - 1), (",", commas))
    for sep, want in expected:
        positions = np.flatnonzero(buf == ord(sep))
        per_line = np.diff(np.searchsorted(positions, ends), prepend=0)
        bad = np.flatnonzero(per_line != want)
        if bad.size:
            raise ValueError("{}: line {} has {} {!r} separators, expected {} for {} slots".format(
                raw_path, int(bad[0]) + 1, int(per_line[bad[0]]), sep, want, n_slots))


def _parse_atd_raw(raw_path, n_slots, n_frames, player_channels=DEFAULT_CHANNELS,
                   ball_channels=DEFAULT_CHANNELS):
    """Parse a raw ATD .txt into float32 arrays indexed by video frame (frame_id - 1).

    Fails loudly unless the file has exactly n_frames lines, frame_ids 1..n_frames in order, and
    exactly len(player_channels) * n_slots player values plus len(ball_channels) ball values on
    every line. The channel ids (_atd_meta) give the column of x, y and speed. Without a
    SPEED_CHANNEL for the players, `ps` is NaN everywhere (engine.tracks then uses position
    differences where it needs a speed).

    The ball speed column is ignored: it is empty on ~90% of frames (PLAN.md 1.1).
    The player speed column is labelled "speed_kph" in the XML but is already m/s: its median
    ratio to position-derived speed on graded frames is 0.995 (measured for P0-A). It is
    returned unconverted, which is the column PLAN.md Section 1 thresholds were measured on.
    """
    with open(raw_path, "rb") as handle:
        raw = handle.read()
    player_values, ball_values = len(player_channels), len(ball_channels)
    _check_atd_layout(raw, raw_path, n_slots, n_frames, player_values, ball_values)

    n_cols = 1 + player_values * n_slots + ball_values
    table = pd.read_csv(io.BytesIO(raw.replace(b":", b",").replace(b";", b",")), header=None,
                        dtype=np.float64, engine="c", names=list(range(n_cols)))
    del raw
    values = table.to_numpy()
    del table
    if values.shape != (n_frames, n_cols):
        raise ValueError("{}: parsed shape {} != {}".format(raw_path, values.shape,
                                                            (n_frames, n_cols)))
    frame_ids = values[:, 0]
    if not np.array_equal(frame_ids, np.arange(1, n_frames + 1, dtype=np.float64)):
        raise ValueError("{}: frame_id is not contiguous 1..{}".format(raw_path, n_frames))

    players = values[:, 1:1 + player_values * n_slots].reshape(n_frames, n_slots, player_values)
    ball = values[:, 1 + player_values * n_slots:]
    column = list(player_channels).index
    if SPEED_CHANNEL in player_channels:
        speed = players[:, :, column(SPEED_CHANNEL)]
    else:
        speed = np.full((n_frames, n_slots), np.nan)
    return {
        "px": np.ascontiguousarray(players[:, :, column(X_CHANNEL)], dtype=np.float32),
        "py": np.ascontiguousarray(players[:, :, column(Y_CHANNEL)], dtype=np.float32),
        "ps": np.ascontiguousarray(speed, dtype=np.float32),
        "bx": np.ascontiguousarray(ball[:, list(ball_channels).index(X_CHANNEL)],
                                   dtype=np.float32),
        "by": np.ascontiguousarray(ball[:, list(ball_channels).index(Y_CHANNEL)],
                                   dtype=np.float32),
    }


# -- cache ---------------------------------------------------------------------------------------

def _cache_key(game, source, paths):
    """JSON string identifying a cache: version, game, source, and size + mtime of each input."""
    files = []
    for name in sorted(paths):
        stat = os.stat(paths[name])  # follows the data/ symlinks
        files.append([name, os.path.basename(paths[name]), stat.st_size, stat.st_mtime_ns])
    return json.dumps({"version": CACHE_VERSION, "game": game, "source": source,
                       "files": files}, sort_keys=True)


def _read_cache(path, key):
    """Arrays from an npz cache if it exists and its key matches, else None."""
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as npz:
            if CACHE_KEY_FIELD not in npz.files or str(npz[CACHE_KEY_FIELD]) != key:
                return None
            return {name: npz[name] for name in npz.files if name != CACHE_KEY_FIELD}
    except Exception:  # unreadable/partial cache: rebuild rather than fail
        return None


def _write_cache(path, key, arrays):
    """Write arrays + key to an npz atomically (temp file, then os.replace)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = "{}.tmp-{}".format(path, os.getpid())
    try:
        with open(tmp, "wb") as handle:  # a handle, so numpy does not append ".npz"
            np.savez(handle, **dict(arrays, **{CACHE_KEY_FIELD: np.array(key)}))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _load_atd(game, source, data_dir, cache_dir, use_cache, with_valid):
    """Shared body of load_tactical / load_broadcast."""
    game = resolve_game(game)
    inputs = ["atd_meta", "atd_raw"] + (["homography"] if with_valid else [])
    paths = dict(zip(inputs, require(inputs, game, data_dir, source)))
    n_frames = frame_count(game, source)
    cache_path = _cache_path(cache_dir, game, source)
    key = _cache_key(game, source, paths)

    arrays = _read_cache(cache_path, key) if use_cache else None
    if arrays is None:
        meta = _atd_meta(paths["atd_meta"], game)
        arrays = _parse_atd_raw(paths["atd_raw"], len(meta["team_of_slot"]), n_frames,
                                meta["player_channels"], meta["ball_channels"])
        arrays["team_of_slot"] = meta["team_of_slot"]
        arrays["team_names"] = np.array(meta["team_names"])
        if with_valid:
            arrays["valid"] = load_homography(source, game, data_dir)[2]
        if use_cache:
            _write_cache(cache_path, key, arrays)

    out = dict(arrays)
    if out["px"].shape[0] != n_frames:
        raise ValueError("{} cache has {} frames, expected {}".format(
            source, out["px"].shape[0], n_frames))
    out["team_names"] = tuple(str(name) for name in out["team_names"])
    out.update(n_frames=n_frames, n_slots=int(out["px"].shape[1]),
               fps=int(games.GAMES[game]["fps"]), source=source, game=game)
    return out


def load_tactical(game=None, data_dir=None, cache_dir=None, use_cache=True):
    """Tactical ATD on the tactical video index (the submission index).

    Returns a dict:
      px, py, ps    float32 (N, n_slots)  normalised 0-1 positions, speed m/s; NaN = missing
      bx, by        float32 (N,)          ball, normalised 0-1; NaN = missing
      team_of_slot  int8 (n_slots,)       0 = home (first Team in the XML), 1 = away
      team_names    (home, away)          club names from the ATD XML
      n_frames, n_slots, fps, source, game

    Row i is video frame i (ATD frame_id - 1). Cached in <cache_dir>/tactical.npz, keyed by the
    size and mtime of the input files, so a changed bundle is re-parsed. A match other than
    games.GAME uses <cache_dir>/<game>_tactical.npz.
    """
    return _load_atd(game, "tactical", data_dir, cache_dir, use_cache, with_valid=False)


def load_broadcast(game=None, data_dir=None, cache_dir=None, use_cache=True):
    """Broadcast ATD on the broadcast video's own index (NOT aligned to tactical; PLAN.md 1.5).

    Same keys as load_tactical(), plus `valid`: bool (N,) from the broadcast homography (False on
    replays / close-ups, where no pitch transform exists). Cached in <cache_dir>/broadcast.npz.
    """
    return _load_atd(game, "broadcast", data_dir, cache_dir, use_cache, with_valid=True)


# -- homography ----------------------------------------------------------------------------------

def load_homography(source=games.DEFAULT_SOURCE, game=None, data_dir=None):
    """(H, reprojection_error, valid) for one source, indexed by that source's video frame.

    H is float64 (N, 3, 3), mapping normalised pitch (0-1) to image pixels, NaN where invalid;
    reprojection_error float64 (N,), lower is better; valid bool (N,).
    Fails loudly unless the `frame` column is exactly 0..N-1 with N from games.py.
    """
    game = resolve_game(game)
    path = require(["homography"], game, data_dir, source)[0]
    table = pd.read_csv(path)
    missing = [c for c in ["frame", "reprojection_error", "valid"] + HOMOGRAPHY_COLUMNS
               if c not in table.columns]
    if missing:
        raise ValueError("{}: missing columns {}".format(path, missing))
    n_frames = frame_count(game, source)
    if not np.array_equal(table["frame"].to_numpy(), np.arange(n_frames)):
        raise ValueError("{}: frame column is not 0..{}".format(path, n_frames - 1))
    valid_raw = table["valid"].to_numpy()
    if not np.isin(valid_raw, [0, 1]).all():
        raise ValueError("{}: valid column is not 0/1".format(path))
    valid = valid_raw.astype(bool)
    matrices = table[HOMOGRAPHY_COLUMNS].to_numpy(dtype=np.float64).reshape(-1, 3, 3)
    if not np.isfinite(matrices[valid]).all():
        raise ValueError("{}: non-finite matrix on a valid frame".format(path))
    return matrices, table["reprojection_error"].to_numpy(dtype=np.float64), valid


# -- Smart Tagging -------------------------------------------------------------------------------

def _compact(text):
    """Lower-case alphanumerics only: 'Atletic Club' -> 'atleticclub'."""
    return re.sub(r"[^0-9a-z]", "", text.lower())


def _is_subsequence(short, long):
    """True if `short` can be obtained from `long` by deleting characters."""
    chars = iter(long)
    return all(c in chars for c in short)


def _team_side(value, team_names):
    """Map a Smart Tagging Team value to "H" / "A" / None, using the ATD team names.

    Case-insensitive. Tries, in order: no-team markers ("N/A" -> None), exact match, substring
    either way, then subsequence either way (tolerates a dropped letter such as
    "Athletic" / "Atletic"). Each step must identify exactly one side; anything else raises so a
    new export fails loudly instead of silently losing its team prior.
    """
    if value is None or value.strip().lower() in NO_TEAM_VALUES:
        return None
    tag = _compact(value)
    names = [_compact(name) for name in team_names]
    tests = (lambda a, b: a == b,
             lambda a, b: a in b or b in a,
             lambda a, b: _is_subsequence(a, b) or _is_subsequence(b, a))
    for test in tests:
        hits = [side for side, name in enumerate(names) if tag and name and test(tag, name)]
        if len(hits) == 1:
            return SIDE_LETTER[hits[0]]
        if len(hits) > 1:
            break
    raise ValueError("Smart Tagging team {!r} matches neither / both of {}".format(
        value, list(team_names)))


def _half_number(value):
    """'1st Half' -> 1, '2nd Half' -> 2; raises on anything without a leading integer."""
    match = HALF_PATTERN.match(value or "")
    if not match:
        raise ValueError("Smart Tagging half {!r} has no leading half number".format(value))
    return int(match.group(1))


def _snake(text):
    """'Max Players in the box' -> 'max_players_in_the_box'."""
    return re.sub(r"[^0-9a-z]+", "_", text.strip().lower()).strip("_")


def load_tagging(game=None, data_dir=None):
    """Smart Tagging instances as a DataFrame on the tactical video index, in file order.

    Columns: id, code (upper-case), team ("H"/"A"/None), type (str or None), half (int),
    start_s, end_s, start_frame, end_frame (half-open: end_frame is the first frame after),
    then one snake_case column per remaining label group (None where absent), sorted by name.

    Frames are round(seconds x fps), as in starter/load_smart_tagging.py (the XML clock is zero at
    the first tactical video frame). Team names come from the tactical ATD XML, first = home.
    It is a phase/set-piece prior, not ground truth (PLAN.md 1.6).

    A match whose games.py entry provides no "smart_tagging" asset gets an empty DataFrame with
    the nine columns above. A provided asset whose file is missing is an error.
    """
    game = resolve_game(game)
    if not has_asset("smart_tagging", game):
        return _tagging_frame([], set())
    path = require(["smart_tagging"], game, data_dir)[0]
    meta_path = require(["atd_meta"], game, data_dir)[0]
    team_names = _atd_meta(meta_path, game)["team_names"]
    fps = games.GAMES[game]["fps"]

    rows, extra_groups = [], set()
    for instance in ET.parse(path).iter("instance"):
        tags = {}
        for label in instance.findall("label"):
            group = (label.findtext("group") or "").strip()
            if group in tags:
                raise ValueError("{}: instance {} repeats label group {!r}".format(
                    path, instance.findtext("ID"), group))
            tags[group] = label.findtext("text")
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        id_text = instance.findtext("ID")
        row = {
            "id": int(id_text) if id_text is not None else -1,
            "code": (instance.findtext("code") or "").strip().upper(),
            "team": _team_side(tags.get(TAG_TEAM), team_names),
            "type": tags[TAG_TYPE].strip() if tags.get(TAG_TYPE) is not None else None,
            "half": _half_number(tags.get(TAG_HALF)),
            "start_s": start_s,
            "end_s": end_s,
            "start_frame": int(round(start_s * fps)),
            "end_frame": int(round(end_s * fps)),
        }
        for group, text in tags.items():
            if group not in (TAG_TEAM, TAG_TYPE, TAG_HALF):
                row[_snake(group)] = text
                extra_groups.add(_snake(group))
        rows.append(row)
    return _tagging_frame(rows, extra_groups)


def _tagging_frame(rows, extra_groups):
    """The load_tagging() DataFrame of these rows, with the column types of a full table."""
    columns = ["id", "code", "team", "type", "half", "start_s", "end_s", "start_frame",
               "end_frame"] + sorted(extra_groups)
    frame = pd.DataFrame(rows, columns=columns)
    frame = frame.astype({"id": np.int64, "half": np.int64, "start_frame": np.int64,
                          "end_frame": np.int64})
    if not rows:
        frame = frame.astype({"start_s": np.float64, "end_s": np.float64})
    for column in ["team", "type"] + sorted(extra_groups):
        frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
    return frame


# -- match constants (all from games.py) ---------------------------------------------------------

def graded_blocks(game=None):
    """[(half, first_frame, last_frame_inclusive)] of graded video frames, from games.py."""
    return [(int(h), int(a), int(b)) for h, a, b in games.half_video_blocks(resolve_game(game))]


def pitch_m(game=None):
    """(length, width) of the pitch in metres, from games.py."""
    length, width = games.GAMES[resolve_game(game)]["pitch"]
    return float(length), float(width)


def kickoff_frames(game=None):
    """{1: first graded frame of half 1, 2: second-half kickoff anchor} as video frames.

    Half 1 uses the start of the graded block (play is alive from the first graded frame,
    PLAN.md Section 0); half 2 uses games.py's kickoff_2nd anchor, because the second graded
    block opens ~2 minutes before the kickoff (those frames are D).

    An entry that sets the optional field KICKOFF_ANCHOR_FIELD to True takes half 1 from its
    "kickoff" anchor instead. That is for a match whose first graded block opens before the
    kickoff. An entry without the field keeps the rule above.
    """
    game = resolve_game(game)
    entry = games.GAMES[game]
    first_half = graded_blocks(game)[0][1]
    if entry.get(KICKOFF_ANCHOR_FIELD, False):
        first_half = int(games.anchor_video_frame(entry["kickoff"]))
    return {1: first_half,
            2: int(games.anchor_video_frame(entry["kickoff_2nd"]))}
