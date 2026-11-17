"""Team-swap symmetry of the engine on a synthetic match (P9-D, IDEAS-REVIEW Blend D).

The test writes a small ATD-style bundle of two halves to a temporary folder: an ATD XML, a raw
tracking file and a Smart Tagging XML. It adds a games.py-like entry for the bundle with
monkeypatch and runs engine.possession.run() on it. Then it writes the same match again with the
two teams swapped: the other team is first in the XML, so it is home, and its players fill the
first slots. The positions, the ball and the tags do not change. The output of the second run
must be the output of the first run with H and A swapped, and with D and X unchanged. A
difference shows a hidden home default in the engine.

The scene is built from simple physics and is the same on every run. It has no randomness:
  * Each team keeps a formation in its own half and swaps ends at half time. Each player moves
    on a sine path of his own around his formation spot, at about 3 m/s.
  * Each half starts with a kickoff from the centre mark (Law 8): the other team kicks off the
    second half. The second graded block opens 2 s before its kickoff.
  * The ball goes from player to player at PASS_SPEED and stays at a receiver's feet for HOLD
    frames. Each possession chain ends with a pass to the other team.
  * Once in each half a pass goes out over a touchline. The tracker loses the ball, the players
    slow down, and the ball comes back moving, thrown in by the other team (Law 15).

The test needs no match data.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import games
from engine import data, possession
from engine.config import Config

FPS = 25
PITCH = (105.0, 68.0)
N_FRAMES = 9200
# Graded blocks of video frames (inclusive), and the kickoff frame of each half.
BLOCKS = {1: (50, 4549), 2: (4650, 9149)}
KICKOFF = {1: 50, 2: 4700}
TEAM_NAMES = ("Blue", "Red")       # the first team is home in the original bundle
N_PER_TEAM = 11
MARK = (PITCH[0] / 2.0, PITCH[1] / 2.0)
# The formation of a team that defends the goal at x = 0, in metres. Slot 0 is the goalkeeper.
FORMATION = [(6.0, 34.0), (22.0, 10.0), (20.0, 26.0), (20.0, 42.0), (22.0, 58.0),
             (36.0, 12.0), (34.0, 28.0), (34.0, 40.0), (36.0, 56.0), (46.0, 24.0), (46.0, 44.0)]
KICKER = 9                         # the forward of the kicking team stands at the mark
KICKER_SPOT = (51.5, 34.0)         # 1 m from the mark: within KICKOFF_TAKER_M
SWING_M = (1.2, 3.5)               # sine amplitude of a goalkeeper, of an outfield player
PASS_SPEED = 14.0                  # m/s, a ground pass below AIRBORNE_SPEED
THROW_SPEED = 10.0                 # m/s
HOLD = 25                          # frames a receiver keeps the ball
BALL_OFFSET = (0.4, 0.0)           # the ball at a receiver's feet, metres
CHAINS = (4, 3, 5, 2, 4, 3, 5, 4, 3, 4, 2, 5, 3, 4, 4, 3, 5, 2, 4, 3, 4, 5, 3, 4)
OUT_CHAIN = 2                      # this chain of each half ends with a pass out of play
OUT_GAP = 200                      # frames the tracker loses the ball after it goes out
STOP_LEVEL = 0.1                   # sine amplitude factor of the players during the stoppage


def _kicking_team(half):
    return 0 if half == 1 else 1


def _defends_left(team, half):
    """True if `team` (0 or 1) defends the goal at x = 0 in `half`."""
    return (team == 0) == (half == 1)


def _base(team, index, half):
    """Formation spot (m) of player `index` of `team` in `half`."""
    x, y = FORMATION[index]
    if team == _kicking_team(half) and index == KICKER:
        x, y = KICKER_SPOT
    if _defends_left(team, half):
        return x, y
    return PITCH[0] - x, PITCH[1] - y


def _envelope(frames, half, stoppage):
    """Sine amplitude factor: 0 before the kickoff, a ramp after it, a dip in the stoppage."""
    k = KICKOFF[half]
    env = np.clip((frames - k) / 75.0, 0.0, 1.0)
    lo, hi = stoppage
    down = np.clip((frames - lo) / 25.0, 0.0, 1.0)
    up = np.clip((frames - hi) / 50.0, 0.0, 1.0)
    dip = 1.0 - (1.0 - STOP_LEVEL) * down * (1.0 - up)
    return env * dip


def _player_paths(half, stoppage):
    """(px, py) float64 (N_FRAMES, 22) in metres for one half, NaN outside its block.

    Columns 0-10 are team 0, columns 11-21 are team 1. The first half also fills the frames
    before its block, with the players on their kickoff spots.
    """
    first, last = BLOCKS[half]
    lo = 0 if half == 1 else first
    frames = np.arange(lo, last + 1, dtype=np.float64)
    env = _envelope(frames, half, stoppage)
    px = np.full((N_FRAMES, 2 * N_PER_TEAM), np.nan)
    py = np.full((N_FRAMES, 2 * N_PER_TEAM), np.nan)
    seconds = frames / FPS
    for team in (0, 1):
        for index in range(N_PER_TEAM):
            col = team * N_PER_TEAM + index
            x0, y0 = _base(team, index, half)
            amp = SWING_M[0] if index == 0 else SWING_M[1]
            period_x, period_y = 5.0 + 0.37 * col, 6.0 + 0.29 * col
            phase_x = 2 * np.pi * seconds / period_x + 0.9 * col
            phase_y = 2 * np.pi * seconds / period_y + 0.4 * col
            px[lo:last + 1, col] = x0 + env * amp * np.sin(phase_x)
            py[lo:last + 1, col] = y0 + env * amp * np.cos(phase_y)
    return px, py


def _receivers(start):
    """Endless cycle of outfield player indices of a chain, never the same one twice in a row."""
    index = start
    while True:
        index = (index * 3 + 1) % 10 + 1
        yield index


def _ball_and_tags(half, px, py):
    """(bx, by, tags, stoppage): the ball path of one half in metres, its tags, its stoppage.

    `tags` is [(code, team index or None, type or None, first frame, end frame)].
    """
    first, last = BLOCKS[half]
    bx = np.full(N_FRAMES, np.nan)
    by = np.full(N_FRAMES, np.nan)
    lo = 0 if half == 1 else first
    k = KICKOFF[half]
    bx[lo:k + 1], by[lo:k + 1] = MARK

    def feet(team, index, t):
        col = team * N_PER_TEAM + index
        return px[t, col] + BALL_OFFSET[0], py[t, col] + BALL_OFFSET[1]

    def fly(t0, p0, p1, speed):
        """Linear flight from p0 at t0 to p1. Returns the arrival frame."""
        dist = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        steps = max(8, int(round(dist / speed * FPS)))
        w = np.arange(1, steps + 1) / float(steps)
        bx[t0 + 1:t0 + steps + 1] = p0[0] + w * (p1[0] - p0[0])
        by[t0 + 1:t0 + steps + 1] = p0[1] + w * (p1[1] - p0[1])
        return t0 + steps

    def hold(team, index, t0, t1):
        cols = team * N_PER_TEAM + index
        bx[t0:t1 + 1] = px[t0:t1 + 1, cols] + BALL_OFFSET[0]
        by[t0:t1 + 1] = py[t0:t1 + 1, cols] + BALL_OFFSET[1]

    tags, stoppage = [], (last + 1, last + 1)
    team = _kicking_team(half)
    t, pos = k, MARK
    budget = last - 4 * FPS                 # stop the script 4 s before the block end
    holder = None
    for chain, length in enumerate(CHAINS):
        pick = _receivers(chain)
        chain_start, chain_team = t, team
        for step in range(length):
            # The last pass of a chain goes to the other team: an interception.
            receiver_team = team if step < length - 1 else 1 - team
            index = next(pick) if receiver_team == team else 5 + (chain % 4)
            target_t = t + 30
            if target_t + HOLD + 120 > budget:
                break
            arrive = fly(t, pos, feet(receiver_team, index, min(target_t, last)), PASS_SPEED)
            hold(receiver_team, index, arrive, arrive + HOLD)
            holder = (receiver_team, index)
            t, pos = arrive + HOLD, feet(receiver_team, index, arrive + HOLD)
            if chain == OUT_CHAIN and step == length - 2:
                # A pass out over the touchline y = 0, then a throw-in by the other team.
                out_x = float(np.clip(pos[0], 20.0, 85.0))
                cross = fly(t, pos, (out_x, -2.0), THROW_SPEED)
                gap_end = cross + OUT_GAP
                stoppage = (cross - 10, gap_end)
                thrower = 1 - team
                index = 5 if _defends_left(thrower, half) else 8
                p1 = feet(thrower, index, gap_end + 40)
                arrive = fly(gap_end, (out_x, 0.3), p1, THROW_SPEED)
                bx[cross + 1:gap_end + 1] = np.nan
                by[cross + 1:gap_end + 1] = np.nan
                hold(thrower, index, arrive, arrive + HOLD)
                tags.append(("SET PIECES", None, "Throw-in", cross + 40, arrive + 25))
                holder = (thrower, index)
                t, pos = arrive + HOLD, feet(thrower, index, arrive + HOLD)
                team = thrower
                break
        else:
            team = 1 - team
        if chain in (0, 1):
            code = "BUILD UP" if chain == 0 else "PROGRESSION"
            tags.append((code, chain_team, None, chain_start, t))
        if t + HOLD + 150 > budget:
            break
    if holder is not None:
        hold(holder[0], holder[1], t, last)
    return bx, by, tags, stoppage


def scene(untracked_kicker=()):
    """(px, py, bx, by, tags) of the whole synthetic match, in metres and video frames.

    For each half in `untracked_kicker` the kicker is missing from the tracking for the 3 s
    before the kickoff and the first 0.4 s after it. The centre mark rule then finds no taker.
    """
    px = np.full((N_FRAMES, 2 * N_PER_TEAM), np.nan)
    py = np.full((N_FRAMES, 2 * N_PER_TEAM), np.nan)
    bx = np.full(N_FRAMES, np.nan)
    by = np.full(N_FRAMES, np.nan)
    tags = []
    for half in (1, 2):
        # The stoppage frames come from the ball script, and the player paths need them.
        # The script reads the players' paths only up to the stoppage on the first pass.
        hpx, hpy = _player_paths(half, (N_FRAMES, N_FRAMES))
        _, _, _, stoppage = _ball_and_tags(half, hpx, hpy)
        hpx, hpy = _player_paths(half, stoppage)
        hbx, hby, htags, _ = _ball_and_tags(half, hpx, hpy)
        first, last = BLOCKS[half]
        lo = 0 if half == 1 else first
        px[lo:last + 1], py[lo:last + 1] = hpx[lo:last + 1], hpy[lo:last + 1]
        bx[lo:last + 1], by[lo:last + 1] = hbx[lo:last + 1], hby[lo:last + 1]
        tags += [tag + (half,) for tag in htags]
        if half in untracked_kicker:
            col = _kicking_team(half) * N_PER_TEAM + KICKER
            k = KICKOFF[half]
            px[max(0, k - 75):k + 10, col] = np.nan
            py[max(0, k - 75):k + 10, col] = np.nan
    return px, py, bx, by, tags


def _speeds(px, py):
    """Player speed (m/s) from the central difference of the positions, NaN where missing."""
    vx = np.gradient(px, axis=0) * FPS
    vy = np.gradient(py, axis=0) * FPS
    return np.hypot(vx, vy)


def _number(value):
    return "NaN" if not np.isfinite(value) else "{:.6f}".format(value)


def write_bundle(folder, px, py, bx, by, tags, home_first):
    """Write meta.xml, track.txt and tags.xml of the scene to `folder`.

    With home_first False the second team is first in the XML and its players fill slots
    0-10. The positions of every player and of the ball are the same.
    """
    os.makedirs(folder, exist_ok=True)
    order = [0, 1] if home_first else [1, 0]
    cols = np.concatenate([np.arange(N_PER_TEAM) + t * N_PER_TEAM for t in order])
    root = ET.Element("Metadata")
    ET.SubElement(ET.SubElement(root, "GlobalConfig"), "FrameRate").text = str(FPS)
    teams = ET.SubElement(root, "Teams")
    for t in order:
        team = ET.SubElement(teams, "Team", id="T{}".format(t))
        ET.SubElement(team, "Name").text = TEAM_NAMES[t]
    players = ET.SubElement(root, "Players")
    for col in cols:
        ET.SubElement(players, "Player", id="P{}".format(col),
                      teamId="T{}".format(col // N_PER_TEAM))
    ET.ElementTree(root).write(os.path.join(folder, "meta.xml"))

    speed = _speeds(px, py)
    nx, ny = px / PITCH[0], py / PITCH[1]
    lines = []
    for f in range(N_FRAMES):
        groups = ";".join("{},{},{}".format(_number(nx[f, c]), _number(ny[f, c]),
                                            _number(speed[f, c])) for c in cols)
        ball = "{},{},NaN".format(_number(bx[f] / PITCH[0]), _number(by[f] / PITCH[1]))
        lines.append("{}:{}:{}".format(f + 1, groups, ball))
    with open(os.path.join(folder, "track.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")

    root = ET.Element("file")
    instances = ET.SubElement(root, "ALL_INSTANCES")
    for i, (code, team, kind, start, end, half) in enumerate(tags):
        node = ET.SubElement(instances, "instance")
        ET.SubElement(node, "ID").text = str(i + 1)
        ET.SubElement(node, "start").text = "{:.2f}".format(start / float(FPS))
        ET.SubElement(node, "end").text = "{:.2f}".format(end / float(FPS))
        ET.SubElement(node, "code").text = code
        labels = [("Team", "N/A" if team is None else TEAM_NAMES[team]),
                  ("Half", "1st Half" if half == 1 else "2nd Half")]
        if kind is not None:
            labels.append(("Type", kind))
        for group, text in labels:
            label = ET.SubElement(node, "label")
            ET.SubElement(label, "group").text = group
            ET.SubElement(label, "text").text = text
    ET.ElementTree(root).write(os.path.join(folder, "tags.xml"))


def _entry(folder):
    """A games.py-like entry for a bundle in `folder` (relative to the data folder)."""
    halves = {h: (a + 1, b + 1) for h, (a, b) in BLOCKS.items()}
    k2 = KICKOFF[2]
    return {
        "n_video_frames": N_FRAMES, "fps": FPS, "offsets": {1: 1, 2: 1}, "halves": halves,
        "kickoff": {"minute": 0, "second": KICKOFF[1] // FPS, "frame": KICKOFF[1] % FPS},
        "kickoff_2nd": {"minute": k2 // (60 * FPS), "second": (k2 // FPS) % 60,
                        "frame": k2 % FPS},
        "pitch": PITCH,
        "assets": {"atd_meta": folder + "/meta.xml", "atd_raw": folder + "/track.txt",
                   "smart_tagging": folder + "/tags.xml"},
    }


SWAP = {"H": "A", "A": "H", "D": "D", "X": "X"}
# The scenes of the test. "untracked" hides both kickers from the centre mark rule and drops the
# team phase tags, so the kickoff teams come from no rule and the owner starts UNKNOWN.
SCENES = {
    "tracked": {"untracked_kicker": (), "team_tags": True, "kick_teams": {1: "H", 2: "A"}},
    "untracked": {"untracked_kicker": (1, 2), "team_tags": False,
                  "kick_teams": {1: None, 2: None}},
}


@pytest.fixture(scope="module", params=sorted(SCENES))
def runs(request, tmp_path_factory):
    """(scene name, original labels, swapped-bundle labels, original parts, swapped ATD)."""
    spec = SCENES[request.param]
    root = str(tmp_path_factory.mktemp("symmetry_" + request.param))
    px, py, bx, by, tags = scene(spec["untracked_kicker"])
    if not spec["team_tags"]:
        tags = [tag for tag in tags if tag[1] is None]
    write_bundle(os.path.join(root, "home_first"), px, py, bx, by, tags, True)
    write_bundle(os.path.join(root, "away_first"), px, py, bx, by, tags, False)
    patch = pytest.MonkeyPatch()
    try:
        patch.setattr(games, "DATA_DIR", root)
        patch.setattr(data, "DEFAULT_CACHE_DIR", os.path.join(root, "cache"))
        patch.setitem(games.GAMES, "SYNTH-HOME-FIRST", _entry("home_first"))
        patch.setitem(games.GAMES, "SYNTH-AWAY-FIRST", _entry("away_first"))
        cfg = Config.default()
        first, parts = possession.run(cfg, return_parts=True, game="SYNTH-HOME-FIRST")
        second = possession.run(cfg, game="SYNTH-AWAY-FIRST")
        tactical = data.load_tactical("SYNTH-AWAY-FIRST")
    finally:
        patch.undo()
    return request.param, np.asarray(first), np.asarray(second), parts, tactical


def test_the_scene_is_a_plausible_match(runs):
    """The scene gives possessions of both teams, a stoppage in each half and X off the blocks."""
    name, first, _, parts, tactical = runs
    assert tactical["team_names"] == (TEAM_NAMES[1], TEAM_NAMES[0])
    graded = np.zeros(N_FRAMES, dtype=bool)
    for a, b in BLOCKS.values():
        graded[a:b + 1] = True
    assert set(first[~graded].tolist()) == {"X"}
    assert "X" not in set(first[graded].tolist())
    for half, (a, b) in BLOCKS.items():
        labels = first[a:b + 1]
        changes = int(np.sum(labels[1:] != labels[:-1]))
        assert {"H", "A", "D"} <= set(labels.tolist()), half
        assert changes >= 10, half
    assert parts["kick_teams"] == SCENES[name]["kick_teams"]
    assert (parts["tag_lo"] != 0).any() == SCENES[name]["team_tags"]


def test_team_swap_gives_h_and_a_swapped(runs):
    """IDEAS-REVIEW Blend D: swap the teams' slots and names, and H and A swap exactly."""
    _, first, second, _, _ = runs
    expected = np.array([SWAP[v] for v in first.tolist()])
    diff = np.flatnonzero(second != expected)
    assert diff.size == 0, "{} frames differ, first at {}".format(diff.size, diff[:10].tolist())
