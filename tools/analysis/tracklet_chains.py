"""Tracklet chains and the team repair of RESEARCH-6 item T1. Not part of the submission.

    python tools/analysis/tracklet_chains.py report --game NAME [--link-frames 3]
        [--cache-dir DIR] [--out OUT.json]
    python tools/analysis/tracklet_chains.py selftest --game NAME [--seed 0] [--rate-per-min 1.0]
        [--near-ball-m 0] [--link-frames 3] [--cache-dir DIR] [--out OUT.json]

`report` reads the tracking of one match and prints summary numbers. It reads no label. It can
run on the competition match, because it prints summary statistics only (team rules).
`selftest` puts wrong-team episodes with a known truth into an external match, runs the repair
and prints its precision and recall. It refuses the competition match. Set MTAH_DATA_DIR for the
match, as engine.run needs it.

The defect. A tracker can put a player in a slot of the other team for some seconds. The slot of
his own team is empty during that time. The player then counts for the wrong team in every
feature of the engine. On BAR-ATH a team has more than 11 visible slots on 8.1 % (home) and
4.7 % (away) of the graded frames (PLAN.md 1.2).

Definitions.

- Tracklet. A maximal run of frames in which one slot is visible.
- Continuation link. Tracklet A ends and tracklet B starts in another slot at most LINK_FRAMES
  frames later. The first position of B is within NOISE_M + V_MAX x (time between them) of the
  last position of A. A player cannot move farther in that time. The tool keeps a link only if
  B is the nearest start for A and A is the nearest end for B.
- Chain. Tracklets joined by continuation links. A chain follows one player.
- Mixed chain. A chain with tracklets of both teams. One of its team labels is wrong.

The repair (team_moves). In a mixed chain the team with the larger share of the frames is the
team of the player, if that share is at least MAJOR_SHARE. The tracklets of the other team move
to that team. If no team has that share, the count decides for each tracklet: the tracklet moves
if its team has more than MAX_TEAM visible slots and the other team has fewer than MAX_TEAM on
more than half of its frames. apply_moves() puts each moved tracklet into a new slot of the
other team. It changes no position.

Deterministic: fixed order of slots and frames, ties go to the lower index, a seeded
numpy RandomState for the injected episodes. numpy only. Python 3.9.
"""

import argparse
import json
import os
import sys
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Top speed bound of a footballer, m/s. It is above every measured sprint (about 10 to 11 m/s).
# docs/plan/calibration/STRESS_RATES.md Section 1 uses the same value for the teleport test.
V_MAX = 12.0
# Position noise margin of a one-frame step, m (STRESS_RATES.md Section 1: the p99.9 one-frame
# step is 0.31 to 0.37 m on every match).
NOISE_M = 0.5
# A continuation starts at most this many frames after the end. Design value: a re-assigned
# slot continues at once, and a longer gap is a lost player, which this tool does not link.
LINK_FRAMES = 3
# Players of one team on the pitch (Laws of the Game, Law 3).
MAX_TEAM = 11
# Share of the frames of a chain that makes one team the team of the player. Design value.
MAJOR_SHARE = 0.6
# Profile of the injected episodes of `selftest`: episodes per minute, median and 90 % length in
# seconds. Design values near the BAR-ATH over-count profile. Measure them with `report` first.
EPISODES_PER_MIN = 1.0
EPISODE_MEDIAN_S = 2.6
EPISODE_P90_S = 16.0
# z value of the 90 % point of a normal distribution.
Z90 = 1.2816


def runs(mask):
    """(starts, stops) of the True runs of a 1-D bool array. A stop is one past the last frame."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def tracklets(visible):
    """[(slot, start, stop)] of every tracklet, by slot and then by start frame."""
    visible = np.asarray(visible, dtype=bool)
    spans = []
    for slot in range(visible.shape[1]):
        starts, stops = runs(visible[:, slot])
        spans.extend((slot, int(s), int(e)) for s, e in zip(starts, stops))
    return spans


def continuation_links(x, y, spans, fps, link_frames=LINK_FRAMES):
    """{index of tracklet A: index of tracklet B} of the continuation links (module docstring).

    `x` and `y` are positions in metres, shape (frames, slots), NaN where a slot is not visible.
    """
    starts = {}
    for index, (_, start, _) in enumerate(spans):
        starts.setdefault(start, []).append(index)
    n_frames = x.shape[0]
    best_for_start = {}
    for index, (slot, _, stop) in enumerate(spans):
        if stop >= n_frames:
            continue
        choice = None
        for frame in range(stop, min(n_frames, stop + link_frames)):
            limit = NOISE_M + V_MAX * (frame - stop + 1) / float(fps)
            for other in starts.get(frame, ()):
                other_slot = spans[other][0]
                if other_slot == slot:
                    continue
                dist = float(np.hypot(x[frame, other_slot] - x[stop - 1, slot],
                                      y[frame, other_slot] - y[stop - 1, slot]))
                if dist <= limit and (choice is None or dist < choice[0]):
                    choice = (dist, other)
        if choice is None:
            continue
        dist, other = choice
        if other not in best_for_start or dist < best_for_start[other][0]:
            best_for_start[other] = (dist, index)
    return {index: other for other, (_, index) in sorted(best_for_start.items())}


def chains(n_spans, succ):
    """[[tracklet index, ...]] of the chains with two tracklets or more, in time order."""
    has_pred = set(succ.values())
    out = []
    for index in range(n_spans):
        if index in has_pred or index not in succ:
            continue
        chain = [index]
        while chain[-1] in succ:
            chain.append(succ[chain[-1]])
        out.append(chain)
    return out


def team_counts(visible, team_of_slot):
    """int array (2, frames): the visible slots of the home team and of the away team."""
    team_of_slot = np.asarray(team_of_slot)
    return np.stack([visible[:, team_of_slot == team].sum(axis=1) for team in (0, 1)])


def team_moves(spans, succ, team_of_slot, visible, major_share=MAJOR_SHARE):
    """[(slot, start, stop)] of the tracklets that the repair moves to the other team."""
    team_of_slot = np.asarray(team_of_slot)
    counts = team_counts(visible, team_of_slot)
    moves = []
    for chain in chains(len(spans), succ):
        teams = [int(team_of_slot[spans[index][0]]) for index in chain]
        if len(set(teams)) < 2:
            continue
        frames = [0, 0]
        for index, team in zip(chain, teams):
            frames[team] += spans[index][2] - spans[index][1]
        major = int(frames[1] > frames[0])
        share = frames[major] / float(frames[0] + frames[1])
        for index, team in zip(chain, teams):
            slot, start, stop = spans[index]
            if share >= major_share:
                if team != major:
                    moves.append((slot, start, stop))
                continue
            over = (counts[team][start:stop] > MAX_TEAM) & (counts[1 - team][start:stop] < MAX_TEAM)
            if over.mean() > 0.5:
                moves.append((slot, start, stop))
    return sorted(moves)


def apply_moves(tactical, moves):
    """A copy of the tactical dict with each moved tracklet in a new slot of the other team."""
    out = dict(tactical)
    if not moves:
        return out
    team_of_slot = np.asarray(tactical["team_of_slot"])
    n_frames = np.asarray(tactical["px"]).shape[0]
    extra = np.full((n_frames, len(moves)), np.nan, dtype=np.float32)
    arrays = {key: np.concatenate([np.asarray(tactical[key], dtype=np.float32), extra], axis=1)
              for key in ("px", "py", "ps")}
    teams = np.concatenate([team_of_slot, np.zeros(len(moves), dtype=team_of_slot.dtype)])
    base = team_of_slot.size
    for number, (slot, start, stop) in enumerate(moves):
        teams[base + number] = 1 - team_of_slot[slot]
        for values in arrays.values():
            values[start:stop, base + number] = values[start:stop, slot]
            values[start:stop, slot] = np.nan
    out.update(arrays)
    out["team_of_slot"] = teams
    out["n_slots"] = int(teams.size)
    return out


def metres(tactical, pitch):
    """(x, y, visible) of the player slots in metres."""
    x = np.asarray(tactical["px"], dtype=np.float64) * float(pitch[0])
    y = np.asarray(tactical["py"], dtype=np.float64) * float(pitch[1])
    return x, y, np.isfinite(x) & np.isfinite(y)


def repair(tactical, pitch, fps, link_frames=LINK_FRAMES):
    """(repaired tactical dict, moves, spans, links) of the team repair on one match."""
    x, y, visible = metres(tactical, pitch)
    spans = tracklets(visible)
    succ = continuation_links(x, y, spans, fps, link_frames)
    moves = team_moves(spans, succ, tactical["team_of_slot"], visible)
    return apply_moves(tactical, moves), moves, spans, succ


def graded_mask(blocks, n_frames):
    """bool (n_frames,): the frames of the graded blocks [(half, first, last inclusive)]."""
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        mask[int(first):int(last) + 1] = True
    return mask


def count_anatomy(visible, team_of_slot, graded):
    """Label-free numbers that separate a wrong-team tracklet from an extra object."""
    counts = team_counts(visible, team_of_slot)[:, graded]
    out = {}
    for team, name in ((0, "home"), (1, "away")):
        over = counts[team] > MAX_TEAM
        short = counts[1 - team] < MAX_TEAM
        starts, stops = runs(over)
        lengths = (stops - starts).astype(np.float64)
        full = counts.sum(axis=0)[over] == 2 * MAX_TEAM
        out[name] = {
            "over_share": round(float(over.mean()), 4),
            "other_short_when_over": round(float(short[over].mean()), 4) if over.any() else None,
            "other_short_base": round(float(short.mean()), 4),
            "total_22_when_over": round(float(full.mean()), 4) if over.any() else None,
            "over_episodes": int(starts.size),
            "over_episode_frames_p50_p90": ([float(np.percentile(lengths, 50)),
                                             float(np.percentile(lengths, 90))]
                                            if lengths.size else None),
        }
    return out


def report(tactical, pitch, fps, blocks, link_frames=LINK_FRAMES):
    """Label-free summary of the tracklets, the links and the repair of one match."""
    x, y, visible = metres(tactical, pitch)
    team_of_slot = np.asarray(tactical["team_of_slot"])
    graded = graded_mask(blocks, visible.shape[0])
    fixed, moves, spans, succ = repair(tactical, pitch, fps, link_frames)
    cross = sum(1 for a, b in succ.items()
                if team_of_slot[spans[a][0]] != team_of_slot[spans[b][0]])
    mixed = 0
    for chain in chains(len(spans), succ):
        mixed += len({int(team_of_slot[spans[index][0]]) for index in chain}) > 1
    bx = np.asarray(tactical["bx"], dtype=np.float64) * float(pitch[0])
    by = np.asarray(tactical["by"], dtype=np.float64) * float(pitch[1])
    near_ball = 0
    for slot, start, stop in moves:
        dist = np.hypot(x[start:stop, slot] - bx[start:stop], y[start:stop, slot] - by[start:stop])
        near_ball += bool(np.any(dist <= 3.0))
    moved = np.array([stop - start for _, start, stop in moves], dtype=np.float64)
    _, _, fixed_visible = metres(fixed, pitch)
    return {
        "tracklets": len(spans),
        "links": len(succ),
        "links_to_other_team": int(cross),
        "mixed_chains": int(mixed),
        "moves": len(moves),
        "moved_frames": int(moved.sum()),
        "moved_frames_p50_p90": ([float(np.percentile(moved, 50)), float(np.percentile(moved, 90))]
                                 if moved.size else None),
        "moves_with_ball_within_3m": int(near_ball),
        "counts_before": count_anatomy(visible, team_of_slot, graded),
        "counts_after": count_anatomy(fixed_visible, fixed["team_of_slot"], graded),
    }


def inject_episodes(tactical, blocks, fps, pitch, seed, rate_per_min=EPISODES_PER_MIN,
                    median_s=EPISODE_MEDIAN_S, p90_s=EPISODE_P90_S, near_ball_m=0.0):
    """(tactical dict with wrong-team episodes, [(slot, start, stop)] of the episodes).

    Each episode moves one visible player to a new slot of the other team for a log-normal
    time. With near_ball_m above 0 the player is drawn with the weight
    exp(-distance to the ball / near_ball_m), so the episodes sit where the ball is.
    """
    x, y, visible = metres(tactical, pitch)
    n_frames = visible.shape[0]
    graded = np.flatnonzero(graded_mask(blocks, n_frames))
    rng = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    sigma = np.log(float(p90_s) / float(median_s)) / Z90
    count = int(rng.poisson(graded.size / float(fps) / 60.0 * rate_per_min))
    bx = np.asarray(tactical["bx"], dtype=np.float64) * float(pitch[0])
    by = np.asarray(tactical["by"], dtype=np.float64) * float(pitch[1])
    busy = np.zeros_like(visible)
    episodes = []
    for _ in range(count):
        frame = int(graded[rng.randint(graded.size)])
        length = max(2, int(round(np.exp(np.log(median_s) + sigma * rng.randn()) * fps)))
        pick = rng.uniform()
        slots = np.flatnonzero(visible[frame] & ~busy[frame])
        if slots.size == 0:
            continue
        if near_ball_m > 0 and np.isfinite(bx[frame]):
            dist = np.hypot(x[frame, slots] - bx[frame], y[frame, slots] - by[frame])
            weight = np.cumsum(np.exp(-dist / float(near_ball_m)))
            slot = int(slots[min(int(np.searchsorted(weight / weight[-1], pick)), slots.size - 1)])
        else:
            slot = int(slots[min(int(pick * slots.size), slots.size - 1)])
        stop = frame
        while (stop < n_frames and stop < frame + length and visible[stop, slot]
               and not busy[stop, slot]):
            stop += 1
        if stop - frame < 2:
            continue
        busy[max(0, frame - LINK_FRAMES):stop + LINK_FRAMES, slot] = True
        episodes.append((slot, frame, stop))
    return apply_moves(tactical, episodes), episodes


def selftest(tactical, blocks, fps, pitch, seed, rate_per_min, near_ball_m,
             link_frames=LINK_FRAMES):
    """Precision and recall of the repair on injected episodes of one external match."""
    base = tactical["team_of_slot"].size
    bad, episodes = inject_episodes(tactical, blocks, fps, pitch, seed, rate_per_min,
                                    near_ball_m=near_ball_m)
    truth = {(base + number, start, stop) for number, (_, start, stop) in enumerate(episodes)}
    _, moves, _, _ = repair(bad, pitch, fps, link_frames)
    found = set(moves)
    _, clean_moves, _, _ = repair(tactical, pitch, fps, link_frames)
    return {"episodes": len(truth), "moves": len(found), "true_moves": len(found & truth),
            "false_moves": len(found - truth), "missed_episodes": len(truth - found),
            "moves_on_the_clean_match": len(clean_moves)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("report", "selftest"))
    parser.add_argument("--game", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rate-per-min", type=float, default=EPISODES_PER_MIN)
    parser.add_argument("--near-ball-m", type=float, default=0.0)
    parser.add_argument("--link-frames", type=int, default=LINK_FRAMES)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    import games
    from engine import data

    game = data.resolve_game(args.game)
    if args.mode == "selftest" and game == games.GAME:
        raise SystemExit("selftest changes the tracking. It does not run on {}.".format(game))
    tactical = data.load_tactical(game, cache_dir=args.cache_dir)
    pitch, fps, blocks = data.pitch_m(game), data.fps(game), data.graded_blocks(game)
    if args.mode == "report":
        result = report(tactical, pitch, fps, blocks, args.link_frames)
    else:
        seed = zlib.crc32(game.encode("utf-8")) if args.seed is None else args.seed
        result = selftest(tactical, blocks, fps, pitch, seed, args.rate_per_min, args.near_ball_m,
                          args.link_frames)
    result["game"] = game
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
