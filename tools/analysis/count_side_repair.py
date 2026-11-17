"""Count-side team repair at cross-team links (C24 backlog H3, step 4 prototype). Not part of
the submission.

    python tools/analysis/count_side_repair.py --game NAME [--link-frames 3] [--cache-dir DIR]
        [--out OUT.json] [--no-veto]

Label-free. It reads the tracking of one match and prints summary numbers only, so it can run
on the competition match (team rules). It changes nothing in engine/.

The rule. T1 (tracklet_chains.team_moves) moved the minority of a mixed chain. The link count
of tools/analysis/link_structure.py shows that on BAR-ATH most cross-team links join two
tracklets while both teams stay at 11 or fewer, where the team totals cannot say which member
is wrong, and the majority choice raised the over-count. This rule moves a tracklet only where
the totals at the link say which side is wrong:

- Link A (team X) -> B (team Y), gap at most LINK_FRAMES, within the T1 gate, mutual nearest.
- If X shows more than MAX_TEAM tracks on the frame before the link and Y shows at most
  MAX_TEAM on the frame after it, A was the wrong-team tracklet: A moves to Y.
- Else if Y shows more than MAX_TEAM on the frame after the link and X at most MAX_TEAM on
  the frame before it, B is the wrong-team tracklet: B moves to X.
- Otherwise nothing moves. A clean match (no team ever above MAX_TEAM) gets no move.

Veto (default on): a move is dropped if the receiving team would show more than MAX_TEAM
tracks on more than VETO_SHARE of the tracklet's frames after the move.

Only the tracklet next to the link moves, not its chain. Deterministic, numpy only.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.analysis import tracklet_chains as tc  # noqa: E402

MAX_TEAM = tc.MAX_TEAM
VETO_SHARE = 0.05


def count_side_moves(spans, succ, team_of_slot, counts, veto=True, veto_share=VETO_SHARE):
    """[(slot, start, stop)] of the tracklets the rule moves, and a dict of reasons."""
    team_of_slot = np.asarray(team_of_slot)
    reasons = {"ended_side_over": 0, "new_side_over": 0, "both_over": 0, "neither_over": 0,
               "vetoed": 0}
    moves = {}
    for a, b in succ.items():
        slot_a, start_a, stop_a = spans[a]
        slot_b, start_b, stop_b = spans[b]
        x_team, y_team = int(team_of_slot[slot_a]), int(team_of_slot[slot_b])
        if x_team == y_team:
            continue
        before, after = counts[:, stop_a - 1], counts[:, start_b]
        ended_over = before[x_team] > MAX_TEAM and after[y_team] <= MAX_TEAM
        new_over = after[y_team] > MAX_TEAM and before[x_team] <= MAX_TEAM
        if ended_over:
            reasons["ended_side_over"] += 1
            candidate, to_team = (slot_a, start_a, stop_a), y_team
        elif new_over:
            reasons["new_side_over"] += 1
            candidate, to_team = (slot_b, start_b, stop_b), x_team
        elif before[x_team] > MAX_TEAM and after[y_team] > MAX_TEAM:
            reasons["both_over"] += 1
            continue
        else:
            reasons["neither_over"] += 1
            continue
        if veto:
            _, start, stop = candidate
            bad = (counts[to_team, start:stop] + 1 > MAX_TEAM).mean()
            if bad > veto_share:
                reasons["vetoed"] += 1
                continue
        moves[candidate] = to_team
    return sorted(moves), reasons


def run(tactical, pitch, fps, blocks, link_frames, veto):
    x, y, visible = tc.metres(tactical, pitch)
    team_of_slot = np.asarray(tactical["team_of_slot"])
    graded = tc.graded_mask(blocks, visible.shape[0])
    counts = tc.team_counts(visible, team_of_slot)
    spans = tc.tracklets(visible)
    succ = tc.continuation_links(x, y, spans, fps, link_frames)
    moves, reasons = count_side_moves(spans, succ, team_of_slot, counts, veto)
    fixed = tc.apply_moves(tactical, moves)
    _, _, fixed_visible = tc.metres(fixed, pitch)
    bx = np.asarray(tactical["bx"], dtype=np.float64) * float(pitch[0])
    by = np.asarray(tactical["by"], dtype=np.float64) * float(pitch[1])
    near_ball, graded_moves, moved_graded = 0, 0, 0
    for slot, start, stop in moves:
        dist = np.hypot(x[start:stop, slot] - bx[start:stop], y[start:stop, slot] - by[start:stop])
        near_ball += bool(np.any(dist <= 3.0))
        graded_moves += bool(graded[start:stop].any())
        moved_graded += int(graded[start:stop].sum())
    moved = np.array([stop - start for _, start, stop in moves], dtype=np.float64)
    before = tc.count_anatomy(visible, team_of_slot, graded)
    after = tc.count_anatomy(fixed_visible, fixed["team_of_slot"], graded)
    fixed_counts = tc.team_counts(fixed_visible, np.asarray(fixed["team_of_slot"]))
    new_over = {name: int(((fixed_counts[t] > MAX_TEAM) & (counts[t] <= MAX_TEAM) & graded).sum())
                for t, name in ((0, "home"), (1, "away"))}
    return {
        "link_frames": link_frames, "veto": veto, "links": len(succ), "reasons": reasons,
        "moves": len(moves), "moves_touching_graded_frames": graded_moves,
        "moved_frames": int(moved.sum()), "moved_graded_frames": moved_graded,
        "moved_frames_p50_p90": ([float(np.percentile(moved, 50)),
                                  float(np.percentile(moved, 90))] if moved.size else None),
        "moves_with_ball_within_3m": int(near_ball),
        "over_share_before": {k: v["over_share"] for k, v in before.items()},
        "over_share_after": {k: v["over_share"] for k, v in after.items()},
        "over_frames_before": {k: int(round(v["over_share"] * graded.sum()))
                               for k, v in before.items()},
        "over_frames_after": {k: int(round(v["over_share"] * graded.sum()))
                              for k, v in after.items()},
        "new_over_frames_on_receiving_team": new_over,
        "counts_before": before, "counts_after": after,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--link-frames", type=int, default=tc.LINK_FRAMES)
    parser.add_argument("--no-veto", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    from engine import data

    game = data.resolve_game(args.game)
    tactical = data.load_tactical(game, cache_dir=args.cache_dir)
    pitch, fps, blocks = data.pitch_m(game), data.fps(game), data.graded_blocks(game)
    result = run(tactical, pitch, fps, blocks, args.link_frames, not args.no_veto)
    result["game"] = game
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
