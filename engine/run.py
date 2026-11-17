"""One command to produce a submission (PLAN.md Sections 3.8 and 4, card P0-C).

    python -m engine.run --out evaluation/results/<name>.csv [--mode engine|baseline50|m1|m2]
                         [--config path] [--game name]

Writes the CSV, runs the organisers' validator on it, prints the full-match summary with the
PLAN.md 5.3 vetoes, and prints the file's md5.

  --mode engine      (default) the submission path: engine.possession.run(cfg), milestone M3
                     (card P1-C): event and dead-ball evidence, first-order Viterbi decoder
  --mode baseline50  starter/baseline.py's baseline_smooth50 reimplemented on engine.data arrays;
                     must equal evaluation/results/baseline_smooth50.csv byte-for-byte (Phase 0 gate)
  --mode m1          milestone M1 (card P1-B): baseline owner + engine.deadball overlay
                     (engine.possession.run_m1)
  --mode m2          milestone M2 (card P1-A): owner from engine.events, M1 dead-ball state
                     machine on top (engine.possession.run_m2)
  --game name        the match of games.py to label (default: games.GAME). The engine reads
                     every match constant from that entry. Write the CSV of another match
                     outside evaluation/results/.
"""

import argparse
import sys

import numpy as np

from engine import data, submission

# starter/baseline.py VARIANTS: ("baseline_smooth50", 50), half-width of the majority window.
BASELINE_WINDOW = 50
HOME, AWAY, NOBODY = 0, 1, -1  # starter/baseline.py team codes

def nearest_team_codes(tactical, pitch):
    """Per-frame team code (HOME/AWAY) of the player nearest the ball; NOBODY if no ball.

    Mirrors starter/baseline.py::nearest_player_labels: both axes scaled to metres, squared
    distance dx*dx + dy*dy in float64 with the same operation order, first slot wins on ties
    (strict `<`, slots in order == argmin), missing players skipped, missing ball -> NOBODY.

    engine.data stores float32; the baseline uses float64 of the text. The plain cast is enough:
    measured for P0-C, recovering the exact text values changes the nearest team on 0 frames.
    """
    length, width = pitch
    bx, by = tactical["bx"].astype(np.float64), tactical["by"].astype(np.float64)
    px, py = tactical["px"].astype(np.float64), tactical["py"].astype(np.float64)
    if not np.array_equal(np.isnan(px), np.isnan(py)) or \
            not np.array_equal(np.isnan(bx), np.isnan(by)):
        # The baseline only tests x for NaN; a lone NaN y would poison its running minimum.
        raise ValueError("ATD has x/y missing-ness mismatch; baseline semantics undefined")

    ball_x = (bx * length)[:, None]
    ball_y = (by * width)[:, None]
    dx = px * length - ball_x
    dy = py * width - ball_y
    distance = dx * dx + dy * dy
    distance[np.isnan(distance)] = np.inf
    nearest = np.argmin(distance, axis=1)

    codes = tactical["team_of_slot"][nearest].astype(np.int8)
    codes[np.isnan(bx) | np.isinf(distance.min(axis=1))] = NOBODY
    return codes


def smooth_majority(codes, window):
    """starter/baseline.py::smooth: centred majority vote +/- window, ties to H, D if nobody."""
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))
    home_cumulative = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_cumulative = np.concatenate(([0], np.cumsum(codes == AWAY)))
    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    home = home_cumulative[high] - home_cumulative[low]
    away = away_cumulative[high] - away_cumulative[low]
    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels


def baseline50_labels(game=None):
    """Labels of the organisers' baseline_smooth50, from engine.data (no X anywhere).

    `game` None means games.GAME.
    """
    codes = nearest_team_codes(data.load_tactical(game), data.pitch_m(game))
    return smooth_majority(codes, BASELINE_WINDOW)


def _resolve_config(path):
    """Config for engine.possession.run. Config files are a later card; guarded import."""
    if path is None:
        return None
    try:
        from engine import config
    except Exception:  # absent, or mid-edit by another card: never crash the CLI on it
        print("note: --config {} ignored (engine.config not available yet)".format(path))
        return None
    loader = getattr(config, "load", None) or getattr(config, "load_config", None)
    return loader(path) if callable(loader) else path


def main(argv=None):
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="CSV to write")
    parser.add_argument("--mode", choices=("engine", "baseline50", "m1", "m2"), default="engine")
    parser.add_argument("--config", default=None, help="config file (engine mode)")
    parser.add_argument("--game", default=None, choices=sorted(data.games.GAMES),
                        help="match of games.py (default: games.GAME)")
    args = parser.parse_args(argv)
    game = data.resolve_game(args.game)

    if args.mode == "baseline50":
        labels = baseline50_labels(game)
    elif args.mode == "m1":
        from engine import possession
        labels = possession.run_m1(_resolve_config(args.config), game=game)
    elif args.mode == "m2":
        from engine import possession
        labels = possession.run_m2(_resolve_config(args.config), game=game)
    else:
        from engine import possession
        labels = possession.run(_resolve_config(args.config), game=game)

    path = submission.write_submission(labels, args.out, game=game)
    if game == data.games.GAME:
        print(submission.validate(path).rstrip())
    else:
        # evaluation/validate_submission.py checks the length and graded frames of games.GAME
        # only, so it cannot check the CSV of another match.
        print("validator skipped: it checks {} only".format(data.games.GAME))
    print()
    print(submission.format_summary(submission.summarize(labels, game=game)))
    print()
    print("wrote {}".format(path))
    print("md5   {}".format(submission.md5_of(path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
