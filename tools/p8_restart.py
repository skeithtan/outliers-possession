"""Measure a match's restart delay from its tracking, then write the P8-R1 comparison.

    python -m tools.p8_restart --game METRICA-S3 --out-dir /tmp/p8-r1/s3
    python -m tools.p8_restart --game METRICA-S3 --out-dir /tmp/p8-r1/s3-matched \
        --data-root /path/to/matched --cache-dir /tmp/p8-r1/cache-matched

Set MTAH_DATA_DIR to the input view, or pass --data-root with a separate --cache-dir.
The tool reads no reference labels and computes no scores.
It writes baseline.csv, candidate.csv, both full config snapshots and summary.json outside
the repository. Both CSVs pass the official format validator. Use tools/score_game.py afterwards
for the declared comparison.

The estimator follows M16 in tools/analysis/r2_restart_combo.py. The P8 design memo defines
its boundaries, ties and fallback. This experiment does not change the submission pipeline.
"""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import config, data, possession, submission  # noqa: E402

# M16's peak_end() searches two seconds beyond the rise start.
PEAK_SEARCH_S = 2.0
# M16 ignores a descending step until speed exceeds the starting speed by this amount.
PEAK_RISE_MPS = 0.5
# SPLITS.md reserves these two matches for the scheduled final check.
SEALED_GAMES = ("DFL-J03WQQ", "DFL-J03WR9")


def _support_bounds(speed, visible, run_id):
    """Return inclusive bounds of each finite visible run, or -1 at invalid frames."""
    n_frames = speed.size
    left = np.full(n_frames, -1, dtype=np.int64)
    right = np.full(n_frames, -1, dtype=np.int64)
    valid = visible & np.isfinite(speed) & (run_id >= 0)
    if n_frames == 0:
        return left, right
    joined = valid[1:] & valid[:-1] & (run_id[1:] == run_id[:-1])
    starts = np.flatnonzero(valid & ~np.concatenate(([False], joined)))
    stops = np.flatnonzero(valid & ~np.concatenate((joined, [False])))
    for first, last in zip(starts, stops):
        left[first:last + 1] = first
        right[first:last + 1] = last
    return left, right


def _peak_end(speed, frame, last):
    """Return the first qualifying speed peak, or the first maximum in the search interval."""
    for at in range(frame, last):
        if speed[at + 1] < speed[at] and speed[at] > speed[frame] + PEAK_RISE_MPS:
            return at
    return frame + int(np.argmax(speed[frame:last + 1]))


def estimate_restart_timing(inputs, cfg=None):
    """Return the P8-R1 median delay and aggregate diagnostics, without reference labels.

    Use unique unshifted kick frames, including unconfirmed candidates. Require one graded
    block and one finite visible run across every impulse window. Keep the earliest maximum.
    Round the median with Python's ties-to-even rule. With no sample, retain the config delay.
    """
    cfg = cfg or config.Config.default()
    ball = inputs["ball"]
    speed = np.asarray(ball["speed"], dtype=np.float64)
    visible = np.asarray(ball["visible"], dtype=bool)
    run_id = np.asarray(ball["run_id"])
    if speed.ndim != 1 or visible.shape != speed.shape or run_id.shape != speed.shape:
        raise ValueError("speed, visible and run_id must be equal-length vectors")
    fps = float(ball["fps"])
    half_window = int(cfg.CONTACT_HALF_WINDOW)
    if not np.isfinite(fps) or fps <= 0 or half_window < 1:
        raise ValueError("fps and CONTACT_HALF_WINDOW must be positive")
    horizon = max(1, int(round(PEAK_SEARCH_S * fps)))
    left, right = _support_bounds(speed, visible, run_id)
    table = inputs["restarts"]
    kick_frames = table.loc[table["kind"] == "kick", "frame"].to_numpy(dtype=np.int64)
    frames = np.unique(kick_frames)
    rejected = {"outside_graded": 0, "invalid_ball": 0, "no_impulse_support": 0}
    offsets = []
    for raw_frame in frames:
        frame = int(raw_frame)
        block = next(((int(first), int(last)) for _, first, last in inputs["blocks"]
                      if first <= frame <= last), None)
        if block is None:
            rejected["outside_graded"] += 1
            continue
        if not 0 <= frame < speed.size or left[frame] < 0:
            rejected["invalid_ball"] += 1
            continue
        first = max(block[0], int(left[frame]))
        last = min(block[1], int(right[frame]))
        search_end = min(last, frame + horizon)
        peak = min(last, max(_peak_end(speed, frame, search_end), frame + 1))
        centers = np.arange(max(frame, first + half_window),
                            min(peak, last - half_window) + 1, dtype=np.int64)
        if centers.size == 0:
            rejected["no_impulse_support"] += 1
            continue
        impulse = speed[centers + half_window] - speed[centers - half_window]
        center = int(centers[int(np.argmax(impulse))])
        offsets.append(center - frame)

    fallback = len(offsets) == 0
    percentiles = (None if fallback else
                   dict(zip(("p10", "p50", "p90"),
                            [float(value) for value in np.percentile(offsets, [10, 50, 90])])))
    delay = int(cfg.DELTA_RESTART_KICK) if fallback else int(round(percentiles["p50"]))
    return {
        "method": "p8_r1_m16_match_median",
        "baseline_delay_frames": int(cfg.DELTA_RESTART_KICK),
        "delay_frames": delay,
        "fallback": fallback,
        "kick_rows": int(kick_frames.size),
        "unique_kick_frames": int(frames.size),
        "duplicate_kick_rows": int(kick_frames.size - frames.size),
        "eligible_kicks": len(offsets),
        "rejected": rejected,
        "offset_percentiles_frames": percentiles,
        "contact_half_window_frames": half_window,
        "peak_search_s": PEAK_SEARCH_S,
        "peak_rise_mps": PEAK_RISE_MPS,
    }


def candidate_config(inputs, cfg=None):
    """Return the config with the measured kick delay, plus the measurement diagnostics."""
    cfg = cfg or config.Config.default()
    estimate = estimate_restart_timing(inputs, cfg)
    return dataclasses.replace(cfg, DELTA_RESTART_KICK=estimate["delay_frames"]), estimate


def _external_output(path):
    """Resolve an output folder and reject repository paths, including symlink aliases."""
    resolved = Path(path).expanduser().resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError("write P8 predictions outside the repository")
    return resolved


def _selected_restarts(inputs, cfg):
    """Return the selected unshifted restarts, before the timing offset applies."""
    _, selected = possession.dead_intervals(
        inputs["n_frames"], inputs["onsets"], inputs["e_dead"], inputs["restarts"],
        inputs["blocks"], inputs["kickoffs"], cfg, inputs["fps"], inputs.get("holds"))
    return selected


def _selection_hash(selected):
    """Fingerprint the local selected restart list without exposing its rows."""
    encoded = json.dumps(selected, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_comparison(game, out_dir, cfg=None, estimate_only=False):
    """Write the declared comparison and return summaries. Refuse the sealed matches."""
    game = data.resolve_game(game)
    if game in SEALED_GAMES:
        raise ValueError("P8-R1 does not open sealed matches")
    out_dir = _external_output(out_dir)
    cfg = cfg or config.Config.default()
    inputs = possession.prepare(cfg, game=game)
    candidate, estimate = candidate_config(inputs, cfg)
    report = {"game": game, "estimate": estimate,
              "baseline_config_hash": cfg.config_hash(),
              "candidate_config_hash": candidate.config_hash()}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, snapshot in (("baseline", cfg), ("candidate", candidate)):
        (out_dir / (name + "_config.json")).write_text(
            json.dumps(snapshot.as_dict(), indent=2, sort_keys=True) + "\n")
    if not estimate_only:
        base_selection = _selected_restarts(inputs, cfg)
        candidate_selection = _selected_restarts(inputs, candidate)
        report["selection"] = {
            "baseline_sha256": _selection_hash(base_selection),
            "candidate_sha256": _selection_hash(candidate_selection),
            "unchanged": base_selection == candidate_selection,
            "selected_restart_count": len(base_selection),
            "detection_key_unchanged": (possession.detection_key(cfg) ==
                                        possession.detection_key(candidate)),
        }
        labels = {}
        for name, run_cfg in (("baseline", cfg), ("candidate", candidate)):
            labels[name] = possession.run(run_cfg, inputs=inputs, game=game)
            path = out_dir / (name + ".csv")
            submission.write_submission(labels[name], str(path), game=game)
            submission.validate(str(path), rows=inputs["n_frames"])
            report[name] = {"md5": submission.md5_of(path), "validated": True,
                            "summary": submission.summarize(labels[name], cfg=run_cfg, game=game)}
        report["changed_frames"] = int(np.sum(labels["baseline"] != labels["candidate"]))
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None):
    """Run one comparison on the requested data root. Print aggregate diagnostics only."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", required=True, choices=sorted(data.games.GAMES))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--data-root", help="input view root, otherwise use MTAH_DATA_DIR")
    parser.add_argument("--cache-dir", help="separate cache directory for this input view")
    parser.add_argument("--config", help="base config overrides")
    parser.add_argument("--estimate-only", action="store_true", help="write no predictions")
    args = parser.parse_args(argv)
    if args.data_root and not args.cache_dir:
        parser.error("--data-root requires --cache-dir, because the cache key omits the data root")
    old_root = data.games.DATA_DIR
    old_env = os.environ.get("MTAH_DATA_DIR")
    old_cache = data.DEFAULT_CACHE_DIR
    try:
        if args.data_root:
            root = str(Path(args.data_root).expanduser().resolve())
            os.environ["MTAH_DATA_DIR"] = root
            # games reads MTAH_DATA_DIR at import, before argparse reads the CLI arguments.
            data.games.DATA_DIR = root
        if args.cache_dir:
            data.DEFAULT_CACHE_DIR = str(_external_output(args.cache_dir))
        cfg = config.load(args.config) if args.config else config.Config.default()
        report = run_comparison(args.game, args.out_dir, cfg, args.estimate_only)
    finally:
        data.games.DATA_DIR = old_root
        data.DEFAULT_CACHE_DIR = old_cache
        if old_env is None:
            os.environ.pop("MTAH_DATA_DIR", None)
        else:
            os.environ["MTAH_DATA_DIR"] = old_env
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
