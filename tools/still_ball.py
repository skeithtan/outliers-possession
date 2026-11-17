"""DFL still ball removed (round 4, step C of LEARNED_STAGE_PROTOCOL_R4.md). Not part of the submission.

    # an ATD bundle copy of a DFL development match with the still ball marked missing
    python tools/still_ball.py --game DFL-J03WMX --in-root ~/mtah2027/refs --out-root DIR

The rule (label-free, judgement, not tuned): in DFL tracking, on every run of MIN_FRAMES (13,
0.52 s at 25 fps) or more consecutive frames in which the ball position does not change at all
(x and y exactly equal to the previous frame's, as parsed from the bundle), the ball is marked
missing on every frame of that run. DFL keeps the ball at its last position while it is dead
(tools/dfl_bundle.py: "the ball keeps the position of the file on every frame"); real broadcast
tracking loses it. A frame that is already missing ends a run. Nothing else changes: the players,
the frame grid and the frames of shorter still stretches stay as they are.

Two ways to apply it, one rule (still_mask()):

- In memory, where the learned-stage tools load DFL tracking (remove_still(), the switch
  `--dfl-still removed` of tools/learned_spike/build.py, aug_build.py and ballfree_build.py).
- As a bundle copy (build_bundle(), this command), so that engine.run (the rule engine c23) runs on
  the same input with MTAH_DATA_DIR=<out-root>. The metadata XML is copied byte for byte; on a
  removed frame the ball group of the raw line becomes "NaN,NaN" (the missing-ball text of the
  Metrica bundles); every other line keeps its text byte for byte. The tool also writes
  `<raw file>.still.json` with the counts and the md5 of inputs and outputs.

DFL development matches only: the two sealed DFL matches (J03WQQ, J03WR9) are refused before any
file is read, as is every other match. Python 3.9, numpy and pandas (engine.data).
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine import data  # noqa: E402

MIN_FRAMES = 13          # 0.52 s at 25 fps (protocol R4 step C)
SWITCH_VALUES = ("kept", "removed")
DEV_DFL = ("DFL-J03WMX", "DFL-J03WN1", "DFL-J03WOH", "DFL-J03WOY", "DFL-J03WPY")
SEALED = ("DFL-J03WQQ", "DFL-J03WR9")
MISSING_TEXT = "NaN"
SUFFIX = ".still.json"


def check_game(game):
    """Return `game` if it is one of the five DFL development matches; else raise SystemExit."""
    if game in SEALED:
        raise SystemExit("{} is sealed: the still-ball tools never read it".format(game))
    if game not in DEV_DFL:
        raise SystemExit("{}: the still-ball rule applies to the five DFL development matches "
                         "only".format(game))
    return game


def applies(game, still):
    """True if the switch value `still` removes the still ball of `game` (DFL only).

    "kept" (the default everywhere) never applies. "removed" applies to the DFL development
    matches and refuses the sealed ones; other providers are left alone.
    """
    if still not in SWITCH_VALUES:
        raise ValueError("dfl still switch: {!r} not in {}".format(still, SWITCH_VALUES))
    if still == "kept":
        return False
    if game in SEALED:
        raise SystemExit("{} is sealed: the still-ball tools never read it".format(game))
    return game.startswith("DFL-") and check_game(game) is not None


def still_mask(bx, by, min_frames=MIN_FRAMES):
    """bool (N,): the frames of every run of >= min_frames frames with exactly zero displacement.

    A run is a maximal stretch of visible frames (finite x and y) in which each frame's x and y
    equal the previous frame's exactly. Its first frame (the frame the ball arrives at the
    position) belongs to the run, so a run of k equal positions marks k frames.
    """
    bx = np.asarray(bx)
    by = np.asarray(by)
    n = len(bx)
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    vis = np.isfinite(bx) & np.isfinite(by)
    same = np.zeros(n, dtype=bool)
    same[1:] = vis[1:] & vis[:-1] & (bx[1:] == bx[:-1]) & (by[1:] == by[:-1])
    # run starts: a visible frame whose next frame repeats it, and that does not repeat the one
    # before. Label run ids by cumulative starts over "not same".
    start = vis & ~same
    run_id = np.cumsum(start)
    run_id[~vis] = 0
    if not vis.any():
        return out
    lengths = np.bincount(run_id)
    long_run = lengths >= int(min_frames)
    long_run[0] = False
    out = long_run[run_id]
    return out


def remove_still(tac, game, still="removed"):
    """A copy of a tactical dict with the ball missing on the still frames (if the switch applies).

    Return (tactical, mask). With still="kept" or a match outside DFL, the dict itself and an
    all-False mask.
    """
    n = len(tac["bx"])
    if not applies(game, still):
        return tac, np.zeros(n, dtype=bool)
    mask = still_mask(tac["bx"], tac["by"])
    out = dict(tac)
    for key in ("bx", "by"):
        arr = np.array(tac[key], copy=True)
        arr[mask] = np.nan
        out[key] = arr
    return out, mask


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rewrite_lines(lines, meta, mask):
    """Raw ATD lines with the ball x and y written as NaN on the masked frames."""
    cols = [list(meta["ball_channels"]).index(c) for c in (data.X_CHANNEL, data.Y_CHANNEL)]
    out = []
    for i, line in enumerate(lines):
        if not mask[i]:
            out.append(line)
            continue
        frame, players, ball = line.split(":")
        parts = ball.split(",")
        for c in cols:
            parts[c] = MISSING_TEXT
        out.append("{}:{}:{}".format(frame, players, ",".join(parts)))
    return out


def build_bundle(game, in_root, out_root):
    """Write the bundle copy of one DFL development match; return the summary dict."""
    check_game(game)
    in_root, out_root = os.path.abspath(in_root), os.path.abspath(out_root)
    if in_root == out_root:
        raise SystemExit("--out-root must differ from --in-root")
    src = data.asset_paths(game, in_root)
    meta_path, raw_path = src["atd_meta"], src["atd_raw"]
    meta = data._atd_meta(meta_path, game)
    n_frames = data.frame_count(game)
    arrays = data._parse_atd_raw(raw_path, len(meta["team_of_slot"]), n_frames,
                                 meta["player_channels"], meta["ball_channels"])
    mask = still_mask(arrays["bx"], arrays["by"])
    with open(raw_path, "r", encoding="ascii", newline="") as handle:
        lines = handle.read().splitlines()
    new = rewrite_lines(lines, meta, mask)
    rel = {k: os.path.relpath(src[k], in_root) for k in ("atd_meta", "atd_raw")}
    dst = {k: os.path.join(out_root, rel[k]) for k in rel}
    os.makedirs(os.path.dirname(dst["atd_raw"]), exist_ok=True)
    _write(dst["atd_raw"], "\n".join(new) + "\n")
    with open(meta_path, "rb") as handle:
        _write(dst["atd_meta"], handle.read())
    vis = np.isfinite(arrays["bx"]) & np.isfinite(arrays["by"])
    runs = int(np.sum(mask[1:] & ~mask[:-1]) + (1 if mask[:1].any() else 0))
    summary = {"game": game, "rule": "runs of >= {} frames with zero ball displacement -> "
                                     "missing".format(MIN_FRAMES),
               "min_frames": MIN_FRAMES, "frames": int(n_frames),
               "ball_visible_before": int(vis.sum()), "frames_removed": int(mask.sum()),
               "runs_removed": runs, "inputs_md5": {k: _md5(src[k]) for k in rel},
               "outputs": rel, "outputs_md5": {k: _md5(dst[k]) for k in rel}}
    _write(dst["atd_raw"] + SUFFIX, json.dumps(summary, sort_keys=True, indent=1) + "\n")
    return summary


def _write(path, content):
    tmp = "{}.tmp-{}".format(path, os.getpid())
    mode = "wb" if isinstance(content, bytes) else "w"
    kwargs = {} if mode == "wb" else {"encoding": "ascii", "newline": ""}
    with open(tmp, mode, **kwargs) as handle:
        handle.write(content)
    os.replace(tmp, path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--game", required=True, nargs="+")
    p.add_argument("--in-root", required=True, help="MTAH_DATA_DIR of the raw DFL bundles")
    p.add_argument("--out-root", required=True, help="new root of the copies")
    a = p.parse_args(argv)
    for g in a.game:
        check_game(g)
    for g in a.game:
        print(json.dumps(build_bundle(g, os.path.expanduser(a.in_root),
                                      os.path.expanduser(a.out_root)), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
