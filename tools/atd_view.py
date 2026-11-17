"""ATD-like smoothing view of an ATD-style bundle (P9-A, part 4). Not part of the submission.

    python tools/atd_view.py --game DFL-J03WMX --in-root ~/mtah2027/refs \
        --out-root ~/mtah2027/refs-smooth
    python tools/atd_view.py --game METRICA-S3 --in-root ~/mtah2027/sample-data/data \
        --out-root ~/mtah2027/refs-smooth --sigma 9.5 --players off

The competition ATD ball is smoothed by a symmetric kernel. RESEARCH-2 M13 measured it on 46
isolated strong acceleration peaks of the full match: half-width at half-maximum 11 frames,
sigma about 9.5 frames. The external references carry a raw ball. This tool writes a copy of
one bundle in which the ball track is convolved with a Gaussian kernel of that sigma, so that
the engine can run on a ball track that looks like the competition match's.

The tool reads the `atd_meta` and `atd_raw` files that engine.data.asset_paths() gives for the
match under `--in-root`. It writes them under `--out-root` with the same relative paths. So
`MTAH_DATA_DIR=<out-root> python -m engine.run --game <name>` runs the engine on the view, and
games.py does not change. The metadata XML is copied byte for byte. The tool also writes
`<raw file>.view.json` with the sigma, the player switch and the md5 of the inputs.

How the kernel is applied:

* In metres, with the pitch size of games.py. The tool converts back to normalised 0-1
  coordinates only when it writes a value.
* Within each run of the track only. A run ends at a missing frame and at the edge of a graded
  block (engine.data.graded_blocks()), so the kernel never bridges a gap or a half boundary.
  For the players a run also ends at the first frame of a DataFormatSpecification block,
  because a slot can hold another player after it.
* The kernel is truncated at TRUNCATE_SIGMAS sigmas and renormalised over the frames of the
  run that it covers. So a constant stretch stays constant: a frozen dead ball stays frozen
  away from the frames where it moves, and a run edge is not pulled toward zero.
* A run is not cut at a large one-frame step. The one-frame jump of about 6 m that the DFL ball
  makes at a restart is smoothed like any other displacement, as a smoothing provider would
  smooth it. The engine's teleport rule (3 m in one frame) then no longer sees it.
* The view adds no gaps, blanks no frame and inserts no slot birth.

`--players on` smooths the player positions with the same kernel. The player speed channel is
left as it is. P9-A4 measured that the BAR-ATH players are not smoothed like the ball
(docs/plan/calibration/ATD_VIEW.md), so the default is off.

Deterministic: no clock, no randomness, fixed number format. Python 3.9, numpy and pandas.
Refuses sealed matches unless `--allow-sealed` is explicit. Use that flag only at the final selection
(docs/plan/calibration/SEALED_OPENING.md). Prints summary statistics only.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import games  # noqa: E402
from engine import data  # noqa: E402

# Sigma of the ATD ball kernel in frames. RESEARCH-2 M13 (LEARNINGS C8): HWHM 11 frames, sigma
# about 9.5 frames, measured on 46 isolated strong acceleration peaks of the BAR-ATH ball.
ATD_SIGMA_FRAMES = 9.5
# The kernel is cut at this many sigmas (the Gaussian weight there is 3e-4 of the centre).
TRUNCATE_SIGMAS = 4.0
# Number format of a written position, the format of tools/dfl_bundle.py and metrica_bundle.py.
POSITION_FORMAT = "{:.5f}"
# The two sealed DFL matches of docs/plan/calibration/SPLITS.md, as games.py names them.
SEALED_GAMES = ("DFL-J03WQQ", "DFL-J03WR9")
VIEW_SUFFIX = ".view.json"


def gaussian_kernel(sigma):
    """Unnormalised Gaussian weights over -r..r frames, r = ceil(TRUNCATE_SIGMAS * sigma)."""
    radius = int(math.ceil(TRUNCATE_SIGMAS * sigma))
    lags = np.arange(-radius, radius + 1, dtype=np.float64)
    return np.exp(-0.5 * (lags / sigma) ** 2)


def run_starts(visible, breaks):
    """bool (N,): True on the first frame of each run of `visible`, a run cut at `breaks`."""
    visible = np.asarray(visible, dtype=bool)
    prev = np.concatenate([[False], visible[:-1]])
    return visible & (~prev | np.asarray(breaks, dtype=bool))


def smooth_runs(values, visible, breaks, sigma):
    """Gaussian-smoothed copy of a (N,) series, run by run (see the module docstring).

    Frames outside `visible` keep their value. Each run is convolved with the truncated kernel
    and divided by the kernel mass that falls inside the run.
    """
    values = np.asarray(values, dtype=np.float64)
    out = values.copy()
    if sigma <= 0:
        return out
    kernel = gaussian_kernel(sigma)
    radius = (kernel.size - 1) // 2
    visible = np.asarray(visible, dtype=bool)
    breaks = np.asarray(breaks, dtype=bool)
    starts = np.flatnonzero(run_starts(visible, breaks))
    goes_on = np.zeros(values.size, dtype=bool)
    goes_on[:-1] = visible[1:] & ~breaks[1:]
    stops = np.flatnonzero(visible & ~goes_on) + 1
    for start, stop in zip(starts, stops):
        seg = values[start:stop]
        num = np.convolve(seg, kernel, mode="full")[radius:radius + seg.size]
        den = np.convolve(np.ones(seg.size), kernel, mode="full")[radius:radius + seg.size]
        out[start:stop] = num / den
    return out


def block_breaks(n_frames, game):
    """bool (N,): True at the first frame of each graded block and at the frame after it."""
    breaks = np.zeros(n_frames + 1, dtype=bool)
    for _, first, last in data.graded_blocks(game):
        breaks[first] = True
        breaks[last + 1] = True
    return breaks[:n_frames]


def spec_breaks(meta_path, n_frames):
    """bool (N,): True at the first video frame of every DataFormatSpecification block."""
    breaks = np.zeros(n_frames, dtype=bool)
    for spec in ET.parse(meta_path).getroot().iter("DataFormatSpecification"):
        start = spec.get("startFrame")
        if start is not None and 1 <= int(start) <= n_frames:
            breaks[int(start) - 1] = True
    return breaks


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group(text, columns, values):
    """`text` (one comma-separated object group) with `values` written at `columns`."""
    parts = text.split(",")
    for column, value in zip(columns, values):
        parts[column] = POSITION_FORMAT.format(value)
    return ",".join(parts)


def rewrite_lines(lines, meta, ball_xy, player_xy=None):
    """New raw lines with the smoothed positions written in place of the old ones.

    `ball_xy` is (x, y, changed) of normalised (N,) arrays and a bool (N,) mask of the frames to
    write. `player_xy` is the same with (N, S) arrays, or None to keep every player group. A
    frame that is not in a mask keeps its text byte for byte.
    """
    ball_cols = [list(meta["ball_channels"]).index(c) for c in (data.X_CHANNEL, data.Y_CHANNEL)]
    player_cols = [list(meta["player_channels"]).index(c)
                   for c in (data.X_CHANNEL, data.Y_CHANNEL)]
    bx, by, bmask = ball_xy
    out = []
    for i, line in enumerate(lines):
        frame, players, ball = line.split(":")
        if bmask[i]:
            ball = _group(ball, ball_cols, (bx[i], by[i]))
        if player_xy is not None and player_xy[2][i].any():
            px, py, pmask = player_xy
            groups = players.split(";")
            for s in np.flatnonzero(pmask[i]):
                groups[s] = _group(groups[s], player_cols, (px[i, s], py[i, s]))
            players = ";".join(groups)
        out.append("{}:{}:{}".format(frame, players, ball))
    return out


def build_view(game, in_root, out_root, sigma=ATD_SIGMA_FRAMES, players=False, *,
               allow_sealed=False):
    """Write one smoothed bundle and return summary statistics.

    Set `allow_sealed` only at the final selection. Refuse sealed matches before reading any asset
    unless that argument is true. Record an allowed sealed opening in the view manifest.
    """
    if game in SEALED_GAMES and not allow_sealed:
        raise SystemExit("{} is sealed (docs/plan/calibration/SPLITS.md). Pass --allow-sealed "
                         "only at the final selection. No view was built."
                         .format(game))
    src = data.asset_paths(game, in_root)
    meta_path, raw_path = src["atd_meta"], src["atd_raw"]
    for path in (meta_path, raw_path):
        if not os.path.isfile(path):
            raise IOError("missing input {}".format(path))
    rel = {name: os.path.relpath(src[name], in_root) for name in ("atd_meta", "atd_raw")}
    if any(r.startswith("..") for r in rel.values()):
        raise ValueError("the assets of {} are not under {}".format(game, in_root))

    meta = data._atd_meta(meta_path, game)
    n_frames = data.frame_count(game)
    arrays = data._parse_atd_raw(raw_path, len(meta["team_of_slot"]), n_frames,
                                 meta["player_channels"], meta["ball_channels"])
    length, width = data.pitch_m(game)
    breaks = block_breaks(n_frames, game)

    bx_m = arrays["bx"].astype(np.float64) * length
    by_m = arrays["by"].astype(np.float64) * width
    bvis = np.isfinite(bx_m) & np.isfinite(by_m)
    sbx = smooth_runs(bx_m, bvis, breaks, sigma) / length
    sby = smooth_runs(by_m, bvis, breaks, sigma) / width
    ball_xy = (sbx, sby, bvis)

    player_xy = None
    if players:
        pbreaks = breaks | spec_breaks(meta_path, n_frames)
        px_m = arrays["px"].astype(np.float64) * length
        py_m = arrays["py"].astype(np.float64) * width
        pvis = np.isfinite(px_m) & np.isfinite(py_m)
        spx, spy = px_m.copy(), py_m.copy()
        for s in range(px_m.shape[1]):
            spx[:, s] = smooth_runs(px_m[:, s], pvis[:, s], pbreaks, sigma)
            spy[:, s] = smooth_runs(py_m[:, s], pvis[:, s], pbreaks, sigma)
        player_xy = (spx / length, spy / width, pvis)

    with open(raw_path, "r", encoding="ascii", newline="") as handle:
        lines = handle.read().splitlines()
    new_lines = rewrite_lines(lines, meta, ball_xy, player_xy)

    dst = {name: os.path.join(out_root, rel[name]) for name in rel}
    os.makedirs(os.path.dirname(dst["atd_raw"]), exist_ok=True)
    os.makedirs(os.path.dirname(dst["atd_meta"]), exist_ok=True)
    _write_atomic(dst["atd_raw"], "\n".join(new_lines) + "\n")
    with open(meta_path, "rb") as handle:
        _write_atomic(dst["atd_meta"], handle.read())

    shift = np.hypot((sbx - arrays["bx"]) * length, (sby - arrays["by"]) * width)[bvis]
    summary = {
        "game": game, "sigma_frames": sigma, "truncate_sigmas": TRUNCATE_SIGMAS,
        "players": bool(players), "frames": int(n_frames),
        "ball_visible_frames": int(bvis.sum()),
        "ball_runs": int(run_starts(bvis, breaks).sum()),
        "ball_shift_m_p50_p99_max": [round(float(v), 3) for v in
                                     (np.quantile(shift, 0.5), np.quantile(shift, 0.99),
                                      shift.max())] if shift.size else None,
        "inputs_md5": {name: _md5(src[name]) for name in ("atd_meta", "atd_raw")},
        "outputs": {name: rel[name] for name in sorted(rel)},
    }
    if game in SEALED_GAMES:
        summary["sealed_opening"] = True
    _write_atomic(dst["atd_raw"] + VIEW_SUFFIX, json.dumps(summary, sort_keys=True, indent=1)
                  + "\n")
    return summary


def _write_atomic(path, content):
    """Write text or bytes to `path` through a temporary file and os.replace."""
    tmp = "{}.tmp-{}".format(path, os.getpid())
    mode = "wb" if isinstance(content, bytes) else "w"
    kwargs = {} if mode == "wb" else {"encoding": "ascii", "newline": ""}
    try:
        with open(tmp, mode, **kwargs) as handle:
            handle.write(content)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", required=True, choices=sorted(games.GAMES))
    parser.add_argument("--in-root", required=True, help="MTAH_DATA_DIR of the raw bundle")
    parser.add_argument("--out-root", required=True, help="root of the view")
    parser.add_argument("--sigma", type=float, default=ATD_SIGMA_FRAMES,
                        help="kernel sigma in frames (default: RESEARCH-2 M13)")
    parser.add_argument("--players", choices=("on", "off"), default="off",
                        help="smooth the player positions too")
    parser.add_argument("--allow-sealed", action="store_true",
                        help="build a sealed match view (the final selection only)")
    args = parser.parse_args(argv)
    in_root = os.path.abspath(os.path.expanduser(args.in_root))
    out_root = os.path.abspath(os.path.expanduser(args.out_root))
    if in_root == out_root:
        raise SystemExit("--out-root must differ from --in-root")
    summary = build_view(args.game, in_root, out_root, args.sigma, args.players == "on",
                         allow_sealed=args.allow_sealed)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
