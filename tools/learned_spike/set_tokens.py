"""Round 8 arm S: player and ball tokens per 5 fps step, and their token cache. Not part of the submission.

    python -m tools.learned_spike.set_tokens build --labels CA --engine-cache DIR --out TOK \
        --games G ... [--aug-k 4] [--views train val]

A step (BIN = 5 frames at 25 fps) holds N_TOKENS = 23 tokens: up to 11 tokens of the home team
(slots 0-10), up to 11 of the away team (11-21) and the ball (22). Each token has N_COLS = 11
columns:

    0 x, 1 y          position in metres from the pitch centre, / 10
    2 vx, 3 vy        velocity, / 5 for a player, / 10 for the ball (m/s, clipped)
    4 present         player: visible in the step; ball: share of visible frames in the step
    5 team            +1 home ("own", attacks +x after orientation), -1 away, 0 ball
    6 dx, 7 dy        player minus ball, / 10 (0 when the ball is not seen in the step)
    8 dist / t_since  player: distance to the ball, clipped 30 m, / 10 (0 when the ball is
                      not seen); ball: log1p(seconds since the ball was last seen, clipped 60 s)
    9 ball_missing    1 when the ball is not seen in any frame of the step
   10 is_ball         1 on the ball token

The tracks and the orientation are those of feature version stage50-v2t
(features.oriented_arrays(away_fallback=True, compare_teams=True); the ball cleaned by
tracks.clean_ball with Config.default(); no player track cleaning, as in v2t). Per step a
column is the mean over the frames of the step where it is defined; a player slot is present
when visible in at least one frame. When more than 11 slots of a team are present (ghosts of
the defect copies), the 11 nearest the ball are kept (stable sort by slot).

Symmetries (transform()): a team swap exchanges the two blocks of player tokens, negates the
team flag, and turns the pitch 180 degrees (x, y, vx, vy, dx, dy negated); a mirror negates
y, vy and dy. Both are exact relabellings of the tactical input (tests/test_set_tokens.py).

The cache: TOK/<view>/<game>.npy float16 (bins, 23, 11), view "train" (DFL from input
c_train_j, others raw), "val" (DFL from input c, others raw; the scoring input) and "k<k>"
(defect copy k of augment.augment(version r3) on the train input: the first copies of the
recipe's aug2tj folder). Labels are read from the existing caches by set_train.py.

Matches pass build.check (no competition match, no sealed match). Python 3.9, numpy.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from engine import data, tracks  # noqa: E402
from engine.config import Config  # noqa: E402
from tools.learned_spike import features  # noqa: E402

BIN = features.BIN
FPS = features.FPS
PER_TEAM = 11
N_TOKENS = 2 * PER_TEAM + 1
BALL = N_TOKENS - 1
N_COLS = 11
CLIP_D = 30.0
CLIP_V = 12.0
CLIP_BV = 35.0
X_COLS = (0, 2, 6)
Y_COLS = (1, 3, 7)
TEAM_COL, PRESENT_COL = 5, 4


def _bins(a):
    """(nb, BIN, ...) view of a (n, ...) array, tail padded with the last frame."""
    n = a.shape[0]
    nb = -(-n // BIN)
    pad = nb * BIN - n
    if pad:
        a = np.concatenate([a, np.repeat(a[-1:], pad, axis=0)])
    return a.reshape((nb, BIN) + a.shape[1:])


def _masked_mean(values, ok):
    """Mean over axis 1 of `values` where ok, 0 where no frame is ok."""
    cnt = ok.sum(axis=1)
    s = np.where(ok, values, 0.0).sum(axis=1)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0), cnt


def tokens(tac, game):
    """float32 (bins, N_TOKENS, N_COLS) tokens of a tactical dict (see the module docstring)."""
    length, width = data.pitch_m(game)
    px, py, bx, by = features.oriented_arrays(tac, game, away_fallback=True, compare_teams=True)
    team = np.asarray(tac["team_of_slot"])
    ball = tracks.clean_ball(bx, by, (length, width), Config.default(), fps=FPS)
    vis = np.asarray(ball["visible"], dtype=bool)
    bxm = features._fill_nearest(np.nan_to_num(ball["x"]), vis)
    bym = features._fill_nearest(np.nan_to_num(ball["y"]), vis)
    bvx = np.where(vis, np.nan_to_num(ball["vx"]), 0.0)
    bvy = np.where(vis, np.nan_to_num(ball["vy"]), 0.0)
    since = np.log1p(np.minimum(features._time_since(vis), 60 * FPS) / FPS)

    pxm, pym = px * length, py * width
    pvis = np.isfinite(pxm) & np.isfinite(pym)
    pvx, pvy = tracks._difference_velocity(pxm, pym, pvis, FPS)

    # Steps.
    bvis_b = _bins(vis)
    bpres = bvis_b.mean(axis=1)
    bseen = bvis_b.any(axis=1)
    bx_s = _bins(bxm).mean(axis=1) - length / 2.0
    by_s = _bins(bym).mean(axis=1) - width / 2.0
    bvx_s = _bins(bvx).mean(axis=1)
    bvy_s = _bins(bvy).mean(axis=1)
    since_s = _bins(since).mean(axis=1)
    nb = bx_s.shape[0]

    pv_b = _bins(pvis)
    x_s, cnt = _masked_mean(_bins(np.nan_to_num(pxm)), pv_b)
    y_s, _ = _masked_mean(_bins(np.nan_to_num(pym)), pv_b)
    vok = _bins(pvis & np.isfinite(pvx) & np.isfinite(pvy))
    vx_s, _ = _masked_mean(_bins(np.nan_to_num(pvx)), vok)
    vy_s, _ = _masked_mean(_bins(np.nan_to_num(pvy)), vok)
    present = cnt > 0
    x_s = x_s - length / 2.0
    y_s = y_s - width / 2.0
    dx = x_s - bx_s[:, None]
    dy = y_s - by_s[:, None]
    dist = np.hypot(dx, dy)
    sp = np.hypot(vx_s, vy_s)
    scale = np.where(sp > CLIP_V, CLIP_V / np.maximum(sp, 1e-9), 1.0)
    vx_s, vy_s = vx_s * scale, vy_s * scale

    out = np.zeros((nb, N_TOKENS, N_COLS), dtype=np.float64)
    rows = np.arange(nb)[:, None]
    for t, (lo, sign) in enumerate(((0, 1.0), (PER_TEAM, -1.0))):
        cols = np.flatnonzero(team == t)
        key = np.where(present[:, cols], dist[:, cols], np.inf)
        order = np.argsort(key, axis=1, kind="stable")[:, :PER_TEAM]
        sel = cols[order]                                  # (nb, <=11) slot indices
        k = sel.shape[1]
        ok = present[rows, sel]
        blk = out[:, lo:lo + k]
        blk[..., 0] = x_s[rows, sel] / 10.0
        blk[..., 1] = y_s[rows, sel] / 10.0
        blk[..., 2] = vx_s[rows, sel] / 5.0
        blk[..., 3] = vy_s[rows, sel] / 5.0
        blk[..., 4] = 1.0
        blk[..., 5] = sign
        seen = bseen[:, None]
        blk[..., 6] = np.where(seen, dx[rows, sel], 0.0) / 10.0
        blk[..., 7] = np.where(seen, dy[rows, sel], 0.0) / 10.0
        blk[..., 8] = np.where(seen, np.minimum(dist[rows, sel], CLIP_D), 0.0) / 10.0
        blk[..., 9] = np.where(seen, 0.0, 1.0)
        blk *= ok[..., None]
        out[:, lo:lo + k] = blk
    bsp = np.hypot(bvx_s, bvy_s)
    bscale = np.where(bsp > CLIP_BV, CLIP_BV / np.maximum(bsp, 1e-9), 1.0)
    out[:, BALL, 0] = bx_s / 10.0
    out[:, BALL, 1] = by_s / 10.0
    out[:, BALL, 2] = bvx_s * bscale / 10.0
    out[:, BALL, 3] = bvy_s * bscale / 10.0
    out[:, BALL, 4] = bpres
    out[:, BALL, 8] = since_s
    out[:, BALL, 9] = np.where(bseen, 0.0, 1.0)
    out[:, BALL, 10] = 1.0
    return out.astype(np.float32)


def transform(tok, swap=False, mirror=False):
    """Tokens of the symmetric copy (works on (..., N_TOKENS, N_COLS) arrays)."""
    out = np.array(tok, dtype=np.float32, copy=True)
    if swap:
        home = out[..., :PER_TEAM, :].copy()
        out[..., :PER_TEAM, :] = out[..., PER_TEAM:2 * PER_TEAM, :]
        out[..., PER_TEAM:2 * PER_TEAM, :] = home
        out[..., list(X_COLS) + list(Y_COLS) + [TEAM_COL]] *= -1.0
    if mirror:
        out[..., list(Y_COLS)] *= -1.0
    return out + 0.0          # no negative zeros


# -- cache -------------------------------------------------------------------------------------

def tok_path(out, view, game):
    return os.path.join(out, view, game + ".npy")


def _save(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npy"
    np.save(tmp, arr.astype(np.float16))
    os.replace(tmp, path)


def build_game(game, labels, engine_cache, out, aug_k, views):
    from tools.learned_spike import augment, build, r4_inputs, r5_inputs
    build.check(game)
    t0 = time.time()
    rep = {}
    need_train = "train" in views or aug_k > 0
    raw = r4_inputs._raw_tac(game, engine_cache)
    codes = r4_inputs._codes(labels, game)
    n_bins = -(-len(codes) // BIN)
    if game.startswith("DFL-"):
        tac_train = r5_inputs.load_dfl(raw, game, "c_train_j")[0] if need_train else None
        tac_val = r4_inputs.load_dfl(raw, game, "c")[0] if "val" in views else None
    else:
        tac_train = tac_val = raw
    todo = []
    if "train" in views:
        todo.append(("train", tac_train))
    if "val" in views:
        todo.append(("val", tac_val))
    for view, tac in todo:
        p = tok_path(out, view, game)
        if not os.path.isfile(p):
            tk = tokens(tac, game)
            assert tk.shape[0] == n_bins, (game, tk.shape, n_bins)
            _save(p, tk)
    for k in range(aug_k):
        p = tok_path(out, "k{}".format(k), game)
        if os.path.isfile(p):
            continue
        aug, _ = augment.augment(tac_train, codes, game, k, data.graded_blocks(game),
                                 data.pitch_m(game), version="r3")
        tk = tokens(aug, game)
        assert tk.shape[0] == n_bins
        _save(p, tk)
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--labels", required=True, help="cache folder whose labels/ hold raw codes")
    b.add_argument("--engine-cache", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--games", nargs="+", required=True)
    b.add_argument("--aug-k", type=int, default=4)
    b.add_argument("--views", nargs="+", default=["train", "val"], choices=["train", "val"])
    args = p.parse_args(argv)
    for g in args.games:
        rep = build_game(g, args.labels, args.engine_cache, args.out, args.aug_k, args.views)
        print(g, json.dumps(rep), file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
