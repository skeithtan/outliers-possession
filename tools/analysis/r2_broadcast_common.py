"""R2-a shared loader: tactical ball/features, engine output, and the aligned broadcast ball.

load(out_dir) returns a dict. The engine parts are cached in <out_dir>/engine_parts.pkl (scratch,
never in the repository). The aligned broadcast ball comes from r2_broadcast_align.py
(<out_dir>/bb_on_tactical.npz). Deterministic. Never reads the full-match ground truth.
"""
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from engine import data, possession  # noqa: E402
from engine.config import Config  # noqa: E402

L, W = data.pitch_m()


def load(out_dir):
    cache = os.path.join(out_dir, "engine_parts.pkl")
    if os.path.isfile(cache):
        with open(cache, "rb") as fh:
            d = pickle.load(fh)
    else:
        cfg = Config.default()
        inputs = possession.prepare(cfg)
        labels, parts = possession.run(cfg, return_parts=True, inputs=inputs)
        ball = {k: inputs["ball"][k] for k in ("x", "y", "visible", "speed")}
        feats = {k: inputs["feats"][k] for k in ("dH", "dA", "nearest_slot_H", "nearest_slot_A",
                                                  "nvisH", "nvisA", "nearest_team")}
        d = {"labels": np.asarray(labels), "events": parts["events"], "owner": parts["owner"],
             "stoppage": parts["stoppage"], "raw_events": inputs["events"],
             "ball": ball, "feats": feats, "config_hash": cfg.config_hash()}
        with open(cache, "wb") as fh:
            pickle.dump(d, fh, protocol=4)
    bb = np.load(os.path.join(out_dir, "bb_on_tactical.npz"))
    d["bb"] = {k: bb[k] for k in bb.files}
    tac = data.load_tactical()
    d["px"] = tac["px"].astype(float) * L
    d["py"] = tac["py"].astype(float) * W
    d["team_of_slot"] = np.asarray(tac["team_of_slot"])
    d["T_raw_x"] = tac["bx"].astype(float) * L
    d["T_raw_y"] = tac["by"].astype(float) * W
    blocks = data.graded_blocks()
    graded = np.zeros(tac["n_frames"], bool)
    half = np.zeros(tac["n_frames"], np.int8)
    for h, lo, hi in blocks:
        graded[lo:hi + 1] = True
        half[lo:hi + 1] = h
    d["graded"], d["half"], d["blocks"] = graded, half, blocks
    return d


def nearest_per_team(px, py, team_of_slot, x, y):
    """For points (x, y) per frame: (dist (N,2), slot (N,2)) of the nearest player of H and A."""
    n = px.shape[0]
    dist = np.full((n, 2), np.nan)
    slot = np.full((n, 2), -1, np.int64)
    for t in (0, 1):
        cols = np.flatnonzero(team_of_slot == t)
        d = np.hypot(px[:, cols] - x[:, None], py[:, cols] - y[:, None])
        dd = np.where(np.isfinite(d), d, np.inf)
        j = np.argmin(dd, 1)
        m = dd[np.arange(n), j]
        ok = np.isfinite(m)
        dist[ok, t] = m[ok]
        slot[ok, t] = cols[j[ok]]
    return dist, slot
