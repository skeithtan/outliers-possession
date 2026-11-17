"""C0 preparation: label-free run-time rules for the 19 class U fields, on one match and view.

    MTAH_DATA_DIR=... python tools/analysis/c0_u_fields.py --game BAR-ATH --view raw \
        --cache-dir DIR --json OUT.json [--ladder 7 8 9 10 11 12 13] [--skip-ladder]

The organisers' "Clarification of the rules" may rule that a fixed value set from a label-free
statistic of BAR-ATH's own tracking is a tuned parameter. This script asks, for each class U
field of docs/plan/calibration/provenance_current.json, what a fixed rule computed at run time
from the engine's own input would give. Every statistic below is label-free: the script reads
the tracking through engine.data, the Smart Tagging file (an input of the engine) and the
engine's own output (possession.run). It never opens evaluation/gt, a reference CSV or a video.
It refuses the sealed matches. It prints summary numbers only, never a tracking row.

Rules (each written here before the first run; docs/plan/calibration/C0_U_FIELDS_PLAN.md):

  kernel        CONTACT_HALF_WINDOW = round(HWHM) + 1 of the median |a| profile around the 46
                strongest isolated ball peaks (tools/analysis/atd_kernel.py, M13).
  kick_centre   DELTA_RESTART_KICK = median over the graded dv_9 rises of the offset from
                deadball._rise_start to the argmax of the impulse |v(t+h) - v(t-h)| over the
                rise, h = CONTACT_HALF_WINDOW (M16).
  ghost         GHOST_NOBODY_M = ghost_valley.measure(): valley of the nearest-player distance
                histogram over the stationary-ball frames, 0.5 m bins.
  jitter        speed quantiles of the cleaned ball on stationary frames (HEADING_MIN_SPEED,
                AIRBORNE_DECEL_MIN_SPEED): candidate rules p90, p99 and max, rounded to 0.5 m/s.
  sep           team-centroid separation and mean player speed inside long ball gaps (> 2 s,
                PLAN 1.1) against the other graded frames (SEP_DEAD): candidate rules are the
                floor of the long-gap median and the rounded midpoint of the two medians.
  gk_share      per block and team: the own-box share of the best slot and the largest share of
                any other slot with more than 60 s visible (GK_OWN_BOX_SHARE): the rule is the
                midpoint of the gap, rounded to 0.1; the inert band is the gap itself.
  frozen        share of visible player frames with a NaN ATD speed (before
                speed_from_positions), the 1 s displacement quantiles of those frames
                (FROZEN_MOVE_M, FROZEN_WIN) and, where both goalkeeper-like slots exist, the
                distance from the live goalkeeper-like slot beyond its half-way line to the
                frozen other goalkeeper-like slot (GK_SWAP_R): p50, p90, p95, p100 and the
                smallest multiple of 5 m that covers 90 %.
  touch_rate    local maxima of |dv_3| above DV_TOUCH with a player within R_TOUCH, per
                visible-ball minute, at 1.0, 1.5, 2.0 m/s (DV_TOUCH): the loosest value inside
                the 15 to 50 per minute band.
  chain_gap     median gap in frames between consecutive kick candidates of deadball.restarts()
                (unshifted) that both lie on engine-alive frames with no stoppage frame between
                (RESTART_CHAIN_FRAMES); the same for the engine's release rows.
  deep_stop     the slowest 20 s window of mean player speed on engine-alive frames
                (GK_HOLD_DEEP_STOP_SPEED): rule floor to 0.5 m/s of the minimum, minus 0.5 m/s
                margin is NOT applied (the record is the raw minimum); counts of deep-stop runs
                (< 1.0 m/s for >= 10 s) on alive and on stoppage frames.
  inward        reappearances of the ball beyond a touchline after a gap: frames to the first
                inward motion over 2 to 10 frames (OUT_INWARD_FRAMES): the count that moves
                inward at each window.
  set_pieces    count of SET PIECES tags (VETO_MIN_D_SEGMENTS): rule count - 4.
  ladder        H<->A switches per alive minute of the engine output at DV_CONTACT 7 to 13
                (DV_CONTACT): the loosest value with a rate of at most 3.0.

Deterministic, no seed, no clock. Python 3.9, numpy, pandas.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SEALED = ("DFL-J03WQQ", "DFL-J03WR9")
PFF_FORBIDDEN = ("PFF-3813", "PFF-3817", "PFF-3821", "PFF-3825", "PFF-3829", "PFF-3833",
                 "PFF-3837", "PFF-3842", "PFF-3847", "PFF-3851", "PFF-3855", "PFF-3859",
                 "PFF-10506", "PFF-10510", "PFF-10515", "PFF-3812", "PFF-3840", "PFF-10502",
                 "PFF-10517")
LONG_GAP_S = 2.0
BAND_RULE = 3.0
TOUCH_BAND = (15.0, 50.0)
LADDER = (7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0)
DEEP_WINDOW_S = 20.0


def q(values, probs):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {str(p): None for p in probs}
    return {str(p): round(float(np.percentile(values, p)), 3) for p in probs}


def runs_of(mask):
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def graded_mask(game, n):
    from engine import data

    out = np.zeros(n, dtype=bool)
    for _, first, last in data.graded_blocks(game):
        out[first:last + 1] = True
    return out


def labels_md5(labels):
    return hashlib.md5("".join(np.asarray(labels).astype("<U1").tolist()).encode()).hexdigest()


# -- the measurements ----------------------------------------------------------------------------

def kernel(game):
    import atd_kernel

    out = atd_kernel.measure(game)
    k = out["ball_kernel"]
    hwhm = k.get("hwhm")
    k["rule_half_window"] = None if hwhm is None or not np.isfinite(hwhm) else int(round(hwhm)) + 1
    return k


def kick_centre(game, inputs, cfg):
    from engine import deadball, events, tracks

    ball, fps = inputs["ball"], inputs["fps"]
    speed = ball["speed"]
    n = speed.shape[0]
    k = int(cfg.RESTART_K)
    graded = graded_mask(game, n)
    rise = deadball._above(tracks.velocity_step(speed, k, ball["run_id"]), cfg.DV_RESTART)
    size = events.impulse(ball, int(cfg.CONTACT_HALF_WINDOW))
    offsets = []
    for start, stop in runs_of(rise):
        if not graded[start]:
            continue
        end = min(n - 1, stop - 1 + k)
        rs = deadball._rise_start(speed, int(start), end)
        seg = size[rs:end + 1]
        if seg.size == 0 or not np.isfinite(seg).any():
            continue
        offsets.append(int(np.nanargmax(seg)))
    offsets = np.asarray(offsets)
    # Variant b: the window ends at rise_start + 2h (one bump), so a later second kick inside a
    # long dv_9 run does not pull the centre. Variant c: only the kick rows of
    # deadball.restarts() (the restart candidates the engine stamps), window as in b.
    h = int(cfg.CONTACT_HALF_WINDOW)
    kick_frames = set(int(f) for f in inputs["restarts"].loc[
        inputs["restarts"]["kind"] == "kick", "frame"].to_numpy())
    off_b, off_c = [], []
    for start, stop in runs_of(rise):
        if not graded[start]:
            continue
        end = min(n - 1, stop - 1 + k)
        rs = deadball._rise_start(speed, int(start), end)
        seg = size[rs:min(n, rs + 2 * h + 1)]
        if seg.size == 0 or not np.isfinite(seg).any():
            continue
        off_b.append(int(np.nanargmax(seg)))
        if rs in kick_frames:
            off_c.append(int(np.nanargmax(seg)))
    off_b, off_c = np.asarray(off_b), np.asarray(off_c)
    return {"n_rises": int(offsets.size), "quantiles": q(offsets, (10, 25, 50, 75, 90)),
            "rule_delta": None if offsets.size == 0 else int(round(float(np.median(offsets)))),
            "variant_b_n": int(off_b.size), "variant_b": q(off_b, (10, 25, 50, 75, 90)),
            "variant_c_n": int(off_c.size), "variant_c": q(off_c, (10, 25, 50, 75, 90))}


def ghost(game):
    import ghost_valley

    out = ghost_valley.measure(game)
    return {k: out[k] for k in ("stationary_frames", "frames_with_player", "mode_bin",
                                "valley_bin", "valley_m", "run_valley_m", "smoothed")}


def jitter(game, inputs, cfg):
    from engine import deadball

    ball = inputs["ball"]
    mask = deadball.stationary_mask(ball, cfg) & ball["visible"] & graded_mask(
        game, ball["visible"].shape[0])
    speed = np.asarray(ball["speed"], dtype=np.float64)[mask]
    speed = speed[np.isfinite(speed)]
    out = {"n_frames": int(speed.size), "quantiles": q(speed, (50, 90, 95, 99, 100))}
    for name, p in (("p90", 90), ("p99", 99), ("max", 100)):
        val = out["quantiles"][str(p)]
        out["rule_" + name + "_half"] = None if val is None else round(np.ceil(val * 2) / 2.0, 2)
    return out


def separation(game, inputs, cfg):
    ball, feats, fps = inputs["ball"], inputs["feats"], inputs["fps"]
    n = ball["visible"].shape[0]
    graded = graded_mask(game, n)
    missing = ~ball["visible"] & graded
    long_gap = np.zeros(n, dtype=bool)
    min_len = int(round(LONG_GAP_S * fps))
    n_gaps = 0
    for a, b in runs_of(missing):
        if b - a > min_len:
            long_gap[a:b] = True
            n_gaps += 1
    other = graded & ~long_gap
    sep, meanv = feats["sep"], feats["meanv_all"]
    out = {"n_long_gaps": n_gaps, "long_gap_share": round(float(long_gap.sum() / graded.sum()), 4),
           "sep_gap": q(sep[long_gap], (25, 50, 75)), "sep_other": q(sep[other], (25, 50, 75)),
           "meanv_gap": q(meanv[long_gap], (25, 50, 75)),
           "meanv_other": q(meanv[other], (25, 50, 75))}
    g, o = out["sep_gap"]["50"], out["sep_other"]["50"]
    out["rule_floor_gap_median"] = None if g is None else int(np.floor(g))
    out["rule_midpoint"] = None if g is None or o is None else int(round((g + o) / 2.0))
    return out


def gk_share(game, tactical, cfg):
    from engine import data, tracks

    pitch = data.pitch_m(game)
    blocks = data.graded_blocks(game)
    fps = float(data.fps(game))
    team_of_slot = np.asarray(tactical["team_of_slot"])
    px, py, visible = tracks._player_metres(tactical, pitch)
    length = float(pitch[0])
    rows = []
    for half, first, last in blocks:
        sl = slice(int(first), int(last) + 1)
        for team in (0, 1):
            cols = np.flatnonzero(team_of_slot == team)
            bvis = visible[sl][:, cols]
            team_x = px[sl][:, cols][bvis]
            if team_x.size == 0:
                continue
            own_goal_x = 0.0 if np.median(team_x) < length / 2.0 else length
            inside = bvis & tracks.penalty_area_mask(px[sl][:, cols], py[sl][:, cols], own_goal_x,
                                                    pitch, cfg)
            counts = inside.sum(axis=0)
            n_visible = bvis.sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                shares = np.where(n_visible > 0, counts / np.maximum(n_visible, 1), 0.0)
            best = int(np.argmax(counts))
            others = shares[(np.arange(cols.size) != best) & (n_visible > 60 * fps)]
            rows.append({"half": int(half), "team": "HA"[team],
                         "best_share": round(float(shares[best]), 3),
                         "best_own_box_frames": int(counts[best]),
                         "other_max_share": round(float(others.max()), 3) if others.size else None})
    best_min = min(r["best_share"] for r in rows) if rows else None
    other_max = max(r["other_max_share"] for r in rows
                    if r["other_max_share"] is not None) if rows else None
    out = {"rows": rows, "gap": [other_max, best_min]}
    out["rule_midpoint"] = (None if best_min is None or other_max is None
                            else round(round((best_min + other_max) / 2.0, 1), 2))
    return out


def frozen(game, tactical, cfg):
    from engine import data, tracks

    pitch = data.pitch_m(game)
    blocks = data.graded_blocks(game)
    px, py, visible = tracks._player_metres(tactical, pitch)
    ps = np.asarray(tactical["ps"], dtype=np.float64)
    n = px.shape[0]
    graded = graded_mask(game, n)
    vis = visible & graded[:, None]
    nan_speed = vis & np.isnan(ps)
    out = {"visible_player_frames": int(vis.sum()),
           "nan_speed_share": round(float(nan_speed.sum() / max(vis.sum(), 1)), 4)}
    half = int(cfg.FROZEN_WIN) // 2
    index = np.arange(n)
    before = np.clip(index - half, 0, n - 1)
    after = np.clip(index + half, 0, n - 1)
    move = np.hypot(px[after] - px[before], py[after] - py[before])
    moves = move[nan_speed]
    moves = moves[np.isfinite(moves)]
    out["nan_speed_move_1s"] = q(moves, (50, 90, 95, 99))
    out["nan_speed_share_below_0_25"] = (round(float((moves < cfg.FROZEN_MOVE_M).mean()), 4)
                                         if moves.size else None)
    # Histogram valley of the 1 s displacement of the NaN-speed frames (ghost_valley method,
    # 0.05 m bins to 3 m): a run-time rule for FROZEN_MOVE_M.
    import ghost_valley

    counts, _ = ghost_valley.histogram(moves, bin_m=0.05, max_m=3.0)
    mode, valley = ghost_valley.first_valley(ghost_valley.smooth3(counts))
    out["move_valley_m"] = ghost_valley.valley_value(valley, bin_m=0.05)
    out["move_hist_first_20"] = counts[:20].tolist()
    # the engine's own frozen mask and the live-to-frozen distance of the guard
    frozen_mask = tracks.frozen_mask(tactical, pitch, cfg)
    keepers = tracks.goalkeeper_slots(tactical, blocks, pitch, cfg)
    out["frozen_share_of_visible"] = round(float((frozen_mask & vis).sum() / max(vis.sum(), 1)), 4)
    gk_cols = set(v["slot"] for v in keepers.values() if v["slot"] >= 0)
    if gk_cols:
        on_gk = (frozen_mask & vis)[:, sorted(gk_cols)].sum()
        out["frozen_share_on_gk_slots"] = round(float(on_gk / max((frozen_mask & vis).sum(), 1)), 4)
    dists = []
    length = float(pitch[0])
    for half_id, first, last in blocks:
        frames = np.arange(int(first), int(last) + 1)
        for team in (0, 1):
            own, other = keepers.get((half_id, team)), keepers.get((half_id, 1 - team))
            if own is None or other is None or own["slot"] < 0 or other["slot"] < 0:
                continue
            g, o = own["slot"], other["slot"]
            dist = np.hypot(px[frames, g] - px[frames, o], py[frames, g] - py[frames, o])
            with np.errstate(invalid="ignore"):
                beyond = np.abs(px[frames, g] - own["own_goal_x"]) > length / 2.0
            fire = visible[frames, g] & ~frozen_mask[frames, g] & frozen_mask[frames, o] & beyond
            dists.append(dist[fire])
    dists = np.concatenate(dists) if dists else np.array([])
    dists = dists[np.isfinite(dists)]
    out["swap_candidate_frames"] = int(dists.size)
    out["swap_distance"] = q(dists, (50, 90, 95, 100))
    p90 = out["swap_distance"]["90"]
    out["rule_swap_r_cover90_5m"] = None if p90 is None else int(np.ceil(p90 / 5.0) * 5)
    out["keepers"] = {"{}-{}".format(h, "HA"[t]): v["slot"] for (h, t), v in keepers.items()}
    return out


def touch_rate(game, inputs, cfg):
    ball, feats, fps = inputs["ball"], inputs["feats"], inputs["fps"]
    n = ball["visible"].shape[0]
    graded = graded_mask(game, n)
    dv = np.abs(np.asarray(ball["dv_k"], dtype=np.float64))
    near = np.isfinite(feats["dmin"]) & (np.nan_to_num(feats["dmin"], nan=np.inf) <= cfg.R_TOUCH)
    with np.errstate(invalid="ignore"):
        peak = np.zeros(n, dtype=bool)
        peak[1:-1] = (dv[1:-1] > dv[:-2]) & (dv[1:-1] >= dv[2:])
    minutes = float((ball["visible"] & graded).sum()) / fps / 60.0
    out = {"visible_minutes": round(minutes, 2), "rates": {}}
    for t in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        with np.errstate(invalid="ignore"):
            count = int((peak & (dv > t) & near & graded).sum())
        out["rates"][str(t)] = round(count / minutes, 2) if minutes else None
    inside = [t for t in (1.0, 1.5, 2.0, 2.5, 3.0)
              if out["rates"][str(t)] is not None and TOUCH_BAND[0] <= out["rates"][str(t)] <= TOUCH_BAND[1]]
    out["rule_loosest_in_band"] = max(inside) if inside else None
    return out


def chain_gap(game, inputs, parts, cfg):
    stoppage = np.asarray(parts["stoppage"], dtype=bool)
    n = stoppage.shape[0]
    graded = graded_mask(game, n)
    dead_cum = np.concatenate(([0], np.cumsum(stoppage)))

    def gaps_of(frames):
        frames = np.asarray(sorted(set(int(f) for f in frames)))
        frames = frames[(frames >= 0) & (frames < n)]
        frames = frames[graded[frames] & ~stoppage[frames]]
        out = []
        for a, b in zip(frames[:-1], frames[1:]):
            if dead_cum[b] - dead_cum[a] == 0:
                out.append(int(b - a))
        return np.asarray(out)

    restarts = inputs["restarts"]
    kicks = restarts.loc[restarts["kind"] == "kick", "frame"].to_numpy()
    events = inputs["events"]
    releases = events.loc[events["kind"] == "release", "frame"].to_numpy()
    kg, rg = gaps_of(kicks), gaps_of(releases)
    return {"n_kick_candidates": int(kicks.size), "kick_gaps_n": int(kg.size),
            "kick_gap": q(kg, (25, 50, 75)), "n_releases": int(releases.size),
            "release_gap": q(rg, (25, 50, 75)),
            "rule_median_kick_gap": None if kg.size == 0 else int(round(float(np.median(kg))))}


def deep_stop(game, inputs, parts, cfg):
    feats, fps = inputs["feats"], inputs["fps"]
    stoppage = np.asarray(parts["stoppage"], dtype=bool)
    n = stoppage.shape[0]
    graded = graded_mask(game, n)
    meanv = np.asarray(feats["meanv_all"], dtype=np.float64)
    w = int(round(DEEP_WINDOW_S * fps))
    ok = graded & ~stoppage & np.isfinite(meanv)
    csum = np.concatenate(([0.0], np.cumsum(np.where(ok, meanv, 0.0))))
    cok = np.concatenate(([0], np.cumsum(ok)))
    means = (csum[w:] - csum[:-w]) / w
    full = (cok[w:] - cok[:-w]) == w
    alive_min = float(means[full].min()) if full.any() else None
    below = np.isfinite(meanv) & (meanv < cfg.GK_HOLD_DEEP_STOP_SPEED) & graded
    min_len = int(round(cfg.GK_HOLD_DEEP_STOP_S * fps))
    n_alive = n_dead = 0
    for a, b in runs_of(below):
        if b - a >= min_len:
            if stoppage[a:b].mean() > 0.5:
                n_dead += 1
            else:
                n_alive += 1
    return {"slowest_alive_20s_mean": None if alive_min is None else round(alive_min, 3),
            "deep_runs_on_stoppage": n_dead, "deep_runs_on_alive": n_alive,
            "rule_floor_half": None if alive_min is None else float(np.floor(alive_min * 2) / 2.0)}


def inward(game, inputs, cfg):
    from engine import data

    ball = inputs["ball"]
    length, width = data.pitch_m(game)
    x, y, visible, vy = ball["x"], ball["y"], ball["visible"], ball["vy"]
    n = visible.shape[0]
    graded = graded_mask(game, n)
    cases = []
    for start, _ in runs_of(visible & graded):
        f = int(start)
        if f == 0 or visible[f - 1]:
            continue
        if not (0.0 <= x[f] <= length) or (0.0 <= y[f] <= width):
            continue
        cases.append(f)
    counts = {}
    for win in range(1, 11):
        n_in = 0
        for f in cases:
            last = min(f + win, n - 1)
            while last > f and not visible[last]:
                last -= 1
            dy = y[last] - y[f] if last > f else vy[f]
            if np.isfinite(dy) and ((dy > 0.0) if y[f] < 0.0 else (dy < 0.0)):
                n_in += 1
        counts[str(win)] = n_in
    return {"n_touchline_reappearances": len(cases), "inward_by_window": counts}


def set_pieces(game):
    from engine import data

    try:
        tags = data.load_tagging(game)
    except Exception as exc:  # no tagging asset on an external match
        return {"n_set_pieces": None, "note": type(exc).__name__}
    if tags is None or len(tags) == 0:
        return {"n_set_pieces": 0}
    n = int((tags["code"].astype(str).str.upper() == "SET PIECES").sum())
    return {"n_set_pieces": n, "rule_minus_4": n - 4}


def switch_stats(labels, blocks, fps):
    labels = np.asarray(labels).astype("<U1")
    switches = alive = poss = 0
    for _, first, last in blocks:
        seg = labels[int(first):int(last) + 1]
        if seg.size == 0:
            continue
        alive += int(np.isin(seg, ("H", "A")).sum())
        starts = np.concatenate(([0], np.flatnonzero(seg[1:] != seg[:-1]) + 1))
        team = np.isin(seg[starts], ("H", "A"))
        poss += int(team.sum())
        switches += int((team[1:] & team[:-1]).sum())
    minutes = alive / float(fps) / 60.0
    return {"switches": switches, "alive_min": round(minutes, 2),
            "rate": round(switches / minutes, 3) if minutes > 0 else None, "possessions": poss}


def ladder(game, cfg, values, base_labels, base_value):
    from engine import data, possession

    blocks = data.graded_blocks(game)
    fps = float(data.fps(game))
    rows = {}
    for v in values:
        if float(v) == float(base_value):
            labels = base_labels
        else:
            labels = possession.run(dataclasses.replace(cfg, DV_CONTACT=float(v)), game=game)
        rows[str(float(v))] = dict(switch_stats(labels, blocks, fps), md5=labels_md5(labels)[:8])
    inside = [float(v) for v in values if rows[str(float(v))]["rate"] is not None
              and rows[str(float(v))]["rate"] <= BAND_RULE]
    return {"rows": rows, "rule_loosest_le_3": min(inside) if inside else None}


# -- driver ----------------------------------------------------------------------------------------

def measure(game, want_ladder, values):
    from engine import data, possession
    from engine.config import Config

    if game in SEALED or game in PFF_FORBIDDEN:
        raise SystemExit("{} is sealed or forbidden".format(game))
    cfg = Config.default()
    out = {"game": game, "data_dir": os.environ.get("MTAH_DATA_DIR"),
           "cache_dir": data.DEFAULT_CACHE_DIR, "config_hash": cfg.config_hash()}
    tactical_raw = data.load_tactical(game)
    out["kernel"] = kernel(game)
    out["ghost"] = ghost(game)
    out["gk_share"] = gk_share(game, tactical_raw, cfg)
    out["frozen"] = frozen(game, tactical_raw, cfg)
    out["set_pieces"] = set_pieces(game)
    inputs = possession.prepare(cfg, game)
    labels, parts = possession.run(cfg, return_parts=True, inputs=inputs)
    out["labels_md5"] = labels_md5(labels)
    out["kick_centre"] = kick_centre(game, inputs, cfg)
    out["jitter"] = jitter(game, inputs, cfg)
    out["separation"] = separation(game, inputs, cfg)
    out["touch_rate"] = touch_rate(game, inputs, cfg)
    out["chain_gap"] = chain_gap(game, inputs, parts, cfg)
    out["deep_stop"] = deep_stop(game, inputs, parts, cfg)
    out["inward"] = inward(game, inputs, cfg)
    out["base_switch"] = switch_stats(labels, data.graded_blocks(game), float(data.fps(game)))
    if want_ladder:
        out["ladder"] = ladder(game, cfg, values, labels, cfg.DV_CONTACT)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default="BAR-ATH")
    parser.add_argument("--view", default="raw")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--ladder", nargs="*", type=float, default=list(LADDER))
    parser.add_argument("--skip-ladder", action="store_true")
    args = parser.parse_args(argv)
    from engine import data

    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        data.DEFAULT_CACHE_DIR = args.cache_dir
    result = measure(args.game, not args.skip_ladder, args.ladder)
    result["view"] = args.view
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=1, sort_keys=True, default=str)
    brief = {k: result[k] for k in ("game", "view", "labels_md5")}
    brief["rules"] = {
        "CONTACT_HALF_WINDOW": result["kernel"].get("rule_half_window"),
        "DELTA_RESTART_KICK": result["kick_centre"]["rule_delta"],
        "GHOST_NOBODY_M": result["ghost"]["valley_m"],
        "SEP_DEAD floor/mid": [result["separation"]["rule_floor_gap_median"],
                               result["separation"]["rule_midpoint"]],
        "GK_OWN_BOX_SHARE": result["gk_share"]["rule_midpoint"],
        "GK_SWAP_R": result["frozen"]["rule_swap_r_cover90_5m"],
        "DV_TOUCH": result["touch_rate"]["rule_loosest_in_band"],
        "RESTART_CHAIN_FRAMES": result["chain_gap"]["rule_median_kick_gap"],
        "deep_stop_min": result["deep_stop"]["slowest_alive_20s_mean"],
        "DV_CONTACT": result.get("ladder", {}).get("rule_loosest_le_3"),
    }
    sys.stdout.write(json.dumps(brief, indent=1, sort_keys=True, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
