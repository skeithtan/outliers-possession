"""Dead-ball forensics on the full match (F1). The script reads no ground truth.

    python tools/analysis/deadball_forensics.py --labels /tmp/c8.csv --out-dir /tmp/deadball_f1
        [--versions-dir <dir with p2_calibrated.csv ... c8_bump.csv>]

The script runs `engine.possession.prepare()` and `engine.possession.run()` once with the
default config. It checks that the labels it decodes equal the labels CSV byte for byte. Then it
writes these files to `--out-dir`. All of them hold match data, so keep them outside the
repository:

    segments.csv          one row for each `D` segment on graded frames
    stoppages.csv         one row for each stoppage of dead_intervals()
    alive_windows.csv     the physics of 20 s windows of alive play (the reference)
    cases.txt             the traces of the case segments (untagged and longer than 20 s, the
                          10 longest, and the set-piece tags with no `D` within 10 s)
    tag_versions.csv      the Smart Tagging agreement of each version (only with --versions-dir)
    tag_split.csv         the stretches that change the Smart Tagging agreement between two
                          consecutive versions (only with --versions-dir)
    tables.md             the aggregate tables and the per-segment table of the report
    summary.json          the headline numbers

The report `docs/plan/research/DEADBALL-FORENSICS.md` holds only the aggregate tables.

Terms:
  * A stoppage is one interval of `engine.possession.dead_intervals()`. It starts at an entry and
    ends at a confirmed restart candidate or at the end of the block.
  * The entry kind is `out` or `pursuit_stop` (an onset of `engine.deadball.dead_onsets()`),
    `evidence` (the start of a run of e_D above DEAD_EVIDENCE_HIGH that lasts at least
    COLLECTIVE_STOP_S: in practice a ball gap with slow players) or `pre_kickoff`.
  * The restart kind is `kick` or `reappear` (`engine.deadball.restarts()`). The players test
    holds when meanv_all rises above V_STOP within RESTART_CONFIRM_S of the candidate. A
    candidate that fails the players test and is still confirmed uses the C11 chain.
  * A suppressed contact is a bump contact (`engine.events._bump_contacts()`) on a frame where
    e_D is above EVENT_SUPPRESS_DEAD. The engine does not detect it, because the detector runs
    on alive frames only.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (ROOT, os.path.join(ROOT, "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import _common as cm  # noqa: E402
import sanity  # noqa: E402
from engine import deadball, events, possession  # noqa: E402
from engine.config import Config  # noqa: E402
from engine.tracks import _runs  # noqa: E402

TAG_MARGIN_S = 15.0          # brief: SET PIECES or GOALS tag within 15 s of the segment
LONG_UNTAGGED_S = 20.0       # brief: untagged segments longer than 20 s are cases
N_LONGEST = 10               # brief: the 10 longest segments are cases
TRACE_HALF_S = 5.0           # brief: trace +-5 s around the onset
TRACE_STEP = 10              # frames between trace rows (0.4 s at 25 fps)
PROFILE_STEP_S = 2.0         # seconds per row of the segment profile
ALIVE_WINDOW_S = 20.0        # reference windows of alive play, same length as the case limit
REENGAGE_S = 6.0             # RESTART_CONFIRM_S: a window this long tests "play goes on"
EVENT_WINDOW = 25            # frames: an event this near a stretch start explains it (1 s)
DEEP_STOP_MS = 1.0           # m/s: players this slow stand (alive 20 s windows: minimum 1.36)
DEEP_STOP_S = 10.0           # seconds of a deep stop in a row
STATIONARY_ALONE_S = 1.0     # seconds of a stationary ball with nobody within R_TOUCH
KEEPER_BALL_MS = 3.0         # m/s: a ball this slow at the goalkeeper can be in his hands
GOAL_AREA_DEPTH_M = 5.5      # Law 1: goal area depth
GOAL_AREA_HALF_WIDTH_M = 9.16  # Law 1: 5.5 m beyond each post, posts 7.32 m apart
VERSIONS = ("p2_calibrated", "c5_dribble", "c6_kickoff", "c7_touch", "c10_touchline",
            "c9_gk_guard", "c11_restart_chain", "c8_bump")
CYCLE_OF = {"c5_dribble": "C5", "c6_kickoff": "C6", "c7_touch": "C7", "c10_touchline": "C10",
            "c9_gk_guard": "C9", "c11_restart_chain": "C11", "c8_bump": "C8"}


# -- engine parts ---------------------------------------------------------------------------------

def engine_parts(cfg, labels_path):
    """(inputs, parts, codes): prepare() and run() of the engine, checked against the CSV."""
    inputs = possession.prepare(cfg)
    labels, parts = possession.run(cfg, return_parts=True, inputs=inputs)
    codes = cm.read_results(labels_path)
    decoded = cm.pm.label_codes(labels)
    if not np.array_equal(decoded, codes):
        raise SystemExit("{} differs from the engine output on {} frames".format(
            labels_path, int((decoded != codes).sum())))
    return inputs, parts, codes


def stoppages(inputs, cfg):
    """[dict] of the stoppages of dead_intervals(), with the entry and restart kinds.

    The loop is the loop of possession.dead_intervals(). It also records why each stoppage
    started and which candidate ended it. The function checks that its mask equals the mask of
    dead_intervals().
    """
    n, blocks, kickoffs, fps = (inputs["n_frames"], inputs["blocks"], inputs["kickoffs"],
                                inputs["fps"])
    onsets, cands, e_dead = inputs["onsets"], inputs["restarts"], inputs["e_dead"]
    feats = inputs["feats"]
    holds = inputs.get("holds")
    high = np.asarray(e_dead) > cfg.DEAD_EVIDENCE_HIGH
    dwell = max(1, int(round(cfg.COLLECTIVE_STOP_S * fps)))
    arm = max(1, int(round(cfg.RESTART_ARM_S * fps)))
    confirm = max(1, int(round(cfg.RESTART_CONFIRM_S * fps)))
    starts, stops = _runs(high)
    dwell_entries = set(int(s) for s in starts[(stops - starts) >= dwell])
    onset_rows = onsets[onsets["mechanism"] != "collective_stop"]
    onset_kind = {}
    for frame, mech in zip(onset_rows["frame"], onset_rows["mechanism"]):
        onset_kind.setdefault(int(frame), []).append(str(mech))
    entries = np.array(sorted(dwell_entries | set(onset_kind)), dtype=np.int64)
    moving = np.isfinite(feats["meanv_all"]) & (np.nan_to_num(feats["meanv_all"]) > cfg.V_STOP)
    usable = cands[cands["confirmed"]].reset_index(drop=True)
    cand_frames = usable["frame"].to_numpy()

    out = []
    dead = np.zeros(n, dtype=bool)
    for half, first, last in blocks:
        stop = last + 1
        pos = first
        kick = kickoffs.get(half)
        if kick is not None and first < kick <= last:
            out.append({"enter": first, "entry": "pre_kickoff", "leave": int(kick),
                        "restart": "kickoff", "restart_team": None, "players_test": None,
                        "chain": False, "half": half})
            dead[first:kick] = True
            pos = kick
        while pos < stop:
            later = entries[(entries >= pos) & (entries < stop)]
            if later.size == 0:
                break
            enter = int(later[0])
            kinds = list(onset_kind.get(enter, []))
            if enter in dwell_entries:
                kinds.append("evidence")
            ok = np.flatnonzero((cand_frames >= enter + arm) & (cand_frames < stop))
            end = int(cand_frames[ok[0]]) if ok.size else stop
            if holds is not None and holds[0][enter] and enter not in onset_kind \
                    and not holds[1][enter:end].any():
                pos = enter + 1          # C13: a goalkeeper hold is play
                continue
            row = {"enter": enter, "entry": "+".join(kinds), "half": half}
            if ok.size == 0:
                row.update({"leave": stop, "restart": "block_end", "restart_team": None,
                            "players_test": None, "chain": False})
                dead[enter:stop] = True
                out.append(row)
                break
            cand = usable.iloc[int(ok[0])]
            leave = int(cand["frame"])
            players = bool(moving[leave:min(n, leave + confirm + 1)].any())
            row.update({"leave": leave, "restart": str(cand["kind"]),
                        "restart_team": cand["team"], "players_test": players,
                        "chain": not players})
            dead[enter:leave] = True
            out.append(row)
            pos = leave
    check, _ = possession.dead_intervals(n, onsets, e_dead, cands, blocks, kickoffs, cfg, fps,
                                         holds)
    if not np.array_equal(check, dead):
        raise SystemExit("the stoppage loop differs from dead_intervals()")
    shift = {"kick": int(cfg.DELTA_RESTART_KICK), "reappear": int(cfg.DELTA_RESTART_REAPPEAR)}
    for row in out:
        block_last = next(last for h, first, last in blocks if h == row["half"])
        row["leave_shifted"] = int(min(row["leave"] + shift.get(row["restart"], 0),
                                       block_last + 1))
    return out


def suppressed_contacts(inputs, cfg):
    """int64 frames of the bump contacts on frames where e_D is above EVENT_SUPPRESS_DEAD."""
    ball, feats = inputs["ball"], inputs["feats"]
    every = np.ones(ball["visible"].shape[0], dtype=bool)
    rows = events._bump_contacts(ball, feats, None, every, cfg)
    frames = np.array([int(r[0]) for r in rows], dtype=np.int64)
    e_dead = np.asarray(inputs["e_dead"], dtype=np.float64)
    return frames[e_dead[frames] > cfg.EVENT_SUPPRESS_DEAD]


# -- per-segment features -------------------------------------------------------------------------

def window_features(inputs, lo, hi, cfg, suppressed=None):
    """dict of the physics features over frames [lo, hi)."""
    ball, feats = inputs["ball"], inputs["feats"]
    vis = ball["visible"][lo:hi]
    meanv = feats["meanv_all"][lo:hi]
    sep = feats["sep"][lo:hi]
    dmin = feats["dmin"][lo:hi]
    speed = ball["speed"][lo:hi]
    pitch_len, pitch_wid = cm.games.GAMES[inputs["game"]]["pitch"]
    x, y = ball["x"][lo:hi], ball["y"][lo:hi]
    with np.errstate(invalid="ignore"):
        near = vis & np.isfinite(dmin) & (dmin <= cfg.R_TOUCH)
        outside = vis & ((x < 0) | (x > pitch_len) | (y < 0) | (y > pitch_wid))
        slow = np.isfinite(meanv) & (meanv < cfg.V_STOP)
        moving_ball = vis & np.isfinite(speed) & (speed > 3.0)
    n_vis = int(vis.sum())
    out = {
        "vis_share": float(vis.mean()) if hi > lo else np.nan,
        "meanv": float(np.nanmean(meanv)) if np.isfinite(meanv).any() else np.nan,
        "slow_share": float(slow.mean()) if hi > lo else np.nan,
        "sep": float(np.nanmean(sep)) if np.isfinite(sep).any() else np.nan,
        "near_share_vis": float(near.sum()) / n_vis if n_vis else np.nan,
        "outside_share_vis": float(outside.sum()) / n_vis if n_vis else np.nan,
        "fast_ball_share_vis": float(moving_ball.sum()) / n_vis if n_vis else np.nan,
    }
    if suppressed is not None:
        out["suppressed"] = int(((suppressed >= lo) & (suppressed < hi)).sum())
    return out


def tags_near(tagging, lo, hi, margin):
    """[(code, type, team, start, end, id)] of SET PIECES and GOALS rows near [lo, hi)."""
    rows = tagging[tagging["code"].isin(("SET PIECES", "GOALS"))]
    out = []
    for row in rows.itertuples(index=False):
        if int(row.start_frame) < hi + margin and int(row.end_frame) > lo - margin:
            out.append((row.code, row.type, row.team, int(row.start_frame), int(row.end_frame),
                        int(row.id)))
    return out


def d_segments(codes, blocks):
    """[(start, stop, half)] of the `D` segments on graded frames (half-open)."""
    out = []
    for half, first, last in blocks:
        for a, b, k in cm.pm.segments(codes, [(first, last + 1)]):
            if k == cm.D:
                out.append((a, b, half))
    return out


def segment_table(inputs, parts, codes, cfg, keepers):
    """[dict]: one row for each `D` segment, with its stoppages, tags and physics."""
    fps = inputs["fps"]
    stops = stoppages(inputs, cfg)
    supp = suppressed_contacts(inputs, cfg)
    masks = term_masks(inputs, cfg)
    table = inputs["events"]
    ev_frames = table.loc[table["kind"].isin(("release", "control", "touch", "dribble")),
                          "frame"].to_numpy()
    bump = table["notes"].astype(str).isin(("", "receive", "redirect")) & table["kind"].isin(
        ("release", "control", "touch", "contested"))
    bump_frames = table.loc[bump, "frame"].to_numpy()
    margin = int(round(TAG_MARGIN_S * fps))
    rows = []
    for a, b, half in d_segments(codes, inputs["blocks"]):
        inside = [s for s in stops if s["enter"] < b and s["leave_shifted"] > a]
        first = inside[0] if inside else None
        last = inside[-1] if inside else None
        tags = tags_near(inputs["tagging"], a, b, margin)
        row = {"start": a, "end": b, "length_s": (b - a) / fps, "half": half,
               "n_stoppages": len(inside),
               "entry": first["entry"] if first else "decoder", "entry_frame":
               first["enter"] if first else -1,
               "restart": last["restart"] if last else "decoder",
               "restart_frame": last["leave"] if last else -1,
               "restart_shifted": last["leave_shifted"] if last else -1,
               "restart_team": (last["restart_team"] or "") if last else "",
               "chain": bool(last["chain"]) if last else False,
               "tags": "; ".join("{} {} {}".format(c, t, tm or "-") for c, t, tm, _, _, _ in tags),
               "tag_ends": ";".join(str(e) for _, _, _, _, e, _ in tags),
               "tagged": bool(tags),
               "goal": any(c == "GOALS" for c, _, _, _, _, _ in tags),
               "events_inside": int(((ev_frames >= a) & (ev_frames < b)).sum()),
               "contacts_inside": int(((bump_frames >= a) & (bump_frames < b)).sum())}
        row.update(window_features(inputs, a, b, cfg, supp))
        entry = row["entry_frame"] if row["entry_frame"] >= 0 else a
        row["entry_terms"] = evidence_terms(inputs, entry, cfg, masks)
        for prefix, frame, before in (("in_", entry, True),
                                      ("re_", row["restart_frame"] if row["restart_frame"] >= 0
                                       else b, False)):
            ctx = ball_context(inputs, min(frame, inputs["n_frames"] - 1), keepers, half, cfg,
                               before)
            for key, value in ctx.items():
                row[prefix + key] = value
        rows.append(row)
    return rows, stops, supp


def alive_reference(inputs, codes, cfg, supp):
    """[dict] of window_features() over 20 s windows inside one run of alive (H or A) frames.

    The windows do not overlap. They start at the start of each alive run.
    """
    fps = inputs["fps"]
    width = int(round(ALIVE_WINDOW_S * fps))
    out = []
    for _, first, last in inputs["blocks"]:
        alive = (codes[first:last + 1] == cm.H) | (codes[first:last + 1] == cm.A)
        starts, stops = _runs(alive)
        for a, b in zip(starts + first, stops + first):
            for lo in range(int(a), int(b) - width + 1, width):
                row = window_features(inputs, lo, lo + width, cfg, supp)
                row["start"] = lo
                out.append(row)
    return out


def keeper_slots(cfg, inputs):
    """{(half, team letter): slot} of the goalkeeper-like slots (tracks.goalkeeper_slots)."""
    from engine import tracks
    tactical = possession.engine_tactical(cfg)
    found = tracks.goalkeeper_slots(tactical, inputs["blocks"],
                                    cm.games.GAMES[inputs["game"]]["pitch"], cfg)
    return {(half, "H" if team == 0 else "A"): int(info["slot"])
            for (half, team), info in found.items()}


def ball_context(inputs, frame, keepers, half, cfg, before=True):
    """dict: the ball and the nearest player at the last visible frame at or before `frame`
    (before=True), or at the first visible frame at or after it (before=False)."""
    ball, feats = inputs["ball"], inputs["feats"]
    vis = ball["visible"]
    f = int(frame)
    n = vis.shape[0]
    if before:
        while f > 0 and not vis[f]:
            f -= 1
    else:
        while f < n - 1 and not vis[f]:
            f += 1
    pitch_len, pitch_wid = cm.games.GAMES[inputs["game"]]["pitch"]
    x, y = float(ball["x"][f]), float(ball["y"][f])
    team = deadball._team_letter(feats["nearest_team"][f])
    slot = int(feats["nearest_slot_" + team][f]) if team else -1
    in_box = (abs(y - pitch_wid / 2.0) < cfg.PENALTY_AREA_WIDTH_M / 2.0
              and (x < cfg.PENALTY_AREA_DEPTH_M or x > pitch_len - cfg.PENALTY_AREA_DEPTH_M))
    inside = 0.0 <= x <= pitch_len and 0.0 <= y <= pitch_wid
    return {"frame": f, "gap_s": abs(int(frame) - f) / inputs["fps"], "x": x, "y": y,
            "v": float(ball["speed"][f]), "team": team or "", "dist": float(feats["dmin"][f]),
            "keeper": bool(team and keepers.get((half, team), -2) == slot),
            "in_box": bool(in_box), "inside": bool(inside)}


def term_masks(inputs, cfg):
    """{name: bool (N,)} of the e_D terms that need a whole-array computation."""
    ball, feats = inputs["ball"], inputs["feats"]
    pitch = cm.games.GAMES[inputs["game"]]["pitch"]
    return {"outside": deadball.outside_pitch(ball, pitch, cfg.PITCH_MARGIN_M),
            "setpiece": deadball.setpiece_mask(ball, feats, cfg)}


def evidence_terms(inputs, frame, cfg, masks):
    """str: the active e_D terms at `frame` (deadball.dead_evidence)."""
    ball, feats = inputs["ball"], inputs["feats"]
    f = int(frame)
    slow = deadball._below(feats["meanv_all"][f:f + 1], cfg.V_STOP)[0]
    terms = []
    if not ball["visible"][f] and slow:
        terms.append("gap+slow")
    elif not ball["visible"][f]:
        terms.append("gap")
    if slow:
        terms.append("slow")
    if deadball._above(feats["sep"][f:f + 1], cfg.SEP_DEAD)[0]:
        terms.append("sep")
    if masks["outside"][f]:
        terms.append("outside")
    if masks["setpiece"][f]:
        terms.append("setpiece")
    return "+".join(terms)


# -- traces ---------------------------------------------------------------------------------------

def trace_lines(inputs, parts, codes, centre, cfg, half_s=TRACE_HALF_S, step=TRACE_STEP):
    """Text rows of the ball and player trace on [centre - half_s, centre + half_s]."""
    ball, feats, fps = inputs["ball"], inputs["feats"], inputs["fps"]
    e_dead = inputs["e_dead"]
    half = int(round(half_s * fps))
    lo, hi = max(0, centre - half), min(inputs["n_frames"], centre + half + 1)
    table = inputs["events"]
    lines = ["  {:>7} {:>4} {:>3} {:>6} {:>6} {:>5} {:>5} {:>5} {:>5} {:>5} {:>4} {:>3}  {}".format(
        "frame", "rel", "vis", "x", "y", "v", "dH", "dA", "meanv", "sep", "e_D", "lab", "events")]
    for f in range(lo, hi, step):
        ev = table[(table["frame"] >= f) & (table["frame"] < f + step)]
        evtext = ",".join("{}{}@{}".format(k, t or "?", int(fr))
                          for fr, k, t in zip(ev["frame"], ev["kind"], ev["team"]))
        lines.append("  {:>7} {:>+4.1f} {:>3} {:>6.1f} {:>6.1f} {:>5.1f} {:>5.1f} {:>5.1f} "
                     "{:>5.2f} {:>5.1f} {:>4.2f} {:>3}  {}".format(
                         f, (f - centre) / fps, "y" if ball["visible"][f] else ".",
                         ball["x"][f], ball["y"][f], ball["speed"][f], feats["dH"][f],
                         feats["dA"][f], feats["meanv_all"][f], feats["sep"][f], e_dead[f],
                         cm.LETTER[int(codes[f])], evtext))
    return lines


def profile_lines(inputs, a, b, cfg, supp):
    """Text rows of window_features() over PROFILE_STEP_S bins of [a - 4 s, b + 4 s)."""
    fps = inputs["fps"]
    step = int(round(PROFILE_STEP_S * fps))
    pad = 2 * step
    lines = ["  {:>7} {:>6} {:>4} {:>5} {:>5} {:>5} {:>5} {:>5} {:>4}".format(
        "frame", "rel_s", "vis", "meanv", "slow", "sep", "near", "fast", "supp")]
    for lo in range(max(0, a - pad), min(inputs["n_frames"], b + pad), step):
        f = window_features(inputs, lo, lo + step, cfg, supp)
        lines.append("  {:>7} {:>+6.0f} {:>4.2f} {:>5.2f} {:>5.2f} {:>5.1f} {:>5.2f} {:>5.2f} "
                     "{:>4d}".format(lo, (lo - a) / fps, f["vis_share"], f["meanv"],
                                     f["slow_share"], f["sep"], f["near_share_vis"],
                                     f["fast_ball_share_vis"], f["suppressed"]))
    return lines


def moving_cover(inputs, a, b, cfg):
    """(seconds, [(start, stop)]): the frames of [a, b) covered by REENGAGE_S windows that lie
    inside [a, b) and have meanv_all above V_STOP on at least 80 % of their frames."""
    fps = inputs["fps"]
    width = int(round(REENGAGE_S * fps))
    if b - a < width:
        return 0.0, []
    meanv = inputs["feats"]["meanv_all"][a:b]
    with np.errstate(invalid="ignore"):
        moving = (np.isfinite(meanv) & (meanv > cfg.V_STOP)).astype(np.float64)
    csum = np.concatenate(([0.0], np.cumsum(moving)))
    share = (csum[width:] - csum[:-width]) / width
    cover = np.zeros(b - a, dtype=np.int64)
    for s in np.flatnonzero(share >= 0.8):
        cover[s] += 1
        if s + width < cover.shape[0]:
            cover[s + width] -= 1
    covered = np.cumsum(cover) > 0
    starts, stops = _runs(covered)
    return float(covered.sum()) / fps, [(int(s) + a, int(e) + a) for s, e in zip(starts, stops)]


def reengage_frame(inputs, a, b, cfg, skip_s=2.0):
    """First frame in [a + skip_s, b) that starts REENGAGE_S of meanv_all above V_STOP on at
    least 80 % of frames, or -1. It marks players moving like open play inside the segment."""
    fps = inputs["fps"]
    meanv = inputs["feats"]["meanv_all"]
    width = int(round(REENGAGE_S * fps))
    with np.errstate(invalid="ignore"):
        moving = (np.isfinite(meanv) & (meanv > cfg.V_STOP)).astype(np.float64)
    lo = a + int(round(skip_s * fps))
    if b - lo < width:
        return -1
    csum = np.concatenate(([0.0], np.cumsum(moving[lo:b])))
    share = (csum[width:] - csum[:-width]) / width
    hits = np.flatnonzero(share >= 0.8)
    return int(lo + hits[0]) if hits.size else -1


# -- tag agreement split --------------------------------------------------------------------------

def tag_paint(inputs):
    """int8 (N,) Smart Tagging team paint of tools/sanity.py on the graded frames."""
    runs = cm.runs_from_blocks(inputs["blocks"])
    return cm.tag_team_codes(inputs["tagging"], inputs["n_frames"], sanity.TAG_PHASE_CODES, runs)


def agreement(codes, paint):
    """(agree frames, tagged-and-alive frames) as in sanity._tag_agreement."""
    sure = (paint == 0) | (paint == 1)
    alive = (codes == cm.H) | (codes == cm.A)
    return int((sure & alive & (codes == paint)).sum()), int((sure & alive).sum())


def change_stretches(old, new, paint, blocks):
    """[dict] of the maximal runs of graded frames where old != new with one (old, new) pair.

    Each row has the agree and alive deltas on the tagged frames of the stretch.
    """
    sure = (paint == 0) | (paint == 1)
    out = []
    for _, first, last in blocks:
        lo, hi = first, last + 1
        diff = old[lo:hi] != new[lo:hi]
        key = np.where(diff, old[lo:hi].astype(np.int64) * 4 + new[lo:hi], -1)
        breaks = np.flatnonzero(key[1:] != key[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [hi - lo]))
        for i in range(len(bounds) - 1):
            s, e = lo + int(bounds[i]), lo + int(bounds[i + 1])
            if key[s - lo] < 0:
                continue
            seg = slice(s, e)
            t = sure[seg]
            o, nw, p = old[seg], new[seg], paint[seg]
            o_alive, n_alive = (o == cm.H) | (o == cm.A), (nw == cm.H) | (nw == cm.A)
            d_agree = int((t & n_alive & (nw == p)).sum()) - int((t & o_alive & (o == p)).sum())
            d_alive = int((t & n_alive).sum()) - int((t & o_alive).sum())
            tagged = int(t.sum())
            team = ""
            if tagged:
                h, a = int((p[t] == 0).sum()), int((p[t] == 1).sum())
                team = "H" if h >= a else "A"
                if h and a:
                    team = "H{}/A{}".format(h, a)
            out.append({"start": s, "end": e, "frames": e - s, "old": cm.LETTER[int(o[0])],
                        "new": cm.LETTER[int(nw[0])], "tagged": tagged, "tag_team": team,
                        "d_agree": d_agree, "d_alive": d_alive})
    return out


def events_near(table, frame, team, window=EVENT_WINDOW):
    """str: the owner-kind rows of `team` within +-window frames of `frame` (kind/note@frame)."""
    rows = table[(table["frame"] >= frame - window) & (table["frame"] <= frame + window)
                 & (table["team"] == team)
                 & table["kind"].isin(("release", "control", "dribble", "touch"))]
    out = []
    for fr, kind, note in zip(rows["frame"], rows["kind"], rows["notes"]):
        note = str(note) if isinstance(note, str) and not str(note).startswith("end") else ""
        out.append("{}/{}@{}".format(kind, note.split(";")[0], int(fr)))
    return ";".join(out)


def c8_class(bump_text, dv3_text):
    """str: why a C8 stretch changed, from the new team's rows near its start.

    receive     a bump `control` row (note receive or redirect) of the new team: the
                intended C8 mechanism (receptions as owner events)
    deflection  a bump `release` of the new team where the dv3 rows had a release with the
                note deflection: the dv3 deflection filter does not act on bump rows
    other       anything else (a stretch that the old team's event at its end decides, a
                decoder edge, or a release stamp that moved)
    """
    if "control/receive" in bump_text or "control/redirect" in bump_text:
        return "receive"
    if "release/@" in bump_text and "release/deflection" in dv3_text:
        return "deflection"
    return "other"


def tag_split(inputs, versions_dir, cfg):
    """(per-version agreement rows, per-cycle stretch rows)."""
    import dataclasses
    bump = inputs["events"]
    dv3 = possession.prepare(dataclasses.replace(cfg, CONTACT_DETECTOR="dv3"))["events"]
    paint = tag_paint(inputs)
    codes = {}
    for name in VERSIONS:
        raw = cm.read_results(os.path.join(versions_dir, name + ".csv"))
        codes[name] = cm.pm.fill_abstentions(raw, cm.runs_from_blocks(inputs["blocks"]))
    version_rows = []
    for name in VERSIONS:
        agree, alive = agreement(codes[name], paint)
        version_rows.append({"version": name, "agree": agree, "alive": alive,
                             "agreement": agree / float(alive)})
    stretch_rows = []
    for prev, cur in zip(VERSIONS[:-1], VERSIONS[1:]):
        a0, n0 = agreement(codes[prev], paint)
        a1, n1 = agreement(codes[cur], paint)
        r0 = a0 / float(n0)
        for row in change_stretches(codes[prev], codes[cur], paint, inputs["blocks"]):
            # exact split: a1/n1 - a0/n0 = sum over stretches of (d_agree - r0 d_alive) / n1
            row["contrib_pp"] = 100.0 * (row["d_agree"] - r0 * row["d_alive"]) / n1
            row["cycle"] = CYCLE_OF[cur]
            row["from"], row["to"] = prev, cur
            new_team = row["new"] if row["new"] in ("H", "A") else None
            row["bump_new_start"] = events_near(bump, row["start"], new_team) if new_team else ""
            row["dv3_new_start"] = events_near(dv3, row["start"], new_team) if new_team else ""
            old_team = row["old"] if row["old"] in ("H", "A") else None
            row["bump_old_end"] = events_near(bump, row["end"], old_team) if old_team else ""
            row["c8_class"] = (c8_class(row["bump_new_start"], row["dv3_new_start"])
                               if row["cycle"] == "C8" else "")
            stretch_rows.append(row)
    return version_rows, stretch_rows


# -- cases ----------------------------------------------------------------------------------------

def case_rows(rows, fps):
    """(untagged long rows, longest rows): the two case groups of the brief."""
    untagged = [r for r in rows if not r["tagged"] and r["length_s"] > LONG_UNTAGGED_S]
    longest = sorted(rows, key=lambda r: (-r["length_s"], r["start"]))[:N_LONGEST]
    return untagged, longest


def setpieces_without_d(inputs, codes):
    """sanity._set_piece_check() on the labels (the tags with no `D` within 10 s)."""
    runs = cm.runs_from_blocks(inputs["blocks"])
    filled = cm.pm.fill_abstentions(codes, runs)
    return sanity._set_piece_check(filled, inputs["tagging"], runs, inputs["fps"])


def write_cases(path, inputs, parts, codes, rows, cfg, supp):
    """Write the traces of every case segment and of the set-piece tags with no `D`."""
    fps = inputs["fps"]
    untagged, longest = case_rows(rows, fps)
    lines = []
    for title, group in (("UNTAGGED LONGER THAN 20 S", untagged), ("10 LONGEST", longest)):
        lines += ["", "=" * 100, title, "=" * 100]
        for r in group:
            lines += ["", "-" * 100,
                      "D {}-{} ({:.1f} s, half {}) entry {} @{} [{}] -> restart {} @{} "
                      "(shifted {}, team {}, chain {}) stoppages {} reengage {}".format(
                          r["start"], r["end"], r["length_s"], r["half"], r["entry"],
                          r["entry_frame"], r["entry_terms"], r["restart"], r["restart_frame"],
                          r["restart_shifted"], r["restart_team"], r["chain"],
                          r["n_stoppages"], r["reengage"]),
                      "  tags: {} (ends {})".format(r["tags"] or "none", r["tag_ends"] or "-"),
                      "  inside: vis {:.2f} meanv {:.2f} slow {:.2f} sep {:.1f} near {:.2f} "
                      "fast {:.2f} contacts {} suppressed {}".format(
                          r["vis_share"], r["meanv"], r["slow_share"], r["sep"],
                          r["near_share_vis"], r["fast_ball_share_vis"],
                          r["contacts_inside"], r["suppressed"]),
                      "  ball before entry: f {} ({:.1f} s before) x {:.1f} y {:.1f} v {:.1f} "
                      "nearest {} {:.1f} m keeper {} box {}".format(
                          r["in_frame"], r["in_gap_s"], r["in_x"], r["in_y"], r["in_v"],
                          r["in_team"], r["in_dist"], r["in_keeper"], r["in_in_box"]),
                      "  ball at restart: f {} (+{:.1f} s) x {:.1f} y {:.1f} v {:.1f} "
                      "nearest {} {:.1f} m keeper {} box {}".format(
                          r["re_frame"], r["re_gap_s"], r["re_x"], r["re_y"], r["re_v"],
                          r["re_team"], r["re_dist"], r["re_keeper"], r["re_in_box"]),
                      "  trace around the entry:"]
            lines += trace_lines(inputs, parts, codes, int(r["entry_frame"]), cfg)
            lines += ["  trace around the restart:"]
            lines += trace_lines(inputs, parts, codes, int(min(r["restart_frame"],
                                                               inputs["n_frames"] - 1)), cfg)
            lines += ["  profile ({:.0f} s bins):".format(PROFILE_STEP_S)]
            lines += profile_lines(inputs, r["start"], r["end"], cfg, supp)
    check = setpieces_without_d(inputs, codes)
    lines += ["", "=" * 100, "SET PIECES TAGS WITH NO D WITHIN 10 S: {} of {}".format(
        check["n_without_d"], check["n_checked"]), "=" * 100]
    for item in check["without_d"]:
        lines += ["", "id {} half {} {} team {} frames {}-{}".format(
            item["id"], item["half"], item["type"], item["team"], item["start_frame"],
            item["end_frame"])]
        lines += trace_lines(inputs, parts, codes, int(item["start_frame"]), cfg,
                             half_s=12.0, step=12)
        lines += ["  profile:"]
        lines += profile_lines(inputs, int(item["start_frame"]), int(item["end_frame"]), cfg,
                               supp)
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


# -- physics flags --------------------------------------------------------------------------------

def _longest_run_s(mask, fps):
    """Seconds of the longest run of True in `mask`."""
    starts, stops = _runs(np.asarray(mask, dtype=bool))
    return float((stops - starts).max()) / fps if starts.size else 0.0


def physics_flags(inputs, row, cfg, stationary):
    """dict of the stoppage and alive signatures of one segment (report, case rule).

    Stoppage signatures (any one supports `D`):
      deep_stop    meanv_all below DEEP_STOP_MS for at least DEEP_STOP_S in a row
      sep_stop     sep above SEP_DEAD and meanv_all below V_STOP together on half the frames
      ball_out     the visible ball beyond a line inside the segment, or the onset is an
                   observed `out`
      placed       a stationary ball (deadball.stationary_mask) with nobody within R_TOUCH
      goal_area    the restart ball inside a goal area (a goal kick position)
      no_chase     the onset is a pursuit stop (nobody chases a moving ball)
      block_end    the segment runs to the end of its block with no player tracked at the end
    Alive signature:
      keeper_hold  the ball vanished within R_TOUCH of the own goalkeeper-like slot inside its
                   penalty area at under KEEPER_BALL_MS, the entry is `evidence`, mean sep is
                   at most SEP_DEAD and there is no deep stop (Law 12: the ball in the
                   goalkeeper's hands is in play)
    """
    fps = inputs["fps"]
    a, b = int(row["start"]), int(row["end"])
    feats, ball = inputs["feats"], inputs["ball"]
    pitch_len, pitch_wid = cm.games.GAMES[inputs["game"]]["pitch"]
    meanv, sep = feats["meanv_all"][a:b], feats["sep"][a:b]
    with np.errstate(invalid="ignore"):
        deep = np.isfinite(meanv) & (meanv < DEEP_STOP_MS)
        sep_slow = (np.isfinite(sep) & (sep > cfg.SEP_DEAD) & np.isfinite(meanv)
                    & (meanv < cfg.V_STOP))
        dmin = feats["dmin"][a:b]
        alone = ~(np.isfinite(dmin) & (dmin <= cfg.R_TOUCH))
    deep_s = _longest_run_s(deep, fps)
    flags = {
        "deep_stop_s": deep_s,
        "deep_stop": deep_s >= DEEP_STOP_S,
        "sep_stop": bool(sep_slow.mean() >= 0.5) if b > a else False,
        "ball_out": bool(row["outside_share_vis"] > 0) or (row["entry"].startswith("out")
                                                         and not row["in_inside"]),
        "placed": _longest_run_s(stationary[a:b] & alone & ball["visible"][a:b], fps)
        >= STATIONARY_ALONE_S,
        "goal_area": bool(abs(row["re_y"] - pitch_wid / 2.0) < GOAL_AREA_HALF_WIDTH_M
                          and (row["re_x"] < GOAL_AREA_DEPTH_M
                               or row["re_x"] > pitch_len - GOAL_AREA_DEPTH_M)),
        "no_chase": row["entry"].startswith("pursuit_stop"),
        "block_end": row["restart"] == "block_end" and not np.isfinite(
            feats["meanv_all"][b - 1]),
    }
    flags["keeper_hold"] = bool(row["entry"] == "evidence" and row["in_keeper"]
                                and row["in_in_box"] and row["in_dist"] <= cfg.R_TOUCH
                                and row["in_v"] < KEEPER_BALL_MS and row["sep"] <= cfg.SEP_DEAD
                                and not flags["deep_stop"])
    stop_keys = ("deep_stop", "sep_stop", "ball_out", "placed", "goal_area", "no_chase",
                 "block_end")
    flags["signatures"] = "+".join(k for k in stop_keys if flags[k])
    if flags["keeper_hold"]:
        flags["signatures"] += ("+" if flags["signatures"] else "") + "KEEPER_HOLD"
    stop = bool(flags["signatures"].replace("KEEPER_HOLD", "").strip("+"))
    if stop and not flags["keeper_hold"]:
        flags["flag_class"] = "supported"
    elif flags["keeper_hold"] and not stop:
        flags["flag_class"] = "unsupported"
    else:
        flags["flag_class"] = "doubtful"
    return flags


# -- output ---------------------------------------------------------------------------------------

def _fmt(value, spec="{:.2f}"):
    """A table cell: '-' for NaN, else the formatted value."""
    if isinstance(value, float) and not np.isfinite(value):
        return "-"
    return spec.format(value)


def markdown_tables(rows, alive, version_rows, stretch_rows):
    """list of Markdown lines: the aggregate tables and the per-segment table of the report."""
    out = []

    def table(header, body):
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "---|" * len(header))
        for line in body:
            out.append("| " + " | ".join(str(c) for c in line) + " |")
        out.append("")

    def med(values):
        values = [v for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
        return float(np.median(values)) if values else float("nan")

    total_s = sum(r["length_s"] for r in rows)
    out.append("Entry kinds ({} segments, {:.0f} s):".format(len(rows), total_s))
    out.append("")
    body = []
    for kind in sorted(set(r["entry"] for r in rows)):
        group = [r for r in rows if r["entry"] == kind]
        body.append((kind, len(group), "{:.0f}".format(sum(r["length_s"] for r in group)),
                     "{:.1f}".format(med([r["length_s"] for r in group])),
                     sum(1 for r in group if r["tagged"]),
                     sum(1 for r in group if r["flag_class"] == "supported")))
    table(("entry", "segments", "seconds", "median s", "tagged", "flags supported"), body)

    out.append("Restart kinds:")
    out.append("")
    body = []
    for kind in sorted(set(r["restart"] for r in rows)):
        group = [r for r in rows if r["restart"] == kind]
        body.append((kind, len(group), "{:.0f}".format(sum(r["length_s"] for r in group)),
                     sum(1 for r in group if r["chain"])))
    table(("restart", "segments", "seconds", "chain used"), body)

    out.append("Length classes:")
    out.append("")
    body = []
    for lo, hi in ((0, 10), (10, 20), (20, 60), (60, 1e9)):
        group = [r for r in rows if lo <= r["length_s"] < hi]
        tagged = [r for r in group if r["tagged"]]
        body.append(("{}-{} s".format(lo, int(hi)) if hi < 1e9 else ">= {} s".format(lo),
                     len(group), "{:.0f}".format(sum(r["length_s"] for r in group)),
                     len(tagged), len(group) - len(tagged),
                     "{:.0f}".format(sum(r["length_s"] for r in group if not r["tagged"]))))
    table(("length", "segments", "seconds", "tagged", "untagged", "untagged seconds"), body)

    out.append("Physics inside the segments (medians) against 20 s windows of alive play:")
    out.append("")
    keys = ("vis_share", "meanv", "slow_share", "sep", "near_share_vis", "fast_ball_share_vis")
    groups = (("tagged D", [r for r in rows if r["tagged"]]),
              ("untagged D", [r for r in rows if not r["tagged"]]),
              ("untagged D > 20 s", [r for r in rows if not r["tagged"]
                                     and r["length_s"] > LONG_UNTAGGED_S]),
              ("alive windows", alive))
    body = []
    for name, group in groups:
        body.append([name, len(group)] + [_fmt(med([g[k] for g in group])) for k in keys])
    table(("group", "n", "ball visible", "meanv m/s", "slow share", "sep m", "near share",
           "fast ball share"), body)

    out.append("Flag classes (tools rule, before review):")
    out.append("")
    body = []
    for cls in ("supported", "doubtful", "unsupported"):
        group = [r for r in rows if r["flag_class"] == cls]
        body.append((cls, len(group), "{:.0f}".format(sum(r["length_s"] for r in group)),
                     sum(1 for r in group if r["tagged"])))
    table(("class", "segments", "seconds", "tagged"), body)

    out.append("All `D` segments on graded frames:")
    out.append("")
    body = []
    for r in rows:
        body.append((r["start"], r["end"], "{:.1f}".format(r["length_s"]), r["half"],
                     "{} {}".format(r["entry"], r["entry_frame"]),
                     "{} {}{}".format(r["restart"], r["restart_frame"],
                                      " chain" if r["chain"] else ""),
                     r["tags"].replace("SET PIECES ", "SP ") or "-",
                     _fmt(r["vis_share"]), _fmt(r["meanv"]), _fmt(r["sep"], "{:.1f}"),
                     _fmt(r["near_share_vis"]), "{}/{}".format(r["contacts_inside"],
                                                               r["suppressed"]),
                     r["signatures"] or "-"))
    table(("start", "end", "s", "half", "entry", "restart", "tags within 15 s", "vis",
           "meanv", "sep", "near", "contacts/supp", "signatures"), body)

    if version_rows:
        out.append("Smart Tagging agreement per version:")
        out.append("")
        body = [(v["version"], v["agree"], v["alive"], "{:.2f}".format(100 * v["agreement"]))
                for v in version_rows]
        table(("version", "agree", "tagged and alive", "agreement %"), body)
        out.append("Per cycle (contributions in pp, exact split):")
        out.append("")
        body = []
        for cycle in [CYCLE_OF[v] for v in VERSIONS[1:]]:
            group = [s for s in stretch_rows if s["cycle"] == cycle]
            effect = [s for s in group if s["contrib_pp"] != 0]
            net = sum(s["contrib_pp"] for s in group)
            top = sorted(group, key=lambda s: -abs(s["contrib_pp"]))[:3]
            body.append((cycle, len(group), sum(s["frames"] for s in group), len(effect),
                         "{:+.3f}".format(net),
                         "{:+.3f}".format(sum(s["contrib_pp"] for s in effect
                                              if s["contrib_pp"] < 0)),
                         "{:+.3f}".format(sum(s["contrib_pp"] for s in effect
                                              if s["contrib_pp"] > 0)),
                         "{:+.3f}".format(sum(s["contrib_pp"] for s in top))))
        table(("cycle", "stretches", "frames", "with effect", "net", "losses", "gains",
               "top 3"), body)
        cols = ("cycle", "start", "end", "frames", "old", "new", "tag_team", "contrib_pp",
                "bump_new_start")
        for title, key in (("15 largest losses", lambda s: s["contrib_pp"]),
                           ("15 largest gains", lambda s: -s["contrib_pp"])):
            out.append(title + ":")
            out.append("")
            chosen = sorted([s for s in stretch_rows if s["contrib_pp"] != 0], key=key)[:15]
            table(cols, [[("{:+.3f}".format(s[c]) if c == "contrib_pp" else (s[c] or "-"))
                          for c in cols] for s in chosen])
        out.append("C8 stretches by class:")
        out.append("")
        body = []
        for cls in ("receive", "deflection", "other"):
            group = [s for s in stretch_rows if s["cycle"] == "C8" and s["c8_class"] == cls]
            body.append((cls, len(group), sum(s["frames"] for s in group),
                         "{:+.3f}".format(sum(s["contrib_pp"] for s in group)),
                         "{:+.3f}".format(sum(s["contrib_pp"] for s in group
                                              if s["contrib_pp"] < 0)),
                         "{:+.3f}".format(sum(s["contrib_pp"] for s in group
                                              if s["contrib_pp"] > 0))))
        table(("class", "stretches", "frames", "net", "losses", "gains"), body)
    return out


def write_csv(path, rows):
    """Write a list of dicts as CSV, columns in the order of the first row."""
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--labels", required=True, help="labels CSV of the default config")
    parser.add_argument("--out-dir", required=True, help="directory outside the repository")
    parser.add_argument("--versions-dir", default=None,
                        help="directory with the CSV of each version in VERSIONS")
    parser.add_argument("--trace", type=int, nargs="*", default=None,
                        help="print the trace around these frames and stop")
    parser.add_argument("--trace-s", type=float, default=TRACE_HALF_S,
                        help="half width of --trace in seconds")
    parser.add_argument("--trace-step", type=int, default=TRACE_STEP,
                        help="frames between --trace rows")
    args = parser.parse_args(argv)
    if os.path.abspath(args.out_dir).startswith(ROOT + os.sep):
        raise SystemExit("--out-dir must be outside the repository (match data)")
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config.default()
    inputs, parts, codes = engine_parts(cfg, args.labels)
    if args.trace:
        for frame in args.trace:
            print("\n".join(["trace at {}".format(frame)] + trace_lines(
                inputs, parts, codes, frame, cfg, half_s=args.trace_s, step=args.trace_step)))
        return 0
    keepers = keeper_slots(cfg, inputs)
    rows, stops, supp = segment_table(inputs, parts, codes, cfg, keepers)
    stationary = deadball.stationary_mask(inputs["ball"], cfg)
    for row in rows:
        row["reengage"] = reengage_frame(inputs, row["start"], row["end"], cfg)
        row["moving_s"], spans = moving_cover(inputs, row["start"], row["end"], cfg)
        row["moving_spans"] = ";".join("{}-{}".format(s, e) for s, e in spans)
        row.update(physics_flags(inputs, row, cfg, stationary))
    write_cases(os.path.join(args.out_dir, "cases.txt"), inputs, parts, codes, rows, cfg, supp)
    write_csv(os.path.join(args.out_dir, "segments.csv"), rows)
    write_csv(os.path.join(args.out_dir, "stoppages.csv"), stops)
    alive = alive_reference(inputs, codes, cfg, supp)
    write_csv(os.path.join(args.out_dir, "alive_windows.csv"), alive)
    untagged, _ = case_rows(rows, inputs["fps"])
    summary = {"n_segments": len(rows),
               "d_frames": int(sum(r["end"] - r["start"] for r in rows)),
               "flag_classes": {c: sum(1 for r in rows if r["flag_class"] == c)
                                for c in ("supported", "doubtful", "unsupported")},
               "untagged_long": [[int(r["start"]), int(r["end"]), r["flag_class"]]
                                 for r in untagged],
               "suppressed_contacts": int(sum(r["suppressed"] for r in rows))}
    version_rows, stretch_rows = [], []
    if args.versions_dir:
        version_rows, stretch_rows = tag_split(inputs, args.versions_dir, cfg)
        write_csv(os.path.join(args.out_dir, "tag_versions.csv"), version_rows)
        write_csv(os.path.join(args.out_dir, "tag_split.csv"), stretch_rows)
    with open(os.path.join(args.out_dir, "tables.md"), "w") as handle:
        handle.write("\n".join(markdown_tables(rows, alive, version_rows, stretch_rows)) + "\n")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
