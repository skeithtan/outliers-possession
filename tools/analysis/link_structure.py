"""Label-free structure of the cross-team continuation links (C24 backlog H3, step 1).

    python tools/analysis/link_structure.py --game NAME [--link-frames 3 25] [--cache-dir DIR]
        [--out OUT.json]

Reads the tracking of one match and prints summary counts only. It reads no label, so it can
run on the competition match (team rules). Not part of the submission.

The question (C23_CYCLE.md Section 1, ANSWER-010). T1's repair raised the BAR-ATH over-count.
Two hypotheses fit: (A) a duel opponent, one track ends and an opponent's track starts nearby;
(B) a true re-labelling, where the majority rule moved the wrong member of the chain. The
team totals just before and just after each cross-team link separate them:

- relabel_start: the team of the new tracklet goes 11 -> 12 and the team of the ended tracklet
  goes 11 -> 10. One player changed label. (B)
- relabel_end: the mirror, 12 -> 11 and 10 -> 11. The wrong label ends. (B)
- lost_found: the ended team goes 11 -> 10 and the new team goes 10 -> 11. One player lost
  and another found. (A), or an occlusion.
- handover: one over-count passes to the other team (12 -> 11 and 11 -> 12), or the mirror.
- other: everything else, with the raw (before, after) pairs tallied for the record.

For each link the gap (frames between the end and the start; 0 is the next frame), the
distance in metres, whether a third visible player of either team is within DUEL_M of the new
tracklet's first position (a duel) and whether the ball is within BALL_M are counted.

The same transition is tallied at the start and the end of each over-count episode (a run of
frames where a team shows more than MAX_TEAM tracks), whether or not a link explains it:
how the other team's count moves at that frame, and whether a cross-team link sits there.

Deterministic, numpy only, Python 3.9.
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.analysis import tracklet_chains as tc  # noqa: E402

MAX_TEAM = tc.MAX_TEAM
DUEL_M = 2.0
BALL_M = 3.0
# A link within this many frames of an over-count episode edge "sits there".
EDGE_FRAMES = 3


def classify(before_a, after_a, before_b, after_b):
    """The case of one cross-team link: team A ends a tracklet, team B starts one."""
    if before_a == MAX_TEAM and after_a == MAX_TEAM - 1 and before_b == MAX_TEAM \
            and after_b == MAX_TEAM + 1:
        return "relabel_start"
    if before_a == MAX_TEAM + 1 and after_a == MAX_TEAM and before_b == MAX_TEAM - 1 \
            and after_b == MAX_TEAM:
        return "relabel_end"
    if before_a == MAX_TEAM and after_a == MAX_TEAM - 1 and before_b == MAX_TEAM - 1 \
            and after_b == MAX_TEAM:
        return "lost_found"
    if before_a == MAX_TEAM + 1 and after_a == MAX_TEAM and before_b == MAX_TEAM \
            and after_b == MAX_TEAM + 1:
        return "handover"
    # Generalised forms: the same shape at other levels (a team already short or over).
    if after_b > MAX_TEAM and before_b <= MAX_TEAM and after_a < MAX_TEAM:
        return "over_begins_other_short"
    if before_a > MAX_TEAM and after_a <= MAX_TEAM and after_b <= MAX_TEAM:
        return "over_ends_other_filled"
    if after_b <= MAX_TEAM and after_a <= MAX_TEAM and before_a <= MAX_TEAM \
            and before_b <= MAX_TEAM:
        return "both_within_11"
    return "other"


def percentiles(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    return [round(float(np.percentile(values, p)), 2) for p in (10, 50, 90)]


def link_table(x, y, visible, spans, succ, team_of_slot, counts, bx, by, graded):
    """Per-link classification, summarised."""
    rows = []
    for a, b in succ.items():
        slot_a, start_a, stop_a = spans[a]
        slot_b, start_b, stop_b = spans[b]
        ta, tb = int(team_of_slot[slot_a]), int(team_of_slot[slot_b])
        if ta == tb:
            continue
        f_before, f_after = stop_a - 1, start_b
        before = counts[:, f_before]
        after = counts[:, f_after]
        case = classify(int(before[ta]), int(after[ta]), int(before[tb]), int(after[tb]))
        xa, ya = x[f_before, slot_a], y[f_before, slot_a]
        xb, yb = x[f_after, slot_b], y[f_after, slot_b]
        dist = float(np.hypot(xb - xa, yb - ya))
        # Third player within DUEL_M of B's first position, at B's first frame.
        others = visible[f_after].copy()
        others[slot_b] = False
        others[slot_a] = False
        d3 = np.hypot(x[f_after] - xb, y[f_after] - yb)
        near3 = others & (d3 <= DUEL_M)
        duel_team = None
        if near3.any():
            teams_near = set(int(t) for t in team_of_slot[near3])
            duel_team = "both" if len(teams_near) == 2 else ("same_as_a" if ta in teams_near
                                                              else "same_as_b")
        ball = float(np.hypot(xb - bx[f_after], yb - by[f_after])) if np.isfinite(bx[f_after]) \
            else np.nan
        rows.append({
            "case": case, "ta": ta, "gap": int(start_b - stop_a), "dist": dist,
            "duel": bool(near3.any()), "duel_team": duel_team,
            "near_ball": bool(np.isfinite(ball) and ball <= BALL_M),
            "len_a": stop_a - start_a, "len_b": stop_b - start_b,
            "pair": "{}{}->{}{} | {}{}->{}{}".format("A" if ta else "H", int(before[ta]),
                                                     "A" if ta else "H", int(after[ta]),
                                                     "A" if tb else "H", int(before[tb]),
                                                     "A" if tb else "H", int(after[tb])),
            "graded": bool(graded[f_after]),
        })
    out = {"cross_links": len(rows)}
    cases = Counter(r["case"] for r in rows)
    out["cases"] = dict(sorted(cases.items()))
    out["cases_near_ball"] = dict(sorted(Counter(r["case"] for r in rows if r["near_ball"]).items()))
    out["cases_graded"] = dict(sorted(Counter(r["case"] for r in rows if r["graded"]).items()))
    out["cases_with_duel"] = dict(sorted(Counter(r["case"] for r in rows if r["duel"]).items()))
    out["duel_team_by_case"] = {}
    for case in cases:
        out["duel_team_by_case"][case] = dict(sorted(Counter(
            str(r["duel_team"]) for r in rows if r["case"] == case).items()))
    out["gap_by_case"] = {case: dict(sorted(Counter(r["gap"] for r in rows
                                                    if r["case"] == case).items()))
                          for case in cases}
    out["dist_p10_p50_p90_by_case"] = {case: percentiles([r["dist"] for r in rows
                                                          if r["case"] == case])
                                       for case in cases}
    out["len_a_p10_p50_p90_by_case"] = {case: percentiles([r["len_a"] for r in rows
                                                           if r["case"] == case])
                                        for case in cases}
    out["len_b_p10_p50_p90_by_case"] = {case: percentiles([r["len_b"] for r in rows
                                                           if r["case"] == case])
                                        for case in cases}
    out["pairs_top"] = dict(Counter(r["pair"] for r in rows).most_common(16))
    out["pairs_other"] = dict(Counter(r["pair"] for r in rows if r["case"] == "other")
                              .most_common(12))
    return out, rows


def episode_table(counts, spans, succ, team_of_slot, x, y, bx, by, graded, n_frames):
    """What happens to the other team's count at each over-count episode start and end."""
    # Cross-team link frames: the new tracklet's first frame -> (a, b).
    link_at = {}
    for a, b in succ.items():
        if team_of_slot[spans[a][0]] != team_of_slot[spans[b][0]]:
            link_at.setdefault(spans[b][1], []).append((a, b))
    link_end_at = {}
    for a, b in succ.items():
        if team_of_slot[spans[a][0]] != team_of_slot[spans[b][0]]:
            link_end_at.setdefault(spans[a][2], []).append((a, b))
    out = {}
    for team, name in ((0, "home"), (1, "away")):
        over = counts[team] > MAX_TEAM
        starts, stops = tc.runs(over)
        keep = graded[np.clip(starts, 0, n_frames - 1)]
        starts, stops = starts[keep], stops[keep]
        other = 1 - team
        start_cases, end_cases = Counter(), Counter()
        start_link, end_link = 0, 0
        start_near_ball, end_near_ball = 0, 0
        lengths = (stops - starts).astype(np.float64)
        start_pairs, end_pairs = Counter(), Counter()
        sum_at_start = Counter()
        for s, e in zip(starts, stops):
            if s == 0:
                continue
            ob, oa = int(counts[other, s - 1]), int(counts[other, s])
            tb_, ta_ = int(counts[team, s - 1]), int(counts[team, s])
            start_pairs["{}->{} | other {}->{}".format(tb_, ta_, ob, oa)] += 1
            sum_at_start[int(counts[:, s].sum())] += 1
            if ob == MAX_TEAM and oa == MAX_TEAM - 1:
                start_cases["other 11->10 (relabel)"] += 1
            elif oa == ob:
                start_cases["other unchanged (extra track)"] += 1
            elif oa < ob:
                start_cases["other falls, from below 11"] += 1
            else:
                start_cases["other rises"] += 1
            linked = any(abs(f - s) <= EDGE_FRAMES for f in link_at)
            start_link += linked
            # the slot(s) that appeared at s on `team`
            if np.isfinite(bx[s]):
                new = [sl for sl, st, _ in spans if st == s and team_of_slot[sl] == team]
                if new and np.hypot(x[s, new[0]] - bx[s], y[s, new[0]] - by[s]) <= BALL_M:
                    start_near_ball += 1
            if e >= n_frames:
                continue
            ob, oa = int(counts[other, e - 1]), int(counts[other, e])
            tb_, ta_ = int(counts[team, e - 1]), int(counts[team, e])
            end_pairs["{}->{} | other {}->{}".format(tb_, ta_, ob, oa)] += 1
            if ob == MAX_TEAM - 1 and oa == MAX_TEAM:
                end_cases["other 10->11 (relabel ends)"] += 1
            elif oa == ob:
                end_cases["other unchanged (extra track ends)"] += 1
            elif oa > ob:
                end_cases["other rises, from elsewhere"] += 1
            else:
                end_cases["other falls"] += 1
            end_link += any(abs(f - e) <= EDGE_FRAMES for f in link_end_at)
            if np.isfinite(bx[e - 1]):
                gone = [sl for sl, _, sp in spans if sp == e and team_of_slot[sl] == team]
                if gone and np.hypot(x[e - 1, gone[0]] - bx[e - 1],
                                     y[e - 1, gone[0]] - by[e - 1]) <= BALL_M:
                    end_near_ball += 1
        out[name] = {
            "episodes": int(starts.size),
            "frames_p10_p50_p90": percentiles(lengths),
            "start_cases": dict(sorted(start_cases.items())),
            "end_cases": dict(sorted(end_cases.items())),
            "start_with_cross_link_within_3": int(start_link),
            "end_with_cross_link_within_3": int(end_link),
            "start_new_track_near_ball": int(start_near_ball),
            "end_gone_track_near_ball": int(end_near_ball),
            "start_pairs_top": dict(start_pairs.most_common(10)),
            "end_pairs_top": dict(end_pairs.most_common(10)),
            "total_tracks_at_start": dict(sorted(sum_at_start.items())),
        }
    return out


SHADOW_M = 1.0


def nearest_other(x, y, visible, slot, frames, cols):
    """Per-frame distance from `slot` to the nearest visible slot of `cols`, NaN if none."""
    out = np.full(len(frames), np.nan)
    for k, f in enumerate(frames):
        vis = cols[visible[f, cols]]
        if vis.size:
            out[k] = np.min(np.hypot(x[f, vis] - x[f, slot], y[f, vis] - y[f, slot]))
    return out


def birth_anatomy(x, y, visible, spans, team_of_slot, counts, bx, by, fps, slot, s, stop_b, e):
    """Label-free features of one tracklet born at `s` (episode end `e`, or None)."""
    team = int(team_of_slot[slot])
    other = 1 - team
    same_cols = np.flatnonzero(team_of_slot == team)
    same_cols = same_cols[same_cols != slot]
    other_cols = np.flatnonzero(team_of_slot == other)
    life_end = min(stop_b, e) if e is not None else stop_b
    frames = np.arange(s, life_end)
    if frames.size > 400:
        frames = frames[np.linspace(0, frames.size - 1, 400).astype(int)]
    d_same = nearest_other(x, y, visible, slot, frames, same_cols)
    d_other = nearest_other(x, y, visible, slot, frames, other_cols)
    med_same = float(np.nanmedian(d_same)) if np.isfinite(d_same).any() else np.nan
    med_other = float(np.nanmedian(d_other)) if np.isfinite(d_other).any() else np.nan
    if np.isfinite(med_other) and med_other <= SHADOW_M and not (np.isfinite(med_same)
                                                                 and med_same < med_other):
        shadow = "shadow_of_opponent"
    elif np.isfinite(med_same) and med_same <= SHADOW_M:
        shadow = "shadow_of_teammate"
    else:
        shadow = "free"
    # The other team's most recent loss before s.
    last = None
    for sl, st, sp in spans:
        if team_of_slot[sl] == other and sp <= s and (last is None or sp > last[2]):
            last = (sl, st, sp)
    if last is not None:
        gap = s - last[2]
        dist = float(np.hypot(x[s, slot] - x[last[2] - 1, last[0]],
                              y[s, slot] - y[last[2] - 1, last[0]]))
    else:
        gap, dist = None, None
    ball = (float(np.hypot(x[s, slot] - bx[s], y[s, slot] - by[s]))
            if np.isfinite(bx[s]) else np.nan)
    return {
        "len": int(stop_b - s), "ends_episode": bool(e is not None and stop_b == e),
        "shadow": shadow, "med_same": med_same, "med_other": med_other,
        "d_other_birth": float(d_other[0]) if frames.size else np.nan,
        "d_same_birth": float(d_same[0]) if frames.size else np.nan,
        "gap_last_loss": gap, "dist_last_loss": dist,
        "loss_plausible": (dist is not None and dist <= tc.NOISE_M + tc.V_MAX * gap / float(fps)),
        "near_ball": bool(np.isfinite(ball) and ball <= BALL_M),
        "total": int(counts[:, s].sum()), "other_count": int(counts[other, s]),
    }


def summarise_births(rows):
    out = {"n": len(rows)}
    if not rows:
        return out
    out["shadow"] = dict(sorted(Counter(r["shadow"] for r in rows).items()))
    out["shadow_by_total"] = {}
    for total in sorted({r["total"] for r in rows}):
        out["shadow_by_total"][str(total)] = dict(sorted(Counter(
            r["shadow"] for r in rows if r["total"] == total).items()))
    out["ends_episode"] = int(sum(r["ends_episode"] for r in rows))
    out["near_ball_at_birth"] = int(sum(r["near_ball"] for r in rows))
    out["len_p10_p50_p90"] = percentiles([r["len"] for r in rows])
    out["d_other_birth_p10_p50_p90"] = percentiles([r["d_other_birth"] for r in rows
                                                    if np.isfinite(r["d_other_birth"])])
    out["d_same_birth_p10_p50_p90"] = percentiles([r["d_same_birth"] for r in rows
                                                   if np.isfinite(r["d_same_birth"])])
    out["med_other_p10_p50_p90"] = percentiles([r["med_other"] for r in rows
                                                if np.isfinite(r["med_other"])])
    out["med_same_p10_p50_p90"] = percentiles([r["med_same"] for r in rows
                                               if np.isfinite(r["med_same"])])
    for limit in (2.0, 3.0):
        out["med_same_le_{:.0f}m".format(limit)] = int(sum(
            np.isfinite(r["med_same"]) and r["med_same"] <= limit for r in rows))
        out["d_same_birth_le_{:.0f}m".format(limit)] = int(sum(
            np.isfinite(r["d_same_birth"]) and r["d_same_birth"] <= limit for r in rows))
        out["med_other_le_{:.0f}m".format(limit)] = int(sum(
            np.isfinite(r["med_other"]) and r["med_other"] <= limit for r in rows))
    gaps = [r["gap_last_loss"] for r in rows if r["gap_last_loss"] is not None]
    out["gap_last_loss_frames_p10_p50_p90"] = percentiles(gaps)
    out["gap_last_loss_le_25"] = int(sum(g <= 25 for g in gaps))
    out["gap_last_loss_le_125"] = int(sum(g <= 125 for g in gaps))
    out["dist_last_loss_p10_p50_p90"] = percentiles([r["dist_last_loss"] for r in rows
                                                     if r["dist_last_loss"] is not None])
    out["dist_last_loss_le_3m"] = int(sum(r["dist_last_loss"] is not None
                                          and r["dist_last_loss"] <= 3.0 for r in rows))
    out["loss_plausible"] = int(sum(r["loss_plausible"] for r in rows))
    out["other_count_at_birth"] = dict(sorted(Counter(r["other_count"] for r in rows).items()))
    return out


def episode_anatomy(x, y, visible, spans, team_of_slot, counts, bx, by, fps, graded, n_frames):
    """The tracklet born at each over-count episode start, against a control of normal births."""
    by_start = {}
    for index, (sl, st, sp) in enumerate(spans):
        by_start.setdefault(st, []).append(index)
    out = {}
    for team, name in ((0, "home"), (1, "away")):
        over = counts[team] > MAX_TEAM
        starts, stops = tc.runs(over)
        keep = graded[np.clip(starts, 0, n_frames - 1)]
        starts, stops = starts[keep], stops[keep]
        episode_rows, multi = [], 0
        episode_starts = set(int(s) for s in starts)
        for s, e in zip(starts, stops):
            born = [spans[i] for i in by_start.get(int(s), []) if team_of_slot[spans[i][0]] == team]
            if len(born) != 1:
                multi += 1
                continue
            sl, st, sp = born[0]
            episode_rows.append(birth_anatomy(x, y, visible, spans, team_of_slot, counts, bx, by,
                                              fps, sl, st, sp, int(e)))
        # Control: births of this team on graded frames that leave the team at 11 or fewer,
        # with the count at 11 after the birth (the same count level, no over-count).
        control_rows = []
        for st, indices in by_start.items():
            if st in episode_starts or st >= n_frames or not graded[st] or st == 0:
                continue
            for i in indices:
                sl, _, sp = spans[i]
                if team_of_slot[sl] != team or counts[team, st] != MAX_TEAM \
                        or counts[team, st - 1] != MAX_TEAM - 1:
                    continue
                control_rows.append(birth_anatomy(x, y, visible, spans, team_of_slot, counts,
                                                  bx, by, fps, sl, st, sp, None))
        out[name] = {"episode_births": summarise_births(episode_rows),
                     "episodes_with_several_births": int(multi),
                     "control_births_10_to_11": summarise_births(control_rows)}
    return out


CLUSTER_GAP = 250
TOUCHLINE_M = 3.0


def slot_table(x, y, visible, team_of_slot, counts, pitch, graded, bx, by):
    """Per slot: how much of its visible time the team is over-counted, and where it lives."""
    out = {}
    for team, name in ((0, "home"), (1, "away")):
        over = counts[team] > MAX_TEAM
        cols = np.flatnonzero(team_of_slot == team)
        other_cols = np.flatnonzero(team_of_slot == 1 - team)
        rows = []
        for slot in cols:
            vis = visible[:, slot] & graded
            n_vis = int(vis.sum())
            if n_vis == 0:
                continue
            n_over = int((vis & over).sum())
            fr = np.flatnonzero(vis)
            sample = fr[np.linspace(0, fr.size - 1, min(fr.size, 300)).astype(int)]
            ys = y[sample, slot]
            touch = float(np.mean((ys <= TOUCHLINE_M) | (ys >= pitch[1] - TOUCHLINE_M)))
            same = cols[cols != slot]
            d_same = nearest_other(x, y, visible, slot, sample, same)
            d_other = nearest_other(x, y, visible, slot, sample, other_cols)
            ball = np.hypot(x[sample, slot] - bx[sample], y[sample, slot] - by[sample])
            rows.append({
                "slot": int(slot), "visible": n_vis, "over": n_over,
                "over_share": round(n_over / float(n_vis), 3),
                "touchline_share": round(touch, 3),
                "med_same_m": round(float(np.nanmedian(d_same)), 2) if np.isfinite(d_same).any()
                else None,
                "med_other_m": round(float(np.nanmedian(d_other)), 2)
                if np.isfinite(d_other).any() else None,
                "ball_within_3m_share": round(float(np.nanmean(ball <= BALL_M)), 3)
                if np.isfinite(ball).any() else None,
                "tracklets": int(tc.runs(visible[:, slot])[0].size),
            })
        rows.sort(key=lambda r: -r["over_share"])
        total_over = int(over[graded].sum())
        out[name] = {
            "slots_used": len(rows), "over_frames_graded": total_over,
            "over_frames_in_slots_with_share_ge_0.5": int(sum(
                r["over"] for r in rows if r["over_share"] >= 0.5)),
            "over_frames_in_slots_with_share_ge_0.3": int(sum(
                r["over"] for r in rows if r["over_share"] >= 0.3)),
            "top_slots": rows[:14],
            "over_share_histogram": dict(sorted(Counter(
                "{:.1f}".format(np.floor(r["over_share"] * 10) / 10) for r in rows).items())),
        }
    return out


def cluster_table(counts, graded, n_frames, gap=CLUSTER_GAP):
    """Over-count episodes merged across gaps of at most `gap` frames."""
    out = {}
    for team, name in ((0, "home"), (1, "away")):
        over = counts[team] > MAX_TEAM
        starts, stops = tc.runs(over)
        clusters = []
        for s, e in zip(starts, stops):
            if clusters and s - clusters[-1][1] <= gap:
                clusters[-1][1] = int(e)
                clusters[-1][2] += 1
                clusters[-1][3] += int(e - s)
            else:
                clusters.append([int(s), int(e), 1, int(e - s)])
        clusters = [c for c in clusters if graded[c[0]:c[1]].any()]
        spans = np.array([c[1] - c[0] for c in clusters], dtype=np.float64)
        eps = np.array([c[2] for c in clusters], dtype=np.float64)
        out[name] = {
            "clusters": len(clusters),
            "span_frames_p10_p50_p90": percentiles(spans),
            "episodes_per_cluster_p50_p90": percentiles(eps)[1:] if eps.size else None,
            "over_frames_in_clusters_with_ge_3_episodes": int(sum(
                c[3] for c in clusters if c[2] >= 3)),
            "over_frames_total": int(sum(c[3] for c in clusters)),
            "largest_clusters_span_episodes_over": sorted(
                [[c[1] - c[0], c[2], c[3]] for c in clusters], key=lambda r: -r[2])[:8],
        }
    return out


def analyse(tactical, pitch, fps, blocks, link_frames_list):
    x, y, visible = tc.metres(tactical, pitch)
    team_of_slot = np.asarray(tactical["team_of_slot"])
    n_frames = visible.shape[0]
    graded = tc.graded_mask(blocks, n_frames)
    counts = tc.team_counts(visible, team_of_slot)
    bx = np.asarray(tactical["bx"], dtype=np.float64) * float(pitch[0])
    by = np.asarray(tactical["by"], dtype=np.float64) * float(pitch[1])
    spans = tc.tracklets(visible)
    result = {"tracklets": len(spans), "links": {}}
    rows_by_link = {}
    for lf in link_frames_list:
        succ = tc.continuation_links(x, y, spans, fps, lf)
        table, rows = link_table(x, y, visible, spans, succ, team_of_slot, counts, bx, by, graded)
        table["all_links"] = len(succ)
        table["all_links_graded"] = int(sum(1 for a, b in succ.items()
                                            if graded[spans[b][1]]))
        result["links"][str(lf)] = table
        rows_by_link[lf] = (succ, rows)
    succ3 = rows_by_link[link_frames_list[0]][0]
    result["episodes"] = episode_table(counts, spans, succ3, team_of_slot, x, y, bx, by,
                                       graded, n_frames)
    result["episode_anatomy"] = episode_anatomy(x, y, visible, spans, team_of_slot, counts,
                                                bx, by, fps, graded, n_frames)
    result["slots"] = slot_table(x, y, visible, team_of_slot, counts, pitch, graded, bx, by)
    result["clusters"] = cluster_table(counts, graded, n_frames)
    result["n_slots"] = {"home": int((team_of_slot == 0).sum()),
                         "away": int((team_of_slot == 1).sum())}
    # Count anatomy on graded frames for the record.
    result["count_distribution_graded"] = {
        name: {str(k): int(v) for k, v in sorted(Counter(counts[t, graded].tolist()).items())}
        for t, name in ((0, "home"), (1, "away"))}
    return result, rows_by_link


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--link-frames", type=int, nargs="+", default=[3, 25])
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    from engine import data

    game = data.resolve_game(args.game)
    tactical = data.load_tactical(game, cache_dir=args.cache_dir)
    pitch, fps, blocks = data.pitch_m(game), data.fps(game), data.graded_blocks(game)
    result, _ = analyse(tactical, pitch, fps, blocks, args.link_frames)
    result["game"] = game
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
