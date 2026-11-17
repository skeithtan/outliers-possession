"""R2-c: is D under-called or over-called on the full match? (RESEARCH open question 4).

Measurements only. Reads the engine output (possession.prepare + run), the tactical ATD, Smart
Tagging and, for the public window only, evaluation/gt/BAR-ATH_possession_gt_public.csv through
tools/gt.py. Never opens the full-match ground truth.

Parts:
  1. under-call candidates: graded H/A stretches >= 3 s where a collective-stop signal holds
       S1  meanv_near < 1.2 m/s (engine feature, defined only while the ball is visible)
       S1h meanv_near_held < 1.2 (6 nearest to the last seen ball position, defined in gaps too)
       S2  meanv_all < 1.6 and sep > 13 (the e_D collective-stop terms)
     with ball state, set-piece window, events, onsets (incl. pre-confirmation ones) and a class.
  2. over-call candidates: every decoded D segment with entry mechanism, exit kind, movement.
  3. public window: GT label on every candidate stretch and on every engine D segment.
  4. expected stoppage count: Smart Tagging restarts and the ball-gap structure.
  5. spellF1 cost on the public window: delete each engine D segment, insert a 3 s D segment
     into the middle of each predicted spell.
  6. out onsets fired by a ball that reappears beyond a line (throw-in being taken), and a scratch
     variant of the engine without them (inputs copied, engine code untouched).
  7. stationary ball on alive frames (quick restart candidates).
  8. hard onsets confirmed only by a collective stop that began before them.
  9. ball exits over a pitch line and engine D coverage.
 10. D segments by cause (out of play, other, goal, pre-kickoff).

Usage: python tools/analysis/r2_dcall.py <out_dir>
Writes CSVs into <out_dir> and prints summaries. Deterministic.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from engine import data, deadball, possession  # noqa: E402
from engine.config import Config  # noqa: E402
from engine.tracks import _runs  # noqa: E402
import possession_metrics as pm  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)
FPS = 25
MIN_LEN = 75                       # 3 s
L, W = data.pitch_m()
PUB = (11185, 18686)               # public GT valid run (half-open)
PA_X, PA_HALF_Y = 16.5, 20.16      # penalty area depth, half width (laws of the game)
EDGE_M = 3.0                       # "near a line"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)


# -- load -------------------------------------------------------------------------------------
cfg = Config.default()
inputs = possession.prepare(cfg)
labels, parts = possession.run(cfg, return_parts=True, inputs=inputs)
ball, feats, ev, onsets = inputs["ball"], inputs["feats"], inputs["events"], inputs["onsets"]
cands, tagging, blocks = inputs["restarts"], inputs["tagging"], inputs["blocks"]
kickoffs, e_dead, ghost = inputs["kickoffs"], inputs["e_dead"], inputs["ghost"]
N = labels.shape[0]
graded = np.zeros(N, dtype=bool)
half_of = np.zeros(N, dtype=np.int8)
for h, f, l in blocks:
    graded[f:l + 1] = True
    half_of[f:l + 1] = h
vis = ball["visible"]
bx, by, bspeed = ball["x"], ball["y"], ball["speed"]
stoppage = parts["stoppage"]
isD = labels == "D"
alive = graded & ~isD

tac = data.load_tactical()
px = np.asarray(tac["px"], dtype=np.float64) * L
py = np.asarray(tac["py"], dtype=np.float64) * W
ps = np.asarray(tac["ps"], dtype=np.float64)
pvis = np.isfinite(px) & np.isfinite(py)
team_of_slot = np.asarray(tac["team_of_slot"])

# ball position held through gaps (last visible position), NaN before the first sighting
idx = np.where(vis, np.arange(N), -1)
idx = np.maximum.accumulate(idx)
hx = np.where(idx >= 0, bx[np.maximum(idx, 0)], np.nan)
hy = np.where(idx >= 0, by[np.maximum(idx, 0)], np.nan)
d_held = np.hypot(px - hx[:, None], py - hy[:, None])
d_held = np.where(np.isfinite(d_held), d_held, np.inf)
cols6 = np.argpartition(d_held, 5, axis=1)[:, :6]
rows = np.arange(N)[:, None]
sp6 = np.where(np.isfinite(d_held[rows, cols6]), ps[rows, cols6], np.nan)
import warnings  # noqa: E402
with np.errstate(invalid="ignore"), warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    meanv_near_held = np.nanmean(np.where(np.isfinite(sp6), sp6, np.nan), axis=1)

# own goal side per team and half: team centroid at the half's kickoff frame
own_goal_x = {}
for h, f, l in blocks:
    k = kickoffs[h]
    for t, key in ((0, "H"), (1, "A")):
        cx = feats["cx" + key][k]
        own_goal_x[(h, t)] = 0.0 if cx < L / 2 else L


def lt(v, thr):
    v = np.asarray(v, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.isfinite(v) & (v < thr)


def gt_(v, thr):
    v = np.asarray(v, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.isfinite(v) & (v > thr)


SIGNALS = {
    "S1": lt(feats["meanv_near"], 1.2),
    "S1h": lt(meanv_near_held, 1.2),
    "S2": lt(feats["meanv_all"], 1.6) & gt_(feats["sep"], 13.0),
}

# pre-confirmation onset candidates (everything the out / pursuit rules propose)
_horizon = int(round(cfg.ONSET_CONFIRM_S * FPS))
_all = (np.array([0]), np.array([N]))
raw_out = deadball._out_onsets(ball, cfg, (L, W), FPS, _all[0], _all[1], _horizon)
raw_pur = deadball._pursuit_onsets(ball, feats, cfg, _all[0], _all[1], _horizon)
raw_onsets = pd.DataFrame(sorted(set(raw_out + raw_pur)), columns=["frame", "mechanism", "conf"])
kept = set(zip(onsets["frame"], onsets["mechanism"]))
raw_onsets["confirmed"] = [(f, m) in kept for f, m in zip(raw_onsets["frame"],
                                                           raw_onsets["mechanism"])]

setp = tagging[tagging["code"].isin(["SET PIECES", "GOALS"])].reset_index(drop=True)
EVK = ("touch", "release", "control", "dribble", "contested")
ev_frames = {k: ev.loc[ev["kind"] == k, "frame"].to_numpy() for k in EVK}
in_pa = (vis & ((bx < PA_X) | (bx > L - PA_X)) & (np.abs(by - W / 2) < PA_HALF_Y))
outside = deadball.outside_pitch(ball, (L, W), cfg.PITCH_MARGIN_M)
near_touch = vis & ((by < EDGE_M) | (by > W - EDGE_M))
near_goal_line = vis & ((bx < EDGE_M) | (bx > L - EDGE_M))


def deepest_mask():
    """bool (N,): the nearest player to the (visible) ball is his team's deepest outfield-or-GK
    player (closest to his own goal line) and within 2 m: a goalkeeper proxy (no roles in ATD)."""
    out = np.zeros(N, dtype=bool)
    for h, f, l in blocks:
        for t, key in ((0, "H"), (1, "A")):
            cols = np.flatnonzero(team_of_slot == t)
            gx = own_goal_x[(h, t)]
            dist_goal = np.where(pvis[f:l + 1][:, cols], np.abs(px[f:l + 1][:, cols] - gx), np.inf)
            deepest = cols[np.argmin(dist_goal, axis=1)]
            slot = feats["nearest_slot_" + key][f:l + 1]
            ok = (feats["nearest_team"][f:l + 1] == t) & (slot == deepest) & lt(
                feats["dmin"][f:l + 1], 2.0)
            out[f:l + 1] |= ok
    return out


keeper_near = deepest_mask()


def nearest_setpiece(s, e):
    """(distance in frames from [s, e) to the nearest SET PIECES/GOALS window, type, code)."""
    best = (np.inf, None, None, None)
    for r in setp.itertuples(index=False):
        a, b = int(r.start_frame), int(r.end_frame)
        dist = 0 if (a < e and s < b) else (a - e if a >= e else s - b)
        if dist < best[0]:
            best = (dist, r.type if r.code == "SET PIECES" else "GOAL", r.team, a)
    return best


def ball_disp(s, e):
    v = vis[s:e]
    if v.sum() < 2:
        return np.nan, np.nan
    x, y = bx[s:e][v], by[s:e][v]
    diag = float(np.hypot(x.max() - x.min(), y.max() - y.min()))
    path = float(np.nansum(np.hypot(np.diff(x), np.diff(y))))
    return diag, path


def d_runs():
    """[(start, stop)] decoded D runs on the graded frames, per block."""
    out = []
    for h, f, l in blocks:
        s, e = _runs(isD[f:l + 1])
        out += [(int(a) + f, int(b) + f) for a, b in zip(s, e)]
    return out


DRUNS = d_runs()
D_START = np.array([a for a, _ in DRUNS])
D_STOP = np.array([b for _, b in DRUNS])


def why_no_d(s, e):
    """Why the state machine did not enter (or hold) D on [s, e)."""
    win_lo = s - 10 * FPS
    near = onsets[(onsets["frame"] >= win_lo) & (onsets["frame"] < e)]
    hard = near[near["mechanism"] != "collective_stop"]
    soft = near[near["mechanism"] == "collective_stop"]
    raw = raw_onsets[(raw_onsets["frame"] >= win_lo) & (raw_onsets["frame"] < e)
                     & ~raw_onsets["confirmed"]]
    prev = D_STOP[D_STOP <= s]
    gap_prev = int(s - prev.max()) if prev.size else 10 ** 6
    nxt = D_START[D_START >= e]
    gap_next = int(nxt.min() - e) if nxt.size else 10 ** 6
    eD_hi = float(np.mean(e_dead[s:e] > cfg.DEAD_EVIDENCE_HIGH))
    if len(hard):
        r = hard.iloc[-1]
        reason = "entered_exited" if gap_prev <= (e - s) + 10 * FPS else "onset_ignored?"
        detail = "{}@{}".format(r["mechanism"], int(r["frame"]))
    elif len(raw):
        r = raw.iloc[-1]
        reason, detail = "onset_unconfirmed", "{}@{}".format(r["mechanism"], int(r["frame"]))
    elif len(soft):
        reason, detail = "collective_soft_only", "collective@{}".format(int(soft.iloc[-1]["frame"]))
    else:
        reason, detail = "no_onset", ""
    return reason, detail, gap_prev, gap_next, eD_hi


def classify(row):
    """Physical class of an under-call stretch (rules in the order given)."""
    if row["ghost_frac"] > 0.5 or row["ntracks_med"] < 15:
        return "tracking_artefact"
    if row["outside_frames"] > 0:
        return "ball_out_seen"
    if row["vis_frac"] < 0.5:
        return "ball_gap"
    if row["keeper_frac"] > 0.5:
        return "keeper_holding"
    if (row["disp_diag"] < 2.0) and row["n_touch_like"] == 0:
        return "ball_still_no_touch"
    if row["n_touch_like"] >= 1 or row["disp_diag"] >= 5.0:
        return "slow_play_ball_moving"
    return "ball_slow_drift"


def candidates(sig_name):
    mask = SIGNALS[sig_name] & alive
    rows_ = []
    for h, f, l in blocks:
        s_, e_ = _runs(mask[f:l + 1])
        for s, e in zip(s_ + f, e_ + f):
            s, e = int(s), int(e)
            if e - s < MIN_LEN:
                continue
            labs = labels[s:e]
            diag, path = ball_disp(s, e)
            v = vis[s:e]
            n_ev = {k: int(((ev_frames[k] >= s) & (ev_frames[k] < e)).sum()) for k in EVK}
            dist, sptype, spteam, spstart = nearest_setpiece(s, e)
            reason, detail, gp, gn, eD_hi = why_no_d(s, e)
            sp = bspeed[s:e][v]
            rows_.append(dict(
                signal=sig_name, half=h, start=s, end=e, length=e - s, secs=(e - s) / FPS,
                label="".join(sorted(set(labs))), vis_frac=float(v.mean()),
                ball_speed_med=float(np.median(sp)) if sp.size else np.nan,
                ball_speed_max=float(sp.max()) if sp.size else np.nan,
                disp_diag=diag, path=path,
                ball_x=float(hx[s + (e - s) // 2]), ball_y=float(hy[s + (e - s) // 2]),
                in_pa_frac=float(in_pa[s:e].mean()), outside_frames=int(outside[s:e].sum()),
                near_touch_frac=float(near_touch[s:e].mean()),
                near_goal_line_frac=float(near_goal_line[s:e].mean()),
                keeper_frac=float(keeper_near[s:e][v].mean()) if v.any() else 0.0,
                ghost_frac=float(ghost[s:e].mean()),
                ntracks_med=float(np.median(feats["n_tracks"][s:e])),
                meanv_all_med=float(np.nanmedian(feats["meanv_all"][s:e])),
                meanv_near_med=float(np.nanmedian(meanv_near_held[s:e])),
                sep_med=float(np.nanmedian(feats["sep"][s:e])),
                dmin_med=float(np.nanmedian(feats["dmin"][s:e])) if v.any() else np.nan,
                n_touch_like=n_ev["touch"] + n_ev["release"] + n_ev["control"] + n_ev["dribble"],
                **{"n_" + k: n_ev[k] for k in EVK},
                sp_dist=dist, sp_type=sptype, sp_team=spteam, sp_start=spstart,
                why=reason, why_detail=detail, gap_prev_D=gp, gap_next_D=gn, eD_hi_frac=eD_hi))
    tab = pd.DataFrame(rows_)
    if len(tab):
        tab["cls"] = tab.apply(classify, axis=1)
    return tab


def part1():
    print("\n=== PART 1: under-call candidates (graded, labelled H/A, signal >= 3 s) ===")
    allc = []
    for name in SIGNALS:
        tab = candidates(name)
        allc.append(tab)
        print("\n-- {}: {} stretches, {} frames ({:.1f} s)".format(
            name, len(tab), int(tab["length"].sum()) if len(tab) else 0,
            tab["length"].sum() / FPS if len(tab) else 0))
        if not len(tab):
            continue
        g = tab.groupby("cls").agg(n=("length", "size"), frames=("length", "sum"),
                                   med_len=("length", "median"))
        print(g.to_string())
        print(tab.groupby("why").agg(n=("length", "size"), frames=("length", "sum")).to_string())
        tab["sp_within_10s"] = tab["sp_dist"] <= 250
        print("within 10 s of a SET PIECES/GOALS window:", int(tab["sp_within_10s"].sum()))
        show = ["start", "end", "secs", "label", "cls", "why", "why_detail", "vis_frac",
                "ball_speed_med", "disp_diag", "path", "ball_x", "ball_y", "in_pa_frac",
                "keeper_frac", "meanv_all_med", "meanv_near_med", "sep_med", "dmin_med",
                "n_touch_like", "sp_dist", "sp_type", "gap_prev_D", "gap_next_D", "eD_hi_frac"]
        print(tab[show].round(2).to_string())
    allc = pd.concat(allc, ignore_index=True)
    allc.to_csv(os.path.join(OUT, "undercall_candidates.csv"), index=False)
    return allc




# -- part 2: over-call candidates ------------------------------------------------------------
DWELL = int(round(cfg.COLLECTIVE_STOP_S * FPS))
_hs, _he = _runs(e_dead > cfg.DEAD_EVIDENCE_HIGH)
DWELL_STARTS = set(int(a) for a, b in zip(_hs, _he) if b - a >= DWELL)
HARD_ONSETS = onsets[onsets["mechanism"] != "collective_stop"]
CONF_CANDS = cands[cands["confirmed"]]
RUN_FAST = 2.0       # m/s, mean speed of all players clearly above V_STOP: play is going on
ALIVE_RUN_S = 5.0    # a 5 s stretch of fast players with the ball being played


def longest_true(mask):
    s, e = _runs(mask)
    return (int((e - s).max()), int(s[np.argmax(e - s)])) if len(s) else (0, -1)


def entry_kind(s):
    if s in kickoffs.values():
        return "kickoff"
    for h, f, l in blocks:
        if s == f:
            return "block_start(pre-kickoff)"
    rows_ = HARD_ONSETS[HARD_ONSETS["frame"] == s]
    if len(rows_):
        return rows_["mechanism"].iloc[0]
    if s in DWELL_STARTS:
        return "eD_dwell"
    near = HARD_ONSETS[(HARD_ONSETS["frame"] - s).abs() <= 25]
    if len(near):
        return "~" + near["mechanism"].iloc[0]
    return "decoder"


def exit_kind(e):
    for h, f, l in blocks:
        if e == l + 1:
            return "block_end"
    if e in kickoffs.values():
        return "kickoff"
    rows_ = CONF_CANDS[CONF_CANDS["frame"] == e]
    if len(rows_):
        return rows_["kind"].iloc[0]
    # DELTA_RESTART_REAPPEAR shifts reappear exits
    rows_ = CONF_CANDS[CONF_CANDS["frame"] == e - int(cfg.DELTA_RESTART_REAPPEAR)]
    if len(rows_):
        return rows_["kind"].iloc[0] + "+d"
    near = CONF_CANDS[(CONF_CANDS["frame"] - e).abs() <= 25]
    if len(near):
        return "~" + near["kind"].iloc[0]
    return "decoder"


def part2():
    print("\n=== PART 2: every decoded D segment ===")
    rows_ = []
    mv = feats["meanv_all"]
    fast = gt_(mv, RUN_FAST)
    touchlike = np.zeros(N, dtype=bool)
    for k in ("touch", "release", "control", "dribble"):
        touchlike[ev_frames[k]] = True
    for s, e in DRUNS:
        diag, path = ball_disp(s, e)
        v = vis[s:e]
        n_tl = int(touchlike[s:e].sum())
        lf, lf_at = longest_true(fast[s:e])
        # inner alive stretch: fast players for >= ALIVE_RUN_S with >= 2 touch-like events in it
        inner = []
        fs, fe = _runs(fast[s:e])
        for a, b in zip(fs + s, fe + s):
            if b - a >= ALIVE_RUN_S * FPS and touchlike[a:b].sum() >= 2:
                inner.append((int(a), int(b)))
        dist, sptype, spteam, spstart = nearest_setpiece(s, e)
        inside_c = cands[(cands["frame"] > s) & (cands["frame"] < e)]
        rows_.append(dict(
            half=int(half_of[s]), start=s, end=e, length=e - s, secs=(e - s) / FPS,
            entry=entry_kind(s), exit=exit_kind(e),
            prev_label=labels[s - 1] if graded[s - 1] else "X",
            next_label=labels[e] if e < N and graded[e] else "X",
            vis_frac=float(v.mean()), ball_speed_max=float(np.nanmax(bspeed[s:e][v])) if v.any()
            else np.nan, disp_diag=diag, path=path,
            meanv_all_med=float(np.nanmedian(mv[s:e])), meanv_all_max=float(np.nanmax(mv[s:e])),
            frac_fast=float(fast[s:e].mean()), longest_fast_s=lf / FPS,
            longest_fast_at=lf_at + s if lf_at >= 0 else -1,
            n_touch_like=n_tl, n_inner_alive=len(inner),
            inner_alive=";".join("{}-{}".format(a, b) for a, b in inner),
            n_cand_inside=len(inside_c), n_cand_unconf_inside=int((~inside_c["confirmed"]).sum()),
            sp_dist=dist, sp_type=sptype, sp_start=spstart))
    tab = pd.DataFrame(rows_)
    tab["alive_flag"] = (tab["n_inner_alive"] > 0) | ((tab["path"] > 30) & (tab["n_touch_like"] >= 3)
                                                     & (tab["frac_fast"] > 0.3))
    tab.to_csv(os.path.join(OUT, "d_segments.csv"), index=False)
    print(tab.drop(columns=["inner_alive"]).round(2).to_string())
    print("\nentry kinds:\n", tab.groupby("entry").agg(n=("length", "size"), frames=("length", "sum"),
                                                      med_s=("secs", "median")).to_string())
    print("\nexit kinds:\n", tab.groupby("exit").agg(n=("length", "size"), frames=("length", "sum"),
                                                    med_s=("secs", "median")).to_string())
    for h, f, l in blocks:
        nh = l + 1 - f
        sub = tab[tab["half"] == h]
        print("\nhalf {} D share {:.3f}  ({} segs)".format(h, sub["length"].sum() / nh, len(sub)))
        print((sub.groupby("entry")["length"].sum() / nh).round(4).to_string())
    print("\nlength distribution (s):", tab["secs"].describe().round(1).to_dict())
    print("segments > 60 s:", int((tab["secs"] > 60).sum()), " frames", int(tab.loc[tab["secs"] > 60, "length"].sum()))
    fl = tab[tab["alive_flag"]]
    print("\nalive-looking D segments: {}  frames {}".format(len(fl), int(fl["length"].sum())))
    print(fl[["start", "end", "secs", "entry", "exit", "frac_fast", "longest_fast_s", "path",
              "n_touch_like", "inner_alive", "sp_type", "sp_dist"]].round(2).to_string())
    return tab


# -- part 3: public window -------------------------------------------------------------------
def load_public():
    from gt import public_gt_codes
    gtc = public_gt_codes()
    return gtc, pm.valid_runs(gtc)


def to_codes(lab):
    m = {"H": 0, "A": 1, "D": 2, "X": 3}
    return np.array([m[c] for c in lab], dtype=np.int8)


def part3(under, dseg, gtc, runs):
    print("\n=== PART 3: public window {}..{} ===".format(PUB[0], PUB[1] - 1))
    lab = {0: "H", 1: "A", 2: "D", 3: "X"}
    segs = [(a, b, lab[k]) for a, b, k in pm.segments(gtc, runs)]
    print("GT D segments:", [(a, b, b - a) for a, b, k in segs if k == "D"])
    print("engine D in public:", int(isD[PUB[0]:PUB[1]].sum()), "of", PUB[1] - PUB[0],
          " GT D:", int((gtc[PUB[0]:PUB[1]] == 2).sum()))
    pub = dseg[(dseg["start"] < PUB[1]) & (dseg["end"] > PUB[0])].copy()
    for i, r in pub.iterrows():
        g = gtc[r["start"]:r["end"]]
        pub.loc[i, "gt_D_frac"] = float((g == 2).mean())
    print(pub[["start", "end", "length", "entry", "exit", "frac_fast", "longest_fast_s",
               "path", "alive_flag", "gt_D_frac"]].round(2).to_string())
    inpub = under[(under["start"] < PUB[1]) & (under["end"] > PUB[0])].copy()
    for i, r in inpub.iterrows():
        g = gtc[r["start"]:r["end"]]
        inpub.loc[i, "gt_labels"] = "".join(lab[k] for k in sorted(set(g.tolist())))
        inpub.loc[i, "gt_D_frac"] = float((g == 2).mean())
    print("under-call candidates in the public window:")
    print(inpub[["signal", "start", "end", "secs", "label", "cls", "gt_labels", "gt_D_frac"]]
          .to_string() if len(inpub) else "  none")
    # the signals on public GT D vs alive frames (frame level), for calibration
    w = slice(PUB[0], PUB[1])
    gd = gtc[w] == 2
    for name, sig in SIGNALS.items():
        s = sig[w]
        print("  {}: on GT D {:.3f}, on GT alive {:.3f}".format(name, s[gd].mean(), s[~gd].mean()))
    # distribution test: signal rates on full-match engine alive / D frames vs public GT
    extra = dict(SIGNALS)
    extra["slow_all<1.6"] = lt(feats["meanv_all"], 1.6)
    extra["fast_all>2.0"] = gt_(feats["meanv_all"], 2.0)
    extra["ball_visible"] = vis.copy()
    print("  signal rate: public GT D | public GT alive | full engine D | full engine alive"
          " | full engine D minus goals/pre-kickoff")
    goalish = np.zeros(N, dtype=bool)
    for r in tagging[tagging["code"] == "GOALS"].itertuples(index=False):
        goalish[int(r.start_frame):int(r.end_frame) + 250] = True
    for h, f, l in blocks:
        if kickoffs[h] > f:
            goalish[f:kickoffs[h]] = True
    for name, sig in extra.items():
        s = sig[w]
        print("    {:>14s}: {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f}".format(
            name, s[gd].mean(), s[~gd].mean(), sig[graded & isD].mean(), sig[alive].mean(),
            sig[graded & isD & ~goalish].mean()))
    print("  D share excluding GOALS windows (+10 s) and pre-kickoff frames: {:.3f}".format(
        isD[graded & ~goalish].mean()))
    # run-level: >= 3 s runs of each signal in the public window, GT label composition
    for name, sig in SIGNALS.items():
        s_, e_ = _runs(sig[w])
        out = []
        for a, b in zip(s_ + PUB[0], e_ + PUB[0]):
            if b - a >= MIN_LEN:
                out.append((int(a), int(b), round(float((gtc[a:b] == 2).mean()), 2),
                            round(float((labels[a:b] == "D").mean()), 2)))
        print("  {} runs >= 3 s (start, end, GT D frac, engine D frac): {}".format(name, out))
    return pub, inpub


# -- part 4: expected stoppage count ---------------------------------------------------------
def part4():
    print("\n=== PART 4: expected number of stoppages ===")
    sp = tagging[tagging["code"] == "SET PIECES"]
    print("SET PIECES by type:", sp["type"].value_counts().to_dict(), " GOALS:",
          int((tagging["code"] == "GOALS").sum()))
    # set pieces with / without an engine D segment within +-10 s
    hit = []
    for r in sp.itertuples(index=False):
        a, b = int(r.start_frame) - 250, int(r.end_frame) + 250
        hit.append(bool(np.any((D_START < b) & (D_STOP > a))))
    print("set pieces with a D segment overlapping window +-10 s: {}/{}".format(sum(hit), len(hit)))
    miss = sp[~np.array(hit)]
    print(miss[["type", "team", "start_frame", "end_frame"]].to_string() if len(miss) else "")
    # D segments with no tagged set piece / goal window within +-10 s
    nots = 0
    for a, b in DRUNS:
        ok = ((setp["start_frame"] - 250 < b) & (setp["end_frame"] + 250 > a)).any()
        nots += int(not ok)
    print("engine D segments with no SET PIECES/GOALS window within 10 s: {}/{}".format(nots, len(DRUNS)))
    # tracking structure: dead-evidence episodes
    slow = lt(feats["meanv_all"], cfg.V_STOP)
    ev_mask = ((~vis & slow) | outside | deadball.setpiece_mask(ball, feats, cfg)
               | (slow & gt_(feats["sep"], cfg.SEP_DEAD))) & graded
    s_, e_ = _runs(ev_mask)
    keep = (e_ - s_) >= 2 * FPS
    s_, e_ = s_[keep], e_[keep]
    print("dead-evidence runs >= 2 s on graded frames: {}".format(len(s_)))
    for merge_s in (5, 10, 15, 20):
        clusters = []
        for a, b in zip(s_, e_):
            if clusters and a - clusters[-1][1] <= merge_s * FPS and half_of[a] == half_of[clusters[-1][0]]:
                clusters[-1][1] = b
            else:
                clusters.append([a, b])
        cl = np.array(clusters)
        dur = (cl[:, 1] - cl[:, 0]) / FPS
        in_d = np.array([isD[a:b].mean() for a, b in cl])
        print("  merge gap {:>2d} s: {} episodes (>= 5 s: {}), {} with < 10 % engine D, "
              "{} with >= 50 % engine D".format(merge_s, len(cl), int((dur >= 5).sum()),
                                                int((in_d < 0.1).sum()), int((in_d >= 0.5).sum())))
    # long ball gaps
    gs, ge = _runs(~vis & graded)
    m = (ge - gs) > 2 * FPS
    gs, ge = gs[m], ge[m]
    slowfrac = np.array([np.nanmean(lt(feats["meanv_all"][a:b], cfg.V_STOP)) for a, b in zip(gs, ge)])
    dfrac = np.array([isD[a:b].mean() for a, b in zip(gs, ge)])
    print("ball gaps > 2 s: {}  slow (>50 % frames meanv_all<1.6): {}".format(len(gs), int((slowfrac > 0.5).sum())))
    print("  slow gaps: fully D {}, partly D {}, no D {}".format(
        int(((slowfrac > 0.5) & (dfrac == 1)).sum()), int(((slowfrac > 0.5) & (dfrac > 0) & (dfrac < 1)).sum()),
        int(((slowfrac > 0.5) & (dfrac == 0)).sum())))
    print("  fast gaps: fully D {}, partly D {}, no D {}".format(
        int(((slowfrac <= 0.5) & (dfrac == 1)).sum()), int(((slowfrac <= 0.5) & (dfrac > 0) & (dfrac < 1)).sum()),
        int(((slowfrac <= 0.5) & (dfrac == 0)).sum())))
    for merge_s in (10, 20):
        cl = []
        for a, b, sf in zip(gs, ge, slowfrac):
            if sf <= 0.5:
                continue
            if cl and a - cl[-1][1] <= merge_s * FPS:
                cl[-1][1] = b
            else:
                cl.append([a, b])
        print("  slow gaps clustered within {} s: {} episodes".format(merge_s, len(cl)))
    nd = [(int(a), int(b)) for a, b, sf, df in zip(gs, ge, slowfrac, dfrac) if sf > 0.5 and df == 0]
    print("  slow gaps with no D:", nd)


# -- part 5: spellF1 cost on the public window -----------------------------------------------
def spellf1(codes, gtc, runs):
    return pm.spell_scores(gtc, pm.fill_abstentions(codes, runs), runs)["mean_f1"]


def part5(gtc, runs):
    print("\n=== PART 5: spellF1 cost of D edits on the public window ===")
    base = to_codes(labels)
    b0 = spellf1(base, gtc, runs)
    print("baseline public spellF1 {:.4f}".format(b0))
    lab = {0: "H", 1: "A", 2: "D"}
    segs = pm.segments(base, runs)
    res = []
    # (i) delete each engine D segment: fill with the neighbours (same team: merge; else split
    # at the midpoint, the earlier half to the previous team)
    for i, (a, b, k) in enumerate(segs):
        if k != 2:
            continue
        prev_k = segs[i - 1][2] if i > 0 else None
        next_k = segs[i + 1][2] if i + 1 < len(segs) else None
        c = base.copy()
        if prev_k is None and next_k is None:
            continue
        if prev_k is None:
            c[a:b] = next_k
        elif next_k is None or prev_k == next_k:
            c[a:b] = prev_k
        else:
            mid = (a + b) // 2
            c[a:mid], c[mid:b] = prev_k, next_k
        f = spellf1(c, gtc, runs)
        res.append(("delete_D", a, b, "{}D{}".format(lab.get(prev_k, "-"), lab.get(next_k, "-")),
                    f, f - b0))
    # (ii) insert a 3 s D in the middle of each predicted spell
    gsp = pm.possession_spells(gtc, runs)
    psp = pm.possession_spells(base, runs)
    ov = pm.spell_overlaps(gsp, psp)
    best = {}
    for iou, gi, pj in ov:
        best[pj] = max(best.get(pj, 0.0), iou)
    for j, (a, b, k) in enumerate(psp):
        mid = (a + b) // 2
        lo, hi = mid - MIN_LEN // 2, mid - MIN_LEN // 2 + MIN_LEN
        if hi - lo >= b - a:
            continue
        c = base.copy()
        c[lo:hi] = 2
        f = spellf1(c, gtc, runs)
        res.append(("insert_3s_D", a, b, "{} best IoU {:.2f}".format(lab[k], best.get(j, 0.0)),
                    f, f - b0))
    # (iii) the converse for GT: a GT spell that the engine cuts with D (none here) and a real
    # stoppage the engine misses: replace each GT D segment's engine labels... (GT-side edits are
    # not possible on the submission; use (i) as the missed-stoppage proxy)
    tab = pd.DataFrame(res, columns=["op", "start", "end", "context", "spellF1", "delta"])
    print(tab.round(4).to_string())
    for op in ("delete_D", "insert_3s_D"):
        sub = tab[tab["op"] == op]
        print("{}: n {}, mean delta {:.4f}, median {:.4f}".format(op, len(sub), sub["delta"].mean(),
                                                                  sub["delta"].median()))
    gs = pm.possession_spells(gtc, runs)
    print("public GT spells {}, predicted spells {}".format(len(gs), len(psp)))
    tab.to_csv(os.path.join(OUT, "public_d_edits.csv"), index=False)
    return tab


# -- part 6: out onsets fired by a ball that reappears outside the pitch -----------------------
def reappear_outside_onsets():
    """Out onsets whose first frame is the first frame of a visible run (the ball reappears
    already beyond the line) and whose ball moves toward the pitch over the next 5 frames:
    a throw-in or corner being taken, seen from the tracker's first sighting."""
    def dout(i):
        return max(-bx[i], bx[i] - L, -by[i], by[i] - W, 0.0)
    rows_ = []
    for f, m in zip(onsets["frame"], onsets["mechanism"]):
        f = int(f)
        if m != "out" or f == 0 or vis[f - 1] or not vis[f]:
            continue
        inward = bool(f + 5 < N and vis[f + 5] and dout(f + 5) < dout(f))
        restart_here = bool(((CONF_CANDS["frame"] - f).abs() <= 3).any())
        side = "touchline" if (by[f] < 0 or by[f] > W) else "goal_line"
        rows_.append((f, inward, restart_here, labels[f - 1], bool(graded[f]), side))
    return pd.DataFrame(rows_, columns=["frame", "inward", "confirmed_restart_within_3",
                                        "label_before", "graded", "side"])


def spell_count(lab):
    n = 0
    for h, f, l in blocks:
        seg = lab[f:l + 1]
        s, e = _runs(seg != "D")
        for a, b in zip(s, e):
            sub = seg[a:b]
            n += 1 + int((sub[1:] != sub[:-1]).sum())
    return n


def part6(gtc, runs):
    print("\n=== PART 6: out onsets on a ball that reappears outside the pitch ===")
    tab = reappear_outside_onsets()
    tab = tab[tab["graded"]]
    print("out onsets on graded frames:", int(((onsets["mechanism"] == "out")
                                                & graded[onsets["frame"].to_numpy()]).sum()))
    print("of which on a reappearance outside the pitch: {}  inward {}  with a confirmed restart "
          "within 3 frames {}  inward & restart {}".format(
              len(tab), int(tab["inward"].sum()), int(tab["confirmed_restart_within_3"].sum()),
              int((tab["inward"] & tab["confirmed_restart_within_3"]).sum())))
    print("label on the frame before:", tab["label_before"].value_counts().to_dict())
    print("by side:", tab.groupby("side")["inward"].agg(["size", "sum"]).to_dict())
    lab2 = None
    for variant in ("inward, any line", "inward, touchline only"):
        sel = tab["inward"] & ((tab["side"] == "touchline") | (variant == "inward, any line"))
        print("\n-- variant: drop out onsets on reappearances outside ({}): {} onsets".format(
            variant, int(sel.sum())))
        lab2 = _variant(set(tab.loc[sel, "frame"].astype(int)), gtc, runs)
    tab.to_csv(os.path.join(OUT, "reappear_outside_outs.csv"), index=False)
    return tab, lab2


def _variant(drop, gtc, runs):
    mod = dict(inputs)
    on2 = onsets[~((onsets["mechanism"] == "out") & onsets["frame"].isin(drop))].reset_index(drop=True)
    mod["onsets"] = on2
    lab2 = possession.run(cfg, inputs=mod)
    d2 = lab2 == "D"
    print("  D share {:.4f} -> {:.4f}".format(
        isD[graded].mean(), d2[graded].mean()))
    for h, f, l in blocks:
        print("  half {}: {:.4f} -> {:.4f}".format(h, isD[f:l + 1].mean(), d2[f:l + 1].mean()))
    n1 = sum(len(_runs(isD[f:l + 1])[0]) for h, f, l in blocks)
    n2 = sum(len(_runs(d2[f:l + 1])[0]) for h, f, l in blocks)
    print("  D segments {} -> {};  spells (possessions) {} -> {}".format(
        n1, n2, spell_count(labels), spell_count(lab2)))
    changed = np.flatnonzero((labels != lab2) & graded)
    print("  frames changed {} (D->alive {}, alive->D {})".format(
        changed.size, int((isD & ~d2 & graded).sum()), int((~isD & d2 & graded).sum())))
    s_, e_ = _runs((labels != lab2) & graded)
    print("  changed stretches:", [(int(a), int(b), labels[a], lab2[a]) for a, b in zip(s_, e_)])
    print("  public spellF1 {:.4f} -> {:.4f}".format(spellf1(to_codes(labels), gtc, runs),
                                                     spellf1(to_codes(lab2), gtc, runs)))
    return lab2


# -- part 7: quick restarts: stationary ball on alive frames -----------------------------------
def part7(gtc):
    print("\n=== PART 7: stationary ball (>= 1.5 s, < 0.5 m) on frames labelled H/A ===")
    st = deadball.stationary_mask(ball, cfg) & alive
    s_, e_ = _runs(st)
    rows_ = []
    for a, b in zip(s_, e_):
        a, b = int(a), int(b)
        d = feats["dmin"][a:b]
        mv = feats["meanv_all"][a:b]
        dist, typ, team, _ = nearest_setpiece(a, b)
        rows_.append(dict(start=a, end=b, secs=(b - a) / FPS, label="".join(sorted(set(labels[a:b]))),
                          dmin_med=float(np.nanmedian(d)), d2_med=float(np.nanmedian(feats["d_second"][a:b])),
                          meanv_all_med=float(np.nanmedian(mv)),
                          meanv_near_med=float(np.nanmedian(feats["meanv_near"][a:b])),
                          keeper_frac=float(keeper_near[a:b].mean()), ghost_frac=float(ghost[a:b].mean()),
                          in_pa=float(in_pa[a:b].mean()), x=float(bx[a]), y=float(by[a]),
                          sp_dist=dist, sp_type=typ,
                          gt=("".join(sorted(set("HADX"[k] for k in gtc[a:b]))) if PUB[0] <= a < PUB[1] else "")))
    tab = pd.DataFrame(rows_)
    if not len(tab):
        print("none")
        return tab
    def cls(r):
        if r["ghost_frac"] > 0.5:
            return "ghost"
        if r["keeper_frac"] > 0.5:
            return "keeper_holding"
        if r["dmin_med"] <= 1.0 and r["d2_med"] > 3.0:
            return "one_player_on_ball"
        if r["dmin_med"] <= 1.5:
            return "player_on_ball_contested"
        return "nobody_near"
    tab["cls"] = tab.apply(cls, axis=1)
    print(tab.round(2).to_string())
    print(tab.groupby("cls").agg(n=("secs", "size"), secs=("secs", "sum")).to_string())
    tab.to_csv(os.path.join(OUT, "stationary_alive.csv"), index=False)
    return tab


# -- part 8: onsets confirmed by a collective stop that began before the onset -----------------
def part8(dseg):
    print("\n=== PART 8: hard onsets confirmed only by a stop run that started before them ===")
    fps = FPS
    horizon = int(round(cfg.ONSET_CONFIRM_S * fps))
    ss, se = deadball._stop_runs(feats, cfg, fps)
    rows_ = []
    for f, m in zip(HARD_ONSETS["frame"], HARD_ONSETS["mechanism"]):
        f = int(f)
        if not graded[f]:
            continue
        hit = (ss <= f + horizon) & (se > f)
        if not hit.any():
            continue  # observed outs need no confirmation
        only_before = bool(np.all(ss[hit] < f))
        is_entry = bool(f in set(dseg["start"]))
        prev = D_STOP[D_STOP <= f]
        since_restart = int(f - prev.max()) if prev.size else 10 ** 6
        seg = dseg[dseg["start"] == f]
        rows_.append(dict(frame=f, mechanism=m, only_prior_stop=only_before, entry=is_entry,
                          frames_since_prev_D=since_restart,
                          seg_len=int(seg["length"].iloc[0]) if len(seg) else 0,
                          seg_frac_fast=float(seg["frac_fast"].iloc[0]) if len(seg) else np.nan,
                          seg_touch=int(seg["n_touch_like"].iloc[0]) if len(seg) else 0))
    tab = pd.DataFrame(rows_)
    print("hard onsets with a confirming stop run: {}; confirmed only by a run that began earlier: "
          "{} (entries: {})".format(len(tab), int(tab["only_prior_stop"].sum()),
                                    int((tab["only_prior_stop"] & tab["entry"]).sum())))
    print(tab[tab["only_prior_stop"]].round(2).to_string())
    tab.to_csv(os.path.join(OUT, "onset_prior_stop.csv"), index=False)
    return tab


# -- part 9: ball exits over a line (throw-ins, corners, goal kicks) and D coverage -------------
def part9():
    print("\n=== PART 9: ball exits over a pitch line and engine D coverage ===")
    beyond = deadball.outside_pitch(ball, (L, W), 0.0)
    rows_ = []
    # (a) observed crossing: visible inside on frame t-1, visible beyond the line on frame t
    cross = np.flatnonzero(beyond[1:] & vis[:-1] & ~beyond[:-1]) + 1
    for f in cross:
        side = "touch" if (by[f] < 0 or by[f] > W) else "goal"
        rows_.append((int(f), "observed", side))
    # (b) the ball vanishes within EDGE_M of a line, moving toward it, for > 1 s
    rs, re_ = _runs(vis)
    for idx in range(len(re_) - 1):
        last = int(re_[idx]) - 1
        nxt = int(rs[idx + 1])
        if nxt - last < FPS or beyond[last]:
            continue
        x, y, vx, vy = bx[last], by[last], ball["vx"][last], ball["vy"][last]
        if (y < EDGE_M and vy < 0) or (y > W - EDGE_M and vy > 0):
            rows_.append((last + 1, "vanish", "touch"))
        elif (x < EDGE_M and vx < 0) or (x > L - EDGE_M and vx > 0):
            rows_.append((last + 1, "vanish", "goal"))
    tab = pd.DataFrame(sorted(rows_), columns=["frame", "how", "side"])
    tab = tab[graded[tab["frame"].to_numpy()]].reset_index(drop=True)
    # merge exits within 10 s (one stoppage), keep the first
    keep, last_f = [], -10 ** 9
    for f in tab["frame"]:
        keep.append(f - last_f > 10 * FPS)
        last_f = f if f - last_f > 10 * FPS else last_f
    tab = tab[np.array(keep)].reset_index(drop=True)
    tab["label_at"] = [labels[f] for f in tab["frame"]]
    tab["D_within_5s"] = [bool(isD[f:min(N, f + 5 * FPS)].any()) for f in tab["frame"]]
    tab["D_within_10s"] = [bool(isD[max(0, f - 2 * FPS):min(N, f + 10 * FPS)].any()) for f in tab["frame"]]
    tab["alive_after_3s"] = [float(np.nanmedian(feats["meanv_all"][f + FPS:f + 4 * FPS])) for f in tab["frame"]]
    print("exits (merged within 10 s): {}  by side {}  by how {}".format(
        len(tab), tab["side"].value_counts().to_dict(), tab["how"].value_counts().to_dict()))
    print("already inside D at the exit: {}".format(int((tab["label_at"] == "D").sum())))
    al = tab[tab["label_at"] != "D"]
    print("from alive: {}; D within 5 s: {}; D within 10 s: {}".format(
        len(al), int(al["D_within_5s"].sum()), int(al["D_within_10s"].sum())))
    miss = al[~al["D_within_10s"]]
    print("exits from alive with no D within 10 s:")
    print(miss.round(2).to_string() if len(miss) else "  none")
    tab.to_csv(os.path.join(OUT, "line_exits.csv"), index=False)
    return tab


def part10(dseg):
    """Each D segment as out-type (a ball exit or out onset from 10 s before it to its end) or
    not (foul / offside / injury / goal type): compares with the literature mix."""
    print("\n=== PART 10: D segments by cause (out of play vs other) ===")
    beyond = deadball.outside_pitch(ball, (L, W), 0.0)
    cross = np.flatnonzero(beyond[1:] & vis[:-1] & ~beyond[:-1]) + 1
    outs = onsets.loc[onsets["mechanism"] == "out", "frame"].to_numpy()
    lines = pd.read_csv(os.path.join(OUT, "line_exits.csv"))["frame"].to_numpy()
    allx = np.unique(np.concatenate([cross, outs, lines]))
    goals = tagging[tagging["code"] == "GOALS"]
    kinds = []
    for r in dseg.itertuples(index=False):
        a, b = int(r.start), int(r.end)
        if r.entry.startswith("block_start"):
            kinds.append("pre_kickoff")
            continue
        if ((goals["start_frame"] < b) & (goals["end_frame"] > a)).any() and r.secs > 60:
            kinds.append("goal")
            continue
        hit = allx[(allx >= a - 10 * FPS) & (allx < b)]
        kinds.append("out_of_play" if hit.size else "other(foul/offside/injury)")
    dseg = dseg.assign(cause=kinds)
    print(dseg.groupby("cause").agg(n=("length", "size"), frames=("length", "sum"),
                                    med_s=("secs", "median")).to_string())
    print("per half:", dseg.groupby(["half", "cause"]).size().to_dict())
    oth = dseg[dseg["cause"].str.startswith("other")]
    print(oth[["start", "end", "secs", "entry", "exit", "prev_label", "next_label", "sp_type",
               "sp_dist"]].round(1).to_string())
    dseg.to_csv(os.path.join(OUT, "d_segments.csv"), index=False)
    return dseg


if __name__ == "__main__":
    print("labels D share graded {:.4f}, D runs {}".format(isD[graded].mean(), len(DRUNS)))
    under = part1()
    dseg = part2()
    gtc, runs = load_public()
    part3(under, dseg, gtc, runs)
    part4()
    part5(gtc, runs)
    part6(gtc, runs)
    part7(gtc)
    part8(dseg)
    part9()
    part10(dseg)
