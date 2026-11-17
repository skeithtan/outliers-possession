"""Q3 detector view: what the contact detector saw at each reference live hand-over.

    python q3_detector.py     -> out/q3_detector.json, out/handover_detector_<view>.csv

For every reference direct hand-over r (old team W -> new team R) the script rebuilds the bump
contacts of engine.events (impulse |v(t+h) - v(t-h)|, h = CONTACT_HALF_WINDOW 12, window maxima
over +-h, on frames with e_D <= EVENT_SUPPRESS_DEAD) at several thresholds, assigns each maximum
to the nearer team over [c-h, c+h] as _bump_contacts does, and classifies the hand-over by the
best evidence for the new team within +-12 frames of r (ladder below). It does the same over the
whole reference possession of R for the T_missed_absorbed episodes of q4.py.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
WORK = C.WORK
from engine import events as E  # noqa: E402
from engine.config import Config  # noqa: E402

CFG = Config.default()
H_WIN = int(CFG.CONTACT_HALF_WINDOW)
THR = float(CFG.DV_CONTACT)
LOW = (6.0, 4.0)
R_TOUCH = float(CFG.R_TOUCH)
R_REACH = float(CFG.R_REACH_M)
MARGIN = float(CFG.CONTESTED_MARGIN_M)
WIN = 12

LADDER = ("A_row_new_owner", "B_row_new_nonowner", "C_bump_new_no_row", "D_bump_new_reach_no_row",
          "E_bump_contested", "F_bump_old_team", "G_subthr6_new", "H_subthr4_new",
          "I_subthr_other", "J_no_bump_new_near", "K_no_bump_new_far")


def maxima_table(M):
    """DataFrame of window maxima of |J| >= min(LOW) on event-alive frames, with team geometry."""
    ball = {"vx": M["vx"].astype(np.float64), "vy": M["vy"].astype(np.float64),
            "run_id": M["run_id"]}
    size = E.impulse(ball, H_WIN)
    alive = ~(np.isfinite(M["e_dead"]) & (M["e_dead"] > CFG.EVENT_SUPPRESS_DEAD))
    cand = np.flatnonzero(np.isfinite(size) & (size >= min(LOW)) & alive)
    peaks = E.window_maxima(size, H_WIN, cand)
    dH = np.where(np.isfinite(M["dH"]), M["dH"], np.inf).astype(np.float64)
    dA = np.where(np.isfinite(M["dA"]), M["dA"], np.inf).astype(np.float64)
    speed = M["speed"].astype(np.float64)
    n = len(size)
    rows = []
    for c in peaks:
        lo, hi = max(0, c - H_WIN), min(n, c + H_WIN + 1)
        dh, da = dH[lo:hi].min(), dA[lo:hi].min()
        team = "H" if dh <= da else "A"
        d1, d2 = min(dh, da), max(dh, da)
        contested = d2 <= R_TOUCH and (d2 - d1) < MARGIN
        ch = speed[min(n - 1, c + H_WIN)] - speed[max(0, c - H_WIN)]
        split = CFG.REDIRECT_FRACTION * size[c]
        kind = "kick" if ch >= split else ("receive" if ch <= -split else "redirect")
        rows.append((int(c), float(size[c]), team, float(d1), float(d2), bool(contested), kind))
    return pd.DataFrame(rows, columns=["frame", "J", "team", "d1", "d2", "contested", "kind"]), size


def classify(r, R, W, mx, ev, owner_set, dR, lo_off=-WIN, hi_off=WIN):
    """Ladder label of the best new-team evidence in [r+lo_off, r+hi_off]."""
    a, b = r + lo_off, r + hi_off
    e = ev[(ev.frame >= a) & (ev.frame <= b)]
    eR = e[e.team == R]
    if len(eR):
        if any((int(f), R) in owner_set for f in eR.frame):
            return "A_row_new_owner"
        return "B_row_new_nonowner"
    m = mx[(mx.frame >= a) & (mx.frame <= b)]
    strong = m[m.J >= THR]
    if len(strong):
        sR = strong[(strong.team == R) & ~strong.contested]
        if len(sR):
            if (sR.d1 <= R_TOUCH).any():
                return "C_bump_new_no_row"
            if (sR.d1 <= R_REACH).any():
                return "D_bump_new_reach_no_row"
        if strong.contested.any():
            return "E_bump_contested"
        return "F_bump_old_team"
    s6 = m[(m.J >= LOW[0]) & (m.team == R) & ~m.contested & (m.d1 <= R_TOUCH)]
    if len(s6):
        return "G_subthr6_new"
    s4 = m[(m.J >= LOW[1]) & (m.team == R) & ~m.contested & (m.d1 <= R_TOUCH)]
    if len(s4):
        return "H_subthr4_new"
    if len(m):
        return "I_subthr_other"
    near = np.nanmin(dR[max(0, a):b + 1]) if b >= a else np.inf
    return "J_no_bump_new_near" if near <= R_TOUCH else "K_no_bump_new_far"


def analyse(view, game, episodes_df):
    M = C.load(view, game)
    mx, size = maxima_table(M)
    ev = M["events"].copy()
    ev["team"] = ev["team"].where(ev["team"].notna(), None)
    owner_set = set((int(f), str(t)) for f, t in M["owner_events"])
    gt, ours, runs = M["gt"], M["ours"], M["runs"]
    changes = [c for c in C.owner_changes(gt, runs) if c["kind"] == "direct"]
    our_ch = [c for c in C.owner_changes(ours, runs) if c["kind"] != "same_D"]
    ref_all = [c for c in C.owner_changes(gt, runs) if c["kind"] != "same_D"]
    pairs, _, _ = C.match_changes(ref_all, our_ch, 25)
    matched = set(ref_all[i]["frame"] for i, _ in pairs)
    dist = {"H": M["dH"].astype(np.float64), "A": M["dA"].astype(np.float64)}
    rows = []
    for c in changes:
        r = c["frame"]
        R, W = "HA"[c["dst"]], "HA"[c["src"]]
        lab = classify(r, R, W, mx, ev, owner_set, dist[R])
        jwin = size[max(0, r - WIN):r + WIN + 1]
        rows.append({"view": view, "game": game, "group": M["group"], "frame": r, "src": W,
                     "dst": R, "det25": r in matched, "new_dur_s": (c["next_stop"] - r) / C.FPS,
                     "ladder": lab,
                     "jmax": float(np.nanmax(jwin)) if np.isfinite(jwin).any() else None,
                     "dR_min": float(np.nanmin(dist[R][max(0, r - WIN):r + WIN + 1])),
                     "dW_min": float(np.nanmin(dist[W][max(0, r - WIN):r + WIN + 1])),
                     "vis": bool(M["visible"][r])})
    # absorbed episodes: evidence over the whole reference possession of the new team
    ab = []
    sel = episodes_df[(episodes_df.game == game) & (episodes_df.cls == "T_missed_absorbed")]
    for _, e in sel.iterrows():
        a, b = int(e.start), int(e.stop)
        R, W = e.ref, e.our
        lab = classify(a, R, W, mx, ev, owner_set, dist[R], lo_off=-WIN, hi_off=(b - a) + WIN)
        ab.append({"view": view, "game": game, "group": M["group"], "start": a, "len": b - a,
                   "ladder_whole": lab,
                   "ladder_at_start": classify(a, R, W, mx, ev, owner_set, dist[R])})
    return rows, ab


def summarise(df, key):
    out = {}
    for name, sub in df.groupby("group"):
        out[name] = sub[key].value_counts(normalize=False).to_dict()
    out["pooled"] = df[key].value_counts().to_dict()
    return out


def main():
    report = {}
    for view in ("matched", "raw"):
        eps = pd.read_csv(os.path.join(WORK, "out", "episodes_%s.csv" % view))
        rows, ab = [], []
        for game in C.OPEN_GAMES:
            r, a = analyse(view, game, eps)
            rows += r
            ab += a
        df = pd.DataFrame(rows)
        adf = pd.DataFrame(ab)
        df.to_csv(os.path.join(WORK, "out", "handover_detector_%s.csv" % view), index=False)
        adf.to_csv(os.path.join(WORK, "out", "absorbed_detector_%s.csv" % view), index=False)
        rep = {}
        for grp in ("Metrica", "DFL", "pooled"):
            d = df if grp == "pooled" else df[df.group == grp]
            a = adf if grp == "pooled" else adf[adf.group == grp]
            rep[grp] = {
                "n": int(len(d)),
                "ladder_missed": d[~d.det25].ladder.value_counts().reindex(LADDER).fillna(0).astype(int).to_dict(),
                "ladder_detected": d[d.det25].ladder.value_counts().reindex(LADDER).fillna(0).astype(int).to_dict(),
                "det25_by_ladder": {k: (float(d[d.ladder == k].det25.mean()) if (d.ladder == k).any() else None)
                                    for k in LADDER},
                "jmax_missed_q": C.quantiles(d[~d.det25].jmax.dropna()),
                "jmax_detected_q": C.quantiles(d[d.det25].jmax.dropna()),
                "dR_min_missed_q": C.quantiles(d[~d.det25].dR_min.replace(np.inf, np.nan).dropna()),
                "dR_min_detected_q": C.quantiles(d[d.det25].dR_min.replace(np.inf, np.nan).dropna()),
                "absorbed_n": int(len(a)),
                "absorbed_len_s_q": C.quantiles(a["len"] / C.FPS) if len(a) else None,
                "absorbed_len_bins": {k: int(v) for k, v in pd.cut(a["len"] / C.FPS, [0, 1, 2, 5, 1e9], right=False,
                                                                   labels=["<1s", "1-2s", "2-5s", ">=5s"]).value_counts().sort_index().items()} if len(a) else None,
                "absorbed_ladder_whole": a.ladder_whole.value_counts().reindex(LADDER).fillna(0).astype(int).to_dict() if len(a) else None,
                "absorbed_ladder_at_start": a.ladder_at_start.value_counts().reindex(LADDER).fillna(0).astype(int).to_dict() if len(a) else None,
            }
        report[view] = rep
    with open(os.path.join(WORK, "out", "q3_detector.json"), "w") as handle:
        json.dump(report, handle, indent=1, sort_keys=True,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(report["matched"], indent=1)[:6000])


if __name__ == "__main__":
    main()
