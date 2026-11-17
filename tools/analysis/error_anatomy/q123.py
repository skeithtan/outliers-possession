"""Q1, Q2, Q3 and Q6 of the error anatomy: owner changes of the reference against ours.

    python q123.py            -> out/q123.json, out/changes_<view>.csv

Owner change = change of team between consecutive alive runs inside one graded run of the
reference (direct H<->A, or across a D stretch). Matching: common.match_changes (one-to-one per
destination team, max cardinality then least |latency|). Only the eight OPEN references.
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

TOLS = (5, 12, 25, 50)
SPUR_TOL = 25
SWITCH_WIN = 3           # SWITCH_EVENT_WINDOW of engine/config.py
ALIVE_WIN = 5            # ALIVE_EVENT_WINDOW
R_TOUCH = 1.5
TEAM_KINDS = ("touch", "release", "control", "dribble")
FLICKER_BINS = (0.5, 1.0, 2.0, 5.0, 10.0)
DUR_BINS = (1.0, 2.0, 5.0)
OWNER_MIN_CONF = 0.5


def ev_label(kind, notes, dist):
    notes = str(notes)
    if kind == "release":
        lab = "kick"
    elif kind == "control":
        if "keeper_hold" in notes:
            lab = "keeper_hold"
        elif "receive" in notes:
            lab = "receive"
        elif "redirect" in notes:
            lab = "redirect_ctrl"
        else:
            lab = "control_run"
    elif kind == "touch":
        lab = "redirect_touch" if "redirect" in notes else "touch"
    elif kind == "dribble":
        lab = "dribble_contested" if "contested" in notes else "dribble"
    else:
        lab = str(kind)
    if np.isfinite(dist) and dist > R_TOUCH:
        lab += "+reach"
    return lab


class Context(object):
    """Per match-view lookups of events, owner events and restarts."""

    def __init__(self, M):
        self.M = M
        ev = M["events"].copy()
        ev["team"] = ev["team"].where(ev["team"].notna(), None)
        self.ev = ev
        self.ev_frame = ev["frame"].to_numpy(dtype=np.int64)
        self.ev_team = ev["team"].to_numpy(dtype=object)
        self.ev_kind = ev["kind"].to_numpy(dtype=object)
        self.ev_conf = ev["confidence"].to_numpy(dtype=np.float64)
        self.ev_lab = np.array([ev_label(k, n, d) for k, n, d in
                                zip(ev["kind"], ev["notes"], ev["dist_m"])], dtype=object)
        owners = set((int(f), str(t)) for f, t in M["owner_events"])
        self.ev_owner = np.array([(int(f), t) in owners for f, t in
                                  zip(self.ev_frame, self.ev_team)], dtype=bool)
        self.ev_teamrow = np.isin(self.ev_kind, TEAM_KINDS) & np.isin(self.ev_team, ["H", "A"])
        rs = M["restarts"]
        conf = rs[rs["confirmed"]]
        self.rs_frame = conf["frame"].to_numpy(dtype=np.int64)
        self.rs_kind = conf["kind"].to_numpy(dtype=object)
        self.rs_team = conf["team"].where(conf["team"].notna(), None).to_numpy(dtype=object)
        self.re = sorted(M["restart_events"], key=lambda x: (x[0], str(x[1])))
        self.re_frame = np.array([f for f, _ in self.re], dtype=np.int64)
        self.kick_frames = set(int(f) for f in M["kickoffs"].values())
        self.n = len(M["gt"])

    def events_near(self, frame, lo, hi, team=None):
        """Indices of events with frame in [frame+lo, frame+hi] (team filter optional)."""
        a = np.searchsorted(self.ev_frame, frame + lo, side="left")
        b = np.searchsorted(self.ev_frame, frame + hi, side="right")
        idx = np.arange(a, b)
        if team is not None:
            idx = idx[self.ev_team[idx] == team]
        return idx

    def trigger(self, ch, letter):
        """Label of what made our change at ch (dict from owner_changes, dst team letter)."""
        f = ch["frame"]
        out = {}
        if ch["kind"] == "across_D":
            k = np.flatnonzero(np.abs(self.re_frame - f) <= ALIVE_WIN)
            teams = [self.re[i][1] for i in k]
            if f in self.kick_frames or any(int(self.re[i][0]) in self.kick_frames for i in k):
                lab = "restart:kickoff"
            elif letter in teams:
                kinds = [str(self.rs_kind[j]) for j in np.flatnonzero(
                    (np.abs(self.rs_frame - f) <= ALIVE_WIN) & (self.rs_team == letter))]
                lab = "restart:team_" + (kinds[0] if kinds else "goal_kickoff")
            elif len(k):
                lab = "restart:other_or_none_team"
            else:
                lab = "restart:no_candidate"
            out["trigger"] = lab
            idx = self.events_near(f, -SWITCH_WIN, SWITCH_WIN, letter)
            idx = idx[self.ev_teamrow[idx]]
            out["event_at_switch"] = (self.ev_lab[idx[np.argmin(np.abs(self.ev_frame[idx] - f))]]
                                      if len(idx) else None)
            return out
        idx = self.events_near(f, -SWITCH_WIN, SWITCH_WIN, letter)
        idx = idx[self.ev_teamrow[idx]]
        if len(idx):
            j = idx[np.lexsort((-self.ev_owner[idx].astype(int), np.abs(self.ev_frame[idx] - f)))][0]
            out["trigger"] = self.ev_lab[j]
            out["trigger_owner_event"] = bool(self.ev_owner[j])
            out["trigger_conf"] = float(self.ev_conf[j])
            out["trigger_offset"] = int(self.ev_frame[j] - f)
        else:
            idx = self.events_near(f, -25, 25, letter)
            idx = idx[self.ev_teamrow[idx]]
            out["trigger"] = "no_event_pm3" + ("(event_pm25)" if len(idx) else "(none_pm25)")
            out["trigger_owner_event"] = False
        return out


def ball_info(M, f, half):
    n = len(M["visible"])
    a, b = max(0, f - half), min(n, f + half + 1)
    vis = M["visible"][a:b]
    sp = M["speed"][max(0, f - 5):min(n, f + 6)]
    spv = sp[np.isfinite(sp)]
    return {"vis_at": bool(M["visible"][f]), "vis_frac": float(vis.mean()),
            "speed_at": float(M["speed"][f]) if np.isfinite(M["speed"][f]) else None,
            "speed_med5": float(np.median(spv)) if spv.size else None}


def analyse(view, game):
    M = C.load(view, game)
    ctx = Context(M)
    gt, ours, runs, fps = M["gt"], M["ours"], M["runs"], float(M["fps"])
    ref_all = [c for c in C.owner_changes(gt, runs) if c["kind"] != "same_D"]
    our_all = [c for c in C.owner_changes(ours, runs) if c["kind"] != "same_D"]
    our_same = [c for c in C.owner_changes(ours, runs) if c["kind"] == "same_D"]
    ref_same = [c for c in C.owner_changes(gt, runs) if c["kind"] == "same_D"]
    letters = "HA"
    res = {"view": view, "game": game, "group": M["group"], "hours": M["hours"],
           "n_ref_direct": sum(c["kind"] == "direct" for c in ref_all),
           "n_ref_across": sum(c["kind"] == "across_D" for c in ref_all),
           "n_ref_same_D": len(ref_same),
           "n_our_direct": sum(c["kind"] == "direct" for c in our_all),
           "n_our_across": sum(c["kind"] == "across_D" for c in our_all),
           "n_our_same_D": len(our_same)}
    matchings = {tol: C.match_changes(ref_all, our_all, tol) for tol in TOLS}
    rows = []
    # reference changes
    m50 = dict(matchings[50][0])
    det = {tol: set(i for i, _ in matchings[tol][0]) for tol in TOLS}
    pair25 = dict(matchings[25][0])
    for i, c in enumerate(ref_all):
        r = c["frame"]
        dst = letters[c["dst"]]
        # new possession of the reference: from r to the end of that alive run
        dur = (c["next_stop"] - r) / fps
        # what ends it: a direct return (flicker) or D / run end
        end = c["next_stop"]
        ends_direct = end < len(gt) and gt[end] in (C.H, C.A) and any(
            a <= end < b for a, b in runs)
        row = {"view": view, "game": game, "group": M["group"], "side": "ref", "idx": i,
               "frame": r, "kind": c["kind"], "src": letters[c["src"]], "dst": dst,
               "gap": c["gap"], "new_dur_s": dur, "new_ends_direct": bool(ends_direct)}
        for tol in TOLS:
            row["det_%d" % tol] = i in det[tol]
        if i in m50:
            oc = our_all[m50[i]]
            row["offset50"] = oc["frame"] - r
            row["our_kind50"] = oc["kind"]
            row["our_dstart50"] = oc["d_start"] - r if oc["kind"] == "across_D" else None
        if i in pair25:
            oc = our_all[pair25[i]]
            tr = ctx.trigger(oc, dst)
            row["our_trigger25"] = tr.get("trigger")
        row.update({"ball_" + k: v for k, v in ball_info(M, r, 25).items()})
        row["ball_vis_pm12"] = float(M["visible"][max(0, r - 12):r + 13].mean())
        # events of the new team around the change
        for win, name in ((25, "1s"), (75, "3s")):
            idx = ctx.events_near(r, -win, win, dst)
            row["n_ev_new_%s" % name] = int(len(idx))
            row["n_teamrow_new_%s" % name] = int(ctx.ev_teamrow[idx].sum())
            row["n_owner_new_%s" % name] = int(ctx.ev_owner[idx].sum())
            row["labs_new_%s" % name] = ";".join(sorted(set(ctx.ev_lab[idx]))) if len(idx) else ""
            lowc = idx[ctx.ev_teamrow[idx] & ~ctx.ev_owner[idx]]
            row["n_nonowner_teamrow_%s" % name] = int(len(lowc))
            row["labs_nonowner_%s" % name] = ";".join(sorted(set(
                "{}{}".format(ctx.ev_lab[j], "<conf" if ctx.ev_conf[j] < OWNER_MIN_CONF else "")
                for j in lowc))) if len(lowc) else ""
            cont = ctx.events_near(r, -win, win)
            row["n_contested_%s" % name] = int(np.sum(ctx.ev_kind[cont] == "contested"))
        # nearest owner event of the new team (signed offset)
        idx = ctx.events_near(r, -75, 75, dst)
        idx = idx[ctx.ev_owner[idx]]
        row["nearest_owner_new_off"] = (int((ctx.ev_frame[idx] - r)[np.argmin(np.abs(ctx.ev_frame[idx] - r))])
                                        if len(idx) else None)
        # owner track and Viterbi path over the new reference possession (capped at 3 s)
        seg_end = min(c["next_stop"], r + 75)
        lo = max(0, r - 25)
        ot = M["owner"][lo:seg_end]
        row["owner_new_frac"] = float(np.mean(ot == c["dst"])) if len(ot) else None
        row["owner_unknown_frac"] = float(np.mean(ot == -1)) if len(ot) else None
        vit = M["vit"][lo:seg_end]
        row["vit_has_new"] = bool(np.any(vit == c["dst"]))
        row["final_has_new"] = bool(np.any(ours[lo:seg_end] == c["dst"]))
        row["our_dead_frac"] = float(np.mean(ours[r:c["next_stop"]] == C.D))
        rows.append(row)
    # our changes
    m25_our = {j: i for i, j in matchings[25][0]}
    for j, c in enumerate(our_all):
        f = c["frame"]
        dst = letters[c["dst"]]
        row = {"view": view, "game": game, "group": M["group"], "side": "ours", "idx": j,
               "frame": f, "kind": c["kind"], "src": letters[c["src"]], "dst": dst,
               "gap": c["gap"], "matched25": j in m25_our}
        # nearest reference change of any destination
        ref_frames = np.array([rc["frame"] for rc in ref_all], dtype=np.int64)
        row["nearest_ref_any"] = int(np.min(np.abs(ref_frames - f))) if len(ref_frames) else None
        # return time: next our change (B->A) in the same run
        nxt = our_all[j + 1] if j + 1 < len(our_all) else None
        if nxt is not None and nxt["run"] == c["run"]:
            row["return_s"] = (nxt["frame"] - f) / fps
            row["return_kind"] = nxt["kind"]
        else:
            row["return_s"] = None
            row["return_kind"] = None
        row["our_run_s"] = (c["next_stop"] - f) / fps
        row.update(ctx.trigger(c, dst))
        row.update({"ball_" + k: v for k, v in ball_info(M, f, 12).items()})
        # reference label at our change and over our new run
        row["ref_at"] = "HADX"[gt[f]]
        seg = gt[f:c["next_stop"]]
        row["ref_new_frac"] = float(np.mean(seg == c["dst"])) if len(seg) else None
        row["ref_dead_frac"] = float(np.mean(seg == C.D)) if len(seg) else None
        row["owner_at"] = int(M["owner"][f])
        row["owner_prev"] = int(M["owner"][f - 1])
        rows.append(row)
    res["detect"] = {}
    for kind in ("direct", "across_D"):
        idx = [i for i, c in enumerate(ref_all) if c["kind"] == kind]
        res["detect"][kind] = {str(t): C.frac(sum(i in det[t] for i in idx), len(idx)) for t in TOLS}
        res["detect"][kind]["n"] = len(idx)
    res["n_our_spurious25"] = len(matchings[25][2])
    return res, rows


def bins_count(values, edges):
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    names, counts = [], []
    lo = 0.0
    for e in edges:
        names.append("<{:g}s".format(e) if lo == 0 else "{:g}-{:g}s".format(lo, e))
        counts.append(int(np.sum((v >= lo) & (v < e))))
        lo = e
    names.append(">={:g}s".format(lo))
    counts.append(int(np.sum(v >= lo)))
    return dict(zip(names, counts))


def summarise(df, per):
    """Aggregates for one subset (a group or pooled) of the change rows."""
    out = {}
    ref = df[df.side == "ref"]
    ours = df[df.side == "ours"]
    hours = per["hours"].sum()
    for kind in ("direct", "across_D"):
        r = ref[ref.kind == kind]
        d = {"n": int(len(r)), "per_hour": len(r) / hours}
        for t in TOLS:
            d["det_%d" % t] = float(r["det_%d" % t].mean()) if len(r) else None
        off = r["offset50"].dropna().to_numpy()
        d["offset50"] = {"n": int(off.size), "median": float(np.median(off)) if off.size else None,
                         "p25": float(np.percentile(off, 25)) if off.size else None,
                         "p75": float(np.percentile(off, 75)) if off.size else None,
                         "late": C.frac(np.sum(off > 0), off.size),
                         "early": C.frac(np.sum(off < 0), off.size),
                         "zero": C.frac(np.sum(off == 0), off.size),
                         "abs_median": float(np.median(np.abs(off))) if off.size else None,
                         "late_gt12": C.frac(np.sum(off > 12), off.size),
                         "early_lt-12": C.frac(np.sum(off < -12), off.size)}
        m25 = r[r.det_25]
        d["matched25_across_ours"] = C.frac(np.sum(r["our_kind50"].eq("across_D") & r.det_25), len(m25))
        d["ball_missing_at"] = C.frac(np.sum(~r["ball_vis_at"].astype(bool)), len(r))
        d["ball_missing_pm12_all"] = C.frac(np.sum(r["ball_vis_pm12"] == 0.0), len(r))
        d["new_dur_bins"] = bins_count(r["new_dur_s"], DUR_BINS)
        # recall by duration of the new reference possession
        d["det25_by_new_dur"] = {}
        for name, lo, hi in (("<1s", 0, 1), ("1-2s", 1, 2), ("2-5s", 2, 5), (">=5s", 5, 1e9)):
            s = r[(r.new_dur_s >= lo) & (r.new_dur_s < hi)]
            d["det25_by_new_dur"][name] = {"n": int(len(s)),
                                           "det25": float(s.det_25.mean()) if len(s) else None}
        out["ref_" + kind] = d
    # spurious
    sp = ours[~ours.matched25.astype(bool)]
    out["ours"] = {"n": int(len(ours)), "per_hour": len(ours) / hours,
                   "n_direct": int((ours.kind == "direct").sum()),
                   "n_across": int((ours.kind == "across_D").sum()),
                   "spurious": int(len(sp)), "spurious_per_hour": len(sp) / hours,
                   "spurious_share": C.frac(len(sp), len(ours)),
                   "spurious_loose_no_ref_any_pm25": int((ours.nearest_ref_any > SPUR_TOL).sum()),
                   "spurious_by_kind": sp.kind.value_counts().to_dict()}
    spd = sp[sp.kind == "direct"]
    out["ours"]["spurious_direct_return_bins"] = bins_count(
        [x if x is not None and not (isinstance(x, float) and np.isnan(x)) else 1e9
         for x in spd.return_s], FLICKER_BINS)
    out["ours"]["spurious_all_return_bins"] = bins_count(
        [x if x is not None and not (isinstance(x, float) and np.isnan(x)) else 1e9
         for x in sp.return_s], FLICKER_BINS)
    out["ours"]["spurious_return_kind"] = sp.return_kind.fillna("none").value_counts().to_dict()
    out["ours"]["spurious_trigger"] = sp.trigger.value_counts().to_dict()
    out["ours"]["spurious_trigger_owner_event"] = C.frac(
        np.sum(spd.trigger_owner_event.fillna(False).astype(bool)), len(spd))
    out["ours"]["spurious_ball_missing_at"] = C.frac(np.sum(~sp.ball_vis_at.astype(bool)), len(sp))
    out["ours"]["matched_ball_missing_at"] = C.frac(
        np.sum(~ours[ours.matched25.astype(bool)].ball_vis_at.astype(bool)),
        int(ours.matched25.astype(bool).sum()))
    out["ours"]["spurious_ref_at"] = sp.ref_at.value_counts().to_dict()
    out["ours"]["spurious_ref_new_frac_median"] = float(sp.ref_new_frac.median()) if len(sp) else None
    out["ours"]["matched_trigger"] = ours[ours.matched25.astype(bool)].trigger.value_counts().to_dict()
    out["ours"]["all_trigger"] = ours.trigger.value_counts().to_dict()
    # missed direct hand-overs
    rd = ref[ref.kind == "direct"]
    ms = rd[~rd.det_25.astype(bool)]
    mo = {"n": int(len(ms)), "share_of_direct": C.frac(len(ms), len(rd)),
          "per_hour": len(ms) / hours}
    mo["detected_25_50"] = int(ms.det_50.sum())
    mo["ball_missing_at"] = C.frac(np.sum(~ms.ball_vis_at.astype(bool)), len(ms))
    mo["ball_vis_frac_pm1s_median"] = float(ms.ball_vis_frac.median()) if len(ms) else None
    mo["ball_vis_pm1s_all_missing"] = C.frac(np.sum(ms.ball_vis_frac == 0), len(ms))
    mo["speed_at_quantiles"] = C.quantiles(ms.ball_speed_at.dropna())
    det_rows = rd[rd.det_25.astype(bool)]
    mo["speed_at_quantiles_detected"] = C.quantiles(det_rows.ball_speed_at.dropna())
    mo["ball_missing_at_detected"] = C.frac(np.sum(~det_rows.ball_vis_at.astype(bool)), len(det_rows))
    for name in ("1s", "3s"):
        mo["any_new_event_%s" % name] = C.frac(np.sum(ms["n_ev_new_%s" % name] > 0), len(ms))
        mo["any_teamrow_%s" % name] = C.frac(np.sum(ms["n_teamrow_new_%s" % name] > 0), len(ms))
        mo["any_owner_event_%s" % name] = C.frac(np.sum(ms["n_owner_new_%s" % name] > 0), len(ms))
        mo["detected_any_owner_event_%s" % name] = C.frac(
            np.sum(det_rows["n_owner_new_%s" % name] > 0), len(det_rows))
    mo["new_dur_bins"] = bins_count(ms.new_dur_s, DUR_BINS)
    mo["new_dur_bins_detected"] = bins_count(det_rows.new_dur_s, DUR_BINS)
    mo["new_ends_direct"] = C.frac(np.sum(ms.new_ends_direct.astype(bool)), len(ms))
    mo["vit_has_new"] = C.frac(np.sum(ms.vit_has_new.astype(bool)), len(ms))
    mo["final_has_new"] = C.frac(np.sum(ms.final_has_new.astype(bool)), len(ms))
    mo["owner_new_frac_median"] = float(ms.owner_new_frac.median()) if len(ms) else None
    # reason ladder
    reasons = []
    for _, x in ms.iterrows():
        if x.det_50:
            reasons.append("detected 25-50 frames off")
        elif x.n_owner_new_3s > 0 and x.owner_new_frac and x.owner_new_frac > 0:
            if x.vit_has_new and not x.final_has_new:
                reasons.append("owner track flipped; Viterbi took it; min-duration pass removed")
            elif x.final_has_new:
                reasons.append("owner track flipped; our label has new team but change >50 off")
            else:
                reasons.append("owner track flipped; decoder did not switch")
        elif x.n_owner_new_3s > 0:
            reasons.append("owner event of new team, owner track never new team in window")
        elif x.n_teamrow_new_3s > 0:
            reasons.append("only non-owner rows of new team (touch/low conf)")
        elif x.n_ev_new_3s > 0:
            reasons.append("only other rows of new team")
        else:
            reasons.append("no event of new team within 3 s")
    mo["reasons"] = pd.Series(reasons, dtype=object).value_counts().to_dict() if reasons else {}
    lab_counts = {}
    for labs in ms["labs_nonowner_1s"]:
        for lab in [t for t in str(labs).split(";") if t]:
            lab_counts[lab] = lab_counts.get(lab, 0) + 1
    mo["nonowner_labels_1s"] = dict(sorted(lab_counts.items(), key=lambda kv: -kv[1]))
    lab_counts = {}
    for labs in ms["labs_new_1s"]:
        for lab in [t for t in str(labs).split(";") if t]:
            lab_counts[lab] = lab_counts.get(lab, 0) + 1
    mo["new_team_labels_1s"] = dict(sorted(lab_counts.items(), key=lambda kv: -kv[1]))
    mo["contested_1s"] = C.frac(np.sum(ms.n_contested_1s > 0), len(ms))
    out["missed_direct"] = mo
    # late detections: what fired our change, what was near the reference change
    late = rd[rd.det_50.astype(bool) & (rd.offset50 > 12)]
    out["late_direct"] = {"n": int(len(late)),
                          "our_trigger": late.our_trigger25.fillna("not_matched_at_25").value_counts().to_dict(),
                          "nearest_owner_new_off_quantiles": C.quantiles(late.nearest_owner_new_off.dropna())}
    early = rd[rd.det_50.astype(bool) & (rd.offset50 < -12)]
    out["early_direct"] = {"n": int(len(early)),
                           "our_trigger": early.our_trigger25.fillna("not_matched_at_25").value_counts().to_dict()}
    return out


def main():
    os.makedirs(os.path.join(WORK, "out"), exist_ok=True)
    report = {}
    for view in ("matched", "raw"):
        per, all_rows = [], []
        for game in C.OPEN_GAMES:
            res, rows = analyse(view, game)
            per.append(res)
            all_rows += rows
        df = pd.DataFrame(all_rows)
        df.to_csv(os.path.join(WORK, "out", "changes_%s.csv" % view), index=False)
        perdf = pd.DataFrame(per)
        rep = {"per_match": per}
        for name, sel in (("Metrica", perdf.group == "Metrica"), ("DFL", perdf.group == "DFL"),
                          ("pooled", perdf.group.notna())):
            games = set(perdf[sel].game)
            rep[name] = summarise(df[df.game.isin(games)], perdf[sel])
        report[view] = rep
    with open(os.path.join(WORK, "out", "q123.json"), "w") as handle:
        json.dump(report, handle, indent=1, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote out/q123.json")


if __name__ == "__main__":
    main()
