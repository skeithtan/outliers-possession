"""Context of every owner change, from the error episodes of q4.py (frame-exact).

    python q13_context.py    -> out/q13_context.json, out/ref_context_<view>.csv,
                                out/our_context_<view>.csv

Reference direct hand-over r (W -> R), by the team episode that starts or ends at r:
  exact            no episode at r: we change W -> R at r
  late<=25 / late>25         a T_late episode starts at r (we follow after its length)
  early<=25 / early>25       a T_early episode ends at r (we changed before r)
  absorbed_start   a T_missed_absorbed episode starts at r (R's whole possession sits in ours)
  absorbed_return  a T_missed_absorbed episode ends at r (the return edge of an absorbed spell)
  missed_to_dead   a T_missed_to_dead episode starts at r (we never follow before a D)
  our_D            our label at r or r-1 is D
  other            anything else (restart or swap episodes)
Our change f (W -> R'), by the episode that ends or starts at f:
  exact, late_adoption<=25 / >25 (ends a T_late), early<=25 / >25 (starts a T_early),
  true_spurious_flicker (starts a T_spurious that ends with our return), true_spurious_to_dead,
  true_spurious_return (ends a T_spurious episode), restart_fix (ends a T_restart_* episode),
  wrong_restart (starts a T_restart_* episode), late_restart_right_team (our D exit after the
  reference restarted with the same team), ref_D (the reference is D at f), other.
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

TEAM_CLASSES = ("T_late", "T_missed_absorbed", "T_missed_to_dead", "T_early", "T_spurious",
                "T_restart_wrong", "T_restart_after_missedD", "T_swap")
REF_ORDER = ("exact", "late<=25", "late>25", "early<=25", "early>25", "absorbed_start",
             "absorbed_return", "missed_to_dead", "our_D", "other")
OUR_ORDER = ("exact", "late_adoption<=25", "late_adoption>25", "early<=25", "early>25",
             "true_spurious_flicker", "true_spurious_to_dead", "true_spurious_return",
             "restart_fix", "wrong_restart", "late_restart_right_team", "ref_D", "other")


def ref_context(r, gt, ours, by_start, by_stop):
    if ours[r] == C.D or ours[r - 1] == C.D:
        return "our_D", None
    s = by_start.get(r)
    if s is not None:
        if s["cls"] == "T_late":
            return ("late<=25" if s["len"] <= 25 else "late>25"), s["len"]
        if s["cls"] == "T_missed_absorbed":
            return "absorbed_start", s["len"]
        if s["cls"] == "T_missed_to_dead":
            return "missed_to_dead", s["len"]
    e = by_stop.get(r)
    if e is not None:
        if e["cls"] == "T_early":
            return ("early<=25" if e["len"] <= 25 else "early>25"), -e["len"]
        if e["cls"] == "T_missed_absorbed":
            return "absorbed_return", None
    if ours[r] == gt[r] and ours[r - 1] == gt[r - 1]:
        return "exact", 0
    return "other", None


def our_context(f, gt, ours, by_start, by_stop):
    if gt[f] == C.D:
        return "ref_D", None
    e = by_stop.get(f)
    s = by_start.get(f)
    if e is not None and e["cls"] == "T_late":
        return ("late_adoption<=25" if e["len"] <= 25 else "late_adoption>25"), e["len"]
    if s is not None and s["cls"] == "T_early":
        return ("early<=25" if s["len"] <= 25 else "early>25"), -s["len"]
    if s is not None and s["cls"] == "T_spurious":
        flick = s["sub"].endswith(">our_change")
        return ("true_spurious_flicker" if flick else "true_spurious_to_dead"), s["len"]
    if e is not None and e["cls"].startswith("T_restart"):
        return "restart_fix", e["len"]
    if e is not None and e["cls"] == "T_spurious":
        return "true_spurious_return", e["len"]
    if ours[f - 1] == C.D and gt[f - 1] == gt[f] and ours[f] == gt[f]:
        return "late_restart_right_team", None
    if s is not None and s["cls"].startswith("T_restart"):
        return "wrong_restart", s["len"]
    if ours[f] == gt[f] and ours[f - 1] == gt[f - 1]:
        return "exact", 0
    return "other", None


def analyse(view, game, eps):
    M = C.load(view, game)
    gt, ours, runs = M["gt"], M["ours"], M["runs"]
    e = eps[(eps.game == game) & eps.cls.isin(TEAM_CLASSES)]
    by_start = {int(x.start): x for _, x in e.iterrows()}
    by_stop = {int(x.stop): x for _, x in e.iterrows()}
    ref_all = [c for c in C.owner_changes(gt, runs) if c["kind"] != "same_D"]
    our_all = [c for c in C.owner_changes(ours, runs) if c["kind"] != "same_D"]
    pairs, _, our_un = C.match_changes(ref_all, our_all, 25)
    ref_m = set(i for i, _ in pairs)
    our_m = set(j for _, j in pairs)
    rrows, orows = [], []
    for i, c in enumerate(ref_all):
        if c["kind"] != "direct":
            continue
        ctx, lag = ref_context(c["frame"], gt, ours, by_start, by_stop)
        rrows.append({"view": view, "game": game, "group": M["group"], "frame": c["frame"],
                      "context": ctx, "lag": lag, "det25": i in ref_m,
                      "new_dur_s": (c["next_stop"] - c["frame"]) / C.FPS})
    for j, c in enumerate(our_all):
        ctx, lag = our_context(c["frame"], gt, ours, by_start, by_stop)
        nxt = our_all[j + 1] if j + 1 < len(our_all) and our_all[j + 1]["run"] == c["run"] else None
        orows.append({"view": view, "game": game, "group": M["group"], "frame": c["frame"],
                      "kind": c["kind"], "context": ctx, "lag": lag, "matched25": j in our_m,
                      "return_s": (nxt["frame"] - c["frame"]) / C.FPS if nxt else None})
    return rrows, orows


def main():
    report = {}
    for view in ("matched", "raw"):
        eps = pd.read_csv(os.path.join(WORK, "out", "episodes_%s.csv" % view))
        rr, oo = [], []
        for game in C.OPEN_GAMES:
            a, b = analyse(view, game, eps)
            rr += a
            oo += b
        rdf, odf = pd.DataFrame(rr), pd.DataFrame(oo)
        rdf.to_csv(os.path.join(WORK, "out", "ref_context_%s.csv" % view), index=False)
        odf.to_csv(os.path.join(WORK, "out", "our_context_%s.csv" % view), index=False)
        hours = {}
        for game in C.OPEN_GAMES:
            hours[game] = C.load(view, game)["hours"]
        rep = {}
        for grp in ("Metrica", "DFL", "pooled"):
            r = rdf if grp == "pooled" else rdf[rdf.group == grp]
            o = odf if grp == "pooled" else odf[odf.group == grp]
            h = sum(hours[g] for g in set(o.game))
            ou = o[~o.matched25]
            rep[grp] = {
                "ref_direct_n": int(len(r)),
                "ref_context": r.context.value_counts().reindex(REF_ORDER).fillna(0).astype(int).to_dict(),
                "ref_context_missed25": r[~r.det25].context.value_counts().reindex(REF_ORDER).fillna(0).astype(int).to_dict(),
                "late_gt25_lag_q": C.quantiles(r[r.context == "late>25"].lag),
                "early_gt25_lead_q": C.quantiles(-r[r.context == "early>25"].lag),
                "absorbed_len_q": C.quantiles(r[r.context == "absorbed_start"].lag / C.FPS),
                "missed_to_dead_len_q": C.quantiles(r[r.context == "missed_to_dead"].lag / C.FPS),
                "our_n": int(len(o)), "our_per_hour": len(o) / h,
                "our_unmatched_n": int(len(ou)), "our_unmatched_per_hour": len(ou) / h,
                "our_context_all": o.context.value_counts().reindex(OUR_ORDER).fillna(0).astype(int).to_dict(),
                "our_context_unmatched": ou.context.value_counts().reindex(OUR_ORDER).fillna(0).astype(int).to_dict(),
                "late_adoption_gt25_lag_q": C.quantiles(o[o.context == "late_adoption>25"].lag),
                "flicker_return_bins": {str(k): int(v) for k, v in pd.cut(
                    o[o.context == "true_spurious_flicker"].return_s, [0, 0.5, 1, 2, 5, 10, 1e9],
                    right=False).value_counts().sort_index().items()},
                "hours": h,
            }
        report[view] = rep
    with open(os.path.join(WORK, "out", "q13_context.json"), "w") as handle:
        json.dump(report, handle, indent=1, sort_keys=True,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for view in ("matched", "raw"):
        for grp in ("Metrica", "DFL", "pooled"):
            print(view, grp, json.dumps(report[view][grp], default=str)[:2500])


if __name__ == "__main__":
    main()
