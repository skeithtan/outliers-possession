"""Build the tables of report.md from out/*.json and out/*.csv (deterministic).

    python report_tables.py   -> out/tables.md
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("ERROR_ANATOMY_WORK", os.path.join("/tmp", "error_anatomy"))  # outputs, never the repo
OUT = os.path.join(WORK, "out")
VIEWS = ("matched", "raw")
GROUPS = ("Metrica", "DFL", "pooled")


def J(name):
    with open(os.path.join(OUT, name)) as h:
        return json.load(h)


def pct(x, d=0):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else ("{:.%df}%%" % d).format(100 * x)


def num(x, fmt="{:.1f}"):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


tb = lambda s: s.astype(str).str.lower().eq("true")  # noqa: E731


def main():
    q123, q4, q13, q56, qd, qa, qp = (J("q123.json"), J("q4.json"), J("q13_context.json"),
                                      J("q56.json"), J("q3_detector.json"), J("q_attr.json"),
                                      J("q_plaus.json"))
    md = []
    # ---- Q1
    md.append("### T1. Reference owner changes and our detection (one-to-one per destination team)\n")
    rows = []
    for v in VIEWS:
        for g in GROUPS:
            for kind, lab in (("ref_direct", "direct live"), ("ref_across_D", "across D")):
                d = q123[v][g][kind]
                o = d["offset50"]
                rows.append([v, g, lab, d["n"], num(d["per_hour"]), pct(d["det_5"]), pct(d["det_12"]),
                             pct(d["det_25"]), pct(d["det_50"]), num(o["median"], "{:+.0f}"),
                             "{} .. {}".format(num(o["p25"], "{:+.0f}"), num(o["p75"], "{:+.0f}")),
                             pct(o["late"]), pct(o["early"]), pct(o["zero"]), pct(o["late_gt12"]),
                             pct(o["early_lt-12"])])
    md.append(table(["view", "group", "kind", "n", "per h", "det ±5", "±12", "±25", "±50",
                     "offset median (fr)", "IQR", "late", "early", "exact", "late >12", "early <-12"], rows))
    md.append("\nOffsets: our change frame minus reference frame, over the pairs matched at ±50. "
              "Positive = we are late.\n")

    md.append("### T2. Context of every reference direct hand-over (frame-exact, from the error episodes)\n")
    order = ("exact", "late<=25", "early<=25", "late>25", "early>25", "absorbed_start",
             "absorbed_return", "missed_to_dead", "our_D", "other")
    rows = []
    for v in VIEWS:
        for g in GROUPS:
            d = q13[v][g]
            n = d["ref_direct_n"]
            rows.append([v, g, n] + [pct(d["ref_context"][k] / n) for k in order])
    md.append(table(["view", "group", "n"] + list(order), rows))
    rows = []
    for v in VIEWS:
        for g in GROUPS:
            d = q13[v][g]
            rows.append([v, g, num(d["late_gt25_lag_q"]["50"], "{:.0f}"),
                         "{:.0f} .. {:.0f}".format(d["late_gt25_lag_q"]["25"], d["late_gt25_lag_q"]["75"]),
                         num(d["early_gt25_lead_q"]["50"], "{:.0f}"),
                         num(d["absorbed_len_q"]["50"], "{:.2f}"),
                         "{:.2f} .. {:.2f}".format(d["absorbed_len_q"]["25"], d["absorbed_len_q"]["75"]),
                         num(d["missed_to_dead_len_q"]["50"], "{:.2f}")])
    md.append("\n" + table(["view", "group", "late>25 lag median (fr)", "IQR", "early>25 lead median",
                            "absorbed possession median (s)", "IQR (s)", "missed-to-D possession median (s)"], rows))

    # ---- Q2
    md.append("\n### T3. Our owner changes that match no reference change within ±25 frames\n")
    order2 = ("late_adoption>25", "early>25", "true_spurious_flicker", "true_spurious_return",
              "true_spurious_to_dead", "restart_fix", "wrong_restart", "late_restart_right_team",
              "ref_D", "exact", "other")
    rows = []
    for v in VIEWS:
        for g in GROUPS:
            d = q13[v][g]
            q = q123[v][g]["ours"]
            n = d["our_unmatched_n"]
            rows.append([v, g, d["our_n"], num(d["our_per_hour"]), n, num(d["our_unmatched_per_hour"]),
                         q["spurious_loose_no_ref_any_pm25"]] +
                        [d["our_context_unmatched"][k] for k in order2])
    md.append(table(["view", "group", "our changes", "per h", "unmatched ±25", "per h",
                     "no ref change of any team ±25"] + list(order2), rows))
    md.append("\nReturn time of the unmatched changes (to our next owner change), and of the true "
              "wrong-team flickers:\n")
    rows = []
    for v in VIEWS:
        for g in GROUPS:
            q = q123[v][g]["ours"]
            b = q["spurious_all_return_bins"]
            f = q13[v][g]["flicker_return_bins"]
            rows.append([v, g] + [b[k] for k in ("<0.5s", "0.5-1s", "1-2s", "2-5s", "5-10s", ">=10s")] +
                        [" / ".join(str(f[k]) for k in sorted(f, key=lambda s: float(s.split(",")[0][1:])))])
    md.append(table(["view", "group", "<0.5 s", "0.5-1 s", "1-2 s", "2-5 s", "5-10 s", ">=10 s or never",
                     "true flickers by return bin [0,.5)/[.5,1)/[1,2)/[2,5)/[5,10)/[10,inf)"], rows))
    md.append("\nWhat fired the change (the new team's row within ±3 frames of a direct change, or the "
              "restart team of an across-D change), unmatched vs matched at ±25:\n")
    for v in VIEWS:
        rows = []
        for g in ("Metrica", "DFL"):
            q = q123[v][g]["ours"]
            keys = sorted(set(q["spurious_trigger"]) | set(q["matched_trigger"]))
            for k in keys:
                rows.append([g, k, q["spurious_trigger"].get(k, 0), q["matched_trigger"].get(k, 0)])
        md.append("view {}:\n".format(v))
        md.append(table(["group", "trigger", "unmatched", "matched"], rows) + "\n")
    rows = []
    for v in VIEWS:
        for g in GROUPS:
            q = q123[v][g]["ours"]
            rows.append([v, g, pct(q["spurious_ball_missing_at"], 1), pct(q["matched_ball_missing_at"], 1),
                         pct(q["spurious_trigger_owner_event"])])
    md.append(table(["view", "group", "ball missing at unmatched change", "at matched change",
                     "direct unmatched changes fired by an owner event"], rows))

    # ---- Q3
    md.append("\n### T4. Missed hand-overs (reference direct, unmatched at ±25): new-team evidence\n")
    rows = []
    for v in VIEWS:
        rc = pd.read_csv(os.path.join(OUT, "ref_context_%s.csv" % v))
        ch = pd.read_csv(os.path.join(OUT, "changes_%s.csv" % v))
        ref = ch[(ch.side == "ref") & (ch.kind == "direct")]
        m = rc.merge(ref, on=["game", "frame"], suffixes=("", "_c"))
        for g in ("Metrica", "DFL"):
            gg = m[m.group == g]
            for ctx in ("absorbed_start", "late>25", "missed_to_dead", "matched<=25"):
                x = gg[gg.context.isin(["exact", "late<=25", "early<=25"])] if ctx == "matched<=25" else gg[gg.context == ctx]
                rows.append([v, g, ctx, len(x), pct((x.n_ev_new_1s > 0).mean()), pct((x.n_owner_new_1s > 0).mean()),
                             pct((x.n_ev_new_3s > 0).mean()), pct((x.n_owner_new_3s > 0).mean()),
                             pct((x.n_contested_1s > 0).mean()), pct((~tb(x.ball_vis_at)).mean(), 1),
                             num(x.ball_speed_at.median()), pct(tb(x.vit_has_new).mean())])
    md.append(table(["view", "group", "context", "n", "new-team row ±1 s", "owner event ±1 s",
                     "row ±3 s", "owner ±3 s", "contested row ±1 s", "ball missing at r",
                     "ball speed at r (m/s, median)", "Viterbi had new team (window)"], rows))

    md.append("\n### T5. Detector ladder at reference direct hand-overs (best new-team evidence within ±12 frames)\n")
    lad = ("A_row_new_owner", "B_row_new_nonowner", "C_bump_new_no_row", "D_bump_new_reach_no_row",
           "E_bump_contested", "F_bump_old_team", "G_subthr6_new", "H_subthr4_new", "I_subthr_other",
           "J_no_bump_new_near", "K_no_bump_new_far")
    rows = []
    for v in VIEWS:
        rc = pd.read_csv(os.path.join(OUT, "ref_context_%s.csv" % v))
        hd = pd.read_csv(os.path.join(OUT, "handover_detector_%s.csv" % v))
        m = rc.merge(hd[["game", "frame", "ladder"]], on=["game", "frame"])
        for g in ("Metrica", "DFL"):
            gg = m[m.group == g]
            for name, sel in (("missed ±25 (all)", ~gg.det25),
                              ("absorbed_start", gg.context == "absorbed_start"),
                              ("late>25", gg.context == "late>25"),
                              ("missed_to_dead", gg.context == "missed_to_dead"),
                              ("matched ±25", gg.det25)):
                x = gg[sel]
                vc = x.ladder.value_counts()
                rows.append([v, g, name, len(x)] + [pct(vc.get(k, 0) / max(1, len(x))) for k in lad])
    md.append(table(["view", "group", "set", "n"] + [k.split("_", 1)[0] for k in lad], rows))
    md.append("\nLadder: A new-team owner-event row; B new-team row that is not an owner event; C bump "
              "|J| >= 9 m/s nearest new team within 1.5 m but no row (unconfirmed redirect); D same with "
              "the new team at 1.5-3 m (reach not confirmed); E bump >= 9 contested; F bump >= 9 nearer "
              "the OLD team; G bump 6-9 m/s nearest new team; H bump 4-6 nearest new team; I bump 4-9 "
              "other team/contested/far; J no bump >= 4 but a new-team player within 1.5 m; K nothing.\n")

    md.append("### T6. Team attribution of strong bumps (|J| >= 9 m/s): engine window vs alternatives\n")
    rows = []
    for v in VIEWS:
        for g in ("Metrica", "DFL"):
            for pop in ("handover", "inside"):
                for r in ("engine", "post", "post6", "at_c", "pre"):
                    d = qa[v][g][pop][r]
                    rows.append([v, g, pop, r, d["n"], pct(d["correct"]), pct(d["wrong"]),
                                 pct(d["contested"]), pct(d["far"])])
    md.append(table(["view", "group", "bumps", "rule", "n", "right team", "wrong team", "contested",
                     "nobody within 1.5 m"], rows))

    md.append("\n### T7. Tracking plausibility of reference possessions (share of frames with the nearest player within 1.5 m)\n")
    rows = []
    for v in VIEWS:
        for g in ("Metrica", "DFL"):
            for k, d in sorted(qp[v][g].items()):
                rows.append([v, g, k.replace("|", " "), int(d["n"]), pct(d["own"]), pct(d["oth"]),
                             pct(d["none"]), pct(d["own_any"])])
    md.append(table(["view", "group", "possessions", "n", "own team nearest", "other team nearest",
                     "nobody", "any own-team frame"], rows))

    # ---- Q4
    md.append("\n### T8. Shapley decomposition of the spellF1 loss by error class (exact, 2^12 subsets)\n")
    cls = ("T_missed_absorbed", "T_missed_to_dead", "T_late", "T_early", "T_spurious",
           "T_restart_wrong", "T_restart_after_missedD", "T_swap", "D_missed", "D_onset",
           "D_restart", "D_spurious")
    rows = []
    for v in VIEWS:
        for g in ("Metrica", "DFL", "balanced"):
            d = q4[v][g]
            rows.append([v, g, num(d["base"], "{:.3f}"), num(1 - d["base"], "{:.3f}")] +
                        [num(d["shapley_mean"][c], "{:.3f}") for c in cls])
    md.append(table(["view", "group", "spellF1", "loss"] + list(cls), rows))
    md.append("\nAlone (fix only this class) and last (fix everything else first) gains:\n")
    rows = []
    for v in VIEWS:
        for g in ("Metrica", "DFL"):
            d = q4[v][g]
            rows.append([v, g, "alone"] + [num(d["alone_mean"][c], "{:+.3f}") for c in cls])
            rows.append([v, g, "last"] + [num(d["last_mean"][c], "{:+.3f}") for c in cls])
    md.append(table(["view", "group", "gain"] + list(cls), rows))
    rows = []
    for v in VIEWS:
        for g in ("Metrica", "DFL"):
            d = q4[v][g]
            rows.append([v, g, num(d["team_all"], "{:+.3f}"), num(d["dead_all"], "{:+.3f}")])
    md.append("\nAll team classes at once (= D2 oracle_team) and all D classes at once:\n")
    md.append(table(["view", "group", "fix all T_*", "fix all D_*"], rows))

    md.append("\n### T9. Unmatched possessions at IoU 0.5, attributed to the class whose fix alone recovers them\n")
    rows = []
    for v in VIEWS:
        for g in ("Metrica", "DFL"):
            d = q4[v][g]
            ra, pa = d["ref_attr"], d["pred_attr"]
            rows.append([v, g, "reference", d["n_ref_spells"], d["ref_unmatched"]] +
                        ["{} ({:.0f})".format(ra["single_exclusive"][c], ra["fractional_share"][c]) for c in cls] +
                        [ra["ambiguous_single"], ra["compound_n"]])
            rows.append([v, g, "predicted", d["n_pred_spells"], d["pred_unmatched"]] +
                        ["{} ({:.0f})".format(pa["single_exclusive"][c], pa["fractional_share"][c]) for c in cls] +
                        [pa["ambiguous_single"], pa["compound_n"]])
    md.append(table(["view", "group", "side", "possessions", "unmatched"] + list(cls) +
                    ["several single fixes", "needs 2+ classes"], rows))
    md.append("\nCell = possessions recovered by that class alone and by no other single class "
              "(fractional share in brackets: ambiguous and compound cases split equally over the "
              "classes of their minimal fixing subsets).\n")
    md.append("### T10. Error episodes per class (count, frames)\n")
    rows = []
    for v in VIEWS:
        for g in ("Metrica", "DFL"):
            d = q4[v][g]
            rows.append([v, g] + ["{} / {}".format(d["episodes"][c]["n"], d["episodes"][c]["frames"]) for c in cls])
    md.append(table(["view", "group"] + list(cls), rows))

    # ---- Q5 Q6
    md.append("\n### T11. Possession durations and recall by duration (IoU 0.5)\n")
    rows = []
    for v in VIEWS:
        for g in GROUPS:
            d = q56[v][g]
            for side in ("ref", "our"):
                q = d[side + "_q"]
                s = d[side + "_share"]
                rows.append([v, g, side, d[side + "_n"], num(d[side + "_mean_s"]), num(q["10"], "{:.2f}"),
                             num(q["25"], "{:.2f}"), num(q["50"], "{:.2f}"), num(q["75"], "{:.2f}"),
                             num(q["90"], "{:.2f}"), pct(s["<1s"]), pct(s["<2s"]), pct(s["<5s"])])
    md.append(table(["view", "group", "side", "n", "mean s", "p10", "p25", "p50", "p75", "p90",
                     "<1 s", "<2 s", "<5 s"], rows))
    rows = []
    for v in VIEWS:
        for g in GROUPS:
            d = q56[v][g]
            rows.append([v, g, "recall of ref"] + ["{} ({})".format(pct(d["recall_by_ref_dur"][k]["recall"]),
                                                                    d["recall_by_ref_dur"][k]["n"])
                                                   for k in ("<1s", "1-2s", "2-5s", "5-10s", ">=10s")])
            rows.append([v, g, "precision of ours"] + ["{} ({})".format(pct(d["precision_by_our_dur"][k]["precision"]),
                                                                        d["precision_by_our_dur"][k]["n"])
                                                       for k in ("<1s", "1-2s", "2-5s", "5-10s", ">=10s")])
    md.append("\n" + table(["view", "group", "measure", "<1 s", "1-2 s", "2-5 s", "5-10 s", ">=10 s"], rows))
    md.append("\n### T12. Ball missing in our input at reference hand-overs\n")
    rows = []
    for v in VIEWS:
        for g in GROUPS:
            d = q56[v][g]
            rows.append([v, g, d["n_direct"], pct(d["q6_direct_ball_missing_at"], 1),
                         pct(d["q6_direct_ball_missing_any_pm12"], 1), d["n_across"],
                         pct(d["q6_across_ball_missing_at_restart"], 1),
                         pct(d["q6_alive_frames_ball_missing"], 2)])
    md.append(table(["view", "group", "direct hand-overs", "ball missing at r", "any missing frame in ±12",
                     "across-D changes", "ball missing at the restart frame", "reference-alive frames with ball missing"], rows))
    with open(os.path.join(OUT, "tables.md"), "w") as h:
        h.write("\n".join(md) + "\n")
    print("wrote out/tables.md", sum(len(x) for x in md))


if __name__ == "__main__":
    main()
