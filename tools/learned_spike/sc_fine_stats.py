"""The two SkillCorner references side by side (coarse: phases of play; fine: possession.group),
with the Metrica and DFL references as provided and the Metrica loose-touch variant.

    python tools/learned_spike/sc_fine_stats.py --cache CACHE --fine OUT/cache_possgroup \
        --raw-root RAW --out DIR [--metrica-variant DIR]

Statistics per match at 25 fps on the graded frames (possession_metrics.valid_runs of the
reference): possessions (maximal H or A runs inside a graded run), possessions per live minute
(live = H or A frames), share under 1 s and under 2 s, median and mean length, D share. For
SkillCorner also: frame agreement of the two references on live frames, boundary nesting (is each
coarse boundary a fine boundary, exactly and within 1 s; the share of fine boundaries that are
coarse ones), team changes of the fine view inside phases with and without
`team_possession_loss_in_phase`, and what `possession.group` is made of (10 Hz, in periods:
named share, whether a named group always comes with a `player_id` of that team).

External data only (SkillCorner Open Data, MIT, SkillCorner; Metrica sample data; open DFL). It
reads cache labels and the SkillCorner raw files, nothing else. Python 3.9, numpy.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import possession_metrics as pm  # noqa: E402
from tools.learned_spike import folds  # noqa: E402

FPS = 25
H, A, D, X = pm.H, pm.A, pm.D, pm.X
TOL = FPS            # 1 s tolerance for boundary nesting


def spells(codes, runs):
    out = []
    for a, b in runs:
        w = codes[a:b]
        br = np.flatnonzero(w[1:] != w[:-1]) + 1
        st = np.concatenate(([0], br)) + a
        sp = np.concatenate((br, [b - a])) + a
        for s, e in zip(st, sp):
            if codes[s] in (H, A):
                out.append((int(s), int(e), int(codes[s])))
    return out


def boundaries(codes, runs):
    """Frames f (inside a graded run) with codes[f] != codes[f - 1]."""
    out = []
    for a, b in runs:
        w = codes[a:b]
        out += (np.flatnonzero(w[1:] != w[:-1]) + 1 + a).tolist()
    return np.asarray(out, dtype=np.int64)


def profile(codes, runs):
    codes = np.asarray(codes, dtype=np.int8)
    sp = spells(codes, runs)
    L = np.array([(e - s) / float(FPS) for s, e, _ in sp])
    graded = sum(b - a for a, b in runs)
    live = int(sum(np.isin(codes[a:b], (H, A)).sum() for a, b in runs))
    return {"possessions": len(sp), "live_min": live / (60.0 * FPS),
            "per_live_min": len(sp) / (live / (60.0 * FPS)),
            "under_1s": float((L < 1).mean()), "under_2s": float((L < 2).mean()),
            "median_s": float(np.median(L)), "mean_s": float(L.mean()),
            "d_share": float(sum((codes[a:b] == D).sum() for a, b in runs)) / graded}


def nesting(coarse, fine, runs):
    bc, bf = boundaries(coarse, runs), boundaries(fine, runs)
    sf = set(bf.tolist())
    exact = float(np.mean([b in sf for b in bc])) if len(bc) else 1.0
    if len(bf):
        idx = np.searchsorted(bf, bc)
        near = []
        for b, i in zip(bc, idx):
            d = min(abs(b - bf[j]) for j in (i - 1, i) if 0 <= j < len(bf))
            near.append(d <= TOL)
        within = float(np.mean(near))
    else:
        within = 0.0
    # coarse H<->A switches only (the D edges are shared by construction)
    ha = [b for b in bc if coarse[b] in (H, A) and coarse[b - 1] in (H, A)]
    ha_exact = float(np.mean([b in sf for b in ha])) if ha else 1.0
    ha_fine_same = float(np.mean([fine[b] == coarse[b] and fine[b - 1] == coarse[b - 1]
                                  for b in ha])) if ha else 1.0
    return {"coarse_boundaries": len(bc), "fine_boundaries": len(bf),
            "coarse_in_fine_exact": exact, "coarse_in_fine_1s": within,
            "coarse_ha_switches": len(ha), "coarse_ha_in_fine_exact": ha_exact,
            "coarse_ha_same_both_sides": ha_fine_same,
            "fine_that_are_coarse": float(np.mean([b in set(bc.tolist()) for b in bf]))
            if len(bf) else 1.0}


def raw_field(match_dir, mid):
    """possession.group / player_id facts at 10 Hz, and the phases with their loss flag."""
    with open(os.path.join(match_dir, mid + "_match.json")) as h:
        m = json.load(h)
    home = int(m["home_team"]["id"])
    team_of = {int(p["id"]): int(p["team_id"]) for p in m["players"]}
    named = with_pid = pid_team_ok = n_per = pid_without_group = 0
    group10 = []
    with open(os.path.join(match_dir, mid + "_tracking_extrapolated.jsonl")) as h:
        for line in h:
            if not line.strip():
                continue
            d = json.loads(line)
            pos = d.get("possession") or {}
            g, pid = pos.get("group"), pos.get("player_id")
            code = H if g == "home team" else A if g == "away team" else X
            group10.append(code)
            if not d.get("period"):
                continue
            n_per += 1
            if pid is not None and g is None:
                pid_without_group += 1
            if g is None:
                continue
            named += 1
            if pid is not None:
                with_pid += 1
                t = team_of.get(int(pid))
                pid_team_ok += int((t == home) == (code == H))
    phases = []
    with open(os.path.join(match_dir, mid + "_phases_of_play.csv"), newline="") as h:
        for r in csv.DictReader(h):
            loss = str(r.get("team_possession_loss_in_phase", "")).strip().lower() in (
                "true", "1", "1.0")
            phases.append((int(r["frame_start"]), int(r["frame_end"]), loss))
    return {"in_periods": n_per, "named_share": named / float(n_per),
            "named_with_player_share": with_pid / float(max(named, 1)),
            "player_team_agrees_share": pid_team_ok / float(max(with_pid, 1)),
            "player_without_group": pid_without_group}, phases, np.asarray(group10, np.int8)


def loss_check(phases, fine, n_video):
    """Team changes of the fine view inside each phase (25 Hz span of the 10 Hz phase)."""
    out = {"loss": [0, 0], "no_loss": [0, 0]}     # [phases, phases with a fine team change]
    for a, b, loss in phases:
        va, vb = -((-a * 25) // 10), min((b * 25) // 10, n_video)
        if vb - va < 2:
            continue
        w = fine[va:vb]
        w = w[np.isin(w, (H, A))]
        change = bool(len(w) > 1 and (w[1:] != w[:-1]).any())
        key = "loss" if loss else "no_loss"
        out[key][0] += 1
        out[key][1] += int(change)
    return out


def load(path, key="raw"):
    with np.load(path) as z:
        return z[key].astype(np.int8)


def fmt_row(name, p, extra=""):
    return "| {} | {} | {:.1f} | {:.2f} | {:.2f} | {:.2f} | {:.1f} | {:.1f} | {:.3f} |{}".format(
        name, p["possessions"], p["live_min"], p["per_live_min"], p["under_1s"], p["under_2s"],
        p["median_s"], p["mean_s"], p["d_share"], extra)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--fine", required=True, help="cache_possgroup of skillcorner_fine.py")
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrica-variant", default=None, help="folder of METRICA-S*_variant.npy")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    res = {"skillcorner": {}, "others": {}}
    head = ("| match | possessions | live min | per live min | <1 s | <2 s | median s | "
            "mean s | D share |")
    sep = "|---" * 9 + "|"
    lines = ["# SkillCorner coarse (phases) and fine (possession.group) references", "",
             "Generated by tools/learned_spike/sc_fine_stats.py. 25 fps, graded frames of each "
             "reference; a possession is a maximal H or A run. SkillCorner Open Data (MIT, "
             "SkillCorner).", ""]
    rows_c, rows_f, rows_n = [], [], []
    for game in folds.SC:
        mid = game[3:]
        coarse = load(os.path.join(a.cache, "labels", game + ".npz"))
        fine = load(os.path.join(a.fine, "labels", game + ".npz"))
        runs = pm.valid_runs(coarse)
        assert runs == pm.valid_runs(fine)
        pc, pf = profile(coarse, runs), profile(fine, runs)
        live = np.isin(coarse, (H, A))
        agree = float((coarse[live] == fine[live]).mean())
        field, phases, _ = raw_field(os.path.join(a.raw_root, mid), mid)
        res["skillcorner"][game] = {"coarse": pc, "fine": pf, "live_agreement": agree,
                                    "nesting": nesting(coarse, fine, runs), "field": field,
                                    "loss_flag": loss_check(phases, fine, len(fine))}
        rows_c.append(fmt_row(game, pc))
        rows_f.append(fmt_row(game, pf, " {:.3f} |".format(agree)))
    sc = res["skillcorner"]

    def mean_of(kind, key):
        return float(np.mean([sc[g][kind][key] for g in folds.SC]))

    def sc_mean_row(kind):
        p = {k: mean_of(kind, k) for k in sc[folds.SC[0]][kind]}
        p["possessions"] = int(round(p["possessions"]))
        return fmt_row("**mean of 20**", p)
    lines += ["## Coarse (phases of play, the reference used today)", "", head, sep] + rows_c + [
        sc_mean_row("coarse"), "",
        "## Fine (possession.group, held through empty frames)", "",
        head[:-1] + " live agreement with coarse |", sep + "---|"] + rows_f + [
        sc_mean_row("fine") + " {:.3f} |".format(
            float(np.mean([sc[g]["live_agreement"] for g in folds.SC]))), ""]
    # nesting and loss flags
    lines += ["## How the fine view relates to the coarse one", "",
              "| match | coarse boundaries | fine boundaries | coarse in fine (exact) | "
              "coarse in fine (1 s) | coarse H-A switches | of them in fine (exact) | "
              "fine that are coarse | phases with loss: fine change | phases without loss: "
              "fine change |", "|---" * 10 + "|"]
    for g in folds.SC:
        n, lf = sc[g]["nesting"], sc[g]["loss_flag"]
        lines.append("| {} | {} | {} | {:.3f} | {:.3f} | {} | {:.3f} | {:.3f} | {}/{} | {}/{} |"
                     .format(g, n["coarse_boundaries"], n["fine_boundaries"],
                             n["coarse_in_fine_exact"], n["coarse_in_fine_1s"],
                             n["coarse_ha_switches"], n["coarse_ha_in_fine_exact"],
                             n["fine_that_are_coarse"], lf["loss"][1], lf["loss"][0],
                             lf["no_loss"][1], lf["no_loss"][0]))
    tot = {k: sum(sc[g]["loss_flag"][k][i] for g in folds.SC) for k in ("loss", "no_loss")
           for i in (0,)}
    chg = {k: sum(sc[g]["loss_flag"][k][1] for g in folds.SC) for k in ("loss", "no_loss")}
    lines += ["", "Totals: phases flagged with a loss {} (fine team change inside {}, {:.1%}); "
              "phases without {} (fine team change inside {}, {:.1%}).".format(
                  tot["loss"], chg["loss"], chg["loss"] / float(max(tot["loss"], 1)),
                  tot["no_loss"], chg["no_loss"], chg["no_loss"] / float(max(tot["no_loss"], 1))),
              "", "## The field itself (10 Hz, frames in periods)", "",
              "| match | named share | named with a player_id | player's team = group | "
              "player_id without group |", "|---|---|---|---|---|"]
    for g in folds.SC:
        f = sc[g]["field"]
        lines.append("| {} | {:.3f} | {:.4f} | {:.4f} | {} |".format(
            g, f["named_share"], f["named_with_player_share"], f["player_team_agrees_share"],
            f["player_without_group"]))
    # other providers
    lines += ["", "## Other providers, references as provided (and the Metrica loose-touch "
              "variant)", "", head, sep]
    for g in folds.OPEN:
        ref = load(os.path.join(a.cache, "labels", g + ".npz"))
        runs = pm.valid_runs(ref)
        p = profile(ref, runs)
        res["others"][g] = p
        lines.append(fmt_row(g, p))
        if a.metrica_variant and g.startswith("METRICA"):
            var = np.load(os.path.join(a.metrica_variant, g + "_variant.npy")).astype(np.int8)
            pv = profile(var, runs)
            res["others"][g + " loose-touch"] = pv
            lines.append(fmt_row(g + " loose-touch", pv))
    for prov, keys in (("Metrica", [g for g in folds.OPEN if g.startswith("METRICA")]),
                       ("Metrica loose-touch", [g + " loose-touch" for g in folds.OPEN
                                                if g.startswith("METRICA")]),
                       ("DFL", [g for g in folds.OPEN if g.startswith("DFL")])):
        keys = [k for k in keys if k in res["others"]]
        if not keys:
            continue
        p = {k: float(np.mean([res["others"][g][k] for g in keys])) for k in res["others"][keys[0]]}
        p["possessions"] = int(round(p["possessions"]))
        lines.append(fmt_row("**mean {}**".format(prov), p))
    with open(os.path.join(a.out, "SC_FINE_STATS.md"), "w") as h:
        h.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out, "sc_fine_stats.json"), "w") as h:
        json.dump(res, h, indent=1, sort_keys=True)
        h.write("\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
