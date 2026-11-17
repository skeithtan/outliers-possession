"""Round 8 arm N: the four-test table and the verdict of protocol R8 Section 4 (Arm N rule). Not part of the submission.

    python -m tools.learned_spike.r8_scripts.n_report --score DIR --pff DIR --seeds 01 \
        --base E2 --cands K200 K400 [--repro DIR_E2_4SEEDS --repro-pff DIR] > TABLES.md

Reads n_score.json (n_score.py) and n_pff.json (n_pff.py). Tests: (1) T-Moof, (2) T-D,
(3) T-S, (4) PFF 2 s; gains over c23 at (12, 12) with the member at 0.25. Rule (registered
16 Nov 2026 00:40): four-test mean at least 0.005 above the baseline's; no test lower by more
than 0.005; T-D and PFF not lower by more than 0.003; share of D inside SkillCorner stoppages
with the ball in view not lower than E2's 0.573 by more than 0.01 (also shown against the
same-seed baseline). Python 3.9.
"""

import argparse
import json
import os
import sys

TESTS = (("T-Moof", "(1) Metrica out of fold"), ("T-D", "(2) left-out DFL"),
         ("T-S", "(3) left-out SkillCorner, possession groups"), ("PFF", "(4) open PFF, 2 s"))
E2_D_IN_VIEW = 0.573


def load(score_dir, pff_dir, name, seeds):
    s = json.load(open(os.path.join(score_dir, "n_score.json")))["systems"][name]
    p = json.load(open(os.path.join(pff_dir, "n_pff.json")))["systems"]["{}_s{}".format(
        name, seeds)]
    out = {}
    for t in ("T-Moof", "T-D", "T-S", "T-Sp"):
        if t in s["summary"]:
            out[t] = {"summary": s["summary"][t],
                      "per": {g: r["delta"] for g, r in s["tests"][t].items()}}
    ps = p["summary"]
    out["PFF"] = {"summary": dict(ps, gain=ps["gain_2s"], up=ps["up_2s"], n=ps["n"],
                                  worst=ps["worst_2s"]),
                  "per": {g: r["ctl2"]["delta"] for g, r in p["per_match"].items()}}
    return out


def f4(v):
    return "-" if v is None else "{:+.4f}".format(v)


def f3(v):
    return "-" if v is None else "{:.3f}".format(v)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--score", required=True)
    p.add_argument("--pff", required=True)
    p.add_argument("--seeds", default="01")
    p.add_argument("--base", default="E2")
    p.add_argument("--cands", nargs="+", default=["K200", "K400"])
    args = p.parse_args(argv)
    base = load(args.score, args.pff, args.base, args.seeds)
    systems = {args.base: base}
    for c in args.cands:
        systems[c] = load(args.score, args.pff, c, args.seeds)
    L = ["## Four tests, seeds {} (gain over c23 at (12, 12), member F5T x 0.25)".format(
        ",".join(args.seeds)), "",
        "| system | " + " | ".join(n for _, n in TESTS) + " | four-test mean |",
        "|---|" + "---|" * (len(TESTS) + 1)]
    means = {}
    for name, S in systems.items():
        g = [S[t]["summary"]["gain"] for t, _ in TESTS]
        means[name] = sum(g) / len(g)
        L.append("| {} | {} | {} |".format(name, " | ".join(
            "{} ({}/{} up, worst {})".format(f4(S[t]["summary"]["gain"]), S[t]["summary"]["up"],
                                            S[t]["summary"]["n"], f4(S[t]["summary"]["worst"]))
            for t, _ in TESTS), f4(means[name])))
    L += ["", "## Against {} (same seeds): difference, matches better / worse".format(args.base),
          "", "| system | " + " | ".join(n for _, n in TESTS) + " | mean |",
          "|---|" + "---|" * (len(TESTS) + 1)]
    verdicts = {}
    for name in args.cands:
        S = systems[name]
        cells, diffs = [], {}
        for t, _ in TESTS:
            d = S[t]["summary"]["gain"] - base[t]["summary"]["gain"]
            diffs[t] = d
            better = sum(S[t]["per"][g] > base[t]["per"][g] + 1e-12 for g in S[t]["per"])
            worse = sum(S[t]["per"][g] < base[t]["per"][g] - 1e-12 for g in S[t]["per"])
            cells.append("{} ({} / {})".format(f4(d), better, worse))
        dm = means[name] - means[args.base]
        L.append("| {} | {} | {} |".format(name, " | ".join(cells), f4(dm)))
        dv = S["T-S"]["summary"]["D_in_view_net"]
        checks = {
            "mean +0.005": dm >= 0.005,
            "no test below -0.005": all(d >= -0.005 for d in diffs.values()),
            "DFL not below -0.003": diffs["T-D"] >= -0.003,
            "PFF not below -0.003": diffs["PFF"] >= -0.003,
            "D in view >= 0.573 - 0.01": dv >= E2_D_IN_VIEW - 0.01}
        verdicts[name] = (all(checks.values()), checks, dv)
    L += ["", "## Guards and verdict (protocol R8 Section 4, Arm N rule)", "",
          "| system | " + " | ".join(verdicts[args.cands[0]][1]) + " | D in view (base) | "
          "verdict |", "|---|" + "---|" * (len(verdicts[args.cands[0]][1]) + 2)]
    bv = base["T-S"]["summary"]["D_in_view_net"]
    for name, (ok, checks, dv) in verdicts.items():
        L.append("| {} | {} | {} ({}) | **{}** |".format(
            name, " | ".join("yes" if v else "NO" for v in checks.values()), f3(dv), f3(bv),
            "passes" if ok else "fails"))
    L += ["", "## Other measures", "",
          "| system | test | merge (c23) | split (c23) | possessions / ref (c23) | D share | "
          "seed disagreement pairwise / any |", "|---|---|---|---|---|---|---|"]
    for name, S in systems.items():
        for t in ("T-Moof", "T-D", "T-S", "T-Sp"):
            if t not in S:
                continue
            x = S[t]["summary"]
            L.append("| {} | {} | {} ({}) | {} ({}) | {:.1f} / {:.1f} ({:.1f}) | {} | {} / {} |"
                     .format(name, t, f3(x["merge"]), f3(x["merge_c23"]), f3(x["split"]),
                             f3(x["split_c23"]), x["poss"], x["n_ref"], x["poss_c23"],
                             f3(x["d_share"]), f3(x.get("seed_pairwise")),
                             f3(x.get("seed_any"))))
        x = S["PFF"]["summary"]
        L.append("| {} | PFF 2 s | {} | {} | {:.1f} / {:.1f} | - | {} / {} |".format(
            name, f3(x["merge_2s"]), f3(x["split_2s"]), x["poss_2s"], x["n_ref_2s"],
            f3(x.get("seed_pairwise")), f3(x.get("seed_any"))))
    L += ["", "SkillCorner stoppages >= 2 s, share labelled D (ball in view / hidden; c23 in "
          "brackets):", "", "| system | in view | hidden | D segments per 90 |", "|---|---|---|---|"]
    for name, S in systems.items():
        x = S["T-S"]["summary"]
        L.append("| {} | {} ({}) | {} ({}) | {:.1f} ({:.1f}) |".format(
            name, f3(x["D_in_view_net"]), f3(x["D_in_view_c23"]), f3(x["D_hidden_net"]),
            f3(x["D_hidden_c23"]), x["d90_net"], x["d90_c23"]))
    L += ["", "PFF as provided: " + ", ".join("{} {}".format(
        n, f4(S["PFF"]["summary"]["gain_provided"])) for n, S in systems.items()), ""]
    L += ["## Per match (gain over c23)", ""]
    for t, label in TESTS:
        L += ["### " + label, "", "| match | " + " | ".join(systems) + " |",
              "|---|" + "---|" * len(systems)]
        for g in sorted(base[t]["per"]):
            L.append("| {} | {} |".format(g, " | ".join(f4(S[t]["per"][g])
                                                         for S in systems.values())))
        L.append("")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
