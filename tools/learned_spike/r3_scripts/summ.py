import json, sys, numpy as np
d = json.load(open(sys.argv[1]))["games"]
games = list(d)
names = list(d[games[0]]["variants"])
def prov(g): return g.split("-")[0]
print("ref counts raw / ctl2 / median raw s / median ctl2 s")
for g in games:
    r=d[g]; print(g, r["ref_raw"]["n"], r["ref_ctl2"]["n"], round(r["ref_raw"]["median_s"],2), round(r["ref_ctl2"]["median_s"],2), "fold cost", r["fold_cost"], "nested", r["nested_pair"], "mbr", r["mbr_pick"], "sampleN", round(r["sample_n_mean"]))
print()
print("%-22s %s" % ("variant", "  ".join("%-34s" % p for p in ["METRICA F1raw(dC22) ratio med F1ctl2", "DFL F1raw(dC22) ratio med F1ctl2"])))
for n in names:
    row=[]
    for p in ["METRICA","DFL"]:
        gs=[g for g in games if prov(g)==p]
        f=np.mean([d[g]["variants"][n]["spellF1_raw"] for g in gs])
        c=np.mean([d[g]["variants"]["c22"]["spellF1_raw"] for g in gs])
        wins=sum(d[g]["variants"][n]["spellF1_raw"]>d[g]["variants"]["c22"]["spellF1_raw"] for g in gs)
        ratio=np.median([d[g]["variants"][n]["count_ratio"] for g in gs])
        med=np.median([d[g]["variants"][n]["median_s"] for g in gs])
        f2=np.mean([d[g]["variants"][n]["spellF1_ctl2"] for g in gs])
        row.append("%.4f(%+.3f,%d/%d) %.2f %5.2f %.4f" % (f, f-c, wins, len(gs), ratio, med, f2))
    print("%-22s %s" % (n, "  ".join("%-34s" % x for x in row)))
print()
for g in games:
    print(g, {n: (v["spellF1_raw"], v["n"]) for n,v in d[g]["variants"].items()})
print()
for g in games: print(g, "MBR exp", d[g]["mbr_expected"]); print(g, "MBR true", d[g]["mbr_true"])
