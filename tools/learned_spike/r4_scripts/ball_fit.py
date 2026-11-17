"""Fit of R4_AUG_BALL.md Section 2.2: python ball_fit.py ball_stats.json fit.json (from r4_ball_measure)."""
import json, sys, numpy as np
d = json.load(open(sys.argv[1]))
MD = [g for g in d if not g.startswith("SC-")]; SC = sorted(g for g in d if g.startswith("SC-"))
def inv(xs, ys, y):   # ys decreasing in xs
    ys = np.asarray(ys); xs = np.asarray(xs)
    # monotone: take first crossing
    for i in range(len(xs)-1):
        if (ys[i]-y)*(ys[i+1]-y) <= 0 and ys[i] != ys[i+1]:
            return float(xs[i] + (y-ys[i])*(xs[i+1]-xs[i])/(ys[i+1]-ys[i]))
    return float("nan")
print("ratio(sigma) per M/D:")
for g in MD:
    print(g, [round(d[g]["sigma"][s]["ratio"],3) for s in sorted(d[g]["sigma"],key=float)], "touch:", [round(d[g]["sigma"][s]["touch_d"],2) for s in sorted(d[g]["sigma"],key=float)])
print("sigmas", sorted(d[MD[0]]["sigma"],key=float))
for g in MD:
    GS = sorted({float(k.split("|")[0]) for k in d[g]["grid"]}); GA = sorted({float(k.split("|")[1]) for k in d[g]["grid"]})
    print(g, "touch grid rows sigma", GS, "cols A", GA); [print("   ", s_, [round(d[g]["grid"]["{}|{}".format(s_, a)],2) for a in GA]) for s_ in GS]
print("resampled ratio:")
for g in MD:
    print(g, [round(d[g]["resampled"][s]["ratio"],3) for s in sorted(d[g]["resampled"],key=float)])
res = {}
print("\n| SC match | ratio | touch_d m | near<=1m | sigma eq. (median over 8, min-max) | sigma on 10Hz-resampled (median) | A eq. m at fitted sigma (median, min-max) |")
for sc in SC:
    r = d[sc]["raw"]["ratio"]; td = d[sc]["raw"]["touch_d"]
    sig = []; sigr = []; A = []
    for g in MD:
        ks = sorted(d[g]["sigma"], key=float); ss = [float(s) for s in ks]; ys = [d[g]["sigma"][s]["ratio"] for s in ks]
        s_eq = inv(ss, ys, r); sig.append(s_eq)
        kr = sorted(d[g]["resampled"], key=float); ssr = [float(s) for s in kr]; ysr = [d[g]["resampled"][s]["ratio"] for s in kr]
        sigr.append(inv(ssr, ysr, r))
        # touch_d grid: interpolate in sigma at each A, then invert in A
        GS = sorted({float(k.split("|")[0]) for k in d[g]["grid"]}); GA = sorted({float(k.split("|")[1]) for k in d[g]["grid"]})
        tdA = []
        for a in GA:
            col = [d[g]["grid"]["{}|{}".format(s, a)] for s in GS]
            tdA.append(float(np.interp(s_eq, GS, col)))
        A.append(inv(GA, [-t for t in tdA], -td))
    res[sc] = dict(ratio=r, touch_d=td, sigma=sig, sigma_resampled=sigr, A=A)
    f = lambda v: "{:.2f} ({:.2f}-{:.2f})".format(np.nanmedian(v), np.nanmin(v), np.nanmax(v))
    print("| {} | {:.3f} | {:.2f} | {:.2f} | {} | {} | {} |".format(sc, r, td, d[sc]["raw"]["near"], f(sig), f(sigr), f(A)))
allsig = np.array([x for v in res.values() for x in v["sigma"]]); allA = np.array([x for v in res.values() for x in v["A"]])
print("all sigma: min %.2f p50 %.2f max %.2f" % (np.nanmin(allsig), np.nanmedian(allsig), np.nanmax(allsig)))
print("per-SC median sigma range: %.2f - %.2f" % (min(np.nanmedian(v["sigma"]) for v in res.values()), max(np.nanmedian(v["sigma"]) for v in res.values())))
for prov in ("METRICA", "DFL"):
    idx = [i for i, g in enumerate(MD) if g.startswith(prov)]
    s = np.array([[v["sigma"][i] for i in idx] for v in res.values()]); a = np.array([[v["A"][i] for i in idx] for v in res.values()])
    print(prov, "sigma %.2f-%.2f (p50 %.2f)  A %.2f-%.2f (p50 %.2f)" % (np.nanmin(s), np.nanmax(s), np.nanmedian(s), np.nanmin(a), np.nanmax(a), np.nanmedian(a)))
print("all A: min %.2f p50 %.2f max %.2f nan %d" % (np.nanmin(allA), np.nanmedian(allA), np.nanmax(allA), np.isnan(allA).sum()))
json.dump(res, open(sys.argv[2], "w"), indent=1)
