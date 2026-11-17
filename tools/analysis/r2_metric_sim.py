"""R2-f: full-match spellF1 simulation under a measured error model (research direction F).

    python tools/analysis/r2_metric_sim.py [--draws 200] [--out DIR]

THIS IS A MODEL, not a measurement. It builds synthetic "truth-like" possession sequences with the
known full-match scale (two halves of 79,467 and 79,390 graded frames, D share swept, ~97 D
segments, ~250 possessions, public-window spell-length shape) and applies an error model read off
the public window (c5_dribble transition alignment from r2_metric_decomp.py). It reports the
simulated spellF1 and its sensitivity to one error component at a time. Two sanity checks:
(a) the same error model applied to the real public GT must land near the observed public score;
(b) a baseline_smooth50 error model must land near the organisers' published full-match 0.2433.

Truth model, per half: alive episodes separated by D segments. Episode lengths are equal shares of
the alive time with lognormal jitter; each episode is filled with alternating possessions of
lognormal length (median and sigma from the 17 public spells, scaled so the match has the target
possession count); the team after a D equals the team before it with probability P_SAME_AFTER_D.

Error model: every GT transition is dropped (missed) or emitted with a latency drawn from a
per-kind mixture; a dropped D onset means the D is missed (the pre-D team is held, then the restart
switches team or merges); a restart may carry the wrong team. Spurious other-team excursions and
spurious D segments are overlaid at a rate per alive minute. The prediction is then the label of
the latest emitted transition, exactly as an event-gated decoder holds its state.

Scores with a fast re-implementation of pm.spell_scores, checked equal to the scorer on the first
draws. Fixed seeds. Never opens any ground-truth file except the public one (via tools/gt.py).
"""
import argparse
import copy
import json
import os
import tempfile
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from gt import public_gt_codes  # noqa: E402
import possession_metrics as pm  # noqa: E402

H, A, D, X = 0, 1, 2, 3
FPS = 25
HALVES = (79467, 79390)          # graded frames per half (11185..90651, 110808..190197)
GAP = 100                        # X frames between the halves in the synthetic array
PUB_LOG_MEDIAN = np.log(200.7)   # geometric mean of the 17 public spell lengths (frames)
PUB_LOG_SIGMA = 1.196            # sd of log length, public spells (ddof=1)
P_SAME_AFTER_D = 0.35            # public: 1 of 4 D segments has the same team either side
D_LOG_SIGMA = 0.9                # D length spread (public 138-511 f; full match has long tails)
D_MIN = 50                       # 2 s: no D under 2 s in the M2 full-match output (RESEARCH M6)
EPISODE_LOG_SIGMA = 0.6
ORDER_GAP = 5                    # frames between two emitted switches when a late one queues
DEFAULT_OUT = os.path.join(tempfile.gettempdir(), "r2_research", "f_metric")

# ---------------------------------------------------------------------------- error models
# Latency mixtures: list of (weight, kind, a, b, p_late). kind "normal": N(a, b).
# kind "mag": |latency| ~ U(a, b), sign + with probability p_late.
C5 = {
    "name": "c5_dribble",
    # 12 public H<->A transitions read order-preserving (each pred switch is the next GT switch,
    # late): dropped 1 (15799, the 90 f A spell has no pred A at all); within +-15: 3 (-13, +9,
    # +11); 16-50: 2 (+36, +45); large: 6 (+106, -112, +150, +155, +225, +336; 5 late, 1 early).
    # The large group is the "no event at the hand-over" class (M3: 11351, 11553, 13785, 15766).
    "ha_miss": 1 / 12.0,
    "ha_lat": [(3 / 11.0, "normal", 2.0, 10.0, 0), (2 / 11.0, "mag", 16, 50, 1.0),
               (6 / 11.0, "mag", 100, 340, 5 / 6.0)],
    # 4 onsets: 0, -10, -9, -2. 4 restarts: 0, -7, -2, +26. None missed, none spurious.
    "on_miss": 0.0, "on_lat": [(1.0, "normal", -5.0, 5.0, 0)],
    "re_lat": [(1.0, "normal", 4.0, 15.0, 0)],
    "re_wrong_team": 0.0,
    # genuinely spurious other-team excursions under the order-preserving reading: 12818-12851
    # (33 f) only, 1 in 4.0 public alive minutes -> 0.25 per alive minute.
    "exc_rate": 0.25, "exc_log_median": np.log(60.0), "exc_log_sigma": 0.9,
    "spD_rate": 0.0, "spD_log_median": np.log(150.0), "spD_log_sigma": 0.8,
}
SMOOTH50 = {
    "name": "baseline_smooth50",
    # 12 H<->A: unpaired 15346, 15365; paired +6, +2 | -38, -47 | +117, +151, +177, -117, -87, -64
    "ha_miss": 2 / 12.0,
    "ha_lat": [(2 / 10.0, "normal", 0.0, 8.0, 0), (2 / 10.0, "mag", 16, 50, 0.0),
               (6 / 10.0, "mag", 60, 180, 0.5)],
    # onsets: +30, +130, two missed; restarts: -60, -123, -62, one missed
    "on_miss": 0.5, "on_lat": [(0.5, "mag", 20, 40, 1.0), (0.5, "mag", 100, 160, 1.0)],
    "re_lat": [(1.0, "mag", 50, 130, 0.0)],
    "re_wrong_team": 0.0,
    # rates calibrated in calibrate_baseline() to the published 594 possessions / 133 D segments
    "exc_rate": 2.5, "exc_log_median": np.log(60.0), "exc_log_sigma": 0.8,
    "spD_rate": 0.8, "spD_log_median": np.log(80.0), "spD_log_sigma": 0.8,
}


def draw_latency(rng, mix):
    w = np.array([m[0] for m in mix])
    k = rng.choice(len(mix), p=w / w.sum())
    _, kind, a, b, p_late = mix[k]
    if kind == "normal":
        return int(round(rng.normal(a, b)))
    mag = rng.uniform(a, b)
    return int(round(mag if rng.rand() < p_late else -mag))


# ---------------------------------------------------------------------------- truth model

def gen_half(rng, n_frames, d_share, n_d, poss_scale):
    """[(start, stop, label)] covering 0..n_frames, first segment a kickoff possession."""
    d_total = d_share * n_frames
    dl = np.exp(rng.normal(0.0, D_LOG_SIGMA, n_d))
    dl = np.maximum(D_MIN, dl / dl.sum() * d_total).astype(int)
    alive_total = n_frames - dl.sum()
    ew = np.exp(rng.normal(0.0, EPISODE_LOG_SIGMA, n_d + 1))
    el = np.maximum(30, ew / ew.sum() * alive_total).astype(int)
    el[-1] = n_frames - dl.sum() - el[:-1].sum()
    if el[-1] < 30:
        el[-1] = 30
    segs = []
    t = 0
    team = int(rng.randint(2))
    for e in range(n_d + 1):
        left = int(el[e])
        first = True
        while left > 0:
            if not first:
                team = 1 - team
            first = False
            n = int(np.exp(rng.normal(PUB_LOG_MEDIAN, PUB_LOG_SIGMA)) * poss_scale)
            n = max(10, n)
            if left - n < 10:
                n = left
            segs.append((t, t + n, team))
            t += n
            left -= n
        if e < n_d:
            segs.append((t, t + int(dl[e]), D))
            t += int(dl[e])
            if rng.rand() >= P_SAME_AFTER_D:
                team = 1 - team
            # the next episode's first possession uses `team` (first=True)
    # trim or extend to n_frames
    out = []
    for s, e, k in segs:
        if s >= n_frames:
            break
        out.append((s, min(e, n_frames), k))
    if out[-1][1] < n_frames:
        s, e, k = out[-1]
        out[-1] = (s, n_frames, k)
    return out


def to_array(halves_segs):
    parts = []
    for i, segs in enumerate(halves_segs):
        if i:
            parts.append(np.full(GAP, X, np.int8))
        arr = np.empty(segs[-1][1], np.int8)
        for s, e, k in segs:
            arr[s:e] = k
        parts.append(arr)
    return np.concatenate(parts)


# ---------------------------------------------------------------------------- error model

def perturb_half(rng, segs, model):
    n = segs[-1][1]
    events = []                       # (frame, order, to_label)
    first = segs[0][2]
    for i in range(1, len(segs)):
        f, prev, cur = segs[i][0], segs[i - 1][2], segs[i][2]
        if cur == D:                                   # onset
            if rng.rand() < model["on_miss"]:
                continue
            lat = draw_latency(rng, model["on_lat"])
            to = D
        elif prev == D:                                # restart
            lat = draw_latency(rng, model["re_lat"])
            to = cur if rng.rand() >= model["re_wrong_team"] else 1 - cur
        else:                                          # H<->A
            if rng.rand() < model["ha_miss"]:
                continue
            lat = draw_latency(rng, model["ha_lat"])
            to = cur
        events.append((f + lat, i, to))
    # Order preserving: a decoder emits its switches in the order of the evidence, so a late
    # switch delays the next one instead of overtaking it (c5: 11351 +150, 11553 +155, 11624 +336
    # is one late chain). Minimum gap ORDER_GAP frames.
    last_f = 0
    for k in range(len(events)):
        f = max(events[k][0], last_f + ORDER_GAP, 1)
        events[k] = (f, events[k][1], events[k][2])
        last_f = f
    events = [e for e in events if e[0] < n]
    pred = np.empty(n, np.int8)
    cur, last = first, 0
    for f, _, to in events:
        pred[last:f] = cur
        cur, last = to, f
    pred[last:] = cur
    alive_min = float(np.sum(pred != D)) / FPS / 60.0
    # spurious other-team excursions, clipped at the next predicted D
    for _ in range(rng.poisson(model["exc_rate"] * alive_min)):
        s = int(rng.randint(0, n))
        if pred[s] == D:
            continue
        L = int(np.clip(np.exp(rng.normal(model["exc_log_median"], model["exc_log_sigma"])),
                        5, 400))
        w = pred[s:s + L]
        stop = np.flatnonzero(w == D)
        L = int(stop[0]) if len(stop) else len(w)
        pred[s:s + L] = 1 - pred[s]
    # spurious D segments (the team after is the team before: a split)
    for _ in range(rng.poisson(model["spD_rate"] * alive_min)):
        s = int(rng.randint(0, n))
        if pred[s] == D:
            continue
        L = int(np.clip(np.exp(rng.normal(model["spD_log_median"], model["spD_log_sigma"])),
                        D_MIN // 2, 1500))
        pred[s:s + L] = D
    return pred


# ---------------------------------------------------------------------------- fast scorer

def spells_of(codes, runs):
    return pm.possession_spells(codes, runs)


def fast_spell_f1(gt_spells, pred_spells):
    """Same result as pm.spell_scores(...)['mean_f1'] (checked), O(n log n) overlap search."""
    if not gt_spells or not pred_spells:
        return 0.0, len(pred_spells)
    ps = np.array([p[0] for p in pred_spells])
    pe = np.array([p[1] for p in pred_spells])
    out = []
    for i, (gs, ge, gk) in enumerate(gt_spells):
        lo = int(np.searchsorted(pe, gs, side="right"))
        hi = int(np.searchsorted(ps, ge, side="left"))
        for j in range(lo, hi):
            if pred_spells[j][2] != gk:
                continue
            inter = min(ge, pe[j]) - max(gs, ps[j])
            if inter <= 0:
                continue
            union = max(ge, pe[j]) - min(gs, ps[j])
            out.append((inter / float(union), i, j))
    out.sort(key=lambda it: (-it[0], it[1], it[2]))
    ng, npred = len(gt_spells), len(pred_spells)
    f1s = []
    for t in pm.SPELL_IOUS:
        tp = int(pm.matched_gt_flags(out, ng, t).sum())
        f1s.append(2.0 * tp / (ng + npred) if tp else 0.0)
    return float(np.mean(f1s)), npred


def simulate(model, n_draws, seed, d_share=0.5, n_d=97, target_poss=250, check=False):
    rng = np.random.RandomState(seed)
    scale = poss_scale_for(d_share, n_d, target_poss)
    scores, counts, gtc, dsegs, dshare = [], [], [], [], []
    for draw in range(n_draws):
        truth = [gen_half(rng, n, d_share, n_d // 2 + (k == 0 and n_d % 2), scale)
                 for k, n in enumerate(HALVES)]
        preds = [perturb_half(rng, segs, model) for segs in truth]
        gt = to_array(truth)
        pred = np.concatenate([preds[0], np.full(GAP, X, np.int8), preds[1]])
        runs = pm.valid_runs(gt)
        gsp, psp = spells_of(gt, runs), spells_of(pred, runs)
        s, npred = fast_spell_f1(gsp, psp)
        if check and draw < 2:
            ref = pm.spell_scores(gt, pred, runs)["mean_f1"]
            assert abs(ref - s) < 1e-12, (ref, s)
        scores.append(s)
        counts.append(npred)
        gtc.append(len(gsp))
        segs = pm.segments(pred, runs)
        dsegs.append(sum(1 for a in segs if a[2] == D))
        dshare.append(float(np.mean(pred[gt != X] == D)))
    sc = np.array(scores)
    return {"mean": float(sc.mean()), "sd": float(sc.std(ddof=1)),
            "se": float(sc.std(ddof=1) / np.sqrt(len(sc))),
            "p5": float(np.percentile(sc, 5)), "p95": float(np.percentile(sc, 95)),
            "pred_poss": float(np.mean(counts)), "gt_poss": float(np.mean(gtc)),
            "pred_dsegs": float(np.mean(dsegs)), "pred_dshare": float(np.mean(dshare)),
            "scores": sc}


_SCALE_CACHE = {}


def poss_scale_for(d_share, n_d, target):
    """Possession-length scale that gives about `target` GT possessions (searched, fixed seed)."""
    key = (d_share, n_d, target)
    if key in _SCALE_CACHE:
        return _SCALE_CACHE[key]
    lo, hi = 0.3, 4.0
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        rng = np.random.RandomState(12345)
        c = []
        for _ in range(6):
            c.append(sum(sum(1 for s in gen_half(rng, n, d_share, n_d // 2, mid) if s[2] != D)
                         for n in HALVES))
        if np.mean(c) > target:
            lo = mid
        else:
            hi = mid
    _SCALE_CACHE[key] = 0.5 * (lo + hi)
    return _SCALE_CACHE[key]


def public_check(model, n_draws, seed):
    """Apply the error model to the real public GT, many draws."""
    gt = public_gt_codes()
    runs = pm.valid_runs(gt)
    s0, s1 = runs[0]
    segs = [(a - s0, b - s0, k) for a, b, k in pm.segments(gt, runs)]
    rng = np.random.RandomState(seed)
    sc, cnt = [], []
    gsp = [(a, b, k) for a, b, k in segs if k != D]
    for _ in range(n_draws):
        pred = perturb_half(rng, segs, model)
        r = [(0, len(pred))]
        psp = pm.possession_spells(pred, r)
        s, n = fast_spell_f1(gsp, psp)
        sc.append(s)
        cnt.append(n)
    sc = np.array(sc)
    return {"mean": float(sc.mean()), "sd": float(sc.std(ddof=1)),
            "p5": float(np.percentile(sc, 5)), "p95": float(np.percentile(sc, 95)),
            "pred_poss": float(np.mean(cnt))}


def variants(base):
    """One error component changed at a time (the task list plus a few extra)."""
    out = []

    def mk(name, fn):
        m = copy.deepcopy(base)
        fn(m)
        out.append((name, m))

    mk("base", lambda m: None)
    mk("halve H<->A miss rate", lambda m: m.update(ha_miss=m["ha_miss"] / 2))
    mk("H<->A miss rate 0", lambda m: m.update(ha_miss=0.0))
    mk("halve spurious excursion rate", lambda m: m.update(exc_rate=m["exc_rate"] / 2))
    mk("spurious excursions 0", lambda m: m.update(exc_rate=0.0))
    mk("double spurious excursion rate", lambda m: m.update(exc_rate=m["exc_rate"] * 2))

    def late_to_small(m):
        mix = m["ha_lat"]
        w_large = mix[2][0]
        m["ha_lat"] = [(mix[0][0] + w_large / 2, ) + mix[0][1:], mix[1],
                       (w_large / 2, ) + mix[2][1:]]
    mk("halve the large-latency (>100 f) H<->A share (moved to +-10)", late_to_small)

    def no_large(m):
        mix = m["ha_lat"]
        m["ha_lat"] = [(mix[0][0] + mix[2][0], ) + mix[0][1:], mix[1]]
    mk("no large-latency H<->A (all moved to +-10)", no_large)

    def medium_to_small(m):
        mix = m["ha_lat"]
        m["ha_lat"] = [(mix[0][0] + mix[1][0], ) + mix[0][1:], mix[2]]
    mk("no medium-latency (16-50 f) H<->A (moved to +-10)", medium_to_small)

    def jitter_half(m):
        mix = m["ha_lat"]
        m["ha_lat"] = [mix[0][:3] + (mix[0][3] / 2, 0)] + mix[1:]
        m["on_lat"] = [(1.0, "normal", m["on_lat"][0][2], m["on_lat"][0][3] / 2, 0)]
        m["re_lat"] = [(1.0, "normal", m["re_lat"][0][2], m["re_lat"][0][3] / 2, 0)]
    mk("halve small-latency jitter sigma (H<->A, onset, restart)", jitter_half)
    mk("restarts +10 frames", lambda m: m.update(
        re_lat=[(1.0, "normal", m["re_lat"][0][2] + 10, m["re_lat"][0][3], 0)]))
    mk("restarts +25 frames", lambda m: m.update(
        re_lat=[(1.0, "normal", m["re_lat"][0][2] + 25, m["re_lat"][0][3], 0)]))
    mk("onsets +25 frames late", lambda m: m.update(
        on_lat=[(1.0, "normal", m["on_lat"][0][2] + 25, m["on_lat"][0][3], 0)]))
    for k in (50, 125, 250):
        mk("restarts +{} frames (D wider at the restart end)".format(k), lambda m, k=k: m.update(
            re_lat=[(1.0, "normal", m["re_lat"][0][2] + k, m["re_lat"][0][3], 0)]))
    mk("onsets 125 frames early (D wider at the onset end)", lambda m: m.update(
        on_lat=[(1.0, "normal", m["on_lat"][0][2] - 125, m["on_lat"][0][3], 0)]))
    mk("restart sigma 15 -> 40", lambda m: m.update(
        re_lat=[(1.0, "normal", m["re_lat"][0][2], 40.0, 0)]))
    mk("miss 10% of D segments", lambda m: m.update(on_miss=m["on_miss"] + 0.10))
    mk("miss 25% of D segments", lambda m: m.update(on_miss=m["on_miss"] + 0.25))
    mk("add spurious D = 10% of n_D", lambda m: m.update(spD_rate=m["spD_rate"] + SPD_10PCT))
    mk("add spurious D = 25% of n_D", lambda m: m.update(spD_rate=m["spD_rate"] + 2.5 * SPD_10PCT))
    mk("wrong restart team 10%", lambda m: m.update(re_wrong_team=0.10))
    return out


# 10 % of 97 D segments = 9.7 spurious D per match, over ~53-79 alive minutes; per alive minute at
# D share 50 % (52.9 alive min) this is 0.183. Kept as one rate for all D shares.
SPD_10PCT = 9.7 / 52.9


def calibrate_baseline(model, d_share, n_draws=20):
    """Tune exc_rate and spD_rate so the smooth50 model gives ~594 possessions, ~133 D segments
    at the given truth D share (two-pass coordinate search, fixed seed)."""
    m = copy.deepcopy(model)
    for _ in range(3):
        for key, target, stat in (("exc_rate", 594.0, "pred_poss"), ("spD_rate", 133.0,
                                                                      "pred_dsegs")):
            lo, hi = 0.0, 8.0
            for _ in range(10):
                mid = 0.5 * (lo + hi)
                m[key] = mid
                r = simulate(m, n_draws, 777, d_share=d_share)
                if r[stat] > target:
                    hi = mid
                else:
                    lo = mid
            m[key] = 0.5 * (lo + hi)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    lines = ["# r2_metric_sim output (draws per cell: {}, seed 2027)\n".format(args.draws)]
    res = {}

    lines.append("## sanity (a): error model applied to the real public GT (2000 draws)\n")
    lines.append("| model | sim mean | sd | p5 | p95 | pred spells | observed public |")
    lines.append("|---|---|---|---|---|---|---|")
    for model, obs in ((C5, 0.5529), (SMOOTH50, 0.3256)):
        r = public_check(model, 2000, 11)
        res["public_" + model["name"]] = r
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.1f} | {} |".format(
            model["name"], r["mean"], r["sd"], r["p5"], r["p95"], r["pred_poss"], obs))
    lines.append("")

    lines.append("## sanity (b): baseline_smooth50 error model on the synthetic full match\n")
    lines.append("exc_rate and spD_rate calibrated to the published 594 possessions and 133 D"
                 " segments at each truth D share; published full-match spellF1 0.2433.\n")
    lines.append("| truth D share | exc_rate /alive min | spD_rate /alive min | sim spellF1 | sd |"
                 " pred poss | pred D segs | pred D share (published 32.3%) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for ds in (0.35, 0.42, 0.50):
        m = calibrate_baseline(SMOOTH50, ds)
        r = simulate(m, args.draws // 2, 2027, d_share=ds, check=True)
        res["smooth50_full_{}".format(ds)] = {k: v for k, v in r.items() if k != "scores"}
        lines.append("| {:.2f} | {:.2f} | {:.2f} | {:.4f} | {:.4f} | {:.0f} | {:.0f} | {:.1%} |"
                     .format(ds, m["exc_rate"], m["spD_rate"], r["mean"], r["sd"],
                             r["pred_poss"], r["pred_dsegs"], r["pred_dshare"]))
        m_pub = copy.deepcopy(m)
        rp = public_check(m_pub, 1000, 12)
        lines.append("|  | (same rates on the public GT: {:.4f}, {:.1f} spells) | | | | | | |"
                     .format(rp["mean"], rp["pred_poss"]))
    lines.append("")

    lines.append("## c5 error model on the synthetic full match: sensitivity, one component at a"
                 " time\n")
    lines.append("Same seed for every variant (common random numbers), so differences are paired;"
                 " the paired SE of a difference is given.\n")
    for ds in (0.35, 0.42, 0.50):
        lines.append("### truth D share {:.0%}, n_D 97, ~250 GT possessions\n".format(ds))
        lines.append("| variant | spellF1 | Δ vs base | paired SE of Δ | pred poss | GT poss |"
                     " pred D segs | pred D share |")
        lines.append("|---|---|---|---|---|---|---|---|")
        base_scores = None
        for name, m in variants(C5):
            r = simulate(m, args.draws, 2027, d_share=ds, check=(name == "base"))
            if base_scores is None:
                base_scores = r["scores"]
                base_mean = r["mean"]
            diff = r["scores"] - base_scores
            se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if name != "base" else 0.0
            res["c5_{}_{}".format(ds, name)] = {k: v for k, v in r.items() if k != "scores"}
            lines.append("| {} | {:.4f} | {:+.4f} | {:.4f} | {:.0f} | {:.0f} | {:.0f} | {:.1%} |"
                         .format(name, r["mean"], r["mean"] - base_mean, se, r["pred_poss"],
                                 r["gt_poss"], r["pred_dsegs"], r["pred_dshare"]))
        lines.append("")

    lines.append("## excess-D scenarios: engine D share ~50% while the truth is 40%\n")
    lines.append("The c5 engine writes 50.5% D. If the truth is 40%, the extra ~10% of graded"
                 " frames (~16,000 f) must be spurious D segments or wider D. Each row adds one"
                 " explanation to the c5 base model at truth D share 40%.\n")
    lines.append("| scenario | spellF1 | Δ vs c5 base at 40% | pred D share | pred D segs |"
                 " pred poss |")
    lines.append("|---|---|---|---|---|---|")
    base40 = simulate(C5, args.draws, 2027, d_share=0.40)
    scen = [("c5 base (truth 40%)", copy.deepcopy(C5))]
    m = copy.deepcopy(C5)
    m["re_lat"] = [(1.0, "normal", 4.0 + 165, 60.0, 0)]
    scen.append(("restarts late by ~165 f (6.6 s) on average, sd 60", m))
    m = copy.deepcopy(C5)
    m["re_lat"] = [(1.0, "normal", 4.0 + 82, 40.0, 0)]
    m["on_lat"] = [(1.0, "normal", -5.0 - 82, 40.0, 0)]
    scen.append(("both D ends ~82 f wider, sd 40", m))
    m = copy.deepcopy(C5)
    m["spD_rate"] = 0.8
    m["spD_log_median"] = np.log(750.0)
    scen.append(("spurious D segments, median 30 s, 0.8 per alive min", m))
    for nm, m in scen:
        r = simulate(m, args.draws, 2027, d_share=0.40)
        lines.append("| {} | {:.4f} | {:+.4f} | {:.1%} | {:.0f} | {:.0f} |".format(
            nm, r["mean"], r["mean"] - base40["mean"], r["pred_dshare"], r["pred_dsegs"],
            r["pred_poss"]))
    lines.append("")

    lines.append("## c5 base model: n_D and possession-count sweep (D share 50%)\n")
    lines.append("| n_D | target GT poss | spellF1 | sd | pred poss |\n|---|---|---|---|---|")
    for n_d, tp in ((80, 250), (97, 225), (97, 270), (120, 250), (140, 250)):
        r = simulate(C5, args.draws // 2, 2027, d_share=0.5, n_d=n_d, target_poss=tp)
        lines.append("| {} | {} | {:.4f} | {:.4f} | {:.0f} |".format(n_d, tp, r["mean"], r["sd"],
                                                                      r["pred_poss"]))
    lines.append("")
    text = "\n".join(lines)
    with open(os.path.join(args.out, "sim.md"), "w") as fh:
        fh.write(text)
    with open(os.path.join(args.out, "sim.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(text)


if __name__ == "__main__":
    main()
