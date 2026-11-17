"""Round 5: step 0's breakdown (part 1) for the round 5 model, inside the label-free step. Not part of the submission.

    (engine venv) python -m tools.learned_spike.r5_step0 --run RUN --work W --feature-cache DIR \
        [--engine-cache ... --sc-engine-cache ... --ref-cache ...] [--as-match GAME]

LEARNED_STAGE_PROTOCOL_R5.md Section 4 (G7): "In the same run, recorded: ... step 0's breakdown
for the round 5 model". The definitions are those of step 0, part 1
(docs/plan/research/learned-r5/STEP0_PLAN.md, result in docs/plan/research/learned-r5/step0/),
ported from its scripts. It reads only what the label-free step of r3_eval.py already wrote and
computes nothing on the network:

- the probability dumps W/probs/final/<match>/{seed,copy}_<i>.npz of the label-free match and
  W/probs/fold/<SC>/... of the 20 SkillCorner matches (r3_eval agreement and bar);
- the tracking: the label-free match's raw ATD (the competition match through engine.data, or
  `--as-match` a development match through the external views, as r3_eval's Match), the
  SkillCorner matches through evaluate.ext_loader;
- the 5 fps features for the band condition: the label-free match through
  bar_report.bar_features() in the version of --feature-cache, SkillCorner from --feature-cache
  feats/raw; the training band of the 28 external matches from --feature-cache.

Conditions per graded frame (step 0's definitions): dup (a team with more than 11 visible
tracks), frag (a tracklet start or end, not at a block edge, within +-2 s), few (a team with
fewer than 9), band (a team-spread feature of the bin outside the 1 to 99 % training band:
H_sx H_sy A_sx A_sy for stage50-v2 and stage50-v2s, H_iqrx H_iqry A_iqrx A_iqry for stage-v3), ball_missing,
ball_still (ball visible and within 1 m of itself over +-1 s). Track defects = dup | frag | few |
band. The four exclusive cells: track_only, still_only, both, neither. For each cell and
condition: the share of graded frames, and the seed and copy disagreement inside (any member's
arg-max differs from the first's, on the upsampled 5 fps probabilities, as
bar_report.prob_group). Contributions to the excess (label-free match minus SkillCorner mean)
as step 0's part1_aggregate.

Writes W/step0/part1.json and W/step0/PART1.md. Recorded only; it decides nothing. Python 3.9.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from engine import data  # noqa: E402
from tools.learned_spike import folds  # noqa: E402

SPREAD = {"stage50-v2": ("H_sx", "H_sy", "A_sx", "A_sy"),
          "stage50-v2s": ("H_sx", "H_sy", "A_sx", "A_sy"),
          "stage50-v2t": ("H_sx", "H_sy", "A_sx", "A_sy"),
          "stage-v3": ("H_iqrx", "H_iqry", "A_iqrx", "A_iqry")}
MAX_TEAM, MIN_TEAM = 11, 9
FRAG_S = 2.0
STILL_M = 1.0
STILL_WIN_S = 2.0
BIN = 5
CONDS = ("dup", "frag", "few", "band", "ball_missing", "ball_still")
CELLS = ("track_only", "still_only", "both", "neither")
ROWS = CELLS + ("track_any",) + CONDS + ("all",)


def graded_mask(game):
    n = data.frame_count(game)
    m = np.zeros(n, dtype=bool)
    for _, a, b in data.graded_blocks(game):
        m[a:b + 1] = True
    return m


def _window_any(event, half):
    c = np.concatenate(([0], np.cumsum(event.astype(np.int64))))
    n = event.size
    idx = np.arange(n)
    lo = np.clip(idx - half, 0, n)
    hi = np.clip(idx + half + 1, 0, n)
    return (c[hi] - c[lo]) > 0


def conditions(tac, game, feats, band, version):
    """{condition: bool (N,)} (module docstring; step 0's common.conditions)."""
    n = data.frame_count(game)
    fps = data.fps(game)
    length, width = data.pitch_m(game)
    px = np.asarray(tac["px"], dtype=np.float64)[:n]
    py = np.asarray(tac["py"], dtype=np.float64)[:n]
    vis = np.isfinite(px) & np.isfinite(py)
    team = np.asarray(tac["team_of_slot"])
    counts = [vis[:, team == t].sum(axis=1) for t in (0, 1)]
    dup = (counts[0] > MAX_TEAM) | (counts[1] > MAX_TEAM)
    few = (counts[0] < MIN_TEAM) | (counts[1] < MIN_TEAM)
    event = np.zeros(n, dtype=bool)
    for _, a, b in data.graded_blocks(game):
        v = vis[a:b + 1]
        pad = np.zeros((1, v.shape[1]), dtype=bool)
        d = np.diff(np.vstack([pad, v, pad]).astype(np.int8), axis=0)
        starts = np.argwhere(d == 1)[:, 0]
        stops = np.argwhere(d == -1)[:, 0]
        L = v.shape[0]
        event[a + starts[starts > 0]] = True
        event[a + stops[stops < L] - 1] = True
    frag = _window_any(event, int(round(FRAG_S * fps)))
    bx = np.asarray(tac["bx"], dtype=np.float64)[:n] * length
    by = np.asarray(tac["by"], dtype=np.float64)[:n] * width
    bvis = np.isfinite(bx) & np.isfinite(by)
    half = int(round(STILL_WIN_S * fps / 2.0))
    maxd = np.zeros(n)
    for off in range(-half, half + 1):
        if off == 0:
            continue
        j = np.arange(n) + off
        ok = (j >= 0) & (j < n)
        jj = np.clip(j, 0, n - 1)
        dd = np.hypot(bx[jj] - bx, by[jj] - by)
        dd[~ok | ~bvis[jj]] = 0.0
        maxd = np.maximum(maxd, np.nan_to_num(dd, nan=0.0))
    out = {"dup": dup, "frag": frag, "few": few, "ball_missing": ~bvis,
           "ball_still": bvis & (maxd < STILL_M)}
    from tools.learned_spike import features
    lo, hi = band
    cols = features.column_indices(list(SPREAD[version]), version)
    f = np.asarray(feats, dtype=np.float64)[:, cols]
    out_bin = ((f < lo[cols]) | (f > hi[cols])).any(axis=1)
    out["band"] = out_bin[np.minimum(np.arange(n) // BIN, out_bin.size - 1)]
    return out


def any_disagree(p5s, n, graded):
    from tools.learned_spike import bar_report as br
    tops = np.vstack([(np.asarray(p, dtype=np.float64) if len(p) == n
                       else br.upsample(np.asarray(p, dtype=np.float64), n))[graded].argmax(axis=1)
                      for p in p5s])
    return (tops != tops[:1]).any(axis=0)


def read_dumps(work, tag, game):
    folder = os.path.join(work, "probs", tag, game)
    def one(path):
        with np.load(path) as z:
            return z["p25"] if "p25" in z.files else z["p5"]     # p25: round 6, R6E members
    seeds = [one(os.path.join(folder, "seed_%d.npz" % i)) for i in range(4)]
    copies = [one(os.path.join(folder, "copy_%d.npz" % i)) for i in range(4)]
    return seeds, copies


def match_stats(game, tac, feats, seeds, copies, band, version):
    n = data.frame_count(game)
    graded = graded_mask(game)
    cond = conditions(tac, game, feats, band, version)
    sd = any_disagree(seeds, n, graded)
    cd = any_disagree(copies, n, graded)
    g = {k: v[graded] for k, v in cond.items()}
    track = g["dup"] | g["frag"] | g["few"] | g["band"]
    still = g["ball_still"]
    masks = {"all": np.ones(int(graded.sum()), dtype=bool)}
    masks.update({c: g[c] for c in CONDS})
    masks.update({"track_only": track & ~still, "still_only": still & ~track,
                  "both": track & still, "neither": ~track & ~still, "track_any": track})
    return {"game": game, "graded": int(graded.sum()), "seed_total": float(sd.mean()),
            "copy_total": float(cd.mean()),
            "stats": {k: {"frames": int(m.sum()), "seed_dis": int(sd[m].sum()),
                          "copy_dis": int(cd[m].sum())} for k, m in masks.items()}}


def per_match(m, row):
    s, g = m["stats"][row], m["graded"]
    f = s["frames"]
    return {"share": f / g, "seed_rate": s["seed_dis"] / f if f else None,
            "copy_rate": s["copy_dis"] / f if f else None,
            "seed_sxr": s["seed_dis"] / g, "copy_sxr": s["copy_dis"] / g}


def dataset_table(matches):
    out = {}
    for row in ROWS:
        pm = [per_match(m, row) for m in matches]
        fr = sum(m["stats"][row]["frames"] for m in matches)
        sd = sum(m["stats"][row]["seed_dis"] for m in matches)
        cd = sum(m["stats"][row]["copy_dis"] for m in matches)
        out[row] = {"share_mean": float(np.mean([p["share"] for p in pm])),
                    "seed_rate_pooled": sd / fr if fr else None,
                    "copy_rate_pooled": cd / fr if fr else None,
                    "seed_sxr_mean": float(np.mean([p["seed_sxr"] for p in pm])),
                    "copy_sxr_mean": float(np.mean([p["copy_sxr"] for p in pm])),
                    "frames": int(fr)}
    return out


def breakdown(bar, sc):
    """Tables and contributions to the excess (step 0's part1_aggregate, part 1 only)."""
    tables = {"match": dataset_table([bar]), "sc": dataset_table(sc)}
    contrib = {}
    for key in ("seed", "copy"):
        sc_mean = float(np.mean([m[key + "_total"] for m in sc]))
        excess = bar[key + "_total"] - sc_mean
        rows = {}
        for row in ROWS:
            b = tables["match"][row][key + "_sxr_mean"]
            s = tables["sc"][row][key + "_sxr_mean"]
            rows[row] = {"match_sxr": b, "sc_mean_sxr": s, "contribution": b - s,
                         "share_of_excess": (b - s) / excess if excess else None}
        track = rows["track_only"]["contribution"] + rows["both"]["contribution"]
        contrib[key] = {"rows": rows, "match_total": bar[key + "_total"], "sc_mean": sc_mean,
                        "excess": excess, "ratio_to_sc_mean": bar[key + "_total"] / sc_mean
                        if sc_mean else None,
                        "track_cells_share_of_excess": track / excess if excess else None}
    return tables, contrib


def part1_md(rec):
    w = ["# Step 0 breakdown (part 1) for the round 5 model", ""]
    if rec["rehearsal"]:
        w += ["**REHEARSAL on {} in place of the competition match: not a result.**".format(
            rec["match"]), ""]
    w += ["Recorded only (LEARNED_STAGE_PROTOCOL_R5.md Section 4). Feature version {}; band of "
          "the {} external matches in {}.".format(rec["feature_version"], len(rec["band_games"]),
                                                  rec["feature_cache"]), ""]
    for key in ("seed", "copy"):
        c = rec["contributions"][key]
        w.append("{}: match {:.4f}, SkillCorner fold mean {:.4f}, ratio {}, track cells carry "
                 "{} of the excess.".format(
                     key, c["match_total"], c["sc_mean"],
                     "-" if c["ratio_to_sc_mean"] is None else "{:.2f}".format(
                         c["ratio_to_sc_mean"]),
                     "-" if c["track_cells_share_of_excess"] is None else "{:.0%}".format(
                         c["track_cells_share_of_excess"])))
    w += ["", "| frames | share, match | seed dis. inside | copy dis. inside | share, SkillCorner "
          "| seed dis. inside | copy dis. inside |", "|---|---|---|---|---|---|---|"]
    f = lambda v: "-" if v is None else "{:.4f}".format(v)  # noqa: E731
    for row in ROWS:
        m, s = rec["tables"]["match"][row], rec["tables"]["sc"][row]
        w.append("| {} | {:.3f} | {} | {} | {:.3f} | {} | {} |".format(
            row, m["share_mean"], f(m["seed_rate_pooled"]), f(m["copy_rate_pooled"]),
            s["share_mean"], f(s["seed_rate_pooled"]), f(s["copy_rate_pooled"])))
    return "\n".join(w) + "\n"


def main(argv=None):
    from tools.learned_spike import bar_report as br
    from tools.learned_spike import build
    from tools.learned_spike import evaluate as ev
    from tools.learned_spike import r3_eval
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--feature-cache", required=True)
    p.add_argument("--as-match", default=None,
                   help="rehearsal: the development match of the label-free step")
    for name in ("--engine-cache", "--sc-engine-cache", "--ref-cache"):
        p.add_argument(name)
    p.add_argument("--refs-root", default=ev.DEFAULT_REFS_ROOT)
    p.add_argument("--sample-root", default=ev.DEFAULT_SAMPLE_ROOT)
    p.add_argument("--mtah-root", default=ev.DEFAULT_MTAH_ROOT)
    args = p.parse_args(argv)
    ctx = r3_eval.make_ctx(args)
    match = r3_eval.Match(ctx, args.as_match)
    version = build.folder_version(args.feature_cache)
    if version not in SPREAD:
        raise SystemExit("no spread features defined for {}".format(version))
    lo, hi, used = br.training_band(args.feature_cache, list(folds.train_games(pff=False)))
    band = (lo, hi)
    final_rec = r3_eval.read_json(os.path.join(ctx.work, "npz", "final.json")) \
        if os.path.isfile(os.path.join(ctx.work, "npz", "final.json")) else {}
    # The label-free match: its raw ATD for the conditions; the features the network read.
    seeds, copies = read_dumps(ctx.work, "final", match.game)
    feats = br.bar_features(match.game, lambda g: match.tactical(guarded=True), version)
    bar = match_stats(match.game, match.tactical(guarded=False), feats, seeds, copies, band,
                      version)
    load = ev.ext_loader(ctx)
    sc = []
    for game in folds.SC:
        s, c = read_dumps(ctx.work, "fold", game)
        with np.load(os.path.join(args.feature_cache, "feats", "raw", game + ".npz")) as z:
            f5 = np.asarray(z["F"])
        sc.append(match_stats(game, load(game), f5, s, c, band, version))
    tables, contrib = breakdown(bar, sc)
    rec = {"match": match.game, "rehearsal": match.rehearsal, "feature_version": version,
           "feature_cache": os.path.abspath(args.feature_cache), "band_games": used,
           "final_npz": final_rec.get("out"), "tables": tables, "contributions": contrib,
           "match_stats": bar, "sc_stats": {m["game"]: m for m in sc},
           "definitions": "tools/learned_spike/r5_step0.py docstring (step 0, part 1)"}
    out = os.path.join(ctx.work, "step0")
    os.makedirs(out, exist_ok=True)
    r3_eval.write_json(os.path.join(out, "part1.json"), rec)
    text = part1_md(rec)
    with open(os.path.join(out, "PART1.md"), "w") as handle:
        handle.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
