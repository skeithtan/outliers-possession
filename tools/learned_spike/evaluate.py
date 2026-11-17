"""The evaluation driver of the learned stage: protocol Sections 4, 5, 7 and 8, and round 2. Not part of the submission.

This tool scores the candidates M (model alone) and H (hybrid) against c22. It implements
docs/plan/calibration/LEARNED_STAGE_PROTOCOL.md Sections 4 to 8 and the round 2 changes of
LEARNED_STAGE_PROTOCOL_R2.md (criteria 9 and 10 from the vetoes V2 and V3, the veto V1 on the
sealed PFF set, the 8c band over the round 2 training set, learned_v2.npz,
LEARNED_STAGE_RESULT_R2). Each scored label comes from engine.run with a config override, so the
scored code is the shipped code.

Names in the commands below:

- STAGE: the output folder of run_stage.sh. It holds folds/fold{k}/net_s{seed}.pt and
  costs.json, and final/net_s{seed}.pt and costs.json.
- W: a work folder outside the repository. The tool refuses a folder inside it.
- EXT: the c22.py cache. It holds views/raw/engine_data and refs/METRICA-*_ref.csv.
- SCX: the engine.data cache of the SkillCorner matches.
- CACHE: the build.py feature cache (feats/raw/<game>.npz), for Section 8c.
- T: the python of the torch venv. E: the python of the engine venv (~/.venvs/barca).

Run these steps in this order. Each step is resumable: it caches every engine output under a
key of the match, the view, the full config, the md5 of the weights file and the source hash of
engine/ (and of corrupt.py for a stressed view), and skips a run whose output exists.

1. Export the weights. Run this step with the torch venv:
   $T -m tools.learned_spike.evaluate export --stage STAGE --work W
       --final-out engine/models/learned_v2.npz
   (the tool refuses to overwrite the round 1 file learned_v1.npz)
2. Run the engine on the held-out matches (c22, M and H with the fold costs, M and H again
   with the grand-median costs):
   $E -m tools.learned_spike.evaluate run --stage STAGE --work W --engine-cache
       EXT/views/raw/engine_data --sc-engine-cache SCX --ref-cache EXT --c22-ext EXT
3. Score the runs (writes W/scores.json):
   $E -m tools.learned_spike.evaluate score (the arguments of step 2)
4. Run the stress views of criteria 4 and 5 (writes W/stress.json):
   $E -m tools.learned_spike.evaluate stress (the arguments of step 2)
5. Make the label-free BAR-ATH numbers of criterion 6 and Section 8 (writes W/bar.json):
   $E -m tools.learned_spike.evaluate bar (the arguments of step 2)
       --final-npz engine/models/learned_v2.npz --feature-cache CACHE
   The 8c band pools the round 2 training set: the 28 external matches and the 43 PFF
   training matches (folds.TRAIN_ALL), so CACHE must hold all 71 (R2 Section 2).
6. Run the vetoes V2 and V3 of R2 Section 3, criteria 9 and 10 (writes W/v23.json):
   $E -m tools.learned_spike.evaluate v23 (the arguments of step 2)
       --final-npz engine/models/learned_v2.npz
7. Write W/repro.json for criterion 8 (protocol Section 6), for example
   {"pass": true, "evidence": "two runs md5 ..., repro_fresh_clone.sh ..., numpy-torch ..."}.
   If W/bar.json lists features outside the band (Section 8c), write W/explanations.json:
   {"<feature>": {"explanation": "...", "covered_by_stress": true}}.
8. Decide (writes docs/plan/calibration/LEARNED_STAGE_RESULT_R2.md and .json):
   $E -m tools.learned_spike.evaluate decide --work W --repro W/repro.json
       [--explanations W/explanations.json]
9. Only if decide chose M or H: the veto V1 on the sealed PFF set S (R2 Section 3). First
   record the md5 of learned_v2.npz and the engine commit in the result files, commit them,
   then run the 15 matches (writes W/sealed.json):
   $E -m tools.learned_spike.evaluate sealed-record --work W --stage STAGE
   $E -m tools.learned_spike.evaluate sealed --work W --stage STAGE

`decide` refuses to run until every input exists, covers the 28 matches and is not a smoke
output. It prints the list of missing inputs. `sealed` refuses to run without a result file
that names M or H and holds the V1 record, or when the weights, the final costs or the engine
source differ from that record.

These are the conventions that the protocol leaves to the tool. Each one is in the result file.

- Bootstrap: one RandomState(0). Each draw resamples the matches of each provider with
  replacement (providers in the order Metrica, DFL, SkillCorner, matches sorted by name). The
  draw value is the mean of the three provider means of the paired differences. The interval is
  the 5th and 95th percentile of 1,000 draws.
- Reference shift by s frames: the reference label of frame t moves to frame t + s. A frame
  with no source frame gets X, so the scorer ignores it. The predictions do not move.
- Criterion 4: the drop of a system is its unstressed provider-balanced mean minus its stressed
  provider-balanced mean (a loss is positive). "Half of its unstressed value" compares the
  stressed provider-balanced delta with half of the unstressed delta (fold costs, 28 matches).
- Criterion 5: hidden frames are the graded frames where the raw view has a ball and the
  uniform30 view has none (corrupt.py --hidden-out). Reference-alive frames are H or A in the
  reference as provided (the 2 s view has the same alive frames). The share pools the frames of
  the matches of a provider. The uniform30 view hides the ball only: corrupt.py --ball
  uniform30 --no-frag --no-swap --ghost-per-10min 0 (protocol Section 12, reading 4).
- Section 8a and 8b: the rate is the share of graded frames where the arg max of the members
  differs (bar_report.py "share_any_disagree"). The SkillCorner mean is the mean over the 20
  SkillCorner matches, each predicted by the ensemble of the fold that holds it out (protocol
  Section 12, reading 1). The in-sample rate of the final ensemble is recorded.
- Section 8e: the envelope is the range of the 28 references as provided. The tool checks the
  median and the 90th percentile of the possession lengths, and for M also of the D segment
  lengths. The range of the 2 s references is recorded.
- Section 8f: the mean fold-ensemble labels come from one weights file with the 32 fold models
  (equal to the mean of the eight fold ensembles), decoded by engine.run with the grand-median
  costs in the same mode.
- Section 7, transfer rule: H must qualify to ship. If M fails criterion 4, 5 or 7, H passes
  them and H fails another criterion, c22 ships and the result file says so. Round 2 keeps the
  rule verbatim: the transfer condition looks at criteria 4, 5 and 7 only, and to qualify a
  candidate passes criteria 1 to 10.

Round 2 conventions (R2 Section 3):

- V2 (criterion 9): the view is corrupt.py --ball uniform30 --no-frag --no-swap
  --ghost-per-10min 0 --seed 0 on BAR-ATH (corrupt.py --allow-competition), each system with
  the final weights and the final costs. "seed 0" replaces the crc32 of the match name in
  corrupt.seed_of(), so the ball gaps draw from RandomState(0 + SEED_BALL). K is the graded
  frames whose ball the view hides (--hidden-out). A_Y is the frames of K that system Y labels
  H or A on the raw view (the bar step's run), flip_Y the share of A_Y that Y labels D on the
  hidden view. M passes if flip_M <= flip_c22 + 0.10. H is exempt (its D frames are c22's).
- V3 (criterion 10): the share of graded BAR-ATH frames with no ball in the raw tracking that a
  system labels D (bar_report.system_stats "d_share_ball_missing", here unrounded). M passes if
  its share is at most c22's + 0.02. H is exempt.
- V1 (`sealed`): X and c22 with the final weights and costs on the 15 sealed PFF matches, raw
  view, scored at shift 0 against the 2 s and the provided references. The bootstrap is over
  matches (one provider): matches sorted by name, RandomState(0), 1,000 draws, 5-95 %. Rules
  of PFF_TRANSFER_TEST.md Section 5 with T replaced by S: a. the 5 % bound of the 2 s mean
  delta is above 0; b. the provided mean delta is above 0 and its 5 % bound is above -0.010;
  c. delta > 0 (2 s) on at least 60 % of the matches (5 x wins >= 3 x 15, so 9 of 15). The
  fallback is sequential: if X = M fails and H qualified, H runs and ships only if it passes;
  else c22 ships. A system whose runs do not finish S fails. The shift -10 and +10 rows, the D
  share and the possession count per match, and c22's mean on S against its mean on the pilot
  set P (PFF_REFERENCES.md Section 6, quoted, not re-run) are recorded, never decisive.

The tool never reads evaluation/gt and never scores BAR-ATH. It runs a sealed PFF match only
in `sealed`, and never the sealed DFL matches. BAR-ATH outputs stay in W. Python 3.9.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from tools.learned_spike import folds  # noqa: E402
from tools.learned_spike import pff  # noqa: E402

PROVIDERS = ("metrica", "dfl", "skillcorner")
PROVIDER_TITLES = {"metrica": "Metrica", "dfl": "DFL", "skillcorner": "SkillCorner"}
CANDIDATES = ("M", "H")
SEEDS = (0, 1, 2, 3)
CONTROL_FRAMES = 50          # the 2 s controlled reference at 25 fps (protocol Section 2)
SHIFTS = (-10, 0, 10)        # reference shifts of the recorded rows (protocol Section 4)
N_DRAWS = 1000
BOOT_SEED = 0
PERCENTILES = (5, 95)
KEEP_SHARE = 0.5             # criterion 4: keep at least half of the unstressed delta
MAX_EXTRA_DROP = 0.05        # criterion 4: absolute drop of X minus that of c22
MAX_EXTRA_D = 0.10           # criterion 5: 10 points
TIE = 0.010                  # Section 7
BOUND_FACTOR = 1.5           # Section 8a, 8b
MIN_AGREEMENT = 0.90         # Section 8f
SLOT_COUNT_FEATURES = ("n_tracks", "H_nvis", "A_nvis")
ENVELOPE_KEYS = {"M": ("possession_s.p50", "possession_s.p90", "d_segment_s.p50",
                       "d_segment_s.p90"),
                 "H": ("possession_s.p50", "possession_s.p90")}
VIEW_RAW, VIEW_STRESS, VIEW_U30 = "raw", "stress", "uniform30"
# Round 3 records the matched input view of the 8 open matches (LEARNED_STAGE_PROTOCOL_R3.md
# Section 4). Raw and matched run through external_sweep --engine-child; the other views through
# corrupt.py engine.
VIEW_MATCHED = "matched"
ENGINE_CHILD_VIEWS = (VIEW_RAW, VIEW_MATCHED)
# corrupt.py arguments of each stressed view. Criterion 5 uses the ball gaps alone (protocol
# Section 12, reading 4): no fragmentation, no swaps, no ghosts.
VIEW_ARGS = {VIEW_STRESS: ("--ball", "matched"),
             VIEW_U30: ("--ball", "uniform30", "--no-frag", "--no-swap",
                        "--ghost-per-10min", "0")}
RUN_SYSTEMS = ("c22", "M", "H", "M_med", "H_med")
STRESS_SYSTEMS = ("c22", "M", "H")
BAR_SYSTEMS = ("c22", "M", "H", "Mfold", "Hfold")
COMPETITION = folds.COMPETITION
CALIBRATION = os.path.join(ROOT, "docs", "plan", "calibration")
RESULT_MD = os.path.join(CALIBRATION, "LEARNED_STAGE_RESULT_R2.md")
ROUND1_RESULT_MD = os.path.join(CALIBRATION, "LEARNED_STAGE_RESULT.md")
FINAL_NPZ = os.path.join(ROOT, "engine", "models", "learned_v2.npz")      # R2 Section 2
ROUND1_NPZ = os.path.join(ROOT, "engine", "models", "learned_v1.npz")     # never overwritten
# Round 2 vetoes (R2 Section 3).
VIEW_V2 = "v2_ball_hidden"
V2_SEED = 0
VIEW_ARGS[VIEW_V2] = VIEW_ARGS[VIEW_U30] + ("--seed", str(V2_SEED))
V2_SYSTEMS = ("c22", "M", "H")
MAX_EXTRA_FLIP = 0.10        # V2, criterion 9
MAX_EXTRA_D_MISSING = 0.02   # V3, criterion 10
N_CRITERIA = 10
EXEMPT_H = ("9", "10")       # H's D frames are c22's (R2 Section 3)
HONESTY_NOTE = ("Change B, V3's margin of 0.02 and the decision to keep the 1.5x bounds were "
                "chosen knowing round 1's numbers (219 = 219, 0.9113, 0.0594). They are "
                "pre-declared for round 2, but they are not blind to round 1 "
                "(LEARNED_STAGE_PROTOCOL_R2.md Section 7).")
# V1 on the sealed PFF set (R2 Section 3, PFF_TRANSFER_TEST.md Sections 4 and 5).
V1_RAW_FLOOR = -0.010        # rule b
V1_WIN_SHARE = (3, 5)        # rule c: wins / n >= 3 / 5 (60 %), exact integer test
V1_GATE = "14 November 2026 23:59 CET"     # Gate 3 (R2 Section 4), procedural
# c22 on the pilot set P (PFF_REFERENCES.md Section 6), quoted for the recorded gap flag only.
PILOT_C22_MEAN = {"ctl2": 0.3672, "provided": 0.3296}
MAX_PILOT_GAP = 0.05


def log(text):
    print(text, file=sys.stderr, flush=True)


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(obj, handle, indent=1, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def md5_file(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# -- statistics (pure, no data) -------------------------------------------------------------------

def by_provider(games):
    """{provider: sorted games} for the providers present."""
    out = {}
    for game in sorted(games):
        out.setdefault(folds.provider(game), []).append(game)
    return out


def provider_weights(games):
    """{game: weight}: each provider present weighs the same, shared equally by its matches."""
    groups = by_provider(games)
    return {g: 1.0 / (len(groups) * len(names)) for names in groups.values() for g in names}


def balanced_mean(values):
    """The provider-balanced mean of {game: value}."""
    weights = provider_weights(values)
    return float(sum(weights[g] * values[g] for g in values))


def stratified_bootstrap(deltas, n_draws=N_DRAWS, seed=BOOT_SEED):
    """(low, high): 5-95 % of the provider-balanced mean under a provider-stratified bootstrap.

    Each draw resamples the matches of each provider with replacement, in the order PROVIDERS
    and with the matches sorted by name, from one RandomState(seed).
    """
    groups = by_provider(deltas)
    arrays = [np.array([deltas[g] for g in groups[p]], dtype=np.float64)
              for p in PROVIDERS if p in groups]
    rng = np.random.RandomState(seed)
    draws = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        draws[i] = np.mean([a[rng.randint(0, a.size, a.size)].mean() for a in arrays])
    low, high = np.percentile(draws, PERCENTILES)
    return float(low), float(high)


def delta_summary(x, base, n_draws=N_DRAWS, seed=BOOT_SEED):
    """Provider-balanced means, the paired delta x - base and its interval, per provider too."""
    if sorted(x) != sorted(base):
        raise ValueError("the two systems must name the same matches")
    deltas = {g: float(x[g]) - float(base[g]) for g in x}
    low, high = stratified_bootstrap(deltas, n_draws, seed)
    per = {}
    for p, names in by_provider(x).items():
        d = [deltas[g] for g in names]
        per[p] = {"n": len(names), "x_mean": float(np.mean([x[g] for g in names])),
                  "c22_mean": float(np.mean([base[g] for g in names])),
                  "delta": float(np.mean(d)), "wins": int(sum(v > 0 for v in d))}
    return {"x_mean": balanced_mean(x), "c22_mean": balanced_mean(base),
            "delta": balanced_mean(deltas), "low": low, "high": high, "per_provider": per,
            "per_match": {g: {"x": float(x[g]), "c22": float(base[g]), "delta": deltas[g]}
                          for g in sorted(x)}}


def shift_codes(codes, shift, fill=3):
    """The reference label of frame t moved to frame t + shift; frames with no source get X."""
    codes = np.asarray(codes)
    out = np.full_like(codes, fill)
    n = codes.size
    if shift >= 0:
        out[shift:] = codes[:n - shift]
    else:
        out[:n + shift] = codes[-shift:]
    return out


# -- criteria and choice (pure, no data) ----------------------------------------------------------

def crit1(fold_ctl2, median_ctl2):
    ok = fold_ctl2["low"] > 0 and median_ctl2["low"] > 0
    return {"pass": bool(ok), "numbers": {
        "fold_costs": [fold_ctl2["delta"], fold_ctl2["low"], fold_ctl2["high"]],
        "median_costs": [median_ctl2["delta"], median_ctl2["low"], median_ctl2["high"]]}}


def two_thirds(wins, n):
    """True if wins is at least 2/3 of n (exact integer test)."""
    return 3 * int(wins) >= 2 * int(n)


def crit2(fold_ctl2):
    rows, ok = {}, True
    for p in PROVIDERS:
        r = fold_ctl2["per_provider"].get(p)
        if r is None:
            rows[p] = None
            ok = False
            continue
        good = r["delta"] >= 0 and two_thirds(r["wins"], r["n"])
        rows[p] = {"delta": r["delta"], "wins": r["wins"], "n": r["n"], "pass": bool(good)}
        ok = ok and good
    return {"pass": bool(ok), "numbers": rows}


def crit3(fold_provided):
    return {"pass": bool(fold_provided["low"] > 0), "numbers": [
        fold_provided["delta"], fold_provided["low"], fold_provided["high"]]}


def crit4(stressed, unstressed, drop_x, drop_c22):
    """stressed, unstressed: delta_summary() of X - c22 on the 2 s references; drops balanced."""
    bound = stressed["low"] > 0
    keep = stressed["delta"] >= KEEP_SHARE * unstressed["delta"]
    extra = drop_x - drop_c22
    absolute = extra <= MAX_EXTRA_DROP
    return {"pass": bool(bound and keep and absolute), "numbers": {
        "stressed_delta": [stressed["delta"], stressed["low"], stressed["high"]],
        "unstressed_delta": unstressed["delta"], "keep_ratio": (
            stressed["delta"] / unstressed["delta"] if unstressed["delta"] else None),
        "bound_above_0": bool(bound), "keeps_half": bool(keep),
        "drop_x": drop_x, "drop_c22": drop_c22, "extra_drop": extra,
        "extra_drop_ok": bool(absolute)}}


def crit5(share_x, share_c22):
    """share_*: {provider: share of hidden reference-alive frames labelled D}."""
    rows, ok = {}, True
    for p in PROVIDERS:
        if p not in share_x or p not in share_c22:
            rows[p] = None
            ok = False
            continue
        diff = share_x[p] - share_c22[p]
        rows[p] = {"x": share_x[p], "c22": share_c22[p], "diff": diff,
                   "pass": bool(diff <= MAX_EXTRA_D)}
        ok = ok and diff <= MAX_EXTRA_D
    return {"pass": bool(ok), "numbers": rows}


def veto_limits():
    """The PLAN 5.3 veto bounds of engine.config (read, not changed)."""
    try:
        from engine.config import Config
        cfg = Config()
        return {"poss_min": cfg.VETO_POSSESSIONS_MIN, "poss_max": cfg.VETO_POSSESSIONS_MAX,
                "d_min": cfg.VETO_D_SHARE_MIN, "d_max": cfg.VETO_D_SHARE_MAX,
                "dseg_min": cfg.VETO_MIN_D_SEGMENTS, "x_max": cfg.VETO_MAX_GRADED_X}
    except ImportError:
        return {"poss_min": 180, "poss_max": 450, "d_min": 0.18, "d_max": 0.55,
                "dseg_min": 55, "x_max": 0}


def crit6(stats, limits=None):
    lim = limits or veto_limits()
    checks = {
        "possessions": lim["poss_min"] <= stats["possessions"] <= lim["poss_max"],
        "d_share": lim["d_min"] <= stats["d_share"] <= lim["d_max"],
        "d_segments": stats["d_segments"] >= lim["dseg_min"],
        "graded_x": stats["graded_x"] <= lim["x_max"],
    }
    return {"pass": bool(all(checks.values())), "numbers": {
        k: {"value": stats[k], "pass": bool(v)} for k, v in checks.items()}}


def bounds8(cand, bar, explanations=None):
    """Section 8 for one candidate from the bar.json record. Returns {part: {...}, "pass"}."""
    explanations = explanations or {}
    out = {}
    for part, key in (("a", "seed"), ("b", "copy")):
        r = bar["disagreement"][key]
        limit = BOUND_FACTOR * r["sc_mean_fold"]
        out[part] = {"bar": r["bar"], "sc_mean": r["sc_mean_fold"], "limit": limit,
                     "sc_mean_final_recorded": r.get("sc_mean_final"),
                     "pass": bool(r["bar"] <= limit)}
    outside = list(bar["features"]["outside"])
    rows, ok = {}, True
    for name in outside:
        note = explanations.get(name) or {}
        explained = bool(note.get("explanation"))
        if name in SLOT_COUNT_FEATURES:
            good = explained and bool(note.get("covered_by_stress"))
        else:
            good = explained
        rows[name] = {"slot_count": name in SLOT_COUNT_FEATURES, "explained": explained,
                      "covered_by_stress": bool(note.get("covered_by_stress")),
                      "pass": bool(good)}
        ok = ok and good
    out["c"] = {"outside": rows, "pass": bool(ok)}
    out["d"] = {"reported": True}
    env = bar["envelope_provided"][cand]
    erows = {}
    for key in ENVELOPE_KEYS[cand]:
        erows[key] = env.get(key)
    eok = all(r is not None and r["inside"] for r in erows.values())
    out["e"] = {"checked": erows, "pass": bool(eok),
                "exempt_for_H": ["d_share", "d_segments", "d_segment_s"] if cand == "H" else []}
    agree = bar["agreement_final_vs_fold"][cand]
    out["f"] = {"agreement": agree, "pass": bool(agree >= MIN_AGREEMENT)}
    out["g"] = {"reported": True, "x": bar["systems"][cand].get("d_share_ball_missing"),
                "c22": bar["systems"]["c22"].get("d_share_ball_missing")}
    out["pass"] = bool(all(out[p]["pass"] for p in ("a", "b", "c", "e", "f")))
    return out


def crit7(b8):
    return {"pass": bool(b8["pass"]), "numbers": {p: b8[p]["pass"] for p in "abcef"}}


def crit8(repro):
    return {"pass": bool(repro.get("pass") is True), "numbers": repro}


def v2_counts(raw_codes, hidden_codes, hidden, graded):
    """V2 for one system: K = hidden graded frames, A = frames of K labelled H or A on the raw
    view, flip = share of A labelled D on the hidden view (R2 Section 3)."""
    raw_codes = np.asarray(raw_codes)
    hidden_codes = np.asarray(hidden_codes)
    k = np.asarray(hidden, dtype=bool) & np.asarray(graded, dtype=bool)
    alive = k & np.isin(raw_codes, (0, 1))
    n = int(alive.sum())
    d = int(np.sum(hidden_codes[alive] == 2))
    return {"k_frames": int(k.sum()), "alive_raw": n, "flipped_to_d": d,
            "flip": d / float(n) if n else None}


def d_share_missing(codes, runs, ball_missing):
    """V3 for one system: the share of graded frames with no ball (raw tracking) labelled D.

    The same statistic as bar_report.system_stats "d_share_ball_missing" (filled abstentions),
    unrounded.
    """
    import possession_metrics as pm
    from tools.learned_spike import bar_report as br
    filled = pm.fill_abstentions(np.asarray(codes, dtype=np.int8), runs)
    sel = br.graded_mask(runs, len(filled)) & np.asarray(ball_missing, dtype=bool)
    return float(np.mean(filled[sel] == pm.D)) if sel.any() else None


def crit9(v23, cand):
    """V2: flip_X <= flip_c22 + 0.10. H is exempt."""
    sysv = v23["v2"]["systems"]
    x, c = sysv[cand]["flip"], sysv["c22"]["flip"]
    nums = {"flip_x": x, "flip_c22": c, "limit": None if c is None else c + MAX_EXTRA_FLIP,
            "alive_raw_x": sysv[cand]["alive_raw"], "k_frames": v23["v2"]["k_frames"]}
    if cand == "H":
        return {"pass": True, "exempt": True, "numbers": nums}
    ok = x is not None and c is not None and x <= c + MAX_EXTRA_FLIP
    return {"pass": bool(ok), "numbers": nums}


def crit10(v23, cand):
    """V3: X's D share on missing-ball graded frames <= c22's + 0.02. H is exempt."""
    x, c = v23["v3"][cand], v23["v3"]["c22"]
    nums = {"d_missing_x": x, "d_missing_c22": c,
            "limit": None if c is None else c + MAX_EXTRA_D_MISSING}
    if cand == "H":
        return {"pass": True, "exempt": True, "numbers": nums}
    ok = x is not None and c is not None and x <= c + MAX_EXTRA_D_MISSING
    return {"pass": bool(ok), "numbers": nums}


def qualifies(criteria):
    """Criteria 1 to 10 (round 1 Section 5 and R2 Section 3) all pass."""
    return all(criteria[str(i)]["pass"] for i in range(1, N_CRITERIA + 1))


def choose(results):
    """Section 7. results: {"M"|"H": {"criteria": {"1".."10": {"pass"}}, "mean": 2 s mean}}."""
    q = {c: qualifies(results[c]["criteria"]) for c in CANDIDATES}
    m_fail = [i for i in ("4", "5", "7") if not results["M"]["criteria"][i]["pass"]]
    h_pass = all(results["H"]["criteria"][i]["pass"] for i in ("4", "5", "7"))
    transfer = bool(m_fail) and h_pass
    note = None
    if transfer and q["H"]:
        return {"choice": "H", "qualifies": q, "transfer_rule": True,
                "reason": "transfer rule: M fails criterion {} and H passes 4, 5 and 7".format(
                    ", ".join(m_fail))}
    if transfer and not q["H"]:
        note = ("the transfer-rule condition holds (M fails criterion {}, H passes 4, 5 and 7) "
                "but H fails another criterion; the tool reads Section 7 as requiring H to "
                "qualify".format(", ".join(m_fail)))
    if q["M"] and q["H"]:
        diff = results["M"]["mean"] - results["H"]["mean"]
        if abs(diff) < TIE:
            choice, reason = "H", "both qualify; 2 s means differ by {:.4f} < {:.3f}: H".format(
                abs(diff), TIE)
        else:
            choice = "M" if diff > 0 else "H"
            reason = "both qualify; higher provider-balanced 2 s mean ({:+.4f} M - H)".format(diff)
    elif q["M"]:
        choice, reason = "M", "only M qualifies"
    elif q["H"]:
        choice, reason = "H", "only H qualifies"
    else:
        choice, reason = "c22", "neither M nor H qualifies"
    out = {"choice": choice, "qualifies": q, "transfer_rule": False, "reason": reason}
    if note:
        out["note"] = note
    return out


# -- configuration of the runs --------------------------------------------------------------------

class Ctx(object):
    """Paths and roots shared by the subcommands."""

    def __init__(self, args):
        self.args = args
        self.work = os.path.abspath(os.path.expanduser(args.work))
        if os.path.commonpath([self.work, ROOT]) == ROOT:
            raise SystemExit("--work must be outside the repository (BAR-ATH outputs go there)")
        self.stage = os.path.abspath(os.path.expanduser(args.stage)) if getattr(
            args, "stage", None) else None
        self.smoke = bool(getattr(args, "smoke_costs", None))
        self._md5 = {}

    def npz(self, k):
        return os.path.join(self.work, "npz", "fold{}.npz".format(k))

    def folds_all(self):
        return os.path.join(self.work, "npz", "folds_all.npz")

    def weights_md5(self, path):
        if path not in self._md5:
            self._md5[path] = md5_file(path)
        return self._md5[path]


def costs_record(path, smoke_costs):
    """{"M": (ha, ad), "H": ha, "source"} from a costs.json, or the smoke costs if it is missing."""
    if os.path.isfile(path):
        rec = read_json(path)
        h = (rec.get("H") or {}).get("cost_ha")      # None in round 4 (no candidate H)
        out = {"M": (float(rec["M"]["cost_ha"]), float(rec["M"]["cost_ad"])),
               "H": None if h is None else float(h), "source": path, "smoke": False}
        if rec.get("pf"):
            out["pf"] = rec["pf"]                      # round 6: the per-frame member
        return out
    if smoke_costs:
        ha, ad, hh = smoke_costs
        return {"M": (float(ha), float(ad)), "H": float(hh), "source": "smoke", "smoke": True}
    raise SystemExit("{} is missing (give --smoke-costs only for a smoke test)".format(path))


def fold_costs(ctx, k):
    return costs_record(os.path.join(ctx.stage, "folds", "fold{}".format(k), "costs.json"),
                        ctx.args.smoke_costs)


def final_costs(ctx):
    return costs_record(os.path.join(ctx.stage, "final", "costs.json"), ctx.args.smoke_costs)


def pf_overrides(costs):
    """Round 6 (variant R6E): Config.LEARNED_PF on with the weights and weight of costs["pf"]
    ({"weights", "weight", "md5"}); {} without a per-frame member."""
    pf = (costs or {}).get("pf")
    if not pf:
        return {}
    from tools.learned_spike import pf_gate
    # the engine runs use the final member ("{fold}" -> "final"; pf_gate.pf_path)
    return {"LEARNED_PF": "on",
            "LEARNED_PF_WEIGHTS": os.path.abspath(pf_gate.pf_path(pf["weights"])),
            "LEARNED_PF_WEIGHT": float(pf["weight"])}


def overrides(system, weights, costs):
    """The engine config override of one system ("c22", "M", "H")."""
    if system == "c22":
        return {}
    if system == "M":
        over = {"LEARNED_MODE": "model", "LEARNED_WEIGHTS": weights,
                "LEARNED_COST_HA": costs["M"][0], "LEARNED_COST_AD": costs["M"][1]}
        if costs.get("structure"):
            # Round 3, R3_DEVIATION_STRUCTURE.md: dead ball before each half's kickoff.
            over["LEARNED_STRUCTURE"] = True
        if costs.get("prior_bonus"):
            # Round 4: the rule-engine prior (Config.LEARNED_PRIOR_BONUS); absent when 0.
            over["LEARNED_PRIOR_BONUS"] = float(costs["prior_bonus"])
        over.update(pf_overrides(costs))
        return over
    if system == "H":
        return dict({"LEARNED_MODE": "hybrid", "LEARNED_WEIGHTS": weights,
                     "LEARNED_COST_HA_HYBRID": costs["H"]}, **pf_overrides(costs))
    raise ValueError(system)


def fold_of(game):
    for k in range(folds.N_FOLDS):
        if game in folds.fold(k)["test"]:
            return k
    raise ValueError(game)


def selected_games(args):
    games = list(args.games) if args.games else list(folds.ALL)
    return folds.check_all(games)


def roots(args):
    return (os.path.abspath(os.path.expanduser(args.refs_root)),
            os.path.abspath(os.path.expanduser(args.sample_root)),
            os.path.abspath(os.path.expanduser(args.mtah_root)))


def data_cache(ctx, game):
    args = ctx.args
    cache = args.sc_engine_cache if game.startswith("SC-") else args.engine_cache
    return os.path.abspath(os.path.expanduser(cache)) if cache else os.path.join(
        ctx.work, "engine_data")


def source_hash(paths):
    digest = hashlib.sha1()
    for path in paths:
        digest.update(os.path.relpath(path, ROOT).encode("utf-8"))
        with open(path, "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()


def corrupt_hash():
    here = os.path.dirname(os.path.abspath(__file__))
    return source_hash([os.path.join(here, "corrupt.py"), os.path.join(here, "slot_stats.py")])


def make_job(ctx, game, system, label, view, weights, costs):
    """A run description: match, system label, view, override and the cache key."""
    from analysis import external_sweep as xs
    over = overrides(system, weights, costs)
    if view in ENGINE_CHILD_VIEWS:
        root = xs.data_root(game, view, *roots(ctx.args))
    else:
        from tools.learned_spike import slot_stats
        root = xs.data_root(game, "raw", slot_stats.REFS_ROOT, slot_stats.SAMPLE_ROOT,
                            slot_stats.MTAH_ROOT)
    resolved, _ = xs.resolve_config(over)
    key_obj = {"game": game, "view": view, "config": resolved, "data_dir": root,
               "engine": engine_hash(), "weights_md5": ctx.weights_md5(weights) if weights
               else None, "corrupt": corrupt_hash() if view not in ENGINE_CHILD_VIEWS else None,
               "corrupt_args": list(VIEW_ARGS.get(view, ()))}
    if over.get("LEARNED_PF") == "on":
        key_obj["pf_md5"] = ctx.weights_md5(over["LEARNED_PF_WEIGHTS"])
    key = hashlib.sha1(json.dumps(key_obj, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    where = os.path.join(ctx.work, "runs", view, game)
    return {"game": game, "system": label, "view": view, "overrides": over, "data_dir": root,
            "key": key, "csv": os.path.join(where, key + ".csv"),
            "hidden": (os.path.join(where, key + ".hidden.npz")
                       if view not in ENGINE_CHILD_VIEWS else None),
            # The matched view has its own engine.data cache: the cache is keyed by match name.
            "engine_cache": (os.path.join(ctx.work, "views", view, "engine_data")
                             if view == VIEW_MATCHED else None),
            "weights": weights, "weights_md5": key_obj["weights_md5"],
            "costs": None if system == "c22" else costs.get("source")}


_ENGINE_HASH = []


def engine_hash():
    """external_sweep.engine_source_hash(), computed once."""
    if not _ENGINE_HASH:
        from analysis import external_sweep as xs
        _ENGINE_HASH.append(xs.engine_source_hash())
    return _ENGINE_HASH[0]


def raw_jobs(ctx, games, systems=RUN_SYSTEMS):
    """The raw-view jobs of the held-out matches: {(game, system label): job}."""
    med = final_costs(ctx) if any(s.endswith("_med") for s in systems) else None
    jobs = {}
    for game in games:
        k = fold_of(game)
        weights = ctx.npz(k)
        fc = fold_costs(ctx, k)
        for label in systems:
            base = label.split("_")[0]
            costs = med if label.endswith("_med") else fc
            if base != "c22" and not os.path.isfile(weights):
                raise SystemExit("{} is missing: run `export` first".format(weights))
            jobs[(game, label)] = make_job(ctx, game, base, label, VIEW_RAW,
                                           weights if base != "c22" else None, costs)
    return jobs


def stress_jobs(ctx, games):
    jobs = {}
    for game in games:
        k = fold_of(game)
        fc = fold_costs(ctx, k)
        for view in (VIEW_STRESS, VIEW_U30):
            for label in STRESS_SYSTEMS:
                weights = ctx.npz(k) if label != "c22" else None
                if weights and not os.path.isfile(weights):
                    raise SystemExit("{} is missing: run `export` first".format(weights))
                jobs[(game, view, label)] = make_job(ctx, game, label, label, view, weights, fc)
    return jobs


def run_one(ctx, job):
    """Run engine.run (raw view) or corrupt.py engine (stressed view) unless the CSV exists."""
    if os.path.isfile(job["csv"]) and (job["hidden"] is None or os.path.isfile(job["hidden"])):
        return job["csv"]
    folder = os.path.dirname(job["csv"])
    os.makedirs(folder, exist_ok=True)
    base = job["csv"][:-len(".csv")]
    if job["view"] == VIEW_RAW and job["system"] == "c22" and ctx.args.c22_ext:
        from tools.learned_spike import c22
        cj = c22.job_of(job["game"], ctx.args.c22_ext)
        src = c22.path_of(job["game"], ctx.args.c22_ext)
        if cj["data_dir"] == job["data_dir"] and os.path.isfile(src):
            shutil.copyfile(src, base + ".tmp.csv")
            write_json(base + ".meta.json", dict(job, copied_from=src))
            os.replace(base + ".tmp.csv", job["csv"])
            return job["csv"]
    config_path = base + ".config.json"
    write_json(config_path, job["overrides"])
    tmp = base + ".tmp.csv"
    env = dict(os.environ, MTAH_DATA_DIR=job["data_dir"])
    cache = job.get("engine_cache") or data_cache(ctx, job["game"])
    if job["view"] in ENGINE_CHILD_VIEWS:
        os.makedirs(cache, exist_ok=True)
        from analysis import external_sweep as xs
        command = [sys.executable, os.path.abspath(xs.__file__), "--engine-child", cache,
                   "--game", job["game"], "--config", config_path, "--out", tmp]
    else:
        command = [sys.executable, "-m", "tools.learned_spike.corrupt", "engine", "--game",
                   job["game"], "--out", tmp, "--config", config_path, "--hidden-out",
                   base + ".hidden.tmp.npz"] + list(VIEW_ARGS[job["view"]])
        for flag, value in (("--engine-cache", ctx.args.engine_cache),
                            ("--sc-engine-cache", ctx.args.sc_engine_cache)):
            if value:
                command += [flag, os.path.abspath(os.path.expanduser(value))]
    log("run {} {} {} ({})".format(job["view"], job["game"], job["system"], job["key"]))
    with open(base + ".log", "w") as out:
        code = subprocess.call(command, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
    if code != 0:
        raise RuntimeError("run failed: {} {} {}, see {}.log".format(
            job["view"], job["game"], job["system"], base))
    if job["hidden"]:
        os.replace(base + ".hidden.tmp.npz", job["hidden"])
    write_json(base + ".meta.json", dict(job, output_md5=md5_file(tmp)))
    os.replace(tmp, job["csv"])
    return job["csv"]


def run_all(ctx, jobs):
    todo = sorted(jobs.values(), key=lambda j: (j["view"], j["game"], j["system"]))
    seen, unique = set(), []
    for job in todo:
        if job["csv"] not in seen:
            seen.add(job["csv"])
            unique.append(job)
    n = max(1, int(ctx.args.jobs))
    if n == 1:
        for job in unique:
            run_one(ctx, job)
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(lambda j: run_one(ctx, j), unique))


def missing_runs(jobs):
    return sorted("{} {} {}".format(j["view"], j["game"], j["system"]) for j in jobs.values()
                  if not os.path.isfile(j["csv"]))


# -- references and scores ------------------------------------------------------------------------

def label_codes(labels):
    lut = {"H": 0, "A": 1, "D": 2, "X": 3}
    return np.array([lut[str(v)] for v in labels], dtype=np.int8)


def references(ctx, game):
    """{"provided": codes, "ctl2": codes, "provided_md5": file md5}."""
    from analysis import external_sweep as xs
    from analysis import control_reference as cref
    import possession_metrics as pm
    refs_root, sample_root, _ = roots(ctx.args)
    cache = os.path.abspath(os.path.expanduser(ctx.args.ref_cache)) if ctx.args.ref_cache \
        else ctx.work
    path, digest = xs.prepare_reference(game, refs_root, sample_root, cache)
    provided = pm.label_codes(pm.read_label_csv(path)).astype(np.int8)
    labels = np.asarray(pm.LABELS)[provided]
    ctl2 = label_codes(cref.control_labels(labels, CONTROL_FRAMES))
    return {"provided": provided, "ctl2": ctl2, "provided_md5": digest, "path": path}


def codes_md5(codes):
    return hashlib.md5(np.ascontiguousarray(codes, dtype=np.int8).tobytes()).hexdigest()


def score_csv(ctx, job, ref_codes, shift=0):
    """{"spell_f1", "macro_f1"} of one run against one (shifted) reference, cached."""
    import score_game as sg
    ref = shift_codes(ref_codes, shift) if shift else ref_codes
    path = os.path.join(ctx.work, "scores", job["game"], "{}_{}_s{}.json".format(
        job["key"], codes_md5(ref)[:12], shift))
    if os.path.isfile(path):
        return read_json(path)
    pred = sg.read_codes(job["csv"], len(ref))
    result = sg.score_codes(ref, pred)[0]
    rec = {"spell_f1": float(result["spell_f1_mean"]), "macro_f1": float(result["macro_f1"])}
    write_json(path, rec)
    return rec


def table(scores, game_list, label, ref, shift=0):
    return {g: scores[g][label][ref][str(shift)]["spell_f1"] for g in game_list}


def summaries(scores, games):
    """Every delta row of Section 4 from the per-match score table."""
    out = {}
    for ref in ("ctl2", "provided"):
        for kind, suffix in (("fold", ""), ("median", "_med")):
            for cand in CANDIDATES:
                out["{}|{}|{}|0".format(cand, kind, ref)] = delta_summary(
                    table(scores, games, cand + suffix, ref), table(scores, games, "c22", ref))
        for shift in SHIFTS:
            for cand in CANDIDATES:
                out["{}|fold|{}|{}".format(cand, ref, shift)] = delta_summary(
                    table(scores, games, cand, ref, shift),
                    table(scores, games, "c22", ref, shift))
    return out


# -- subcommands ----------------------------------------------------------------------------------

def check_final_out(path):
    """Refuse the round 1 weights file as an output (R2 Section 2: learned_v1.npz stays)."""
    if path and os.path.abspath(path) == os.path.abspath(ROUND1_NPZ):
        raise SystemExit("{} is the round 1 file: write round 2 to {}".format(path, FINAL_NPZ))
    return path


def cmd_export(ctx):
    """Fold weights to W/npz/fold{k}.npz, the 32 fold models to folds_all.npz, the final file."""
    import torch
    from tools.learned_spike import export
    check_final_out(ctx.args.final_out)
    done = []
    for k in range(folds.N_FOLDS):
        nets = [os.path.join(ctx.stage, "folds", "fold{}".format(k), "net_s{}.pt".format(s))
                for s in SEEDS]
        if not all(os.path.isfile(p) for p in nets):
            log("fold {}: seeds missing, skipped".format(k))
            continue
        inputs = {p: md5_file(p) for p in nets}
        out, rec = ctx.npz(k), ctx.npz(k)[:-4] + ".json"
        if os.path.isfile(out) and os.path.isfile(rec) and read_json(rec).get("inputs") == inputs:
            log("fold {}: exported, skipped".format(k))
            done.append(k)
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        export.write(out, [torch.load(p, map_location="cpu", weights_only=True) for p in nets],
                     export.recorded_version(nets))
        write_json(rec, {"inputs": inputs, "npz_md5": md5_file(out)})
        log("fold {}: wrote {}".format(k, out))
        done.append(k)
    if len(done) == folds.N_FOLDS:
        merge_npz([ctx.npz(k) for k in range(folds.N_FOLDS)], ctx.folds_all())
        log("wrote {}".format(ctx.folds_all()))
    if ctx.args.final_out:
        nets = [os.path.join(ctx.stage, "final", "net_s{}.pt".format(s)) for s in SEEDS]
        if all(os.path.isfile(p) for p in nets):
            export.write(ctx.args.final_out,
                         [torch.load(p, map_location="cpu", weights_only=True) for p in nets],
                         export.recorded_version(nets))
            write_json(os.path.join(ctx.work, "npz", "final.json"), {
                "inputs": {p: md5_file(p) for p in nets}, "out": os.path.abspath(
                    ctx.args.final_out), "npz_md5": md5_file(ctx.args.final_out)})
            log("wrote {}".format(ctx.args.final_out))
        else:
            log("final seeds missing, final weights not written")
    return 0


def merge_npz(paths, out):
    """One weights file with the models of `paths` in order (numpy only)."""
    from engine import learned
    arrays, k = {}, 0
    version = None
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            version = str(archive["feature_version"])
            for m in range(int(archive["n_models"])):
                for name in learned.LAYER_NAMES:
                    arrays["m{}/{}".format(k, name)] = archive["m{}/{}".format(m, name)]
                k += 1
    arrays["n_models"] = np.int64(k)
    arrays["feature_version"] = np.str_(version)
    np.savez(out, **arrays)
    learned.load_weights(out)


def cmd_run(ctx):
    games = selected_games(ctx.args)
    jobs = raw_jobs(ctx, games)
    run_all(ctx, jobs)
    log("run: {} runs present".format(len(jobs) - len(missing_runs(jobs))))
    return 0


def score_raw(ctx, games):
    """{game: {system: {ref: {shift: score}}}} for the raw-view runs, and the reference md5s."""
    jobs = raw_jobs(ctx, games)
    missing = missing_runs(jobs)
    if missing:
        raise SystemExit("runs missing (run `run` first):\n  " + "\n  ".join(missing))
    scores, ref_md5 = {}, {}
    for game in games:
        refs = references(ctx, game)
        ref_md5[game] = refs["provided_md5"]
        scores[game] = {}
        for label in RUN_SYSTEMS:
            job = jobs[(game, label)]
            row = {}
            for ref in ("ctl2", "provided"):
                shifts = SHIFTS if label in ("c22", "M", "H") else (0,)
                row[ref] = {str(s): score_csv(ctx, job, refs[ref], s) for s in shifts}
            scores[game][label] = row
    return scores, ref_md5, jobs


def cmd_score(ctx):
    games = selected_games(ctx.args)
    scores, ref_md5, jobs = score_raw(ctx, games)
    rec = {"games": games, "complete": sorted(games) == sorted(folds.ALL) and not ctx.smoke,
           "smoke": ctx.smoke, "scores": scores, "reference_md5": ref_md5,
           "summaries": summaries(scores, games),
           "costs": {"fold": {str(k): fold_costs(ctx, k) for k in sorted({fold_of(g)
                                                                          for g in games})},
                     "median": final_costs(ctx)},
           "runs": {"{}|{}".format(*k): {"key": j["key"], "weights_md5": j["weights_md5"]}
                    for k, j in jobs.items()},
           "engine_source_hash": engine_hash(),
           "conventions": {"bootstrap": "provider-stratified over matches, RandomState(0), "
                                        "1000 draws, 5-95 %",
                           "shift": "reference label of frame t moved to t + s; edge frames X"}}
    write_json(os.path.join(ctx.work, "scores.json"), rec)
    print_summaries(rec["summaries"])
    return 0


def print_summaries(rows):
    for key in sorted(rows):
        r = rows[key]
        print("{:<26} X {:.4f}  c22 {:.4f}  delta {:+.4f} [{:+.4f}, {:+.4f}]".format(
            key, r["x_mean"], r["c22_mean"], r["delta"], r["low"], r["high"]))


def d_share_hidden(hidden, ref_provided, codes):
    """(frames, frames labelled D) of the hidden reference-alive frames."""
    sel = np.asarray(hidden, dtype=bool) & np.isin(ref_provided, (0, 1))
    return int(sel.sum()), int(np.sum(np.asarray(codes)[sel] == 2))


def pooled_share(counts):
    """{provider: share} from {game: (frames, d)}."""
    out = {}
    for p, names in by_provider(counts).items():
        frames = sum(counts[g][0] for g in names)
        d = sum(counts[g][1] for g in names)
        out[p] = d / float(frames) if frames else None
    return out


def drops(unstressed, stressed):
    """{"balanced", "per_provider"}: unstressed mean minus stressed mean of one system."""
    diff = {g: unstressed[g] - stressed[g] for g in unstressed}
    per = {p: float(np.mean([diff[g] for g in names])) for p, names in by_provider(diff).items()}
    rel = {p: (float(np.mean([diff[g] for g in names]))
               / float(np.mean([unstressed[g] for g in names])))
           for p, names in by_provider(diff).items()}
    return {"balanced": balanced_mean(diff), "per_provider": per, "relative_per_provider": rel}


def cmd_stress(ctx):
    import score_game as sg
    games = selected_games(ctx.args)
    base = raw_jobs(ctx, games, systems=STRESS_SYSTEMS)
    sjobs = stress_jobs(ctx, games)
    run_all(ctx, dict(list(base.items()) + list(sjobs.items())))
    missing = missing_runs(sjobs) + missing_runs(base)
    if missing:
        raise SystemExit("runs missing:\n  " + "\n  ".join(missing))
    unstressed = {s: {} for s in STRESS_SYSTEMS}
    stressed = {s: {} for s in STRESS_SYSTEMS}
    dcount = {v: {s: {} for s in STRESS_SYSTEMS} for v in (VIEW_STRESS, VIEW_U30)}
    mask_same = {v: {} for v in (VIEW_STRESS, VIEW_U30)}
    for game in games:
        refs = references(ctx, game)
        n = len(refs["provided"])
        for s in STRESS_SYSTEMS:
            unstressed[s][game] = score_csv(ctx, base[(game, s)], refs["ctl2"])["spell_f1"]
            stressed[s][game] = score_csv(ctx, sjobs[(game, VIEW_STRESS, s)],
                                          refs["ctl2"])["spell_f1"]
        for view in (VIEW_STRESS, VIEW_U30):
            masks = []
            for s in STRESS_SYSTEMS:
                job = sjobs[(game, view, s)]
                with np.load(job["hidden"]) as npz:
                    hidden = np.asarray(npz["hidden"], dtype=bool)
                masks.append(hidden)
                codes = sg.read_codes(job["csv"], n)
                dcount[view][s][game] = d_share_hidden(hidden, refs["provided"], codes)
            mask_same[view][game] = bool(all(np.array_equal(masks[0], m) for m in masks[1:]))
    rec = {"games": games, "complete": sorted(games) == sorted(folds.ALL) and not ctx.smoke,
           "smoke": ctx.smoke, "unstressed": unstressed, "stressed": stressed,
           "summaries": {}, "drops": {s: drops(unstressed[s], stressed[s])
                                      for s in STRESS_SYSTEMS},
           "d_hidden": {v: {s: {"per_match": {g: list(c) for g, c in dcount[v][s].items()},
                                "per_provider": pooled_share(dcount[v][s])}
                            for s in STRESS_SYSTEMS} for v in dcount},
           "hidden_mask_identical": mask_same,
           "runs": {"|".join(k): {"key": j["key"]} for k, j in sjobs.items()}}
    for cand in CANDIDATES:
        rec["summaries"][cand] = {
            "stressed": delta_summary(stressed[cand], stressed["c22"]),
            "unstressed": delta_summary(unstressed[cand], unstressed["c22"])}
    write_json(os.path.join(ctx.work, "stress.json"), rec)
    for cand in CANDIDATES:
        s, u = rec["summaries"][cand]["stressed"], rec["summaries"][cand]["unstressed"]
        print("{} stressed delta {:+.4f} [{:+.4f}, {:+.4f}]  unstressed {:+.4f}".format(
            cand, s["delta"], s["low"], s["high"], u["delta"]))
    for s in STRESS_SYSTEMS:
        print("drop {:<4} {:+.4f}  D|hidden alive (uniform30) {}".format(
            s, rec["drops"][s]["balanced"], rec["d_hidden"][VIEW_U30][s]["per_provider"]))
    return 0


def member_frames_pf(nets, tactical, game, pf):
    """Round 6 (variant R6E): (seed members, copy members) as frame probabilities (N, 3), the
    label-free stability members of the candidate network + per-frame member.

    The network runs as engine.learned does with LEARNED_PF on (the member's swap mode:
    "columns", the symmetric copies of transform(); "tracking", predict_ensemble_paired's copies
    from the swapped tracking). A seed member is the per-frame member's frame
    probabilities blended (engine.learned.blend_pf, weight pf["weight"]) with ONE network seed
    (that model's mean over its four copies, rounded, upsampled); a copy member is the member
    blended with ONE symmetric copy of the network (that copy's mean over the seeds, rounded,
    upsampled). Copies in engine.learned.SYMMETRIES order (H and A exchanged back)."""
    from engine import learned, learned_pf
    model = learned_pf.load_pf(pf["weights"])
    f = learned.features_for(nets, tactical, game)
    n = int(tactical["n_frames"])
    p_pf = learned_pf.frame_probabilities_pf(tactical, game, model)
    w = float(pf["weight"])
    tracking = model.get("swap_mode", "columns") == "tracking"
    if tracking:
        g = learned.features_for(nets, learned_pf.swapped_tactical(tactical), game)
        copies = np.stack([f, learned.lf.transform(f, mirror=True), g,
                           learned.lf.transform(g, mirror=True)]).astype(np.float64)
    else:
        copies = learned.symmetric_copies(f)       # SYMMETRIES order, as the engine
    per = []                                       # (models, 4, bins, 3), H and A as given
    for net in nets:
        if tracking:
            p = learned.softmax(learned.forward(net, copies))
            p[2] = p[2][:, [1, 0, 2]]
            p[3] = p[3][:, [1, 0, 2]]
        else:
            p = learned.copy_probabilities(net, copies)
        per.append(p)
    per = np.stack(per)

    def frames(p5):
        return learned.blend_pf(learned.upsample(np.round(p5, learned.PROB_DECIMALS), n), p_pf, w)
    if tracking:
        seeds = [frames(((per[i, 0] + per[i, 2]) + (per[i, 1] + per[i, 3])) / 4.0)
                 for i in range(per.shape[0])]
    else:
        seeds = [frames(per[i].sum(axis=0) / 4.0) for i in range(per.shape[0])]
    copies_out = [frames(per[:, c].sum(axis=0) / per.shape[0]) for c in range(4)]
    return seeds, copies_out


def member_probs(nets, tactical, game):
    """float64 (models, 4 copies, bins, 3): the probabilities of each model and copy (numpy).

    The features are those of the models' feature version (engine.learned.features_for()).
    """
    from engine import learned
    feats = learned.features_for(nets, tactical, game)
    copies = learned.symmetric_copies(feats)
    return np.stack([learned.copy_probabilities(net, copies) for net in nets])


def dump_probs(ctx, tag, game, weights, load, pf=None):
    """Write seed_{i}.npz and copy_{c}.npz (p5) of one match. Return (seed paths, copy paths).

    pf (round 6, variant R6E: {"weights", "weight"}): the members of member_frames_pf(), stored
    as frame probabilities under the key "p25" (bar_report.prob_group reads either key)."""
    from engine import learned
    folder = os.path.join(ctx.work, "probs", tag, game)
    rec_path = os.path.join(folder, "record.json")
    md5 = ctx.weights_md5(weights)
    if pf:
        md5 = "{}+pf{}w{:g}".format(md5, ctx.weights_md5(pf["weights"]), float(pf["weight"]))
    seeds = [os.path.join(folder, "seed_{}.npz".format(i)) for i in SEEDS]
    copies = [os.path.join(folder, "copy_{}.npz".format(c)) for c in range(4)]
    if os.path.isfile(rec_path) and read_json(rec_path).get("weights_md5") == md5:
        return seeds, copies
    os.makedirs(folder, exist_ok=True)
    log("probabilities {} {}".format(tag, game))
    nets = learned.load_weights(weights)
    if len(nets) != len(SEEDS):
        raise SystemExit("{}: {} models, expected {}".format(weights, len(nets), len(SEEDS)))
    if pf:
        seed_f, copy_f = member_frames_pf(nets, load(game), game, pf)
        for i, path in enumerate(seeds):
            np.savez(path, p25=seed_f[i])
        for c, path in enumerate(copies):
            np.savez(path, p25=copy_f[c])
        write_json(rec_path, {"weights": weights, "weights_md5": md5, "game": game,
                              "pf": pf})
        return seeds, copies
    probs = member_probs(nets, load(game), game)
    for i, path in enumerate(seeds):
        np.savez(path, p5=probs[i].mean(axis=0))
    for c, path in enumerate(copies):
        np.savez(path, p5=probs[:, c].mean(axis=0))
    write_json(rec_path, {"weights": weights, "weights_md5": md5, "game": game})
    return seeds, copies


def bar_run(ctx, label, over):
    """engine.run on the competition match. Its summary goes to a log file, not to stdout."""
    from analysis import external_sweep as xs
    resolved, _ = xs.resolve_config(over)
    weights = over.get("LEARNED_WEIGHTS")
    key_obj = {"game": COMPETITION, "config": resolved, "engine": engine_hash(),
               "weights_md5": ctx.weights_md5(weights) if weights else None}
    key = hashlib.sha1(json.dumps(key_obj, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    folder = os.path.join(ctx.work, "bar", "runs")
    csv = os.path.join(folder, "{}_{}.csv".format(label, key))
    if os.path.isfile(csv):
        return csv
    os.makedirs(folder, exist_ok=True)
    base = csv[:-4]
    write_json(base + ".config.json", over)
    env = dict(os.environ)
    env.pop("MTAH_DATA_DIR", None)
    log("run {} {}".format(COMPETITION, label))
    with open(base + ".log", "w") as out:
        code = subprocess.call([sys.executable, "-m", "engine.run", "--out", base + ".tmp.csv",
                                "--config", base + ".config.json"], cwd=ROOT, env=env,
                               stdout=out, stderr=subprocess.STDOUT)
    if code != 0:
        raise RuntimeError("engine.run failed on {} {}, see {}.log".format(
            COMPETITION, label, base))
    os.replace(base + ".tmp.csv", csv)
    return csv


def ext_loader(ctx):
    """game -> the tactical dict the probability dumps read. BAR-ATH reads the ATD of the engine,
    with the C9 goalkeeper guard (possession.engine_tactical(), R2 Section 9, D1)."""
    from engine import data
    from engine.possession import engine_tactical
    from analysis import external_sweep as xs

    def load(game):
        if game == COMPETITION:
            return engine_tactical(None, game)
        folds.check(game)
        root = xs.data_root(game, "raw", *roots(ctx.args))
        return data.load_tactical(game, data_dir=root, cache_dir=data_cache(ctx, game))
    return load


def cmd_bar(ctx):
    import argparse as ap
    from engine import data
    from tools.learned_spike import bar_report as br
    args = ctx.args
    final = os.path.abspath(args.final_npz)
    for path in (final, ctx.folds_all()):
        if not os.path.isfile(path):
            raise SystemExit("{} is missing: run `export` first".format(path))
    if not args.feature_cache:
        raise SystemExit("bar needs --feature-cache (Section 8c)")
    med = final_costs(ctx)
    preds = {"c22": bar_run(ctx, "c22", {})}
    for cand, weights, label in (("M", final, "M"), ("H", final, "H"),
                                 ("M", ctx.folds_all(), "Mfold"),
                                 ("H", ctx.folds_all(), "Hfold")):
        preds[label] = bar_run(ctx, label, overrides(cand, weights, med))
    load = ext_loader(ctx)
    bar_seeds, bar_copies = dump_probs(ctx, "final", COMPETITION, final, load)
    sc_rates = {"final": {}, "fold": {}}
    for game in folds.SC:
        n = data.frame_count(game)
        graded = br.graded_mask(br.graded_runs(game), n)
        for tag, weights in (("final", final), ("fold", ctx.npz(fold_of(game)))):
            seeds, copies = dump_probs(ctx, tag, game, weights, load)
            sc_rates[tag][game] = {"seed": br.prob_group(seeds, graded)["share_any_disagree"],
                                   "copy": br.prob_group(copies, graded)["share_any_disagree"]}
    # Envelope references: as provided (checked) and the 2 s view (recorded).
    ext_provided, ext_ctl2 = [], {}
    for game in folds.ALL:
        refs = references(ctx, game)
        ext_provided += ["M={}:{}".format(game, refs["path"]), "H={}:{}".format(game, refs["path"])]
        ext_ctl2[game] = refs["ctl2"]
    ns = ap.Namespace(pred=["{}={}".format(k, preds[k]) for k in BAR_SYSTEMS], restart_ref="c22",
                      ext=ext_provided,
                      seed_probs=["s{}={}".format(i, p) for i, p in enumerate(bar_seeds)],
                      copy_probs=["c{}={}".format(i, p) for i, p in enumerate(bar_copies)],
                      feature_cache=os.path.abspath(args.feature_cache),
                      feature_games=list(folds.TRAIN_ALL))      # 8c band: R2 Section 2
    report = br.build_report(ns)
    with open(os.path.join(ctx.work, "bar", "bar_report.txt"), "w") as handle:
        br.print_report(report, handle)
    write_json(os.path.join(ctx.work, "bar", "bar_report.json"), report)
    ctl2_stats = [dict(br.system_stats(c, br.graded_runs(g), None), game=g)
                  for g, c in ext_ctl2.items()]
    rec = {
        "complete": not ctx.smoke, "smoke": ctx.smoke, "final_npz": final,
        "final_npz_md5": ctx.weights_md5(final), "median_costs": med,
        "systems": {k: report["systems"][k] for k in BAR_SYSTEMS},
        "vetoes": {c: crit6(report["systems"][c]) for c in ("c22",) + CANDIDATES},
        "disagreement": {
            key: {"bar": report[key + "_disagreement"]["share_any_disagree"],
                  "bar_mean_pairwise": report[key + "_disagreement"]["mean_pairwise_disagree"],
                  "sc_mean_final": float(np.mean([r[key] for r in sc_rates["final"].values()])),
                  "sc_mean_fold": float(np.mean([r[key] for r in sc_rates["fold"].values()]))}
            for key in ("seed", "copy")},
        "sc_rates": sc_rates,
        "features": {"outside": report["features"]["outside"],
                     "train_games": report["features"]["train_games"],
                     "most_bins_outside": report["features"]["most_bins_outside"],
                     "rows": {n: report["features"]["features"][n]
                              for n in report["features"]["outside"]}},
        "envelope_provided": {c: report["envelope"][c] for c in CANDIDATES},
        "envelope_ctl2_recorded": {c: br.envelope(report["systems"][c], ctl2_stats)
                                   for c in CANDIDATES},
        "agreement_final_vs_fold": {
            "M": report["pairs"]["M vs Mfold"]["agreement"],
            "H": report["pairs"]["H vs Hfold"]["agreement"]},
        "anatomy_vs_c22": {c: report["pairs"]["c22 vs {}".format(c)] for c in CANDIDATES},
    }
    write_json(os.path.join(ctx.work, "bar.json"), rec)
    for c in ("c22",) + CANDIDATES:
        s = rec["systems"][c]
        print("{:<4} possessions {} D share {:.3f} D segments {} graded X {}  vetoes {}".format(
            c, s["possessions"], s["d_share"], s["d_segments"], s["graded_x"],
            "pass" if rec["vetoes"][c]["pass"] else "FAIL"))
    return 0


# -- round 2 vetoes V2 and V3 (criteria 9 and 10) ------------------------------------------------

def bar_hidden_run(ctx, label, over):
    """corrupt.py engine on BAR-ATH with the V2 view (R2 Section 3). Return (CSV, hidden npz).

    The run is cached under a key of the config, the engine and corrupt.py source, the weights
    md5 and the view arguments, so the step is resumable. Its summary goes to a log file.
    """
    from analysis import external_sweep as xs
    resolved, _ = xs.resolve_config(over)
    weights = over.get("LEARNED_WEIGHTS")
    key_obj = {"game": COMPETITION, "view": VIEW_V2, "config": resolved, "engine": engine_hash(),
               "weights_md5": ctx.weights_md5(weights) if weights else None,
               "corrupt": corrupt_hash(), "corrupt_args": list(VIEW_ARGS[VIEW_V2])}
    key = hashlib.sha1(json.dumps(key_obj, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    folder = os.path.join(ctx.work, "bar", VIEW_V2)
    csv = os.path.join(folder, "{}_{}.csv".format(label, key))
    hidden = csv[:-4] + ".hidden.npz"
    if os.path.isfile(csv) and os.path.isfile(hidden):
        return csv, hidden
    os.makedirs(folder, exist_ok=True)
    base = csv[:-4]
    write_json(base + ".config.json", over)
    env = dict(os.environ)
    env.pop("MTAH_DATA_DIR", None)
    command = [sys.executable, "-m", "tools.learned_spike.corrupt", "engine", "--game",
               COMPETITION, "--out", base + ".tmp.csv", "--config", base + ".config.json",
               "--hidden-out", base + ".hidden.tmp.npz",
               "--allow-competition"] + list(VIEW_ARGS[VIEW_V2])
    log("run {} {} {}".format(VIEW_V2, COMPETITION, label))
    with open(base + ".log", "w") as out:
        code = subprocess.call(command, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
    if code != 0:
        raise RuntimeError("corrupt.py engine failed on {} {}, see {}.log".format(
            COMPETITION, label, base))
    write_json(base + ".meta.json", dict(key_obj, label=label, output_md5=md5_file(
        base + ".tmp.csv")))
    os.replace(base + ".hidden.tmp.npz", hidden)
    os.replace(base + ".tmp.csv", csv)
    return csv, hidden


def ball_seed():
    """The RandomState seed of corrupt.py's ball gaps under --seed V2_SEED."""
    from tools.learned_spike import corrupt as cr
    return cr.seed_of(COMPETITION, cr.SEED_BALL, V2_SEED)


def cmd_v23(ctx):
    """V2 and V3 of R2 Section 3 on BAR-ATH with the final weights and costs (W/v23.json)."""
    import score_game as sg
    from engine import data
    from tools.learned_spike import bar_report as br
    final = os.path.abspath(ctx.args.final_npz)
    if not os.path.isfile(final):
        raise SystemExit("{} is missing: run `export` first".format(final))
    med = final_costs(ctx)
    n = data.frame_count(COMPETITION)
    runs = br.graded_runs(COMPETITION)
    graded = br.graded_mask(runs, n)
    raw, hid, masks = {}, {}, {}
    for label in V2_SYSTEMS:
        over = overrides(label, final, med)
        raw[label] = sg.read_codes(bar_run(ctx, label, over), n)
        csv, mask_path = bar_hidden_run(ctx, label, over)
        hid[label] = sg.read_codes(csv, n)
        with np.load(mask_path) as npz:
            masks[label] = np.asarray(npz["hidden"], dtype=bool)
    same = all(np.array_equal(masks["c22"], masks[s]) for s in V2_SYSTEMS)
    if not same:
        raise SystemExit("V2: the hidden-ball masks differ between systems")
    hidden = masks["c22"]
    tac = data.load_tactical(COMPETITION)
    ball_missing = ~np.isfinite(np.asarray(tac["bx"], dtype=np.float64))
    v2 = {"systems": {s: v2_counts(raw[s], hid[s], hidden, graded) for s in V2_SYSTEMS},
          "k_frames": int((hidden & graded).sum()),
          "ball_visible_graded_frames": int((graded & ~ball_missing).sum()),
          "hidden_share_of_visible": float((hidden & graded).sum()) / max(
              1, int((graded & ~ball_missing).sum())),
          "hidden_mask_identical": bool(same), "view": list(VIEW_ARGS[VIEW_V2]),
          "seed": V2_SEED, "ball_random_state": ball_seed()}
    v3 = {s: d_share_missing(raw[s], runs, ball_missing) for s in V2_SYSTEMS}
    rec = {"complete": not ctx.smoke, "smoke": ctx.smoke, "final_npz": final,
           "final_npz_md5": ctx.weights_md5(final), "median_costs": med,
           "engine_source_hash": engine_hash(), "v2": v2, "v3": v3,
           "v3_ball_missing_graded_frames": int((graded & ball_missing).sum()),
           "criteria": {c: {"9": crit9({"v2": v2, "v3": v3}, c),
                            "10": crit10({"v2": v2, "v3": v3}, c)} for c in CANDIDATES}}
    write_json(os.path.join(ctx.work, "v23.json"), rec)
    for s in V2_SYSTEMS:
        r = v2["systems"][s]
        print("{:<4} V2 flip {}  ({} of {} alive frames of K)  V3 D share, ball missing {}".format(
            s, fmt(r["flip"]), r["flipped_to_d"], r["alive_raw"], fmt(v3[s])))
    for c in CANDIDATES:
        print("{} criterion 9 {}  criterion 10 {}".format(
            c, fmt(rec["criteria"][c]["9"]["pass"]), fmt(rec["criteria"][c]["10"]["pass"])))
    return 0


# -- decide ---------------------------------------------------------------------------------------

def check_inputs(work, repro_path):
    """(inputs, missing): load every input of `decide` and list what is missing or incomplete."""
    missing, inputs = [], {}
    for name in ("scores", "stress", "bar", "v23"):
        path = os.path.join(work, name + ".json")
        if not os.path.isfile(path):
            missing.append("{} (run `{}`)".format(path, name if name != "scores" else "score"))
            continue
        rec = read_json(path)
        if rec.get("smoke"):
            missing.append("{}: smoke output, not admissible".format(path))
        elif not rec.get("complete"):
            missing.append("{}: incomplete (not all 28 matches)".format(path))
        inputs[name] = rec
    for name in ("scores", "stress"):
        rec = inputs.get(name)
        if rec is not None and sorted(rec.get("games", [])) != sorted(folds.ALL):
            missing.append("{}.json: games differ from the 28 of the protocol".format(name))
    bar, v23 = inputs.get("bar"), inputs.get("v23")
    if bar is not None:
        band = (bar.get("features") or {}).get("train_games")
        if band is None or sorted(band) != sorted(folds.TRAIN_ALL):
            missing.append("bar.json: the 8c band is not over the round 2 training set (28 + 43)")
    if bar is not None and v23 is not None:
        for key in ("final_npz_md5", "median_costs"):
            if bar.get(key) != v23.get(key):
                missing.append("bar.json and v23.json differ in {}".format(key))
    if not repro_path or not os.path.isfile(repro_path):
        missing.append("reproducibility file (criterion 8, --repro): {}".format(repro_path))
    else:
        repro = read_json(repro_path)
        if "pass" not in repro:
            missing.append("{}: no \"pass\" field".format(repro_path))
        inputs["repro"] = repro
    return inputs, missing


def evaluate_all(inputs, explanations=None):
    """{"M"|"H": {"criteria", "mean", "bounds8"}} and the Section 7 choice."""
    sm = inputs["scores"]["summaries"]
    st = inputs["stress"]
    bar = inputs["bar"]
    results = {}
    for cand in CANDIDATES:
        b8 = bounds8(cand, bar, explanations)
        crits = {
            "1": crit1(sm["{}|fold|ctl2|0".format(cand)], sm["{}|median|ctl2|0".format(cand)]),
            "2": crit2(sm["{}|fold|ctl2|0".format(cand)]),
            "3": crit3(sm["{}|fold|provided|0".format(cand)]),
            "4": crit4(st["summaries"][cand]["stressed"], st["summaries"][cand]["unstressed"],
                       st["drops"][cand]["balanced"], st["drops"]["c22"]["balanced"]),
            "5": crit5(st["d_hidden"][VIEW_U30][cand]["per_provider"],
                       st["d_hidden"][VIEW_U30]["c22"]["per_provider"]),
            "6": crit6(bar["systems"][cand]),
            "7": crit7(b8),
            "8": crit8(inputs["repro"]),
            "9": crit9(inputs["v23"], cand),
            "10": crit10(inputs["v23"], cand),
        }
        results[cand] = {"criteria": crits, "mean": sm["{}|fold|ctl2|0".format(cand)]["x_mean"],
                         "bounds8": b8}
    return results, choose(results)


def fmt(v, spec="{:.4f}"):
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "pass" if v else "FAIL"
    if isinstance(v, (int, np.integer)):
        return str(v)
    return spec.format(v)


def result_markdown(inputs, results, decision):
    sc, st, bar, v23 = inputs["scores"], inputs["stress"], inputs["bar"], inputs["v23"]
    sm = sc["summaries"]
    games = sorted(sc["games"], key=lambda g: (PROVIDERS.index(folds.provider(g)), g))
    w = []
    w.append("# LEARNED_STAGE_RESULT_R2: round 2 of the learned stage against the frozen "
             "protocol\n")
    w.append("Written by `tools/learned_spike/evaluate.py decide` from W/scores.json, "
             "W/stress.json, W/bar.json, W/v23.json and the reproducibility file. Protocols: "
             "`LEARNED_STAGE_PROTOCOL.md` and `LEARNED_STAGE_PROTOCOL_R2.md` (both frozen). "
             "Nothing here was tuned after a score was seen.\n")
    w.append("**Honesty note** (R2 Section 7): {}\n".format(HONESTY_NOTE))
    w.append("## Verdict of `decide` (before V1)\n")
    w.append("**{}** is chosen. Reason: {}.".format(decision["choice"], decision["reason"]))
    if decision.get("note"):
        w.append(" Note: {}.".format(decision["note"]))
    w.append("\n\nQualifies (criteria 1–10): M {}, H {}.\n".format(
        fmt(decision["qualifies"]["M"]), fmt(decision["qualifies"]["H"])))
    if decision["choice"] in CANDIDATES:
        w.append("A candidate ships only if it also passes V1 on the sealed PFF set (R2 "
                 "Sections 3 and 5): `evaluate.py sealed-record`, commit, then "
                 "`evaluate.py sealed`. Until then c22 stays the shipped system.\n")
    else:
        w.append("V1 does not run: c22 ships (R2 Section 5).\n")
    w.append("## Criteria (round 1 Section 5, R2 Section 3 for 9 and 10)\n")
    w.append("| # | M | H | numbers M | numbers H |\n|---|---|---|---|---|")
    for i in range(1, N_CRITERIA + 1):
        cm, ch = results["M"]["criteria"][str(i)], results["H"]["criteria"][str(i)]
        w.append("| {} | {} | {} | `{}` | `{}` |".format(
            i, fmt(cm["pass"]), fmt(ch["pass"]), json.dumps(cm["numbers"], sort_keys=True,
                                                             default=float),
            json.dumps(ch["numbers"], sort_keys=True, default=float)))
    w.append("")
    w.append("## Provider-balanced deltas (Section 4)\n")
    w.append("Bootstrap: provider-stratified over matches, 1,000 draws, seed 0, 5–95 %. "
             "Shift s: the reference label of frame t moves to t + s; edge frames are X.\n")
    w.append("| row | X mean | c22 mean | Δ | 5 % | 95 % | Metrica Δ (wins/n) | DFL Δ (wins/n) "
             "| SkillCorner Δ (wins/n) |\n|---|---|---|---|---|---|---|---|---|")
    for key in sorted(sm):
        r = sm[key]
        cells = []
        for p in PROVIDERS:
            q = r["per_provider"].get(p)
            cells.append("{} ({}/{})".format(fmt(q["delta"], "{:+.4f}"), q["wins"], q["n"])
                         if q else "–")
        w.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            key.replace("|", " / "), fmt(r["x_mean"]), fmt(r["c22_mean"]),
            fmt(r["delta"], "{:+.4f}"),
            fmt(r["low"], "{:+.4f}"), fmt(r["high"], "{:+.4f}"), " | ".join(cells)))
    w.append("\nRow key: candidate / costs (fold or grand median) / reference (ctl2 = 2 s, "
             "provided) / shift in frames.\n")
    w.append("## Per-match spellF1 (raw view)\n")
    w.append("| match | ref | c22 | M | H | M (median costs) | H (median costs) |\n"
             "|---|---|---|---|---|---|---|")
    for g in games:
        for ref in ("ctl2", "provided"):
            row = sc["scores"][g]
            w.append("| {} | {} | {} |".format(g, ref, " | ".join(
                fmt(row[s][ref]["0"]["spell_f1"]) for s in RUN_SYSTEMS)))
    w.append("\n## Stress view (criterion 4) and missing-ball D guard (criterion 5)\n")
    w.append("| system | drop balanced | Metrica | DFL | SkillCorner | relative Metrica | "
             "relative DFL | relative SkillCorner |\n|---|---|---|---|---|---|---|---|")
    for s in STRESS_SYSTEMS:
        d = st["drops"][s]
        w.append("| {} | {} | {} | {} |".format(s, fmt(d["balanced"], "{:+.4f}"), " | ".join(
            fmt(d["per_provider"].get(p), "{:+.4f}") for p in PROVIDERS), " | ".join(
            fmt(d["relative_per_provider"].get(p), "{:+.3f}") for p in PROVIDERS)))
    w.append("\nShare of hidden reference-alive frames labelled D (pooled per provider):\n")
    w.append("| view | system | Metrica | DFL | SkillCorner |\n|---|---|---|---|---|")
    for view in (VIEW_U30, VIEW_STRESS):
        for s in STRESS_SYSTEMS:
            pp = st["d_hidden"][view][s]["per_provider"]
            w.append("| {} | {} | {} |".format(view + (" (checked)" if view == VIEW_U30 else
                                                      " (recorded)"), s, " | ".join(
                fmt(pp.get(p)) for p in PROVIDERS)))
    w.append("\n## BAR-ATH, label-free (criterion 6 and Section 8)\n")
    w.append("| system | possessions | D share | D segments | graded X | D share, ball missing |"
             "\n|---|---|---|---|---|---|")
    for s in BAR_SYSTEMS:
        r = bar["systems"][s]
        w.append("| {} | {} | {} | {} | {} | {} |".format(
            s, r["possessions"], fmt(r["d_share"]), r["d_segments"], r["graded_x"],
            fmt(r.get("d_share_ball_missing"))))
    w.append("\n| bound | M | H |\n|---|---|---|")
    for part in "abcef":
        w.append("| 8{} | {} | {} |".format(part, fmt(results["M"]["bounds8"][part]["pass"]),
                                            fmt(results["H"]["bounds8"][part]["pass"])))
    w.append("\nSection 8 details: `{}`\n".format(json.dumps(
        {c: results[c]["bounds8"] for c in CANDIDATES}, sort_keys=True, default=float)))
    w.append("Seed and copy disagreement: `{}`\n".format(json.dumps(bar["disagreement"],
                                                                   sort_keys=True)))
    w.append("8c band: the 1–99 % band of the graded bins of the {} round 2 training matches "
             "(28 external + 43 PFF, R2 Section 2).\n".format(len(bar["features"]["train_games"])))
    w.append("## Vetoes V2 and V3 on BAR-ATH (criteria 9 and 10, R2 Section 3)\n")
    w.append("V2 view: `corrupt.py {}` (seed 0 replaces the crc32 of the match name; ball "
             "RandomState {}), final weights `{}`, final costs. K = {} hidden graded frames. "
             "H is exempt from both.\n".format(" ".join(v23["v2"]["view"]),
                                                 v23["v2"]["ball_random_state"],
                                                 v23["final_npz_md5"], v23["v2"]["k_frames"]))
    w.append("| system | A (H or A on raw, in K) | flipped to D | flip | V3: D share, ball "
             "missing |\n|---|---|---|---|---|")
    for s in V2_SYSTEMS:
        r = v23["v2"]["systems"][s]
        w.append("| {} | {} | {} | {} | {} |".format(s, r["alive_raw"], r["flipped_to_d"],
                                                     fmt(r["flip"]), fmt(v23["v3"][s])))
    w.append("\nBounds: flip_M ≤ flip_c22 + {:.2f}; V3_M ≤ V3_c22 + {:.2f}.\n".format(
        MAX_EXTRA_FLIP, MAX_EXTRA_D_MISSING))
    w.append("## Conventions the protocol leaves to the tool\n")
    w.append("See the docstring of `tools/learned_spike/evaluate.py`: bootstrap, reference "
             "shift, drop sign, hidden-frame mask, Section 8a/8b rate and SkillCorner mean "
             "(fold ensembles), 8e reference set (as provided), 8f fold-mean labels, transfer "
             "rule (criteria 4, 5, 7; to qualify, 1–10), V2 seed and denominator, V3 "
             "statistic, V1 bootstrap and rules.\n")
    return "\n".join(w) + "\n"


def cmd_decide(args):
    work = os.path.abspath(os.path.expanduser(args.work))
    if os.path.abspath(args.out_md) == os.path.abspath(ROUND1_RESULT_MD):
        raise SystemExit("{} is the round 1 record: round 2 writes {}".format(
            args.out_md, RESULT_MD))
    out_json = os.path.abspath(args.out_md)[:-3] + ".json"
    if os.path.isfile(out_json) and read_json(out_json).get("v1_record"):
        raise SystemExit("{} already holds the V1 record: decide does not run again after "
                         "`sealed-record`".format(out_json))
    inputs, missing = check_inputs(work, args.repro)
    if missing:
        raise SystemExit("decide refuses to run; missing inputs:\n  " + "\n  ".join(missing))
    explanations = read_json(args.explanations) if args.explanations else {}
    results, decision = evaluate_all(inputs, explanations)
    out_md = os.path.abspath(args.out_md)
    with open(out_md, "w") as handle:
        handle.write(result_markdown(inputs, results, decision))
    write_json(out_md[:-3] + ".json", {
        "decision": decision, "results": results, "explanations": explanations,
        "final_npz": inputs["bar"]["final_npz"], "final_npz_md5": inputs["bar"]["final_npz_md5"],
        "median_costs": inputs["bar"]["median_costs"],
        "engine_source_hash_v23": inputs["v23"].get("engine_source_hash"),
        "honesty_note": HONESTY_NOTE})
    print("{}: {}".format(decision["choice"], decision["reason"]))
    return 0


# -- V1: the sealed PFF set (R2 Section 3) --------------------------------------------------------

def match_bootstrap(deltas, n_draws=N_DRAWS, seed=BOOT_SEED):
    """(low, high): 5-95 % of the mean of {game: delta} under a bootstrap over matches.

    One provider (PFF_TRANSFER_TEST.md Section 4): matches sorted by name, one RandomState(seed),
    each draw resamples the matches with replacement. Equal to stratified_bootstrap() on the
    matches of a single provider.
    """
    names = sorted(deltas)
    a = np.array([deltas[g] for g in names], dtype=np.float64)
    rng = np.random.RandomState(seed)
    draws = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        draws[i] = a[rng.randint(0, a.size, a.size)].mean()
    low, high = np.percentile(draws, PERCENTILES)
    return float(low), float(high)


def win_share_ok(wins, n):
    """Rule c: wins / n >= 60 %, as the exact integer test 5 x wins >= 3 x n."""
    return V1_WIN_SHARE[1] * int(wins) >= V1_WIN_SHARE[0] * int(n)


def v1_rules(x, base, n_expected=None):
    """Rules a, b and c of PFF_TRANSFER_TEST.md Section 5 with T replaced by S (shift 0).

    x, base: {"ctl2": {game: spellF1}, "provided": {game: spellF1}}. With n_expected, a system
    that did not score every match fails (R2 Section 3: a run that does not finish S fails).
    """
    games = sorted(base["ctl2"])
    complete = (sorted(x["ctl2"]) == games and sorted(x["provided"]) == games
                and sorted(base["provided"]) == games
                and (n_expected is None or len(games) == n_expected))
    if not complete or not games:
        return {"complete": False, "pass": False, "a": False, "b": False, "c": False}
    d2 = {g: x["ctl2"][g] - base["ctl2"][g] for g in games}
    dp = {g: x["provided"][g] - base["provided"][g] for g in games}
    lo2, hi2 = match_bootstrap(d2)
    lop, hip = match_bootstrap(dp)
    mean2, meanp = float(np.mean(list(d2.values()))), float(np.mean(list(dp.values())))
    wins = int(sum(v > 0 for v in d2.values()))
    a = lo2 > 0
    b = meanp > 0 and lop > V1_RAW_FLOOR
    c = win_share_ok(wins, len(games))
    return {"complete": True, "n": len(games),
            "ctl2": {"delta": mean2, "low": lo2, "high": hi2},
            "provided": {"delta": meanp, "low": lop, "high": hip},
            "wins": wins, "win_share": wins / float(len(games)),
            "a": bool(a), "b": bool(b), "c": bool(c), "pass": bool(a and b and c),
            "per_match": {g: {"ctl2": d2[g], "provided": dp[g]} for g in games}}


def v1_verdict(choice, qualifies_h, passes):
    """The V1 outcome. passes: {"M"|"H": bool or None (not run)}. Return (ships, reason, run_h).

    run_h is True when H must run (X = M failed and H qualified). Call again with passes["H"].
    """
    if choice not in CANDIDATES:
        raise SystemExit("V1 runs only after decide chose M or H (choice: {})".format(choice))
    if passes.get(choice):
        return choice, "{} passes V1 a, b and c".format(choice), False
    if choice == "M" and qualifies_h:
        if passes.get("H") is None:
            return None, "M fails V1; H qualified, so H runs on S", True
        if passes["H"]:
            return "H", "M fails V1; H qualified and passes V1 (sequential fallback)", False
        return "c22", "M fails V1; H qualified but fails V1", False
    if choice == "M":
        return "c22", "M fails V1 and H did not qualify", False
    return "c22", "H fails V1 (X = H): c22 ships", False


def git_out(*args):
    try:
        return subprocess.check_output(("git",) + args, cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# The engine code that the V1 record pins by commit. The weights are pinned by their md5, so an
# untracked or newly committed engine/models/learned_v2.npz does not count as an engine change.
ENGINE_PATHS = ("engine/*.py", "games.py")


def v1_paths(args):
    result_json = os.path.abspath(args.result_md)[:-3] + ".json"
    return os.path.abspath(args.result_md), result_json


def load_result(args):
    """The decide result: refuse without it or without a chosen M or H."""
    result_md, result_json = v1_paths(args)
    if not os.path.isfile(result_json):
        raise SystemExit("{} is missing: V1 runs only after `decide`".format(result_json))
    rec = read_json(result_json)
    choice = (rec.get("decision") or {}).get("choice")
    if choice not in CANDIDATES:
        raise SystemExit("{}: decide chose {}, so V1 does not run (c22 ships)".format(
            result_json, choice))
    return rec


def v1_current(args, stage):
    """The md5 of the final weights, the final costs, the commit and engine source now."""
    final = os.path.abspath(args.final_npz)
    if not os.path.isfile(final):
        raise SystemExit("{} is missing".format(final))
    from analysis import external_sweep as xs
    costs_path = os.path.join(stage, "final", "costs.json")
    if not os.path.isfile(costs_path):
        raise SystemExit("{} is missing".format(costs_path))
    return {"final_npz": final, "final_npz_md5": md5_file(final),
            "final_costs": costs_record(costs_path, None), "final_costs_md5": md5_file(costs_path),
            "engine_commit": git_out("rev-parse", "HEAD"),
            "engine_dirty": bool(git_out("status", "--porcelain", "--untracked-files=no", "--",
                                         *ENGINE_PATHS)),
            "engine_source_hash": xs.engine_source_hash()}


def cmd_sealed_record(args):
    """Write the md5 of learned_v2.npz and the engine commit into the result files (V1 first)."""
    rec = load_result(args)
    stage = os.path.abspath(os.path.expanduser(args.stage))
    now = v1_current(args, stage)
    if now["engine_dirty"]:
        raise SystemExit("engine/*.py or games.py has uncommitted changes: commit first, so "
                         "the recorded commit names the code that runs")
    if now["final_npz_md5"] != rec.get("final_npz_md5"):
        raise SystemExit("{} md5 {} differs from the weights that decide judged ({})".format(
            now["final_npz"], now["final_npz_md5"], rec.get("final_npz_md5")))
    costs = now["final_costs"]
    if [list(costs["M"]), costs["H"]] != [[rec["median_costs"]["M"][0],
                                           rec["median_costs"]["M"][1]],
                                          rec["median_costs"]["H"]]:
        raise SystemExit("the final costs differ from the costs that decide judged")
    record = {"final_npz": os.path.relpath(now["final_npz"], ROOT)
              if now["final_npz"].startswith(ROOT) else now["final_npz"],
              "final_npz_md5": now["final_npz_md5"], "engine_commit": now["engine_commit"],
              "engine_source_hash": now["engine_source_hash"],
              "final_costs": {"M": list(costs["M"]), "H": costs["H"]},
              "final_costs_md5": now["final_costs_md5"]}
    if rec.get("v1_record"):
        if rec["v1_record"] != record:
            raise SystemExit("the result file already holds a different V1 record:\n{}".format(
                json.dumps(rec["v1_record"], sort_keys=True)))
        print("V1 record already present and identical")
        return 0
    result_md, result_json = v1_paths(args)
    rec["v1_record"] = record
    write_json(result_json, rec)
    with open(result_md, "a") as handle:
        handle.write("\n## V1 record, written before the first run on the sealed set\n\n"
                     "`learned_v2.npz` md5 `{}`; engine commit `{}` (engine source hash `{}`); "
                     "final costs `{}`. X = {} and c22 run on S with this commit "
                     "(R2 Section 3, PFF_TRANSFER_TEST.md Section 3).\n".format(
                         record["final_npz_md5"], record["engine_commit"],
                         record["engine_source_hash"], json.dumps(record["final_costs"]),
                         rec["decision"]["choice"]))
    print("V1 record written to {} and {}. Commit both files, then run `sealed`.".format(
        result_md, result_json))
    return 0


def check_v1_record(rec, now, commit_has_engine):
    """[text] of every difference between the V1 record and the code and files now."""
    record = rec.get("v1_record")
    if not record:
        return ["no V1 record in the result file: run `sealed-record` first"]
    bad = []
    if now["final_npz_md5"] != record["final_npz_md5"]:
        bad.append("weights md5 {} differs from the recorded {}".format(
            now["final_npz_md5"], record["final_npz_md5"]))
    if now["final_costs_md5"] != record["final_costs_md5"]:
        bad.append("final costs.json differs from the recorded one")
    if now["engine_source_hash"] != record["engine_source_hash"]:
        bad.append("engine source hash {} differs from the recorded {}".format(
            now["engine_source_hash"], record["engine_source_hash"]))
    if now["engine_dirty"]:
        bad.append("engine/ or games.py has uncommitted changes")
    if not commit_has_engine:
        bad.append("engine/ or games.py differs from the recorded commit {}".format(
            record["engine_commit"]))
    return bad


def sealed_job(work, game, label, system, weights, weights_md5, costs, refs_root):
    """The engine run of one system on one sealed PFF match (raw view)."""
    from analysis import external_sweep as xs
    over = overrides(system, weights, costs)
    resolved, _ = xs.resolve_config(over)
    key_obj = {"game": game, "view": VIEW_RAW, "config": resolved, "data_dir": refs_root,
               "engine": engine_hash(), "weights_md5": weights_md5 if system != "c22" else None}
    key = hashlib.sha1(json.dumps(key_obj, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    where = os.path.join(work, "sealed", "runs", game)
    return {"game": game, "system": label, "overrides": over, "data_dir": refs_root, "key": key,
            "csv": os.path.join(where, "{}_{}.csv".format(label, key))}


def sealed_run(work, job):
    """external_sweep --engine-child on one sealed match, unless the CSV exists."""
    from analysis import external_sweep as xs
    if os.path.isfile(job["csv"]):
        return job["csv"]
    base = job["csv"][:-4]
    os.makedirs(os.path.dirname(base), exist_ok=True)
    write_json(base + ".config.json", job["overrides"])
    cache = os.path.join(work, "sealed", "engine_data")
    os.makedirs(cache, exist_ok=True)
    command = [sys.executable, os.path.abspath(xs.__file__), "--engine-child", cache, "--game",
               job["game"], "--config", base + ".config.json", "--out", base + ".tmp.csv"]
    log("run sealed {} {} ({})".format(job["game"], job["system"], job["key"]))
    with open(base + ".log", "w") as out:
        code = subprocess.call(command, cwd=ROOT, env=dict(os.environ,
                                                            MTAH_DATA_DIR=job["data_dir"]),
                               stdout=out, stderr=subprocess.STDOUT)
    if code != 0:
        raise RuntimeError("engine.run failed on {} {}, see {}.log".format(
            job["game"], job["system"], base))
    write_json(base + ".meta.json", dict(job, output_md5=md5_file(base + ".tmp.csv")))
    os.replace(base + ".tmp.csv", job["csv"])
    return job["csv"]


def sealed_references(game, refs_root):
    """{"provided", "ctl2", "md5"} of a sealed PFF match (md5 checked against R2 Section 8)."""
    import possession_metrics as pm
    from analysis import control_reference as cref
    path, digest = pff.reference(folds.check_sealed_v1(game), refs_root, sealed=True)
    provided = pm.label_codes(pm.read_label_csv(path)).astype(np.int8)
    labels = np.asarray(pm.LABELS)[provided]
    return {"provided": provided, "ctl2": label_codes(cref.control_labels(labels, CONTROL_FRAMES)),
            "md5": digest}


def sealed_system(work, label, system, weights, weights_md5, costs, refs_root, refs):
    """Run and score one system on S. Return {"scores", "stats", "complete", "error"}."""
    import score_game as sg
    from tools.learned_spike import bar_report as br
    scores = {"ctl2": {}, "provided": {}}
    shifted = {str(s): {"ctl2": {}, "provided": {}} for s in SHIFTS}
    stats, error = {}, None
    for game in folds.PFF_SEALED:
        job = sealed_job(work, game, label, system, weights, weights_md5, costs, refs_root)
        try:
            csv = sealed_run(work, job)
        except RuntimeError as exc:
            error = str(exc)
            log(error)
            break
        ref = refs[game]
        pred = sg.read_codes(csv, len(ref["provided"]))
        for kind in ("ctl2", "provided"):
            for shift in SHIFTS:
                r = shift_codes(ref[kind], shift) if shift else ref[kind]
                value = float(sg.score_codes(r, pred)[0]["spell_f1_mean"])
                shifted[str(shift)][kind][game] = value
                if shift == 0:
                    scores[kind][game] = value
        st = br.system_stats(pred, br.graded_runs(game), None)
        stats[game] = {"possessions": st["possessions"], "d_share": st["d_share"],
                       "graded_frames": int(br.graded_mask(br.graded_runs(game),
                                                           len(pred)).sum()),
                       "csv_md5": md5_file(csv)}
    complete = error is None and len(scores["ctl2"]) == len(folds.PFF_SEALED)
    return {"scores": scores, "shifted": shifted, "stats": stats, "complete": complete,
            "error": error}


def sealed_band_diagnostic(cache, refs_root):
    """PFF_TRANSFER_TEST.md Section 4, recorded only: per sealed match, the share of feature
    medians (graded bins, raw view) inside the 1-99 % band of the round 2 training set."""
    from engine import data
    from tools.learned_spike import bar_report as br
    from tools.learned_spike import build, features
    version = build.folder_version(cache)     # the band's feature version (round 5: stage-v3)
    lo, hi, used = br.training_band(cache, list(folds.TRAIN_ALL))
    out = {}
    for game in folds.PFF_SEALED:
        tac = data.load_tactical(folds.check_sealed_v1(game), data_dir=refs_root, use_cache=False)
        f5 = features.downsample(features.frame_features(tac, game, version=version)).astype(
            np.float64)
        med = np.median(f5[br.graded_bins(game, f5.shape[0])], axis=0)
        inside = (med >= lo) & (med <= hi)
        out[game] = {"share_inside": float(inside.mean()),
                     "outside": [n for (n, _), ok in zip(features.feature_spec(version), inside)
                                 if not ok]}
    return {"train_games": len(used), "per_match": out,
            "mean_share_inside": float(np.mean([r["share_inside"] for r in out.values()]))}


def pooled_d(stats):
    frames = sum(s["graded_frames"] for s in stats.values())
    return sum(s["d_share"] * s["graded_frames"] for s in stats.values()) / float(frames) \
        if frames else None


def cmd_sealed(args):
    """V1 of R2 Section 3: X and c22 (and H under the fallback) on the 15 sealed PFF matches."""
    import datetime
    print("V1 started {} (Gate 3: V1, fallback included, finishes by {}).".format(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip(), V1_GATE))
    rec = load_result(args)
    stage = os.path.abspath(os.path.expanduser(args.stage))
    work = os.path.abspath(os.path.expanduser(args.work))
    if os.path.commonpath([work, ROOT]) == ROOT:
        raise SystemExit("--work must be outside the repository")
    out_path = os.path.join(work, "sealed.json")
    if os.path.isfile(out_path) and read_json(out_path).get("verdict"):
        print("{} already holds the V1 verdict: {}".format(out_path,
                                                         read_json(out_path)["verdict"]))
        return 0
    now = v1_current(args, stage)
    record = rec.get("v1_record") or {}
    same_commit = bool(record.get("engine_commit")) and subprocess.call(
        ["git", "diff", "--quiet", record["engine_commit"], "--"] + list(ENGINE_PATHS),
        cwd=ROOT) == 0
    bad = check_v1_record(rec, now, same_commit)
    if bad:
        raise SystemExit("sealed refuses to run:\n  " + "\n  ".join(bad))
    refs_root = os.path.abspath(os.path.expanduser(args.refs_root))
    pff.check_references(refs_root, folds.PFF_SEALED)
    refs = {g: sealed_references(g, refs_root) for g in folds.PFF_SEALED}
    choice = rec["decision"]["choice"]
    qualifies_h = bool(rec["decision"]["qualifies"].get("H"))
    costs = {"M": tuple(record["final_costs"]["M"]), "H": float(record["final_costs"]["H"])}
    weights, wmd5 = now["final_npz"], now["final_npz_md5"]
    systems = {}
    for label in ("c22", choice):
        systems[label] = sealed_system(work, label, label, weights, wmd5, costs, refs_root, refs)
    n = len(folds.PFF_SEALED)
    rules = {}

    def judge(label):
        if not systems["c22"]["complete"]:
            return {"complete": False, "pass": False, "error": "c22 did not finish S"}
        if not systems[label]["complete"]:
            return {"complete": False, "pass": False, "error": systems[label]["error"]}
        return v1_rules(systems[label]["scores"], systems["c22"]["scores"], n)

    rules[choice] = judge(choice)
    ships, reason, run_h = v1_verdict(choice, qualifies_h, {choice: rules[choice]["pass"]})
    if run_h:
        systems["H"] = sealed_system(work, "H", "H", weights, wmd5, costs, refs_root, refs)
        rules["H"] = judge("H")
        ships, reason, _ = v1_verdict(choice, qualifies_h, {choice: rules[choice]["pass"],
                                                            "H": rules["H"]["pass"]})
    recorded = {}
    for label in systems:
        if label == "c22" or not systems[label]["complete"] or not systems["c22"]["complete"]:
            continue
        recorded[label] = {s: {kind: float(np.mean([systems[label]["shifted"][s][kind][g]
                                                    - systems["c22"]["shifted"][s][kind][g]
                                                    for g in folds.PFF_SEALED]))
                               for kind in ("ctl2", "provided")} for s in systems[label]["shifted"]}
    c22_mean = {k: float(np.mean(list(systems["c22"]["scores"][k].values())))
                for k in ("ctl2", "provided")} if systems["c22"]["complete"] else None
    ref_pooled = {}
    for g in folds.PFF_SEALED:
        from tools.learned_spike import bar_report as br
        st = br.system_stats(refs[g]["provided"], br.graded_runs(g), None)
        ref_pooled[g] = {"d_share": st["d_share"], "possessions": st["possessions"],
                         "graded_frames": int(br.graded_mask(br.graded_runs(g),
                                                             len(refs[g]["provided"])).sum())}
    out = {
        "verdict": {"ships": ships, "reason": reason}, "choice": choice,
        "qualifies_h": qualifies_h, "rules": rules, "matches": list(folds.PFF_SEALED),
        "reference_md5": {g: refs[g]["md5"] for g in folds.PFF_SEALED},
        "v1_record": record, "engine_source_hash": now["engine_source_hash"],
        "engine_commit_now": now["engine_commit"],
        "systems": {k: {"complete": v["complete"], "error": v["error"], "scores": v["scores"],
                        "stats": v["stats"], "pooled_d_share": pooled_d(v["stats"])}
                    for k, v in systems.items()},
        "recorded": {"mean_delta_by_shift": recorded, "c22_mean_on_S": c22_mean,
                     "c22_mean_on_P_quoted": PILOT_C22_MEAN,
                     "c22_gap_P_vs_S_flag": None if c22_mean is None else bool(
                         abs(c22_mean["ctl2"] - PILOT_C22_MEAN["ctl2"]) > MAX_PILOT_GAP),
                     "reference_pooled_d_share": pooled_d(ref_pooled),
                     "reference_stats": ref_pooled,
                     "band_8c_diagnostic": sealed_band_diagnostic(
                         os.path.abspath(os.path.expanduser(args.feature_cache)), refs_root)
                     if args.feature_cache else None},
        "conventions": {"bootstrap": "over matches, sorted, RandomState(0), 1000 draws, 5-95 %",
                        "rules": "a: 2 s low > 0; b: provided mean > 0 and low > -0.010; "
                                 "c: 5 x wins >= 3 x 15", "shift": 0, "gate": V1_GATE},
    }
    if ships is None:
        raise RuntimeError("internal: no V1 verdict")
    write_json(out_path, out)
    print("V1: {} ships ({}).".format(ships, reason))
    for label, r in sorted(rules.items()):
        if r.get("complete"):
            print("{}: 2 s delta {:+.4f} [{:+.4f}, {:+.4f}]  provided {:+.4f} [{:+.4f}]  wins "
                  "{}/{}  a {} b {} c {}".format(
                      label, r["ctl2"]["delta"], r["ctl2"]["low"], r["ctl2"]["high"],
                      r["provided"]["delta"], r["provided"]["low"], r["wins"], r["n"],
                      fmt(r["a"]), fmt(r["b"]), fmt(r["c"])))
        else:
            print("{}: did not finish S ({})".format(label, r.get("error")))
    print("V1 finished {} (Gate 3: {}).".format(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), V1_GATE))
    return 0


# -- CLI ------------------------------------------------------------------------------------------

DEFAULT_REFS_ROOT = os.path.join("~", "mtah2027", "refs")             # external_sweep defaults
DEFAULT_SAMPLE_ROOT = os.path.join("~", "mtah2027", "sample-data", "data")
DEFAULT_MTAH_ROOT = os.path.join("~", "mtah2027")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--work", required=True, help="work folder outside the repository")
    common.add_argument("--stage", required=True, help="run_stage.sh output folder")
    common.add_argument("--smoke-costs", type=float, nargs=3, metavar=("HA", "AD", "HA_H"),
                        help="smoke test only: costs where a costs.json is missing")
    runs = argparse.ArgumentParser(add_help=False)
    runs.add_argument("--games", nargs="+", help="restrict to these matches (smoke test)")
    runs.add_argument("--jobs", type=int, default=1, help="engine processes at a time")
    runs.add_argument("--engine-cache", help="engine.data cache of the open references")
    runs.add_argument("--sc-engine-cache", help="engine.data cache of SkillCorner")
    runs.add_argument("--ref-cache", help="folder with refs/METRICA-*_ref.csv (c22.py --ext)")
    runs.add_argument("--c22-ext", help="c22.py cache: reuse its c22 runs")
    runs.add_argument("--refs-root", default=DEFAULT_REFS_ROOT)
    runs.add_argument("--sample-root", default=DEFAULT_SAMPLE_ROOT)
    runs.add_argument("--mtah-root", default=DEFAULT_MTAH_ROOT)
    p = sub.add_parser("export", parents=[common])
    p.add_argument("--final-out", help="the final weights file (engine/models/learned_v1.npz)")
    sub.add_parser("run", parents=[common, runs])
    sub.add_parser("score", parents=[common, runs])
    sub.add_parser("stress", parents=[common, runs])
    p = sub.add_parser("bar", parents=[common, runs])
    p.add_argument("--final-npz", default=FINAL_NPZ)
    p.add_argument("--feature-cache", help="build.py cache with feats/raw/<game>.npz (71 matches)")
    p = sub.add_parser("v23", parents=[common, runs], help="vetoes V2 and V3 (criteria 9, 10)")
    p.add_argument("--final-npz", default=FINAL_NPZ)
    p = sub.add_parser("decide")
    p.add_argument("--work", required=True)
    p.add_argument("--repro", required=True, help="criterion 8 JSON with a boolean \"pass\"")
    p.add_argument("--explanations", help="Section 8c JSON: {feature: {explanation, "
                                          "covered_by_stress}}")
    p.add_argument("--out-md", default=RESULT_MD)
    for name in ("sealed-record", "sealed"):
        p = sub.add_parser(name, help="veto V1 on the sealed PFF set (R2 Section 3)")
        p.add_argument("--work", required=True)
        p.add_argument("--stage", required=True, help="run_stage.sh output (final/costs.json)")
        p.add_argument("--final-npz", default=FINAL_NPZ)
        p.add_argument("--result-md", default=RESULT_MD)
        p.add_argument("--refs-root", default=DEFAULT_REFS_ROOT)
        p.add_argument("--feature-cache", help="sealed only: build.py cache of the 71 training "
                                               "matches, for the recorded 8c diagnostic")
    args = parser.parse_args(argv)
    if args.cmd == "decide":
        return cmd_decide(args)
    if args.cmd == "sealed-record":
        return cmd_sealed_record(args)
    if args.cmd == "sealed":
        return cmd_sealed(args)
    ctx = Ctx(args)
    return {"export": cmd_export, "run": cmd_run, "score": cmd_score, "stress": cmd_stress,
            "bar": cmd_bar, "v23": cmd_v23}[args.cmd](ctx)


if __name__ == "__main__":
    sys.exit(main())
