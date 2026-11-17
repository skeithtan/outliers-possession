"""Round 3 scoring of a finished learned run against c23 (LEARNED_STAGE_PROTOCOL_R3.md). Not part of the submission.

The one-pass driver is tools/learned_spike/r3_score.sh RUN_DIR OUT_DIR, which calls the
subcommands below in this order and stops at the first failed criterion where the protocol says
to stop. Each subcommand is resumable (engine outputs are cached by a key of the match, view,
full config, weights md5 and engine source hash, as in evaluate.py).

    E = engine venv python, T = torch venv python; RUN = the r3_launch.sh output folder;
    W = OUT/work; every output stays outside the repository.
    $E -m tools.learned_spike.r3_eval preflight --run RUN --work W [--variant R3B|R3C]
    $T -m tools.learned_spike.evaluate export --stage RUN --work W --final-out OUT/learned_v3.npz
    $E -m tools.learned_spike.r3_eval costs  ARGS   # nested costs (rule B), H costs, kill check
    $E -m tools.learned_spike.r3_eval run    ARGS   # raw view: c23, M, M_fin; matched view
    $E -m tools.learned_spike.r3_eval score  ARGS   # W/scores.json, criteria 1 to 3
    $E -m tools.learned_spike.r3_eval stress ARGS   # W/stress.json, criteria 4 and 5
    $E -m tools.learned_spike.r3_eval bar    ARGS --final-npz OUT/learned_v3.npz   # ONCE
    $E -m tools.learned_spike.r3_eval v23    ARGS --final-npz OUT/learned_v3.npz   # 9, 10
    $E -m tools.learned_spike.r3_eval repro  ARGS ...                              # 8
    $E -m tools.learned_spike.r3_eval pair12a ARGS     # Section 12a walk, only if 8e failed
    (bar, v23, repro) ... --pair H,A                    # criteria 6 to 10 at the 12a pair
    $E -m tools.learned_spike.r3_eval gate --work W --criteria 3 [4 ...]
    $E -m tools.learned_spike.r3_eval decide --work W --out-dir OUT

Variants (R3_DEVIATION_AUG.md Section 13): R3b, and R3c = R3b plus the aug training switch.
`preflight --variant` checks the switches the models recorded against the variant (an R3c model
carries switches["aug"], an R3b model none) and records the variant; `decide` names the result
LEARNED_STAGE_RESULT_<variant>. Inference does not change (the weights format, the export and
the engine are the same; checked with repro_check on an R3c fold model). Every criterion, the
rule 12 pair (on the run's own out-of-fold probabilities, r3_views) and the structure step are
the same for both variants.

Section 12a (R3_DEVIATION_COSTS.md 12a): `pair12a` reads views.json (ranking by Score, ties to
the lower H-A then alive-dead), decodes the final ensemble's stored probabilities of the
label-free step (W/probs/final/<match>/seed_*.npz; no network run, no engine run) at each pair
in rank order with the structure mask, after checking that this decode reproduces the engine's
M output at the rule's pair byte for byte, and writes pair12a.json and pair12a.md (rank, pair,
Score, the four 8e statistics, inside or outside, the chosen pair). `--pair H,A` on bar, v23 and
repro then reruns the label-free criteria at that pair in <folder>_12a/, recorded as
"Section 12a"; it refuses a pair that the walk did not choose.

What differs from round 2 (evaluate.py), and why:

- The baseline is c23 (R3 Section 0.3). It is the engine of this branch with LEARNED_MODE off:
  the c23 engine (tag `submission`, engine tree 241e70d6...) with the learned stage merged in.
  The switch-off identity (`preflight --identity`) checks the 16 external outputs against
  docs/plan/calibration/learned_switch_off_md5_c23.json and BAR-ATH against 7844cad8...
  Nothing is copied from the c22 cache (round 2 copied c22 CSVs there).
- Costs (R3 change B): a development match is decoded with the pair chosen on the other seven
  (RUN/final/nested_costs.json, from r3_nested check on the 4-seed out-of-fold probabilities);
  the final costs are the pair chosen on all eight. The fold costs of kfold (validation pair)
  decide nothing and are not read. Reading (not in the protocol): a SkillCorner match has no
  nested pair (SkillCorner does not enter the choice), so it is decoded with the final costs,
  which no SkillCorner reference chose. "M_fin" is every match with the final costs (round 1's
  "grand-median" row).
- Criteria (R3 Section 4): 1 and 2 (2 s references) are computed and recorded, never blocking;
  3 (as provided, provider-balanced over Metrica, DFL and SkillCorner, 5 % bound above 0) is
  the primary score criterion; 4 to 8 verbatim with c23; 9 and 10 are V2 and V3 of R2. Criterion
  4 is computed on the 2 s references, as its text says, and recorded on the references as
  provided too (see `conventions` in the result for this ambiguity).
- R3b reads no PFF file (R3 Section 9): the 8c band pools the 28 external matches of R3b's
  training set (folds.train_games(pff=False)), no PFF reference md5 is checked, and V1 (sealed
  PFF) does not exist. The sealed judge is the two DFL matches J03WQQ and J03WR9, once, after
  everything else passes and with the owner's explicit go; this tool never opens them.
- Choice: c23 or M (R3 Section 4). H is not eligible; its raw development scores are computed
  and recorded (`run --systems H H_fin`), with its H-A cost chosen by the same nested rule on
  the H grid of decoders.GRID_HA over the c23 labels.
- The label-free step on the competition match (`bar`, `v23`, the second run of `repro`) runs
  once, with the final weights and costs, after criteria 1 to 5 (`bar` refuses before). For a
  rehearsal, `--as-match GAME` runs the same code on a development match in place of the
  competition match, and every output is marked as a rehearsal; decide then titles the result
  "REHEARSAL".

The tool never reads evaluation/gt, never scores the competition match, never opens a sealed
match (folds.check) and never reads a PFF file. Python 3.9.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from tools.learned_spike import evaluate as ev  # noqa: E402
from tools.learned_spike import folds  # noqa: E402

BASE = "c23"
C23_BAR_MD5 = "7844cad862e38d55cf9a3fc29b18ae85"          # R3 Section 0.3
C23_DEV_MD5 = {                                           # R3 Section 8 (raw view)
    "DFL-J03WMX": "6236515bf0272ca2e9f5e57cec75c8b8", "DFL-J03WN1": "8c2f6272925845a80ae03a117ad6eb89",
    "DFL-J03WOH": "bdb5845aa3226b9ac61f7dd18f136c91", "DFL-J03WOY": "9471a2254e5697cbd9539d388ddf16ea",
    "DFL-J03WPY": "6025967bd31a4f05f948713412e5905c", "METRICA-S1": "61d4b60a5dcf9491d5998f4c5aeace61",
    "METRICA-S2": "330d7a6f24bc47548327eca9000f8bb9", "METRICA-S3": "e3d03c7c31dc8a678a3ba7ffa97077ce"}
SWITCH_OFF_RECORD = os.path.join(ROOT, "docs", "plan", "calibration",
                                 "learned_switch_off_md5_c23.json")
DEV = tuple(folds.OPEN)
ALL = tuple(folds.ALL)
SEEDS = (0, 1, 2, 3)
STEPS = 2000
TRAIN_CODE_PATHS = ("tools/learned_spike", "engine/learned_features.py")
R3B_SWITCHES = {"target": "raw", "short_weight": "off", "mix": "sc_quarter"}
# Variants (R3_DEVIATION_AUG.md Section 13): R3c is R3b plus the AUG switch of train.configure()
# (branch learned-r3-aug), which the model records as switches["aug"] = {"cache", "k", "share",
# "manifest_md5"} (train.aug_record()). R3b models carry no "aug" key; an R3c model must carry it
# with these values, and one manifest for every model of the run. The aug cache path is recorded,
# not checked (it is a folder outside the repository).
R3C_AUG = {"k": 16, "share": 0.5}
VARIANTS = ("R3B", "R3C", "R4", "R5", "R6", "R6E", "R7", "R7E")
VARIANT_TITLES = {"R3B": "R3b", "R3C": "R3c", "R4": "R4", "R5": "R5", "R6": "R6",
                  "R6E": "R6E", "R7": "R7", "R7E": "R7E"}
# Round 5 (LEARNED_STAGE_PROTOCOL_R5.md): the round 4 machinery (R4_LIKE) with three changes.
# The models carry r4["feature_version"] = R5_FEATURE_VERSION (train.r4_record()); G7's limits
# 8a and 8b are each the lower of BOUND_FACTOR x the round 5 model's SkillCorner fold mean and
# round 4's limit (R5_LIMIT_CAPS, Section 4: a limit cannot loosen); the ratio of the label-free
# match's disagreement to the SkillCorner fold mean is recorded with 8a and 8b.
R4_LIKE = ("R4", "R5", "R6", "R6E", "R7", "R7E")
R5_FEATURE_VERSION = "stage-v3"
R5_LIMIT_CAPS = {"a": 0.0717, "b": 0.0289}
# Round 6: variant R6 is R5 for feature version stage50-v2s (the version of the training
# records, never of the width), with one change: G7 (criteria 6, 7 and 7x) is recorded, not
# blocking ("the owner decides"); G6 (criterion 5) and G8 (criterion 8) still block.
# Round 7: variants R7 and R7E are R6 and R6E for feature version stage50-v2t (the retrained
# network), identical in every rule.
CAPPED = ("R5", "R6", "R6E", "R7", "R7E")
FEATURE_VERSIONS = {"R5": R5_FEATURE_VERSION, "R6": "stage50-v2s", "R6E": "stage50-v2s",
                    "R7": "stage50-v2t", "R7E": "stage50-v2t"}
# Variant R6E: R6 for the candidate network + per-frame member (Config.LEARNED_PF on): the
# member's weights file and weight come with the pair (costs-fixed --pf-weights, --pf-weight)
# and reach every engine run (evaluate.pf_overrides); the label-free stability members are
# those of evaluate.member_frames_pf (a seed member: the member blended with one network seed; a
# copy member: the member blended with one symmetric copy of the network, over the seeds).
PF_VARIANTS = ("R6E", "R7E")
R6_LIKE = ("R6", "R6E", "R7", "R7E")
R6_BLOCKING = ("5", "8")
ROUND_OF = {"R4": "4", "R5": "5", "R6": "6", "R6E": "6", "R7": "7", "R7E": "7"}
R6_RECORDED = ("6", "7", "7x")
# Round 4 (LEARNED_STAGE_PROTOCOL_R4.md, Sections 3 and 5): the models carry the round 4 record
# of train.r4_record(): step A views ["raw"], step C dfl_still "removed" (with its val cache),
# optionally B (sc_labels) or B' (sc_alive_dead), D' (ballfree) and E (short_weight sqrt), on
# top of the aug switch of R3c. Its final scoring here: preflight, export, costs-fixed (the pair
# and prior bonus of Section 4, given on the command line: no cost search, no walk), stress
# (criterion 5 = G6; criterion 4 recorded), the label-free step once (criteria 6 and 7 = G7 with
# the four extra bounds of `r4bounds`, criterion "7x"), criterion 8 (G8). No interim kill check,
# no criteria 1 to 4 or 9 and 10 as gates (G1 to G5 come from r4_screen and r4_pff_test).
R4_SHORT_WEIGHTS = ("off", "sqrt")
R4_REQUIRED = {"views": ["raw"], "dfl_still": "removed"}
R4_OPTIONAL = ("sc_labels", "sc_alive_dead", "ballfree", "val_cache")
R4_BLOCKING = ("5", "6", "7", "7x", "8")
RAW_SYSTEMS = ("c23", "M", "M_fin")
H_SYSTEMS = ("H", "H_fin")
MATCHED_SYSTEMS = ("c23", "M")
STRESS_SYSTEMS = ("c23", "M")
BAR_SYSTEMS = ("c23", "M", "Mfold")
BLOCKING = ("3", "4", "5", "6", "7", "8", "9", "10")
RECORDED = ("1", "2")
RESULT_NAME = "LEARNED_STAGE_RESULT_R3B"


def result_name(variant="R3B"):
    """LEARNED_STAGE_RESULT_R3B or LEARNED_STAGE_RESULT_R3C."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    return "LEARNED_STAGE_RESULT_" + variant


DISCLOSURE_R3C = (
    " R3c (R3_DEVIATION_AUG.md Section 13) is R3b with one training switch added, --aug-cache "
    "(16 pre-defected copies per match, half of the crops; R3_AUG.md); it was started because "
    "its first seed scored higher than R3b's on the stress view. Which of R3b and R3c is the "
    "candidate is decided on the development matches only (Section 13); this file scores R3c "
    "against c23 under the same criteria.")
DISCLOSURE = (
    "Round 3 reopens a stage that R2 had closed ('There is no round 3, whatever the outcome'), on "
    "the owner's decision of 12 November 2026, and the engine rule is lifted only if M passes "
    "every criterion against c23 (R3 Section 0). R3b trains without PFF for licence reasons "
    "(R3 Section 9); it was written after the with-PFF seed 0 had passed its interim check, and "
    "its only changes are the removal of a provider and one mix weight taken from outside. "
    "Change B's grid and objective were chosen knowing the stage 1 development scores of the "
    "round 2 models (R3 Section 0.5). R3b's development scores measure providers the model has "
    "seen (about 60 % of the crops are Metrica or DFL), the competition match comes from a "
    "provider it has not seen (R3 Section 10).")


def disclosure(variant="R3B"):
    return DISCLOSURE + (DISCLOSURE_R3C if variant == "R3C" else "")


def log(text):
    print(text, file=sys.stderr, flush=True)


read_json = ev.read_json
write_json = ev.write_json
md5_file = ev.md5_file


# -- pure helpers (tested) ------------------------------------------------------------------------

def nested_index(scores, games, grid_len, test=None):
    """(index, value): the grid index with the best mean of the Metrica and DFL means over
    `games` without `test`; ties go to the lower index (r3_nested.nested_pick on any grid)."""
    use = [g for g in games if g != test]
    best, best_v = None, -1.0
    for i in range(grid_len):
        v = np.mean([np.mean([scores[g][i] for g in use if folds.provider(g) == p])
                     for p in ("metrica", "dfl")])
        if v > best_v + 1e-12:
            best, best_v = i, float(v)
    return best, best_v


def match_costs(costs, game):
    """The evaluate.overrides() cost dict of one match: nested pair for a development match,
    the final pair otherwise (SkillCorner)."""
    m = costs["M_nested"].get(game, costs["M_final"])
    h = costs["H_nested"].get(game, costs["H_final"]) if costs.get("H_nested") else None
    out = {"M": (float(m[0]), float(m[1])), "H": None if h is None else float(h),
           "source": "nested" if game in costs["M_nested"] else "final"}
    return _r4_costs(costs, out)


def _r4_costs(costs, out):
    """Round 4 (costs-fixed): the structure step and the prior bonus travel with the pair."""
    if costs.get("fixed"):
        out["structure"] = bool(costs.get("structure"))
        out["prior_bonus"] = float(costs.get("prior_bonus", 0.0))
        out["source"] = "fixed"
        if costs.get("pf"):
            out["pf"] = costs["pf"]                   # round 6, variant R6E
    return out


def final_costs(costs):
    h = costs.get("H_final")
    return _r4_costs(costs, {"M": (float(costs["M_final"][0]), float(costs["M_final"][1])),
                             "H": None if h is None else float(h), "source": "final"})


def rename_base(summary):
    """evaluate.delta_summary() output with its "c22" names read as the baseline c23."""
    out = dict(summary)
    out["c23_mean"] = out.pop("c22_mean")
    out["per_provider"] = {p: dict((("c23_mean" if k == "c22_mean" else k), v)
                                   for k, v in r.items())
                           for p, r in summary["per_provider"].items()}
    out["per_match"] = {g: {"x": r["x"], "c23": r["c22"], "delta": r["delta"]}
                        for g, r in summary["per_match"].items()}
    return out


def summary(x, base):
    return rename_base(ev.delta_summary(x, base))


def crit1(nested_ctl2, final_ctl2):
    r = ev.crit1(nested_ctl2, final_ctl2)
    r["numbers"] = {"nested_costs": r["numbers"]["fold_costs"],
                    "final_costs": r["numbers"]["median_costs"]}
    return dict(r, blocking=False)


def crit2(nested_ctl2):
    return dict(ev.crit2(nested_ctl2), blocking=False)


def crit3(nested_provided, dev_only=None):
    r = ev.crit3(nested_provided)
    if dev_only is not None:
        r["recorded_metrica_dfl_only"] = [dev_only["delta"], dev_only["low"], dev_only["high"]]
    return dict(r, blocking=True)


def crit4(stressed, unstressed, drop_x, drop_base, recorded_provided=None):
    r = ev.crit4(stressed, unstressed, drop_x, drop_base)
    r["numbers"]["drop_c23"] = r["numbers"].pop("drop_c22")
    r["reference"] = "2 s (ctl2), as the criterion text says"
    if recorded_provided is not None:
        r["recorded_provided"] = recorded_provided
    return dict(r, blocking=True)


def crit5(share_x, share_base):
    r = ev.crit5(share_x, share_base)
    for row in r["numbers"].values():
        if row is not None:
            row["c23"] = row.pop("c22")
    return dict(r, blocking=True)


def crit6(stats, limits=None):
    return dict(ev.crit6(stats, limits), blocking=True)


def limit8(part, sc_mean_fold, variant=None):
    """The limit of bound 8a (part "a", seeds) or 8b ("b", copies): BOUND_FACTOR x the
    SkillCorner fold mean; for R5 the lower of that and round 4's limit (R5_LIMIT_CAPS)."""
    limit = ev.BOUND_FACTOR * sc_mean_fold
    if variant in CAPPED:
        limit = min(limit, R5_LIMIT_CAPS[part])
    return limit


def bounds8(bar, explanations=None, variant=None):
    """Section 8 a, b, c, e, f, g for M from the r3 bar.json record (round 1 bounds, c23).

    variant R5: limits 8a and 8b by limit8(), and the ratio bar / SkillCorner fold mean."""
    explanations = explanations or {}
    out = {}
    for part, key in (("a", "seed"), ("b", "copy")):
        r = bar["disagreement"][key]
        limit = limit8(part, r["sc_mean_fold"], variant)
        out[part] = {"bar": r["bar"], "sc_mean": r["sc_mean_fold"], "limit": limit,
                     "sc_mean_final_recorded": r.get("sc_mean_final"),
                     "pass": bool(r["bar"] <= limit)}
        if variant in CAPPED:
            out[part].update(
                limit_rule="min(1.5 x SkillCorner fold mean, round 4's {})".format(
                    R5_LIMIT_CAPS[part]),
                limit_uncapped=ev.BOUND_FACTOR * r["sc_mean_fold"],
                ratio_to_sc_fold_mean=(r["bar"] / r["sc_mean_fold"]
                                       if r["sc_mean_fold"] else None))
    rows, ok = {}, True
    for name in bar["features"]["outside"]:
        note = explanations.get(name) or {}
        explained = bool(note.get("explanation"))
        good = explained and (bool(note.get("covered_by_stress"))
                              if name in ev.SLOT_COUNT_FEATURES else True)
        rows[name] = {"slot_count": name in ev.SLOT_COUNT_FEATURES, "explained": explained,
                      "covered_by_stress": bool(note.get("covered_by_stress")),
                      "pass": bool(good)}
        ok = ok and good
    out["c"] = {"outside": rows, "pass": bool(ok)}
    out["d"] = {"reported": True}
    env = bar["envelope_provided"]
    erows = {k: env.get(k) for k in ev.ENVELOPE_KEYS["M"]}
    out["e"] = {"checked": erows, "pass": bool(all(r is not None and r["inside"]
                                                    for r in erows.values()))}
    agree = bar["agreement_final_vs_fold"]
    out["f"] = {"agreement": agree, "pass": bool(agree >= ev.MIN_AGREEMENT)}
    out["g"] = {"reported": True, "x": bar["systems"]["M"].get("d_share_ball_missing"),
                "c23": bar["systems"]["c23"].get("d_share_ball_missing")}
    out["pass"] = bool(all(out[p]["pass"] for p in "abcef"))
    return out


def crit7(b8):
    return {"pass": bool(b8["pass"]), "numbers": {p: b8[p]["pass"] for p in "abcef"},
            "blocking": True}


def crit8(repro):
    return {"pass": bool(repro.get("pass") is True), "numbers": repro, "blocking": True}


def crit9(v23):
    sysv = v23["v2"]["systems"]
    x, c = sysv["M"]["flip"], sysv["c23"]["flip"]
    ok = x is not None and c is not None and x <= c + ev.MAX_EXTRA_FLIP
    return {"pass": bool(ok), "blocking": True, "numbers": {
        "flip_x": x, "flip_c23": c, "limit": None if c is None else c + ev.MAX_EXTRA_FLIP,
        "alive_raw_x": sysv["M"]["alive_raw"], "k_frames": v23["v2"]["k_frames"]}}


def crit10(v23):
    x, c = v23["v3"]["M"], v23["v3"]["c23"]
    ok = x is not None and c is not None and x <= c + ev.MAX_EXTRA_D_MISSING
    return {"pass": bool(ok), "blocking": True, "numbers": {
        "d_missing_x": x, "d_missing_c23": c,
        "limit": None if c is None else c + ev.MAX_EXTRA_D_MISSING}}


def blocking_of(variant):
    """The blocking criteria of a variant: R6_BLOCKING for R6 (G7 recorded), R4_BLOCKING for R4
    and R5, BLOCKING for round 3."""
    if variant in R6_LIKE:
        return R6_BLOCKING
    return R4_BLOCKING if variant in R4_LIKE else BLOCKING


def verdict(criteria, kill_passed, stop=None, blocking=BLOCKING):
    """(choice, reason): c23 or M (R3 Section 4). criteria: {"1".."10": {"pass"} or None}.
    Round 4: blocking=R4_BLOCKING (criteria 5, 6, 7, 7x, 8 = G6, G7, G8)."""
    if not kill_passed:
        return BASE, "the interim kill check K1 to K3 failed (R3 Section 3)"
    failed = [i for i in blocking if criteria.get(i) is not None and not criteria[i]["pass"]]
    if failed:
        return BASE, "M fails criterion {}".format(", ".join(failed))
    missing = [i for i in blocking if criteria.get(i) is None]
    if missing:
        return None, "incomplete: criterion {} not computed{}".format(
            ", ".join(missing), " ({})".format(stop) if stop else "")
    if tuple(blocking) == R6_BLOCKING:
        g7 = [criteria.get(k) for k in R6_RECORDED]
        state = ("not computed" if any(c is None for c in g7)
                 else "pass" if all(c["pass"] for c in g7) else "fail")
        return "M", ("M passes G6 and G8 against c23 at the fixed pair and prior bonus; G7: {} "
                     "(recorded; the owner decides). The other gates (G1 to G5, G3b, G4b) are "
                     "decided by r4_screen and r4_pff_test, and G9 runs last".format(state))
    if tuple(blocking) == R4_BLOCKING:
        return "M", ("M passes G6, G7 (criteria 5, 6 and 7 of round 3 and the four extra bounds) "
                     "and G8 against c23 at the fixed pair and prior bonus. The other gates of "
                     "LEARNED_STAGE_PROTOCOL_R4.md Section 5 (G1 to G5, G3b, G4b) are decided by "
                     "r4_screen and r4_pff_test, and G9 runs last")
    return "M", ("M passes criteria 3 to 10 against c23 (1 and 2 recorded). It ships only after "
                 "the sealed DFL judge (J03WQQ, J03WR9: mean of Δ(M − c23) as provided ≥ 0), "
                 "run once with the owner's explicit go (R3 Sections 4 and 9)")


# -- context --------------------------------------------------------------------------------------

def make_ctx(args):
    """An evaluate.Ctx for the shared run machinery (never copies a c22 run)."""
    ns = argparse.Namespace(**vars(args))
    ns.stage = args.run
    ns.smoke_costs = None
    ns.c22_ext = None
    ns.games = None
    for name in ("engine_cache", "sc_engine_cache", "ref_cache"):
        if not hasattr(ns, name):
            setattr(ns, name, None)
    # One engine process at a time (round 3). R3_EVAL_JOBS > 1 runs that many external engine
    # runs at once (the label-free step stays one run at a time: competition_run is serial).
    ns.jobs = max(1, int(os.environ.get("R3_EVAL_JOBS", "1")))
    return ev.Ctx(ns)


def load_costs(work):
    path = os.path.join(work, "costs_r3.json")
    if not os.path.isfile(path):
        raise SystemExit("{} is missing: run `costs` first".format(path))
    return read_json(path)


def pf_of(work):
    """The per-frame member of the fixed costs ({"weights", "weight", "md5"}; None without)."""
    path = os.path.join(work, "costs_r3.json")
    return read_json(path).get("pf") if os.path.isfile(path) else None


def pf_for(pf, fold=None):
    """The member of one set of models: {"weights": the file ("{fold}" filled in: the fold
    number, or "final" for fold None), "weight"}; None without a member."""
    if not pf:
        return None
    from tools.learned_spike import pf_gate
    return {"weights": pf_gate.pf_path(pf["weights"], fold=fold), "weight": pf["weight"]}


def rehearsal_of(work):
    path = os.path.join(work, "preflight.json")
    return bool(read_json(path).get("rehearsal")) if os.path.isfile(path) else False


# -- preflight ------------------------------------------------------------------------------------

def git_out(*args):
    try:
        return subprocess.check_output(("git",) + args, cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def same_training_code(commits):
    """True if the training code paths are identical at every commit in `commits`."""
    commits = sorted(set(c for c in commits if c))
    for c in commits[1:]:
        if subprocess.call(["git", "diff", "--quiet", commits[0], c, "--"] +
                           list(TRAIN_CODE_PATHS), cwd=ROOT) != 0:
            return False
    return True


def check_model_record(rec, k=None, variant="R3B"):
    """[problem] of one model record (model_s*.json or final_s*.json) under R3b or R3c."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    bad = []
    train = list(rec.get("train") or [])
    if any(g.startswith("PFF-") for g in train):
        bad.append("a PFF match in the training list")
    if any(g in folds.SEALED or g == folds.COMPETITION for g in train):
        bad.append("a sealed or competition match in the training list")
    if k is not None:
        spec = folds.fold(k, pff=False)
        held = set(spec["test"]) | set(spec["val"])
        if held & set(train):
            bad.append("a held-out match of fold {} in the training list".format(k))
        if sorted(train) != sorted(spec["train"]):
            bad.append("training list differs from folds.fold({}, pff=False)".format(k))
    switches = rec.get("switches") or {}
    title = VARIANT_TITLES[variant]
    if variant in R4_LIKE:
        return bad + check_r4_switches(switches, variant) + (
            [] if rec.get("steps") == STEPS else ["steps {} != {}".format(rec.get("steps"), STEPS)])
    for key, want in R3B_SWITCHES.items():
        if switches.get(key) != want:
            bad.append("switch {} = {!r}, {} needs {!r}".format(key, switches.get(key), title,
                                                                 want))
    aug = switches.get("aug")
    if variant == "R3B" and aug is not None:
        bad.append("switch aug is set (an R3c model): R3b has no aug switch (score it with "
                   "--variant R3C)")
    if variant == "R3C":
        if not isinstance(aug, dict):
            bad.append("switch aug is missing: R3c needs --aug-cache (R3_DEVIATION_AUG.md 13)")
        else:
            for key, want in R3C_AUG.items():
                if aug.get(key) != want:
                    bad.append("switch aug.{} = {!r}, R3c needs {!r}".format(key, aug.get(key),
                                                                              want))
            if not aug.get("manifest_md5"):
                bad.append("switch aug has no manifest md5")
    extra = sorted(set(switches) - set(R3B_SWITCHES) - {"short_median_bins", "aug"})
    if extra:
        bad.append("unknown switches {} (not part of {})".format(extra, title))
    if rec.get("steps") != STEPS:
        bad.append("steps {} != {}".format(rec.get("steps"), STEPS))
    return bad


def check_r4_switches(switches, variant="R4"):
    """[problem] of the switches of a round 4 model (train.recipe / train_model records).

    variant R5: the round 4 record must also carry feature_version R5_FEATURE_VERSION."""
    bad = []
    for key in ("target", "mix"):
        if switches.get(key) != R3B_SWITCHES[key]:
            bad.append("switch {} = {!r}, R4 needs {!r}".format(key, switches.get(key),
                                                                R3B_SWITCHES[key]))
    if switches.get("short_weight") not in R4_SHORT_WEIGHTS:
        bad.append("switch short_weight = {!r}".format(switches.get("short_weight")))
    aug = switches.get("aug")
    if not isinstance(aug, dict) or any(aug.get(k) != v for k, v in R3C_AUG.items()) \
            or not aug.get("manifest_md5"):
        bad.append("switch aug missing or not k 16, share 0.5 with a manifest (R4 step A)")
    r4 = switches.get("r4")
    if not isinstance(r4, dict):
        bad.append("no round 4 record (switches['r4']): not a round 4 model")
        r4 = {}
    for key, want in R4_REQUIRED.items():
        if r4.get(key) != want:
            bad.append("r4 {} = {!r}, R4 needs {!r}".format(key, r4.get(key), want))
    if r4.get("sc_labels") and r4.get("sc_alive_dead"):
        bad.append("both B and B' are set")
    if variant in FEATURE_VERSIONS and r4.get("feature_version") != FEATURE_VERSIONS[variant]:
        bad.append("r4 feature_version = {!r}, {} needs {!r}".format(
            r4.get("feature_version"), variant, FEATURE_VERSIONS[variant]))
    allowed = set(R4_REQUIRED) | set(R4_OPTIONAL) | ({"feature_version"}
                                                     if variant in FEATURE_VERSIONS else set())
    extra = sorted(set(r4) - allowed)
    if extra:
        bad.append("unknown round 4 switches {}".format(extra))
    extra = sorted(set(switches) - set(R3B_SWITCHES) - {"short_median_bins", "aug", "r4"})
    if extra:
        bad.append("unknown switches {}".format(extra))
    return bad


def variant_of(work):
    path = os.path.join(work, "preflight.json")
    return (read_json(path).get("variant") or "R3B") if os.path.isfile(path) else "R3B"


def cmd_preflight(args):
    """The finished run against its variant (R3b, or R3c = R3b + the aug switch): 32 fold models,
    4 final models, nested costs on 4 seeds, training lists without PFF or held-out matches, one
    training code (and for R3c one aug manifest), kill check passed."""
    run = os.path.abspath(args.run)
    variant = getattr(args, "variant", None) or "R3B"
    problems, commits, models, manifests = [], [], {}, set()
    for k in range(folds.N_FOLDS):
        for s in SEEDS:
            net = os.path.join(run, "folds", "fold{}".format(k), "net_s{}.pt".format(s))
            rec = os.path.join(run, "folds", "fold{}".format(k), "model_s{}.json".format(s))
            if not (os.path.isfile(net) and os.path.isfile(rec)):
                problems.append("fold {} seed {}: model missing".format(k, s))
                continue
            r = read_json(rec)
            if r.get("seed") != s:
                problems.append("fold {} seed {}: record says seed {}".format(k, s, r.get("seed")))
            problems += ["fold {} seed {}: {}".format(k, s, p)
                         for p in check_model_record(r, k, variant)]
            manifests.add(((r.get("switches") or {}).get("aug") or {}).get("manifest_md5"))
            commits.append((r.get("environment") or {}).get("git_commit"))
            models["fold{}/s{}".format(k, s)] = {"md5": md5_file(net),
                                                  "best_step": r.get("best_step")}
    for s in SEEDS:
        net = os.path.join(run, "final", "net_s{}.pt".format(s))
        rec = os.path.join(run, "final", "final_s{}.json".format(s))
        if not (os.path.isfile(net) and os.path.isfile(rec)):
            problems.append("final seed {}: model missing".format(s))
            continue
        r = read_json(rec)
        problems += ["final seed {}: {}".format(s, p)
                     for p in check_model_record(r, None, variant)]
        manifests.add(((r.get("switches") or {}).get("aug") or {}).get("manifest_md5"))
        if r.get("train") and sorted(r["train"]) != sorted(folds.train_games(pff=False)):
            problems.append("final seed {}: training list is not the 28 matches".format(s))
        commits.append((r.get("environment") or {}).get("git_commit"))
        models["final/s{}".format(s)] = {"md5": md5_file(net), "stop_at": r.get("stop_at")}
    fold_best = [m["best_step"] for key, m in sorted(models.items()) if key.startswith("fold")]
    if len(fold_best) == folds.N_FOLDS * len(SEEDS):
        want = folds.final_steps(fold_best)
        for key, m in models.items():
            if key.startswith("final/") and m.get("stop_at") != want:
                problems.append("{}: stop_at {} != median fold best step {}".format(
                    key, m.get("stop_at"), want))
    md5s = [m["md5"] for key, m in models.items() if key.startswith("fold0/")]
    if len(set(md5s)) < len(md5s):
        problems.append("fold 0: two seeds have identical weights")
    if not same_training_code(commits):
        problems.append("the models come from commits with different training code: {}".format(
            sorted(set(c for c in commits if c))))
    if variant in ("R3C",) + R4_LIKE and len(manifests - {None}) > 1:
        problems.append("the models were trained on different aug caches (manifest md5s {})".format(
            sorted(manifests - {None})))
    if variant in R4_LIKE:
        kill = {"passed": True, "note": "rounds 4 and 5 have no interim kill check"}
    else:
        interim = os.path.join(run, "interim", "check.json")
        kill = read_json(interim)["kill_check"] if os.path.isfile(interim) else None
        if kill is None:
            problems.append("interim/check.json missing (R3 Section 3)")
        elif not kill.get("passed"):
            problems.append("interim kill check failed: {}".format(kill.get("reasons")))
        nested = os.path.join(run, "final", "nested_costs.json")
        if not os.path.isfile(nested):
            problems.append("final/nested_costs.json missing")
    oof = os.path.join(run, "oof_p5")
    seeds_seen = {}
    for game in DEV:
        path = os.path.join(oof, game + "_p5.npz")
        if not os.path.isfile(path):
            problems.append("oof_p5/{}_p5.npz missing".format(game))
            continue
        with np.load(path) as z:
            seeds_seen[game] = [int(v) for v in z["seeds"]]
        if seeds_seen[game] != list(SEEDS):
            problems.append("oof_p5/{}: seeds {} (rule B needs the 4-seed ensembles)".format(
                game, seeds_seen[game]))
    rec = {"run": run, "variant": variant, "aug_manifest_md5": sorted(manifests - {None}),
           "rehearsal": bool(args.rehearsal), "problems": problems,
           "models": models, "commits": sorted(set(c for c in commits if c)),
           "kill_check_interim": kill, "oof_seeds": seeds_seen, "pass": not problems,
           "tool_commit": git_out("rev-parse", "HEAD")}
    if args.identity:
        rec["identity"] = switch_off_identity(args)
        if not rec["identity"]["pass"]:
            problems.append("switch-off identity failed: {}".format(rec["identity"]["diff"]))
            rec["pass"] = False
    write_json(os.path.join(args.work, "preflight.json"), rec)
    for p in problems:
        print("preflight: " + p)
    if problems and not args.rehearsal:
        print("PREFLIGHT FAILED: the run is not a finished {} run; do not score it.".format(
            VARIANT_TITLES[variant]))
        return 2
    print("preflight: {}{}".format("pass" if not problems else "problems recorded",
                                   " (rehearsal: problems are recorded, not fatal)"
                                   if args.rehearsal else ""))
    return 0


def competition_data_dir():
    """The data folder of the competition match (MTAH_DATA_DIR, else data/ of the checkout)."""
    env = os.environ.get("MTAH_DATA_DIR")
    for d in ([env] if env else []) + [os.path.join(ROOT, "data")]:
        if os.path.isdir(os.path.join(d, "tactical")):
            return os.path.abspath(d)
    raise SystemExit("the competition data folder is not found: set MTAH_DATA_DIR")


def switch_off_identity(args):
    """The switch-off identity of this tree against c23: 16 external outputs and BAR-ATH."""
    from tools.analysis import switch_off_identity as soi
    cache = os.path.join(os.path.abspath(args.work), "switch_off_cache")
    measured = soi.measure(ROOT, cache, 1)
    recorded = read_json(SWITCH_OFF_RECORD)
    diff = soi.compare(recorded, measured)
    raw_ok = all(measured.get("{}/raw".format(g)) == m for g, m in C23_DEV_MD5.items())
    out = os.path.join(os.path.abspath(args.work), "identity", "BAR-ATH_off.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    env = dict(os.environ, MTAH_DATA_DIR=competition_data_dir())
    with open(out[:-4] + ".log", "w") as handle:
        code = subprocess.call([sys.executable, "-m", "engine.run", "--out", out], cwd=ROOT,
                               env=env, stdout=handle, stderr=subprocess.STDOUT)
    bar_md5 = md5_file(out) if code == 0 and os.path.isfile(out) else None
    return {"external": measured, "diff": diff, "raw_equals_protocol_section_8": raw_ok,
            "bar_ath_md5": bar_md5, "bar_ath_expected": C23_BAR_MD5,
            "pass": bool(not diff and raw_ok and bar_md5 == C23_BAR_MD5),
            "engine_source_hash": ev.engine_hash()}


# -- costs ----------------------------------------------------------------------------------------

def cmd_costs(args):
    """Rule B costs from the run (verified by recomputation) and the nested H costs."""
    import score_game as sg
    from engine import data
    from tools.learned_spike import decoders, r3_nested
    from tools.learned_spike import r3_scope as r
    run = os.path.abspath(args.run)
    nested = read_json(os.path.join(run, "final", "nested_costs.json"))
    oof = os.path.join(run, "oof_p5")
    probs, refs, base = {}, {}, {}
    scores_m, scores_h = {}, {}
    grid_h = list(decoders.GRID_HA)
    for game in DEV:
        folds.check(game)
        lab = np.load(os.path.join(args.cache, "labels", game + ".npz"))
        refs[game] = lab["raw"].astype(np.int8)
        p5 = np.load(os.path.join(oof, game + "_p5.npz"))["p5"]
        probs[game] = decoders.upsample(p5, len(refs[game]))
        base[game] = sg.read_codes(c23_dev_csv(args, game), len(refs[game]))
        paths = decoders.decode_model(probs[game], data.graded_blocks(game), r3_nested.GRID)
        scores_m[game] = [r.spell_f1(refs[game], p) for p in paths]
        paths_h = decoders.decode_hybrid(probs[game], base[game], grid_h)
        scores_h[game] = [r.spell_f1(refs[game], p) for p in paths_h]
        log("costs: {} decoded on both grids".format(game))
    m_nested, h_nested, rows = {}, {}, {}
    mismatch = []
    for game in DEV:
        i, v = nested_index(scores_m, DEV, len(r3_nested.GRID), game)
        j, w = nested_index(scores_h, DEV, len(grid_h), game)
        m_nested[game] = list(r3_nested.GRID[i])
        h_nested[game] = grid_h[j]
        recorded = nested["matches"][game]["costs"]
        if [float(x) for x in recorded] != [float(x) for x in m_nested[game]]:
            mismatch.append("{}: recomputed {} vs nested_costs.json {}".format(
                game, m_nested[game], recorded))
        rows[game] = {"M": m_nested[game], "M_nested_val": v, "M_spellF1": scores_m[game][i],
                      "H": h_nested[game], "H_nested_val": w, "H_spellF1": scores_h[game][j]}
    fi, fv = nested_index(scores_m, DEV, len(r3_nested.GRID))
    hj, hw = nested_index(scores_h, DEV, len(grid_h))
    m_final, h_final = list(r3_nested.GRID[fi]), grid_h[hj]
    if [float(x) for x in nested["final_costs"]] != [float(x) for x in m_final]:
        mismatch.append("final: recomputed {} vs nested_costs.json {}".format(
            m_final, nested["final_costs"]))
    rec = {"M_nested": m_nested, "M_final": m_final, "H_nested": h_nested, "H_final": h_final,
           "rows": rows, "final_val": {"M": fv, "H": hw}, "grid_M": r3_nested.GRID,
           "grid_H": grid_h, "nested_costs_json": nested, "recompute_mismatch": mismatch,
           "kill_check_4seed": nested.get("kill_check"),
           "rehearsal": rehearsal_of(args.work)}
    write_json(os.path.join(args.work, "costs_r3.json"), rec)
    # The final costs in the costs.json format that repro_check.py and evaluate read.
    write_json(os.path.join(args.work, "costs_final.json"), {
        "M": {"cost_ha": m_final[0], "cost_ad": m_final[1]}, "H": {"cost_ha": h_final},
        "source": "r3 rule B, nested_costs.json final_costs (M); nested H rule (H)"})
    print("costs: M final {} H final {}; nested M {}".format(m_final, h_final, m_nested))
    if mismatch:
        print("COSTS MISMATCH between nested_costs.json and the recomputation:\n  " +
              "\n  ".join(mismatch))
        return 2
    return 0


def cmd_costs_fixed(args):
    """R4: W/costs_r3.json with one pair for every match and the prior bonus (no search)."""
    if variant_of(args.work) not in R4_LIKE:
        raise SystemExit("costs-fixed is the round 4 and 5 step: run preflight --variant R4 or "
                         "R5 first")
    a, b = float(args.pair[0]), float(args.pair[1])
    if a > 2 * b:
        raise SystemExit("pair ({}, {}): team-change cost above twice the alive-dead cost".format(
            a, b))
    if args.prior_bonus not in (0.0, 1.0, 2.0):
        raise SystemExit("the prior bonus of Section 4 is 0, 1 or 2")
    rec = {"fixed": True, "M_nested": {}, "M_final": [a, b], "H_nested": {}, "H_final": None,
           "structure": not args.no_structure, "prior_bonus": float(args.prior_bonus),
           "source": "LEARNED_STAGE_PROTOCOL_R4.md Section 4, given on the command line",
           "rehearsal": rehearsal_of(args.work)}
    pf_path = getattr(args, "pf_weights", None)
    if variant_of(args.work) in PF_VARIANTS:
        if not pf_path:
            raise SystemExit("variant {} needs --pf-weights (the per-frame member)".format(
                variant_of(args.work)))
        from engine import learned_pf
        from tools.learned_spike import pf_gate
        final_pf = pf_gate.pf_path(pf_path)
        learned_pf.load_pf(final_pf)                  # refuse a file the engine cannot run
        weight = float(args.pf_weight)
        if not 0.0 <= weight <= 1.0:
            raise SystemExit("--pf-weight must lie in [0, 1]")
        # "{fold}" in the path: the fold members for the fold dumps of 8a/8b, "final" else
        rec["pf"] = {"weights": os.path.abspath(pf_path), "md5": md5_file(final_pf),
                     "weight": weight, "pf_shared": pf_gate.pf_shared(pf_path)}
    elif pf_path:
        raise SystemExit("--pf-weights belongs to variants {} only".format(PF_VARIANTS))
    path = os.path.join(args.work, "costs_r3.json")
    if os.path.isfile(path) and read_json(path) != rec:
        raise SystemExit("{} already holds other costs: {}".format(path, read_json(path)))
    write_json(path, rec)
    write_json(os.path.join(args.work, "costs_final.json"), dict({
        "M": {"cost_ha": a, "cost_ad": b}, "H": {"cost_ha": None},
        "prior_bonus": float(args.prior_bonus), "structure": rec["structure"],
        "source": rec["source"]}, **({"pf": rec["pf"]} if rec.get("pf") else {})))
    print("costs-fixed: M {} structure {} prior bonus {:g}".format((a, b), rec["structure"],
                                                                   rec["prior_bonus"]))
    return 0


R4_REF_KEYS = ("possessions_per_90min", "possession_s.p10", "possession_s.p25")


def _stat(st, key):
    if "." in key:
        a, q = key.split(".")
        return st[a][q]
    return st[key]


def r4_bounds(m_stats, c23_stats, ref_stats, ratio_range):
    """The four extra reject-only bounds of G7. m_stats, c23_stats: bar_report.system_stats of
    M and c23 on the label-free match; ref_stats: the same of the 28 external references as
    provided; ratio_range: (min, max) of possessions(M) / possessions(c23) over the 28 external
    out-of-fold outputs at the same pair and bonus."""
    rows = {}
    for key in R4_REF_KEYS:
        vals = [_stat(r, key) for r in ref_stats if _stat(r, key) is not None]
        lo, hi = min(vals), max(vals)
        v = _stat(m_stats, key)
        rows[key] = {"value": None if v is None else float(v), "range": [float(lo), float(hi)],
                     "inside": bool(v is not None and lo <= v <= hi)}
    ratio = m_stats["possessions"] / float(c23_stats["possessions"])
    lo, hi = float(ratio_range[0]), float(ratio_range[1])
    rows["possessions_over_c23"] = {"value": float(ratio), "range": [lo, hi],
                                    "inside": bool(lo <= ratio <= hi)}
    return {"bounds": rows, "pass": bool(all(r["inside"] for r in rows.values()))}


def cmd_r4bounds(args):
    """Criterion 7x (R4 G7): the four extra bounds, from bar.json and the external references."""
    from tools.learned_spike import bar_report as br
    ctx = make_ctx(args)
    match = Match(ctx, args.as_match, False)
    bar = read_json(os.path.join(match.folder, "bar.json"))
    oof = read_json(args.oof_ratio)
    costs = bar["final_costs"]
    if [float(x) for x in oof["pair"]] != [float(x) for x in costs["M"]] or \
            float(oof["prior_bonus"]) != float(costs.get("prior_bonus", 0.0)):
        raise SystemExit("the out-of-fold ratio file is for pair {} bonus {}, the label-free step "
                         "for {} bonus {}".format(oof["pair"], oof["prior_bonus"], costs["M"],
                                                  costs.get("prior_bonus", 0.0)))
    refs = []
    for game in ALL:
        r = ev.references(ctx, game)["provided"]
        refs.append(br.system_stats(r, br.graded_runs(game), None))
    rec = r4_bounds(bar["systems"]["M"], bar["systems"][BASE], refs,
                    (oof["ratio_min"], oof["ratio_max"]))
    rec.update(game=match.game, rehearsal=match.rehearsal, oof_ratio=args.oof_ratio,
               pair=list(costs["M"]), prior_bonus=costs.get("prior_bonus", 0.0))
    write_json(os.path.join(match.folder, "r4bounds.json"), rec)
    for k, r in rec["bounds"].items():
        print("7x {}: {:.3f} in [{:.3f}, {:.3f}]: {}".format(k, r["value"], r["range"][0],
                                                            r["range"][1], r["inside"]))
    print("criterion 7x: {}".format("pass" if rec["pass"] else "FAIL"))
    return 0


def c23_dev_csv(args, game):
    """The c23 output of a development match, checked against R3 Section 8."""
    path = os.path.join(os.path.abspath(args.c23_dev), game + ".csv")
    if md5_file(path) != C23_DEV_MD5[game]:
        raise SystemExit("{}: md5 differs from R3 Section 8".format(path))
    return path


# -- engine runs on the external matches ----------------------------------------------------------

def raw_jobs(ctx, costs, systems, view=ev.VIEW_RAW, games=ALL):
    """{(game, label): job} of the external engine runs."""
    jobs = {}
    for game in games:
        k = ev.fold_of(game)
        for label in systems:
            base = label.split("_")[0]
            c = final_costs(costs) if label.endswith("_fin") else match_costs(costs, game)
            if base == BASE:
                jobs[(game, label)] = ev.make_job(ctx, game, "c22", label, view, None, c)
                continue
            weights = ctx.npz(k)
            if not os.path.isfile(weights):
                raise SystemExit("{} is missing: run `export` first".format(weights))
            jobs[(game, label)] = ev.make_job(ctx, game, base, label, view, weights, c)
    return jobs


def cmd_run(args):
    ctx = make_ctx(args)
    costs = load_costs(ctx.work)
    systems = tuple(args.systems)
    jobs = raw_jobs(ctx, costs, systems)
    if not args.no_matched and any(s in MATCHED_SYSTEMS for s in systems):
        jobs.update({(g, "matched|" + l): j for (g, l), j in raw_jobs(
            ctx, costs, [s for s in systems if s in MATCHED_SYSTEMS], ev.VIEW_MATCHED,
            DEV).items()})
    ev.run_all(ctx, jobs)
    missing = ev.missing_runs(jobs)
    print("run: {} of {} runs present".format(len(jobs) - len(missing), len(jobs)))
    return 1 if missing else 0


# -- scores ---------------------------------------------------------------------------------------

def score_table(ctx, jobs, games, labels, shifts_for):
    """{game: {label: {ref: {shift: score}}}}, and the reference md5s."""
    scores, ref_md5 = {}, {}
    for game in games:
        refs = ev.references(ctx, game)
        ref_md5[game] = refs["provided_md5"]
        scores[game] = {}
        for label in labels:
            job = jobs.get((game, label))
            if job is None or not os.path.isfile(job["csv"]):
                continue
            scores[game][label] = {ref: {str(s): ev.score_csv(ctx, job, refs[ref], s)
                                         for s in shifts_for(label)}
                                   for ref in ("ctl2", "provided")}
    return scores, ref_md5


def table(scores, games, label, ref, shift=0):
    return {g: scores[g][label][ref][str(shift)]["spell_f1"] for g in games
            if label in scores[g] and str(shift) in scores[g][label][ref]}


def summaries(scores, games, cands):
    out = {}
    for ref in ("ctl2", "provided"):
        base = table(scores, games, BASE, ref)
        for cand in cands:
            for suffix, kind in (("", "nested"), ("_fin", "final")):
                x = table(scores, games, cand + suffix, ref)
                if sorted(x) == sorted(base) and x:
                    out["{}|{}|{}|0".format(cand, kind, ref)] = summary(x, base)
            for shift in ev.SHIFTS:
                if shift == 0:
                    continue
                x = table(scores, games, cand, ref, shift)
                b = table(scores, games, BASE, ref, shift)
                if sorted(x) == sorted(b) and x:
                    out["{}|nested|{}|{}".format(cand, ref, shift)] = summary(x, b)
    return out


def cmd_score(args):
    ctx = make_ctx(args)
    costs = load_costs(ctx.work)
    labels = RAW_SYSTEMS + H_SYSTEMS
    jobs = raw_jobs(ctx, costs, [s for s in labels if s in RAW_SYSTEMS or costs.get("H_nested")])
    missing = [k for k in ev.missing_runs({k: v for k, v in jobs.items()
                                           if k[1] in RAW_SYSTEMS})]
    if missing:
        raise SystemExit("runs missing (run `run` first):\n  " + "\n  ".join(missing))
    shifts = lambda label: ev.SHIFTS if label in (BASE, "M", "H") else (0,)  # noqa: E731
    scores, ref_md5 = score_table(ctx, jobs, ALL, labels, shifts)
    mjobs = raw_jobs(ctx, costs, MATCHED_SYSTEMS, ev.VIEW_MATCHED, DEV)
    matched, _ = score_table(ctx, mjobs, DEV, MATCHED_SYSTEMS, lambda label: (0,))
    cands = ["M"] + (["H"] if all("H" in scores[g] for g in ALL) else [])
    sm = summaries(scores, ALL, cands)
    dev_sm = summaries({g: scores[g] for g in DEV}, DEV, cands)
    matched_sm = summaries(matched, DEV, ["M"]) if all(
        "M" in matched[g] and BASE in matched[g] for g in DEV) else {}
    rec = {"games": list(ALL), "complete": all(all(l in scores[g] for l in RAW_SYSTEMS)
                                               for g in ALL),
           "rehearsal": rehearsal_of(ctx.work), "scores": scores, "reference_md5": ref_md5,
           "summaries": sm, "summaries_metrica_dfl": dev_sm, "matched_scores": matched,
           "summaries_matched": matched_sm,
           "costs": {"M_nested": costs["M_nested"], "M_final": costs["M_final"],
                     "H_nested": costs.get("H_nested"), "H_final": costs.get("H_final")},
           "runs": {"{}|{}".format(*k): {"key": j["key"], "weights_md5": j["weights_md5"]}
                    for k, j in jobs.items() if os.path.isfile(j["csv"])},
           "engine_source_hash": ev.engine_hash()}
    write_json(os.path.join(ctx.work, "scores.json"), rec)
    for key in sorted(sm):
        r = sm[key]
        print("{:<26} X {:.4f}  c23 {:.4f}  delta {:+.4f} [{:+.4f}, {:+.4f}]".format(
            key, r["x_mean"], r["c23_mean"], r["delta"], r["low"], r["high"]))
    return 0


# -- stress ---------------------------------------------------------------------------------------

def stress_games(work):
    """The matches of the stress step: all 28; in a rehearsal only (preflight.json says so), the
    subset named in R3_EVAL_REHEARSAL_GAMES (space-separated), to rehearse at low load."""
    names = os.environ.get("R3_EVAL_REHEARSAL_GAMES", "").split()
    if not names:
        return ALL
    if not rehearsal_of(work):
        raise SystemExit("R3_EVAL_REHEARSAL_GAMES is for rehearsals only")
    return tuple(folds.check_all(names))


def cmd_stress(args):
    import score_game as sg
    ctx = make_ctx(args)
    costs = load_costs(ctx.work)
    games = stress_games(ctx.work)
    base = raw_jobs(ctx, costs, STRESS_SYSTEMS, games=games)
    sjobs = {}
    for game in games:
        k = ev.fold_of(game)
        c = match_costs(costs, game)
        for view in (ev.VIEW_STRESS, ev.VIEW_U30):
            for label in STRESS_SYSTEMS:
                system = "c22" if label == BASE else label
                weights = ctx.npz(k) if label != BASE else None
                sjobs[(game, view, label)] = ev.make_job(ctx, game, system, label, view,
                                                         weights, c)
    ev.run_all(ctx, dict(list(base.items()) + list(sjobs.items())))
    missing = ev.missing_runs(sjobs) + ev.missing_runs(base)
    if missing:
        raise SystemExit("runs missing:\n  " + "\n  ".join(missing))
    unstressed = {r: {s: {} for s in STRESS_SYSTEMS} for r in ("ctl2", "provided")}
    stressed = {r: {s: {} for s in STRESS_SYSTEMS} for r in ("ctl2", "provided")}
    dcount = {v: {s: {} for s in STRESS_SYSTEMS} for v in (ev.VIEW_STRESS, ev.VIEW_U30)}
    mask_same = {v: {} for v in (ev.VIEW_STRESS, ev.VIEW_U30)}
    for game in games:
        refs = ev.references(ctx, game)
        n = len(refs["provided"])
        for ref in ("ctl2", "provided"):
            for s in STRESS_SYSTEMS:
                unstressed[ref][s][game] = ev.score_csv(ctx, base[(game, s)],
                                                        refs[ref])["spell_f1"]
                stressed[ref][s][game] = ev.score_csv(ctx, sjobs[(game, ev.VIEW_STRESS, s)],
                                                      refs[ref])["spell_f1"]
        for view in (ev.VIEW_STRESS, ev.VIEW_U30):
            masks = []
            for s in STRESS_SYSTEMS:
                job = sjobs[(game, view, s)]
                with np.load(job["hidden"]) as npz:
                    hidden = np.asarray(npz["hidden"], dtype=bool)
                masks.append(hidden)
                codes = sg.read_codes(job["csv"], n)
                dcount[view][s][game] = ev.d_share_hidden(hidden, refs["provided"], codes)
            mask_same[view][game] = bool(all(np.array_equal(masks[0], m) for m in masks[1:]))
    rec = {"games": list(games), "complete": games == ALL, "rehearsal": rehearsal_of(ctx.work),
           "unstressed": unstressed, "stressed": stressed,
           "drops": {r: {s: ev.drops(unstressed[r][s], stressed[r][s]) for s in STRESS_SYSTEMS}
                     for r in ("ctl2", "provided")},
           "d_hidden": {v: {s: {"per_match": {g: list(c) for g, c in dcount[v][s].items()},
                                "per_provider": ev.pooled_share(dcount[v][s])}
                            for s in STRESS_SYSTEMS} for v in dcount},
           "hidden_mask_identical": mask_same,
           "summaries": {r: {"stressed": summary(stressed[r]["M"], stressed[r][BASE]),
                             "unstressed": summary(unstressed[r]["M"], unstressed[r][BASE])}
                         for r in ("ctl2", "provided")},
           "runs": {"|".join(k): {"key": j["key"]} for k, j in sjobs.items()}}
    write_json(os.path.join(ctx.work, "stress.json"), rec)
    for ref in ("ctl2", "provided"):
        s, u = rec["summaries"][ref]["stressed"], rec["summaries"][ref]["unstressed"]
        print("M {} stressed delta {:+.4f} [{:+.4f}, {:+.4f}]  unstressed {:+.4f}".format(
            ref, s["delta"], s["low"], s["high"], u["delta"]))
    return 0


# -- the label-free step (competition match, or a development match in a rehearsal) ---------------

class Match(object):
    """The match of the label-free step: the competition match, or `--as-match GAME`."""

    def __init__(self, ctx, as_match=None, section12a=False):
        from analysis import external_sweep as xs
        self.ctx = ctx
        self.rehearsal = as_match is not None
        self.section12a = bool(section12a)
        if self.rehearsal:
            if as_match not in DEV:
                raise SystemExit("--as-match takes a development match: {}".format(DEV))
            folds.check(as_match)
            self.game = as_match
            self.root = xs.data_root(as_match, "raw", *ev.roots(ctx.args))
            self.folder = os.path.join(ctx.work, "bar_as_" + as_match)
        else:
            self.game = folds.COMPETITION
            self.root = competition_data_dir()
            self.folder = os.path.join(ctx.work, "bar")
        # The step at the rule's pair stays in `folder`; a rerun at the Section 12a pair writes
        # its own folder beside it (bar_12a/, bar_as_<GAME>_12a/), and pair12a.json stays in
        # the folder of the rule's pair.
        self.rule_folder = self.folder
        if self.section12a:
            self.folder = self.folder + "_12a"

    def run(self, label, system, weights, costs, view=ev.VIEW_RAW, must_exist=False):
        """CSV (and hidden mask for the V2 view) of one system on this match, cached.
        must_exist: return the cached run, or raise SystemExit; never run the engine."""
        if self.rehearsal:
            job = ev.make_job(self.ctx, self.game, system, label, view, weights, costs)
            if must_exist and not os.path.isfile(job["csv"]):
                raise SystemExit("{} {} is not in the cache ({})".format(self.game, label,
                                                                          job["csv"]))
            ev.run_one(self.ctx, job)
            return job["csv"], job["hidden"]
        over = ev.overrides(system, weights, costs)
        if self.section12a and system == "c22":
            # c23 does not depend on the pair: reuse the run of the step at the rule's pair
            # (same cache key) when it exists, never a second c23 run on this match.
            try:
                return competition_run(self.ctx, self.rule_folder, label, over, view, self.root,
                                       must_exist=True)
            except SystemExit:
                pass
        return competition_run(self.ctx, self.folder, label, over, view, self.root, must_exist)

    def tactical(self, guarded):
        from engine import data
        from engine.possession import engine_tactical
        if not self.rehearsal:
            return engine_tactical(None, self.game) if guarded else data.load_tactical(self.game)
        tac = data.load_tactical(self.game, data_dir=self.root,
                                 cache_dir=ev.data_cache(self.ctx, self.game))
        return guard(tac, self.game) if guarded else tac


def guard(tac, game):
    """engine_tactical() on an already loaded tactical dict."""
    from engine import data, tracks
    from engine.config import Config
    cfg = Config.default()
    pitch = data.pitch_m(game)
    patched, _, _ = tracks.guard_goalkeeper_swaps(tracks.speed_from_positions(tac, pitch),
                                                  data.graded_blocks(game), cfg, pitch)
    return patched


def competition_run(ctx, folder, label, over, view, data_dir, must_exist=False):
    """engine.run (raw) or corrupt.py engine (V2 view) on the competition match, with
    MTAH_DATA_DIR set to its data folder (round 2's bar_run removed it, which fails in a
    checkout without data symlinks). Cached by a key as evaluate.bar_run."""
    from analysis import external_sweep as xs
    resolved, _ = xs.resolve_config(over)
    weights = over.get("LEARNED_WEIGHTS")
    key_obj = {"game": folds.COMPETITION, "config": resolved, "engine": ev.engine_hash(),
               "weights_md5": ctx.weights_md5(weights) if weights else None}
    if view != ev.VIEW_RAW:
        key_obj.update(view=view, corrupt=ev.corrupt_hash(), corrupt_args=list(ev.VIEW_ARGS[view]))
    if over.get("LEARNED_PF") == "on":
        key_obj["pf_md5"] = ctx.weights_md5(over["LEARNED_PF_WEIGHTS"])
    key = hashlib.sha1(json.dumps(key_obj, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    where = os.path.join(folder, "runs" if view == ev.VIEW_RAW else view)
    csv = os.path.join(where, "{}_{}.csv".format(label, key))
    hidden = csv[:-4] + ".hidden.npz" if view != ev.VIEW_RAW else None
    if os.path.isfile(csv) and (hidden is None or os.path.isfile(hidden)):
        return csv, hidden
    if must_exist:
        raise SystemExit("{} {} is not in the cache ({})".format(folds.COMPETITION, label, csv))
    os.makedirs(where, exist_ok=True)
    base = csv[:-4]
    write_json(base + ".config.json", over)
    env = dict(os.environ, MTAH_DATA_DIR=data_dir)
    if view == ev.VIEW_RAW:
        command = [sys.executable, "-m", "engine.run", "--out", base + ".tmp.csv", "--config",
                   base + ".config.json"]
    else:
        command = [sys.executable, "-m", "tools.learned_spike.corrupt", "engine", "--game",
                   folds.COMPETITION, "--out", base + ".tmp.csv", "--config",
                   base + ".config.json", "--hidden-out", base + ".hidden.tmp.npz",
                   "--allow-competition"] + list(ev.VIEW_ARGS[view])
    log("run {} {} {}".format(view, folds.COMPETITION, label))
    with open(base + ".log", "w") as out:
        code = subprocess.call(command, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
    if code != 0:
        raise RuntimeError("run failed on {} {}, see {}.log".format(folds.COMPETITION, label,
                                                                   base))
    write_json(base + ".meta.json", dict(key_obj, label=label,
                                         output_md5=md5_file(base + ".tmp.csv")))
    if hidden:
        os.replace(base + ".hidden.tmp.npz", hidden)
    os.replace(base + ".tmp.csv", csv)
    return csv, hidden


def require_criteria(work, numbers, rehearsal):
    """Refuse the label-free step before criteria 1 to 5 are scored and 3 to 5 pass (R3 4)."""
    crits = criteria_so_far(work)
    bad = [i for i in numbers if crits.get(i) is None or not crits[i]["pass"]]
    if bad:
        text = "criterion {} not passed or not computed".format(", ".join(bad))
        if not rehearsal:
            raise SystemExit("the label-free step runs only after criteria 1 to 5 (R3 Section "
                             "4): " + text)
        log("rehearsal: " + text + " (recorded, the rehearsal continues)")
        return text
    return None


def label_free_costs(work, rule, structure, pair=None):
    """The costs of the label-free step (and of the configuration that would ship).

    rule "b" (default, R3_DEVIATION_COSTS.md, owner's approval of 12 Nov 2026): the pair of
    r3_views rule b on all eight development matches (W/views.json); "a": rule B as frozen
    (final/nested_costs.json). The frozen pair is recorded beside it either way. `structure`
    turns on Config.LEARNED_STRUCTURE for M (R3_DEVIATION_STRUCTURE.md).
    `pair` (H-A, alive-dead): the Section 12a pair (R3_DEVIATION_COSTS.md 12a), which replaces
    the rule's pair; the rule's pair is recorded beside it as "rule_pair".
    """
    costs = load_costs(work)
    out = final_costs(costs)
    out["frozen_rule_pair"] = list(out["M"])
    out["rule"] = "a (rule B as frozen)"
    if rule == "fixed" or costs.get("fixed"):
        if rule != "fixed" or not costs.get("fixed") or pair is not None:
            raise SystemExit("round 4: the label-free costs are the costs-fixed pair "
                             "(--costs-rule fixed, no --pair)")
        out["rule"] = ("fixed: pair {} and prior bonus {:g} given on the command line "
                       "(LEARNED_STAGE_PROTOCOL_R4.md Section 4; no cost search, no walk)".format(
                           tuple(out["M"]), out["prior_bonus"]))
        out["structure"] = bool(costs.get("structure"))
        out["source"] = "label-free: " + out["rule"]
        return out
    if rule in ("b", "c"):
        path = os.path.join(work, "views.json")
        if not os.path.isfile(path):
            raise SystemExit("{} is missing: the label-free costs of rule {} come from "
                             "r3_views grid".format(path, rule))
        views = read_json(path)
        out["M"] = tuple(float(x) for x in views["grid"][views["rules"][rule]["final"]])
        out["rule"] = "{} (r3_views rule {}, score {:+.4f})".format(
            "best worst case over the as-provided and loose-touch views" if rule == "b"
            else "equal weight of the two views", rule, views["rules"][rule]["final_score"])
    if pair is not None:
        out["rule_pair"] = list(out["M"])
        out["rule_of_rule_pair"] = out["rule"]
        out["M"] = (float(pair[0]), float(pair[1]))
        out["rule"] = ("Section 12a (R3_DEVIATION_COSTS.md 12a: the first pair of the rule {} "
                       "ranking with Score > 0 whose output is inside the four 8e bounds; the "
                       "rule's pair {} failed 8e)".format(rule, tuple(out["rule_pair"])))
        out["section12a"] = True
    out["structure"] = bool(structure)
    out["source"] = "label-free: " + out["rule"]
    return out


def final_inputs(ctx, args, match):
    final = os.path.abspath(args.final_npz)
    for path in (final, ctx.folds_all()):
        if not os.path.isfile(path):
            raise SystemExit("{} is missing: run `export` first".format(path))
    pair = getattr(args, "pair", None)
    if pair is not None:
        check_pair12a(match, pair)
    costs = label_free_costs(ctx.work, args.costs_rule, not args.no_structure, pair)
    prev = os.path.join(match.folder, "bar.json")
    if os.path.isfile(prev) and read_json(prev).get("final_npz_md5") != ctx.weights_md5(final):
        raise SystemExit("{} was made with other weights: the label-free step runs once "
                         "(R3 Section 4)".format(prev))
    return final, costs


def cmd_bar(args):
    import argparse as ap
    from engine import data
    from tools.learned_spike import bar_report as br
    ctx = make_ctx(args)
    match = Match(ctx, args.as_match, args.pair is not None)
    order_note = require_criteria(ctx.work, ("5",) if variant_of(ctx.work) in R4_LIKE
                                  else ("3", "4", "5"), match.rehearsal)
    final, costs = final_inputs(ctx, args, match)
    preds = {}
    preds[BASE], _ = match.run(BASE, "c22", None, costs)
    expected = C23_BAR_MD5 if not match.rehearsal else C23_DEV_MD5[match.game]
    base_md5 = md5_file(preds[BASE])
    if base_md5 != expected:
        raise SystemExit("c23 on {}: md5 {} != {} (switch-off identity)".format(
            match.game, base_md5, expected))
    preds["M"], _ = match.run("M", "M", final, costs)
    preds["Mfold"], _ = match.run("Mfold", "M", ctx.folds_all(), costs)

    def load(game):
        if game == match.game:
            return match.tactical(guarded=True)
        return ev.ext_loader(ctx)(game)
    pf = pf_of(ctx.work)                              # round 6, variant R6E (None otherwise)
    bar_seeds, bar_copies = ev.dump_probs(ctx, "final", match.game, final, load, pf_for(pf))
    sc_rates = {"final": {}, "fold": {}}
    for game in folds.SC:
        n = data.frame_count(game)
        graded = br.graded_mask(br.graded_runs(game), n)
        for tag, weights in (("final", final), ("fold", ctx.npz(ev.fold_of(game)))):
            seeds, copies = ev.dump_probs(ctx, tag, game, weights, load,
                                          pf_for(pf, None if tag == "final" else
                                                 ev.fold_of(game)))
            sc_rates[tag][game] = {"seed": br.prob_group(seeds, graded)["share_any_disagree"],
                                   "copy": br.prob_group(copies, graded)["share_any_disagree"]}
    ext_provided, ext_ctl2 = [], {}
    for game in ALL:
        refs = ev.references(ctx, game)
        ext_provided.append("M={}:{}".format(game, refs["path"]))
        ext_ctl2[game] = refs["ctl2"]
    band_games = list(folds.train_games(pff=False))       # R3b: no PFF file is read
    ns = ap.Namespace(pred=["{}={}".format(k, preds[k]) for k in BAR_SYSTEMS], restart_ref=BASE,
                      ext=ext_provided,
                      seed_probs=["s{}={}".format(i, p) for i, p in enumerate(bar_seeds)],
                      copy_probs=["c{}={}".format(i, p) for i, p in enumerate(bar_copies)],
                      feature_cache=os.path.abspath(args.feature_cache),
                      feature_games=band_games,
                      # Rehearsal only: the development match and its guarded ATD (the ball
                      # columns are those of the raw ATD; the features are what the network
                      # reads, as bar_features() on the competition match).
                      game=match.game if match.rehearsal else None,
                      load_tactical=(lambda g: match.tactical(guarded=True))
                      if match.rehearsal else None)
    report = br.build_report(ns)
    os.makedirs(match.folder, exist_ok=True)
    with open(os.path.join(match.folder, "bar_report.txt"), "w") as handle:
        br.print_report(report, handle)
    write_json(os.path.join(match.folder, "bar_report.json"), report)
    ctl2_stats = [dict(br.system_stats(c, br.graded_runs(g), None), game=g)
                  for g, c in ext_ctl2.items()]
    lf_path = os.path.join(ctx.work, "costs_label_free.json")
    if match.section12a and os.path.isfile(lf_path) and not read_json(lf_path).get("section12a"):
        # keep the record of the rule's pair before the 12a pair replaces it
        write_json(os.path.join(ctx.work, "costs_label_free.rule_pair.json"), read_json(lf_path))
    write_json(lf_path, dict({
        "M": {"cost_ha": costs["M"][0], "cost_ad": costs["M"][1]},
        "H": {"cost_ha": costs["H"]}, "rule": costs["rule"],
        "frozen_rule_pair": costs["frozen_rule_pair"], "structure": costs["structure"],
        "prior_bonus": costs.get("prior_bonus", 0.0)},
        **({"pf": costs["pf"]} if costs.get("pf") else {}),
        **({"section12a": True, "rule_pair": costs["rule_pair"]} if match.section12a else {})))
    rec = {
        "complete": True, "rehearsal": match.rehearsal or rehearsal_of(ctx.work),
        "section12a": match.section12a,
        "game": match.game, "order_note": order_note, "final_npz": final,
        "final_npz_md5": ctx.weights_md5(final), "folds_all_md5": ctx.weights_md5(ctx.folds_all()),
        "final_costs": costs, "c23_md5": base_md5,
        "md5": {k: md5_file(preds[k]) for k in BAR_SYSTEMS},
        "systems": {k: report["systems"][k] for k in BAR_SYSTEMS},
        "vetoes": {c: ev.crit6(report["systems"][c]) for c in (BASE, "M")},
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
        "envelope_provided": report["envelope"]["M"],
        "envelope_ctl2_recorded": br.envelope(report["systems"]["M"], ctl2_stats),
        "agreement_final_vs_fold": report["pairs"]["M vs Mfold"]["agreement"],
        "anatomy_vs_c23": report["pairs"]["c23 vs M"],
        "engine_source_hash": ev.engine_hash(),
    }
    if match.section12a:
        # the engine run at the 12a pair must write the labels the walk decoded
        import score_game as sg
        walk = read_json(os.path.join(match.rule_folder, "pair12a.json"))["chosen"]
        got = ev.codes_md5(sg.read_codes(preds["M"], data.frame_count(match.game)))
        rec["walk_labels_md5"] = {"walk": walk.get("labels_md5"), "engine": got,
                                  "equal": walk.get("labels_md5") == got}
        print("Section 12a: engine M labels md5 {} {} the walk's {}".format(
            got, "equals" if rec["walk_labels_md5"]["equal"] else "DIFFERS FROM",
            walk.get("labels_md5")))
    write_json(os.path.join(match.folder, "bar.json"), rec)
    for c in (BASE, "M"):
        s = rec["systems"][c]
        print("{:<4} possessions {} D share {:.3f} D segments {} graded X {}  vetoes {}".format(
            c, s["possessions"], s["d_share"], s["d_segments"], s["graded_x"],
            "pass" if rec["vetoes"][c]["pass"] else "FAIL"))
    return 0


def cmd_agreement(args):
    """Section 8a and 8b baselines on the external matches only (no competition match): the
    seed and copy disagreement of the fold ensembles (out of sample, R1 Section 12.1) and of the
    final ensemble on the 20 SkillCorner matches, and the limits 1.5 x the fold means. The
    probability dumps are the ones `bar` reuses."""
    from engine import data
    from tools.learned_spike import bar_report as br
    ctx = make_ctx(args)
    final = os.path.abspath(args.final_npz)
    load = ev.ext_loader(ctx)
    pf = pf_of(ctx.work)                              # round 6, variant R6E (None otherwise)
    rates = {"final": {}, "fold": {}}
    for game in folds.SC:
        n = data.frame_count(game)
        graded = br.graded_mask(br.graded_runs(game), n)
        for tag, weights in (("final", final), ("fold", ctx.npz(ev.fold_of(game)))):
            seeds, copies = ev.dump_probs(ctx, tag, game, weights, load,
                                          pf_for(pf, None if tag == "final" else
                                                 ev.fold_of(game)))
            rates[tag][game] = {"seed": br.prob_group(seeds, graded)["share_any_disagree"],
                                "copy": br.prob_group(copies, graded)["share_any_disagree"]}
    rec = {"sc_rates": rates, "final_npz_md5": ctx.weights_md5(final),
           "rehearsal": rehearsal_of(ctx.work)}
    variant = variant_of(ctx.work)
    for key in ("seed", "copy"):
        fold_mean = float(np.mean([r[key] for r in rates["fold"].values()]))
        part = "a" if key == "seed" else "b"
        rec[key] = {"sc_mean_fold": fold_mean,
                    "sc_mean_final": float(np.mean([r[key] for r in rates["final"].values()])),
                    "limit_8" + part: limit8(part, fold_mean, variant)}
        if variant in CAPPED:
            rec[key]["limit_rule"] = "min(1.5 x fold mean {:.4f}, round 4's {})".format(
                fold_mean, R5_LIMIT_CAPS[part])
    write_json(os.path.join(ctx.work, "agreement.json"), rec)
    print("8a seed: SkillCorner fold mean {:.4f}, limit {:.4f} (final in-sample {:.4f}); "
          "8b copy: fold mean {:.4f}, limit {:.4f} (final {:.4f})".format(
              rec["seed"]["sc_mean_fold"], rec["seed"]["limit_8a"], rec["seed"]["sc_mean_final"],
              rec["copy"]["sc_mean_fold"], rec["copy"]["limit_8b"], rec["copy"]["sc_mean_final"]))
    return 0


def cmd_v23(args):
    import score_game as sg
    from engine import data
    from tools.learned_spike import bar_report as br
    ctx = make_ctx(args)
    match = Match(ctx, args.as_match, args.pair is not None)
    require_criteria(ctx.work, ("3", "4", "5"), match.rehearsal)
    final, costs = final_inputs(ctx, args, match)
    game = match.game
    n = data.frame_count(game)
    runs = br.graded_runs(game)
    graded = br.graded_mask(runs, n)
    raw, hid, masks = {}, {}, {}
    for label, system, weights in ((BASE, "c22", None), ("M", "M", final)):
        csv, _ = match.run(label, system, weights, costs)
        raw[label] = sg.read_codes(csv, n)
        csv, mask_path = match.run(label, system, weights, costs, ev.VIEW_V2)
        hid[label] = sg.read_codes(csv, n)
        with np.load(mask_path) as npz:
            masks[label] = np.asarray(npz["hidden"], dtype=bool)
    if not np.array_equal(masks[BASE], masks["M"]):
        raise SystemExit("V2: the hidden-ball masks differ between systems")
    hidden = masks[BASE]
    tac = match.tactical(guarded=False)
    ball_missing = ~np.isfinite(np.asarray(tac["bx"], dtype=np.float64))
    v2 = {"systems": {s: ev.v2_counts(raw[s], hid[s], hidden, graded) for s in (BASE, "M")},
          "k_frames": int((hidden & graded).sum()), "view": list(ev.VIEW_ARGS[ev.VIEW_V2]),
          "seed": ev.V2_SEED, "hidden_mask_identical": True}
    v3 = {s: ev.d_share_missing(raw[s], runs, ball_missing) for s in (BASE, "M")}
    rec = {"complete": True, "rehearsal": match.rehearsal or rehearsal_of(ctx.work),
           "game": game, "final_npz": final, "final_npz_md5": ctx.weights_md5(final),
           "final_costs": costs, "engine_source_hash": ev.engine_hash(), "v2": v2, "v3": v3,
           "v3_ball_missing_graded_frames": int((graded & ball_missing).sum())}
    rec["criteria"] = {"9": crit9(rec), "10": crit10(rec)}
    write_json(os.path.join(match.folder, "v23.json"), rec)
    for s in (BASE, "M"):
        r = v2["systems"][s]
        print("{:<4} V2 flip {} ({} of {})  V3 {}".format(s, ev.fmt(r["flip"]), r["flipped_to_d"],
                                                       r["alive_raw"], ev.fmt(v3[s])))
    return 0


# -- criterion 8 ----------------------------------------------------------------------------------

def cmd_repro(args):
    """Assemble criterion 8 (R1 Section 6) from its parts.

    a. the final M on the label-free match a second time: same md5 as the `bar` run;
    b. tools/repro_fresh_clone.sh (log given with --fresh-clone-log, PASS on its last lines);
    c. repro_check.py run (numpy engine against torch on the 8 open matches): its JSON;
    d. the learned-stage unit tests: the pytest log given with --tests-log.
    """
    ctx = make_ctx(args)
    match = Match(ctx, args.as_match, args.pair is not None)
    final, costs = final_inputs(ctx, args, match)
    bar = read_json(os.path.join(match.folder, "bar.json"))
    over = ev.overrides("M", final, costs)
    tmp = os.path.join(match.folder, "repro", "M_run2.csv")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    if match.rehearsal:
        from analysis import external_sweep as xs
        cfg = tmp[:-4] + ".config.json"
        write_json(cfg, over)
        command = [sys.executable, os.path.abspath(xs.__file__), "--engine-child",
                   ev.data_cache(ctx, match.game), "--game", match.game, "--config", cfg,
                   "--out", tmp]
    else:
        cfg = tmp[:-4] + ".config.json"
        write_json(cfg, over)
        command = [sys.executable, "-m", "engine.run", "--out", tmp, "--config", cfg]
    with open(tmp[:-4] + ".log", "w") as handle:
        code = subprocess.call(command, cwd=ROOT, env=dict(os.environ, MTAH_DATA_DIR=match.root),
                               stdout=handle, stderr=subprocess.STDOUT)
    run2 = md5_file(tmp) if code == 0 else None
    part_a = {"run1_md5": bar["md5"]["M"], "run2_md5": run2,
              "pass": bool(run2 is not None and run2 == bar["md5"]["M"])}
    part_b = {"pass": None, "note": "not run (needs a commit with the config defaults and "
                                    "pip access, see the adoption runbook)"}
    if args.fresh_clone_log:
        text = open(args.fresh_clone_log).read()
        part_b = {"log": args.fresh_clone_log, "pass": "PASS" in text.splitlines()[-5:][-1]
                  if text.strip() else False}
        part_b["pass"] = bool(any(line.strip().startswith("PASS")
                                  for line in text.splitlines()[-10:]))
    part_c = {"pass": None, "note": "repro_check.py JSON not given"}
    if args.repro_check and os.path.isfile(args.repro_check):
        rc = read_json(args.repro_check)
        v = rc.get("summary", {})          # repro_check.verdict(): M decides, H is recorded
        part_c = {"record": args.repro_check, "pass": bool(v.get("M", {}).get("pass")),
                  "M": v.get("M"), "c23_equals_switch_off_record_all":
                  rc.get("c22_equals_switch_off_record_all"),
                  "weights_md5": rc.get("weights_md5")}
    part_d = {"pass": None, "note": "tests log not given"}
    if args.tests_log and os.path.isfile(args.tests_log):
        lines = open(args.tests_log).read().splitlines()
        tail = " ".join(lines[-3:])
        part_d = {"log": args.tests_log, "summary": tail,
                  "pass": bool(" passed" in tail and " failed" not in tail
                               and " error" not in tail)}
    parts = {"a": part_a, "b": part_b, "c": part_c, "d": part_d}
    complete = all(p["pass"] is not None for p in parts.values())
    rec = {"pass": bool(complete and all(p["pass"] for p in parts.values())) if complete
           else (False if any(p["pass"] is False for p in parts.values()) else None),
           "complete": complete, "parts": parts, "rehearsal": match.rehearsal,
           "weights_md5": ctx.weights_md5(final), "platform": "macOS arm64 (this machine)",
           "linux": "not part of criterion 8 here; R3 Section 5 asks for Linux x86-64 byte "
                    "identity unless the owner accepts macOS arm64 only (adoption runbook)"}
    write_json(os.path.join(match.folder, "repro.json"), rec)
    print("criterion 8: a {} b {} c {} d {} -> {}".format(
        *[ev.fmt(parts[p]["pass"]) for p in "abcd"], ev.fmt(rec["pass"])))
    return 0


# -- Section 12a: the pair walk if the rule's pair fails 8e (R3_DEVIATION_COSTS.md 12a) -----------

E_KEYS = ev.ENVELOPE_KEYS["M"]      # the four 8e statistics of M


def score_ranking(views, rule="b"):
    """[(grid index, Score)] of every pair of views.json, best Score first; ties go to the lower
    H-A, then the lower alive-dead (the grid order), as Section 12 (r3_views.pick)."""
    from tools.learned_spike import r3_views
    score_of = r3_views.rule_scores(views["spell_f1"], views["c23"], list(DEV), rule)
    rows = [(i, float(score_of(i))) for i in range(len(views["grid"]))]
    return sorted(rows, key=lambda r: (-round(r[1], 12), r[0]))


def stats_8e(codes, runs, ext_stats, bounds=None):
    """(the four 8e rows of envelope(), inside all four) of one decoded output. `bounds`
    ({key: (lo, hi)}, rehearsal only) replaces the external bounds of those keys."""
    from tools.learned_spike import bar_report as br
    stats = br.system_stats(codes, runs, None)
    env = br.envelope(stats, ext_stats)
    rows = rebound({k: env.get(k) for k in E_KEYS}, bounds)
    extra = {"short_d_segments": short_d_segments(codes, runs),
             "d_segments": stats["d_segments"], "possessions": stats["possessions"]}
    return rows, bool(all(r is not None and r["inside"] for r in rows.values())), extra


def short_d_segments(codes, runs, max_frames=2):
    """The number of D segments of at most `max_frames` frames (filled codes, graded runs, as
    bar_report.system_stats counts D segments). Recorded so that a decoder effect is visible:
    with cost H-A > 2 x cost alive-dead the Viterbi path changes team through a one-frame D."""
    import possession_metrics as pm
    filled = pm.fill_abstentions(np.asarray(codes, dtype=np.int8), runs)
    return int(sum(1 for a, b, code in pm.segments(filled, runs)
                   if code == pm.D and b - a <= max_frames))


def rebound(rows, bounds):
    """envelope() rows with the bounds of `bounds` ({key: (lo, hi)}) put in, inside recomputed."""
    out = {}
    for k, r in rows.items():
        if r is not None and bounds and k in bounds:
            lo, hi = bounds[k]
            r = dict(r, ext_min=lo, ext_max=hi, inside=bool(lo <= r["bar"] <= hi),
                     rehearsal_bound=True)
        out[k] = r
    return out


def parse_bound(text):
    """KEY=LO,HI of --rehearse-bound."""
    try:
        key, rng = text.split("=", 1)
        lo, hi = (float(x) for x in rng.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("--rehearse-bound takes KEY=LO,HI")
    if key not in E_KEYS:
        raise argparse.ArgumentTypeError("--rehearse-bound: KEY is one of {}".format(E_KEYS))
    return key, (lo, hi)


def walk_12a(ranking, grid, decode_stats, full=False):
    """(rows, chosen row or None) of the Section 12a walk.

    ranking: score_ranking(); decode_stats(pair) -> (8e rows, inside), or None when 12a does not
    apply (nothing is decoded). Pairs with Score <= 0 are never used (and not decoded). The
    chosen pair is the first with Score > 0 whose output is inside all four bounds; with
    full=True every pair with Score > 0 is decoded and listed.
    """
    rows, chosen = [], None
    for rank, (i, score) in enumerate(ranking, 1):
        pair = [float(grid[i][0]), float(grid[i][1])]
        row = {"rank": rank, "index": i, "pair": pair, "score": score}
        if score <= 0:
            row.update(decoded=False, inside=None, note="Score <= 0: never used (12a)")
        elif decode_stats is None:
            row.update(decoded=False, inside=None, note="12a does not apply: not decoded")
        elif chosen is not None and not full:
            row.update(decoded=False, inside=None, note="after the chosen pair: not decoded")
        else:
            got = decode_stats(tuple(pair))
            stats, inside = got[0], got[1]
            row.update(decoded=True, stats=stats, inside=inside)
            if len(got) > 2:
                row.update(got[2])
            if inside and chosen is None:
                chosen = row
                row["chosen"] = True
        rows.append(row)
    return rows, chosen


def applies_12a(bar, explanations=None, ignore=()):
    """(applies, reason): 12a moves the pair only if the rule's pair fails 8e and passes the
    other parts of criteria 6 and 7 (for any other failure Section 12's sentence stands)."""
    b8 = bounds8(bar, explanations)
    c6 = crit6(bar["systems"]["M"])
    if b8["e"]["pass"]:
        return False, "the rule's pair passes 8e: Section 12a does not apply"
    other = [name for name, ok in (("criterion 6", c6["pass"]), ("8a", b8["a"]["pass"]),
                                   ("8b", b8["b"]["pass"]), ("8f", b8["f"]["pass"]))
             if not ok and name not in ignore]
    if other:
        return False, ("the rule's pair also fails {}: Section 12's sentence stands, c23 ships"
                       .format(", ".join(other)))
    note = "" if b8["c"]["pass"] else " (8c is not yet passed: it does not depend on the pair)"
    return True, "the rule's pair fails 8e only{}".format(note)


def ensemble_from_dumps(seed_paths, n_frames):
    """Frame probabilities of the final ensemble from the label-free step's dumps: the mean of
    the 4 seed files (each the mean over the 4 symmetric copies), rounded to PROB_DECIMALS at 5
    fps and upsampled, as engine.learned.frame_probabilities() (the network does not run)."""
    from engine import learned
    p5 = []
    for path in seed_paths:
        with np.load(path) as npz:
            p5.append(np.asarray(npz["p5"], dtype=np.float64))
    mean = np.round(np.mean(np.stack(p5), axis=0), learned.PROB_DECIMALS)
    return learned.upsample(mean, n_frames)


def pair12a_markdown(rec):
    w = ["## Section 12a walk on {}{} (R3_DEVIATION_COSTS.md 12a)\n".format(
        rec["game"], ", REHEARSAL on a development match" if rec["rehearsal"] else "")]
    if rec.get("rehearsal_bounds"):
        w.append("**Rehearsal bounds, not the real 8e bounds**: {} (set with --rehearse-bound "
                 "so that the walk moves).\n".format(json.dumps(rec["rehearsal_bounds"])))
    w.append("Rule's pair {} ({}); 8e at that pair: {}. {}.\n".format(
        tuple(rec["rule_pair"]), rec["rule"], "inside" if rec["rule_pair_inside_8e"]
        else "outside", rec["applies_reason"]))
    w.append("Ranking: Score of rule {} on the 80 pairs of views.json, ties to the lower H-A then "
             "alive-dead. Decode: the final ensemble's stored probabilities ({}), "
             "LEARNED_STRUCTURE {}; the network did not run. The decode at the rule's pair "
             "reproduces the engine output of the label-free step: {}.\n".format(
                 rec["ranking_rule"], rec["probabilities"], "on" if rec["structure"] else "off",
                 rec["reproduces_engine"]))
    w.append("Cells: value on the match [external min, max]. Last columns: D segments of at most "
             "2 frames / all D segments / possessions of the decoded output (recorded; with H-A "
             "> 2 x alive-dead the path changes team through a one-frame D).\n")
    w.append("| rank | pair (H-A / alive-dead) | Score | " + " | ".join(E_KEYS) +
             " | 8e | D <= 2 frames / D segments / possessions |\n|---|---|---|" +
             "---|" * (len(E_KEYS) + 2))
    for row in rec["rows"]:
        if row["decoded"]:
            cells = ["{} [{}, {}]".format(row["stats"][k]["bar"], row["stats"][k]["ext_min"],
                                          row["stats"][k]["ext_max"]) if row["stats"].get(k)
                     else "–" for k in E_KEYS]
            verdict = "inside" if row["inside"] else "outside"
            if row.get("chosen"):
                verdict = "**inside: chosen**"
        else:
            cells = ["–"] * len(E_KEYS)
            verdict = row["note"]
        segs = "{} / {} / {}".format(row["short_d_segments"], row["d_segments"],
                                     row["possessions"]) if "short_d_segments" in row else "–"
        w.append("| {} | {:g} / {:g} | {:+.4f} | {} | {} | {} |".format(
            row["rank"], row["pair"][0], row["pair"][1], row["score"], " | ".join(cells),
            verdict, segs))
    ch = rec["chosen"]
    w.append("\nChosen pair: {}.\n".format(
        "{} (rank {}, Score {:+.4f})".format(tuple(ch["pair"]), ch["rank"], ch["score"])
        if ch else "none" + (" (no pair with Score > 0 is inside the four 8e bounds: c23 ships)"
                              if rec["applies"] else "")))
    return "\n".join(w) + "\n"


def cmd_pair12a(args):
    """The Section 12a walk on the label-free match: rank the pairs by Score (views.json), decode
    the stored probabilities of the final ensemble at each pair in that order (structure on),
    compute the four 8e statistics, and write the table and the chosen pair (pair12a.json and
    pair12a.md in the folder of the label-free step). No engine run, no network run."""
    from engine import data, learned
    from tools.learned_spike import bar_report as br
    import score_game as sg
    ctx = make_ctx(args)
    match = Match(ctx, args.as_match)
    folder = match.folder
    bar_path = os.path.join(folder, "bar.json")
    if not os.path.isfile(bar_path):
        raise SystemExit("{} is missing: the walk runs after the label-free step (`bar`)".format(
            bar_path))
    bar = read_json(bar_path)
    report = read_json(os.path.join(folder, "bar_report.json"))
    views = read_json(os.path.join(ctx.work, "views.json"))
    costs = bar["final_costs"]
    if costs.get("section12a"):
        raise SystemExit("{} is a Section 12a run: the walk reads the step at the rule's pair"
                         .format(bar_path))
    rule = args.costs_rule
    ranking = score_ranking(views, rule)
    grid = [tuple(float(x) for x in p) for p in views["grid"]]
    first = grid[ranking[0][0]]
    rule_pair = tuple(float(x) for x in costs["M"])
    if first != rule_pair:
        raise SystemExit("the first pair of the ranking {} is not the pair of the label-free "
                         "step {} (costs rule {} vs {})".format(first, rule_pair, rule,
                                                                costs.get("rule")))
    game = match.game
    n = data.frame_count(game)
    blocks = data.graded_blocks(game)
    runs = br.graded_runs(game)
    structure = bool(costs.get("structure"))
    forced = learned.pre_kickoff_mask(n, blocks, data.kickoff_frames(game)) if structure else None
    seeds = [os.path.join(ctx.work, "probs", "final", game, "seed_{}.npz".format(i))
             for i in SEEDS]
    rec_probs = read_json(os.path.join(ctx.work, "probs", "final", game, "record.json"))
    if rec_probs.get("weights_md5") != bar["final_npz_md5"]:
        raise SystemExit("the stored probabilities of {} come from other weights".format(game))
    probs = ensemble_from_dumps(seeds, n)
    ext_stats = report["external_stats"]["M"]

    def decode(pair):
        return learned.decode_model(probs, blocks, float(pair[0]), float(pair[1]), forced)

    # The decode must reproduce the engine output of the label-free step at the rule's pair.
    csv, _ = match.run("M", "M", os.path.abspath(bar["final_npz"]), costs, must_exist=True)
    if md5_file(csv) != bar["md5"]["M"]:
        raise SystemExit("{}: md5 differs from bar.json".format(csv))
    engine_codes = sg.read_codes(csv, n)
    rule_codes = decode(rule_pair)
    diff = int(np.sum(rule_codes != engine_codes))
    if diff:
        raise SystemExit("the decode of the stored probabilities at {} differs from engine M on "
                         "{} frames: the walk would not decode what the engine writes".format(
                             rule_pair, diff))
    rule_stats, rule_inside, rule_extra = stats_8e(rule_codes, runs, ext_stats)
    bar_env = {k: bar["envelope_provided"].get(k) for k in E_KEYS}
    if {k: (v or {}).get("bar") for k, v in rule_stats.items()} != {
            k: (v or {}).get("bar") for k, v in bar_env.items()}:
        raise SystemExit("the 8e statistics at the rule's pair differ from bar.json: {} vs {}"
                         .format(rule_stats, bar_env))
    bounds = dict(args.rehearse_bound or [])
    if bounds:
        if not match.rehearsal:
            raise SystemExit("--rehearse-bound is a rehearsal option (--as-match)")
        rule_stats, rule_inside, rule_extra = stats_8e(rule_codes, runs, ext_stats, bounds)
        bar = dict(bar, envelope_provided=dict(bar["envelope_provided"], **rebound(
            {k: bar["envelope_provided"].get(k) for k in bounds}, bounds)))
    ignore = tuple(args.rehearse_ignore or ())
    if ignore and not match.rehearsal:
        raise SystemExit("--rehearse-ignore is a rehearsal option (--as-match)")
    applies, reason = applies_12a(bar, read_json(args.explanations) if args.explanations
                                  else None, ignore)
    if ignore:
        reason += " (rehearsal: {} ignored)".format(", ".join(ignore))

    def decode_stats(pair):
        log("12a: decode {}".format(pair))
        codes = decode(pair)
        rows8, inside, extra = stats_8e(codes, runs, ext_stats, bounds)
        return rows8, inside, dict(extra, labels_md5=ev.codes_md5(codes))
    # Only when 12a applies is the competition match decoded at other pairs.
    rows, chosen = walk_12a(ranking, grid, decode_stats if applies else None,
                            full=args.full_table)
    if not applies:
        rows[0].update(rule_extra, decoded=True, stats=rule_stats, inside=rule_inside,
                       note="the label-free step's own pair")
    rec = {"game": game, "rehearsal": match.rehearsal or bool(bar.get("rehearsal")),
           "rule": costs.get("rule"), "ranking_rule": rule, "rule_pair": list(rule_pair),
           "rule_pair_inside_8e": rule_inside, "applies": applies, "applies_reason": reason,
           "structure": structure,
           "reproduces_engine": "yes: 0 frames differ; label md5 {} (decode) = {} (engine CSV "
                                "labels); CSV md5 {} = bar.json md5.M".format(
                                    ev.codes_md5(rule_codes), ev.codes_md5(engine_codes),
                                    bar["md5"]["M"]),
           "probabilities": "W/probs/final/{}/seed_0..3.npz, weights md5 {}".format(
               game, bar["final_npz_md5"]),
           "final_npz_md5": bar["final_npz_md5"], "rows": rows,
           "rehearsal_bounds": {k: list(v) for k, v in bounds.items()},
           "rehearsal_ignored": list(ignore),
           "chosen": chosen if applies else None,
           "chosen_if_applied": chosen, "full_table": bool(args.full_table),
           "engine_source_hash": ev.engine_hash(), "tool_commit": git_out("rev-parse", "HEAD")}
    write_json(os.path.join(folder, "pair12a.json"), rec)
    md = pair12a_markdown(rec)
    with open(os.path.join(folder, "pair12a.md"), "w") as handle:
        handle.write(md)
    print(md)
    if applies and chosen:
        print("Section 12a: the pair of the label-free step is {} ({}). Rerun the label-free "
              "criteria with --pair {:g},{:g} (r3_score.sh: PAIR_12A={:g},{:g}).".format(
                  tuple(chosen["pair"]), reason, chosen["pair"][0], chosen["pair"][1],
                  chosen["pair"][0], chosen["pair"][1]))
    elif applies:
        print("Section 12a: no pair with Score > 0 is inside the four 8e bounds: c23 ships.")
    else:
        print("Section 12a: " + reason)
    return 0


def rule_attempt(bar):
    """The label-free step at the rule's pair, kept in the result after a Section 12a rerun."""
    b8 = bounds8(bar)
    return {"pair": list(bar["final_costs"]["M"]), "rule": bar["final_costs"].get("rule"),
            "md5": bar["md5"], "criterion6": crit6(bar["systems"]["M"]),
            "criterion7": crit7(b8), "bounds8e": b8["e"],
            "systems": {k: bar["systems"][k] for k in BAR_SYSTEMS if k in bar["systems"]}}


def rule_attempt_markdown(bar):
    r = rule_attempt(bar)
    w = ["## The label-free step at the rule's pair {} (failed, kept; R3_DEVIATION_COSTS.md "
         "12a)\n".format(tuple(r["pair"]))]
    w.append("Rule: {}. Criterion 6: {}. Criterion 7: {} (parts {}).\n".format(
        r["rule"], ev.fmt(r["criterion6"]["pass"]), ev.fmt(r["criterion7"]["pass"]),
        json.dumps(r["criterion7"]["numbers"], sort_keys=True)))
    w.append("| 8e statistic | value | external min | external max | inside |\n"
             "|---|---|---|---|---|")
    for k, row in sorted(r["bounds8e"]["checked"].items()):
        if row:
            w.append("| {} | {} | {} | {} | {} |".format(k, row["bar"], row["ext_min"],
                                                       row["ext_max"], row["inside"]))
    w.append("\n| system | md5 | possessions | D share | D segments |\n|---|---|---|---|---|")
    for k, row in r["systems"].items():
        w.append("| {} | `{}` | {} | {} | {} |".format(k, r["md5"].get(k), row["possessions"],
                                                     fmt_num(row["d_share"]), row["d_segments"]))
    return "\n".join(w) + "\n"


def check_pair12a(match, pair):
    """Refuse a --pair that is not the chosen pair of pair12a.json (the walk on this match)."""
    path = os.path.join(match.rule_folder, "pair12a.json")
    if not os.path.isfile(path):
        raise SystemExit("--pair needs {} (run `pair12a` first)".format(path))
    rec = read_json(path)
    chosen = rec.get("chosen")
    if not rec.get("applies") or not chosen:
        raise SystemExit("{}: Section 12a does not choose a pair ({})".format(
            path, rec.get("applies_reason")))
    if [float(x) for x in chosen["pair"]] != [float(x) for x in pair]:
        raise SystemExit("--pair {} differs from the pair chosen by the walk {}".format(
            tuple(pair), tuple(chosen["pair"])))


def parse_pair(text):
    try:
        a, b = (float(x) for x in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("--pair takes H-A,alive-dead, e.g. 32,16")
    return (a, b)


# -- gate and decide ------------------------------------------------------------------------------

def label_free_folder(work):
    """The folder of the label-free step: bar/ or a rehearsal bar_as_<GAME>/ (the newest); its
    Section 12a rerun (<folder>_12a/) when that holds a bar.json."""
    real = os.path.join(work, "bar")
    if os.path.isfile(os.path.join(real, "bar.json")):
        folder = real
    else:
        cands = sorted(d for d in os.listdir(work) if d.startswith("bar_as_")
                       and not d.endswith("_12a")) if os.path.isdir(work) else []
        folder = os.path.join(work, cands[-1]) if cands else real
    if os.path.isfile(os.path.join(folder + "_12a", "bar.json")):
        return folder + "_12a"
    return folder


def criteria_so_far(work, explanations=None):
    """{"1".."10": criterion or None} from the files present in W."""
    out = {str(i): None for i in range(1, 11)}
    sc_path = os.path.join(work, "scores.json")
    if os.path.isfile(sc_path):
        sm = read_json(sc_path)["summaries"]
        dev = read_json(sc_path).get("summaries_metrica_dfl", {})
        if "M|nested|ctl2|0" in sm and "M|final|ctl2|0" in sm:
            out["1"] = crit1(sm["M|nested|ctl2|0"], sm["M|final|ctl2|0"])
            out["2"] = crit2(sm["M|nested|ctl2|0"])
        if "M|nested|provided|0" in sm:
            out["3"] = crit3(sm["M|nested|provided|0"], dev.get("M|nested|provided|0"))
    st_path = os.path.join(work, "stress.json")
    if os.path.isfile(st_path):
        st = read_json(st_path)
        rec_p = {"stressed_delta": [st["summaries"]["provided"]["stressed"][k]
                                    for k in ("delta", "low", "high")],
                 "unstressed_delta": st["summaries"]["provided"]["unstressed"]["delta"],
                 "drop_x": st["drops"]["provided"]["M"]["balanced"],
                 "drop_c23": st["drops"]["provided"][BASE]["balanced"]}
        out["4"] = crit4(st["summaries"]["ctl2"]["stressed"], st["summaries"]["ctl2"]["unstressed"],
                         st["drops"]["ctl2"]["M"]["balanced"],
                         st["drops"]["ctl2"][BASE]["balanced"], rec_p)
        out["5"] = crit5(st["d_hidden"][ev.VIEW_U30]["M"]["per_provider"],
                         st["d_hidden"][ev.VIEW_U30][BASE]["per_provider"])
    folder = label_free_folder(work)
    bar_path = os.path.join(folder, "bar.json")
    if os.path.isfile(bar_path):
        bar = read_json(bar_path)
        out["6"] = crit6(bar["systems"]["M"])
        b8 = bounds8(bar, explanations, variant_of(work))
        out["7"] = dict(crit7(b8), bounds8=b8)
    v_path = os.path.join(folder, "v23.json")
    if os.path.isfile(v_path):
        v = read_json(v_path)
        out["9"], out["10"] = crit9(v), crit10(v)
    r_path = os.path.join(folder, "repro.json")
    if os.path.isfile(r_path):
        rep = read_json(r_path)
        if rep.get("pass") is not None:
            out["8"] = crit8(rep)
    x_path = os.path.join(folder, "r4bounds.json")
    out["7x"] = None
    if os.path.isfile(x_path):
        xb = read_json(x_path)
        out["7x"] = {"pass": bool(xb["pass"]), "numbers": xb["bounds"], "blocking": True}
    return out


def cmd_gate(args):
    expl = read_json(args.explanations) if args.explanations else None
    crits = criteria_so_far(os.path.abspath(args.work), expl)
    for i in args.criteria:
        c = crits.get(i)
        if c is None:
            print("GATE: criterion {} was not computed: stop.".format(i))
            return 4
        if not c["pass"]:
            print("GATE: criterion {} FAILS: {}. Under R3 Section 4 M does not qualify and c23 "
                  "ships; the scoring stops here.".format(i, json.dumps(
                      c.get("numbers"), sort_keys=True, default=float)[:600]))
            return 3
        print("gate: criterion {} passes".format(i))
    return 0


def fmt_num(v, spec="{:.4f}"):
    return ev.fmt(v, spec)


def result_markdown(ctx_work, crits, choice, reason, inputs, rehearsal, stop, variant="R3B"):
    w = []
    name = result_name(variant)
    title = "# {}: round {}, variant {}, against the frozen protocol".format(
        name, ROUND_OF.get(variant, "3"),
        VARIANT_TITLES[variant])
    if rehearsal:
        title = ("# REHEARSAL ({}, not a result): {}".format(
            "short-training models" if variant in R4_LIKE else "seed 0 fold models only", name))
    w.append(title + "\n")
    w.append("Written by `tools/learned_spike/r3_eval.py decide` from W = `{}`. Protocols: "
             "`LEARNED_STAGE_PROTOCOL.md`, `LEARNED_STAGE_PROTOCOL_R2.md`, "
             "`LEARNED_STAGE_PROTOCOL_R3.md` (Sections 0 to 10, frozen; R3b in Section 9{}). "
             "Nothing here was tuned after a score was seen.\n".format(
                 ctx_work, "; R3c in R3_DEVIATION_AUG.md Section 13" if variant == "R3C" else ""))
    if variant in PF_VARIANTS:
        w.append("Variant {}: the candidate is the network + the per-frame member (Config."
                 "LEARNED_PF on), as engine.learned decodes it. For the label-free stability "
                 "bounds 8a and 8b, a **seed member** is the per-frame member's frame "
                 "probabilities averaged (at the member's weight) with ONE network seed (that "
                 "seed's mean over its four symmetric copies), and a **copy member** is the "
                 "per-frame member averaged with ONE symmetric copy of the network (that copy's "
                 "mean over the four seeds); the network's swapped copies come from the swapped "
                 "tracking (engine.learned.predict_ensemble_paired).\n".format(variant))
    if variant in R6_LIKE:
        g7 = [crits.get(k) for k in R6_RECORDED]
        state = ("not computed" if any(c is None for c in g7)
                 else "pass" if all(c["pass"] for c in g7) else "fail")
        w.append("Round {}, variant {} (feature version {}): **G7: {} (recorded; the "
                 "owner decides)**. G7's limits 8a and 8b are the lower of 1.5 x this model's "
                 "SkillCorner fold mean and round 4's limits (0.0717, 0.0289); criteria 6, 7 and "
                 "7x are recorded and do not set the verdict below, which G6 (criterion 5) and "
                 "G8 (criterion 8) decide.\n".format(ROUND_OF[variant], VARIANT_TITLES[variant],
                                                     FEATURE_VERSIONS[variant], state))
    if variant == "R5":
        w.append("Round 5 (`LEARNED_STAGE_PROTOCOL_R5.md`): this is a **second attempt at G7**. "
                 "Round 4's two verdicts stand beside this one: under round 4's protocol c23 "
                 "ships, and round 4's candidate failed G7 (seeds 0.0754 against 0.0717, copies "
                 "0.0303 against 0.0289). G7's limits 8a and 8b here are the lower of 1.5 x the "
                 "round 5 model's SkillCorner fold mean and round 4's limits.\n")
    if rehearsal and variant in R4_LIKE:
        w.append("**This file is a rehearsal of the tooling.** The models are short-training "
                 "rehearsal models, the pair and prior bonus were given on the command line, "
                 "and the label-free step ran on a development match in place of the "
                 "competition match. No number here is a result of round {}.\n".format(
                     ROUND_OF.get(variant, "4")))
    elif rehearsal:
        w.append("**This file is a rehearsal of the tooling.** The 'ensembles' are the seed 0 "
                 "fold models copied four times (final = the seed 0 models of folds 0 to 3), "
                 "the costs are those of the interim check, and the label-free step ran on a "
                 "development match in place of the competition match. No number here is a "
                 "result of round 3.\n")
    w.append("**Disclosure** (R3 Sections 0, 9, 10{}): {}\n".format(
        ", 13" if variant == "R3C" else "", disclosure(variant)))
    w.append("## Verdict\n")
    w.append("**{}**. Reason: {}.\n".format(choice or "no verdict yet", reason))
    if stop:
        w.append("Stopped: {}.\n".format(stop))
    w.append("## Criteria (R1 Section 5 and R2 Section 3 with c23; R3 Section 4)\n")
    w.append("| # | blocking | M | numbers |\n|---|---|---|---|")
    blocking = blocking_of(variant)
    keys = ["4", "5", "6", "7", "7x", "8"] if variant in R4_LIKE else [
        str(i) for i in range(1, 11)]
    for i in keys:
        c = crits.get(str(i))
        if c is None:
            w.append("| {} | {} | not computed | |".format(i, "yes" if str(i) in blocking
                                                           else "no (recorded)"))
            continue
        if variant in R4_LIKE and str(i) not in blocking:
            c = dict(c, blocking=False)
        nums = {k: v for k, v in c.items() if k not in ("pass", "blocking", "bounds8")}
        w.append("| {} | {} | {} | `{}` |".format(
            i, "yes" if c.get("blocking") else "no (recorded)", ev.fmt(c["pass"]),
            json.dumps(nums, sort_keys=True, default=float)))
    w.append("")
    sc = inputs.get("scores")
    if sc:
        w.append("## Provider-balanced deltas against c23 (raw view)\n")
        w.append("Bootstrap: provider-stratified over matches, 1,000 draws, seed 0, 5–95 %. "
                 "Row key: candidate / costs (nested = rule B per match, SkillCorner with the "
                 "final pair; final = the pair chosen on all eight) / reference / shift.\n")
        for name, rows in (("All 28 matches (Metrica, DFL, SkillCorner)", sc["summaries"]),
                           ("Metrica and DFL only (recorded)", sc.get("summaries_metrica_dfl",
                                                                      {})),
                           ("Matched input view, Metrica and DFL (recorded)",
                            sc.get("summaries_matched", {}))):
            if not rows:
                continue
            w.append("### {}\n".format(name))
            w.append("| row | X mean | c23 mean | Δ | 5 % | 95 % | Metrica Δ (wins/n) | DFL Δ "
                     "(wins/n) | SkillCorner Δ (wins/n) |\n|---|---|---|---|---|---|---|---|---|")
            for key in sorted(rows):
                r = rows[key]
                cells = []
                for p in ev.PROVIDERS:
                    q = r["per_provider"].get(p)
                    cells.append("{} ({}/{})".format(fmt_num(q["delta"], "{:+.4f}"), q["wins"],
                                                     q["n"]) if q else "–")
                w.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                    key.replace("|", " / "), fmt_num(r["x_mean"]), fmt_num(r["c23_mean"]),
                    fmt_num(r["delta"], "{:+.4f}"), fmt_num(r["low"], "{:+.4f}"),
                    fmt_num(r["high"], "{:+.4f}"), " | ".join(cells)))
            w.append("")
        w.append("## Per-match spellF1 (raw view; matched view for the open matches)\n")
        labels = [l for l in RAW_SYSTEMS + H_SYSTEMS
                  if any(l in sc["scores"][g] for g in sc["scores"])]
        w.append("| match | ref | costs M (nested) | {} | matched c23 | matched M |\n|{}".format(
            " | ".join(labels), "---|" * (len(labels) + 5)))
        games = sorted(sc["scores"], key=lambda g: (ev.PROVIDERS.index(folds.provider(g)), g))
        for g in games:
            for ref in ("provided", "ctl2"):
                row = sc["scores"][g]
                mrow = (sc.get("matched_scores") or {}).get(g, {})
                cells = [fmt_num(row[l][ref]["0"]["spell_f1"]) if l in row else "–"
                         for l in labels]
                mc = [fmt_num(mrow[l][ref]["0"]["spell_f1"]) if l in mrow else "–"
                      for l in MATCHED_SYSTEMS]
                w.append("| {} | {} | {} | {} | {} |".format(
                    g, ref, sc["costs"]["M_nested"].get(g, sc["costs"]["M_final"]),
                    " | ".join(cells), " | ".join(mc)))
        w.append("")
    st = inputs.get("stress")
    if st:
        w.append("## Stress view (criterion 4) and missing-ball D guard (criterion 5)\n")
        w.append("| reference | system | drop balanced | Metrica | DFL | SkillCorner |\n"
                 "|---|---|---|---|---|---|")
        for ref in ("ctl2", "provided"):
            for s in STRESS_SYSTEMS:
                d = st["drops"][ref][s]
                w.append("| {} | {} | {} | {} |".format(ref, s, fmt_num(d["balanced"], "{:+.4f}"),
                                                       " | ".join(fmt_num(d["per_provider"].get(p),
                                                                          "{:+.4f}")
                                                                  for p in ev.PROVIDERS)))
        w.append("\n| view | system | Metrica | DFL | SkillCorner |\n|---|---|---|---|---|")
        for view in (ev.VIEW_U30, ev.VIEW_STRESS):
            for s in STRESS_SYSTEMS:
                pp = st["d_hidden"][view][s]["per_provider"]
                w.append("| {} | {} | {} |".format(view, s, " | ".join(fmt_num(pp.get(p))
                                                                       for p in ev.PROVIDERS)))
        w.append("")
    bar = inputs.get("bar")
    if bar:
        w.append("## Label-free step on {} (criterion 6, Section 8{}{})\n".format(
            bar["game"], ", REHEARSAL on a development match" if bar.get("rehearsal") else "",
            ", at the Section 12a pair" if bar.get("section12a") else ""))
        if bar.get("section12a"):
            w.append("**Section 12a** (R3_DEVIATION_COSTS.md 12a): the rule's pair {} failed 8e; "
                     "criteria 6 to 10 are computed at the pair {} only (walk below).\n".format(
                         tuple(bar["final_costs"].get("rule_pair") or ()),
                         tuple(bar["final_costs"]["M"])))
        w.append("Final weights md5 `{}`, final costs `{}`.\n".format(
            bar["final_npz_md5"], json.dumps(bar["final_costs"])))
        w.append("| system | md5 | possessions | D share | D segments | graded X | possession "
                 "median s | D share, ball missing |\n|---|---|---|---|---|---|---|---|")
        for s in BAR_SYSTEMS:
            r = bar["systems"][s]
            w.append("| {} | `{}` | {} | {} | {} | {} | {} | {} |".format(
                s, bar["md5"][s], r["possessions"], fmt_num(r["d_share"]), r["d_segments"],
                r["graded_x"], r["possession_s"].get("p50"), fmt_num(r.get("d_share_ball_missing"))))
        w.append("\nSeed and copy disagreement: `{}`\n".format(json.dumps(bar["disagreement"],
                                                                         sort_keys=True)))
        w.append("8c band: the 1–99 % band of the graded bins of the {} {} training matches "
                 "(no PFF).\n".format(len(bar["features"]["train_games"]),
                                      VARIANT_TITLES[variant]))
    rule_bar = inputs.get("bar_rule")
    if rule_bar:
        w.append(rule_attempt_markdown(rule_bar))
    if inputs.get("pair12a"):
        w.append(pair12a_markdown(inputs["pair12a"]))
    v23 = inputs.get("v23")
    if v23:
        w.append("## V2 and V3 (criteria 9 and 10)\n")
        w.append("| system | A (alive raw, in K) | flipped to D | flip | V3 |\n|---|---|---|---|---|")
        for s in (BASE, "M"):
            r = v23["v2"]["systems"][s]
            w.append("| {} | {} | {} | {} | {} |".format(s, r["alive_raw"], r["flipped_to_d"],
                                                       fmt_num(r["flip"]), fmt_num(v23["v3"][s])))
        w.append("")
    costs = inputs.get("costs")
    if costs and costs.get("fixed"):
        w.append("## Costs (round 4, fixed)\n")
        w.append("M {} for every match, kickoff rule {}, prior bonus {:g}; {}.\n".format(
            costs["M_final"], "on" if costs.get("structure") else "off",
            costs.get("prior_bonus", 0.0), costs.get("source")))
        costs = None
    if costs:
        w.append("## Costs (rule B)\n")
        w.append("M final {}, H final {}; per development match (nested): {}. Recomputed from "
                 "the out-of-fold probabilities: {}.\n".format(
                     costs["M_final"], costs["H_final"], json.dumps(costs["M_nested"]),
                     "identical" if not costs["recompute_mismatch"] else
                     "MISMATCH " + "; ".join(costs["recompute_mismatch"])))
    pre = inputs.get("preflight")
    if pre:
        w.append("## Preflight\n")
        w.append("Run `{}` (variant {}{}); problems: {}; model commits {}; interim kill check {}; "
                 "identity {}.\n".format(pre["run"], VARIANT_TITLES[pre.get("variant", "R3B")],
                                         ", aug manifest md5 {}".format(pre["aug_manifest_md5"])
                                         if pre.get("aug_manifest_md5") else "",
                                         pre["problems"] or "none", pre["commits"],
                                         pre["kill_check_interim"],
                                         (pre.get("identity") or {}).get("pass", "not run")))
    agr = inputs.get("agreement")
    if agr and not inputs.get("bar"):
        w.append("## Seed and copy agreement on SkillCorner (Section 8a and 8b baselines)\n")
        w.append("Seed: fold-ensemble mean {:.4f}, limit 8a {:.4f}, final ensemble (in sample) "
                 "{:.4f}. Copy: fold mean {:.4f}, limit 8b {:.4f}, final {:.4f}. The bounds are "
                 "read on the competition match in the label-free step.\n".format(
                     agr["seed"]["sc_mean_fold"], agr["seed"]["limit_8a"],
                     agr["seed"]["sc_mean_final"], agr["copy"]["sc_mean_fold"],
                     agr["copy"]["limit_8b"], agr["copy"]["sc_mean_final"]))
    views = inputs.get("views")
    if views:
        from tools.learned_spike import r3_views
        w.append(r3_views.markdown(views))
    w.append("## Readings this tool applies where the protocol is silent or ambiguous\n")
    for item in READINGS:
        w.append("- " + item)
    w.append("")
    return "\n".join(w) + "\n"


READINGS = (
    "Criterion 3 pools Metrica, DFL and SkillCorner (R1 Section 4's provider-balanced mean, "
    "verbatim); the Metrica-and-DFL-only interval is recorded next to it.",
    "A SkillCorner match has no nested pair (rule B uses Metrica and DFL only): it is decoded "
    "with the final pair, which no SkillCorner reference chose.",
    "Criterion 4 is computed on the 2 s references, as its text says (R3 Section 4 names only "
    "criteria 1 and 2 as moved to 'recorded'); the same numbers on the references as provided "
    "are recorded.",
    "Criterion 2's per-provider test is on the 2 s references and therefore recorded only; no "
    "per-provider test on the references as provided blocks.",
    "The matched input view (R3 Section 4) is scored and recorded for Metrica and DFL; no "
    "criterion is defined on it.",
    "H is not eligible (R3 Section 4); its raw development scores are recorded, with its H-A "
    "cost chosen by the nested rule on decoders.GRID_HA over the c23 labels.",
    "The baseline c23 is the engine of this branch with LEARNED_MODE off; its identity with tag "
    "`submission` is checked on 16 external outputs and on BAR-ATH (7844cad8...).",
    "Criterion 8 part b (fresh clone) needs a commit whose config defaults select M and a pip "
    "install from requirements.txt; until it has run, criterion 8 is not computed.",
)


def cmd_decide(args):
    work = os.path.abspath(args.work)
    out_dir = os.path.abspath(args.out_dir)
    if os.path.commonpath([out_dir, ROOT]) == ROOT:
        raise SystemExit("--out-dir must be outside the repository")
    expl = read_json(args.explanations) if args.explanations else {}
    crits = criteria_so_far(work, expl)
    pre = read_json(os.path.join(work, "preflight.json")) if os.path.isfile(
        os.path.join(work, "preflight.json")) else {}
    kill = (pre.get("kill_check_interim") or {}).get("passed", False)
    variant = pre.get("variant") or getattr(args, "variant", None) or "R3B"
    choice, reason = verdict(crits, kill, args.stop,
                             blocking_of(variant))
    if getattr(args, "variant", None) and args.variant != variant:
        raise SystemExit("--variant {} but preflight.json of {} says {}".format(
            args.variant, work, variant))
    result = result_name(variant)
    folder = label_free_folder(work)
    rule_folder = folder[:-len("_12a")] if folder.endswith("_12a") else folder
    inputs = {}
    for name, path in (("scores", os.path.join(work, "scores.json")),
                       ("stress", os.path.join(work, "stress.json")),
                       ("bar", os.path.join(folder, "bar.json")),
                       ("v23", os.path.join(folder, "v23.json")),
                       ("repro", os.path.join(folder, "repro.json")),
                       ("costs", os.path.join(work, "costs_r3.json")),
                       ("preflight", os.path.join(work, "preflight.json")),
                       ("views", os.path.join(work, "views.json")),
                       ("agreement", os.path.join(work, "agreement.json")),
                       ("pair12a", os.path.join(rule_folder, "pair12a.json")),
                       ("bar_rule", os.path.join(rule_folder, "bar.json")
                        if rule_folder != folder else "")):
        if os.path.isfile(path):
            inputs[name] = read_json(path)
    rehearsal = bool(pre.get("rehearsal")) or any(
        (inputs.get(k) or {}).get("rehearsal") for k in ("bar", "v23", "repro"))
    os.makedirs(out_dir, exist_ok=True)
    md = os.path.join(out_dir, result + ".md")
    with open(md, "w") as handle:
        handle.write(result_markdown(work, crits, choice, reason, inputs, rehearsal, args.stop,
                                     variant))
    write_json(os.path.join(out_dir, result + ".json"), {
        "variant": variant, "rehearsal": rehearsal, "choice": choice, "reason": reason,
        "stop": args.stop,
        "criteria": crits, "kill_check_interim": pre.get("kill_check_interim"),
        "final_npz_md5": (inputs.get("bar") or {}).get("final_npz_md5"),
        "final_costs": (inputs.get("bar") or {}).get("final_costs"),
        "engine_source_hash": (inputs.get("scores") or {}).get("engine_source_hash"),
        "rule_pair_attempt": rule_attempt(inputs["bar_rule"]) if inputs.get("bar_rule")
        else None,
        "readings": list(READINGS), "disclosure": disclosure(variant),
        "tool_commit": git_out("rev-parse", "HEAD")})
    print("{}{}: {}".format("REHEARSAL " if rehearsal else "", choice, reason))
    print("wrote {}".format(md))
    return 0


# -- CLI ------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--work", required=True, help="work folder outside the repository")
    common.add_argument("--run", required=True, help="r3_launch.sh output folder")
    runs = argparse.ArgumentParser(add_help=False)
    runs.add_argument("--engine-cache", help="engine.data cache of the open references")
    runs.add_argument("--sc-engine-cache", help="engine.data cache of SkillCorner")
    runs.add_argument("--ref-cache", help="folder with refs/METRICA-*_ref.csv")
    runs.add_argument("--refs-root", default=ev.DEFAULT_REFS_ROOT)
    runs.add_argument("--sample-root", default=ev.DEFAULT_SAMPLE_ROOT)
    runs.add_argument("--mtah-root", default=ev.DEFAULT_MTAH_ROOT)
    p = sub.add_parser("preflight", parents=[common])
    p.add_argument("--rehearsal", action="store_true", help="record problems, do not stop")
    p.add_argument("--identity", action="store_true", help="also the switch-off identity")
    p.add_argument("--variant", choices=VARIANTS, default="R3B",
                   help="R3B (default) or R3C = R3B + the aug switch (R3_DEVIATION_AUG.md 13)")
    p = sub.add_parser("costs", parents=[common])
    p.add_argument("--cache", required=True, help="build.py cache (labels/<game>.npz)")
    p.add_argument("--c23-dev", required=True, help="c23 CSVs of the 8 development matches")
    p = sub.add_parser("run", parents=[common, runs])
    p.add_argument("--systems", nargs="+", default=list(RAW_SYSTEMS),
                   choices=list(RAW_SYSTEMS + H_SYSTEMS))
    p.add_argument("--no-matched", action="store_true")
    sub.add_parser("score", parents=[common, runs])
    sub.add_parser("stress", parents=[common, runs])
    for name in ("bar", "v23", "repro"):
        p = sub.add_parser(name, parents=[common, runs])
        p.add_argument("--final-npz", required=True)
        p.add_argument("--as-match", help="rehearsal: a development match in place of BAR-ATH")
        p.add_argument("--costs-rule", choices=("a", "b", "c", "fixed"), default="b",
                       help="costs of the label-free step: b = R3_DEVIATION_COSTS.md (default), "
                            "a = rule B as frozen; the frozen pair is recorded either way")
        p.add_argument("--no-structure", action="store_true",
                       help="M without LEARNED_STRUCTURE (default: on, R3_DEVIATION_STRUCTURE.md)")
        if name == "bar":
            p.add_argument("--feature-cache", required=True, help="build.py cache (8c band)")
        p.add_argument("--pair", type=parse_pair,
                       help="H-A,alive-dead: the Section 12a pair chosen by `pair12a` "
                            "(R3_DEVIATION_COSTS.md 12a); replaces the --costs-rule pair, "
                            "runs in <label-free folder>_12a/")
        if name == "repro":
            p.add_argument("--repro-check", help="JSON of repro_check.py run")
            p.add_argument("--fresh-clone-log", help="log of tools/repro_fresh_clone.sh")
            p.add_argument("--tests-log", help="pytest log of the learned-stage tests")
    p = sub.add_parser("pair12a", parents=[common, runs])
    p.add_argument("--as-match", help="rehearsal: a development match in place of BAR-ATH")
    p.add_argument("--costs-rule", choices=("b", "c"), default="b",
                   help="the ranking rule (b, R3_DEVIATION_COSTS.md 12)")
    p.add_argument("--explanations", help="Section 8c JSON (recorded only)")
    p.add_argument("--full-table", action="store_true",
                   help="decode every pair with Score > 0, not only up to the chosen one")
    p.add_argument("--rehearse-bound", type=parse_bound, action="append",
                   help="rehearsal only (--as-match): KEY=LO,HI replaces one 8e bound, so that "
                        "the walk can be rehearsed on a match where the rule's pair passes")
    p.add_argument("--rehearse-ignore", action="append", choices=("criterion 6", "8a", "8b", "8f"),
                   help="rehearsal only: a part whose failure does not stop the walk")
    p = sub.add_parser("costs-fixed", parents=[common],
                       help="R4: the pair and prior bonus of Section 4 from the command line")
    p.add_argument("--pair", type=parse_pair, required=True)
    p.add_argument("--prior-bonus", type=float, required=True)
    p.add_argument("--pf-weights", default=None, help="R6E: the per-frame member's weights")
    p.add_argument("--pf-weight", type=float, default=0.5, help="R6E: its weight in the blend")
    p.add_argument("--no-structure", action="store_true")
    p = sub.add_parser("r4bounds", parents=[common, runs],
                       help="R4: the four extra reject-only bounds of G7 (criterion 7x)")
    p.add_argument("--as-match", help="rehearsal: the development match of the label-free step")
    p.add_argument("--oof-ratio", required=True, help="JSON of r4_oof_ratio.py at the same "
                                                      "pair and bonus")
    p = sub.add_parser("agreement", parents=[common, runs])
    p.add_argument("--final-npz", required=True)
    p = sub.add_parser("gate")
    p.add_argument("--work", required=True)
    p.add_argument("--criteria", nargs="+", required=True)
    p.add_argument("--explanations")
    p = sub.add_parser("decide")
    p.add_argument("--work", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--explanations", help="Section 8c JSON")
    p.add_argument("--stop", help="the reason the scoring stopped early, recorded")
    p.add_argument("--variant", choices=VARIANTS,
                   help="check against preflight.json (the variant is read from there)")
    args = parser.parse_args(argv)
    for name in ("work", "out_dir"):
        if getattr(args, name, None) and os.path.commonpath(
                [os.path.abspath(getattr(args, name)), ROOT]) == ROOT:
            raise SystemExit("--{} must be outside the repository".format(name.replace("_", "-")))
    if getattr(args, "work", None):
        os.makedirs(args.work, exist_ok=True)
    return {"preflight": cmd_preflight, "costs": cmd_costs, "run": cmd_run, "score": cmd_score,
            "stress": cmd_stress, "agreement": cmd_agreement, "bar": cmd_bar, "v23": cmd_v23, "repro": cmd_repro,
            "pair12a": cmd_pair12a, "costs-fixed": cmd_costs_fixed, "r4bounds": cmd_r4bounds,
            "gate": cmd_gate, "decide": cmd_decide}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
