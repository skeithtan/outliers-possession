"""Score frozen ball masks against external labels (PLAN Phase 10, P10-B).

Create the source profile once with ``profile --out PROFILE.json``. Run each open
match with ``run --game NAME --gt REF.csv --profile PROFILE.json --out-dir DIR``.
Use ``summarize DIR`` to average seeds within matches before paired match intervals.

U places observed gap lengths uniformly. S places them by their source gap class.
S_alt doubles the alive-class weight. These mixtures are assumptions, not truth.
The fraction is additional hidden observations divided by all graded frames.
Native gaps remain missing. Draws can overlap them, but only new missing frames
count toward the target. Rejections and the achieved distribution are reported.

Reference labels can place S masks and score predictions. They never enter the
engine. Each engine run starts from the masked raw ball and recomputes features.
Keep profiles, masks and predictions outside the repository. Reports print only
summary statistics. No sealed match is accepted by this tool.
"""

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT, ROOT / "tools", ROOT / "evaluation"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from tools import gapmask as gm
from tools import score_game as sg
from tools import failure_classes as fc
import possession_metrics as pm

OPEN_GAMES = ("DFL-J03WMX", "DFL-J03WN1", "DFL-J03WOH", "DFL-J03WOY",
              "DFL-J03WPY", "METRICA-S1", "METRICA-S2", "METRICA-S3")
CLASSES = ("inside", "restart", "alive", "onset")
VIEWS = ("U", "S", "S_alt")
SYSTEMS = ("current", "gap_stop_off")
# P9-E stress level, now divided by all graded frames as required by P10-B.
DEFAULT_FRACTION = 0.15
# The alternative doubles live-gap sampling without changing any engine setting.
ALTERNATIVE_ALIVE_WEIGHT = 2.0
# Recovery requires one second correct within five seconds after reappearance.
RECOVERY_DWELL_S = 1.0
RECOVERY_LIMIT_S = 5.0
SCHEMA_VERSION = 1


def digest_json(value):
    """Hash canonical JSON rather than timestamped archive bytes."""
    return hashlib.md5(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False).encode("utf8")).hexdigest()


def array_md5(values):
    """Hash array shape, dtype and values in a deterministic order."""
    values = np.ascontiguousarray(values)
    return hashlib.md5(str(values.shape).encode("ascii") + values.dtype.str.encode("ascii")
                       + values.tobytes()).hexdigest()


def write_json(path, value):
    """Write sorted JSON with no nonfinite numeric values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def outside_repo(path):
    """Reject paths inside the checkout, including symlink aliases."""
    path = Path(path).resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("benchmark artifacts must stay outside the repository")
    return path


def gap_class(codes, start, stop):
    """Classify an interval by its live/dead states, ignoring team changes."""
    values = np.asarray(codes)[start:stop]
    if len(values) == 0 or (values == pm.X).any():
        raise ValueError("a gap must lie inside graded reference frames")
    dead = values == pm.D
    if dead.all():
        return "inside"
    if not dead.any():
        return "alive"
    if (dead[:-1] & ~dead[1:]).any():
        return "restart"
    return "onset"


def length_summary(lengths, fps):
    """Report duration quantiles and the P9-E duration buckets."""
    seconds = np.asarray(lengths, dtype=float) / float(fps)
    return {
        "gaps": len(seconds),
        "median_s": float(np.median(seconds)) if len(seconds) else None,
        "p90_s": float(np.percentile(seconds, 90)) if len(seconds) else None,
        "max_s": float(np.max(seconds)) if len(seconds) else None,
        "buckets": {gm.bucket_name(a, b): int(((seconds >= a) & (seconds < b)).sum())
                    for a, b in gm.BUCKETS_S},
    }


def missing_summary(missing, blocks, fps, codes=None):
    """Summarize missingness inside blocks, including censored edge gaps."""
    graded = gm.graded_mask(len(missing), blocks)
    lengths, counts, frames = [], {k: 0 for k in CLASSES}, {k: 0 for k in CLASSES}
    for _, first, last in blocks:
        starts, stops = gm.runs_of(np.asarray(missing)[first:last + 1])
        for a, b in zip(starts, stops):
            lengths.append(int(b - a))
            if codes is not None:
                kind = gap_class(codes, first + int(a), first + int(b))
                counts[kind] += 1
                frames[kind] += int(b - a)
    return dict(length_summary(lengths, fps), graded_frames=int(graded.sum()),
                missing_frames=int((missing & graded).sum()),
                missing_fraction=float((missing & graded).sum()) / int(graded.sum()),
                class_counts=counts if codes is not None else None,
                class_frames=frames if codes is not None else None)


def build_profile(missing, blocks, fps, baseline, source="BAR-ATH", public=None):
    """Freeze observed durations and engine-based gap classes without private labels."""
    missing = np.asarray(missing, dtype=bool)
    baseline = np.asarray(baseline, dtype=np.int8)
    graded = gm.graded_mask(len(missing), blocks)
    if len(baseline) != len(missing) or (baseline[graded] == pm.X).any():
        raise ValueError("source labels must cover every graded frame")
    gaps, cut = gm.observed_gaps(missing, blocks)
    if not gaps:
        raise ValueError("source profile has no uncensored observed gaps")
    entries = [{"duration_s": (b - a) / float(fps), "class": gap_class(baseline, a, b)}
               for a, b in gaps]
    counts = {k: sum(e["class"] == k for e in entries) for k in CLASSES}
    profile = {"schema": SCHEMA_VERSION, "source": source, "fps": float(fps),
               "source_missing_md5": array_md5(missing),
               "source_labels_md5": array_md5(baseline), "edge_gaps_excluded": cut,
               "entries": entries, "class_counts": counts,
               "mixture": {k: counts[k] / float(len(entries)) for k in CLASSES},
               "observed": missing_summary(missing, blocks, fps, baseline),
               "assumption": "Gap classes use the frozen engine output, not full-match truth.",
               "alternative": "S_alt doubles the alive weight, then normalizes all weights."}
    if public is not None:
        public = np.asarray(public, dtype=np.int8)
        covered = [(i + 1, a, b - 1) for i, (a, b) in enumerate(pm.valid_runs(public))]
        profile["public_context"] = missing_summary(missing, covered, fps, public)
        profile["public_context"]["stoppages"] = sum(
            k == pm.D for _, _, k in pm.segments(public, pm.valid_runs(public)))
    profile["md5"] = digest_json(profile)
    return profile


def check_profile(profile):
    """Validate a saved profile before any draw."""
    content = dict(profile)
    expected = content.pop("md5", None)
    if content.get("schema") != SCHEMA_VERSION or digest_json(content) != expected:
        raise ValueError("profile checksum or schema mismatch")
    if not content.get("entries") or any(
            e["class"] not in CLASSES or not np.isfinite(e["duration_s"])
            or e["duration_s"] <= 0 for e in content["entries"]):
        raise ValueError("invalid profile durations or classes")


def _prefix(mask):
    return np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))


def _window_sum(prefix, length):
    return prefix[length:] - prefix[:-length]


def draw_mask(missing, blocks, fps, reference, profile, game, view, seed,
              fraction=DEFAULT_FRACTION, max_draws=gm.MAX_DRAWS):
    """Draw disjoint intervals and count only newly hidden graded observations."""
    check_profile(profile)
    if view not in VIEWS or not 0.0 < fraction < 1.0:
        raise ValueError("invalid view or fraction")
    missing, reference = np.asarray(missing, bool), np.asarray(reference, np.int8)
    if len(missing) != len(reference):
        raise ValueError("reference length mismatch")
    n = len(missing)
    graded = gm.graded_mask(n, blocks) & (reference != pm.X)
    # Require each interval and margin to lie inside one block, even when blocks touch.
    block_id = np.full(n, -1, dtype=int)
    for i, (_, a, b) in enumerate(blocks):
        block_id[a:b + 1] = i
    available = graded.copy()
    visible = graded & ~missing
    target = int(np.ceil(fraction * int(graded.sum())))
    entries = profile["entries"]
    lengths = np.maximum(1, np.rint([e["duration_s"] * fps for e in entries]).astype(int))
    classes = np.asarray([e["class"] for e in entries])
    weights = np.array([profile["mixture"][k] for k in CLASSES], dtype=float)
    if view == "S_alt":
        weights[CLASSES.index("alive")] *= ALTERNATIVE_ALIVE_WEIGHT
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    prefix_visible = _prefix(visible)
    dead = reference == pm.D
    prefix_dead = _prefix(dead)
    restart = np.zeros(n, bool)
    restart[1:] = dead[:-1] & (reference[1:] < pm.D)
    prefix_restart = _prefix(restart)
    starts, selected, selected_classes = [], [], []
    candidate_cache = {}
    rejected = {k: 0 for k in CLASSES}
    draws = hidden = 0
    while hidden < target and draws < max_draws:
        draws += 1
        if view == "U":
            index = int(rng.integers(len(entries)))
            kind = str(classes[index])
        else:
            kind = CLASSES[int(rng.choice(len(CLASSES), p=weights))]
            pool = np.flatnonzero(classes == kind)
            index = int(pool[rng.integers(len(pool))])
        length = int(lengths[index])
        width = length + 2 * gm.MARGIN
        if width > n:
            rejected[kind] += 1
            continue
        cache_key = (length, kind if view != "U" else "U")
        if cache_key in candidate_cache:
            candidates = candidate_cache[cache_key]
        else:
            candidates = np.flatnonzero(_window_sum(_prefix(available), width) == width) + gm.MARGIN
            if len(candidates):
                candidates = candidates[block_id[candidates - gm.MARGIN]
                                        == block_id[candidates + length + gm.MARGIN - 1]]
            if len(candidates):
                added = prefix_visible[candidates + length] - prefix_visible[candidates]
                candidates = candidates[added > 0]
            if view != "U" and len(candidates):
                n_dead = prefix_dead[candidates + length] - prefix_dead[candidates]
                n_restart = (prefix_restart[candidates + length]
                             - prefix_restart[candidates + 1])
                valid = {"inside": n_dead == length, "alive": n_dead == 0,
                         "restart": n_restart > 0,
                         "onset": (n_dead > 0) & (n_dead < length) & (n_restart == 0)}[kind]
                candidates = candidates[valid]
            candidate_cache[cache_key] = candidates
        if not len(candidates):
            rejected[kind] += 1
            continue
        start = int(candidates[rng.integers(len(candidates))])
        candidate_cache.clear()
        starts.append(start)
        selected.append(length)
        selected_classes.append(gap_class(reference, start, start + length))
        hidden += int(prefix_visible[start + length] - prefix_visible[start])
        available[max(0, start - gm.MARGIN):min(n, start + length + gm.MARGIN)] = False
    mask = {"schema": SCHEMA_VERSION, "game": game, "view": view, "seed": int(seed),
            "fraction": float(fraction), "fraction_denominator": "all graded frames",
            "fps": float(fps), "profile_md5": profile["md5"],
            "native_missing_md5": array_md5(missing), "reference_md5": array_md5(reference),
            "blocks": [[int(v) for v in b] for b in blocks], "max_draws": int(max_draws),
            "starts": starts, "lengths": selected, "classes": selected_classes,
            "target_frames": target, "hidden_frames": hidden, "draws": draws,
            "rejected": int(sum(rejected.values())), "rejected_by_draw_class": rejected,
            "reached": hidden >= target, "mixture": dict(zip(CLASSES, weights.tolist()))}
    mask["md5"] = digest_json(mask)
    return mask


def frozen_mask(path, missing, blocks, fps, reference, profile, game, view, seed,
                fraction=DEFAULT_FRACTION, require_existing=False):
    """Reuse an immutable mask only when all placement inputs match."""
    path = Path(path)
    if not path.exists():
        if require_existing:
            raise ValueError("required frozen mask does not exist: {}".format(path))
        mask = draw_mask(missing, blocks, fps, reference, profile, game, view, seed, fraction)
        write_json(path, mask)
        return mask
    mask = json.loads(path.read_text())
    content = dict(mask)
    checksum = content.pop("md5", None)
    expected = {"schema": SCHEMA_VERSION, "game": game, "view": view, "seed": int(seed),
                "fraction": float(fraction), "fps": float(fps), "profile_md5": profile["md5"],
                "native_missing_md5": array_md5(np.asarray(missing, bool)),
                "reference_md5": array_md5(np.asarray(reference, np.int8)),
                "blocks": [[int(v) for v in b] for b in blocks]}
    if digest_json(content) != checksum or any(mask.get(k) != v for k, v in expected.items()):
        raise ValueError("saved mask does not match its content or placement inputs")
    return mask


@contextmanager
def masked_tactical(tactical, hidden, game):
    """Provide raw masked observations before the engine computes any features."""
    from engine import data
    original = data.load_tactical
    masked = gm.apply_mask(tactical, hidden)

    def load(requested=None, *args, **kwargs):
        return masked if data.resolve_game(requested) == game else original(requested, *args, **kwargs)

    data.load_tactical = load
    try:
        yield
    finally:
        data.load_tactical = original


def predict(tactical, hidden, game, cfg):
    """Run the submission algorithm without passing labels or derived features."""
    from engine import possession
    with masked_tactical(tactical, hidden, game):
        letters = possession.run(cfg, game=game)
    return pm.label_codes(letters)


def _ratio(numerator, denominator):
    return float(numerator) / int(denominator) if denominator else None


def frame_metrics(reference, prediction, where):
    """Count D and X predictions as errors on every reference-live masked frame."""
    live = where & (reference < pm.D)
    both_alive = live & (prediction < pm.D)
    correct = int((live & (reference == prediction)).sum())
    return {"frames": int(where.sum()), "reference_live_frames": int(live.sum()),
            "false_d_frames": int((live & (prediction == pm.D)).sum()),
            "false_d_rate": _ratio((live & (prediction == pm.D)).sum(), live.sum()),
            "ownership_accuracy": _ratio(correct, live.sum()),
            "ownership_accuracy_given_alive": _ratio(correct, both_alive.sum()),
            "predicted_alive_frames": int(both_alive.sum())}


def recovery_metrics(reference, prediction, missing, added, blocks, fps):
    """Measure stable label recovery after effective gaps touched by the mask.

    Search ends at the next gap, reference transition or block edge. Unrecovered
    cases remain censored rather than receiving the latency of another spell.
    """
    dwell = max(1, int(round(RECOVERY_DWELL_S * fps)))
    horizon = max(dwell, int(round(RECOVERY_LIMIT_S * fps)))
    latencies, censored, skipped = [], 0, 0
    for _, first, last in blocks:
        starts, stops = gm.runs_of(missing[first:last + 1])
        for a, b in zip(starts + first, stops + first):
            a, b = int(a), int(b)
            if not added[a:b].any() or b > last or reference[b] == pm.X:
                continue
            end = min(last + 1, b + horizon)
            stop_points = np.flatnonzero(missing[b:end] | (reference[b:end] != reference[b]))
            if len(stop_points):
                end = b + int(stop_points[0])
            correct = prediction[b:end] == reference[b:end]
            if len(correct) < dwell:
                skipped += 1
                continue
            good = np.flatnonzero(_window_sum(_prefix(correct), dwell) == dwell)
            if len(good):
                latencies.append(int(good[0]))
            else:
                censored += 1
    return {"recovered": len(latencies), "censored": censored,
            "insufficient_visible_context": skipped, "dwell_s": RECOVERY_DWELL_S,
            "horizon_s": RECOVERY_LIMIT_S,
            "mean_latency_frames": float(np.mean(latencies)) if latencies else None,
            "median_latency_frames": float(np.median(latencies)) if latencies else None,
            "p90_latency_frames": float(np.percentile(latencies, 90)) if latencies else None}


def spurious_spell_changes(reference, prediction, clean, runs, iou=0.5):
    """Count new and removed false spells after same-team IoU correspondence.

    First identify each output's unmatched spells with the official greedy rule.
    Then match those false spells one to one across outputs at the same IoU.
    A net count can hide a new false spell when an old false spell disappears.
    """
    truth = pm.possession_spells(reference, runs)

    def unmatched(codes):
        spells = pm.possession_spells(pm.fill_abstentions(codes, runs), runs)
        used = set(fc._greedy_pairs(pm.spell_overlaps(truth, spells), iou).values())
        return [spell for index, spell in enumerate(spells) if index not in used]

    old, new = unmatched(clean), unmatched(prediction)
    persisted = len(fc._greedy_pairs(pm.spell_overlaps(old, new), iou))
    return {"new": len(new) - persisted, "removed": len(old) - persisted,
            "persisted": persisted, "delta": len(new) - len(old), "iou": float(iou)}


def score_cell(reference, prediction, clean, mask, missing, blocks, fps, draws=1000):
    """Score one prediction with the official metric and gap-specific diagnostics."""
    reference, prediction, clean = [np.asarray(v, np.int8) for v in (reference, prediction, clean)]
    requested = gm.hidden_mask(len(reference), mask["starts"], mask["lengths"])
    graded = gm.graded_mask(len(reference), blocks) & (reference != pm.X)
    added = requested & ~missing & graded
    result, ref_stats, pred_stats = sg.score_codes(reference, prediction)
    runs = pm.valid_runs(reference)
    filled = pm.fill_abstentions(prediction, runs)
    ref_turns = [t for t in pm.transitions(reference, runs)
                 if t[1] < pm.D and t[2] < pm.D and added[t[0]]]
    pred_turns = [t for t in pm.transitions(filled, runs) if t[1] < pm.D and t[2] < pm.D]
    matched, missed, _ = pm.match_transitions(ref_turns, pred_turns, max(pm.TAUS), typed="pair")
    buckets = []
    for index, (low, high) in enumerate(gm.BUCKETS_S):
        chosen = [(s, length) for s, length in zip(mask["starts"], mask["lengths"])
                  if gm.bucket_of(length, fps) == index]
        bucket = gm.hidden_mask(len(reference), [s for s, _ in chosen], [v for _, v in chosen])
        buckets.append(dict(frame_metrics(reference, prediction, bucket & added),
                            bucket=gm.bucket_name(low, high), gaps=len(chosen)))
    false_spells = spurious_spell_changes(reference, prediction, clean, runs)
    return {"score": sg.summary(result, ref_stats, pred_stats),
            "paired_vs_clean": sg.paired_delta(reference, prediction, clean, n_draws=draws),
            "masked": frame_metrics(reference, prediction, added),
            "buckets": buckets, "d_precision": result["per_class"]["D"]["precision"],
            "d_recall": result["per_class"]["D"]["recall"],
            "d_edges": fc.dead_edges(reference, prediction, runs),
            "true_hidden_turnovers": {"n": len(ref_turns), "matched": len(matched),
                                      "missed": len(missed),
                                      "recall": _ratio(len(matched), len(ref_turns))},
            "spurious_spells_iou_05": result["spell_f1"][0.5]["n_spurious"],
            "spurious_added_spells_iou_05": false_spells["new"],
            "spurious_spells_delta_iou_05": false_spells["delta"],
            "spurious_spell_changes": false_spells,
            "changes": gm.label_changes(clean, prediction, added, graded),
            "recovery": recovery_metrics(reference, prediction, missing | requested,
                                         added, blocks, fps),
            "missing_before": missing_summary(missing, blocks, fps, reference),
            "missing_after": missing_summary(missing | requested, blocks, fps, reference),
            "inserted": length_summary(mask["lengths"], fps),
            "x_graded": int((graded & (prediction == pm.X)).sum())}


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def match_interval(values, draws=1000):
    """Bootstrap match means after averaging seeds, never treating seeds as matches."""
    if draws <= 0:
        raise ValueError("match bootstrap draws must be positive")
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"matches": 0, "mean": None, "low": None, "high": None}
    rng = np.random.default_rng(0)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    low, high = np.percentile(samples, pm.BOOTSTRAP_PERCENTILES)
    return {"matches": len(values), "mean": float(values.mean()),
            "low": float(low), "high": float(high)}


def aggregate(reports):
    """Average each match's replicates before system comparisons and provider means."""
    groups = {}
    profiles = {r["profile_md5"] for r in reports}
    engines = {r["engine_md5"] for r in reports}
    if len(profiles) > 1 or len(engines) > 1:
        raise ValueError("mixed profile or engine versions in the aggregate")
    configurations = {}
    for report in reports:
        configurations.setdefault(report["system"], set()).add(report["config_hash"])
    if any(len(hashes) > 1 for hashes in configurations.values()):
        raise ValueError("mixed configurations within one benchmark system")
    reference_versions = {}
    for report in reports:
        key = (report["input_view"], report["game"])
        reference_versions.setdefault(key, set()).add(report["reference_md5"])
    if any(len(hashes) > 1 for hashes in reference_versions.values()):
        raise ValueError("mixed references across replicates of one match")
    paired_cells = {}
    for report in reports:
        cell = (report["input_view"], report["game"], report["view"], report["seed"])
        pair = paired_cells.setdefault(cell, {})
        if report["system"] in pair:
            raise ValueError("duplicate benchmark cell")
        pair[report["system"]] = report
    for pair in paired_cells.values():
        if set(pair) != set(SYSTEMS):
            raise ValueError("incomplete system pair or unequal seed sets")
        left, right = (pair[name] for name in SYSTEMS)
        if any(left[key] != right[key] for key in
               ("mask_md5", "profile_md5", "engine_md5", "reference_md5")):
            raise ValueError("paired systems have different masks or provenance")
    for report in reports:
        key = (report["input_view"], report["game"], report["system"], report["view"])
        groups.setdefault(key, []).append(report)
    match_rows = []
    for (input_view, game, system, view), rows in sorted(groups.items()):
        seeds = [r["seed"] for r in rows]
        if len(set(seeds)) != len(seeds):
            raise ValueError("duplicate benchmark cell")
        match_rows.append({"input_view": input_view, "game": game, "system": system,
                           "view": view, "replicates": len(rows),
                           "spell_f1": _mean([r["metrics"]["score"]["spell_f1_mean"] for r in rows]),
                           "delta_vs_clean": _mean([r["metrics"]["paired_vs_clean"]["delta"] for r in rows]),
                           "false_d_rate": _mean([r["metrics"]["masked"]["false_d_rate"] for r in rows]),
                           "ownership_accuracy": _mean([r["metrics"]["masked"]["ownership_accuracy"] for r in rows]),
                           "missing_fraction": _mean([r["metrics"]["missing_after"]["missing_fraction"] for r in rows])})
    pairs, degradation = [], []
    for input_view in sorted(set(r["input_view"] for r in match_rows)):
        for view in ("clean",) + VIEWS:
            lookup = {(r["game"], r["system"]): r for r in match_rows
                      if r["input_view"] == input_view and r["view"] == view}
            for provider in ("all", "DFL", "METRICA"):
                games = [g for g in OPEN_GAMES if (provider == "all" or g.startswith(provider))
                         and all((g, s) in lookup for s in SYSTEMS)]
                deltas = [lookup[g, "gap_stop_off"]["spell_f1"]
                          - lookup[g, "current"]["spell_f1"] for g in games]
                if games:
                    for system in SYSTEMS:
                        changes = [lookup[g, system]["delta_vs_clean"] for g in games]
                        degradation.append(dict(match_interval(changes), input_view=input_view,
                                                view=view, provider=provider, system=system,
                                                comparison="masked minus clean"))
                    pairs.append(dict(match_interval(deltas), input_view=input_view,
                                      view=view, provider=provider,
                                      comparison="gap_stop_off minus current"))
    required = {(game, system, view, seed) for game in OPEN_GAMES for system in SYSTEMS
                for view, seeds in (("clean", (None,)), ("U", range(4)), ("S", range(4)))
                for seed in seeds}
    completion = {}
    for input_view in sorted(set(r["input_view"] for r in reports)):
        actual = {(r["game"], r["system"], r["view"], r["seed"]) for r in reports
                  if r["input_view"] == input_view and r["view"] != "S_alt"}
        completion[input_view] = {"complete": actual == required,
                                  "expected_cells": len(required), "actual_cells": len(actual),
                                  "missing_cells": len(required - actual),
                                  "unexpected_cells": len(actual - required)}
    return {"cells": len(reports), "match_means": match_rows, "paired_match_intervals": pairs, "degradation_intervals": degradation,
            "complete_primary_matrix": bool(completion) and all(c["complete"] for c in completion.values()),
            "completion_by_input_view": completion,
            "primary_cells": sum(r["view"] != "S_alt" for r in reports),
            "alternative_cells": sum(r["view"] == "S_alt" for r in reports),
            "method": "Average seeds within each match, then bootstrap paired match differences."}


def reference_blocks(reference, engine_blocks):
    """Split game blocks at every reference X interval."""
    reference = np.asarray(reference, np.int8)
    graded = gm.graded_mask(len(reference), engine_blocks)
    if ((reference != pm.X) & ~graded).any():
        raise ValueError("reference grades a frame outside the game blocks")
    blocks = []
    for half, first, last in engine_blocks:
        starts, stops = gm.runs_of(reference[first:last + 1] != pm.X)
        blocks.extend((half, first + int(a), first + int(b) - 1)
                      for a, b in zip(starts, stops))
    if not blocks:
        raise ValueError("reference has no graded frames")
    return blocks


def conditioned_missingness(missing, codes, blocks):
    """Count missing observations by live/dead state without consulting a prediction score."""
    missing, codes = np.asarray(missing, bool), np.asarray(codes, np.int8)
    if missing.shape != codes.shape or codes.ndim != 1:
        raise ValueError("missingness and labels must be equal-length vectors")
    graded = gm.graded_mask(len(codes), blocks) & (codes != pm.X)
    count = int(graded.sum())
    if not count:
        raise ValueError("the observation audit needs graded frames")
    result = {"graded_frames": count,
              "missing_frames": int((missing & graded).sum()),
              "missing_fraction": float((missing & graded).sum()) / count, "states": {}}
    for name, selected in (("alive", codes < pm.D), ("dead", codes == pm.D)):
        selected = selected & graded
        frames, absent = int(selected.sum()), int((selected & missing).sum())
        result["states"][name] = {"frames": frames, "missing_frames": absent,
                                  "missing_fraction": _ratio(absent, frames),
                                  "share_of_graded": frames / float(count)}
    return result


def hide_only_feasibility(source, native):
    """Test necessary marginal conditions for a hide-only source-rate match.

    Targets are the source rates, rounded to the nearest target frame. A native
    excess cannot be repaired by hiding more observations. Feasible marginals do
    not establish feasible durations, placement, class mixtures or equivalent data.
    """
    states, impossible = {}, []
    implied = 0
    for name in ("alive", "dead"):
        target_rate = source["states"][name]["missing_fraction"]
        own = native["states"][name]
        if target_rate is None or own["frames"] == 0:
            states[name] = {"status": "undefined_state_support", "target_rate": target_rate}
            impossible.append(name)
            continue
        target = int(round(target_rate * own["frames"]))
        budget = target - own["missing_frames"]
        implied += target
        states[name] = {"target_rate": target_rate, "native_rate": own["missing_fraction"],
                        "target_missing_frames": target, "additional_frames": max(0, budget),
                        "irreducible_excess_frames": max(0, -budget),
                        "status": "native_excess" if budget < 0 else "marginally_feasible"}
        if budget < 0:
            impossible.append(name)
    total_target = int(round(source["missing_fraction"] * native["graded_frames"]))
    return {"states": states, "impossible_states": impossible,
            "hide_only_marginals_feasible": not impossible,
            "source_total_target_fraction": source["missing_fraction"],
            "source_total_target_additional_frames": max(0, total_target - native["missing_frames"]),
            "source_total_target_excess_frames": max(0, native["missing_frames"] - total_target),
            "conditional_targets_imply_fraction": implied / float(native["graded_frames"]),
            "accepted_as_source_simulation": False,
            "status": "native_excess" if impossible else "marginally_feasible_not_validated",
            "reason": ("Native gaps must stay missing. Marginal feasibility alone does not "
                       "validate a source-like observation process.")}


def distribution_distance(source_summary, achieved_summary):
    """Report continuous distribution mismatches without fitted acceptance thresholds."""
    def shares(summary, key):
        values = summary[key]
        total = float(sum(values.values()))
        return {name: value / total for name, value in values.items()} if total else None

    distances = {}
    for key in ("buckets", "class_counts"):
        left, right = shares(source_summary, key), shares(achieved_summary, key)
        distances[key + "_total_variation"] = (
            0.5 * sum(abs(left[name] - right[name]) for name in left)
            if left is not None and right is not None else None)
    distances["missing_fraction_difference"] = (achieved_summary["missing_fraction"]
                                                - source_summary["missing_fraction"])
    for key in ("median_s", "p90_s"):
        left, right = source_summary[key], achieved_summary[key]
        distances[key + "_difference"] = right - left if left is not None and right is not None else None
    return distances


def observation_audit(args):
    """Audit source targets, native feasibility and existing masks before new scoring.

    This command reads no masked prediction or score report. The frozen source
    baseline defines assumed states on BAR-ATH. External labels define states only
    for placement diagnostics. No engine inference runs in this command.
    """
    from engine import data
    from tools.gt import public_gt_codes

    output = outside_repo(args.out)
    run_root = outside_repo(args.run_root)
    profile = json.loads(Path(args.profile).read_text())
    check_profile(profile)
    source = data.load_tactical("BAR-ATH", data_dir=str(Path(args.source_data_root).resolve()))
    baseline = sg.read_codes(args.source_baseline, source["n_frames"])
    missing = ~np.isfinite(source["bx"]) | ~np.isfinite(source["by"])
    if array_md5(missing) != profile["source_missing_md5"] or array_md5(baseline) != profile["source_labels_md5"]:
        raise ValueError("source observations or baseline differ from the frozen profile")
    source_context = conditioned_missingness(missing, baseline, data.graded_blocks("BAR-ATH"))
    public = public_gt_codes()
    public_context = conditioned_missingness(missing, public, data.graded_blocks("BAR-ATH"))
    matches, mask_checks = [], []
    for game in OPEN_GAMES:
        tactical = data.load_tactical(game, data_dir=str(Path(args.data_root).resolve()),
                                     cache_dir=str(run_root / game / "cache"))
        native = ~np.isfinite(tactical["bx"]) | ~np.isfinite(tactical["by"])
        reference = sg.read_codes(Path(args.references) / (game + ".csv"), tactical["n_frames"])
        blocks = reference_blocks(reference, data.graded_blocks(game))
        own = conditioned_missingness(native, reference, blocks)
        feasibility = hide_only_feasibility(source_context, own)
        matches.append({"game": game, "native": own, "feasibility": feasibility})
        for view in VIEWS:
            for seed in range(4):
                path = run_root / game / "masks" / (game + "_" + view + "_" + str(seed) + ".json")
                mask = frozen_mask(path, native, blocks, tactical["fps"], reference, profile,
                                   game, view, seed, require_existing=True)
                requested = gm.hidden_mask(len(native), mask["starts"], mask["lengths"])
                achieved = conditioned_missingness(native | requested, reference, blocks)
                shape = missing_summary(native | requested, blocks, tactical["fps"], reference)
                mask_checks.append({"game": game, "view": view, "seed": seed,
                                    "mask_md5": mask["md5"], "conditioned": achieved,
                                    "distribution": shape,
                                    "distance_from_source": distribution_distance(profile["observed"], shape),
                                    "accepted_as_source_simulation": False,
                                    "status": "diagnostic_stress_mask"})
    result = {"schema": SCHEMA_VERSION, "profile_md5": profile["md5"],
              "source_authority": "Frozen engine labels are assumed states, not full-match truth.",
              "source_engine_context": source_context, "source_public_context": public_context,
              "criterion": ("Nearest-frame source conditional missingness targets. Native excess "
                            "is impossible under hide-only masking. No score-derived tolerances."),
              "matches": matches, "masks": mask_checks,
              "accepted_as_source_simulation": False,
              "conclusion": ("All masks remain diagnostic. Matching marginal rates cannot prove "
                             "equivalent durations or state-conditioned placement.")}
    result["md5"] = digest_json(result)
    write_json(output, result)
    print("audited {} matches and {} frozen masks without prediction scores".format(
        len(matches), len(mask_checks)))
    return result


def engine_fingerprint():
    """Hash all submission modules for report provenance."""
    return digest_json({str(p.relative_to(ROOT)): gm.file_md5(str(p))
                        for p in sorted((ROOT / "engine").glob("*.py"))})


def run_match(args):
    """Run clean and frozen corrupted views for both declared systems."""
    from engine import data
    from engine.config import Config, load
    from engine.submission import write_submission, validate
    if args.game not in OPEN_GAMES:
        raise ValueError("only the eight open reference matches are allowed")
    out = outside_repo(args.out_dir)
    profile = json.loads(Path(args.profile).read_text())
    check_profile(profile)
    data.DEFAULT_CACHE_DIR = str(out / "cache")
    tactical = data.load_tactical(args.game)
    n, engine_blocks = tactical["n_frames"], data.graded_blocks(args.game)
    fps = float(tactical["fps"])
    missing = ~np.isfinite(tactical["bx"]) | ~np.isfinite(tactical["by"])
    reference = sg.read_codes(args.gt, n)
    expected_graded = gm.graded_mask(n, engine_blocks)
    blocks = reference_blocks(reference, engine_blocks)
    masks = [("clean", None, {"starts": [], "lengths": [], "md5": None})]
    for view in args.views.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            mask_root = Path(args.reuse_masks).resolve() if args.reuse_masks else out / "masks"
            path = mask_root / (args.game + "_" + view + "_" + str(seed) + ".json")
            mask = frozen_mask(path, missing, blocks, fps, reference, profile, args.game,
                               view, seed, args.fraction, require_existing=bool(args.reuse_masks))
            masks.append((view, seed, mask))
    cfg = load(args.config) if args.config else Config.default()
    source_hash = engine_fingerprint()
    for system in SYSTEMS:
        chosen = cfg if system == "current" else replace(cfg, W_GAP_STOP=0.0)
        clean = None
        for view, seed, mask in masks:
            requested = gm.hidden_mask(n, mask["starts"], mask["lengths"])
            codes = predict(tactical, requested, args.game, chosen)
            if view == "clean":
                clean = codes
            if (codes[expected_graded] == pm.X).any() or not (codes[~expected_graded] == pm.X).all():
                raise ValueError("engine output violates graded or ungraded X contract")
            name = "{}_{}_{}_{}".format(args.game, system, view, "clean" if seed is None else seed)
            csv_path = out / "predictions" / (name + ".csv")
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            # The official writer validates this game's frame count and label contract.
            write_submission(np.asarray(pm.LABELS)[codes], str(csv_path), n_frames=n, blocks=engine_blocks)
            validate(str(csv_path), rows=n)
            metrics = score_cell(reference, codes, clean, mask, missing, blocks, fps, args.draws)
            report = {"schema": SCHEMA_VERSION, "game": args.game, "system": system,
                      "input_view": args.input_view, "view": view, "seed": seed,
                      "engine_md5": source_hash, "config_hash": chosen.config_hash(),
                      "prediction_md5": gm.file_md5(str(csv_path)), "mask_md5": mask["md5"],
                      "profile_md5": profile["md5"], "reference_md5": array_md5(reference),
                      "mask": mask, "source_profile": {k: profile[k] for k in
                                                        ("observed", "mixture", "assumption", "alternative")},
                      "metrics": metrics}
            write_json(out / "reports" / (name + ".json"), report)
            print("{}: spellF1 {:.4f}, delta {:+.4f}, mask {}".format(
                name, metrics["score"]["spell_f1_mean"], metrics["paired_vs_clean"]["delta"],
                mask["md5"]), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    profile_parser = sub.add_parser("profile", help="freeze the BAR-ATH source profile")
    profile_parser.add_argument("--out", required=True)
    profile_parser.add_argument("--baseline", help="validated current-best CSV, otherwise run the engine")
    run_parser = sub.add_parser("run", help="score one open match")
    run_parser.add_argument("--game", required=True, choices=OPEN_GAMES)
    run_parser.add_argument("--gt", required=True)
    run_parser.add_argument("--profile", required=True)
    run_parser.add_argument("--out-dir", required=True)
    run_parser.add_argument("--input-view", choices=("raw", "matched", "smooth95"), default="raw")
    run_parser.add_argument("--views", default=",".join(VIEWS))
    run_parser.add_argument("--seeds", default="0,1,2,3")
    run_parser.add_argument("--fraction", type=float, default=DEFAULT_FRACTION)
    run_parser.add_argument("--config")
    run_parser.add_argument("--reuse-masks", help="read existing masks only from this folder")
    run_parser.add_argument("--draws", type=int, default=1000)
    audit_parser = sub.add_parser("audit", help="audit observation feasibility without prediction scores")
    audit_parser.add_argument("--profile", required=True)
    audit_parser.add_argument("--source-baseline", required=True)
    audit_parser.add_argument("--source-data-root", required=True)
    audit_parser.add_argument("--data-root", required=True)
    audit_parser.add_argument("--references", required=True)
    audit_parser.add_argument("--run-root", required=True, help="existing raw runs and frozen masks")
    audit_parser.add_argument("--out", required=True)
    summary_parser = sub.add_parser("summarize", help="aggregate saved reports")
    summary_parser.add_argument("directory")
    args = parser.parse_args(argv)
    if args.command == "profile":
        from engine import data
        from engine.config import Config
        from tools.gt import public_gt_codes
        output = outside_repo(args.out)
        if output.exists():
            saved = json.loads(output.read_text())
            check_profile(saved)
            print("reused source profile " + saved["md5"])
            return 0
        tactical = data.load_tactical("BAR-ATH")
        n = tactical["n_frames"]
        baseline = (sg.read_codes(args.baseline, n) if args.baseline else
                    predict(tactical, np.zeros(n, bool), "BAR-ATH", Config.default()))
        missing = ~np.isfinite(tactical["bx"]) | ~np.isfinite(tactical["by"])
        result = build_profile(missing, data.graded_blocks("BAR-ATH"), tactical["fps"],
                               baseline, public=public_gt_codes())
        write_json(output, result)
        print("froze source profile " + result["md5"])
    elif args.command == "run":
        run_match(args)
    elif args.command == "audit":
        observation_audit(args)
    else:
        directory = outside_repo(args.directory)
        reports = [json.loads(p.read_text()) for p in sorted(directory.rglob("reports/*.json"))]
        result = aggregate(reports)
        write_json(directory / "summary.json", result)
        print("summarized {} cells".format(result["cells"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
