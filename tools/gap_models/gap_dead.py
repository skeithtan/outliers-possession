"""Identify dead-ball intervals supported only by gap evidence (PLAN P10-D).

The provenance audit freezes the original detections and removes only the gap
weight from the evidence. Physical anchors protect their whole final dead interval.
An optional frozen model can restore complete gaps with agreeing live neighbours.
"""

from dataclasses import replace
import hashlib
import json
import math
import re

import numpy as np

from engine import data, deadball, possession
from engine.tracks import _runs
from tools.gap_models import gap_owner

FORMAT = "gap-dead-logistic-v1"
FEATURE_NAMES = tuple(
    "{}_{}_t{:+d}s".format(name, kind, context)
    for context in gap_owner.CONTEXT_SECONDS for name in gap_owner.BASE_FEATURES
    for kind in ("sum", "absolute_difference", "observed_sum", "observed_difference")
)


def _vector(value, n, name, dtype=None):
    """Read a vector of the declared length without changing its source."""
    result = np.asarray(value, dtype=dtype)
    if result.shape != (n,):
        raise ValueError("{} must have shape ({},)".format(name, n))
    return result


def _stamp_mask(n, frames, radius=0):
    """Mark valid frame stamps and their inclusive radius."""
    result = np.zeros(n, dtype=bool)
    for frame in frames:
        frame = int(frame)
        if 0 <= frame < n:
            result[max(0, frame - radius):min(n, frame + radius + 1)] = True
    return result


def gap_only_mask(inputs, run_parts, cfg, game=None):
    """Return a conservative P10-D write mask and the support of each dead interval.

    ``inputs`` is the original ``possession.prepare`` result. ``run_parts`` is
    the matching decoder result with its final labels under ``labels``. Both
    mappings must describe the same run. No label reference is read here.

    The counterfactual changes only ``W_GAP_STOP`` in ``dead_evidence``. It uses
    the original onsets, restart candidates, hold masks and timing offsets.
    A complete final D interval is protected when any frame has other support.
    Chosen kick restarts protect the interval through their existing decoder
    window. Reappearance restarts do not establish a physical stoppage alone.

    The audit contains interval coordinates. Keep it in local experiment files,
    outside the public bundle. Publish only its aggregate counts.
    """
    n = int(inputs["n_frames"])
    labels = _vector(run_parts["labels"], n, "labels")
    if not np.isin(labels, ("H", "A", "D", "X")).all():
        raise ValueError("labels must contain H, A, D or X")
    selected_game = inputs.get("game") if game is None else data.resolve_game(game)
    if game is not None and inputs.get("game") != selected_game:
        raise ValueError("inputs describe a different game")
    if "key" in inputs and inputs["key"] != possession.detection_key(cfg):
        raise ValueError("inputs describe a different detector configuration")
    blocks = inputs["blocks"]
    kickoffs = inputs["kickoffs"]
    pitch = inputs.get("pitch")
    if pitch is None:
        pitch = data.pitch_m(selected_game)
    ball, feats = inputs["ball"], inputs["feats"]
    visible = _vector(ball["visible"], n, "ball visibility", bool)
    evidence = _vector(inputs["e_dead"], n, "dead evidence", float)
    if not np.isfinite(evidence).all() or ((evidence < 0) | (evidence > 1)).any():
        raise ValueError("dead evidence must be finite probabilities")
    before_kickoff = np.zeros(n, dtype=bool)
    prior_stop = 0
    for half, first, last in blocks:
        if not (0 <= first <= last < n) or first < prior_stop:
            raise ValueError("blocks must be ordered, disjoint and inside the match")
        prior_stop = last + 1
        kick = kickoffs.get(half, first)
        if kick is not None and kick > first:
            before_kickoff[first:min(kick, last + 1)] = True

    def decision(e_dead):
        raw, exits = possession.dead_intervals(
            n, inputs["onsets"], e_dead, inputs["restarts"], blocks, kickoffs,
            cfg, inputs["fps"], inputs.get("holds"))
        shifted, shifted_exits = possession.shift_stoppage(
            raw, exits, inputs["onsets"], inputs["restarts"], cfg, blocks, kickoffs)
        return shifted, exits, shifted_exits

    base, exits, shifted_exits = decision(evidence)
    observed_base = _vector(run_parts["stoppage"], n, "stoppage", bool)
    if not np.array_equal(base, observed_base):
        raise ValueError("run parts do not match the original stoppage decision")
    no_gap_evidence = deadball.dead_evidence(
        ball, feats, inputs.get("tagging"), blocks, replace(cfg, W_GAP_STOP=0.0),
        kickoffs=kickoffs, pitch=pitch)
    without_gap, _, _ = decision(no_gap_evidence)

    onsets = inputs["onsets"]
    reasons = {
        "counterfactual_support": without_gap,
        "outside_stoppage": ~base,
        "pre_kickoff": before_kickoff,
        "out_onset": _stamp_mask(n, onsets.loc[onsets["mechanism"] == "out", "frame"]),
        "pursuit_stop": _stamp_mask(
            n, onsets.loc[onsets["mechanism"] == "pursuit_stop", "frame"]),
        "observed_out": deadball.outside_pitch(ball, pitch, cfg.PITCH_MARGIN_M),
        "placement": deadball.setpiece_mask(ball, feats, cfg),
        "ghost": _vector(inputs["ghost"], n, "ghost mask", bool),
    }
    holds = inputs.get("holds")
    reasons["keeper_hold"] = (np.zeros(n, dtype=bool) if holds is None else
                              _vector(holds[0], n, "keeper hold mask", bool))
    hold_rows = inputs.get("hold_rows")
    if hold_rows is not None:
        reasons["keeper_hold"] = reasons["keeper_hold"].copy()
        for row in hold_rows.itertuples(index=False):
            note = str(row.notes)
            if "keeper_hold;end=" not in note:
                raise ValueError("keeper hold row has no interval endpoint")
            stop = int(note.rsplit("end=", 1)[1]) + 1
            start = int(row.frame)
            if not 0 <= start < stop <= n:
                raise ValueError("keeper hold row is outside the match")
            reasons["keeper_hold"][start:stop] = True
    events = inputs.get("events")
    contact = (np.zeros(n, dtype=bool) if events is None else
               _stamp_mask(n, events["frame"]))
    reasons["observed_contact"] = contact & visible

    # Use the same first confirmed candidate at a frame as dead_intervals.
    first_kind = {}
    for row in inputs["restarts"].itertuples(index=False):
        if row.confirmed:
            first_kind.setdefault(int(row.frame), str(row.kind))
    kick_mask = np.zeros(n, dtype=bool)
    for (raw_frame, _), (shifted_frame, _) in zip(exits, shifted_exits):
        if raw_frame not in kickoffs.values() and first_kind.get(raw_frame) == "kick":
            block = next((row for row in blocks if row[1] <= raw_frame <= row[2]), None)
            if block is None:
                raise ValueError("chosen kick restart has no source block")
            for stamp in (raw_frame, shifted_frame):
                lo = max(block[1], stamp - cfg.ALIVE_EVENT_WINDOW)
                hi = min(block[2] + 1, stamp + cfg.ALIVE_EVENT_WINDOW + 1)
                kick_mask[lo:hi] = True
    reasons["kick_restart"] = kick_mask

    mask = np.zeros(n, dtype=bool)
    segments = []
    counts = {name: 0 for name in sorted(reasons)}
    for half, first, last in blocks:
        starts, stops = _runs(labels[first:last + 1] == "D")
        for start, stop in zip(starts + first, stops + first):
            start, stop = int(start), int(stop)
            support = sorted(name for name, values in reasons.items()
                             if values[start:stop].any())
            for name in support:
                counts[name] += 1
            missing = ~visible[start:stop]
            if not support:
                mask[start:stop] = missing
            segments.append({"half": int(half), "start": start, "stop": stop,
                             "support": support, "gap_only": not bool(support),
                             "missing_frames": int(missing.sum()),
                             "eligible_frames": int(mask[start:stop].sum())})
    return mask, {
        "definition": "gap-only-dead-v1",
        "segments": segments,
        "segment_count": len(segments),
        "gap_only_segments": sum(row["gap_only"] for row in segments),
        "eligible_frames": int(mask.sum()),
        "protected_by_reason": counts,
        "counterfactual_stoppage_frames": int(without_gap.sum()),
        "base_stoppage_frames": int(base.sum()),
    }


def features_from_teams(base, blocks, fps):
    """Return symmetric geometry and movement contexts from per-team features."""
    base = np.asarray(base)
    if base.ndim != 3 or base.shape[1:] != (2, len(gap_owner.BASE_FEATURES)):
        raise ValueError("team features must have shape (frames, 2, base features)")
    blocks = gap_owner._blocks(blocks, len(base))
    fps = gap_owner._positive(fps, "fps")
    result = np.zeros((len(base), len(FEATURE_NAMES)), dtype=np.float32)
    width = 4 * len(gap_owner.BASE_FEATURES)
    for context_index, seconds in enumerate(gap_owner.CONTEXT_SECONDS):
        offset = int(round(seconds * fps))
        column = context_index * width
        for _, first, last in blocks:
            begin, end = max(first, first - offset), min(last + 1, last + 1 - offset)
            if begin >= end:
                continue
            values = base[begin + offset:end + offset]
            finite = np.isfinite(values)
            filled = np.where(finite, values, 0.0)
            observed = finite.astype(float)
            result[begin:end, column:column + width:4] = filled[:, 0] + filled[:, 1]
            result[begin:end, column + 1:column + width:4] = np.abs(filled[:, 0] - filled[:, 1])
            result[begin:end, column + 2:column + width:4] = observed[:, 0] + observed[:, 1]
            result[begin:end, column + 3:column + width:4] = np.abs(observed[:, 0] - observed[:, 1])
    return result, list(FEATURE_NAMES)


def feature_matrix(tactical, blocks, pitch, fps, min_players=None):
    """Return player-only live/dead features and their fixed ordered schema."""
    base, _ = gap_owner.team_feature_matrix(tactical, blocks, pitch, fps, min_players)
    return features_from_teams(base, blocks, fps)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("{} must be a finite number".format(name))
    return float(value)


def validate_artifact(artifact):
    """Reject malformed models before inference or source-data loading."""
    required = {"format", "feature_names", "feature_spec", "means", "scales", "coefficients",
                "intercept", "threshold", "enabled", "training"}
    if not isinstance(artifact, dict) or not required.issubset(artifact) or \
            set(artifact) - required - {"selection"}:
        raise ValueError("state artefact has missing or unknown fields")
    if artifact["format"] != FORMAT or artifact["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("state artefact schema does not match")
    spec = artifact["feature_spec"]
    if not isinstance(spec, dict) or set(spec) != {"min_players"}:
        raise ValueError("feature_spec must declare min_players")
    gap_owner._minimum(spec["min_players"])
    for name in ("means", "scales", "coefficients"):
        value = artifact[name]
        if not isinstance(value, list) or len(value) != len(FEATURE_NAMES):
            raise ValueError("{} must match the state feature dimension".format(name))
        checked = np.array([_number(v, name) for v in value])
        if name == "scales" and (checked <= 0).any():
            raise ValueError("scales must be positive")
    _number(artifact["intercept"], "intercept")
    threshold = _number(artifact["threshold"], "threshold")
    if not 0.5 < threshold < 1.0:
        raise ValueError("threshold must be between one half and one")
    if not isinstance(artifact["enabled"], bool):
        raise ValueError("enabled must be boolean")
    training = artifact["training"]
    if not isinstance(training, dict) or set(training) != {"games", "reference_sha256", "dataset_sha256"}:
        raise ValueError("training provenance has missing or unknown fields")
    games = training["games"]
    if not isinstance(games, list) or not games or any(not isinstance(g, str) or not g for g in games) or \
            games != sorted(set(games)):
        raise ValueError("training games must be sorted unique names")
    hashes = training["reference_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != set(games):
        raise ValueError("reference hashes must match the training games")
    for digest in list(hashes.values()) + [training["dataset_sha256"]]:
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("training hashes must be SHA-256 strings")
    if "selection" in artifact and not isinstance(artifact["selection"], dict):
        raise ValueError("selection must be a mapping")
    try:
        json.dumps(artifact, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError("state artefact must contain finite JSON") from error
    return artifact


def load_artifact(path):
    """Read finite JSON and reject duplicate keys."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate artefact key: " + key)
            result[key] = value
        return result

    with open(path, encoding="utf8") as stream:
        model = json.load(stream, object_pairs_hook=pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              ValueError("nonfinite artefact value: " + value)))
    return validate_artifact(model)


def artifact_hash(artifact):
    """Hash the complete model and training provenance."""
    validate_artifact(artifact)
    return hashlib.sha256(json.dumps(artifact, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf8")).hexdigest()


def predict_log_odds(features, artifact):
    """Return log odds of live play from frozen training-side scaling."""
    validate_artifact(artifact)
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES) or not np.isfinite(features).all():
        raise ValueError("state features must be finite with the declared dimension")
    coefficients = np.asarray(artifact["coefficients"], dtype=np.float64)
    scales = np.asarray(artifact["scales"], dtype=np.float64)
    means = np.asarray(artifact["means"], dtype=np.float64)
    weights = coefficients / scales
    with np.errstate(over="ignore", invalid="ignore"):
        logits = features @ weights + (artifact["intercept"] - means @ weights)
    if not np.isfinite(logits).all():
        raise ValueError("state model produced nonfinite log odds")
    return logits


def actionable_runs(labels, eligible, blocks):
    """Return complete eligible D runs with agreeing immediate live neighbours."""
    labels = np.asarray(labels)
    eligible = _vector(eligible, len(labels), "eligible mask", bool)
    blocks = gap_owner._blocks(blocks, len(labels))
    result = []
    for _, first, last in blocks:
        starts, stops = _runs(eligible[first:last + 1])
        for start, stop in zip(starts + first, stops + first):
            start, stop = int(start), int(stop)
            if start <= first or stop > last or not (labels[start:stop] == "D").all():
                continue
            team = labels[start - 1]
            if team in ("H", "A") and labels[stop] == team:
                result.append((start, stop, str(team)))
    return result


def correct(labels, log_odds, eligible, blocks, threshold, enabled=True):
    """Restore a complete eligible run, or retain it without fragmentation."""
    labels = np.asarray(labels)
    logits = _vector(log_odds, len(labels), "live log odds", float)
    threshold = _number(threshold, "threshold")
    if not np.isfinite(logits).all() or not 0.5 < threshold < 1.0:
        raise ValueError("invalid log odds or live threshold")
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be boolean")
    output = labels.copy()
    if not enabled:
        return output
    cutoff = math.log(threshold / (1.0 - threshold))
    for start, stop, team in actionable_runs(labels, eligible, blocks):
        # Centre first so a constant exact-cutoff run cannot flip from summation rounding.
        if float((logits[start:stop] - cutoff).mean()) > 0.0:
            output[start:stop] = team
    return output


def apply(labels, inputs, parts, artifact, cfg, game=None):
    """Apply the optional state model after B0 without changing known live frames."""
    labels = np.asarray(labels)
    if artifact is None:
        return labels.copy(), {"enabled": False, "changed_frames": 0}
    validate_artifact(artifact)
    combined = dict(parts, labels=labels)
    eligible, audit = gap_only_mask(inputs, combined, cfg, game)
    runs = actionable_runs(labels, eligible, inputs["blocks"])
    audit.update(actionable_runs=len(runs), actionable_frames=sum(b - a for a, b, _ in runs),
                 enabled=artifact["enabled"], model_sha256=artifact_hash(artifact))
    if not artifact["enabled"]:
        audit.update(changed_frames=0, removed_runs=0)
        return labels.copy(), audit
    tactical = inputs.get("tactical")
    if tactical is None:
        tactical = possession.engine_tactical(cfg, inputs.get("game", game))
    pitch = inputs.get("pitch")
    if pitch is None:
        pitch = data.pitch_m(inputs.get("game", game))
    features, _ = feature_matrix(tactical, inputs["blocks"], pitch, inputs["fps"],
                                 artifact["feature_spec"]["min_players"])
    output = correct(labels, predict_log_odds(features, artifact), eligible,
                     inputs["blocks"], artifact["threshold"])
    changed = output != labels
    if changed[~eligible].any() or changed[labels != "D"].any():
        raise AssertionError("state correction changed a protected label")
    audit.update(changed_frames=int(changed.sum()),
                 removed_runs=sum(bool(changed[a:b].any()) for a, b, _ in runs))
    return output, audit
