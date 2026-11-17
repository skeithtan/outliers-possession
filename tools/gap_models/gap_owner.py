"""Optional player-only ownership correction for missing-ball intervals (PLAN P10-C).

The model is a frozen, zero-intercept logistic regression. Features change sign when
the team identities swap. The experiment tools train the model and read the labels.
This module reads no label.
The feature schema fixes the one-second and two-second contexts specified by PLAN P10-C.
Decoder costs live in the model artefact and come from training-side match folds.
"""

import hashlib
import json
import math
import re

import numpy as np

from engine.config import Config
from engine.tracks import _runs


FORMAT = "gap-owner-logistic-v1"
# These contexts and feature families are the fixed P10-C hypothesis, not fitted values.
CONTEXT_SECONDS = (-2, -1, 0, 1, 2)
BASE_FEATURES = (
    "centroid_forward_m", "longitudinal_spread_m", "longitudinal_extent_m",
    "lateral_spread_m", "visible_count", "advanced_count", "mean_speed_mps",
    "centroid_forward_velocity_mps", "centroid_lateral_velocity_mps",
    "centroid_lateral_m",
)
FEATURE_NAMES = tuple(
    "{}_{}_t{:+d}s".format(name, kind, context)
    for context in CONTEXT_SECONDS for name in BASE_FEATURES for kind in ("value", "observed")
)


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError("{} must be a positive number".format(name))
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError("{} must be a positive number".format(name))
    return float(value)


def _minimum(value):
    if value is None:
        value = Config.default().SHAPE_MIN_PLAYERS
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("min_players must be a positive integer")
    return int(value)


def _blocks(blocks, n):
    """Validate inclusive graded blocks and return them in their input order."""
    checked = []
    previous = -1
    halves = set()
    for row in blocks:
        if len(row) != 3 or any(isinstance(v, (bool, np.bool_)) or
                                not isinstance(v, (int, np.integer)) for v in row):
            raise ValueError("blocks must contain integer half, first, last triples")
        half, first, last = map(int, row)
        if first < 0 or last < first or last >= n or first <= previous or half in halves:
            raise ValueError("blocks must be ordered, distinct and within the frame array")
        checked.append((half, first, last))
        previous = last
        halves.add(half)
    return checked


def _mean(values, valid):
    count = valid.sum(axis=1)
    total = np.where(valid, values, 0.0).sum(axis=1)
    return np.divide(total, count, out=np.full(len(count), np.nan), where=count > 0)


def _velocity(values, valid, fps):
    """Differentiate within one block, without connecting separated observations."""
    result = np.full(values.shape, np.nan)
    if len(values) < 2:
        return result
    adjacent = valid[1:] & valid[:-1]
    delta = (values[1:] - values[:-1]) * fps
    result[:-1] = np.where(adjacent, delta, result[:-1])
    result[1:] = np.where(adjacent, delta, result[1:])
    if len(values) > 2:
        central = valid[:-2] & valid[1:-1] & valid[2:]
        result[1:-1] = np.where(central, (values[2:] - values[:-2]) * (fps / 2),
                                result[1:-1])
    return result


def team_feature_matrix(tactical, blocks, pitch, fps, min_players=None):
    """Return raw player features shaped (frames, H/A, features), plus ordered names.

    Coordinates arrive normalised, then become metres. Each team's attacking direction
    comes from its median longitudinal centroid relative to its opponent in that half.
    An unresolved direction leaves directional features missing. NaN marks missing evidence.
    The two team columns swap exactly when team identities swap.
    """
    px = np.asarray(tactical["px"], dtype=np.float64)
    py = np.asarray(tactical["py"], dtype=np.float64)
    teams = np.asarray(tactical["team_of_slot"])
    if px.ndim != 2 or py.shape != px.shape or teams.shape != (px.shape[1],):
        raise ValueError("player positions and team slots have incompatible shapes")
    if not np.isin(teams, (0, 1)).all():
        raise ValueError("team slots must contain only 0 and 1")
    if len(pitch) != 2:
        raise ValueError("pitch must contain length and width")
    length, width = (_positive(v, "pitch") for v in pitch)
    fps = _positive(fps, "fps")
    minimum = _minimum(min_players)
    blocks = _blocks(blocks, len(px))
    x, y = px * length, py * width
    visible = np.isfinite(x) & np.isfinite(y)
    base = np.full((len(px), 2, len(BASE_FEATURES)), np.nan)
    for _, first, last in blocks:
        sl = slice(first, last + 1)
        medians = []
        summaries = []
        for team in (0, 1):
            columns = teams == team
            tx, ty, seen = x[sl, :][:, columns], y[sl, :][:, columns], visible[sl, :][:, columns]
            count = seen.sum(axis=1)
            cx, cy = _mean(tx, seen), _mean(ty, seen)
            enough = count >= minimum
            candidates = cx[enough & np.isfinite(cx)]
            medians.append(float(np.median(candidates)) if len(candidates) else np.nan)
            summaries.append((tx, ty, seen, count, cx, cy, enough))
        direction = np.sign(medians[1] - medians[0])
        if not np.isfinite(direction) or direction == 0:
            direction = np.nan
        for team, (tx, ty, seen, count, cx, cy, enough) in enumerate(summaries):
            sign = direction if team == 0 else -direction
            row = base[sl, team, :]
            row[:, 0] = np.where(enough, (cx - length / 2) * sign, np.nan)
            row[:, 1] = np.where(enough, np.sqrt(_mean((tx - cx[:, None]) ** 2, seen)), np.nan)
            if tx.shape[1]:
                extent = np.max(np.where(seen, tx, -np.inf), axis=1) - np.min(
                    np.where(seen, tx, np.inf), axis=1)
            else:
                extent = np.full(len(cx), np.nan)
            row[:, 2] = np.where(enough, extent, np.nan)
            row[:, 3] = np.where(enough, np.sqrt(_mean((ty - cy[:, None]) ** 2, seen)), np.nan)
            row[:, 4] = count
            row[:, 5] = np.where(enough & np.isfinite(sign),
                                  (seen & ((tx - length / 2) * sign > 0)).sum(axis=1), np.nan)
            vx, vy = _velocity(tx, seen, fps), _velocity(ty, seen, fps)
            speed_valid = np.isfinite(vx) & np.isfinite(vy)
            row[:, 6] = np.where(speed_valid.sum(axis=1) >= minimum,
                                  _mean(np.hypot(vx, vy), speed_valid), np.nan)
            centroid_valid = enough & np.isfinite(cx) & np.isfinite(cy)
            row[:, 7] = _velocity(cx, centroid_valid, fps) * sign
            row[:, 8] = _velocity(cy, centroid_valid, fps) * sign
            row[:, 9] = np.where(enough, (cy - width / 2) * sign, np.nan)
    return base, list(BASE_FEATURES)


def feature_matrix(tactical, blocks, pitch, fps, min_players=None):
    """Return player-only antisymmetric features and their ordered names.

    Missing team features become zero, with separate H-minus-A observation indicators.
    One-second and two-second contexts stay within the same graded block.
    """
    blocks = _blocks(blocks, len(tactical["px"]))
    fps = _positive(fps, "fps")
    base, _ = team_feature_matrix(tactical, blocks, pitch, fps, min_players)
    result = np.zeros((len(base), len(FEATURE_NAMES)), dtype=np.float64)
    for context_index, seconds in enumerate(CONTEXT_SECONDS):
        offset = int(round(seconds * fps))
        column = context_index * len(BASE_FEATURES) * 2
        for _, first, last in blocks:
            begin, end = max(first, first - offset), min(last + 1, last + 1 - offset)
            if begin >= end:
                continue
            sample = base[begin + offset:end + offset]
            observed = np.isfinite(sample)
            value = np.where(observed, sample, 0.0)
            result[begin:end, column:column + 2 * len(BASE_FEATURES):2] = value[:, 0] - value[:, 1]
            result[begin:end, column + 1:column + 2 * len(BASE_FEATURES):2] = (
                observed[:, 0].astype(float) - observed[:, 1].astype(float))
    return result, list(FEATURE_NAMES)


def _numeric_vector(value, name):
    if not isinstance(value, list) or len(value) != len(FEATURE_NAMES):
        raise ValueError("{} must match the feature dimension".format(name))
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        raise ValueError("{} must contain numbers".format(name))
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("{} must contain finite numbers".format(name))
    return result


def _decoder(value):
    keys = {"switch_cost", "baseline_log_odds", "endpoint_log_odds"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("decoder must declare exactly {}".format(sorted(keys)))
    for key, number in value.items():
        if isinstance(number, bool) or not isinstance(number, (int, float)) or \
                not math.isfinite(number) or number < 0:
            raise ValueError("decoder {} must be finite and nonnegative".format(key))
    return value


def validate_artifact(artifact):
    """Check the frozen schema, symmetry contract and training provenance."""
    required = {"format", "feature_names", "feature_spec", "means", "scales", "coefficients",
                "decoder", "training"}
    if not isinstance(artifact, dict) or not required.issubset(artifact) or \
            set(artifact) - required - {"selection"}:
        raise ValueError("artefact has missing or unknown fields")
    if artifact["format"] != FORMAT or artifact["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("artefact format or feature schema does not match")
    spec = artifact["feature_spec"]
    if not isinstance(spec, dict) or set(spec) != {"min_players"}:
        raise ValueError("feature_spec must declare min_players")
    _minimum(spec["min_players"])
    if np.any(_numeric_vector(artifact["means"], "means") != 0):
        raise ValueError("means must be zero for team-swap symmetry")
    if np.any(_numeric_vector(artifact["scales"], "scales") <= 0):
        raise ValueError("scales must be positive")
    _numeric_vector(artifact["coefficients"], "coefficients")
    _decoder(artifact["decoder"])
    training = artifact["training"]
    if not isinstance(training, dict) or set(training) != {"games", "reference_sha256", "dataset_sha256"}:
        raise ValueError("training provenance has missing or unknown fields")
    games = training["games"]
    if not isinstance(games, list) or not games or any(not isinstance(g, str) or not g for g in games) or \
            games != sorted(set(games)):
        raise ValueError("training games must be unique sorted names")
    references = training["reference_sha256"]
    if not isinstance(references, dict) or set(references) != set(games):
        raise ValueError("reference hashes must cover the training games")
    for digest in list(references.values()) + [training["dataset_sha256"]]:
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("training hashes must be lowercase SHA-256 values")
    if "selection" in artifact and not isinstance(artifact["selection"], dict):
        raise ValueError("selection must be an object")
    try:
        json.dumps(artifact, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("artefact metadata must be finite JSON") from exc
    return artifact


def artifact_hash(artifact):
    """Return the SHA-256 of a validated artefact in canonical JSON form."""
    validate_artifact(artifact)
    payload = json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_artifact(path):
    """Read an artefact and reject duplicate keys and non-finite JSON numbers."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate artefact key: {}".format(key))
            result[key] = value
        return result

    def bad_constant(value):
        raise ValueError("non-finite artefact constant: {}".format(value))

    with open(path, encoding="utf-8") as handle:
        artifact = json.load(handle, object_pairs_hook=unique, parse_constant=bad_constant)
    return validate_artifact(artifact)


def predict_log_odds(features, artifact):
    """Return H-versus-A log odds from the frozen symmetric logistic model."""
    validate_artifact(artifact)
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES) or not np.isfinite(matrix).all():
        raise ValueError("features must be a finite matrix with the declared schema")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = (matrix / np.asarray(artifact["scales"])) @ np.asarray(artifact["coefficients"])
    if not np.isfinite(result).all():
        raise ValueError("model log odds are not finite")
    return result


def _labels(labels):
    value = np.asarray(labels)
    if value.ndim != 1 or not np.isin(value, ("H", "A", "D", "X")).all():
        raise ValueError("labels must be a one-dimensional H/A/D/X array")
    return value.astype("<U1", copy=True)


def _mask(value, n, name):
    result = np.asarray(value)
    if result.shape != (n,) or result.dtype.kind != "b":
        raise ValueError("{} must be a boolean frame mask".format(name))
    return result


def eligible_mask(labels, parts, cfg=None, game=None):
    """Mark missing-ball live frames outside protected physical evidence.

    Contact guards extend CONTACT_HALF_WINDOW frames around each event. The guard shares
    the contact detector's physical support. Half boundaries and pre-kickoff frames remain protected.
    """
    cfg = cfg or Config.default()
    labels = _labels(labels)
    n = len(labels)
    blocks = _blocks(parts["blocks"], n)
    visible = _mask(parts["ball"]["visible"], n, "ball visibility")
    mask = ~visible & np.isin(labels, ("H", "A"))
    graded = np.zeros(n, dtype=bool)
    kickoffs = parts["kickoffs"]
    for half, first, last in blocks:
        start = max(first, int(kickoffs.get(half, first)))
        graded[start:last + 1] = True
    mask &= graded
    mask &= ~_mask(parts["ghost"], n, "ghost")
    holds = parts.get("holds")
    if holds is not None:
        mask &= ~_mask(holds[0], n, "keeper holds")
    # A held-ball control stores the full protected gap in its note, even if its D override is off.
    hold_rows = parts.get("hold_rows")
    if hold_rows is not None and len(hold_rows):
        for frame, note in zip(hold_rows["frame"], hold_rows["notes"]):
            match = re.fullmatch(r"keeper_hold;end=(\d+)", str(note))
            if match:
                mask[max(0, int(frame)):min(n, int(match.group(1)) + 1)] = False
    radius = int(cfg.CONTACT_HALF_WINDOW)
    for key in ("events", "raw_events"):
        table = parts.get(key)
        if table is not None:
            for frame in table["frame"]:
                frame = int(frame)
                mask[max(0, frame - radius):min(n, frame + radius + 1)] = False
    return mask


def correct(labels, log_odds, mask, blocks, decoder):
    """Decode each eligible interval, with soft endpoint and baseline ownership priors.

    A zero model log odd preserves the baseline frame exactly. Equal-cost paths prefer
    the corresponding baseline team, which also preserves team-swap symmetry.
    """
    output = _labels(labels)
    baseline = output.copy()
    n = len(output)
    log_odds = np.asarray(log_odds, dtype=np.float64)
    if log_odds.shape != (n,) or not np.isfinite(log_odds).all():
        raise ValueError("log odds must be a finite frame vector")
    eligible = _mask(mask, n, "eligibility") & np.isin(output, ("H", "A")) & (log_odds != 0)
    blocks = _blocks(blocks, n)
    costs = _decoder(decoder)
    switch = costs["switch_cost"]
    for _, first, last in blocks:
        starts, stops = _runs(eligible[first:last + 1])
        for local_start, local_stop in zip(starts, stops):
            start, stop = int(first + local_start), int(first + local_stop)
            signs = np.where(baseline[start:stop] == "H", 1.0, -1.0)
            odds = log_odds[start:stop] + costs["baseline_log_odds"] * signs
            if start > first and baseline[start - 1] in ("H", "A"):
                odds[0] += costs["endpoint_log_odds"] * (1 if baseline[start - 1] == "H" else -1)
            if stop <= last and baseline[stop] in ("H", "A"):
                odds[-1] += costs["endpoint_log_odds"] * (1 if baseline[stop] == "H" else -1)
            if not np.isfinite(odds).all():
                raise ValueError("decoder log odds overflow")
            emissions = np.column_stack((np.maximum(-odds, 0), np.maximum(odds, 0)))
            back = np.zeros((stop - start, 2), dtype=np.int8)
            value = emissions[0].copy()
            for index in range(1, stop - start):
                value -= value.min()
                preferred = 0 if baseline[start + index - 1] == "H" else 1
                updated = np.empty(2)
                for state in (0, 1):
                    left = value[0] + (switch if state != 0 else 0)
                    right = value[1] + (switch if state != 1 else 0)
                    previous = (0 if left < right else 1) if left != right else preferred
                    back[index, state] = previous
                    updated[state] = (left if previous == 0 else right) + emissions[index, state]
                value = updated
            preferred = 0 if baseline[stop - 1] == "H" else 1
            state = (0 if value[0] < value[1] else 1) if value[0] != value[1] else preferred
            for index in range(stop - start - 1, -1, -1):
                output[start + index] = "H" if state == 0 else "A"
                state = int(back[index, state])
    return output


def apply(labels, parts, artifact, cfg=None, game=None):
    """Return corrected labels and aggregate audit counts. None keeps the baseline unchanged."""
    baseline = _labels(labels)
    if artifact is None:
        return baseline, {"enabled": False, "eligible_frames": 0, "changed_frames": 0,
                          "changed_intervals": 0}
    cfg = cfg or Config.default()
    validate_artifact(artifact)
    from engine import data, possession

    game = game or parts.get("game")
    tactical = parts.get("tactical")
    if tactical is None:
        tactical = possession.engine_tactical(cfg, game)
    pitch = parts.get("pitch")
    if pitch is None:
        pitch = data.pitch_m(game)
    features, _ = feature_matrix(tactical, parts["blocks"], pitch, parts["fps"],
                                 artifact["feature_spec"]["min_players"])
    odds = predict_log_odds(features, artifact)
    mask = eligible_mask(baseline, parts, cfg, game)
    output = correct(baseline, odds, mask, parts["blocks"], artifact["decoder"])
    changed = output != baseline
    if np.any(changed & ~mask) or not np.array_equal(np.isin(output, ("D", "X")),
                                                    np.isin(baseline, ("D", "X"))):
        raise AssertionError("gap owner correction escaped its output contract")
    return output, {"enabled": True, "artifact_sha256": artifact_hash(artifact),
                    "eligible_frames": int(mask.sum()), "changed_frames": int(changed.sum()),
                    "changed_intervals": int(len(_runs(changed)[0]))}
