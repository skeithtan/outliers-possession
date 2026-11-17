"""Freeze reference-live gap sensitivity masks for P10-C.

The protocol is docs/plan/calibration/GAP_OWNER_EXPERIMENT.md. These masks never
select model settings. Reference labels only constrain interval placement.
"""

import json
from pathlib import Path

import numpy as np

from tools import gap_benchmark as gb
from tools import gapmask as gm


TARGETS = (0.045, 0.09, 0.18)


def live_view(target_fraction):
    """Return the stable name of a predeclared sensitivity view."""
    if target_fraction not in TARGETS:
        raise ValueError("target must be a predeclared live missingness fraction")
    return "L{:03d}".format(int(round(1000 * target_fraction)))


def draw_live_mask(missing, blocks, fps, reference, profile, game,
                   target_fraction, seed=0, max_draws=gm.MAX_DRAWS):
    """Hide live observations until total live missingness reaches the target.

    Native gaps stay missing. A complete final interval can overshoot the target.
    H/A transitions do not split live spans. D and X frames are never hidden.
    """
    gb.check_profile(profile)
    view = live_view(target_fraction)
    missing = np.asarray(missing, dtype=bool)
    reference = np.asarray(reference, dtype=np.int8)
    if missing.ndim != 1 or reference.shape != missing.shape:
        raise ValueError("reference and missingness must be equal-length vectors")
    if not np.isin(reference, (gb.pm.H, gb.pm.A, gb.pm.D, gb.pm.X)).all():
        raise ValueError("invalid reference labels")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    live = gm.graded_mask(len(reference), blocks) & (reference < gb.pm.D)
    total_live = int(live.sum())
    native_live = int((live & missing).sum())
    target_total = int(np.ceil(target_fraction * total_live))
    target_added = max(0, target_total - native_live)

    entries = [dict(row) for row in profile["entries"] if row["class"] == "alive"]
    if not entries:
        raise ValueError("source profile has no live-gap durations")
    live_profile = {"schema": gb.SCHEMA_VERSION, "entries": entries,
                    "mixture": {kind: float(kind == "alive") for kind in gb.CLASSES},
                    "parent_md5": profile["md5"]}
    live_profile["md5"] = gb.digest_json(live_profile)
    if target_added:
        # Removing D from the placement reference also changes the denominator to live frames.
        placement = np.where(live, reference, gb.pm.X).astype(np.int8)
        drawn = gb.draw_mask(missing, blocks, fps, placement, live_profile, game, "S",
                             seed, target_added / float(total_live), max_draws)
        starts, lengths = drawn["starts"], drawn["lengths"]
        draws, rejected = drawn["draws"], drawn["rejected"]
    else:
        starts, lengths, draws, rejected = [], [], 0, 0
    hidden = gm.hidden_mask(len(reference), starts, lengths)
    added = hidden & ~missing
    if (hidden & ~live).any():
        raise AssertionError("live mask crossed a dead or ungraded frame")
    achieved_live = int((live & (missing | hidden)).sum())
    result = {
        "schema": gb.SCHEMA_VERSION, "game": game, "view": view, "seed": int(seed),
        "fraction": float(target_fraction), "fraction_denominator": "reference-live frames",
        "fps": float(fps), "profile_md5": profile["md5"],
        "draw_profile_md5": live_profile["md5"],
        "native_missing_md5": gb.array_md5(missing),
        "reference_md5": gb.array_md5(reference),
        "blocks": [[int(v) for v in row] for row in blocks],
        "starts": starts, "lengths": lengths, "classes": ["alive"] * len(starts),
        "live_frames": total_live, "native_missing_live_frames": native_live,
        "native_missing_live_fraction": native_live / float(total_live) if total_live else None,
        "target_total_live_frames": target_total, "target_frames": target_added,
        "hidden_frames": int(added.sum()), "achieved_missing_live_frames": achieved_live,
        "achieved_missing_live_fraction": achieved_live / float(total_live) if total_live else None,
        "overshoot_frames": max(0, achieved_live - target_total),
        "reached": achieved_live >= target_total, "draws": int(draws),
        "rejected": int(rejected), "max_draws": int(max_draws),
        "duration_summary": gb.length_summary(lengths, fps),
        "assumption": "Live-only sensitivity. Native dead-ball missingness is unchanged.",
    }
    result["md5"] = gb.digest_json(result)
    return result


def frozen_live_mask(path, missing, blocks, fps, reference, profile, game,
                     target_fraction, seed=0):
    """Create or verify a fixed mask without silently replacing changed inputs."""
    path = gb.outside_repo(path)
    expected = {"schema": gb.SCHEMA_VERSION, "game": game,
                "view": live_view(target_fraction), "seed": int(seed),
                "fraction": float(target_fraction), "fps": float(fps),
                "profile_md5": profile["md5"],
                "native_missing_md5": gb.array_md5(np.asarray(missing, bool)),
                "reference_md5": gb.array_md5(np.asarray(reference, np.int8)),
                "blocks": [[int(v) for v in row] for row in blocks]}
    gb.check_profile(profile)
    if path.exists():
        saved = json.loads(path.read_text())
        content = dict(saved)
        checksum = content.pop("md5", None)
        if gb.digest_json(content) != checksum or any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError("saved live mask differs from placement inputs or checksum")
        return saved
    result = draw_live_mask(missing, blocks, fps, reference, profile, game, target_fraction, seed)
    gb.write_json(path, result)
    return result
