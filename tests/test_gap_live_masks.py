"""Independent placement and immutable-input checks for live-gap sensitivity."""

import json

import numpy as np
import pytest

from tools import gap_benchmark as gb
from tools import gap_live_masks as lm
from tools import gapmask as gm


def scene():
    blocks = [(1, 10, 499), (2, 500, 989)]
    ref = np.full(1000, gb.pm.X, np.int8)
    ref[10:990] = gb.pm.H
    ref[100:150] = gb.pm.D
    ref[300:500] = gb.pm.A
    ref[600:700] = gb.pm.A
    missing = ref == gb.pm.D
    profile = {"schema": gb.SCHEMA_VERSION,
               "entries": [{"class": "alive", "duration_s": 1.0}],
               "mixture": {k: float(k == "alive") for k in gb.CLASSES}}
    profile["md5"] = gb.digest_json(profile)
    return missing, blocks, ref, profile


@pytest.mark.parametrize("fraction", lm.TARGETS)
def test_live_masks_preserve_dead_and_half_boundaries(fraction):
    missing, blocks, ref, profile = scene()
    result = lm.draw_live_mask(missing, blocks, 25, ref, profile, "synthetic", fraction)
    assert result == lm.draw_live_mask(missing, blocks, 25, ref, profile, "synthetic", fraction)
    hidden = gm.hidden_mask(len(ref), result["starts"], result["lengths"])
    assert not hidden[ref >= gb.pm.D].any()
    assert result["achieved_missing_live_fraction"] >= fraction
    assert result["hidden_frames"] == int(hidden.sum())
    for start, length in zip(result["starts"], result["lengths"]):
        assert any(a <= start and start + length - 1 <= b for _, a, b in blocks)
    swapped = ref.copy()
    swapped[ref == gb.pm.H], swapped[ref == gb.pm.A] = gb.pm.A, gb.pm.H
    other = lm.draw_live_mask(missing, blocks, 25, swapped, profile, "synthetic", fraction)
    assert result["starts"] == other["starts"]


def test_native_missingness_above_target_is_preserved():
    missing, blocks, ref, profile = scene()
    missing[200:400] = True
    result = lm.draw_live_mask(missing, blocks, 25, ref, profile, "synthetic", 0.09)
    assert result["starts"] == []
    assert result["native_missing_live_frames"] == 200
    assert result["hidden_frames"] == 0
    assert result["achieved_missing_live_frames"] == 200


def test_frozen_mask_rejects_changed_inputs_and_tampering(tmp_path):
    missing, blocks, ref, profile = scene()
    path = tmp_path / "mask.json"
    result = lm.frozen_live_mask(path, missing, blocks, 25, ref, profile, "synthetic", 0.09)
    assert lm.frozen_live_mask(path, missing, blocks, 25, ref, profile, "synthetic", 0.09) == result
    with pytest.raises(ValueError, match="placement"):
        lm.frozen_live_mask(path, missing, blocks, 25, ref, profile, "synthetic", 0.18)
    result["starts"][0] += 1
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="checksum"):
        lm.frozen_live_mask(path, missing, blocks, 25, ref, profile, "synthetic", 0.09)


def test_empty_live_reference_is_reported_without_division():
    missing, blocks, ref, profile = scene()
    ref[ref < gb.pm.D] = gb.pm.D
    result = lm.draw_live_mask(missing, blocks, 25, ref, profile, "synthetic", 0.09)
    assert result["starts"] == []
    assert result["achieved_missing_live_fraction"] is None
