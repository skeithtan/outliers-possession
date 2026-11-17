"""C9: the goalkeeper slot-swap guard on this match (RESEARCH-2 1d, M14).

The slot numbers and frames of this match live here as assertions and nowhere in engine/
(PLAN.md 5.5). The engine finds the goalkeeper-like slot of each team in each graded block with
the own-box rule of tracks.goalkeeper_slots(). The tests need the match data bundle and are
skipped without it.
"""

import os

import numpy as np
import pytest

import games
from engine import data, tracks
from engine.config import Config

CFG = Config.default()
_ASSETS = games.asset_paths()
HAVE_DATA = all(os.path.isfile(_ASSETS[k]) for k in ("atd_meta", "atd_raw"))
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="match data bundle not present")

KEEPERS = {(1, 0): 23, (1, 1): 45, (2, 0): 23, (2, 1): 45}
SWAP_FRAMES = 7999          # graded frames where the guard fires at GK_SWAP_R = 15 m (M14)


@pytest.fixture(scope="module")
def guarded():
    tactical = data.load_tactical()
    patched, swaps, keepers = tracks.guard_goalkeeper_swaps(tactical, data.graded_blocks(), CFG,
                                                            data.pitch_m())
    return tactical, patched, swaps, keepers


@pytest.mark.slow
@needs_data
def test_keeper_slots_of_this_match(guarded):
    _, _, _, keepers = guarded
    length = data.pitch_m()[0]
    assert {key: info["slot"] for key, info in keepers.items()} == KEEPERS
    for info in keepers.values():
        assert 0.86 <= info["share"] <= 0.96          # M14
    # The teams change ends at half time.
    assert keepers[(1, 0)]["own_goal_x"] == 0.0 and keepers[(2, 0)]["own_goal_x"] == length
    assert keepers[(1, 1)]["own_goal_x"] == length and keepers[(2, 1)]["own_goal_x"] == 0.0


@pytest.mark.slow
@needs_data
def test_swaps_of_this_match(guarded):
    tactical, patched, swaps, _ = guarded
    assert len(swaps) == SWAP_FRAMES
    assert set(map(tuple, swaps[:, 1:].tolist())) == {(23, 45), (45, 23)}
    by_frame = {int(f): (int(g), int(o)) for f, g, o in swaps}
    assert by_frame[11708] == (45, 23)      # A slot 45 in the H box, H slot 23 frozen 3.8 m away
    assert by_frame[15654] == (23, 45)      # H slot 23 dribbles in the A box, A slot 45 frozen
    for frame in (115403, 115409):          # the other keeper slot is not visible there
        assert frame not in by_frame
    frames = swaps[:, 0]
    assert np.array_equal(patched["px"][frames, swaps[:, 2]], tactical["px"][frames, swaps[:, 1]])
    assert np.isnan(patched["px"][frames, swaps[:, 1]]).all()
