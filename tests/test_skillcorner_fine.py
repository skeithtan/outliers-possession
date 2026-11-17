"""tools/skillcorner_fine.py: the fine SkillCorner reference from `possession.group`.

Synthetic input only: no match data, no network.
"""

import json
import os

import numpy as np
import pytest

from tools import skillcorner_bundle as scb
from tools import skillcorner_fine as scf

H, A, D, X = scf.H, scf.A, scf.D, scf.X


# -- the 10 Hz rule ---------------------------------------------------------------------------

def test_named_frames_take_the_field_and_dead_frames_are_unchanged():
    coarse = np.array([X, D, H, H, H, H, D, A, A, X], dtype=np.int8)
    group = np.array([H, A, A, H, A, H, H, H, A, A], dtype=np.int8)
    fine, c = scf.fine_source_labels(coarse, group)
    assert fine.tolist() == [X, D, A, H, A, H, D, H, A, X]
    assert c["live_frames"] == 6 and c["live_stretches"] == 2
    assert c["held_frames"] == 0 and c["head_frames"] == 0
    assert c["fine_differs_from_coarse_frames"] == 3
    # the field named a team on the dead frames 1 and 6: counted, never used
    assert c["group_named_outside_live_frames"] == 2


def test_hold_last_team_through_empty_frames():
    coarse = np.array([D, H, H, H, H, H, H, D], dtype=np.int8)
    group = np.array([X, H, X, A, X, X, H, X], dtype=np.int8)
    fine, c = scf.fine_source_labels(coarse, group)
    assert fine.tolist() == [D, H, H, A, A, A, H, D]
    assert c["held_frames"] == 3 and c["head_frames"] == 0
    assert c["group_empty_frames"] == 3


def test_head_of_a_stretch_takes_the_phase_team():
    # the field is empty at the start of the stretch: phase team (A) until it names a team
    coarse = np.array([D, A, A, A, A, D], dtype=np.int8)
    group = np.array([X, X, X, H, X, X], dtype=np.int8)
    fine, c = scf.fine_source_labels(coarse, group)
    assert fine.tolist() == [D, A, A, H, H, D]
    assert c["head_frames"] == 2 and c["held_frames"] == 1


def test_hold_never_crosses_a_dead_or_ungraded_frame():
    coarse = np.array([H, H, D, H, H, X, A, A], dtype=np.int8)
    group = np.array([A, X, X, X, X, X, X, H], dtype=np.int8)
    fine, c = scf.fine_source_labels(coarse, group)
    # stretch 1: A then held A; stretch 2: empty -> phase team H (HEAD); stretch 3: HEAD A, then H
    assert fine.tolist() == [A, A, D, H, H, X, A, H]
    assert c["live_stretches"] == 3
    assert c["held_frames"] == 1 and c["head_frames"] == 3


def test_hold_runs_across_adjacent_phases():
    # two adjacent phases (H then A) form one live stretch: the hold carries over the boundary
    coarse = np.array([D, H, H, A, A, D], dtype=np.int8)
    group = np.array([X, H, X, X, A, X], dtype=np.int8)
    fine, _ = scf.fine_source_labels(coarse, group)
    assert fine.tolist() == [D, H, H, H, A, D]


def test_all_empty_field_gives_the_coarse_labels():
    coarse = np.array([X, D, H, H, A, A, D, X], dtype=np.int8)
    fine, c = scf.fine_source_labels(coarse, np.full(8, X, dtype=np.int8))
    assert np.array_equal(fine, coarse)
    assert c["head_frames"] == 4 and c["fine_differs_from_coarse_frames"] == 0


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        scf.fine_source_labels(np.array([H, H], dtype=np.int8), np.array([H], dtype=np.int8))
    with pytest.raises(ValueError):
        scf.fine_source_labels(np.array([H], dtype=np.int8), np.array([D], dtype=np.int8))


def test_to_video_uses_the_bundle_grid():
    labels10 = np.array([X, D, H, A, A, D, X], dtype=np.int8)
    blocks = {1: scb.video_block(1, 5)}
    out = scf.to_video(labels10, blocks[1][1] + 1, blocks)
    nearest = scb.nearest_source(len(out))
    for v in range(len(out)):
        want = labels10[nearest[v]] if blocks[1][0] <= v <= blocks[1][1] else X
        assert out[v] == want


# -- one synthetic match end to end ----------------------------------------------------------

MID, HOME_ID, AWAY_ID = "900002", 11, 22
PERIODS = {1: (2, 12), 2: (15, 26)}
N = 30


def _period(f):
    for half, (a, b) in PERIODS.items():
        if a <= f <= b:
            return half
    return None


def _group(f):
    if _period(f) is None:
        return None
    if f in (6, 7, 20):
        return None                      # empty field inside live play
    return "away team" if f in (5, 8, 10, 11, 21) else "home team"


def _write(folder):
    os.makedirs(folder)
    raw = {"id": int(MID), "date_time": "", "home_team": {"id": HOME_ID, "name": "H FC"},
           "away_team": {"id": AWAY_ID, "name": "A FC"},
           "players": [{"id": 1, "team_id": HOME_ID, "player_role": {"acronym": "GK"}},
                       {"id": 2, "team_id": AWAY_ID, "player_role": {"acronym": "GK"}}],
           "match_periods": [{"period": h, "start_frame": a, "end_frame": b}
                             for h, (a, b) in sorted(PERIODS.items())],
           "pitch_length": 100, "pitch_width": 60,
           "home_team_side": ["left_to_right", "right_to_left"]}
    with open(os.path.join(folder, MID + "_match.json"), "w") as h:
        json.dump(raw, h)
    lines = []
    for f in range(N):
        half = _period(f)
        players = []
        if half is not None:
            gx = -40.0 if half == 1 else 40.0
            players = [{"player_id": 1, "x": gx, "y": 0.0, "is_detected": True},
                       {"player_id": 2, "x": -gx, "y": 0.0, "is_detected": True}]
        lines.append(json.dumps({
            "frame": f, "period": half,
            "ball_data": {"x": 0.0 if half else None, "y": 0.0 if half else None, "z": 0.0,
                          "is_detected": bool(half)},
            "possession": {"player_id": None, "group": _group(f)}, "player_data": players}))
    with open(os.path.join(folder, MID + "_tracking_extrapolated.jsonl"), "w") as h:
        h.write("\n".join(lines) + "\n")
    with open(os.path.join(folder, MID + "_phases_of_play.csv"), "w") as h:
        h.write("index,match_id,frame_start,frame_end,period,team_in_possession_id\n")
        for i, (a, b, half, team) in enumerate([(3, 10, 1, HOME_ID), (10, 12, 1, AWAY_ID),
                                                (17, 24, 2, HOME_ID)]):
            h.write("{},{},{},{},{},{}\n".format(i, MID, a, b, half, team))


def _setup(tmp_path):
    raw_dir = str(tmp_path / "raw" / MID)
    _write(raw_dir)
    refs = str(tmp_path / "refs" / MID)
    bundle, _ = scb.build_match(raw_dir, refs)
    cache = tmp_path / "cache"
    (cache / "labels").mkdir(parents=True)
    (cache / "feats" / "raw").mkdir(parents=True)
    np.savez(str(cache / "labels" / ("SC-" + MID + ".npz")), raw=bundle["labels"],
             ctl2=bundle["labels"])
    np.savez(str(cache / "labels" / "METRICA-S1.npz"), raw=np.zeros(3, np.int8))
    return raw_dir, refs, str(cache), bundle


def test_build_match_keeps_grid_dead_ball_and_mask(tmp_path):
    raw_dir, refs, cache, bundle = _setup(tmp_path)
    out = str(tmp_path / "out")
    s = scf.build_match(raw_dir, refs, cache, out)
    with np.load(os.path.join(out, "cache_possgroup", "labels", "SC-" + MID + ".npz")) as z:
        fine, ctl2, phases = z["raw"], z["ctl2"], z["phases"]
    coarse = bundle["labels"]
    assert np.array_equal(phases, coarse)
    assert len(fine) == len(coarse)
    assert np.array_equal(fine == X, coarse == X)
    assert np.array_equal(fine == D, coarse == D)
    # the 10 Hz fine labels by hand: phase 3..10 H, 10..12 A, 17..24 H
    want10 = np.full(N, X, dtype=np.int8)
    for half, (a, b) in PERIODS.items():
        want10[a:b + 1] = D
    want10[3:12] = [H, H, A, A, A, A, H, A, A]      # 5 A; 6, 7 held A; 8 A; 9 H; 10, 11 A
    want10[17:24] = [H, H, H, H, A, H, H]            # 20 held H; 21 A
    assert np.array_equal(fine, scf.to_video(want10, len(coarse), bundle["blocks"]))
    assert s["counts_10hz"]["held_frames"] == 3 and s["counts_10hz"]["head_frames"] == 0
    assert len(ctl2) == len(fine)
    csv = scf.read_csv_codes(os.path.join(out, "refs", "SC-" + MID + "_reference_possgroup.csv"))
    assert np.array_equal(csv, fine)
    # deterministic
    out2 = str(tmp_path / "out2")
    s2 = scf.build_match(raw_dir, refs, cache, out2)
    assert s2["outputs"] == s["outputs"]


def test_build_match_refuses_a_stale_reference(tmp_path):
    raw_dir, refs, cache, bundle = _setup(tmp_path)
    bad = bundle["labels"].copy()
    bad[np.flatnonzero(bad == H)[0]] = A
    np.savez(os.path.join(cache, "labels", "SC-" + MID + ".npz"), raw=bad, ctl2=bad)
    with pytest.raises(ValueError):
        scf.build_match(raw_dir, refs, cache, str(tmp_path / "out"))


def test_link_cache_never_writes_into_the_cache(tmp_path):
    raw_dir, refs, cache, _ = _setup(tmp_path)
    out = str(tmp_path / "out")
    before = open(os.path.join(cache, "labels", "SC-" + MID + ".npz"), "rb").read()
    scf.link_cache(cache, out, set())              # links every label, the SC one included
    scf.build_match(raw_dir, refs, cache, out)     # replaces the link, not its target
    assert open(os.path.join(cache, "labels", "SC-" + MID + ".npz"), "rb").read() == before
    lab = os.path.join(out, "cache_possgroup", "labels")
    assert not os.path.islink(os.path.join(lab, "SC-" + MID + ".npz"))
    assert os.path.islink(os.path.join(lab, "METRICA-S1.npz"))
    assert os.path.islink(os.path.join(out, "cache_possgroup", "feats"))


def test_cli_refuses_out_inside_the_cache(tmp_path):
    _, refs, cache, _ = _setup(tmp_path)
    with pytest.raises(SystemExit):
        scf.main(["build", "--raw-root", str(tmp_path / "raw"), "--refs-root",
                  str(tmp_path / "refs"), "--cache", cache, "--out",
                  os.path.join(cache, "x"), "--games", "SC-1874553"])
