"""P1.3 (RESEARCH-6 3.4, RESEARCH-2 open question 1): frozen slots that are not goalkeepers.

    python tools/analysis/frozen_slots_probe.py [--labels c22.csv] [--game BAR-ATH]

A slot is frozen-exact on a frame when its normalised position equals the previous frame's to
the bit (the millimetre and below). The tool reports the player-frames in frozen-exact runs of
2 s or more, the runs of 10 s or more (slot, team, span, goalkeeper-like per
engine.tracks.goalkeeper_slots), the overlap with the engine's own frozen_mask, and the engine
events (engine.possession.prepare / run with return_parts) whose nearest player of the event's
team is a frozen slot at the event frame. Summary statistics only. Reads no ground truth and no
video. Python 3.9.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from engine import data, possession, tracks  # noqa: E402
from engine.config import Config  # noqa: E402

LABELS = "HADX"


def runs_of(mask):
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def read_labels(path, n_frames):
    codes = np.full(n_frames, 3, dtype=np.int8)
    with open(path) as handle:
        next(handle)
        for line in handle:
            frame, label = line.strip().split(",")
            codes[int(frame)] = LABELS.index(label)
    return codes


def frozen_exact(px, py, min_frames):
    """bool (N, S): the slot sits in a run of >= min_frames frames with identical position."""
    same = np.zeros(px.shape, dtype=bool)
    same[1:] = (px[1:] == px[:-1]) & (py[1:] == py[:-1]) & np.isfinite(px[1:]) & np.isfinite(py[1:])
    out = np.zeros(px.shape, dtype=bool)
    for s in range(px.shape[1]):
        for a, b in runs_of(same[:, s]):
            # frames a-1 .. b-1 share one position: b - a + 1 identical frames
            if b - a + 1 >= min_frames:
                out[a - 1:b, s] = True
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", default=None)
    parser.add_argument("--game", default=None)
    parser.add_argument("--min-s", type=float, default=2.0)
    parser.add_argument("--long-s", type=float, default=10.0)
    args = parser.parse_args(argv)
    game = data.resolve_game(args.game)
    cfg = Config.default()
    fps = data.fps(game)
    pitch = data.pitch_m(game)
    blocks = data.graded_blocks(game)
    raw = data.load_tactical(game)
    n_frames, n_slots = raw["n_frames"], raw["n_slots"]
    team_of_slot = np.asarray(raw["team_of_slot"])
    half_of = np.zeros(n_frames, dtype=np.int8)
    for h, lo, hi in blocks:
        half_of[lo:hi + 1] = h
    graded = half_of > 0

    min_frames = int(round(args.min_s * fps))
    long_frames = int(round(args.long_s * fps))
    fz = frozen_exact(raw["px"], raw["py"], min_frames)
    visible = np.isfinite(raw["px"])
    print("game {} slots {} frames {} | frozen-exact >= {:.0f} s".format(game, n_slots, n_frames,
                                                                        args.min_s))
    print("== 1. player-frames ==")
    print("visible player-frames {} | frozen-exact {} ({:.2f}%) | on graded frames {} ({:.2f}% of graded visible)".format(
        visible.sum(), fz.sum(), 100.0 * fz.sum() / visible.sum(), (fz & graded[:, None]).sum(),
        100.0 * (fz & graded[:, None]).sum() / (visible & graded[:, None]).sum()))
    ps = np.asarray(raw["ps"])
    print("ATD speed on frozen-exact frames: NaN {} | == 0 {} | > 0 {}".format(
        (fz & np.isnan(ps)).sum(), (fz & (ps == 0)).sum(), (fz & (ps > 0)).sum()))
    eng_frozen = tracks.frozen_mask(raw, pitch, cfg)
    print("engine frozen_mask frames {} | overlap with frozen-exact {} | frozen-exact not in engine mask {}".format(
        eng_frozen.sum(), (eng_frozen & fz).sum(), (fz & ~eng_frozen).sum()))
    for team in (0, 1):
        cols = team_of_slot == team
        print("  team {}: frozen-exact player-frames {} (graded {})".format(
            "HA"[team], fz[:, cols].sum(), (fz & graded[:, None])[:, cols].sum()))

    keepers = tracks.goalkeeper_slots(raw, blocks, pitch, cfg)
    gk_by_half = {h: {keepers[(h, t)]["slot"] for t in (0, 1)} for h, _, _ in blocks}
    print("goalkeeper-like slots per half: {}".format(
        {h: {"HA"[t]: keepers[(h, t)]["slot"] for t in (0, 1)} for h, _, _ in blocks}))

    print("\n== 2. runs ==")
    all_runs = []
    for s in range(n_slots):
        for a, b in runs_of(fz[:, s]):
            all_runs.append((s, a, b))
    lens = np.array([b - a for _, a, b in all_runs])
    print("frozen-exact runs {} | length (s): median {:.1f} p90 {:.1f} max {:.1f}".format(
        len(all_runs), np.median(lens) / fps, np.quantile(lens, 0.9) / fps, lens.max() / fps))
    print("runs >= {:.0f} s: {}".format(args.long_s, (lens >= long_frames).sum()))
    for s, a, b in sorted(all_runs, key=lambda r: r[1]):
        if b - a < long_frames:
            continue
        halves = sorted(set(half_of[a:b].tolist()))
        is_gk = any(s in gk_by_half.get(h, set()) for h in halves if h)
        x_m = float(raw["px"][a, s]) * pitch[0]
        y_m = float(raw["py"][a, s]) * pitch[1]
        print("  slot {:2d} team {} frames {:6d}-{:6d} ({:6.1f} s) halves {} graded-frames {:5d} at ({:5.1f},{:5.1f}) m  GK-like: {}".format(
            s, "HA"[team_of_slot[s]], a, b - 1, (b - a) / fps,
            [h for h in halves if h] or ["outside"], int(graded[a:b].sum()), x_m, y_m, is_gk))
    gk_frames = sum(int(fz[lo:hi + 1, s].sum()) for h, lo, hi in blocks for s in gk_by_half[h] if s >= 0)
    print("frozen-exact graded player-frames on goalkeeper-like slots {} | on other slots {}".format(
        gk_frames, int((fz & graded[:, None]).sum()) - gk_frames))

    print("\n== 3. engine events whose nearest player is a frozen-exact slot ==")
    inputs = possession.prepare(cfg, game)
    labels, parts = possession.run(cfg, return_parts=True, inputs=inputs, game=game)
    feats = inputs["feats"]
    table = inputs["events"]
    frames = table["frame"].to_numpy(dtype=np.int64)
    teams = table["team"].to_numpy(dtype=object)
    kinds = table["kind"].to_numpy(dtype=object)
    owner = possession.owner_mask(table, cfg)
    near = {t: np.asarray(feats["nearest_slot_" + t], dtype=np.int64) for t in ("H", "A")}
    hit = np.zeros(len(table), dtype=bool)
    hit_eng = np.zeros(len(table), dtype=bool)
    for i, (f, t) in enumerate(zip(frames, teams)):
        if t not in ("H", "A"):
            continue
        slot = near[t][f]
        if slot >= 0:
            hit[i] = fz[f, slot]
            hit_eng[i] = eng_frozen[f, slot]
    print("event rows {} (team H/A {}) | nearest slot frozen-exact at the frame: {} | engine-frozen: {}".format(
        len(table), np.isin(teams, ("H", "A")).sum(), hit.sum(), hit_eng.sum()))
    print("owner-gate rows {} | with frozen-exact nearest slot: {} | engine-frozen: {}".format(
        owner.sum(), (owner & hit).sum(), (owner & hit_eng).sum()))
    for kind in sorted(set(kinds.tolist())):
        sel = kinds == kind
        print("  kind {:12s} rows {:5d} frozen-exact {:3d} engine-frozen {:3d} owner-gate+frozen-exact {:3d}".format(
            kind, sel.sum(), (sel & hit).sum(), (sel & hit_eng).sum(), (sel & hit & owner).sum()))
    if hit.any():
        for i in np.flatnonzero(hit)[:30]:
            print("    frame {} kind {} team {} conf {:.3f} slot {} owner-gate {}".format(
                frames[i], kinds[i], teams[i], table["confidence"].iloc[i], near[teams[i]][frames[i]], bool(owner[i])))
    owner_events = parts["owner_events"]
    print("owner events after shift/flights: {}".format(len(owner_events)))

    print("\n== 4. frames where a frozen-exact slot is the nearest player (graded) ==")
    any_fz = fz.any(axis=1)
    nt = np.asarray(feats["nearest_team"])
    near_slot = np.where(nt == 0, near["H"], np.where(nt == 1, near["A"], -1))
    near_is_fz = np.zeros(n_frames, dtype=bool)
    ok = near_slot >= 0
    near_is_fz[ok] = fz[np.flatnonzero(ok), near_slot[ok]]
    print("graded frames with any frozen-exact slot {} | nearest player (either team) frozen-exact {}".format(
        (any_fz & graded).sum(), (near_is_fz & graded).sum()))
    for t in ("H", "A"):
        sel = near[t] >= 0
        team_fz = np.zeros(n_frames, dtype=bool)
        team_fz[sel] = fz[np.flatnonzero(sel), near[t][sel]]
        d = np.asarray(feats["d" + t])
        close = team_fz & graded & (d <= cfg.R_TOUCH)
        print("  team {} nearest slot frozen-exact on {} graded frames; within R_TOUCH {:.1f} m of the ball on {}".format(
            t, (team_fz & graded).sum(), cfg.R_TOUCH, close.sum()))
    if args.labels:
        codes = read_labels(args.labels, n_frames)
        sel = near_is_fz & graded
        print("labels of {} on frames whose nearest player is frozen-exact: {}".format(
            os.path.basename(args.labels), {LABELS[k]: int((codes[sel] == k).sum()) for k in range(4)}))
        runs = runs_of(near_is_fz & graded)
        print("  runs {} | length (s) median {:.1f} max {:.1f}".format(
            len(runs), np.median([b - a for a, b in runs]) / fps if runs else 0,
            max([b - a for a, b in runs]) / fps if runs else 0))
    # Features that would change if frozen slots were dropped: n_tracks, hulls, centroids.
    print("graded frames where dropping frozen-exact slots changes n_tracks: {} (of {})".format(
        (any_fz & graded).sum(), graded.sum()))

    print("\n== 5. the long runs: over-count and the C9 guard on their graded frames ==")
    _, swaps, _ = tracks.guard_goalkeeper_swaps(raw, blocks, cfg, pitch)
    swaps = np.asarray(swaps).reshape(-1, 3)
    print("C9 guard rows (frame, live_slot, frozen_slot) in the match: {}".format(len(swaps)))
    for s, a, b in sorted(all_runs, key=lambda r: r[1]):
        if b - a < long_frames:
            continue
        team = int(team_of_slot[s])
        cols = team_of_slot == team
        sel = np.zeros(n_frames, dtype=bool)
        sel[a:b] = True
        sel &= graded
        if not sel.any():
            continue
        count = visible[sel][:, cols].sum(axis=1)
        over = count > cfg.MAX_TEAM_TRACKS
        over_without = (count - 1) > cfg.MAX_TEAM_TRACKS
        in_run = swaps[(swaps[:, 0] >= a) & (swaps[:, 0] < b)] if len(swaps) else swaps
        print("  slot {:2d} team {} graded frames {:5d}: team visible count median {:.0f} | over-counted (> {}) {} frames, "
              "{} without the frozen slot | C9 guard rows in the run {} (frozen_slot == this slot: {})".format(
                  s, "HA"[team], sel.sum(), np.median(count), cfg.MAX_TEAM_TRACKS, over.sum(),
                  over_without.sum(), len(in_run), int((in_run[:, 2] == s).sum()) if len(in_run) else 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
