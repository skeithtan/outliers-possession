"""Possession engine: per-frame H, A, D labels for the whole tactical video (PLAN.md 3.7).

`run(cfg)` is milestone M3 (card P1-C) and the submission path of `python -m engine.run`. It
builds per-frame log evidence for H, A and D on each graded block, decodes it with
engine.decoder, and writes `X` outside the graded blocks only (PLAN.md 1.7).

`run_m1(cfg)` is milestone M1 (card P1-B): the baseline's owner signal with the dead-ball overlay
of engine.deadball (`python -m engine.run --mode m1`).

`run_m2(cfg)` is milestone M2 (card P1-A): the owner is the team of the last confident release or
control of engine.events, held otherwise, with the M1 dead-ball state machine on top
(`python -m engine.run --mode m2`). It has no dynamic programming.

`run(cfg)` also applies the GT-semantics switches of card P2: POSSESSION_RULE,
FLIGHT_ASSIGNMENT and the DELTA_* offsets. `prepare(cfg)` holds the detection steps, which the
switches do not change. `run(cfg, inputs=prepare(cfg))` decodes a variant without a new
detection (tools/calibrate.py).

M1 and M2 stay as they were scored. They ignore the P2 offsets. run() reuses their parts:
dead_intervals(), owner_events(). All three read the ATD through engine_tactical(), which applies
the goalkeeper slot-swap guard of C9 before tracks.prepare().
"""

import numpy as np

from engine import data
from engine.tracks import _runs


def engine_tactical(cfg=None, game=None):
    """The tactical ATD that the engine reads: data.load_tactical() with the C9 guard applied.

    tracks.guard_goalkeeper_swaps() moves a goalkeeper slot that the tracker carried onto the
    other goalkeeper back to the right team (RESEARCH-2 1d). Before the guard,
    tracks.speed_from_positions() gives an ATD without a speed channel a player speed. An ATD
    with a speed channel passes unchanged. The --mode baseline50 path of engine.run reads the
    raw ATD and does not call this function. `game` None means games.GAME.
    """
    from engine import tracks
    from engine.config import Config

    cfg = cfg or Config.default()
    game = data.resolve_game(game)
    pitch = data.pitch_m(game)
    tactical = tracks.speed_from_positions(data.load_tactical(game), pitch)
    patched, _, _ = tracks.guard_goalkeeper_swaps(tactical, data.graded_blocks(game), cfg, pitch)
    return patched


def match_inputs(game=None):
    """(game, n_frames, blocks, kickoffs, pitch) of one match, all from games.py.

    `game` None means games.GAME. run(), run_m1() and run_m2() read the match through this
    function only, so `python -m engine.run --game <name>` changes every input at once.
    """
    from engine import deadball

    game = data.resolve_game(game)
    n_frames = data.frame_count(game)
    return (game, n_frames, data.graded_blocks(game), deadball.kickoff_frames(game),
            data.pitch_m(game))


# -- M1: dead-ball overlay on the baseline owner signal (card P1-B) -----------------------------
# A simple, explicit, deterministic state machine. P1-C replaces it with the decoder.

def hold_owner(baseline_labels, blocks):
    """Baseline H/A/D labels -> H/A everywhere inside the graded blocks: a baseline `D` (no ball
    within the smoothing window) holds the last owner (a gap alone is never D, PLAN 1.1, 3.6);
    a block that opens on `D` is back-filled from its first H/A frame."""
    owner = np.asarray(baseline_labels).astype("<U1").copy()
    for _, first, last in blocks:
        seg = owner[first:last + 1]
        known = np.flatnonzero(seg != "D")
        if known.size == 0:
            continue
        index = np.where(seg != "D", np.arange(seg.size), -1)
        index = np.maximum.accumulate(index)
        index[index < 0] = known[0]
        owner[first:last + 1] = seg[index]
    return owner


def dead_intervals(n_frames, onsets, e_dead, candidates, blocks, kickoffs, cfg, fps,
                   holds=None, knee=None):
    """M1 dead-ball decision: (dead bool (N,), [(restart_frame, team or None)]).

    Per graded block:
      * frames before the block's kickoff (kickoffs[half] > first) are D; the kickoff is a restart
        with no team (the owner signal decides);
      * enter D at an onset event (dead_onsets), or at the start of a run of e_D > DEAD_EVIDENCE_HIGH
        lasting >= COLLECTIVE_STOP_S (the gap + collective-stop case; the run must last as long as a
        collective stop; 46-frame public alive gap 17161-17207 stays out);
        collective_stop onsets are soft hints and do not enter D on their own at M1;
      * leave D at the first CONFIRMED restart candidate at least RESTART_ARM_S after entry.
        There is no exit on low e_D: after an out the ball is retrieved and moved at 4-7 m/s
        (18116-18150), and after an offside it is re-played inside D (17388-17442), both with
        low e_D (P1-B forensics). D ends at the block end if no restart follows;
      * skip an e_D entry that is no onset when `holds` (keeper_holds()) marks the entry frame
        as a goalkeeper hold and the stoppage it would start holds no frame of the stop
        signature (C13, Law 12). The next entry then decides. None keeps every entry;
      * `knee` (knee_retimer(), ONSET_KNEE) is None or a function (entry, pos) -> frame. It
        gets every e_D entry that is no onset and starts a stoppage, with `pos`, the first
        frame the state machine could enter. The stoppage then starts at the returned frame,
        held inside [pos, entry]. The restart choice, the C13 test and the next search
        position stay on the detected entry, so the rule moves an edge, never a choice
        (LEARNINGS C3 (5)). None keeps every entry frame.
    """
    dead = np.zeros(n_frames, dtype=bool)
    restart_list = []
    high = np.asarray(e_dead) > cfg.DEAD_EVIDENCE_HIGH
    dwell = max(1, int(round(cfg.COLLECTIVE_STOP_S * fps)))
    arm = max(1, int(round(cfg.RESTART_ARM_S * fps)))
    run_starts, run_stops = _runs(high)
    dwell_entries = run_starts[(run_stops - run_starts) >= dwell]
    onset_entries = onsets.loc[onsets["mechanism"] != "collective_stop", "frame"].to_numpy()
    entries = np.unique(np.concatenate([dwell_entries, onset_entries]).astype(np.int64))
    onset_set = set(int(frame) for frame in onset_entries)
    usable = candidates[candidates["confirmed"]]
    cand_frames = usable["frame"].to_numpy()
    cand_teams = usable["team"].tolist()

    for half, first, last in blocks:
        stop = last + 1
        pos = first
        kick = kickoffs.get(half)
        if kick is not None and first < kick <= last:
            dead[first:kick] = True
            restart_list.append((int(kick), None))
            pos = kick
        while pos < stop:
            later = entries[(entries >= pos) & (entries < stop)]
            if later.size == 0:
                break
            enter = int(later[0])
            ok = np.flatnonzero((cand_frames >= enter + arm) & (cand_frames < stop))
            leave = int(cand_frames[ok[0]]) if ok.size else stop
            if holds is not None and holds[0][enter] and enter not in onset_set \
                    and not holds[1][enter:leave].any():
                pos = enter + 1          # C13: a goalkeeper hold is play
                continue
            start = enter
            if knee is not None and enter not in onset_set:
                start = max(pos, min(enter, int(knee(enter, pos))))
            if ok.size == 0:
                dead[start:stop] = True
                break
            dead[start:leave] = True
            restart_list.append((leave, cand_teams[ok[0]]))
            pos = leave
    return dead, restart_list


def knee_retimer(ball, feats, events, cfg, fps, records=None, floors=None):
    """The `knee` function of dead_intervals() for ONSET_KNEE (COMPONENT_KNEE.md, IDEAS 2).

    The function (entry, pos) -> frame moves a gap-plus-stop dwell entry, one where the ball is
    not visible at the entry, to the players' deceleration knee plus DELTA_KNEE. It returns
    every other entry unchanged. The knee search starts at `lo`, the largest of:
      * `pos`, the first frame the state machine could enter,
      * the entry minus KNEE_LOOKBACK_S,
      * the last confident contact at or before the entry: an event row with confidence at
        least OWNER_MIN_CONFIDENCE (the confident test of owner_mask()),
      * with KNEE_GAP_CLAMP, the first frame of the ball gap that holds the entry.
    deadball.deceleration_knee() gives the knee k. The new entry is k + DELTA_KNEE, clipped to
    [lo, entry]. No knee keeps the entry. `floors` {entry: frame} (spell_floors(), variant 2)
    replaces the new frame of the entries it names. `records`, a list, receives one dict per
    gap entry: entry, lo, bound (the name of the binding lower bound), knee (None if none),
    new, and floor (the replaced frame, else None).
    """
    from engine import deadball

    smooth = deadball.smoothed_speed(feats["meanv_all"], cfg, fps)
    visible = np.asarray(ball["visible"], dtype=bool)
    gap0 = deadball.gap_starts(visible)
    look = int(round(float(cfg.KNEE_LOOKBACK_S) * fps))
    if events is not None and len(events):
        sure = (events["confidence"] >= cfg.OWNER_MIN_CONFIDENCE).to_numpy()
        contacts = np.sort(events["frame"].to_numpy(dtype=np.int64)[sure])
    else:
        contacts = np.zeros(0, dtype=np.int64)

    def retime(entry, pos):
        entry = int(entry)
        if visible[entry]:
            return entry
        bounds = [("pos", int(pos)), ("lookback", entry - look)]
        j = int(np.searchsorted(contacts, entry, side="right")) - 1
        if j >= 0:
            bounds.append(("contact", int(contacts[j])))
        if cfg.KNEE_GAP_CLAMP:
            bounds.append(("gap", int(gap0[entry])))
        bound, lo = max(bounds, key=lambda item: item[1])
        k = deadball.deceleration_knee(smooth, entry, lo, cfg, fps)
        new = entry if k is None else int(np.clip(k + int(cfg.DELTA_KNEE), lo, entry))
        floor = None
        if floors and entry in floors:
            floor, new = new, int(floors[entry])
        if records is not None:
            records.append({"entry": entry, "lo": int(lo), "bound": bound, "knee": k,
                            "new": new, "floor": floor})
        return new

    return retime


def spell_floors(labels, records, min_frames):
    """{entry: frame} of the re-timed entries that cut their preceding spell (variant 2).

    COMPONENT_KNEE.md Section 8. `labels` is the decoded output with the knee rows `records`
    of knee_retimer(). For a re-timed entry (new < entry) whose frame new - 1 is `H` or `A`,
    s is the first frame of that spell. If new - s < `min_frames` (KNEE_MIN_PRE_SPELL_S), the
    entry moves later to s + min_frames, or back to the entry when that is later than the
    entry. Other entries are not in the dict.
    """
    labels = np.asarray(labels)
    out = {}
    for row in records:
        entry, new = int(row["entry"]), int(row["new"])
        if new >= entry or new < 1 or labels[new - 1] not in TEAM_INDEX:
            continue
        team, start = labels[new - 1], new - 1
        while start > 0 and labels[start - 1] == team:
            start -= 1
        if new - start < min_frames:
            target = start + int(min_frames)
            out[entry] = target if target <= entry else entry
    return out


def backdate_changes(labels, d_home, d_away, visible, blocks, cfg, fps, records=None):
    """'<U1' copy of `labels` with each direct H <-> A change moved back (K2).

    calibration/COMPONENT_BACKDATE.md, RESEARCH-4 Section 4.1. `labels` is the decoded output.
    `d_home` and `d_away` are the nearest distances of each team to the ball (NaN without a
    ball or a player). `visible` is the cleaned ball mask. For each frame t1 of a graded block
    where the label changes from one team V to the other team U:

      * s is the first frame of the V spell that ends at t1 - 1, in `labels`.
      * The walk may go down to lo = max(s + floor, t1 - cap, block start). The floor is
        BACKDATE_MIN_PRE_SPELL_S and the cap is BACKDATE_CAP_S, both in frames.
      * From k = t1 - 1 down to lo, the walk goes on while the ball is visible, the nearest
        player of U is within BACKDATE_RADIUS_M of the ball and no farther than the nearest
        player of V (or V has no visible player), and the label is V.
      * If the stretch [t0, t1) holds at least BACKDATE_MIN_S in frames, its frames take U.
      * K2b: a change is not moved when the U spell that starts at t1 (in `labels`, inside the
        block) is shorter than BACKDATE_MIN_POST_SPELL_S in frames.

    The changes are read on `labels`, not on the output, so the result does not depend on the
    order of the moves. A move never crosses a block edge, never touches a D frame and never
    makes the spell before it shorter than the floor. It never moves the start of a spell shorter
    than the post-spell floor. It adds and removes no change.
    `records`, if given, receives one dict per moved change: {"t1", "t0", "team"}.
    """
    labels = np.asarray(labels)
    out = labels.copy()
    d_home = np.asarray(d_home, dtype=np.float64)
    d_away = np.asarray(d_away, dtype=np.float64)
    visible = np.asarray(visible, dtype=bool)
    radius = float(cfg.BACKDATE_RADIUS_M)
    cap = int(round(float(cfg.BACKDATE_CAP_S) * fps))
    min_frames = int(round(float(cfg.BACKDATE_MIN_S) * fps))
    floor = int(round(float(cfg.BACKDATE_MIN_PRE_SPELL_S) * fps))
    post_floor = int(round(float(cfg.BACKDATE_MIN_POST_SPELL_S) * fps))
    for _, first, last in blocks:
        spell_start = int(first)
        for t1 in range(int(first) + 1, int(last) + 1):
            before, after = labels[t1 - 1], labels[t1]
            if after == before:
                continue
            s, spell_start = spell_start, t1
            if before not in TEAM_INDEX or after not in TEAM_INDEX:
                continue
            j = t1
            while j <= int(last) and j - t1 < post_floor and labels[j] == after:
                j += 1
            if j - t1 < post_floor:
                continue
            own, other = (d_home, d_away) if after == "H" else (d_away, d_home)
            low = max(s + floor, t1 - cap, int(first))
            k = t1 - 1
            while (k >= low and visible[k] and np.isfinite(own[k]) and own[k] <= radius
                   and (not np.isfinite(other[k]) or own[k] <= other[k])
                   and labels[k] == before):
                k -= 1
            t0 = k + 1
            if t1 - t0 >= min_frames and t1 > t0:
                out[t0:t1] = after
                if records is not None:
                    records.append({"t1": int(t1), "t0": int(t0), "team": str(after)})
    return out


def keeper_positions(tactical, blocks, pitch, cfg):
    """tracks.goalkeeper_positions() with the goalkeeper-like slots of `tactical` (C13).

    `tactical` is the guarded ATD of engine_tactical().
    """
    from engine import tracks

    slots = tracks.goalkeeper_slots(tactical, blocks, pitch, cfg)
    return tracks.goalkeeper_positions(tactical, slots, blocks, pitch)


def keeper_holds(tactical, ball, feats, blocks, pitch, cfg, keepers=None):
    """(hold, stop) bool (N,) masks for dead_intervals(), or None when GK_HOLD_IS_PLAY is off.

    `hold` is deadball.keeper_hold_mask() with the goalkeeper-like slots of
    tracks.goalkeeper_slots() on `tactical` (the guarded ATD of engine_tactical()). `stop` is
    deadball.stop_signature_mask(). `keepers` is keeper_positions(), computed here when None.
    C13, F1 Section 6 rank 2.
    """
    from engine import deadball

    if not cfg.GK_HOLD_IS_PLAY:
        return None
    if keepers is None:
        keepers = keeper_positions(tactical, blocks, pitch, cfg)
    return (deadball.keeper_hold_mask(ball, keepers, pitch, cfg),
            deadball.stop_signature_mask(feats, cfg, ball["fps"]))


def keeper_hold_rows(ball, feats, slot_conf, keepers, pitch, cfg):
    """Event rows: a `control` of the goalkeeper's team at the start of each hold gap (C13b).

    One row for each gap of deadball.keeper_hold_gaps(), at the first frame of the gap. The
    confidence is events.confidence() of a control: the goalkeeper's distance to the ball and
    the margin to the nearest player of the other team, on the last visible frame, and the slot
    confidence of the goalkeeper's slot. The note is "keeper_hold;end=<last frame of the gap>".
    Physics: Law 12, a goalkeeper who catches the ball controls it. Public evidence: the ground
    truth gives the hold 13755-13800 to the goalkeeper's team from 13785. Returns None when
    GK_HOLD_OWNER is off.
    """
    import pandas as pd

    from engine import deadball, events

    if not cfg.GK_HOLD_OWNER:
        return None
    rows = []
    for start, stop, team, dist in deadball.keeper_hold_gaps(ball, keepers, pitch, cfg):
        last = start - 1
        letter = "H" if team == 0 else "A"
        other = feats["dA"][last] if letter == "H" else feats["dH"][last]
        margin = float(other) - dist if np.isfinite(other) else np.inf
        slot = int(keepers["slot"][last, team])
        conf_slot = 1.0 if slot_conf is None or slot < 0 else float(slot_conf[last, slot])
        rows.append((start, "control", letter, events.confidence(dist, margin, conf_slot, cfg),
                     dist, "keeper_hold;end={}".format(stop - 1)))
    table = pd.DataFrame(rows, columns=events.COLUMNS)
    return table.astype({"frame": np.int64, "confidence": np.float64, "dist_m": np.float64})


def add_keeper_rows(table, rows, stoppage):
    """`table` plus the keeper_hold_rows() whose gap has no frame in the `stoppage` mask (C13b).

    A hold that becomes a stoppage keeps the owner logic of C13: no row is added, so the owner
    after its restart does not change. The result is sorted as engine.events.detect_events()
    sorts its rows. `rows` None or empty returns `table` unchanged.
    """
    import pandas as pd

    from engine import events

    if rows is None or len(rows) == 0:
        return table
    ends = rows["notes"].str.split("end=").str[-1].astype(np.int64).to_numpy()
    keep = [not np.asarray(stoppage)[int(a):int(b) + 1].any()
            for a, b in zip(rows["frame"], ends)]
    rows = rows[np.asarray(keep, dtype=bool)]
    if len(rows) == 0:
        return table
    out = pd.concat([table, rows], ignore_index=True)
    order = sorted(range(len(out)), key=lambda i: (int(out["frame"].iloc[i]),
                                                    events.KINDS.index(out["kind"].iloc[i]), i))
    out = out.iloc[order].reset_index(drop=True)
    out["team"] = out["team"].astype(object).where(out["team"].notna(), None)
    return out


def m1_labels(baseline_labels, dead, restart_list, ghost, blocks, n_frames):
    """Compose M1 labels: held baseline owner, ghost-ball freeze, restart team, D, X outside.

    The restart team (if any) owns from the restart until the held baseline owner next changes
    (otherwise smooth50 would flip it back within its window); ghost-ball stretches keep the
    owner of the frame before them (R-F5)."""
    owner = hold_owner(baseline_labels, blocks)
    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    starts, stops = _runs(np.asarray(ghost, dtype=bool) & graded)
    for start, stop in zip(starts, stops):
        if start > 0 and graded[start - 1]:
            owner[start:stop] = owner[start - 1]
    changes = np.flatnonzero(owner[1:] != owner[:-1]) + 1
    for frame, team in restart_list:
        if team not in ("H", "A"):
            continue
        after = changes[changes > frame]
        block_end = next(last + 1 for _, first, last in blocks if first <= frame <= last)
        end = min(int(after[0]) if after.size else n_frames, block_end)
        owner[frame:end] = team
    labels = np.where(dead, "D", owner).astype("<U1")
    labels[~graded] = "X"
    return labels


def run_m1(cfg=None, return_parts=False, game=None):
    """M1 labels for the whole tactical video (card P1-B): baseline owner + dead-ball overlay.

    With return_parts=True also returns a dict of the intermediate arrays/tables (for tools/).
    `game` None means games.GAME.
    """
    from engine import deadball, tracks
    from engine.config import Config
    from engine.run import BASELINE_WINDOW, nearest_team_codes, smooth_majority

    cfg = cfg or Config.default()
    game, n_frames, blocks, kickoffs, pitch = match_inputs(game)
    tactical = engine_tactical(cfg, game)
    ball, feats, _ = tracks.prepare(tactical, cfg)
    tagging = data.load_tagging(game)

    e_dead = deadball.dead_evidence(ball, feats, tagging, blocks, cfg, kickoffs=kickoffs,
                                    pitch=pitch)
    onsets = deadball.dead_onsets(ball, feats, cfg, pitch=pitch)
    candidates = deadball.restarts(ball, feats, None, tagging, cfg, onsets=onsets,
                                   pitch=pitch)
    ghost = deadball.ghost_mask(ball, feats, cfg)
    fps = float(ball["fps"])
    holds = keeper_holds(tactical, ball, feats, blocks, pitch, cfg)
    dead, restart_list = dead_intervals(n_frames, onsets, e_dead, candidates, blocks, kickoffs,
                                        cfg, fps, holds)
    baseline = smooth_majority(nearest_team_codes(tactical, pitch), BASELINE_WINDOW)
    labels = m1_labels(baseline, dead, restart_list, ghost, blocks, n_frames)
    if return_parts:
        return labels, {"e_dead": e_dead, "onsets": onsets, "restarts": candidates,
                        "ghost": ghost, "dead": dead, "restart_list": restart_list,
                        "baseline": baseline}
    return labels


# -- M2: owner from events, dead-ball state machine of M1 on top (card P1-A) -------------------
# A thin, explicit, deterministic orchestrator. P1-C replaces it with the decoder.

# The dribble is an owner event of the release_or_control rule only (RESEARCH 4 rank 2).
RULE_KINDS = {"release": ("release",),
              "release_or_control": ("release", "control", "dribble")}


def owner_mask(events, cfg):
    """bool (len(events),): True on the rows of `events` that set the owner.

    A row sets the owner when all of these hold:
      * its kind is in the POSSESSION_RULE kinds (RULE_KINDS),
      * its team is "H" or "A",
      * it is not a deflection and its confidence is at least OWNER_MIN_CONFIDENCE.
    A deflection row of a rule kind sets the owner at its own frame only if a qualifying row of
    the same team follows within DEFLECTION_CONFIRM_FRAMES (PLAN 3.3, deflection filter on).
    """
    if events is None or len(events) == 0:
        return np.zeros(0, dtype=bool)
    kinds = RULE_KINDS[cfg.POSSESSION_RULE]
    rule = (events["kind"].isin(kinds) & events["team"].isin(("H", "A"))).to_numpy()
    deflection = events["notes"].astype(str).str.split(";").apply(
        lambda n: "deflection" in n).to_numpy(dtype=bool)
    confident = (events["confidence"] >= cfg.OWNER_MIN_CONFIDENCE).to_numpy()
    plain = rule & ~deflection & confident
    frames = events["frame"].to_numpy(dtype=np.int64)
    teams = events["team"].to_numpy(dtype=object)
    mask = plain.copy()
    q_frames, q_teams = frames[plain], teams[plain]
    for idx in np.flatnonzero(rule & deflection):
        later = (q_frames > frames[idx]) & (q_frames <= frames[idx] + cfg.DEFLECTION_CONFIRM_FRAMES)
        if np.any(later & (q_teams == teams[idx])):
            mask[idx] = True
    return mask


def owner_events(events, cfg):
    """[(frame, team)] of the events that set the owner (owner_mask), sorted by frame.

    FLIGHT_ASSIGNMENT does not act here. run() applies it after this step (flight_assignment).
    """
    if events is None or len(events) == 0:
        return []
    rows = events[owner_mask(events, cfg)]
    return sorted((int(f), str(t)) for f, t in zip(rows["frame"], rows["team"]))


def m2_owner(owner_list, restart_list, blocks, n_frames):
    """H/A owner on every graded frame, '' elsewhere.

    Per graded block the owner is the team of the last owner event at or before the frame. A
    restart with a team (dead_intervals) is an owner event at its frame, as in M1. Frames before
    the first owner event of a block take the kickoff team if the kickoff restart gives one,
    else the team of the first owner event. A block without any owner event is all "H".
    """
    owner = np.full(n_frames, "", dtype="<U1")
    changes = list(owner_list) + [(int(f), t) for f, t in restart_list if t in ("H", "A")]
    changes.sort(key=lambda item: item[0])
    frames = np.array([f for f, _ in changes], dtype=np.int64)
    teams = [t for _, t in changes]
    for _, first, last in blocks:
        inside = np.flatnonzero((frames >= first) & (frames <= last))
        if inside.size == 0:
            owner[first:last + 1] = "H"
            continue
        kickoff_team = next((t for f, t in restart_list
                         if first <= f <= last and f <= frames[inside[0]] and t in ("H", "A")),
                        None)
        current = kickoff_team or teams[inside[0]]
        pos = first
        for idx in inside:
            frame = int(frames[idx])
            owner[pos:frame] = current
            current, pos = teams[idx], frame
        owner[pos:last + 1] = current
    return owner


def run_m2(cfg=None, return_parts=False, game=None):
    """M2 labels for the whole tactical video (card P1-A): event owner + dead-ball overlay.

    Steps:
      1. tracks.prepare() and deadball.dead_evidence() as in M1.
      2. events.detect_events(), suppressed where e_D > EVENT_SUPPRESS_DEAD.
      3. deadball.dead_onsets() and deadball.restarts() with the real events.
      4. dead_intervals() of M1, unchanged.
      5. The owner from owner_events() and m2_owner(), `D` on dead frames, `X` outside the
         graded blocks (the writer refuses `X` on a graded frame).
    M2 ignores the P2 offsets and raises NotImplementedError with FLIGHT_ASSIGNMENT on. M2
    always uses the "dv3" contact rows it was scored with (CONTACT_DETECTOR, dv3_config).
    With return_parts=True also returns a dict of the intermediate arrays and tables.
    `game` None means games.GAME.
    """
    from engine import deadball, events, tracks
    from engine.config import Config

    cfg = dv3_config(cfg or Config.default())
    if cfg.FLIGHT_ASSIGNMENT:
        raise NotImplementedError("M2 runs as it was scored, with FLIGHT_ASSIGNMENT off")
    game, n_frames, blocks, kickoffs, pitch = match_inputs(game)
    tactical = engine_tactical(cfg, game)
    ball, feats, slot_conf = tracks.prepare(tactical, cfg)
    tagging = data.load_tagging(game)

    e_dead = deadball.dead_evidence(ball, feats, tagging, blocks, cfg, kickoffs=kickoffs,
                                pitch=pitch)
    table = events.detect_events(ball, feats, tactical, slot_conf, cfg, e_dead=e_dead)
    onsets = deadball.dead_onsets(ball, feats, cfg, pitch=pitch, events=table)
    candidates = deadball.restarts(ball, feats, table, tagging, cfg, onsets=onsets,
                               pitch=pitch)
    fps = float(ball["fps"])
    holds = keeper_holds(tactical, ball, feats, blocks, pitch, cfg)
    dead, restart_list = dead_intervals(n_frames, onsets, e_dead, candidates, blocks, kickoffs,
                                    cfg, fps, holds)
    owners = owner_events(table, cfg)
    owner = m2_owner(owners, restart_list, blocks, n_frames)
    labels = np.where(dead, "D", owner).astype("<U1")
    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    labels[~graded] = "X"
    if return_parts:
        return labels, {"e_dead": e_dead, "events": table, "onsets": onsets,
                        "restarts": candidates, "dead": dead, "restart_list": restart_list,
                        "owner_events": owners, "owner": owner}
    return labels


# -- M3: evidence and the first-order decoder (card P1-C) ---------------------------------------

UNKNOWN = -1
TEAM_INDEX = {"H": 0, "A": 1}
TEAM_EVENT_KINDS = ("touch", "release", "control", "dribble")
# The event kinds that take part in the owner walk and the switch costs, for each
# POSSESSION_RULE. The `release` rule ignores the control and dribble rows completely (P2 card,
# RESEARCH 4 rank 2). A dribble of the new team opens the cheap switch as a control does.
RULE_TEAM_KINDS = {"release": ("touch", "release"),
                   "release_or_control": TEAM_EVENT_KINDS}


GOALS_CODE = "GOALS"           # Smart Tagging code of a goal window (PLAN 1.6)
OTHER_TEAM = {"H": "A", "A": "H"}


def mark_team(mark, frame, cfg):
    """(team, distance): the kickoff team by the centre mark rule (Law 8, RESEARCH rank 3).

    `mark` is the tracks.mark_nearest() dict. The rule takes the single tracked player nearest
    the centre mark over [frame - KICKOFF_SEARCH_FRAMES, frame]. The team is "H" or "A" if
    that player is within KICKOFF_TAKER_M of the mark, else None (the taker is not tracked).
    The distance is that player's distance in metres, inf without a tracked player. The first
    frame of the minimum wins a tie.
    """
    if mark is None:
        return None, float("inf")
    dist = np.asarray(mark["dist"], dtype=np.float64)
    hi = min(int(frame) + 1, dist.shape[0])
    lo = max(0, int(frame) - int(cfg.KICKOFF_SEARCH_FRAMES))
    if hi <= lo:
        return None, float("inf")
    window = np.where(np.isfinite(dist[lo:hi]), dist[lo:hi], np.inf)
    at = lo + int(np.argmin(window))
    best = float(window[at - lo])
    team = int(mark["team"][at])
    if not np.isfinite(best) or best > cfg.KICKOFF_TAKER_M or team not in data.SIDE_LETTER:
        return None, best
    return data.SIDE_LETTER[team], best


def kickoff_decisions(kickoffs, candidates, cfg, mark=None):
    """{half: (team, rule, distance)}: the team that takes each kickoff of games.py.

    The rules, in this order:
      * "mark": mark_team() gives a team (Law 8: the taker stands at the centre mark).
      * "restart": the team of the first confirmed restart candidate within
        ALIVE_EVENT_WINDOW frames of the kickoff frame (the kicker is the player nearest the
        ball, P1-C card).
      * "law": the other half of the same pair (halves 1 and 2, 3 and 4) has a team from one
        of the two rules above. Law 8: the team that did not take the kickoff of the first half
        takes the kickoff of the second half.
      * "none": no rule gives a team. The owner is unknown until the first owner event of
        the block.
    The law rule only fills a missing team. It does not change a team that a rule above gives.
    The distance is the mark_team() distance in metres.
    """
    usable = candidates[candidates["confirmed"] & candidates["team"].isin(("H", "A"))]
    frames = usable["frame"].to_numpy(dtype=np.int64)
    teams = usable["team"].tolist()
    out = {}
    for half in sorted(kickoffs):
        frame = int(kickoffs[half])
        team, dist = mark_team(mark, frame, cfg)
        if team is not None:
            out[half] = (team, "mark", dist)
            continue
        near = np.flatnonzero(np.abs(frames - frame) <= cfg.ALIVE_EVENT_WINDOW)
        if near.size:
            out[half] = (teams[int(near[0])], "restart", dist)
        else:
            out[half] = (None, "none", dist)
    for half in sorted(out):
        partner = half + 1 if half % 2 == 1 else half - 1
        team, rule, dist = out[half]
        if team is None and partner in out and out[partner][1] in ("mark", "restart"):
            out[half] = (OTHER_TEAM[out[partner][0]], "law", dist)
    return out


def kickoff_teams(kickoffs, candidates, cfg, mark=None):
    """{half: "H", "A" or None}: the team of each kickoff of games.py (kickoff_decisions).

    The first-half kickoff is the first graded frame, so dead_intervals() has no kickoff entry
    for it. This function covers both halves.
    """
    return {half: row[0] for half, row in kickoff_decisions(kickoffs, candidates, cfg,
                                                       mark).items()}


def goal_kickoffs(tagging, stoppage, restart_list, blocks, cfg, mark=None):
    """[(frame, team, rule, distance)]: the kickoffs after a goal.

    The end of a Smart Tagging GOALS window is the kickoff after the goal (PLAN 1.6). If that
    frame is inside a stoppage of a graded block, the restart that ends the stoppage is the
    kickoff. Its frame is the first alive frame after the stoppage, and it must be in
    `restart_list` (dead_intervals(), after shift_stoppage()). A GOALS end on an alive frame or
    a stoppage that runs to the end of the block gives no kickoff.
    The team is "mark" if mark_team() at the kickoff frame gives one. Else it is "restart":
    the team the restart already has in `restart_list` (None if it has none). The GOALS team is
    not used, because the documentation does not say if it names the scoring team (RESEARCH
    open question 2).
    """
    stoppage = np.asarray(stoppage, dtype=bool)
    restart_team = dict((int(f), t) for f, t in restart_list)
    ends = set()
    if tagging is not None and len(tagging):
        codes = tagging["code"].astype(str).str.upper()
        ends = set(int(f) for f in tagging.loc[codes == GOALS_CODE, "end_frame"])
    rows = []
    for end in sorted(ends):
        block = next(((first, last) for _, first, last in blocks if first <= end <= last), None)
        if block is None or not stoppage[end]:
            continue
        alive = np.flatnonzero(~stoppage[end:block[1] + 1])
        if alive.size == 0:
            continue
        frame = end + int(alive[0])
        if frame not in restart_team:
            continue
        team, dist = mark_team(mark, frame, cfg)
        if team is not None:
            rows.append((frame, team, "mark", dist))
        else:
            rows.append((frame, restart_team[frame], "restart", dist))
    return rows


def apply_goal_kickoffs(restart_list, restart_events, rows):
    """(restart_list, restart_events) with the team of each goal_kickoffs() row.

    In both lists an entry on the frame of a row takes the team of the row. A row with no
    team changes nothing. A row whose frame is not in restart_events is added to it.
    """
    teams = dict((int(f), t) for f, t, _, _ in rows if t in TEAM_INDEX)
    if not teams:
        return list(restart_list), list(restart_events)
    new_list = [(f, teams.get(int(f), t)) for f, t in restart_list]
    new_events = [(f, teams.get(int(f), t)) for f, t in restart_events]
    present = set(int(f) for f, _ in restart_events)
    new_events += [(f, t) for f, t in sorted(teams.items()) if f not in present]
    new_events.sort(key=lambda item: (item[0], str(item[1])))
    return new_list, new_events


def team_event_rows(events, deflections=True, kinds=TEAM_EVENT_KINDS):
    """[(frame, team)] of the rows of `kinds` that have a team, sorted by frame.

    With deflections=False the rows with the note `deflection` are left out. run() passes the
    kinds of RULE_TEAM_KINDS, so the `release` rule leaves out the control and dribble rows.
    """
    if events is None or len(events) == 0:
        return []
    rows = events[events["kind"].isin(kinds) & events["team"].isin(("H", "A"))]
    if not deflections:
        notes = rows["notes"].astype(str).str.split(";")
        rows = rows[~notes.apply(lambda n: "deflection" in n).to_numpy()]
    return sorted((int(f), str(t)) for f, t in zip(rows["frame"], rows["team"]))


def release_rows(events, cfg):
    """[(frame, team)] of the rows that can release the owner, sorted by frame (owner_track).

    The kinds are those of RULE_TEAM_KINDS[POSSESSION_RULE] that are also in
    OWNER_RELEASE_KINDS. Deflections are left out. A bare touch does not release the owner
    (config provenance, C7). It still opens the cheap switch of the decoder (team_event_rows).
    """
    kinds = tuple(k for k in RULE_TEAM_KINDS[cfg.POSSESSION_RULE]
                  if k in cfg.OWNER_RELEASE_KINDS)
    return team_event_rows(events, deflections=False, kinds=kinds)


def owner_track(owner_list, restart_list, maybe_rows, blocks, n_frames, kick_teams=None,
                kickoffs=None):
    """int8 (n_frames,): the event owner, 0 for H, 1 for A, UNKNOWN (-1) where it is not known.

    The walk goes through each graded block in time order:
      * an owner event (owner_events(): rule v1, deflection confirm) sets the owner to its team,
      * a restart of restart_list with a team sets the owner to that team, as in M2,
      * a kickoff with a team (kick_teams, kickoffs: {half: ...}) sets the owner to that team,
      * an event of maybe_rows by the team that does not own the ball makes the owner UNKNOWN
        until the next owner event, restart or kickoff with a team. run() passes
        release_rows(): the releases, controls and dribbles that are not owner events and not
        deflections. The delayed nearest player decides the team there (PLAN 2.2, P1-C card:
        nearest-player evidence may move the owner only near a new-team event). An
        unconfirmed deflection, a contested touch and a bare touch change nothing (PLAN 3.3,
        C7).
    Before the first owner event of a block the owner is UNKNOWN unless the kickoff has a team.
    Frames outside the blocks are UNKNOWN.
    """
    owner = np.full(int(n_frames), UNKNOWN, dtype=np.int8)
    sets = [(int(f), 0, TEAM_INDEX[t]) for f, t in owner_list]
    sets += [(int(f), 0, TEAM_INDEX[t]) for f, t in restart_list if t in TEAM_INDEX]
    for half, team in sorted((kick_teams or {}).items()):
        if team in TEAM_INDEX:
            sets.append((int(kickoffs[half]), 0, TEAM_INDEX[team]))
    setting = set((f, t) for f, t in owner_list)
    maybes = [(int(f), 1, TEAM_INDEX[t]) for f, t in maybe_rows if (f, t) not in setting]
    changes = sorted(sets + maybes)
    frames = np.array([c[0] for c in changes], dtype=np.int64)
    for _, first, last in blocks:
        state, pos = UNKNOWN, first
        for idx in np.flatnonzero((frames >= first) & (frames <= last)):
            frame, kind, team = changes[int(idx)]
            owner[pos:frame] = state
            if kind == 0:
                state = team
            elif state != UNKNOWN and team != state:
                state = UNKNOWN
            pos = frame
        owner[pos:last + 1] = state
    return owner


def stretch_mean(values, unknown, boundaries):
    """float64 (N,): `values` replaced by their mean over each stretch of UNKNOWN owner.

    A stretch is a run of `unknown` frames cut at every frame of `boundaries` (the team events
    and the edges of the stoppage). Possession changes only when a team plays the ball
    (PLAN 1.3), so the team cannot change between two events. A constant value per stretch
    gives the decoder no reason to switch between events, and the sum still decides the team of
    the stretch. Frames with a known owner keep 0.
    """
    values = np.asarray(values, dtype=np.float64)
    unknown = np.asarray(unknown, dtype=bool)
    n = values.shape[0]
    cut = np.zeros(n, dtype=bool)
    frames = np.asarray(boundaries, dtype=np.int64)
    cut[frames[(frames >= 0) & (frames < n)]] = True
    cut[0] = True
    cut[1:] |= unknown[1:] != unknown[:-1]
    stretch = np.cumsum(cut) - 1
    sums = np.bincount(stretch[unknown], weights=values[unknown],
                       minlength=stretch[-1] + 1).astype(np.float64)
    counts = np.bincount(stretch[unknown], minlength=stretch[-1] + 1)
    mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return np.where(unknown, mean[stretch], 0.0)


def _log_odds(p):
    """log(p / (1 - p))."""
    return float(np.log(p / (1.0 - p)))


def nearest_log_odds(feats, ghost, cfg):
    """float64 (N,): H-versus-A log-odds of the delayed nearest player (PLAN 1.2, 1.3).

    The value at frame t comes from the nearest player at t - NEAREST_LAG. It is
    +-log-odds(P_NEAR_CLOSE) within NEAR_CLOSE_M of the ball, +-log-odds(P_NEAR_FAR) within
    NEAR_FAR_M, and 0 without a ball, beyond NEAR_FAR_M, or on a ghost-ball frame (R-F5). The
    sign is + for H.
    """
    dist = np.asarray(feats["dmin"], dtype=np.float64)
    team = np.asarray(feats["nearest_team"])
    finite = np.isfinite(dist)
    safe = np.where(finite, dist, np.inf)
    size = np.where(safe <= cfg.NEAR_CLOSE_M, _log_odds(cfg.P_NEAR_CLOSE),
                    np.where(safe <= cfg.NEAR_FAR_M, _log_odds(cfg.P_NEAR_FAR), 0.0))
    sign = np.where(team == data.HOME, 1.0, np.where(team == data.AWAY, -1.0, 0.0))
    raw = np.where(finite & ~np.asarray(ghost, dtype=bool), sign * size, 0.0)
    lag = int(cfg.NEAREST_LAG)
    out = np.zeros_like(raw)
    if lag <= 0:
        return raw
    out[lag:] = raw[:-lag]
    return out


def tag_log_odds(tagging, n_frames, cfg):
    """float64 (N,): +-log(TAG_TEAM_PRIOR_RATIO) on frames of a team phase of Smart Tagging.

    Only codes in TAG_TEAM_PHASE_CODES with a team count (PLAN 1.6). The sign is + for H. A
    frame tagged for both teams gets 0.
    """
    home = np.zeros(int(n_frames), dtype=bool)
    away = np.zeros(int(n_frames), dtype=bool)
    if tagging is not None and len(tagging):
        wanted = set(code.upper() for code in cfg.TAG_TEAM_PHASE_CODES)
        for row in tagging.itertuples(index=False):
            if str(row.code).upper() not in wanted or row.team not in TEAM_INDEX:
                continue
            lo, hi = max(0, int(row.start_frame)), min(int(n_frames), int(row.end_frame))
            if hi > lo:
                (home if row.team == "H" else away)[lo:hi] = True
    step = float(np.log(cfg.TAG_TEAM_PRIOR_RATIO))
    return np.where(home & ~away, step, np.where(away & ~home, -step, 0.0))


def dead_probability(e_dead, stoppage, cfg):
    """float64 (N,): P(D) per frame, before the floor.

    With `stoppage` None it is e_D. With a bool stoppage mask (the `dead` output of
    dead_intervals) it is max(e_D, P_DEAD_STOPPAGE) inside the stoppage and
    min(e_D, 1 - P_DEAD_STOPPAGE) outside it (config provenance: 15889, 17161-17207).
    """
    e_dead = np.asarray(e_dead, dtype=np.float64)
    if stoppage is None:
        return e_dead
    stoppage = np.asarray(stoppage, dtype=bool)
    return np.where(stoppage, np.maximum(e_dead, cfg.P_DEAD_STOPPAGE),
                    np.minimum(e_dead, 1.0 - cfg.P_DEAD_STOPPAGE))


def emissions(owner, unknown_lo, tag_lo, p_dead, cfg):
    """float64 (N, 3): log evidence for H, A and D.

    The H-versus-A log-odds z is +-log-odds(P_OWNER) plus the Smart Tagging prior where the
    event owner is known. Where it is UNKNOWN, z is `unknown_lo`: the stretch mean of the
    delayed nearest-player log-odds plus the Smart Tagging prior (stretch_mean). Then
    log P(H) = log(1 - p_D) + log sigmoid(z), log P(A) = log(1 - p_D) + log sigmoid(-z) and
    log P(D) = log p_D, with p_D clipped to [P_FLOOR, 1 - P_FLOOR].
    """
    owner = np.asarray(owner)
    step = _log_odds(cfg.P_OWNER)
    known = np.where(owner == 0, step, -step) + np.asarray(tag_lo, dtype=np.float64)
    z = np.where(owner == UNKNOWN, np.asarray(unknown_lo, dtype=np.float64), known)
    p_dead = np.clip(np.asarray(p_dead, dtype=np.float64), cfg.P_FLOOR, 1.0 - cfg.P_FLOOR)
    alive = np.log1p(-p_dead)
    out = np.empty((owner.shape[0], 3), dtype=np.float64)
    out[:, 0] = alive - np.logaddexp(0.0, -z)
    out[:, 1] = alive - np.logaddexp(0.0, z)
    out[:, 2] = np.log(p_dead)
    return out


def decode_blocks(log_e, blocks, kickoffs, team_rows, onset_frames, restart_events, cfg,
                  n_frames):
    """'<U1' (n_frames,): decoded H/A/D on the graded blocks, X elsewhere.

    Each block is decoded on its own (engine.decoder.decode). Frames of a block before its
    kickoff are forced to D (PLAN 3.4: the kickoff comes from games.py). Event frames are
    global frame numbers. The function makes them local to the block.
    """
    from engine import decoder

    labels = np.full(int(n_frames), "X", dtype="<U1")
    letters = np.array(decoder.STATES)
    team_frames = {t: np.array([f for f, tt in team_rows if tt == t], dtype=np.int64)
                   for t in TEAM_INDEX}
    onset_frames = np.asarray(onset_frames, dtype=np.int64)
    for half, first, last in blocks:
        stop = last + 1
        n = stop - first

        def local(frames):
            frames = np.asarray(frames, dtype=np.int64)
            return frames[(frames >= first) & (frames < stop)] - first

        restarts = [(int(f) - first, t) for f, t in restart_events if first <= f < stop]
        costs = decoder.switch_costs(n, cfg, {t: local(fr) for t, fr in team_frames.items()},
                                 local(onset_frames), restarts)
        forced = np.full(n, decoder.NO_FORCE, dtype=np.int64)
        kick = kickoffs.get(half)
        if kick is not None and first < kick <= last:
            forced[:kick - first] = decoder.D
        path = decoder.decode(log_e[first:stop], costs, cfg, forced)
        labels[first:stop] = letters[path]
    return labels


# -- P2: GT-semantics switches (card P2, PLAN.md 1.3, 1.4, 3.7) ---------------------------------

# The Config fields that run() reads after prepare(). They change only the decoder input, never
# the detection, so one prepare() result serves every value of them (tools/calibrate.py).
SEMANTIC_FIELDS = ("POSSESSION_RULE", "OWNER_RELEASE_KINDS", "FLIGHT_ASSIGNMENT",
                   "DELTA_SWITCH", "DELTA_DEAD", "DELTA_RESTART_KICK", "DELTA_RESTART_REAPPEAR",
                   "ONSET_KNEE", "DELTA_KNEE", "KNEE_LOOKBACK_S", "KNEE_SMOOTH_S",
                   "KNEE_PRE_FROM_S", "KNEE_PRE_TO_S", "KNEE_LEVEL_FRAC", "KNEE_GAP_CLAMP",
                   "KNEE_MIN_PRE_SPELL_S", "BACKDATE_CHANGES", "BACKDATE_RADIUS_M",
                   "BACKDATE_CAP_S", "BACKDATE_MIN_S", "BACKDATE_MIN_PRE_SPELL_S",
                   "BACKDATE_MIN_POST_SPELL_S")


def detection_key(cfg):
    """((name, value), ...) of every Config field that prepare() reads, sorted by name."""
    return tuple(sorted((k, v) for k, v in cfg.as_dict().items() if k not in SEMANTIC_FIELDS))


def shift_frames(frames, delta, blocks):
    """int64 copy of `frames` plus `delta`, clipped to the graded block of each frame.

    A shift never moves a frame across a block edge: a frame inside block [first, last] stays
    in it. A frame outside every block keeps its value.
    """
    frames = np.asarray(frames, dtype=np.int64)
    out = frames.copy()
    if int(delta) == 0:
        return out
    for _, first, last in blocks:
        inside = (frames >= first) & (frames <= last)
        out[inside] = np.clip(frames[inside] + int(delta), first, last)
    return out


def shift_events(events, delta, blocks):
    """The event table with every frame moved by `delta` (DELTA_SWITCH), still sorted by frame.

    With delta 0 the function returns the same table. A stable sort keeps the order of rows
    that end on one frame.
    """
    if events is None or len(events) == 0 or int(delta) == 0:
        return events
    out = events.copy()
    out["frame"] = shift_frames(out["frame"].to_numpy(), delta, blocks)
    return out.sort_values("frame", kind="mergesort").reset_index(drop=True)


def shift_onsets(onsets, delta, blocks):
    """The onset table with every frame moved by `delta` (DELTA_DEAD). Delta 0 returns it."""
    if onsets is None or len(onsets) == 0 or int(delta) == 0:
        return onsets
    out = onsets.copy()
    out["frame"] = shift_frames(out["frame"].to_numpy(), delta, blocks)
    return out.sort_values("frame", kind="mergesort").reset_index(drop=True)


def shift_restarts(candidates, cfg, blocks):
    """The restart table with each frame moved by the offset of its kind, sorted again.

    A `kick` row moves by DELTA_RESTART_KICK and a `reappear` row by DELTA_RESTART_REAPPEAR.
    The table keeps the order of deadball.restarts(): frame, then kind. With both offsets at 0
    the function returns the same table.
    """
    kick, reappear = int(cfg.DELTA_RESTART_KICK), int(cfg.DELTA_RESTART_REAPPEAR)
    if candidates is None or len(candidates) == 0 or (kick == 0 and reappear == 0):
        return candidates
    out = candidates.copy()
    frames = out["frame"].to_numpy(dtype=np.int64)
    kinds = out["kind"].to_numpy(dtype=object)
    moved = frames.copy()
    for kind, delta in (("kick", kick), ("reappear", reappear)):
        rows = kinds == kind
        moved[rows] = shift_frames(frames[rows], delta, blocks)
    out["frame"] = moved
    return out.sort_values(["frame", "kind"], kind="mergesort").reset_index(drop=True)


def shift_stoppage(stoppage, restart_list, onsets, candidates, cfg, blocks, kickoffs):
    """(stoppage, restart_list) with the edges of each stoppage moved by the offset of its kind.

    dead_intervals() chooses the entries and the exits on the frames before any shift. This
    function then moves an entry at an onset row (not collective_stop) by DELTA_DEAD and an
    exit at a confirmed restart candidate by the DELTA_RESTART_* of its kind. So an offset moves
    the time of a stoppage edge only. It never changes which candidate ends the stoppage (the
    RESTART_ARM_S test stays on the detected frames). A kickoff edge never moves. An edge stays
    inside its graded block. With all three offsets at 0 the inputs come back unchanged.
    """
    deltas = {"kick": int(cfg.DELTA_RESTART_KICK), "reappear": int(cfg.DELTA_RESTART_REAPPEAR)}
    if int(cfg.DELTA_DEAD) == 0 and not any(deltas.values()):
        return stoppage, restart_list
    entries = set(onsets.loc[onsets["mechanism"] != "collective_stop", "frame"].astype(int))
    exit_delta = {}
    confirmed = candidates[candidates["confirmed"]]
    for frame, kind in zip(confirmed["frame"], confirmed["kind"]):
        exit_delta.setdefault(int(frame), deltas.get(kind, 0))   # first row wins, as in M1
    kick_frames = set(int(f) for f in kickoffs.values())
    out = np.zeros_like(np.asarray(stoppage, dtype=bool))
    moved = {}
    for _, first, last in blocks:
        starts, stops = _runs(np.asarray(stoppage[first:last + 1], dtype=bool))
        for start, stop in zip(starts + first, stops + first):
            start, stop = int(start), int(stop)
            if start in entries and start not in kick_frames:
                start = int(np.clip(start + int(cfg.DELTA_DEAD), first, last))
            if stop in exit_delta and stop not in kick_frames and stop <= last:
                new = int(np.clip(stop + exit_delta[stop], first, last + 1))
                moved[stop] = new
                stop = new
            if stop > start:
                out[start:stop] = True
    shifted = [(moved.get(int(f), int(f)), t) for f, t in restart_list]
    return out, shifted


def flight_assignment(events, owner, stoppage, blocks, kinds=TEAM_EVENT_KINDS):
    """Flights of a release to the other team: (flights, blocked), both lists of tuples.

    `events` is the event table, `owner` the owner_mask() of its rows, `stoppage` the bool (N,)
    stoppage mask of dead_intervals(), `kinds` the RULE_TEAM_KINDS of the rule.

    A flight starts at a release row by team T that sets the owner. The next event is the first
    later row of `kinds` or of kind contested, inside the same graded block (P2 card):
      * if the next event sets the owner and its team U is not T, the flight is
        (release frame, T, U, next event frame), and U owns the ball from the release
        (PLAN 1.3, R-F9: a pass that is intercepted belongs to the interceptor from the kick),
      * a next event of T, a contested next event, or a next event of U that does not set the
        owner (a touch, an unconfirmed deflection) gives no flight,
      * a stoppage frame between the release and the next event gives no flight. The ball went
        dead first, and the assignment stops at the onset (PLAN 1.3: 17380, where the next
        touch is inside D). These go to `blocked` as (release frame, T, first stoppage frame).
    flights: [(release_frame, T, U, end_frame)]. Offline lookahead, deterministic (PLAN 3.7).
    """
    flights, blocked = [], []
    if events is None or len(events) == 0:
        return flights, blocked
    frames = events["frame"].to_numpy(dtype=np.int64)
    kind = events["kind"].to_numpy(dtype=object)
    team = events["team"].to_numpy(dtype=object)
    owner = np.asarray(owner, dtype=bool)
    stoppage = np.asarray(stoppage, dtype=bool)
    stream = np.flatnonzero(np.isin(kind, list(kinds) + ["contested"]))
    dead = np.concatenate(([0], np.cumsum(stoppage.astype(np.int64))))
    for idx in np.flatnonzero(owner & (kind == "release")):
        start, kicker = int(frames[idx]), team[idx]
        block = next(((f, l) for _, f, l in blocks if f <= start <= l), None)
        later = stream[frames[stream] > start]
        if block is None or kicker not in TEAM_INDEX or later.size == 0:
            continue
        j = int(later[0])
        frame = int(frames[j])
        if frame > block[1]:
            continue
        if dead[frame + 1] - dead[start] > 0:
            blocked.append((start, kicker, start + int(np.argmax(stoppage[start:frame + 1]))))
            continue
        if owner[j] and team[j] in TEAM_INDEX and team[j] != kicker:
            flights.append((start, kicker, team[j], frame))
    return flights, blocked


def apply_flights(rows, flights):
    """[(frame, team)] with the release row (frame, T) of each flight given to its team U.

    The result is sorted. The release frame stays where it is. Only the team changes, so the
    owner, the switch costs and the owner walk all give the flight to U from the release.
    """
    change = {(int(f), t): u for f, t, u, _ in flights}
    return sorted((int(f), change.get((int(f), t), t)) for f, t in rows)


def dv3_config(cfg):
    """`cfg` with CONTACT_DETECTOR "dv3" (the P1-A touch rows), or `cfg` itself if it has it."""
    import dataclasses

    if cfg.CONTACT_DETECTOR == "dv3":
        return cfg
    return dataclasses.replace(cfg, CONTACT_DETECTOR="dv3")


def prepare(cfg=None, game=None):
    """Detection inputs of run(), as a dict. They do not depend on SEMANTIC_FIELDS.

    Steps: tracks.prepare(), deadball.dead_evidence(), events.detect_events(),
    deadball.dead_onsets(), deadball.restarts(), deadball.ghost_mask(),
    tracks.mark_nearest() (the kickoff team, under "mark"), keeper_holds() (the C13
    goalkeeper holds, under "holds") and keeper_hold_rows() (under "hold_rows"). Pass the dict to
    run(cfg, inputs=...) to decode many semantic variants without a new detection.
    `game` None means games.GAME. The dict keeps the resolved name under "game".
    """
    from engine import deadball, events, tracks
    from engine.config import Config

    cfg = cfg or Config.default()
    game, n_frames, blocks, kickoffs, pitch = match_inputs(game)
    tactical = engine_tactical(cfg, game)
    ball, feats, slot_conf = tracks.prepare(tactical, cfg)
    tagging = data.load_tagging(game)

    e_dead = deadball.dead_evidence(ball, feats, tagging, blocks, cfg, kickoffs=kickoffs,
                                pitch=pitch)
    table = events.detect_events(ball, feats, tactical, slot_conf, cfg, e_dead=e_dead)
    onsets = deadball.dead_onsets(ball, feats, cfg, pitch=pitch, events=table)
    candidates = deadball.restarts(ball, feats, table, tagging, cfg, onsets=onsets,
                               pitch=pitch)
    ghost = deadball.ghost_mask(ball, feats, cfg)
    mark = tracks.mark_nearest(tactical, pitch)
    keepers = None
    if cfg.GK_HOLD_IS_PLAY or cfg.GK_HOLD_OWNER:
        keepers = keeper_positions(tactical, blocks, pitch, cfg)
    holds = keeper_holds(tactical, ball, feats, blocks, pitch, cfg, keepers)
    hold_rows = (keeper_hold_rows(ball, feats, slot_conf, keepers, pitch, cfg)
                 if keepers is not None else None)
    return {"key": detection_key(cfg), "game": game, "n_frames": n_frames, "blocks": blocks,
            "kickoffs": kickoffs, "fps": float(ball["fps"]), "ball": ball, "feats": feats,
            "tagging": tagging, "e_dead": e_dead, "events": table, "onsets": onsets,
            "restarts": candidates, "ghost": ghost, "mark": mark, "holds": holds,
            "hold_rows": hold_rows}


def run(cfg=None, return_parts=False, inputs=None, game=None):
    """M3 labels for the whole tactical video (cards P1-C, P2). The submission path.

    Steps:
      1. prepare(): tracks, dead-ball evidence, events, onsets, restart candidates. `inputs`
         is an earlier prepare() result. Its detection fields must equal those of `cfg`.
      2. The P2 offsets move the event frames of each kind: the event table by DELTA_SWITCH,
         the onsets by DELTA_DEAD, the restart candidates by DELTA_RESTART_KICK or
         DELTA_RESTART_REAPPEAR (shift_events, shift_onsets, shift_restarts). The kickoff
         frames do not move.
      3. dead_intervals() of M1 gives the stoppage mask and the restarts with a team, on the
         frames before the shift. With ONSET_KNEE, a gap-plus-stop dwell entry starts at the
         players' deceleration knee plus DELTA_KNEE (knee_retimer()). With
         KNEE_MIN_PRE_SPELL_S > 0, an entry that cuts its preceding spell below that length
         moves later (spell_floors()) and steps 3 to 6 run once more (COMPONENT_KNEE.md 8). shift_stoppage() then moves the stoppage edges by the
         offsets, so an offset never changes which restart candidate ends a stoppage.
         kickoff_decisions() gives the team of each kickoff of games.py: the centre mark rule,
         else the restart candidates before the shift, else Law 8 (the other half's team
         flipped). goal_kickoffs() finds the kickoff after each GOALS window and gives it the
         centre mark team where the rule gives one (apply_goal_kickoffs).
      4. add_keeper_rows() adds the goalkeeper controls of the holds that stay alive (C13b).
         The owner events of POSSESSION_RULE (owner_events). The event rows of RULE_TEAM_KINDS
         give the switch costs. The rows of release_rows() (no bare touch) can release the
         owner in the owner walk. With FLIGHT_ASSIGNMENT on, a release to the
         other team gives the ball to that team from the release (flight_assignment).
      5. The event owner (owner_track), the delayed nearest player (nearest_log_odds), the Smart
         Tagging prior (tag_log_odds) and P(D) (dead_probability) give the log evidence
         (emissions).
      6. decode_blocks() runs the first-order Viterbi decoder and the minimum-duration post-pass
         on each graded block. The switch costs come from the team events, all onsets, and the
         confirmed restart candidates plus the kickoffs.
      7. With BACKDATE_CHANGES (K2), backdate_changes() moves each direct H <-> A change of
         the decoded labels back over the frames where the new team was the nearer team and
         within touch range of a visible ball (calibration/COMPONENT_BACKDATE.md).
      8. X outside the graded blocks. The writer refuses X on a graded frame.
      9. With LEARNED_MODE "hybrid", engine.learned keeps the D and X frames of step 8 and
         gives each alive run the team of the learned ensemble. With "model", run() skips
         steps 1 to 8 and returns the labels of the ensemble alone (engine.learned.apply).
         With LEARNED_STRUCTURE the model decoder forces D before each half's kickoff, the
         rule of steps 3 and 6 (engine.learned.pre_kickoff_mask). With LEARNED_PRIOR_BONUS > 0
         run() runs steps 1 to 8 in mode "model" too, and the model decoder adds the bonus to
         the log-probability of the class of step 8's label on each H, A or D frame (the
         labels "off" writes, after back-dating; engine.learned.apply_prior).
         Both modes read the features from engine_tactical(), the ATD with the C9 guard.
         With return_parts=True the model mode runs steps 1 to 8 too, and parts keeps the
         rule-engine labels under "c22_labels" (the key name is historical: they are the
         labels of the current rule engine). "off" (the default) loads nothing learned.
    With return_parts=True also returns a dict of the intermediate arrays and tables.
    `game` None means games.GAME, or the game of `inputs` when `inputs` is given.
    """
    from engine.config import Config

    cfg = cfg or Config.default()
    mode = learned_mode(cfg)
    if (mode == "model" and not return_parts and inputs is None
            and not getattr(cfg, "LEARNED_PRIOR_BONUS", 0.0)):
        # Candidate M needs no c22 label, so run() skips the detection and the decoder. With
        # LEARNED_PRIOR_BONUS > 0 it needs the labels of steps 1 to 8 and runs them.
        from engine import learned
        game, _, blocks, _, _ = match_inputs(game)
        return learned.apply(cfg, game, blocks, tactical=engine_tactical(cfg, game))
    if inputs is None:
        inputs = prepare(cfg, game)
    elif inputs["key"] != detection_key(cfg):
        raise ValueError("inputs come from a config with other detection fields")
    elif game is not None and inputs.get("game") != data.resolve_game(game):
        raise ValueError("inputs come from game {!r}, not {!r}".format(inputs.get("game"), game))
    n_frames, blocks, kickoffs = inputs["n_frames"], inputs["blocks"], inputs["kickoffs"]
    feats, tagging, e_dead = inputs["feats"], inputs["tagging"], inputs["e_dead"]

    onsets = shift_onsets(inputs["onsets"], cfg.DELTA_DEAD, blocks)
    candidates = shift_restarts(inputs["restarts"], cfg, blocks)
    knee, knee_rows = None, []
    # Synthetic inputs without the ball (tests/test_calibration.py) skip the knee rule.
    if cfg.ONSET_KNEE and "ball" in inputs:
        knee = knee_retimer(inputs["ball"], feats, inputs["events"], cfg, inputs["fps"],
                        knee_rows)

    def decode(knee):
        """(labels, parts) of steps 3 to 6 with the `knee` hook of dead_intervals()."""
        stoppage, restart_list = dead_intervals(n_frames, inputs["onsets"], e_dead,
                                            inputs["restarts"], blocks, kickoffs, cfg,
                                            inputs["fps"], inputs.get("holds"), knee)
        table = add_keeper_rows(inputs["events"], inputs.get("hold_rows"), stoppage)
        table = shift_events(table, cfg.DELTA_SWITCH, blocks)
        stoppage, restart_list = shift_stoppage(stoppage, restart_list, inputs["onsets"],
                                            inputs["restarts"], cfg, blocks, kickoffs)
        mark = inputs.get("mark")
        kick_rules = kickoff_decisions(kickoffs, inputs["restarts"], cfg, mark)
        kick_teams = {half: row[0] for half, row in kick_rules.items()}
        goal_rows = goal_kickoffs(tagging, stoppage, restart_list, blocks, cfg, mark)
        restart_list, _ = apply_goal_kickoffs(restart_list, [], goal_rows)

        kinds = RULE_TEAM_KINDS[cfg.POSSESSION_RULE]
        owners = owner_events(table, cfg)
        team_rows = team_event_rows(table, kinds=kinds)
        maybe_rows = release_rows(table, cfg)
        flights, blocked = [], []
        if cfg.FLIGHT_ASSIGNMENT:
            flights, blocked = flight_assignment(table, owner_mask(table, cfg), stoppage, blocks,
                                             kinds)
            owners = apply_flights(owners, flights)
            team_rows = apply_flights(team_rows, flights)
            maybe_rows = apply_flights(maybe_rows, flights)
        owner = owner_track(owners, restart_list, maybe_rows, blocks, n_frames, kick_teams,
                        kickoffs)
        near_lo = nearest_log_odds(feats, inputs["ghost"], cfg)
        tag_lo = tag_log_odds(tagging, n_frames, cfg)
        edges = np.flatnonzero(np.diff(stoppage.astype(np.int8)) != 0) + 1
        unknown_lo = stretch_mean(near_lo + tag_lo, owner == UNKNOWN,
                              np.concatenate([[f for f, _ in team_rows], edges]))
        p_dead = dead_probability(e_dead, stoppage, cfg)
        log_e = emissions(owner, unknown_lo, tag_lo, p_dead, cfg)

        confirmed = candidates[candidates["confirmed"]]
        restart_events = [(int(f), t if t in TEAM_INDEX else None)
                          for f, t in zip(confirmed["frame"], confirmed["team"])]
        restart_events += [(int(kickoffs[h]), kick_teams[h]) for h in sorted(kickoffs)]
        restart_events.sort(key=lambda item: (item[0], str(item[1])))
        _, restart_events = apply_goal_kickoffs([], restart_events, goal_rows)
        labels = decode_blocks(log_e, blocks, kickoffs, team_rows, onsets["frame"].to_numpy(),
                           restart_events, cfg, n_frames)
        return labels, {"e_dead": e_dead, "events": table, "onsets": onsets,
                        "restarts": candidates, "stoppage": stoppage,
                        "restart_list": restart_list, "owner_events": owners, "owner": owner,
                        "near_lo": near_lo, "tag_lo": tag_lo, "unknown_lo": unknown_lo,
                        "kick_teams": kick_teams, "kick_rules": kick_rules,
                        "goal_kickoffs": goal_rows, "p_dead": p_dead,
                        "log_e": log_e, "restart_events": restart_events,
                        "flights": flights, "flights_blocked": blocked}

    labels, parts = decode(knee)
    if cfg.ONSET_KNEE and cfg.KNEE_MIN_PRE_SPELL_S > 0:
        floors = spell_floors(labels, knee_rows,
                              int(round(float(cfg.KNEE_MIN_PRE_SPELL_S) * inputs["fps"])))
        if floors:
            knee_rows = []
            knee = knee_retimer(inputs["ball"], feats, inputs["events"], cfg, inputs["fps"],
                                knee_rows, floors)
            labels, parts = decode(knee)
    backdate_rows = []
    # Synthetic inputs without the ball (tests/test_calibration.py) skip the K2 rule.
    if cfg.BACKDATE_CHANGES and "ball" in inputs:
        labels = backdate_changes(labels, feats["dH"], feats["dA"], inputs["ball"]["visible"],
                                  blocks, cfg, inputs["fps"], backdate_rows)
    if mode != "off":
        from engine import learned
        parts["c22_labels"] = labels
        labels, parts["learned_probs"] = learned.apply(
            cfg, inputs["game"], blocks, labels, return_probs=True,
            tactical=engine_tactical(cfg, inputs["game"]))
    if return_parts:
        parts["knee"] = knee_rows
        parts["backdate"] = backdate_rows
        return labels, parts
    return labels


def learned_mode(cfg):
    """cfg.LEARNED_MODE, checked. "off" loads nothing learned (engine.learned.check_mode)."""
    if cfg.LEARNED_MODE == "off":
        return "off"
    from engine import learned
    return learned.check_mode(cfg)
