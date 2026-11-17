"""First-order Viterbi decoder over the states H, A and D (PLAN.md 3.7, card P1-C).

    STATES                                    ("H", "A", "D"), the state index is the position
    near_mask(frames, n, width)               bool (n,): frames within +-width of any frame
    switch_costs(n, cfg, team_events, onsets, restarts)
                                              float64 (n, 3, 3): cost of i -> j into frame t
    viterbi(log_e, costs, forced=None)        int8 (n,): the best state path
    min_durations(states, log_e, costs, cfg, forced=None)
                                              int8 (n,): the path after the minimum durations
    decode(log_e, costs, cfg, forced=None)    viterbi() and then min_durations()

The decoder works on one graded block at a time. The caller gives the per-frame log evidence
`log_e` (n, 3) and the switch costs. A cost is in nats and the decoder subtracts it from the
path score. A switch is cheap near an event and expensive elsewhere:

  * H <-> A costs C_SWITCH. Within SWITCH_EVENT_WINDOW frames of an event of the new team it
    costs C_SWITCH_EVENT.
  * alive -> D costs C_DEAD. Within DEAD_EVENT_WINDOW frames of an onset it costs C_DEAD_EVENT.
  * D -> alive costs C_ALIVE. Within ALIVE_EVENT_WINDOW frames of a restart it costs
    C_ALIVE_EVENT, and RESTART_TEAM_BONUS less for the team of that restart. The result can be
    below zero. The path has no cycles in time, so a negative cost is a bonus and nothing more.

Ties have a fixed order. A frame keeps the state of the frame before it if that is as good as a
switch. After that the order is H, then A, then D. The post-pass merges a possession shorter than
MIN_POSS frames and a D segment shorter than MIN_DEAD frames into the neighbour that costs less.

numpy only. Deterministic. The module reads no match data and never reads the ground truth.
"""

import numpy as np

STATES = ("H", "A", "D")
H, A, D = 0, 1, 2
ALIVE = (H, A)
N_STATES = len(STATES)
NO_FORCE = -1

# For each destination state j: the order in which predecessors win a tie. Staying comes first,
# then H, A, D (card P1-C: "prefer staying, then H").
_ORDER = np.array([[j] + [k for k in range(N_STATES) if k != j] for j in range(N_STATES)],
                  dtype=np.int64)
_COLS = np.repeat(np.arange(N_STATES)[:, None], N_STATES, axis=1)
_ROWS = np.arange(N_STATES)


def near_mask(frames, n, width):
    """bool (n,): True on every frame within +-width of a frame in `frames` (clipped to 0..n-1)."""
    frames = np.asarray(frames, dtype=np.int64).ravel()
    diff = np.zeros(int(n) + 1, dtype=np.int64)
    if frames.size:
        lo = np.clip(frames - int(width), 0, n)
        hi = np.clip(frames + int(width) + 1, 0, n)
        keep = hi > lo
        np.add.at(diff, lo[keep], 1)
        np.add.at(diff, hi[keep], -1)
    return np.cumsum(diff[:n]) > 0


def switch_costs(n, cfg, team_events=None, onsets=None, restarts=None):
    """float64 (n, 3, 3): cost[t, i, j] of a switch from state i at t - 1 to state j at t.

    team_events  {"H": frames, "A": frames}: frames of events of each team (local indices).
    onsets       frames of alive -> D onsets (local indices).
    restarts     [(frame, team or None)]: D -> alive restarts (local indices).
    The diagonal is 0. Row 0 is unused by viterbi() (frame 0 has no predecessor).
    """
    team_events = team_events or {}
    costs = np.zeros((int(n), N_STATES, N_STATES), dtype=np.float64)
    for team, src, dst in (("A", H, A), ("H", A, H)):
        near = near_mask(team_events.get(team, ()), n, cfg.SWITCH_EVENT_WINDOW)
        costs[:, src, dst] = np.where(near, cfg.C_SWITCH_EVENT, cfg.C_SWITCH)
    near_onset = near_mask(onsets if onsets is not None else (), n, cfg.DEAD_EVENT_WINDOW)
    to_dead = np.where(near_onset, cfg.C_DEAD_EVENT, cfg.C_DEAD)
    costs[:, H, D] = to_dead
    costs[:, A, D] = to_dead
    restarts = list(restarts or ())
    frames = [f for f, _ in restarts]
    near_restart = near_mask(frames, n, cfg.ALIVE_EVENT_WINDOW)
    to_alive = np.where(near_restart, cfg.C_ALIVE_EVENT, cfg.C_ALIVE)
    for team, dst in (("H", H), ("A", A)):
        own = near_mask([f for f, t in restarts if t == team], n, cfg.ALIVE_EVENT_WINDOW)
        costs[:, D, dst] = to_alive - np.where(own, cfg.RESTART_TEAM_BONUS, 0.0)
    return costs


def _apply_forced(log_e, forced):
    """Copy of log_e with -inf on every state other than the forced one of a forced frame."""
    log_e = np.array(log_e, dtype=np.float64)
    if forced is None:
        return log_e
    forced = np.asarray(forced, dtype=np.int64)
    rows = np.flatnonzero(forced != NO_FORCE)
    if rows.size:
        keep = log_e[rows, forced[rows]]
        log_e[rows, :] = -np.inf
        log_e[rows, forced[rows]] = keep
    return log_e


def viterbi(log_e, costs, forced=None):
    """int8 (n,): the state path with the highest score sum(log_e) - sum(costs).

    `log_e` is (n, 3) log evidence for H, A, D. `costs` is the (n, 3, 3) output of
    switch_costs(). `forced` is an optional int (n,) array: a state index where the frame must
    take that state, NO_FORCE elsewhere. Exact dynamic programming with the tie order of the
    module docstring. The loop runs over frames and is vectorised over the 3 states.
    """
    log_e = _apply_forced(log_e, forced)
    n = log_e.shape[0]
    path = np.zeros(n, dtype=np.int8)
    if n == 0:
        return path
    pointer = np.zeros((n, N_STATES), dtype=np.int8)
    score = log_e[0].copy()
    for t in range(1, n):
        cand = score[:, None] - costs[t]            # cand[i, j]: from i at t - 1 to j at t
        ordered = cand[_ORDER, _COLS]               # ordered[j, p]: the p-th predecessor of j
        pos = np.argmax(ordered, axis=1)            # the first maximum wins the tie
        pointer[t] = _ORDER[_ROWS, pos]
        score = ordered[_ROWS, pos] + log_e[t]
    path[-1] = int(np.argmax(score))                # tie order H, A, D at the last frame
    for t in range(n - 1, 0, -1):
        path[t - 1] = pointer[t, path[t]]
    return path


def _runs(states):
    """(starts, stops, labels) of the maximal constant runs of an int array (half-open)."""
    states = np.asarray(states)
    if states.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    breaks = np.flatnonzero(states[1:] != states[:-1]) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [states.size]))
    return starts, stops, states[starts]


def _relabel_loss(log_e, costs, states, start, stop, label):
    """Score lost if frames [start, stop) take `label` instead of their state (lower is better)."""
    old = int(states[start])
    loss = float(log_e[start:stop, old].sum() - log_e[start:stop, label].sum())
    n = states.shape[0]
    if start > 0:
        left = int(states[start - 1])
        loss += float(costs[start, left, label] - costs[start, left, old])
    if stop < n:
        right = int(states[stop])
        loss += float(costs[stop, label, right] - costs[stop, old, right])
    return loss


def min_durations(states, log_e, costs, cfg, forced=None):
    """int8 (n,): `states` after the minimum-duration post-pass (PLAN 3.7).

    A run of H or A shorter than MIN_POSS frames, or a run of D shorter than MIN_DEAD frames, is
    short. The pass takes the shortest short run (the earliest one on a tie) and gives it the
    label of the neighbour run whose label loses the least score: the log evidence of the run
    plus the switch costs at its two edges. On a tie the left neighbour wins. A run that holds a
    forced frame is never changed. The pass repeats until no short run is left that it can
    change. It is deterministic.
    """
    states = np.array(states, dtype=np.int8)
    log_e = _apply_forced(log_e, forced)
    forced = (np.full(states.shape[0], NO_FORCE, dtype=np.int64) if forced is None
              else np.asarray(forced, dtype=np.int64))
    minimum = {H: int(cfg.MIN_POSS), A: int(cfg.MIN_POSS), D: int(cfg.MIN_DEAD)}
    frozen = set()
    while True:
        starts, stops, labels = _runs(states)
        short = []
        for idx, (start, stop, label) in enumerate(zip(starts, stops, labels)):
            start, stop = int(start), int(stop)
            if (stop - start) >= minimum[int(label)] or start in frozen:
                continue
            if np.any(forced[start:stop] != NO_FORCE):
                continue
            short.append((stop - start, start, idx))
        if not short:
            return states
        _, start, idx = min(short)
        stop = int(stops[idx])
        options = []
        if idx > 0:
            options.append(int(labels[idx - 1]))
        if idx + 1 < len(labels):
            options.append(int(labels[idx + 1]))
        if not options:
            frozen.add(start)                       # a block with one run: nothing to merge into
            continue
        losses = [_relabel_loss(log_e, costs, states, start, stop, lab) for lab in options]
        best = options[int(np.argmin(losses))]      # the first minimum: the left neighbour
        states[start:stop] = best


def decode(log_e, costs, cfg, forced=None):
    """int8 (n,): viterbi() followed by min_durations(), for one graded block."""
    path = viterbi(log_e, costs, forced)
    return min_durations(path, log_e, costs, cfg, forced)
