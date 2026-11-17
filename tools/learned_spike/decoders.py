"""The decoders and the cost choice of the learned stage, numpy only. Not part of the submission.

This module implements the decoding steps of LEARNED_STAGE_PROTOCOL.md (Sections 1 to 3, with
the advisor changes of 9 November 2026). It has no torch dependency, so engine/ can copy it.

Probabilities:

1. ensemble_p5() averages the 5 fps class probabilities (H, A, D) over every model and every
   symmetric copy, in float64, and rounds the mean once to ROUND_DECIMALS (4) decimals.
2. upsample() interpolates the 5 fps values linearly onto every 25 fps frame (bin centres).

Decoders:

- Candidate M, decode_model(): a three-state Viterbi (H, A, D) in each graded block, with a
  cost for a change between H and A and a cost for a change between alive and dead. Frames
  outside the graded blocks are X.
- Candidate H, decode_hybrid(): the alive and dead frames of c22, unchanged. On each alive run
  of c22 (H or A), a two-state Viterbi over the H and A probabilities, renormalised over H and
  A, with the H-A cost, gives the team.

Both decoders take the log of the probabilities clipped at LOG_FLOOR (1e-4). They decode a whole
grid of costs in one pass (viterbi_segments()). For each cost the path is the same as the path
of tools/learned_spike/model.viterbi(), including the choice on ties (the lowest state).

The cost grid of candidate M is GRID_HA x GRID_AD without the pairs whose H-A cost is above
MAX_HA_PER_AD (2) times the alive-dead cost (LEARNED_STAGE_PROTOCOL_R2.md Section 1, change B):
a direct H-A change is then never dearer than a pass through D. That leaves 32 of the 42 pairs.
The H-A grid of candidate H (GRID_HA) does not change. choose() takes the highest mean spellF1
over the validation matches. On a tie it takes the lowest H-A cost, then the lowest alive-dead
cost.

Python 3.9.
"""

import numpy as np

ROUND_DECIMALS = 4
LOG_FLOOR = 1e-4
BIN = 5                       # features.BIN: 25 fps -> 5 fps
GRID_HA = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
GRID_AD = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
MAX_HA_PER_AD = 2.0           # R2 change B: keep only H-A <= 2 x alive-dead in M's grid
H, A, D, X = 0, 1, 2, 3       # possession_metrics codes


def ensemble_p5(prob_list):
    """float64 (bins, 3): the mean of the (bins, 3) arrays, rounded to ROUND_DECIMALS."""
    total = np.zeros_like(np.asarray(prob_list[0], dtype=np.float64))
    for p in prob_list:
        total += np.asarray(p, dtype=np.float64)
    return np.round(total / len(prob_list), ROUND_DECIMALS)


def upsample(probs, n_frames, bin_size=BIN):
    """Linear interpolation of bin probabilities (bin centres) onto every frame."""
    centres = np.arange(probs.shape[0]) * bin_size + bin_size // 2
    frames = np.arange(n_frames)
    out = np.empty((n_frames, probs.shape[1]))
    for k in range(probs.shape[1]):
        out[:, k] = np.interp(frames, centres, probs[:, k])
    return out


def viterbi_segments(logp, trans, segments):
    """int8 (C, N) best paths for C transition matrices, -1 outside the segments.

    logp: (N, K) log-probabilities. trans: (C, K, K) costs, trans[c, i, j] for a change from
    state i to state j. segments: (first, end) pairs, end exclusive, in increasing order and
    without overlap. Each segment is decoded alone.
    """
    logp = np.asarray(logp, dtype=np.float64)
    trans = np.asarray(trans, dtype=np.float64)
    n, k = logp.shape
    c = trans.shape[0]
    paths = np.full((c, n), -1, dtype=np.int8)
    cols = np.arange(c)
    for first, end in segments:
        if end <= first:
            continue
        m = end - first
        back = np.empty((m, c, k), dtype=np.int8)
        score = np.repeat(logp[first][None], c, axis=0)
        for t in range(1, m):
            cand = score[:, :, None] - trans
            arg = cand.argmax(axis=1)
            back[t] = arg
            score = np.take_along_axis(cand, arg[:, None, :], axis=1)[:, 0, :] + logp[first + t]
        state = score.argmax(axis=1).astype(np.int8)
        paths[:, end - 1] = state
        for t in range(m - 1, 0, -1):
            state = back[t, cols, state]
            paths[:, first + t - 1] = state
    return paths


def block_segments(blocks):
    """(first, end) pairs of graded blocks [(half, first, last_inclusive)]."""
    return [(int(a), int(b) + 1) for _, a, b in blocks]


def runs_of(mask):
    """(first, end) pairs of the runs of True in a boolean array."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def model_trans(costs):
    """(C, 3, 3) transition costs of the three-state decoder for [(cost_ha, cost_ad)]."""
    out = np.empty((len(costs), 3, 3))
    for i, (cha, cad) in enumerate(costs):
        out[i] = [[0.0, cha, cad], [cha, 0.0, cad], [cad, cad, 0.0]]
    return out


def decode_model(frame_probs, blocks, costs):
    """int8 (C, N) codes of candidate M for each (cost_ha, cost_ad) in `costs`."""
    logp = np.log(np.clip(frame_probs, LOG_FLOOR, 1.0))
    paths = viterbi_segments(logp, model_trans(costs), block_segments(blocks))
    paths[paths < 0] = X
    return paths


def team_logp(frame_probs):
    """(N, 2) log-probabilities of H and A renormalised over H and A, clipped at LOG_FLOOR."""
    team = frame_probs[:, :2] / np.maximum(frame_probs[:, :2].sum(axis=1, keepdims=True), 1e-9)
    return np.log(np.clip(team, LOG_FLOOR, 1.0))


def decode_hybrid(frame_probs, c22_codes, costs_ha):
    """int8 (C, N) codes of candidate H for each H-A cost in `costs_ha`."""
    base = np.asarray(c22_codes, dtype=np.int8)
    trans = np.array([[[0.0, c], [c, 0.0]] for c in costs_ha], dtype=np.float64)
    alive = np.isin(base, (H, A))
    paths = viterbi_segments(team_logp(frame_probs), trans, runs_of(alive))
    out = np.repeat(base[None], len(costs_ha), axis=0)
    out[:, alive] = paths[:, alive]
    return out


def full_model_grid():
    """[(cost_ha, cost_ad)] of the round 1 grid (42 pairs), H-A major."""
    return [(cha, cad) for cha in GRID_HA for cad in GRID_AD]


def model_grid():
    """[(cost_ha, cost_ad)] of M's round 2 grid, H-A major: H-A <= MAX_HA_PER_AD x alive-dead."""
    return [(cha, cad) for cha, cad in full_model_grid() if cha <= MAX_HA_PER_AD * cad]


def choose(rows):
    """The row with the highest "mean"; on a tie the lowest "cost_ha", then "cost_ad"."""
    return max(rows, key=lambda r: (r["mean"], -r["cost_ha"], -r.get("cost_ad", 0.0)))
