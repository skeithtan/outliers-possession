"""Round 8 arm DK: study decoders D1, D2, D3 and the kickoff rule K(a). Not part of the submission.

Protocol R8 Section 3 rows D and K. Every decoder reads the frame probabilities E2's decoder
reads (N, 3) and returns int8 codes 0 H, 1 A, 2 D, 3 X (X outside the graded blocks), with the
kickoff rule of E2 (frames before each block's kickoff forced to D, engine.learned.
pre_kickoff_mask). At their neutral setting each decoder gives engine.learned.decode_model's
path (D1 f = 1, D2 s = 0, D3 none: checked by dk_lopo.py --self-check).

- D1 (decode_d1): the alive -> D and D -> alive cost into frame t is cost_ad x f when the ball
  is not in the tracking at frame t, cost_ad elsewhere.
- D2 (decode_d2): the dead state remembers the team before it (two dead chains, each with a
  run-length counter up to T frames). Returning from D to the same team after a dead run of
  fewer than T frames costs cost_ad + s; every other D -> alive change costs cost_ad. A block
  that opens in D has no team before it (no surcharge).
- D3 (decode_d3): pass 1 is E2's decode; pass 2 keeps its D frames and decides the team again
  on each alive run with team-change cost c (engine.learned.decode_hybrid on pass 1's codes).
- D4 (geometric_pool, protocol R8 note 6.1): the members pooled per class as
  network^0.75 x member^0.25 (each clipped at POOL_FLOOR first), renormalised over H, A, D,
  then E2's decoder at (12, 12).
- K(a) (kickoff_forced): from each half's kickoff frame the first KICK_FLOOR frames are forced
  to the kickoff team in the Viterbi (half 1: the team of E2's first alive frame at or after
  the kickoff; half 2: the other team, Law 8); after the floor the decoder decides, so the
  forced team lasts until the decoder's first change of team or dead ball.

Ties: the lowest predecessor index wins, as in engine.learned.viterbi(). numpy only.
"""
import numpy as np

H, A, D, X = 0, 1, 2, 3
FPS = 25
KICK_FLOOR_S = 1.0              # protocol R8 arm K: floor of the forced kickoff team, fixed 16 Nov
GRID_D1 = (2.0, 4.0)
GRID_D2 = tuple((t, s) for t in (1.0, 2.0, 3.0) for s in (4.0, 8.0))
GRID_D3 = (8.0, 10.0, 14.0, 16.0)
POOL_FLOOR = 1e-6               # protocol R8 note 6.1: floor before the powers of D4
W_PF = 0.25


def geometric_pool(p_net, p_pf, w_pf=W_PF, floor=POOL_FLOOR, decimals=None):
    """float64 (N, 3): p_net^(1 - w) x p_pf^w per class, renormalised; rounded if asked."""
    a = np.clip(np.asarray(p_net, np.float64), floor, 1.0)
    b = np.clip(np.asarray(p_pf, np.float64), floor, 1.0)
    g = np.exp((1.0 - w_pf) * np.log(a) + w_pf * np.log(b))
    g = g / g.sum(axis=1, keepdims=True)
    return np.round(g, decimals) if decimals is not None else g


def _logp(frame_probs, forced_dead):
    from engine import learned
    logp = learned.log_emissions(frame_probs)
    if forced_dead is not None:
        fd = np.asarray(forced_dead, dtype=bool)
        logp[fd, H] = -np.inf
        logp[fd, A] = -np.inf
    return logp


def viterbi_tv(logp, cost_ha, cost_ad_t):
    """engine.learned.viterbi with a per-frame alive <-> D cost cost_ad_t[t] (into frame t)."""
    n = logp.shape[0]
    back = np.empty((n, 3), dtype=np.int8)
    score = logp[0].copy()
    trans = np.array([[0.0, cost_ha, 0.0], [cost_ha, 0.0, 0.0], [0.0, 0.0, 0.0]])
    cols = (0, 1, 2)
    for t in range(1, n):
        c = cost_ad_t[t]
        trans[0, 2] = trans[1, 2] = trans[2, 0] = trans[2, 1] = c
        cand = score[:, None] - trans
        arg = cand.argmax(axis=0)
        back[t] = arg
        score = cand[arg, cols] + logp[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_d1(frame_probs, blocks, cost_ha, cost_ad, forced_dead, ball_in_view, f):
    logp = _logp(frame_probs, forced_dead)
    cad = np.where(np.asarray(ball_in_view, dtype=bool), cost_ad, cost_ad * float(f))
    out = np.full(frame_probs.shape[0], X, dtype=np.int8)
    for _, first, last in blocks:
        out[first:last + 1] = viterbi_tv(logp[first:last + 1], cost_ha, cad[first:last + 1])
    return out


def viterbi_d2(logp, cost_ha, cost_ad, short_frames, surcharge):
    """Path of the remembering decoder on one block (codes 0 H, 1 A, 2 D).

    States: H, A, then per team k in (H, A) a chain of dead states with run length 1 .. T - 1
    (index j = length - 1) and a long state (length >= T). D -> same team from a chain state
    costs cost_ad + surcharge.
    """
    n = logp.shape[0]
    T = int(short_frames)
    m = T - 1                                          # chain states of a short run
    ninf = -np.inf
    # scores
    sH, sA = logp[0, H], logp[0, A]
    chH = np.full(m, ninf)
    chA = np.full(m, ninf)
    lgH = lgA = logp[0, D]
    # back pointers: for H and A, the predecessor code: 0 H, 1 A, 2 chainH (with index),
    # 3 longH, 4 chainA, 5 longA; for the long states, 0 from its own long, 1 from chain end.
    bH = np.zeros((n, 2), dtype=np.int32)
    bA = np.zeros((n, 2), dtype=np.int32)
    bLH = np.zeros(n, dtype=np.int8)
    bLA = np.zeros(n, dtype=np.int8)
    for t in range(1, n):
        e = logp[t]
        iH = int(np.argmax(chH)) if m else 0
        iA = int(np.argmax(chA)) if m else 0
        mH = chH[iH] if m else ninf
        mA = chA[iA] if m else ninf
        # into H: H, A, chainH (+s), longH, chainA, longA ; first maximum wins
        cands = (sH, sA - cost_ha, mH - cost_ad - surcharge, lgH - cost_ad,
                 mA - cost_ad, lgA - cost_ad)
        kH = int(np.argmax(cands))
        nH = cands[kH] + e[H]
        bH[t] = (kH, iH if kH == 2 else iA)
        cands = (sH - cost_ha, sA, mH - cost_ad, lgH - cost_ad,
                 mA - cost_ad - surcharge, lgA - cost_ad)
        kA = int(np.argmax(cands))
        nA = cands[kA] + e[A]
        bA[t] = (kA, iH if kA == 2 else iA)
        # long dead states: stay, or from the last chain state
        endH = chH[-1] if m else sH - cost_ad
        endA = chA[-1] if m else sA - cost_ad
        if lgH >= endH:
            nLH, bLH[t] = lgH, 0
        else:
            nLH, bLH[t] = endH, 1
        if lgA >= endA:
            nLA, bLA[t] = lgA, 0
        else:
            nLA, bLA[t] = endA, 1
        if m:
            chH = np.concatenate(([sH - cost_ad], chH[:-1]))
            chA = np.concatenate(([sA - cost_ad], chA[:-1]))
        chH = chH + e[D]
        chA = chA + e[D]
        sH, sA, lgH, lgA = nH, nA, nLH + e[D], nLA + e[D]
    # final state, order H, A, chainH..., longH, chainA..., longA
    finals = [sH, sA] + list(chH) + [lgH] + list(chA) + [lgA]
    k = int(np.argmax(finals))
    if k == 0:
        st = ("H", 0)
    elif k == 1:
        st = ("A", 0)
    elif k < 2 + m:
        st = ("cH", k - 2)
    elif k == 2 + m:
        st = ("LH", 0)
    elif k < 3 + 2 * m:
        st = ("cA", k - 3 - m)
    else:
        st = ("LA", 0)
    path = np.empty(n, dtype=np.int8)
    names = {0: ("H", 0), 1: ("A", 0), 3: ("LH", 0), 5: ("LA", 0)}
    for t in range(n - 1, -1, -1):
        kind, j = st
        path[t] = H if kind == "H" else A if kind == "A" else D
        if t == 0:
            break
        if kind in ("H", "A"):
            code, idx = (bH if kind == "H" else bA)[t]
            if code == 2:
                st = ("cH", int(idx))
            elif code == 4:
                st = ("cA", int(idx))
            else:
                st = names[int(code)]
        elif kind in ("cH", "cA"):
            if j == 0:
                st = ("H", 0) if kind == "cH" else ("A", 0)
            else:
                st = (kind, j - 1)
        else:                                          # long
            src = (bLH if kind == "LH" else bLA)[t]
            team = "H" if kind == "LH" else "A"
            if src == 0:
                st = (kind, 0)
            elif m:
                st = ("c" + team, m - 1)
            else:
                st = (team, 0)
    return path


def decode_d2(frame_probs, blocks, cost_ha, cost_ad, forced_dead, t_s, s):
    logp = _logp(frame_probs, forced_dead)
    out = np.full(frame_probs.shape[0], X, dtype=np.int8)
    T = int(round(float(t_s) * FPS))
    for _, first, last in blocks:
        out[first:last + 1] = viterbi_d2(logp[first:last + 1], cost_ha, cost_ad, T, float(s))
    return out


def decode_stock(frame_probs, blocks, cost_ha, cost_ad, forced_dead):
    from engine import learned
    return learned.decode_model(frame_probs, blocks, cost_ha, cost_ad, forced_dead)


def decode_d3(frame_probs, blocks, cost_ha, cost_ad, forced_dead, c, pass1=None):
    from engine import learned
    if pass1 is None:
        pass1 = decode_stock(frame_probs, blocks, cost_ha, cost_ad, forced_dead)
    return learned.decode_hybrid(frame_probs, pass1, float(c))


# -- K(a) ---------------------------------------------------------------------------------------

def kickoff_teams_a(stock_codes, kickoffs, blocks):
    """{half: team code}: half 1's team is E2's first alive code at or after its kickoff frame,
    half 2's team the other one (Law 8). {} when half 1 has no alive frame in its block."""
    k1 = kickoffs.get(1)
    blk = next(((f, l) for h, f, l in blocks if h == 1), None)
    if k1 is None or blk is None:
        return {}
    seg = np.asarray(stock_codes[k1:blk[1] + 1])
    alive = np.flatnonzero(seg < D)
    if alive.size == 0:
        return {}
    t1 = int(seg[alive[0]])
    return {1: t1, 2: 1 - t1}


def kickoff_forced(n, blocks, kickoffs, teams, floor_frames):
    """int (n,) forced team per frame (-1 elsewhere): [kick, kick + floor) of each half."""
    forced = np.full(int(n), -1, dtype=np.int64)
    for half, first, last in blocks:
        kick = kickoffs.get(half)
        if kick is None or half not in teams or not (first <= kick <= last):
            continue
        forced[kick:min(kick + int(floor_frames), last + 1)] = teams[half]
    return forced


def decode_k(frame_probs, blocks, cost_ha, cost_ad, forced_dead, kickoffs, stock_codes=None,
             floor_s=KICK_FLOOR_S):
    """E2's decode with the kickoff team forced on the floor after each half's kickoff."""
    from engine import learned
    n = frame_probs.shape[0]
    if stock_codes is None:
        stock_codes = decode_stock(frame_probs, blocks, cost_ha, cost_ad, forced_dead)
    teams = kickoff_teams_a(stock_codes, kickoffs, blocks)
    ft = kickoff_forced(n, blocks, kickoffs, teams, int(round(floor_s * FPS)))
    logp = _logp(frame_probs, forced_dead)
    rows = np.flatnonzero(ft >= 0)
    keep = logp[rows, ft[rows]]
    logp[rows, :] = -np.inf
    logp[rows, ft[rows]] = keep
    out = np.full(n, X, dtype=np.int8)
    for _, first, last in blocks:
        out[first:last + 1] = learned.viterbi(logp[first:last + 1], cost_ha, cost_ad)
    return out, teams
