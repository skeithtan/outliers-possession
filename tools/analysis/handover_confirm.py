"""Hand-over confirmation measurements for card P9-B. The script measures. It designs no rule.

    python tools/analysis/handover_confirm.py collect --game NAME [--ref REFERENCE.csv]
        [--engine-csv ENGINE.csv] [--off-csv REDIRECT_OFF.csv] [--public-gt PUBLIC_GT.csv]
        [--cache-dir DIR] [--view raw|smooth] [--config CONFIG.json] --out OUT.json
    python tools/analysis/handover_confirm.py report OUT.json [OUT.json ...]

`collect` runs the engine (possession.prepare and possession.run with the default Config, or the
overrides of `--config` through engine.config.load) on one match and stores per-contact summary
numbers in a JSON file. `report` prints the tables of docs/plan/calibration/P9B_measurements.md
from the JSON files. Set MTAH_DATA_DIR for each match
as tools/score_game.py needs it. `--cache-dir` points the engine's npz cache somewhere else (the
smoothed view uses its own cache). Write the JSON files outside the repository. Never run
`collect` on a sealed match of calibration/SPLITS.md.

Definitions.

- Graded frame. A frame of a graded block of games.py (engine.data.graded_blocks).
- Reference hand-over. A frame t inside a graded block where the reference label changes from H
  to A or from A to H, with t - 1 in the same block. The old team is the label at t - 1, the new
  team the label at t. A change through `D` or `X` is no hand-over.
- Candidate contact. A row of events._bump_contacts() of the default engine (kinds kick,
  receive, redirect, contested), on graded frames. The frame c is the bump centre. h is
  CONTACT_HALF_WINDOW. Under REACH_SHAPE_CONFIRM (C18) the rows include the reach contacts
  (d1 above R_TOUCH), built with the shape array as the engine builds them. A reach contact
  has the key `reach` in its record and keeps its team in every attribution column (cat_cur,
  cat_centre, cat_inst), because the attribution alternatives do not apply to it. The wide list
  leaves out the maxima that are reach contacts. With REACH_SHAPE_CONFIRM false the output has
  no `reach` key and equals the output before C18. The dv3 rows (events._touch_rows() with CONTACT_DETECTOR "dv3", kinds
  touch, release, contested) are the comparison candidates.
- Matching. Each reference hand-over takes the candidate within MATCH_TOL frames with the
  smallest |c - t|. The matching is one to one and greedy by that difference. Ties go to the
  earlier contact, then to the earlier hand-over. A matched contact is a hand-over contact (HO).
  A contact with no reference hand-over within MATCH_TOL frames is a non-hand-over contact
  (NHO). A contact within MATCH_TOL of a hand-over that another contact took is in neither
  class. NHO-other is the NHO subset whose contact team is not the reference label at c (a touch
  of the team without the ball that did not change the reference owner).
- Old and new team of a contact. HO: the old and new team of its hand-over. NHO: the old team is
  the reference label at c, the new team the other team. A contact whose reference label at c
  is not H or A is in no class.
- Team forward velocity. For team T and frame t: the mean over the visible T players of the
  player x velocity (tracks._difference_velocity, metres per second), times +1 when T defends
  x = 0 in that block and -1 when T defends x = length (tracks.goalkeeper_slots own_goal_x: the
  median x of the team's players over the block). Then a centred mean over +-SHAPE_HALF frames
  (NaN-aware). NaN outside the graded blocks.
- Shape response. R_T(L) = vf_T(c + L) - vf_T(c). The feature at lag L is
  R_new(L) - R_old(L). A contact with a reference `D` frame in [c, c + CLEAR_S] is left out of
  the shape tables (the players stop at a stoppage whatever happened at the contact).
- Separation. AUC = P(feature of a random HO > feature of a random NHO), ties count one half
  (Mann-Whitney from average ranks). "Above median" is the share of HO features above the
  median of the NHO features of the same match.
- Stream. The uncontested candidate contacts and the control runs (events._control_rows) in
  frame order, a contact before a control run on one frame: the items of the C8 redirect
  confirmation (events._redirect_kinds). The next item of a contact is the stream item after it.
  "Clean" means no frame of the engine's dead-ball mask (e_dead > EVENT_SUPPRESS_DEAD) lies
  between the two items.
- Rule decision. The re-walk of events._redirect_kinds gives each uncontested contact the stream
  owner before it. Kinds by that owner: kick_own, receive_own, redirect_own (a `touch` row),
  kick_other, receive_other, redirect_confirmed (a `control` row), redirect_dropped (no row),
  and kick_none, receive_none when a dead frame has reset the owner (the rule then treats a
  redirect like one of the other team).
  Under REDIRECT_SHAPE_CONFIRM (C15) the re-walk and the row rebuild take the shape array of
  events.match_team_shape(), as the engine does, and a redirect that only the shape check
  (events.shape_confirms) confirms is redirect_confirmed with the clause `shape`.
  The re-walk is checked against events._redirect_kinds on every run. A dropped redirect fails
  "dead_between" (a dead frame before the next item), "next_other_team" (the next item is by
  another team) or "no_next".
- Audit truth. A contact is a true hand-over when a reference hand-over to the contact's team
  lies within MATCH_TOL frames of c. The redirect class is redirect_dropped among the contacts
  of the other team (kick_other, receive_other, redirect_confirmed, redirect_dropped). Precision
  of the redirect class = share of redirect_dropped that is not a true hand-over. Recall = share
  of the other-team contacts that are not true hand-overs and that the rule drops.
- Fraction. |ds| / |J| of a contact, ds = speed(c + h) - speed(c - h), J the impulse
  (events.impulse). The rule's redirect is |ds| < REDIRECT_FRACTION |J|.
- dv3 deflection test (F1 rank 3). At t = c - DV_K // 2: heading_change_k(t) <
  DEFLECTION_MAX_HEADING_DEG and |dv_signed_k(t)| / speed(t) < DEFLECTION_MAX_SPEED_FRAC (the
  rule of events._touch_rows). The bump variant uses the heading between v(c - h) and v(c + h)
  and |ds| / speed(c - h).
- Splice (measurement 3). The frames where the engine output and the REDIRECT_FRACTION = 0 run
  (`--off-csv`) differ form stretches. A stretch belongs to the dropped redirect nearest its
  first frame within [-SPLICE_BACK, +SPLICE_FWD] frames. The splice puts the off labels into the
  stretches of one class (true hand-over, not a hand-over, no dropped redirect) and scores it.
- Nobody near (measurement 4). For a reference hand-over at t: the smallest distance of a
  new-team player to the visible ball over [t - h, t + h]. "No ball" when the ball is not visible
  on any frame of the window.
- Attribution instants (measurement 5). For a contact: at_H and at_A, the frames of the smallest
  H and A distance over [c - h, c + h] (the rule of _bump_contacts). The separation is
  |at_H - at_A| where both teams are visible. Single-instant attributions use both distances
  at one frame: the bump centre c, or the frame of the overall nearest player. The category of
  an attribution is none (nearest above R_TOUCH), contested, H or A. "Rows" re-runs
  events._redirect_kinds on the contacts with the other attribution and compares the final row
  (release, control, touch, contested or none, with its team). The owner after a contact is the
  reference label at c + h + 1.
- Airborne. A bump receive with ball speed above AIRBORNE_SPEED (Config) at c. Its accuracy is
  the share whose team equals the reference owner after the contact.

The script stores and prints summary numbers per contact (distances, frames, speeds), never a
tracking row. `--public-gt` (BAR-ATH only) reads the public ground truth through tools/ for the
four public flips of PLAN 1.3 only. numpy and pandas only. Deterministic.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (ROOT, os.path.join(ROOT, "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

TEAMS = ("H", "A")
OTHER = {"H": "A", "A": "H"}
MATCH_TOL = 25             # frames, the card's window (1 s at 25 fps)
LAGS_S = (0.5, 1.0, 1.5, 2.0)                       # the card's lags
LAG_GRID_S = tuple(0.25 * i for i in range(1, 13))  # 0.25 to 3.0 s, to find the largest separation
SHAPE_HALF = 6             # frames, centred mean of the team forward velocity (about 0.25 s)
CLEAR_S = 2.0              # seconds after c without a reference D for the shape tables
RHYTHM_S = (1.0, 2.0, 3.0)
RADII = (1.5, 2.0, 2.5, 3.0)
SPLICE_BACK = 50           # frames before a stretch start where its dropped redirect may lie
SPLICE_FWD = 5             # frames after it
FRACTION_EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)
PUBLIC_FLIPS = (11351, 11553, 13785, 15766)         # PLAN 1.3: no new-team player within 1.5 m
CATEGORIES = ("none", "contested", "H", "A")


# -- small helpers --------------------------------------------------------------------------------

def finite_or_inf(values):
    """float64 copy with NaN replaced by +inf (a missing player is far away)."""
    values = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(values), values, np.inf)


def read_labels(path, n_frames=None):
    """The `label` column of a `frame,label` CSV as an array of one-letter strings."""
    table = pd.read_csv(path, dtype={"frame": np.int64, "label": str})
    labels = table.sort_values("frame")["label"].to_numpy().astype("<U1")
    if n_frames is not None and labels.size != n_frames:
        raise ValueError("{}: {} rows, expected {}".format(path, labels.size, n_frames))
    return labels


def labels_md5(labels):
    """md5 of the labels written as engine.submission writes them (frame,label rows)."""
    text = "frame,label\n" + "".join("{},{}\n".format(i, v) for i, v in enumerate(labels))
    return hashlib.md5(text.encode("ascii")).hexdigest()


def graded_mask(blocks, n_frames):
    """bool (n_frames,): frames of the graded blocks."""
    mask = np.zeros(int(n_frames), dtype=bool)
    for _, first, last in blocks:
        mask[int(first):int(last) + 1] = True
    return mask


def block_index(blocks, n_frames):
    """int64 (n_frames,): index of the graded block of each frame, -1 outside."""
    out = np.full(int(n_frames), -1, dtype=np.int64)
    for i, (_, first, last) in enumerate(blocks):
        out[int(first):int(last) + 1] = i
    return out


def centred_nanmean(values, half):
    """float64 (N,): mean of the finite values over [t - half, t + half], NaN if none."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    csum = np.concatenate(([0.0], np.cumsum(np.where(finite, values, 0.0))))
    ccount = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    n = values.size
    lo = np.clip(np.arange(n) - int(half), 0, n)
    hi = np.clip(np.arange(n) + int(half) + 1, 0, n)
    count = ccount[hi] - ccount[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (csum[hi] - csum[lo]) / count
    return np.where(count > 0, out, np.nan)


def quantiles(values, qs=(10, 50, 90)):
    """{n, p10, p50, p90} of the finite values (n only when empty)."""
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    values = values[np.isfinite(values)]
    out = {"n": int(values.size)}
    for q in qs:
        out["p{}".format(q)] = float(np.percentile(values, q)) if values.size else None
    return out


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(pos == neg) over the finite values, None if a class is empty."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return None
    ranks = pd.Series(np.concatenate((pos, neg))).rank(method="average").to_numpy()
    n1 = float(pos.size)
    return float((ranks[:pos.size].sum() - n1 * (n1 + 1.0) / 2.0) / (n1 * neg.size))


def share_above(pos, ref_values):
    """Share of the finite `pos` above the median of the finite `ref_values`."""
    pos = np.asarray(pos, dtype=np.float64)
    ref_values = np.asarray(ref_values, dtype=np.float64)
    pos, ref_values = pos[np.isfinite(pos)], ref_values[np.isfinite(ref_values)]
    if pos.size == 0 or ref_values.size == 0:
        return None
    return float(np.mean(pos > np.median(ref_values)))


# -- reference hand-overs and the matching --------------------------------------------------------

def reference_handovers(labels, blocks):
    """[(t, old, new)]: the H <-> A changes of `labels` inside each graded block."""
    labels = np.asarray(labels)
    out = []
    for _, first, last in blocks:
        seg = labels[int(first):int(last) + 1]
        if seg.size < 2:
            continue
        prev, cur = seg[:-1], seg[1:]
        change = np.isin(prev, TEAMS) & np.isin(cur, TEAMS) & (prev != cur)
        for i in np.flatnonzero(change):
            out.append((int(first) + int(i) + 1, str(prev[i]), str(cur[i])))
    return out


def match_handovers(ho_frames, contact_frames, tol=MATCH_TOL):
    """[contact index or -1] for each hand-over: one to one, greedy by |c - t| within `tol`.

    The pair with the smallest difference goes first. Ties go to the earlier contact, then to
    the earlier hand-over.
    """
    contacts = np.asarray(contact_frames, dtype=np.int64)
    order = np.argsort(contacts, kind="stable")
    sorted_c = contacts[order]
    pairs = []
    for i, t in enumerate(ho_frames):
        lo = int(np.searchsorted(sorted_c, int(t) - tol, side="left"))
        hi = int(np.searchsorted(sorted_c, int(t) + tol, side="right"))
        for k in range(lo, hi):
            j = int(order[k])
            pairs.append((abs(int(contacts[j]) - int(t)), int(contacts[j]), int(t), i, j))
    out, used = [-1] * len(ho_frames), set()
    for _, _, _, i, j in sorted(pairs):
        if out[i] == -1 and j not in used:
            out[i] = j
            used.add(j)
    return out


def near_handover(frames, ho_frames, tol=MATCH_TOL):
    """bool per frame: a hand-over lies within `tol` frames."""
    ho = np.sort(np.asarray(ho_frames, dtype=np.int64))
    out = []
    for f in frames:
        lo = int(np.searchsorted(ho, int(f) - tol, side="left"))
        hi = int(np.searchsorted(ho, int(f) + tol, side="right"))
        out.append(hi > lo)
    return out


# -- team shape -----------------------------------------------------------------------------------

def team_forward_velocity(pvx, team_of_slot, own_goal_x, blocks, half=SHAPE_HALF):
    """{"H", "A": float64 (N,)}: forward velocity of each team (module docstring).

    `pvx` is (N, S) player x velocity in m/s, NaN where not visible. `own_goal_x` maps
    (block index, team code) to 0.0 or the pitch length (any value > 0 means "defends x =
    length").
    """
    pvx = np.asarray(pvx, dtype=np.float64)
    team_of_slot = np.asarray(team_of_slot)
    n = pvx.shape[0]
    blk = block_index(blocks, n)
    out = {}
    for code, letter in ((0, "H"), (1, "A")):
        cols = np.flatnonzero(team_of_slot == code)
        sub = pvx[:, cols]
        finite = np.isfinite(sub)
        count = finite.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(finite, sub, 0.0).sum(axis=1) / count
        mean = np.where(count > 0, mean, np.nan)
        sign = np.full(n, np.nan)
        for b in range(len(blocks)):
            goal = own_goal_x.get((b, code))
            if goal is None:
                continue
            sign[blk == b] = 1.0 if float(goal) == 0.0 else -1.0
        out[letter] = centred_nanmean(mean * sign, half)
    return out


def shape_response(vf, frame, lags):
    """{"H", "A": [R_T(L) for L in lags]}: vf_T(frame + L) - vf_T(frame), NaN off the array."""
    out = {}
    for team in TEAMS:
        series = vf[team]
        n = series.shape[0]
        base = series[frame] if 0 <= frame < n else np.nan
        vals = []
        for lag in lags:
            f = int(frame) + int(lag)
            vals.append(float(series[f] - base) if 0 <= f < n else float("nan"))
        out[team] = vals
    return out


def shape_feature(resp, new, old, index):
    """R_new(L) - R_old(L) at the lag of `index`."""
    return resp[new][index] - resp[old][index]


# -- contacts, stream and rule walk ---------------------------------------------------------------

def category(d_h, d_a, cfg):
    """none, contested, H or A for one pair of team distances (the rule of _bump_contacts)."""
    team = "H" if d_h <= d_a else "A"
    d1, d2 = (d_h, d_a) if team == "H" else (d_a, d_h)
    if not d1 <= cfg.R_TOUCH:
        return "none"
    if d2 <= cfg.R_TOUCH and (d2 - d1) < cfg.CONTESTED_MARGIN_M:
        return "contested"
    return team


def engine_contacts(ball, feats, alive, cfg, shape_all):
    """(contacts, shape, reach frames): the bump contacts as events._contact_rows builds them.

    `shape_all` is events.match_team_shape() or None. The contacts read it under
    REACH_SHAPE_CONFIRM (a reach contact, C18). The redirect rule reads `shape` under
    REDIRECT_SHAPE_CONFIRM (C15), else None. The reach frames are the contacts with a distance
    above R_TOUCH.
    """
    from engine import events

    shape = shape_all if cfg.REDIRECT_SHAPE_CONFIRM else None
    contacts = events._bump_contacts(ball, feats, None, alive, cfg,
                                     shape=shape_all if cfg.REACH_SHAPE_CONFIRM else None)
    return contacts, shape, {int(c[0]) for c in contacts if not c[4] <= cfg.R_TOUCH}


def build_stream(contacts, control_rows):
    """[(frame, order, team, kind, contact index or -1)] in the order of _redirect_kinds."""
    stream = [(int(c[0]), 0, c[2], c[1], i) for i, c in enumerate(contacts)
              if c[1] != "contested"]
    stream += [(int(r[0]), 1, r[2], "control", -1) for r in control_rows]
    stream.sort(key=lambda item: (item[0], item[1]))
    return stream


def walk_rule(contacts, control_rows, alive, shape=None, cfg=None):
    """(info per contact index, stream): the re-walk of events._redirect_kinds.

    info[i] = {"owner": stream owner before the contact, "decision": rule decision,
    "clause": failing clause of a dropped redirect, "shape" for a redirect that only the C15
    shape check confirms, else None, "pos": stream position}. `shape` and `cfg` as in
    events._redirect_kinds (None: the C8 rule without the shape check).
    """
    from engine import events

    dead = np.concatenate(([0], np.cumsum(~np.asarray(alive, dtype=bool))))

    def clean(a, b):
        return dead[b + 1] - dead[a] == 0

    stream = build_stream(contacts, control_rows)
    info = {}
    owner, last = None, None
    for pos, (frame, _, team, kind, idx) in enumerate(stream):
        if last is not None and not clean(last, frame):
            owner = None
        last = frame
        before = owner
        decision, clause = None, None
        if kind != "redirect":
            if idx >= 0:
                side = "none" if before is None else ("own" if before == team else "other")
                decision = "{}_{}".format(kind, side)
            owner = team
        elif owner == team:
            decision = "redirect_own"
        else:
            nxt = stream[pos + 1] if pos + 1 < len(stream) else None
            if nxt is not None and nxt[2] == team and clean(frame, nxt[0]):
                decision = "redirect_confirmed"
                owner = team
            elif events.shape_confirms(shape, frame, team, dead, cfg):
                decision, clause = "redirect_confirmed", "shape"
                owner = team
            else:
                decision = "redirect_dropped"
                if nxt is None:
                    clause = "no_next"
                elif not clean(frame, nxt[0]):
                    clause = "dead_between"
                else:
                    clause = "next_other_team"
        if idx >= 0:
            info[idx] = {"owner": before, "decision": decision, "clause": clause, "pos": pos}
    return info, stream


def rows_after_rule(contacts, control_rows, alive, shape=None, cfg=None):
    """{frame: (row kind, team)} of the bump rows that _contact_rows would give the contacts.

    `shape` and `cfg` go to events._redirect_kinds, as the engine passes them.
    """
    from engine import events

    kinds = events._redirect_kinds(contacts, control_rows, alive, shape, cfg)
    out = {}
    for frame, kind, team, _, _ in contacts:
        if kind == "contested":
            out[int(frame)] = ("contested", None)
        elif kind == "kick":
            out[int(frame)] = ("release", team)
        elif kind == "receive":
            out[int(frame)] = ("control", team)
        elif frame in kinds:
            out[int(frame)] = (kinds[frame], team)
        else:
            out[int(frame)] = ("none", team)
    return out


def next_items(stream, pos, alive_cum, count=2):
    """[(frame, team, clean)] of the `count` stream items after position `pos`."""
    frame = stream[pos][0]
    out = []
    for item in stream[pos + 1:pos + 1 + count]:
        clean = alive_cum[item[0] + 1] - alive_cum[frame] == 0
        out.append((int(item[0]), item[2], bool(clean)))
    return out


def next_after_frame(stream_frames, stream, frame, alive_cum, count=2):
    """[(frame, team, clean)] of the `count` stream items with a frame above `frame`."""
    pos = int(np.searchsorted(stream_frames, int(frame), side="right"))
    out = []
    for item in stream[pos:pos + count]:
        clean = alive_cum[item[0] + 1] - alive_cum[int(frame)] == 0
        out.append((int(item[0]), item[2], bool(clean)))
    return out


def window_min(values, lo, hi, mask=None):
    """Smallest finite value over [lo, hi) (optionally where `mask`), NaN if none."""
    seg = np.asarray(values[lo:hi], dtype=np.float64)
    if mask is not None:
        seg = np.where(mask[lo:hi], seg, np.nan)
    seg = seg[np.isfinite(seg)]
    return float(seg.min()) if seg.size else float("nan")


# -- collect --------------------------------------------------------------------------------------

def _f(value):
    """JSON-friendly float (None for NaN or inf)."""
    value = float(value)
    return value if np.isfinite(value) else None


def collect(args):
    """The `collect` command: per-contact summary numbers for one match."""
    import games
    from engine import data, events, possession, tracks
    from engine.config import Config, load as load_config

    if args.cache_dir:
        data.DEFAULT_CACHE_DIR = os.path.abspath(os.path.expanduser(args.cache_dir))
    game = args.game
    cfg = load_config(args.config) if getattr(args, "config", None) else Config.default()
    inputs = possession.prepare(cfg, game)
    labels = np.asarray(possession.run(cfg, inputs=inputs))
    n = labels.size
    blocks = inputs["blocks"]
    fps = float(inputs["fps"])
    graded = graded_mask(blocks, n)
    ball, feats = inputs["ball"], inputs["feats"]
    e_dead = np.asarray(inputs["e_dead"], dtype=np.float64)
    alive = ~(np.isfinite(e_dead) & (e_dead > cfg.EVENT_SUPPRESS_DEAD))
    alive_cum = np.concatenate(([0], np.cumsum(~alive)))
    h = int(cfg.CONTACT_HALF_WINDOW)
    d_h, d_a = finite_or_inf(feats["dH"]), finite_or_inf(feats["dA"])
    dist = {"H": d_h, "A": d_a}
    visible = np.asarray(ball["visible"], dtype=bool)
    speed = np.asarray(ball["speed"], dtype=np.float64)
    vx = np.asarray(ball["vx"], dtype=np.float64)
    vy = np.asarray(ball["vy"], dtype=np.float64)
    heading_k = np.asarray(ball["heading_change_k"], dtype=np.float64)
    dv_signed = np.asarray(ball["dv_signed_k"], dtype=np.float64)
    airborne = np.asarray(feats["airborne_suspect"], dtype=bool)
    size = events.impulse(ball, h)

    controls = events._control_rows(ball, feats, None, alive, cfg)
    # The shape array of C15 and C18, as engine/events.detect_events builds it.
    wants_shape = (cfg.CONTACT_DETECTOR == "bump"
                   and (cfg.REDIRECT_SHAPE_CONFIRM or cfg.REACH_SHAPE_CONFIRM))
    shape_all = (events.match_team_shape(possession.engine_tactical(cfg, game), cfg)
                 if wants_shape else None)
    contacts, shape, reach_frames = engine_contacts(ball, feats, alive, cfg, shape_all)
    info, stream = walk_rule(contacts, controls, alive, shape, cfg)
    stream_frames = np.asarray([s[0] for s in stream], dtype=np.int64)
    # Check 1: the re-walk equals events._redirect_kinds.
    official = events._redirect_kinds(contacts, controls, alive, shape, cfg)
    rewalk = {}
    for i, c in enumerate(contacts):
        dec = info.get(i, {}).get("decision")
        if dec == "redirect_own":
            rewalk[int(c[0])] = "touch"
        elif dec == "redirect_confirmed":
            rewalk[int(c[0])] = "control"
    if rewalk != {int(k): v for k, v in official.items()}:
        raise RuntimeError("re-walk differs from events._redirect_kinds")
    # Check 2: the bump rows of the engine table equal the contacts after the rule.
    table = inputs["events"]
    bump_kinds = table[table["kind"].isin(("release", "touch", "contested"))
                       | ((table["kind"] == "control") & table["notes"].isin(("receive",
                                                                               "redirect")))]
    expected = sorted((f, k, t) for f, (k, t) in rows_after_rule(contacts, controls, alive,
                                                                 shape, cfg).items()
                      if k != "none")
    got = sorted((int(f), str(k), None if t is None else str(t))
                 for f, k, t in zip(bump_kinds["frame"], bump_kinds["kind"], bump_kinds["team"]))
    if expected != got:
        raise RuntimeError("contact rows differ from the engine event table")

    # dv3 rows (comparison candidates).
    cfg3 = possession.dv3_config(cfg)
    dv3_rows = [(int(r[0]), str(r[1]), r[2]) for r in events._touch_rows(ball, feats, None,
                                                                           alive, cfg3)]
    stream3 = [(f, 0, t, k, -1) for f, k, t in dv3_rows if k != "contested"]
    stream3 += [(int(r[0]), 1, r[2], "control", -1) for r in controls]
    stream3.sort(key=lambda item: (item[0], item[1]))
    stream3_frames = np.asarray([s[0] for s in stream3], dtype=np.int64)

    # Team forward velocity.
    tactical = possession.engine_tactical(cfg, game)
    pitch = data.pitch_m(game)
    px, py, pvis = tracks._player_metres(tactical, pitch)
    pvx, _ = tracks._difference_velocity(px, py, pvis, fps)
    slots = tracks.goalkeeper_slots(tactical, blocks, pitch, cfg)
    own_goal = {}
    for b, (half, _, _) in enumerate(blocks):
        for code in (0, 1):
            if (int(half), code) in slots:
                own_goal[(b, code)] = float(slots[(int(half), code)]["own_goal_x"])
    vf = team_forward_velocity(pvx, tactical["team_of_slot"], own_goal, blocks)
    del px, py, pvis, pvx
    lag_frames = [int(round(s * fps)) for s in LAG_GRID_S]

    ref = None
    hos = []
    if args.ref:
        ref = read_labels(args.ref, int(games.GAMES[game]["n_video_frames"]))
        hos = reference_handovers(ref, blocks)
    ho_frames = [t for t, _, _ in hos]

    # Per-contact records.
    recs = []
    alt_cats = []
    k3 = int(ball.get("k", cfg.DV_K))
    for i, (c, kind, team, _, d1) in enumerate(contacts):
        c = int(c)
        lo, hi = max(0, c - h), min(n, c + h + 1)
        at_h = lo + int(np.argmin(d_h[lo:hi]))
        at_a = lo + int(np.argmin(d_a[lo:hi]))
        reach = c in reach_frames       # a C18 contact: nobody within R_TOUCH over the window
        cat_cur = category(d_h[at_h], d_a[at_a], cfg)
        cur_expected = "none" if reach else ("contested" if kind == "contested" else team)
        if cat_cur != cur_expected:
            raise RuntimeError("attribution re-computation differs at {}".format(c))
        cat_centre = category(d_h[c], d_a[c], cfg)
        both = np.minimum(d_h[lo:hi], d_a[lo:hi])
        t_star = lo + int(np.argmin(both))
        cat_inst = category(d_h[t_star], d_a[t_star], cfg)
        if reach:
            cat_cur = cat_centre = cat_inst = team
        s_in, s_out = speed[c - h], speed[c + h]
        ds = s_out - s_in
        split = cfg.REDIRECT_FRACTION * size[c]
        phys = "kick" if ds >= split else ("receive" if ds <= -split else "redirect")
        if kind != "contested" and phys != kind:
            raise RuntimeError("kind re-computation differs at {}".format(c))
        alt_cats.append((c, phys, cat_centre, cat_inst))
        frac = abs(ds) / size[c] if size[c] > 0 else float("nan")
        v_in = np.array([vx[c - h], vy[c - h]])
        v_out = np.array([vx[c + h], vy[c + h]])
        with np.errstate(invalid="ignore"):
            cosang = float(np.dot(v_in, v_out) / (np.hypot(*v_in) * np.hypot(*v_out)))
        heading_bump = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))) \
            if np.isfinite(cosang) else float("nan")
        t3 = max(0, c - k3 // 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel3 = abs(dv_signed[t3]) / speed[t3] if speed[t3] > 0 else float("inf")
            rel_bump = abs(ds) / s_in if s_in > 0 else float("inf")
        defl3 = bool(np.isfinite(heading_k[t3]) and heading_k[t3] < cfg.DEFLECTION_MAX_HEADING_DEG
                     and rel3 < cfg.DEFLECTION_MAX_SPEED_FRAC)
        defl_bump = bool(np.isfinite(heading_bump)
                         and heading_bump < cfg.DEFLECTION_MAX_HEADING_DEG
                         and rel_bump < cfg.DEFLECTION_MAX_SPEED_FRAC)
        sep = abs(at_h - at_a) if (np.isfinite(d_h[at_h]) and np.isfinite(d_a[at_a])) else None
        rec = {"c": c, "kind": kind, "phys_kind": phys, "team": team, "d1": _f(d1),
               "dH": _f(d_h[at_h]), "dA": _f(d_a[at_a]), "at_sep": sep,
               "cat_cur": cat_cur, "cat_centre": cat_centre, "cat_inst": cat_inst,
               "J": _f(size[c]), "ds": _f(ds), "frac": _f(frac), "s_in": _f(s_in),
               "s_c": _f(speed[c]), "s_out": _f(s_out), "heading_bump": _f(heading_bump),
               "defl3": defl3, "defl_bump": defl_bump,
               "airborne_c": bool(airborne[c]), "airborne_win": bool(airborne[lo:hi].any()),
               "graded": bool(graded[c])}
        if reach:
            rec["reach"] = True
        rec.update(info.get(i, {"owner": None, "decision": None, "clause": None, "pos": None}))
        if rec["pos"] is not None:
            rec["next"] = next_items(stream, rec["pos"], alive_cum)
        else:
            rec["next"] = next_after_frame(stream_frames, stream, c, alive_cum)
        rec["shape"] = shape_response(vf, c, lag_frames)
        rec["engine_at_c"] = str(labels[c])
        if ref is not None:
            rec["ref_c"] = str(ref[c])
            rec["ref_after"] = str(ref[min(n - 1, c + h + 1)])
            clear_hi = min(n, c + int(round(CLEAR_S * fps)) + 1)
            rec["ref_clear"] = bool(not (ref[c:clear_hi] == "D").any())
            near_to = [t for t, _, new in hos if abs(t - c) <= MATCH_TOL and new == team]
            rec["true_ho"] = bool(near_to)
        recs.append(rec)

    # Attribution alternatives: rows after the rule.
    rows_cur = rows_after_rule(contacts, controls, alive, shape, cfg)
    row_changes = {}
    for col, name in ((2, "centre"), (3, "instant")):
        alt = []
        for item in alt_cats:
            cat = item[col]
            if cat == "none":
                continue
            alt.append((item[0], "contested" if cat == "contested" else item[1],
                        None if cat == "contested" else cat, 0.0, 0.0))
        rows_alt = rows_after_rule(alt, controls, alive, shape, cfg)
        changed = 0
        for f, row in rows_cur.items():
            if not graded[f]:
                continue
            if rows_alt.get(f, ("none", None)) != row:
                changed += 1
        row_changes[name] = {"changed": changed,
                             "n": int(sum(1 for f in rows_cur if graded[f]))}

    # Hand-over matching (bump and dv3) and the hand-over records.
    graded_idx = [i for i, r in enumerate(recs) if r["graded"]]
    match = match_handovers(ho_frames, [recs[i]["c"] for i in graded_idx])
    dv3_graded = [r for r in dv3_rows if graded[r[0]]]
    match3 = match_handovers(ho_frames, [r[0] for r in dv3_graded])
    ho_recs = []
    for k, (t, old, new) in enumerate(hos):
        j = match[k]
        lo, hi = max(0, t - h), min(n, t + h + 1)
        d_new = window_min(dist[new], lo, hi, visible)
        d_new_wide = window_min(dist[new], max(0, t - 40), min(n, t + 7), visible)
        rec = {"t": t, "old": old, "new": new, "contact": None, "dv3": None,
               "ball_in_window": bool(visible[lo:hi].any()), "d_new": _f(d_new),
               "d_new_wide": _f(d_new_wide),
               "shape": shape_response(vf, t, lag_frames),
               "next": next_after_frame(stream_frames, stream, t, alive_cum),
               "clear": bool(not (ref[t:min(n, t + int(round(CLEAR_S * fps)) + 1)]
                                  == "D").any())}
        if j >= 0:
            idx = graded_idx[j]
            recs[idx]["ho"] = k
            rec["contact"] = recs[idx]["c"]
            rec["contact_kind"] = recs[idx]["kind"]
            rec["contact_team"] = recs[idx]["team"]
            rec["contact_decision"] = recs[idx]["decision"]
        if match3[k] >= 0:
            rec["dv3"] = dv3_graded[match3[k]][0]
        ho_recs.append(rec)
    if ref is not None:
        near = near_handover([r["c"] for r in recs], ho_frames)
        for r, is_near in zip(recs, near):
            if "ho" in r:
                r["cls"] = "HO"
            elif not is_near and r["ref_c"] in TEAMS:
                r["cls"] = "NHO"
            else:
                r["cls"] = None

    # dv3 records (shape and rhythm only).
    dv3_recs = []
    if ref is not None:
        matched3 = {}
        for k, j in enumerate(match3):
            if j >= 0:
                matched3[j] = k
        near3 = near_handover([r[0] for r in dv3_graded], ho_frames)
        for j, ((f, kind, team), is_near) in enumerate(zip(dv3_graded, near3)):
            if j in matched3:
                _, old, new = hos[matched3[j]]
                cls = "HO"
            elif not is_near and ref[f] in TEAMS:
                old, new, cls = str(ref[f]), OTHER[str(ref[f])], "NHO"
            else:
                continue
            clear_hi = min(n, f + int(round(CLEAR_S * fps)) + 1)
            dv3_recs.append({"c": f, "kind": kind, "team": team, "cls": cls, "old": old,
                             "new": new, "clear": bool(not (ref[f:clear_hi] == "D").any()),
                             "shape": shape_response(vf, f, lag_frames),
                             "next": next_after_frame(stream3_frames, stream3, f, alive_cum)})

    # Wider radius: impulse maxima whose nearest player lies beyond R_TOUCH, up to the widest
    # radius of RADII (the maxima _bump_contacts drops: not the reach contacts of C18).
    strong = np.isfinite(size) & (size >= cfg.DV_CONTACT) & alive
    wide = []
    i10 = lag_index({"lags_s": list(LAG_GRID_S)}, 1.0)
    for c in events.window_maxima(size, h, np.flatnonzero(strong)):
        c = int(c)
        if not graded[c] or c in reach_frames:
            continue
        lo, hi = max(0, c - h), min(n, c + h + 1)
        dmh, dma = float(d_h[lo:hi].min()), float(d_a[lo:hi].min())
        team = "H" if dmh <= dma else "A"
        d1 = min(dmh, dma)
        if not (cfg.R_TOUCH < d1 <= RADII[-1]):
            continue
        resp = shape_response(vf, c, lag_frames)
        rec = {"c": c, "d1": d1, "d2": _f(max(dmh, dma)), "team": team,
               "engine_at_c": str(labels[c]),
               "shape10": _f(shape_feature(resp, team, OTHER[team], i10)),
               "next": next_after_frame(stream_frames, stream, c, alive_cum)}
        if ref is not None:
            rec["ref_c"] = str(ref[c])
            rec["true_ho"] = bool([t for t, _, new in hos if abs(t - c) <= MATCH_TOL
                                   and new == team])
        wide.append(rec)

    # Engine hand-overs (label-free): nearest new-team distance around each.
    eng_hos = reference_handovers(labels, blocks)
    eng_d = []
    for t, _, new in eng_hos:
        lo, hi = max(0, t - h), min(n, t + h + 1)
        eng_d.append(_f(window_min(dist[new], lo, hi, visible)))

    # Gaps between consecutive clean stream items and between consecutive contacts (graded).
    def gaps_of(frames):
        frames = sorted(int(f) for f in frames if graded[int(f)])
        out = []
        blk = block_index(blocks, n)
        for a, b in zip(frames[:-1], frames[1:]):
            if b > a and blk[a] == blk[b] and alive_cum[b + 1] - alive_cum[a] == 0:
                out.append(b - a)
        return out

    stream_gaps = gaps_of([s[0] for s in stream])
    contact_gaps = gaps_of([r["c"] for r in recs])

    # Splice of the REDIRECT_FRACTION = 0 run by class of the dropped redirect.
    splice = None
    if args.off_csv and ref is not None:
        splice = splice_scores(labels, read_labels(args.off_csv, n), ref, recs)

    # Public flips (BAR-ATH): frames and distances only.
    public = None
    if args.public_gt:
        public = public_flips(args.public_gt, n, dist, visible, h, vf, lag_frames, stream,
                              stream_frames, alive_cum, recs, labels)

    engine_equal = None
    if args.engine_csv:
        engine_equal = bool(np.array_equal(read_labels(args.engine_csv, n), labels))
    out = {"game": game, "view": args.view, "fps": fps, "h": h, "lags_s": list(LAG_GRID_S),
           "engine_md5": labels_md5(labels), "engine_equal_csv": engine_equal,
           "n_frames": int(n), "n_graded": int(graded.sum()),
           "visible_graded_min": float((visible & graded).sum() / fps / 60.0),
           "alive_graded_min": float((alive & graded).sum() / fps / 60.0),
           "has_ref": ref is not None, "contacts": recs, "handovers": ho_recs,
           "dv3": dv3_recs, "n_dv3_graded": len(dv3_graded), "row_changes": row_changes,
           "engine_ho_d_new": eng_d, "stream_gaps": stream_gaps, "contact_gaps": contact_gaps,
           "splice": splice, "public": public, "wide": wide,
           "cfg": {"R_TOUCH": cfg.R_TOUCH, "REDIRECT_FRACTION": cfg.REDIRECT_FRACTION,
                   "CONTACT_HALF_WINDOW": h, "AIRBORNE_SPEED": cfg.AIRBORNE_SPEED,
                   "DV_CONTACT": cfg.DV_CONTACT}}
    write_json(args.out, out)
    return out


def stretches_of(mask):
    """[(start, stop)] half-open runs of True."""
    mask = np.asarray(mask, dtype=np.int8)
    edges = np.diff(np.concatenate(([0], mask, [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def splice_scores(base, off, ref, recs):
    """spellF1 of the engine, the off run, and the off labels spliced in by class (docstring)."""
    import score_game as sg

    drops = sorted((r["c"], bool(r.get("true_ho"))) for r in recs
                   if r["decision"] == "redirect_dropped")
    drop_frames = np.asarray([d[0] for d in drops], dtype=np.int64)
    classes = {"wrong": [], "right": [], "none": []}
    for start, stop in stretches_of(base != off):
        lo = int(np.searchsorted(drop_frames, start - SPLICE_BACK, side="left"))
        hi = int(np.searchsorted(drop_frames, start + SPLICE_FWD, side="right"))
        if hi > lo:
            best = min(range(lo, hi), key=lambda k: (abs(int(drop_frames[k]) - start), k))
            key = "wrong" if drops[best][1] else "right"
        else:
            key = "none"
        classes[key].append((start, stop))
    gt_codes = sg.pm.label_codes(ref)

    def score(labels):
        result, _, _ = sg.score_codes(gt_codes, sg.pm.label_codes(labels))
        return float(result["spell_f1_mean"])

    out = {"base": score(base), "off": score(off),
           "n_stretches": {k: len(v) for k, v in classes.items()},
           "frames": {k: int(sum(b - a for a, b in v)) for k, v in classes.items()}}
    for key, spans in classes.items():
        spliced = base.copy()
        for a, b in spans:
            spliced[a:b] = off[a:b]
        out["splice_" + key] = score(spliced)
    return out


def public_flips(path, n, dist, visible, h, vf, lag_frames, stream, stream_frames, alive_cum,
                 recs, labels):
    """Frames, distances and confirmations at the four public flips of PLAN 1.3 (BAR-ATH)."""
    gt = read_labels(path)
    frames = pd.read_csv(path, usecols=["frame"])["frame"].to_numpy()
    first = int(frames.min())
    out = []
    contact_frames = np.asarray([r["c"] for r in recs], dtype=np.int64)
    for flip in PUBLIC_FLIPS:
        i = flip - first
        old, new = str(gt[i - 1]), str(gt[i])
        if old not in TEAMS or new not in TEAMS or old == new:
            raise RuntimeError("{} is not an H/A flip of the public file".format(flip))
        lo, hi = max(0, flip - h), min(n, flip + h + 1)
        nearest = int(np.argmin(np.abs(contact_frames - flip))) if contact_frames.size else -1
        rec = {"t": flip, "old": old, "new": new,
               "d_new": _f(window_min(dist[new], lo, hi, visible)),
               "d_new_wide": _f(window_min(dist[new], max(0, flip - 40), min(n, flip + 7),
                                           visible)),
               "d_old": _f(window_min(dist[old], lo, hi, visible)),
               "ball_frames": int(visible[lo:hi].sum()),
               "shape": shape_response(vf, flip, lag_frames),
               "next": next_after_frame(stream_frames, stream, flip, alive_cum),
               "engine_at": str(labels[flip])}
        if nearest >= 0:
            r = recs[nearest]
            rec.update(nearest_contact=r["c"], nearest_kind=r["kind"],
                       nearest_team=r["team"], nearest_decision=r["decision"])
        out.append(rec)
    return out


def write_json(path, payload):
    """Write `payload` as sorted, indented JSON (deterministic)."""
    folder = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.write("\n")


# -- report ---------------------------------------------------------------------------------------

def _fmt(value, digits=2):
    """A number for a table cell, `-` for None."""
    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return "{:.{}f}".format(float(value), digits)


def _pct(num, den):
    """`num/den (p %)`."""
    if not den:
        return "0/0"
    return "{}/{} ({:.0f} %)".format(int(num), int(den), 100.0 * num / den)


def short(game):
    """A short column name for a game."""
    return game.replace("DFL-", "").replace("METRICA-", "")


def lag_index(payload, seconds):
    """Index of `seconds` in the lag grid of a payload."""
    grid = [round(x, 3) for x in payload["lags_s"]]
    return grid.index(round(seconds, 3))


def contact_classes(p):
    """(HO, NHO, NHO-other) lists of contact records with old/new teams set (graded only)."""
    ho_by_idx = {k: r for k, r in enumerate(p["handovers"])}
    ho, nho, nho_other = [], [], []
    for r in p["contacts"]:
        if not r["graded"] or r.get("cls") is None:
            continue
        if r["cls"] == "HO":
            h = ho_by_idx[r["ho"]]
            ho.append(dict(r, old=h["old"], new=h["new"]))
        else:
            old = r["ref_c"]
            item = dict(r, old=old, new=OTHER[old])
            nho.append(item)
            if r["team"] is not None and r["team"] != old:
                nho_other.append(item)
    return ho, nho, nho_other


def features(items, index, clear_key="ref_clear"):
    """Shape features R_new - R_old at lag `index` of the items with a clear window."""
    out = []
    for r in items:
        if not r.get(clear_key, True):
            continue
        out.append(shape_feature(r["shape"], r["new"], r["old"], index))
    return np.asarray(out, dtype=np.float64)


def report_m1(payloads, lines):
    """Measurement 1: team-shape response."""
    refs = [p for p in payloads if p["has_ref"]]
    lines.append("### 1a. Shape response by class, pooled (R_new - R_old, m/s)")
    lines.append("")
    lines.append("| lag s | class | n | p10 | p50 | p90 | R_new p50 | R_old p50 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    pooled = {}
    for lag in LAGS_S:
        for name, pick in (("HO", 0), ("NHO", 1), ("NHO-other", 2)):
            vals, rnew, rold = [], [], []
            for p in refs:
                items = contact_classes(p)[pick]
                i = lag_index(p, lag)
                vals += list(features(items, i))
                rnew += [r["shape"][r["new"]][i] for r in items if r["ref_clear"]]
                rold += [r["shape"][r["old"]][i] for r in items if r["ref_clear"]]
            q = quantiles(vals)
            pooled[(lag, name)] = vals
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                lag, name, q["n"], _fmt(q["p10"]), _fmt(q["p50"]), _fmt(q["p90"]),
                _fmt(quantiles(rnew)["p50"]), _fmt(quantiles(rold)["p50"])))
    lines.append("")
    for title, neg in (("HO against NHO", 1), ("HO against NHO-other", 2)):
        lines.append("### 1{}. AUC, {} (per match and pooled)".format(
            "b" if neg == 1 else "c", title))
        lines.append("")
        lines.append("| match | n HO | n neg | " + " | ".join("{} s".format(l) for l in LAGS_S)
                     + " | best lag on the 0.25 s grid (AUC) |")
        lines.append("|---|---|---|" + "---|" * len(LAGS_S) + "---|")
        pool_pos = {l: [] for l in LAG_GRID_S}
        pool_neg = {l: [] for l in LAG_GRID_S}
        for p in refs:
            cls = contact_classes(p)
            cells, grid = [], []
            for l in LAG_GRID_S:
                i = lag_index(p, l)
                pos, ng = features(cls[0], i), features(cls[neg], i)
                pool_pos[l] += list(pos)
                pool_neg[l] += list(ng)
                grid.append((auc(pos, ng), l))
            for l in LAGS_S:
                i = lag_index(p, l)
                cells.append(_fmt(auc(features(cls[0], i), features(cls[neg], i))))
            best = max((g for g in grid if g[0] is not None), default=(None, None))
            lines.append("| {} | {} | {} | {} | {} s ({}) |".format(
                short(p["game"]), int(np.isfinite(features(cls[0], 0)).sum()),
                int(np.isfinite(features(cls[neg], 0)).sum()),
                " | ".join(cells), _fmt(best[1]), _fmt(best[0])))
        grid = [(auc(pool_pos[l], pool_neg[l]), l) for l in LAG_GRID_S]
        best = max((g for g in grid if g[0] is not None), default=(None, None))
        lines.append("| pooled | {} | {} | {} | {} s ({}) |".format(
            int(np.isfinite(pool_pos[LAG_GRID_S[0]]).sum()),
            int(np.isfinite(pool_neg[LAG_GRID_S[0]]).sum()),
            " | ".join(_fmt(auc(pool_pos[l], pool_neg[l])) for l in LAGS_S),
            _fmt(best[1]), _fmt(best[0])))
        lines.append("")
        lines.append("Pooled AUC on the whole grid: " + ", ".join(
            "{:.2f} s {}".format(l, _fmt(a)) for a, l in grid) + ".")
        lines.append("")
    lines.append("### 1d. Share of HO features above the NHO median (per match)")
    lines.append("")
    lines.append("| match | " + " | ".join("{} s".format(l) for l in LAGS_S) + " |")
    lines.append("|---|" + "---|" * len(LAGS_S))
    for p in refs:
        cls = contact_classes(p)
        lines.append("| {} | {} |".format(short(p["game"]), " | ".join(
            _fmt(share_above(features(cls[0], lag_index(p, l)),
                             features(cls[1], lag_index(p, l)))) for l in LAGS_S)))
    lines.append("| pooled | {} |".format(" | ".join(
        _fmt(share_above(pooled[(l, "HO")], pooled[(l, "NHO")])) for l in LAGS_S)))
    lines.append("")
    lines.append("### 1e. The dv3 rows as candidates: AUC HO against NHO (pooled and per match)")
    lines.append("")
    lines.append("| match | n HO | n NHO | " + " | ".join("{} s".format(l) for l in LAGS_S)
                 + " |")
    lines.append("|---|---|---|" + "---|" * len(LAGS_S))
    pp = {l: ([], []) for l in LAGS_S}
    for p in refs:
        pos_items = [r for r in p["dv3"] if r["cls"] == "HO"]
        neg_items = [r for r in p["dv3"] if r["cls"] == "NHO"]
        cells = []
        for l in LAGS_S:
            i = lag_index(p, l)
            a, b = features(pos_items, i, "clear"), features(neg_items, i, "clear")
            pp[l][0].extend(a)
            pp[l][1].extend(b)
            cells.append(_fmt(auc(a, b)))
        i0 = lag_index(p, LAGS_S[0])
        lines.append("| {} | {} | {} | {} |".format(
            short(p["game"]), int(np.isfinite(features(pos_items, i0, "clear")).sum()),
            int(np.isfinite(features(neg_items, i0, "clear")).sum()), " | ".join(cells)))
    lines.append("| pooled | {} | {} | {} |".format(
        int(np.isfinite(pp[LAGS_S[0]][0]).sum()), int(np.isfinite(pp[LAGS_S[0]][1]).sum()),
        " | ".join(_fmt(auc(*pp[l])) for l in LAGS_S)))
    lines.append("")


def rhythm_row(items, fps, team_key):
    """Cells of the rhythm table for a list of records with `next` items."""
    n = len(items)
    dts, dts2, by_new = [], [], {s: 0 for s in RHYTHM_S}
    second_new, clean = 0, 0
    for r in items:
        nxt = r["next"]
        if not nxt:
            continue
        frame, team, ok = nxt[0]
        dt = (frame - r[team_key]) / r.get("fps", fps)
        dts.append(dt)
        clean += int(ok)
        for s in RHYTHM_S:
            by_new[s] += int(team == r["new"] and dt <= s and ok)
        if len(nxt) > 1:
            dts2.append((nxt[1][0] - r[team_key]) / r.get("fps", fps))
            second_new += int(nxt[1][1] == r["new"])
    q = quantiles(dts)
    return [str(n), _fmt(q["p50"]), _fmt(q["p10"]), _fmt(q["p90"]),
            _fmt(quantiles(dts2)["p50"])] + \
        [_fmt(by_new[s] / float(n) if n else None) for s in RHYTHM_S] + \
        [_fmt(second_new / float(n) if n else None), _fmt(clean / float(n) if n else None)]


def other_team_items(p, decisions=None):
    """Graded contacts of a team other than the stream owner, oriented contact team vs other.

    `new` is the contact team and `old` the other team (label-free orientation). The truth is
    `true_ho`. `decisions` limits the rule decisions (default OTHER_DECISIONS).
    """
    decisions = decisions or OTHER_DECISIONS
    out = []
    for r in p["contacts"]:
        if r["graded"] and r["decision"] in decisions:
            out.append(dict(r, new=r["team"], old=OTHER[r["team"]], fps=p["fps"]))
    return out


def report_other_team(payloads, lines):
    """Shape and rhythm on the contacts that the redirect rule judges (true HO or not)."""
    refs = [p for p in payloads if p["has_ref"]]
    for title, decisions in (("all other-team contacts", OTHER_DECISIONS),
                             ("other-team redirects (confirmed and dropped)",
                              ("redirect_confirmed", "redirect_dropped"))):
        lines.append("#### Shape AUC, true hand-over against not, on {} "
                     "(contact team minus other team)".format(title))
        lines.append("")
        lines.append("| match | n true | n not | " + " | ".join("{} s".format(l) for l in LAGS_S)
                     + " |")
        lines.append("|---|---|---|" + "---|" * len(LAGS_S))
        pool = {l: ([], []) for l in LAGS_S}
        for p in refs:
            items = other_team_items(p, decisions)
            pos = [r for r in items if r["true_ho"]]
            neg = [r for r in items if not r["true_ho"]]
            cells = []
            for l in LAGS_S:
                i = lag_index(p, l)
                a, b = features(pos, i), features(neg, i)
                pool[l][0].extend(a)
                pool[l][1].extend(b)
                cells.append(_fmt(auc(a, b)))
            lines.append("| {} | {} | {} | {} |".format(short(p["game"]), len(pos), len(neg),
                                                       " | ".join(cells)))
        lines.append("| pooled | {} | {} | {} |".format(
            int(np.isfinite(pool[LAGS_S[0]][0]).sum()), int(np.isfinite(pool[LAGS_S[0]][1]).sum()),
            " | ".join(_fmt(auc(*pool[l])) for l in LAGS_S)))
        lines.append("")
        lines.append("#### Rhythm on {}: the next stream item is by the contact team "
                     "(`new`)".format(title))
        lines.append("")
        lines.append("| match | truth | n | dt p50 | dt p10 | dt p90 | dt 2nd p50 | new <= 1 s | "
                     "new <= 2 s | new <= 3 s | 2nd new | next clean |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        pooled = {True: [], False: []}
        for p in refs:
            items = other_team_items(p, decisions)
            for truth in (True, False):
                sel = [r for r in items if bool(r["true_ho"]) == truth]
                pooled[truth] += sel
                lines.append("| {} | {} | {} |".format(short(p["game"]),
                                                       "HO" if truth else "not HO",
                                                       " | ".join(rhythm_row(sel, p["fps"], "c"))))
        for truth in (True, False):
            lines.append("| pooled | {} | {} |".format("HO" if truth else "not HO", " | ".join(
                rhythm_row(pooled[truth], 25.0, "c"))))
        lines.append("")


def report_m2(payloads, lines):
    """Measurement 2: contact rhythm."""
    refs = [p for p in payloads if p["has_ref"]]
    lines.append("### 2a. Next stream item after a contact (bump candidates)")
    lines.append("")
    lines.append("dt in seconds. `new <= s`: the next item is by the new team, clean, within s "
                 "seconds. `2nd new`: the second-next item is by the new team.")
    lines.append("")
    head = ("| match | class | n | dt p50 | dt p10 | dt p90 | dt 2nd p50 | new <= 1 s | "
            "new <= 2 s | new <= 3 s | 2nd new | next clean |")
    rule = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines.append(head)
    lines.append(rule)
    pooled = {"HO": [], "NHO": [], "NHO-other": []}
    for p in refs:
        cls = contact_classes(p)
        for name, items in zip(("HO", "NHO", "NHO-other"), cls):
            items = [dict(r, fps=p["fps"]) for r in items]
            pooled[name] += items
            lines.append("| {} | {} | {} |".format(short(p["game"]), name, " | ".join(
                rhythm_row(items, p["fps"], "c"))))
    for name, items in pooled.items():
        lines.append("| pooled | {} | {} |".format(name, " | ".join(
            rhythm_row(items, 25.0, "c"))))
    lines.append("")
    lines.append("### 2b. The dv3 rows as candidates (pooled)")
    lines.append("")
    lines.append(head)
    lines.append(rule)
    for name in ("HO", "NHO"):
        items = [r for p in refs for r in p["dv3"] if r["cls"] == name]
        lines.append("| pooled | {} | {} |".format(name, " | ".join(
            rhythm_row(items, 25.0, "c"))))
    lines.append("")
    lines.append("### 2c. Dropped redirects: gap to the next stream item (F2's 47 frames)")
    lines.append("")
    lines.append("| match | clause | truth | n | gap p50 frames | p10 | p90 | <= 50 frames | "
                 "<= 100 frames |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for p in payloads:
        groups = {}
        for r in p["contacts"]:
            if not r["graded"] or r["decision"] != "redirect_dropped" or not r["next"]:
                continue
            truth = ("HO" if r.get("true_ho") else "not HO") if p["has_ref"] else "no labels"
            groups.setdefault((r["clause"], truth), []).append(r["next"][0][0] - r["c"])
        for (clause, truth) in sorted(groups):
            g = groups[(clause, truth)]
            q = quantiles(g)
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                short(p["game"]), clause, truth, len(g), _fmt(q["p50"], 0), _fmt(q["p10"], 0),
                _fmt(q["p90"], 0), sum(1 for x in g if x <= 50), sum(1 for x in g if x <= 100)))
    lines.append("")
    lines.append("### 2d. Gap between consecutive clean items, graded frames (label-free)")
    lines.append("")
    lines.append("| match | stream items: n | p10 | p25 | p50 | p75 | p90 | contacts: n | p50 | "
                 "p90 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for p in payloads:
        q = quantiles(p["stream_gaps"], (10, 25, 50, 75, 90))
        qc = quantiles(p["contact_gaps"], (50, 90))
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            short(p["game"]), q["n"], _fmt(q["p10"], 0), _fmt(q["p25"], 0), _fmt(q["p50"], 0),
            _fmt(q["p75"], 0), _fmt(q["p90"], 0), qc["n"], _fmt(qc["p50"], 0),
            _fmt(qc["p90"], 0)))
    lines.append("")


OTHER_DECISIONS = ("kick_other", "receive_other", "redirect_confirmed", "redirect_dropped")
OWN_DECISIONS = ("kick_own", "receive_own", "redirect_own", "kick_none", "receive_none")


def truth_of(r, truth):
    """The audit truth of a contact: "ho" (true_ho) or "after" (owner after = contact team)."""
    if truth == "after":
        return r["ref_after"] == r["team"]
    return bool(r.get("true_ho"))


def audit_counts(p, truth="ho"):
    """{decision: (n, n true)} on graded uncontested contacts, under one audit truth."""
    out = {}
    for r in p["contacts"]:
        if not r["graded"] or r["decision"] is None:
            continue
        if truth == "after" and r["ref_after"] not in TEAMS:
            continue
        n, k = out.get(r["decision"], (0, 0))
        out[r["decision"]] = (n + 1, k + int(truth_of(r, truth)))
    return out


def redirect_pr(counts):
    """(precision, recall) of the redirect class (module docstring)."""
    dn, dk = counts.get("redirect_dropped", (0, 0))
    neg = sum(counts.get(d, (0, 0))[0] - counts.get(d, (0, 0))[1] for d in OTHER_DECISIONS)
    prec = (dn - dk) / float(dn) if dn else None
    rec = (dn - dk) / float(neg) if neg else None
    return prec, rec


def report_m3(payloads, lines, label=""):
    """Measurement 3: the redirect rule audit."""
    refs = [p for p in payloads if p["has_ref"]]
    for truth, title in (("ho", "a reference hand-over to the contact team within 25 frames"),
                         ("after", "the reference owner at c + h + 1 is the contact team")):
        report_audit(refs, lines, label, truth, title)


def report_audit(refs, lines, label, truth, title):
    """3a: the rule decisions against one audit truth."""
    lines.append("### 3a{}{}. Rule decision against the reference: share where {}".format(
        label, "" if truth == "ho" else " (truth: owner after)", title))
    lines.append("")
    cols = OTHER_DECISIONS + OWN_DECISIONS
    lines.append("| match | " + " | ".join(cols) + " | redirect precision | redirect recall |")
    lines.append("|---|" + "---|" * (len(cols) + 2))
    pool = {}
    for p in refs:
        counts = audit_counts(p, truth)
        for d, (n, k) in counts.items():
            a, b = pool.get(d, (0, 0))
            pool[d] = (a + n, b + k)
        prec, rec = redirect_pr(counts)
        lines.append("| {} | {} | {} | {} |".format(short(p["game"]), " | ".join(
            _pct(counts.get(d, (0, 0))[1], counts.get(d, (0, 0))[0]) for d in cols),
            _fmt(prec), _fmt(rec)))
    prec, rec = redirect_pr(pool)
    lines.append("| pooled | {} | {} | {} |".format(" | ".join(
        _pct(pool.get(d, (0, 0))[1], pool.get(d, (0, 0))[0]) for d in cols), _fmt(prec),
        _fmt(rec)))
    lines.append("")


def report_m3_rest(payloads, lines, label=""):
    """3b to 3d: splice, fraction buckets and deflection tests."""
    refs = [p for p in payloads if p["has_ref"]]
    if any(p.get("splice") for p in refs):
        lines.append("### 3b{}. spellF1 of the REDIRECT_FRACTION = 0 run spliced in by class of "
                     "the dropped redirect".format(label))
        lines.append("")
        lines.append("| match | engine | off | splice true-HO drops | splice not-HO drops | "
                     "splice no drop | stretches HO / not / none | frames HO / not / none |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for p in refs:
            s = p.get("splice")
            if not s:
                continue
            lines.append("| {} | {:.4f} | {:.4f} ({:+.4f}) | {:.4f} ({:+.4f}) | {:.4f} ({:+.4f}) "
                         "| {:.4f} ({:+.4f}) | {} / {} / {} | {} / {} / {} |".format(
                             short(p["game"]), s["base"], s["off"], s["off"] - s["base"],
                             s["splice_wrong"], s["splice_wrong"] - s["base"],
                             s["splice_right"], s["splice_right"] - s["base"],
                             s["splice_none"], s["splice_none"] - s["base"],
                             s["n_stretches"]["wrong"], s["n_stretches"]["right"],
                             s["n_stretches"]["none"], s["frames"]["wrong"],
                             s["frames"]["right"], s["frames"]["none"]))
        lines.append("")
    lines.append("### 3c{}. Other-team contacts by |ds| / |J| bucket (pooled; share true hand-over)"
                 .format(label))
    lines.append("")
    lines.append("| |ds|/|J| | n (ds > 0) | true HO (ds > 0) | n (ds < 0) | true HO (ds < 0) | "
                 "all |")
    lines.append("|---|---|---|---|---|---|")
    edges = FRACTION_EDGES
    for lo, hi in zip(edges[:-1], edges[1:]):
        pos = [r for p in refs for r in p["contacts"] if r["graded"]
               and r["decision"] in OTHER_DECISIONS and r["frac"] is not None
               and lo <= r["frac"] < hi]
        up = [r for r in pos if r["ds"] > 0]
        down = [r for r in pos if r["ds"] <= 0]
        lines.append("| {:.1f}-{:.1f} | {} | {} | {} | {} | {} |".format(
            lo, min(hi, 1.0), len(up), _fmt(np.mean([r["true_ho"] for r in up]) if up else None),
            len(down), _fmt(np.mean([r["true_ho"] for r in down]) if down else None),
            _pct(sum(r["true_ho"] for r in pos), len(pos))))
    lines.append("")
    lines.append("### 3d{}. Other-team contacts by deflection test (pooled; share true hand-over)"
                 .format(label))
    lines.append("")
    lines.append("| population | test | deflection | not deflection |")
    lines.append("|---|---|---|---|")
    for pop_name, decisions in (("redirects of the other team", ("redirect_confirmed",
                                                                   "redirect_dropped")),
                                ("kicks of the other team", ("kick_other",)),
                                ("receives of the other team", ("receive_other",)),
                                ("all other-team contacts", OTHER_DECISIONS)):
        pop = [r for p in refs for r in p["contacts"] if r["graded"]
               and r["decision"] in decisions]
        for key in ("defl3", "defl_bump"):
            yes = [r for r in pop if r[key]]
            no = [r for r in pop if not r[key]]
            lines.append("| {} | {} | {} | {} |".format(
                pop_name, "dv3 at c" if key == "defl3" else "over +-h",
                _pct(sum(r["true_ho"] for r in yes), len(yes)),
                _pct(sum(r["true_ho"] for r in no), len(no))))
    lines.append("")


def report_not_ho(payloads, lines, label=""):
    """3e: the other-team contacts that are no true hand-over, by the reference owner at c."""
    refs = [p for p in payloads if p["has_ref"]]
    lines.append("### 3e{}. Other-team contacts that are no true hand-over: reference owner at c"
                 .format(label))
    lines.append("")
    lines.append("`own` means the reference already gives the ball to the contact team at c (the "
                 "stream owner of the engine differs from the reference owner).")
    lines.append("")
    lines.append("| match | decision | n not HO | ref owner = contact team | ref owner = other "
                 "team | ref D or X | owner after = contact team |")
    lines.append("|---|---|---|---|---|---|---|")
    pool = {}
    for p in refs:
        for dec in ("redirect_dropped", "redirect_confirmed", "kick_other", "receive_other"):
            sel = [r for r in p["contacts"] if r["graded"] and r["decision"] == dec
                   and not r["true_ho"]]
            own = sum(1 for r in sel if r["ref_c"] == r["team"])
            oth = sum(1 for r in sel if r["ref_c"] == OTHER[r["team"]])
            after = sum(1 for r in sel if r["ref_after"] == r["team"])
            rest = len(sel) - own - oth
            a = pool.get(dec, (0, 0, 0, 0, 0))
            pool[dec] = (a[0] + len(sel), a[1] + own, a[2] + oth, a[3] + rest, a[4] + after)
            if dec == "redirect_dropped":
                lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                    short(p["game"]), dec, len(sel), _pct(own, len(sel)), _pct(oth, len(sel)),
                    rest, _pct(after, len(sel))))
    for dec in ("redirect_dropped", "redirect_confirmed", "kick_other", "receive_other"):
        a = pool.get(dec, (0, 0, 0, 0, 0))
        lines.append("| pooled | {} | {} | {} | {} | {} | {} |".format(
            dec, a[0], _pct(a[1], a[0]), _pct(a[2], a[0]), a[3], _pct(a[4], a[0])))
    lines.append("")


def report_wide(payloads, lines, label=""):
    """4d: impulse maxima with the nearest player beyond R_TOUCH (a wider radius)."""
    lines.append("### 4d{}. Impulse maxima with the nearest player beyond 1.5 m (what a wider "
                 "radius would add)".format(label))
    lines.append("")
    lines.append("Graded, alive impulse maxima with |J| >= DV_CONTACT. `switch` = the nearest "
                 "team is not the engine label at c. `shape` = the 1.0 s feature (nearest team "
                 "minus other) above the median of the NHO contacts of the match (BAR-ATH: of all "
                 "graded contacts oriented by the engine owner). `rhythm` = the next stream item "
                 "is by the nearest team, clean, within 2 s. `true HO` = a reference hand-over to "
                 "the nearest team within 25 frames.")
    lines.append("")
    lines.append("| match | band m | n | per alive min | switch | switch, true HO | switch + "
                 "shape | ... true HO | switch + rhythm | ... true HO | switch + both | ... true "
                 "HO |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    bands = list(zip(RADII[:-1], RADII[1:]))
    for p in payloads:
        if "wide" not in p:
            continue
        i = lag_index(p, 1.0)
        if p["has_ref"]:
            med = np.median(features(contact_classes(p)[1], i))
        else:
            own = [shape_feature(r["shape"], OTHER[r["engine_at_c"]], r["engine_at_c"], i)
                   for r in p["contacts"] if r["graded"] and r["engine_at_c"] in TEAMS]
            med = float(np.nanmedian(own)) if own else np.nan
        for lo, hi in bands:
            sel = [r for r in p["wide"] if lo < r["d1"] <= hi]
            sw = [r for r in sel if r["engine_at_c"] in TEAMS and r["engine_at_c"] != r["team"]]

            def ok_shape(r):
                return r["shape10"] is not None and r["shape10"] > med

            def ok_rhythm(r):
                nxt = r["next"]
                return bool(nxt and nxt[0][1] == r["team"] and nxt[0][2]
                            and (nxt[0][0] - r["c"]) / p["fps"] <= 2.0)

            groups = [sw, [r for r in sw if ok_shape(r)], [r for r in sw if ok_rhythm(r)],
                      [r for r in sw if ok_shape(r) and ok_rhythm(r)]]
            cells = []
            for k, g in enumerate(groups):
                cells.append(str(len(g)))
                if p["has_ref"]:
                    cells.append(_pct(sum(1 for r in g if r["true_ho"]), len(g)))
                else:
                    cells.append("-")
            lines.append("| {} | {}-{} | {} | {:.2f} | {} |".format(
                short(p["game"]), lo, hi, len(sel), len(sel) / p["alive_graded_min"],
                " | ".join(cells)))
    lines.append("")


def report_m4(payloads, lines, label=""):
    """Measurement 4: nobody of the new team within R_TOUCH."""
    refs = [p for p in payloads if p["has_ref"]]
    best = 1.0
    if not refs:
        report_m4_engine(payloads, lines, label)
        return
    lines.append("### 4a{}. Reference hand-overs with no new-team player within 1.5 m in "
                 "[t - h, t + h]".format(label))
    lines.append("")
    lines.append("Shape confirmation: the feature at {} s above the NHO median of the match. "
                 "Rhythm confirmation: the next stream item after t is by the new team, clean, "
                 "within 2 s.".format(best))
    lines.append("")
    lines.append("| match | hand-overs | no ball | nobody <= 1.5 m | d_new p50 | d_new p90 | "
                 "reach 2.0 | reach 2.5 | reach 3.0 | shape | rhythm | either | matched contact |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    pool = {"n": 0, "noball": 0, "far": [], "shape": 0, "rhythm": 0, "either": 0,
            "matched": 0}
    for p in refs:
        i = lag_index(p, best)
        cls = contact_classes(p)
        nho_med = np.median(features(cls[1], i)) if len(features(cls[1], i)) else np.nan
        hos = p["handovers"]
        noball = [r for r in hos if not r["ball_in_window"]]
        far = [r for r in hos if r["ball_in_window"] and (r["d_new"] is None
                                                           or r["d_new"] > 1.5)]
        d = [r["d_new"] if r["d_new"] is not None else np.inf for r in far]
        shape = [bool(r["clear"] and shape_feature(r["shape"], r["new"], r["old"], i)
                      > nho_med) for r in far]
        rhythm = []
        for r in far:
            nxt = r["next"]
            rhythm.append(bool(nxt and nxt[0][1] == r["new"] and nxt[0][2]
                               and (nxt[0][0] - r["t"]) / p["fps"] <= 2.0))
        either = [a or b for a, b in zip(shape, rhythm)]
        matched = sum(1 for r in far if r["contact"] is not None)
        q = quantiles([x for x in d if np.isfinite(x)])
        reach = [sum(1 for x in d if x <= rad) for rad in RADII[1:]]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            short(p["game"]), len(hos), len(noball), len(far), _fmt(q["p50"]), _fmt(q["p90"]),
            reach[0], reach[1], reach[2], sum(shape), sum(rhythm), sum(either), matched))
        pool["n"] += len(hos)
        pool["noball"] += len(noball)
        pool["far"] += d
        pool["shape"] += sum(shape)
        pool["rhythm"] += sum(rhythm)
        pool["either"] += sum(either)
        pool["matched"] += matched
    q = quantiles([x for x in pool["far"] if np.isfinite(x)])
    reach = [sum(1 for x in pool["far"] if x <= rad) for rad in RADII[1:]]
    lines.append("| pooled | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        pool["n"], pool["noball"], len(pool["far"]), _fmt(q["p50"]), _fmt(q["p90"]), reach[0],
        reach[1], reach[2], pool["shape"], pool["rhythm"], pool["either"], pool["matched"]))
    lines.append("")
    report_m4_engine(payloads, lines, label)


def report_m4_engine(payloads, lines, label=""):
    """Measurement 4, label-free: nearest new-team distance at the engine's hand-overs."""
    lines.append("### 4b{}. Nearest new-team distance at the engine's own hand-overs (label-free)"
                 .format(label))
    lines.append("")
    lines.append("| match | engine hand-overs | p50 m | p90 m | <= 1.5 m | <= 2.0 m | <= 2.5 m | "
                 "<= 3.0 m | no ball |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for p in payloads:
        d = [x for x in p["engine_ho_d_new"]]
        fin = [x for x in d if x is not None]
        q = quantiles(fin)
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            short(p["game"]), len(d), _fmt(q["p50"]), _fmt(q["p90"]),
            *([_pct(sum(1 for x in fin if x <= rad), len(d)) for rad in RADII]
              + [len(d) - len(fin)])))
    lines.append("")


def report_public(payloads, lines):
    """Measurement 4, BAR-ATH: the four public flips."""
    for p in payloads:
        if not p.get("public"):
            continue
        i = lag_index(p, 1.0)
        own = []
        for r in p["contacts"]:
            if r["graded"] and r["engine_at_c"] in TEAMS:
                own.append(shape_feature(r["shape"], OTHER[r["engine_at_c"]],
                                         r["engine_at_c"], i))
        med = float(np.nanmedian(own)) if own else None
        lines.append("### 4c. BAR-ATH public flips of PLAN 1.3 (frames and distances)")
        lines.append("")
        lines.append("Shape feature at 1.0 s; for scale, the median over all graded BAR-ATH "
                     "contacts oriented by the engine owner at c is {} m/s.".format(_fmt(med)))
        lines.append("")
        lines.append("| flip | d_new min [t-h, t+h] m | d_new min [t-40, t+6] m | d_old min "
                     "m | ball frames in window | reach 2.0 / 2.5 / 3.0 | shape 1.0 s | "
                     "next item dt s, team, clean | nearest bump contact (dt, kind, team, "
                     "decision) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in p["public"]:
            d = r["d_new"]
            reach = " / ".join("yes" if (d is not None and d <= rad) else "no"
                               for rad in RADII[1:])
            nxt = r["next"][0] if r["next"] else None
            nxt_txt = "{:.2f}, {}, {}".format((nxt[0] - r["t"]) / p["fps"],
                                             "new" if nxt[1] == r["new"] else "old",
                                             "clean" if nxt[2] else "dead between") \
                if nxt else "-"
            near = "{:+d}, {}, {}, {}".format(r["nearest_contact"] - r["t"], r["nearest_kind"],
                                             "new" if r["nearest_team"] == r["new"] else
                                             ("old" if r["nearest_team"] == r["old"] else "none"),
                                             r["nearest_decision"]) \
                if "nearest_contact" in r else "-"
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r["t"], _fmt(d), _fmt(r["d_new_wide"]), _fmt(r["d_old"]), r["ball_frames"],
                reach, _fmt(shape_feature(r["shape"], r["new"], r["old"], i)), nxt_txt, near))
        lines.append("")


def report_m5(payloads, lines):
    """Measurement 5: attribution instants and airborne receives."""
    lines.append("### 5a. Separation of the two teams' argmin instants and single-instant "
                 "attributions")
    lines.append("")
    lines.append("Category changes are against the current attribution (none, contested, H, "
                 "A). `rows` re-runs the redirect rule and counts graded contacts whose final "
                 "row (kind and team) changes.")
    lines.append("")
    lines.append("| match | contacts | sep p50 | sep p90 | sep >= 12 | centre: any change | "
                 "team flip | contested flag | rows | instant: any change | team flip | "
                 "contested flag | rows |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in payloads:
        cs = [r for r in p["contacts"] if r["graded"]]
        seps = [r["at_sep"] for r in cs if r["at_sep"] is not None]
        q = quantiles(seps)
        cells = []
        for key in ("cat_centre", "cat_inst"):
            any_c = sum(1 for r in cs if r[key] != r["cat_cur"])
            flip = sum(1 for r in cs if r[key] in TEAMS and r["cat_cur"] in TEAMS
                       and r[key] != r["cat_cur"])
            cont = sum(1 for r in cs if (r[key] == "contested") != (r["cat_cur"] == "contested"))
            rc = p["row_changes"]["centre" if key == "cat_centre" else "instant"]
            cells += [_pct(any_c, len(cs)), str(flip), str(cont), _pct(rc["changed"], rc["n"])]
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            short(p["game"]), len(cs), _fmt(q["p50"], 0), _fmt(q["p90"], 0),
            _pct(sum(1 for s in seps if s >= 12), len(seps)), " | ".join(cells)))
    lines.append("")
    lines.append("### 5b. Agreement of the attributed team with the reference owner after the "
                 "contact")
    lines.append("")
    lines.append("Contacts whose reference label at c + h + 1 is H or A. A contact without a "
                 "team under an attribution counts as a disagreement in `all`; `with team` "
                 "counts only contacts that have a team under that attribution.")
    lines.append("")
    lines.append("| match | n | current all | centre all | instant all | current with team | "
                 "centre with team | instant with team | centre differs: current right / "
                 "centre right | instant differs: current right / instant right |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    pool = {"n": 0}
    for p in payloads:
        if not p["has_ref"]:
            continue
        cs = [r for r in p["contacts"] if r["graded"] and r["ref_after"] in TEAMS]
        cells_all, cells_team = [], []
        for key in ("cat_cur", "cat_centre", "cat_inst"):
            ok = sum(1 for r in cs if r[key] == r["ref_after"])
            withteam = [r for r in cs if r[key] in TEAMS]
            okt = sum(1 for r in withteam if r[key] == r["ref_after"])
            cells_all.append(_pct(ok, len(cs)))
            cells_team.append(_pct(okt, len(withteam)))
            pool[key] = tuple(np.add(pool.get(key, (0, 0, 0, 0)), (ok, len(cs), okt,
                                                                   len(withteam))))
        pair = []
        for key in ("cat_centre", "cat_inst"):
            diff = [r for r in cs if r[key] != r["cat_cur"]]
            cur_ok = sum(1 for r in diff if r["cat_cur"] == r["ref_after"])
            alt_ok = sum(1 for r in diff if r[key] == r["ref_after"])
            pair.append("{} / {} of {}".format(cur_ok, alt_ok, len(diff)))
            pool["pair_" + key] = tuple(np.add(pool.get("pair_" + key, (0, 0, 0)),
                                               (cur_ok, alt_ok, len(diff))))
        dcells = " | ".join(pair)
        pool["n"] += len(cs)
        lines.append("| {} | {} | {} | {} | {} |".format(short(p["game"]), len(cs),
                                                         " | ".join(cells_all),
                                                         " | ".join(cells_team), dcells))
    if pool["n"]:
        lines.append("| pooled | {} | {} | {} | {} / {} of {} | {} / {} of {} |".format(
            pool["n"], " | ".join(_pct(pool[k][0], pool[k][1]) for k in ("cat_cur", "cat_centre",
                                                                        "cat_inst")),
            " | ".join(_pct(pool[k][2], pool[k][3]) for k in ("cat_cur", "cat_centre",
                                                              "cat_inst")),
            *(pool["pair_cat_centre"] + pool["pair_cat_inst"])))
    lines.append("")
    lines.append("### 5c. Bump receives with a fast ball (speed at c above AIRBORNE_SPEED)")
    lines.append("")
    lines.append("| match | receives | fast at c | fast at c - h | accuracy fast | accuracy "
                 "slow | airborne_suspect at c: fast / slow | airborne_suspect in window: fast "
                 "/ slow | fast receives that change the owner |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for p in payloads:
        thr = p["cfg"]["AIRBORNE_SPEED"]
        rc = [r for r in p["contacts"] if r["graded"] and r["kind"] == "receive"]
        fast = [r for r in rc if r["s_c"] is not None and r["s_c"] > thr]
        slow = [r for r in rc if not (r["s_c"] is not None and r["s_c"] > thr)]
        fast_in = sum(1 for r in rc if r["s_in"] is not None and r["s_in"] > thr)
        if p["has_ref"]:
            fa = [r for r in fast if r["ref_after"] in TEAMS]
            sa = [r for r in slow if r["ref_after"] in TEAMS]
            acc_f = _pct(sum(1 for r in fa if r["team"] == r["ref_after"]), len(fa))
            acc_s = _pct(sum(1 for r in sa if r["team"] == r["ref_after"]), len(sa))
        else:
            acc_f = acc_s = "no labels"
        change = sum(1 for r in fast if r["decision"] == "receive_other")
        lines.append("| {} | {} | {} | {} | {} | {} | {} / {} | {} / {} | {} |".format(
            short(p["game"]), len(rc), _pct(len(fast), len(rc)), _pct(fast_in, len(rc)), acc_f,
            acc_s, sum(r["airborne_c"] for r in fast), sum(r["airborne_c"] for r in slow),
            sum(r["airborne_win"] for r in fast), sum(r["airborne_win"] for r in slow),
            change))
    lines.append("")


def report_overview(payloads, lines):
    """Per-match counts and the engine check."""
    lines.append("### 0. Matches, engine check and counts")
    lines.append("")
    lines.append("| match | view | engine md5 | equals CSV | bump contacts (graded) | per visible "
                 "min | reference hand-overs | matched by bump | matched by dv3 | HO / NHO / "
                 "NHO-other |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for p in payloads:
        cs = [r for r in p["contacts"] if r["graded"]]
        hos = p["handovers"]
        if p["has_ref"]:
            cls = contact_classes(p)
            counts = "{} / {} / {}".format(len(cls[0]), len(cls[1]), len(cls[2]))
            mb = _pct(sum(1 for r in hos if r["contact"] is not None), len(hos))
            md = _pct(sum(1 for r in hos if r["dv3"] is not None), len(hos))
        else:
            counts, mb, md = "-", "-", "-"
        lines.append("| {} | {} | `{}` | {} | {} | {:.1f} | {} | {} | {} | {} |".format(
            short(p["game"]), p["view"], p["engine_md5"][:8], p["engine_equal_csv"], len(cs),
            len(cs) / p["visible_graded_min"] if p["visible_graded_min"] else 0.0, len(hos),
            mb, md, counts))
    lines.append("")


def report(args):
    """The `report` command: print the markdown tables."""
    payloads = []
    for path in args.paths:
        with open(path) as handle:
            payloads.append(json.load(handle))
    lines = []
    report_overview(payloads, lines)
    raw = [p for p in payloads if p["view"] == "raw"]
    smooth = [p for p in payloads if p["view"] == "smooth"]
    for name, group in (("raw", raw), ("smooth", smooth)):
        if not group:
            continue
        lines.append("## View: {}".format(name))
        lines.append("")
        if any(p["has_ref"] for p in group):
            report_m1(group, lines)
            report_m2(group, lines)
            lines.append("### 2e. The population the redirect rule judges")
            lines.append("")
            report_other_team(group, lines)
            report_m3(group, lines, label="")
            report_m3_rest(group, lines)
            report_not_ho(group, lines)
            report_m4(group, lines, label="")
        else:
            report_m2(group, lines)
            report_m4(group, lines, label="")
        report_public(group, lines)
        report_wide(group, lines)
        report_m5(group, lines)
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text)
    sys.stdout.write(text)
    return text


def main(argv=None):
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command")
    c = sub.add_parser("collect")
    c.add_argument("--game", required=True)
    c.add_argument("--ref")
    c.add_argument("--engine-csv")
    c.add_argument("--off-csv")
    c.add_argument("--public-gt")
    c.add_argument("--cache-dir")
    c.add_argument("--view", default="raw", choices=("raw", "smooth"))
    c.add_argument("--config", help="JSON of Config overrides (engine.config.load)")
    c.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("paths", nargs="+")
    r.add_argument("--out")
    args = parser.parse_args(argv)
    if args.command == "collect":
        collect(args)
    elif args.command == "report":
        report(args)
    else:
        parser.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
