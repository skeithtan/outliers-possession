"""Re-measure the "public case only" constants of engine/config.py on external references (P9-F).

    python tools/analysis/provenance_recheck.py measure --game NAME --ref REFERENCE.csv
        [--data-dir DIR] [--config OVERRIDES.json] --out OUT.json
    python tools/analysis/provenance_recheck.py report OUT.json [OUT.json ...]
    python tools/analysis/provenance_recheck.py bucketb OUT.json x8 [--json SELECTION.json]
        [--lag L --near-close M]

`measure` loads one external match through the engine data layer, runs the feature functions of
engine.tracks, and reads the reference labels as the owner. It does not run the decoder and it
changes no engine file. `report` prints the markdown tables of
docs/plan/calibration/PROVENANCE_RECHECK.md from the JSON files. `bucketb` applies the
pre-registered bucket b rules of docs/plan/calibration/grids.json to the eight open references
(see "Bucket b" below) and prints the tables of docs/plan/calibration/BUCKET_B.md. Write the JSON
files outside the
repository. The script prints summary statistics only. It refuses BAR-ATH and the two sealed
matches of docs/plan/calibration/SPLITS.md. It reads no ground truth file.

Definitions. A label is H, A, D or X. Alive means H or A. The owner of a frame is its reference
label. All distances are in metres and all times in frames at the match frame rate.

Measurement 1, the nearest player (PLAN 1.2, 1.3, NEAREST_LAG, P_NEAR_*, NEAR_*_M).

- Usable source frame. The cleaned ball is visible, the nearest player has a team, and the frame
  is not a ghost-ball frame (the decoder zeroes the evidence there).
- Agreement at lag L. Take every alive frame t with a usable source frame t - L. The pair agrees
  when the team of the nearest player at t - L is the owner at t. The decoder reads the evidence
  at lag NEAREST_LAG. A negative L reads a source frame after t. The curve runs from -10 to 45.
- Bands. The band is set by the distance of the nearest player at the source frame: at most 0.5,
  1.0 or 1.5 m, from 1 to 3 m, from 3 to 5 m, from 1 to 5 m (the range that P_NEAR_FAR covers),
  at most 5 m, beyond 5 m, and all.
- Log-odds. log(p / (1 - p)) of an agreement rate p. The decoder uses it with the sign of the
  nearest team. The shipped constants are P_NEAR_CLOSE within NEAR_CLOSE_M and P_NEAR_FAR within
  NEAR_FAR_M.
- Flip lead. For each reference frame t that starts a possession (label alive, label before it
  different), let T be its team. Let ok(u) hold when the source frame u is usable and its nearest
  player is of team T. The flip is the first frame u0 of the first run of five consecutive ok
  frames that starts at or after t - 25 (the definition of tools/analysis/analyse_public.py),
  moved back while ok holds on the frame before, for at most 100 frames. The lead is t - u0. A
  positive lead means the nearest team flips before the label. A hand-over is a flip from the
  other alive team. A restart is a flip from D.

Measurement 2, ONSET_CONFIRM_S. A reference onset is a frame t with label D and an alive label
at t - 1. The stop runs are engine.deadball._stop_runs: runs of meanv_all < V_STOP with sep >
SEP_DEAD that last at least COLLECTIVE_STOP_S. The delay of an onset is max(0, start - t) for
the first stop run that ends after t. It is 0 when the onset lies inside a run. A delay of at
most h frames is the engine's confirmed() test with a horizon of h. The false rate for horizon
h is the same test on every alive frame t that has no non-alive frame in t .. t + h. It is the
chance that a random alive moment would confirm an onset. The same delay is reported for the
onsets that a pursuit candidate owns (measurement 3) and for the near losses (measurement 5).
Those are the two kinds of onset that the engine confirms.

Measurement 3, PURSUIT_KICK_LOOKBACK. A pursuit candidate is a fast ball run that passes every
test of engine.deadball._pursuit_onsets except the confirmation by a collective stop. The script
re-implements the loop and checks that its onsets equal the engine's for a confirmation that
always holds. The kicking team is the reference owner at the run start. The look-back offset is
the run start minus the latest frame at or before the run start where the kicking team is nearer
to the ball than the other team by CONTESTED_MARGIN_M or more. It is 0 when the margin holds at
the run start. The search goes back at most 100 frames. The engine's mean over
PURSUIT_KICK_LOOKBACK frames gives its kicking team. The script counts how often that team is
the reference owner. Two groups are reported: every candidate, and the candidates whose onset
lies within 50 frames of a reference onset.

Measurement 4, TOUCH_ONSET_MAX_FRAMES and RELEASE_K. A reference hand-over is a frame t with an
alive label and a different alive label at t - 1. The dv3 peak is the largest local maximum of
dv_3 above DV_TOUCH in t - 25 .. t + 25. The contact is engine.events.contact_frame with a cap
of 60 frames. The walk-back length is peak - contact. The cap truncates the walk when the length
exceeds TOUCH_ONSET_MAX_FRAMES. A hand-over counts when a player is within R_TOUCH of the ball
between the contact and the peak plus DV_K. A kick is a counted contact where the speed rises by
more than DV_RELEASE within 30 frames. The peak-speed length is the number of frames from the
contact to the speed maximum within 30 frames, in the same visible run. The gain length is the
number of frames to the first gain above DV_RELEASE, which is what RELEASE_K must cover. The
release share at K is the share of counted contacts where the speed rises by more than
DV_RELEASE within K frames, from the contact that the shipped cap gives (events._rise_after).
Everything is computed for the raw ball and for the ball smoothed with the ATD kernel
(tools/atd_view.py, sigma 9.5).

Measurement 5, DELTA_OUT_LAG and OUT_MAX_RANGE_M. The loss frame f of a reference onset t is the
last frame of the visible ball track that holds t, or the last visible frame before t. The lag
is t - f, negative when the ball stays visible after the onset. The loss is near when the lag is
at most 250 frames in size. The visible track is the cleaned ball of the engine, so a ball that
the teleport rule rejects counts as lost. At the loss frame inside the pitch and at a speed of at
least HEADING_MIN_SPEED the heading distance is engine.deadball._boundary_distance. The roll
reaches the line when speed squared is at least twice FRICTION_DECEL times that distance. The lag
rule serves a loss with a heading distance of at most OUT_MAX_RANGE_M and no roll to the line.
Its error is f + DELTA_OUT_LAG - t. The extrapolation serves a loss with the roll to the line.
Its error is the extrapolated crossing minus t. An exit is a near loss inside the pitch whose next
visible ball lies within 2 m of a pitch line. The crossing offset is, for an onset with the
visible ball beyond a line, t minus the first frame of that excursion.

Bucket b (COMPLIANCE_PROTOCOL.md, written before the selection run). `measure` also stores the
band-limited lag curve: the agreement over the lags 0 to 45 in bands of 0.5 m from 0 to 8 m and
one band beyond 8 m. A band (a, b] on the 0.5 m grid is the sum of its bands. `bucketb` applies
the rules in the order of the advisor ruling of 8 Nov 2026 (recorded in
docs/plan/calibration/BUCKET_B.md). The median of eight values is the mean of the middle two. A
probability is rounded to 3 decimals.

- NEAR_FAR_M at a lag. The smallest whole edge E such that, on every match, the pooled agreement
  over (E, 8] m is within 3 points of the agreement over (5, 8] m of that match.
- NEAREST_LAG within a NEAR_FAR_M. Per match, the lag in 0..45 with the highest agreement in the
  band (0, NEAR_FAR_M], the first on a tie. The value is the median over the eight matches, a
  half rounded up. A match with an empty band takes the argmax of the all-distance curve.
- The two fields are iterated to a fixed point from lag 0: NEAR_FAR_M at lag 0, the lag within
  it, NEAR_FAR_M at that lag, and so on, until neither changes. On a cycle the state with the
  smallest lag wins. The whole trajectory is printed.
- NEAR_CLOSE_M. The 0.5 m band table at the final lag, each band the median over the eight
  matches. The value is the edge between two adjacent bands with the largest fall of that median
  rate, the first on a tie.
- P_NEAR_CLOSE. The median of the agreement in (0, NEAR_CLOSE_M] at the final lag.
- P_NEAR_FAR. The median of the agreement in (NEAR_CLOSE_M, NEAR_FAR_M] at the final lag (the
  C17 rule).
- OUT_MAX_RANGE_M. The heading distances of the exits of measurement 5 on the three Metrica
  matches, pooled. The value is the smallest multiple of 5 m that holds at least 90 % of the
  headings. An exit at a ball slower than HEADING_MIN_SPEED has no heading. The value with all
  exits as the denominator is printed for the record.

With `--lag L --near-close M`, `bucketb` runs no iteration. The appendix of
COMPLIANCE_PROTOCOL.md (precedence of PR #1, 8 Nov 2026) fixes NEAREST_LAG and NEAR_CLOSE_M from
EXTERNAL_SETTINGS.md. The command then gives NEAR_FAR_M by the pooled rule at lag L and
P_NEAR_FAR over (M, NEAR_FAR_M] at lag L. It prints the per-match pooled gaps for E = 1 to 5.

Deterministic. numpy only, plus the engine and tools modules of this repository.
"""

import argparse
import dataclasses
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (ROOT, os.path.join(ROOT, "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# The matches that this script never touches: the competition match (its labels are the public
# window, and its numbers are in PLAN 1.2) and the two sealed DFL matches (SPLITS.md).
REFUSED = ("BAR-ATH", "DFL-J03WQQ", "DFL-J03WR9")
ALIVE = ("H", "A")
# Lags of the report table (frames). The curve itself runs over CURVE_LAGS.
REPORT_LAGS = (0, 5, 10, 15, 20, 25, 30, 40)
CURVE_LAGS = tuple(range(-10, 46))
# Distance bands of the nearest player: (name, exclusive lower bound, inclusive upper bound).
BANDS = (("all", -1.0, np.inf), ("le0.5", -1.0, 0.5), ("le1.0", -1.0, 1.0), ("le1.5", -1.0, 1.5),
         ("1-1.5", 1.0, 1.5), ("1.5-2", 1.5, 2.0), ("2-3", 2.0, 3.0), ("3-4", 3.0, 4.0),
         ("4-5", 4.0, 5.0), ("5-8", 5.0, 8.0), ("1-3", 1.0, 3.0), ("3-5", 3.0, 5.0),
         ("1-5", 1.0, 5.0), ("le5", -1.0, 5.0), ("gt5", 5.0, np.inf))
# The plateau of the lag curve holds the lags in 0..45 whose rate is within this of the maximum.
PLATEAU_TOL = 0.005
# The flip lead: look from this many frames before the label, need this many ok frames in a row,
# and walk back for at most this many frames.
FLIP_START = -25
FLIP_RUN = 5
FLIP_BACK = 100
# Horizons of the confirmation test, seconds.
HORIZONS_S = (2.0, 3.0, 4.0, 6.0, 8.0, 10.0)
# A pursuit candidate belongs to a reference onset when its onset lies this close to it.
OWN_TOL = 50
# The look-back search of measurement 3 stops after this many frames.
KICK_BACK_MAX = 100
# Measurement 4: window around a hand-over for the dv3 peak, cap of the walk-back, rise window.
PEAK_WINDOW = 25
WALK_CAP = 60
RISE_WINDOW = 30
RELEASE_KS = (5, 9, 12, 15, 20, 30)
# Measurement 5: an exit restarts within this distance of a pitch line, and a loss is near an
# onset when the lag is at most this many frames (10 s).
EXIT_LINE_M = 2.0
LOSS_NEAR = 250
# Sigma in frames of the ATD ball kernel (tools/atd_view.py, RESEARCH-2 M13).
SMOOTH_SIGMA = 9.5
# The band-limited lag curve of the bucket b rules (COMPLIANCE_PROTOCOL.md, grids.json): bands of
# 0.5 m from 0 to 8 m and one band beyond 8 m, over the lags 0 to 45. Any band with edges on the
# 0.5 m grid is a sum of these bands (band_pair).
FINE_STEP_M = 0.5
FINE_MAX_M = 8.0
FINE_LAGS = tuple(range(0, 46))
FINE_BANDS = tuple(
    ("{:.1f}-{:.1f}".format(i * FINE_STEP_M, (i + 1) * FINE_STEP_M),
     -1.0 if i == 0 else i * FINE_STEP_M, (i + 1) * FINE_STEP_M)
    for i in range(int(round(FINE_MAX_M / FINE_STEP_M)))) + (("gt8.0", FINE_MAX_M, np.inf),)
# Measurement 5: the heading distances of the exits are counted at these multiples of 5 m.
COVER_STEPS_M = tuple(range(5, 55, 5))
# The bucket b rules: the open references, the tolerance of the NEAR_FAR_M rule, the share of
# the OUT_MAX_RANGE_M rule.
OPEN_REFERENCES = ("DFL-J03WMX", "DFL-J03WN1", "DFL-J03WOH", "DFL-J03WOY", "DFL-J03WPY",
                   "METRICA-S1", "METRICA-S2", "METRICA-S3")
FAR_TOL = 0.03
COVER_SHARE = 0.9


# -- small helpers ----------------------------------------------------------------------------

def read_labels(path, n_frames):
    """'<U1' (n_frames,): the labels of a `frame,label` CSV. Missing frames are X."""
    import pandas as pd

    table = pd.read_csv(path, dtype={"label": str})
    labels = np.full(int(n_frames), "X", dtype="<U1")
    frames = table["frame"].to_numpy(dtype=np.int64)
    keep = (frames >= 0) & (frames < n_frames)
    labels[frames[keep]] = table["label"].to_numpy()[keep]
    return labels


def owner_codes(labels):
    """int8 (n,): 0 for H, 1 for A, -1 for any other label."""
    labels = np.asarray(labels)
    return np.where(labels == "H", 0, np.where(labels == "A", 1, -1)).astype(np.int8)


def quantiles(values):
    """{n, min, p10, p25, p50, p75, p90, max} of the finite values, None for an empty input."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    keys = ("min", "p10", "p25", "p50", "p75", "p90", "max")
    if values.size == 0:
        return dict([("n", 0)] + [(key, None) for key in keys])
    q = np.percentile(values, [0, 10, 25, 50, 75, 90, 100])
    return dict([("n", int(values.size))] + [(key, float(v)) for key, v in zip(keys, q)])


def share(count, total):
    """count / total as a float, None when total is 0."""
    return float(count) / float(total) if total else None


def log_odds(p):
    """log(p / (1 - p)), None for p outside (0, 1) or None."""
    if p is None or not 0.0 < p < 1.0:
        return None
    return float(np.log(p / (1.0 - p)))


def reference_onsets(labels):
    """int64: the frames t with label D and an alive label at t - 1."""
    labels = np.asarray(labels)
    alive = (labels == "H") | (labels == "A")
    return np.flatnonzero((labels[1:] == "D") & alive[:-1]) + 1


def reference_handovers(labels):
    """int64: the frames t of an alive label that differs from the alive label at t - 1."""
    labels = np.asarray(labels)
    alive = (labels == "H") | (labels == "A")
    return np.flatnonzero(alive[1:] & alive[:-1] & (labels[1:] != labels[:-1])) + 1


def reference_possession_starts(labels):
    """[(t, team, kind)]: frames where an alive label starts, kind "handover" or "restart"."""
    labels = np.asarray(labels)
    out = []
    for t in np.flatnonzero((labels[1:] != labels[:-1]) & ((labels[1:] == "H") |
                                                          (labels[1:] == "A"))) + 1:
        if labels[t - 1] in ALIVE:
            out.append((int(t), str(labels[t]), "handover"))
        elif labels[t - 1] == "D":
            out.append((int(t), str(labels[t]), "restart"))
    return out


# -- measurement 1: the nearest player -----------------------------------------------------------

def nearest_agreement(owner, nearest_team, dmin, usable, lags, bands=BANDS):
    """{lag: {band: [agree, n]}}: the agreement of the lagged nearest team with the owner.

    owner         int8 (n,) 0 H, 1 A, -1 other (the reference label)
    nearest_team  int8 (n,) 0 H, 1 A, -1 none
    dmin          float (n,) distance of the nearest player
    usable        bool (n,) the frames that can be a source
    The pair is (owner[t], nearest_team[t - lag]) for every t with an alive owner and a usable
    source frame. A band holds the pairs whose source distance lies in (low, high].
    """
    owner = np.asarray(owner)
    nearest_team = np.asarray(nearest_team)
    dmin = np.asarray(dmin, dtype=np.float64)
    usable = np.asarray(usable, dtype=bool)
    n = owner.shape[0]
    out = {}
    for lag in lags:
        lag = int(lag)
        if lag >= 0:
            own, team, dist, ok = owner[lag:], nearest_team[:n - lag], dmin[:n - lag], \
                usable[:n - lag]
        else:
            k = -lag
            own, team, dist, ok = owner[:n - k], nearest_team[k:], dmin[k:], usable[k:]
        valid = (own >= 0) & ok
        agree = team == own
        row = {}
        for name, low, high in bands:
            with np.errstate(invalid="ignore"):
                member = valid & (dist > low) & (dist <= high)
            row[name] = [int((member & agree).sum()), int(member.sum())]
        out[str(lag)] = row
    return out


def best_lag(curve, band):
    """The lag with the highest agreement rate in a band (the first on a tie), and that rate."""
    best = (None, -1.0)
    for lag in sorted(curve, key=int):
        agree, total = curve[lag][band]
        if total and agree / float(total) > best[1]:
            best = (int(lag), agree / float(total))
    return best


def plateau(curve, band, tol=PLATEAU_TOL, lo=0, hi=45):
    """(first lag, last lag, best lag, best rate, rate at 20): the plateau of one band.

    The plateau holds the lags from `lo` to `hi` whose rate is within `tol` of the best rate.
    The first and last lag are the ends of the span of those lags. The rate at lag 20 is
    included for the comparison with NEAREST_LAG.
    """
    rates = {}
    for lag in range(lo, hi + 1):
        agree, total = curve[str(lag)][band]
        if total:
            rates[lag] = agree / float(total)
    top_lag = max(sorted(rates), key=lambda k: rates[k])
    inside = [k for k in rates if rates[k] >= rates[top_lag] - tol]
    return min(inside), max(inside), top_lag, rates[top_lag], rates.get(20)


def flip_leads(labels, nearest_team, usable):
    """{"handover": [lead], "restart": [lead]}: the flip lead of each possession start.

    See the module docstring. The lead is None when no run of FLIP_RUN ok frames starts from
    t + FLIP_START to t + 100.
    """
    labels = np.asarray(labels)
    n = labels.shape[0]
    team_code = {"H": 0, "A": 1}
    leads = {"handover": [], "restart": []}
    for t, team, kind in reference_possession_starts(labels):
        ok = np.asarray(usable, dtype=bool) & (np.asarray(nearest_team) == team_code[team])
        hit = None
        for k in range(FLIP_START, 100):
            lo = t + k
            if lo < 0 or lo + FLIP_RUN > n:
                continue
            if ok[lo:lo + FLIP_RUN].all():
                hit = lo
                break
        if hit is None:
            leads[kind].append(None)
            continue
        first = hit
        floor = max(0, t - FLIP_BACK)
        while first - 1 >= floor and ok[first - 1]:
            first -= 1
        leads[kind].append(int(t - first))
    return leads


def lead_summary(leads):
    """Quantiles of the flip leads and the shares used in the report."""
    found = [v for v in leads if v is not None]
    return {"n": len(leads), "found": len(found), "q": quantiles(found),
            "share_14_25": share(sum(1 for v in found if 14 <= v <= 25), len(leads)),
            "share_early": share(sum(1 for v in found if v > 0), len(leads)),
            "share_ge_100": share(sum(1 for v in found if v >= FLIP_BACK), len(leads))}


# -- measurement 2: the collective stop after an onset ---------------------------------------------

def stop_delays(onsets, stop_starts, stop_stops):
    """float64 (len(onsets),): max(0, start - t) of the first stop run that ends after t, else inf."""
    onsets = np.asarray(onsets, dtype=np.int64)
    stop_starts = np.asarray(stop_starts, dtype=np.int64)
    stop_stops = np.asarray(stop_stops, dtype=np.int64)
    out = np.full(onsets.shape[0], np.inf)
    if stop_starts.size == 0:
        return out
    idx = np.searchsorted(stop_stops, onsets, side="right")
    has = idx < stop_starts.size
    out[has] = np.maximum(0, stop_starts[idx[has]] - onsets[has])
    return out


def false_confirm_rate(labels, stop_starts, stop_stops, horizon):
    """(confirmed, eligible): alive frames that pass the confirmation test, and all that qualify.

    A frame t qualifies when t .. t + horizon holds only alive labels. It passes when a stop run
    starts at or before t + horizon and ends after t.
    """
    labels = np.asarray(labels)
    n = labels.shape[0]
    alive = (labels == "H") | (labels == "A")
    # frames until the next non-alive frame (n - t for none)
    until = np.zeros(n, dtype=np.int64)
    nxt = n
    for t in range(n - 1, -1, -1):
        nxt = t if not alive[t] else nxt
        until[t] = nxt - t
    eligible = alive & (until > horizon)
    t_idx = np.flatnonzero(eligible)
    stop_starts = np.asarray(stop_starts, dtype=np.int64)
    stop_stops = np.asarray(stop_stops, dtype=np.int64)
    if stop_starts.size == 0:
        return 0, int(t_idx.size)
    idx = np.searchsorted(stop_stops, t_idx, side="right")
    has = idx < stop_starts.size
    passed = np.zeros(t_idx.shape[0], dtype=bool)
    passed[has] = stop_starts[idx[has]] <= t_idx[has] + horizon
    return int(passed.sum()), int(t_idx.size)


# -- measurement 3: the pursuit stop and the kicking team ------------------------------------------

def pursuit_candidates(ball, feats, cfg, horizon, chase):
    """list of dicts: the pursuit candidates before the collective-stop confirmation.

    The loop is the one of engine.deadball._pursuit_onsets, with a row for every candidate that
    passes all tests but the confirmation. Each row holds start, onset, kick ("H" or "A"),
    mean_h and mean_a (the mean nearest distances over the look-back).
    """
    from engine import deadball as db

    rows = []
    speed = ball["speed"]
    near = feats["meanv_near"]
    n = speed.shape[0]
    fast = ball["visible"] & db._above(speed, cfg.PURSUIT_MIN_BALL_SPEED)
    starts, stops = db._runs(fast)
    for start, stop in zip(starts, stops):
        start, stop = int(start), int(stop)
        lo = max(0, start - int(cfg.PURSUIT_KICK_LOOKBACK))
        if not ball["visible"][lo:start].all() or start == 0:
            continue
        peak = db._first_peak(speed, start, stop)
        before = near[lo:start + 1]
        if not np.isfinite(before).any() or np.nanmax(before) < cfg.PURSUIT_STOP_SPEED:
            continue
        with np.errstate(invalid="ignore"):
            mean_h = np.nanmean(feats["dH"][lo:start + 1]) if np.isfinite(
                feats["dH"][lo:start + 1]).any() else np.inf
            mean_a = np.nanmean(feats["dA"][lo:start + 1]) if np.isfinite(
                feats["dA"][lo:start + 1]).any() else np.inf
        if not np.isfinite(min(mean_h, mean_a)):
            continue
        if abs(mean_h - mean_a) < cfg.CONTESTED_MARGIN_M:
            continue
        d_kick = feats["dH"] if mean_h <= mean_a else feats["dA"]
        end = min(n, start + horizon)
        slow = np.flatnonzero(db._below(near[start:end], cfg.PURSUIT_STOP_SPEED))
        if slow.size == 0:
            continue
        drop = start + int(slow[0])
        descent = drop
        while descent - 1 >= 0 and np.isfinite(near[descent - 1]) and \
                near[descent - 1] > near[descent]:
            descent -= 1
        onset = max(peak, descent)
        if not (np.isfinite(d_kick[onset]) and np.isfinite(d_kick[start])
                and d_kick[onset] > d_kick[start]):
            continue
        after = d_kick[onset:min(n, onset + chase)]
        if np.isfinite(after).any() and np.nanmin(after) <= cfg.R_TOUCH:
            continue
        rows.append({"start": start, "onset": int(onset), "kick": "H" if mean_h <= mean_a else "A",
                     "mean_h": float(mean_h), "mean_a": float(mean_a)})
    return rows


def engine_pursuit_onsets(ball, feats, cfg, horizon):
    """Sorted onsets of engine.deadball._pursuit_onsets when the confirmation always holds."""
    from engine import deadball as db

    n = ball["speed"].shape[0]
    rows = db._pursuit_onsets(ball, feats, cfg, np.array([0], dtype=np.int64),
                              np.array([n + 1], dtype=np.int64), horizon)
    return sorted(int(r[0]) for r in rows)


def is_owned(row, ref_onsets):
    """True when a pursuit candidate row lies within OWN_TOL frames of a reference onset."""
    ref_onsets = np.asarray(ref_onsets, dtype=np.int64)
    return bool(ref_onsets.size and (np.abs(ref_onsets - row["onset"]) <= OWN_TOL).any())


def lookback_offset(start, team, d_h, d_a, visible, margin, back_max=KICK_BACK_MAX):
    """Frames from `start` back to the latest frame where `team` is nearer by `margin`, or None."""
    for t in range(int(start), max(-1, int(start) - int(back_max) - 1), -1):
        if not visible[t] or not (np.isfinite(d_h[t]) and np.isfinite(d_a[t])):
            continue
        gap = d_a[t] - d_h[t] if team == "H" else d_h[t] - d_a[t]
        if gap >= margin:
            return int(start - t)
    return None


def kick_summary(rows, labels, feats, ball, cfg, ref_onsets):
    """The look-back statistics of the candidate rows, for all rows and the owned rows."""
    labels = np.asarray(labels)
    onsets = np.asarray(ref_onsets, dtype=np.int64)

    def owned(row):
        return is_owned(row, onsets)

    out = {}
    for name, keep in (("all", lambda r: True), ("owned", owned)):
        offsets, correct, total, never, mean_gap = [], 0, 0, 0, []
        for row in rows:
            if not keep(row):
                continue
            team = str(labels[row["start"]])
            if team not in ALIVE:
                continue
            total += 1
            correct += int(row["kick"] == team)
            mean_gap.append(abs(row["mean_h"] - row["mean_a"]))
            off = lookback_offset(row["start"], team, feats["dH"], feats["dA"], ball["visible"],
                                  cfg.CONTESTED_MARGIN_M)
            if off is None:
                never += 1
            else:
                offsets.append(off)
        out[name] = {"n": total, "engine_team_is_owner": correct,
                     "offset_q": quantiles(offsets), "none_within_100": never,
                     "share_le_10": share(sum(1 for v in offsets if v <= cfg.PURSUIT_KICK_LOOKBACK),
                                          total),
                     "share_zero": share(sum(1 for v in offsets if v == 0), total),
                     "mean_gap_q": quantiles(mean_gap)}
    return out


# -- measurement 4: the rise of a hand-over --------------------------------------------------------

def handover_rises(ball, feats, handovers, cfg):
    """dict: the walk-back and rise statistics of the hand-overs (see the module docstring).

    The walk-back length is peak - contact for the contact with a cap of WALK_CAP frames. A kick
    is a contact where the speed rises by more than DV_RELEASE within RISE_WINDOW frames. The
    rise length and size are reported for kicks only: the speed of a received ball falls. The
    gain length is the number of frames from the contact to the first speed gain above
    DV_RELEASE, the quantity that RELEASE_K must cover.
    """
    from engine import events

    speed = np.asarray(ball["speed"], dtype=np.float64)
    run_id = np.asarray(ball["run_id"], dtype=np.int64)
    dv = np.asarray(ball["dv_k"], dtype=np.float64)
    dmin = np.asarray(feats["dmin"], dtype=np.float64)
    n = dv.shape[0]
    k = int(cfg.DV_K)
    long_cfg = dataclasses.replace(cfg, TOUCH_ONSET_MAX_FRAMES=WALK_CAP)
    with np.errstate(invalid="ignore"):
        peaks = events._local_maxima(dv) & (dv > cfg.DV_TOUCH)
    walk, walk_kick, rise_len, rise_mag, gain_len, near_count = [], [], [], [], [], 0
    release = dict((kk, 0) for kk in RELEASE_KS)
    n_peak = 0
    for t in handovers:
        lo, hi = max(0, int(t) - PEAK_WINDOW), min(n, int(t) + PEAK_WINDOW + 1)
        cand = np.flatnonzero(peaks[lo:hi]) + lo
        if cand.size == 0:
            continue
        n_peak += 1
        peak = int(cand[int(np.argmax(dv[cand]))])
        contact = events.contact_frame(speed, run_id, peak, long_cfg)
        stop = min(n, peak + k + 1)
        seg = dmin[contact:stop]
        if not (np.isfinite(seg).any() and np.nanmin(seg) <= cfg.R_TOUCH):
            continue
        near_count += 1
        walk.append(peak - contact)
        window = np.arange(contact, min(n, contact + RISE_WINDOW + 1))
        values = speed[window[run_id[window] == run_id[contact]]]
        if values.size and np.isfinite(values).any():
            top = int(np.nanargmax(values))
            if values[top] - speed[contact] > cfg.DV_RELEASE:
                walk_kick.append(peak - contact)
                rise_len.append(top)
                rise_mag.append(float(values[top] - speed[contact]))
                with np.errstate(invalid="ignore"):
                    gained = np.flatnonzero(values - speed[contact] > cfg.DV_RELEASE)
                gain_len.append(int(gained[0]))
        shipped = events.contact_frame(speed, run_id, peak, cfg)
        for kk in RELEASE_KS:
            sub = dataclasses.replace(cfg, RELEASE_K=kk)
            rise = events._rise_after(speed, run_id, shipped, sub)
            release[kk] += int(np.isfinite(rise) and rise > cfg.DV_RELEASE)
    cap = int(cfg.TOUCH_ONSET_MAX_FRAMES)
    return {"handovers": int(len(handovers)), "with_peak": n_peak, "with_player": near_count,
            "walk_q": quantiles(walk), "walk_kick_q": quantiles(walk_kick), "cap": cap,
            "share_walk_gt_cap": share(sum(1 for v in walk if v > cap), len(walk)),
            "share_walk_kick_gt_cap": share(sum(1 for v in walk_kick if v > cap), len(walk_kick)),
            "walk_hist": dict((str(v), int(sum(1 for w in walk if w == v))) for v in range(0, 13)),
            "kicks": len(rise_len), "rise_len_q": quantiles(rise_len),
            "rise_mag_q": quantiles(rise_mag),
            "share_rise_len_le_k": share(sum(1 for v in rise_len if v <= cfg.RELEASE_K),
                                         len(rise_len)),
            "gain_len_q": quantiles(gain_len),
            "share_gain_len_le_k": share(sum(1 for v in gain_len if v <= cfg.RELEASE_K),
                                         len(gain_len)),
            "release_share": dict((str(kk), share(release[kk], near_count))
                                  for kk in RELEASE_KS)}


# -- measurement 5: the lost ball ------------------------------------------------------------------

def last_next_visible(visible):
    """(last, next): int64 arrays, the last visible frame at or before t and the first at or after."""
    visible = np.asarray(visible, dtype=bool)
    n = visible.shape[0]
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(visible, idx, -1))
    nxt = np.minimum.accumulate(np.where(visible, idx, n)[::-1])[::-1]
    return last, nxt


def out_lag_rows(ball, pitch, cfg, onsets):
    """(summary dict, near-loss onsets): measurement 5 (see the module docstring).

    For each reference onset t the loss frame f is the last frame of the visible run that holds
    t, or the last visible frame before t. The signed lag is t - f. It is negative when the ball
    stays visible after the onset. The loss is near when the lag is at most LOSS_NEAR frames in
    size. Only a near loss is a ball that goes missing at a stoppage.
    """
    from engine import deadball as db

    length, width = pitch
    visible = ball["visible"]
    n = visible.shape[0]
    idx = np.arange(n)
    last, nxt = last_next_visible(visible)
    first_missing = np.minimum.accumulate(np.where(~visible, idx, n)[::-1])[::-1]
    fps = float(ball["fps"])
    lags, heading, exit_heading = [], [], []
    serve_lags, serve_err, extrap_err = [], [], []
    far = 0
    inside_count, moving_count, exits, exits_in_range, exits_serving = 0, 0, 0, 0, 0
    visible_after = 0
    near_onsets = []
    for t in onsets:
        t = int(t)
        f = int(first_missing[t]) - 1 if visible[t] else int(last[t])
        if f < 0 or f >= n - 1:
            far += 1
            continue
        lag = t - f
        if abs(lag) > LOSS_NEAR:
            far += 1
            continue
        near_onsets.append(t)
        lags.append(lag)
        visible_after += int(lag < 0)
        x, y, speed = ball["x"][f], ball["y"][f], ball["speed"][f]
        inside = 0.0 <= x <= length and 0.0 <= y <= width
        inside_count += int(inside)
        dist = np.inf
        if inside and np.isfinite(speed) and speed >= cfg.HEADING_MIN_SPEED:
            moving_count += 1
            ux, uy = ball["vx"][f] / speed, ball["vy"][f] / speed
            dist = db._boundary_distance(x, y, ux, uy, pitch)
        in_range = bool(np.isfinite(dist) and dist <= cfg.OUT_MAX_RANGE_M)
        roll = bool(np.isfinite(dist) and speed * speed >= 2.0 * cfg.FRICTION_DECEL * dist)
        if np.isfinite(dist):
            heading.append(dist)
        if in_range and not roll:
            serve_lags.append(lag)
            serve_err.append(f + int(cfg.DELTA_OUT_LAG) - t)
        if in_range and roll:
            seconds = (speed - np.sqrt(speed * speed - 2.0 * cfg.FRICTION_DECEL * dist)) \
                / cfg.FRICTION_DECEL
            extrap_err.append(f + int(round(seconds * fps)) - t)
        g = int(nxt[f + 1])
        if inside and g < n:
            gx, gy = ball["x"][g], ball["y"][g]
            if min(gx, length - gx, gy, width - gy) <= EXIT_LINE_M:
                exits += 1
                exits_in_range += int(in_range)
                exits_serving += int(in_range and not roll)
                if np.isfinite(dist):
                    exit_heading.append(dist)
    near = len(near_onsets)
    return ({"onsets": int(len(onsets)), "near_loss": near, "far": far,
             "visible_after_onset": visible_after, "lag_q": quantiles(lags),
             "share_lag_ge_21": share(sum(1 for v in lags if v >= cfg.DELTA_OUT_LAG), near),
             "share_lag_le_0": share(sum(1 for v in lags if v <= 0), near),
             "inside": inside_count, "moving_inside": moving_count,
             "heading_q": quantiles(heading), "heading_in_range": sum(
                 1 for v in heading if v <= cfg.OUT_MAX_RANGE_M),
             "serving_n": len(serve_lags), "serving_lag_q": quantiles(serve_lags),
             "serving_err_q": quantiles(serve_err), "extrap_n": len(extrap_err),
             "extrap_err_q": quantiles(extrap_err), "exits": exits,
             "exits_in_range": exits_in_range, "exits_serving": exits_serving,
             "exit_heading_q": quantiles(exit_heading),
             "exit_heading_cover": dict((str(m), sum(1 for v in exit_heading if v <= m))
                                        for m in COVER_STEPS_M)},
            np.array(near_onsets, dtype=np.int64))


def crossing_offsets(ball, pitch, onsets):
    """list: t minus the first frame of the visible excursion beyond a line, for onsets beyond one."""
    from engine import deadball as db

    beyond = db.outside_pitch(ball, pitch, 0.0)
    out = []
    for t in onsets:
        t = int(t)
        if not beyond[t]:
            continue
        first = t
        while first - 1 >= 0 and beyond[first - 1]:
            first -= 1
        out.append(t - first)
    return out


# -- one match -------------------------------------------------------------------------------------

def load_match(game, data_dir, cfg, sigma=None):
    """(tactical, ball, feats): the arrays of the engine for one match, as engine_tactical gives.

    With `sigma` the ball track is smoothed with tools/atd_view.smooth_runs before the features.
    """
    from engine import data, tracks

    pitch = data.pitch_m(game)
    tactical = tracks.speed_from_positions(data.load_tactical(game, data_dir=data_dir), pitch)
    tactical, _, _ = tracks.guard_goalkeeper_swaps(tactical, data.graded_blocks(game), cfg, pitch)
    if sigma:
        import atd_view

        length, width = pitch
        bx = np.asarray(tactical["bx"], dtype=np.float64) * length
        by = np.asarray(tactical["by"], dtype=np.float64) * width
        vis = np.isfinite(bx) & np.isfinite(by)
        breaks = atd_view.block_breaks(bx.shape[0], game)
        tactical = dict(tactical)
        tactical["bx"] = atd_view.smooth_runs(bx, vis, breaks, sigma) / length
        tactical["by"] = atd_view.smooth_runs(by, vis, breaks, sigma) / width
    ball, feats, _ = tracks.prepare(tactical, cfg, pitch)
    return tactical, ball, feats


def measure(game, ref_path, data_dir=None, cfg=None):
    """All measurements of one match as a dict of summary statistics."""
    from engine import data, deadball as db
    from engine.config import Config

    if game in REFUSED:
        raise SystemExit("{} is refused: see the module docstring".format(game))
    cfg = cfg or Config.default()
    pitch = data.pitch_m(game)
    tactical, ball, feats = load_match(game, data_dir, cfg)
    n = int(tactical["n_frames"])
    labels = read_labels(ref_path, n)
    owner = owner_codes(labels)
    alive = owner >= 0
    fps = float(ball["fps"])
    ghost = db.ghost_mask(ball, feats, cfg)
    usable = ball["visible"] & np.isfinite(feats["dmin"]) & (feats["nearest_team"] >= 0) & ~ghost

    out = {"game": game, "n_frames": n, "alive_frames": int(alive.sum()),
           "alive_visible": int((alive & ball["visible"]).sum()), "ghost_frames": int(ghost.sum()),
           "fps": fps}
    curve = nearest_agreement(owner, feats["nearest_team"], feats["dmin"], usable, CURVE_LAGS)
    out["m1"] = {"curve": curve,
                 "best_lag": dict((band, best_lag(curve, band)[0])
                                  for band in ("all", "le1.0", "1-5")),
                 "plateau": dict((band, list(plateau(curve, band)))
                                 for band in ("all", "le1.0", "1-5")),
                 "flip": dict((kind, lead_summary(vals)) for kind, vals in
                              flip_leads(labels, feats["nearest_team"], usable).items()),
                 "fine": nearest_agreement(owner, feats["nearest_team"], feats["dmin"], usable,
                                           FINE_LAGS, FINE_BANDS)}

    out["m5"], loss_onsets = out_lag_rows(ball, pitch, cfg, reference_onsets(labels))
    out["m5"]["crossing_q"] = quantiles(crossing_offsets(ball, pitch, reference_onsets(labels)))

    horizon = db._frames(cfg.ONSET_CONFIRM_S, fps)
    chase = db._frames(cfg.COLLECTIVE_STOP_S, fps)
    starts, stops = db._stop_runs(feats, cfg, fps)
    onsets = reference_onsets(labels)
    rows = pursuit_candidates(ball, feats, cfg, horizon, chase)
    owned_onsets = np.array([t for t in onsets if any(abs(t - r["onset"]) <= OWN_TOL
                                                       for r in rows)], dtype=np.int64)
    delays = stop_delays(onsets, starts, stops)
    sweep = {}
    for seconds in HORIZONS_S:
        h = db._frames(seconds, fps)
        confirmed, eligible = false_confirm_rate(labels, starts, stops, h)
        sweep[str(seconds)] = {"recall": share(int((delays <= h).sum()), onsets.size),
                               "false_confirmed": confirmed, "false_eligible": eligible}

    def delay_group(group):
        d = stop_delays(group, starts, stops)
        return {"n": int(group.size), "delay_q": quantiles(d), "never": int((~np.isfinite(d)).sum()),
                "share_within": share(int((d <= horizon).sum()), group.size)}

    out["m2"] = {"onsets": int(onsets.size), "stop_runs": int(starts.size),
                 "delay_q": quantiles(delays), "never": int((~np.isfinite(delays)).sum()),
                 "inside": int((delays == 0).sum()),
                 "share_within": share(int((delays <= horizon).sum()), onsets.size),
                 "horizon_frames": int(horizon), "sweep": sweep,
                 "pursuit_owned": delay_group(owned_onsets), "near_loss": delay_group(loss_onsets)}

    engine_rows = engine_pursuit_onsets(ball, feats, cfg, horizon)
    conf = [bool(np.any((starts <= r["onset"] + horizon) & (stops > r["onset"]))) for r in rows]
    own = [is_owned(r, onsets) for r in rows]
    out["m3"] = {"candidates": len(rows), "confirmed": int(sum(conf)),
                 "owned": int(sum(own)),
                 "confirmed_owned": int(sum(1 for c, o in zip(conf, own) if c and o)),
                 "matches_engine": sorted(r["onset"] for r in rows) == engine_rows,
                 "summary": kick_summary(rows, labels, feats, ball, cfg, onsets)}

    handovers = reference_handovers(labels)
    out["m4"] = {"raw": handover_rises(ball, feats, handovers, cfg)}
    _, sball, sfeats = load_match(game, data_dir, cfg, sigma=SMOOTH_SIGMA)
    out["m4"]["smooth"] = handover_rises(sball, sfeats, handovers, cfg)
    return out


# -- output ----------------------------------------------------------------------------------------

def write_json(path, payload):
    """Write `payload` as sorted, indented JSON (numpy scalars as Python numbers)."""
    def convert(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        raise TypeError(type(value))
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, default=convert)
        handle.write("\n")


def _rate(pair):
    """agree / total of an [agree, total] pair, None for an empty band."""
    return share(pair[0], pair[1])


def _pc(value, digits=1):
    """A percentage string, '-' for None."""
    return "-" if value is None else "{:.{d}f}".format(100.0 * value, d=digits)


def _num(value, digits=1):
    """A number string, '-' for None."""
    return "-" if value is None else "{:.{d}f}".format(value, d=digits)


def _q(q, keys=("p50", "p90")):
    """'n / p50 / p90' for a quantiles dict."""
    return " / ".join([str(q["n"])] + [_num(q[k]) for k in keys])


def report(payloads):
    """The markdown tables of docs/plan/calibration/PROVENANCE_RECHECK.md as one string."""
    lines = []
    add = lines.append

    def table(title, header, rows):
        add("")
        add("**{}**".format(title))
        add("")
        add("| " + " | ".join(header) + " |")
        add("|" + "|".join("---" for _ in header) + "|")
        for row in rows:
            add("| " + " | ".join(str(c) for c in row) + " |")

    def seconds_of(p, q, key):
        return None if q[key] is None else q[key] / p["fps"]

    table("Frames", ["match", "frames", "alive", "alive and ball visible", "ghost frames"],
          [[p["game"], p["n_frames"], p["alive_frames"], p["alive_visible"], p["ghost_frames"]]
           for p in payloads])
    bands = ("all", "le0.5", "le1.0", "le1.5", "1-3", "3-5", "1-5", "gt5")
    names = ("all", "<=0.5 m", "<=1.0 m", "<=1.5 m", "1-3 m", "3-5 m", "1-5 m", ">5 m")
    for lag in ("0", "20"):
        table("Measurement 1: agreement of the nearest team with the owner at lag " + lag +
              " (percent, n)", ["match"] + list(names),
              [[p["game"]] + ["{} ({})".format(_pc(_rate(p["m1"]["curve"][lag][b])),
                                               p["m1"]["curve"][lag][b][1]) for b in bands]
               for p in payloads])
    fine = ("le1.0", "1-1.5", "1.5-2", "2-3", "3-4", "4-5", "5-8")
    table("Measurement 1: agreement by distance band at lag 20 (percent)",
          ["match", "<=1", "1-1.5", "1.5-2", "2-3", "3-4", "4-5", "5-8"],
          [[p["game"]] + [_pc(_rate(p["m1"]["curve"]["20"][b])) for b in fine] for p in payloads])
    table("Measurement 1: agreement by lag, all distances / within 1.0 m (percent)",
          ["match"] + ["lag {}".format(lag) for lag in REPORT_LAGS],
          [[p["game"]] + ["{} / {}".format(_pc(_rate(p["m1"]["curve"][str(lag)]["all"])),
                                           _pc(_rate(p["m1"]["curve"][str(lag)]["le1.0"])))
                          for lag in REPORT_LAGS] for p in payloads])
    rows = []
    for p in payloads:
        row = [p["game"]]
        for band in ("all", "le1.0"):
            first, last, top, rate, at20 = p["m1"]["plateau"][band]
            row.append("{}..{}, best {} ({}), lag 20 {}".format(first, last, top, _pc(rate),
                                                               _pc(at20)))
        row.append("{} / {}".format(_pc(_rate(p["m1"]["curve"]["-10"]["all"])),
                                    _pc(_rate(p["m1"]["curve"]["-5"]["all"]))))
        rows.append(row)
    table("Measurement 1: plateau of the lag curve (lags within 0.5 points of the best, 0 to 45)",
          ["match", "all distances", "within 1.0 m", "all, lag -10 / -5"], rows)
    rows = []
    for p in payloads:
        row = [p["game"]]
        for lag in ("0", "20"):
            c = p["m1"]["curve"][lag]
            row.append("{} / {} / {}".format(_num(log_odds(_rate(c["le1.0"])), 2),
                                             _num(log_odds(_rate(c["1-5"])), 2),
                                             _num(log_odds(_rate(c["3-5"])), 2)))
        rows.append(row)
    table("Measurement 1: log-odds within 1.0 m / from 1 to 5 m / from 3 to 5 m (shipped 2.27 / "
          "0.41 / 0.41)", ["match", "lag 0", "lag 20"], rows)
    rows = []
    for p in payloads:
        for kind in ("handover", "restart"):
            f = p["m1"]["flip"][kind]
            rows.append([p["game"], kind, f["n"], f["found"], _num(f["q"]["p50"], 0),
                         _num(f["q"]["p25"], 0), _num(f["q"]["p75"], 0),
                         _pc(f["share_14_25"], 0), _pc(f["share_early"], 0),
                         _pc(f["share_ge_100"], 0)])
    table("Measurement 1: flip lead of the nearest team before a possession start (frames)",
          ["match", "kind", "starts", "found", "p50", "p25", "p75", "% in 14-25", "% early (>0)",
           "% capped (>=100)"], rows)

    table("Measurement 2: delay from a reference D onset to a collective stop",
          ["match", "onsets", "stop runs", "inside a run", "never", "p50 s", "p90 s",
           "% within 4 s"],
          [[p["game"], p["m2"]["onsets"], p["m2"]["stop_runs"], p["m2"]["inside"],
            p["m2"]["never"], _num(seconds_of(p, p["m2"]["delay_q"], "p50")),
            _num(seconds_of(p, p["m2"]["delay_q"], "p90")), _pc(p["m2"]["share_within"])]
           for p in payloads])
    rows = []
    for p in payloads:
        for name, key in (("pursuit-owned", "pursuit_owned"), ("near loss", "near_loss")):
            g = p["m2"][key]
            rows.append([p["game"], name, g["n"], g["never"],
                         _num(seconds_of(p, g["delay_q"], "p50")),
                         _num(seconds_of(p, g["delay_q"], "p90")), _pc(g["share_within"])])
    table("Measurement 2: the same delay for the onsets the engine confirms",
          ["match", "group", "onsets", "never", "p50 s", "p90 s", "% within 4 s"], rows)
    rows = []
    for p in payloads:
        row = [p["game"]]
        for seconds in HORIZONS_S:
            s = p["m2"]["sweep"][str(seconds)]
            row.append("{} / {}".format(_pc(s["recall"], 0),
                                        _pc(share(s["false_confirmed"], s["false_eligible"]))))
        rows.append(row)
    table("Measurement 2: onset recall / alive-frame false rate (percent) by horizon",
          ["match"] + ["{:g} s".format(h) for h in HORIZONS_S], rows)

    rows = []
    for p in payloads:
        m3 = p["m3"]
        for name in ("all", "owned"):
            s = m3["summary"][name]
            rows.append([p["game"], name, m3["candidates"] if name == "all" else m3["owned"],
                         m3["confirmed"] if name == "all" else m3["confirmed_owned"],
                         s["n"], _pc(share(s["engine_team_is_owner"], s["n"]), 0),
                         _num(s["offset_q"]["p50"], 0), _num(s["offset_q"]["p90"], 0),
                         _pc(s["share_zero"], 0), _pc(s["share_le_10"], 0),
                         s["none_within_100"], "yes" if m3["matches_engine"] else "NO"])
    table("Measurement 3: look-back offset of the kicking team (frames)",
          ["match", "group", "candidates", "confirmed", "n with owner", "engine team = owner",
           "p50", "p90", "% at 0", "% within 10", "none in 100", "replica = engine"], rows)

    for view in ("raw", "smooth"):
        table("Measurement 4, " + view + " ball: walk-back of the contact at reference hand-overs",
              ["match", "hand-overs", "with peak", "with player", "walk p50", "walk p90",
               "% walk > 9", "kicks", "kick walk p50", "kick walk p90", "% kick walk > 9"],
              [[p["game"], p["m4"][view]["handovers"], p["m4"][view]["with_peak"],
                p["m4"][view]["with_player"], _num(p["m4"][view]["walk_q"]["p50"], 0),
                _num(p["m4"][view]["walk_q"]["p90"], 0),
                _pc(p["m4"][view]["share_walk_gt_cap"], 0), p["m4"][view]["kicks"],
                _num(p["m4"][view]["walk_kick_q"]["p50"], 0),
                _num(p["m4"][view]["walk_kick_q"]["p90"], 0),
                _pc(p["m4"][view]["share_walk_kick_gt_cap"], 0)] for p in payloads])
        table("Measurement 4, " + view + " ball: rise of the kicks (speed gain above 3 m/s)",
              ["match", "kicks", "peak-speed length p50", "p90", "gain length p50", "p90",
               "% gain length <= 9", "rise m/s p50"],
              [[p["game"], p["m4"][view]["kicks"], _num(p["m4"][view]["rise_len_q"]["p50"], 0),
                _num(p["m4"][view]["rise_len_q"]["p90"], 0),
                _num(p["m4"][view]["gain_len_q"]["p50"], 0),
                _num(p["m4"][view]["gain_len_q"]["p90"], 0),
                _pc(p["m4"][view]["share_gain_len_le_k"], 0),
                _num(p["m4"][view]["rise_mag_q"]["p50"])] for p in payloads])
        table("Measurement 4, " + view + " ball: share of contacts that are releases (percent) "
              "by RELEASE_K, and K 9 as a share of K 30",
              ["match"] + ["K {}".format(k) for k in RELEASE_KS] + ["K 9 / K 30"],
              [[p["game"]] + [_pc(p["m4"][view]["release_share"][str(k)], 0) for k in RELEASE_KS] +
               [_pc(share(p["m4"][view]["release_share"]["9"],
                          p["m4"][view]["release_share"]["30"]), 0)] for p in payloads])

    table("Measurement 5: the ball loss near a reference onset (lag = onset minus last visible "
          "frame, frames)",
          ["match", "onsets", "near loss", "ball visible after onset", "lag p10", "lag p50",
           "lag p90", "% lag <= 0", "% lag >= 21"],
          [[p["game"], p["m5"]["onsets"], p["m5"]["near_loss"], p["m5"]["visible_after_onset"],
            _num(p["m5"]["lag_q"]["p10"], 0), _num(p["m5"]["lag_q"]["p50"], 0),
            _num(p["m5"]["lag_q"]["p90"], 0), _pc(p["m5"]["share_lag_le_0"], 0),
            _pc(p["m5"]["share_lag_ge_21"], 0)] for p in payloads])
    table("Measurement 5: geometry at the last visible frame, and the two rules",
          ["match", "near loss", "inside pitch", "moving inside", "heading <= 20 m",
           "lag-rule onsets", "error of f + 21 p50 / p90", "extrapolated onsets",
           "error of the roll p50 / p90", "exits", "exits <= 20 m", "exit heading p50 / p90 m"],
          [[p["game"], p["m5"]["near_loss"], p["m5"]["inside"], p["m5"]["moving_inside"],
            p["m5"]["heading_in_range"], p["m5"]["serving_n"],
            "{} / {}".format(_num(p["m5"]["serving_err_q"]["p50"], 0),
                             _num(p["m5"]["serving_err_q"]["p90"], 0)), p["m5"]["extrap_n"],
            "{} / {}".format(_num(p["m5"]["extrap_err_q"]["p50"], 0),
                             _num(p["m5"]["extrap_err_q"]["p90"], 0)), p["m5"]["exits"],
            p["m5"]["exits_in_range"],
            "{} / {}".format(_num(p["m5"]["exit_heading_q"]["p50"]),
                             _num(p["m5"]["exit_heading_q"]["p90"]))] for p in payloads])
    table("Measurement 5: label offset after a seen crossing (frames, onset minus first frame "
          "beyond the line)", ["match", "n", "p50", "p90"],
          [[p["game"], p["m5"]["crossing_q"]["n"], _num(p["m5"]["crossing_q"]["p50"], 0),
            _num(p["m5"]["crossing_q"]["p90"], 0)] for p in payloads])
    return "\n".join(lines) + "\n"


# -- bucket b: the pre-registered rules of grids.json -----------------------------------------------

def band_pair(row, low, high):
    """[agree, n] of the fine bands of one lag row that lie inside (low, high].

    `row` is one lag of the "fine" curve. The edges must lie on the FINE_STEP_M grid. `high` can
    be np.inf. A low edge of 0 includes a distance of 0.
    """
    agree = total = 0
    for name, b_low, b_high in FINE_BANDS:
        edge = 0.0 if b_low < 0 else b_low
        if edge >= low - 1e-9 and b_high <= high + 1e-9:
            agree += row[name][0]
            total += row[name][1]
    return [agree, total]


def band_rate(fine, lag, low, high):
    """Agreement rate of the band (low, high] at one lag, None for an empty band."""
    return _rate(band_pair(fine[str(lag)], low, high))


def argmax_lag(fine, low, high, lags=FINE_LAGS):
    """The lag with the highest rate in the band (low, high] (the first on a tie), or None."""
    best_lag, best_rate = None, -1.0
    for lag in lags:
        rate = band_rate(fine, lag, low, high)
        if rate is not None and rate > best_rate:
            best_lag, best_rate = int(lag), rate
    return best_lag


def median_half_up(values):
    """Median of integers, rounded to a whole number (a half rounds up)."""
    return int(np.floor(np.median(np.asarray(values, dtype=np.float64)) + 0.5))


def largest_drop_edge(rates, step=FINE_STEP_M):
    """(edge, drop): the edge between bands i and i + 1 with the largest rates[i] - rates[i + 1].

    rates[i] is the rate of the band [i step, (i + 1) step]. The edge is (i + 1) step. The first
    edge wins a tie. None values are skipped.
    """
    best = (None, -np.inf)
    for i in range(len(rates) - 1):
        if rates[i] is None or rates[i + 1] is None:
            continue
        drop = rates[i] - rates[i + 1]
        if drop > best[1]:
            best = ((i + 1) * step, drop)
    return best


def far_edge(fines, lag, tol=FAR_TOL, top=FINE_MAX_M, ref=(5.0, 8.0)):
    """Smallest whole edge E such that, on every match, the pooled agreement over (E, top] is
    within tol of the agreement over the reference band (5, 8] m (advisor ruling, 8 Nov 2026).

    fines  {match: fine curve}. E runs from 1 to the lower edge of the reference band, where the
    test holds by construction. A match with an empty band fails the test.
    """
    for edge in range(1, int(ref[0]) + 1):
        ok = True
        for fine in fines.values():
            pooled = band_rate(fine, lag, float(edge), top)
            base = band_rate(fine, lag, ref[0], ref[1])
            if pooled is None or base is None or abs(pooled - base) > tol + 1e-12:
                ok = False
                break
        if ok:
            return edge
    return int(ref[0])


def median_best_lag(fines, far_m):
    """(median lag, {match: lag}): the per-match argmax within (0, far_m], a half rounded up.

    A match with an empty band takes the argmax of the all-distance curve (the fallback of the
    rule).
    """
    lags = {}
    for game, fine in fines.items():
        lag = argmax_lag(fine, 0.0, far_m)
        lags[game] = lag if lag is not None else argmax_lag(fine, 0.0, np.inf)
    return median_half_up([lags[g] for g in sorted(lags)]), lags


def iterate_lag_far(fines, start_lag=0, max_steps=50):
    """(lag, far, trajectory, cycle): the NEAREST_LAG and NEAR_FAR_M fixed point.

    From start_lag: NEAR_FAR_M = far_edge at the lag, then the lag = median_best_lag within that
    NEAR_FAR_M, then again, until neither changes (advisor ruling, 8 Nov 2026). The trajectory
    lists the states (lag, far, per-match lags). When a state repeats before a fixed point, the
    states of the cycle are returned in `cycle` and the result is the cycle state with the
    smallest lag.
    """
    lag = int(start_lag)
    far = far_edge(fines, lag)
    trajectory = [{"lag": lag, "far": far, "per_match": None}]
    seen = [(lag, far)]
    for _ in range(max_steps):
        new_lag, per_match = median_best_lag(fines, float(far))
        new_far = far_edge(fines, new_lag)
        trajectory.append({"lag": new_lag, "far": new_far, "per_match": per_match})
        if (new_lag, new_far) == (lag, far):
            return lag, far, trajectory, None
        if (new_lag, new_far) in seen:
            cycle = seen[seen.index((new_lag, new_far)):]
            lag, far = min(cycle)
            return lag, far, trajectory, cycle
        seen.append((new_lag, new_far))
        lag, far = new_lag, new_far
    raise RuntimeError("no fixed point or cycle within {} steps".format(max_steps))


def cover_multiple(cover, denominator, share_min=COVER_SHARE):
    """Smallest step m with cover[m] / denominator >= share_min, or None."""
    for step in sorted(cover, key=float):
        if denominator and cover[step] / float(denominator) >= share_min:
            return int(float(step))
    return None


def select_bucket_b(payloads):
    """The bucket b values of the nearest-player fields and OUT_MAX_RANGE_M, with the evidence.

    Order of the advisor ruling (8 Nov 2026): NEAREST_LAG and NEAR_FAR_M by the fixed-point
    iteration from lag 0, then at the final lag NEAR_CLOSE_M, P_NEAR_CLOSE within NEAR_CLOSE_M
    and P_NEAR_FAR over (NEAR_CLOSE_M, NEAR_FAR_M].
    """
    games = [p["game"] for p in payloads]
    fine = dict((p["game"], p["m1"]["fine"]) for p in payloads)
    out = {"games": games}

    lag, far, trajectory, cycle = iterate_lag_far(fine)
    out["trajectory"] = trajectory
    out["cycle"] = cycle
    out["NEAREST_LAG"] = lag
    out["NEAR_FAR_M"] = float(far)
    out["lag_all_distance"] = dict((g, argmax_lag(fine[g], 0.0, np.inf)) for g in games)
    out["lag_per_match"] = dict((g, argmax_lag(fine[g], 0.0, float(far))) for g in games)
    out["pooled_gap"] = dict(
        (str(e), max(abs(band_rate(fine[g], lag, float(e), FINE_MAX_M)
                         - band_rate(fine[g], lag, 5.0, 8.0)) for g in games))
        for e in range(1, 6))

    n_fine = int(round(FINE_MAX_M / FINE_STEP_M))
    half = dict((g, [band_rate(fine[g], lag, i * FINE_STEP_M, (i + 1) * FINE_STEP_M)
                     for i in range(n_fine)]) for g in games)
    median_half = [float(np.median([half[g][i] for g in games])) for i in range(n_fine)]
    edge, drop = largest_drop_edge(median_half)
    out["half_m_table"] = half
    out["half_m_median"] = median_half
    out["half_m_edge_per_match"] = dict((g, largest_drop_edge(half[g])[0]) for g in games)
    out["NEAR_CLOSE_M"] = edge
    out["near_close_drop"] = drop

    close = dict((g, band_rate(fine[g], lag, 0.0, edge)) for g in games)
    out["close_per_match"] = close
    out["P_NEAR_CLOSE"] = round(float(np.median([close[g] for g in games])), 3)

    out["one_m_table"] = dict((g, [band_rate(fine[g], lag, float(k), float(k + 1))
                                   for k in range(int(FINE_MAX_M))]) for g in games)
    out["ref_5_8"] = dict((g, band_rate(fine[g], lag, 5.0, 8.0)) for g in games)
    far_rates = dict((g, band_rate(fine[g], lag, edge, float(far))) for g in games)
    out["far_per_match"] = far_rates
    out["P_NEAR_FAR"] = round(float(np.median([far_rates[g] for g in games])), 3)

    metrica = [p for p in payloads if p["game"].startswith("METRICA-")]
    if metrica:
        cover = {}
        for p in metrica:
            for step, count in p["m5"]["exit_heading_cover"].items():
                cover[step] = cover.get(step, 0) + count
        headings = sum(p["m5"]["exit_heading_q"]["n"] for p in metrica)
        exits = sum(p["m5"]["exits"] for p in metrica)
        out["exit_cover"] = dict(sorted(cover.items(), key=lambda kv: float(kv[0])))
        out["exit_headings"] = headings
        out["exits"] = exits
        out["OUT_MAX_RANGE_M"] = cover_multiple(cover, headings)
        out["OUT_MAX_RANGE_M_over_exits"] = cover_multiple(cover, exits)
    return out


def select_fixed_lag(payloads, lag, near_close):
    """NEAR_FAR_M and P_NEAR_FAR at a lag and a NEAR_CLOSE_M fixed by another protocol.

    COMPLIANCE_PROTOCOL.md appendix, precedence of PR #1 (8 Nov 2026): NEAREST_LAG and
    NEAR_CLOSE_M come from EXTERNAL_SETTINGS.md. NEAR_FAR_M is the pooled far_edge() rule at
    that lag. P_NEAR_FAR is the median over the eight matches of the agreement over
    (near_close, NEAR_FAR_M] at that lag, rounded to 0.001. No iteration runs.
    """
    games = [p["game"] for p in payloads]
    fine = dict((p["game"], p["m1"]["fine"]) for p in payloads)
    far = far_edge(fine, int(lag))
    out = {"games": games, "NEAREST_LAG": int(lag), "NEAR_CLOSE_M": float(near_close),
           "NEAR_FAR_M": float(far)}
    out["pooled_gap"] = dict(
        (str(e), max(abs(band_rate(fine[g], lag, float(e), FINE_MAX_M)
                         - band_rate(fine[g], lag, 5.0, 8.0)) for g in games))
        for e in range(1, 6))
    out["pooled_gap_per_match"] = dict(
        (str(e), dict((g, band_rate(fine[g], lag, float(e), FINE_MAX_M)
                       - band_rate(fine[g], lag, 5.0, 8.0)) for g in games))
        for e in range(1, 6))
    out["one_m_table"] = dict((g, [band_rate(fine[g], lag, float(k), float(k + 1))
                                   for k in range(int(FINE_MAX_M))]) for g in games)
    out["ref_5_8"] = dict((g, band_rate(fine[g], lag, 5.0, 8.0)) for g in games)
    far_rates = dict((g, band_rate(fine[g], lag, float(near_close), float(far))) for g in games)
    out["far_per_match"] = far_rates
    out["P_NEAR_FAR"] = round(float(np.median([far_rates[g] for g in games])), 3)
    return out


def fixed_lag_report(sel):
    """Markdown summary of select_fixed_lag()."""
    games = sel["games"]
    pc = _pc
    lines = ["NEAR_FAR_M at the fixed lag {}: pooled (E, 8] m against 5-8 m".format(
        sel["NEAREST_LAG"]), "",
        "| E | " + " | ".join(games) + " | largest gap |", "|---|" + "---|" * (len(games) + 1)]
    for e, per in sel["pooled_gap_per_match"].items():
        lines.append("| {} | {} | {:.1f} |".format(
            e, " | ".join("{:+.1f}".format(100.0 * per[g]) for g in games),
            100.0 * sel["pooled_gap"][e]))
    lines.append("NEAR_FAR_M = {:g} m".format(sel["NEAR_FAR_M"]))
    lines.append("")
    k = len(next(iter(sel["one_m_table"].values())))
    lines.append("| match | " + " | ".join("{}-{}".format(i, i + 1) for i in range(k))
                 + " | 5-8 |")
    lines.append("|---|" + "---|" * (k + 1))
    for g in games:
        lines.append("| {} | {} | {} |".format(
            g, " | ".join(pc(v) for v in sel["one_m_table"][g]), pc(sel["ref_5_8"][g])))
    lines.append("")
    lines.append("P_NEAR_FAR over ({:g}, {:g}] m at lag {}: {} -> median {:.3f}".format(
        sel["NEAR_CLOSE_M"], sel["NEAR_FAR_M"], sel["NEAREST_LAG"],
        ", ".join(pc(sel["far_per_match"][g]) for g in games), sel["P_NEAR_FAR"]))
    return "\n".join(lines) + "\n"


def bucket_b_report(sel):
    """Markdown summary of select_bucket_b()."""
    games = sel["games"]
    pc = _pc
    lines = ["NEAREST_LAG and NEAR_FAR_M: fixed-point iteration from lag 0", "",
             "| step | lag | NEAR_FAR_M | per-match argmax within the previous NEAR_FAR_M |",
             "|---|---|---|---|"]
    for i, state in enumerate(sel["trajectory"]):
        per = state["per_match"]
        lines.append("| {} | {} | {} | {} |".format(
            i, state["lag"], state["far"],
            "start" if per is None else ", ".join(str(per[g]) for g in games)))
    lines.append("cycle: {}".format(sel["cycle"] or "none"))
    lines.append("NEAREST_LAG = {}, NEAR_FAR_M = {:g}".format(sel["NEAREST_LAG"],
                                                             sel["NEAR_FAR_M"]))
    lines.append("largest pooled gap (E, 8] vs 5-8 m at the final lag, points: " + ", ".join(
        "E={}: {:.1f}".format(e, 100.0 * v) for e, v in sel["pooled_gap"].items()))
    lines.append("argmax lag, all distances: " + ", ".join(
        "{} {}".format(g, sel["lag_all_distance"][g]) for g in games))
    lines.append("")
    n_fine = len(sel["half_m_median"])
    lines.append("0.5 m band table at lag {} (percent)".format(sel["NEAREST_LAG"]))
    lines.append("")
    lines.append("| match | " + " | ".join("{:g}-{:g}".format(i * FINE_STEP_M,
                                                            (i + 1) * FINE_STEP_M)
                                            for i in range(n_fine)) + " | largest drop at |")
    lines.append("|---|" + "---|" * (n_fine + 1))
    for g in games:
        lines.append("| {} | {} | {:g} |".format(
            g, " | ".join(pc(v) for v in sel["half_m_table"][g]), sel["half_m_edge_per_match"][g]))
    lines.append("| median | {} | {:g} |".format(" | ".join(pc(v) for v in sel["half_m_median"]),
                                                 sel["NEAR_CLOSE_M"]))
    lines.append("NEAR_CLOSE_M = {:g} m (median drop {:.1f} points)".format(
        sel["NEAR_CLOSE_M"], 100.0 * sel["near_close_drop"]))
    lines.append("P_NEAR_CLOSE within {:g} m: {} -> median {:.3f}".format(
        sel["NEAR_CLOSE_M"], ", ".join(pc(sel["close_per_match"][g]) for g in games),
        sel["P_NEAR_CLOSE"]))
    lines.append("")
    k = len(next(iter(sel["one_m_table"].values())))
    lines.append("1 m band table at lag {} (percent), and 5-8 m".format(sel["NEAREST_LAG"]))
    lines.append("")
    lines.append("| match | " + " | ".join("{}-{}".format(i, i + 1) for i in range(k))
                 + " | 5-8 |")
    lines.append("|---|" + "---|" * (k + 1))
    for g in games:
        lines.append("| {} | {} | {} |".format(
            g, " | ".join(pc(v) for v in sel["one_m_table"][g]), pc(sel["ref_5_8"][g])))
    lines.append("")
    lines.append("P_NEAR_FAR over ({:g}, {:g}] m at lag {}: {} -> median {:.3f}".format(
        sel["NEAR_CLOSE_M"], sel["NEAR_FAR_M"], sel["NEAREST_LAG"],
        ", ".join(pc(sel["far_per_match"][g]) for g in games), sel["P_NEAR_FAR"]))
    if "exit_cover" in sel:
        lines.append("")
        lines.append("OUT_MAX_RANGE_M: Metrica exits {}, with a heading {}".format(
            sel["exits"], sel["exit_headings"]))
        lines.append("exit headings within m: " + ", ".join(
            "{}: {}".format(k_, v) for k_, v in sel["exit_cover"].items()))
        lines.append("OUT_MAX_RANGE_M = {} (share of headings), {} (share of all exits)".format(
            sel["OUT_MAX_RANGE_M"], sel["OUT_MAX_RANGE_M_over_exits"]))
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")
    m = sub.add_parser("measure", help="all measurements on one external reference")
    m.add_argument("--game", required=True)
    m.add_argument("--ref", required=True, help="the reference `frame,label` CSV")
    m.add_argument("--data-dir", default=None, help="MTAH_DATA_DIR for this match")
    m.add_argument("--out", required=True)
    m.add_argument("--config", default=None, help="JSON overrides of engine/config.py")
    r = sub.add_parser("report", help="print the markdown tables")
    r.add_argument("inputs", nargs="+")
    b = sub.add_parser("bucketb", help="apply the bucket b rules to the eight open references")
    b.add_argument("inputs", nargs="+")
    b.add_argument("--json", default=None)
    b.add_argument("--lag", type=int, default=None,
                   help="fixed NEAREST_LAG: no iteration, only NEAR_FAR_M and P_NEAR_FAR")
    b.add_argument("--near-close", type=float, default=None,
                   help="fixed NEAR_CLOSE_M, with --lag")
    args = parser.parse_args(argv)
    if args.command == "measure":
        from engine import config as engine_config

        data_dir = os.path.expanduser(args.data_dir) if args.data_dir else None
        cfg = engine_config.load(args.config) if args.config else None
        write_json(args.out, measure(args.game, os.path.expanduser(args.ref), data_dir, cfg))
    elif args.command == "bucketb":
        payloads = []
        for path in args.inputs:
            with open(path) as handle:
                payloads.append(json.load(handle))
        names = sorted(p["game"] for p in payloads)
        if names != sorted(OPEN_REFERENCES):
            raise SystemExit("bucketb needs the eight open references, got {}".format(names))
        if (args.lag is None) != (args.near_close is None):
            raise SystemExit("give --lag and --near-close together")
        if args.lag is not None:
            sel = select_fixed_lag(payloads, args.lag, args.near_close)
            report_text = fixed_lag_report(sel)
        else:
            sel = select_bucket_b(payloads)
            report_text = bucket_b_report(sel)
        if args.json:
            write_json(args.json, sel)
        sys.stdout.write(report_text)
    elif args.command == "report":
        payloads = []
        for path in args.inputs:
            with open(path) as handle:
                payloads.append(json.load(handle))
        sys.stdout.write(report(payloads))
    else:
        parser.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
