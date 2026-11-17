"""Wider contact radius gated by the team shape (P9-B clause 3). It measures. It designs no rule.

    python tools/analysis/wider_radius.py collect --game NAME --view VIEW [--ref REFERENCE.csv]
        [--public-gt PUBLIC_GT.csv] [--cache-dir DIR] [--config CONFIG.json] --out OUT.json
    python tools/analysis/wider_radius.py report OUT.json [OUT.json ...] [--out REPORT.md]

`collect` runs the engine (possession.prepare and possession.run with the default Config, or the
overrides of `--config`) on one match and stores per-candidate summary numbers in a JSON file.
`report` prints the tables of docs/plan/calibration/P9B_wider_radius.md. Set MTAH_DATA_DIR for
each match as tools/score_game.py needs it. `--cache-dir` points the engine's npz cache somewhere
else (each data root needs its own cache folder). `--view` is a free label (raw, smooth,
matched): the report groups the matches by it. Write the JSON files outside the repository. The
script refuses the sealed matches of calibration/SPLITS.md.

Definitions.

- Graded frame, reference hand-over, old and new team of a hand-over: as in
  tools/analysis/handover_confirm.py (an H <-> A change of the reference label inside one graded
  block).
- Impulse maximum. A frame c that events.window_maxima() keeps from the frames with
  |J| >= DV_CONTACT on an alive frame (e_dead <= EVENT_SUPPRESS_DEAD), J = events.impulse() with
  h = CONTACT_HALF_WINDOW: the frames that events._bump_contacts() examines. Only graded maxima
  count.
- Owner at c. The engine label at c. A maximum where the engine label is not H or A is left out
  (counted as "no owner"). The new team is the team that is not the owner.
- Team distance. For each team, the smallest player distance to the ball over [c - h, c + h]
  (the rule of _bump_contacts). d_new and d_own are the new and the owner team's distance.
- Candidate. A maximum of one of two classes:
    nobody      no player of either team within R_TOUCH (min(d_H, d_A) > R_TOUCH). Today
                _bump_contacts gives no row.
    owner_near  the owner team within R_TOUCH and the new team beyond it (d_own <= R_TOUCH <
                d_new). Today a kick, receive or redirect of the owner (an owner contact).
  and d_new inside one band of BANDS (lower bound excluded, upper bound included). A maximum where
  the new team is within R_TOUCH is no candidate (today a contact of the new team or contested).
- Shape. events.shape_response(shape, c, new, SHAPE_LAG_FRAMES) with the shape array of
  events.match_team_shape(): the C15 definition, R_new - R_own at +25 frames. "Clean" means no dead
  frame in [c, c + SHAPE_LAG_FRAMES], the condition of events.shape_confirms(). The gate "shape >=
  T" holds when the shape is defined, clean and at least T. At T = SHAPE_CONFIRM_MS the gate is
  checked against events.shape_confirms() on every candidate.
- Speed and airborne. Ball speed at c (m/s) and feats["airborne_suspect"] at c. The gate
  "shape >= T, not airborne" adds airborne False. The gate "shape >= T, slow" adds a speed under
  SLOW_MS.
- Truth. A candidate is true when a reference hand-over to its new team lies within MATCH_TOL
  frames of c.
- Cost class of an admitted false candidate, the first that holds:
    shift     a reference hand-over to the new team within SHIFT_TOL frames of c that the engine
              finds (an engine hand-over to the same team within MATCH_TOL frames of it): a
              switch here would move a transition the engine already has;
    spurious  the reference label at c is the owner: a switch here opens a spell inside a
              reference possession of the owner;
    ref_new   the reference label at c is already the new team (the engine owner is wrong at c);
    ref_dead  the reference label at c is D or X.
- Contact. A row of events._bump_contacts() on a graded frame (kick, receive, redirect or
  contested). The recall ceiling of a view is the share of reference hand-overs with no contact
  within MATCH_TOL frames. Such a hand-over is reached when a candidate with d_new <= REACH_M,
  the gate shape >= REACH_T and the hand-over's new team lies within MATCH_TOL frames.
- Splice. Not an engine run. Start from the engine labels. For each admitted candidate in frame
  order (band upper bound <= B, gate shape >= T), set the label to the new team from c up to the
  first engine owner event after c (possession.run parts["owner_events"]), the end of the graded
  block, or the first frame whose engine label is not H or A, whichever comes first. Score the
  splice against the reference with tools/score_game.paired_delta() (the `--versus` bootstrap of
  tools/score_game.py: spliced minus engine, 1000 draws, seed 0). The grid is SPLICE_B x
  SPLICE_T, once for the candidates of both classes (the card) and once for the nobody class
  alone.
- Public flips (BAR-ATH). The four flips of PLAN 1.3 with no new-team player within 1.5 m. The
  public file names the old and new team of each flip. For each impulse maximum within FLIP_TOL
  frames of a flip the report prints the offset, the flip's new-team and old-team distance, the
  band, the shape of the flip's new team, the speed, the airborne flag, the engine owner at c and
  whether the maximum is a candidate. Frames and distances only. The report also lists the
  candidates inside the public window with shape >= LIST_T, from the engine labels alone.

The script stores per-candidate summary numbers (distances, frames, speeds), never a tracking
row. numpy and pandas only. Deterministic.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "tools", "analysis")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from tools.analysis.handover_confirm import (OTHER, PUBLIC_FLIPS, TEAMS, _fmt,  # noqa: E402
                                             _pct, block_index, finite_or_inf, graded_mask,
                                             labels_md5, read_labels, reference_handovers,
                                             short, write_json)

SEALED = ("DFL-J03WQQ", "DFL-J03WR9")       # calibration/SPLITS.md
BANDS = ((1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0))   # the card's bands, m
THRESHOLDS = (0.5, 1.0, 1.45, 2.0, 2.5, 3.0)                # the card's gate thresholds, m/s
MATCH_TOL = 25             # frames, the card's truth window
SHIFT_TOL = 50             # frames, the card's "near a transition the engine finds"
SLOW_MS = 8.0              # m/s, the card's slow-ball gate
REACH_M = 3.0              # m, the card's recall-ceiling radius
REACH_T = 1.45             # m/s, the card's recall-ceiling gate (= SHAPE_CONFIRM_MS)
SPLICE_B = (2.5, 3.0)      # m, the card's splice band upper bounds
SPLICE_T = (1.45, 2.0, 2.5)                                 # m/s, the card's splice thresholds
SPLICE_CLASSES = (None, "nobody")   # both classes (the card), and the nobody class alone
FLIP_TOL = 50              # frames around a public flip
PUBLIC_WINDOW = (11185, 18684)   # team rules: the public window of BAR-ATH, video frames
LIST_T = 1.0               # m/s, the loosest gate of the public-window list
CLASSES = ("nobody", "owner_near")


# -- the candidate finder (pure, tested on synthetic arrays) --------------------------------------

def band_of(distance, bands=BANDS):
    """Index of the band (lo, hi] that holds `distance`, None outside every band."""
    if distance is None or not np.isfinite(distance):
        return None
    for i, (lo, hi) in enumerate(bands):
        if lo < distance <= hi:
            return i
    return None


def impulse_maxima(size, alive, cfg):
    """int64 array: the impulse maxima of events._bump_contacts (module docstring)."""
    from engine import events

    size = np.asarray(size, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        strong = np.isfinite(size) & (size >= cfg.DV_CONTACT) & np.asarray(alive, dtype=bool)
    return events.window_maxima(size, int(cfg.CONTACT_HALF_WINDOW), np.flatnonzero(strong))


def find_candidates(maxima, d_h, d_a, owner, graded, h, r_touch, bands=BANDS):
    """(candidates, counts) of the impulse maxima (module docstring).

    `d_h`, `d_a` are the per-frame nearest distances of each team (inf where none), `owner` the
    engine label per frame. Each candidate is a dict {c, cls, owner, new, d_new, d_own, band}.
    `counts` holds the graded maxima, the ones with no owner, and the ones whose new team lies
    within `r_touch` or whose d_new lies outside every band.
    """
    d_h = finite_or_inf(d_h)
    d_a = finite_or_inf(d_a)
    n = d_h.shape[0]
    counts = {"maxima": 0, "no_owner": 0, "new_near": 0, "out_of_band": 0}
    out = []
    for c in np.asarray(maxima, dtype=np.int64):
        c = int(c)
        if not graded[c]:
            continue
        counts["maxima"] += 1
        own = str(owner[c])
        if own not in TEAMS:
            counts["no_owner"] += 1
            continue
        new = OTHER[own]
        lo, hi = max(0, c - int(h)), min(n, c + int(h) + 1)
        dist = {"H": float(d_h[lo:hi].min()), "A": float(d_a[lo:hi].min())}
        d_new, d_own = dist[new], dist[own]
        if d_new <= r_touch:
            counts["new_near"] += 1
            continue
        cls = "owner_near" if d_own <= r_touch else "nobody"
        band = band_of(d_new, bands)
        if band is None:
            counts["out_of_band"] += 1
            continue
        out.append({"c": c, "cls": cls, "owner": own, "new": new, "d_new": d_new,
                    "d_own": d_own if np.isfinite(d_own) else None, "band": band})
    return out, counts


def gate(cand, threshold, extra=None, slow_ms=SLOW_MS):
    """True when `cand` passes "shape >= threshold" (and `extra`: "not_airborne" or "slow")."""
    value = cand.get("shape")
    if value is None or not cand.get("shape_clean") or value < threshold:
        return False
    if extra == "not_airborne":
        return not cand["airborne"]
    if extra == "slow":
        return cand["speed"] is not None and cand["speed"] < slow_ms
    return True


def cost_class(cand, ref, found_frames_by_team, shift_tol=SHIFT_TOL):
    """shift, spurious, ref_new or ref_dead for an admitted false candidate (module docstring).

    `found_frames_by_team` maps a team to the sorted frames of the reference hand-overs to that
    team that the engine finds.
    """
    frames = np.asarray(found_frames_by_team.get(cand["new"], []), dtype=np.int64)
    c = int(cand["c"])
    lo = int(np.searchsorted(frames, c - shift_tol, side="left"))
    hi = int(np.searchsorted(frames, c + shift_tol, side="right"))
    if hi > lo:
        return "shift"
    label = str(ref[c])
    if label == cand["owner"]:
        return "spurious"
    if label == cand["new"]:
        return "ref_new"
    return "ref_dead"


def splice_labels(labels, admitted, owner_frames, blocks):
    """Engine labels with the owner set to the new team after each admitted candidate.

    `admitted` is [(c, new team)], `owner_frames` the frames of the engine owner events. From c
    the label becomes the new team up to the first owner event after c, the end of the graded
    block, or the first frame whose engine label is not H or A (module docstring).
    """
    labels = np.asarray(labels).astype("<U1")
    out = labels.copy()
    n = labels.size
    blk = block_index(blocks, n)
    ends = {i: int(last) + 1 for i, (_, _, last) in enumerate(blocks)}
    owner_frames = np.sort(np.asarray(owner_frames, dtype=np.int64))
    live = np.isin(labels, TEAMS)
    for c, new in sorted((int(c), str(t)) for c, t in admitted):
        if blk[c] < 0 or not live[c]:
            continue
        k = int(np.searchsorted(owner_frames, c, side="right"))
        stop = min(int(owner_frames[k]) if k < owner_frames.size else n, ends[int(blk[c])])
        dead = np.flatnonzero(~live[c:stop])
        if dead.size:
            stop = c + int(dead[0])
        out[c:stop] = new
    return out


def near_frames(frames, sorted_targets, tol):
    """bool per frame: a target lies within `tol` frames."""
    targets = np.asarray(sorted_targets, dtype=np.int64)
    out = []
    for f in frames:
        lo = int(np.searchsorted(targets, int(f) - tol, side="left"))
        hi = int(np.searchsorted(targets, int(f) + tol, side="right"))
        out.append(hi > lo)
    return out


# -- collect --------------------------------------------------------------------------------------

def _f(value):
    """JSON-friendly float (None for NaN or inf)."""
    value = float(value)
    return value if np.isfinite(value) else None


def collect(args):
    """The `collect` command: per-candidate summary numbers for one match."""
    import games
    import score_game as sg
    from engine import data, events, possession
    from engine.config import Config, load as load_config

    game = args.game
    if game in SEALED:
        raise SystemExit("{} is sealed (calibration/SPLITS.md)".format(game))
    if args.public_gt and game != "BAR-ATH":
        raise SystemExit("--public-gt is for BAR-ATH only")
    if args.cache_dir:
        data.DEFAULT_CACHE_DIR = os.path.abspath(os.path.expanduser(args.cache_dir))
    cfg = load_config(args.config) if getattr(args, "config", None) else Config.default()
    inputs = possession.prepare(cfg, game)
    labels, parts = possession.run(cfg, return_parts=True, inputs=inputs)
    labels = np.asarray(labels).astype("<U1")
    n = labels.size
    blocks = inputs["blocks"]
    fps = float(inputs["fps"])
    graded = graded_mask(blocks, n)
    ball, feats = inputs["ball"], inputs["feats"]
    e_dead = np.asarray(inputs["e_dead"], dtype=np.float64)
    alive = ~(np.isfinite(e_dead) & (e_dead > cfg.EVENT_SUPPRESS_DEAD))
    dead_cum = np.concatenate(([0], np.cumsum(~alive)))
    h = int(cfg.CONTACT_HALF_WINDOW)
    lag = int(cfg.SHAPE_LAG_FRAMES)
    d_h, d_a = finite_or_inf(feats["dH"]), finite_or_inf(feats["dA"])
    speed = np.asarray(ball["speed"], dtype=np.float64)
    airborne = np.asarray(feats["airborne_suspect"], dtype=bool)
    size = events.impulse(ball, h)
    shape = events.match_team_shape(possession.engine_tactical(cfg, game), cfg)

    maxima = impulse_maxima(size, alive, cfg)
    contacts = events._bump_contacts(ball, feats, None, alive, cfg)
    contact_frames = sorted(int(c[0]) for c in contacts if graded[int(c[0])])
    # Check: the contact rows sit on the maxima, and a maximum has no row exactly when nobody is
    # within R_TOUCH.
    max_set = set(int(c) for c in maxima)
    if not set(int(c[0]) for c in contacts) <= max_set:
        raise RuntimeError("a contact row is not an impulse maximum")
    with_row = set(int(c[0]) for c in contacts)
    for c in maxima:
        lo, hi = max(0, int(c) - h), min(n, int(c) + h + 1)
        nobody = min(d_h[lo:hi].min(), d_a[lo:hi].min()) > cfg.R_TOUCH
        if nobody == (int(c) in with_row):
            raise RuntimeError("row presence differs from the R_TOUCH test at {}".format(c))

    cands, counts = find_candidates(maxima, d_h, d_a, labels, graded, h, cfg.R_TOUCH)
    contact_kind = {int(c[0]): c[1] for c in contacts}
    for cand in cands:
        c = cand["c"]
        value = events.shape_response(shape, c, cand["new"], lag)
        end = c + lag
        clean = bool(end < n and dead_cum[end + 1] - dead_cum[c] == 0)
        cand["shape"] = _f(value)
        cand["shape_clean"] = clean
        cand["speed"] = _f(speed[c])
        cand["airborne"] = bool(airborne[c])
        cand["owner_kind"] = contact_kind.get(c)
        if gate(cand, cfg.SHAPE_CONFIRM_MS) != events.shape_confirms(shape, c, cand["new"],
                                                                      dead_cum, cfg):
            raise RuntimeError("gate differs from events.shape_confirms at {}".format(c))

    # Splice grid (always: label-free counts on BAR-ATH, scored where a reference exists).
    owner_frames = [int(f) for f, _ in parts["owner_events"]]
    ref = None
    hos = []
    if args.ref:
        ref = read_labels(args.ref, int(games.GAMES[game]["n_video_frames"]))
        hos = reference_handovers(ref, blocks)
    gt_codes = sg.pm.label_codes(ref) if ref is not None else None
    base_codes = sg.pm.label_codes(labels)
    base_f1 = (float(sg.score_codes(gt_codes, base_codes)[0]["spell_f1_mean"])
               if ref is not None else None)
    splices = []
    for cls in SPLICE_CLASSES:
        for b_hi in SPLICE_B:
            for t in SPLICE_T:
                admitted = [(cd["c"], cd["new"]) for cd in pick(cands, cls=cls, band_hi=b_hi)
                            if gate(cd, t)]
                spliced = splice_labels(labels, admitted, owner_frames, blocks)
                row = {"cls": cls, "band_hi": b_hi, "T": t, "admitted": len(admitted),
                       "frames_changed": int((spliced != labels).sum())}
                if ref is not None:
                    codes = sg.pm.label_codes(spliced)
                    row["base"] = base_f1
                    row["spliced"] = float(sg.score_codes(gt_codes, codes)[0]["spell_f1_mean"])
                    row["paired"] = sg.paired_delta(gt_codes, codes, base_codes)
                splices.append(row)

    ho_recs = []
    if ref is not None:
        eng_hos = reference_handovers(labels, blocks)
        eng_by_team = {t: sorted(f for f, _, new in eng_hos if new == t) for t in TEAMS}
        found = {t: [] for t in TEAMS}
        for t, _, new in hos:
            if near_frames([t], eng_by_team[new], MATCH_TOL)[0]:
                found[new].append(t)
        ho_by_team = {t: sorted(f for f, _, new in hos if new == t) for t in TEAMS}
        for cand in cands:
            cand["true_ho"] = bool(near_frames([cand["c"]], ho_by_team[cand["new"]],
                                               MATCH_TOL)[0])
            cand["ref_c"] = str(ref[cand["c"]])
            cand["cost"] = None if cand["true_ho"] else cost_class(cand, ref, found)
        cand_frames = {t: sorted(cd["c"] for cd in cands if cd["new"] == t) for t in TEAMS}
        reach = {t: sorted(cd["c"] for cd in cands if cd["new"] == t and cd["d_new"] <= REACH_M
                           and gate(cd, REACH_T)) for t in TEAMS}
        for t, old, new in hos:
            ho_recs.append({"t": t, "old": old, "new": new,
                            "contact": bool(near_frames([t], contact_frames, MATCH_TOL)[0]),
                            "found": bool(near_frames([t], eng_by_team[new], MATCH_TOL)[0]),
                            "cand_any": bool(near_frames([t], cand_frames[new], MATCH_TOL)[0]),
                            "cand_reach": bool(near_frames([t], reach[new], MATCH_TOL)[0])})

    public = None
    if args.public_gt:
        public = public_flips(args.public_gt, maxima, d_h, d_a, labels, graded, h, shape, lag,
                              dead_cum, speed, airborne, cands, cfg)

    out = {"game": game, "view": args.view, "fps": fps, "h": h,
           "engine_md5": labels_md5(labels), "n_frames": int(n),
           "alive_graded_min": float((alive & graded).sum() / fps / 60.0),
           "has_ref": ref is not None, "counts": counts, "n_contacts": len(contact_frames),
           "candidates": cands, "handovers": ho_recs, "splices": splices, "public": public,
           "cfg": {"R_TOUCH": cfg.R_TOUCH, "CONTACT_HALF_WINDOW": h, "DV_CONTACT": cfg.DV_CONTACT,
                   "SHAPE_LAG_FRAMES": lag, "SHAPE_CONFIRM_MS": cfg.SHAPE_CONFIRM_MS,
                   "config_hash": cfg.config_hash()}}
    write_json(args.out, out)
    return out


def public_flips(path, maxima, d_h, d_a, labels, graded, h, shape, lag, dead_cum, speed,
                 airborne, cands, cfg):
    """Impulse maxima near the four public flips of PLAN 1.3 (frames and distances only)."""
    import pandas as pd
    from engine import events

    gt = read_labels(path)
    first = int(pd.read_csv(path, usecols=["frame"])["frame"].min())
    n = d_h.shape[0]
    by_frame = {cd["c"]: cd for cd in cands}
    out = []
    for flip in PUBLIC_FLIPS:
        i = flip - first
        old, new = str(gt[i - 1]), str(gt[i])
        if old not in TEAMS or new not in TEAMS or old == new:
            raise RuntimeError("{} is not an H/A flip of the public file".format(flip))
        dist = {"H": d_h, "A": d_a}
        rows = []
        for c in maxima:
            c = int(c)
            if abs(c - flip) > FLIP_TOL or not graded[c]:
                continue
            lo, hi = max(0, c - h), min(n, c + h + 1)
            dn, do = float(dist[new][lo:hi].min()), float(dist[old][lo:hi].min())
            end = c + lag
            clean = bool(end < n and dead_cum[end + 1] - dead_cum[c] == 0)
            cd = by_frame.get(c)
            rows.append({"dt": c - flip, "d_new": _f(dn), "d_old": _f(do),
                         "band": band_of(dn), "shape": _f(events.shape_response(shape, c, new,
                                                                                 lag)),
                         "shape_clean": clean, "speed": _f(speed[c]),
                         "airborne": bool(airborne[c]),
                         "engine_owner": "new" if labels[c] == new else (
                             "old" if labels[c] == old else str(labels[c])),
                         "candidate": None if cd is None else cd["cls"]})
        out.append({"t": flip, "maxima": rows})
    return out


# -- report ---------------------------------------------------------------------------------------

def band_name(i):
    """`1.5-2.0` for a band index, `all` for None."""
    if i is None:
        return "all"
    lo, hi = BANDS[i]
    return "{:.1f}-{:.1f}".format(lo, hi)


def pick(cands, band=None, cls=None, band_hi=None):
    """Candidates of one band (index), one class, or with the band's upper bound <= band_hi."""
    out = []
    for cd in cands:
        if band is not None and cd["band"] != band:
            continue
        if cls is not None and cd["cls"] != cls:
            continue
        if band_hi is not None and BANDS[cd["band"]][1] > band_hi:
            continue
        out.append(cd)
    return out


def pooled(group):
    """All candidates of a group of payloads."""
    return [cd for p in group for cd in p["candidates"]]


def report_candidates(group, lines):
    """Measurement 1: candidates and true share per band and class, pooled."""
    lines.append("#### 1a. Candidates per band and class, pooled: n, true hand-overs (share)")
    lines.append("")
    lines.append("| band m | nobody | owner_near | both |")
    lines.append("|---|---|---|---|")
    cands = pooled(group)
    for band in list(range(len(BANDS))) + [None]:
        cells = []
        for cls in CLASSES + (None,):
            sel = pick(cands, band, cls)
            cells.append(_pct(sum(1 for c in sel if c["true_ho"]), len(sel)))
        lines.append("| {} | {} |".format(band_name(band), " | ".join(cells)))
    lines.append("")
    lines.append("#### 1b. Per match: candidates, true (share), both classes")
    lines.append("")
    lines.append("| match | " + " | ".join(band_name(b) for b in range(len(BANDS)))
                 + " | gate 1.45, <= 3.0 m: true / false |")
    lines.append("|---|" + "---|" * len(BANDS) + "---|")
    for p in group:
        cells = []
        for b in range(len(BANDS)):
            sel = pick(p["candidates"], b)
            cells.append(_pct(sum(1 for c in sel if c["true_ho"]), len(sel)))
        adm = [c for c in pick(p["candidates"], band_hi=3.0) if gate(c, 1.45)]
        tt = sum(1 for c in adm if c["true_ho"])
        lines.append("| {} | {} | {} / {} |".format(short(p["game"]), " | ".join(cells), tt,
                                                    len(adm) - tt))
    lines.append("")


def gate_cells(sel, extra=None):
    """`true/false (precision)` for each threshold."""
    cells = []
    for t in THRESHOLDS:
        adm = [c for c in sel if gate(c, t, extra)]
        tt = sum(1 for c in adm if c["true_ho"])
        cells.append("{}/{} ({})".format(tt, len(adm) - tt,
                                         "-" if not adm else "{:.0f} %".format(100.0 * tt
                                                                               / len(adm))))
    return cells


def report_gates(group, lines):
    """Measurement 1: the gate curves per band and pooled."""
    cands = pooled(group)
    header = ("| band m | class | n (true) | " + " | ".join("T {}".format(t) for t in THRESHOLDS)
              + " |")
    rule = "|---|---|---|" + "---|" * len(THRESHOLDS)
    for extra, title in ((None, "shape >= T"), ("not_airborne", "shape >= T and not airborne"),
                         ("slow", "shape >= T and ball speed under {:.0f} m/s".format(SLOW_MS))):
        lines.append("#### 1c. Gate {}: admitted true / false (precision), pooled".format(title))
        lines.append("")
        lines.append(header)
        lines.append(rule)
        rows = [(b, None, None) for b in range(len(BANDS))]
        rows += [(None, None, 2.5), (None, None, 3.0), (None, None, None)]
        if extra is None:
            rows += [(None, "nobody", 2.5), (None, "owner_near", 2.5),
                     (None, "nobody", 3.0), (None, "owner_near", 3.0)]
        for band, cls, hi in rows:
            sel = pick(cands, band, cls, hi)
            name = band_name(band) if hi is None else "<= {:.1f}".format(hi)
            lines.append("| {} | {} | {} ({}) | {} |".format(
                name, cls or "both", len(sel), sum(1 for c in sel if c["true_ho"]),
                " | ".join(gate_cells(sel, extra))))
        lines.append("")


def report_cost(group, lines):
    """Measurement 2: cost classes of the admitted false candidates."""
    cands = pooled(group)
    lines.append("#### 2. Admitted false candidates by cost class, pooled: shift / spurious / "
                 "ref_new / ref_dead")
    lines.append("")
    lines.append("| band m | " + " | ".join("T {}".format(t) for t in THRESHOLDS) + " |")
    lines.append("|---|" + "---|" * len(THRESHOLDS))
    rows = [(b, None) for b in range(len(BANDS))] + [(None, 2.5), (None, 3.0), (None, None)]
    for band, hi in rows:
        sel = pick(cands, band, band_hi=hi)
        cells = []
        for t in THRESHOLDS:
            bad = [c for c in sel if gate(c, t) and not c["true_ho"]]
            cells.append(" / ".join(str(sum(1 for c in bad if c["cost"] == k))
                                    for k in ("shift", "spurious", "ref_new", "ref_dead")))
        name = band_name(band) if hi is None else "<= {:.1f}".format(hi)
        lines.append("| {} | {} |".format(name, " | ".join(cells)))
    lines.append("")


def report_ceiling(group, lines):
    """Measurement 3: the recall ceiling of the contact table and the wider-radius reach."""
    lines.append("#### 3. Recall ceiling: reference hand-overs with no contact within {} "
                 "frames".format(MATCH_TOL))
    lines.append("")
    lines.append("| match | hand-overs | engine finds | no contact | of those: any candidate | "
                 "of those: <= {:.1f} m and shape >= {} |".format(REACH_M, REACH_T))
    lines.append("|---|---|---|---|---|---|")
    tot = [0, 0, 0, 0, 0]
    for p in group:
        hos = p["handovers"]
        none = [r for r in hos if not r["contact"]]
        vals = [len(hos), sum(1 for r in hos if r["found"]), len(none),
                sum(1 for r in none if r["cand_any"]), sum(1 for r in none if r["cand_reach"])]
        tot = [a + b for a, b in zip(tot, vals)]
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            short(p["game"]), vals[0], _pct(vals[1], vals[0]), _pct(vals[2], vals[0]),
            _pct(vals[3], vals[2]), _pct(vals[4], vals[2])))
    lines.append("| pooled | {} | {} | {} | {} | {} |".format(
        tot[0], _pct(tot[1], tot[0]), _pct(tot[2], tot[0]), _pct(tot[3], tot[2]),
        _pct(tot[4], tot[2])))
    lines.append("")


def report_splice(group, lines):
    """Measurement 5: the splice tables, both classes and the nobody class alone."""
    for cls in SPLICE_CLASSES:
        report_splice_class(group, lines, cls)


def report_splice_class(group, lines, cls):
    """Measurement 5: the splice table of one candidate class (None: both)."""
    lines.append("#### 5{}. Splice (not an engine run), candidates {}: spliced minus engine, "
                 "spellF1, paired 5-95 %".format("a" if cls is None else "b",
                                                 "of both classes" if cls is None
                                                 else "of class " + cls))
    lines.append("")
    lines.append("| match | band <= m | T | admitted (true) | frames changed | engine | spliced | "
                 "delta [5-95 %] |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for p in group:
        trues = {}
        for b_hi in SPLICE_B:
            for t in SPLICE_T:
                trues[(b_hi, t)] = sum(1 for cd in pick(p["candidates"], cls=cls, band_hi=b_hi)
                                       if gate(cd, t) and cd.get("true_ho"))
        for row in p["splices"]:
            if row["cls"] != cls:
                continue
            pr = row["paired"]
            lines.append("| {} | {:.1f} | {} | {} ({}) | {} | {:.4f} | {:.4f} | {:+.4f} [{:+.4f}, "
                         "{:+.4f}] |".format(short(p["game"]), row["band_hi"], row["T"],
                                             row["admitted"], trues.get((row["band_hi"],
                                                                         row["T"]), 0),
                                             row["frames_changed"], row["base"], row["spliced"],
                                             pr["delta"], pr["low"], pr["high"]))
    lines.append("")
    lines.append("Mean delta over the matches of the group:")
    lines.append("")
    lines.append("| band <= m | T | mean delta | matches above 0 / below 0 | intervals above 0 / "
                 "below 0 |")
    lines.append("|---|---|---|---|---|")
    for b_hi in SPLICE_B:
        for t in SPLICE_T:
            rows = [r for p in group for r in p["splices"]
                    if r["band_hi"] == b_hi and r["T"] == t and r["cls"] == cls]
            ds = [r["paired"]["delta"] for r in rows]
            lines.append("| {:.1f} | {} | {:+.4f} | {} / {} | {} / {} |".format(
                b_hi, t, float(np.mean(ds)), sum(1 for d in ds if d > 0),
                sum(1 for d in ds if d < 0), sum(1 for r in rows if r["paired"]["low"] > 0),
                sum(1 for r in rows if r["paired"]["high"] < 0)))
    lines.append("")


def report_barath(p, lines):
    """Measurement 4: BAR-ATH, label-free counts and the public flips."""
    cands = p["candidates"]
    minutes = p["alive_graded_min"]
    lines.append("#### 4a. BAR-ATH full match (label-free): candidates per band and gate, count "
                 "(per alive minute)")
    lines.append("")
    lines.append("Engine md5 `{}`; {} graded impulse maxima, {} contact rows, {:.1f} alive graded "
                 "minutes; maxima left out: {} with no engine owner, {} with the new team within "
                 "R_TOUCH, {} with d_new beyond 4.0 m.".format(
                     p["engine_md5"], p["counts"]["maxima"], p["n_contacts"], minutes,
                     p["counts"]["no_owner"], p["counts"]["new_near"],
                     p["counts"]["out_of_band"]))
    lines.append("")
    gates = (None, 1.45, 2.0, 2.5, 3.0)
    lines.append("| band m | class | " + " | ".join("all" if g is None else "shape >= {}".format(g)
                                                    for g in gates)
                 + " | 1.45, not airborne | 1.45, slow |")
    lines.append("|---|---|" + "---|" * (len(gates) + 2))
    rows = [(b, cls, None) for b in range(len(BANDS)) for cls in CLASSES]
    rows += [(None, None, 2.5), (None, None, 3.0), (None, None, None)]
    for band, cls, hi in rows:
        sel = pick(cands, band, cls, hi)
        cells = []
        for g in gates:
            k = len(sel) if g is None else sum(1 for c in sel if gate(c, g))
            cells.append("{} ({:.2f})".format(k, k / minutes if minutes else 0.0))
        for extra in ("not_airborne", "slow"):
            k = sum(1 for c in sel if gate(c, 1.45, extra))
            cells.append("{} ({:.2f})".format(k, k / minutes if minutes else 0.0))
        name = band_name(band) if hi is None else "<= {:.1f}".format(hi)
        lines.append("| {} | {} | {} |".format(name, cls or "both", " | ".join(cells)))
    lines.append("")
    lines.append("Splice grid on BAR-ATH (label-free, nothing scored): admitted candidates and "
                 "frames changed.")
    lines.append("")
    lines.append("| class | band <= m | T | admitted | frames changed |")
    lines.append("|---|---|---|---|---|")
    for row in p["splices"]:
        lines.append("| {} | {:.1f} | {} | {} | {} |".format(
            row["cls"] or "both", row["band_hi"], row["T"], row["admitted"],
            row["frames_changed"]))
    lines.append("")
    lines.append("#### 4b. BAR-ATH public window {}..{} (label-free): candidates with shape >= {} "
                 "(frames and distances only)".format(PUBLIC_WINDOW[0], PUBLIC_WINDOW[1], LIST_T))
    lines.append("")
    lines.append("| frame | class | band | d_new m | d_own m | shape m/s | speed m/s | airborne |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cd in cands:
        if PUBLIC_WINDOW[0] <= cd["c"] <= PUBLIC_WINDOW[1] and gate(cd, LIST_T):
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                cd["c"], cd["cls"], band_name(cd["band"]), _fmt(cd["d_new"]), _fmt(cd["d_own"]),
                _fmt(cd["shape"]), _fmt(cd["speed"], 1), cd["airborne"]))
    lines.append("")
    if not p.get("public"):
        return
    lines.append("#### 4c. BAR-ATH public flips of PLAN 1.3: impulse maxima within {} frames "
                 "(frames and distances only)".format(FLIP_TOL))
    lines.append("")
    lines.append("| flip | dt | d_new m | d_old m | band of d_new | shape new m/s | clean | speed "
                 "m/s | airborne | engine owner at c | candidate |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for flip in p["public"]:
        if not flip["maxima"]:
            lines.append("| {} | none | - | - | - | - | - | - | - | - | - |".format(flip["t"]))
        for m in flip["maxima"]:
            lines.append("| {} | {:+d} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                flip["t"], m["dt"], _fmt(m["d_new"]), _fmt(m["d_old"]),
                "-" if m["band"] is None else band_name(m["band"]), _fmt(m["shape"]),
                m["shape_clean"], _fmt(m["speed"], 1), m["airborne"], m["engine_owner"],
                m["candidate"] or "no"))
    lines.append("")


def report_overview(payloads, lines):
    """Per-match engine md5 and counts."""
    lines.append("### 0. Matches and counts")
    lines.append("")
    lines.append("| match | view | engine md5 | graded maxima | contact rows | no owner | new "
                 "team near | d_new > 4 m | candidates nobody / owner_near | reference "
                 "hand-overs |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for p in payloads:
        k = p["counts"]
        cands = p["candidates"]
        lines.append("| {} | {} | `{}` | {} | {} | {} | {} | {} | {} / {} | {} |".format(
            short(p["game"]), p["view"], p["engine_md5"][:8], k["maxima"], p["n_contacts"],
            k["no_owner"], k["new_near"], k["out_of_band"],
            sum(1 for c in cands if c["cls"] == "nobody"),
            sum(1 for c in cands if c["cls"] == "owner_near"),
            len(p["handovers"]) if p["has_ref"] else "-"))
    lines.append("")


def report(args):
    """The `report` command: print the markdown tables."""
    payloads = []
    for path in args.paths:
        with open(path) as handle:
            payloads.append(json.load(handle))
    lines = []
    report_overview(payloads, lines)
    views = []
    for p in payloads:
        if p["has_ref"] and p["view"] not in views:
            views.append(p["view"])
    for view in views:
        group = [p for p in payloads if p["view"] == view and p["has_ref"]]
        lines.append("### View: {} ({} matches)".format(view, len(group)))
        lines.append("")
        report_candidates(group, lines)
        report_gates(group, lines)
        report_cost(group, lines)
        report_ceiling(group, lines)
        report_splice(group, lines)
    for p in payloads:
        if not p["has_ref"]:
            lines.append("### BAR-ATH")
            lines.append("")
            report_barath(p, lines)
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
    c.add_argument("--view", required=True, help="group label of the report: raw, smooth, ...")
    c.add_argument("--ref")
    c.add_argument("--public-gt")
    c.add_argument("--cache-dir")
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
