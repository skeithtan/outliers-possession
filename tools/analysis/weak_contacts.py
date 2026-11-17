"""Weak contacts of the team without the ball, gated by the team shape (RESEARCH-3 R3-A).

    python tools/analysis/weak_contacts.py collect --game NAME --view VIEW [--ref REFERENCE.csv]
        [--alt-ref VARIANT.csv] [--public-gt PUBLIC_GT.csv] [--dv-ladder] [--cache-dir DIR]
        [--config CONFIG.json] --out OUT.json
    python tools/analysis/weak_contacts.py report OUT.json [OUT.json ...] [--out REPORT.md]

`collect` runs the engine (possession.prepare and possession.run with the default Config, or the
overrides of `--config`) on one match. It stores the switch rates and per-candidate summary
numbers in a JSON file. `report` prints the tables of docs/plan/calibration/R3A_weak_contacts.md.
Set MTAH_DATA_DIR for each match as tools/score_game.py needs it. `--cache-dir` points the
engine's npz cache somewhere else (each data root needs its own cache folder). `--view` is a free
label (raw, smooth, matched): the report groups the matches by it. Write the JSON files outside
the repository. The script refuses the sealed matches of calibration/SPLITS.md. It designs no
rule and changes no engine file.

Definitions.

- Graded frame, reference hand-over, old and new team of a hand-over: as in
  tools/analysis/handover_confirm.py (an H <-> A change of the label inside one graded block).
- Switch rate. Inside each graded block the labels are cut into runs of one value. A switch is a
  run of H or A that directly follows a run of the other team (an H -> A or A -> H change with
  no D between). Alive frames are the graded frames labelled H or A. The rate is switches per
  alive minute (alive frames / fps / 60). On the engine output this is the rule that set
  DV_CONTACT (LEARNINGS C8, RESEARCH-2 1g1). The script prints it for the engine labels and for
  the reference labels of each view. `--dv-ladder` also runs the engine at DV_LADDER values of
  DV_CONTACT and prints their rates (label-free, for the band line on the shipped engine).
- Impulse. J = events.impulse() with h = CONTACT_HALF_WINDOW. Alive means e_dead <=
  EVENT_SUPPRESS_DEAD, the rule of events.detect_events().
- Weak maximum. A frame c that events.window_maxima() keeps from the alive frames with |J| >=
  J_LO, and whose |J| is below DV_CONTACT. window_maxima() compares c with every frame of
  [c - h, c + h], so no impulse maximum of _bump_contacts lies within h frames of a weak maximum
  (checked on every run). Only graded maxima count.
- Owner at c. The engine label at c. A maximum where the engine label is not H or A is left out
  (counted as "no owner"). The new team is the team that is not the owner.
- Team distance. For each team, the smallest player distance to the ball over [c - h, c + h]
  (the attribution rule of _bump_contacts). d_new and d_own.
- Candidate. A weak maximum with d_new <= R_TOUCH. It is "contested" when d_own <= R_TOUCH too,
  "uncontested" otherwise. A weak maximum with d_new > R_TOUCH is counted as "new team far".
  The |J| band is one of J_BANDS (lower bound included, upper bound excluded). The kind (kick,
  receive, redirect) comes from ds = speed(c + h) - speed(c - h) against REDIRECT_FRACTION x
  |J|, the rule of _bump_contacts.
- Shape. events.shape_response(shape, c, new, SHAPE_LAG_FRAMES) with the shape array of
  events.match_team_shape(): the C15 definition. "Clean" means no dead frame in
  [c, c + SHAPE_LAG_FRAMES]. The gate "shape >= T" holds when the shape is defined, clean and at
  least T. At T = SHAPE_CONFIRM_MS it is checked against events.shape_confirms() on every
  candidate. The owner-side guard admits only uncontested candidates.
- Truth. A candidate is true when a reference hand-over to its new team lies within MATCH_TOL
  frames of c. The matched hand-over is the nearest such one (the earlier one on a tie).
- Reference spell of a hand-over at t. The frames from t to the first later frame whose
  reference label differs from the label at t, or the end of the graded block. Touch-and-lose
  check: the share of admitted true candidates whose matched hand-over opens a spell under
  SHORT_S seconds. Reach: the distinct reference hand-overs matched by an admitted true
  candidate, split by spell length (under or at least 2 s) and by whether a contact row of
  _bump_contacts (with the C18 reach rows when REACH_SHAPE_CONFIRM is on) lies within MATCH_TOL
  frames of it.
- Alternative reference (`--alt-ref`, Sample Game 3 only in practice: the F2 variant of
  REDIRECT-FORENSICS 2). The same truth, spells and splice against that file.
- Cost class of an admitted false candidate, the first that holds (wider_radius.cost_class):
  shift (a reference hand-over to the new team within SHIFT_TOL frames that the engine finds),
  spurious (the reference label at c is the owner), ref_new (it is the new team), ref_dead.
- Splice. Not an engine run. wider_radius.splice_labels(): for each admitted candidate in frame
  order the label becomes the new team from c up to the first engine owner event after c, the
  end of the graded block, or the first frame whose engine label is not H or A. The grid is
  SPLICE_FLOORS (the weakest |J| admitted, up to DV_CONTACT) x THRESHOLDS x GUARDS. Each splice
  is scored with tools/score_game.paired_delta() (the `--versus` bootstrap, 1000 draws, seed 0).
  The report adds the mean over the Metrica matches, over the DFL matches, and over all matches
  of a view with the cross-match interval of tools/paired_summary.paired_summary().
- Public window (BAR-ATH, `--public-gt`). For each candidate inside PUBLIC_WINDOW with shape >=
  LIST_T the report prints the frame, |J|, band, new team, contested flag and shape, and from
  the public file the label at c, the label at c + MATCH_TOL and whether the public labels hand
  over to the new team within MATCH_TOL frames of c. Frames and labels only.

The script stores per-candidate summary numbers (frames, impulses, distances, shape values),
never a tracking row. numpy and pandas only. Deterministic.
"""

import argparse
import dataclasses
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "tools", "analysis")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from tools.analysis.handover_confirm import (OTHER, TEAMS, _fmt, _pct,  # noqa: E402
                                             finite_or_inf, graded_mask, labels_md5,
                                             read_labels, reference_handovers, short,
                                             write_json)
from tools.analysis.wider_radius import cost_class, near_frames, splice_labels  # noqa: E402

SEALED = ("DFL-J03WQQ", "DFL-J03WR9")       # calibration/SPLITS.md
J_LO = 5.0                 # m/s, the card's population floor (RESEARCH-3 R3-A)
J_BANDS = ((5.0, 6.0), (6.0, 7.0), (7.0, 8.0), (8.0, 9.0))  # m/s, the card's |J| bands
THRESHOLDS = (1.45, 2.0, 2.5, 3.0)          # m/s, the card's gate thresholds (1.45: C15)
GUARDS = ("all", "uncontested")             # the card's owner-side guard off and on
MATCH_TOL = 25             # frames, the card's truth window
SHIFT_TOL = 50             # frames, as wider_radius.py
SHORT_S = (1.0, 2.0)       # s, the card's touch-and-lose spell lengths
SPLICE_FLOORS = (8.0, 7.0, 6.0, 5.0)        # m/s, weakest |J| admitted by a splice
DV_LADDER = (7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0)   # m/s, the C8 band steps (LEARNINGS C8)
BAND_RULE = 3.0            # switches per alive minute, the C8 band (RESEARCH-2 1g1)
PUBLIC_WINDOW = (11185, 18684)   # team rules: the public window of BAR-ATH, video frames
LIST_T = 1.45              # m/s, the loosest gate of the public-window list
CLASSES = ("uncontested", "contested")
KINDS = ("kick", "receive", "redirect")


# -- pure helpers (tested on synthetic arrays) ----------------------------------------------------

def switch_stats(labels, blocks, fps):
    """{switches, alive_frames, alive_min, rate, possessions} of `labels` (module docstring)."""
    labels = np.asarray(labels).astype("<U1")
    switches = alive = poss = 0
    for _, first, last in blocks:
        seg = labels[int(first):int(last) + 1]
        if seg.size == 0:
            continue
        alive += int(np.isin(seg, TEAMS).sum())
        starts = np.concatenate(([0], np.flatnonzero(seg[1:] != seg[:-1]) + 1))
        vals = seg[starts]
        team = np.isin(vals, TEAMS)
        poss += int(team.sum())
        switches += int((team[1:] & team[:-1]).sum())
    minutes = alive / float(fps) / 60.0
    return {"switches": switches, "alive_frames": alive, "alive_min": minutes,
            "rate": switches / minutes if minutes > 0 else None, "possessions": poss}


def j_band_of(value, bands=J_BANDS):
    """Index of the band [lo, hi) that holds `value`, None outside every band."""
    if value is None or not np.isfinite(value):
        return None
    for i, (lo, hi) in enumerate(bands):
        if lo <= value < hi:
            return i
    return None


def weak_maxima(size, alive, h, lo, hi):
    """int64 array: window maxima of the alive frames with |J| >= lo, kept when |J| < hi."""
    from engine import events

    size = np.asarray(size, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        frames = np.flatnonzero(np.isfinite(size) & (size >= lo) & np.asarray(alive, dtype=bool))
    keep = events.window_maxima(size, int(h), frames)
    return keep[size[keep] < hi] if keep.size else keep


def find_weak(maxima, size, speed, d_h, d_a, owner, graded, h, r_touch, fraction,
              bands=J_BANDS):
    """(candidates, counts) of the weak maxima (module docstring).

    Each candidate is a dict {c, j, band, kind, owner, new, d_new, d_own, contested}. `counts`
    holds the graded weak maxima, the ones with no owner and the ones whose new team is beyond
    `r_touch`.
    """
    d_h = finite_or_inf(d_h)
    d_a = finite_or_inf(d_a)
    size = np.asarray(size, dtype=np.float64)
    speed = np.asarray(speed, dtype=np.float64)
    n = d_h.shape[0]
    counts = {"maxima": 0, "no_owner": 0, "new_far": 0}
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
        if not d_new <= r_touch:
            counts["new_far"] += 1
            continue
        j = float(size[c])
        kind = None
        if c - h >= 0 and c + h < n:
            change = speed[c + h] - speed[c - h]
            split = fraction * j
            kind = "kick" if change >= split else ("receive" if change <= -split else "redirect")
        out.append({"c": c, "j": j, "band": j_band_of(j, bands), "kind": kind, "owner": own,
                    "new": new, "d_new": d_new, "d_own": d_own if np.isfinite(d_own) else None,
                    "contested": bool(d_own <= r_touch)})
    return out, counts


def gate(cand, threshold, guard="all"):
    """True when `cand` passes "shape >= threshold" and the guard ("uncontested": no owner near)."""
    value = cand.get("shape")
    if value is None or not cand.get("shape_clean") or value < threshold:
        return False
    return guard == "all" or not cand["contested"]


def nearest_in(frames, c, tol):
    """The frame of sorted `frames` nearest to c within `tol` (the earlier on a tie), or None."""
    frames = np.asarray(frames, dtype=np.int64)
    lo = int(np.searchsorted(frames, int(c) - tol, side="left"))
    hi = int(np.searchsorted(frames, int(c) + tol, side="right"))
    if hi <= lo:
        return None
    window = frames[lo:hi]
    return int(window[int(np.argmin(np.abs(window - int(c))))])


def spell_frames(labels, t, block_end):
    """Frames from t to the first later frame whose label differs, capped at block_end + 1."""
    labels = np.asarray(labels)
    seg = labels[int(t):int(block_end) + 1]
    change = np.flatnonzero(seg != seg[0])
    return int(change[0]) if change.size else int(seg.size)


def flags_text(flags):
    """The spell match flags (5, n_gt) as one string of 0 and 1 per IoU row."""
    return ["".join("1" if v else "0" for v in row) for row in np.asarray(flags)]


def flags_array(rows):
    """Inverse of flags_text()."""
    return np.asarray([[1 if ch == "1" else 0 for ch in row] for row in rows], dtype=np.int8)


def pick(cands, band=None, cls=None, floor=None):
    """Candidates of one |J| band (index), one class, or with |J| >= floor."""
    out = []
    for cd in cands:
        if band is not None and cd["band"] != band:
            continue
        if cls is not None and (cd["contested"] != (cls == "contested")):
            continue
        if floor is not None and cd["j"] < floor:
            continue
        out.append(cd)
    return out


# -- collect --------------------------------------------------------------------------------------

def _f(value):
    """JSON-friendly float (None for NaN or inf)."""
    value = float(value)
    return value if np.isfinite(value) else None


def _truth(cands, ref, blocks, labels, key):
    """Truth, matched hand-over, spell and cost class of each candidate against `ref`."""
    hos = reference_handovers(ref, blocks)
    eng_hos = reference_handovers(labels, blocks)
    eng_by_team = {t: sorted(f for f, _, new in eng_hos if new == t) for t in TEAMS}
    ho_by_team = {t: sorted(f for f, _, new in hos if new == t) for t in TEAMS}
    found = {t: [] for t in TEAMS}
    for t, _, new in hos:
        if near_frames([t], eng_by_team[new], MATCH_TOL)[0]:
            found[new].append(t)
    end_of = _block_end_index(blocks, ref.size)
    for cd in cands:
        ho = nearest_in(ho_by_team[cd["new"]], cd["c"], MATCH_TOL)
        cd[key] = {"true": ho is not None, "ho_t": ho,
                   "spell": None if ho is None else spell_frames(ref, ho, end_of[ho]),
                   "ref_c": str(ref[cd["c"]]),
                   "cost": None if ho is not None else cost_class(cd, ref, found, SHIFT_TOL)}
    return hos, eng_by_team, end_of


def _block_end_index(blocks, n):
    """int64 (n,): the last frame of the graded block of each frame, -1 outside."""
    out = np.full(int(n), -1, dtype=np.int64)
    for _, first, last in blocks:
        out[int(first):int(last) + 1] = int(last)
    return out


def _handover_records(ref, blocks, labels, contact_frames, end_of):
    """Per reference hand-over: frame, new team, spell frames, contact and engine-found flags."""
    eng_hos = reference_handovers(labels, blocks)
    eng_by_team = {t: sorted(f for f, _, new in eng_hos if new == t) for t in TEAMS}
    out = []
    for t, old, new in reference_handovers(ref, blocks):
        out.append({"t": t, "old": old, "new": new, "spell": spell_frames(ref, t, end_of[t]),
                    "contact": bool(near_frames([t], contact_frames, MATCH_TOL)[0]),
                    "found": bool(near_frames([t], eng_by_team[new], MATCH_TOL)[0])})
    return out


def _splices(cands, labels, owner_frames, blocks, refs, fps):
    """The splice grid; `refs` maps a reference key to (gt_codes, base_codes, runs)."""
    import score_game as sg

    out = []
    for guard in GUARDS:
        for floor in SPLICE_FLOORS:
            for t in THRESHOLDS:
                admitted = [(cd["c"], cd["new"]) for cd in pick(cands, floor=floor)
                            if gate(cd, t, guard)]
                spliced = splice_labels(labels, admitted, owner_frames, blocks)
                row = {"guard": guard, "floor": floor, "T": t, "admitted": len(admitted),
                       "frames_changed": int((spliced != labels).sum()),
                       "switch_rate": switch_stats(spliced, blocks, fps)["rate"]}
                codes = sg.pm.label_codes(spliced)
                for key, (gt_codes, base_codes, runs) in refs.items():
                    flags, n_pred = sg._spell_flags(gt_codes, codes, runs)
                    row[key] = {"spliced": float(sg.score_codes(gt_codes, codes)[0]
                                                 ["spell_f1_mean"]),
                                "paired": sg.paired_delta(gt_codes, codes, base_codes),
                                "flags": flags_text(flags), "n_pred": int(n_pred)}
                out.append(row)
    return out


def _public(path, cands, n):
    """Public labels around each public-window candidate with shape >= LIST_T."""
    import pandas as pd

    table = pd.read_csv(path, dtype={"frame": np.int64, "label": str})
    lab = np.full(int(n), "?", dtype="<U1")
    lab[table["frame"].to_numpy()] = table["label"].to_numpy().astype("<U1")
    out = []
    for cd in cands:
        c = cd["c"]
        if not (PUBLIC_WINDOW[0] <= c <= PUBLIC_WINDOW[1]) or not gate(cd, LIST_T):
            continue
        lo, hi = max(PUBLIC_WINDOW[0], c - MATCH_TOL), min(PUBLIC_WINDOW[1], c + MATCH_TOL)
        seg = lab[lo:hi + 1]
        hand = bool(((seg[1:] == cd["new"]) & (seg[:-1] == cd["owner"])).any())
        out.append({"c": c, "public_c": str(lab[c]),
                    "public_after": str(lab[min(c + MATCH_TOL, PUBLIC_WINDOW[1])]),
                    "public_handover": hand})
    return out


def collect(args):
    """The `collect` command: switch rates and per-candidate summary numbers for one match."""
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
    if abs(J_BANDS[-1][1] - cfg.DV_CONTACT) > 1e-9:
        raise SystemExit("the top band must end at DV_CONTACT {}".format(cfg.DV_CONTACT))
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
    size = events.impulse(ball, h)
    shape = events.match_team_shape(possession.engine_tactical(cfg, game), cfg)

    maxima = weak_maxima(size, alive, h, J_LO, cfg.DV_CONTACT)
    with np.errstate(invalid="ignore"):
        strong_frames = np.flatnonzero(np.isfinite(size) & (size >= cfg.DV_CONTACT) & alive)
    strong = events.window_maxima(size, h, strong_frames)
    # Check: no strong maximum lies within h frames of a weak maximum.
    for c in maxima:
        k = int(np.searchsorted(strong, int(c) - h, side="left"))
        if k < strong.size and abs(int(strong[k]) - int(c)) <= h:
            raise RuntimeError("a strong maximum lies within h of the weak maximum {}".format(c))
    reach_shape = shape if cfg.REACH_SHAPE_CONFIRM else None
    contacts = events._bump_contacts(ball, feats, None, alive, cfg, shape=reach_shape)
    contact_frames = sorted(int(c[0]) for c in contacts if graded[int(c[0])])

    cands, counts = find_weak(maxima, size, speed, d_h, d_a, labels, graded, h, cfg.R_TOUCH,
                              cfg.REDIRECT_FRACTION)
    for cd in cands:
        c = cd["c"]
        end = c + lag
        cd["shape"] = _f(events.shape_response(shape, c, cd["new"], lag))
        cd["shape_clean"] = bool(end < n and dead_cum[end + 1] - dead_cum[c] == 0)
        cd["speed"] = _f(speed[c])
        if gate(cd, cfg.SHAPE_CONFIRM_MS) != events.shape_confirms(shape, c, cd["new"],
                                                                   dead_cum, cfg):
            raise RuntimeError("gate differs from events.shape_confirms at {}".format(c))

    out = {"game": game, "view": args.view, "fps": fps, "h": h,
           "engine_md5": labels_md5(labels), "n_frames": int(n),
           "switch": {"engine": switch_stats(labels, blocks, fps)},
           "has_ref": bool(args.ref), "has_alt": bool(args.alt_ref), "counts": counts,
           "n_contacts": len(contact_frames), "n_strong": int(sum(1 for c in strong
                                                                 if graded[int(c)])),
           "candidates": cands, "handovers": [], "alt_handovers": [], "splices": [],
           "public": None, "ladder": None,
           "cfg": {"R_TOUCH": cfg.R_TOUCH, "CONTACT_HALF_WINDOW": h,
                   "DV_CONTACT": cfg.DV_CONTACT, "SHAPE_LAG_FRAMES": lag,
                   "SHAPE_CONFIRM_MS": cfg.SHAPE_CONFIRM_MS, "config_hash": cfg.config_hash()}}

    refs = {}
    base_codes = sg.pm.label_codes(labels)
    for key, path, ho_key in (("ref", args.ref, "handovers"),
                              ("alt", args.alt_ref, "alt_handovers")):
        if not path:
            continue
        ref = read_labels(path, int(games.GAMES[game]["n_video_frames"]))
        _, _, end_of = _truth(cands, ref, blocks, labels, key)
        out[ho_key] = _handover_records(ref, blocks, labels, contact_frames, end_of)
        out["switch"][key] = switch_stats(ref, blocks, fps)
        gt_codes = sg.pm.label_codes(ref)
        runs = sg.pm.valid_runs(gt_codes)
        base_flags, base_n = sg._spell_flags(gt_codes, base_codes, runs)
        out["base_" + key] = {"spell_f1": float(sg.score_codes(gt_codes, base_codes)[0]
                                                ["spell_f1_mean"]),
                              "flags": flags_text(base_flags), "n_pred": int(base_n),
                              "name": os.path.basename(path)}
        refs[key] = (gt_codes, base_codes, runs)
    owner_frames = [int(f) for f, _ in parts["owner_events"]]
    out["splices"] = _splices(cands, labels, owner_frames, blocks, refs, fps)
    if args.public_gt:
        out["public"] = _public(args.public_gt, cands, n)
    if args.dv_ladder:
        ladder = []
        for dv in DV_LADDER:
            cfg_dv = dataclasses.replace(cfg, DV_CONTACT=float(dv))
            lab_dv = np.asarray(possession.run(cfg_dv, game=game)).astype("<U1")
            ladder.append(dict(switch_stats(lab_dv, blocks, fps), dv=float(dv),
                               md5=labels_md5(lab_dv)))
        out["ladder"] = ladder
    write_json(args.out, out)
    return out


# -- report ---------------------------------------------------------------------------------------

def band_name(i):
    """`5-6` for a band index, `5-9` for None."""
    if i is None:
        return "{:g}-{:g}".format(J_BANDS[0][0], J_BANDS[-1][1])
    lo, hi = J_BANDS[i]
    return "{:g}-{:g}".format(lo, hi)


def provider(game):
    """`metrica`, `dfl` or `bar` from the game name."""
    if game.startswith("METRICA"):
        return "metrica"
    if game.startswith("DFL"):
        return "dfl"
    return "bar"


def _truth_of(cd, key="ref"):
    return bool(cd.get(key) and cd[key]["true"])


def _gate_cell(sel, t, guard, key="ref"):
    adm = [c for c in sel if gate(c, t, guard)]
    tt = sum(1 for c in adm if _truth_of(c, key))
    return "{}/{} ({})".format(tt, len(adm) - tt,
                               "-" if not adm else "{:.0f} %".format(100.0 * tt / len(adm)))


def report_overview(payloads, lines):
    """Per-match engine md5, switch rates and counts."""
    lines.append("### 0. Matches and counts")
    lines.append("")
    lines.append("| match | view | engine md5 | alive min (engine) | strong maxima | weak maxima | "
                 "no owner | new team far | candidates uncontested / contested | reference "
                 "hand-overs |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for p in payloads:
        k = p["counts"]
        cands = p["candidates"]
        lines.append("| {} | {} | `{}` | {:.1f} | {} | {} | {} | {} | {} / {} | {} |".format(
            short(p["game"]), p["view"], p["engine_md5"][:8], p["switch"]["engine"]["alive_min"],
            p["n_strong"], k["maxima"], k["no_owner"], k["new_far"],
            sum(1 for c in cands if not c["contested"]), sum(1 for c in cands if c["contested"]),
            len(p["handovers"]) if p["has_ref"] else "-"))
    lines.append("")


def report_switch(payloads, lines):
    """Measurement 1: switch rates of the engine and of the references."""
    lines.append("### 1. Switch rate: H<->A switches per alive minute (band rule: at most "
                 "{:.1f} on the engine)".format(BAND_RULE))
    lines.append("")
    lines.append("| match | view | engine switches | engine alive min | engine rate | engine "
                 "possessions | reference switches | reference alive min | reference rate | "
                 "reference possessions | engine / reference |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    groups = {}
    for p in payloads:
        e = p["switch"]["engine"]
        r = p["switch"].get("ref")
        cells = ["-"] * 5
        if r:
            cells = [str(r["switches"]), "{:.1f}".format(r["alive_min"]),
                     "{:.2f}".format(r["rate"]), str(r["possessions"]),
                     "{:.2f}".format(e["rate"] / r["rate"]) if r["rate"] else "-"]
            groups.setdefault((p["view"], provider(p["game"])), []).append((e["rate"],
                                                                           r["rate"]))
        lines.append("| {} | {} | {} | {:.1f} | {:.2f} | {} | {} |".format(
            short(p["game"]), p["view"], e["switches"], e["alive_min"], e["rate"],
            e["possessions"], " | ".join(cells)))
    lines.append("")
    if groups:
        lines.append("Mean rate per view and provider (engine / reference):")
        lines.append("")
        lines.append("| view | provider | matches | engine rate | reference rate |")
        lines.append("|---|---|---|---|---|")
        for (view, prov), rows in groups.items():
            lines.append("| {} | {} | {} | {:.2f} | {:.2f} |".format(
                view, prov, len(rows), float(np.mean([a for a, _ in rows])),
                float(np.mean([b for _, b in rows]))))
        lines.append("")
    for p in payloads:
        if not p.get("ladder"):
            continue
        lines.append("DV_CONTACT ladder on {} (label-free, the engine with only DV_CONTACT "
                     "changed):".format(short(p["game"])))
        lines.append("")
        lines.append("| DV_CONTACT m/s | switches | alive min | rate | possessions | md5 |")
        lines.append("|---|---|---|---|---|---|")
        for row in p["ladder"]:
            lines.append("| {:g} | {} | {:.1f} | {:.2f} | {} | `{}` |".format(
                row["dv"], row["switches"], row["alive_min"], row["rate"], row["possessions"],
                row["md5"][:8]))
        lines.append("")


def report_population(group, lines, key="ref"):
    """Measurement 2: the candidates per band and class, pooled and per match."""
    cands = [cd for p in group for cd in p["candidates"]]
    minutes = sum(p["switch"]["engine"]["alive_min"] for p in group)
    lines.append("#### 2a. Candidates per |J| band and class, pooled: n (per alive minute), true "
                 "hand-overs (share)")
    lines.append("")
    lines.append("| band m/s | uncontested | contested | both |")
    lines.append("|---|---|---|---|")
    for band in list(range(len(J_BANDS))) + [None]:
        cells = []
        for cls in CLASSES + (None,):
            sel = pick(cands, band, cls)
            tt = sum(1 for c in sel if _truth_of(c, key))
            cells.append("{} ({:.2f}), {}".format(len(sel), len(sel) / minutes if minutes else 0,
                                                  _pct(tt, len(sel))))
        lines.append("| {} | {} |".format(band_name(band), " | ".join(cells)))
    lines.append("")
    lines.append("#### 2b. Per match: candidates (per alive minute), true share; uncontested / "
                 "contested")
    lines.append("")
    lines.append("| match | " + " | ".join(band_name(b) for b in range(len(J_BANDS)))
                 + " | all |")
    lines.append("|---|" + "---|" * (len(J_BANDS) + 1))
    for p in group:
        m = p["switch"]["engine"]["alive_min"]
        cells = []
        for b in list(range(len(J_BANDS))) + [None]:
            parts = []
            for cls in CLASSES:
                sel = pick(p["candidates"], b, cls)
                tt = sum(1 for c in sel if _truth_of(c, key))
                parts.append("{} ({:.2f}) {}".format(len(sel), len(sel) / m if m else 0,
                                                     _pct(tt, len(sel)).split(" ", 1)[-1]
                                                     if sel else "-"))
            cells.append(" / ".join(parts))
        lines.append("| {} | {} |".format(short(p["game"]), " | ".join(cells)))
    lines.append("")
    lines.append("#### 2c. Candidates by kind (ds rule of _bump_contacts), pooled: n, true share, "
                 "gate 1.45 true / false")
    lines.append("")
    lines.append("| kind | uncontested | contested | uncontested, gate 1.45 | contested, "
                 "gate 1.45 |")
    lines.append("|---|---|---|---|---|")
    for kind in KINDS:
        sel = [c for c in cands if c["kind"] == kind]
        cells = []
        for cls in CLASSES:
            s = pick(sel, cls=cls)
            cells.append(_pct(sum(1 for c in s if _truth_of(c, key)), len(s)))
        for cls in CLASSES:
            cells.append(_gate_cell(pick(sel, cls=cls), 1.45, "all", key))
        lines.append("| {} | {} |".format(kind, " | ".join(cells)))
    lines.append("")


def report_gates(group, lines, key="ref", title=""):
    """Measurement 3: the gate curves per band, class and guard, pooled."""
    cands = [cd for p in group for cd in p["candidates"]]
    lines.append("#### 3a. Gate shape >= T{}: admitted true / false (precision), pooled".format(
        title))
    lines.append("")
    lines.append("| band m/s | candidates | n (true) | " + " | ".join(
        "T {}".format(t) for t in THRESHOLDS) + " |")
    lines.append("|---|---|---|" + "---|" * len(THRESHOLDS))
    rows = [(b, None) for b in range(len(J_BANDS))] + [(None, None)]
    rows += [(b, "uncontested") for b in range(len(J_BANDS))] + [(None, "uncontested")]
    rows += [(None, "contested")]
    for band, cls in rows:
        sel = pick(cands, band, cls)
        lines.append("| {} | {} | {} ({}) | {} |".format(
            band_name(band), cls or "both (guard off)", len(sel),
            sum(1 for c in sel if _truth_of(c, key)),
            " | ".join(_gate_cell(sel, t, "all", key) for t in THRESHOLDS)))
    lines.append("")
    lines.append("Floors (|J| from the floor up to DV_CONTACT), guard on (uncontested):")
    lines.append("")
    lines.append("| floor m/s | n (true) | " + " | ".join("T {}".format(t) for t in THRESHOLDS)
                 + " |")
    lines.append("|---|---|" + "---|" * len(THRESHOLDS))
    for floor in SPLICE_FLOORS:
        sel = pick(cands, cls="uncontested", floor=floor)
        lines.append("| {:g} | {} ({}) | {} |".format(
            floor, len(sel), sum(1 for c in sel if _truth_of(c, key)),
            " | ".join(_gate_cell(sel, t, "all", key) for t in THRESHOLDS)))
    lines.append("")
    lines.append("#### 3b. Per match, gate 1.45: admitted true / false, guard off and on")
    lines.append("")
    lines.append("| match | guard off | guard on (uncontested) | guard on, |J| >= 7 |")
    lines.append("|---|---|---|---|")
    for p in group:
        c = p["candidates"]
        lines.append("| {} | {} | {} | {} |".format(
            short(p["game"]), _gate_cell(c, 1.45, "all", key),
            _gate_cell(c, 1.45, "uncontested", key),
            _gate_cell(pick(c, floor=7.0), 1.45, "uncontested", key)))
    lines.append("")


def report_touch_and_lose(group, lines, key="ref", ho_key="handovers"):
    """Measurement 3: spells under 1 s and 2 s among admitted true candidates, and the reach."""
    fps = group[0]["fps"]
    cands = [cd for p in group for cd in p["candidates"]]
    hos = [r for p in group for r in p[ho_key]]
    lines.append("#### 3c. Touch-and-lose: admitted true candidates whose reference spell lasts "
                 "under 1 s / under 2 s")
    lines.append("")
    base1 = sum(1 for r in hos if r["spell"] < SHORT_S[0] * fps)
    base2 = sum(1 for r in hos if r["spell"] < SHORT_S[1] * fps)
    lines.append("All reference hand-overs of the view: {} ; under 1 s {} ; under 2 s {}.".format(
        len(hos), _pct(base1, len(hos)), _pct(base2, len(hos))))
    lines.append("")
    lines.append("| guard | " + " | ".join("T {}".format(t) for t in THRESHOLDS) + " |")
    lines.append("|---|" + "---|" * len(THRESHOLDS))
    for guard in GUARDS:
        cells = []
        for t in THRESHOLDS:
            tr = [c for c in cands if gate(c, t, guard) and _truth_of(c, key)]
            s1 = sum(1 for c in tr if c[key]["spell"] < SHORT_S[0] * fps)
            s2 = sum(1 for c in tr if c[key]["spell"] < SHORT_S[1] * fps)
            cells.append("{} true: {} / {}".format(len(tr), _pct(s1, len(tr)).split(" ", 1)[-1]
                                                   if tr else "-",
                                                   _pct(s2, len(tr)).split(" ", 1)[-1]
                                                   if tr else "-"))
        lines.append("| {} | {} |".format(guard, " | ".join(cells)))
    lines.append("")
    lines.append("#### 3d. Reach: distinct reference hand-overs matched by an admitted true "
                 "candidate, by spell length and contact")
    lines.append("")
    short_n = sum(1 for r in hos if r["spell"] < SHORT_S[1] * fps)
    long_n = len(hos) - short_n
    none_n = sum(1 for r in hos if not r["contact"])
    lines.append("Hand-overs: {} with a spell under 2 s, {} at least 2 s; {} with no contact row "
                 "within {} frames (the recall ceiling).".format(short_n, long_n, none_n,
                                                                 MATCH_TOL))
    lines.append("")
    lines.append("| guard | T | reached | spell < 2 s | spell >= 2 s | short / long recall ratio | "
                 "no contact |")
    lines.append("|---|---|---|---|---|---|---|")
    for guard in GUARDS:
        for t in THRESHOLDS:
            reached = set()
            for p in group:
                for c in p["candidates"]:
                    if gate(c, t, guard) and _truth_of(c, key):
                        reached.add((p["game"], c[key]["ho_t"]))
            recs = [(p["game"], r) for p in group for r in p[ho_key]]
            hit = [r for g, r in recs if (g, r["t"]) in reached]
            hs = sum(1 for r in hit if r["spell"] < SHORT_S[1] * fps)
            hl = len(hit) - hs
            hn = sum(1 for r in hit if not r["contact"])
            ratio = ("{:.2f}".format((hs / short_n) / (hl / long_n))
                     if short_n and long_n and hl else "-")
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                guard, t, len(hit), _pct(hs, short_n), _pct(hl, long_n), ratio,
                _pct(hn, none_n)))
    lines.append("")


def report_cost(group, lines, key="ref"):
    """Measurement 4: cost classes of the admitted false candidates."""
    cands = [cd for p in group for cd in p["candidates"]]
    lines.append("#### 4a. Admitted false candidates by cost class, pooled: shift / spurious / "
                 "ref_new / ref_dead")
    lines.append("")
    lines.append("| guard | floor m/s | " + " | ".join("T {}".format(t) for t in THRESHOLDS)
                 + " |")
    lines.append("|---|---|" + "---|" * len(THRESHOLDS))
    for guard in GUARDS:
        for floor in (7.0, 5.0):
            cells = []
            for t in THRESHOLDS:
                bad = [c for c in pick(cands, floor=floor)
                       if gate(c, t, guard) and not _truth_of(c, key)]
                cells.append(" / ".join(str(sum(1 for c in bad if c[key]["cost"] == k))
                                        for k in ("shift", "spurious", "ref_new", "ref_dead")))
            lines.append("| {} | {:g} | {} |".format(guard, floor, " | ".join(cells)))
    lines.append("")


def _cross(group, key, guard, floor, t):
    """paired_summary() of the spliced labels against the engine over the matches of `group`."""
    import paired_summary as ps

    side_a, side_b = {}, {}
    for p in group:
        base = p["base_" + key]
        row = [r for r in p["splices"] if r["guard"] == guard and r["floor"] == floor
               and r["T"] == t][0]
        name = p["game"] + "/" + base["name"]
        side_a[name] = (name, flags_array(row[key]["flags"]), row[key]["n_pred"], "spliced")
        side_b[name] = (name, flags_array(base["flags"]), base["n_pred"], "engine")
    return ps.paired_summary(side_a, side_b, 1000, 0)


def report_splice(group, lines, key="ref", title=""):
    """Measurement 4: the splice grid per match and the means by provider."""
    lines.append("#### 4b. Splice (not an engine run){}: spliced minus engine, spellF1, paired "
                 "delta (admitted, true)".format(title))
    lines.append("")
    names = [short(p["game"]) for p in group]
    metrica = [p for p in group if provider(p["game"]) == "metrica"]
    dfl = [p for p in group if provider(p["game"]) == "dfl"]
    lines.append("| guard | floor | T | " + " | ".join(names) + " | Metrica mean | DFL mean | "
                 "mean of {} [5-95 %] |".format(len(group)))
    lines.append("|---|---|---|" + "---|" * len(names) + "---|---|---|")
    summary = []
    for guard in GUARDS:
        for floor in SPLICE_FLOORS:
            for t in THRESHOLDS:
                cells = []
                ds = {}
                for p in group:
                    row = [r for r in p["splices"] if r["guard"] == guard
                           and r["floor"] == floor and r["T"] == t][0]
                    tt = sum(1 for c in pick(p["candidates"], floor=floor)
                             if gate(c, t, guard) and _truth_of(c, key))
                    pr = row[key]["paired"]
                    mark = "*" if pr["low"] > 0 else ("!" if pr["high"] < 0 else "")
                    cells.append("{:+.4f}{} ({}, {})".format(pr["delta"], mark, row["admitted"],
                                                             tt))
                    ds[p["game"]] = pr["delta"]
                mm = ("{:+.4f}".format(np.mean([ds[p["game"]] for p in metrica]))
                      if metrica else "-")
                dm = "{:+.4f}".format(np.mean([ds[p["game"]] for p in dfl])) if dfl else "-"
                cr = _cross(group, key, guard, floor, t)
                summary.append((guard, floor, t, mm, dm, cr))
                lines.append("| {} | {:g} | {} | {} | {} | {} | {:+.4f} [{:+.4f}, {:+.4f}] |"
                             .format(guard, floor, t, " | ".join(cells), mm, dm,
                                     cr["mean_delta"], cr["low"], cr["high"]))
    lines.append("")
    lines.append("`*`: the match's paired interval lies above zero; `!`: below zero.")
    lines.append("")


def report_barath(p, lines):
    """Measurement 5: BAR-ATH, label-free counts and the public-window candidates."""
    cands = p["candidates"]
    minutes = p["switch"]["engine"]["alive_min"]
    lines.append("#### 5a. BAR-ATH full match (label-free): candidates per band, class and gate, "
                 "count (per alive minute)")
    lines.append("")
    lines.append("Engine md5 `{}`; {} graded strong maxima, {} contact rows, {} graded weak "
                 "maxima, {:.1f} alive graded minutes (engine labels H or A); weak maxima left "
                 "out: {} with no engine owner, {} with the new team beyond R_TOUCH.".format(
                     p["engine_md5"], p["n_strong"], p["n_contacts"], p["counts"]["maxima"],
                     minutes, p["counts"]["no_owner"], p["counts"]["new_far"]))
    lines.append("")
    gates = (None,) + THRESHOLDS
    lines.append("| band m/s | class | " + " | ".join(
        "all" if g is None else "shape >= {}".format(g) for g in gates) + " |")
    lines.append("|---|---|" + "---|" * len(gates))
    rows = [(b, cls) for b in range(len(J_BANDS)) for cls in CLASSES]
    rows += [(None, "uncontested"), (None, "contested"), (None, None)]
    for band, cls in rows:
        sel = pick(cands, band, cls)
        cells = []
        for g in gates:
            k = len(sel) if g is None else sum(1 for c in sel if gate(c, g))
            cells.append("{} ({:.2f})".format(k, k / minutes if minutes else 0.0))
        lines.append("| {} | {} | {} |".format(band_name(band), cls or "both", " | ".join(cells)))
    lines.append("")
    lines.append("Splice grid on BAR-ATH (label-free, nothing scored): admitted candidates / "
                 "frames changed / H<->A switches per alive minute of the spliced labels (engine "
                 "{:.2f}).".format(p["switch"]["engine"]["rate"]))
    lines.append("")
    lines.append("| guard | floor m/s | " + " | ".join("T {}".format(t) for t in THRESHOLDS)
                 + " |")
    lines.append("|---|---|" + "---|" * len(THRESHOLDS))
    for guard in GUARDS:
        for floor in SPLICE_FLOORS:
            cells = []
            for t in THRESHOLDS:
                row = [r for r in p["splices"] if r["guard"] == guard and r["floor"] == floor
                       and r["T"] == t][0]
                rate = row.get("switch_rate")
                cells.append("{} / {} / {}".format(row["admitted"], row["frames_changed"],
                                                   "-" if rate is None else
                                                   "{:.2f}".format(rate)))
            lines.append("| {} | {:g} | {} |".format(guard, floor, " | ".join(cells)))
    lines.append("")
    if p.get("public") is None:
        return
    pub = {r["c"]: r for r in p["public"]}
    lines.append("#### 5b. BAR-ATH public window {}..{}: candidates with shape >= {} (frames and "
                 "labels only)".format(PUBLIC_WINDOW[0], PUBLIC_WINDOW[1], LIST_T))
    lines.append("")
    lines.append("| frame | abs J m/s | band | owner -> new | contested | d_new m | d_own m | "
                 "shape m/s | admitted at T | public label at c | public label at c + {} | "
                 "public hand-over to new within {} |".format(MATCH_TOL, MATCH_TOL))
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cd in cands:
        if cd["c"] not in pub:
            continue
        r = pub[cd["c"]]
        adm = ", ".join("{}".format(t) for t in THRESHOLDS if gate(cd, t))
        lines.append("| {} | {:.2f} | {} | {} -> {} | {} | {} | {} | {} | {} | {} | {} | {} |"
                     .format(cd["c"], cd["j"], band_name(cd["band"]), cd["owner"], cd["new"],
                             cd["contested"], _fmt(cd["d_new"]), _fmt(cd["d_own"]),
                             _fmt(cd["shape"]), adm, r["public_c"], r["public_after"],
                             r["public_handover"]))
    if not pub:
        lines.append("| none | - | - | - | - | - | - | - | - | - | - | - |")
    lines.append("")


def report(args):
    """The `report` command: print the markdown tables."""
    payloads = []
    for path in args.paths:
        with open(path) as handle:
            payloads.append(json.load(handle))
    lines = []
    report_overview(payloads, lines)
    report_switch(payloads, lines)
    views = []
    for p in payloads:
        if p["has_ref"] and p["view"] not in views:
            views.append(p["view"])
    for view in views:
        group = [p for p in payloads if p["view"] == view and p["has_ref"]]
        lines.append("### View: {} ({} matches)".format(view, len(group)))
        lines.append("")
        report_population(group, lines)
        report_gates(group, lines)
        report_touch_and_lose(group, lines)
        report_cost(group, lines)
        report_splice(group, lines)
        alt = [p for p in group if p["has_alt"]]
        if alt:
            lines.append("#### Alternative reference (F2 variant): {}".format(
                ", ".join(short(p["game"]) for p in alt)))
            lines.append("")
            report_gates(alt, lines, "alt", " against the variant reference")
            report_touch_and_lose(alt, lines, "alt", "alt_handovers")
            report_splice(alt, lines, "alt", ", against the variant reference")
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
    c.add_argument("--alt-ref", help="a second reference of the same match (the F2 variant)")
    c.add_argument("--public-gt")
    c.add_argument("--dv-ladder", action="store_true",
                   help="also run the engine at each DV_LADDER value (label-free)")
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
