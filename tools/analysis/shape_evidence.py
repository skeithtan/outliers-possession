"""Team-shape evidence on the frames the decoder decides without a contact (P9-B). It measures.

    python tools/analysis/shape_evidence.py collect --game NAME --view VIEW [--ref REFERENCE.csv]
        [--public-gt PUBLIC_GT.csv] [--cache-dir DIR] [--config CONFIG.json] --out OUT.json
    python tools/analysis/shape_evidence.py report OUT.json [OUT.json ...] [--out REPORT.md]

`collect` runs the engine (possession.prepare and possession.run with the default Config, or the
overrides of `--config`) on one match. It stores counts, per-stretch summary numbers and the splice
scores in a JSON file. `report` prints the tables of docs/plan/calibration/P9B_shape_evidence.md.
Set MTAH_DATA_DIR for each match as tools/score_game.py needs it. `--cache-dir` points the
engine's npz cache somewhere else (each data root needs its own cache folder). `--view` is a free
label (raw, smooth, matched, barath): the report groups the matches by it. Write the JSON files
outside the repository. The script refuses the sealed matches of calibration/SPLITS.md. It
designs no rule and changes no engine file.

Definitions.

- Graded frame: a frame of a graded block of games.py. Alive frame: a graded frame outside the
  engine stoppage mask (possession.run parts["stoppage"]).
- Owner anchors, as in possession.owner_track: the engine owner events (parts["owner_events"]),
  the restarts of parts["restart_list"] that have a team, and the kickoffs with a team
  (parts["kick_teams"]).
- Contact rows: the rows of parts["events"] of the kinds touch, release, control, dribble and
  contested that are not owner events (team None for contested and for a contested dribble).
- Blind frame. A walk through each graded block in frame order. The state is a team or blind. It
  starts blind. On each frame the walk takes four steps in this order. A stoppage frame makes the
  state blind (cause "dead"). A contact row with no team makes it blind (cause "contested"). A
  contact row of the other team while the state is a team makes it blind (cause "contact"). An
  owner anchor sets the state to its team. An alive frame whose state is blind after these steps is a blind frame. The
  cause "start" is a block that opens before its first owner anchor. So a blind frame is an alive
  frame after a dead frame or after a contact that set no owner, up to the next owner anchor. The
  engine's own unknown owner (parts["owner"] == UNKNOWN on an alive frame) is a subset: the
  script counts the engine-unknown frames and checks that each one is a blind frame. On a blind
  frame where the engine owner is known, the engine holds the owner from before the dead frame or
  the contact ("held").
- Blind stretch: a maximal run of blind frames, cut at every frame with a contact row (the cut
  of possession.stretch_mean). The cause of a stretch is the cause at its first frame.
- Shape signal. vf_T is tracks.team_forward_velocity() (events.match_team_shape, the C15 array:
  mean forward x velocity of the visible players, centred mean over +-SHAPE_HALF_FRAMES, NaN under
  SHAPE_MIN_PLAYERS players). s_T(t) = vf_T(t) - vf_T(t - L), L = SHAPE_LAG_FRAMES (25: the C15
  rise read backwards), and d(t) = s_H(t) - s_A(t), in m/s. d(t) equals
  events.shape_response(shape, t - L, "H", L), and the script checks that on a sample of frames.
  d is NaN when a value is undefined, when t - L lies before the array, or when a stoppage frame
  lies in [t - L, t] (the clean condition of events.shape_confirms). The shape team is H when
  d > 0 and A when d < 0. The centred mean reads vf up to SHAPE_HALF_FRAMES (6) frames after t,
  so the signal is causal up to 0.24 s.
- Nearest team: the sign of possession.nearest_log_odds at t (parts["near_lo"]: the nearest
  player at t - NEAREST_LAG within NEAR_FAR_M, not on a ghost frame), H when > 0, A when < 0,
  no vote at 0. The per-frame term the decoder uses on an unknown owner. "Nearest stretch" is the
  sign of the mean of near_lo + tag_lo over the blind stretch: the value of
  possession.stretch_mean that the emission reads where the owner is unknown.
- Agreement against a reference: frames where the reference label is H or A and the signal has a
  vote. |d| buckets: BUCKETS. The combined matrix crosses the shape vote and the nearest vote:
  both agree, both disagree (which one the reference backs), one vote only, no vote. Per
  stretch: the majority reference label of the stretch (H or A only) against the sign of d_bar
  (below), the nearest stretch and the majority engine label, by |d_bar| bucket.
- Splice. Not an engine run. Start from the engine labels. For each blind stretch whose mean d
  over its frames with a finite d (d_bar) has |d_bar| >= T and whose shape team differs from the
  majority engine label (H or A) of the stretch, set every H or A frame of the stretch to the
  shape team. A stretch with no finite d is never changed. Score the splice against the reference
  with tools/score_game.paired_delta() (the `--versus` bootstrap of tools/score_game.py: spliced
  minus engine, 1000 draws, seed 0). T in SPLICE_T.
- BAR-ATH. Label-free on the full match. With `--public-gt` the script reads the public file for
  the blind stretches that lie inside the public window PUBLIC_WINDOW only: frame ranges, the
  majority public label and its share. Nothing outside the window is read from it.

The script stores counts and per-stretch summary numbers (frames, means, labels), never a tracking
row. numpy and pandas only. Deterministic.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "tools", "analysis")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from tools.analysis.handover_confirm import (TEAMS, _fmt, _pct, graded_mask,  # noqa: E402
                                             labels_md5, quantiles, read_labels, short,
                                             write_json)

SEALED = ("DFL-J03WQQ", "DFL-J03WR9")       # calibration/SPLITS.md
CONTACT_KINDS = ("touch", "release", "control", "dribble", "contested")
BUCKETS = ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, np.inf))   # |d| m/s, the card's
DISAGREE_T = 1.45          # m/s, the card's disagreement threshold (= SHAPE_CONFIRM_MS)
SPLICE_T = (1.45, 2.0, 2.5)                  # m/s, the card's splice thresholds
HALF_SHARE_T = 1.45        # m/s, measurement 4: |d| >= this on at least half the stretch frames
LENGTH_EDGES = (0, 10, 25, 50, 125, 250, 10 ** 9)   # frames, the stretch length histogram
PUBLIC_WINDOW = (11185, 18684)   # team rules: the public window of BAR-ATH, video frames
CAUSES = ("start", "dead", "contested", "contact")
POPULATIONS = ("blind", "blind_unknown", "blind_held", "nonblind")
CHECK_STEP = 997           # frames between two checks of d against events.shape_response


# -- pure parts (tested on synthetic arrays) --------------------------------------------------

def blind_walk(n, blocks, stoppage, anchors, contacts):
    """(blind bool (n,), cause '<U9' (n,), cut bool (n,)) of the walk in the module docstring.

    `anchors` is [(frame, team)] of the owner anchors, `contacts` [(frame, team or None)] of the
    contact rows. `cause` is '' where the frame is not blind. `cut` marks the frames with a
    contact row.
    """
    stoppage = np.asarray(stoppage, dtype=bool)
    blind = np.zeros(int(n), dtype=bool)
    cause = np.full(int(n), "", dtype="<U9")
    cut = np.zeros(int(n), dtype=bool)
    by_frame_c, by_frame_a = {}, {}
    for f, t in contacts:
        by_frame_c.setdefault(int(f), []).append(t if t in TEAMS else None)
    for f, t in anchors:
        if t in TEAMS:
            by_frame_a.setdefault(int(f), []).append(t)
    for f in by_frame_c:
        if 0 <= f < n:
            cut[f] = True
    for _, first, last in blocks:
        state, why = None, "start"
        for t in range(int(first), int(last) + 1):
            if stoppage[t]:
                state, why = None, "dead"
            for team in by_frame_c.get(t, ()):
                if team is None:
                    state, why = None, "contested"
                elif state is not None and team != state:
                    state, why = None, "contact"
            for team in by_frame_a.get(t, ()):
                state = team
            if state is None and not stoppage[t]:
                blind[t] = True
                cause[t] = why
    return blind, cause, cut


def stretches(blind, cut):
    """[(start, stop)] half-open: maximal runs of `blind`, cut at every `cut` frame."""
    blind = np.asarray(blind, dtype=bool)
    cut = np.asarray(cut, dtype=bool)
    n = blind.size
    out = []
    t = 0
    while t < n:
        if not blind[t]:
            t += 1
            continue
        start = t
        t += 1
        while t < n and blind[t] and not cut[t]:
            t += 1
        out.append((start, t))
    return out


def shape_signal(vf_h, vf_a, lag, stoppage):
    """float64 (n,): d(t) = s_H(t) - s_A(t), s_T(t) = vf_T(t) - vf_T(t - lag) (module docstring).

    NaN where a value is undefined, t < lag, or a `stoppage` frame lies in [t - lag, t].
    """
    vf_h = np.asarray(vf_h, dtype=np.float64)
    vf_a = np.asarray(vf_a, dtype=np.float64)
    n = vf_h.size
    lag = int(lag)
    d = np.full(n, np.nan)
    if n > lag:
        d[lag:] = (vf_h[lag:] - vf_h[:-lag]) - (vf_a[lag:] - vf_a[:-lag])
    cum = np.concatenate(([0], np.cumsum(np.asarray(stoppage, dtype=np.int64))))
    lo = np.clip(np.arange(n) - lag, 0, n)
    dirty = (cum[np.arange(n) + 1] - cum[lo]) > 0
    d[dirty] = np.nan
    d[:lag] = np.nan
    return d


def vote(values):
    """'<U1' (n,): H where value > 0, A where < 0, '' at 0 or NaN."""
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.size, "", dtype="<U1")
    with np.errstate(invalid="ignore"):
        out[values > 0] = "H"
        out[values < 0] = "A"
    return out


def bucket_of(abs_d):
    """int64 (n,): index of the |d| bucket of BUCKETS, -1 for NaN."""
    abs_d = np.asarray(abs_d, dtype=np.float64)
    out = np.full(abs_d.size, -1, dtype=np.int64)
    for i, (lo, hi) in enumerate(BUCKETS):
        with np.errstate(invalid="ignore"):
            out[(abs_d >= lo) & (abs_d < hi)] = i
    return out


def agreement(mask, ref, shape_v, abs_d, near_v, near_s, engine):
    """Counts on the frames of `mask` (module docstring). All inputs are (n,) arrays."""
    mask = np.asarray(mask, dtype=bool)
    ref_ha = mask & np.isin(ref, TEAMS)
    out = {"frames": int(mask.sum()),
           "ref": {lab: int((mask & (ref == lab)).sum()) for lab in ("H", "A", "D", "X")},
           "engine_right": int((ref_ha & (engine == ref)).sum()),
           "ref_ha": int(ref_ha.sum())}
    has_s = ref_ha & (shape_v != "")
    has_n = ref_ha & (near_v != "")
    s_ok = shape_v == ref
    n_ok = near_v == ref
    out["shape"] = {"vote": int(has_s.sum()), "right": int((has_s & s_ok).sum())}
    out["near"] = {"vote": int(has_n.sum()), "right": int((has_n & n_ok).sum())}
    has_ns = ref_ha & (near_s != "")
    out["near_stretch"] = {"vote": int(has_ns.sum()),
                           "right": int((has_ns & (near_s == ref)).sum())}
    buckets = bucket_of(abs_d)
    out["shape_bucket"] = []
    out["near_bucket"] = []
    for i in range(len(BUCKETS)):
        sel = has_s & (buckets == i)
        out["shape_bucket"].append({"vote": int(sel.sum()), "right": int((sel & s_ok).sum())})
        seln = has_n & (buckets == i)
        out["near_bucket"].append({"vote": int(seln.sum()), "right": int((seln & n_ok).sum())})
    both = has_s & has_n
    agree = both & (shape_v == near_v)
    disagree = both & (shape_v != near_v)
    strong = disagree & (abs_d >= DISAGREE_T)
    out["combined"] = {
        "agree": int(agree.sum()), "agree_right": int((agree & s_ok).sum()),
        "disagree": int(disagree.sum()), "disagree_shape": int((disagree & s_ok).sum()),
        "disagree_near": int((disagree & n_ok).sum()),
        "shape_only": int((has_s & ~has_n).sum()),
        "shape_only_right": int((has_s & ~has_n & s_ok).sum()),
        "near_only": int((has_n & ~has_s).sum()),
        "near_only_right": int((has_n & ~has_s & n_ok).sum()),
        "none": int((ref_ha & ~has_s & ~has_n).sum()),
        "strong_disagree": int(strong.sum()), "strong_shape": int((strong & s_ok).sum()),
        "strong_near": int((strong & n_ok).sum())}
    return out


def stretch_agreement(rows):
    """Per |d_bar| bucket: stretches whose majority reference label is H or A, and how many of
    them the shape team, the nearest stretch and the majority engine label name right.

    `rows` are the stretch records of collect() with the key "ref". Bucket -1 (no d_bar) is the
    last entry.
    """
    out = [{"n": 0, "shape": 0, "near_vote": 0, "near": 0, "engine": 0, "frames": 0}
           for _ in range(len(BUCKETS) + 1)]
    for r in rows:
        if r.get("ref") not in TEAMS:
            continue
        if r["d_bar"] is None:
            i = len(BUCKETS)
        else:
            i = int(bucket_of([abs(r["d_bar"])])[0])
        node = out[i]
        node["n"] += 1
        node["frames"] += r["len"]
        if r["d_bar"] is not None and ("H" if r["d_bar"] > 0 else "A") == r["ref"]:
            node["shape"] += 1
        if r["near_stretch"]:
            node["near_vote"] += 1
            node["near"] += int(r["near_stretch"] == r["ref"])
        node["engine"] += int(r["engine"] == r["ref"])
    return out


def splice_stretches(labels, rows, threshold):
    """(spliced labels, stretches changed) for one threshold (module docstring, Splice).

    `rows` are the stretch records of collect(): start, stop, d_bar, engine (majority H or A or
    '').
    """
    labels = np.asarray(labels).astype("<U1")
    out = labels.copy()
    changed = 0
    for r in rows:
        if r["d_bar"] is None or abs(r["d_bar"]) < threshold or r["engine"] not in TEAMS:
            continue
        team = "H" if r["d_bar"] > 0 else "A"
        if team == r["engine"]:
            continue
        seg = out[r["start"]:r["stop"]]
        seg[np.isin(seg, TEAMS)] = team
        changed += 1
    return out, changed


def majority(values):
    """(most common non-empty value, its share of all values) with a fixed tie order H, A, D, X."""
    values = np.asarray(values)
    if values.size == 0:
        return "", 0.0
    best, count = "", 0
    for lab in ("H", "A", "D", "X"):
        c = int((values == lab).sum())
        if c > count:
            best, count = lab, c
    return best, float(count) / values.size


def length_histogram(lengths):
    """Counts of stretch lengths in the bins of LENGTH_EDGES."""
    lengths = np.asarray(lengths, dtype=np.int64)
    return [int(((lengths >= lo) & (lengths < hi)).sum())
            for lo, hi in zip(LENGTH_EDGES[:-1], LENGTH_EDGES[1:])]


# -- collect ----------------------------------------------------------------------------------

def _f(value):
    """JSON-friendly float (None for NaN or inf)."""
    value = float(value)
    return value if np.isfinite(value) else None


def collect(args):
    """The `collect` command: counts and stretch records for one match."""
    import games
    import score_game as sg
    from engine import data, events, possession
    from engine.config import Config, load as load_config

    game = args.game
    if game in SEALED:
        raise SystemExit("{} is sealed (calibration/SPLITS.md)".format(game))
    if args.public_gt and game != "BAR-ATH":
        raise SystemExit("--public-gt is for BAR-ATH only")
    if game == "BAR-ATH" and args.ref:
        raise SystemExit("BAR-ATH has no full reference: use --public-gt")
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
    stoppage = np.asarray(parts["stoppage"], dtype=bool)
    alive = graded & ~stoppage
    lag = int(cfg.SHAPE_LAG_FRAMES)

    # Anchors and contact rows, as possession.owner_track reads them.
    owner_list = [(int(f), str(t)) for f, t in parts["owner_events"]]
    anchors = list(owner_list)
    anchors += [(int(f), t) for f, t in parts["restart_list"] if t in TEAMS]
    for half, team in sorted(parts["kick_teams"].items()):
        if team in TEAMS:
            anchors.append((int(inputs["kickoffs"][half]), team))
    setting = set(owner_list)
    table = parts["events"]
    rows = table[table["kind"].isin(CONTACT_KINDS)]
    contacts = []
    for f, t in zip(rows["frame"], rows["team"]):
        t = t if t in TEAMS else None
        if (int(f), t) in setting:
            continue
        contacts.append((int(f), t))
    blind, cause, cut = blind_walk(n, blocks, stoppage, anchors, contacts)
    engine_unknown = alive & (np.asarray(parts["owner"]) == possession.UNKNOWN)
    if np.any(engine_unknown & ~blind):
        raise RuntimeError("an engine-unknown alive frame is not a blind frame")

    shape = events.match_team_shape(possession.engine_tactical(cfg, game), cfg)
    d = shape_signal(shape["H"], shape["A"], lag, stoppage)
    for t in range(lag, n, CHECK_STEP):
        if np.isfinite(d[t]):
            ref_value = events.shape_response(shape, t - lag, "H", lag)
            if not np.isclose(ref_value, d[t]):
                raise RuntimeError("d differs from events.shape_response at {}".format(t))
    abs_d = np.abs(d)
    shape_v = vote(d)
    near_lo = np.asarray(parts["near_lo"], dtype=np.float64)
    tag_lo = np.asarray(parts["tag_lo"], dtype=np.float64)
    near_v = vote(near_lo)
    spans = stretches(blind, cut)
    near_s_value = np.zeros(n)
    for a, b in spans:
        near_s_value[a:b] = float(np.mean(near_lo[a:b] + tag_lo[a:b]))
    near_s = vote(near_s_value)

    ref = None
    if args.ref:
        ref = read_labels(args.ref, int(games.GAMES[game]["n_video_frames"]))
    public = None
    if args.public_gt:
        public = read_labels(args.public_gt, n)

    # Stretch records.
    stretch_rows = []
    for a, b in spans:
        dd = d[a:b]
        fin = dd[np.isfinite(dd)]
        eng, eng_share = majority(labels[a:b])
        rec = {"start": int(a), "stop": int(b), "len": int(b - a), "cause": str(cause[a]),
               "engine_unknown": int(engine_unknown[a:b].sum()),
               "engine": eng if eng in TEAMS else "", "engine_share": eng_share,
               "d_bar": _f(fin.mean()) if fin.size else None,
               "d_defined": int(fin.size),
               "strong_share": float((np.abs(fin) >= HALF_SHARE_T).sum()) / (b - a),
               "near_stretch": str(near_s[a]), "near_stretch_value": _f(near_s_value[a]),
               "near_frames_h": int((near_v[a:b] == "H").sum()),
               "near_frames_a": int((near_v[a:b] == "A").sum())}
        if ref is not None:
            lab, share = majority(ref[a:b])
            rec["ref"], rec["ref_share"] = lab, share
        stretch_rows.append(rec)

    # The public window list (BAR-ATH): the public file is read on these stretches only.
    public_rows = []
    if public is not None:
        lo, hi = PUBLIC_WINDOW
        for rec in stretch_rows:
            if rec["start"] < lo or rec["stop"] - 1 > hi:
                continue
            lab, share = majority(public[rec["start"]:rec["stop"]])
            row = dict(rec)
            row["public"], row["public_share"] = lab, share
            public_rows.append(row)

    # Agreement tables.
    pops = {"blind": blind, "blind_unknown": blind & engine_unknown,
            "blind_held": blind & ~engine_unknown, "nonblind": alive & ~blind}
    counts = {"alive": int(alive.sum()), "graded": int(graded.sum()),
              "blind": int(blind.sum()), "engine_unknown": int(engine_unknown.sum()),
              "stretches": len(spans),
              "by_cause": {c: {"frames": int((blind & (cause == c)).sum()),
                               "stretches": sum(1 for r in stretch_rows if r["cause"] == c)}
                           for c in CAUSES},
              "unknown_stretches": sum(1 for r in stretch_rows if r["engine_unknown"] > 0),
              "lengths": quantiles([r["len"] for r in stretch_rows], (10, 50, 90)),
              "length_hist": length_histogram([r["len"] for r in stretch_rows]),
              "d_defined_blind": int((blind & np.isfinite(d)).sum()),
              "half_strong": sum(1 for r in stretch_rows if r["strong_share"] >= 0.5),
              "half_strong_frames": sum(r["len"] for r in stretch_rows
                                        if r["strong_share"] >= 0.5),
              "unknown_half_strong": sum(1 for r in stretch_rows if r["strong_share"] >= 0.5
                                         and r["engine_unknown"] > 0)}
    agree = None
    splices = []
    if ref is not None:
        agree = {}
        for name, mask in pops.items():
            agree[name] = agreement(mask, ref, shape_v, abs_d, near_v, near_s, labels)
        for c in CAUSES:
            agree["cause_" + c] = agreement(blind & (cause == c), ref, shape_v, abs_d, near_v,
                                            near_s, labels)
        agree["stretch"] = stretch_agreement(stretch_rows)
        gt_codes = sg.pm.label_codes(ref)
        base_codes = sg.pm.label_codes(labels)
        base_f1 = float(sg.score_codes(gt_codes, base_codes)[0]["spell_f1_mean"])
        for t in SPLICE_T:
            spliced, changed = splice_stretches(labels, stretch_rows, t)
            diff = spliced != labels
            row = {"T": t, "stretches": changed, "frames_changed": int(diff.sum()),
                   "frames_ref_backs": int((diff & (spliced == ref)).sum()),
                   "frames_ref_engine": int((diff & (labels == ref)).sum()), "base": base_f1}
            codes = sg.pm.label_codes(spliced)
            row["spliced"] = float(sg.score_codes(gt_codes, codes)[0]["spell_f1_mean"])
            row["paired"] = sg.paired_delta(gt_codes, codes, base_codes)
            splices.append(row)
    else:
        for t in SPLICE_T:
            spliced, changed = splice_stretches(labels, stretch_rows, t)
            splices.append({"T": t, "stretches": changed,
                            "frames_changed": int((spliced != labels).sum())})

    out = {"game": game, "view": args.view, "fps": fps, "engine_md5": labels_md5(labels),
           "n_frames": int(n), "has_ref": ref is not None, "counts": counts,
           "agreement": agree, "splices": splices, "public": public_rows if public is not None
           else None,
           "stretches": stretch_rows if game == "BAR-ATH" else None,
           "cfg": {"SHAPE_LAG_FRAMES": lag, "NEAREST_LAG": cfg.NEAREST_LAG,
                   "NEAR_FAR_M": cfg.NEAR_FAR_M, "config_hash": cfg.config_hash()}}
    write_json(args.out, out)
    return out


# -- report -----------------------------------------------------------------------------------

def bucket_name(i):
    """`0.0-0.5` or `> 2.5` for a bucket index."""
    lo, hi = BUCKETS[i]
    return "> {:.1f}".format(lo) if not np.isfinite(hi) else "{:.1f}-{:.1f}".format(lo, hi)


def _acc(node):
    """`right/vote (p %)` of a {vote, right} node."""
    return _pct(node["right"], node["vote"])


def _sum(nodes):
    """Element-wise sum of nested count dicts and lists."""
    first = nodes[0]
    if isinstance(first, dict):
        return {k: _sum([nd[k] for nd in nodes]) for k in first}
    if isinstance(first, list):
        return [_sum([nd[i] for nd in nodes]) for i in range(len(first))]
    return sum(nodes)


def report_counts(payloads, lines):
    """Measurement 1: blind frames and stretches per match."""
    lines.append("| match | view | engine md5 | alive frames | blind frames (share) | engine "
                 "unknown | stretches | after dead | contested | other-team contact | block "
                 "start | length p10 / p50 / p90 | engine right on blind (ref H/A) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in payloads:
        c = p["counts"]
        bc = c["by_cause"]

        def cell(k):
            return "{} / {}".format(bc[k]["stretches"], bc[k]["frames"])

        q = c["lengths"]
        lens = "-" if not q["n"] else "{:.0f} / {:.0f} / {:.0f}".format(q["p10"], q["p50"],
                                                                         q["p90"])
        right = "-"
        if p["agreement"]:
            b = p["agreement"]["blind"]
            right = _pct(b["engine_right"], b["ref_ha"])
        lines.append("| {} | {} | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            short(p["game"]), p["view"], p["engine_md5"][:8], c["alive"],
            _pct(c["blind"], c["alive"]), c["engine_unknown"], c["stretches"], cell("dead"),
            cell("contested"), cell("contact"), cell("start"), lens, right))
    lines.append("")
    lines.append("Cause cells are stretches / frames.")
    lines.append("")
    lines.append("| match | view | " + " | ".join(
        "{}-{}".format(lo, hi if hi < 10 ** 8 else "") for lo, hi in
        zip(LENGTH_EDGES[:-1], LENGTH_EDGES[1:])) + " |")
    lines.append("|---|---|" + "---|" * (len(LENGTH_EDGES) - 1))
    for p in payloads:
        lines.append("| {} | {} | {} |".format(short(p["game"]), p["view"], " | ".join(
            str(x) for x in p["counts"]["length_hist"])))
    lines.append("")


def report_reference_labels(group, lines):
    """Measurement 1: the reference label on the blind frames and the engine's accuracy there."""
    lines.append("| match | blind frames | ref H | ref A | ref D | ref X | engine right "
                 "(ref H/A) | held: engine right | non-blind alive: engine right |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for p in group + [None]:
        if p is None:
            ag = {k: _sum([q["agreement"][k] for q in group]) for k in POPULATIONS}
            name = "pooled"
        else:
            ag, name = p["agreement"], short(p["game"])
        b = ag["blind"]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            name, b["frames"], b["ref"]["H"], b["ref"]["A"], b["ref"]["D"], b["ref"]["X"],
            _pct(b["engine_right"], b["ref_ha"]),
            _pct(ag["blind_held"]["engine_right"], ag["blind_held"]["ref_ha"]),
            _pct(ag["nonblind"]["engine_right"], ag["nonblind"]["ref_ha"])))
    lines.append("")


def report_agreement(group, lines):
    """Measurement 2: shape, nearest and combined agreement."""
    pooled = {k: _sum([q["agreement"][k] for q in group]) for k in group[0]["agreement"]}
    lines.append("#### 2a. Agreement with the reference owner, per match: right / vote (share)")
    lines.append("")
    lines.append("| match | blind: shape | blind: nearest | blind: nearest stretch | blind: "
                 "engine | non-blind: shape | non-blind: nearest |")
    lines.append("|---|---|---|---|---|---|---|")
    for p in group + [None]:
        ag = pooled if p is None else p["agreement"]
        name = "pooled" if p is None else short(p["game"])
        b, nb = ag["blind"], ag["nonblind"]
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            name, _acc(b["shape"]), _acc(b["near"]), _acc(b["near_stretch"]),
            _pct(b["engine_right"], b["ref_ha"]), _acc(nb["shape"]), _acc(nb["near"])))
    lines.append("")
    lines.append("#### 2b. Agreement by |d| bucket, pooled: right / vote (share)")
    lines.append("")
    lines.append("| |d| m/s | blind: shape | blind: nearest (same frames' bucket) | non-blind: "
                 "shape | non-blind: nearest |")
    lines.append("|---|---|---|---|---|")
    for i in range(len(BUCKETS)):
        lines.append("| {} | {} | {} | {} | {} |".format(
            bucket_name(i), _acc(pooled["blind"]["shape_bucket"][i]),
            _acc(pooled["blind"]["near_bucket"][i]), _acc(pooled["nonblind"]["shape_bucket"][i]),
            _acc(pooled["nonblind"]["near_bucket"][i])))
    lines.append("")
    lines.append("#### 2c. Shape against nearest on blind frames (ref H/A), per match")
    lines.append("")
    lines.append("| match | both vote, agree: right | both vote, disagree | disagree: shape "
                 "right | disagree: nearest right | shape only: right | nearest only: right | "
                 "no vote |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for p in group + [None]:
        cb = (pooled if p is None else p["agreement"])["blind"]["combined"]
        name = "pooled" if p is None else short(p["game"])
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            name, _pct(cb["agree_right"], cb["agree"]), cb["disagree"],
            _pct(cb["disagree_shape"], cb["disagree"]),
            _pct(cb["disagree_near"], cb["disagree"]),
            _pct(cb["shape_only_right"], cb["shape_only"]),
            _pct(cb["near_only_right"], cb["near_only"]), cb["none"]))
    lines.append("")
    lines.append("#### 2d. Blind frames by cause, pooled: right / vote (share)")
    lines.append("")
    lines.append("| cause | frames | engine right | shape | shape, |d| >= 1.5 | nearest |")
    lines.append("|---|---|---|---|---|---|")
    for c in CAUSES:
        ag = pooled["cause_" + c]
        strong = _sum(ag["shape_bucket"][3:])
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            c, ag["frames"], _pct(ag["engine_right"], ag["ref_ha"]), _acc(ag["shape"]),
            _acc(strong), _acc(ag["near"])))
    lines.append("")
    lines.append("#### 2e. Blind stretches by |d_bar|, pooled: right / stretches with a majority "
                 "reference H or A")
    lines.append("")
    lines.append("| |d_bar| m/s | stretches (frames) | shape team | nearest stretch (right / "
                 "vote) | engine majority |")
    lines.append("|---|---|---|---|---|")
    st = pooled["stretch"]
    for i in range(len(BUCKETS) + 1):
        node = st[i]
        name = "no d_bar" if i == len(BUCKETS) else bucket_name(i)
        lines.append("| {} | {} ({}) | {} | {} | {} |".format(
            name, node["n"], node["frames"], _pct(node["shape"], node["n"]),
            _pct(node["near"], node["near_vote"]), _pct(node["engine"], node["n"])))
    lines.append("")
    return pooled


def report_disagree(group, lines):
    """Measurement 3: shape and nearest disagree with |d| >= 1.45."""
    lines.append("| match | frames | reference backs shape | reference backs nearest |")
    lines.append("|---|---|---|---|")
    tot = [0, 0, 0]
    for p in group:
        cb = p["agreement"]["blind"]["combined"]
        tot = [tot[0] + cb["strong_disagree"], tot[1] + cb["strong_shape"],
               tot[2] + cb["strong_near"]]
        lines.append("| {} | {} | {} | {} |".format(
            short(p["game"]), cb["strong_disagree"], _pct(cb["strong_shape"],
                                                          cb["strong_disagree"]),
            _pct(cb["strong_near"], cb["strong_disagree"])))
    lines.append("| pooled | {} | {} | {} |".format(tot[0], _pct(tot[1], tot[0]),
                                                     _pct(tot[2], tot[0])))
    lines.append("")


def report_splice(group, lines):
    """Measurement 5: the splice per match and threshold."""
    lines.append("| T m/s | match | stretches flipped | frames changed | of which ref backs "
                 "shape / engine | engine spellF1 | spliced minus engine | paired 5-95 % |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for t in SPLICE_T:
        deltas = []
        for p in group:
            row = next(r for r in p["splices"] if r["T"] == t)
            pr = row["paired"]
            deltas.append(pr["delta"])
            lines.append("| {} | {} | {} | {} | {} / {} | {:.4f} | {:+.4f} | [{:+.4f}, {:+.4f}] "
                         "|".format(t, short(p["game"]), row["stretches"],
                                    row["frames_changed"], row["frames_ref_backs"],
                                    row["frames_ref_engine"], row["base"], pr["delta"],
                                    pr["low"], pr["high"]))
        lines.append("| {} | mean of {} | | | | | {:+.4f} | {} of {} above zero |".format(
            t, len(group), float(np.mean(deltas)), sum(1 for x in deltas if x > 0),
            len(group)))
    lines.append("")


def report_barath(p, lines):
    """Measurement 4: BAR-ATH blind stretches, label-free, and the public window list."""
    c = p["counts"]
    alive_min = c["alive"] / p["fps"] / 60.0
    lines.append("| quantity | value |")
    lines.append("|---|---|")
    lines.append("| engine md5 | `{}` |".format(p["engine_md5"]))
    lines.append("| alive graded frames | {} ({:.1f} min) |".format(c["alive"], alive_min))
    lines.append("| blind frames / stretches | {} / {} |".format(c["blind"], c["stretches"]))
    lines.append("| engine-unknown frames / stretches holding one | {} / {} |".format(
        c["engine_unknown"], c["unknown_stretches"]))
    for k in CAUSES:
        lines.append("| cause {}: stretches / frames | {} / {} |".format(
            k, c["by_cause"][k]["stretches"], c["by_cause"][k]["frames"]))
    q = c["lengths"]
    if q["n"]:
        lines.append("| stretch length p10 / p50 / p90 | {:.0f} / {:.0f} / {:.0f} |".format(
            q["p10"], q["p50"], q["p90"]))
    lines.append("| blind frames with d defined | {} |".format(c["d_defined_blind"]))
    lines.append("| stretches with |d| >= {} on at least half their frames (frames) | {} ({}) "
                 "|".format(HALF_SHARE_T, c["half_strong"], c["half_strong_frames"]))
    lines.append("| of which hold engine-unknown frames | {} |".format(c["unknown_half_strong"]))
    for row in p["splices"]:
        lines.append("| splice T {}: stretches / frames that would change | {} / {} |".format(
            row["T"], row["stretches"], row["frames_changed"]))
    lines.append("")
    rows = p.get("stretches") or []
    unk = [r for r in rows if r["engine_unknown"] > 0]
    if unk:
        lines.append("Engine-unknown stretches (label-free): the shape sign against the "
                     "engine label.")
        lines.append("")
        agree = sum(1 for r in unk if r["d_bar"] is not None and r["engine"] in TEAMS
                    and ("H" if r["d_bar"] > 0 else "A") == r["engine"])
        defined = sum(1 for r in unk if r["d_bar"] is not None and r["engine"] in TEAMS)
        near_agree = sum(1 for r in unk if r["near_stretch"] and r["near_stretch"] == r["engine"])
        lines.append("| stretches | frames | d_bar defined | shape team = engine label | "
                     "nearest stretch = engine label |")
        lines.append("|---|---|---|---|---|")
        lines.append("| {} | {} | {} | {} | {} |".format(
            len(unk), sum(r["engine_unknown"] for r in unk), defined, _pct(agree, defined),
            _pct(near_agree, len(unk))))
        lines.append("")
    if p.get("public") is not None:
        lines.append("Public window {}..{}: every blind stretch inside it.".format(*PUBLIC_WINDOW))
        lines.append("")
        lines.append("| frames | cause | engine unknown frames | engine label | public label "
                     "(share) | nearest stretch | nearest frames H / A | d_bar m/s | shape sign "
                     "| share |d| >= 1.45 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in p["public"]:
            sign = "-" if r["d_bar"] is None else ("H" if r["d_bar"] > 0 else "A")
            lines.append("| {}..{} | {} | {} | {} | {} ({:.0f} %) | {} | {} / {} | {} | {} | "
                         "{:.0f} % |".format(
                             r["start"], r["stop"] - 1, r["cause"], r["engine_unknown"],
                             r["engine"] or "-", r["public"], 100 * r["public_share"],
                             r["near_stretch"] or "-", r["near_frames_h"], r["near_frames_a"],
                             _fmt(r["d_bar"]), sign, 100 * r["strong_share"]))
        lines.append("")


def report(args):
    """The `report` command: the tables of calibration/P9B_shape_evidence.md."""
    import json

    payloads = []
    for path in args.files:
        with open(path) as handle:
            payloads.append(json.load(handle))
    lines = ["### 0. Matches and blind-frame counts", ""]
    report_counts(payloads, lines)
    views = []
    for p in payloads:
        if p["has_ref"] and p["view"] not in views:
            views.append(p["view"])
    for view in views:
        group = [p for p in payloads if p["view"] == view and p["has_ref"]]
        lines += ["### View: {} ({} matches)".format(view, len(group)), ""]
        lines += ["#### 1. The reference label on blind frames", ""]
        report_reference_labels(group, lines)
        report_agreement(group, lines)
        lines += ["#### 3. Shape and nearest disagree, |d| >= {}: which one the reference "
                  "backs".format(DISAGREE_T), ""]
        report_disagree(group, lines)
        lines += ["#### 5. Splice (not an engine run): blind stretches flipped to the shape "
                  "team, spliced minus engine, spellF1", ""]
        report_splice(group, lines)
    for p in payloads:
        if p["game"] == "BAR-ATH":
            lines += ["### 4. BAR-ATH (label-free; the public file only inside the window)", ""]
            report_barath(p, lines)
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text)
    sys.stdout.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd")
    c = sub.add_parser("collect")
    c.add_argument("--game", required=True)
    c.add_argument("--view", required=True)
    c.add_argument("--ref")
    c.add_argument("--public-gt")
    c.add_argument("--cache-dir")
    c.add_argument("--config")
    c.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("files", nargs="+")
    r.add_argument("--out")
    args = parser.parse_args(argv)
    if args.cmd == "collect":
        collect(args)
    elif args.cmd == "report":
        report(args)
    else:
        parser.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
