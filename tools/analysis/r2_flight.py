"""Research R2 direction e: flight assignment re-measured with dribble events (C5 HANDOFF 6).

    python tools/analysis/r2_flight.py --out-dir DIR [--cache DIR/inputs.pkl] [--write-csv]
        [--cap-dist] [--variants off,on,cap91+geom+norise+peak,...]

--cap-dist prints the full-match distribution of release-to-next-event gaps (next event within
R_TOUCH), by the team of the next event. --write-csv writes
evaluation/results/rx_e_flight_<variant>.csv for each variant ("+" becomes "_").

Runs the engine with FLIGHT_ASSIGNMENT on, plus physics variants of the flight walk, through
engine.possession.run(cfg, return_parts=True, inputs=...) with the flight functions
monkeypatched. Nothing under engine/ is edited. For each variant it prints and writes to JSON:
the full-match flight numbers (owner releases, flights, flight length distribution, walk
outcomes), the full-match sanity numbers, the public-window score, the latency of every GT
transition, the matched set against flight off, and a leave-one-transition-out verdict.

Variants (all under POSSESSION_RULE release_or_control):
  on          the engine's flight_assignment, unchanged. The walk stops at the first row of the
              rule kinds or contested. A flight needs that row to be an owner event of the
              other team. A stoppage cancels the flight.
  cap<N>      `on` plus: the next event must come within N frames of the release.
  conf        the walk skips non-deflection rows below OWNER_MIN_CONFIDENCE. A low-confidence
              touch is not evidence that anyone played the ball. A deflection of any confidence
              ends the walk without a flight. A confident owner row of the other team fires.
  confall     the walk skips every row below OWNER_MIN_CONFIDENCE, deflections included (only
              confident rows and contested rows end the walk).
  defl        `on` plus: a deflection row of the other team (any confidence) also fires (the
              kick reached the other team, who touched it).
  geom        `on` plus: the straight line from the ball at the release to the ball at the
              reception is within DEFLECTION_MAX_HEADING_DEG of the release velocity direction.
  norise      `on` plus: no kick rise (speed gain above DV_RELEASE within RELEASE_K frames, the
              engine's kick test) between the end of the release's own monotone rise and the
              reception (no intervening kick).
  anyu        `on` plus: any team row of the other team (touch, deflection, any confidence)
              fires: the first player the kick reached was of the other team.
  peak        a flight's switch moves from the release contact to the last frame of the
              release's monotone speed rise (rise_peak: the kick stops accelerating the ball).
  shift       a flight's switch moves from the release contact to contact + RELEASE_K (the
              engine's kick rise length). The kicker's own row stays at the contact.
  shift<N>    as shift, with N frames (diagnostic only).
Combinations join names with "+".

Rules: the public GT is read through tools/gt.py only. The engine never sees it.
"""

import argparse
import dataclasses
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import _common as cm  # noqa: E402
from _common import pm  # noqa: E402
import calibrate  # noqa: E402
import forensics  # noqa: E402
import sanity  # noqa: E402
from gt import public_gt_codes  # noqa: E402

from engine import data, possession, submission  # noqa: E402
from engine.config import Config  # noqa: E402

PUBLIC = (11185, 18685)
CASES = (13704, 14846, 15365, 15346)
TEAM_KINDS = possession.TEAM_EVENT_KINDS

_ORIG_FLIGHT = possession.flight_assignment
_ORIG_APPLY = possession.apply_flights
STATE = {"records": None, "switch": {}}


# -- the generalised flight walk ----------------------------------------------------------------

def _is_deflection(notes):
    return "deflection" in str(notes).split(";")


def release_direction(ball, contact, cfg):
    """Unit vector of the mean ball velocity over [contact + 3, contact + RELEASE_K], or None."""
    lo, hi = contact + 3, contact + int(cfg.RELEASE_K) + 1
    vx = np.asarray(ball["vx"][lo:hi], dtype=np.float64)
    vy = np.asarray(ball["vy"][lo:hi], dtype=np.float64)
    ok = np.isfinite(vx) & np.isfinite(vy)
    if not ok.any():
        return None
    v = np.array([vx[ok].mean(), vy[ok].mean()])
    n = np.hypot(*v)
    return None if n < 1e-6 else v / n


def ball_at(ball, frame, back=True, reach=25):
    """(x, y) of the visible ball at `frame`, else the nearest visible frame within reach."""
    x, y = np.asarray(ball["x"]), np.asarray(ball["y"])
    for d in range(reach + 1):
        for f in ((frame - d, frame + d) if back else (frame + d, frame - d)):
            if 0 <= f < x.shape[0] and np.isfinite(x[f]) and np.isfinite(y[f]):
                return np.array([x[f], y[f]])
    return None


def path_angle(ball, contact, end, cfg):
    """Degrees between the release direction and the line ball(contact) -> ball(end)."""
    u = release_direction(ball, contact, cfg)
    p0, p1 = ball_at(ball, contact), ball_at(ball, end)
    if u is None or p0 is None or p1 is None:
        return None
    d = p1 - p0
    n = np.hypot(*d)
    if n < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(u, d / n), -1.0, 1.0))))


def rise_peak(ball, contact):
    """Last frame of the monotone speed rise that starts at the release contact (same run)."""
    speed = np.asarray(ball["speed"], dtype=np.float64)
    run_id = np.asarray(ball["run_id"], dtype=np.int64)
    end = contact
    while (end + 1 < speed.shape[0] and run_id[end + 1] == run_id[contact]
           and np.isfinite(speed[end + 1]) and speed[end + 1] >= speed[end]):
        end += 1
    return end


def max_rise(ball, contact, end, cfg):
    """Largest kick rise speed[t + RELEASE_K] - speed[t] for t from the end of the release's
    own monotone rise (rise_peak + 1) to end - RELEASE_K, inside one visible run, m/s.

    0 if the window is empty. A rise above DV_RELEASE is the engine's definition of a kick
    (events.py), so a rise inside the window means something kicked the ball between the
    release and the reception. On this smooth ATD track the release's own rise lasts a median
    23 frames (r2 measurement), so the window cannot start at contact + RELEASE_K.
    """
    speed = np.asarray(ball["speed"], dtype=np.float64)
    run_id = np.asarray(ball["run_id"], dtype=np.int64)
    k = int(cfg.RELEASE_K)
    lo, hi = rise_peak(ball, contact) + 1, end - k
    if hi < lo:
        return 0.0
    t = np.arange(lo, hi + 1)
    t = t[(t + k) < speed.shape[0]]
    same = run_id[t] == run_id[t + k]
    rise = np.where(same, speed[t + k] - speed[t], np.nan)
    rise = rise[np.isfinite(rise)]
    return float(rise.max()) if rise.size else 0.0


def walk(events, owner, stoppage, blocks, kinds, ball, cfg, opts):
    """(flights, blocked, records). See the module docstring for the options.

    records: one dict per owner release on a graded frame, with the event that ends the walk and
    the outcome: flight, same_team, contested, other_not_owner, stoppage, block_end, none,
    capped, geometry, rise.
    """
    flights, blocked, records = [], [], []
    frames = events["frame"].to_numpy(dtype=np.int64)
    kind = events["kind"].to_numpy(dtype=object)
    team = events["team"].to_numpy(dtype=object)
    conf = events["confidence"].to_numpy(dtype=np.float64)
    dist = events["dist_m"].to_numpy(dtype=np.float64)
    defl = np.array([_is_deflection(n) for n in events["notes"]], dtype=bool)
    owner = np.asarray(owner, dtype=bool)
    stoppage = np.asarray(stoppage, dtype=bool)
    stream = np.flatnonzero(np.isin(kind, list(kinds) + ["contested"]))
    dead = np.concatenate(([0], np.cumsum(stoppage.astype(np.int64))))
    for idx in np.flatnonzero(owner & (kind == "release")):
        start, kicker = int(frames[idx]), team[idx]
        block = next(((f, l) for _, f, l in blocks if f <= start <= l), None)
        if block is None or kicker not in possession.TEAM_INDEX:
            continue
        rec = {"release": start, "kicker": kicker, "outcome": "none"}
        records.append(rec)
        later = stream[frames[stream] > start]
        j = None
        for cand in later:
            cand = int(cand)
            if opts.get("conf") and not defl[cand] and kind[cand] != "contested" \
                    and conf[cand] < cfg.OWNER_MIN_CONFIDENCE:
                continue
            if opts.get("confall") and kind[cand] != "contested" \
                    and conf[cand] < cfg.OWNER_MIN_CONFIDENCE:
                continue
            j = cand
            break
        if j is None:
            continue
        frame = int(frames[j])
        rec.update({"next_frame": frame, "next_kind": kind[j], "next_team": team[j],
                    "next_conf": float(conf[j]), "next_defl": bool(defl[j]),
                    "next_owner": bool(owner[j]), "next_dist": float(dist[j]),
                    "gap": frame - start})
        if frame > block[1]:
            rec["outcome"] = "block_end"
            continue
        if dead[frame + 1] - dead[start] > 0:
            rec["outcome"] = "stoppage"
            blocked.append((start, kicker, start + int(np.argmax(stoppage[start:frame + 1]))))
            continue
        other = team[j] in possession.TEAM_INDEX and team[j] != kicker
        if kind[j] == "contested" or team[j] not in possession.TEAM_INDEX:
            rec["outcome"] = "contested"
            continue
        if not other:
            rec["outcome"] = "same_team"
            continue
        fires = bool(owner[j]) or (opts.get("defl") and bool(defl[j])) or bool(opts.get("anyu"))
        if not fires:
            rec["outcome"] = "other_not_owner"
            continue
        rec["angle"] = path_angle(ball, start, frame, cfg)
        rec["rise"] = max_rise(ball, start, frame, cfg)
        cap = opts.get("cap")
        if cap is not None and frame - start > cap:
            rec["outcome"] = "capped"
            continue
        if opts.get("geom") and (rec["angle"] is None
                                 or rec["angle"] > cfg.DEFLECTION_MAX_HEADING_DEG):
            rec["outcome"] = "geometry"
            continue
        if opts.get("norise") and rec["rise"] > cfg.DV_RELEASE:
            rec["outcome"] = "rise"
            continue
        rec["outcome"] = "flight"
        flights.append((start, kicker, team[j], frame))
    return flights, blocked, records


def all_records(inputs, cfg, stoppage):
    """Walk records of the plain `on` walk, but with every outcome kept for measurement."""
    owner = possession.owner_mask(inputs["events"], cfg)
    return walk(inputs["events"], owner, stoppage, inputs["blocks"],
                possession.RULE_TEAM_KINDS[cfg.POSSESSION_RULE], inputs["ball"], cfg, {})[2]


def install(opts, inputs, cfg):
    """Monkeypatch possession.flight_assignment / apply_flights for one variant."""
    ball = inputs["ball"]

    def flight_assignment(events, owner, stoppage, blocks, kinds=TEAM_KINDS):
        flights, blocked, records = walk(events, owner, stoppage, blocks, kinds, ball, cfg, opts)
        STATE["records"] = records
        return flights, blocked

    shift = opts.get("shift", 0)

    def switch_frame(f, end):
        if shift == "peak":
            return min(rise_peak(ball, int(f)), int(end))
        return min(int(f) + int(shift), int(end))

    def apply_flights(rows, flights):
        if shift == 0:
            return _ORIG_APPLY(rows, flights)
        out = list((int(f), t) for f, t in rows)
        present = set(out)
        for f, t, u, end in flights:
            if (int(f), t) in present:           # the kicker keeps his row at the contact
                out.append((switch_frame(f, end), u))
        return sorted(out)

    possession.flight_assignment = flight_assignment
    possession.apply_flights = apply_flights


def uninstall():
    possession.flight_assignment = _ORIG_FLIGHT
    possession.apply_flights = _ORIG_APPLY
    STATE["records"] = None


def parse(name, cfg):
    """Options dict of a variant name such as 'cap50+geom'."""
    opts = {}
    if name in ("off", "on"):
        return opts
    for part in name.split("+"):
        if part == "on":
            continue
        if part.startswith("cap"):
            opts["cap"] = int(part[3:])
        elif part in ("conf", "confall", "defl", "anyu", "geom", "norise"):
            opts[part] = True
        elif part == "shift":
            opts["shift"] = int(cfg.RELEASE_K)
        elif part.startswith("shift") and part[5:].isdigit():
            opts["shift"] = int(part[5:])
        elif part == "peak":
            opts["shift"] = "peak"
        else:
            raise ValueError(part)
    return opts


# -- context and scoring ------------------------------------------------------------------------

class Ctx(calibrate.Context):
    """calibrate.Context on the CURRENT default config (not the P2 start point)."""

    def __init__(self, inputs):
        self.base = Config.default()
        self.inputs = inputs
        self.gt = public_gt_codes()
        self.runs = pm.valid_runs(self.gt)
        self.gt_trans = pm.transitions(self.gt, self.runs)
        tactical = data.load_tactical()
        self.ball_missing = np.isnan(tactical["bx"])
        self.kind = {g[0]: forensics.kind_of(g[1], g[2], self.ball_missing, g[0],
                                             forensics.REAPPEAR_LOOKBACK)
                     for g in self.gt_trans}
        self.tags = data.load_tagging()
        self.blocks = data.graded_blocks()
        self.graded_runs = cm.runs_from_blocks(self.blocks)
        margin = int(round(sanity.SET_PIECE_MARGIN_S * cm.FPS))
        pieces = self.tags[self.tags["code"].str.upper() == sanity.SET_PIECE_CODE]
        self.setpiece_windows = [(int(r.start_frame) - margin, int(r.end_frame) + margin)
                                 for r in pieces.itertuples(index=False)
                                 if any(a < int(r.end_frame) + margin and int(r.start_frame)
                                        - margin < b for a, b in self.graded_runs)]


def load_inputs(cache):
    if cache and os.path.exists(cache):
        with open(cache, "rb") as handle:
            inputs = pickle.load(handle)
        if inputs["key"] == possession.detection_key(Config.default()):
            return inputs
    inputs = possession.prepare(Config.default())
    if cache:
        with open(cache, "wb") as handle:
            pickle.dump(inputs, handle)
    return inputs


def pct(values, q):
    return None if len(values) == 0 else float(np.percentile(values, q))


def run_variant(ctx, name):
    cfg = dataclasses.replace(ctx.base, FLIGHT_ASSIGNMENT=(name != "off"))
    opts = parse(name, cfg)
    install(opts, ctx.inputs, cfg)
    try:
        labels, parts = possession.run(cfg, return_parts=True, inputs=ctx.inputs)
        records = STATE["records"] or []
    finally:
        uninstall()
    point = calibrate.score(ctx, labels, parts)
    codes = pm.label_codes(np.asarray(labels))
    rep = pm.evaluate(ctx.gt, codes, taus=(12, 25))
    point["per_iou"] = {str(k): v["n_matched"] for k, v in rep["spell_f1"].items()}
    point["spurious_12"] = rep["transitions"][12].get("n_spurious")
    point["missed_12"] = rep["transitions"][12].get("n_missed")
    summ = submission.summarize(labels, blocks=ctx.blocks)
    point["d_share_half"] = {int(h): r["d_share"] for h, r in summ["halves"].items()}
    graded = np.zeros(ctx.inputs["n_frames"], dtype=bool)
    for _, f, l in ctx.blocks:
        graded[f:l + 1] = True
    flights = [f for f in parts["flights"] if graded[f[0]]]
    lengths = np.array([e - s for s, _, _, e in flights], dtype=np.float64)
    outcomes = {}
    for r in records:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    point["flight"] = {"owner_releases": len(records), "flights": len(flights),
                       "len_p10": pct(lengths, 10), "len_p50": pct(lengths, 50),
                       "len_p90": pct(lengths, 90),
                       "len_max": None if not lengths.size else float(lengths.max()),
                       "outcomes": outcomes,
                       "public": [list(map(lambda v: v if isinstance(v, str) else int(v), f))
                                  for f in flights if PUBLIC[0] <= f[0] < PUBLIC[1]]}
    point["records_public"] = [r for r in records if PUBLIC[0] - 200 <= r["release"] < PUBLIC[1]]
    return labels, point


def loo(ctx, on, off):
    """Leave-one-transition-out of 'on beats off': strictly more matched, none lost."""
    rows = []
    for g in ctx.gt_trans:
        ex = (g[0],)
        lost = calibrate.lost(off, on, ex)
        n_on, n_off = len(calibrate.matched_set(on, ex)), len(calibrate.matched_set(off, ex))
        rows.append({"left_out": g[0], "on": n_on, "off": n_off, "lost": lost,
                     "on_wins": (not lost) and n_on > n_off})
    full_lost = calibrate.lost(off, on)
    full_win = (not full_lost) and len(calibrate.matched_set(on)) > len(calibrate.matched_set(off))
    return {"full_win": full_win, "full_lost": full_lost,
            "gained": sorted(calibrate.matched_set(on) - calibrate.matched_set(off)),
            "n_loo_win": sum(r["on_wins"] for r in rows), "rows": rows,
            "survives": full_win and all(r["on_wins"] for r in rows)}


def cap_distribution(ctx):
    """Release-to-next-event gaps on the full match (graded), next event within R_TOUCH."""
    cfg = ctx.base
    stoppage = possession.run(cfg, return_parts=True, inputs=ctx.inputs)[1]["stoppage"]
    recs = all_records(ctx.inputs, cfg, stoppage)
    out = {}
    for label, test in (
            ("other_team_any", lambda r: r.get("next_team") in ("H", "A")
             and r["next_team"] != r["kicker"]),
            ("other_team_owner", lambda r: r.get("next_team") in ("H", "A")
             and r["next_team"] != r["kicker"] and r["next_owner"]),
            ("same_team_any", lambda r: r.get("next_team") == r["kicker"]),
            ("same_team_owner", lambda r: r.get("next_team") == r["kicker"] and r["next_owner"]),
            ("same_team_touchlike", lambda r: r.get("next_team") == r["kicker"]
             and r["next_kind"] in ("touch", "release"))):
        gaps = np.array([r["gap"] for r in recs if "gap" in r and r["outcome"] != "stoppage"
                         and r["next_dist"] <= cfg.R_TOUCH and test(r)], dtype=np.float64)
        out[label] = {"n": int(gaps.size), **{"p{}".format(q): pct(gaps, q)
                                              for q in (10, 25, 50, 75, 90, 95)},
                      "max": None if not gaps.size else float(gaps.max()),
                      "share_le_50": None if not gaps.size else float(np.mean(gaps <= 50)),
                      "share_le_75": None if not gaps.size else float(np.mean(gaps <= 75))}
    return out, recs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--variants", default="off,on")
    parser.add_argument("--write-csv", action="store_true",
                        help="write evaluation/results/rx_e_<variant>.csv for each variant")
    parser.add_argument("--cap-dist", action="store_true")
    args = parser.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    ctx = Ctx(load_inputs(args.cache))
    result = {}
    if args.cap_dist:
        dist, recs = cap_distribution(ctx)
        result["cap_distribution"] = dist
        with open(os.path.join(args.out_dir, "release_records.json"), "w") as h:
            json.dump(recs, h, indent=0, default=calibrate._jsonable)
        print(json.dumps(dist, indent=1))
    points = {}
    for name in args.variants.split(","):
        labels, point = run_variant(ctx, name)
        points[name] = point
        if args.write_csv and name != "off":
            tag = name.replace("+", "_")
            path = os.path.join(ROOT, "evaluation", "results", "rx_e_flight_{}.csv".format(tag))
            submission.write_submission(labels, path)
            point["csv"] = path
            point["md5"] = submission.md5_of(path)
    for name, point in points.items():
        if name != "off" and "off" in points:
            point["loo"] = loo(ctx, point, points["off"])
    for name, point in points.items():
        point["per"] = {str(k): v for k, v in point["per"].items()}
    result["points"] = points
    with open(os.path.join(args.out_dir, "r2_flight.json"), "w") as h:
        json.dump(result, h, indent=1, default=calibrate._jsonable)
    print_table(ctx, points)
    return 0


def print_table(ctx, points):
    head = ("{:<22} {:>5} {:>8} {:>3} {:>4} {:>5} {:>5} {:>5} {:>4} {:>6} {:>6} {:>6} {:>3} "
            "{:>5} {:<10} {}").format(
        "variant", "spF1", "m/mi/sp", "m12", "fl", "p50", "p90", "max", "pos", "D", "D1", "D2",
        "Dsg", "tag", "LOO", "gained/lost")
    print(head)
    for name, p in points.items():
        fl = p["flight"]
        lo = p.get("loo")
        print("{:<22} {:.3f} {:>2}/{:>2}/{:>2} {:>3} {:>4} {:>5} {:>5} {:>5} {:>4} {:>6.1%} {:>6.1%} "
              "{:>6.1%} {:>3} {:>5.1%} {:<10} {}".format(
                  name, p["spell_f1"], p["matched"], p["missed"], p["spurious"], p["matched_12"],
                  fl["flights"], _n(fl["len_p50"]), _n(fl["len_p90"]), _n(fl["len_max"]),
                  p["possessions"], p["d_share"], p["d_share_half"][1], p["d_share_half"][2],
                  p["d_segments"], p["tag_agreement"] or 0.0,
                  "-" if lo is None else "{}{}/20".format("S " if lo["survives"] else "",
                                                          lo["n_loo_win"]),
                  "-" if lo is None else "+{} -{}".format(lo["gained"], lo["full_lost"])))
    print()
    print("latency per GT transition (m matched at 25; . nearest same destination)")
    print("{:<22} ".format("") + " ".join("{:>6}".format(g[0]) for g in ctx.gt_trans))
    for name, p in points.items():
        cells = []
        for g in ctx.gt_trans:
            rec = p["per"][str(g[0])] if str(g[0]) in p["per"] else p["per"][g[0]]
            lat = rec["latency"]
            cells.append("{:>6}".format(("-" if lat is None else "{:+d}".format(lat))
                                        + ("m" if rec["status"] == "matched" else ".")))
        print("{:<22} ".format(name[:22]) + " ".join(cells))


def _n(v):
    return "-" if v is None else "{:.0f}".format(v)


if __name__ == "__main__":
    sys.exit(main())
