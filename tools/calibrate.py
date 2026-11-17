"""GT-semantics calibration on the public window (card P2, PLAN.md 2 item 3, 5.1).

    python tools/calibrate.py [--json PATH]

The tool runs the engine on a small grid of semantic settings and prints one table row for each
grid point. The settings are POSSESSION_RULE, FLIGHT_ASSIGNMENT and the four DELTA_* offsets.
The tool then prints a leave-one-transition-out table, the flight measurement on the full match,
and the release rise lengths.

Rules (team rules, PLAN 5.1):
  * The public score never promotes a version. The table prints it, and no choice reads it.
  * The tool calls the engine only through engine.possession.run(cfg) with a
    dataclasses.replace() of the default Config. It never edits a file.
  * The tool calls engine.possession.prepare() once. Every grid point reuses that result, so a
    grid point costs only the decoder.
  * The tool reads the public GT through tools/gt.py. The engine never sees it.

Procedure (P2 card):
  1. Stage A reads the latency of each transition kind at the start point (START, the M3
     values).
  2. Stage B runs the rule and flight switches with all offsets at 0. A switch is kept only if
     it loses no transition that the start point matches, passes the vetoes, and matches
     strictly more transitions at tol 25. Ties keep the incumbent (rule v1, flight off). The
     tool then repeats stage B with each public transition left out. If the choices disagree,
     the incumbent stays (PLAN 5.1: the simpler rule).
  3. Stage C runs one offset kind at a time on that rule and flight. The grid is the card grid
     plus the value that cancels the median latency of the kind. The choice is the grid value
     nearest to minus the median matched latency at offset 0. Ties go to the smaller |delta|.
     A choice that loses a matched transition, fails a veto, or (for a restart offset) raises
     the count of set-piece tags with no D within 10 s falls back to 0.
  4. Leave-one-out: stage C again with each public transition left out. If the choices
     disagree, the final value of the kind is the choice with the smallest |delta|.
  5. Stage D runs the final choice. The tool then measures flight assignment on the full match
     and the rise length of every owner release.
"""

import argparse
import dataclasses
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as cm  # noqa: E402
from _common import A, D, H, LETTER, pm  # noqa: E402
import forensics  # noqa: E402
import sanity  # noqa: E402
from gt import public_gt_codes  # noqa: E402  (the only permitted GT reader)

from engine import data, possession, submission  # noqa: E402
from engine.config import Config  # noqa: E402

TOL = 25                           # PLAN 2 item 3: the latency at tolerance 25 frames
KINDS = forensics.KINDS            # H<->A, alive->D, D->alive (kick), D->alive (reappear)
DELTA_OF_KIND = {KINDS[0]: "DELTA_SWITCH", KINDS[1]: "DELTA_DEAD",
                 KINDS[2]: "DELTA_RESTART_KICK", KINDS[3]: "DELTA_RESTART_REAPPEAR"}
# The grid of the brief for each offset kind. Stage C adds the cancel value of each kind.
CARD_GRID = {"DELTA_SWITCH": (0, 3, 6), "DELTA_DEAD": (-5, 0, 5),
             "DELTA_RESTART_KICK": (0, 5), "DELTA_RESTART_REAPPEAR": (0, 10)}
RESTART_DELTAS = ("DELTA_RESTART_KICK", "DELTA_RESTART_REAPPEAR")
# Stage B candidates in the tie order: the incumbent first (rule v1 of PLAN 3.7, flight off).
RULE_FLIGHT = (("release_or_control", False), ("release", False),
               ("release_or_control", True), ("release", True))
# The start point of the grid: rule v1 of M3 (PLAN 3.7), flight off, every offset 0. The grid
# starts here whatever the current defaults in engine/config.py are.
START = {"POSSESSION_RULE": "release_or_control", "FLIGHT_ASSIGNMENT": False,
         "DELTA_SWITCH": 0, "DELTA_DEAD": 0, "DELTA_RESTART_KICK": 0,
         "DELTA_RESTART_REAPPEAR": 0}
# The two short public possessions (PLAN 1.3): matched at IoU >= SHORT_IOU is the target.
SHORT_SPELLS = ((15346, 15365), (15766, 15799))
SHORT_IOU = 0.5


def _round(value):
    """Nearest integer, halves away from zero (no banker's rounding)."""
    return int(np.sign(value) * np.floor(abs(value) + 0.5))


class Context(object):
    """The data that every grid point shares: the prepare() result, the GT and the tags."""

    def __init__(self, base=None):
        self.base = dataclasses.replace(base or Config.default(), **START)
        self.inputs = possession.prepare(self.base)
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
        margin = int(round(sanity.SET_PIECE_MARGIN_S * cm.fps()))
        pieces = self.tags[self.tags["code"].str.upper() == sanity.SET_PIECE_CODE]
        self.setpiece_windows = [(int(r.start_frame) - margin, int(r.end_frame) + margin)
                                 for r in pieces.itertuples(index=False)
                                 if any(a < int(r.end_frame) + margin and int(r.start_frame)
                                        - margin < b for a, b in self.graded_runs)]

    def run(self, **changes):
        """(cfg, labels, parts) of engine.possession.run() for the default plus `changes`."""
        cfg = dataclasses.replace(self.base, **changes)
        labels, parts = possession.run(cfg, return_parts=True, inputs=self.inputs)
        return cfg, labels, parts


def short_spell_iou(gt, filled, runs, spell):
    """(best IoU with a same-team predicted spell, matched at SHORT_IOU by the scorer)."""
    gt_spells = pm.possession_spells(gt, runs)
    pred_spells = pm.possession_spells(filled, runs)
    index = [i for i, (a, b, _) in enumerate(gt_spells) if (a, b) == spell]
    if not index:
        return None, False
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    best = max([iou for iou, i, _ in overlaps if i == index[0]] or [0.0])
    flags = pm.matched_gt_flags(overlaps, len(gt_spells), SHORT_IOU)
    return float(best), bool(flags[index[0]])


def score(ctx, labels, parts=None):
    """Every number of one grid point, as a dict (public window and full match)."""
    codes = pm.label_codes(np.asarray(labels))
    report = pm.evaluate(ctx.gt, codes, taus=(12, TOL))
    filled = pm.fill_abstentions(codes, ctx.runs)
    pred_trans = pm.transitions(filled, ctx.runs)
    pairs, missed, spurious = pm.match_transitions(ctx.gt_trans, pred_trans, TOL)
    pair_of = {g[0]: p for g, p in pairs}
    per = {}
    for g in ctx.gt_trans:
        if g[0] in pair_of:
            per[g[0]] = {"status": "matched", "latency": pair_of[g[0]][0] - g[0]}
        else:
            near = forensics.nearest_same_destination(g, pred_trans)
            per[g[0]] = {"status": "missed", "latency": None if near is None else near[0] - g[0]}
    summary = submission.summarize(labels, blocks=ctx.blocks)
    stats = sanity.sanity_stats(labels, tags=ctx.tags, blocks=ctx.blocks, use_ball=False)
    short = [short_spell_iou(ctx.gt, filled, ctx.runs, s) for s in SHORT_SPELLS]
    restarts = np.array([f for f, a, b in pm.transitions(pm.label_codes(np.asarray(labels)),
                                                         ctx.graded_runs)
                         if a == D and b in (H, A)], dtype=np.int64)
    out = {"spell_f1": report["spell_f1_mean"], "macro_f1": report["macro_f1"],
           "n_pred_spells": report["n_pred_spells"],
           "matched": len(pairs), "missed": len(missed), "spurious": len(spurious),
           "matched_12": report["transitions"][12]["n_matched"],
           "spurious_frames": [int(p[0]) for p in spurious],
           "per": per, "possessions": summary["total"]["possessions"],
           "d_share": summary["total"]["d_share"], "d_segments": summary["total"]["d_segments"],
           "vetoes_ok": bool(summary["ok"]),
           "setpiece_without_d": stats["set_pieces"]["n_without_d"],
           "setpiece_with_restart": int(sum(
               1 for lo, hi in ctx.setpiece_windows
               if np.any((restarts >= lo) & (restarts < hi)))),
           "tag_agreement": stats["tags"]["total"]["agreement"],
           "short": short}
    if parts is not None:
        out["n_flights"] = len(parts.get("flights", []))
        out["n_flights_blocked"] = len(parts.get("flights_blocked", []))
    return out


def kind_latencies(ctx, point, exclude=()):
    """{kind: [matched latencies at TOL]} of one grid point, without the `exclude` frames."""
    out = {k: [] for k in KINDS}
    for frame, rec in sorted(point["per"].items()):
        if frame in exclude or rec["status"] != "matched":
            continue
        out[ctx.kind[frame]].append(rec["latency"])
    return out


def _stat(values, fn):
    return None if len(values) == 0 else float(fn(values))


def matched_set(point, exclude=()):
    """GT frames matched at TOL, without the `exclude` frames."""
    return set(f for f, r in point["per"].items() if r["status"] == "matched" and f not in exclude)


def lost(reference, point, exclude=()):
    """Sorted GT frames that `reference` matches and `point` does not."""
    return sorted(matched_set(reference, exclude) - matched_set(point, exclude))


def choose_rule_flight(ref, points, exclude=()):
    """The stage B choice: (rule, flight). See the module docstring, step 2."""
    best, best_n = RULE_FLIGHT[0], len(matched_set(ref, exclude))
    for key in RULE_FLIGHT[1:]:
        point = points[key]
        if lost(ref, point, exclude) or not point["vetoes_ok"]:
            continue
        n = len(matched_set(point, exclude))
        if n > best_n:
            best, best_n = key, n
    return best


def choose_delta(ctx, field, zero, grid_points, exclude=()):
    """The stage C choice for one offset field. See the module docstring, step 3."""
    kind = [k for k, f in DELTA_OF_KIND.items() if f == field][0]
    values = kind_latencies(ctx, zero, exclude)[kind]
    if not values:
        return 0
    target = -float(np.median(values))
    grid = sorted(grid_points, key=lambda d: (abs(d - target), abs(d), d))
    choice = grid[0]
    point = grid_points[choice]
    bad = lost(zero, point, exclude) or not point["vetoes_ok"]
    if field in RESTART_DELTAS and point["setpiece_without_d"] > zero["setpiece_without_d"]:
        bad = True
    return 0 if bad else int(choice)


def rise_lengths(ball, events, owner, blocks, cfg):
    """Release rise lengths on the graded frames: {"monotone": [...], "to_peak": [...]}.

    monotone  frames from the contact to the last frame of the speed rise. The rise goes on
              while the one-frame speed change is at least DV_TOUCH / DV_K inside one visible
              run (the walk-back rule of events.contact_frame, forward).
    to_peak   frames from the contact to the highest speed within RELEASE_K frames.
    Only release rows that set the owner count.
    """
    speed = np.asarray(ball["speed"], dtype=np.float64)
    run_id = np.asarray(ball["run_id"], dtype=np.int64)
    least = cfg.DV_TOUCH / float(cfg.DV_K)
    graded = np.zeros(speed.shape[0], dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    mono, peak = {}, {}
    rows = events[np.asarray(owner, dtype=bool) & (events["kind"] == "release").to_numpy()]
    for frame in rows["frame"].to_numpy(dtype=np.int64):
        c = int(frame)
        if not graded[c]:
            continue
        end = c
        while (end + 1 < speed.shape[0] and run_id[end + 1] == run_id[c]
               and np.isfinite(speed[end + 1]) and speed[end + 1] - speed[end] >= least):
            end += 1
        mono[c] = end - c
        hi = min(speed.shape[0], c + int(cfg.RELEASE_K) + 1)
        window = np.where(run_id[c:hi] == run_id[c], speed[c:hi], -np.inf)
        peak[c] = int(np.argmax(window))
    return mono, peak


def flight_measurement(ctx, rule):
    """Full-match flight numbers with FLIGHT_ASSIGNMENT on, against the same rule with it off."""
    cfg_off, labels_off, parts_off = ctx.run(POSSESSION_RULE=rule, FLIGHT_ASSIGNMENT=False)
    cfg_on, labels_on, parts_on = ctx.run(POSSESSION_RULE=rule, FLIGHT_ASSIGNMENT=True)
    table = parts_on["events"]
    owner = possession.owner_mask(table, cfg_on)
    graded = np.zeros(ctx.inputs["n_frames"], dtype=bool)
    for _, first, last in ctx.blocks:
        graded[first:last + 1] = True
    releases = table[owner & (table["kind"] == "release").to_numpy()]
    n_releases = int(graded[releases["frame"].to_numpy(dtype=np.int64)].sum())
    flights = [f for f in parts_on["flights"] if graded[f[0]]]
    lengths = np.array([e - s for s, _, _, e in flights], dtype=np.float64)
    onset_frames = parts_on["onsets"]["frame"].to_numpy(dtype=np.int64)
    through_onset = sum(1 for s, _, _, e in flights
                        if np.any((onset_frames > s) & (onset_frames <= e)))
    stoppage = parts_on["stoppage"]
    through_d = sum(1 for s, _, _, e in flights if stoppage[s:e + 1].any())
    off = submission.summarize(labels_off, blocks=ctx.blocks)["total"]
    on = submission.summarize(labels_on, blocks=ctx.blocks)["total"]
    changed = int(np.sum(np.asarray(labels_on) != np.asarray(labels_off)))
    return {"rule": rule, "owner_releases": n_releases, "flights": len(flights),
            "fraction_of_releases": len(flights) / float(n_releases) if n_releases else None,
            "fraction_of_possessions_off": len(flights) / float(off["possessions"]),
            "flight_len_median": _stat(lengths, np.median),
            "flight_len_p90": _stat(lengths, lambda v: np.percentile(v, 90)),
            "flight_len_max": _stat(lengths, np.max),
            "blocked_by_stoppage": len([b for b in parts_on["flights_blocked"] if graded[b[0]]]),
            "blocked": [(int(s), t, int(h)) for s, t, h in parts_on["flights_blocked"]
                        if graded[s]],
            "flights_through_stoppage": through_d, "flights_over_any_onset_row": through_onset,
            "possessions_off": off["possessions"], "possessions_on": on["possessions"],
            "d_share_off": off["d_share"], "d_share_on": on["d_share"],
            "d_segments_off": off["d_segments"], "d_segments_on": on["d_segments"],
            "frames_changed": changed,
            "public_flights": [f for f in parts_on["flights"] if 11185 <= f[0] < 18685]}


def _lat(value):
    return "    -" if value is None else "{:+5.1f}".format(value)


def _point_line(name, ctx, point):
    lat = kind_latencies(ctx, point)
    cells = "".join(" {}/{}".format(_lat(_stat(lat[k], np.mean)), _lat(_stat(lat[k], np.median)))
                    for k in KINDS)
    short = " ".join("{:.2f}{}".format(i or 0.0, "*" if m else "") for i, m in point["short"])
    return ("{:<34} {:.3f} {:.3f} {:>2}/{:>2}/{:>2} {:>2} {}  {:>3} {:>5.1%} {:>3} {:>2} {:<4} {}"
            .format(name, point["spell_f1"], point["macro_f1"], point["matched"], point["missed"],
                    point["spurious"], point["n_pred_spells"], cells, point["possessions"],
                    point["d_share"], point["d_segments"], point["setpiece_without_d"],
                    "ok" if point["vetoes_ok"] else "VETO", short))


HEADER = ("{:<34} {:>5} {:>5} {:>8} {:>2} {:^13}{:^13}{:^13}{:^13}  {:>3} {:>6} {:>3} {:>2} {:<4} {}"
          .format("grid point", "spF1", "mcF1", "m/mi/sp", "sp", "H<->A", "alive->D",
                  "D->alive kick", "D->alive reap", "pos", "D", "Dsg", "SP", "veto",
                  "short IoU (* = matched 0.5)"))


def calibrate(ctx):
    """Run stages A to D and the leave-one-out. Returns a dict of every number."""
    result = {"points": {}, "order": []}

    def add(name, point):
        result["points"][name] = point
        result["order"].append(name)
        return point

    # Stage A and B: rule x flight with all offsets at 0. The first one is M3.
    rf = {}
    for rule, flight in RULE_FLIGHT:
        _, labels, parts = ctx.run(POSSESSION_RULE=rule, FLIGHT_ASSIGNMENT=flight)
        name = "rule={} flight={}".format(rule, "on" if flight else "off")
        rf[(rule, flight)] = add(name, score(ctx, labels, parts))
    ref = rf[RULE_FLIGHT[0]]
    rule, flight = choose_rule_flight(ref, rf)
    result["rule_flight"] = (rule, flight)
    rf_loo = {g[0]: choose_rule_flight(ref, rf, (g[0],)) for g in ctx.gt_trans}
    rf_options = set([(rule, flight)] + list(rf_loo.values()))
    if len(rf_options) > 1:           # disagreement: the simpler choice (PLAN 5.1)
        rule, flight = RULE_FLIGHT[0]
    result["rule_flight_final"] = (rule, flight)
    zero = rf[(rule, flight)]

    # Stage C: one offset kind at a time.
    grids = {}
    for kind in KINDS:
        field = DELTA_OF_KIND[kind]
        values = kind_latencies(ctx, zero)[kind]
        cancel = _round(-np.median(values)) if values else 0
        grid = sorted(set(CARD_GRID[field]) | {cancel})
        grids[field] = {0: zero}
        for delta in grid:
            if delta == 0:
                continue
            _, labels, parts = ctx.run(POSSESSION_RULE=rule, FLIGHT_ASSIGNMENT=flight,
                                       **{field: delta})
            grids[field][delta] = add("  {}={:+d}".format(field, delta),
                                      score(ctx, labels, parts))
        result.setdefault("cancel", {})[field] = cancel
    full = {f: choose_delta(ctx, f, zero, grids[f]) for f in grids}
    result["choice_full"] = full

    # Leave-one-out of the offsets, on the final rule and flight.
    loo = []
    for g in ctx.gt_trans:
        ex = (g[0],)
        deltas = {f: choose_delta(ctx, f, zero, grids[f], ex) for f in grids}
        loo.append({"frame": g[0], "kind": ctx.kind[g[0]],
                    "from_to": "{}->{}".format(LETTER[g[1]], LETTER[g[2]]),
                    "rule_flight": rf_loo[g[0]], "deltas": deltas,
                    "moves": rf_loo[g[0]] != result["rule_flight"] or deltas != full})
    result["loo"] = loo
    final = {}
    for field in grids:
        options = [full[field]] + [row["deltas"][field] for row in loo]
        final[field] = full[field] if len(set(options)) == 1 else \
            sorted(options, key=lambda d: (abs(d), d))[0]
    result["choice_final"] = final

    # Stage D: the combined final choice.
    fr, ff = result["rule_flight_final"]
    _, labels, parts = ctx.run(POSSESSION_RULE=fr, FLIGHT_ASSIGNMENT=ff, **final)
    add("FINAL rule={} flight={} {}".format(fr, "on" if ff else "off", " ".join(
        "{}={:+d}".format(k.replace("DELTA_", ""), v) for k, v in sorted(final.items()))),
        score(ctx, labels, parts))
    result["lost_vs_m3"] = {name: lost(ref, p) for name, p in result["points"].items()}
    result["flight"] = flight_measurement(ctx, fr)
    mono, peak = rise_lengths(ctx.inputs["ball"], ctx.inputs["events"],
                              possession.owner_mask(ctx.inputs["events"], ctx.base),
                              ctx.blocks, ctx.base)
    result["rise"] = {"monotone": mono, "to_peak": peak}
    return result


def _distribution(values):
    values = np.asarray(list(values), dtype=np.float64)
    if not values.size:
        return "n=0"
    return "n={} mean {:.1f} median {:.0f} IQR {:.0f}-{:.0f} p10-p90 {:.0f}-{:.0f} sd {:.1f}".format(
        values.size, values.mean(), np.median(values), np.percentile(values, 25),
        np.percentile(values, 75), np.percentile(values, 10), np.percentile(values, 90),
        values.std())


def format_result(ctx, res):
    """The printout of calibrate() as a list of lines."""
    lines = ["CALIBRATION (public window, tol {}; latency = ours - GT, frames, mean/median of "
             "matched; SP = set-piece tags with no D within 10 s, full match)".format(TOL),
             HEADER]
    for name in res["order"]:
        lines.append(_point_line(name, ctx, res["points"][name]))
    lines += ["", "lost vs M3 (matched by M3 at tol 25, not by the grid point):"]
    for name in res["order"]:
        if res["lost_vs_m3"][name]:
            lines.append("  {:<34} {}".format(name, res["lost_vs_m3"][name]))
    lines += ["", "per-transition latency at tol 25 (m = matched, . = nearest same destination):",
              "  {:<10}".format("GT frame") + " ".join("{:>6}".format(g[0]) for g in ctx.gt_trans)]
    for name in res["order"]:
        cells = []
        for g in ctx.gt_trans:
            rec = res["points"][name]["per"][g[0]]
            lat = rec["latency"]
            text = "-" if lat is None else "{:+d}".format(lat)
            cells.append("{:>6}".format(text + ("m" if rec["status"] == "matched" else ".")))
        lines.append("  {:<10}".format(name[:10]) + " ".join(cells))
    lines += ["", "stage B choice: rule={} flight={}; after its leave-one-out: rule={} flight={}"
              .format(*(res["rule_flight"] + res["rule_flight_final"])),
              "set-piece tags with a D->alive within 10 s (full match, of {}): ".format(
                  len(ctx.setpiece_windows)) + ", ".join(
                  "{} {}".format(n.strip()[:28], res["points"][n]["setpiece_with_restart"])
                  for n in res["order"]),
              "cancel values: " + ", ".join("{}={:+d}".format(k, v)
                                            for k, v in sorted(res["cancel"].items())),
              "stage C choice (all 20): " + ", ".join("{}={:+d}".format(k, v)
                                                      for k, v in sorted(res["choice_full"].items())),
              "", "LEAVE-ONE-OUT (choice recomputed without the transition)",
              "  {:>6} {:<6} {:<20} {:<26} {:>6} {:>6} {:>6} {:>6} moves".format(
                  "frame", "from", "kind", "rule / flight", "switch", "dead", "kick", "reapp")]
    for row in res["loo"]:
        d = row["deltas"]
        lines.append("  {:>6} {:<6} {:<20} {:<26} {:>+6d} {:>+6d} {:>+6d} {:>+6d} {}".format(
            row["frame"], row["from_to"], row["kind"],
            "{} / {}".format(row["rule_flight"][0], "on" if row["rule_flight"][1] else "off"),
            d["DELTA_SWITCH"], d["DELTA_DEAD"], d["DELTA_RESTART_KICK"],
            d["DELTA_RESTART_REAPPEAR"], "MOVES" if row["moves"] else "-"))
    n_moves = sum(1 for row in res["loo"] if row["moves"])
    lines += ["  {} of {} leave-outs move the choice".format(n_moves, len(res["loo"])),
              "final (simpler choice and smallest |delta| on disagreement): rule={} flight={} {}"
              .format(res["rule_flight_final"][0], res["rule_flight_final"][1],
                  ", ".join("{}={:+d}".format(k, v) for k, v in sorted(res["choice_final"].items())))]
    fl = res["flight"]
    lines += ["", "FLIGHT ASSIGNMENT, full match (rule {})".format(fl["rule"]),
              "  owner releases on graded frames {}, flights {} ({:.1%} of releases, {:.1%} of the "
              "{} possessions with flight off)".format(
                  fl["owner_releases"], fl["flights"], fl["fraction_of_releases"] or 0.0,
                  fl["fraction_of_possessions_off"], fl["possessions_off"]),
              "  flight length (release to the interceptor's owner event), frames: median {} p90 {} "
              "max {}".format(fl["flight_len_median"], fl["flight_len_p90"], fl["flight_len_max"]),
              "  walks cut by a stoppage frame: {}; flights through a stoppage frame: {}; flights "
              "over any onset row (incl. collective_stop hints): {}".format(
                  fl["blocked_by_stoppage"], fl["flights_through_stoppage"],
                  fl["flights_over_any_onset_row"]),
              "  possessions off/on {}/{}; D share off/on {:.1%}/{:.1%}; D segments off/on {}/{}; "
              "frames changed {}".format(fl["possessions_off"], fl["possessions_on"],
                                         fl["d_share_off"], fl["d_share_on"],
                                         fl["d_segments_off"], fl["d_segments_on"],
                                         fl["frames_changed"]),
              "  public flights (release, from, to, end): {}".format(fl["public_flights"]),
              "  public walks cut by a stoppage: {}".format(
                  [b for b in fl["blocked"] if 11185 <= b[0] < 18685])]
    mono, peak = res["rise"]["monotone"], res["rise"]["to_peak"]
    lines += ["", "RELEASE RISE LENGTHS (owner releases, graded frames)",
              "  contact to end of monotone rise: " + _distribution(mono.values()),
              "  contact to highest speed within RELEASE_K: " + _distribution(peak.values())]
    for gt_frame in (13704, 14846, 15365):
        before = [c for c in mono if c <= gt_frame and gt_frame - c <= 2 * TOL]
        if before:
            c = max(before)
            lines.append("  GT {}: release contact {} (GT - contact {:+d}), monotone rise {}, "
                         "to peak {}".format(gt_frame, c, gt_frame - c, mono[c], peak[c]))
    return lines


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(type(value))


def main(argv=None):
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", default=None, help="also write every number to this file")
    args = parser.parse_args(argv)
    ctx = Context()
    res = calibrate(ctx)
    print("\n".join(format_result(ctx, res)))
    if args.json:
        rise = {k: {str(f): v for f, v in d.items()} for k, d in res["rise"].items()}
        dump = dict(res, rise=rise)
        for point in dump["points"].values():
            point["per"] = {str(f): r for f, r in point["per"].items()}
        with open(args.json, "w") as handle:
            json.dump(dump, handle, indent=1, default=_jsonable)
        print("\nwrote {}".format(args.json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
