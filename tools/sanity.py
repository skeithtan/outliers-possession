"""Full-match sanity dashboard (PLAN.md 5.3, card P0-D). Uses NO ground truth.

    python tools/sanity.py evaluation/results/<name>.csv [--json] [--strict] [--game name]

Per half and in total, computed INSIDE each graded block (a spell never straddles halftime, as in
the scorer): possessions, duration stats, D share and D segments, transitions per minute,
agreement with Smart Tagging team phases, set-piece tags without a nearby D, label changes inside
long ball gaps, X on graded frames, and the D segments that end at a block end other than the
half's end (a D segment that runs to the end of a block because no restart follows it). The PLAN 5.3 VETOES are printed beside each number and
violations are marked; the WATCHED ranges are shown as comments only. Vetoes catch pathology,
they are not a ranking (PLAN 1.7: a possession band would keep the worse baseline).

`sanity_stats()` is importable (engine/submission.py may call it for its end-of-run printout).
This module deliberately does not import tools.gt or touch evaluation/gt.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as cm  # noqa: E402
from _common import A, D, H, X, pm  # noqa: E402

# -- thresholds: PLAN.md 5.3 (vetoes) and 1.7 / 5.3 (watched). Not engine config. ----------------
VETO_POSSESSIONS = (180, 450)          # PLAN 5.3
VETO_D_SHARE = (0.18, 0.55)            # PLAN 5.3
VETO_MIN_D_SEGMENTS = 55               # PLAN 5.3 (59 tagged set pieces minus margin)
VETO_MAX_GRADED_X = 0                  # PLAN 1.7 / 5.3
WATCH_POSSESSIONS = (230, 330)         # PLAN 5.3
WATCH_D_SEGMENTS = (85, 150)           # PLAN 1.7 / 5.3
WATCH_MIN_TAG_AGREEMENT = 0.85         # PLAN 5.3 / card P0-D
WATCH_SHORT_SHARE = "a few percent (public window: 1 of 17)"
# Smart Tagging phases whose team is the team in possession (PLAN 1.6, card P0-D). DEFENDING*,
# DEFENSIVE TRANSITION and PLAYERS IN THE BOX name the defending / sub-phase side, so they are
# not used as a possession prior.
TAG_PHASE_CODES = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
                   "LONG BALL", "BALL IN FINAL THIRD", "BALL IN THE BOX")
SET_PIECE_CODE = "SET PIECES"
SET_PIECE_MARGIN_S = 10.0              # card P0-D: "no D segment within +-10 s"
LONG_GAP_S = 2.0                       # PLAN 1.1: long ball gap = missing for more than 2 s
SHORT_SPELL_S, LONG_SPELL_S = 1.0, 60.0
LIST_LIMIT = 20


def _as_codes(labels):
    """Accept int codes (0..3) or label strings; return int8 codes."""
    arr = np.asarray(labels)
    if arr.dtype.kind in "iu":
        return arr.astype(np.int8)
    return pm.label_codes(arr)


def _scope_stats(filled, raw, runs, fps=None):
    """Per-scope numbers for a set of half-open runs (one half, or both), at `fps` frames/s.

    `fps` None means the frame rate of games.GAME.
    """
    fps = fps or cm.fps()
    n_graded = int(sum(b - a for a, b in runs))
    spells = pm.possession_spells(filled, runs)
    lengths = np.array([b - a for a, b, _ in spells], dtype=np.float64)
    seconds = lengths / fps
    d_segments = [(a, b) for a, b, k in pm.segments(filled, runs) if k == D]
    d_frames = int(sum(int((filled[a:b] == D).sum()) for a, b in runs))
    n_transitions = len(pm.transitions(filled, runs))
    minutes = n_graded / fps / 60.0
    return {
        "graded_frames": n_graded,
        "graded_minutes": minutes,
        "possessions": len(spells),
        "poss_mean_s": float(seconds.mean()) if len(seconds) else None,
        "poss_median_s": float(np.median(seconds)) if len(seconds) else None,
        "poss_share_lt_1s": float((seconds < SHORT_SPELL_S).mean()) if len(seconds) else None,
        "poss_share_gt_60s": float((seconds > LONG_SPELL_S).mean()) if len(seconds) else None,
        "poss_longest_s": float(seconds.max()) if len(seconds) else None,
        "d_share": d_frames / float(n_graded) if n_graded else None,
        "d_frames": d_frames,
        "d_segments": len(d_segments),
        "d_segment_len": cm.segment_stats([b - a for a, b in d_segments], fps),
        "transitions": n_transitions,
        "transitions_per_min": n_transitions / minutes if minutes else None,
        "x_graded": int(sum(int((raw[a:b] == X).sum()) for a, b in runs)),
    }


def _tag_agreement(filled, tags, runs, n_frames):
    """Fraction of tagged graded frames where our alive label equals the Smart Tagging team."""
    paint = cm.tag_team_codes(tags, n_frames, TAG_PHASE_CODES, runs)
    sure = (paint == 0) | (paint == 1)
    alive = (filled == H) | (filled == A)
    n_conflict = int((paint == 2).sum())
    n_tagged = int(sure.sum())
    n_alive = int((sure & alive).sum())
    n_agree = int((sure & alive & (filled == paint)).sum())
    return {
        "tagged_frames": n_tagged,
        "conflict_frames_dropped": n_conflict,
        "tagged_and_our_alive": n_alive,
        "agree_frames": n_agree,
        # primary: of tagged frames where we say H/A, share equal to the tag team (card P0-D)
        "agreement": n_agree / float(n_alive) if n_alive else None,
        # strict: a D (or X) on a tagged frame counts as a disagreement
        "agreement_all_tagged": n_agree / float(n_tagged) if n_tagged else None,
        "our_d_share_on_tagged": (float((sure & (filled == D)).sum()) / n_tagged
                                  if n_tagged else None),
    }


def _set_piece_check(filled, tags, runs, fps=None):
    """Set-piece tags with no D frame within +-SET_PIECE_MARGIN_S of the tag window.

    `fps` None means the frame rate of games.GAME.
    """
    margin = int(round(SET_PIECE_MARGIN_S * (fps or cm.fps())))
    n = len(filled)
    graded = np.zeros(n, dtype=bool)
    for a, b in runs:
        graded[a:b] = True
    is_d = (filled == D) & graded
    rows = tags[tags["code"].str.upper() == SET_PIECE_CODE]
    n_considered, missing = 0, []
    for row in rows.itertuples(index=False):
        lo, hi = max(0, int(row.start_frame) - margin), min(n, int(row.end_frame) + margin)
        if not graded[lo:hi].any():
            continue  # tag lies wholly outside the graded blocks: nothing to check
        n_considered += 1
        if not is_d[lo:hi].any():
            missing.append({"id": int(row.id), "type": row.type, "team": row.team,
                            "half": int(row.half), "start_frame": int(row.start_frame),
                            "end_frame": int(row.end_frame)})
    return {"n_set_pieces": int(len(rows)), "n_checked": n_considered,
            "n_without_d": len(missing), "without_d": missing}


def _gap_label_changes(filled, ball_missing, runs, fps=None):
    """Label changes while the ball is missing inside a gap longer than LONG_GAP_S.

    A change at frame f counts when the ball is missing at f-1 and f, both inside one gap that
    is longer than LONG_GAP_S, inside one graded run (never across halftime). `fps` None means
    the frame rate of games.GAME.
    """
    min_len = int(round(LONG_GAP_S * (fps or cm.fps())))
    changes, n_gaps, n_gaps_with_change = [], 0, 0
    for run_start, run_stop in runs:
        miss = ball_missing[run_start:run_stop].astype(np.int8)
        edges = np.diff(np.concatenate(([0], miss, [0])))
        starts = np.flatnonzero(edges == 1) + run_start
        stops = np.flatnonzero(edges == -1) + run_start
        for gs, ge in zip(starts, stops):
            if ge - gs <= min_len:
                continue
            n_gaps += 1
            idx = np.flatnonzero(filled[gs + 1:ge] != filled[gs:ge - 1]) + gs + 1
            if len(idx):
                n_gaps_with_change += 1
            for f in idx.tolist():
                changes.append({"frame": f, "from": cm.LETTER[int(filled[f - 1])],
                                "to": cm.LETTER[int(filled[f])], "gap_start": int(gs),
                                "gap_frames": int(ge - gs)})
    return {"long_gaps": n_gaps, "gaps_with_change": n_gaps_with_change,
            "n_changes": len(changes), "first": changes[:LIST_LIMIT]}


def _d_at_block_end(filled, blocks, half_ends):
    """D segments whose last frame is the last frame of a graded block.

    `blocks` is [(half, first, last_inclusive)]. `half_ends` is {half: last frame of the half}
    from games.py. A D segment at the end of a block that is not the end of its half has no
    restart after it (CODE-REVIEW-RESPONSE finding 4). A D segment at the half's end is the
    end of play.
    Returns {"n_at_block_end", "n_not_half_end", "segments": [{half, start, end, half_end}]}.
    """
    rows = []
    for half, first, last in blocks:
        segs = pm.segments(filled, [(int(first), int(last) + 1)])
        if not segs or segs[-1][2] != D:
            continue
        start, stop, _ = segs[-1]
        rows.append({"half": int(half), "start": int(start), "end": int(stop - 1),
                     "half_end": half_ends.get(int(half)) == int(stop - 1)})
    return {"n_at_block_end": len(rows),
            "n_not_half_end": sum(1 for r in rows if not r["half_end"]), "segments": rows}


def _veto_table(total):
    """[(name, value, rule text, violated)] for the PLAN 5.3 vetoes, plus watched flags."""
    p, d, s = total["possessions"], total["d_share"], total["d_segments"]
    vetoes = [
        {"name": "possessions", "value": p, "rule": "{}-{}".format(*VETO_POSSESSIONS),
         "violated": not (VETO_POSSESSIONS[0] <= p <= VETO_POSSESSIONS[1])},
        {"name": "d_share", "value": d,
         "rule": "{:.0f}-{:.0f}%".format(VETO_D_SHARE[0] * 100, VETO_D_SHARE[1] * 100),
         "violated": not (VETO_D_SHARE[0] <= d <= VETO_D_SHARE[1])},
        {"name": "d_segments", "value": s, "rule": ">= {}".format(VETO_MIN_D_SEGMENTS),
         "violated": s < VETO_MIN_D_SEGMENTS},
        {"name": "x_graded", "value": total["x_graded"], "rule": "== {}".format(VETO_MAX_GRADED_X),
         "violated": total["x_graded"] != VETO_MAX_GRADED_X},
    ]
    return vetoes


def sanity_stats(labels, tags=None, ball_missing=None, blocks=None, use_tags=True, use_ball=True,
                 game=None):
    """All sanity numbers for one label vector (int codes 0..3 or H/A/D/X strings), as a dict.

    labels         length-N vector, one entry per tactical video frame
    tags           Smart Tagging DataFrame (engine.data.load_tagging()); loaded if None
    ball_missing   bool (N,) ball-missing mask (from engine.data.load_tactical()); loaded if None
    blocks         [(half, first, last_inclusive)] graded blocks; engine.data.graded_blocks()
    use_tags / use_ball  False skips those checks (no data bundle needed)
    game           the match of games.py that gives the defaults and the fps (None: games.GAME)
    Returns {"halves": {1: .., 2: ..}, "total": .., "tags": .., "set_pieces": .., "gaps": ..,
    "vetoes": [..], "violations": [names], "watched": {...}}.
    """
    raw = _as_codes(labels)
    fps = cm.fps(game)
    if blocks is None or (use_tags and tags is None) or (use_ball and ball_missing is None):
        from engine import data  # local: sanity_stats on bare arrays should not need the bundle
        blocks = blocks if blocks is not None else data.graded_blocks(game)
        if use_tags and tags is None:
            tags = data.load_tagging(game)
        if use_ball and ball_missing is None:
            ball_missing = np.isnan(data.load_tactical(game)["bx"])
    runs = cm.runs_from_blocks(blocks)
    filled = pm.fill_abstentions(raw, runs)  # X -> held label, exactly as the scorer does

    halves = {int(h): _scope_stats(filled, raw, [(a, b + 1)], fps) for h, a, b in blocks}
    total = _scope_stats(filled, raw, runs, fps)
    out = {"halves": halves, "total": total, "tag_phase_codes": list(TAG_PHASE_CODES)}
    half_ends = {int(h): int(b) for h, _, b in cm.games.half_video_blocks(cm.game_name(game))}
    out["d_block_end"] = _d_at_block_end(filled, blocks, half_ends)
    if use_tags:
        out["tag_codes_present"] = sorted(set(tags["code"].str.upper()))
        out["tag_codes_missing"] = [c for c in TAG_PHASE_CODES + (SET_PIECE_CODE,)
                                    if c not in out["tag_codes_present"]]
        out["tags"] = {"total": _tag_agreement(filled, tags, runs, len(raw)),
                       "halves": {int(h): _tag_agreement(filled, tags, [(a, b + 1)], len(raw))
                                  for h, a, b in blocks}}
        out["set_pieces"] = _set_piece_check(filled, tags, runs, fps)
    if use_ball:
        out["gaps"] = _gap_label_changes(filled, np.asarray(ball_missing, dtype=bool), runs, fps)
    out["vetoes"] = _veto_table(total)
    out["violations"] = [v["name"] for v in out["vetoes"] if v["violated"]]
    p, s = total["possessions"], total["d_segments"]
    out["watched"] = {
        "possessions_outside_230_330": not (WATCH_POSSESSIONS[0] <= p <= WATCH_POSSESSIONS[1]),
        "d_segments_outside_85_150": not (WATCH_D_SEGMENTS[0] <= s <= WATCH_D_SEGMENTS[1]),
        "tag_agreement_below_85pct": bool(
            use_tags and (out["tags"]["total"]["agreement"] or 0.0) < WATCH_MIN_TAG_AGREEMENT),
        "gap_changes_nonzero": bool(use_ball and out["gaps"]["n_changes"] > 0),
    }
    return out


# -- printing ------------------------------------------------------------------------------------

def _f(value, fmt="{:.1f}", none="-"):
    return none if value is None else fmt.format(value)


def format_report(stats, name=""):
    """The dashboard as a list of text lines."""
    t, hv = stats["total"], stats["halves"]
    cols = [("H1", hv.get(1)), ("H2", hv.get(2)), ("TOTAL", t)]
    lines = ["SANITY  {}   (no ground truth used)".format(name),
             "{:<30}{:>12}{:>12}{:>12}".format("", *[c for c, _ in cols])]

    def row(label, key, fmt="{:.1f}", scale=1.0, sub=None):
        cells = []
        for _, s in cols:
            v = s[key] if sub is None else s[key][sub]
            cells.append(_f(None if v is None else v * scale, fmt))
        lines.append("{:<30}{:>12}{:>12}{:>12}".format(label, *cells))

    row("graded minutes", "graded_minutes")
    row("possessions", "possessions", "{:.0f}")
    row("  mean duration (s)", "poss_mean_s")
    row("  median duration (s)", "poss_median_s")
    row("  share < 1 s (%)", "poss_share_lt_1s", "{:.1f}", 100)
    row("  share > 60 s (%)", "poss_share_gt_60s", "{:.1f}", 100)
    row("  longest (s)", "poss_longest_s")
    row("D share of graded (%)", "d_share", "{:.1f}", 100)
    row("D segments", "d_segments", "{:.0f}")
    row("  length min (s)", "d_segment_len", "{:.1f}", 1, "min_s")
    row("  length median (s)", "d_segment_len", "{:.1f}", 1, "median_s")
    row("  length p90 (s)", "d_segment_len", "{:.1f}", 1, "p90_s")
    row("  length max (s)", "d_segment_len", "{:.1f}", 1, "max_s")
    row("transitions / min", "transitions_per_min", "{:.2f}")
    row("X on graded frames", "x_graded", "{:.0f}")
    if "d_block_end" in stats:
        ends = stats["d_block_end"]
        lines.append("D segments that end at a block end other than the half's end: {}".format(
            ends["n_not_half_end"]))
        for seg in ends["segments"]:
            lines.append("  half {}  D {}-{}  {}".format(
                seg["half"], seg["start"], seg["end"],
                "at the half's end" if seg["half_end"] else "at a block end, no restart"))

    if "tags" in stats:
        lines += ["", "Smart Tagging team phases ({})".format(", ".join(stats["tag_phase_codes"]))]
        lines.append("  codes present in load_tagging(): " + ", ".join(stats["tag_codes_present"]))
        if stats["tag_codes_missing"]:
            lines.append("  !! expected codes NOT present: " + ", ".join(stats["tag_codes_missing"]))
        for label, key in (("H1", 1), ("H2", 2), ("TOTAL", None)):
            ag = stats["tags"]["total"] if key is None else stats["tags"]["halves"][key]
            lines.append("  {:<6} agreement {} of {} tagged-and-alive frames "
                         "(strict incl. our D: {}; our D on tagged: {}; {} tagged frames, "
                         "{} two-team frames dropped)".format(
                             label, _f(None if ag["agreement"] is None else ag["agreement"] * 100,
                                       "{:.1f}%"), ag["tagged_and_our_alive"],
                             _f(None if ag["agreement_all_tagged"] is None
                                else ag["agreement_all_tagged"] * 100, "{:.1f}%"),
                             _f(None if ag["our_d_share_on_tagged"] is None
                                else ag["our_d_share_on_tagged"] * 100, "{:.1f}%"),
                             ag["tagged_frames"], ag["conflict_frames_dropped"]))
        sp = stats["set_pieces"]
        lines += ["", "Set-piece tags with no D within +-{:.0f} s of the tag: {} of {} checked "
                      "({} tags total)".format(SET_PIECE_MARGIN_S, sp["n_without_d"],
                                               sp["n_checked"], sp["n_set_pieces"])]
        for item in sp["without_d"]:
            lines.append("  id {:>4}  half {}  {!s:<22} team {!s:<4} frames {}-{}".format(
                item["id"], item["half"], item["type"], item["team"], item["start_frame"],
                item["end_frame"]))
    if "gaps" in stats:
        g = stats["gaps"]
        lines += ["", "Label changes inside ball gaps > {:.0f} s: {} changes in {} of {} long "
                      "gaps".format(LONG_GAP_S, g["n_changes"], g["gaps_with_change"],
                                    g["long_gaps"])]
        for c in g["first"]:
            lines.append("  frame {:>6}  {}->{}  (gap from {}, {} frames)".format(
                c["frame"], c["from"], c["to"], c["gap_start"], c["gap_frames"]))
        if g["n_changes"] > len(g["first"]):
            lines.append("  ... {} more".format(g["n_changes"] - len(g["first"])))

    lines += ["", "VETOES (PLAN 5.3; a violation rejects the version regardless of public score)"]
    for v in stats["vetoes"]:
        value = ("{:.1f}%".format(v["value"] * 100) if v["name"] == "d_share"
                 else "{:.0f}".format(v["value"]))
        lines.append("  {:<14}{:>9}   veto {:<10} {}".format(
            v["name"], value, v["rule"], "VIOLATION" if v["violated"] else "ok"))
    w = stats["watched"]
    lines += ["WATCHED (comments only; PLAN 5.3, not gates)",
              "  possessions 230-330        {}".format("outside" if w["possessions_outside_230_330"] else "inside"),
              "  D segments ~85-150         {}".format("outside" if w["d_segments_outside_85_150"] else "inside"),
              "  share of spells < 1 s      {}".format(WATCH_SHORT_SHARE),
              "  Smart Tagging agreement > 85%   {}".format(
                  "BELOW" if w["tag_agreement_below_85pct"] else "ok"),
              "  label changes in gaps ~ 0  {}".format(
                  "nonzero" if w["gap_changes_nonzero"] else "ok"),
              "RESULT: {}".format("VETO VIOLATED: " + ", ".join(stats["violations"])
                                  if stats["violations"] else "no veto violated")]
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results", help="results CSV (frame,label)")
    parser.add_argument("--json", action="store_true",
                        help="also write evaluation/reports/<name>_sanity.json")
    parser.add_argument("--strict", action="store_true", help="exit 1 if a veto is violated")
    parser.add_argument("--game", default=None, choices=sorted(cm.games.GAMES),
                        help="match of games.py (default: games.GAME)")
    args = parser.parse_args(argv)

    name = cm.result_name(args.results)
    codes = cm.read_results(args.results, args.game)
    stats = sanity_stats(codes, game=args.game)
    print("\n".join(format_report(stats, name)))
    if args.json:
        os.makedirs(cm.REPORTS_DIR, exist_ok=True)
        path = os.path.join(cm.REPORTS_DIR, name + "_sanity.json")
        with open(path, "w") as handle:
            json.dump(stats, handle, indent=2, sort_keys=True, default=_jsonable)
        print("\nWrote {}".format(path))
    return 1 if (args.strict and stats["violations"]) else 0


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(type(value))


if __name__ == "__main__":
    sys.exit(main())
