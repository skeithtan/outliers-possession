"""Three diagnostics of the default engine, 8 Nov 2026. Not part of the submission.

    python tools/analysis/diagnostics_8nov.py [--jobs N] [--cache DIR] [--out FILE.json]
        [--refs-root DIR] [--sample-root DIR] [--mtah-root DIR] [--bar-data DIR]

The diagnostics decide nothing. They inform the team. The report is
docs/plan/calibration/DIAGNOSTICS_8NOV.md.

D1. Minimum-duration bypass. engine/decoder.py runs viterbi() and then min_durations(). The tool
wraps engine.decoder.decode in its own process, so it sees the Viterbi path before the post-pass
and the path after it. It never edits engine/. It counts, per match:

  (a) changes of owner (H to A or A to H) across a D run shorter than MIN_DEAD in the Viterbi path,
  (b) of those, the changes that survive the post-pass (a direct switch with the D absorbed, or a
      switch across a D that stays),
  (c) all changes of owner between consecutive alive runs, with or without D between them,
  (d) the share (a) / (c).

D2. Oracle decomposition on the external references. spellF1 and macroF1 of (i) our output,
(ii) our D mask with the reference team (oracle_team), (iii) the reference D mask with our team
(oracle_dead).

D3. Ball process. The share of frames with the ball visible and stationary, split by the reference
D and alive frames (DFL) and by our D and alive frames (all matches, BAR-ATH included). The kinds
of our stoppage entries and exits per graded hour.

BAR-ATH is label-free here. The tool reads only its tracking and our engine output. It never
opens evaluation/gt/*. It runs the public scorer on no match. It refuses the sealed matches.

Runs. Each match and view runs in a child process (`--child`) with MTAH_DATA_DIR set to the root
of that view (external_sweep.data_root) and the npz cache of engine.data in
<cache>/<view>/engine_data. The child writes <cache>/<view>/<game>.npz (per-frame arrays) and
<game>.json (summaries). The parent prints summaries only: counts, shares and scores.

Deterministic: matches in sorted order, sorted JSON, no clock in the outputs. Python 3.9.
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, "tools")
ANALYSIS = os.path.join(TOOLS, "analysis")
for _path in (ROOT, TOOLS, ANALYSIS, os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

H, A, D, X = 0, 1, 2, 3          # the codes of possession_metrics.LABELS and engine.decoder
ALIVE = (H, A)
NO_PATH = -1                     # frames outside the graded blocks in a stored path
BAR = "BAR-ATH"
SEALED = ("DFL-J03WQQ", "DFL-J03WR9")
OPEN_GAMES = ("METRICA-S1", "METRICA-S2", "METRICA-S3",
              "DFL-J03WMX", "DFL-J03WN1", "DFL-J03WOH", "DFL-J03WOY", "DFL-J03WPY")
VIEWS = ("raw", "matched")
DEFAULT_CACHE = os.path.join(ROOT, "cache", "diagnostics_8nov")
DEFAULT_BAR_DATA = os.path.join("~", "mtah2027", "data")
# The advisor's decision rule for D1: a fix is warranted at 1 % or more on BAR-ATH.
D1_THRESHOLD = 0.01
# The order in which a dwell entry gets its kind when more than one e_D term is active.
DWELL_KINDS = ("dwell_out", "dwell_placed", "dwell_gap_stop", "dwell_tag_tail", "dwell_other")


# -- pure helpers ----------------------------------------------------------------------------

def runs(codes):
    """(starts, stops, labels) of the maximal constant runs of an int array (half-open)."""
    codes = np.asarray(codes)
    if codes.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    breaks = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.concatenate(([0], breaks)).astype(np.int64)
    stops = np.concatenate((breaks, [codes.size])).astype(np.int64)
    return starts, stops, codes[starts]


def owner_changes(path):
    """(changes, direct) of a block path of H, A and D codes.

    changes: pairs of consecutive alive runs with different teams, with or without D between.
    direct: H to A or A to H on adjacent frames.
    """
    path = np.asarray(path)
    alive = path[np.isin(path, ALIVE)]
    _, _, labels = runs(alive)
    changes = max(0, len(labels) - 1)
    direct = int(np.sum(np.isin(path[:-1], ALIVE) & np.isin(path[1:], ALIVE)
                        & (path[:-1] != path[1:])))
    return changes, direct


def _nearest_alive(path, frame, step):
    """The code of the first alive frame from `frame` in direction `step`, or None."""
    t = int(frame)
    while 0 <= t < len(path):
        if path[t] in ALIVE:
            return int(path[t])
        t += step
    return None


def best_direct(log_e, costs, start, stop, left, right):
    """(score, k) of the best path that switches left -> right once, with no D, on [start, stop).

    Frames start..k-1 take `left`, frames k..stop-1 take `right`. The switch costs
    costs[k, left, right]. k runs over start..stop. Frame `stop` is the first frame of the right
    run, so k == stop keeps `left` on the whole segment.
    """
    seg_l = np.concatenate(([0.0], np.cumsum(log_e[start:stop, left])))
    seg_r = np.concatenate(([0.0], np.cumsum(log_e[start:stop, right])))
    best, best_k = -np.inf, start
    for k in range(start, stop + 1):
        score = seg_l[k - start] + seg_r[-1] - seg_r[k - start] - costs[k, left, right]
        if score > best:
            best, best_k = score, k
    return float(best), int(best_k)


def bypass_records(vit, final, log_e, costs, min_dead):
    """[dict]: the changes of owner across a D run shorter than `min_dead` in the Viterbi path.

    `vit` and `final` are the paths of one block before and after min_durations(). `log_e` (n, 3)
    and `costs` (n, 3, 3) are the decoder inputs. Each record holds:
      start, stop   the D run [start, stop) in `vit`
      left, right   the teams of the alive runs on each side (they differ)
      entry, exit   costs[start, left, D] and costs[stop, D, right]
      detour        the score of the D run: sum log_e[., D] - entry - exit
      direct        the best score of one direct switch on the same frames (best_direct())
      advantage     detour - direct, in nats. Viterbi is exact, so it is >= 0 on unforced frames
      survival      "direct" if the final path still changes team across [start, stop) with no D
                    there, "with_d" if it changes team and keeps a D frame there, "removed"
                    if the post-pass gives both sides the same team
    """
    vit = np.asarray(vit)
    final = np.asarray(final)
    starts, stops, labels = runs(vit)
    out = []
    for idx in range(1, len(labels) - 1):
        start, stop = int(starts[idx]), int(stops[idx])
        left, right = int(labels[idx - 1]), int(labels[idx + 1])
        if labels[idx] != D or stop - start >= min_dead:
            continue
        if left not in ALIVE or right not in ALIVE or left == right:
            continue
        entry = float(costs[start, left, D])
        exit_ = float(costs[stop, D, right])
        detour = float(log_e[start:stop, D].sum()) - entry - exit_
        direct, _ = best_direct(log_e, costs, start, stop, left, right)
        before = _nearest_alive(final, start - 1, -1)
        after = _nearest_alive(final, stop, 1)
        if before is None or after is None or before == after:
            survival = "removed"
        elif np.any(final[start:stop] == D):
            survival = "with_d"
        else:
            survival = "direct"
        out.append({"start": start, "stop": stop, "length": stop - start, "left": left,
                    "right": right, "entry": entry, "exit": exit_, "detour": detour,
                    "direct": direct, "advantage": detour - direct, "survival": survival})
    return out


def d_run_profile(path, min_dead):
    """{counts and lengths} of the D runs of a block path, by their neighbours.

    d_runs: all D runs. short: D runs shorter than `min_dead`. Each class splits by the alive
    runs on its two sides: "change" (two alive runs of different teams), "same" (the same team)
    or "edge" (a block edge on one side). change_lengths: the lengths of the "change" runs.
    """
    starts, stops, labels = runs(path)
    out = {"d_runs": 0, "short": 0, "short_change": 0, "short_same": 0, "short_edge": 0,
           "change": 0, "change_lengths": []}
    for idx, label in enumerate(labels):
        if label != D:
            continue
        length = int(stops[idx] - starts[idx])
        if idx == 0 or idx == len(labels) - 1:
            kind = "edge"
        elif labels[idx - 1] != labels[idx + 1]:
            kind = "change"
        else:
            kind = "same"
        out["d_runs"] += 1
        if kind == "change":
            out["change"] += 1
            out["change_lengths"].append(length)
        if length < min_dead:
            out["short"] += 1
            out["short_" + kind] += 1
    return out


def merge_profiles(profiles):
    """The sum of d_run_profile() dicts, with the change lengths summarised as quantiles."""
    out = {"d_runs": 0, "short": 0, "short_change": 0, "short_same": 0, "short_edge": 0,
           "change": 0}
    lengths = []
    for prof in profiles:
        for key in out:
            out[key] += prof[key]
        lengths += prof["change_lengths"]
    out["change_length_min"] = int(min(lengths)) if lengths else None
    out["change_length_p10"] = float(np.percentile(lengths, 10)) if lengths else None
    out["change_length_median"] = float(np.median(lengths)) if lengths else None
    return out


def d1_counts(records, changes_vit, changes_final, cfg_costs):
    """The D1 summary of one match from its bypass records and owner-change counts.

    `cfg_costs` is {"C_DEAD_EVENT", "C_ALIVE"}: an entry at C_DEAD_EVENT is near an onset and an
    exit below C_ALIVE is near a restart.
    """
    n_a = len(records)
    survive = [r for r in records if r["survival"] != "removed"]
    near_onset = [r["entry"] <= cfg_costs["C_DEAD_EVENT"] for r in records]
    near_restart = [r["exit"] < cfg_costs["C_ALIVE"] for r in records]
    windows = {"onset_and_restart": 0, "onset_only": 0, "restart_only": 0, "neither": 0}
    for on, re in zip(near_onset, near_restart):
        key = ("onset_and_restart" if on and re else "onset_only" if on
               else "restart_only" if re else "neither")
        windows[key] += 1
    adv = [r["advantage"] for r in records]
    return {
        "a_bypass": n_a,
        "b_survive": len(survive),
        "b_survive_direct": sum(r["survival"] == "direct" for r in records),
        "b_survive_with_d": sum(r["survival"] == "with_d" for r in records),
        "b_removed": sum(r["survival"] == "removed" for r in records),
        "c_changes_viterbi": int(changes_vit[0]),
        "c_direct_viterbi": int(changes_vit[1]),
        "c_changes_final": int(changes_final[0]),
        "c_direct_final": int(changes_final[1]),
        "d_share": n_a / float(changes_vit[0]) if changes_vit[0] else 0.0,
        "b_share_final": len(survive) / float(changes_final[0]) if changes_final[0] else 0.0,
        "windows": windows,
        "d_length_median": float(np.median([r["length"] for r in records])) if records else None,
        "advantage_median": float(np.median(adv)) if adv else None,
        "advantage_max": float(np.max(adv)) if adv else None,
    }


def oracle_team(ours, ref):
    """int8 codes: our D mask with the reference team (D2 view ii).

    On our alive frames: the reference team where the reference is alive, our team where the
    reference says D or X. Our D and X frames stay as they are.
    """
    ours = np.asarray(ours, dtype=np.int8)
    ref = np.asarray(ref, dtype=np.int8)
    out = ours.copy()
    take = np.isin(ours, ALIVE) & np.isin(ref, ALIVE)
    out[take] = ref[take]
    return out


def nearest_alive_fill(codes, segments):
    """int8 codes: each frame takes the team of the nearest alive frame of `codes`.

    The search stays inside each half-open segment (start, stop). On a tie the earlier frame wins.
    A segment with no alive frame keeps its codes.
    """
    codes = np.asarray(codes, dtype=np.int8)
    out = codes.copy()
    for start, stop in segments:
        seg = codes[start:stop]
        alive = np.flatnonzero(np.isin(seg, ALIVE))
        if alive.size == 0:
            continue
        idx = np.arange(seg.size)
        pos = np.searchsorted(alive, idx)               # first alive index >= idx
        right = alive[np.minimum(pos, alive.size - 1)]
        left = alive[np.maximum(pos - 1, 0)]
        use_left = ((np.abs(idx - left) <= np.abs(right - idx)) & (pos > 0)) | (pos >= alive.size)
        pick = np.where(use_left, left, right)
        out[start:stop] = seg[pick]
    return out


def oracle_dead(ours, ref, segments):
    """int8 codes: the reference D mask with our team (D2 view iii).

    On reference D frames: D. On reference alive frames: our team where we are alive, else the
    team of our nearest alive frame in the same graded segment (nearest_alive_fill(), ties to the
    earlier frame). Reference X frames keep our codes (the scorer does not grade them).
    """
    ours = np.asarray(ours, dtype=np.int8)
    ref = np.asarray(ref, dtype=np.int8)
    filled = nearest_alive_fill(ours, segments)
    out = ours.copy()
    out[ref == D] = D
    alive = np.isin(ref, ALIVE)
    out[alive] = filled[alive]
    return out


def share(mask, within):
    """Share of the True frames of `within` where `mask` is True (0.0 if `within` is empty)."""
    within = np.asarray(within, dtype=bool)
    total = int(within.sum())
    return float(np.sum(np.asarray(mask, dtype=bool) & within)) / total if total else 0.0


def dwell_kind(flags):
    """The kind of a dwell entry from its active e_D terms {"out", "placed", "gap_stop", ...}.

    The first active kind of DWELL_KINDS wins. No active term gives "dwell_other".
    """
    for kind in DWELL_KINDS[:-1]:
        if flags.get(kind[len("dwell_"):]):
            return kind
    return DWELL_KINDS[-1]


# -- the child: one engine run with the decoder wrapped -------------------------------------

def _entry_exit_kinds(inputs, cfg):
    """({kind: count} of stoppage entries, {kind: count} of exits) before the stoppage shift.

    The stoppage is possession.dead_intervals() on the detection inputs. An entry at an onset row
    takes the onset mechanism (out split by confidence). Any other entry is a dwell of e_D and
    takes dwell_kind() of its terms. An exit takes the kind of its restart candidate (kick split
    into placed and open), "kickoff", or "block_end" when no restart follows.
    """
    from engine import deadball, possession
    from engine.tracks import _runs, velocity_step

    ball, feats, blocks = inputs["ball"], inputs["feats"], inputs["blocks"]
    kickoffs, fps = inputs["kickoffs"], inputs["fps"]
    pitch = possession.match_inputs(inputs["game"])[4]
    stoppage, restart_list = possession.dead_intervals(
        inputs["n_frames"], inputs["onsets"], inputs["e_dead"], inputs["restarts"], blocks,
        kickoffs, cfg, fps, inputs.get("holds"))
    onsets = inputs["onsets"]
    onset_kind = {}
    out_name = {cfg.CONF_OUT_OBSERVED: "out_observed", cfg.CONF_OUT_EXTRAPOLATED:
                "out_extrapolated", cfg.CONF_OUT_LAG: "out_lag"}
    for frame, mech, conf in zip(onsets["frame"], onsets["mechanism"], onsets["confidence"]):
        if mech == "collective_stop":
            continue
        name = out_name.get(float(conf), "out_other") if mech == "out" else str(mech)
        onset_kind.setdefault(int(frame), name)
    slow = deadball._below(feats["meanv_all"], cfg.V_STOP)
    terms = {"out": deadball.outside_pitch(ball, pitch, cfg.PITCH_MARGIN_M),
             "placed": deadball.setpiece_mask(ball, feats, cfg),
             "gap_stop": ~ball["visible"] & slow,
             "tag_tail": deadball.tag_tail_mask(inputs["tagging"], inputs["n_frames"], cfg, fps)}
    kick_frames = set(int(f) for f in kickoffs.values())
    entries = {}
    for half, first, last in blocks:
        starts, _ = _runs(stoppage[first:last + 1])
        for start in starts + first:
            start = int(start)
            if start == first and int(kickoffs.get(half, -1)) > first:
                continue                                 # the pre-kickoff stretch
            kind = onset_kind.get(start) or dwell_kind({k: bool(v[start])
                                                         for k, v in terms.items()})
            entries[kind] = entries.get(kind, 0) + 1
    # Placed-ball kicks, rebuilt as deadball.restarts() builds them.
    speed, n, k = ball["speed"], ball["speed"].shape[0], int(cfg.RESTART_K)
    rise = deadball._above(velocity_step(speed, k, ball["run_id"]), cfg.DV_RESTART)
    r_starts, r_stops = _runs(rise)
    rises = [(deadball._rise_start(speed, int(s), min(n - 1, int(e) + k)), int(e))
             for s, e in zip(r_starts, r_stops)]
    placed = set()
    if cfg.PLACED_BALL_RESTART:
        off = deadball.outside_pitch(ball, pitch, cfg.PITCH_MARGIN_M)
        placed = set(f for f, _ in deadball.placed_kicks(
            ball, deadball.stationary_mask(ball, cfg), rises, off))
    confirmed = inputs["restarts"][inputs["restarts"]["confirmed"]]
    cand_kind = {}
    for frame, kind in zip(confirmed["frame"], confirmed["kind"]):
        cand_kind.setdefault(int(frame), str(kind))
    exits = {}
    for frame, _ in restart_list:
        frame = int(frame)
        if frame in kick_frames:
            kind = "kickoff"
        else:
            kind = cand_kind.get(frame, "other")
            if kind == "kick":
                kind = "kick_placed" if frame in placed else "kick_open"
        exits[kind] = exits.get(kind, 0) + 1
    n_stoppages = sum(entries.values())
    n_ends = sum(v for key, v in exits.items() if key != "kickoff")
    if n_stoppages > n_ends:
        exits["block_end"] = n_stoppages - n_ends
    return entries, exits, stoppage


def child(game, out_prefix):
    """Run the default engine on `game`, wrap the decoder, write <prefix>.npz and <prefix>.json.

    MTAH_DATA_DIR and the engine.data cache are set by the parent. The function returns 0.
    """
    from engine import decoder, deadball, possession
    from engine.config import Config

    cfg = Config.default()
    blocks_seen = []
    original = decoder.decode

    def traced(log_e, costs, cfg_, forced=None):
        vit = decoder.viterbi(log_e, costs, forced)
        final = decoder.min_durations(vit, log_e, costs, cfg_, forced)
        blocks_seen.append((vit, final, decoder._apply_forced(log_e, forced), costs))
        return final

    decoder.decode = traced
    try:
        inputs = possession.prepare(cfg, game)
        labels, parts = possession.run(cfg, return_parts=True, inputs=inputs)
    finally:
        decoder.decode = original

    n = int(inputs["n_frames"])
    vit_full = np.full(n, NO_PATH, dtype=np.int8)
    records, ch_v, ch_f = [], [0, 0], [0, 0]
    prof_v, prof_f = [], []
    for (half, first, last), (vit, final, log_e, costs) in zip(inputs["blocks"], blocks_seen):
        vit_full[first:last + 1] = vit
        for rec in bypass_records(vit, final, log_e, costs, int(cfg.MIN_DEAD)):
            rec["start"] += first
            rec["stop"] += first
            records.append(rec)
        prof_v.append(d_run_profile(vit, int(cfg.MIN_DEAD)))
        prof_f.append(d_run_profile(final, int(cfg.MIN_DEAD)))
        for acc, path in ((ch_v, vit), (ch_f, final)):
            c, d = owner_changes(path)
            acc[0] += c
            acc[1] += d
    codes = np.full(n, X, dtype=np.int8)
    for idx, letter in enumerate("HADX"):
        codes[labels == letter] = idx
    ball = inputs["ball"]
    stationary = deadball.stationary_mask(ball, cfg)
    entries, exits, stoppage = _entry_exit_kinds(inputs, cfg)
    onset_counts = {}
    for mech, conf in zip(inputs["onsets"]["mechanism"], inputs["onsets"]["confidence"]):
        key = str(mech)
        if mech == "out":
            key = {cfg.CONF_OUT_OBSERVED: "out_observed",
                   cfg.CONF_OUT_EXTRAPOLATED: "out_extrapolated",
                   cfg.CONF_OUT_LAG: "out_lag"}.get(float(conf), "out_other")
        onset_counts[key] = onset_counts.get(key, 0) + 1
    graded = int(sum(last - first + 1 for _, first, last in inputs["blocks"]))
    lengths = []
    for _, first, last in inputs["blocks"]:
        st, sp, lab = runs(np.asarray(parts["stoppage"][first:last + 1], dtype=np.int8))
        lengths += [int(b - a) for a, b, v in zip(st, sp, lab)
                    if v == 1 and a > 0 and b < last + 1 - first]   # inner stoppages only
    np.savez_compressed(out_prefix + ".npz", labels=codes, vit=vit_full,
                        visible=np.asarray(ball["visible"], dtype=bool),
                        stationary=np.asarray(stationary, dtype=bool),
                        speed=np.asarray(ball["speed"], dtype=np.float32),
                        stoppage=np.asarray(stoppage, dtype=bool))
    summary = {
        "game": game, "fps": float(inputs["fps"]), "graded_frames": graded,
        "d1": d1_counts(records, ch_v, ch_f, {"C_DEAD_EVENT": cfg.C_DEAD_EVENT,
                                              "C_ALIVE": cfg.C_ALIVE}),
        "d_runs_viterbi": merge_profiles(prof_v), "d_runs_final": merge_profiles(prof_f),
        # The shortest stoppage of the decoder input (after shift_stoppage), inner runs only.
        "stoppage_min_frames": int(min(lengths)) if lengths else None,
        "onsets": onset_counts, "entries": entries, "exits": exits,
        "stationary_speed_mps": cfg.STATIONARY_MAX_DISP_M / cfg.STATIONARY_S,
        "min_dead": int(cfg.MIN_DEAD),
    }
    with open(out_prefix + ".json", "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


# -- the parent ------------------------------------------------------------------------------

def check_game(game):
    """Raise SystemExit for a match the tool must not run."""
    if game in SEALED:
        raise SystemExit("{} is sealed (docs/plan/calibration/SPLITS.md)".format(game))
    if game != BAR and game not in OPEN_GAMES:
        raise SystemExit("{} is not one of the matches of this diagnostic".format(game))


def run_child(job):
    """Run one child process unless its outputs exist. Return the JSON path."""
    game, view, root, cache = job["game"], job["view"], job["root"], job["cache"]
    check_game(game)
    prefix = os.path.join(cache, view, game)
    if os.path.isfile(prefix + ".json") and os.path.isfile(prefix + ".npz"):
        return prefix + ".json"
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    env = dict(os.environ, MTAH_DATA_DIR=root)
    command = [sys.executable, os.path.abspath(__file__), "--child", game, prefix,
               os.path.join(cache, view, "engine_data")]
    with open(prefix + ".log", "w") as log:
        code = subprocess.call(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if code != 0:
        raise RuntimeError("child failed on {} {}, see {}.log".format(game, view, prefix))
    return prefix + ".json"


def _score(gt, pred):
    """(spellF1, macroF1) of tools/score_game.py."""
    import score_game as sg

    result, _, _ = sg.score_codes(gt, pred)
    return float(result["spell_f1_mean"]), float(result["macro_f1"])


def _means(rows, key):
    """{"all", "metrica", "dfl"}: the mean of rows[game][key] per group."""
    out = {}
    for name, prefix in (("all", ""), ("metrica", "METRICA-"), ("dfl", "DFL-")):
        vals = [rows[g][key] for g in sorted(rows) if g.startswith(prefix)]
        out[name] = float(np.mean(vals)) if vals else None
    return out


def main(argv=None):
    """CLI entry point. Return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--child", nargs=3, metavar=("GAME", "PREFIX", "DATA_CACHE"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--out", default=None, help="write the full summary JSON here")
    parser.add_argument("--refs-root", default=None)
    parser.add_argument("--sample-root", default=None)
    parser.add_argument("--mtah-root", default=None)
    parser.add_argument("--bar-data", default=DEFAULT_BAR_DATA)
    args = parser.parse_args(argv)
    if args.child:
        game, prefix, data_cache = args.child
        check_game(game)
        from engine import data
        data.DEFAULT_CACHE_DIR = data_cache
        return child(game, prefix)

    import external_sweep as es
    import score_game as sg

    refs_root = os.path.abspath(os.path.expanduser(args.refs_root or es.DEFAULT_REFS_ROOT))
    sample_root = os.path.abspath(os.path.expanduser(args.sample_root or es.DEFAULT_SAMPLE_ROOT))
    mtah_root = os.path.abspath(os.path.expanduser(args.mtah_root or es.DEFAULT_MTAH_ROOT))
    cache = os.path.abspath(args.cache)
    jobs = [{"game": BAR, "view": "bar", "cache": cache,
             "root": os.path.abspath(os.path.expanduser(args.bar_data))}]
    for view in VIEWS:
        for game in OPEN_GAMES:
            jobs.append({"game": game, "view": view, "cache": cache,
                         "root": es.data_root(game, view, refs_root, sample_root, mtah_root)})
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        paths = list(pool.map(run_child, jobs))
    summaries = {}
    for job, path in zip(jobs, paths):
        with open(path) as handle:
            summaries[(job["view"], job["game"])] = json.load(handle)

    report = {"d1": {}, "d2": {}, "d3": {}}
    # D1: every match and view.
    for (view, game), summ in sorted(summaries.items()):
        row = dict(summ["d1"], d_runs_viterbi=summ["d_runs_viterbi"],
                   d_runs_final=summ["d_runs_final"],
                   stoppage_min_frames=summ["stoppage_min_frames"])
        report["d1"].setdefault(view, {})[game] = row
    bar = summaries[("bar", BAR)]["d1"]
    report["d1_verdict"] = {"bar_share": bar["d_share"], "threshold": D1_THRESHOLD,
                            "fix_warranted": bar["d_share"] >= D1_THRESHOLD}

    # D2: the oracle decomposition on the externals.
    refs = {g: es.prepare_reference(g, refs_root, sample_root, cache)[0] for g in OPEN_GAMES}
    gts = {g: sg.read_codes(refs[g]) for g in OPEN_GAMES}
    for view in VIEWS:
        rows = {}
        for game in OPEN_GAMES:
            gt = np.asarray(gts[game], dtype=np.int8)
            ours = np.load(os.path.join(cache, view, game + ".npz"))["labels"]
            if ours.shape != gt.shape:
                raise SystemExit("{}: {} frames, reference {}".format(game, ours.size, gt.size))
            import possession_metrics as pm
            segments = pm.valid_runs(gt)
            row = {}
            for name, pred in (("ours", ours), ("oracle_team", oracle_team(ours, gt)),
                               ("oracle_dead", oracle_dead(ours, gt, segments))):
                row[name + "_spell"], row[name + "_macro"] = _score(gt, pred)
            rows[game] = row
        means = {key: _means(rows, key) for key in sorted(next(iter(rows.values())))}
        report["d2"][view] = {"games": rows, "means": means}

    # D3: ball process on the raw DFL views (reference split) and every match (our split).
    d3 = {}
    for view, game in [("bar", BAR)] + [("raw", g) for g in OPEN_GAMES]:
        arrays = np.load(os.path.join(cache, view, game + ".npz"))
        summ = summaries[(view, game)]
        labels = arrays["labels"]
        visible = arrays["visible"]
        stat = arrays["stationary"] & visible
        speed = arrays["speed"].astype(np.float64)
        slow = visible & np.isfinite(speed) & (speed < summ["stationary_speed_mps"])
        graded = labels != X
        ours_d = graded & (labels == D)
        ours_alive = graded & np.isin(labels, ALIVE)
        hours = summ["graded_frames"] / summ["fps"] / 3600.0
        row = {"graded_hours": hours,
               "ours_d_share": share(ours_d, graded),
               "stationary_in_our_d": share(stat, ours_d),
               "stationary_in_our_alive": share(stat, ours_alive),
               "slow_in_our_d": share(slow, ours_d),
               "slow_in_our_alive": share(slow, ours_alive),
               "visible_in_our_d": share(visible, ours_d),
               "visible_in_our_alive": share(visible, ours_alive),
               "onsets_per_hour": {k: v / hours for k, v in sorted(summ["onsets"].items())},
               "entries_per_hour": {k: v / hours for k, v in sorted(summ["entries"].items())},
               "exits_per_hour": {k: v / hours for k, v in sorted(summ["exits"].items())},
               "entries": summ["entries"], "exits": summ["exits"], "onsets": summ["onsets"]}
        if game.startswith("DFL-"):
            gt = np.asarray(gts[game], dtype=np.int8)
            ref_d, ref_alive = gt == D, np.isin(gt, ALIVE)
            row.update({"ref_d_share": share(ref_d, gt != X),
                        "stationary_in_ref_d": share(stat, ref_d),
                        "stationary_in_ref_alive": share(stat, ref_alive),
                        "slow_in_ref_d": share(slow, ref_d),
                        "slow_in_ref_alive": share(slow, ref_alive),
                        "visible_in_ref_d": share(visible, ref_d),
                        "visible_in_ref_alive": share(visible, ref_alive)})
        d3[game] = row
    report["d3"] = d3
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
