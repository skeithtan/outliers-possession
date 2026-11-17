"""r2 follow-up: components T, H, K, G of the restart study, alone and stacked, on engine C6.

    python tools/analysis/r2_restart_combo.py [--write]

T  throw-in seen as out: an observed `out` onset whose first frame beyond the line has no visible
   frame before it (the crossing was not seen), lies beyond a TOUCHLINE (y outside [0, W], x inside
   [0, L]) and moves inward (y over the next 4 frames moves toward the pitch) is dropped: the ball
   is being thrown in. Goal-line cases are kept. deadball.dead_onsets is monkeypatched and
   possession.prepare() rerun, so the restart priors see the same onsets.
H  chain confirmation: a candidate is also confirmed if the next players-confirmed candidate starts
   within CHAIN = 87 frames (median gap between kick candidates in alive play).
K  DELTA_RESTART_KICK = KICK_CENTRE (median offset from _rise_start to the centre of the 24-frame
   acceleration bump, argmax of v(t+12) - v(t-12) over the rise, all graded kick candidates).
G  goal kickoff seen far from the mark: after a GOALS window end inside a stoppage, the first
   visible ball frame farther than 9.15 m (centre-circle radius, Law 8) from the centre mark is a
   confirmed restart candidate of kind `kickoff_seen` (no offset); team = nearest player within
   REAPPEAR_TEAM_M, else None. A ball first seen inside the circle is the kick itself and already
   gives a reappear candidate.
--write writes evaluation/results/rx_b_<R2_PREFIX>_<combo>.csv for T, TH, THK, THKG (the
others are scored, then removed). R2_PREFIX defaults to c7 (the engine on disk). Deterministic; reads the public GT only.
"""

import dataclasses
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_restart_common as c  # noqa: E402
from engine import deadball, possession  # noqa: E402
from engine.config import Config  # noqa: E402

CHAIN = 87
CIRCLE_M = 9.15
REF_MD5 = {"c6": "cb7392a5a076d3c47698c2f925c09fb9", "c7": "ff4ecef5b8720ccb6ec0a79e5af90c1d"}
PREFIX = os.environ.get("R2_PREFIX", "c7")
WRITE = ("T", "TH", "THK", "THKG")


# -- T -------------------------------------------------------------------------------------------
def throwin_mask_rows(onsets, ball, pitch):
    """Index labels of `out` onset rows that T drops, with (frame, side, inward dy)."""
    length, width = pitch
    x, y, vis = ball["x"], ball["y"], ball["visible"]
    drop = []
    for idx, row in onsets.iterrows():
        f = int(row["frame"])
        if row["mechanism"] != "out" or not vis[f] or (f > 0 and vis[f - 1]):
            continue
        if not (0.0 <= x[f] <= length) or (0.0 <= y[f] <= width):
            continue  # not beyond a touchline (goal line or inside): keep
        j = f + 4
        while j > f and not vis[j]:
            j -= 1
        dy = y[j] - y[f] if j > f else ball["vy"][f]
        inward = dy > 0 if y[f] < 0 else dy < 0
        if inward:
            drop.append(idx)
    return drop


def patched_prepare(cfg):
    """possession.prepare with deadball.dead_onsets filtered by T (this process only)."""
    orig = deadball.dead_onsets

    def filtered(ball, feats, cfg=None, pitch=None, events=None):
        table = orig(ball, feats, cfg, pitch=pitch, events=events)
        drop = throwin_mask_rows(table, ball, c.data.pitch_m())
        return table.drop(index=drop).reset_index(drop=True)
    deadball.dead_onsets = filtered
    try:
        return possession.prepare(cfg)
    finally:
        deadball.dead_onsets = orig


# -- H, K, G --------------------------------------------------------------------------------------
def chain_confirmed(table):
    frames = table["frame"].to_numpy()
    cur = table["confirmed"].to_numpy().copy()
    conf_frames = frames[cur]
    out = cur.copy()
    for i in range(len(frames)):
        if not out[i]:
            nxt = conf_frames[conf_frames > frames[i]]
            out[i] = bool(nxt.size and nxt[0] - frames[i] <= CHAIN)
    return out


def kick_centre(inp):
    """(median, p10, p50, p90, per-frame dict) of the bump-centre offset over graded kicks."""
    b = inp["ball"]
    v, rid = b["speed"], b["run_id"]
    n = v.shape[0]
    table = inp["restarts"]
    offs = {}
    for f in table[table["kind"] == "kick"]["frame"].to_numpy():
        f = int(f)
        if not c.graded(f, inp["blocks"]):
            continue
        p = max(c.peak_end(v, f), f + 1)
        best, at = -np.inf, None
        for t in range(f, p + 1):
            if t - 12 < 0 or t + 12 >= n or not (rid[t - 12] == rid[f] == rid[t + 12]):
                continue
            a = v[t + 12] - v[t - 12]
            if np.isfinite(a) and a > best:
                best, at = a, t
        if at is not None:
            offs[f] = at - f
    vals = np.array(list(offs.values()), dtype=float)
    p10, p50, p90 = np.percentile(vals, [10, 50, 90])
    return int(round(p50)), p10, p50, p90, offs


def goal_rows(inp, cfg, stoppage):
    """Extra `kickoff_seen` candidate rows for component G."""
    b, feats, tg = inp["ball"], inp["feats"], inp["tagging"]
    length, width = c.data.pitch_m()
    ends = sorted(int(e) for e in tg[tg["code"] == "GOALS"]["end_frame"])
    rows = []
    for end in ends:
        if not stoppage[end]:
            continue
        t = end
        n = b["visible"].shape[0]
        while t < n and stoppage[t]:
            if b["visible"][t]:
                d = np.hypot(b["x"][t] - length / 2, b["y"][t] - width / 2)
                if d > CIRCLE_M:
                    dm = feats["dmin"][t]
                    team = None
                    if np.isfinite(dm) and dm <= cfg.REAPPEAR_TEAM_M:
                        team = deadball.TEAM_LETTER.get(int(feats["nearest_team"][t]))
                    rows.append((t, team, "kickoff_seen", 1.0, True, end, round(float(d), 1)))
                break  # first visible frame decides (inside the circle: the reappear stands)
            t += 1
    return rows


def with_goal_candidates(inp, cfg, confirmed):
    """inputs copy with G rows appended to the restart table (confirmed column applied)."""
    table = inp["restarts"].copy()
    table["confirmed"] = np.asarray(confirmed, dtype=bool)
    lab0, parts0 = c.run_with_confirmed(inp, cfg, confirmed)
    rows = goal_rows(inp, cfg, parts0["stoppage"])
    extra = pd.DataFrame([r[:5] for r in rows], columns=deadball.RESTART_COLUMNS)
    table = pd.concat([table, extra], ignore_index=True)
    table = table.sort_values(["frame", "kind"], kind="mergesort").reset_index(drop=True)
    table = table.astype({"frame": np.int64, "confidence": np.float64, "confirmed": bool})
    table["team"] = table["team"].astype(object).where(table["team"].notna(), None)
    alt = dict(inp)
    alt["restarts"] = table
    return alt, rows


# -- reporting -------------------------------------------------------------------------------------
def stretches(a, b):
    """[(start, stop, from, to)] runs where label vectors differ."""
    diff = np.asarray(a) != np.asarray(b)
    from engine.tracks import _runs
    s, e = _runs(diff)
    return [(int(x), int(y), str(a[x]), str(b[x])) for x, y in zip(s, e)]


def report(name, lab, base_lab, write):
    import sanity
    pairs, missed, spur, res = c.public_report(lab)
    pub = {g[0]: p[0] - g[0] for g, p in pairs}
    if write:
        path, md5, st = c.write_and_sanity(lab, "rx_b_%s_%s" % (PREFIX, name))
        if name not in WRITE:
            os.remove(path)
    else:
        st, md5 = sanity.sanity_stats(lab), ""
    h = st["halves"]
    ch = stretches(base_lab, lab)
    row = {"combo": name, "md5": md5, "spellF1": round(res["spell_f1_mean"], 4),
           "macroF1": round(res["macro_f1"], 4), "matched": len(pairs),
           "lat": {g: pub.get(g) for g in c.PUBLIC_RESTARTS},
           "poss": st["total"]["possessions"], "D1": round(100 * h[1]["d_share"], 2),
           "D2": round(100 * h[2]["d_share"], 2), "Dtot": round(100 * st["total"]["d_share"], 2),
           "Dseg": st["total"]["d_segments"], "violations": st["violations"],
           "changed_frames": int(sum(e - s for s, e, _, _ in ch)), "n_stretches": len(ch)}
    print(json.dumps(row, default=int))
    return row, ch


def main(write=False):
    cfg = Config.default()
    base = c.load(cfg, tag="base")
    tin = c.load(cfg, tag="T", prepare=patched_prepare)
    k_med, p10, p50, p90, offs = kick_centre(base)
    print("K: rise_start -> bump centre over %d graded kicks: p10/p50/p90 %.1f/%.1f/%.1f; "
          "16393 -> %s, 18355 -> %s; DELTA_RESTART_KICK = %d" % (
              len(offs), p10, p50, p90, offs.get(16393), offs.get(18355), k_med))
    cfg_k = dataclasses.replace(cfg, DELTA_RESTART_KICK=k_med)

    # T: dropped onsets
    on_b, on_t = base["onsets"], tin["onsets"]
    key = lambda t: set(zip(t["frame"].astype(int), t["mechanism"]))  # noqa: E731
    dropped = sorted(key(on_b) - key(on_t))
    tg = base["tagging"]
    sp = tg[tg["code"] == "SET PIECES"]
    print("T: %d onsets dropped (base %d -> %d):" % (len(dropped), len(on_b), len(on_t)))
    dlist = []
    for f, m in dropped:
        r = sp[(sp["start_frame"] - 125 <= f) & (f < sp["end_frame"] + 125)]
        dlist.append(f)
        print("   %d %s graded=%s tag=%s" % (f, m, c.graded(f, base["blocks"]),
                                             r["type"].iloc[0] if len(r) else "-"))
    dcall = os.path.join(os.path.dirname(c.SCRATCH), "c_dcall", "out",
                         "reappear_outside_outs.csv")
    if os.path.exists(dcall):
        dc = pd.read_csv(dcall)
        dct = set(dc[(dc["side"] == "touchline") & dc["inward"]]["frame"].astype(int))
        print("   cross-check with c_dcall touchline+inward (%d): both %s; only c_dcall %s; "
              "only T %s" % (len(dct), sorted(dct & set(dlist)), sorted(dct - set(dlist)),
                             sorted(set(dlist) - dct)))

    lab0, _ = c.run_with_confirmed(base, cfg, base["restarts"]["confirmed"])
    rows, out = [], {}

    def run(name, inp, cf, H=False, G=False):
        conf = chain_confirmed(inp["restarts"]) if H else inp["restarts"]["confirmed"].to_numpy()
        if G:
            alt, grows = with_goal_candidates(inp, cf, conf)
            lab = possession.run(cf, inputs=alt)
            print("   G rows: %s" % grows)
        else:
            lab, _ = c.run_with_confirmed(inp, cf, conf)
        row, ch = report(name, lab, lab0, write)
        rows.append(row)
        out[name] = ch
        return lab

    run(PREFIX, base, cfg)
    assert rows[0]["md5"] in ("", REF_MD5[PREFIX]), rows[0]["md5"]
    run("T", tin, cfg)
    run("H", base, cfg, H=True)
    run("K", base, cfg_k)
    run("G", base, cfg, G=True)
    run("TH", tin, cfg, H=True)
    run("THK", tin, cfg_k, H=True)
    run("THKG", tin, cfg_k, H=True, G=True)
    for name in ("T", "G"):
        print("label change stretches %s vs base (%d):" % (name, len(out[name])))
        for s in out[name]:
            print("   %s" % (s,))
    with open(os.path.join(c.SCRATCH, "combo_%s.json" % PREFIX), "w") as fh:
        json.dump({"rows": rows, "stretches": out, "kick_centre": [k_med, p10, p50, p90],
                   "dropped": dlist}, fh, indent=1, default=int)


if __name__ == "__main__":
    main(write="--write" in sys.argv)
