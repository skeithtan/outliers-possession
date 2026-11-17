import os, sys, json, dataclasses
import numpy as np
sys.path.insert(0, sys.argv[1])
from tools.learned_spike import features, build
from engine import possession, config, data
CA, EXT = sys.argv[2], sys.argv[3]
DEV = ["METRICA-S1","METRICA-S2","METRICA-S3","DFL-J03WMX","DFL-J03WN1","DFL-J03WOH","DFL-J03WOY","DFL-J03WPY"]
out = {}
for game in DEV + ["BAR-ATH"]:
    if game == "BAR-ATH":
        tac = possession.engine_tactical(None, game)
    else:
        tac = build.load_view(game, "raw", os.path.join(EXT, "views/raw/engine_data"))
    length, width = data.pitch_m(game)
    px, py, bx, by = features.oriented_arrays(tac, game)
    team = np.asarray(tac["team_of_slot"])
    features.STABLE_SORT = False
    f0 = features.frame_features(tac, game)
    features.STABLE_SORT = True
    f1 = features.frame_features(tac, game)
    features.STABLE_SORT = False
    # ties: recompute distances as frame_features does (ball cleaned and filled)
    from engine import tracks
    cfg = config.Config.default()
    ball = tracks.clean_ball(bx, by, (length, width), cfg, fps=features.FPS)
    vis = ball["visible"]
    bxm = features._fill_nearest(np.nan_to_num(ball["x"]), vis)
    bym = features._fill_nearest(np.nan_to_num(ball["y"]), vis)
    pxm, pym = px * length, py * width
    pvis = np.isfinite(pxm) & np.isfinite(pym)
    dist = np.where(pvis, np.hypot(pxm - bxm[:, None], pym - bym[:, None]), np.inf)
    tie = {}
    for key, t in (("H", 0), ("A", 1)):
        sub = np.sort(dist[:, team == t], axis=1)
        tie[key] = int(np.sum(np.isfinite(sub[:, 0]) & (sub[:, 0] == sub[:, 1])))
    rowdiff = int(np.any(f0 != f1, axis=1).sum())
    d0, d1 = features.downsample(f0), features.downsample(f1)
    rec = {"frames": int(len(f0)), "slots_per_team": [int((team == 0).sum()), int((team == 1).sum())],
           "tie_frames": tie, "frames_differ_stable": rowdiff,
           "bins_differ_stable": int(np.any(d0 != d1, axis=1).sum())}
    if game != "BAR-ATH":
        cached = np.load(os.path.join(CA, "feats", "raw", game + ".npz"))["F"]
        rec["default_equals_cache"] = bool(cached.shape == d0.shape and cached.tobytes() == d0.astype(cached.dtype).tobytes())
        rec["stable_equals_cache"] = bool(cached.shape == d1.shape and cached.tobytes() == d1.astype(cached.dtype).tobytes())
    out[game] = rec
    print(game, json.dumps(rec), flush=True)
json.dump(out, open(sys.argv[4], "w"), indent=1)
