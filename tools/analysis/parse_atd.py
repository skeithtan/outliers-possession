"""Parse the tactical ATD into numpy arrays and cache to /tmp. Not part of the repo."""
import os, sys, time
import numpy as np
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "cache", "atd_tactical.npz")
sys.path.insert(0, ROOT)
from games import asset_paths

OUT = CACHE

def main():
    p = asset_paths()
    root = ET.parse(p["atd_meta"]).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    names = [t.findtext("Name") for t in root.iter("Team")]
    slots = [pl.get("teamId") for pl in root.iter("Player")]
    team_of_slot = np.array([0 if s == team_ids[0] else 1 for s in slots], dtype=np.int8)
    n_slots = len(slots)
    print("teams", names, "slots", n_slots)

    N = 199319
    px = np.full((N, n_slots), np.nan, np.float32)
    py = np.full((N, n_slots), np.nan, np.float32)
    ps = np.full((N, n_slots), np.nan, np.float32)
    bx = np.full(N, np.nan, np.float32)
    by = np.full(N, np.nan, np.float32)
    bs = np.full(N, np.nan, np.float32)
    t0 = time.time()
    with open(p["atd_raw"]) as fh:
        for line in fh:
            fid, players, ball = line.rstrip("\n").split(":")
            i = int(fid) - 1
            vals = np.array(players.replace(";", ",").split(","), dtype=np.float32).reshape(-1, 3)
            px[i, :vals.shape[0]] = vals[:, 0]
            py[i, :vals.shape[0]] = vals[:, 1]
            ps[i, :vals.shape[0]] = vals[:, 2]
            b = ball.split(",")
            bx[i], by[i], bs[i] = float(b[0]), float(b[1]), float(b[2])
    print("parsed in %.1fs" % (time.time() - t0))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, px=px, py=py, ps=ps, bx=bx, by=by, bs=bs, team_of_slot=team_of_slot)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
