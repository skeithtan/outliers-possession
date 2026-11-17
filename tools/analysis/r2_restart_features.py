"""r2 task 1: characterise the restart confirmation signal on the full match (no ground truth).

    python tools/analysis/r2_restart_features.py

Writes <SCRATCH>/features.csv (one row per graded restart candidate, with its class) and
<SCRATCH>/label_exit_features.csv (one row per D->H/A change of the current output), and prints
p10/p50/p90 per class. Deterministic.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_restart_common as c  # noqa: E402
from engine.config import Config  # noqa: E402

COLS = ["t_all", "t_near", "disp1", "disp2", "disp4", "disp6", "peak2", "sp0", "dv9", "n5",
        "t_all_from_peak", "rest_after"]


def main():
    cfg = Config.default()
    inp = c.load(cfg)
    cand, dead, rl, iv = c.classify(inp, cfg)
    feat = c.features(inp, cand)
    table = pd.concat([cand.reset_index(drop=True), feat.drop(columns=["frame"])], axis=1)
    table.to_csv(os.path.join(c.SCRATCH, "features.csv"), index=False)
    labels, _ = c.run_with_confirmed(inp, cfg, inp["restarts"]["confirmed"])
    ex = c.label_exits(labels, inp["blocks"])
    lf = c.features(inp, pd.DataFrame({"frame": [f for f, _ in ex]}))
    lf.to_csv(os.path.join(c.SCRATCH, "label_exit_features.csv"), index=False)

    print("state-machine exits:", int((table["cls"] == "exit").sum()),
          " label D->H/A changes:", len(ex))
    print(table.groupby(["cls", "kind"]).size().to_string())
    groups = [("exit", table[table["cls"] == "exit"]),
              ("confirm_rej", table[table["cls"] == "confirm_rej"]),
              ("other (alive + not_armed)", table[table["cls"].isin(["alive", "not_armed"])]),
              ("label exits", lf)]
    for name, g in groups:
        print("\n== %s (n=%d)" % (name, len(g)))
        for col in COLS:
            print("  %-16s %s" % (col, c.pct(g[col])))
        print("  vis %.2f  sp_tag %.2f  sp_tag_tail %.2f" % (
            g["vis"].mean(), g["sp_tag"].mean(), g["sp_tag_tail"].mean()))
    # re-play vs late-restart split of the confirmation rejects
    rej = table[table["cls"] == "confirm_rej"].copy()
    replay = (rej["disp6"] < 10) & ~(rej["t_all"] <= 150)
    print("\nconfirm_rej: n=%d; replay-like (disp6 < 10 m, players slow > 6 s): %d; "
          "ball moves >= 10 m in 6 s: %d; ball invisible at the candidate: %d" % (
              len(rej), int(replay.sum()), int((rej["disp6"] >= 10).sum()),
              int((~rej["vis"]).sum())))
    pd.set_option("display.width", 250)
    show = ["frame", "kind", "team", "entry", "t_all", "t_near", "disp1", "disp2", "disp4",
            "disp6", "peak2", "sp0", "dv9", "n5", "vis", "sp_tag_tail", "sp_type",
            "t_all_from_peak", "rest_after"]
    print(rej[show].to_string(index=False))


if __name__ == "__main__":
    main()
