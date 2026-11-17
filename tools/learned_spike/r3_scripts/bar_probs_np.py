"""Stage 1 script (LEARNED-R3-SCOPING Section 4): BAR-ATH probabilities of a weights file and its
M codes. The codes go through learned.apply()'s decoder with the match-structure step of
Config.LEARNED_STRUCTURE (dead ball before each half's kickoff, R3_DEVIATION_STRUCTURE.md), so
this path no longer bypasses it. Args: REPO WEIGHTS OUT.npz [COST_HA COST_AD] [--no-structure]."""
import sys, numpy as np, time
sys.path.insert(0, sys.argv[1])
from engine import learned, config, possession, data
import dataclasses
args = [a for a in sys.argv[4:] if not a.startswith("--")]
structure = "--no-structure" not in sys.argv
cost_ha, cost_ad = (float(args[0]), float(args[1])) if len(args) >= 2 else (16.0, 12.0)
cfg = dataclasses.replace(config.Config(), LEARNED_MODE="model", LEARNED_STRUCTURE=structure)
nets = learned.load_weights(sys.argv[2])
t = time.time()
tac = possession.engine_tactical(cfg, "BAR-ATH")
p5, probs = learned.frame_probabilities(nets, tac, "BAR-ATH", cfg)
blocks = data.graded_blocks("BAR-ATH")
forced = (learned.pre_kickoff_mask(len(probs), blocks, data.kickoff_frames("BAR-ATH"))
          if cfg.LEARNED_STRUCTURE else None)
codes = learned.decode_model(probs, blocks, cost_ha, cost_ad, forced)
np.savez_compressed(sys.argv[3], p5=p5, n_frames=int(tac["n_frames"]), m_codes=codes,
                    structure=bool(structure), costs=np.array([cost_ha, cost_ad]))
print("done", time.time() - t, p5.shape, "structure", structure)
