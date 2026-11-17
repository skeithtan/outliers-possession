"""H2' decision count on BAR-ATH (ANSWER-016 §5). Label-free: c23's own output and the tracking only."""
import sys, os
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), 'tools'), os.path.join(os.getcwd(), 'evaluation')]
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# player_signals and runs_of: advisor/ASK-016/gap_signals.py on branch advisor/remote (copy it next to this file)
from gap_signals import player_signals, runs_of
from engine import data, possession
from engine.config import Config
game = 'BAR-ATH'
cfg = Config.default()
inputs = possession.prepare(cfg, game)
labels = np.asarray(possession.run(cfg, inputs=inputs))
tac = data.load_tactical(game)
pitch = data.pitch_m(game); fps = float(data.fps(game))
blocks = data.graded_blocks(game)
graded = np.zeros(len(labels), dtype=bool)
for _, a, b in blocks: graded[int(a):int(b) + 1] = True
sig, _ = player_signals(tac, pitch, fps, blocks)
no_ball = ~np.isfinite(np.asarray(tac['bx'], dtype=np.float64))
LONG, STEP = int(2 * fps), 5
frames = []
for a, b in runs_of(no_ball & graded):
    if b - a > LONG: frames.extend(range(a, b, STEP))
frames = np.asarray(frames)
lab = labels[frames]; sp = sig['mean_speed'][frames]
edges = [0, 1.0, 1.6, 2.5, 3.5, np.inf]
def bands(m):
    v = sp[m & np.isfinite(sp)]; n = len(v)
    return n, [round(100 * np.mean((v >= lo) & (v < hi)), 1) for lo, hi in zip(edges[:-1], edges[1:])]
print('graded frames', int(graded.sum()), 'long-gap graded frames', int(sum(b - a for a, b in runs_of(no_ball & graded) if b - a > LONG)))
print('rows', len(frames), 'engine D share of long-gap rows', round(100 * np.mean(lab == 'D'), 1))
for name, m in [('c23 D', lab == 'D'), ('c23 H/A', np.isin(lab, ['H', 'A']))]:
    n, b = bands(m); print(name, 'n', n, 'bands <1 / 1-1.6 / 1.6-2.5 / 2.5-3.5 / >3.5 :', b, 'above 2.5:', round(b[3] + b[4], 1))
