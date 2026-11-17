"""Leave-one-match-out evaluation of the learned spike. Not part of the submission.

    python -m tools.learned_spike.lomo --cache DIR --out DIR --test G [--steps 1500] [--seed 0]

One fold: the test match is held out. One other match is the validation match (val_match()).
The model trains on the remaining matches, all views (raw, matched, sigma95, degraded), with
random 2-minute crops and a random symmetric copy per crop. The validation match (raw view)
picks the training checkpoint (lowest validation loss) and the two Viterbi switch costs
(highest spellF1 against its 2 s reference). The test match is touched once, at the end: the
model predicts its raw view, the decoder uses the chosen costs, and the tool writes
OUT/<test>/pred.csv, OUT/<test>/probs.npz (5 fps probabilities) and OUT/<test>/fold.json.

Loss: cross entropy on the 2 s controlled labels, X frames masked, frames within BOUNDARY_BINS
bins of a label change weighted BOUNDARY_WEIGHT times. Deterministic for a given seed and
thread count. Python 3.9.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import games  # noqa: E402
import possession_metrics as pm  # noqa: E402
import score_game as sg  # noqa: E402
from engine import data  # noqa: E402
from tools.learned_spike import features, model  # noqa: E402

OPEN_GAMES = ("METRICA-S1", "METRICA-S2", "METRICA-S3",
              "DFL-J03WMX", "DFL-J03WN1", "DFL-J03WOH", "DFL-J03WOY", "DFL-J03WPY")
FORBIDDEN = {"DFL-J03WQQ", "DFL-J03WR9", games.GAME}
TRAIN_VIEWS = ("raw", "matched", "sigma95", "degraded")
CROP_BINS = 600            # 2 minutes at 5 fps
BATCH = 16
LR = 2e-3
BOUNDARY_BINS = 5          # +-1 s at 5 fps
BOUNDARY_WEIGHT = 3.0
EVAL_EVERY = 200
INPUT_NOISE = 0.02
COST_HA = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
COST_AD = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


def provider(game):
    return game.split("-")[0]


def val_match(test, pool):
    """The validation match of a fold: the first match of the other provider after the test
    match in the cyclic order of `pool`. The costs come from another provider, as they would
    for the competition match."""
    k = pool.index(test)
    order = pool[k + 1:] + pool[:k]
    for game in order:
        if provider(game) != provider(test):
            return game
    return order[0]


def load_game(cache, game, views):
    lab = np.load(os.path.join(cache, "labels", game + ".npz"))
    feats = {v: np.load(os.path.join(cache, "feats", v, game + ".npz"))["F"] for v in views}
    return {"feats": feats, "raw": lab["raw"], "ctl2": lab["ctl2"],
            "bins_ctl2": features.bin_labels(lab["ctl2"])}


def boundary_weights(codes):
    """Weight per bin: BOUNDARY_WEIGHT within BOUNDARY_BINS of a change of label, else 1."""
    change = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    w = np.ones(len(codes), dtype=np.float32)
    near = np.zeros(len(codes) + 1, dtype=np.int64)
    for c in change:
        near[max(0, c - BOUNDARY_BINS)] += 1
        near[min(len(codes), c + BOUNDARY_BINS)] -= 1
    w[np.cumsum(near)[:-1] > 0] = BOUNDARY_WEIGHT
    w[codes == 3] = 0.0
    return w


def sample_batch(rng, train, views):
    xs, ys, ws = [], [], []
    for _ in range(BATCH):
        game = train[rng.randint(len(train))]
        item = GAMES_DATA[game]
        view = views[rng.randint(len(views))]
        feats = item["feats"][view]
        n = feats.shape[0]
        start = rng.randint(0, max(1, n - CROP_BINS))
        x = feats[start:start + CROP_BINS]
        y = item["bins_ctl2"][start:start + CROP_BINS].astype(np.int64)
        swap, mirror = bool(rng.randint(2)), bool(rng.randint(2))
        x = features.transform(x, swap, mirror)
        if swap:
            y = model.swap_labels(y)
        w = boundary_weights(y)
        y = np.where(y == 3, 0, y)
        xs.append(x)
        ys.append(y)
        ws.append(w)
    x = torch.from_numpy(np.stack(xs)).float()
    x = x + INPUT_NOISE * torch.from_numpy(rng.standard_normal(x.shape).astype(np.float32))
    return x, torch.from_numpy(np.stack(ys)), torch.from_numpy(np.stack(ws))


@torch.no_grad()
def val_loss(net, item):
    net.eval()
    x = torch.from_numpy(item["feats"]["raw"])[None].float()
    y = item["bins_ctl2"].astype(np.int64)
    w = torch.from_numpy(boundary_weights(y))
    logits = net(x)[0]
    ce = torch.nn.functional.cross_entropy(logits, torch.from_numpy(np.where(y == 3, 0, y)),
                                           reduction="none")
    return float((ce * w).sum() / w.sum())


def train(train_games, val_game, steps, seed):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    net = model.Net()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps,
                                                pct_start=0.1)
    best = (float("inf"), None, -1)
    history = []
    for step in range(1, steps + 1):
        net.train()
        x, y, w = sample_batch(rng, train_games, TRAIN_VIEWS)
        logits = net(x)
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, model.N_CLASSES),
                                               y.reshape(-1), reduction="none")
        loss = (ce * w.reshape(-1)).sum() / w.sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % EVAL_EVERY == 0 or step == steps:
            vl = val_loss(net, GAMES_DATA[val_game])
            history.append({"step": step, "train": float(loss.detach()), "val": vl})
            print("step {} train {:.4f} val {:.4f}".format(step, float(loss.detach()), vl),
                  file=sys.stderr, flush=True)
            if vl < best[0]:
                best = (vl, {k: v.clone() for k, v in net.state_dict().items()}, step)
    net.load_state_dict(best[1])
    return net, best[2], history


def frame_probs(net, game):
    item = GAMES_DATA[game]
    p5 = model.predict_symmetric(net, item["feats"]["raw"])
    return p5, model.upsample(p5, len(item["raw"]))


def spell_f1(ref_codes, pred_codes):
    return float(sg.score_codes(ref_codes, pred_codes)[0]["spell_f1_mean"])


def choose_costs(probs, game):
    """(cost_ha, cost_ad, table) with the highest spellF1 on the 2 s reference of `game`."""
    blocks = data.graded_blocks(game)
    ref = GAMES_DATA[game]["ctl2"]
    table = []
    for cha in COST_HA:
        for cad in COST_AD:
            pred = model.decode(probs, blocks, cha, cad)
            table.append((spell_f1(ref, pred), cha, cad))
    best = max(table, key=lambda r: (r[0], -r[1], -r[2]))
    return best[1], best[2], table


def write_csv(path, codes):
    labels = np.asarray(pm.LABELS)[codes]
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        for i, lab in enumerate(labels):
            handle.write("{},{}\n".format(i, lab))


GAMES_DATA = {}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--games", nargs="+", default=list(OPEN_GAMES))
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    pool = list(args.games)
    for game in pool:
        if game in FORBIDDEN:
            raise SystemExit("{} is forbidden".format(game))
    test = args.test
    val = val_match(test, pool)
    train_games = [g for g in pool if g not in (test, val)]
    for game in pool:
        GAMES_DATA[game] = load_game(args.cache, game, TRAIN_VIEWS)
    t0 = time.time()
    net, best_step, history = train(train_games, val, args.steps, args.seed)
    t_train = time.time() - t0
    vp5, vprobs = frame_probs(net, val)
    cha, cad, table = choose_costs(vprobs, val)
    t1 = time.time()
    p5, probs = frame_probs(net, test)
    pred = model.decode(probs, data.graded_blocks(test), cha, cad)
    t_pred = time.time() - t1
    out = os.path.join(args.out, test)
    os.makedirs(out, exist_ok=True)
    write_csv(os.path.join(out, "pred.csv"), pred)
    np.savez_compressed(os.path.join(out, "probs.npz"), p5=p5.astype(np.float32))
    torch.save(net.state_dict(), os.path.join(out, "net.pt"))
    item = GAMES_DATA[test]
    record = {
        "test": test, "val": val, "train": train_games, "steps": args.steps, "seed": args.seed,
        "best_step": best_step, "history": history, "cost_ha": cha, "cost_ad": cad,
        "val_cost_table": [list(r) for r in table],
        "val_spell_f1_ctl2": max(r[0] for r in table),
        "test_spell_f1_ctl2": spell_f1(item["ctl2"], pred),
        "test_spell_f1_raw": spell_f1(item["raw"], pred),
        "train_seconds": t_train, "predict_decode_seconds": t_pred,
    }
    with open(os.path.join(out, "fold.json"), "w") as handle:
        json.dump(record, handle, indent=1, sort_keys=True)
    print(json.dumps({k: record[k] for k in ("test", "val", "best_step", "cost_ha", "cost_ad",
                                             "val_spell_f1_ctl2", "test_spell_f1_ctl2",
                                             "test_spell_f1_raw", "train_seconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
