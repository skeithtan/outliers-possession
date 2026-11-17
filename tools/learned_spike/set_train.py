"""Round 8 arm S: train the set network on one left-out provider, and predict. Not part of the submission.

    python -m tools.learned_spike.set_train train --provider metrica|dfl --seed 0 \
        --tokens TOK --labels CACHE_C_TRAIN2TJ --sc-labels PG --out OUT [--aug-k 16] \
        [--device cpu|mps] [--threads 2] [--steps 2000]
    python -m tools.learned_spike.set_train predict --net OUT/metrica/seed0/net.pt \
        --tokens TOK --games G ... --out P5DIR [--device cpu]

The recipe is AQCTJ where it applies (train.py constants and switches): STEPS 2000, BATCH 16
crops of CROP_BINS 600 bins (2 minutes at 5 fps), one-cycle Adam with peak LR 2e-3 (10 %
warm-up), gradient clip 1.0, the four symmetric copies drawn per crop, boundary weights,
target "raw" labels, mix "sc_quarter" (a SkillCorner match at weight 1/4), SkillCorner labels
from the possession groups with the dead-against-live loss only (train.sc_alive_dead_ce), DFL
from input c_train_j for training (still ball removed, half of the runs kept in view with
jitter) and from input c for validation, and the defect copies: half of the crops take one of
the aug_k copies of set_tokens.py, the other half the training view. Input noise
INPUT_NOISE on the continuous columns of the present tokens. The pool (training and
validation matches) is r3_diag.pool("no-<provider>") as r4_screen; the checkpoint of the lowest
mean validation loss (every EVAL_EVERY steps) is kept.

Outputs: OUT/<provider>/seed<s>/net.pt and model.json; predict writes <game>.npy (bins, 3)
float64 symmetric means. No PFF, no competition match (folds.check_all). Python 3.9, torch.
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
from tools.learned_spike import features, folds, r3_diag, train  # noqa: E402
from tools.learned_spike import set_model as sm  # noqa: E402
from tools.learned_spike import set_tokens as st  # noqa: E402
from tools.learned_spike.lomo import boundary_weights  # noqa: E402

NOISE_COLS = (0, 1, 2, 3, 6, 7, 8)


def log(text):
    print(time.strftime("%H:%M:%S"), text, file=sys.stderr, flush=True)


def labels_of(labels, sc_labels, game):
    base = sc_labels if folds.provider(game) == "skillcorner" else labels
    lab = np.load(os.path.join(base, "labels", game + ".npz"))
    return features.bin_labels(lab["raw"])


def load(tokens, labels, sc_labels, games, aug_k, val=False):
    out = {}
    for g in folds.check_all(games):
        bins = labels_of(labels, sc_labels, g)
        view = "val" if val else "train"
        tk = np.load(st.tok_path(tokens, view, g), mmap_mode="r")
        if tk.shape[0] != len(bins):
            raise SystemExit("{}: {} token steps, {} label bins".format(g, tk.shape[0], len(bins)))
        item = {"tok": tk, "bins": bins, "provider": folds.provider(g)}
        if not val:
            item["aug"] = [np.load(st.tok_path(tokens, "k{}".format(k), g), mmap_mode="r")
                           for k in range(aug_k)]
        out[g] = item
    return out


def sample_batch(rng, data, train_games):
    """As train.sample_batch with switches AQCTJ (mix sc_quarter, aug share 0.5, sc alive-dead)."""
    xs, ys, ws, scs = [], [], [], []
    w = np.array([train.MATCH_WEIGHTS.get(folds.provider(g), 1.0) for g in train_games])
    c = np.cumsum(w)
    for _ in range(train.BATCH):
        game = train_games[min(int(np.searchsorted(c, rng.random_sample() * c[-1],
                                                    side="right")), len(train_games) - 1)]
        item = data[game]
        if item["aug"] and rng.random_sample() < train.AUG_SHARE:
            tk = item["aug"][rng.randint(len(item["aug"]))]
        else:
            tk = item["tok"]
        n = tk.shape[0]
        start = rng.randint(0, max(1, n - train.CROP_BINS))
        x = np.asarray(tk[start:start + train.CROP_BINS], dtype=np.float32)
        y = item["bins"][start:start + train.CROP_BINS].astype(np.int64)
        swap, mirror = bool(rng.randint(2)), bool(rng.randint(2))
        x = st.transform(x, swap, mirror)
        if swap:
            y = np.where(y == 0, 1, np.where(y == 1, 0, y))
        wt = boundary_weights(y)
        y = np.where(y == 3, 0, y)
        xs.append(x)
        ys.append(y)
        ws.append(wt)
        scs.append(item["provider"] == "skillcorner")
    x = np.stack(xs)
    noise = train.INPUT_NOISE * rng.standard_normal(x.shape[:-1] + (len(NOISE_COLS),))
    present = (x[..., st.PRESENT_COL] > 0)
    present[..., st.BALL] = True
    x[..., list(NOISE_COLS)] += (noise * present[..., None]).astype(np.float32)
    return (torch.from_numpy(x), torch.from_numpy(np.stack(ys)),
            torch.from_numpy(np.stack(ws)), torch.from_numpy(np.array(scs, dtype=bool)))


@torch.no_grad()
def val_loss(net, item, device):
    net.eval()
    x = torch.from_numpy(np.asarray(item["tok"], dtype=np.float32))[None].to(device)
    y = item["bins"].astype(np.int64)
    w = torch.from_numpy(boundary_weights(y))
    logits = net(x, chunk=4096)[0].float().cpu()
    yt = torch.from_numpy(np.where(y == 3, 0, y))
    sc = torch.tensor([item["provider"] == "skillcorner"])
    ce = train.sc_alive_dead_ce(logits[None], yt[None], sc)[0]
    return float((ce * w).sum() / w.sum())


def cmd_train(args):
    torch.set_num_threads(args.threads)
    if args.device == "cpu":
        torch.use_deterministic_algorithms(True)
    spec = r3_diag.pool("no-" + args.provider)
    assert not any(folds.provider(g) == args.provider for g in spec["train"] + spec["val"])
    data = load(args.tokens, args.labels, args.sc_labels, spec["train"], args.aug_k)
    data.update(load(args.tokens, args.labels, args.sc_labels, spec["val"], 0, val=True))
    out = os.path.join(args.out, args.provider, "seed{}".format(args.seed))
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    dev = torch.device(args.device)
    net = sm.SetNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=train.LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=train.LR, total_steps=args.steps,
                                                pct_start=train.PCT_START)
    best = (float("inf"), None, -1)
    history = []
    t0 = time.time()
    t_eval = 0.0
    train_games = spec["train"]
    for step in range(1, args.steps + 1):
        net.train()
        x, y, w, sc = sample_batch(rng, data, train_games)
        logits = net(x.to(dev)).float().cpu() if args.device != "cpu" else net(x)
        ce = train.sc_alive_dead_ce(logits, y, sc).reshape(-1)
        loss = (ce * w.reshape(-1)).sum() / w.sum().clamp_min(1.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), train.CLIP)
        opt.step()
        sched.step()
        if not np.isfinite(float(loss.detach())):
            raise SystemExit("non-finite loss at step {}".format(step))
        check = step % args.eval_every == 0 or step == args.steps
        if check or step % 50 == 0 or step <= 3:
            entry = {"step": step, "train": round(float(loss.detach()), 6),
                     "elapsed": round(time.time() - t0, 2)}
            if check:
                te = time.time()
                losses = [val_loss(net, data[g], dev) for g in spec["val"]]
                t_eval += time.time() - te
                entry["val"] = [round(v, 6) for v in losses]
                entry["val_mean"] = float(np.mean(losses))
                if entry["val_mean"] < best[0]:
                    best = (entry["val_mean"],
                            {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                            step)
            history.append(entry)
            log("{} seed {} step {} train {:.4f}{} {:.1f}s".format(
                args.provider, args.seed, step, entry["train"],
                " val {:.4f}".format(entry["val_mean"]) if check else "", entry["elapsed"]))
    seconds = time.time() - t0
    torch.save(best[1], os.path.join(out, "net.pt"))
    rec = {"provider_left_out": args.provider, "seed": args.seed, "steps": args.steps,
           "best_step": best[2], "best_val_mean": best[0], "train": train_games,
           "val": spec["val"], "aug_k": args.aug_k, "device": args.device,
           "threads": args.threads, "train_seconds": round(seconds, 1),
           "eval_seconds": round(t_eval, 1),
           "seconds_per_step": round((seconds - t_eval) / args.steps, 4),
           "params": int(sum(p.numel() for p in net.parameters())),
           "tokens": os.path.abspath(args.tokens), "labels": os.path.abspath(args.labels),
           "sc_labels": os.path.abspath(args.sc_labels), "history": history,
           "torch": torch.__version__}
    with open(os.path.join(out, "model.json"), "w") as h:
        json.dump(rec, h, indent=1)
    log("done: best step {} val {:.4f}, {:.1f} s".format(best[2], best[0], seconds))
    return 0


def cmd_predict(args):
    torch.set_num_threads(args.threads)
    net = sm.SetNet()
    net.load_state_dict(torch.load(args.net, weights_only=True))
    dev = torch.device(args.device)
    net.to(dev)
    os.makedirs(args.out, exist_ok=True)
    for g in folds.check_all(args.games):
        tk = np.load(st.tok_path(args.tokens, "val", g))
        t0 = time.time()
        p = sm.predict_symmetric(net, tk, device=dev)
        np.save(os.path.join(args.out, g + ".npy"), p)
        log("{} predicted {} bins in {:.1f}s".format(g, p.shape[0], time.time() - t0))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--provider", choices=("metrica", "dfl", "skillcorner"), required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--tokens", required=True)
    t.add_argument("--labels", required=True)
    t.add_argument("--sc-labels", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--aug-k", type=int, default=16)
    t.add_argument("--steps", type=int, default=train.STEPS)
    t.add_argument("--eval-every", type=int, default=train.EVAL_EVERY)
    t.add_argument("--device", default="cpu")
    t.add_argument("--threads", type=int, default=2)
    q = sub.add_parser("predict")
    q.add_argument("--net", required=True)
    q.add_argument("--tokens", required=True)
    q.add_argument("--games", nargs="+", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--device", default="cpu")
    q.add_argument("--threads", type=int, default=2)
    args = p.parse_args(argv)
    return cmd_train(args) if args.cmd == "train" else cmd_predict(args)


if __name__ == "__main__":
    sys.exit(main())
