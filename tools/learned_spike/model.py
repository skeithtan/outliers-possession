"""The sequence model, the symmetric prediction and the Viterbi decoder. Not part of the submission.

Net: a per-step linear layer, one residual temporal convolution, a two-layer bidirectional GRU
and a linear head with three classes (H, A, D). Input: the 5 fps features of features.py. The
input width is that of the feature version (features.n_features(); net_for_version()), the
default Net() takes "stage50-v2".

predict_symmetric() averages the class probabilities over the four symmetric copies of the
input (as is, mirror, swap, swap and mirror). A swapped copy predicts with H and A exchanged,
so its H and A columns are exchanged back before the average.

viterbi() decodes 25 fps log-probabilities over the states H, A, D with a cost for a change
between H and A and a cost for a change between alive and dead.

Python 3.9. torch for the network, numpy for the decoder.
"""

import numpy as np
import torch
from torch import nn

from tools.learned_spike import features

N_CLASSES = 3
SYMMETRIES = ((False, False), (False, True), (True, False), (True, True))


class Net(nn.Module):
    def __init__(self, n_in=features.NF, hidden=64):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU())
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.act = nn.GELU()
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True,
                          dropout=0.1)
        self.head = nn.Linear(2 * hidden, N_CLASSES)

    @property
    def n_in(self):
        return int(self.inp[0].in_features)

    @property
    def feature_version(self):
        """The feature version of the input width (features.version_of_width())."""
        return features.version_of_width(self.n_in)

    def forward(self, x):
        h = self.inp(x)
        h = h + self.act(self.conv(h.transpose(1, 2)).transpose(1, 2))
        h, _ = self.gru(h)
        return self.head(h)


def net_for_version(version, hidden=64):
    """A Net whose input width is that of a feature version."""
    return Net(n_in=features.n_features(version), hidden=hidden)


def net_for_state(state, hidden=None):
    """A Net shaped for a state dictionary (input width and hidden size from inp.0.weight)."""
    h, n_in = tuple(state["inp.0.weight"].shape)
    return Net(n_in=int(n_in), hidden=int(h) if hidden is None else hidden)


def swap_labels(codes):
    """H <-> A on possession_metrics codes (D and X unchanged)."""
    out = codes.copy()
    out[codes == 0] = 1
    out[codes == 1] = 0
    return out


@torch.no_grad()
def predict_symmetric(net, feats, symmetries=SYMMETRIES):
    """float64 (bins, 3) mean class probabilities over the symmetric copies."""
    net.eval()
    total = np.zeros((feats.shape[0], N_CLASSES))
    for swap, mirror in symmetries:
        x = torch.from_numpy(np.ascontiguousarray(features.transform(feats, swap, mirror)))
        p = torch.softmax(net(x[None].float())[0], dim=-1).double().numpy()
        if swap:
            p = p[:, [1, 0, 2]]
        total += p
    return total / len(symmetries)


def upsample(probs, n_frames, bin_size=features.BIN):
    """Linear interpolation of bin probabilities (bin centres) onto every frame."""
    centres = np.arange(probs.shape[0]) * bin_size + bin_size // 2
    frames = np.arange(n_frames)
    out = np.empty((n_frames, probs.shape[1]))
    for k in range(probs.shape[1]):
        out[:, k] = np.interp(frames, centres, probs[:, k])
    return out


def viterbi(logp, cost_ha, cost_ad):
    """Best state path (int8 codes 0 H, 1 A, 2 D) for (N, 3) log-probabilities."""
    n = logp.shape[0]
    trans = np.array([[0.0, cost_ha, cost_ad],
                      [cost_ha, 0.0, cost_ad],
                      [cost_ad, cost_ad, 0.0]])
    back = np.empty((n, 3), dtype=np.int8)
    score = logp[0].copy()
    for t in range(1, n):
        cand = score[:, None] - trans
        arg = cand.argmax(axis=0)
        back[t] = arg
        score = cand[arg, (0, 1, 2)] + logp[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(frame_probs, blocks, cost_ha, cost_ad, floor=1e-4):
    """int8 codes for a match: Viterbi in each graded block, X (3) outside the blocks."""
    n = frame_probs.shape[0]
    out = np.full(n, 3, dtype=np.int8)
    logp = np.log(np.clip(frame_probs, floor, 1.0))
    for _, first, last in blocks:
        out[first:last + 1] = viterbi(logp[first:last + 1], cost_ha, cost_ad)
    return out
