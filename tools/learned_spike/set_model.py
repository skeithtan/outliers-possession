"""Round 8 arm S: the set network over player and ball tokens. Not part of the submission.

SetNet: per 5 fps step, a linear embedding of the 23 tokens of set_tokens.py to WIDTH, LAYERS
pre-norm self-attention layers over the tokens of the step (HEADS heads, absent player tokens
masked; the ball token is always attended), a final layer norm, then masked mean pooling over
the home ("own") tokens, over the away tokens, and the ball token: three vectors concatenated
(3 x WIDTH). Then a two-layer bidirectional GRU over time (HIDDEN units each way, as the
network of the shipped engine) and a linear head on three classes (H, A, D).

predict_symmetric() averages the class probabilities over the four symmetric copies of the
tokens (set_tokens.transform()); a swapped copy has its H and A columns exchanged back.

Python 3.9, torch.
"""

import numpy as np
import torch
from torch import nn

from tools.learned_spike import set_tokens as st

N_CLASSES = 3
WIDTH = 64
HEADS = 4
LAYERS = 2
HIDDEN = 64
FFN = 128
SYMMETRIES = ((False, False), (False, True), (True, False), (True, True))


class SetNet(nn.Module):
    def __init__(self, n_cols=st.N_COLS, width=WIDTH, heads=HEADS, layers=LAYERS,
                 hidden=HIDDEN, ffn=FFN):
        super().__init__()
        self.embed = nn.Linear(n_cols, width)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, dim_feedforward=ffn, dropout=0.0,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.gru = nn.GRU(3 * width, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.head = nn.Linear(2 * hidden, N_CLASSES)

    def encode(self, x):
        """(S, 23, C) tokens of S steps -> (S, 3 * WIDTH)."""
        present = x[..., st.PRESENT_COL] > 0
        present[:, st.BALL] = True
        pad = ~present
        h = self.embed(x)
        for blk in self.blocks:
            # Masked tokens never serve as keys; their own rows are excluded from the pooling.
            h = blk(h, src_key_padding_mask=pad)
        h = self.norm(h)
        team = x[..., st.TEAM_COL]
        own = ((team > 0.5) & present).to(h.dtype)
        oth = ((team < -0.5) & present).to(h.dtype)

        def pool(m):
            return (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1.0)
        return torch.cat([pool(own), pool(oth), h[:, st.BALL]], dim=-1)

    def forward(self, tok, chunk=None):
        """(B, T, 23, C) -> (B, T, 3) logits. chunk: encode at most this many steps at once."""
        b, t, n, c = tok.shape
        x = tok.reshape(b * t, n, c)
        if chunk is None or x.shape[0] <= chunk:
            z = self.encode(x)
        else:
            z = torch.cat([self.encode(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)])
        z, _ = self.gru(z.reshape(b, t, -1))
        return self.head(z)


@torch.no_grad()
def predict_symmetric(net, tok, chunk=4096, device="cpu"):
    """float64 (bins, 3) mean class probabilities over the four symmetric copies."""
    net.eval()
    total = np.zeros((tok.shape[0], N_CLASSES))
    for swap, mirror in SYMMETRIES:
        x = torch.from_numpy(st.transform(np.asarray(tok, dtype=np.float32), swap, mirror))
        p = torch.softmax(net(x[None].to(device), chunk=chunk)[0], dim=-1)
        p = p.double().cpu().numpy()
        if swap:
            p = p[:, [1, 0, 2]]
        total += p
    return total / len(SYMMETRIES)
