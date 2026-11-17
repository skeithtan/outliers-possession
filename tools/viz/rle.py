"""Run-length encoding of label codes (site/data/SCHEMA.md, Section 1).

A run is `[start, stop, letter]` with a half-open frame range. Codes are the scorer's codes
`H=0`, `A=1`, `D=2`, `X=3`. A frame in no run decodes to `X`.
"""

import numpy as np

# Label letters in code order (SCHEMA.md Section 1, evaluation/possession_metrics.LABELS).
LETTERS = ("H", "A", "D", "X")
# Code of `X`, the label of a frame in no run (SCHEMA.md Section 1).
X_CODE = 3


def encode(codes, start=0, stop=None):
    """Maximal constant runs of `codes[start:stop]` as `[start, stop, letter]` lists.

    The frame numbers are absolute. Runs of `X` are kept, so the runs cover the whole range.
    """
    codes = np.asarray(codes)
    stop = len(codes) if stop is None else int(stop)
    start = int(start)
    window = codes[start:stop]
    if not len(window):
        return []
    breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(window)]))
    return [[start + int(bounds[i]), start + int(bounds[i + 1]), LETTERS[int(window[bounds[i]])]]
            for i in range(len(bounds) - 1)]


def decode(runs, n_frames):
    """int8 codes of length `n_frames` from runs. A frame in no run is `X`."""
    out = np.full(int(n_frames), X_CODE, dtype=np.int8)
    for start, stop, letter in runs:
        out[int(start):int(stop)] = LETTERS.index(letter)
    return out
