"""Make a controlled-possession view of an external reference. Not part of the submission.

    python tools/analysis/control_reference.py REFERENCE.csv --seconds 2 --out OUT.csv

DFL gives the ball to a team at each touch. The challenge needs controlled possession
(docs/challenge/CHALLENGE.md, "Controlled" possession). One proxy in that text is "a player
keeps the ball for a few seconds". This tool applies that proxy to a reference:

- An alive stretch is a run of frames labelled H or A.
- In each alive stretch, a team run shorter than the minimum is not a change of possession.
  The tool gives its frames to the team that had the ball before it.
- The first run of an alive stretch never changes. A restart always has a clear possessor.
- D and X frames never change.

The minimum is a convention, not a fitted value. The default is 2 s, from the challenge text.
Score the 1 s and 3 s views as stress views.

The rule runs left to right. A short run that the rule absorbs does not become the new owner,
so two short runs of the new team in a row both go back to the old team. The output keeps the
header and the frame column of the input. Deterministic. Python 3.9.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

FPS = 25
DEFAULT_SECONDS = 2.0


def control_labels(labels, min_frames):
    """Return a copy of the labels with team runs under min_frames absorbed. See the docstring."""
    out = np.array(labels, dtype=object).copy()
    if min_frames <= 0:
        return out
    alive = np.isin(out, ("H", "A"))
    n = len(out)
    i = 0
    while i < n:
        if not alive[i]:
            i += 1
            continue
        end = i
        while end < n and alive[end]:
            end += 1
        owner = out[i]
        k = i
        while k < end:
            m = k
            while m < end and out[m] == out[k]:
                m += 1
            if out[k] != owner and (m - k) < min_frames:
                out[k:m] = owner
            else:
                owner = out[k]
            k = m
        i = end
    return out


def seconds_to_frames(seconds):
    """Frames of a duration at 25 fps, rounded to the nearest frame."""
    return int(round(float(seconds) * FPS))


def write_control_reference(src, dst, seconds=DEFAULT_SECONDS):
    """Write the controlled view of the reference src to dst. Return dst."""
    table = pd.read_csv(src, dtype={"label": str})
    labels = table["label"].str.strip().str.upper().to_numpy()
    table["label"] = control_labels(labels, seconds_to_frames(seconds))
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmp = dst + ".tmp"
    table.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, dst)
    return dst


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", help="reference CSV with the columns frame,label")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS,
                        help="minimum control time in seconds (default: 2)")
    parser.add_argument("--out", required=True, help="output CSV")
    args = parser.parse_args(argv)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if os.path.abspath(args.reference).startswith(os.path.join(root, "evaluation")):
        parser.error("this tool is for external references only")
    print(write_control_reference(args.reference, args.out, args.seconds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
