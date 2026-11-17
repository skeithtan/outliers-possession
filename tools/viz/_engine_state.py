"""One engine run shared by both exporters (plan "Exporters", `_engine_state.py`).

engine_state() runs `possession.prepare()` and `possession.run(return_parts=True)` once. It
refuses a results file whose labels differ from the labels of the engine, because the bundle
would then show internals that did not make those labels.
"""

import numpy as np

from tools import _common as cm
from tools.viz import rle


class EngineMismatch(ValueError):
    """The results file does not hold the labels that the engine makes now."""


def engine_state(cfg=None, game=None, results_path=None, allow_mismatch=False):
    """(labels_codes, inputs, parts, md5) of one engine run.

    `labels_codes` are the int8 codes of the results file (H=0 A=1 D=2 X=3), not of the engine.
    `inputs` is the prepare() dict and `parts` the dict of run(return_parts=True). `md5` is the
    md5 of the results file. With `allow_mismatch` True, a results file that differs from the
    engine passes. The baseline smoke test needs that.
    """
    from engine import possession, submission
    from engine.config import Config

    cfg = cfg or Config.default()
    codes = cm.read_results(results_path, game)
    inputs = possession.prepare(cfg, game)
    labels, parts = possession.run(cfg, return_parts=True, inputs=inputs)
    engine_codes = _letters_to_codes(labels)
    differ = int(np.sum(engine_codes != codes))
    if differ and not allow_mismatch:
        raise EngineMismatch(
            "{} frames of {} differ from the engine output. Rebuild the results file with "
            "python -m engine.run, or pass --allow-md5-mismatch".format(differ, results_path))
    return codes, inputs, parts, submission.md5_of(results_path)


def _letters_to_codes(labels):
    """int8 codes of an array of label letters."""
    labels = np.asarray(labels).astype(str)
    out = np.full(len(labels), rle.X_CODE, dtype=np.int8)
    for code, letter in enumerate(rle.LETTERS):
        out[labels == letter] = code
    return out
