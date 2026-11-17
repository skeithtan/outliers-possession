/**
 * A copy of the temporal metrics of `evaluation/possession_metrics.py`, for the page.
 *
 * Each function has the same name and the same result as its Python original, in camel case.
 * The page uses them to animate the matching and checks the counts against `scores.json`.
 * Codes: H=0, A=1, D=2, X=3. Every interval is half-open.
 *
 * @module util/spells
 */

export const H = 0;
export const A = 1;
export const D = 2;
export const X = 3;
/** `SPELL_IOUS` of the scorer. */
export const SPELL_IOUS = [0.3, 0.4, 0.5, 0.6, 0.7];
/** `TAUS` of the scorer, in frames. */
export const TAUS = [12, 25];

/**
 * `valid_runs`: half-open spans where the ground truth is not X.
 * @param {Uint8Array} gt
 * @returns {Array<[number, number]>}
 */
export function validRuns(gt) {
  const out = [];
  let start = -1;
  for (let i = 0; i < gt.length; i++) {
    const graded = gt[i] !== X;
    if (graded && start < 0) start = i;
    else if (!graded && start >= 0) {
      out.push([start, i]);
      start = -1;
    }
  }
  if (start >= 0) out.push([start, gt.length]);
  return out;
}

/**
 * `fill_abstentions`: forward-fill X inside each run, then back-fill the head of the run.
 * @param {Uint8Array} pred
 * @param {Array<[number, number]>} runs
 * @returns {Uint8Array} A copy.
 */
export function fillAbstentions(pred, runs) {
  const out = pred.slice();
  for (const [start, stop] of runs) {
    let first = -1;
    for (let i = start; i < stop; i++) {
      if (pred[i] !== X) {
        first = i;
        break;
      }
    }
    if (first < 0) continue;
    let last = pred[first];
    for (let i = start; i < stop; i++) {
      if (pred[i] !== X) last = pred[i];
      out[i] = last;
    }
  }
  return out;
}

/**
 * `segments`: constant-label runs, encoded inside each valid run separately.
 * @param {Uint8Array} codes
 * @param {Array<[number, number]>} runs
 * @returns {Array<[number, number, number]>}
 */
export function segments(codes, runs) {
  const out = [];
  for (const [start, stop] of runs) {
    if (stop <= start) continue;
    let s = start;
    for (let i = start + 1; i <= stop; i++) {
      if (i === stop || codes[i] !== codes[s]) {
        out.push([s, i, codes[s]]);
        s = i;
      }
    }
  }
  return out;
}

/**
 * `transitions`: label changes inside each run, as [frame, fromCode, toCode].
 * @param {Uint8Array} codes
 * @param {Array<[number, number]>} runs
 * @returns {Array<[number, number, number]>}
 */
export function transitions(codes, runs) {
  const out = [];
  for (const [start, stop] of runs) {
    for (let i = start + 1; i < stop; i++) {
      if (codes[i] !== codes[i - 1]) out.push([i, codes[i - 1], codes[i]]);
    }
  }
  return out;
}

/**
 * `possession_spells`: maximal H or A runs.
 * @param {Uint8Array} codes
 * @param {Array<[number, number]>} runs
 * @returns {Array<[number, number, number]>}
 */
export function possessionSpells(codes, runs) {
  return segments(codes, runs).filter((s) => s[2] === H || s[2] === A);
}

/**
 * `spell_overlaps`: [iou, gtIndex, predIndex] for every same-team overlapping pair, sorted by
 * (-iou, gtIndex, predIndex) exactly as the scorer sorts them.
 * @param {Array<[number, number, number]>} gtSpells
 * @param {Array<[number, number, number]>} predSpells
 * @returns {Array<[number, number, number]>}
 */
export function spellOverlaps(gtSpells, predSpells) {
  const out = [];
  gtSpells.forEach(([gs, ge, gc], i) => {
    predSpells.forEach(([ps, pe, pc], j) => {
      if (pc !== gc || pe <= gs || ge <= ps) return;
      const inter = Math.min(ge, pe) - Math.max(gs, ps);
      const union = Math.max(ge, pe) - Math.min(gs, ps);
      out.push([inter / union, i, j]);
    });
  });
  out.sort((a, b) => (b[0] - a[0]) || (a[1] - b[1]) || (a[2] - b[2]));
  return out;
}

/**
 * `matched_gt_flags`, extended to return the pairs: a greedy one-to-one match on the
 * IoU-descending list at one threshold.
 * @param {Array<[number, number, number]>} overlaps
 * @param {number} nGt
 * @param {number} threshold
 * @returns {{flags: boolean[], pairs: Array<[number, number, number]>, predMatched: Set<number>}}
 */
export function matchedGtFlags(overlaps, nGt, threshold) {
  const flags = new Array(nGt).fill(false);
  const predMatched = new Set();
  const pairs = [];
  for (const [iou, gi, pj] of overlaps) {
    if (iou < threshold) break;
    if (flags[gi] || predMatched.has(pj)) continue;
    flags[gi] = true;
    predMatched.add(pj);
    pairs.push([iou, gi, pj]);
  }
  return { flags, pairs, predMatched };
}

/**
 * `match_spells` plus precision, recall and F1, as `spell_scores` computes them.
 * @param {Array} overlaps
 * @param {number} nGt
 * @param {number} nPred
 * @param {number} threshold
 * @returns {{tp: number, fp: number, fn: number, precision: number, recall: number, f1: number,
 *           pairs: Array, flags: boolean[], predMatched: Set<number>}}
 */
export function matchSpells(overlaps, nGt, nPred, threshold) {
  const m = matchedGtFlags(overlaps, nGt, threshold);
  const tp = m.pairs.length;
  const fp = nPred - tp;
  const fn = nGt - tp;
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  return Object.assign({ tp, fp, fn, precision, recall, f1 }, m);
}

/**
 * Prepare one comparison as `evaluate` does: valid runs from the ground truth, the filled
 * prediction, both spell lists and the sorted overlaps.
 * @param {Uint8Array} gt
 * @param {Uint8Array} pred
 */
export function prepare(gt, pred) {
  const runs = validRuns(gt);
  const filled = fillAbstentions(pred, runs);
  const gtSpells = possessionSpells(gt, runs);
  const predSpells = possessionSpells(filled, runs);
  const overlaps = spellOverlaps(gtSpells, predSpells);
  return { runs, filled, gtSpells, predSpells, overlaps };
}

/**
 * `spell_scores`: the match at every threshold and the mean F1.
 * @param {Uint8Array} gt
 * @param {Uint8Array} pred
 * @param {number[]} [ious]
 */
export function spellScores(gt, pred, ious = SPELL_IOUS) {
  const p = prepare(gt, pred);
  const perIou = {};
  let sum = 0;
  for (const t of ious) {
    const m = matchSpells(p.overlaps, p.gtSpells.length, p.predSpells.length, t);
    perIou[t] = m;
    sum += m.f1;
  }
  return Object.assign(p, { perIou, meanF1: ious.length ? sum / ious.length : 0 });
}

/**
 * `match_transitions` with typed="to": one-to-one match on the destination label within
 * +/- tol frames, earliest to earliest.
 * @param {Array<[number, number, number]>} gtTrans
 * @param {Array<[number, number, number]>} predTrans
 * @param {number} tol
 * @returns {{pairs: Array, missed: Array, spurious: Array}}
 */
export function matchTransitions(gtTrans, predTrans, tol) {
  const pairs = [];
  const missed = [];
  const spurious = [];
  const keys = Array.from(new Set(gtTrans.map((t) => t[2]).concat(predTrans.map((t) => t[2]))))
    .sort((a, b) => a - b);
  for (const key of keys) {
    const gts = gtTrans.filter((t) => t[2] === key);
    const preds = predTrans.filter((t) => t[2] === key);
    let i = 0;
    let j = 0;
    while (i < gts.length && j < preds.length) {
      const delta = preds[j][0] - gts[i][0];
      if (delta < -tol) {
        spurious.push(preds[j]);
        j += 1;
      } else if (delta > tol) {
        missed.push(gts[i]);
        i += 1;
      } else {
        pairs.push([gts[i], preds[j]]);
        i += 1;
        j += 1;
      }
    }
    for (; i < gts.length; i++) missed.push(gts[i]);
    for (; j < preds.length; j++) spurious.push(preds[j]);
  }
  return { pairs, missed, spurious };
}

/**
 * Our transition into the same label closest in time, as `tools/forensics.py`
 * `nearest_same_destination` finds it.
 * @param {[number, number, number]} gtTr
 * @param {Array<[number, number, number]>} predTrans
 * @returns {[number, number, number]|null}
 */
export function nearestSameDestination(gtTr, predTrans) {
  let best = null;
  for (const p of predTrans) {
    if (p[2] !== gtTr[2]) continue;
    if (!best) best = p;
    else {
      const d = Math.abs(p[0] - gtTr[0]);
      const bd = Math.abs(best[0] - gtTr[0]);
      if (d < bd || (d === bd && p[0] < best[0])) best = p;
    }
  }
  return best;
}

/**
 * IoU of two half-open intervals.
 * @returns {number}
 */
export function iou(a0, a1, b0, b1) {
  const inter = Math.max(0, Math.min(a1, b1) - Math.max(a0, b0));
  const union = Math.max(a1, b1) - Math.min(a0, b0);
  return union > 0 ? inter / union : 0;
}
