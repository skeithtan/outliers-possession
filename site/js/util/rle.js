/**
 * Helpers for RLE label runs `[start, stop, letter]`, see SCHEMA.md Section 1.
 *
 * @module util/rle
 */

import { decodeRuns, LETTERS } from '../data.js';

export { decodeRuns, LETTERS };

/**
 * Clip runs to `[lo, hi)`.
 * @param {Array} runs
 * @param {number} lo
 * @param {number} hi
 * @returns {Array}
 */
export function clipRuns(runs, lo, hi) {
  const out = [];
  for (const [a, b, k] of runs) {
    if (b <= lo || a >= hi) continue;
    out.push([Math.max(a, lo), Math.min(b, hi), k]);
  }
  return out;
}

/**
 * Index of the first run whose stop is above `frame`, by binary search.
 * @param {Array} runs
 * @param {number} frame
 * @returns {number}
 */
export function firstRunAfter(runs, frame) {
  let lo = 0;
  let hi = runs.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (runs[mid][1] <= frame) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/**
 * Label fractions per pixel column. Column `c` covers the frames
 * `[d0 + c * fpc, d0 + (c + 1) * fpc)` with `fpc = (d1 - d0) / cols`. A frame occupies the
 * continuous span `[f, f + 1)`. A frame in no run counts as X.
 * @param {Array} runs  Sorted runs.
 * @param {number} d0  First frame of the domain, can be fractional.
 * @param {number} d1  End of the domain.
 * @param {number} cols  Number of columns.
 * @returns {Float32Array} `cols * 4` fractions in the order H, A, D, X.
 */
export function columnFractions(runs, d0, d1, cols) {
  const out = new Float32Array(cols * 4);
  const fpc = (d1 - d0) / cols;
  if (!(fpc > 0)) return out;
  for (let i = firstRunAfter(runs, d0); i < runs.length; i++) {
    const r = runs[i];
    if (r[0] >= d1) break;
    const code = LETTERS.indexOf(r[2]);
    const a = Math.max(r[0], d0);
    const b = Math.min(r[1], d1);
    if (b <= a) continue;
    let c0 = Math.floor((a - d0) / fpc);
    const c1 = Math.min(cols - 1, Math.floor((b - d0) / fpc));
    for (let c = Math.max(0, c0); c <= c1; c++) {
      const cs = d0 + c * fpc;
      const ce = cs + fpc;
      const ov = Math.min(b, ce) - Math.max(a, cs);
      if (ov > 0) out[c * 4 + code] += ov / fpc;
    }
  }
  for (let c = 0; c < cols; c++) {
    const s = out[c * 4] + out[c * 4 + 1] + out[c * 4 + 2] + out[c * 4 + 3];
    out[c * 4 + 3] += Math.max(0, 1 - s);
  }
  return out;
}

/**
 * Maximal runs of one letter, for example the possession spells of H.
 * @param {Array} runs
 * @param {string[]} letters
 * @returns {Array}
 */
export function runsOf(runs, letters) {
  return runs.filter((r) => letters.indexOf(r[2]) >= 0);
}
