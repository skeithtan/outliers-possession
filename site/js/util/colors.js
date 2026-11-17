/**
 * Colours for canvas drawing, read from the CSS tokens in `css/tokens.css`.
 *
 * SVG elements take their colours from CSS classes. A canvas needs resolved values, so this
 * module reads the custom properties once and caches them.
 *
 * @module util/colors
 */

let cache = null;

/**
 * Read one custom property of `:root`.
 * @param {string} name  For example '--h'.
 * @returns {string}
 */
export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/**
 * Parse "#rrggbb" or "#rgb" to [r, g, b].
 * @param {string} hex
 * @returns {number[]}
 */
export function hexRgb(hex) {
  let h = hex.replace('#', '').trim();
  if (h.length === 3) h = h.split('').map((c) => c + c).join('');
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/**
 * Resolved palette. Label colours are in code order H, A, D, X.
 * @returns {{label: string[], labelRgb: number[][], gt: string, baseline: string, bg: string,
 *           panel: string, hairline: string, text: string, muted: string, onset: string,
 *           restart: string}}
 */
export function palette() {
  if (cache) return cache;
  const label = ['--h', '--a', '--d', '--x'].map(cssVar);
  cache = {
    label,
    labelRgb: label.map(hexRgb),
    gt: cssVar('--gt'),
    baseline: cssVar('--baseline'),
    bg: cssVar('--bg'),
    panel: cssVar('--panel'),
    hairline: cssVar('--hairline'),
    text: cssVar('--text'),
    muted: cssVar('--muted'),
    onset: cssVar('--onset'),
    restart: cssVar('--restart'),
  };
  return cache;
}

/** Class name of a label letter, for SVG fills: `lab-H`, `lab-A`, `lab-D`, `lab-X`. */
export function labelClass(letter) {
  return 'lab-' + letter;
}

/**
 * A hatch pattern for X, made once per context.
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} [scale]  Device pixel ratio.
 * @returns {CanvasPattern}
 */
export function hatchPattern(ctx, scale = 1) {
  const p = palette();
  const size = Math.max(6, Math.round(7 * scale));
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const g = c.getContext('2d');
  g.fillStyle = p.bg;
  g.fillRect(0, 0, size, size);
  g.strokeStyle = p.hairline;
  g.lineWidth = Math.max(1, scale);
  g.beginPath();
  g.moveTo(-1, size + 1);
  g.lineTo(size + 1, -1);
  g.moveTo(-1, 1);
  g.lineTo(1, -1);
  g.moveTo(size - 1, size + 1);
  g.lineTo(size + 1, size - 1);
  g.stroke();
  return ctx.createPattern(c, 'repeat');
}
