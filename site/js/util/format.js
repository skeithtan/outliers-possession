/**
 * Number and time formats for the page. Every format uses tabular numerals in CSS (`.tnum`).
 *
 * @module util/format
 */

const MINUS = '−';

/**
 * An integer with thousands separators: 199319 becomes "199,319".
 * @param {number} n
 * @returns {string}
 */
export function int(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return '–';
  const s = Math.round(Math.abs(n)).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  return (n < 0 ? MINUS : '') + s;
}

/**
 * A fixed-decimal number with a true minus sign.
 * @param {number} n
 * @param {number} [digits]
 * @returns {string}
 */
export function fixed(n, digits = 3) {
  if (n === null || n === undefined || Number.isNaN(n)) return '–';
  const s = Math.abs(n).toFixed(digits);
  return (n < 0 && Number(s) !== 0 ? MINUS : '') + s;
}

/**
 * A signed number: "+4.0" or "−2.6".
 * @param {number} n
 * @param {number} [digits]
 * @returns {string}
 */
export function signed(n, digits = 1) {
  if (n === null || n === undefined || Number.isNaN(n)) return '–';
  const s = Math.abs(n).toFixed(digits);
  if (Number(s) === 0) return s;
  return (n < 0 ? MINUS : '+') + s;
}

/**
 * A share as a percentage: 0.498 becomes "49.8%".
 * @param {number} share
 * @param {number} [digits]
 * @returns {string}
 */
export function pct(share, digits = 1) {
  if (share === null || share === undefined || Number.isNaN(share)) return '–';
  return (share * 100).toFixed(digits) + '%';
}

/**
 * Video time of a frame as "mm:ss.ff", with ff the frame inside the second.
 * @param {number} frame
 * @param {number} [fps]
 * @returns {string}
 */
export function videoTime(frame, fps = 25) {
  const f = Math.max(0, Math.round(frame));
  const total = Math.floor(f / fps);
  const mm = Math.floor(total / 60);
  const ss = total % 60;
  const ff = f % fps;
  return pad(mm) + ':' + pad(ss) + '.' + pad(ff);
}

/**
 * Match clock of a frame: minutes and seconds since the kickoff of its half, plus 45 minutes in
 * the second half. Returns null before the first kickoff.
 * @param {number} frame
 * @param {Array<{half: number, frame: number}>} kickoffs
 * @param {number} [fps]
 * @returns {{half: number, text: string}|null}
 */
export function matchClock(frame, kickoffs, fps = 25) {
  const ks = (kickoffs || []).slice().sort((a, b) => a.frame - b.frame);
  let k = null;
  for (const item of ks) if (frame >= item.frame) k = item;
  if (!k) return null;
  const secs = (frame - k.frame) / fps + (k.half === 2 ? 45 * 60 : 0);
  const mm = Math.floor(secs / 60);
  const ss = Math.floor(secs % 60);
  return { half: k.half, text: mm + '′' + pad(ss) + '″' };
}

/**
 * Seconds for a frame count.
 * @param {number} frames
 * @param {number} [fps]
 * @returns {string}
 */
export function seconds(frames, fps = 25) {
  return (frames / fps).toFixed(1) + ' s';
}

function pad(n) {
  return n < 10 ? '0' + n : String(n);
}

/**
 * Escape text for HTML.
 * @param {*} s
 * @returns {string}
 */
export function esc(s) {
  return String(s === null || s === undefined ? '' : s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * The club name to show for a team name of the bundle. The organisers' Smart Tagging spells the
 * away club "Atletic Club" and the home club "Barcelona". The page shows the clubs' own names.
 * Any other name passes through unchanged.
 * @param {string} name
 * @returns {string}
 */
export function teamDisplayName(name) {
  const clubs = { Barcelona: 'FC Barcelona', 'Atletic Club': 'Athletic Club' };
  return Object.prototype.hasOwnProperty.call(clubs, name) ? clubs[name] : name;
}
