/**
 * Data loader API. See `site/data/SCHEMA.md` for the full schema of both bundles.
 *
 * All URLs are relative, for example `./data/public/...`, so the site works from any path,
 * including GitHub Pages. Nothing here reads `data/` or `cache/` of the repository.
 *
 * Report key names. `scores.json` copies the keys of the scorer report unchanged. Inside
 * `spell_f1[iou]` the report names the counts `n_matched`, `n_spurious` and `n_missed`, and
 * SCHEMA.md Section 2.6 calls them `tp`, `fp` and `fn`. `spellCounts()` reads either name.
 * The bootstrap bounds are `bootstrap.lo` and `bootstrap.hi`, or `spell_bootstrap_low` and
 * `spell_bootstrap_high` in a raw report. `bootstrapBounds()` reads either name.
 *
 * Label codes in decoded arrays: H=0, A=1, D=2, X=3.
 *
 * @module data
 */

/** Label letters in code order. */
export const LETTERS = ['H', 'A', 'D', 'X'];

/** @typedef {[number, number, 'H'|'A'|'D'|'X']} Run  A half-open RLE run `[start, stop, letter]`. */

/**
 * @typedef {Object} Manifest
 * @property {number} schema
 * @property {string} version
 * @property {string} md5
 * @property {string} config_hash
 * @property {{commit: string, commit_date: string, tag: (string|null), dirty: boolean}} git
 * @property {string} game
 * @property {number} n_frames
 * @property {number} fps
 * @property {[number, number]} pitch_m
 * @property {{H: string, A: string}} team_names
 * @property {Array<[number, number]>} graded_blocks
 * @property {Array<{half: number, frame: number}>} kickoffs
 * @property {{start: number, stop: number}} public_window
 * @property {string[]} team
 * @property {{command: string, python: string, pins: Object<string, string>, n_tests: number}} repro
 * @property {Object<string, number>} files
 */

/**
 * @typedef {Object} BinHeader
 * @property {string} [dtype]
 * @property {number} [scale]
 * @property {number} [sentinel]
 * @property {number} frame_start
 * @property {number} frame_stop
 * @property {number} n_slots
 * @property {Object<string, {offset: number, shape: number[], dtype: (string|undefined), scale: (number|undefined)}>} arrays
 */

/**
 * @typedef {Object} BinFile
 * @property {BinHeader} header
 * @property {Object<string, (Int16Array|Uint16Array|Int8Array|Uint8Array)>} views
 *   One typed view for each array in the header, over the same buffer.
 */

/**
 * @typedef {Object} PublicBundle
 * @property {Manifest} manifest                      manifest.json
 * @property {{window: {start: number, stop: number}, ours: Run[], gt_public: Run[],
 *             baseline_smooth50: Run[], baseline_smooth125: Run[]}} labelsWindow
 * @property {Object<string, {name: string, n_frames: number, runs: Run[]}>} baselines
 *   Keyed by file stem: `smooth50`, `smooth125`.
 * @property {{bin_frames: number, n_bins: number, ours: Object[], baseline_smooth50: Object[]}} minuteBins
 * @property {{events: Array, owner_events: Array, onsets: Array, restarts: Array, stoppages: Array}} windowDetail
 * @property {Object<string, Object>} scores          scores.json
 * @property {Object} transitions                     transitions.json
 * @property {Object} sanity                          sanity.json
 * @property {{config_hash: string, n_fields: number, sections: Object[]}} config   config.json
 * @property {{rows: Object[], rejected: Object[]}} ladder    ladder.json
 * @property {Object[]} cycles                        cycles.json
 */

/**
 * @typedef {Object} LocalBundle
 * @property {Manifest & {halves: Object[]}} manifest      local manifest.json
 * @property {Run[]} labelsOurs                            full-match runs of ours
 * @property {Object} eventsXy                             events_xy.json
 * @property {Object} deadballXy                           deadball_xy.json
 * @property {Object} transitionsKin                       transitions_kin.json
 * @property {{team_of_slot: string[], visible_counts: number[]}} slots   slots.json
 * @property {Object<number, {positions: BinFile, frames: BinFile}>} halves   keyed by half, 1 or 2
 */

const PUBLIC_DIR = './data/public/';
const LOCAL_DIR = './data/local/';

/** Bundle key for each public file. A file not in this map goes to `bundle.extra`. */
const PUBLIC_KEYS = {
  'labels_window.json': 'labelsWindow',
  'minute_bins.json': 'minuteBins',
  'window_detail.json': 'windowDetail',
  'scores.json': 'scores',
  'transitions.json': 'transitions',
  'sanity.json': 'sanity',
  'config.json': 'config',
  'ladder.json': 'ladder',
  'cycles.json': 'cycles',
};

/** Local JSON files, see SCHEMA.md Section 3.4. */
const LOCAL_KEYS = {
  'labels_ours.json': 'labelsOurs',
  'events_xy.json': 'eventsXy',
  'deadball_xy.json': 'deadballXy',
  'transitions_kin.json': 'transitionsKin',
  'slots.json': 'slots',
};

/** Typed array constructor for each dtype name of a binary header. */
const DTYPES = {
  int16: Int16Array,
  uint16: Uint16Array,
  int8: Int8Array,
  uint8: Uint8Array,
};

/**
 * Reason that the last `probeLocal` call returned null because of an md5 difference.
 * `null` when there was no mismatch. Read it after `probeLocal` to show a banner.
 * @type {{reason: string, publicMd5: string, localMd5: string}|null}
 */
export let mismatch = null;

/**
 * Fetch one JSON file. Reject with a clear message when the response is not OK.
 * @param {string} url
 * @returns {Promise<*>}
 */
async function getJson(url) {
  const res = await fetchRetry(url);
  if (!res.ok) throw new Error('fetch ' + url + ' failed: HTTP ' + res.status);
  return res.json();
}

/**
 * Fetch with up to three retries after a dropped connection. `python -m http.server` has a short
 * listen queue and can reset a connection when the page asks for many files at once. An HTTP
 * error status is a response, not a dropped connection, so it is not retried.
 * @param {string} url
 * @param {RequestInit} [init]
 * @returns {Promise<Response>}
 */
async function fetchRetry(url, init) {
  for (let attempt = 0; ; attempt++) {
    try {
      return await fetch(url, init);
    } catch (err) {
      if (attempt >= 3) throw err;
      await new Promise((r) => setTimeout(r, 200 * (attempt + 1)));
    }
  }
}

/**
 * Fetch every public file listed in `./data/public/manifest.json` `files`.
 * Paths are relative. A failed fetch rejects the promise.
 * @returns {Promise<PublicBundle>}
 */
export async function loadPublic() {
  const manifest = await getJson(PUBLIC_DIR + 'manifest.json');
  const names = Object.keys(manifest.files || {}).sort();
  const bodies = await Promise.all(names.map((n) => getJson(PUBLIC_DIR + n)));
  const bundle = { manifest, baselines: {}, extra: {} };
  names.forEach((name, i) => {
    const body = bodies[i];
    const m = /^labels_baseline_(.+)\.json$/.exec(name);
    if (m) bundle.baselines[m[1]] = body;
    else if (PUBLIC_KEYS[name]) bundle[PUBLIC_KEYS[name]] = body;
    else bundle.extra[name] = body;
  });
  return bundle;
}

/**
 * Probe for the local bundle at `./data/local/manifest.json`.
 * Returns null on any failure, for example a 404. Returns null and sets `mismatch` when the
 * local `md5` differs from `publicManifest.md5`. On success it loads the JSON files and opens the
 * binary pair of each half with `openBin`.
 * @param {Manifest} publicManifest
 * @returns {Promise<LocalBundle|null>}
 */
export async function probeLocal(publicManifest) {
  mismatch = null;
  let manifest;
  try {
    const res = await fetch(LOCAL_DIR + 'manifest.json', { cache: 'no-store' });
    if (!res.ok) return null;
    manifest = await res.json();
  } catch (err) {
    return null;
  }
  if (!publicManifest || manifest.md5 !== publicManifest.md5) {
    mismatch = {
      reason: 'The local bundle comes from a different results file than the public bundle.',
      publicMd5: publicManifest ? publicManifest.md5 : '',
      localMd5: manifest.md5 || '',
    };
    return null;
  }
  try {
    const local = { manifest, halves: {} };
    const names = Object.keys(LOCAL_KEYS);
    const bodies = await Promise.all(names.map((n) => getJson(LOCAL_DIR + n)));
    names.forEach((n, i) => {
      local[LOCAL_KEYS[n]] = bodies[i];
    });
    // labels_ours.json holds {name, n_frames, runs}. The API exposes the runs.
    if (local.labelsOurs && !Array.isArray(local.labelsOurs)) {
      local.labelsOurs = local.labelsOurs.runs;
    }
    const halves = manifest.halves || [];
    const opened = await Promise.all(halves.map((h) => Promise.all([
      openBin(LOCAL_DIR + h.files.positions),
      openBin(LOCAL_DIR + h.files.frames),
    ])));
    halves.forEach((h, i) => {
      local.halves[h.half] = { positions: opened[i][0], frames: opened[i][1] };
    });
    return local;
  } catch (err) {
    console.error('local bundle found but could not be loaded', err);
    mismatch = {
      reason: 'The local bundle is incomplete: ' + err.message,
      publicMd5: publicManifest.md5,
      localMd5: manifest.md5 || '',
    };
    return null;
  }
}

/**
 * Decode RLE runs to one label code for each frame.
 * Codes: H=0, A=1, D=2, X=3. A frame in no run is X.
 * @param {Run[]} runs  Half-open runs sorted by start.
 * @param {number} nFrames  Length of the result.
 * @returns {Uint8Array}
 */
export function decodeRuns(runs, nFrames) {
  const out = new Uint8Array(nFrames).fill(3);
  for (const [start, stop, letter] of runs) {
    const code = LETTERS.indexOf(letter);
    if (code < 0) throw new Error('bad label letter ' + letter);
    out.fill(code, Math.max(0, start), Math.min(nFrames, stop));
  }
  return out;
}

/**
 * Look up the label of one frame in RLE runs, by binary search. Returns 'X' for a frame in no run.
 * @param {Run[]} runs
 * @param {number} frame
 * @returns {'H'|'A'|'D'|'X'}
 */
export function labelAt(runs, frame) {
  let lo = 0;
  let hi = runs.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const r = runs[mid];
    if (frame < r[0]) hi = mid - 1;
    else if (frame >= r[1]) lo = mid + 1;
    else return r[2];
  }
  return 'X';
}

/**
 * Open a binary pair. Fetch the sidecar header JSON, then the `.bin` file next to it (same stem,
 * extension `.bin`). Make one typed view for each array at its header offset, with `Int16Array`,
 * `Uint16Array`, `Int8Array` or `Uint8Array` by the array `dtype` (or the header `dtype`).
 * Offsets are even bytes, so 16-bit views are always aligned.
 * @param {string} headerJsonUrl  For example `./data/local/positions_h1.json`.
 * @returns {Promise<BinFile>}
 */
export async function openBin(headerJsonUrl) {
  const header = await getJson(headerJsonUrl);
  const binUrl = headerJsonUrl.replace(/\.json$/, '.bin');
  const res = await fetchRetry(binUrl);
  if (!res.ok) throw new Error('fetch ' + binUrl + ' failed: HTTP ' + res.status);
  const buffer = await res.arrayBuffer();
  const views = {};
  for (const name of Object.keys(header.arrays).sort()) {
    const spec = header.arrays[name];
    const dtype = spec.dtype || header.dtype;
    const Ctor = DTYPES[dtype];
    if (!Ctor) throw new Error('unknown dtype ' + dtype + ' for ' + name);
    const length = spec.shape.reduce((a, b) => a * b, 1);
    if (spec.offset % Ctor.BYTES_PER_ELEMENT !== 0) {
      throw new Error('array ' + name + ' is not aligned at offset ' + spec.offset);
    }
    if (spec.offset + length * Ctor.BYTES_PER_ELEMENT > buffer.byteLength) {
      throw new Error('array ' + name + ' runs past the end of ' + binUrl);
    }
    views[name] = new Ctor(buffer, spec.offset, length);
  }
  return { header, views };
}

/**
 * Read the counts of one IoU row of `scores.json`, whatever name the exporter kept.
 * @param {Object} row  `scores[version].spell_f1[iou]`
 * @returns {{tp: number, fp: number, fn: number, f1: number}}
 */
export function spellCounts(row) {
  const pick = (a, b) => (row[a] !== undefined ? row[a] : row[b]);
  return {
    tp: pick('tp', 'n_matched'),
    fp: pick('fp', 'n_spurious'),
    fn: pick('fn', 'n_missed'),
    f1: row.f1,
  };
}

/**
 * Read the bootstrap bounds of one `scores.json` entry, or null when the run had no bootstrap.
 * @param {Object} entry
 * @returns {{lo: number, hi: number}|null}
 */
export function bootstrapBounds(entry) {
  if (!entry) return null;
  const b = entry.bootstrap || {};
  const lo = b.lo !== undefined ? b.lo : entry.spell_bootstrap_low;
  const hi = b.hi !== undefined ? b.hi : entry.spell_bootstrap_high;
  if (lo === null || lo === undefined || hi === null || hi === undefined) return null;
  return { lo, hi };
}

/**
 * Find the IoU row of `spell_f1` for a threshold. JSON keys are strings such as "0.3".
 * @param {Object} spellF1  `scores[version].spell_f1`
 * @param {number} iou
 * @returns {Object|undefined}
 */
export function iouRow(spellF1, iou) {
  if (!spellF1) return undefined;
  for (const key of Object.keys(spellF1)) {
    if (Math.abs(parseFloat(key) - iou) < 1e-9) return spellF1[key];
  }
  return undefined;
}

/**
 * Find the tolerance row of `transitions` for `tol` frames. JSON keys are strings such as "25".
 * @param {Object} transitions  `scores[version].transitions`
 * @param {number} tol
 * @returns {Object|undefined}
 */
export function tolRow(transitions, tol) {
  return transitions ? transitions[String(tol)] : undefined;
}

