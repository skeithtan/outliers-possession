/**
 * Replay panel: the playback loop, the HUD, the transport controls, the mini seek strip and the
 * jump menus around one pitch canvas.
 *
 * Two sources feed the same panel:
 *   local      the real tracking of the local bundle (`site/data/local/`), decoded lazily per half.
 *              The playhead is `state.frame`, written once per rendered frame and followed when
 *              another section seeks.
 *   synthetic  the scripted scenes of synthetic.js, for public mode. Scene frames are local to the
 *              scene, so the panel never writes them to `state.frame`.
 *
 *   const replay = createReplay(panelEl, ctx, {stripEl})
 *   replay.showCase({frame, from, to, insets, title})   local mode: seek and loop [from, to)
 *   replay.showScenes(ids, {index})                     synthetic: a playlist of scene ids
 *   replay.play() / pause() / seek(frame) / setActive(bool) / stats() / destroy()
 *
 * The loop draws only the latest frame: a frame accumulator advances 25 x speed frames per second.
 *
 * @module replay
 */

import { createPitch, readTokens, DEFAULT_LAYERS, LAYER_NAMES } from './pitch.js';
import { createScenes, SCENE_NAMES, configAccessor } from './synthetic.js';

const SPEEDS = [0.25, 0.5, 1, 2, 4];
const TRAIL = 10;
const BALL_TRAIL = 8;
const STRIP_HALF_S = 30;
const FLAG = { stoppage: 1, visible: 2, airborne: 4, ghost: 8, teleport: 16, graded: 32 };
const LETTER = ['H', 'A', 'D', 'X'];

// -- bundle access (accept both the snake_case file names and the camelCase loader names) -------

function pick(obj, ...names) {
  if (!obj) return undefined;
  for (const n of names) if (obj[n] != null) return obj[n];
  return undefined;
}

function runsOf(x) {
  if (!x) return null;
  if (Array.isArray(x)) return x;
  return x.runs || null;
}

/** Label of one frame in RLE runs `[start, stop, letter]`, by binary search. */
export function runLabel(runs, frame) {
  if (!runs || !runs.length) return null;
  let lo = 0, hi = runs.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const r = runs[mid];
    if (frame < r[0]) hi = mid - 1;
    else if (frame >= r[1]) lo = mid + 1;
    else return r[2];
  }
  return 'X';
}

/** First index of rows (sorted by row[0]) with row[0] >= frame. */
function lowerBound(rows, frame) {
  let lo = 0, hi = rows.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (rows[mid][0] < frame) lo = mid + 1; else hi = mid;
  }
  return lo;
}

const VIEW = { int16: Int16Array, uint16: Uint16Array, int8: Int8Array, uint8: Uint8Array };

/** Open a binary pair without data.js: the sidecar header, then the .bin (SCHEMA.md 3.2). */
async function fetchRetry(url) {
  for (let attempt = 0; ; attempt++) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error('fetch failed: ' + res.status + ' ' + url);
      return res;
    } catch (err) {
      if (attempt >= 2) throw err;
    }
  }
}

async function fetchBin(headerUrl) {
  const header = await (await fetchRetry(headerUrl)).json();
  const binUrl = headerUrl.replace(/\.json$/, '.bin');
  const buf = await (await fetchRetry(binUrl)).arrayBuffer();
  const views = {};
  for (const [name, a] of Object.entries(header.arrays)) {
    const Ctor = VIEW[a.dtype || header.dtype];
    const count = a.shape.reduce((p, q) => p * q, 1);
    views[name] = new Ctor(buf, a.offset, count);
  }
  return { header, views };
}

// -- the local source ---------------------------------------------------------------------------

function createLocalSource(ctx, cfg) {
  const local = ctx.local;
  const bundle = ctx.bundle || {};
  const manifest = local.manifest;
  const halves = manifest.halves || [];
  const nFrames = manifest.n_frames;
  const fps = manifest.fps || 25;
  const slots = pick(local, 'slots') || {};
  const teamOf = (slots.team_of_slot || []).map((t) => (t === 'H' ? 0 : t === 'A' ? 1 : 255));
  const nSlots = teamOf.length || 46;
  const team = Uint8Array.from({ length: nSlots }, (_, s) => (teamOf[s] == null ? 255 : teamOf[s]));
  const ours = runsOf(pick(local, 'labels_ours', 'labelsOurs'));
  const lw = pick(bundle, 'labels_window', 'labelsWindow') || {};
  const gt = runsOf(lw.gt_public);
  const bl = pick(bundle, 'baselines') || {};
  const base = runsOf(pick(bl, 'smooth50', 'baseline_smooth50')) || runsOf(pick(bundle, 'labels_baseline_smooth50'))
    || runsOf(lw.baseline_smooth50);
  const evx = pick(local, 'events_xy', 'eventsXy') || {};
  const events = (evx.events || []).slice().sort((p, q) => p[0] - q[0]);
  const dbx = pick(local, 'deadball_xy', 'deadballXy') || {};
  const onsets = (dbx.onsets || []).slice().sort((p, q) => p[0] - q[0]);
  const restarts = (dbx.restarts || []).slice().sort((p, q) => p[0] - q[0]);
  const stoppages = dbx.stoppages || [];
  const kickoffs = (bundle.manifest && bundle.manifest.kickoffs) || manifest.kickoffs || [];
  const span = cfg.has('CONTACT_HALF_WINDOW') ? cfg.get('CONTACT_HALF_WINDOW') : 12;
  const bins = {};
  const loading = {};

  function halfOf(frame) {
    for (const h of halves) if (frame >= h.frame_start && frame < h.frame_stop) return h.half;
    return halves.length ? (frame < halves[0].frame_start ? halves[0].half : halves[halves.length - 1].half) : 1;
  }

  async function openHalf(half) {
    if (bins[half]) return bins[half];
    if (loading[half]) return loading[half];
    loading[half] = (async () => {
      const given = (pick(local, 'positions') || {})[half] || (local.halves && local.halves[half] && local.halves[half].positions);
      const givenF = (pick(local, 'frames') || {})[half] || (local.halves && local.halves[half] && local.halves[half].frames);
      const h = halves.find((x) => x.half === half);
      const base = (ctx.localBase || './data/local/');
      const pos = given || (await fetchBin(base + h.files.positions));
      const fr = givenF || (await fetchBin(base + h.files.frames));
      bins[half] = { pos, fr, start: pos.header.frame_start, stop: pos.header.frame_stop };
      return bins[half];
    })();
    return loading[half];
  }

  function scaleOf(binFile, name) {
    const a = binFile.header.arrays[name];
    return a && a.scale != null ? a.scale : binFile.header.scale != null ? binFile.header.scale : 1;
  }

  function decoder(b) {
    const P = b.pos, F = b.fr;
    const sP = P.header.scale, sent = P.header.sentinel;
    const v = P.views, fv = F.views;
    const sc = {};
    for (const k of Object.keys(fv)) sc[k] = scaleOf(F, k);
    return {
      P, F, v, fv, sc, sent, sP,
      pos(arr, i) { const x = arr[i]; return x === sent ? NaN : x * sP; },
      u16(name, i) { const x = fv[name][i]; return x === 65535 ? NaN : x * sc[name]; },
      i16(name, i) { const x = fv[name][i]; return x === -32768 ? NaN : x * sc[name]; },
    };
  }

  function clock(frame) {
    const h = halfOf(frame);
    const ko = kickoffs.find((k) => k.half === h);
    const base = ko ? ko.frame : (halves.find((x) => x.half === h) || { frame_start: 0 }).frame_start;
    const rel = frame - base;
    const neg = rel < 0;
    const a = Math.abs(rel);
    const secs = Math.floor(a / fps) + (h === 2 && !neg ? 45 * 60 : 0);
    const ff = a % fps;
    const mm = Math.floor(secs / 60), ss = secs % 60;
    return (neg ? '-' : '') + String(mm).padStart(2, '0') + ':' + String(ss).padStart(2, '0') + '.' + String(ff).padStart(2, '0');
  }

  function near(rows, frame, lo, hi) {
    const out = [];
    for (let k = lowerBound(rows, frame - lo); k < rows.length && rows[k][0] <= frame + hi; k++) out.push(rows[k]);
    return out;
  }

  /** Frame state of one real frame, or null while its half is not loaded. */
  function frameAt(frame, insetMode) {
    insetModeNow = insetMode;
    const half = halfOf(frame);
    const b = bins[half];
    if (!b) { openHalf(half); return null; }
    const d = decoder(b);
    const row = Math.max(0, Math.min(b.stop - b.start - 1, frame - b.start));
    const ns = nSlots;
    const px = new Float32Array(ns), py = new Float32Array(ns);
    for (let s = 0; s < ns; s++) { px[s] = d.pos(d.v.px, row * ns + s); py[s] = d.pos(d.v.py, row * ns + s); }
    const tn = Math.min(TRAIL, row);
    const tpx = new Float32Array(tn * ns), tpy = new Float32Array(tn * ns);
    for (let k = 0; k < tn; k++) {
      const r = row - tn + k;
      for (let s = 0; s < ns; s++) { tpx[k * ns + s] = d.pos(d.v.px, r * ns + s); tpy[k * ns + s] = d.pos(d.v.py, r * ns + s); }
    }
    const bn = Math.min(BALL_TRAIL, row);
    const btx = new Float32Array(bn), bty = new Float32Array(bn);
    for (let k = 0; k < bn; k++) { btx[k] = d.pos(d.v.bx, row - bn + k); bty[k] = d.pos(d.v.by, row - bn + k); }
    const flags = d.fv.flags[row];
    const owner = d.fv.owner[row];
    const slotH = d.fv.slot_h[row], slotA = d.fv.slot_a[row];
    let same = 0;
    for (let k = 1; k <= 8 && row - k >= 0; k++) if (d.fv.owner[row - k] === owner) same++;
    const pDead = d.fv.p_dead[row] * d.sc.p_dead;
    const bx = d.pos(d.v.bx, row), by = d.pos(d.v.by, row);
    const speed = d.u16('speed', row);
    const fs = {
      frame, nSlots: ns, px, py, team,
      trail: { n: tn, px: tpx, py: tpy },
      ball: { x: bx, y: by, visible: (flags & FLAG.visible) !== 0 && bx === bx,
        rawX: d.pos(d.v.bx_raw, row), rawY: d.pos(d.v.by_raw, row), rawVisible: true,
        airborne: (flags & FLAG.airborne) !== 0, ghost: (flags & FLAG.ghost) !== 0, teleport: (flags & FLAG.teleport) !== 0, speed },
      ballTrail: { n: bn, x: btx, y: bty },
      glow: owner >= 0 ? { slot: owner === 0 ? slotH : slotA, team: owner, alpha: same / 8 } : null,
      pDead, stoppage: (flags & FLAG.stoppage) !== 0,
      nearest: [{ slot: slotH, d: d.u16('d_h', row) }, { slot: slotA, d: d.u16('d_a', row) }],
      events: near(events, frame, span, span).map((e) => ({ kind: e[1], team: e[2], dt: frame - e[0], x: e[4], y: e[5], label: e[1] + (e[2] ? ' ' + e[2] : '') })),
      onsets: near(onsets, frame, span * 2, 2).map((o) => { const p = onsetXY(o, d, b); return { mechanism: o[1], dt: frame - o[0], x: p[0], y: p[1] }; }),
      restarts: near(restarts, frame, span * 2, 0).map((r) => ({ kind: r[2], team: r[1], dt: frame - r[0], x: r[5], y: r[6], confirmed: r[4] })),
      labels: { ours: runLabel(ours, frame), gt: runLabel(gt, frame) === 'X' && !inWindow(frame) ? null : runLabel(gt, frame), baseline: runLabel(base, frame) },
      hud: { speed, dH: d.u16('d_h', row), dA: d.u16('d_a', row), owner, pDead, graded: (flags & FLAG.graded) !== 0 },
      overlays: [], insets: [],
      camera: camera(d, row, b, px, py, ns),
      half,
    };
    if (insetMode) addInsets(fs, d, row, b, insetMode);
    return fs;
  }

  /** Onset position: the row's own x, y, else the last visible ball before the onset (a lost ball). */
  const onsetCache = new Map();
  function onsetXY(o, d, b) {
    if (o[3] != null) return [o[3], o[4]];
    if (onsetCache.has(o[0])) return onsetCache.get(o[0]);
    let p = [NaN, NaN];
    for (let r = Math.min(b.stop - b.start - 1, o[0] - b.start); r >= Math.max(0, o[0] - b.start - 250); r--) {
      const x = d.pos(d.v.bx, r);
      if (x === x) { p = [x, d.pos(d.v.by, r)]; break; }
    }
    onsetCache.set(o[0], p);
    return p;
  }

  /** Follow camera: the centred mean of the visible ball over +-30 frames, else the players. */
  function camera(d, row, b, px, py, ns) {
    let sx = 0, sy = 0, c = 0;
    const lo = Math.max(0, row - 30), hi = Math.min(b.stop - b.start - 1, row + 30);
    for (let r = lo; r <= hi; r += 2) {
      const x = d.pos(d.v.bx, r);
      if (x === x && (d.fv.flags[r] & FLAG.visible)) { sx += x; sy += d.pos(d.v.by, r); c++; }
    }
    if (!c) for (let s = 0; s < ns; s++) if (px[s] === px[s]) { sx += px[s]; sy += py[s]; c++; }
    return c ? { x: sx / c, y: sy / c, zoom: zoomFor(insetModeNow) } : null;
  }

  let insetModeNow = null;
  function zoomFor(mode) {
    return mode === 'pursuit' || mode === 'deadball' ? 1.35 : 1.75;
  }

  function inWindow(frame) {
    const w = lw.window || (bundle.manifest && bundle.manifest.public_window);
    return !!w && frame >= w.start && frame < w.stop;
  }

  function addInsets(fs, d, row, b, mode) {
    const W = 75;
    const lo = Math.max(0, row - W), hi = Math.min(b.stop - b.start, row + W + 1);
    const n = hi - lo;
    const cursor = row - lo;
    if (mode === 'impulse') {
      const h = cfg.get('CONTACT_HALF_WINDOW');
      const dv = cfg.get('DV_CONTACT');
      const sp = new Float32Array(n), J = new Float32Array(n).fill(NaN);
      for (let k = 0; k < n; k++) sp[k] = d.u16('speed', lo + k);
      for (let k = 0; k < n; k++) {
        const r = lo + k;
        if (r - h < 0 || r + h >= b.stop - b.start) continue;
        const ax = d.i16('vx', r - h), ay = d.i16('vy', r - h), bx2 = d.i16('vx', r + h), by2 = d.i16('vy', r + h);
        J[k] = Math.hypot(bx2 - ax, by2 - ay);
      }
      const markers = [];
      for (const e of events) {
        if (e[0] < b.start + lo || e[0] >= b.start + hi) continue;
        markers.push({ i: e[0] - b.start - lo, v: J[e[0] - b.start - lo], color: 'ev-' + e[1], label: e[1] + (e[2] ? ' ' + e[2] : '') });
      }
      let jMax = dv * 1.5;
      for (let k = 0; k < n; k++) if (J[k] > jMax) jMax = J[k];
      fs.insets.push({ type: 'spark', title: 'real ball speed and velocity change over ±' + h + ' frames', n, cursor, yMax: Math.min(45, jMax * 1.1), reveal: false,
        series: [{ values: sp, color: 'muted', label: 'speed' }, { values: J, color: 'onset', fill: true, label: 'change', labelX: 52 }],
        threshold: { value: dv, label: 'contact threshold ' + dv, color: 'text' }, markers,
        readout: 'change ' + (J[cursor] === J[cursor] ? J[cursor].toFixed(1) : 'n/a') + ' m/s' });
    } else if (mode === 'pursuit' || mode === 'deadball') {
      const mv = new Float32Array(n), pd = new Float32Array(n);
      for (let k = 0; k < n; k++) { mv[k] = d.u16('meanv', lo + k); pd[k] = d.fv.p_dead[lo + k] * d.sc.p_dead * 4; }
      fs.insets.push({ type: 'spark', title: 'mean player speed and P(dead) ×4', n, cursor, yMax: 4.5, reveal: false,
        series: [{ values: mv, color: 'text', label: 'mean speed' }, { values: pd, color: 'onset', fill: true, label: 'P(dead)', labelX: 52 }],
        threshold: { value: cfg.get('V_STOP'), label: 'stop speed ' + cfg.get('V_STOP'), color: 'onset' },
        markers: onsets.filter((o) => o[0] >= b.start + lo && o[0] < b.start + hi).map((o) => ({ i: o[0] - b.start - lo, color: 'onset', label: o[1] })),
        readout: 'mean speed ' + (mv[cursor] === mv[cursor] ? mv[cursor].toFixed(2) : 'n/a') });
      if (mode === 'pursuit') fs.insets.push(nearestBars(fs, d, row));
    } else if (mode === 'restart') {
      const mv = new Float32Array(n);
      for (let k = 0; k < n; k++) mv[k] = d.u16('meanv', lo + k);
      fs.insets.push({ type: 'spark', title: 'mean player speed and restart candidates', n, cursor, yMax: 4.5, reveal: false,
        series: [{ values: mv, color: 'text', label: 'mean speed' }],
        threshold: { value: cfg.get('V_STOP'), label: 'stop speed ' + cfg.get('V_STOP'), color: 'restart' },
        markers: restarts.filter((r) => r[0] >= b.start + lo && r[0] < b.start + hi)
          .map((r) => ({ i: r[0] - b.start - lo, color: 'restart', label: r[2] + (r[4] ? '' : ' (unconfirmed)'), dashed: !r[4] })),
        readout: 'mean speed ' + (mv[cursor] === mv[cursor] ? mv[cursor].toFixed(2) : 'n/a') });
    } else if (mode === 'logE') {
      const rows = [new Float32Array(n), new Float32Array(n), new Float32Array(n)];
      const path = new Int8Array(n);
      for (let k = 0; k < n; k++) {
        const r = lo + k;
        const l = [d.i16('log_e', r * 3), d.i16('log_e', r * 3 + 1), d.i16('log_e', r * 3 + 2)];
        const m = Math.max(l[0], l[1], l[2]);
        const e = l.map((x) => (x === x ? Math.exp(x - m) : 0));
        const z = e[0] + e[1] + e[2] || 1;
        for (let j = 0; j < 3; j++) rows[j][k] = e[j] / z;
        const lab = runLabel(ours, b.start + r);
        path[k] = lab === 'H' ? 0 : lab === 'A' ? 1 : lab === 'D' ? 2 : -1;
      }
      fs.insets.push({ type: 'lanes', title: 'real evidence for each label and our decoded path', n, cursor, weight: 2.6,
        rows: [{ label: 'H', color: 'h', values: rows[0] }, { label: 'A', color: 'a', values: rows[1] }, { label: 'D', color: 'd', values: rows[2] }],
        path,
        markers: events.filter((e) => e[0] >= b.start + lo && e[0] < b.start + hi && e[2]).map((e) => ({ i: e[0] - b.start - lo, color: 'ev-' + e[1] })) });
    } else if (mode === 'cleaning') {
      const step = new Float32Array(n).fill(NaN), tele = [];
      for (let k = 1; k < n; k++) {
        const r = lo + k;
        const x1 = d.pos(d.v.bx_raw, r - 1), y1 = d.pos(d.v.by_raw, r - 1), x2 = d.pos(d.v.bx_raw, r), y2 = d.pos(d.v.by_raw, r);
        step[k] = Math.hypot(x2 - x1, y2 - y1);
        if (d.fv.flags[r] & FLAG.teleport) tele.push({ i: k, color: 'teleport' });
      }
      const mj = cfg.get('BALL_MAX_JUMP_M');
      fs.insets.push({ type: 'spark', title: 'raw ball step per frame, m', n, cursor, yMax: mj * 4, reveal: false,
        series: [{ values: step, color: 'text', label: 'step' }], threshold: { value: mj, label: 'max step ' + mj + ' m', color: 'teleport' },
        markers: tele, readout: 'step ' + (step[cursor] === step[cursor] ? step[cursor].toFixed(2) : 'n/a') + ' m' });
    }
  }

  function nearestBars(fs, d, row) {
    const k = cfg.get('PURSUIT_N_NEAREST');
    const list = [];
    if (fs.ball.visible) {
      for (let s = 0; s < fs.nSlots; s++) {
        if (!(fs.px[s] === fs.px[s])) continue;
        list.push([Math.hypot(fs.px[s] - fs.ball.x, fs.py[s] - fs.ball.y), s]);
      }
    }
    list.sort((p, q) => p[0] - q[0]);
    const bars = list.slice(0, k).map(([, s]) => {
      const r0 = Math.max(0, row - 2), r1 = row + 2;
      const v = Math.hypot(d.pos(d.v.px, r1 * fs.nSlots + s) - d.pos(d.v.px, r0 * fs.nSlots + s),
        d.pos(d.v.py, r1 * fs.nSlots + s) - d.pos(d.v.py, r0 * fs.nSlots + s)) * fps / (r1 - r0);
      return { label: String(s), value: v, color: team[s] === 0 ? 'h' : 'a' };
    });
    const mean = bars.length ? bars.reduce((p, q) => p + (q.value === q.value ? q.value : 0), 0) / bars.length : NaN;
    for (const b of bars) fs.overlays.push({ type: 'circle', slot: Number(b.label), r: 1.3, color: 'onset', width: 1.2, alpha: 0.75 });
    return { type: 'bars', title: 'speed of the ' + k + ' players nearest the ball, m/s', bars, max: 7,
      threshold: { value: cfg.get('PURSUIT_STOP_SPEED'), label: 'stop speed ' + cfg.get('PURSUIT_STOP_SPEED'), color: 'onset' },
      mean: { value: mean, label: 'mean ' + (mean === mean ? mean.toFixed(2) : 'n/a'), color: 'text' } };
  }

  return {
    kind: 'local', nFrames, fps, halves, halfOf, openHalf, frameAt, clock, ours, gt, events, onsets, restarts, stoppages,
    ready: (frame) => !!bins[halfOf(frame)],
  };
}

// -- the panel ----------------------------------------------------------------------------------

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

/**
 * Create the replay panel inside `container`.
 * @param {HTMLElement} container  the pitch panel element, emptied
 * @param {{bundle: Object, local: Object|null, state: Object, mode: string}} ctx
 * @param {{stripEl?: HTMLElement}} [opts]
 */
export function createReplay(container, ctx, opts) {
  const options = opts || {};
  const state = ctx.state;
  const bundle = ctx.bundle || {};
  const manifest = bundle.manifest || (ctx.local && ctx.local.manifest) || {};
  const pitchM = manifest.pitch_m || [105.3, 68];
  const fps = manifest.fps || 25;
  const cfg = configAccessor(bundle.config);
  const reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const debug = /[?&]debug=1/.test(location.search);
  const params = {
    rTouch: cfg.has('R_TOUCH') ? cfg.get('R_TOUCH') : NaN,
    ringFrames: cfg.has('CONTACT_HALF_WINDOW') ? cfg.get('CONTACT_HALF_WINDOW') : 12,
  };
  const isLocal = ctx.mode === 'local' && !!ctx.local;

  // DOM
  container.classList.add('rp');
  container.textContent = '';
  const hud = el('div', 'rp-hud tnum');
  const stage = el('div', 'rp-stage');
  const canvas = el('canvas', 'rp-canvas');
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', 'Pitch replay');
  const caption = el('div', 'rp-caption');
  const note = el('div', 'rp-note');
  const msg = el('div', 'rp-msg');
  // the caption and the note share one row over the top of the canvas, so they never overlap
  const top = el('div', 'rp-top');
  top.append(caption, note);
  stage.append(canvas, top, msg);
  const controls = el('div', 'rp-controls');
  container.append(hud, stage, controls);

  const hudCells = {};
  for (const [key, label] of [['frame', 'frame'], ['clock', 'clock'], ['half', 'half'], ['ours', isLocal ? 'ours' : 'labels'], ['gt', 'truth'], ['base', 'baseline'],
    ['owner', 'owner'], ['pdead', 'P(dead)'], ['speed', 'ball'], ['dh', 'nearest H'], ['da', 'nearest A']]) {
    const cell = el('div', 'rp-cell rp-cell-' + key);
    const k = el('span', 'rp-k', label);
    const v = el('span', 'rp-v', '–');
    cell.append(k, v);
    if (key === 'pdead') { const bar = el('span', 'rp-bar'); const fill = el('span', 'rp-bar-fill'); bar.append(fill); cell.append(bar); hudCells.pbar = fill; }
    hud.append(cell);
    hudCells[key] = v;
  }
  if (!isLocal) for (const k of ['gt', 'base']) hudCells[k].parentElement.hidden = true;

  // controls
  const btnPlay = el('button', 'rp-btn rp-play', 'Play');
  btnPlay.type = 'button';
  btnPlay.setAttribute('aria-label', 'Play or pause (space)');
  const speedBox = el('div', 'rp-speeds');
  const speedBtns = SPEEDS.map((s) => {
    const b = el('button', 'rp-btn rp-speed', s + '×');
    b.type = 'button';
    b.addEventListener('click', () => setSpeed(s));
    speedBox.append(b);
    return b;
  });
  const btnBack = el('button', 'rp-btn', '− 1f');
  const btnFwd = el('button', 'rp-btn', '+ 1f');
  btnBack.type = btnFwd.type = 'button';
  btnBack.addEventListener('click', () => step(-1));
  btnFwd.addEventListener('click', () => step(1));
  const chipBox = el('div', 'rp-chips');
  const halfSel = el('select', 'rp-select');
  halfSel.setAttribute('aria-label', 'Half');
  const btnKeep = el('button', 'rp-btn rp-keep', 'Keep playing');
  btnKeep.type = 'button';
  btnKeep.hidden = true;
  btnKeep.addEventListener('click', () => { loop = null; btnKeep.hidden = true; play(); });
  const layersBtn = el('button', 'rp-btn', 'Layers');
  layersBtn.type = 'button';
  const jumpBtn = el('button', 'rp-btn', 'Jump to');
  jumpBtn.type = 'button';
  const pop = el('div', 'rp-pop');
  pop.hidden = true;
  controls.append(btnPlay, btnBack, btnFwd, speedBox, chipBox, halfSel, btnKeep, layersBtn, jumpBtn, pop);

  // the mini seek strip
  const strip = options.stripEl || el('div');
  strip.classList.add('rp-strip');
  strip.textContent = '';
  const stripCanvas = el('canvas', 'rp-strip-canvas');
  stripCanvas.setAttribute('aria-label', 'Seek strip, plus and minus 30 seconds');
  const stripLegend = el('div', 'rp-strip-legend');
  strip.append(stripLegend, stripCanvas);
  if (!options.stripEl) container.append(strip);

  let tokens = readTokens();
  const pitch = createPitch(canvas, { pitchM, tokens,
    marks: { boxDepth: cfg.has('PENALTY_AREA_DEPTH_M') ? cfg.get('PENALTY_AREA_DEPTH_M') : 16.5,
      boxWidth: cfg.has('PENALTY_AREA_WIDTH_M') ? cfg.get('PENALTY_AREA_WIDTH_M') : 40.32 } });

  // playback state
  let layers = Object.assign({}, DEFAULT_LAYERS);
  // v2: the follow camera default changed to off, an older saved choice no longer applies
  try { const saved = JSON.parse(localStorage.getItem('rp-layers-v2') || 'null'); if (saved) layers = Object.assign(layers, saved); } catch (e) { /* storage blocked */ }
  let playing = false;
  let speed = 1;
  let frame = 0;
  let acc = 0;
  let last = 0;
  let loop = null;
  let dirty = true;
  let visible = true;
  let active = false;
  let raf = 0;
  let lastWritten = null;
  let insetMode = null;
  const drawTimes = [];
  let source = null;
  let local = null;
  let scenes = null;
  let playlist = [];
  let plIndex = 0;
  let scene = null;
  let destroyed = false;
  let jumpBuilder = null;

  if (isLocal) {
    local = createLocalSource(ctx, cfg);
    for (const h of local.halves) {
      const o = el('option', null, 'Half ' + h.half);
      o.value = String(h.half);
      halfSel.append(o);
    }
    halfSel.addEventListener('change', () => {
      const h = local.halves.find((x) => String(x.half) === halfSel.value);
      const ko = (manifest.kickoffs || []).find((k) => k.half === h.half);
      loop = null;
      btnKeep.hidden = true;
      seek(ko ? ko.frame : h.frame_start);
    });
    source = local;
    note.textContent = 'real tracking, local bundle';
    buildJumpMenu();
  } else {
    halfSel.hidden = true;
    jumpBtn.hidden = true;
    note.innerHTML = '';
    const a = el('span', null, 'synthetic illustration, real tracking is never published');
    const b2 = el('span', 'rp-hint', 'with the match data, this command builds a replay of the real tracking on your computer: ');
    const code = el('code', null, 'python -m tools.viz.build_replay');
    b2.append(code);
    note.append(a, b2);
    try {
      scenes = createScenes({ config: bundle.config, pitchM, fps });
    } catch (err) {
      msg.textContent = 'The scenes need config.json: ' + err.message;
    }
  }

  // layer menu
  function buildLayerMenu() {
    pop.textContent = '';
    pop.append(el('div', 'rp-pop-title', 'Layers'));
    for (const key of Object.keys(DEFAULT_LAYERS)) {
      const lab = el('label', 'rp-check');
      const box = el('input');
      box.type = 'checkbox';
      box.checked = !!layers[key];
      box.addEventListener('change', () => {
        layers[key] = box.checked;
        try { localStorage.setItem('rp-layers-v2', JSON.stringify(layers)); } catch (e) { /* storage blocked */ }
        dirty = true;
      });
      lab.append(box, document.createTextNode(' ' + LAYER_NAMES[key]));
      pop.append(lab);
    }
  }

  function buildJumpMenu() {
    jumpBuilder = () => {
      pop.textContent = '';
      const tr = pick(bundle, 'transitions') || {};
      const gts = tr.gt_transitions || tr.gt || [];
      const sec = (title) => { const d = el('div', 'rp-pop-sec'); d.append(el('div', 'rp-pop-title', title)); pop.append(d); return d; };
      const row = (parent, text, cls, onClick) => {
        const b = el('button', 'rp-row ' + (cls || ''), text);
        b.type = 'button';
        b.addEventListener('click', () => { onClick(); pop.hidden = true; });
        parent.append(b);
      };
      const s1 = sec('Public transitions (' + gts.length + ' GT, ' + (tr.spurious || []).length + ' spurious)');
      for (const g of gts) {
        const lat = g.latency == null ? 'none' : (g.latency > 0 ? '+' : '') + g.latency;
        row(s1, g.gt_frame + '  ' + g.from + '→' + g.to + '  ' + g.status + '  ' + lat, 'is-' + g.status,
          () => showCase({ frame: g.gt_frame, from: g.gt_frame - 75, to: g.gt_frame + 75, insets: insetMode }));
      }
      for (const sp of tr.spurious || []) {
        row(s1, sp.our_frame + '  ' + sp.from + '→' + sp.to + '  spurious', 'is-spurious',
          () => showCase({ frame: sp.our_frame, from: sp.our_frame - 75, to: sp.our_frame + 75, insets: insetMode }));
      }
      const byKind = (rows, idx) => {
        const m = new Map();
        for (const r of rows) m.set(r[idx], (m.get(r[idx]) || 0) + 1);
        return Array.from(m.entries()).sort((p, q) => q[1] - p[1]);
      };
      const next = (rows, test) => {
        const k = lowerBound(rows, frame + 1);
        for (let i = k; i < rows.length; i++) if (test(rows[i])) return rows[i][0];
        for (let i = 0; i < k; i++) if (test(rows[i])) return rows[i][0];
        return null;
      };
      const go = (f) => { if (f != null) showCase({ frame: f, from: f - 50, to: f + 100, insets: insetMode }); };
      const s2 = sec('Events by kind (next after the playhead)');
      for (const [kind, c] of byKind(local.events, 1)) row(s2, kind + '  ' + c, '', () => go(next(local.events, (r) => r[1] === kind)));
      const s3 = sec('Stoppages and restarts');
      for (const [mech, c] of byKind(local.onsets, 1)) row(s3, 'onset ' + mech + '  ' + c, '', () => go(next(local.onsets, (r) => r[1] === mech)));
      for (const [kind, c] of byKind(local.restarts, 2)) row(s3, 'restart ' + kind + '  ' + c, '', () => go(next(local.restarts, (r) => r[2] === kind)));
      const goals = findGoals();
      const s4 = sec('Goals');
      if (goals.length) goals.forEach((g, k) => row(s4, 'goal ' + (k + 1) + '  ' + g.label, '', () => go(g.frame)));
      else s4.append(el('div', 'rp-pop-note', 'The bundle does not list the goals.'));
    };
  }

  /** Goal frames, when the bundle exposes them (manifest `goals` or sanity set pieces). */
  function findGoals() {
    const out = [];
    const g = manifest.goals || (bundle.manifest && bundle.manifest.goals);
    if (Array.isArray(g)) for (const x of g) out.push({ frame: x.frame != null ? x.frame : x, label: x.team || '' });
    const sp = bundle.sanity && bundle.sanity.ours && bundle.sanity.ours.set_pieces;
    if (!out.length && sp && Array.isArray(sp.goals)) for (const x of sp.goals) out.push({ frame: x.frame || x.start, label: x.team || '' });
    return out;
  }

  layersBtn.addEventListener('click', () => {
    const open = pop.hidden || pop.dataset.kind !== 'layers';
    pop.dataset.kind = 'layers';
    if (open) { buildLayerMenu(); pop.hidden = false; } else pop.hidden = true;
  });
  jumpBtn.addEventListener('click', () => {
    const open = pop.hidden || pop.dataset.kind !== 'jump';
    pop.dataset.kind = 'jump';
    if (open && jumpBuilder) { jumpBuilder(); pop.hidden = false; } else pop.hidden = true;
  });
  document.addEventListener('pointerdown', onDocPointer, true);
  function onDocPointer(e) {
    if (!pop.hidden && !pop.contains(e.target) && e.target !== layersBtn && e.target !== jumpBtn) pop.hidden = true;
  }

  // -- playback ----------------------------------------------------------------------------

  function length() {
    return local ? local.nFrames : scene ? scene.n : 1;
  }

  function setSpeed(s) {
    speed = s;
    speedBtns.forEach((b, k) => b.classList.toggle('is-on', SPEEDS[k] === s));
    if (local) state.set({ speed: s });
  }

  function play() {
    if (playing) return;
    playing = true;
    acc = 0;
    btnPlay.textContent = 'Pause';
    btnPlay.classList.add('is-on');
    if (local) state.set({ playing: true });
  }

  function pause() {
    if (!playing) return;
    playing = false;
    btnPlay.textContent = 'Play';
    btnPlay.classList.remove('is-on');
    if (local) state.set({ playing: false });
  }

  function seek(f, keepAcc) {
    frame = Math.max(0, Math.min(length() - 1, Math.round(f)));
    if (!keepAcc) acc = 0;
    dirty = true;
  }

  function step(df) {
    pause();
    seek(frame + df);
  }

  function advance(k) {
    let f = frame + k;
    if (loop) {
      const len = loop[1] - loop[0];
      if (f >= loop[1]) f = loop[0] + ((f - loop[0]) % len);
    } else if (scene && f >= scene.n) {
      if (playlist.length > 1) { nextScene(); return; }
      f = 0;
    }
    seek(f, true);
  }

  function nextScene() {
    plIndex = (plIndex + 1) % playlist.length;
    loadScene(playlist[plIndex]);
  }

  btnPlay.addEventListener('click', () => (playing ? pause() : play()));
  setSpeed(1);

  function tick(now) {
    if (destroyed) return;
    raf = requestAnimationFrame(tick);
    const dt = last ? Math.min(1000, now - last) : 0;
    last = now;
    if (!visible || document.hidden) return;
    if (playing && source) {
      acc += (dt / 1000) * fps * speed;
      const k = Math.floor(acc);
      if (k > 0) { acc -= k; advance(k); }
    }
    if (dirty) render();
  }

  function render() {
    if (!source) return;
    const fs = local ? local.frameAt(frame, insetMode) : scene ? scene.frameAt(frame) : null;
    if (!fs) {
      msg.textContent = 'Loading the real tracking of half ' + (local ? local.halfOf(frame) : '') + '…';
      if (local) local.openHalf(local.halfOf(frame)).then(() => { msg.textContent = ''; dirty = true; }).catch((err) => { msg.textContent = 'Could not load the local bundle: ' + err.message; });
      dirty = false;
      return;
    }
    dirty = false;
    msg.textContent = '';
    const ms = pitch.draw(fs, layers, params);
    drawTimes.push(ms);
    if (drawTimes.length > 120) drawTimes.shift();
    updateHud(fs);
    drawStrip();
    if (local) {
      lastWritten = frame;
      state.set({ frame, half: fs.half });
    }
    if (scene) caption.textContent = scene.caption(frame);
  }

  function setChip(cell, letter) {
    cell.textContent = letter == null ? '–' : letter;
    cell.dataset.label = letter == null ? '' : letter;
  }

  function fmt(v, d) {
    return typeof v === 'number' && isFinite(v) ? v.toFixed(d) : '–';
  }

  function updateHud(fs) {
    const h = fs.hud || {};
    if (local) {
      hudCells.frame.textContent = String(frame);
      hudCells.clock.textContent = local.clock(frame);
      hudCells.half.textContent = String(fs.half);
      if (String(fs.half) !== halfSel.value) halfSel.value = String(fs.half);
    } else {
      hudCells.frame.textContent = String(frame) + ' / ' + (scene ? scene.n : 0);
      const s = frame / fps;
      hudCells.clock.textContent = '00:' + String(Math.floor(s)).padStart(2, '0') + '.' + String(frame % fps).padStart(2, '0');
      hudCells.half.textContent = '–';
    }
    setChip(hudCells.ours, fs.labels && fs.labels.ours);
    setChip(hudCells.gt, fs.labels && fs.labels.gt);
    setChip(hudCells.base, fs.labels && fs.labels.baseline);
    const ow = h.owner;
    setChip(hudCells.owner, ow === 0 ? 'H' : ow === 1 ? 'A' : null);
    const pd = typeof h.pDead === 'number' ? h.pDead : fs.pDead || 0;
    hudCells.pdead.textContent = pd.toFixed(2);
    hudCells.pbar.style.transform = 'scaleX(' + Math.max(0, Math.min(1, pd)).toFixed(3) + ')';
    hudCells.speed.textContent = fmt(h.speed, 1) + ' m/s';
    hudCells.dh.textContent = fmt(h.dH, 2) + ' m';
    hudCells.da.textContent = fmt(h.dA, 2) + ' m';
    if (debug) {
      const avg = drawTimes.reduce((p, q) => p + q, 0) / drawTimes.length;
      note.dataset.debug = 'draw ' + avg.toFixed(2) + ' ms';
    }
  }

  // -- the seek strip ----------------------------------------------------------------------

  const stripState = { lo: 0, hi: 1, drag: false };
  function stripRange() {
    if (local) {
      const half = STRIP_HALF_S * fps;
      if (stripState.drag) return [stripState.lo, stripState.hi];
      return [frame - half, frame + half];
    }
    return [0, scene ? scene.n : 1];
  }

  function drawStrip() {
    const rect = stripCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, rect.width), h = Math.max(1, rect.height);
    if (stripCanvas.width !== Math.round(w * dpr) || stripCanvas.height !== Math.round(h * dpr)) {
      stripCanvas.width = Math.round(w * dpr);
      stripCanvas.height = Math.round(h * dpr);
    }
    const g = stripCanvas.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const [lo, hi] = stripRange();
    const X = (f) => ((f - lo) / (hi - lo)) * w;
    const lanes = local ? [['ours', local.ours], ['GT', local.gt], ['events', null]] : [['ours', null], ['events', null]];
    const laneH = (h - 4) / lanes.length;
    const col = { H: tokens.h, A: tokens.a, D: tokens.d, X: tokens.hairline };
    lanes.forEach(([name, runs], k) => {
      const y = 2 + k * laneH;
      g.fillStyle = 'rgba(232,236,243,0.04)';
      g.fillRect(0, y, w, laneH - 2);
      if (name === 'events') {
        const marks = local
          ? local.events.slice(lowerBound(local.events, lo), lowerBound(local.events, hi)).map((e) => ({ i: e[0], color: 'ev-' + e[1] }))
            .concat(local.onsets.slice(lowerBound(local.onsets, lo), lowerBound(local.onsets, hi)).map((o) => ({ i: o[0], color: 'onset' })))
            .concat(local.restarts.slice(lowerBound(local.restarts, lo), lowerBound(local.restarts, hi)).map((r) => ({ i: r[0], color: 'restart' })))
          : (scene ? scene.timeline.marks : []);
        for (const m of marks) {
          g.fillStyle = tokens[m.color] || tokens.text;
          g.fillRect(X(m.i) - 1, y + 2, 2, laneH - 6);
        }
        return;
      }
      if (local) {
        if (!runs) return;
        for (let i = Math.max(0, lowerBound(runs, lo) - 1); i < runs.length && runs[i][0] < hi; i++) {
          const r = runs[i];
          if (r[1] < lo) continue;
          g.fillStyle = col[r[2]] || tokens.d;
          g.fillRect(X(r[0]), y, Math.max(1, X(r[1]) - X(r[0])), laneH - 2);
        }
      } else if (scene) {
        const labs = scene.timeline.labels;
        let s = 0;
        for (let i = 1; i <= labs.length; i++) {
          if (i === labs.length || labs[i] !== labs[s]) {
            g.fillStyle = col[LETTER[labs[s]]];
            g.fillRect(X(s), y, Math.max(1, X(i) - X(s)), laneH - 2);
            s = i;
          }
        }
      }
    });
    if (loop) {
      g.fillStyle = 'rgba(255,209,102,0.10)';
      g.fillRect(X(loop[0]), 0, X(loop[1]) - X(loop[0]), h);
    }
    const px = X(frame);
    g.fillStyle = tokens.text;
    g.fillRect(px - 1, 0, 2, h);
    stripLegend.textContent = local ? 'our labels, ground truth and events, ±' + STRIP_HALF_S + ' s' : 'the rule engine’s labels and events, whole scene';
  }

  function stripFrame(e) {
    const rect = stripCanvas.getBoundingClientRect();
    const [lo, hi] = stripRange();
    return lo + ((e.clientX - rect.left) / Math.max(1, rect.width)) * (hi - lo);
  }

  stripCanvas.addEventListener('pointerdown', (e) => {
    const [lo, hi] = stripRange();
    stripState.lo = lo;
    stripState.hi = hi;
    stripState.drag = true;
    stripCanvas.setPointerCapture(e.pointerId);
    loop = null;
    btnKeep.hidden = true;
    seek(stripFrame(e));
  });
  stripCanvas.addEventListener('pointermove', (e) => { if (stripState.drag) seek(stripFrame(e)); });
  const endDrag = () => { stripState.drag = false; dirty = true; };
  stripCanvas.addEventListener('pointerup', endDrag);
  stripCanvas.addEventListener('pointercancel', endDrag);

  // -- scenes and cases --------------------------------------------------------------------

  function renderChips() {
    chipBox.textContent = '';
    if (playlist.length < 2) return;
    playlist.forEach((id, k) => {
      const b = el('button', 'rp-chip' + (k === plIndex ? ' is-on' : ''), SCENE_NAMES[id] || id);
      b.type = 'button';
      b.addEventListener('click', () => { plIndex = k; loadScene(id); play(); });
      chipBox.append(b);
    });
  }

  function loadScene(id) {
    if (!scenes) return;
    try {
      scene = scenes.get(id);
    } catch (err) {
      msg.textContent = 'Scene ' + id + ' failed: ' + err.message;
      console.error(err);
      return;
    }
    source = scene;
    loop = null;
    seek(reduced ? scene.keyFrame : 0);
    renderChips();
  }

  /** Synthetic mode: play a playlist of scenes. */
  function showScenes(ids, o) {
    if (!scenes) return;
    playlist = ids.slice();
    plIndex = (o && o.index) || 0;
    loadScene(playlist[plIndex]);
    if (!reduced && (!o || o.autoplay !== false)) play();
    else pause();
  }

  /** Local mode: seek to a case and loop [from, to). */
  function showCase(c) {
    if (!local) return;
    insetMode = c.insets || null;
    const from = Math.max(0, c.from != null ? c.from : c.frame - 50);
    const to = Math.min(local.nFrames, c.to != null ? c.to : c.frame + 100);
    loop = c.loop === false ? null : [from, to];
    btnKeep.hidden = !loop;
    caption.textContent = c.title || '';
    seek(reduced ? c.frame : from);
    if (!reduced && c.autoplay !== false) play();
    local.openHalf(local.halfOf(c.frame)).then(() => { dirty = true; }).catch((err) => { msg.textContent = 'Could not load the local bundle: ' + err.message; });
  }

  // -- keyboard ----------------------------------------------------------------------------

  function onKey(e) {
    if (!active && !container.contains(document.activeElement)) return;
    const t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA')) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === ' ' || e.code === 'Space') { e.preventDefault(); playing ? pause() : play(); }
    else if (e.key === '[') { const k = SPEEDS.indexOf(speed); setSpeed(SPEEDS[Math.max(0, k - 1)]); }
    else if (e.key === ']') { const k = SPEEDS.indexOf(speed); setSpeed(SPEEDS[Math.min(SPEEDS.length - 1, k + 1)]); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); step(e.shiftKey ? -fps : -1); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); step(e.shiftKey ? fps : 1); }
  }
  document.addEventListener('keydown', onKey);

  // -- external state ----------------------------------------------------------------------

  const offs = [];
  if (local) {
    offs.push(state.on('frame', (v) => {
      if (v === lastWritten || typeof v !== 'number') return;
      loop = null;
      btnKeep.hidden = true;
      seek(v);
    }));
    offs.push(state.on('playing', (v) => { if (v && !playing) play(); else if (!v && playing) pause(); }));
    offs.push(state.on('speed', (v) => { if (SPEEDS.indexOf(v) >= 0 && v !== speed) setSpeed(v); }));
  }

  // -- size, visibility, theme -------------------------------------------------------------

  /**
   * Fit the canvas inside the stage at the pitch aspect: the largest box that fits both the stage
   * width and the stage height, centred. The top row reads the offsets through --cx and --cy.
   * The camera zooms inside this box and never changes its size.
   */
  function fitCanvas() {
    const w = stage.clientWidth, h = stage.clientHeight;
    if (!w || !h) return;
    let cw = w, ch = w / pitch.aspect;
    if (ch > h) { ch = h; cw = h * pitch.aspect; }
    cw = Math.floor(cw);
    ch = Math.floor(ch);
    const x = Math.floor((w - cw) / 2), y = Math.floor((h - ch) / 2);
    canvas.style.width = cw + 'px';
    canvas.style.height = ch + 'px';
    canvas.style.left = x + 'px';
    canvas.style.top = y + 'px';
    stage.style.setProperty('--cx', x + 'px');
    stage.style.setProperty('--cy', y + 'px');
  }

  const ro = new ResizeObserver(() => { fitCanvas(); pitch.resize(); dirty = true; });
  ro.observe(stage);
  ro.observe(stripCanvas);
  const io = new IntersectionObserver((entries) => { for (const en of entries) visible = en.isIntersecting; if (visible) dirty = true; });
  io.observe(container);
  const mo = new MutationObserver(() => {
    tokens = readTokens();
    pitch.setTokens(tokens);
    dirty = true;
  });
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'data-theme', 'style', 'data-arena'] });

  raf = requestAnimationFrame(tick);

  return {
    el: container,
    showScenes,
    showCase,
    play,
    pause,
    seek(f) { loop = null; btnKeep.hidden = true; seek(f); },
    setActive(v) { active = !!v; if (!active) pop.hidden = true; },
    setInsets(m) { insetMode = m; dirty = true; },
    get frame() { return frame; },
    get playing() { return playing; },
    get mode() { return local ? 'local' : 'synthetic'; },
    get scene() { return scene; },
    /** Draw n consecutive frames now and time them (frame state build and pitch draw). */
    bench(n) {
      const build = [], draw = [];
      const start = frame;
      for (let k = 0; k < (n || 100); k++) {
        const t0 = performance.now();
        const fs = local ? local.frameAt(start + k, insetMode) : scene ? scene.frameAt((start + k) % scene.n) : null;
        const t1 = performance.now();
        if (!fs) continue;
        draw.push(pitch.draw(fs, layers, params));
        build.push(t1 - t0);
      }
      const q = (v, p) => v.slice().sort((x, y) => x - y)[Math.min(v.length - 1, Math.floor(v.length * p))] || 0;
      const mean = (v) => v.reduce((x, y) => x + y, 0) / Math.max(1, v.length);
      dirty = true;
      return { n: draw.length, drawMean: mean(draw), drawP95: q(draw, 0.95), buildMean: mean(build), buildP95: q(build, 0.95),
        canvas: [canvas.width, canvas.height] };
    },
    stats() {
      const v = drawTimes.slice().sort((p, q) => p - q);
      return { n: v.length, mean: v.reduce((p, q) => p + q, 0) / Math.max(1, v.length), p95: v[Math.floor(v.length * 0.95)] || 0, max: v[v.length - 1] || 0,
        canvas: [canvas.width, canvas.height] };
    },
    destroy() {
      destroyed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      mo.disconnect();
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('pointerdown', onDocPointer, true);
      for (const off of offs) off();
    },
  };
}
