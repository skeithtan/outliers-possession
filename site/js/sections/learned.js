/**
 * Section "From tracking to labels": the engine step by step.
 *
 * `mount(el, ctx)` builds the section inside the empty
 * `<section id="learned" class="story-section" data-section="learned">`:
 *   header                               kicker, title, lede
 *   .scrolly > .steps > .step[0..6]      the copy, one step per stage of the engine
 *   .scrolly > .figure                   sticky: the pipeline diagram and one panel per step
 *
 * The scroller sets `state.activeStep = {section: 'learned', step, progress}`. On a new step the
 * section lights the stages of the pipeline and shows the panel of the step. The panels are
 * diagrams with synthetic numbers: the hosted page never holds tracking or full-match labels.
 * The decoder panel runs the engine's three-state Viterbi (engine/learned.py viterbi()) on
 * synthetic probabilities, with the change costs of `bundle.config`.
 *
 * Value chips show the value of a constant of engine/config.py. The tooltip names the constant.
 *
 * @module sections/learned
 */

import { configAccessor } from '../synthetic.js';
import { reducedMotion } from '../util/dom.js';
import { fixed } from '../util/format.js';

const SVGNS = 'http://www.w3.org/2000/svg';

function h(tag, attrs, children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k === 'html') e.innerHTML = v;
    else e.setAttribute(k, v);
  }
  for (const c of [].concat(children || [])) if (c !== null && c !== undefined) e.append(c);
  return e;
}

function s(tag, attrs, children) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'text') e.textContent = v;
    else e.setAttribute(k, v);
  }
  for (const c of [].concat(children || [])) if (c) e.append(c);
  return e;
}

/* -- the pipeline ----------------------------------------------------------------------------- */

// Two layouts of the same diagram: a wide one with a second line of text in each box, and a
// narrow one for phones. Nodes are centred at (x, y) with width w and height h.
const WIDE = {
  w: 904, h: 236, rx: 10, label: 14, sub: 10.5, head: 6,
  nodes: {
    tracking: { x: 62, y: 92, w: 104, h: 52, label: 'tracking', sub: 'tactical ATD' },
    clean: { x: 184, y: 92, w: 104, h: 52, label: 'clean', sub: 'ball, goalkeepers' },
    features: { x: 306, y: 92, w: 104, h: 52, label: 'features', sub: '50 per frame' },
    network: { x: 452, y: 40, w: 140, h: 52, label: 'the network', sub: 'GRU, 5 fps' },
    perframe: { x: 452, y: 144, w: 140, h: 52, label: 'per-frame model', sub: 'trees, 25 fps' },
    blend: { x: 598, y: 92, w: 104, h: 52, label: 'blend', sub: '' },
    decoder: { x: 720, y: 92, w: 104, h: 52, label: 'decoder', sub: '' },
    labels: { x: 842, y: 92, w: 104, h: 52, label: 'labels', sub: 'H, A, D' },
    rule: { x: 520, y: 206, w: 220, h: 38, label: 'the rule engine', sub: 'the fallback, one setting away', dashed: true },
  },
  paths: {
    'clean>rule': { d: 'M184 118 V206 H405', end: [407, 206], dir: 'r' },
    'rule>labels': { d: 'M630 206 H842 V122', end: [842, 120], dir: 'u' },
  },
};

const NARROW = {
  w: 360, h: 110, rx: 6, label: 9.5, sub: 0, head: 4,
  nodes: {
    clean: { x: 27, y: 40, w: 46, h: 26, label: 'clean' },
    features: { x: 84, y: 40, w: 48, h: 26, label: 'features' },
    network: { x: 152, y: 15, w: 58, h: 24, label: 'network' },
    perframe: { x: 152, y: 65, w: 58, h: 24, label: 'per-frame' },
    blend: { x: 220, y: 40, w: 46, h: 26, label: 'blend' },
    decoder: { x: 276, y: 40, w: 48, h: 26, label: 'decoder' },
    labels: { x: 332, y: 40, w: 46, h: 26, label: 'labels' },
    rule: { x: 180, y: 96, w: 150, h: 22, label: 'the rule engine, the fallback', dashed: true },
  },
  paths: {
    'clean>rule': { d: 'M27 53 V96 H103', end: [104, 96], dir: 'r' },
    'rule>labels': { d: 'M255 96 H332 V56', end: [332, 54], dir: 'u' },
  },
};

const EDGES = ['tracking>clean', 'clean>features', 'features>network', 'features>perframe',
  'network>blend', 'perframe>blend', 'blend>decoder', 'decoder>labels', 'clean>rule', 'rule>labels'];

/** Stages and edges lit at each step. */
const LIT = [
  { nodes: ['tracking', 'clean'], edges: ['tracking>clean'] },
  { nodes: ['features'], edges: ['clean>features'] },
  { nodes: ['network'], edges: ['features>network'] },
  { nodes: ['perframe'], edges: ['features>perframe'] },
  { nodes: ['blend'], edges: ['network>blend', 'perframe>blend'] },
  { nodes: ['decoder', 'labels'], edges: ['blend>decoder', 'decoder>labels'] },
  { nodes: ['rule', 'labels'], edges: ['clean>rule', 'rule>labels'] },
];

function edgeGeom(L, id) {
  if (L.paths[id]) return L.paths[id];
  const [from, to] = id.split('>');
  const a = L.nodes[from], b = L.nodes[to];
  if (!a || !b) return null;
  const x0 = a.x + a.w / 2, y0 = a.y, x1 = b.x - b.w / 2 - 2, y1 = b.y;
  const mx = (x0 + x1) / 2;
  const d = y0 === y1 ? 'M' + x0 + ' ' + y0 + ' H' + (x1 - 1)
    : 'M' + x0 + ' ' + y0 + ' C' + mx + ' ' + y0 + ' ' + mx + ' ' + y1 + ' ' + (x1 - 1) + ' ' + y1;
  return { d, end: [x1, y1], dir: 'r' };
}

function headPath(end, dir, k) {
  const [x, y] = end;
  if (dir === 'u') return 'M' + (x - k * 0.75) + ' ' + (y + k) + ' L' + x + ' ' + y + ' L' + (x + k * 0.75) + ' ' + (y + k);
  return 'M' + (x - k) + ' ' + (y - k * 0.75) + ' L' + x + ' ' + y + ' L' + (x - k) + ' ' + (y + k * 0.75);
}

function buildPipeline(L, cls, subs) {
  const root = s('svg', { viewBox: '0 0 ' + L.w + ' ' + L.h, class: 'lp ' + cls, role: 'img',
    'aria-label': 'The engine: tracking, cleaning, 50 features, the network and the per-frame model, the blend, the decoder, the labels. The rule engine is the fallback.' });
  const edges = {};
  for (const id of EDGES) {
    const g = edgeGeom(L, id);
    if (!g) continue;
    const grp = s('g', { class: 'lp-edge' + (L.paths[id] ? ' lp-edge--fallback' : ''), 'data-edge': id });
    grp.append(s('path', { d: g.d, class: 'lp-line' }), s('path', { d: headPath(g.end, g.dir, L.head), class: 'lp-head' }));
    root.append(grp);
    edges[id] = grp;
  }
  const nodes = {};
  for (const [id, n] of Object.entries(L.nodes)) {
    const sub = subs[id] !== undefined ? subs[id] : n.sub;
    const g = s('g', { class: 'lp-node' + (n.dashed ? ' lp-node--fallback' : ''), 'data-node': id, tabindex: '0', role: 'button',
      'aria-label': n.label });
    g.append(s('rect', { x: n.x - n.w / 2, y: n.y - n.h / 2, width: n.w, height: n.h, rx: L.rx, class: 'lp-box' }));
    const two = L.sub && sub;
    g.append(s('text', { x: n.x, y: two ? n.y - 3 : n.y + L.label * 0.36, class: 'lp-label', 'text-anchor': 'middle',
      'font-size': L.label, text: n.label }));
    if (two) g.append(s('text', { x: n.x, y: n.y + 13, class: 'lp-sub', 'text-anchor': 'middle', 'font-size': L.sub, text: sub }));
    root.append(g);
    nodes[id] = g;
  }
  return { root, nodes, edges };
}

/* -- synthetic data ----------------------------------------------------------------------------- */

/** A small deterministic generator, so that the page looks the same on every load. */
function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

const sig = (x) => 1 / (1 + Math.exp(-x));

function softmax(z) {
  const m = Math.max(z[0], z[1], z[2]);
  const e = z.map((v) => Math.exp(v - m));
  const t = e[0] + e[1] + e[2];
  return e.map((v) => v / t);
}

/**
 * Seven synthetic seconds at 25 frames a second: H has the ball, a player of A comes close for a
 * moment, A wins the ball at 3 s, the ball goes out at 5.2 s. The network's probabilities are
 * smooth and enter dead ball late. The per-frame model's are sharp and noisy. Rows are [H, A, D].
 */
function synthetic(weightPf) {
  const fps = 25, n = 7 * fps;
  const rnd = mulberry32(7);
  const gauss = () => {
    let u = 0, v = 0;
    while (u === 0) u = rnd();
    while (v === 0) v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const tHand = 3 * fps, tOut = 5.2 * fps, tNear = 1.5 * fps, nearLen = 9;
  let drift = [0, 0, 0];
  const net = [], pf = [];
  for (let f = 0; f < n; f++) {
    drift = drift.map((v) => 0.9 * v + 0.12 * gauss());
    const a = sig((f - tHand) / (0.22 * fps));
    const d = sig((f - (tOut + 0.4 * fps)) / (0.25 * fps));
    const near = Math.exp(-((f - (tNear + nearLen / 2)) ** 2) / (2 * (0.2 * fps) ** 2));
    net.push(softmax([1.6 - 3.4 * a + drift[0], -1.6 + 3.4 * a + 3.4 * near + drift[1], -2.6 + 6.2 * d + drift[2]]));
    const a2 = sig((f - tHand) / (0.05 * fps));
    const d2 = sig((f - tOut) / (0.05 * fps));
    const touch = f >= tNear && f < tNear + nearLen ? 1 : 0;
    pf.push(softmax([1.8 - 3.8 * a2 + 1.1 * gauss(), -1.8 + 3.8 * a2 + 5 * touch + 1.1 * gauss(), -2.8 + 6.4 * d2 + 1.1 * gauss()]));
  }
  const blend = net.map((p, i) => p.map((v, k) => (1 - weightPf) * v + weightPf * pf[i][k]));
  const t0 = Math.round(tNear);
  return { fps, n, net, pf, blend, near: [t0 - 6, t0 + nearLen + 6] };
}

/** The engine's decoder (engine/learned.py viterbi()): best path of 0 H, 1 A, 2 D. */
function viterbi(probs, costHA, costAD) {
  const n = probs.length;
  const T = [[0, costHA, costAD], [costHA, 0, costAD], [costAD, costAD, 0]];
  const lp = probs.map((p) => p.map((v) => Math.log(Math.max(v, 1e-4))));
  const back = new Int8Array(n * 3);
  let score = lp[0].slice();
  for (let t = 1; t < n; t++) {
    const next = [0, 0, 0];
    for (let j = 0; j < 3; j++) {
      let bi = 0, bv = -Infinity;
      for (let i = 0; i < 3; i++) {
        const v = score[i] - T[i][j];
        if (v > bv) { bv = v; bi = i; }
      }
      back[t * 3 + j] = bi;
      next[j] = bv + lp[t][j];
    }
    score = next;
  }
  const path = new Int8Array(n);
  let best = 0;
  for (let i = 1; i < 3; i++) if (score[i] > score[best]) best = i;
  path[n - 1] = best;
  for (let t = n - 1; t > 0; t--) path[t - 1] = back[t * 3 + path[t]];
  return path;
}

const argmax = (p) => (p[0] >= p[1] && p[0] >= p[2] ? 0 : p[1] >= p[2] ? 1 : 2);
const changes = (a) => { let c = 0; for (let i = 1; i < a.length; i++) if (a[i] !== a[i - 1]) c++; return c; };
const LETTER = 'HAD';
const CLS = ['h', 'a', 'd'];

/* -- small chart pieces ------------------------------------------------------------------------- */

/** Stacked probability areas: H from the bottom, D in the middle, A on top. */
function areaChart(probs, cls) {
  const n = probs.length;
  const svg = s('svg', { viewBox: '0 0 ' + (n - 1) + ' 100', preserveAspectRatio: 'none', class: 'lw-area ' + (cls || ''), 'aria-hidden': 'true' });
  const yH = probs.map((p) => 100 - 100 * p[0]);
  const yHD = probs.map((p) => 100 - 100 * (p[0] + p[2]));
  const line = (ys) => ys.map((y, i) => (i ? 'L' : 'M') + i + ' ' + y.toFixed(2)).join(' ');
  const back = (ys) => ys.map((y, i) => 'L' + (ys.length - 1 - i) + ' ' + ys[ys.length - 1 - i].toFixed(2)).join(' ');
  svg.append(
    s('path', { d: line(yH) + ' L' + (n - 1) + ' 100 L0 100 Z', class: 'lw-a-h' }),
    s('path', { d: line(yHD) + ' ' + back(yH) + ' Z', class: 'lw-a-d' }),
    s('path', { d: 'M0 0 L' + (n - 1) + ' 0 ' + back(yHD) + ' Z', class: 'lw-a-a' }),
  );
  return svg;
}

/** One strip of labels, a rectangle for each run. */
function labelStrip(codes, cls) {
  const n = codes.length;
  const svg = s('svg', { viewBox: '0 0 ' + n + ' 10', preserveAspectRatio: 'none', class: 'lw-strip ' + (cls || ''), 'aria-hidden': 'true' });
  fillStrip(svg, codes);
  return svg;
}

function fillStrip(svg, codes) {
  svg.textContent = '';
  let start = 0;
  for (let i = 1; i <= codes.length; i++) {
    if (i === codes.length || codes[i] !== codes[start]) {
      svg.append(s('rect', { x: start, y: 0, width: i - start, height: 10, class: 'lw-s-' + CLS[codes[start]] }));
      start = i;
    }
  }
}

function timeAxis(seconds) {
  const ax = h('div', { class: 'lw-axis', 'aria-hidden': 'true' });
  for (let t = 0; t <= seconds; t++) {
    ax.append(h('span', { style: 'left:' + (100 * t / seconds).toFixed(3) + '%', text: t + (t === seconds ? ' s' : '') }));
  }
  return ax;
}

function panelHead(k, title) {
  return h('p', { class: 'lw-ph' }, [h('b', { class: 'tnum', text: String(k + 1) }), document.createTextNode(title)]);
}

/* -- the panels --------------------------------------------------------------------------------- */

function panelClean(c) {
  // A synthetic ball: a curved pass, sampled every frame, with three tracker faults.
  const pts = [];
  for (let i = 0; i < 46; i++) {
    const t = i / 45;
    pts.push({ x: 24 + 352 * t, y: 176 - 104 * Math.sin(Math.PI * t * 0.82) + 14 * t, kind: 'kept' });
  }
  pts[9] = { ...pts[9], y: pts[9].y + 52, kind: 'jump' };
  for (const i of [21, 22, 23]) pts[i] = { ...pts[i], y: pts[i].y - 44 - 4 * (i - 21), x: pts[i].x + 10, kind: 'jump' };
  for (const i of [31, 32, 33, 34, 35]) pts[i] = { ...pts[i], kind: 'gap' };
  pts[32] = { ...pts[32], x: pts[32].x + 6, y: pts[32].y + 46, kind: 'short' };
  pts[33] = { ...pts[33], x: pts[33].x + 10, y: pts[33].y + 50, kind: 'short' };
  const svg = s('svg', { viewBox: '0 0 400 220', class: 'lw-clean', 'aria-hidden': 'true' });
  const kept = pts.filter((p) => p.kind === 'kept');
  const d = kept.map((p, i) => (i ? 'L' : 'M') + p.x.toFixed(1) + ' ' + p.y.toFixed(1)).join(' ');
  svg.append(s('path', { d, class: 'lw-clean-path', pathLength: '1' }));
  for (const p of pts) {
    if (p.kind === 'gap') continue;
    if (p.kind === 'kept') svg.append(s('circle', { cx: p.x, cy: p.y, r: 3.1, class: 'lw-dot' }));
    else {
      const k = 4.2;
      svg.append(s('path', { d: 'M' + (p.x - k) + ' ' + (p.y - k) + ' L' + (p.x + k) + ' ' + (p.y + k) + ' M' + (p.x - k) + ' ' + (p.y + k) + ' L' + (p.x + k) + ' ' + (p.y - k),
        class: 'lw-x lw-x--' + p.kind }));
    }
  }
  const jump = c('BALL_MAX_JUMP_M', 'm'), run = c('BALL_MIN_RUN', 'frames');
  const legend = h('ul', { class: 'lw-legend' }, [
    h('li', {}, [h('i', { class: 'lw-key lw-key--dot' }), document.createTextNode('a ball detection, kept')]),
    h('li', {}, [h('i', { class: 'lw-key lw-key--jump' }), document.createTextNode('more than '), jump, document.createTextNode(' from the last good frame: removed')]),
    h('li', {}, [h('i', { class: 'lw-key lw-key--short' }), document.createTextNode('a visible run under '), run, document.createTextNode(': dropped')]),
    h('li', {}, [h('i', { class: 'lw-key lw-key--line' }), document.createTextNode('the cleaned ball that the models read')]),
  ]);
  const checks = h('ul', { class: 'lw-checks' }, [
    h('li', { text: 'player speed from the positions, where the tracker gives none' }),
    h('li', { text: 'ball cleaning' }),
    h('li', { text: 'goalkeeper-swap correction' }),
  ]);
  return [panelHead(0, 'Clean the tracking: a synthetic ball'), h('div', { class: 'lw-fill lw-fill--clean' }, [svg]), legend, checks];
}

function panelFeatures() {
  const rnd = mulberry32(11);
  const cols = 44;
  const groups = [{ n: 18, cls: 'g' }, { n: 16, cls: 'h' }, { n: 16, cls: 'a' }];
  const rowH = 4, gap = 6;
  const height = 50 * rowH + 2 * gap;
  const svg = s('svg', { viewBox: '0 0 ' + cols * 4 + ' ' + height, preserveAspectRatio: 'none', class: 'lw-heat', 'aria-hidden': 'true' });
  let y = 0;
  for (const g of groups) {
    for (let r = 0; r < g.n; r++) {
      let v = rnd();
      const speed = 0.15 + 0.5 * rnd();
      for (let col = 0; col < cols; col++) {
        v = Math.min(1, Math.max(0, v + (rnd() - 0.5) * speed));
        svg.append(s('rect', { x: col * 4 + 0.3, y: y + 0.3, width: 3.4, height: rowH - 0.6, class: 'lw-cell lw-cell--' + g.cls,
          style: 'opacity:' + (0.12 + 0.88 * v).toFixed(2) + ';--col:' + col }));
      }
      y += rowH;
    }
    y += gap;
  }
  const names = h('div', { class: 'lw-heat-names' }, [
    h('span', { class: 'lw-hn lw-hn--g', style: 'flex:18', html: '<b class="tnum">18</b> the ball and all players' }),
    h('span', { class: 'lw-hn lw-hn--h', style: 'flex:16', html: '<b class="tnum">16</b> the home team' }),
    h('span', { class: 'lw-hn lw-hn--a', style: 'flex:16', html: '<b class="tnum">16</b> the away team' }),
  ]);
  const heat = h('div', { class: 'lw-fill lw-heatwrap' }, [names, h('div', { class: 'lw-heatbox' }, [svg, h('span', { class: 'lw-heat-t', text: 'time' })])]);
  const pitch = (flip) => {
    const p = s('svg', { viewBox: '0 0 120 66', class: 'lw-mini', 'aria-hidden': 'true' });
    p.append(s('rect', { x: 1, y: 1, width: 118, height: 64, rx: 4, class: 'lw-mini-pitch' }),
      s('path', { d: 'M60 1 V65', class: 'lw-mini-line' }), s('circle', { cx: 60, cy: 33, r: 9, class: 'lw-mini-line' }),
      s('path', { d: flip ? 'M84 22 H36 M44 16 L36 22 L44 28' : 'M36 22 H84 M76 16 L84 22 L76 28', class: 'lw-mini-raw' }),
      s('path', { d: 'M36 46 H84 M76 40 L84 46 L76 52', class: 'lw-mini-home' }));
    return p;
  };
  const turn = h('div', { class: 'lw-turn' }, [
    h('figure', {}, [pitch(false), h('figcaption', { text: 'first half' })]),
    h('figure', {}, [pitch(true), h('figcaption', { text: 'second half' })]),
    h('p', { class: 'lw-turn-key', html: '<i class="lw-key lw-key--raw"></i>the home team’s attack in the tracking<br><i class="lw-key lw-key--home"></i>after the turn: the same way in both halves' }),
  ]);
  return [panelHead(1, 'What the models read: 50 features a frame, no player identity'), heat, turn];
}

function panelNetwork() {
  const arrow = () => h('span', { class: 'lw-arrow', 'aria-hidden': 'true', text: '→' });
  const block = (title, sub, cls) => h('div', { class: 'lw-block ' + (cls || '') }, [h('b', { text: title }), h('span', { text: sub })]);
  const gru = h('div', { class: 'lw-block lw-block--gru' }, [
    h('b', { text: 'two recurrent layers' }),
    h('span', { class: 'lw-gru-dir', html: '<i>→</i> 64 forward' }),
    h('span', { class: 'lw-gru-dir', html: '<i>←</i> 64 backward' }),
  ]);
  const chain = h('div', { class: 'lw-chain' }, [
    block('50 features', 'averaged over 5 frames'), arrow(),
    block('linear layer', '64 units'), arrow(),
    block('convolution', 'over time'), arrow(),
    gru, arrow(),
    h('div', { class: 'lw-block lw-block--head' }, [h('b', { text: 'head' }),
      h('span', { class: 'lw-hads', html: '<i class="chip chip--h">H</i><i class="chip chip--a">A</i><i class="chip chip--d">D</i>' })]),
  ]);
  const copies = ['as it is', 'teams swapped', 'pitch mirrored', 'both'];
  const grid = h('div', { class: 'lw-ens' });
  grid.append(h('span', { class: 'lw-ens-corner', text: '' }));
  copies.forEach((t) => grid.append(h('span', { class: 'lw-ens-col', text: t })));
  // each cell is one output for one frame: the shares of H, D and A, a little different each time
  const rnd = mulberry32(5);
  let sumH = 0, sumHD = 0;
  for (let r = 0; r < 4; r++) {
    grid.append(h('span', { class: 'lw-ens-row', text: 'seed ' + (r + 1) }));
    for (let col = 0; col < 4; col++) {
      const pH = 0.5 + 0.18 * (rnd() - 0.5), pD = 0.12 + 0.08 * rnd();
      sumH += pH; sumHD += pH + pD;
      grid.append(h('i', { class: 'lw-ens-cell', style: '--k:' + (r * 4 + col) + ';--p1:' + (100 * pH).toFixed(1) + '%;--p2:' + (100 * (pH + pD)).toFixed(1) + '%' }));
    }
  }
  const mean = h('i', { class: 'lw-ens-cell lw-ens-cell--mean', style: '--p1:' + (100 * sumH / 16).toFixed(1) + '%;--p2:' + (100 * sumHD / 16).toFixed(1) + '%' });
  const out = h('div', { class: 'lw-ens-out' }, [
    h('span', { class: 'lw-arrow', 'aria-hidden': 'true', text: '→' }),
    h('div', { class: 'lw-block lw-block--mean' }, [h('b', { text: 'mean of 16' }), mean, h('span', { text: 'interpolated to 25 frames a second' })]),
  ]);
  return [panelHead(2, 'The network, at 5 frames a second'), chain,
    h('p', { class: 'lw-sub-h', text: 'Four networks from different random seeds, each on four mirrored copies of the match' }),
    h('div', { class: 'lw-ens-wrap' }, [grid, out])];
}

function treeIcon(cls) {
  const t = s('svg', { viewBox: '0 0 64 48', class: 'lw-tree ' + cls, 'aria-hidden': 'true' });
  const nodes = [[32, 6], [16, 24], [48, 24], [8, 42], [24, 42], [40, 42], [56, 42]];
  t.append(s('path', { d: 'M32 6 L16 24 M32 6 L48 24 M16 24 L8 42 M16 24 L24 42 M48 24 L40 42 M48 24 L56 42', class: 'lw-tree-e' }));
  nodes.forEach(([x, y], i) => t.append(s('circle', { cx: x, cy: y, r: i < 3 ? 4 : 3.4,
    class: i < 3 ? 'lw-tree-n' : 'lw-tree-l' + (i % 2 ? ' lw-tree-l--b' : '') })));
  return t;
}

function panelPerFrame() {
  const bar = h('div', { class: 'lw-bar' }, [
    h('span', { class: 'lw-seg lw-seg--base', style: 'flex:50' }),
    h('span', { class: 'lw-seg lw-seg--touch', style: 'flex:8' }),
    h('span', { class: 'lw-seg lw-seg--win', style: 'flex:126' }),
  ]);
  const key = h('div', { class: 'lw-bar-key' }, [
    h('span', {}, [h('i', { class: 'lw-key lw-key--base' }), document.createTextNode('the 50 features')]),
    h('span', {}, [h('i', { class: 'lw-key lw-key--touch' }), document.createTextNode('8 of proximity touches: the ball visible, a team’s nearest player within 1.8 m')]),
    h('span', {}, [h('i', { class: 'lw-key lw-key--win' }), document.createTextNode('means over windows of 0.5 to 4.5 s, never across a half')]),
  ]);
  const trees = h('div', { class: 'lw-trees' }, [
    h('div', { class: 'lw-treecard' }, [treeIcon('lw-tree--d'), h('div', {}, [h('b', { text: 'dead ball or live play' }),
      h('span', { text: 'trained on DFL, Metrica and SkillCorner, SkillCorner at a quarter weight' })])]),
    h('div', { class: 'lw-treecard' }, [treeIcon('lw-tree--ha'), h('div', {}, [h('b', { text: 'home or away, on live frames' }),
      h('span', { text: 'trained on Metrica and DFL' })])]),
  ]);
  const formula = h('p', { class: 'lw-formula', html: '<span><i>p</i>(D) from the first model</span>' +
    '<span><i>p</i>(H) = (1 − <i>p</i>(D)) × <i>p</i>(home | live)</span>' });
  return [panelHead(3, 'The per-frame model, at 25 frames a second'),
    h('div', { class: 'lw-bar-wrap' }, [h('p', { class: 'lw-bar-total', html: '<b class="tnum">184</b> features on every frame' }), bar, key]),
    trees, formula];
}

function panelBlend(data, wNet, wPf) {
  const rows = h('div', { class: 'lw-rows' });
  const row = (label, weight, chart, cls) => {
    rows.append(h('div', { class: 'lw-row ' + (cls || '') }, [
      h('div', { class: 'lw-row-label' }, [h('b', { text: label }), weight ? h('span', { class: 'lw-weight' }, weight) : null]),
      h('div', { class: 'lw-row-chart' }, [chart]),
    ]));
  };
  row('the network', [document.createTextNode('× '), wNet], areaChart(data.net, 'is-net'), 'lw-row--net');
  row('the per-frame model', [document.createTextNode('× '), wPf], areaChart(data.pf, 'is-pf'), 'lw-row--pf');
  row('the blend', null, areaChart(data.blend, 'is-blend'), 'lw-row--blend');
  rows.append(h('div', { class: 'lw-row lw-row--axis' }, [h('div', { class: 'lw-row-label' }), timeAxis(7)]));
  const legend = h('ul', { class: 'lw-legend lw-legend--inline' }, [
    h('li', {}, [h('i', { class: 'lw-key lw-key--h' }), document.createTextNode('p(H)')]),
    h('li', {}, [h('i', { class: 'lw-key lw-key--d' }), document.createTextNode('p(D)')]),
    h('li', {}, [h('i', { class: 'lw-key lw-key--a' }), document.createTextNode('p(A)')]),
  ]);
  return [panelHead(4, 'The blend, frame by frame: synthetic probabilities'), rows, legend,
    h('p', { class: 'lw-cap', text: 'The network is smooth and enters dead ball late. The per-frame model is sharp and noisy. A player of A comes close at 1.5 s; A wins the ball at 3 s; the ball goes out at 5.2 s.' })];
}

function panelDecoder(data, costHA, costAD, chipHA) {
  const most = data.blend.map(argmax);
  const rows = h('div', { class: 'lw-rows lw-rows--dec' });
  const chartCell = h('div', { class: 'lw-row-chart' }, [areaChart(data.blend, 'is-blend')]);
  const note = h('div', { class: 'lw-flag' });
  chartCell.append(note);
  rows.append(h('div', { class: 'lw-row lw-row--blend' }, [h('div', { class: 'lw-row-label' }, [h('b', { text: 'the blend' })]), chartCell]));
  rows.append(h('div', { class: 'lw-row lw-row--strip' }, [h('div', { class: 'lw-row-label' }, [h('b', { text: 'most likely label' })]),
    h('div', { class: 'lw-row-chart' }, [labelStrip(most, 'is-most')])]));
  const decodedSvg = labelStrip(most, 'is-decoded');
  const decodedLabel = h('b', { text: 'decoded' });
  rows.append(h('div', { class: 'lw-row lw-row--strip lw-row--decoded' }, [h('div', { class: 'lw-row-label' }, [decodedLabel]),
    h('div', { class: 'lw-row-chart' }, [decodedSvg])]));
  rows.append(h('div', { class: 'lw-row lw-row--axis' }, [h('div', { class: 'lw-row-label' }), timeAxis(7)]));

  // the frames around the moment where a player of A comes close
  const [n0, n1] = data.near;
  const nearFrames = [];
  for (let f = n0; f < n1; f++) if (most[f] === 1) nearFrames.push(f);
  const gainA = nearFrames.reduce((acc, f) => acc + Math.log(Math.max(data.blend[f][1], 1e-4)) - Math.log(Math.max(data.blend[f][0], 1e-4)), 0);
  if (nearFrames.length) {
    note.style.left = (100 * nearFrames[0] / (data.n - 1)).toFixed(2) + '%';
    note.style.width = (100 * (nearFrames[nearFrames.length - 1] + 1 - nearFrames[0]) / (data.n - 1)).toFixed(2) + '%';
  }

  const slider = h('input', { type: 'range', min: '0', max: '30', step: '0.5', value: String(costHA), class: 'lw-range',
    'aria-label': 'Cost of each change of label, in log-probability' });
  const costOut = h('b', { class: 'tnum lw-cost-v' });
  const tick = h('span', { class: 'lw-range-tick', style: 'left:' + (100 * costHA / 30).toFixed(2) + '%' }, [
    h('span', {}, [document.createTextNode('the engine, '), chipHA])]);
  const readout = h('p', { class: 'lw-readout' });
  const verdict = h('p', { class: 'lw-verdict' });
  const control = h('div', { class: 'lw-control' }, [
    h('label', { class: 'lw-range-label' }, [document.createTextNode('cost of each change '), costOut]),
    h('div', { class: 'lw-range-wrap' }, [slider, tick]),
  ]);

  function update() {
    const cost = Number(slider.value);
    // the slider moves both costs together, as the submitted engine sets them equal
    const ratio = costHA > 0 ? costAD / costHA : 1;
    const dec = viterbi(data.blend, cost, cost * ratio);
    fillStrip(decodedSvg, dec);
    costOut.textContent = fixed(cost, cost % 1 ? 1 : 0);
    const kept = nearFrames.length && nearFrames.some((f) => dec[f] === 1);
    note.classList.toggle('is-kept', !!kept);
    readout.textContent = 'Changes of label in these 7 s: ' + changes(most) + ' for the most likely label, ' + changes(dec) + ' decoded.';
    if (nearFrames.length) {
      verdict.textContent = 'At 1.5 s, A is the most likely label on ' + nearFrames.length + ' frames. On them A beats H by ' +
        fixed(gainA, 1) + ' in log-probability, against ' + fixed(2 * cost, cost % 1 ? 1 : 0) + ' for the two changes. ' +
        (kept ? 'The decoder writes a short A possession.' : 'The decoder keeps H.');
    }
  }
  slider.addEventListener('input', update);
  update();
  return {
    parts: [panelHead(5, 'The decoder, on the blend: synthetic probabilities'), rows, control, verdict, readout],
    decodedSvg,
  };
}

function panelFallback(c) {
  const lane = (value, title, sub, md5, cls) => h('div', { class: 'lw-lane ' + cls }, [
    h('span', { class: 'lw-switch' }, [value]),
    h('span', { class: 'lw-arrow', 'aria-hidden': 'true', text: '→' }),
    h('div', { class: 'lw-lane-body' }, [h('b', { text: title }), h('span', { text: sub })]),
    h('span', { class: 'lw-arrow', 'aria-hidden': 'true', text: '→' }),
    h('div', { class: 'lw-lane-out' }, [h('span', { text: 'output md5' }), h('code', { text: md5 })]),
  ]);
  const off = h('span', { class: 'k-const tnum', title: 'the other value of the same setting' }, [h('b', { text: 'off' })]);
  const lanes = h('div', { class: 'lw-lanes' }, [
    lane(c('LEARNED_MODE', ''), 'two learned models and the decoder', 'the engine, with the kickoff team rule', 'd7f167a2401387bd8339d1bcbb292f2c', 'lw-lane--model is-on'),
    lane(off, 'the rule engine', 'physical rules for stoppages and owner events, with Smart Tagging as a weak prior', '7844cad862e38d55cf9a3fc29b18ae85', 'lw-lane--off'),
  ]);
  const btn = h('button', { type: 'button', class: 'btn lw-flip', 'aria-pressed': 'false', text: 'Flip the switch' });
  btn.addEventListener('click', () => {
    const on = btn.getAttribute('aria-pressed') !== 'true';
    btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    lanes.querySelector('.lw-lane--model').classList.toggle('is-on', !on);
    lanes.querySelector('.lw-lane--off').classList.toggle('is-on', on);
  });
  return [panelHead(6, 'The fallback: one setting, two outputs'), lanes,
    h('div', { class: 'lw-fb-foot' }, [btn, h('p', { class: 'lw-cap', text: 'Both read the same cleaned tracking and give each half the same dead ball before its kickoff. Only the rule engine reads Smart Tagging.' })])];
}

/* -- the copy ----------------------------------------------------------------------------------- */

function stepCopy(c, link) {
  const t = (x) => document.createTextNode(x);
  return [
    {
      kicker: '1 · Clean', title: 'Clean the tracking',
      body: [
        [t('The engine reads the tactical ATD. Where the tracker gives no speed for a player, it computes one from the positions.')],
        [t('A ball frame more than '), c('BALL_MAX_JUMP_M', 'm'), t(' from the last good frame is removed: 3 m in one frame is 75 m/s, faster than any shot. A visible run of the ball shorter than '),
          c('BALL_MIN_RUN', 'frames'), t(' is dropped.')],
        [t('The tracker sometimes carries one goalkeeper’s track onto the other goalkeeper. The goalkeeper-swap correction moves the position back before any feature is computed. These three steps come from the code of the rule engine.')],
      ],
    },
    {
      kicker: '2 · Features', title: 'What the models read',
      body: [
        [t('Each frame gets 50 features of the tracking. They describe the ball: visible or not, position, speed, the time since it was last seen and until it is next seen, out of play.')],
        [t('They give the distance from the ball to the nearest players of each team, and how many players of each team are within 1.5, 3 and 5 m of it. And they give each team’s position, spread, speed, share of near-stationary players and deepest player. No player identity enters.')],
        [t('Home and away come from the ATD metadata: the first team it lists is home, and it gives each track its team. The pitch is turned so that the home team attacks the same way in both halves: in each half, towards the away team’s side, as the two teams’ median positions over the half show it.')],
      ],
    },
    {
      kicker: '3 · The network', title: 'The network: context at 5 frames a second',
      body: [
        [t('A small bidirectional recurrent network, a GRU, reads the features averaged over five frames, at 5 frames a second. It runs in both directions, so each moment sees what comes before it and after it.')],
        [t('A linear layer of 64 units, a convolution over time, two recurrent layers of 64 units each way and a head on three classes: H, A and D. Each network trained for 2,000 steps on batches of 16 two-minute windows, with a cross-entropy loss and Adam. A SkillCorner match was drawn for a training window a quarter as often as a Metrica or DFL match, and SkillCorner taught only dead ball against live play.')],
        [t('Four networks, trained from different random seeds, each read four mirrored copies of the match: as it is, mirrored across its long axis, with home and away swapped, and both. The engine averages the 16 outputs and interpolates them to 25 frames a second.')],
      ],
    },
    {
      kicker: '4 · The per-frame model', title: 'The per-frame model: short events at 25 frames a second',
      body: [
        [t('Two gradient-boosted tree models read 184 features on every frame: the 50 above, eight features of proximity touches, and means of some features over windows of 0.5 to 4.5 s, never across a half.')],
        [t('A proximity touch is a frame where the ball is visible and a team’s nearest player is within 1.8 m. Its eight features say which team touched last and which touches next, and when.')],
        [t('One model decides dead ball against live play, on all three providers. The other decides home against away on live frames, on Metrica and DFL only. The engine walks their trees in '), h('code', { text: 'numpy' }), t('.')],
      ],
    },
    {
      kicker: '5 · The blend', title: 'The blend: three parts network, one part per-frame model',
      body: [
        [t('On every frame the engine averages the two sets of probabilities: the network at weight '), c('LEARNED_PF_WEIGHT', '', 'net'),
          t(', the per-frame model at '), c('LEARNED_PF_WEIGHT', ''), t('.')],
        [t('At 0.25 the blend beats the network alone on Metrica, SkillCorner and the PFF design matches and trails it by 0.006 on DFL. A weight of 0.5 gave up more on DFL to gain more on PFF.')],
        [t('The network alone missed stoppages while the ball was missing. On the PFF design matches, 43 PFF FC World Cup matches that helped us compare designs, it labelled only 82.5% of such dead frames D. Averaged in, the per-frame model raised spellF1 on 43 of the 43 PFF design matches.')],
      ],
    },
    {
      kicker: '6 · The decoder', title: 'The decoder: whole possessions, not frames',
      body: [
        [t('A three-state Viterbi decoder finds the best sequence of H, A and D for each graded half, from the log-probabilities of the blend.')],
        [t('Each change of team costs '), c('LEARNED_COST_HA', ''), t(', and each change into or out of dead ball costs '), c('LEARNED_COST_AD', ''),
          t(', in log-probability. A run of another label must gain more than its two changes cost, so a short false possession needs strong evidence. A rule fixed in advance chose the cost: the best mean gain on Metrica and DFL, each labelled by models that never saw its provider. A cost of 12 for both kinds of change came within 0.0002 of the best pair. '),
          h('code', { text: 'spellF1' }), t(' asks for that: it charges a false possession like a missed long one.')],
        [t('The frames before the kickoff of each half are dead ball. The kickoff frames come from the organisers’ '), h('code', { text: 'games.py' }),
          t('. The rule is the rule engine’s.')],
        [t('From each half’s kickoff frame, the first '), c('LEARNED_KICKOFF_FLOOR_S', 's'),
          t(' goes to the team that takes the kickoff: in the first half the team that the decoder finds first, in the second half the other team (Law 8). The decoder then decides everything after it. The 1 s is not tuned, because the kicking team plays the ball first and keeps it at least for the first pass. Measured on four of the tests of Section 03, the rule moves each gain by at most 0.0004.')],
        [t('Drag the cost in the figure to see how the decoded labels change.')],
      ],
    },
    {
      kicker: '7 · The fallback', title: 'The fallback: one setting restores the rule engine',
      body: [
        [t('The rule engine is a set of physical rules. It finds the stoppages from the movement of the ball and the players, and changes the team only when a player of the other team kicks, receives, controls or dribbles the ball. '),
          link, t(' walks through it.')],
        [t('One setting switches the learned models off. The engine then writes the rule engine’s labels, byte for byte. The learned models do not read Smart Tagging. Only the rule engine does.')],
      ],
    },
  ];
}

/* -- mount -------------------------------------------------------------------------------------- */

/**
 * Mount the section.
 * @param {HTMLElement} el  the empty section element
 * @param {{bundle: Object, state: Object}} ctx
 */
export function mount(el, ctx) {
  const bundle = ctx.bundle || {};
  const cfg = configAccessor(bundle.config);
  const num = (name, fallback) => (cfg.has(name) ? Number(cfg.get(name)) : fallback);
  const wPf = num('LEARNED_PF_WEIGHT', 0.25);
  const costHA = num('LEARNED_COST_HA', 12);
  const costAD = num('LEARNED_COST_AD', 12);
  const fmtNum = (v) => String(Math.round(v * 1e6) / 1e6);

  /** A value chip: the value of a constant, its name in the tooltip. `net` shows one minus it. */
  function c(name, unit, mode) {
    const chip = h('span', { class: 'k-const tnum' });
    let v = 'n/a';
    if (cfg.has(name)) {
      const raw = cfg.get(name);
      v = mode === 'net' ? fmtNum(1 - Number(raw)) : typeof raw === 'number' ? fmtNum(raw) : String(raw);
    }
    chip.title = (mode === 'net' ? 'one minus the setting ' : 'setting ') + name + ' in engine/config.py';
    chip.append(h('b', { text: v }), document.createTextNode(unit ? ' ' + unit : ''));
    return chip;
  }

  el.textContent = '';
  el.classList.add('lw-section');

  const header = h('header', { class: 'section-head lw-head' }, [
    h('p', { class: 'kicker', html: '<span class="num">01</span> The approach, and why <span class="sep">/</span> the engine' }),
    h('h2', { text: 'From tracking to labels: two learned models and a decoder' }),
    h('p', { class: 'lede', text: 'The engine cleans the tracking, describes every frame with 50 features, asks two models for the probability of H, A and D, averages them and decodes whole possessions. ' +
      'Both models trained offline on labelled matches of DFL, Metrica and SkillCorner. ' +
      'Hover a value to see its name in the code. The figure is a diagram with synthetic numbers: the hosted page never holds tracking data.' }),
  ]);

  const link = h('a', { href: '#engine', text: 'Section 04' });
  const steps = h('div', { class: 'steps lw-steps' });
  stepCopy(c, link).forEach((st, k) => {
    const copy = h('div', { class: 'step-copy' }, [h('p', { class: 'lw-kicker tnum', text: st.kicker }), h('h3', { text: st.title })]);
    for (const para of st.body) copy.append(h('p', {}, para));
    steps.append(h('div', { class: 'step lw-step', 'data-step': String(k) }, [copy]));
  });

  const subs = { blend: fmtNum(1 - wPf) + ' and ' + fmtNum(wPf), decoder: 'costs ' + fmtNum(costHA) + ' and ' + fmtNum(costAD) };
  const wide = buildPipeline(WIDE, 'lp--wide', subs);
  const narrow = buildPipeline(NARROW, 'lp--narrow', {});

  const data = synthetic(wPf);
  const dec = panelDecoder(data, costHA, costAD, c('LEARNED_COST_HA', ''));
  const panelParts = [
    panelClean(c),
    panelFeatures(),
    panelNetwork(),
    panelPerFrame(),
    panelBlend(data, c('LEARNED_PF_WEIGHT', '', 'net'), c('LEARNED_PF_WEIGHT', '')),
    dec.parts,
    panelFallback(c),
  ];
  const stage = h('div', { class: 'lw-stage' });
  const panels = panelParts.map((parts, k) => {
    const p = h('div', { class: 'lw-panel lw-panel--' + k, 'data-panel': String(k), 'aria-hidden': 'true' }, parts);
    stage.append(p);
    return p;
  });

  const fig = h('div', { class: 'fig lw-fig' }, [h('div', { class: 'lw-pipe' }, [wide.root, narrow.root]), stage]);
  const figure = h('div', { class: 'figure lw-figure' }, [fig]);
  el.append(header, h('div', { class: 'scrolly lw-scrolly' }, [steps, figure]));

  let current = -1;
  function show(k) {
    k = Math.max(0, Math.min(LIT.length - 1, k | 0));
    if (k === current) return;
    current = k;
    const lit = LIT[k];
    for (const p of [wide, narrow]) {
      for (const [id, node] of Object.entries(p.nodes)) node.classList.toggle('is-active', lit.nodes.indexOf(id) >= 0);
      for (const [id, edge] of Object.entries(p.edges)) edge.classList.toggle('is-active', lit.edges.indexOf(id) >= 0);
      p.root.classList.toggle('is-fallback', k === LIT.length - 1);
    }
    panels.forEach((p, i) => {
      const on = i === k;
      p.classList.toggle('is-on', on);
      p.setAttribute('aria-hidden', on ? 'false' : 'true');
      if (on && !reducedMotion()) {
        p.classList.remove('is-play');
        // restart the entry animation on the next frame
        requestAnimationFrame(() => p.classList.add('is-play'));
      }
    });
  }

  for (const p of [wide, narrow]) {
    for (const [id, node] of Object.entries(p.nodes)) {
      const go = () => {
        const k = LIT.findIndex((x) => x.nodes[0] === id || (id === 'labels' && x.nodes.indexOf('decoder') >= 0));
        const target = steps.querySelector('.step[data-step="' + Math.max(0, k) + '"]');
        if (target) target.scrollIntoView({ behavior: reducedMotion() ? 'auto' : 'smooth', block: 'center' });
        if (k >= 0) show(k);
      };
      node.addEventListener('click', go);
      node.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
    }
  }

  const off = ctx.state.on('activeStep', (st) => {
    if (st && st.section === 'learned') show(st.step);
  });
  const now = ctx.state.get().activeStep;
  show(now && now.section === 'learned' ? now.step : 0);

  return { show, destroy() { off(); } };
}

export default { mount };
