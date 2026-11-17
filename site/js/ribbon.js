/**
 * The match ribbon: label lanes over all frames of the match, drawn on one canvas.
 *
 * Each lane is either RLE runs at frame resolution or per-minute share bins. A runs lane is
 * drawn by exact pixel-column averaging: each column gets the mean colour of the frames under
 * it, so a file that flickers between H and A looks mixed at full-match scale and resolves into
 * clean runs when you zoom in. X frames show the hatch underneath.
 *
 * Zoom with d3.zoom on the plot (pinch, or Ctrl and wheel, or the buttons) and with a d3.brushX
 * on the overview strip. A gold bracket marks the public window. The playhead follows
 * `setPlayhead(frame)`. A click calls `onSeek(frame)`.
 *
 * @module ribbon
 */

import { columnFractions } from './util/rle.js';
import { labelAt } from './data.js';
import { palette, hatchPattern } from './util/colors.js';
import { videoTime, matchClock, int, pct, esc } from './util/format.js';
import { dur, easeOut, showTip, hideTip, observeSize } from './util/dom.js';

const TICK_STEPS_S = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900];
const LETTER_NAME = { H: 'H', A: 'A', D: 'D', X: 'X' };

/**
 * @typedef {Object} Lane
 * @property {string} id
 * @property {string} title        Lane title, always shown, so colour is never the only channel.
 * @property {string} [sub]        Second line of the title.
 * @property {'runs'|'bins'} type
 * @property {Array} [runs]        RLE runs for a runs lane.
 * @property {Array} [bins]        Minute bin rows for a bins lane.
 * @property {[number, number]} [region]  Frames where the lane has data. Outside it the lane says so.
 * @property {string} [accent]     CSS colour of the title marker.
 * @property {number} [weight]     Relative height, 1 by default.
 */

/**
 * Create a ribbon in `container`.
 * @param {HTMLElement} container
 * @param {Object} opts
 * @param {Object} opts.d3
 * @param {number} opts.nFrames
 * @param {number} opts.fps
 * @param {{start: number, stop: number}} opts.window
 * @param {Array<[number, number]>} opts.gradedBlocks
 * @param {Array<{half: number, frame: number}>} opts.kickoffs
 * @param {Lane[]} opts.lanes
 * @param {string} opts.overviewLane   Id of the lane drawn in the overview strip.
 * @param {function(number): void} [opts.onSeek]
 * @param {function([number, number]): void} [opts.onDomain]
 */
export function createRibbon(container, opts) {
  const d3 = opts.d3;
  const N = opts.nFrames;
  const fps = opts.fps || 25;
  const win = opts.window;
  const lanes = opts.lanes;
  const pal = palette();

  container.classList.add('ribbon');
  container.innerHTML = `
    <div class="ribbon-bar">
      <div class="ribbon-legend">
        <span><i class="sw sw--h"></i>H</span><span><i class="sw sw--a"></i>A</span>
        <span><i class="sw sw--d"></i>D dead ball</span><span><i class="sw sw--x"></i>X not graded</span>
        <span><i class="sw sw--gt"></i>public window</span>
      </div>
      <div class="ribbon-readout tnum" aria-live="off"></div>
      <div class="ribbon-tools" role="group" aria-label="Ribbon zoom">
        <button class="btn btn--ghost" type="button" data-act="full">Full match</button>
        <button class="btn btn--ghost" type="button" data-act="window">Public window</button>
        <button class="btn btn--ghost btn--icon" type="button" data-act="out" aria-label="Zoom out">&minus;</button>
        <button class="btn btn--ghost btn--icon" type="button" data-act="in" aria-label="Zoom in">+</button>
      </div>
    </div>
    <div class="ribbon-body">
      <div class="ribbon-titles"></div>
      <div class="ribbon-plot"><canvas class="ribbon-canvas" role="img"></canvas></div>
    </div>
    <div class="ribbon-mini">
      <div class="ribbon-mini-label">overview</div>
      <div class="ribbon-mini-plot"><canvas class="ribbon-mini-canvas"></canvas><svg class="ribbon-brush"></svg></div>
    </div>`;

  const titles = container.querySelector('.ribbon-titles');
  const plot = container.querySelector('.ribbon-plot');
  const canvas = container.querySelector('.ribbon-canvas');
  const miniPlot = container.querySelector('.ribbon-mini-plot');
  const miniCanvas = container.querySelector('.ribbon-mini-canvas');
  const brushSvg = container.querySelector('.ribbon-brush');
  const readout = container.querySelector('.ribbon-readout');
  canvas.setAttribute('aria-label', 'Label lanes over all ' + int(N) + ' frames: ' +
    lanes.map((l) => l.title).join(', '));

  lanes.forEach((lane) => {
    const t = document.createElement('div');
    t.className = 'ribbon-title';
    t.dataset.lane = lane.id;
    t.innerHTML = '<i style="background:' + (lane.accent || pal.muted) + '"></i><span><b>' +
      esc(lane.title) + '</b>' + (lane.sub ? '<small>' + esc(lane.sub) + '</small>' : '') + '</span>';
    titles.appendChild(t);
  });

  // -- state -------------------------------------------------------------------------------
  let W = 0;
  let H = 0;
  let dpr = 1;
  let geo = null;
  let x0 = d3.scaleLinear().domain([0, N]);
  let x = x0.copy();
  let transform = d3.zoomIdentity;
  let reveal = 1;
  let playhead = null;
  let hoverX = null;
  let focusId = null;
  let focusAlpha = 0;
  let cache = null;
  let cacheKey = '';
  let hatch = null;
  let syncing = false;
  let miniW = 0;
  let miniH = 0;
  const ctx = canvas.getContext('2d');
  const mctx = miniCanvas.getContext('2d');
  const rowCanvas = document.createElement('canvas');
  const rowCtx = rowCanvas.getContext('2d');

  // -- geometry ----------------------------------------------------------------------------
  function layout() {
    const rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
    const axisH = Math.round(rem * 1.7);
    const topH = Math.round(rem * 1.5);
    const gap = Math.max(6, Math.round(rem * 0.5));
    const avail = Math.max(60, H - axisH - topH - gap * (lanes.length - 1));
    const totalW = lanes.reduce((s, l) => s + (l.weight || 1), 0);
    // The cap only bites on very tall panels (4K): at 4.6 rem the lanes filled a third of it.
    const unit = Math.min(rem * 12, avail / totalW);
    let y = topH;
    const rows = lanes.map((l) => {
      const h = Math.max(10, Math.round(unit * (l.weight || 1)));
      const row = { lane: l, y, h };
      y += h + gap;
      return row;
    });
    const used = y - gap + axisH;
    return { rows, axisY: y - gap + Math.round(rem * 0.35), topH, rem, used };
  }

  function resize() {
    W = plot.clientWidth;
    H = plot.clientHeight;
    if (!W || !H) return;
    dpr = Math.min(3, window.devicePixelRatio || 1);
    geo = layout();
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    titles.style.paddingTop = geo.topH + 'px';
    Array.from(titles.children).forEach((t, i) => {
      t.style.height = geo.rows[i].h + 'px';
      t.classList.toggle('is-compact', geo.rows[i].h < geo.rem * 2.2);
      t.style.marginBottom = (i < geo.rows.length - 1 ? geo.rows[i + 1].y - geo.rows[i].y - geo.rows[i].h : 0) + 'px';
    });
    hatch = hatchPattern(ctx, dpr);
    x0 = d3.scaleLinear().domain([0, N]).range([0, W]);
    x = transform.rescaleX(x0);
    zoom.extent([[0, 0], [W, H]]).translateExtent([[0, 0], [W, H]])
      .scaleExtent([1, Math.max(2, (N * 8) / W)]);
    miniW = miniPlot.clientWidth;
    miniH = miniPlot.clientHeight;
    miniCanvas.width = Math.round(miniW * dpr);
    miniCanvas.height = Math.round(miniH * dpr);
    miniCanvas.style.width = miniW + 'px';
    miniCanvas.style.height = miniH + 'px';
    d3.select(brushSvg).attr('width', miniW).attr('height', miniH);
    brush.extent([[0, 0], [miniW, miniH]]);
    brushG.call(brush);
    cacheKey = '';
    drawMini();
    syncBrush();
    draw();
  }

  // -- lane rendering ----------------------------------------------------------------------
  function drawRunsLane(g, row, d0, d1) {
    const lane = row.lane;
    const cols = Math.max(1, Math.round(W * dpr));
    const fr = columnFractions(lane.runs, d0, d1, cols);
    if (rowCanvas.width !== cols) {
      rowCanvas.width = cols;
      rowCanvas.height = 1;
    }
    const img = rowCtx.createImageData(cols, 1);
    const data = img.data;
    const rgb = pal.labelRgb;
    for (let c = 0; c < cols; c++) {
      const h = fr[c * 4];
      const a = fr[c * 4 + 1];
      const d = fr[c * 4 + 2];
      const s = h + a + d;
      const o = c * 4;
      if (s <= 1e-6) {
        data[o + 3] = 0;
        continue;
      }
      data[o] = (h * rgb[0][0] + a * rgb[1][0] + d * rgb[2][0]) / s;
      data[o + 1] = (h * rgb[0][1] + a * rgb[1][1] + d * rgb[2][1]) / s;
      data[o + 2] = (h * rgb[0][2] + a * rgb[1][2] + d * rgb[2][2]) / s;
      data[o + 3] = Math.min(255, Math.round(s * 255));
    }
    rowCtx.putImageData(img, 0, 0);
    g.imageSmoothingEnabled = false;
    g.drawImage(rowCanvas, 0, 0, cols, 1, 0, row.y * dpr, cols, row.h * dpr);
  }

  function drawBinsLane(g, row) {
    const bins = row.lane.bins || [];
    const y = row.y * dpr;
    const h = row.h * dpr;
    for (const b of bins) {
      const xa = x(b.start) * dpr;
      const xb = x(b.stop) * dpr;
      if (xb < 0 || xa > W * dpr) continue;
      const wpx = xb - xa;
      const inset = wpx > 6 * dpr ? Math.min(1.5 * dpr, wpx * 0.06) : 0;
      const bx = xa + inset * 0.5;
      const bw = Math.max(0.5, wpx - inset);
      let yy = y;
      const parts = [[b.share_h, pal.label[0]], [b.share_d, pal.label[2]], [b.share_a, pal.label[1]]];
      const total = (b.share_h || 0) + (b.share_a || 0) + (b.share_d || 0);
      for (const [share, color] of parts) {
        const hh = (share || 0) * h;
        if (hh <= 0) continue;
        g.fillStyle = color;
        g.fillRect(bx, yy, bw, hh);
        yy += hh;
      }
      if (total < 0.999) {
        g.fillStyle = hatch;
        g.fillRect(bx, yy, bw, y + h - yy);
      }
    }
  }

  function laneBackground(g, row, d0, d1) {
    const lane = row.lane;
    const y = row.y * dpr;
    const h = row.h * dpr;
    g.fillStyle = '#0e121b';
    g.fillRect(0, y, W * dpr, h);
    const reg = lane.region || [0, N];
    const xa = Math.max(0, x(reg[0]));
    const xb = Math.min(W, x(reg[1]));
    if (xb > xa) {
      g.fillStyle = hatch;
      g.fillRect(xa * dpr, y, (xb - xa) * dpr, h);
    }
    if (lane.region) {
      // Outside the region the lane has no public data. Say so in the lane.
      g.strokeStyle = 'rgba(154,163,178,0.25)';
      g.setLineDash([2 * dpr, 4 * dpr]);
      g.lineWidth = dpr;
      g.beginPath();
      g.moveTo(0, y + h / 2);
      g.lineTo(Math.max(0, xa) * dpr, y + h / 2);
      g.moveTo(Math.min(W, xb) * dpr, y + h / 2);
      g.lineTo(W * dpr, y + h / 2);
      g.stroke();
      g.setLineDash([]);
    }
    if (lane.type === 'runs') drawRunsLane(g, row, d0, d1);
    else drawBinsLane(g, row);
  }

  /** Write "public window only" in the empty parts of the lanes that have a region. */
  function drawRegionNotes(g) {
    g.font = Math.round(geo.rem * 0.68 * dpr) + 'px ' + getComputedStyle(document.body).fontFamily;
    g.textBaseline = 'middle';
    g.textAlign = 'center';
    for (const row of geo.rows) {
      const lane = row.lane;
      if (!lane.region) continue;
      const y = row.y * dpr;
      const h = row.h * dpr;
      const xa = Math.max(0, x(lane.region[0]));
      const xb = Math.min(W, x(lane.region[1]));
      // Put each note in the graded part of an empty span, never in the half-time veil.
      const blocks = opts.gradedBlocks && opts.gradedBlocks.length ? opts.gradedBlocks : [[0, N]];
      const spans = [];
      [[0, xa], [xb, W]].forEach(([a, b]) => {
        let best = null;
        blocks.forEach(([f0, f1]) => {
          const pa = Math.max(a, x(f0));
          const pb = Math.min(b, x(f1));
          if (pb - pa > 220 && (!best || pb - pa > best[1] - best[0])) best = [pa, pb];
        });
        if (best) spans.push(best);
      });
      for (const [a, b] of spans) {
        const tx = ((a + b) / 2) * dpr;
        const label = lane.regionNote || 'public window only';
        const tw = g.measureText(label).width + 12 * dpr;
        g.fillStyle = '#0e121b';
        g.fillRect(tx - tw / 2, y + h / 2 - 8 * dpr, tw, 16 * dpr);
        g.fillStyle = 'rgba(154,163,178,0.75)';
        g.fillText(label, tx, y + h / 2);
      }
    }
  }

  function renderCache() {
    const d = x.domain();
    const key = W + 'x' + H + '@' + dpr + ':' + d[0].toFixed(3) + ',' + d[1].toFixed(3);
    if (key === cacheKey && cache) return;
    if (!cache) cache = document.createElement('canvas');
    cache.width = canvas.width;
    cache.height = canvas.height;
    const g = cache.getContext('2d');
    g.clearRect(0, 0, cache.width, cache.height);
    for (const row of geo.rows) laneBackground(g, row, d[0], d[1]);
    // The scorer never grades frames outside the graded blocks. Some files still label them,
    // for example the organisers' baselines, so veil those frames with the X hatch.
    const spans = ungradedSpans();
    g.save();
    g.globalAlpha = 0.86;
    g.fillStyle = hatch;
    for (const row of geo.rows) {
      for (const [a, b] of spans) {
        const xa = Math.max(0, x(a));
        const xb = Math.min(W, x(b));
        if (xb > xa) g.fillRect(xa * dpr, row.y * dpr, (xb - xa) * dpr, row.h * dpr);
      }
    }
    g.restore();
    drawRegionNotes(g);
    cacheKey = key;
  }

  /** Half-open frame spans outside every graded block. */
  function ungradedSpans() {
    const blocks = (opts.gradedBlocks || []).slice().sort((a, b) => a[0] - b[0]);
    if (!blocks.length) return [];
    const out = [];
    let f = 0;
    for (const [a, b] of blocks) {
      if (a > f) out.push([f, a]);
      f = Math.max(f, b);
    }
    if (f < N) out.push([f, N]);
    return out;
  }

  function isGraded(frame) {
    return (opts.gradedBlocks || []).some((b) => frame >= b[0] && frame < b[1]);
  }

  // -- axis and overlays -------------------------------------------------------------------
  function tickStep() {
    const d = x.domain();
    const secsPerPx = (d[1] - d[0]) / fps / W;
    const minPx = geo.rem * 5.2;
    for (const s of TICK_STEPS_S) if (s / secsPerPx >= minPx) return s;
    return TICK_STEPS_S[TICK_STEPS_S.length - 1];
  }

  function drawAxis(g) {
    const step = tickStep();
    const d = x.domain();
    const y = geo.axisY * dpr;
    g.font = Math.round(geo.rem * 0.7 * dpr) + 'px ' + getComputedStyle(document.body).fontFamily;
    g.textBaseline = 'top';
    g.textAlign = 'center';
    const ks = (opts.kickoffs || []).slice().sort((a, b) => a.frame - b.frame);
    ks.forEach((k) => {
      const block = (opts.gradedBlocks || []).find((b) => k.frame >= b[0] && k.frame < b[1]);
      const stop = block ? block[1] : N;
      const stepF = step * fps;
      const nStart = Math.max(0, Math.ceil((d[0] - k.frame) / stepF));
      for (let n = nStart; ; n++) {
        const f = k.frame + n * stepF;
        if (f > stop || f > d[1]) break;
        const px = x(f) * dpr;
        const secs = n * step + (k.half === 2 ? 45 * 60 : 0);
        const mm = Math.floor(secs / 60);
        const ss = Math.round(secs % 60);
        const label = step >= 60 ? mm + '′' : mm + '′' + (ss < 10 ? '0' : '') + ss + '″';
        g.fillStyle = 'rgba(255,255,255,0.10)';
        g.fillRect(px, geo.topH * dpr, Math.max(1, dpr * 0.6), y - geo.topH * dpr - 3 * dpr);
        g.fillStyle = n === 0 ? pal.text : pal.muted;
        const text = n === 0 ? (k.half === 1 ? 'kickoff' : '2nd half') : label;
        const half = g.measureText(text).width / 2 + 2 * dpr;
        g.fillText(text, Math.max(half, Math.min(W * dpr - half, px)), y);
      }
    });
    // Half-time gap label.
    const blocks = opts.gradedBlocks || [];
    if (blocks.length === 2) {
      const a = x(blocks[0][1]);
      const b = x(blocks[1][0]);
      if (b - a > 70 && b > 0 && a < W) {
        g.fillStyle = pal.muted;
        g.fillText('half-time', ((Math.max(0, a) + Math.min(W, b)) / 2) * dpr, y);
      }
    }
  }

  function drawWindow(g) {
    const xa = x(win.start);
    const xb = x(win.stop);
    if (xb < -2 || xa > W + 2) return;
    const top = (geo.topH - geo.rem * 0.55) * dpr;
    const bottom = (geo.rows[geo.rows.length - 1].y + geo.rows[geo.rows.length - 1].h + 4) * dpr;
    const ca = Math.max(-4, xa) * dpr;
    const cb = Math.min(W + 4, xb) * dpr;
    g.fillStyle = 'rgba(255,209,102,0.06)';
    g.fillRect(ca, top, cb - ca, bottom - top);
    g.strokeStyle = pal.gt;
    g.lineWidth = 1.5 * dpr;
    g.shadowColor = 'rgba(255,209,102,0.6)';
    g.shadowBlur = 8 * dpr;
    g.beginPath();
    const tick = 6 * dpr;
    g.moveTo(ca + tick, top);
    g.lineTo(ca, top);
    g.lineTo(ca, bottom);
    g.lineTo(ca + tick, bottom);
    g.moveTo(cb - tick, top);
    g.lineTo(cb, top);
    g.lineTo(cb, bottom);
    g.lineTo(cb - tick, bottom);
    g.stroke();
    g.shadowBlur = 0;
    g.font = '700 ' + Math.round(geo.rem * 0.62 * dpr) + 'px ' + getComputedStyle(document.body).fontFamily;
    g.fillStyle = pal.gt;
    g.textBaseline = 'bottom';
    const label = 'PUBLIC WINDOW';
    const tw = g.measureText(label).width;
    const mid = (ca + cb) / 2;
    if (cb - ca > tw + 10 * dpr) {
      g.textAlign = 'center';
      g.fillText(label, mid, top - 3 * dpr);
    } else {
      g.textAlign = 'left';
      g.fillText(label, Math.min(cb + 6 * dpr, W * dpr - tw - 2 * dpr), top + 12 * dpr);
    }
  }

  function drawFocus(g) {
    if (focusAlpha <= 0.001) return;
    for (const row of geo.rows) {
      if (focusId && row.lane.id === focusId) continue;
      g.fillStyle = 'rgba(11,14,20,' + (0.62 * focusAlpha).toFixed(3) + ')';
      g.fillRect(0, row.y * dpr - 1, W * dpr, row.h * dpr + 2);
    }
  }

  function drawPlayhead(g) {
    if (playhead === null) return;
    const px = x(playhead + 0.5);
    if (px < 0 || px > W) return;
    const top = (geo.topH - geo.rem * 0.2) * dpr;
    const bottom = geo.axisY * dpr - 2 * dpr;
    g.fillStyle = '#ffffff';
    g.shadowColor = 'rgba(255,255,255,0.8)';
    g.shadowBlur = 6 * dpr;
    g.fillRect(px * dpr - dpr * 0.75, top, 1.5 * dpr, bottom - top);
    g.shadowBlur = 0;
    g.beginPath();
    g.moveTo(px * dpr - 5 * dpr, top - 6 * dpr);
    g.lineTo(px * dpr + 5 * dpr, top - 6 * dpr);
    g.lineTo(px * dpr, top);
    g.closePath();
    g.fill();
  }

  function drawHover(g) {
    if (hoverX === null) return;
    const top = geo.topH * dpr;
    const bottom = geo.axisY * dpr - 2 * dpr;
    g.fillStyle = 'rgba(255,255,255,0.55)';
    g.fillRect(hoverX * dpr - dpr * 0.5, top, dpr, bottom - top);
  }

  function draw() {
    if (!geo || !W) return;
    renderCache();
    const g = ctx;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, canvas.width, canvas.height);
    if (reveal < 1) {
      g.save();
      g.beginPath();
      g.rect(0, 0, reveal * W * dpr, canvas.height);
      g.clip();
      g.drawImage(cache, 0, 0);
      g.restore();
      // Leading edge glow.
      const ex = reveal * W * dpr;
      const grad = g.createLinearGradient(ex - 40 * dpr, 0, ex, 0);
      grad.addColorStop(0, 'rgba(255,255,255,0)');
      grad.addColorStop(1, 'rgba(255,255,255,0.35)');
      g.fillStyle = grad;
      for (const row of geo.rows) g.fillRect(ex - 40 * dpr, row.y * dpr, 40 * dpr, row.h * dpr);
    } else {
      g.drawImage(cache, 0, 0);
    }
    drawFocus(g);
    drawAxis(g);
    if (reveal >= 1) drawWindow(g);
    drawHover(g);
    drawPlayhead(g);
  }

  // -- overview strip ----------------------------------------------------------------------
  let xm = d3.scaleLinear().domain([0, N]);
  function drawMini() {
    if (!miniW) return;
    xm = d3.scaleLinear().domain([0, N]).range([0, miniW]);
    const g = mctx;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, miniCanvas.width, miniCanvas.height);
    const lane = lanes.find((l) => l.id === opts.overviewLane) || lanes[0];
    const cols = Math.max(1, Math.round(miniW * dpr));
    const h = miniCanvas.height;
    g.fillStyle = hatchPattern(g, dpr);
    g.fillRect(0, 0, cols, h);
    if (lane.type === 'runs') {
      const fr = columnFractions(lane.runs, 0, N, cols);
      const rc = document.createElement('canvas');
      rc.width = cols;
      rc.height = 1;
      const rcx = rc.getContext('2d');
      const img = rcx.createImageData(cols, 1);
      const rgb = pal.labelRgb;
      for (let c = 0; c < cols; c++) {
        const s = fr[c * 4] + fr[c * 4 + 1] + fr[c * 4 + 2];
        if (s <= 1e-6) continue;
        for (let k = 0; k < 3; k++) {
          img.data[c * 4 + k] = (fr[c * 4] * rgb[0][k] + fr[c * 4 + 1] * rgb[1][k] + fr[c * 4 + 2] * rgb[2][k]) / s;
        }
        img.data[c * 4 + 3] = Math.round(s * 200);
      }
      rcx.putImageData(img, 0, 0);
      g.imageSmoothingEnabled = false;
      g.drawImage(rc, 0, 0, cols, 1, 0, 0, cols, h);
    }
    g.save();
    g.globalAlpha = 0.86;
    g.fillStyle = hatchPattern(g, dpr);
    for (const [a, b] of ungradedSpans()) g.fillRect(xm(a) * dpr, 0, (xm(b) - xm(a)) * dpr, h);
    g.restore();
    // The public window in the overview.
    g.fillStyle = pal.gt;
    const a = xm(win.start) * dpr;
    const b = xm(win.stop) * dpr;
    g.fillRect(a, 0, Math.max(dpr, b - a), 2 * dpr);
    g.fillRect(a, h - 2 * dpr, Math.max(dpr, b - a), 2 * dpr);
  }

  const brush = d3.brushX().on('brush end', (event) => {
    if (syncing || !event.sourceEvent) return;
    let sel = event.selection;
    if (!sel) {
      // A click in the overview centres the view on the clicked frame.
      const [px] = d3.pointer(event.sourceEvent, brushSvg);
      const d = x.domain();
      const half = (d[1] - d[0]) / 2;
      const c = Math.max(half, Math.min(N - half, xm.invert(px)));
      setDomain([c - half, c + half], 0);
      return;
    }
    setDomain([xm.invert(sel[0]), xm.invert(sel[1])], 0);
  });
  const brushG = d3.select(brushSvg).append('g').attr('class', 'brush');

  function syncBrush() {
    if (!miniW) return;
    const d = x.domain();
    syncing = true;
    brushG.call(brush.move, [xm(d[0]), Math.max(xm(d[0]) + 3, xm(d[1]))]);
    syncing = false;
  }

  // -- zoom --------------------------------------------------------------------------------
  const zoom = d3.zoom()
    .filter((event) => {
      if (event.type === 'wheel') return event.ctrlKey || event.metaKey;
      if (event.type === 'dblclick') return true;
      if (event.type === 'touchstart') return event.touches && event.touches.length > 1;
      return !event.button;
    })
    .on('zoom', (event) => {
      transform = event.transform;
      x = transform.rescaleX(x0);
      draw();
      syncBrush();
      if (opts.onDomain) opts.onDomain(x.domain());
    });

  const canvasSel = d3.select(canvas);
  canvasSel.call(zoom).on('dblclick.zoom', null);
  canvas.style.touchAction = 'pan-y';

  function transformFor(domain) {
    const [f0, f1] = domain;
    const span = Math.max(20, f1 - f0);
    const k = Math.max(1, N / span);
    return d3.zoomIdentity.scale(k).translate(-x0(f0), 0);
  }

  /**
   * Show the frames `[f0, f1)`.
   * @param {[number, number]} domain
   * @param {number} [ms]
   * @returns {Promise<void>}
   */
  function setDomain(domain, ms = 900) {
    const t = transformFor(domain);
    const d = dur(ms);
    if (!d) {
      canvasSel.call(zoom.transform, t);
      return Promise.resolve();
    }
    return canvasSel.transition().duration(d).ease(d3.easeCubicInOut).call(zoom.transform, t).end()
      .catch(() => {});
  }

  // -- pointer -----------------------------------------------------------------------------
  function laneAt(py) {
    for (const row of geo.rows) if (py >= row.y && py <= row.y + row.h) return row;
    return null;
  }

  function describe(frame) {
    const clock = matchClock(frame, opts.kickoffs, fps);
    let html = '<div class="tt-title tnum">frame ' + int(frame) + '</div>';
    html += '<div class="tt-row"><span>video</span><b>' + videoTime(frame, fps) + '</b></div>';
    if (clock) html += '<div class="tt-row"><span>match clock, half ' + clock.half + '</span><b>' + clock.text + '</b></div>';
    if (!isGraded(frame)) html += '<div class="tt-row"><span>not graded by the scorer</span><b>X</b></div>';
    for (const lane of lanes) {
      let v;
      if (lane.type === 'runs') {
        const inRegion = !lane.region || (frame >= lane.region[0] && frame < lane.region[1]);
        v = inRegion ? LETTER_NAME[labelAt(lane.runs, frame)] : 'not public';
        if (inRegion) v = '<span class="chip chip--' + v.toLowerCase() + '">' + v + '</span>';
      } else {
        const b = (lane.bins || []).find((r) => frame >= r.start && frame < r.stop);
        v = b ? 'H ' + pct(b.share_h, 0) + ' / A ' + pct(b.share_a, 0) + ' / D ' + pct(b.share_d, 0) : 'X';
      }
      html += '<div class="tt-row"><span>' + esc(lane.title) + '</span><b>' + v + '</b></div>';
    }
    return html;
  }

  canvas.addEventListener('pointermove', (e) => {
    const r = canvas.getBoundingClientRect();
    const px = e.clientX - r.left;
    const py = e.clientY - r.top;
    if (!geo || py < geo.topH - 8 || py > geo.axisY) {
      hoverX = null;
      hideTip();
      draw();
      return;
    }
    hoverX = px;
    const frame = Math.max(0, Math.min(N - 1, Math.floor(x.invert(px))));
    readout.textContent = 'frame ' + int(frame) + '  ' + videoTime(frame, fps);
    showTip(describe(frame), e.clientX, e.clientY);
    draw();
  });
  canvas.addEventListener('pointerleave', () => {
    hoverX = null;
    hideTip();
    readout.textContent = '';
    draw();
  });
  canvas.addEventListener('click', (e) => {
    const r = canvas.getBoundingClientRect();
    const py = e.clientY - r.top;
    if (!geo || !laneAt(py)) return;
    const frame = Math.max(0, Math.min(N - 1, Math.floor(x.invert(e.clientX - r.left))));
    if (opts.onSeek) opts.onSeek(frame);
  });

  container.querySelector('.ribbon-tools').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-act]');
    if (!b) return;
    const d = x.domain();
    const c = (d[0] + d[1]) / 2;
    const span = d[1] - d[0];
    if (b.dataset.act === 'full') setDomain([0, N]);
    else if (b.dataset.act === 'window') setDomain([win.start, win.stop]);
    else if (b.dataset.act === 'in') setDomain([c - span / 6, c + span / 6], 500);
    else if (b.dataset.act === 'out') setDomain([c - span * 1.5, c + span * 1.5], 500);
  });

  // -- public API --------------------------------------------------------------------------
  let focusRaf = 0;
  function setFocus(id) {
    focusId = id || null;
    titles.querySelectorAll('.ribbon-title').forEach((t) => {
      t.classList.toggle('is-dim', !!focusId && t.dataset.lane !== focusId);
      t.classList.toggle('is-focus', !!focusId && t.dataset.lane === focusId);
    });
    const target = focusId ? 1 : 0;
    const from = focusAlpha;
    const ms = dur(350);
    cancelAnimationFrame(focusRaf);
    if (!ms) {
      focusAlpha = target;
      draw();
      return;
    }
    const t0 = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - t0) / ms);
      focusAlpha = from + (target - from) * easeOut(t);
      draw();
      if (t < 1) focusRaf = requestAnimationFrame(tick);
    };
    focusRaf = requestAnimationFrame(tick);
  }

  function drawIn(ms) {
    const d = dur(ms === undefined ? 2400 : ms);
    if (!d) {
      reveal = 1;
      draw();
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      const t0 = performance.now();
      reveal = 0;
      const tick = (now) => {
        const t = Math.min(1, (now - t0) / d);
        reveal = easeOut(t);
        draw();
        if (t < 1) requestAnimationFrame(tick);
        else {
          reveal = 1;
          draw();
          resolve();
        }
      };
      requestAnimationFrame(tick);
    });
  }

  const stopObs = observeSize(container, resize);

  return {
    element: container,
    setFocus,
    setDomain,
    zoomToWindow: (ms) => setDomain([win.start, win.stop], ms),
    reset: (ms) => setDomain([0, N], ms),
    setPlayhead(frame) {
      playhead = frame === null || frame === undefined ? null : frame;
      draw();
    },
    hide() {
      reveal = 0;
      draw();
    },
    drawIn,
    domain: () => x.domain(),
    redraw() {
      cacheKey = '';
      draw();
    },
    destroy() {
      stopObs();
      canvasSel.on('.zoom', null);
    },
  };
}
