/**
 * The problem and the metric, in four scenes driven by the scroll steps.
 *
 *  A  step 0     the crossing chart: the public window picks the wrong baseline.
 *  B  steps 1-2  spell matching by IoU, with a threshold handle that the scroll moves.
 *  C  step 3     the shortest public spell and how fast its IoU falls when it shifts.
 *  D  step 4     transition latency, baseline against ours.
 *
 * The spells come from `labels_window.json`. `util/spells.js` replicates the scorer, and the
 * module checks its counts against `scores.json` for every IoU threshold. The scorer sees only
 * the public window, because the public ground truth is X everywhere else and `valid_runs`
 * returns that one block.
 *
 * @module sections/problem
 */

import { decodeRuns } from '../util/rle.js';
import * as S from '../util/spells.js';
import { spellCounts, iouRow, tolRow } from '../data.js';
import { int, fixed, signed, esc, videoTime } from '../util/format.js';
import { observeSize, dur, showTip, hideTip, countUp, el } from '../util/dom.js';

/*
 * Organisers' published full-match spellF1 of the two baselines. They are not in the bundle.
 * Source: the organisers' EVALUATION.md, "The public window is not a leaderboard", and
 * starter/baseline.py: baseline_smooth50 0.2433, baseline_smooth125 0.2242.
 * The public values come from scores.json. The organisers publish them too (0.3256, 0.4242).
 */
const FULL_MATCH_SPELL_F1 = { baseline_smooth50: 0.2433, baseline_smooth125: 0.2242 };

const LETTER = ['H', 'A', 'D', 'X'];
/** Scroll progress within a step after which the step takes the IoU handle back from a drag. */
const TAU_RESUME_PROGRESS = 0.1;
const SCENES = ['crossing', 'matching', 'matching', 'shift', 'latency'];

/**
 * @param {HTMLElement} root
 * @param {{bundle: Object, state: Object, d3: Object}} ctx
 */
export function mount(root, ctx) {
  const { bundle, state, d3 } = ctx;
  const m = bundle.manifest;
  const N = m.n_frames;
  const win = m.public_window;
  const version = m.version;
  const lw = bundle.labelsWindow;
  const fig = root.querySelector('#problem-fig');

  // -- the scorer replica -----------------------------------------------------------------
  const gt = decodeRuns(lw.gt_public, N);
  const ours = decodeRuns(lw.ours, N);
  const base50 = decodeRuns(lw.baseline_smooth50, N);
  const prep = S.prepare(gt, ours);
  const check = verify(bundle.scores, version, gt, {
    [version]: ours,
    baseline_smooth50: base50,
    baseline_smooth125: decodeRuns(lw.baseline_smooth125, N),
  });

  // -- scene hosts ------------------------------------------------------------------------
  fig.classList.add('fig--scenes');
  const hosts = {};
  ['crossing', 'matching', 'shift', 'latency'].forEach((name) => {
    const h = el('div', 'scene scene--' + name);
    h.dataset.scene = name;
    fig.appendChild(h);
    hosts[name] = h;
  });

  let active = 'crossing';
  let tau = 0.5;
  let tauDragged = false;
  let tauDragAt = 0;
  let progressNow = 0;
  let shift = 0;
  let shiftDragged = false;
  let lastStep = -1;
  const drawn = {};

  function show(name) {
    active = name;
    Object.keys(hosts).forEach((k) => hosts[k].classList.toggle('is-on', k === name));
    if (!drawn[name]) {
      drawn[name] = true;
      render(name, true);
    }
  }

  function render(name, animate) {
    const h = hosts[name];
    const w = fig.clientWidth;
    const hh = fig.clientHeight;
    if (!w || !hh) return;
    if (name === 'crossing') renderCrossing(h, w, hh, animate);
    if (name === 'matching') renderMatching(h, w, hh, animate);
    if (name === 'shift') renderShift(h, w, hh);
    if (name === 'latency') renderLatency(h, w, hh, animate);
  }

  observeSize(fig, () => {
    Object.keys(drawn).forEach((k) => render(k, false));
  });

  state.on('activeStep', (s) => {
    if (s.section !== 'problem') return;
    const scene = SCENES[s.step] || 'crossing';
    if (s.step !== lastStep) {
      if (s.step === 2) tauDragged = false;
      if (s.step === 3) shiftDragged = false;
      lastStep = s.step;
    }
    progressNow = s.progress;
    // After a drag the step takes control back once the reader scrolls on by more than a tenth.
    if (tauDragged && Math.abs(s.progress - tauDragAt) > TAU_RESUME_PROGRESS) tauDragged = false;
    if (scene !== active || !drawn[scene]) show(scene);
    if (s.step === 1 && !tauDragged) setTau(0.5);
    if (s.step === 2 && !tauDragged) setTau(0.3 + 0.4 * s.progress);
    if (s.step === 3 && !shiftDragged) setShift(Math.round(Math.min(1, s.progress * 1.25) * 14));
  });
  show('crossing');

  // -- A: crossing chart ------------------------------------------------------------------
  function renderCrossing(host, w, h, animate) {
    host.innerHTML = '';
    const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
    const pad = { l: rem * 4, r: rem * 9.5, t: rem * 4.5, b: rem * 4 };
    const svg = d3.select(host).append('svg').attr('width', w).attr('height', h);
    svg.append('text').attr('class', 'svg-title').attr('x', pad.l - rem * 2.5).attr('y', rem * 2)
      .text('Organisers’ baselines, spellF1');
    svg.append('text').attr('class', 'svg-label').attr('x', pad.l - rem * 2.5).attr('y', rem * 3.1)
      .text('Same algorithm, two smoothing windows. The public window and the full match rank them in opposite orders.');
    const x = d3.scalePoint().domain(['public', 'full']).range([pad.l + rem * 2, w - pad.r]);
    const y = d3.scaleLinear().domain([0.15, 0.5]).range([h - pad.b, pad.t]);
    const grid = svg.append('g').attr('class', 'grid');
    y.ticks(7).forEach((t) => {
      grid.append('line').attr('x1', pad.l).attr('x2', w - pad.r + rem * 2).attr('y1', y(t)).attr('y2', y(t));
    });
    svg.append('g').attr('class', 'axis').attr('transform', 'translate(' + pad.l + ',0)')
      .call(d3.axisLeft(y).ticks(7).tickFormat(d3.format('.2f')).tickSize(0).tickPadding(8));
    const labels = {
      public: ['Public window', 'five minutes, ' + (bundle.scores[version] ? bundle.scores[version].n_gt_spells : 17) + ' possessions'],
      full: ['Full match', 'organisers’ published score'],
    };
    ['public', 'full'].forEach((k) => {
      const g = svg.append('g').attr('transform', 'translate(' + x(k) + ',' + (h - pad.b + rem * 1.6) + ')');
      g.append('text').attr('class', 'svg-title').attr('text-anchor', 'middle').text(labels[k][0]);
      g.append('text').attr('class', 'svg-label').attr('text-anchor', 'middle').attr('y', rem * 1.15).text(labels[k][1]);
      svg.append('line').attr('class', 'stem').attr('x1', x(k)).attr('x2', x(k)).attr('y1', pad.t - rem).attr('y2', h - pad.b);
    });
    const series = [
      { key: 'baseline_smooth125', name: 'smooth125', cls: 'line--b2' },
      { key: 'baseline_smooth50', name: 'smooth50', cls: 'line--b1' },
    ].map((s) => ({
      ...s,
      pub: bundle.scores[s.key] ? bundle.scores[s.key].spell_f1_mean : null,
      full: FULL_MATCH_SPELL_F1[s.key],
    })).filter((s) => s.pub !== null);
    const t = dur(animate ? 1400 : 0);
    series.forEach((s, i) => {
      const g = svg.append('g').attr('class', 'series ' + s.cls);
      const path = g.append('line').attr('class', 'series-line')
        .attr('x1', x('public')).attr('y1', y(s.pub)).attr('x2', x('public')).attr('y2', y(s.pub));
      path.transition().delay(t ? 300 + i * 250 : 0).duration(t).ease(d3.easeCubicOut)
        .attr('x2', x('full')).attr('y2', y(s.full));
      [['public', s.pub, 4], ['full', s.full, 4]].forEach(([k, v, digits], j) => {
        const dot = g.append('circle').attr('class', 'series-dot').attr('cx', x(k)).attr('cy', y(v)).attr('r', 0);
        dot.transition().delay(t ? 300 + i * 250 + j * t : 0).duration(dur(300)).attr('r', rem * 0.42);
        const left = k === 'public';
        g.append('text').attr('class', 'series-val tnum')
          .attr('x', x(k) + (left ? -rem * 0.9 : rem * 0.9)).attr('y', y(v) + rem * 0.35)
          .attr('text-anchor', left ? 'end' : 'start')
          .text(fixed(v, digits)).style('opacity', 0)
          .transition().delay(t ? 300 + i * 250 + j * t : 0).duration(dur(300)).style('opacity', 1);
      });
      g.append('text').attr('class', 'series-name').attr('x', x('full') + rem * 0.9)
        .attr('y', y(s.full) + (s.key === 'baseline_smooth50' ? -rem * 0.9 : rem * 1.5))
        .text(s.name).style('opacity', 0)
        .transition().delay(t ? 300 + i * 250 + t : 0).duration(dur(300)).style('opacity', 1);
    });
    // The crossing point.
    if (series.length === 2) {
      const [a, b] = series;
      const den = (a.pub - b.pub) - (a.full - b.full);
      if (den !== 0) {
        const u = (a.pub - b.pub) / den;
        const cx = x('public') + u * (x('full') - x('public'));
        const cy = y(a.pub + u * (a.full - a.pub));
        const c = svg.append('g').attr('class', 'cross').style('opacity', 0);
        c.append('circle').attr('cx', cx).attr('cy', cy).attr('r', rem * 0.9);
        c.append('text').attr('x', cx).attr('y', cy - rem * 1.5).attr('text-anchor', 'middle')
          .text('the order reverses');
        c.transition().delay(t ? 300 + t * 1.2 : 0).duration(dur(500)).style('opacity', 1);
      }
    }
  }

  // -- B: spell matching ------------------------------------------------------------------
  let matchApi = null;

  function renderMatching(host, w, h, animate) {
    host.innerHTML = '';
    const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
    const narrow = w < 560;
    const pad = { l: narrow ? rem * 3.2 : rem * 7, r: rem * 1.5, t: rem * 2.4 };
    const panelH = narrow ? rem * 12.5 : rem * 8.5;
    const plotH = Math.max(rem * 9, h - pad.t - panelH - rem * 1.5);
    const gtY = pad.t + plotH * 0.08;
    const barH = Math.max(rem * 1.4, plotH * 0.16);
    const prY = pad.t + plotH * 0.72 - barH;
    const x = d3.scaleLinear().domain([win.start, win.stop]).range([pad.l, w - pad.r]);
    const svg = d3.select(host).append('svg').attr('width', w).attr('height', h);

    // Lane titles.
    const lt = svg.append('g').attr('class', 'lane-titles');
    lt.append('text').attr('class', 'svg-title lane-gt').attr('x', pad.l - rem * 0.6).attr('y', gtY + barH / 2 + rem * 0.3)
      .attr('text-anchor', 'end').text(narrow ? 'GT' : 'Ground truth');
    lt.append('text').attr('class', 'svg-title').attr('x', pad.l - rem * 0.6).attr('y', prY + barH / 2 + rem * 0.3)
      .attr('text-anchor', 'end').text('Ours');

    // Time axis between the lanes.
    const axis = d3.axisBottom(x).tickValues(d3.range(win.start, win.stop + 1, 1500))
      .tickFormat((f) => ((f - win.start) / m.fps / 60).toFixed(0) + '′').tickSize(4);
    svg.append('g').attr('class', 'axis').attr('transform', 'translate(0,' + (prY + barH + rem * 0.5) + ')').call(axis);

    // D segments as thin context bars.
    const segG = svg.append('g').attr('class', 'dsegs');
    const dsegs = (codes) => S.segments(codes, prep.runs).filter((s) => s[2] === S.D);
    dsegs(gt).forEach(([a, b]) => segG.append('rect').attr('class', 'dseg').attr('x', x(a)).attr('width', x(b) - x(a))
      .attr('y', gtY + barH * 0.35).attr('height', barH * 0.3));
    dsegs(prep.filled).forEach(([a, b]) => segG.append('rect').attr('class', 'dseg').attr('x', x(a)).attr('width', x(b) - x(a))
      .attr('y', prY + barH * 0.35).attr('height', barH * 0.3));

    // Bands for every same-team overlap.
    const bandG = svg.append('g').attr('class', 'bands');
    const bands = bandG.selectAll('path').data(prep.overlaps).join('path')
      .attr('class', (d) => 'band band--' + LETTER[prep.gtSpells[d[1]][2]])
      .attr('d', (d) => {
        const g = prep.gtSpells[d[1]];
        const p = prep.predSpells[d[2]];
        const y0 = gtY + barH;
        const y1 = prY;
        const ym = (y0 + y1) / 2;
        return 'M' + x(g[0]) + ',' + y0 + 'L' + x(g[1]) + ',' + y0 +
          'C' + x(g[1]) + ',' + ym + ' ' + x(p[1]) + ',' + ym + ' ' + x(p[1]) + ',' + y1 +
          'L' + x(p[0]) + ',' + y1 +
          'C' + x(p[0]) + ',' + ym + ' ' + x(g[0]) + ',' + ym + ' ' + x(g[0]) + ',' + y0 + 'Z';
      });
    const iouLabels = bandG.selectAll('text').data(prep.overlaps).join('text')
      .attr('class', 'band-iou tnum')
      .attr('x', (d) => {
        const g = prep.gtSpells[d[1]];
        const p = prep.predSpells[d[2]];
        return (x(Math.max(g[0], p[0])) + x(Math.min(g[1], p[1]))) / 2;
      })
      .attr('y', (gtY + barH + prY) / 2 + rem * 0.3)
      .attr('text-anchor', 'middle')
      .text((d) => d[0].toFixed(2));

    // Spells.
    const spellG = svg.append('g');
    const gtRects = spellG.selectAll('rect.gt').data(prep.gtSpells).join('rect')
      .attr('class', (d) => 'spell gt lab-' + LETTER[d[2]])
      .attr('x', (d) => x(d[0]) + 0.5).attr('width', (d) => Math.max(1.5, x(d[1]) - x(d[0]) - 1))
      .attr('y', gtY).attr('height', barH).attr('rx', 2);
    const prRects = spellG.selectAll('rect.pr').data(prep.predSpells).join('rect')
      .attr('class', (d) => 'spell pr lab-' + LETTER[d[2]])
      .attr('x', (d) => x(d[0]) + 0.5).attr('width', (d) => Math.max(1.5, x(d[1]) - x(d[0]) - 1))
      .attr('y', prY).attr('height', barH).attr('rx', 2);
    spellG.append('line').attr('class', 'gt-rule').attr('x1', pad.l).attr('x2', w - pad.r)
      .attr('y1', gtY - 3).attr('y2', gtY - 3);

    function tip(e, html) {
      showTip(html, e.clientX, e.clientY);
    }
    gtRects.on('pointermove', (e, d) => tip(e, spellTip('Ground-truth spell', d)))
      .on('pointerleave', hideTip);
    prRects.on('pointermove', (e, d) => tip(e, spellTip('Our spell', d)))
      .on('pointerleave', hideTip);
    bands.on('pointermove', (e, d) => tip(e, '<div class="tt-title">IoU ' + d[0].toFixed(3) + '</div>' +
      '<div class="tt-row"><span>ground truth</span><b>' + int(prep.gtSpells[d[1]][0]) + ' to ' + int(prep.gtSpells[d[1]][1]) + '</b></div>' +
      '<div class="tt-row"><span>ours</span><b>' + int(prep.predSpells[d[2]][0]) + ' to ' + int(prep.predSpells[d[2]][1]) + '</b></div>'))
      .on('pointerleave', hideTip);

    // Threshold control and score panel.
    const panelY = h - panelH;
    const ctl = svg.append('g').attr('class', 'tau-ctl').attr('transform', 'translate(0,' + panelY + ')');
    const panelW = Math.min(w * 0.44, rem * 26);
    const tx = d3.scaleLinear().domain([0.3, 0.7])
      .range([pad.l, narrow ? w - pad.r - rem : Math.max(pad.l + rem * 8, w - panelW - rem * 4.5)]).clamp(true);
    // The label sits above the handle readout so the two never overlap near 0.3.
    ctl.append('text').attr('class', 'svg-label').attr('x', pad.l).attr('y', -rem * 1.3).text('IoU threshold');
    ctl.append('line').attr('class', 'tau-track').attr('x1', tx(0.3)).attr('x2', tx(0.7)).attr('y1', rem * 1.4).attr('y2', rem * 1.4);
    const fill = ctl.append('line').attr('class', 'tau-fill').attr('x1', tx(0.3)).attr('y1', rem * 1.4).attr('y2', rem * 1.4);
    S.SPELL_IOUS.forEach((t) => {
      ctl.append('line').attr('class', 'tau-tick').attr('x1', tx(t)).attr('x2', tx(t)).attr('y1', rem * 0.9).attr('y2', rem * 1.9);
      ctl.append('text').attr('class', 'svg-label tnum').attr('x', tx(t)).attr('y', rem * 3).attr('text-anchor', 'middle').text(t.toFixed(1));
    });
    const handle = ctl.append('g').attr('class', 'tau-handle').attr('tabindex', 0).attr('role', 'slider')
      .attr('aria-label', 'IoU threshold').attr('aria-valuemin', 0.3).attr('aria-valuemax', 0.7);
    handle.append('circle').attr('r', rem * 0.7);
    const handleText = handle.append('text').attr('class', 'tnum').attr('y', -rem * 1.05).attr('text-anchor', 'middle');
    handle.call(d3.drag().on('start drag', (e) => {
      tauDragged = true;
      tauDragAt = progressNow;
      setTau(tx.invert(e.x));
    }));
    handle.on('keydown', (e) => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowUp') { tauDragged = true; tauDragAt = progressNow; setTau(tau + 0.01); e.preventDefault(); }
      if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') { tauDragged = true; tauDragAt = progressNow; setTau(tau - 0.01); e.preventDefault(); }
    });

    // Score panel on the right (below on narrow screens).
    const sp = el('div', 'match-panel');
    sp.innerHTML =
      '<div class="mp-counts">' +
      '<div><span class="mp-k">matched</span><b class="mp-tp tnum">0</b></div>' +
      '<div><span class="mp-k">missed</span><b class="mp-fn tnum">0</b></div>' +
      '<div><span class="mp-k">spurious</span><b class="mp-fp tnum">0</b></div>' +
      '<div><span class="mp-k">F1</span><b class="mp-f1 tnum">0</b></div></div>' +
      '<div class="mp-bars"></div>' +
      '<div class="mp-mean"><span class="mp-k">spellF1, the mean of five</span><b class="mp-meanv tnum">0.000</b></div>' +
      '<div class="mp-check ' + (check.ok ? 'is-ok' : 'is-bad') + '">' +
      (check.ok
        ? '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8.5l3.2 3L13 4.5"/></svg>verified against the scorer report'
        : 'differs from the scorer report: ' + esc(check.reason)) + '</div>';
    host.appendChild(sp);
    sp.style.top = (panelY - rem * 0.4) + 'px';
    // Keep the whole panel, the verified line included, inside the clipped figure.
    const spClip = (host.closest('.fig') || host).getBoundingClientRect();
    const spOver = sp.getBoundingClientRect().bottom - (Math.min(spClip.bottom, host.getBoundingClientRect().top + h) - rem * 0.75);
    if (spOver > 0) sp.style.top = (panelY - rem * 0.4 - spOver) + 'px';
    const bars = sp.querySelector('.mp-bars');
    const finals = S.SPELL_IOUS.map((t) => S.matchSpells(prep.overlaps, prep.gtSpells.length, prep.predSpells.length, t));
    finals.forEach((f, i) => {
      const b = el('div', 'mp-bar');
      b.style.setProperty('--h', f.f1.toFixed(4));
      b.innerHTML = '<i></i><span class="tnum">' + fixed(f.f1, 2) + '</span><em>' + S.SPELL_IOUS[i].toFixed(1) + '</em>';
      bars.appendChild(b);
    });

    const legend = svg.append('g').attr('class', 'match-legend').attr('transform', 'translate(' + pad.l + ',' + (pad.t - rem * 1.1) + ')');
    const items = [['sw-solid', 'matched'], ['sw-hollow', 'missed or spurious'], ['sw-band', 'band opacity = IoU']]
      .slice(0, narrow ? 2 : 3);
    let lx = 0;
    items.forEach(([cls, txt]) => {
      const g = legend.append('g').attr('transform', 'translate(' + lx + ',0)');
      g.append('rect').attr('class', cls).attr('width', rem * 0.8).attr('height', rem * 0.6).attr('y', -rem * 0.5).attr('rx', 2);
      g.append('text').attr('class', 'svg-label').attr('x', rem * 1.1).text(txt);
      lx += rem * 1.6 + txt.length * rem * 0.42;
    });

    let lastMean = 0;
    matchApi = {
      update(t) {
        const r = S.matchSpells(prep.overlaps, prep.gtSpells.length, prep.predSpells.length, t);
        const matchedSet = new Set(r.pairs.map((p) => p[1] + ':' + p[2]));
        bands.classed('is-matched', (d) => matchedSet.has(d[1] + ':' + d[2]))
          .classed('is-below', (d) => d[0] < t)
          .style('fill-opacity', (d) => (matchedSet.has(d[1] + ':' + d[2]) ? 0.25 + 0.6 * d[0] : 0.04 + 0.22 * d[0]));
        iouLabels.classed('is-on', (d) => matchedSet.has(d[1] + ':' + d[2]) && x(Math.min(prep.gtSpells[d[1]][1], prep.predSpells[d[2]][1])) - x(Math.max(prep.gtSpells[d[1]][0], prep.predSpells[d[2]][0])) > rem * 2);
        gtRects.classed('is-unmatched', (d, i) => !r.flags[i]);
        prRects.classed('is-unmatched', (d, j) => !r.predMatched.has(j));
        handle.attr('transform', 'translate(' + tx(t) + ',' + rem * 1.4 + ')').attr('aria-valuenow', t.toFixed(2));
        handleText.text('IoU ≥ ' + t.toFixed(2));
        fill.attr('x2', tx(t));
        sp.querySelector('.mp-tp').textContent = r.tp;
        sp.querySelector('.mp-fn').textContent = r.fn;
        sp.querySelector('.mp-fp').textContent = r.fp;
        sp.querySelector('.mp-f1').textContent = fixed(r.f1, 3);
        // Bars light up once the threshold reaches them. The mean covers the lit bars.
        let sum = 0;
        let n = 0;
        Array.from(bars.children).forEach((b, i) => {
          const on = S.SPELL_IOUS[i] <= t + 1e-9;
          b.classList.toggle('is-on', on);
          b.classList.toggle('is-current', Math.abs(S.SPELL_IOUS[i] - t) < 0.05);
          if (on) {
            sum += finals[i].f1;
            n += 1;
          }
        });
        const full = t >= 0.7 - 1e-9;
        const mean = full ? finals.reduce((a, f) => a + f.f1, 0) / finals.length : (n ? sum / n : 0);
        const meanEl = sp.querySelector('.mp-meanv');
        if (Math.abs(mean - lastMean) > 1e-9) {
          countUp(meanEl, mean, { from: lastMean, duration: 500, format: (v) => fixed(v, 3) });
          lastMean = mean;
        }
        sp.classList.toggle('is-final', full);
      },
    };
    matchApi.update(tau);
    if (animate) {
      gtRects.style('opacity', 0).transition().duration(dur(500)).delay((d, i) => dur(i * 30)).style('opacity', 1);
      prRects.style('opacity', 0).transition().duration(dur(500)).delay((d, i) => dur(200 + i * 30)).style('opacity', 1);
      bands.style('opacity', 0).transition().duration(dur(700)).delay(dur(600)).style('opacity', 1);
    }
  }

  function spellTip(title, d) {
    return '<div class="tt-title">' + title + ', <span class="chip chip--' + LETTER[d[2]].toLowerCase() + '">' + LETTER[d[2]] + '</span></div>' +
      '<div class="tt-row"><span>frames</span><b>' + int(d[0]) + ' to ' + int(d[1] - 1) + '</b></div>' +
      '<div class="tt-row"><span>length</span><b>' + ((d[1] - d[0]) / m.fps).toFixed(2) + ' s</b></div>';
  }

  function setTau(t) {
    tau = Math.max(0.3, Math.min(0.7, t));
    if (matchApi) matchApi.update(tau);
  }

  // -- C: the short spell -----------------------------------------------------------------
  const shortest = prep.gtSpells.reduce((best, s) => (!best || s[1] - s[0] < best[1] - best[0] ? s : best), null);
  const segsGt = S.segments(gt, prep.runs);
  let mergeNote = '';
  if (shortest) {
    root.querySelectorAll('[data-bind="shortLen"]').forEach((n) => { n.textContent = String(shortest[1] - shortest[0]); });
    const idx = segsGt.findIndex((s) => s[0] === shortest[0]);
    const prev = segsGt[idx - 1];
    const next = segsGt[idx + 1];
    if (prev && next && prev[2] === next[2] && (prev[2] === S.H || prev[2] === S.A)) {
      const a = S.iou(prev[0], prev[1], prev[0], next[1]);
      const b = S.iou(next[0], next[1], prev[0], next[1]);
      mergeNote = 'If the engine misses it, the two ' + LETTER[prev[2]] + ' spells on either side merge into one. ' +
        'Their IoUs against the merged spell are ' + a.toFixed(2) + ' and ' + b.toFixed(2) + '.';
    }
  }
  root.querySelectorAll('[data-bind="mergeNote"]').forEach((n) => { n.textContent = mergeNote; });

  let shiftApi = null;
  function renderShift(host, w, h) {
    host.innerHTML = '';
    if (!shortest) return;
    const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
    const L = shortest[1] - shortest[0];
    const ctxF = Math.max(30, L * 2.2);
    const f0 = shortest[0] - ctxF;
    const f1 = shortest[1] + ctxF;
    const pad = { l: rem * 2, r: rem * 2 };
    const x = d3.scaleLinear().domain([f0, f1]).range([pad.l, w - pad.r]);
    const svg = d3.select(host).append('svg').attr('width', w).attr('height', h);
    const y1 = h * 0.2;
    const y2 = h * 0.43;
    const bh = Math.max(rem * 2, h * 0.12);
    svg.append('text').attr('class', 'svg-title').attr('x', pad.l).attr('y', rem * 2)
      .text('Ground truth, frames ' + int(f0) + ' to ' + int(f1));
    // Frame grid.
    const grid = svg.append('g').attr('class', 'frame-grid');
    d3.range(Math.ceil(f0), Math.floor(f1) + 1).forEach((f) => {
      grid.append('line').attr('x1', x(f)).attr('x2', x(f)).attr('y1', y1 - rem * 0.3).attr('y2', y2 + bh + rem * 0.3)
        .attr('class', (f - shortest[0]) % 5 === 0 ? 'major' : '');
    });
    segsGt.filter((s) => s[1] > f0 && s[0] < f1).forEach((s) => {
      svg.append('rect').attr('class', 'lab-' + LETTER[s[2]] + (s === shortest || s[0] === shortest[0] ? ' short' : ' dim'))
        .attr('x', x(Math.max(f0, s[0]))).attr('width', x(Math.min(f1, s[1])) - x(Math.max(f0, s[0])))
        .attr('y', y1).attr('height', bh).attr('rx', 2);
    });
    svg.append('text').attr('class', 'svg-label').attr('x', x(shortest[0])).attr('y', y1 - rem * 0.6)
      .text(L + ' frames, ' + (L / m.fps).toFixed(2) + ' s');
    // The shifted copy.
    const copy = svg.append('g').attr('class', 'shift-copy').style('cursor', 'ew-resize');
    const copyRect = copy.append('rect').attr('class', 'lab-' + LETTER[shortest[2]]).attr('y', y2).attr('height', bh).attr('rx', 2)
      .attr('width', x(shortest[1]) - x(shortest[0]));
    const ghost = svg.append('rect').attr('class', 'shift-ghost').attr('x', x(shortest[0])).attr('width', x(shortest[1]) - x(shortest[0]))
      .attr('y', y2).attr('height', bh).attr('rx', 2);
    const overlapRect = svg.append('rect').attr('class', 'shift-overlap').attr('y', y2 - 3).attr('height', bh + 6);
    svg.append('text').attr('class', 'svg-label').attr('x', pad.l).attr('y', y2 + bh + rem * 1.4)
      .text('A copy of the spell, shifted. Drag it.');
    // The drag subject is the left edge of the copy, so e.x is its new left edge.
    copy.call(d3.drag()
      .subject(() => ({ x: x(shortest[0] + shift), y: y2 }))
      .on('start drag', (e) => {
        shiftDragged = true;
        setShift(x.invert(e.x) - shortest[0]);
      }));

    // Readout.
    const ro = svg.append('g').attr('transform', 'translate(' + pad.l + ',' + (h * 0.68) + ')');
    const shiftT = ro.append('text').attr('class', 'shift-val tnum').attr('y', 0);
    const iouT = ro.append('text').attr('class', 'shift-iou tnum').attr('x', w * 0.32).attr('y', 0);
    ro.append('text').attr('class', 'svg-label').attr('y', rem * 1.3).text('shift in frames');
    ro.append('text').attr('class', 'svg-label').attr('x', w * 0.32).attr('y', rem * 1.3).text('IoU with the original');
    const pills = ro.append('g').attr('transform', 'translate(' + (w * 0.58) + ',' + (-rem * 1.6) + ')');
    const pillW = Math.min(rem * 4, (w * 0.42 - pad.r) / 5 - rem * 0.4);
    const pillEls = S.SPELL_IOUS.map((t, i) => {
      const g = pills.append('g').attr('class', 'pill').attr('transform', 'translate(' + i * (pillW + rem * 0.4) + ',0)');
      g.append('rect').attr('width', pillW).attr('height', rem * 2).attr('rx', rem);
      g.append('text').attr('x', pillW / 2).attr('y', rem * 1.35).attr('text-anchor', 'middle').attr('class', 'tnum').text(t.toFixed(1));
      return g;
    });
    pills.append('text').attr('class', 'svg-label').attr('y', rem * 2.9).text('matched at these thresholds');

    shiftApi = {
      update(s) {
        const xs = x(shortest[0] + s);
        copyRect.attr('x', xs);
        ghost.style('opacity', s === 0 ? 0 : 1);
        const a = Math.max(shortest[0], shortest[0] + s);
        const b = Math.min(shortest[1], shortest[1] + s);
        overlapRect.attr('x', x(a)).attr('width', Math.max(0, x(b) - x(a)));
        const v = S.iou(shortest[0], shortest[1], shortest[0] + s, shortest[1] + s);
        shiftT.text((s > 0 ? '+' : s < 0 ? '−' : '') + Math.abs(s));
        iouT.text(v.toFixed(2));
        pillEls.forEach((g, i) => g.classed('is-pass', v >= S.SPELL_IOUS[i]));
      },
    };
    shiftApi.update(shift);
  }

  function setShift(s) {
    const L = shortest ? shortest[1] - shortest[0] : 0;
    shift = Math.max(-L - 4, Math.min(L + 4, Math.round(s)));
    if (shiftApi) shiftApi.update(shift);
  }

  // -- D: latency -------------------------------------------------------------------------
  function latencyRows(codes) {
    const runs = prep.runs;
    const filled = S.fillAbstentions(codes, runs);
    const gtT = S.transitions(gt, runs);
    const prT = S.transitions(filled, runs);
    const mt = S.matchTransitions(gtT, prT, 25);
    const pairOf = new Map(mt.pairs.map(([g, p]) => [g[0], p]));
    return gtT.map((g) => {
      const p = pairOf.get(g[0]);
      const near = p || S.nearestSameDestination(g, prT);
      return { frame: g[0], to: g[2], from: g[1], matched: !!p, latency: near ? near[0] - g[0] : null };
    });
  }

  function renderLatency(host, w, h, animate) {
    host.innerHTML = '';
    const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
    const rows = [
      { key: 'baseline_smooth50', name: 'Organisers’ baseline', cls: 'lat--base', data: latencyRows(base50) },
      { key: version, name: 'Ours, the engine', cls: 'lat--ours', data: latencyRows(ours) },
    ];
    const pad = { l: w < 560 ? rem * 1.5 : rem * 13, r: rem * 2, t: rem * 4.5, b: rem * 4 };
    const lim = 60;
    const x = d3.scaleLinear().domain([-lim, lim]).range([pad.l, w - pad.r]);
    const svg = d3.select(host).append('svg').attr('width', w).attr('height', h);
    svg.append('text').attr('class', 'svg-title').attr('x', rem * 1.5).attr('y', rem * 2)
      .text('Latency of each ground-truth transition, in frames');
    svg.append('text').attr('class', 'svg-label').attr('x', rem * 1.5).attr('y', rem * 3.1)
      .text('Filled: one of our transitions lies within 25 frames. Hollow: none does, and the dot sits at our nearest transition to the same label.');
    const bandTop = pad.t;
    const bandBot = h - pad.b;
    svg.append('rect').attr('class', 'tol25').attr('x', x(-25)).attr('width', x(25) - x(-25)).attr('y', bandTop).attr('height', bandBot - bandTop);
    svg.append('rect').attr('class', 'tol12').attr('x', x(-12)).attr('width', x(12) - x(-12)).attr('y', bandTop).attr('height', bandBot - bandTop);
    svg.append('line').attr('class', 'zero').attr('x1', x(0)).attr('x2', x(0)).attr('y1', bandTop).attr('y2', bandBot);
    [[-25, '−25'], [-12, '−12'], [12, '+12'], [25, '+25']].forEach(([v, t]) => {
      svg.append('text').attr('class', 'svg-label tnum').attr('x', x(v)).attr('y', bandTop - rem * 0.4).attr('text-anchor', 'middle').text(t);
    });
    svg.append('g').attr('class', 'axis').attr('transform', 'translate(0,' + bandBot + ')')
      .call(d3.axisBottom(x).tickValues([-60, -40, -20, 0, 20, 40, 60]).tickFormat((v) => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v)).tickSize(4));
    svg.append('text').attr('class', 'svg-label').attr('x', x(-lim)).attr('y', bandBot + rem * 2.6).text('early');
    svg.append('text').attr('class', 'svg-label').attr('x', x(lim)).attr('y', bandBot + rem * 2.6).attr('text-anchor', 'end').text('late');

    const rowH = (bandBot - bandTop) / rows.length;
    rows.forEach((row, ri) => {
      const cy = bandTop + rowH * (ri + 0.5);
      const g = svg.append('g').attr('class', 'lat-row ' + row.cls);
      if (pad.l > rem * 3) {
        g.append('text').attr('class', 'svg-title').attr('x', pad.l - rem).attr('y', cy - rem * 0.2).attr('text-anchor', 'end').text(row.name);
      }
      const matched = row.data.filter((d) => d.matched);
      const sc = bundle.scores[row.key];
      const tr = sc ? tolRow(sc.transitions, 25) : null;
      const meanLat = tr && tr.mean_latency !== null ? tr.mean_latency : null;
      const sub = matched.length + ' of ' + row.data.length + ' within 25 frames' + (meanLat !== null ? ', mean ' + signed(meanLat, 1) : '');
      g.append('text').attr('class', 'svg-label').attr('x', pad.l > rem * 3 ? pad.l - rem : x(-lim)).attr('y', pad.l > rem * 3 ? cy + rem * 1.1 : cy - rowH * 0.32)
        .attr('text-anchor', pad.l > rem * 3 ? 'end' : 'start').text(sub);
      // Beeswarm-like stacking: dots at the same rounded latency stack vertically.
      const used = new Map();
      const r = Math.max(4, rem * 0.38);
      row.data.forEach((d, i) => {
        if (d.latency === null) return;
        const v = Math.max(-lim, Math.min(lim, d.latency));
        const key = Math.round(x(v) / (r * 2.2));
        const k = used.get(key) || 0;
        used.set(key, k + 1);
        const dy = (k % 2 ? 1 : -1) * Math.ceil(k / 2) * r * 2.3;
        const clamped = d.latency !== v;
        const dot = g.append('g').attr('class', 'lat-dot' + (d.matched ? ' is-matched' : ' is-missed') + (clamped ? ' is-clamped' : ''))
          .attr('transform', 'translate(' + x(v) + ',' + (cy + dy) + ')');
        if (clamped) {
          dot.append('path').attr('d', v < 0 ? 'M' + (-r * 2.2) + ',0l' + r + ',' + (-r * 0.8) + 'v' + r * 1.6 + 'z' : 'M' + r * 2.2 + ',0l' + (-r) + ',' + (-r * 0.8) + 'v' + r * 1.6 + 'z');
        }
        dot.append('circle').attr('r', animate ? 0 : r)
          .transition().delay(dur(animate ? 200 + i * 40 + ri * 300 : 0)).duration(dur(300)).attr('r', r);
        dot.on('pointermove', (e) => showTip(
          '<div class="tt-title">' + LETTER[d.from] + ' to ' + LETTER[d.to] + ' at frame ' + int(d.frame) + '</div>' +
          '<div class="tt-row"><span>video</span><b>' + videoTime(d.frame, m.fps) + '</b></div>' +
          '<div class="tt-row"><span>latency</span><b>' + signed(d.latency, 0) + ' frames</b></div>' +
          '<div class="tt-row"><span>status</span><b>' + (d.matched ? 'matched' : 'missed') + '</b></div>', e.clientX, e.clientY))
          .on('pointerleave', hideTip);
      });
      if (meanLat !== null) {
        g.append('line').attr('class', 'lat-mean').attr('x1', x(meanLat)).attr('x2', x(meanLat))
          .attr('y1', cy - rowH * 0.38).attr('y2', cy + rowH * 0.38);
        g.append('text').attr('class', 'lat-mean-label tnum').attr('x', x(meanLat)).attr('y', cy - rowH * 0.42)
          .attr('text-anchor', 'middle').text('mean ' + signed(meanLat, 1));
      }
    });
  }
}

/**
 * Compare the replica with `scores.json` for every version whose labels the page holds.
 * Logs the result. Returns `{ok, reason}`.
 * @param {Object} scores
 * @param {string} version
 * @param {Uint8Array} gt
 * @param {Object<string, Uint8Array>} preds
 */
function verify(scores, version, gt, preds) {
  const bad = [];
  let compared = 0;
  Object.keys(preds).forEach((name) => {
    const sc = scores[name];
    if (!sc || !sc.spell_f1) return;
    const r = S.spellScores(gt, preds[name]);
    S.SPELL_IOUS.forEach((t) => {
      const row = iouRow(sc.spell_f1, t);
      if (!row) {
        bad.push(name + ' has no IoU ' + t);
        return;
      }
      const c = spellCounts(row);
      const mm = r.perIou[t];
      compared += 1;
      if (c.tp !== mm.tp || c.fp !== mm.fp || c.fn !== mm.fn || Math.abs(c.f1 - mm.f1) > 1e-9) {
        bad.push(name + ' at IoU ' + t + ': page ' + mm.tp + '/' + mm.fp + '/' + mm.fn + ', report ' + c.tp + '/' + c.fp + '/' + c.fn);
      }
    });
    if (Math.abs(r.meanF1 - sc.spell_f1_mean) > 1e-9) bad.push(name + ' mean ' + r.meanF1 + ' against ' + sc.spell_f1_mean);
  });
  const ok = bad.length === 0 && compared > 0 && !!scores[version];
  if (ok) {
    console.info('[problem] spell matching verified against the scorer report: ' + compared +
      ' IoU rows over ' + Object.keys(preds).filter((n) => scores[n]).join(', '));
  } else {
    console.warn('[problem] spell matching differs from the scorer report', bad);
  }
  return { ok, reason: bad[0] || (compared ? '' : 'no report to compare') };
}
