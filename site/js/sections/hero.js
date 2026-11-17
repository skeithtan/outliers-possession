/**
 * Hero: the whole match on one ribbon, and two full-match counters against the baseline.
 *
 * Hosted mode draws the organisers' baseline at frame resolution, our per-minute share stripe,
 * and our labels and the ground truth inside the public window only. Local mode draws our
 * full-match labels at frame resolution from the local bundle. Every number comes from
 * `sanity.json`. The headline scores sit in the glance section, led by the external matches.
 *
 * @module sections/hero
 */

import { createRibbon } from '../ribbon.js';
import { int, pct, esc } from '../util/format.js';
import { countUp, el, dur, onFirstView } from '../util/dom.js';
import { palette } from '../util/colors.js';

/**
 * @param {HTMLElement} root  The `#hero` section.
 * @param {{bundle: Object, local: (Object|null), state: Object, d3: Object, mode: string}} ctx
 */
export function mount(root, ctx) {
  const { bundle, local, state, d3, mode } = ctx;
  const m = bundle.manifest;
  const pal = palette();
  const win = m.public_window;
  const base = bundle.baselines.smooth50;
  const lw = bundle.labelsWindow;
  const ours = bundle.sanity.ours.total;
  const baseStats = bundle.sanity.baseline_smooth50.total;

  root.querySelectorAll('[data-bind="basePoss"]').forEach((n) => { n.textContent = int(baseStats.possessions); });
  root.querySelectorAll('[data-bind="oursPoss"]').forEach((n) => { n.textContent = int(ours.possessions); });

  // -- lanes ------------------------------------------------------------------------------
  const lanes = [{
    id: 'baseline',
    title: 'Organisers’ baseline',
    sub: 'smooth50, every frame',
    type: 'runs',
    runs: base.runs,
    accent: pal.baseline,
  }];
  if (local) {
    lanes.push({
      id: 'ours',
      title: 'Ours, the engine',
      sub: 'every frame, local bundle',
      type: 'runs',
      runs: local.labelsOurs,
      accent: pal.text,
      weight: 1.25,
    });
  } else {
    lanes.push({
      id: 'ours',
      title: 'Ours, per minute',
      sub: 'share of H, D and A in 60 s',
      type: 'bins',
      bins: bundle.minuteBins.ours,
      accent: pal.text,
      weight: 1.25,
    });
    lanes.push({
      id: 'ours-window',
      title: 'Ours, every frame',
      sub: 'public window only',
      type: 'runs',
      runs: lw.ours,
      region: [win.start, win.stop],
      accent: pal.text,
      weight: 0.8,
    });
  }
  lanes.push({
    id: 'gt',
    title: 'Ground truth',
    sub: 'public window only',
    type: 'runs',
    runs: lw.gt_public,
    region: [win.start, win.stop],
    regionNote: 'no public ground truth',
    accent: pal.gt,
    weight: 0.8,
  });

  const ribbonHost = root.querySelector('#hero-ribbon');
  const ribbon = createRibbon(ribbonHost, {
    d3,
    nFrames: m.n_frames,
    fps: m.fps,
    window: win,
    gradedBlocks: m.graded_blocks,
    kickoffs: m.kickoffs,
    lanes,
    overviewLane: local ? 'ours' : 'baseline',
    onSeek: (frame) => state.set({ frame }),
  });
  ribbon.setPlayhead(state.get().frame);
  state.on('frame', (f) => ribbon.setPlayhead(f));

  // -- counters ---------------------------------------------------------------------------
  const counters = root.querySelector('#hero-counters');
  const tiles = [
    {
      label: 'Possessions',
      ours: ours.possessions,
      base: baseStats.possessions,
      fmt: (v) => int(v),
      note: 'full match, our sanity range ' + bundle.sanity.limits.possessions.join(' to '),
    },
    {
      label: 'Dead-ball share',
      ours: ours.d_share,
      base: baseStats.d_share,
      fmt: (v) => pct(v, 1),
      note: 'full match, our sanity range ' + bundle.sanity.limits.d_share.map((v) => pct(v, 0)).join(' to '),
    },
  ];
  const valueEls = [];
  tiles.forEach((t) => {
    const max = Math.max(t.ours || 0, t.base || 0) || 1;
    const tile = el('div', 'counter');
    tile.innerHTML =
      '<div class="counter-label">' + esc(t.label) + '</div>' +
      '<div class="counter-value display tnum">' + t.fmt(0) + '</div>' +
      '<div class="counter-bars" aria-hidden="true">' +
      '<span class="cb cb--ours" style="--w:' + ((t.ours || 0) / max).toFixed(4) + '"></span>' +
      '<span class="cb cb--base" style="--w:' + ((t.base || 0) / max).toFixed(4) + '"></span></div>' +
      '<div class="counter-base"><i class="sw sw--base"></i>baseline <b class="tnum">' +
      (t.baseFmt || t.fmt)(t.base) + '</b><span class="counter-note">' + esc(t.note) + '</span></div>';
    counters.appendChild(tile);
    valueEls.push([tile.querySelector('.counter-value'), t]);
  });

  let counted = false;
  function runCounters() {
    if (counted) return;
    counted = true;
    counters.classList.add('is-on');
    valueEls.forEach(([node, t], i) => {
      setTimeout(() => countUp(node, t.ours, { duration: 1500, format: t.fmt }), dur(i * 140));
    });
  }

  // -- caption from the active step -------------------------------------------------------
  const caption = root.querySelector('#hero-caption');
  const steps = Array.from(root.querySelectorAll('.step[data-step]'));
  const dots = el('div', 'caption-dots');
  steps.forEach(() => dots.appendChild(el('i')));
  const capText = el('div', 'caption-text');
  caption.appendChild(dots);
  caption.appendChild(capText);

  let shown = -1;
  function showStep(i) {
    if (i === shown) return;
    const copy = steps[i] && steps[i].querySelector('.step-copy');
    if (!copy) return;
    shown = i;
    Array.from(dots.children).forEach((d, k) => d.classList.toggle('is-on', k === i));
    capText.classList.remove('is-in');
    // Restart the fade on the next frame.
    requestAnimationFrame(() => {
      capText.innerHTML = copy.innerHTML;
      capText.classList.add('is-in');
    });
  }

  let zoomed = false;
  function applyStep(i) {
    showStep(i);
    if (i === 1) ribbon.setFocus('baseline');
    else if (i === 2) ribbon.setFocus('ours');
    else ribbon.setFocus(null);
    if (i >= 3 && !zoomed) {
      zoomed = true;
      ribbon.zoomToWindow(1400);
    } else if (i < 3 && zoomed) {
      zoomed = false;
      ribbon.reset(1000);
    }
  }

  state.on('activeStep', (s) => {
    if (s.section === 'hero') applyStep(s.step);
  });
  showStep(0);

  // Draw the lanes in, then count up, the first time the hero is on screen.
  // Wait for the loading screen to go, so the draw-in is seen.
  ribbon.hide();
  const start = () => onFirstView(ribbonHost, () => {
    ribbon.drawIn(2400).then(runCounters);
  }, 0.2);
  if (document.documentElement.dataset.ready === '1') start();
  else window.addEventListener('story-ready', start, { once: true });

  root.dataset.mode = mode;
}
