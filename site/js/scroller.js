/**
 * Scroll steps and arena mode.
 *
 * An IntersectionObserver watches every `.step[data-step]` inside each `.story-section`. The
 * active step is the step that crosses the middle of the viewport. The scroller writes
 * `state.activeStep = {section, step, progress}` with progress from 0 to 1 through the step.
 * The observer uses thresholds every 0.05. A scroll listener refines the progress of the active
 * step once per animation frame.
 *
 * Arena mode (`?arena=1` or key A) sets `state.arena`, scales the type by 1.25, moves to the next
 * step every 9 s and loops to the top after the last step. In arena mode the progress of each
 * step runs from 0 to 1 over the dwell time. Key F toggles fullscreen and Esc leaves arena mode.
 *
 * @module scroller
 */

import { state } from './state.js';
import { reducedMotion } from './util/dom.js';

/** Dwell time of one step in arena mode, ms (plan: 9 s). */
export const ARENA_DWELL_MS = 9000;
const THRESHOLDS = Array.from({ length: 21 }, (_, i) => i / 20);

let steps = [];
let io = null;
let current = null;
let raf = 0;
let arenaTimer = 0;
let arenaRaf = 0;
let arenaIndex = -1;
let arenaStepStart = 0;

/** Every step of the page in document order. */
function collect() {
  return Array.from(document.querySelectorAll('.story-section .step[data-step]'));
}

function sectionOf(step) {
  const sec = step.closest('.story-section');
  return sec ? (sec.dataset.section || sec.id) : '';
}

/** Progress of a step: 0 when its top reaches the middle line, 1 when its bottom does. */
function progressOf(step) {
  const r = step.getBoundingClientRect();
  const mid = window.innerHeight * 0.5;
  if (r.height <= 0) return 0;
  return Math.min(1, Math.max(0, (mid - r.top) / r.height));
}

/** The step that crosses the middle line, or the nearest one above it. */
function findActive() {
  const mid = window.innerHeight * 0.5;
  let best = null;
  for (const s of steps) {
    const r = s.getBoundingClientRect();
    if (r.top <= mid && r.bottom > mid) return s;
    if (r.bottom <= mid) best = s;
  }
  return best;
}

function emit(step, progress) {
  const section = sectionOf(step);
  const index = Number(step.dataset.step);
  if (current !== step) {
    if (current) current.classList.remove('is-active');
    step.classList.add('is-active');
    current = step;
  }
  state.set({ activeStep: { section, step: index, progress } });
}

/** Mark the top-bar link of the section that crosses the middle line. */
function updateNav() {
  const mid = window.innerHeight * 0.5;
  let id = '';
  document.querySelectorAll('.story-section').forEach((sec) => {
    const r = sec.getBoundingClientRect();
    // A section that continues a chapter lights the link of that chapter (data-nav-group).
    if (r.top <= mid && r.bottom > mid) id = sec.dataset.navGroup || sec.id;
  });
  document.querySelectorAll('.topnav a[data-nav]').forEach((a) => {
    a.classList.toggle('is-current', a.dataset.nav === id);
  });
}

function update() {
  raf = 0;
  updateProgressBar();
  updateNav();
  if (state.get().arena) return;
  const step = findActive();
  if (!step) return;
  emit(step, reducedMotion() ? 1 : progressOf(step));
}

function schedule() {
  if (!raf) raf = requestAnimationFrame(update);
}

function updateProgressBar() {
  const bar = document.getElementById('scroll-progress-bar');
  if (!bar) return;
  const max = document.documentElement.scrollHeight - window.innerHeight;
  const p = max > 0 ? window.scrollY / max : 0;
  bar.style.transform = 'scaleX(' + p.toFixed(4) + ')';
}

/**
 * Collect the steps again and observe them. Call it after a section adds steps to the page.
 */
export function refresh() {
  steps = collect();
  if (io) io.disconnect();
  io = new IntersectionObserver(schedule, { threshold: THRESHOLDS });
  steps.forEach((s) => io.observe(s));
  schedule();
}

/* -- arena ------------------------------------------------------------------------------- */

function arenaGo(index) {
  steps = collect();
  if (!steps.length) return;
  if (index >= steps.length) {
    // Loop: back to the top, then start again with the first step.
    index = 0;
  }
  arenaIndex = index;
  const step = steps[index];
  const r = step.getBoundingClientRect();
  const sticky = step.closest('.scrolly--caption');
  // Put the step on the middle line, so that the figure shows it.
  const y = window.scrollY + r.top - (sticky ? 0 : window.innerHeight * 0.5) + Math.min(r.height * 0.15, 80);
  window.scrollTo({ top: Math.max(0, index === 0 ? 0 : y), behavior: reducedMotion() ? 'auto' : 'smooth' });
  arenaStepStart = performance.now();
  emit(step, reducedMotion() ? 1 : 0);
  cancelAnimationFrame(arenaRaf);
  const tick = (now) => {
    if (!state.get().arena) return;
    const p = Math.min(1, (now - arenaStepStart) / ARENA_DWELL_MS);
    emit(step, reducedMotion() ? 1 : p);
    if (p < 1) arenaRaf = requestAnimationFrame(tick);
  };
  arenaRaf = requestAnimationFrame(tick);
  clearTimeout(arenaTimer);
  arenaTimer = setTimeout(() => arenaGo(arenaIndex + 1), ARENA_DWELL_MS);
}

/**
 * Turn arena mode on or off.
 * @param {boolean} on
 */
export function setArena(on) {
  state.set({ arena: !!on });
}

function applyArena(on) {
  document.documentElement.dataset.arena = on ? '1' : '0';
  const btn = document.getElementById('arena-toggle');
  if (btn) btn.setAttribute('aria-pressed', on ? 'true' : 'false');
  clearTimeout(arenaTimer);
  cancelAnimationFrame(arenaRaf);
  if (on) {
    const active = findActive();
    steps = collect();
    const start = active ? Math.max(0, steps.indexOf(active)) : 0;
    // Let the type scale settle before the first scroll.
    setTimeout(() => arenaGo(start), 120);
  } else {
    schedule();
  }
}

function toggleFullscreen() {
  const d = document;
  if (d.fullscreenElement) {
    if (d.exitFullscreen) d.exitFullscreen();
  } else if (d.documentElement.requestFullscreen) {
    d.documentElement.requestFullscreen().catch(() => {});
  }
}

function isTyping(e) {
  const t = e.target;
  return t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName));
}

/**
 * Start the scroller and the arena keys. Call it once after every section has mounted.
 */
export function start() {
  refresh();
  window.addEventListener('scroll', schedule, { passive: true });
  window.addEventListener('resize', schedule);
  state.on('arena', applyArena);

  const btn = document.getElementById('arena-toggle');
  if (btn) btn.addEventListener('click', () => setArena(!state.get().arena));

  document.addEventListener('keydown', (e) => {
    if (isTyping(e) || e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key;
    if (k === 'a' || k === 'A') {
      setArena(!state.get().arena);
    } else if (k === 'f' || k === 'F') {
      toggleFullscreen();
    } else if (k === 'Escape' && state.get().arena) {
      setArena(false);
      if (document.fullscreenElement && document.exitFullscreen) document.exitFullscreen();
    }
  });

  // In arena mode a wheel or touch by a person hands the page back to them.
  const stopArena = () => {
    if (state.get().arena) setArena(false);
  };
  window.addEventListener('wheel', stopArena, { passive: true });
  window.addEventListener('touchstart', stopArena, { passive: true });

  const params = new URLSearchParams(window.location.search);
  if (params.get('arena') === '1') setArena(true);
}
