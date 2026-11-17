/**
 * Small DOM helpers shared by the sections: motion preference, count-up numbers, a shared
 * tooltip, size observers and copy buttons.
 *
 * @module util/dom
 */

const motionQuery = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;

/** @returns {boolean} True when the viewer asks for reduced motion. */
export function reducedMotion() {
  return !!(motionQuery && motionQuery.matches);
}

/**
 * Duration in ms, or 0 under reduced motion.
 * @param {number} ms
 * @returns {number}
 */
export function dur(ms) {
  return reducedMotion() ? 0 : ms;
}

/** Cubic ease out. */
export function easeOut(t) {
  return 1 - Math.pow(1 - t, 3);
}

/**
 * Animate a number into an element. Under reduced motion it writes the final value at once.
 * @param {Element} el
 * @param {number} to
 * @param {{from?: number, duration?: number, format: function(number): string}} opts
 * @returns {Promise<void>}
 */
export function countUp(el, to, opts) {
  const from = opts.from || 0;
  const ms = dur(opts.duration === undefined ? 1400 : opts.duration);
  if (!ms || to === null || to === undefined || Number.isNaN(to)) {
    el.textContent = opts.format(to);
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    const t0 = performance.now();
    function tick(now) {
      const t = Math.min(1, (now - t0) / ms);
      el.textContent = opts.format(from + (to - from) * easeOut(t));
      if (t < 1) requestAnimationFrame(tick);
      else {
        el.textContent = opts.format(to);
        resolve();
      }
    }
    requestAnimationFrame(tick);
  });
}

let tipEl = null;

/**
 * Show the shared tooltip near a pointer position.
 * @param {string} html  Trusted HTML built by the caller with escaped values.
 * @param {number} clientX
 * @param {number} clientY
 */
export function showTip(html, clientX, clientY) {
  if (!tipEl) {
    tipEl = document.createElement('div');
    tipEl.className = 'tooltip';
    tipEl.setAttribute('role', 'status');
    document.body.appendChild(tipEl);
  }
  tipEl.innerHTML = html;
  tipEl.classList.add('is-on');
  const pad = 14;
  const r = tipEl.getBoundingClientRect();
  let x = clientX + pad;
  let y = clientY + pad;
  if (x + r.width > window.innerWidth - 8) x = clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 8) y = clientY - r.height - pad;
  tipEl.style.transform = 'translate(' + Math.max(8, x) + 'px,' + Math.max(8, y) + 'px)';
}

/** Hide the shared tooltip. */
export function hideTip() {
  if (tipEl) tipEl.classList.remove('is-on');
}

/**
 * Call `fn(width, height)` now and on every size change of `el`, at most once per frame.
 * @param {Element} el
 * @param {function(number, number): void} fn
 * @returns {function(): void} Disconnect.
 */
export function observeSize(el, fn) {
  let raf = 0;
  let last = '';
  const ro = new ResizeObserver(() => {
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => {
      const w = el.clientWidth;
      const h = el.clientHeight;
      const key = w + 'x' + h;
      if (key === last || w === 0) return;
      last = key;
      try {
        fn(w, h);
      } catch (err) {
        console.error('render failed for', el.id || el.className, err);
      }
    });
  });
  ro.observe(el);
  return () => ro.disconnect();
}

/**
 * Copy text to the clipboard. Fall back to a selection and `execCommand` when the Clipboard API
 * is missing, for example on plain http from another host.
 * @param {string} text
 * @returns {Promise<boolean>}
 */
export async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch (err) {
    /* fall through to the selection fallback */
  }
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.setAttribute('readonly', '');
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  let ok = false;
  try {
    ok = document.execCommand('copy');
  } catch (err) {
    ok = false;
  }
  document.body.removeChild(ta);
  return ok;
}

/**
 * Make a copy button that copies `getText()` and shows "Copied" for 1.6 s.
 * @param {string} label
 * @param {function(): string} getText
 * @returns {HTMLButtonElement}
 */
export function copyButton(label, getText) {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'btn btn--copy';
  b.textContent = label;
  b.addEventListener('click', async () => {
    const ok = await copyText(getText());
    b.textContent = ok ? 'Copied' : 'Select and copy';
    b.classList.toggle('is-done', ok);
    setTimeout(() => {
      b.textContent = label;
      b.classList.remove('is-done');
    }, 1600);
  });
  return b;
}

/**
 * Create an element with a class and optional HTML.
 * @param {string} tag
 * @param {string} [cls]
 * @param {string} [html]
 * @returns {HTMLElement}
 */
export function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

/**
 * Run `fn` once, the first time `target` is at least `ratio` visible.
 * @param {Element} target
 * @param {function(): void} fn
 * @param {number} [ratio]
 */
export function onFirstView(target, fn, ratio = 0.25) {
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        io.disconnect();
        fn();
        return;
      }
    }
  }, { threshold: ratio });
  io.observe(target);
}

/**
 * Scroll an element into view, smooth unless the viewer asks for reduced motion.
 * @param {Element} target
 * @param {ScrollLogicalPosition} [block]
 */
export function scrollToEl(target, block = 'start') {
  if (!target) return;
  target.scrollIntoView({ behavior: reducedMotion() ? 'auto' : 'smooth', block });
}

/** Fill every `[data-bind]` element under `root` from `values`. */
export function bindText(root, values) {
  root.querySelectorAll('[data-bind]').forEach((node) => {
    const key = node.getAttribute('data-bind');
    if (values[key] !== undefined) node.textContent = values[key];
  });
}
