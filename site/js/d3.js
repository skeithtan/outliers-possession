/**
 * D3 7.9.0 as an ES module export.
 *
 * `index.html` loads the pinned CDN file as a classic script before the module script. If the
 * CDN fails, its `onerror` handler loads `./vendor/d3.min.js`, the same file, and fires
 * `d3-ready` on `window`. `ready` resolves with `window.d3` in both cases.
 *
 * @module d3
 */

/** The D3 namespace. It is null until `ready` resolves when the fallback had to load. */
export let d3 = window.d3 || null;

/** Resolves with the D3 namespace. Rejects when neither copy loads within 15 s. */
export const ready = new Promise((resolve, reject) => {
  if (window.d3) {
    resolve(window.d3);
    return;
  }
  const done = () => {
    d3 = window.d3;
    if (d3) resolve(d3);
    else reject(new Error('D3 did not load'));
  };
  window.addEventListener('d3-ready', done, { once: true });
  window.addEventListener('d3-failed', () => reject(new Error('D3 did not load from the CDN or from vendor/')), { once: true });
  setTimeout(() => {
    if (window.d3) done();
    else reject(new Error('D3 did not load within 15 s'));
  }, 15000);
}).then((v) => {
  d3 = v;
  return v;
});

export default ready;
