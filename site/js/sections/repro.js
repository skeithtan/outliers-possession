/**
 * Reproduction card: the commands of Section 8 of the technical document, the git tag they check
 * out, the md5 of the submission, the pinned packages, and the configuration that switches the
 * learned models off with the md5 of the rule engine's labels.
 *
 * The tag, the expected md5s and the weights files are the ones the technical document gives. The
 * md5, the pins and the test count of the bundle come from `manifest.json`. If the bundle md5
 * differs from the document md5, the card shows both.
 *
 * @module sections/repro
 */

import { esc, int } from '../util/format.js';
import { copyButton, onFirstView, reducedMotion } from '../util/dom.js';

/**
 * Git ref and md5 of the reproduction, Section 8 of the technical document: the tag
 * `submission` on main.
 */
const DOC_COMMIT = 'submission';
const DOC_MD5 = 'd7f167a2401387bd8339d1bcbb292f2c';
/** The rule engine's labels, written with the learned models switched off: Section 8. */
const RULE_ENGINE_MD5 = '7844cad862e38d55cf9a3fc29b18ae85';
/* The content of off.json, the configuration of Section 8 that switches the learned models off. */
const OFF_CONFIG = JSON.stringify({ LEARNED_MODE: 'off' });
const OFF_COMMAND = 'python -m engine.run --config off.json --out rules.csv';
const REPO_URL = 'https://github.com/skeithtan/outliers-possession.git';

/** The commands of Section 8, as [command, comment] rows. */
const COMMANDS = [
  ['git clone ' + REPO_URL + ' repo && cd repo', ''],
  ['git checkout ' + DOC_COMMIT, ''],
  ['python3.9 -m venv .venv && source .venv/bin/activate', ''],
  ['pip install -r requirements.txt', ''],
  ['export MTAH_DATA_DIR=/path/to/mtah2027', 'or unpack the bundle into data/'],
  ['python -m engine.run --out "The Outliers.csv"', ''],
  ['python evaluation/validate_submission.py "The Outliers.csv"', ''],
];

/**
 * @param {HTMLElement} root
 * @param {{bundle: Object}} ctx
 */
export function mount(root, ctx) {
  const m = ctx.bundle.manifest;
  const r = m.repro || {};
  const fig = root.querySelector('#repro-fig');
  const pins = r.pins || {};
  const md5 = String(m.md5 || DOC_MD5);
  const differs = md5 !== DOC_MD5;
  const runtime = ['numpy', 'pandas'].filter((k) => pins[k]);

  fig.innerHTML = `
    <div class="rp-md5">
      <span class="rp-label">md5 of the submitted file</span>
      <div class="rp-hash mono" aria-label="${esc(md5)}"><span class="rp-typed"></span><span class="rp-caret" aria-hidden="true"></span></div>
      ${differs ? '<p class="rp-det rp-dirty">This bundle comes from a file with a different md5 than the technical document, <code>' + esc(DOC_MD5) + '</code>.</p>' : ''}
      <div class="rp-md5-tools"></div>
    </div>
    <div class="rp-grid">
      <div class="rp-cmd">
        <span class="rp-label">Run it from a fresh clone</span>
        <pre class="mono">${COMMANDS.map(([c, note]) => '<span class="rp-prompt">$ </span>' + esc(c) +
          (note ? '  <span class="rp-c"># ' + esc(note) + '</span>' : '')).join('\n')}</pre>
        <div class="rp-cmd-tools"></div>
        <p class="rp-det"><code>engine.run</code> writes and validates the CSV, then prints the sanity checks and the md5.
          It loads two weights files, <code>engine/models/learned_v7.npz</code> and <code>engine/models/learned_pf_v7.npz</code>, and never trains.
          It needs <code>numpy</code> and <code>pandas</code> only. There is no randomness, network or wall clock. The test
          <code>tests/test_repro.py</code> runs the engine twice and compares the outputs.
          This match takes about 1 minute on an Apple silicon Mac, with up to 3.6 GB of memory. The same md5 came out on macOS arm64 (two runs and a fresh clone), on Linux arm64 and on emulated Linux x86-64 (two runs each). On Intel and AMD hardware, with AVX2 and AVX-512, under Linux, Windows and macOS, the engine wrote the same bytes as the Mac on the three Metrica sample games. This match's data stays on our machine. There we reproduced the one arithmetic difference that we found between the two processor families, a fused multiply-add in one interpolation, and the md5 stayed the same.</p>
        <div class="rp-off">
          <span class="rp-label">The rule engine, one configuration setting away</span>
          <p class="rp-det">Save this configuration as <code>off.json</code>. It switches the learned models off, and the
            engine writes the rule engine's labels, md5 <code class="mono">${RULE_ENGINE_MD5}</code>.</p>
          <pre class="mono"><span class="rp-c"># off.json</span>
${esc(OFF_CONFIG)}
<span class="rp-prompt">$ </span>${esc(OFF_COMMAND)}</pre>
        </div>
      </div>
      <dl class="rp-facts">
        <div><dt>Tag</dt><dd class="mono">${esc(DOC_COMMIT)}</dd></div>
        <div><dt>Python</dt><dd class="mono">3.9 to 3.12</dd></div>
        <div><dt>Packages</dt><dd class="mono">${runtime.map((k) => esc(k) + '==' + esc(pins[k])).join('<br>')}</dd></div>
        <div><dt>Tests</dt><dd class="tnum">${int(r.n_tests)} test functions in <code>tests/</code></dd></div>
        <div><dt>Data</dt><dd>the organisers' bundle in <code>data/</code>, or <code>MTAH_DATA_DIR</code></dd></div>
      </dl>
    </div>`;

  fig.querySelector('.rp-md5-tools').appendChild(copyButton('Copy md5', () => md5));
  fig.querySelector('.rp-cmd-tools').appendChild(copyButton('Copy commands', () => COMMANDS.map(([c]) => c).join('\n')));

  const typed = fig.querySelector('.rp-typed');
  if (reducedMotion()) {
    typed.textContent = md5;
    fig.querySelector('.rp-hash').classList.add('is-done');
    return;
  }
  typed.textContent = '';
  onFirstView(fig, () => {
    let i = 0;
    const tick = () => {
      i += 1;
      typed.textContent = md5.slice(0, i);
      if (i < md5.length) setTimeout(tick, 28 + (i % 4 === 0 ? 30 : 0));
      else fig.querySelector('.rp-hash').classList.add('is-done');
    };
    setTimeout(tick, 250);
  }, 0.4);
}
