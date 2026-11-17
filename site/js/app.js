/**
 * Application. `main.js` imports it, so a failed module load is caught there. Load the public
 * bundle, probe for the local bundle, set the mode, then mount every section with the same
 * context `{bundle, local, state, d3, mode}`.
 *
 * The engine section belongs to its own module. If `sections/engine.js` is missing or fails to
 * load, the page works without it.
 *
 * @module app
 */

import { ready as d3Ready } from './d3.js';
import { state } from './state.js';
import { loadPublic, probeLocal, mismatch } from './data.js';
import * as scroller from './scroller.js';
import { esc, int, teamDisplayName } from './util/format.js';
import { bindText } from './util/dom.js';

import { mount as mountHero } from './sections/hero.js';
import { mount as mountGlance } from './sections/glance.js';
import { mount as mountProblem } from './sections/problem.js';
import { mount as mountLearned } from './sections/learned.js';
import { mount as mountResults } from './sections/results.js';
import { mount as mountForensics } from './sections/forensics.js';
import { mount as mountSanity } from './sections/sanity.js';
import { mount as mountConfig } from './sections/config.js';
import { mount as mountRepro } from './sections/repro.js';
import { mount as mountTeam } from './sections/team.js';

const BADGE = {
  local: 'replay: real tracking (local)',
  public: 'replay: synthetic illustration',
};

function setMode(mode) {
  document.documentElement.dataset.mode = mode;
  const badge = document.getElementById('mode-badge');
  if (badge) {
    badge.dataset.mode = mode;
    badge.querySelector('.mode-text').textContent = BADGE[mode];
  }
}

function showBanner(m) {
  const b = document.getElementById('banner');
  if (!b || !m) return;
  b.innerHTML = '<strong>Local replay refused.</strong> ' + esc(m.reason) +
    ' Public md5 <code>' + esc(m.publicMd5) + '</code>, local md5 <code>' + esc(m.localMd5) +
    '</code>. Rebuild it with <code>python -m tools.viz.build_replay</code> for the same version. ' +
    'The page shows the synthetic replay.' +
    ' <button class="btn btn--ghost" type="button" id="banner-close">Dismiss</button>';
  b.hidden = false;
  b.querySelector('#banner-close').addEventListener('click', () => {
    b.hidden = true;
  });
}

/**
 * Copy the header text of each column of a `.rt` table into the `data-label` of its cells. On a
 * narrow screen the CSS turns each row into a card and shows the label before each cell.
 * @param {ParentNode} root
 */
function labelTableCells(root) {
  root.querySelectorAll('table.rt').forEach((table) => {
    const heads = Array.from(table.querySelectorAll('thead th')).map((th) => th.textContent.trim());
    table.querySelectorAll('tbody tr').forEach((tr) => {
      Array.from(tr.children).forEach((td, i) => {
        if (heads[i] && !td.hasAttribute('data-label')) td.setAttribute('data-label', heads[i]);
      });
    });
  });
}

/**
 * Mount one section. A failing section logs its error and leaves the others alone.
 * @param {string} id
 * @param {Function|null} mount
 * @param {Object} ctx
 */
async function mountSection(id, mount, ctx) {
  const el = document.getElementById(id);
  if (!el || typeof mount !== 'function') return;
  try {
    await mount(el, ctx);
  } catch (err) {
    console.error('section "' + id + '" failed to mount', err);
    el.classList.add('is-broken');
  }
}

export async function main() {
  const d3 = await d3Ready;
  const bundle = await loadPublic();
  const local = await probeLocal(bundle.manifest);
  const mode = local ? 'local' : 'public';
  setMode(mode);
  state.set({ mode });
  if (!local && mismatch) showBanner(mismatch);

  const m = bundle.manifest;
  bindText(document, {
    game: m.game,
    teamH: teamDisplayName(m.team_names.H),
    teamA: teamDisplayName(m.team_names.A),
    nFrames: int(m.n_frames),
    winStart: int(m.public_window.start),
    winLast: int(m.public_window.stop - 1),
    version: m.version,
    nFields: bundle.config ? int(bundle.config.n_fields) : '',
  });

  const ctx = { bundle, local, state, d3, mode };
  const { mount: mountEngine } = await import('./sections/engine.js').catch((err) => {
    console.warn('sections/engine.js is not available, the engine section stays empty', err);
    return { mount: null };
  });

  const order = [
    ['hero', mountHero],
    ['glance', mountGlance],
    ['problem', mountProblem],
    ['learned', mountLearned],
    ['results', mountResults],
    ['sanity', mountSanity],
    ['engine', mountEngine],
    ['rules', mountConfig],
    ['forensics', mountForensics],
    ['repro', mountRepro],
    ['team', mountTeam],
  ];
  for (const [id, fn] of order) {
    await mountSection(id, fn, ctx);
  }
  labelTableCells(document);

  const l = document.getElementById('loading');
  if (l) {
    l.classList.add('is-done');
    setTimeout(() => l.remove(), 700);
  }
  // Sections that animate on arrival wait for this, so the animation is not hidden by the
  // loading screen.
  document.documentElement.dataset.ready = '1';
  window.dispatchEvent(new Event('story-ready'));
  scroller.start();
  // A hash in the URL points into content that only exists now.
  if (window.location.hash) {
    const target = document.querySelector(window.location.hash);
    if (target) target.scrollIntoView();
  }
}
