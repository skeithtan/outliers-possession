/**
 * Team: the five names from `manifest.json`, the link to the public repository on GitHub and the
 * address of this page.
 *
 * @module sections/team
 */

import { esc, teamDisplayName } from '../util/format.js';

/** The public repository, and the address of this page, which GitHub Pages serves from it. */
const REPO_URL = 'https://github.com/skeithtan/outliers-possession';
const PAGES_URL = 'https://skeithtan.github.io/outliers-possession/';

/**
 * @param {HTMLElement} root
 * @param {{bundle: Object}} ctx
 */
export function mount(root, ctx) {
  const m = ctx.bundle.manifest;
  const host = root.querySelector('#team-fig');
  const names = m.team || [];
  const initials = (n) => n.split(/\s+/).filter(Boolean).map((p) => p[0]).slice(0, 2).join('').toUpperCase();
  host.innerHTML = `
    <div class="tm">
      <p class="kicker"><span class="num">08</span> Team</p>
      <h2 class="tm-title">Built by five people</h2>
      <ul class="tm-names">
        ${names.map((n, i) => '<li style="--i:' + i + '"><span class="tm-mono" aria-hidden="true">' + esc(initials(n)) + '</span><span>' + esc(n) + '</span></li>').join('')}
      </ul>
      <div class="tm-links">
        <a class="tm-link tm-link--repo" href="${REPO_URL}" rel="noopener">
          <span class="tm-link-k">Repository</span><span class="tm-link-v mono">${esc(REPO_URL.replace(/^https:\/\//, ''))}</span></a>
        <a class="tm-link" href="${PAGES_URL}" rel="noopener">
          <span class="tm-link-k">This page</span><span class="tm-link-v mono">${esc(PAGES_URL.replace(/^https:\/\//, '').replace(/\/$/, ''))}</span></a>
      </div>
      <p class="tm-foot muted">More than a Hack 2027, Phase 1. ${esc(teamDisplayName(m.team_names.H))} v ${esc(teamDisplayName(m.team_names.A))}, ${esc(m.game)}.
        The hosted page holds no tracking positions and no full-match labels of ours. Charts drawn with D3 7.9.0.</p>
    </div>`;
}
