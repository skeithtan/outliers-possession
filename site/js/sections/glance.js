/**
 * At a glance: the five key numbers of the glance page of the technical document
 * in the same order.
 *
 * The external numbers are the engine's, from `LEARNED_EXTERNAL` (learned_external.js): the gain
 * in spellF1 over the rule engine with each provider unseen, and on the PFF design matches. The
 * match numbers come from `sanity.json`, the public window from `scores.json`. The public window
 * comes last: it was scored once, for information.
 *
 * @module sections/glance
 */

import { int, pct, fixed, signed, esc } from '../util/format.js';
import { el } from '../util/dom.js';
import { LEARNED_EXTERNAL } from '../learned_external.js';

const LEFT_OUT = 'Unseen provider: no model that labels a match saw any match of its provider.';
const TWO_SECOND = 'The 2-second labels: a team’s run under 2 s inside live play does not change the ' +
  'possession. The 43 PFF design matches helped us compare designs, so they are not an untouched test.';

/**
 * The tiles as `{value, label, note, kind, tip}` rows. A missing file leaves its tiles out.
 * @param {Object} bundle
 * @returns {Object[]}
 */
export function glanceTiles(bundle) {
  const tiles = [];
  const ext = Object.fromEntries(LEARNED_EXTERNAL.map((r) => [r.id, r]));
  const gainTile = (r, label) => ({
    value: signed(r.gain, 3), label,
    note: 'gain in spellF1 over the rule engine, above it on ' + r.above + ' of ' + r.n,
    kind: 'ext', tip: LEFT_OUT,
  });
  if (ext.metrica_left_out) tiles.push(gainTile(ext.metrica_left_out, 'Metrica sample games, unseen provider'));
  if (ext.dfl_left_out) tiles.push(gainTile(ext.dfl_left_out, 'Sportec Open DFL, unseen provider'));
  const pff = ext.pff_benchmark_2s;
  if (pff) {
    tiles.push({
      value: signed(pff.gain, 3), label: pff.n + ' PFF design matches, 2-second labels',
      note: 'gain over the rule engine, above it on ' + pff.above + ' of ' + pff.n + '. spellF1 ' +
        fixed(pff.engine, 3) + ' against ' + fixed(pff.rule, 3),
      kind: 'ext', tip: TWO_SECOND,
    });
  }
  const ours = bundle.sanity && bundle.sanity.ours && bundle.sanity.ours.total;
  if (ours) {
    tiles.push({
      value: int(ours.possessions), label: 'This match, possessions',
      note: 'D ' + pct(ours.d_share, 1) + ' in ' + int(ours.d_segments) + ' stoppages, ' +
        (ours.x_graded === 0 ? 'no graded X' : int(ours.x_graded) + ' graded X'),
      kind: 'match', tip: '',
    });
  }
  const v = bundle.manifest.version;
  const sc = bundle.scores && bundle.scores[v];
  const base = bundle.scores && bundle.scores.baseline_smooth50;
  if (sc) {
    const b = sc.bootstrap || {};
    tiles.push({
      value: fixed(sc.spell_f1_mean, 4), label: 'Public window, spellF1',
      note: 'scored once, for information. 5 to 95%: ' + fixed(b.lo, 2) + ' to ' + fixed(b.hi, 2) +
        (base ? '. baseline_smooth50 ' + fixed(base.spell_f1_mean, 3) : ''),
      kind: 'check', tip: 'Scored once, after the settings were final.',
    });
  }
  return tiles;
}

/**
 * @param {HTMLElement} root  The `#glance` section.
 * @param {{bundle: Object}} ctx
 */
export function mount(root, ctx) {
  const host = root.querySelector('#glance-fig');
  if (!host) return;
  glanceTiles(ctx.bundle).forEach((t) => {
    const tile = el('div', 'gl-tile gl-tile--' + t.kind);
    if (t.tip) tile.title = t.tip;
    tile.innerHTML =
      '<div class="gl-label">' + esc(t.label) + '</div>' +
      '<div class="gl-value tnum' + (t.value.length > 7 ? ' gl-value--long' : '') + '">' + esc(t.value) + '</div>' +
      '<div class="gl-note">' + esc(t.note) + '</div>';
    host.appendChild(tile);
  });
}
