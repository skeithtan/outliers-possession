/**
 * Config constants: the origin class of every engine constant, and every field of
 * `engine/config.py` with its value and comment, from `config.json`.
 *
 * The comments are developer notes. They cite internal working notes that a visitor cannot read,
 * so the table hides them until the visitor switches them on. The search then also reads them.
 *
 * The class tiles count the constants of each class of our record of the source of every setting
 * (`config.json` `classes`). A search box and class chips filter the full table.
 *
 * The module mounts on `#rules`, the rule engine's part of section 04, which holds both hosts.
 *
 * @module sections/config
 */

import { int, esc } from '../util/format.js';
import { el } from '../util/dom.js';

/** Origin classes, in the order of the technical document, with a plain name for each. */
const CLASSES = [
  { id: 'E', label: 'external labelled matches', note: 'selected or measured on other matches' },
  { id: 'F', label: 'physics or the Laws', note: 'fixed by a physical bound or a Law of the Game' },
  { id: 'U', label: 'label-free statistics', note: 'measured on the unlabelled full match' },
  { id: 'D', label: 'defaults', note: 'never varied against any label' },
  { id: 'P', label: 'public labels', note: 'no setting comes from the public window' },
];

function fmtValue(v) {
  if (v === null || v === undefined) return 'None';
  if (typeof v === 'boolean') return v ? 'True' : 'False';
  if (Array.isArray(v)) return '(' + v.map(fmtValue).join(', ') + ')';
  if (typeof v === 'string') return '"' + v + '"';
  return String(v);
}

/**
 * @param {HTMLElement} root
 * @param {{bundle: Object}} ctx
 */
export function mount(root, ctx) {
  const cfg = ctx.bundle.config;
  const fig = root.querySelector('#config-fig');
  const classHost = root.querySelector('#config-classes');
  if (!cfg) return;
  const counts = (cfg.classes && cfg.classes.counts) || {};

  // -- class tiles -----------------------------------------------------------------------------
  if (classHost) {
    classHost.innerHTML = CLASSES.map((c) =>
      '<div class="cf-class cf-class--' + c.id + (counts[c.id] ? '' : ' is-zero') + '">' +
      '<b class="tnum">' + int(counts[c.id] || 0) + '</b>' +
      '<span class="cf-class-name"><i class="cls cls--' + c.id + '">' + c.id + '</i>' + esc(c.label) + '</span>' +
      '<em>' + esc(c.note) + '</em></div>').join('');
  }
  if (!fig) return;

  fig.innerHTML = `
    <div class="cf-head">
      <div class="cf-stat"><b class="tnum">${int(cfg.n_fields)}</b><span>constants</span></div>
      <div class="cf-stat cf-stat--hash"><b class="mono">${esc(cfg.config_hash)}</b><span>configuration hash</span></div>
      <label class="cf-notes"><input type="checkbox" class="cf-notes-box"> Show the developer notes</label>
      <label class="cf-search"><span class="sr">Search the constants</span>
        <input type="search" placeholder="Search name or value" autocomplete="off" spellcheck="false"></label>
    </div>
    <div class="cf-chips" role="group" aria-label="Filter by origin class"></div>
    <div class="cf-table cf-notes-off" role="table" aria-label="Engine constants"></div>
    <p class="cf-empty muted" hidden>No constant matches.</p>`;

  const chips = fig.querySelector('.cf-chips');
  const table = fig.querySelector('.cf-table');
  const input = fig.querySelector('input[type="search"]');
  const notesBox = fig.querySelector('.cf-notes-box');
  let notes = false;
  const empty = fig.querySelector('.cf-empty');
  let cls = null;
  let query = '';

  const all = el('button', 'chip-btn is-on', 'All <span class="tnum">' + int(cfg.n_fields) + '</span>');
  all.type = 'button';
  all.dataset.cls = '';
  chips.appendChild(all);
  CLASSES.forEach((c) => {
    if (!counts[c.id]) return;
    const b = el('button', 'chip-btn chip-btn--cls-' + c.id, esc(c.label) + ' <span class="tnum">' + counts[c.id] + '</span>');
    b.type = 'button';
    b.dataset.cls = c.id;
    chips.appendChild(b);
  });
  chips.addEventListener('click', (e) => {
    const b = e.target.closest('button[data-cls]');
    if (!b) return;
    cls = b.dataset.cls || null;
    chips.querySelectorAll('button').forEach((x) => x.classList.toggle('is-on', x === b));
    apply();
  });

  // Build the rows once and filter by toggling `hidden`.
  const label = new Map(CLASSES.map((c) => [c.id, c.label]));
  const rows = [];
  cfg.sections.forEach((sec) => {
    const group = el('div', 'cf-group');
    // The section titles of engine/config.py carry internal names. Show plain ones.
    const bare = sec.title.replace(/\s*\(.*\)\s*$/, '')
      .replace(/^GT semantics: Phase 2 calibration$/, 'reading the labels: offsets and conventions')
      .replace(/^K2: owner changes/, 'back-dating: owner changes')
      .replace(/^learned stage:/, 'the learned models:')
      .replace(/sanity vetoes/, 'sanity checks');
    const title = bare.charAt(0).toUpperCase() + bare.slice(1);
    group.innerHTML = '<div class="cf-group-head" role="row"><h4>' + esc(title) + '</h4><span class="cf-n tnum"></span></div>';
    const body = el('div', 'cf-rows');
    sec.fields.forEach((f) => {
      const row = el('div', 'cf-row');
      row.setAttribute('role', 'row');
      const c = f.class || '';
      row.innerHTML = '<div class="cf-name mono" role="cell">' + esc(f.name) + '</div>' +
        '<div class="cf-value mono tnum" role="cell">' + esc(fmtValue(f.value)) + '</div>' +
        '<div class="cf-comment" role="cell"><p>' + esc(f.comment) + '</p></div>' +
        '<div class="cf-srcs" role="cell">' + (c ? '<span class="cls cls--' + c + '" title="' + esc(label.get(c) || c) + '">' +
          esc(c) + '</span><span class="cf-cls-name">' + esc(label.get(c) || '') + '</span>' : '') + '</div>';
      row.addEventListener('click', () => row.classList.toggle('is-open'));
      body.appendChild(row);
      rows.push({
        row, group, cls: c,
        text: (f.name + ' ' + fmtValue(f.value)).toLowerCase(),
        note: String(f.comment || '').toLowerCase(),
      });
    });
    group.appendChild(body);
    table.appendChild(group);
  });

  function apply() {
    let shown = 0;
    const groups = new Map();
    rows.forEach((r) => {
      const hit = !query || r.text.indexOf(query) >= 0 || (notes && r.note.indexOf(query) >= 0);
      const ok = (!cls || r.cls === cls) && hit;
      r.row.hidden = !ok;
      if (ok) shown += 1;
      groups.set(r.group, (groups.get(r.group) || 0) + (ok ? 1 : 0));
    });
    groups.forEach((n, g) => {
      g.hidden = n === 0;
      g.querySelector('.cf-n').textContent = n + (n === 1 ? ' constant' : ' constants');
    });
    empty.hidden = shown > 0;
  }

  notesBox.addEventListener('change', () => {
    notes = notesBox.checked;
    table.classList.toggle('cf-notes-off', !notes);
    input.placeholder = notes ? 'Search name, value or note' : 'Search name or value';
    apply();
  });

  let timer = 0;
  input.addEventListener('input', () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      query = input.value.trim().toLowerCase();
      apply();
    }, 80);
  });

  apply();
}
