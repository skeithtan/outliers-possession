/**
 * On this match, without labels: four arc gauges of the sanity checks and the label-free
 * statistics behind them, then the learned models' own checks.
 *
 * The gauges and the cards come from `sanity.json`: `tools.sanity.sanity_stats()` for the
 * submitted engine and for `baseline_smooth50`, the `VETO_*` limits of `Config` and the expected
 * ranges of `tools/sanity.py`. The checks reject a broken output. They never rank two engines.
 * The learned models' stability and possession-count checks are not in the bundle. They come from
 * Section 3 of the technical document.
 *
 * The last card is the public window, from `scores.json`: scored once, for information, after the
 * settings were final. It comes after the label-free checks, as in Section 3 of the technical document.
 *
 * @module sections/sanity
 */

import { int, pct, fixed, signed, esc } from '../util/format.js';
import { observeSize, dur, el, onFirstView, countUp } from '../util/dom.js';

/*
 * Agreement of the public ground truth with the Smart Tagging team phases, measured in
 * RESEARCH-2 and quoted in Section 6 of the technical document. It is
 * not in the bundle, because it needs the ground truth and the tags together.
 */
const GT_TAG_AGREEMENT = 0.76;

/*
 * The two models' checks on this match, with limits written down before the run: Section 3 of the
 * technical document. The limits are 1.5 times the same figures on
 * SkillCorner, each match labelled by models that had not seen it. The possession range is the
 * lowest and the highest ratio of the engine's possession count to the rule engine's on the 28
 * external matches, scored the same way, whose median is 1.14. The models wrote 265 possessions
 * (0.950 times); the kickoff team rule removes one false possession, so the engine writes 264 against
 * the rule engine's 279: 0.9462 times, 0.0002 below the range.
 */
const STABILITY = [
  { label: 'The four networks give a different most likely class', value: 0.0518, limit: 0.0548 },
  { label: 'The four mirrored copies give a different most likely class', value: 0.0201, limit: 0.0211 },
];
const STABILITY_MAX = 0.06;
const POSSESSION_RATIO = { value: 0.9462, range: [0.9464, 1.387], median: 1.14, domain: [0.8, 1.5] };

/* The rule engine's spellF1 on the public window: Section 3 of the technical document. */
const RULE_ENGINE_WINDOW = 0.682;

/**
 * @param {HTMLElement} root
 * @param {{bundle: Object, d3: Object}} ctx
 */
export function mount(root, ctx) {
  const { bundle, d3 } = ctx;
  const s = bundle.sanity;
  const ours = s.ours;
  const base = s.baseline_smooth50;
  const lim = s.limits;
  const wat = s.watched || {};
  const fig = root.querySelector('#sanity-fig');

  fig.innerHTML = `
    <div class="sn-legend">
      <span><i class="sn-k sn-k--ours"></i>the engine</span>
      <span><i class="sn-k sn-k--base"></i>baseline: baseline_smooth50</span>
      <span><i class="sn-k sn-k--veto"></i>allowed range</span>
      <span><i class="sn-k sn-k--watch"></i>expected range: outside it we look closer, it rejects nothing</span>
    </div>
    <div class="sn-gauges"></div>
    <div class="sn-grid">
      <div class="sn-card sn-halves"><h4>Each half</h4><div class="sn-halves-body"></div></div>
      <div class="sn-card sn-dlen"><h4>Length of the stoppages</h4><div class="sn-dlen-body"></div></div>
      <div class="sn-card sn-tags"><h4>Agreement with Smart Tagging</h4><div class="sn-tags-body"></div></div>
    </div>
    <div class="sn-card sn-learned"><h4>The two models' own checks, with limits written down before the run</h4><div class="sl-body"></div></div>
    <div class="sn-check"></div>`;

  const gauges = [
    {
      key: 'possessions', title: 'Possessions', value: ours.total.possessions, base: base.total.possessions,
      domain: [0, 700], veto: lim.possessions, watch: wat.possessions, fmt: (v) => int(v),
      rule: 'allowed ' + lim.possessions.join(' to '),
    },
    {
      key: 'd_share', title: 'Dead-ball share', value: ours.total.d_share, base: base.total.d_share,
      domain: [0, 0.7], veto: lim.d_share, watch: null, fmt: (v) => pct(v, 1),
      rule: 'allowed ' + lim.d_share.map((v) => pct(v, 0)).join(' to '),
    },
    {
      key: 'd_segments', title: 'Stoppages', value: ours.total.d_segments, base: base.total.d_segments,
      domain: [0, 200], veto: [lim.min_d_segments, 200], watch: wat.d_segments, fmt: (v) => int(v),
      rule: 'at least ' + lim.min_d_segments,
    },
    {
      key: 'x_graded', title: 'X on graded frames', value: ours.total.x_graded, base: base.total.x_graded,
      domain: [0, 100], veto: [0, lim.max_graded_x], watch: null, fmt: (v) => int(v),
      rule: 'allowed ' + lim.max_graded_x,
    },
  ];

  const gaugeHost = fig.querySelector('.sn-gauges');
  const gaugeEls = gauges.map((g) => {
    const card = el('div', 'sn-gauge');
    gaugeHost.appendChild(card);
    return { g, card };
  });

  let seen = false;
  function renderGauges(animate) {
    gaugeEls.forEach(({ g, card }) => drawGauge(card, g, animate));
  }

  function drawGauge(card, g, animate) {
    card.innerHTML = '';
    const w = card.clientWidth;
    if (!w) return;
    const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
    const size = Math.min(w, rem * 16);
    const r = size * 0.42;
    const cx = w / 2;
    const cy = r + rem * 1.2;
    const h = cy + r * 0.62 + rem * 3.2;
    const a0 = -Math.PI * 2 / 3;
    const a1 = Math.PI * 2 / 3;
    const sc = d3.scaleLinear().domain(g.domain).range([a0, a1]).clamp(true);
    const svg = d3.select(card).append('svg').attr('width', w).attr('height', h).attr('role', 'img')
      .attr('aria-label', g.title + ': the engine ' + g.fmt(g.value) + ', baseline_smooth50 ' + g.fmt(g.base) + ', ' + g.rule);
    const gg = svg.append('g').attr('transform', 'translate(' + cx + ',' + cy + ')');
    const arc = d3.arc().innerRadius(r * 0.84).outerRadius(r).cornerRadius(2);
    gg.append('path').attr('class', 'g-track').attr('d', arc({ startAngle: a0, endAngle: a1 }));
    // Out-of-band zones are red, the band is green, the watched range is brighter.
    const vlo = sc(g.veto[0]);
    const vhi = sc(Math.min(g.veto[1], g.domain[1]));
    if (vlo > a0) gg.append('path').attr('class', 'g-out').attr('d', arc({ startAngle: a0, endAngle: vlo }));
    if (vhi < a1) gg.append('path').attr('class', 'g-out').attr('d', arc({ startAngle: vhi, endAngle: a1 }));
    if (vhi - vlo > 0.01) gg.append('path').attr('class', 'g-veto').attr('d', arc({ startAngle: vlo, endAngle: vhi }));
    else gg.append('path').attr('class', 'g-veto g-veto--tick').attr('d', arc({ startAngle: vlo - 0.02, endAngle: vlo + 0.02 }));
    if (g.watch) {
      const warc = d3.arc().innerRadius(r * 0.7).outerRadius(r * 0.8).cornerRadius(2);
      gg.append('path').attr('class', 'g-watch').attr('d', warc({ startAngle: sc(g.watch[0]), endAngle: sc(g.watch[1]) }));
    }
    // Scale ends.
    [[g.domain[0], a0], [g.domain[1], a1]].forEach(([v, a]) => {
      gg.append('text').attr('class', 'svg-label tnum').attr('text-anchor', 'middle')
        .attr('x', Math.sin(a) * (r + rem * 0.9)).attr('y', -Math.cos(a) * (r + rem * 0.9) + rem * 0.3).text(g.fmt(v));
    });
    // Needles.
    const needle = (cls, len, width) => {
      const n = gg.append('g').attr('class', cls);
      n.append('path').attr('d', 'M' + (-width) + ',0L0,' + (-len) + 'L' + width + ',0Z');
      n.append('circle').attr('r', width * 1.4);
      return n;
    };
    const baseN = needle('g-needle g-needle--base', r * 0.92, rem * 0.18);
    const ourN = needle('g-needle g-needle--ours', r * 0.98, rem * 0.26);
    const t = dur(animate ? 1600 : 0);
    const deg = (v) => (sc(v) * 180) / Math.PI;
    baseN.attr('transform', 'rotate(' + deg(animate ? g.domain[0] : g.base) + ')')
      .transition().duration(t).delay(t ? 200 : 0).ease(d3.easeCubicOut).attr('transform', 'rotate(' + deg(g.base) + ')');
    ourN.attr('transform', 'rotate(' + deg(animate ? g.domain[0] : g.value) + ')')
      .transition().duration(t).delay(t ? 500 : 0).ease(d3.easeElasticOut.amplitude(1).period(0.45)).attr('transform', 'rotate(' + deg(g.value) + ')');

    const pass = g.value >= g.veto[0] && g.value <= g.veto[1];
    const valueT = gg.append('text').attr('class', 'g-value tnum').attr('text-anchor', 'middle').attr('y', r * 0.42).text(g.fmt(animate ? g.domain[0] : g.value));
    if (animate) countUp(valueT.node(), g.value, { from: g.domain[0], duration: 1600, format: g.fmt });
    gg.append('text').attr('class', 'g-base tnum').attr('text-anchor', 'middle').attr('y', r * 0.42 + rem * 1.25)
      .text('baseline ' + g.fmt(g.base));
    svg.append('text').attr('class', 'g-title').attr('x', cx).attr('y', h - rem * 1.6).attr('text-anchor', 'middle').text(g.title);
    svg.append('text').attr('class', 'g-rule ' + (pass ? 'is-pass' : 'is-fail')).attr('x', cx).attr('y', h - rem * 0.35)
      .attr('text-anchor', 'middle').text((pass ? 'passes, ' : 'fails, ') + g.rule);
  }

  observeSize(gaugeHost, () => renderGauges(false));
  onFirstView(gaugeHost, () => {
    seen = true;
    renderGauges(true);
  }, 0.3);

  // -- each half ---------------------------------------------------------------------------
  const halves = fig.querySelector('.sn-halves-body');
  const hrows = [];
  ['1', '2'].forEach((hk) => {
    const o = ours.halves[hk];
    const b = base.halves[hk];
    if (!o || !b) return;
    hrows.push({ label: 'Half ' + hk + ', possessions', o: o.possessions, b: b.possessions, max: 350, fmt: (v) => int(v) });
    hrows.push({ label: 'Half ' + hk + ', D share', o: o.d_share, b: b.d_share, max: 0.7, fmt: (v) => pct(v, 1), limit: lim.d_share[1] });
  });
  halves.innerHTML = hrows.map((r) => {
    const lp = r.limit !== undefined ? '<span class="hb-limit" style="left:' + (r.limit / r.max * 100).toFixed(2) + '%" title="limit ' + r.fmt(r.limit) + '"></span>' : '';
    return '<div class="hb"><div class="hb-label">' + esc(r.label) + '</div>' +
      '<div class="hb-track"><span class="hb-bar hb-bar--ours" style="--w:' + Math.min(1, r.o / r.max).toFixed(4) + '"></span>' +
      '<span class="hb-bar hb-bar--base" style="--w:' + Math.min(1, r.b / r.max).toFixed(4) + '"></span>' + lp + '</div>' +
      '<div class="hb-vals tnum"><b>' + r.fmt(r.o) + '</b><span>' + r.fmt(r.b) + '</span></div></div>';
  }).join('') + '<p class="sn-note">The dashed line is the 55% limit.</p>';
  if (ours.halves['2']) {
    const h2 = ours.halves['2'].d_share;
    halves.querySelector('.sn-note').textContent = 'The dashed line is the ' + pct(lim.d_share[1], 0) +
      ' limit. The engine\u2019s closest value is half 2: ' + pct(h2, 1) + ', ' + fixed((lim.d_share[1] - h2) * 100, 1) + ' points under it.';
  }

  // -- D-segment lengths ------------------------------------------------------------------
  const dlen = fig.querySelector('.sn-dlen-body');
  function drawDlen() {
    dlen.innerHTML = '';
    const w = dlen.clientWidth;
    if (!w) return;
    const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
    const rows = [['ours', ours.total.d_segment_len, ours.total.d_segments], ['base', base.total.d_segment_len, base.total.d_segments]];
    const x = d3.scaleLog().domain([1, 400]).range([rem * 0.5, w - rem * 0.5]).clamp(true);
    const h = rem * 9;
    const svg = d3.select(dlen).append('svg').attr('width', w).attr('height', h);
    svg.append('g').attr('class', 'axis').attr('transform', 'translate(0,' + (h - rem * 1.6) + ')')
      .call(d3.axisBottom(x).tickValues([1, 3, 10, 30, 100, 300]).tickFormat((v) => v + ' s').tickSize(3));
    rows.forEach(([cls, st, n], i) => {
      if (!st || st.min_s === null) return;
      const y = rem * 1.6 + i * rem * 3;
      const g = svg.append('g').attr('class', 'dl dl--' + cls);
      g.append('line').attr('class', 'dl-range').attr('x1', x(st.min_s)).attr('x2', x(st.max_s)).attr('y1', y).attr('y2', y);
      g.append('rect').attr('class', 'dl-box').attr('x', x(st.median_s)).attr('width', Math.max(2, x(st.p90_s) - x(st.median_s)))
        .attr('y', y - rem * 0.45).attr('height', rem * 0.9).attr('rx', 2);
      g.append('line').attr('class', 'dl-med').attr('x1', x(st.median_s)).attr('x2', x(st.median_s)).attr('y1', y - rem * 0.6).attr('y2', y + rem * 0.6);
      g.append('text').attr('class', 'svg-label').attr('x', x(1)).attr('y', y - rem * 0.75)
        .text((cls === 'ours' ? 'engine' : 'baseline') + ': ' + int(n) + ' stoppages, median ' + fixed(st.median_s, 1) + ' s, 90% under ' + fixed(st.p90_s, 0) + ' s');
    });
  }
  observeSize(dlen, drawDlen);

  // -- tags and set pieces ----------------------------------------------------------------
  const tagsBody = fig.querySelector('.sn-tags-body');
  const ta = ours.tags && ours.tags.total ? ours.tags.total.agreement : null;
  const tb = base.tags && base.tags.total ? base.tags.total.agreement : null;
  const tw = wat.min_tag_agreement;
  const pos = (v) => (((v - 0.5) / 0.5) * 100).toFixed(2) + '%';
  tagsBody.innerHTML =
    '<div class="tg-meter">' +
    (tw ? '<span class="tg-watch" style="left:' + pos(tw) + '"><em>look closer below ' + pct(tw, 0) + '</em></span>' : '') +
    '<span class="tg-gt" style="left:' + pos(GT_TAG_AGREEMENT) + '"><em>ground truth ' + pct(GT_TAG_AGREEMENT, 0) + '</em></span>' +
    (tb !== null ? '<span class="tg-dot tg-dot--base" style="left:' + pos(tb) + '"></span>' : '') +
    (ta !== null ? '<span class="tg-dot tg-dot--ours" style="left:' + pos(ta) + '"></span>' : '') +
    '</div><div class="tg-scale"><span>50%</span><span>75%</span><span>100%</span></div>' +
    '<div class="tg-vals"><div><b class="tnum">' + pct(ta, 1) + '</b><span>engine</span></div><div><b class="tnum tg-b">' + pct(tb, 1) + '</b><span>baseline</span></div></div>' +
    '<p class="sn-note">Share of tagged frames where the engine\u2019s live label names the tagged team. ' +
    'The engine does not read Smart Tagging, so the tags are an outside check here. ' +
    'The ground truth itself agrees on about ' + pct(GT_TAG_AGREEMENT, 0) + ' of the public frames, so a high value is not the goal.</p>' +
    (ours.set_pieces ? '<div class="sn-mini"><b class="tnum">' + int(ours.set_pieces.n_without_d) + '</b> of ' + int(ours.set_pieces.n_checked) +
      ' tagged set pieces have no <code>D</code> within 10 s.</div>' : '') +
    (ours.gaps ? '<div class="sn-mini"><b class="tnum">' + int(ours.gaps.n_changes) + '</b> label changes inside the ' + int(ours.gaps.long_gaps) +
      ' ball gaps longer than 2 s.</div>' : '');

  // -- the learned models' own checks -----------------------------------------------------
  const sl = fig.querySelector('.sl-body');
  const meter = (v, max) => Math.min(100, (v / max) * 100).toFixed(2) + '%';
  const pr = POSSESSION_RATIO;
  const rpos = (v) => (((v - pr.domain[0]) / (pr.domain[1] - pr.domain[0])) * 100).toFixed(2) + '%';
  sl.innerHTML = '<div class="sl-grid">' + STABILITY.map((c) => {
    const pass = c.value <= c.limit;
    return '<div class="sl-item"><span class="sl-k">' + esc(c.label) + '</span>' +
      '<div class="sl-track"><span class="sl-fill" style="width:' + meter(c.value, STABILITY_MAX) + '"></span>' +
      '<span class="sl-limit" style="left:' + meter(c.limit, STABILITY_MAX) + '"></span></div>' +
      '<div class="sl-vals"><b class="tnum">' + pct(c.value, 2) + '</b><span class="tnum">of graded frames, limit ' + pct(c.limit, 2) + '</span>' +
      '<em class="' + (pass ? 'is-pass' : 'is-fail') + '">' + (pass ? 'within the limit' : 'over the limit') + '</em></div></div>';
  }).join('') +
    '<div class="sl-item"><span class="sl-k">Possessions, as a multiple of the rule engine\u2019s</span>' +
    '<div class="sl-range"><span class="sl-band" style="left:' + rpos(pr.range[0]) + ';width:calc(' + rpos(pr.range[1]) + ' - ' + rpos(pr.range[0]) + ')"></span>' +
    '<span class="sl-med" style="left:' + rpos(pr.median) + '" title="median of the 28 external matches"></span>' +
    '<span class="sl-dot" style="left:' + rpos(pr.value) + '"></span></div>' +
    '<div class="sl-vals"><b class="tnum">' + fixed(pr.value, 4) + '</b><span class="tnum">times, range ' + fixed(pr.range[0], 4) + ' to ' +
    fixed(pr.range[1], 3) + '</span>' + (pr.value >= pr.range[0] && pr.value <= pr.range[1]
      ? '<em class="is-pass">inside the range</em>'
      : '<em class="is-fail">' + fixed(Math.abs(pr.value < pr.range[0] ? pr.range[0] - pr.value : pr.value - pr.range[1]), 4) +
        (pr.value < pr.range[0] ? ' below' : ' above') + ' the range, after the kickoff team rule</em>') + '</div></div></div>' +
    '<p class="sn-note">Each limit is 1.5 times the same figure on the 20 SkillCorner matches, each labelled by models ' +
    'that had not seen it. The engine submits the average of the four networks, so their disagreement measures how ' +
    'sure the models are on this match\u2019s tracking, not an error rate. The range of the possession count runs ' +
    'from the lowest to the highest ratio on the 28 external matches, scored the same way, whose median is ' +
    fixed(pr.median, 2) + ' (the tick). The models wrote 265 possessions, 0.950 times the rule engine\u2019s, inside ' +
    'the range. A check of the output against Law 8 then showed that the models gave the first 0.84 s of the second ' +
    'half to the team that had kicked off the first half. The kickoff team rule removes that possession, which ' +
    'leaves 264.</p>';

  // -- the public window, for information -----------------------------------------------------------
  const sc = bundle.scores && bundle.scores[bundle.manifest.version];
  const smooth50 = bundle.scores && bundle.scores.baseline_smooth50;
  if (sc) {
    const t25 = sc.transitions && sc.transitions['25'];
    const b = sc.bootstrap || {};
    const stat = (v, label) => '<div class="sc-stat"><b class="tnum">' + v + '</b><span>' + esc(label) + '</span></div>';
    fig.querySelector('.sn-check').innerHTML =
      '<h4>Last, the public window, a pipeline check</h4>' +
      '<p class="sn-note">These numbers were scored once, for information, after the settings were final. With ' +
      int(sc.n_gt_spells) + ' possessions in the window, gaps below about 0.33 in spellF1 are noise. So the window only ' +
      'checks the validator, the frame alignment and the reproduction.</p>' +
      '<div class="sc-stats">' +
      stat(fixed(sc.spell_f1_mean, 4), 'spellF1, bootstrap 5 to 95%: ' + fixed(b.lo, 2) + ' to ' + fixed(b.hi, 2)) +
      stat(fixed(sc.macro_f1, 3), 'macroF1') +
      stat(int(sc.n_pred_spells) + ' / ' + int(sc.n_gt_spells), 'possessions written / in the ground truth') +
      (t25 ? stat(signed(t25.mean_latency, 1), 'mean transition latency, frames (negative is early)') : '') +
      stat(fixed(RULE_ENGINE_WINDOW, 3), 'spellF1 of the rule engine, whose fitted settings come from external matches') +
      (smooth50 ? stat(fixed(smooth50.spell_f1_mean, 3), 'spellF1 of baseline_smooth50') : '') +
      '</div>';
  }

  root.addEventListener('focusin', () => {
    if (!seen) {
      seen = true;
      renderGauges(true);
    }
  }, { once: true });
}
