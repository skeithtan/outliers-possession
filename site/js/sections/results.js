/**
 * Results on matches the models never saw.
 *
 * The section follows Section 3 of the technical document:
 *
 *   1. the engine's gain over the rule engine on each external test, from `LEARNED_EXTERNAL` in
 *      `learned_external.js` (the numbers of Section 3 of the technical document), with the PFF
 *      scores and the held-out PFF test as text
 *   2. the rule engine, the reference of every gain: spellF1 per match against
 *      `baseline_smooth50`, from `external_summary.json` (`tools/viz/build_site.py` writes it from
 *      our file of external numbers: summary numbers of the eight Metrica and DFL matches and the
 *      two sealed DFL matches, never a frame of this match)
 *   3. the rule engine in detail, in a disclosure: the summary table, the component switch-offs,
 *      where its error is (the reason why the two models target the home-or-away decisions), what the
 *      fitted contact point adds and its SkillCorner score
 *
 * @module sections/results
 */

import { fixed, signed, pct, int, esc } from '../util/format.js';
import { observeSize, onFirstView, dur, showTip, hideTip } from '../util/dom.js';
import { LEARNED_EXTERNAL } from '../learned_external.js';

/** Match groups of the rule engine's per-match chart. */
const GROUPS = [
  { title: 'Metrica Sample Games', test: (g) => g.startsWith('METRICA-') },
  { title: 'Sportec Open DFL, five matches', test: (g) => g.startsWith('DFL-') },
];

/** Group titles of the gain chart, in the order of `LEARNED_EXTERNAL`. */
const GAIN_GROUPS = {
  'unseen provider': 'Unseen provider',
  'PFF FC World Cup 2022': 'PFF FC World Cup 2022',
};

/** Domain of the gain chart, gain in spellF1 over the rule engine. */
const GAIN_DOMAIN = [-0.02, 0.22];

/*
 * A number of the engine that is not a row of the chart: Section 3 of the technical document.
 */
const PFF_AS_PROVIDED = { engine: 0.570, rule: 0.409 };

/**
 * The rule engine's switch-offs, in the order of our results table, with the reading of each
 * row. `fields` names the ablation rows of `external_summary.json`. A row with `flat` is shown as
 * text: its component cannot be tested on the external matches.
 */
const SWITCH_OFFS = [
  { fields: ['REDIRECT_SHAPE_CONFIRM'], label: 'team-movement check on a redirect', read: () => 'helps on both providers' },
  { fields: ['DRIBBLE_FRAMES'], label: 'dribble events', read: () => 'helps on both providers' },
  { fields: ['CONTACT_POINT'], label: 'fitted contact point', read: () => 'helps with an ATD-like ball, on both providers' },
  {
    fields: ['W_GAP_STOP'], label: 'ball-gap stoppage (ball missing, players stopped)',
    read: (r) => 'Metrica ' + signed(r.own.metrica, 3) + ' and ' + signed(r.atd.metrica, 3) + '. Zero on DFL, whose dead ball stays visible',
  },
  {
    fields: ['REACH_SHAPE_CONFIRM'], label: 'distant touches, credited up to 3 m away',
    read: () => 'helps a little with an ATD-like ball. The fitted contact point reaches most of these touchers',
  },
  { fields: ['PURSUIT_MIN_BALL_SPEED'], label: 'pursuit stop', read: () => 'helps on DFL, small cost on Metrica' },
  {
    fields: ['ONSET_KNEE'], label: 'stoppage start dated at the knee',
    read: (r) => 'Metrica ' + signed(r.own.metrica, 3) + ' and ' + signed(r.atd.metrica, 3) + ', DFL 0',
  },
  { fields: ['RESTART_CHAIN_FRAMES'], label: 'chained restarts', read: () => 'small' },
  {
    fields: ['CONTACT_DETECTOR'], label: 'contacts over ±12 frames, replaced by a 3-frame velocity step',
    read: (r) => 'ATD-like Metrica ' + signed(r.atd.metrica, 3) + ', DFL ' + signed(r.atd.dfl, 3) + ': helps with an ATD-like ball',
  },
  {
    fields: ['REDIRECT_FRACTION'], label: 'redirects (without them, every contact is a kick or a reception)',
    read: (r) => 'costs on DFL (' + signed(r.own.dfl, 3) + ', ' + signed(r.atd.dfl, 3) + ') and on Metrica (' +
      signed(r.own.metrica, 3) + ', ' + signed(r.atd.metrica, 3) + ')',
  },
  {
    fields: ['GK_SWAP_R', 'TAG_TEAM_PRIOR_RATIO'], label: 'goalkeeper-swap correction, Smart Tagging hints', flat: true,
    read: () => 'not testable: the external matches have no swapped tracks and no tags',
  },
];

/*
 * Organisers' published full-match spellF1 of baseline_smooth50 on this match, the same constant
 * as in sections/problem.js (starter/baseline.py). It is not in the bundle.
 */
const BASELINE_FULL_MATCH = 0.2433;

/** Domain of the switch-off bars, change in mean spellF1. */
const SWITCH_DOMAIN = [-0.05, 0.05];

/*
 * What the fitted contact point adds to the rule engine: the paired change in mean spellF1 with
 * and without it on the eight Metrica and DFL matches, at three smoothings of the ball. These
 * numbers are not in external.json. They come from our record of the contact point test.
 */
const CONTACT_POINT_GAIN = [
  { ball: 'own tracking', both: [0.001, -0.001, 0.003], metrica: [0.001, -0.002, 0.004], dfl: [0.002, -0.002, 0.005], above: '4 of 8' },
  { ball: 'ATD-like', both: [0.019, 0.014, 0.024], metrica: [0.025, 0.017, 0.034], dfl: [0.013, 0.006, 0.019], above: '8 of 8' },
  { ball: 'smoothed with the full kernel measured on this match (σ = 9.5 frames)', both: [0.049, 0.041, 0.058], metrica: [0.043, 0.030, 0.056], dfl: [0.056, 0.045, 0.065], above: '8 of 8' },
];

/** Short row label of an external match: "Sample Game 1" or the DFL match id. */
function matchLabel(game) {
  if (game.startsWith('METRICA-S')) return 'Sample Game ' + game.slice(-1);
  return game.replace(/^DFL-/, '');
}

/** "+0.261 [+0.239, +0.282]" with the interval in a muted span. */
function withCi(mean, low, high) {
  return '<span class="tnum">' + signed(mean, 3) + '</span> <span class="ci tnum">[' + signed(low, 3) + ', ' + signed(high, 3) + ']</span>';
}

/**
 * @param {HTMLElement} root  The `#results` section.
 * @param {{bundle: Object, d3: Object}} ctx
 */
export function mount(root, ctx) {
  const { bundle, d3 } = ctx;
  const ext = bundle.extra && bundle.extra['external_summary.json'];
  const fig = root.querySelector('#results-fig');
  if (!fig) return;

  const pffBench = LEARNED_EXTERNAL.find((t) => t.id === 'pff_benchmark_2s');
  const sealedPff = LEARNED_EXTERNAL.find((t) => t.id === 'pff_sealed_2s');
  const scFine = LEARNED_EXTERNAL.find((t) => t.id === 'skillcorner_left_out_fine');

  fig.innerHTML = `
    <div class="le-legend">
      <span><i class="le-k le-k--dot"></i>mean gain, the two models</span>
      <span><i class="le-k le-k--ring"></i>mean gain, the earlier pair of models</span>
      <span><i class="le-k le-k--bound"></i>to the lower 5% bound of the mean</span>
    </div>
    <div class="le-chart" role="img" aria-label="${esc(gainAria())}"></div>
    <p class="rs-note">Each dot is the mean gain of the engine over the rule engine on a test's matches, in
      <code>spellF1</code>. The bar runs back to the lower 5% bound of the mean (paired bootstrap) where we
      computed it. On SkillCorner's possession field the gain, ${signed(scFine.gain, 3)} on ${scFine.above} of
      ${scFine.n} matches, is level with the rule engine.</p>
    <div class="le-facts">
      <div class="le-fact"><span class="le-fact-k">PFF design matches, 2-second labels</span>
        <b class="tnum">${fixed(pffBench.engine, 3)}</b><em class="tnum">spellF1, against ${fixed(pffBench.rule, 3)} for the rule engine</em></div>
      <div class="le-fact"><span class="le-fact-k">PFF design matches, labels as provided</span>
        <b class="tnum">${fixed(PFF_AS_PROVIDED.engine, 3)}</b><em class="tnum">spellF1, against ${fixed(PFF_AS_PROVIDED.rule, 3)} for the rule engine</em></div>
    </div>
    <p class="rs-note">Every other external test that we set passes as well.</p>
    <p class="rs-note"><strong>Held-out PFF matches.</strong> The ${sealedPff.n} held-out matches ran once, with a
      pass condition set before the run, on the earlier pair of models. That pair had the same design and settings,
      but was trained with an earlier attacking-direction rule, which turned the pitch by comparing the home team
      with the halfway line. It gained ${signed(sealedPff.gain, 3)} (lower 5% bound ${signed(sealedPff.lo5, 3)}), above the rule engine on
      ${sealedPff.above} of ${sealedPff.n}. The two models of the engine were not run there.</p>

    <h3 class="rs-h">The rule engine, the reference</h3>
    <div class="rs-ref"></div>
    <details class="rs-more">
      <summary><span class="rs-more-t">The rule engine in detail</span>
        <span class="rs-more-s">its summary table, what each component is worth, where its error is and what the
          fitted contact point adds</span></summary>
      <div class="rs-more-body"></div>
    </details>`;

  drawGainChart(fig.querySelector('.le-chart'), d3);

  const ref = fig.querySelector('.rs-ref');
  if (!ext) {
    ref.innerHTML = '<p class="muted">This bundle has no external numbers of the rule engine. Rebuild it with <code>python -m tools.viz.build_site</code>.</p>';
    fig.querySelector('.rs-more').remove();
    return;
  }
  mountRuleEngine(ref, fig.querySelector('.rs-more-body'), ext, d3);
}

/** One sentence for screen readers: every test of the gain chart. */
function gainAria() {
  return 'Gain in spellF1 of the engine over the rule engine on each external test. ' + LEARNED_EXTERNAL.map((t) =>
    t.label + ', ' + t.note + ': ' + signed(t.gain, 3) + (t.lo5 !== null ? ', lower 5% bound ' + signed(t.lo5, 3) : '') +
    ', above the rule engine on ' + t.above + ' of ' + t.n).join('. ') + '.';
}

/**
 * The gain chart: one row per external test, a dot at the mean gain over the rule engine, a bar
 * back to the lower 5% bound where it was computed, and the matches above the rule engine.
 * @param {HTMLElement} host
 * @param {Object} d3
 */
function drawGainChart(host, d3) {
  const rows = [];
  let lastGroup = null;
  LEARNED_EXTERNAL.forEach((t) => {
    if (t.group !== lastGroup) {
      rows.push({ head: GAIN_GROUPS[t.group] || t.group });
      lastGroup = t.group;
    }
    rows.push({ t });
  });
  let grown = false;
  let grow = null;

  function draw(w) {
    host.innerHTML = '';
    if (!w) return;
    const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
    const narrow = w < 40 * rem;
    const labelW = narrow ? 0 : 15 * rem;
    const aboveW = narrow ? 0 : 10.5 * rem;
    const meterW = 4.5 * rem;
    const rowH = narrow ? 4.6 * rem : 3 * rem;
    const headH = narrow ? 2.1 * rem : 2.3 * rem;
    const top = 2.2 * rem;
    const bottom = 2.4 * rem;
    let h = top + bottom;
    rows.forEach((r) => { h += r.head ? headH : rowH; });
    const x0 = labelW + (narrow ? 0.4 * rem : 0.8 * rem);
    const x1 = w - aboveW - (narrow ? 3.6 * rem : 1.2 * rem);
    const x = d3.scaleLinear().domain(GAIN_DOMAIN).range([x0, x1]);
    const svg = d3.select(host).append('svg').attr('width', w).attr('height', h);

    // Grid and axis on top.
    const grid = svg.append('g').attr('class', 'le-grid');
    const ticks = narrow ? [0, 0.1, 0.2] : [0, 0.05, 0.1, 0.15, 0.2];
    ticks.forEach((v) => {
      if (v !== 0) grid.append('line').attr('x1', x(v)).attr('x2', x(v)).attr('y1', top - 0.35 * rem).attr('y2', h - bottom + 0.3 * rem);
      grid.append('text').attr('class', 'svg-label tnum').attr('x', x(v)).attr('y', top - 0.75 * rem)
        .attr('text-anchor', 'middle').text(v === 0 ? '0' : signed(v, 2));
    });
    if (!narrow) {
      svg.append('text').attr('class', 'svg-label le-axis-t').attr('x', 0).attr('y', top - 0.75 * rem).text('test');
      svg.append('text').attr('class', 'svg-label le-axis-t').attr('x', w).attr('y', top - 0.75 * rem)
        .attr('text-anchor', 'end').text('matches above the rule engine');
    }
    // The rule engine at zero.
    svg.append('line').attr('class', 'le-zero').attr('x1', x(0)).attr('x2', x(0))
      .attr('y1', top - 0.35 * rem).attr('y2', h - bottom + 0.3 * rem);
    svg.append('text').attr('class', 'svg-label le-zero-t').attr('x', narrow ? x(0) - 0.2 * rem : x(0))
      .attr('y', h - bottom + 1.3 * rem).attr('text-anchor', narrow ? 'start' : 'middle').text('rule engine');
    svg.append('text').attr('class', 'svg-label le-axis-t').attr('x', x1).attr('y', h - bottom + 1.3 * rem)
      .attr('text-anchor', 'end').text('gain in spellF1 →');

    const stems = [];
    const dots = [];
    const halos = [];
    const vals = [];
    const bounds = [];
    let y = top;
    rows.forEach((r) => {
      if (r.head) {
        svg.append('text').attr('class', 'rs-group le-group').attr('x', 0).attr('y', y + headH - 0.6 * rem).text(r.head);
        svg.append('line').attr('class', 'le-sep').attr('x1', 0).attr('x2', w)
          .attr('y1', y + headH - 0.2 * rem).attr('y2', y + headH - 0.2 * rem);
        y += headH;
        return;
      }
      const t = r.t;
      const g = svg.append('g').attr('class', 'le-row' + (t.earlier ? ' is-earlier' : ''));
      g.append('rect').attr('class', 'le-hit').attr('x', 0).attr('y', y).attr('width', w).attr('height', rowH);
      const yc = narrow ? y + rowH - 1.25 * rem : y + rowH / 2;
      // Labels.
      if (narrow) {
        g.append('text').attr('class', 'rs-label le-label').attr('x', 0).attr('y', y + 1.05 * rem).text(t.label);
        g.append('text').attr('class', 'svg-label le-note').attr('x', 0).attr('y', y + 2.05 * rem).text(t.note);
        g.append('text').attr('class', 'le-above-t tnum').attr('x', w).attr('y', y + 1.05 * rem).attr('text-anchor', 'end')
          .text(t.above + ' of ' + t.n + ' above');
      } else {
        g.append('text').attr('class', 'rs-label le-label').attr('x', 0.8 * rem).attr('y', yc - 0.15 * rem).text(t.label);
        g.append('text').attr('class', 'svg-label le-note').attr('x', 0.8 * rem).attr('y', yc + 0.95 * rem).text(t.note);
        // Matches above the rule engine: a small meter and the count.
        const mx = w - aboveW + 0.6 * rem;
        g.append('rect').attr('class', 'le-meter').attr('x', mx).attr('y', yc - 0.2 * rem).attr('width', meterW)
          .attr('height', 0.4 * rem).attr('rx', 0.2 * rem);
        g.append('rect').attr('class', 'le-meter-fill').attr('x', mx).attr('y', yc - 0.2 * rem)
          .attr('width', meterW * (t.above / t.n)).attr('height', 0.4 * rem).attr('rx', 0.2 * rem);
        g.append('text').attr('class', 'le-above-t tnum').attr('x', w).attr('y', yc + 0.32 * rem).attr('text-anchor', 'end')
          .text(t.above + ' of ' + t.n);
      }
      // Stem from the rule engine to the mean, the bound, the dot and the value.
      stems.push({ t, el: g.append('line').attr('class', 'le-stem').attr('x1', x(0)).attr('x2', x(0)).attr('y1', yc).attr('y2', yc) });
      if (t.lo5 !== null && t.lo5 !== undefined) {
        const b = g.append('g').attr('class', 'le-bound');
        b.append('rect').attr('x', x(t.lo5)).attr('y', yc - 0.3 * rem).attr('width', Math.max(1, x(t.gain) - x(t.lo5)))
          .attr('height', 0.6 * rem).attr('rx', 0.3 * rem);
        b.append('line').attr('x1', x(t.lo5)).attr('x2', x(t.lo5)).attr('y1', yc - 0.55 * rem).attr('y2', yc + 0.55 * rem);
        b.append('text').attr('class', 'svg-label tnum le-bound-t').attr('x', x(t.lo5)).attr('y', yc + 1.35 * rem)
          .attr('text-anchor', 'middle').text('5% bound ' + signed(t.lo5, 3));
        bounds.push(b);
      }
      halos.push({ t, el: g.append('circle').attr('class', 'le-halo').attr('cx', x(0)).attr('cy', yc).attr('r', 0.85 * rem) });
      dots.push({ t, el: g.append('circle').attr('class', 'le-dot').attr('cx', x(0)).attr('cy', yc).attr('r', 0.42 * rem) });
      vals.push({
        t, el: g.append('text').attr('class', 'le-val tnum').attr('x', x(t.gain) + 0.85 * rem).attr('y', yc + 0.34 * rem)
          .text(signed(t.gain, 3)),
      });
      g.on('pointermove', (e) => showTip(gainTip(t), e.clientX, e.clientY)).on('pointerleave', hideTip);
      y += rowH;
    });

    grow = (animate) => {
      if (!animate || !dur(1)) {
        // Final state at once, without a timer, so that a redraw after a resize never waits on one.
        stems.forEach((st) => st.el.attr('x2', x(st.t.gain)));
        dots.forEach((st) => st.el.attr('cx', x(st.t.gain)));
        halos.forEach((st) => st.el.attr('cx', x(st.t.gain)));
        vals.forEach((st) => st.el.style('opacity', 1));
        bounds.forEach((bd) => bd.style('opacity', 1));
        return;
      }
      const ms = dur(1100);
      const delay = (i) => dur(120 + i * 110);
      const ease = d3.easeCubicOut;
      stems.forEach((st, i) => st.el.transition().duration(ms).delay(delay(i)).ease(ease).attr('x2', x(st.t.gain)));
      dots.forEach((st, i) => st.el.transition().duration(ms).delay(delay(i)).ease(ease).attr('cx', x(st.t.gain)));
      halos.forEach((st, i) => st.el.transition().duration(ms).delay(delay(i)).ease(ease).attr('cx', x(st.t.gain)));
      vals.forEach((st, i) => st.el.transition().duration(dur(400)).delay(delay(i) + ms * 0.7).style('opacity', 1));
      bounds.forEach((bd) => bd.transition().duration(dur(500)).delay(delay(rows.length) + ms * 0.5).style('opacity', 1));
    };
    if (grown) grow(false);
    else {
      vals.forEach((s) => s.el.style('opacity', 0));
      bounds.forEach((b) => b.style('opacity', 0));
    }
  }

  observeSize(host, draw);
  onFirstView(host, () => {
    grown = true;
    if (grow) grow(true);
  }, 0.3);
}

/** Tooltip of one test of the gain chart. */
function gainTip(t) {
  const row = (k, v) => '<div class="tt-row"><span>' + esc(k) + '</span><b>' + esc(v) + '</b></div>';
  return '<div class="tt-title">' + esc(t.label) + '</div>' +
    '<div class="tt-row"><span>' + esc(t.note) + '</span></div>' +
    row('mean gain over the rule engine', signed(t.gain, 3)) +
    (t.lo5 !== null && t.lo5 !== undefined ? row('lower 5% bound of the mean', signed(t.lo5, 3)) : '') +
    row('matches above the rule engine', t.above + ' of ' + t.n) +
    (t.engine !== undefined ? row('spellF1, engine and rule engine', fixed(t.engine, 3) + ' and ' + fixed(t.rule, 3)) : '') +
    (t.earlier ? '<div class="tt-note">The earlier pair of models: the same design and settings, an earlier attacking-direction rule.</div>' : '');
}

/**
 * The rule engine: the per-match chart against baseline_smooth50, then the details.
 * @param {HTMLElement} ref   Host of the reference paragraph and chart.
 * @param {HTMLElement} more  Host of the details.
 * @param {Object} ext        `external_summary.json`.
 * @param {Object} d3
 */
function mountRuleEngine(ref, more, ext, d3) {
  const s = ext.summary;
  ref.innerHTML = `
    <p class="rs-note">The rule engine is our earlier engine of physical rules, which the two models replace and
      which one configuration setting restores (Section 07). It finds the stoppages from the movement of the ball and the players, and
      changes the team only when a player of the other team kicks, receives, controls or dribbles the ball. On the
      providers' own tracking it scores a mean <code>spellF1</code> of ${fixed(s.metrica.own.engine_spell_f1, 3)} on
      Metrica's three matches and ${fixed(s.dfl.own.engine_spell_f1, 3)} on the five DFL training matches, against
      ${fixed(s.metrica.own.baseline_spell_f1, 3)} and ${fixed(s.dfl.own.baseline_spell_f1, 3)} for
      <code>baseline_smooth50</code>. Its settings were selected on these eight matches. We score each of them on
      the provider's <strong>own tracking</strong> and on an <strong>ATD-like</strong> copy, whose ball we smooth until
      its kicks look like this match's kicks (σ = 3 frames on DFL, 5 on Metrica).</p>
    <div class="rs-legend">
      <span><i class="rs-k rs-k--own"></i>rule engine, own tracking</span>
      <span><i class="rs-k rs-k--atd"></i>rule engine, ATD-like ball</span>
      <span><i class="rs-k rs-k--base"></i>baseline_smooth50</span>
    </div>
    <div class="rs-chart" role="img" aria-label="spellF1 on each external match, the rule engine against baseline_smooth50"></div>
    <p class="rs-note">J03WMX and the other codes are DFL match ids. <code>baseline_smooth50</code> scores lower on this match than on any external match, so by its
      measure this match is at least as hard as the DFL matches. DFL scores lower mainly because its labels give the
      ball to a team for a single touch.</p>`;

  more.innerHTML = `
    <div class="rs-table-wrap"></div>
    <p class="rs-note">The two sealed DFL matches ran once, on an earlier rule engine without the fitted contact point
      and the back-dating. They passed a test written before the run, which checked the timing of stoppages only.
      On the 20 SkillCorner matches, on which no setting of the rule engine was selected, the rule engine scores a mean
      <code>spellF1</code> of 0.508 (0.439 to 0.561 by match) against the provider's phases of play. The back-dating
      adds +0.012 to the rule engine there (paired bootstrap 5 to 95%: +0.010 to +0.013) and gains on all 20 matches.</p>

    <h4 class="rs-h4">What each component of the rule engine is worth</h4>
    <p class="rs-note">We switched off one component of the rule engine at a time on the eight Metrica and DFL matches.
      Each bar is the change in the rule engine's mean <code>spellF1</code>, with its paired bootstrap 5 to 95%
      interval. A negative change means the component helps.</p>
    <div class="rs-switch"></div>
    <p class="rs-note">Switched off alone, the back-dating changes the rule engine's mean <code>spellF1</code> by
      −0.005 <span class="ci">[−0.008, −0.003]</span> on Metrica's own tracking and −0.008
      <span class="ci">[−0.011, −0.005]</span> ATD-like, and by −0.002 <span class="ci">[−0.005, +0.000]</span> and
      −0.004 <span class="ci">[−0.007, −0.002]</span> on DFL. It helps on Metrica and on DFL ATD-like. On DFL's own
      tracking the interval reaches zero.</p>
    <p class="rs-note">The rule engine's ±12-frame contacts cost on DFL's unsmoothed ball and gain with an ATD-like
      ball, the case of this match. The redirect, a third kind of contact beside kicks and receptions, costs the rule
      engine on both providers. The rule engine keeps it, because dropping it on this result would be a choice outside
      its selection rule.</p>

    <h4 class="rs-h4">Where the rule engine's error is</h4>
    <p class="rs-note">We replaced one part of the rule engine's output at a time with the providers' labels, on the
      providers' own tracking.</p>
    <div class="rs-oracle"></div>
    <p class="rs-note rs-oracle-note"></p>
    <div class="rs-recall"></div>

    <h4 class="rs-h4">What the fitted contact point adds to the rule engine</h4>
    <p class="rs-note">We compared the rule engine with and without its fitted contact point on the eight Metrica and DFL
      matches, with the ball at three smoothings. Each cell is the paired change in mean <code>spellF1</code>, with
      each provider weighted equally.</p>
    <div class="rs-kink"></div>
    <p class="rs-note">The gain grows with the smoothing, from nothing on an unsmoothed ball to +0.049 at the measured
      kernel: the pattern to expect from a fix for the corner-cutting of the smoothed ball. The rule moves only the
      owner, so the share of stoppage starts and restarts that the rule engine finds does not change.</p>`;

  // -- 1. per-match chart ------------------------------------------------------------------
  const chart = ref.querySelector('.rs-chart');
  const rows = [];
  GROUPS.forEach((g) => {
    rows.push({ head: g.title });
    ext.matches.filter((m) => g.test(m.game)).forEach((m) => rows.push({
      label: matchLabel(m.game), own: m.own.engine_spell_f1, atd: m.atd.engine_spell_f1,
      base: m.own.baseline_spell_f1, delta: m.own.engine_minus_baseline,
    }));
  });
  rows.push({ head: 'Sportec Open DFL, sealed, scored once', headWide: ' on an earlier rule engine, without the fitted contact point and the back-dating' });
  ext.sealed.forEach((m) => rows.push({
    label: matchLabel(m.game), own: m.own.engine_spell_f1, atd: m.atd.engine_spell_f1, base: null, delta: null,
  }));

  function drawChart(w) {
    chart.innerHTML = '';
    const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
    const narrow = w < 40 * rem;
    const labelW = narrow ? 6.2 * rem : 9.5 * rem;
    const deltaW = narrow ? 0 : 11 * rem;
    const rowH = 1.55 * rem;
    const headH = 1.9 * rem;
    const top = 1.8 * rem;
    let h = top;
    rows.forEach((r) => { h += r.head ? headH : rowH; });
    h += 2.4 * rem;
    const x = d3.scaleLinear().domain([0, 0.8]).range([labelW, w - deltaW - rem * 0.6]);
    const svg = d3.select(chart).append('svg').attr('width', w).attr('height', h);
    const grid = svg.append('g').attr('class', 'rs-grid');
    const ticks = narrow ? [0, 0.2, 0.4, 0.6, 0.8] : d3.range(0, 0.81, 0.1);
    ticks.forEach((t) => {
      grid.append('line').attr('x1', x(t)).attr('x2', x(t)).attr('y1', top - 0.4 * rem).attr('y2', h - 2 * rem);
      grid.append('text').attr('class', 'svg-label tnum').attr('x', x(t)).attr('y', top - 0.7 * rem)
        .attr('text-anchor', 'middle').text(t.toFixed(1));
    });
    svg.append('text').attr('class', 'svg-label').attr('x', 0).attr('y', top - 0.7 * rem).text('spellF1');
    if (deltaW) {
      svg.append('text').attr('class', 'svg-label rs-dhead').attr('x', w).attr('y', top - 0.7 * rem)
        .attr('text-anchor', 'end').text('minus baseline_smooth50');
    }
    // baseline_smooth50 on the full match of this match, for scale.
    const fx = x(BASELINE_FULL_MATCH);
    svg.append('line').attr('class', 'rs-floor').attr('x1', fx).attr('x2', fx).attr('y1', top - 0.4 * rem).attr('y2', h - 2 * rem);
    svg.append('text').attr('class', 'svg-label rs-floor-t').attr('y', h - 0.7 * rem)
      .attr('text-anchor', narrow ? 'start' : 'end')
      .attr('x', narrow ? 0 : fx - 0.3 * rem)
      .text('baseline_smooth50 on this match: ' + fixed(BASELINE_FULL_MATCH, 3));
    let y = top;
    rows.forEach((r) => {
      if (r.head) {
        svg.append('text').attr('class', 'rs-group').attr('x', 0).attr('y', y + headH - 0.55 * rem)
          .text(r.head + (narrow ? '' : r.headWide || ''));
        y += headH;
        return;
      }
      const yc = y + rowH / 2;
      svg.append('text').attr('class', 'rs-label').attr('x', narrow ? 0 : 0.8 * rem).attr('y', yc + 0.3 * rem).text(r.label);
      if (r.base !== null) {
        svg.append('line').attr('class', 'rs-link').attr('x1', x(r.base)).attr('x2', x(r.own)).attr('y1', yc).attr('y2', yc);
        const d = 0.32 * rem;
        svg.append('path').attr('class', 'rs-base')
          .attr('d', 'M' + x(r.base) + ',' + (yc - d) + 'L' + (x(r.base) + d) + ',' + yc + 'L' + x(r.base) + ',' + (yc + d) + 'L' + (x(r.base) - d) + ',' + yc + 'Z');
      }
      svg.append('circle').attr('class', 'rs-atd').attr('cx', x(r.atd)).attr('cy', yc).attr('r', 0.27 * rem);
      svg.append('circle').attr('class', 'rs-own').attr('cx', x(r.own)).attr('cy', yc).attr('r', 0.32 * rem);
      if (deltaW) {
        const t = svg.append('text').attr('class', 'rs-delta tnum').attr('x', w).attr('y', yc + 0.3 * rem).attr('text-anchor', 'end');
        if (r.delta) {
          t.text(signed(r.delta.mean, 3) + ' [' + signed(r.delta.low, 3) + ', ' + signed(r.delta.high, 3) + ']');
        } else {
          t.attr('class', 'rs-delta rs-delta--none').text('no baseline run');
        }
      }
      y += rowH;
    });
  }
  observeSize(chart, drawChart);

  // -- 2. summary table ---------------------------------------------------------------------
  const sumRow = (name, g) => '<tr><td>' + name + '</td>' +
    '<td class="tnum">' + fixed(g.own.engine_spell_f1, 3) + ' / ' + fixed(g.atd.engine_spell_f1, 3) + '</td>' +
    '<td class="tnum">' + fixed(g.own.engine_macro_f1, 3) + '</td>' +
    '<td class="tnum">' + fixed(g.own.baseline_spell_f1, 3) + '</td>' +
    '<td>' + withCi(g.own.engine_minus_baseline.mean, g.own.engine_minus_baseline.low, g.own.engine_minus_baseline.high) + '</td></tr>';
  const sealed = ext.sealed;
  more.querySelector('.rs-table-wrap').innerHTML =
    '<table class="rt rt--num"><thead><tr><th>matches</th><th>rule engine spellF1, own / ATD-like</th>' +
    '<th>rule engine macroF1, own</th><th>baseline_smooth50 spellF1</th><th>rule engine minus baseline_smooth50, paired bootstrap 5 to 95%</th></tr></thead><tbody>' +
    sumRow('Metrica Sample Games 1 to 3', s.metrica) +
    sumRow('Sportec Open DFL, 5 matches', s.dfl) +
    sumRow('All 8 Metrica and DFL matches', s.all) +
    '<tr class="rt-sealed"><td>Sportec Open DFL, 2 sealed matches (an earlier rule engine, without the fitted contact point and the back-dating)</td>' +
    '<td class="tnum">' + sealed.map((m) => fixed(m.own.engine_spell_f1, 3)).join(' and ') + ' / ' +
    sealed.map((m) => fixed(m.atd.engine_spell_f1, 3)).join(' and ') + '</td>' +
    '<td class="tnum">' + sealed.map((m) => fixed(m.own.engine_macro_f1, 3)).join(' and ') + '</td>' +
    '<td>not run</td><td>not run</td></tr></tbody></table>';

  // -- 3. switch-offs -------------------------------------------------------------------------
  const byField = new Map(ext.ablations.map((a) => [a.field, a]));
  const pos = (v) => (((Math.max(SWITCH_DOMAIN[0], Math.min(SWITCH_DOMAIN[1], v)) - SWITCH_DOMAIN[0]) /
    (SWITCH_DOMAIN[1] - SWITCH_DOMAIN[0])) * 100).toFixed(2) + '%';
  const bar = (view, d) => {
    const lo = Math.min(0, d.mean);
    const hi = Math.max(0, d.mean);
    return '<div class="sw-bar sw-bar--' + view + (d.mean < 0 ? ' is-help' : d.mean > 0 ? ' is-cost' : '') + '">' +
      '<span class="sw-track"><span class="sw-fill" style="left:' + pos(lo) + ';width:calc(' + pos(hi) + ' - ' + pos(lo) + ')"></span>' +
      '<span class="sw-ci" style="left:' + pos(d.low) + ';width:calc(' + pos(d.high) + ' - ' + pos(d.low) + ')"></span>' +
      '<span class="sw-zero" style="left:' + pos(0) + '"></span></span>' +
      '<b class="sw-val tnum">' + signed(d.mean, 3) + '</b></div>';
  };
  const sw = more.querySelector('.rs-switch');
  let swHtml = '<div class="sw-head"><span></span><span class="sw-scale"><span class="sw-scale-in"><em>' +
    signed(SWITCH_DOMAIN[0], 2) + '</em><em>helps ← 0 → costs</em><em>' + signed(SWITCH_DOMAIN[1], 2) +
    '</em></span><span></span></span></div>';
  SWITCH_OFFS.forEach((spec) => {
    const r = byField.get(spec.fields[0]);
    if (!r) return;
    const flat = !!spec.flat;
    swHtml += '<div class="sw-row' + (flat ? ' sw-row--flat' : '') + '">' +
      '<div class="sw-text"><b>' + esc(spec.label) + '</b><span>' + esc(spec.read(r)) + '</span></div>' +
      (flat ? '<div class="sw-bars"><span class="sw-flat tnum">0 on own and ATD-like tracking</span></div>'
        : '<div class="sw-bars">' + bar('own', r.own) + bar('atd', r.atd) + '</div>') + '</div>';
  });
  swHtml += '<div class="sw-key"><span><i class="sw-k sw-k--own"></i>own tracking</span>' +
    '<span><i class="sw-k sw-k--atd"></i>ATD-like ball</span><span><i class="sw-k sw-k--ci"></i>paired 5 to 95%</span></div>';
  sw.innerHTML = swHtml;

  // -- 4. oracle split and recall by duration ---------------------------------------------
  const or = ext.oracle.own;
  const orRows = [
    ['the rule engine', 'engine'],
    ['its dead ball, with the providers’ teams', 'reference_teams'],
    ['the providers’ dead ball, with its teams', 'reference_dead_ball'],
  ];
  const orCol = (prov, title) => {
    const base = or[prov].engine;
    return '<div class="or-col"><h4>' + title + '</h4>' + orRows.map(([label, key]) => {
      const v = or[prov][key];
      const gain = key === 'engine' ? '' : ' <em class="tnum">(' + signed(v - base, 3) + ')</em>';
      return '<div class="or-row or-row--' + key + '"><span class="or-label">' + esc(label) + '</span>' +
        '<span class="or-track"><span class="or-fill" style="width:' + (v * 100).toFixed(2) + '%"></span>' +
        '<span class="or-base" style="left:' + (base * 100).toFixed(2) + '%"></span></span>' +
        '<b class="tnum">' + fixed(v, 3) + gain + '</b></div>';
    }).join('') + '</div>';
  };
  more.querySelector('.rs-oracle').innerHTML = orCol('metrica', 'Metrica') + orCol('dfl', 'DFL');
  const rec = s.all.own.recall_by_duration;
  const nRef = rec.reduce((n, b) => n + b.n, 0);
  more.querySelector('.rs-oracle-note').innerHTML = 'For the rule engine, home-or-away decisions, not stoppage timing, carry ' +
    'most of the error. That is why the two models target those decisions. Short possessions are the hard part. ' +
    'Of the <span class="tnum">' + int(nRef) + '</span> possessions in the labels of the eight Metrica and DFL matches, ' +
    'the rule engine finds this share at temporal IoU 0.5, on the providers’ own tracking:';
  const bucketName = (b) => b.replace('< ', 'under ').replace('> ', 'over ').replace('-', ' to ');
  more.querySelector('.rs-recall').innerHTML = rec.map((b) =>
    '<div class="rc-row"><span class="rc-label">' + esc(bucketName(b.bucket)) + '</span>' +
    '<span class="rc-track"><span class="rc-fill" style="width:' + (b.recall * 100).toFixed(2) + '%"></span></span>' +
    '<b class="tnum">' + pct(b.recall, 1) + '</b><em class="tnum">' + int(b.recalled) + ' of ' + int(b.n) + '</em></div>').join('');

  // -- 5. fitted contact point ---------------------------------------------------------------
  const cell = (t) => '<td>' + withCi(t[0], t[1], t[2]) + '</td>';
  more.querySelector('.rs-kink').innerHTML =
    '<table class="rt rt--num"><thead><tr><th>ball</th><th>both providers</th><th>Metrica</th><th>DFL</th>' +
    '<th>matches above zero</th></tr></thead><tbody>' +
    CONTACT_POINT_GAIN.map((r) => '<tr><td>' + esc(r.ball) + '</td>' + cell(r.both) + cell(r.metrica) + cell(r.dfl) +
      '<td class="tnum">' + esc(r.above) + '</td></tr>').join('') + '</tbody></table>';
}
