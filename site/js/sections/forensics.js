/**
 * The public window, transition by transition: every ground-truth transition, with the engine's latency.
 *
 * The records come from `transitions.json` (`tools.forensics.analyse()`): one record for each GT
 * transition and one for each spurious transition of the engine. Each lollipop is one GT
 * transition. Its height is the engine's latency in frames, negative means early. The detail card
 * also lists the rule engine's events near the transition, which the submitted engine does not use. A missed transition is hollow with a
 * cross, at our nearest transition into the same label. A spurious transition is a violet
 * diamond. The counters come from `scores.json`, at the loosest tolerance of the report.
 *
 * Local mode adds the kinematic trace from `transitions_kin.json` and a button that sends the
 * replay to the transition.
 *
 * @module sections/forensics
 */

import { int, signed, fixed, esc, videoTime, matchClock } from '../util/format.js';
import { tolRow } from '../data.js';
import { edgeLabelRows } from '../util/chart-labels.js';
import { observeSize, dur, showTip, hideTip, countUp, el, scrollToEl, onFirstView } from '../util/dom.js';

const KIND_CLASS = {
  'H<->A': 'k-handover',
  'alive->D': 'k-onset',
  'D->alive (kick)': 'k-kick',
  'D->alive (reappear)': 'k-reappear',
};
const KIND_TEXT = {
  'H<->A': 'Hand-over, H to A or A to H',
  'alive->D': 'Ball goes dead',
  'D->alive (kick)': 'Restart by a kick',
  'D->alive (reappear)': 'Restart, the ball reappears',
};
const LIM = 60;
/**
 * The rule engine's event kinds in the words of the page. A "control" is a reception or a slow-ball
 * control. The submitted engine does not use these events: the panel shows them as context.
 */
const EVENT_TEXT = { release: 'kick', control: 'reception or slow-ball control', dribble: 'dribble', touch: 'light touch' };
/** How the rule engine's kickoff rule finds the kickoff team, in plain words. The bundle stores the rule name. */
const KICKOFF_RULE_TEXT = {
  mark: 'the player at the centre mark',
  restart: 'the first confirmed restart',
  law: 'Law 8, from the other half',
  none: 'no rule',
};

/**
 * @param {HTMLElement} root
 * @param {{bundle: Object, local: (Object|null), state: Object, d3: Object, mode: string}} ctx
 */
export function mount(root, ctx) {
  const { bundle, local, state, d3 } = ctx;
  const m = bundle.manifest;
  const win = m.public_window;
  const fig = root.querySelector('#forensics-fig');
  const tr = bundle.transitions || {};
  const recs = tr.gt_transitions || [];
  const spur = tr.spurious || [];
  const tol = tr.tol || 25;
  const score = bundle.scores[m.version];
  const t25 = score ? tolRow(score.transitions, tol) : null;
  const kin = local && local.transitionsKin ? local.transitionsKin : null;

  fig.innerHTML = `
    <div class="fx-top">
      <div class="fx-stat"><span>matched</span><b class="tnum fx-tp">0</b><em>of <span class="fx-ngt">0</span>, within ${tol} frames</em></div>
      <div class="fx-stat fx-stat--miss"><span>missed</span><b class="tnum fx-fn">0</b><em>hollow, with a cross</em></div>
      <div class="fx-stat fx-stat--spur"><span>spurious</span><b class="tnum fx-fp">0</b><em>violet diamonds</em></div>
      <div class="fx-stat fx-stat--lat"><span>mean latency</span><b class="tnum fx-lat">0</b><em>frames, matched only</em></div>
    </div>
    <div class="fx-legend">
      <span><i class="kd k-handover"></i>hand-over</span>
      <span><i class="kd k-onset"></i>ball goes dead</span>
      <span><i class="kd k-kick"></i>restart, kick</span>
      <span><i class="kd k-reappear"></i>restart, ball reappears</span>
      <span><i class="kd kd--miss"></i>missed</span>
      <span><i class="kd kd--spur"></i>spurious</span>
    </div>
    <div class="fx-chart"></div>
    <div class="fx-detail" aria-live="polite"></div>
    <h3 class="fx-h">Latency by kind</h3>
    <div class="fx-kinds"></div>
    <div class="fx-foot"></div>`;

  const chart = fig.querySelector('.fx-chart');
  const detail = fig.querySelector('.fx-detail');
  let selected = recs.findIndex((r) => r.status === 'missed');
  if (selected < 0) selected = 0;
  let api = null;

  // -- counters ---------------------------------------------------------------------------
  const nMatched = t25 ? t25.n_matched : recs.filter((r) => r.status === 'matched').length;
  const nMissed = t25 ? t25.n_missed : recs.filter((r) => r.status === 'missed').length;
  const nSpur = t25 ? t25.n_spurious : spur.length;
  const meanLat = t25 ? t25.mean_latency : null;
  fig.querySelector('.fx-ngt').textContent = String(t25 ? t25.n_gt : recs.length);
  onFirstView(fig, () => {
    countUp(fig.querySelector('.fx-tp'), nMatched, { format: (v) => String(Math.round(v)) });
    countUp(fig.querySelector('.fx-fn'), nMissed, { format: (v) => String(Math.round(v)) });
    countUp(fig.querySelector('.fx-fp'), nSpur, { format: (v) => String(Math.round(v)) });
    countUp(fig.querySelector('.fx-lat'), meanLat, { format: (v) => signed(v, 1) });
    hasGrown = true;
    if (api) api.grow();
  }, 0.15);

  // -- chart ------------------------------------------------------------------------------
  function render() {
    chart.innerHTML = '';
    const w = chart.clientWidth;
    if (!w) return;
    const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
    const narrow = w < 640;
    let h = narrow ? rem * 20 : rem * 24;
    const pad = { l: narrow ? rem * 2.4 : rem * 6.5, r: rem * 1.2, t: rem * 1.5 };
    const laneH = rem * 0.9;
    const x = d3.scaleLinear().domain([win.start, win.stop]).range([pad.l, w - pad.r]);
    const svg = d3.select(chart).append('svg').attr('width', w);

    // Separate nearby edge markers, then put every numeric label in a free row.
    const nudge = new Map();
    const lastX = { 1: -Infinity, '-1': -Infinity };
    const clipped = recs.filter((d) => d.latency !== null && Math.abs(d.latency) > LIM);
    clipped.slice().sort((a, b) => a.gt_frame - b.gt_frame).forEach((d) => {
      const sgn = d.latency > 0 ? 1 : -1;
      const px = Math.max(x(d.gt_frame), lastX[sgn] + rem * 1.4);
      nudge.set(d, px - x(d.gt_frame));
      lastX[sgn] = px;
    });
    const dx = (d) => nudge.get(d) || 0;
    const probe = svg.append('text').attr('class', 'pop-over tnum');
    const labels = edgeLabelRows(clipped.map((d) => ({
      record: d, side: d.latency > 0 ? 1 : -1, x: x(d.gt_frame) + dx(d),
      width: probe.text(signed(d.latency, 0)).node().getComputedTextLength(),
    })), pad.l, w - pad.r, rem * 0.4);
    probe.remove();
    const labelByRecord = new Map(labels.map((label) => [label.record, label]));
    const extra = (side) => Math.max(0, ...labels.filter((label) => label.side === side).map((label) => label.row)) * rem * 1.1;
    const extraTop = extra(1), extraBottom = extra(-1);
    h += extraTop + extraBottom;
    pad.t += extraTop;
    const lanesY = h - rem * 5.2;
    const y = d3.scaleLinear().domain([LIM, -LIM]).range([pad.t, lanesY - rem * 1.6 - extraBottom]);
    svg.attr('height', h);

    // Tolerance bands.
    svg.append('rect').attr('class', 'fx-b25').attr('x', pad.l).attr('width', w - pad.l - pad.r)
      .attr('y', y(25)).attr('height', y(-25) - y(25));
    svg.append('rect').attr('class', 'fx-b12').attr('x', pad.l).attr('width', w - pad.l - pad.r)
      .attr('y', y(12)).attr('height', y(-12) - y(12));
    svg.append('line').attr('class', 'fx-zero').attr('x1', pad.l).attr('x2', w - pad.r).attr('y1', y(0)).attr('y2', y(0));
    svg.append('g').attr('class', 'axis').attr('transform', 'translate(' + pad.l + ',0)')
      .call(d3.axisLeft(y).tickValues([-60, -25, -12, 0, 12, 25, 60]).tickFormat((v) => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v)).tickSize(0).tickPadding(6));
    if (!narrow) {
      svg.append('text').attr('class', 'svg-label').attr('x', rem * 0.8).attr('y', y(LIM * 0.7)).text('late');
      svg.append('text').attr('class', 'svg-label').attr('x', rem * 0.8).attr('y', y(-LIM * 0.7)).text('early');
    }

    // Label lanes.
    const lanes = [['Ground truth', bundle.labelsWindow.gt_public, 'gt'], ['Engine', bundle.labelsWindow.ours, 'ours']];
    lanes.forEach(([name, runs, cls], i) => {
      const ly = lanesY + i * (laneH + rem * 0.35);
      const g = svg.append('g').attr('class', 'fx-lane fx-lane--' + cls);
      runs.forEach(([a, b, k]) => g.append('rect').attr('class', 'lab-' + k)
        .attr('x', x(a)).attr('width', Math.max(0.5, x(b) - x(a))).attr('y', ly).attr('height', laneH));
      if (!narrow) {
        g.append('text').attr('class', 'svg-label' + (cls === 'gt' ? ' lane-gt' : '')).attr('x', pad.l - rem * 0.5)
          .attr('y', ly + laneH * 0.8).attr('text-anchor', 'end').text(name);
      }
    });
    svg.append('g').attr('class', 'axis').attr('transform', 'translate(0,' + (lanesY + 2 * laneH + rem * 0.8) + ')')
      .call(d3.axisBottom(x).tickValues(d3.range(win.start, win.stop + 1, 1500))
        .tickFormat((f) => {
          const c = matchClock(f, m.kickoffs, m.fps);
          return c ? c.text : int(f);
        }).tickSize(3));

    // Spurious diamonds on the zero line.
    const sp = svg.append('g').attr('class', 'fx-spur').selectAll('path').data(spur).join('path')
      .attr('transform', (d) => 'translate(' + x(d.our_frame) + ',' + y(0) + ') rotate(45)')
      .attr('d', () => {
        const s = rem * 0.42;
        return 'M' + (-s) + ',' + (-s) + 'h' + 2 * s + 'v' + 2 * s + 'h' + (-2 * s) + 'z';
      });
    sp.on('pointermove', (e, d) => showTip(spurTip(d), e.clientX, e.clientY)).on('pointerleave', hideTip);

    // Lollipops.
    const pops = svg.append('g').attr('class', 'fx-pops').selectAll('g').data(recs).join('g')
      .attr('class', (d) => 'pop ' + (KIND_CLASS[d.kind] || '') + ' is-' + d.status)
      .attr('transform', (d) => 'translate(' + x(d.gt_frame) + ',0)')
      .attr('tabindex', 0).attr('role', 'button')
      .attr('aria-label', (d) => d.from + ' to ' + d.to + ' at frame ' + d.gt_frame + ', ' + d.status +
        (d.latency !== null ? ', latency ' + d.latency + ' frames' : ''));
    const lat = (d) => (d.latency === null ? 0 : Math.max(-LIM, Math.min(LIM, d.latency)));
    pops.append('line').attr('class', 'pop-tick').attr('y1', lanesY - rem * 0.5).attr('y2', lanesY + 2 * laneH + rem * 0.4);
    const stems = pops.append('line').attr('class', 'pop-stem').attr('y1', y(0)).attr('y2', y(0));
    const heads = pops.append('g').attr('class', 'pop-head').attr('transform', (d) => 'translate(' + dx(d) + ',' + y(0) + ')');
    heads.filter((d) => dx(d) > 0).append('line').attr('class', 'pop-stem').attr('x1', (d) => -dx(d)).attr('x2', 0);
    heads.append('circle').attr('r', rem * 0.5);
    heads.filter((d) => d.status === 'missed').append('path').attr('class', 'pop-cross')
      .attr('d', () => {
        const s = rem * 0.28;
        return 'M' + (-s) + ',' + (-s) + 'L' + s + ',' + s + 'M' + s + ',' + (-s) + 'L' + (-s) + ',' + s;
      });
    heads.filter((d) => labelByRecord.has(d)).append('text').attr('class', 'pop-over tnum')
      .attr('text-anchor', 'middle').attr('x', (d) => labelByRecord.get(d).x - x(d.gt_frame) - dx(d))
      .attr('y', (d) => {
        const row = labelByRecord.get(d).row;
        return rem * (d.latency > 0 ? -0.8 - row * 1.1 : 1.3 + row * 1.1);
      })
      .text((d) => signed(d.latency, 0));
    pops.on('pointermove', (e, d) => showTip(recTip(d), e.clientX, e.clientY))
      .on('pointerleave', hideTip)
      .on('click', (e, d) => select(recs.indexOf(d)))
      .on('keydown', (e, d) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          select(recs.indexOf(d));
        }
      });

    function grow(animate) {
      const t = dur(animate ? 900 : 0);
      stems.transition().duration(t).delay((d, i) => (animate ? dur(i * 45) : 0)).ease(d3.easeBackOut.overshoot(1.4))
        .attr('y2', (d) => y(lat(d)));
      heads.transition().duration(t).delay((d, i) => (animate ? dur(i * 45) : 0)).ease(d3.easeBackOut.overshoot(1.4))
        .attr('transform', (d) => 'translate(' + dx(d) + ',' + y(lat(d)) + ')');
    }
    api = {
      grow: () => grow(true),
      mark(i) {
        pops.classed('is-sel', (d, k) => k === i);
      },
    };
    if (hasGrown) grow(false);
    api.mark(selected);
  }
  // The lollipops grow the first time the section is on screen, and stay grown after a resize.
  let hasGrown = false;
  observeSize(chart, render);

  // -- detail card ------------------------------------------------------------------------
  function traceHtml(rows, big) {
    if (!rows || !rows.length) return '';
    const cells = (key) => rows.map((r) => '<i class="tc tc--' + String(r[key]).toLowerCase() + (r.off === 0 ? ' tc--zero' : '') +
      '" title="offset ' + r.off + ': ' + r[key] + '"></i>').join('');
    return '<div class="trace' + (big ? ' trace--big' : '') + '">' +
      '<div class="trace-row"><span>truth</span><div class="trace-cells">' + cells('gt') + '</div></div>' +
      '<div class="trace-row"><span>engine</span><div class="trace-cells">' + cells('ours') + '</div></div>' +
      '<div class="trace-axis"><span>' + signed(rows[0].off, 0) + '</span><span>0</span><span>' + signed(rows[rows.length - 1].off, 0) + '</span></div></div>';
  }

  function kinHtml(frame) {
    if (!kin || !kin.gt_transitions) return '';
    const rec = kin.gt_transitions.find((r) => r.gt_frame === frame);
    if (!rec || !rec.labels_trace || !rec.labels_trace.length) return '';
    const rows = rec.labels_trace;
    const series = [['speed', 'ball speed, m/s', 'k-speed'], ['d_h', 'nearest H, m', 'k-dh'], ['d_a', 'nearest A, m', 'k-da']];
    const W = 320;
    const H = 70;
    const offs = rows.map((r) => r.off);
    const xs = d3.scaleLinear().domain([d3.min(offs), d3.max(offs)]).range([4, W - 4]);
    const vals = [];
    series.forEach(([k]) => rows.forEach((r) => { if (r[k] !== null && r[k] !== undefined) vals.push(r[k]); }));
    const ys = d3.scaleLinear().domain([0, Math.max(1, d3.max(vals) || 1)]).range([H - 4, 4]);
    const paths = series.map(([k, , cls]) => {
      const line = d3.line().defined((r) => r[k] !== null && r[k] !== undefined).x((r) => xs(r.off)).y((r) => ys(r[k]));
      return '<path class="' + cls + '" d="' + line(rows) + '"/>';
    }).join('');
    return '<div class="kin"><svg viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none">' +
      '<line class="kin-zero" x1="' + xs(0) + '" x2="' + xs(0) + '" y1="0" y2="' + H + '"/>' + paths + '</svg>' +
      '<div class="kin-legend">' + series.map(([, t, cls]) => '<span><i class="' + cls + '"></i>' + t + '</span>').join('') + '</div></div>';
  }

  function select(i) {
    selected = i;
    if (api) api.mark(i);
    const d = recs[i];
    if (!d) {
      detail.innerHTML = '';
      return;
    }
    const c = matchClock(d.gt_frame, m.kickoffs, m.fps);
    const tags = (d.tags || []).slice(0, 6).map((t) => '<li><b>' + esc(t.code) + '</b> ' + esc(t.team || '') +
      ' <span class="tnum">' + signed(t.start_frame - d.gt_frame, 0) + ' to ' + signed(t.end_frame - d.gt_frame, 0) + '</span></li>').join('');
    const evs = (d.events || []).map((e) => '<li><b>' + esc(EVENT_TEXT[e[1]] || e[1]) + '</b> ' + esc(e[2] || 'no team') +
      ' <span class="tnum">' + signed(e[0] - d.gt_frame, 0) + ' frames, confidence ' + fixed(e[3], 2) + '</span></li>').join('');
    detail.innerHTML =
      '<div class="fxd-head">' +
      '<div><span class="fxd-kind ' + (KIND_CLASS[d.kind] || '') + '">' + esc(KIND_TEXT[d.kind] || d.kind) + '</span>' +
      '<h4><span class="chip chip--' + d.from.toLowerCase() + '">' + d.from + '</span> to <span class="chip chip--' + d.to.toLowerCase() + '">' + d.to + '</span>' +
      ' at frame <span class="tnum">' + int(d.gt_frame) + '</span></h4>' +
      '<p class="muted tnum">video ' + videoTime(d.gt_frame, m.fps) + (c ? ', match clock ' + c.text : '') + '</p></div>' +
      '<div class="fxd-verdict is-' + d.status + '"><b>' + d.status + '</b>' +
      (d.latency !== null ? '<span class="tnum">' + signed(d.latency, 0) + ' frames' + (d.our_frame !== null ? ', the engine\u2019s at ' + int(d.our_frame) : '') + '</span>' : '<span>no transition of the engine to ' + d.to + '</span>') +
      '</div>' +
      (local ? '<button class="btn fxd-play" type="button">Play this in the replay</button>' : '') +
      '</div>' +
      '<div class="fxd-body"><div><h5>Labels every 3 frames</h5>' + traceHtml(d.labels_trace, true) + kinHtml(d.gt_frame) + '</div>' +
      '<div><h5>Rule engine events near it</h5><p class="fxd-aside">Not used by the engine.</p>' +
      (evs ? '<ul>' + evs + '</ul>' : '<p class="muted">No rule engine event within ' + (tr.trace_half_width || 50) + ' frames.</p>') + '</div>' +
      '<div><h5>Smart Tagging phases</h5><p class="fxd-aside">Not read by the engine.</p>' +
      (tags ? '<ul>' + tags + '</ul>' : '<p class="muted">No tag overlaps.</p>') + '</div></div>' +
      '<div class="fxd-nav"><button class="btn btn--ghost" type="button" data-nav="-1">Previous</button>' +
      '<span class="muted tnum">' + (i + 1) + ' of ' + recs.length + '</span>' +
      '<button class="btn btn--ghost" type="button" data-nav="1">Next</button></div>';
    const play = detail.querySelector('.fxd-play');
    if (play) {
      play.addEventListener('click', () => {
        state.set({ frame: Math.max(win.start, d.gt_frame - 50), playing: true });
        scrollToEl(document.getElementById('engine'));
      });
    }
    detail.querySelectorAll('[data-nav]').forEach((b) => b.addEventListener('click', () => {
      select((selected + Number(b.dataset.nav) + recs.length) % recs.length);
    }));
  }

  function recTip(d) {
    return '<div class="tt-title">' + d.from + ' to ' + d.to + ', frame ' + int(d.gt_frame) + '</div>' +
      '<div class="tt-row"><span>kind</span><b>' + esc(d.kind) + '</b></div>' +
      '<div class="tt-row"><span>status</span><b>' + esc(d.status) + '</b></div>' +
      '<div class="tt-row"><span>latency</span><b>' + (d.latency === null ? 'none' : signed(d.latency, 0) + ' frames') + '</b></div>' +
      traceHtml(d.labels_trace, false) + '<div class="tt-note">Click for the details.</div>';
  }

  function spurTip(d) {
    const n = d.nearest_gt_transition;
    return '<div class="tt-title">Spurious ' + d.from + ' to ' + d.to + ', frame ' + int(d.our_frame) + '</div>' +
      '<div class="tt-row"><span>ground truth there</span><b>' + esc(d.gt_label_at_frame) + '</b></div>' +
      (n ? '<div class="tt-row"><span>nearest ground-truth transition</span><b>' + esc(n.from_to) + ' at ' + int(n.frame) + ', ' + n.distance + ' frames</b></div>' : '') +
      traceHtml(d.labels_trace, false);
  }

  select(selected);

  // -- latency by kind --------------------------------------------------------------------
  const kinds = fig.querySelector('.fx-kinds');
  const lbk = tr.latency_by_kind || {};
  Object.keys(KIND_CLASS).forEach((k) => {
    const v = lbk[k];
    if (!v) return;
    const card = el('div', 'fx-kind ' + KIND_CLASS[k]);
    const dots = (v.nearest_latencies || []).map((l) => {
      const c = Math.max(-LIM, Math.min(LIM, l));
      const pctX = ((c + LIM) / (2 * LIM)) * 100;
      const inTol = Math.abs(l) <= tol;
      return '<i class="' + (inTol ? 'in' : 'out') + '" style="left:' + pctX.toFixed(2) + '%" title="' + signed(l, 0) + ' frames"></i>';
    }).join('');
    card.innerHTML = '<div class="fk-name">' + esc(KIND_TEXT[k]) + '</div>' +
      '<div class="fk-big tnum">' + v.n_matched + '<small> of ' + v.n_gt + ' matched</small></div>' +
      '<div class="fk-strip"><span class="fk-tol" style="left:' + (((LIM - tol) / (2 * LIM)) * 100) + '%;right:' + (((LIM - tol) / (2 * LIM)) * 100) + '%"></span><span class="fk-zero"></span>' + dots + '</div>' +
      '<div class="fk-meta tnum">mean ' + signed(v.matched_mean, 1) + ', median ' + signed(v.matched_median, 1) + ' frames</div>';
    kinds.appendChild(card);
  });

  // -- kickoffs and sandwiches ------------------------------------------------------------
  const foot = fig.querySelector('.fx-foot');
  const kos = tr.kickoffs || [];
  const sw = tr.sandwiches || [];
  let html = '';
  if (kos.length) {
    html += '<div class="fx-note"><h5>Kickoffs in the window</h5><ul>' + kos.map((k) =>
      '<li><b>' + esc(k.kind) + '</b>, frame <span class="tnum">' + int(k.frame) + '</span>: the engine\u2019s label <span class="chip chip--' +
      String(k.ours).toLowerCase() + '">' + esc(k.ours) + '</span>. The rule engine\u2019s kickoff detection, ' +
      esc(KICKOFF_RULE_TEXT[k.rule] || k.rule) + ', gives team ' + esc(k.team || 'none') + '.</li>').join('') + '</ul></div>';
  }
  if (sw.length) {
    html += '<div class="fx-note"><h5>Same team either side of a stoppage</h5><ul>' + sw.map((s) =>
      '<li>Team ' + esc(s.team) + ', ground-truth <code>D</code> from <span class="tnum">' + int(s.gt_d_start) + '</span> to <span class="tnum">' +
      int(s.gt_d_stop) + '</span>: ' + (s.separate ? 'the engine keeps the two possessions apart' : 'the engine merges the two possessions') +
      ', its <code>D</code> covers ' + Math.round(s.our_d_share_of_gt_d * 100) + '% of it.</li>').join('') + '</ul></div>';
  }
  foot.innerHTML = html;

  // A frame from elsewhere (a public-case chip) selects the nearest transition.
  state.on('frame', (f) => {
    if (f < win.start || f >= win.stop || !recs.length) return;
    let best = 0;
    recs.forEach((r, i) => {
      if (Math.abs(r.gt_frame - f) < Math.abs(recs[best].gt_frame - f)) best = i;
    });
    if (Math.abs(recs[best].gt_frame - f) <= LIM && best !== selected) select(best);
  });
}
