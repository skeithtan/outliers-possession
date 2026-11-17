/**
 * Section "The rule engine, step by step: the fallback" in section 04: seven scroll steps, the
 * pipeline diagram and the pitch panel. The rule engine writes the whole output when the learned
 * models are switched off. Every sentence names it "the rule engine".
 *
 * `mount(el, ctx)` builds the whole section inside the empty
 * `<section id="engine" class="story-section" data-section="engine">`:
 *   .steps > .step[data-step="0..6"]   the copy, one step per mechanism
 *   .figure                             sticky: pipeline SVG, pitch panel, mini seek strip
 *
 * The scroller sets `state.activeStep = {section: 'engine', step, progress}`. On a new step the
 * section lights the pipeline node and starts the scene for the step. In local mode the panel plays
 * the real tracking of the documented public case, looped over 6 s. In public mode it plays the
 * synthetic scene. Constant values in the copy come from `bundle.config` by name.
 *
 * @module sections/engine
 */

import { createReplay } from '../replay.js';
import { configAccessor, engineDefault, STEP_SCENES } from '../synthetic.js';

const NODES = [
  { id: 'tracks', label: 'tracks', file: 'tracks.py' },
  { id: 'events', label: 'events', file: 'events.py' },
  { id: 'deadball', label: 'dead ball', file: 'deadball.py' },
  { id: 'possession', label: 'possession', file: 'possession.py' },
  { id: 'decoder', label: 'decoder', file: 'decoder.py' },
  { id: 'submission', label: 'submission', file: 'submission.py' },
];

/** Local-mode cases: the public frames of the plan, 2 s before to 4 s after (6 s loop). */
const CASES = [
  [{ gap: true, label: 'ball back after a gap', insets: 'cleaning' }],
  [{ frame: 13704, label: '13704 kick', insets: 'impulse' }, { frame: 11624, label: '11624 reception', insets: 'impulse' }, { frame: 11553, label: '11553 fitted contact point', insets: 'impulse' }],
  [{ frame: 17838, label: '17838 slow ball', insets: 'impulse' }],
  [{ frame: 12851, label: '12851 out', insets: 'deadball' }, { frame: 15889, label: '15889 pursuit stop', insets: 'pursuit' }],
  [{ frame: 18389, label: '18389 chained restart', insets: 'restart' }],
  [{ frame: 11624, label: '11624 owner switch', insets: 'logE' }, { frame: 13704, label: '13704 owner switch', insets: 'logE' }],
  [{ frame: 15654, label: '15654 goalkeeper swap, corrected', insets: null }],
];

const STEP_NODES = [['tracks'], ['events'], ['events'], ['deadball'], ['deadball'], ['possession', 'decoder', 'submission'], ['tracks']];

function h(tag, attrs, children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k === 'html') e.innerHTML = v;
    else e.setAttribute(k, v);
  }
  for (const c of [].concat(children || [])) e.append(c);
  return e;
}

function svg(tag, attrs) {
  const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  return e;
}

/**
 * Copy of the seven steps. `c(name, unit)` renders the bundle value of one constant as a chip. The
 * chip shows the value and its unit. Its tooltip gives the name of the constant in engine/config.py.
 * Plain sentences for a reader who has not seen the code.
 */
function stepCopy(c, cfg) {
  return [
    {
      kicker: '1  Tracks', title: 'Clean the ball first',
      body: [
        ['The tracked ball has glitches. A frame that lands more than ', c('BALL_MAX_JUMP_M', 'm'),
          ' from the last good frame is a teleport: 75 m/s, faster than any shot. It is removed.'],
        ['A visible run shorter than ', c('BALL_MIN_RUN', 'frames'), ' is an isolated detection and is dropped. Every later step reads the cleaned ball. The engine uses the same cleaning.'],
      ],
    },
    {
      kicker: '2  Events', title: 'A contact is the peak of a bump',
      body: [
        ['The provider smooths the ball, so a kick shows as a bump of acceleration, not a step. The rule engine measures the change of ball velocity across ',
          c('CONTACT_HALF_WINDOW', 'frames'), ' on each side, the measured half-width of the smoothing plus one.'],
        ['A peak above ', c('DV_CONTACT', 'm/s'), ' with a player within ', c('R_TOUCH', 'm'),
          ', the touch radius, is a contact, placed at the centre of the bump. If the ball speeds up, the contact is a kick. If it slows down, it is a reception. Both are owner events: the owner becomes the team that made them.'],
        ...(cfg.has('CONTACT_POINT') && cfg.get('CONTACT_POINT') === 'kink_min'
          ? [['The smoothing also cuts the corner where the ball turns, so the smoothed ball can pass about 2 m from the player who played it. ',
            'Around each peak the rule engine fits a straight path with one change of velocity, seen through the same smoothing. Its corner is the fitted contact point. ',
            'Each team counts the smaller of two distances: its nearest distance over the window, and its distance to the fitted contact point.']]
          : []),
        ['A redirect turns the ball more than it changes its speed. A redirect by the team without the ball changes the owner only with proof: ',
          'a next contact by that team, or the team-movement check. A team’s forward speed is the mean velocity of its players towards the goal it attacks. ',
          'The check passes when the forward speed of the toucher’s team rises by ', c('SHAPE_CONFIRM_MS', 'm/s'),
          ' against the other team’s within ', c('SHAPE_LAG_FRAMES', 'frames'), ' of the touch. Both teams need enough visible players, and the window must stay inside one half. Without proof the rule engine ignores the redirect.'],
        ['A ball in the air shows on the ground, metres from the player who heads it. The rule engine normally ignores a peak with no player within ', c('R_TOUCH', 'm'),
          '. It still becomes a contact, a distant touch, when the nearest player is within ', c('R_REACH_M', 'm'),
          ' and the team-movement check confirms it for that player’s team. Its kind follows the usual rules, so a redirect still needs proof.'],
      ],
    },
    {
      kicker: '3  Events', title: 'Possession without a kick',
      body: [
        ['A ball within ', c('R_CONTROL', 'm'), ' of one player, slower than ', c('CONTROL_MAX_SPEED', 'm/s'), ', for ',
          c('CONTROL_FRAMES', 'frames'), ' is a slow-ball control.'],
        ['A player who keeps a moving ball within ', c('R_DRIBBLE', 'm'), ' for ', c('DRIBBLE_FRAMES', 'frames'),
          ' dribbles, if the median speed of the ball relative to the player stays under ', c('DRIBBLE_REL_SPEED', 'm/s'), '. The whole run decides, not one frame.'],
      ],
    },
    {
      kicker: '4  Dead ball', title: 'Three ways the ball dies',
      body: [
        ['Out: the ball is seen beyond a line, or lost while heading for a line within ', c('OUT_MAX_RANGE_M', 'm'), '. The rule engine rolls it on with a friction of ',
          c('FRICTION_DECEL', 'm/s²'), ...(cfg.get('DELTA_OUT_LAG') === 0
            ? ['. If the roll stops short, the stoppage starts at the last sighting, with no added delay. The players must still confirm the stop.']
            : ['. If the roll stops short, the stoppage starts ', c('DELTA_OUT_LAG', 'frames'), ' after the last sighting.'])],
        ['Pursuit stop: a fast ball, and the ', c('PURSUIT_N_NEAREST', 'nearest players'), ' fall below ', c('PURSUIT_STOP_SPEED', 'm/s'),
          '. Collective stop: the players’ mean speed below ', c('V_STOP', 'm/s'), '. A stoppage start counts only if the players stop within ', c('ONSET_CONFIRM_S', 's'), '.'],
        ['When the ball is missing, the rule engine adds weighted signals into a dead-ball score from 0 to 1, P(dead): the ball missing while the players are slow, the players slow, and the centres of the two teams far apart. ',
          'A collective stop in a ball gap, a ball-gap stoppage, needs the score above ', c('DEAD_EVIDENCE_HIGH', ''), ' for ', c('COLLECTIVE_STOP_S', 's'), '.'],
        ...(cfg.has('ONSET_KNEE') && cfg.get('ONSET_KNEE')
          ? [['When the ball is missing, a collective stop is dated at the knee of the players’ deceleration: ', c('DELTA_KNEE', 'frames'),
            ' from the half-way point of their fall, never before the last sighting of the ball.']]
          : []),
      ],
    },
    {
      kicker: '5  Dead ball', title: 'Play resumes as a sequence',
      body: [
        ['A restart is a kick, a speed rise above ', c('DV_RESTART', 'm/s'), ' over ', c('RESTART_K', 'frames'),
          ' with a player at the ball, or a ball that reappears faster than ', c('REAPPEAR_SPEED', 'm/s'), '.'],
        ['It is confirmed when the players start to move within ', c('RESTART_CONFIRM_S', 's'), ', or when the next confirmed restart kick follows within ',
          c('RESTART_CHAIN_FRAMES', 'frames'), ', a chained restart. Play resumes ', c('DELTA_RESTART_KICK', 'frames'), ' after the kick’s rise begins, at the centre of its bump.'],
        ['A confirmed reappearance resumes play at the frame where the ball is seen again, with ', c('DELTA_RESTART_REAPPEAR', 'frames'), ' of added delay.'],
      ],
    },
    {
      kicker: '6  Decoder', title: 'One label per frame, switched once',
      body: [
        ['The owner is the team of the last kick, reception, slow-ball control or dribble. A Viterbi decoder with three states, H, A and D, turns it into labels. On each frame the owner gets a probability of ',
          c('P_OWNER', ''), '.'],
        ['A team switch costs ', c('C_SWITCH', 'nats'), ' (a log-probability penalty), and only ', c('C_SWITCH_EVENT', 'nats'), ' within ', c('SWITCH_EVENT_WINDOW', 'frames'),
          ' of an owner event of the new team. So a flickering nearest player never flips the label, and an event flips it once.'],
        ['The new team’s first owner event often comes a second or two after it took the ball. So the rule engine moves each change of owner back over the frames where the new team’s nearest player is within ',
          c('BACKDATE_RADIUS_M', 'm'), ' of the visible ball and at least as near as the other team’s, for at most ', c('BACKDATE_CAP_S', 's'),
          ', as long as the possessions on both sides keep at least ', c('BACKDATE_MIN_PRE_SPELL_S', 's'), '. This back-dating changes no stoppage.'],
        ['Where no event decides, the nearest player ', c('NEAREST_LAG', 'frames'), ' earlier gives weak evidence: the raw nearest player runs ahead of the true owner. ',
          'Smart Tagging adds only a weak prior: a tagged team phase adds a small team prior (likelihood ratio ', c('TAG_TEAM_PRIOR_RATIO', ''), ') where no event decides, and a set-piece tag adds dead-ball evidence and restart confidence.'],
      ],
    },
    {
      kicker: '7  Tracks', title: 'Fix swapped goalkeeper tracks',
      body: [
        ['One tracker fault fooled every detector: the track of one goalkeeper jumped onto the other goalkeeper’s body, while that goalkeeper’s own track froze. The rule engine credited the wrong team at full confidence.'],
        ['The rule engine finds each goalkeeper’s track from the tracking alone: at least ', c('GK_OWN_BOX_SHARE', ''), ' of its frames in its own penalty area. When a moving goalkeeper track is beyond its half-way line and the other goalkeeper’s track is frozen within ',
          c('GK_SWAP_R', 'm'), ', the rule engine swaps the positions back before any detector runs. This happened at frame 15654 of the public window. The engine uses the same correction.'],
      ],
    },
  ];
}

function buildPipeline() {
  const W = 720, H = 76;
  const root = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'pl', role: 'img', 'aria-label': 'The rule engine: tracks, events, dead ball, possession, decoder, submission' });
  const defs = svg('defs');
  const grad = svg('radialGradient', { id: 'pl-glow' });
  grad.append(svg('stop', { offset: '0%', 'stop-color': 'currentColor', 'stop-opacity': '0.55' }), svg('stop', { offset: '100%', 'stop-color': 'currentColor', 'stop-opacity': '0' }));
  defs.append(grad);
  root.append(defs);
  const gap = W / NODES.length;
  const nodeW = gap - 26, nodeH = 44, y = (H - nodeH) / 2;
  const nodes = {};
  NODES.forEach((n, k) => {
    const x = k * gap + 13;
    if (k > 0) {
      const x0 = x - 26, x1 = x;
      const edge = svg('path', { d: 'M' + x0 + ' ' + H / 2 + ' L' + (x1 - 4) + ' ' + H / 2, class: 'pl-edge', 'data-to': n.id });
      const head = svg('path', { d: 'M' + (x1 - 7) + ' ' + (H / 2 - 4) + ' L' + (x1 - 1) + ' ' + H / 2 + ' L' + (x1 - 7) + ' ' + (H / 2 + 4), class: 'pl-head', 'data-to': n.id });
      root.append(edge, head);
    }
    const g = svg('g', { class: 'pl-node', 'data-node': n.id, tabindex: '0', role: 'button', 'aria-label': n.label });
    g.append(svg('ellipse', { cx: x + nodeW / 2, cy: H / 2, rx: nodeW * 0.62, ry: nodeH * 0.9, fill: 'url(#pl-glow)', class: 'pl-halo' }));
    g.append(svg('rect', { x, y, width: nodeW, height: nodeH, rx: 9, class: 'pl-box' }));
    const t1 = svg('text', { x: x + nodeW / 2, y: y + 19, class: 'pl-label', 'text-anchor': 'middle' });
    t1.textContent = n.label;
    const t2 = svg('text', { x: x + nodeW / 2, y: y + 34, class: 'pl-file', 'text-anchor': 'middle' });
    t2.textContent = n.file;
    g.append(t1, t2);
    root.append(g);
    nodes[n.id] = g;
  });
  return { root, nodes };
}

/**
 * Mount the section.
 * @param {HTMLElement} el  the empty section element
 * @param {{bundle: Object, local: Object|null, state: Object, d3?: Object, mode: string}} ctx
 * @returns {{replay: Object, destroy: function(): void}}
 */
export function mount(el, ctx) {
  const bundle = ctx.bundle || {};
  const cfg = configAccessor(bundle.config);
  const isLocal = ctx.mode === 'local' && !!ctx.local;
  el.textContent = '';
  el.classList.add('engine-section');

  function c(name, unit) {
    const chip = h('span', { class: 'k-const tnum', title: 'setting ' + name + ' in engine/config.py' });
    let v = 'n/a';
    // an older bundle has no R_REACH_M: show the engine default, as the scenes use it
    const known = cfg.has(name) || engineDefault(name) !== undefined;
    if (known) {
      const raw = cfg.has(name) ? cfg.get(name) : engineDefault(name);
      v = typeof raw === 'number' ? String(raw) : Array.isArray(raw) ? raw.join(', ') : String(raw);
    }
    chip.append(h('b', { text: v }), document.createTextNode(unit ? ' ' + unit : ''));
    return chip;
  }

  const header = h('header', { class: 'engine-head' }, [
    h('p', { class: 'kicker', html: '<span class="num">04</span> The possession rules <span class="sep">/</span> the fallback, step by step' }),
    h('h2', { class: 'engine-title', text: 'The rule engine, step by step: the fallback' }),
    h('p', { class: 'engine-lede', text: 'The rule engine is a set of physical rules, and the fallback of the engine: one configuration setting makes the engine write its labels, byte for byte. ' +
      'It decides dead ball first, then finds the owner events, then decodes whole possessions. ' +
      'An owner event is a moment that shows which team has the ball: a kick, a reception, a slow-ball control or a dribble. ' +
      'Each step shows one mechanism with the values that the rule engine uses. Hover a value to see its name in the code. ' +
      (isLocal ? 'The panel plays the real tracking of a moment in the public window.' : 'The panel plays a synthetic scene: the hosted page never holds tracking data.') }),
  ]);

  const steps = h('div', { class: 'steps' });
  const copy = stepCopy(c, cfg);
  const caseRows = [];
  copy.forEach((s, k) => {
    const step = h('article', { class: 'step', 'data-step': String(k) });
    step.append(h('p', { class: 'step-kicker tnum', text: s.kicker }), h('h3', { class: 'step-title', text: s.title }));
    for (const para of s.body) step.append(h('p', { class: 'step-body' }, para));
    if (isLocal) {
      const row = h('div', { class: 'step-cases' });
      CASES[k].forEach((cs, j) => {
        const b = h('button', { class: 'case-chip tnum', type: 'button', text: cs.label });
        b.addEventListener('click', () => { current = -1; applyStep(k, j); });
        row.append(b);
      });
      step.append(row);
      caseRows.push(row);
    }
    steps.append(step);
  });

  const pipe = buildPipeline();
  const panel = h('div', { class: 'engine-panel' });
  const strip = h('div', { class: 'engine-strip' });
  const figure = h('div', { class: 'figure engine-figure' }, [h('div', { class: 'engine-pipe' }, [pipe.root]), panel, strip]);
  el.append(header, steps, figure);

  const replay = createReplay(panel, ctx, { stripEl: strip });

  for (const [id, node] of Object.entries(pipe.nodes)) {
    const go = () => {
      const k = STEP_NODES.findIndex((ns) => ns.indexOf(id) >= 0);
      const target = steps.querySelector('.step[data-step="' + (k < 0 ? 0 : k) + '"]');
      if (target) target.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'center' });
      if (k >= 0) applyStep(k);
    };
    node.addEventListener('click', go);
    node.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
  }

  let current = -1;
  let currentCase = 0;
  let seq = 0;

  function light(k) {
    const on = new Set(STEP_NODES[k] || []);
    for (const [id, node] of Object.entries(pipe.nodes)) node.classList.toggle('is-active', on.has(id));
    pipe.root.querySelectorAll('.pl-edge, .pl-head').forEach((e) => e.classList.toggle('is-active', on.has(e.getAttribute('data-to'))));
    steps.querySelectorAll('.step').forEach((s) => s.classList.toggle('is-active', Number(s.dataset.step) === k));
  }

  // Minimum run of invisible frames that counts as a gap (1 s at 25 fps, the engine's own smallest
  // visible run is shorter). The case is picked from the local frame flags at run time.
  const GAP_MIN_FRAMES = 25;
  const FLAG_VISIBLE = 2;

  /** The first frame of the public window where the ball comes back after a gap, with the gap length. */
  async function gapCase() {
    const local = ctx.local;
    const w = (bundle.manifest && bundle.manifest.public_window) || { start: 11185, stop: 18685 };
    const half = 1;
    const hinfo = local.manifest.halves.find((x) => x.half === half);
    const fr = (local.frames && local.frames[half]) || (local.halves && local.halves[half] && local.halves[half].frames);
    let flags = fr && fr.views && fr.views.flags;
    if (!flags) {
      const base = ctx.localBase || './data/local/';
      const header = await (await fetch(base + hinfo.files.frames)).json();
      const buf = await (await fetch(base + hinfo.files.frames.replace(/\.json$/, '.bin'))).arrayBuffer();
      const a = header.arrays.flags;
      flags = new Uint8Array(buf, a.offset, a.shape[0]);
    }
    let off = 0;
    for (let f = w.start; f < w.stop; f++) {
      if (flags[f - hinfo.frame_start] & FLAG_VISIBLE) {
        if (off >= GAP_MIN_FRAMES) {
          return { frame: f, title: 'the ball comes back after a gap of ' + off + ' frames, the raw and cleaned tracks are drawn together' };
        }
        off = 0;
      } else off++;
    }
    const ko = ((bundle.manifest && bundle.manifest.kickoffs) || []).find((x) => x.half === half);
    return { frame: ko ? ko.frame : w.start, title: 'no ball gap of ' + GAP_MIN_FRAMES + ' frames in the window, the kickoff frame is shown' };
  }

  const w0 = (bundle.manifest && bundle.manifest.public_window) || { start: 11185 };

  function applyStep(k, j, autoplay) {
    const caseIdx = j || 0;
    const auto = autoplay !== false;
    if (k === current && caseIdx === currentCase) return;
    current = k;
    currentCase = caseIdx;
    light(k);
    if (isLocal) {
      caseRows.forEach((row, r) => row.querySelectorAll('.case-chip').forEach((b, q) => b.classList.toggle('is-on', r === k && q === caseIdx)));
      const cs = CASES[k][caseIdx];
      const start = (f, title) => replay.showCase({ frame: f, from: f - 50, to: f + 100, insets: cs.insets, title: title || cs.label, autoplay: auto });
      const token = ++seq;
      // the search is async: a later step must win over a late answer
      if (cs.gap) gapCase().then((g) => { if (token === seq) start(g.frame, g.title); }).catch(() => { if (token === seq) start(w0.start, 'the gap search failed, the first frame of the window is shown'); });
      else start(cs.frame);
    } else {
      ++seq;
      replay.showScenes(STEP_SCENES[k], { autoplay: auto });
    }
  }

  // On entry, a seek from elsewhere (hero ribbon, forensics, constants) has cleared the case loop
  // and hidden "Keep playing". Keep that frame until the scroll that brought the reader here stops
  // for 250 ms, so a smooth scroll through the steps does not replace it. A paused case (the
  // step 0 preview) starts playing on entry.
  const keepBtn = replay.el.querySelector('.rp-keep');
  let wasActive = false;
  let holding = false;
  let holdTimer = 0;
  const onHoldScroll = () => {
    clearTimeout(holdTimer);
    holdTimer = setTimeout(() => { holding = false; window.removeEventListener('scroll', onHoldScroll); }, 250);
  };
  const off = ctx.state.on('activeStep', (s) => {
    if (!s || s.section !== 'engine') { wasActive = false; replay.setActive(false); return; }
    replay.setActive(true);
    const k = Math.max(0, Math.min(6, s.step | 0));
    if (!wasActive) {
      wasActive = true;
      if (isLocal && keepBtn && keepBtn.hidden) {
        holding = true;
        window.addEventListener('scroll', onHoldScroll, { passive: true });
        onHoldScroll();
        current = -2;
        // the panel no longer shows a documented case
        const cap = replay.el.querySelector('.rp-caption');
        if (cap) cap.textContent = '';
        caseRows.forEach((row) => row.querySelectorAll('.case-chip').forEach((b) => b.classList.remove('is-on')));
      }
      else if (!replay.playing) current = -1;
    }
    if (holding && keepBtn.hidden) {
      if (k !== current) { current = k; currentCase = 0; light(k); }
      return;
    }
    if (k !== current) applyStep(k);
  });
  const now = ctx.state.get().activeStep;
  if (now && now.section === 'engine') { wasActive = true; replay.setActive(true); applyStep(now.step | 0); }
  else applyStep(0, 0, false);

  return {
    replay,
    applyStep,
    destroy() { off(); replay.destroy(); },
  };
}

export default { mount };
