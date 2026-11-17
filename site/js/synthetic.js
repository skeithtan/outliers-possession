/**
 * Synthetic scenes for the engine section in public mode. Real tracking is never published, so
 * the hosted page illustrates each mechanism on a scripted, deterministic scene.
 *
 * Every scene is built the same way:
 *   1. a seeded agent simulation moves 22 players and a raw ball (kicks, receptions, dribbles,
 *      friction from FRICTION_DECEL);
 *   2. the raw ball is smoothed with a Gaussian kernel, as the ATD provider does (RESEARCH-2 M13).
 *      The kernel width comes from CONTACT_HALF_WINDOW: h = HWHM + 1, sigma = (h - 1) / sqrt(2 ln 2);
 *   3. ports of the engine detectors run on the result with the constants of `bundle.config`:
 *      the bump contacts with the C18 reach clause and the K1 kink contact point (CONTACT_POINT
 *      "kink_min"), control and dribble rows with the redirect rule
 *      and its team-shape clause (engine/events.py, engine/tracks.py), out and pursuit onsets and
 *      restart candidates with the chain rule (engine/deadball.py), the teleport rule
 *      (engine/tracks.py), the goalkeeper guard (engine/tracks.py) and the Viterbi decoder
 *      (engine/decoder.py);
 *   4. the scene turns each frame into a frame state for pitch.js, with overlays and signal cards.
 *
 * The markers on screen are what the ported detector finds on the scene, not scripted labels.
 * No threshold is written in this file: every engine constant is read by name from the config.
 * Randomness comes from mulberry32 with a fixed seed per scene, so a scene is the same on every
 * load. Math.random is never used.
 *
 *   createScenes({config, pitchM, fps}) -> {get(id), ids, steps, cfg}
 *   scene = {id, title, n, fps, frameAt(i), caption(i), timeline, keyFrame}
 *
 * @module synthetic
 */

export const NS = 46;
const H_SLOTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const A_SLOTS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33];
const TRAIL = 10;
const BALL_TRAIL = 8;

/** Scenes for each of the seven steps, in play order. */
export const STEP_SCENES = [
  ['cleaning'],
  ['contacts', 'kink', 'redirect', 'redirect-flat', 'redirect-reach', 'redirect-reach-flat'],
  ['dribble'],
  ['out-roll', 'out-lag', 'pursuit', 'collective'],
  ['reappear', 'chain'],
  ['decoder'],
  ['gk'],
];

// -- config -----------------------------------------------------------------------------------

/**
 * Accessor over `config.json` (`{sections: [{fields: [{name, value}]}]}`) or a flat object.
 * `get(name)` throws when the constant is missing, so a scene never runs on an invented value.
 */
export function configAccessor(config) {
  const map = new Map();
  if (config && Array.isArray(config.sections)) {
    for (const sec of config.sections) for (const f of sec.fields || []) map.set(f.name, f.value);
  } else if (config && typeof config === 'object') {
    for (const k of Object.keys(config)) map.set(k, config[k]);
  }
  return {
    get(name) {
      if (!map.has(name)) throw new Error('config.json has no constant ' + name);
      return map.get(name);
    },
    has(name) { return map.has(name); },
    size: map.size,
  };
}

// -- small numeric helpers --------------------------------------------------------------------

/** mulberry32: a small seeded PRNG. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const isF = (v) => typeof v === 'number' && isFinite(v);
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const sigmoid = (z) => 1 / (1 + Math.exp(-z));

function median(arr) {
  const v = arr.filter(isF).sort((p, q) => p - q);
  if (!v.length) return NaN;
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

function runsOf(mask, n) {
  const out = [];
  let s = -1;
  for (let i = 0; i <= n; i++) {
    const on = i < n && mask[i];
    if (on && s < 0) s = i;
    if (!on && s >= 0) { out.push([s, i]); s = -1; }
  }
  return out;
}

// -- world simulation -------------------------------------------------------------------------

/**
 * A seeded world of 22 agents and a raw ball. Agents steer to a target with a speed cap.
 * The default target is a 4-3-3 block that shifts with the ball, plus a slow wander.
 */
function createWorld(env, seed) {
  const { L, W, fps } = env;
  const rng = mulberry32(seed);
  const team = new Uint8Array(NS).fill(255);
  const agents = [];
  const rows = [
    ['gk', 0, 0], ['def', -14, -19], ['def', -15, -6.5], ['def', -15, 6.5], ['def', -14, 19],
    ['mid', -3, -14], ['mid', -4, 0], ['mid', -3, 14], ['fwd', 9, -17], ['fwd', 10, 0], ['fwd', 9, 17],
  ];
  for (const [t, slots] of [[0, H_SLOTS], [1, A_SLOTS]]) {
    slots.forEach((slot, k) => {
      team[slot] = t;
      const [role, ox, oy] = rows[k];
      agents.push({
        slot, team: t, role, ox: t === 0 ? ox : -ox, oy: t === 0 ? oy : -oy,
        x: NaN, y: NaN, vx: 0, vy: 0, visible: true,
        w1: 0.35 + rng() * 0.5, w2: 0.3 + rng() * 0.5, p1: rng() * 6.283, p2: rng() * 6.283,
        a1: 1.6 + rng() * 1.6, a2: 1.2 + rng() * 1.4, override: null,
      });
    });
  }
  const world = {
    env, rng, team, agents, t: 0,
    ball: { x: L / 2, y: W / 2, vx: 0, vy: 0, vis: true, attach: -1, dirX: 1, dirY: 0, off: 0.6, hold: false },
    cap: () => 3.4,
    shift: { hx: 0, ax: 0 },
    bySlot: new Map(),
  };
  for (const a of agents) world.bySlot.set(a.slot, a);

  world.blockTarget = function blockTarget(a, t) {
    const b = world.ball;
    const bx = isF(b.x) ? b.x : L / 2;
    const by = isF(b.y) ? b.y : W / 2;
    const cx = clamp(bx * 0.62 + L * 0.19, 28, L - 28);
    const cy = W / 2 + 0.3 * (by - W / 2);
    const ts = t / fps;
    let x, y;
    if (a.role === 'gk') {
      x = a.team === 0 ? 5 : L - 5;
      y = W / 2 + 0.12 * (by - W / 2);
    } else {
      x = cx + (a.team === 0 ? -2 : 2) + a.ox + (a.team === 0 ? world.shift.hx : world.shift.ax);
      y = cy + a.oy;
    }
    x += a.a1 * Math.sin(a.w1 * ts + a.p1);
    y += a.a2 * Math.cos(a.w2 * ts + a.p2);
    return { x: clamp(x, 1, L - 1), y: clamp(y, 1, W - 1) };
  };

  world.place = function place() {
    for (const a of agents) {
      const p = world.blockTarget(a, 0);
      a.x = p.x;
      a.y = p.y;
    }
  };

  world.stepPlayers = function stepPlayers(t) {
    const dt = 1 / fps;
    for (const a of agents) {
      let tgt = a.override ? a.override(t, a, world) : null;
      if (!tgt) tgt = world.blockTarget(a, t);
      const cap = tgt.cap != null ? tgt.cap : world.cap(t, a);
      if (tgt.teleport) { a.x = tgt.x; a.y = tgt.y; a.vx = 0; a.vy = 0; continue; }
      const dx = tgt.x - a.x, dy = tgt.y - a.y;
      const dist = Math.hypot(dx, dy);
      const want = Math.min(cap, dist * (tgt.gain || 1.3));
      const dvx = dist > 1e-6 ? (dx / dist) * want : 0;
      const dvy = dist > 1e-6 ? (dy / dist) * want : 0;
      const k = Math.min(1, (tgt.accel || 3.2) * dt);
      a.vx += (dvx - a.vx) * k;
      a.vy += (dvy - a.vy) * k;
      a.x += a.vx * dt;
      a.y += a.vy * dt;
    }
  };

  world.stepBall = function stepBall(t) {
    const b = world.ball;
    const dt = 1 / fps;
    if (b.attach >= 0) {
      const a = world.bySlot.get(b.attach);
      const sp = Math.hypot(a.vx, a.vy);
      if (sp > 0.4) { b.dirX = a.vx / sp; b.dirY = a.vy / sp; }
      const off = b.hold ? 0.35 : b.off + 0.22 * Math.sin((2 * Math.PI * t) / 11);
      const nx = a.x + b.dirX * off, ny = a.y + b.dirY * off;
      b.vx = (nx - b.x) * fps;
      b.vy = (ny - b.y) * fps;
      b.x = nx;
      b.y = ny;
      return;
    }
    const s = Math.hypot(b.vx, b.vy);
    if (s > 0) {
      const s2 = Math.max(0, s - env.friction * dt);
      b.vx *= s2 / s;
      b.vy *= s2 / s;
    }
    b.x += b.vx * dt;
    b.y += b.vy * dt;
  };

  /** Kick from where the ball is, to a point, at a speed. */
  world.kickTo = function kickTo(x, y, speed) {
    const b = world.ball;
    const dx = x - b.x, dy = y - b.y;
    const d = Math.hypot(dx, dy) || 1;
    b.attach = -1;
    b.hold = false;
    b.vx = (dx / d) * speed;
    b.vy = (dy / d) * speed;
  };
  /** Put a player at a point before the first frame. */
  world.put = function put(slot, x, y) {
    const a = world.bySlot.get(slot);
    a.x = x;
    a.y = y;
  };
  world.receive = function receive(slot, hold) {
    world.ball.attach = slot;
    world.ball.hold = !!hold;
  };
  return world;
}

/** Run a world for n frames. `script(t, world)` acts before each frame. Returns raw arrays. */
function runWorld(world, n, script, init) {
  const px = new Float32Array(n * NS).fill(NaN);
  const py = new Float32Array(n * NS).fill(NaN);
  const rx = new Float32Array(n), ry = new Float32Array(n);
  const rvis = new Uint8Array(n);
  world.place();
  if (init) init(world);
  for (let t = 0; t < n; t++) {
    world.t = t;
    script(t, world);
    world.stepBall(t);
    world.stepPlayers(t);
    for (const a of world.agents) {
      if (!a.visible) continue;
      px[t * NS + a.slot] = a.x;
      py[t * NS + a.slot] = a.y;
    }
    rx[t] = world.ball.x;
    ry[t] = world.ball.y;
    rvis[t] = world.ball.vis ? 1 : 0;
  }
  return { n, px, py, team: world.team, rx, ry, rvis };
}

// -- ball smoothing and kinematics ------------------------------------------------------------

/** Gaussian smoothing inside visible runs: the ATD provider's symmetric kernel. */
function smoothRuns(x, vis, n, sigma) {
  const out = new Float32Array(n).fill(NaN);
  const R = Math.ceil(3 * sigma);
  const w = new Float32Array(2 * R + 1);
  for (let k = -R; k <= R; k++) w[k + R] = Math.exp(-(k * k) / (2 * sigma * sigma));
  for (const [s, e] of runsOf(vis, n)) {
    for (let t = s; t < e; t++) {
      let sum = 0, ws = 0;
      for (let k = -R; k <= R; k++) {
        const u = t + k;
        let v;
        // beyond the run ends, continue the raw track at its end velocity (no edge shrink)
        if (u < s) v = e - s > 1 ? x[s] + (x[s] - x[s + 1]) * (s - u) : x[s];
        else if (u >= e) v = e - s > 1 ? x[e - 1] + (x[e - 1] - x[e - 2]) * (u - e + 1) : x[e - 1];
        else v = x[u];
        sum += v * w[k + R];
        ws += w[k + R];
      }
      out[t] = sum / ws;
    }
  }
  return out;
}

/** Velocity by central difference inside visible runs, one-sided at run ends (tracks.py). */
function kinematics(x, y, vis, n, fps) {
  const vx = new Float32Array(n).fill(NaN), vy = new Float32Array(n).fill(NaN);
  const speed = new Float32Array(n).fill(NaN);
  const runId = new Int32Array(n).fill(-1);
  let id = 0;
  for (const [s, e] of runsOf(vis, n)) {
    for (let t = s; t < e; t++) {
      runId[t] = id;
      const a = t > s ? t - 1 : t, b = t < e - 1 ? t + 1 : t;
      if (b === a) continue;
      vx[t] = ((x[b] - x[a]) * fps) / (b - a);
      vy[t] = ((y[b] - y[a]) * fps) / (b - a);
      speed[t] = Math.hypot(vx[t], vy[t]);
    }
    id++;
  }
  return { vx, vy, speed, runId };
}

/** Ball arrays from raw positions: smoothing, visibility, kinematics. */
function ballFrom(raw, env, smooth) {
  const n = raw.n;
  const vis = new Uint8Array(n);
  for (let t = 0; t < n; t++) vis[t] = raw.rvis[t];
  const x = smooth ? smoothRuns(raw.rx, vis, n, env.sigma) : Float32Array.from(raw.rx, (v, t) => (vis[t] ? v : NaN));
  const y = smooth ? smoothRuns(raw.ry, vis, n, env.sigma) : Float32Array.from(raw.ry, (v, t) => (vis[t] ? v : NaN));
  const k = kinematics(x, y, vis, n, env.fps);
  return Object.assign({ x, y, vis, rawX: Float32Array.from(x), rawY: Float32Array.from(y), rawVis: Uint8Array.from(vis),
    teleport: new Uint8Array(n), airborne: new Uint8Array(n) }, k);
}

// -- per-frame features (engine/tracks.py frame_features, the parts the scenes need) ----------

function features(sc, env) {
  const { n, px, py, team } = sc;
  const b = sc.ball;
  const nearH = new Int16Array(n).fill(-1), nearA = new Int16Array(n).fill(-1);
  const dH = new Float32Array(n).fill(Infinity), dA = new Float32Array(n).fill(Infinity);
  const relH = new Float32Array(n).fill(NaN), relA = new Float32Array(n).fill(NaN);
  const meanvAll = new Float32Array(n).fill(NaN), meanvNear = new Float32Array(n).fill(NaN);
  const sep = new Float32Array(n).fill(NaN);
  const ps = new Float32Array(n * NS).fill(NaN);
  const pvx = new Float32Array(n * NS).fill(NaN), pvy = new Float32Array(n * NS).fill(NaN);
  for (let t = 0; t < n; t++) {
    for (let s = 0; s < NS; s++) {
      const i = t * NS + s;
      if (!isF(px[i])) continue;
      const a = t > 0 && isF(px[i - NS]) ? t - 1 : t;
      const c = t < n - 1 && isF(px[i + NS]) ? t + 1 : t;
      if (a === c) continue;
      pvx[i] = ((px[c * NS + s] - px[a * NS + s]) * env.fps) / (c - a);
      pvy[i] = ((py[c * NS + s] - py[a * NS + s]) * env.fps) / (c - a);
      ps[i] = Math.hypot(pvx[i], pvy[i]);
    }
  }
  const nNear = env.cfg.get('PURSUIT_N_NEAREST');
  const dists = [];
  for (let t = 0; t < n; t++) {
    let sum = 0, cnt = 0, sxH = 0, cH = 0, sxA = 0, cA = 0;
    dists.length = 0;
    for (let s = 0; s < NS; s++) {
      const i = t * NS + s;
      if (!isF(px[i])) continue;
      if (isF(ps[i])) { sum += ps[i]; cnt++; }
      if (team[s] === 0) { sxH += px[i]; cH++; } else if (team[s] === 1) { sxA += px[i]; cA++; }
      if (b.vis[t] && isF(b.x[t])) {
        const d = Math.hypot(px[i] - b.x[t], py[i] - b.y[t]);
        dists.push([d, ps[i]]);
        if (team[s] === 0 && d < dH[t]) { dH[t] = d; nearH[t] = s; }
        if (team[s] === 1 && d < dA[t]) { dA[t] = d; nearA[t] = s; }
      }
    }
    meanvAll[t] = cnt ? sum / cnt : NaN;
    if (cH && cA) sep[t] = Math.abs(sxH / cH - sxA / cA);
    if (dists.length >= nNear) {
      dists.sort((p, q) => p[0] - q[0]);
      let m = 0;
      for (let k = 0; k < nNear; k++) m += isF(dists[k][1]) ? dists[k][1] : 0;
      meanvNear[t] = m / nNear;
    }
    for (const [slot, out] of [[nearH[t], relH], [nearA[t], relA]]) {
      if (slot < 0 || !isF(b.vx[t])) continue;
      const i = t * NS + slot;
      if (isF(pvx[i])) out[t] = Math.hypot(b.vx[t] - pvx[i], b.vy[t] - pvy[i]);
    }
  }
  return { nearH, nearA, dH, dA, relH, relA, meanvAll, meanvNear, sep, ps, pvx };
}

// -- detector ports ---------------------------------------------------------------------------

/** |v(t + h) - v(t - h)|, NaN unless both ends lie in one visible run (events.impulse). */
function impulse(b, n, h) {
  const J = new Float32Array(n).fill(NaN);
  for (let t = h; t < n - h; t++) {
    const r0 = b.runId[t - h], r1 = b.runId[t + h];
    if (r0 < 0 || r0 !== r1) continue;
    J[t] = Math.hypot(b.vx[t + h] - b.vx[t - h], b.vy[t + h] - b.vy[t - h]);
  }
  return J;
}

/** Event confidence (events.confidence). */
function confidence(d1, margin, cfg) {
  const floor = cfg.get('EVENT_CONF_FACTOR_FLOOR');
  const rT = cfg.get('R_TOUCH'), cm = cfg.get('CONTESTED_MARGIN_M');
  const factor = (v, full, fl) => {
    if (!isF(v)) return full > fl ? 1 : floor;
    const pos = (v - full) / (fl - full);
    return clamp(1 - (1 - floor) * pos, floor, 1);
  };
  return cfg.get('EVENT_CONF_MAX') * factor(d1, 0, rT) * factor(margin, rT, cm);
}

/**
 * Bump contacts (events._bump_contacts) on alive frames. `shape` is the team forward velocity or
 * null. With `shape`, a maximum with nobody within R_TOUCH is a reach contact (C18) when the
 * nearest player is within R_REACH_M and the shape response of that player's team confirms it.
 * Without `shape` such a maximum has no row. A reach contact has `reach: true` and d1 above R_TOUCH.
 */
function bumpContacts(sc, env, alive, shape) {
  const { cfg } = env;
  const h = cfg.get('CONTACT_HALF_WINDOW');
  const dv = cfg.get('DV_CONTACT');
  const rT = cfg.get('R_TOUCH'), cm = cfg.get('CONTESTED_MARGIN_M'), rf = cfg.get('REDIRECT_FRACTION');
  const { n, ball: b, f } = sc;
  const J = sc.J;
  const out = [];
  const reachM = cfgOr(cfg, 'R_REACH_M');
  const dead = new Int32Array(n + 1);
  for (let t = 0; t < n; t++) dead[t + 1] = dead[t] + (alive[t] ? 0 : 1);
  const centres = [];
  for (let c = 0; c < n; c++) {
    if (!isF(J[c]) || J[c] < dv || !alive[c]) continue;
    const lo = Math.max(0, c - h), hi = Math.min(n, c + h + 1);
    let best = lo;
    for (let u = lo; u < hi; u++) if ((isF(J[u]) ? J[u] : -Infinity) > (isF(J[best]) ? J[best] : -Infinity)) best = u;
    if (best === c) centres.push(c);
  }
  // K1: the kink contact point of each maximum, when the bundle ships CONTACT_POINT "kink_min"
  const points = contactPoint(cfg) === 'kink_min' ? kinkPoints(b, centres, cfg) : null;
  centres.forEach((c, index) => {
    const lo = Math.max(0, c - h), hi = Math.min(n, c + h + 1);
    let atH = lo, atA = lo;
    for (let u = lo; u < hi; u++) {
      if (f.dH[u] < f.dH[atH]) atH = u;
      if (f.dA[u] < f.dA[atA]) atA = u;
    }
    // `win` is the C8 window distance of each team, `kink` the fitted point and its team distances
    const win = { H: f.dH[atH], A: f.dA[atA] };
    let kink = null;
    let d = win;
    if (points) {
      kink = Object.assign({}, points[index], kinkTeamDistances(sc, points[index], cfg));
      d = { H: Math.min(win.H, kink.H), A: Math.min(win.A, kink.A) };
    }
    const teamL = d.H <= d.A ? 'H' : 'A';
    const other = teamL === 'H' ? 'A' : 'H';
    const d1 = d[teamL], d2 = d[other];
    let reach = false;
    if (!(d1 <= rT)) {
      // C18: nobody within R_TOUCH. Keep the maximum only for the nearest team, within R_REACH_M, with a shape response.
      reach = !!shape && d1 <= reachM && shapeConfirms(shape, c, teamL, dead, cfg);
      if (!reach) { out.push({ frame: c, kind: 'none', team: null, near: teamL, J: J[c], d1, win, kink, reason: 'nobody within the touch radius' }); return; }
    }
    if (d2 <= rT && d2 - d1 < cm) { out.push({ frame: c, kind: 'contested', team: null, J: J[c], d1, win, kink, conf: confidence(d1, Infinity, cfg) }); return; }
    const ds = b.speed[c + h] - b.speed[c - h];
    const split = rf * J[c];
    const kind = ds >= split ? 'kick' : ds <= -split ? 'receive' : 'redirect';
    out.push({ frame: c, kind, team: teamL, J: J[c], d1, win, kink, ds, reach, conf: confidence(d1, d2 - d1, cfg) });
  });
  return out;
}

/**
 * The contact-point rule of the bundle. A bundle built before K1 has no CONTACT_POINT field, and
 * its engine used the window rule, so the scenes use "window" for it.
 */
function contactPoint(cfg) {
  return cfg.has('CONTACT_POINT') ? cfg.get('CONTACT_POINT') : 'window';
}

/** erf by Abramowitz and Stegun 7.1.26 (error under 1.5e-7). The engine uses math.erf. */
function erf(v) {
  const a = Math.abs(v);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return v < 0 ? -y : y;
}

/** G(u) = u Phi(u / sigma) + sigma phi(u / sigma): a unit velocity step seen through the kernel (events.gauss_ramp). */
function gaussRamp(u, sigma) {
  const z = u / sigma;
  return u * 0.5 * (1 + erf(z / Math.SQRT2)) + (sigma * Math.exp(-0.5 * z * z)) / Math.sqrt(2 * Math.PI);
}

/**
 * Kink contact points (events.kink_points): {x, y, tau} for each impulse maximum, NaN when the
 * fit has too few frames. The model is a straight path with one velocity change at tau, seen
 * through a Gaussian kernel: x(t) = x_c + v1 (t - tau) + (v2 - v1) G(t - tau). For each sigma of
 * KINK_SIGMAS and each tau of the KINK_TAU_HALF and KINK_TAU_STEP grid, the fit solves the three
 * normal equations of each axis and keeps the first (sigma, tau) with the smallest residual.
 */
function kinkPoints(b, centres, cfg) {
  const half = cfgOr(cfg, 'KINK_HALF_WINDOW'), side = cfgOr(cfg, 'KINK_SIDE_FRAMES'), minPts = cfgOr(cfg, 'KINK_MIN_POINTS');
  const sigmas = cfgOr(cfg, 'KINK_SIGMAS'), step = cfgOr(cfg, 'KINK_TAU_STEP');
  const steps = Math.round(cfgOr(cfg, 'KINK_TAU_HALF') / step);
  const n = b.x.length;
  return centres.map((c, i) => {
    const none = { x: NaN, y: NaN, tau: NaN };
    let left = c - half, right = c + half;
    if (i > 0) left = Math.max(left, Math.floor((centres[i - 1] + c) / 2));
    if (i + 1 < centres.length) right = Math.min(right, Math.floor((centres[i + 1] + c) / 2));
    left = Math.max(0, left);
    right = Math.min(n - 1, right);
    if (b.runId[c] < 0 || right < left) return none;
    const idx = [];
    for (let t = left; t <= right; t++) if (b.runId[t] === b.runId[c]) idx.push(t);
    if (idx.length < minPts || idx[0] >= c - side || idx[idx.length - 1] <= c + side) return none;
    // relative to the ball at c, as the engine does, so the sums stay small
    const ox = b.x[c], oy = b.y[c];
    let best = Infinity, out = none;
    for (const sigma of sigmas) {
      for (let k = -steps; k <= steps; k++) {
        const shift = k * step;
        const us = idx.map((t) => t - c - shift);
        const gs = us.map((u) => gaussRamp(u, sigma));
        let a00 = 0, a01 = 0, a02 = 0, a11 = 0, a12 = 0, a22 = 0;
        us.forEach((u, j) => { const g = gs[j]; a00 += 1; a01 += u; a02 += g; a11 += u * u; a12 += u * g; a22 += g * g; });
        const c00 = a11 * a22 - a12 * a12, c01 = a02 * a12 - a01 * a22, c02 = a01 * a12 - a02 * a11;
        const c11 = a00 * a22 - a02 * a02, c12 = a01 * a02 - a00 * a12, c22 = a00 * a11 - a01 * a01;
        const det = a00 * c00 + a01 * c01 + a02 * c02;
        if (!(Math.abs(det) > 1e-9 * Math.max(1, Math.abs(a00 * a11 * a22)))) continue;
        let resid = 0;
        const heads = [];
        for (const [arr, o] of [[b.x, ox], [b.y, oy]]) {
          let r0 = 0, r1 = 0, r2 = 0;
          idx.forEach((t, j) => { const v = arr[t] - o; r0 += v; r1 += v * us[j]; r2 += v * gs[j]; });
          const b0 = (c00 * r0 + c01 * r1 + c02 * r2) / det;
          const b1 = (c01 * r0 + c11 * r1 + c12 * r2) / det;
          const b2 = (c02 * r0 + c12 * r1 + c22 * r2) / det;
          idx.forEach((t, j) => { const e = arr[t] - o - (b0 + b1 * us[j] + b2 * gs[j]); resid += e * e; });
          heads.push(b0);
        }
        if (resid < best) { best = resid; out = { x: ox + heads[0], y: oy + heads[1], tau: c + shift }; }
      }
    }
    return out;
  });
}

/** {H, A}: nearest distance of each team to a kink point on the frames round(tau) +- KINK_PLAYER_HALF (events.kink_team_distances). */
function kinkTeamDistances(sc, p, cfg) {
  const out = { H: Infinity, A: Infinity };
  if (!isF(p.x) || !isF(p.tau)) return out;
  const reach = cfgOr(cfg, 'KINK_PLAYER_HALF');
  const centre = Math.round(p.tau);
  for (let t = Math.max(0, centre - reach); t <= Math.min(sc.n - 1, centre + reach); t++) {
    for (let s = 0; s < NS; s++) {
      const x = sc.px[t * NS + s];
      if (!isF(x) || (sc.team[s] !== 0 && sc.team[s] !== 1)) continue;
      const d = Math.hypot(x - p.x, sc.py[t * NS + s] - p.y);
      const key = sc.team[s] === 0 ? 'H' : 'A';
      if (d < out[key]) out[key] = d;
    }
  }
  return out;
}

/** Control rows (events._control_rows). */
function controlRows(sc, env, alive) {
  const { cfg } = env;
  const { n, ball: b, f } = sc;
  const maxV = cfg.get('CONTROL_MAX_SPEED'), rC = cfg.get('R_CONTROL'), rT = cfg.get('R_TOUCH');
  const cm = cfg.get('CONTESTED_MARGIN_M'), need = cfg.get('CONTROL_FRAMES');
  const key = new Int8Array(n).fill(-1);
  for (let t = 0; t < n; t++) {
    const d1 = Math.min(f.dH[t], f.dA[t]), d2 = Math.max(f.dH[t], f.dA[t]);
    const contested = d2 <= rT && d2 - d1 < cm;
    const ok = b.vis[t] && isF(b.speed[t]) && b.speed[t] < maxV && d1 <= rC && !contested && alive[t];
    if (ok) key[t] = f.dH[t] <= f.dA[t] ? 0 : 1;
  }
  sc.controlKey = key;
  const rows = [];
  for (const code of [0, 1]) {
    for (const [s, e] of runsOf(Array.from(key, (k) => k === code), n)) {
      if (e - s < need) continue;
      const own = code === 0 ? f.dH[s] : f.dA[s], oth = code === 0 ? f.dA[s] : f.dH[s];
      rows.push({ frame: s, end: e - 1, kind: 'control', team: code === 0 ? 'H' : 'A', note: 'slow ball', conf: confidence(own, oth - own, cfg) });
    }
  }
  return rows.sort((p, q) => p.frame - q.frame);
}

/** Dribble rows (events._dribble_rows). */
function dribbleRows(sc, env, alive) {
  const { cfg } = env;
  const { n, ball: b, f } = sc;
  const rD = cfg.get('R_DRIBBLE'), need = cfg.get('DRIBBLE_FRAMES'), rel = cfg.get('DRIBBLE_REL_SPEED');
  const maxV = cfg.get('CONTROL_MAX_SPEED'), rT = cfg.get('R_TOUCH'), cm = cfg.get('CONTESTED_MARGIN_M');
  const key = new Int16Array(n).fill(-1);
  for (let t = 0; t < n; t++) {
    const home = f.dH[t] <= f.dA[t];
    const d1 = home ? f.dH[t] : f.dA[t];
    const slot = home ? f.nearH[t] : f.nearA[t];
    if (b.vis[t] && d1 <= rD && slot >= 0 && alive[t]) key[t] = slot;
  }
  sc.dribbleKey = key;
  const rows = [];
  let s = 0;
  while (s < n) {
    if (key[s] < 0) { s++; continue; }
    let e = s;
    while (e < n && key[e] === key[s]) e++;
    if (e - s >= need) {
      const home = f.dH[s] <= f.dA[s];
      const relRun = [], spRun = [];
      for (let t = s; t < e; t++) { relRun.push(home ? f.relH[t] : f.relA[t]); spRun.push(b.speed[t]); }
      const mRel = median(relRun), mSp = median(spRun);
      if (mRel < rel && mSp > maxV) {
        const own = home ? f.dH[s] : f.dA[s], oth = home ? f.dA[s] : f.dH[s];
        const contested = oth <= rT && oth - own < cm;
        rows.push({ frame: s, end: e - 1, kind: 'dribble', team: contested ? null : home ? 'H' : 'A', slot: key[s],
          medRel: mRel, medSpeed: mSp, conf: confidence(own, contested ? Infinity : oth - own, cfg) });
      }
    }
    s = e;
  }
  return rows;
}

/**
 * Engine defaults of the C15 constants (engine/config.py). A bundle built before C15 has no such
 * field in `config.json`: `cfgOr` then falls back to the default, so the old bundle still renders.
 */
const C15_DEFAULTS = {
  REDIRECT_SHAPE_CONFIRM: true, SHAPE_LAG_FRAMES: 25, SHAPE_CONFIRM_MS: 1.45, SHAPE_HALF_FRAMES: 6, SHAPE_MIN_PLAYERS: 6,
};

/** Engine defaults of the C18 constants (engine/config.py), with the same fallback for an old bundle. */
const C18_DEFAULTS = { REACH_SHAPE_CONFIRM: true, R_REACH_M: 3.0 };

/** Engine defaults of the knee onset constants (engine/config.py, COMPONENT_KNEE.md variant 2). */
const KNEE_DEFAULTS = {
  ONSET_KNEE: true, DELTA_KNEE: -15, KNEE_LOOKBACK_S: 4.0, KNEE_SMOOTH_S: 0.5,
  KNEE_PRE_FROM_S: 4.0, KNEE_PRE_TO_S: 2.0, KNEE_LEVEL_FRAC: 0.5, KNEE_GAP_CLAMP: true,
};

/** Engine defaults of the K1 kink fit (engine/config.py, calibration/K1_contact_point.md). */
const KINK_DEFAULTS = {
  KINK_HALF_WINDOW: 30, KINK_SIGMAS: [1, 3, 5, 7, 9.5, 12], KINK_TAU_HALF: 6, KINK_TAU_STEP: 0.5,
  KINK_MIN_POINTS: 12, KINK_SIDE_FRAMES: 3, KINK_PLAYER_HALF: 2,
};

/** Engine default of a C15, C18, knee or kink-fit constant, or undefined for any other name. */
export function engineDefault(name) {
  if (name in KINK_DEFAULTS) return KINK_DEFAULTS[name];
  if (name in KNEE_DEFAULTS) return KNEE_DEFAULTS[name];
  return name in C18_DEFAULTS ? C18_DEFAULTS[name] : C15_DEFAULTS[name];
}

function cfgOr(cfg, name) {
  return cfg.has(name) ? cfg.get(name) : engineDefault(name);
}

/** NaN-aware centred mean of `values` over [t - half, t + half] (tracks.centred_nanmean). */
function centredNanmean(values, half) {
  const n = values.length;
  const csum = new Float64Array(n + 1), ccount = new Int32Array(n + 1);
  for (let t = 0; t < n; t++) {
    const ok = isF(values[t]);
    csum[t + 1] = csum[t] + (ok ? values[t] : 0);
    ccount[t + 1] = ccount[t] + (ok ? 1 : 0);
  }
  const out = new Float64Array(n).fill(NaN);
  for (let t = 0; t < n; t++) {
    const lo = Math.max(0, t - half), hi = Math.min(n, t + half + 1);
    const count = ccount[hi] - ccount[lo];
    if (count > 0) out[t] = (csum[hi] - csum[lo]) / count;
  }
  return out;
}

/**
 * Forward velocity of each team, m/s (tracks.team_forward_velocity, C15): the mean x velocity of
 * the visible players of the team, signed towards the goal it attacks, then a centred mean over
 * +-SHAPE_HALF_FRAMES. NaN where fewer than SHAPE_MIN_PLAYERS players of the team are visible.
 * The scene worlds are one block: H defends x = 0 and attacks +x, A defends x = L.
 */
function teamForwardVelocity(sc, env) {
  const { cfg } = env;
  const { n, f, team } = sc;
  const minPlayers = cfgOr(cfg, 'SHAPE_MIN_PLAYERS');
  const half = cfgOr(cfg, 'SHAPE_HALF_FRAMES');
  const out = {};
  for (const [code, key, sign] of [[0, 'H', 1], [1, 'A', -1]]) {
    const mean = new Float64Array(n).fill(NaN);
    for (let t = 0; t < n; t++) {
      let sum = 0, count = 0;
      for (let s = 0; s < NS; s++) {
        if (team[s] !== code) continue;
        const v = f.pvx[t * NS + s];
        if (isF(v)) { sum += v; count++; }
      }
      if (count > 0 && count >= minPlayers) mean[t] = sign * (sum / count);
    }
    out[key] = centredNanmean(mean, half);
    // P10-A: the centered mean must preserve undefined player-count frames.
    for (let t = 0; t < n; t++) if (!isF(mean[t])) out[key][t] = NaN;
  }
  return out;
}

/** R_team - R_other at `frame`, with R_T = vf_T(frame + lag) - vf_T(frame); NaN when undefined (events.shape_response). */
function shapeResponse(shape, frame, team, lag) {
  const other = team === 'H' ? 'A' : 'H';
  const n = shape[team].length, end = frame + lag;
  if (!(frame >= 0 && frame < n && end >= 0 && end < n)) return NaN;
  return (shape[team][end] - shape[team][frame]) - (shape[other][end] - shape[other][frame]);
}

/** True when the C15 shape response confirms a redirect of `team` at `frame` (events.shape_confirms). `dead` counts dead frames before t, with a leading 0. */
function shapeConfirms(shape, frame, team, dead, cfg) {
  if (!shape) return false;
  const lag = cfgOr(cfg, 'SHAPE_LAG_FRAMES');
  const end = frame + lag;
  if (end >= dead.length - 1 || dead[end + 1] - dead[frame] !== 0) return false;
  const value = shapeResponse(shape, frame, team, lag);
  return isF(value) && value >= cfgOr(cfg, 'SHAPE_CONFIRM_MS');
}

/**
 * Redirect kinds (events._redirect_kinds), then the event rows of the bump detector. A redirect by
 * the team without the ball is a hand-over (a `control` row) when the next contact is by that team
 * (C8) or when the team shape confirms it (C15). Otherwise it has no row. `shape` null is the C8 rule.
 */
function contactEvents(contacts, controls, alive, shape, cfg) {
  const n = alive.length;
  const dead = new Int32Array(n + 1);
  for (let t = 0; t < n; t++) dead[t + 1] = dead[t] + (alive[t] ? 0 : 1);
  const clean = (a, b2) => dead[b2 + 1] - dead[a] === 0;
  const stream = contacts.filter((c) => c.kind !== 'contested' && c.kind !== 'none').map((c) => [c.frame, 0, c.team, c.kind])
    .concat(controls.map((r) => [r.frame, 1, r.team, 'control']));
  stream.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
  const kinds = new Map();
  let owner = null, last = null;
  stream.forEach(([frame, , team, kind], i) => {
    if (last !== null && !clean(last, frame)) owner = null;
    last = frame;
    if (kind !== 'redirect') { owner = team; return; }
    if (owner === team) { kinds.set(frame, 'touch'); return; }
    if (i + 1 < stream.length && stream[i + 1][2] === team && clean(frame, stream[i + 1][0])) {
      kinds.set(frame, 'control');
      owner = team;
    } else if (shapeConfirms(shape, frame, team, dead, cfg)) {
      kinds.set(frame, 'control');
      owner = team;
    }
  });
  const rows = [];
  for (const c of contacts) {
    if (c.kind === 'contested') rows.push({ frame: c.frame, kind: 'contested', team: null, note: 'contested', J: c.J });
    else if (c.kind === 'kick') rows.push({ frame: c.frame, kind: 'release', team: c.team, note: 'kick', J: c.J });
    else if (c.kind === 'receive') rows.push({ frame: c.frame, kind: 'control', team: c.team, note: 'receive', J: c.J });
    else if (c.kind === 'redirect' && kinds.has(c.frame)) rows.push({ frame: c.frame, kind: kinds.get(c.frame), team: c.team, note: 'redirect', J: c.J });
  }
  return rows;
}

/** Teleport rule (tracks._teleports) and the minimum run (BALL_MIN_RUN). */
function cleanRaw(rx, ry, rvis, n, maxJump, minRun) {
  const step = new Float32Array(n).fill(Infinity);
  for (let t = 1; t < n; t++) step[t] = Math.hypot(rx[t] - rx[t - 1], ry[t] - ry[t - 1]);
  const rejected = new Uint8Array(n);
  let resume = 0;
  for (let t0 = 1; t0 < n; t0++) {
    if (!(rvis[t0] && rvis[t0 - 1] && step[t0] > maxJump) || t0 < resume) continue;
    const lastOk = t0 - 1;
    let t = t0;
    while (t < n && rvis[t]) {
      const back = Math.hypot(rx[t] - rx[lastOk], ry[t] - ry[lastOk]) <= maxJump * (t - lastOk);
      if (step[t] > maxJump && !back) { rejected[t] = 1; t++; } else break;
    }
    resume = t + 1;
  }
  const vis = new Uint8Array(n);
  for (let t = 0; t < n; t++) vis[t] = rvis[t] && !rejected[t] ? 1 : 0;
  const short = [];
  for (const [s, e] of runsOf(vis, n)) {
    if (e - s < minRun) { for (let t = s; t < e; t++) vis[t] = 0; short.push([s, e]); }
  }
  return { vis, rejected, step, short };
}

/** Signed speed[t + k] - speed[t] inside one run (tracks.velocity_step). */
function velocityStep(b, n, k) {
  const out = new Float32Array(n).fill(NaN);
  for (let t = 0; t + k < n; t++) {
    if (b.runId[t] < 0 || b.runId[t] !== b.runId[t + k]) continue;
    out[t] = b.speed[t + k] - b.speed[t];
  }
  return out;
}

/** Restart candidates, kick and reappear (deadball.restarts) with the players test and chain. */
function restartCandidates(sc, env, barriers) {
  const { cfg, fps } = env;
  const { n, ball: b, f } = sc;
  const k = cfg.get('RESTART_K');
  const dvR = cfg.get('DV_RESTART'), rP = cfg.get('RESTART_PLAYER_M');
  const reV = cfg.get('REAPPEAR_SPEED'), reM = cfg.get('REAPPEAR_TEAM_M');
  const vStop = cfg.get('V_STOP');
  const confirmN = Math.round(cfg.get('RESTART_CONFIRM_S') * fps);
  const chainN = cfg.get('RESTART_CHAIN_FRAMES');
  const dv9 = velocityStep(b, n, k);
  sc.dv9 = dv9;
  const nearestIn = (lo, hi) => {
    let best = Infinity, team = null;
    for (let t = lo; t < hi; t++) {
      const d = Math.min(f.dH[t], f.dA[t]);
      if (d < best) { best = d; team = f.dH[t] <= f.dA[t] ? 'H' : 'A'; }
    }
    return [best, team];
  };
  const rows = [];
  for (const [s, e] of runsOf(Array.from(dv9, (v) => isF(v) && v > dvR), n)) {
    // _rise_start: largest one-frame increase in [s, e + k), then walk back while rising
    const last = Math.min(n - 1, e + k);
    let frame = s, bestA = -Infinity;
    for (let t = s; t < last; t++) {
      const acc = b.speed[t + 1] - b.speed[t];
      if (isF(acc) && acc > bestA) { bestA = acc; frame = t; }
    }
    while (frame - 1 >= s && isF(b.speed[frame - 1]) && b.speed[frame - 1] < b.speed[frame]) frame--;
    const [d, team] = nearestIn(Math.max(0, frame - k), Math.min(n, frame + k + 1));
    if (d > rP) continue;
    rows.push({ frame, kind: 'kick', team, d });
  }
  for (const [s] of runsOf(b.vis, n)) {
    if (s === 0 || !(b.speed[s] > reV)) continue;
    const [d, team] = nearestIn(s, Math.min(n, s + k + 1));
    rows.push({ frame: s, kind: 'reappear', team: d <= reM ? team : null, d, speed: b.speed[s] });
  }
  rows.sort((p, q) => p.frame - q.frame || (p.kind < q.kind ? -1 : 1));
  for (const r of rows) {
    r.players = false;
    r.movingAt = -1;
    for (let t = r.frame; t < Math.min(n, r.frame + confirmN + 1); t++) {
      if (f.meanvAll[t] > vStop) { r.players = true; r.movingAt = t; break; }
    }
    if (r.movingAt < 0) for (let t = r.frame; t < n; t++) if (f.meanvAll[t] > vStop) { r.movingAt = t; break; }
  }
  // chain_confirmed: a failed candidate is confirmed by the next players-confirmed candidate
  // within RESTART_CHAIN_FRAMES with no barrier (out or pursuit_stop onset) between
  let following = null;
  for (let i = rows.length - 1; i >= 0; i--) {
    const r = rows[i];
    r.confirmed = r.players;
    r.chainedTo = null;
    if (!r.players && following && following.frame - r.frame <= chainN && chainN > 0) {
      const blocked = barriers.some((x) => x > r.frame && x <= following.frame);
      if (!blocked) { r.confirmed = true; r.chainedTo = following.frame; }
    }
    if (r.players) following = r;
  }
  return rows;
}

/** Distance along a unit heading to the first pitch line (deadball._boundary_distance). */
function boundaryDistance(x, y, ux, uy, L, W) {
  let best = Infinity;
  for (const [pos, step, lo, hi] of [[x, ux, 0, L], [y, uy, 0, W]]) {
    if (step > 0) best = Math.min(best, (hi - pos) / step);
    else if (step < 0) best = Math.min(best, (lo - pos) / step);
  }
  return best;
}

/** Collective-stop runs that last COLLECTIVE_STOP_S (deadball._stop_runs). */
function stopRuns(sc, env) {
  const { cfg, fps } = env;
  const vStop = cfg.get('V_STOP'), sepDead = cfg.get('SEP_DEAD');
  const need = Math.round(cfg.get('COLLECTIVE_STOP_S') * fps);
  const mask = Array.from(sc.f.meanvAll, (v, t) => v < vStop && sc.f.sep[t] > sepDead);
  return runsOf(mask, sc.n).filter(([s, e]) => e - s >= need);
}

function confirmedBy(runs, onset, horizon) {
  for (const [s, e] of runs) if (s <= onset + horizon && e > onset) return s;
  return -1;
}

/** Out onsets: observed crossings and the missing-ball extrapolation (deadball._out_onsets). */
function outOnsets(sc, env) {
  const { cfg, fps, L, W } = env;
  const { n, ball: b } = sc;
  const margin = cfg.get('PITCH_MARGIN_M');
  const decel = cfg.get('FRICTION_DECEL');
  const horizon = Math.round(cfg.get('ONSET_CONFIRM_S') * fps);
  const runs = stopRuns(sc, env);
  const rows = [];
  const beyond = (t, m) => b.vis[t] && (b.x[t] < -m || b.x[t] > L + m || b.y[t] < -m || b.y[t] > W + m);
  for (let t = 0; t < n; t++) {
    if (!beyond(t, margin) || (t > 0 && beyond(t - 1, margin))) continue;
    let frame = t;
    while (frame - 1 >= 0 && beyond(frame - 1, 0)) frame--;
    // C10: a reappearance beyond a touchline that moves inward is a throw-in
    const inward = (() => {
      if (frame > 0 && b.vis[frame - 1]) return false;
      if (!(b.x[frame] >= 0 && b.x[frame] <= L) || (b.y[frame] >= 0 && b.y[frame] <= W)) return false;
      let last = Math.min(frame + cfg.get('OUT_INWARD_FRAMES'), n - 1);
      while (last > frame && !b.vis[last]) last--;
      const dy = last > frame ? b.y[last] - b.y[frame] : b.vy[frame];
      return b.y[frame] < 0 ? dy > 0 : dy < 0;
    })();
    if (inward) { sc.throwIn = frame; continue; }
    rows.push({ frame, mechanism: 'out', how: 'observed', conf: cfg.get('CONF_OUT_OBSERVED'), x: b.x[frame], y: b.y[frame] });
  }
  for (const [s, e] of runsOf(b.vis, n)) {
    const last = e - 1;
    if (e >= n) continue;
    const next = runsOf(b.vis, n).find(([s2]) => s2 > last);
    const nextVisible = next ? next[0] : n;
    if (!(b.x[last] >= 0 && b.x[last] <= L && b.y[last] >= 0 && b.y[last] <= W)) continue;
    const sp = b.speed[last];
    if (!isF(sp) || sp < cfg.get('HEADING_MIN_SPEED')) continue;
    const ux = b.vx[last] / sp, uy = b.vy[last] / sp;
    const dist = boundaryDistance(b.x[last], b.y[last], ux, uy, L, W);
    if (dist > cfg.get('OUT_MAX_RANGE_M')) continue;
    let onset, conf, how, cross;
    if (sp * sp >= 2 * decel * dist) {
      const secs = (sp - Math.sqrt(sp * sp - 2 * decel * dist)) / decel;
      onset = last + Math.round(secs * fps);
      conf = cfg.get('CONF_OUT_EXTRAPOLATED');
      how = 'extrapolated';
      cross = [b.x[last] + ux * dist, b.y[last] + uy * dist];
    } else {
      onset = last + cfg.get('DELTA_OUT_LAG');
      conf = cfg.get('CONF_OUT_LAG');
      how = 'lag';
      const roll = (sp * sp) / (2 * decel);
      cross = [b.x[last] + ux * dist, b.y[last] + uy * dist];
      sc.rollStop = [b.x[last] + ux * roll, b.y[last] + uy * roll, dist - roll];
    }
    if (onset >= nextVisible) continue;
    const stopAt = confirmedBy(runs, onset, horizon);
    sc.lastSeen = { frame: last, x: b.x[last], y: b.y[last], ux, uy, speed: sp, dist, how, cross, onset, stopAt };
    if (stopAt >= 0) rows.push({ frame: onset, mechanism: 'out', how, conf, x: cross[0], y: cross[1], last, stopAt });
  }
  return rows;
}

/** Pursuit-stop onsets (deadball._pursuit_onsets). */
function pursuitOnsets(sc, env) {
  const { cfg, fps } = env;
  const { n, ball: b, f } = sc;
  const minV = cfg.get('PURSUIT_MIN_BALL_SPEED'), stopV = cfg.get('PURSUIT_STOP_SPEED');
  const back = cfg.get('PURSUIT_KICK_LOOKBACK'), cm = cfg.get('CONTESTED_MARGIN_M'), rT = cfg.get('R_TOUCH');
  const chase = Math.round(cfg.get('COLLECTIVE_STOP_S') * fps);
  const horizon = Math.round(cfg.get('ONSET_CONFIRM_S') * fps);
  const runs = stopRuns(sc, env);
  const rows = [];
  for (const [start, stop] of runsOf(Array.from(b.speed, (v, t) => b.vis[t] && v > minV), n)) {
    const lo = Math.max(0, start - back);
    let seen = start > 0;
    for (let t = lo; t < start; t++) if (!b.vis[t]) seen = false;
    if (!seen) continue;
    let peak = stop - 1;
    for (let t = start; t < stop - 1; t++) if (b.speed[t + 1] < b.speed[t]) { peak = t; break; }
    let mx = -Infinity;
    for (let t = lo; t <= start; t++) if (isF(f.meanvNear[t])) mx = Math.max(mx, f.meanvNear[t]);
    if (mx < stopV) continue;
    let mh = 0, ma = 0, c = 0;
    for (let t = lo; t <= start; t++) { mh += f.dH[t]; ma += f.dA[t]; c++; }
    mh /= c; ma /= c;
    if (Math.abs(mh - ma) < cm) continue;
    const kick = mh <= ma ? 'H' : 'A';
    const dK = kick === 'H' ? f.dH : f.dA;
    let drop = -1;
    for (let t = start; t < Math.min(n, start + horizon); t++) if (f.meanvNear[t] < stopV) { drop = t; break; }
    if (drop < 0) continue;
    let descent = drop;
    while (descent - 1 >= 0 && isF(f.meanvNear[descent - 1]) && f.meanvNear[descent - 1] > f.meanvNear[descent]) descent--;
    const onset = Math.max(peak, descent);
    if (!(dK[onset] > dK[start])) continue;
    let reach = false;
    for (let t = onset; t < Math.min(n, onset + chase); t++) if (dK[t] <= rT) reach = true;
    if (reach) continue;
    const stopAt = confirmedBy(runs, onset, horizon);
    sc.pursuit = { start, peak, drop, descent, onset, kick, stopAt };
    if (stopAt >= 0) rows.push({ frame: onset, mechanism: 'pursuit_stop', conf: cfg.get('CONF_PURSUIT'), x: b.x[onset], y: b.y[onset], kick });
  }
  return rows;
}

// -- decoder (engine/decoder.py) --------------------------------------------------------------

function nearMask(frames, n, width) {
  const m = new Uint8Array(n);
  for (const fr of frames) for (let t = Math.max(0, fr - width); t <= Math.min(n - 1, fr + width); t++) m[t] = 1;
  return m;
}

/** switch_costs: cost[t, i, j] flattened to t * 9 + i * 3 + j. */
function switchCosts(n, cfg, teamEvents, onsets, restarts) {
  const c = new Float64Array(n * 9);
  const cS = cfg.get('C_SWITCH'), cSE = cfg.get('C_SWITCH_EVENT'), wS = cfg.get('SWITCH_EVENT_WINDOW');
  const nearAev = nearMask(teamEvents.A, n, wS), nearHev = nearMask(teamEvents.H, n, wS);
  const nearOn = nearMask(onsets, n, cfg.get('DEAD_EVENT_WINDOW'));
  const nearRs = nearMask(restarts.map((r) => r[0]), n, cfg.get('ALIVE_EVENT_WINDOW'));
  const ownH = nearMask(restarts.filter((r) => r[1] === 'H').map((r) => r[0]), n, cfg.get('ALIVE_EVENT_WINDOW'));
  const ownA = nearMask(restarts.filter((r) => r[1] === 'A').map((r) => r[0]), n, cfg.get('ALIVE_EVENT_WINDOW'));
  const bonus = cfg.get('RESTART_TEAM_BONUS');
  for (let t = 0; t < n; t++) {
    const o = t * 9;
    c[o + 0 * 3 + 1] = nearAev[t] ? cSE : cS;
    c[o + 1 * 3 + 0] = nearHev[t] ? cSE : cS;
    const toDead = nearOn[t] ? cfg.get('C_DEAD_EVENT') : cfg.get('C_DEAD');
    c[o + 0 * 3 + 2] = toDead;
    c[o + 1 * 3 + 2] = toDead;
    const toAlive = nearRs[t] ? cfg.get('C_ALIVE_EVENT') : cfg.get('C_ALIVE');
    c[o + 2 * 3 + 0] = toAlive - (ownH[t] ? bonus : 0);
    c[o + 2 * 3 + 1] = toAlive - (ownA[t] ? bonus : 0);
  }
  return c;
}

const ORDER = [[0, 1, 2], [1, 0, 2], [2, 0, 1]];

/** Exact Viterbi with the engine's tie order: stay first, then H, A, D. */
export function viterbi(logE, costs, n) {
  const path = new Int8Array(n);
  if (!n) return path;
  const ptr = new Int8Array(n * 3);
  let score = [logE[0], logE[1], logE[2]];
  for (let t = 1; t < n; t++) {
    const next = [0, 0, 0];
    for (let j = 0; j < 3; j++) {
      let best = -Infinity, arg = ORDER[j][0];
      for (const p of ORDER[j]) {
        const v = score[p] - costs[t * 9 + p * 3 + j];
        if (v > best) { best = v; arg = p; }
      }
      next[j] = best + logE[t * 3 + j];
      ptr[t * 3 + j] = arg;
    }
    score = next;
  }
  let last = 0;
  for (let j = 1; j < 3; j++) if (score[j] > score[last]) last = j;
  path[n - 1] = last;
  for (let t = n - 1; t > 0; t--) path[t - 1] = ptr[t * 3 + path[t]];
  return path;
}

/** The owner event kinds of each POSSESSION_RULE (possession.RULE_KINDS). */
const RULE_KINDS = { release: ['release'], release_or_control: ['release', 'control', 'dribble'] };

/**
 * Owner walk, emission and decoding for one scene. Owner events are the rows with a team whose
 * kind is an owner event of POSSESSION_RULE. The emission follows the owner (P_OWNER) and the
 * stoppage (P_DEAD_STOPPAGE).
 */
function decodeScene(sc, env, opts) {
  const { cfg } = env;
  const n = sc.n;
  const kinds = RULE_KINDS[cfg.has('POSSESSION_RULE') ? cfg.get('POSSESSION_RULE') : 'release_or_control'];
  const ownerEv = sc.events.filter((e) => e.team && kinds.indexOf(e.kind) >= 0 && !sc.stoppage[e.frame]);
  const owner = new Int8Array(n).fill(-1);
  let cur = opts.initialOwner == null ? -1 : opts.initialOwner;
  let k = 0;
  for (let t = 0; t < n; t++) {
    while (k < ownerEv.length && ownerEv[k].frame <= t) { cur = ownerEv[k].team === 'H' ? 0 : 1; k++; }
    if (sc.stoppage[t]) cur = opts.keepOwnerInD ? cur : cur;
    owner[t] = cur;
  }
  const pO = cfg.get('P_OWNER'), pD = cfg.get('P_DEAD_STOPPAGE'), floor = cfg.get('P_FLOOR');
  const logE = new Float64Array(n * 3);
  for (let t = 0; t < n; t++) {
    const d = sc.stoppage[t] ? pD : 1 - pD;
    const pH = owner[t] === 0 ? pO : owner[t] === 1 ? 1 - pO : 0.5;
    logE[t * 3] = Math.log(Math.max(floor, (1 - d) * pH));
    logE[t * 3 + 1] = Math.log(Math.max(floor, (1 - d) * (1 - pH)));
    logE[t * 3 + 2] = Math.log(Math.max(floor, d));
  }
  const teamEvents = { H: [], A: [] };
  for (const e of sc.events) if (e.team && !sc.stoppage[e.frame]) teamEvents[e.team].push(e.frame);
  const costs = switchCosts(n, cfg, teamEvents, sc.onsets.map((o) => o.frame), sc.exits.map((r) => [r.exit, r.team]));
  const path = viterbi(logE, costs, n);
  sc.owner = owner;
  sc.logE = logE;
  sc.costs = costs;
  sc.labels = Uint8Array.from(path);
}

// -- scene assembly ---------------------------------------------------------------------------

function baseScene(env, raw, ball) {
  const sc = { env, n: raw.n, px: raw.px, py: raw.py, team: raw.team, ball,
    events: [], contacts: [], onsets: [], restarts: [], exits: [], beats: [], marks: [] };
  sc.f = features(sc, env);
  sc.J = impulse(ball, sc.n, env.cfg.get('CONTACT_HALF_WINDOW'));
  sc.stoppage = new Uint8Array(sc.n);
  sc.pDead = new Float32Array(sc.n);
  return sc;
}

function detectEvents(sc, env, alive) {
  // C15 and C18: REDIRECT_SHAPE_CONFIRM and REACH_SHAPE_CONFIRM each switch one use of the team shape
  const redirectOn = cfgOr(env.cfg, 'REDIRECT_SHAPE_CONFIRM'), reachOn = cfgOr(env.cfg, 'REACH_SHAPE_CONFIRM');
  const shape = redirectOn || reachOn ? teamForwardVelocity(sc, env) : null;
  const contacts = bumpContacts(sc, env, alive, reachOn ? shape : null);
  const controls = controlRows(sc, env, alive);
  const dribbles = dribbleRows(sc, env, alive);
  sc.shape = redirectOn ? shape : null;
  sc.alive = alive;
  const rows = contactEvents(contacts, controls, alive, sc.shape, env.cfg).concat(controls, dribbles);
  rows.sort((p, q) => p.frame - q.frame);
  sc.contacts = contacts;
  sc.controls = controls;
  sc.dribbles = dribbles;
  sc.events = rows;
}

/** Fill the stoppage from [start, stop) ranges and fade p_dead in and out over a few frames. */
function setStoppage(sc, ranges) {
  for (const [s, e] of ranges) for (let t = Math.max(0, s); t < Math.min(sc.n, e); t++) sc.stoppage[t] = 1;
  const k = 0.18;
  let v = sc.stoppage[0];
  for (let t = 0; t < sc.n; t++) {
    v += (sc.stoppage[t] - v) * k;
    sc.pDead[t] = Math.max(sc.pDead[t], v);
  }
}

function eventLabel(e) {
  const team = e.team ? ' ' + e.team : '';
  if (e.kind === 'control' && e.note === 'receive') return 'reception' + team;
  if (e.kind === 'control' && e.note === 'slow ball') return 'slow-ball control' + team;
  if (e.kind === 'release') return 'kick' + team;
  if (e.kind === 'control') return 'owner change to' + (team || ' nobody');
  return e.kind + team;
}

/** Finish a scene: the generic frameAt, the caption and the timeline. */
function finish(sc, meta) {
  const { env } = sc;
  const n = sc.n;
  const span = env.cfg.get('CONTACT_HALF_WINDOW');
  sc.id = meta.id;
  sc.title = meta.title;
  sc.fps = env.fps;
  sc.keyFrame = meta.keyFrame != null ? meta.keyFrame : Math.floor(n * 0.6);
  sc.decorate = meta.decorate || null;
  sc.beats = meta.beats || [];
  sc.posX = meta.posX || null;
  // follow camera: the centred mean of the visible ball over +-CAM frames, else the players
  const CAM = 30;
  const camX = new Float32Array(n), camY = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let sx = 0, sy = 0, c = 0;
    for (let k = Math.max(0, i - CAM); k <= Math.min(n - 1, i + CAM); k++) {
      if (sc.ball.vis[k] && isF(sc.ball.x[k])) { sx += sc.ball.x[k]; sy += sc.ball.y[k]; c++; }
    }
    if (!c) {
      for (let s2 = 0; s2 < NS; s2++) { const x = sc.px[i * NS + s2]; if (isF(x)) { sx += x; sy += sc.py[i * NS + s2]; c++; } }
    }
    camX[i] = c ? sx / c : env.L / 2;
    camY[i] = c ? sy / c : env.W / 2;
    if (meta.camAt) {
      const p = meta.camAt(i, camX[i], camY[i]);
      camX[i] = p[0];
      camY[i] = p[1];
    }
  }
  const zoom = meta.zoom || 1.7;
  const ev = sc.events.map((e) => ({ frame: e.frame, kind: e.kind, team: e.team, x: sc.ball.x[e.frame], y: sc.ball.y[e.frame], label: eventLabel(e) }));
  sc.timeline = {
    labels: sc.labels,
    marks: ev.map((e) => ({ i: e.frame, color: 'ev-' + e.kind, label: e.label }))
      .concat(sc.onsets.map((o) => ({ i: o.frame, color: 'onset', label: 'onset ' + o.mechanism })))
      .concat(sc.exits.map((r) => ({ i: r.frame, color: 'restart', label: 'restart ' + r.kind }))),
  };
  sc.caption = function caption(i) {
    let text = meta.caption || '';
    for (const b of sc.beats) if (i >= b.from) text = b.text;
    return text;
  };
  sc.frameAt = function frameAt(i) {
    i = clamp(i | 0, 0, n - 1);
    const px = sc.posX ? sc.posX(i, 'x') : sc.px.subarray(i * NS, (i + 1) * NS);
    const py = sc.posX ? sc.posX(i, 'y') : sc.py.subarray(i * NS, (i + 1) * NS);
    const tn = Math.min(TRAIL, i);
    const tpx = new Float32Array(tn * NS), tpy = new Float32Array(tn * NS);
    for (let k = 0; k < tn; k++) {
      const u = i - tn + k;
      tpx.set(sc.posX ? sc.posX(u, 'x') : sc.px.subarray(u * NS, (u + 1) * NS), k * NS);
      tpy.set(sc.posX ? sc.posX(u, 'y') : sc.py.subarray(u * NS, (u + 1) * NS), k * NS);
    }
    const bn = Math.min(BALL_TRAIL, i);
    const btx = new Float32Array(bn), bty = new Float32Array(bn);
    for (let k = 0; k < bn; k++) {
      const u = i - bn + k;
      btx[k] = sc.ball.vis[u] ? sc.ball.x[u] : NaN;
      bty[k] = sc.ball.vis[u] ? sc.ball.y[u] : NaN;
    }
    const b = sc.ball;
    const owner = sc.owner ? sc.owner[i] : -1;
    const ownerSlot = owner === 0 ? sc.f.nearH[i] : owner === 1 ? sc.f.nearA[i] : -1;
    const lab = sc.labels ? sc.labels[i] : 3;
    const fs = {
      frame: i, nSlots: NS, px, py, team: sc.team,
      trail: { n: tn, px: tpx, py: tpy },
      ball: { x: b.x[i], y: b.y[i], visible: !!b.vis[i], rawX: b.rawX[i], rawY: b.rawY[i], rawVisible: !!b.rawVis[i],
        airborne: !!b.airborne[i], ghost: false, teleport: !!b.teleport[i], speed: b.speed[i] },
      ballTrail: { n: bn, x: btx, y: bty },
      glow: ownerSlot >= 0 && lab < 2 ? { slot: ownerSlot, team: owner, alpha: 1 } : null,
      pDead: sc.pDead[i],
      stoppage: !!sc.stoppage[i],
      nearest: [{ slot: sc.f.nearH[i], d: sc.f.dH[i] }, { slot: sc.f.nearA[i], d: sc.f.dA[i] }],
      events: ev.filter((e) => Math.abs(i - e.frame) <= span).map((e) => Object.assign({ dt: i - e.frame }, e)),
      onsets: sc.onsets.filter((o) => i - o.frame >= -2 && i - o.frame <= span * 2)
        .map((o) => ({ mechanism: o.mechanism, dt: i - o.frame, x: o.x, y: o.y, edge: o.edge })),
      restarts: sc.exits.filter((r) => i - r.frame >= 0 && i - r.frame <= span * 2)
        .map((r) => ({ kind: r.kind, team: r.team, dt: i - r.frame, x: b.x[r.frame], y: b.y[r.frame], confirmed: r.confirmed })),
      labels: { ours: 'HADX'[lab], gt: null, baseline: null },
      hud: { speed: b.vis[i] ? b.speed[i] : NaN, dH: sc.f.dH[i], dA: sc.f.dA[i], owner, pDead: sc.pDead[i] },
      overlays: [], insets: [],
      camera: { x: camX[i], y: camY[i], zoom },
    };
    if (sc.decorate) sc.decorate(i, fs);
    return fs;
  };
  return sc;
}

// -- shared decoration helpers ----------------------------------------------------------------

/** Velocity arrows v(t - h), v(t + h) and the impulse J at the ball. */
function impulseArrows(sc, i, fs, h) {
  const b = sc.ball;
  if (i - h < 0 || i + h >= sc.n || !isF(sc.J[i]) || sc.J[i] < 2 || !b.vis[i]) return;
  const k = 0.45;
  const x = b.x[i], y = b.y[i];
  fs.overlays.push({ type: 'path', pts: [x, y, x + b.vx[i - h] * k, y + b.vy[i - h] * k], color: 'muted', width: 2, top: true });
  fs.overlays.push({ type: 'path', pts: [x, y, x + b.vx[i + h] * k, y + b.vy[i + h] * k], color: 'text', width: 2, top: true });
  fs.overlays.push({ type: 'path', pts: [x + b.vx[i - h] * k, y + b.vy[i - h] * k, x + b.vx[i + h] * k, y + b.vy[i + h] * k],
    color: 'onset', width: 2.5, dash: [4, 3], top: true });
}

function fmt(v, d) {
  return isF(v) ? v.toFixed(d == null ? 1 : d) : 'n/a';
}

// -- the scenes -------------------------------------------------------------------------------

function sceneCleaning(env) {
  const { cfg } = env;
  const n = 160;
  const world = createWorld(env, 101);
  const passer = world.bySlot.get(6), recv = world.bySlot.get(9);
  passer.override = (t) => (t < 30 ? { x: 42, y: 30, cap: 2 } : null);
  recv.override = (t, a) => (t < 130 ? { x: 66, y: 36, cap: 3 } : null);
  const raw = runWorld(world, n, (t, w) => {
    if (t === 0) { w.ball.x = 42.6; w.ball.y = 30.2; w.ball.vx = 0; w.ball.vy = 0; w.receive(6, true); }
    if (t === 30) w.kickTo(66, 36, 11);
    if (t > 30 && w.ball.attach < 0 && Math.hypot(w.ball.x - recv.x, w.ball.y - recv.y) < 0.9) w.receive(9);
  }, (w) => { w.put(6, 42, 30); w.put(9, 66.5, 36.2); });
  const ball = ballFrom(raw, env, true);
  // inject the tracker faults into the raw track: a one-frame spike, a fast streak, a short run
  const maxJump = cfg.get('BALL_MAX_JUMP_M');
  const minRun = cfg.get('BALL_MIN_RUN');
  const rx = Float32Array.from(ball.x), ry = Float32Array.from(ball.y), rvis = Uint8Array.from(ball.vis);
  const spike = 45;
  rx[spike] += maxJump * 2.4; ry[spike] -= maxJump * 0.6;
  const streak = 92;
  for (let k = 0; k < 3; k++) { rx[streak + k] = rx[streak - 1] - (k + 1) * maxJump * 1.7; ry[streak + k] = ry[streak - 1] + (k + 1) * maxJump * 0.5; }
  const gapA = 108, gapB = gapA + 14;
  for (let t = gapA; t < gapB; t++) rvis[t] = 0;
  rvis[gapA + 5] = 1; rvis[gapA + 6] = 1;
  rx[gapA + 5] = 83; ry[gapA + 5] = 12; rx[gapA + 6] = 83.2; ry[gapA + 6] = 12.1;
  const cl = cleanRaw(rx, ry, rvis, n, maxJump, minRun);
  const clean = {
    x: Float32Array.from(rx, (v, t) => (cl.vis[t] ? v : NaN)), y: Float32Array.from(ry, (v, t) => (cl.vis[t] ? v : NaN)),
    vis: cl.vis, rawX: Float32Array.from(rx, (v, t) => (rvis[t] ? v : NaN)), rawY: Float32Array.from(ry, (v, t) => (rvis[t] ? v : NaN)),
    rawVis: rvis, teleport: cl.rejected, airborne: new Uint8Array(n),
  };
  Object.assign(clean, kinematics(clean.x, clean.y, clean.vis, n, env.fps));
  const sc = baseScene(env, raw, clean);
  detectEvents(sc, env, new Uint8Array(n).fill(1));
  decodeScene(sc, env, { initialOwner: 0 });
  const stepSeries = Float32Array.from(cl.step, (v, t) => (rvis[t] && rvis[t - 1] ? v : NaN));
  const faults = [];
  for (let t = 0; t < n; t++) if (cl.rejected[t]) faults.push({ i: t, kind: 'teleport' });
  for (const [s] of cl.short) faults.push({ i: s, kind: 'short' });
  const lastOk = (t) => { let u = t - 1; while (u > 0 && !cl.vis[u]) u--; return u; };
  return finish(sc, {
    id: 'cleaning', zoom: 1.8, title: 'Ball cleaning', keyFrame: streak + 2,
    beats: [
      { from: 0, text: 'The tracked ball, with three tracker faults. The cleaner checks every step from one frame to the next.' },
      { from: spike - 3, text: 'A one-frame spike: the step exceeds ' + maxJump + ' m and the next frame comes back, so only the spike is dropped.' },
      { from: streak - 3, text: 'A streak of impossible speed: every frame whose own step exceeds ' + maxJump + ' m is a teleport.' },
      { from: gapA, text: 'A two-frame detection after a gap: a visible run shorter than ' + minRun + ' frames is dropped.' },
      { from: gapB + 4, text: 'What is left is the cleaned ball. Every later step reads it.' },
    ],
    decorate(i, fs) {
      for (const fl of faults) {
        const age = i - fl.i;
        if (age < 0 || age > 30) continue;
        const a = 1 - age / 30;
        if (fl.kind === 'teleport') {
          const u = lastOk(fl.i);
          const jump = Math.hypot(rx[fl.i] - rx[fl.i - 1], ry[fl.i] - ry[fl.i - 1]);
          fs.overlays.push({ type: 'cross', x: rx[fl.i], y: ry[fl.i], color: 'teleport', alpha: a, size: 0.9, top: true });
          if (fl.i === spike || fl.i === streak) {
            fs.overlays.push({ type: 'line', x1: rx[u], y1: ry[u], x2: rx[fl.i], y2: ry[fl.i], color: 'teleport', dash: [3, 3], alpha: a,
              label: jump.toFixed(1) + ' m > ' + maxJump + ' m: teleport', top: true });
          }
        } else {
          fs.overlays.push({ type: 'circle', x: rx[fl.i], y: ry[fl.i], r: 1.6, color: 'teleport', dash: [2, 3], alpha: a,
            label: '2-frame run, under ' + minRun + ' frames', top: true });
        }
      }
      fs.insets.push({ type: 'spark', title: 'raw step per frame, m', n, cursor: i, yMax: maxJump * 4.5,
        series: [{ values: stepSeries, color: 'text', label: 'step' }],
        threshold: { value: maxJump, label: 'max step ' + maxJump + ' m', color: 'teleport' },
        markers: faults.map((fl) => ({ i: fl.i, label: fl.kind === 'teleport' ? '' : 'short run', color: 'teleport', dashed: fl.kind !== 'teleport' })),
        readout: 'step ' + fmt(stepSeries[i], 2) + ' m' });
      fs.insets.push({ type: 'checklist', title: 'cleaning rules', items: [
        { text: 'step > ' + maxJump + ' m from the last good frame: teleport', ok: i >= spike ? true : null },
        { text: 'frames rejected so far: ' + cl.rejected.reduce((p, v, t) => p + (v && t <= i ? 1 : 0), 0), ok: i >= streak ? true : null },
        { text: 'runs under ' + minRun + ' frames dropped', ok: i >= gapA + 5 ? true : null },
      ] });
    },
  });
}

function sceneContacts(env) {
  const { cfg } = env;
  const n = 190;
  const h = cfg.get('CONTACT_HALF_WINDOW');
  const dv = cfg.get('DV_CONTACT');
  const world = createWorld(env, 202);
  const kicker = world.bySlot.get(6), recv = world.bySlot.get(28), tapper = recv;
  const K = 30;
  const kickV = dv * 1.75;
  const from = [40, 40], to = [66, 28];
  const dist = Math.hypot(to[0] - from[0], to[1] - from[1]);
  const fr = cfg.get('FRICTION_DECEL');
  const tArr = (kickV - Math.sqrt(kickV * kickV - 2 * fr * dist)) / fr;
  const R = K + Math.round(tArr * env.fps);
  kicker.override = (t) => (t < K + 6 ? { x: from[0] - 0.5, y: from[1] + 0.2, cap: 1.2 } : null);
  recv.override = (t) => (t < R ? { x: to[0] + 0.3, y: to[1] - 0.1, cap: 3.6 } : t < 150 ? { x: 74, y: 22, cap: 2.2 } : null);
  let got = -1;
  const raw = runWorld(world, n, (t, w) => {
    if (t === 0) { w.ball.x = from[0]; w.ball.y = from[1]; w.ball.vx = 0; w.ball.vy = 0; w.receive(6, true); }
    if (t === K) w.kickTo(to[0], to[1], kickV);
    if (t > K && got < 0 && Math.hypot(w.ball.x - recv.x, w.ball.y - recv.y) < 0.8) { got = t; w.receive(28); w.ball.off = 0.55; }
    if (t === 150) w.kickTo(50, 30, dv * 0.45);
  }, (w) => { w.put(6, from[0] - 0.5, from[1] + 0.2); w.put(28, to[0] + 6, to[1] - 5); });
  void tapper;
  const ball = ballFrom(raw, env, true);
  const sc = baseScene(env, raw, ball);
  detectEvents(sc, env, new Uint8Array(n).fill(1));
  decodeScene(sc, env, { initialOwner: 0 });
  const J = sc.J;
  let jMax = 0;
  for (let t = 0; t < n; t++) if (isF(J[t])) jMax = Math.max(jMax, J[t]);
  const weak = [];
  for (let t = 1; t < n - 1; t++) {
    if (isF(J[t]) && J[t] < dv && J[t] > dv * 0.25 && J[t] >= J[t - 1] && J[t] > J[t + 1]) {
      if (!sc.contacts.some((c) => Math.abs(c.frame - t) < h * 2)) weak.push(t);
    }
  }
  const markers = sc.contacts.map((c) => ({ i: c.frame, v: c.J, color: 'onset', label: c.kind === 'kick' ? 'kick' : c.kind === 'receive' ? 'reception' : c.kind }))
    .concat(weak.map((t) => ({ i: t, v: J[t], color: 'muted', label: 'below the threshold', dashed: true })));
  const rel = sc.contacts.find((c) => c.kind === 'kick');
  const rec = sc.contacts.find((c) => c.kind === 'receive');
  return finish(sc, {
    id: 'contacts', zoom: 1.7, title: 'A contact at the peak of a bump', keyFrame: rec ? rec.frame + 2 : R,
    beats: [
      { from: 0, text: 'The provider smooths the ball, so a kick shows as a bump of acceleration about ' + (2 * h) + ' frames wide.' },
      { from: K - h, text: 'The change of ball velocity across ' + h + ' frames on each side grows across the kick.' },
      { from: rel ? rel.frame : K, text: 'Its peak passes ' + dv + ' m/s: a contact, placed at the centre of the bump. The ball sped up, so it is a kick.' },
      { from: rec ? rec.frame - h : R - h, text: 'The interception: the ball slows at the A player’s feet. A speed loss over half the velocity change is a reception, an owner event.' },
      { from: 145, text: 'A light touch: the velocity change peaks below ' + dv + ' m/s, so it is not a contact.' },
    ],
    decorate(i, fs) {
      impulseArrows(sc, i, fs, h);
      fs.insets.push({ type: 'spark', title: 'ball speed and velocity change, m/s', n, cursor: i, yMax: Math.max(jMax, kickV) * 1.12,
        series: [{ values: ball.speed, color: 'muted', label: 'speed', labelRow: 0 }, { values: J, color: 'onset', fill: true, label: 'velocity change over ±' + h + ' frames', labelX: 52, labelRow: 0 }],
        threshold: { value: dv, label: 'contact threshold ' + dv, color: 'text' },
        brackets: i - h >= 0 && i + h < n ? [{ i0: i - h, i1: i + h, color: 'muted', label: '±' + h }] : [],
        markers, readout: 'change ' + fmt(J[i], 1) + ' m/s' });
      const c = sc.contacts.filter((x) => i >= x.frame).pop();
      const items = c && i - c.frame < 50 ? [
        { text: 'peak change over ±' + h + ' frames: ' + fmt(c.J, 1) + ' m/s', ok: true },
        { text: 'nearest player ' + fmt(c.d1, 2) + ' m, touch radius ' + cfg.get('R_TOUCH') + ' m', ok: c.d1 <= cfg.get('R_TOUCH') },
        { text: 'speed ' + (c.ds >= 0 ? 'rose ' : 'fell ') + fmt(Math.abs(c.ds), 1) + ' m/s' + (Math.abs(c.ds) > cfg.get('REDIRECT_FRACTION') * c.J ? ', over half' : ', under half') + ': ' + (c.kind === 'kick' ? 'kick' : c.kind === 'receive' ? 'reception' : c.kind), ok: true },
      ] : [{ text: 'waiting for a peak above ' + dv + ' m/s', ok: null }];
      fs.insets.push({ type: 'checklist', title: 'contact test', items });
    },
  });
}

/**
 * The kink contact point (K1). H passes, and a player of A strikes the ball first time. The raw
 * ball turns at his foot, but the smoothed ball cuts the corner and passes about 2 m from him, and
 * close to a player of H who stands inside the turn. The window rule gives the contact to H. The
 * kink fit finds the corner, so the A player is nearest to the fitted contact point. Each team
 * keeps the smaller of its two distances, and the contact is a release by A. With a bundle built
 * before K1 (no CONTACT_POINT) the detector uses the window rule, and the scene says so.
 */
function sceneKink(env) {
  const { cfg } = env;
  const n = 140;
  const h = cfg.get('CONTACT_HALF_WINDOW');
  const rT = cfg.get('R_TOUCH');
  const world = createWorld(env, 707);
  const kicker = world.bySlot.get(6), toucher = world.bySlot.get(28), inside = world.bySlot.get(7);
  const K = 20;
  const from = [42, 40], to = [55, 36];
  const speedIn = 9, speedOut = 20;
  // the A player plays the ball on, 60 degrees to the left of its path
  const inDir = [(to[0] - from[0]) / Math.hypot(to[0] - from[0], to[1] - from[1]), (to[1] - from[1]) / Math.hypot(to[0] - from[0], to[1] - from[1])];
  const turn = -Math.PI / 3;
  const outDir = [inDir[0] * Math.cos(turn) - inDir[1] * Math.sin(turn), inDir[0] * Math.sin(turn) + inDir[1] * Math.cos(turn)];
  const away = [to[0] + outDir[0] * 30, to[1] + outDir[1] * 30];
  // the smoothed ball is displaced from the corner along v2 - v1; the H player stands beyond it
  const dvx = outDir[0] * speedOut - inDir[0] * speedIn, dvy = outDir[1] * speedOut - inDir[1] * speedIn;
  const dvn = Math.hypot(dvx, dvy);
  const cut = (dvn / env.fps) * env.sigma / Math.sqrt(2 * Math.PI);
  const insideAt = [to[0] + (dvx / dvn) * (cut + 1.0), to[1] + (dvy / dvn) * (cut + 1.0)];
  const standAt = [to[0] + inDir[0] * 0.3, to[1] + inDir[1] * 0.3];
  let R = -1;
  const hold = (a, t) => {
    const b = world.ball, sx = b.x, sy = b.y;
    b.x = to[0]; b.y = to[1];
    const p = world.blockTarget(a, t);
    b.x = sx; b.y = sy;
    return p;
  };
  for (const a of world.agents) {
    if (a === kicker || a === toucher || a === inside) continue;
    a.override = (t, ag) => {
      const p = hold(ag, t);
      // keep the rest of both blocks clear of the corner, so the scene has two candidates
      const dx = p.x - to[0], dy = p.y - to[1], d = Math.hypot(dx, dy);
      if (d < 7) return { x: to[0] + (dx / (d || 1)) * 7, y: to[1] + (dy / (d || 1)) * 7 };
      return p;
    };
  }
  kicker.override = (t) => (t < K + 6 ? { x: from[0] - 0.5, y: from[1] + 0.2, cap: 1.2 } : hold(kicker, t));
  toucher.override = () => ({ x: standAt[0], y: standAt[1], cap: 1.5 });
  inside.override = () => ({ x: insideAt[0], y: insideAt[1], cap: 1.5 });
  const raw = runWorld(world, n, (t, w) => {
    if (t === 0) { w.ball.x = from[0]; w.ball.y = from[1]; w.ball.vx = 0; w.ball.vy = 0; w.receive(6, true); }
    if (t === K) w.kickTo(to[0], to[1], speedIn);
    // the raw ball turns where it reaches `to`
    if (R < 0 && t > K && (w.ball.x - to[0]) * inDir[0] + (w.ball.y - to[1]) * inDir[1] >= 0) { R = t; w.kickTo(away[0], away[1], speedOut); }
  }, (w) => { w.put(6, from[0] - 0.5, from[1] + 0.2); w.put(28, standAt[0], standAt[1]); w.put(7, insideAt[0], insideAt[1]); });
  const ball = ballFrom(raw, env, true);
  const sc = baseScene(env, raw, ball);
  detectEvents(sc, env, new Uint8Array(n).fill(1));
  decodeScene(sc, env, { initialOwner: 0 });
  const c = sc.contacts.filter((x) => x.frame > K + h).sort((p, q) => Math.abs(p.frame - R) - Math.abs(q.frame - R))[0] || null;
  const cf = c ? c.frame : R;
  const kinkOn = !!(c && c.kink);
  // the fitted point is shown also when the bundle predates K1, but then it decides nothing
  let kp = c && c.kink;
  if (c && !kp) kp = Object.assign({}, kinkPoints(ball, [cf], cfg)[0], kinkTeamDistances(sc, kinkPoints(ball, [cf], cfg)[0], cfg));
  const win = c ? c.win : { H: NaN, A: NaN };
  const verdict = c && c.team ? (c.kind === 'kick' ? 'kick' : c.kind === 'receive' ? 'reception' : c.kind) + ' by ' + c.team
    : c && c.kind === 'contested' ? 'contested' : 'no contact';
  const owner = c && c.team === 'A' && c.kind !== 'redirect' ? 'The owner is now A.' : 'The owner stays H.';
  const result = (verdict === 'contested' ? 'The contact is contested. ' : verdict === 'no contact' ? 'The peak gives no contact. ' : 'The contact is a ' + verdict + '. ') + owner;
  const dH = Float32Array.from(sc.f.dH, (v) => (isF(v) ? Math.min(v, 6) : 6));
  const dA = Float32Array.from(sc.f.dA, (v) => (isF(v) ? Math.min(v, 6) : 6));
  const tau = kp && isF(kp.tau) ? kp.tau : cf;
  const beats = [
    { from: 0, text: 'H plays a pass. H owns the ball.' },
    { from: cf - h, text: 'A player of A strikes the ball first time. The true ball turns at the player\u2019s foot, but the smoothed ball cuts the corner.' },
    { from: cf, text: 'Over ±' + h + ' frames the smoothed ball passes ' + fmt(win.A, 1) + ' m from that player and ' + fmt(win.H, 1) + ' m from a player of H inside the turn.' +
      (win.H <= rT && win.H < win.A ? ' The nearest distance over the window alone gives the contact to H.' : '') },
    kinkOn
      ? { from: cf + 12, text: 'The rule engine fits a straight path with one change of velocity, seen through the same smoothing. Its corner, the fitted contact point, is ' + fmt(kp.A, 2) + ' m from the A player.' }
      : { from: cf + 12, text: 'This data bundle has no fitted contact point, so the nearest distance over the window alone decides the contact.' },
    { from: cf + 36, text: kinkOn
      ? 'Each team keeps the smaller of its two distances: H ' + fmt(Math.min(win.H, kp.H), 2) + ' m, A ' + fmt(Math.min(win.A, kp.A), 2) + ' m. ' + result
      : result },
  ];
  return finish(sc, {
    id: 'kink', zoom: 1.6, camAt: () => [to[0] + 2, to[1] - 2], keyFrame: cf + 36,
    title: 'Contact at the fitted contact point',
    beats,
    decorate(i, fs) {
      // the true path of the ball, which the tracker never shows
      const lo = Math.max(K, cf - 30), hi = Math.min(i, cf + 30, n - 1);
      if (hi > lo) {
        const pts = [];
        for (let t = lo; t <= hi; t++) pts.push(raw.rx[t], raw.ry[t]);
        fs.overlays.push({ type: 'path', pts, color: 'muted', width: 1.4, dash: [2, 3], label: i > cf + 6 ? 'true path of the ball' : '' });
      }
      if (c && i >= cf) {
        const fade = Math.max(0.45, 1 - (i - cf) / 90);
        fs.overlays.push({ type: 'circle', slot: 7, r: 1.4, color: 'h', alpha: fade, label: 'H: ' + fmt(win.H, 1) + ' m to the smoothed ball' });
        fs.overlays.push({ type: 'circle', slot: 28, r: 1.4, color: 'a', alpha: fade, label: 'A: ' + fmt(win.A, 1) + ' m' });
      }
      if (kp && isF(kp.x) && i >= cf + 12) {
        fs.overlays.push({ type: 'cross', x: kp.x, y: kp.y, color: 'onset', size: 0.6, top: true, label: 'fitted contact point' });
      }
      fs.insets.push({ type: 'spark', title: 'distance of each team to the smoothed ball, m', n, cursor: i, yMax: 6,
        series: [{ values: dH, color: 'h', label: 'H' }, { values: dA, color: 'a', label: 'A', labelX: 30 }],
        threshold: { value: rT, label: 'touch radius ' + rT, color: 'onset' },
        brackets: c && i >= cf ? [{ i0: cf - h, i1: cf + h, color: 'muted', label: '±' + h }] : [],
        markers: c ? [{ i: cf, color: 'onset', label: 'contact' }, { i: Math.round(tau), color: 'muted', label: 'contact time from the fit', dashed: true }] : [],
        readout: 'H ' + fmt(sc.f.dH[i], 1) + ' m, A ' + fmt(sc.f.dA[i], 1) + ' m' });
      const seen = c && i >= cf, fitted = c && i >= cf + 12, done = c && i >= cf + 36;
      fs.insets.push({ type: 'checklist', title: 'who played the ball', items: [
        { text: seen ? 'nearest over the window: H ' + fmt(win.H, 1) + ' m, A ' + fmt(win.A, 1) + ' m' : 'nearest over the window: not yet', ok: seen ? false : null },
        { text: fitted ? 'fitted contact point: H ' + fmt(kp.H, 1) + ' m, A ' + fmt(kp.A, 1) + ' m' : 'fitted contact point: not yet', ok: fitted ? kinkOn : null },
        { text: done ? 'contact: ' + verdict : 'contact: not yet decided', ok: done ? c.team === 'A' : null },
      ] });
    },
  });
}

/**
 * Redirect by the team without the ball (C15) and the reach contact (C18). One seed and one ball
 * path, four variants. In `shape` the players respond to the touch (A moves forward, H drops
 * back) and the shape confirms the hand-over. In `flat` the shapes stay as they were and the owner
 * stays H. `reach` and `reach-flat` are the same two cases with the toucher of A standing
 about 2.4 m from the ball, outside R_TOUCH and inside R_REACH_M: without the C18 clause the
 * contact has no row in either variant.
 */
function sceneRedirect(env, variant) {
  const { cfg } = env;
  const flat = variant === 'flat' || variant === 'reach-flat';
  const reach = variant === 'reach' || variant === 'reach-flat';
  const n = 150;
  const h = cfg.get('CONTACT_HALF_WINDOW');
  const lag = cfgOr(cfg, 'SHAPE_LAG_FRAMES'), need = cfgOr(cfg, 'SHAPE_CONFIRM_MS');
  const world = createWorld(env, 606);
  const kicker = world.bySlot.get(6), toucher = world.bySlot.get(28);
  const K = 25;
  const from = [44, 40], to = [59.6, 33.7], away = [50, 14];
  // the toucher stands where the smoothed ball passes. The reach variants move it sideways, at
  // right angles to the ball path, 75 % of the way from R_TOUCH to R_REACH_M (2.6 m with the
  // shipped constants, 2.4 m from the smoothed ball), so nobody is within R_TOUCH and the
  // toucher is in reach
  const rTouch = cfg.get('R_TOUCH'), rReach = cfgOr(cfg, 'R_REACH_M');
  const standOff = rTouch + 0.75 * (rReach - rTouch);
  const pathLen = Math.hypot(to[0] - from[0], to[1] - from[1]);
  const side = reach ? [-(to[1] - from[1]) / pathLen, (to[0] - from[0]) / pathLen] : [0, 0];
  const stand = [58.1 + side[0] * standOff, 33.1 + side[1] * standOff];
  const speedIn = 11, speedOut = 8;
  let R = -1;
  // every other player holds a block that does not follow the ball, so only the response moves them
  const hold = (a, t) => {
    const b = world.ball, sx = b.x, sy = b.y;
    b.x = 58; b.y = 34;
    const p = world.blockTarget(a, t);
    b.x = sx; b.y = sy;
    return p;
  };
  // the response: A (the toucher's team) moves towards x = 0, the goal it attacks, and H drops back
  const pushA = 3, pushH = 1.6;
  for (const a of world.agents) {
    if (a === kicker || a === toucher) continue;
    a.override = (t, ag) => {
      const p = hold(ag, t);
      // a block mate of the toucher stands clear of the ball path, so the touch has one toucher
      if (ag.slot === 29) p.y += 9;
      if (flat || R < 0 || t <= R || ag.role === 'gk') return p;
      const dx = ((t - R) / env.fps) * (ag.team === 1 ? pushA : pushH);
      return { x: p.x - dx, y: p.y, cap: 3.4 };
    };
  }
  kicker.override = (t) => (t < K + 6 ? { x: from[0] - 0.5, y: from[1] + 0.2, cap: 1.2 } : hold(kicker, t));
  toucher.override = () => ({ x: stand[0], y: stand[1], cap: 2 });
  const raw = runWorld(world, n, (t, w) => {
    if (t === 0) { w.ball.x = from[0]; w.ball.y = from[1]; w.ball.vx = 0; w.ball.vy = 0; w.receive(6, true); }
    if (t === K) w.kickTo(to[0], to[1], speedIn);
    // the raw ball turns where it passes `to`; the smoothed ball cuts the corner and passes the toucher
    if (R < 0 && t > K && (w.ball.x - to[0]) * (to[0] - from[0]) + (w.ball.y - to[1]) * (to[1] - from[1]) >= 0) { R = t; w.kickTo(away[0], away[1], speedOut); }
  }, (w) => { w.put(6, from[0] - 0.5, from[1] + 0.2); w.put(28, stand[0], stand[1]); });
  const ball = ballFrom(raw, env, true);
  const sc = baseScene(env, raw, ball);
  detectEvents(sc, env, new Uint8Array(n).fill(1));
  decodeScene(sc, env, { initialOwner: 0 });
  // a redirect, or the maximum that C8 drops for want of a player within R_TOUCH (kind none)
  const c = sc.contacts.find((x) => x.kind === 'redirect' || x.kind === 'none');
  const cf = c ? c.frame : R;
  const row = sc.events.find((e) => e.note === 'redirect');
  // the scene always shows the response, also when the bundle switches the clause off
  const shape = sc.shape || teamForwardVelocity(sc, env);
  const respond = (t) => (c && t >= cf && t <= cf + lag ? shapeResponse(shape, cf, 'A', t - cf) : NaN);
  const series = Float32Array.from({ length: n }, (_, t) => respond(t));
  const final = c ? shapeResponse(shape, cf, 'A', lag) : NaN;
  sc.redirectInfo = { frame: cf, response: final, row: row || null };
  const centroid = (t, code) => {
    let sx = 0, sy = 0, k = 0;
    for (let s2 = 0; s2 < NS; s2++) if (sc.team[s2] === code && isF(sc.px[t * NS + s2])) { sx += sc.px[t * NS + s2]; sy += sc.py[t * NS + s2]; k++; }
    return [sx / k, sy / k];
  };
  const nearest = c ? c.d1 : NaN;
  const missed = flat
    ? 'The teams did not move. The team-movement value is ' + fmt(final, 1) + ' m/s, under ' + need + ' m/s. The check fails, and H keeps the ball.'
    : 'A moved forward and H dropped back. The team-movement value is ' + fmt(final, 1) + ' m/s, at least ' + need + ' m/s. The check passes. The owner is now A.';
  const beats = reach ? [
    { from: 0, text: 'H plays a long ball. H owns the ball.' },
    { from: cf - h, text: 'A player of A heads the ball on. The tracker shows a ball in the air on the ground, so the nearest player is ' + fmt(nearest, 1) + ' m from the smoothed ball.' },
    { from: cf, text: 'Nobody is within the ' + rTouch + ' m touch radius, so the plain contact test finds nothing. The rule engine looks further: the nearest player is of A and within ' + rReach + ' m, so it runs the team-movement check.' },
    { from: cf + lag, text: flat ? missed : missed.replace('The owner is now A.', 'The peak is a distant touch of A. The owner is now A.') },
  ] : [
    { from: 0, text: 'H plays a pass. H owns the ball.' },
    { from: cf - h, text: 'A player of A turns the ball. The direction changes more than the speed, so the touch is a redirect.' },
    { from: cf, text: 'A redirect by the team without the ball needs proof. A makes no next contact, so the rule engine runs the team-movement check.' },
    { from: cf + lag, text: missed },
  ];
  return finish(sc, {
    id: 'redirect' + (reach ? '-reach' : '') + (flat ? '-flat' : ''), zoom: 1.2, camAt: () => [56, 34], keyFrame: cf + lag,
    title: (reach ? 'Distant touch, ' : 'Redirect, ') + (flat ? 'check fails' : 'check passes'),
    beats,
    decorate(i, fs) {
      impulseArrows(sc, i, fs, h);
      if (c && i >= cf) {
        const fade = Math.max(0.35, 1 - (i - cf) / 60);
        if (reach) {
          const bx = sc.ball.x[cf], by = sc.ball.y[cf];
          fs.overlays.push({ type: 'circle', x: bx, y: by, r: rTouch, color: 'onset', dash: [3, 3], alpha: fade });
          fs.overlays.push({ type: 'circle', x: bx, y: by, r: rReach, color: 'a', dash: [3, 3], label: 'touch radius ' + rTouch + ' m, reach ' + rReach + ' m', alpha: fade });
        }
        fs.overlays.push({ type: 'circle', slot: 28, r: 1.6, color: 'a', label: reach ? 'nearest, ' + fmt(nearest, 1) + ' m' : 'redirect by A', alpha: fade });
        const t = Math.min(i, cf + lag);
        for (const [code, col] of [[0, 'h'], [1, 'a']]) {
          const p0 = centroid(cf, code), p1 = centroid(t, code);
          fs.overlays.push({ type: 'path', pts: [p0[0], p0[1], p1[0], p1[1]], color: col, width: 3, top: true });
        }
      }
      fs.insets.push({ type: 'spark', title: 'team-movement check: forward speed of A minus H, m/s', n, cursor: i, yMax: Math.max(need * 2.4, isF(final) ? final * 1.15 : 0),
        series: [{ values: series, color: 'a', label: 'since the touch', fill: true }],
        threshold: { value: need, label: 'needed ' + need, color: 'onset' },
        brackets: c && i >= cf ? [{ i0: cf, i1: cf + lag, color: 'muted', label: lag + ' frames' }] : [],
        markers: c ? [{ i: cf, color: 'a', label: reach ? 'touch' : 'redirect' }] : [],
        readout: c && i >= cf ? 'response ' + fmt(respond(Math.min(i, cf + lag)), 1) + ' m/s' : '' });
      const seen = c && i >= cf;
      const done = c && i >= cf + lag;
      const response = { text: done ? 'team movement ' + fmt(final, 1) + ' vs ' + need + ' m/s' : 'team movement: waiting ' + lag + ' frames', ok: done ? !!row : null };
      const verdict = { text: done ? (row ? 'a contact of A: the owner is A' : 'ignored: the owner stays H') : 'not yet decided', ok: done ? !!row : null };
      fs.insets.push({ type: 'checklist', title: reach ? 'distant touch by A, owner H' : 'redirect by A, owner H', items: reach ? [
        { text: 'nearest player ' + fmt(nearest, 1) + ' m: beyond touch radius', ok: seen ? false : null },
        { text: 'within ' + rReach + ' m, and of A', ok: seen ? true : null },
        response, verdict,
      ] : [
        { text: 'speed change under half the peak: redirect', ok: seen ? true : null },
        { text: 'next contact by A: none', ok: seen ? false : null },
        response, verdict,
      ] });
    },
  });
}

function sceneDribble(env) {
  const { cfg } = env;
  const n = 210;
  const maxV = cfg.get('CONTROL_MAX_SPEED'), need = cfg.get('CONTROL_FRAMES');
  const rD = cfg.get('R_DRIBBLE'), dN = cfg.get('DRIBBLE_FRAMES'), relMax = cfg.get('DRIBBLE_REL_SPEED');
  const fr = cfg.get('FRICTION_DECEL');
  const world = createWorld(env, 303);
  const ctrl = world.bySlot.get(29), drib = world.bySlot.get(31);
  // a slow loose ball rolls to the A player: it falls under CONTROL_MAX_SPEED near his feet
  const v0 = maxV + fr * 1.6;
  const start = [48, 36];
  const travel = (v0 * v0 - (maxV * 0.5) ** 2) / (2 * fr);
  const meet = [start[0] + travel, start[1]];
  ctrl.override = (t) => (t < 95 ? { x: meet[0] + 0.7, y: meet[1] + 0.1, cap: t < 30 ? 1.5 : 0.6 } : null);
  const P = 95, Rv = 112;
  drib.override = (t) => {
    if (t < Rv) return { x: meet[0] - 9, y: meet[1] - 7, cap: 2 };
    return { x: 30, y: 24, cap: 4.4, accel: 2.2 };
  };
  const raw = runWorld(world, n, (t, w) => {
    if (t === 0) { w.ball.x = start[0]; w.ball.y = start[1]; w.ball.vx = v0; w.ball.vy = 0; }
    if (t === 72) w.receive(29, true);
    if (t === P) w.kickTo(meet[0] - 8.6, meet[1] - 6.8, 9);
    if (t === Rv) { w.receive(31); w.ball.off = 0.6; }
  }, (w) => { w.put(29, meet[0] + 4, meet[1] + 2); w.put(31, meet[0] - 9, meet[1] - 7); });
  const ball = ballFrom(raw, env, true);
  const sc = baseScene(env, raw, ball);
  detectEvents(sc, env, new Uint8Array(n).fill(1));
  decodeScene(sc, env, { initialOwner: 0 });
  const ctrlRow = sc.controls.find((r) => r.team === 'A');
  const dribRow = sc.dribbles[0];
  const runLen = (key, i, slot) => { let k = 0; for (let u = i; u >= 0 && (slot == null ? key[u] >= 0 : key[u] === slot); u--) k++; return k; };
  return finish(sc, {
    id: 'dribble', zoom: 2.0, title: 'Slow ball and dribble', keyFrame: dribRow ? dribRow.frame + dN : 150,
    beats: [
      { from: 0, text: 'A loose ball rolls slowly to an A player. No kick, no reception bump: nothing for the contact detector.' },
      { from: ctrlRow ? ctrlRow.frame - 6 : 60, text: 'Slow-ball control: within ' + cfg.get('R_CONTROL') + ' m, slower than ' + maxV + ' m/s, for ' + need + ' frames.' },
      { from: Rv, text: 'A teammate runs with the ball. The same player stays within ' + rD + ' m of a moving ball.' },
      { from: dribRow ? dribRow.frame + dN : 150, text: 'After ' + dN + ' frames, the medians over the run decide: a dribble, placed at the first frame of the run.' },
    ],
    decorate(i, fs) {
      const ck = sc.controlKey;
      if (ck[i] >= 0) {
        const slot = ck[i] === 0 ? sc.f.nearH[i] : sc.f.nearA[i];
        const k = runLen(ck, i, ck[i]);
        fs.overlays.push({ type: 'progress', slot, r: 1.9, frac: k / need, color: 'ev-control',
          label: 'control ' + Math.min(k, need) + '/' + need + '  ' + fmt(ball.speed[i], 1) + ' m/s' });
      }
      const dk = sc.dribbleKey;
      if (i >= Rv - 2 && dk[i] >= 0) {
        const k = runLen(dk, i, dk[i]);
        fs.overlays.push({ type: 'progress', slot: dk[i], r: 2.1, frac: k / dN, color: 'ev-dribble', label: 'dribble ' + Math.min(k, dN) + '/' + dN });
      }
      fs.insets.push({ type: 'spark', title: 'ball speed, m/s', n, cursor: i, yMax: 10,
        series: [{ values: ball.speed, color: 'text', label: 'speed' }],
        threshold: { value: maxV, label: 'control speed ' + maxV, color: 'ev-control' },
        markers: sc.events.filter((e) => e.kind === 'control' || e.kind === 'dribble').map((e) => ({ i: e.frame, v: ball.speed[e.frame], color: 'ev-' + e.kind, label: e.kind })),
        readout: fmt(ball.speed[i], 2) + ' m/s' });
      let items;
      if (i < Rv) {
        const d = Math.min(sc.f.dH[i], sc.f.dA[i]);
        items = [
          { text: 'ball ' + fmt(d, 2) + ' m from a player, limit ' + cfg.get('R_CONTROL') + ' m', ok: d <= cfg.get('R_CONTROL') },
          { text: 'speed ' + fmt(ball.speed[i], 2) + ' m/s (needs < ' + maxV + ')', ok: ball.speed[i] < maxV },
          { text: need + ' frames in a row, uncontested', ok: ctrlRow && i >= ctrlRow.frame + need ? true : null },
        ];
      } else {
        const s0 = dribRow ? dribRow.frame : Rv;
        const upto = Math.max(s0, i);
        const rel = [], sp = [];
        for (let t = s0; t <= upto; t++) { rel.push(sc.f.relA[t]); sp.push(ball.speed[t]); }
        const k = runLen(dk, i, dk[i]);
        items = [
          { text: 'same player within ' + rD + ' m: ' + Math.min(k, dN) + ' of ' + dN + ' frames', ok: k >= dN ? true : null },
          { text: 'relative speed ' + fmt(median(rel), 2) + ' m/s (needs < ' + relMax + ')', ok: median(rel) < relMax },
          { text: 'ball speed ' + fmt(median(sp), 2) + ' m/s (needs > ' + maxV + ')', ok: median(sp) > maxV },
        ];
      }
      fs.insets.push({ type: 'checklist', title: i < Rv ? 'control test' : 'dribble test', items });
    },
  });
}

function sceneOut(env, variant) {
  const { cfg, L, W, fps } = env;
  const n = 175;
  const world = createWorld(env, variant === 'lag' ? 404 : 405);
  const lagN = cfg.get('DELTA_OUT_LAG');
  const decel = cfg.get('FRICTION_DECEL');
  // roll: kicked hard toward the touchline, lost 12 to 14 m out with the roll reaching the line
  // lag: a slow roll lost 13 m out (the 12851 geometry), the roll stops short of the line
  const lose = variant === 'lag' ? 70 : 52;
  let from, dir, v0;
  if (variant === 'lag') {
    const vLoss = Math.sqrt(2 * decel * 13) * 0.92;
    v0 = vLoss + decel * (lose - 10) / fps;
    dir = [0.2, 0.98];
    const travel = (v0 * v0 - vLoss * vLoss) / (2 * decel);
    from = [62, W - 13 - travel * dir[1]];
  } else {
    v0 = 12.5;
    dir = [0.3, 0.954];
    from = [52, 32];
  }
  const kicker = world.bySlot.get(8);
  kicker.override = (t) => (t < 14 ? { x: from[0] - dir[0] * 0.5, y: from[1] - dir[1] * 0.5, cap: 1 } : null);
  let stopFrom = 1e9;
  world.cap = (t) => (t >= stopFrom ? 0.7 : 3.4);
  const raw = runWorld(world, n, (t, w) => {
    if (t === 0) { w.ball.x = from[0]; w.ball.y = from[1]; w.ball.vx = 0; w.ball.vy = 0; w.receive(8, true); }
    if (t === 10) { const d = 30; w.kickTo(w.ball.x + dir[0] * d, w.ball.y + dir[1] * d, v0); }
    if (t === lose) w.ball.vis = false;
    if (t === lose + (variant === 'lag' ? lagN : 30) + 18) { stopFrom = t; w.shift.hx = -5; w.shift.ax = 5; }
  }, (w) => w.put(8, from[0] - dir[0] * 0.5, from[1] - dir[1] * 0.5));
  const ball = ballFrom(raw, env, true);
  const sc = baseScene(env, raw, ball);
  const pre = outOnsets(sc, env);
  void pre;
  detectEvents(sc, env, new Uint8Array(n).fill(1));
  sc.onsets = outOnsets(sc, env).map((o) => Object.assign(o, { edge: 'bottom' }));
  const on = sc.onsets[0];
  if (on) setStoppage(sc, [[on.frame, n]]);
  sc.events = sc.events.filter((e) => !sc.stoppage[e.frame]);
  decodeScene(sc, env, { initialOwner: 0 });
  const ls = sc.lastSeen;
  // the extrapolated path: friction-decayed roll from the last sighting
  const path = [];
  if (ls) {
    const roll = Math.min(ls.dist + 1.5, (ls.speed * ls.speed) / (2 * decel));
    for (let k = 0; k <= 24; k++) { const d = (roll * k) / 24; path.push(ls.x + ls.ux * d, ls.y + ls.uy * d); }
  }
  const horizonS = cfg.get('ONSET_CONFIRM_S');
  const explainTiming = Math.max(lose + 4, on ? on.frame : lose + lagN);
  return finish(sc, {
    id: 'out-' + variant, zoom: 1.6,
    camAt: (i, x) => [ls ? ls.x + 2 : x, W - 13],
    title: variant === 'lag' ? 'Out, the roll stops short' : 'Out, by extrapolation', keyFrame: on ? on.frame + 3 : 100,
    beats: variant === 'lag' ? [
      { from: 0, text: 'A slow ball rolls toward the touchline. The tracker loses it 13 m from the line, as at frame 12851 of the public window.' },
      { from: lose, text: 'Last sighting: heading for a line within ' + cfg.get('OUT_MAX_RANGE_M') + ' m. A roll with a friction of ' + decel + ' m/s² stops short of it.' },
      { from: explainTiming, text: lagN === 0
        ? 'Once the players confirm the stop, the stoppage starts at the last sighting, with no added delay.'
        : 'So the stoppage starts ' + lagN + ' frames after the last sighting.' },
      { from: Math.max(explainTiming + 4, on ? on.frame + 25 : 120), text: 'Kept because the players stop within ' + horizonS + ' s.' },
    ] : [
      { from: 0, text: 'A hard ball toward the touchline. The tracker loses it in flight.' },
      { from: lose, text: 'Last sighting: heading for a line within ' + cfg.get('OUT_MAX_RANGE_M') + ' m. The rule engine extrapolates a roll with a friction of ' + decel + ' m/s².' },
      { from: on ? on.frame : 80, text: 'The roll reaches the line: the stoppage starts at the predicted crossing.' },
      { from: on ? on.frame + 25 : 120, text: 'Kept because the players stop within ' + horizonS + ' s.' },
    ],
    decorate(i, fs) {
      if (ls && i >= ls.frame) {
        fs.overlays.push({ type: 'path', pts: path, color: 'onset', dash: [5, 4], width: 2, alpha: 0.9 });
        fs.overlays.push({ type: 'cross', x: ls.x, y: ls.y, color: 'muted', size: 0.6, label: 'last seen ' + fmt(ls.speed, 1) + ' m/s' });
        if (ls.how === 'lag' && sc.rollStop) {
          fs.overlays.push({ type: 'circle', x: sc.rollStop[0], y: sc.rollStop[1], r: 0.7, color: 'onset', label: 'roll stops ' + fmt(sc.rollStop[2], 1) + ' m short' });
        } else if (ls.cross) {
          fs.overlays.push({ type: 'circle', x: ls.cross[0], y: ls.cross[1], r: 0.8, color: 'onset', label: 'crossing at +' + (ls.onset - ls.frame) + ' frames' });
        }
        if (ls.how === 'lag' && on && i < on.frame) {
          fs.overlays.push({ type: 'label', x: ls.x + 10, y: ls.y - 6, text: 'delay ' + (i - ls.frame) + '/' + lagN, color: 'onset', top: true });
        }
      }
      const items = [
        { text: 'ball lost heading for a line ' + (ls ? fmt(ls.dist, 1) : '?') + ' m away', ok: ls && i >= ls.frame ? ls.dist <= cfg.get('OUT_MAX_RANGE_M') : null },
        { text: ls && ls.how === 'lag'
          ? (lagN === 0 ? 'roll stops short: last sighting, no added delay' : 'roll stops short: last seen + ' + lagN + ' frames')
          : 'roll reaches the line: crossing frame', ok: on && i >= on.frame ? true : null },
        { text: 'collective stop within ' + horizonS + ' s: kept', ok: on && on.stopAt >= 0 && i >= on.stopAt ? true : null },
      ];
      fs.insets.push({ type: 'checklist', title: 'ball out: stoppage start', items });
      fs.insets.push({ type: 'spark', title: 'mean player speed, m/s', n, cursor: i, yMax: 4.5,
        series: [{ values: sc.f.meanvAll, color: 'text', label: 'mean speed' }],
        threshold: { value: cfg.get('V_STOP'), label: 'stop speed ' + cfg.get('V_STOP'), color: 'onset' },
        markers: on ? [{ i: on.frame, color: 'onset', label: 'stoppage start' }] : [], readout: fmt(sc.f.meanvAll[i], 2) + ' m/s' });
    },
  });
}

function scenePursuit(env) {
  const { cfg, L } = env;
  const n = 175;
  const world = createWorld(env, 506);
  const K = 20, whistle = 78;
  const stopV = cfg.get('PURSUIT_STOP_SPEED');
  const nNear = cfg.get('PURSUIT_N_NEAREST');
  const kicker = world.bySlot.get(5);
  kicker.override = (t) => (t < K + 4 ? { x: 35.5, y: 33.6, cap: 1 } : null);
  world.cap = (t, a) => (t >= whistle ? 0.6 : 3.6);
  // chasers: everyone near the flight runs after the ball until the whistle
  for (const a of world.agents) {
    if (a.slot === 5) continue;
    const prev = a.override;
    a.override = (t, me, w) => {
      if (prev) { const r = prev(t, me, w); if (r) return r; }
      if (t > K + 5 && t < whistle && Math.hypot(me.x - w.ball.x, me.y - w.ball.y) < 26 && a.role !== 'gk') {
        return { x: w.ball.x, y: w.ball.y, cap: a.team === 1 ? 5.2 : 4.4 };
      }
      if (t >= whistle) return { x: me.x + me.vx * 0.2, y: me.y + me.vy * 0.2, cap: 0.5, accel: 2.4 };
      return null;
    };
  }
  const raw = runWorld(world, n, (t, w) => {
    if (t === 0) { w.ball.x = 36; w.ball.y = 34; w.receive(5, true); }
    if (t === K) w.kickTo(L - 8, 22, 15.5);
    if (t === whistle + 20) { w.shift.hx = -6; w.shift.ax = 6; }
  }, (w) => w.put(5, 35.5, 33.6));
  const ball = ballFrom(raw, env, true);
  const air = cfg.get('AIRBORNE_SPEED');
  for (let t = 0; t < n; t++) ball.airborne[t] = ball.speed[t] > air ? 1 : 0;
  const sc = baseScene(env, raw, ball);
  detectEvents(sc, env, new Uint8Array(n).fill(1));
  sc.onsets = pursuitOnsets(sc, env);
  const on = sc.onsets[0];
  if (on) setStoppage(sc, [[on.frame, n]]);
  sc.events = sc.events.filter((e) => !sc.stoppage[e.frame]);
  decodeScene(sc, env, { initialOwner: 0 });
  const nearestSix = (i) => {
    const out = [];
    for (let s = 0; s < NS; s++) {
      const x = sc.px[i * NS + s];
      if (!isF(x) || !ball.vis[i]) continue;
      out.push([Math.hypot(x - ball.x[i], sc.py[i * NS + s] - ball.y[i]), s]);
    }
    out.sort((p, q) => p[0] - q[0]);
    return out.slice(0, nNear).map((p) => p[1]);
  };
  const pu = sc.pursuit;
  return finish(sc, {
    id: 'pursuit', zoom: 1.2, title: 'Pursuit stop', keyFrame: on && pu ? Math.max(on.frame + 20, pu.drop + 2) : 100,
    beats: [
      { from: 0, text: 'A long ball, faster than ' + cfg.get('PURSUIT_MIN_BALL_SPEED') + ' m/s. The players chase it.' },
      { from: whistle, text: 'The whistle goes with the ball still rolling. The ' + nNear + ' players nearest the ball slow down.' },
      { from: on ? on.frame : whistle + 10, text: 'They fall below ' + stopV + ' m/s and nobody of the kicking team reaches the ball: a pursuit stop, as at frame 15889 of the public window.' },
      { from: on ? on.frame + 30 : 130, text: 'The stoppage is dated at the start of their final deceleration, never before the ball’s peak speed.' },
    ],
    decorate(i, fs) {
      const six = nearestSix(i);
      for (const s of six) fs.overlays.push({ type: 'circle', slot: s, r: 1.3, color: 'onset', width: 1.2, alpha: 0.75 });
      const bars = six.map((s) => ({ label: String(s), value: sc.f.ps[i * NS + s], color: sc.team[s] === 0 ? 'h' : 'a' }));
      fs.insets.push({ type: 'bars', title: 'speed of the ' + nNear + ' nearest players, m/s', bars, max: 6,
        threshold: { value: stopV, label: 'stop speed ' + stopV, color: 'onset' },
        mean: { value: sc.f.meanvNear[i], label: 'mean ' + fmt(sc.f.meanvNear[i], 2), color: 'text' } });
      fs.insets.push({ type: 'checklist', title: 'pursuit stop', items: [
        { text: 'ball over ' + cfg.get('PURSUIT_MIN_BALL_SPEED') + ' m/s after the kick', ok: pu && i >= pu.start ? true : null },
        { text: 'nobody of ' + (pu ? pu.kick : '?') + ' reaches the ball', ok: pu && i >= pu.onset ? true : null },
        { text: 'nearest under ' + stopV + ' m/s within ' + cfg.get('ONSET_CONFIRM_S') + ' s', ok: pu && i >= pu.drop ? true : null },
        { text: 'no ' + (pu ? pu.kick : '') + ' player within touch radius for ' + cfg.get('COLLECTIVE_STOP_S') + ' s', ok: on && i >= on.frame + 50 ? true : null },
      ] });
    },
  });
}

function sceneCollective(env) {
  const { cfg, fps } = env;
  const n = 190;
  const world = createWorld(env, 607);
  const gone = 38, slow = 46;
  world.cap = (t) => (t >= slow ? 0.55 : 3.4);
  const raw = runWorld(world, n, (t, w) => {
    if (t === 0) { w.ball.x = 58; w.ball.y = 40; w.ball.vx = 6; w.ball.vy = -2; }
    if (t === gone) w.ball.vis = false;
    if (t === slow) { w.shift.hx = -5; w.shift.ax = 5; }
  });
  const ball = ballFrom(raw, env, true);
  const sc = baseScene(env, raw, ball);
  detectEvents(sc, env, new Uint8Array(n).fill(1));
  const vStop = cfg.get('V_STOP'), sepDead = cfg.get('SEP_DEAD'), high = cfg.get('DEAD_EVIDENCE_HIGH');
  const bias = cfg.get('DEAD_LOGIT_BIAS');
  const terms = (t) => {
    const isSlow = sc.f.meanvAll[t] < vStop;
    const gap = !ball.vis[t];
    const sep = sc.f.sep[t] > sepDead;
    return [
      { label: 'base level', value: bias, active: true },
      { label: 'ball missing and players slow', value: cfg.get('W_GAP_STOP'), active: gap && isSlow },
      { label: 'players under ' + vStop + ' m/s', value: cfg.get('W_SLOW'), active: isSlow },
      { label: 'mean positions of the teams over ' + sepDead + ' m apart', value: cfg.get('W_SEP'), active: sep },
    ];
  };
  const eD = new Float32Array(n);
  for (let t = 0; t < n; t++) eD[t] = sigmoid(terms(t).reduce((p, q) => p + (q.active ? q.value : 0), 0));
  const dwell = Math.round(cfg.get('COLLECTIVE_STOP_S') * fps);
  let runStart = -1, enter = -1;
  for (let t = 0; t < n; t++) {
    if (eD[t] > high) { if (runStart < 0) runStart = t; if (t - runStart + 1 >= dwell && enter < 0) enter = t; } else runStart = -1;
  }
  // Knee onset (possession.knee_retimer, deadball.deceleration_knee): the ball is missing at the
  // entry, so the entry moves back to the players' deceleration knee plus DELTA_KNEE.
  const dwellEnd = enter;
  let knee = -1;
  if (enter >= 0 && cfgOr(cfg, 'ONSET_KNEE') && !ball.vis[enter]) {
    const half = Math.round(cfgOr(cfg, 'KNEE_SMOOTH_S') * fps / 2);
    const smooth = centredNanmean(Array.from(sc.f.meanvAll), half);
    const pre = median(smooth.slice(Math.max(0, enter - Math.round(cfgOr(cfg, 'KNEE_PRE_FROM_S') * fps)),
      Math.max(0, enter - Math.round(cfgOr(cfg, 'KNEE_PRE_TO_S') * fps))));
    let lo = Math.max(0, enter - Math.round(cfgOr(cfg, 'KNEE_LOOKBACK_S') * fps));
    if (cfgOr(cfg, 'KNEE_GAP_CLAMP')) lo = Math.max(lo, gone);
    if (lo < enter && isF(pre) && pre > vStop) {
      const level = vStop + cfgOr(cfg, 'KNEE_LEVEL_FRAC') * (pre - vStop);
      knee = lo;
      for (let t = lo; t <= enter; t++) if (isF(smooth[t]) && smooth[t] >= level) knee = t;
      enter = Math.min(enter, Math.max(lo, knee + cfgOr(cfg, 'DELTA_KNEE')));
    }
  }
  sc.onsets = [];
  if (enter >= 0) {
    setStoppage(sc, [[enter, n]]);
    sc.onsets = [{ frame: enter, mechanism: 'collective_stop', x: NaN, y: NaN }];
  }
  for (let t = 0; t < n; t++) if (!sc.stoppage[t]) sc.pDead[t] = Math.max(sc.pDead[t], eD[t] * 0.8);
  decodeScene(sc, env, { initialOwner: 1 });
  const first = (() => { for (let t = 0; t < n; t++) if (eD[t] > high) return t; return -1; })();
  return finish(sc, {
    id: 'collective', zoom: 1.0, title: 'Collective stop, ball missing', keyFrame: enter >= 0 ? enter + 10 : 150,
    beats: [
      { from: 0, text: 'Open play. Then the tracker loses the ball. A gap alone is no evidence: balls go missing in play.' },
      { from: slow, text: 'The players slow below ' + vStop + ' m/s with the ball still missing: the strongest term, ball missing and players slow, switches on.' },
      { from: first >= 0 ? first : 70, text: 'The dead-ball evidence passes ' + high + '. It must stay there for ' + cfg.get('COLLECTIVE_STOP_S') + ' s.' },
      { from: dwellEnd >= 0 ? dwellEnd : 140, text: knee >= 0
        ? 'The wait is complete: the ball is dead. The rule engine dates the stoppage at the knee, ' + Math.abs(cfgOr(cfg, 'DELTA_KNEE')) + ' frames before the half-way point of the players\u2019 deceleration, never before the last sighting of the ball.'
        : 'The wait is complete: the ball is dead.' },
    ],
    decorate(i, fs) {
      const tm = terms(i);
      const z = tm.reduce((p, q) => p + (q.active ? q.value : 0), 0);
      fs.insets.push({ type: 'terms', title: 'dead-ball evidence: sigmoid of the sum of the active terms', terms: tm,
        result: { label: 'evidence', value: sigmoid(z).toFixed(2), color: sigmoid(z) > high ? 'onset' : 'text' } });
      fs.insets.push({ type: 'spark', title: 'dead-ball evidence', n, cursor: i, yMax: 1,
        series: [{ values: eD, color: 'onset', fill: true, label: 'evidence' }],
        threshold: { value: high, label: 'threshold ' + high, color: 'text' },
        brackets: first >= 0 && i >= first ? [{ i0: first, i1: Math.min(first + dwell, n - 1), color: 'onset', label: cfg.get('COLLECTIVE_STOP_S') + ' s wait' }] : [],
        readout: 'evidence ' + eD[i].toFixed(2) });
      if (!ball.vis[i]) fs.overlays.push({ type: 'label', x: env.L / 2, y: 6, text: 'ball missing', color: 'muted', top: true });
    },
  });
}

/** Stoppage exits from confirmed candidates after RESTART_ARM_S in D, with the kind offsets. */
function exitsFrom(sc, env, cands, deadFrom) {
  const { cfg, fps } = env;
  const arm = Math.round(cfg.get('RESTART_ARM_S') * fps);
  for (const c of cands) {
    if (!c.confirmed || c.frame - deadFrom < arm) continue;
    const delta = c.kind === 'kick' ? cfg.get('DELTA_RESTART_KICK') : cfg.get('DELTA_RESTART_REAPPEAR');
    return Object.assign({}, c, { exit: Math.min(sc.n - 1, c.frame + delta), delta });
  }
  return null;
}

function sceneReappear(env) {
  const { cfg, W } = env;
  const n = 160;
  const world = createWorld(env, 708);
  const R = 45;
  const thrower = world.bySlot.get(3);
  thrower.override = (t) => (t < R + 8 ? { x: 50, y: W + 1.6, cap: 1.5 } : null);
  let move = 1e9;
  world.cap = (t) => (t >= move ? 3.8 : 0.7);
  const raw = runWorld(world, n, (t, w) => {
    if (t === 0) { w.ball.vis = false; w.ball.x = 50; w.ball.y = W + 0.6; }
    if (t === R) { w.ball.vis = true; w.ball.x = 50.2; w.ball.y = W + 1.3; w.kickTo(56, W - 14, cfg.get('REAPPEAR_SPEED') * 2.1); }
    if (t === R + 18) move = t;
  }, (w) => w.put(3, 50, W + 1.6));
  const ball = ballFrom(raw, env, false);
  const sc = baseScene(env, raw, ball);
  const alive = Uint8Array.from({ length: n }, (_, t) => (t >= R ? 1 : 0));
  detectEvents(sc, env, alive);
  sc.onsets = outOnsets(sc, env);
  sc.restarts = restartCandidates(sc, env, []);
  const ex = exitsFrom(sc, env, sc.restarts.filter((c) => c.kind === 'reappear'), 0);
  sc.exits = ex ? [ex] : [];
  setStoppage(sc, [[0, ex ? ex.exit : n]]);
  sc.events = sc.events.filter((e) => !sc.stoppage[e.frame]);
  decodeScene(sc, env, { initialOwner: 0 });
  const c = sc.restarts.find((r) => r.kind === 'reappear');
  return finish(sc, {
    id: 'reappear', zoom: 1.8,
    camAt: (i, x) => [x, W - 12],
    title: 'Restart by reappearance', keyFrame: ex ? ex.exit + 5 : 80,
    beats: [
      { from: 0, text: 'The ball is out of play and out of sight. The match is in D.' },
      { from: R, text: 'It reappears beyond the touchline, thrown in faster than ' + cfg.get('REAPPEAR_SPEED') + ' m/s: a possible restart.' },
      { from: R + 3, text: 'It moves toward the pitch within ' + cfg.get('OUT_INWARD_FRAMES') + ' frames, so it is a throw-in, not a new ball out.' },
      { from: Math.max(R + 4, ex ? ex.exit : R + cfg.get('DELTA_RESTART_REAPPEAR')),
        text: cfg.get('DELTA_RESTART_REAPPEAR') === 0
          ? 'The later player movement confirms the restart. Play resumes at the reappearance, with no added offset.'
          : 'Play resumes ' + cfg.get('DELTA_RESTART_REAPPEAR') + ' frames after the reappearance. The players then move: confirmed.' },
    ],
    decorate(i, fs) {
      fs.insets.push({ type: 'checklist', title: 'possible restart', items: [
        { text: 'reappears at ' + fmt(c && c.speed, 1) + ' m/s (needs > ' + cfg.get('REAPPEAR_SPEED') + ')', ok: c && i >= c.frame ? true : null },
        { text: 'nearest player ' + fmt(c && c.d, 2) + ' m (≤ ' + cfg.get('REAPPEAR_TEAM_M') + ' m): team ' + (c && c.team), ok: c && i >= c.frame ? c.team != null : null },
        { text: 'moving inward: a throw-in, not a ball out', ok: sc.throwIn != null && i >= R + cfg.get('OUT_INWARD_FRAMES') ? true : null },
        { text: 'players above the stop speed within ' + cfg.get('RESTART_CONFIRM_S') + ' s', ok: c && c.players && i >= c.movingAt ? true : null },
      ] });
      fs.insets.push({ type: 'spark', title: 'mean player speed, m/s', n, cursor: i, yMax: 4.5,
        series: [{ values: sc.f.meanvAll, color: 'text', label: 'mean speed' }],
        threshold: { value: cfg.get('V_STOP'), label: 'stop speed ' + cfg.get('V_STOP'), color: 'restart' },
        markers: c ? [{ i: c.frame, color: 'restart', label: 'ball back' }].concat(ex ? [{ i: ex.exit, color: 'restart', label: '+' + ex.delta, dashed: true }] : []) : [],
        readout: fmt(sc.f.meanvAll[i], 2) + ' m/s' });
    },
  });
}

function sceneChain(env) {
  const { cfg, W, fps } = env;
  const confirmN = Math.round(cfg.get('RESTART_CONFIRM_S') * fps);
  const chainN = cfg.get('RESTART_CHAIN_FRAMES');
  const K1 = 50;
  const K2 = K1 + Math.round(chainN * 0.7);
  const moveAt = K1 + confirmN + 12;
  const n = Math.max(moveAt + 50, K2 + 120);
  const world = createWorld(env, 809);
  const gk = world.bySlot.get(0), def = world.bySlot.get(2);
  gk.override = (t) => (t < K1 + 4 ? { x: 5.6, y: W / 2 - 4, cap: 1 } : t < K2 + 30 ? { x: 9, y: W / 2 - 6, cap: 1.4 } : null);
  const meet = [22, 20];
  def.override = (t) => (t < K2 + 4 ? { x: meet[0], y: meet[1], cap: 1.4 } : null);
  world.cap = (t) => (t >= moveAt ? 4.2 : 0.9);
  world.shift.hx = -16;
  world.shift.ax = 4;
  let r1 = -1;
  const raw = runWorld(world, n, (t, w) => {
    if (t === 0) { w.ball.x = 6; w.ball.y = W / 2 - 4; w.ball.vx = 0; w.ball.vy = 0; }
    if (t === K1) w.kickTo(meet[0], meet[1], 11);
    if (t > K1 && r1 < 0 && Math.hypot(w.ball.x - def.x, w.ball.y - def.y) < 0.9) { r1 = t; w.receive(2); }
    if (t === K2) w.kickTo(10, W / 2 - 6, 10);
    if (t > K2 + 10 && w.ball.attach < 0 && Math.hypot(w.ball.x - gk.x, w.ball.y - gk.y) < 0.9) w.receive(0, true);
  }, (w) => { w.put(0, 5.4, W / 2 - 4); w.put(2, meet[0] + 0.5, meet[1]); });
  const ball = ballFrom(raw, env, true);
  const sc = baseScene(env, raw, ball);
  sc.restarts = restartCandidates(sc, env, []);
  const kicks = sc.restarts.filter((c) => c.kind === 'kick');
  const ex = exitsFrom(sc, env, kicks, 0);
  sc.exits = ex ? [ex] : [];
  setStoppage(sc, [[0, ex ? ex.exit : n]]);
  const alive = Uint8Array.from(sc.stoppage, (v) => (v ? 0 : 1));
  detectEvents(sc, env, alive);
  decodeScene(sc, env, { initialOwner: 0 });
  const c1 = kicks[0], c2 = kicks[1];
  const vStop = cfg.get('V_STOP');
  return finish(sc, {
    id: 'chain', zoom: 1.5,
    camAt: (i, x, y) => [Math.max(20, x), y],
    title: 'Chained restart', keyFrame: c2 ? c2.frame + 10 : 120,
    beats: [
      { from: 0, text: 'A goal kick. The ball is dead and placed.' },
      { from: c1 ? c1.frame : K1, text: 'The kick: the speed rises by more than ' + cfg.get('DV_RESTART') + ' m/s over ' + cfg.get('RESTART_K') + ' frames, with the goalkeeper at the ball.' },
      { from: c1 ? c1.frame + 20 : 60, text: 'But everybody walks: nobody goes above ' + vStop + ' m/s within ' + cfg.get('RESTART_CONFIRM_S') + ' s. Alone, this kick would fail, as the kick at frame 18355 of the public window did.' },
      { from: c2 ? c2.frame : K2, text: 'The next kick of the sequence comes ' + (c2 && c1 ? c2.frame - c1.frame : '?') + ' frames later, within ' + chainN + ' frames, and the players speed up after it.' },
      { from: moveAt, text: 'Play goes on: the second kick confirms the first. Play resumes ' + cfg.get('DELTA_RESTART_KICK') + ' frames after the first kick\u2019s rise begins.' },
    ],
    decorate(i, fs) {
      const brackets = [];
      if (c1 && i >= c1.frame) brackets.push({ i0: c1.frame, i1: Math.min(n - 1, c1.frame + confirmN), color: c1.players ? 'restart' : 'a', label: cfg.get('RESTART_CONFIRM_S') + ' s confirmation window' });
      if (c1 && c2 && i >= c2.frame) brackets.push({ i0: c1.frame, i1: c2.frame, color: 'restart', label: '≤ ' + chainN });
      fs.insets.push({ type: 'spark', title: 'mean player speed and restart candidates', n, cursor: i, yMax: 4.5,
        series: [{ values: sc.f.meanvAll, color: 'text', label: 'mean speed' }],
        threshold: { value: vStop, label: 'stop speed ' + vStop, color: 'restart' }, brackets,
        markers: kicks.map((c, k) => ({ i: c.frame, color: 'restart', label: 'kick ' + (k + 1) })).concat(ex ? [{ i: ex.exit, color: 'restart', label: 'play resumes', dashed: true }] : []),
        readout: fmt(sc.f.meanvAll[i], 2) + ' m/s' });
      fs.insets.push({ type: 'checklist', title: 'confirmation', items: [
        { text: 'kick 1: players speed up within ' + cfg.get('RESTART_CONFIRM_S') + ' s', ok: c1 && i >= c1.frame + confirmN ? c1.players : null },
        { text: 'kick 2: players speed up within ' + cfg.get('RESTART_CONFIRM_S') + ' s', ok: c2 && i >= c2.movingAt && c2.movingAt >= 0 ? c2.players : null },
        { text: 'gap ' + (c1 && c2 ? c2.frame - c1.frame : '?') + ' frames (needs ≤ ' + chainN + '), no stoppage', ok: c2 && i >= c2.frame ? true : null },
        { text: 'kick 1 confirmed by kick 2', ok: c1 && c1.chainedTo != null && i >= c1.frame + confirmN ? true : null },
      ] });
    },
  });
}

function sceneDecoder(env) {
  const { cfg, W } = env;
  const n = 250;
  const world = createWorld(env, 910);
  const dribbler = world.bySlot.get(7), challenger = world.bySlot.get(27), recv = world.bySlot.get(29);
  dribbler.override = (t) => (t < 112 ? { x: 44 + t * 0.12, y: 30, cap: 2.6 } : null);
  challenger.override = (t) => (t > 40 && t < 88 ? { x: 44 + t * 0.12 + 0.3, y: 31.2, cap: 3 } : t >= 88 && t < 150 ? { x: 40, y: 44, cap: 4.5 } : null);
  const passTo = [70, 44];
  recv.override = (t) => (t < 150 ? { x: passTo[0] + 0.5, y: passTo[1], cap: 3.4 } : t < 200 ? { x: 79, y: W - 11, cap: 5.2, accel: 2.5 } : null);
  let r = -1;
  const raw = runWorld(world, n, (t, w) => {
    if (t === 0) { w.ball.x = 44.6; w.ball.y = 30; w.receive(7); }
    if (t === 112) w.kickTo(passTo[0], passTo[1], 15.5);
    if (t > 112 && r < 0 && Math.hypot(w.ball.x - recv.x, w.ball.y - recv.y) < 0.8) { r = t; w.receive(29); }
    if (t === 192) w.kickTo(w.ball.x + 6, W + 10, 16);
  }, (w) => { w.put(7, 44, 30); w.put(29, passTo[0] + 4, passTo[1] + 3); w.put(27, 47, 36); });
  const ball = ballFrom(raw, env, true);
  const sc = baseScene(env, raw, ball);
  detectEvents(sc, env, new Uint8Array(n).fill(1));
  sc.onsets = outOnsets(sc, env).filter((o) => o.how === 'observed').map((o) => Object.assign(o, { edge: 'bottom' }));
  const on = sc.onsets[0];
  if (on) setStoppage(sc, [[on.frame, n]]);
  sc.events = sc.events.filter((e) => !sc.stoppage[e.frame]);
  decodeScene(sc, env, { initialOwner: 0 });
  const rowsH = new Float32Array(n), rowsA = new Float32Array(n), rowsD = new Float32Array(n);
  const near = new Float32Array(n), nearA = new Float32Array(n);
  const cost = new Float32Array(n);
  for (let t = 0; t < n; t++) {
    rowsH[t] = Math.exp(sc.logE[t * 3]);
    rowsA[t] = Math.exp(sc.logE[t * 3 + 1]);
    rowsD[t] = Math.exp(sc.logE[t * 3 + 2]);
    const nh = sc.f.dH[t] <= sc.f.dA[t];
    near[t] = ball.vis[t] && nh ? 0.9 : 0;
    nearA[t] = ball.vis[t] && !nh ? 0.9 : 0;
    const st = sc.labels[t];
    cost[t] = st === 0 ? sc.costs[t * 9 + 1] : st === 1 ? sc.costs[t * 9 + 3] : Math.min(sc.costs[t * 9 + 6], sc.costs[t * 9 + 7]);
  }
  const flicker = [];
  for (let t = 1; t < n; t++) if (ball.vis[t] && (sc.f.dH[t] <= sc.f.dA[t]) !== (sc.f.dH[t - 1] <= sc.f.dA[t - 1])) flicker.push(t);
  const cS = cfg.get('C_SWITCH'), cSE = cfg.get('C_SWITCH_EVENT');
  const ctrlA = sc.events.find((e) => e.team === 'A' && (e.kind === 'control' || e.kind === 'release'));
  return finish(sc, {
    id: 'decoder', zoom: 1.6, title: 'Owner and decoder', keyFrame: ctrlA ? ctrlA.frame + 15 : 160,
    beats: [
      { from: 0, text: 'H keeps the ball. The owner is the team of the last owner event. On each frame it gets a probability of ' + cfg.get('P_OWNER') + '.' },
      { from: 42, text: 'An A player closes in. The nearest team flips ' + (flicker.length === 1 ? 'once' : flicker.length + ' times') + ', but with no event a switch costs ' + cS + ' nats, a log-probability penalty: no switch.' },
      { from: 112, text: 'H plays a pass. The A player receives it: a reception, an owner event of the new team.' },
      { from: ctrlA ? ctrlA.frame - 3 : 140, text: 'Within ' + cfg.get('SWITCH_EVENT_WINDOW') + ' frames of the event the switch costs only ' + cSE + ' nats. The decoded path switches here, once.' },
      { from: on ? on.frame - 3 : 200, text: 'The ball crosses the line: the stoppage start makes the switch to D cheap (a cost of ' + cfg.get('C_DEAD_EVENT') + ' nats).' },
    ],
    decorate(i, fs) {
      fs.insets.push({ type: 'lanes', title: 'evidence for each label, decoded path, switch cost', n, cursor: i, weight: 2.6,
        rows: [
          { label: 'near H', color: 'h', values: near },
          { label: 'near A', color: 'a', values: nearA },
          { label: 'H', color: 'h', values: rowsH },
          { label: 'A', color: 'a', values: rowsA },
          { label: 'D', color: 'd', values: rowsD },
        ],
        path: Int8Array.from(sc.labels, (v) => v + 2),
        cost: cost, costMax: Math.max(cS, cfg.get('C_DEAD')), costLabel: 'switch cost ' + cS + ', near an event ' + cSE,
        markers: sc.events.filter((e) => e.team).map((e) => ({ i: e.frame, color: 'ev-' + e.kind })) });
      if (i >= 42 && i < 95) {
        const s = sc.f.nearA[i];
        if (s >= 0 && sc.f.dA[i] < sc.f.dH[i]) fs.overlays.push({ type: 'circle', slot: s, r: 1.8, color: 'a', dash: [3, 3], label: 'nearest, no event' });
      }
    },
  });
}

function sceneGoalkeeper(env) {
  const { cfg, L, W, fps } = env;
  const n = 240;
  const world = createWorld(env, 1011);
  const swapAt = 100;
  const gkH = world.bySlot.get(0), gkA = world.bySlot.get(23), shooter = world.bySlot.get(9);
  const body = { x: L - 4.5, y: W / 2 + 1 };
  gkA.override = (t) => (t < 40 ? { x: L - 4.2, y: W / 2 + 1.5, cap: 2 } : t < swapAt ? { x: L - 5, y: W / 2 + 1.5, cap: 0.6 }
    : { x: L - 14, y: W / 2 + 7, cap: 2.4, accel: 2.5 });
  shooter.override = (t) => (t < 18 ? { x: L - 26, y: W / 2 - 3, cap: 1 } : null);
  const raw = runWorld(world, n, (t, w) => {
    if (t === 0) { w.ball.x = L - 25.4; w.ball.y = W / 2 - 3; w.ball.vx = 0; w.ball.vy = 0; w.receive(9, true); }
    if (t === 15) w.kickTo(L - 4.2, W / 2 + 1.5, 19);
    if (t > 15 && w.ball.attach < 0 && Math.hypot(w.ball.x - gkA.x, w.ball.y - gkA.y) < 1.0) w.receive(23, true);
  }, (w) => { w.put(9, L - 26, W / 2 - 3); w.put(23, L - 4.2, W / 2 + 1.5); });
  void body;
  void gkH;
  // the tracker fault: from swapAt the H keeper slot carries the A keeper body, and the A keeper
  // slot freezes where it was. The H keeper body is untracked.
  const rawPx = Float32Array.from(raw.px), rawPy = Float32Array.from(raw.py);
  const fx = raw.px[(swapAt - 1) * NS + 23], fy = raw.py[(swapAt - 1) * NS + 23];
  for (let t = swapAt; t < n; t++) {
    rawPx[t * NS + 0] = raw.px[t * NS + 23];
    rawPy[t * NS + 0] = raw.py[t * NS + 23];
    rawPx[t * NS + 23] = fx;
    rawPy[t * NS + 23] = fy;
  }
  // the guard (tracks.goalkeeper_swaps): frozen over a centred FROZEN_WIN window, and the rest
  const win = cfg.get('FROZEN_WIN'), half = Math.floor(win / 2), moveM = cfg.get('FROZEN_MOVE_M');
  const swapR = cfg.get('GK_SWAP_R');
  const frozen = new Uint8Array(n), fire = new Uint8Array(n);
  const distGK = new Float32Array(n);
  for (let t = 0; t < n; t++) {
    const a = clamp(t - half, 0, n - 1), b = clamp(t + half, 0, n - 1);
    const mv = Math.hypot(rawPx[b * NS + 23] - rawPx[a * NS + 23], rawPy[b * NS + 23] - rawPy[a * NS + 23]);
    // a synthetic slot has a speed until the fault; from then the frozen slot has none
    frozen[t] = t >= swapAt && mv < moveM ? 1 : 0;
    distGK[t] = Math.hypot(rawPx[t * NS] - rawPx[t * NS + 23], rawPy[t * NS] - rawPy[t * NS + 23]);
    const beyond = Math.abs(rawPx[t * NS] - 0) > L / 2;
    fire[t] = isF(rawPx[t * NS]) && frozen[t] && distGK[t] <= swapR && beyond ? 1 : 0;
  }
  const firstFire = fire.indexOf(1);
  const patchedPx = Float32Array.from(rawPx), patchedPy = Float32Array.from(rawPy);
  for (let t = 0; t < n; t++) {
    if (!fire[t]) continue;
    patchedPx[t * NS + 23] = rawPx[t * NS];
    patchedPy[t * NS + 23] = rawPy[t * NS];
    patchedPx[t * NS] = NaN;
    patchedPy[t * NS] = NaN;
  }
  const show = 45;
  const showGuard = firstFire >= 0 ? firstFire + show : n;
  const ball = ballFrom(raw, env, true);
  // the engine and the decoded labels see the guarded positions
  const guarded = Object.assign({}, raw, { px: patchedPx, py: patchedPy });
  const sc = baseScene(env, guarded, ball);
  detectEvents(sc, env, new Uint8Array(n).fill(1));
  decodeScene(sc, env, { initialOwner: 0 });
  // without the guard: the same detectors on the raw positions
  const rawSc = baseScene(env, Object.assign({}, raw, { px: rawPx, py: rawPy }), ball);
  detectEvents(rawSc, env, new Uint8Array(n).fill(1));
  decodeScene(rawSc, env, { initialOwner: 0 });
  sc.rawEvents = rawSc.events;
  const view = (i) => (i >= showGuard ? [patchedPx, patchedPy] : [rawPx, rawPy]);
  const sl = (arr, i) => arr.subarray(i * NS, (i + 1) * NS);
  return finish(sc, {
    id: 'gk', zoom: 1.4,
    camAt: (i, x) => [Math.min(x, L - 16), W / 2 + 1],
    title: 'Goalkeeper-swap correction', keyFrame: showGuard + 10,
    posX: (i, axis) => sl(view(i)[axis === 'x' ? 0 : 1], i),
    beats: [
      { from: 0, text: 'A shot. The A goalkeeper catches the ball near the A goal.' },
      { from: swapAt, text: 'The tracker fault: the H goalkeeper’s track jumps onto the A goalkeeper’s body, and the A goalkeeper’s track freezes a few metres away.' },
      { from: swapAt + 10, text: 'Without the correction, the rule engine credits H with the ball, at full confidence. This happened at frame 15654 of the public window.' },
      { from: showGuard - 15, text: 'The correction fires: a moving goalkeeper track beyond its half-way line, and the other goalkeeper’s track frozen within ' + swapR + ' m.' },
      { from: showGuard, text: 'The moving position goes back to the frozen track. The body is A again, before any detector runs.' },
    ],
    decorate(i, fs) {
      if (i >= swapAt && i < showGuard) {
        fs.overlays.push({ type: 'circle', x: fx, y: fy, r: 1.4, color: 'a', dash: [2, 3],
          label: frozen[i] ? 'A goalkeeper track frozen: < ' + moveM + ' m in ' + win + ' frames' : 'A goalkeeper track still' });
        fs.overlays.push({ type: 'circle', slot: 0, r: 2.2, color: 'h', dash: [4, 3], label: 'H goalkeeper track on the A goalkeeper' });
        fs.overlays.push({ type: 'line', x1: L / 2, y1: 0, x2: L / 2, y2: W, color: 'gt', dash: [6, 5], alpha: 0.6, label: 'half-way' });
        fs.labels.ours = 'HADX'[rawSc.labels[i]];
        fs.badge = 'without the correction: owner ' + (rawSc.owner[i] === 0 ? 'H' : rawSc.owner[i] === 1 ? 'A' : '?');
      }
      if (i >= showGuard && i < showGuard + 20) {
        const a = 1 - (i - showGuard) / 20;
        fs.overlays.push({ type: 'circle', slot: 23, r: 1.2 + (i - showGuard) * 0.25, color: 'a', alpha: a, width: 3, label: 'A goalkeeper position restored', top: true });
      }
      fs.insets.push({ type: 'checklist', title: 'goalkeeper-swap correction, frame ' + i, items: [
        { text: 'H goalkeeper track visible and moving', ok: i >= swapAt ? true : null },
        { text: 'A goalkeeper frozen: < ' + moveM + ' m in ' + win + ' frames', ok: i >= swapAt ? !!frozen[i] || i >= showGuard : null },
        { text: 'distance ' + fmt(distGK[i], 1) + ' m (needs ≤ ' + swapR + ')', ok: i >= swapAt ? distGK[i] <= swapR : null },
        { text: 'H goalkeeper beyond its half-way line', ok: i >= swapAt ? true : null },
        { text: 'correction: ' + fire.reduce((p, v, t) => p + (v && t <= i ? 1 : 0), 0) + ' frames fixed', ok: firstFire >= 0 && i >= firstFire ? true : null },
      ] });
    },
  });
}

const BUILDERS = {
  cleaning: sceneCleaning,
  contacts: sceneContacts,
  kink: sceneKink,
  dribble: sceneDribble,
  'out-roll': (env) => sceneOut(env, 'roll'),
  'out-lag': (env) => sceneOut(env, 'lag'),
  pursuit: scenePursuit,
  collective: sceneCollective,
  reappear: sceneReappear,
  chain: sceneChain,
  decoder: sceneDecoder,
  gk: sceneGoalkeeper,
  redirect: (env) => sceneRedirect(env, 'shape'),
  'redirect-flat': (env) => sceneRedirect(env, 'flat'),
  'redirect-reach': (env) => sceneRedirect(env, 'reach'),
  'redirect-reach-flat': (env) => sceneRedirect(env, 'reach-flat'),
};

/** Short names of the scenes, for the chips. */
export const SCENE_NAMES = {
  cleaning: 'cleaning', contacts: 'contacts', kink: 'fitted contact point', dribble: 'slow ball, dribble', 'out-roll': 'out, extrapolated',
  'out-lag': 'out, roll stops short', pursuit: 'pursuit stop', collective: 'collective stop', reappear: 'reappearance',
  chain: 'chained restart', decoder: 'decoder', gk: 'goalkeeper swap',
  redirect: 'redirect, confirmed', 'redirect-flat': 'redirect, not confirmed',
  'redirect-reach': 'distant touch, confirmed', 'redirect-reach-flat': 'distant touch, not confirmed',
};

/**
 * Create the scene library. Scenes are built lazily on first use and cached.
 * @param {{config: Object, pitchM: [number, number], fps?: number}} opts
 */
export function createScenes(opts) {
  const cfg = configAccessor(opts.config);
  const fps = opts.fps || 25;
  const h = cfg.get('CONTACT_HALF_WINDOW');
  const env = {
    cfg, fps, L: opts.pitchM[0], W: opts.pitchM[1],
    friction: cfg.get('FRICTION_DECEL'),
    // the provider's kernel: half-width at half-maximum h - 1 frames (CONTACT_HALF_WINDOW = HWHM + 1)
    sigma: (h - 1) / Math.sqrt(2 * Math.LN2),
  };
  const cache = new Map();
  return {
    cfg, env, ids: Object.keys(BUILDERS), steps: STEP_SCENES,
    get(id) {
      if (!cache.has(id)) {
        if (!BUILDERS[id]) throw new Error('no synthetic scene ' + id);
        cache.set(id, BUILDERS[id](env));
      }
      return cache.get(id);
    },
  };
}
