/**
 * Pitch renderer for the replay panel. Canvas 2D at devicePixelRatio.
 *
 * `createPitch(canvas, {pitchM, tokens, marks})` returns a renderer. `draw(frameState, layers)`
 * paints one frame. The renderer keeps no playback state and fetches nothing: everything it draws
 * comes from the frame state. The pitch markings are drawn once to an offscreen canvas on resize.
 *
 * Coordinates are metres. x runs along the pitch length from 0 to pitchM[0], y across the width
 * from 0 to pitchM[1], and y grows downwards on screen.
 *
 * Frame state (all keys optional unless noted):
 *   frame        number            frame shown in captions
 *   nSlots       number            required, number of player slots (46 for the ATD)
 *   px, py       Float32Array      required, player positions in metres, NaN = not visible
 *   team         Uint8Array        team per slot, 0 = H, 1 = A, 255 = unknown
 *   trail        {n, px, py}       the n frames before this one, oldest first, n x nSlots each
 *   ball         {x, y, visible, rawX, rawY, rawVisible, airborne, ghost, teleport, speed}
 *   ballTrail    {n, x, y}         ball positions of the frames before this one, oldest first
 *   glow         {slot, team, alpha}  owner glow around one player
 *   pDead        number 0..1       dead-ball dimming
 *   stoppage     boolean           the engine marks the frame as a stoppage
 *   nearest      [{slot, d}, {slot, d}]  nearest H and nearest A player and their distance to the ball
 *   events       [{kind, team, dt, x, y, label}]       dt = this frame minus the event frame
 *   onsets       [{mechanism, dt, x, y}]
 *   restarts     [{kind, team, dt, x, y, confirmed}]
 *   overlays     [overlay]         scene annotations in pitch coordinates, see drawOverlay
 *   insets       [inset]           glass cards drawn over the bottom of the pitch, see drawInsets
 *   badge        string            text of the top badge, replaces the default dead-ball badge
 *   camera       {x, y, zoom}      follow camera: the view centres on (x, y) in metres at this zoom
 *
 * Layers (booleans): camera, trails, glow, dim, rings, onset, restart, touch, slotIds, airborne, ghost,
 * insets, overlays. All default true except camera: the follow camera zooms in and cuts off the
 * rest of the pitch, so the default view shows the whole pitch.
 *
 * @module pitch
 */

/** Token names read from CSS custom properties, with fallbacks used only if a property is empty. */
const TOKEN_FALLBACK = {
  bg: '#0b0e14', panel: '#121722', hairline: '#232a38', text: '#e8ecf3', muted: '#9aa3b2',
  h: '#4f8dff', a: '#ff4d6d', d: '#8b93a7', gt: '#ffd166', baseline: '#c084fc',
  onset: '#ff8a3d', restart: '#7CFFB2', release: '#7CFFB2',
  'ev-release': '#7CFFB2', 'ev-control': '#5ad1ff', 'ev-touch': '#e8ecf3',
  'ev-contested': '#8b93a7', 'ev-dribble': '#c6f36b', teleport: '#ff3b5c',
  'pitch-1': '#0c1d18', 'pitch-2': '#0f241d', 'pitch-line': 'rgba(232,236,243,0.42)',
  'pitch-vignette': 'rgba(5,7,12,0.55)', 'dim': '5,7,12',
  'font-sans': 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
  'font-mono': 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
};

/** Layer names and their defaults. */
export const DEFAULT_LAYERS = {
  camera: false, trails: true, glow: true, dim: true, rings: true, onset: true, restart: true, touch: true,
  slotIds: true, airborne: true, ghost: true, insets: true, overlays: true,
};

/** Human names of the layers, for the toggle menu. */
export const LAYER_NAMES = {
  camera: 'Follow camera', trails: 'Trails', glow: 'Owner glow', dim: 'Dead-ball dimming', rings: 'Event rings',
  onset: 'Stoppage-start flash', restart: 'Restart pulse', touch: 'Touch-radius circles', slotIds: 'Track ids',
  airborne: 'Airborne halo', ghost: 'Raw ball position', insets: 'Signal cards', overlays: 'Annotations',
};

/**
 * Read the colour and font tokens from CSS custom properties on the root element.
 * @param {Element} [root]
 * @returns {Object<string, string>}
 */
export function readTokens(root) {
  const el = root || document.documentElement;
  const style = getComputedStyle(el);
  const out = {};
  for (const key of Object.keys(TOKEN_FALLBACK)) {
    const v = style.getPropertyValue('--' + key).trim();
    out[key] = v || TOKEN_FALLBACK[key];
  }
  return out;
}

/** Colour of an event kind. */
export function kindColor(tokens, kind) {
  return tokens['ev-' + kind] || tokens['ev-touch'];
}

/** Parse a CSS colour to [r, g, b] through a 1x1 canvas (cached). */
const rgbCache = new Map();
let probeCtx = null;
function rgbOf(color) {
  if (rgbCache.has(color)) return rgbCache.get(color);
  if (!probeCtx) {
    const c = document.createElement('canvas');
    c.width = c.height = 1;
    probeCtx = c.getContext('2d', { willReadFrequently: true });
  }
  probeCtx.clearRect(0, 0, 1, 1);
  probeCtx.fillStyle = '#000';
  probeCtx.fillStyle = color;
  probeCtx.fillRect(0, 0, 1, 1);
  const d = probeCtx.getImageData(0, 0, 1, 1).data;
  const rgb = [d[0], d[1], d[2]];
  rgbCache.set(color, rgb);
  return rgb;
}

/** `rgba()` string of a colour with an alpha. */
export function rgba(color, alpha) {
  const [r, g, b] = rgbOf(color);
  return 'rgba(' + r + ',' + g + ',' + b + ',' + Math.max(0, Math.min(1, alpha)).toFixed(3) + ')';
}

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

function finite(v) {
  return typeof v === 'number' && isFinite(v);
}

/** Margin around the pitch in metres: room for a ball beyond the line and the goals. */
const MARGIN_M = 4;

/**
 * Create a pitch renderer on a canvas. The canvas size follows its CSS box; call `resize()` when
 * the box changes (the replay does this with a ResizeObserver). `aspect` on the result is the
 * width over the height of the pitch with its surround: a box of that aspect has no empty bands.
 * @param {HTMLCanvasElement} canvas
 * @param {{pitchM: [number, number], tokens?: Object, marks?: {boxDepth?: number, boxWidth?: number},
 *          maxBackingWidth?: number}} opts
 */
export function createPitch(canvas, opts) {
  const L = opts.pitchM[0];
  const W = opts.pitchM[1];
  const marks = Object.assign({ boxDepth: 16.5, boxWidth: 40.32 }, opts.marks || {});
  const maxBacking = opts.maxBackingWidth || 3600;
  let tokens = opts.tokens || readTokens();
  const ctx = canvas.getContext('2d');
  const bgCanvas = document.createElement('canvas');
  const bgCtx = bgCanvas.getContext('2d');
  const sprites = {};

  // layout in CSS pixels
  const lay = { cssW: 0, cssH: 0, dpr: 1, s: 1, s0: 1, ox: 0, oy: 0, pw: 0, ph: 0, r: 6, font: 12, kb: 1, zoom: 1 };
  const fullW = L + 2 * MARGIN_M;
  const fullH = W + 2 * MARGIN_M;
  /** Largest camera zoom; the background is pre-rendered sharp enough for it. */
  const ZOOM_MAX = 2.6;

  /** Pitch metres to CSS pixels. */
  function X(x) { return lay.ox + (x + MARGIN_M) * lay.s; }
  function Y(y) { return lay.oy + (y + MARGIN_M) * lay.s; }

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const cssW = Math.max(1, rect.width);
    const cssH = Math.max(1, rect.height);
    let dpr = window.devicePixelRatio || 1;
    if (cssW * dpr > maxBacking) dpr = maxBacking / cssW;
    lay.cssW = cssW;
    lay.cssH = cssH;
    lay.dpr = dpr;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    lay.s0 = Math.min(cssW / fullW, cssH / fullH);
    lay.font = clamp(Math.min(cssW / 66, cssH / 38), 9, 42);
    // background resolution: sharp at the largest zoom, capped at 4096 device pixels wide
    lay.kb = Math.min(lay.s0 * ZOOM_MAX * dpr, 4096 / fullW);
    setView(null, 0);
    renderBackground();
    renderSprites();
  }

  /**
   * Place the pitch for one frame. Without a camera the whole pitch fits the canvas. With a camera
   * {x, y, zoom} the view centres on (x, y) above the signal cards and never leaves the surround.
   */
  function setView(cam, insetH) {
    const zoom = cam && cam.zoom > 1 ? Math.min(cam.zoom, ZOOM_MAX) : 1;
    const s = lay.s0 * zoom;
    lay.s = s;
    lay.zoom = zoom;
    lay.pw = fullW * s;
    lay.ph = fullH * s;
    if (zoom === 1 || !cam) {
      lay.ox = (lay.cssW - lay.pw) / 2;
      lay.oy = (lay.cssH - lay.ph) / 2;
    } else {
      let ox = lay.cssW / 2 - (cam.x + MARGIN_M) * s;
      let oy = (lay.cssH - insetH) / 2 - (cam.y + MARGIN_M) * s;
      if (lay.pw >= lay.cssW) ox = clamp(ox, lay.cssW - lay.pw, 0); else ox = (lay.cssW - lay.pw) / 2;
      const room = lay.cssH - insetH;
      if (lay.ph >= room) oy = clamp(oy, room - lay.ph, 0); else oy = (room - lay.ph) / 2;
      lay.ox = ox;
      lay.oy = oy;
    }
    lay.r = clamp(s * 0.6, 4, 28);
  }

  function insetHeight() {
    return clamp(lay.cssH * 0.3, 84, 640);
  }

  function renderBackground() {
    // draw in background pixels: temporarily map metres with the background scale
    const keep = { s: lay.s, ox: lay.ox, oy: lay.oy, pw: lay.pw, ph: lay.ph };
    lay.s = lay.kb;
    lay.ox = 0;
    lay.oy = 0;
    lay.pw = fullW * lay.kb;
    lay.ph = fullH * lay.kb;
    bgCanvas.width = Math.ceil(lay.pw);
    bgCanvas.height = Math.ceil(lay.ph);
    const g = bgCtx;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, bgCanvas.width, bgCanvas.height);
    // surround
    const x0 = X(-MARGIN_M), y0 = Y(-MARGIN_M);
    g.fillStyle = tokens['pitch-1'];
    roundRect(g, x0, y0, lay.pw, lay.ph, Math.min(14, lay.s * 1.2));
    g.fill();
    // mowing stripes, 18 bands along the length
    g.save();
    roundRect(g, x0, y0, lay.pw, lay.ph, Math.min(14, lay.s * 1.2));
    g.clip();
    const bands = 18;
    for (let i = 0; i < bands; i++) {
      if (i % 2) continue;
      g.fillStyle = tokens['pitch-2'];
      g.fillRect(X((L / bands) * i), y0, (L / bands) * lay.s, lay.ph);
    }
    // soft light from the top, like stadium floodlights
    const lg = g.createLinearGradient(0, y0, 0, y0 + lay.ph);
    lg.addColorStop(0, 'rgba(255,255,255,0.035)');
    lg.addColorStop(0.5, 'rgba(255,255,255,0)');
    lg.addColorStop(1, 'rgba(0,0,0,0.12)');
    g.fillStyle = lg;
    g.fillRect(x0, y0, lay.pw, lay.ph);
    // vignette
    const cx = X(L / 2), cy = Y(W / 2);
    const rg = g.createRadialGradient(cx, cy, lay.ph * 0.25, cx, cy, lay.pw * 0.62);
    rg.addColorStop(0, 'rgba(0,0,0,0)');
    rg.addColorStop(1, tokens['pitch-vignette']);
    g.fillStyle = rg;
    g.fillRect(x0, y0, lay.pw, lay.ph);
    g.restore();

    // lines (Law 1)
    g.strokeStyle = tokens['pitch-line'];
    g.lineWidth = Math.max(1, lay.s * 0.12);
    g.lineJoin = 'round';
    g.strokeRect(X(0), Y(0), L * lay.s, W * lay.s);
    line(g, X(L / 2), Y(0), X(L / 2), Y(W));
    circle(g, X(L / 2), Y(W / 2), 9.15 * lay.s);
    g.stroke();
    dot(g, X(L / 2), Y(W / 2), Math.max(1.5, lay.s * 0.25), tokens['pitch-line']);
    for (const side of [0, 1]) {
      const sx = side ? -1 : 1;
      const gx = side ? L : 0;
      const bd = marks.boxDepth, bw = marks.boxWidth;
      g.strokeRect(Math.min(X(gx), X(gx + sx * bd)), Y(W / 2 - bw / 2), bd * lay.s, bw * lay.s);
      g.strokeRect(Math.min(X(gx), X(gx + sx * 5.5)), Y(W / 2 - 9.16), 5.5 * lay.s, 18.32 * lay.s);
      // goal
      g.strokeRect(Math.min(X(gx), X(gx - sx * 2)), Y(W / 2 - 3.66), 2 * lay.s, 7.32 * lay.s);
      // penalty spot and arc
      const spot = gx + sx * 11;
      dot(g, X(spot), Y(W / 2), Math.max(1.2, lay.s * 0.2), tokens['pitch-line']);
      const off = Math.acos((bd - 11) / 9.15);
      g.beginPath();
      if (side === 0) g.arc(X(spot), Y(W / 2), 9.15 * lay.s, -off, off);
      else g.arc(X(spot), Y(W / 2), 9.15 * lay.s, Math.PI - off, Math.PI + off);
      g.stroke();
    }
    // corner arcs
    const cr = 1 * lay.s;
    g.beginPath(); g.arc(X(0), Y(0), cr, 0, Math.PI / 2); g.stroke();
    g.beginPath(); g.arc(X(L), Y(0), cr, Math.PI / 2, Math.PI); g.stroke();
    g.beginPath(); g.arc(X(0), Y(W), cr, -Math.PI / 2, 0); g.stroke();
    g.beginPath(); g.arc(X(L), Y(W), cr, Math.PI, Math.PI * 1.5); g.stroke();
    Object.assign(lay, keep);
  }

  function makeGlow(color, inner) {
    const c = document.createElement('canvas');
    const size = 160;
    c.width = c.height = size;
    const g = c.getContext('2d');
    const m = size / 2;
    const rg = g.createRadialGradient(m, m, 0, m, m, m);
    rg.addColorStop(0, rgba(color, inner));
    rg.addColorStop(0.35, rgba(color, inner * 0.45));
    rg.addColorStop(1, rgba(color, 0));
    g.fillStyle = rg;
    g.fillRect(0, 0, size, size);
    return { canvas: c };
  }

  function renderSprites() {
    sprites.glowH = makeGlow(tokens.h, 0.55);
    sprites.glowA = makeGlow(tokens.a, 0.55);
    sprites.ballGlow = makeGlow('#ffffff', 0.5);
    sprites.pH = makeGlow(tokens.h, 0.35);
    sprites.pA = makeGlow(tokens.a, 0.35);
  }

  /** Draw a glow sprite centred on (x, y) with radius r in CSS pixels. */
  function blit(sprite, x, y, alpha, r) {
    if (alpha <= 0.002) return;
    ctx.globalAlpha = alpha;
    ctx.drawImage(sprite.canvas, x - r, y - r, r * 2, r * 2);
    ctx.globalAlpha = 1;
  }

  function teamColor(t) {
    return t === 0 ? tokens.h : t === 1 ? tokens.a : tokens.d;
  }

  function resolveColor(c) {
    if (!c) return tokens.text;
    return tokens[c] || c;
  }

  // -- layers -------------------------------------------------------------------------------

  function drawTrails(fs) {
    const tr = fs.trail;
    if (!tr || tr.n < 1) return;
    const n = tr.n, ns = fs.nSlots;
    ctx.lineCap = 'round';
    ctx.lineWidth = Math.max(1.2, lay.r * 0.5);
    for (let team = 0; team < 2; team++) {
      const col = teamColor(team);
      for (let k = 0; k < n; k++) {
        ctx.strokeStyle = rgba(col, 0.05 + 0.4 * ((k + 1) / n));
        ctx.beginPath();
        for (let s = 0; s < ns; s++) {
          if (fs.team && fs.team[s] !== team) continue;
          const x1 = tr.px[k * ns + s], y1 = tr.py[k * ns + s];
          const x2 = k + 1 < n ? tr.px[(k + 1) * ns + s] : fs.px[s];
          const y2 = k + 1 < n ? tr.py[(k + 1) * ns + s] : fs.py[s];
          if (!(x1 === x1 && y1 === y1 && x2 === x2 && y2 === y2)) continue;
          ctx.moveTo(X(x1), Y(y1));
          ctx.lineTo(X(x2), Y(y2));
        }
        ctx.stroke();
      }
    }
  }

  function drawBallTrail(fs) {
    const bt = fs.ballTrail;
    const b = fs.ball;
    if (!bt || bt.n < 1 || !b || !b.visible) return;
    const speed = finite(b.speed) ? b.speed : 0;
    const wide = clamp(speed / 20, 0.15, 1);
    ctx.lineCap = 'round';
    for (let k = 0; k < bt.n; k++) {
      const x1 = bt.x[k], y1 = bt.y[k];
      const x2 = k + 1 < bt.n ? bt.x[k + 1] : b.x;
      const y2 = k + 1 < bt.n ? bt.y[k + 1] : b.y;
      if (!(x1 === x1 && x2 === x2)) continue;
      const f = (k + 1) / bt.n;
      ctx.strokeStyle = 'rgba(255,255,255,' + (0.05 + 0.5 * f * wide).toFixed(3) + ')';
      ctx.lineWidth = Math.max(1, lay.r * (0.25 + 0.75 * f) * (0.4 + wide));
      ctx.beginPath();
      ctx.moveTo(X(x1), Y(y1));
      ctx.lineTo(X(x2), Y(y2));
      ctx.stroke();
    }
  }

  function drawGlow(fs) {
    const g = fs.glow;
    if (!g || g.slot == null || g.slot < 0) return;
    const x = fs.px[g.slot], y = fs.py[g.slot];
    if (!(x === x && y === y)) return;
    blit(g.team === 0 ? sprites.glowH : sprites.glowA, X(x), Y(y), clamp(g.alpha == null ? 1 : g.alpha, 0, 1), lay.s * 4.5);
  }

  function drawTouch(fs, rTouch) {
    const b = fs.ball;
    if (!b || !b.visible || !finite(b.x) || !finite(rTouch)) return;
    const bx = X(b.x), by = Y(b.y);
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = 'rgba(232,236,243,0.55)';
    circle(ctx, bx, by, rTouch * lay.s);
    ctx.stroke();
    ctx.setLineDash([]);
    if (!fs.nearest) return;
    for (let team = 0; team < 2; team++) {
      const nr = fs.nearest[team];
      if (!nr || nr.slot == null || nr.slot < 0 || !finite(nr.d)) continue;
      const x = fs.px[nr.slot], y = fs.py[nr.slot];
      if (!(x === x)) continue;
      const inside = nr.d <= rTouch;
      const col = teamColor(team);
      ctx.strokeStyle = rgba(col, inside ? 0.95 : 0.45);
      ctx.lineWidth = inside ? 2 : 1.2;
      ctx.setLineDash(inside ? [] : [3, 4]);
      ctx.beginPath();
      ctx.moveTo(bx, by);
      ctx.lineTo(X(x), Y(y));
      ctx.stroke();
      ctx.setLineDash([]);
      const mx = (bx + X(x)) / 2, my = (by + Y(y)) / 2;
      pill(ctx, mx, my - lay.font * 0.9, nr.d.toFixed(2) + ' m', col, inside, lay.font * 0.82);
    }
  }

  function drawPlayers(fs, layers) {
    const ns = fs.nSlots;
    const r = lay.r;
    // soft under-glow
    for (let s = 0; s < ns; s++) {
      const x = fs.px[s], y = fs.py[s];
      if (!(x === x && y === y)) continue;
      const t = fs.team ? fs.team[s] : 255;
      if (t === 0 || t === 1) blit(t === 0 ? sprites.pH : sprites.pA, X(x), Y(y), 0.9, r * 2.2);
    }
    ctx.lineWidth = Math.max(1, r * 0.22);
    for (let s = 0; s < ns; s++) {
      const x = fs.px[s], y = fs.py[s];
      if (!(x === x && y === y)) continue;
      const t = fs.team ? fs.team[s] : 255;
      const cx = X(x), cy = Y(y);
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = teamColor(t);
      ctx.fill();
      ctx.strokeStyle = 'rgba(8,10,16,0.85)';
      ctx.stroke();
    }
    if (layers.slotIds && r >= 6.5) {
      ctx.font = '700 ' + Math.round(r * 0.95) + 'px ' + tokens['font-mono'];
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = 'rgba(8,10,16,0.92)';
      for (let s = 0; s < ns; s++) {
        const x = fs.px[s], y = fs.py[s];
        if (!(x === x && y === y)) continue;
        ctx.fillText(String(s), X(x), Y(y) + 0.5);
      }
    }
  }

  function drawBall(fs, layers) {
    const b = fs.ball;
    if (!b) return;
    if (layers.ghost && finite(b.rawX) && b.rawVisible !== false) {
      const rx = X(b.rawX), ry = Y(b.rawY);
      const far = b.visible && finite(b.x) ? Math.hypot(b.rawX - b.x, b.rawY - b.y) : Infinity;
      if (far > 0.25) {
        const col = b.teleport ? tokens.teleport : 'rgba(232,236,243,0.7)';
        ctx.strokeStyle = col;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([2, 3]);
        circle(ctx, rx, ry, lay.r * 0.75);
        ctx.stroke();
        if (b.visible && finite(b.x) && far < 40) {
          ctx.beginPath();
          ctx.moveTo(rx, ry);
          ctx.lineTo(X(b.x), Y(b.y));
          ctx.stroke();
        }
        ctx.setLineDash([]);
        if (b.teleport) {
          const k = lay.r * 0.9;
          ctx.lineWidth = 2.2;
          ctx.strokeStyle = tokens.teleport;
          ctx.beginPath();
          ctx.moveTo(rx - k, ry - k); ctx.lineTo(rx + k, ry + k);
          ctx.moveTo(rx + k, ry - k); ctx.lineTo(rx - k, ry + k);
          ctx.stroke();
        }
      }
    }
    if (!b.visible || !finite(b.x)) return;
    const bx = X(b.x), by = Y(b.y);
    const rb = Math.max(2.6, lay.r * 0.55);
    blit(sprites.ballGlow, bx, by, 0.9, Math.max(lay.r * 2.2, lay.s * 1.4));
    if (layers.airborne && b.airborne) {
      // shadow offset and a halo: a ball in the air is metres from its ground projection
      ctx.fillStyle = 'rgba(0,0,0,0.45)';
      ctx.beginPath();
      ctx.ellipse(bx + rb * 0.9, by + rb * 1.4, rb * 1.1, rb * 0.6, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.7)';
      ctx.lineWidth = 1.2;
      ctx.setLineDash([2, 2]);
      circle(ctx, bx, by, rb * 2.4);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.beginPath();
    ctx.arc(bx, by, rb, 0, Math.PI * 2);
    ctx.fillStyle = b.ghost ? 'rgba(255,255,255,0.45)' : '#ffffff';
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(8,10,16,0.9)';
    ctx.stroke();
  }

  function drawRings(fs, ringFrames) {
    if (!fs.events) return;
    const span = ringFrames || 12;
    for (const ev of fs.events) {
      if (!finite(ev.x) || Math.abs(ev.dt) > span) continue;
      const col = kindColor(tokens, ev.kind);
      const cx = X(ev.x), cy = Y(ev.y);
      if (ev.dt >= 0) {
        const f = ev.dt / span;
        const rr = (0.8 + f * 3.4) * lay.s;
        ctx.strokeStyle = rgba(col, 1 - f);
        ctx.lineWidth = Math.max(1.5, lay.r * 0.35 * (1 - f) + 1);
        circle(ctx, cx, cy, rr);
        ctx.stroke();
        ctx.strokeStyle = rgba(col, 0.5 * (1 - f));
        ctx.lineWidth = 1;
        circle(ctx, cx, cy, rr * 0.6);
        ctx.stroke();
        const label = ev.label || ev.kind;
        const tcol = ev.team === 'H' || ev.team === 0 ? tokens.h : ev.team === 'A' || ev.team === 1 ? tokens.a : col;
        tag(ctx, cx, cy - rr - lay.font * 0.9, label, col, tcol, 1 - f * 0.7, lay.font);
      } else {
        const f = 1 + ev.dt / span;
        ctx.strokeStyle = rgba(col, 0.35 * f);
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 3]);
        circle(ctx, cx, cy, (4.2 - 3.4 * f) * lay.s);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }

  function nearestEdge(x, y) {
    const d = [[y, 'top'], [W - y, 'bottom'], [x, 'left'], [L - x, 'right']];
    d.sort((p, q) => p[0] - q[0]);
    return d[0][1];
  }

  function drawEdgeFlash(edge, along, alpha, color, spanM) {
    const half = (spanM || 18) / 2;
    ctx.save();
    ctx.lineCap = 'round';
    for (const [w, a] of [[lay.s * 1.6, 0.18], [lay.s * 0.7, 0.45], [Math.max(2, lay.s * 0.22), 1]]) {
      ctx.strokeStyle = rgba(color, a * alpha);
      ctx.lineWidth = w;
      ctx.beginPath();
      if (edge === 'top' || edge === 'bottom') {
        const yy = edge === 'top' ? 0 : W;
        ctx.moveTo(X(clamp(along - half, 0, L)), Y(yy));
        ctx.lineTo(X(clamp(along + half, 0, L)), Y(yy));
      } else {
        const xx = edge === 'left' ? 0 : L;
        ctx.moveTo(X(xx), Y(clamp(along - half, 0, W)));
        ctx.lineTo(X(xx), Y(clamp(along + half, 0, W)));
      }
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawOnsets(fs, span) {
    if (!fs.onsets) return;
    for (const on of fs.onsets) {
      if (Math.abs(on.dt) > span * 2 || on.dt < -2) continue;
      const alpha = clamp(1 - on.dt / (span * 2), 0, 1);
      if (on.mechanism === 'out' && finite(on.x)) {
        const edge = on.edge || nearestEdge(on.x, on.y);
        const along = edge === 'top' || edge === 'bottom' ? on.x : on.y;
        drawEdgeFlash(edge, along, alpha, tokens.onset, 22);
      } else {
        // a stop is not at a line: flash the whole pitch outline faintly
        ctx.strokeStyle = rgba(tokens.onset, 0.5 * alpha);
        ctx.lineWidth = Math.max(2, lay.s * 0.35);
        ctx.strokeRect(X(0), Y(0), L * lay.s, W * lay.s);
      }
      if (finite(on.x)) {
        tag(ctx, X(clamp(on.x, 0, L)), Y(clamp(on.y, 0, W)) - lay.font * 2.4,
          'onset: ' + on.mechanism.replace('_', ' '), tokens.onset, tokens.onset, alpha, lay.font);
      }
    }
  }

  function drawRestarts(fs, span) {
    if (!fs.restarts) return;
    for (const rs of fs.restarts) {
      if (!finite(rs.x) || rs.dt < 0 || rs.dt > span * 2) continue;
      const f = rs.dt / (span * 2);
      const col = tokens.restart;
      const cx = X(rs.x), cy = Y(rs.y);
      for (let k = 0; k < 3; k++) {
        const ff = (f + k / 3) % 1;
        ctx.strokeStyle = rgba(col, (1 - ff) * (1 - f * 0.5));
        ctx.lineWidth = 2;
        ctx.setLineDash(rs.confirmed === false ? [4, 4] : []);
        circle(ctx, cx, cy, (0.6 + ff * 5) * lay.s);
        ctx.stroke();
      }
      ctx.setLineDash([]);
      const text = 'restart: ' + rs.kind + (rs.confirmed === false ? ' (unconfirmed)' : '');
      tag(ctx, cx, cy + lay.s * 6 + lay.font, text, col, col, 1 - f * 0.6, lay.font);
    }
  }

  function drawDim(fs) {
    const p = clamp(fs.pDead || 0, 0, 1);
    if (p > 0.01) {
      ctx.fillStyle = 'rgba(' + tokens.dim + ',' + (0.62 * p).toFixed(3) + ')';
      ctx.fillRect(0, 0, lay.cssW, lay.cssH);
    }
  }

  function drawBadge(fs) {
    const p = clamp(fs.pDead || 0, 0, 1);
    const text = fs.badge || (fs.stoppage || p >= 0.5 ? 'D  dead ball  p ' + p.toFixed(2) : null);
    if (!text) return;
    const size = lay.font * 1.05;
    ctx.font = '700 ' + size + 'px ' + tokens['font-mono'];
    const w = ctx.measureText(text).width + size * 1.4;
    const x = lay.cssW - w - 8, y = size * 3.4;
    ctx.fillStyle = 'rgba(11,14,20,0.82)';
    roundRect(ctx, x, y, w, size * 1.9, size * 0.5);
    ctx.fill();
    ctx.strokeStyle = rgba(tokens.d, 0.9);
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = tokens.text;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, x + size * 0.7, y + size * 0.95);
  }

  // -- overlays -----------------------------------------------------------------------------

  function drawOverlay(o, fs) {
    const alpha = o.alpha == null ? 1 : o.alpha;
    if (alpha <= 0.01) return;
    const col = resolveColor(o.color);
    ctx.setLineDash(o.dash || []);
    ctx.lineWidth = o.width || 1.6;
    ctx.strokeStyle = rgba(col, alpha);
    switch (o.type) {
      case 'circle': {
        let x = o.x, y = o.y;
        if (o.slot != null) { x = fs.px[o.slot]; y = fs.py[o.slot]; }
        if (!finite(x)) break;
        circle(ctx, X(x), Y(y), o.r * lay.s);
        if (o.fill) { ctx.fillStyle = rgba(resolveColor(o.fill), (o.fillAlpha || 0.15) * alpha); ctx.fill(); }
        ctx.stroke();
        if (o.label) tag(ctx, X(x), Y(y) - o.r * lay.s - lay.font * 0.9, o.label, col, col, alpha, lay.font);
        break;
      }
      case 'progress': {
        let x = o.x, y = o.y;
        if (o.slot != null) { x = fs.px[o.slot]; y = fs.py[o.slot]; }
        if (!finite(x)) break;
        const rr = (o.r || 1.6) * lay.s;
        ctx.setLineDash([]);
        ctx.lineWidth = Math.max(2, lay.r * 0.4);
        ctx.strokeStyle = rgba(col, 0.2 * alpha);
        circle(ctx, X(x), Y(y), rr);
        ctx.stroke();
        ctx.strokeStyle = rgba(col, alpha);
        ctx.beginPath();
        ctx.arc(X(x), Y(y), rr, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * clamp(o.frac, 0, 1));
        ctx.stroke();
        if (o.label) tag(ctx, X(x), Y(y) + rr + lay.font * 1.1, o.label, col, col, alpha, lay.font);
        break;
      }
      case 'line': {
        ctx.beginPath();
        ctx.moveTo(X(o.x1), Y(o.y1));
        ctx.lineTo(X(o.x2), Y(o.y2));
        ctx.stroke();
        if (o.label) tag(ctx, (X(o.x1) + X(o.x2)) / 2, (Y(o.y1) + Y(o.y2)) / 2 - lay.font, o.label, col, col, alpha, lay.font);
        break;
      }
      case 'path': {
        const p = o.pts;
        if (!p || p.length < 4) break;
        ctx.beginPath();
        ctx.moveTo(X(p[0]), Y(p[1]));
        for (let i = 2; i < p.length; i += 2) ctx.lineTo(X(p[i]), Y(p[i + 1]));
        ctx.stroke();
        if (o.label) {
          const i = p.length - 2;
          tag(ctx, X(p[i]), Y(p[i + 1]) - lay.font * 1.2, o.label, col, col, alpha, lay.font);
        }
        break;
      }
      case 'cross': {
        const k = (o.size || 0.8) * lay.s;
        const cx = X(o.x), cy = Y(o.y);
        ctx.setLineDash([]);
        ctx.lineWidth = o.width || 2.4;
        ctx.beginPath();
        ctx.moveTo(cx - k, cy - k); ctx.lineTo(cx + k, cy + k);
        ctx.moveTo(cx + k, cy - k); ctx.lineTo(cx - k, cy + k);
        ctx.stroke();
        if (o.label) tag(ctx, cx, cy - k - lay.font, o.label, col, col, alpha, lay.font);
        break;
      }
      case 'label': {
        tag(ctx, X(o.x), Y(o.y), o.text, col, o.textColor ? resolveColor(o.textColor) : col, alpha, lay.font * (o.scale || 1));
        break;
      }
      case 'edge': {
        drawEdgeFlash(o.edge, o.along, alpha, col, o.span);
        break;
      }
      case 'zone': {
        ctx.fillStyle = rgba(col, (o.fillAlpha || 0.08) * alpha);
        ctx.fillRect(X(o.x0), Y(o.y0), (o.x1 - o.x0) * lay.s, (o.y1 - o.y0) * lay.s);
        if (o.stroke) { ctx.strokeRect(X(o.x0), Y(o.y0), (o.x1 - o.x0) * lay.s, (o.y1 - o.y0) * lay.s); }
        if (o.label) tag(ctx, X((o.x0 + o.x1) / 2), Y(o.y0) + lay.font * 1.2, o.label, col, col, alpha, lay.font);
        break;
      }
      default:
        break;
    }
    ctx.setLineDash([]);
  }

  // -- insets -------------------------------------------------------------------------------

  /** Cut a text with an ellipsis to fit a width, in the current font. */
  function fit(text, maxW) {
    if (ctx.measureText(text).width <= maxW) return text;
    let t = text;
    while (t.length > 1 && ctx.measureText(t + '\u2026').width > maxW) t = t.slice(0, -1);
    return t + '\u2026';
  }

  function card(x, y, w, h, title) {
    ctx.fillStyle = 'rgba(11,14,20,0.84)';
    roundRect(ctx, x, y, w, h, 8);
    ctx.fill();
    ctx.strokeStyle = 'rgba(232,236,243,0.12)';
    ctx.lineWidth = 1;
    ctx.stroke();
    if (title) {
      ctx.font = '600 ' + (lay.font * 0.82) + 'px ' + tokens['font-sans'];
      ctx.fillStyle = tokens.muted;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(title, x + 10, y + 7);
    }
    return { x: x + 10, y: y + 10 + lay.font * 1.1, w: w - 20, h: h - 18 - lay.font * 1.1 };
  }

  function insetSpark(inset, r) {
    const n = inset.n;
    const yMax = inset.yMax || 1;
    const px = (i) => r.x + (i / Math.max(1, n - 1)) * r.w;
    const py = (v) => r.y + r.h - clamp(v / yMax, 0, 1.05) * r.h;
    ctx.font = (lay.font * 0.72) + 'px ' + tokens['font-mono'];
    if (inset.brackets) {
      for (const b of inset.brackets) {
        const col = resolveColor(b.color);
        ctx.fillStyle = rgba(col, 0.12);
        ctx.fillRect(px(b.i0), r.y, px(b.i1) - px(b.i0), r.h);
        ctx.strokeStyle = rgba(col, 0.8);
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(px(b.i0), r.y + 3); ctx.lineTo(px(b.i0), r.y); ctx.lineTo(px(b.i1), r.y); ctx.lineTo(px(b.i1), r.y + 3);
        ctx.stroke();
        if (b.label) {
          ctx.fillStyle = col;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'top';
          ctx.fillText(b.label, (px(b.i0) + px(b.i1)) / 2, r.y + 3);
        }
      }
    }
    for (const th of [].concat(inset.threshold || [])) {
      const col = resolveColor(th.color || 'onset');
      ctx.strokeStyle = rgba(col, 0.85);
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(r.x, py(th.value));
      ctx.lineTo(r.x + r.w, py(th.value));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = col;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'bottom';
      ctx.fillText(th.label, r.x + r.w, py(th.value) - 2);
    }
    const upto = inset.reveal === false ? n : Math.min(n, (inset.cursor == null ? n - 1 : inset.cursor) + 1);
    let legendX = 0;
    for (const s of inset.series) {
      const col = resolveColor(s.color);
      const v = s.values;
      if (s.fill) {
        ctx.fillStyle = rgba(col, 0.14);
        ctx.beginPath();
        ctx.moveTo(px(0), py(0));
        for (let i = 0; i < upto; i++) ctx.lineTo(px(i), py(finite(v[i]) ? v[i] : 0));
        ctx.lineTo(px(upto - 1), py(0));
        ctx.closePath();
        ctx.fill();
      }
      ctx.strokeStyle = col;
      ctx.lineWidth = s.width || 1.6;
      ctx.beginPath();
      let pen = false;
      for (let i = 0; i < upto; i++) {
        if (!finite(v[i])) { pen = false; continue; }
        if (!pen) { ctx.moveTo(px(i), py(v[i])); pen = true; } else ctx.lineTo(px(i), py(v[i]));
      }
      ctx.stroke();
      if (s.label) {
        // legend along the bottom edge of the plot, one entry after the other
        ctx.font = (lay.font * 0.72) + 'px ' + tokens['font-mono'];
        ctx.fillStyle = col;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'bottom';
        const lx = r.x + 2 + legendX;
        ctx.fillText(s.label, lx, r.y + r.h - 1);
        legendX += ctx.measureText(s.label).width + lay.font;
      }
    }
    if (inset.markers) {
      for (const m of inset.markers) {
        if (m.i > upto - 1 && inset.reveal !== false) continue;
        const col = resolveColor(m.color);
        const x = px(m.i);
        ctx.strokeStyle = rgba(col, 0.9);
        ctx.setLineDash(m.dashed ? [2, 3] : []);
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(x, r.y + r.h);
        ctx.lineTo(x, finite(m.v) ? py(m.v) : r.y);
        ctx.stroke();
        ctx.setLineDash([]);
        if (finite(m.v)) {
          ctx.fillStyle = col;
          ctx.beginPath();
          ctx.arc(x, py(m.v), 3.2, 0, Math.PI * 2);
          ctx.fill();
        }
        if (m.label) {
          ctx.fillStyle = col;
          ctx.textAlign = x > r.x + r.w * 0.8 ? 'right' : 'left';
          ctx.textBaseline = 'bottom';
          const ly = finite(m.v) ? Math.max(r.y + lay.font * 0.8, py(m.v) - 4) : r.y + lay.font * 0.8;
          ctx.fillText(m.label, x + (ctx.textAlign === 'left' ? 4 : -4), ly);
        }
      }
    }
    if (inset.cursor != null) {
      const x = px(inset.cursor);
      ctx.strokeStyle = 'rgba(232,236,243,0.65)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, r.y);
      ctx.lineTo(x, r.y + r.h);
      ctx.stroke();
      if (inset.readout) {
        ctx.font = '600 ' + (lay.font * 0.8) + 'px ' + tokens['font-mono'];
        ctx.fillStyle = tokens.text;
        ctx.textAlign = 'right';
        ctx.textBaseline = 'top';
        ctx.fillText(inset.readout, r.x + r.w, r.y - lay.font * 1.05);
      }
    }
  }

  function insetBars(inset, r) {
    const bars = inset.bars;
    const max = inset.max || 1;
    const bw = r.w / bars.length;
    const py = (v) => r.y + r.h - lay.font - clamp(v / max, 0, 1) * (r.h - lay.font * 1.2);
    ctx.font = (lay.font * 0.72) + 'px ' + tokens['font-mono'];
    bars.forEach((b, i) => {
      const x = r.x + i * bw + bw * 0.18;
      const w = bw * 0.64;
      const col = resolveColor(b.color);
      const top = py(finite(b.value) ? b.value : 0);
      ctx.fillStyle = rgba(col, b.dim ? 0.35 : 0.9);
      ctx.fillRect(x, top, w, r.y + r.h - lay.font - top);
      ctx.fillStyle = tokens.muted;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(b.label, x + w / 2, r.y + r.h);
      if (finite(b.value)) {
        ctx.fillStyle = tokens.text;
        ctx.fillText(b.value.toFixed(1), x + w / 2, top - 2);
      }
    });
    if (inset.threshold) {
      const th = inset.threshold;
      const col = resolveColor(th.color || 'onset');
      ctx.strokeStyle = col;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(r.x, py(th.value));
      ctx.lineTo(r.x + r.w, py(th.value));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = col;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'bottom';
      ctx.fillText(th.label, r.x + r.w, py(th.value) - 2);
    }
    if (inset.mean) {
      const m = inset.mean;
      ctx.strokeStyle = resolveColor(m.color || 'text');
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(r.x, py(m.value));
      ctx.lineTo(r.x + r.w, py(m.value));
      ctx.stroke();
      ctx.fillStyle = resolveColor(m.color || 'text');
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText(m.label, r.x, py(m.value) - 2);
    }
  }

  function insetLanes(inset, r) {
    const n = inset.n;
    const rows = inset.rows;
    const costH = inset.cost ? r.h * 0.3 : 0;
    const laneH = (r.h - costH - 4) / rows.length;
    ctx.font = '600 ' + (lay.font * 0.72) + 'px ' + tokens['font-mono'];
    // the label gutter fits the widest row label, and is never narrower than 28 px
    const gut = Math.max(28, Math.ceil(Math.max(...rows.map((row) => ctx.measureText(row.label).width))) + 6);
    const px = (i) => r.x + gut + (i / Math.max(1, n - 1)) * (r.w - gut);
    const cw = Math.max(1, (r.w - gut) / n + 0.6);
    rows.forEach((row, k) => {
      const y = r.y + k * laneH;
      const col = resolveColor(row.color);
      ctx.fillStyle = col;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(row.label, r.x, y + laneH / 2);
      for (let i = 0; i < n; i++) {
        const v = row.values[i];
        if (!finite(v) || v <= 0.02) continue;
        ctx.fillStyle = rgba(col, 0.08 + 0.8 * clamp(v, 0, 1));
        ctx.fillRect(px(i), y + 1, cw, laneH - 2);
      }
    });
    if (inset.path) {
      // the decoded path as a stepped line through the lane centres
      ctx.strokeStyle = tokens.text;
      ctx.lineWidth = 2;
      ctx.beginPath();
      let prev = -1;
      for (let i = 0; i < n; i++) {
        const st = inset.path[i];
        if (st < 0 || st > 2) { prev = -1; continue; }
        const y = r.y + st * laneH + laneH / 2;
        if (prev < 0) ctx.moveTo(px(i), y);
        else if (prev !== st) { ctx.lineTo(px(i), r.y + prev * laneH + laneH / 2); ctx.lineTo(px(i), y); }
        else ctx.lineTo(px(i) + cw, y);
        prev = st;
      }
      ctx.stroke();
    }
    if (inset.cost) {
      const y0 = r.y + r.h - costH;
      const max = inset.costMax || 8;
      ctx.fillStyle = tokens.muted;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText('cost', r.x, y0);
      ctx.strokeStyle = tokens.onset;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const v = inset.cost[i];
        const y = y0 + (1 - clamp(v / max, 0, 1)) * (costH - 2);
        if (i === 0) ctx.moveTo(px(i), y); else ctx.lineTo(px(i), y);
      }
      ctx.stroke();
      if (inset.costLabel) {
        ctx.fillStyle = tokens.onset;
        ctx.textAlign = 'right';
        ctx.fillText(inset.costLabel, r.x + r.w, y0);
      }
    }
    if (inset.markers) {
      for (const m of inset.markers) {
        ctx.strokeStyle = rgba(resolveColor(m.color), 0.8);
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(px(m.i), r.y);
        ctx.lineTo(px(m.i), r.y + r.h);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    if (inset.cursor != null) {
      ctx.strokeStyle = 'rgba(232,236,243,0.85)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px(inset.cursor), r.y - 2);
      ctx.lineTo(px(inset.cursor), r.y + r.h);
      ctx.stroke();
    }
  }

  function insetChecklist(inset, r) {
    const size = lay.font * 0.8;
    ctx.font = size + 'px ' + tokens['font-sans'];
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    inset.items.forEach((it, k) => {
      const y = r.y + size * 0.7 + k * size * 1.55;
      if (y > r.y + r.h) return;
      const col = it.ok === true ? tokens.restart : it.ok === false ? tokens.a : tokens.muted;
      ctx.strokeStyle = col;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(r.x + size * 0.5, y, size * 0.45, 0, Math.PI * 2);
      ctx.stroke();
      if (it.ok === true) {
        ctx.beginPath();
        ctx.moveTo(r.x + size * 0.25, y);
        ctx.lineTo(r.x + size * 0.45, y + size * 0.22);
        ctx.lineTo(r.x + size * 0.78, y - size * 0.22);
        ctx.stroke();
      } else if (it.ok === false) {
        ctx.beginPath();
        ctx.moveTo(r.x + size * 0.3, y - size * 0.2); ctx.lineTo(r.x + size * 0.7, y + size * 0.2);
        ctx.moveTo(r.x + size * 0.7, y - size * 0.2); ctx.lineTo(r.x + size * 0.3, y + size * 0.2);
        ctx.stroke();
      }
      ctx.fillStyle = it.ok == null ? tokens.muted : tokens.text;
      ctx.fillText(fit(it.text, r.w - size * 1.4), r.x + size * 1.4, y);
    });
  }

  function insetTerms(inset, r) {
    const size = lay.font * 0.78;
    ctx.font = size + 'px ' + tokens['font-mono'];
    ctx.textBaseline = 'middle';
    inset.terms.forEach((t, k) => {
      const y = r.y + size * 0.7 + k * size * 1.45;
      ctx.fillStyle = t.active ? tokens.text : rgba(tokens.muted, 0.6);
      ctx.textAlign = 'left';
      ctx.fillText(t.label, r.x, y);
      ctx.textAlign = 'right';
      ctx.fillText((t.value >= 0 ? '+' : '') + t.value.toFixed(2), r.x + r.w, y);
    });
    if (inset.result) {
      const y = r.y + r.h - size * 0.6;
      ctx.fillStyle = resolveColor(inset.result.color || 'text');
      ctx.font = '700 ' + size * 1.05 + 'px ' + tokens['font-mono'];
      ctx.textAlign = 'left';
      ctx.fillText(inset.result.label, r.x, y);
      ctx.textAlign = 'right';
      ctx.fillText(inset.result.value, r.x + r.w, y);
    }
  }

  function drawInsets(fs) {
    const list = fs.insets;
    if (!list || !list.length) return;
    const pad = 8;
    const areaX = pad;
    const areaW = lay.cssW - 2 * pad;
    const h = insetHeight();
    // a canvas that is not laid out yet has no room for the cards
    if (areaW < 4 * pad || lay.cssH < h + 2 * pad) return;
    const y = lay.cssH - h - pad;
    const weights = list.map((it) => it.weight || (it.type === 'checklist' || it.type === 'terms' ? 0.7 : 1));
    const total = weights.reduce((a, b) => a + b, 0);
    const maxW = Math.min(areaW, lay.font * 30 * total);
    let x = areaX + (areaW - maxW) / 2;
    const gap = 8;
    list.forEach((it, k) => {
      const w = (maxW - gap * (list.length - 1)) * weights[k] / total;
      const r = card(x, y, w, h, it.title);
      if (it.type === 'spark') insetSpark(it, r);
      else if (it.type === 'bars') insetBars(it, r);
      else if (it.type === 'lanes') insetLanes(it, r);
      else if (it.type === 'checklist') insetChecklist(it, r);
      else if (it.type === 'terms') insetTerms(it, r);
      x += w + gap;
    });
  }

  // -- main ---------------------------------------------------------------------------------

  /**
   * Draw one frame.
   * @param {Object} fs  the frame state, see the module docstring
   * @param {Object} [layers]  layer switches, see DEFAULT_LAYERS
   * @param {{rTouch?: number, ringFrames?: number}} [params]  constants from the config
   * @returns {number} draw time in milliseconds
   */
  function draw(fs, layers, params) {
    const t0 = performance.now();
    const ly = Object.assign({}, DEFAULT_LAYERS, layers || {});
    const pr = params || {};
    const insetH = ly.insets && fs && fs.insets && fs.insets.length ? insetHeight() + 8 : 0;
    setView(ly.camera && fs ? fs.camera : null, insetH);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = tokens.bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(bgCanvas, lay.ox * lay.dpr, lay.oy * lay.dpr, lay.pw * lay.dpr, lay.ph * lay.dpr);
    ctx.setTransform(lay.dpr, 0, 0, lay.dpr, 0, 0);
    if (!fs) return performance.now() - t0;
    const span = pr.ringFrames || 12;
    if (ly.onset) drawOnsets(fs, span);
    if (ly.glow) drawGlow(fs);
    if (ly.trails) { drawTrails(fs); drawBallTrail(fs); }
    if (ly.touch) drawTouch(fs, pr.rTouch);
    if (ly.overlays && fs.overlays) for (const o of fs.overlays) if (!o.top) drawOverlay(o, fs);
    drawPlayers(fs, ly);
    drawBall(fs, ly);
    if (ly.dim) drawDim(fs);
    if (ly.rings) drawRings(fs, span);
    if (ly.restart) drawRestarts(fs, span);
    if (ly.overlays && fs.overlays) for (const o of fs.overlays) if (o.top) drawOverlay(o, fs);
    drawBadge(fs);
    if (ly.insets) drawInsets(fs);
    return performance.now() - t0;
  }

  function setTokens(t) {
    tokens = t || readTokens();
    rgbCache.clear();
    renderBackground();
    renderSprites();
  }

  /** CSS-pixel position of a pitch point, and back. */
  function toScreen(x, y) { return [X(x), Y(y)]; }
  function toPitch(sx, sy) { return [(sx - lay.ox) / lay.s - MARGIN_M, (sy - lay.oy) / lay.s - MARGIN_M]; }

  resize();
  return { draw, resize, setTokens, toScreen, toPitch, layout: lay, pitchM: [L, W], aspect: fullW / fullH,
    get tokens() { return tokens; } };
}

// -- small drawing helpers ------------------------------------------------------------------

function roundRect(g, x, y, w, h, r) {
  // a box narrower than its corners, on a canvas that is not laid out yet, gives a radius of 0
  const rr = Math.max(0, Math.min(r, w / 2, h / 2));
  g.beginPath();
  g.moveTo(x + rr, y);
  g.arcTo(x + w, y, x + w, y + h, rr);
  g.arcTo(x + w, y + h, x, y + h, rr);
  g.arcTo(x, y + h, x, y, rr);
  g.arcTo(x, y, x + w, y, rr);
  g.closePath();
}

function line(g, x1, y1, x2, y2) {
  g.beginPath();
  g.moveTo(x1, y1);
  g.lineTo(x2, y2);
  g.stroke();
}

function circle(g, x, y, r) {
  g.beginPath();
  g.arc(x, y, Math.max(0.5, r), 0, Math.PI * 2);
}

function dot(g, x, y, r, color) {
  g.fillStyle = color;
  g.beginPath();
  g.arc(x, y, r, 0, Math.PI * 2);
  g.fill();
}

/** Text label on a dark pill. */
function tag(g, x, y, text, border, color, alpha, size) {
  if (alpha <= 0.02 || !text) return;
  const fs = Math.round(size * 0.82);
  g.font = '600 ' + fs + 'px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
  const w = g.measureText(text).width + fs * 1.0;
  const h = fs * 1.6;
  g.globalAlpha = alpha;
  g.fillStyle = 'rgba(11,14,20,0.86)';
  roundRect(g, x - w / 2, y - h / 2, w, h, h / 2);
  g.fill();
  g.strokeStyle = border;
  g.lineWidth = 1;
  g.stroke();
  g.fillStyle = color;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(text, x, y + 0.5);
  g.globalAlpha = 1;
}

/** Distance label pill, filled when inside the radius. */
function pill(g, x, y, text, color, filled, size) {
  const fs = Math.round(size);
  g.font = '600 ' + fs + 'px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
  const w = g.measureText(text).width + fs * 0.9;
  const h = fs * 1.5;
  g.fillStyle = filled ? color : 'rgba(11,14,20,0.82)';
  roundRect(g, x - w / 2, y - h / 2, w, h, h / 2);
  g.fill();
  g.strokeStyle = color;
  g.lineWidth = 1;
  g.stroke();
  g.fillStyle = filled ? 'rgba(8,10,16,0.95)' : color;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(text, x, y + 0.5);
}
