/* The transit stage: a starfield, a planet orbiting its star, and the light curve drawing itself.
   Everything is derived from the analysed star: period, dip depth, dip length and secondary eclipse.
   The planet is drawn larger than life so the dip can be seen. */
(() => {
  'use strict';

  const TAU = Math.PI * 2;
  const LOOP_SECONDS = 13;
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');

  const stageEl = document.getElementById('stage');
  const sky = document.getElementById('sky');
  const cv = document.getElementById('transit');
  const toggle = document.getElementById('stage-toggle');
  if (!stageEl || !sky || !cv) return;
  const sctx = sky.getContext('2d');
  const ctx = cv.getContext('2d');

  const state = {
    p: null,            // parameters derived from the report
    g: null,            // geometry derived from the canvas size
    u: 0,               // loop position 0..1
    playing: !reduceMotion.matches,
    visible: true,
    last: 0,
    stars: [],
  };

  /* ------------------------------------------------------------------ maths */
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const smooth = (x) => { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); };

  /** Area of the overlap of two discs (radii R and r, centres d apart). */
  function overlap(d, R, r) {
    if (d >= R + r) return 0;
    if (d <= Math.abs(R - r)) return Math.PI * Math.min(R, r) ** 2;
    const a = R * R * Math.acos((d * d + R * R - r * r) / (2 * d * R));
    const b = r * r * Math.acos((d * d + r * r - R * R) / (2 * d * r));
    const c = 0.5 * Math.sqrt((-d + R + r) * (d + R - r) * (d - R + r) * (d + R + r));
    return a + b - c;
  }

  /* --------------------------------------------------------------- parameters */
  function paramsFrom(report) {
    const c = report.candidate, f = report.features;
    const depth = c.depth_ppm / 1e6;
    const sec = clamp(f.secondary_ratio || 0, 0, 1.5);
    return {
      period: c.period_days,
      depthPpm: c.depth_ppm,
      ratio: clamp(Math.sqrt(depth) * 3.2, 0.075, 0.42),  // exaggerated planet / star radius
      secondary: sec,
      luminous: sec >= 0.2,   // a companion that shines: its secondary eclipse is visible
      name: report.target.name,
    };
  }

  /** Geometry that depends on the canvas size, plus the dip curve and the time warp. */
  function layout() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const box = cv.parentElement.getBoundingClientRect();
    const W = Math.max(280, Math.round(box.width));
    const H = W < 520 ? 330 : 400;
    for (const c of [cv]) {
      c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
      c.style.width = `${W}px`; c.style.height = `${H}px`;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const sb = stageEl.getBoundingClientRect();
    sky.width = Math.round(sb.width * dpr); sky.height = Math.round(sb.height * dpr);
    sky.style.width = `${sb.width}px`; sky.style.height = `${sb.height}px`;
    sctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    state.skyW = sb.width; state.skyH = sb.height;
    if (!state.stars.length || state.stars.length < Math.round(sb.width * sb.height / 7000)) seedStars(sb.width, sb.height);

    const sceneH = H - (W < 520 ? 118 : 134);
    const Rs = Math.min(W * 0.15, sceneH * 0.3, 92);
    const g = { W, H, sceneH, Rs, cx: W / 2, cy: sceneH * 0.52, A: W * 0.43, B: Rs * 0.2,
      curveTop: sceneH + 22, curveH: H - sceneH - 52, curveL: 14, curveR: W - 14 };
    state.g = g;
    if (state.p) { buildCurve(); buildWarp(); }
  }

  function buildCurve() {
    const { p, g } = state;
    const rp = g.Rs * p.ratio;
    const N = 720, vals = new Float32Array(N + 1);
    for (let i = 0; i <= N; i++) {
      const th = -Math.PI + (i / N) * TAU;
      const z = Math.cos(th);
      const d = Math.hypot(g.A * Math.sin(th), g.B * z);
      const s = overlap(d, g.Rs, rp) / (Math.PI * rp * rp);   // fraction of the small disc over the star
      vals[i] = z >= 0 ? s : (p.luminous ? p.secondary * s : 0);
    }
    state.vals = vals;
    state.rp = rp;
    state.thetaT = Math.asin(Math.min(0.99, (g.Rs + rp) / g.A));
  }

  /** Orbit is played in slow motion around the transit (and the secondary eclipse), fast elsewhere. */
  function buildWarp() {
    const { p, thetaT } = state;
    const M = 1440, zone = 2.2 * thetaT, vmin = 0.11;
    const cum = new Float64Array(M + 1);
    for (let i = 1; i <= M; i++) {
      const th = -Math.PI + ((i - 0.5) / M) * TAU;
      let d = Math.abs(th);
      if (p.luminous) d = Math.min(d, Math.PI - Math.abs(th));
      const speed = vmin + (1 - vmin) * smooth(d / zone);
      cum[i] = cum[i - 1] + 1 / speed;
    }
    const total = cum[M], K = 2000, table = new Float32Array(K + 1);
    let j = 0;
    for (let k = 0; k <= K; k++) {
      const target = (k / K) * total;
      while (j < M && cum[j + 1] < target) j++;
      const seg = cum[j + 1] - cum[j] || 1;
      table[k] = -Math.PI + ((j + (target - cum[j]) / seg) / M) * TAU;
    }
    state.warp = table;
    // start a moment before the first transit so the viewer sees it straight away
    let k = 0; while (k < K && table[k] < -2.6 * thetaT) k++;
    state.u = k / K;
  }

  const thetaAt = (u) => state.warp[Math.min(2000, Math.max(0, Math.round(u * 2000)))];

  /* ------------------------------------------------------------------ starfield */
  function seedStars(w, h) {
    const n = Math.round(w * h / 6500);
    let s = 7;
    const rnd = () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
    state.stars = Array.from({ length: n }, () => ({
      x: rnd(), y: rnd(), r: 0.35 + rnd() ** 3 * 1.5, ph: rnd() * TAU, sp: 0.4 + rnd() * 1.3,
      tint: rnd() < 0.18 ? 1 : 0,
    }));
  }

  function drawSky(t) {
    sctx.clearRect(0, 0, state.skyW, state.skyH);
    for (const s of state.stars) {
      const tw = state.playing ? 0.55 + 0.45 * Math.sin(t * s.sp + s.ph) : 0.8;
      sctx.globalAlpha = clamp(tw, 0.1, 1) * (0.35 + s.r * 0.35);
      sctx.fillStyle = s.tint ? '#ffd9a8' : '#cfe0ff';
      sctx.beginPath(); sctx.arc(s.x * state.skyW, s.y * state.skyH, s.r, 0, TAU); sctx.fill();
    }
    sctx.globalAlpha = 1;
  }

  /* ------------------------------------------------------------------ scene */
  function drawScene(theta, dip, reveal) {
    const { g, p, rp } = state;
    const x = g.cx + g.A * Math.sin(theta), z = Math.cos(theta), y = g.cy + g.B * z;
    ctx.clearRect(0, 0, g.W, g.H);

    // orbit
    ctx.save();
    ctx.strokeStyle = 'rgba(190,210,255,0.16)'; ctx.lineWidth = 1; ctx.setLineDash([2, 7]);
    ctx.beginPath(); ctx.ellipse(g.cx, g.cy, g.A, g.B, 0, 0, TAU); ctx.stroke();
    ctx.restore();

    const star = () => {
      const dim = 1 - 0.5 * clamp(p.luminous && z < 0 ? dip / Math.max(p.secondary, 0.01) * 0.4 : dip, 0, 1);
      const reach = Math.min(g.Rs * 3.4, g.cy - 1);          // fade to nothing before the canvas edge
      const glow = ctx.createRadialGradient(g.cx, g.cy, g.Rs * 0.4, g.cx, g.cy, reach);
      glow.addColorStop(0, `rgba(255,186,105,${0.5 * dim})`);
      glow.addColorStop(0.4, `rgba(255,140,60,${0.14 * dim})`);
      glow.addColorStop(1, 'rgba(255,140,60,0)');
      ctx.fillStyle = glow; ctx.fillRect(0, 0, g.W, g.sceneH + 20);
      const disc = ctx.createRadialGradient(g.cx - g.Rs * 0.18, g.cy - g.Rs * 0.2, g.Rs * 0.05, g.cx, g.cy, g.Rs);
      disc.addColorStop(0, '#fff9e8'); disc.addColorStop(0.5, '#ffd78f');
      disc.addColorStop(0.88, '#f8a447'); disc.addColorStop(1, '#d4702a');
      ctx.fillStyle = disc; ctx.beginPath(); ctx.arc(g.cx, g.cy, g.Rs, 0, TAU); ctx.fill();
    };

    const trail = () => {
      for (let k = 14; k >= 1; k--) {
        const th = theta - k * 0.045;
        const tx = g.cx + g.A * Math.sin(th), tz = Math.cos(th), ty = g.cy + g.B * tz;
        ctx.globalAlpha = (1 - k / 15) * 0.22;
        ctx.fillStyle = p.luminous ? '#ff9a62' : '#9fc3ff';
        ctx.beginPath(); ctx.arc(tx, ty, Math.max(1, rp * (1 - k / 18) * 0.5), 0, TAU); ctx.fill();
      }
      ctx.globalAlpha = 1;
    };

    const body = () => {
      if (p.luminous) {
        const h = ctx.createRadialGradient(x, y, rp * 0.3, x, y, rp * 3.2);
        h.addColorStop(0, 'rgba(255,120,70,0.45)'); h.addColorStop(1, 'rgba(255,120,70,0)');
        ctx.fillStyle = h; ctx.beginPath(); ctx.arc(x, y, rp * 3.2, 0, TAU); ctx.fill();
        const b = ctx.createRadialGradient(x - rp * 0.3, y - rp * 0.3, rp * 0.1, x, y, rp);
        b.addColorStop(0, '#ffd2a6'); b.addColorStop(1, '#e0572e');
        ctx.fillStyle = b;
      } else {
        const b = ctx.createRadialGradient(x - rp * 0.4, y - rp * 0.4, rp * 0.1, x, y, rp);
        b.addColorStop(0, '#1a2540'); b.addColorStop(1, '#03050b');
        ctx.fillStyle = b;
      }
      ctx.beginPath(); ctx.arc(x, y, rp, 0, TAU); ctx.fill();
      if (!p.luminous) { ctx.strokeStyle = 'rgba(150,190,255,0.45)'; ctx.lineWidth = 1; ctx.stroke(); }
    };

    if (z >= 0) { star(); trail(); body(); } else { trail(); body(); star(); }

    // clock
    const day = ((theta + Math.PI) / TAU) * p.period;
    ctx.font = '14px "Atkinson Hyperlegible", system-ui, sans-serif';
    ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(205,218,240,0.78)';
    ctx.fillText(`Day ${day.toFixed(day < 10 ? 2 : 1)} of ${p.period.toFixed(p.period < 10 ? 3 : 2)}`, 14, 24);
    ctx.textAlign = 'right';
    ctx.fillText('Planet drawn larger than life', g.W - 14, 24);
  }

  function drawCurve(theta, reveal) {
    const { g, p, vals } = state;
    const L = g.curveL, w = g.curveR - g.curveL, top = g.curveTop + 4, h = g.curveH - 18;
    const yOf = (v) => top + v * h * 0.8;
    const N = vals.length - 1;
    const upto = reveal ? N : Math.round(((theta + Math.PI) / TAU) * N);

    // baseline
    ctx.strokeStyle = 'rgba(190,210,255,0.18)'; ctx.lineWidth = 1; ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(L, top); ctx.lineTo(L + w, top); ctx.stroke();

    // full curve as a guide
    ctx.strokeStyle = 'rgba(190,210,255,0.2)'; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i <= N; i++) { const X = L + (i / N) * w, Y = yOf(vals[i]); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
    ctx.stroke();

    // revealed part
    if (upto > 0) {
      ctx.beginPath();
      for (let i = 0; i <= upto; i++) { const X = L + (i / N) * w, Y = yOf(vals[i]); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
      const xEnd = L + (upto / N) * w;
      ctx.save();
      ctx.lineTo(xEnd, top); ctx.lineTo(L, top); ctx.closePath();
      const fill = ctx.createLinearGradient(0, top, 0, top + h * 0.85);
      fill.addColorStop(0, 'rgba(235,104,52,0.04)'); fill.addColorStop(1, 'rgba(235,104,52,0.38)');
      ctx.fillStyle = fill; ctx.fill(); ctx.restore();
      ctx.strokeStyle = '#f07a45'; ctx.lineWidth = 2.5; ctx.lineJoin = 'round';
      ctx.beginPath();
      for (let i = 0; i <= upto; i++) { const X = L + (i / N) * w, Y = yOf(vals[i]); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
      ctx.stroke();
    }

    // playhead
    if (!reveal) {
      const X = L + (upto / N) * w, Y = yOf(vals[upto]);
      ctx.strokeStyle = 'rgba(255,255,255,0.14)'; ctx.setLineDash([2, 4]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(X, top - 6); ctx.lineTo(X, top + h); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(X, Y, 5.5, 0, TAU); ctx.fill();
      ctx.fillStyle = '#eb6834'; ctx.beginPath(); ctx.arc(X, Y, 3.5, 0, TAU); ctx.fill();
    }

    // labels
    ctx.font = '13px "Atkinson Hyperlegible", system-ui, sans-serif';
    ctx.fillStyle = 'rgba(205,218,240,0.78)'; ctx.textAlign = 'center';
    const bottom = g.H - 8;
    ctx.fillText('transit', L + w / 2, bottom);
    ctx.textAlign = 'left'; ctx.fillText('Brightness over one orbit', L, bottom);
    ctx.textAlign = 'right'; ctx.fillText(`dips ${Math.round(p.depthPpm).toLocaleString('en-US')} ppm`, L + w, bottom);
  }

  function draw(t) {
    const { g, p } = state;
    if (!g || !p || !state.vals) return;
    const stat = !state.playing && reduceMotion.matches;       // reduced motion: one still frame
    const theta = stat ? 0 : thetaAt(state.u);
    const z = Math.cos(theta);
    const i = clamp(Math.round(((theta + Math.PI) / TAU) * (state.vals.length - 1)), 0, state.vals.length - 1);
    drawSky(t);
    drawScene(theta, state.vals[i], stat);
    drawCurve(theta, stat);
  }

  /* ------------------------------------------------------------------ loop */
  function tick(now) {
    const t = now / 1000;
    if (state.playing && state.visible && !document.hidden && state.p) {
      const dt = Math.min(0.05, t - state.last);
      state.u = (state.u + dt / LOOP_SECONDS) % 1;
      draw(t);
    }
    state.last = t;
    requestAnimationFrame(tick);
  }

  function syncToggle() {
    toggle.setAttribute('aria-pressed', String(state.playing));
    toggle.textContent = state.playing ? 'Pause animation' : 'Play animation';
  }
  toggle.addEventListener('click', () => {
    state.playing = !state.playing;
    syncToggle();
    draw(performance.now() / 1000);
  });
  reduceMotion.addEventListener('change', () => { state.playing = !reduceMotion.matches; syncToggle(); draw(performance.now() / 1000); });
  new IntersectionObserver(([e]) => { state.visible = e.isIntersecting; }, { threshold: 0.01 }).observe(stageEl);
  new ResizeObserver(() => { if (!stageEl.closest('[hidden]')) { layout(); draw(performance.now() / 1000); } }).observe(stageEl);

  window.ExoStage = {
    set(report) {
      state.p = paramsFrom(report);
      layout();
      const desc = state.p.luminous
        ? `A star is eclipsed by a luminous companion every ${state.p.period.toFixed(2)} days, with a second, shallower dip when it passes behind.`
        : `A planet crosses its star every ${state.p.period.toFixed(2)} days and dims it by ${Math.round(state.p.depthPpm).toLocaleString('en-US')} parts per million.`;
      cv.setAttribute('aria-label', `Animation. ${desc}`);
      syncToggle();
      draw(performance.now() / 1000);
    },
  };
  syncToggle();
  requestAnimationFrame(tick);
})();
