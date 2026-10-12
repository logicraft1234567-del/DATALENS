/* DataLens cinematic intro
   Timeline (seconds):
     0.00 - 1.25  WARP    data streams rush out of a single point, accelerating
     1.25 - 2.50  VORTEX  everything spirals in and forms the lens; bars rise, scanner sweeps
     2.50 - 4.00  IMPACT  flash, shockwaves, screen shake, glitch-decoded wordmark
     4.00 - 4.95  REVEAL  the camera flies through the lens into the real page
   Click, tap, Esc, Enter or Space skips to the reveal. Plays once per browser
   session; add ?intro to the URL to force it. Skipped for reduced-motion users. */
(() => {
  'use strict';

  const root = document.getElementById('intro');
  if (!root) return;

  const html = document.documentElement;
  const KEY = 'dl_intro_seen';
  const TAU = Math.PI * 2;
  const T = { warp: 1.25, impact: 2.5, reveal: 4.0, end: 4.95 };

  let inerted = [];
  let raf = 0;
  let done = false;

  const forced = /[?&]intro(=|&|$)/.test(location.search);
  let seen = false;
  try { seen = sessionStorage.getItem(KEY) === '1'; } catch (_) { /* storage blocked */ }
  const reduce = !!(window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches);

  function finish() {
    if (done) return;
    done = true;
    cancelAnimationFrame(raf);
    window.removeEventListener('resize', resize);
    document.removeEventListener('keydown', onKey);
    inerted.forEach((el) => { el.inert = false; });
    inerted = [];
    html.classList.remove('intro-active');
    root.remove();
    setTimeout(() => html.classList.remove('intro-reveal'), 1300);
    window.dispatchEvent(new CustomEvent('datalens:intro-done'));
  }

  if (reduce || (seen && !forced)) { finish(); return; }
  try { sessionStorage.setItem(KEY, '1'); } catch (_) { /* ignore */ }

  const q = (s) => root.querySelector(s);
  const stage = q('.intro-stage');
  const canvas = q('.intro-canvas');
  const ctx = canvas.getContext('2d');
  const wordEl = q('.intro-word');
  const tagEl = q('.intro-tag');
  const logo = q('.intro-logo');
  const statusEl = q('.intro-status span');
  const countEl = q('.intro-count');
  const skipBtn = q('.intro-skip');
  if (!ctx) { finish(); return; }
  if (logo) logo.addEventListener('error', () => logo.remove());

  /* Keep the page underneath from taking focus while the intro plays */
  html.classList.add('intro-active');
  for (const el of document.body.children) {
    if (el !== root && el.tagName !== 'SCRIPT' && !el.inert) { el.inert = true; inerted.push(el); }
  }

  /* ---------- helpers ---------- */
  const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
  const ease3 = (x) => x * x * x;
  const ease3out = (x) => 1 - Math.pow(1 - x, 3);
  const easeOutBack = (k) => { const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(k - 1, 3) + c1 * Math.pow(k - 1, 2); };

  /* ---------- wordmark letters ---------- */
  const WORD = 'DataLens';
  const GLYPHS = '01#$%&@<>/\\=+*?';
  wordEl.dataset.text = WORD;
  wordEl.textContent = '';
  const letters = [...WORD].map((ch) => { const s = document.createElement('span'); wordEl.appendChild(s); return s; });
  const letterState = letters.map(() => 0); // 0 hidden, 1 scrambling, 2 resolved
  let widthsLocked = false;
  function lockLetterWidths() {
    if (widthsLocked) return; widthsLocked = true;
    letters.forEach((s, i) => { s.textContent = WORD[i]; });
    const ws = letters.map((s) => s.getBoundingClientRect().width);
    letters.forEach((s, i) => { s.style.width = ws[i] + 'px'; s.textContent = ''; });
  }

  /* ---------- sizing ---------- */
  let w = 0, h = 0, dpr = 1, cx = 0, cy = 0, R = 100, maxR = 600, fs = 72;
  function resize() {
    const rect = root.getBoundingClientRect();
    w = rect.width || window.innerWidth; h = rect.height || window.innerHeight;
    dpr = Math.min(window.devicePixelRatio || 1, 1.75);
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cx = w / 2; cy = h * (h < 520 ? 0.36 : 0.42);
    R = clamp(Math.min(w, h) * 0.2, 58, 150);
    maxR = Math.hypot(w, h) / 2 * 1.15;
    fs = clamp(Math.min(w * 0.11, h * 0.14), 30, 96);
    const wordY = cy + R * 1.3 + fs * 0.75;
    wordEl.style.fontSize = fs + 'px';
    wordEl.style.top = wordY + 'px';
    tagEl.style.top = (wordY + fs * 0.5 + 16) + 'px';
    if (logo) { logo.style.width = Math.round(R * 1.05) + 'px'; }
    root.style.setProperty('--cx', cx + 'px');
    root.style.setProperty('--cy', cy + 'px');
    widthsLocked = false; // re-measure if the font size changed
    if (letterState.some((s) => s > 0)) { lockLetterWidths(); }
  }
  resize();
  window.addEventListener('resize', resize);

  /* ---------- particles ---------- */
  const COUNT = w < 700 ? 380 : 760;
  const P = [];
  const G = [[], [], []]; // teal, white, violet
  for (let i = 0; i < COUNT; i++) {
    const roll = Math.random();
    const g = roll < 0.6 ? 0 : roll < 0.82 ? 1 : 2;
    const p = {
      a: Math.random() * TAU, r: 2 + Math.random() * maxR * 0.18,
      px: 0, py: 0, fresh: true, sp: Math.random(), dir: Math.random() < 0.85 ? 1 : -1,
      off: (Math.random() - 0.5) * 14, vr: 0, burst: false, g
    };
    P.push(p); G[g].push(p);
  }

  function stepParticles(t, dt) {
    const u = clamp((t - T.warp) / (T.impact - T.warp));
    for (const p of P) {
      if (t < T.warp) {
        const k = 0.35 + 8 * ease3(t / T.warp);
        p.r += (p.r * 0.9 + 60 * (0.4 + p.sp)) * k * dt;
        if (p.r > maxR) { p.r = 2 + Math.random() * 30; p.a = Math.random() * TAU; p.fresh = true; }
      } else if (t < T.impact) {
        p.r += (R + p.off - p.r) * (1 - Math.exp(-dt * (1.5 + 5 * u)));
        p.a += p.dir * (0.6 + p.sp * 1.4 + 8 * u * u) * dt;
      } else if (t < T.reveal) {
        if (!p.burst) { p.burst = true; p.vr = 250 + Math.random() * 900; }
        p.r += p.vr * dt; p.vr *= Math.exp(-dt * 2.4);
        p.r += (R + p.off * 1.5 - p.r) * (1 - Math.exp(-dt * 0.8));
        p.a += p.dir * (1.2 + p.sp * 2) * dt;
      } else {
        const v = clamp((t - T.reveal) / (T.end - T.reveal));
        p.r *= 1 + dt * (2 + 10 * v * v);
        p.a += p.dir * (1.2 + p.sp * 2) * dt;
      }
    }
  }

  const COLORS = ['45,226,196', '235,255,252', '143,123,255'];
  const WIDTHS = [1, 0.8, 1.2];
  function drawParticles(t) {
    const e = ease3(clamp(t / T.warp));
    const alpha = t < T.warp ? 0.35 + 0.55 * e : t < T.reveal ? 0.85 : 0.6;
    const lw = t < T.warp ? 0.8 + 1.6 * e : 1.5;
    ctx.globalCompositeOperation = 'lighter';
    ctx.lineCap = 'round';
    for (let g = 0; g < 3; g++) {
      ctx.beginPath();
      for (const p of G[g]) {
        const x = cx + Math.cos(p.a) * p.r, y = cy + Math.sin(p.a) * p.r;
        if (p.fresh) { p.px = x; p.py = y; p.fresh = false; }
        ctx.moveTo(p.px, p.py); ctx.lineTo(x + 0.01, y);
        p.px = x; p.py = y;
      }
      ctx.strokeStyle = `rgba(${COLORS[g]},${alpha})`;
      ctx.lineWidth = lw * WIDTHS[g];
      ctx.stroke();
    }
  }

  /* ---------- the lens ---------- */
  const BARS = [0.35, 0.6, 0.45, 0.9, 0.7, 1, 0.55];
  function drawLens(t) {
    const start = T.warp * 0.85;
    if (t < start) return;
    const p = ease3out(clamp((t - start) / (T.impact - start) * 1.15));
    const after = Math.max(0, t - T.impact);
    const pulse = after ? 1 + 0.05 * Math.sin(after * 9) * Math.exp(-after * 2) : 1;
    const rr = R * pulse;
    const sweepFade = Math.exp(-after * 3.5);

    ctx.save();
    ctx.translate(cx, cy);
    ctx.globalCompositeOperation = 'lighter';

    // Scanner sweep
    if (ctx.createConicGradient && sweepFade > 0.02) {
      const ang = t * 3.2;
      const g = ctx.createConicGradient(ang, 0, 0);
      g.addColorStop(0, 'rgba(45,226,196,0)');
      g.addColorStop(0.72, 'rgba(45,226,196,0)');
      g.addColorStop(1, `rgba(45,226,196,${0.38 * p * sweepFade})`);
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(0, 0, rr * 0.86, 0, TAU); ctx.fill();
      ctx.strokeStyle = `rgba(160,255,240,${0.8 * p * sweepFade})`; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(Math.cos(ang) * rr * 0.86, Math.sin(ang) * rr * 0.86); ctx.stroke();
    }

    // Rising data bars (they become the logo at impact)
    const barP = ease3out(clamp((t - T.warp - 0.25) / 0.7));
    const barA = 0.55 * Math.exp(-after * 9);
    if (barP > 0 && barA > 0.01) {
      const n = BARS.length, bw = rr * 0.14, gap = rr * 0.06, total = n * bw + (n - 1) * gap;
      for (let i = 0; i < n; i++) {
        const bh = BARS[i] * rr * 0.7 * clamp(barP * 1.4 - i * 0.07);
        const bx = -total / 2 + i * (bw + gap), by = rr * 0.3;
        ctx.fillStyle = `rgba(45,226,196,${barA})`; ctx.fillRect(bx, by - bh, bw, bh);
        ctx.fillStyle = `rgba(235,255,252,${barA * 1.4})`; ctx.fillRect(bx, by - bh, bw, 2);
      }
    }

    // Main ring (draws itself clockwise), glows hard on impact
    ctx.shadowColor = 'rgba(45,226,196,.9)';
    ctx.shadowBlur = 18 + 22 * Math.exp(-after * 2.5);
    ctx.strokeStyle = 'rgba(120,255,235,.95)';
    ctx.lineWidth = 2.2 + 2 * Math.exp(-after * 3);
    ctx.beginPath(); ctx.arc(0, 0, rr, -Math.PI / 2, -Math.PI / 2 + TAU * p); ctx.stroke();
    ctx.shadowBlur = 0;

    // Rotating tick ring
    ctx.save();
    ctx.rotate(t * 0.5);
    ctx.strokeStyle = `rgba(45,226,196,${0.6 * p})`; ctx.lineWidth = 1.2;
    const n = 72;
    ctx.beginPath();
    for (let i = 0; i < n * p; i++) {
      const a = (i / n) * TAU, len = i % 6 === 0 ? 11 : 5, r0 = rr * 1.14;
      ctx.moveTo(Math.cos(a) * r0, Math.sin(a) * r0);
      ctx.lineTo(Math.cos(a) * (r0 + len), Math.sin(a) * (r0 + len));
    }
    ctx.stroke();
    ctx.restore();

    // Inner dashed ring, counter-rotating
    ctx.save();
    ctx.rotate(-t * 0.9);
    ctx.setLineDash([3, 9]);
    ctx.strokeStyle = `rgba(235,255,252,${0.5 * p})`; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.arc(0, 0, rr * 0.9, 0, TAU); ctx.stroke();
    ctx.restore();
    ctx.setLineDash([]);

    // Crosshair ticks at the four compass points
    ctx.strokeStyle = `rgba(160,255,240,${0.8 * p})`; ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i < 4; i++) {
      const a = (i * Math.PI) / 2, c = Math.cos(a), s = Math.sin(a);
      ctx.moveTo(c * rr * 0.72, s * rr * 0.72); ctx.lineTo(c * rr * 0.8, s * rr * 0.8);
    }
    ctx.stroke();

    ctx.restore();
  }

  /* ---------- impact: flash + shockwaves ---------- */
  function drawImpact(t) {
    const after = t - T.impact;
    if (after < 0) return;
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    if (after < 0.45) {
      const a = Math.pow(1 - after / 0.45, 2) * 0.85;
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, maxR * 0.7);
      g.addColorStop(0, `rgba(255,255,255,${a})`);
      g.addColorStop(0.25, `rgba(120,255,235,${a * 0.5})`);
      g.addColorStop(1, 'rgba(45,226,196,0)');
      ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
    }
    for (let i = 0; i < 3; i++) {
      const tt = after - i * 0.13;
      if (tt <= 0 || tt > 1.1) continue;
      const k = tt / 1.1;
      ctx.strokeStyle = `rgba(${i === 1 ? '235,255,252' : '45,226,196'},${Math.pow(1 - k, 1.5) * 0.8})`;
      ctx.lineWidth = 0.5 + 3 * (1 - k);
      ctx.beginPath(); ctx.arc(cx, cy, R * 0.6 + tt * maxR * 1.1, 0, TAU); ctx.stroke();
    }
    ctx.restore();
  }

  /* ---------- reveal: lens opens onto the real page ---------- */
  function drawReveal(t) {
    if (t < T.reveal) return;
    const v = clamp((t - T.reveal) / (T.end - T.reveal));
    const rr = Math.pow(v, 2.2) * maxR * 1.05;
    root.style.setProperty('--r', rr + 'px');
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.shadowColor = 'rgba(45,226,196,1)'; ctx.shadowBlur = 30;
    ctx.strokeStyle = `rgba(160,255,240,${1 - v * 0.6})`;
    ctx.lineWidth = 2 + 6 * (1 - v);
    ctx.beginPath(); ctx.arc(cx, cy, rr + 3, 0, TAU); ctx.stroke();
    ctx.restore();
  }

  /* ---------- DOM layer (wordmark, logo, HUD) ---------- */
  const STATUS = [
    [0, 'BOOTING DATA CORE'],
    [T.warp * 0.8, 'INGESTING 1,048,576 ROWS'],
    [T.warp + 0.45, 'MAPPING PATTERNS'],
    [T.impact - 0.45, 'FOCUSING LENS'],
    [T.impact, 'INSIGHT READY']
  ];
  let statusIdx = -1, lastPct = -1, tagOn = false, shaking = false, revealOn = false, frame = 0;

  function updateDom(t) {
    // HUD
    let si = 0; for (let i = 0; i < STATUS.length; i++) if (t >= STATUS[i][0]) si = i;
    if (si !== statusIdx) { statusIdx = si; statusEl.textContent = STATUS[si][1]; }
    const pct = Math.round(100 * ease3out(clamp(t / T.impact)));
    if (pct !== lastPct) { lastPct = pct; countEl.textContent = String(pct).padStart(3, '0') + '%'; }

    const after = t - T.impact;
    const v = t >= T.reveal ? clamp((t - T.reveal) / (T.end - T.reveal)) : 0;

    // Screen shake
    if (after >= 0 && after < 0.6) {
      const amp = 10 * Math.exp(-after * 7);
      stage.style.transform = `translate(${(Math.random() - 0.5) * 2 * amp}px, ${(Math.random() - 0.5) * 2 * amp}px)`;
      shaking = true;
    } else if (shaking) { stage.style.transform = ''; shaking = false; }

    // Logo pops in at impact, then flies at the camera on reveal
    if (logo) {
      if (after < 0) { logo.style.opacity = '0'; }
      else {
        const k = clamp(after / 0.55);
        let sc = easeOutBack(k) * (1 + 0.015 * Math.sin(after * 3)), op = Math.min(1, after * 4);
        if (v > 0) { sc = 1 + v * 7; op = 1 - clamp(v * 2.2); }
        logo.style.opacity = String(op);
        logo.style.transform = `translate(-50%,-50%) scale(${sc})`;
      }
    }

    // Wordmark: letters decode from glitch characters, left to right
    if (after >= 0) {
      if (!wordEl.classList.contains('on')) { lockLetterWidths(); wordEl.classList.add('on'); }
      for (let i = 0; i < letters.length; i++) {
        const s0 = 0.05 + i * 0.07, s1 = s0 + 0.35 + i * 0.02;
        const el = letters[i];
        if (after < s0) continue;
        if (after < s1) {
          if (letterState[i] !== 1) { letterState[i] = 1; el.classList.add('s'); }
          if (frame % 3 === 0) el.textContent = GLYPHS[(Math.random() * GLYPHS.length) | 0];
        } else if (letterState[i] !== 2) {
          letterState[i] = 2; el.classList.remove('s'); el.textContent = WORD[i];
        }
      }
      if (!tagOn && after > 0.55) { tagOn = true; tagEl.classList.add('on'); }
    }
    if (v > 0) {
      const o = String(1 - clamp(v * 2.5));
      wordEl.style.opacity = o; tagEl.style.opacity = o;
      wordEl.style.transform = `translate(-50%,-50%) scale(${1 + v * 1.4})`;
      tagEl.style.animation = 'none';
    }
    if (v > 0 && !revealOn) { revealOn = true; html.classList.add('intro-reveal'); }
  }

  /* ---------- main loop ---------- */
  let t = 0, last = 0;
  function tick(now) {
    if (done) return;
    if (!last) last = now;
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now; t += dt; frame++;
    if (t >= T.end) { finish(); return; }

    ctx.globalCompositeOperation = 'source-over';
    const fade = t < T.warp ? 0.3 : t < T.impact ? 0.16 : t < T.reveal ? 0.2 : 0.3;
    ctx.fillStyle = `rgba(3,8,11,${fade})`;
    ctx.fillRect(0, 0, w, h);

    stepParticles(t, dt);
    drawParticles(t);
    drawLens(t);
    drawImpact(t);
    drawReveal(t);
    updateDom(t);

    raf = requestAnimationFrame(tick);
  }

  /* ---------- skipping ---------- */
  let lastSkip = 0;
  function skip() {
    const now = performance.now();
    if (now - lastSkip < 250) return;
    lastSkip = now;
    if (t < T.reveal) t = T.reveal; else finish();
  }
  function onKey(e) {
    if (e.key === 'Escape' || e.key === 'Enter' || e.key === ' ') { e.preventDefault(); skip(); }
  }
  root.addEventListener('click', skip);
  skipBtn.addEventListener('click', (e) => { e.stopPropagation(); skip(); });
  document.addEventListener('keydown', onKey);

  // Paint the first frame solid so there is no flash of the page underneath
  ctx.fillStyle = '#03080b'; ctx.fillRect(0, 0, w, h);
  raf = requestAnimationFrame(tick);
})();