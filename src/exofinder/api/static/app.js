/* Exoplanet Finder dashboard. Plain JS + Plotly; all API text is inserted with textContent. */
(() => {
  'use strict';

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const KIND_LABEL = {
    planet: 'Planet', eclipsing_binary: 'Eclipsing binary', grazing_eb: 'Grazing binary',
    systematic: 'Instrument glitch', noise: 'Noise only',
  };
  const MODEL_LABEL = { cnn: 'Neural network', baseline: 'Gradient boosting', ensemble: 'Both averaged', auto: 'Auto' };
  const FEATURE_LABEL = {
    log_period: 'Orbital period (log10 days)', log_duration_hr: 'Dip length (log10 hours)',
    duty_cycle: 'Dip length / period', log_depth_ppm: 'Dip depth (log10 ppm)',
    log_snr: 'Signal-to-noise (log10)', odd_even_sigma: 'Odd vs even depth (sigma)',
    secondary_ratio: 'Secondary / primary depth', log_n_transits: 'Transits seen (log10)',
    log_noise_ppm: 'Point noise (log10 ppm)',
  };
  const STATUS_WORD = { pass: 'Pass', warn: 'Borderline', fail: 'Fails' };
  const ICON = {
    pass: '<svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="9" fill="#0ca30c"/><path d="M5.5 10.5l3 3 6-6.5" fill="none" stroke="#fff" stroke-width="2"/></svg>',
    warn: '<svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true"><path d="M10 1.5L19 17.5H1z" fill="#fab219"/><path d="M10 7v5" stroke="#16202c" stroke-width="2"/><circle cx="10" cy="14.6" r="1.1" fill="#16202c"/></svg>',
    fail: '<svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true"><path d="M10 .8L19.2 10 10 19.2.8 10z" fill="#d03b3b"/><path d="M6.8 6.8l6.4 6.4M13.2 6.8l-6.4 6.4" stroke="#fff" stroke-width="2"/></svg>',
  };

  const STATE = { report: null, info: null, kind: 'planet', seed: 1, view: 'raw', model: 'auto', busy: false, lastRun: null };

  /* ---------------------------------------------------------------- helpers */
  function el(tag, props = {}, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (k === 'class') e.className = v;
      else if (k === 'text') e.textContent = v;
      else if (v !== false && v != null) e.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat()) if (kid != null) e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    return e;
  }
  const num = (n, d = 0) => Number(n).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
  const pct = (x) => `${Math.round(x * 100)}%`;
  const list = (a) => (a.length < 2 ? a.join('') : `${a.slice(0, -1).join(', ')} and ${a[a.length - 1]}`);

  async function request(url, init) {
    const r = await fetch(url, init);
    let body = null;
    try { body = await r.json(); } catch { /* non-JSON error body */ }
    if (!r.ok) {
      const d = body && body.detail;
      throw new Error(typeof d === 'string' ? d : d ? JSON.stringify(d) : `Request failed (${r.status})`);
    }
    return body;
  }

  function setStatus(msg, isError = false) {
    const s = $('#status');
    s.textContent = msg;
    s.classList.toggle('error', isError);
  }
  function setBusy(b, msg) {
    STATE.busy = b;
    $('#report').classList.toggle('is-loading', b);
    $$('.chip, #redraw, #panel-star .btn').forEach((x) => { x.disabled = b; });
    if (b) setStatus(msg || 'Analyzing…');
  }
  async function analyze(fn, msg) {
    if (STATE.busy) return;
    setBusy(true, msg);
    try {
      STATE.report = await fn();
      render(STATE.report);
      setStatus('');
    } catch (e) {
      setStatus(e.message, true);
    } finally {
      setBusy(false);
    }
  }

  /* ---------------------------------------------------------------- requests */
  function runDemo() {
    STATE.lastRun = runDemo;
    return analyze(() => request(`/api/demo?kind=${STATE.kind}&seed=${STATE.seed}&model=${STATE.model}`),
      'Simulating a star and searching for repeating dips…');
  }
  function runUpload(file) {
    STATE.lastRun = () => runUpload(file);
    return analyze(() => {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('model', STATE.model);
      return request('/api/analyze/upload', { method: 'POST', body: fd });
    }, `Analyzing ${file.name}…`);
  }
  function runStar(target, mission) {
    STATE.lastRun = () => runStar(target, mission);
    return analyze(() => request('/api/analyze/target', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target, mission, model: STATE.model }),
    }), 'Fetching from NASA MAST and searching for repeating dips. This can take a minute…');
  }

  /* ---------------------------------------------------------------- theme */
  function currentTheme() {
    return document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  }
  try { const t = localStorage.getItem('exofinder-theme'); if (t) document.documentElement.dataset.theme = t; } catch { /* storage blocked */ }
  $('#theme').addEventListener('click', () => {
    const next = currentTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('exofinder-theme', next); } catch { /* storage blocked */ }
    if (STATE.report) drawCharts(STATE.report);
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (STATE.report) drawCharts(STATE.report); });

  /* ---------------------------------------------------------------- verdict */
  function renderVerdict(r) {
    const v = r.verdict;
    const flagged = r.checks.filter((c) => c.status !== 'pass');
    const issues = flagged.map((c) => c.issue);
    $('#verdict-title').textContent = v.planet_like ? 'Looks like a planet.' : 'Probably not a planet.';
    $('#verdict-sub').textContent = v.planet_like
      ? (flagged.length ? `Worth a closer look: ${list(issues)}.` : 'The dips repeat on a steady clock, share one shape, and pass every vetting check.')
      : (flagged.length ? `What stands out: ${list(issues)}.` : 'The dips do not resemble the confirmed planets the model learned from.');

    const ruler = $('#ruler');
    ruler.style.setProperty('--score', v.score);
    ruler.style.setProperty('--cut', v.threshold);
    ruler.setAttribute('aria-label', `Planet-likeness score ${v.score.toFixed(2)} on a scale from 0 to 1. The cut-off for calling a planet is ${v.threshold.toFixed(2)}.`);
    const val = $('#ruler-val');
    val.textContent = `Score ${v.score.toFixed(2)}`;
    val.className = 'ruler-val' + (v.score < 0.12 ? ' left' : v.score > 0.88 ? ' right' : '');
    const cl = $('#ruler-cut');
    cl.textContent = `Cut-off ${v.threshold.toFixed(2)}`;
    cl.className = 'ruler-cutlabel' + (v.threshold < 0.1 ? ' left' : v.threshold > 0.9 ? ' right' : '');

    const pm = $('#per-model');
    pm.replaceChildren();
    for (const [k, m] of Object.entries(v.per_model)) {
      const used = v.model === k || v.model === 'ensemble';
      pm.append(el('li', {}, el('strong', { text: MODEL_LABEL[k] }), ` ${m.score.toFixed(2)} (cut-off ${m.threshold.toFixed(2)})`, used ? ' · used' : ''));
    }
    if (v.model === 'ensemble') pm.append(el('li', { text: 'Verdict uses the average of both.' }));

    const truth = $('#truth');
    if (r.truth) {
      const ok = r.truth.is_planet === v.planet_like;
      const per = r.truth.period > 0 ? ` with a true period of ${num(r.truth.period, 2)} days` : '';
      truth.replaceChildren(el('strong', { text: 'Answer key: ' }),
        `this star was generated as ${(KIND_LABEL[r.truth.kind] || r.truth.kind).toLowerCase()}${per}. The model's call is ${ok ? 'correct' : 'wrong'}.`);
      truth.hidden = false;
    } else {
      truth.hidden = true;
    }

    const c = r.candidate, t = r.target;
    const fact = (label, value, unit, wide = false) =>
      el('div', { class: wide ? 'wide' : '' }, el('dt', { text: label }), el('dd', {}, value, unit ? el('small', { text: unit }) : null));
    $('#facts').replaceChildren(
      fact('Orbital period', num(c.period_days, c.period_days < 10 ? 3 : 2), 'days'),
      fact('Dip depth', num(c.depth_ppm), 'ppm'),
      fact('Dip length', num(c.duration_hours, 1), 'hours'),
      fact('Signal-to-noise', num(c.snr, 1)),
      fact('Transits seen', String(c.transits_observed)),
      fact('Search strength', num(c.sde, 1), 'sigma'),
      fact('Light curve', `${t.name}: ${num(t.points)} points over ${num(t.baseline_days, 1)} days, ${num(t.noise_ppm)} ppm noise`, '', true),
    );
  }

  function renderChecks(r) {
    const ul = $('#checks');
    ul.replaceChildren();
    for (const c of r.checks) {
      const icon = el('span');
      icon.innerHTML = ICON[c.status]; // static, trusted markup
      ul.append(el('li', {}, icon,
        el('div', {},
          el('div', { class: 'head' }, c.label, ' ', el('span', { text: `· ${STATUS_WORD[c.status]}` })),
          el('p', { text: c.detail }))));
    }
  }

  function table(cols, rows, maxRows = 150) {
    const step = Math.max(1, Math.ceil(rows.length / maxRows));
    const shown = rows.filter((_, i) => i % step === 0);
    const t = el('table', {}, el('thead', {}, el('tr', {}, cols.map((c) => el('th', { text: c })))),
      el('tbody', {}, shown.map((row) => el('tr', {}, row.map((x) => el('td', { text: x }))))));
    const frag = document.createDocumentFragment();
    frag.append(t);
    if (step > 1) frag.append(el('p', { class: 'table-note', text: `Showing every ${step}th of ${num(rows.length)} rows.` }));
    return frag;
  }

  function renderCard(r) {
    const body = $('#card-body');
    body.replaceChildren();
    const metrics = STATE.info && STATE.info.metrics;
    const key = r.verdict.model;
    const m = metrics && metrics[key];
    if (!m || !m.test) {
      body.append(el('p', { class: 'card-note', text: 'No evaluation numbers found for this model. Run exofinder train to produce them.' }));
    } else {
      const t = m.test;
      body.append(
        el('dl', { class: 'metrics' },
          el('div', {}, el('dt', { text: 'PR-AUC' }), el('dd', { text: t.pr_auc == null ? 'n/a' : t.pr_auc.toFixed(2) })),
          el('div', {}, el('dt', { text: 'Recall' }), el('dd', { text: pct(t.recall ?? 0) })),
          el('div', {}, el('dt', { text: 'Precision' }), el('dd', { text: pct(t.precision ?? 0) }))),
        el('p', { class: 'cap', text: 'Recall is the share of real planets caught. Precision is the share of planet calls that were real. Measured on stars the model never saw during training.' }));
      if (m.by_kind) {
        const order = Object.keys(m.by_kind).sort((a, b) => (a === 'planet' ? -1 : b === 'planet' ? 1 : a.localeCompare(b)));
        const wrap = el('div', { class: 'table-wrap kinds' });
        wrap.append(table(['Signal type', 'Tested', 'Called planet-like'],
          order.map((k) => [KIND_LABEL[k] || k, String(m.by_kind[k].n), pct(m.by_kind[k].flagged_as_planet)])));
        body.append(wrap, el('p', { class: 'card-note', text: 'Planets should score high here and everything else low.' }));
      }
    }
    const ds = metrics && metrics.dataset;
    if (ds) {
      const synthetic = ds.kinds && ds.kinds.planet !== undefined;
      body.append(el('p', { class: 'card-note', text: synthetic
        ? `Trained and tested on ${num(ds.n)} simulated light curves, so accuracy on real stars will be lower.`
        : `Trained and tested on ${num(ds.n)} labelled Kepler or TESS candidates.` }));
    }
  }

  function renderFeatures(r) {
    $('#features').replaceChildren(table(['Feature', 'Value'],
      Object.entries(r.features).map(([k, v]) => [FEATURE_LABEL[k] || k, num(v, 3)])));
  }

  /* ---------------------------------------------------------------- charts */
  const tokens = () => {
    const cs = getComputedStyle(document.documentElement);
    const g = (n) => cs.getPropertyValue(n).trim();
    return { ink: g('--ink'), ink2: g('--ink-2'), muted: g('--muted'), rule: g('--rule'), grid: g('--grid'),
      s1: g('--series-1'), s2: g('--series-2'), surface: g('--surface'), font: g('--font-ui') };
  };
  const axis = (t, title, extra = {}) => ({
    title: { text: title, standoff: 8, font: { size: 12, color: t.ink2 } },
    gridcolor: t.grid, zeroline: false, linecolor: t.rule, tickcolor: t.rule, ticks: 'outside', ticklen: 4,
    automargin: true, tickfont: { size: 12, color: t.ink2 }, ...extra,
  });
  const base = (t, o) => ({
    height: o.height, margin: { l: 64, r: 14, t: o.legend ? 30 : 10, b: o.b ?? 48 },
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: t.font, size: 12, color: t.ink2 },
    hoverlabel: { bgcolor: t.surface, bordercolor: t.rule, font: { family: t.font, size: 13, color: t.ink } },
    showlegend: !!o.legend,
    legend: { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', font: { size: 12, color: t.ink2 } },
    hovermode: o.hover || 'x unified', dragmode: 'zoom', hoverdistance: 30,
    ...(o.extra || {}),
  });
  const CONFIG = { displayModeBar: false, responsive: true, doubleClick: 'reset' };
  const spikes = (t) => ({ showspikes: true, spikemode: 'across', spikethickness: 1, spikecolor: t.muted, spikedash: 'solid', spikesnap: 'cursor' });

  function drawLC(r) {
    const t = tokens(), L = r.lightcurve, flat = STATE.view === 'flat';
    const y = flat ? L.flat : L.raw;
    const top = Math.max(...y), bottom = Math.min(...y), pad = (top - bottom) * 0.07 || 1;
    const traces = [{ type: 'scatter', mode: 'markers', x: L.time, y, name: flat ? 'Brightness, drift removed' : 'Brightness',
      marker: { size: 3.5, color: t.s1, opacity: 0.6 }, hovertemplate: '%{y:,.0f} ppm' }];
    if (!flat) traces.push({ type: 'scatter', mode: 'lines', x: L.trend ? L.time : [], y: L.trend, name: 'Slow drift',
      line: { color: t.ink2, width: 2 }, hovertemplate: '%{y:,.0f} ppm' });
    traces.push({ type: 'scatter', mode: 'markers', x: L.transit_times, y: L.transit_times.map(() => top + pad),
      name: 'Predicted transit', marker: { symbol: 'triangle-down', size: 10, color: t.s2 }, hovertemplate: 'Transit expected' });
    const w = Math.max(L.duration_days, (L.time[L.time.length - 1] - L.time[0]) / 500);
    const shapes = L.transit_times.length > 400 ? [] : L.transit_times.map((tt) => ({
      type: 'rect', xref: 'x', yref: 'paper', x0: tt - w / 2, x1: tt + w / 2, y0: 0, y1: 1,
      fillcolor: t.s2, opacity: 0.16, line: { width: 0 }, layer: 'below' }));
    Plotly.react('plot-lc', traces, base(t, { height: 320, legend: true, extra: {
      shapes, xaxis: axis(t, 'Time (days)', { hoverformat: '.2f', ...spikes(t) }),
      yaxis: axis(t, 'Brightness change (ppm)', { tickformat: ',.0f', range: [bottom - pad, top + 2.2 * pad] }) } }), CONFIG);
  }

  function drawPG(r) {
    const t = tokens(), P = r.periodogram, box = $('#plot-pg');
    if (!P.period.length) {
      Plotly.purge(box);
      box.replaceChildren(el('p', { class: 'cap', text: 'No search was run because the period was supplied.' }));
      return;
    }
    if (box.querySelector('p')) box.replaceChildren();
    const ymax = Math.max(...P.sde) * 1.1, ymin = Math.min(...P.sde, 0);
    const vline = (x, color, width, dash, text) => ({ type: 'scatter', mode: text ? 'lines+text' : 'lines', x: [x, x], y: [ymin, ymax],
      line: { color, width, dash }, text: text ? ['', text] : undefined, textposition: 'top right', textfont: { color: t.ink, size: 12 },
      hoverinfo: 'skip', showlegend: false, cliponaxis: false });
    const lo = P.period[0], hi = P.period[P.period.length - 1];
    const traces = [{ type: 'scatter', mode: 'lines', x: P.period, y: P.sde, name: 'Signal strength',
      line: { color: t.s1, width: 1.5 }, hovertemplate: '%{x:.3f} days<br>strength %{y:.1f}<extra></extra>' }];
    for (const [f, lab] of [[0.5, 'half'], [2, 'double']]) {
      const x = P.best * f;
      if (x > lo && x < hi) traces.push(vline(x, t.muted, 1, 'dot', lab));
    }
    traces.push(vline(P.best, t.s2, 2.5, 'solid', `${P.best.toFixed(3)} days`));
    Plotly.react('plot-pg', traces, base(t, { height: 230, hover: 'closest', extra: {
      xaxis: axis(t, 'Orbital period (days)', { type: 'log', ...spikes(t) }),
      yaxis: axis(t, 'Signal strength', { range: [ymin, ymax] }) } }), CONFIG);
  }

  function drawFolds(r) {
    const t = tokens(), F = r.fold;
    const dots = (x, y) => ({ type: 'scatter', mode: 'markers', x, y, name: 'Each measurement',
      marker: { size: 3.5, color: t.s1, opacity: 0.4 }, hovertemplate: '%{y:,.0f} ppm<extra></extra>' });
    const line = (x, y, name) => ({ type: 'scatter', mode: 'lines', x, y, name,
      line: { color: t.s2, width: 3 }, hovertemplate: '%{y:,.0f} ppm<extra></extra>' });
    $('#fold-title').textContent = `Every orbit stacked together at ${r.candidate.period_days.toFixed(3)} days`;
    Plotly.react('plot-fg', [dots(F.global.phase_days, F.global.flux), line(F.global.bin_phase_days, F.global.bin_flux, 'Typical value per slice')],
      base(t, { height: 280, legend: true, hover: 'closest', extra: {
        xaxis: axis(t, 'Days from mid-transit', { ...spikes(t) }), yaxis: axis(t, 'Brightness change (ppm)', { tickformat: ',.0f' }) } }), CONFIG);

    const bins = F.local.bin_phase_hours;
    const step = (bins[bins.length - 1] - bins[0]) / (bins.length - 1);
    const half = bins[bins.length - 1] + step / 2;
    const hasAttn = !!r.attention;
    Plotly.react('plot-fl', [dots(F.local.phase_hours, F.local.flux), line(bins, F.local.bin_flux, 'Typical value per slice')],
      base(t, { height: hasAttn ? 232 : 280, legend: true, hover: 'closest', b: hasAttn ? 30 : 48, extra: {
        xaxis: axis(t, hasAttn ? '' : 'Hours from mid-transit', { range: [-half, half], ...spikes(t), showticklabels: !hasAttn || true }),
        yaxis: axis(t, 'Brightness change (ppm)', { tickformat: ',.0f' }) } }), CONFIG);

    $('#attn-wrap').hidden = !hasAttn;
    $('#cap-attn').hidden = !hasAttn;
    if (hasAttn) {
      const a = r.attention.local.map((v) => Math.max(v, 0));
      const n = a.length;
      const xs = a.map((_, i) => -half + ((i + 0.5) * 2 * half) / n);
      $('#cap-attn').textContent = 'The shaded area under the zoom shows where the neural network looked: the taller it is, the more erasing that stretch of the curve lowers the score.';
      Plotly.react('plot-attn', [{ type: 'scatter', mode: 'lines', x: xs, y: a, fill: 'tozeroy',
        line: { color: t.s1, width: 2 }, fillcolor: 'rgba(42,120,214,0.25)', name: 'Network focus',
        hovertemplate: '%{x:.1f} h from mid-transit<br>score drop %{y:.2f}<extra></extra>' }],
      base(t, { height: 120, hover: 'x', b: 40, extra: {
        xaxis: axis(t, 'Hours from mid-transit', { range: [-half, half] }),
        yaxis: axis(t, 'Score drop', { rangemode: 'tozero', nticks: 3, tickformat: '.1f', fixedrange: true }) } }), CONFIG);
    } else {
      Plotly.purge('plot-attn');
    }
  }

  function drawCharts(r) {
    $('#cap-lc').textContent = 'Each orange marker is a predicted transit; the grey line is the slow drift removed before searching. Drag to zoom, double-click to reset.';
    drawLC(r); drawPG(r); drawFolds(r);
  }

  /* ---------------------------------------------------------------- tables (accessible alternative to charts) */
  const TABLES = {
    lc: (r) => { const L = r.lightcurve; return [['Day', 'Brightness (ppm)', 'Drift (ppm)', 'Drift removed (ppm)'],
      L.time.map((x, i) => [num(x, 3), num(L.raw[i]), num(L.trend[i]), num(L.flat[i])])]; },
    pg: (r) => [['Period (days)', 'Signal strength'], r.periodogram.period.map((p, i) => [num(p, 4), num(r.periodogram.sde[i], 2)])],
    fold: (r) => { const f = r.fold.local, a = r.attention && r.attention.local; const n = f.bin_phase_hours.length;
      return [['Hours from mid-transit', 'Typical brightness (ppm)'],
        f.bin_phase_hours.map((x, i) => [num(x, 2), num(f.bin_flux[i])])]; },
  };
  $$('details.data[data-table]').forEach((d) => d.addEventListener('toggle', () => {
    if (!d.open || !STATE.report) return;
    const [cols, rows] = TABLES[d.dataset.table](STATE.report);
    d.querySelector('.table-wrap').replaceChildren(table(cols, rows));
  }));

  /* ---------------------------------------------------------------- render */
  function render(r) {
    $('#report').hidden = false;
    renderVerdict(r);
    renderChecks(r);
    renderCard(r);
    renderFeatures(r);
    drawCharts(r);
    $$('details.data[data-table][open]').forEach((d) => {
      const [cols, rows] = TABLES[d.dataset.table](r);
      d.querySelector('.table-wrap').replaceChildren(table(cols, rows));
    });
  }

  /* ---------------------------------------------------------------- controls */
  function selectTab(id) {
    for (const tab of $$('.tab')) {
      const on = tab.id === `tab-${id}`;
      tab.setAttribute('aria-selected', on);
      tab.tabIndex = on ? 0 : -1;
      $(`#${tab.getAttribute('aria-controls')}`).hidden = !on;
    }
  }
  $$('.tab').forEach((tab, i, all) => {
    tab.addEventListener('click', () => selectTab(tab.id.slice(4)));
    tab.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      const next = all[(i + (e.key === 'ArrowRight' ? 1 : all.length - 1)) % all.length];
      selectTab(next.id.slice(4));
      next.focus();
    });
  });

  $$('.chip').forEach((chip) => chip.addEventListener('click', () => {
    STATE.kind = chip.dataset.kind;
    STATE.seed = 1 + Math.floor(Math.random() * 9998);
    $$('.chip').forEach((c) => c.setAttribute('aria-pressed', c === chip));
    runDemo();
  }));
  $('#redraw').addEventListener('click', () => { STATE.seed = 1 + Math.floor(Math.random() * 9998); runDemo(); });

  $('#file').addEventListener('change', (e) => { const f = e.target.files[0]; if (f) runUpload(f); e.target.value = ''; });
  const drop = $('#drop');
  ['dragenter', 'dragover'].forEach((n) => drop.addEventListener(n, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((n) => drop.addEventListener(n, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', (e) => { const f = e.dataTransfer.files[0]; if (f) runUpload(f); });

  $('#panel-star').addEventListener('submit', (e) => {
    e.preventDefault();
    const id = $('#star-id').value.trim();
    if (!id) { setStatus('Enter a numeric ID such as KIC 8462852 or TIC 261136679.', true); return; }
    runStar(id, $('#mission').value);
  });

  $$('.seg button').forEach((b) => b.addEventListener('click', () => {
    STATE.view = b.dataset.view;
    $$('.seg button').forEach((x) => x.setAttribute('aria-pressed', x === b));
    if (STATE.report) drawLC(STATE.report);
  }));

  function setupModelSelect(info) {
    const sel = $('#model-select');
    if (info.models.length < 2) return;
    sel.replaceChildren(el('option', { value: 'auto', text: 'Auto' }), ...info.models.map((m) => el('option', { value: m, text: MODEL_LABEL[m] })));
    sel.hidden = false;
    $('#model-label').hidden = false;
    sel.addEventListener('change', () => { STATE.model = sel.value; if (STATE.lastRun) STATE.lastRun(); });
  }

  (async function init() {
    try {
      STATE.info = await request('/api/model');
    } catch (e) {
      setStatus(`Could not reach the API: ${e.message}`, true);
      return;
    }
    if (!STATE.info.models.length) {
      setStatus('No trained model found. Run "exofinder train" to create one, then reload this page.', true);
      return;
    }
    setupModelSelect(STATE.info);
    runDemo();
  })();
})();
