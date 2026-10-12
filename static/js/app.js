/* DataLens front end. Authentication and authorisation are always enforced by the server;
   this file only renders what the API returns. User data is inserted with textContent, never as HTML. */
(() => {
  'use strict';

  /* ===================== Helpers ===================== */
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const COLORS = ['#7c5cff', '#38bdf8', '#34d399', '#fbbf24', '#f472b6', '#f87171', '#a78bfa', '#22d3ee'];

  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === undefined || v === null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'text') el.textContent = v;
      else if (k === 'style') el.style.cssText = v; // CSSOM is allowed by the CSP; inline style attributes are not
      else if (k === 'dataset') Object.assign(el.dataset, v);
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat()) {
      if (kid === null || kid === undefined || kid === false) continue;
      el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }
  const ico = (name) => h('i', { 'data-lucide': name });
  const refreshIcons = () => { if (window.lucide) window.lucide.createIcons(); };
  const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); return el; };
  const mount = (el, ...kids) => { clear(el).append(...kids.flat().filter(Boolean)); refreshIcons(); return el; };

  const nfmt = new Intl.NumberFormat();
  const fmt = (n, d = 2) => (n === null || n === undefined || Number.isNaN(n)) ? '—'
    : (Number.isInteger(n) ? nfmt.format(n) : nfmt.format(Math.round(n * 10 ** d) / 10 ** d));
  const fmtBytes = (b) => b < 1024 ? `${b} B` : b < 1048576 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(1)} MB`;
  const fmtDate = (iso) => iso ? new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' }) : '—';
  const fmtDay = (iso) => new Date(iso + 'T00:00:00').toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  function timeAgo(iso) {
    if (!iso) return '—';
    const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
    if (s < 60) return 'just now';
    if (s < 3600) return `${Math.floor(s / 60)} min ago`;
    if (s < 86400) return `${Math.floor(s / 3600)} hr ago`;
    if (s < 604800) return `${Math.floor(s / 86400)} d ago`;
    return fmtDate(iso);
  }
  const debounce = (fn, ms = 300) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

  /* ===================== State ===================== */
  const state = {
    user: null, csrf: document.querySelector('meta[name="csrf-token"]')?.content || '',
    ai: { configured: false }, limits: {}, datasets: [], dashDs: null,
    an: { id: null, data: null, tab: 'overview' }, chat: { dsId: null, conv: null, convs: [], sending: false, chartCtx: null, status: null },
  };
  const charts = {};
  function drawChart(key, canvas, config) {
    if (charts[key]) charts[key].destroy();
    if (!window.Chart || !canvas) return null;
    charts[key] = new window.Chart(canvas, config);
    return charts[key];
  }
  function destroyCharts(prefix) {
    for (const k of Object.keys(charts)) if (k.startsWith(prefix)) { charts[k].destroy(); delete charts[k]; }
  }
  function setupChartDefaults() {
    if (!window.Chart) return;
    const C = window.Chart;
    C.defaults.color = '#8d95bb';
    C.defaults.font.family = '"Inter", system-ui, sans-serif';
    C.defaults.borderColor = 'rgba(129,140,248,0.12)';
    C.defaults.maintainAspectRatio = false;
    C.defaults.plugins.legend.labels.boxWidth = 10;
    C.defaults.plugins.legend.labels.usePointStyle = true;
    C.defaults.plugins.tooltip.backgroundColor = '#1a2349';
    C.defaults.plugins.tooltip.borderColor = 'rgba(129,140,248,0.4)';
    C.defaults.plugins.tooltip.borderWidth = 1;
    C.defaults.plugins.tooltip.padding = 10;
  }

  /* ===================== API ===================== */
  class ApiError extends Error {
    constructor(message, status, code, fields, detail) { super(message); this.status = status; this.code = code; this.fields = fields; this.detail = detail || {}; }
  }
  async function api(method, url, body) {
    const headers = { 'X-CSRFToken': state.csrf, Accept: 'application/json' };
    let payload;
    if (body instanceof FormData) payload = body;
    else if (body !== undefined) { headers['Content-Type'] = 'application/json'; payload = JSON.stringify(body); }
    let res;
    try { res = await fetch(url, { method, headers, body: payload, credentials: 'same-origin' }); }
    catch (_) { throw new ApiError('Could not reach the server. Check your connection and try again.', 0, 'network'); }
    let data = null;
    try { data = await res.json(); } catch (_) { /* non-JSON */ }
    if (data && data.csrf_token) state.csrf = data.csrf_token;
    if (!res.ok || !data || data.ok === false) {
      const e = data && data.error || {};
      const err = new ApiError(e.message || 'Something went wrong. Please try again.', res.status, e.code, e.fields, e);
      if (res.status === 401 && e.code === 'auth_required' && state.user) onSessionExpired();
      if (res.status === 403 && e.code === 'password_change_required') openForcedPassword();
      throw err;
    }
    return data;
  }
  const get = (u) => api('GET', u);
  const post = (u, b) => api('POST', u, b === undefined ? {} : b);
  const patch = (u, b) => api('PATCH', u, b);
  const del = (u, b) => api('DELETE', u, b === undefined ? {} : b);
  const qs = (o) => Object.entries(o).filter(([, v]) => v !== undefined && v !== null && v !== '').map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`).join('&');

  async function downloadFile(url, button) {
    if (button) button.setAttribute('aria-busy', 'true');
    try {
      const res = await fetch(url, { credentials: 'same-origin' });
      if (!res.ok) {
        let msg = 'The download failed.';
        try { msg = (await res.json()).error.message; } catch (_) { /* keep default */ }
        throw new Error(msg);
      }
      const blob = await res.blob();
      const m = /filename="?([^";]+)"?/.exec(res.headers.get('Content-Disposition') || '');
      const a = h('a', { href: URL.createObjectURL(blob), download: m ? m[1] : 'download' });
      document.body.append(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 4000);
      toast('Download ready.', 'success');
    } catch (e) { toast(e.message, 'error'); }
    finally { if (button) button.removeAttribute('aria-busy'); }
  }

  /* ===================== Toasts and dialogs ===================== */
  function toast(message, type = 'info', ms = 4800) {
    const box = h('div', { class: `toast ${type}`, role: type === 'error' ? 'alert' : 'status' },
      ico(type === 'error' ? 'circle-alert' : type === 'success' ? 'circle-check' : 'info'), h('span', { text: message }));
    $('#toasts').append(box); refreshIcons();
    setTimeout(() => box.remove(), ms);
  }
  function confirmDialog({ title, text, okLabel = 'Confirm', danger = true }) {
    const dlg = $('#dlg-confirm');
    $('#dlg-confirm-title').textContent = title;
    $('#dlg-confirm-text').textContent = text || '';
    const ok = $('#dlg-confirm-ok'); ok.textContent = okLabel; ok.className = danger ? 'btn btn-danger' : 'btn btn-primary';
    return new Promise((resolve) => {
      dlg.returnValue = '';
      dlg.addEventListener('close', () => resolve(dlg.returnValue === 'ok'), { once: true });
      dlg.showModal();
    });
  }
  function promptDialog({ title, text, label, value = '', type = 'text', okLabel = 'Save', validate }) {
    const dlg = $('#dlg-prompt'), input = $('#dlg-prompt-input'), err = $('#dlg-prompt-error');
    $('#dlg-prompt-title').textContent = title; $('#dlg-prompt-text').textContent = text || '';
    $('#dlg-prompt-label').textContent = label; $('#dlg-prompt-ok').textContent = okLabel;
    input.type = type; input.value = value; err.textContent = '';
    return new Promise((resolve) => {
      const ok = $('#dlg-prompt-ok');
      const onOk = (e) => {
        const msg = validate ? validate(input.value) : null;
        if (msg) { e.preventDefault(); err.textContent = msg; input.focus(); }
      };
      ok.addEventListener('click', onOk);
      dlg.returnValue = '';
      dlg.addEventListener('close', () => { ok.removeEventListener('click', onOk); resolve(dlg.returnValue === 'ok' ? input.value : null); }, { once: true });
      dlg.showModal(); input.focus(); input.select();
    });
  }

  function emptyState(iconName, title, text, action) {
    return h('div', { class: 'empty-state' }, h('div', { class: 'es-icon' }, ico(iconName)), h('h4', { text: title }),
      text ? h('p', { text }) : null, action || null);
  }
  const skeleton = (n = 3) => h('div', {}, Array.from({ length: n }, () => h('div', { class: 'skeleton', style: 'margin:10px 0;height:22px' })));
  function setBusy(btn, busy) { if (!btn) return; if (busy) btn.setAttribute('aria-busy', 'true'); else btn.removeAttribute('aria-busy'); btn.disabled = !!busy; }
  function showFormError(form, err) {
    $$('.field', form).forEach((f) => f.classList.remove('has-error'));
    $$('.field-error', form).forEach((e) => { e.textContent = ''; });
    const alertBox = $('[data-form-alert]', form);
    const fields = err && err.fields;
    if (fields) {
      for (const [name, msg] of Object.entries(fields)) {
        const slot = $(`[data-error-for="${name}"]`, form);
        if (slot) { slot.textContent = msg; slot.closest('.field')?.classList.add('has-error'); }
      }
    }
    if (alertBox) {
      alertBox.classList.remove('success');
      alertBox.hidden = !err || (fields && Object.keys(fields).length > 0);
      if (err && alertBox.hidden === false) alertBox.textContent = err.message;
    }
    const first = $('.has-error input, .has-error select', form);
    if (first) first.focus();
  }

  /* ===================== Authentication ===================== */
  function showScreen(name) {
    $('#boot').hidden = true;
    $('#auth-screen').hidden = name !== 'auth';
    $('#app-shell').hidden = name !== 'app';
    document.body.classList.toggle('is-auth', name === 'auth');
  }
  function setAuthTab(tab) {
    $$('[data-auth-tab].seg').forEach((b) => { const on = b.dataset.authTab === tab; b.classList.toggle('active', on); b.setAttribute('aria-selected', on); });
    $$('[data-auth-form]').forEach((f) => { f.hidden = f.dataset.authForm !== tab; });
    const first = $(`[data-auth-form="${tab}"] input`); if (first) first.focus();
  }
  function wireAuth() {
    document.addEventListener('click', (e) => {
      const t = e.target.closest('[data-auth-tab]'); if (t) setAuthTab(t.dataset.authTab);
      const pw = e.target.closest('[data-toggle-password]');
      if (pw) {
        const input = document.getElementById(pw.dataset.togglePassword);
        const show = input.type === 'password'; input.type = show ? 'text' : 'password';
        pw.setAttribute('aria-label', show ? 'Hide password' : 'Show password');
        mount(pw, ico(show ? 'eye-off' : 'eye'));
      }
    });
    $('#login-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const f = e.currentTarget, btn = $('button[type=submit]', f);
      const body = Object.fromEntries(new FormData(f));
      if (!body.identifier.trim() || !body.password) return showFormError(f, { fields: { identifier: body.identifier.trim() ? '' : 'Enter your email or username.', password: body.password ? '' : 'Enter your password.' } });
      setBusy(btn, true); showFormError(f, null);
      try { const d = await post('/api/auth/login', body); f.reset(); await afterAuth(d.user); }
      catch (err) { showFormError(f, err); }
      finally { setBusy(btn, false); }
    });
    $('#signup-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const f = e.currentTarget, btn = $('button[type=submit]', f);
      setBusy(btn, true); showFormError(f, null);
      try { const d = await post('/api/auth/signup', Object.fromEntries(new FormData(f))); f.reset(); toast('Welcome to DataLens.', 'success'); await afterAuth(d.user); }
      catch (err) { showFormError(f, err); }
      finally { setBusy(btn, false); }
    });
  }
  async function afterAuth(user) {
    state.user = user;
    try { const me = await get('/api/me'); state.ai = me.ai; state.limits = me.limits || {}; } catch (_) { /* non-fatal */ }
    enterApp();
  }
  function enterApp() {
    showScreen('app');
    const u = state.user;
    $('#user-avatar').textContent = (u.full_name || u.username || '?').trim()[0] || '?';
    $('#user-btn-name').textContent = u.full_name.split(' ')[0] || u.username;
    $('#um-name').textContent = u.full_name; $('#um-email').textContent = u.email;
    $('#admin-nav').hidden = !u.is_admin;
    refreshIcons();
    if (u.must_change_password) return openForcedPassword();
    if (!location.hash || location.hash === '#') location.hash = '#/dashboard';
    route();
  }
  async function logout() {
    try { await post('/api/auth/logout'); } catch (_) { /* ignore */ }
    resetAfterLogout();
  }
  function resetAfterLogout() {
    state.user = null; state.datasets = []; state.dashDs = null; state.an = { id: null, data: null, tab: 'overview' };
    state.chat = { dsId: null, conv: null, convs: [], sending: false, chartCtx: null, status: null };
    Object.keys(charts).forEach((k) => { charts[k].destroy(); delete charts[k]; });
    location.hash = '';
    for (const d of $$('dialog[open]')) d.close();
    showScreen('auth'); setAuthTab('login');
  }
  function onSessionExpired() { if (!state.user) return; toast('Your session expired. Please log in again.', 'error'); resetAfterLogout(); }

  function wireForcedPassword() {
    const dlg = $('#dlg-force-pw'), form = $('#force-pw-form');
    dlg.addEventListener('cancel', (e) => e.preventDefault());
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const btn = $('button[type=submit]', form); setBusy(btn, true); showFormError(form, null);
      try {
        const d = await post('/api/auth/password', Object.fromEntries(new FormData(form)));
        state.user = d.user; form.reset(); dlg.close(); toast('Password updated.', 'success'); enterApp();
      } catch (err) { showFormError(form, err); } finally { setBusy(btn, false); }
    });
  }
  function openForcedPassword() { const d = $('#dlg-force-pw'); if (!d.open) { d.showModal(); refreshIcons(); } }

  /* ===================== Router and shell ===================== */
  const ROUTES = {
    dashboard: { title: 'Dashboard', load: loadDashboard }, upload: { title: 'Upload CSV', load: loadUpload },
    datasets: { title: 'My datasets', load: loadDatasets }, analysis: { title: 'Analyses', load: loadAnalysis },
    chat: { title: 'AI chat', load: loadChat }, reports: { title: 'Reports', load: loadReports },
    assistants: { title: 'My AIs', load: loadAssistants },
    api: { title: 'API & Developers', load: loadApiPage },
    settings: { title: 'Settings', load: loadSettings },
    'admin-analytics': { title: 'Analytics', load: loadAdminAnalytics, admin: true },
    'admin-users': { title: 'User management', load: loadAdminUsers, admin: true },
    'admin-logs': { title: 'System logs', load: loadAdminLogs, admin: true },
  };
  let currentRoute = null;
  function route() {
    if (!state.user || state.user.must_change_password) return;
    const [name, arg] = (location.hash.replace(/^#\/?/, '') || 'dashboard').split('/');
    const def = ROUTES[name];
    if (!def || (def.admin && !state.user.is_admin)) { location.hash = '#/dashboard'; return; }
    if (currentRoute && currentRoute !== name) leaveView(currentRoute);
    currentRoute = name;
    $$('.view').forEach((v) => { v.hidden = v.dataset.view !== name; });
    $$('.nav-link').forEach((a) => { const on = a.dataset.nav === name; a.classList.toggle('active', on); if (on) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
    $('#page-title').textContent = def.title; document.title = `${def.title} · DataLens`;
    closeDrawer(); closeUserMenu();
    window.scrollTo({ top: 0 });
    Promise.resolve(def.load(arg)).catch((e) => { if (e.code !== 'auth_required') toast(e.message, 'error'); });
  }
  function leaveView(name) {
    if (name === 'dashboard') destroyCharts('dash-');
    if (name === 'analysis') { destroyCharts('ch-'); destroyCharts('ml-'); }
    if (name === 'admin-analytics') destroyCharts('ad-');
    if (name === 'api') destroyCharts('api-');
  }
  function openDrawer() { $('#sidebar').classList.add('open'); $('#scrim').hidden = false; $('#menu-btn').setAttribute('aria-expanded', 'true'); }
  function closeDrawer() { $('#sidebar').classList.remove('open'); $('#scrim').hidden = true; $('#menu-btn').setAttribute('aria-expanded', 'false'); }
  function closeUserMenu() { $('#user-menu').hidden = true; $('#user-menu-btn').setAttribute('aria-expanded', 'false'); }
  function wireShell() {
    $('#menu-btn').addEventListener('click', openDrawer);
    $('#drawer-close').addEventListener('click', closeDrawer);
    $('#scrim').addEventListener('click', closeDrawer);
    $('#user-menu-btn').addEventListener('click', (e) => {
      e.stopPropagation(); const m = $('#user-menu'); m.hidden = !m.hidden; $('#user-menu-btn').setAttribute('aria-expanded', String(!m.hidden));
    });
    document.addEventListener('click', (e) => {
      if (!e.target.closest('.user-menu-wrap')) closeUserMenu();
      const act = e.target.closest('[data-action]');
      if (act && act.dataset.action === 'logout') { closeUserMenu(); logout(); }
      const closer = e.target.closest('[data-close-dialog]'); if (closer) document.getElementById(closer.dataset.closeDialog).close();
      const focus = e.target.closest('[data-focus="password"]'); if (focus) setTimeout(() => $('#pw-current')?.focus(), 80);
    });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { closeDrawer(); closeUserMenu(); } });
    window.addEventListener('hashchange', route);
    $('#global-search-form').addEventListener('submit', (e) => {
      e.preventDefault();
      $('#datasets-search').value = $('#global-search').value.trim();
      if (location.hash === '#/datasets') loadDatasets(); else location.hash = '#/datasets';
    });
    $$('dialog').forEach((d) => d.addEventListener('click', (e) => { if (e.target === d && d.id !== 'dlg-force-pw') d.close('cancel'); }));
  }

  /* ===================== Datasets (shared) ===================== */
  async function refreshDatasets(q = '', sort = 'recent') {
    const d = await get(`/api/datasets?${qs({ q, sort })}`);
    if (!q) state.datasets = d.datasets;
    return d;
  }
  function datasetOptions(select, selected) {
    clear(select);
    for (const d of state.datasets) select.append(h('option', { value: d.id, text: d.name, selected: d.id === selected }));
  }
  const uploadCta = () => h('a', { class: 'btn btn-primary', href: '#/upload' }, ico('upload'), 'Upload a CSV');
  function openAnalysis(id) { state.an.id = id; location.hash = `#/analysis/${id}`; }
  async function renameDataset(ds) {
    const name = await promptDialog({ title: 'Rename dataset', label: 'Dataset name', value: ds.name, validate: (v) => v.trim() ? null : 'Enter a name.' });
    if (name === null) return false;
    try { await patch(`/api/datasets/${ds.id}`, { name: name.trim() }); toast('Dataset renamed.', 'success'); return true; }
    catch (e) { toast(e.message, 'error'); return false; }
  }
  async function deleteDataset(ds) {
    const ok = await confirmDialog({ title: `Delete “${ds.name}”?`, text: 'This permanently removes the file, cleaning steps, AI conversations, and any assistants built on it (their API keys stop working). This can’t be undone.', okLabel: 'Delete dataset' });
    if (!ok) return false;
    try { await del(`/api/datasets/${ds.id}`); toast('Dataset deleted.', 'success'); if (state.an.id === ds.id) state.an = { id: null, data: null, tab: 'overview' }; if (state.dashDs === ds.id) state.dashDs = null; return true; }
    catch (e) { toast(e.message, 'error'); return false; }
  }

  /* ===================== Dashboard ===================== */
  const EVENT_ICONS = { dataset_uploaded: ['upload', 'blue'], analysis_completed: ['chart-no-axes-column', ''], transform_applied: ['wand-sparkles', 'amber'],
    ai_request_success: ['sparkles', ''], ai_request_failed: ['triangle-alert', 'red'], csv_download: ['download', 'green'], report_download: ['file-text', 'green'],
    login: ['log-in', 'blue'], conversation_created: ['message-square', ''], model_trained: ['cpu', 'green'], ai_report_generated: ['sparkles', ''] };
  function statCard(iconName, tone, label, value, sub, subTone) {
    return h('div', { class: 'stat-card' }, h('div', { class: `stat-icon ${tone}` }, ico(iconName)),
      h('div', { class: 'stat-body' }, h('p', { class: 'stat-label', text: label }), h('p', { class: 'stat-value', text: value }),
        sub ? h('p', { class: `stat-sub ${subTone || ''}`, text: sub }) : null));
  }
  function changeText(c) {
    if (!c || c.change_pct === null) return c && c.last_7_days ? [`${c.last_7_days} in the last 7 days`, ''] : ['No activity in the last 7 days', ''];
    return [`${c.change_pct >= 0 ? '↑' : '↓'} ${Math.abs(c.change_pct)}% vs previous week`, c.change_pct >= 0 ? 'up' : 'down'];
  }
  async function loadDashboard() {
    const days = $('#dash-range').value;
    const [d] = await Promise.all([get(`/api/dashboard?days=${days}`), refreshDatasets()]);
    const hour = new Date().getHours();
    $('#dash-greeting').textContent = `Good ${hour < 12 ? 'morning' : hour < 18 ? 'afternoon' : 'evening'}, ${state.user.full_name.split(' ')[0]}`;
    const t = d.totals, c = d.changes;
    mount($('#dash-stats'),
      statCard('database', '', 'Total datasets', fmt(t.datasets), ...changeText(c.datasets)),
      statCard('chart-no-axes-column', 'blue', 'Total analyses', fmt(t.analyses), ...changeText(c.analyses)),
      statCard('sparkles', 'green', 'AI queries', fmt(t.ai_queries), ...changeText(c.ai_queries)),
      statCard('download', 'amber', 'Successful downloads', fmt(t.downloads), ...changeText(c.downloads)));
    // trend chart
    drawChart('dash-trend', $('#dash-trend'), { type: 'line', data: { labels: d.labels.map(fmtDay), datasets: [
      { label: 'Datasets uploaded', data: d.uploads, borderColor: '#7c5cff', backgroundColor: 'rgba(124,92,255,0.18)', fill: true, tension: 0.35, pointRadius: 3 },
      { label: 'Analyses completed', data: d.analyses, borderColor: '#38bdf8', backgroundColor: 'rgba(56,189,248,0.12)', fill: true, tension: 0.35, pointRadius: 3 }] },
      options: { interaction: { mode: 'index', intersect: false }, scales: { y: { beginAtZero: true, ticks: { precision: 0 } }, x: { grid: { display: false } } }, plugins: { legend: { position: 'bottom' } } } });
    // usage donut
    const usage = d.feature_usage, total = usage.reduce((a, b) => a + b.count, 0);
    mount($('#dash-usage-total'), total ? [h('strong', { text: fmt(total) }), h('span', { text: 'events' })] : [h('span', { text: 'No activity yet' })]);
    drawChart('dash-usage', $('#dash-usage'), { type: 'doughnut', data: { labels: usage.map((u) => u.label), datasets: [{ data: total ? usage.map((u) => u.count) : [1], backgroundColor: total ? COLORS : ['rgba(129,140,248,0.15)'], borderWidth: 0 }] },
      options: { cutout: '72%', plugins: { legend: { display: false }, tooltip: { enabled: !!total } } } });
    mount($('#dash-usage-legend'), usage.map((u, i) => h('li', {}, h('span', { class: 'dot', style: `background:${COLORS[i % COLORS.length]}` }), u.label, h('span', { class: 'val', text: total ? `${Math.round(u.count / total * 100)}%` : '0%' }))));
    // activity
    const act = $('#dash-activity');
    if (!d.recent_activity.length) mount(act, emptyState('activity', 'No activity yet', 'Upload a dataset to get started.'));
    else mount(act, d.recent_activity.map((e) => { const [i, tone] = EVENT_ICONS[e.type] || ['activity', '']; return h('li', {}, h('div', { class: `feed-ico ${tone}` }, ico(i)), h('div', { class: 'feed-text' }, h('strong', { text: e.label }), h('span', { text: e.dataset_name || '' })), h('span', { class: 'feed-time', text: timeAgo(e.at) })); }));
    // recent datasets
    const tbl = $('#dash-datasets'), emptyBox = $('#dash-datasets-empty');
    if (!d.recent_datasets.length) { clear(tbl); mount(emptyBox, emptyState('database', 'No datasets yet', 'Upload your first CSV to see statistics, charts and AI insights.', uploadCta())); }
    else {
      clear(emptyBox);
      mount(tbl, h('thead', {}, h('tr', {}, ...['Name', 'Rows', 'Columns', 'Uploaded', ''].map((x, i) => h('th', { class: i === 1 || i === 2 ? 'num' : '', text: x })))),
        h('tbody', {}, d.recent_datasets.map((ds) => h('tr', {}, h('td', { class: 'truncate' }, ico('file-spreadsheet'), h('a', { class: 'link', href: `#/analysis/${ds.id}`, text: ds.name })),
          h('td', { class: 'num', text: fmt(ds.rows) }), h('td', { class: 'num', text: fmt(ds.cols) }), h('td', { text: fmtDate(ds.created_at) }),
          h('td', {}, h('div', { class: 'btn-row' }, h('a', { class: 'icon-btn', href: `#/analysis/${ds.id}`, 'aria-label': `Open ${ds.name}` }, ico('eye')), h('button', { class: 'icon-btn', 'aria-label': `Ask AI about ${ds.name}`, onclick: () => { state.chat.dsId = ds.id; location.hash = '#/chat'; } }, ico('message-square'))))))));
    }
    // conversations
    const cv = $('#dash-convs');
    if (!d.recent_conversations.length) mount(cv, emptyState('message-square', 'No conversations yet', 'Ask the AI assistant a question about a dataset.', h('a', { class: 'btn btn-soft', href: '#/chat' }, 'Open AI chat')));
    else mount(cv, d.recent_conversations.map((c) => h('li', {}, h('button', { class: 'feed-item', onclick: () => { state.chat.dsId = c.dataset_id; state.chat.openConv = c.id; location.hash = '#/chat'; } },
      h('div', { class: 'feed-ico' }, ico('message-square')), h('div', { class: 'feed-text' }, h('strong', { text: c.title }), h('span', { text: c.preview || c.dataset_name || '' })), h('span', { class: 'feed-time', text: timeAgo(c.updated_at) })))));
    // profile
    mount($('#dash-profile'), h('div', { class: 'profile-row' }, h('span', { class: 'avatar', text: (state.user.full_name[0] || '?') }), h('div', { class: 'feed-text' }, h('strong', { text: state.user.full_name }), h('span', { text: `@${state.user.username}` }))),
      h('div', { class: 'profile-links' }, h('a', { href: '#/settings' }, 'Account settings'), h('a', { href: '#/settings', 'data-focus': 'password' }, 'Change password'), h('button', { class: 'danger', 'data-action': 'logout' }, ico('log-out'), 'Log out')));
    // quality + insights for the selected dataset
    const sel = $('#dash-ds-select');
    if (!state.datasets.length) { sel.hidden = true; mount($('#dash-quality'), emptyState('shield-check', 'Nothing to check yet', 'Quality metrics appear after you upload a dataset.')); mount($('#dash-insights'), emptyState('lightbulb', 'No insights yet', 'Insights are generated from your data.')); return; }
    sel.hidden = false;
    if (!state.datasets.some((x) => x.id === state.dashDs)) state.dashDs = (d.recent_analyses[0] && state.datasets.some((x) => x.id === d.recent_analyses[0].dataset_id)) ? d.recent_analyses[0].dataset_id : state.datasets[0].id;
    datasetOptions(sel, state.dashDs);
    await renderDashQuality();
  }
  async function renderDashQuality() {
    const box = $('#dash-quality'), ins = $('#dash-insights');
    mount(box, skeleton(3));
    try {
      const a = await get(`/api/datasets/${state.dashDs}/analysis`);
      const o = a.overview, r = 52, circ = 2 * Math.PI * r;
      const ring = h('div', { class: 'q-ring' });
      ring.innerHTML = `<svg viewBox="0 0 120 120" aria-hidden="true"><circle class="track" cx="60" cy="60" r="${r}"></circle><circle class="value" cx="60" cy="60" r="${r}" stroke-dasharray="${circ * o.completeness / 100} ${circ}"></circle></svg>`;
      ring.append(h('div', { class: 'center' }, h('strong', { text: `${Math.round(o.completeness)}%` }), h('span', { text: 'Completeness' })));
      const li = (color, label, val) => h('li', {}, h('span', { class: 'dot', style: `width:9px;height:9px;border-radius:50%;background:${color}` }), label, h('span', { class: 'val', text: fmt(val) }));
      mount(box, h('div', { class: 'quality' }, ring, h('ul', { class: 'q-list' }, li('#34d399', 'Complete rows', o.complete_rows), li('#fbbf24', 'Missing values', o.missing_cells), li('#f87171', 'Duplicate rows', o.duplicate_rows), li('#a78bfa', 'Potential outliers', o.outlier_cells))),
        h('div', { class: 'ai-promo' }, ico('sparkles'), h('p', { text: 'Ask the AI assistant about your data to find insights and patterns.' }), h('a', { class: 'btn btn-primary btn-sm', href: '#/chat', onclick: () => { state.chat.dsId = state.dashDs; } }, ico('send'))));
      $('#dash-insights-link').setAttribute('href', `#/analysis/${state.dashDs}`);
      const top = a.insights.filter((x) => x.kind !== 'interpretation').slice(0, 4);
      mount(ins, top.map((x) => h('li', {}, h('div', { class: `feed-ico ${x.severity === 'warning' ? 'amber' : x.severity === 'good' ? 'green' : 'blue'}` }, ico(x.severity === 'warning' ? 'triangle-alert' : 'lightbulb')), h('div', { class: 'feed-text' }, h('strong', { text: x.title }), h('span', { text: x.text })))));
    } catch (e) { mount(box, emptyState('triangle-alert', 'Couldn’t load this dataset', e.message)); mount(ins, ''); }
  }

  /* ===================== Upload ===================== */
  function wireUpload() {
    const dz = $('#dropzone'), input = $('#file-input');
    const pick = () => input.click();
    $('#pick-file').addEventListener('click', (e) => { e.stopPropagation(); pick(); });
    dz.addEventListener('click', pick);
    dz.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
    ['dragenter', 'dragover'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('drag'); }));
    ['dragleave', 'drop'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('drag'); }));
    dz.addEventListener('drop', (e) => { const f = e.dataTransfer.files[0]; if (f) uploadFile(f); });
    input.addEventListener('change', () => { if (input.files[0]) uploadFile(input.files[0]); input.value = ''; });
  }
  function loadUpload() {
    const l = state.limits;
    $('#upload-limits').textContent = `CSV, TSV or delimited text up to ${l.max_upload_bytes ? fmtBytes(l.max_upload_bytes) : '25 MB'}, ${l.max_rows ? fmt(l.max_rows) : '200,000'} rows and ${l.max_cols || 200} columns.`;
  }
  function uploadFile(file) {
    const result = $('#upload-result'); clear(result);
    const max = state.limits.max_upload_bytes;
    if (!/\.(csv|tsv|txt)$/i.test(file.name)) return mount(result, h('div', { class: 'result-box error', role: 'alert' }, h('strong', { text: 'That file type isn’t supported.' }), h('p', { class: 'muted', text: 'Choose a .csv file (or a .tsv/.txt file with delimited text).' })));
    if (max && file.size > max) return mount(result, h('div', { class: 'result-box error', role: 'alert' }, h('strong', { text: 'That file is too large.' }), h('p', { class: 'muted', text: `The limit is ${fmtBytes(max)}; this file is ${fmtBytes(file.size)}.` })));
    const wrap = $('#upload-progress'), bar = $('#upload-bar'), status = $('#upload-status');
    wrap.hidden = false; $('#upload-file-name').textContent = file.name; $('#upload-pct').textContent = '0%'; $('span', bar).style.width = '0%'; bar.setAttribute('aria-valuenow', '0');
    status.textContent = 'Uploading…';
    const fd = new FormData(); fd.append('file', file);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/datasets'); xhr.setRequestHeader('X-CSRFToken', state.csrf); xhr.setRequestHeader('Accept', 'application/json');
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      const p = Math.round(e.loaded / e.total * 100);
      $('span', bar).style.width = `${p}%`; bar.setAttribute('aria-valuenow', String(p)); $('#upload-pct').textContent = `${p}%`;
      if (p === 100) status.textContent = 'Validating and analysing…';
    };
    const fail = (msg) => { wrap.hidden = true; mount(result, h('div', { class: 'result-box error', role: 'alert' }, h('strong', { text: 'The upload didn’t work.' }), h('p', { class: 'muted', text: msg }))); };
    xhr.onerror = () => fail('Could not reach the server. Check your connection and try again.');
    xhr.onload = async () => {
      let data = null; try { data = JSON.parse(xhr.responseText); } catch (_) { /* ignore */ }
      if (data && data.csrf_token) state.csrf = data.csrf_token;
      if (xhr.status === 401) return onSessionExpired();
      if (xhr.status >= 200 && xhr.status < 300 && data && data.ok) {
        wrap.hidden = true; const ds = data.dataset;
        mount(result, h('div', { class: 'result-box' }, h('div', { class: 'btn-row' }, h('span', { class: 'badge badge-ok' }, ico('circle-check'), 'Upload complete'), h('strong', { text: ds.name })),
          h('p', { class: 'muted', text: `${fmt(ds.rows)} rows · ${fmt(ds.cols)} columns · ${fmtBytes(ds.size_bytes)} · ${data.encoding}, delimiter “${data.delimiter === '\t' ? 'tab' : data.delimiter}”` }),
          data.warnings.length ? h('ul', {}, data.warnings.map((w) => h('li', { text: w }))) : null,
          h('div', { class: 'btn-row' }, h('a', { class: 'btn btn-primary', href: `#/analysis/${ds.id}` }, ico('chart-no-axes-column'), 'Open analysis'),
            h('button', { class: 'btn btn-soft', onclick: () => { state.chat.dsId = ds.id; location.hash = '#/chat'; } }, ico('message-square'), 'Ask AI'))));
        toast('Dataset uploaded.', 'success'); refreshDatasets().catch(() => {});
      } else fail((data && data.error && data.error.message) || 'The server rejected the file.');
    };
    xhr.send(fd);
  }

  /* ===================== My datasets ===================== */
  async function loadDatasets() {
    const q = $('#datasets-search').value.trim(), sort = $('#datasets-sort').value, list = $('#datasets-list');
    mount(list, skeleton(2));
    const d = await refreshDatasets(q, sort);
    $('#datasets-count').textContent = `${d.total} dataset${d.total === 1 ? '' : 's'}${q ? ` · ${d.datasets.length} match “${q}”` : ''}`;
    if (!d.datasets.length) return mount(list, h('div', { class: 'card', style: 'grid-column:1/-1' }, q ? emptyState('search', 'No datasets match', 'Try a different name.') : emptyState('database', 'No datasets yet', 'Upload a CSV to start exploring your data.', uploadCta())));
    mount(list, d.datasets.map((ds) => h('article', { class: 'card dataset-card' },
      h('div', { class: 'ds-top' }, h('div', { class: 'stat-icon green' }, ico('file-spreadsheet')), h('div', {}, h('h3', { text: ds.name, title: ds.name }), h('p', { class: 'muted small', text: ds.filename }))),
      h('div', { class: 'ds-meta' }, h('span', { class: 'badge', text: `${fmt(ds.rows)} rows` }), h('span', { class: 'badge', text: `${fmt(ds.cols)} columns` }), h('span', { class: 'badge', text: fmtBytes(ds.size_bytes) }), ds.has_transforms ? h('span', { class: 'badge badge-purple', text: `${ds.transform_count} cleaning step${ds.transform_count === 1 ? '' : 's'}` }) : null),
      h('p', { class: 'muted small', text: `Uploaded ${fmtDate(ds.created_at)}` }),
      h('div', { class: 'ds-actions' },
        h('a', { class: 'btn btn-primary btn-sm', href: `#/analysis/${ds.id}` }, ico('chart-no-axes-column'), 'Analyze'),
        h('button', { class: 'btn btn-soft btn-sm', onclick: () => { state.chat.dsId = ds.id; location.hash = '#/chat'; } }, ico('message-square'), 'Ask AI'),
        h('button', { class: 'icon-btn', 'aria-label': `Rename ${ds.name}`, onclick: async () => { if (await renameDataset(ds)) loadDatasets(); } }, ico('pencil')),
        h('button', { class: 'icon-btn', 'aria-label': `Download original ${ds.name}`, onclick: (e) => downloadFile(`/api/datasets/${ds.id}/download/original`, e.currentTarget) }, ico('download')),
        h('button', { class: 'icon-btn', 'aria-label': `Delete ${ds.name}`, onclick: async () => { if (await deleteDataset(ds)) loadDatasets(); } }, ico('trash-2'))))));
  }

  /* ===================== Analysis workspace ===================== */
  const AGGS = [['sum', 'Sum'], ['mean', 'Average'], ['median', 'Median'], ['min', 'Minimum'], ['max', 'Maximum'], ['count', 'Count'], ['std', 'Std. deviation']];
  const OPS_BY_KIND = {
    numeric: [['eq', 'equals'], ['ne', 'does not equal'], ['gt', 'is greater than'], ['ge', 'is at least'], ['lt', 'is less than'], ['le', 'is at most'], ['isnull', 'is empty'], ['notnull', 'is not empty']],
    categorical: [['eq', 'equals'], ['ne', 'does not equal'], ['contains', 'contains'], ['not_contains', 'does not contain'], ['startswith', 'starts with'], ['isnull', 'is empty'], ['notnull', 'is not empty']],
  };
  OPS_BY_KIND.datetime = OPS_BY_KIND.numeric;
  const A = () => state.an.data;
  const cols = (...kinds) => A().columns.filter((c) => kinds.includes(c.kind));

  function typeBadge(p) {
    if (p.mixed) return h('span', { class: 'badge badge-warn', title: 'Contains both numbers and text', text: 'mixed' });
    const tone = p.kind === 'numeric' ? 'badge-blue' : p.kind === 'datetime' ? 'badge-purple' : '';
    return h('span', { class: `badge ${tone}`, text: p.detail || p.kind });
  }
  function missingBar(pct) { return h('div', { class: 'bar-cell' }, h('div', { class: `mini-bar ${pct >= 20 ? 'warn' : ''}` }, h('span', { style: `width:${Math.min(100, pct)}%` })), h('span', { class: 'small', text: `${pct}%` })); }

  function sortableTable(table, columns, rows, init = {}) {
    let key = init.key || null, dir = init.dir || 'desc';
    const render = () => {
      const col = columns.find((c) => c.key === key);
      let data = rows;
      if (col) {
        data = [...rows].sort((a, b) => { const x = col.value(a), y = col.value(b);
          if (x === y) return 0; if (x === null || x === undefined) return 1; if (y === null || y === undefined) return -1;
          return (typeof x === 'string' ? x.localeCompare(y) : x - y) * (dir === 'asc' ? 1 : -1); });
      }
      mount(table, h('thead', {}, h('tr', {}, columns.map((c) => h('th', { class: `${c.num ? 'num' : ''} sort`, tabindex: 0, role: 'columnheader', 'aria-sort': c.key === key ? (dir === 'asc' ? 'ascending' : 'descending') : 'none',
        onclick: () => { if (key === c.key) dir = dir === 'asc' ? 'desc' : 'asc'; else { key = c.key; dir = c.num ? 'desc' : 'asc'; } render(); },
        onkeydown: (e) => { if (e.key === 'Enter') e.currentTarget.click(); } }, c.label, h('span', { class: 'arrow', text: c.key === key ? (dir === 'asc' ? '▲' : '▼') : '' }))))),
        h('tbody', {}, data.map((r) => h('tr', {}, columns.map((c) => h('td', { class: c.num ? 'num' : '' }, c.cell ? c.cell(r) : String(c.value(r) ?? '—')))))));
    };
    render();
  }
  const badgeCount = (n, warn) => h('span', { class: `badge ${n ? (warn ? 'badge-warn' : '') : 'badge-ok'}`, text: fmt(n) });

  async function loadAnalysis(arg) {
    const empty = $('#an-empty'), body = $('#an-body'), actions = $('#an-actions'), sel = $('#an-select');
    await refreshDatasets();
    if (!state.datasets.length) {
      body.hidden = true; actions.hidden = true; sel.hidden = true; $('#an-meta').textContent = '';
      return mount(empty, h('div', { class: 'card' }, emptyState('chart-no-axes-column', 'No dataset to analyze', 'Upload a CSV to see statistics, charts, data-quality checks and insights.', uploadCta())));
    }
    clear(empty); sel.hidden = false;
    let id = arg || state.an.id;
    if (arg && !state.datasets.some((d) => d.id === arg)) { toast('That dataset doesn’t exist.', 'error'); id = state.datasets[0].id; }
    if (!state.datasets.some((d) => d.id === id)) id = state.datasets[0].id;
    datasetOptions(sel, id);
    if (arg !== id) history.replaceState(null, '', `#/analysis/${id}`);
    await loadAnalysisData(id);
  }

  async function loadAnalysisData(id) {
    state.an.id = id; body_loading(true);
    try {
      const data = await get(`/api/datasets/${id}/analysis`);
      state.an.data = data; state.an.ex = { page: 1, size: Number($('#ex-size').value) || 25, q: '', sort: null, dir: 'asc', filters: [], cols: null, dups: false };
      state.an.chart = null; state.an.autoDrawn = false;
      $('#ex-search').value = ''; $('#ex-dups').checked = false; clear($('#ex-filters')); $('#ex-col-picker').hidden = true;
      $('#an-meta').textContent = `${data.dataset.filename} · uploaded ${fmtDate(data.dataset.created_at)} · ${fmtBytes(data.dataset.size_bytes)}`;
      $('#an-actions').hidden = false; $('#an-body').hidden = false;
      renderOverview(); renderStatistics(); renderQuality(); renderInsights(); setupCharts(); setupClean(); resetAiTabs();
      showTab(state.an.tab === 'explorer' ? 'explorer' : state.an.tab);
    } catch (e) { $('#an-body').hidden = true; mount($('#an-empty'), h('div', { class: 'card' }, emptyState('triangle-alert', 'This dataset couldn’t be opened', e.message))); }
    finally { body_loading(false); }
  }
  function body_loading(on) { if (on) { clear($('#an-empty')); mount($('#ov-stats'), skeleton(1)); } }

  function showTab(tab) {
    state.an.tab = tab;
    $$('#an-tabs .tab').forEach((t) => { const on = t.dataset.tab === tab; t.classList.toggle('active', on); t.setAttribute('aria-selected', on); });
    $$('.tab-panel').forEach((p) => { p.hidden = p.dataset.panel !== tab; });
    if (tab === 'explorer') loadRows();
    if (tab === 'aireport' && A()) loadAiReport();
    if (tab === 'train' && A()) loadTrainTab();
    if (tab === 'charts' && !state.an.autoDrawn && A()) { state.an.autoDrawn = true; const s = A().suggestions[0]; if (s) applySuggestion(s, false); }
  }

  /* ----- Overview ----- */
  function renderOverview() {
    const a = A(), o = a.overview;
    mount($('#ov-stats'),
      statCard('rows-3', '', 'Rows', fmt(o.rows)), statCard('columns-3', 'blue', 'Columns', fmt(o.cols)),
      statCard('badge-check', 'green', 'Completeness', `${o.completeness}%`), statCard('circle-help', 'amber', 'Missing cells', fmt(o.missing_cells), `${o.missing_pct}% of all cells`),
      statCard('copy', 'red', 'Duplicate rows', fmt(o.duplicate_rows)), statCard('memory-stick', '', 'Memory', fmtBytes(o.memory_bytes), 'while analysing'));
    const e = o.explain;
    mount($('#ov-explain'), [e.types, e.completeness, e.missing, e.duplicates, e.outliers].map((t) => h('li', { text: t })));
    mount($('#ov-columns'), h('thead', {}, h('tr', {}, ['Column', 'Type', 'Filled', 'Missing', 'Distinct'].map((x, i) => h('th', { class: i > 1 && i !== 3 ? 'num' : '', text: x })))),
      h('tbody', {}, a.columns.map((p) => h('tr', {}, h('td', { class: 'truncate', title: p.name, text: p.name }), h('td', {}, typeBadge(p)), h('td', { class: 'num', text: fmt(p.non_null) }), h('td', {}, missingBar(p.missing_pct)), h('td', { class: 'num', text: fmt(p.unique) })))));
    const sug = $('#ov-suggest');
    if (!a.suggestions.length) mount(sug, h('p', { class: 'muted', text: 'No chart suggestions for this dataset.' }));
    else mount(sug, a.suggestions.map((s) => h('button', { class: 'chip', onclick: () => applySuggestion(s, true) }, h('strong', { text: s.title }), h('span', { text: s.reason }))));
    const id = a.dataset.id, dl = (kind, label, icon) => h('button', { class: 'btn btn-soft', onclick: (ev) => downloadFile(`/api/datasets/${id}/download/${kind}`, ev.currentTarget) }, ico(icon), label);
    mount($('#ov-downloads'), dl('original', 'Original CSV', 'file-spreadsheet'), dl('cleaned', a.dataset.has_transforms ? 'Cleaned CSV' : 'Data as analysed (CSV)', 'wand-sparkles'), dl('summary', 'Summary statistics CSV', 'sigma'), dl('missing', 'Missing-value report CSV', 'circle-help'), dl('report', 'PDF report', 'file-down'));
  }

  /* ----- Statistics ----- */
  function renderStatistics() {
    const a = A(), nums = a.columns.filter((c) => c.kind === 'numeric' && c.stats);
    const sn = (k) => (r) => r.stats[k];
    const numCols = [{ key: 'name', label: 'Column', value: (r) => r.name, cell: (r) => h('span', { title: r.name, text: r.name }) }, { key: 'count', label: 'Valid', num: true, value: sn('count'), cell: (r) => fmt(r.stats.count) }];
    for (const [k, l] of [['mean', 'Mean'], ['median', 'Median'], ['std', 'Std. dev.'], ['min', 'Min'], ['q1', 'Q1'], ['q3', 'Q3'], ['max', 'Max'], ['iqr', 'IQR']]) numCols.push({ key: k, label: l, num: true, value: sn(k), cell: (r) => fmt(r.stats[k], 3) });
    if (nums.length) { clear($('#st-numeric-empty')); sortableTable($('#st-numeric'), numCols, nums, { key: 'name', dir: 'asc' }); }
    else { clear($('#st-numeric')); mount($('#st-numeric-empty'), emptyState('sigma', 'No numeric columns', 'Numeric statistics appear when a column contains numbers only.')); }
    const cats = a.columns.filter((c) => c.kind === 'categorical');
    if (cats.length) {
      clear($('#st-cats-empty'));
      mount($('#st-cats'), cats.map((p) => h('div', { class: 'cat-card' }, h('h4', { title: p.name, text: p.name }),
        h('div', { class: 'btn-row' }, typeBadge(p), h('span', { class: 'small muted', text: `${fmt(p.non_null)} filled · ${fmt(p.unique)} distinct` })),
        ...(p.top_values || []).slice(0, 6).map((t) => h('div', { class: 'cat-row' }, h('span', { class: 'name', title: t.value, text: t.value }), h('div', { class: 'mini-bar' }, h('span', { style: `width:${t.pct}%` })), h('span', { class: 'cnt', text: `${fmt(t.count)} (${t.pct}%)` }))),
        p.unique > 6 ? h('span', { class: 'small muted', text: `…and ${fmt(p.unique - 6)} more values` }) : (p.non_null ? null : h('span', { class: 'small muted', text: 'This column is empty.' })))));
    } else { clear($('#st-cats')); mount($('#st-cats-empty'), emptyState('list', 'No categorical columns', 'Text columns with repeated values appear here.')); }
    const dts = a.columns.filter((c) => c.kind === 'datetime');
    if (dts.length) {
      clear($('#st-dates-empty'));
      mount($('#st-dates'), h('thead', {}, h('tr', {}, ['Column', 'Earliest', 'Latest', 'Span', 'Valid', 'Missing'].map((x) => h('th', { text: x })))),
        h('tbody', {}, dts.map((p) => h('tr', {}, h('td', { text: p.name }), h('td', { text: p.stats ? p.stats.min.slice(0, 10) : '—' }), h('td', { text: p.stats ? p.stats.max.slice(0, 10) : '—' }), h('td', { text: p.stats ? `${fmt(p.stats.span_days)} days` : '—' }), h('td', { text: fmt(p.non_null) }), h('td', { text: fmt(p.missing) })))));
    } else { clear($('#st-dates')); mount($('#st-dates-empty'), emptyState('calendar', 'No date columns', 'Columns whose values are all valid dates are detected automatically.')); }
  }

  /* ----- Quality ----- */
  function renderQuality() {
    const a = A(), q = a.quality, o = a.overview;
    mount($('#q-stats'), statCard('circle-help', 'amber', 'Missing cells', fmt(q.missing_cells), `${q.missing_pct}% of cells`), statCard('rows-3', 'blue', 'Rows with missing values', fmt(q.rows_with_missing), `${fmt(q.complete_rows)} complete rows`),
      statCard('copy', 'red', 'Duplicate rows', fmt(q.duplicate_rows), 'exact repeats of an earlier row'), statCard('activity', '', 'Potential outliers', fmt(o.outlier_cells), 'IQR rule'));
    sortableTable($('#q-missing'), [
      { key: 'column', label: 'Column', value: (r) => r.column }, { key: 'kind', label: 'Type', value: (r) => r.kind },
      { key: 'missing', label: 'Missing', num: true, value: (r) => r.missing, cell: (r) => fmt(r.missing) },
      { key: 'pct', label: 'Share missing', value: (r) => r.pct, cell: (r) => missingBar(r.pct) }], q.missing_by_column, { key: 'missing', dir: 'desc' });
    const rows = [];
    rows.push(['Duplicate rows', q.duplicate_rows ? `${fmt(q.duplicate_rows)}. First at row ${q.duplicate_row_numbers.slice(0, 8).join(', ')}${q.duplicate_rows > 8 ? '…' : ''}` : 'None found']);
    rows.push(['Constant columns', q.constant_columns.length ? q.constant_columns.join(', ') : 'None']);
    rows.push(['Empty columns', q.empty_columns.length ? q.empty_columns.join(', ') : 'None']);
    rows.push(['Mixed-type columns', q.mixed_type_columns.length ? q.mixed_type_columns.map((m) => `${m.column} (${m.numeric_fraction}% numeric)`).join(', ') : 'None']);
    const struct = h('div', { class: 'kv' }, rows.map(([k, v]) => h('div', { class: 'kv-row' }, h('span', { text: k }), h('span', { text: v }))));
    const extra = [];
    if (q.duplicate_rows) extra.push(h('button', { class: 'btn btn-soft btn-sm', onclick: () => { state.an.ex.dups = true; $('#ex-dups').checked = true; state.an.ex.page = 1; showTab('explorer'); } }, ico('eye'), 'Show duplicates'));
    if (q.mixed_type_columns.length) extra.push(h('button', { class: 'btn btn-soft btn-sm', onclick: () => { showTab('clean'); $('#cl-op').value = 'convert_type'; renderCleanFields(); } }, ico('wand-sparkles'), 'Convert in Clean tab'));
    mount($('#q-struct'), struct, extra.length ? h('div', { class: 'btn-row', style: 'margin-top:12px' }, extra) : null);
    $('#q-outlier-note').textContent = q.outlier_note;
    if (q.outliers.length) { clear($('#q-outliers-empty')); mount($('#q-outliers'), h('thead', {}, h('tr', {}, ['Column', 'Outliers', 'Share', 'Expected range', 'Most extreme'].map((x) => h('th', { text: x })))),
      h('tbody', {}, q.outliers.map((r) => h('tr', {}, h('td', { text: r.column }), h('td', { text: `${fmt(r.count)} (${r.low} low, ${r.high} high)` }), h('td', { text: `${r.pct}%` }), h('td', { text: `${fmt(r.lower_bound, 3)} to ${fmt(r.upper_bound, 3)}` }), h('td', { class: 'truncate', text: r.examples.join(', ') }))))); }
    else { clear($('#q-outliers')); mount($('#q-outliers-empty'), emptyState('circle-check', 'No potential outliers', 'No numeric column has values outside the IQR range.')); }
  }

  /* ----- Insights ----- */
  function renderInsights() {
    const items = A().insights, kindTone = { observation: 'badge-blue', interpretation: 'badge-purple', suggestion: 'badge-ok' };
    mount($('#in-list'), items.length ? items.map((i) => h('li', { class: `insight ${i.severity}` }, h('div', { class: `feed-ico ${i.severity === 'warning' ? 'amber' : i.severity === 'good' ? 'green' : 'blue'}` }, ico(i.severity === 'warning' ? 'triangle-alert' : i.kind === 'interpretation' ? 'brain' : 'lightbulb')),
      h('div', {}, h('h4', {}, i.title, h('span', { class: `badge ${kindTone[i.kind]}`, text: i.kind })), h('p', { text: i.text })))) : [emptyState('lightbulb', 'No insights yet', 'Upload more data to see insights.')]);
  }

  /* ----- Charts ----- */
  const CHART_TYPES = [['histogram', 'Histogram'], ['bar', 'Category frequency'], ['compare', 'Compare groups'], ['line', 'Time series'], ['scatter', 'Scatter plot'], ['correlation', 'Correlation heatmap'], ['missing', 'Missing values']];
  function availableChartTypes() {
    const num = cols('numeric').length, nonDate = cols('numeric', 'categorical').length, dates = cols('datetime').length;
    return { histogram: num >= 1, bar: nonDate >= 1, compare: nonDate >= 1 && num >= 1, line: dates >= 1, scatter: num >= 2, correlation: num >= 2, missing: true };
  }
  const selectField = (name, label, options, value, scope = 'chf') => h('div', { class: 'field' }, h('label', { for: `${scope}-${name}`, text: label }),
    h('select', { id: `${scope}-${name}`, 'data-param': name }, options.map((o) => { const [v, l] = Array.isArray(o) ? o : [o, o || '(none)']; return h('option', { value: v, text: l, selected: v === value }); })));
  const numberField = (name, label, ph, value) => h('div', { class: 'field' }, h('label', { for: `chf-${name}`, text: label }), h('input', { id: `chf-${name}`, type: 'number', min: 3, max: 100, 'data-param': name, placeholder: ph, value: value || '' }));

  function setupCharts() {
    const avail = availableChartTypes(), type = $('#ch-type');
    clear(type);
    for (const [v, l] of CHART_TYPES) type.append(h('option', { value: v, text: l + (avail[v] ? '' : ' (not available)'), disabled: !avail[v] }));
    const first = CHART_TYPES.find(([v]) => avail[v]); type.value = first ? first[0] : 'missing';
    renderChartFields();
    const sug = A().suggestions;
    mount($('#ch-suggest'), sug.length ? sug.map((s) => h('button', { class: 'chip', onclick: () => applySuggestion(s, true) }, h('strong', { text: s.title }), h('span', { text: s.reason }))) : h('p', { class: 'muted', text: 'No suggestions available.' }));
    mount($('#ch-message'), ''); $('#ch-box').hidden = true; $('#ch-heatmap').hidden = true; $('#ch-note').textContent = ''; $('#ch-title').textContent = 'Choose a chart'; $('#ch-ask').hidden = true;
    destroyCharts('ch-');
  }
  function renderChartFields(preset = {}) {
    const t = $('#ch-type').value, num = cols('numeric').map((c) => c.name), dates = cols('datetime').map((c) => c.name);
    const rank = (c) => (c.kind === 'categorical' ? (['category', 'boolean'].includes(c.detail) ? 0 : 2) : (c.unique <= 30 ? 1 : 3));
    const any = cols('numeric', 'categorical').slice().sort((a, b) => rank(a) - rank(b)).map((c) => c.name);
    const f = [];
    if (t === 'histogram') f.push(selectField('column', 'Numeric column', num, preset.column), numberField('bins', 'Bins (optional)', 'auto', preset.bins));
    if (t === 'bar') f.push(selectField('column', 'Column', any, preset.column), numberField('top_n', 'Show top', '15', preset.top_n));
    if (t === 'compare') f.push(selectField('group_column', 'Group by', any, preset.group_column), selectField('value_column', 'Value', num, preset.value_column), selectField('agg', 'Calculation', AGGS, preset.agg || 'mean'));
    if (t === 'line') f.push(selectField('date_column', 'Date column', dates, preset.date_column), selectField('value_column', 'Value (optional)', ['', ...num], preset.value_column), selectField('agg', 'Calculation', AGGS, preset.agg || 'mean'), selectField('freq', 'Group by', [['', 'Automatic'], ['D', 'Day'], ['W', 'Week'], ['M', 'Month'], ['Q', 'Quarter'], ['Y', 'Year']], preset.freq));
    if (t === 'scatter') f.push(selectField('x', 'X axis', num, preset.x), selectField('y', 'Y axis', num, preset.y || num[1]));
    mount($('#ch-fields'), f);
  }
  function applySuggestion(s, scroll) {
    if (state.an.tab !== 'charts') { state.an.autoDrawn = true; showTab('charts'); }
    $('#ch-type').value = s.type; renderChartFields(s.params); drawSelectedChart();
    if (scroll) $('#ch-title').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  async function drawSelectedChart() {
    const type = $('#ch-type').value, params = {};
    $$('#ch-fields [data-param]').forEach((el) => { if (el.value !== '') params[el.dataset.param] = el.value; });
    const msg = $('#ch-message'), btn = $('#ch-draw');
    setBusy(btn, true);
    try {
      const d = await get(`/api/datasets/${A().dataset.id}/chart?${qs({ type, ...params })}`);
      $('#ch-box').hidden = true; $('#ch-heatmap').hidden = true; destroyCharts('ch-');
      if (d.unsuitable) { state.an.chart = null; $('#ch-ask').hidden = true; $('#ch-title').textContent = 'This chart isn’t suitable'; $('#ch-note').textContent = ''; return mount(msg, emptyState('info', 'Can’t draw this chart', d.message)); }
      clear(msg); state.an.chart = { type, params };
      renderChart(d.chart); $('#ch-ask').hidden = false;
    } catch (e) { mount(msg, emptyState('triangle-alert', 'The chart couldn’t be drawn', e.message)); }
    finally { setBusy(btn, false); }
  }
  function renderChart(c) {
    $('#ch-title').textContent = c.title; $('#ch-note').textContent = c.note || (c.summary ? `${fmt(c.summary.count)} values · mean ${fmt(c.summary.mean, 3)} · median ${fmt(c.summary.median, 3)} · range ${fmt(c.summary.min, 3)} to ${fmt(c.summary.max, 3)}` : '');
    if (c.type === 'correlation') return renderHeatmap(c);
    $('#ch-box').hidden = false;
    const axis = (t, extra) => ({ title: { display: !!t, text: t }, ...extra });
    const grid = { grid: { display: false } };
    let cfg;
    if (c.type === 'scatter') {
      const ds = [{ label: 'Rows', data: c.points, backgroundColor: 'rgba(124,92,255,0.55)', pointRadius: 3 }];
      if (c.line) ds.push({ label: 'Trend line', type: 'line', data: c.line, borderColor: '#38bdf8', borderWidth: 2, pointRadius: 0, borderDash: [6, 4] });
      cfg = { type: 'scatter', data: { datasets: ds }, options: { scales: { x: axis(c.x_label), y: axis(c.y_label) }, plugins: { legend: { position: 'bottom' } } } };
    } else if (c.type === 'line') {
      cfg = { type: 'line', data: { labels: c.labels, datasets: [{ label: c.datasets[0].label, data: c.datasets[0].data, borderColor: '#7c5cff', backgroundColor: 'rgba(124,92,255,0.16)', fill: true, tension: 0.25, pointRadius: c.labels.length > 60 ? 0 : 3 }] },
        options: { scales: { x: axis(c.x_label, { ticks: { maxTicksLimit: 12 }, ...grid }), y: axis(c.y_label) }, plugins: { legend: { position: 'bottom' } } } };
    } else {
      const horizontal = c.type === 'missing';
      cfg = { type: 'bar', data: { labels: c.labels, datasets: [{ label: c.datasets[0].label, data: c.datasets[0].data, backgroundColor: c.type === 'histogram' ? 'rgba(124,92,255,0.75)' : c.labels.map((_, i) => COLORS[i % COLORS.length] + 'cc'), borderRadius: 4, borderWidth: 0, categoryPercentage: c.type === 'histogram' ? 1 : 0.8, barPercentage: c.type === 'histogram' ? 0.98 : 0.8 }] },
        options: { indexAxis: horizontal ? 'y' : 'x', scales: { x: axis(horizontal ? c.y_label : c.x_label, { ticks: { maxRotation: 55, autoSkip: true }, ...grid }), y: axis(horizontal ? '' : c.y_label, { beginAtZero: true }) },
          plugins: { legend: { position: 'bottom' }, tooltip: c.percent ? { callbacks: { afterLabel: (i) => `${c.percent[i.dataIndex]}% of rows` } } : {} } } };
    }
    drawChart('ch-main', $('#ch-canvas'), cfg);
  }
  function renderHeatmap(c) {
    const box = $('#ch-heatmap'), n = c.columns.length; box.hidden = false;
    const grid = h('div', { class: 'heatmap', role: 'table', 'aria-label': 'Correlation matrix' });
    grid.style.gridTemplateColumns = `130px repeat(${n}, 56px)`;
    grid.append(h('div', {}));
    c.columns.forEach((n2) => grid.append(h('div', { class: 'heat-label col', title: n2, text: n2 })));
    c.matrix.forEach((row, i) => {
      grid.append(h('div', { class: 'heat-label', title: c.columns[i], text: c.columns[i] }));
      row.forEach((v) => {
        const cell = h('div', { class: 'heat-cell', title: v === null ? 'n/a' : v.toFixed(3), text: v === null ? '–' : v.toFixed(2) });
        if (v !== null) { const a = Math.min(1, Math.abs(v)); cell.style.background = v >= 0 ? `rgba(124,92,255,${0.12 + a * 0.78})` : `rgba(56,189,248,${0.12 + a * 0.78})`; cell.style.color = a > 0.5 ? '#fff' : '#c4c9e6'; }
        grid.append(cell);
      });
    });
    mount(box, grid);
  }

  /* ----- Explorer ----- */
  const EX = () => state.an.ex;
  async function loadRows() {
    const ex = EX(), id = A().dataset.id, table = $('#ex-table');
    const params = { page: ex.page, page_size: ex.size, q: ex.q, sort: ex.sort, dir: ex.dir, dups: ex.dups ? 'only' : '', cols: ex.cols ? ex.cols.join(',') : '' };
    const filters = ex.filters.filter((f) => f.column && (f.operator === 'isnull' || f.operator === 'notnull' || f.value !== ''));
    if (filters.length) params.filters = JSON.stringify(filters);
    try {
      const d = await get(`/api/datasets/${id}/rows?${qs(params)}`);
      const head = h('tr', {}, h('th', { class: 'num', text: '#' }), d.columns.map((c) => h('th', { class: 'sort', tabindex: 0, 'aria-sort': ex.sort === c.name ? (ex.dir === 'asc' ? 'ascending' : 'descending') : 'none', title: `${c.kind}, ${c.missing} missing`,
        onclick: () => { if (ex.sort === c.name) ex.dir = ex.dir === 'asc' ? 'desc' : 'asc'; else { ex.sort = c.name; ex.dir = 'asc'; } ex.page = 1; loadRows(); }, onkeydown: (e) => { if (e.key === 'Enter') e.currentTarget.click(); } },
        c.name, h('span', { class: 'arrow', text: ex.sort === c.name ? (ex.dir === 'asc' ? '▲' : '▼') : '' }))));
      const rows = d.rows.map((r) => h('tr', { class: r.dup ? 'dup' : '', tabindex: 0, title: r.dup ? 'Duplicate of an earlier row' : '', onclick: () => openRow(r.n - 1), onkeydown: (e) => { if (e.key === 'Enter') openRow(r.n - 1); } },
        h('td', { class: 'rn', text: r.n }), r.cells.map((v) => v === null ? h('td', { class: 'null-cell', text: 'empty' }) : h('td', { title: v, text: v }))));
      mount(table, h('thead', {}, head), h('tbody', {}, rows.length ? rows : [h('tr', {}, h('td', { colspan: d.columns.length + 1 }, emptyState('search', 'No rows match', 'Try clearing the search or filters.')))]));
      const from = d.total ? (d.page - 1) * d.page_size + 1 : 0, to = Math.min(d.total, d.page * d.page_size);
      mount($('#ex-pager'), h('span', { text: `Showing ${fmt(from)}–${fmt(to)} of ${fmt(d.total)}${d.total !== d.total_unfiltered ? ` (filtered from ${fmt(d.total_unfiltered)})` : ''}` }),
        h('div', { class: 'btn-row' }, h('button', { class: 'btn btn-soft btn-sm', disabled: d.page <= 1, onclick: () => { ex.page--; loadRows(); } }, ico('chevron-left'), 'Previous'),
          h('span', { text: `Page ${d.page} of ${d.pages}` }), h('button', { class: 'btn btn-soft btn-sm', disabled: d.page >= d.pages, onclick: () => { ex.page++; loadRows(); } }, 'Next', ico('chevron-right'))));
    } catch (e) { mount(table, h('tbody', {}, h('tr', {}, h('td', {}, emptyState('triangle-alert', 'Couldn’t load rows', e.message))))); clear($('#ex-pager')); }
  }
  async function openRow(i) {
    try {
      const d = await get(`/api/datasets/${A().dataset.id}/row/${i}`);
      $('#dlg-row-title').textContent = `Row ${fmt(d.row_number)}${d.duplicate ? ' · duplicate' : ''}`;
      mount($('#dlg-row-body'), h('div', { class: 'kv' }, d.cells.map((c) => h('div', { class: 'kv-row' }, h('span', { text: c.column }), h('span', { class: c.value === null ? 'null-cell' : '', text: c.value === null ? 'empty' : c.value })))));
      $('#dlg-row').showModal();
    } catch (e) { toast(e.message, 'error'); }
  }
  function addFilterRow() {
    const f = { column: A().columns[0].name, operator: 'eq', value: '' }; EX().filters.push(f);
    const row = h('div', { class: 'filter-row' });
    const colSel = h('select', { 'aria-label': 'Filter column' }, A().columns.map((c) => h('option', { value: c.name, text: c.name })));
    const opSel = h('select', { 'aria-label': 'Filter condition' }); const val = h('input', { type: 'text', 'aria-label': 'Filter value', placeholder: 'Value' });
    const fillOps = () => { const kind = A().columns.find((c) => c.name === f.column).kind; mount(opSel, (OPS_BY_KIND[kind] || OPS_BY_KIND.categorical).map(([v, l]) => h('option', { value: v, text: l }))); f.operator = opSel.value; val.hidden = f.operator === 'isnull' || f.operator === 'notnull'; };
    colSel.addEventListener('change', () => { f.column = colSel.value; fillOps(); EX().page = 1; loadRows(); });
    opSel.addEventListener('change', () => { f.operator = opSel.value; val.hidden = f.operator === 'isnull' || f.operator === 'notnull'; EX().page = 1; loadRows(); });
    val.addEventListener('input', debounce(() => { f.value = val.value; EX().page = 1; loadRows(); }, 400));
    row.append(colSel, opSel, val, h('button', { class: 'icon-btn', 'aria-label': 'Remove filter', onclick: () => { EX().filters.splice(EX().filters.indexOf(f), 1); row.remove(); EX().page = 1; loadRows(); } }, ico('x')));
    $('#ex-filters').append(row); fillOps(); refreshIcons();
  }
  function wireExplorer() {
    $('#ex-search').addEventListener('input', debounce((e) => { EX().q = e.target.value.trim(); EX().page = 1; loadRows(); }));
    $('#ex-size').addEventListener('change', (e) => { EX().size = Number(e.target.value); EX().page = 1; loadRows(); });
    $('#ex-dups').addEventListener('change', (e) => { EX().dups = e.target.checked; EX().page = 1; loadRows(); });
    $('#ex-filter-btn').addEventListener('click', () => { if (EX().filters.length < 5) addFilterRow(); else toast('You can add up to 5 filters.', 'info'); });
    $('#ex-cols-btn').addEventListener('click', () => {
      const picker = $('#ex-col-picker'); picker.hidden = !picker.hidden; if (picker.hidden) return;
      const visible = new Set(EX().cols || A().columns.map((c) => c.name));
      mount(picker, A().columns.map((c) => h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: visible.has(c.name), onchange: () => {
        const chosen = $$('input', picker).map((i, idx) => i.checked ? A().columns[idx].name : null).filter(Boolean);
        if (!chosen.length) { toast('Keep at least one column visible.', 'info'); $$('input', picker)[0].checked = true; return; }
        EX().cols = chosen.length === A().columns.length ? null : chosen; EX().page = 1; loadRows(); } }), h('span', { text: c.name }))));
    });
  }

  /* ----- Clean ----- */
  function checkList(name, list, hint) {
    return h('div', { class: 'field' }, h('label', { text: hint }), h('div', { class: 'col-picker', 'data-checks': name }, list.length ? list.map((c) => h('label', { class: 'check' }, h('input', { type: 'checkbox', value: c.name }), h('span', { text: c.name }))) : [h('span', { class: 'muted small', text: 'No matching columns.' })]));
  }
  const checked = (name) => $$(`[data-checks="${name}"] input:checked`).map((i) => i.value);
  function renderCleanFields() {
    const op = $('#cl-op').value, a = A(), box = $('#cl-fields'), all = a.columns;
    const withMissing = (k) => all.filter((c) => c.kind === k && c.missing > 0);
    const f = [];
    if (op === 'drop_duplicates') f.push(checkList('cols', all, 'Only compare these columns (leave empty to compare whole rows)'));
    if (op === 'drop_missing') f.push(checkList('cols', all.filter((c) => c.missing > 0), 'Only these columns (leave empty to check every column)'));
    if (op === 'fill_numeric') f.push(selectField('method', 'Fill with', [['mean', 'Average (mean)'], ['median', 'Median'], ['mode', 'Most common value'], ['min', 'Minimum'], ['max', 'Maximum'], ['zero', 'Zero']], undefined, 'clf'), checkList('cols', withMissing('numeric'), 'Columns (leave empty for all numeric columns with gaps)'));
    if (op === 'fill_categorical') f.push(h('div', { class: 'field' }, h('label', { for: 'cl-value', text: 'Fill with this text' }), h('input', { id: 'cl-value', type: 'text', maxlength: 200, placeholder: 'Unknown' })), checkList('cols', withMissing('categorical'), 'Columns (leave empty for all text columns with gaps)'));
    if (op === 'convert_type') f.push(selectField('to', 'Convert to', [['numeric', 'Numbers'], ['datetime', 'Dates']], undefined, 'clf'), checkList('cols', all.filter((c) => c.kind === 'categorical'), 'Text columns to convert. Values that can’t be converted become empty.'));
    if (op === 'rename') f.push(selectField('column', 'Column', all.map((c) => c.name), undefined, 'clf'), h('div', { class: 'field' }, h('label', { for: 'cl-newname', text: 'New name' }), h('input', { id: 'cl-newname', type: 'text', maxlength: 100 })));
    if (op === 'drop_columns') f.push(checkList('cols', all, 'Columns to remove'));
    if (op === 'filter') {
      const colSel = selectField('column', 'Column', all.map((c) => c.name), undefined, 'clf'), opSel = selectField('operator', 'Condition', OPS_BY_KIND.categorical, undefined, 'clf');
      const fillOps = () => { const kind = all.find((c) => c.name === $('select', colSel).value).kind; mount($('select', opSel), (OPS_BY_KIND[kind] || OPS_BY_KIND.categorical).map(([v, l]) => h('option', { value: v, text: l }))); };
      $('select', colSel).addEventListener('change', fillOps);
      f.push(colSel, opSel, h('div', { class: 'field' }, h('label', { for: 'cl-fvalue', text: 'Value' }), h('input', { id: 'cl-fvalue', type: 'text' })));
      queueMicrotask(fillOps);
    }
    mount(box, f); clear($('#cl-preview-box')); $('#cl-preview-box').hidden = true;
  }
  function collectOp() {
    const type = $('#cl-op').value, op = { type };
    if (['drop_duplicates', 'drop_missing', 'drop_columns', 'fill_numeric', 'fill_categorical', 'convert_type'].includes(type)) op.columns = checked('cols');
    if (type === 'fill_numeric') op.method = $('#cl-fields [data-param="method"]').value;
    if (type === 'fill_categorical') op.value = $('#cl-value').value;
    if (type === 'convert_type') op.to = $('#cl-fields [data-param="to"]').value;
    if (type === 'rename') op.mapping = { [$('#cl-fields [data-param="column"]').value]: $('#cl-newname').value.trim() };
    if (type === 'filter') { op.column = $('#cl-fields [data-param="column"]').value; op.operator = $('#cl-fields [data-param="operator"]').value; op.value = $('#cl-fvalue').value; }
    return op;
  }
  function setupClean() { $('#cl-op').value = 'drop_duplicates'; renderCleanFields(); renderSteps(A().transforms); }
  function renderSteps(steps) {
    const list = $('#cl-steps');
    if (!steps.length) { clear(list); mount($('#cl-steps-empty'), emptyState('wand-sparkles', 'No cleaning steps yet', 'You’re looking at your data exactly as uploaded.')); }
    else { clear($('#cl-steps-empty')); mount(list, steps.map((s) => h('li', {}, h('span', { text: s.description })))); }
    $('#cl-undo').disabled = $('#cl-reset').disabled = !steps.length;
  }
  async function previewOp() {
    const btn = $('#cl-preview'); setBusy(btn, true);
    try {
      const d = await post(`/api/datasets/${A().dataset.id}/transform/preview`, collectOp()), p = d.preview;
      const row = (label, a, b) => h('div', { class: 'delta' }, h('span', { text: label }), h('span', {}, fmt(a), h('span', { class: 'arrow', text: '  →  ' }), h('strong', { text: fmt(b) })));
      const box = $('#cl-preview-box'); box.hidden = false;
      mount(box, h('strong', { text: d.description }), row('Rows', p.rows_before, p.rows_after), row('Columns', p.cols_before, p.cols_after), row('Missing cells', p.missing_before, p.missing_after), row('Duplicate rows', p.duplicates_before, p.duplicates_after),
        h('p', { class: 'muted small', text: 'Nothing has been changed yet. Apply the step to keep it. You can undo or reset it later.' }));
    } catch (e) { toast(e.message, 'error'); } finally { setBusy(btn, false); }
  }
  async function transformCall(path, body, okMsg) {
    try { await post(`/api/datasets/${A().dataset.id}/transform/${path}`, body); toast(okMsg, 'success'); await loadAnalysisData(A().dataset.id); showTab('clean'); }
    catch (e) { toast(e.message, 'error'); }
  }
  function wireClean() {
    $('#cl-op').addEventListener('change', renderCleanFields);
    $('#cl-preview').addEventListener('click', previewOp);
    $('#cl-apply').addEventListener('click', () => transformCall('apply', collectOp(), 'Cleaning step applied.'));
    $('#cl-undo').addEventListener('click', () => transformCall('undo', {}, 'Last step undone.'));
    $('#cl-reset').addEventListener('click', async () => { if (await confirmDialog({ title: 'Reset all cleaning steps?', text: 'The dataset returns to exactly what you uploaded. Your original file is never changed.', okLabel: 'Reset', danger: false })) transformCall('reset', {}, 'Dataset reset to the original.'); });
  }

  function wireAnalysis() {
    $('#an-select').addEventListener('change', (e) => { location.hash = `#/analysis/${e.target.value}`; });
    $('#an-tabs').addEventListener('click', (e) => { const t = e.target.closest('.tab'); if (t) showTab(t.dataset.tab); });
    $('#ch-type').addEventListener('change', () => renderChartFields());
    $('#ch-draw').addEventListener('click', drawSelectedChart);
    $('#ch-ask').addEventListener('click', () => {
      if (!state.an.chart) return;
      state.chat.dsId = A().dataset.id; state.chat.chartCtx = { ...state.an.chart, label: $('#ch-title').textContent }; location.hash = '#/chat';
    });
    $('#an-ask').addEventListener('click', () => { state.chat.dsId = A().dataset.id; location.hash = '#/chat'; });
    $('#an-report').addEventListener('click', (e) => downloadFile(`/api/datasets/${A().dataset.id}/download/report`, e.currentTarget));
    $('#an-rename').addEventListener('click', async () => { if (await renameDataset(A().dataset)) { await refreshDatasets(); loadAnalysis(A().dataset.id); } });
    $('#an-delete').addEventListener('click', async () => { if (await deleteDataset(A().dataset)) { state.an.id = null; location.hash = '#/datasets'; } });
    wireExplorer(); wireClean(); wireAi();
  }

  /* ===================== AI full analysis and train-your-own model ===================== */
  const MLS = () => state.an.ml;
  const LEVEL = { good: ['Strong result', 'badge-ok', 'circle-check'], ok: ['Useful, but not perfect', '', 'info'], poor: ['Weak result', 'badge-warn', 'triangle-alert'] };
  const levelBadge = (lvl) => h('span', { class: `badge ${(LEVEL[lvl] || LEVEL.ok)[1]}`, text: ({ good: 'Strong', ok: 'Fair', poor: 'Weak' })[lvl] || 'Fair' });
  const short = (v) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : Math.abs(v) >= 1000 ? nfmt.format(Math.round(v)) : Math.abs(v) >= 100 ? nfmt.format(Math.round(v * 10) / 10) : String(+Number(v).toPrecision(3));
  const headlineText = (m) => (m.headline.value === null || m.headline.value === undefined ? '—' : m.task === 'classification' ? `${Math.round(m.headline.value * 100)}%` : m.headline.value.toFixed(2));

  function resetAiTabs() {
    state.an.air = { loaded: false, report: null, configured: state.ai.configured, running: false };
    state.an.ml = { loaded: false, options: null, models: [], sel: null, features: null, target: null };
    destroyCharts('ml-');
    mount($('#air-body'), ''); $('#air-stale').hidden = true; $('#air-copy').hidden = true; $('#air-meta').textContent = '';
    clear($('#ml-detail')); $('#ml-detail').hidden = true; mount($('#ml-form-box'), ''); mount($('#ml-list'), ''); mount($('#ml-list-empty'), '');
  }

  /* ----- AI report ----- */
  async function loadAiReport() {
    const air = state.an.air, id = A().dataset.id;
    if (air.loaded) return;
    mount($('#air-body'), skeleton(4));
    try {
      const d = await get(`/api/datasets/${id}/ai-report`);
      if (state.an.air !== air) return;
      Object.assign(air, { loaded: true, report: d.report, configured: d.ai_configured });
      renderAiReport();
    } catch (e) { mount($('#air-body'), emptyState('triangle-alert', 'Couldn’t load the AI analysis', e.message)); }
  }
  function renderAiReport() {
    const air = state.an.air, rep = air.report, body = $('#air-body');
    $('#air-run-label').textContent = rep ? 'Run it again' : 'Run full AI analysis';
    $('#air-run').disabled = !air.configured || air.running;
    $('#air-copy').hidden = !rep; $('#air-stale').hidden = !(rep && rep.stale);
    $('#air-meta').textContent = rep ? `Written ${timeAgo(rep.created_at)} from the data as it is now analysed.` : '';
    if (!air.configured) return mount(body, emptyState('sparkles', 'AI isn’t set up on this server', 'The administrator needs to set GROQ_API_KEY. Statistics, charts, cleaning and model training work without it.'));
    if (!rep) return mount(body, emptyState('sparkles', 'Let the AI analyse everything', 'One click: DataLens calculates statistics, groups, correlations, trends and quality problems from every row, and the AI writes a complete report with findings, risks and next steps.'));
    mount(body, h('div', { class: 'ai-report' }, renderMarkdown(rep.content)));
  }
  async function runAiReport() {
    const air = state.an.air, id = A().dataset.id;
    if (air.running) return;
    air.running = true; setBusy($('#air-run'), true);
    mount($('#air-body'), h('div', { class: 'ai-working' }, h('div', { class: 'spinner', 'aria-hidden': 'true' }),
      h('div', {}, h('strong', { text: 'Analysing your dataset…' }),
        h('p', { class: 'muted small', text: 'DataLens is calculating statistics, group comparisons, correlations and trends from every row, then the AI writes it up. This usually takes 10 to 40 seconds.' }))));
    try {
      const d = await post(`/api/datasets/${id}/ai-report`);
      air.report = d.report; air.loaded = true; state.chat.status = null; toast('AI analysis is ready.', 'success');
    } catch (e) { toast(e.message, 'error'); if (e.code === 'ai_not_configured') air.configured = false; }
    finally { air.running = false; setBusy($('#air-run'), false); if (state.an.air === air) renderAiReport(); }
  }

  /* ----- Train your own model ----- */
  async function loadTrainTab() {
    const ml = MLS(), id = A().dataset.id;
    if (ml.loaded) return;
    mount($('#ml-form-box'), skeleton(4));
    try {
      const [opt, mods] = await Promise.all([get(`/api/datasets/${id}/ml/options`), get(`/api/datasets/${id}/models`)]);
      if (state.an.ml !== ml) return;
      Object.assign(ml, { loaded: true, options: opt, models: mods.models });
      renderTrainForm(); renderModelList();
    } catch (e) { mount($('#ml-form-box'), emptyState('triangle-alert', 'Couldn’t load model training', e.message)); }
  }
  function renderTrainForm() {
    const ml = MLS(), o = ml.options, box = $('#ml-form-box');
    if (!o.available) return mount(box, emptyState('cpu', 'Model training isn’t installed', 'The server administrator needs to install scikit-learn (pip install scikit-learn).'));
    if (!o.targets.length) return mount(box, emptyState('target', 'Nothing here can be predicted yet', 'Training needs a column with at least 30 filled rows and some variety: a category with up to 20 values, or a number. Cleaning missing values first can help.'));
    ml.target = ml.target || o.targets[0].column;
    const targetSel = selectField('target', 'What do you want to predict?', o.targets.map((t) => [t.column, t.column]), ml.target, 'ml');
    const algoSel = selectField('algorithm', 'Learning method', o.algorithms.map((a) => [a.id, a.label]), 'forest', 'ml');
    const algoHelp = h('p', { class: 'hint', id: 'ml-algo-help' });
    const showHelp = () => { algoHelp.textContent = (o.algorithms.find((a) => a.id === $('select', algoSel).value) || {}).help || ''; };
    $('select', targetSel).addEventListener('change', loadMlFeatures);
    $('select', algoSel).addEventListener('change', showHelp);
    mount(box, h('div', { class: 'form' }, targetSel, h('p', { class: 'hint', id: 'ml-task-hint' }), algoSel, algoHelp,
      h('div', { class: 'field', id: 'ml-feats' }),
      h('div', { class: 'field' }, h('label', { for: 'ml-name', text: 'Model name (optional)' }), h('input', { id: 'ml-name', type: 'text', maxlength: 120, placeholder: 'e.g. Churn predictor' })),
      h('button', { class: 'btn btn-primary', type: 'button', id: 'ml-train', onclick: trainModel }, ico('cpu'), 'Train my model'),
      h('p', { class: 'muted small', id: 'ml-train-status', role: 'status' }),
      h('p', { class: 'muted small', text: 'The model learns from 80% of your rows and is tested on the 20% it never saw, so the score you get is honest. It learns from your data as currently cleaned, and your data never leaves your account.' })));
    showHelp(); loadMlFeatures();
  }
  async function loadMlFeatures() {
    const ml = MLS(), id = A().dataset.id, target = $('#ml-target').value, box = $('#ml-feats');
    ml.target = target;
    const t = ml.options.targets.find((x) => x.column === target);
    $('#ml-task-hint').textContent = !t ? '' : t.task === 'classification' ? `DataLens will learn to predict which category “${target}” falls into (${t.unique} possible values).` : `DataLens will learn to predict the number in “${target}”.`;
    mount(box, skeleton(1));
    try {
      const d = await get(`/api/datasets/${id}/ml/options?${qs({ target })}`);
      if (state.an.ml !== ml || ml.target !== target) return;
      const u = d.features.usable;
      mount(box, h('label', { text: `Columns the model may learn from (${u.length})` }),
        h('div', { class: 'col-picker', 'data-checks': 'mlfeat' }, u.map((c) => h('label', { class: 'check' }, h('input', { type: 'checkbox', value: c.name, checked: true }), h('span', { text: c.name })))),
        d.features.skipped.length ? h('p', { class: 'muted small', text: 'Not used: ' + d.features.skipped.map((s) => `${s.column} (${s.reason})`).join(', ') + '.' }) : null);
    } catch (e) { mount(box, h('p', { class: 'muted small', text: e.message })); }
  }
  async function trainModel() {
    const ml = MLS(), id = A().dataset.id, btn = $('#ml-train'), features = checked('mlfeat');
    if (!features.length) return toast('Select at least one column for the model to learn from.', 'error');
    setBusy(btn, true); $('#ml-train-status').textContent = 'Training and testing your model… this can take up to a minute for large datasets.';
    try {
      const d = await post(`/api/datasets/${id}/models`, { target: $('#ml-target').value, algorithm: $('#ml-algorithm').value, features, name: $('#ml-name').value.trim() });
      toast('Model trained.', 'success');
      const list = await get(`/api/datasets/${id}/models`);
      if (state.an.ml !== ml) return;
      ml.models = list.models; showModel(d.model);
      $('#ml-detail').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) { toast(e.message, 'error'); }
    finally { setBusy(btn, false); const s = $('#ml-train-status'); if (s) s.textContent = ''; }
  }
  function renderModelList() {
    const ml = MLS(), list = $('#ml-list');
    if (!ml.models.length) { clear(list); return mount($('#ml-list-empty'), emptyState('cpu', 'No models yet', 'Train your first model with the form and it will be saved here.')); }
    clear($('#ml-list-empty'));
    mount(list, ml.models.map((m) => h('li', {}, h('button', { class: `conv-item ${ml.sel && ml.sel.id === m.id ? 'active' : ''}`, type: 'button', onclick: () => openModel(m.id) },
      h('div', { class: 'feed-ico' }, ico('cpu')),
      h('div', { class: 'feed-text' }, h('strong', { text: m.name }), h('span', { text: `${m.task === 'classification' ? 'Category' : 'Number'} · ${m.headline.label} ${headlineText(m)} · ${timeAgo(m.created_at)}` })),
      levelBadge(m.level)))));
  }
  async function openModel(id) {
    const ml = MLS();
    try { const d = await get(`/api/models/${id}`); if (state.an.ml === ml) { showModel(d.model); $('#ml-detail').scrollIntoView({ behavior: 'smooth', block: 'start' }); } }
    catch (e) { toast(e.message, 'error'); }
  }
  function showModel(m) {
    const ml = MLS(); ml.sel = m; renderModelList(); destroyCharts('ml-');
    const mt = m.metrics, cls = m.task === 'classification', pct = (v) => `${Math.round(v * 100)}%`, box = $('#ml-detail');
    const algo = (ml.options.algorithms.find((a) => a.id === m.algorithm) || {}).label || m.algorithm;
    const [lvlTitle, , lvlIcon] = LEVEL[mt.level] || LEVEL.ok;
    const stats = cls
      ? [statCard('target', 'green', 'Accuracy', pct(mt.accuracy), 'on rows it never saw'), statCard('dices', 'amber', 'Guessing baseline', pct(mt.baseline_accuracy), 'always the most common value'),
        statCard('scale', 'blue', 'Balanced accuracy', pct(mt.balanced_accuracy), 'fair across categories'), statCard('crosshair', '', 'F1 score', fmt(mt.f1_macro, 2), 'macro average'),
        statCard('rows-3', '', 'Rows tested', fmt(m.n_test), `${fmt(m.n_train)} used to learn`)]
      : [statCard('target', 'green', 'R² score', mt.r2.toFixed(2), '1 is perfect; 0 is no better than the average'), statCard('ruler', 'blue', 'Typical error', short(mt.mae), 'average miss (MAE)'),
        statCard('activity', '', 'RMSE', short(mt.rmse), 'punishes big misses more'), statCard('dices', 'amber', 'Guessing baseline', short(mt.baseline_mae), 'error when always guessing the average'),
        statCard('rows-3', '', 'Rows tested', fmt(m.n_test), `${fmt(m.n_train)} used to learn`)];

    const importanceCard = h('article', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', { text: 'What the model relied on' })),
      mt.importance && mt.importance.length
        ? [h('p', { class: 'muted small', text: 'How much of the model’s skill comes from each column, measured by shuffling the column and watching the score drop. It shows what the model uses, not what causes the outcome.' }),
          h('div', { class: 'chart-box chart-box-lg' }, h('canvas', { id: 'ml-imp-canvas', role: 'img', 'aria-label': 'Importance of each input column' }))]
        : h('p', { class: 'muted', text: 'No single column stood out, so importance isn’t shown.' }));

    let sideCard;
    if (cls) {
      const cm = mt.confusion, labels = cm.labels;
      sideCard = h('article', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', { text: 'Where it gets things right and wrong' })),
        h('p', { class: 'muted small', text: 'Rows are the real value, columns are what the model predicted. Bright diagonal cells are correct predictions.' }),
        h('div', { class: 'table-wrap' }, h('table', { class: 'table cm-table' },
          h('thead', {}, h('tr', {}, h('th', { text: 'Actual ↓ / Predicted →' }), labels.map((l) => h('th', { class: 'num', title: l, text: l })))),
          h('tbody', {}, cm.matrix.map((row, i) => { const sum = row.reduce((a, b) => a + b, 0) || 1;
            return h('tr', {}, h('th', { title: labels[i], text: labels[i] }), row.map((v, j) => h('td', { class: 'num', style: `background:rgba(${i === j ? '52,211,153' : '248,113,113'},${(0.06 + 0.5 * v / sum).toFixed(2)})`, text: fmt(v) }))); })))));
    } else {
      sideCard = h('article', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', { text: 'How to read the result' })),
        h('div', { class: 'kv' }, h('div', { class: 'kv-row' }, h('span', { text: `Average ${m.target}` }), h('span', { text: short(mt.target_mean) })),
          h('div', { class: 'kv-row' }, h('span', { text: 'Typical spread (std. deviation)' }), h('span', { text: short(mt.target_std) })),
          h('div', { class: 'kv-row' }, h('span', { text: 'Model’s typical error' }), h('span', { text: short(mt.mae) })),
          h('div', { class: 'kv-row' }, h('span', { text: 'Error if you just guessed the average' }), h('span', { text: short(mt.baseline_mae) }))),
        h('p', { class: 'muted small', text: 'A useful model has a typical error clearly below the guessing error.' }));
    }

    const inputs = m.inputs.map((f, i) => {
      const id = `ml-in-${i}`;
      let control;
      if (f.kind === 'categorical') control = h('select', { id, 'data-ml-input': f.name }, h('option', { value: '', text: '(leave empty)' }), f.options.map((o) => h('option', { value: o, text: o })));
      else if (f.kind === 'datetime') control = h('input', { id, type: 'date', 'data-ml-input': f.name, min: (f.min || '').slice(0, 10), max: (f.max || '').slice(0, 10) });
      else control = h('input', { id, type: 'number', step: 'any', 'data-ml-input': f.name, placeholder: `e.g. ${short(f.median)}` });
      return h('div', { class: 'field' }, h('label', { for: id, text: f.name }), control);
    });
    const tryCard = h('article', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', { text: `Try your model: predict ${m.target}` })),
      h('p', { class: 'muted small', text: 'Enter values for a new, made-up row. Leave a field empty if you don’t know it. The model fills the gap with a typical value.' }),
      h('div', { class: 'predict-grid' }, inputs),
      h('div', { class: 'btn-row' }, h('button', { class: 'btn btn-primary', type: 'button', id: 'ml-predict', onclick: () => predictNow(m) }, ico('wand-sparkles'), 'Predict')),
      h('div', { class: 'predict-result', id: 'ml-result', 'aria-live': 'polite' }));

    const explainCard = h('article', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', { text: 'Explain this model in plain language' }),
        h('button', { class: 'btn btn-soft btn-sm', type: 'button', id: 'ml-explain', disabled: !state.ai.configured, title: state.ai.configured ? '' : 'AI isn’t configured on this server', onclick: () => explainModel(m) }, ico('sparkles'), 'Explain with AI')),
      h('div', { class: 'ai-report', id: 'ml-explain-out' }, h('p', { class: 'muted small', text: 'The AI reads the test results above and explains how good the model is, what it relied on, and how to improve it.' })));

    mount(box,
      h('article', { class: 'card' },
        h('div', { class: 'card-head' }, h('div', {}, h('h3', { text: m.name }), h('p', { class: 'muted small', text: `Predicts “${m.target}” (${cls ? 'a category' : 'a number'}) · ${algo} · trained ${timeAgo(m.created_at)}` })),
          h('button', { class: 'btn btn-danger-soft btn-sm', type: 'button', onclick: () => deleteModel(m) }, ico('trash-2'), 'Delete model')),
        h('div', { class: `verdict ${mt.level}` }, ico(lvlIcon), h('div', {}, h('strong', { text: lvlTitle }), h('p', { text: mt.verdict }))),
        mt.notes && mt.notes.length ? h('ul', { class: 'explain' }, mt.notes.map((n) => h('li', { text: n }))) : null,
        h('div', { class: 'grid stats stats-5' }, stats)),
      h('div', { class: 'grid two' }, importanceCard, sideCard), tryCard, explainCard);
    box.hidden = false;
    if (mt.importance && mt.importance.length) {
      drawChart('ml-imp', $('#ml-imp-canvas'), { type: 'bar', data: { labels: mt.importance.map((r) => r.feature), datasets: [{ label: 'Share of importance', data: mt.importance.map((r) => +(r.share * 100).toFixed(1)), backgroundColor: COLORS[0], borderRadius: 6 }] },
        options: { indexAxis: 'y', plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${c.parsed.x}%` } } }, scales: { x: { ticks: { callback: (v) => `${v}%` } }, y: { grid: { display: false } } } } });
    }
  }
  async function predictNow(m) {
    const values = {};
    $$('#ml-detail [data-ml-input]').forEach((el) => { if (el.value !== '') values[el.dataset.mlInput] = el.value; });
    if (!Object.keys(values).length) return toast('Fill in at least one value first.', 'error');
    const btn = $('#ml-predict'); setBusy(btn, true);
    try {
      const r = (await post(`/api/models/${m.id}/predict`, { values })).result, box = $('#ml-result');
      if (m.task === 'classification') {
        mount(box, h('p', { class: 'muted small', text: `Predicted ${m.target}` }), h('p', { class: 'predict-value', text: r.prediction }),
          r.probabilities.map((p) => h('div', { class: 'prob-row' }, h('span', { class: 'truncate', title: p.value, text: p.value }),
            h('div', { class: 'mini-bar' }, h('span', { style: `width:${(p.probability * 100).toFixed(1)}%` })), h('span', { class: 'small', text: `${Math.round(p.probability * 100)}%` }))),
          h('p', { class: 'muted small', text: 'The percentages show how confident the model is. They are estimates, not guarantees.' }));
      } else {
        mount(box, h('p', { class: 'muted small', text: `Predicted ${m.target}` }), h('p', { class: 'predict-value', text: `≈ ${short(r.prediction)}` }),
          h('p', { class: 'muted small', text: `On rows it had never seen, the model was typically off by about ${short(r.typical_error)}.` }));
      }
    } catch (e) { toast(e.message, 'error'); } finally { setBusy(btn, false); }
  }
  async function explainModel(m) {
    const btn = $('#ml-explain'), out = $('#ml-explain-out'); setBusy(btn, true); mount(out, skeleton(3));
    try { const d = await post(`/api/models/${m.id}/explain`); mount(out, renderMarkdown(d.explanation)); state.chat.status = null; }
    catch (e) { toast(e.message, 'error'); mount(out, h('p', { class: 'muted small', text: e.message })); }
    finally { setBusy(btn, false); }
  }
  async function deleteModel(m) {
    if (!await confirmDialog({ title: `Delete “${m.name}”?`, text: 'The trained model is removed permanently. Your dataset is not affected.', okLabel: 'Delete model' })) return;
    const ml = MLS();
    try {
      await del(`/api/models/${m.id}`); toast('Model deleted.', 'success');
      ml.sel = null; destroyCharts('ml-'); clear($('#ml-detail')); $('#ml-detail').hidden = true;
      ml.models = ml.models.filter((x) => x.id !== m.id); renderModelList();
    } catch (e) { toast(e.message, 'error'); }
  }
  function wireAi() {
    $('#air-run').addEventListener('click', runAiReport);
    $('#air-copy').addEventListener('click', async () => {
      const r = state.an.air && state.an.air.report; if (!r) return;
      try { await navigator.clipboard.writeText(r.content); toast('Report copied.', 'success'); } catch (_) { toast('Couldn’t copy automatically. Select the text and copy it.', 'error'); }
    });
    $('#air-chat').addEventListener('click', () => { state.chat.dsId = A().dataset.id; location.hash = '#/chat'; });
  }

  /* ===================== Safe Markdown (DOM nodes only, no innerHTML) ===================== */
  function inlineMd(text, parent) {
    const parts = text.split(/(`[^`\n]+`|\*\*[^*\n]+\*\*|\*[^*\s][^*\n]*\*)/g);
    for (const part of parts) {
      if (!part) continue;
      if (part.startsWith('`') && part.endsWith('`') && part.length > 2) parent.append(h('code', { text: part.slice(1, -1) }));
      else if (part.startsWith('**') && part.endsWith('**') && part.length > 4) parent.append(h('strong', { text: part.slice(2, -2) }));
      else if (part.startsWith('*') && part.endsWith('*') && part.length > 2) parent.append(h('em', { text: part.slice(1, -1) }));
      else parent.append(document.createTextNode(part));
    }
  }
  function renderMarkdown(src) {
    const frag = document.createDocumentFragment();
    const lines = String(src).replace(/\r/g, '').split('\n');
    const isTableSep = (l) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$/.test(l);
    const cells = (l) => l.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
    let i = 0, para = [];
    const flush = () => { if (!para.length) return; const p = h('p'); para.forEach((t, k) => { if (k) p.append(h('br')); inlineMd(t, p); }); frag.append(p); para = []; };
    while (i < lines.length) {
      const line = lines[i];
      if (/^```/.test(line)) { flush(); const code = []; i++; while (i < lines.length && !/^```/.test(lines[i])) code.push(lines[i++]); i++; frag.append(h('pre', {}, h('code', { text: code.join('\n') }))); continue; }
      if (/^#{1,6}\s+/.test(line)) { flush(); const hd = h('h4'); inlineMd(line.replace(/^#{1,6}\s+/, ''), hd); frag.append(hd); i++; continue; }
      if (line.includes('|') && i + 1 < lines.length && isTableSep(lines[i + 1])) {
        flush(); const head = cells(line); i += 2; const rows = [];
        while (i < lines.length && lines[i].includes('|') && lines[i].trim()) rows.push(cells(lines[i++]));
        const t = h('table', {}, h('thead', {}, h('tr', {}, head.map((c) => { const th = h('th'); inlineMd(c, th); return th; }))),
          h('tbody', {}, rows.map((r) => h('tr', {}, r.map((c) => { const td = h('td'); inlineMd(c, td); return td; })))));
        frag.append(t); continue;
      }
      const ul = /^\s*[-*•]\s+(.*)/.exec(line), ol = /^\s*\d+[.)]\s+(.*)/.exec(line);
      if (ul || ol) {
        flush(); const list = h(ul ? 'ul' : 'ol'); const rx = ul ? /^\s*[-*•]\s+(.*)/ : /^\s*\d+[.)]\s+(.*)/;
        while (i < lines.length && rx.test(lines[i])) { const li = h('li'); inlineMd(rx.exec(lines[i])[1], li); list.append(li); i++; }
        frag.append(list); continue;
      }
      if (!line.trim()) { flush(); i++; continue; }
      para.push(line); i++;
    }
    flush();
    return frag;
  }

  /* ===================== AI chat ===================== */
  const STARTERS = ['What does this dataset contain?', 'Explain the summary statistics in simple terms', 'Which columns have the most missing values?', 'Are there unusual values I should investigate?',
    'What are the most important patterns?', 'Suggest cleaning steps for this dataset', 'What questions should I investigate next?', 'Summarize the dataset for a business report'];
  const NO_RETRY = new Set(['ai_not_configured', 'quota_exceeded', 'dataset_missing', 'too_long', 'validation', 'csrf', 'auth_error', 'model_unavailable', 'permission_denied']);
  const C = () => state.chat;
  const dsName = (id) => (state.datasets.find((d) => d.id === id) || {}).name || 'this dataset';

  async function loadChat() {
    const c = C(); await refreshDatasets();
    try { c.status = await get('/api/ai/status'); state.ai.configured = c.status.configured; } catch (_) { /* keep previous */ }
    const banner = $('#chat-banner');
    banner.hidden = state.ai.configured;
    if (!state.ai.configured) banner.textContent = 'AI chat needs to be configured on the server (set GROQ_API_KEY). Statistics, charts, cleaning and reports work without it.';
    if (!state.datasets.length) {
      $('#chat-ds').disabled = true; $('#chat-new').disabled = true; mount($('#chat-convs'), ''); mount($('#chat-suggest'), '');
      mount($('#chat-messages'), emptyState('database', 'Upload a dataset to start chatting', 'The assistant answers questions about one of your datasets.', uploadCta()));
      setComposerEnabled(false); $('#chat-ds-badge').textContent = ''; return;
    }
    $('#chat-ds').disabled = false; $('#chat-new').disabled = false;
    if (!state.datasets.some((d) => d.id === c.dsId)) c.dsId = state.datasets[0].id;
    datasetOptions($('#chat-ds'), c.dsId); updateAttach();
    await loadConvs();
    if (c.openConv) { const id = c.openConv; c.openConv = null; await openConversation(id); }
    else if (!c.conv || c.conv.dataset_id !== c.dsId) startNew(false);
    else renderChatHeader();
    updateFoot(); setComposerEnabled(true);
  }
  function setComposerEnabled(on) {
    const ok = on && state.ai.configured;
    $('#chat-input').disabled = !ok; $('#chat-send').disabled = !ok;
    $('#chat-input').placeholder = ok ? 'Ask about this dataset…' : (on ? 'AI chat isn’t configured on this server' : 'Upload a dataset first');
  }
  function updateFoot() {
    const s = C().status, f = $('#chat-foot');
    f.textContent = s && state.ai.configured ? `${s.remaining} of ${s.daily_limit} AI questions left today. Answers use calculations from your data; AI can make mistakes, so verify important figures.` : 'Answers use calculations from your data. AI can make mistakes; verify important figures.';
  }
  async function loadConvs() {
    const c = C();
    try { c.convs = (await get(`/api/conversations?${qs({ dataset: c.dsId })}`)).conversations; } catch (_) { c.convs = []; }
    renderConvList();
  }
  function renderConvList() {
    const c = C(), list = $('#chat-convs');
    if (!c.convs.length) { clear(list); mount($('#chat-convs-empty'), h('p', { class: 'muted small', text: 'No conversations about this dataset yet.' })); return; }
    clear($('#chat-convs-empty'));
    mount(list, c.convs.map((cv) => h('li', {}, h('button', { class: `conv-item ${c.conv && c.conv.id === cv.id ? 'active' : ''}`, onclick: () => openConversation(cv.id) },
      h('div', { class: 'feed-ico' }, ico('message-square')), h('div', { class: 'feed-text' }, h('strong', { text: cv.title }), h('span', { text: timeAgo(cv.updated_at) }))))));
  }
  function renderChatHeader() {
    const c = C();
    $('#chat-title').textContent = c.conv ? c.conv.title : 'New conversation';
    $('#chat-ds-badge').textContent = `Active dataset: ${dsName(c.dsId)}`;
    $('#chat-rename').hidden = $('#chat-delete').hidden = !c.conv;
  }
  function welcome() {
    const box = h('div', { class: 'empty-state', id: 'chat-welcome' }, h('div', { class: 'es-icon' }, ico('sparkles')), h('h4', { text: `Ask anything about “${dsName(C().dsId)}”` }),
      h('p', { text: 'I calculate figures from your actual data first, then explain them in plain language.' }));
    return box;
  }
  function renderStarters(show) {
    mount($('#chat-suggest'), show ? STARTERS.map((q) => h('button', { class: 'suggest-btn', type: 'button', onclick: () => sendMessage(q) }, q)) : '');
  }
  function startNew(showMain = true) {
    const c = C(); c.conv = null; renderChatHeader(); renderConvList();
    mount($('#chat-messages'), welcome()); renderStarters(true);
    $('#chat-layout').classList.toggle('show-main', showMain);
    if (showMain) $('#chat-input').focus();
  }
  async function openConversation(id) {
    try {
      const d = await get(`/api/conversations/${id}`), c = C();
      c.conv = d.conversation; c.dsId = d.conversation.dataset_id || c.dsId;
      datasetOptions($('#chat-ds'), c.dsId); renderChatHeader(); renderConvList();
      const box = clear($('#chat-messages'));
      if (!d.messages.length) box.append(welcome()); else d.messages.forEach((m) => appendMessage(m));
      renderStarters(!d.messages.length); scrollChat(); refreshIcons();
      $('#chat-layout').classList.add('show-main');
    } catch (e) { toast(e.message, 'error'); startNew(false); }
  }
  const scrollChat = () => { const m = $('#chat-messages'); m.scrollTop = m.scrollHeight; };
  function appendMessage(m) {
    const isUser = m.role === 'user';
    const body = h('div', { class: `msg-body ${m.status === 'failed' ? 'failed' : ''}` });
    if (isUser) body.textContent = m.content; else body.append(renderMarkdown(m.content));
    const col = h('div', { class: 'msg-col' }, body, h('div', { class: 'msg-meta', text: m.created_at ? timeAgo(m.created_at) : '' }));
    const root = h('div', { class: `msg ${isUser ? 'msg-user' : 'msg-ai'}` }, h('div', { class: 'msg-avatar', 'aria-hidden': 'true' }, isUser ? (state.user.full_name[0] || 'U') : ico('sparkles')), col);
    $('#chat-messages').append(root);
    if (isUser && m.status === 'failed') showFailure(root, { message: 'This question wasn’t answered.', code: 'unknown' }, m.content);
    refreshIcons(); scrollChat();
    return root;
  }
  function showFailure(root, err, text) {
    const col = $('.msg-col', root); $('.msg-error', col)?.remove();
    const row = h('div', { class: 'msg-error', role: 'alert' }, ico('triangle-alert'), h('span', { text: err.message }));
    if (!NO_RETRY.has(err.code)) row.append(h('button', { class: 'btn btn-soft btn-sm', type: 'button', onclick: () => { row.remove(); sendMessage(text, true, root); } }, ico('refresh-cw'), 'Retry'));
    col.append(row); $('.msg-body', root).classList.add('failed'); refreshIcons(); scrollChat();
  }
  function typingNode() {
    const n = h('div', { class: 'msg msg-ai', id: 'chat-typing', 'aria-label': 'The assistant is working' }, h('div', { class: 'msg-avatar' }, ico('sparkles')),
      h('div', { class: 'msg-col' }, h('div', { class: 'msg-body' }, h('span', { class: 'typing' }, h('span'), h('span'), h('span')))));
    $('#chat-messages').append(n); refreshIcons(); scrollChat(); return n;
  }
  async function sendMessage(text, retry = false, existing = null) {
    const c = C(); text = (text || '').trim();
    if (!text || c.sending || !state.ai.configured) return;
    c.sending = true; $('#chat-send').disabled = true;
    let userNode = existing, typing = null;
    try {
      if (!c.conv) { const d = await post('/api/conversations', { dataset_id: c.dsId }); c.conv = d.conversation; renderChatHeader(); }
      $('#chat-welcome')?.remove(); renderStarters(false);
      if (!userNode) userNode = appendMessage({ role: 'user', content: text, status: 'pending' });
      $('.msg-body', userNode).classList.remove('failed');
      typing = typingNode();
      const body = { content: text, retry };
      if (c.chartCtx) body.context = { chart: { type: c.chartCtx.type, params: c.chartCtx.params } };
      const d = await post(`/api/conversations/${c.conv.id}/messages`, body);
      typing.remove(); appendMessage(d.message);
      c.conv = d.conversation; c.status = { ...(c.status || {}), remaining: d.remaining, daily_limit: (c.status || {}).daily_limit }; renderChatHeader(); updateFoot();
      loadConvs();
    } catch (e) {
      if (typing) typing.remove();
      if (e.code === 'ai_not_configured') { state.ai.configured = false; $('#chat-banner').hidden = false; $('#chat-banner').textContent = e.message; setComposerEnabled(true); }
      if (userNode) showFailure(userNode, e, text); else toast(e.message, 'error');
      if (e.code === 'quota_exceeded') toast(e.message, 'error');
    } finally { c.sending = false; $('#chat-send').disabled = !state.ai.configured; }
  }
  function updateAttach() {
    const a = C().chartCtx; $('#chat-attach').hidden = !a;
    if (a) $('#chat-attach-text').textContent = `Chart attached: ${a.label || a.type}. The assistant will see its data.`;
    refreshIcons();
  }
  function wireChat() {
    const input = $('#chat-input');
    $('#chat-form').addEventListener('submit', (e) => { e.preventDefault(); const t = input.value; if (!t.trim()) return; input.value = ''; autosize(); sendMessage(t); });
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); $('#chat-form').requestSubmit(); } });
    const autosize = () => { input.style.height = 'auto'; input.style.height = `${Math.min(input.scrollHeight, 160)}px`; };
    input.addEventListener('input', autosize);
    $('#chat-ds').addEventListener('change', async (e) => { C().dsId = e.target.value; C().conv = null; await loadConvs(); startNew(false); });
    $('#chat-new').addEventListener('click', () => startNew(true));
    $('#chat-back').addEventListener('click', () => $('#chat-layout').classList.remove('show-main'));
    $('#chat-attach-x').addEventListener('click', () => { C().chartCtx = null; updateAttach(); });
    $('#chat-rename').addEventListener('click', async () => {
      const t = await promptDialog({ title: 'Rename conversation', label: 'Title', value: C().conv.title, validate: (v) => v.trim() ? null : 'Enter a title.' });
      if (t === null) return;
      try { const d = await patch(`/api/conversations/${C().conv.id}`, { title: t.trim() }); C().conv = d.conversation; renderChatHeader(); loadConvs(); } catch (e) { toast(e.message, 'error'); }
    });
    $('#chat-delete').addEventListener('click', async () => {
      if (!await confirmDialog({ title: 'Delete this conversation?', text: 'The messages are permanently removed.', okLabel: 'Delete' })) return;
      try { await del(`/api/conversations/${C().conv.id}`); toast('Conversation deleted.', 'success'); C().conv = null; await loadConvs(); startNew(false); } catch (e) { toast(e.message, 'error'); }
    });
    $('#chat-clear').addEventListener('click', clearAllChats);
  }
  async function clearAllChats() {
    if (!await confirmDialog({ title: 'Delete all chat history?', text: 'Every conversation with the AI assistant is permanently removed.', okLabel: 'Delete everything' })) return false;
    try { const d = await del('/api/conversations'); toast(`Deleted ${d.deleted} conversation${d.deleted === 1 ? '' : 's'}.`, 'success'); C().conv = null; C().convs = []; if (currentRoute === 'chat') { renderConvList(); startNew(false); } return true; }
    catch (e) { toast(e.message, 'error'); return false; }
  }

  /* ===================== Reports ===================== */
  async function loadReports() {
    await refreshDatasets();
    const table = $('#rep-table'), empty = $('#rep-empty');
    if (!state.datasets.length) { clear(table); return mount(empty, emptyState('file-text', 'No reports yet', 'Upload a dataset to generate downloads and a PDF report.', uploadCta())); }
    clear(empty);
    const dlBtn = (ds, kind, label) => h('button', { class: 'btn btn-soft btn-sm', onclick: (e) => downloadFile(`/api/datasets/${ds.id}/download/${kind}`, e.currentTarget) }, label);
    mount(table, h('thead', {}, h('tr', {}, ['Dataset', 'Rows', 'Uploaded', 'Downloads'].map((x, i) => h('th', { class: i === 1 ? 'num' : '', text: x })))),
      h('tbody', {}, state.datasets.map((ds) => h('tr', {}, h('td', { class: 'truncate' }, h('a', { class: 'link', href: `#/analysis/${ds.id}`, text: ds.name })), h('td', { class: 'num', text: fmt(ds.rows) }), h('td', { text: fmtDate(ds.created_at) }),
        h('td', {}, h('div', { class: 'btn-row' }, dlBtn(ds, 'original', 'Original CSV'), dlBtn(ds, 'cleaned', ds.has_transforms ? 'Cleaned CSV' : 'Analysed CSV'), dlBtn(ds, 'summary', 'Statistics CSV'), dlBtn(ds, 'missing', 'Missing values CSV'), dlBtn(ds, 'report', 'PDF report')))))));
  }

  /* ===================== Settings ===================== */
  async function loadSettings() {
    const u = state.user;
    $('#pf-name').value = u.full_name; $('#pf-username').value = u.username; $('#pf-email').value = u.email;
    const box = $('#set-ai');
    try {
      const s = await get('/api/ai/status'); state.ai.configured = s.configured;
      mount(box, h('div', { class: 'kv' },
        h('div', { class: 'kv-row' }, h('span', { text: 'Status' }), h('span', {}, h('span', { class: `badge ${s.configured ? 'badge-ok' : 'badge-warn'}`, text: s.configured ? 'Configured' : 'Not configured' }))),
        s.configured ? h('div', { class: 'kv-row' }, h('span', { text: 'Model' }), h('span', { text: s.model })) : null,
        h('div', { class: 'kv-row' }, h('span', { text: 'Today’s questions' }), h('span', { text: `${s.used_today} of ${s.daily_limit}` }))),
        h('p', { class: 'muted small', text: s.configured ? 'Limits reset at midnight UTC. Availability and rate limits depend on the server’s Groq account.' : 'The server administrator must set GROQ_API_KEY to enable AI chat. Everything else works without it.' }));
    } catch (e) { mount(box, h('p', { class: 'muted', text: e.message })); }
  }
  function wireSettings() {
    $('#profile-form').addEventListener('submit', async (e) => {
      e.preventDefault(); const f = e.currentTarget, btn = $('button[type=submit]', f); setBusy(btn, true); showFormError(f, null);
      try { const d = await patch('/api/profile', { full_name: $('#pf-name').value, email: $('#pf-email').value }); state.user = d.user; enterAppHeader(); toast('Profile updated.', 'success'); }
      catch (err) { showFormError(f, err); } finally { setBusy(btn, false); }
    });
    $('#password-form').addEventListener('submit', async (e) => {
      e.preventDefault(); const f = e.currentTarget, btn = $('button[type=submit]', f); setBusy(btn, true); showFormError(f, null);
      try { await post('/api/auth/password', Object.fromEntries(new FormData(f))); f.reset(); toast('Password updated.', 'success'); }
      catch (err) { showFormError(f, err); } finally { setBusy(btn, false); }
    });
    $('#set-clear-chats').addEventListener('click', clearAllChats);
    $('#set-delete-account').addEventListener('click', async () => {
      if (state.user.is_admin) return toast('Administrator accounts can’t be deleted here.', 'error');
      const pw = await promptDialog({ title: 'Delete your account?', text: 'This permanently deletes your datasets, conversations and profile. Enter your password to confirm.', label: 'Password', type: 'password', okLabel: 'Delete account', validate: (v) => v ? null : 'Enter your password.' });
      if (pw === null) return;
      try { await del('/api/profile', { password: pw }); toast('Your account was deleted.', 'success'); resetAfterLogout(); } catch (e) { toast(e.message, 'error'); }
    });
  }
  function enterAppHeader() {
    const u = state.user; $('#user-avatar').textContent = u.full_name.trim()[0] || '?'; $('#user-btn-name').textContent = u.full_name.split(' ')[0];
    $('#um-name').textContent = u.full_name; $('#um-email').textContent = u.email;
  }

  /* ===================== Admin ===================== */
  const pretty = (s) => s.replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase());
  async function loadAdminAnalytics() {
    const d = await get(`/api/admin/metrics?days=${$('#ad-range').value}`), u = d.users, ds = d.datasets, ai = d.ai, dl = d.downloads;
    $('#ad-updated').textContent = `Real usage from the database · updated ${new Date(d.generated_at).toLocaleTimeString()}`;
    mount($('#ad-stats'),
      statCard('users', '', 'Registered users', fmt(u.total), `${fmt(u.active_accounts)} active accounts`), statCard('activity', 'blue', 'Daily active users', fmt(u.dau), 'last 24 hours'),
      statCard('activity', 'blue', 'Weekly active users', fmt(u.wau), 'last 7 days'), statCard('activity', 'blue', 'Monthly active users', fmt(u.mau), 'last 30 days'),
      statCard('database', 'green', 'Datasets stored', fmt(ds.total), `${fmt(ds.uploads_in_period)} uploaded in period`), statCard('rows-3', 'green', 'Rows processed', fmt(ds.rows_processed), 'at upload, all time'),
      statCard('chart-no-axes-column', 'amber', 'Analyses completed', fmt(ds.analyses_total), 'all time'), statCard('sparkles', '', 'AI requests', fmt(ai.attempted), `${fmt(ai.succeeded)} ok · ${fmt(ai.failed)} failed`),
      statCard('triangle-alert', 'red', 'AI rate-limit errors', fmt(ai.rate_limit_errors), `${fmt(ai.blocked)} blocked by quota`), statCard('coins', 'amber', 'AI tokens', fmt(ai.tokens), `from ${fmt(ai.requests_with_usage)} requests with usage data`),
      statCard('download', 'green', 'CSV downloads', fmt(dl.csv), 'in period'), statCard('file-text', 'green', 'Report downloads', fmt(dl.reports), 'in period'));
    const labels = d.labels.map(fmtDay), line = (label, data, color) => ({ label, data, borderColor: color, backgroundColor: color + '33', tension: 0.3, pointRadius: 2, fill: false });
    const scales = { y: { beginAtZero: true, ticks: { precision: 0 } }, x: { grid: { display: false }, ticks: { maxTicksLimit: 10 } } };
    drawChart('ad-signups', $('#ad-signups'), { type: 'bar', data: { labels, datasets: [{ label: 'New users', data: u.signups, backgroundColor: '#7c5cffcc', borderRadius: 4 }] }, options: { scales, plugins: { legend: { display: false } } } });
    drawChart('ad-uploads', $('#ad-uploads'), { type: 'line', data: { labels, datasets: [line('Uploads', ds.uploads, '#7c5cff'), line('Analyses', ds.analyses, '#38bdf8')] }, options: { scales, plugins: { legend: { position: 'bottom' } } } });
    drawChart('ad-ai', $('#ad-ai'), { type: 'bar', data: { labels, datasets: [{ label: 'Successful', data: ai.success_series, backgroundColor: '#34d399cc' }, { label: 'Failed', data: ai.failed_series, backgroundColor: '#f87171cc' }] }, options: { scales: { ...scales, x: { ...scales.x, stacked: true }, y: { ...scales.y, stacked: true } }, plugins: { legend: { position: 'bottom' } } } });
    drawChart('ad-dl', $('#ad-dl'), { type: 'line', data: { labels, datasets: [line('CSV downloads', dl.csv_series, '#fbbf24'), line('Report downloads', dl.report_series, '#f472b6')] }, options: { scales, plugins: { legend: { position: 'bottom' } } } });
    const hbar = (items, color) => ({ type: 'bar', data: { labels: items.map((x) => pretty(x.label || x.feature)), datasets: [{ data: items.map((x) => x.count), backgroundColor: color, borderRadius: 4 }] }, options: { indexAxis: 'y', scales: { x: { beginAtZero: true, ticks: { precision: 0 } }, y: { grid: { display: false } } }, plugins: { legend: { display: false } } } });
    drawChart('ad-features', $('#ad-features'), hbar(ds.feature_usage, '#7c5cffcc'));
    if (ai.features.length) { clear($('#ad-topics-empty')); drawChart('ad-topics', $('#ad-topics'), hbar(ai.features, '#38bdf8cc')); }
    else { destroyCharts('ad-topics'); mount($('#ad-topics-empty'), h('p', { class: 'muted small', text: 'No AI chat requests in this period.' })); }
    const tu = $('#ad-topusers');
    if (ai.top_users.length) { clear($('#ad-topusers-empty')); mount(tu, h('thead', {}, h('tr', {}, h('th', { text: 'User' }), h('th', { class: 'num', text: 'AI requests' }))), h('tbody', {}, ai.top_users.map((x) => h('tr', {}, h('td', { text: x.username }), h('td', { class: 'num', text: fmt(x.requests) }))))); }
    else { clear(tu); mount($('#ad-topusers-empty'), h('p', { class: 'muted small', text: 'No AI usage in this period.' })); }
    mount($('#ad-recent'), h('thead', {}, h('tr', {}, ['User', 'Joined', 'Last active'].map((x) => h('th', { text: x })))), h('tbody', {}, u.recent.map((x) => h('tr', {}, h('td', { text: x.username }), h('td', { text: fmtDate(x.created_at) }), h('td', { text: timeAgo(x.last_active_at) })))));
    mount($('#ad-defs'), [d.definitions.active_users, d.definitions.rows_processed, d.definitions.tokens].map((t) => h('li', { text: t })));
  }
  async function loadAdminUsers() {
    const q = $('#au-search').value.trim(), d = await get(`/api/admin/users?${qs({ q })}`);
    $('#au-count').textContent = `${d.total} registered user${d.total === 1 ? '' : 's'}`;
    mount($('#au-table'), h('thead', {}, h('tr', {}, ['User', 'Email', 'Role', 'Status', 'Datasets', 'AI (30d)', 'Joined', 'Last active', ''].map((x) => h('th', { text: x })))),
      h('tbody', {}, d.users.map((u) => h('tr', {}, h('td', {}, h('strong', { text: u.username }), h('div', { class: 'muted small', text: u.full_name })), h('td', { text: u.email }), h('td', {}, h('span', { class: `badge ${u.is_admin ? 'badge-purple' : ''}`, text: u.is_admin ? 'Admin' : 'User' })),
        h('td', {}, h('span', { class: `badge ${u.is_active ? 'badge-ok' : 'badge-bad'}`, text: u.is_active ? 'Active' : 'Disabled' })), h('td', { class: 'num', text: fmt(u.datasets) }), h('td', { class: 'num', text: fmt(u.ai_requests_30d) }),
        h('td', { text: fmtDate(u.created_at) }), h('td', { text: timeAgo(u.last_active_at) }),
        h('td', {}, u.id === state.user.id ? h('span', { class: 'muted small', text: 'You' }) : h('button', { class: `btn btn-sm ${u.is_active ? 'btn-danger-soft' : 'btn-soft'}`, onclick: async () => {
          if (u.is_active && !await confirmDialog({ title: `Disable ${u.username}?`, text: 'They will be signed out and unable to log in until re-enabled. Their data is kept.', okLabel: 'Disable' })) return;
          try { await patch(`/api/admin/users/${u.id}`, { is_active: !u.is_active }); toast(`${u.username} ${u.is_active ? 'disabled' : 'enabled'}.`, 'success'); loadAdminUsers(); } catch (e) { toast(e.message, 'error'); } } }, u.is_active ? 'Disable' : 'Enable'))))));
  }
  async function loadAdminLogs() {
    const d = await get(`/api/admin/logs?${qs({ failed: $('#al-filter').value === 'failed' ? 1 : '', limit: 100 })}`);
    const tone = (t) => /failed|blocked/.test(t) ? 'badge-bad' : /success|upload|download/.test(t) ? 'badge-ok' : '';
    mount($('#al-table'), h('thead', {}, h('tr', {}, ['Time', 'Event', 'User', 'Details'].map((x) => h('th', { text: x })))),
      h('tbody', {}, d.events.length ? d.events.map((e) => h('tr', {}, h('td', { text: new Date(e.at).toLocaleString() }), h('td', {}, h('span', { class: `badge ${tone(e.type)}`, text: e.type })), h('td', { text: e.username || '—' }),
        h('td', { class: 'truncate', text: [...Object.entries(e.meta).map(([k, v]) => `${k}: ${v}`), e.tokens ? `tokens: ${e.tokens}` : '', e.rows ? `rows: ${e.rows}` : ''].filter(Boolean).join(' · ') || '—' }))) : [h('tr', {}, h('td', { colspan: 4 }, emptyState('terminal', 'No events', 'Nothing matches this filter.')))]));
  }
  function wireAdmin() {
    $('#ad-range').addEventListener('change', () => loadAdminAnalytics().catch((e) => toast(e.message, 'error')));
    $('#au-search').addEventListener('input', debounce(() => loadAdminUsers().catch((e) => toast(e.message, 'error'))));
    $('#al-filter').addEventListener('change', () => loadAdminLogs().catch((e) => toast(e.message, 'error')));
  }

  /* ===================== Start-up ===================== */
  function wireMisc() {
    $('#datasets-search').addEventListener('input', debounce(() => { if (currentRoute === 'datasets') loadDatasets().catch((e) => toast(e.message, 'error')); }));
    $('#datasets-sort').addEventListener('change', () => loadDatasets().catch((e) => toast(e.message, 'error')));
    $('#dash-range').addEventListener('change', () => loadDashboard().catch((e) => toast(e.message, 'error')));
    $('#dash-ds-select').addEventListener('change', (e) => { state.dashDs = e.target.value; renderDashQuality(); });
  }

  /* ===================== My AIs (Build Your AI) ===================== */
  const AS = { opts: null, cur: null, conv: null, convs: [], sending: false, step: 0, tab: 'chat' };
  const asRoot = () => $('#as-root');
  const statusBadge = (a) => h('span', { class: `badge ${a.status === 'published' ? 'badge-ok' : 'badge-purple'}`, text: a.status === 'published' ? 'Published' : 'Draft' });
  async function loadAssistants(arg) {
    const [opts] = await Promise.all([get('/api/assistants/options'), refreshDatasets()]);
    AS.opts = opts; mount(asRoot(), skeleton(3));
    if (arg === 'new') return renderAsWizard(null);
    if (arg) { const d = await get(`/api/assistants/${arg}`); AS.cur = d.assistant; AS.conv = null; AS.tab = 'chat'; return renderAsWorkspace(); }
    const d = await get('/api/assistants'); renderAsList(d.assistants);
  }
  function renderAsList(items) {
    const cards = items.map((a) => h('article', { class: 'card dataset-card' },
      h('div', { class: 'ds-top' }, h('h3', { text: a.name }), statusBadge(a)),
      h('p', { class: 'muted small', text: a.description || 'No description.' }),
      h('p', { class: 'small muted', text: `${a.dataset ? a.dataset.name : 'dataset missing'} · ${a.dataset_version.label}${a.update_available ? ' · newer data available' : ''}` }),
      h('div', { class: 'btn-row' }, h('a', { class: 'btn btn-primary btn-sm', href: `#/assistants/${a.id}` }, 'Open'), h('a', { class: 'btn btn-soft btn-sm', href: `#/api/${a.id}` }, 'API'))));
    mount(asRoot(), h('div', { class: 'page-head' }, h('div', {}, h('h2', { text: 'My AIs' }),
        h('p', { class: 'muted', text: 'Dataset-aware assistants you configure, test and publish. They use an existing AI model plus safe calculations on your data. They are not newly trained models.' })),
      h('div', { class: 'head-actions' }, h('a', { class: 'btn btn-primary', href: '#/assistants/new' }, ico('plus'), 'Create assistant'))),
      items.length ? h('div', { class: 'dataset-cards' }, cards)
        : emptyState('bot', 'No assistants yet', 'Pick a dataset, describe what the assistant is for, test it, then publish it through an API.',
          state.datasets.length ? h('a', { class: 'btn btn-primary', href: '#/assistants/new' }, 'Create your first assistant') : uploadCta()));
  }
  /* The same form is used as a 3-step wizard (new) and as one page (edit). */
  function asForm(a, edit) {
    const o = AS.opts, v = a || {};
    const f = (label, ctl, hint) => h('div', { class: 'field' }, h('label', { text: label }), ctl, hint ? h('p', { class: 'muted small', text: hint }) : null);
    const name = h('input', { id: 'af-name', maxlength: 80, value: v.name || '', placeholder: 'e.g. Sales helper' });
    const dsSel = h('select', { id: 'af-ds', class: 'select' }, state.datasets.map((d) => h('option', { value: d.id, text: d.name })));
    const desc = h('input', { id: 'af-desc', maxlength: 300, value: v.description || '', placeholder: 'What is this assistant for?' });
    const welcome = h('input', { id: 'af-welcome', maxlength: 300, value: v.welcome_message || '', placeholder: 'Hi! Ask me about the sales data.' });
    const ins = h('textarea', { id: 'af-ins', rows: 5, maxlength: 1500, placeholder: 'Purpose, audience and tone. Example: You help shop managers; keep answers short and use plain language.' }); ins.value = v.instructions || '';
    const style = h('select', { id: 'af-style', class: 'select' }, o.styles.map((s) => h('option', { value: s.id, text: s.label, selected: (v.style || 'balanced') === s.id })));
    const model = h('select', { id: 'af-model', class: 'select' }, h('option', { value: '', text: `Server default${o.default_model ? ` (${o.default_model})` : ''}` }), o.models.map((m) => h('option', { value: m, text: m, selected: v.model === m })));
    const caps = o.capabilities.map((c) => h('label', { class: 'check', style: 'align-items:flex-start' }, h('input', { type: 'checkbox', value: c.id, checked: (v.capabilities || o.capabilities.map((x) => x.id)).includes(c.id) }), h('span', {}, h('strong', { text: c.label }), h('br'), h('span', { class: 'muted small', text: c.help }))));
    const ex = h('textarea', { id: 'af-ex', rows: 4, placeholder: 'One question per line (up to 5)' }); ex.value = (v.examples || []).join('\n');
    const panes = [
      h('div', { class: 'tab-panel', 'data-pane': 0 }, f('Name', name), edit ? h('p', { class: 'muted small', text: `Dataset: ${v.dataset ? v.dataset.name : '—'} (${v.dataset_version.label}). To use another dataset, create a new assistant.` }) : f('Dataset', dsSel, 'The assistant can only see this dataset. Its current cleaning steps become the version the assistant is bound to.'), f('Description', desc), f('Welcome message (optional)', welcome)),
      h('div', { class: 'tab-panel', 'data-pane': 1 }, f('Instructions', ins, 'These shape purpose and tone. They can’t override safety rules or make the assistant ignore the data.'), f('Response style', style), o.models.length ? f('Model', model, 'Provider: Groq.') : null),
      h('div', { class: 'tab-panel', 'data-pane': 2 }, h('div', { class: 'field' }, h('label', { text: 'What may it calculate?' }), h('div', { class: 'kv' }, caps), h('p', { class: 'muted small', text: 'Private-looking columns (emails, phones, addresses…) are always hidden.' })), f('Example questions (optional)', ex)),
    ];
    const collect = () => {
      const p = { name: name.value, description: desc.value, welcome_message: welcome.value, instructions: ins.value, style: style.value, model: o.models.length ? model.value : '',
        capabilities: $$('input[type=checkbox]:checked', caps[0].parentNode).map((i) => i.value), examples: ex.value.split('\n').map((x) => x.trim()).filter(Boolean).slice(0, 5) };
      if (!edit) p.dataset_id = dsSel.value; return p;
    };
    return { panes, collect };
  }
  function renderAsWizard(a) {
    const form = asForm(null, false), titles = ['Basics', 'Behaviour', 'Abilities'];
    const err = h('p', { class: 'field-error', role: 'alert' });
    const stepBar = h('div', { class: 'tabs' }, titles.map((t, i) => h('button', { type: 'button', class: 'tab', onclick: () => go(i), text: `${i + 1}. ${t}` })));
    const back = h('button', { type: 'button', class: 'btn btn-soft', onclick: () => go(AS.step - 1), text: 'Back' });
    const next = h('button', { type: 'button', class: 'btn btn-primary', onclick: () => go(AS.step + 1), text: 'Next' });
    const save = h('button', { type: 'button', class: 'btn btn-primary', text: 'Save draft and test' });
    function go(i) { AS.step = Math.max(0, Math.min(2, i)); form.panes.forEach((p, k) => { p.hidden = k !== AS.step; }); $$('.tab', stepBar).forEach((t, k) => t.classList.toggle('active', k === AS.step)); back.hidden = AS.step === 0; next.hidden = AS.step === 2; save.hidden = AS.step !== 2; }
    save.addEventListener('click', async () => {
      setBusy(save, true); err.textContent = '';
      try { const d = await post('/api/assistants', form.collect()); toast('Draft saved. Test it in the chat before publishing.', 'success'); location.hash = `#/assistants/${d.assistant.id}`; }
      catch (e) { err.textContent = Object.values(e.fields || {})[0] || e.message; if (e.fields?.name) go(0); } finally { setBusy(save, false); }
    });
    mount(asRoot(), h('div', { class: 'page-head' }, h('div', {}, h('h2', { text: 'Create an assistant' }), h('p', { class: 'muted', text: 'Three short steps. Sensible defaults are already filled in.' }))),
      state.datasets.length ? h('div', { class: 'card' }, stepBar, form.panes, err, h('div', { class: 'btn-row' }, back, next, save)) : emptyState('database', 'Upload a dataset first', 'An assistant answers from one of your datasets.', uploadCta()));
    if (state.datasets.length) go(0);
  }
  function renderAsWorkspace() {
    const a = AS.cur;
    const tabs = [['chat', 'Test chat', 'message-square'], ['config', 'Configure', 'sliders-horizontal'], ['publish', 'Publish', 'rocket']];
    const body = h('div', { id: 'as-body' });
    const bar = h('div', { class: 'tabs' }, tabs.map(([id, label, ic]) => h('button', { type: 'button', class: `tab ${AS.tab === id ? 'active' : ''}`, onclick: () => { AS.tab = id; renderAsWorkspace(); } }, ico(ic), label)));
    const banner = a.update_available ? h('div', { class: 'notice' }, h('strong', { text: 'The dataset was edited. ' }),
      `This assistant still answers from “${a.dataset_version.label}” (${fmt(a.dataset_version.rows)} rows). Latest: ${a.latest_version.label} (${fmt(a.latest_version.rows)} rows). `,
      h('div', { class: 'btn-row', style: 'margin-top:8px' }, h('button', { class: 'btn btn-primary btn-sm', onclick: () => asDatasetChoice('update'), text: 'Update to latest data' }), h('button', { class: 'btn btn-soft btn-sm', onclick: () => asDatasetChoice('keep'), text: 'Keep current version' }))) : null;
    mount(asRoot(), h('div', { class: 'page-head' }, h('div', {}, h('h2', { text: a.name }), h('p', { class: 'muted small', text: `${a.dataset ? a.dataset.name : 'dataset missing'} · ${a.dataset_version.label} · ${fmt(a.dataset_version.rows)} rows · v${a.version}` })),
      h('div', { class: 'head-actions' }, statusBadge(a), h('a', { class: 'btn btn-soft btn-sm', href: '#/assistants' }, 'All assistants'))), banner, bar, body);
    if (AS.tab === 'chat') renderAsChat(body); else if (AS.tab === 'config') renderAsConfig(body); else renderAsPublish(body);
  }
  async function asDatasetChoice(action) {
    try { const d = await post(`/api/assistants/${AS.cur.id}/dataset`, { action }); AS.cur = d.assistant; toast(action === 'update' ? 'Assistant updated to the latest data. Test it again before publishing.' : 'Keeping the current data version.', 'success'); renderAsWorkspace(); }
    catch (e) { toast(e.message, 'error'); }
  }
  function renderAsConfig(body) {
    const form = asForm(AS.cur, true), err = h('p', { class: 'field-error', role: 'alert' });
    const btn = h('button', { class: 'btn btn-primary', text: 'Save changes', onclick: async () => {
      setBusy(btn, true); err.textContent = '';
      try { const d = await patch(`/api/assistants/${AS.cur.id}`, form.collect()); AS.cur = d.assistant; toast('Saved. Test the assistant again before publishing.', 'success'); renderAsWorkspace(); }
      catch (e) { err.textContent = Object.values(e.fields || {})[0] || e.message; } finally { setBusy(btn, false); }
    } });
    mount(body, h('div', { class: 'card' }, form.panes.map((p) => h('div', {}, p)), err, h('div', { class: 'btn-row' }, btn)));
  }
  function renderAsPublish(body) {
    const a = AS.cur, o = AS.opts;
    const row = (okv, text) => h('li', {}, ico(okv ? 'circle-check' : 'circle-alert'), h('span', { text }));
    const ack = h('input', { type: 'checkbox', id: 'as-ack' });
    const can = o.ai_configured && a.tested && a.dataset;
    const pub = h('button', { class: 'btn btn-primary', text: 'Publish', disabled: true, onclick: async () => {
      setBusy(pub, true);
      try { const d = await post(`/api/assistants/${a.id}/publish`, { acknowledge: true }); AS.cur = d.assistant; toast('Published. Create an API key to call it.', 'success'); location.hash = `#/api/${a.id}`; }
      catch (e) { toast(e.message, 'error'); } finally { setBusy(pub, false); }
    } });
    ack.addEventListener('change', () => { pub.disabled = !(ack.checked && can); });
    const del_ = h('button', { class: 'btn btn-danger-soft', text: 'Delete assistant', onclick: async () => {
      const typed = await promptDialog({ title: 'Delete this assistant?', text: `This permanently deletes its conversations and revokes its API keys. Type “${a.name}” to confirm.`, label: 'Assistant name', okLabel: 'Delete', validate: (x) => x === a.name ? null : 'The name doesn’t match.' });
      if (typed === null) return;
      try { await del(`/api/assistants/${a.id}`, { confirm_name: typed }); toast('Assistant deleted.', 'success'); location.hash = '#/assistants'; } catch (e) { toast(e.message, 'error'); }
    } });
    const live = a.status === 'published';
    mount(body, h('div', { class: 'card' }, h('h3', { text: live ? 'This assistant is live' : 'Publish through an API' }),
      h('ul', { class: 'steps' }, row(o.ai_configured, o.ai_configured ? 'AI provider is configured on the server' : 'The server needs GROQ_API_KEY before assistants can answer'),
        row(a.tested, a.tested ? 'Tested with a real answer' : 'Send at least one message in Test chat (editing re-opens this check)'), row(!!a.dataset, `Bound to ${a.dataset_version.label}`)),
      live ? h('div', { class: 'btn-row' }, h('a', { class: 'btn btn-primary', href: `#/api/${a.id}` }, 'Open API & keys'), h('button', { class: 'btn btn-soft', text: 'Unpublish', onclick: async () => {
          try { const d = await post(`/api/assistants/${a.id}/unpublish`); AS.cur = d.assistant; toast('Unpublished. API calls now fail with 403.', 'success'); renderAsWorkspace(); } catch (e) { toast(e.message, 'error'); } } }))
        : h('div', {}, h('label', { class: 'check', for: 'as-ack' }, ack, ' I understand that anyone with an API key can read answers drawn from this dataset version.'), h('div', { class: 'btn-row', style: 'margin-top:12px' }, pub))),
      h('div', { class: 'card' }, h('h3', { text: 'Danger zone' }), h('p', { class: 'muted small', text: 'Deleting removes the assistant, its chats and API keys. The dataset is not touched.' }), del_));
  }
  /* ---- test chat ---- */
  function asMsg(box, m, evidence) {
    const isUser = m.role === 'user', bodyEl = h('div', { class: `msg-body ${m.status === 'failed' ? 'failed' : ''}` });
    if (isUser) bodyEl.textContent = m.content; else bodyEl.append(renderMarkdown(m.content));
    const used = (evidence || []).filter((e) => e.title).map((e) => e.title);
    const col = h('div', { class: 'msg-col' }, bodyEl, h('div', { class: 'msg-meta', text: used.length ? `Calculated: ${used.join('; ')}` : (m.created_at ? timeAgo(m.created_at) : '') }));
    const root = h('div', { class: `msg ${isUser ? 'msg-user' : 'msg-ai'}` }, h('div', { class: 'msg-avatar', 'aria-hidden': 'true' }, isUser ? (state.user.full_name[0] || 'U') : ico('sparkles')), col);
    box.append(root); refreshIcons(); box.scrollTop = box.scrollHeight; return root;
  }
  async function renderAsChat(body) {
    const a = AS.cur, box = h('div', { class: 'messages', id: 'as-messages', 'aria-live': 'polite', style: 'min-height:260px;max-height:55vh;overflow:auto' });
    const input = h('textarea', { id: 'as-input', rows: 1, maxlength: 2000, placeholder: `Ask ${a.name}…` });
    const sendBtn = h('button', { type: 'submit', class: 'btn btn-primary send-btn', 'aria-label': 'Send' }, ico('send'));
    const sel = h('select', { class: 'select select-sm', 'aria-label': 'Conversation' });
    const starters = h('div', { class: 'chips' });
    const showWelcome = () => { clear(box); box.append(h('div', { class: 'empty-inline', text: a.welcome_message || `Ask ${a.name} about ${a.dataset ? a.dataset.name : 'its dataset'}. Try the examples below.` }));
      mount(starters, (a.examples.length ? a.examples : ['What does this dataset contain?', 'What are the most important numbers here?']).map((q) => h('button', { type: 'button', class: 'chip', onclick: () => send(q) }, h('strong', { text: q })))); };
    async function loadConvs() { const d = await get(`/api/assistants/${a.id}/conversations`); AS.convs = d.conversations;
      mount(sel, h('option', { value: '', text: 'New conversation' }), AS.convs.map((c) => h('option', { value: c.id, text: c.title, selected: AS.conv && AS.conv.id === c.id }))); }
    async function open(id) { if (!id) { AS.conv = null; showWelcome(); return; }
      const d = await get(`/api/assistant-conversations/${id}`); AS.conv = d.conversation; clear(box); clear(starters); d.messages.forEach((m) => asMsg(box, m)); }
    async function send(text) {
      text = (text || '').trim(); if (!text || AS.sending) return; AS.sending = true; sendBtn.disabled = true; clear(starters);
      let userNode = null, typing = null;
      try {
        if (!AS.conv) { AS.conv = (await post(`/api/assistants/${a.id}/conversations`)).conversation; }
        if (box.querySelector('.empty-inline')) clear(box);
        userNode = asMsg(box, { role: 'user', content: text, status: 'pending' });
        typing = h('div', { class: 'msg msg-ai' }, h('div', { class: 'msg-avatar' }, ico('sparkles')), h('div', { class: 'msg-col' }, h('div', { class: 'msg-body' }, h('span', { class: 'typing' }, h('span'), h('span'), h('span'))))); box.append(typing); refreshIcons(); box.scrollTop = box.scrollHeight;
        const d = await post(`/api/assistant-conversations/${AS.conv.id}/messages`, { content: text });
        typing.remove(); asMsg(box, d.message, d.evidence); AS.cur = d.assistant; await loadConvs();
        if (!a.tested) { a.tested = true; toast('Tested. You can now publish this assistant.', 'success'); }
      } catch (e) {
        typing?.remove(); const col = userNode && $('.msg-col', userNode);
        if (col) { const retry = h('button', { class: 'btn btn-soft btn-sm', type: 'button', onclick: async () => { userNode.remove(); AS.sending = false; try { await post(`/api/assistant-conversations/${AS.conv.id}/messages`, { content: text, retry: true }); await open(AS.conv.id); } catch (e2) { toast(e2.message, 'error'); } } }, ico('refresh-cw'), 'Retry');
          col.append(h('div', { class: 'msg-error', role: 'alert' }, ico('triangle-alert'), h('span', { text: e.message }), retry)); refreshIcons(); } else toast(e.message, 'error');
      } finally { AS.sending = false; sendBtn.disabled = false; }
    }
    const form = h('form', { class: 'composer', onsubmit: (e) => { e.preventDefault(); const t = input.value; input.value = ''; send(t); } }, input, sendBtn);
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); } });
    sel.addEventListener('change', () => open(sel.value).catch((e) => toast(e.message, 'error')));
    const delBtn = h('button', { class: 'btn btn-danger-soft btn-sm', type: 'button', text: 'Delete chat', onclick: async () => {
      if (!AS.conv || !await confirmDialog({ title: 'Delete this conversation?', text: 'The messages are permanently removed.', okLabel: 'Delete' })) return;
      try { await del(`/api/assistant-conversations/${AS.conv.id}`); AS.conv = null; await loadConvs(); showWelcome(); } catch (e) { toast(e.message, 'error'); } } });
    mount(body, h('div', { class: 'card' }, h('div', { class: 'btn-row' }, sel, delBtn), box, starters, form,
      h('p', { class: 'muted small', text: 'Answers use calculations on the bound dataset version. AI can make mistakes; check important figures.' })));
    if (!AS.opts.ai_configured) body.prepend(h('div', { class: 'ai-banner' }, 'The server needs GROQ_API_KEY before this assistant can answer.'));
    showWelcome(); await loadConvs();
  }

  /* ===================== API & Developers ===================== */
  const copyText = async (t, msg = 'Copied.') => { try { await navigator.clipboard.writeText(t); toast(msg, 'success'); } catch (_) { toast('Copy failed. Select the text and copy it manually.', 'error'); } };
  function showSecret(secret, note) {
    $('#dlg-secret-val').textContent = secret; $('#dlg-secret-note').textContent = note;
    $('#dlg-secret-copy').onclick = () => copyText(secret, 'Key copied.'); $('#dlg-secret').showModal(); refreshIcons();
  }
  async function loadApiPage(arg) {
    const root = $('#api-root'); mount(root, skeleton(4));
    const list = (await get('/api/assistants')).assistants;
    if (!list.length) return mount(root, h('div', { class: 'page-head' }, h('h2', { text: 'API & Developers' })), emptyState('code-xml', 'No assistants yet', 'Create and publish an assistant to get an API endpoint.', h('a', { class: 'btn btn-primary', href: '#/assistants/new' }, 'Create assistant')));
    const id = arg && list.some((a) => a.id === arg) ? arg : list[0].id;
    const d = await get(`/api/assistants/${id}/api`), a = d.assistant, docs = d.docs;
    const pick = h('select', { class: 'select', 'aria-label': 'Assistant', onchange: (e) => { location.hash = `#/api/${e.target.value}`; } }, list.map((x) => h('option', { value: x.id, text: x.name, selected: x.id === id })));
    const reload = () => loadApiPage(id).catch((e) => toast(e.message, 'error'));
    const rowFor = (k) => h('tr', {}, h('td', { text: k.name }), h('td', {}, h('code', { text: `${k.prefix}…` })), h('td', {}, h('span', { class: `badge ${k.status === 'active' ? 'badge-ok' : 'badge-bad'}`, text: k.status })),
      h('td', { text: k.last_used_at ? timeAgo(k.last_used_at) : 'never' }), h('td', { text: `${fmt(k.requests)} (${fmt(k.errors)} errors)` }), h('td', { text: `${fmt(k.requests_today)} / ${fmt(k.daily_limit)}` }),
      h('td', {}, k.status === 'active' ? h('div', { class: 'btn-row' },
        h('button', { class: 'btn btn-soft btn-sm', text: 'Rotate', onclick: async () => { if (!await confirmDialog({ title: 'Rotate this key?', text: 'A new key is created and this one stops working immediately.', okLabel: 'Rotate', danger: false })) return;
          try { const r = await post(`/api/assistants/${id}/keys/${k.id}/rotate`); showSecret(r.secret, r.secret_notice); reload(); } catch (e) { toast(e.message, 'error'); } } }),
        h('button', { class: 'btn btn-danger-soft btn-sm', text: 'Revoke', onclick: async () => { if (!await confirmDialog({ title: 'Revoke this key?', text: 'Apps using it will immediately get 401 errors.', okLabel: 'Revoke' })) return;
          try { await post(`/api/assistants/${id}/keys/${k.id}/revoke`); toast('Key revoked.', 'success'); reload(); } catch (e) { toast(e.message, 'error'); } } })) : '—'));
    const newKey = h('button', { class: 'btn btn-primary btn-sm', text: 'Create key', onclick: async () => {
      const name = await promptDialog({ title: 'Create API key', label: 'Key name', text: 'Name the app or server that will use it.', okLabel: 'Create', validate: (v) => v.trim() ? null : 'Enter a name.' });
      if (name === null) return;
      try { const r = await post(`/api/assistants/${id}/keys`, { name: name.trim() }); showSecret(r.secret, r.secret_notice); reload(); } catch (e) { toast(e.message, 'error'); } } });
    const snip = (title, text) => h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', { text: title }), h('button', { class: 'btn btn-soft btn-sm', onclick: () => copyText(text) }, ico('copy'), 'Copy')), h('pre', { class: 'code-block' }, h('code', { text })));
    const canvas = h('canvas', { 'aria-label': 'Requests in the last 7 days' });
    mount(root, h('div', { class: 'page-head' }, h('div', {}, h('h2', { text: 'API & Developers' }), h('p', { class: 'muted', text: 'Call a published assistant from your own server.' })), h('div', { class: 'head-actions' }, pick)),
      h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', { text: a.name }), statusBadge(a)),
        h('div', { class: 'kv' }, h('div', { class: 'kv-row' }, h('span', { text: 'Endpoint' }), h('span', {}, h('code', { text: `POST ${docs.endpoint}` }), ' ', h('button', { class: 'icon-btn', 'aria-label': 'Copy endpoint', onclick: () => copyText(docs.endpoint) }, ico('copy')))),
          h('div', { class: 'kv-row' }, h('span', { text: 'Data version' }), h('span', { text: `${a.dataset_version.label}${a.update_available ? ' (newer data exists)' : ''}` })),
          h('div', { class: 'kv-row' }, h('span', { text: 'AI provider' }), h('span', { text: d.provider_ready ? `Groq · ${d.model}` : 'Not configured (set GROQ_API_KEY): calls return 503' }))),
        a.status !== 'published' ? h('div', { class: 'notice' }, 'This assistant isn’t published, so API calls return 403. ', h('a', { class: 'link', href: `#/assistants/${id}` }, 'Test and publish it.')) : null),
      h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', { text: 'API keys' }), newKey),
        d.keys.length ? h('div', { class: 'table-wrap' }, h('table', { class: 'data-table' }, h('thead', {}, h('tr', {}, ['Name', 'Key', 'Status', 'Last used', 'Requests', 'Today', ''].map((t) => h('th', { text: t })))), h('tbody', {}, d.keys.map(rowFor))))
          : h('p', { class: 'muted', text: 'No keys yet. A key is shown once when created; only a hash is stored.' }),
        h('p', { class: 'muted small', text: `Limits per key: ${docs.limits.per_minute_per_key} requests/minute, ${docs.limits.per_day_per_key}/day. Account total: ${docs.limits.per_day_per_account}/day.` })),
      h('div', { class: 'card' }, h('h3', { text: `Usage, last 7 days: ${fmt(d.usage.requests_7d)} requests, ${fmt(d.usage.errors_7d)} errors` }), h('div', { class: 'chart-box' }, canvas),
        d.usage.recent_errors.length ? h('div', {}, h('h4', { text: 'Recent errors' }), h('ul', { class: 'feed' }, d.usage.recent_errors.map((r) => h('li', {}, h('span', { text: `${r.status} ${r.code || ''}${r.key ? ` · key “${r.key}”` : ''}` }), h('span', { class: 'feed-time', text: timeAgo(r.at) }))))) : null),
      snip('curl', docs.snippets.curl), snip('JavaScript (server-side)', docs.snippets.javascript), snip('Python', docs.snippets.python),
      snip('Example request body', JSON.stringify(docs.request_example, null, 2)), snip('Example response', JSON.stringify(docs.response_example, null, 2)),
      h('div', { class: 'card' }, h('h3', { text: 'Errors' }), h('div', { class: 'table-wrap' }, h('table', { class: 'data-table' }, h('thead', {}, h('tr', {}, ['Status', 'Code', 'Meaning'].map((t) => h('th', { text: t })))), h('tbody', {}, docs.errors.map((e) => h('tr', {}, h('td', { text: String(e.status) }), h('td', {}, h('code', { text: e.code })), h('td', { text: e.meaning })))))),
        h('ul', { class: 'steps' }, docs.notes.map((n) => h('li', {}, ico('info'), h('span', { text: n }))))));
    drawChart('api-usage', canvas, { type: 'bar', data: { labels: d.usage.days.map((x) => fmtDay(x.date)), datasets: [{ label: 'Requests', data: d.usage.days.map((x) => x.requests), backgroundColor: '#7c5cff', borderRadius: 4 }, { label: 'Errors', data: d.usage.days.map((x) => x.errors), backgroundColor: '#f87171', borderRadius: 4 }] }, options: { scales: { y: { beginAtZero: true, ticks: { precision: 0 } } } } });
    refreshIcons();
  }

  async function init() {
    setupChartDefaults(); wireAuth(); wireShell(); wireUpload(); wireAnalysis(); wireChat(); wireSettings(); wireAdmin(); wireForcedPassword(); wireMisc();
    if (!window.Chart) toast('The charts library couldn’t load. Check your internet connection and refresh.', 'error', 9000);
    try {
      const me = await get('/api/me');
      state.ai = me.ai; state.limits = me.limits || {};
      if (me.authenticated) { state.user = me.user; enterApp(); } else { showScreen('auth'); setAuthTab('login'); }
    } catch (e) { showScreen('auth'); toast(e.message, 'error'); }
    refreshIcons();
  }
  init();
})();