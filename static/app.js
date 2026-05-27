'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
let currentJobId   = null;
let pollTimer      = null;   // kept for fallback health checks only
let eventSource    = null;
let activeTab      = 'paste';

// ── Boot ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadModels();
  loadFamilies();
  setupFileDropzone();

  document.getElementById('contract-text').addEventListener('input', e => {
    document.getElementById('char-count').textContent =
      `${e.target.value.length} characters`;
  });
});

// ── Catalogue loaders ─────────────────────────────────────────────────────────
async function loadModels() {
  try {
    const data = await apiFetch('/models');
    const sel  = document.getElementById('model-select');
    sel.innerHTML = '';
    data.models.forEach(m => {
      const opt   = document.createElement('option');
      opt.value   = m.id;
      opt.textContent = m.display_name;
      if (m.id === data.default) opt.selected = true;
      sel.appendChild(opt);
    });
  } catch {
    document.getElementById('model-select').innerHTML =
      '<option value="claude-haiku-4-5-20251001">Claude Haiku (default)</option>' +
      '<option value="claude-sonnet-4-6">Claude Sonnet</option>';
  }
}

async function loadFamilies() {
  try {
    const data = await apiFetch('/families');
    const grid = document.getElementById('family-grid');
    grid.innerHTML = '';
    data.families.forEach(f => {
      const lbl   = document.createElement('label');
      const cb    = document.createElement('input');
      cb.type     = 'checkbox';
      cb.value    = f.id;
      cb.checked  = true;
      cb.id       = `fam-${f.id}`;
      lbl.appendChild(cb);
      lbl.appendChild(document.createTextNode(f.display_name));
      lbl.title = f.description;
      grid.appendChild(lbl);
    });
  } catch {
    // Fallback if /families fails
    const fallback = [
      ['assignment',       'Assignment'],
      ['change_of_control','Change of Control'],
      ['termination',      'Termination'],
      ['exclusivity',      'Exclusivity / Non-Compete'],
    ];
    const grid = document.getElementById('family-grid');
    grid.innerHTML = '';
    fallback.forEach(([id, name]) => {
      const lbl = document.createElement('label');
      const cb  = document.createElement('input');
      cb.type   = 'checkbox'; cb.value = id; cb.checked = true; cb.id = `fam-${id}`;
      lbl.appendChild(cb);
      lbl.appendChild(document.createTextNode(name));
      grid.appendChild(lbl);
    });
  }
}

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(tab) {
  activeTab = tab;
  document.getElementById('tab-paste').classList.toggle('active', tab === 'paste');
  document.getElementById('tab-file').classList.toggle('active',  tab === 'file');
  document.getElementById('panel-paste').style.display = tab === 'paste' ? '' : 'none';
  document.getElementById('panel-file').style.display  = tab === 'file'  ? '' : 'none';
  setValidationError('');
}

// ── File dropzone ─────────────────────────────────────────────────────────────
function setupFileDropzone() {
  const drop  = document.getElementById('file-drop');
  const input = document.getElementById('file-input');

  input.addEventListener('change', () => showFileName(input.files[0]));

  drop.addEventListener('dragover',  e => { e.preventDefault(); drop.classList.add('drag-over'); });
  drop.addEventListener('dragleave', ()  => drop.classList.remove('drag-over'));
  drop.addEventListener('drop', e => {
    e.preventDefault();
    drop.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) {
      // Assign to file input so it's picked up on submit
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      showFileName(file);
    }
  });
}

function showFileName(file) {
  if (!file) return;
  const name = document.getElementById('file-name');
  if (file.name.endsWith('.pdf')) {
    setValidationError('PDF is not yet supported. Please convert to .txt first.');
    name.textContent = '';
    return;
  }
  name.textContent = `✓ ${file.name}`;
  setValidationError('');
}

// ── Submit ─────────────────────────────────────────────────────────────────────
async function submitReview() {
  setValidationError('');

  // Collect selected families
  const families = [...document.querySelectorAll('#family-grid input[type=checkbox]:checked')]
    .map(cb => cb.value);
  if (families.length === 0) {
    setValidationError('Select at least one clause family.');
    return;
  }

  const form = new FormData();
  form.append('families', families.join(','));
  form.append('model',    document.getElementById('model-select').value);

  if (activeTab === 'paste') {
    const text = document.getElementById('contract-text').value.trim();
    if (text.length < 300) {
      setValidationError(`Text too short (${text.length} chars). Minimum is 300.`);
      return;
    }
    form.append('contract_text', text);
  } else {
    const fileInput = document.getElementById('file-input');
    if (!fileInput.files.length) {
      setValidationError('No file selected.');
      return;
    }
    const file = fileInput.files[0];
    if (file.name.endsWith('.pdf')) {
      setValidationError('PDF is not yet supported. Please convert to .txt first.');
      return;
    }
    form.append('file', file);
  }

  // Lock UI
  setSubmitting(true);

  try {
    const res = await fetch('/review', { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    currentJobId = data.job_id;
    startStreaming();
  } catch (err) {
    setValidationError(err.message);
    setSubmitting(false);
  }
}

// ── Streaming (SSE) ───────────────────────────────────────────────────────────
function startStreaming() {
  showResultsSection();
  setStatus('running', 'Starting review pipeline…');

  if (eventSource) eventSource.close();

  eventSource = new EventSource(`/review/${currentJobId}/stream`);

  eventSource.onmessage = (e) => {
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }

    if (msg.type === 'card') {
      appendClauseCard(msg.card);
      const family = (msg.card.clause_family || '').replace(/_/g, ' ');
      setStatus('running', `Received ${family} clause…`);
    } else if (msg.type === 'done') {
      eventSource.close();
      finishReview(msg.overall_risk);
    } else if (msg.type === 'error') {
      eventSource.close();
      setStatus('failed', `Failed: ${msg.message || 'unknown error'}`);
      setSubmitting(false);
    }
  };

  eventSource.onerror = () => {
    // Connection dropped unexpectedly — fall back to polling the status endpoint
    eventSource.close();
    if (currentJobId) pollTimer = setInterval(fallbackPoll, 2500);
  };
}

async function fallbackPoll() {
  try {
    const data = await apiFetch(`/review/${currentJobId}`);
    if (data.status === 'done') {
      clearInterval(pollTimer);
      renderResults(data.result);
      setStatus('done', 'Review complete.');
      setSubmitting(false);
    } else if (data.status === 'failed') {
      clearInterval(pollTimer);
      setStatus('failed', `Failed: ${data.error || 'unknown error'}`);
      setSubmitting(false);
    }
  } catch (err) {
    console.warn('Fallback poll error:', err);
  }
}

function appendClauseCard(card) {
  const container = document.getElementById('cards-container');
  const el = buildClauseCard(card);
  el.classList.add('card-stream-in');
  container.appendChild(el);
}

function finishReview(overallRisk) {
  const risk  = overallRisk || 'none';
  const badge = document.getElementById('overall-badge');
  badge.textContent = risk.toUpperCase();
  badge.className   = `risk-badge ${risk}`;
  document.getElementById('summary-card').style.display = '';
  setStatus('done', 'Review complete.');
  setSubmitting(false);
}

// ── Render results (used by fallback poll only) ───────────────────────────────
function renderResults(result) {
  const container = document.getElementById('cards-container');
  container.innerHTML = '';
  (result.clause_cards || []).forEach(card => container.appendChild(buildClauseCard(card)));
  finishReview(result.overall_risk_rating);
  if (result.overall_summary) {
    renderSummaryData({ overall_summary: result.overall_summary, top_red_flags: result.top_red_flags });
  }
}

function buildClauseCard(card) {
  const risk    = card.llm_generated_risk_rating || 'none';
  const family  = (card.clause_family || '').replace(/_/g, ' ');
  const found   = card.clause_found;
  const notes   = card.confidence_uncertainty_notes || [];
  const similar = card.similar_precedents || [];
  const contrast= card.contrasting_precedents || [];

  const el = document.createElement('div');
  el.className = 'clause-card';
  el.innerHTML = `
    <div class="clause-header">
      <span class="family-name">${family}</span>
      <span class="found-badge ${found ? 'found' : 'not-found'}">${found ? 'Clause found' : 'Not found'}</span>
      <span class="risk-badge ${risk}">${risk === 'none' ? 'N/A' : risk.toUpperCase()}</span>
    </div>
    <div class="clause-body">
      <div class="rationale">${escHtml(card.risk_rationale || 'No rationale generated.')}</div>
      ${notes.length ? `<div class="notes"><ul>${notes.map(n => `<li>${escHtml(n)}</li>`).join('')}</ul></div>` : ''}
      ${buildPrecedents('Similar Precedents', similar, p => escHtml(p.why_similar), p => p.contract_id)}
      ${buildPrecedents('Contrasting Precedents', contrast, p => escHtml(p.why_contrasting), p => p.contract_id)}
    </div>`;
  return el;
}

function buildPrecedents(title, items, reasonFn, idFn) {
  if (!items.length) return '';
  const uid = `prec-${Math.random().toString(36).slice(2)}`;
  const rows = items.map(p => `
    <div class="precedent-item">
      <strong>${escHtml(idFn(p))}</strong>
      ${reasonFn(p)}
    </div>`).join('');
  return `
    <div class="precedent-section">
      <button class="precedent-toggle" onclick="togglePrecedents(this,'${uid}')">
        <span class="arrow">▶</span> ${title} (${items.length})
      </button>
      <div class="precedent-list" id="${uid}">${rows}</div>
    </div>`;
}

function togglePrecedents(btn, uid) {
  btn.classList.toggle('open');
  const list = document.getElementById(uid);
  list.classList.toggle('open');
}

// ── Summarize ─────────────────────────────────────────────────────────────────
async function requestSummary() {
  const btn = document.getElementById('summarize-btn');
  btn.disabled = true;
  btn.textContent = 'Generating summary…';

  try {
    const data = await apiFetch(`/review/${currentJobId}/summarize`, { method: 'POST' });
    renderSummaryData(data);
  } catch (err) {
    document.getElementById('summary-output').innerHTML =
      `<p class="error-msg">Summary failed: ${escHtml(err.message)}</p>`;
  } finally {
    btn.style.display = 'none';
  }
}

function renderSummaryData(data) {
  const flags = (data.top_red_flags || []);
  const flagHtml = flags.length
    ? `<ul class="flags-list">${flags.map(f => `<li>${escHtml(f)}</li>`).join('')}</ul>`
    : '<p style="color:var(--muted);font-size:.85rem">No red flags flagged.</p>';

  document.getElementById('summary-output').innerHTML = `
    <div class="summary-text">${escHtml(data.overall_summary || '')}</div>
    <div style="font-size:.8rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:8px">Red Flags</div>
    ${flagHtml}`;

  document.getElementById('summarize-btn').style.display = 'none';
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function showResultsSection() {
  document.getElementById('results-section').style.display = '';
  document.getElementById('cards-container').innerHTML = '';
  document.getElementById('summary-card').style.display = 'none';
  document.getElementById('summary-output').innerHTML = '';
  document.getElementById('summarize-btn').disabled = false;
  document.getElementById('summarize-btn').style.display = '';
  document.getElementById('summarize-btn').textContent = 'Generate Summary & Red Flags';
  document.getElementById('results-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function setStatus(type, msg) {
  const bar     = document.getElementById('status-bar');
  const spinner = document.getElementById('status-spinner');
  const text    = document.getElementById('status-text');
  bar.className = `status-bar ${type}`;
  spinner.style.display = type === 'running' || type === 'pending' ? '' : 'none';
  text.textContent = msg;
}

function setSubmitting(on) {
  const btn = document.getElementById('submit-btn');
  btn.disabled = on;
  btn.querySelector('span').textContent = on ? 'Reviewing…' : 'Review Contract';
}

function setValidationError(msg) {
  document.getElementById('validation-error').textContent = msg;
}

async function apiFetch(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || res.statusText);
  }
  return res.json();
}

function escHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
