/* Labeling canvas UI */

const canvas = document.getElementById('label-canvas');
const ctx = canvas.getContext('2d');
const img = document.getElementById('frame-img');

// ── State ─────────────────────────────────────────────────────────────────────

let classes = [];       // [{id, name, color}]  — source of truth, always use id refs
let annotations = [];   // [{id, class_id, class_name, color, x, y, width, height}]
let selectedClass = null;
let editingClassId = null;  // which class is in rename mode
let _renameBusy = false;    // guard against double-blur firing
let drawing = false;
let startX = 0, startY = 0, curX = 0, curY = 0;

// ── Init ──────────────────────────────────────────────────────────────────────

function init() {
  classes = INITIAL_CLASSES.map(c => Object.assign({}, c));
  annotations = INITIAL_ANNOTATIONS.map(a => Object.assign({}, a));

  img.onload = resizeCanvas;
  if (img.complete) resizeCanvas();

  renderClassList();
  renderAnnotationList();
}

function resizeCanvas() {
  canvas.width = img.naturalWidth || img.clientWidth;
  canvas.height = img.naturalHeight || img.clientHeight;
  canvas.style.width = img.clientWidth + 'px';
  canvas.style.height = img.clientHeight + 'px';
  drawAll();
}

window.addEventListener('resize', resizeCanvas);

// ── Class list rendering ──────────────────────────────────────────────────────

function renderClassList() {
  const container = document.getElementById('cls-list');
  container.innerHTML = '';

  if (classes.length === 0) {
    const hint = document.createElement('p');
    hint.className = 'text-muted text-sm';
    hint.textContent = 'No classes yet. Add one below.';
    container.appendChild(hint);
  }

  classes.forEach(cls => {
    const row = document.createElement('div');
    row.className = 'cls-row' + (selectedClass?.id === cls.id ? ' selected' : '');

    if (editingClassId === cls.id) {
      // ── Rename mode ──
      row.innerHTML = `
        <span class="cls-dot" style="background:${cls.color}"></span>
        <input class="cls-name-input" value="${escapeHtml(cls.name)}"
               onkeydown="handleRenameKey(event, ${cls.id})"
               onblur="commitRename(${cls.id}, this.value)">
        <button class="cls-btn del" onclick="cancelRename()" title="Cancel">✕</button>
      `;
      setTimeout(() => row.querySelector('input')?.focus(), 0);
    } else {
      // ── Normal mode ──
      row.innerHTML = `
        <span class="cls-dot" style="background:${cls.color}"
              onclick="selectClassById(${cls.id})" title="Select"></span>
        <span class="cls-name" onclick="selectClassById(${cls.id})">${escapeHtml(cls.name)}</span>
        <button class="cls-btn" onclick="startRename(${cls.id})" title="Rename">✎</button>
        <button class="cls-btn del" onclick="deleteClass(${cls.id})" title="Delete">×</button>
      `;
    }
    container.appendChild(row);
  });
}

// ── Class selection ───────────────────────────────────────────────────────────

function selectClassById(id) {
  selectedClass = classes.find(c => c.id === id) || null;
  renderClassList();
}

// ── Class rename ──────────────────────────────────────────────────────────────

function startRename(id) {
  editingClassId = id;
  renderClassList();
}

function cancelRename() {
  editingClassId = null;
  renderClassList();
}

function handleRenameKey(e, id) {
  if (e.key === 'Enter') { e.target.blur(); }   // triggers onblur → commitRename
  if (e.key === 'Escape') { editingClassId = null; renderClassList(); }
}

async function commitRename(id, newName) {
  if (_renameBusy) return;
  _renameBusy = true;
  editingClassId = null;

  newName = newName.trim();
  const cls = classes.find(c => c.id === id);

  if (!newName || !cls || cls.name === newName) {
    renderClassList();
    _renameBusy = false;
    return;
  }

  const resp = await fetch(`/classes/${id}/rename`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: `name=${encodeURIComponent(newName)}`,
  });

  if (resp.ok) {
    cls.name = newName;
    // Update display names in current annotations (ids remain unchanged)
    annotations.forEach(a => { if (a.class_id === id) a.class_name = newName; });
    drawAll();
    renderAnnotationList();
  }
  renderClassList();
  _renameBusy = false;
}

// ── Add class ─────────────────────────────────────────────────────────────────

function showAddClass() {
  const row = document.getElementById('add-cls-row');
  row.style.display = 'flex';
  document.getElementById('new-cls-input').focus();
}

function cancelAddClass() {
  document.getElementById('add-cls-row').style.display = 'none';
  document.getElementById('new-cls-input').value = '';
}

async function commitAddClass() {
  const input = document.getElementById('new-cls-input');
  const name = input.value.trim();
  if (!name) return;

  const resp = await fetch(`/projects/${PROJECT_ID}/classes`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: `name=${encodeURIComponent(name)}`,
  });

  if (resp.ok) {
    const cls = await resp.json();
    classes.push(cls);
    input.value = '';
    document.getElementById('add-cls-row').style.display = 'none';
    renderClassList();
  }
}

// ── Delete class ──────────────────────────────────────────────────────────────

async function deleteClass(id) {
  const cls = classes.find(c => c.id === id);
  const hasAnnotations = annotations.some(a => a.class_id === id);
  const msg = hasAnnotations
    ? `Delete class "${cls?.name}"? Annotations using it on this frame will also be removed.`
    : `Delete class "${cls?.name}"?`;
  if (!confirm(msg)) return;

  const resp = await fetch(`/classes/${id}/delete`, { method: 'POST' });
  if (resp.ok) {
    classes = classes.filter(c => c.id !== id);
    annotations = annotations.filter(a => a.class_id !== id);
    if (selectedClass?.id === id) selectedClass = null;
    renderClassList();
    renderAnnotationList();
    drawAll();
  }
}

// ── Drawing ───────────────────────────────────────────────────────────────────

canvas.addEventListener('mousedown', e => {
  if (!selectedClass) { alert('Select a class first.'); return; }
  drawing = true;
  const p = canvasPoint(e);
  startX = p.x; startY = p.y; curX = p.x; curY = p.y;
  e.preventDefault();
});

canvas.addEventListener('mousemove', e => {
  if (!drawing) return;
  const p = canvasPoint(e);
  curX = p.x; curY = p.y;
  drawAll();
  const x = Math.min(startX, curX), y = Math.min(startY, curY);
  const w = Math.abs(curX - startX), h = Math.abs(curY - startY);
  ctx.strokeStyle = selectedClass.color;
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 3]);
  ctx.strokeRect(x, y, w, h);
  ctx.setLineDash([]);
});

canvas.addEventListener('mouseup', e => {
  if (!drawing) return;
  drawing = false;
  const p = canvasPoint(e);
  curX = p.x; curY = p.y;

  const w = Math.abs(curX - startX), h = Math.abs(curY - startY);
  if (w < 5 || h < 5) { drawAll(); return; }

  const cw = canvas.width, ch = canvas.height;
  const nx = (Math.min(startX, curX) + w / 2) / cw;
  const ny = (Math.min(startY, curY) + h / 2) / ch;

  annotations.push({
    id: null,
    class_id: selectedClass.id,
    class_name: selectedClass.name,
    color: selectedClass.color,
    x: nx, y: ny, width: w / cw, height: h / ch,
  });

  drawAll();
  renderAnnotationList();
});

function canvasPoint(e) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left) * (canvas.width / rect.width),
    y: (e.clientY - rect.top) * (canvas.height / rect.height),
  };
}

// ── Canvas rendering ──────────────────────────────────────────────────────────

function drawAll() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const cw = canvas.width, ch = canvas.height;

  annotations.forEach(ann => {
    const px = (ann.x - ann.width / 2) * cw;
    const py = (ann.y - ann.height / 2) * ch;
    const pw = ann.width * cw;
    const ph = ann.height * ch;

    ctx.fillStyle = ann.color + '33';
    ctx.fillRect(px, py, pw, ph);
    ctx.strokeStyle = ann.color;
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.strokeRect(px, py, pw, ph);

    ctx.fillStyle = ann.color;
    ctx.font = 'bold 12px system-ui';
    const tw = ctx.measureText(ann.class_name).width;
    ctx.fillRect(px, py - 16, tw + 6, 16);
    ctx.fillStyle = '#fff';
    ctx.fillText(ann.class_name, px + 3, py - 3);
  });
}

// ── Annotation list ───────────────────────────────────────────────────────────

function renderAnnotationList() {
  document.getElementById('ann-count').textContent = annotations.length;
  const container = document.getElementById('ann-items');
  container.innerHTML = '';
  annotations.forEach((ann, i) => {
    const div = document.createElement('div');
    div.className = 'ann-item';
    div.innerHTML = `
      <span style="color:${ann.color}">■</span>
      <span>${escapeHtml(ann.class_name)}</span>
      <button class="ann-delete" onclick="deleteAnnotation(${i})" title="Delete">×</button>
    `;
    container.appendChild(div);
  });
}

function deleteAnnotation(index) {
  annotations.splice(index, 1);
  drawAll();
  renderAnnotationList();
}

function clearAll() {
  if (annotations.length === 0 || confirm('Clear all annotations?')) {
    annotations = [];
    drawAll();
    renderAnnotationList();
  }
}

// ── Save ──────────────────────────────────────────────────────────────────────

async function saveAnnotations() {
  const payload = {
    annotations: annotations.map(a => ({
      class_id: a.class_id,
      x: a.x, y: a.y, width: a.width, height: a.height,
    })),
  };

  const resp = await fetch(`/projects/${PROJECT_ID}/frames/${FRAME_ID}/annotations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (resp.ok) {
    const status = document.createElement('span');
    status.textContent = ' Saved ✓';
    status.style.cssText = 'color:#10b981; margin-left:0.5rem';
    const saveBtn = document.querySelector('[onclick="saveAnnotations()"]');
    saveBtn.after(status);
    setTimeout(() => status.remove(), 2000);
  } else {
    alert('Save failed: ' + resp.statusText);
  }
}

async function saveAndNext() {
  await saveAnnotations();
  if (NEXT_FRAME_ID) {
    window.location.href = `/projects/${PROJECT_ID}/label/${NEXT_FRAME_ID}`;
  }
}

// ── Utility ───────────────────────────────────────────────────────────────────

function escapeHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Start ─────────────────────────────────────────────────────────────────────
init();
