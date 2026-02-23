/* Labeling canvas UI */

const canvas = document.getElementById('label-canvas');
const ctx = canvas.getContext('2d');
const img = document.getElementById('frame-img');

// ── Constants ─────────────────────────────────────────────────────────────────

const HANDLE_RADIUS = 6;
const HANDLE_CURSORS = {
  tl: 'nw-resize', tr: 'ne-resize',
  bl: 'sw-resize', br: 'se-resize',
  t:  'n-resize',  b:  's-resize',
  l:  'w-resize',  r:  'e-resize',
};

// ── State ─────────────────────────────────────────────────────────────────────

let classes = [];       // [{id, name, color}]
let annotations = [];   // [{id, class_id, class_name, color, x, y, width, height}]
let selectedClass = null;
let editingClassId = null;
let _renameBusy = false;

let dirty = false;
let drawing = false;
let startX = 0, startY = 0, curX = 0, curY = 0;

let resizing = false;
let resizeAnnIndex = null;
let resizeHandle = null;
let resizeOrigAnn = null;

let hoveredAnnIndex = null;

// ── Init ──────────────────────────────────────────────────────────────────────

function init() {
  classes = INITIAL_CLASSES.map(c => Object.assign({}, c));
  annotations = INITIAL_ANNOTATIONS.map(a => Object.assign({}, a));

  img.onload = resizeCanvas;
  if (img.complete) resizeCanvas();

  renderClassList();
  renderAnnotationList();
  updateNegativeBanner();

  window.addEventListener('beforeunload', e => {
    if (dirty) { e.preventDefault(); e.returnValue = ''; }
  });
}

let labelStatus = (typeof LABEL_STATUS !== 'undefined') ? LABEL_STATUS : 'unlabeled';

function updateNegativeBanner() {
  const banner = document.getElementById('negative-banner');
  if (banner) banner.style.display = (labelStatus === 'negative') ? 'block' : 'none';
}

function resizeCanvas() {
  canvas.width = img.naturalWidth || img.clientWidth;
  canvas.height = img.naturalHeight || img.clientHeight;
  canvas.style.width = img.clientWidth + 'px';
  canvas.style.height = img.clientHeight + 'px';

  // Lock wrapper to current size so browser resize can't cause bbox misalignment
  const wrapper = document.getElementById('canvas-wrapper');
  wrapper.style.width = img.clientWidth + 'px';

  drawAll();
}

function markDirty() { dirty = true; }
function markClean() { dirty = false; }

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
      row.innerHTML = `
        <span class="cls-dot" style="background:${cls.color}"></span>
        <input class="cls-name-input" value="${escapeHtml(cls.name)}"
               onkeydown="handleRenameKey(event, ${cls.id})"
               onblur="commitRename(${cls.id}, this.value)">
        <button class="cls-btn del" onclick="cancelRename()" title="Cancel">✕</button>
      `;
      setTimeout(() => row.querySelector('input')?.focus(), 0);
    } else {
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
  if (e.key === 'Enter') { e.target.blur(); }
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
    annotations.forEach(a => { if (a.class_id === id) a.class_name = newName; });
    drawAll();
    renderAnnotationList();
  }
  renderClassList();
  _renameBusy = false;
}

// ── Add class ─────────────────────────────────────────────────────────────────

function showAddClass() {
  document.getElementById('add-cls-row').style.display = 'flex';
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

// ── Coordinate helper ─────────────────────────────────────────────────────────

// Returns canvas-space coordinates clamped to [0, canvas dimensions].
// Works for events fired anywhere on the document (during drag).
function clampedPoint(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: Math.max(0, Math.min(canvas.width,  (e.clientX - rect.left) * scaleX)),
    y: Math.max(0, Math.min(canvas.height, (e.clientY - rect.top)  * scaleY)),
  };
}

// ── Resize helpers ────────────────────────────────────────────────────────────

function getHandlePositions(ann) {
  const cw = canvas.width, ch = canvas.height;
  const px = (ann.x - ann.width  / 2) * cw;
  const py = (ann.y - ann.height / 2) * ch;
  const pw = ann.width  * cw;
  const ph = ann.height * ch;
  return {
    tl: { x: px,        y: py        },
    tr: { x: px + pw,   y: py        },
    bl: { x: px,        y: py + ph   },
    br: { x: px + pw,   y: py + ph   },
    t:  { x: px + pw/2, y: py        },
    r:  { x: px + pw,   y: py + ph/2 },
    b:  { x: px + pw/2, y: py + ph   },
    l:  { x: px,        y: py + ph/2 },
  };
}

function hitTestHandles(mx, my) {
  const r2 = HANDLE_RADIUS * HANDLE_RADIUS;
  for (let i = annotations.length - 1; i >= 0; i--) {
    const handles = getHandlePositions(annotations[i]);
    for (const [name, pos] of Object.entries(handles)) {
      const dx = mx - pos.x, dy = my - pos.y;
      if (dx * dx + dy * dy <= r2) return { annIndex: i, handle: name };
    }
  }
  return null;
}

function hitTestAnnotation(mx, my) {
  const cw = canvas.width, ch = canvas.height;
  for (let i = annotations.length - 1; i >= 0; i--) {
    const ann = annotations[i];
    const px = (ann.x - ann.width  / 2) * cw;
    const py = (ann.y - ann.height / 2) * ch;
    const pw = ann.width  * cw;
    const ph = ann.height * ch;
    if (mx >= px && mx <= px + pw && my >= py && my <= py + ph) return i;
  }
  return null;
}

function applyResize(annIndex, handle, mx, my) {
  const orig = resizeOrigAnn;
  const cw = canvas.width, ch = canvas.height;
  let left   = (orig.x - orig.width  / 2) * cw;
  let right  = (orig.x + orig.width  / 2) * cw;
  let top    = (orig.y - orig.height / 2) * ch;
  let bottom = (orig.y + orig.height / 2) * ch;

  const cx = Math.max(0, Math.min(cw, mx));
  const cy = Math.max(0, Math.min(ch, my));

  // Move the appropriate edge(s), enforcing 5px minimum size
  if (handle === 'l' || handle === 'tl' || handle === 'bl') left   = Math.min(cx, right  - 5);
  if (handle === 'r' || handle === 'tr' || handle === 'br') right  = Math.max(cx, left   + 5);
  if (handle === 't' || handle === 'tl' || handle === 'tr') top    = Math.min(cy, bottom - 5);
  if (handle === 'b' || handle === 'bl' || handle === 'br') bottom = Math.max(cy, top    + 5);

  const pw = right - left;
  const ph = bottom - top;
  const ann = annotations[annIndex];
  ann.x      = (left + pw / 2) / cw;
  ann.y      = (top  + ph / 2) / ch;
  ann.width  = pw / cw;
  ann.height = ph / ch;
}

// ── Canvas events ─────────────────────────────────────────────────────────────

canvas.addEventListener('mousedown', e => {
  const p = clampedPoint(e);

  // Resize handle takes priority over drawing
  const hit = hitTestHandles(p.x, p.y);
  if (hit) {
    resizing      = true;
    resizeAnnIndex = hit.annIndex;
    resizeHandle  = hit.handle;
    resizeOrigAnn = Object.assign({}, annotations[hit.annIndex]);
    e.preventDefault();
    startDrag();
    return;
  }

  if (!selectedClass) { alert('Select a class first.'); return; }
  drawing = true;
  startX = p.x; startY = p.y; curX = p.x; curY = p.y;
  e.preventDefault();
  startDrag();
});

canvas.addEventListener('mousemove', e => {
  if (drawing || resizing) return; // handled by document listeners during drag
  const p = clampedPoint(e);
  const hit = hitTestHandles(p.x, p.y);
  if (hit) {
    canvas.style.cursor = HANDLE_CURSORS[hit.handle];
    hoveredAnnIndex = hit.annIndex;
  } else {
    hoveredAnnIndex = hitTestAnnotation(p.x, p.y);
    canvas.style.cursor = 'crosshair';
  }
  drawAll();
});

canvas.addEventListener('mouseleave', () => {
  if (!drawing && !resizing) {
    hoveredAnnIndex = null;
    canvas.style.cursor = 'crosshair';
    drawAll();
  }
});

// Attaches document-level drag listeners so the mouse can leave the canvas
// during drawing or resizing without dropping the operation.
function startDrag() {
  const onMove = e => {
    const p = clampedPoint(e);
    if (resizing)     applyResize(resizeAnnIndex, resizeHandle, p.x, p.y);
    else if (drawing) { curX = p.x; curY = p.y; }
    drawAll();
  };

  const onUp = e => {
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup',   onUp);

    const p = clampedPoint(e);

    if (resizing) {
      applyResize(resizeAnnIndex, resizeHandle, p.x, p.y);
      resizing = false; resizeAnnIndex = null; resizeHandle = null; resizeOrigAnn = null;
      markDirty();
    } else if (drawing) {
      drawing = false;
      curX = p.x; curY = p.y;
      const w = Math.abs(curX - startX), h = Math.abs(curY - startY);
      if (w >= 5 && h >= 5) {
        const cw = canvas.width, ch = canvas.height;
        annotations.push({
          id: null,
          class_id:   selectedClass.id,
          class_name: selectedClass.name,
          color:      selectedClass.color,
          x:      (Math.min(startX, curX) + w / 2) / cw,
          y:      (Math.min(startY, curY) + h / 2) / ch,
          width:  w / cw,
          height: h / ch,
        });
        markDirty();
        renderAnnotationList();
      }
    }

    canvas.style.cursor = 'crosshair';
    drawAll();
  };

  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup',   onUp);
}

// ── Canvas rendering ──────────────────────────────────────────────────────────

function drawAll() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const cw = canvas.width, ch = canvas.height;
  const showHandlesFor = resizing ? resizeAnnIndex : hoveredAnnIndex;

  annotations.forEach((ann, i) => {
    const px = (ann.x - ann.width  / 2) * cw;
    const py = (ann.y - ann.height / 2) * ch;
    const pw = ann.width  * cw;
    const ph = ann.height * ch;

    ctx.fillStyle = ann.color + '33';
    ctx.fillRect(px, py, pw, ph);

    ctx.strokeStyle = ann.color;
    ctx.lineWidth = i === showHandlesFor ? 2.5 : 2;
    ctx.setLineDash([]);
    ctx.strokeRect(px, py, pw, ph);

    // Label: draw below box when too close to top edge
    const labelY = py >= 18 ? py : py + ph + 16;
    ctx.font = 'bold 12px system-ui';
    const tw = ctx.measureText(ann.class_name).width;
    ctx.fillStyle = ann.color;
    ctx.fillRect(px, labelY - 16, tw + 6, 16);
    ctx.fillStyle = '#fff';
    ctx.fillText(ann.class_name, px + 3, labelY - 3);

    // Resize handles (shown on hover or while resizing)
    if (i === showHandlesFor) {
      const handles = getHandlePositions(ann);
      for (const pos of Object.values(handles)) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, HANDLE_RADIUS, 0, Math.PI * 2);
        ctx.fillStyle = '#fff';
        ctx.fill();
        ctx.strokeStyle = ann.color;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
  });

  // In-progress draw rect
  if (drawing && selectedClass) {
    const x = Math.min(startX, curX), y = Math.min(startY, curY);
    const w = Math.abs(curX - startX), h = Math.abs(curY - startY);
    ctx.strokeStyle = selectedClass.color;
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
  }
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
  markDirty();
  drawAll();
  renderAnnotationList();
}

function clearAll() {
  if (annotations.length === 0) return;
  if (!confirm('Clear all annotations?')) return;
  annotations = [];
  markDirty();
  drawAll();
  renderAnnotationList();
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
    markClean();
    const status = document.createElement('span');
    status.textContent = ' Saved ✓';
    status.style.cssText = 'color:#10b981; margin-left:0.5rem';
    const saveBtn = document.querySelector('[onclick="saveAnnotations()"]');
    saveBtn.after(status);
    setTimeout(() => status.remove(), 2000);
    return true;
  } else {
    alert('Save failed: ' + resp.statusText);
    return false;
  }
}

async function saveAndNext() {
  const ok = await saveAnnotations();
  if (ok && NEXT_FRAME_ID) {
    window.location.href = `/projects/${PROJECT_ID}/label/${NEXT_FRAME_ID}`;
  }
}

async function markAsNegative() {
  if (labelStatus === 'negative' && annotations.length === 0) {
    // Already negative with no new annotations — undo back to unlabeled
    const resp = await fetch(`/projects/${PROJECT_ID}/frames/${FRAME_ID}/annotations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ annotations: [] }),
    });
    if (resp.ok) {
      labelStatus = 'unlabeled';
      markClean();
      updateNegativeBanner();
    }
    return;
  }

  const resp = await fetch(`/projects/${PROJECT_ID}/frames/${FRAME_ID}/mark_negative`, {
    method: 'POST',
  });

  if (resp.ok) {
    annotations = [];
    labelStatus = 'negative';
    markClean();
    drawAll();
    renderAnnotationList();
    updateNegativeBanner();

    const status = document.createElement('span');
    status.textContent = ' Saved ✓';
    status.style.cssText = 'color:#10b981; margin-left:0.5rem';
    const btn = document.getElementById('mark-negative-btn');
    btn.after(status);
    setTimeout(() => status.remove(), 2000);
  } else {
    alert('Failed to mark as negative: ' + resp.statusText);
  }
}

// ── Utility ───────────────────────────────────────────────────────────────────

function escapeHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Start ─────────────────────────────────────────────────────────────────────
init();
