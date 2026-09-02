// Preview + controls for the live engine.  No framework, no build step: this
// file is served as-is and runs on whatever browser is at the rig.
'use strict';

const canvas = document.getElementById('preview');
const ctx = canvas.getContext('2d', { alpha: false });

let geometry = null;      // {xy, size, count, generation}
let colours = null;       // Uint8Array, 3 bytes per dot
let generation = -1;
let dragging = null;      // control the operator is holding, so status can't stomp it
const camera = { yaw: 28, pitch: 14, distance: 1.6, aspect: 16 / 9 };

// --------------------------------------------------------------------------
// Preview
// --------------------------------------------------------------------------

function resize() {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  dirty = true;
  const aspect = rect.width / Math.max(1, rect.height);
  if (Math.abs(aspect - camera.aspect) > 0.02) {
    camera.aspect = aspect;
    postCamera();
  }
}
window.addEventListener('resize', resize);

let dirty = true;
let drawn = 0, drawnAt = performance.now();

// Colour strings are the one real allocation in the draw loop -- one per lit
// dot per frame, ~130 000 a second at 40 Hz.  A frame only ever uses the
// colours in the current palette ramp, so caching them is nearly a total hit.
const styles = new Map();
function styleFor(r, g, b) {
  const key = (r << 16) | (g << 8) | b;
  let s = styles.get(key);
  if (s === undefined) {
    if (styles.size > 16384) styles.clear();
    s = `rgb(${r},${g},${b})`;
    styles.set(key, s);
  }
  return s;
}

function draw() {
  requestAnimationFrame(draw);
  // Only repaint when something changed.  The feed runs at the engine's frame
  // rate, so painting on every animation frame would redraw the same pixels.
  if (!dirty) return;
  dirty = false;

  const w = canvas.width, h = canvas.height;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, w, h);
  if (!geometry || !colours) return;

  // Additive, because two lit fixtures overlapping on screen should get
  // brighter -- the same reason the engine composites effects by adding.
  ctx.globalCompositeOperation = 'lighter';
  const scale = Math.min(w, h) / 540;
  const { xy, size, count } = geometry;
  for (let i = 0; i < count; i++) {
    const r = colours[i * 3], g = colours[i * 3 + 1], b = colours[i * 3 + 2];
    if (r + g + b < 3) continue;                  // skip the dark majority
    const d = Math.max(1, size[i] * 2.6 * scale);
    ctx.fillStyle = styleFor(r, g, b);
    ctx.fillRect(xy[i * 2] * w - d / 2, xy[i * 2 + 1] * h - d / 2, d, d);
  }
  ctx.globalCompositeOperation = 'source-over';

  drawn++;
  const now = performance.now();
  if (now - drawnAt >= 1000) {
    document.getElementById('draw').textContent =
      `${Math.round(drawn * 1000 / (now - drawnAt))} fps preview`;
    drawn = 0; drawnAt = now;
  }
}

// --------------------------------------------------------------------------
// Camera: drag to orbit, wheel to zoom
// --------------------------------------------------------------------------

let pending = null;
function postCamera() {
  clearTimeout(pending);
  pending = setTimeout(async () => {
    const res = await fetch('/api/camera', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(camera),
    });
    const body = await res.json();
    Object.assign(camera, body.camera);
    await loadGeometry();
  }, 60);
}

canvas.addEventListener('pointerdown', (e) => {
  canvas.setPointerCapture(e.pointerId);
  canvas.dataset.drag = `${e.clientX},${e.clientY}`;
});
canvas.addEventListener('pointerup', () => { delete canvas.dataset.drag; });
canvas.addEventListener('pointermove', (e) => {
  if (!canvas.dataset.drag) return;
  const [x, y] = canvas.dataset.drag.split(',').map(Number);
  camera.yaw += (e.clientX - x) * 0.35;
  camera.pitch = Math.max(-80, Math.min(80, camera.pitch + (e.clientY - y) * 0.25));
  canvas.dataset.drag = `${e.clientX},${e.clientY}`;
  postCamera();
});
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  camera.distance = Math.max(0.4, Math.min(6, camera.distance * (1 + e.deltaY * 0.001)));
  postCamera();
}, { passive: false });

async function loadGeometry() {
  const body = await (await fetch('/api/geometry')).json();
  geometry = {
    xy: Float32Array.from(body.xy),
    size: Float32Array.from(body.size),
    count: body.count,
  };
  generation = body.generation;
  dirty = true;
}

// --------------------------------------------------------------------------
// Controls, generated from the server's schema
// --------------------------------------------------------------------------

const widgets = {};

function buildControls(schema) {
  const host = document.getElementById('controls');
  host.innerHTML = '';
  for (const item of schema) {
    if (item.key === 'blackout') continue;         // has its own big button
    const box = document.createElement('div');
    box.className = 'control';

    if (item.kind === 'bool') {
      box.classList.add('toggle');
      box.innerHTML = `<label style="margin:0">${item.label}</label>`;
      const input = Object.assign(document.createElement('input'),
        { type: 'checkbox' });
      input.onchange = () => patch({ [item.key]: input.checked });
      box.appendChild(input);
      widgets[item.key] = { set: (v) => { input.checked = v; } };
    } else if (item.kind === 'choice') {
      box.innerHTML = `<label>${item.label}</label>`;
      const select = document.createElement('select');
      select.innerHTML = item.options
        .map((o) => `<option value="${o}">${o}</option>`).join('');
      select.onchange = () => patch({ [item.key]: select.value });
      box.appendChild(select);
      widgets[item.key] = { set: (v) => { select.value = v; } };
    } else {
      const label = document.createElement('label');
      const value = document.createElement('span');
      label.textContent = item.label;
      label.appendChild(value);
      const input = Object.assign(document.createElement('input'), {
        type: 'range', min: item.min, max: item.max, step: item.step,
      });
      const show = (v) => { value.textContent = Number(v).toFixed(
        item.step < 0.1 ? 2 : (item.step < 1 ? 2 : 0)); };
      input.oninput = () => { show(input.value); patch({ [item.key]: +input.value }); };
      input.onpointerdown = () => { dragging = item.key; };
      input.onpointerup = input.onblur = () => { dragging = null; };
      box.append(label, input);
      widgets[item.key] = { set: (v) => { input.value = v; show(v); } };
    }
    host.appendChild(box);
  }
}

async function patch(body) {
  const res = await fetch('/api/settings', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) console.warn('settings rejected:', await res.text());
}

function applySettings(settings) {
  for (const [key, value] of Object.entries(settings)) {
    if (key === dragging) continue;                // never fight the operator
    if (widgets[key]) widgets[key].set(value);
  }
  const button = document.getElementById('blackout');
  button.classList.toggle('on', !!settings.blackout);
}

document.getElementById('blackout').onclick = () => {
  const on = document.getElementById('blackout').classList.contains('on');
  patch({ blackout: !on });
};

// --------------------------------------------------------------------------
// Presets
// --------------------------------------------------------------------------

async function refreshPresets(names) {
  if (!names) names = (await (await fetch('/api/presets')).json()).presets;
  const list = document.getElementById('preset-list');
  list.innerHTML = '';
  for (const name of names) {
    const li = document.createElement('li');
    const label = document.createElement('span');
    label.textContent = name;
    const load = Object.assign(document.createElement('button'),
      { textContent: 'Load' });
    load.onclick = async () => {
      const res = await fetch(`/api/presets/${encodeURIComponent(name)}/load`,
        { method: 'POST' });
      if (res.ok) applySettings((await res.json()).settings);
    };
    const drop = Object.assign(document.createElement('button'),
      { textContent: '×', title: 'delete' });
    drop.onclick = async () => {
      const res = await fetch(`/api/presets/${encodeURIComponent(name)}`,
        { method: 'DELETE' });
      if (res.ok) refreshPresets((await res.json()).presets);
    };
    li.append(label, load, drop);
    list.appendChild(li);
  }
}

document.getElementById('preset-save').onclick = async () => {
  const field = document.getElementById('preset-name');
  const name = field.value.trim();
  if (!name) { field.focus(); return; }
  const res = await fetch(`/api/presets/${encodeURIComponent(name)}`,
    { method: 'PUT' });
  if (res.ok) { field.value = ''; refreshPresets((await res.json()).presets); }
  else alert(await res.text());
};

// --------------------------------------------------------------------------
// Live link
// --------------------------------------------------------------------------

function setStatus(status) {
  const put = (id, v) => { document.getElementById(id).textContent = v; };
  put('s-fps', status.fps.toFixed(1));
  put('s-render', status.render_ms.toFixed(2));
  put('s-scene', status.scene);
  put('s-pattern', status.pattern);
  put('s-bpm', status.bpm.toFixed(0));
  put('s-late', status.late_frames);
  if (status.audio) {
    put('s-conf', status.confidence.toFixed(2));
    put('s-barconf', status.bar_confidence.toFixed(2));
    put('s-bar', status.bar);
    // The audio-drive gauges: what the low and high end are reading as, and
    // how fast the material is running against the clock as a result.
    put('s-bass', status.bass_drive.toFixed(2));
    put('s-air', status.air.toFixed(2));
    put('s-rate', `${status.drive_rate.toFixed(2)}x`);
    document.querySelector('#bar i').style.width =
      `${(1 - status.bar_phase) * 100}%`;
    document.getElementById('why').textContent = status.free_running
      ? 'beat clock free-running — no beats detected'
      : (status.reason ? `state: ${status.reason}` : '');
  } else {
    put('s-conf', '—'); put('s-barconf', '—'); put('s-bar', '—');
    put('s-bass', '—'); put('s-air', '—'); put('s-rate', '—');
    document.getElementById('why').textContent = 'scripted show — no audio input';
  }
  document.getElementById('target').textContent = status.target;
  document.querySelector('#beat i').style.width =
    `${(1 - status.beat_phase) * 100}%`;
}

function connect() {
  const link = document.getElementById('link');
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => { link.textContent = 'live'; link.className = 'pill good'; };
  ws.onclose = () => {
    link.textContent = 'reconnecting'; link.className = 'pill bad';
    // The daemon must survive the browser going away and vice versa, so this
    // just keeps trying rather than needing anything restarted.
    setTimeout(connect, 1000);
  };
  ws.onmessage = (event) => {
    if (typeof event.data === 'string') {
      const body = JSON.parse(event.data);
      setStatus(body.status);
      applySettings(body.settings);
      if (body.generation !== generation) loadGeometry();
      return;
    }
    colours = new Uint8Array(event.data);
    dirty = true;
  };
}

(async function start() {
  const schema = await (await fetch('/api/schema')).json();
  buildControls(schema.settings);
  if (schema.unaddressed.length) {
    document.getElementById('warning').textContent =
      `${schema.unaddressed.length} models sit outside every controller's ` +
      `channel space and are not reachable over DDP (${schema.unaddressed[0]} ...).`;
  }
  await loadGeometry();
  await refreshPresets();
  applySettings(await (await fetch('/api/settings')).json());
  resize();
  draw();
  connect();
})();
