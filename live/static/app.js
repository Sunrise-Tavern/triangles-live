// Controls and readout for the live engine.  No framework, no build step:
// this file is served as-is and runs on whatever browser is at the rig.  The
// preview itself lives in preview.js, shared with the game page.
'use strict';

let dragging = null;      // control the operator is holding, so status can't stomp it

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

(async function start() {
  const schema = await (await fetch('/api/schema')).json();
  buildControls(schema.settings);
  if (schema.unaddressed.length) {
    document.getElementById('warning').textContent =
      `${schema.unaddressed.length} models sit outside every controller's ` +
      `channel space and are not reachable over DDP (${schema.unaddressed[0]} ...).`;
  }
  await refreshPresets();
  applySettings(await (await fetch('/api/settings')).json());
  await Preview.init(document.getElementById('preview'), {
    link: document.getElementById('link'),
    drawPill: document.getElementById('draw'),
    onStatus: (body) => { setStatus(body.status); applySettings(body.settings); },
  });
})();
