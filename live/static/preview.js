// The rig preview, shared by the show panel (/) and the game page (/game):
// the canvas, the orbit camera, the geometry and the one WebSocket.  A page
// calls Preview.init(canvas, {...}) once and gets status, settings and game
// messages back through callbacks; it never touches the socket itself.
'use strict';

const Preview = (() => {
  let canvas = null, ctx = null;
  let geometry = null;      // {xy, size, count}
  let colours = null;       // Uint8Array, 3 bytes per dot
  let generation = -1;
  let dirty = true;
  let drawn = 0, drawnAt = performance.now();
  let hooks = {};
  const camera = { yaw: 28, pitch: 14, distance: 1.6, aspect: 16 / 9 };

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

  // Colour strings are the one real allocation in the draw loop -- one per
  // lit dot per frame, ~130 000 a second at 40 Hz.  A frame only ever uses
  // the colours in the current palette ramp, so caching them is nearly a
  // total hit.
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
    // Only repaint when something changed.  The feed runs at the engine's
    // frame rate, so painting on every animation frame would redraw the
    // same pixels.
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
    if (now - drawnAt >= 1000 && hooks.drawPill) {
      hooks.drawPill.textContent =
        `${Math.round(drawn * 1000 / (now - drawnAt))} fps preview`;
      drawn = 0; drawnAt = now;
    }
  }

  // -- camera: drag to orbit, wheel to zoom ---------------------------------

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

  function bindCamera() {
    canvas.addEventListener('pointerdown', (e) => {
      // A page may keep touch for itself (the game page steers by swipe).
      if (hooks.touchOrbit === false && e.pointerType === 'touch') return;
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
  }

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

  // -- the live link ---------------------------------------------------------

  function connect() {
    const link = hooks.link;
    const ws = new WebSocket(`ws://${location.host}/ws`);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      if (link) { link.textContent = 'live'; link.className = 'pill good'; }
    };
    ws.onclose = () => {
      if (link) { link.textContent = 'reconnecting'; link.className = 'pill bad'; }
      // The daemon must survive the browser going away and vice versa, so
      // this just keeps trying rather than needing anything restarted.
      setTimeout(connect, 1000);
    };
    ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        const body = JSON.parse(event.data);
        if (body.game) {
          if (hooks.onGame) hooks.onGame(body.game);
          return;
        }
        if (hooks.onStatus) hooks.onStatus(body);
        if (body.generation !== generation) loadGeometry();
        return;
      }
      colours = new Uint8Array(event.data);
      dirty = true;
    };
  }

  async function init(element, options) {
    canvas = element;
    ctx = canvas.getContext('2d', { alpha: false });
    hooks = options || {};
    window.addEventListener('resize', resize);
    bindCamera();
    await loadGeometry();
    resize();
    draw();
    connect();
  }

  return { init, camera };
})();
