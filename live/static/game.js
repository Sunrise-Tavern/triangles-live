// The game page: the rig preview on top, the controls underneath.  Steering
// goes to POST /api/game; the rig knobs (game mode, speed, bounce) are plain
// settings.  State comes back over the shared WebSocket the moment the
// snake moves, so the board here and the rig never disagree for long.
'use strict';

let board = null;         // {game, rows, cols, cells, walls?, free?, games}
let state = null;         // the last snapshot from the engine
let settings = {};
let dragging = null;
let fetchingBoard = false;

// --------------------------------------------------------------------------
// Input
// --------------------------------------------------------------------------

async function send(action, dir) {
  const res = await fetch('/api/game', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(dir ? { action, dir } : { action }),
  });
  if (res.ok) applyState((await res.json()).state);
  else console.warn('game input rejected:', await res.text());
}

async function patch(body) {
  const res = await fetch('/api/settings', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) console.warn('settings rejected:', await res.text());
}

const KEYS = {
  ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right',
  w: 'up', s: 'down', a: 'left', d: 'right',
  W: 'up', S: 'down', A: 'left', D: 'right',
};

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' && e.target.type !== 'checkbox'
      && e.target.type !== 'range') return;
  const dir = KEYS[e.key];
  if (dir) {
    e.preventDefault();
    flash(dir);
    send('turn', dir);
  } else if (e.key === ' ') {
    e.preventDefault();
    send('toggle');
  } else if (e.key === 'r' || e.key === 'R') {
    send('reset');
  } else if (e.key === 'g' || e.key === 'G') {
    patch({ game_mode: !settings.game_mode });
  }
});

function flash(dir) {
  const button = document.querySelector(`#pad button[data-dir="${dir}"]`);
  if (!button) return;
  button.classList.add('held');
  setTimeout(() => button.classList.remove('held'), 120);
}

for (const button of document.querySelectorAll('#pad button[data-dir]')) {
  // pointerdown, not click: a tap on a phone should steer on the way down.
  button.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    send('turn', button.dataset.dir);
  });
}
document.getElementById('go').onclick = () => send('toggle');
document.getElementById('reset').onclick = () => send('reset');

// Swipe to steer on the preview: the natural phone control, and one that
// does not need a thumb to find a button mid-game.  A finger that swipes
// steers; a mouse still orbits (preview.js leaves touch alone here).
(function swipes() {
  const stage = document.getElementById('stage');
  let start = null;
  stage.addEventListener('pointerdown', (e) => {
    if (e.pointerType !== 'touch') return;
    start = { x: e.clientX, y: e.clientY, at: performance.now() };
  });
  stage.addEventListener('pointerup', (e) => {
    if (!start || e.pointerType !== 'touch') return;
    const dx = e.clientX - start.x, dy = e.clientY - start.y;
    const quick = performance.now() - start.at < 600;
    start = null;
    if (Math.max(Math.abs(dx), Math.abs(dy)) < 24 || !quick) return;
    const dir = Math.abs(dx) > Math.abs(dy)
      ? (dx > 0 ? 'right' : 'left') : (dy > 0 ? 'down' : 'up');
    flash(dir);
    send('turn', dir);
  });
  stage.addEventListener('pointercancel', () => { start = null; });
})();

// --------------------------------------------------------------------------
// Rig knobs: game mode, speed, bounce -- ordinary settings
// --------------------------------------------------------------------------

const modeBox = document.getElementById('game-mode');
modeBox.onchange = () => patch({ game_mode: modeBox.checked });
const kindBox = document.getElementById('game-kind');
kindBox.onchange = () => patch({ game: kindBox.value });

function slider(id, key, valueId, digits) {
  const input = document.getElementById(id);
  const value = document.getElementById(valueId);
  const show = (v) => { value.textContent = Number(v).toFixed(digits); };
  input.oninput = () => { show(input.value); patch({ [key]: +input.value }); };
  input.onpointerdown = () => { dragging = key; };
  input.onpointerup = input.onblur = () => { dragging = null; };
  return { set: (v) => { input.value = v; show(v); } };
}
const widgets = {
  game_speed: slider('game-speed', 'game_speed', 'v-speed', 1),
  game_bounce: slider('game-bounce', 'game_bounce', 'v-bounce', 2),
};

function applySettings(next) {
  settings = next;
  for (const [key, widget] of Object.entries(widgets)) {
    if (key !== dragging) widget.set(next[key]);
  }
  modeBox.checked = !!next.game_mode;
  document.getElementById('mode').classList.toggle('on', !!next.game_mode);
  if (next.game && kindBox.value !== next.game) kindBox.value = next.game;
  describe();
}

async function loadBoard() {
  // The board is static per game; it is fetched once, and again whenever
  // the engine reports a state from a different game than we have.
  if (fetchingBoard) return;
  fetchingBoard = true;
  try {
    const game = await (await fetch('/api/game')).json();
    board = game.board;
    kindBox.innerHTML = board.games
      .map((g) => `<option value="${g}">${g === 'pacman' ? 'Pac-Man' : g}</option>`)
      .join('');
    kindBox.value = board.game;
    applyState(game.state);
  } finally {
    fetchingBoard = false;
  }
}

// --------------------------------------------------------------------------
// The board, drawn flat
// --------------------------------------------------------------------------

const boardCanvas = document.getElementById('board');
const bctx = boardCanvas.getContext('2d');

let animating = false;
function drawBoard() {
  animating = false;
  const w = boardCanvas.width, h = boardCanvas.height;
  bctx.fillStyle = '#000';
  bctx.fillRect(0, 0, w, h);
  if (!board) return;
  const cell = Math.min(w / board.cols, h / board.rows);
  const ox = (w - cell * board.cols) / 2, oy = (h - cell * board.rows) / 2;
  const box = (r, c, colour, inset) => {
    bctx.fillStyle = colour;
    bctx.fillRect(ox + c * cell + inset, oy + r * cell + inset,
      cell - 2 * inset, cell - 2 * inset);
  };
  for (const [r, c] of board.cells) box(r, c, '#1a2028', 1);
  if (!state || state.game !== board.game) return;
  if (board.game === 'pacman') { drawPacman(box, cell); return; }
  const n = state.body.length;
  state.body.forEach(([r, c], i) => {
    const level = 1 - 0.6 * (i / Math.max(1, n - 1));
    const colour = state.alive
      ? `hsl(135, ${i === 0 ? 45 : 90}%, ${Math.round(30 + 35 * level)}%)`
      : `hsl(0, 90%, ${Math.round(12 + 20 * level)}%)`;
    box(r, c, colour, 1);
  });
  if (state.food && state.alive) {
    const [r, c] = state.food;
    box(r, c, 'hsl(315, 90%, 60%)', Math.max(1, cell * 0.2));
  }
}

const GHOSTS = ['#ff2a18', '#ff73bf', '#28f2ff', '#ffa628'];

function drawPacman(box, cell) {
  for (const [r, c] of board.walls) box(r, c, '#0d1f5c', 0.5);
  const dot = Math.max(1, cell * 0.33), power = Math.max(1, cell * 0.18);
  board.free.forEach(([r, c], i) => {
    const mark = state.pellets[i];
    if (mark === '1') box(r, c, '#8a7a5a', dot);
    else if (mark === '2') box(r, c, '#ffd9a0', power);
  });
  const w = boardCanvas.width, h = boardCanvas.height;
  const ox = (w - cell * board.cols) / 2, oy = (h - cell * board.rows) / 2;
  const centre = ([r, c]) => [ox + (c + 0.5) * cell, oy + (r + 0.5) * cell];
  state.ghosts.forEach((g, i) => {
    const colour = g.mode === 'eyes' ? '#4a4a7a'
      : (g.mode === 'fright' ? '#2a2af0' : GHOSTS[i % GHOSTS.length]);
    const [x, y] = centre(g.cell);
    bctx.fillStyle = colour;
    bctx.beginPath();
    bctx.arc(x, y, cell * (g.mode === 'eyes' ? 0.55 : 0.85), 0, Math.PI * 2);
    bctx.fill();
  });
  // Pac-Man as on the rig: nearly three cells across, mouth toward the way
  // he faces, chomping while he moves (the phase is local -- the engine
  // only reports cells, the mouth is decoration).
  const [x, y] = centre(state.pac);
  const heading = { up: -Math.PI / 2, down: Math.PI / 2, left: Math.PI, right: 0 }[state.facing || 'left'];
  const moving = state.running && state.direction && !state.over;
  const mouth = state.over ? Math.PI * 0.9
    : (moving ? 0.12 + 0.68 * Math.abs(Math.sin(performance.now() / 250)) : 0.4);
  bctx.fillStyle = state.over ? '#7a1000' : '#ffd800';
  bctx.beginPath();
  bctx.moveTo(x, y);
  bctx.arc(x, y, cell * 1.4, heading + mouth, heading - mouth + Math.PI * 2);
  bctx.closePath();
  bctx.fill();
  if (moving && !animating) { animating = true; requestAnimationFrame(drawBoard); }
}

function describe() {
  const el = document.getElementById('state');
  el.className = '';
  if (!state) { el.textContent = ''; return; }
  if (!settings.game_mode) {
    el.textContent = 'game mode is off — the rig is still running the show';
  } else if (state.game === 'pacman') {
    if (state.over) {
      el.textContent = `game over at ${state.score} — R or New game`;
      el.className = 'bad';
    } else if (!state.running) {
      el.textContent = state.steps ? 'paused — space or GO' : 'press an arrow to start';
    } else {
      el.textContent = `level ${state.level} · ${state.pellets_left} pellets left`
        + (state.fright ? ' · ghosts are edible!' : '');
      el.className = state.fright ? 'good' : '';
    }
  } else if (state.won) {
    el.textContent = 'you filled the triangle — R for a new game';
    el.className = 'good';
  } else if (!state.alive) {
    el.textContent = `game over at ${state.score} — R or New game`;
    el.className = 'bad';
  } else if (!state.running) {
    el.textContent = state.steps ? 'paused — space or GO' : 'press an arrow to start';
  } else {
    el.textContent = 'playing';
    el.className = 'good';
  }
}

function applyState(next) {
  state = next;
  if (board && next.game !== board.game) { loadBoard(); return; }
  document.getElementById('g-score').textContent = next.score;
  document.getElementById('g-best').textContent = next.best;
  const pacman = next.game === 'pacman';
  document.getElementById('g-third').textContent = pacman ? next.lives : next.length;
  document.getElementById('g-third-label').textContent = pacman ? 'lives' : 'length';
  drawBoard();
  describe();
}

function setStatus(status) {
  document.getElementById('target').textContent = status.target;
  document.getElementById('why').textContent = status.audio
    ? (status.free_running ? 'audio: no beats detected' : `audio: level ${status.level.toFixed(2)}`)
    : 'no audio input — the food pulses on the scripted tempo';
}

(async function start() {
  applySettings(await (await fetch('/api/settings')).json());
  await loadBoard();
  await Preview.init(document.getElementById('preview'), {
    link: document.getElementById('link'),
    drawPill: document.getElementById('draw'),
    onStatus: (body) => { setStatus(body.status); applySettings(body.settings); },
    onGame: applyState,
    touchOrbit: false,          // a finger on the preview steers instead
  });
})();
