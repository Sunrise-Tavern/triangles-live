"""Game mode: the nets become a screen, the corridor a scoreboard.

With ``game_mode`` on, the engine stops rendering the show and hands the
canvas to :class:`Game` every frame instead.  Nothing about the pipeline
below the canvas changes -- the same gather, the same DDP packets, the same
preview -- so a game is played on the rig with exactly the tools an effect
has: paint float RGB into the buffers, in source order.

**Boards are grids of square cells over a surface.**  A game wants square
cells, not the 465-node triangular lattice each net actually is.  So a
surface -- the big triangle (``Canvas.big_geo``) or the whole net array
(``Canvas.all_geo``) -- is cut into ``columns`` columns and however many rows
keep the cells square in the surface's own plane, and every real pixel is
binned into a cell.  Cells on the sloping edges that catch only a sliver of
pixels are *void*, and so is anything that is not part of a decently sized
4-connected region, because a cell reachable only diagonally is a trap that
reads as a bug.  Measured on the 2026-08-30 layout:

* the big triangle at 14 columns: 12 rows, 94 cells of ~20 pixels;
* the whole array at 48 columns: 15 rows, 212 cells of ~14 pixels, in four
  islands -- the three small nets and the big triangle.

**Tunnels.**  On the whole array the islands are separated by the void
between the triangles.  Stepping off an island's edge into void carries on
in the same row or column and comes out on the next island, the way
Pac-Man's side tunnels do -- so Pac-Man leaves one triangle and appears on
the next, and the maze is one maze across all seven nets.  Void is not a
wall: walls are the maze's own, placed inside the islands, and they stop a
step dead.

**Snake** plays on the big triangle; **Pac-Man** on the whole array, each
small net releasing one ghost.  The ``game`` knob picks; ``game_speed`` is
cells a second for the player, and ``game_bounce`` how much the measured
kick bounces the board's brightness.  The corridor keeps the score in both:
one arch per bite for the snake, pellets eaten as a bar for Pac-Man, a white
pulse down the tunnel for a bite or a ghost, red for a death.

**Timing is the engine's clock.**  Everything moves on show time ``t``, so a
speed means exactly that at 40 fps or 60, and a paused game resumes where it
was rather than racing to catch up.

Input arrives from the web thread; rendering happens on the render thread.
Everything that touches a game goes through one lock, which is cheap
because a step is a handful of tuple operations.
"""

from __future__ import annotations

import random
import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import palette as pal
from .frame import Canvas

#: Cells across the big triangle's base.  Rows follow from the aspect.
COLUMNS = 14
#: Cells across the whole net array, for games that use every triangle.
COLUMNS_ALL = 48
#: A cell with fewer real pixels than this is void: it sits on a sloping
#: edge and would light as a lone dot or two.
MIN_PIXELS = 5
#: A 4-connected region smaller than this is a stray, not an island.
MIN_REGION = 6
#: Snake length at the start.
START_LENGTH = 3

DIRECTIONS: dict[str, tuple[int, int]] = {
    "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
}
_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}

Cell = tuple[int, int]


class Board:
    """A surface of the rig as a grid of square cells.

    ``cell`` is a ``(k, nodes)`` int array over the surface's buffer rows,
    the cell index of each pixel or -1 where it is void or padding, so
    painting the board is one fancy-index gather from a colour table.
    ``regions`` are the 4-connected islands, largest first, and
    :meth:`neighbour` is how a game moves: one step, or a tunnel across the
    void to the next island.
    """

    def __init__(self, canvas: Canvas, surface: str = "big",
                 columns: int | None = None,
                 min_pixels: int = MIN_PIXELS) -> None:
        if surface == "big":
            geo, self.slice = canvas.big_geo, canvas.big
            aspect = canvas.big_aspect
            columns = columns or COLUMNS
        elif surface == "all":
            geo, self.slice = canvas.all_geo, slice(0, len(canvas.net_names))
            aspect = canvas.all_geo.aspect
            columns = columns or COLUMNS_ALL
        else:
            raise ValueError(f"no such surface {surface!r}")
        self.surface = surface
        mask = canvas.net_mask[self.slice]
        self.cols = int(columns)
        # Square cells in the surface's own plane, so a move up covers the
        # same distance as a move across.
        self.rows = max(2, int(round(self.cols / max(aspect, 1e-6))))
        cx = np.clip((geo.x * self.cols).astype(np.int32), 0, self.cols - 1)
        cy = np.clip((geo.y * self.rows).astype(np.int32), 0, self.rows - 1)
        raw = cy * self.cols + cx
        counts = np.bincount(raw[mask], minlength=self.rows * self.cols)
        playable = (counts >= min_pixels).reshape(self.rows, self.cols)
        self.regions = [r for r in self._components(playable)
                        if len(r) >= MIN_REGION]
        self.regions.sort(key=len, reverse=True)
        playable = np.zeros_like(playable)
        self.region_of: dict[Cell, int] = {}
        for i, region in enumerate(self.regions):
            for r, c in region:
                playable[r, c] = True
                self.region_of[(r, c)] = i
        self.playable = playable            # (rows, cols) bool
        #: Every real pixel's position in cell units (x across, y down),
        #: and which pixels are real: what a sprite is drawn with, so it
        #: can be any size and shape and not a stack of cells.
        self.px = (geo.x * self.cols).astype(np.float32)
        self.py = (geo.y * self.rows).astype(np.float32)
        self.mask = mask
        keep = mask & playable.reshape(-1)[raw]
        self.cell = np.where(keep, raw, -1).astype(np.int32)
        self.pixels = counts                # per cell, for diagnostics
        self.cells: list[Cell] = [
            (int(r), int(c)) for r, c in zip(*np.nonzero(playable))]
        self.count = len(self.cells)
        self.covered = float(keep.sum()) / max(1, int(mask.sum()))
        self._lookup = set(self.cells)

    @staticmethod
    def _components(playable: np.ndarray) -> list[list[Cell]]:
        """4-connected regions of a boolean grid."""
        rows, cols = playable.shape
        seen = np.zeros_like(playable)
        out: list[list[Cell]] = []
        for start in zip(*np.nonzero(playable)):
            start = (int(start[0]), int(start[1]))
            if seen[start]:
                continue
            region = [start]
            seen[start] = True
            queue = deque([start])
            while queue:
                r, c = queue.popleft()
                for dr, dc in DIRECTIONS.values():
                    n = (r + dr, c + dc)
                    if (0 <= n[0] < rows and 0 <= n[1] < cols
                            and playable[n] and not seen[n]):
                        seen[n] = True
                        region.append(n)
                        queue.append(n)
            out.append(region)
        return out

    def __contains__(self, cell: Cell) -> bool:
        return cell in self._lookup

    def index(self, cell: Cell) -> int:
        return cell[0] * self.cols + cell[1]

    def adjacent(self, cell: Cell) -> list[Cell]:
        """Plain 4-neighbours on the same island; no tunnels."""
        r, c = cell
        return [(r + dr, c + dc) for dr, dc in DIRECTIONS.values()
                if (r + dr, c + dc) in self]

    def neighbour(self, cell: Cell, direction: str,
                  walls: frozenset[Cell] | set[Cell] = frozenset()) -> Cell | None:
        """One step, or through the void to the next island -- or nothing.

        A wall stops the step; void is crossed until the next cell or the
        edge of the grid.  That is what makes the three small nets and the
        big triangle one board.
        """
        dr, dc = DIRECTIONS[direction]
        r, c = cell
        while True:
            r, c = r + dr, c + dc
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                return None
            n = (r, c)
            if n in walls:
                return None
            if n in self._lookup:
                return n

    def centre(self, cells: list[Cell] | None = None) -> Cell:
        """The cell nearest the centroid -- of the board or of some cells."""
        cells = cells if cells is not None else self.cells
        rows = np.array([r for r, _ in cells], dtype=np.float64)
        cols = np.array([c for _, c in cells], dtype=np.float64)
        cr, cc = rows.mean(), cols.mean()
        best = int(np.argmin((rows - cr) ** 2 + (cols - cc) ** 2))
        return cells[best]

    def run(self, cell: Cell, direction: str) -> int:
        """How many cells lie ahead of ``cell`` on its island before an edge."""
        dr, dc = DIRECTIONS[direction]
        n = 0
        r, c = cell
        while (r + dr, c + dc) in self:
            r, c = r + dr, c + dc
            n += 1
        return n

    def target(self, canvas: Canvas) -> np.ndarray:
        return canvas.nets[self.slice]

    def paint(self, target: np.ndarray, colours: np.ndarray) -> None:
        """``colours`` is ``(rows*cols, 3)``; void and padding stay black."""
        table = np.vstack([colours.astype(np.float32),
                           np.zeros((1, 3), dtype=np.float32)])
        index = np.where(self.cell < 0, len(colours), self.cell)
        target[:] = table[index]

    def sprite(self, target: np.ndarray, at: tuple[float, float], radius: float,
               colour, facing: tuple[float, float] | None = None,
               mouth: float = 0.0) -> int:
        """A disc of ``radius`` cells centred on cell ``at`` (row, column,
        fractional), painted straight onto the pixels.  ``mouth`` is the
        half-angle, radians, of a wedge left unpainted on the ``facing``
        side -- Pac-Man's mouth.  Returns how many pixels it covered."""
        y, x = at[0] + 0.5, at[1] + 0.5
        dx, dy = self.px - x, self.py - y
        inside = (dx * dx + dy * dy < radius * radius) & self.mask
        if facing is not None and mouth > 0.0:
            fx, fy = facing
            along = (dx * fx + dy * fy) / np.maximum(np.hypot(dx, dy), 1e-6)
            inside &= np.arccos(np.clip(along, -1.0, 1.0)) > mouth
        target[inside] = colour
        return int(inside.sum())

    def describe(self) -> dict:
        return {"rows": self.rows, "cols": self.cols,
                "cells": [[r, c] for r, c in self.cells]}


# --------------------------------------------------------------------------- #
# Snake
# --------------------------------------------------------------------------- #


class Snake:
    """The rules, and nothing about pixels or time.

    ``step`` is one move.  Turns queue up to two deep so a quick
    up-then-left between two steps is honoured, as players expect, and a
    reversal is ignored rather than fatal.  Food never lands on the snake.
    """

    def __init__(self, board: Board, seed: int = 7) -> None:
        self.board = board
        self.seed = int(seed)
        self.games = 0
        self.best = 0
        self.reset()

    def reset(self) -> None:
        self.games += 1
        self.rng = random.Random(self.seed * 1000 + self.games)
        head = self.board.centre()
        # Face the longest free run, and unroll the body behind the head.
        self.direction = max(DIRECTIONS, key=lambda d: self.board.run(head, d))
        dr, dc = DIRECTIONS[self.direction]
        self.body: deque[Cell] = deque([head])
        for i in range(1, START_LENGTH):
            tail = (head[0] - dr * i, head[1] - dc * i)
            if tail not in self.board:
                break
            self.body.append(tail)
        self.queue: deque[str] = deque()
        self.score = 0
        self.alive = True
        self.won = False
        self.running = False
        self.steps = 0
        self.food = self._spawn()

    def _spawn(self) -> Cell | None:
        taken = set(self.body)
        free = [c for c in self.board.cells if c not in taken]
        return self.rng.choice(free) if free else None

    @property
    def head(self) -> Cell:
        return self.body[0]

    def turn(self, direction: str) -> None:
        if direction not in DIRECTIONS:
            raise ValueError(f"no such direction {direction!r}")
        last = self.queue[-1] if self.queue else self.direction
        if direction == last or direction == _OPPOSITE[last]:
            return
        if len(self.queue) < 2:
            self.queue.append(direction)

    def step(self) -> str:
        """Advance one cell.  Returns "moved", "ate", "died", "won" or "idle"."""
        if not (self.alive and self.running) or self.won:
            return "idle"
        if self.queue:
            self.direction = self.queue.popleft()
        dr, dc = DIRECTIONS[self.direction]
        head = (self.head[0] + dr, self.head[1] + dc)
        growing = head == self.food
        # The tail moves out of the way this same step unless we are growing
        # -- so chasing your own tail is legal, as in every snake since 1976.
        occupied = set(self.body) if growing else set(list(self.body)[:-1])
        self.steps += 1
        if head not in self.board or head in occupied:
            self.alive = False
            self.running = False
            self.best = max(self.best, self.score)
            return "died"
        self.body.appendleft(head)
        if growing:
            self.score += 1
            self.best = max(self.best, self.score)
            self.food = self._spawn()
            if self.food is None:
                self.won = True
                self.running = False
                return "won"
            return "ate"
        self.body.pop()
        return "moved"

    def snapshot(self) -> dict:
        return {
            "alive": self.alive, "running": self.running, "won": self.won,
            "score": self.score, "best": self.best, "length": len(self.body),
            "direction": self.direction, "steps": self.steps,
            "games": self.games,
            "body": [[r, c] for r, c in self.body],
            "food": list(self.food) if self.food else None,
        }


class SnakeGame:
    """Snake on the big triangle: the clock, the input and the paint."""

    name = "snake"
    surface = "big"

    def __init__(self, board: Board, seed: int = 7) -> None:
        self.board = board
        self.snake = Snake(board, seed=seed)
        self._due: float | None = None
        self._ate_at = -1e9
        self._died_at = -1e9
        self._table = np.zeros((board.rows * board.cols, 3), dtype=np.float32)

    def describe(self) -> dict:
        return {"game": self.name, **self.board.describe()}

    def snapshot(self) -> dict:
        return {"game": self.name, **self.snake.snapshot()}

    def input(self, action: str, direction: str | None = None) -> None:
        snake = self.snake
        if action == "turn":
            if direction is None:
                raise ValueError("turn needs a direction")
            snake.turn(direction)
            # An arrow key is also the start button, as on every handheld.
            if snake.alive and not snake.won:
                snake.running = True
        elif action == "start":
            if snake.alive and not snake.won:
                snake.running = True
        elif action == "pause":
            snake.running = False
            self._due = None
        elif action == "toggle":
            if snake.alive and not snake.won:
                snake.running = not snake.running
                if not snake.running:
                    self._due = None
        elif action == "reset":
            snake.reset()
            self._due = None
        else:
            raise ValueError(f"no such action {action!r}")

    def advance(self, t: float, speed: float) -> bool:
        """Move the snake up to show time ``t``.  True if anything changed."""
        snake = self.snake
        if not snake.running:
            return False
        period = 1.0 / max(0.5, float(speed))
        if self._due is None or self._due < t - 2 * period:
            # Freshly started, resumed, or the render loop stalled: move
            # next period from now rather than sprinting.
            self._due = t + period
        # One step a frame at most -- 40 a second is already far past any
        # speed the knob allows.
        if t < self._due:
            return False
        outcome = snake.step()
        self._due += period
        if outcome == "ate":
            self._ate_at = t
        elif outcome == "died":
            self._died_at = t
        return True

    def _hue(self) -> float:
        # The snake's colour walks the hue circle by the golden angle every
        # few bites, so a long game is a colour journey, not a longer green.
        return (135.0 + (self.snake.score // 3) * pal.GOLDEN_ANGLE) % 360.0

    def paint(self, canvas: Canvas, t: float, gain: float,
              level: float | None, beat_phase: float) -> None:
        snake, board, table = self.snake, self.board, self._table
        body, food, alive = list(snake.body), snake.food, snake.alive
        table[:] = 0.0
        hue = self._hue()
        # The playable area, just visible, so the board reads as a board.
        table[board.playable.reshape(-1)] = pal.hsv_to_rgb(
            np.float32(hue), np.float32(0.6), np.float32(0.05))
        n = len(body)
        if n:
            # Head bright and a little white, tail dim: direction is legible
            # from across the room.
            fade = np.linspace(1.0, 0.35, n, dtype=np.float32)
            sat = np.full(n, 0.9, dtype=np.float32)
            sat[0] = 0.45
            if not alive:
                # Dead: the snake goes dark red and stays where it fell.
                rgb = pal.hsv_to_rgb(np.zeros(n, dtype=np.float32),
                                     np.ones(n, dtype=np.float32), fade * 0.5)
            else:
                rgb = pal.hsv_to_rgb(np.full(n, hue, dtype=np.float32), sat,
                                     fade)
            table[[board.index(c) for c in body]] = rgb
        if food is not None and alive:
            # Complementary, and it pulses on the beat so it can be found.
            pulse = 0.55 + 0.45 * (1.0 - beat_phase) ** 2
            table[board.index(food)] = pal.hsv_to_rgb(
                np.float32(hue + 180.0), np.float32(0.85), np.float32(pulse))
        # A bite lights the whole board for a flash; a death washes it red.
        since_ate = t - self._ate_at
        if 0.0 <= since_ate < 0.25:
            table += 0.35 * (1.0 - since_ate / 0.25)
        since_died = t - self._died_at
        if 0.0 <= since_died < 1.2:
            table[board.playable.reshape(-1), 0] += 0.6 * (1.0 - since_died / 1.2)
        table *= gain
        np.clip(table, 0.0, 1.0, out=table)
        board.paint(board.target(canvas), table)
        self._paint_meter(canvas, t, level)
        _paint_corridor(canvas, t, hue, min(snake.score, canvas.arches.shape[0])
                        / canvas.arches.shape[0], self._ate_at, self._died_at)

    def _paint_meter(self, canvas: Canvas, t: float, level: float | None) -> None:
        """The small nets: a level meter with audio, a slow breath without."""
        _, small = canvas.net_pair()
        if level is None:
            value = 0.10 + 0.08 * (0.5 + 0.5 * np.sin(t * 1.2))
        else:
            value = 0.06 + 0.74 * level
        rgb = pal.hsv_to_rgb(np.float32(self._hue()), np.float32(0.8),
                             np.float32(value))
        # Brighter toward the bottom, so the meter reads as filling up.
        y = canvas.net_y[small]
        canvas.nets[small] = rgb * (0.4 + 0.6 * y)[..., None]
        canvas.nets[small] *= canvas.net_mask[small][..., None]


def _paint_corridor(canvas: Canvas, t: float, hue: float, progress: float,
                    pulse_at: float, died_at: float) -> None:
    """Arches lit from the front by ``progress``, a white pulse down the
    tunnel after ``pulse_at``, a red wash after ``died_at``."""
    n = canvas.arches.shape[0]
    lit = np.zeros(n, dtype=np.float32)
    lit[:int(round(min(max(progress, 0.0), 1.0) * n))] = 0.55
    rgb = pal.hsv_to_rgb(np.full(n, hue, dtype=np.float32),
                         np.full(n, 0.85, dtype=np.float32), lit)
    since = t - pulse_at
    if 0.0 <= since < 0.7:
        front = since / 0.6 * n
        wave = np.clip(1.0 - np.abs(np.arange(n) - front) / 3.0, 0.0, 1.0)
        rgb += (wave * (1.0 - since / 0.7))[:, None] * 0.8
    since = t - died_at
    if 0.0 <= since < 1.5:
        rgb[:, 0] += 0.8 * (1.0 - since / 1.5)
    np.clip(rgb, 0.0, 1.0, out=rgb)
    canvas.arches[:] = rgb[:, None, :]


# --------------------------------------------------------------------------- #
# Pac-Man
# --------------------------------------------------------------------------- #

#: Seconds a power pellet keeps the ghosts frightened; they flash for the
#: last two.
FRIGHT_S = 6.0
#: Ghost speed against the player's: chasing, frightened, and as eyes
#: heading home after being eaten.
GHOST_PACE = {"chase": 0.8, "fright": 0.55, "eyes": 1.6}
#: Seconds between ghosts leaving home after a (re)spawn, one at a time, as
#: the arcade does.  Without it the nearest ghost is one tunnel hop away and
#: reaches a player who has not moved yet in about two seconds (measured:
#: 2.2 s from respawn to death at 4 cells/s).
GHOST_RELEASE_S = 2.0
GHOST_COLOURS = np.array([(1.0, 0.15, 0.08), (1.0, 0.45, 0.75),
                          (0.15, 0.95, 1.0), (1.0, 0.65, 0.15)],
                         dtype=np.float32)
PELLET = 10
POWER = 50
#: Sprite radii in cells.  Pac-Man is nearly three cells across -- about 25
#: lattice columns on a small net -- so the mouth reads from the floor; the
#: ghosts are under two, so three of them fit a small triangle.
PAC_RADIUS = 1.4
GHOST_RADIUS = 0.85
EYES_RADIUS = 0.55


class Maze:
    """Walls and pellets on a board, generated so every corridor connects.

    Pillars on every odd row and column give the classic loop-everywhere
    layout with no dead ends; some pillars are stretched into short bars
    (seeded, so the same seed is the same maze) to break the monotony.  Then
    two passes fix what the triangle edges break: walls between separated
    corridors come out until everything is reachable through the tunnels,
    and a wall next to a dead end comes out when that gives it a second
    exit.  Anything still unreachable is bricked over.
    """

    def __init__(self, board: Board, rng: random.Random) -> None:
        self.board = board
        walls = {(r, c) for r, c in board.cells if r % 2 == 1 and c % 2 == 1}
        for r, c in sorted(walls):
            if rng.random() < 0.3:
                n = (r, c + 1) if rng.random() < 0.5 else (r + 1, c)
                if n in board and (n[0] + (n[0] - r), n[1] + (n[1] - c)) in walls:
                    walls.add(n)
        self.walls: set[Cell] = walls
        self._connect()
        self._open_dead_ends()
        self.free: list[Cell] = [c for c in board.cells if c not in walls]
        self.index: dict[Cell, int] = {c: i for i, c in enumerate(self.free)}
        self._exits: dict[Cell, dict[str, Cell]] = {
            c: {d: n for d in DIRECTIONS
                if (n := board.neighbour(c, d, self.walls)) is not None}
            for c in self.free}
        self._distances: dict[Cell, dict[Cell, int]] = {}

    def exits(self, cell: Cell) -> dict[str, Cell]:
        return self._exits[cell]

    def _components(self) -> list[set[Cell]]:
        free = [c for c in self.board.cells if c not in self.walls]
        seen: set[Cell] = set()
        out: list[set[Cell]] = []
        for start in free:
            if start in seen:
                continue
            comp = {start}
            queue = deque([start])
            while queue:
                cell = queue.popleft()
                for d in DIRECTIONS:
                    n = self.board.neighbour(cell, d, self.walls)
                    if n is not None and n not in comp:
                        comp.add(n)
                        queue.append(n)
            seen |= comp
            out.append(comp)
        out.sort(key=len, reverse=True)
        return out

    def _connect(self) -> None:
        for _ in range(64):
            comps = self._components()
            if len(comps) <= 1:
                return
            fixed = False
            for comp in comps[1:]:
                for cell in sorted(comp):
                    for wall in self.board.adjacent(cell):
                        if wall not in self.walls:
                            continue
                        beyond = [n for n in self.board.adjacent(wall)
                                  if n not in self.walls and n not in comp]
                        if beyond:
                            self.walls.discard(wall)
                            fixed = True
                            break
                    if fixed:
                        break
                if fixed:
                    break
            if not fixed:
                # Sealed off by void, not walls: brick it over.
                for comp in comps[1:]:
                    self.walls |= comp
                return

    def _open_dead_ends(self) -> None:
        for _ in range(4):
            changed = False
            for cell in sorted(self.board.cells):
                if cell in self.walls:
                    continue
                exits = [d for d in DIRECTIONS
                         if self.board.neighbour(cell, d, self.walls) is not None]
                if len(exits) > 1:
                    continue
                for wall in self.board.adjacent(cell):
                    if wall not in self.walls:
                        continue
                    if any(n not in self.walls and n != cell
                           for n in self.board.adjacent(wall)):
                        self.walls.discard(wall)
                        changed = True
                        break
            if not changed:
                return

    def distances(self, source: Cell) -> dict[Cell, int]:
        """BFS steps from ``source`` to every free cell, memoised."""
        found = self._distances.get(source)
        if found is None:
            if len(self._distances) > 512:
                self._distances.clear()
            found = {source: 0}
            queue = deque([source])
            while queue:
                cell = queue.popleft()
                for n in self._exits[cell].values():
                    if n not in found:
                        found[n] = found[cell] + 1
                        queue.append(n)
            self._distances[source] = found
        return found


@dataclass
class Ghost:
    home: Cell
    cell: Cell
    prev: Cell
    direction: str | None = None
    mode: str = "chase"
    due: float | None = None
    #: Seconds of waiting at home still owed before this ghost may move.
    wait: float = 0.0
    #: Show time of the last step, for sliding from ``prev`` to ``cell``.
    moved_at: float = -1e9


@dataclass
class Pacman:
    """The rules of Pac-Man on a :class:`Maze`; time is the game's affair."""

    maze: Maze
    seed: int = 7
    games: int = 0
    best: int = 0
    rng: random.Random = field(default_factory=random.Random)
    cell: Cell = (0, 0)
    prev: Cell = (0, 0)
    direction: str | None = None
    wanted: str | None = None
    #: The last direction actually moved in: which way the mouth points
    #: while standing still.
    facing: str = "left"
    ghosts: list[Ghost] = field(default_factory=list)
    pellets: set[Cell] = field(default_factory=set)
    power: set[Cell] = field(default_factory=set)
    score: int = 0
    lives: int = 3
    level: int = 1
    combo: int = 0
    over: bool = False
    running: bool = False
    steps: int = 0

    def __post_init__(self) -> None:
        board, maze = self.maze.board, self.maze
        # The player starts on the biggest island; each other island releases
        # a ghost from its middle.  With fewer islands than ghosts (the big
        # triangle alone) the rest start as far from the player as the maze
        # allows.
        islands = [[c for c in region if c not in maze.walls]
                   for region in board.regions]
        islands = [i for i in islands if i]
        self.start = self._open_spot(islands[0])
        homes = [self._open_spot(i) for i in islands[1:4]]
        far = self.maze.distances(self.start)
        while len(homes) < 3:
            pick = max((c for c in maze.free if c not in homes and c != self.start),
                       key=lambda c: far.get(c, -1))
            homes.append(pick)
        self.homes = homes
        # One power pellet per island, at the far end of it from its middle.
        self.power_cells: list[Cell] = []
        for island, middle in zip(islands, [self.start, *homes]):
            dist = self.maze.distances(middle)
            self.power_cells.append(
                max(island, key=lambda c: (dist.get(c, -1), c)))
        self.reset()

    def _open_spot(self, cells: list[Cell]) -> Cell:
        """The cell nearest the middle of ``cells`` that is a junction.

        The plain centroid can land between two pillars, and a player who
        starts boxed in on the axis they first press reads it as broken.
        """
        middle = self.maze.board.centre(cells)
        ranked = sorted(cells, key=lambda c: (abs(c[0] - middle[0])
                                              + abs(c[1] - middle[1]), c))
        for wanted in (3, 2):
            for cell in ranked:
                if len(self.maze.exits(cell)) >= wanted:
                    return cell
        return middle

    # -- state ------------------------------------------------------------- #

    def reset(self) -> None:
        self.games += 1
        self.rng = random.Random(self.seed * 1000 + self.games)
        self.score, self.lives, self.level = 0, 3, 1
        self.over = self.running = False
        self.steps = 0
        self._lay_pellets()
        self._respawn()

    def _lay_pellets(self) -> None:
        keep = {self.start, *self.homes}
        self.pellets = {c for c in self.maze.free if c not in keep}
        self.power = {c for c in self.power_cells if c in self.pellets}

    def _respawn(self) -> None:
        self.cell = self.prev = self.start
        self.direction = self.wanted = None
        self.ghosts = [Ghost(home=h, cell=h, prev=h, wait=GHOST_RELEASE_S * (i + 1))
                       for i, h in enumerate(self.homes)]
        self.combo = 0

    @property
    def pace(self) -> float:
        """Speed multiplier for the level: +10 % a level, capped."""
        return min(1.6, 1.0 + 0.1 * (self.level - 1))

    def turn(self, direction: str) -> None:
        if direction not in DIRECTIONS:
            raise ValueError(f"no such direction {direction!r}")
        self.wanted = direction

    # -- moves --------------------------------------------------------------- #

    def move(self) -> str:
        """The player's step: "moved", "pellet", "power", "clear", "stuck"."""
        exits = self.maze.exits(self.cell)
        if self.wanted in exits:
            self.direction = self.wanted
        if self.direction not in exits:
            return "stuck"
        self.steps += 1
        self.facing = self.direction
        self.prev, self.cell = self.cell, exits[self.direction]
        if self.cell not in self.pellets:
            return "moved"
        self.pellets.discard(self.cell)
        outcome = "pellet"
        self.score += PELLET
        if self.cell in self.power:
            self.power.discard(self.cell)
            self.score += POWER
            self.combo = 0
            for g in self.ghosts:
                if g.mode != "eyes":
                    g.mode = "fright"
                    g.direction = (_OPPOSITE[g.direction]
                                   if g.direction else None)
            outcome = "power"
        self.best = max(self.best, self.score)
        if not self.pellets:
            self.level += 1
            self._lay_pellets()
            return "clear"
        return outcome

    def move_ghost(self, g: Ghost) -> None:
        exits = self.maze.exits(g.cell)
        if not exits:
            return
        if g.mode == "eyes" and g.cell == g.home:
            g.mode = "chase"
        options = dict(exits)
        if g.direction and len(options) > 1:
            options.pop(_OPPOSITE[g.direction], None)
        if g.mode == "eyes":
            dist = self.maze.distances(g.home)
            pick = min(options, key=lambda d: (dist.get(options[d], 9999),
                                               self.rng.random()))
        elif g.mode == "fright":
            dist = self.maze.distances(self.cell)
            pick = max(options, key=lambda d: (dist.get(options[d], -1),
                                               self.rng.random()))
        elif self.rng.random() < 0.8:
            dist = self.maze.distances(self.cell)
            pick = min(options, key=lambda d: (dist.get(options[d], 9999),
                                               self.rng.random()))
        else:
            pick = self.rng.choice(sorted(options))
        g.direction = pick
        g.prev, g.cell = g.cell, options[pick]

    def collisions(self) -> str:
        """After anyone moved: "ghost" (one eaten), "died", or ""."""
        for g in self.ghosts:
            met = (g.cell == self.cell
                   or (g.cell == self.prev and g.prev == self.cell))
            if not met or g.mode == "eyes":
                continue
            if g.mode == "fright":
                self.score += 200 * (2 ** self.combo)
                self.combo += 1
                self.best = max(self.best, self.score)
                g.mode = "eyes"
                return "ghost"
            self.lives -= 1
            if self.lives <= 0:
                self.over = True
                self.running = False
            return "died"
        return ""

    def snapshot(self) -> dict:
        marks = ["0"] * len(self.maze.free)
        for c in self.pellets:
            marks[self.maze.index[c]] = "2" if c in self.power else "1"
        return {
            "running": self.running, "over": self.over,
            "score": self.score, "best": self.best, "lives": self.lives,
            "level": self.level, "steps": self.steps, "games": self.games,
            "pellets_left": len(self.pellets),
            "pac": list(self.cell), "direction": self.direction,
            "facing": self.facing,
            "ghosts": [{"cell": list(g.cell), "mode": g.mode} for g in self.ghosts],
            "pellets": "".join(marks),
        }


class PacmanGame:
    """Pac-Man across every triangle: the clock, the input and the paint."""

    name = "pacman"
    surface = "all"

    def __init__(self, board: Board, seed: int = 7) -> None:
        self.board = board
        self.maze = Maze(board, random.Random(seed))
        self.pac = Pacman(self.maze, seed=seed)
        self._fright_until = -1e9
        self._hold_until: float | None = None
        self._ate_at = -1e9
        self._ghost_at = -1e9
        self._died_at = -1e9
        self._clear_at = -1e9
        #: When Pac-Man last stepped, and how far the mouth has chomped:
        #: one open-and-shut every two cells, frozen while standing.
        self._moved_at = -1e9
        self._period = 0.25
        self._chomp = 0.0
        self._table = np.zeros((board.rows * board.cols, 3), dtype=np.float32)
        self._wall_index = np.array([board.index(c) for c in self.maze.walls],
                                    dtype=np.int64)

    def describe(self) -> dict:
        return {"game": self.name, **self.board.describe(),
                "walls": [[r, c] for r, c in sorted(self.maze.walls)],
                "free": [[r, c] for r, c in self.maze.free]}

    def snapshot(self) -> dict:
        return {"game": self.name, "fright": any(g.mode == "fright"
                                                  for g in self.pac.ghosts),
                **self.pac.snapshot()}

    def input(self, action: str, direction: str | None = None) -> None:
        pac = self.pac
        if action == "turn":
            if direction is None:
                raise ValueError("turn needs a direction")
            pac.turn(direction)
            if not pac.over:
                pac.running = True
        elif action == "start":
            if not pac.over:
                pac.running = True
        elif action == "pause":
            pac.running = False
            self._freeze()
        elif action == "toggle":
            if not pac.over:
                pac.running = not pac.running
                if not pac.running:
                    self._freeze()
        elif action == "reset":
            pac.reset()
            self._freeze()
            self._hold_until = None
        else:
            raise ValueError(f"no such action {action!r}")

    def _freeze(self) -> None:
        self._due = None
        self._last_t = None
        for g in self.pac.ghosts:
            g.due = None

    _due: float | None = None
    _last_t: float | None = None

    def advance(self, t: float, speed: float) -> bool:
        before = self._last_t
        changed = self._advance(t, speed)
        pac = self.pac
        if (before is not None and pac.running and self._hold_until is None
                and pac.direction in self.maze.exits(pac.cell)):
            # The mouth chomps while he is actually going somewhere.
            self._chomp += max(0.0, min(t - before, 0.1)) / self._period
        self._last_t = t
        return changed

    def _advance(self, t: float, speed: float) -> bool:
        pac = self.pac
        if not pac.running:
            return False
        if self._hold_until is not None:
            # The moment after a death or a cleared level: everyone stands
            # still, then the board is reset and play resumes.
            if t < self._hold_until:
                return False
            self._hold_until = None
            pac._respawn()
            self._freeze()
            return True
        period = 1.0 / (max(0.5, float(speed)) * pac.pace)
        self._period = period
        changed = False
        if self._last_t is None:
            self._last_t = t
        if self._due is None or self._due < t - 2 * period:
            self._due = t + period
        if t >= self._due:
            self._due += period
            outcome = pac.move()
            changed = outcome != "stuck"
            if changed:
                self._moved_at = t
            if outcome in ("pellet", "power", "clear"):
                self._ate_at = t
            if outcome == "power":
                self._fright_until = t + FRIGHT_S
            if outcome == "clear":
                self._clear_at = t
                self._hold_until = t + 2.0
                return True
            if self._settle(t):
                return True
        if t >= self._fright_until:
            for g in pac.ghosts:
                if g.mode == "fright":
                    g.mode = "chase"
                    changed = True
        for g in pac.ghosts:
            gp = period / GHOST_PACE[g.mode]
            if g.due is None or g.due < t - 2 * gp:
                g.due = t + gp
            if g.wait > 0.0:
                # Still owed time at home: it comes off as the clock runs,
                # so a pause does not release anyone.
                g.wait -= max(0.0, min(t - self._last_t, 0.1))
                g.due = t + gp
                continue
            if t >= g.due:
                g.due += gp
                pac.move_ghost(g)
                g.moved_at = t
                changed = True
                if self._settle(t):
                    return True
        return changed

    def _settle(self, t: float) -> bool:
        """Collisions after a move.  True if play was interrupted."""
        outcome = self.pac.collisions()
        if outcome == "ghost":
            self._ghost_at = t
        elif outcome == "died":
            self._died_at = t
            if not self.pac.over:
                self._hold_until = t + 1.5
            return True
        return False

    def paint(self, canvas: Canvas, t: float, gain: float,
              level: float | None, beat_phase: float) -> None:
        pac, board, maze, table = self.pac, self.board, self.maze, self._table
        table[:] = 0.0
        # The maze: walls in the classic blue, pellets as warm dots, power
        # pellets pulsing on the beat.
        table[self._wall_index] = (0.02, 0.05, 0.22)
        if pac.pellets:
            table[[board.index(c) for c in pac.pellets]] = (0.22, 0.18, 0.12)
        if pac.power:
            pulse = 0.5 + 0.5 * (1.0 - beat_phase) ** 2
            table[[board.index(c) for c in pac.power]] = (
                0.9 * pulse, 0.7 * pulse, 0.45 * pulse)
        since = t - self._ghost_at
        if 0.0 <= since < 0.3:
            table += 0.3 * (1.0 - since / 0.3)
        since = t - self._clear_at
        if 0.0 <= since < 2.0:
            flash = 0.35 * (0.5 + 0.5 * np.cos(since * 12.0)) * (1.0 - since / 2.0)
            table[board.playable.reshape(-1)] += flash
        since = t - self._died_at
        if 0.0 <= since < 1.5:
            table[board.playable.reshape(-1), 0] += 0.5 * (1.0 - since / 1.5)
        table *= gain
        np.clip(table, 0.0, 1.0, out=table)
        target = board.target(canvas)
        board.paint(target, table)
        self._paint_sprites(target, t, gain)
        total = max(1, len(maze.free) - 4)
        progress = 1.0 - len(pac.pellets) / total
        _paint_corridor(canvas, t, 48.0, progress,
                        max(self._ate_at, self._ghost_at), self._died_at)
        fright_left = self._fright_until - t
        if fright_left > 0.0 and not pac.over:
            canvas.arches[..., 2] += 0.25 * min(1.0, fright_left / 2.0)
            np.clip(canvas.arches, 0.0, 1.0, out=canvas.arches)

    @staticmethod
    def _slide(prev: Cell, cell: Cell, moved_at: float, t: float,
               period: float) -> tuple[float, float]:
        """Where a sprite is between two cells: ``prev`` sliding to ``cell``
        over one period, except through a tunnel, where it just appears."""
        if abs(cell[0] - prev[0]) + abs(cell[1] - prev[1]) != 1:
            return float(cell[0]), float(cell[1])
        f = min(1.0, max(0.0, (t - moved_at) / max(period, 1e-6)))
        return (prev[0] + (cell[0] - prev[0]) * f,
                prev[1] + (cell[1] - prev[1]) * f)

    def _paint_sprites(self, target: np.ndarray, t: float, gain: float) -> None:
        pac, board = self.pac, self.board
        fright_left = self._fright_until - t
        for i, g in enumerate(pac.ghosts):
            gp = self._period / GHOST_PACE[g.mode]
            at = self._slide(g.prev, g.cell, g.moved_at, t, gp)
            if g.mode == "eyes":
                colour, radius = (0.25, 0.25, 0.45), EYES_RADIUS
            elif g.mode == "fright":
                # Flash white for the last two seconds, twice a second.
                flashing = fright_left < 2.0 and int(t * 4) % 2 == 0
                colour = (0.85, 0.85, 0.85) if flashing else (0.15, 0.15, 0.95)
                radius = GHOST_RADIUS
            else:
                colour = tuple(GHOST_COLOURS[i % len(GHOST_COLOURS)])
                radius = GHOST_RADIUS
            board.sprite(target, at, radius, np.array(colour, np.float32) * gain)

        # Pac-Man: a disc with a wedge for a mouth, opening and shutting once
        # every two cells while he moves.  Dying, the mouth opens all the way
        # round as he fades -- the arcade's death, at 14 pixels a cell.
        at = self._slide(pac.prev, pac.cell, self._moved_at, t, self._period)
        dr, dc = DIRECTIONS[pac.facing]
        facing = (float(dc), float(dr))
        since_died = t - self._died_at
        if 0.0 <= since_died < 1.5 or pac.over:
            k = min(1.0, max(0.0, since_died / 1.5))
            mouth = 0.5 + (np.pi - 0.5) * k
            colour = np.array((1.0, 0.85, 0.05), np.float32) * (1.0 - 0.7 * k)
        else:
            mouth = 0.12 + 0.68 * abs(np.sin(np.pi * self._chomp / 2.0))
            colour = np.array((1.0, 0.85, 0.05), np.float32)
        board.sprite(target, at, PAC_RADIUS, colour * gain, facing, float(mouth))


GAMES = {SnakeGame.name: SnakeGame, PacmanGame.name: PacmanGame}


# --------------------------------------------------------------------------- #
# The engine's side
# --------------------------------------------------------------------------- #


class Game:
    """What the engine drives: picks the game, steps it on show time, paints.

    Games are built lazily and kept, so switching back to one resumes its
    best score; boards are shared between games on the same surface.
    """

    def __init__(self, canvas: Canvas, seed: int = 7, kind: str = "snake") -> None:
        self.canvas = canvas
        self.seed = int(seed)
        self.lock = threading.Lock()
        #: Bumped on every visible change, so a pusher can poll an int.
        self.serial = 0
        self._boards: dict[str, Board] = {}
        self._games: dict[str, SnakeGame | PacmanGame] = {}
        self.kind = ""
        self.select(kind)
        self._pump = 0.0
        self._pump_t: float | None = None

    def _board(self, surface: str) -> Board:
        if surface not in self._boards:
            self._boards[surface] = Board(self.canvas, surface)
        return self._boards[surface]

    def select(self, kind: str) -> None:
        """Make ``kind`` the current game; a no-op if it already is."""
        if kind not in GAMES:
            raise ValueError(f"no such game {kind!r}; pick one of {list(GAMES)}")
        if kind not in self._games:
            cls = GAMES[kind]
            self._games[kind] = cls(self._board(cls.surface), seed=self.seed)
        if kind != self.kind:
            self.kind = kind
            self.serial += 1

    @property
    def current(self):
        return self._games[self.kind]

    @property
    def board(self) -> Board:
        return self.current.board

    def describe(self) -> dict:
        with self.lock:
            return {**self.current.describe(), "games": list(GAMES)}

    # -- input (web thread) ------------------------------------------------ #

    def input(self, action: str, direction: str | None = None) -> dict:
        with self.lock:
            self.current.input(action, direction)
            self.serial += 1
            out = self.current.snapshot()
        out["serial"] = self.serial
        return out

    def snapshot(self) -> dict:
        with self.lock:
            out = self.current.snapshot()
        out["serial"] = self.serial
        return out

    # -- render (engine thread) -------------------------------------------- #

    def render(self, t: float, *, kind: str | None = None, speed: float = 4.0,
               bounce: float = 0.6, features=None,
               beat_phase: float = 0.0) -> None:
        """Advance the current game to show time ``t`` and paint the rig."""
        # The kick, peak-held and decaying over about a bar, as the arranger
        # does for showpieces.  ``level`` (RMS against the set's own peak) is
        # what a meter shows; the board bounces on the kick.
        if self._pump_t is None:
            self._pump_t = t
        decayed = self._pump * float(np.exp(-(t - self._pump_t) * 1.6))
        instant = (min(1.0, float(features.kick) / 4.0)
                   if features is not None else 0.0)
        self._pump = max(decayed, instant)
        self._pump_t = t
        bounce = min(max(float(bounce), 0.0), 1.0)
        gain = 1.0
        level = None
        if features is not None:
            gain = 1.0 - 0.45 * bounce + 0.45 * bounce * self._pump
            level = min(1.0, float(features.level)) * (0.2 + 0.8 * bounce)

        with self.lock:
            if kind is not None and kind in GAMES:
                self.select(kind)
            game = self.current
            if game.advance(t, speed):
                self.serial += 1
            self.canvas.clear()
            game.paint(self.canvas, t, gain, level, beat_phase)
