"""The live effect vocabulary: corridor patterns and net effects.

The offline arranger schedules xLights effects with start times and fades.  A
real-time renderer has no schedule -- it is asked "what does this look like
*now*" forty times a second -- so the same ideas take a different shape:

* a **corridor pattern** is a pure function ``(n_arches, phase) -> levels``,
  where ``phase`` runs 0..1 across one phrase.  No timeline, no fades: the
  falloff that the offline code got from an effect's fade-out is here just the
  shape of the level curve.
* a **net effect** paints into ``canvas.nets`` using the geometry vectors.

Both are deterministic functions of their arguments -- including the random
ones, which take a frame index rather than a live RNG.  That is what keeps a
capture byte-comparable with a re-render, which is how M1's oracle works.

The one rule carried over from the offline show, because it was learned from
sequences that came out visually dead: **lit arches must overlap.**  A comet
whose tail is shorter than the gap between arches reads as 24 things blinking
in sequence, not as one band travelling through a tunnel.
"""

from __future__ import annotations

import numpy as np

from .frame import Canvas
from .palette import Palette

# --------------------------------------------------------------------------- #
# Corridor patterns:  (n, phase) -> (n,) levels in 0..1
# --------------------------------------------------------------------------- #


def _head(n: int, phase: float, tail: float, reverse: bool) -> np.ndarray:
    """One head of light at ``phase``, trailing ``tail`` arches behind it."""
    index = np.arange(n, dtype=np.float32)
    if reverse:
        index = index[::-1]
    distance = phase * (n + tail) - index
    return np.clip(1.0 - distance / tail, 0.0, 1.0) * (distance >= 0.0)


def comet(n: int, phase: float, tail: float = 4.0, reverse: bool = False) -> np.ndarray:
    """One head running the length of the corridor."""
    return _head(n, phase, tail, reverse)


def bounce(n: int, phase: float, tail: float = 4.0, reverse: bool = False) -> np.ndarray:
    """Down the corridor and back, inside one phrase."""
    if phase < 0.5:
        return _head(n, phase * 2.0, tail, reverse)
    return _head(n, (phase - 0.5) * 2.0, tail, not reverse)


def converge(n: int, phase: float, tail: float = 3.0, reverse: bool = False) -> np.ndarray:
    """Both ends run inward and meet in the middle."""
    half = n // 2
    front = _head(half, phase, tail, False)
    return np.concatenate([front, front[::-1]])[:n]


def diverge(n: int, phase: float, tail: float = 3.0, reverse: bool = False) -> np.ndarray:
    """Opens from the middle outward -- reads as the tunnel splitting."""
    half = n // 2
    back = _head(half, phase, tail, True)
    return np.concatenate([back, back[::-1]])[:n]


def alternate(n: int, phase: float, steps: int = 8, reverse: bool = False) -> np.ndarray:
    """Odd and even arches trade places, ``steps`` times per phrase."""
    step = int(phase * steps)
    level = np.zeros(n, dtype=np.float32)
    level[step % 2::2] = 1.0
    return level * _decay(phase * steps - step)


def pairs(n: int, phase: float, size: int = 4, reverse: bool = False) -> np.ndarray:
    """Blocks of arches step down the corridor -- chunkier than a comet."""
    groups = -(-n // size)
    position = phase * groups
    index = np.arange(n, dtype=np.float32) // size
    if reverse:
        index = (groups - 1) - index
    distance = position - index
    return np.clip(1.0 - distance / 1.6, 0.0, 1.0) * (distance >= 0.0)


def strobe(n: int, phase: float, steps: int = 8, reverse: bool = False) -> np.ndarray:
    """The whole corridor pulses as one."""
    step = int(phase * steps)
    return np.full(n, _decay(phase * steps - step), dtype=np.float32)


def sparkle(n: int, phase: float, steps: int = 16, fraction: float = 0.15,
            seed: int = 0, reverse: bool = False) -> np.ndarray:
    """Scattered arches blink at random.

    This is what keeps a quiet passage alive.  An intro should be low and slow,
    but low is not the same as dark -- with nothing moving the rig looks
    powered off rather than restrained.  Seeded by the step index, so it is
    reproducible frame for frame.
    """
    step = int(phase * steps)
    rng = np.random.default_rng((seed, step))
    level = np.zeros(n, dtype=np.float32)
    chosen = rng.choice(n, size=max(1, int(n * fraction)), replace=False)
    level[chosen] = rng.uniform(0.6, 1.0, size=chosen.size)
    return level * _decay(phase * steps - step)


def chase(n: int, phase: float, heads: int = 3, tail: float = 3.0,
          reverse: bool = False) -> np.ndarray:
    """Several evenly spaced heads circulating the corridor -- a comet's busier
    sibling.  Wraps, so the tunnel never empties between heads."""
    index = np.arange(n, dtype=np.float32)
    if reverse:
        index = index[::-1]
    level = np.zeros(n, dtype=np.float32)
    for k in range(heads):
        position = ((phase + k / heads) % 1.0) * n
        distance = (position - index) % n
        level = np.maximum(level, np.clip(1.0 - distance / tail, 0.0, 1.0))
    return level


def wave(n: int, phase: float, cycles: float = 2.0, reverse: bool = False) -> np.ndarray:
    """A smooth sine travelling the tunnel: no head, no tail, just swell."""
    depth = np.arange(n, dtype=np.float32) / max(1, n - 1)
    if reverse:
        depth = depth[::-1]
    level = 0.5 + 0.5 * np.sin(2 * np.pi * (cycles * depth - phase))
    return (level ** 1.5).astype(np.float32)


def swell(n: int, phase: float, reverse: bool = False) -> np.ndarray:
    """The whole corridor breathes once per phrase -- the calm cousin of
    ``strobe``."""
    level = 0.5 + 0.5 * np.sin(2 * np.pi * phase - np.pi / 2)
    return np.full(n, float(level) ** 1.2, dtype=np.float32)


def fill(n: int, phase: float, soft: float = 1.5, reverse: bool = False) -> np.ndarray:
    """Fills from the mouth to the back, then drains the same way."""
    index = np.arange(n, dtype=np.float32)
    if reverse:
        index = index[::-1]
    if phase < 0.5:
        edge = phase * 2.0 * (n + soft)
        return np.clip((edge - index) / soft, 0.0, 1.0).astype(np.float32)
    edge = (phase - 0.5) * 2.0 * (n + soft) - soft
    return np.clip((index - edge) / soft, 0.0, 1.0).astype(np.float32)


def shower(n: int, phase: float, heads: int = 4, tail: float = 2.5,
           seed: int = 0, reverse: bool = False) -> np.ndarray:
    """Comets at different speeds and offsets, so they overtake each other.

    Seeded, so a re-render matches; unlike ``sparkle`` the seed is the only
    randomness, and the motion inside the phrase is continuous."""
    rng = np.random.default_rng(seed)
    speeds = rng.integers(1, 4, size=heads)
    offsets = rng.uniform(0.0, 1.0, size=heads)
    index = np.arange(n, dtype=np.float32)
    if reverse:
        index = index[::-1]
    level = np.zeros(n, dtype=np.float32)
    for speed, offset in zip(speeds, offsets):
        position = ((phase * speed + offset) % 1.0) * (n + tail)
        distance = position - index
        level = np.maximum(level, np.clip(1.0 - distance / tail, 0.0, 1.0)
                           * (distance >= 0.0))
    return level


def heartbeat(n: int, phase: float, steps: int = 4, reverse: bool = False) -> np.ndarray:
    """Two quick pulses then a rest, ``steps`` times a phrase -- da-dum."""
    within = phase * steps - int(phase * steps)
    first = float(np.exp(-within * 9.0))
    second = float(np.exp(-(within - 0.28) * 9.0)) * 0.75 if within >= 0.28 else 0.0
    return np.full(n, max(first, second), dtype=np.float32)


def _decay(within: float, hold: float = 0.35) -> float:
    """A flash's envelope inside one step: full, then a soft tail."""
    if within <= hold:
        return 1.0
    return float(max(0.0, 1.0 - (within - hold) / (1.0 - hold)) ** 1.5)


PATTERNS = {
    "sparkle": sparkle, "comet": comet, "converge": converge, "diverge": diverge,
    "bounce": bounce, "pairs": pairs, "alternate": alternate, "strobe": strobe,
    "chase": chase, "wave": wave, "swell": swell, "fill": fill,
    "shower": shower, "heartbeat": heartbeat,
}

#: Patterns that take a ``seed`` keyword.
SEEDED = frozenset({"sparkle", "shower"})

#: Roughly how much light each pattern puts in the corridor per unit time, 0-1.
#: Selection is weighted by this so the corridor's business tracks the music:
#: picking uniformly lets a drop draw a sparse comet while a quiet passage
#: draws a full strobe, which inverts the dynamics.  Ported unchanged from the
#: offline arranger, where the numbers were tuned against real tracks.
DENSITY = {
    "sparkle": 0.20, "comet": 0.30, "converge": 0.42, "diverge": 0.42,
    "bounce": 0.55, "pairs": 0.72, "alternate": 0.90, "strobe": 1.00,
    # The second generation, placed by the same eye: how much of the
    # corridor is lit, how much of the time.
    "shower": 0.38, "chase": 0.45, "wave": 0.50, "swell": 0.52,
    "fill": 0.60, "heartbeat": 0.78,
}

#: How many of the nearest patterns a phrase may choose from.  Three when the
#: vocabulary was eight; with fourteen there are enough near neighbours at
#: every energy to widen it without reaching for something wrong.
VOCABULARY = 4


def vocabulary(target: float, quiet: bool = False) -> list[str]:
    """Pattern names whose density suits ``target``, nearest first."""
    names = list(DENSITY)
    if quiet:
        # Never strobe the corridor during a quiet passage, however the numbers
        # happen to fall out.
        names = [n for n in names if DENSITY[n] <= 0.45]
    names.sort(key=lambda n: abs(DENSITY[n] - target))
    return names[:VOCABULARY]


# --------------------------------------------------------------------------- #
# Compositing
# --------------------------------------------------------------------------- #


def corridor(canvas: Canvas, levels: np.ndarray, near: Palette,
             far: Palette | None = None, *, brightness: float = 1.0,
             height: float = 0.0, mode: str = "add") -> None:
    """Paint per-arch ``levels`` into the corridor with a depth gradient.

    ``height`` graduates each arch along its own run: 0 is flat, 1 fades the
    bases out and leaves the apex bright.  ``far`` makes the tunnel travel
    through *hue* down its depth rather than merely getting darker -- cyan at
    the mouth to violet at the back, which is what makes it read as depth.
    """
    depth = canvas.depth
    colors = near.ramp(depth)
    if far is not None:
        colors = colors * (1 - depth[:, None]) + far.ramp(depth) * depth[:, None]
    contribution = (levels[:, None, None].astype(np.float32)
                    * colors[:, None, :] * brightness)
    if height > 0:
        profile = (1.0 - height) + height * canvas.arch_h
        contribution = contribution * profile[None, :, None]
    _blend(canvas.arches, contribution, mode)


def wash(canvas: Canvas, palette: Palette, level: float = 1.0, *,
         targets: slice | None = None, gradient: float = 0.0,
         mode: str = "add") -> None:
    """Flat colour over the nets, optionally graduated apex to base."""
    colors = palette.ramp(canvas.net_y * gradient if gradient else
                          np.zeros_like(canvas.net_y))
    _blend_nets(canvas, colors * level, targets, mode)


def bars(canvas: Canvas, palette: Palette, phase: float, *, count: int = 3,
         angle: float = 0.0, width: float = 0.35, level: float = 1.0,
         targets: slice | None = None, mode: str = "add") -> None:
    """Bands sweeping across the nets.  ``angle`` 0 = vertical, 1 = horizontal."""
    coord = canvas.net_x * (1.0 - angle) + canvas.net_y * angle
    cycle = (coord * count - phase * count) % 1.0
    profile = np.clip(1.0 - np.abs(cycle - 0.5) / max(1e-3, width), 0.0, 1.0)
    colors = palette.ramp(cycle) * profile[:, None] * level
    _blend_nets(canvas, colors, targets, mode)


def radial(canvas: Canvas, palette: Palette, phase: float, *, width: float = 0.25,
           level: float = 1.0, targets: slice | None = None,
           mode: str = "add") -> None:
    """A ring expanding from each net's centre -- the kick's natural shape."""
    ring = np.clip(1.0 - np.abs(canvas.net_r - phase) / max(1e-3, width), 0.0, 1.0)
    colors = palette.ramp(canvas.net_r) * (ring ** 2)[:, None] * level
    _blend_nets(canvas, colors, targets, mode)


def pinwheel(canvas: Canvas, palette: Palette, phase: float, *, arms: int = 3,
             level: float = 1.0, targets: slice | None = None,
             mode: str = "add") -> None:
    """Arms rotating about each net's centre."""
    spin = (canvas.net_angle * arms + phase) % 1.0
    profile = np.clip(1.0 - np.abs(spin - 0.5) * 2.0, 0.0, 1.0) ** 2
    colors = palette.ramp(canvas.net_r) * profile[:, None] * level
    _blend_nets(canvas, colors, targets, mode)


def spiral(canvas: Canvas, palette: Palette, phase: float, *, arms: int = 2,
           twist: float = 1.5, level: float = 1.0, targets: slice | None = None,
           mode: str = "add") -> None:
    """A pinwheel whose arms curl: the angle advances with radius."""
    spin = (canvas.net_angle * arms + canvas.net_r * twist + phase) % 1.0
    profile = np.clip(1.0 - np.abs(spin - 0.5) * 2.0, 0.0, 1.0) ** 2
    colors = palette.ramp(canvas.net_r) * profile[:, None] * level
    _blend_nets(canvas, colors, targets, mode)


def ripples(canvas: Canvas, palette: Palette, phase: float, *, rings: float = 3.0,
            level: float = 1.0, targets: slice | None = None,
            mode: str = "add") -> None:
    """Concentric rings flowing outward -- ``radial`` repeated, and smooth."""
    field = 0.5 + 0.5 * np.sin(2 * np.pi * (canvas.net_r * rings - phase))
    profile = field ** 2
    colors = palette.ramp(canvas.net_r) * profile[:, None] * level
    _blend_nets(canvas, colors, targets, mode)


def checker(canvas: Canvas, palette: Palette, step: int, *, cells: int = 4,
            level: float = 1.0, targets: slice | None = None,
            mode: str = "add") -> None:
    """A checkerboard that flips parity on every ``step``."""
    cx = np.floor(canvas.net_x * cells).astype(np.int32)
    cy = np.floor(canvas.net_y * cells).astype(np.int32)
    on = ((cx + cy + step) % 2 == 0).astype(np.float32)
    colors = palette.ramp((cx / cells).astype(np.float32)) * on[:, None] * level
    _blend_nets(canvas, colors, targets, mode)


def orbit(canvas: Canvas, palette: Palette, phase: float, *, radius: float = 0.55,
          width: float = 0.18, level: float = 1.0, targets: slice | None = None,
          mode: str = "add") -> None:
    """One soft blob circling each net's centre."""
    gap = np.abs(((canvas.net_angle - phase) + 0.5) % 1.0 - 0.5)
    profile = np.exp(-(gap / width) ** 2) * np.exp(-((canvas.net_r - radius) / 0.35) ** 2)
    colors = palette.ramp(canvas.net_angle) * profile[:, None] * level
    _blend_nets(canvas, colors, targets, mode)


def halves(canvas: Canvas, palette: Palette, left: bool, strength: float = 1.0, *,
           level: float = 1.0, targets: slice | None = None,
           mode: str = "add") -> None:
    """Light one half of each net -- left or right -- with a soft seam."""
    edge = canvas.net_x - 0.5
    side = np.clip((-edge if left else edge) / 0.08 + 0.5, 0.0, 1.0)
    colors = palette.ramp(canvas.net_y) * (side * strength)[:, None] * level
    _blend_nets(canvas, colors, targets, mode)


def apex(canvas: Canvas, palette: Palette, strength: float, *, level: float = 1.0,
         targets: slice | None = None, mode: str = "add") -> None:
    """A flash that starts at the apex and dies toward the base -- the kick's
    shape on a triangle."""
    profile = ((1.0 - canvas.net_y) ** 2) * strength
    colors = palette.ramp(canvas.net_y) * profile[:, None] * level
    _blend_nets(canvas, colors, targets, mode)


def plasma(canvas: Canvas, palette: Palette, t: float, *, scale: float = 3.0,
           speed: float = 0.5, level: float = 1.0,
           targets: slice | None = None, mode: str = "add") -> None:
    """Slow interference of three sines -- texture for beds and breakdowns."""
    x, y = canvas.net_x * scale, canvas.net_y * scale
    phase = t * speed
    field = (np.sin(x + phase) + np.sin(y * 1.3 - phase * 0.7)
             + np.sin((x + y) * 0.7 + phase * 1.3))
    value = (field / 3.0 + 1.0) * 0.5
    colors = palette.ramp(value) * (value ** 1.5)[:, None] * level
    _blend_nets(canvas, colors, targets, mode)


def net_sparkle(canvas: Canvas, palette: Palette, frame: int, *,
                density: float = 0.02, level: float = 1.0, seed: int = 0,
                targets: slice | None = None,
                mode: str = "add") -> None:
    """Random pixels glint.  Seeded by frame, so a re-render matches exactly."""
    view = _net_view(canvas, targets)
    rng = np.random.default_rng((seed, frame))
    hits = rng.random(view.shape[:2]) < density
    colors = palette.ramp(rng.random(view.shape[:2])) * level
    _blend(view, colors * hits[..., None], mode)


def par(canvas: Canvas, color: np.ndarray | tuple[float, ...], level: float = 1.0,
        white: float = 0.0) -> None:
    """The DJ par: one RGBW pixel, driven separately from everything else."""
    canvas.par[:3] = np.asarray(color, dtype=np.float32)[:3] * level
    if canvas.par.size > 3:
        canvas.par[3] = white * level


# --------------------------------------------------------------------------- #


def _blend_nets(canvas: Canvas, colors: np.ndarray,
                targets: slice | None, mode: str) -> None:
    _blend(_net_view(canvas, targets), colors, mode)


def _net_view(canvas: Canvas, targets: slice | None) -> np.ndarray:
    """A writable view of some nets.

    Deliberately slice-only: fancy indexing would hand back a *copy*, and every
    effect painted into it would vanish silently.  The show's net groups are
    contiguous anyway -- see :meth:`Canvas.net_slice`.
    """
    if targets is None:
        return canvas.nets
    if not isinstance(targets, slice):
        raise TypeError(
            "targets must be a slice (use Canvas.net_slice); an index array "
            "would copy and the paint would be lost"
        )
    return canvas.nets[targets]


def _blend(buffer: np.ndarray, contribution: np.ndarray, mode: str) -> None:
    if mode == "add":
        buffer += contribution
    elif mode == "max":
        np.maximum(buffer, contribution, out=buffer)
    elif mode == "set":
        buffer[...] = contribution
    else:
        raise ValueError(f"unknown blend mode {mode!r}")
