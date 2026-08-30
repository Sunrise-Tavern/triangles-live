"""A deterministic test pattern whose failures are legible in xLights.

Every stage isolates one thing that :mod:`live.layout` could get wrong, so
watching the capture in xLights' 3D preview tells you *which* mapping is off
rather than just "it looks wrong":

=========  ==================  ===================================================
stage      what you should see  what a failure means
=========  ==================  ===================================================
identify   models light one at  a model stays dark, or two light together ->
           a time, front to     start channel wrong
           back
primaries  the whole rig goes   nets red while arches are green -> string colour
           red, green, blue     order (the arches are GRB, the nets RGB)
corridor   one band travels     the band jumps around -> the Tunnel group order
           front to back        is not physical order
arch       each arch fills      it fills unevenly or from the middle -> the
           base -> apex -> base polyline segment split is wrong
nets       a bar sweeps left    the bar scatters -> the custom-model node map is
           to right across      wrong
           each net
par        the par cycles       nothing -> the DMX slot map is wrong
           R, G, B, W
=========  ==================  ===================================================

The pattern is a pure function of frame index, so the same run always produces
the same bytes -- which is what makes the byte-exact round-trip test possible.
"""

from __future__ import annotations

import numpy as np

from .layout import Layout

#: (name, seconds).  Total is 30 s, matching the plan's fixed verification run.
STAGES: tuple[tuple[str, float], ...] = (
    ("identify", 6.0),
    ("primaries", 6.0),
    ("corridor", 6.0),
    ("arch", 3.0),
    ("nets", 6.0),
    ("par", 3.0),
)
DURATION = sum(seconds for _, seconds in STAGES)


def stage_at(t: float) -> tuple[str, float]:
    """Stage name and 0..1 progress within it, for show time ``t`` seconds."""
    t = t % DURATION
    for name, seconds in STAGES:
        if t < seconds:
            return name, t / seconds
        t -= seconds
    return STAGES[-1][0], 1.0


def frame(layout: Layout, t: float) -> np.ndarray:
    """One full channel array for show time ``t``."""
    out = layout.blank_channels()
    name, progress = stage_at(t)
    ordered = [layout[n] for n in (*layout.nets, *([layout.par] if layout.par else []),
                                   *layout.arches)]

    if name == "identify":
        which = int(progress * len(ordered)) % len(ordered)
        model = ordered[which]
        rgb = np.full((model.nodes, 3), 0, dtype=np.uint8)
        rgb[:] = _hue(which / len(ordered))
        model.pack(rgb, out)

    elif name == "primaries":
        colour = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][int(progress * 3) % 3]
        for model in ordered:
            rgb = np.empty((model.nodes, 3), dtype=np.uint8)
            rgb[:] = colour
            model.pack(rgb, out)

    elif name == "corridor":
        count = len(layout.arches)
        head = progress * (count + 3)
        for i, name_ in enumerate(layout.arches):
            model = layout[name_]
            distance = head - i
            level = 0.0 if distance < 0 else max(0.0, 1.0 - distance / 3.0)
            rgb = np.empty((model.nodes, 3), dtype=np.uint8)
            rgb[:] = (0, round(255 * level), round(255 * level))
            model.pack(rgb, out)

    elif name == "arch":
        for name_ in layout.arches:
            model = layout[name_]
            t_along = model.strip_t if model.strip_t is not None else np.linspace(
                0, 1, model.nodes, dtype=np.float32
            )
            lit = (t_along <= progress).astype(np.uint8) * 255
            rgb = np.zeros((model.nodes, 3), dtype=np.uint8)
            rgb[:, 0] = lit          # red: pure, so a GRB slip shows as green
            model.pack(rgb, out)

    elif name == "nets":
        for name_ in layout.nets:
            model = layout[name_]
            height, width = model.grid
            bar = progress * width
            col = model.coords[:, 1].astype(np.float32)
            level = np.clip(1.0 - np.abs(col - bar) / 4.0, 0.0, 1.0)
            rgb = np.zeros((model.nodes, 3), dtype=np.uint8)
            rgb[:, 0] = np.rint(level * 255).astype(np.uint8)
            rgb[:, 2] = np.rint(level * 255).astype(np.uint8)
            model.pack(rgb, out)

    elif name == "par" and layout.par:
        model = layout[layout.par]
        slot = int(progress * 4) % 4
        rgb = np.zeros((1, 4), dtype=np.uint8)
        rgb[0, slot] = 255
        model.pack(rgb, out)

    return out


def frames(layout: Layout, fps: float = 40.0, seconds: float = DURATION):
    """Yield ``fps * seconds`` frames of the pattern, in order."""
    for i in range(int(round(fps * seconds))):
        yield frame(layout, i / fps)


def _hue(h: float) -> tuple[int, int, int]:
    """Cheap HSV(h, 1, 1) -> RGB, so consecutive models look different."""
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    q, t_ = int(255 * (1 - f)), int(255 * f)
    return [(255, t_, 0), (q, 255, 0), (0, 255, t_),
            (0, q, 255), (t_, 0, 255), (255, 0, q)][i]
