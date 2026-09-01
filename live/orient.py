"""Orientation check: does "up" on the rig mean up, and "front" mean front?

Three stages, looping, each a single bright band you can follow with your
eyes.  Each one answers one question about the mapping, and a wrong answer
points at one place:

=========  =========================  ======================================
stage      what you should see        what a failure means
=========  =========================  ======================================
nets up    on every net at once, one  the band falls, or runs sideways ->
           band rising from the base  that net's custom model is mounted
           to the apex                upside down / rotated in xLights
big up     one band rising across     the four nets rise out of step, or
           the four big nets as one   the middle one is wrong -> the
           triangle -- the two base   world positions in xLights do not
           nets first, then the       match how the big triangle is hung
           inverted middle, the top
           net last
tunnel     one arch lit at a time,    arches light out of order -> the
           front of the corridor to   Tunnel group is not in physical
           the back                   order
=========  =========================  ======================================

The band is white with a dim blue floor under it, so a net that is present
but never crossed by the band still shows as *there*.  Pure function of
time, like :mod:`live.testpattern`.
"""

from __future__ import annotations

import numpy as np

from .frame import Canvas

#: (name, seconds)
STAGES: tuple[tuple[str, float], ...] = (
    ("nets up", 5.0),
    ("big up", 5.0),
    ("tunnel", 6.0),
)
DURATION = sum(seconds for _, seconds in STAGES)

FLOOR = 0.06
BAND = 0.07


def stage_at(t: float) -> tuple[str, float]:
    t = t % DURATION
    for name, seconds in STAGES:
        if t < seconds:
            return name, t / seconds
        t -= seconds
    return STAGES[-1][0], 1.0


def hold(stage: str, t: float) -> float:
    """Map wall time onto one stage, looping it, so it can be held."""
    offset = 0.0
    for name, seconds in STAGES:
        if name == stage:
            return offset + (t % seconds)
        offset += seconds
    raise ValueError(f"unknown stage {stage!r}")


def _band(coord: np.ndarray, at: float, width: float = BAND) -> np.ndarray:
    return np.clip(1.0 - np.abs(coord - at) / width, 0.0, 1.0)


def paint_one_arch(canvas: Canvas, index: int) -> None:
    """Arch ``index`` (0-based, tunnel front-to-back order) solid white,
    every other arch a dim blue -- for walking the corridor and checking
    which physical arch answers to which position."""
    canvas.clear()
    canvas.arches[..., 2] = FLOOR
    canvas.arches[index] = 1.0


def paint(canvas: Canvas, t: float) -> str:
    """Paint the orientation pattern for show time ``t``; returns the stage."""
    canvas.clear()
    name, progress = stage_at(t)
    # Hold each end for a moment so the start and finish are unmistakable.
    run = float(np.clip((progress - 0.08) / 0.84, 0.0, 1.0))

    if name == "nets up":
        canvas.nets[..., 2] = FLOOR
        level = _band(canvas.net_y, 1.0 - run)          # y 1 = base, 0 = apex
        canvas.nets += level[..., None]
        canvas.arches[..., 2] = FLOOR * 0.5
    elif name == "big up":
        canvas.nets[..., 2] = FLOOR
        level = _band(canvas.big_geo.y, 1.0 - run)
        canvas.nets[canvas.big] += level[..., None]
        canvas.arches[..., 2] = FLOOR * 0.5
    elif name == "tunnel":
        canvas.nets[..., 2] = FLOOR * 0.5
        n = canvas.arches.shape[0]
        index = np.arange(n, dtype=np.float32) / max(1, n - 1)
        level = _band(index, run, width=1.0 / max(1, n - 1))
        canvas.arches[..., 2] = FLOOR
        canvas.arches += level[:, None, None]
    np.clip(canvas._source, 0.0, 1.0, out=canvas._source)
    return name
