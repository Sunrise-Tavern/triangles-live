"""A drift-free frame clock.

``time.sleep(1 / fps)`` in a loop accumulates every scheduling overshoot, so a
40 fps render drifts seconds over a set -- which is fatal here, because the
whole design bets on firing effects at a *predicted* beat.  This clock sleeps
until an absolute deadline computed from the start time, so late frames are
absorbed rather than compounded.

It reports lateness instead of hiding it: :attr:`FrameClock.late_frames` and
:meth:`stats` are how M6's "sustained 40 fps for 10 minutes" gets measured, and
how the Pi tells us it has run out of headroom.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ClockStats:
    frames: int = 0
    late_frames: int = 0
    skipped_frames: int = 0
    worst_late_ms: float = 0.0
    total_late_ms: float = 0.0
    elapsed_s: float = 0.0

    @property
    def mean_late_ms(self) -> float:
        return self.total_late_ms / self.frames if self.frames else 0.0

    @property
    def actual_fps(self) -> float:
        return self.frames / self.elapsed_s if self.elapsed_s > 0 else 0.0

    def summary(self) -> str:
        return (
            f"{self.frames} frames in {self.elapsed_s:.1f}s "
            f"({self.actual_fps:.1f} fps), late {self.late_frames} "
            f"(mean {self.mean_late_ms:.1f} ms, worst {self.worst_late_ms:.1f} ms), "
            f"skipped {self.skipped_frames}"
        )


@dataclass
class FrameClock:
    """Iterate to get one tick per frame period.

    ``for frame_index, t in clock: ...`` yields the frame number and the
    *scheduled* show time in seconds -- scheduled, not measured, so anything
    derived from it (beat phase, effect position) stays on the nominal grid
    even when a frame runs long.
    """

    fps: float = 40.0
    #: Beyond this much lateness, give up on the missed frames and resync
    #: rather than sprinting to catch up.
    max_catchup_s: float = 0.25
    stats: ClockStats = field(default_factory=ClockStats)
    _start: float = 0.0
    _frame: int = 0

    @property
    def period(self) -> float:
        return 1.0 / self.fps

    def start(self, at: float | None = None) -> None:
        self._start = at if at is not None else time.perf_counter()
        self._frame = 0
        self.stats = ClockStats()

    def wait(self) -> tuple[int, float]:
        """Block until the next frame is due.  Returns (frame index, show time)."""
        if self._start == 0.0:
            self.start()

        deadline = self._start + self._frame * self.period
        now = time.perf_counter()
        late = now - deadline

        if late > self.max_catchup_s:
            # Hopelessly behind (a stall, a suspend).  Jump the frame counter
            # to now so show time stays wall-clock accurate; count the loss.
            missed = int(late / self.period)
            self._frame += missed
            self.stats.skipped_frames += missed
            deadline = self._start + self._frame * self.period
            late = now - deadline
        elif late < 0:
            time.sleep(-late)
            late = 0.0

        if late > 0:
            self.stats.late_frames += 1
            self.stats.total_late_ms += late * 1000.0
            self.stats.worst_late_ms = max(self.stats.worst_late_ms, late * 1000.0)

        # frame / fps, never frame * period: the two disagree in the last bit
        # of a float, which is enough to flip an 8-bit level by one and make a
        # capture differ from the same content rendered offline.
        index, show_time = self._frame, self._frame / self.fps
        self._frame += 1
        self.stats.frames += 1
        self.stats.elapsed_s = time.perf_counter() - self._start
        return index, show_time

    def __iter__(self):
        self.start()
        while True:
            yield self.wait()


class Ticker:
    """Fixed-count version of :class:`FrameClock`, for scripted runs.

    ``for i, t in Ticker(fps=40, seconds=30): ...`` is the shape every
    verification script in the plan uses -- a fixed 30 s of frames, rendered
    and captured, then opened in xLights.
    """

    def __init__(self, fps: float = 40.0, seconds: float = 30.0, realtime: bool = True):
        self.clock = FrameClock(fps=fps)
        self.frames = int(round(fps * seconds))
        self.realtime = realtime

    def __iter__(self):
        self.clock.start()
        for i in range(self.frames):
            if self.realtime:
                yield self.clock.wait()
            else:
                yield i, i / self.clock.fps
