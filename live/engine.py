"""The daemon: a render loop in a thread, with knobs anyone can turn.

One :class:`Engine` owns the canvas, the show, the frame clock and the DDP
socket.  It runs in its own thread so the web server cannot starve it -- a late
frame at the rig is a visible glitch, a late HTTP response is not.

Communication with the web layer is deliberately dumb:

* **settings in** -- plain attribute writes on a shared :class:`Settings`.
  Every field is one scalar, so no lock is needed for a single knob; the render
  loop simply sees the new value on its next frame.
* **frames out** -- the engine publishes the last frame it sent by rebinding
  one attribute.  A reader always gets a whole, self-consistent frame because
  the engine writes into a *different* buffer than the one it published.

Nothing here knows about audio yet.  M4 replaces the metronome inside
:class:`live.script.Script` with a real beat clock; the engine does not change.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import effects as fx
from .ddp import DDP_PORT, DDPSender
from .frame import Canvas
from .fseq import FseqWriter
from .layout import Layout, load_layout
from .script import SCENES, Script
from .settings import CHOICES, Settings
from .timing import FrameClock

# The vocabularies the UI offers.  Registered here rather than in settings.py
# because this is where the effect and scene tables actually live.
CHOICES["pattern"] = ["auto", *fx.PATTERNS]
CHOICES["scene"] = ["auto", *(scene.kind for scene in SCENES)]


@dataclass
class EngineStatus:
    running: bool = False
    frames: int = 0
    fps: float = 0.0
    late_frames: int = 0
    skipped_frames: int = 0
    render_ms: float = 0.0
    packets_sent: int = 0
    scene: str = "-"
    pattern: str = "-"
    bpm: float = 0.0
    beat_phase: float = 0.0
    target: str = "-"
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


class Engine:
    def __init__(self, layout: Layout | None = None, settings: Settings | None = None,
                 *, fps: float = 40.0, host: str | None = None, port: int = DDP_PORT,
                 controller: str | None = None, record: Path | None = None,
                 seed: int = 7) -> None:
        self.layout = layout or load_layout()
        self.settings = settings or Settings()
        self.fps = float(fps)
        self.host = host
        self.port = port
        self.record = Path(record) if record else None

        self.canvas = Canvas(self.layout)
        self.script = Script(self.canvas, settings=self.settings, seed=seed)
        self.status = EngineStatus()

        self.span = slice(0, self.layout.channel_count)
        if controller:
            self.span = self.layout.output(controller).slice

        # Two buffers: the loop always fills the one it did not publish, so a
        # reader can never see half of two different frames.
        self._buffers = [self.layout.blank_channels(), self.layout.blank_channels()]
        self._which = 0
        self.frame: np.ndarray = self._buffers[1]

        self._sender: DDPSender | None = None
        self._writer: FseqWriter | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._listeners: list = []

    # -- lifecycle --------------------------------------------------------- #

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("engine already started")
        if self.host:
            self._sender = DDPSender(self.host, port=self.port)
        if self.record:
            self.record.parent.mkdir(parents=True, exist_ok=True)
            self._writer = FseqWriter(self.record, self.layout.channel_count,
                                      step_time_ms=int(round(1000.0 / self.fps)))
        self.status.target = (f"{self.host}:{self.port}" if self.host
                              else "no output")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="render",
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self._sender is not None:
            self._sender.close()
            self._sender = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self.status.running = False

    def __enter__(self) -> "Engine":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- the loop ---------------------------------------------------------- #

    def _run(self) -> None:
        clock = FrameClock(fps=self.fps)
        clock.start()
        self.status.running = True
        render_ms = 0.0
        settings = self.settings

        while not self._stop.is_set():
            index, t = clock.wait()
            started = time.perf_counter()

            out = self._buffers[self._which]
            self.script.render(index, t)
            self.canvas.to_channels(out, brightness=settings.brightness,
                                    gamma=settings.gamma)
            if settings.blackout:
                # Blackout is applied to the *frame*, not the canvas, so the
                # preview shows black too -- an operator hitting it must see
                # the rig go dark, not watch a show that is secretly still lit.
                out[:] = 0

            if self._sender is not None and settings.output_enabled:
                self._sender.send_frame(out[self.span], offset=self.span.start)
            if self._writer is not None:
                self._writer.add_frame(out)

            self.frame = out
            self._which ^= 1
            render_ms += (time.perf_counter() - started) * 1000

            # Every four frames: 10 Hz at 40 fps, which is often enough for
            # the beat bar to look continuous and rare enough to be free.
            if index % 4 == 0:
                self._publish(clock, render_ms / 4.0, t)
                render_ms = 0.0

        self.status.running = False

    def _publish(self, clock: FrameClock, render_ms: float, t: float) -> None:
        _, scene, _ = self.script.locate(t)
        s = self.status
        s.frames = clock.stats.frames
        s.fps = round(clock.stats.actual_fps, 2)
        s.late_frames = clock.stats.late_frames
        s.skipped_frames = clock.stats.skipped_frames
        s.render_ms = round(render_ms, 3)
        s.elapsed_s = round(clock.stats.elapsed_s, 1)
        s.scene = scene.kind
        s.pattern = self.script.pattern_for(scene)
        s.bpm = self.script.bpm
        s.beat_phase = round(self.script.beat_phase(t), 3)
        s.packets_sent = self._sender.packets_sent if self._sender else 0


if __name__ == "__main__":
    engine = Engine()
    with engine:
        time.sleep(2.0)
        print(engine.status.to_dict())
        print(f"frame nonzero channels: {int((engine.frame > 0).sum())}")
