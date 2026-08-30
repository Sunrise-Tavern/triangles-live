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
from . import palette as pal
from .arranger import Arranger
from .audio import AudioSource
from .ddp import DDP_PORT, DDPSender
from .listener import Listener
from .state import STATES, StateMachine, StateThresholds
from .frame import Canvas
from .fseq import FseqWriter
from .layout import Layout, load_layout
from .script import SCENES, Script
from .settings import CHOICES, Settings
from .timing import FrameClock

# The vocabularies the UI offers.  Registered here rather than in settings.py
# because this is where the effect and scene tables actually live.
CHOICES["pattern"] = ["auto", *fx.PATTERNS]
CHOICES["scheme"] = ["auto", *pal.SCHEMES]
# With audio the "scene" knob holds a *state*; without it, a script scene.
CHOICES["scene"] = ["auto", *STATES, *(scene.kind for scene in SCENES)]


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
    # -- only meaningful when driven by audio --
    audio: bool = False
    confidence: float = 0.0
    bar_phase: float = 0.0
    bar: int = 0
    bar_confidence: float = 0.0
    locked: bool = False
    free_running: bool = False
    energy: float = 0.0
    level: float = 0.0
    audio_lag_ms: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class Sink:
    """One DDP receiver and the slice of the show it is responsible for."""

    host: str
    port: int
    span: slice
    name: str = ""
    sender: DDPSender | None = None

    def __str__(self) -> str:
        return f"{self.name or self.host}@{self.host}:{self.port}"


class Engine:
    def __init__(self, layout: Layout | None = None, settings: Settings | None = None,
                 *, fps: float = 40.0, host: str | None = None, port: int = DDP_PORT,
                 controller: str | None = None, record: Path | None = None,
                 seed: int = 7, audio: AudioSource | None = None,
                 backend: str = "aubio",
                 thresholds: StateThresholds | None = None,
                 window: int = 2048, silence_dbfs: float = -70.0,
                 session=None) -> None:
        self.layout = layout or load_layout()
        self.settings = settings or Settings()
        self.fps = float(fps)
        self.host = host
        self.port = port
        self.record = Path(record) if record else None

        self.canvas = Canvas(self.layout)
        self.audio = audio
        self.listener: Listener | None = None
        self.machine: StateMachine | None = None
        if audio is not None:
            self.listener = Listener(audio, backend=backend, window=window,
                                     silence_dbfs=silence_dbfs)
            self.machine = StateMachine(thresholds)
            self.script = Arranger(self.canvas, self.listener,
                                   settings=self.settings, state=self.machine,
                                   seed=seed)
        else:
            self.script = Script(self.canvas, settings=self.settings, seed=seed)
        self.session = session
        self.status = EngineStatus(audio=audio is not None)

        # Where frames go.  The show spans two Falcons -- nets on one,
        # corridor on the other -- so this is a list, not a host.
        #   host="auto"   every DDP controller in xlights_networks.xml
        #   host=<addr>   that one address, clipped by `controller` if given
        #   host=""       nowhere; render only, which is the laptop default
        self.targets: list[Sink] = []
        if host and host.lower() == "auto":
            if controller:
                picks = [self.layout.output(controller)]
            else:
                picks = self.layout.ddp_targets()
            self.targets = [Sink(c.ip, port, c.slice, c.name) for c in picks]
        elif host:
            span = (self.layout.output(controller).slice if controller
                    else slice(0, self.layout.channel_count))
            self.targets = [Sink(host, port, span, controller or "all")]
        self.status.target = (", ".join(str(t) for t in self.targets)
                              if self.targets else "no output")

        # Two buffers: the loop always fills the one it did not publish, so a
        # reader can never see half of two different frames.
        self._buffers = [self.layout.blank_channels(), self.layout.blank_channels()]
        self._which = 0
        self.frame: np.ndarray = self._buffers[1]

        self._writer: FseqWriter | None = None
        self._thread: threading.Thread | None = None
        self._audio_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._listeners: list = []

    # -- lifecycle --------------------------------------------------------- #

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("engine already started")
        for sink in self.targets:
            sink.sender = DDPSender(sink.host, port=sink.port)
        if self.record:
            self.record.parent.mkdir(parents=True, exist_ok=True)
            self._writer = FseqWriter(self.record, self.layout.channel_count,
                                      step_time_ms=int(round(1000.0 / self.fps)))
        self._stop.clear()
        if self.listener is not None:
            # Audio first: the render loop's show time is the *audio's*
            # timeline, because that is the timeline the beat clock's anchor
            # lives on.  Starting the frame clock from the moment audio began
            # keeps the two within a block of each other.
            self.listener.on_beat = self._on_beat
            self._audio_thread = threading.Thread(
                target=self._pump_audio, name="listen", daemon=True)
            self._audio_thread.start()
        self._thread = threading.Thread(target=self._run, name="render",
                                        daemon=True)
        self._thread.start()

    def _on_beat(self, event) -> None:
        # Accepted beats only -- the listener calls this after the clock has
        # fitted the event -- so the recording shows what the grid was built
        # from, not everything the tracker said.
        if self.session is not None:
            clock = self.listener.clock
            self.session.beat(event.t, clock.tempo, clock.confidence)

    def _pump_audio(self) -> None:
        """Analysis on its own thread; the render loop only ever reads."""
        assert self.listener is not None and self.machine is not None
        self.listener.stats.running = True
        try:
            for block in self.listener.source.blocks():
                if self._stop.is_set():
                    break
                features = self.listener.step(block)
                self.machine.push(features)
                if self.session is not None:
                    self.session.observe(block, features, self.listener.clock,
                                         self.machine)
        finally:
            self.listener.source.close()
            self.listener.stats.running = False

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self._audio_thread is not None:
            self._audio_thread.join(timeout=timeout)
            self._audio_thread = None
        for sink in self.targets:
            if sink.sender is not None:
                sink.sender.close()
                sink.sender = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self.session is not None:
            self.session.close()
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

            if self.machine is not None:
                # The panel's state knobs, applied live.  Copied rather than
                # shared so the state machine keeps working with no UI at all.
                self.machine.t.quiet_enter = settings.quiet_enter
                self.machine.t.build_high_share = settings.build_high_share
                self.machine.t.drop_kick = settings.drop_kick
            if self.listener is not None:
                self.listener.clock.latency = settings.latency_ms / 1000.0

            out = self._buffers[self._which]
            self.script.render(index, t)
            self.canvas.to_channels(out, brightness=settings.brightness,
                                    gamma=settings.gamma)
            if settings.blackout:
                # Blackout is applied to the *frame*, not the canvas, so the
                # preview shows black too -- an operator hitting it must see
                # the rig go dark, not watch a show that is secretly still lit.
                out[:] = 0

            if settings.output_enabled:
                for sink in self.targets:
                    if sink.sender is not None:
                        # Offset 0, not span.start: a DDP offset addresses the
                        # *receiver's* own channel space, which starts at its
                        # first channel however far into the show that is.
                        # Sending the corridor at 11160 puts every packet past
                        # the end of a controller that owns 25 920 channels,
                        # and it drops them without a word.
                        sink.sender.send_frame(out[sink.span])
            if self._writer is not None:
                self._writer.add_frame(out)

            if self.session is not None:
                phase = (self.listener.clock.phase(t)
                         if self.listener is not None else -1.0)
                self.session.frame(index, t, out, self.layout, phase)
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
        if self.listener is not None and self.machine is not None:
            beat = self.listener.clock
            state = beat.state(t)
            s.confidence = round(state.confidence, 3)
            s.bar_phase = round(state.bar_phase, 3)
            s.bar = state.bar
            s.locked = state.locked
            s.free_running = state.free_running
            s.bar_confidence = round(self.listener.bars.confidence, 3)
            s.audio_lag_ms = round(self.listener.stats.lag_ms, 1)
            report = self.machine.report
            s.energy = round(report.energy, 3)
            s.reason = report.reason or s.reason
            if self.listener.features is not None:
                s.level = round(self.listener.features.level, 3)
        s.frames = clock.stats.frames
        s.fps = round(clock.stats.actual_fps, 2)
        s.late_frames = clock.stats.late_frames
        s.skipped_frames = clock.stats.skipped_frames
        s.render_ms = round(render_ms, 3)
        s.elapsed_s = round(clock.stats.elapsed_s, 1)
        s.scene = scene.kind
        s.pattern = self.script.pattern_for(scene,
                                            self.script.phrase_index(t, scene))
        s.bpm = self.script.bpm
        s.beat_phase = round(self.script.beat_phase(t), 3)
        s.packets_sent = sum(t.sender.packets_sent for t in self.targets
                             if t.sender is not None)


if __name__ == "__main__":
    engine = Engine()
    with engine:
        time.sleep(2.0)
        print(engine.status.to_dict())
        print(f"frame nonzero channels: {int((engine.frame > 0).sum())}")
