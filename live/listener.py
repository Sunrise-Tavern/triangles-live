"""Audio source -> features -> beats -> clock, on its own thread.

The engine's render loop must never wait on audio, and the audio must never
wait on the render loop, so this owns a thread and publishes state the same way
:class:`live.engine.Engine` publishes frames: by rebinding one attribute.  A
reader always gets a whole, self-consistent :class:`~live.analysis.Features`.

The clock is shared rather than copied -- the render loop calls
``clock.phase(now)`` forty times a second and this thread nudges the same
model when a beat arrives.  That is safe because the render loop only reads,
and it reads scalars that are individually consistent; the worst case is a
frame whose phase comes from a model a millisecond out of date.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .analysis import Analyzer, Features
from .audio import AudioSource, Block
from .beats import BeatEvent, make_backend
from .clock import BeatClock
from .downbeat import BarTracker


@dataclass
class ListenerStats:
    blocks: int = 0
    beats: int = 0
    relocks: int = 0
    analysis_ms: float = 0.0
    #: How far the audio thread is behind the wall clock.  Rising means the
    #: analysis cannot keep up, which on a Pi is the first thing to check.
    lag_ms: float = 0.0
    running: bool = False

    def as_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


class Listener:
    def __init__(self, source: AudioSource, *, backend: str = "aubio",
                 clock: BeatClock | None = None, backend_kwargs: dict | None = None,
                 on_beat=None, bars: BarTracker | None = None,
                 window: int = 2048, silence_dbfs: float = -70.0) -> None:
        self.source = source
        self.analyzer = Analyzer(source.samplerate, source.blocksize,
                                 window=window, silence_dbfs=silence_dbfs)
        self.backend = make_backend(
            backend,
            **{"samplerate": source.samplerate, "blocksize": source.blocksize,
               "window": window, **(backend_kwargs or {})}
            if backend == "aubio" else (backend_kwargs or {}),
        )
        self.clock = clock or BeatClock()
        self.bars = bars or BarTracker()
        self.stats = ListenerStats()
        self.features: Features | None = None
        self.on_beat = on_beat
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        #: Show time of the newest analysed audio.  The render loop uses this
        #: rather than wall time, so a file streamed faster than real time and
        #: a live input behave identically.
        self.audio_time = 0.0

    # -- one block, the whole chain ---------------------------------------- #

    def step(self, block: Block) -> Features:
        started = time.perf_counter()
        features = self.analyzer.push(block)
        event = self.backend.push(block, features)
        # The clock resolves half-beat slips from where the low end lands, so
        # it needs every block, not just the ones with a beat on them.
        self.clock.observe(features.t, features.kick)
        self.clock.tick(features.t)
        self.bars.push(features, self.clock)
        if event is not None:
            self.stats.beats += 1
            if self.clock.on_beat(event) and self.on_beat is not None:
                self.on_beat(event)
        self.features = features
        self.audio_time = features.t
        self.stats.blocks += 1
        self.stats.relocks = self.clock.relocks
        elapsed = (time.perf_counter() - started) * 1000
        self.stats.analysis_ms += (elapsed - self.stats.analysis_ms) * 0.05
        return features

    def run(self) -> None:
        """Consume the source until it ends or :meth:`stop` is called."""
        self.stats.running = True
        started = time.perf_counter()
        try:
            for block in self.source.blocks():
                if self._stop.is_set():
                    break
                self.step(block)
                self.stats.lag_ms = (time.perf_counter() - started
                                     - self.audio_time) * 1000
        finally:
            self.stats.running = False
            self.source.close()

    # -- lifecycle --------------------------------------------------------- #

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("listener already started")
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, name="listen", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def __enter__(self) -> "Listener":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
