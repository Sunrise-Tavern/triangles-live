"""Beat detection, behind an interface thin enough to swap.

``BeatBackend`` exists so the tracker is a *choice*, not a load-bearing
assumption.  aubio is the default -- it is C, it streams, and Debian packages
it for the Pi.  The plan reserves the right to A/B it against BeatNet or
BTrack later, and that only stays cheap if nothing downstream knows which one
is running.  :class:`live.clock.BeatClock` consumes :class:`BeatEvent` and
nothing else.

Installing aubio is the one rough edge: the 2019 release needs two build flags
(see ``setup.sh``).  If it is missing, :func:`make_backend` says so plainly
rather than failing at the first block of a set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .analysis import Features
from .audio import BLOCKSIZE, SAMPLERATE, Block


@dataclass
class BeatEvent:
    #: When the beat happened, in show time -- *not* when it was noticed.
    t: float
    tempo: float
    confidence: float
    source: str = ""


class BeatBackend(Protocol):
    name: str

    def push(self, block: Block, features: Features) -> BeatEvent | None: ...
    def reset(self) -> None: ...


class AubioBackend:
    """aubio's tempo tracker, one call per hop.

    ``get_last_s()`` is the important part: aubio reports *when the beat was*,
    which is a little in the past by the time we hear about it.  Passing that
    through rather than the arrival time is what keeps the clock's phase
    estimate honest -- the detection lag becomes a known quantity instead of a
    silent bias.
    """

    name = "aubio"

    def __init__(self, samplerate: int = SAMPLERATE, blocksize: int = BLOCKSIZE,
                 window: int = 2048, method: str = "default",
                 threshold: float | None = None, silence_db: float = -70.0) -> None:
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.window = window
        self.method = method
        self.threshold = threshold
        self.silence_db = silence_db
        self._tempo = None
        self.reset()

    def reset(self) -> None:
        try:
            import aubio
        except ImportError as exc:                  # pragma: no cover
            raise RuntimeError(
                "aubio is not installed.  Run ./setup.sh (it needs two build "
                "flags), or on a Pi: sudo apt install python3-aubio."
            ) from exc
        self._tempo = aubio.tempo(self.method, self.window, self.blocksize,
                                  self.samplerate)
        self._tempo.set_silence(self.silence_db)
        if self.threshold is not None:
            self._tempo.set_threshold(self.threshold)

    def push(self, block: Block, features: Features) -> BeatEvent | None:
        samples = block.samples
        if len(samples) != self.blocksize:
            raise ValueError(f"aubio wants exactly {self.blocksize} samples per hop")
        hit = self._tempo(np.ascontiguousarray(samples, dtype=np.float32))
        if not hit[0]:
            return None
        # get_last_s() counts from the start of the stream, which is the same
        # origin as Block.t -- both start at zero when the source does.
        return BeatEvent(
            t=float(self._tempo.get_last_s()),
            tempo=float(self._tempo.get_bpm()),
            confidence=float(self._tempo.get_confidence()),
            source=self.name,
        )


class MetronomeBackend:
    """A perfect tracker, for testing everything downstream of detection.

    When a clock test fails with this backend the bug is in the clock; when it
    only fails with aubio the bug is in detection or in the audio.  Keeping
    those separable is worth the twenty lines.
    """

    name = "metronome"

    def __init__(self, bpm: float = 128.0, offset: float = 0.0,
                 jitter: float = 0.0, seed: int = 0) -> None:
        self.bpm = bpm
        self.offset = offset
        self.jitter = jitter
        self._rng = np.random.default_rng(seed)
        self._next = offset

    def reset(self) -> None:
        self._next = self.offset
        self._rng = np.random.default_rng(0)

    def push(self, block: Block, features: Features) -> BeatEvent | None:
        period = 60.0 / self.bpm
        end = block.t + len(block.samples) / SAMPLERATE
        if end < self._next:
            return None
        t = self._next
        self._next += period
        if self.jitter:
            t += float(self._rng.normal(0.0, self.jitter))
        return BeatEvent(t=t, tempo=self.bpm, confidence=1.0, source=self.name)


BACKENDS = {"aubio": AubioBackend, "metronome": MetronomeBackend}


def make_backend(name: str = "aubio", **kwargs) -> BeatBackend:
    if name not in BACKENDS:
        raise ValueError(f"unknown beat backend {name!r}; have {sorted(BACKENDS)}")
    return BACKENDS[name](**kwargs)
