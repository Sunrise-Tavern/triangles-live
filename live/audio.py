"""Audio in, one interface, two sources.

The rig hears a line from the XR16 through a USB interface; the Mac hears an
mp3 played back in real time.  Nothing downstream is told which -- the whole
point is that a simulation run and a show run exercise the same code, so a bug
found on the sofa is the bug that would have happened at the gig.

Blocks are 512 samples at 44.1 kHz -- 11.6 ms, and the hop size aubio's tempo
tracker wants.  Each block carries the show time of its **first sample**, not
the time it was handed over, so analysis timestamps mean the same thing whether
the audio arrived from a sound card or from a file being read faster than real
time.

The file source decodes through ffmpeg rather than a Python decoder: it is
already a hard dependency of the offline generator, it handles anything the DJ
might hand us, and it streams instead of loading a whole track into memory.
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

SAMPLERATE = 44100
BLOCKSIZE = 512


@dataclass
class Block:
    """One hop of mono audio."""

    samples: np.ndarray     # float32, length blocksize
    t: float                # show time of the first sample, seconds
    index: int
    #: What :class:`AutoGain` multiplied these samples by, so a consumer can
    #: recover the level at the *input*.  Every relative measure downstream
    #: wants the gained samples -- that is the point of the gain -- but the
    #: one absolute measure, the silence gate, must not move when the gain
    #: does, or it is testing a different threshold every minute.
    gain: float = 1.0

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.samples.astype(np.float64) ** 2)))

    @property
    def input_rms(self) -> float:
        """RMS as it arrived at the sound card, before any auto-gain."""
        return self.rms / max(self.gain, 1e-9)


class AudioSource(Protocol):
    samplerate: int
    blocksize: int

    def blocks(self) -> Iterator[Block]: ...
    def close(self) -> None: ...


class AutoGain:
    """Track a very slow peak and scale toward a target level.

    Its job is the DJ's gain knob, which drifts over **minutes** -- not musical
    dynamics, which move over seconds.  Getting that wrong is not a small
    mis-tuning: a compressor-like 0.25 s attack actively inverts the thing the
    state machine reads, because a loud passage pulls the gain down and the
    measured level *falls*.  Measured on a track with a real drop, that took
    the drop-to-breakdown contrast from 4.6 to 2.7 and dropped the show out of
    `hot` twenty-three seconds early; on a live input it reached `quiet` while
    the music was still loud.

    At 60 s attack / 180 s release the contrast is 4.6, indistinguishable from
    no auto-gain at all, while a 12 dB drift across a track is still corrected.

    Worth knowing: the analysis does not actually *need* this. `energy` is
    already relative to a 45-second baseline, so it handles drift on its own --
    measured, 4.4 with auto-gain off on the same drifted signal. This exists to
    keep a very quiet feed clear of the noise floor, not to help the analysis.
    """

    def __init__(self, target: float = 0.25, attack: float = 60.0,
                 release: float = 180.0, ceiling: float = 12.0,
                 block_s: float = BLOCKSIZE / SAMPLERATE) -> None:
        self.target = target
        self.gain = 1.0
        self.peak = target
        self._up = 1.0 - np.exp(-block_s / attack)
        self._down = 1.0 - np.exp(-block_s / release)
        self.ceiling = ceiling

    def apply(self, samples: np.ndarray) -> np.ndarray:
        level = float(np.abs(samples).max())
        alpha = self._up if level > self.peak else self._down
        self.peak += (level - self.peak) * alpha
        if self.peak > 1e-5:
            self.gain = min(self.ceiling, self.target / self.peak)
        return samples * self.gain


class FileSource:
    """Stream a file through ffmpeg, optionally paced to real time.

    ``realtime=False`` runs as fast as the CPU allows, which is what the
    verification harness wants: comparing 90 seconds of predicted beats against
    ground truth should take a second, not ninety.
    """

    def __init__(self, path: str | Path, *, samplerate: int = SAMPLERATE,
                 blocksize: int = BLOCKSIZE, realtime: bool = True,
                 start: float = 0.0, gain: AutoGain | None = None,
                 loop: bool = False) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.realtime = realtime
        self.start = start
        self.gain = gain
        self.loop = loop
        self._proc: subprocess.Popen | None = None

    def _spawn(self) -> subprocess.Popen:
        return subprocess.Popen(
            ["ffmpeg", "-v", "quiet", "-nostdin",
             "-ss", f"{self.start:.3f}", "-i", str(self.path),
             "-f", "f32le", "-ac", "1", "-ar", str(self.samplerate), "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

    def blocks(self) -> Iterator[Block]:
        want = self.blocksize * 4          # float32
        index = 0
        started = time.perf_counter()
        while True:
            self._proc = self._spawn()
            assert self._proc.stdout is not None
            while True:
                raw = self._proc.stdout.read(want)
                if len(raw) < want:
                    break
                samples = np.frombuffer(raw, dtype=np.float32).copy()
                if self.gain is not None:
                    samples = self.gain.apply(samples)
                    applied = self.gain.gain
                else:
                    applied = 1.0
                t = index * self.blocksize / self.samplerate
                if self.realtime:
                    # Absolute deadline, so a slow consumer cannot make the
                    # stream drift away from its own timestamps.
                    delay = started + t - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                yield Block(samples=samples, t=t, index=index, gain=applied)
                index += 1
            self.close()
            if not self.loop:
                return
            started = time.perf_counter() - index * self.blocksize / self.samplerate

    def close(self) -> None:
        if self._proc is not None:
            if self._proc.stdout is not None:
                self._proc.stdout.close()
            self._proc.terminate()
            self._proc.wait(timeout=2)
            self._proc = None


class ArraySource:
    """Stream an in-memory signal.  Fault injection and unit tests."""

    def __init__(self, samples: np.ndarray, *, samplerate: int = SAMPLERATE,
                 blocksize: int = BLOCKSIZE, realtime: bool = False,
                 gain: AutoGain | None = None) -> None:
        self.samples = np.asarray(samples, dtype=np.float32)
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.realtime = realtime
        self.gain = gain

    def blocks(self) -> Iterator[Block]:
        started = time.perf_counter()
        total = len(self.samples) // self.blocksize
        for index in range(total):
            chunk = self.samples[index * self.blocksize:(index + 1) * self.blocksize]
            if self.gain is not None:
                chunk = self.gain.apply(chunk)
            t = index * self.blocksize / self.samplerate
            if self.realtime:
                delay = started + t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            yield Block(samples=chunk.copy(), t=t, index=index)

    def close(self) -> None:
        pass


class LineInSource:
    """The rig: a USB interface's line input, via PortAudio.

    The callback runs on PortAudio's thread and must not block, so it only
    drops a block into a queue.  A full queue means the consumer has stalled;
    the oldest block is discarded rather than the newest, because stale audio
    is worthless to a tracker that is trying to predict the *next* beat.
    """

    def __init__(self, device: int | str | None = None, *,
                 samplerate: int = SAMPLERATE, blocksize: int = BLOCKSIZE,
                 channel: int = 0, gain: AutoGain | None = None,
                 depth: int = 32) -> None:
        self.device = device
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channel = channel
        self.gain = gain if gain is not None else AutoGain()
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=depth)
        self._stream = None
        self.overruns = 0
        self._stop = threading.Event()

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            self.overruns += 1
        mono = indata[:, min(self.channel, indata.shape[1] - 1)].copy()
        try:
            self._queue.put_nowait(mono)
        except queue.Full:
            self.overruns += 1
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(mono)
            except (queue.Empty, queue.Full):
                pass

    def blocks(self) -> Iterator[Block]:
        import sounddevice as sd

        self._stream = sd.InputStream(
            device=self.device, channels=1, samplerate=self.samplerate,
            blocksize=self.blocksize, dtype="float32", callback=self._callback,
        )
        self._stream.start()
        index = 0
        while not self._stop.is_set():
            try:
                samples = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            applied = 1.0
            if self.gain is not None:
                samples = self.gain.apply(samples)
                applied = self.gain.gain
            yield Block(samples=samples, t=index * self.blocksize / self.samplerate,
                        index=index, gain=applied)
            index += 1

    def close(self) -> None:
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def devices() -> str:
    """Human-readable list of capture devices, for setting up at the rig."""
    try:
        import sounddevice as sd
    except Exception as exc:                        # noqa: BLE001
        return f"sounddevice unavailable: {exc}"
    lines = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            lines.append(f"  [{i}] {dev['name']}  "
                         f"{dev['max_input_channels']} in @ "
                         f"{dev['default_samplerate']:.0f} Hz")
    return "\n".join(lines) or "  (no input devices)"


if __name__ == "__main__":
    print("input devices:")
    print(devices())
