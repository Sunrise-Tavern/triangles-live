"""Record a whole session: the audio, what we made of it, and what we lit.

Three rounds of guessing at a detection problem is three rounds too many.  The
fix for that is not a better guess, it is a recording -- the exact audio that
went in, every number the analysis derived from it, every decision the state
machine made and why, and a summary of what the lights actually did, all on one
timeline.

The audio is the important part: with it, the entire pipeline can be replayed
offline, deterministically, as many times as it takes.  The traces matter
because a replay only proves what the code does *now*, and the question is
usually what it did *then* -- with that config, that device, that gain.

    ./live.sh serve --audio-device 2 --record-session sessions/friday
    ./live.sh analyze sessions/friday
    ./live.sh analyze sessions/friday --replay      # rerun the chain on the audio

Written to be cheap enough to leave on: int16 audio is 5 MB a minute, the
per-block trace about 1 MB, and the frame summary a few hundred KB.  A four
hour set is under 1.5 GB, which is worth it the first time something is wrong
and nobody can say what the input was.
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SAMPLERATE = 44100
#: Per-block numbers worth keeping.  Everything the state machine reads, plus
#: the raw levels needed to second-guess it.
FEATURE_FIELDS = ("t", "rms", "peak", "energy", "level", "bass", "mid", "high",
                  "bass_share", "high_share", "flux", "onset", "kick",
                  "novelty", "centroid", "baseline", "warm", "silent",
                  "dbfs", "near_floor")
CLOCK_FIELDS = ("tempo", "confidence", "beat", "bar", "beat_phase", "bar_phase",
                "locked", "free_running", "since_beat")
STATE_FIELDS = ("energy", "slope", "brightness")


class WavWriter:
    """16-bit mono WAV, written by hand.

    No soundfile dependency: the live engine deliberately does not install the
    offline generator's stack, and a recorder that only works on the dev
    machine is no use at the rig.
    """

    def __init__(self, path: Path, samplerate: int = SAMPLERATE) -> None:
        self.path = Path(path)
        self.samplerate = samplerate
        self.frames = 0
        self._fh = self.path.open("wb")
        self._fh.write(b"\0" * 44)          # header patched on close

    def write(self, samples: np.ndarray) -> None:
        clipped = np.clip(samples, -1.0, 1.0)
        self._fh.write((clipped * 32767.0).astype("<i2").tobytes())
        self.frames += len(clipped)

    def close(self) -> None:
        if self._fh.closed:
            return
        data = self.frames * 2
        self._fh.seek(0)
        self._fh.write(b"RIFF" + struct.pack("<I", 36 + data) + b"WAVEfmt "
                       + struct.pack("<IHHIIHH", 16, 1, 1, self.samplerate,
                                     self.samplerate * 2, 2, 16)
                       + b"data" + struct.pack("<I", data))
        self._fh.close()


@dataclass
class SessionMeta:
    started: str = ""
    duration_s: float = 0.0
    blocks: int = 0
    frames: int = 0
    samplerate: int = SAMPLERATE
    blocksize: int = 512
    fps: float = 40.0
    config: dict = field(default_factory=dict)
    transitions: list = field(default_factory=list)
    beats: list = field(default_factory=list)
    notes: list = field(default_factory=list)


class Recorder:
    """Writes a session bundle.  Safe to call from the audio and render threads."""

    def __init__(self, directory: str | Path, *, config=None, fps: float = 40.0,
                 blocksize: int = 512, max_minutes: float = 240.0,
                 frame_stride: int = 4) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = SessionMeta(
            started=time.strftime("%Y-%m-%d %H:%M:%S"), fps=fps,
            blocksize=blocksize,
            config=config.to_dict() if config is not None else {},
        )
        self.limit_blocks = int(max_minutes * 60 * SAMPLERATE / blocksize)
        #: Frames are summarised every Nth, because forty a second of
        #: per-model brightness is a lot of rows for something only ever read
        #: as a curve.
        self.frame_stride = frame_stride
        self._wav = WavWriter(self.dir / "audio.wav")
        self._features: list[tuple] = []
        self._clock: list[tuple] = []
        self._state: list[tuple] = []
        self._frames: list[tuple] = []
        self._closed = False

    # -- called from the audio thread -------------------------------------- #

    def observe(self, block, features, clock, machine) -> None:
        if self._closed or self.meta.blocks >= self.limit_blocks:
            return
        self._wav.write(block.samples)
        self._features.append(tuple(
            float(getattr(features, name)) for name in FEATURE_FIELDS))
        state = clock.state(features.t)
        self._clock.append(tuple(float(getattr(state, name))
                                 for name in CLOCK_FIELDS))
        report = machine.report
        self._state.append((STATES_INDEX.get(report.state, -1),)
                           + tuple(float(getattr(report, n)) for n in STATE_FIELDS))
        if report.changed:
            self.meta.transitions.append(
                [round(features.t, 3), report.state, report.reason])
        self.meta.blocks += 1
        self.meta.duration_s = features.t

    def beat(self, when: float, tempo: float, confidence: float) -> None:
        self.meta.beats.append([round(when, 4), round(tempo, 2),
                                round(confidence, 3)])

    # -- called from the render thread ------------------------------------- #

    def frame(self, index: int, t: float, channels: np.ndarray, layout,
              beat_phase: float = -1.0) -> None:
        """One rendered frame's summary.

        ``beat_phase`` is recorded because inferring "did the lights fire on
        the beat" from a brightness curve does not work: at every-4th-frame
        sampling the curve resolves to about 100 ms against a 470 ms beat, so
        the answer comes out barely above chance whatever the truth is.  With
        the phase logged the question is direct.
        """
        if self._closed or index % self.frame_stride:
            return
        # A brightness curve per fixture family is enough to see whether the
        # lights moved when the music did; the full frames are what --record
        # writes to an fseq if that is ever needed.
        self._frames.append((
            round(t, 4),
            float(channels.mean()),
            float(channels[layout.nets_slice].mean()) if layout.nets_slice else 0.0,
            float(channels[layout.arches_slice].mean()) if layout.arches_slice else 0.0,
            float(beat_phase),
        ))
        self.meta.frames += 1

    # -- lifecycle --------------------------------------------------------- #

    def note(self, text: str) -> None:
        self.meta.notes.append(text)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._wav.close()
        np.savez_compressed(
            self.dir / "trace.npz",
            features=np.array(self._features, dtype=np.float32),
            features_fields=np.array(FEATURE_FIELDS),
            clock=np.array(self._clock, dtype=np.float32),
            clock_fields=np.array(CLOCK_FIELDS),
            state=np.array(self._state, dtype=np.float32),
            state_fields=np.array(("state",) + STATE_FIELDS),
            frames=np.array(self._frames, dtype=np.float32),
            frames_fields=np.array(("t", "mean", "nets", "arches", "beat_phase")),
        )
        (self.dir / "session.json").write_text(
            json.dumps(asdict(self.meta), indent=2) + "\n")

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


from .state import STATES                                    # noqa: E402
STATES_INDEX = {name: i for i, name in enumerate(STATES)}


# --------------------------------------------------------------------------- #
# Reading a session back
# --------------------------------------------------------------------------- #


def _column(data, group: str, name: str) -> np.ndarray:
    # Tolerate both namings; early recordings used the singular.
    key = f"{group}_fields"
    if key not in data:
        key = f"{group.rstrip('s')}_fields"
    fields = [str(f) for f in data[key]]
    return data[group][:, fields.index(name)]


def predicted_beats(t: np.ndarray, tempo: np.ndarray,
                    beat_phase: np.ndarray, beat: np.ndarray) -> np.ndarray:
    """The beat times the renderer actually used, from the per-block trace.

    Each block records the clock's phase and tempo, so the beat it was
    counting from is ``t - phase * period``.  The last block before each beat
    index changes holds the prediction that was standing when the lights
    fired -- the number that decides whether they were on time -- which is
    not the same as where the tracker later said the beat was.
    """
    period = 60.0 / np.maximum(tempo, 1.0)
    origin = t - beat_phase * period
    index = beat.astype(int)
    last = np.flatnonzero(np.diff(index) != 0)      # last block of each beat
    return origin[last]


def kick_on(times: np.ndarray, t: np.ndarray, kick: np.ndarray,
            window: float = 0.06) -> float:
    """Mean peak kick within ``window`` of each time; 0 if none fall in range."""
    if len(times) == 0:
        return 0.0
    lo = np.searchsorted(t, times - window)
    hi = np.searchsorted(t, times + window)
    peaks = [kick[a:b].max() for a, b in zip(lo, hi) if b > a]
    return float(np.mean(peaks)) if peaks else 0.0


@dataclass
class Window:
    start: float
    end: float
    tempo: float
    confidence: float
    locked: float
    free_running: float
    beats: int
    kick_ours: float
    kick_offbeat: float
    energy: float
    state: str

    @property
    def polarity(self) -> str:
        if self.beats < 4 or max(self.kick_ours, self.kick_offbeat) < 0.5:
            return "-"
        # Same bar the clock's own polarity check uses: below it a syncopated
        # passage (kicks on the and) reads as a tie, not a slip.
        if self.kick_offbeat > 1.35 * self.kick_ours:
            return "OFFBEAT"
        if self.kick_ours > 1.35 * self.kick_offbeat:
            return "ok"
        return "tie"


def windows(t, tempo, confidence, locked, free_running, beat, beat_phase,
            kick, energy, state, width: float) -> list[Window]:
    """Everything that matters, per stretch of the session.

    Session-wide numbers hide every failure that lasts under a minute, and
    those are the ones that matter: a half-beat slip for twenty seconds, a
    lock on pulseless material, a tempo that wandered and came back.  Thirty
    seconds is about eight bars, long enough for the kick evidence to mean
    something and short enough to localise a problem to a passage.
    """
    beats = predicted_beats(t, tempo, beat_phase, beat)
    out = []
    for start in np.arange(0.0, float(t[-1]), width):
        end = start + width
        m = (t >= start) & (t < end)
        if m.sum() < 10:
            continue
        ours = beats[(beats >= start) & (beats < end)]
        half = 30.0 / max(float(np.median(tempo[m])), 1.0)
        counts = np.bincount(state[m].astype(int), minlength=len(STATES))
        out.append(Window(
            start=float(start), end=float(min(end, t[-1])),
            tempo=float(np.median(tempo[m])),
            confidence=float(confidence[m].mean()),
            locked=float(locked[m].mean()),
            free_running=float(free_running[m].mean()),
            beats=len(ours),
            kick_ours=kick_on(ours, t, kick),
            kick_offbeat=kick_on(ours + half, t, kick),
            energy=float(np.median(energy[m])),
            state=STATES[int(counts.argmax())],
        ))
    return out


def window_table(rows: list[Window]) -> str:
    lines = ["  window      tempo  conf  lock  kick ours/off  phase    energy  state",
             "  " + "-" * 72]
    for w in rows:
        lines.append(
            f"  {w.start:5.0f}-{w.end:4.0f}s  {w.tempo:5.1f}  {w.confidence:4.2f}"
            f"  {100 * w.locked:3.0f}%  {w.kick_ours:5.2f}/{w.kick_offbeat:<5.2f}"
            f"  {w.polarity:<7}  {w.energy:5.2f}  {w.state}")
    return "\n".join(lines)


def window_findings(rows: list[Window]) -> list[str]:
    """The checks that would have named each failure seen so far."""
    findings = []
    run: list[Window] = []
    for w in rows + [None]:
        if w is not None and w.polarity == "OFFBEAT":
            run.append(w)
            continue
        if run:
            findings.append(
                f"on the wrong half of the beat {run[0].start:.0f}-{run[-1].end:.0f}s "
                f"(kick {np.mean([r.kick_offbeat for r in run]):.1f} on our offbeat "
                f"vs {np.mean([r.kick_ours for r in run]):.1f} on our beat)")
            run = []
    for w in rows:
        if w.locked > 0.5 and max(w.kick_ours, w.kick_offbeat) < 0.5:
            findings.append(f"locked {100 * w.locked:.0f}% through {w.start:.0f}-"
                            f"{w.end:.0f}s with no low end to lock to")
    for a, b in zip(rows, rows[1:]):
        if a.confidence > 0.5 and b.confidence > 0.5 and \
                abs(b.tempo - a.tempo) > 0.04 * a.tempo:
            findings.append(f"tempo {a.tempo:.1f} -> {b.tempo:.1f} at {b.start:.0f}s"
                            + (" (a metrical ratio: counting the same music "
                               "differently)" if _near_ratio(b.tempo / a.tempo)
                               else ""))
    return findings


def _near_ratio(ratio: float) -> bool:
    return any(abs(ratio - r) < 0.04 * r for r in (0.5, 2 / 3, 0.75, 4 / 3, 1.5, 2.0))


def analyze(directory: Path, replay: bool = False, width: float = 30.0) -> int:
    """Report what happened, and flag the things that look wrong.

    Deliberately opinionated: a dump of numbers is what we already had.  The
    value is in the checks -- "it called this quiet while the level was high"
    is the sentence that would have saved three rounds of guessing.
    """
    directory = Path(directory)
    meta = json.loads((directory / "session.json").read_text())
    data = np.load(directory / "trace.npz", allow_pickle=False)

    t = _column(data, "features", "t")
    rms = _column(data, "features", "rms")
    energy = _column(data, "features", "energy")
    silent = _column(data, "features", "silent")
    high = _column(data, "features", "high_share")
    kick = _column(data, "features", "kick")
    tempo = _column(data, "clock", "tempo")
    confidence = _column(data, "clock", "confidence")
    locked = _column(data, "clock", "locked")
    free_running = _column(data, "clock", "free_running")
    beat = _column(data, "clock", "beat")
    beat_phase = _column(data, "clock", "beat_phase")
    state = data["state"][:, 0]
    duration = float(meta["duration_s"])

    print(f"session   {directory}")
    print(f"  started {meta['started']}, {duration / 60:.1f} min, "
          f"{meta['blocks']} blocks, {meta['frames']} frame samples")
    audio = directory / "audio.wav"
    if audio.exists():
        print(f"  audio   {audio} ({audio.stat().st_size / 1e6:.0f} MB)")
    cfg = meta.get("config", {}).get("audio", {})
    print(f"  input   device {cfg.get('device') or 'default'}, autogain "
          f"{cfg.get('autogain')}, silence_dbfs {cfg.get('silence_dbfs')}")

    print("\nlevels")
    for name, values in (("rms", rms), ("energy", energy)):
        print(f"  {name:<8} p5 {np.percentile(values, 5):8.4f}  "
              f"p50 {np.percentile(values, 50):8.4f}  "
              f"p95 {np.percentile(values, 95):8.4f}")
    loud = rms > np.percentile(rms, 70)

    print("\nstate")
    for index, name in enumerate(STATES):
        share = float(np.mean(state == index))
        if share > 0.001:
            print(f"  {name:<10} {100 * share:5.1f}% of the session")
    changes = meta.get("transitions", [])
    print(f"  {len(changes)} transitions"
          + (f", {len(changes) / max(duration / 60, 1e-6):.1f}/min" if duration else ""))
    for when, name, why in changes[:24]:
        print(f"    {when:8.1f}s  -> {name:<9} {why}")
    if len(changes) > 24:
        print(f"    ... {len(changes) - 24} more")

    print("\nclock")
    print(f"  tempo p5/p50/p95  {np.percentile(tempo, 5):.1f} / "
          f"{np.percentile(tempo, 50):.1f} / {np.percentile(tempo, 95):.1f}")
    print(f"  locked            {100 * np.mean(confidence >= 0.5):.0f}% of the session")
    beats = meta.get("beats", [])
    if beats:
        print(f"  {len(beats)} accepted tracker beats recorded")

    rows = windows(t, tempo, confidence, locked, free_running, beat, beat_phase,
                   kick, energy, state, width)
    print(f"\nper {width:.0f} s  (kick = low end on our beats / on our offbeat; "
          "the beat is where the kick is)")
    print(window_table(rows))

    print("\nsuspicious")
    findings = []
    quiet_index = STATES.index("quiet")
    silent_index = STATES.index("silent")
    for label, index in (("quiet", quiet_index), ("silent", silent_index)):
        both = loud & (state == index)
        if both.any():
            findings.append(
                f"{label} while the level was in its top 30%: "
                f"{100 * np.mean(both):.1f}% of the session "
                f"(first at {t[np.argmax(both)]:.1f}s)")
    if silent.any() and (silent.astype(bool) & loud).any():
        findings.append("the silence gate fired on loud audio -- silence_dbfs "
                        "is set too high for this input")
    if duration > 60 and len(changes) / (duration / 60) > 8:
        findings.append(f"{len(changes) / (duration / 60):.1f} state changes a "
                        "minute; that is flapping, not structure")
    findings.extend(window_findings(rows))
    steady = [w for w in rows if w.confidence > 0.5]
    if len(steady) >= 2:
        spread = max(w.tempo for w in steady) - min(w.tempo for w in steady)
        if spread > 12:
            findings.append(f"tempo ranged {spread:.0f} BPM across locked windows"
                            " -- several tracks, or one that would not sit still")
    if np.mean(confidence >= 0.5) < 0.5:
        findings.append(f"the beat clock was only locked "
                        f"{100 * np.mean(confidence >= 0.5):.0f}% of the time")
    if np.median(high[state == quiet_index]) > 0.5 if (state == quiet_index).any() else False:
        findings.append("quiet passages are broadband -- this looks like room "
                        "noise rather than music")
    print("  " + ("\n  ".join(findings) if findings else "nothing stood out"))

    if replay:
        print("\nreplaying the recorded audio through the current code...")
        _replay(directory, meta, width)
    return 0


def _replay(directory: Path, meta: dict, width: float = 30.0) -> None:
    """Rerun the chain on the recorded audio -- does today's code do better?

    Same table as the recording, so the two can be read side by side, plus
    the clock's own counters, which the trace does not carry.
    """
    import wave

    from .audio import ArraySource
    from .listener import Listener
    from .state import StateMachine

    with wave.open(str(directory / "audio.wav"), "rb") as handle:
        raw = handle.readframes(handle.getnframes())
    samples = (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)
    cfg = meta.get("config", {}).get("audio", {})
    machine = StateMachine()
    listener = Listener(
        ArraySource(samples, blocksize=int(meta.get("blocksize", 512))),
        silence_dbfs=float(cfg.get("silence_dbfs", -70.0)),
        window=int(cfg.get("window", 2048)))
    rows = []
    for block in listener.source.blocks():
        features = listener.step(block)
        machine.push(features)
        clock = listener.clock.state(features.t)
        rows.append((features.t, clock.tempo, clock.confidence, clock.locked,
                     clock.free_running, clock.beat, clock.beat_phase,
                     features.kick, features.energy,
                     STATES_INDEX[machine.state]))
    trace = np.array(rows, dtype=np.float64).T
    table = windows(*trace[:9], trace[9], width)
    clock = listener.clock
    print(f"  clock: relocks {clock.relocks}, half-beat slips {clock.slips}, "
          f"flips onto the tracker's half {clock.flips}, "
          f"offbeat detections {clock.offbeat_events}")
    print(window_table(table))
    for line in window_findings(table):
        print(f"  ! {line}")
    before = [(float(w), n) for w, n, _ in meta.get("transitions", [])]
    after = [(float(w), n) for w, n in machine.history]

    def same(a, b):
        return a[1] == b[1] and abs(a[0] - b[0]) < 0.5

    print(f"  {len(after)} transitions now vs {len(before)} when recorded")
    for x in after:
        mark = "" if any(same(x, y) for y in before) else "   (new)"
        print(f"    {x[0]:8.1f}s  -> {x[1]}{mark}")
    for x in before:
        if not any(same(x, y) for y in after):
            print(f"    {x[0]:8.1f}s  -> {x[1]}   (no longer)")
