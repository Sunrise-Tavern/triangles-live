"""Does the beat clock actually land on the beat?

The only honest way to answer that is against ground truth, so this measures
the live pipeline the way the renderer uses it -- stepping a 40 fps loop
through the audio and recording, for each beat, the prediction that was
*standing one frame before it happened*.  That is the number that decides
whether a light is on time, and it is not the same as how accurately the
tracker reported beats after the fact.

Two references:

* ``.cache/test_track.wav`` is synthesised at exactly 128 BPM from t=0, so its
  grid is arithmetic and exact.  It also contains passages with no kick at all
  (its intro and break), which is where a tracker is supposed to struggle.
* Real tracks are compared against librosa's offline beat tracker, which sees
  the whole file at once.  That is not truth, and it turns out to matter: on
  two of the five test tracks librosa's grid sits on the *offbeat*, so
  measuring against it reported us 60 % wrong when we were right.

Which is why there is a third measurement, and it is the one to trust on
phase: **how much kick lands on each grid**.  It needs no tracker's opinion --
for the music this rig plays, the beat is where the low end hits.  A grid whose
beats carry twice the kick energy of the alternative is the correct grid,
whoever produced it.

Fault injection covers what a DJ set does to a tracker: a silent break, and a
tempo change.  Free-run and re-lock are the normal case here, not the error
case, so they get measured rather than asserted.

    ./live.sh beats .cache/test_track.wav
    ./live.sh beats out/Eric_Prydz_Opus_OUT_NOW.mp3
    ./live.sh beats --faults
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import SAMPLERATE, ArraySource, FileSource
from .clock import BeatClock
from .listener import Listener


@dataclass
class Fired:
    index: int
    predicted: float        # when the renderer thought the beat would be
    lead: float             # how far ahead that prediction was made


def _verdict(ours: float, reference: float, tie: float = 0.05) -> str:
    if abs(ours - reference) <= tie * max(ours, reference, 1e-9):
        return "same grid"
    return "ours carries the low end" if ours > reference else \
           "the reference carries the low end"


def kick_alignment(times: np.ndarray, feature_t: np.ndarray,
                   kick: np.ndarray, window: float = 0.06) -> float:
    """Mean peak kick strength within ``window`` of each candidate beat.

    The objective tiebreaker: no tracker's opinion involved, just where the
    low end actually is.
    """
    scores = [kick[m].max() for T in times
              if (m := np.abs(feature_t - T) < window).any()]
    return float(np.mean(scores)) if scores else 0.0


@dataclass
class Report:
    label: str
    fired: list[Fired]
    reference: np.ndarray
    listener: Listener
    #: (ours, reference, our-offbeat) mean kick strength; the phase tiebreaker.
    kick_scores: tuple[float, float, float] | None = None

    @property
    def period(self) -> float:
        """The reference's own beat period -- the scale errors mean anything on."""
        if len(self.reference) < 2:
            return 0.5
        return float(np.median(np.diff(self.reference)))

    @property
    def scored(self) -> np.ndarray:
        """Fired beats that fall inside the reference's coverage.

        librosa's grid does not always reach the end of a file -- on one of
        the test tracks it stops 20 seconds early.  Scoring beats past that
        against the last reference beat produced "errors" of 8 seconds, which
        said nothing about the clock and everything about the yardstick.
        """
        if not self.fired or not len(self.reference):
            return np.zeros(0, dtype=bool)
        predicted = np.array([f.predicted for f in self.fired])
        inside = (predicted >= self.reference[0] - self.period / 2) & \
                 (predicted <= self.reference[-1] + self.period / 2)
        # ...and not inside a hole.  The fault tests deliberately contain a
        # silent break, where there is no reference to be right or wrong
        # against; scoring free-run beats there measures nothing.
        gaps = np.diff(self.reference)
        for start, span in zip(self.reference[:-1], gaps):
            if span > 1.6 * self.period:
                inside &= ~((predicted > start + self.period / 2) &
                            (predicted < start + span - self.period / 2))
        return inside

    @property
    def times(self) -> np.ndarray:
        return np.array([f.predicted for f in self.fired])[self.scored]

    @property
    def errors(self) -> np.ndarray:
        """Signed error to the nearest reference beat, milliseconds."""
        predicted = self.times
        if not len(predicted):
            return np.zeros(0)
        nearest = self.reference[np.abs(
            predicted[:, None] - self.reference[None, :]).argmin(axis=1)]
        return (predicted - nearest) * 1000.0

    @property
    def offbeat_share(self) -> float:
        """Fraction of beats landing nearer the offbeat than the beat.

        Split out from the median because the two failures are different
        animals: phase noise makes everything a bit late, a polarity error
        makes some beats exactly wrong.  A number that mixes them hides which
        one you have.
        """
        e = np.abs(self.errors)
        return float(np.mean(e > 0.35 * self.period * 1000)) if len(e) else 0.0

    @property
    def drift_ms_per_min(self) -> float:
        """Trend in the error -- a clock running fast shows up here, not above."""
        t = self.times
        if len(t) < 8:
            return 0.0
        return float(np.polyfit(t, self.errors, 1)[0] * 60.0)

    def summary(self) -> str:
        e = self.errors
        clock = self.listener.clock
        if not len(e):
            return f"{self.label}: no beats fired"
        absolute = np.abs(e)
        expected = len(self.reference)
        span = self.reference[-1] - self.reference[0]
        rate = len(e) / span * self.period if span else 0.0
        return (
            f"{self.label}\n"
            f"  scored  {len(e)} of {len(self.fired)} fired beats against "
            f"{expected} reference ({60 / self.period:.1f} BPM reference, "
            f"{rate:.2f}x rate -- 0.5 or 2.0 means the octave is wrong)\n"
            f"  error   median {np.median(absolute):5.1f} ms   "
            f"p90 {np.percentile(absolute, 90):5.1f} ms   "
            f"bias {e.mean():+6.1f} ms\n"
            f"  within  30 ms {100 * np.mean(absolute < 30):5.1f} %   "
            f"50 ms {100 * np.mean(absolute < 50):5.1f} %   "
            f"offbeat {100 * self.offbeat_share:4.1f} %\n"
            f"  drift   {self.drift_ms_per_min:+.1f} ms/min\n"
            + (f"  kick    ours {self.kick_scores[0]:.2f}   reference "
               f"{self.kick_scores[1]:.2f}   our offbeat "
               f"{self.kick_scores[2]:.2f}   "
               f"({_verdict(*self.kick_scores[:2])})\n" if self.kick_scores else "")
            + f"  clock   tempo {clock.tempo:.2f}  confidence "
            f"{clock.confidence:.2f}  relocks {clock.relocks}  slips "
            f"{clock.slips}  beats seen {clock.beats_seen}"
        )


def run(source, *, fps: float = 40.0, backend: str = "aubio",
        clock: BeatClock | None = None, label: str = "",
        watch=None) -> tuple[list[Fired], Listener]:
    """Step a renderer-shaped loop through the audio and record predictions.

    The interleaving matters: frames are advanced only up to the time of the
    audio that has actually been analysed, so the clock is never asked about a
    moment it could not yet know about.  Running the whole file through
    analysis first and then replaying would quietly measure a clock with
    lookahead, which is the one thing this design does not have.
    """
    listener = Listener(source, backend=backend, clock=clock)
    fired: list[Fired] = []
    pending: dict[int, tuple[float, float]] = {}
    frame_period = 1.0 / fps
    lead = frame_period
    now = 0.0
    previous = 0.0

    for block in source.blocks():
        features = listener.step(block)
        while now <= features.t:
            beat_clock = listener.clock
            upcoming = beat_clock.beat_index_at(now) + 1
            predicted = beat_clock.next_beat(now)
            # Keep the newest prediction that still had a frame of warning.
            if predicted - now >= lead:
                pending[upcoming] = (predicted, predicted - now)
            for index in beat_clock.crossed(previous, now):
                when, ahead = pending.pop(
                    index, (beat_clock.beat_time(index), 0.0))
                fired.append(Fired(index=index, predicted=when, lead=ahead))
            if watch is not None:
                watch(now, listener)
            previous = now
            now += frame_period
    return fired, listener


# --------------------------------------------------------------------------- #
# References
# --------------------------------------------------------------------------- #


def synthetic_grid(duration: float, bpm: float = 128.0,
                   offset: float = 0.0) -> np.ndarray:
    period = 60.0 / bpm
    return offset + np.arange(int((duration - offset) / period) + 1) * period


def librosa_grid(path: Path) -> np.ndarray:
    """Offline ground truth: the whole file at once, which we never get live."""
    import librosa

    y, sr = librosa.load(str(path), sr=None, mono=True)
    _tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    return np.asarray(beats)


def check_file(path: Path, *, fps: float = 40.0, backend: str = "aubio",
               bpm: float | None = None) -> Report:
    feature_t: list[float] = []
    kick: list[float] = []
    source = FileSource(path, realtime=False)
    fired, listener = run(
        source, fps=fps, backend=backend,
        watch=lambda now, l: None,
    )
    # Second pass purely to collect the kick track for the tiebreaker.  Cheap
    # (analysis runs ~150x faster than real time) and keeps `run` honest --
    # it must not be handed anything the live path would not have.
    from .listener import Listener as _L
    replay = _L(FileSource(path, realtime=False))
    for block in replay.source.blocks():
        features = replay.step(block)
        feature_t.append(features.t)
        kick.append(features.kick)
    feature_t_a = np.array(feature_t)
    kick_a = np.array(kick)

    if bpm is not None:
        reference = synthetic_grid(listener.audio_time, bpm)
    else:
        reference = librosa_grid(path)

    ours = np.array([f.predicted for f in fired])
    period = float(np.median(np.diff(reference))) if len(reference) > 1 else 0.5
    scores = (kick_alignment(ours, feature_t_a, kick_a),
              kick_alignment(reference, feature_t_a, kick_a),
              kick_alignment(ours + period / 2, feature_t_a, kick_a))
    return Report(label=f"{path.name} [{backend}]", fired=fired,
                  reference=reference, listener=listener, kick_scores=scores)


# --------------------------------------------------------------------------- #
# Fault injection
# --------------------------------------------------------------------------- #


def click_track(segments: list[tuple[float, float]], *,
                samplerate: int = SAMPLERATE) -> tuple[np.ndarray, np.ndarray]:
    """Build a click track from (bpm, seconds) segments.  ``bpm=0`` is silence.

    Returns the audio and the true beat grid.  Clicks are a kick-ish thump
    plus a tick, because a pure impulse is easier to track than any real music
    and would flatter the result.
    """
    audio: list[np.ndarray] = []
    grid: list[float] = []
    cursor = 0.0
    rng = np.random.default_rng(3)
    for bpm, seconds in segments:
        span = np.zeros(int(seconds * samplerate), dtype=np.float32)
        if bpm > 0:
            period = 60.0 / bpm
            n = int(0.12 * samplerate)
            t = np.arange(n) / samplerate
            body = (np.sin(2 * np.pi * (140 * np.exp(-t * 30) + 50)
                           * np.cumsum(np.ones(n)) / samplerate)
                    * np.exp(-t * 22)).astype(np.float32)
            body += (rng.standard_normal(n) * np.exp(-t * 120) * 0.25
                     ).astype(np.float32)
            at = 0.0
            while at < seconds:
                start = int(at * samplerate)
                end = min(len(span), start + n)
                if end > start:
                    span[start:end] += body[:end - start]
                grid.append(cursor + at)
                at += period
        audio.append(span)
        cursor += seconds
    return np.concatenate(audio) * 0.6, np.array(grid)


def check_faults(fps: float = 40.0, backend: str = "aubio") -> list[str]:
    """A silent break and a tempo change -- what a DJ set does to a tracker."""
    lines: list[str] = []

    audio, grid = click_track([(128.0, 24.0), (0.0, 7.5), (128.0, 24.0)])
    trace: list[tuple[float, float, float, bool]] = []
    fired, listener = run(
        ArraySource(audio), fps=fps, backend=backend,
        watch=lambda now, l: trace.append(
            (now, l.clock.tempo, l.clock.confidence, l.clock.free_running)),
    )
    report = Report("silence: 4 bars", fired, grid, listener)
    during = [row for row in trace if 26.0 <= row[0] <= 31.0]
    kept = [f for f in fired if 24.0 <= f.predicted <= 31.5]
    after = [f for f in fired if f.predicted > 33.0]
    err_after = Report("after", after, grid, listener).errors
    lines.append(report.summary())
    lines.append(
        f"  free-run  {len(kept)} beats predicted through the silence "
        f"(expected ~{int(7.5 / (60 / 128))}); free_running "
        f"{sum(1 for r in during if r[3])}/{len(during)} frames; confidence "
        f"{during[0][2]:.2f} -> {during[-1][2]:.2f}\n"
        f"  re-lock   median error after the break "
        f"{np.median(np.abs(err_after)):.1f} ms over {len(after)} beats"
    )

    audio, grid = click_track([(128.0, 24.0), (140.0, 24.0)])
    fired, listener = run(ArraySource(audio), fps=fps, backend=backend)
    late = [f for f in fired if f.predicted > 30.0]
    tail = Report("tempo nudge 128 -> 140", late, grid, listener)
    lines.append(tail.summary())
    lines.append(f"  settled tempo {listener.clock.tempo:.1f} (want 140.0), "
                 f"{listener.clock.relocks} relocks")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--bpm", type=float,
                    help="known tempo: compare against an exact grid instead "
                         "of librosa")
    ap.add_argument("--fps", type=float, default=40.0)
    ap.add_argument("--backend", default="aubio")
    ap.add_argument("--faults", action="store_true",
                    help="run the silence and tempo-change tests")
    args = ap.parse_args(argv)

    for path in args.files:
        bpm = args.bpm
        if bpm is None and path.name == "test_track.wav":
            bpm = 128.0                     # synthesised, so the grid is exact
        print(check_file(path, fps=args.fps, backend=args.backend, bpm=bpm).summary())
        print()
    if args.faults or not args.files:
        for line in check_faults(fps=args.fps, backend=args.backend):
            print(line)
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
