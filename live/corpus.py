"""Run a pile of tracks through the live engine and see which ones break it.

Every threshold in this project was chosen by looking at one or two files.
That is how you end up with a show that works on the track you developed
against.  This runs the whole live chain over a directory of music and puts one
row per track on the table, so a bad one *stands out* instead of hiding behind
an average.

**Most of what it measures needs no oracle**, deliberately.  The offline
generator is better-informed than the live one -- it sees the whole file -- but
it is not ground truth, and treating it as such is a mistake already made once
in this project: librosa's beat grid turned out to be on the *offbeat* for two
of five tracks, so measuring against it reported us 60 % wrong when we were
right.  So the primary signals here are ones that are true or false on their
own terms:

* **kick alignment** -- does the low end land on our beats, or on our offbeat?
  No tracker's opinion involved.  ``offbeat > ours`` means we are on the wrong
  half, full stop.
* **clock health** -- share of the track spent locked, re-locks, half-beat
  corrections, how far the tempo wandered.
* **bar stability** -- a tracker that shifts the bar line once has found it;
  one that shifts it fifteen times has not.
* **state behaviour** -- transitions per minute (flapping), and whether the
  machine ever leaves one state at all.
* **cost** -- how much faster than real time the whole chain runs, which is
  what decides whether it fits on a Pi.

Offline section boundaries are compared too, but reported as a *flag* rather
than a score: where the two disagree is a place to go and listen, not evidence
that either is wrong.

    ./live.sh corpus tracks/*.mp3
    ./live.sh corpus tracks/ --offline        # also run the offline segmenter
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .arranger import Arranger
from .audio import FileSource
from .frame import Canvas
from .layout import Layout, load_layout
from .listener import Listener
from .state import STATES, StateMachine
from .verify import kick_alignment

AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".aiff", ".aif", ".ogg", ".opus"}


@dataclass
class TrackResult:
    name: str
    duration: float = 0.0
    realtime: float = 0.0
    render_ms: float = 0.0

    tempo: float = 0.0
    tempo_spread: float = 0.0
    locked_share: float = 0.0
    relocks: int = 0
    slips: int = 0
    offbeat_used: int = 0

    kick_ours: float = 0.0
    kick_offbeat: float = 0.0

    bar_shifts: int = 0
    bar_confidence: float = 0.0

    transitions: float = 0.0
    state_share: dict = field(default_factory=dict)

    offline_sections: int = 0
    boundary_median: float | None = None

    notes: list[str] = field(default_factory=list)

    @property
    def on_beat(self) -> bool:
        """Is the low end on our beats rather than our offbeats?"""
        return self.kick_ours >= self.kick_offbeat

    def verdict(self) -> str:
        """Short flags.  Empty means nothing stood out."""
        flags = []
        if not self.on_beat:
            flags.append("OFFBEAT")
        if self.locked_share < 0.6:
            flags.append(f"unlocked {100 * self.locked_share:.0f}%")
        if self.bar_shifts > 4:
            flags.append(f"bar unstable x{self.bar_shifts}")
        if self.transitions > 6:
            flags.append(f"flapping {self.transitions:.1f}/min")
        if len(self.state_share) <= 1:
            flags.append("one state only")
        if self.tempo_spread > 4.0:
            flags.append(f"tempo wander {self.tempo_spread:.1f}")
        if self.realtime < 3.0:
            flags.append(f"slow {self.realtime:.1f}x")
        return ", ".join(flags)


def run_track(path: Path, layout: Layout, *, fps: float = 40.0,
              backend: str = "aubio", render: bool = True,
              offline: bool = False) -> TrackResult:
    result = TrackResult(name=path.name)
    listener = Listener(FileSource(path, realtime=False), backend=backend)
    machine = StateMachine()
    canvas = arranger = out = None
    if render:
        canvas = Canvas(layout)
        arranger = Arranger(canvas, listener, state=machine)
        out = np.zeros(layout.channel_count, dtype=np.uint8)

    stamps: list[float] = []
    kicks: list[float] = []
    tempi: list[float] = []
    beat_times: list[float] = []
    previous_beat = 0.0
    locked = 0
    total = 0
    now = 0.0
    index = 0
    render_ms = 0.0
    started = time.perf_counter()

    for block in listener.source.blocks():
        features = listener.step(block)
        machine.push(features)
        stamps.append(features.t)
        kicks.append(features.kick)
        tempi.append(listener.clock.tempo)
        for index in listener.clock.crossed(previous_beat, features.t):
            beat_times.append(listener.clock.beat_time(index))
        previous_beat = features.t
        total += 1
        locked += int(listener.clock.locked)
        if render:
            while now <= features.t:
                t0 = time.perf_counter()
                arranger.render(index, now)
                canvas.to_channels(out)
                render_ms += (time.perf_counter() - t0) * 1000
                index += 1
                now += 1.0 / fps

    elapsed = time.perf_counter() - started
    result.duration = listener.audio_time
    result.realtime = result.duration / max(elapsed, 1e-6)
    result.render_ms = render_ms / max(index, 1)

    clock = listener.clock
    result.tempo = clock.tempo
    result.tempo_spread = float(np.percentile(tempi, 95) - np.percentile(tempi, 5))
    result.locked_share = locked / max(total, 1)
    result.relocks = clock.relocks
    result.slips = clock.slips
    result.offbeat_used = clock.offbeat_events

    # The oracle-free phase check: where did the low end land relative to the
    # beats we actually fired?
    #
    # Not a grid rebuilt from the final tempo and anchor, which was the first
    # attempt: on a track whose tempo wandered 43 BPM that grid never existed,
    # and comparing against it reported a phase error the show never had.  The
    # fired beats are what the renderer really used.
    stamp_a = np.array(stamps)
    kick_a = np.array(kicks)
    beats = np.array(beat_times)
    if len(beats) > 2:
        period = float(np.median(np.diff(beats)))
        result.kick_ours = kick_alignment(beats, stamp_a, kick_a)
        result.kick_offbeat = kick_alignment(beats + period / 2, stamp_a, kick_a)

    result.bar_shifts = listener.bars.shifts
    result.bar_confidence = listener.bars.confidence

    history = machine.history
    minutes = max(result.duration / 60.0, 1e-6)
    result.transitions = len(history) / minutes
    spans: dict[str, float] = {}
    marks = [(0.0, "cruising"), *history, (result.duration, history[-1][1]
                                           if history else "cruising")]
    for (t0, state), (t1, _) in zip(marks, marks[1:]):
        spans[state] = spans.get(state, 0.0) + max(0.0, t1 - t0)
    result.state_share = {k: round(v / max(result.duration, 1e-6), 3)
                          for k, v in sorted(spans.items()) if v > 0.5}

    if offline:
        result.offline_sections, result.boundary_median = _offline_boundaries(
            path, history)
    return result


def _offline_boundaries(path: Path, history) -> tuple[int, float | None]:
    """Where the offline segmenter put its boundaries, and how far we were.

    A flag, not a score.  The offline tool sees the whole file and is better
    informed; it is still only an opinion, and a disagreement is a cue to go
    and listen rather than proof that either side is wrong.
    """
    try:
        from triseq.analysis import analyze
        from triseq.structure import find_sections
    except Exception as exc:                       # noqa: BLE001
        return 0, None
    try:
        features = analyze(path, verbose=False)
        sections = find_sections(features, verbose=False)
    except Exception:                              # noqa: BLE001
        return 0, None
    boundaries = [s.start for s in sections[1:]]
    if not boundaries or not history:
        return len(sections), None
    ours = np.array([t for t, _ in history])
    distance = [float(np.abs(ours - b).min()) for b in boundaries]
    return len(sections), float(np.median(distance))


def collect(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.iterdir()
                                if p.suffix.lower() in AUDIO_SUFFIXES))
        elif path.suffix.lower() in AUDIO_SUFFIXES:
            files.append(path)
    return files


def table(results: list[TrackResult]) -> str:
    lines = [
        f"{'track':<40} {'len':>5} {'bpm':>6} {'lock':>5} {'kick':>11} "
        f"{'bar':>7} {'st/min':>6} {'x rt':>5}  flags",
        "-" * 118,
    ]
    for r in results:
        kick = f"{r.kick_ours:4.2f}/{r.kick_offbeat:4.2f}"
        lines.append(
            f"{r.name[:40]:<40} {r.duration:5.0f} {r.tempo:6.1f} "
            f"{100 * r.locked_share:4.0f}% {kick:>11} "
            f"{r.bar_shifts:3d}/{r.bar_confidence:.2f} {r.transitions:6.1f} "
            f"{r.realtime:5.1f}  {r.verdict()}"
        )
    lines.append("")
    lines.append("kick = low-end strength on our beats / on our offbeats. "
                 "Lower on the left means we are on the wrong half.")
    lines.append("bar  = shifts / final confidence. One shift then stable is "
                 "the tracker working.")
    return "\n".join(lines)


def summary(results: list[TrackResult]) -> str:
    if not results:
        return "no tracks"
    flagged = [r for r in results if r.verdict()]
    offbeat = [r for r in results if not r.on_beat]
    lines = [
        f"{len(results)} tracks, {sum(r.duration for r in results) / 60:.0f} "
        f"minutes of audio",
        f"  phase        : {len(results) - len(offbeat)}/{len(results)} with "
        f"the low end on our beats",
        f"  locked       : median {100 * float(np.median([r.locked_share for r in results])):.0f}% "
        f"of the track",
        f"  bar shifts   : median {float(np.median([r.bar_shifts for r in results])):.0f}",
        f"  transitions  : median {float(np.median([r.transitions for r in results])):.1f}/min",
        f"  speed        : slowest {min(r.realtime for r in results):.1f}x real time",
        f"  flagged      : {len(flagged)} track(s)",
    ]
    used = set()
    for r in results:
        used |= set(r.state_share)
    missing = [s for s in STATES if s not in used]
    if missing:
        lines.append(f"  never reached: {', '.join(missing)} on any track")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--backend", default="aubio")
    ap.add_argument("--fps", type=float, default=40.0)
    ap.add_argument("--no-render", action="store_true",
                    help="analysis only; skip the renderer")
    ap.add_argument("--offline", action="store_true",
                    help="also run the offline segmenter, to compare boundaries")
    ap.add_argument("--out", type=Path, help="write the report here as well")
    args = ap.parse_args(argv)

    files = collect(args.paths)
    if not files:
        print("No audio found.  Drop full-length tracks into a folder and "
              "point this at it -- see live/README.md for what to include.")
        return 1

    layout = load_layout()
    results = []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name}", flush=True)
        try:
            results.append(run_track(path, layout, fps=args.fps,
                                     backend=args.backend,
                                     render=not args.no_render,
                                     offline=args.offline))
        except Exception as exc:                   # noqa: BLE001
            print(f"    failed: {type(exc).__name__}: {exc}")
            failed = TrackResult(name=path.name)
            failed.notes.append(f"{type(exc).__name__}: {exc}")
            results.append(failed)

    report = table(results) + "\n\n" + summary(results)
    if args.offline:
        report += "\n\noffline comparison (a flag, not a score):\n"
        for r in results:
            if r.boundary_median is not None:
                report += (f"  {r.name[:44]:<46} {r.offline_sections} sections, "
                           f"median {r.boundary_median:.1f}s to our nearest "
                           f"state change\n")
    print()
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
