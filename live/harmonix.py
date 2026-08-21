"""Ground truth, from the Harmonix Set.

912 Western pop and dance tracks annotated with **beats, downbeats and
functional segments**, plus BPM and time signature (Nieto et al., ISMIR 2019;
MIT-licensed annotations at github.com/urinieto/harmonixset).  That is the
first real oracle this project has had for three things it could previously
only check against a synthesised file:

* **downbeats** -- the bar tracker was only ever tested on audio we generated;
* **sections** -- the state machine was validated against one hand-written
  arrangement;
* **beats on real music** -- where M4 is weakest, and where librosa turned out
  to be a poor oracle (it sat on the *offbeat* for two of five test tracks).

The set ships annotations, not audio.  So the workflow is: point :func:`scan`
at a music library, see which files are covered, and score those.  Nothing is
downloaded -- the repo does carry a ``youtube_urls.csv``, which this module
deliberately ignores.

Segment labels map onto our states through the pop analogue of build-and-drop:
a **prechorus is a build** and a **chorus is a drop**.  That mapping is a
judgement, not a fact, so it lives in one visible table rather than being
scattered through the scoring.
"""

from __future__ import annotations

import csv
import difflib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .state import BUILDING, CRUISING, HOT, QUIET

ROOT = Path(__file__).resolve().parent.parent / "datasets" / "harmonix"
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".aiff", ".aif", ".ogg", ".opus"}

#: Harmonix segment label -> our state.  A judgement call, stated once.
#: "prechorus" is a build and "chorus" is a drop -- the pop analogue of the
#: dance-music shape the arranger was written for.  Anything sparse or
#: terminal is quiet; anything that is just the song running is cruising.
LABEL_STATE = {
    "intro": QUIET, "outro": QUIET, "end": QUIET, "silence": QUIET,
    "break": QUIET, "breakdown": QUIET, "fadein": QUIET,
    "verse": CRUISING, "inst": CRUISING, "instrumental": CRUISING,
    "solo": CRUISING, "bridge": CRUISING, "section": CRUISING,
    "prechorus": BUILDING, "transition": BUILDING, "build": BUILDING,
    "chorus": HOT, "postchorus": HOT, "refrain": HOT,
}


def state_for(label: str) -> str | None:
    """Map a Harmonix label to one of our states, or None if we have no view."""
    base = re.sub(r"\d+$", "", label.strip().lower())
    return LABEL_STATE.get(base)


@dataclass
class Entry:
    file: str
    title: str
    artist: str
    duration: float
    bpm: float
    time_signature: str
    genre: str
    musicbrainz: str = ""

    @property
    def four_four(self) -> bool:
        return self.time_signature == "4|4"

    def beats(self, root: Path = ROOT) -> tuple[np.ndarray, np.ndarray]:
        """(beat times, beat position in bar).  Position 1 is a downbeat."""
        path = root / "dataset" / "beats_and_downbeats" / f"{self.file}.txt"
        rows = np.loadtxt(path)
        return rows[:, 0], rows[:, 1].astype(int)

    def downbeats(self, root: Path = ROOT) -> np.ndarray:
        times, position = self.beats(root)
        return times[position == 1]

    def segments(self, root: Path = ROOT) -> list[tuple[float, str]]:
        path = root / "dataset" / "segments" / f"{self.file}.txt"
        out = []
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                out.append((float(parts[0]), parts[1]))
        return out

    def state_timeline(self, root: Path = ROOT) -> list[tuple[float, str]]:
        """Segments translated into our vocabulary, unmappable ones dropped."""
        return [(t, s) for t, label in self.segments(root)
                if (s := state_for(label)) is not None]


def load(root: Path = ROOT) -> list[Entry]:
    path = root / "dataset" / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No Harmonix annotations at {root}.  Clone them with:\n"
            "  git clone --depth 1 https://github.com/urinieto/harmonixset.git "
            "datasets/harmonix"
        )
    entries = []
    for row in csv.DictReader(path.open()):
        try:
            entries.append(Entry(
                file=row["File"], title=row["Title"], artist=row["Artist"],
                duration=float(row["Duration"]), bpm=float(row["BPM"]),
                time_signature=row["Time Signature"], genre=row["Genre"],
                musicbrainz=row.get("MusicBrainz Id", "") or "",
            ))
        except (KeyError, ValueError):
            continue
    return entries


# --------------------------------------------------------------------------- #
# Matching a local file to an entry
# --------------------------------------------------------------------------- #


def _norm(text: str) -> str:
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", text.lower())
    text = re.sub(r"\b(feat|ft|featuring|remaster(ed)?|radio edit|official"
                  r"|video|audio|lyrics?)\b", " ", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def tags(path: Path) -> dict:
    """Title/artist/duration from the file itself, via ffprobe.

    ffprobe rather than a tag library because ffmpeg is already a hard
    dependency of this project and one fewer install is one fewer thing to go
    wrong on the Pi.
    """
    try:
        raw = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(path)],
            capture_output=True, text=True, timeout=30).stdout
        blob = json.loads(raw or "{}").get("format", {})
    except Exception:                              # noqa: BLE001
        return {}
    meta = {k.lower(): v for k, v in (blob.get("tags") or {}).items()}
    meta["duration"] = float(blob.get("duration", 0.0) or 0.0)
    return meta


@dataclass
class Match:
    path: Path
    entry: Entry
    score: float
    how: str
    #: Seconds between the file's duration and the annotated one.  Annotations
    #: are timed to one specific edition; a remaster or radio edit will line up
    #: at the start and drift, which looks like a tracking failure but is not.
    duration_gap: float = 0.0

    @property
    def suspect_edition(self) -> bool:
        return abs(self.duration_gap) > 2.0


def match_file(path: Path, entries: list[Entry],
               threshold: float = 0.82) -> Match | None:
    meta = tags(path)
    title = meta.get("title") or path.stem
    artist = meta.get("artist") or ""
    mbid = (meta.get("musicbrainz_trackid")
            or meta.get("musicbrainz_releasetrackid") or "")
    duration = meta.get("duration", 0.0)

    if mbid:
        for entry in entries:
            if entry.musicbrainz and entry.musicbrainz == mbid:
                return Match(path, entry, 1.0, "musicbrainz id",
                             duration - entry.duration)

    want_title, want_artist = _norm(title), _norm(artist)
    haystack = _norm(f"{artist} {title}") or _norm(path.stem)
    best: Match | None = None
    for entry in entries:
        et, ea = _norm(entry.title), _norm(entry.artist)
        if not et:
            continue
        if want_artist and et == want_title and ea == want_artist:
            return Match(path, entry, 1.0, "title + artist",
                         duration - entry.duration)
        # No usable tags: fall back to the filename, which usually contains
        # both but in an unknown order and with junk around them.
        score = difflib.SequenceMatcher(None, et + ea, haystack).ratio()
        score = max(score, difflib.SequenceMatcher(None, et, want_title).ratio()
                    * (0.75 + 0.25 * bool(ea and ea in haystack)))
        if best is None or score > best.score:
            best = Match(path, entry, score, "fuzzy", duration - entry.duration)
    if best is not None and best.score >= threshold:
        return best
    return None


def scan(paths: list[Path], entries: list[Entry] | None = None,
         threshold: float = 0.82) -> tuple[list[Match], list[Path]]:
    """Walk a library and report which files the Harmonix Set covers."""
    entries = entries if entries is not None else load()
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*")
                                if p.suffix.lower() in AUDIO_SUFFIXES))
        elif path.suffix.lower() in AUDIO_SUFFIXES:
            files.append(path)
    matched, missed = [], []
    for path in files:
        found = match_file(path, entries, threshold)
        (matched.append(found) if found else missed.append(path))
    return matched, missed


# --------------------------------------------------------------------------- #
# Rendering an annotation as audio
# --------------------------------------------------------------------------- #


def render(entry: Entry, root: Path = ROOT, samplerate: int = 44100,
           seed: int = 5, grid: tuple[np.ndarray, np.ndarray] | None = None
           ) -> np.ndarray:
    """Turn one track's annotation into audio that follows it exactly.

    Not a substitute for the real recording -- the timbres are ours, so this
    says nothing about how aubio copes with a dense modern mix.  What it does
    give, without owning a single file, is **912 real musical structures**:
    real tempo curves including drift and mid-track changes, real bar layouts
    including the 22 tracks that are not in 4/4, and real section
    arrangements.  Against that, our one hand-written synthetic arrangement is
    a sample of size one.

    Each beat gets a kick; each downbeat gets a bass note and a crash, which
    are the cues :mod:`live.downbeat` keys on; sections get the level and band
    balance their mapped state implies, so a "prechorus" really does sweep the
    high band.
    """
    times, position = grid if grid is not None else entry.beats(root)
    if len(times) < 8:
        raise ValueError(f"{entry.file}: too few annotated beats")
    timeline = entry.state_timeline(root) or [(0.0, CRUISING)]
    if grid is not None:
        # The caller re-timed the grid; move the sections with it, or a
        # scaled track would keep its original section boundaries.
        original, _ = entry.beats(root)
        scale = (times[-1] - times[0]) / max(original[-1] - original[0], 1e-9)
        shift = times[0] - original[0] * scale
        timeline = [(t * scale + shift, state) for t, state in timeline]
    rng = np.random.default_rng(seed)
    total = int((times[-1] + 2.0) * samplerate)
    audio = np.zeros(total, dtype=np.float32)

    n = int(0.18 * samplerate)
    t = np.arange(n) / samplerate
    sweep = 150 * np.exp(-t * 40) + 45
    kick = (np.sin(2 * np.pi * np.cumsum(sweep) / samplerate)
            * np.exp(-t * 18)).astype(np.float32)
    crash = (rng.standard_normal(n) * np.exp(-t * 7)).astype(np.float32) * 0.30
    hn = int(0.05 * samplerate)
    hat = (rng.standard_normal(hn)
           * np.exp(-np.arange(hn) / samplerate * 90)).astype(np.float32)

    #: state -> (kick, bass, hat, high-band sweep)
    voicing = {QUIET: (0.20, 0.30, 0.15, 0.0), CRUISING: (0.75, 0.55, 0.30, 0.0),
               BUILDING: (0.50, 0.40, 0.55, 1.0), HOT: (1.00, 0.90, 0.60, 0.0)}

    def state_at(when: float) -> str:
        current = timeline[0][1]
        for start, value in timeline:
            if start <= when:
                current = value
            else:
                break
        return current

    for index, (when, pos) in enumerate(zip(times, position)):
        state = state_at(when)
        k_lv, bass_lv, hat_lv, rise = voicing[state]
        at = int(when * samplerate)
        _mix(audio, kick * k_lv, at)
        if pos == 1:
            # The bar line, marked the way real music marks it and the way the
            # bar tracker looks for it: low end plus a broadband hit.
            _mix(audio, crash * (0.5 + 0.5 * k_lv), at)
            span = times[min(index + 1, len(times) - 1)] - when
            note = 55 * 2 ** (rng.integers(0, 4) / 12)
            seg = np.arange(int(max(span, 0.1) * 4 * samplerate)) / samplerate
            _mix(audio, (np.sin(2 * np.pi * note * seg) * bass_lv * 0.8
                         * np.exp(-seg * 0.7)).astype(np.float32), at)
        step = (times[min(index + 1, len(times) - 1)] - when) or 0.4
        for h in range(2):
            _mix(audio, hat * hat_lv, int((when + h * step / 2) * samplerate))
        if rise:
            seg = int(step * samplerate)
            progress = index / max(len(times) - 1, 1)
            _mix(audio, (rng.standard_normal(seg)
                         * (0.06 + 0.30 * progress)).astype(np.float32) * 0.7, at)

    peak = float(np.abs(audio).max())
    return (audio / (peak * 1.05)).astype(np.float32) if peak > 0 else audio


def _mix(buffer: np.ndarray, signal: np.ndarray, at: int) -> None:
    end = min(at + len(signal), len(buffer))
    if 0 <= at < len(buffer):
        buffer[at:end] += signal[:end - at]


# --------------------------------------------------------------------------- #
# Scoring our pipeline against the annotations
# --------------------------------------------------------------------------- #


@dataclass
class Score:
    match: Match
    beats_within_30ms: float = 0.0
    beat_median_ms: float = 0.0
    beats_scored: int = 0
    downbeats_within_50ms: float = 0.0
    downbeats_scored: int = 0
    bar_shifts: int = 0
    tempo: float = 0.0
    state_agreement: float = 0.0
    boundary_median_s: float | None = None
    notes: list[str] = field(default_factory=list)

    def line(self) -> str:
        entry = self.match.entry
        return (f"{entry.title[:30]:<32}{entry.artist[:18]:<20}"
                f"{entry.bpm:>5.0f}/{self.tempo:<6.1f}"
                f"{100 * self.beats_within_30ms:>6.0f}%"
                f"{100 * self.downbeats_within_50ms:>7.0f}%"
                f"{100 * self.state_agreement:>7.0f}%"
                f"  {entry.time_signature:<5}{'; '.join(self.notes)}")


def score(path: Path, match: Match, *, fps: float = 40.0,
          backend: str = "aubio") -> Score:
    """Run the live chain over one matched file and grade it."""
    from .audio import FileSource
    from .clock import BeatClock
    from .listener import Listener
    from .state import StateMachine
    from .verify import run

    result = Score(match=match)
    entry = match.entry
    if match.suspect_edition:
        result.notes.append(
            f"duration off by {match.duration_gap:+.1f}s -- probably a "
            "different edition, timings will drift")
    if not entry.four_four:
        result.notes.append(f"{entry.time_signature} -- bar_length is hardcoded to 4")

    clock = BeatClock()
    machine = StateMachine()
    fired, listener = run(FileSource(path, realtime=False), fps=fps,
                          backend=backend, clock=clock)

    # A second pass for the state machine; the beat pass above owns the clock.
    machine_listener = Listener(FileSource(path, realtime=False), backend=backend)
    for block in machine_listener.source.blocks():
        machine.push(machine_listener.step(block))

    truth_beats, _positions = entry.beats()
    ours = np.array([f.predicted for f in fired])
    if len(ours) and len(truth_beats):
        inside = (ours >= truth_beats[0]) & (ours <= truth_beats[-1])
        ours_in = ours[inside]
        if len(ours_in):
            delta = np.abs(ours_in[:, None] - truth_beats[None, :]).min(axis=1) * 1000
            result.beats_within_30ms = float(np.mean(delta < 30))
            result.beat_median_ms = float(np.median(delta))
            result.beats_scored = int(len(ours_in))

    truth_down = entry.downbeats()
    mine_down = np.array([f.predicted for f in fired if f.downbeat])
    if len(mine_down) and len(truth_down):
        inside = (mine_down >= truth_down[0]) & (mine_down <= truth_down[-1])
        mine_in = mine_down[inside]
        if len(mine_in):
            delta = np.abs(mine_in[:, None] - truth_down[None, :]).min(axis=1) * 1000
            result.downbeats_within_50ms = float(np.mean(delta < 50))
            result.downbeats_scored = int(len(mine_in))
    result.bar_shifts = listener.bars.shifts if listener.bars else 0
    result.tempo = clock.tempo

    timeline = entry.state_timeline()
    if timeline and machine.history:
        result.state_agreement, result.boundary_median_s = _compare_states(
            timeline, machine.history, entry.duration)
    return result


def _compare_states(truth: list[tuple[float, str]],
                    ours: list[tuple[float, str]],
                    duration: float) -> tuple[float, float | None]:
    """Share of the track we agree on, and how far our boundaries land.

    Sampled at 4 Hz rather than compared as event lists: what matters is how
    much of the show was in the right mode, not whether every boundary has a
    partner.
    """
    grid = np.arange(0.0, duration, 0.25)

    def at(timeline, t):
        state = timeline[0][1]
        for start, value in timeline:
            if start <= t:
                state = value
            else:
                break
        return state

    ours_full = [(0.0, CRUISING), *ours]
    agree = np.mean([at(truth, t) == at(ours_full, t) for t in grid])
    boundaries = [t for t, _ in truth[1:]]
    mine = np.array([t for t, _ in ours])
    gap = (float(np.median([float(np.abs(mine - b).min()) for b in boundaries]))
           if boundaries and len(mine) else None)
    return float(agree), gap


# --------------------------------------------------------------------------- #
# Batch: score the whole chain against rendered annotations
# --------------------------------------------------------------------------- #


@dataclass
class SynthScore:
    entry: Entry
    #: Share of *our* beats that land on an annotated one.  Halves if we run
    #: at double tempo, which recall alone would hide completely.
    beats_pct: float = 0.0
    #: Share of *annotated* beats we put a beat on.  Stays high at double
    #: tempo, so the pair together identify an octave error.
    beats_recall: float = 0.0
    beat_median_ms: float = 0.0
    downbeats_pct: float = 0.0
    bar_shifts: int = 0
    tempo: float = 0.0
    #: p95 - p5 of the tempo estimate across the track.  A correct final
    #: tempo can hide a clock that wandered badly in the middle, which is
    #: exactly what this corpus keeps catching.
    tempo_spread: float = 0.0
    state_agreement: float = 0.0
    locked: float = 0.0
    error: str = ""

    @property
    def tempo_error(self) -> float:
        return abs(self.tempo - self.entry.bpm) / max(self.entry.bpm, 1e-6)

    def line(self) -> str:
        e = self.entry
        if self.error:
            return f"{e.title[:28]:<30}{e.genre[:12]:<14}  FAILED {self.error[:44]}"
        return (f"{e.title[:28]:<30}{e.genre[:12]:<14}{e.bpm:>5.0f}"
                f"{self.tempo:>7.1f}{e.time_signature:>6}"
                f"{100 * self.beats_pct:>6.0f}%{100 * self.beats_recall:>6.0f}%"
                f"{100 * self.downbeats_pct:>8.0f}%"
                f"{self.bar_shifts:>5}{self.tempo_spread:>7.0f}"
                f"{100 * self.state_agreement:>7.0f}%{100 * self.locked:>6.0f}%")


def score_synthetic(entry: Entry, *, fps: float = 40.0,
                    backend: str = "aubio", root: Path = ROOT) -> SynthScore:
    """Render one annotation as audio and grade the live chain against it."""
    from .audio import ArraySource
    from .clock import BeatClock
    from .listener import Listener
    from .state import StateMachine
    from .verify import run

    result = SynthScore(entry=entry)
    try:
        audio = render(entry, root)
    except Exception as exc:                       # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    clock = BeatClock()
    fired, listener = run(ArraySource(audio), fps=fps, backend=backend,
                          clock=clock)
    truth, _positions = entry.beats(root)
    ours = np.array([f.predicted for f in fired])
    if len(ours) and len(truth):
        delta = np.abs(ours[:, None] - truth[None, :]).min(axis=1) * 1000
        result.beats_pct = float(np.mean(delta < 30))
        result.beat_median_ms = float(np.median(delta))
        back = np.abs(truth[:, None] - ours[None, :]).min(axis=1) * 1000
        result.beats_recall = float(np.mean(back < 30))

    down = entry.downbeats(root)
    mine = np.array([f.predicted for f in fired if f.downbeat])
    if len(mine) and len(down):
        delta = np.abs(mine[:, None] - down[None, :]).min(axis=1) * 1000
        result.downbeats_pct = float(np.mean(delta < 50))
    result.bar_shifts = listener.bars.shifts
    result.tempo = clock.tempo

    machine = StateMachine()
    second = Listener(ArraySource(audio), backend=backend)
    locked = total = 0
    trace: list[float] = []
    for block in second.source.blocks():
        features = second.step(block)
        machine.push(features)
        total += 1
        locked += int(second.clock.locked)
        trace.append(second.clock.tempo)
    result.locked = locked / max(total, 1)
    if trace:
        result.tempo_spread = float(np.percentile(trace, 95)
                                    - np.percentile(trace, 5))
    timeline = entry.state_timeline(root)
    if timeline:
        # The rendered length, not the metadata duration: annotations do not
        # always cover the whole track (one spans 5.2-135.5 s of a 173 s
        # song), and comparing over the difference scores us against nothing.
        result.state_agreement, _gap = _compare_states(
            timeline, machine.history, len(audio) / 44100.0)
    return result


def sample(entries: list[Entry], count: int, seed: int = 0) -> list[Entry]:
    """A spread across genre, tempo and metre rather than the first N.

    The set is 46 % pop; taking the head of the list would measure pop.
    """
    rng = np.random.default_rng(seed)
    buckets: dict[tuple, list[Entry]] = {}
    for entry in entries:
        key = (entry.genre, entry.four_four, int(entry.bpm // 20))
        buckets.setdefault(key, []).append(entry)
    keys = sorted(buckets)
    picked: list[Entry] = []
    while len(picked) < count and keys:
        for key in list(keys):
            pool = buckets[key]
            if not pool:
                keys.remove(key)
                continue
            picked.append(pool.pop(rng.integers(len(pool))))
            if len(picked) >= count:
                break
    return picked


# --------------------------------------------------------------------------- #
# Crossfades
# --------------------------------------------------------------------------- #


@dataclass
class Transition:
    """Two tracks mixed, with the true grid on each side of the blend."""

    audio: np.ndarray
    before: np.ndarray          # track A's beat times, in mix time
    after: np.ndarray           # track B's beat times, in mix time
    after_downbeats: np.ndarray
    blend_start: float
    blend_end: float
    tempo_a: float
    tempo_b: float
    label: str = ""

    @property
    def duration(self) -> float:
        return len(self.audio) / 44100.0


def crossfade(a: Entry, b: Entry, *, lead: float = 40.0, overlap: float = 16.0,
              tail: float = 40.0, match_tempo: bool = True,
              beat_offset: int = 0, root: Path = ROOT,
              samplerate: int = 44100, label: str = "") -> Transition:
    """Mix two annotated tracks the way a DJ would, and keep both grids.

    ``match_tempo`` beatmatches B to A, which is what actually happens on a
    CDJ; without it the two tempos genuinely differ across the blend, which is
    the harder case.  ``beat_offset`` slides B's downbeat, so a mix can be
    tempo-matched but bar-misaligned -- a real and common mistake, and the one
    that most confuses a bar tracker.

    Scaling the annotation times *is* the tempo change: the audio is rendered
    from those times, so no resampling is involved and the ground truth stays
    exact.
    """
    a_times, a_pos = a.beats(root)
    b_times, b_pos = b.beats(root)
    a_period = float(np.median(np.diff(a_times)))
    b_period = float(np.median(np.diff(b_times)))
    scale = (a_period / b_period) if match_tempo else 1.0

    # A runs from 0; B starts at the top of the blend, shifted so its beats
    # line up with A's (plus any deliberate offset).
    blend_start = lead
    a_keep = a_times[a_times <= blend_start + overlap + 1e-6]
    if len(a_keep) < 8:
        raise ValueError(f"{a.file}: not enough beats before the blend")
    b_scaled = b_times * scale
    b_scaled = b_scaled - b_scaled[0]
    grid_at_blend = a_times[np.searchsorted(a_times, blend_start)]
    b_shift = grid_at_blend + beat_offset * a_period
    b_placed = b_scaled + b_shift
    keep = b_placed <= blend_start + overlap + tail
    b_placed, b_kept_pos = b_placed[keep], b_pos[keep]
    if len(b_placed) < 8:
        raise ValueError(f"{b.file}: not enough beats after the blend")

    audio_a = render(a, root, samplerate, grid=(a_keep, a_pos[:len(a_keep)]))
    audio_b = render(b, root, samplerate, seed=9, grid=(b_placed, b_kept_pos))

    total = int(max(len(audio_a), b_shift * samplerate + len(audio_b)))
    mix = np.zeros(total + samplerate, dtype=np.float32)

    # Equal-power, which is what a mixer does; a linear blend dips in the
    # middle and the analyser would read that dip as a breakdown.
    fade = np.ones(len(audio_a), dtype=np.float32)
    i0 = int(blend_start * samplerate)
    i1 = min(len(audio_a), int((blend_start + overlap) * samplerate))
    if i1 > i0:
        t = np.linspace(0.0, 1.0, i1 - i0, dtype=np.float32)
        fade[i0:i1] = np.cos(t * np.pi / 2)
        fade[i1:] = 0.0
    mix[:len(audio_a)] += audio_a * fade

    start_b = int(b_shift * samplerate)
    up = np.ones(len(audio_b), dtype=np.float32)
    j1 = min(len(audio_b), int(overlap * samplerate))
    if j1 > 0:
        t = np.linspace(0.0, 1.0, j1, dtype=np.float32)
        up[:j1] = np.sin(t * np.pi / 2)
    end_b = min(len(mix), start_b + len(audio_b))
    mix[start_b:end_b] += audio_b[:end_b - start_b] * up[:end_b - start_b]

    peak = float(np.abs(mix).max())
    if peak > 0:
        mix = (mix / (peak * 1.05)).astype(np.float32)
    return Transition(
        audio=mix, before=a_keep, after=b_placed,
        after_downbeats=b_placed[b_kept_pos == 1],
        blend_start=blend_start, blend_end=blend_start + overlap,
        tempo_a=60.0 / a_period, tempo_b=60.0 / (b_period * scale), label=label,
    )
