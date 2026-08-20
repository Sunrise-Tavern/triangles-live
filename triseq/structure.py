"""Song structure: find the sections, name them, and choose the excerpt.

Section labels drive everything the arranger does, so this module's job is to
turn a continuous track into a short list of named, bar-aligned spans.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analysis import Features

#: Section kinds, ordered roughly by intensity.
KINDS = ("intro", "break", "verse", "build", "drop", "outro")

#: Below this spread between a track's loud and quiet sections (on the 0-1
#: normalized intensity scale), treat it as having no dynamic structure at all
#: rather than ranking sections against each other.
FLAT_TRACK_CONTRAST = 0.12


@dataclass
class Section:
    start: float
    end: float
    kind: str
    #: Mean normalized energy, 0-1.
    energy: float
    #: Mean spectral centroid, 0-1. Drives palette warmth.
    brightness: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def shifted(self, by: float) -> "Section":
        return Section(self.start - by, self.end - by, self.kind,
                       self.energy, self.brightness)

    def __str__(self) -> str:
        return (f"{self.start:6.1f}-{self.end:6.1f}s  {self.kind:<6} "
                f"energy={self.energy:.2f}")


def find_sections(f: Features, *, verbose: bool = True,
                  stats: tuple[float, float] | None = None) -> list[Section]:
    """Segment the track, then label each segment by its role.

    Pass `stats` from `label_scale()` on the full track to keep labels on the
    song's own scale while the boundaries are found within a shorter window.
    """
    import librosa

    if len(f.beat_times) < 8:
        # Too short or too rubato to segment; treat it as one span.
        return [_make_section(f, 0.0, f.duration, "verse")]

    # Beat-synchronous timbre + harmony features. Beat-syncing means segment
    # boundaries can only land on beats, which is most of the way to bar-aligned.
    beat_frames = librosa.time_to_frames(f.beat_times, sr=f.sr, hop_length=512)
    beat_frames = np.clip(beat_frames, 0, len(f.frame_times) - 1)

    chroma = librosa.feature.chroma_cqt(y=f.y, sr=f.sr, hop_length=512)
    mfcc = librosa.feature.mfcc(y=f.y, sr=f.sr, hop_length=512, n_mfcc=13)

    sync = np.vstack([
        librosa.util.sync(chroma, beat_frames, aggregate=np.median),
        librosa.util.sync(mfcc, beat_frames, aggregate=np.mean),
    ])
    sync = librosa.util.normalize(sync, axis=0)

    k = int(np.clip(round(f.duration / 8.0), 6, 12))
    k = min(k, sync.shape[1] - 1)
    if k < 2:
        return [_make_section(f, 0.0, f.duration, "verse")]

    if verbose:
        print(f"  segmenting into {k} parts")
    bounds = librosa.segment.agglomerative(sync, k)
    bound_times = [float(f.beat_times[min(b, len(f.beat_times) - 1)]) for b in bounds]

    # Snap to bar lines, dedupe, and bracket the whole track.
    edges = sorted({0.0, *(f.snap_to_bar(t) for t in bound_times), f.duration})
    # Drop segments shorter than two bars -- they are analysis noise, not sections.
    min_len = max(2 * f.bar_period, 4.0)
    merged = [edges[0]]
    for t in edges[1:]:
        if t - merged[-1] >= min_len:
            merged.append(t)
    if merged[-1] < f.duration:
        merged[-1] = f.duration
    if len(merged) < 2:
        merged = [0.0, f.duration]

    spans = list(zip(merged[:-1], merged[1:]))
    return _label(f, spans, stats)


def _make_section(f: Features, start: float, end: float, kind: str) -> Section:
    return Section(
        start=start, end=end, kind=kind,
        energy=f.mean_over(f.rms, start, end),
        brightness=f.mean_over(f.centroid, start, end),
    )


def _intensity(f: Features, sections: list[Section]) -> np.ndarray:
    """Per-section intensity: loudness, weighted toward the low end.

    Bass is what separates a drop from a build. Both are loud and bright, but a
    build is a riser -- hats, noise sweeps, no kick -- while a drop lands the low
    end. Weighting highs too heavily promotes every build to a drop.
    """
    energies = np.array([s.energy for s in sections])
    lows = np.array([f.mean_over(f.bands["bass"], s.start, s.end) for s in sections])
    highs = np.array([f.mean_over(f.bands["high"], s.start, s.end) for s in sections])
    return 0.50 * energies + 0.35 * lows + 0.15 * highs


def label_scale(f: Features, sections: list[Section]) -> tuple[float, float]:
    """The (midpoint, spread) that section labelling is measured against.

    Exposed so a whole track's scale can be reused when labelling a short
    excerpt of it -- see `find_sections(stats=...)`.
    """
    intensity = _intensity(f, sections)
    med = float(np.median(intensity))
    iqr = float(np.percentile(intensity, 75) - np.percentile(intensity, 25))
    # Bound the spread at both ends. Too small and a uniformly-loud track gets
    # its tiny variations amplified into imaginary structure; too large -- which
    # is what a sharply bimodal track produces -- and the threshold grows past
    # the gap between quiet and loud, so genuine drops read as ordinary verses.
    return med, min(max(iqr, 0.08), 0.25)


def _label(f: Features, spans: list[tuple[float, float]],
           stats: tuple[float, float] | None = None) -> list[Section]:
    """Assign a role to each span from its energy relative to the track.

    `stats` overrides the midpoint and spread that intensities are compared
    against. Without it the scale is derived from `spans` themselves, which is
    correct for a whole track but wrong for an excerpt: a window selected *for*
    being energetic re-baselines against its own contents, and its drop -- the
    reason the window was chosen -- gets demoted to a verse because it is merely
    typical of that window.
    """
    raw = [_make_section(f, s, e, "verse") for s, e in spans]
    intensity = _intensity(f, raw)
    med, spread = stats if stats is not None else label_scale(f, raw)

    # Some music simply has no drops. A track that holds one energy level for
    # its whole length -- steady dream-house, most ambient, a lot of rock --
    # still produces a highest and a lowest section, and dividing those tiny
    # differences by the floored spread manufactures a "drop" out of a section
    # that is two percent louder than its neighbours. The arranger then lights
    # it with strobes and white accents, which is conspicuously wrong for music
    # that never actually lifts.
    #
    # So check the real contrast first, and if the track is flat, say so.
    contrast = float(np.percentile(intensity, 90) - np.percentile(intensity, 10))
    if contrast < FLAT_TRACK_CONTRAST:
        for sec in raw:
            sec.kind = "verse"
        _mark_edges(raw, intensity)
        return raw

    z = (intensity - med) / spread

    for i, sec in enumerate(raw):
        if z[i] >= 0.5:
            sec.kind = "drop"
        elif z[i] <= -0.5:
            sec.kind = "break"
        else:
            sec.kind = "verse"

    # A section before a drop is a build only if it actually rises into it.
    # Without the slope test, every quiet breakdown that happens to precede a
    # drop gets relabelled, and the arranger then ramps where it should rest.
    for i in range(len(raw) - 1):
        if raw[i + 1].kind != "drop" or raw[i].kind == "drop":
            continue
        if _rises(f, raw[i]):
            raw[i].kind = "build"

    _mark_edges(raw, _intensity(f, raw))
    return raw


def _mark_edges(raw: list[Section], intensity=None) -> None:
    """Give the first and last sections their structural names.

    The last section only becomes an `outro` if the music is actually winding
    down. An excerpt is usually cut out of the middle of a track, so its final
    section is often still at full energy -- labelling that an outro makes the
    rig fade toward black while the music is still going, which reads as the
    show breaking rather than ending.
    """
    if not raw:
        return
    have = intensity is not None and len(intensity) == len(raw)
    med = float(np.median(intensity)) if have else None

    # Same test at both ends: an edge section is an intro/outro only if the
    # music is actually quieter there than in the middle. An excerpt usually
    # starts mid-track with the pulse already going, and dimming that opening
    # as an "intro" throws away the first bars of exactly what the window was
    # chosen for.
    if raw[0].kind != "drop" and (not have or float(intensity[0]) <= med):
        raw[0].kind = "intro"
    if len(raw) >= 2 and raw[-1].kind != "drop" and \
            (not have or float(intensity[-1]) <= med):
        raw[-1].kind = "outro"


def _rises(f: Features, sec: Section) -> bool:
    """Does energy trend upward across this section?

    Fits a line to the RMS envelope; a build ramps, a breakdown sits flat or
    sags even though both can sit at the same average level.
    """
    lo, hi = np.searchsorted(f.frame_times, [sec.start, sec.end])
    seg = f.rms[lo:max(hi, lo + 2)]
    if len(seg) < 4:
        return False
    x = np.linspace(0.0, 1.0, len(seg))
    slope = float(np.polyfit(x, seg, 1)[0])
    return slope > 0.05


def pick_excerpt(f: Features, sections: list[Section], *, target: float | None = None,
                 lo: float = 90.0, hi: float = 120.0,
                 verbose: bool = True) -> tuple[float, float]:
    """Choose the most show-worthy window. Returns (start, duration).

    With `target` unset the *length* is chosen too, anywhere in [lo, hi]. A
    fixed length has to cut somewhere arbitrary: on a track with a long build
    and a late drop, 90 seconds starting at a sensible place leaves the drop
    clipped to the last few bars. Letting the window stretch means it can run to
    the end of the drop instead of through the middle of it.
    """
    if target is not None:
        candidates = [float(np.clip(target, lo, hi))]
    else:
        # Try lengths at 4-bar granularity so every candidate stays phrase-aligned.
        step = max(f.bar_period * 4.0, 5.0)
        candidates = []
        t = lo
        while t <= hi + 1e-6:
            candidates.append(round(t, 3))
            t += step
        if candidates[-1] < hi:
            candidates.append(hi)

    shortest = min(candidates)
    if f.duration <= shortest:
        return 0.0, f.duration

    boundaries = {round(s.start, 2) for s in sections}
    ends = {round(s.end, 2) for s in sections}
    lead_in_kinds = {"intro", "build", "break"}

    best: tuple[float, float] = (0.0, shortest)
    best_score = -1e9

    for span in candidates:
        starts = f.bar_times[f.bar_times <= f.duration - span]
        if len(starts) == 0:
            continue
        for start in starts:
            score = _score_window(f, sections, float(start), span,
                                  boundaries, ends, lead_in_kinds)
            # A touch of preference for longer windows, but far too small to
            # override a genuinely better-placed shorter one.
            score += 0.15 * (span - shortest) / max(hi - shortest, 1e-6)
            if score > best_score:
                best, best_score = (float(start), span), score

    if verbose:
        print(f"  excerpt {best[0]:.1f}s -> {best[0] + best[1]:.1f}s "
              f"({best[1]:.0f}s, score {best_score:.2f})")
    return best


def _score_window(f: Features, sections: list[Section], start: float, target: float,
              boundaries: set, ends: set, lead_in_kinds: set) -> float:
    end = start + target
    score = f.mean_over(f.rms, start, end) * 1.2

    covered = [s for s in sections if s.start < end and s.end > start]
    # A window that contains a whole drop is worth far more than one that
    # merely clips the loudest part of it.
    if any(s.kind == "drop" and s.start >= start and s.end <= end for s in covered):
        score += 1.5
    elif any(s.kind == "drop" for s in covered):
        score += 0.4

    # Reward variety -- a window spanning several kinds has structure.
    score += 0.25 * len({s.kind for s in covered})

    # Weight by how much of the window is actually *doing* something. A
    # window can start on a section boundary and contain two drops while
    # still being mostly breakdown, which makes for a dull show.
    active = sum(
        min(s.end, end) - max(s.start, start)
        for s in covered if s.kind in ("drop", "build", "verse")
    )
    idle = sum(
        min(s.end, end) - max(s.start, start)
        for s in covered if s.kind in ("break", "intro")
    )
    score += 1.2 * (active / target)
    if idle > target * 0.4:
        score -= 1.5 * (idle / target)

    opening = next((s for s in covered if s.start <= start < s.end), None)

    # Prefer starting on a section boundary, ideally a quiet one so the
    # excerpt has somewhere to build from.
    if round(float(start), 2) in boundaries:
        score += 0.5
        if opening and opening.kind in lead_in_kinds:
            score += 1.0
    # Opening cold in the middle of a drop gives the show no lead-in and
    # nowhere to go; a loud window is not automatically a good one.
    if opening and opening.kind == "drop":
        score -= 0.8 if round(float(start), 2) in boundaries else 1.4

    # Ending mid-drop is an abrupt cut; prefer landing somewhere resolved.
    closing = next((s for s in covered if s.start < end <= s.end), None)
    if closing and closing.kind in ("outro", "break"):
        score += 0.3

    # Landing on a section edge is the whole point of a variable length:
    # given the choice, stretch or shrink so the window ends where the music
    # does rather than partway through a drop.
    if round(end, 2) in ends or round(end, 2) in boundaries:
        score += 0.8
    elif closing and closing.kind == "drop":
        # Clipped mid-drop. How bad depends on how little of it we kept.
        kept = (end - max(closing.start, start)) / max(closing.duration, 1e-6)
        if kept < 0.6:
            score -= 1.0 * (1.0 - kept)

    # Avoid starting in the last third; endings tend to trail off.
    if start > f.duration * 0.75:
        score -= 0.5

    return score


def clip_sections(sections: list[Section], start: float,
                  duration: float) -> list[Section]:
    """Trim sections to the excerpt window and rebase them to t=0."""
    end = start + duration
    out = []
    for s in sections:
        if s.end <= start or s.start >= end:
            continue
        clipped = Section(
            start=max(s.start, start) - start,
            end=min(s.end, end) - start,
            kind=s.kind,
            energy=s.energy,
            brightness=s.brightness,
        )
        if clipped.duration > 0.5:
            out.append(clipped)

    if not out:
        return []

    # Close any gaps left by the trim so the timeline is continuous.
    out[0].start = 0.0
    out[-1].end = duration
    for a, b in zip(out, out[1:]):
        a.end = b.start
    return out
