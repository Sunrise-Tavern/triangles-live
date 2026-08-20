"""Audio feature extraction.

Everything the arranger needs to make the lights land on the music: a beat
grid, bar lines, per-band energy envelopes, and the kick times that accents
hang off.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 22050
HOP = 512

#: Frequency splits for the three energy bands we track.
BANDS = {
    "bass": (20.0, 250.0),
    "mid": (250.0, 4000.0),
    "high": (4000.0, 11000.0),
}


@dataclass
class Features:
    """Analysis results for one track. All times are seconds from the start."""

    y: np.ndarray
    sr: int
    duration: float
    tempo: float
    beat_times: np.ndarray
    bar_times: np.ndarray
    frame_times: np.ndarray
    onset_env: np.ndarray
    rms: np.ndarray
    bands: dict[str, np.ndarray]
    centroid: np.ndarray
    kick_times: np.ndarray
    kick_strength: np.ndarray
    #: Mean energy per pitch class (C, C#, D, ...), normalized 0-1.
    chroma: np.ndarray

    #: Chroma per frame, for per-section harmonic queries.
    chroma_frames: np.ndarray | None = None
    #: How strongly the track states a regular pulse, 0-1. Four-on-the-floor
    #: scores near 1; rubato or texture-led music scores low.
    pulse_clarity: float = 0.5
    #: Detected onsets per beat. Separates busy tracks from sparse ones
    #: independently of tempo.
    onsets_per_beat: float = 2.0

    @property
    def drive(self) -> float:
        """One 0-1 number for how hard the track pushes.

        Combines pulse clarity with onset density. Percussive energy ratio
        (HPSS) looks like the obvious ingredient and is not usable: it measures
        the production more than the genre, and rates a pure-synth four-on-the-
        floor track *lower* than a live-drum one because a sine bass is filed as
        harmonic.
        """
        pulse = np.clip((self.pulse_clarity - 0.30) / 0.65, 0.0, 1.0)
        density = np.clip((self.onsets_per_beat - 0.40) / 2.10, 0.0, 1.0)
        return float(np.clip(0.6 * pulse + 0.4 * density, 0.0, 1.0))

    def quality(self, start: float, end: float) -> float:
        """How major-sounding a span is, -1 (minor) to +1 (major).

        Chroma is matched against major and minor triad templates at all 12
        roots; the score is the normalized gap between the best major fit and
        the best minor fit.

        This is deliberately the *only* harmonic quantity fed to the lighting.
        Chord identity is detectable but not perceivable -- an audience cannot
        connect "that hue" to "that chord" -- whereas major versus minor is felt
        directly as warm versus cold. Driving hue from harmony would also fight
        the per-section hue rotation: most songs sit in one key, so sections
        would collapse back onto a single colour.
        """
        if self.chroma_frames is None or not self.chroma_frames.size:
            return 0.0
        lo, hi = np.searchsorted(self.frame_times, [start, end])
        seg = self.chroma_frames[:, lo:max(hi, lo + 1)]
        if not seg.size:
            return 0.0
        v = seg.mean(axis=1)
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            return 0.0
        v = v / norm

        best_maj = max(float(v @ np.roll(_MAJOR, r)) for r in range(12))
        best_min = max(float(v @ np.roll(_MINOR, r)) for r in range(12))
        total = best_maj + best_min
        if total < 1e-9:
            return 0.0
        return float(np.clip((best_maj - best_min) / total * 6.0, -1.0, 1.0))

    @property
    def key_hue(self) -> float:
        """A base hue in degrees, derived from the track's dominant pitch class.

        Mapping the chromatic circle onto the colour wheel gives each song its
        own colour identity for free, and one that is a property of the music
        rather than an arbitrary constant -- two different tracks reliably open
        in different colours.
        """
        if self.chroma is None or not len(self.chroma):
            return 210.0
        return float(int(np.argmax(self.chroma)) * 30.0)

    @property
    def beat_period(self) -> float:
        """Seconds per beat."""
        if len(self.beat_times) > 1:
            return float(np.median(np.diff(self.beat_times)))
        return 60.0 / max(self.tempo, 1.0)

    @property
    def bar_period(self) -> float:
        if len(self.bar_times) > 1:
            return float(np.median(np.diff(self.bar_times)))
        return self.beat_period * 4

    def sample(self, curve: np.ndarray, t: float) -> float:
        """Read a frame-rate curve at an arbitrary time."""
        return float(np.interp(t, self.frame_times, curve))

    def mean_over(self, curve: np.ndarray, start: float, end: float) -> float:
        """Average a frame-rate curve across a time window."""
        lo, hi = np.searchsorted(self.frame_times, [start, end])
        hi = max(hi, lo + 1)
        seg = curve[lo:hi]
        return float(seg.mean()) if len(seg) else 0.0

    def beats_between(self, start: float, end: float) -> np.ndarray:
        return self.beat_times[(self.beat_times >= start) & (self.beat_times < end)]

    def bars_between(self, start: float, end: float) -> np.ndarray:
        return self.bar_times[(self.bar_times >= start) & (self.bar_times < end)]

    def kicks_between(self, start: float, end: float) -> np.ndarray:
        return self.kick_times[(self.kick_times >= start) & (self.kick_times < end)]

    def beat_grid(self, start: float, end: float, per_beat: int = 1) -> np.ndarray:
        """Real beat times in [start, end), optionally subdivided.

        Subdivisions are interpolated between *adjacent detected beats*, never
        stepped off a constant period.

        Extrapolating `beats[0] + i * beat_period` looks equivalent and is not:
        `beat_period` is the median spacing, real spacing varies either side of
        it, and the difference accumulates. Over a 90-second sequence that walks
        the grid more than three whole beats away from the music -- the lights
        start in time and end audibly late. Anchoring every point to a detected
        beat keeps the error bounded by detection accuracy instead of letting it
        compound.
        """
        b = self.beat_times
        if len(b) < 2:
            return np.array([], dtype=float)

        pad = self.beat_period * 1.5
        sel = b[(b >= start - pad) & (b <= end + pad)]
        if len(sel) < 2:
            return sel[(sel >= start) & (sel < end)]

        if per_beat <= 1:
            pts = sel
        else:
            steps = np.arange(per_beat) / float(per_beat)
            spans = sel[:-1, None] + np.diff(sel)[:, None] * steps[None, :]
            pts = np.append(spans.reshape(-1), sel[-1])

        return pts[(pts >= start) & (pts < end)]

    def snap(self, t: float, start: float, end: float, per_beat: int = 1) -> float:
        """Nearest point on the real beat grid."""
        grid = self.beat_grid(start, end, per_beat)
        if not len(grid):
            return t
        return float(grid[int(np.argmin(np.abs(grid - t)))])

    def snap_to_bar(self, t: float) -> float:
        """Nearest bar line. Sections that do not start on a bar feel late."""
        if len(self.bar_times) == 0:
            return t
        return float(self.bar_times[int(np.argmin(np.abs(self.bar_times - t)))])


def _normalize(x: np.ndarray) -> np.ndarray:
    """Scale to 0-1, robust to outliers."""
    lo = np.percentile(x, 2)
    hi = np.percentile(x, 98)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def analyze(path: Path, *, verbose: bool = True) -> Features:
    import librosa

    if verbose:
        print(f"  loading {path.name}")
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    duration = float(len(y) / sr)

    stft = np.abs(librosa.stft(y, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    frame_times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr, hop_length=HOP)

    onset_env = librosa.onset.onset_strength(S=librosa.power_to_db(stft**2), sr=sr,
                                             hop_length=HOP)

    if verbose:
        print("  tracking beats")
    tempo_raw, beats = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=HOP, trim=False, units="time"
    )
    tempo = float(np.atleast_1d(tempo_raw)[0])
    beat_times = np.asarray(beats, dtype=float)

    bar_times = _find_downbeats(beat_times, onset_env, frame_times)

    rms = _normalize(librosa.feature.rms(S=stft, hop_length=HOP)[0])
    centroid = _normalize(
        librosa.feature.spectral_centroid(S=stft, sr=sr, hop_length=HOP)[0]
    )

    bands = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        bands[name] = _normalize(stft[mask].mean(axis=0))

    chroma_frames = librosa.feature.chroma_stft(S=stft**2, sr=sr)
    chroma = chroma_frames.mean(axis=1)
    chroma = chroma / (chroma.max() or 1.0)

    if verbose:
        print("  separating percussion")
    kick_times, kick_strength = _find_kicks(y, sr)

    pulse_clarity = _pulse_clarity(onset_env, tempo, sr)
    peaks = librosa.util.peak_pick(
        onset_env / (onset_env.max() or 1.0),
        pre_max=4, post_max=4, pre_avg=8, post_avg=8, delta=0.12, wait=4,
    )
    onsets_per_beat = len(peaks) / max(duration * tempo / 60.0, 1e-6)

    if verbose:
        print(f"  {tempo:.1f} BPM, {len(beat_times)} beats, "
              f"{len(bar_times)} bars, {len(kick_times)} onsets")

    return Features(
        y=y, sr=sr, duration=duration, tempo=tempo,
        beat_times=beat_times, bar_times=bar_times,
        frame_times=frame_times, onset_env=onset_env, rms=rms,
        bands=bands, centroid=centroid, chroma=chroma,
        chroma_frames=chroma_frames,
        pulse_clarity=pulse_clarity, onsets_per_beat=onsets_per_beat,
        kick_times=kick_times, kick_strength=kick_strength,
    )


def _pulse_clarity(onset_env: np.ndarray, tempo: float, sr: int) -> float:
    """How strongly the onset envelope repeats at the beat period, 0-1.

    A tight four-on-the-floor autocorrelates hard at the beat lag; music led by
    texture or with a loose feel does not. This is what separates a track that
    wants hits on every beat from one where that would look mechanical.
    """
    import librosa

    if len(onset_env) < 8 or tempo <= 0:
        return 0.5
    centred = onset_env - onset_env.mean()
    ac = librosa.autocorrelate(centred, max_size=len(centred) // 2)
    if not len(ac) or ac[0] <= 0:
        return 0.5
    ac = ac / ac[0]
    lag = int(round((60.0 / tempo) * sr / HOP))
    lo, hi = max(1, lag - 2), min(len(ac), lag + 3)
    if lo >= hi:
        return 0.5
    return float(np.clip(np.max(ac[lo:hi]), 0.0, 1.0))


def _find_downbeats(beat_times: np.ndarray, onset_env: np.ndarray,
                    frame_times: np.ndarray) -> np.ndarray:
    """Pick which of every 4 beats is the downbeat.

    Assumes 4/4.  The downbeat is normally the most strongly accented of the
    four, so we score each of the 4 possible phases by summed onset strength
    and keep the winner.
    """
    if len(beat_times) < 4:
        return beat_times.copy()

    strength = np.interp(beat_times, frame_times, onset_env)
    scores = [strength[phase::4].sum() for phase in range(4)]
    phase = int(np.argmax(scores))
    return beat_times[phase::4]


#: Triad templates as pitch-class masks, rooted on C. Rolling them gives all
#: 12 roots. Major is root/major-third/fifth; minor flattens the third.
_MAJOR = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
_MINOR = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)
_MAJOR /= np.linalg.norm(_MAJOR)
_MINOR /= np.linalg.norm(_MINOR)

#: Only bins below this are considered when hunting for kicks.
KICK_FMAX = 200.0


def _find_kicks(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Kick-drum onsets, used to place accent hits.

    Two filters, both necessary:

    * Harmonic content is stripped so sustained synths and vocals do not
      register as hits.
    * Onset strength is measured over *low frequencies only*. Without this the
      detector fires on hi-hats and every other piece of percussion, so a track
      with busy percussion produces hits several times a second regardless of
      its tempo -- the lights end up tracking the hats instead of the pulse,
      and the show visibly speeds up whenever the percussion gets denser even
      though the BPM never changed.
    """
    import librosa

    _, y_perc = librosa.effects.hpss(y)
    env = librosa.onset.onset_strength(
        y=y_perc, sr=sr, hop_length=HOP, fmax=KICK_FMAX, n_mels=32
    )
    env = env / (env.max() or 1.0)

    # `wait` is in frames (~23ms each); 8 enforces a ~185ms refractory period,
    # which is under a 16th note at 200 BPM but well above hi-hat spacing.
    peaks = librosa.util.peak_pick(
        env, pre_max=4, post_max=4, pre_avg=8, post_avg=8, delta=0.15, wait=8
    )
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=HOP)
    return np.asarray(times, dtype=float), env[peaks]
