"""Causal, per-block audio features.

Everything here can be computed from audio that has already happened.  That is
the constraint that separates this from ``triseq/analysis.py``, which is free to
look at a whole track at once and does -- the two are deliberately not shared.

One 2048-point Hann window per 512-sample hop: long enough that the bass band
has usable resolution (21.5 Hz per bin, so a 50 Hz kick is not smeared across
the DC bin), short enough that spectral flux still peaks on the transient.

The numbers that matter downstream are all *relative*:

* ``onset`` is flux over its own rolling mean, so it means the same thing at
  any input level;
* ``energy`` is loudness over a 45-second baseline, so "loud" means loud *for
  this set* rather than loud in dBFS.

That is what lets the state machine in M5 have thresholds at all.  Absolute
levels would just track the DJ's gain knob.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import BLOCKSIZE, SAMPLERATE, Block

WINDOW = 2048

#: Bands in the coarse profile used for novelty.  Enough to separate a bass
#: note change from a hi-hat, few enough that noise averages out.
NOVELTY_BANDS = 16

#: How far below the running baseline a block counts as silence, rather than
#: as a quiet passage worth learning from.
SILENCE_GATE = 0.05

#: Absolute floor, dBFS, below which there is no music playing at all.
#:
#: Every other loudness measure here is relative on purpose, so the show tracks
#: the music and not the DJ's gain knob.  But a purely relative measure cannot
#: detect silence: it normalises by whatever it is hearing, so a dead input
#: reads as perfectly average.  Measured on a silent feed followed by music,
#: `energy` was 1.00 then 1.56 and `level` 1.18 then 1.00 -- neither separates
#: them, while the broadband noise floor made `high_share` read 0.80, i.e. a
#: permanent build.
#:
#: "Is this loud for this set" is relative.  "Is there any signal" is not, and
#: this is the one place an absolute number belongs.
#:
#: Applied to the **smoothed** level, not per-block RMS.  Measured: the median
#: block of a click track is digitally silent (-180 dBFS) because most blocks
#: fall between the clicks, so a per-block threshold calls busy music silent.
#: Smoothed over three seconds the separation is clean -- a dead input sits at
#: -90 dBFS, and music attenuated twentyfold still reads -51 -- so the floor
#: goes between them with about 20 dB of margin either side.
SILENCE_DBFS = -70.0
SILENCE_RMS = 10.0 ** (SILENCE_DBFS / 20.0)

#: (name, low Hz, high Hz).  Bass is the kick, high is where builds live.
BANDS = (("bass", 20.0, 160.0), ("mid", 160.0, 2000.0), ("high", 2000.0, 10000.0))


@dataclass
class Features:
    t: float                # show time of the newest sample in the window
    rms: float
    peak: float
    bass: float
    mid: float
    high: float
    #: Spectral flux, raw.
    flux: float
    #: Flux over its own rolling mean -- 1.0 is "as busy as usual".
    onset: float
    #: How different the spectrum is from what has been playing for the last
    #: couple of seconds, 0..1-ish.  A kick repeats every beat, so it lives in
    #: the average and does *not* spike this; a bassline changing note or a new
    #: loop starting does.  That is the bar-line cue :mod:`live.downbeat` needs
    #: -- transient energy alone cannot find a downbeat in four-on-the-floor,
    #: because every beat has the same kick on it.
    novelty: float
    #: Flux in the bass band only, over its own rolling mean: a kick detector.
    #: Deliberately not the bass *level* -- a bassline plays continuously, so
    #: level barely moves when the kick lands, while the transient is
    #: unmistakable.  This is what tells the clock which half of the beat it
    #: is on.
    kick: float
    #: RMS over a slow *mean* -- 1.0 is "about as loud as usual lately".
    energy: float
    #: RMS over a slowly-decaying *peak* -- 1.0 is "as loud as this set gets".
    #: The mean chases the music, which compresses exactly the distinction that
    #: matters: measured on the test track, a verse and a drop differ by 1.8x
    #: in raw level but only 1.55 vs 1.71 in ``energy``, because the mean has
    #: already risen by the time the drop lands.  Against a peak they are 0.55
    #: and 1.00.
    level: float
    #: Share of spectral energy in the bass band, 0..1, over a short window.
    #: Energy-weighted rather than a running mean of per-frame ratios: in the
    #: gaps between transients almost all that is left is the noise floor,
    #: which is broadband, so averaging the ratios makes any sparse material
    #: look like a build.
    bass_share: float
    #: Share in the high band.  A build sweeps this upward and it is the one
    #: cue that separates a build from a drop without reference to loudness:
    #: 0.57 during the test track's builds against 0.25 or less elsewhere.
    high_share: float
    #: Spectral centroid in Hz; a build sweeps it upward.
    centroid: float
    baseline: float
    #: No music is playing.  Absolute, unlike every other level here.
    silent: bool
    #: How much of the baseline's window has actually been heard, 0..1.
    #: Until this is near 1 the loudness baseline is a small, unrepresentative
    #: sample, and "loud for this set" does not mean anything yet -- the very
    #: start of a track always reads as average, because it is all there is.
    warm: float

    def as_dict(self) -> dict:
        return {k: round(v, 5) for k, v in self.__dict__.items()}


class _Ema:
    """One-pole smoother specified by a time constant, with an honest start.

    A plain EMA seeded at zero spends its first time constant claiming the
    signal is far above baseline -- which for a 45-second baseline means the
    first three quarters of a minute of a set read as the loudest thing that
    has ever happened.  So until it has seen a time constant's worth of blocks
    this is a running mean of everything so far, which is unbiased, and only
    then does it become a forgetting average.
    """

    def __init__(self, tau_s: float, block_s: float, initial: float = 0.0) -> None:
        self.alpha = 1.0 - float(np.exp(-block_s / max(tau_s, 1e-6)))
        self.value = initial
        self.count = 0
        self._warm = max(1, int(round(1.0 / self.alpha)))

    def push(self, x: float) -> float:
        self.count += 1
        if self.count <= self._warm:
            self.value += (x - self.value) / self.count
        else:
            self.value += (x - self.value) * self.alpha
        return self.value


class Analyzer:
    def __init__(self, samplerate: int = SAMPLERATE, blocksize: int = BLOCKSIZE,
                 window: int = WINDOW, baseline_s: float = 45.0,
                 flux_tau_s: float = 1.5, novelty_tau_s: float = 2.0,
                 peak_release_s: float = 120.0) -> None:
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.window = window
        block_s = blocksize / samplerate

        self._buffer = np.zeros(window, dtype=np.float32)
        self._hann = np.hanning(window).astype(np.float32)
        self._prev_mag = np.zeros(window // 2 + 1, dtype=np.float32)
        freqs = np.fft.rfftfreq(window, 1.0 / samplerate)
        self._freqs = freqs
        self._band_slices = [
            (name, np.searchsorted(freqs, lo), np.searchsorted(freqs, hi))
            for name, lo, hi in BANDS
        ]

        self._flux_mean = _Ema(flux_tau_s, block_s, initial=1e-6)
        self._kick_mean = _Ema(flux_tau_s, block_s, initial=1e-6)

        # A coarse log-spaced view of the spectrum, and a slow average of it.
        # Coarse on purpose: at this resolution a note change moves several
        # bins while vibrato and noise do not.
        edges = np.geomspace(40.0, 12000.0, NOVELTY_BANDS + 1)
        self._novelty_bins = [
            (np.searchsorted(freqs, lo), max(np.searchsorted(freqs, hi),
                                             np.searchsorted(freqs, lo) + 1))
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
        self._profile = np.zeros(NOVELTY_BANDS, dtype=np.float32)
        self._profile_alpha = 1.0 - float(np.exp(-block_s / novelty_tau_s))
        self._novelty_mean = _Ema(flux_tau_s * 4, block_s, initial=1e-6)
        self._baseline = _Ema(baseline_s, block_s, initial=0.0)
        # The peak tracks a *smoothed* level, not the raw RMS.  Following raw
        # RMS with a fast attack measures crest factor -- the gap between a
        # kick and the gap after it -- which is a property of the mix, not of
        # the section.  Measured that way a verse and a drop both read 0.85.
        self._smooth_rms = _Ema(3.0, block_s, initial=0.0)
        # Short window: energy-weighting is what makes the shares robust, so
        # the smoothing only has to steady them, not rescue them.  Long enough
        # and the drop arrives before the band shape catches up.
        self._band_ema = {name: _Ema(0.4, block_s, initial=0.0)
                          for name, _, _ in BANDS}
        self._peak = 0.0
        self._peak_up = 1.0 - float(np.exp(-block_s / 5.0))
        self._peak_down = 1.0 - float(np.exp(-block_s / peak_release_s))
        self.blocks = 0

    def push(self, block: Block) -> Features:
        # Slide the window and drop the newest hop in at the end.
        self._buffer[:-self.blocksize] = self._buffer[self.blocksize:]
        self._buffer[-self.blocksize:] = block.samples[:self.blocksize]

        spectrum = np.abs(np.fft.rfft(self._buffer * self._hann)).astype(np.float32)

        # Half-wave rectified flux: only *increases* in energy are onsets.
        # Falling energy is a note ending, which is not an event to fire on.
        diff = spectrum - self._prev_mag
        np.maximum(diff, 0.0, out=diff)
        flux = float(diff.sum()) / self.window
        self._prev_mag = spectrum

        bands = {
            name: float(spectrum[lo:hi].sum()) / self.window
            for name, lo, hi in self._band_slices
        }
        low_lo, low_hi = self._band_slices[0][1], self._band_slices[0][2]
        kick_flux = float(diff[low_lo:low_hi].sum()) / self.window
        total = float(spectrum.sum())
        centroid = (float((self._freqs * spectrum).sum()) / total) if total > 1e-9 else 0.0

        rms = block.rms
        # The baseline only learns from audible passages.  Letting a breakdown
        # pull it down means the first sound afterwards reads as enormous.
        #
        # The gate is a fraction of the baseline itself, not an absolute level.
        # An absolute one would be the single thing in this module that is not
        # level-independent -- and it fails in the direction that matters: a
        # quiet feed from the desk would fall entirely below it, and the
        # baseline would never learn anything at all.
        smooth = self._smooth_rms.push(rms)
        silent = smooth < SILENCE_RMS
        baseline = self._baseline.value
        # Silence must not teach the baseline anything.  Letting it meant that
        # starting the engine before the music left the baseline at the noise
        # floor, so the first track read as 20x "normal" and the state machine
        # sat in `hot` for over a minute.
        if not silent and rms > max(SILENCE_GATE * baseline, 0.0):
            baseline = self._baseline.push(rms)
        flux_mean = self._flux_mean.push(flux)
        kick_mean = self._kick_mean.push(kick_flux)

        profile = np.array([spectrum[lo:hi].mean() for lo, hi in self._novelty_bins],
                           dtype=np.float32)
        total = profile.sum()
        if total > 1e-9:
            profile /= total          # shape, not loudness: novelty must not
        # simply follow the volume, or every drop reads as a bar line.
        raw_novelty = float(np.maximum(profile - self._profile, 0.0).sum())
        self._profile += (profile - self._profile) * self._profile_alpha
        novelty_mean = self._novelty_mean.push(raw_novelty)

        self._peak += (smooth - self._peak) * (self._peak_up if smooth > self._peak
                                               else self._peak_down)
        smoothed_bands = {name: self._band_ema[name].push(value)
                          for name, value in bands.items()}
        total_bands = sum(smoothed_bands.values()) + 1e-12

        self.blocks += 1
        return Features(
            t=block.t + self.blocksize / self.samplerate,
            rms=rms, peak=float(np.abs(block.samples).max()),
            bass=bands["bass"], mid=bands["mid"], high=bands["high"],
            flux=flux, onset=flux / max(flux_mean, 1e-9),
            kick=kick_flux / max(kick_mean, 1e-9),
            novelty=raw_novelty / max(novelty_mean, 1e-9),
            silent=silent,
            warm=min(1.0, self._baseline.count / self._baseline._warm),
            energy=rms / max(baseline, 1e-6),
            level=smooth / max(self._peak, 1e-6),
            bass_share=smoothed_bands["bass"] / total_bands,
            high_share=smoothed_bands["high"] / total_bands,
            centroid=centroid,
            baseline=baseline,
        )
