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

#: How far below the running baseline a block counts as silence, rather than
#: as a quiet passage worth learning from.
SILENCE_GATE = 0.05

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
    #: Flux in the bass band only, over its own rolling mean: a kick detector.
    #: Deliberately not the bass *level* -- a bassline plays continuously, so
    #: level barely moves when the kick lands, while the transient is
    #: unmistakable.  This is what tells the clock which half of the beat it
    #: is on.
    kick: float
    #: RMS over a slow baseline -- 1.0 is "as loud as this set has been".
    energy: float
    #: Spectral centroid in Hz; a build sweeps it upward.
    centroid: float
    baseline: float

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
                 flux_tau_s: float = 1.5) -> None:
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
        self._baseline = _Ema(baseline_s, block_s, initial=0.0)
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
        baseline = self._baseline.value
        if rms > max(SILENCE_GATE * baseline, 0.0) and rms > 0.0:
            baseline = self._baseline.push(rms)
        flux_mean = self._flux_mean.push(flux)
        kick_mean = self._kick_mean.push(kick_flux)

        self.blocks += 1
        return Features(
            t=block.t + self.blocksize / self.samplerate,
            rms=rms, peak=float(np.abs(block.samples).max()),
            bass=bands["bass"], mid=bands["mid"], high=bands["high"],
            flux=flux, onset=flux / max(flux_mean, 1e-9),
            kick=kick_flux / max(kick_mean, 1e-9),
            energy=rms / max(baseline, 1e-6), centroid=centroid,
            baseline=baseline,
        )
