#!/usr/bin/env python3
"""Synthesize a structured test track, so the pipeline can be exercised offline.

Not musical, but it has what the analyser looks for: a steady 128 BPM grid,
four-on-the-floor kicks, hats, and sections that differ in energy and
brightness.  Useful for checking structure detection without downloading
anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
BPM = 128.0
BEAT = 60.0 / BPM
BAR = BEAT * 4

# (kind, bars) -- a conventional dance arrangement.
ARRANGEMENT = [
    ("intro", 8), ("verse", 8), ("build", 8), ("drop", 16),
    ("break", 8), ("build", 8), ("drop", 16), ("outro", 8),
]

LEVELS = {
    "intro": dict(kick=0.0, bass=0.25, hat=0.15, lead=0.0, noise=0.02),
    "verse": dict(kick=0.7, bass=0.5, hat=0.3, lead=0.15, noise=0.03),
    "build": dict(kick=0.5, bass=0.4, hat=0.55, lead=0.35, noise=0.12),
    "drop":  dict(kick=1.0, bass=0.9, hat=0.6, lead=0.7, noise=0.06),
    "break": dict(kick=0.0, bass=0.2, hat=0.1, lead=0.3, noise=0.02),
    "outro": dict(kick=0.3, bass=0.2, hat=0.15, lead=0.05, noise=0.02),
}


def env(n: int, attack: int, decay: int) -> np.ndarray:
    e = np.ones(n)
    a = min(attack, n)
    e[:a] = np.linspace(0, 1, a)
    d = min(decay, n)
    e[-d:] *= np.linspace(1, 0, d)
    return e


def kick(dur: float = 0.18) -> np.ndarray:
    n = int(SR * dur)
    t = np.arange(n) / SR
    freq = 150 * np.exp(-t * 40) + 45
    tone = np.sin(2 * np.pi * np.cumsum(freq) / SR)
    return tone * np.exp(-t * 18)


def hat(dur: float = 0.05) -> np.ndarray:
    n = int(SR * dur)
    rng = np.random.default_rng(1)
    return rng.standard_normal(n) * np.exp(-np.arange(n) / SR * 90)


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "test_track.wav")

    total_bars = sum(b for _, b in ARRANGEMENT)
    total = total_bars * BAR
    n = int(SR * total)
    audio = np.zeros(n)

    k, h = kick(), hat()
    rng = np.random.default_rng(7)

    bar_index = 0
    for kind, bars in ARRANGEMENT:
        lv = LEVELS[kind]
        for b in range(bars):
            bar_start = (bar_index + b) * BAR
            frac = b / max(bars - 1, 1)

            for beat in range(4):
                t0 = bar_start + beat * BEAT
                i0 = int(t0 * SR)

                if lv["kick"] > 0:
                    _add(audio, k * lv["kick"], i0)

                # Builds add a snare roll that subdivides as they progress.
                div = 1 if kind != "build" else (1 if frac < 0.5 else 2)
                for s in range(2 * div):
                    j0 = int((t0 + s * BEAT / (2 * div)) * SR)
                    _add(audio, h * lv["hat"], j0)

            # Bass and lead as sustained tones over the bar.
            i0, i1 = int(bar_start * SR), int((bar_start + BAR) * SR)
            seg = np.arange(i1 - i0) / SR
            if lv["bass"] > 0:
                note = 55 * 2 ** (rng.integers(0, 3) / 12)
                _add(audio, np.sin(2 * np.pi * note * seg) * lv["bass"]
                     * env(len(seg), 200, 400), i0)
            if lv["lead"] > 0:
                note = 440 * 2 ** (rng.integers(0, 12) / 12)
                saw = 2 * (seg * note % 1) - 1
                _add(audio, saw * lv["lead"] * 0.3 * env(len(seg), 400, 800), i0)
            if lv["noise"] > 0:
                # Rising noise sweep gives builds their brightness.
                amt = lv["noise"] * (1 + 3 * frac if kind == "build" else 1)
                _add(audio, rng.standard_normal(len(seg)) * amt * 0.3, i0)

        bar_index += bars

    audio /= np.abs(audio).max() * 1.05
    sf.write(out, audio.astype(np.float32), SR)

    print(f"Wrote {out} -- {total:.1f}s, {BPM:.0f} BPM")
    print("Arrangement: " + " | ".join(f"{k}({b})" for k, b in ARRANGEMENT))
    return 0


def _add(buf: np.ndarray, sig: np.ndarray, at: int) -> None:
    end = min(at + len(sig), len(buf))
    if at < len(buf):
        buf[at:end] += sig[:end - at]


if __name__ == "__main__":
    sys.exit(main())
