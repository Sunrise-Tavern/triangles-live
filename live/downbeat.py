"""Which beat starts the bar.

The beat clock counts beats consistently but has no idea where a bar begins,
and phrase-level behaviour -- "this build resolves in two bars" -- is worthless
counted from the wrong place.  aubio does not provide downbeats, and the
BeatNet spike (see LIVE_PLAN.md) found that the modes we could actually run
live do not either.  So it is inferred here, from the same kind of evidence
that resolved half-beat polarity in :mod:`live.clock`, one level up.

**Transient energy alone cannot do it.**  In four-on-the-floor every beat
carries the same kick, so "which beat is loudest" is noise.  Two cues that do
carry bar information, and which fail in different places, so both are used:

* ``kick`` -- bass-band transient.  Works when the bar line is reinforced, a
  bass note struck with the kick or a heavier first beat.  Useless when every
  kick is identical.
* ``novelty`` -- how much the *shape* of the spectrum differs from the last
  couple of seconds.  A kick that repeats every beat is part of that average
  and does not register; a bassline changing note, or a new loop starting,
  does.  This is the cue that survives a machine-perfect drum pattern.

Measured on the synthetic 128 BPM track, taking each beat's peak: ``kick``
leads the other three bar positions by 1.66x and ``novelty`` by 1.27x, both
pointing at the true bar line, while broadband onset (1.07x) and the mid and
high bands (~1.0x) say nothing at all.

Moving the bar line is deliberately reluctant.  Being wrong for a few bars is
survivable; flipping back and forth across a phrase boundary is not, because
every phrase-aligned decision downstream inherits the wobble.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .analysis import Features
from .clock import BeatClock


@dataclass
class BarTracker:
    bar_length: int = 4
    #: Bars of evidence in the running average, per position.
    tau_bars: float = 8.0
    #: How far ahead a challenger must be before the bar line moves.
    margin: float = 1.25
    #: ...and for how many bars it must stay ahead.
    hold_bars: float = 2.0
    #: Relative weight of the two cues.
    kick_weight: float = 1.0
    novelty_weight: float = 1.0

    #: Running evidence per bar position, index 0 being the current downbeat.
    scores: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float32))
    shifts: int = 0
    bars_seen: int = 0

    _beat: int | None = None
    _peak: float = 0.0
    _leader: int = 0
    _leading_for: float = 0.0

    def __post_init__(self) -> None:
        if len(self.scores) != self.bar_length:
            self.scores = np.zeros(self.bar_length, dtype=np.float32)
        self._alpha = 1.0 - math.exp(-1.0 / max(self.tau_bars, 1e-6))

    # -- evidence ---------------------------------------------------------- #

    def score(self, features: Features) -> float:
        return (self.kick_weight * features.kick
                + self.novelty_weight * features.novelty)

    def push(self, features: Features, clock: BeatClock) -> None:
        """One audio block.  Folds into the bar estimate at each beat boundary."""
        index = clock.beat_index_at(features.t)
        if self._beat is None:
            self._beat = index
        elif index != self._beat:
            # Each beat contributes its *peak*, not its mean: the cue is a
            # transient a block or two wide, and averaging over the ~40 blocks
            # in a beat buries it in whatever else is sounding.
            self._commit(clock)
            self._beat = index
            self._peak = 0.0
        self._peak = max(self._peak, self.score(features))

    def _commit(self, clock: BeatClock) -> None:
        position = (self._beat - clock.downbeat) % self.bar_length
        self.scores[position] += (self._peak - self.scores[position]) * self._alpha
        if position == self.bar_length - 1:
            self.bars_seen += 1
        self._decide(clock)

    def _decide(self, clock: BeatClock) -> None:
        if self.bars_seen < 2 or clock.confidence < 0.3:
            # Nothing to say until the beat grid itself is worth counting on.
            return
        best = int(np.argmax(self.scores))
        if best == 0:
            self._leading_for = 0.0
            return
        rest = float(np.mean(np.delete(self.scores, best)))
        if rest <= 1e-9 or self.scores[best] < self.margin * max(self.scores[0], 1e-9):
            self._leading_for = 0.0
            return
        if best != self._leader:
            self._leader, self._leading_for = best, 0.0
        self._leading_for += 1.0 / self.bar_length      # one beat, in bars
        if self._leading_for >= self.hold_bars:
            clock.downbeat += best
            # Rotate the evidence with the bar line, so the accumulated
            # history stays attached to the musical positions it was measured
            # on rather than being thrown away.
            self.scores = np.roll(self.scores, -best)
            self.shifts += 1
            self._leading_for = 0.0
            self._leader = 0

    # -- reporting --------------------------------------------------------- #

    @property
    def confidence(self) -> float:
        """How much the bar line stands out, 0 (nothing) to 1 (unambiguous)."""
        total = float(self.scores.sum())
        if total <= 1e-9 or self.bars_seen < 2:
            return 0.0
        share = float(self.scores[0]) / total
        even = 1.0 / self.bar_length
        return float(np.clip((share - even) / even, 0.0, 1.0))

    def state(self) -> dict:
        return {
            "bar_confidence": round(self.confidence, 3),
            "bar_shifts": self.shifts,
            "bars_seen": self.bars_seen,
            "evidence": [round(float(v), 2) for v in self.scores],
        }
