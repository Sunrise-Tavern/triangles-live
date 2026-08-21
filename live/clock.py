"""The beat clock: tempo, phase, and the *next* beat.

This is the piece the whole design rests on.  A tracker tells you a beat has
happened, which is already too late -- by the time audio has been captured,
analysed, turned into pixels and pushed over DDP, a light fired on detection
lands visibly behind the room.  So the clock does not report beats; it
maintains a model (a tempo and an anchor) and *extrapolates*, and the renderer
asks "where are we now" rather than being told "a beat just happened".

Three consequences worth stating, because they drive the code:

* **Detection latency stops mattering for firing.**  A beat noticed 40 ms late
  still refines the same model, and the model's next prediction is on time.
  It only has to be accurate, not prompt.
* **Losing the tracker is survivable.**  With a model, silence means "keep
  going at the last known tempo", not "stop".  Live DJ sets defeat beat
  trackers routinely -- breakdowns, transitions, tempo nudges -- so free-run
  and re-lock are the normal case, not the error case.
* **Confidence is a first-class output.**  M5's arranger should lean on the
  grid when it is trustworthy and fall back to energy-driven behaviour when it
  is not, so the clock has to say how sure it is.

Phase locking is a PLL rather than a fit over a window of beats: it degrades
gracefully (a bad beat nudges rather than jumps), it costs nothing, and its
one tunable -- how hard to pull toward each observation -- is exactly the
"trust the tracker vs trust the model" dial the problem actually has.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .beats import BeatEvent

#: Dance music, folded into one octave.  A tracker reporting 64 or 256 BPM has
#: almost always halved or doubled, and following it there makes the corridor
#: crawl or strobe for no musical reason.
TEMPO_MIN = 70.0
TEMPO_MAX = 180.0

#: Ratios at which a tracker's tempo readout is describing the *same* music at
#: a different metrical level -- counting half-bars, or triplets -- rather than
#: reporting that the music changed speed.  Measured on a 140 BPM track,
#: aubio's readout alternates between 142 and 94, and 142 x 2/3 = 95: it is
#: changing its mind about the pulse, not hearing a tempo change.  Following
#: that is how a clock ends up wandering 13 BPM across a track that never
#: varied.
#:
#: A real tempo change does not land here.  A DJ nudging pitch moves a few
#: percent, and 128 -> 140 is a ratio of 1.09, nowhere near any of these.
METRICAL_RATIOS = (1 / 3, 1 / 2, 2 / 3, 3 / 4, 4 / 3, 3 / 2, 2.0, 3.0)


def _metrical(hint: float, tempo: float, tolerance: float) -> bool:
    """Is ``hint`` the same pulse counted differently, rather than a new tempo?"""
    if tempo <= 1e-6 or hint <= 1e-6:
        return False
    ratio = hint / tempo
    return any(abs(ratio - r) <= tolerance * r for r in METRICAL_RATIOS)


@dataclass
class ClockState:
    tempo: float
    confidence: float
    beat: int
    bar: int
    beat_phase: float
    bar_phase: float
    next_beat: float
    locked: bool
    free_running: bool
    since_beat: float

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class BeatClock:
    tempo: float = 128.0
    #: Show time of the beat numbered :attr:`anchor_index`.
    anchor: float = 0.0
    anchor_index: int = 0
    confidence: float = 0.0
    bar_length: int = 4
    #: Beat number that starts a bar.  M5 re-anchors this on a downbeat.
    downbeat: int = 0

    #: How hard to pull the model toward each observed beat.  Low when
    #: confident (the model is better than any single observation), high when
    #: not (the observation is all we have).
    lock_gain: float = 0.12
    loose_gain: float = 0.5
    #: How fast the period follows the *residuals* -- the frequency half of the
    #: loop.  Much smaller than the phase gain: a loop that chases tempo as
    #: eagerly as phase oscillates instead of settling.
    freq_gain: float = 0.04
    #: Beyond this fraction of a period, an observation is not this beat.
    tolerance: float = 0.3
    #: Consecutive mismatches before giving up and snapping to the tracker.
    relock_after: int = 3
    #: Beats of silence before the clock calls itself free-running.
    free_run_beats: float = 3.0
    #: Confidence half-life while free-running, seconds.
    decay_s: float = 6.0
    #: Shift every prediction by this much, to line the lights up with the PA.
    latency: float = 0.0
    tempo_range: tuple[float, float] = (TEMPO_MIN, TEMPO_MAX)

    # -- offbeat detection ------------------------------------------------- #
    #
    # A half-beat slip is the failure that matters here: the grid stays
    # plausible, the tempo stays right, and every effect fires on the wrong
    # half of the bar.  It happens where a tracker has least to work with --
    # a breakdown with no kick, where hats and pads sit on the offbeat -- and
    # once the model has moved there nothing about the timing looks wrong.
    #
    # Phase alone cannot resolve it, because both answers fit the beat grid
    # equally well.  Low-end can: for the music this rig plays, the kick is on
    # the beat.  So the clock watches how much bass arrives near its predicted
    # beat versus near the halfway point, and if the offbeat is consistently
    # heavier it moves.
    #: How much stronger the offbeat must be before the clock believes it.
    polarity_ratio: float = 1.35
    #: Evidence window, seconds.
    polarity_tau: float = 4.0
    #: How long the offbeat must stay ahead before shifting.
    polarity_hold: float = 1.5
    #: Evidence multiplier while confidence is low.
    polarity_unsure: float = 2.0
    #: Relative tempo disagreement that counts as a real change rather than
    #: the tracker's standing bias.
    tempo_jump: float = 0.04
    #: ...and how many beats it must persist for.
    tempo_jump_beats: int = 3
    #: How hard to jump once convinced.  The gate above is what makes this
    #: safe to make decisive: bias never trips it, a real change always does.
    tempo_jump_gain: float = 0.5
    #: How close to a metrical ratio counts as one, relative.
    metrical_tolerance: float = 0.06
    #: Below this confidence, a metrical disagreement is followed anyway --
    #: otherwise a clock that locked onto the wrong pulse to begin with could
    #: never be talked out of it.
    metrical_trust: float = 0.6
    #: How many beats of sustained metrical disagreement to hold out for.
    #: Without a limit the guard cuts both ways: if the clock ever settles on
    #: the wrong pulse, every correction back looks metrical and gets refused
    #: forever.  Measured on one track the clock sat at 149 BPM while the
    #: tracker said 117 -- a 3/4 ratio -- and the guard kept it there.
    #: Holding against a momentary flip is the point; holding against a
    #: tracker that has said the same thing for twenty-odd beats is not.
    metrical_patience: int = 20
    metrical_rejects: int = 0
    #: Accepted beats of history before the long-baseline estimate is trusted.
    baseline_beats: int = 24
    baseline_gain: float = 0.12
    #: Confidence is otherwise a measure of *self-consistency* -- do incoming
    #: beats fit the model we already hold -- which a slow drift satisfies
    #: perfectly.  Fed a grid sliding 121 to 170 BPM, the clock reported 0.99
    #: throughout, and the arranger uses that number to decide whether to trust
    #: the grid at all.  So a tempo that will not sit still caps it.
    stability_beats: int = 12
    #: Relative spread across those beats at which confidence is fully capped.
    stability_spread: float = 0.06
    block_s: float = 512 / 44100
    slips: int = 0
    offbeat_events: int = 0

    last_beat_t: float = -1e9
    beats_seen: int = 0
    relocks: int = 0
    _misses: int = 0
    _now: float = 0.0
    _last_tick: float | None = None
    _on_energy: float = 0.0
    _off_energy: float = 0.0
    _slip_for: float = 0.0
    _disagree: int = 0
    _metrical_run: int = 0
    #: (time, beat index) for accepted beats -- the long tempo baseline.
    history: list[tuple[float, int]] = field(default_factory=list)
    #: Recent tempo readings, for the stability cap on confidence.
    _tempo_log: list[float] = field(default_factory=list)

    # -- model ------------------------------------------------------------- #

    @property
    def period(self) -> float:
        return 60.0 / self.tempo

    def fold_tempo(self, bpm: float) -> float:
        """Bring a tempo into the working octave without changing its phase."""
        low, high = self.tempo_range
        if not (bpm and bpm == bpm and bpm > 1.0):     # zero, NaN, nonsense
            return self.tempo
        while bpm < low:
            bpm *= 2.0
        while bpm > high:
            bpm /= 2.0
        return bpm

    def beat_index_at(self, now: float) -> int:
        return self.anchor_index + int((now + self.latency - self.anchor)
                                       // self.period)

    def beat_time(self, index: int) -> float:
        return self.anchor + (index - self.anchor_index) * self.period

    def phase(self, now: float) -> float:
        """Position within the current beat, 0 on the beat, approaching 1."""
        return ((now + self.latency - self.anchor) / self.period) % 1.0

    def audio_phase(self, now: float) -> float:
        """Phase in the *audio's* timeline, with no output latency applied.

        :attr:`latency` exists to line the lights up with the PA; measuring
        what the music is doing must not be shifted by it.
        """
        return ((now - self.anchor) / self.period) % 1.0

    def bar_phase(self, now: float) -> float:
        """Position within the current bar, 0 on the downbeat."""
        offset = (self.beat_index_at(now) - self.downbeat) % self.bar_length
        return (offset + self.phase(now)) / self.bar_length

    def next_beat(self, now: float) -> float:
        """Show time of the next beat -- the thing effects should fire on."""
        return self.beat_time(self.beat_index_at(now) + 1) - self.latency

    def crossed(self, since: float, now: float) -> list[int]:
        """Beat indices falling in ``(since, now]`` -- one call per frame.

        Returns indices rather than times so a caller can tell a downbeat from
        an offbeat without recomputing anything.
        """
        first = self.beat_index_at(since) + 1
        last = self.beat_index_at(now)
        return list(range(first, last + 1)) if last >= first else []

    # -- updates ----------------------------------------------------------- #

    def on_beat(self, event: BeatEvent) -> bool:
        """Fold an observed beat into the model.  True if it fitted.

        A beat that does not fit is not immediately believed: a single spurious
        detection during a breakdown should not throw away a grid that has been
        right for a minute.  Three in a row is a different story -- that is a
        transition, and then the tracker is right and the model is stale.
        """
        t = event.t
        if t <= self.last_beat_t:            # duplicate or out of order
            return False
        self.beats_seen += 1
        self.last_beat_t = t

        hint = self.fold_tempo(event.tempo)

        if self.confidence <= 0.01 and self.beats_seen <= 1:
            self._snap(t, hint)
            return True

        steps = round((t - self.anchor) / self.period)
        if steps <= 0:
            return False
        error = t - (self.anchor + steps * self.period)
        relative = error / self.period

        if abs(relative) > self.tolerance:
            fraction = relative - round(relative)
            if abs(abs(fraction) - 0.5) < 0.15:
                # An offbeat detection -- routine in a breakdown, where the
                # kick drops out and the tracker latches onto hats or a pad.
                # It is not evidence of a tempo change, and letting it count
                # toward a re-lock is how a grid ends up half a beat out with
                # the tempo still right and nothing looking wrong.
                #
                # But it is still a phase reference: an offbeat sits exactly
                # half a period from a beat, so it pins the grid just as well
                # once you account for the offset.  Ignoring these outright
                # leaves the model with nothing to correct against, and it
                # free-runs into the very half-beat error it was trying to
                # avoid -- measured, 45 seconds of that got it there exactly.
                # Which half is the beat is a separate question, settled by
                # observe() on the low end where the kick is.
                self.offbeat_events += 1
                target = 0.5 if fraction > 0 else -0.5
                offset = (fraction - target) * self.period
                gain = self.lock_gain if self.confidence > 0.5 else self.loose_gain
                self.anchor += gain * offset
                period = self.period + self.freq_gain * offset / max(1, steps)
                self.tempo = min(max(60.0 / period, self.tempo_range[0]),
                                 self.tempo_range[1])
                return False
            self._misses += 1
            self.confidence *= 0.6
            if self._misses >= self.relock_after:
                self.relocks += 1
                self._snap(t, hint)
            return False

        self._misses = 0
        period = self.period
        gain = self.lock_gain if self.confidence > 0.5 else self.loose_gain
        self.anchor += steps * period + gain * error
        self.anchor_index += steps

        # Frequency, in two speeds.
        #
        # Fine: the residuals.  A beat that lands consistently late means the
        # period is short.  This is used in preference to the tracker's own BPM
        # readout because following that continuously imports its bias -- aubio
        # reports 129.8 for a track that is exactly 128.0, which is 1.4% and
        # costs about 6 ms on every prediction.
        period += self.freq_gain * error / steps
        tempo = 60.0 / period

        # Coarse: the tracker's readout, but only when it disagrees by more
        # than bias could explain, and keeps disagreeing.  A phase loop alone
        # cannot follow a real tempo change -- the phase term absorbs the error
        # each beat, leaving almost nothing to drive the frequency term, so it
        # crawls.  Measured on a 128 -> 140 step it reached 130.9 and stalled,
        # which tracks each beat while predicting the next one 116 ms wrong.
        if abs(hint - tempo) > self.tempo_jump * tempo:
            if (self.confidence > self.metrical_trust
                    and self._metrical_run < self.metrical_patience
                    and _metrical(hint, tempo, self.metrical_tolerance)):
                # The tracker is counting a different pulse, not hearing a
                # different tempo.  Our own estimate comes from observed beat
                # times and is anchored to the grid that is currently working,
                # so it wins -- but only while we are confident in it, or a
                # bad initial lock could never be corrected.
                self.metrical_rejects += 1
                self._metrical_run += 1
                self._disagree = 0
            else:
                self._metrical_run = 0
                self._disagree += 1
            if self._disagree >= self.tempo_jump_beats:
                tempo += (hint - tempo) * self.tempo_jump_gain
                # Those beats were played at the old tempo.  Keeping them
                # would have the long baseline below pulling back toward it
                # for the next half minute.
                self.history.clear()
        else:
            self._disagree = 0

        # Long baseline.  The residual term is an integrator with no damping,
        # so under jitter the tempo does a random walk with nothing to pull it
        # back -- measured, 25 ms of jitter moved it 0.65 BPM over 300 beats.
        # The span between two accepted beats two dozen apart is a far quieter
        # estimate, and unlike the tracker's readout it is unbiased: a constant
        # detection lag cancels out of an interval.
        self.history.append((t, self.anchor_index))
        if len(self.history) > 64:
            del self.history[:-64]
        if len(self.history) >= self.baseline_beats:
            estimate = self._baseline_tempo()
            if estimate:
                tempo += (estimate - tempo) * self.baseline_gain

        self.tempo = min(max(tempo, self.tempo_range[0]), self.tempo_range[1])
        agreement = 1.0 - abs(relative) / self.tolerance
        self.confidence += (1.0 - self.confidence) * 0.25 * max(agreement, 0.1)

        self._tempo_log.append(self.tempo)
        if len(self._tempo_log) > self.stability_beats:
            del self._tempo_log[:-self.stability_beats]
        if len(self._tempo_log) >= self.stability_beats:
            middle = sorted(self._tempo_log)[len(self._tempo_log) // 2]
            spread = (max(self._tempo_log) - min(self._tempo_log)) / max(middle, 1e-6)
            steady = 1.0 - min(spread / max(self.stability_spread, 1e-6), 1.0)
            self.confidence = min(self.confidence, 0.15 + 0.85 * steady)
        return True

    def _baseline_tempo(self) -> float | None:
        """Least-squares tempo over the whole beat history.

        Fitting a line through every beat rather than differencing the two
        ends: with a few dozen beats it is roughly five times quieter, and the
        endpoints of a jittery series are the worst two points to build an
        estimate from.  Written out longhand to keep this module free of numpy
        -- it runs once per beat, on at most 64 points.
        """
        n = len(self.history)
        sum_i = sum_t = sum_ii = sum_it = 0.0
        for t, i in self.history:
            sum_i += i
            sum_t += t
            sum_ii += i * i
            sum_it += i * t
        denominator = n * sum_ii - sum_i * sum_i
        if denominator <= 0:
            return None
        period = (n * sum_it - sum_i * sum_t) / denominator
        return 60.0 / period if period > 1e-6 else None

    def _snap(self, t: float, tempo: float) -> None:
        """Hard re-lock: believe the tracker, keep counting beats."""
        steps = max(1, round((t - self.anchor) / self.period)) if self.beats_seen > 1 else 1
        self.anchor = t
        self.anchor_index += steps
        self.tempo = tempo or self.tempo
        self.confidence = max(self.confidence, 0.2)
        self._misses = 0

    def observe(self, t: float, low: float) -> None:
        """Feed one block's kick strength, for the offbeat check.

        Only the windows around the beat and around the halfway point count;
        what happens between them says nothing about which of the two is the
        beat.
        """
        alpha = 1.0 - math.exp(-self.block_s / max(self.polarity_tau, 1e-6))
        phase = self.audio_phase(t)
        if min(phase, 1.0 - phase) < 0.15:
            self._on_energy += (low - self._on_energy) * alpha
        elif abs(phase - 0.5) < 0.15:
            self._off_energy += (low - self._off_energy) * alpha

    def _check_polarity(self, dt: float) -> None:
        on, off = self._on_energy, self._off_energy
        loud_enough = on + off > 0.2
        # Confidence scales how much evidence is demanded, rather than
        # switching the check off.  A hard gate at 0.4 deadlocks: sitting on
        # the offbeat is exactly what keeps confidence low, so the one
        # mechanism that could fix the phase was disabled precisely when it
        # was needed.  Measured on a 128 BPM track, that left the clock half a
        # beat out for its entire five minutes at confidence 0.19.
        needed = (self.polarity_ratio if self.confidence > 0.4
                  else self.polarity_ratio * self.polarity_unsure)
        if (loud_enough and self.confidence > 0.1 and not self.free_running
                and off > on * needed):
            self._slip_for += dt
            if self._slip_for >= self.polarity_hold:
                self.anchor += self.period / 2.0
                self._on_energy, self._off_energy = off, on
                self._slip_for = 0.0
                self.slips += 1
                # The phase loop is about to see every incoming beat land half
                # a period away.  Tell it this was deliberate, or it spends the
                # next three beats "correcting" the fix.
                self._misses = 0
                self.last_beat_t = self._now
        else:
            self._slip_for = 0.0

    def tick(self, now: float) -> None:
        """Advance wall time.  Call once a frame, beats or no beats."""
        dt = 0.0 if self._last_tick is None else max(0.0, now - self._last_tick)
        self._last_tick = now
        self._now = now
        self._check_polarity(dt)
        if now - self.last_beat_t > self.free_run_beats * self.period:
            # Halve the confidence every decay_s, but never quite reach zero:
            # a stale grid is still better than no grid through a breakdown,
            # and it is what lets the clock re-lock instead of cold-starting.
            self.confidence = max(0.02,
                                  self.confidence * 0.5 ** (dt / self.decay_s))

    @property
    def free_running(self) -> bool:
        return self._now - self.last_beat_t > self.free_run_beats * self.period

    @property
    def locked(self) -> bool:
        return self.confidence >= 0.5 and not self.free_running

    def state(self, now: float | None = None) -> ClockState:
        now = self._now if now is None else now
        index = self.beat_index_at(now)
        return ClockState(
            tempo=self.tempo, confidence=self.confidence, beat=index,
            bar=(index - self.downbeat) // self.bar_length,
            beat_phase=self.phase(now), bar_phase=self.bar_phase(now),
            next_beat=self.next_beat(now), locked=self.locked,
            free_running=self.free_running,
            since_beat=max(0.0, now - self.last_beat_t),
        )

    def set_downbeat(self, now: float) -> None:
        """Declare that the beat around ``now`` starts a bar."""
        self.downbeat = self.beat_index_at(now)
