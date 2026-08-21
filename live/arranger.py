"""State and clock in, pixels out -- the offline recipes as a stepping loop.

The offline arranger has the whole song and writes a timeline.  This has the
present moment and a prediction of the next beat, so the same ideas are
expressed as "given where we are, what should this frame look like":

* the **palette journey** advances on each state change rather than each
  section boundary, so the show still travels through colour;
* the **corridor phrase** rotates every few bars, picking a pattern by density
  the same way the offline arranger does, so the tunnel keeps introducing new
  material instead of looping one gesture;
* **articulation** comes from how clear the pulse actually is right now
  (`clock.confidence`) rather than from an offline `drive` number.

Everything time-varying is driven from :meth:`live.clock.BeatClock.phase` and
``bar_phase``, which are *predictions*.  That is the entire point of M4: an
accent placed at ``phase == 0`` lands with the room rather than a pipeline's
worth of latency behind it.

When the clock is not confident, the arranger leans on energy instead of the
grid -- beat-locked strobing off a wrong grid looks far worse than a wash that
merely breathes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import effects as fx
from . import palette as pal
from .analysis import Features
from .frame import Canvas
from .listener import Listener
from .settings import Settings
from .state import BUILDING, CRUISING, HOT, QUIET, SILENT, StateMachine


@dataclass(frozen=True)
class Treatment:
    """How one state looks.  The offline show's section recipes, condensed."""

    kind: str
    scheme: str
    value: float
    brightness: float
    #: Corridor pattern when articulation sits at its midpoint.
    pattern: str
    #: Bars per corridor phrase.
    phrase_bars: float
    #: Never strobe the corridor here, however the numbers fall out.
    quiet: bool = False


TREATMENTS: dict[str, Treatment] = {
    SILENT:   Treatment(SILENT,   "analogous",     0.45, 0.40, "comet",    16.0, True),
    QUIET:    Treatment(QUIET,    "analogous",     0.55, 0.55, "sparkle",   4.0, True),
    CRUISING: Treatment(CRUISING, "split",         0.85, 0.80, "comet",     4.0),
    BUILDING: Treatment(BUILDING, "complementary", 0.95, 0.90, "pairs",     2.0),
    HOT:      Treatment(HOT,      "triadic",       1.00, 1.00, "alternate", 1.0),
}


class Arranger:
    """Drop-in replacement for :class:`live.script.Script`, driven by audio."""

    def __init__(self, canvas: Canvas, listener: Listener, *,
                 settings: Settings | None = None,
                 state: StateMachine | None = None,
                 base_hue: float = 190.0, seed: int = 7) -> None:
        self.canvas = canvas
        self.listener = listener
        self.settings = settings
        self.machine = state or StateMachine()
        self.base_hue = base_hue
        self.seed = seed
        self.big = canvas.net_slice("Big Triangle")
        self.small = canvas.net_slice("Small Triangle Nets")
        self._palettes: dict[tuple, pal.Palette] = {}
        #: Advances once per state change -- the hue journey's step counter.
        self.journey = 0
        self._last_state = self.machine.state

    # -- knobs and derived values ------------------------------------------ #

    @property
    def clock(self):
        return self.listener.clock

    @property
    def features(self) -> Features | None:
        return self.listener.features

    @property
    def bpm(self) -> float:
        return self.clock.tempo

    @property
    def duration(self) -> float:
        return 0.0          # a live show does not have one

    def beat_phase(self, t: float) -> float:
        return self.clock.phase(t)

    def treatment(self) -> Treatment:
        held = self.settings.scene if self.settings else "auto"
        if held != "auto" and held in TREATMENTS:
            return TREATMENTS[held]
        return TREATMENTS.get(self.machine.state, TREATMENTS[CRUISING])

    def locate(self, t: float):
        """(journey index, treatment, progress through the current phrase)."""
        treat = self.treatment()
        rate = self.settings.corridor_rate if self.settings else 1.0
        bars = self.clock.bar_phase(t) + self._bars_elapsed(t)
        phrase = max(0.25, treat.phrase_bars / max(rate, 1e-3))
        return self.journey, treat, (bars % phrase) / phrase

    def _bars_elapsed(self, t: float) -> float:
        index = self.clock.beat_index_at(t)
        return (index - self.clock.downbeat) // self.clock.bar_length

    def palette_for(self, treat: Treatment) -> pal.Palette:
        offset = self.settings.hue_offset if self.settings else 0.0
        lock = self.settings.hue_lock if self.settings else False
        hue = self.base_hue + offset + (0.0 if lock
                                        else self.journey * pal.GOLDEN_ANGLE)
        key = (round(hue, 2), treat.scheme, treat.value, treat.kind)
        cached = self._palettes.get(key)
        if cached is None:
            if len(self._palettes) > 256:
                self._palettes.clear()
            cached = pal.generate(hue, treat.scheme, value=treat.value,
                                  white=treat.kind == HOT).floored()
            self._palettes[key] = cached
        return cached

    def pattern_for(self, treat: Treatment) -> str:
        if self.settings is not None and self.settings.pattern != "auto":
            return self.settings.pattern
        articulation = self.settings.articulation if self.settings else 0.5
        # A grid we do not trust should not drive busy, tightly-placed
        # patterns; fall back toward the sparse end instead.
        trust = self.clock.confidence
        target = (fx.DENSITY[treat.pattern] + (articulation - 0.5)) * (0.5 + 0.5 * trust)
        return fx.vocabulary(min(max(target, 0.0), 1.0), quiet=treat.quiet)[0]

    # -- render ------------------------------------------------------------ #

    def render(self, frame: int, t: float) -> None:
        canvas = self.canvas
        canvas.clear()

        if self.machine.state != self._last_state:
            self.journey += 1
            self._last_state = self.machine.state

        _, treat, phrase = self.locate(t)
        if treat.kind == SILENT:
            # Nothing is playing, so nothing here is driven by the beat clock:
            # it is free-running on no evidence and following it would make the
            # rig twitch at an imaginary tempo.  Everything below is a function
            # of wall time, slow enough that you have to watch to see it move.
            self._silent(t)
            return
        palette = self.palette_for(treat)
        far = palette.rotated(70.0)
        beat = self.clock.phase(t)
        bar = self.clock.bar_phase(t)
        kick = _kick(beat)
        features = self.features

        levels = fx.PATTERNS[self.pattern_for(treat)](
            len(canvas.arch_names), phrase,
            **({"seed": self.seed} if self.pattern_for(treat) == "sparkle" else {}))
        fx.corridor(canvas, levels, palette, far,
                    brightness=treat.brightness, height=0.35)

        getattr(self, f"_{treat.kind}")(frame, t, phrase, palette, beat, bar,
                                        kick, features)

    # Each treatment is the offline show's recipe for that kind of section.

    #: Seconds for one full traverse of the corridor while idle, and for one
    #: turn of the hue.  Both deliberately long: this is what the rig does
    #: between sets, and anything that reads as "an effect" is wrong here.
    IDLE_SWEEP_S = 48.0
    IDLE_HUE_S = 300.0

    def _silent(self, t: float) -> None:
        canvas = self.canvas
        offset = self.settings.hue_offset if self.settings else 0.0
        lock = self.settings.hue_lock if self.settings else False
        drift = 0.0 if lock else 360.0 * (t / self.IDLE_HUE_S)
        palette = pal.generate(self.base_hue + offset + drift, "analogous",
                               value=0.45).floored()
        far = palette.rotated(40.0)

        # One smooth swell travelling the tunnel, never fully off at either
        # end -- a corridor that goes dark reads as a fault rather than as
        # rest.  No pattern function: those all have discrete steps in them.
        phase = t / self.IDLE_SWEEP_S
        levels = 0.38 + 0.30 * np.sin(
            2 * np.pi * (canvas.depth * 0.6 - phase)).astype(np.float32)
        fx.corridor(canvas, levels, palette, far, brightness=0.40, height=0.25)

        # A very slow plasma, and nothing else: no sparkle, no accents, no
        # per-net trading.  Barely changing is the point.
        fx.plasma(canvas, palette, t, scale=1.6, speed=0.06, level=0.45)
        breath = 0.5 + 0.5 * float(np.sin(2 * np.pi * t / (self.IDLE_SWEEP_S / 2)))
        fx.par(canvas, palette.color(0), 0.10 + 0.08 * breath)

    def _quiet(self, frame, t, phrase, palette, beat, bar, kick, features) -> None:
        fx.plasma(self.canvas, palette, t, scale=2.5, speed=0.35, level=0.7)
        fx.net_sparkle(self.canvas, pal.WHITE, frame, density=0.006, level=0.5,
                       seed=self.seed)
        fx.par(self.canvas, palette.color(0), 0.25 + 0.15 * phrase)

    def _cruising(self, frame, t, phrase, palette, beat, bar, kick, features) -> None:
        fx.wash(self.canvas, palette.dimmed(0.35), 1.0, gradient=0.8)
        # Big and small nets trade bars, as in the offline show.  This is the
        # first thing that needs a *correct* bar line rather than a consistent
        # one -- counted from the wrong beat it trades on the offbeat.
        lead = self.big if int(self._bars_elapsed(t)) % 2 == 0 else self.small
        fx.bars(self.canvas, palette, phrase * 4.0, count=3, angle=0.15,
                width=0.3, level=0.9, targets=lead)
        fx.par(self.canvas, palette.color(1), 0.35 + 0.45 * kick)

    def _building(self, frame, t, phrase, palette, beat, bar, kick, features) -> None:
        # Tension tracks how far into the build we are, but it is *abortable*:
        # if the sweep stops without a drop, this simply relaxes.  Never
        # pre-fire the resolution.
        tension = min(1.0, self.machine.report.since_s / 8.0)
        fx.wash(self.canvas, palette.dimmed(0.25), 1.0, gradient=1.0)
        fx.pinwheel(self.canvas, palette, t * (1.0 + 5.0 * tension) * 0.35,
                    arms=3, level=0.8, targets=self.big)
        rate = 2 + int(tension * 6)
        flash = _kick((t / max(1e-6, 60.0 / self.bpm) * rate) % 1.0, sharp=3.0)
        fx.wash(self.canvas, pal.WHITE, 0.55 * flash * tension, targets=self.small)
        fx.par(self.canvas, palette.color(0), 0.4 + 0.6 * tension * flash,
               white=0.3 * tension)

    def _hot(self, frame, t, phrase, palette, beat, bar, kick, features) -> None:
        fx.radial(self.canvas, palette, (t * 2.0) % 1.0, width=0.3, level=0.9)
        fx.wash(self.canvas, palette.dimmed(0.3), kick, targets=self.big)
        fx.pinwheel(self.canvas, palette, -t * 0.8, arms=5, level=0.6,
                    targets=self.small)
        fx.net_sparkle(self.canvas, pal.WHITE, frame, density=0.02, level=0.9,
                       seed=self.seed)
        fx.par(self.canvas, pal.WHITE.color(0), kick, white=kick)


def _kick(phase: float, sharp: float = 2.0) -> float:
    return float(max(0.0, 1.0 - phase) ** sharp)
