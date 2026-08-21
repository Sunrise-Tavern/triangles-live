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

import zlib
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
    #: Bars for one traverse of the corridor -- how fast the gesture moves.
    phrase_bars: float
    #: Bars before a *different* pattern is drawn.  Separate from the gesture
    #: rate on purpose: a drop wants a fast comet (one bar) but not a new idea
    #: every bar, which reads as thrashing rather than energy.
    pattern_bars: float
    #: Never strobe the corridor here, however the numbers fall out.
    quiet: bool = False


#: Net gestures each state may draw, rotated one per phrase by the same
#: deterministic walk the corridor uses.  The offline show does this too --
#: "butterfly/spirals/fan, rotating every 4 bars" for a drop -- and for the
#: same reason the corridor does: one gesture held for a whole section reads
#: as one idea, however well it tracks the music.
#:
#: Each state's list is ordered loosely from calm to busy, and every gesture in
#: it has to make sense at that energy: a drop can strobe, a breakdown cannot.
NET_GESTURES: dict[str, tuple[str, ...]] = {
    QUIET:    ("plasma", "twinkle", "breathe"),
    CRUISING: ("bars", "trade", "slow_wheel", "rings"),
    BUILDING: ("wheel_up", "strobe_small", "bars_fast"),
    HOT:      ("rings", "fast_wheel", "bars_fast", "flare"),
}


TREATMENTS: dict[str, Treatment] = {
    SILENT:   Treatment(SILENT,   "analogous",     0.45, 0.40, "comet",    16.0, 16.0, True),
    QUIET:    Treatment(QUIET,    "analogous",     0.55, 0.55, "sparkle",   4.0,  8.0, True),
    CRUISING: Treatment(CRUISING, "split",         0.85, 0.80, "comet",     4.0,  4.0),
    BUILDING: Treatment(BUILDING, "complementary", 0.95, 0.90, "pairs",     2.0,  4.0),
    HOT:      Treatment(HOT,      "triadic",       1.00, 1.00, "alternate", 1.0,  4.0),
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
        #: How many times each state has been entered.  Song structure is
        #: repetition with variation -- a second chorus is the first one, more
        #: so -- and while nothing causal can name a section "chorus", the
        #: *return* is detectable by simply counting.  Measured before this
        #: existed, the track's two drops drew identical patterns and identical
        #: gestures and differed only in colour, which had moved by accident.
        self.visits: dict[str, int] = {}
        self.gesture = "bars"
        self._walks: dict[str, tuple[int | None, int]] = {}
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
        bars = self.clock.bar_phase(t) + self._bars_elapsed(t)
        phrase = self._phrase_bars(treat)
        return self.journey, treat, (bars % phrase) / phrase

    def _phrase_bars(self, treat: Treatment) -> float:
        rate = self.settings.corridor_rate if self.settings else 1.0
        return max(0.25, treat.phrase_bars / max(rate, 1e-3))

    def phrase_index(self, t: float, treat: Treatment) -> int:
        """Which phrase we are in.  The corridor draws a new pattern on each.

        Counted in ``pattern_bars``, not ``phrase_bars``: the gesture rate and
        the redraw rate are different questions.  Sharing one number gave a
        drop a new pattern every 1.9 s, which reads as thrashing.
        """
        rate = self.settings.corridor_rate if self.settings else 1.0
        bars = self.clock.bar_phase(t) + self._bars_elapsed(t)
        return int(bars // max(0.5, treat.pattern_bars / max(rate, 1e-3)))

    def _bars_elapsed(self, t: float) -> float:
        index = self.clock.beat_index_at(t)
        return (index - self.clock.downbeat) // self.clock.bar_length

    #: Degrees the hue steps per phrase inside a section, and how many steps
    #: before it comes back.  Small and bounded on purpose: the golden-angle
    #: jump between sections is the journey, and this is only so a ninety
    #: second drop is not one flat colour the whole way through.
    PHRASE_HUE_STEP = 9.0
    PHRASE_HUE_CYCLE = 4

    def palette_for(self, treat: Treatment, phrase: int = 0) -> pal.Palette:
        offset = self.settings.hue_offset if self.settings else 0.0
        lock = self.settings.hue_lock if self.settings else False
        drift = (phrase % self.PHRASE_HUE_CYCLE) * self.PHRASE_HUE_STEP
        hue = self.base_hue + offset + (0.0 if lock
                                        else self.journey * pal.GOLDEN_ANGLE + drift)
        key = (round(hue, 2), treat.scheme, treat.value, treat.kind)
        cached = self._palettes.get(key)
        if cached is None:
            if len(self._palettes) > 256:
                self._palettes.clear()
            cached = pal.generate(hue, treat.scheme, value=treat.value,
                                  white=treat.kind == HOT).floored()
            self._palettes[key] = cached
        return cached

    def _walk(self, key: str, phrase: int, count: int) -> int:
        """A deterministic walk that never lands twice in a row.

        The obvious ``(phrase * step) % count`` does not do this: the step is
        derived per phrase, so two consecutive phrases can land on the same
        index -- measured, a drop drew `bars_fast` for two phrases running.
        Accumulating a step that is never a multiple of ``count`` does
        guarantee it, and stays reproducible because it only ever moves
        forward from a fresh start.
        """
        if count < 2:
            return 0
        last, index = self._walks.get(key, (None, 0))
        if last == phrase:
            return index
        first = phrase if last is None else last + 1
        for step_phrase in range(first, phrase + 1):
            seed = f"{self.seed}:{key}:{step_phrase}".encode()
            index = (index + 1 + zlib.crc32(seed) % (count - 1)) % count
        self._walks[key] = (phrase, index)
        return index

    def gesture_for(self, treat: Treatment, phrase: int) -> str:
        """Which net gesture this phrase draws."""
        options = NET_GESTURES.get(treat.kind, ("bars",))
        visit = self.visits.get(treat.kind, 1)
        return options[self._walk(f"nets:{treat.kind}:{visit}", phrase,
                                  len(options))]

    #: How much busier each return of a state is than the one before, and how
    #: many returns it keeps escalating for.  Small steps: a second drop should
    #: read as more, not as a different show.
    RETURN_STEP = 0.07
    RETURN_CAP = 3

    def escalation(self, treat: Treatment) -> float:
        """0 the first time in a state, rising a little on each return."""
        return self.RETURN_STEP * min(
            max(self.visits.get(treat.kind, 1) - 1, 0), self.RETURN_CAP)

    def pattern_for(self, treat: Treatment, phrase: int = 0) -> str:
        """Which corridor pattern this phrase draws.

        ``vocabulary`` returns the three patterns nearest the energy we want,
        and the corridor takes a different one each phrase rather than the
        nearest one every time.  The offline show learned this: a single
        travelling comet repeated for ninety seconds reads as one idea however
        well it tracks the music.  Holding one pattern for a whole section was
        measured here as four patterns across a hundred and fifty seconds.

        The walk is deterministic -- a step derived from the phrase number,
        never zero -- so consecutive phrases always differ and the whole show
        still renders identically twice, which is what keeps the fseq
        comparison usable as an oracle.  It uses crc32 rather than ``hash()``,
        which is salted per process: with ``hash()`` two runs of the same show
        drew different patterns.
        """
        if self.settings is not None and self.settings.pattern != "auto":
            return self.settings.pattern
        articulation = self.settings.articulation if self.settings else 0.5
        # A grid we do not trust should not drive busy, tightly-placed
        # patterns; fall back toward the sparse end instead.
        trust = self.clock.confidence
        target = ((fx.DENSITY[treat.pattern] + (articulation - 0.5))
                  * (0.5 + 0.5 * trust) + self.escalation(treat))
        candidates = fx.vocabulary(min(max(target, 0.0), 1.0), quiet=treat.quiet)
        # The walk is keyed on the visit as well, so a return does not replay
        # the same sequence of patterns in the same order.
        visit = self.visits.get(treat.kind, 1)
        return candidates[self._walk(f"corridor:{treat.kind}:{visit}", phrase,
                                     len(candidates))]

    # -- render ------------------------------------------------------------ #

    def render(self, frame: int, t: float) -> None:
        canvas = self.canvas
        canvas.clear()

        if self.machine.state != self._last_state:
            self.journey += 1
            self.visits[self.machine.state] = self.visits.get(
                self.machine.state, 0) + 1
            self._last_state = self.machine.state

        _, treat, phrase = self.locate(t)
        if treat.kind == SILENT:
            # Nothing is playing, so nothing here is driven by the beat clock:
            # it is free-running on no evidence and following it would make the
            # rig twitch at an imaginary tempo.  Everything below is a function
            # of wall time, slow enough that you have to watch to see it move.
            self._silent(t)
            return
        beat = self.clock.phase(t)
        bar = self.clock.bar_phase(t)
        kick = _kick(beat)
        features = self.features

        index = self.phrase_index(t, treat)
        self.gesture = self.gesture_for(treat, index)
        name = self.pattern_for(treat, index)
        palette = self.palette_for(treat, index)
        far = palette.rotated(70.0)
        levels = fx.PATTERNS[name](
            len(canvas.arch_names), phrase,
            **({"seed": self.seed} if name == "sparkle" else {}))
        fx.corridor(canvas, levels, palette, far,
                    brightness=min(1.0, treat.brightness + self.escalation(treat)),
                    height=0.35)

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

    def _gesture(self, name: str, frame: int, t: float, phrase: float,
                 palette, beat: float, kick: float, tension: float = 0.0) -> None:
        """Paint one net gesture.  The bed and the par stay with the state."""
        canvas = self.canvas
        if name == "plasma":
            fx.plasma(canvas, palette, t, scale=2.5, speed=0.35, level=0.7)
        elif name == "twinkle":
            fx.plasma(canvas, palette, t, scale=3.0, speed=0.2, level=0.45)
            fx.net_sparkle(canvas, pal.WHITE, frame, density=0.008, level=0.6,
                           seed=self.seed)
        elif name == "breathe":
            swell = 0.35 + 0.35 * float(np.sin(2 * np.pi * phrase))
            fx.wash(canvas, palette, swell, gradient=0.9)
        elif name == "bars":
            fx.bars(canvas, palette, phrase * 4.0, count=3, angle=0.15,
                    width=0.3, level=0.9)
        elif name == "bars_fast":
            fx.bars(canvas, palette, phrase * 8.0, count=4, angle=0.35,
                    width=0.22, level=0.95)
        elif name == "trade":
            # Big and small nets take turns, as in the offline show.  This is
            # the gesture that needs a *correct* bar line rather than merely a
            # consistent one -- counted from the wrong beat it trades offbeat.
            lead = self.big if int(self._bars_elapsed(t)) % 2 == 0 else self.small
            fx.bars(canvas, palette, phrase * 4.0, count=3, angle=0.15,
                    width=0.3, level=0.95, targets=lead)
        elif name == "slow_wheel":
            fx.pinwheel(canvas, palette, t * 0.12, arms=3, level=0.75)
        elif name == "wheel_up":
            fx.pinwheel(canvas, palette, t * (1.0 + 5.0 * tension) * 0.35,
                        arms=3, level=0.8, targets=self.big)
        elif name == "fast_wheel":
            fx.pinwheel(canvas, palette, -t * 0.8, arms=5, level=0.7)
        elif name == "rings":
            fx.radial(canvas, palette, (t * 2.0) % 1.0, width=0.3, level=0.9)
        elif name == "flare":
            fx.radial(canvas, palette, kick, width=0.45, level=0.95)
            fx.net_sparkle(canvas, pal.WHITE, frame, density=0.02, level=0.9,
                           seed=self.seed)
        elif name == "strobe_small":
            rate = 2 + int(tension * 6)
            flash = _kick((t / max(1e-6, 60.0 / self.bpm) * rate) % 1.0, sharp=3.0)
            fx.pinwheel(canvas, palette, t * 0.4, arms=3, level=0.6,
                        targets=self.big)
            fx.wash(canvas, pal.WHITE, 0.55 * flash * max(tension, 0.3),
                    targets=self.small)

    def _quiet(self, frame, t, phrase, palette, beat, bar, kick, features) -> None:
        self._gesture(self.gesture, frame, t, phrase, palette, beat, kick)
        fx.par(self.canvas, palette.color(0), 0.25 + 0.15 * phrase)

    def _cruising(self, frame, t, phrase, palette, beat, bar, kick, features) -> None:
        fx.wash(self.canvas, palette.dimmed(0.35), 1.0, gradient=0.8)
        self._gesture(self.gesture, frame, t, phrase, palette, beat, kick)
        fx.par(self.canvas, palette.color(1), 0.35 + 0.45 * kick)

    def _building(self, frame, t, phrase, palette, beat, bar, kick, features) -> None:
        # Tension tracks how far into the build we are, but it is *abortable*:
        # if the sweep stops without a drop, this simply relaxes.  Never
        # pre-fire the resolution.
        tension = min(1.0, self.machine.report.since_s / 8.0)
        fx.wash(self.canvas, palette.dimmed(0.25), 1.0, gradient=1.0)
        self._gesture(self.gesture, frame, t, phrase, palette, beat, kick,
                      tension=tension)
        flash = _kick((t / max(1e-6, 60.0 / self.bpm)
                       * (2 + int(tension * 6))) % 1.0, sharp=3.0)
        fx.par(self.canvas, palette.color(0), 0.4 + 0.6 * tension * flash,
               white=0.3 * tension)

    def _hot(self, frame, t, phrase, palette, beat, bar, kick, features) -> None:
        fx.wash(self.canvas, palette.dimmed(0.3), kick * 0.8)
        self._gesture(self.gesture, frame, t, phrase, palette, beat, kick,
                      tension=1.0)
        fx.par(self.canvas, pal.WHITE.color(0), kick, white=kick)


def _kick(phase: float, sharp: float = 2.0) -> float:
    return float(max(0.0, 1.0 - phase) ** sharp)
