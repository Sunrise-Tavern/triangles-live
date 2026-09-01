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

Two later additions, both about not repeating ourselves:

* **variety per visit**: each time a state is entered it picks a colour
  scheme, a corridor depth rotation and a saturation of its own, so a second
  drop is not the first one again in a different hue;
* **transitions**: when the material changes -- a new phrase, a new state --
  the outgoing look is painted alongside the incoming one for a moment and
  the two are mixed by a style chosen per change: a cut, a crossfade, a wipe
  down the tunnel, or a dip.  A drop always cuts; nothing else has to.

Both are driven by the same seeded, deterministic walks as everything else,
so "random" here still means "the same show for the same audio".
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
    #: Light the corridor keeps under the pattern, 0..1.  A sparse pattern
    #: leaves most arches dark most of the time, which is right for a drop and
    #: wrong for a quiet passage: measured on a real session, a track's 45
    #: second intro rendered the arches at 2.8/255 against 2.9 for dead air
    #: between tracks -- a track everyone in the room could hear building
    #: looked exactly like no music at all.
    floor: float = 0.0


#: Net gestures each state may draw, rotated one per phrase by the same
#: deterministic walk the corridor uses.  The offline show does this too --
#: "butterfly/spirals/fan, rotating every 4 bars" for a drop -- and for the
#: same reason the corridor does: one gesture held for a whole section reads
#: as one idea, however well it tracks the music.
#:
#: Each state's list is ordered loosely from calm to busy, and every gesture in
#: it has to make sense at that energy: a drop can strobe, a breakdown cannot.
#:
#: The ``big_*`` gestures paint the four "Big Triangle" nets as one surface
#: (see :meth:`Canvas._big_geometry`): one wheel turning about the big
#: triangle's centre, one ring leaving it, bands sweeping across all four.
#: The small nets echo the same effect at their own scale, dimmer.
#:
#: The ``all_*`` gestures paint every net as one surface -- the whole array,
#: three singles and the big triangle side by side -- so a thing can cross
#: the room: a band sweeping end to end and back, a ball bouncing along it,
#: a wave rolling through, a burst from the centre, the nets lit one after
#: another left to right.
NET_GESTURES: dict[str, tuple[str, ...]] = {
    QUIET:    ("plasma", "twinkle", "breathe", "orbit", "ripples_slow",
               "big_plasma", "big_orbit", "all_plasma", "all_wave_slow",
               "all_drift"),
    CRUISING: ("bars", "trade", "slow_wheel", "rings", "spiral", "ripples",
               "rain", "orbit", "big_bars", "big_wheel", "big_spiral",
               "big_ripples", "all_sweep", "all_ball", "all_wave", "all_fall",
               "all_diagonal", "all_scan"),
    BUILDING: ("wheel_up", "strobe_small", "bars_fast", "rain_fast",
               "checker", "spiral_fast", "big_wheel_up", "big_rain",
               "all_rise", "all_scan_up", "all_squeeze"),
    HOT:      ("rings", "fast_wheel", "bars_fast", "flare", "checker",
               "halves", "apex_flash", "spiral_fast", "ripples_fast",
               "big_rings", "big_bars_fast", "big_wheel_fast", "big_apex",
               "all_burst", "all_sweep_fast", "all_ball_fast", "all_scan_fast",
               "all_slam", "all_diagonal_fast"),
}

#: Colour schemes each state may draw, one chosen per visit.  Every scheme in
#: a list has to suit the energy: a drop can take four corners of the wheel,
#: a breakdown wants one hue in three depths.
SCHEME_OPTIONS: dict[str, tuple[str, ...]] = {
    SILENT:   ("analogous",),
    QUIET:    ("analogous", "mono", "sweep", "neighbours"),
    CRUISING: ("split", "analogous", "accent", "sweep", "neighbours"),
    BUILDING: ("complementary", "accent", "split"),
    HOT:      ("triadic", "tetradic", "complementary", "split", "accent"),
}

#: How the outgoing look gives way to the incoming one, and over how many
#: beats.  Weighted by repetition: a cut is still the commonest move, because
#: a show that always dissolves reads as soft.
TRANSITIONS: tuple[tuple[str, float], ...] = (
    ("cut", 0.0), ("cut", 0.0),
    ("fade", 1.0), ("fade", 2.0), ("fade", 4.0),
    ("wipe_back", 1.0), ("wipe_back", 2.0), ("wipe_front", 2.0),
    ("dip", 2.0),
)


#: Showpieces: composed sequences that own the whole rig for their stretch --
#: tunnel and triangles telling one story -- unlike a pattern or a gesture,
#: which each paint their half.  Each name maps to the states it suits, and to
#: a ``_piece_<name>`` method.  They join the rotation at PIECE_SHARE, and the
#: panel's "Piece" knob forces one for a look.
PIECES: dict[str, tuple[str, ...]] = {
    "charge": (CRUISING, BUILDING, HOT),
    "dna": (CRUISING, HOT),
}

#: Share of material stretches the rotation gives a showpiece, in the states
#: it suits, after the clip roll has passed.
PIECE_SHARE = 0.15


TREATMENTS: dict[str, Treatment] = {
    SILENT:   Treatment(SILENT,   "analogous",     0.45, 0.40, "comet",    16.0, 16.0, True, 0.00),
    QUIET:    Treatment(QUIET,    "analogous",     0.60, 0.60, "sparkle",   4.0,  8.0, True, 0.30),
    CRUISING: Treatment(CRUISING, "split",         0.85, 0.80, "comet",     2.0,  4.0, False, 0.12),
    BUILDING: Treatment(BUILDING, "complementary", 0.95, 0.90, "pairs",     2.0,  4.0, False, 0.06),
    HOT:      Treatment(HOT,      "triadic",       1.00, 1.00, "alternate", 1.0,  4.0, False, 0.00),
}


class Arranger:
    """Drop-in replacement for :class:`live.script.Script`, driven by audio."""

    def __init__(self, canvas: Canvas, listener: Listener, *,
                 settings: Settings | None = None,
                 state: StateMachine | None = None,
                 base_hue: float = 190.0, seed: int = 7,
                 clips=None) -> None:
        self.canvas = canvas
        self.listener = listener
        self.settings = settings
        self.machine = state or StateMachine()
        #: A live.clips.Clips library, or None.  With one, some phrases play
        #: a canned xLights loop -- beat-locked, enveloped by the music --
        #: instead of a painted look; without one (every offline tool, the
        #: selftests) the show is exactly as before, and deterministic.
        self.clips = clips
        self.base_hue = base_hue
        self._seed = seed
        self.big, self.small = canvas.net_pair()
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
        #: The look being painted, and the one it is replacing.
        self._look: tuple | None = None
        self._outgoing: tuple | None = None
        self._transition: tuple[str, float, float] | None = None
        self._changes = 0
        self._scratch: Canvas | None = None
        #: (when the current phrase started, its index).
        self._phrase: tuple[float | None, int] = (None, 0)
        #: Per state, the (kind, visit, phrase) a pattern was chosen for and
        #: the pattern -- so the choice holds for the phrase.
        self._pattern_held: dict[str, tuple[tuple, str]] = {}

    # -- knobs and derived values ------------------------------------------ #

    @property
    def clock(self):
        return self.listener.clock

    @property
    def seed(self) -> int:
        return int(self.settings.seed) if self.settings else self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seed = int(value)

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

        Counted *here*, not read off the clock's beat index: that index moves
        whenever the clock relocks or changes its mind about the tempo, and
        measured on a real session it moved often enough during a drop (the
        tracker at 0.03 confidence) that the pattern changed every half second
        against a nominal four bars.  So a phrase is a span of wall time,
        ended at the first bar line after ``pattern_bars`` have passed -- or
        outright once half again as long has gone by with no bar line found.
        """
        rate = self.settings.corridor_rate if self.settings else 1.0
        span = (max(0.5, treat.pattern_bars / max(rate, 1e-3))
                * self.clock.bar_length * 60.0 / max(self.bpm, 1e-3))
        started, count = self._phrase
        if started is None:
            self._phrase = (t, 0)
            return 0
        elapsed = t - started
        if elapsed < 0.0:
            # Time moved backwards (a replay restarted): start over.
            self._phrase = (t, count + 1)
            return count + 1
        if elapsed >= span * 1.5 or (
                elapsed >= span * 0.85 and self.clock.bar_phase(t) < 0.12):
            self._phrase = (t, count + 1)
            return count + 1
        return count

    def _beats(self, t: float) -> float:
        """Beats elapsed, continuous -- the clock for every gesture.

        Gestures used to spin on wall time (``t * 0.12``), which has no
        relation to the music: measured at 128.8 BPM the cruising wheel took
        17.9 beats a turn and the orbit 26.8, and none of it moved with the
        tempo.  Everything periodic now counts in beats off the clock.
        """
        return float(self.clock.beat_index_at(t) + self.clock.phase(t))

    def _bars_elapsed(self, t: float) -> float:
        index = self.clock.beat_index_at(t)
        return (index - self.clock.downbeat) // self.clock.bar_length

    #: Degrees the hue moves per phrase inside a section.  This used to be a
    #: 9-degree nudge cycling over four phrases, barely visible; now the
    #: phrase is the *colour* clock -- the pattern and gesture hold for
    #: several phrases (``pattern_hold``) and what changes between them is
    #: the colour, so the step has to read as a change.  Monotonic, so a
    #: long section keeps travelling rather than snapping back.
    PHRASE_HUE_STEP = 24.0

    def material_index(self, phrase: int) -> int:
        """Which stretch of held material ``phrase`` falls in."""
        hold = self.settings.pattern_hold if self.settings else 4.0
        return int(phrase // max(1, int(round(hold))))

    def _visit_roll(self, key: str, treat: Treatment) -> float:
        """A stable 0..1 for this visit to this state, per ``key``."""
        visit = self.visits.get(treat.kind, 1)
        seed = f"{self.seed}:{key}:{treat.kind}:{visit}".encode()
        return (zlib.crc32(seed) % 1000) / 999.0

    def scheme_for(self, treat: Treatment) -> str:
        """Which colour scheme this visit to the state draws.

        The treatment's own scheme is the first option and the walk starts
        from it, so the first visit looks as it always did and returns vary.
        """
        held = self.settings.scheme if self.settings else "auto"
        if held != "auto" and held in pal.SCHEMES:
            return held
        options = SCHEME_OPTIONS.get(treat.kind, (treat.scheme,))
        visit = self.visits.get(treat.kind, 1)
        if visit <= 1:
            return options[0]
        return options[self._walk(f"scheme:{treat.kind}", visit - 2, len(options))]

    def saturation_for(self, treat: Treatment) -> float:
        """Full colour for a drop; elsewhere each visit sits a little off it,
        so two verses do not read as the same wash."""
        if treat.kind == HOT:
            return 1.0
        return 1.0 - 0.18 * self._visit_roll("sat", treat)

    def far_rotation_for(self, treat: Treatment) -> float:
        """Degrees the corridor's far end is turned from its mouth.  Was a
        fixed 70; now 45-110 per visit, so the tunnel's depth reads
        differently each time round."""
        return 45.0 + 65.0 * self._visit_roll("far", treat)

    def palette_for(self, treat: Treatment, phrase: int = 0) -> pal.Palette:
        offset = self.settings.hue_offset if self.settings else 0.0
        lock = self.settings.hue_lock if self.settings else False
        drift = phrase * self.PHRASE_HUE_STEP
        hue = self.base_hue + offset + (0.0 if lock
                                        else self.journey * pal.GOLDEN_ANGLE + drift)
        scheme = self.scheme_for(treat)
        sat = round(self.saturation_for(treat), 3)
        key = (round(hue, 2), scheme, sat, treat.value, treat.kind)
        cached = self._palettes.get(key)
        if cached is None:
            if len(self._palettes) > 256:
                self._palettes.clear()
            cached = pal.generate(hue, scheme, value=treat.value, sat=sat,
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
        return options[self._walk(f"nets:{treat.kind}:{visit}",
                                  self.material_index(phrase), len(options))]

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
        # Decided once per phrase.  The density target below moves with the
        # clock's confidence, and re-deciding every frame let a wobbling
        # tracker (0.03 confidence through a drop, measured) reshuffle the
        # candidates under the walk's index: the pattern changed every half
        # second against a nominal four bars.
        visit = self.visits.get(treat.kind, 1)
        phrase = self.material_index(phrase)
        key = (treat.kind, visit, phrase)
        held = self._pattern_held.get(treat.kind)
        if held is not None and held[0] == key:
            return held[1]
        articulation = self.settings.articulation if self.settings else 0.5
        # A grid we do not trust should not drive busy, tightly-placed
        # patterns; fall back toward the sparse end instead.
        trust = self.clock.confidence
        target = ((fx.DENSITY[treat.pattern] + (articulation - 0.5))
                  * (0.5 + 0.5 * trust) + self.escalation(treat))
        candidates = fx.vocabulary(min(max(target, 0.0), 1.0), quiet=treat.quiet)
        # The walk is keyed on the visit as well, so a return does not replay
        # the same sequence of patterns in the same order.
        name = candidates[self._walk(f"corridor:{treat.kind}:{visit}", phrase,
                                     len(candidates))]
        self._pattern_held[treat.kind] = (key, name)
        return name

    def transition_for(self, change: int, *, state_change: bool) -> tuple[str, float]:
        """(style, beats) for the ``change``-th change of look.

        A change of *state* always cuts: the music changed, and a four-beat
        dissolve out of quiet read as the lights lagging a track that had
        plainly started (reported).  A drop is only the loudest case of
        that.  Within a state -- a new phrase, a new colour -- the change is
        ours rather than the music's, and it may roll.
        """
        if state_change:
            return ("cut", 0.0)
        share = self.settings.transitions if self.settings else 0.75
        seed = f"{self.seed}:transition:{change}".encode()
        roll = zlib.crc32(seed)
        if (roll % 1000) / 1000.0 >= share:
            return ("cut", 0.0)
        return TRANSITIONS[(roll // 1000) % len(TRANSITIONS)]

    # -- render ------------------------------------------------------------ #

    def render(self, frame: int, t: float) -> None:
        canvas = self.canvas
        canvas.clear()

        if self.machine.state != self._last_state:
            self.journey += 1
            self.visits[self.machine.state] = self.visits.get(
                self.machine.state, 0) + 1
            self._last_state = self.machine.state
            # A new state is a new phrase, whatever the old one had left.
            self._phrase = (t, self._phrase[1] + 1)

        _, treat, phrase = self.locate(t)
        if treat.kind == SILENT:
            # Nothing is playing, so nothing here is driven by the beat clock:
            # it is free-running on no evidence and following it would make the
            # rig twitch at an imaginary tempo.  Everything below is a function
            # of wall time, slow enough that you have to watch to see it move.
            self._look = (SILENT, "", "", None, 0.0, treat, 0.0)
            self._outgoing = None
            self._silent(t)
            return

        index = self.phrase_index(t, treat)
        piece_name = self.piece_for(treat, index)
        clip_name = None if piece_name else self.clip_for(treat, index)
        palette = self.palette_for(treat, index)
        if piece_name is not None:
            self.gesture = "piece"
            anchor = self._beats(self._phrase[0] if self._phrase[0] is not None
                                 else t)
            look = (treat.kind, f"piece:{piece_name}", "piece", palette,
                    self.far_rotation_for(treat), treat, anchor)
        elif clip_name is not None:
            # This stretch of material is a canned loop.  Anchored in beats
            # at the phrase start, so it begins at its first frame and its
            # authored rhythm rides the clock.
            self.gesture = "clip"
            anchor = self._beats(self._phrase[0] if self._phrase[0] is not None
                                 else t)
            look = (treat.kind, f"clip:{clip_name}", "clip", palette,
                    self.far_rotation_for(treat), treat, anchor)
        else:
            self.gesture = self.gesture_for(treat, index)
            name = self.pattern_for(treat, index)
            look = (treat.kind, name, self.gesture, palette,
                    self.far_rotation_for(treat), treat, 0.0)

        if self._look is not None and _identity(look) != _identity(self._look):
            # The material changed.  Decide how the old gives way to the new;
            # a change arriving mid-transition simply replaces the outgoing
            # look, which is what a cut would have shown anyway.
            self._changes += 1
            style, beats = self.transition_for(
                self._changes, state_change=treat.kind != self._look[0])
            if style == "cut":
                self._outgoing, self._transition = None, None
            else:
                seconds = beats * 60.0 / max(self.bpm, 1e-3)
                self._outgoing = self._look
                self._transition = (style, t, max(seconds, 1e-3))
        self._look = look

        self._paint(canvas, look, frame, t, phrase)

        if self._outgoing is not None and self._transition is not None:
            style, started, seconds = self._transition
            mix = (t - started) / seconds
            if mix >= 1.0:
                self._outgoing, self._transition = None, None
                return
            if self._scratch is None:
                self._scratch = Canvas(canvas.layout)
            self._scratch.clear()
            self._paint(self._scratch, self._outgoing, frame, t, phrase)
            _mix_canvases(canvas, self._scratch, style, max(mix, 0.0))

    def _paint(self, canvas: Canvas, look: tuple, frame: int, t: float,
               phrase: float) -> None:
        """One look, fully painted: corridor, nets, par."""
        kind, name, gesture, palette, far_degrees, treat, anchor = look
        beat = self.clock.phase(t)
        bar = self.clock.bar_phase(t)
        kick = _kick(beat)
        features = self.features

        if name.startswith("clip:"):
            self._paint_clip(canvas, name[5:], kind, t, anchor, beat, kick)
            return
        if name.startswith("piece:"):
            getattr(self, f"_piece_{name[6:]}")(canvas, kind, t, anchor,
                                                palette, beat, kick, features)
            return

        far = palette.rotated(far_degrees)
        levels = fx.PATTERNS[name](
            len(canvas.arch_names), phrase,
            **({"seed": self.seed} if name in fx.SEEDED else {}))
        if treat.floor > 0.0:
            # Blended, not clamped: the pattern still reads on top of the
            # floor rather than being flattened by it.
            levels = treat.floor + (1.0 - treat.floor) * levels
        fx.corridor(canvas, levels, palette, far,
                    brightness=min(1.0, treat.brightness + self.escalation(treat)),
                    height=0.35)

        getattr(self, f"_{kind}")(canvas, gesture, frame, t, phrase, palette,
                                  beat, bar, kick, features)

    def clip_for(self, treat: Treatment, phrase: int) -> str | None:
        """The clip this stretch of material plays, or None for a painted look.

        Decided once per pattern_hold span by the same seeded machinery as
        everything else: a roll against ``clip_share`` says whether this
        stretch is a clip at all, and the no-repeat walk picks which, from
        the clips whose *measured* energy suits the state.  A clip that is
        not decoded yet is skipped for this stretch (the loader is already
        on it) rather than waited for.
        """
        if self.clips is None or treat.kind == SILENT:
            return None
        share = self.settings.clip_share if self.settings else 0.3
        if share <= 0.0:
            return None
        options = self.clips.vocabulary(treat.kind)
        if not options:
            return None
        visit = self.visits.get(treat.kind, 1)
        material = self.material_index(phrase)
        roll = zlib.crc32(f"{self.seed}:cliproll:{treat.kind}:{visit}:"
                          f"{material}".encode())
        if (roll % 1000) / 1000.0 >= share:
            return None
        name = options[self._walk(f"clips:{treat.kind}:{visit}", material,
                                  len(options))]
        return name if self.clips.get(name) is not None else None

    def piece_for(self, treat: Treatment, phrase: int) -> str | None:
        """The showpiece this stretch plays, or None.

        The "Piece" knob wins: a named piece plays always, "off" removes
        them from the rotation, "auto" rolls -- the same seeded machinery as
        clips, so the same audio and seed still render the same show.
        """
        held = self.settings.piece if self.settings else "auto"
        if held in PIECES:
            return held
        if held == "off" or treat.kind == SILENT:
            return None
        options = [n for n, kinds in PIECES.items() if treat.kind in kinds]
        if not options:
            return None
        visit = self.visits.get(treat.kind, 1)
        material = self.material_index(phrase)
        roll = zlib.crc32(f"{self.seed}:pieceroll:{treat.kind}:{visit}:"
                          f"{material}".encode())
        if (roll % 1000) / 1000.0 >= PIECE_SHARE:
            return None
        return options[self._walk(f"pieces:{treat.kind}:{visit}", material,
                                  len(options))]

    #: The charge cycle: bars down the tunnel, one bar of blow, bars back.
    CHARGE_RUN_BARS = 4.0

    def _piece_charge(self, canvas: Canvas, kind: str, t: float, anchor: float,
                      palette, beat: float, kick: float, features) -> None:
        """A comet charges from the back of the tunnel; reaching the mouth it
        blows through every triangle at once; then it runs home.

        The run takes four bars each way and the hue steps 45 degrees on
        every bar from a base that is randomized (seeded) per cycle, so no
        two charges wear the same colours.  The blow lands exactly on the
        bar line the run arrives on -- the clock predicts it, nothing reacts
        late -- and its flash core rides the kick.
        """
        bars = (self._beats(t) - anchor) / max(1, self.clock.bar_length)
        run = self.CHARGE_RUN_BARS
        cycle_bars = 2.0 * run
        cycle, u = int(bars // cycle_bars), bars % cycle_bars
        base = (zlib.crc32(f"{self.seed}:charge:{cycle}".encode()) % 3600) / 10.0
        hue = base + 45.0 * int(u)
        paint = pal.generate(hue, "complementary", value=0.95).floored()
        far = paint.rotated(60.0)

        position = 1.0 - u / run if u < run else (u - run) / run
        glow = np.exp(-((canvas.depth - position) * 6.0) ** 2)
        levels = (0.06 + 0.94 * glow * (0.75 + 0.25 * kick)).astype(np.float32)
        fx.corridor(canvas, levels, paint, far, brightness=1.0, height=0.35)

        blow = u - run          # bars since the head hit the mouth
        if 0.0 <= blow < 1.0:
            G = canvas.all_geo
            fx.radial(canvas, paint, blow, width=0.30, level=1.0 - 0.6 * blow,
                      geo=G)
            fx.blob(canvas, pal.WHITE, 0.5, 0.5, radius=0.30,
                    level=(1.0 - blow) ** 2 * (0.5 + 0.5 * kick), geo=G)
        else:
            fx.plasma(canvas, paint.dimmed(0.5), t, scale=2.0, speed=0.4,
                      level=0.30)

    def _piece_dna(self, canvas: Canvas, kind: str, t: float, anchor: float,
                   palette, beat: float, kick: float, features) -> None:
        """A double helix screws through the tunnel toward the mouth while
        the triangles take the bass on the chin.

        The crush is driven by the *measured* kick -- bass flux over its own
        average, the analyzer's kick detector -- scaled into 0..1, and shaped
        by the predicted beat so it lands with the room: on a hit, two bands
        slam from the ends of the array into its centre and a flash blooms
        out as it decays.  No bass, no crush; a heavier hit crushes harder.
        """
        beats = self._beats(t) - anchor
        fx.helix(canvas, palette, beats / 4.0, turns=2.0, level=0.95)

        strength = min(1.0, float(getattr(features, "kick", 0.0)) / 4.0)             if features is not None else 0.5
        env = _kick(beat, sharp=2.5) * strength
        G = canvas.all_geo
        fx.plasma(canvas, palette.dimmed(0.35), t, scale=2.2, speed=0.5,
                  level=0.30)
        if env > 0.02:
            at = 0.5 * env
            fx.sweep(canvas, palette, at, width=0.10, level=env, geo=G)
            fx.sweep(canvas, palette, 1.0 - at, width=0.10, level=env, geo=G)
            fx.blob(canvas, pal.WHITE, 0.5, 0.5,
                    radius=0.12 + 0.40 * (1.0 - env), level=env * 0.9, geo=G)

    def _paint_clip(self, canvas: Canvas, name: str, kind: str, t: float,
                    anchor: float, beat: float, kick: float) -> None:
        """A canned loop as this phrase's material, ridden by the music.

        The playhead advances in *beats* from the phrase it started in, so
        the authored motion speeds up and slows down with the track; the
        level envelope is the state's -- the kick pulses it while cruising
        and hot, a build's flashes quicken with its tension, quiet breathes.
        The clip's own colours are kept: that is what it is for.
        """
        clip = self.clips.get(name) if self.clips is not None else None
        if clip is None:                # evicted or failed mid-phrase
            fx.wash(canvas, self.palette_for(TREATMENTS[kind]), 0.5)
            return
        canvas.from_channels(clip.frame_at_beats(self._beats(t) - anchor))
        if kind == QUIET:
            envelope = 0.65 + 0.15 * float(np.sin(2 * np.pi * self.clock.bar_phase(t)))
        elif kind == BUILDING:
            tension = min(1.0, self.machine.report.since_s / 8.0)
            flash = _kick((self._beats(t) * (2 + int(tension * 6))) % 1.0,
                          sharp=3.0)
            envelope = 0.55 + 0.15 * tension + 0.30 * tension * flash
        elif kind == HOT:
            envelope = 0.70 + 0.30 * kick
        else:
            envelope = 0.75 + 0.25 * kick
        canvas.nets *= envelope
        canvas.arches *= envelope

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

    def _rest(self, canvas: Canvas, targets: slice, t: float, palette) -> None:
        """Give the group that is not leading something to do.

        Three gestures lead with the big or the small triangles and leave the
        other group on the bed wash alone.  The offline show does the same,
        and there it reads as call-and-response; live, with the corridor
        moving underneath, the resting group reads as frozen -- a bar at a
        time in ``trade``, a whole phrase in ``wheel_up``.  So by default it
        breathes: a slow, dim plasma, clearly subordinate to the lead.
        ``rest_level`` 0 restores the hold.
        """
        level = self.settings.rest_level if self.settings else 0.4
        if level > 0.0:
            fx.plasma(canvas, palette, t, scale=2.0, speed=2.2,
                      level=level, targets=targets)

    def _gesture(self, canvas: Canvas, name: str, frame: int, t: float,
                 phrase: float, palette, beat: float, kick: float,
                 tension: float = 0.0) -> None:
        """Paint one net gesture.  The bed and the par stay with the state."""
        beat_s = max(1e-6, 60.0 / self.bpm)
        beats = self._beats(t)
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
            even = int(self._bars_elapsed(t)) % 2 == 0
            lead, rest = (self.big, self.small) if even else (self.small, self.big)
            self._rest(canvas, rest, t, palette)
            fx.bars(canvas, palette, phrase * 4.0, count=3, angle=0.15,
                    width=0.3, level=0.95, targets=lead)
        elif name == "slow_wheel":
            # One turn per two bars.
            fx.pinwheel(canvas, palette, beats / 8.0, arms=3, level=0.75)
        elif name == "wheel_up":
            # One turn per bar, tightening to one per beat as the build rises.
            self._rest(canvas, self.small, t, palette)
            fx.pinwheel(canvas, palette, beats * (0.25 + 0.75 * tension),
                        arms=3, level=0.8, targets=self.big)
        elif name == "fast_wheel":
            # One turn per two beats, the other way.
            fx.pinwheel(canvas, palette, -beats / 2.0, arms=5, level=0.7)
        elif name == "rings":
            # One ring out per beat, launched on the beat.
            fx.radial(canvas, palette, beat, width=0.3, level=0.9)
        elif name == "flare":
            fx.radial(canvas, palette, kick, width=0.45, level=0.95)
            fx.net_sparkle(canvas, pal.WHITE, frame, density=0.02, level=0.9,
                           seed=self.seed)
        elif name == "strobe_small":
            rate = 2 + int(tension * 6)
            flash = _kick((t / beat_s * rate) % 1.0, sharp=3.0)
            fx.pinwheel(canvas, palette, beats / 4.0, arms=3, level=0.6,
                        targets=self.big)
            # The small nets only flash here, faintly until the build has
            # some tension; between flashes they would otherwise hold.
            self._rest(canvas, self.small, t, palette)
            fx.wash(canvas, pal.WHITE, 0.55 * flash * max(tension, 0.3),
                    targets=self.small)
        # -- the second generation ------------------------------------------ #
        elif name == "orbit":
            # One orbit per two bars; the small nets go the other way at a
            # different rate so the two groups are not in step.
            fx.orbit(canvas, palette, beats / 8.0, level=0.85, targets=self.big)
            fx.orbit(canvas, palette, -beats / 6.0, width=0.22, level=0.7,
                     targets=self.small)
        elif name == "ripples_slow":
            # One ring per bar.
            fx.ripples(canvas, palette, beats / 4.0, rings=2.0, level=0.7)
        elif name == "ripples":
            # One ring per two beats.
            fx.ripples(canvas, palette, beats / 2.0, rings=3.0, level=0.85)
        elif name == "ripples_fast":
            # One ring per beat, so the rings fire with the room.
            fx.ripples(canvas, palette, beat, rings=2.0, level=0.95)
        elif name == "spiral":
            # One turn per two bars.
            fx.spiral(canvas, palette, beats / 8.0, arms=2, twist=1.5, level=0.8)
        elif name == "spiral_fast":
            # One turn per two beats, one per beat at full tension.
            fx.spiral(canvas, palette, -beats * (0.5 + 0.5 * tension), arms=3,
                      twist=2.5, level=0.85)
        elif name == "rain":
            # One band per two beats, falling apex to base; the nets are
            # triangles, so this reads as something pouring into the wide end.
            fx.bars(canvas, palette, -beats / 2.0, count=3, angle=1.0,
                    width=0.3, level=0.85)
        elif name == "rain_fast":
            # One band per beat.
            fx.bars(canvas, palette, -beats, count=4, angle=1.0,
                    width=0.22, level=0.9)
        elif name == "checker":
            # Flips on every beat; on a build, twice as often as it tightens.
            step = int(beats * (1 + int(tension > 0.6)))
            fx.checker(canvas, palette, step, cells=4,
                       level=0.85 * _kick(beat, sharp=1.2) + 0.15)
        elif name == "halves":
            # Left and right trade on the beat, with the kick's decay.
            left = int(beats) % 2 == 0
            fx.halves(canvas, palette, left, 0.35 + 0.65 * kick, level=0.95)
        elif name == "apex_flash":
            fx.pinwheel(canvas, palette, beats / 4.0, arms=3, level=0.45)
            fx.apex(canvas, pal.WHITE, kick, level=0.9)
        # -- the big triangle as one surface -------------------------------- #
        elif name.startswith("big_"):
            self._big(canvas, name[4:], frame, t, phrase, palette, beat, kick,
                      tension, beats)
        # -- every net as one surface --------------------------------------- #
        elif name.startswith("all_"):
            self._all(canvas, name[4:], frame, t, phrase, palette, beat, kick,
                      tension, beats)

    def _big(self, canvas: Canvas, name: str, frame: int, t: float,
             phrase: float, palette, beat: float, kick: float, tension: float,
             beats: float) -> None:
        """One gesture over the big triangle, echoed small on the small nets.

        Every call paints the big nets through ``canvas.big_geo`` -- the
        four of them as one triangle -- and the small nets with the same
        effect in their own frame at a lower level, so the two groups read
        as the same idea at two scales rather than one group resting.
        """
        big, small = self.big, self.small
        G = canvas.big_geo
        echo = 0.45
        if name == "plasma":
            fx.plasma(canvas, palette, t, scale=1.6, speed=0.3, level=0.75,
                      targets=big, geo=G)
            fx.plasma(canvas, palette, t, scale=2.5, speed=0.35, level=echo,
                      targets=small)
        elif name == "orbit":
            fx.orbit(canvas, palette, beats / 16.0, radius=0.6, width=0.12,
                     level=0.9, targets=big, geo=G)
            fx.orbit(canvas, palette, -beats / 8.0, width=0.22, level=echo,
                     targets=small)
        elif name == "bars":
            # Three bands across the whole big triangle, one bar per pass.
            fx.bars(canvas, palette, phrase * 2.0, count=2, angle=0.2,
                    width=0.3, level=0.95, targets=big, geo=G)
            fx.bars(canvas, palette, phrase * 4.0, count=3, angle=0.15,
                    width=0.3, level=echo, targets=small)
        elif name == "bars_fast":
            fx.bars(canvas, palette, beats / 2.0, count=3, angle=0.35,
                    width=0.22, level=0.95, targets=big, geo=G)
            fx.bars(canvas, palette, beats, count=4, angle=0.35, width=0.22,
                    level=echo, targets=small)
        elif name == "wheel":
            # One turn per four bars about the big triangle's centre.
            fx.pinwheel(canvas, palette, beats / 16.0, arms=3, level=0.85,
                        targets=big, geo=G)
            fx.pinwheel(canvas, palette, -beats / 8.0, arms=3, level=echo,
                        targets=small)
        elif name == "wheel_up":
            # One turn per two bars, tightening to one per beat.
            fx.pinwheel(canvas, palette, beats * (0.125 + 0.875 * tension),
                        arms=3, level=0.85, targets=big, geo=G)
            self._rest(canvas, small, t, palette)
        elif name == "wheel_fast":
            fx.pinwheel(canvas, palette, -beats / 4.0, arms=6, level=0.8,
                        targets=big, geo=G)
            fx.pinwheel(canvas, palette, beats / 2.0, arms=5, level=echo,
                        targets=small)
        elif name == "spiral":
            fx.spiral(canvas, palette, beats / 16.0, arms=2, twist=2.0,
                      level=0.85, targets=big, geo=G)
            fx.spiral(canvas, palette, -beats / 8.0, arms=2, twist=1.5,
                      level=echo, targets=small)
        elif name == "ripples":
            # One ring per bar out of the big centre.
            fx.ripples(canvas, palette, beats / 4.0, rings=2.5, level=0.85,
                       targets=big, geo=G)
            fx.ripples(canvas, palette, beats / 2.0, rings=3.0, level=echo,
                       targets=small)
        elif name == "rings":
            # One ring per beat, launched on the beat, across the big triangle.
            fx.radial(canvas, palette, beat, width=0.25, level=0.95,
                      targets=big, geo=G)
            fx.radial(canvas, palette, beat, width=0.3, level=echo,
                      targets=small)
        elif name == "rain":
            # Bands falling the height of the big triangle, faster as it builds.
            fx.bars(canvas, palette, -beats * (0.25 + 0.75 * tension), count=3,
                    angle=1.0, width=0.25, level=0.9, targets=big, geo=G)
            self._rest(canvas, small, t, palette)
        elif name == "apex":
            # The kick flashes from the big apex and dies toward the base.
            fx.pinwheel(canvas, palette, beats / 8.0, arms=3, level=0.4,
                        targets=big, geo=G)
            fx.apex(canvas, pal.WHITE, kick, level=0.95, targets=big, geo=G)
            fx.apex(canvas, pal.WHITE, kick, level=echo, targets=small)
        else:
            raise ValueError(f"unknown big gesture {name!r}")

    def _all(self, canvas: Canvas, name: str, frame: int, t: float,
             phrase: float, palette, beat: float, kick: float, tension: float,
             beats: float) -> None:
        """One gesture across every net as one surface.

        Positions are in ``canvas.all_geo``: x 0 at the left end of the
        array, 1 at the right; y 0 at the top.  Everything periodic counts
        in beats; a "bounce" is a triangle wave of the beat count, so the
        thing turns round exactly on a bar line.
        """
        G = canvas.all_geo
        bed = 0.10          # so the nets the thing is not on are not off

        def bounce(period_beats: float) -> float:
            u = (beats / period_beats) % 1.0
            return 2.0 * u if u < 0.5 else 2.0 - 2.0 * u

        if name == "plasma":
            fx.plasma(canvas, palette, t, scale=1.2, speed=0.25, level=0.75, geo=G)
        elif name == "wave_slow":
            # One long wave rolling through the whole array, one crest per
            # array, a bar and a half per crossing.
            fx.bars(canvas, palette, beats / 6.0, count=1, width=0.5,
                    level=0.8, geo=G)
        elif name == "wave":
            fx.bars(canvas, palette, beats / 4.0, count=2, width=0.4,
                    level=0.85, geo=G)
        elif name == "drift":
            # A soft spot wandering slowly across the room.
            fx.wash(canvas, palette.dimmed(bed), 1.0)
            fx.blob(canvas, palette, bounce(32.0), 0.5 + 0.3 * np.sin(beats / 5.0),
                    radius=0.35, level=0.8, geo=G)
        elif name == "sweep":
            # A band crossing end to end over two bars, then back.
            fx.wash(canvas, palette.dimmed(bed), 1.0)
            fx.sweep(canvas, palette, bounce(16.0), width=0.12, level=0.95, geo=G)
        elif name == "sweep_fast":
            fx.wash(canvas, palette.dimmed(bed), 1.0)
            fx.sweep(canvas, palette, bounce(8.0), width=0.10, level=1.0, geo=G)
        elif name == "fall":
            # Top to bottom, one drop per bar, over every net at once.
            fx.wash(canvas, palette.dimmed(bed), 1.0)
            fx.sweep(canvas, palette, (beats / 4.0) % 1.0, angle=1.0, width=0.2,
                     level=0.9, geo=G)
        elif name == "rise":
            # Bottom to top, quickening with the build.
            fx.wash(canvas, palette.dimmed(bed), 1.0)
            fx.sweep(canvas, palette, 1.0 - (beats * (0.25 + 0.75 * tension)) % 1.0,
                     angle=1.0, width=0.2, level=0.9, geo=G)
        elif name == "diagonal":
            fx.wash(canvas, palette.dimmed(bed), 1.0)
            fx.sweep(canvas, palette, bounce(16.0), angle=0.4, width=0.14,
                     level=0.95, geo=G)
        elif name == "diagonal_fast":
            fx.wash(canvas, palette.dimmed(bed), 1.0)
            fx.sweep(canvas, palette, bounce(4.0), angle=0.6, width=0.12,
                     level=1.0, geo=G)
        elif name == "ball":
            # A ball crossing the room over four bars and bouncing on the
            # floor once a bar.
            fx.wash(canvas, palette.dimmed(bed), 1.0)
            hop = abs(np.sin(np.pi * (beats / 4.0)))
            fx.blob(canvas, palette, bounce(32.0), 0.85 - 0.6 * hop,
                    radius=0.22, level=1.0, geo=G)
        elif name == "ball_fast":
            # Crosses in two bars, bounces on every beat.
            fx.wash(canvas, palette.dimmed(bed), 1.0)
            hop = abs(np.sin(np.pi * beats))
            fx.blob(canvas, palette, bounce(16.0), 0.85 - 0.65 * hop,
                    radius=0.2, level=1.0, geo=G)
            fx.blob(canvas, pal.WHITE, bounce(16.0), 0.85 - 0.65 * hop,
                    radius=0.06, level=0.7 * kick, geo=G)
        elif name == "burst":
            # A ring out of the centre of the whole array on every beat.
            fx.wash(canvas, palette.dimmed(bed), 1.0)
            fx.radial(canvas, palette, beat, width=0.18, level=1.0, geo=G)
        elif name == "slam":
            # Both ends rush in and meet in the middle on the beat.
            fx.wash(canvas, palette.dimmed(bed), 1.0)
            fx.sweep(canvas, palette, 0.5 * beat, width=0.1, level=1.0, geo=G)
            fx.sweep(canvas, palette, 1.0 - 0.5 * beat, width=0.1, level=1.0, geo=G)
            fx.blob(canvas, pal.WHITE, 0.5, 0.5, radius=0.25,
                    level=_kick(beat, sharp=4.0) * 0.8, geo=G)
        elif name == "squeeze":
            # The two ends close in as the build tightens, slowly.
            fx.wash(canvas, palette.dimmed(bed), 1.0)
            at = 0.5 * tension * (0.5 + 0.5 * np.sin(np.pi * beats / 4.0))
            fx.sweep(canvas, palette, at, width=0.12, level=0.95, geo=G)
            fx.sweep(canvas, palette, 1.0 - at, width=0.12, level=0.95, geo=G)
        elif name in ("scan", "scan_up", "scan_fast"):
            # The nets one after another, left to right.
            order = canvas.net_order
            per_beat = {"scan": 0.5, "scan_up": 0.5 + tension, "scan_fast": 2.0}[name]
            position = (beats * per_beat) % len(order)
            fx.wash(canvas, palette.dimmed(bed), 1.0)
            for rank, index in enumerate(order):
                gap = min(abs(position - rank), len(order) - abs(position - rank))
                glow = max(0.0, 1.0 - gap / 1.2) ** 1.5
                if glow > 0.0:
                    fx.wash(canvas, palette, glow * 0.95, gradient=0.6,
                            targets=slice(index, index + 1))
        else:
            raise ValueError(f"unknown all gesture {name!r}")

    def _quiet(self, canvas, gesture, frame, t, phrase, palette, beat, bar,
               kick, features) -> None:
        self._gesture(canvas, gesture, frame, t, phrase, palette, beat, kick)
        fx.par(canvas, palette.color(0), 0.25 + 0.15 * phrase)

    def _cruising(self, canvas, gesture, frame, t, phrase, palette, beat, bar,
                  kick, features) -> None:
        fx.wash(canvas, palette.dimmed(0.35), 1.0, gradient=0.8)
        self._gesture(canvas, gesture, frame, t, phrase, palette, beat, kick)
        fx.par(canvas, palette.color(1), 0.35 + 0.45 * kick)

    def _building(self, canvas, gesture, frame, t, phrase, palette, beat, bar,
                  kick, features) -> None:
        # Tension tracks how far into the build we are, but it is *abortable*:
        # if the sweep stops without a drop, this simply relaxes.  Never
        # pre-fire the resolution.
        tension = min(1.0, self.machine.report.since_s / 8.0)
        fx.wash(canvas, palette.dimmed(0.25), 1.0, gradient=1.0)
        self._gesture(canvas, gesture, frame, t, phrase, palette, beat, kick,
                      tension=tension)
        flash = _kick((t / max(1e-6, 60.0 / self.bpm)
                       * (2 + int(tension * 6))) % 1.0, sharp=3.0)
        fx.par(canvas, palette.color(0), 0.4 + 0.6 * tension * flash,
               white=0.3 * tension)

    def _hot(self, canvas, gesture, frame, t, phrase, palette, beat, bar,
             kick, features) -> None:
        fx.wash(canvas, palette.dimmed(0.3), kick * 0.8)
        self._gesture(canvas, gesture, frame, t, phrase, palette, beat, kick,
                      tension=1.0)
        fx.par(canvas, pal.WHITE.color(0), kick, white=kick)


def _kick(phase: float, sharp: float = 2.0) -> float:
    return float(max(0.0, 1.0 - phase) ** sharp)


def _identity(look: tuple) -> tuple:
    """What makes a look different from the last: material, or colour."""
    palette = look[3]
    return (*look[:3], palette.name if palette is not None else "")


def _smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _mix_canvases(canvas: Canvas, old: Canvas, style: str, mix: float) -> None:
    """Blend the outgoing look (``old``) into ``canvas`` by ``style``.

    ``mix`` runs 0 (all old) to 1 (all new).  The corridor wipes along its
    depth and the nets along their height, so a wipe reads as one motion
    across the whole rig rather than two unrelated ones.
    """
    m = _smooth(mix)
    if style == "fade":
        canvas.arches *= m
        canvas.arches += old.arches * (1.0 - m)
        canvas.nets *= m
        canvas.nets += old.nets * (1.0 - m)
    elif style in ("wipe_back", "wipe_front"):
        soft = 0.25
        depth = canvas.depth if style == "wipe_back" else 1.0 - canvas.depth
        edge = m * (1.0 + soft)
        arch_w = np.clip((edge - depth) / soft, 0.0, 1.0)[:, None, None]
        net_w = np.clip((edge - canvas.net_y) / soft, 0.0, 1.0)[:, :, None]
        canvas.arches *= arch_w
        canvas.arches += old.arches * (1.0 - arch_w)
        canvas.nets *= net_w
        canvas.nets += old.nets * (1.0 - net_w)
    elif style == "dip":
        # A crossfade through a dimmer middle: the rig drops to 40% at the
        # halfway point, which reads as a breath between two ideas.
        dip = 1.0 - 0.6 * float(np.sin(np.pi * m))
        canvas.arches *= m * dip
        canvas.arches += old.arches * ((1.0 - m) * dip)
        canvas.nets *= m * dip
        canvas.nets += old.nets * ((1.0 - m) * dip)
    canvas.par *= m
    canvas.par += old.par * (1.0 - m)
