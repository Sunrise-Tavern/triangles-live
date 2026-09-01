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

import hashlib
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
    "volley": (CRUISING, HOT),
    "tide": (CRUISING, BUILDING),
    "swarm": (QUIET, CRUISING),
    "storm": (BUILDING, HOT),
    "pendulum": (CRUISING, BUILDING),
}


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
        #: Peak-hold envelope of the *measured* kick (bass flux over its own
        #: average): jumps on a hit, decays over about a bar.  This is what
        #: lets a whole showpiece ride the bass -- the instantaneous detector
        #: is a spike a frame wide, unusable as a level.
        self._pump = 0.0
        self._pump_t: float | None = None
        #: Beat anchor per (piece, kind, visit, material): a showpiece's arc
        #: must run from the start of its *material stretch*, not from each
        #: phrase -- anchored per phrase, the 8-bar cycle restarted every 4
        #: bars and the blow never fired (measured: bars 0-3 only).
        self._piece_anchors: dict[tuple, float] = {}
        #: Storm bolts in flight: launch times.  Spawned by measured kicks.
        self._bolts: list[float] = []
        self._bolt_last = -1e9
        #: Per clip stretch: (last beat count, playhead position in beats).
        #: The playhead integrates a bass-scaled rate, so it cannot jump.
        self._clip_heads: dict[tuple, tuple[float, float]] = {}
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
        return (_hash(f"{self.seed}:{key}:{treat.kind}:{visit}") % 1000) / 999.0

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

        The start is hashed too, not 0: stepping from a fixed start meant
        the first phrase of every visit could land anywhere *except* index
        0, so the option the treatment was tuned around never opened a
        visit -- audited at exactly 0.000 share.
        """
        if count < 2:
            return 0
        last, index = self._walks.get(key, (None, None))
        if last == phrase:
            return index
        if last is None:
            index = _hash(f"{self.seed}:{key}:init") % count
            first = phrase - phrase          # walk every step from 0
        else:
            first = last + 1
        for step_phrase in range(first, phrase + 1):
            step = _hash(f"{self.seed}:{key}:{step_phrase}")
            index = (index + 1 + step % (count - 1)) % count
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
        roll = _hash(f"{self.seed}:transition:{change}")
        if (roll % 1000) / 1000.0 >= share:
            return ("cut", 0.0)
        return TRANSITIONS[(roll // 1000) % len(TRANSITIONS)]

    # -- render ------------------------------------------------------------ #

    def render(self, frame: int, t: float) -> None:
        canvas = self.canvas
        canvas.clear()

        features_now = self.features
        if self._pump_t is None:
            self._pump_t = t
        decayed = self._pump * float(np.exp(-(t - self._pump_t) * 1.6))
        instant = (min(1.0, float(features_now.kick) / 4.0)
                   if features_now is not None else 0.0)
        self._pump = max(decayed, instant)
        self._pump_t = t

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
            key = (piece_name, treat.kind, self.visits.get(treat.kind, 1),
                   self.material_index(index))
            if len(self._piece_anchors) > 64:
                self._piece_anchors.clear()
            anchor = self._piece_anchors.setdefault(key, self._beats(t))
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
        roll = _hash(f"{self.seed}:cliproll:{treat.kind}:{visit}:{material}")
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
        share = self.settings.piece_share if self.settings else 0.25
        visit = self.visits.get(treat.kind, 1)
        material = self.material_index(phrase)
        roll = _hash(f"{self.seed}:pieceroll:{treat.kind}:{visit}:{material}")
        if (roll % 1000) / 1000.0 >= share:
            return None
        return options[self._walk(f"pieces:{treat.kind}:{visit}", material,
                                  len(options))]

    #: Both showpieces share one arc, in bars: travel in from the back of
    #: the tunnel, hit the triangles, travel home.  Eight bars a cycle, the
    #: hit landing on a downbeat because the runs are whole bars.
    CHARGE_ARC = (4.0, 1.0, 3.0)      # in, blow, out
    DNA_ARC = (4.0, 2.0, 2.0)         # in, bounce, out

    def _piece_charge(self, canvas: Canvas, kind: str, t: float, anchor: float,
                      palette, beat: float, kick: float, features) -> None:
        """A comet charges from the back of the tunnel, shatters the
        triangles at the mouth, and runs home.

        Four bars in, hue stepping 45 degrees per bar from a base randomized
        (seeded) per cycle; one full bar of blow -- the tunnel's mouth glows
        and dies while every triangle pixel ignites, burns white, fades at
        its own rate and drops out (fx.shatter, re-seeded per cycle); three
        bars back.  The blow lands exactly on a bar line: the clock predicts
        the arrival, nothing reacts late.
        """
        run_in, blow_bars, run_out = self.CHARGE_ARC
        cycle_bars = run_in + blow_bars + run_out
        bars = (self._beats(t) - anchor) / max(1, self.clock.bar_length)
        cycle, u = int(bars // cycle_bars), bars % cycle_bars
        pump = self._pump
        punch = kick * (0.4 + 0.6 * pump)          # beat-shaped, bass-sized
        base = (_hash(f"{self.seed}:charge:{cycle}") % 3600) / 10.0
        paint = pal.generate(base + 45.0 * int(u), "complementary",
                             value=0.80 + 0.20 * punch).floored()
        far = paint.rotated(60.0)

        G = canvas.all_geo
        if u < run_in or u >= run_in + blow_bars:  # -- travelling
            if u < run_in:
                position = 1.0 - u / run_in
            else:
                position = (u - run_in - blow_bars) / run_out
            glow = np.exp(-((canvas.depth - position) * 6.0) ** 2)
            levels = (0.05 + 0.05 * punch + 0.90 * glow * (0.65 + 0.35 * punch))
            # The triangles keep time with the bass while the comet travels:
            # a wash that lands with each hit, and a soft centre bounce.
            fx.plasma(canvas, paint.dimmed(0.5), t, scale=2.0, speed=0.4,
                      level=0.25 + 0.25 * pump)
            fx.blob(canvas, paint, 0.5, 0.72 - 0.30 * pump * kick,
                    radius=0.24, level=0.35 + 0.45 * punch, geo=G)
        else:                                      # -- the blow
            blow = (u - run_in) / blow_bars
            levels = (0.05 + 0.95 * np.exp(-canvas.depth * 5.0)
                      * (1.0 - 0.7 * blow))
            fx.shatter(canvas, paint, blow,
                       seed=_hash(f"{self.seed}:blow:{cycle}"),
                       level=0.8 + 0.2 * pump, geo=G)
            if blow < 0.15:
                fx.blob(canvas, pal.WHITE, 0.5, 0.5, radius=0.25,
                        level=(1.0 - blow / 0.15) * (0.5 + 0.5 * pump), geo=G)
        fx.corridor(canvas, levels.astype(np.float32), paint, far,
                    brightness=1.0, height=0.35)

    def _piece_dna(self, canvas: Canvas, kind: str, t: float, anchor: float,
                   palette, beat: float, kick: float, features) -> None:
        """A DNA segment travels the tunnel, and the triangles bounce.

        The helix is a *travelling* stretch about two rings long (a focus
        window over six turns of thread), screwing toward the mouth over
        four bars.  Arrived, it parks at the mouth and spins while the
        triangles bounce for two bars: two balls hopping in counter-phase
        across the whole array, hop depth driven by the measured kick --
        bass flux over its own average -- with a white core flashing on the
        hit.  Then it screws home in two.  A soft centre hop keeps the
        triangles breathing with the bass even while the helix travels.
        """
        run_in, bounce_bars, run_out = self.DNA_ARC
        cycle_bars = run_in + bounce_bars + run_out
        beats = self._beats(t) - anchor
        bars = beats / max(1, self.clock.bar_length)
        u = bars % cycle_bars

        strength = min(1.0, float(getattr(features, "kick", 0.0)) / 4.0)             if features is not None else 0.5
        hop = abs(float(np.sin(np.pi * beats)))
        G = canvas.all_geo

        if u < run_in:
            centre = 1.0 - u / run_in
        elif u < run_in + bounce_bars:
            centre = 0.0
        else:
            centre = (u - run_in - bounce_bars) / run_out
        fx.helix(canvas, palette, beats / 4.0, turns=6.0, level=0.95,
                 focus=(centre, 5.5))
        fx.corridor(canvas, np.full(len(canvas.arch_names), 0.05,
                                    dtype=np.float32),
                    palette.dimmed(0.6), palette.rotated(60.0),
                    brightness=1.0, height=0.35)

        fx.plasma(canvas, palette.dimmed(0.35), t, scale=2.2, speed=0.5,
                  level=0.25)
        if run_in <= u < run_in + bounce_bars:
            # The hit: two balls bouncing in counter-phase, as deep as the
            # bass hits hard.
            depth = 0.30 + 0.55 * strength
            hop2 = abs(float(np.sin(np.pi * (beats + 0.5))))
            fx.blob(canvas, palette, 0.32, 0.88 - depth * hop,
                    radius=0.20, level=0.95, geo=G)
            fx.blob(canvas, palette.rotated(180.0), 0.68, 0.88 - depth * hop2,
                    radius=0.20, level=0.95, geo=G)
            env = _kick(beat, sharp=2.5) * strength
            if env > 0.05:
                fx.blob(canvas, pal.WHITE, 0.5, 0.5,
                        radius=0.12 + 0.30 * (1.0 - env), level=env * 0.8,
                        geo=G)
        else:
            # Travelling: the triangles keep a small bounce with the bass.
            fx.blob(canvas, palette, 0.5, 0.80 - 0.25 * strength * hop,
                    radius=0.22, level=0.45, geo=G)

    def _piece_volley(self, canvas: Canvas, kind: str, t: float, anchor: float,
                      palette, beat: float, kick: float, features) -> None:
        """A rally: down the tunnel, off the back wall, out again, then
        across the triangles and back.  One leg per bar, so every bounce
        lands on a downbeat; each bounce flashes white, as hard as the bass
        hits.  Four bars a rally, two rallies a cycle."""
        pump = self._pump
        punch = kick * (0.4 + 0.6 * pump)
        bars = (self._beats(t) - anchor) / max(1, self.clock.bar_length)
        unit = bars % 4.0
        G = canvas.all_geo
        turn = _kick(unit % 1.0, sharp=3.0) * (0.3 + 0.7 * pump)

        levels = np.full(len(canvas.arch_names), 0.05 + 0.05 * punch,
                         dtype=np.float32)
        if unit < 2.0:                     # ball in the tunnel
            position = unit if unit < 1.0 else 2.0 - unit
            levels += 0.95 * np.exp(-((canvas.depth - position) * 7.0) ** 2) \
                * (0.7 + 0.3 * punch)
            if unit < 1.0 and unit > 0.85:      # about to hit the back wall
                levels[-2:] += turn
            fx.plasma(canvas, palette.dimmed(0.4), t, scale=2.2, speed=0.5,
                      level=0.22 + 0.20 * pump)
        else:                              # ball across the triangles
            x = unit - 2.0 if unit < 3.0 else 4.0 - unit
            fx.blob(canvas, palette, x, 0.5, radius=0.20,
                    level=0.85 + 0.15 * punch, geo=G)
            fx.blob(canvas, pal.WHITE, x, 0.5, radius=0.08, level=turn, geo=G)
        fx.corridor(canvas, levels, palette, palette.rotated(60.0),
                    brightness=1.0, height=0.35)

    def _piece_tide(self, canvas: Canvas, kind: str, t: float, anchor: float,
                    palette, beat: float, kick: float, features) -> None:
        """Water floods the tunnel toward the mouth, crashes on the
        triangles as foam, and drains back.  Four bars in, two of crash --
        white spray raining apex to base, splash sized by the bass -- and
        two to drain."""
        pump = self._pump
        punch = kick * (0.4 + 0.6 * pump)
        bars = (self._beats(t) - anchor) / max(1, self.clock.bar_length)
        u = bars % 8.0
        surge = 0.85 + 0.15 * float(np.sin(2 * np.pi * bars))    # the swell
        if u < 4.0:                          # flooding, back toward the mouth
            front = 1.0 - u / 4.0
            levels = np.clip((canvas.depth - front) / 0.25 + 1.0, 0.0, 1.0)
        elif u < 6.0:                        # crashed: the tunnel sloshes full
            levels = np.full(len(canvas.arch_names), 0.8)
            crash = (u - 4.0) / 2.0
            G = canvas.all_geo
            fx.sweep(canvas, pal.WHITE, crash, angle=1.0,
                     width=0.20 + 0.15 * pump, level=(1.0 - crash * 0.5), geo=G)
            fx.net_sparkle(canvas, pal.WHITE, int(t * 40),
                           density=0.01 + 0.04 * pump, level=0.9,
                           seed=self.seed)
        else:                                # draining, mouth toward the back
            front = (u - 6.0) / 2.0
            levels = np.clip((canvas.depth - front) / 0.25 + 1.0, 0.0, 1.0) * 0.7
        fx.corridor(canvas, (levels * surge * (0.75 + 0.25 * punch)
                             ).astype(np.float32),
                    palette, palette.rotated(40.0), brightness=0.95,
                    height=0.25)
        fx.wash(canvas, palette.dimmed(0.30 + 0.25 * pump), 1.0, gradient=0.9)

    def _piece_swarm(self, canvas: Canvas, kind: str, t: float, anchor: float,
                     palette, beat: float, kick: float, features) -> None:
        """A cloud of sparks drifts through the tunnel, settles on the
        triangles as two counter-spinning wheels, and swarms home.  The
        bass scatters it: a heavy hit widens the cloud and thickens the
        sparks."""
        pump = self._pump
        bars = (self._beats(t) - anchor) / max(1, self.clock.bar_length)
        beats = self._beats(t) - anchor
        u = bars % 8.0
        if u < 3.0:
            centre = 1.0 - u / 3.0
        elif u < 5.0:
            centre = 0.0
        else:
            centre = (u - 5.0) / 3.0
        window = np.exp(-((canvas.depth - centre) * (5.0 - 2.5 * pump)) ** 2)
        twinkle = fx.sparkle(len(canvas.arch_names), (bars / 2.0) % 1.0,
                             steps=16, fraction=0.35 + 0.25 * pump,
                             seed=self.seed)
        levels = (0.04 + (0.5 + 0.5 * twinkle) * window).astype(np.float32)
        fx.corridor(canvas, np.clip(levels, 0.0, 1.0), palette,
                    palette.rotated(50.0), brightness=0.9, height=0.3)
        fx.plasma(canvas, palette.dimmed(0.35), t, scale=2.5, speed=0.3,
                  level=0.25 + 0.15 * pump)
        if 3.0 <= u < 5.0:
            # Settled: the swarm organizes into wheels, pulsing on the beat.
            spin = 0.75 + 0.25 * kick * (0.4 + 0.6 * pump)
            fx.pinwheel(canvas, palette, beats / 8.0, arms=3, level=spin,
                        targets=self.big, geo=canvas.big_geo)
            fx.pinwheel(canvas, palette.rotated(40.0), -beats / 6.0, arms=3,
                        level=spin * 0.8, targets=self.small)
            fx.net_sparkle(canvas, pal.WHITE, int(t * 40),
                           density=0.004 + 0.01 * pump, level=0.8,
                           seed=self.seed)

    def _piece_storm(self, canvas: Canvas, kind: str, t: float, anchor: float,
                     palette, beat: float, kick: float, features) -> None:
        """Clouds gather on the triangles; then the bass throws lightning.

        During the strike bars every qualifying kick -- the *measured*
        detector, not the grid -- launches a bolt from the triangles down
        the tunnel, streaking back with an afterglow, while the triangles
        flash white at the moment of birth.  No bass, no bolts: the storm
        is only as violent as the music.  Two bars gathering, four of
        strikes, two of afterglow."""
        pump = self._pump
        bars = (self._beats(t) - anchor) / max(1, self.clock.bar_length)
        u = bars % 8.0
        beat_s = max(1e-6, 60.0 / self.bpm)

        instant = float(getattr(features, "kick", 0.0)) if features is not None else 0.0
        if (2.0 <= u < 6.0 and instant >= 3.0
                and t - self._bolt_last >= 0.5 * beat_s):
            self._bolts.append(t)
            self._bolt_last = t
        self._bolts = [b for b in self._bolts if t - b < 1.2]

        levels = np.full(len(canvas.arch_names), 0.03, dtype=np.float32)
        flash = 0.0
        for born in self._bolts:
            age = (t - born) / 1.2
            position = age * 1.4                    # mouth -> back, and out
            levels += (np.exp(-((canvas.depth - position) * 5.0) ** 2)
                       * (1.0 - age)).astype(np.float32)
            flash = max(flash, (1.0 - age * 4.0))
        if u >= 6.0:
            levels += 0.10 * (1.0 - (u - 6.0) / 2.0)
        fx.corridor(canvas, np.clip(levels, 0.0, 1.0), palette.floored(),
                    pal.WHITE, brightness=0.95, height=0.4)

        # The cloud: heavy, slow, rumbling with the pump.
        fx.plasma(canvas, palette.dimmed(0.30 + 0.30 * pump), t, scale=1.8,
                  speed=0.8, level=0.35 + 0.25 * pump)
        if flash > 0.0:
            fx.wash(canvas, pal.WHITE, flash * 0.9)

    def _piece_pendulum(self, canvas: Canvas, kind: str, t: float, anchor: float,
                        palette, beat: float, kick: float, features) -> None:
        """A pendulum swings the width of the array, one full swing per
        bar, so it strikes an end on every other beat -- and each strike
        launches a pulse down the tunnel.  The bass sets how hard: swing
        brightness, strike flash and pulse depth all ride the pump."""
        pump = self._pump
        punch = kick * (0.4 + 0.6 * pump)
        bars = (self._beats(t) - anchor) / max(1, self.clock.bar_length)
        G = canvas.all_geo

        swing = float(np.cos(2 * np.pi * bars))          # 1 = left end
        x = 0.5 - 0.46 * swing
        y = 0.42 + 0.40 * ((x - 0.5) / 0.46) ** 2        # the arc of the bob
        fx.wash(canvas, palette.dimmed(0.18 + 0.12 * pump), 1.0, gradient=0.8)
        fx.blob(canvas, palette, x, y, radius=0.16 + 0.04 * punch,
                level=0.8 + 0.2 * punch, geo=G)
        strike = _kick((bars * 2.0) % 1.0, sharp=3.0) * (0.3 + 0.7 * pump)
        if strike > 0.05:
            end_x = 0.04 if swing > 0 else 0.96
            fx.blob(canvas, pal.WHITE, end_x, 0.75, radius=0.12,
                    level=strike, geo=G)

        # Each strike sends a pulse into the tunnel; two live at once.
        levels = np.full(len(canvas.arch_names), 0.05 + 0.05 * punch,
                         dtype=np.float32)
        for offset in (0.0, 0.5):
            phase = (bars - offset) % 1.0
            levels += (np.exp(-((canvas.depth - phase) * 6.0) ** 2)
                       * (1.0 - phase) * (0.5 + 0.5 * pump)).astype(np.float32)
        fx.corridor(canvas, np.clip(levels, 0.0, 1.0), palette,
                    palette.rotated(70.0), brightness=0.95, height=0.35)

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
        # The playhead runs on a bass-scaled rate: the authored motion pushes
        # harder when the low end does, eases off in a lull.  Integrated, not
        # multiplied -- position must stay continuous as the rate moves.
        pump = self._pump
        now_beats = self._beats(t)
        key = (name, round(anchor, 4))
        if len(self._clip_heads) > 64:
            self._clip_heads.clear()
        last, position = self._clip_heads.get(key, (now_beats, 0.0))
        if now_beats > last:            # a transition paints twice per frame
            position += (now_beats - last) * (0.65 + 0.70 * pump)
            self._clip_heads[key] = (now_beats, position)
        canvas.from_channels(clip.frame_at_beats(position))

        if kind == QUIET:
            envelope = 0.65 + 0.15 * float(np.sin(2 * np.pi * self.clock.bar_phase(t)))
            bounce = 0.0
        elif kind == BUILDING:
            tension = min(1.0, self.machine.report.since_s / 8.0)
            flash = _kick((now_beats * (2 + int(tension * 6))) % 1.0, sharp=3.0)
            envelope = 0.55 + 0.15 * tension + 0.30 * tension * flash
            bounce = 0.25 * tension * pump
        elif kind == HOT:
            envelope = 0.70 + 0.30 * kick
            bounce = 0.40 * pump
        else:
            envelope = 0.75 + 0.25 * kick
            bounce = 0.30 * pump
        canvas.nets *= envelope
        canvas.arches *= envelope
        if bounce > 0.01:
            # The content itself bounces with the bass: a brightness wave
            # rolls apex-to-base through the triangles on each beat, and a
            # ripple runs the clip's tunnel front to back with it.
            wave = np.exp(-((canvas.net_y - beat) * 2.5) ** 2)
            canvas.nets *= (1.0 - bounce) + (2.0 * bounce) * wave[..., None]
            ripple = np.exp(-((canvas.depth - beat) * 3.0) ** 2)
            canvas.arches *= ((1.0 - bounce)
                              + (2.0 * bounce) * ripple[:, None, None])

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


def _hash(text: str) -> int:
    """A stable, well-mixed 32-bit hash of ``text``.

    Not crc32, which this replaced: crc is *linear*, and the walk's keys
    differ only in a trailing digit, so its low bits -- exactly what a
    ``% (count - 1)`` keeps, worst when that is a power of two -- inherited
    the digit's structure instead of mixing it.  Audited over 12000 draws:
    a 9-option list came out with shares 0.06-0.18 against a fair 0.11.
    blake2b's are 0.107-0.115.  Not ``hash()``, which is salted per process
    and would render a different show every run.
    """
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=4).digest(),
                          "big")


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
