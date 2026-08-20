"""Turn analysed audio into an effect timeline.

The rig has three pieces and each wants a different kind of motion:

* **Tunnel** -- 24 arches. The group element carries a colour bed; the
  corridor's signature move is built from *per-arch* effects with staggered
  starts, because the arches all sit at the same 3D position in the layout, so
  a group-level sweep has no depth to travel through.
* **Nets** -- real 2D buffers, so pattern effects (Butterfly, Spirals,
  Pinwheel, Shockwave) read well.
* **DJ par** -- one DMX fixture, effectively a single pixel. Flat colour only.

``ModelBlending`` is on in the sequence header, so the Tunnel group bed and the
per-arch waves blend rather than one overwriting the other.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from . import effects as fx
from . import palettes as pal
from .analysis import Features
from .palettes import Palette
from .show import Show
from .structure import Section
from .xsq import SequenceBuilder

#: Layer indices in the written file. In xLights the FIRST EffectLayer is the
#: top of the stack, and with the default "Normal" blend an effect there covers
#: whatever is below it for as long as it runs. So the short accents go on
#: layer 0 and the sustained beds on layer 1 -- the same shape as the hand-made
#: sequence in this show. Getting this backwards puts a wall-to-wall wash on
#: top and hides every accent underneath it, which renders as smooth glowing
#: with no beat at all however many hits the file contains.
ACCENT, BASE = 0, 1

#: A corridor wave must take long enough that consecutive arches land on
#: different 50ms frames, otherwise the whole tunnel just flashes at once.
MIN_TRAVEL_S = 1.3

#: Minimum peak channel per section kind, 0-255. Quiet sections are allowed to
#: be dim but never invisible -- an LED below roughly 40/255 reads as off.
FLOOR = {
    "intro": 64,
    "break": 64,
    "outro": 84,
    "verse": 110,
}

#: Harmony scheme per section kind. Contrast rises with energy: quiet sections
#: sit in neighbouring hues, drops use three evenly spaced ones.
SCHEME = {
    "intro": "analogous",
    "break": "analogous",
    "outro": "analogous",
    "verse": "split",
    "build": "complementary",
    "drop": "triadic",
}

#: Value (brightness) and saturation per section kind, before the floor.
VALUE = {"intro": 0.60, "break": 0.55, "outro": 0.50,
         "verse": 0.88, "build": 0.96, "drop": 1.00}
SAT = {"intro": 0.92, "break": 0.92, "outro": 0.90,
       "verse": 0.95, "build": 0.80, "drop": 0.88}
SPARKLE = {"build": 60, "drop": 85}

#: Multiplies the corridor traversal length per section kind, so a drop stays
#: lively and an intro stays calm regardless of the track's overall character.
GESTURE_SCALE = {
    "intro": 1.8, "break": 1.8, "outro": 2.0,
    "verse": 1.0, "build": 0.8, "drop": 0.6,
}


@dataclass
class Style:
    """Knobs that shift the overall feel."""

    name: str = "edm"
    #: Multiplies how many accent hits get placed.
    accent_density: float = 1.0
    #: Multiplies corridor wave frequency.
    wave_density: float = 1.0
    #: Minimum normalized onset strength for an accent hit.
    accent_threshold: float = 0.35
    #: Bars per corridor traversal. Lower = the tunnel crosses more often and
    #: feels faster; higher = longer, calmer sweeps. This is the single knob
    #: between "looks slow" and "too much blinking", so it is worth tuning by
    #: eye in the 3D preview rather than by any measurement.
    gesture_bars: float = 1.0


STYLES = {
    "edm": Style("edm", accent_density=1.0, wave_density=1.0,
                 accent_threshold=0.35, gesture_bars=1.0),
    "rock": Style("rock", accent_density=0.8, wave_density=0.7,
                  accent_threshold=0.45, gesture_bars=1.5),
    "ambient": Style("ambient", accent_density=0.3, wave_density=0.4,
                     accent_threshold=0.55, gesture_bars=2.0),
}


def detect_style(f: Features) -> Style:
    """Derive the arrangement's density from the track itself.

    The named presets are three points on what is really a continuum, and
    picking between them by hand asks the operator to classify the genre when
    the audio already answers the question. `Features.drive` combines how
    clearly the track states a pulse with how many onsets it puts in each beat;
    everything below scales off that.

    A tight, busy track gets fast corridor gestures and hits on most beats. A
    loose, sparse one gets long sweeps and only the strongest accents, because
    hitting every beat of music that does not insist on its beat looks
    mechanical rather than tight.
    """
    d = f.drive
    style = Style(
        name=f"auto({d:.2f})",
        accent_density=round(0.30 + 0.70 * d, 3),
        wave_density=round(0.50 + 0.50 * d, 3),
        accent_threshold=round(0.55 - 0.20 * d, 3),
        gesture_bars=round(2.00 - 1.20 * d, 3),
    )
    return style


class Arranger:
    def __init__(self, show: Show, features: Features, sections: list[Section],
                 builder: SequenceBuilder, *, seed: int = 0,
                 style: str | None = None):
        self.show = show
        self.f = features
        self.sections = sections
        self.b = builder
        # Everything that does not name a layer is a sustained bed.
        self.b.default_layer = BASE
        self.rng = random.Random(seed)
        # No named style means derive one from the audio.
        self.style = STYLES[style] if style in STYLES else detect_style(features)
        self.arches = show.tunnel_arches
        self._arch_pos = {name: i for i, name in enumerate(self.arches)}
        self._wave_dir = 0
        #: Last corridor pattern used, so consecutive phrases never repeat.
        self._last_pattern: str | None = None
        #: Sections already articulated on the beat; onset accents skip these.
        self._articulated: set[int] = set()
        #: Colour identity for this track, taken from its dominant pitch class.
        self._base_hue = features.key_hue
        self._sec_index = {id(s): i for i, s in enumerate(sections)}

        # Coarse-to-fine net groups. The show defines three, but fall back
        # gracefully if someone removes one in xLights.
        groups = show.net_groups
        self._all = groups[0]
        self._big = groups[1] if len(groups) > 1 else groups[0]
        self._small = groups[2] if len(groups) > 2 else self._big

    # -- entry point --------------------------------------------------------

    def arrange(self) -> None:
        # Declaring up front fixes the element order in the xLights timeline:
        # the groups you actually watch first, then the arches, then the par.
        self.b.declare(self.show.tunnel, *self.show.net_groups, *self.arches,
                       self.show.dj)

        self._timing_tracks()

        for i, sec in enumerate(self.sections):
            prev = self.sections[i - 1] if i else None
            nxt = self.sections[i + 1] if i + 1 < len(self.sections) else None
            handler = getattr(self, f"_{sec.kind}", self._verse)
            handler(sec, prev, nxt)

        self._accents()

    # -- timing tracks ------------------------------------------------------

    def _timing_tracks(self) -> None:
        """Beats, bars and named sections.

        These are as valuable as the effects: with them, hand-editing the
        result in xLights snaps to the real musical grid.
        """
        dur = self.b.duration_s

        beats = [t for t in self.f.beat_times if 0 <= t < dur]
        self.b.add_timing_track("Beats", [
            (t, nxt, "") for t, nxt in zip(beats, [*beats[1:], dur])
        ])

        bars = [t for t in self.f.bar_times if 0 <= t < dur]
        self.b.add_timing_track("Bars", [
            (t, nxt, "") for t, nxt in zip(bars, [*bars[1:], dur])
        ])

        self.b.add_timing_track("Sections", [
            (s.start, s.end, s.kind) for s in self.sections
        ])

    # -- palette helpers ----------------------------------------------------

    def _hue(self, sec: Section) -> float:
        """This section's base hue, in degrees.

        Starts from the track's own key and rotates by the golden angle each
        section, so consecutive sections are always far apart on the wheel and
        the sequence takes a long time to revisit a hue. The brightness nudge
        keeps two sections of the same kind from landing on the same colour
        just because they sit at the same point in the rotation.
        """
        i = self._sec_index.get(id(sec), 0)
        return (self._base_hue
                + i * pal.GOLDEN_ANGLE
                + (sec.brightness - 0.5) * 40.0) % 360.0

    def _palette(self, sec: Section, *, dim: float = 1.0,
                 hue_shift: float = 0.0) -> Palette:
        kind = sec.kind
        # -1 minor .. +1 major. Harmony sets the *character* of the colour, never
        # its identity: a minor section leans cool, desaturated and tightly
        # spaced; a major one leans warm and open. Hue itself stays on the
        # rotation, so two sections in the same key still look different.
        q = self.f.quality(sec.start, sec.end)

        scheme = SCHEME.get(kind, "analogous")
        if kind in ("verse", "break", "intro", "outro"):
            # Minor pulls toward the closest-spaced scheme, major toward the
            # most open one the section's energy allows.
            if q < -0.25:
                scheme = "analogous"
            elif q > 0.25:
                scheme = "split"

        p = pal.generate(
            # Warm sections drift a little toward the warm half of the wheel,
            # cold ones the other way -- a nudge, not a relocation.
            self._hue(sec) + hue_shift + q * 18.0,
            scheme,
            value=VALUE.get(kind, 0.9) * dim * (1.0 + 0.06 * q),
            sat=SAT.get(kind, 0.92) * (1.0 - 0.12 * max(0.0, -q)),
            sparkles=SPARKLE.get(kind, 0),
            white=kind in ("build", "drop"),
        )
        # Backstop against a dim multiplier landing below the visible threshold.
        return p.floored(FLOOR.get(kind, 90))

    def _depth(self, palette: Palette, amount: float = 0.3,
               floor: int = 44, hue_shift: float = 55.0) -> Palette:
        """The colour the far end of the corridor sits at.

        Keeping the hue and dropping the value gives a light-to-dark run down
        the tunnel -- light blue at the near end, deep blue at the far end --
        which reads as depth rather than as 24 identical arches.

        Floored as well: dimming an already-quiet palette for depth is the
        second way to arrive at an invisible far end, and losing the back half
        of the corridor costs more than the gradient gains.

        The hue rotates across the corridor too, so the tunnel runs through a
        colour rather than just fading out -- cyan at the mouth to violet at the
        back reads as far more depth than cyan to dark-cyan.
        """
        return palette.rotated(hue_shift).dimmed(amount).floored(floor)

    def _cycles(self, start: float, end: float, per_bar: float = 0.5) -> float:
        """Cycle count for a sweeping effect, scaled to the span it covers.

        Effect parameters like `cycles` are absolute, but section lengths are
        not: the same `cycles=1.5` that sweeps nicely across two seconds is
        motionless across thirty. Deriving the count from how many bars the
        effect spans keeps the *rate* constant, so a wash moves at the same
        musical speed whether it covers one bar or a whole drop.
        """
        bars = max((end - start) / max(self.f.bar_period, 1e-6), 1.0)
        return round(max(0.25, bars * per_bar), 2)

    def _phrases(self, sec: Section, bars_per: int = 2) -> list[tuple[float, float]]:
        """Split a section into phrase-length spans on bar lines.

        Sustained effects get chunked over these rather than spanning the whole
        section. A single effect covering thirty seconds reads as a still image
        no matter how energetic its parameters are.
        """
        bars = list(self.f.bars_between(sec.start, sec.end))
        if len(bars) < 2:
            return [(sec.start, sec.end)]
        edges = [float(b) for b in bars[::bars_per]] + [sec.end]
        return [(a, b) for a, b in zip(edges, edges[1:]) if b - a > 0.2]

    def _calm_look(self, fade_in: float = 1.0, fade_out: float = 0.0):
        """A slow, organic net texture for intros, breakdowns and outros.

        Quiet sections used one fixed effect each, so every intro in every
        sequence looked the same. These all move slowly enough to stay restful.
        """
        pick = self.rng.choice(("twinkle", "galaxy", "tendril", "plasma", "fire"))
        if pick == "twinkle":
            return fx.twinkle(count=self.rng.randint(20, 34), steps=36,
                              fade_in=fade_in, fade_out=fade_out)
        if pick == "galaxy":
            return fx.galaxy(revolutions=720, duration=95, end_radius=85,
                             start_width=10, end_width=40,
                             fade_in=fade_in, fade_out=fade_out)
        if pick == "tendril":
            return fx.build("Tendril", fade_in=fade_in, fade_out=fade_out,
                            Tendril_Movement=self.rng.choice(
                                ("Circle", "Random", "Vertical Zig Zag")),
                            Tendril_Speed=self.rng.randint(2, 5),
                            Tendril_Length=self.rng.randint(40, 80),
                            Tendril_Thickness=self.rng.randint(2, 5),
                            Tendril_Trails=self.rng.randint(1, 4))
        if pick == "plasma":
            return fx.build("Plasma", fade_in=fade_in, fade_out=fade_out,
                            Plasma_Style=self.rng.randint(1, 10),
                            Plasma_Speed=self.rng.randint(4, 14),
                            Plasma_Line_Density=self.rng.randint(1, 3))
        return fx.build("Fire", fade_in=fade_in, fade_out=fade_out,
                        Fire_Height=self.rng.randint(25, 55),
                        Fire_HueShift=self.rng.randint(0, 100),
                        Fire_Location=self.rng.choice(("Bottom", "Top")))

    def _ramp_curve(self, sec: Section, bars) -> list[float]:
        """Per-bar 0-1 intensity ramp for a build.

        Two things this fixes over a plain `i / len(bars)`:

        * That expression never reaches 1.0 -- with 3 bars it stops at 0.67 --
          so the last bar before the drop, the one the whole section exists to
          set up, only ever got two thirds of the intended intensity.
        * A linear ramp ignores the shape of the actual riser. Weighting the
          song's own energy envelope in means a build that holds flat and then
          jumps is lit that way, instead of being lit as a straight line
          regardless of what the music does.

        Forced monotonic and rescaled to peak at 1.0: a build that visibly backs
        off part way through reads as a mistake even when the audio does dip.
        """
        n = len(bars)
        if n <= 1:
            return [1.0]

        energy = [
            self.f.mean_over(self.f.rms, float(t),
                             float(bars[i + 1]) if i + 1 < n else sec.end)
            for i, t in enumerate(bars)
        ]
        lo, hi = min(energy), max(energy)
        shape = ([(v - lo) / (hi - lo) for v in energy] if hi - lo > 1e-6
                 else [i / (n - 1) for i in range(n)])

        out, peak = [], 0.0
        for i, s in enumerate(shape):
            peak = max(peak, 0.5 * (i / (n - 1)) + 0.5 * s)
            out.append(peak)
        top = max(out) or 1.0
        return [v / top for v in out]

    def _tail(self, sec: Section, nxt, default: float = 0.15) -> float:
        """Fade-out for a section's sustained effect.

        The excerpt window can land mid-drop, so the last section is not
        necessarily an outro. Without this the show ends by slamming to black.
        Blending is additive, so a darkening overlay would not work -- the fade
        has to live on the effect that is actually running.
        """
        if nxt is not None:
            return default
        return min(2.0, max(default, sec.duration * 0.5))

    # -- corridor -----------------------------------------------------------

    def _wave(self, start: float, travel: float, palette: Palette,
              *, hold: float | None = None, brightness: int = 100,
              reverse: bool | None = None, layer: int = BASE,
              far_palette: Palette | None = None) -> None:
        """Launch one light wave down the corridor.

        Each arch gets its own short effect, offset by its position in the
        Tunnel group's model order -- which is the physical front-to-back
        order, and is why the arch list is read from the group rather than
        generated from a number.

        `far_palette` paints a colour gradient along the corridor's depth: the
        near arches sit at `palette`, the far ones at `far_palette`, blended in
        between. That gradient is fixed to physical position, not to the wave's
        direction, so the tunnel keeps the same near/far colouring whichever way
        a wave travels through it.
        """
        n = len(self.arches)
        travel = max(travel, MIN_TRAVEL_S)
        stagger = travel / n
        hold = hold if hold is not None else max(stagger * 3.0, 0.15)

        if reverse is None:
            # Alternate direction so repeated waves do not read as a loop.
            reverse = bool(self._wave_dir % 2)
            self._wave_dir += 1

        ramp = (pal.gradient(palette, far_palette, n) if far_palette
                else [palette] * n)

        order = list(reversed(self.arches)) if reverse else self.arches
        for i, arch in enumerate(order):
            t0 = start + i * stagger
            if t0 >= self.b.duration_s:
                break
            self.b.add(
                arch,
                fx.on(brightness, fade_in=hold * 0.25, fade_out=hold * 0.6),
                # Index by physical position so the gradient stays put.
                ramp[self._arch_pos[arch]], t0, t0 + hold, layer=layer,
            )

    # -- corridor pattern library -------------------------------------------
    #
    # The corridor is the most expressive surface in the rig, and a single
    # travelling comet repeated for 90 seconds reads as one idea no matter how
    # well it tracks the music. Each pattern below is a different way of
    # distributing light across the 24 arches over one phrase; the arranger
    # picks a pattern per phrase so the tunnel keeps introducing new material
    # while staying locked to the same beat grid.

    def _ramp(self, palette: Palette, far: Palette | None) -> list[Palette]:
        """Per-arch palettes, so any pattern can carry the depth gradient."""
        n = len(self.arches)
        return pal.gradient(palette, far, n) if far else [palette] * n

    def _lit(self, pos: int, t0: float, dur: float, ramp: list[Palette],
             brightness: int, layer: int = BASE) -> None:
        """Light arch at physical position `pos`."""
        if not (0 <= pos < len(self.arches)) or t0 >= self.b.duration_s:
            return
        self.b.add(self.arches[pos],
                   fx.on(brightness, fade_in=dur * 0.2, fade_out=dur * 0.55),
                   ramp[pos], t0, t0 + dur, layer=layer)

    def _pat_comet(self, start, span, ramp, brightness, reverse=False):
        """One head of light running the length of the corridor."""
        n = len(self.arches)
        step = max(span / n, self.b.frame_ms / 1000.0)
        # Hold each arch well past the next one's start so the lit arches
        # overlap. Without the overlap the corridor reads as 24 things blinking
        # in sequence rather than as one band travelling through it.
        hold = max(step * 5.0, 0.3)
        for i in range(n):
            pos = (n - 1 - i) if reverse else i
            self._lit(pos, start + i * step, hold, ramp, brightness)

    def _pat_bounce(self, start, span, ramp, brightness, reverse=False):
        """Down the corridor and back inside one phrase."""
        half = span / 2.0
        self._pat_comet(start, half, ramp, brightness, reverse=reverse)
        self._pat_comet(start + half, half, ramp, brightness, reverse=not reverse)

    def _pat_converge(self, start, span, ramp, brightness, reverse=False):
        """Both ends run inward and meet in the middle."""
        n = len(self.arches)
        step = max(span / (n / 2), self.b.frame_ms / 1000.0)
        hold = max(step * 2.5, 0.15)
        for i in range(n // 2):
            t = start + i * step
            self._lit(i, t, hold, ramp, brightness)
            self._lit(n - 1 - i, t, hold, ramp, brightness)

    def _pat_diverge(self, start, span, ramp, brightness, reverse=False):
        """Opens from the middle outward -- reads as the tunnel splitting."""
        n = len(self.arches)
        step = max(span / (n / 2), self.b.frame_ms / 1000.0)
        hold = max(step * 2.5, 0.15)
        mid = n // 2
        for i in range(mid):
            t = start + i * step
            self._lit(mid - 1 - i, t, hold, ramp, brightness)
            self._lit(mid + i, t, hold, ramp, brightness)

    def _pat_alternate(self, start, span, ramp, brightness, reverse=False):
        """Odd and even arches trade places on the beat."""
        n = len(self.arches)
        grid = self.f.beat_grid(start, start + span)[::2]
        hold = self.f.beat_period * 1.8
        for s, t in enumerate(grid):
            for pos in range(s % 2, n, 2):
                self._lit(pos, float(t), hold, ramp, brightness)

    def _pat_pairs(self, start, span, ramp, brightness, reverse=False):
        """Blocks of arches step down the corridor -- chunkier than a comet."""
        n, size = len(self.arches), 4
        groups = list(range(0, n, size))
        step = max(span / len(groups), self.b.frame_ms / 1000.0)
        for i, g in enumerate(groups):
            t = start + (len(groups) - 1 - i if reverse else i) * step
            for pos in range(g, min(g + size, n)):
                self._lit(pos, t, step * 1.5, ramp, brightness)

    def _pat_strobe(self, start, span, ramp, brightness, reverse=False):
        """Whole corridor pulses as one, on the beat."""
        # Every second beat, taken off the real grid. Flashing all 24 arches on
        # every beat for a whole phrase stops reading as emphasis and just
        # becomes visual noise.
        grid = self.f.beat_grid(start, start + span)[::2]
        hold = self.f.beat_period * 0.9
        for t in grid:
            for pos in range(len(self.arches)):
                self._lit(pos, float(t), hold, ramp, brightness)

    def _pat_sparkle(self, start, span, ramp, brightness, reverse=False):
        """Scattered arches blink at random.

        This is what keeps a quiet passage alive. An intro should be low and
        slow, but low is not the same as dark -- with nothing moving at all the
        rig looks powered off rather than restrained.
        """
        n = len(self.arches)
        beat = self.f.beat_period
        for t in self.f.beat_grid(start, start + span):
            for pos in self.rng.sample(range(n), k=max(1, n // 8)):
                jitter = self.rng.uniform(0, beat * 0.5)
                self._lit(pos, float(t) + jitter, beat * 0.9, ramp, brightness)

    #: Roughly how much light each pattern puts in the corridor per unit time,
    #: 0-1. Selection is weighted by this so the corridor's business tracks the
    #: music: picking uniformly at random lets a drop draw a sparse comet while
    #: a verse draws a full strobe, which inverts the song's dynamics.
    DENSITY = {
        "sparkle": 0.20,
        "comet": 0.30,
        "converge": 0.42,
        "diverge": 0.42,
        "bounce": 0.55,
        "pairs": 0.72,
        "alternate": 0.90,
        "strobe": 1.00,
    }

    #: Target corridor density per section kind.
    TARGET = {
        "intro": 0.22, "break": 0.25, "outro": 0.22,
        "verse": 0.52, "build": 0.62, "drop": 0.78,
    }

    def _vocabulary(self, sec: Section, quiet: bool) -> list[str]:
        """Patterns whose density suits this section, nearest first."""
        target = self.TARGET.get(sec.kind, 0.5)
        names = list(self.DENSITY)
        if quiet:
            # Never strobe the corridor during a quiet passage, however the
            # numbers fall out.
            names = [n for n in names if self.DENSITY[n] <= 0.45]
        names.sort(key=lambda n: abs(self.DENSITY[n] - target))
        # Keep the closest few so there is still variety inside the band.
        return names[:3]

    def _corridor(self, sec: Section, palette: Palette, *, brightness: int = 90,
                  far: Palette | None = None, phrase_bars: int = 4,
                  quiet: bool = False) -> None:
        """Fill a section with corridor patterns, one per phrase.

        Splitting on phrase boundaries is what turns the tunnel into a sequence
        of ideas rather than one looping gesture: each phrase gets its own
        pattern, chosen without repeating the previous one.
        """
        ramp = self._ramp(palette, far)
        vocabulary = self._vocabulary(sec, quiet)

        bars = list(self.f.bars_between(sec.start, sec.end))
        if len(bars) < 2:
            phrases = [(sec.start, sec.end)]
        else:
            edges = bars[::phrase_bars] + [sec.end]
            phrases = [(float(a), float(b)) for a, b in zip(edges, edges[1:])
                       if b - a > 0.2]

        # How long one traversal of the corridor takes, in bars. Tying this to
        # the bar grid instead of to the phrase length is what keeps the tunnel
        # moving at a musical rate: stretching a single gesture across a whole
        # 4-bar phrase means one comet crawls the length of the corridor over
        # eight seconds, which reads as the rig being slow no matter how well
        # the timings line up with the beat.
        # Scale the traversal length by what the section is doing, not just by
        # the track average. On a full song the global figure is dominated by
        # whatever the track spends most of its time on -- a nine-minute piece
        # with a two-minute ambient opening averages out calm, and its drops
        # then inherit that calm despite being the loudest thing in the show.
        gesture = (self.f.bar_period
                   * self.style.gesture_bars
                   * GESTURE_SCALE.get(sec.kind, 1.0))

        for start, end in phrases:
            choices = [p for p in vocabulary if p != self._last_pattern] or list(vocabulary)
            name = self.rng.choice(choices)
            self._last_pattern = name
            fn = getattr(self, f"_pat_{name}")

            # Repeat the phrase's pattern to fill it, so the phrase still reads
            # as one idea while the idea itself recurs on the bar.
            t = start
            while t < end - 0.05:
                span = min(gesture, end - t)
                if span < self.b.frame_ms / 1000.0 * 4:
                    break
                fn(t, span, ramp, brightness, reverse=bool(self._wave_dir % 2))
                self._wave_dir += 1
                t += span

    def _quantize_hits(self, times, sec: Section, *, per_beat: int = 1) -> list[float]:
        """Snap hit times onto the beat grid, at most one per subdivision.

        Accent placement has to be driven by the tempo, not by how many onsets
        the detector happens to find. Firing on raw onsets means a densely
        percussive passage strobes far faster than a sparse one at the identical
        BPM, which reads as the show arbitrarily speeding up. Snapping to
        beats/`per_beat` and keeping one hit per slot keeps the pulse locked to
        the music while still letting the *audio* decide which slots fire.
        """
        grid = self.f.beat_grid(sec.start, sec.end, per_beat)
        if len(grid) < 2:
            return [float(t) for t in times]

        # Snap to the nearest *real* grid point and keep one hit per slot.
        # Stepping off a constant period instead drifts away from the music as
        # the section goes on -- see Features.beat_grid.
        slots: dict[int, float] = {}
        for t in times:
            i = int(np.argmin(np.abs(grid - float(t))))
            slots.setdefault(i, float(grid[i]))
        return sorted(slots.values())

    def _wave_times(self, sec: Section, per_bar: float) -> list[float]:
        """When to launch waves inside a section."""
        bars = self.f.bars_between(sec.start, sec.end)
        if len(bars) == 0:
            return [sec.start]
        step = max(1, int(round(1.0 / (per_bar * self.style.wave_density))))
        return [float(t) for t in bars[::step]]

    # -- section recipes ----------------------------------------------------

    def _quiet(self, sec: Section, nxt, *, dim: float, brighten: float,
               dj_low: int, dj_high: int, corridor: int,
               phrase_bars: int = 2) -> None:
        """Shared treatment for intros, breakdowns and outros.

        Chunked per phrase, and the treatment *develops* across the section:
        the palette lifts, the nets change texture, the par breathes.

        One sustained effect per section was adequate while sections were ten or
        twenty seconds long. In full-song mode they are not -- a track with a
        long ambient opening produces a 72-second intro, and a single effect
        stretched across it leaves the nets and the par each showing exactly one
        thing for over a minute, which is most of the reason a long intro reads
        as dark and dead rather than as restrained.
        """
        phrases = self._phrases(sec, bars_per=phrase_bars)
        n = max(len(phrases), 1)

        for i, (t0, t1) in enumerate(phrases):
            frac = i / max(n - 1, 1)
            first, last = i == 0, i == n - 1
            # Lift gently across the section so a long quiet passage still has
            # somewhere to go.
            p = self._palette(sec, dim=dim * (1.0 + brighten * frac))
            fade_in = min(2.0, sec.duration * 0.3) if first else 0.0
            fade_out = self._tail(sec, nxt, 0.6) if last else 0.0

            self.b.add(self.show.tunnel,
                       fx.color_wash(cycles=self._cycles(t0, t1, 0.5),
                                     vfade=True, fade_in=fade_in,
                                     fade_out=fade_out),
                       p, t0, t1)
            # A fresh texture each phrase -- _calm_look picks from a pool, so
            # consecutive phrases differ without ever getting busy.
            self.b.add(self._all,
                       self._calm_look(fade_in=fade_in, fade_out=fade_out),
                       p, t0, t1)

            # The par breathes on the bar instead of holding one level.
            level = int(dj_low + (dj_high - dj_low) * frac)
            bars = self.f.bars_between(t0, t1)
            if len(bars) == 0:
                self.b.add(self.show.dj, fx.on(level, fade_in=0.6, fade_out=0.8),
                           p, t0, t1)
            else:
                for k, bt in enumerate(bars[::2]):
                    end = min(float(bt) + self.f.bar_period * 1.6, t1)
                    span = end - float(bt)
                    self.b.add(self.show.dj,
                               fx.on(level, end_brightness=max(8, level // 3),
                                     fade_in=span * 0.25, fade_out=span * 0.5),
                               p, float(bt), end)

            self._corridor(
                Section(t0, t1, sec.kind, sec.energy, sec.brightness),
                p, brightness=corridor, far=self._depth(p),
                quiet=True, phrase_bars=phrase_bars,
            )

    def _intro(self, sec: Section, prev, nxt) -> None:
        self._quiet(sec, nxt, dim=0.62, brighten=0.45,
                    dj_low=30, dj_high=70, corridor=80, phrase_bars=2)

    def _break(self, sec: Section, prev, nxt) -> None:
        self._quiet(sec, nxt, dim=0.58, brighten=0.30,
                    dj_low=35, dj_high=60, corridor=75, phrase_bars=2)

    def _verse(self, sec: Section, prev, nxt) -> None:
        p = self._palette(sec)
        # Chunked on a 4-bar phrase -- gentler than the drop's 2, but a verse
        # can still run long enough for one held wash to go dead.
        verse_phrases = self._phrases(sec, bars_per=4)
        for i, (t0, t1) in enumerate(verse_phrases):
            last = i == len(verse_phrases) - 1
            self.b.add(self.show.tunnel,
                       fx.color_wash(cycles=self._cycles(t0, t1, 0.8),
                                     fade_in=0.4 if i == 0 else 0.0,
                                     fade_out=(self._tail(sec, nxt, 0.4)
                                               if last else 0.0)),
                       p, t0, t1)

        bars = self.f.bars_between(sec.start, sec.end)
        for i, t in enumerate(bars):
            end = float(bars[i + 1]) if i + 1 < len(bars) else sec.end
            self.b.add(self._big,
                       fx.bars(count=2, direction="up", cycles=1.0, fade_out=0.1),
                       p, float(t), end)
            # The small nets answer the big ones a beat later, going the other
            # way. Anchored to the next *detected* beat rather than offset by a
            # nominal period, so the answer stays a beat behind even where the
            # tempo moves.
            answer = self.f.snap(float(t) + self.f.beat_period, sec.start, sec.end)
            self.b.add(self._small,
                       fx.bars(count=2, direction="down", cycles=1.0, fade_out=0.1),
                       p, answer, max(end, answer + self.f.beat_period))

        self._corridor(sec, p, brightness=90, far=self._depth(p),
                       phrase_bars=4)

        # Beat-level articulation.
        #
        # Everything above moves on the *bar*: a wash chunked per phrase, one
        # Bars effect per bar, one par pulse per downbeat. On a track whose
        # pulse is right at the front -- a steady four-on-the-floor -- that
        # comes out as smooth glowing that happens to change colour, because
        # nothing lands on the beat the listener is actually feeling. Bar-level
        # motion is only enough when the beat itself is understated.
        #
        # So punch on the beat in proportion to how hard the track states one,
        # alternating between the big and small nets so it reads as a bounce
        # passing back and forth rather than as everything strobing together.
        pulse = self.f.pulse_clarity
        every = 1 if pulse >= 0.70 else (2 if pulse >= 0.45 else 0)
        if every:
            # Onset-driven accents would land on the same layer and mostly the
            # same instants, so they get suppressed here: two systems writing
            # near-identical hits just means one truncates the other, and the
            # regular pulse is the one this section is built around.
            self._articulated.add(id(sec))
            # A colour flip, not a same-hue flash: the punch sits opposite the
            # bed on the wheel so each beat reads as a change, not a flicker.
            punch = p.rotated(180.0).floored(220)
            hit = min(self.f.beat_period * 0.42, 0.30)
            for i, t in enumerate(self.f.beat_grid(sec.start, sec.end)[::every]):
                target = self._big if i % 2 == 0 else self._small
                # Hard attack, half-length decay. Fade-in plus fade-out must
                # stay well inside the hit, or the two overlap and the punch
                # never reaches full brightness -- it swells instead of hits.
                self.b.add(target,
                           fx.on(100, fade_in=0.0, fade_out=hit * 0.5),
                           punch, float(t), float(t) + hit, layer=ACCENT)

        # The par follows the same pulse: every beat when the track insists on
        # one, otherwise the downbeat.
        if every == 1:
            for t in self.f.beat_grid(sec.start, sec.end):
                hold = self.f.beat_period * 0.55
                self.b.add(self.show.dj, fx.on(90, fade_out=hold * 0.6), p,
                           float(t), float(t) + hold)
        else:
            for t in bars:
                self.b.add(self.show.dj, fx.on(90, fade_out=0.25), p,
                           float(t), float(t) + self.f.beat_period * 0.8)

    def _build(self, sec: Section, prev, nxt) -> None:
        p = self._palette(sec)
        bars = self.f.bars_between(sec.start, sec.end)

        # Chunked per phrase like the drop's bed. A build can run for a minute
        # on a track with a long riser, and one effect stretched across all of
        # it is the single most static thing in the sequence however high its
        # cycle count -- same palette, same effect, no event to mark the bars.
        bed = p.dimmed(0.7).floored(FLOOR.get("build", 90))
        bed_phrases = self._phrases(sec, bars_per=2)
        for i, (t0, t1) in enumerate(bed_phrases):
            last = i == len(bed_phrases) - 1
            self.b.add(self.show.tunnel,
                       fx.color_wash(cycles=self._cycles(t0, t1, 1.0),
                                     shimmer=True,
                                     fade_in=0.3 if i == 0 else 0.0,
                                     fade_out=(self._tail(sec, nxt, 0.1)
                                               if last else 0.0)),
                       bed, t0, t1)

        # Everything ramps together: pinwheel speeds up, waves get shorter and
        # more frequent, the par pulses twice as often each bar. The curve
        # follows the track's own riser rather than a straight line.
        curve = self._ramp_curve(sec, bars)
        for i, t in enumerate(bars):
            end = float(bars[i + 1]) if i + 1 < len(bars) else sec.end
            frac = curve[i]
            self.b.add(self._all,
                       fx.pinwheel(arms=3 + int(frac * 4),
                                   speed=6 + int(frac * 22),
                                   thickness=45 - int(frac * 25),
                                   twist=int(frac * 40)),
                       p, float(t), end)

            # Small nets strobe against the big ones, tightening as it rises.
            self.b.add(self._small,
                       fx.twinkle(count=20 + int(frac * 60),
                                  steps=max(4, 30 - int(frac * 24)),
                                  strobe=frac > 0.5),
                       p, float(t), end)

            travel = max(MIN_TRAVEL_S, self.f.bar_period * (1.0 - 0.55 * frac))
            self._wave(float(t), travel=travel, palette=p,
                       brightness=70 + int(frac * 30),
                       # Gradient flattens as the build peaks, so the corridor
                       # reads as one solid mass by the time the drop lands.
                       far_palette=self._depth(p, 0.3 + 0.6 * frac))

            # Pulse subdivision doubles as the build progresses.
            div = 1 if frac < 0.4 else (2 if frac < 0.75 else 4)
            step = (end - float(t)) / (4 * div)
            for k in range(4 * div):
                t0 = float(t) + k * step
                self.b.add(self.show.dj,
                           fx.on(40 + int(frac * 60), fade_out=step * 0.6),
                           p, t0, t0 + step * 0.9)

    def _drop(self, sec: Section, prev, nxt) -> None:
        p = self._palette(sec)
        # A hue-shifted sibling of the section palette: related enough to read
        # as the same moment, different enough that the corridor and the nets
        # are not painted in one flat colour.
        hot = self._palette(sec, hue_shift=30.0)

        # Chunked per phrase rather than one effect across the whole drop, and
        # the cycle count follows the span so the wash keeps sweeping.
        bed = p.dimmed(0.6).floored(FLOOR.get("drop", 120))
        phrases = self._phrases(sec, bars_per=2)
        for i, (t0, t1) in enumerate(phrases):
            last = i == len(phrases) - 1
            self.b.add(self.show.tunnel,
                       fx.color_wash(cycles=self._cycles(t0, t1, per_bar=1.0),
                                     fade_in=0.15 if i == 0 else 0.0,
                                     fade_out=self._tail(sec, nxt) if last else 0.0),
                       bed, t0, t1)

        # Rotate the net pattern every two bars. Four bars is long enough on the
        # biggest surface in the rig to read as a held image rather than a
        # moving one, which is what makes a drop feel static even while the
        # accents underneath are firing on every beat.
        #
        # The speed parameters are pushed up to match: these effects animate
        # across their own duration, so the same settings that look lively over
        # two bars crawl over eight.
        looks = [
            lambda: fx.butterfly(style=self.rng.choice([1, 2, 3, 5]),
                                 chunks=self.rng.randint(1, 3),
                                 skip=self.rng.randint(1, 3), speed=38),
            lambda: fx.spirals(count=self.rng.randint(2, 4), movement=24.0,
                               thickness=18, rotation=60),
            lambda: fx.fan(blades=self.rng.randint(3, 6), revolutions=2160,
                           duration=45, end_radius=95),
            lambda: fx.pinwheel(arms=self.rng.randint(3, 6), speed=42,
                                thickness=28, twist=30),
            lambda: fx.build("Plasma", Plasma_Style=self.rng.randint(1, 10),
                             Plasma_Speed=self.rng.randint(30, 70),
                             Plasma_Line_Density=self.rng.randint(1, 4)),
            lambda: fx.build("Kaleidoscope",
                             Kaleidoscope_Type=self.rng.choice(
                                 ["Triangle", "6-Fold", "8-Fold", "Radial"]),
                             Kaleidoscope_Size=self.rng.randint(8, 30),
                             Kaleidoscope_Rotation=self.rng.randint(0, 359)),
            lambda: fx.build("Warp",
                             Warp_Type=self.rng.choice(
                                 ["ripple", "banded swirl", "water drops"]),
                             Warp_Speed=self.rng.randint(22, 38),
                             Warp_Frequency=self.rng.randint(14, 30)),
            lambda: fx.build("Circles", Circles_Count=self.rng.randint(3, 8),
                             Circles_Size=self.rng.randint(4, 12),
                             Circles_Speed=self.rng.randint(14, 28),
                             Circles_Bounce=True),
            lambda: fx.build("Spirograph", Spirograph_R=self.rng.randint(20, 60),
                             Spirograph_r=self.rng.randint(5, 30),
                             Spirograph_d=self.rng.randint(10, 60),
                             Spirograph_Speed=self.rng.randint(20, 45),
                             Spirograph_Animate=self.rng.randint(10, 40)),
        ]
        # Shuffle so a long drop does not always walk the list in the same
        # order, while a fixed seed keeps the whole thing reproducible.
        self.rng.shuffle(looks)
        net_phrases = self._phrases(sec, bars_per=2)
        for i, (t0, t1) in enumerate(net_phrases):
            look = looks[i % len(looks)]()
            if nxt is None and i == len(net_phrases) - 1:
                look = fx.with_fade(look, fade_out=self._tail(sec, nxt))
            self.b.add(self._all, look, hot, t0, t1)

        # Small nets punch on the offbeat so the drop has internal rhythm
        # rather than one continuous texture.
        # Real half-beat grid: every second point is an offbeat, and each one
        # sits between its own two detected beats rather than a fixed distance
        # from a nominal one.
        halves = self.f.beat_grid(sec.start, sec.end, per_beat=2)
        for t in halves[1::2]:
            self.b.add(self._small,
                       fx.bars(count=1, direction="Alternate Up", cycles=1.0,
                               fade_out=0.08),
                       hot, float(t), float(t) + self.f.beat_period * 0.4,
                       layer=ACCENT)

        # Corridor waves every two beats, plus a shockwave on the section hit.
        self._corridor(sec, hot, brightness=100, far=self._depth(hot, 0.55),
                       phrase_bars=4)

        self.b.add(self._all,
                   fx.shockwave(start_radius=0, end_radius=110, start_width=4,
                                end_width=22, cycles=1, accel=4, fade_out=0.2),
                   pal.WHITE, sec.start, sec.start + min(1.2, sec.duration),
                   layer=ACCENT)

        # DJ hits the kicks, but never faster than one per beat.
        for t in self._quantize_hits(self.f.kicks_between(sec.start, sec.end),
                                     sec, per_beat=1):
            self.b.add(self.show.dj, fx.on(100, fade_out=0.12), hot,
                       t, t + min(0.2, self.f.beat_period * 0.5))

    def _outro(self, sec: Section, prev, nxt) -> None:
        # Negative `brighten` so the section fades out across its phrases
        # rather than holding level and then dropping at the very end.
        # A gentle settle, not a blackout: an excerpt's outro is often just
        # where the window happens to end, with the music still playing.
        self._quiet(sec, nxt, dim=0.72, brighten=-0.15,
                    dj_low=55, dj_high=30, corridor=72, phrase_bars=4)

    # -- accents ------------------------------------------------------------

    def _accents(self) -> None:
        """Short hits on the strongest percussive onsets.

        These go on layer 1 so the sustained effect underneath keeps running --
        the same two-layer shape the hand-made sequence in this show uses.
        """
        if not len(self.f.kick_times):
            return

        # Group the surviving hits per section so they can be snapped to that
        # section's beat grid.
        keep: dict[int, list[float]] = {}
        for t, strength in zip(self.f.kick_times, self.f.kick_strength):
            if t >= self.b.duration_s:
                break
            sec = self._section_at(float(t))
            if sec is None or sec.kind in ("intro", "outro", "break"):
                continue
            if id(sec) in self._articulated:
                continue

            threshold = self.style.accent_threshold
            if sec.kind != "drop":
                threshold += 0.15
            if strength < threshold:
                continue
            if self.rng.random() > self.style.accent_density:
                continue
            keep.setdefault(self.sections.index(sec), []).append(float(t))

        for idx, times in keep.items():
            sec = self.sections[idx]
            # Half-beat resolution in a drop, one per beat elsewhere.
            per_beat = 2 if sec.kind == "drop" else 1
            dur = 0.15 if sec.kind == "drop" else 0.2
            for t in self._quantize_hits(times, sec, per_beat=per_beat):
                self.b.add(self._big, fx.on(100, fade_out=dur * 0.8), pal.WHITE,
                           t, t + dur, layer=ACCENT)

    def _section_at(self, t: float) -> Section | None:
        for s in self.sections:
            if s.start <= t < s.end:
                return s
        return None


def summarize(sections: list[Section]) -> str:
    return " | ".join(f"{s.kind}:{s.duration:.0f}s" for s in sections)
