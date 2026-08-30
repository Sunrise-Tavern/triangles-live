"""A fixed 30-second show, written by hand.

M2's verification artefact and M5's skeleton.  It is the live renderer asked to
do everything it can -- every corridor pattern, every net effect, a palette
journey, the par -- against a metronome instead of a beat tracker, so the
*rendering* can be judged before the *listening* exists.

Structure is the offline arranger's, compressed: four four-bar scenes at
128 BPM, intro -> verse -> build -> drop, each rotating the base hue by the
golden angle.  Everything is a pure function of the frame index, so rendering
it twice gives identical bytes and ``xLights --fseqcmp`` stays a usable oracle.

    ./live.sh show --out out/show.fseq
    ./live.sh show --send --host 127.0.0.1      # through DDP into the fake Falcon
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import effects as fx
from . import palette as pal
from .frame import Canvas
from .settings import Settings

BPM = 128.0
BEATS_PER_BAR = 4


@dataclass(frozen=True)
class Scene:
    kind: str
    bars: int
    #: Corridor pattern name and how many phrases of it fit in the scene.
    pattern: str
    phrases: int
    scheme: str
    value: float
    brightness: float


SCENES: tuple[Scene, ...] = (
    Scene("intro",  4, "sparkle",   1, "analogous",     0.55, 0.55),
    Scene("verse",  4, "comet",     2, "split",         0.85, 0.80),
    Scene("build",  4, "pairs",     4, "complementary", 0.95, 0.90),
    Scene("drop",   4, "alternate", 8, "triadic",       1.00, 1.00),
)


class Script:
    """Renders one frame of the fixed show into a canvas."""

    def __init__(self, canvas: Canvas, bpm: float = BPM, base_hue: float = 190.0,
                 seed: int = 7, settings: Settings | None = None) -> None:
        self.canvas = canvas
        self.settings = settings
        self.base_hue = base_hue
        self.seed = seed
        self.big, self.small = canvas.net_pair()
        self._bpm = bpm
        self._palettes: dict[tuple, pal.Palette] = {}

    # -- knobs --------------------------------------------------------------#
    #
    # Read through properties rather than cached, so a slider in the browser
    # takes effect on the next frame without the engine restarting anything.

    @property
    def bpm(self) -> float:
        return self.settings.bpm if self.settings else self._bpm

    @property
    def beat(self) -> float:
        return 60.0 / self.bpm

    @property
    def bar(self) -> float:
        return self.beat * BEATS_PER_BAR

    @property
    def duration(self) -> float:
        return sum(scene.bars for scene in SCENES) * self.bar

    # -- where are we -------------------------------------------------------#

    def locate(self, t: float) -> tuple[int, Scene, float]:
        """Scene index, scene, and 0..1 progress through it."""
        held = self.settings.scene if self.settings else "auto"
        span = SCENES[0].bars * self.bar
        if held != "auto":
            index = [s.kind for s in SCENES].index(held)
            return index, SCENES[index], (t % span) / span
        cursor, t = 0.0, t % self.duration
        for i, scene in enumerate(SCENES):
            span = scene.bars * self.bar
            if t < cursor + span:
                return i, scene, (t - cursor) / span
            cursor += span
        return 0, SCENES[0], 0.0

    def beat_phase(self, t: float) -> float:
        """0 on the beat, rising to 1 just before the next one."""
        return (t % self.beat) / self.beat

    def palette_for(self, index: int, scene: Scene) -> pal.Palette:
        """The scene's palette, with the hue journey and any live offset.

        Memoised because a hue only takes a few distinct values while a slider
        sits still, and generating one allocates.
        """
        offset = self.settings.hue_offset if self.settings else 0.0
        lock = self.settings.hue_lock if self.settings else False
        hue = self.base_hue + offset + (0.0 if lock else index * pal.GOLDEN_ANGLE)
        key = (round(hue, 2), scene.scheme, scene.value, scene.kind)
        cached = self._palettes.get(key)
        if cached is None:
            if len(self._palettes) > 256:       # a dragged slider is unbounded
                self._palettes.clear()
            cached = pal.generate(hue, scene.scheme, value=scene.value,
                                  white=scene.kind == "drop").floored()
            self._palettes[key] = cached
        return cached

    def phrase_index(self, t: float, scene: Scene) -> int:
        """The scripted show has fixed patterns; this exists so the engine can
        treat it and the live arranger the same way."""
        return 0

    def pattern_for(self, scene: Scene, phrase: int = 0) -> str:
        """Which corridor pattern to draw, honouring the live knobs.

        At articulation 0.5 this returns the scene's own pattern, so the
        default show is unchanged; turning it up or down slides the choice
        along the density table the offline arranger uses.
        """
        if self.settings is None:
            return scene.pattern
        if self.settings.pattern != "auto":
            return self.settings.pattern
        target = fx.DENSITY[scene.pattern] + (self.settings.articulation - 0.5)
        return fx.vocabulary(min(max(target, 0.0), 1.0),
                             quiet=scene.kind in ("intro", "break"))[0]

    # -- render -------------------------------------------------------------#

    def render(self, frame: int, t: float) -> None:
        canvas = self.canvas
        canvas.clear()
        index, scene, progress = self.locate(t)
        palette = self.palette_for(index, scene)
        # The far end of the corridor is a hue rotation of the near end, so the
        # tunnel reads as depth rather than as a fade to black.
        far = palette.rotated(70.0)
        beat = self.beat_phase(t)
        kick = _kick(beat)

        rate = self.settings.corridor_rate if self.settings else 1.0
        name = self.pattern_for(scene)
        phrase = (progress * scene.phrases * rate) % 1.0
        kwargs = {"seed": self.seed} if name == "sparkle" else {}
        levels = fx.PATTERNS[name](len(canvas.arch_names), phrase, **kwargs)
        fx.corridor(canvas, levels, palette, far,
                    brightness=scene.brightness, height=0.35)

        getattr(self, f"_{scene.kind}")(frame, t, progress, palette, beat, kick)

    # Each scene is deliberately a different *kind* of load on the renderer, so
    # the 30 s says something about worst-case cost and not just typical cost.

    def _intro(self, frame, t, progress, palette, beat, kick) -> None:
        fx.plasma(self.canvas, palette, t, scale=2.5, speed=0.35, level=0.7)
        fx.net_sparkle(self.canvas, pal.WHITE, frame, density=0.006, level=0.5,
                       seed=self.seed)
        fx.par(self.canvas, palette.color(0), 0.25 + 0.15 * progress)

    def _verse(self, frame, t, progress, palette, beat, kick) -> None:
        fx.wash(self.canvas, palette.dimmed(0.35), 1.0, gradient=0.8)
        # Big and small nets trade bars, as in the offline show.
        bar_index = int(t / self.bar)
        lead = self.big if bar_index % 2 == 0 else self.small
        fx.bars(self.canvas, palette, progress * 4.0, count=3, angle=0.15,
                width=0.3, level=0.9, targets=lead)
        fx.par(self.canvas, palette.color(1), 0.35 + 0.45 * kick)

    def _build(self, frame, t, progress, palette, beat, kick) -> None:
        # Everything accelerates with the build: the pinwheel spins faster and
        # the small nets tighten from a pulse into a strobe.
        rate = 1.0 + 5.0 * progress
        fx.wash(self.canvas, palette.dimmed(0.25), 1.0, gradient=1.0)
        fx.pinwheel(self.canvas, palette, t * rate * 0.35, arms=3, level=0.8,
                    targets=self.big)
        strobe_rate = 2 + int(progress * 6)
        flash = _kick((t / self.beat * strobe_rate) % 1.0, sharp=3.0)
        fx.wash(self.canvas, pal.WHITE, 0.55 * flash * progress, targets=self.small)
        fx.par(self.canvas, palette.color(0), 0.4 + 0.6 * progress * flash,
               white=0.3 * progress)

    def _drop(self, frame, t, progress, palette, beat, kick) -> None:
        fx.radial(self.canvas, palette, (t * 2.0) % 1.0, width=0.3, level=0.9)
        fx.wash(self.canvas, palette.dimmed(0.3), kick, targets=self.big)
        fx.pinwheel(self.canvas, palette, -t * 0.8, arms=5, level=0.6,
                    targets=self.small)
        fx.net_sparkle(self.canvas, pal.WHITE, frame, density=0.02, level=0.9,
                       seed=self.seed)
        fx.par(self.canvas, pal.WHITE.color(0), kick, white=kick)


def _kick(phase: float, sharp: float = 2.0) -> float:
    """A percussive envelope over one beat: instant attack, quick decay."""
    return float(max(0.0, 1.0 - phase) ** sharp)


def frames(canvas: Canvas, fps: float = 40.0, seconds: float | None = None,
           **kwargs):
    """Yield ``(index, channel array)`` for the whole script."""
    script = Script(canvas, **kwargs)
    total = int(round(fps * (script.duration if seconds is None else seconds)))
    out = np.zeros(canvas.layout.channel_count, dtype=np.uint8)
    for i in range(total):
        script.render(i, i / fps)
        yield i, canvas.to_channels(out)
