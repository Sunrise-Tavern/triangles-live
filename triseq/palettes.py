"""Color palettes, serialized in xLights' ColorPalette format.

An xLights palette is 8 color slots plus a checkbox per slot saying whether that
slot is active.  Effects consume however many active colors they need.  The
serialized form is comma-separated ``key=value`` pairs sorted by key -- matching
what xLights itself writes.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_COLORS = 8


@dataclass(frozen=True)
class Palette:
    """An ordered list of hex colors, up to 8."""

    name: str
    colors: tuple[str, ...]
    #: Sparkle density 0-200; shows up as white glints on top of the effect.
    sparkles: int = 0

    def serialize(self) -> str:
        if not self.colors:
            raise ValueError(f"Palette {self.name!r} has no colors")
        if len(self.colors) > MAX_COLORS:
            raise ValueError(f"Palette {self.name!r} has more than {MAX_COLORS} colors")

        settings: dict[str, str] = {
            "C_CHECKBOXBRIGHTNESSLEVEL": "0",
            "C_CHECKBOX_Chroma": "0",
            "C_CHECKBOX_MusicSparkles": "0",
            "C_SLIDER_SparkleFrequency": str(self.sparkles),
        }
        for i in range(MAX_COLORS):
            slot = i + 1
            active = i < len(self.colors)
            settings[f"C_BUTTON_Palette{slot}"] = self.colors[i] if active else "#000000"
            settings[f"C_CHECKBOX_Palette{slot}"] = "1" if active else "0"

        return ",".join(f"{k}={settings[k]}" for k in sorted(settings))

    def floored(self, level: int = 48) -> "Palette":
        """Raise the palette until its brightest colour reaches `level`.

        A colour can be arithmetically dim and still be *off* as far as the eye
        is concerned: an LED at 12/255 reads as black at any distance. Quiet
        sections still need to be visible, so every palette gets scaled up until
        something in it clears this floor. Hue and the relative balance between
        slots are preserved, so the palette still looks dim next to a drop.
        """
        peak = max((max(_rgb(c)) for c in self.colors), default=0)
        if peak >= level or peak == 0:
            return self
        return Palette(
            name=f"{self.name}^{level}",
            colors=tuple(_scale(c, level / peak) for c in self.colors),
            sparkles=self.sparkles,
        )

    def rotated(self, degrees: float) -> "Palette":
        """Shift every colour's hue, keeping saturation and value.

        Lets the far end of the corridor sit at a *different colour* from the
        near end rather than merely a darker one, so the tunnel reads as a
        gradient through hue -- cyan at the mouth to violet at the back.
        """
        import colorsys

        out = []
        for c in self.colors:
            r, g, b = (v / 255.0 for v in _rgb(c))
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            out.append(hsv_hex(h * 360.0 + degrees, s, v))
        return Palette(f"{self.name}+{degrees:.0f}", tuple(out), self.sparkles)

    def dimmed(self, factor: float) -> "Palette":
        """A copy with every color scaled toward black.

        Used for breakdowns and outros, where we want the same hues at lower
        intensity rather than a different palette entirely.
        """
        return Palette(
            name=f"{self.name} x{factor:.2f}",
            colors=tuple(_scale(c, factor) for c in self.colors),
            sparkles=self.sparkles,
        )


def blend(a: "Palette", b: "Palette", t: float) -> "Palette":
    """Mix two palettes. t=0 gives `a`, t=1 gives `b`.

    Used to paint a gradient *across* fixtures: give each arch in the corridor
    its own blended palette and the tunnel shows a colour transition along its
    depth at a single instant, rather than every arch being the same colour.
    """
    t = max(0.0, min(1.0, t))
    n = min(len(a.colors), len(b.colors))
    return Palette(
        name=f"{a.name}->{b.name}@{t:.2f}",
        colors=tuple(_mix(a.colors[i], b.colors[i], t) for i in range(n)),
        sparkles=int(round(a.sparkles + (b.sparkles - a.sparkles) * t)),
    )


def gradient(a: "Palette", b: "Palette", n: int) -> list["Palette"]:
    """`n` palettes stepping from `a` to `b`, one per fixture in a run."""
    if n <= 1:
        return [a]
    return [blend(a, b, i / (n - 1)) for i in range(n)]


def _mix(c1: str, c2: str, t: float) -> str:
    a = _rgb(c1)
    b = _rgb(c2)
    return "#%02x%02x%02x" % tuple(
        int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3)
    )


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _scale(hex_color: str, factor: float) -> str:
    scaled = tuple(max(0, min(255, round(c * factor))) for c in _rgb(hex_color))
    return "#%02x%02x%02x" % scaled


def _p(name: str, *colors: str, sparkles: int = 0) -> Palette:
    return Palette(name=name, colors=tuple(colors), sparkles=sparkles)


# ---------------------------------------------------------------------------
# Generative colour.
#
# Palettes are built from a base hue plus a harmony scheme rather than chosen
# from a fixed list. Picking from a short list keyed on one scalar meant any two
# sections with a similar spectral centroid drew the identical palette, so a
# track could spend its entire second half in one colour.
#
# Hue comes from the music (the track's dominant pitch class) and then rotates
# per section, so the show travels through colour space instead of sitting in
# one corner of it.
# ---------------------------------------------------------------------------

#: Hue offsets in degrees from the base. Each scheme has a different amount of
#: internal contrast, which is what makes it suit a particular kind of section.
SCHEMES: dict[str, tuple[float, ...]] = {
    # Neighbouring hues: cohesive and calm. Quiet sections.
    "analogous": (0.0, 25.0, 50.0),
    # A wider fan, still harmonious. Verses.
    "split": (0.0, 150.0, 210.0),
    # Straight opposites: maximum tension. Builds.
    "complementary": (0.0, 180.0, 195.0),
    # Three evenly spaced hues: vivid and busy. Drops.
    "triadic": (0.0, 120.0, 240.0),
    # Cool-to-warm sweep for gradients along the corridor.
    "sweep": (0.0, 40.0, 80.0),
}

#: Rotating the hue by the golden angle between sections gives the longest run
#: before colours start repeating or landing near a previous one.
GOLDEN_ANGLE = 137.508


def hsv_hex(hue: float, sat: float, val: float) -> str:
    """HSV (hue in degrees, sat/val 0-1) to a hex string."""
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(
        (hue % 360.0) / 360.0,
        max(0.0, min(1.0, sat)),
        max(0.0, min(1.0, val)),
    )
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def generate(hue: float, scheme: str = "analogous", *, value: float = 1.0,
             sat: float = 1.0, sparkles: int = 0, white: bool = False,
             name: str | None = None) -> Palette:
    """Build a palette from a base hue and a harmony scheme.

    Slots taper slightly in value and saturation so the palette has internal
    depth rather than three equally-loud colours competing.
    """
    offsets = SCHEMES.get(scheme, SCHEMES["analogous"])
    colors = [
        hsv_hex(hue + off, sat * (1.0 - 0.08 * i), value * (1.0 - 0.10 * i))
        for i, off in enumerate(offsets)
    ]
    if white:
        # A near-white slot gives drops their punch without washing out the hue.
        colors.append(hsv_hex(hue, 0.05, min(1.0, value * 1.05)))
    return Palette(
        name=name or f"{scheme}@{hue % 360:.0f}",
        colors=tuple(colors),
        sparkles=sparkles,
    )


#: A pure-white palette, for strobes and accent hits where hue would muddy it.
WHITE = _p("white", "#ffffff")
#: Black, for hard blackouts.
BLACK = _p("black", "#000000")
