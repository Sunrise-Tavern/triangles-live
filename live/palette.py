"""Generative colour, in numpy floats instead of xLights hex strings.

Same idea as ``triseq/palettes.py`` -- a base hue plus a harmony scheme, rotated
by the golden angle as the show moves on -- but shaped for a renderer rather
than for an ``.xsq`` file: colours are a ``(k, 3)`` float array in 0..1 and the
useful operation is :meth:`Palette.ramp`, which interpolates the palette across
however many pixels an effect is painting.

The two colour rules the offline show learned the hard way carry over unchanged:

* **Quiet is not dark.**  An LED below roughly 40/255 reads as off at any
  distance, so :meth:`Palette.floored` lifts a dim palette until something in
  it clears the floor, keeping hue and internal balance.
* **A gradient should move through hue, not just brightness.**  The corridor
  runs cyan at the mouth to violet at the back because the far palette is a
  *rotation* of the near one, not a dimming of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Hue offsets in degrees from the base.  Each scheme has a different amount of
#: internal contrast, which is what makes it suit a particular kind of section.
SCHEMES: dict[str, tuple[float, ...]] = {
    "analogous": (0.0, 25.0, 50.0),         # cohesive and calm -- quiet passages
    "split": (0.0, 150.0, 210.0),           # a wide fan, still harmonious
    "complementary": (0.0, 180.0, 195.0),   # straight opposites -- builds
    "triadic": (0.0, 120.0, 240.0),         # vivid and busy -- drops
    "sweep": (0.0, 40.0, 80.0),             # cool-to-warm, for depth gradients
    "tetradic": (0.0, 90.0, 180.0, 270.0),  # four corners of the wheel -- loudest
    "mono": (0.0, 0.0, 0.0),                # one hue in three depths -- calmest
    "accent": (0.0, 15.0, 180.0),           # a close pair with one opposite pop
    "neighbours": (0.0, -30.0, 30.0),       # analogous, spread both ways
}

#: Rotating by the golden angle between sections gives the longest run before
#: hues start repeating or landing near a previous one.
GOLDEN_ANGLE = 137.508

#: Below this, an LED reads as off rather than dim.
FLOOR = 48.0 / 255.0


def hsv_to_rgb(hue: np.ndarray, sat: np.ndarray, val: np.ndarray) -> np.ndarray:
    """Vectorised HSV -> RGB.  ``hue`` in degrees, ``sat``/``val`` in 0..1."""
    h = (np.asarray(hue, dtype=np.float32) % 360.0) / 60.0
    s = np.clip(np.asarray(sat, dtype=np.float32), 0.0, 1.0)
    v = np.clip(np.asarray(val, dtype=np.float32), 0.0, 1.0)
    i = np.floor(h).astype(np.int32) % 6
    f = h - np.floor(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    table = np.stack([
        np.stack([v, t, p], axis=-1), np.stack([q, v, p], axis=-1),
        np.stack([p, v, t], axis=-1), np.stack([p, q, v], axis=-1),
        np.stack([t, p, v], axis=-1), np.stack([v, p, q], axis=-1),
    ])
    return np.take_along_axis(table, i[None, ..., None], axis=0)[0].astype(np.float32)


@dataclass(frozen=True)
class Palette:
    name: str
    colors: np.ndarray          # (k, 3) float32 in 0..1

    def __post_init__(self) -> None:
        if self.colors.ndim != 2 or self.colors.shape[1] != 3:
            raise ValueError("palette colours must be (k, 3)")

    def __len__(self) -> int:
        return len(self.colors)

    def color(self, index: int) -> np.ndarray:
        return self.colors[index % len(self.colors)]

    def ramp(self, t: np.ndarray | float) -> np.ndarray:
        """Interpolate across the palette at position(s) ``t`` in 0..1.

        This is how one palette paints a whole fixture: pass the corridor's
        depth vector and every arch gets its own colour; pass a net's ``y`` and
        the triangle graduates from apex to base.
        """
        t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)
        if len(self.colors) == 1:
            return np.broadcast_to(self.colors[0], (*np.shape(t), 3)).copy()
        pos = t * (len(self.colors) - 1)
        lo = np.floor(pos).astype(np.int32)
        lo = np.minimum(lo, len(self.colors) - 2)
        frac = (pos - lo)[..., None]
        return (self.colors[lo] * (1 - frac) + self.colors[lo + 1] * frac).astype(
            np.float32
        )

    def floored(self, level: float = FLOOR) -> "Palette":
        """Lift the palette until its brightest colour clears ``level``."""
        peak = float(self.colors.max()) if self.colors.size else 0.0
        if peak >= level or peak == 0.0:
            return self
        return Palette(f"{self.name}^", np.clip(self.colors * (level / peak), 0, 1))

    def rotated(self, degrees: float) -> "Palette":
        """Shift every hue, keeping saturation and value."""
        h, s, v = _to_hsv(self.colors)
        return Palette(f"{self.name}+{degrees:.0f}", hsv_to_rgb(h + degrees, s, v))

    def dimmed(self, factor: float) -> "Palette":
        return Palette(f"{self.name}x{factor:.2f}",
                       np.clip(self.colors * factor, 0.0, 1.0))


def generate(hue: float, scheme: str = "analogous", *, value: float = 1.0,
             sat: float = 1.0, white: bool = False,
             name: str | None = None) -> Palette:
    """Build a palette from a base hue and a harmony scheme.

    Slots taper slightly in value and saturation, so the palette has internal
    depth rather than three equally loud colours competing.
    """
    offsets = SCHEMES.get(scheme, SCHEMES["analogous"])
    hues = np.array([hue + off for off in offsets], dtype=np.float32)
    index = np.arange(len(offsets), dtype=np.float32)
    colors = hsv_to_rgb(hues, sat * (1.0 - 0.08 * index), value * (1.0 - 0.10 * index))
    if white:
        # A near-white slot gives drops their punch without washing out the hue.
        colors = np.vstack([colors, hsv_to_rgb(
            np.float32(hue), np.float32(0.05), np.float32(min(1.0, value * 1.05)))])
    return Palette(name or f"{scheme}@{hue % 360:.0f}", colors.astype(np.float32))


def blend(a: Palette, b: Palette, t: float) -> Palette:
    t = float(np.clip(t, 0.0, 1.0))
    n = min(len(a), len(b))
    return Palette(f"{a.name}->{b.name}@{t:.2f}",
                   (a.colors[:n] * (1 - t) + b.colors[:n] * t).astype(np.float32))


WHITE = Palette("white", np.ones((1, 3), dtype=np.float32))
BLACK = Palette("black", np.zeros((1, 3), dtype=np.float32))


def _to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    v = rgb.max(axis=-1)
    span = v - rgb.min(axis=-1)
    s = np.where(v > 0, span / np.where(v > 0, v, 1.0), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(
            span == 0, 0.0,
            np.where(v == r, ((g - b) / span) % 6.0,
                     np.where(v == g, (b - r) / span + 2.0, (r - g) / span + 4.0)),
        ) * 60.0
    return h.astype(np.float32), s.astype(np.float32), v.astype(np.float32)


if __name__ == "__main__":
    for scheme in SCHEMES:
        p = generate(200.0, scheme, value=0.9)
        rgb = (p.colors * 255).round().astype(int)
        print(f"{scheme:<14} " + "  ".join("#%02x%02x%02x" % tuple(c) for c in rgb))
    near = generate(180.0, "sweep")
    far = near.rotated(90.0)
    print("\ncorridor ramp, front to back:")
    for i, c in enumerate((blend(near, far, d).colors[0] * 255).round().astype(int)
                          for d in np.linspace(0, 1, 6)):
        print(f"  depth {i / 5:.1f}  #%02x%02x%02x" % tuple(c))
