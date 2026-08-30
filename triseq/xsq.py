"""Writer and validator for xLights ``.xsq`` sequence files.

Format notes, confirmed against the files already in this show folder:

* Section order is fixed: head, ColorPalettes, EffectDB, SequenceMedia,
  DataLayers, DisplayElements, ElementEffects, lastView, TimingTags.
* ``EffectDB`` and ``ColorPalettes`` are flat lists; effects reference them by
  0-based index via the ``ref`` and ``palette`` attributes.  Identical settings
  strings are interned so each distinct one appears once.
* All times are integer milliseconds and must land on the frame grid (50 ms).
* Effects within one ``EffectLayer`` must be ascending and non-overlapping.
* Timing-track effects carry only ``label``/``startTime``/``endTime``.
* ``DisplayElements`` and ``ElementEffects`` must list the same elements in the
  same order, and names must match ``xlights_rgbeffects.xml`` byte for byte.

Unlike every existing sequence here (which are all ``Animation`` with no media),
we write ``sequenceType=Media`` with a ``mediaFile`` so xLights shows the
waveform.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from .effects import EffectSpec
from .palettes import Palette

XLIGHTS_VERSION = "2026.15"

#: Sequence frame grid in milliseconds. xLights accepts 25, 50 or 100.
#:
#: 25 rather than 50 because effect times must land on this grid, and most
#: tempos do not divide into it. At 136 BPM a beat is 441.2ms: on a 50ms grid
#: beats quantize to alternating 450/400ms spacings, a 50ms hiccup every few
#: beats and up to 23.5ms of error. At 25ms that halves to 11.8ms, which is
#: below what reads as a timing error. The cost is a denser render -- 40fps
#: instead of 20 -- which this show's controller handles comfortably.
FRAME_MS = 25
VALID_FRAME_MS = (25, 50, 100)


class SequenceError(RuntimeError):
    pass


@dataclass
class _Effect:
    spec: EffectSpec
    palette: int
    start_ms: int
    end_ms: int
    #: Length it was added with, before any overlap trimming.
    orig_ms: int = 0


@dataclass
class _Mark:
    start_ms: int
    end_ms: int
    label: str = ""


@dataclass
class SequenceBuilder:
    """Accumulates effects, then writes a valid .xsq."""

    duration_s: float
    media_file: Path | None = None
    song: str = ""
    artist: str = ""
    comment: str = ""
    author: str = ""
    frame_ms: int = FRAME_MS
    #: Layer used by add() when none is given. The arranger sets this to its
    #: bed layer so that only accents have to name a layer explicitly.
    default_layer: int = 0

    _effect_db: list[str] = field(default_factory=list)
    _effect_ix: dict[str, int] = field(default_factory=dict)
    _palettes: list[str] = field(default_factory=list)
    _palette_ix: dict[str, int] = field(default_factory=dict)
    # element name -> layer index -> effects
    _layers: dict[str, dict[int, list[_Effect]]] = field(default_factory=dict)
    _timing: dict[str, list[_Mark]] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)
    #: elements that were declared but may legitimately end up empty
    _declared: list[str] = field(default_factory=list)

    # -- interning ----------------------------------------------------------

    def _intern_effect(self, spec: EffectSpec) -> int:
        s = spec.serialize()
        if s not in self._effect_ix:
            self._effect_ix[s] = len(self._effect_db)
            self._effect_db.append(s)
        return self._effect_ix[s]

    def _intern_palette(self, palette: Palette) -> int:
        s = palette.serialize()
        if s not in self._palette_ix:
            self._palette_ix[s] = len(self._palettes)
            self._palettes.append(s)
        return self._palette_ix[s]

    # -- quantization -------------------------------------------------------

    def quantize(self, t_s: float) -> int:
        """Seconds -> milliseconds snapped to the frame grid."""
        return int(round(t_s * 1000.0 / self.frame_ms)) * self.frame_ms

    @property
    def duration_ms(self) -> int:
        return self.quantize(self.duration_s)

    # -- building -----------------------------------------------------------

    def declare(self, *elements: str) -> None:
        """Register elements so they appear in the sequence even if empty.

        Order of declaration is the order they appear in the xLights timeline.
        """
        for e in elements:
            if e not in self._declared:
                self._declared.append(e)

    def add(self, element: str, spec: EffectSpec, palette: Palette,
            start_s: float, end_s: float, layer: int | None = None) -> bool:
        """Add one effect. Returns False if it was dropped as degenerate."""
        if element is None:
            return False        # a fixture the layout no longer has (the par)
        if layer is None:
            layer = self.default_layer
        start = self.quantize(start_s)
        end = self.quantize(end_s)

        # Clamp into the sequence and enforce at least one frame.
        start = max(0, min(start, self.duration_ms - self.frame_ms))
        end = max(start + self.frame_ms, min(end, self.duration_ms))
        if end <= start:
            return False

        self._layers.setdefault(element, {}).setdefault(layer, []).append(
            _Effect(
                spec=spec,
                palette=self._intern_palette(palette),
                start_ms=start,
                end_ms=end,
                orig_ms=end - start,
            )
        )
        if element not in self._declared:
            self._declared.append(element)
        return True

    def add_timing_track(self, name: str, marks: list[tuple[float, float, str]]) -> None:
        """Add a timing track. Marks are (start_s, end_s, label)."""
        out: list[_Mark] = []
        for start_s, end_s, label in marks:
            start = self.quantize(start_s)
            end = self.quantize(end_s)
            start = max(0, min(start, self.duration_ms - self.frame_ms))
            end = max(start + self.frame_ms, min(end, self.duration_ms))
            out.append(_Mark(start, end, label))
        out.sort(key=lambda m: m.start_ms)
        # Timing marks must not overlap either.
        cleaned: list[_Mark] = []
        for m in out:
            if cleaned and m.start_ms < cleaned[-1].end_ms:
                cleaned[-1].end_ms = m.start_ms
                if cleaned[-1].end_ms <= cleaned[-1].start_ms:
                    cleaned.pop()
            if m.end_ms > m.start_ms:
                cleaned.append(m)
        self._timing[name] = cleaned
        if name not in self._order:
            self._order.append(name)

    # -- overlap resolution -------------------------------------------------

    def _resolve(self, effects: list[_Effect]) -> tuple[list[_Effect], int, int]:
        """Sort and de-overlap a layer. Returns (effects, truncated, dropped).

        Later-added effects win: an earlier effect is truncated to make room.

        A truncated remnant is kept only if it is still a meaningful fraction
        of what was asked for. Squeezing a 400ms hit down to a single frame
        does not produce a shorter hit, it produces a flicker, and the fades it
        was built with are then longer than the effect itself -- which is what
        xLights' sequence check reports as a transition issue.
        """
        ordered = sorted(effects, key=lambda e: (e.start_ms, e.end_ms))
        out: list[_Effect] = []
        truncated = dropped = 0
        min_keep = self.frame_ms * 3
        for e in ordered:
            if out and e.start_ms < out[-1].end_ms:
                prev = out[-1]
                prev.end_ms = e.start_ms
                left = prev.end_ms - prev.start_ms
                if left < min_keep or left < 0.35 * max(prev.orig_ms, 1):
                    out.pop()
                    dropped += 1
                else:
                    truncated += 1
            if e.end_ms - e.start_ms >= self.frame_ms:
                out.append(e)
            else:
                dropped += 1
        return out, truncated, dropped

    def _fit_fades(self, spec: EffectSpec, dur_ms: int) -> EffectSpec:
        """Clamp an effect's fade-in/out to fit inside its final duration.

        Fades are authored against the length the effect was *asked* for; after
        overlap trimming that can be much shorter. Fade-in and fade-out that
        together exceed the effect overlap each other, so the effect never
        reaches full brightness. Cap each and their sum.
        """
        s = spec.settings
        fi = float(s.get("T_TEXTCTRL_Fadein", "0") or 0)
        fo = float(s.get("T_TEXTCTRL_Fadeout", "0") or 0)
        if fi <= 0 and fo <= 0:
            return spec
        dur = dur_ms / 1000.0
        cap_i, cap_o, cap_sum = dur * 0.35, dur * 0.65, dur * 0.85
        nfi, nfo = min(fi, cap_i), min(fo, cap_o)
        if nfi + nfo > cap_sum:
            scale = cap_sum / (nfi + nfo)
            nfi, nfo = nfi * scale, nfo * scale
        if abs(nfi - fi) < 1e-6 and abs(nfo - fo) < 1e-6:
            return spec
        new = dict(s)
        new["T_TEXTCTRL_Fadein"] = f"{nfi:.2f}"
        new["T_TEXTCTRL_Fadeout"] = f"{nfo:.2f}"
        return EffectSpec(spec.name, new)

    # -- writing ------------------------------------------------------------

    def write(self, path: Path) -> dict[str, int]:
        path = Path(path)
        root = ET.Element("xsequence", {
            "BaseChannel": "0",
            "ChanCtrlBasic": "0",
            "ChanCtrlColor": "0",
            "FixedPointTiming": "1",
            "ModelBlending": "true",
        })

        # --- head
        head = ET.SubElement(root, "head")
        media = str(self.media_file.resolve()) if self.media_file else ""
        for tag, text in [
            ("version", XLIGHTS_VERSION),
            ("author", self.author),
            ("author-email", ""),
            ("author-website", ""),
            ("song", self.song),
            ("artist", self.artist),
            ("album", ""),
            ("MusicURL", ""),
            ("comment", self.comment),
            ("sequenceTiming", f"{self.frame_ms} ms"),
            ("sequenceType", "Media" if media else "Animation"),
            ("mediaFile", media),
            ("sequenceDuration", f"{self.duration_ms / 1000.0:.3f}"),
            ("imageDir", ""),
        ]:
            ET.SubElement(head, tag).text = text or None

        # --- palettes and effect settings
        cp = ET.SubElement(root, "ColorPalettes")
        for s in self._palettes:
            ET.SubElement(cp, "ColorPalette").text = s

        # Created here to keep the section order xLights expects, but filled
        # after ElementEffects has been walked -- effect settings are interned
        # at write time so fades can be fitted to the final trimmed durations.
        db = ET.SubElement(root, "EffectDB")

        ET.SubElement(root, "SequenceMedia")

        dl = ET.SubElement(root, "DataLayers")
        ET.SubElement(dl, "DataLayer", {
            "lor_params": "0", "channel_offset": "0", "num_channels": "0",
            "num_frames": "0", "data": "<rendered: erase-mode>",
            "source": "<auto-generated>", "name": "Nutcracker",
        })

        # --- element lists (timing tracks first, then models)
        timing_names = [n for n in self._order if n in self._timing]
        model_names = list(self._declared)

        de = ET.SubElement(root, "DisplayElements")
        for name in timing_names:
            ET.SubElement(de, "Element", {
                "collapsed": "false", "type": "timing", "name": name,
                "visible": "true", "views": ",Master View", "active": "true",
            })
        for name in model_names:
            ET.SubElement(de, "Element", {
                "collapsed": "false", "type": "model", "name": name,
                # The 24 individual arches are collapsed out of the way by
                # default; the Tunnel group is what you want to look at.
                "visible": "false" if name.startswith("Poly Line-") else "true",
            })

        # --- the effects themselves
        ee = ET.SubElement(root, "ElementEffects")
        for name in timing_names:
            el = ET.SubElement(ee, "Element", {"type": "timing", "name": name})
            layer = ET.SubElement(el, "EffectLayer")
            for m in self._timing[name]:
                ET.SubElement(layer, "Effect", {
                    "label": m.label,
                    "startTime": str(m.start_ms),
                    "endTime": str(m.end_ms),
                })

        stats = {"effects": 0, "dropped": 0, "truncated": 0, "refitted": 0}
        for name in model_names:
            el = ET.SubElement(ee, "Element", {"type": "model", "name": name})
            layers = self._layers.get(name, {})
            if not layers:
                ET.SubElement(el, "EffectLayer")
                continue
            for li in sorted(layers):
                node = ET.SubElement(el, "EffectLayer")
                resolved, trunc, drop = self._resolve(layers[li])
                stats["dropped"] += drop
                stats["truncated"] += trunc
                stats["effects"] += len(resolved)
                for e in resolved:
                    spec = self._fit_fades(e.spec, e.end_ms - e.start_ms)
                    if spec is not e.spec:
                        stats["refitted"] += 1
                    ET.SubElement(node, "Effect", {
                        "ref": str(self._intern_effect(spec)),
                        "name": spec.name,
                        "startTime": str(e.start_ms),
                        "endTime": str(e.end_ms),
                        "palette": str(e.palette),
                    })

        for text in self._effect_db:
            ET.SubElement(db, "Effect").text = text

        ET.SubElement(root, "lastView").text = "0"
        tt = ET.SubElement(root, "TimingTags")
        for i in range(10):
            ET.SubElement(tt, "Tag", {"number": str(i), "position": "-1"})

        ET.indent(root, space="  ")
        path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)

        stats["palettes"] = len(self._palettes)
        stats["effect_defs"] = len(self._effect_db)
        stats["elements"] = len(model_names) + len(timing_names)
        return stats


def validate(path: Path, valid_names: set[str] | None = None) -> list[str]:
    """Re-parse a written sequence and check every invariant.

    Returns a list of problems; empty means the file is sound.
    """
    problems: list[str] = []
    root = ET.parse(path).getroot()

    if root.tag != "xsequence":
        return [f"root element is {root.tag!r}, expected 'xsequence'"]

    n_db = len(root.findall("./EffectDB/Effect"))
    n_pal = len(root.findall("./ColorPalettes/ColorPalette"))

    head = root.find("head")
    frame_ms = int((head.findtext("sequenceTiming") or "50 ms").split()[0])
    duration_ms = int(round(float(head.findtext("sequenceDuration") or "0") * 1000))

    if head.findtext("sequenceType") == "Media":
        mf = head.findtext("mediaFile") or ""
        if not mf:
            problems.append("sequenceType is Media but mediaFile is empty")
        elif not Path(mf).exists():
            problems.append(f"mediaFile does not exist: {mf}")

    display = [(e.get("type"), e.get("name")) for e in root.findall("./DisplayElements/Element")]
    effects = [(e.get("type"), e.get("name")) for e in root.findall("./ElementEffects/Element")]
    if display != effects:
        problems.append(
            "DisplayElements and ElementEffects disagree:\n"
            f"  display: {display}\n  effects: {effects}"
        )

    if valid_names is not None:
        for kind, name in display:
            if kind == "model" and name not in valid_names:
                problems.append(f"element {name!r} is not a model or group in the show")

    for el in root.findall("./ElementEffects/Element"):
        name = el.get("name")
        is_timing = el.get("type") == "timing"
        for li, layer in enumerate(el.findall("EffectLayer")):
            prev_end = -1
            for fx in layer.findall("Effect"):
                start = int(fx.get("startTime"))
                end = int(fx.get("endTime"))
                where = f"{name!r} layer {li} [{start}-{end}]"

                if start % frame_ms or end % frame_ms:
                    problems.append(f"{where}: not on the {frame_ms}ms frame grid")
                if end <= start:
                    problems.append(f"{where}: end is not after start")
                if end > duration_ms:
                    problems.append(f"{where}: extends past duration {duration_ms}ms")
                if start < prev_end:
                    problems.append(f"{where}: overlaps previous effect ending {prev_end}")
                prev_end = end

                if is_timing:
                    if fx.get("ref") is not None:
                        problems.append(f"{where}: timing mark must not carry a ref")
                    continue

                ref, pal = fx.get("ref"), fx.get("palette")
                if ref is None or not (0 <= int(ref) < n_db):
                    problems.append(f"{where}: ref {ref} out of range (EffectDB has {n_db})")
                if pal is None or not (0 <= int(pal) < n_pal):
                    problems.append(f"{where}: palette {pal} out of range ({n_pal} palettes)")
                if not fx.get("name"):
                    problems.append(f"{where}: missing effect name")

    return problems
