"""The knobs, and named snapshots of them.

Everything the browser can change lives in one flat dataclass.  Flat on
purpose: the UI is generated from :func:`Settings.schema`, so adding a knob is
one line here and it appears with the right widget, range and label -- no
parallel list to forget to update.

Reads and writes cross a thread boundary (web server -> render loop) without a
lock.  That is safe because every field is a single ``bool``/``float``/``str``
and CPython attribute assignment is atomic: the worst case is a frame that sees
brightness from before a change and hue from after, which is invisible.  A
whole-preset load is the one multi-field change, and it takes the lock.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

PRESET_DIR = Path(__file__).resolve().parent / "presets"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,40}$")


@dataclass
class Settings:
    # -- output ------------------------------------------------------------ #
    output_enabled: bool = True
    #: Kill the lights without stopping the engine.  Distinct from disabling
    #: output: the show keeps running underneath, so releasing it is seamless.
    blackout: bool = False
    brightness: float = 1.0
    #: 1.0 is linear, matching the Falcon's DefaultGammaUnderFullControl=1.
    gamma: float = 1.0
    #: Colour saturation over the whole rig: 0 grayscale, 1 as designed.
    saturation: float = 1.0
    #: Contrast about the midpoint: 1 as designed, above it punchier.
    contrast: float = 1.0

    # -- arrangement ------------------------------------------------------- #
    #: Multiplies how many corridor phrases fit in a scene.  Lower = slower
    #: gestures down the tunnel.
    corridor_rate: float = 1.0
    #: 0 sparse, 1 busy.  Picks corridor patterns by their density, the same
    #: way the offline arranger does.
    articulation: float = 0.5
    #: Force one corridor pattern instead of following the scene.
    pattern: str = "auto"
    #: Degrees added to every palette's base hue.
    hue_offset: float = 0.0
    #: Stop the palette journey rotating, so the show holds one colour.
    hue_lock: bool = False
    #: Tempo the scripted show runs at, until M4's beat clock supplies one.
    bpm: float = 128.0
    #: Freeze on one scene instead of following the music.
    scene: str = "auto"
    #: Play a canned xLights loop from clips/ instead of the arranger.
    #: "off" = the show.  Master brightness and gamma still apply.
    clip: str = "off"
    #: Share of phrases the rotation gives to a canned clip instead of a
    #: painted look, beat-locked and enveloped by the music.  0 = never.
    clip_share: float = 0.3
    #: Force one colour scheme instead of letting each visit to a state
    #: choose its own.
    scheme: str = "auto"
    #: Reshuffles every deterministic choice the show makes -- which pattern,
    #: gesture, scheme and transition a phrase gets.  Same seed, same show.
    seed: float = 7.0
    #: How much of the show's material changes with a transition rather than
    #: a cut: 0 is always a cut, 1 lets every phrase fade, wipe or dip.
    transitions: float = 0.75
    #: Phrases a corridor pattern and net gesture are held for.  The colour
    #: still moves every phrase; the material only every this many.
    pattern_hold: float = 4.0
    #: Some net gestures lead with one group of triangles (big or small) and
    #: rest the other.  This is how bright the resting group's slow plasma
    #: is: 0 restores the original behaviour, where it holds the bed wash and
    #: reads as frozen for a bar or a phrase while the corridor moves.
    rest_level: float = 0.4

    # -- state machine (only meaningful when driven by audio) -------------- #
    #: Loudness, relative to a 45 s baseline, below which a passage is quiet.
    quiet_enter: float = 0.62
    #: Share of spectral energy in the high band that marks a build.
    build_high_share: float = 0.40
    #: How hard a kick must hit, out of a build, to call the drop.
    drop_kick: float = 3.0
    #: Shift the lights against the PA, milliseconds. Positive fires later.
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    # -- access ------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def apply(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Validate and apply a partial update.  Returns what actually changed."""
        changed: dict[str, Any] = {}
        with self._lock:
            for key, raw in patch.items():
                if key not in SCHEMA:
                    raise KeyError(f"unknown setting {key!r}")
                value = _coerce(key, raw)
                if getattr(self, key) != value:
                    setattr(self, key, value)
                    changed[key] = value
        return changed

    def load(self, values: dict[str, Any]) -> None:
        """Apply a whole preset, ignoring keys this build does not know.

        Forgiving on purpose: presets saved before a knob existed must still
        load, and presets saved after must not break an older build at the rig.
        """
        self.apply({k: v for k, v in values.items() if k in SCHEMA})


#: field -> (kind, low, high, step, label).  Drives both validation and the UI.
SCHEMA: dict[str, tuple] = {
    "output_enabled": ("bool", None, None, None, "Output"),
    "blackout": ("bool", None, None, None, "Blackout"),
    "brightness": ("float", 0.0, 1.0, 0.01, "Master brightness"),
    "gamma": ("float", 0.5, 3.0, 0.05, "Gamma"),
    "saturation": ("float", 0.0, 2.0, 0.05, "Saturation"),
    "contrast": ("float", 0.5, 1.5, 0.05, "Contrast"),
    "corridor_rate": ("float", 0.25, 4.0, 0.05, "Corridor rate"),
    "articulation": ("float", 0.0, 1.0, 0.01, "Articulation"),
    "pattern": ("choice", None, None, None, "Corridor pattern"),
    "hue_offset": ("float", -180.0, 180.0, 1.0, "Hue offset"),
    "hue_lock": ("bool", None, None, None, "Lock hue"),
    "bpm": ("float", 60.0, 200.0, 0.5, "Tempo"),
    "scene": ("choice", None, None, None, "Scene"),
    "clip": ("choice", None, None, None, "Clip"),
    "clip_share": ("float", 0.0, 1.0, 0.05, "Clips in rotation"),
    "scheme": ("choice", None, None, None, "Colour scheme"),
    "seed": ("float", 0.0, 999.0, 1.0, "Shuffle seed"),
    "transitions": ("float", 0.0, 1.0, 0.05, "Transitions"),
    "pattern_hold": ("float", 1.0, 8.0, 1.0, "Pattern hold (phrases)"),
    "rest_level": ("float", 0.0, 1.0, 0.01, "Resting nets"),
    "quiet_enter": ("float", 0.2, 1.2, 0.01, "Quiet threshold"),
    "build_high_share": ("float", 0.2, 0.8, 0.01, "Build sensitivity"),
    "drop_kick": ("float", 1.5, 8.0, 0.1, "Drop sensitivity"),
    "latency_ms": ("float", -200.0, 200.0, 1.0, "Latency offset (ms)"),
}

#: Filled in by :mod:`live.engine`, which owns the vocabularies.
CHOICES: dict[str, list[str]] = {"pattern": ["auto"], "scene": ["auto"],
                                 "scheme": ["auto"], "clip": ["off"]}


def _coerce(key: str, raw: Any) -> Any:
    kind, low, high, _step, _label = SCHEMA[key]
    if kind == "bool":
        return bool(raw)
    if kind == "float":
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{key}: {raw!r} is not a number") from None
        # Clamp rather than reject: a slider that has drifted a hair past its
        # end should not fail a whole settings patch mid-set.
        return min(max(value, low), high)
    if kind == "choice":
        value = str(raw)
        if value not in CHOICES.get(key, []):
            raise ValueError(f"{key}: {value!r} is not one of {CHOICES.get(key)}")
        return value
    raise ValueError(f"{key}: unknown kind {kind!r}")


def describe() -> list[dict[str, Any]]:
    """The schema as JSON, for the browser to build its controls from."""
    out = []
    for key, (kind, low, high, step, label) in SCHEMA.items():
        entry: dict[str, Any] = {"key": key, "kind": kind, "label": label}
        if kind == "float":
            entry.update(min=low, max=high, step=step)
        if kind == "choice":
            entry["options"] = CHOICES.get(key, [])
        out.append(entry)
    return out


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #


def preset_path(name: str) -> Path:
    if not SAFE_NAME.match(name):
        raise ValueError(
            f"{name!r} is not a usable preset name -- letters, digits, spaces, "
            "hyphens and underscores only"
        )
    return PRESET_DIR / f"{name}.json"


def list_presets() -> list[str]:
    if not PRESET_DIR.exists():
        return []
    return sorted(p.stem for p in PRESET_DIR.glob("*.json"))


def save_preset(name: str, settings: Settings) -> Path:
    path = preset_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def read_preset(name: str) -> dict[str, Any]:
    path = preset_path(name)
    if not path.exists():
        raise FileNotFoundError(f"no preset {name!r}")
    return json.loads(path.read_text())


def delete_preset(name: str) -> None:
    preset_path(name).unlink(missing_ok=True)
