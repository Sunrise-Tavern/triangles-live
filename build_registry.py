#!/usr/bin/env python3
"""Generate effect_registry.json from the installed xLights app bundle.

xLights ships machine-readable metadata for every effect under
``Contents/Resources/effectmetadata`` -- name, properties, types, defaults,
ranges and enum options. That is the authoritative source, and it is far better
than scraping ``strings`` off the binary: a strings dump only catches setting
keys that exist as whole string literals, which misses roughly half of them
because many are assembled at runtime.

Re-run this after upgrading xLights:

    ./.venv/bin/python build_registry.py
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

APP = Path("/Applications/xLights.app")
META = APP / "Contents/Resources/effectmetadata"
BINARY = APP / "Contents/MacOS/xLights"
OUT = Path(__file__).resolve().parent / "effect_registry.json"

#: Widget prefix by (type, controlType), used when the binary does not reveal
#: the key directly. Derived by cross-checking every id that *does* appear.
DEFAULT_PREFIX = {
    ("bool", "checkbox"): "CHECKBOX",
    ("enum", "choice"): "CHOICE",
    ("enum", "custom"): "CHOICE",
    ("int", "slider"): "SLIDER",
    ("int", "spin"): "SPINCTRL",
    ("int", "custom"): "SLIDER",
    ("float", "slider"): "TEXTCTRL",
    ("string", "text"): "TEXTCTRL",
    ("string", "custom"): "TEXTCTRL",
    ("file", "filepicker"): "FILEPICKERCTRL",
    ("file", "custom"): "FILEPICKERCTRL",
    ("font", "fontpicker"): "FONTPICKER",
    ("font", "custom"): "FONTPICKER",
    ("color", "colourpicker"): "COLOURPICKER",
}

#: When several prefixes exist for one id, prefer by the property's type.
#: VALUECURVE is always an alternate encoding, never the plain value.
PREFERENCE = {
    "float": ["TEXTCTRL", "SLIDER"],
    "int": ["SLIDER", "TEXTCTRL", "SPINCTRL"],
    "bool": ["CHECKBOX"],
    "enum": ["CHOICE"],
    "string": ["TEXTCTRL"],
    "file": ["FILEPICKERCTRL", "FILEPICKER"],
    "font": ["FONTPICKER"],
}


def binary_keys() -> dict[str, set[str]]:
    """Map property id -> prefixes observed in the binary."""
    try:
        raw = subprocess.run(["strings", str(BINARY)], capture_output=True,
                             text=True, timeout=120).stdout
    except Exception:
        return {}
    found: dict[str, set[str]] = {}
    for m in re.finditer(r"\bE_([A-Z0-9]+)_([A-Za-z0-9_]+)", raw):
        found.setdefault(m.group(2), set()).add(m.group(1))
    return found


def choose_prefix(prop: dict, observed: set[str]) -> str:
    ty = prop.get("type", "int")
    ct = prop.get("controlType", "slider")
    usable = {p for p in observed if p != "VALUECURVE"}
    for want in PREFERENCE.get(ty, []):
        if want in usable:
            return want
    if len(usable) == 1:
        return next(iter(usable))
    return DEFAULT_PREFIX.get((ty, ct), "SLIDER")


def main() -> int:
    if not META.is_dir():
        print(f"No effect metadata at {META}. Is xLights installed?")
        return 1

    observed = binary_keys()
    effects: dict[str, dict] = {}

    for path in sorted(META.glob("*.json")):
        if path.name.startswith("_"):
            continue
        data = json.loads(path.read_text())
        name = data.get("effectName") or path.stem

        params: dict[str, dict] = {}
        for prop in data.get("properties", []):
            pid = prop.get("id")
            if not pid:
                continue
            prefix = choose_prefix(prop, observed.get(pid, set()))
            entry = {
                "key": f"E_{prefix}_{pid}",
                "type": prop.get("type"),
                "control": prop.get("controlType"),
            }
            for opt in ("default", "min", "max", "options", "divisor"):
                if opt in prop:
                    entry[opt] = prop[opt]
            params[pid] = entry

        effects[name] = {"params": params, "file": path.name}

    OUT.write_text(json.dumps({"effects": effects}, indent=1, sort_keys=True))
    total = sum(len(e["params"]) for e in effects.values())
    print(f"Wrote {OUT.name}: {len(effects)} effects, {total} parameters")
    print(f"  binary confirmed {len(observed)} ids directly; "
          f"the rest use the (type, controlType) convention")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
