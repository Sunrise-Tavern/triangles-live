"""One file the rig is configured from.

At a gig nobody remembers command-line flags, and the machine that matters is
headless and probably being reached over SSH from a phone.  So everything that
changes between the sofa and the venue lives in one commented TOML file, and
the daemon is started the same way every time.

Resolution order is the conventional one and it matters here: **an explicit
flag beats the file, and the file beats the built-in default.**  That way the
config is the thing that runs the show, but you can still override one value
to test something without editing it back afterwards -- which is exactly the
kind of edit that gets left in by accident at 2 a.m.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "live.toml"
ENV_VAR = "TRIANGLES_CONFIG"


@dataclass
class OutputConfig:
    #: The Falcon.  Empty means render but send nothing, which is the default
    #: on a laptop -- an accidental blast at the rig is worse than silence.
    host: str = ""
    port: int = 4048
    #: Clip the send to one controller's channel space.  The arches and the
    #: par sit outside it; see `live layout`.
    controller: str = ""
    fps: float = 40.0
    channels_per_packet: int = 1440


@dataclass
class AudioConfig:
    #: Input device index or name.  Empty means the system default; run
    #: `live listen --devices` at the rig to find the USB interface.
    device: str = ""
    #: Drive the show from a file instead of the line input, for testing.
    file: str = ""
    loop: bool = False
    autogain: bool = True
    backend: str = "aubio"
    #: Samples per analysis hop, and aubio's hop size.  **A last resort on a
    #: slow machine, not a free win.**  1024 does halve the analysis cost, but
    #: measured against exact ground truth it takes beats within 30 ms from
    #: 100% to 68% and the median error from 2.4 ms to 25.6 ms -- the timing
    #: resolution lands right on the threshold rather than comfortably inside
    #: it.  Lower the frame rate first; that costs nothing.
    blocksize: int = 512
    #: FFT window.  Must be at least the block size.  2048 gives the bass band
    #: 21.5 Hz bins, which is what makes the kick detector work.  1024 is
    #: cheaper and worse: 50% of beats within 30 ms in the same test.
    window: int = 2048
    #: Level below which nothing is playing, dBFS.  -70 suits a line feed;
    #: a room microphone wants roughly -45.  `live doctor` measures your input
    #: and tells you what to set.
    silence_dbfs: float = -70.0


@dataclass
class WebConfig:
    bind: str = "0.0.0.0"
    port: int = 8080
    #: 0 means "match the engine's frame rate".
    preview_fps: float = 0.0
    preview_detail: float = 1.0


@dataclass
class ShowConfig:
    brightness: float = 1.0
    gamma: float = 1.0
    #: Shift the lights against the PA.  Positive fires later.
    latency_ms: float = 0.0
    #: Preset to load at startup, by name.
    preset: str = ""
    #: How bright the resting group of triangles glows while the other leads
    #: a gesture.  0 = hold the bed wash (frozen), as the offline show does.
    rest_level: float = 0.4
    #: Start blacked out, so a restart mid-set does not throw light until
    #: someone is watching.
    blackout_on_start: bool = False


@dataclass
class LogConfig:
    file: str = "logs/live.log"
    level: str = "info"
    #: Rotate at this size, keeping `keep` files.  A long set on a small SD
    #: card is not the place to discover an unbounded log.
    max_bytes: int = 8 * 1024 * 1024
    keep: int = 3


@dataclass
class Config:
    output: OutputConfig = field(default_factory=OutputConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    web: WebConfig = field(default_factory=WebConfig)
    show: ShowConfig = field(default_factory=ShowConfig)
    log: LogConfig = field(default_factory=LogConfig)
    #: Where it came from, for `live doctor` to report.
    source: Path | None = None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Read a config, or return defaults if there is not one.

        Missing file is not an error: the whole thing has to work on a laptop
        with nothing set up, or the tests and the dev loop need special cases.
        """
        chosen = Path(path) if path else Path(os.environ.get(ENV_VAR, DEFAULT_PATH))
        config = cls()
        if not chosen.exists():
            return config
        with chosen.open("rb") as handle:
            raw = tomllib.load(handle)
        for section in fields(cls):
            if section.name == "source":
                continue
            values = raw.get(section.name)
            if not isinstance(values, dict):
                continue
            target = getattr(config, section.name)
            known = {f.name: f for f in fields(target)}
            for key, value in values.items():
                if key not in known:
                    raise ValueError(
                        f"{chosen}: unknown setting [{section.name}] {key}"
                    )
                setattr(target, key, _coerce(known[key].type, value, section.name, key))
        config.source = chosen
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            section.name: dict(vars(getattr(self, section.name)))
            for section in fields(self) if section.name != "source"
        }


def _coerce(kind: Any, value: Any, section: str, key: str) -> Any:
    want = {"str": str, "int": int, "float": float, "bool": bool}.get(str(kind))
    if want is None:
        return value
    if want is bool and not isinstance(value, bool):
        raise ValueError(f"[{section}] {key} must be true or false")
    try:
        return want(value)
    except (TypeError, ValueError):
        raise ValueError(f"[{section}] {key} must be a {want.__name__}") from None


def setup_logging(config: LogConfig, quiet: bool = False) -> None:
    """Log to a rotating file as well as the console.

    The Pi runs headless under systemd, so the console goes to the journal --
    but a file next to the code is what someone can actually read over SSH
    without knowing journalctl's flags.
    """
    import logging
    from logging.handlers import RotatingFileHandler

    level = getattr(logging, config.level.upper(), logging.INFO)
    handlers: list[logging.Handler] = []
    if config.file:
        path = Path(config.file)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(
            path, maxBytes=config.max_bytes, backupCount=config.keep))
    if not quiet:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=level, handlers=handlers, force=True,
        format="%(asctime)s %(levelname)-7s %(name)-12s %(message)s",
    )
