"""Canned looks: xLights-rendered .fseq loops, selectable from the panel.

The offline world keeps producing finished material -- ``clips/`` holds
whole-rig sequences designed in xLights against the same layout the engine
drives.  Picking one in the panel loops it on the rig verbatim, in place of
the arranger, until the knob goes back to "off".  Nothing is analysed or
re-rendered: the frames go to the wire as authored (master brightness and
gamma still apply, since they are the operator's, not the author's).

Loading is lazy and off the render thread: a 30 s whole-rig clip is about
40 MB decompressed, half a second of zstd on a laptop and a few on a Pi,
and the show must not freeze while it happens.  Until the clip is ready the
arranger keeps playing.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from .fseq import FseqError, read_all, read_header

log = logging.getLogger(__name__)

#: Sibling of live/ -- the repo root's clips directory.
CLIPS_DIR = Path(__file__).resolve().parent.parent / "clips"


class Clip:
    """One loaded loop: (frames, channels) uint8, and where in it we are."""

    def __init__(self, name: str, frames: np.ndarray, fps: float) -> None:
        self.name = name
        self.frames = frames
        self.fps = fps

    def frame_at(self, seconds: float) -> np.ndarray:
        """The frame ``seconds`` into the loop, wrapping."""
        index = int(seconds * self.fps) % max(1, len(self.frames))
        return self.frames[index]


class Clips:
    """The clip library: names for the panel, lazy loading for the engine.

    ``get`` never blocks: it returns the loaded clip, or None while a
    loader thread works on it (or after a load failed -- the error is
    logged once and the arranger simply keeps the stage).
    """

    def __init__(self, directory: Path | str | None = None,
                 channel_count: int | None = None) -> None:
        self.dir = Path(directory) if directory else CLIPS_DIR
        self.channel_count = channel_count
        self.names: list[str] = []
        self._skipped: list[str] = []
        self._loaded: dict[str, Clip] = {}
        self._failed: set[str] = set()
        self._loading: set[str] = set()
        self._lock = threading.Lock()
        if self.dir.is_dir():
            for path in sorted(self.dir.glob("*.fseq")):
                try:
                    header = read_header(path)
                except (OSError, FseqError) as exc:
                    log.warning("clip %s: unreadable (%s)", path.name, exc)
                    self._skipped.append(path.stem)
                    continue
                if (channel_count is not None
                        and header.channel_count != channel_count):
                    # Rendered against another layout; playing it would put
                    # pixels on the wrong fixtures.  Named at startup so a
                    # stale clip is noticed, not mysteriously absent.
                    log.warning("clip %s: %d channels, the layout has %d -- "
                                "re-render it in xLights", path.name,
                                header.channel_count, channel_count)
                    self._skipped.append(path.stem)
                    continue
                self.names.append(path.stem)

    def get(self, name: str) -> Clip | None:
        """The clip if it is ready; None otherwise (loading, off, unknown)."""
        if name not in self.names:
            return None
        with self._lock:
            clip = self._loaded.get(name)
            if clip is not None or name in self._failed or name in self._loading:
                return clip
            self._loading.add(name)
        threading.Thread(target=self._load, args=(name,), daemon=True,
                         name=f"clip:{name}").start()
        return None

    def _load(self, name: str) -> None:
        path = self.dir / f"{name}.fseq"
        try:
            header, frames = read_all(path)
            clip = Clip(name, frames, header.fps)
            log.info("clip %s: %d frames at %g fps loaded", name,
                     len(frames), header.fps)
            with self._lock:
                self._loaded[name] = clip
                # A little cache: whole-rig clips are ~40 MB each decoded.
                while len(self._loaded) > 3:
                    oldest = next(n for n in self._loaded if n != name)
                    del self._loaded[oldest]
        except (OSError, FseqError, MemoryError) as exc:
            log.error("clip %s failed to load: %s", name, exc)
            with self._lock:
                self._failed.add(name)
        finally:
            with self._lock:
                self._loading.discard(name)
