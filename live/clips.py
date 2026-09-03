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
arranger keeps playing.  A decoded clip is kept as a raw file next to the
fseq (clips/.cache/) and memory-mapped, so a load happens once per clip
ever, the OS pages frames in as needed, and thirty clips do not mean a
gigabyte resident on a Pi.

Clips also carry an *energy* (clips/index.json): how bright and how
flickery the material is, measured from the frames themselves.  That is
what lets the arranger put a slow dim loop in a quiet passage and a strobing
one in a drop without anyone tagging files by hand.  The one hand tag is
``silent_only`` (live.toml, ``[clips]``): names the rotation keeps out of
the music entirely and plays only when nothing is playing at all -- a
figurative loop that is a joke between sets and noise under a drop.
"""

from __future__ import annotations

import json
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

    #: Playback reference: at this tempo a beat-locked clip runs at its
    #: authored speed; faster music plays it proportionally faster.
    REFERENCE_BPM = 128.0

    def frame_at_beats(self, beats: float) -> np.ndarray:
        """The frame ``beats`` into the loop, at the reference tempo's rate."""
        return self.frame_at(beats * 60.0 / self.REFERENCE_BPM)


class Clips:
    """The clip library: names for the panel, lazy loading for the engine.

    ``get`` never blocks: it returns the loaded clip, or None while a
    loader thread works on it (or after a load failed -- the error is
    logged once and the arranger simply keeps the stage).
    """

    def __init__(self, directory: Path | str | None = None,
                 channel_count: int | None = None,
                 silent_only: "list[str] | tuple[str, ...]" = ()) -> None:
        self.dir = Path(directory) if directory else CLIPS_DIR
        self.channel_count = channel_count
        #: Names offered in the silent state only (see vocabulary).
        self.silent_only: tuple[str, ...] = tuple(silent_only)
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
        for name in self.silent_only:
            if name not in self.names:
                # A typo in live.toml would otherwise just mean "no cowboy
                # tonight", noticed by nobody.
                log.warning("clips.silent_only names %r, which is not a "
                            "playable clip (have: %s)", name,
                            ", ".join(self.names) or "none")

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
            header = read_header(path)
            raw = self.dir / ".cache" / f"{name}.raw"
            size = header.frame_count * header.channel_count
            if not (raw.exists() and raw.stat().st_size == size
                    and raw.stat().st_mtime >= path.stat().st_mtime):
                _, frames = read_all(path)
                raw.parent.mkdir(exist_ok=True)
                frames.tofile(raw)
            mapped = np.memmap(raw, dtype=np.uint8, mode="r",
                               shape=(header.frame_count, header.channel_count))
            # Touch every page here, on the loader thread, so the render
            # thread never eats a page fault: measured, the first frames of
            # a cold clip cost up to 46 ms against a 25 ms frame budget.
            int(mapped[::16, ::2048].sum())
            clip = Clip(name, mapped, header.fps)
            log.info("clip %s: %d frames at %g fps mapped", name,
                     header.frame_count, header.fps)
            with self._lock:
                self._loaded[name] = clip
        except (OSError, FseqError, MemoryError) as exc:
            log.error("clip %s failed to load: %s", name, exc)
            with self._lock:
                self._failed.add(name)
        finally:
            with self._lock:
                self._loading.discard(name)

    def preload(self, background: bool = False) -> None:
        """Load every clip, serially, so the rotation finds them ready.

        Without this the rotation's walk kept choosing clips that were not
        decoded yet -- each miss kicks a load, but the *next* stretch walks
        to a different name, so for minutes after a start almost every clip
        stretch fell back to a painted look.  One thread, one clip at a
        time: the memmap cache makes each load cheap after the first ever
        run, and the render thread never competes with a decode burst.
        """
        if background:
            threading.Thread(target=self.preload, daemon=True,
                             name="clip-preload").start()
            return
        for name in self.names:
            with self._lock:
                if name in self._loaded or name in self._failed:
                    continue
                if name in self._loading:
                    continue
                self._loading.add(name)
            self._load(name)

    # -- energy ------------------------------------------------------------ #

    #: Fraction of the ranked list each state draws from.  Bands overlap on
    #: purpose -- a mid-energy clip may serve two or three states.  Widened
    #: from (0.30/0.20-0.70/0.45-0.85/0.65) once the show had run a while:
    #: a set that sits in one state for several minutes -- a long cruise, a
    #: long build -- walked a pool of a dozen and came round to the same
    #: loops.  The overlap is the point: each clip now serves 2.1 states on
    #: average rather than 1.5, so a state's pool is wide enough to stay in
    #: for a while.  Quiet's top edge is still well below the strobes.
    BANDS = {"quiet": (0.0, 0.40), "cruising": (0.12, 0.78),
             "building": (0.32, 0.92), "hot": (0.55, 1.0)}

    def build_index(self, background: bool = False) -> None:
        """Measure every clip's energy into clips/index.json.

        Level is the mean of the frames; flicker the mean frame-to-frame
        change -- both subsampled, both from the decoded cache.  Stale or
        missing entries are measured, existing ones kept, so on a deploy the
        index rsyncs with the clips and the Pi never computes it.
        """
        if background:
            threading.Thread(target=self.build_index, daemon=True,
                             name="clip-index").start()
            return
        index_path = self.dir / "index.json"
        try:
            index = json.loads(index_path.read_text())
        except (OSError, ValueError):
            index = {}
        changed = False
        for name in self.names:
            path = self.dir / f"{name}.fseq"
            entry = index.get(name)
            if entry and entry.get("mtime") == int(path.stat().st_mtime):
                continue
            try:
                _, frames = read_all(path)
            except (OSError, FseqError, MemoryError) as exc:
                log.warning("clip %s: not measurable (%s)", name, exc)
                continue
            sub = np.asarray(frames[::4, ::16], dtype=np.float32) / 255.0
            level = float(sub.mean())
            flicker = float(np.abs(np.diff(sub, axis=0)).mean()) if len(sub) > 1 else 0.0
            index[name] = {"mtime": int(path.stat().st_mtime),
                           "level": round(level, 5), "flicker": round(flicker, 5)}
            changed = True
            log.info("clip %s: level %.3f flicker %.4f", name, level, flicker)
        if changed:
            try:
                index_path.write_text(json.dumps(index, indent=1, sort_keys=True))
            except OSError as exc:
                log.warning("could not write %s: %s", index_path, exc)
        with self._lock:
            self._index = index

    def vocabulary(self, kind: str) -> list[str]:
        """Clip names whose measured energy suits a state, calm to busy.

        Ranked by level + 4x flicker (flicker separates a strobe from a
        bright wash at the same mean) and cut by BANDS.  Empty until the
        index exists, which switches the rotation off rather than guessing.

        ``silent_only`` clips stand outside the ranking entirely: they never
        reach a music state whatever their energy, and the silent state gets
        them and nothing else.  So every music state draws from the measured
        material alone, and the room between sets is the only place the
        tagged loops appear.
        """
        with self._lock:
            index = getattr(self, "_index", None) or {}
        if kind == "silent":
            return [n for n in self.names if n in self.silent_only]
        scored = sorted(
            (index[n]["level"] + 4.0 * index[n]["flicker"], n)
            for n in self.names if n in index and n not in self.silent_only)
        if not scored:
            return []
        lo, hi = self.BANDS.get(kind, (0.0, 1.0))
        count = len(scored)
        picked = [n for rank, (_, n) in enumerate(scored)
                  if lo * (count - 1) <= rank <= hi * (count - 1)]
        return picked or [scored[min(int(lo * count), count - 1)][1]]
