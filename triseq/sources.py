"""Pluggable audio input.

The generator does not care where audio comes from -- it just needs a local
file plus whatever metadata is available.  Adding a new provider means writing
one class and adding one line to ``SOURCES``.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse


class SourceError(RuntimeError):
    pass


@dataclass
class AudioAsset:
    """A local audio file the analyser can read, plus what we know about it."""

    path: Path
    title: str = ""
    artist: str = ""
    source: str = ""

    def __post_init__(self):
        # Uploaders routinely put "Artist - Title" in the title field while
        # yt-dlp also reports the artist separately, which would otherwise
        # produce "Robert Miles - Robert Miles - Children".
        self.title = self.title.strip()
        self.artist = self.artist.strip()
        if self.artist and self.title.lower().startswith(f"{self.artist.lower()} - "):
            self.title = self.title[len(self.artist) + 3:].strip()
        # Uploader names often carry a channel suffix.
        for suffix in (" - Topic", "VEVO", "Official"):
            if self.artist.endswith(suffix):
                self.artist = self.artist[: -len(suffix)].strip()

    @property
    def label(self) -> str:
        if self.artist and self.title:
            return f"{self.artist} - {self.title}"
        return self.title or self.path.stem


class AudioSource(Protocol):
    def resolve(self, workdir: Path) -> AudioAsset:
        """Produce a local audio file in `workdir`."""
        ...


class LocalFileSource:
    """Any audio file already on disk. Nothing is downloaded or transcoded."""

    def __init__(self, spec: str):
        self.path = Path(spec).expanduser()

    def resolve(self, workdir: Path) -> AudioAsset:
        if not self.path.exists():
            raise SourceError(f"No such audio file: {self.path}")
        artist, title = "", self.path.stem
        # "Artist - Title.mp3" is the near-universal convention; use it if present.
        if " - " in self.path.stem:
            artist, _, title = self.path.stem.partition(" - ")
        return AudioAsset(
            path=self.path,
            title=title.strip(),
            artist=artist.strip(),
            source=str(self.path),
        )


class YouTubeSource:
    """Fetch audio via yt-dlp.

    Works for any site yt-dlp supports, not just YouTube -- the name reflects
    the common case.  Note that downloading from YouTube is outside its terms
    of service; LocalFileSource is the clean path if you already have the file.
    """

    def __init__(self, url: str):
        self.url = url

    #: Player clients to fall back through. YouTube serves different playback
    #: URLs per client and rotates which ones are protected, so a 403 from one
    #: often succeeds on another. These work without a JavaScript runtime.
    CLIENTS = ("android_vr", "ios", "tv", "web_safari", "mweb")

    def _attempts(self) -> list[dict]:
        """Option overrides to try in order, stopping at the first success."""
        attempts = [{"_label": "default"}]
        for client in self.CLIENTS:
            attempts.append({
                "_label": f"player_client={client}",
                "extractor_args": {"youtube": {"player_client": [client]}},
            })
        return attempts

    def resolve(self, workdir: Path) -> AudioAsset:
        try:
            import yt_dlp
        except ImportError as e:
            raise SourceError(
                "yt-dlp is not installed. Run ./setup.sh, or pass a local "
                "audio file instead of a URL."
            ) from e

        workdir.mkdir(parents=True, exist_ok=True)
        base = {
            "format": "bestaudio/best",
            "outtmpl": str(workdir / "%(id)s.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "retries": 5,
            "fragment_retries": 5,
            "noplaylist": True,
        }

        info = None
        errors = []
        for attempt in self._attempts():
            opts = {**base, **attempt}
            label = opts.pop("_label", "default")
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(self.url, download=True)
                break
            except Exception as e:
                errors.append(f"  {label}: {e}")

        if info is None:
            raise SourceError(
                f"Could not fetch {self.url} after {len(errors)} attempts:\n"
                + "\n".join(errors)
                + "\n\nYouTube rotates its playback protections, so this is "
                  "usually one of:\n"
                  "  - yt-dlp is out of date:  .venv/bin/pip install -U yt-dlp\n"
                  "  - no JavaScript runtime:  brew install deno\n"
                  "  - or download the audio yourself and pass the file path."
            )

        if info.get("_type") == "playlist":
            entries = [e for e in info.get("entries") or [] if e]
            if not entries:
                raise SourceError(f"{self.url} yielded no playable entries")
            info = entries[0]

        path = workdir / f"{info['id']}.mp3"
        if not path.exists():
            # The postprocessor may have chosen a different extension.
            found = list(workdir.glob(f"{info['id']}.*"))
            if not found:
                raise SourceError(f"Download produced no file for {info.get('id')}")
            path = found[0]

        return AudioAsset(
            path=path,
            title=info.get("track") or info.get("title") or "",
            artist=info.get("artist") or info.get("uploader") or "",
            source=info.get("webpage_url") or self.url,
        )


#: Scheme -> source class. Extend here to add Soundcloud, Bandcamp, etc.
SOURCES: dict[str, type] = {
    "http": YouTubeSource,
    "https": YouTubeSource,
    "file": LocalFileSource,
}


def resolve_source(spec: str) -> AudioSource:
    """Pick a source implementation for a URL or path."""
    scheme = urlparse(spec).scheme
    if scheme in SOURCES:
        cls = SOURCES[scheme]
        return cls(spec if scheme != "file" else urlparse(spec).path)
    if scheme and len(scheme) > 1:
        raise SourceError(
            f"No audio source registered for {scheme!r} URLs. "
            f"Known: {', '.join(sorted(SOURCES))}"
        )
    return LocalFileSource(spec)


def extract_excerpt(src: Path, dest: Path, start_s: float, duration_s: float) -> Path:
    """Cut `duration_s` starting at `start_s` into a fresh mp3.

    Re-encoded rather than stream-copied: a copy would snap to the nearest
    frame boundary and put the audio a few tens of milliseconds out of step
    with a sequence whose timings were computed against the original.
    """
    if not shutil.which("ffmpeg"):
        raise SourceError("ffmpeg not found on PATH. Install it with: brew install ffmpeg")

    # Reading and writing the same file would have ffmpeg truncate its own
    # input. Easy to hit by regenerating from a previous run's excerpt.
    if dest.exists() and src.exists() and dest.resolve() == src.resolve():
        raise SourceError(
            f"Excerpt would overwrite its own source ({src}).\n"
            f"Pass a different -o name, or copy the input elsewhere first."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_s:.3f}",
        "-t", f"{duration_s:.3f}",
        "-i", str(src),
        "-c:a", "libmp3lame", "-b:a", "192k",
        str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SourceError(f"ffmpeg failed:\n{result.stderr.strip()}")
    if not dest.exists():
        raise SourceError(f"ffmpeg produced no output at {dest}")
    return dest
