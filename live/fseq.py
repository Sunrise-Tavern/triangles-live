"""FSEQ v2.0 reader/writer -- uncompressed, streaming.

``.fseq`` is xLights' rendered-frame format: a header, then one flat array of
channel bytes per frame.  Writing it is how the live engine gets a **visual
oracle**: capture what went out over DDP, save it, open it in xLights' 3D
preview next to the real layout.  ``xLights --fseqcmp a.fseq b.fseq`` then
compares two files channel-for-channel, which turns "does the renderer do what
I think" into a byte-exact test.

Header layout (little-endian throughout), reverse-engineered from files this
show folder already contains and cross-checked against xLights' FSEQFile.cpp:

===========  =====================================================
offset       field
===========  =====================================================
0            magic ``PSEQ``
4    u16     offset of the first frame's data
6    u8      minor version
7    u8      major version
8    u16     offset of the variable headers (= 32 + 8*blocks + 6*ranges)
10   u32     channels per frame
14   u32     frame count
18   u8      step time, ms
19   u8      flags (0)
20   u8      high nibble: compression-block count >> 8; low: compression type
21   u8      compression-block count & 0xFF
22   u8      sparse-range count
23   u8      reserved
24   u64     unique id (creation time, microseconds)
32   ...     compression blocks, sparse ranges, then variable headers
===========  =====================================================

We always write compression type 0 with no blocks and no sparse ranges: a Pi
rendering live has no spare cycles for zstd, and these files are scratch.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

MAGIC = b"PSEQ"
HEADER_SIZE = 32
COMPRESSION_NONE = 0
COMPRESSION_ZSTD = 1


class FseqError(RuntimeError):
    pass


@dataclass
class FseqHeader:
    channel_count: int
    frame_count: int
    step_time_ms: int
    data_offset: int
    version: tuple[int, int]
    compression_type: int
    #: v2 compressed files only: number of compression blocks.
    block_count: int
    unique_id: int
    variable_headers: dict[str, bytes]

    @property
    def fps(self) -> float:
        return 1000.0 / self.step_time_ms

    @property
    def duration_s(self) -> float:
        return self.frame_count * self.step_time_ms / 1000.0


class FseqWriter:
    """Append frames as they are produced; the frame count is patched on close.

    Streaming matters: a live capture has no idea how long it will run, and
    buffering 40 fps x 37 084 channels in memory is 1.5 MB/s for nothing.
    """

    def __init__(
        self,
        path: str | Path,
        channel_count: int,
        step_time_ms: int = 25,
        media_file: str | None = None,
        unique_id: int = 0,
    ) -> None:
        self.path = Path(path)
        self.channel_count = int(channel_count)
        self.step_time_ms = int(step_time_ms)
        self.frame_count = 0
        if not 1 <= self.step_time_ms <= 255:
            raise FseqError(f"step time {self.step_time_ms} ms does not fit in a byte")

        variable: list[tuple[str, bytes]] = []
        if media_file:
            variable.append(("mf", media_file.encode("utf-8") + b"\0"))

        blob = b"".join(
            struct.pack("<H", len(data) + 4) + code.encode("ascii") + data
            for code, data in variable
        )
        # xLights pads the data start to a 4-byte boundary.
        data_offset = HEADER_SIZE + len(blob)
        data_offset += (-data_offset) % 4
        self.data_offset = data_offset

        self._fh: BinaryIO = self.path.open("wb")
        self._fh.write(self._header(unique_id))
        self._fh.write(blob)
        self._fh.write(b"\0" * (data_offset - HEADER_SIZE - len(blob)))

    def _header(self, unique_id: int) -> bytes:
        return struct.pack(
            "<4sHBBHIIBBBBBBQ",
            MAGIC,
            self.data_offset,
            0,                      # minor version
            2,                      # major version
            HEADER_SIZE,            # variable headers start right after
            self.channel_count,
            self.frame_count,
            self.step_time_ms,
            0,                      # flags
            COMPRESSION_NONE,
            0,                      # compression blocks
            0,                      # sparse ranges
            0,                      # reserved
            unique_id,
        )

    def add_frame(self, frame: np.ndarray | bytes) -> None:
        if isinstance(frame, np.ndarray):
            if frame.dtype != np.uint8:
                raise FseqError(f"frame dtype must be uint8, got {frame.dtype}")
            data = frame.tobytes()
        else:
            data = bytes(frame)
        if len(data) != self.channel_count:
            raise FseqError(
                f"frame is {len(data)} channels, file is {self.channel_count}"
            )
        self._fh.write(data)
        self.frame_count += 1

    def close(self) -> None:
        if self._fh.closed:
            return
        self._fh.seek(0)
        self._fh.write(self._header(0))
        self._fh.close()

    def __enter__(self) -> "FseqWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_header(path: str | Path) -> FseqHeader:
    with Path(path).open("rb") as fh:
        return _read_header(fh)


def _read_header(fh: BinaryIO) -> FseqHeader:
    raw = fh.read(HEADER_SIZE)
    if len(raw) < HEADER_SIZE or raw[:4] != MAGIC:
        raise FseqError("not an fseq file (bad magic)")
    (
        _magic, data_offset, minor, major, var_offset, channels, frames,
        step_ms, _flags, comp, blocks_lo, _ranges, _reserved, uid,
    ) = struct.unpack("<4sHBBHIIBBBBBBQ", raw)
    if major != 2:
        raise FseqError(f"only fseq v2 is supported, this is v{major}.{minor}")
    # Upper nibble of the compression byte holds the block count's high bits.
    block_count = ((comp >> 4) << 8) | blocks_lo

    variable: dict[str, bytes] = {}
    fh.seek(var_offset)
    while fh.tell() < data_offset - 4:
        head = fh.read(4)
        if len(head) < 4:
            break
        (length,) = struct.unpack("<H", head[:2])
        if length < 4:
            break
        variable[head[2:4].decode("ascii", "replace")] = fh.read(length - 4)

    return FseqHeader(
        channel_count=channels, frame_count=frames, step_time_ms=step_ms,
        data_offset=data_offset, version=(major, minor),
        compression_type=comp & 0x0F, block_count=block_count,
        unique_id=uid, variable_headers=variable,
    )


def read_frames(path: str | Path) -> Iterator[np.ndarray]:
    """Yield each frame as a uint8 array of length ``channel_count``.

    Handles uncompressed files and zstd-compressed ones (what xLights writes
    by default; every clip in ``clips/`` is one).  zstd needs the
    ``zstandard`` package -- in requirements-live.txt, so the Pi has it.
    """
    path = Path(path)
    with path.open("rb") as fh:
        header = _read_header(fh)
        if header.compression_type == COMPRESSION_NONE:
            fh.seek(header.data_offset)
            for _ in range(header.frame_count):
                block = fh.read(header.channel_count)
                if len(block) < header.channel_count:
                    raise FseqError("file truncated mid-frame")
                yield np.frombuffer(block, dtype=np.uint8)
            return
        if header.compression_type != COMPRESSION_ZSTD:
            raise FseqError(
                f"{path.name}: compression type {header.compression_type} is "
                "neither none (0) nor zstd (1); re-render it in xLights."
            )
        try:
            import zstandard
        except ImportError as exc:            # pragma: no cover
            raise FseqError(
                f"{path.name} is zstd-compressed and the zstandard package is "
                "missing -- pip install zstandard (it is in "
                "requirements-live.txt)."
            ) from exc
        # The block table sits right after the 32-byte fixed header: one
        # (first frame, compressed size) pair per block.
        fh.seek(HEADER_SIZE)
        table = [struct.unpack("<II", fh.read(8))
                 for _ in range(header.block_count)]
        fh.seek(header.data_offset)
        decompressor = zstandard.ZstdDecompressor()
        emitted = 0
        for _first_frame, size in table:
            if size == 0:
                continue
            raw = decompressor.decompress(
                fh.read(size),
                max_output_size=header.channel_count * header.frame_count)
            count = len(raw) // header.channel_count
            block = np.frombuffer(
                raw[:count * header.channel_count], dtype=np.uint8
            ).reshape(count, header.channel_count)
            for row in block:
                if emitted >= header.frame_count:
                    return
                emitted += 1
                yield row
        if emitted < header.frame_count:
            raise FseqError(f"{path.name}: {emitted} frames decoded, header "
                            f"promises {header.frame_count}")


def read_all(path: str | Path) -> tuple[FseqHeader, np.ndarray]:
    """Whole file as (header, (frames, channels) uint8 array)."""
    header = read_header(path)
    data = np.stack(list(read_frames(path))) if header.frame_count else np.zeros(
        (0, header.channel_count), dtype=np.uint8
    )
    return header, data


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        h = read_header(arg)
        print(f"{arg}")
        print(f"  v{h.version[0]}.{h.version[1]}  {h.channel_count} channels x "
              f"{h.frame_count} frames @ {h.step_time_ms} ms ({h.fps:g} fps, "
              f"{h.duration_s:.1f} s)")
        print(f"  compression={h.compression_type}  data at {h.data_offset}")
        for code, data in h.variable_headers.items():
            print(f"  [{code}] {data[:80]!r}")
