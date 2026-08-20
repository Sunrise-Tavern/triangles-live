"""A Falcon that isn't there: receive DDP, reassemble frames, write ``.fseq``.

This closes the loop on a Mac with no hardware.  Point the engine's DDP sender
at ``127.0.0.1`` and this stands in for the controller:

* reassembles frames from the packet stream (push flag ends a frame),
* reports what arrived -- packet loss, channels never written, frame rate --
  because a silent renderer bug looks exactly like a working one,
* writes an ``.fseq`` you can open in xLights against the real layout.

xLights is the visual oracle from here on: ``--checksequence`` for validity,
the 3D preview for "does it look right", and ``--fseqcmp`` to compare a capture
against a reference file channel-for-channel.

Run it standalone::

    ./live.sh fake-falcon --out out/capture.fseq --idle 3
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .ddp import DDP_PORT, FrameAssembler
from .fseq import FseqWriter
from .layout import Layout, load_layout


@dataclass
class CaptureStats:
    frames: int = 0
    packets: int = 0
    dropped_packets: int = 0
    out_of_range: int = 0
    first_frame_at: float = 0.0
    last_frame_at: float = 0.0
    #: Channels that were never non-zero in any frame -- usually a model the
    #: renderer forgot, or a start-channel that is off by one.
    lit_channels: int = 0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.last_frame_at - self.first_frame_at)

    @property
    def fps(self) -> float:
        return (self.frames - 1) / self.duration_s if self.duration_s > 0 else 0.0


@dataclass
class FakeFalcon:
    layout: Layout
    port: int = DDP_PORT
    bind: str = "0.0.0.0"
    out: Path | None = None
    fps: float = 40.0
    media_file: str | None = None
    quiet: bool = False
    stats: CaptureStats = field(default_factory=CaptureStats)

    def __post_init__(self) -> None:
        self.assembler = FrameAssembler(self.layout.channel_count)
        self._ever_lit = np.zeros(self.layout.channel_count, dtype=bool)
        self._writer: FseqWriter | None = None
        self.last_frame: np.ndarray | None = None

    # -- lifecycle -------------------------------------------------------- #

    def open(self) -> None:
        if self.out is not None:
            self.out = Path(self.out)
            self.out.parent.mkdir(parents=True, exist_ok=True)
            self._writer = FseqWriter(
                self.out,
                channel_count=self.layout.channel_count,
                step_time_ms=int(round(1000.0 / self.fps)),
                media_file=self.media_file,
            )

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self.stats.lit_channels = int(self._ever_lit.sum())

    def handle(self, datagram: bytes) -> np.ndarray | None:
        frame = self.assembler.feed(datagram)
        if frame is None:
            return None
        now = time.perf_counter()
        if self.stats.frames == 0:
            self.stats.first_frame_at = now
        self.stats.last_frame_at = now
        self.stats.frames += 1
        self.last_frame = frame
        np.logical_or(self._ever_lit, frame > 0, out=self._ever_lit)
        if self._writer is not None:
            self._writer.add_frame(frame)
        return frame

    def run(
        self,
        duration: float | None = None,
        idle_timeout: float | None = 5.0,
        ready: "threading.Event | None" = None,
    ) -> CaptureStats:
        """Receive until ``duration`` elapses or nothing arrives for ``idle_timeout``.

        ``ready`` is set once the socket is bound, so a caller running this in
        a thread can start sending without racing the bind.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # A frame is 26 packets; a generous buffer costs nothing and stops the
        # kernel dropping a burst while we are writing to disk.
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        except OSError:
            pass
        sock.bind((self.bind, self.port))
        sock.settimeout(0.25)

        self.open()
        if ready is not None:
            ready.set()
        started = time.perf_counter()
        last_packet = started
        self._log(f"listening on {self.bind}:{self.port} for "
                  f"{self.layout.channel_count} channels")
        try:
            while True:
                now = time.perf_counter()
                if duration is not None and now - started >= duration:
                    break
                if idle_timeout is not None and self.stats.frames and \
                        now - last_packet > idle_timeout:
                    self._log(f"idle for {idle_timeout:g}s, stopping")
                    break
                try:
                    datagram, _addr = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                last_packet = time.perf_counter()
                self.handle(datagram)
        except KeyboardInterrupt:
            self._log("interrupted")
        finally:
            sock.close()
            self.stats.packets = self.assembler.packets
            self.stats.dropped_packets = self.assembler.dropped_packets
            self.stats.out_of_range = self.assembler.out_of_range
            self.close()
        return self.stats

    # -- reporting -------------------------------------------------------- #

    def _log(self, message: str) -> None:
        if not self.quiet:
            print(f"[fake-falcon] {message}", file=sys.stderr, flush=True)

    def report(self) -> str:
        s = self.stats
        lines = [
            f"frames        : {s.frames} in {s.duration_s:.1f}s ({s.fps:.1f} fps)",
            f"packets       : {s.packets}"
            + (f"  DROPPED {s.dropped_packets}" if s.dropped_packets else ""),
        ]
        if s.out_of_range:
            lines.append(
                f"out of range  : {s.out_of_range} channels past "
                f"{self.layout.channel_count} -- the sender thinks the show is bigger"
            )
        lines.append(
            f"channels lit  : {s.lit_channels} / {self.layout.channel_count}"
        )
        lines.append("")
        lines.append("per model (channels that were ever non-zero):")
        for model in sorted(self.layout.models.values(), key=lambda m: m.start):
            lit = int(self._ever_lit[model.slice].sum())
            mark = "  " if lit else "!!"
            lines.append(
                f" {mark} {model.name:<24} {lit:>6} / {model.channels:<6}"
                + ("" if lit else "   never lit")
            )
        if self.out is not None:
            lines.append("")
            lines.append(f"wrote {self.out}")
        return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("out/capture.fseq"),
                    help="fseq file to write (default: out/capture.fseq)")
    ap.add_argument("--no-out", action="store_true", help="receive but write nothing")
    ap.add_argument("--port", type=int, default=DDP_PORT)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--fps", type=float, default=40.0,
                    help="frame rate to stamp into the fseq (default: 40)")
    ap.add_argument("--media", help="audio file to reference from the fseq")
    ap.add_argument("--duration", type=float, help="stop after N seconds")
    ap.add_argument("--idle", type=float, default=5.0,
                    help="stop after N seconds with no packets (default: 5)")
    args = ap.parse_args(argv)

    falcon = FakeFalcon(
        layout=load_layout(), port=args.port, bind=args.bind,
        out=None if args.no_out else args.out,
        fps=args.fps, media_file=args.media,
    )
    falcon.run(duration=args.duration, idle_timeout=args.idle)
    print(falcon.report())
    return 0 if falcon.stats.frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
