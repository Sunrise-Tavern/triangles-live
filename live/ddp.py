"""DDP (Distributed Display Protocol) sender and packet parser.

DDP is what the Falcon F16V5 is already configured for in this show
(``xlights_networks.xml``: ``Protocol="DDP" ChannelsPerPacket="1440"``), so the
live engine speaks it rather than E1.31 -- one UDP socket, no universe
bookkeeping, and channel offsets are plain byte offsets into the controller's
channel space.

Header, 10 bytes, big-endian where multi-byte::

    byte 0   flags:  V V x T S R Q P
             VV=01 -> version 1; T -> timecode follows; P -> push (end of frame)
    byte 1   sequence number, 1..15 (0 means "not sequenced")
    byte 2   data type:  C R TTT SSS   (TTT=1 RGB, SSS=3 eight-bit)
    byte 3   destination id (1 = the device's default output)
    byte 4-7 data offset, in bytes, from the start of the channel space
    byte 8-9 data length, in bytes

Two details that matter and are easy to miss:

* **The push flag goes on a frame's last packet only.**  Set it on every
  packet and the controller latches mid-frame, which shows up as tearing on
  fast content -- exactly the content this rig runs.
* **Offsets are 0-based bytes, xLights channels are 1-based.**  Channel 1 is
  offset 0.  Everything in this module is 0-based; :mod:`live.layout` is where
  the 1-based channel numbers live.
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

import numpy as np

DDP_PORT = 4048
HEADER_LEN = 10

FLAG_VERSION_1 = 0x40
FLAG_TIMECODE = 0x10
FLAG_PUSH = 0x01

#: type byte: customer=0, reserved=0, type=1 (RGB), pixel size=3 (8 bit)
TYPE_RGB8 = 0x0B
ID_DEFAULT_OUTPUT = 1

#: Matches the Falcon's configured ChannelsPerPacket.  Also a multiple of 3,
#: so a packet never splits a pixel -- not required by the protocol, but it
#: makes packet dumps readable.
DEFAULT_CHANNELS_PER_PACKET = 1440


@dataclass
class DDPPacket:
    sequence: int
    offset: int
    length: int
    push: bool
    data: bytes

    @property
    def end(self) -> int:
        return self.offset + self.length


def parse(datagram: bytes) -> DDPPacket:
    """Parse one datagram.  Raises ValueError on anything malformed."""
    if len(datagram) < HEADER_LEN:
        raise ValueError(f"short datagram: {len(datagram)} bytes")
    flags, sequence, _dtype, _dest, offset, length = struct.unpack(
        ">BBBBIH", datagram[:HEADER_LEN]
    )
    if flags & 0xC0 != FLAG_VERSION_1:
        raise ValueError(f"unsupported DDP version, flags=0x{flags:02x}")
    body = datagram[HEADER_LEN:]
    if flags & FLAG_TIMECODE:
        body = body[4:]
    if len(body) < length:
        raise ValueError(f"datagram claims {length} bytes, carries {len(body)}")
    return DDPPacket(
        sequence=sequence & 0x0F,
        offset=offset,
        length=length,
        push=bool(flags & FLAG_PUSH),
        data=body[:length],
    )


class DDPSender:
    """Sends one frame per call, split across as many packets as it needs.

    Stateless apart from the sequence counter, so it is safe to hold open for
    the life of the daemon and cheap to re-point at a different host.
    """

    def __init__(
        self,
        host: str,
        port: int = DDP_PORT,
        channels_per_packet: int = DEFAULT_CHANNELS_PER_PACKET,
        dest_id: int = ID_DEFAULT_OUTPUT,
        sock: socket.socket | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.channels_per_packet = int(channels_per_packet)
        self.dest_id = int(dest_id)
        self._seq = 0
        self._own_sock = sock is None
        self.sock = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Frames are disposable: never block the render loop on the socket.
        self.sock.setblocking(False)
        self.packets_sent = 0
        self.frames_sent = 0

    def _next_seq(self) -> int:
        self._seq = self._seq % 15 + 1
        return self._seq

    def send_frame(self, channels: np.ndarray | bytes, offset: int = 0) -> int:
        """Send ``channels`` starting at 0-based channel ``offset``.

        Returns the number of packets sent.  The last one carries the push
        flag, so the controller latches exactly one complete frame.
        """
        data = channels.tobytes() if isinstance(channels, np.ndarray) else bytes(channels)
        total = len(data)
        if total == 0:
            return 0
        sent = 0
        pos = 0
        while pos < total:
            chunk = data[pos:pos + self.channels_per_packet]
            last = pos + len(chunk) >= total
            flags = FLAG_VERSION_1 | (FLAG_PUSH if last else 0)
            header = struct.pack(
                ">BBBBIH", flags, self._next_seq(), TYPE_RGB8, self.dest_id,
                offset + pos, len(chunk),
            )
            try:
                self.sock.sendto(header + chunk, (self.host, self.port))
            except BlockingIOError:
                # Kernel buffer full: drop this packet rather than stall the
                # render loop.  The next frame is 25 ms away and complete.
                pass
            sent += 1
            pos += len(chunk)
        self.packets_sent += sent
        self.frames_sent += 1
        return sent

    def close(self) -> None:
        if self._own_sock:
            self.sock.close()

    def __enter__(self) -> "DDPSender":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class FrameAssembler:
    """Rebuilds whole frames from a stream of DDP packets.

    Used by the fake Falcon, and by anything that wants to watch real output.
    A frame is complete when a packet arrives with the push flag set; whatever
    the buffer holds at that moment is the frame.  That mirrors what a real
    controller does, including when packets go missing.
    """

    def __init__(self, channel_count: int) -> None:
        self.channel_count = int(channel_count)
        self.buffer = np.zeros(self.channel_count, dtype=np.uint8)
        self.packets = 0
        self.frames = 0
        self.dropped_packets = 0       # gaps in the sequence numbers
        self.out_of_range = 0          # data past channel_count
        self._last_seq: int | None = None
        self._touched = 0

    def feed(self, datagram: bytes) -> np.ndarray | None:
        """Absorb one datagram; return a frame copy when one completes."""
        packet = parse(datagram)
        self.packets += 1

        if packet.sequence and self._last_seq is not None:
            expected = self._last_seq % 15 + 1
            if packet.sequence != expected:
                self.dropped_packets += (packet.sequence - expected) % 15
        if packet.sequence:
            self._last_seq = packet.sequence

        start = packet.offset
        end = min(packet.end, self.channel_count)
        if start < self.channel_count and end > start:
            self.buffer[start:end] = np.frombuffer(
                packet.data[: end - start], dtype=np.uint8
            )
            self._touched += end - start
        if packet.end > self.channel_count:
            self.out_of_range += packet.end - self.channel_count

        if packet.push:
            self.frames += 1
            self._touched = 0
            return self.buffer.copy()
        return None
