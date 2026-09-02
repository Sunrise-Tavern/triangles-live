"""M1 verification: does a frame survive the whole loop unchanged?

Runs on a Mac with no hardware and no xLights.  It checks the three things
M1 claims, in order of how much they would cost to debug later:

1. the channel map is self-consistent (nothing overlaps, nothing is orphaned),
2. an ``.fseq`` written here reads back byte-identical,
3. a frame sent over a real UDP socket, reassembled by the fake Falcon and
   written to ``.fseq``, is byte-identical to the frame that went in.

(3) is the one that matters: it exercises packet splitting, the push flag,
sequence numbers, offsets and the file writer together.  If the pattern
survives it, anything wrong downstream is a *rendering* question, which
xLights' 3D preview answers.

    ./live.sh selftest
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import struct
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from . import ddp, effects as fx, fseq
from . import palette as pal
from . import settings as knobs
from .fake_falcon import FakeFalcon
from .frame import Canvas
from .layout import STRING_ORDER, Layout, load_layout
from .script import Script
from .testpattern import frame as pattern_frame

LOOPBACK_PORT = 14048  # not 4048: never fight a real listener during a test


class Failure(AssertionError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


# --------------------------------------------------------------------------- #


def test_layout(layout: Layout) -> str:
    models = sorted(layout.models.values(), key=lambda m: m.start)

    occupied: dict[int, str] = {}
    for model in models:
        for ch in range(model.start, model.end + 1):
            if ch in occupied:
                raise Failure(
                    f"channel {ch} claimed by both {occupied[ch]!r} and {model.name!r}"
                )
            occupied[ch] = model.name
    check(len(occupied) == layout.channel_count,
          f"{layout.channel_count - len(occupied)} channels in the range belong to "
          "no model -- a gap in the addressing")

    check(len(layout.arches) == 24, f"expected 24 arches, found {len(layout.arches)}")
    check(len(layout.nets) == 7,
          f"expected 7 nets, found {len(layout.nets)} ({layout.nets}) -- an "
          "overlay (shadow) model must be skipped, not adopted")
    check(all(layout[n].kind == "arch" for n in layout.arches), "Tunnel holds a non-arch")
    sizes = sorted({layout[n].nodes for n in layout.nets})
    check(sizes == [435, 465], f"nets should be 435 and 465 nodes, got {sizes}")

    # Each family's wire order must follow its StringType exactly.  Both are
    # "RGB Nodes" in the 2026-08-30 layout -- the Falcon does the swap now,
    # from the colour order xLights uploaded to each port -- where the arches
    # used to be "GRB Nodes" with xLights swapping.  If the arches come out
    # with red and green exchanged, the Falcon's port config is stale, not
    # this.
    orders = {}
    for name in (*layout.nets, *layout.arches):
        m = layout[name]
        want = STRING_ORDER[m.source.get("StringType", "RGB Nodes")]
        check(m.order == want, f"{name}: order {m.order} does not follow its "
                               f"StringType {m.source.get('StringType')!r}")
        orders[m.kind] = m.order

    # A red frame must land in the red channel of each fixture, whatever its
    # wire order.  This is the assertion that catches a swapped permutation.
    out = layout.blank_channels()
    for model in models:
        rgb = np.zeros((model.nodes, 4), dtype=np.uint8)
        rgb[:, 0] = 255
        model.pack(rgb, out)
    for name in (*layout.nets, *layout.arches):
        m = layout[name]
        slot = m.order.index(0)
        check(out[m.start - 1 + slot] == 255
              and all(out[m.start - 1 + i] == 0 for i in range(3) if i != slot),
              f"{name}: red landed on the wrong channel")
    if layout.par:
        par = layout[layout.par]
        check(out[par.start - 1] == 255, "par: red is not on DMX slot 1")

    return (f"{len(models)} models, {layout.channel_count} channels, "
            f"no overlaps or gaps; nets {orders['net']}, arches {orders['arch']}, "
            f"{'par' if layout.par else 'no par'}")


def test_clips(layout: Layout) -> str:
    """Clips load lazily, loop, and refuse a file from another layout."""
    import time as _time

    import zstandard

    from . import fseq
    from .clips import Clips

    frames = np.arange(24 * 90, dtype=np.uint32).reshape(24, 90) % 256
    frames = frames.astype(np.uint8)

    def v2_header(channels, count, comp, blocks, data_offset):
        return struct.pack("<4sHBBHIIBBBBBBQ", b"PSEQ", data_offset, 0, 2,
                           HEADER := 32, channels, count, 50, 0,
                           comp | ((blocks >> 8) << 4), blocks & 0xFF, 0, 0, 7)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # An uncompressed clip, via the writer the engine already trusts.
        with fseq.FseqWriter(root / "plain.fseq", 90, step_time_ms=50) as w:
            for row in frames:
                w.add_frame(row)
        # The same frames zstd-compressed in two blocks, as xLights writes.
        half = frames[:12].tobytes(), frames[12:].tobytes()
        blobs = [zstandard.ZstdCompressor().compress(b) for b in half]
        table = b"".join(struct.pack("<II", first, len(blob))
                         for first, blob in zip((0, 12), blobs))
        (root / "packed.fseq").write_bytes(
            v2_header(90, 24, fseq.COMPRESSION_ZSTD, 2, 32 + len(table))
            + table + b"".join(blobs))
        # A clip rendered against some other rig.
        with fseq.FseqWriter(root / "othermap.fseq", 33, step_time_ms=50) as w:
            w.add_frame(np.zeros(33, dtype=np.uint8))

        for name in ("plain", "packed"):
            _, back = fseq.read_all(root / f"{name}.fseq")
            check(np.array_equal(back, frames), f"{name}: frames differ after read")

        clips = Clips(root, channel_count=90)
        check(clips.names == ["packed", "plain"],
              f"expected [packed, plain], got {clips.names} -- the 33-channel "
              "file must be skipped, not offered")
        check(clips.get("plain") is None, "get() must not block on first ask")
        for _ in range(100):
            clip = clips.get("plain")
            if clip is not None:
                break
            _time.sleep(0.05)
        check(clip is not None, "the loader thread never finished")
        check(np.array_equal(clip.frame_at(0.0), frames[0])
              and np.array_equal(clip.frame_at(0.10), frames[2])
              and np.array_equal(clip.frame_at(24 * 0.05 + 0.05), frames[1]),
              "frame_at does not step and loop at the clip's own fps")
        # from_channels must be the gather backwards, colour order and all.
        canvas = Canvas(layout)
        rng = np.random.default_rng(3)
        wire = rng.integers(0, 256, layout.channel_count, dtype=np.uint8)
        canvas.from_channels(wire)
        back = canvas.to_channels()
        check(np.array_equal(back, wire),
              "from_channels -> to_channels is not the identity")

        # The energy index ranks clips and hands each state a sensible band.
        clips.build_index()
        vocab = {k: clips.vocabulary(k) for k in ("quiet", "hot")}
        check(all(vocab.values()), f"empty vocabulary from the index: {vocab}")

        # And with clips in its hand the arranger plays one: same audio,
        # clip_share 1, the look must become a clip and stay deterministic.
        from .arranger import Arranger
        from .audio import ArraySource
        from .listener import Listener
        from .settings import Settings
        from .state import StateMachine
        from .verify import arc_track

        with fseq.FseqWriter(root / "rig.fseq", layout.channel_count,
                             step_time_ms=50) as w:
            for i in range(20):
                w.add_frame(np.full(layout.channel_count, 40 + 10 * (i % 4),
                                    dtype=np.uint8))
        rig_clips = Clips(root, channel_count=layout.channel_count)
        rig_clips.build_index()
        rig_clips.get("rig")
        for _ in range(100):
            if rig_clips.get("rig") is not None:
                break
            _time.sleep(0.05)
        check(rig_clips.get("rig") is not None, "rig clip never loaded")

        audio, _ = arc_track()

        def run() -> tuple[list, np.ndarray]:
            cv = Canvas(layout)
            listener = Listener(ArraySource(audio))
            st = Settings()
            st.clip_share = 1.0
            arranger = Arranger(cv, listener, state=StateMachine(),
                                settings=st, clips=rig_clips)
            out = np.zeros(layout.channel_count, dtype=np.uint8)
            looks, sample, now, index = [], [], 0.0, 0
            for block in listener.source.blocks():
                f = listener.step(block)
                arranger.machine.push(f)
                while now <= f.t:
                    arranger.render(index, now)
                    if index % 97 == 0:
                        sample.append(cv.to_channels(out).copy())
                    if arranger._look is not None:
                        looks.append(arranger._look[1])
                    index += 1
                    now += 1.0 / 40.0
            return looks, np.stack(sample)

        looks_a, sample_a = run()
        check(any(n.startswith("clip:") for n in looks_a),
              "clip_share 1 never played a clip")
        looks_b, sample_b = run()
        check(np.array_equal(sample_a, sample_b),
              "the show with clips is not deterministic")
    return ("zstd and plain fseq round-trip, lazy load off the render "
            "thread, wrong-layout clip refused, loop wraps; from_channels "
            "is the gather backwards; the rotation plays a clip and stays "
            "deterministic")


def test_pieces(layout: Layout) -> str:
    """Each showpiece must light tunnel and triangles, and stay finite."""
    from .arranger import PIECES, Arranger
    from .audio import ArraySource
    from .listener import Listener
    from .settings import Settings
    from .state import StateMachine
    from .verify import arc_track

    audio, _ = arc_track()
    audio = audio[: 44100 * 18]
    details = []
    for name in PIECES:
        canvas = Canvas(layout)
        listener = Listener(ArraySource(audio))
        st = Settings()
        st.piece = name
        arranger = Arranger(canvas, listener, state=StateMachine(), settings=st)
        arches = nets = 0.0
        now, index = 0.0, 0
        for block in listener.source.blocks():
            f = listener.step(block)
            arranger.machine.push(f)
            while now <= f.t:
                arranger.render(index, now)
                check(np.isfinite(canvas._source).all(), f"{name}: NaN or inf")
                arches = max(arches, float(canvas.arches.max()))
                nets = max(nets, float(canvas.nets.max()))
                index += 1
                now += 1.0 / 40.0
        check(arranger._look is not None and arranger._look[1] == f"piece:{name}",
              f"{name}: the Piece knob did not hold the look "
              f"({arranger._look and arranger._look[1]})")
        check(arches > 0.5, f"{name}: the tunnel never lit ({arches:.2f})")
        check(nets > 0.5, f"{name}: the triangles never lit ({nets:.2f})")
        details.append(name)
    return f"{', '.join(details)}: tunnel and triangles both lit, held via the knob"


def test_orient(layout: Layout) -> str:
    """The orientation bands must rise base -> apex and run front -> back."""
    from . import orient

    canvas = Canvas(layout)
    mask = canvas.net_mask

    def lit_net_y(t: float, big: bool) -> float:
        orient.paint(canvas, t)
        if big:
            nets, y, m = canvas.nets[canvas.big], canvas.big_geo.y, mask[canvas.big]
        else:
            nets, y, m = canvas.nets, canvas.net_y, mask
        bright = (nets.max(axis=-1) > 0.5) & m
        check(bright.any(), f"nothing lit at {t:.1f}s")
        return float(y[bright].mean())

    early, late = lit_net_y(orient.hold("nets up", 0.6), False), lit_net_y(orient.hold("nets up", 4.4), False)
    check(early > 0.8 and late < 0.2,
          f"nets band runs {early:.2f} -> {late:.2f}; want bottom (1) -> top (0)")

    # ...and "up" must be world up on *every* net, whatever its rotation in
    # xLights.  Counted in grid rows it was sideways on the nets rotated
    # -90 about Z and downward on the ones flipped 180 about X.
    from .geometry import world_positions
    positions = world_positions(layout)
    for i, name in enumerate(canvas.net_names):
        heights = []
        for t in (0.6, 2.5, 4.4):
            orient.paint(canvas, orient.hold("nets up", t))
            bright = (canvas.nets[i].max(axis=-1) > 0.5) & mask[i]
            heights.append(float(positions[name][bright[:canvas.net_nodes[i]], 1].mean()))
        check(heights[0] < heights[1] < heights[2],
              f"{name}: the band does not climb in world Y ({[round(h) for h in heights]})")
    early, late = lit_net_y(orient.hold("big up", 0.6), True), lit_net_y(orient.hold("big up", 4.4), True)
    check(early > 0.8 and late < 0.2,
          f"big band runs {early:.2f} -> {late:.2f}; want base (1) -> apex (0)")

    def lit_arch(t: float) -> int:
        orient.paint(canvas, t)
        return int(np.argmax(canvas.arches.max(axis=(1, 2))))

    first, last = lit_arch(orient.hold("tunnel", 0.3)), lit_arch(orient.hold("tunnel", 5.7))
    check(first == 0 and last == len(layout.arches) - 1,
          f"tunnel runs arch {first} -> {last}; want 0 -> {len(layout.arches) - 1}")
    return (f"{len(orient.STAGES)} stages: nets and the big triangle rise base -> apex, "
            f"tunnel runs front -> back over {len(layout.arches)} arches")


def test_fseq_roundtrip(layout: Layout) -> str:
    rng = np.random.default_rng(1)
    frames = [
        rng.integers(0, 256, layout.channel_count, dtype=np.uint8) for _ in range(7)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "roundtrip.fseq"
        with fseq.FseqWriter(path, layout.channel_count, step_time_ms=25,
                             media_file="track.mp3") as writer:
            for f in frames:
                writer.add_frame(f)
        header, data = fseq.read_all(path)
        check(header.channel_count == layout.channel_count, "channel count changed")
        check(header.frame_count == len(frames),
              f"frame count {header.frame_count} != {len(frames)}")
        check(header.step_time_ms == 25, "step time changed")
        check(header.variable_headers.get("mf", b"").startswith(b"track.mp3"),
              "media file header missing")
        check(np.array_equal(data, np.stack(frames)), "frame data differs")
        size = path.stat().st_size
    return (f"{len(frames)} frames x {layout.channel_count} channels "
            f"({size} bytes) identical after write/read")


def test_ddp_packets(layout: Layout) -> str:
    """Split and reassemble in-process -- no socket, so no loss to hide behind."""
    captured: list[bytes] = []

    class Recorder:
        def sendto(self, data, _addr):
            captured.append(data)

        def setblocking(self, _flag):
            pass

        def close(self):
            pass

    sender = ddp.DDPSender("0.0.0.0", channels_per_packet=1440, sock=Recorder())
    assembler = ddp.FrameAssembler(layout.channel_count)

    source = [pattern_frame(layout, t) for t in (0.0, 7.0, 13.0, 22.0)]
    got: list[np.ndarray] = []
    for f in source:
        captured.clear()
        packets = sender.send_frame(f)
        expected = -(-layout.channel_count // 1440)
        check(packets == expected, f"{packets} packets, expected {expected}")
        pushes = [ddp.parse(p).push for p in captured]
        check(pushes[-1] and not any(pushes[:-1]),
              "the push flag must be on the last packet only")
        for p in captured:
            frame = assembler.feed(p)
            if frame is not None:
                got.append(frame)

    check(len(got) == len(source), f"got {len(got)} frames, sent {len(source)}")
    check(assembler.dropped_packets == 0,
          f"assembler saw {assembler.dropped_packets} sequence gaps")
    for i, (a, b) in enumerate(zip(source, got)):
        check(np.array_equal(a, b), f"frame {i} differs after reassembly")
    return (f"{len(source)} frames split into "
            f"{-(-layout.channel_count // 1440)} packets each, reassembled exactly")


def test_loopback(layout: Layout, seconds: float = 2.0, fps: float = 40.0) -> str:
    """The real thing: UDP over the loopback into the fake Falcon, then fseq."""
    count = int(seconds * fps)
    source = [pattern_frame(layout, i / fps) for i in range(count)]

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "loopback.fseq"
        falcon = FakeFalcon(layout=layout, port=LOOPBACK_PORT, bind="127.0.0.1",
                            out=out, fps=fps, quiet=True)
        ready = threading.Event()
        thread = threading.Thread(
            target=falcon.run,
            kwargs=dict(duration=seconds + 5.0, idle_timeout=1.0, ready=ready),
            daemon=True,
        )
        thread.start()
        check(ready.wait(5.0), "fake Falcon never bound its socket")

        # Blocking socket here: this test is about correctness, not throughput,
        # and a dropped packet would make the comparison meaningless.
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender = ddp.DDPSender("127.0.0.1", port=LOOPBACK_PORT, sock=sock)
        sender.sock.setblocking(True)
        started = time.perf_counter()
        for f in source:
            sender.send_frame(f)
            time.sleep(1.0 / fps / 4)   # pace it; a burst overruns the receive buffer
        elapsed = time.perf_counter() - started
        sender.close()

        thread.join(timeout=10.0)
        check(not thread.is_alive(), "fake Falcon did not stop")

        check(falcon.stats.frames == count,
              f"fake Falcon saw {falcon.stats.frames} frames, sent {count}")
        check(falcon.stats.dropped_packets == 0,
              f"{falcon.stats.dropped_packets} packets lost on the loopback")
        check(falcon.stats.out_of_range == 0,
              f"{falcon.stats.out_of_range} channels addressed past the show")

        header, data = fseq.read_all(out)
        check(header.frame_count == count, "fseq frame count wrong")
        expected = np.stack(source)
        check(np.array_equal(data, expected), "captured frames differ from what was sent")

        lit = int(falcon._ever_lit.sum())
        dark = [m.name for m in layout.models.values()
                if not falcon._ever_lit[m.slice].any()]

    detail = f"{count} frames over UDP in {elapsed:.2f}s, byte-identical in the fseq"
    if dark:
        detail += f"; {len(dark)} model(s) dark in this window ({dark[0]}...)"
    else:
        detail += f"; every model lit ({lit} channels)"
    return detail


def test_canvas(layout: Layout) -> str:
    """The renderer's buffers must land on the same channels M1 verified."""
    canvas = Canvas(layout)
    canvas.nets[:] = (1.0, 0.0, 0.0)
    canvas.arches[:] = (1.0, 0.0, 0.0)
    if canvas.par.size:
        canvas.par[:3] = (1.0, 0.0, 0.0)
    frame = canvas.to_channels()

    for name in (*layout.nets, *layout.arches):
        m = layout[name]
        want = [0, 0, 0]
        want[m.order.index(0)] = 255
        check(list(frame[m.slice][:3]) == want,
              f"{name}: red frame is not red on the wire ({frame[m.slice][:3]})")
        check(frame[m.slice].reshape(-1, 3)[:, m.order.index(0)].min() == 255,
              f"{name}: not every node received the frame -- padding leaked")
    if layout.par:
        par = layout[layout.par]
        check(tuple(frame[par.slice]) == (255, 0, 0, 0), "par: red is not on slot 1")
    # Nothing outside the models may be written: the padded net slots must
    # stay in the buffer.
    check(int((frame > 0).sum()) == sum(layout[n].nodes for n in (*layout.nets, *layout.arches))
          + (1 if layout.par else 0),
          "the frame lit channels that belong to no model")

    # A cleared canvas must be all-off, and painting outside 0..1 must clamp
    # rather than wrap -- an overflowing uint8 shows as a dark flicker.
    canvas.clear()
    check(not canvas.to_channels().any(), "cleared canvas is not black")
    canvas.nets[:] = 4.0
    canvas.arches[:] = -2.0
    frame = canvas.to_channels()
    check(frame[layout[layout.nets[0]].slice].max() == 255, "over-bright did not clamp")
    check(frame[layout[layout.arches[0]].slice].max() == 0, "negative did not clamp")

    # The grade must desaturate to gray, spread about the midpoint, and be a
    # strict no-op at neutral -- the fseq oracle depends on the last one.
    canvas.clear()
    canvas.nets[:] = (0.8, 0.2, 0.4)
    canvas.arches[:] = (0.1, 0.6, 0.3)
    before = canvas._source.copy()
    canvas.grade(1.0, 1.0)
    check(np.array_equal(canvas._source, before), "a neutral grade changed pixels")
    canvas.grade(0.0, 1.0)
    check(np.allclose(canvas.nets[..., 0], canvas.nets[..., 1])
          and np.allclose(canvas.nets[..., 1], canvas.nets[..., 2])
          and np.allclose(canvas.arches[..., 0], canvas.arches[..., 2]),
          "saturation 0 should be grayscale")
    canvas.nets[:] = (0.8, 0.2, 0.4)
    canvas.grade(1.0, 1.4)
    check(float(canvas.nets[..., 0].max()) > 0.9 and float(canvas.nets[..., 1].min()) < 0.1,
          f"contrast 1.4 should spread 0.8/0.2 apart, got "
          f"{canvas.nets[0, 0].round(2)}")

    # Groups must stay contiguous, or an effect aimed at "the small nets" would
    # silently paint a copy and vanish; and between them they must cover
    # every net exactly once.
    big, small = canvas.net_pair()
    covered = sorted([*range(big.start, big.stop), *range(small.start, small.stop)])
    check(covered == list(range(len(layout.nets))),
          f"big {big} + small {small} do not partition the {len(layout.nets)} nets")
    # The big triangle as one surface: its four nets must tile the 0..1 frame
    # -- a top net in the upper half, two base nets in the lower corners, and
    # the inverted one holding the centre -- or big_* gestures paint four
    # unrelated small triangles.
    G, mask = canvas.big_geo, canvas.net_mask[big]
    spans = {}
    for i, name in enumerate(canvas.net_names[big]):
        x, y = G.x[i][mask[i]], G.y[i][mask[i]]
        spans[name] = (float(x.mean()), float(y.mean()), float(G.r[i][mask[i]].min()))
    tops = [n for n, (_, y, _) in spans.items() if y < 0.35]
    centre = [n for n, (_, _, r) in spans.items() if r < 0.05]
    left = [n for n, (x, y, _) in spans.items() if x < 0.35 and y > 0.6]
    right = [n for n, (x, y, _) in spans.items() if x > 0.65 and y > 0.6]
    check(len(tops) == 1 and len(centre) == 1 and len(left) == 1 and len(right) == 1,
          f"big triangle frame is not top/centre/left/right: {spans}")
    allx = np.concatenate([G.x[i][mask[i]] for i in range(mask.shape[0])])
    ally = np.concatenate([G.y[i][mask[i]] for i in range(mask.shape[0])])
    check(allx.min() == 0.0 and allx.max() == 1.0 and ally.min() == 0.0 and ally.max() == 1.0,
          "big triangle frame does not span 0..1")
    # Every net as one surface: the frame must span 0..1 both ways, and the
    # left-to-right order must be the order of the nets' x in it.
    A = canvas.all_geo
    ax = np.concatenate([A.x[i][mask_all[i]] for i in range(len(layout.nets))]) if (mask_all := canvas.net_mask) is not None else None
    ay = np.concatenate([A.y[i][mask_all[i]] for i in range(len(layout.nets))])
    check(ax.min() == 0.0 and ax.max() == 1.0 and ay.min() == 0.0 and ay.max() == 1.0,
          "all-nets frame does not span 0..1")
    means = [float(A.x[i][mask_all[i]].mean()) for i in range(len(layout.nets))]
    check(canvas.net_order == sorted(range(len(layout.nets)), key=means.__getitem__),
          f"net_order {canvas.net_order} is not left-to-right by x")
    check(A.aspect > 2.0, f"the array should be much wider than tall (aspect {A.aspect:.2f})")
    return (f"{canvas.nets.shape} nets ({len(layout.nets)}, padded) + "
            f"{canvas.arches.shape} arches -> {layout.channel_count} channels, "
            f"big {big.start}-{big.stop - 1} ({tops[0]} top, {centre[0]} centre), "
            f"small {small.start}-{small.stop - 1}, array {A.aspect:.1f}:1, "
            f"colour order and clamping hold")


def test_effects(layout: Layout) -> str:
    """Every effect must paint something, stay finite, and stay in range."""
    canvas = Canvas(layout)
    palette = pal.generate(200.0, "triadic").floored()
    n = len(canvas.arch_names)

    for name, pattern in fx.PATTERNS.items():
        peak = 0.0
        for step in range(41):
            level = pattern(n, step / 40.0)
            check(level.shape == (n,), f"{name}: returned {level.shape}, want ({n},)")
            check(np.isfinite(level).all(), f"{name}: produced non-finite levels")
            check(level.min() >= -1e-6 and level.max() <= 1.0 + 1e-6,
                  f"{name}: levels outside 0..1 ({level.min():.2f}..{level.max():.2f})")
            peak = max(peak, float(level.max()))
        check(peak > 0.5, f"{name}: never lights anything (peak {peak:.2f})")

    # Sparkle is random but must be reproducible, or a capture stops being
    # comparable with a re-render.
    check(np.array_equal(fx.sparkle(n, 0.3, seed=5), fx.sparkle(n, 0.3, seed=5)),
          "corridor sparkle is not reproducible")
    for name in fx.SEEDED:
        check(name in fx.PATTERNS, f"SEEDED names {name}, which is not a pattern")
        check(not np.array_equal(fx.PATTERNS[name](n, 0.3, seed=1),
                                 fx.PATTERNS[name](n, 0.3, seed=2)),
              f"{name}: the seed changes nothing")
    # Every gesture the arranger may name must be one it can paint.
    from .arranger import NET_GESTURES, SCHEME_OPTIONS
    for kind, options in SCHEME_OPTIONS.items():
        for scheme in options:
            check(scheme in pal.SCHEMES, f"{kind} names unknown scheme {scheme!r}")
    gestures = sorted({g for names in NET_GESTURES.values() for g in names})

    paints = {
        "corridor": lambda: fx.corridor(canvas, fx.comet(n, 0.5), palette,
                                        palette.rotated(70.0), height=0.4),
        "wash": lambda: fx.wash(canvas, palette, 0.8, gradient=0.6),
        "bars": lambda: fx.bars(canvas, palette, 0.3),
        "radial": lambda: fx.radial(canvas, palette, 0.4),
        "pinwheel": lambda: fx.pinwheel(canvas, palette, 0.2),
        "plasma": lambda: fx.plasma(canvas, palette, 1.0),
        "net_sparkle": lambda: fx.net_sparkle(canvas, pal.WHITE, 3, density=0.05),
        "spiral": lambda: fx.spiral(canvas, palette, 0.3),
        "ripples": lambda: fx.ripples(canvas, palette, 0.3),
        "checker": lambda: fx.checker(canvas, palette, 1),
        "orbit": lambda: fx.orbit(canvas, palette, 0.6),
        "halves": lambda: fx.halves(canvas, palette, True, 0.8),
        "apex": lambda: fx.apex(canvas, palette, 0.8),
        "helix": lambda: fx.helix(canvas, palette, 0.3),
        "shatter": lambda: fx.shatter(canvas, palette, 0.4, seed=5),
        "par": lambda: fx.par(canvas, palette.color(0), 0.9, white=0.2),
    }
    if not layout.par:
        paints.pop("par")      # a layout without one makes fx.par a no-op
    for name, paint in paints.items():
        canvas.clear()
        paint()
        check(np.isfinite(canvas._source).all(), f"{name}: produced NaN or inf")
        check(canvas.to_channels().any(), f"{name}: painted nothing")

    # Targeting a group must leave the other group alone -- this is the failure
    # a fancy-index copy would produce, silently.
    canvas.clear()
    big, small = canvas.net_pair()
    fx.wash(canvas, palette, 1.0, targets=small)
    check(not canvas.nets[big].any(), "a small-net wash leaked onto the big nets")
    check(canvas.nets[small].any(), "a small-net wash painted nothing")
    return (f"{len(fx.PATTERNS)} corridor patterns, {len(paints)} effects, "
            f"{len(gestures)} gestures, {len(pal.SCHEMES)} schemes, all sane")


def test_script(layout: Layout, fps: float = 40.0) -> str:
    """The fixed show must be deterministic and fit the frame budget."""
    canvas = Canvas(layout)
    script = Script(canvas)
    count = int(script.duration * fps)
    out = np.zeros(layout.channel_count, dtype=np.uint8)

    for i in range(40):                       # warm up
        script.render(i, i / fps)
        canvas.to_channels(out)

    first = np.empty((count, layout.channel_count), dtype=np.uint8)
    started = time.perf_counter()
    for i in range(count):
        script.render(i, i / fps)
        first[i] = canvas.to_channels(out)
    per_frame = (time.perf_counter() - started) / count * 1000

    second = Script(Canvas(layout))
    for i in range(0, count, 7):              # spot-check, a full re-render is slow
        second.render(i, i / fps)
        check(np.array_equal(second.canvas.to_channels(), first[i]),
              f"frame {i} differs between two renders -- the script is not "
              "deterministic, so a capture cannot be compared to a re-render")

    check(per_frame < 10.0,
          f"{per_frame:.2f} ms/frame exceeds the 10 ms budget")
    lit = np.unique(np.where(first.any(axis=0))[0])
    check(lit.size > layout.channel_count * 0.9,
          f"only {lit.size} of {layout.channel_count} channels ever light")
    return (f"{count} frames deterministic, {per_frame:.2f} ms/frame "
            f"(budget 10), {lit.size}/{layout.channel_count} channels used")


def test_geometry(layout: Layout) -> str:
    """Preview geometry must be the real rig, not a diagram of it."""
    from .geometry import build_preview, world_positions

    positions = world_positions(layout)
    check(set(positions) == set(layout.models), "a model has no world position")
    for name, points in positions.items():
        check(points.shape == (layout[name].nodes, 3),
              f"{name}: {points.shape} positions for {layout[name].nodes} nodes")
        check(np.isfinite(points).all(), f"{name}: non-finite coordinates")

    # The corridor's depth is the claim worth testing: the arches share one
    # WorldPos, so if PointData were ignored they would all land on top of each
    # other and a wave down the tunnel would have nowhere to travel.
    z = np.array([positions[n][:, 2].mean() for n in layout.arches])
    check(np.all(np.diff(z) > 0),
          "arches are not in increasing Z order -- Tunnel order is not physical")
    spacing = np.diff(z)
    check(spacing.std() < 1.0,
          f"arch spacing is uneven (std {spacing.std():.1f})")

    preview = build_preview(layout)
    check(len(preview) > 1000, f"only {len(preview)} preview dots")
    check(((preview.xy >= -0.5) & (preview.xy <= 1.5)).all(),
          "projected points fall far outside the frame")
    check(preview.channels.max() < layout.channel_count,
          "preview samples a channel past the end of the frame")

    # A red frame must preview as red on a GRB arch too.
    frame = layout.blank_channels()
    for model in layout.models.values():
        rgb = np.zeros((model.nodes, 4), dtype=np.uint8)
        rgb[:, 0] = 255
        model.pack(rgb, frame)
    sampled = preview.sample(frame)
    check(np.all(sampled[:, 0] == 255) and not sampled[:, 1:].any(),
          "the preview does not undo the wire colour order")
    return (f"{len(preview)} dots, corridor spans {z.max() - z.min():.0f} units "
            f"over {len(layout.arches)} arches, {spacing.mean():.0f} apart")


def test_web(layout: Layout) -> str:
    """The daemon's UI: knobs take effect, presets round-trip, frames flow."""
    return asyncio.run(_web(layout))


async def _web(layout: Layout) -> str:
    import aiohttp
    from aiohttp import web as aioweb

    from .engine import Engine
    from .web import Server

    fps = 40.0
    with tempfile.TemporaryDirectory() as tmp:
        record = Path(tmp) / "engine.fseq"
        engine = Engine(layout, fps=fps, record=record)
        server = Server(engine)
        engine.start()
        runner = aioweb.AppRunner(server.app())
        await runner.setup()
        site = aioweb.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        base = f"http://127.0.0.1:{port}"

        async def settle(frames: int = 4) -> None:
            await asyncio.sleep(frames / fps)

        try:
            async with aiohttp.ClientSession() as http:
                async with http.get(f"{base}/api/schema") as r:
                    schema = await r.json()
                check(len(schema["settings"]) == len(knobs.SCHEMA),
                      "the schema the browser gets is missing knobs")
                async with http.get(f"{base}/api/geometry") as r:
                    geo = await r.json()
                check(geo["count"] * 2 == len(geo["xy"]), "geometry xy is malformed")

                async with http.ws_connect(f"{base}/ws") as ws:
                    binary = status = None
                    for _ in range(30):
                        message = await asyncio.wait_for(ws.receive(), timeout=3.0)
                        if message.type == aiohttp.WSMsgType.BINARY:
                            binary = message.data
                        elif message.type == aiohttp.WSMsgType.TEXT:
                            status = json.loads(message.data)
                        if binary is not None and status is not None:
                            break
                    check(binary is not None, "no preview frames over the WebSocket")
                    check(status is not None, "no status over the WebSocket")
                    check(len(binary) == geo["count"] * 3,
                          f"preview frame is {len(binary)} bytes, "
                          f"expected {geo['count'] * 3}")
                    check(status["status"]["running"], "engine reports not running")

                    # Every knob must take effect without a restart, within a
                    # frame or two -- that is the whole point of the panel.
                    await settle()
                    lit = int((engine.frame > 0).sum())
                    check(lit > 0, "engine is not lighting anything to begin with")

                    async with http.post(f"{base}/api/settings",
                                         json={"blackout": True}) as r:
                        check(r.status == 200, f"blackout rejected: {r.status}")
                    await settle(3)
                    check(not engine.frame.any(),
                          "blackout did not reach the wire within 3 frames")

                    # ... and the preview must go dark with it, because it
                    # samples the wire frame rather than the canvas.
                    dark = None
                    for _ in range(20):
                        message = await asyncio.wait_for(ws.receive(), timeout=3.0)
                        if message.type == aiohttp.WSMsgType.BINARY:
                            dark = message.data
                            if not any(dark):
                                break
                    check(dark is not None and not any(dark),
                          "the preview still shows light during a blackout")

                    async with http.post(f"{base}/api/settings",
                                         json={"blackout": False}) as r:
                        pass
                    await settle(3)
                    check(engine.frame.any(), "releasing blackout did not restore output")

                    async with http.post(f"{base}/api/settings",
                                         json={"brightness": 0.25}) as r:
                        pass
                    await settle(3)
                    dim = int(engine.frame.max())
                    async with http.post(f"{base}/api/settings",
                                         json={"brightness": 1.0}) as r:
                        pass
                    await settle(3)
                    check(int(engine.frame.max()) > dim,
                          f"master brightness had no effect ({dim} vs full)")

                    async with http.post(f"{base}/api/settings",
                                         json={"pattern": "strobe"}) as r:
                        pass
                    # The status block is published every 8 frames, so poll for
                    # it rather than assuming one frame time is enough.
                    for _ in range(20):
                        await settle(2)
                        if engine.status.pattern == "strobe":
                            break
                    check(engine.status.pattern == "strobe",
                          f"pattern override ignored ({engine.status.pattern})")

                    async with http.post(f"{base}/api/settings",
                                         json={"brightness": 3.0}) as r:
                        check(r.status == 200, "an out-of-range slider should clamp")
                    async with http.post(f"{base}/api/settings",
                                         json={"nonsense": 1}) as r:
                        check(r.status == 400, "an unknown setting should be refused")

                    # Presets.
                    name = "selftest tmp"
                    knobs.PRESET_DIR = Path(tmp) / "presets"
                    async with http.post(f"{base}/api/settings",
                                         json={"corridor_rate": 2.5}) as r:
                        pass
                    async with http.put(
                            f"{base}/api/presets/{name.replace(' ', '%20')}") as r:
                        check(r.status == 200, f"preset save failed: {r.status}")
                    async with http.post(f"{base}/api/settings",
                                         json={"corridor_rate": 1.0}) as r:
                        pass
                    async with http.post(
                            f"{base}/api/presets/{name.replace(' ', '%20')}/load") as r:
                        loaded = (await r.json())["settings"]
                    check(loaded["corridor_rate"] == 2.5,
                          "loading a preset did not restore the knob")
                    async with http.delete(
                            f"{base}/api/presets/{name.replace(' ', '%20')}") as r:
                        check(name not in (await r.json())["presets"],
                              "deleted preset is still listed")

                    # Moving the camera must rebuild geometry and tell clients.
                    before = geo["generation"]
                    async with http.post(f"{base}/api/camera",
                                         json={"yaw": 90.0}) as r:
                        check((await r.json())["generation"] > before,
                              "moving the camera did not bump the generation")

                # The browser going away must not touch the engine.
                frames_at_close = engine.status.frames
                await settle(8)
                check(engine.status.frames > frames_at_close,
                      "the engine stalled when the browser disconnected")

                async with http.ws_connect(f"{base}/ws") as ws:
                    message = await asyncio.wait_for(ws.receive(), timeout=3.0)
                    check(message.type in (aiohttp.WSMsgType.BINARY,
                                           aiohttp.WSMsgType.TEXT),
                          "could not reconnect after a reload")

            frames = engine.status.frames
        finally:
            await runner.cleanup()
            engine.stop()

        # What the operator saw is what went to disk: the recording must carry
        # the blackout the panel asked for.
        header, data = fseq.read_all(record)
        check(header.frame_count > 10, "the engine recorded almost nothing")
        dark = data.sum(axis=1) == 0
        check(dark.any(),
              "no all-dark frame in the recording, but blackout was pressed")
        check((~dark).any(), "the recording is dark all the way through")

    return (f"{frames} frames served, blackout/brightness/pattern live, "
            f"presets round-trip, reconnect clean")


def test_analysis(layout: Layout) -> str:
    """Features must be finite, level-independent, and spot a kick."""
    from .analysis import Analyzer
    from .audio import ArraySource
    from .verify import click_track

    audio, grid = click_track([(128.0, 12.0)])
    series: dict[str, np.ndarray] = {}
    for scale, label in ((1.0, "unity"), (0.05, "quiet"), (4.0, "hot")):
        analyzer = Analyzer()
        rows = [analyzer.push(b) for b in ArraySource(audio * scale).blocks()]
        settled = rows[len(rows) // 4:]
        for f in settled:
            check(all(np.isfinite(v) for v in vars(f).values()),
                  f"{label}: non-finite feature at t={f.t:.2f}")
        series[label] = np.array([f.energy for f in settled])
        # Averaged over the blocks the baseline actually learns from, energy
        # is 1.0 by construction.  Not over *all* blocks: a click track is
        # three-quarters digital silence, which is gated out on purpose.
        audible = series[label][np.array([f.rms for f in settled]) > 0]
        check(0.7 < audible.mean() < 1.4,
              f"{label}: mean energy over audible blocks should be near 1.0, "
              f"got {audible.mean():.2f} -- the baseline is not tracking level")

        # The kick detector must fire on beats and not between them, at any
        # input level.  That is the whole basis of the clock's polarity check.
        times = np.array([f.t for f in settled])
        kicks = np.array([f.kick for f in settled])
        period = 60.0 / 128.0
        phase = ((times - grid[0]) / period) % 1.0
        on = kicks[np.minimum(phase, 1 - phase) < 0.12]
        off = kicks[np.abs(phase - 0.5) < 0.12]
        check(on.mean() > 2.0 * off.mean(),
              f"{label}: kick strength on the beat ({on.mean():.2f}) is not "
              f"clearly above the offbeat ({off.mean():.2f})")
    # The point of dividing by a baseline: the same music at any input level
    # must produce the same numbers, because every threshold downstream is
    # relative.  Otherwise the show reacts to the mixer's gain knob.
    for label in ("quiet", "hot"):
        spread = np.abs(series[label] - series["unity"]).max()
        check(spread < 0.05,
              f"{label} differs from unity by up to {spread:.3f} -- energy is "
              "not level-independent")
    return ("features finite and identical across an 80x level range; "
            "kick spikes on the beat")


def test_clock(layout: Layout) -> str:
    """The PLL: prediction under jitter, free-run, re-lock, tempo change."""
    from .beats import BeatEvent
    from .clock import BeatClock

    period = 60.0 / 128.0
    rng = np.random.default_rng(0)

    # 1. Predictions must be better than the observations they are built from.
    clock = BeatClock()
    errors = []
    for k in range(300):
        beat = k * period
        if k > 20:
            ask = beat - 0.2 * period          # how a renderer queries: ahead
            clock.tick(ask)
            errors.append((clock.next_beat(ask) - beat) * 1000)
        clock.tick(beat)
        clock.on_beat(BeatEvent(t=beat + rng.normal(0, 0.025), tempo=128.0,
                                confidence=0.9))
    e = np.abs(np.array(errors))
    check(np.median(e) < 10.0,
          f"25 ms of tracker jitter should average down, got {np.median(e):.1f} ms")
    check(abs(clock.tempo - 128.0) < 0.5, f"tempo drifted to {clock.tempo:.2f}")

    # 2. Silence: keep predicting, decay confidence, never stop.
    before = clock.confidence
    start = 300 * period
    fired = 0
    previous = start
    for step in range(1, 400):
        now = start + step * 0.025
        clock.tick(now)
        fired += len(clock.crossed(previous, now))
        previous = now
    check(clock.free_running, "clock did not notice it had lost the beat")
    check(clock.confidence < before * 0.6,
          f"confidence barely moved through 10 s of silence ({clock.confidence:.2f})")
    check(clock.confidence > 0.0, "confidence hit zero -- a stale grid still beats none")
    expected = 10.0 / period
    check(abs(fired - expected) <= 2,
          f"free-run produced {fired} beats over 10 s, expected ~{expected:.0f}")

    # 3. A half-beat slip must be corrected from the low end alone.
    clock = BeatClock(tempo=128.0, confidence=1.0)
    clock.anchor = 0.5 * period          # deliberately on the offbeat
    clock.last_beat_t = 0.0
    for step in range(4000):
        now = step * clock.block_s
        clock.last_beat_t = now          # never free-running
        # kick energy only at the true beats, which the model has wrong
        phase = (now / period) % 1.0
        clock.observe(now, 6.0 if min(phase, 1 - phase) < 0.05 else 0.05)
        clock.tick(now)
    check(clock.slips >= 1, "the clock never noticed it was on the offbeat")
    residual = abs(((clock.anchor / period) % 1.0) - 0.0)
    residual = min(residual, 1 - residual)
    check(residual < 0.1,
          f"after correcting, the anchor is still {residual:.2f} of a beat out")

    # 4. A real tempo change must be followed, not merely tracked.
    clock = BeatClock()
    now = 0.0
    for k in range(120):
        bpm = 128.0 if k < 60 else 140.0
        clock.tick(now)
        clock.on_beat(BeatEvent(t=now, tempo=bpm, confidence=0.9))
        now += 60.0 / bpm
    check(abs(clock.tempo - 140.0) < 2.0,
          f"clock settled at {clock.tempo:.1f} after a 128 -> 140 change")
    return ("prediction beats its input (25 ms jitter -> "
            f"{np.median(e):.1f} ms), free-run, offbeat recovery, tempo step")


def test_beat_pipeline(layout: Layout) -> str:
    """Audio in, beats out: the whole chain against a known grid."""
    from .audio import ArraySource
    from .verify import Report, click_track, run

    audio, grid = click_track([(128.0, 20.0)])
    fired, listener = run(ArraySource(audio))
    report = Report("clicks", fired, grid, listener)
    errors = np.abs(report.errors)
    check(len(errors) > 30, f"only {len(errors)} beats scored")
    within = float(np.mean(errors < 30))
    check(within > 0.85,
          f"only {100 * within:.0f}% of beats within 30 ms (median "
          f"{np.median(errors):.1f} ms)")
    check(report.offbeat_share == 0.0,
          f"{100 * report.offbeat_share:.0f}% of beats landed on the offbeat")
    check(abs(listener.clock.tempo - 128.0) < 2.0,
          f"tempo settled at {listener.clock.tempo:.2f}, not 128")

    # A silent break is the normal case in a DJ set, not an error case.
    audio, grid = click_track([(128.0, 16.0), (0.0, 7.5), (128.0, 16.0)])
    fired, listener = run(ArraySource(audio))
    through = [f for f in fired if 16.5 < f.predicted < 23.0]
    check(len(through) >= 12,
          f"only {len(through)} beats predicted through 7.5 s of silence")
    after = Report("after", [f for f in fired if f.predicted > 25.0], grid, listener)
    relock = np.median(np.abs(after.errors))
    check(relock < 30.0, f"re-locked {relock:.1f} ms off after the break")
    return (f"{100 * within:.0f}% within 30 ms on a click track, no offbeat; "
            f"free-runs a 7.5 s break and re-locks to {relock:.1f} ms")


def test_downbeat(layout: Layout) -> str:
    """Does the bar tracker *find* the bar line, from any wrong start?"""
    from .audio import ArraySource
    from .clock import BeatClock
    from .listener import Listener
    from .verify import click_track

    # A plain click track has no bar line to find, so there would be nothing
    # to test: every beat is identical.  accent_every marks one.
    audio, grid = click_track([(128.0, 40.0)], accent_every=4)
    bar = 60.0 / 128.0 * 4
    truth = np.arange(1, int(40.0 / bar)) * bar

    results = []
    for offset in range(4):
        clock = BeatClock()
        clock.downbeat = offset
        listener = Listener(ArraySource(audio), clock=clock)
        marks: list[tuple[float, int]] = []
        for block in listener.source.blocks():
            features = listener.step(block)
            marks.append((features.t, (clock.beat_index_at(features.t)
                                       - clock.downbeat) % 4))
        # Score by comparing *times*, never "which beat are we in" at the
        # instant of a beat -- that is ambiguous by half a beat and reports
        # about 50% no matter how right the tracker is.
        stamps = np.array([m[0] for m in marks])
        phases = np.array([m[1] for m in marks])
        hits = 0
        for T in truth:
            j = min(int(np.searchsorted(stamps, T)), len(stamps) - 1)
            window = phases[max(0, j - 2):j + 3]
            hits += int((window == 0).any())
        settled = hits / max(1, len(truth))
        results.append((offset, settled, listener.bars.shifts))

    for offset, share, shifts in results:
        check(share > 0.8,
              f"from offset +{offset} only {100 * share:.0f}% of bar lines were "
              f"found ({shifts} shifts)")
    moved = [r for r in results if r[0] != 0]
    check(all(r[2] >= 1 for r in moved),
          "the tracker never moved the bar line from a wrong start")
    check(all(r[2] <= 3 for r in results),
          f"the bar line moved too often: {[r[2] for r in results]}")
    return (f"bar line found from all four starts "
            f"({min(100 * r[1] for r in results):.0f}%+ of bars, "
            f"{max(r[2] for r in results)} shift(s) at most)")


def test_state_machine(layout: Layout) -> str:
    """Quiet -> cruising -> build -> drop, with the drop on time."""
    from .audio import ArraySource
    from .listener import Listener
    from .state import BUILDING, HOT, StateMachine
    from .verify import arc_track

    audio, sections = arc_track()
    machine = StateMachine()
    listener = Listener(ArraySource(audio))
    for block in listener.source.blocks():
        machine.push(listener.step(block))

    history = machine.history
    check(history, "the state machine never changed state at all")
    check(len(history) <= 6,
          f"{len(history)} state changes for a four-section arc -- it is "
          f"flapping: {[(round(t, 1), s) for t, s in history]}")

    wanted = {kind: start for kind, start, _ in sections}
    build = next((t for t, s in history if s == BUILDING), None)
    check(build is not None, "never entered building")
    late = build - wanted[BUILDING]
    check(-0.5 <= late <= 4.0,
          f"entered building {late:+.1f}s from the real boundary "
          f"(allowed -0.5 to +4.0)")
    # The drop is the hot *after* the build.  An earlier hot is allowed: the
    # engine starts cold in this arc, and its cruising section is a kick
    # groove over a bassline, which the cold-start rule calls hot on band
    # shape -- and must then still hand over to the build when the sweep
    # arrives, or the drop's timing is lost.
    drop = next((t for t, s in history if s == HOT and t > build), None)
    check(drop is not None, "never entered hot after the build")
    late = drop - wanted[HOT]
    check(-0.5 <= late <= 2.5,
          f"entered hot {late:+.1f}s from the real boundary "
          f"(allowed -0.5 to +2.5)")
    return (f"{len(history)} transitions, build and drop in order, "
            f"drop {drop - wanted[HOT]:+.1f}s from the boundary")


def test_long_drop(layout: Layout) -> str:
    """A drop longer than the loudness baseline stays hot until the music falls."""
    from .audio import ArraySource
    from .listener import Listener
    from .state import BUILDING, HOT, StateMachine
    from .verify import LONG_DROP_PLAN, arc_track

    audio, sections = arc_track(plan=LONG_DROP_PLAN)
    machine = StateMachine()
    listener = Listener(ArraySource(audio))
    for block in listener.source.blocks():
        machine.push(listener.step(block))

    history = machine.history
    drop_start = next(start for kind, start, _ in sections if kind == "hot")
    breakdown = next(start for kind, start, _ in sections[3:] if kind == "quiet")
    build = next((t for t, s in history if s == BUILDING), None)
    check(build is not None, "never entered building")
    # A cold-start hot before the build is allowed (see test_state_machine);
    # the drop is the hot after it, and there must be exactly one.
    entered = [t for t, s in history if s == HOT and t > build]
    check(entered, "never entered hot after the build")
    check(len(entered) == 1,
          f"entered hot {len(entered)} times: {[(round(t, 1), s) for t, s in history]}")
    left = [t for t, s in history if t > entered[0] and s != HOT]
    check(left, "never left hot after the breakdown")
    check(left[0] >= breakdown - 0.5,
          f"left hot at {left[0]:.1f}s, {breakdown - left[0]:.1f}s before the "
          f"breakdown at {breakdown:.1f}s (drop ran {breakdown - drop_start:.0f}s)")
    check(left[0] - breakdown <= 4.0,
          f"left hot {left[0] - breakdown:+.1f}s after the breakdown")
    return (f"held hot through {breakdown - drop_start:.0f}s of drop (baseline "
            f"45s) and a one-bar dip, left {left[0] - breakdown:+.1f}s into "
            f"the breakdown")


def test_arranger(layout: Layout) -> str:
    """Audio in, pixels out: the whole chain, interleaved as the engine runs it."""
    from .arranger import Arranger
    from .audio import ArraySource
    from .listener import Listener
    from .state import StateMachine
    from .verify import arc_track

    audio, _sections = arc_track()
    fps = 40.0

    def render_all() -> tuple[np.ndarray, int]:
        canvas = Canvas(layout)
        listener = Listener(ArraySource(audio))
        machine = StateMachine()
        arranger = Arranger(canvas, listener, state=machine)
        out = np.zeros(layout.channel_count, dtype=np.uint8)
        lit = np.zeros(layout.channel_count, dtype=bool)
        now, index = 0.0, 0
        frames = []
        for block in listener.source.blocks():
            features = listener.step(block)
            machine.push(features)
            while now <= features.t:
                arranger.render(index, now)
                canvas.to_channels(out)
                np.logical_or(lit, out > 0, out=lit)
                if index % 97 == 0:
                    frames.append(out.copy())
                index += 1
                now += 1.0 / fps
        return lit, index, frames, arranger

    lit, frames_a, sample_a, arranger = render_all()
    check(frames_a > 1000, f"only {frames_a} frames rendered")
    check(lit.all(),
          f"{int((~lit).sum())} channels never lit across the whole arc")
    check(arranger.journey >= 1,
          "the palette never advanced -- the hue journey is not moving")
    check(arranger._changes >= 3,
          f"only {arranger._changes} changes of look across four sections")
    check(arranger._scratch is not None,
          "no transition was ever painted -- every change of look was a cut")

    # Every gesture the tables name must paint the nets, at every energy it
    # is offered at; a misspelt name would otherwise fall through silently.
    from .arranger import NET_GESTURES
    probe = Canvas(layout)
    for kind, names in NET_GESTURES.items():
        for name in names:
            probe.clear()
            arranger._gesture(probe, name, 10, 3.3, 0.4,
                              pal.generate(200.0, "triadic", white=True),
                              0.2, 0.6, tension=0.7)
            check(float(probe.nets.max()) > 0.2,
                  f"gesture {name!r} ({kind}) paints nothing")

    # The same audio must give the same show, or a capture cannot be compared
    # with a re-render and the fseq oracle stops working.
    _lit2, frames_b, sample_b, _ = render_all()
    check(frames_a == frames_b, f"frame counts differ: {frames_a} vs {frames_b}")
    for i, (a, b) in enumerate(zip(sample_a, sample_b)):
        check(np.array_equal(a, b), f"sampled frame {i} differs between runs")
    return (f"{frames_a} frames from {len(audio) / 44100:.0f}s of audio, "
            f"every channel used, {arranger.journey} palette steps, "
            f"{arranger._changes} look changes with transitions, deterministic")


def test_config(layout: Layout) -> str:
    """The file configures the rig; a flag still overrides it for one run."""
    from .__main__ import build_parser
    from .config import Config

    defaults = Config()
    check(defaults.output.host == "",
          "the default config must not point at the Falcon -- a laptop should "
          "render without blasting the rig")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rig.toml"
        path.write_text(
            '[output]\nhost = "10.1.2.3"\nfps = 30.0\n'
            '[web]\nport = 9099\n[show]\nbrightness = 0.5\n'
        )
        config = Config.load(path)
        check(config.output.host == "10.1.2.3", "host not read from the file")
        check(config.output.fps == 30.0, "fps not read from the file")
        check(config.output.port == 4048, "an unset value lost its default")

        args = build_parser(config).parse_args(["serve"])
        check(args.ddp == "10.1.2.3", f"config did not reach the CLI ({args.ddp})")
        check(args.fps == 30.0 and args.port == 9099, "config defaults not applied")
        args = build_parser(config).parse_args(["serve", "--fps", "60"])
        check(args.fps == 60.0, "an explicit flag must beat the file")

        # --config after the subcommand is the form the systemd unit uses.
        args = build_parser(config).parse_args(["serve", "--config", str(path)])
        check(args.config == str(path),
              "--config must be accepted after the subcommand too, or the "
              "unit file's ExecStart line fails")

        bad = Path(tmp) / "bad.toml"
        bad.write_text('[output]\nhsot = "typo"\n')
        try:
            Config.load(bad)
        except ValueError as exc:
            check("hsot" in str(exc), "a typo should be named in the error")
        else:
            raise Failure("an unknown setting was silently ignored")

    missing = Config.load(Path(tmp) / "gone.toml")
    check(missing.output.fps == 40.0,
          "a missing config must fall back to defaults, not fail")
    return "defaults < file < flag, unknown keys rejected, missing file is fine"


def test_doctor(layout: Layout) -> str:
    """Preflight must pass on a machine with nothing set up."""
    import io
    from contextlib import redirect_stdout

    from .config import Config
    from .doctor import FAIL, WARN, Report, check_layout, check_storage, check_tools

    report = Report()
    check_layout(report)
    check_tools(report)
    check_storage(report, Config())
    failures = [c for c in report.checks if c.status == FAIL]
    check(not failures,
          "doctor reports a failure on a working checkout: "
          + "; ".join(f"{c.name}: {c.detail}" for c in failures))
    for c in report.checks:
        if c.status != "ok":
            check(bool(c.remedy),
                  f"check {c.name!r} warns without saying what to do about it")

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = __import__("live.doctor", fromlist=["run_checks"]).run_checks(
            Config(), deep=False)
    check(code == 0, f"doctor exited {code} on a healthy machine")
    check("layout" in buffer.getvalue(), "doctor printed no report")
    return f"{len(report.checks)} checks, every non-ok one carries a remedy"


def test_multi_target(layout: Layout) -> str:
    """Two Falcons, two slices, each addressed the way it was uploaded.

    The show does not fit on one controller, and the failure mode when the
    addressing is wrong is silent: DDP is UDP, so a receiver drops offsets
    outside its input range and reports nothing.  Both Falcons are uploaded
    with xLights' "Keep Channel Numbers", so each expects *absolute* show
    offsets -- for the nets Falcon that happens to equal 0-based, which is
    how sending everything from 0 looked verified while the back half of
    the corridor stayed dark at the rig.  The expected offset comes from
    the layout, so this follows the networks file rather than a guess.
    """
    from .engine import Engine

    engine = Engine(layout, host="auto")
    targets = engine.targets
    check(len(targets) >= 2,
          f"expected the show to span several controllers, got {len(targets)}")

    caught: dict[str, list[bytes]] = {}

    class Recorder:
        def __init__(self, key): self.key = caught.setdefault(key, [])
        def sendto(self, data, _addr): self.key.append(data)
        def setblocking(self, _flag): pass
        def close(self): pass

    for sink in targets:
        sink.sender = ddp.DDPSender(sink.host, port=sink.port,
                                    sock=Recorder(sink.name))

    frame = np.arange(layout.channel_count, dtype=np.int64).astype(np.uint8)
    for sink in targets:
        sink.sender.send_frame(frame[sink.span], offset=sink.offset)

    covered = np.zeros(layout.channel_count, dtype=int)
    for sink in targets:
        covered[sink.span] += 1
        packets = [ddp.parse(p) for p in caught[sink.name]]
        check(bool(packets), f"{sink.name} sent nothing")
        controller = layout.output(sink.name)
        want = controller.offset
        check(packets[0].offset == want,
              f"{sink.name} starts at offset {packets[0].offset}, not {want} -- "
              f"the controller was uploaded with keep_channels="
              f"{controller.keep_channels}")
        width = sink.span.stop - sink.span.start
        check(packets[-1].end == want + width,
              f"{sink.name} covers up to {packets[-1].end}, want {want + width}")
        check(packets[-1].push and not any(q.push for q in packets[:-1]),
              f"{sink.name} must push on its last packet only")
        body = b"".join(bytes(q.data) for q in packets)
        check(body == frame[sink.span].tobytes(),
              f"{sink.name} sent the wrong slice of the show")

    check(not (covered > 1).any(),
          f"{int((covered > 1).sum())} channels are sent to two controllers")
    gap = int((covered == 0).sum())
    offsets = {t.name: layout.output(t.name).offset for t in targets}
    return (f"{len(targets)} controllers, {int((covered == 1).sum())} channels "
            f"covered exactly once, DDP offsets {offsets}"
            + (f"; {gap} unaddressed" if gap else ""))



def test_audio_drive(layout: Layout) -> str:
    """The drive knob must change the show, stay deterministic, keep its
    gauges in range, and cost nothing while off."""
    from .arranger import Arranger
    from .audio import ArraySource
    from .listener import Listener
    from .settings import Settings
    from .state import StateMachine
    from .verify import arc_track

    base = pal.generate(200.0, "triadic", white=True)
    check(base.lit(1.0, 0.0) is base, "lit(1, 0) must return the palette itself")
    lit = base.lit(1.25, 0.3)
    check(np.isfinite(lit.colors).all() and lit.colors.max() <= 1.0,
          "lit palette out of range")
    dull = base.lit(0.75, 0.0)
    check(np.allclose(dull.colors, base.colors * 0.75), "lit gain is not a scale")

    audio, _ = arc_track()
    audio = audio[: 44100 * 30]

    def run(drive: bool, depth: float = 1.0):
        canvas = Canvas(layout)
        listener = Listener(ArraySource(audio))
        st = Settings()
        st.audio_drive = drive
        st.drive_depth = depth
        arranger = Arranger(canvas, listener, state=StateMachine(), settings=st)
        out = np.zeros(layout.channel_count, dtype=np.uint8)
        frames, gauges = [], []
        now, index = 0.0, 0
        for block in listener.source.blocks():
            f = listener.step(block)
            arranger.machine.push(f)
            while now <= f.t:
                arranger.render(index, now)
                check(np.isfinite(canvas._source).all(), "drive: NaN or inf")
                if index % 41 == 0:
                    canvas.to_channels(out)
                    frames.append(out.copy())
                    gauges.append(arranger.gauges)
                index += 1
                now += 1.0 / 40.0
        return np.stack(frames), gauges, arranger

    off_a, gauges_off, _ = run(False)
    off_b, _, _ = run(False)
    check(np.array_equal(off_a, off_b), "the show is not deterministic with the drive off")
    check(all(g["rate"] == 1.0 for g in gauges_off),
          "with the drive off the rate must read exactly 1.0")
    on_a, gauges_on, arranger = run(True)
    on_b, _, _ = run(True)
    check(np.array_equal(on_a, on_b), "the show is not deterministic with the drive on")
    check(not np.array_equal(on_a, off_a), "the drive changes nothing")
    differing = float((on_a != off_a).any(axis=1).mean())
    check(differing > 0.5,
          f"the drive only touched {differing:.0%} of sampled frames")
    zero, _, _ = run(True, depth=0.0)
    check(np.array_equal(zero, off_a), "depth 0 must be the same show as off")

    bass = [g["bass"] for g in gauges_on]
    air = [g["air"] for g in gauges_on]
    rate = [g["rate"] for g in gauges_on]
    check(all(0.0 <= b <= 1.0 for b in bass), "bass gauge out of 0..1")
    check(all(0.0 <= a <= 1.0 for a in air), "air gauge out of 0..1")
    check(all(0.7 - 1e-6 <= r <= 1.3 + 1e-6 for r in rate),
          f"drive rate outside 0.7..1.3 ({min(rate):.2f}..{max(rate):.2f})")
    check(max(bass) > 0.1 and max(air) > 0.3 and min(air) < max(air) - 0.2,
          f"gauges never moved: bass to {max(bass):.2f}, air {min(air):.2f}..{max(air):.2f}")
    lead = arranger._drive - arranger._drive_last
    return (f"on/off differ in {differing:.0%} of frames, both deterministic, "
            f"depth 0 is off; bass to {max(bass):.2f}, air {min(air):.2f}..{max(air):.2f}, "
            f"rate {min(rate):.2f}..{max(rate):.2f}, drive ended {lead:+.1f} beats off the clock")


def test_game(layout: Layout) -> str:
    """Game mode: the board is playable, the rules hold, the rig is painted."""
    from .frame import Canvas
    from .game import (DIRECTIONS, MIN_PIXELS, START_LENGTH, Board, Game,
                       Snake, _OPPOSITE)

    canvas = Canvas(layout)
    board = Board(canvas)
    check(board.count >= 60,
          f"only {board.count} playable cells -- the board is too coarse to play")
    check(board.covered >= 0.9,
          f"only {board.covered:.0%} of the big nets' pixels belong to a cell")
    lit = board.pixels[board.playable.reshape(-1)]
    check(int(lit.min()) >= MIN_PIXELS,
          f"a playable cell has {int(lit.min())} pixels, under {MIN_PIXELS}")
    # Every cell reachable from the start: no diagonal-only islands.
    seen = {board.centre()}
    frontier = [board.centre()]
    while frontier:
        r, c = frontier.pop()
        for dr, dc in DIRECTIONS.values():
            n = (r + dr, c + dc)
            if n in board and n not in seen:
                seen.add(n)
                frontier.append(n)
    check(len(seen) == board.count,
          f"{board.count - len(seen)} cells are unreachable from the start")

    snake = Snake(board, seed=1)
    check(len(snake.body) == START_LENGTH and snake.alive and not snake.running,
          "a new snake should be three long, alive and waiting")
    check(snake.step() == "idle", "the snake moved before it was started")
    snake.running = True
    head, facing = snake.head, snake.direction
    check(snake.step() == "moved", "the first step did not move")
    dr, dc = DIRECTIONS[facing]
    check(snake.head == (head[0] + dr, head[1] + dc),
          f"stepped {facing} from {head} and landed on {snake.head}")
    snake.turn(_OPPOSITE[facing])
    snake.step()
    check(snake.direction == facing, "a reversal was honoured -- that is suicide")
    # Food directly ahead: one bite grows the snake by one and moves the food.
    dr, dc = DIRECTIONS[snake.direction]
    snake.food = (snake.head[0] + dr, snake.head[1] + dc)
    check(snake.food in board, "the test put food on a wall; pick another seed")
    length = len(snake.body)
    check(snake.step() == "ate", "walking onto the food did not eat it")
    check(len(snake.body) == length + 1 and snake.score == 1,
          f"after eating: length {len(snake.body)} (was {length}), score {snake.score}")
    check(snake.food is not None and snake.food not in snake.body,
          "the new food landed on the snake")
    # Into a wall: dead within the board's width, and then inert.
    outcome = "moved"
    for _ in range(board.rows + board.cols):
        outcome = snake.step()
        if outcome == "died":
            break
    check(outcome == "died", "walking straight never hit a wall")
    check(not snake.alive and not snake.running and snake.step() == "idle",
          "a dead snake is still moving")
    check(snake.best == 1, f"best score {snake.best} after a game of 1")
    snake.reset()
    check(snake.alive and snake.games == 2 and snake.score == 0,
          "reset did not start a fresh game")

    # The whole thing on the clock: at 8 cells/s over 1.5 s the snake takes
    # about 11 steps, and the rig is painted every frame.
    game = Game(canvas, seed=3)
    game.input("turn", "left")
    for i in range(60):
        game.render(i / 40.0, speed=8.0)
    snap = game.snapshot()
    check(9 <= snap["steps"] <= 12 or not snap["alive"],
          f"{snap['steps']} steps in 1.5 s at 8 cells/s")
    big = canvas.nets[canvas.big]
    check(float(big.max()) > 0.5, "the board is not lit")
    small = canvas.nets[canvas.net_pair()[1]]
    check(float(small.max()) > 0.05, "the small nets went dark in game mode")
    arches = canvas.arches.max(axis=(1, 2))
    check(int((arches > 0.1).sum()) == min(snap["score"], arches.size)
          or game.current._ate_at > 1.5 - 0.7 or not snap["alive"],
          f"{int((arches > 0.1).sum())} arches lit for a score of {snap['score']}")
    game.input("pause")
    game.render(5.0, speed=4.0)
    first = canvas._source.copy()
    game.render(5.0, speed=4.0)
    check(np.array_equal(first, canvas._source),
          "the same game at the same time painted two different frames")
    try:
        game.input("turn", "sideways")
    except ValueError:
        pass
    else:
        raise Failure("a bogus direction was accepted")

    # Pac-Man: one maze across every triangle, joined by tunnels.
    game.render(6.0, kind="pacman", speed=4.0)
    check(game.snapshot()["game"] == "pacman", "the game knob did not switch")
    pg = game.current
    all_board, maze, pac = pg.board, pg.maze, pg.pac
    check(all_board.surface == "all" and len(all_board.regions) == 4,
          f"the all-nets board has {len(all_board.regions)} islands, not 4")
    check(len(maze.distances(pac.start)) == len(maze.free),
          f"{len(maze.free) - len(maze.distances(pac.start))} maze cells "
          "are unreachable from the start")
    tunnels = sum(1 for c in maze.free for n in maze.exits(c).values()
                  if abs(n[0] - c[0]) + abs(n[1] - c[1]) > 1)
    check(tunnels >= 8, f"only {tunnels} tunnel ends between the triangles")
    dead = sum(1 for c in maze.free if len(maze.exits(c)) <= 1)
    check(dead <= len(maze.free) // 8, f"{dead} dead ends in {len(maze.free)} cells")
    check(len(maze.exits(pac.start)) >= 2, "the player starts boxed in")
    check(len(set(pac.homes)) == 3 and pac.start not in pac.homes,
          f"ghost homes {pac.homes} against start {pac.start}")
    check(len(pac.pellets) == len(maze.free) - 4 and len(pac.power) == 4,
          f"{len(pac.pellets)} pellets, {len(pac.power)} power pellets")
    pac.running = True
    first = next(iter(maze.exits(pac.cell)))
    pac.turn(first)
    check(pac.move() in ("moved", "pellet", "power") and pac.cell != pac.start,
          "the first move went nowhere")
    # A power pellet frightens every ghost; eating one then is worth 200.
    power = next(iter(pac.power))
    from_cell, via = next((c, d) for c in maze.free
                          for d, n in maze.exits(c).items() if n == power)
    pac.cell = pac.prev = from_cell
    pac.turn(via)
    check(pac.move() == "power", "walking onto a power pellet did not fire it")
    check(all(g.mode == "fright" for g in pac.ghosts), "the ghosts are not frightened")
    score = pac.score
    pac.ghosts[0].cell = pac.cell
    check(pac.collisions() == "ghost" and pac.ghosts[0].mode == "eyes"
          and pac.score == score + 200, "a frightened ghost was not eaten")
    pac.ghosts[1].mode = "chase"
    pac.ghosts[1].cell = pac.cell
    check(pac.collisions() == "died" and pac.lives == 2, "a ghost did not cost a life")
    pac.pellets = {maze.exits(pac.cell)[first]} if first in maze.exits(pac.cell) \
        else {next(iter(maze.exits(pac.cell).values()))}
    pac.turn(next(d for d, n in maze.exits(pac.cell).items() if n in pac.pellets))
    check(pac.move() == "clear" and pac.level == 2
          and len(pac.pellets) == len(maze.free) - 4,
          "clearing the pellets did not start the next level")
    # Through the clock: a fresh game, steered along its first exit, moves
    # and paints every net.
    game.input("reset")
    game.input("turn", next(iter(maze.exits(pac.start))))
    for i in range(80):
        game.render(10.0 + i / 40.0, kind="pacman", speed=6.0)
    snap = game.snapshot()
    check(snap["steps"] >= 1 or snap["over"], "Pac-Man did not move on the clock")
    check(len(snap["pellets"]) == len(maze.free), "the pellet string is the wrong length")
    lit_nets = int((canvas.nets.max(axis=(1, 2)) > 0.05).sum())
    check(lit_nets == len(canvas.net_names),
          f"only {lit_nets} of {len(canvas.net_names)} nets lit in Pac-Man")
    # Pac-Man is a sprite, not a cell: a disc of ~100 pixels with a mouth
    # that opens and shuts, so his pixel count swings as he moves.
    game.input("reset")
    game.input("turn", next(iter(maze.exits(pac.start))))
    yellow = []
    for i in range(40):
        game.render(30.0 + i / 40.0, kind="pacman", speed=4.0)
        n = canvas.nets
        yellow.append(int(((n[..., 0] > 0.8) & (n[..., 1] > 0.6)
                           & (n[..., 2] < 0.3)).sum()))
    cell_px = int(np.median(all_board.pixels[all_board.playable.reshape(-1)]))
    check(min(yellow) >= 3 * cell_px,
          f"Pac-Man is {min(yellow)} pixels, a cell is {cell_px}: not a sprite")
    check(max(yellow) - min(yellow) >= cell_px,
          f"the mouth is not moving ({min(yellow)}..{max(yellow)} pixels)")
    game.input("pause")
    game.render(20.0, kind="pacman", speed=4.0)
    first_frame = canvas._source.copy()
    game.render(20.0, kind="pacman", speed=4.0)
    check(np.array_equal(first_frame, canvas._source),
          "the same Pac-Man at the same time painted two different frames")
    game.render(21.0, kind="snake", speed=4.0)
    check(game.snapshot()["game"] == "snake", "switching back to snake failed")

    return asyncio.run(_game_web(
        layout, f"snake {board.rows}x{board.cols}/{board.count} cells "
                f"({board.covered:.0%} of pixels, ~{int(np.median(lit))} px), "
                f"pac-man {all_board.rows}x{all_board.cols}/{len(maze.free)} "
                f"free cells, {tunnels} tunnel ends, {dead} dead ends"))


async def _game_web(layout: Layout, detail: str) -> str:
    """Game mode through the daemon: the knob, the input API, the feed."""
    import aiohttp
    from aiohttp import web as aioweb

    from .engine import Engine
    from .web import Server

    fps = 40.0
    engine = Engine(layout, fps=fps)
    server = Server(engine)
    engine.start()
    runner = aioweb.AppRunner(server.app())
    await runner.setup()
    site = aioweb.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    base = f"http://127.0.0.1:{port}"
    try:
        async with aiohttp.ClientSession() as http:
            async with http.get(f"{base}/game") as r:
                check(r.status == 200 and "game.js" in await r.text(),
                      "/game does not serve the game page")
            async with http.get(f"{base}/api/game") as r:
                body = await r.json()
            check(body["board"]["rows"] > 0 and len(body["board"]["cells"]) > 60,
                  "/api/game does not describe the board")
            check(body["game_mode"] is False, "game mode is on at boot")

            await asyncio.sleep(8 / fps)
            show = engine.frame.copy()
            async with http.post(f"{base}/api/settings",
                                 json={"game_mode": True}) as r:
                check(r.status == 200, "game_mode rejected")
            await asyncio.sleep(10 / fps)
            check(not np.array_equal(show, engine.frame),
                  "switching to game mode did not change the frame")
            check(engine.status.scene == "game",
                  f"status says scene {engine.status.scene!r} in game mode")

            async with http.post(f"{base}/api/game",
                                 json={"action": "turn", "dir": "left"}) as r:
                check(r.status == 200, "a turn was rejected")
                state = (await r.json())["state"]
            check(state["running"], "an arrow key did not start the game")
            async with http.post(f"{base}/api/game",
                                 json={"action": "turn", "dir": "diagonal"}) as r:
                check(r.status == 400, "a bogus direction was accepted")

            async with http.ws_connect(f"{base}/ws") as ws:
                # The first game message is the state as it stands; the
                # snake's first step follows a quarter second later.
                game = None
                for _ in range(200):
                    message = await asyncio.wait_for(ws.receive(), timeout=3.0)
                    if message.type == aiohttp.WSMsgType.TEXT:
                        body = json.loads(message.data)
                        if "game" in body:
                            game = body["game"]
                            if game["steps"] >= 1:
                                break
                check(game is not None, "no game state over the WebSocket")
                check(game["running"] and game["steps"] >= 1,
                      "the WebSocket game state is not moving")

            async with http.post(f"{base}/api/settings",
                                 json={"game": "pacman"}) as r:
                check(r.status == 200, "the game knob rejected pacman")
            await asyncio.sleep(4 / fps)
            async with http.get(f"{base}/api/game") as r:
                body = await r.json()
            check(body["board"]["game"] == "pacman" and body["board"]["walls"]
                  and body["state"]["game"] == "pacman"
                  and "pacman" in body["board"]["games"],
                  "/api/game did not follow the game knob to pacman")
            async with http.post(f"{base}/api/settings",
                                 json={"game": "snake", "game_mode": False}) as r:
                check(r.status == 200, "game_mode off rejected")
            await asyncio.sleep(10 / fps)
            check(engine.status.scene != "game", "the show did not come back")
    finally:
        await runner.cleanup()
        engine.stop()
    return detail


TESTS = (
    ("channel map", test_layout),
    ("fseq round-trip", test_fseq_roundtrip),
    ("DDP split/reassemble", test_ddp_packets),
    ("UDP loopback -> fseq", test_loopback),
    ("multi-controller split", test_multi_target),
    ("canvas -> channels", test_canvas),
    ("orientation", test_orient),
    ("clips", test_clips),
    ("showpieces", test_pieces),
    ("effect vocabulary", test_effects),
    ("fixed script", test_script),
    ("preview geometry", test_geometry),
    ("web UI + engine", test_web),
    ("game mode", test_game),
    ("audio features", test_analysis),
    ("beat clock", test_clock),
    ("audio -> beats", test_beat_pipeline),
    ("bar tracking", test_downbeat),
    ("state machine", test_state_machine),
    ("long drop", test_long_drop),
    ("arranger", test_arranger),
    ("audio drive", test_audio_drive),
    ("config", test_config),
    ("doctor", test_doctor),
)


def main(argv: list[str] | None = None) -> int:
    layout = load_layout()
    failures = 0
    for name, test in TESTS:
        started = time.perf_counter()
        try:
            detail = test(layout)
        except Failure as exc:
            failures += 1
            print(f"FAIL  {name}\n        {exc}")
        except Exception as exc:  # noqa: BLE001 - a crash is a failure too
            failures += 1
            print(f"ERROR {name}\n        {type(exc).__name__}: {exc}")
        else:
            ms = (time.perf_counter() - started) * 1000
            print(f"ok    {name:<22} {detail}  [{ms:.0f} ms]")
    print()
    print("all good" if not failures else f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
