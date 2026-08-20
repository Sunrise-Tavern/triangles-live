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
from .layout import Layout, load_layout
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
    check(len(layout.nets) == 8, f"expected 8 nets, found {len(layout.nets)}")
    check(all(layout[n].kind == "arch" for n in layout.arches), "Tunnel holds a non-arch")

    # Colour order really is different between the two fixture families -- if
    # this ever collapses to one value, someone has "simplified" the packer.
    net_orders = {layout[n].order for n in layout.nets}
    arch_orders = {layout[n].order for n in layout.arches}
    check(net_orders == {(0, 1, 2)}, f"nets should be RGB, got {net_orders}")
    check(arch_orders == {(1, 0, 2)}, f"arches should be GRB, got {arch_orders}")

    # A red frame must land in the red channel of each fixture, whatever its
    # wire order.  This is the assertion that catches a swapped permutation.
    out = layout.blank_channels()
    for model in models:
        rgb = np.zeros((model.nodes, 4), dtype=np.uint8)
        rgb[:, 0] = 255
        model.pack(rgb, out)
    for name in layout.nets:
        m = layout[name]
        check(out[m.start - 1] == 255 and out[m.start] == 0,
              f"{name}: red landed on the wrong channel (RGB model)")
    for name in layout.arches:
        m = layout[name]
        check(out[m.start - 1] == 0 and out[m.start] == 255,
              f"{name}: red landed on the wrong channel (GRB model)")
    par = layout[layout.par]
    check(out[par.start - 1] == 255, "par: red is not on DMX slot 1")

    return (f"{len(models)} models, {layout.channel_count} channels, "
            f"no overlaps or gaps; colour order verified")


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
    canvas.par[:3] = (1.0, 0.0, 0.0)
    frame = canvas.to_channels()

    for name in layout.nets:
        m = layout[name]
        check(tuple(frame[m.slice][:3]) == (255, 0, 0),
              f"{name}: red frame is not red on the wire")
    for name in layout.arches:
        m = layout[name]
        check(tuple(frame[m.slice][:3]) == (0, 255, 0),
              f"{name}: a GRB arch should carry red on channel 2")
    par = layout[layout.par]
    check(tuple(frame[par.slice]) == (255, 0, 0, 0), "par: red is not on slot 1")

    # A cleared canvas must be all-off, and painting outside 0..1 must clamp
    # rather than wrap -- an overflowing uint8 shows as a dark flicker.
    canvas.clear()
    check(not canvas.to_channels().any(), "cleared canvas is not black")
    canvas.nets[:] = 4.0
    canvas.arches[:] = -2.0
    frame = canvas.to_channels()
    check(frame[layout[layout.nets[0]].slice].max() == 255, "over-bright did not clamp")
    check(frame[layout[layout.arches[0]].slice].max() == 0, "negative did not clamp")

    # Groups must stay contiguous, or an effect aimed at "the small nets" would
    # silently paint a copy and vanish.
    check(canvas.net_slice("Big Triangle") == slice(0, 4), "Big Triangle moved")
    check(canvas.net_slice("Small Triangle Nets") == slice(4, 8),
          "Small Triangle Nets moved")
    return (f"{canvas.nets.shape} nets + {canvas.arches.shape} arches -> "
            f"{layout.channel_count} channels, colour order and clamping hold")


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

    paints = {
        "corridor": lambda: fx.corridor(canvas, fx.comet(n, 0.5), palette,
                                        palette.rotated(70.0), height=0.4),
        "wash": lambda: fx.wash(canvas, palette, 0.8, gradient=0.6),
        "bars": lambda: fx.bars(canvas, palette, 0.3),
        "radial": lambda: fx.radial(canvas, palette, 0.4),
        "pinwheel": lambda: fx.pinwheel(canvas, palette, 0.2),
        "plasma": lambda: fx.plasma(canvas, palette, 1.0),
        "net_sparkle": lambda: fx.net_sparkle(canvas, pal.WHITE, 3, density=0.05),
        "par": lambda: fx.par(canvas, palette.color(0), 0.9, white=0.2),
    }
    for name, paint in paints.items():
        canvas.clear()
        paint()
        check(np.isfinite(canvas._source).all(), f"{name}: produced NaN or inf")
        check(canvas.to_channels().any(), f"{name}: painted nothing")

    # Targeting a group must leave the other group alone -- this is the failure
    # a fancy-index copy would produce, silently.
    canvas.clear()
    fx.wash(canvas, palette, 1.0, targets=canvas.net_slice("Small Triangle Nets"))
    check(not canvas.nets[0:4].any(), "a small-net wash leaked onto the big nets")
    check(canvas.nets[4:8].any(), "a small-net wash painted nothing")
    return f"{len(fx.PATTERNS)} corridor patterns, {len(paints)} effects, all sane"


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


TESTS = (
    ("channel map", test_layout),
    ("fseq round-trip", test_fseq_roundtrip),
    ("DDP split/reassemble", test_ddp_packets),
    ("UDP loopback -> fseq", test_loopback),
    ("canvas -> channels", test_canvas),
    ("effect vocabulary", test_effects),
    ("fixed script", test_script),
    ("preview geometry", test_geometry),
    ("web UI + engine", test_web),
    ("audio features", test_analysis),
    ("beat clock", test_clock),
    ("audio -> beats", test_beat_pipeline),
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
