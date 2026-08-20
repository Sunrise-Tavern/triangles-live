"""Command line for the live engine.  ``./live.sh <command>``.

Every command here is meant to be runnable on the Mac with no rig attached.
The ones that matter for M1:

* ``selftest``  -- byte-exact round trip through DDP and fseq (no xLights needed)
* ``demo``      -- sender and fake Falcon in one process; writes a capture
* ``render``    -- test pattern straight to fseq, no network, for xLights
* ``pattern``   -- send the test pattern at the real rig (or another machine)
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np
from pathlib import Path

from .ddp import DDP_PORT, DEFAULT_CHANNELS_PER_PACKET, DDPSender
from .fake_falcon import FakeFalcon
from .engine import Engine
from .fseq import FseqWriter, read_header
from .layout import load_layout
from .frame import Canvas
from .preview import Projection, contact_sheet, write_png
from .script import Script
from .testpattern import DURATION, frame as pattern_frame
from .timing import FrameClock

FALCON_IP = "192.168.50.20"   # from xlights_networks.xml


def cmd_layout(args) -> int:
    layout = load_layout()
    print(layout.describe())
    print()
    print(f"corridor : {' -> '.join(layout.arches[:3])} ... {layout.arches[-1]}")
    print(f"nets     : {', '.join(layout.nets)}")
    print(f"par      : {layout.par}")
    return 0


def cmd_selftest(args) -> int:
    from .selftest import main as selftest_main
    return selftest_main()


def cmd_render(args) -> int:
    """Test pattern -> fseq directly.  The reference the capture is compared to."""
    layout = load_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    step_ms = int(round(1000.0 / args.fps))
    started = time.perf_counter()
    count = int(round(args.fps * args.seconds))
    with FseqWriter(out, layout.channel_count, step_time_ms=step_ms) as writer:
        for i in range(count):
            writer.add_frame(pattern_frame(layout, i / args.fps))
    elapsed = time.perf_counter() - started
    print(f"wrote {out}: {count} frames @ {args.fps:g} fps "
          f"({args.seconds:g}s) in {elapsed:.1f}s "
          f"[{elapsed / count * 1000:.2f} ms/frame]")
    return 0


def cmd_pattern(args) -> int:
    """Stream the test pattern over DDP at a real frame rate."""
    layout = load_layout()
    # A real controller only owns part of the channel space.  Sending it the
    # whole show is harmless but noisy; --controller clips to its slice and
    # sends at offset 0, which is what the device expects.
    span = slice(0, layout.channel_count)
    if args.controller:
        controller = layout.output(args.controller)
        span = controller.slice
        print(f"clipped to {controller.name}: channels "
              f"{controller.start}-{controller.end}")
    sender = DDPSender(args.host, port=args.port,
                       channels_per_packet=args.channels_per_packet)
    clock = FrameClock(fps=args.fps)
    limit = None if args.loop else int(round(args.fps * args.seconds))
    print(f"sending {span.stop - span.start} channels to {args.host}:{args.port} "
          f"@ {args.fps:g} fps"
          + ("" if limit is None else f" for {args.seconds:g}s"))
    try:
        for i, t in clock:
            sender.send_frame(pattern_frame(layout, t)[span])
            if limit is not None and i + 1 >= limit:
                break
    except KeyboardInterrupt:
        print()
    finally:
        sender.close()
    print(clock.stats.summary())
    print(f"{sender.frames_sent} frames, {sender.packets_sent} packets")
    return 0


def cmd_fake_falcon(args) -> int:
    from .fake_falcon import main as falcon_main
    argv = ["--port", str(args.port), "--bind", args.bind, "--fps", str(args.fps),
            "--idle", str(args.idle)]
    argv += ["--no-out"] if args.no_out else ["--out", str(args.out)]
    if args.duration:
        argv += ["--duration", str(args.duration)]
    if args.media:
        argv += ["--media", args.media]
    return falcon_main(argv)


def cmd_demo(args) -> int:
    """Both ends in one process: the M1 loop, start to finish."""
    layout = load_layout()
    out = Path(args.out)
    falcon = FakeFalcon(layout=layout, port=args.port, bind="127.0.0.1",
                        out=out, fps=args.fps, media_file=args.media)
    ready = threading.Event()
    thread = threading.Thread(
        target=falcon.run,
        kwargs=dict(duration=args.seconds + 10.0, idle_timeout=1.5, ready=ready),
        daemon=True,
    )
    thread.start()
    if not ready.wait(5.0):
        print("fake Falcon failed to bind", file=sys.stderr)
        return 1

    sender = DDPSender("127.0.0.1", port=args.port,
                       channels_per_packet=args.channels_per_packet)
    clock = FrameClock(fps=args.fps)
    count = int(round(args.fps * args.seconds))
    render_ms = 0.0
    for i, t in clock:
        t0 = time.perf_counter()
        frame = pattern_frame(layout, t)
        render_ms += (time.perf_counter() - t0) * 1000
        sender.send_frame(frame)
        if i + 1 >= count:
            break
    sender.close()
    thread.join(timeout=15.0)

    print()
    print(falcon.report())
    print()
    print(f"clock  : {clock.stats.summary()}")
    print(f"render : {render_ms / count:.2f} ms/frame mean")
    return 0 if falcon.stats.frames == count else 1


def cmd_show(args) -> int:
    """Render the fixed script: to an fseq, or straight out over DDP."""
    canvas = Canvas(load_layout())
    script = Script(canvas, bpm=args.bpm, base_hue=args.hue)
    seconds = args.seconds if args.seconds else script.duration
    count = int(round(args.fps * seconds))
    out = np.zeros(canvas.layout.channel_count, dtype=np.uint8)
    render_ms = 0.0

    if args.send:
        sender = DDPSender(args.host, port=args.port)
        clock = FrameClock(fps=args.fps)
        print(f"streaming {seconds:g}s of script to {args.host}:{args.port} "
              f"@ {args.fps:g} fps")
        try:
            for i, t in clock:
                t0 = time.perf_counter()
                script.render(i, t)
                canvas.to_channels(out, brightness=args.brightness)
                render_ms += (time.perf_counter() - t0) * 1000
                sender.send_frame(out)
                if i + 1 >= count:
                    break
        except KeyboardInterrupt:
            count = clock.stats.frames
        finally:
            sender.close()
        print(clock.stats.summary())
    else:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with FseqWriter(path, canvas.layout.channel_count,
                        step_time_ms=int(round(1000.0 / args.fps))) as writer:
            for i in range(count):
                t0 = time.perf_counter()
                script.render(i, i / args.fps)
                canvas.to_channels(out, brightness=args.brightness)
                render_ms += (time.perf_counter() - t0) * 1000
                writer.add_frame(out)
        print(f"wrote {path}: {count} frames @ {args.fps:g} fps ({seconds:g}s)")

    print(f"render : {render_ms / max(1, count):.2f} ms/frame mean "
          f"(budget 10 ms at {args.fps:g} fps)")
    return 0


def cmd_bench(args) -> int:
    """How much of the frame budget does rendering actually cost?"""
    canvas = Canvas(load_layout())
    script = Script(canvas)
    out = np.zeros(canvas.layout.channel_count, dtype=np.uint8)
    budget = 1000.0 / args.fps

    for i in range(40):                      # warm up numpy's caches
        script.render(i, i / args.fps)
        canvas.to_channels(out)

    stages = []
    for label, work in (
        ("render", lambda i: script.render(i, i / args.fps)),
        ("pack", lambda i: canvas.to_channels(out)),
    ):
        samples = np.empty(args.frames)
        for i in range(args.frames):
            t0 = time.perf_counter()
            work(i)
            samples[i] = (time.perf_counter() - t0) * 1000
        stages.append((label, samples))

    print(f"{args.frames} frames, {canvas.layout.channel_count} channels, "
          f"budget {budget:.1f} ms/frame at {args.fps:g} fps")
    total = np.zeros(args.frames)
    for label, samples in stages:
        total += samples
        print(f"  {label:<8} mean {samples.mean():6.2f}  p95 "
              f"{np.percentile(samples, 95):6.2f}  max {samples.max():6.2f} ms")
    print(f"  {'total':<8} mean {total.mean():6.2f}  p95 "
          f"{np.percentile(total, 95):6.2f}  max {total.max():6.2f} ms "
          f"({total.mean() / budget * 100:.0f}% of budget)")
    return 0 if total.mean() < budget else 1


def cmd_preview(args) -> int:
    """fseq -> PNG.  A contact sheet by default; --at for a single moment."""
    from .fseq import read_all

    layout = load_layout()
    header, data = read_all(args.file)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.at is not None:
        index = min(int(args.at * header.fps), header.frame_count - 1)
        image = Projection(layout).draw(data[index])
        write_png(out, image)
        print(f"wrote {out}: frame {index} ({index / header.fps:.2f}s) "
              f"of {header.frame_count}")
        return 0

    picks = np.linspace(0, header.frame_count - 1, args.sheet).astype(int)
    image = contact_sheet(layout, [data[i] for i in picks], columns=args.columns)
    write_png(out, image)
    print(f"wrote {out}: {args.sheet} frames sampled across "
          f"{header.duration_s:.1f}s, {image.shape[1]}x{image.shape[0]}")
    return 0


def cmd_serve(args) -> int:
    """Run the engine with its browser UI -- the dev preview and the show desk."""
    from .web import serve

    engine = Engine(load_layout(), fps=args.fps, host=args.ddp, port=args.ddp_port,
                    controller=args.controller,
                    record=Path(args.record) if args.record else None)
    url = f"http://{'localhost' if args.bind in ('0.0.0.0', '') else args.bind}:{args.port}"
    print(f"engine  : {args.fps:g} fps -> "
          f"{args.ddp + ':' + str(args.ddp_port) if args.ddp else 'no output'}")
    print(f"open    : {url}")
    try:
        serve(engine, host=args.bind, port=args.port,
              preview_fps=args.preview_fps, detail=args.preview_detail)
    except KeyboardInterrupt:
        pass
    return 0


def cmd_inspect(args) -> int:
    for path in args.files:
        h = read_header(path)
        print(f"{path}")
        print(f"  v{h.version[0]}.{h.version[1]}  {h.channel_count} channels x "
              f"{h.frame_count} frames @ {h.step_time_ms} ms "
              f"({h.fps:g} fps, {h.duration_s:.1f}s)  compression={h.compression_type}")
        for code, data in h.variable_headers.items():
            print(f"  [{code}] {data[:80]!r}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="live", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("layout", help="print the channel map")
    p.set_defaults(func=cmd_layout)

    p = sub.add_parser("selftest", help="byte-exact round trip, no hardware")
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("render", help="test pattern -> fseq, no network")
    p.add_argument("--out", default="out/pattern.fseq")
    p.add_argument("--fps", type=float, default=40.0)
    p.add_argument("--seconds", type=float, default=DURATION)
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("pattern", help="stream the test pattern over DDP")
    p.add_argument("--host", default="127.0.0.1",
                   help=f"target; the rig's Falcon is {FALCON_IP}")
    p.add_argument("--port", type=int, default=DDP_PORT)
    p.add_argument("--fps", type=float, default=40.0)
    p.add_argument("--seconds", type=float, default=DURATION)
    p.add_argument("--loop", action="store_true", help="run until interrupted")
    p.add_argument("--controller", help="clip the send to one controller's channels, "
                                        "e.g. Falcon_F16V5_0E1C")
    p.add_argument("--channels-per-packet", type=int,
                   default=DEFAULT_CHANNELS_PER_PACKET)
    p.set_defaults(func=cmd_pattern)

    p = sub.add_parser("fake-falcon", help="receive DDP and write an fseq")
    p.add_argument("--out", default="out/capture.fseq")
    p.add_argument("--no-out", action="store_true")
    p.add_argument("--port", type=int, default=DDP_PORT)
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--fps", type=float, default=40.0)
    p.add_argument("--idle", type=float, default=5.0)
    p.add_argument("--duration", type=float)
    p.add_argument("--media")
    p.set_defaults(func=cmd_fake_falcon)

    p = sub.add_parser("demo", help="sender + fake Falcon in one process")
    p.add_argument("--out", default="out/capture.fseq")
    p.add_argument("--port", type=int, default=DDP_PORT)
    p.add_argument("--fps", type=float, default=40.0)
    p.add_argument("--seconds", type=float, default=DURATION)
    p.add_argument("--media")
    p.add_argument("--channels-per-packet", type=int,
                   default=DEFAULT_CHANNELS_PER_PACKET)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("show", help="render the fixed 30 s script")
    p.add_argument("--out", default="out/show.fseq")
    p.add_argument("--send", action="store_true", help="stream over DDP instead")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DDP_PORT)
    p.add_argument("--fps", type=float, default=40.0)
    p.add_argument("--seconds", type=float, help="default: the script's own length")
    p.add_argument("--bpm", type=float, default=128.0)
    p.add_argument("--hue", type=float, default=190.0, help="base hue, degrees")
    p.add_argument("--brightness", type=float, default=1.0)
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("bench", help="render cost against the frame budget")
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--fps", type=float, default=40.0)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("preview", help="fseq -> PNG, no xLights needed")
    p.add_argument("file")
    p.add_argument("--out", default="out/preview.png")
    p.add_argument("--at", type=float, help="single frame at this many seconds")
    p.add_argument("--sheet", type=int, default=12, help="frames in the contact sheet")
    p.add_argument("--columns", type=int, default=4)
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("serve", help="run the engine with its browser UI")
    p.add_argument("--bind", default="0.0.0.0", help="web server address")
    p.add_argument("--port", type=int, default=8080, help="web server port")
    p.add_argument("--ddp", help=f"send frames here; the rig's Falcon is {FALCON_IP}")
    p.add_argument("--ddp-port", type=int, default=DDP_PORT)
    p.add_argument("--controller", help="clip output to one controller's channels")
    p.add_argument("--fps", type=float, default=40.0)
    p.add_argument("--record", help="also write every frame to this fseq")
    p.add_argument("--preview-fps", type=float,
                   help="cap the browser feed (default: the engine's rate)")
    p.add_argument("--preview-detail", type=float, default=1.0,
                   help="preview pixel density; 1.0 is ~3300 dots, 2.0 doubles "
                        "it and the bandwidth")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("inspect", help="print an fseq header")
    p.add_argument("files", nargs="+")
    p.set_defaults(func=cmd_inspect)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
