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
from .audio import AutoGain, FileSource, LineInSource
from .config import Config, setup_logging
from .engine import Engine
from .fseq import FseqWriter, read_header
from .layout import load_layout
from .frame import Canvas
from .preview import Projection, contact_sheet, write_png
from .script import Script
from .testpattern import DURATION, frame as pattern_frame
from . import orient
from .timing import FrameClock

# From xlights_networks.xml.  The show spans two of these -- the nets on the
# first, the corridor on the second -- which is why `serve` takes "auto"
# rather than an address.  These are only for the single-target commands
# (`pattern`, `show --send`) that aim at one controller by hand.
FALCON_NETS = "192.168.1.20"
FALCON_CORRIDOR = "192.168.1.30"


def cmd_layout(args) -> int:
    layout = load_layout()
    print(layout.describe())
    print()
    print(f"corridor : {' -> '.join(layout.arches[:3])} ... {layout.arches[-1]}")
    print(f"nets     : {', '.join(layout.nets)}")
    print(f"par      : {layout.par or 'none'}")
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


def cmd_orient(args) -> int:
    """Stream the orientation pattern: bands rising on the nets, one arch at a
    time down the tunnel.  Sends to both Falcons like `serve` does."""
    layout = load_layout()
    canvas = Canvas(layout)
    if args.host.lower() == "auto":
        picks = ([layout.output(args.controller)] if args.controller
                 else layout.ddp_targets())
        sinks = [(c.ip, c.slice, c.name) for c in picks]
    else:
        span = (layout.output(args.controller).slice if args.controller
                else slice(0, layout.channel_count))
        sinks = [(args.host, span, args.controller or "all")]
    if not sinks:
        print("no DDP controller to send to")
        return 1
    senders = [(DDPSender(host, port=args.port,
                          channels_per_packet=args.channels_per_packet), span, name)
               for host, span, name in sinks]
    for _, span, name in senders:
        print(f"-> {name}: channels {span.start + 1}-{span.stop}")
    print("stages: " + ", ".join(f"{n} ({s:g}s)" for n, s in orient.STAGES)
          + " -- Ctrl-C to stop")
    clock = FrameClock(fps=args.fps)
    out = layout.blank_channels()
    last = None
    try:
        for i, t in clock:
            if args.stage:
                t = orient.hold(args.stage, t)
            stage = orient.paint(canvas, t)
            if stage != last:
                print(f"  {stage}")
                last = stage
            canvas.to_channels(out)
            for sender, span, _ in senders:
                sender.send_frame(out[span])
    except KeyboardInterrupt:
        print()
    finally:
        for sender, _, _ in senders:
            sender.close()
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


def _audio_source(args):
    """Build the audio input from the flags, or None for the scripted show."""
    config = getattr(args, "config_obj", None)
    block = config.audio.blocksize if config else 512
    gain = AutoGain() if getattr(args, "autogain", False) else None
    if getattr(args, "audio", None):
        return FileSource(args.audio, realtime=True, blocksize=block,
                          loop=getattr(args, "loop", False), gain=gain)
    if getattr(args, "audio_device", None) is not None:
        device = args.audio_device
        return LineInSource(device=int(device) if str(device).isdigit() else device,
                            blocksize=block, gain=gain)
    return None


def cmd_serve(args) -> int:
    """Run the engine with its browser UI -- the dev preview and the show desk."""
    from .web import serve

    config: Config = args.config_obj
    audio = _audio_source(args)
    session = None
    if args.record_session:
        from .session import Recorder
        session = Recorder(args.record_session, config=config, fps=args.fps,
                           blocksize=config.audio.blocksize)
        print(f"session : recording to {args.record_session}")
    engine = Engine(load_layout(), fps=args.fps, host=args.ddp or None,
                    port=args.ddp_port, controller=args.controller or None,
                    audio=audio, backend=args.backend, window=config.audio.window,
                    silence_dbfs=config.audio.silence_dbfs, session=session,
                    record=Path(args.record) if args.record else None)
    engine.settings.apply({
        "brightness": config.show.brightness,
        "gamma": config.show.gamma,
        "latency_ms": config.show.latency_ms,
        "blackout": config.show.blackout_on_start,
        "rest_level": config.show.rest_level,
    })
    if config.show.preset:
        from . import settings as knobs
        try:
            engine.settings.load(knobs.read_preset(config.show.preset))
            print(f"preset  : {config.show.preset}")
        except (FileNotFoundError, ValueError) as exc:
            print(f"preset  : {exc}", file=sys.stderr)
    print(f"audio   : {args.audio or args.audio_device or 'none (scripted show)'}")
    url = f"http://{'localhost' if args.bind in ('0.0.0.0', '') else args.bind}:{args.port}"
    print(f"engine  : {args.fps:g} fps -> {engine.status.target}")
    print(f"open    : {url}")
    try:
        serve(engine, host=args.bind, port=args.port,
              preview_fps=args.preview_fps, detail=args.preview_detail)
    except KeyboardInterrupt:
        pass
    return 0


def cmd_beats(args) -> int:
    """Measure the beat clock against ground truth."""
    from .verify import main as verify_main

    argv = [*(str(f) for f in args.files), "--fps", str(args.fps),
            "--backend", args.backend]
    if args.bpm:
        argv += ["--bpm", str(args.bpm)]
    if args.faults:
        argv += ["--faults"]
    return verify_main(argv)


def cmd_listen(args) -> int:
    """Watch the analysis and the clock, live, in the terminal."""
    from .audio import AutoGain, FileSource, LineInSource, devices
    from .listener import Listener

    if args.devices:
        print("input devices:")
        print(devices())
        return 0

    if args.file:
        source = FileSource(args.file, realtime=not args.fast,
                            gain=AutoGain() if args.autogain else None)
    else:
        source = LineInSource(device=args.device)
    listener = Listener(source, backend=args.backend)
    print(f"{'time':>7} {'rms':>6} {'energy':>7} {'onset':>6} {'kick':>6} "
          f"{'tempo':>7} {'conf':>5} {'beat':>6} {'phase':>6}  state")
    last = -1.0
    try:
        for block in source.blocks():
            features = listener.step(block)
            if features.t - last < 0.25:
                continue
            last = features.t
            state = listener.clock.state(features.t)
            flag = ("free-run" if state.free_running else
                    "locked" if state.locked else "seeking")
            print(f"{features.t:7.2f} {features.rms:6.3f} {features.energy:7.2f} "
                  f"{features.onset:6.2f} {features.kick:6.2f} "
                  f"{state.tempo:7.2f} {state.confidence:5.2f} "
                  f"{state.beat:6d} {state.beat_phase:6.2f}  {flag}")
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
    clock = listener.clock
    print(f"\n{listener.stats.blocks} blocks, {clock.beats_seen} beats, "
          f"{clock.relocks} relocks, {clock.slips} half-beat corrections, "
          f"{clock.offbeat_events} offbeat detections used for phase")
    return 0


def cmd_sim(args) -> int:
    """The whole loop on one machine: audio in, DDP out, fseq and preview back.

    This is M6's verification in one command -- it runs the real engine, sends
    real packets, and the fake Falcon writes exactly what a controller would
    have received.  With --play the same audio comes out of the speakers, so a
    person can watch the preview against the music instead of trusting numbers.
    """
    import subprocess

    from .fake_falcon import FakeFalcon

    layout = load_layout()
    falcon = FakeFalcon(layout=layout, port=args.port, bind="127.0.0.1",
                        out=Path(args.out), fps=args.fps, media_file=args.file,
                        quiet=True)
    ready = threading.Event()
    receiver = threading.Thread(
        target=falcon.run,
        kwargs=dict(duration=args.seconds + 30.0, idle_timeout=2.0, ready=ready),
        daemon=True)
    receiver.start()
    if not ready.wait(5.0):
        print("fake Falcon failed to bind", file=sys.stderr)
        return 1

    engine = Engine(layout, fps=args.fps, host="127.0.0.1", port=args.port,
                    audio=FileSource(args.file, realtime=True, loop=args.loop),
                    backend=args.backend)
    player = None
    if args.play:
        player = subprocess.Popen(["afplay", args.file],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
    print(f"simulating {args.file} for {args.seconds:g}s at {args.fps:g} fps")
    engine.start()
    started = time.perf_counter()
    try:
        while time.perf_counter() - started < args.seconds:
            time.sleep(0.5)
            if engine.listener is not None and not engine.listener.stats.running \
                    and engine.status.frames > 10:
                break
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        if player is not None:
            player.terminate()
    receiver.join(timeout=10.0)

    d = engine.status.to_dict()
    print()
    print(falcon.report())
    print()
    print(f"engine  : {d['fps']:.2f} fps, {d['render_ms']:.2f} ms/frame, "
          f"{d['late_frames']} late, {d['skipped_frames']} skipped")
    print(f"audio   : lag {d['audio_lag_ms']:.1f} ms, tempo {d['bpm']:.2f}, "
          f"confidence {d['confidence']:.2f}, bar confidence {d['bar_confidence']:.2f}")
    print(f"last    : state {d['scene']}, corridor {d['pattern']}")
    return 0 if falcon.stats.frames else 1


def cmd_corpus(args) -> int:
    """Run a folder of tracks through the live chain and flag the bad ones."""
    from .corpus import main as corpus_main

    argv = [*(str(p) for p in args.paths), "--backend", args.backend,
            "--fps", str(args.fps)]
    if args.no_render:
        argv.append("--no-render")
    if args.offline:
        argv.append("--offline")
    if args.out:
        argv += ["--out", str(args.out)]
    return corpus_main(argv)


def cmd_harmonix(args) -> int:
    """Ground truth from the Harmonix Set: what it covers, and how we score."""
    from . import harmonix

    entries = harmonix.load()
    if args.list:
        rows = entries
        if args.genre:
            rows = [e for e in rows if args.genre.lower() in e.genre.lower()]
        if args.not_four_four:
            rows = [e for e in rows if not e.four_four]
        if args.min_bpm:
            rows = [e for e in rows if e.bpm >= args.min_bpm]
        if args.max_bpm:
            rows = [e for e in rows if e.bpm <= args.max_bpm]
        print(f"{len(rows)} of {len(entries)} entries match")
        for e in rows[:args.limit]:
            print(f"  {e.bpm:5.0f} {e.time_signature:<5} {e.genre[:16]:<18}"
                  f"{e.title[:38]:<40}{e.artist[:26]}")
        return 0

    if args.crossfade:
        from .harmonix import crossfade, score_transition

        pool = [e for e in entries if "Dance" in e.genre and 118 <= e.bpm <= 142]
        pool = harmonix.sample(pool, 8, seed=3)
        pairs = [(pool[i], pool[i + 1]) for i in range(0, len(pool) - 1, 2)]
        scenarios = [
            ("beatmatched", dict(match_tempo=True, overlap=16.0)),
            ("bar-misaligned", dict(match_tempo=True, overlap=16.0, beat_offset=2)),
            ("tempo jump", dict(match_tempo=False, overlap=16.0)),
            ("hard cut", dict(match_tempo=False, overlap=0.5)),
        ]
        print(f"{'scenario':<26}{'tempo':>13}{'before':>7}{'after':>8}"
              f"{'downbt':>8}{'settle':>8}{'excurs':>8}{'conf':>7}{'rlk':>5}")
        print("-" * 98)
        rows = []
        for name, kwargs in scenarios:
            for a, b in pairs:
                try:
                    tr = crossfade(a, b, label=f"{name}: {a.bpm:.0f}->{b.bpm:.0f}",
                                   **kwargs)
                except ValueError:
                    continue
                score = score_transition(tr, fps=args.fps, backend=args.backend)
                rows.append((name, score))
                print(score.line(), flush=True)
        import numpy as _np
        print()
        for name, _ in scenarios:
            group = [s for n, s in rows if n == name]
            if not group:
                continue
            settled = [s.settle_s for s in group if s.settle_s is not None]
            print(f"  {name:<16} after {100 * _np.median([s.precision_after for s in group]):3.0f}%"
                  f"  downbeats {100 * _np.median([s.downbeats_after for s in group]):3.0f}%"
                  f"  settled {len(settled)}/{len(group)}"
                  + (f" in {_np.median(settled):.1f}s median" if settled else "")
                  + f"  confidence trough {_np.median([s.confidence_trough for s in group]):.2f}")
        return 0

    if args.synth:
        pool = entries
        if args.genre:
            pool = [e for e in pool if args.genre.lower() in e.genre.lower()]
        if args.not_four_four:
            pool = [e for e in pool if not e.four_four]
        if args.min_bpm:
            pool = [e for e in pool if e.bpm >= args.min_bpm]
        if args.max_bpm:
            pool = [e for e in pool if e.bpm <= args.max_bpm]
        picked = harmonix.sample(pool, args.synth, seed=args.seed)
        print(f"rendering and scoring {len(picked)} annotations "
              f"(no audio needed -- the structures are real, the timbres ours)")
        print(f"{'title':<30}{'genre':<14}{'bpm':>5}{'ours':>7}{'sig':>6}"
              f"{'prec':>6}{'rec':>6}{'downbt':>8}{'shift':>5}{'wander':>7}"
              f"{'state':>7}{'lock':>6}")
        print("-" * 112)
        scored = []
        for entry in picked:
            result = harmonix.score_synthetic(entry, fps=args.fps,
                                              backend=args.backend)
            scored.append(result)
            print(result.line(), flush=True)
        ok = [r for r in scored if not r.error]
        if ok:
            import numpy as _np
            print()
            print(f"{len(ok)} scored")
            print(f"  beat precision     : median "
                  f"{100 * float(_np.median([r.beats_pct for r in ok])):.0f}%"
                  f"   worst {100 * min(r.beats_pct for r in ok):.0f}%")
            print(f"  beat recall        : median "
                  f"{100 * float(_np.median([r.beats_recall for r in ok])):.0f}%"
                  f"   (high recall + low precision = we are at double tempo)")
            print(f"  downbeats within 50 ms : median "
                  f"{100 * float(_np.median([r.downbeats_pct for r in ok])):.0f}%"
                  f"   worst {100 * min(r.downbeats_pct for r in ok):.0f}%")
            print(f"  tempo error        : median "
                  f"{100 * float(_np.median([r.tempo_error for r in ok])):.1f}%"
                  f"   over 5%: {sum(r.tempo_error > 0.05 for r in ok)} track(s)")
            print(f"  state agreement    : median "
                  f"{100 * float(_np.median([r.state_agreement for r in ok])):.0f}%")
            print(f"  tempo wander       : median "
                  f"{float(_np.median([r.tempo_spread for r in ok])):.1f} BPM"
                  f"   over 10 BPM: {sum(r.tempo_spread > 10 for r in ok)} track(s)")
            print(f"  clock locked       : median "
                  f"{100 * float(_np.median([r.locked for r in ok])):.0f}%")
            bad = [r for r in ok if r.beats_pct < 0.5]
            if bad:
                print(f"  poor beat tracking : {len(bad)} track(s) below 50% -- "
                      + ", ".join(f"{r.entry.title[:22]} ({r.entry.bpm:.0f}bpm "
                                  f"{r.entry.time_signature})" for r in bad[:6]))
        return 0

    if not args.paths:
        print("Give a music folder to scan, --list to browse the set, or "
              "--synth N to score against rendered annotations.")
        return 1

    matched, missed = harmonix.scan([Path(p) for p in args.paths], entries,
                                    threshold=args.threshold)
    print(f"{len(matched)} of {len(matched) + len(missed)} files have "
          f"annotations")
    for m in matched:
        flag = "  (edition mismatch)" if m.suspect_edition else ""
        print(f"  {m.score:.2f} {m.how:<14} {m.path.name[:44]:<46}"
              f"-> {m.entry.title[:26]} / {m.entry.artist[:18]}{flag}")
    if missed and args.show_missed:
        print(f"\nno annotation for {len(missed)} file(s):")
        for p in missed[:args.limit]:
            print(f"  {p.name}")
    if not matched or not args.score:
        return 0

    print()
    print(f"{'title':<32}{'artist':<20}{'bpm ref/ours':<12}{'beats':>6}"
          f"{'downbt':>7}{'state':>7}  notes")
    print("-" * 116)
    for m in matched:
        result = harmonix.score(m.path, m, fps=args.fps, backend=args.backend)
        print(result.line())
    return 0


def cmd_doctor(args) -> int:
    """Everything that has to be true before the doors open, checked."""
    from .doctor import run_checks
    return run_checks(args.config_obj, deep=not args.quick)


def cmd_analyze(args) -> int:
    """Read back a recorded session and say what went wrong."""
    from .session import analyze
    return analyze(Path(args.directory), replay=args.replay, width=args.window)


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


def build_parser(config: Config | None = None) -> argparse.ArgumentParser:
    config = config or Config()
    ap = argparse.ArgumentParser(prog="live", description=__doc__.splitlines()[0])
    # --config is accepted both before and after the subcommand.  The
    # systemd unit puts it after ("python -m live serve --config ..."), which
    # a bare top-level option silently rejects -- caught by actually running
    # the unit's ExecStart line rather than something like it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", metavar="PATH",
                        help="config file (default: live.toml, or "
                             "$TRIANGLES_CONFIG)")
    ap.add_argument("--config", metavar="PATH",
                    help="config file (default: live.toml, or $TRIANGLES_CONFIG)")
    ap.set_defaults(config_obj=config)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("layout", help="print the channel map", parents=[common])
    p.set_defaults(func=cmd_layout)

    p = sub.add_parser("selftest", help="byte-exact round trip, no hardware", parents=[common])
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("render", help="test pattern -> fseq, no network", parents=[common])
    p.add_argument("--out", default="out/pattern.fseq")
    p.add_argument("--fps", type=float, default=40.0)
    p.add_argument("--seconds", type=float, default=DURATION)
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("pattern", help="stream the test pattern over DDP", parents=[common])
    p.add_argument("--host", default="127.0.0.1",
                   help=f"target; nets {FALCON_NETS}, corridor {FALCON_CORRIDOR}")
    p.add_argument("--port", type=int, default=DDP_PORT)
    p.add_argument("--fps", type=float, default=40.0)
    p.add_argument("--seconds", type=float, default=DURATION)
    p.add_argument("--loop", action="store_true", help="run until interrupted")
    p.add_argument("--controller", help="clip the send to one controller's channels, "
                                        "e.g. Falcon_F16V5_0E1C")
    p.add_argument("--channels-per-packet", type=int,
                   default=DEFAULT_CHANNELS_PER_PACKET)
    p.set_defaults(func=cmd_pattern)

    p = sub.add_parser("orient", help="orientation check: bands rise on the nets, "
                                      "one arch at a time down the tunnel",
                       parents=[common])
    p.add_argument("--host", default="auto",
                   help='"auto" = every Falcon in xlights_networks.xml (default); '
                        "or one address")
    p.add_argument("--port", type=int, default=DDP_PORT)
    p.add_argument("--fps", type=float, default=40.0)
    p.add_argument("--controller", help="send to one controller only")
    p.add_argument("--stage", choices=[n for n, _ in orient.STAGES],
                   help="hold one stage instead of cycling")
    p.add_argument("--channels-per-packet", type=int,
                   default=DEFAULT_CHANNELS_PER_PACKET)
    p.set_defaults(func=cmd_orient)

    p = sub.add_parser("fake-falcon", help="receive DDP and write an fseq", parents=[common])
    p.add_argument("--out", default="out/capture.fseq")
    p.add_argument("--no-out", action="store_true")
    p.add_argument("--port", type=int, default=DDP_PORT)
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--fps", type=float, default=40.0)
    p.add_argument("--idle", type=float, default=5.0)
    p.add_argument("--duration", type=float)
    p.add_argument("--media")
    p.set_defaults(func=cmd_fake_falcon)

    p = sub.add_parser("demo", help="sender + fake Falcon in one process", parents=[common])
    p.add_argument("--out", default="out/capture.fseq")
    p.add_argument("--port", type=int, default=DDP_PORT)
    p.add_argument("--fps", type=float, default=40.0)
    p.add_argument("--seconds", type=float, default=DURATION)
    p.add_argument("--media")
    p.add_argument("--channels-per-packet", type=int,
                   default=DEFAULT_CHANNELS_PER_PACKET)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("show", help="render the fixed 30 s script", parents=[common])
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

    p = sub.add_parser("bench", help="render cost against the frame budget", parents=[common])
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--fps", type=float, default=40.0)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("preview", help="fseq -> PNG, no xLights needed", parents=[common])
    p.add_argument("file")
    p.add_argument("--out", default="out/preview.png")
    p.add_argument("--at", type=float, help="single frame at this many seconds")
    p.add_argument("--sheet", type=int, default=12, help="frames in the contact sheet")
    p.add_argument("--columns", type=int, default=4)
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("serve", help="run the engine with its browser UI", parents=[common])
    p.add_argument("--bind", default=config.web.bind, help="web server address")
    p.add_argument("--port", type=int, default=config.web.port,
                   help="web server port")
    p.add_argument("--ddp", default=config.output.host,
                   metavar="HOST", 
                   help=f"send frames here; nets {FALCON_NETS}, "
                        f"corridor {FALCON_CORRIDOR}")
    p.add_argument("--ddp-port", type=int, default=config.output.port)
    p.add_argument("--controller", default=config.output.controller,
                   help="clip output to one controller's channels")
    p.add_argument("--fps", type=float, default=config.output.fps)
    p.add_argument("--record", help="also write every frame to this fseq")
    p.add_argument("--record-session", metavar="DIR",
                   help="record the audio and every analysis decision, for "
                        "diagnosing what actually happened")
    p.add_argument("--preview-fps", type=float,
                   default=config.web.preview_fps or None,
                   help="cap the browser feed (default: the engine's rate)")
    p.add_argument("--preview-detail", type=float, default=config.web.preview_detail,
                   help="preview pixel density; 1.0 is ~3300 dots, 2.0 doubles "
                        "it and the bandwidth")
    p.add_argument("--audio", default=config.audio.file or None,
                   help="drive the show from this audio file")
    p.add_argument("--audio-device", default=config.audio.device or None,
                   help="drive it from a line input (see 'live listen --devices')")
    p.add_argument("--backend", default=config.audio.backend,
                   help="beat tracker backend")
    p.add_argument("--loop", action="store_true", default=config.audio.loop,
                   help="loop the audio file")
    p.add_argument("--autogain", action="store_true", default=config.audio.autogain)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("beats", help="measure the beat clock against ground truth", parents=[common])
    p.add_argument("files", nargs="*")
    p.add_argument("--bpm", type=float, help="known tempo: compare to an exact grid")
    p.add_argument("--fps", type=float, default=40.0)
    p.add_argument("--backend", default="aubio")
    p.add_argument("--faults", action="store_true",
                   help="silence and tempo-change tests")
    p.set_defaults(func=cmd_beats)

    p = sub.add_parser("listen", help="watch analysis + clock in the terminal", parents=[common])
    p.add_argument("file", nargs="?", help="audio file; omit to use line-in")
    p.add_argument("--device", help="input device index or name")
    p.add_argument("--devices", action="store_true", help="list inputs and exit")
    p.add_argument("--backend", default="aubio")
    p.add_argument("--fast", action="store_true",
                   help="run a file as fast as possible instead of in real time")
    p.add_argument("--autogain", action="store_true")
    p.set_defaults(func=cmd_listen)

    p = sub.add_parser("sim", help="the whole loop: audio -> DDP -> fseq", parents=[common])
    p.add_argument("file", help="audio to drive the show")
    p.add_argument("--out", default="out/sim.fseq")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--fps", type=float, default=40.0)
    p.add_argument("--port", type=int, default=DDP_PORT)
    p.add_argument("--backend", default="aubio")
    p.add_argument("--play", action="store_true",
                   help="play the audio too, so you can watch against it")
    p.add_argument("--loop", action="store_true",
                   help="restart the audio when it ends -- for soak tests")
    p.set_defaults(func=cmd_sim)

    p = sub.add_parser("corpus", help="validate against a folder of tracks", parents=[common])
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--backend", default="aubio")
    p.add_argument("--fps", type=float, default=40.0)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--offline", action="store_true",
                   help="also run the offline segmenter, to compare boundaries")
    p.add_argument("--out", type=Path)
    p.set_defaults(func=cmd_corpus)

    p = sub.add_parser("harmonix", help="ground truth from the Harmonix Set", parents=[common])
    p.add_argument("paths", nargs="*", help="music folders or files to scan")
    p.add_argument("--list", action="store_true", help="browse the annotations")
    p.add_argument("--genre")
    p.add_argument("--not-four-four", action="store_true",
                   help="only tracks that are not in 4/4")
    p.add_argument("--min-bpm", type=float)
    p.add_argument("--max-bpm", type=float)
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--threshold", type=float, default=0.82)
    p.add_argument("--show-missed", action="store_true")
    p.add_argument("--synth", type=int, metavar="N",
                   help="render N annotations as audio and score against them")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--crossfade", action="store_true",
                   help="mix pairs of tracks and score the transition")
    p.add_argument("--score", action="store_true",
                   help="run the live chain on matched files and grade it")
    p.add_argument("--fps", type=float, default=40.0)
    p.add_argument("--backend", default="aubio")
    p.set_defaults(func=cmd_harmonix)

    p = sub.add_parser("doctor", help="preflight: is this machine ready to run", parents=[common])
    p.add_argument("--quick", action="store_true",
                   help="skip the render and beat-tracking benchmarks")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("analyze", help="diagnose a recorded session",
                       parents=[common])
    p.add_argument("directory")
    p.add_argument("--window", type=float, default=30.0,
                   help="seconds per row of the per-window table")
    p.add_argument("--replay", action="store_true",
                   help="rerun the whole chain on the recorded audio")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("inspect", help="print an fseq header", parents=[common])
    p.add_argument("files", nargs="+")
    p.set_defaults(func=cmd_inspect)

    return ap


def main(argv: list[str] | None = None) -> int:
    # Pre-parse just --config, so the real parser can take its defaults from
    # the file.  argparse cannot do this in one pass.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _rest = pre.parse_known_args(argv)
    try:
        config = Config.load(known.config)
    except (ValueError, OSError) as exc:
        print(f"config: {exc}", file=sys.stderr)
        return 2

    args = build_parser(config).parse_args(argv)
    args.config_obj = config
    if args.command in ("serve", "sim"):
        setup_logging(config.log)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
