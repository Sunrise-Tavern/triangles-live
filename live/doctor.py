"""Preflight: is this machine actually ready to run the show?

Written for the ten minutes before doors, over SSH, on a Pi you cannot see.
Every check answers a question that has an obvious remedy, and says which --
"no signal on the input" is a cable, "aubio missing" is an apt install, "12 ms
a frame at 40 fps" means drop to 30.

The distinction that matters is **error versus warning**.  An error means the
show will not run; a warning means it will run in a way you should know about
(no Falcon configured, so nothing lights; no audio, so it falls back to the
scripted show).  A laptop with nothing set up should come out all-green on the
errors, or the check is noise and nobody will run it.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config

OK, WARN, FAIL = "ok", "warn", "FAIL"
MARK = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    remedy: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "", remedy: str = "") -> None:
        self.checks.append(Check(name, status, detail, remedy))

    @property
    def failed(self) -> int:
        return sum(c.status == FAIL for c in self.checks)

    def render(self) -> str:
        width = max(len(c.name) for c in self.checks) + 2
        lines = []
        for c in self.checks:
            lines.append(f"[{MARK[c.status]}] {c.name:<{width}} {c.detail}")
            if c.remedy and c.status != OK:
                lines.append(f"{'':>{width + 10}}-> {c.remedy}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #


def check_layout(report: Report) -> None:
    try:
        from .layout import load_layout
        layout = load_layout()
    except Exception as exc:                       # noqa: BLE001
        report.add("layout", FAIL, f"{type(exc).__name__}: {exc}",
                   "set TRIANGLES_SHOW_DIR, or [xlights] show_dir in live.toml, "
                   "to the xLights show folder")
        return
    orphans = layout.unaddressed()
    from triseq.show import show_dir
    report.add("layout", OK,
               f"{len(layout.models)} models, {layout.channel_count} channels, "
               f"{len(layout.arches)} arches, from {show_dir()}")
    if orphans:
        report.add("addressing", WARN,
                   f"{len(orphans)} models outside every controller "
                   f"({orphans[0].name} ... ch {orphans[-1].end})",
                   "those channels go nowhere: widen a controller's MaxChannels "
                   "in xlights_networks.xml, or move the model")


def check_tools(report: Report) -> None:
    if shutil.which("ffmpeg"):
        report.add("ffmpeg", OK, "present")
    else:
        report.add("ffmpeg", FAIL, "not on PATH",
                   "apt install ffmpeg   (needed to read audio files)")
    try:
        import aubio
        report.add("aubio", OK, f"version {aubio.version}")
    except Exception as exc:                       # noqa: BLE001
        report.add("aubio", FAIL, f"{exc}",
                   "apt install python3-aubio, and make the venv with "
                   "--system-site-packages")


def check_audio(report: Report, config: Config, deep: bool) -> None:
    try:
        import sounddevice as sd
    except Exception as exc:                       # noqa: BLE001
        report.add("audio library", FAIL, str(exc),
                   "apt install libportaudio2 && pip install sounddevice")
        return

    try:
        inputs = [(i, d) for i, d in enumerate(sd.query_devices())
                  if d["max_input_channels"] > 0]
    except Exception as exc:                       # noqa: BLE001
        report.add("audio devices", FAIL, str(exc))
        return
    if not inputs:
        report.add("audio devices", WARN, "no input devices at all",
                   "plug in the USB interface; the show will run scripted")
        return
    report.add("audio devices", OK,
               "; ".join(f"[{i}] {d['name'][:28]}" for i, d in inputs[:3]))

    if config.audio.file:
        path = Path(config.audio.file)
        report.add("audio source", OK if path.exists() else FAIL,
                   f"file {path}" + ("" if path.exists() else " (missing)"),
                   "" if path.exists() else "fix [audio] file in the config")
        return

    wanted = config.audio.device
    device = int(wanted) if str(wanted).isdigit() else (wanted or None)
    if not deep:
        report.add("audio input", OK, f"device {wanted or 'default'} (not opened)")
        return

    # Actually open it and listen.  This is the check that earns its keep: a
    # device can exist, be selected, and be silent because nobody plugged the
    # aux cable in, and nothing else here would notice.
    try:
        recorded = sd.rec(int(2.0 * 44100), samplerate=44100, channels=1,
                          dtype="float32", device=device)
        sd.wait()
    except Exception as exc:                       # noqa: BLE001
        report.add("audio input", FAIL, f"could not open {wanted or 'default'}: {exc}",
                   "check the device index with `live listen --devices`")
        return
    peak = float(np.abs(recorded).max())
    rms = float(np.sqrt(np.mean(recorded.astype(np.float64) ** 2)))
    dbfs = 20 * np.log10(peak) if peak > 1e-9 else -120.0
    rms_dbfs = 20 * np.log10(rms) if rms > 1e-9 else -120.0
    if peak > 0.99:
        report.add("audio input", WARN, f"clipping ({dbfs:.1f} dBFS peak)",
                   "turn the send down; a clipped feed ruins onset detection")
    else:
        report.add("audio input", OK,
                   f"{dbfs:.1f} dBFS peak, {rms_dbfs:.1f} dBFS rms")

    # What "silence" means depends entirely on the input -- about 40 dB
    # between a line feed and a room mic -- so measure it rather than assume.
    # Whatever was playing during the sample sets the reading; the useful run
    # is with nothing playing.
    configured = config.audio.silence_dbfs
    suggested = round(rms_dbfs + 8)
    if rms_dbfs > -45.0:
        # Measured: with a -36 dBFS floor, a track's quiet passages land within
        # 0-4 dB of the room itself at every playback level, so no threshold
        # can separate them.  This is the "never a room mic" rule, as a number.
        report.add("silence threshold", WARN,
                   f"input floor is {rms_dbfs:.0f} dBFS -- too high to tell "
                   f"silence from quiet music",
                   "use a line feed, or a loopback device for desktop audio; "
                   "with a room mic set silence_dbfs = -100 to switch the "
                   "idle state off rather than have it flicker")
    elif rms_dbfs > configured:
        report.add("silence threshold", WARN,
                   f"input is {rms_dbfs:.0f} dBFS but [audio] silence_dbfs is "
                   f"{configured:.0f} -- silence will never be detected",
                   f"if nothing was playing just now, set silence_dbfs = "
                   f"{suggested}")
    else:
        report.add("silence threshold", OK,
                   f"{configured:.0f} dBFS, input measured {rms_dbfs:.0f} "
                   f"({configured - rms_dbfs:+.0f} dB of headroom)")


def check_output(report: Report, config: Config) -> None:
    """Every DDP receiver the show needs, not just the first one.

    A rig on two Falcons fails asymmetrically: the nets light, the corridor
    stays dark, and nothing anywhere reports an error, because DDP is one-way
    UDP.  So each target is checked by name and each gets its own line.
    """
    host = config.output.host
    if not host:
        report.add("falcon", WARN, "no host configured -- rendering only",
                   "set [output] host in the config to light anything")
        return

    if host.lower() == "auto":
        from .layout import load_layout
        try:
            targets = [(c.name, c.ip) for c in load_layout().ddp_targets()]
        except Exception as exc:                        # noqa: BLE001
            report.add("falcon", FAIL, f"cannot read the controller list: {exc}",
                       "check xlights_networks.xml")
            return
        if not targets:
            report.add("falcon", FAIL,
                       "[output] host is 'auto' but no DDP controllers exist",
                       "add them in xlights_networks.xml, or set an address")
            return
    else:
        targets = [("falcon", host)]

    for name, addr in targets:
        label = f"{name} {addr}" if name != "falcon" else addr
        try:
            socket.gethostbyname(addr)
        except OSError as exc:
            report.add("falcon", FAIL, f"{label} does not resolve: {exc}",
                       "check the IP and that this machine is on the show LAN")
            continue
        reachable = subprocess.run(
            ["ping", "-c", "1", "-W", "1", addr] if not _is_mac()
            else ["ping", "-c", "1", "-t", "1", addr],
            capture_output=True).returncode == 0
        if reachable:
            report.add("falcon", OK,
                       f"{label}:{config.output.port} responds to ping")
        else:
            report.add("falcon", WARN, f"{label} does not answer ping",
                       "DDP is one-way UDP so it may still work, but check the "
                       "cable and that the Falcon is powered")


def _is_mac() -> bool:
    import platform
    return platform.system() == "Darwin"


def check_render(report: Report, config: Config) -> None:
    from .frame import Canvas
    from .layout import load_layout
    from .script import Script

    canvas = Canvas(load_layout())
    script = Script(canvas)
    out = np.zeros(canvas.layout.channel_count, dtype=np.uint8)
    for i in range(30):                            # warm up numpy
        script.render(i, i / 40.0)
        canvas.to_channels(out)
    started = time.perf_counter()
    frames = 200
    for i in range(frames):
        script.render(i, i / 40.0)
        canvas.to_channels(out)
    per_frame = (time.perf_counter() - started) / frames * 1000
    budget = 1000.0 / config.output.fps
    share = per_frame / budget
    if share < 0.35:
        report.add("render budget", OK,
                   f"{per_frame:.2f} ms/frame, {100 * share:.0f}% of "
                   f"{budget:.0f} ms at {config.output.fps:g} fps")
    elif share < 0.7:
        report.add("render budget", WARN,
                   f"{per_frame:.2f} ms/frame, {100 * share:.0f}% of budget",
                   "leaves little room for the audio thread; consider 30 fps")
    else:
        report.add("render budget", FAIL,
                   f"{per_frame:.2f} ms/frame, {100 * share:.0f}% of budget",
                   f"drop [output] fps to 30, or thin the analysis hop")


def check_clock(report: Report) -> None:
    from .audio import ArraySource
    from .clock import BeatClock
    from .verify import Report as BeatReport, click_track, run

    audio, grid = click_track([(128.0, 12.0)], accent_every=4)
    clock = BeatClock()
    started = time.perf_counter()
    fired, listener = run(ArraySource(audio), clock=clock)
    elapsed = time.perf_counter() - started
    speed = (len(audio) / 44100.0) / max(elapsed, 1e-6)
    result = BeatReport("clicks", fired, grid, listener)
    errors = np.abs(result.errors)
    within = float(np.mean(errors < 30)) if len(errors) else 0.0
    detail = (f"{100 * within:.0f}% of beats within 30 ms, tempo "
              f"{clock.tempo:.1f}, {speed:.0f}x real time")
    if within > 0.8 and speed > 3.0:
        report.add("beat tracking", OK, detail)
    elif speed <= 3.0:
        report.add("beat tracking", FAIL, detail,
                   "analysis is too slow for real time on this machine")
    else:
        report.add("beat tracking", WARN, detail,
                   "check the aubio build; this should be near 100% on clicks")


def check_storage(report: Report, config: Config) -> None:
    from .config import ROOT
    path = Path(config.log.file)
    if not path.is_absolute():
        path = ROOT / path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        probe = path.parent / ".doctor"
        probe.write_text("x")
        probe.unlink()
    except OSError as exc:
        report.add("log directory", FAIL, f"{path.parent}: {exc}")
        return
    free = shutil.disk_usage(path.parent).free / 1e9
    report.add("log directory", OK if free > 0.5 else WARN,
               f"{path.parent} writable, {free:.1f} GB free",
               "" if free > 0.5 else "low disk; logs rotate but captures do not")


def run_checks(config: Config, deep: bool = True) -> int:
    report = Report()
    report.add("config", OK,
               str(config.source) if config.source else "defaults (no live.toml)")
    check_layout(report)
    check_tools(report)
    check_audio(report, config, deep)
    check_output(report, config)
    check_storage(report, config)
    if deep:
        check_render(report, config)
        check_clock(report)

    print(report.render())
    print()
    warns = sum(c.status == WARN for c in report.checks)
    if report.failed:
        print(f"{report.failed} failure(s), {warns} warning(s) -- "
              f"the show will not run correctly.")
    elif warns:
        print(f"No failures, {warns} warning(s) -- read them, then you are good.")
    else:
        print("All clear.")
    return 1 if report.failed else 0
