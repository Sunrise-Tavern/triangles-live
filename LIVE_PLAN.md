# Live engine — plan (audio-driven)

Real-time lighting for the Triangles rig, driven by **live audio analysis** of
the DJ feed, rendered on a Raspberry Pi, sent to the Falcon over DDP.
Developed and validated on the Mac in simulation first; deployed to the Pi
unchanged.

**Why audio:** the XDJ-RX3 has no LAN port, so Pro DJ Link (beat/track data
over Ethernet) is not available. The only live signal is the sound.

**Decisions**
- Python end to end (headless on the Pi). Analysis via **aubio** (C, streaming)
  — not librosa, which is offline-only and too heavy for a Pi.
- **Predict beats, don't react to them.** The clock tracks tempo + phase and
  fires effects at the *predicted* beat, so pipeline latency doesn't land the
  lights late.
- No offline lookahead exists live, so section structure becomes a **rolling
  state machine** (energy vs. a slow baseline, with hysteresis) rather than
  segmentation.
- XR16 stays off the network for now. Its OSC meters are an optional later add.
- The daemon exposes a **browser UI** (preview + controls + presets) — it is both
  the dev preview and the show-time control surface on the Pi. See M3.
- Third-party engines reviewed (see "Libraries reviewed"): **LedFx** — reference,
  not a base; **BeatNet** — optional second clock backend behind a spike, aubio
  stays the default.

---

## Hardware

### At the event

| Item | Purpose | Notes |
|---|---|---|
| Raspberry Pi 4 (2 GB+) or **Pi 5**, PSU, case with fan, SD card | runs the engine | Pi 5 gives real headroom for the DSP + 40 fps render. Dedicated box, not the FPP one |
| **USB audio interface**, class-compliant, with **line input** (e.g. Behringer UCA222 / U-Phoria UMC22, Focusrite Solo) | the audio feed into the Pi | Pi has no line-in. Mono is enough |
| **Small gigabit switch** (5 ports) or router | Pi ↔ Falcon (↔ Mac during setup) | Falcon is static `192.168.50.20`; give the Pi a static `192.168.50.x` |
| Cat5e/6 cables ×3 (+ spares) | Pi, Falcon, Mac | |
| ¼" TRS / RCA cable | XR16 aux out → interface line-in | |
| *Arches + par controller* | **open question** — see bottom | depends on how they're actually wired today |

Already have: XDJ-RX3, XR16, Falcon F16V5, FPP box, Mac.

### Audio feed (XR16)

Create a **bus** on the XR16: post-fader from the RX3 input channels, mono,
**no EQ/comp/limiter**, routed to an **aux out** → interface line-in. Dedicated
lighting feed nobody touches mid-set. Never a room mic.

Note: the XR16 is **not** usable as a computer sound card — its USB-A port
records to a USB stick (host port). The 18×18 USB interface is on the XR18/X18.
So the small USB interface is required at the rig. For dev on the Mac, skip
hardware entirely and stream an mp3 through the pipeline (see sim).

---

## Architecture

```
mp3 (sim) / line-in (rig)
        │
   audio capture  (sounddevice, ~512-sample blocks @ 44.1k ≈ 11.6 ms)
        │
   causal analysis  ─ onset strength (aubio)
                    ─ tempo + beat tracker (aubio tempo / BTrack) → phase, next-beat prediction
                    ─ band energy (bass / mid / high), rolling RMS
        │
   beat clock  ─ tempo, phase, predicted next beat/bar; confidence; free-runs on loss
   state machine ─ quiet / cruising / building / hot  (energy vs slow baseline + hysteresis)
        │
   live arranger  ─ section-kind → treatment (recipes, palettes, corridor patterns, articulation)
        │
   renderer  ─ per-model pixel buffers @ 40 fps
        │
   DDP sender ─▶ Falcon                       (sim: fake Falcon → .fseq → xLights)
        │
   web server (FastAPI/aiohttp, same process) ─▶ browser
        ─ live preview (frames downsampled over WebSocket onto layout geometry)
        ─ controls: enable/disable, blackout, master brightness, style knobs
        ─ presets: named JSON snapshots of all knobs, load/save
```

Layout: `generated/live/` alongside `triseq/`. Reuse: `show.py`, `palettes.py`,
corridor pattern library, section→treatment recipes, density/articulation logic.
`analysis.py`/`structure.py` are **not** reused — they are offline by design.

---

## Next session — milestones, each with its own verification

Ordered so every step is testable on the Mac with no hardware.

### M1 · DDP out + fake Falcon (the loop's two ends) — **done**
- `live/ddp.py` — DDP sender: 10-byte header (flags, seq, type, dest, 32-bit
  channel offset, 16-bit length), 1440 ch/packet (matches the Falcon config),
  push flag on a frame's last packet; `FrameAssembler` for the other end.
- `live/layout.py` — the channel map, read from `xlights_rgbeffects.xml` +
  `xlights_networks.xml`: controller-relative start channels, per-string colour
  order (nets RGB, arches **GRB**), custom-model node maps, polyline runs.
  Separate from `triseq/show.py`, which knows names but not addresses.
- `live/fake_falcon.py` — UDP receiver reassembling frames, reporting packet
  loss and per-model coverage, writing `.fseq` (v2.0 uncompressed —
  `live/fseq.py`, header documented in its docstring).
- `live/timing.py` — drift-free frame clock that reports lateness.
- `live/testpattern.py` — 30 s pattern in six stages, each isolating one
  mapping so a wrong picture names the bug.
- `live/preview.py` — frames → PNG with no image library, same projection M3's
  browser preview will use.

**Verified.**  `./live.sh selftest` passes: no channel overlaps or gaps, red
lands on red for both RGB and GRB fixtures, `.fseq` round-trips, and 80 frames
over a real UDP socket come back byte-identical.  End to end,
`xLights --fseqcmp` reports **`IDENTICAL: 1200 frames x 37084 channels match
exactly`** between the pattern rendered offline and the same pattern captured
off the wire (30 s at 40 fps, zero packets lost, 0.7 ms/frame).

That comparison earned its keep immediately: it caught `frame * (1/fps)` vs
`frame / fps` disagreeing in the last bit of a float — enough to flip 8-bit
levels by one.  Show time is now always `frame / fps`.

`xLights --checksequence` turned out to be no use on an `.fseq` (it wants a
GUI and hangs); `--fseqcmp` replaces it and is strictly better.

### M2 · Renderer — **done**
- `live/frame.py` — `Canvas`: float RGB buffers plus the geometry effects paint
  by. All eight nets share one 465-node map, so they are one `(8, 465, 3)`
  array; all 24 arches share one base→apex→base run, so the corridor is one
  `(24, 360, 3)` array and a wave down the tunnel is an outer product.
  Effects never touch channel numbers or wire colour order —
  `to_channels()` does that once per frame through a precomputed gather.
- `live/palette.py` — the generative colour from `triseq/palettes.py` in numpy:
  hue journey by golden angle, harmony schemes, `ramp()` across a fixture, and
  both invariants (quiet-is-not-dark floor; gradients move through *hue*).
- `live/effects.py` — the eight corridor patterns ported from `arrange.py` as
  pure functions `(n, phase) → levels`, with their density table intact; net
  effects wash / bars / radial / pinwheel / plasma / sparkle; par.
- `live/script.py` — the fixed 30 s show: four four-bar scenes at 128 BPM,
  intro → verse → build → drop, a pure function of the frame index.

**Verified.**  `./live.sh bench`: **0.55 ms/frame** mean (0.31 render + 0.24
pack), p95 0.70, max 1.28 — 2 % of the 25 ms budget at 40 fps, against a 10 ms
target. Rendered offline, rendered again, and rendered through DDP into the
fake Falcon all compare `IDENTICAL: 1200 frames x 37084 channels` under
`xLights --fseqcmp`. `./live.sh selftest` grew three cases: canvas → channels
(colour order, clamping, group slices), the effect vocabulary (every pattern
finite, in range, and actually lighting something; targeting one net group
leaves the other alone), and the script (deterministic, in budget, every
channel used).

Determinism is a *requirement*, not a nicety: the random effects take a frame
index rather than a live RNG, because a capture that cannot be compared to a
re-render makes the `--fseqcmp` oracle useless.

### M3 · Web UI (preview + controls + presets) — **done**
- `live/engine.py` — the daemon: render thread, DDP out, optional `.fseq`
  recording, status. Its own thread, so the web server cannot starve it.
- `live/settings.py` — every knob in one flat dataclass; validation *and* the
  UI widgets are generated from one schema. Presets are named JSON files that
  ignore unknown keys, so they survive a build that has more or fewer knobs.
- `live/geometry.py` — **real** 3D world positions from the layout, plus an
  orbitable camera. Not the schematic the plan assumed: the preview is the rig.
- `live/web.py` + `live/static/` — aiohttp; REST for settings/presets/camera,
  one WebSocket carrying status (JSON, 5 Hz) and preview frames (raw RGB bytes,
  15 Hz, ~10 KB — the same payload as JSON is 300 KB/s of quotes and commas).
  Plain HTML/JS/canvas, no build toolchain.
- Controls: output, blackout, brightness, gamma, corridor rate, articulation,
  pattern override, hue offset, hue lock, tempo, scene hold. The state-machine
  thresholds land with M5 — the schema takes new knobs one line at a time and
  old presets keep loading.

**Verified**, in `selftest` against a real engine and a real server: blackout
reaches the wire within three frames *and* darkens the preview; brightness
changes the frame; a pattern override shows in the status; an out-of-range
slider clamps while an unknown key is refused; presets round-trip; moving the
camera bumps the geometry generation; closing the browser leaves the engine
running and reconnect works. The run is recorded to `.fseq` and the blackout is
in the file too — what the operator saw is what went to disk. Also driven by
hand in Chrome: preview, orbit, blackout, live scene/pattern readout.

The preview feed was capped at 15 Hz to begin with, straight from this plan's
"~15 fps". That reads as stutter even with the render loop keeping perfect
time — you see 15 of every 40 frames. It now runs at the engine's rate
(measured 39.7 Hz in the browser, 25.0 ms gaps, 0.6 ms to draw) and skips
identical frames, so a blackout costs one frame in three seconds instead of
120. `--preview-fps` and `--preview-detail` are separate dials; the render loop
was never the constraint (1.2 ms of a 25 ms budget, and it holds its rate to
80 fps on the Mac at 13–18 % of budget).

Two things worth carrying forward:
- **Blackout is applied to the frame, not the canvas**, so the preview goes
  dark with the rig. An operator hitting it must see the lights die, not watch
  a show that is secretly still lit underneath.
- The preview samples the **wire frame**, not the canvas, so brightness, gamma
  and blackout are all visible in it. It is the same bytes the Falcon gets.

This also corrected a claim in the offline `README.md`: the 24 arches share one
`WorldPos`, but their `PointData` does not — they sit 100 units apart along Z,
from −995 to +1305, in exactly the `Tunnel` group's order. The corridor has
real depth in the layout, so a group-level sweep would have somewhere to
travel. (The offline generator's per-arch stagger is still the better tool; the
statement of *why* is what was wrong.)

### M4 · Streaming analysis + beat clock — **done, with a caveat**
- `live/audio.py` — one interface, two sources: an ffmpeg file streamer (the
  simulation input, pace-able or as-fast-as-possible) and `sounddevice`
  line-in. Auto-gain lives here, because every threshold downstream is
  relative and must not track the DJ's gain knob.
- `live/analysis.py` — causal per-block features at 40 µs/block (150× real
  time on the Mac): band energies, spectral flux, a **kick** detector (bass
  *flux*, not bass level), RMS over a 45 s baseline.
- `live/beats.py` — `BeatBackend` with aubio as default and a metronome
  backend, so a clock failure can be told apart from a detection failure.
- `live/clock.py` — the model: tempo, phase, **next beat**, confidence,
  free-run and re-lock.
- `live/listener.py` — the chain on a thread. `live/verify.py` — ground truth.

**Verified against exact ground truth**: `test_track.wav` (synthesised 128 BPM)
**100 % of beats within 30 ms**, median 2.3 ms, drift +1.9 ms/min, no octave or
polarity errors; the same track through mp3, 97.3 %. Faults: 16 beats free-run
through a 7.5 s silence, confidence decays 0.89 → 0.50, re-lock to **11 ms**; a
128 → 140 step settles at 140.2 in about 16 s.

**Not verified on real DJ material — 15–38 % within 30 ms** against librosa on
four real tracks, mostly polarity rather than timing noise. librosa is not
truth here (on one track *it* is the one on the offbeat, by a 2.3× margin on
kick energy), so `verify.py` also reports how much kick lands on each grid as
an objective tiebreaker. By that measure we win one, tie one, lose two. This is
the open item for M6, and the reason `BeatBackend` exists: the BeatNet spike
now has a number to beat.

Three findings worth carrying:
- **aubio's BPM readout is biased** (129.8 for a track that is exactly 128.0).
  It is used only to pick the octave and to spot a real tempo change; the
  period comes from the loop's residuals plus a least-squares fit over the last
  64 beats.
- **A phase loop alone cannot follow a tempo change** — the phase term absorbs
  the error, leaving nothing to drive frequency. It reached 130.9 on a 128 →
  140 step and stalled, tracking every beat while predicting the next one
  116 ms wrong.
- **An offbeat detection is still a phase reference**, just half a period out.
  Discarding them left the model with nothing to correct against and it
  free-ran into exactly the half-beat error it was avoiding. Using them took
  the test track from 58 % to 100 % within 30 ms. Which half is the beat is a
  separate question, settled on bass *flux* where the on-beat window carries
  6–8× the offbeat window.

Installing aubio needed two build flags (its 2019 release predates FFmpeg 5 and
modern numpy); `setup.sh` records them, and notes `apt install python3-aubio`
as the easier path on the Pi.

### M5 · State machine + live arranger
- `live/state.py` — quiet/cruising/building/hot from energy vs baseline with
  hysteresis; "building" from a rising slope + high-band sweep; "hot" on the
  first strong low-band kick after a build.
- `live/arranger.py` — the offline recipes as a stepping loop: palette journey,
  corridor phrase rotation, articulation from live pulse clarity, per-state
  treatment.
- **Verify — golden test**: run live end-to-end on a track we have an offline
  sequence for; compare per-element events-per-beat, on-beat alignment and
  brightness-by-state against the offline `.xsq`; render to `.fseq` and view
  beside it in xLights. Won't match exactly (no lookahead) — the point is that
  it's in the same family and beat-tight.

### M6 · Full loop in simulation
- mp3 streamer → analysis → clock → state → arranger → renderer → DDP → fake
  Falcon → preview + `.fseq`, with `afplay` playing the same mp3 so a person can
  watch preview vs sound.
- **Verify**: audio-in → frame latency (should be ~0 on predicted beats,
  ~1 block + 1 frame on reactive accents); CPU per frame; sustained 40 fps for
  10 minutes; behaviour through a simulated track transition (crossfade two
  mp3s of different tempo).

### M7 · Deploy prep
- Config (Falcon IP, audio device, frame rate), `systemd` unit, Pi install
  script, log to file, auto-gain on input. Smoke-test the install on the Mac.
- On the Pi: measure real CPU; if tight, drop to 30 fps or thin the analysis
  hop before touching effects.

---

## Libraries reviewed (2026-08-19)

**LedFx** (github.com/ledfx/ledfx) — full audio-reactive engine: capture →
frequency-band effects → DDP/E1.31/WLED, with a web UI. **Verdict: reference,
not a base.** What it does well is the plumbing we already scoped (DDP out, web
UI, device mapping) — worth reading its DDP sender and its "virtuals" 1D-mapping
code when writing ours. What it lacks is everything that makes our approach
musical: no beat *prediction* (frequency-reactive per frame), no structural
state, no notion of the corridor as one instrument with per-arch stagger, no
palette journey. Bolting our arranger into its per-device effect model means
fighting its architecture for the parts we care most about. Also worth one
evening as a *bench comparison*: run LedFx on the sim feed and confirm our
output is meaningfully tighter — if it isn't, that's important to know early.

**BeatNet** (github.com/mjhydri/BeatNet) — CRNN + particle-filter **online beat,
downbeat and meter tracking**. **Spike run 2026-08-20, ~50 minutes. Verdict: do
not adopt.** The draw was downbeats, and it did not deliver them.

*Installation — better than this plan assumed.* madmom's **git master**
(0.17.dev0) builds and imports cleanly on Python 3.12 with numpy 2.5.2. The
compatibility wall is real only for the released 0.16.1, and the README's
`sitecustomize.py` workaround (which monkey-patches numpy process-wide) is not
needed on that path. What *does* still block: BeatNet's own particle filter
calls `np.in1d`, removed in numpy 2.0, so adoption means a fork or a numpy<2
pin; `pyaudio` and `matplotlib` are imported at module scope on the inference
path (pyaudio needs system portaudio, and can be stubbed for file input); and
its metadata pins `numba==0.54.1`, from 2021. Licence is **CC-BY-4.0** per the
repo, not CC BY-NC-SA as previously recorded here — permissive, so licensing
was never the obstacle.

*Downbeats — the whole reason to consider it — do not work.* On the synthetic
128 BPM 4/4 track and on a 4/4 house track it emits beat-in-bar values of only
{1, 2} — a 2/4 meter — with "downbeat" spacings of 2, 3, 4, 5, 6 and 7 events.
Its downbeats scatter across all four true bar positions (47 % within 30 ms of
a real downbeat, which is barely above chance). This is not misconfiguration:
all three pretrained models behave identically, `BeatNet.py:67` constructs the
online filter with `beats_per_bar=[]`, and the downbeat state space is built
from hardcoded `min/max_beats_per_bar = 2/4` that ignore the argument —
forcing `[4]` crashes it. Nor is it the numpy shim: all four `in1d` calls are
on 1-D arrays, where `isin` is exactly equivalent.

*Beats are decent but not better than what we have.* On the synthetic track:
median 8.7 ms, 89.3 % within 30 ms, and only 262 of 320 beats detected. Our
aubio + PLL on the same file: median 2.3 ms, **100 %** within 30 ms, and every
beat, because it predicts from a model rather than reporting detections. On
Eric Prydz *Opus* — the track our clock is worst on — BeatNet's phase is
genuinely better: kick alignment 4.61 against our 3.91 and librosa's 4.49.
That is the one real win, and it would still have to feed our clock, which is
exactly what `BeatBackend` is for.

*Cost.* 5.3 s to process 97.6 s of audio: **18× real time on the Mac**, against
150× for the whole aubio chain. Roughly 8× the CPU, before the 40 fps renderer,
on a machine far faster than a Pi — and that is on top of torch, librosa,
madmom-from-git and a fork.

**Consequence for M5**: build bar phase ourselves. Correlate band energy at a
4-beat period to find which position carries the pattern change, and anchor
`clock.set_downbeat()` on the first strong kick after a break — the same trick
that resolved half-beat polarity, one level up. No dependency risk, negligible
CPU, and BeatNet's failure here means no shortcut is being passed up. Revisit
it as a *beat* backend only if real-track phase stays a problem; the numbers to
beat are above.

## Later, optional
- **XR16 OSC meters** (UDP 10024) as an energy side-channel — free VU per input,
  no audio path. Needs the XR16 in Ethernet mode (its mode switch is Ethernet /
  AP / Wi-Fi-client, one at a time — check how the DJ controls it).
- **Semi-live cueing**: fingerprint the audio (Chromaprint), recognise the
  track, load its precomputed offline structure for lookahead. Big quality win
  for known tracks; real work.

## Open questions / risks
- **How are the arches and the par actually wired?** Confirmed by M1 rather
  than suspected: the Falcon owns channels 1–11 160 and the eight nets fill it
  exactly; the par (11 161) and the 24 arches (to 37 084) sit outside every
  controller in `xlights_networks.xml`. `./live.sh layout` prints this. If
  they're driven by something xLights doesn't know about, this is
  configuration; if not, it's hardware (~12 pixel ports for 24 arches; DMX for
  the par). Nothing upstream of the output stage cares — the renderer,
  arranger and `.fseq` all treat 33 models as real, and only the sender clips
  per controller.
- Live DJ mixes defeat beat trackers routinely (transitions, breakdowns, tempo
  nudges). Free-run + re-lock is mandatory, and M3's fault tests are the proof.
- Gain: the DJ's level drifts; auto-gain on the input is required, not optional.
- Latency offset RX3 → XR16 → PA vs Pi → Falcon: small and configurable;
  measure at the rig.
