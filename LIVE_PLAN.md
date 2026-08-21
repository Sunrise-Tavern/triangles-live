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

### M5 · State machine + live arranger — **done**
- `live/downbeat.py` — bar phase, which neither aubio nor (per the spike)
  BeatNet gives us live. Transient energy alone cannot find it — in
  four-on-the-floor every beat carries the same kick — so it uses two cues that
  fail in different places: bass *transient*, and **novelty** (how far the
  spectrum's shape is from the last couple of seconds, which a repeating kick
  is part of and a changing bassline is not). Measured per-beat peaks: kick
  leads the other bar positions 1.66×, novelty 1.27×, while broadband onset
  (1.07×) and the mid and high bands (~1.0×) carry nothing. From any of the
  four wrong offsets: **one shift, locked in 4 bars (7.5 s)**, then 100 %.
- `live/state.py` — quiet / cruising / building / hot. Each cue used where it
  is actually discriminative: loudness for quiet, **high-band share** for a
  build (0.57 against ≤0.33 — loudness *falls* during a build), and an
  **event** for the drop, because a verse reads 1.52 and a drop 1.67 and no
  threshold splits them.
- `live/arranger.py` — the offline recipes as a stepping loop: palette journey
  per state change, corridor phrase rotation by density, articulation from
  `clock.confidence`. Anticipation is abortable by construction — a build that
  never resolves simply relaxes.

**Verified** against the test track's known arrangement: every boundary found,
no spurious transitions, drops at **+0.5 s** and builds within 3.3 s. The intro
reads as cruising rather than quiet, which is inherent — nothing causal can
call a passage quiet before it has heard anything loud.

Three things the measurements forced:
- **Band shares must be energy-weighted.** A running mean of per-frame ratios
  is dominated by the near-silent frames between transients, where all that is
  left is a broadband noise floor, so sparse material looks like a permanent
  build.
- **A build has kicks too** — their peaks (9.5–67) completely overlap the
  drop's (18–21). What separates them is where the energy is: 0.19–0.26 bass
  in a build against 0.44–0.63 in a drop.
- **The energy fallback needs a warm baseline.** Early in a track everything is
  "average": a verse reached 1.77 while the track's actual second drop only
  reached 1.73. The build-then-kick rule needs no history and works from bar 1.

The golden test against an offline `.xsq` was not run — comparing event
timelines needs an `.xsq` reader we do not have, and the arrangement of the
synthetic track is exact ground truth for the same question. Worth revisiting
if live and offline output start disagreeing in ways the section timings miss.

### M6 · Full loop in simulation — **done**
- `./live.sh sim track.mp3 --play` — the real engine, real packets, the fake
  Falcon writing exactly what a controller would have received, and the same
  audio out of the speakers so a person can watch the preview against it.
- `./live.sh serve --audio track.mp3` — the same thing with the browser UI;
  `--audio-device N` for line-in at the rig.
- The panel now carries what M3 promised and could not yet deliver: beat and
  bar confidence, bar counter, state and the reason it changed, plus live
  knobs for the quiet threshold, build and drop sensitivity, and a latency
  offset for lining the lights up against the PA.

**Verified** over the whole 150 s test track: **6011 frames at 40.0 fps**,
156 286 packets, all 37 084 channels lit, **4.09 ms/frame**, 15 late frames,
audio lag 0.0 ms, tempo locked at 127.93.

And the ten-minute soak this milestone asks for, through four restarts of the
audio: **24 819 frames in 620.4 s at 40.00 fps, 2.65 ms/frame, 11 late frames,
0 skipped**, tempo still locked at 128.11 with confidence 1.00 at the end.

Still open from this milestone's list: behaviour through a real crossfade
between two tracks of different tempo. The tempo-step fault test covers the
mechanism (128 → 140 settles in ~16 s) but not two tracks overlapping.

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
downbeat and meter tracking**. **Spike run 2026-08-20, all four modes. Verdict:
do not adopt live; useful offline.**

*Installation — better than this plan assumed.* madmom **git master**
(0.17.dev0) builds and imports cleanly on Python 3.12 with numpy 2.5.2. The
compatibility wall is real only for the released 0.16.1, and the README's
`sitecustomize.py` workaround (which monkey-patches numpy process-wide) is not
needed on that path. Still blocking: BeatNet's own particle filter calls
`np.in1d`, removed in numpy 2.0, so adoption means a fork or a numpy<2 pin;
`pyaudio` and `matplotlib` are imported at module scope on the inference path
(pyaudio needs system portaudio, and can be stubbed for file input); metadata
pins `numba==0.54.1`, from 2021. Licence is **CC-BY-4.0** per the repo, not
CC BY-NC-SA as previously recorded here — permissive, so licensing was never
the obstacle.

*The four modes are not variations on one algorithm.* `stream` is microphone;
`realtime` reads a file chunk by chunk; `online` reads the **whole** file into
the CRNN and only the *decoding* is causal; `offline` is the whole file plus
madmom's Viterbi DBN. Only `stream` and `realtime` are the deployment shape.

Measured against the synthesised 128 BPM 4/4 track (exact ground truth), and
Eric Prydz *Opus*:

| mode | beats | within 30 ms | meter | downbeats on the true bar line |
|---|---|---|---|---|
| `online` / PF | 262/320 | 89.3 % | 2/4 | 63.5 % |
| `realtime` / PF | 264/320 | **55.3 %** | 2/4 | 64.0 % |
| `offline` / DBN | 323/320 | 90.4 % | 4/4 | 67.9 % |
| `offline` / DBN forced `[4]` | 323/320 | 90.4 % | 4/4 | 67.9 % |
| **ours (aubio + PLL)** | 322/320 | **100 %** | — | no downbeat yet |

On *Opus*, downbeat spacing in events — a clean tracker gives all 4s:
`online` `{2:15, 3:10, 4:12, 5:2, 6:1, 7:1}`, `realtime` `{2:14, 3:7, 4:17, …}`,
`offline` with `[2,3,4]` `{2:99}` (consistent but wrong meter), `offline` forced
to `[4]` **`{4:51}`** — perfectly clean 4/4 at exactly 125.0 BPM, with the best
kick alignment of anything measured (4.57 against our 3.91 and librosa's 4.49).

*So the conclusion is narrower than "downbeats don't work".* BeatNet's working
downbeats live in the mode we cannot use, and the modes we can use do not
produce them:

- The causal PF path's **meter is broken**, not merely untuned: `BeatNet.py:67`
  builds it with `beats_per_bar=[]`, the downbeat state space is constructed
  from hardcoded `min/max_beats_per_bar = 2/4` that ignore the argument, and
  forcing `[4]` crashes. All three pretrained models behave identically. Not
  the numpy shim either — all four `in1d` calls are on 1-D arrays where `isin`
  is exactly equivalent.
- `realtime`, the mode that matches deployment, is the **worst** on beats
  (55.3 %), well below our aubio + PLL.
- Even at its best, downbeat *phase* is ~68 % right on the synthetic track, with
  most of the remainder landing on the half-bar. (That track may have genuinely
  weak bar cues, so treat this as a floor rather than a verdict on the model.)

*Cost.* 18× real time on the Mac against 150× for our whole aubio chain —
roughly 8× the CPU before the 40 fps renderer, on a machine far faster than a
Pi, plus torch, librosa, madmom-from-git and a fork.

**The one genuinely open door**: the CRNN activations are shared across modes —
it is the *decoder* that differs. Taking BeatNet's CRNN chunk by chunk and
decoding it with our own causal logic (the existing PLL for beats, a bar
tracker for downbeats) is the only configuration that could beat what we have.
It still costs torch on the Pi. Park it behind M5.

**Concretely useful now**: `offline` / DBN with `beats_per_bar=[4]` is an
excellent *precomputation* tool — exact tempo, clean bars. That is precisely
what the "semi-live cueing" idea below needs, so if we ever fingerprint tracks
and load precomputed structure, this is the tool to build it with.

**Consequence for M5**: build bar phase ourselves. Correlate band energy at a
4-beat period to find which position carries the pattern change, and anchor
`clock.set_downbeat()` on the first strong kick after a break — the same trick
that resolved half-beat polarity, one level up. No dependency risk, negligible
CPU, and nothing causal in BeatNet beats it.

## Validation corpus — **open**

`live/corpus.py` (`./live.sh corpus tracks/`) runs a folder of music through
the whole live chain and puts one row per track on the table, flagging the bad
ones. Most of its metrics need no oracle — kick alignment on our grid versus
our offbeat, share of the track locked, bar shifts, state transitions per
minute, speed — because the offline generator is better-informed but is not
ground truth, as the librosa episode in M4 established.

**We have four real tracks, all 90–110 s excerpts, and three synthetic files.**
That is not enough, and every threshold in the project was chosen against
essentially one of them.

Needed, from our own music (nothing is downloaded):
- **full tracks, not excerpts** — the loudness baseline is 45 s and the peak
  reference releases over 120 s, so a 97 s excerpt never fills either;
- **at least one long DJ mix** — transitions are what defeat beat trackers, it
  is literally what the rig will hear, and nothing here has ever been tested
  across a crossfade;
- spread across four-on-the-floor, half-time/DnB, hip-hop, backbeat pop,
  ambient, something with a real tempo change, something live, and something
  not in 4/4 (`bar_length` is hardcoded).

**Ground truth now exists.** The Harmonix Set (912 tracks; beats, downbeats
and functional segments; MIT-licensed annotations) is wired in via
`live/harmonix.py`. `--score` grades any of those 912 you own; `--synth`
renders an annotation *as audio* and grades against it, which tests 912 real
tempo curves, bar layouts and arrangements without owning a note — the timbres
are ours, so it says nothing about aubio on a dense real mix, but our previous
structural sample size was one.

Measured on dance/electronic at 115–145 BPM, which is what this rig plays:
**beat precision 89 %, recall 91 %, downbeats within 50 ms 70 %, tempo error
0.1 %, clock locked 96 %**, best tracks 96–99 %. Across all genres precision
falls to 54 %, and the failures are three specific limits rather than noise:

- **non-4/4 is unusable** (16–29 %) — `bar_length` is hardcoded to 4;
- **sub-80 BPM material is octave-doubled** by the 70–180 folding range
  (66→133, 78→156, 82→164, 87→173);
- **tempo wanders mid-track** on a minority of tracks even when the final value
  is right — "Give It Up" ends at 139.9 against 140 but spends part of the
  track near 92, which is 140 x 2/3.

Precision and recall are reported separately because a single "within 30 ms"
number hides every octave error: recall stays high at double tempo.

Already flagged by the thin corpus of real excerpts, still open:
- **Tempo wanders on real material** in a way it never does on the synthetic
  track: Opus runs 100.7 / 125.2 / 143.0 at p5/p50/p95 and spends about twenty
  seconds around 105 BPM before recovering, against 127.8–128.3 for the whole
  synthetic track. Suspect is M4's coarse tempo term following aubio's BPM
  readout, which never wanders on synthetic input so never fires there.
- **Bar confidence is near zero on all four real tracks**, against 0.13–0.18
  synthetic — the bar line is being found, but not confidently.

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
