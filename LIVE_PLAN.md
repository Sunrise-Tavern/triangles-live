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

### M1 · DDP out + fake Falcon (the loop's two ends)
- `live/ddp.py` — DDP sender: 10-byte header (flags, seq, type, dest, 32-bit
  channel offset, 16-bit length), 1440 ch/packet (matches the Falcon config),
  push flag on a frame's last packet, fixed 40 fps clock.
- `live/fake_falcon.py` — UDP receiver reassembling frames; maps channels →
  models via `show.py` + start channels; **writes `.fseq`** (v1 uncompressed;
  documented in the show's own readme) and a minimal live preview.
- **Verify**: synthetic pattern round-trips exactly; the `.fseq` passes
  `xLights --checksequence` and looks right in xLights' 3D preview against the
  real layout. xLights is the visual oracle from here on.

### M2 · Renderer
- `live/frame.py` — per-model buffers from the layout's node maps (nets 465
  nodes into 59×51; arches 360-node strips base→apex→base; par RGBW).
- Net effects: wash, bars, radial pulse, sparkle, plasma-ish noise. Corridor:
  port the pattern library from `arrange.py` (already per-arch on/off +
  gradient — a real-time renderer's native form).
- **Verify**: fixed 30 s script → `.fseq` → xLights; ≤ 10 ms/frame on the Mac.

### M3 · Web UI (preview + controls + presets)
- `live/web.py` — small async server in the daemon process (aiohttp or FastAPI):
  static single-page UI (plain HTML/JS/canvas, **no build toolchain**), REST for
  settings/presets, WebSocket for state + preview frames.
- Preview: per-model RGB downsampled to ~15 fps, drawn on a canvas using
  geometry exported once from `show.py` (nets as 59×51 grids, arches as a
  corridor of strips front-to-back, par as a swatch). This becomes the primary
  dev preview; `.fseq` → xLights stays the high-fidelity oracle.
- Controls (live, no restart): output enable/disable, **blackout**, master
  brightness, corridor rate, articulation density, palette hue offset/lock,
  state-machine thresholds; current state readout (tempo, confidence,
  quiet/cruising/building/hot).
- Presets: named JSON files of every knob; load/save/delete from the UI.
- **Verify**: change each knob while a sim run is playing and see it take effect
  within a frame or two; kill/reload the browser mid-run (daemon unaffected);
  preview matches what the fake Falcon writes to `.fseq`.

### M4 · Streaming analysis + beat clock
- `live/audio.py` — capture abstraction with two backends: **file streamer**
  (reads an mp3 and yields blocks in real time — the simulation input) and
  `sounddevice` line-in. Same interface, so the rest never knows.
- `live/analysis.py` — aubio onset + tempo per block; band energies; rolling
  RMS with a slow (30–60 s) baseline.
- `live/clock.py` — tempo/phase estimate, **next-beat prediction**, confidence;
  free-runs on last tempo when the tracker loses lock, decays, re-locks.
- Beat detection sits behind a small `BeatBackend` interface (emit beat events +
  tempo estimate) so aubio can later be swapped/AB-tested against BeatNet or
  BTrack without touching the clock.
- **Verify — against ground truth**: stream the mp3s we already have through
  it and compare predicted beats to the offline librosa grid for the same file
  (median error, % within 30 ms, drift over 90 s). Stream the synthetic
  `test_track.wav` (known 128 BPM) for an exact reference. Inject a 4-bar
  silence and a tempo nudge; confirm free-run and re-lock.

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
downbeat and meter tracking**. **Verdict: attractive but risky; optional second
backend, timeboxed spike, not on the critical path.** The draw is real:
downbeats (bar phase) live, which aubio does not give — bar-level moves
currently need phase inference. The risks: depends on **madmom**, which is
unmaintained and fights modern numpy/Python (we already hit this in the offline
tooling); PyTorch inference per hop is heavy for a Pi (fine on the Mac —
"plausible, measure" on a Pi 5); **CC BY-NC-SA license** (fine for this art
project, worth knowing); and particle filters add latency jitter that must be
measured against the ≤1-frame budget. Spike plan: pinned separate venv, run its
streaming mode over our test tracks, compare beat/downbeat accuracy and CPU vs
aubio on the same audio; adopt only if downbeat accuracy is high and Pi CPU
allows. If madmom won't install cleanly in under an hour, stop — aubio +
inferred downbeats is acceptable.

## Later, optional
- **XR16 OSC meters** (UDP 10024) as an energy side-channel — free VU per input,
  no audio path. Needs the XR16 in Ethernet mode (its mode switch is Ethernet /
  AP / Wi-Fi-client, one at a time — check how the DJ controls it).
- **Semi-live cueing**: fingerprint the audio (Chromaprint), recognise the
  track, load its precomputed offline structure for lookahead. Big quality win
  for known tracks; real work.

## Open questions / risks
- **How are the arches and the par actually wired?** In the layout they have no
  controller (start channels 11 161+, past the Falcon's 11 160). If they're
  driven by something xLights doesn't know about, this is configuration; if
  not, it's hardware (~12 pixel ports for 24 arches; DMX for the par).
- Live DJ mixes defeat beat trackers routinely (transitions, breakdowns, tempo
  nudges). Free-run + re-lock is mandatory, and M3's fault tests are the proof.
- Gain: the DJ's level drifts; auto-gain on the input is required, not optional.
- Latency offset RX3 → XR16 → PA vs Pi → Falcon: small and configurable;
  measure at the rig.
