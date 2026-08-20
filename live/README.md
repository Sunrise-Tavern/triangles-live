# Live engine

Real-time lighting for the Triangles rig: audio in, DDP out, no pre-rendered
sequence.  The plan and its milestones live in [`../LIVE_PLAN.md`](../LIVE_PLAN.md).
This file is the operator's guide to what exists **now**.

Status: **M1–M4 complete** — DDP out, fake Falcon, `.fseq`, the renderer and its
effect vocabulary, the browser UI, and streaming analysis with a predictive
beat clock.  The clock is not yet wired into the show (that is M6).
M5–M7 pending.

## Quick start

```bash
./live.sh serve            # the daemon + browser UI -> http://localhost:8080
./live.sh selftest         # everything below, checked, no hardware or xLights
./live.sh layout           # the channel map, and what is unaddressable
./live.sh show             # render the fixed 30 s show -> out/show.fseq
./live.sh preview out/show.fseq --sheet 12 --out out/sheet.png
./live.sh bench            # render cost against the frame budget
```

At the rig, that first line becomes:

```bash
./live.sh serve --ddp 192.168.50.20 --controller Falcon_F16V5_0E1C
```

`live.sh` is the same wrapper idea as `run.sh` — it just runs `python -m live`
inside `.venv`.  Nothing here needs anything the offline generator did not
already install; numpy is the only dependency.

## What M1 gives you

```
test pattern ──► DDPSender ──UDP──► FakeFalcon ──► .fseq ──► xLights / preview PNG
```

| Module | Job |
|---|---|
| `layout.py` | xLights models → absolute channel ranges, colour order, node maps |
| `ddp.py` | DDP sender (packet split, push flag, sequence numbers) and reassembler |
| `fseq.py` | FSEQ v2.0 uncompressed reader/writer, streaming |
| `fake_falcon.py` | UDP → frames → `.fseq`, with a "what actually arrived" report |
| `timing.py` | drift-free frame clock, reports lateness instead of hiding it |
| `testpattern.py` | a 30 s pattern whose failures name the mapping that is wrong |
| `preview.py` | frames → PNG, hand-rolled, so a Pi needs no image library |

## What M2 adds

```
Script ──► effects ──► Canvas ──► to_channels ──► (M1's pipeline)
```

| Module | Job |
|---|---|
| `frame.py` | `Canvas`: float RGB buffers + the geometry effects paint by |
| `palette.py` | generative colour in numpy — hue journey, harmony schemes, ramps |
| `effects.py` | 8 corridor patterns + 8 net/par effects |
| `script.py` | a fixed 30 s show at 128 BPM: intro → verse → build → drop |

Effects work in **float RGB 0..1, in source order**, and never touch channel
numbers or wire colour order — `Canvas.to_channels()` does that once per frame
through a precomputed gather.  An effect cannot address the wrong fixture
because it cannot address fixtures at all.

Two facts about this rig make it vectorise: all eight nets share one 465-node
map over a 59×51 triangular lattice, so they are one `(8, 465, 3)` array; all
24 arches share one 360-node base→apex→base run, so the corridor is one
`(24, 360, 3)` array and a wave down the tunnel is an outer product.

### Corridor patterns

The offline arranger schedules xLights effects with start times and fades.  A
real-time renderer is asked "what does this look like *now*" forty times a
second, so the same eight ideas become pure functions
`(n_arches, phase) -> levels`:

`sparkle` (0.20) · `comet` (0.30) · `converge` / `diverge` (0.42) ·
`bounce` (0.55) · `pairs` (0.72) · `alternate` (0.90) · `strobe` (1.00)

The number is the pattern's density, ported unchanged from the offline show
where it was tuned against real tracks; `effects.vocabulary(target)` picks the
three nearest a section's energy, and refuses to strobe a quiet passage.

The rule that carried over intact: **lit arches must overlap.**  A comet whose
tail is shorter than the gap between arches reads as 24 things blinking in
sequence, not one band travelling through a tunnel.  Here that falloff is just
the shape of the level curve rather than an effect's fade-out.

### Net effects

`wash` · `bars` · `radial` · `pinwheel` · `plasma` · `net_sparkle`, plus `par`
for the single RGBW DMX pixel.  All take a `targets` slice so an effect can
address the big nets or the small ones — a slice, deliberately, because a fancy
index would hand back a copy and the paint would vanish silently.

## What M3 adds

```
Engine (render thread) ──► Settings ◄── REST ──┐
   │                                            ├── browser
   └── frame ──► WebSocket (binary RGB, 15 Hz) ─┘
```

| Module | Job |
|---|---|
| `engine.py` | the daemon: render thread, DDP out, optional fseq recording, status |
| `settings.py` | every knob in one flat dataclass; the UI is generated from it |
| `geometry.py` | real 3D world positions + a camera, for a preview that is the rig |
| `web.py` | aiohttp: static UI, REST for settings/presets/camera, one WebSocket |
| `static/` | one HTML file, one JS file, one stylesheet — no build step |

The engine runs in its own thread so the web server cannot starve it: a late
frame at the rig is a visible glitch, a late HTTP response is not.  Settings
cross the thread boundary without a lock, which is safe because every field is
one `bool`/`float`/`str` and CPython attribute assignment is atomic — the worst
case is a frame that sees brightness from before a change and hue from after,
which is invisible.  A whole-preset load is the one multi-field change and it
takes the lock.

Frames go the other way by rebinding one attribute, with the engine always
filling the buffer it did *not* publish, so a reader can never see half of two
different frames.

**Blackout is applied to the frame, not the canvas**, so the preview goes dark
with the rig.  An operator hitting it must see the lights die, not watch a show
that is secretly still lit underneath.

### The preview is the rig, not a diagram

`geometry.py` reads the real world coordinates out of `xlights_rgbeffects.xml`
and projects them through a perspective camera you can orbit by dragging.  This
corrected something the offline `README.md` still says: the 24 arches share one
`WorldPos`, but their `PointData` does not — they sit **100 units apart along Z,
from −995 to +1305**, in exactly the order the `Tunnel` group lists them.  The
corridor has real depth in the layout.

The preview samples the **wire frame**, not the canvas, so what you watch is
what the Falcon is being sent — master brightness, gamma and blackout included.
It is downsampled (every 2nd net node, every 6th arch node → ~3 300 dots) and
sent as raw RGB bytes **at the engine's own frame rate**: ~10 KB a frame,
390 KB/s at 40 fps, where the same thing as JSON would be several times that in
quotes and commas.  Identical frames are not resent, so a held look or a
blackout drops to nothing (measured: 1 frame in 3 s during blackout, vs 120).

### If the preview looks steppy

It was capped at 15 Hz first, which reads as stutter even though the engine is
keeping perfect time — you were seeing 15 of every 40 frames.  Measured before
and after, with the render loop unchanged in both:

| | render loop | preview feed | browser draw |
|---|---|---|---|
| before | 40.10 fps, 1 late in 320 | 15.0 Hz, 66.6 ms gaps | — |
| after | 40.10 fps, 2 late in 320 | 39.7 Hz, 25.0 ms gaps | 0.6 ms/frame |

Two dials, and they are separate things:

* `--preview-fps N` caps the feed (default: the engine's rate).  Turn it down
  for a phone on bad wifi.
* `--preview-detail 2.0` doubles the pixel density and the bandwidth with it.
  Temporal smoothness is not what this changes.

If the *whole show* should be smoother, raise the engine instead: `--fps 60`.
Measured on the Mac, the loop holds its rate with one late frame per six
seconds all the way to 80 fps, at 13–18 % of the frame budget.  40 stays the
default because the Pi has not been measured yet — that is M7's job.

The browser draws only when a frame actually arrives, not on every animation
frame, and caches colour strings — the one real allocation in a loop that would
otherwise build 130 000 of them a second.

### Controls

Output enable · blackout · master brightness · gamma · corridor rate ·
articulation · corridor pattern override · hue offset · hue lock · tempo ·
scene hold.  Presets are named JSON files under `live/presets/`, saved and
loaded from the panel.

Adding a knob is one line in `settings.py` — the schema drives validation *and*
the widget, so there is no parallel list to forget.  Preset loading ignores
unknown keys on purpose: a preset saved before a knob existed must still load,
and one saved after must not break an older build at the rig.

Articulation at 0.5 reproduces the scripted show exactly; turning it up or down
slides the corridor-pattern choice along the same density table the offline
arranger uses.

## What M4 adds

```
line-in / mp3 ──► Analyzer ──► BeatBackend ──► BeatClock ──► (phase, next beat)
```

| Module | Job |
|---|---|
| `audio.py` | one interface, two sources: ffmpeg file streamer and line-in |
| `analysis.py` | causal per-block features: bands, flux, kick, energy vs baseline |
| `beats.py` | `BeatBackend` — aubio, plus a metronome for testing the clock alone |
| `clock.py` | tempo, phase, **the next beat**, confidence, free-run and re-lock |
| `listener.py` | the chain on its own thread, publishing state |
| `verify.py` | ground-truth measurement and fault injection |

```bash
./live.sh listen track.mp3          # watch features and the clock, live
./live.sh listen --devices          # find the USB interface at the rig
./live.sh beats .cache/test_track.wav --bpm 128
./live.sh beats --faults            # silent break and tempo change
```

### The clock predicts; it does not report

By the time audio is captured, analysed, rendered and pushed over DDP, a light
fired *on detection* lands visibly behind the room.  So the clock keeps a model
— a tempo and an anchor — and the renderer asks "where are we now" rather than
being told "a beat just happened".  Three consequences:

* **Detection latency stops mattering for firing.**  A beat noticed 40 ms late
  refines the same model, and the model's next prediction is on time.  The
  tracker has to be accurate, not prompt.
* **Losing the tracker is survivable.**  Silence means "keep going at the last
  known tempo", not "stop".  Free-run and re-lock are the normal case in a DJ
  set, not the error case.
* **Confidence is an output.**  M5's arranger can lean on the grid when it is
  trustworthy and fall back to energy when it is not.

### Three things the measurements forced

**aubio's BPM readout is biased.**  It reports 129.8 for a track that is
exactly 128.0 — 1.4 %, about 6 ms on every prediction.  So its readout is used
only to pick the octave and to spot a genuine tempo change; the period itself
comes from the loop's own residuals plus a least-squares fit over the last 64
beats.  Fitting a line through every beat rather than differencing the two ends
is roughly five times quieter.

**A phase loop alone cannot follow a tempo change.**  The phase term absorbs
the error each beat, leaving almost nothing to drive the frequency term.
Measured on a 128 → 140 step it reached 130.9 and stalled — tracking every beat
while predicting the next one 116 ms wrong.  A coarse term, gated on the
tracker disagreeing by more than bias could explain and *keeping* it up, fixes
that: 140.2 within about 16 seconds.

**Half-beat slips are the failure that matters.**  The grid stays plausible,
the tempo stays right, and every effect fires on the wrong half.  On the test
track aubio itself jumped to the offbeat at t≈90 s.  Two mechanisms handle it:

* An offbeat detection is *not* evidence of a tempo change, so it never counts
  toward a re-lock — but it **is** a phase reference half a period out, so it
  is used as one.  Discarding those outright left the model with nothing to
  correct against, and it free-ran into exactly the half-beat error it was
  avoiding: measured, 45 seconds to get there.  Using them took the test track
  from 58 % to 100 % of beats within 30 ms.
* Which half is the beat is settled on the **low end**, where the kick is —
  not on level (a bassline plays continuously, so a kick barely moves it) but
  on transient flux in the bass band, where the on-beat window carries 6–8×
  the offbeat window.

### Where it stands, measured

Against exact ground truth — a synthesised 128 BPM grid, and the same track
through mp3:

| | within 30 ms | median | offbeat | drift |
|---|---|---|---|---|
| `test_track.wav` | **100 %** | 2.3 ms | 0 % | +1.9 ms/min |
| `smoke_test.mp3` | **97.3 %** | 17.0 ms | 0 % | −2.9 ms/min |

Faults: 16 beats free-run through a 7.5 s silence (expected ~16), confidence
decays 0.89 → 0.50, re-locks to **11 ms**.  A 128 → 140 step settles at 140.2.

Against librosa on four real tracks, it is **mixed — 15–38 % within 30 ms**,
with most of the disagreement being polarity rather than timing noise.  But
librosa is not truth here, which is why `verify.py` also reports how much kick
lands on each grid:

| track | kick @ ours | kick @ librosa | kick @ our offbeat |
|---|---|---|---|
| Children | **4.04** | 1.75 | 2.06 |
| Midnight Vampires | 5.08 | 5.32 | 2.34 |
| Opus | 3.91 | **4.49** | 3.51 |
| Tove Styrke | 2.45 | 2.63 | **2.93** |

So on Children librosa is the one on the offbeat and its 81 % "error" is
meaningless; on Midnight Vampires the two agree; on Opus and Tove Styrke we are
genuinely worse.  Tove Styrke is the weakest case — the clock settles at
124.6 BPM against a 127.8 reference and reports confidence 0.61, which is at
least honest about it.

**Real DJ material is not solved.**  That is the open item going into M6, and
the reason `BeatBackend` exists: the plan's BeatNet spike has something to
prove against these same numbers.

### Colour

Ported from `triseq/palettes.py` into numpy: a base hue from the music, a
golden-angle rotation per section, and a harmony scheme whose contrast rises
with energy (`analogous` → `split` → `complementary` → `triadic`).  Both colour
invariants survive the port:

* **Quiet is not dark.**  `Palette.floored()` lifts a dim palette until
  something in it clears ~48/255, keeping hue and internal balance.
* **A gradient moves through hue, not just brightness.**  The corridor's far
  palette is a *rotation* of the near one, so the tunnel runs cyan at the mouth
  to violet at the back.

### The three things `layout.py` gets right so nothing else has to

* **Controller-relative start channels.** The nets are addressed
  `!Falcon_F16V5_0E1C:1`, not `1`; controller offsets come from
  `xlights_networks.xml`.
* **String colour order.** Nets are `RGB Nodes`, arches are `GRB Nodes`.
  `Model.pack()` is the single place that permutation happens.
* **Custom-model node maps.** Each net is 465 nodes scattered over a 59×51
  grid; the map comes from `CustomModelCompressed`.

## Verifying on the Mac

**1 · Automated, no dependencies.** `./live.sh selftest` checks that the
channel map has no overlaps or gaps, that a red frame lands on the red channel
of both an RGB net and a GRB arch, that an `.fseq` reads back identical, and
that 80 frames pushed through a real UDP socket into the fake Falcon come out
of the `.fseq` byte-for-byte unchanged.

**2 · Against xLights, byte-exact.** Render the pattern offline, capture the
same pattern over the wire, and let xLights compare them channel-for-channel:

```bash
./live.sh render --out out/pattern.fseq         # no network
./live.sh demo   --out out/capture.fseq         # through DDP
/Applications/xLights.app/Contents/MacOS/xLights \
    --fseqcmp "$PWD/out/pattern.fseq" "$PWD/out/capture.fseq"
# -> IDENTICAL: 1200 frames x 37084 channels match exactly
```

`--fseqcmp` is the sharpest tool in the box: it reports the first differing
frame and channel, groups differences by model, and probes for a frame offset.
It found the one real bug in M1 — `frame * (1/fps)` and `frame / fps` disagree
in the last bit of a float, which was enough to flip 8-bit levels by one.

`--checksequence` is *not* useful on an `.fseq`; it wants a GUI and hangs.

**3 · By eye.** `./live.sh preview` draws the rig flat — corridor as nested
triangles receding, nets as their real 59×51 grids, par as a swatch:

```bash
./live.sh preview out/capture.fseq --sheet 12 --out out/sheet.png   # whole run
./live.sh preview out/capture.fseq --at 14.5 --out out/frame.png    # one moment
```

For full fidelity, open the `.fseq` in xLights against the real layout.  That
stays the visual oracle; the PNG is the fast loop.

**4 · The renderer, three ways.**  The fixed show is a pure function of the
frame index, so all three of these must agree byte-for-byte:

```bash
./live.sh show --out out/show.fseq            # rendered offline
./live.sh show --out out/show2.fseq           # rendered again
./live.sh fake-falcon --out out/capture.fseq --bind 127.0.0.1 &
./live.sh show --send --host 127.0.0.1        # rendered and pushed over DDP
xLights --fseqcmp out/show.fseq out/show2.fseq      # IDENTICAL
xLights --fseqcmp out/show.fseq out/capture.fseq    # IDENTICAL
```

`./live.sh bench` reports the cost split between rendering and packing:

```
600 frames, 37084 channels, budget 25.0 ms/frame at 40 fps
  render   mean   0.31  p95   0.46  max   1.05 ms
  pack     mean   0.24  p95   0.29  max   0.36 ms
  total    mean   0.55  p95   0.70  max   1.28 ms (2% of budget)
```

That is on the Mac.  The Pi number is the one that decides whether 40 fps
survives contact with the DSP — run the same command there.

**5 · The daemon and its panel.**  `selftest`'s last case starts a real engine
and a real server on an ephemeral port and drives it over HTTP: blackout must
reach the wire within three frames *and* darken the preview; master brightness
must change the frame; a pattern override must show in the status; an
out-of-range slider must clamp while an unknown key is refused; a preset must
round-trip; moving the camera must bump the geometry generation; and closing
the browser must leave the engine running, with a reconnect that works.  It
also records the run to an `.fseq` and checks the blackout is there too — what
the operator saw is what went to disk.

### Reading the test pattern

Each stage isolates one mapping, so a wrong picture names the bug:

| Stage | Correct | A failure means |
|---|---|---|
| identify | models light one at a time | start channel wrong |
| primaries | whole rig goes red, green, blue | string colour order (arches are GRB) |
| corridor | one band travels front → back | `Tunnel` group is not physical order |
| arch | each arch fills base → apex → base | polyline segment split wrong |
| nets | a bar sweeps left → right on each net | custom-model node map wrong |
| par | the par cycles R, G, B, W | DMX slot map wrong |

## Verifying at the rig

On the Pi (or the Mac, on the show LAN), aim the pattern at the Falcon and
watch the nets:

```bash
./live.sh pattern --host 192.168.50.20 --controller Falcon_F16V5_0E1C --loop
```

`--controller` clips the send to that device's channel space (1–11 160) and
sends at offset 0, which is what the Falcon expects.  Without it the sender
pushes all 37 084 channels, which is right for the fake Falcon and wrong for a
real one.

To watch what a real controller is being sent, run the fake Falcon on a second
machine and point the sender at it — same code path, and it writes an `.fseq`
you can open next to the real thing.

## Known: the arches and the par are not addressable yet

`./live.sh layout` ends with:

```
!! 25 model(s) sit outside every controller's channel space and cannot be
   driven over DDP:  ADJ Mega Par Profile + (ch 11161) ... Poly Line-9 (ch 37084)
```

This is the open question at the bottom of `LIVE_PLAN.md`, now measured rather
than suspected: the Falcon owns channels 1–11 160, which the eight nets fill
exactly.  The par starts at 11 161 and the 24 arches run to 37 084, and no
controller in `xlights_networks.xml` covers them.

Everything upstream of the wire — renderer, arranger, preview, `.fseq` — treats
all 33 models as real, so nothing needs rewriting when they get a controller.
Only the output stage cares, and it already clips per controller.

## FSEQ v2.0, briefly

`fseq.py` writes uncompressed v2.0: a 32-byte header, optional variable headers
(`mf` names the audio file so xLights loads a waveform), then one flat array of
channel bytes per frame.  Compression is deliberately off — a Pi rendering live
has no cycles for zstd, and these captures are scratch.  The header layout is
documented in the module docstring, decoded from files this show folder already
contains and cross-checked against xLights' own `FSEQFile.cpp`.
