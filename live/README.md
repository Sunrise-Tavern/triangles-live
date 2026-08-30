# Live engine

Real-time lighting for the Triangles rig: audio in, DDP out, no pre-rendered
sequence.  The plan and its milestones live in [`../LIVE_PLAN.md`](../LIVE_PLAN.md).
This file is the operator's guide to what exists **now**.

Status: **M1–M7 complete.**  Audio in, analysed, tracked, arranged, rendered,
and out over DDP, with a browser preview, one config file, a preflight check
and a systemd unit.  What remains is measuring it on the actual Pi.

## Quick start

```bash
./live.sh serve            # the daemon + browser UI -> http://localhost:8080
./live.sh orient           # at the rig: is up up, is front front?  (see below)
./live.sh selftest         # everything below, checked, no hardware or xLights
./live.sh layout           # the channel map, and what is unaddressable
./live.sh show             # render the fixed 30 s show -> out/show.fseq
./live.sh preview out/show.fseq --sheet 12 --out out/sheet.png
./live.sh bench            # render cost against the frame budget
```

At the rig, that first line becomes:

```bash
./live.sh serve --ddp auto                    # both Falcons, from the layout
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

Two facts about this rig make it vectorise: the nets are all triangles on a
lattice — 465 nodes over 59×51 or 435 over 57×49 — so they are one
`(n, 465, 3)` array padded to the widest, with per-net geometry vectors of the
same shape (padding slots are painted like any other and never emitted); all
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
scene hold · colour scheme override · shuffle seed · transitions · resting
nets.  Presets are named JSON files under `live/presets/`, saved and loaded
from the panel.

*Pattern hold* is how many phrases the corridor pattern and net gesture are
kept for — the colour still moves every phrase, the material only every
`pattern_hold` of them.  *Shuffle seed* reseeds every deterministic choice the show makes — which
pattern, gesture, scheme and transition each phrase gets — so two rigs, or two
nights, need not draw the same show from the same music; the same seed always
does.  *Transitions* is how much of the material arrives with a fade, wipe or
dip rather than a cut (0 is always a cut).

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

## What M5 adds — bar phase, state, and the live arranger

| Module | Job |
|---|---|
| `downbeat.py` | which beat starts the bar |
| `state.py` | quiet / cruising / building / hot |
| `arranger.py` | state + clock → pixels, the offline recipes as a stepping loop |

### Finding the bar line

The clock counts beats consistently but has no idea where a bar begins, and
phrase-level behaviour — "this build resolves in two bars" — is worthless
counted from the wrong place.  aubio gives no downbeats, and the BeatNet spike
found the modes we could run live don't either.

**Transient energy alone cannot do it**: in four-on-the-floor every beat has
the same kick, so "which beat is loudest" is noise.  Two cues that do carry
bar information, failing in different places, so both are used:

* `kick` — bass-band *transient*.  Works when the bar line is reinforced.
  Useless when every kick is identical.
* `novelty` — how much the *shape* of the spectrum differs from the last
  couple of seconds.  A kick that repeats every beat is part of that average
  and doesn't register; a bassline changing note does.  This is the cue that
  survives a machine-perfect drum pattern.

Measured, taking each beat's peak: `kick` leads the other three bar positions
by **1.66×** and `novelty` by **1.27×**, both pointing at the true bar line,
while broadband onset (1.07×) and the mid and high bands (~1.0×) say nothing.

From any of the four wrong starting offsets it takes **exactly one shift and
locks in 4 bars (7.5 s)**, then 100 % of bar lines are right.

Caveat worth knowing: it keys on bass and broadband cues, so a bar line marked
only in the mid-range would be missed.

### Silence is its own state

`silent` is separate from `quiet` on purpose: a breakdown is part of a track
and still has a beat grid to be driven by, while silence has none — following
a free-running clock there makes the rig twitch at an imaginary tempo when the
room has gone home.

Detecting it needed the one absolute number in the system.  **Every loudness
measure here is relative by design** so the show tracks the music and not the
DJ's gain knob — and a purely relative measure cannot detect silence, because
it normalises by whatever it is hearing.  Measured on a dead feed followed by
music: `energy` read 1.00 then 1.56, `level` 1.18 then 1.00.  Neither separates
them, while the broadband noise floor made `high_share` 0.80 — so silence read
as a permanent *build*, and once music started the poisoned baseline held the
machine in `hot` for over a minute.

So there is a floor — and it is **configurable**, because "silence" depends
on the input by about 40 dB.  A line feed with nothing on it sits near
−90 dBFS; a microphone in a room sits near −51.  Three attempts to infer it
all failed on one side or the other: a fixed −70 dBFS called room tone
"music"; a noise-floor gate called a quiet intro "silence", because the floor
initialised to it; peak-to-median "peakiness" put room tone at 1.09 and a real
breakdown at 1.16, too thin to split.

`./live.sh doctor`, run with nothing playing, measures your input and tells you
what to set.  Two details still matter:

* it applies to the **3-second smoothed** level, not per-block RMS — the median
  block of a click track is digitally silent (−180 dBFS) because most blocks
  fall between the clicks, so a per-block threshold calls busy music silent;
* **silence never teaches the loudness baseline anything**, which is what
  stopped the first track after a silent start reading as 20× normal.

| input | `silence_dbfs` |
|---|---|
| line feed from the XR16 (the rig) | −70 (default) |
| room mic, or Spotify via loopback in a noisy room | ≈ −45 |

The idle look is deliberately not a show: one smooth swell travelling the
corridor over 48 s, a very slow plasma, hue drifting one turn per five minutes,
and nothing beat-locked — all driven by wall time. Measured, it changes about
**2 units per channel per 5 seconds**. It never goes fully dark at either end
of the corridor, because a dark rig reads as a fault rather than as rest.

### The state machine

Each cue is used where it is actually discriminative, which took measuring
rather than guessing:

| question | cue | why not the obvious one |
|---|---|---|
| is it quiet? | loudness vs a 45 s baseline | — |
| is it a build? | **high-band share** (0.57 vs ≤0.33) | loudness *falls* in a build |
| is it a drop? | **an event**: a kick out of a build, with the energy back in the bass | a verse reads 1.52 and a drop 1.67 — no threshold splits them |

A drop is not a level, it's the moment a build resolves.  Waiting for a
loudness threshold puts the lights behind the room by however long the
smoothing takes; the kick is audible in one block.

Three findings that shaped it:

* **Band shares must be energy-weighted.**  A running mean of per-frame ratios
  is dominated by the near-silent frames between transients, where all that's
  left is a broadband noise floor — so sparse material looks like a permanent
  build.
* **A build has kicks too.**  Their peaks (9.5–67) completely overlap the
  drop's (18–21).  What separates them is where the energy is: 0.19–0.26 bass
  in a build, 0.44–0.63 in a drop.
* **The energy fallback needs a warm baseline.**  Early in a track everything
  is "average", so the first loud passage spikes the ratio whether or not it's
  a drop — a verse reached 1.77 while the track's actual second drop only
  reached 1.73.  The build-then-kick rule needs no history and works from the
  first bar.

Measured against the test track's known arrangement:

| section | true | detected | late by |
|---|---|---|---|
| build | 30.0 s | 32.3 s | +2.3 s |
| **drop** | 45.0 s | **45.5 s** | **+0.5 s** |
| break | 75.0 s | 75.4 s | +0.4 s |
| build | 90.0 s | 93.3 s | +3.3 s |
| **drop** | 105.0 s | **105.5 s** | **+0.5 s** |
| outro | 135.0 s | 135.3 s | +0.3 s |

Every boundary found, no spurious transitions.  The intro reads as *cruising*
rather than *quiet*, and that is inherent: nothing causal can call a passage
quiet before it has heard anything loud.

### The arranger

The offline recipes as a stepping loop.  The palette journey advances on each
state change, the corridor phrase rotates every few bars picking patterns by
density, and articulation comes from how clear the pulse actually is right now
(`clock.confidence`) rather than an offline `drive` number.  When the clock is
not confident it leans on energy instead of the grid — beat-locked strobing off
a *wrong* grid looks far worse than a wash that merely breathes.

Anticipation is **abortable** by construction: a build ramps tension with time
elapsed, and if the sweep ends without a drop it simply relaxes.  Nothing
pre-fires the resolution.

The four "Big Triangle" nets — top, bottom-left, bottom-right and an inverted
one in the middle — are mounted as one large triangle, and the canvas carries
a second frame of reference for them (`Canvas.big_geo`, projected from the
xLights world positions onto the nets' common plane) in which x/y run across
the whole big triangle and r/angle are about its centre.  Any net effect
takes it through its `geo` argument, and the `big_*` gestures use it: one
wheel turning about the big centre, one ring per beat leaving it, bands
sweeping across all four nets, while the small nets echo the same effect at
their own scale, dimmer.

There is a third frame for *every* net at once (`Canvas.all_geo`): the whole
array as one surface, about 3.3:1 wide, with `net_order` listing the nets left
to right.  The `all_*` gestures live on it — a band sweeping end to end and
back, a ball bouncing along the room, a wave rolling through, a burst from the
centre of the array, both ends slamming into the middle on the beat, the nets
lit one after another.  `blob` and `sweep` are the two effects added for
these; both, like everything else, take the frame through `geo`.

Variety comes from three places, all seeded so the same audio still renders
the same show.  The corridor has fourteen patterns and the nets twenty-odd
gestures, walked without repeating the previous one; the phrase is the
*colour* clock (24 degrees of hue per phrase) and the material is held for
`pattern_hold` phrases, four by default, so a look develops through colour
before it is replaced.  Each *visit*
to a state chooses its own colour scheme (a second drop may be tetradic where
the first was triadic), saturation and corridor depth rotation.  And a change
of look — a new phrase, a new state — arrives by a transition chosen per
change: a cut, a one-to-four-beat crossfade, a wipe down the tunnel in either
direction, or a dip through a dimmer middle.  A change of *state* always
cuts — the music changed, and a dissolve there reads as the lights lagging
it; the drop is only the loudest case.

The phrase counter is the arranger's own, rate-limited and ended on a bar
line, rather than read off the clock's beat index: that index moves whenever
the clock relocks, and following it changed the pattern every half second
through a drop where the tracker sat at 0.03 confidence.  The pattern chosen
for a phrase is also held for the phrase, since the density target moves with
clock confidence and re-deciding every frame reshuffled the candidates under
the walk.

## What M6 adds — the loop closes

```bash
./live.sh sim track.mp3 --play          # audio -> DDP -> fseq, with the sound
./live.sh serve --audio track.mp3       # the show, in the browser
./live.sh serve --audio-device 2 --ddp auto
```

`sim` runs the real engine, sends real packets, and the fake Falcon writes
exactly what a controller would have received.  `--play` plays the same audio
so a person can watch the preview against the music rather than trusting
numbers.

Measured over the whole 150 s test track: **6011 frames at 40.0 fps**, 156 286
packets, all 37 084 channels lit, **4.09 ms/frame**, 15 late frames, audio lag
0.0 ms, tempo locked at 127.93.

And the plan's soak — ten minutes unbroken, through four restarts of the audio:

```
frames  : 24819 in 620.4s (40.0 fps)
packets : 645294        channels lit: 37084 / 37084
engine  : 40.00 fps, 2.65 ms/frame, 11 late, 0 skipped
audio   : lag 0.0 ms, tempo 128.11, confidence 1.00, bar confidence 0.77
```

Eleven late frames in ten minutes, none skipped, and the clock still locked at
the end.  Note that a soak capture is large — 37 084 channels x 24 819 frames
is 920 MB — so delete it or point `--out` somewhere disposable.

## Running it at the rig

Everything that changes between the sofa and the venue lives in **`live.toml`**,
because nobody remembers command-line flags at a gig and the machine that
matters is headless.  A flag beats the file; the file beats the default.

```bash
./live.sh doctor                    # the ten-minutes-before-doors check
./live.sh serve                     # uses live.toml
./live.sh serve --fps 30            # override one value for this run
```

`[output] host` is empty by default on purpose — a laptop should render
without blasting the rig.

### `doctor`

Every check answers a question with an obvious remedy, and says which:

```
[  ok  ] layout          33 models, 37084 channels, 24 arches
[ warn ] addressing      25 models outside every controller
                         -> the arches and par have no controller; only the nets will light
[  ok  ] aubio           version 0.4.9
[  ok  ] audio input     signal at -33.0 dBFS peak
[ warn ] falcon          no host configured -- rendering only
[  ok  ] render budget   0.57 ms/frame, 2% of 25 ms at 40 fps
[  ok  ] beat tracking   100% of beats within 30 ms, tempo 128.2, 95x real time
```

The audio check **actually opens the device and listens**, which is the one
that earns its keep: a device can exist, be selected, and be silent because
nobody plugged the aux cable in, and nothing else would notice.  It also warns
on clipping, which ruins onset detection.

Errors mean the show will not run; warnings mean it will run in a way you
should know about.  A laptop with nothing set up comes out all-green on errors,
or nobody would ever run it.

### If the Pi is slow

`doctor` reports the render budget against the configured frame rate.  If it
is tight, the knobs are **not** equally priced:

| knob | saves | costs |
|---|---|---|
| `[output] fps` 40 → 20 | half the render cost | nothing — the offline show has always rendered at 20 fps |
| `[web] preview_fps`, `preview_detail` | real CPU, and bandwidth | only the browser preview |
| `[audio] blocksize` 512 → 1024 | half the analysis cost | **beats within 30 ms: 100 % → 68 %**, median error 2.4 → 25.6 ms |
| `[audio] window` 2048 → 1024 | more | **100 % → 50 %** |

Measured against exact ground truth on the synthetic track.  The frame rate is
free; the DSP knobs are a last resort, because the whole design premise is
landing on the beat.  A faster Pi is cheaper than a third of the beat accuracy.

### Installing on the Pi

```bash
./deploy/install-pi.sh
```

Installs `python3-aubio` **from apt** rather than pip — the 2019 release needs
two build flags to compile against modern numpy and ffmpeg (see `setup.sh`),
and Debian has already done that work — then makes the venv with
`--system-site-packages` so it is visible.

It installs `requirements-live.txt`, deliberately *not* `requirements.txt`:
the offline generator's librosa, numba, scipy and yt-dlp are a slow, fragile
build on a Pi and none of them run at the rig.

Then a systemd unit with `Restart=always`, `Nice=-5` (audio capture and a
40 fps render both suffer from being preempted), and
`After=network-online.target` — without which the service can start before the
interface has its static address and never find the Falcon.

`[show] blackout_on_start` decides whether a mid-set restart comes back dark.

## Checking the orientation, after a remap

Net geometry is taken from each model's world position in the xLights
layout — world X across, world Y up, normalised per net — which is what
xLights' *Per Preview* render style does.  It matters because the nets are
rotated in the layout (two by −90° about Z, the big ones flipped about X, one
inverted): counted in the custom model's grid rows, "up" was sideways on some
nets and downward on others.

`./live.sh orient` sends three looping stages to both Falcons (stop the
service first — `sudo systemctl stop triangles-live` — or the two will fight
over the rig): a white band rising from base to apex on every net at once;
the same band rising across the four "Big Triangle" nets *as one triangle* —
the two base nets first, then the inverted middle one, the top net last; and
one arch lit at a time from the front of the corridor to the back.  `--stage
"big up"` holds one stage.  A band that falls, runs sideways, or crosses the
four big nets out of step points at that model's orientation or world
position in xLights; arches lighting out of order point at the Tunnel group.
`./live.sh pattern` remains the channel-map check (start channels, colour
order, node maps).

## Recording a session, when something is wrong

Three rounds of guessing at a detection problem is three rounds too many.  The
fix for that is a recording, not a better guess:

```bash
./live.sh serve --audio-device 2 --record-session sessions/friday
./live.sh analyze sessions/friday            # what happened, and what looks wrong
./live.sh analyze sessions/friday --replay   # rerun the chain on that exact audio
```

It writes the **audio itself** (16-bit WAV, 5 MB a minute), every per-block
number the analysis derived, the clock's state, every transition *with the
reason it gave*, and a brightness curve per fixture family — all on one
timeline.  A four-hour set is under 1.5 GB.

The audio is the important part: with it the whole pipeline can be replayed
offline, deterministically, as many times as it takes.  The traces matter
because a replay only proves what the code does *now*, and the question is
usually what it did *then* — with that config, that device, that gain.

`analyze` is deliberately opinionated rather than a dump of numbers.  It flags
things like *"quiet while the level was in its top 30 %, first at 84.2 s"* —
which is the sentence that would have saved those three rounds.

## Validating against real music

Every threshold in this project was chosen by looking at one or two files —
mostly `test_track.wav`, which is *synthesised*.  That is how you end up with a
show that works on the track you developed against.

```bash
./live.sh corpus tracks/                    # one row per track, bad ones flagged
./live.sh corpus tracks/ --offline          # also compare offline boundaries
```

**Most of what it measures needs no oracle**, deliberately.  The offline
generator sees the whole file and is better-informed, but it is not ground
truth — a mistake already made once here, when librosa's beat grid turned out
to be on the *offbeat* for two of five tracks and reported us 60 % wrong while
we were right.  So the primary signals are true or false on their own terms:

| signal | what a bad value means |
|---|---|
| `kick` ours / offbeat | left lower ⇒ we are on the wrong half of the beat |
| `lock` | share of the track the clock was confident |
| `bar` shifts / confidence | one shift then stable is the tracker working; fifteen is not |
| `st/min` | state transitions per minute — high means flapping |
| `x rt` | speed vs real time, which is what decides whether it fits on a Pi |

Offline boundaries are compared as a **flag**, not a score: a disagreement is a
place to go and listen, not proof that either side is wrong.

### What to put in `tracks/`

`tracks/` is gitignored — supply your own music; nothing is downloaded.  Two
requirements that are easy to get wrong:

* **Full tracks, not excerpts.**  The loudness baseline is 45 s and the peak
  reference releases over 120 s, so a 97 s excerpt never fills either.  The
  `out/*.mp3` files are excerpts written by `run.sh` and systematically
  under-test the engine for this reason.
* **One long DJ mix is worth more than twenty singles.**  Transitions are what
  defeat beat trackers, and a mix is the only way to test them — plus it is
  literally what the rig will hear.  Nothing in this repo has ever been tested
  across a crossfade.

Diversity matters more than count.  Worth covering:

| kind | what it stresses |
|---|---|
| house / techno, four-on-the-floor | the baseline case |
| drum & bass, half-time | octave errors (tempo folds into 70–180) |
| hip-hop / trap | sparse, swung, half-time feel |
| pop with a backbeat | beat-vs-snare confusion |
| ambient / downtempo | weak pulse — free-run and confidence |
| a long breakdown, or a real tempo change | re-lock |
| live or acoustic | loose timing |
| anything in 3/4 or 6/8 | `bar_length` is hardcoded to 4 |

## Ground truth: the Harmonix Set

912 Western pop and dance tracks annotated with **beats, downbeats and
functional segments**, plus BPM and time signature — MIT-licensed, from
[urinieto/harmonixset](https://github.com/urinieto/harmonixset) ([ISMIR 2019
paper](https://ccrma.stanford.edu/~urinieto/MARL/publications/ISMIR2019-Nieto-Harmonix.pdf)).
The first real oracle this project has had for three things it could only
check against a file we generated ourselves.

```bash
git clone --depth 1 https://github.com/urinieto/harmonixset.git datasets/harmonix

./live.sh harmonix --list --genre Dance --min-bpm 120   # browse what it covers
./live.sh harmonix ~/Music --score                      # score what you own
./live.sh harmonix --synth 30                           # score without owning anything
```

The set ships annotations, not audio.  `--score` matches your library by
MusicBrainz id, then tags, then filename, and warns when a file's duration
disagrees with the annotation — a remaster or radio edit lines up at the start
and drifts, which looks like a tracking failure and is not.

### Testing 912 real structures without owning a note

`--synth` renders an annotation *as audio*: a kick on every annotated beat, a
bass note and crash on every downbeat, hats, and a high-band sweep through
sections mapped to *building*.  The timbres are ours, so this says nothing
about how aubio copes with a dense modern mix — but the **structures are
real**: real tempo curves including drift and mid-track changes, real bar
layouts including the 22 tracks not in 4/4, real arrangements.  Against that,
our one hand-written synthetic arrangement was a sample of size one.

Segment labels map to our states through the pop analogue of build-and-drop:
**prechorus → building, chorus → hot**.  That's a judgement, and it lives in
one visible table in `harmonix.py` rather than scattered through the scoring.

### What it measured

**On dance/electronic at 115–145 BPM — what this rig actually plays** (14 tracks),
before and after fixing the metrical-flip bug below:

| | before | **after** |
|---|---|---|
| beat precision | 89 % | **95 %** (worst 67 %) |
| beat recall | 91 % | **97 %** |
| downbeats within 50 ms | 70 % | **95 %** |
| tempo error | 0.1 % | 0.1 % (1 track over 5 %) |
| tempo wander across the track | 1.6 BPM | 1.1 BPM |
| clock locked | 96 % | **98 %** |
| tracks below 50 % beats | 3 | **0** |

### The metrical-flip bug

The corpus kept reporting tempo wandering mid-track even where the final value
was right.  Instrumenting one case made it obvious: on a 140 BPM track,
**aubio's own readout alternates between 142 and 94** — and 142 × ⅔ = 95.  It
is changing its mind about which pulse to count, not hearing a tempo change.
M4's coarse term followed it every time, so the clock ping-ponged.

The fix is to ask *what kind* of disagreement it is.  A readout at a simple
metrical ratio of our current tempo — ½, ⅔, ¾, 4/3, 3/2, 2, 3 — is the same
music counted differently, and our own estimate is anchored to observed beat
times, so it wins.  A real tempo change does not land on those ratios: a DJ
nudging pitch moves a few percent, and 128 → 140 is a ratio of 1.09.

It only applies while the clock is confident, or a bad initial lock could never
be talked out of it — and it has a **patience limit** for the same reason.  The
guard cuts both ways: if the clock ever settles on the wrong pulse, every
correction back *also* looks metrical.  One track sat at 149 BPM while the
tracker said 117, a ¾ ratio, and the guard would have kept it there
indefinitely.  Twenty beats of sustained disagreement and the tracker wins.

### The polarity deadlock

The other fix the corpus forced.  One track scored **0 %** with a *correct*
tempo (128.8 against an annotated 128) and a perfectly uniform annotated beat
spacing — a median error of 227 ms, which is half of its 469 ms beat.  It sat
on the offbeat for its entire five minutes.

The kick-based polarity check exists precisely to catch that, and it was
disabled: it required `confidence > 0.4`, and *being on the offbeat is what
keeps confidence low*.  Confidence sat at 0.19, so the one mechanism that could
have fixed the phase was switched off exactly when it was needed.

Confidence now scales how much evidence is **demanded** rather than whether the
check runs at all.  That track went from **0 % to 98 %** of beats within 30 ms,
confidence 0.19 → 0.93, with a single half-beat correction.

Corpus-wide across all genres, the two fixes together moved tempo wander from a
median of 13.3 BPM to 5.3, tempo error from 3.7 % to 2.2 %, and poorly-tracked
tracks from 12 of 24 to 9 — the rest being the non-4/4 and sub-80 BPM cases.

### A flaw in the measurement, not the engine

Worth recording because it nearly sent me the wrong way: `corpus.py` originally
rebuilt the beat grid from the clock's **final** tempo and anchor.  On a track
whose tempo wandered 43 BPM that is a grid which never existed, and it reported
Opus as sitting on the offbeat when it was not.  It now uses the beats actually
fired.  The synthetic tracks' phase margin went from 3.7× to 7×, so the metric
is sharper as well as correct.

### Still open: tempo wander on real audio

The fixes above are validated on rendered annotations, where aubio's readout
flips cleanly between metrical levels.  On real mixes it does something else —
it wanders *continuously*, which the ratio guard cannot catch.  Measured
against each track's true tempo:

| track | within 2 % | within 5 % | worst excursion |
|---|---|---|---|
| Opus | **42 %** | 56 % | 21 % |
| Midnight Vampires | 75 % | 81 % | 32 % |
| Children | 85 % | 85 % | 7 % |

Be precise about what this breaks: **phase is still right** — Opus scores 3.81
kick on our beats against 3.53 on our offbeats, so lights still land on beats,
because the PLL re-locks on every one.  A wrong tempo hurts *prediction
between* beats and phrase counting, not the downbeat itself.  The likely fix is
to weight the long-baseline estimate over the tracker's readout, and it needs
real audio to validate rather than renders.

### Crossfades: the case nothing had ever tested

`./live.sh harmonix --crossfade` mixes two annotated tracks the way a DJ does —
beatmatched, bar-misaligned, tempo-jumped, or hard cut — and keeps both grids,
so the clock is scored either side of the blend.  Scaling the annotation times
*is* the tempo change, since the audio is rendered from those times, so the
ground truth stays exact and no resampling is involved.

First results are not good: beat precision runs 49–91 % before a transition and
**4–16 % eight seconds after it**.  Beatmatched and bar-misaligned mixes
re-settle within about 0.2 s in three cases of four; tempo jumps and hard cuts
never re-settle inside the window measured.

Two caveats before treating that as the number.  One pair measured by hand
gives **100 %** after the blend, so the harness can report success and most
sampled pairs really do fail.  And `settle` and `after` look at different
windows — just after the blend, versus eight seconds later — so "settles
quickly then drifts off" is a coherent reading rather than a contradiction.

The confidence trough of **0.11** in every scenario is the encouraging part:
the clock *knows* it has lost the plot, and the arranger already falls back to
energy rather than the grid when confidence is low.  The failure mode is
degraded, not wrong.  Not acted on yet — this needs one more pass.

**On a spread of 30 across every genre**, beat precision drops to 54 % median.
The failures are not random — they are three specific, now-measured limits:

* **Non-4/4 is unusable.** 6/8, 3/4 and 6/4 tracks score 16–29 %.  `bar_length`
  is hardcoded to 4, so this is expected; it is now quantified rather than
  suspected.
* **Slow tracks get octave-doubled.**  66→133, 78→156, 82→164, 87→173: the
  70–180 folding range resolves sub-80 BPM material at double time.  Precision
  and recall together are what reveal this — recall stays high because every
  annotated beat still has one of ours on it.
* ~~Tempo wanders mid-track~~ — **fixed**, see above; that was the
  metrical-flip bug, and on real mixes a different form of it remains open
  (below).

Precision and recall are reported separately for exactly this reason — a single
"beats within 30 ms" number would have hidden every octave error.

### What it found on the four real excerpts we have

All four have the low end on our beats, so phase is right — but the clock's
**tempo wanders on real material in a way it never does on the synthetic
track**:

| track | tempo p5 / p50 / p95 |
|---|---|
| `test_track.wav` (synthetic) | 127.8 / 127.9 / 128.3 |
| Midnight Vampires | 114.4 / 115.4 / 132.4 |
| Opus | **100.7** / 125.2 / **143.0** |

Opus loses the tempo for about twenty seconds mid-track and recovers.  The
suspect is M4's coarse tempo term, which follows aubio's BPM readout when it
disagrees persistently — on the synthetic track that readout never wanders, so
it never fires.  Bar confidence also sits near zero on all four real tracks
against 0.13–0.18 on the synthetic one.

Both are held open rather than tuned away: fixing them against four excerpts
would repeat exactly the mistake this harness exists to catch.

### Song structure, and what is actually knowable live

Nothing causal can name a section.  "This is the second chorus" needs either
the whole file or a model trained to say so, and the BeatNet spike showed the
live-capable options do not deliver even downbeats reliably.  What the state
machine gives is the coarse energy shape — quiet, cruising, building, hot —
which is an *energy* taxonomy, not a structural one: an intro and a breakdown
look identical from inside the moment, and so do a verse and a post-chorus.

But song structure is repetition with variation, and **the repetition is
detectable by counting**.  The show does not need to know a section is a chorus
to know it is *the second time we have been here*.

Measured before this existed: the test track's two drops drew identical
corridor patterns and identical net gestures, differing only in colour, and
that colour difference was incidental — the golden-angle journey happened to
have moved.  The second drop of a track is almost always bigger than the first,
and the show had no way to say so.

Each return now escalates: the density target rises, so a busier pattern comes
into range; brightness lifts; and the walk is re-keyed on the visit number so a
return does not replay the same sequence in the same order.

| | visit | brightness |
|---|---|---|
| drop 1 | 1 | 33.7 |
| **drop 2** | 2 | **43.5** |
| build 1 | 1 | 28.6 |
| **build 2** | 2 | **41.4** |

It is capped after three returns — a fourth chorus should not be blinding.

**Real section names remain an offline problem**, and the plan already has the
shape of the answer under "semi-live cueing": fingerprint the track, look up a
precomputed structure, and get lookahead for known material.  The Harmonix Set
gives 912 annotated tracks to build that against.

### What changes, and when

Three timescales, which is the offline show's structure kept intact:

| timescale | what moves | why |
|---|---|---|
| **per phrase** (4 bars) | corridor pattern, net gesture, a 9° hue step | one gesture held for a section reads as one idea |
| **per state change** | palette jumps a golden angle, treatment changes | the show travels through colour rather than sitting in a corner of it |
| **per beat / bar** | accents, the big/small net trade, par flashes | the grid itself |

The nets rotate through a gesture vocabulary the same way the corridor rotates
patterns — `bars`, `trade`, `slow_wheel`, `rings`, `wheel_up`, `strobe_small`,
`flare`, `plasma`, `twinkle`, `breathe` — each state drawing only from gestures
that make sense at that energy: a drop can strobe, a breakdown cannot.  This
matches what the offline show does for a drop ("butterfly/spirals/fan, rotating
every 4 bars").

The hue steps 9° per phrase on a 4-phrase cycle *inside* a section, so a
ninety-second drop is not one flat colour, while the golden-angle jump between
sections stays the real journey.

Both walks are deterministic and never land twice in a row.  The obvious
`(phrase * step) % count` does **not** give that — the step is derived per
phrase, so consecutive phrases can collide, and measured, a drop drew
`bars_fast` twice running.  Accumulating a step that is never a multiple of the
count does.

### The corridor on `auto`

`vocabulary()` returns the **three** patterns nearest the energy the state
wants, and the corridor draws a different one each phrase rather than the
nearest one every time.  This is the offline show's rule — *"every phrase draws
a different pattern, never repeating the previous one"* — and it exists because
a single travelling comet repeated for ninety seconds reads as one idea however
well it tracks the music.

It was missed at first: `pattern_for` computed the three candidates and then
always took `[0]`, so the corridor held one pattern for a whole section — four
patterns across 150 seconds, changing only when the state changed.  It now
draws 25, from seven.

Two rates, kept separate because they answer different questions:

| | what it controls | quiet / cruising / building / hot |
|---|---|---|
| `phrase_bars` | how fast the gesture travels | 4 / 4 / 2 / **1** bar |
| `pattern_bars` | how often a *new* pattern is drawn | 8 / 4 / 4 / **4** bars |

Sharing one number gave a drop a new pattern every 1.9 s, which reads as
thrashing rather than energy.

The walk is deterministic — a crc32-derived step that is never zero, so
consecutive phrases always differ and the show still renders identically twice.
`hash()` would have been the obvious choice and is wrong: it is salted per
process, so two runs of the same show drew different patterns and the
`--fseqcmp` oracle would have started reporting false differences.

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
./live.sh pattern --host 192.168.1.20 --controller Falcon_F16V5_0E1C --loop
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
