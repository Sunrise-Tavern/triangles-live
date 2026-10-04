# triangles-live

Music-driven lighting for the Triangles rig, in two halves that share the rig
but not the code:

- **`live/`** -- the live engine: listens to the DJ feed and drives the Falcons
  over DDP in real time.  Operator's guide: [`live/README.md`](live/README.md).
- **`triseq/`** -- the offline generator: song in, xLights `.xsq` out.  The
  rest of this file is its guide.

`AGENTS.md` is the map of the whole repo.

## The xLights show folder

Both halves read the rig from an xLights show folder, which is **not** in this
repo: `xlights_rgbeffects.xml` (models, groups, geometry) and, for the live
engine, `xlights_networks.xml` (controllers and addressing).  Tell them where it
is, first match wins:

1. `TRIANGLES_SHOW_DIR=/path/to/show` in the environment
2. `[xlights] show_dir` in `live.toml` (live engine only; relative to the file)
3. the folder this repo sits in -- a checkout inside the show folder needs nothing

```bash
export TRIANGLES_SHOW_DIR=~/xLights/Triangles
./live.sh doctor          # the "layout" line says which folder it read
```

## License

GPL-3.0 (see `LICENSE`).  `effect_registry.json` is derived from xLights'
source, which is GPL-3.0.

# Music-driven sequence generator

Takes a song, analyses it, and writes an xLights `.xsq` that lands on the beat —
driving the arch corridor, the nets, and the DJ par with an intro → build → drop
→ breakdown → outro arc taken from the actual track.

## Setup (once)

```bash
./setup.sh
```

Creates `.venv` on Python 3.12 and installs librosa, soundfile, scipy and yt-dlp.
(The system `python3` is 3.14, which librosa's `numba` dependency doesn't
support yet — hence the pinned interpreter.) Needs `ffmpeg`, already installed.

## Use

```bash
./run.sh "https://www.youtube.com/watch?v=..."     # from a link
./run.sh ~/Music/track.mp3                         # from a file
./run.sh ~/Music/track.mp3 -o warehouse --duration 120 --style edm --seed 3
```

Output lands in `out/`:

- `out/<name>.xsq` — open this in xLights, then render (Ctrl+F3)
- `out/<name>.mp3` — the excerpt, referenced by the sequence so the waveform loads

### Options

| Flag | Meaning |
|---|---|
| `-o, --name` | Output basename (default: from the track title) |
| `--duration` | Pin the length (clamped to **90–120 s**). Omit to let it vary with the song |
| `--start` | Excerpt start, e.g. `1:12`. Omit to auto-pick the best window |
| `--full` | Sequence the **entire song** instead of picking an excerpt |
| `--style` | Force `edm` / `rock` / `ambient`. Omit — it's derived from the track |
| `--corridor-rate` | Bars per corridor traversal. Lower = faster tunnel |
| `--seed` | Makes the arranger's random choices reproducible |
| `--keep-download` | Keep the full downloaded track in `.cache/` |
| `--no-validate` | Skip the post-write validation pass |

## What it generates

Three timing tracks — **Beats**, **Bars**, and **Sections** (labelled
`intro`/`build`/`drop`/…) — so the result snaps to the real musical grid if you
hand-edit it afterwards.

### Arrangement density

`--style` is an override, not a setting you're expected to use. The density is
derived from the audio: `Features.drive` combines **pulse clarity** (how
strongly the onset envelope autocorrelates at the beat period) with **onsets per
beat**, and everything scales off that one number — accent density, accent
threshold, corridor gesture rate, wave frequency.

A tight, busy track gets fast corridor gestures and hits on most beats. A loose,
sparse one gets long sweeps and only the strongest accents, because hitting
every beat of music that doesn't insist on its beat looks mechanical.

Measured on real tracks: a four-on-the-floor dance track lands at drive 0.88–0.90
(≈0.95 bars per corridor traversal), a pop track at 0.66, a long textural
progressive-house build at 0.40 (≈1.5 bars — noticeably calmer).

> Percussive-energy ratio via HPSS looks like the obvious ingredient and is not
> usable. It measures the production, not the genre: a pure-synth track scores
> *lower* than a live-drum one because a sine bass gets filed as harmonic.

### Choosing the window

The length is not fixed. Every bar-aligned start is scored against every
candidate length in 90–120s, so the window can stretch to finish a drop instead
of cutting through the middle of one.

Sections are then found **twice**, and the split matters:

- **Boundaries** come from re-segmenting inside the chosen window. Clustering a
  whole song places boundaries where the largest contrasts across the *song*
  are, which on a track with a long gradual build lumps minutes into one span.
- **Labels** use the whole track's intensity scale (`label_scale()`), passed in
  via `find_sections(stats=...)`. A window is chosen partly *for* being
  energetic, so letting it re-baseline against its own contents demotes the very
  drop that made it worth picking.

### Keeping things moving

Two rules, both learned from sequences that came out visually dead:

- **No sustained effect spans a whole section.** `_phrases()` chunks beds onto
  2-bar (4 in verses) boundaries. A single Color Wash across a 49-second build is
  the most static thing the generator can emit, however high its cycle count.
- **Rates are derived, not constant.** `_cycles()` scales a sweep's cycle count
  to the bars it covers, because `cycles=1.5` sweeps nicely over two seconds and
  is frozen over thirty.

Corridor patterns hold each arch well past the next one's start so lit arches
overlap — without that, a wave reads as 24 things blinking in sequence rather
than one band travelling through the tunnel.

| Section | Tunnel | Nets | DJ |
|---|---|---|---|
| intro | slow wash, dark palette | sparse twinkle | dim, rising |
| verse | wash + per-bar corridor wave | big/small nets trade bars | pulse on the downbeat |
| build | wave accelerates each bar | pinwheel speeds up, small nets strobe | pulse rate doubles |
| drop | wave every 2 beats | butterfly/spirals/fan, rotating every 4 bars | flash on every kick |
| break | dim wash | slow galaxy | held low |
| outro | wash fading out | twinkle fading out | fade to black |

The corridor wave is built from **per-arch effects with staggered starts**, not a
group-level sweep. The 24 arches all sit at the same position in the 3D layout,
so a group buffer has no depth for a wave to travel through — the stagger is
what makes it read as motion down the tunnel.

### Corridor patterns

Each section is split into phrases (4 bars, 2 in intros) and every phrase draws
a different pattern, never repeating the previous one:

`sparkle` (0.20) · `comet` (0.30) · `converge` / `diverge` (0.42) ·
`bounce` (0.55) · `pairs` (0.72) · `alternate` (0.90) · `strobe` (1.00)

The number is the pattern's density. Patterns are chosen from the three nearest
a per-section target, so the corridor's business follows the song's energy —
picking uniformly at random lets a drop draw a sparse comet while a verse draws
a full strobe, which inverts the dynamics. Quiet sections are additionally
capped below 0.45 so a breakdown can never strobe.

### Colour

Palettes are **generated**, not chosen from a list. Three inputs:

1. **Base hue from the music** — the track's dominant pitch class mapped onto
   the colour wheel, so each song has its own colour identity (`Features.key_hue`).
2. **A hue journey** — each section rotates by the golden angle (137.5°), which
   maximises the run before hues start repeating or landing near a previous one.
3. **A harmony scheme per section kind** — contrast rises with energy:
   `analogous` for quiet sections, `split` for verses, `complementary` for
   builds, `triadic` for drops.

Value and saturation come from the section kind, then `Palette.floored()`
guarantees a minimum visible brightness. This replaced a lookup of 16 hardcoded
palettes indexed by spectral centroid, under which any two sections with a
similar centroid drew the *identical* palette — one test track spent its whole
second half in the same green.

Two colour invariants worth keeping in mind when editing:

- Quiet does not mean dark. An LED below ~40/255 reads as off, so
  `FLOOR` and `_depth(..., floor=)` keep intros dim but visible.
- The corridor gradient shifts **hue** as well as brightness, so the tunnel runs
  cyan-to-violet down its depth rather than merely fading out.

## Adding another audio source

`triseq/sources.py` holds a `SOURCES` dict mapping URL scheme → class. Any class
with a `resolve(workdir) -> AudioAsset` method works; add one line to register it.

## Layout

```
triseq/show.py       reads xlights_rgbeffects.xml -> model/group inventory
triseq/sources.py    pluggable audio input (YouTube, local file)
triseq/analysis.py   beats, bars, band energy, kick detection
triseq/structure.py  section detection + labelling, excerpt picking
triseq/arrange.py    sections + beats -> effect timeline
triseq/effects.py    effect vocabulary and parameters
triseq/palettes.py   colour palettes
triseq/xsq.py        .xsq writer + validator
```

### The effect registry

`effect_registry.json` holds all **56 effects** the installed xLights supports,
with every parameter's type, default, range and enum options. It is generated
from xLights' own machine-readable metadata:

```bash
./.venv/bin/python build_registry.py     # re-run after upgrading xLights
```

Use any effect by name with `fx.build(...)`, giving parameters by their metadata
id. Values are checked against the registry — unknown parameter names raise,
numbers clamp to the effect's range, enums must be one of its options. That
check matters because xLights *silently ignores* settings it doesn't recognise,
so a typo would otherwise surface only as an effect rendering with defaults:

```python
fx.build("Plasma", Plasma_Style=4, Plasma_Speed=45, fade_out=0.3)
fx.defaults_for("Warp")     # every parameter and its xLights default
```

Curated builders (`fx.on`, `fx.bars`, `fx.spirals`, …) remain the convenient
path for the effects used most.

Two things this replaced, both worth remembering:

- Parameters were originally scraped from the binary with `strings`, which only
  catches keys existing as whole string literals — that found 393 of 573, so the
  vocabulary was limited by the extraction, not by xLights.
- They were *not* copied from the sequences in this show folder: one of those
  has an `On` effect whose `ref` points at a `Bars` settings string, and
  harvesting would have propagated it.

A wrong effect **name** is still a broken sequence, so `EFFECT_NAMES` comes from
the registry and `EffectSpec` rejects anything outside it.

The arranger deliberately uses a subset. Effects needing external assets or
specific fixture types — `Faces`, `Pictures`, `Video`, `Text`, `Piano`, `Servo`,
`DMX`, `Moving Head`, `Shader`, `Glediator` — are available through `fx.build`
but wrong for this rig.

## Testing offline

`make_test_audio.py` synthesizes a 150 s, 128 BPM track with a known arrangement,
useful for checking structure detection without downloading anything:

```bash
./.venv/bin/python make_test_audio.py .cache/test_track.wav
./run.sh .cache/test_track.wav -o smoke_test
```

## Note on sources

Downloading audio from YouTube is outside its terms of service. Passing a local
file you already have is a first-class path and avoids the issue entirely.
