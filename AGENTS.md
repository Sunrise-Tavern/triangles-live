# Triangles — orientation for agents

Music-driven lighting for the Triangles rig (7 triangular nets of two sizes
-- 465 and 435 nodes -- and a 24-arch corridor; ~35k channels across two
Falcons via DDP; the DJ par was dropped from the layout on 2026-08-30). Two
engines share the rig and the ideas but **not** the code:

| | `triseq/` — offline generator | `live/` — live engine |
|---|---|---|
| input | a song file or URL | the DJ feed (line in), or a file |
| output | xLights `.xsq` in `out/` | DDP packets at 40 fps (+ optional `.fseq` capture) |
| sees | the whole track at once (librosa) | only the past (aubio, streaming) |
| structure | segmented and labelled intro/verse/build/drop/break/outro | rolling state machine: silent/quiet/cruising/building/hot |
| beats | librosa grid | a PLL clock that **predicts** the next beat |
| entry | `./run.sh <song>` → `make_sequence.py` | `./live.sh <command>` → `live/__main__.py` |
| deps | librosa, soundfile, scipy | numpy + aubio only (it must run on a Pi) |

`LIVE_PLAN.md` is the live engine's design rationale, milestones and
measurements. `README.md` is the offline generator's user guide,
`live/README.md` the live engine's operator guide. Read those for depth;
this file is the map.

## Live engine data flow

```
audio.py (device/file → 512-sample blocks)
  → analysis.py   per-block features: rms, bands, flux, kick, novelty, energy
                  (everything is RELATIVE to a 45 s baseline — no absolute levels)
  → beats.py      aubio tempo/beat events
  → clock.py      BeatClock: tempo + phase, predicts beats, confidence, free-run/relock,
                  half-beat (offbeat) correction from where the kick lands
  → downbeat.py   bar line
  → state.py      StateMachine: quiet/cruising/building/hot, hysteresis + dwell
  → arranger.py   state → palette journey, corridor pattern, net gesture, par
  → effects.py / frame.py / geometry.py   pixels on a Canvas
  → ddp.py        → Falcon        (fseq.py captures the same frames to a file)
engine.py owns the threads (audio thread feeds listener.py; render thread at fps);
web.py + static/ is the browser UI, settings.py the knobs it exposes, config.py / live.toml the startup config.
```

Key invariants:
- **Predict beats, don't react to them.** Fire from `clock.state(t)`, never from a detection.
- **Relative, not absolute.** Loudness vs its own baseline, band shares vs their own averages. The one absolute number is `silence_dbfs`.
- **Deterministic rendering.** Same audio + same settings → byte-identical frames (`selftest` pins it). Randomness is seeded by frame index.
- **No lookahead live.** Anything that needs the whole track belongs in `triseq/`, not here.

## Commands you'll actually use

```bash
./setup.sh                          # once: .venv on python3.12, librosa + aubio
./live.sh selftest                  # 17 checks, no hardware — run after ANY change in live/
./live.sh serve [--record-session sessions/NAME]   # daemon + UI at :8080
./live.sh analyze sessions/NAME [--replay] [--window 30]   # diagnose a recording
./live.sh beats .cache/test_track.wav             # clock vs exact 128 BPM ground truth
./live.sh corpus out/                              # every track in a folder, one row each
./live.sh harmonix --synth 24 --min-bpm 110 --max-bpm 150   # ground-truth scoring
./live.sh doctor                    # preflight on the target machine
./run.sh track.mp3                  # offline: .xsq into out/
```

## Measuring before changing

This project's rule is *measure, then change, then measure again*; commit
messages carry the numbers. The tools, from cheapest to most trusted:

1. `selftest` — synthetic, exact, fast. Catches regressions, proves nothing about real music.
2. `beats` on `.cache/test_track.wav` — exact 128 BPM grid with a kickless break.
3. `harmonix --synth N` — 912 real tempo curves/bar layouts rendered as audio (`datasets/harmonix`, `live/harmonix.py`).
4. `corpus` on `out/*.mp3` — real tracks; **kick alignment** (low end on our beats vs our offbeat) is the trustworthy phase signal, no tracker's opinion involved.
5. A recorded session (`sessions/`) — the real input. `analyze --replay` reruns today's code on the recorded audio and diffs.

Do not treat librosa's grid as truth: it sits on the **offbeat** for some
tracks (Robert Miles, Tove Styrke in `out/`). When live and offline disagree
on phase, check where the kick is.

## Where things live

```
live/            the engine (see data flow above)
  selftest.py    the test suite; verify.py / corpus.py / harmonix.py the measurement tools
  session.py     recorder + `analyze`
  presets/       saved knob snapshots
  static/        browser UI
triseq/          offline generator: analysis.py → structure.py → arrange.py → xsq.py
sessions/        recordings: audio.wav + trace.npz + session.json      (gitignored)
out/             rendered .xsq/.fseq/.mp3, also the real-track corpus  (gitignored)
.cache/          synthetic test audio                                  (gitignored)
datasets/        Harmonix annotations, cloned per live/README.md       (gitignored)
deploy/          Pi install script + systemd unit
logs/            live.log
LIVE_PLAN.md     design + milestones + measured numbers + open issues
```

## Known gotchas

- aubio's BPM readout is biased ~+1.3 %; the clock uses it only for octave and real tempo changes.
- aubio's per-beat confidence is not a clean signal (0.0 through intros and clicks); only a *run* of low readings means "no pulse".
- `bar_length` is hardcoded to 4; non-4/4 material is unusable.
- The layout is read from the xLights show folder at startup (not in this
  repo; `TRIANGLES_SHOW_DIR`, else `[xlights] show_dir` in live.toml, else
  the repo's parent folder -- resolved in `triseq/show.py`). Nets are
  padded to the widest node map in the canvas (`Canvas.net_mask`); "Big
  Triangle" is nets 5-8 and the small group is whatever is left
  (`Canvas.net_pair`). Both families are "RGB Nodes" now -- the Falcon does
  the GRB/BGR swap from the port config xLights uploaded, so if a fixture's
  red and green come out exchanged, re-upload the controller from xLights.
- Tempo range folds to 70–180 BPM; sub-80 BPM material is octave-doubled.
- The loudness baseline means the first loud thing after 45 s of quiet reads as "hot" — by design, attempts to fix it cost real drops (see git log).
- Section vocabularies differ on purpose: offline has six kinds, live has four. Don't chase agreement between them.
- Scripts in `.venv` only: system `python3` is 3.14 and librosa won't install there.

## Conventions

- Commit messages: what was wrong, what was measured, what the numbers are now. One concern per commit.
- Every threshold has a comment saying what it was measured on. Keep that up.
- Add a knob by adding one field + one `SCHEMA` line in `live/settings.py`; the UI, presets and validation follow. Startup values go in `live/config.py` + `live.toml`.
- Keep `live/` free of librosa/scipy; it has to run on a Pi.
