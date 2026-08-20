#!/usr/bin/env python3
"""Generate an xLights sequence from a song.

    ./run.sh "https://www.youtube.com/watch?v=..."
    ./run.sh ~/Music/track.mp3 -o my_show --duration 100

Produces out/<name>.xsq and out/<name>.mp3.  Open the .xsq in xLights.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import replace
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"
CACHE_DIR = HERE / ".cache"

#: Hard bounds on the generated sequence. A show longer than two minutes stops
#: being a show and starts being a set, and the arranger's section recipes are
#: tuned for this range. Within these bounds the length is chosen per track:
#: a fixed length has to cut somewhere arbitrary, and on a song with a long
#: build it lands mid-drop.
MIN_DURATION = 90.0
MAX_DURATION = 120.0


def parse_time(value: str) -> float:
    """Accept 90, 1:30 or 0:01:30."""
    parts = value.strip().split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Not a time: {value!r}")
    seconds = 0.0
    for n in nums:
        seconds = seconds * 60 + n
    return seconds


def slugify(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", text).strip()
    slug = re.sub(r"[\s_-]+", "_", slug)
    return slug[:60] or "sequence"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a music-driven xLights sequence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", help="YouTube (or other yt-dlp) URL, or a local audio file")
    p.add_argument("-o", "--name", help="output basename (default: from track title)")
    p.add_argument("--duration", type=parse_time, default=None,
                   help=f"pin the excerpt length in seconds, clamped to "
                        f"{MIN_DURATION:.0f}-{MAX_DURATION:.0f}. "
                        f"Omit to let it vary with the song's structure.")
    p.add_argument("--start", type=parse_time, default=None,
                   help="excerpt start (e.g. 1:12); default is auto-picked")
    p.add_argument("--full", action="store_true",
                   help="sequence the entire song instead of picking an excerpt")
    p.add_argument("--style", choices=["edm", "rock", "ambient"], default=None,
                   help="force an arrangement density. Omit to derive it from "
                        "the track's pulse clarity and onset density.")
    p.add_argument("--corridor-rate", type=float, default=None, metavar="BARS",
                   help="bars per corridor traversal (default 1.0 for edm). "
                        "Lower feels faster, higher calmer.")
    p.add_argument("--frame-ms", type=int, default=None, choices=[25, 50, 100],
                   help="sequence frame grid (default 25). 50 halves the render "
                        "density but adds timing jitter on most tempos.")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the arranger's random choices (default: 0)")
    p.add_argument("--no-validate", action="store_true",
                   help="skip the post-write validation pass")
    p.add_argument("--keep-download", action="store_true",
                   help="keep the full downloaded track in .cache/")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from triseq import analysis, arrange, sources, structure, xsq
    from triseq.show import ShowError, load_show

    duration = None
    if args.duration is not None:
        duration = max(MIN_DURATION, min(MAX_DURATION, args.duration))
        if duration != args.duration:
            print(f"note: clamped duration {args.duration:.0f}s -> {duration:.0f}s "
                  f"(allowed range is {MIN_DURATION:.0f}-{MAX_DURATION:.0f}s)")

    # 1. Show inventory -----------------------------------------------------
    try:
        show = load_show()
    except ShowError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"Show: {len(show.tunnel_arches)} arches, {len(show.nets)} nets, "
          f"DJ = {show.dj}")

    # 2. Audio --------------------------------------------------------------
    print(f"\nFetching {args.input}")
    try:
        source = sources.resolve_source(args.input)
        asset = source.resolve(CACHE_DIR)
    except sources.SourceError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"  {asset.label}")

    # 3. Analysis -----------------------------------------------------------
    print("\nAnalysing")
    feats = analysis.analyze(asset.path)
    print(f"  track is {feats.duration:.1f}s")

    sections = structure.find_sections(feats)
    print("\nStructure")
    for s in sections:
        print(f"  {s}")

    # 4. Excerpt ------------------------------------------------------------
    print("\nChoosing excerpt")
    if args.full:
        start, length = 0.0, feats.duration
        print(f"  whole track: {length:.1f}s ({length/60:.1f} min)")
    elif args.start is not None:
        start = min(args.start, max(0.0, feats.duration - 5.0))
        length = min(duration or MIN_DURATION, feats.duration - start)
        print(f"  using requested window {start:.1f}s -> {start + length:.1f}s")
    else:
        start, length = structure.pick_excerpt(
            feats, sections, target=duration,
            lo=MIN_DURATION, hi=MAX_DURATION,
        )

    name = slugify(args.name or asset.label)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_out = OUT_DIR / f"{name}.mp3"
    seq_out = OUT_DIR / f"{name}.xsq"

    print(f"\nWriting excerpt audio -> {audio_out.name}")
    try:
        sources.extract_excerpt(asset.path, audio_out, start, length)
    except sources.SourceError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    # 5. Arrange and write --------------------------------------------------
    # The analyser ran on the full track, so shift its event grids to be
    # relative to the excerpt before the arranger reads them.
    excerpt_feats = _rebase(feats, start, length)

    # Re-segment *inside* the chosen window rather than reusing the full-track
    # boundaries. Segmentation is relative: clustering a whole song puts its
    # boundaries where the biggest contrasts across the whole song are, and on a
    # track with a long gradual build that lumps several minutes into one span.
    # Re-run on the excerpt and the boundaries land where the contrasts within
    # *this* window are, which is what the sequence actually has to follow.
    print("\nStructure within the excerpt")
    clipped = structure.find_sections(
        excerpt_feats, verbose=False,
        # Boundaries from the excerpt, thresholds from the whole song, so a
        # drop stays a drop even in a window that is mostly loud.
        stats=structure.label_scale(feats, sections),
    )
    if not clipped:
        print("error: excerpt window contained no sections", file=sys.stderr)
        return 1
    for s in clipped:
        print(f"  {s}")

    builder = xsq.SequenceBuilder(
        frame_ms=args.frame_ms or xsq.FRAME_MS,
        duration_s=length,
        media_file=audio_out,
        song=asset.title,
        artist=asset.artist,
        author="triseq",
        comment=(f"Generated from {asset.source or asset.path.name} "
                 f"[{start:.1f}s-{start + length:.1f}s] at "
                 f"{feats.tempo:.1f} BPM. Structure: {arrange.summarize(clipped)}."),
    )

    print("\nArranging")
    arranger = arrange.Arranger(show, excerpt_feats, clipped, builder,
                                seed=args.seed, style=args.style)
    st = arranger.style
    how = "forced" if args.style else "auto"
    print(f"  style ({how}): {st.name}  pulse={feats.pulse_clarity:.2f} "
          f"onsets/beat={feats.onsets_per_beat:.2f}")
    print(f"    accents={st.accent_density:.2f} threshold={st.accent_threshold:.2f} "
          f"corridor={st.gesture_bars:.2f} bars/traversal")
    if args.corridor_rate:
        arranger.style = replace(arranger.style, gesture_bars=args.corridor_rate)
        print(f"  corridor: {args.corridor_rate:g} bars per traversal")
    arranger.arrange()

    stats = builder.write(seq_out)
    print(f"  {stats['effects']} effects across {stats['elements']} elements")
    print(f"  {stats['effect_defs']} distinct settings, {stats['palettes']} palettes")
    if stats["dropped"] or stats["truncated"] or stats["refitted"]:
        print(f"  overlaps: {stats['truncated']} truncated, "
              f"{stats['dropped']} dropped, {stats['refitted']} fades refitted")

    # 6. Validate -----------------------------------------------------------
    if not args.no_validate:
        problems = xsq.validate(seq_out, show.all_names)
        if problems:
            print(f"\nVALIDATION FAILED ({len(problems)} problems):", file=sys.stderr)
            for p in problems[:20]:
                print(f"  {p}", file=sys.stderr)
            if len(problems) > 20:
                print(f"  ... and {len(problems) - 20} more", file=sys.stderr)
            return 1
        print("  validation passed")

    if not args.keep_download and asset.path.parent == CACHE_DIR:
        asset.path.unlink(missing_ok=True)

    print(f"\nDone.\n  {seq_out}\n  {audio_out}")
    print("\nOpen the .xsq in xLights, then render (Ctrl+F3) to preview.")
    return 0


def _rebase(feats, start: float, length: float):
    """Copy the features with all event times relative to the excerpt."""
    import copy

    import numpy as np

    f = copy.copy(feats)
    end = start + length

    # The waveform itself, so anything downstream that re-derives features
    # (segmentation re-runs chroma and MFCC over f.y) sees the excerpt and not
    # the whole track.
    f.y = feats.y[int(start * feats.sr):int(end * feats.sr)]

    def window(times, *extra):
        times = np.asarray(times, dtype=float)
        mask = (times >= start) & (times < end)
        out = [times[mask] - start]
        out.extend(np.asarray(e, dtype=float)[mask] for e in extra)
        return out

    (f.beat_times,) = window(feats.beat_times)
    (f.bar_times,) = window(feats.bar_times)
    f.kick_times, f.kick_strength = window(feats.kick_times, feats.kick_strength)

    # Frame-rate curves keep their samples; only the time axis shifts.
    fmask = (feats.frame_times >= start) & (feats.frame_times < end)
    f.frame_times = feats.frame_times[fmask] - start
    f.onset_env = feats.onset_env[fmask]
    f.rms = feats.rms[fmask]
    f.centroid = feats.centroid[fmask]
    f.bands = {k: v[fmask] for k, v in feats.bands.items()}
    # Chroma is (12, frames) -- slice the frame axis, not the pitch axis, or
    # quality() reads the wrong window when it indexes via frame_times.
    if feats.chroma_frames is not None:
        f.chroma_frames = feats.chroma_frames[:, fmask]
    f.duration = length
    return f


if __name__ == "__main__":
    sys.exit(main())
