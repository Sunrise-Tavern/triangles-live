"""Where the track is, right now: quiet, cruising, building, hot.

The offline generator segments a whole song at once and labels the pieces.
Nothing here can do that -- there is no lookahead -- so structure becomes a
rolling judgement about the present, with three ideas doing the work:

* **Everything is relative.**  Loudness is measured against a 45-second
  baseline, brightness against its own average.  Absolute levels would track
  the DJ's gain knob, not the music.
* **Hysteresis, and a floor on how fast state can change.**  A state machine
  that flickers between cruising and hot at a boundary is worse than one that
  is slightly late, because every effect keyed to state flickers with it.
* **A drop is an event, not a level.**  Waiting for energy to cross a
  threshold puts the lights behind the room by however long the smoothing
  takes.  The kick that starts a drop is audible in one block, so a build
  ending in a strong low-end hit goes straight to hot on that hit.

The states are deliberately coarser than the offline vocabulary
(intro/verse/build/drop/break/outro).  Live, the distinctions that survive are
how much is going on and whether it is increasing -- an intro and a breakdown
look identical from inside the moment, and treating them the same is honest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .analysis import Features

#: No music at all -- between sets, or the feed is dead.  Deliberately its own
#: state rather than a very quiet ``QUIET``: a breakdown is part of a track and
#: should still be driven by its beat grid, while silence has no grid to be
#: driven by and nothing to react to.  Conflating them makes the rig twitch at
#: a free-running clock when the room has gone home.
SILENT = "silent"
QUIET = "quiet"
CRUISING = "cruising"
BUILDING = "building"
HOT = "hot"
STATES = (SILENT, QUIET, CRUISING, BUILDING, HOT)


@dataclass
class StateThresholds:
    """Every number the state machine turns on, in one place for the UI."""

    #: Energy below this is quiet; above the second value it is not.
    quiet_enter: float = 0.62
    quiet_leave: float = 0.78
    #: Energy above this is hot; below the second value it is not.
    hot_enter: float = 1.30
    hot_leave: float = 1.10
    #: Share of spectral energy in the high band that marks a build.  This is
    #: the "high-band sweep": measured on the test track, 0.57 through both
    #: builds against 0.17-0.33 everywhere else -- a cleaner separation than
    #: anything loudness-based, and immune to the DJ's gain.
    build_high_share: float = 0.40
    #: ...and this far above the track's own high-band share.  The absolute
    #: threshold was set on one track; a brighter track sits above it all the
    #: way through (measured: 0.48-0.72 for 86 s, every second of it called
    #: "building").  A sweep is a *rise*, so it is also measured against a
    #: slow average of the share, once that average has had time to fill.
    build_high_ratio: float = 1.3
    #: ...and, once building, the ratio it may fall to before the build is
    #: over.  Without this a track hovering at the ratio flaps
    #: building/cruising every dwell period (measured: 6 changes a minute).
    build_high_ratio_leave: float = 1.1
    #: ...sustained for this long.
    build_hold_s: float = 1.0
    #: A build that has not resolved after this long is not a build any more;
    #: it is the track.  Resolve it by energy, and let the high-band baseline
    #: (which has caught up by now) stop it re-entering.
    max_build_s: float = 20.0
    #: Fallback only: energy this high is hot even with no build before it,
    #: for when we join a track mid-drop.  Deliberately high -- the event rule
    #: below is the one that should normally fire.
    hot_energy: float = 1.75
    #: The lower bar: ``hot_enter`` sustained for this long.  A verse peaks
    #: at 1.77 for a moment and falls back; a track at full tilt sits at
    #: 1.2-1.6 against its own 45 s baseline for minutes, which the 1.75 gate
    #: never reaches -- measured, the loudest 60 s of a session stayed in
    #: cruising.  Duration separates the two where level cannot.
    hot_hold_s: float = 6.0
    #: A kick this far above normal, arriving out of a build, is the drop...
    drop_kick: float = 3.0
    #: ...if the energy has also moved back into the bass.
    drop_bass_share: float = 0.35
    #: ...or, relative to the track's own bass share, this much more.  Bass
    #: share is a property of the mix as much as of the moment: a track whose
    #: drop reads 0.19 bass against 0.10 elsewhere never clears 0.35.
    drop_bass_ratio: float = 1.5
    #: Time constant of the band-share baselines above.
    share_tau_s: float = 45.0
    #: Leaving hot needs the music to actually come down, measured against
    #: the drop's *own* level rather than the 45 s baseline.  The baseline
    #: keeps learning through the drop, so after a minute of it "loud" has
    #: become "normal" and ``energy`` drifts to 1.0 with nothing changing --
    #: measured on a real session, hot ended at 152 s on "energy fell" while
    #: the RMS was 0.038, as loud as anything in the preceding minute (0.026-
    #: 0.074), and the show went hot -> cruising -> building in one second of
    #: unchanged music.  So the fast level must also fall to this fraction of
    #: a slow average taken only while hot: the real ends of hot passages in
    #: that session read 0.34 (159-160 s: 0.045 -> 0.015) and 0.24 (252-253
    #: s); the false exit read 0.78.
    hot_fall: float = 0.6
    #: ...where "slow" is this long.  Long enough that one loud bar does not
    #: make the next ordinary one look like a fall (the same session peaks at
    #: 0.074 for a second inside a 0.03 plateau), short enough that a fade
    #: over a phrase or two still registers.
    hot_level_tau_s: float = 16.0
    #: Safety net for a fade with no step in it: below this fraction of the
    #: set's own peak the passage is not hot whatever the step rule says.
    #: Measured, the loudest 60 s of a session sits at 0.9-1.2 and its real
    #: breakdowns at 0.5-0.65.
    hot_floor_level: float = 0.35
    #: ...and the build actually lasted.
    min_build_s: float = 2.5
    #: Cold start.  Until the loudness baseline has filled, "loud" means
    #: nothing -- an engine started in the middle of a drop learns the drop
    #: as normal, reads energy 1.0 and calls it cruising until the next
    #: breakdown (measured: 26 s of a full-tilt section at 0.45 bass share
    #: with a kick on every beat, rendered as cruising).  Band shape does not
    #: need history: while the baseline is still filling, a passage that
    #: keeps the energy in the bass *and* keeps landing kicks for this long
    #: is hot.  The kick is required so a kickless sub-bass intro, which is
    #: all low end too, does not qualify.
    cold_hot_s: float = 3.0
    #: ...a kick at least ``drop_kick`` strong within this long counts as
    #: "landing kicks".  Just over a beat at 70 BPM.
    kick_recent_s: float = 1.0
    #: How much of the loudness baseline must be filled before the energy
    #: fallback is allowed to declare a drop.  Early in a track everything is
    #: "average", so the ratio spikes on the first loud passage whether or not
    #: it is a drop -- measured, a verse reached 1.77 while the track's actual
    #: second drop only reached 1.73.  The build-then-kick rule below needs no
    #: such history and stays live from the first bar.
    min_warm: float = 0.75
    #: Seconds of no signal before the show goes idle.  Long enough that the
    #: gap between two tracks, or a bar of dead air in a breakdown, does not
    #: drop the rig out of the show.
    silent_hold_s: float = 2.5
    #: Minimum time in a state before it may change again.
    dwell_s: float = 1.2
    #: Smoothing on energy before any of the above is applied.
    energy_tau_s: float = 1.2
    #: Window the slope is measured over.
    slope_tau_s: float = 4.0


@dataclass
class StateReport:
    state: str = CRUISING
    energy: float = 1.0
    slope: float = 0.0
    brightness: float = 1.0
    since_s: float = 0.0
    #: Bars elapsed in this state, if the bar line is known.
    bars: float = 0.0
    changed: bool = False
    #: What triggered the last change, for the readout.
    reason: str = ""

    def as_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


class StateMachine:
    def __init__(self, thresholds: StateThresholds | None = None,
                 block_s: float = 512 / 44100) -> None:
        self.t = thresholds or StateThresholds()
        self.block_s = block_s
        self.state = CRUISING
        self.entered_at = 0.0
        self.report = StateReport()

        self._energy = 1.0
        self._slow = 1.0
        self._bright = 1.0
        self._high = 0.0
        self._high_slow = 0.0
        self._bass_slow = 0.0
        self._share_count = 0
        self._rising_for = 0.0
        self._hot_for = 0.0
        self._quiet_for = 0.0
        #: Absolute smoothed loudness, and its slow average while hot.
        self._loud = 0.0
        self._hot_level = 0.0
        self._level = 1.0
        self._bass_fast = 0.0
        self._last_kick = -1e9
        self._cold_for = 0.0
        #: Hot was called by the cold-start rule, on band shape alone.
        self._provisional = False
        self._now = 0.0
        self._alpha = 1.0 - math.exp(-block_s / self.t.energy_tau_s)
        self._slow_alpha = 1.0 - math.exp(-block_s / self.t.slope_tau_s)
        self._bright_alpha = 1.0 - math.exp(-block_s / 8.0)
        self._share_alpha = 1.0 - math.exp(-block_s / self.t.share_tau_s)
        self._hot_alpha = 1.0 - math.exp(-block_s / self.t.hot_level_tau_s)
        self._share_warm = int(round(1.0 / self._share_alpha))
        self.history: list[tuple[float, str]] = field(default_factory=list)  # type: ignore
        self.history = []

    # -- the update -------------------------------------------------------- #

    def push(self, features: Features) -> StateReport:
        previous = self.state
        self._now = features.t

        self._energy += (features.energy - self._energy) * self._alpha
        self._slow += (features.energy - self._slow) * self._slow_alpha
        # Slope from the gap between a fast and a slow average, scaled to
        # per-second.  A least-squares fit over a window says the same thing
        # and costs a ring buffer; this needs two floats.
        slope = (self._energy - self._slow) / self.t.slope_tau_s
        self._bright += (features.centroid - self._bright) * self._bright_alpha
        brightness = features.centroid / max(self._bright, 1e-6)

        self._loud += (features.rms - self._loud) * self._alpha
        self._level = features.level
        self._bass_fast += (features.bass_share - self._bass_fast) * self._alpha
        if features.kick >= self.t.drop_kick and not features.silent:
            self._last_kick = features.t
        cold = (features.warm < self.t.min_warm and not features.silent
                and self._bass_fast >= self.t.drop_bass_share
                and features.t - self._last_kick <= self.t.kick_recent_s)
        self._cold_for = self._cold_for + self.block_s if cold else 0.0
        if self.state == HOT:
            self._hot_level += (features.rms - self._hot_level) * self._hot_alpha
        self._high += (features.high_share - self._high) * self._alpha
        if not features.silent:
            # Cumulative mean until the window fills, then an EMA -- the same
            # warm-up the loudness baseline uses, so the first bars are not
            # compared against an empty average.
            self._share_count += 1
            a = max(self._share_alpha, 1.0 / self._share_count)
            self._high_slow += (features.high_share - self._high_slow) * a
            self._bass_slow += (features.bass_share - self._bass_slow) * a
        ratio = (self.t.build_high_ratio_leave if self.state == BUILDING
                 else self.t.build_high_ratio)
        rising = self._high >= self.t.build_high_share and (
            self._share_count < self._share_warm
            or self._high >= ratio * self._high_slow)
        self._rising_for = self._rising_for + self.block_s if rising else 0.0
        self._hot_for = (self._hot_for + self.block_s
                         if self._energy > self.t.hot_enter else 0.0)

        reason = self._transition(features, slope, brightness)

        self.report = StateReport(
            state=self.state, energy=self._energy, slope=slope,
            brightness=brightness, since_s=self._now - self.entered_at,
            changed=self.state != previous, reason=reason,
        )
        if self.report.changed:
            self.history.append((self._now, self.state))
        return self.report

    def _transition(self, features: Features, slope: float,
                    brightness: float) -> str:
        """Each cue is used where it is actually discriminative.

        Loudness separates quiet from everything else cleanly and separates a
        verse from a drop badly -- on the test track a verse reads 1.52 and a
        drop 1.67, which no threshold can split.  High-band share separates a
        build from everything else cleanly and says nothing about loudness.
        And a drop is not a level at all: it is the moment a build resolves.
        So: loudness decides quiet, band shape decides building, and an event
        decides hot.
        """
        # Nothing is playing.  Checked before everything else and exempt from
        # the dwell timer: with no signal the relative measures are all
        # meaningless, and the broadband noise floor in particular reads as a
        # high-band sweep, which put the machine in `building` through twenty
        # seconds of silence.
        if features.silent:
            self._quiet_for += self.block_s
            if self._quiet_for >= self.t.silent_hold_s:
                return self._enter(SILENT, "no signal")
            return ""
        was_silent, self._quiet_for = self._quiet_for > 0.0, 0.0
        if self.state == SILENT:
            # Music is back.  Leave immediately -- waiting out a dwell timer
            # here means the first bars of a track play to a dark room.
            return self._enter(CRUISING, "signal returned")

        held = self._now - self.entered_at

        # The drop.  Checked first and exempt from the dwell timer: this is the
        # one moment where being a beat late is obvious to everyone in the room.
        bass_back = (features.bass_share >= self.t.drop_bass_share
                     or (self._share_count >= self._share_warm
                         and features.bass_share
                         >= self.t.drop_bass_ratio * self._bass_slow))
        if (self.state == BUILDING and features.kick >= self.t.drop_kick
                and bass_back
                and self._now - self.entered_at >= self.t.min_build_s):
            # Kick strength alone cannot do this: a build has kicks too, and
            # measured, its peaks (9.5-67) overlap the drop's (18-21)
            # completely.  What separates them is where the energy is -- a
            # build is 0.19-0.26 bass, a drop 0.44-0.63 -- plus the plain fact
            # that a build lasts bars, so anything resolving in under a couple
            # of seconds was not one.
            return self._enter(HOT, "kick out of a build")

        if held < self.t.dwell_s:
            return ""

        if self.state == HOT:
            # A *provisional* hot, called at cold start on band shape, may
            # still hand over to a build: no sweep led into it, so nothing is
            # decaying, and a sweep arriving now is the build the rule could
            # not have seen.  A real hot never goes to building (below).
            if self._provisional and self._rising_for >= self.t.build_hold_s:
                return self._enter(BUILDING, "high band swept up")
            if not self._fallen():
                # Nothing has changed in the music, whatever the baseline
                # says.  Every exit from hot -- to cruising *or* straight to
                # quiet -- waits here for the level to actually step down.
                return ""

        if self._energy < self.t.quiet_enter:
            return self._enter(QUIET, "energy below floor")
        if self.state == QUIET and self._energy < self.t.quiet_leave:
            return ""                       # hysteresis: needs a clear rise

        if self.state == HOT:
            # Only one way out of hot: the energy goes.  Notably *not* into
            # building -- the high band is still falling from the sweep that
            # led into the drop, so allowing it re-entered building about a
            # second after every drop and flapped.  Musically a build follows
            # a lull, never a drop.
            #
            if self._energy < self.t.hot_leave:
                return self._enter(CRUISING, "energy fell")
            return ""

        if self._cold_for >= self.t.cold_hot_s and self.state != BUILDING:
            reason = self._enter(HOT, "bass and kicks at cold start")
            self._provisional = True
            return reason

        if self.state == BUILDING and held >= self.t.max_build_s:
            return self._enter(
                HOT if self._energy > self.t.hot_leave else CRUISING,
                "build outlasted a build")
        if self._rising_for > 0.0:
            if self._rising_for >= self.t.build_hold_s and self.state != BUILDING:
                return self._enter(BUILDING, "high band swept up")
            return ""

        if self.state == BUILDING:
            # The sweep ended without a kick big enough to call a drop.  Not
            # every build resolves into one, and pretending otherwise is worse
            # than under-reacting.
            return self._enter(
                HOT if self._energy > self.t.hot_energy else CRUISING,
                "build ended")

        if features.warm >= self.t.min_warm and (
                self._energy > self.t.hot_energy
                or self._hot_for >= self.t.hot_hold_s):
            return self._enter(HOT, "energy above ceiling")
        if self.state == QUIET:
            return self._enter(CRUISING, "energy above floor")
        return ""

    def _fallen(self) -> bool:
        """Has the level come down from what hot has been running at?"""
        return (self._loud < self.t.hot_fall * self._hot_level
                or self._level < self.t.hot_floor_level)

    def _enter(self, state: str, reason: str) -> str:
        if state == self.state:
            return ""
        if state == HOT:
            # The reference for leaving is what hot sounds like, starting
            # from the level it was entered at.
            self._hot_level = self._loud
            self._provisional = False
        self.state = state
        self.entered_at = self._now
        return reason
