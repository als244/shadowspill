"""Replay one reduction as a chapter, with detailed cuts and accelerated passages."""

from dataclasses import dataclass

from .geometry import subtract_intervals
from .pacing import paced_steps


@dataclass(frozen=True)
class Timing:
    opening_cuts: int
    slow_seconds: float
    explain_seconds: float
    step_seconds: float
    bridge_seconds: float
    bridge_hold_seconds: float
    accelerated_seconds: float
    accelerated_hold_seconds: float
    seed_seconds: float
    final_seconds: float
    cut_zoom: bool
    zoom_seconds: float
    zoom_hold_seconds: float


class Playback:
    def __init__(self, recording, reduction, *, pacing, output):
        self.recording = recording
        self.renderer = recording.renderer
        self.data = self.renderer.data
        self.rid = reduction
        self.reduction = self.data.trace.reductions[reduction]
        self.total = len(self.reduction.cuts)
        self.walk = self.data.trace._walk(reduction)
        self.state = next(self.walk)
        self.pacing = pacing
        self.output = output

    @property
    def count(self):
        return self.state[0]

    def draw(self, **options):
        self.renderer.draw(self.rid, *self.state, **options)

    def save(self, suffix):
        self.renderer.save(
            self.output.with_name(f"{self.output.stem}_r{self.rid:02d}_{suffix}.png")
        )

    def advance(self, target, frames, *, flash=False):
        start = self.count
        touched = set()
        while self.count < target:
            alias = int(self.reduction.cuts[self.count, 4])
            touched.add(alias)
            if self.count == target - 1:
                before = self.data.intervals(alias, self.state[1][alias])
            self.state = next(self.walk)
        removed = subtract_intervals(
            before, self.data.intervals(alias, self.state[1][alias])
        )
        self.draw(
            event=self.count - 1,
            batch_start=start if self.count - start > 1 else None,
            removed=removed,
            touched=touched,
        )
        self.recording.emit(frames=frames, flash=flash)

    def flow(self, target, seconds, hold_seconds):
        steps = (
            [(i, 1) for i in range(self.count + 1, target + 1)]
            if self.pacing == "every-cut"
            else paced_steps(
                self.count,
                target,
                max(1, round(seconds * self.recording.fps)),
                hold_frames=max(1, round(hold_seconds * self.recording.fps)),
            )
        )
        for index, (target_count, frames) in enumerate(steps):
            self.advance(target_count, frames)
            if index % 100 == 0:
                print(
                    f"RENDER reduction={self.rid} cut={target_count}/{self.total} "
                    f"frames={self.recording.frame_count}",
                    flush=True,
                )

    def zoom(self, seconds, *, returning=False, **options):
        frames = max(1, round(seconds * self.recording.fps))
        for frame in range(1, frames + 1):
            fraction = 1 - frame / frames if returning else frame / frames
            self.draw(
                **options,
                journey=True,
                zoom_progress=fraction * fraction * (3 - 2 * fraction),
            )
            self.recording.emit(frames=1)

    def show_cut(self, event, *, explanation, timing):
        alias = int(self.reduction.cuts[event, 4])
        before = self.data.intervals(alias, self.state[1][alias])
        journey = bool(
            explanation
            and timing.cut_zoom
            and self.renderer.focus_mode == "cut"
            and self.data.next_use(self.reduction.cuts[event]) is not None
        )
        selected = dict(
            event=event, phase="selected", journey=False if journey else None
        )
        self.draw(**selected, explain=explanation)
        if explanation:
            self.recording.emit(timing.explain_seconds)
            self.draw(**selected)
        self.recording.emit(timing.slow_seconds * 0.4)
        if journey:
            self.zoom(timing.zoom_seconds, event=event, phase="selected")
            self.recording.emit(timing.zoom_hold_seconds / 2)
        self.state = next(self.walk)
        removed = subtract_intervals(
            before, self.data.intervals(alias, self.state[1][alias])
        )
        self.draw(
            event=event,
            removed=removed,
            touched=[alias],
            journey=True if journey else None,
        )
        self.recording.emit(timing.slow_seconds * 0.6, flash=True)
        if journey:
            self.zoom(timing.zoom_seconds, returning=True, event=event, removed=removed)
            self.draw(event=event, removed=removed, journey=False)
            self.recording.emit(timing.zoom_hold_seconds / 2)
        if event == 0:
            self.save("cut00001")

    def chapter(self, timing, *, details, opening):
        self.draw(event=0 if self.total else None, phase="seed")
        self.save("seed")
        self.recording.emit(timing.seed_seconds)
        slow_events = details or list(
            range(
                min(
                    self.total,
                    timing.opening_cuts if opening else min(1, timing.opening_cuts),
                )
            )
        )
        for event in slow_events:
            if event - self.count <= max(timing.opening_cuts, 24):
                for target in range(self.count + 1, event + 1):
                    self.advance(
                        target,
                        max(1, round(timing.step_seconds * self.recording.fps)),
                        flash=True,
                    )
            else:
                self.flow(event, timing.bridge_seconds, timing.bridge_hold_seconds)
            self.show_cut(event, explanation=event in details, timing=timing)
        if opening:
            for target in range(
                self.count + 1, min(self.total, timing.opening_cuts) + 1
            ):
                self.advance(
                    target,
                    max(1, round(timing.step_seconds * self.recording.fps)),
                    flash=True,
                )
        self.flow(
            self.total, timing.accelerated_seconds, timing.accelerated_hold_seconds
        )
        self.data.assert_required_covered(self.state[1])
        self.draw(event=self.count - 1 if self.count else None, phase="final")
        self.save("final")
        self.recording.emit(timing.final_seconds)
        print(
            f"RENDER reduction={self.rid} complete cuts={self.count} "
            f"frames={self.recording.frame_count}",
            flush=True,
        )
