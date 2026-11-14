"""Select a plan renderer and encode its storyboard to an MP4."""

from __future__ import annotations

import json
from pathlib import Path

from ..encoding import CRF, video_stream
from .storyboard import state_at


def create_renderer(video, data, replay, *, width=3840, title=None):
    if video == "emission-placement":
        from .render_emission_placement import EmissionPlacementRenderer

        return EmissionPlacementRenderer(data, replay, width=width, title=title)
    if video == "execution":
        from .render_execution import ExecutionRenderer

        return ExecutionRenderer(data, replay, width=width, title=title)
    raise ValueError(f"unknown video: {video}")


def render_movie(
    renderer, story, output, *, fps=30, quality="best", motion_fps=15, window_tasks=8
):
    output = Path(output)
    width, height = renderer.width, round(renderer.width * 9 / 16)
    if motion_fps > fps:
        raise ValueError("motion-fps must not exceed fps")
    frames, records = 0, []
    with video_stream(
        output, width=width, height=height, fps=fps, quality=quality
    ) as stream:
        for number, scene in enumerate(story["scenes"]):
            count = max(1, round(scene["seconds"] * fps))
            moving = (
                scene["cursor_start_ns"] != scene["cursor_end_ns"]
                or "placed_start" in scene
            )
            samples = (
                min(count, max(2, round(scene["seconds"] * motion_fps)))
                if moving
                else 1
            )
            print(
                f"scene {number + 1}/{len(story['scenes'])}: "
                f"{scene['title']} ({scene['seconds']:g}s)",
                flush=True,
            )
            records.append({**scene, "first_frame": frames, "frame_count": count})
            for sample in range(samples):
                state = state_at(
                    scene,
                    sample / max(1, samples - 1),
                    renderer.data,
                    window_tasks=window_tasks,
                )
                rgba = renderer.draw(state)
                repeats = round((sample + 1) * count / samples) - round(
                    sample * count / samples
                )
                for _ in range(repeats):
                    stream.write(rgba)
                frames += repeats
            # A review still for every scene, alongside its exact state.
            still = output.parent / (output.stem + "_frames") / f"{number + 1:02d}.png"
            renderer.save(still)
    result = {
        **renderer.data.summary(),
        "storyboard": {**story, "scenes": records},
        "frames": frames,
        "fps": fps,
        "motion_fps": motion_fps,
        "window_tasks": window_tasks,
        "sweep_camera": "fixed_time_window",
        "seconds": frames / fps,
        "width": width,
        "height": height,
        "quality": quality,
        "crf": CRF[quality],
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result
