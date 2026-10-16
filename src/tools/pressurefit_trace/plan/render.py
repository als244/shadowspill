"""Select a plan renderer and encode its storyboard to an MP4."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

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
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ValueError("MP4 rendering requires ffmpeg on PATH")
    width, height = renderer.width, round(renderer.width * 9 / 16)
    if width % 2 or height % 2:
        raise ValueError("video dimensions must be even")
    if motion_fps > fps:
        raise ValueError("motion-fps must not exceed fps")
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        str({"preview": 26, "high": 18, "best": 14}[quality]),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    frames, records = 0, []
    log_path = output.with_suffix(".ffmpeg.log")
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
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
                        process.stdin.write(rgba)
                    frames += repeats
                # A review still for every scene, alongside its exact state.
                still = (
                    output.parent / (output.stem + "_frames") / f"{number + 1:02d}.png"
                )
                renderer.save(still)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f"ffmpeg failed; see {log_path}")
        except BaseException:
            process.kill()
            process.wait()
            raise
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
        "crf": {"preview": 26, "high": 18, "best": 14}[quality],
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result
