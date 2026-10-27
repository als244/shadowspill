"""Shared, bounded RGBA-to-H.264 stream and encoder lifetime."""

import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

CRF = {"preview": 26, "high": 18, "best": 14}


@contextmanager
def video_stream(output, *, width, height, fps, quality):
    """Yield the encoder's input pipe and check completion on normal exit."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ValueError("MP4 rendering requires ffmpeg on PATH")
    if width % 2 or height % 2:
        raise ValueError("video dimensions must be even")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
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
        str(CRF[quality]),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    log_path = output.with_suffix(".ffmpeg.log")
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            yield process.stdin
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f"ffmpeg failed; see {log_path}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if not process.stdin.closed:
                process.stdin.close()
