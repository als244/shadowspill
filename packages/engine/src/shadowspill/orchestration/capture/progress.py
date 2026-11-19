"""Bounded worker-written capture progress; never infer completion from a report."""

import os
from pathlib import Path

from shadowspill.orchestration.launch.publication import _open_regular
from shadowspill_client.protocol.captures import WINDOW_BYTES, NsightWindow
from shadowspill_client.protocol.codec import RecordCodec

_CODEC = RecordCodec({"nsys.window": NsightWindow})


def write_progress(path: Path, value: NsightWindow) -> None:
    """Replace one private node-daemon-assigned metadata file; retain at most two
    copies."""
    raw = _CODEC.encode(value, max_bytes=WINDOW_BYTES)
    temporary = path.with_suffix(".pending")
    with temporary.open("wb", buffering=0) as stream:
        part = memoryview(raw)
        while part:
            count = stream.write(part)
            if type(count) is not int or not 0 < count <= len(part):
                raise OSError("capture progress write made no progress")
            part = part[count:]
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_progress(path: Path) -> NsightWindow:
    with _open_regular(path) as stream:
        value = _CODEC.decode(stream.read(WINDOW_BYTES + 1), max_bytes=WINDOW_BYTES)
    if not isinstance(value, NsightWindow):
        raise ValueError("capture progress requires a window record")
    return value
