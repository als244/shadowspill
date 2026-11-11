"""Shared host phase journals, separate from device/runtime tracing."""

import json
import os
import socket
import sys
import time
from contextlib import contextmanager
from pathlib import Path


class Lifecycle:
    """Record host boundaries without synchronizing or querying an accelerator.

    One process owns each file. Wall timestamps align with external utilization
    samples; monotonic timestamps measure durations on this host. These are
    control-flow boundaries, not CUDA event timings.
    """

    def __init__(self, path, *, rank=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("a", buffering=1)
        self._identity = dict(pid=os.getpid(), hostname=socket.gethostname(), rank=rank)
        self._sequence = 0

    def event(self, name, *, boundary="point", **fields):
        self._sequence += 1
        record = dict(
            timestamp_ns=time.time_ns(),
            monotonic_ns=time.monotonic_ns(),
            **self._identity,
            sequence=self._sequence,
            name=name,
            boundary=boundary,
            **fields,
        )
        self._stream.write(json.dumps(record) + "\n")

    @contextmanager
    def phase(self, name):
        self.event(name, boundary="begin")
        try:
            yield
        except BaseException as error:
            self.event(name, boundary="end", error=type(error).__name__)
            raise
        else:
            self.event(name, boundary="end")

    @contextmanager
    def managed(self, name, manager):
        """Time entry and exit while preserving context-manager error handling."""
        with self.phase(name + ".open"):
            value = manager.__enter__()
        try:
            yield value
        except BaseException:
            with self.phase(name + ".close"):
                suppressed = manager.__exit__(*sys.exc_info())
            if not suppressed:
                raise
        else:
            with self.phase(name + ".close"):
                manager.__exit__(None, None, None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._stream.close()
