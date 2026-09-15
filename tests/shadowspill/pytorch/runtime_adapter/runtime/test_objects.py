"""Public runtime-object references: acquired on the runtime, released by handle."""

from __future__ import annotations

import ctypes
import threading
from types import SimpleNamespace

import pytest

from shadowspill.pytorch.runtime_adapter.runtime import Runtime
from shadowspill.pytorch.runtime_adapter.runtime import objects as objects_module
from shadowspill.pytorch.runtime_adapter.runtime.objects import (
    acquire_object_reference,
)


def test_runtime_object_reference_owns_and_releases_one_runtime_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acquiring needs the adapter's bound runtime; releasing needs only the
    handle, so the two halves are answered by different libraries."""

    class _RuntimeLibrary:
        def __init__(self) -> None:
            self.released: list[int] = []

        def shadowspill_object_handle_acquire(
            self, runtime_handle: int, object_id: int, output: object
        ) -> int:
            assert object_id == 41
            ctypes.cast(output, ctypes.POINTER(ctypes.c_size_t))[0] = 73
            return 0

        def shadowspill_object_handle_release(self, handle: int) -> int:
            self.released.append(handle)
            return 0

    neutral = _RuntimeLibrary()
    monkeypatch.setattr(objects_module, "runtime_library", lambda: neutral)
    runtime = Runtime.__new__(Runtime)
    runtime._lock = threading.RLock()
    runtime._closed = False
    runtime._unusable_reason = None
    runtime._installed = SimpleNamespace(library=object())
    runtime._runtime_handle = 0
    runtime._active_object_references = 0

    reference = acquire_object_reference(runtime, object_id=41, size_bytes=2048)

    assert reference.object_id == 41
    assert reference.size_bytes == 2048
    assert runtime._active_object_references == 1
    reference.close()
    reference.close()
    assert neutral.released == [73]
    assert runtime._active_object_references == 0
