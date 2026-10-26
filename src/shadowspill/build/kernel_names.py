"""Bound generated symbol names so Triton and Inductor agree on cache filenames."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from hashlib import sha256
from threading import RLock
from typing import Any
from unittest.mock import patch

from torch._inductor.codegen import triton

_lock = RLock()


@contextmanager
def portable_kernel_names() -> Iterator[None]:
    # Triton truncates filenames at 150 characters; the static Inductor loader
    # expects the complete symbol. Leave room for its prefix and numeric suffix.
    # Source/module metadata remains complete in the generated wrapper and IR.
    with _lock:
        original: Callable[..., str] = getattr(triton, "get_fused_kernel_name")

        def bounded(*args: Any, **kwargs: Any) -> str:
            name = original(*args, **kwargs)
            if len(name) <= 100:
                return name
            return name[:88] + "_" + sha256(name.encode()).hexdigest()[:10]

        with patch.object(triton, "get_fused_kernel_name", bounded):
            yield
