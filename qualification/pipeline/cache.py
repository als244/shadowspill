"""Optional empty compiler caches, separate from retained phase artifacts."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def compiler_cache(
    environment: dict[str, str] | None = None,
    *,
    empty: bool = False,
    parent: Path | None = None,
    prefix: str = "shadowspill-",
) -> Iterator[dict[str, str] | None]:
    """Yield subprocess settings; remove only this scope's compiler caches."""

    if not empty:
        yield None if environment is None else dict(environment)
        return
    if parent is not None:
        parent = parent.expanduser().resolve()
        parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=prefix, dir=parent) as directory:
        result = dict(os.environ if environment is None else environment)
        result["TORCHINDUCTOR_CACHE_DIR"] = str(Path(directory) / "torchinductor")
        result["TRITON_CACHE_DIR"] = str(Path(directory) / "triton")
        yield result
