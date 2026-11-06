"""Scope Quack host shortcuts to the task whose choices they implement.

A warm eager launch recipe can bypass the autotuner entirely. Capture must
start with an empty dispatch cache; replay may reuse its own recipes, never
another task's or eager execution's. This does not clear tuning/binary caches.
"""

import hashlib
import importlib
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import Any

from ._scope import dispatch_caches

_SOURCES = {
    "quack.gemm_interface": (
        "6215031bf8a870d16a1be224ecad21042076d62096f4621f477d63371eab92d3"
    ),
    "quack.epilogue.frontend": (
        "1fc3c4a0914a23c04e69dfef03232f7efb85e03bbb8a7c6e515509d9582d6f4a"
    ),
}


# dict is only a virtual MutableMapping at runtime; typeshed models it as an
# actual subclass. ABC methods must precede dict's C methods in this adapter.
class _Cache(MutableMapping[Any, Any], dict[Any, Any]):  # type: ignore[misc,metaclass]
    """Keep dict compatibility with provider-owned outer cache wrappers."""

    def __init__(self, eager: MutableMapping[Any, Any]):
        self.eager = eager
        self.key = object()

    def _data(self) -> MutableMapping[Any, Any]:
        caches = dispatch_caches.get()
        return self.eager if caches is None else caches.setdefault(self.key, {})

    def __getitem__(self, key: Any) -> Any:
        return self._data()[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._data()[key] = value

    def __delitem__(self, key: Any) -> None:
        del self._data()[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data())

    def __len__(self) -> int:
        return len(self._data())

    def copy(self) -> dict[Any, Any]:
        return dict(self._data())


def install() -> None:
    modules: dict[str, Any] = {name: importlib.import_module(name) for name in _SOURCES}
    for name, expected in _SOURCES.items():
        if (
            hashlib.sha256(Path(modules[name].__file__).read_bytes()).hexdigest()
            != expected
        ):
            raise ValueError(f"unsupported Quack host dispatch contract: {name}")
    epilogue = modules["quack.epilogue.frontend"].EpiMod
    original = epilogue.__getattribute__
    if original is not object.__getattribute__:
        raise ValueError("unsupported Quack epilogue attribute lookup")
    interface = modules["quack.gemm_interface"]
    interface._gemm_iface_plan_cache = _Cache(interface._gemm_iface_plan_cache)

    def lookup(self: Any, name: str) -> Any:
        if name == "_call_cache" and (caches := dispatch_caches.get()) is not None:
            return caches.setdefault(self, {})
        return original(self, name)

    # Lookup composes with provider-installed cache descriptors and with
    # source-checked __call__ repairs, regardless of registration order.
    epilogue.__getattribute__ = lookup
