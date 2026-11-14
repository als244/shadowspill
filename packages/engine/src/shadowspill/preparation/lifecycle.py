"""Task-owned provider manifests and executor-owned prepared bindings."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from threading import RLock
from typing import Any

import torch

from shadowspill.artifacts._values import digest, freeze, plain

from .description import OperatorPreparation


def _factory(
    description: OperatorPreparation, expected: str | None = None
) -> tuple[Any, str]:
    module_name, name = description.factory.split(":")
    module = import_module(module_name)
    if module.__file__ is None:
        raise ValueError("operator preparation factory has no source")
    source = sha256(Path(module.__file__).read_bytes()).hexdigest()
    if expected is not None and source != expected:
        raise ValueError("operator preparation factory source differs")
    function = getattr(module, name)
    if not callable(function):
        raise TypeError("operator preparation factory is not callable")
    return function, source


def _merge_blobs(destination: dict[str, bytes], blobs: Mapping[str, bytes]) -> None:
    for key, value in blobs.items():
        if not isinstance(value, bytes) or sha256(value).hexdigest() != key:
            raise ValueError("invalid prepared operator binary digest")
        if key in destination and destination[key] != value:
            raise ValueError("conflicting prepared operator binary")
        destination[key] = value


def _close_owner(owner: Any, device: torch.device) -> None:
    with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
        owner.close()


class BuildPreparations:
    """Open once before compilation; capture each existing task call once."""

    def __init__(
        self, descriptions: Sequence[OperatorPreparation], device: torch.device
    ):
        self.stack = ExitStack()
        self.builders: list[Any] = []
        self.providers: list[dict[str, Any]] = []
        self.tasks: dict[str, list[Any]] = {}
        self.blobs: dict[str, bytes] = {}
        self.closed = False
        self.device = device
        try:
            identities = set()
            for description in descriptions:
                if not isinstance(description, OperatorPreparation):
                    raise TypeError(
                        "operator_preparation requires OperatorPreparation descriptions"
                    )
                identity = digest(description.to_dict())
                if identity in identities:
                    raise ValueError("duplicate operator preparation description")
                identities.add(identity)
                factory, source = _factory(description)
                with (
                    torch.cuda.device(device)
                    if device.type == "cuda"
                    else nullcontext()
                ):
                    builder = factory(
                        phase="build", device=device, options=plain(description.options)
                    )
                if callable(getattr(builder, "close", None)):
                    self.stack.callback(_close_owner, builder, device)
                if not all(
                    callable(getattr(builder, name, None))
                    for name in ("capture", "export", "close")
                ):
                    raise TypeError(
                        "operator preparation builder has an incomplete lifecycle"
                    )
                self.builders.append(builder)
                self.providers.append({**description.to_dict(), "source": source})
        except BaseException:
            self.close()
            raise

    @contextmanager
    def capture(self, key: str) -> Iterator[None]:
        if self.closed:
            raise RuntimeError("operator preparation builders are closed")
        if key in self.tasks:
            raise ValueError("operator preparation captured the same task twice")
        with ExitStack() as scope:
            captured = [
                scope.enter_context(builder.capture()) for builder in self.builders
            ]
            yield
        manifests = []
        blobs: dict[str, bytes] = {}
        for builder, selected in zip(self.builders, captured, strict=True):
            manifest, data = builder.export(selected)
            manifests.append(plain(freeze(manifest)))
            _merge_blobs(blobs, data)
        _merge_blobs(self.blobs, blobs)
        self.tasks[key] = manifests

    def metadata(self) -> dict[str, Any]:
        return {"version": 1, "providers": self.providers, "tasks": self.tasks}

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            with self.stack:
                if self.builders and self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)


class LoadedPreparations:
    """Share provider owners across Programs; bind only requested executable keys.

    Open before fixed-layout Plan admission. The caller quiesces streams
    before close. Each returned activation is task-local; modules remain owned
    by this session and never move into a process's unscoped eager kernel cache.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.stack = ExitStack()
        self.loaders: dict[str, Any] = {}
        self.bindings: dict[tuple[str, str], tuple[Any, ...]] = {}
        self.closed = False
        self.lock = RLock()
        self.active = 0

    def add(self, program: Any, keys: Sequence[str]) -> None:
        with self.lock:
            if self.active:
                raise RuntimeError("cannot load operator preparations during execution")
            if not program.bindings.get("operator_preparation"):
                if self.closed:
                    raise RuntimeError("operator preparations are closed")
                return
            with (
                torch.cuda.device(self.device)
                if self.device.type == "cuda"
                else nullcontext()
            ):
                if self.device.type == "cuda":
                    # Device selection/properties need not establish a current
                    # primary context. Establish it without allocating a tensor
                    # before providers load their modules.
                    torch.cuda.synchronize(self.device)
                self._add(program, keys)

    def _add(self, program: Any, keys: Sequence[str]) -> None:
        if self.closed:
            raise RuntimeError("operator preparations are closed")
        metadata = plain(program.bindings.get("operator_preparation", {}))
        if not metadata:
            return
        if (
            set(metadata) != {"version", "providers", "tasks"}
            or metadata["version"] != 1
        ):
            raise ValueError("unsupported operator preparation metadata")
        program_identity = digest(metadata)
        for key in keys:
            if (program_identity, key) in self.bindings:
                continue
            manifests = metadata["tasks"].get(key)
            if not isinstance(manifests, list) or len(manifests) != len(
                metadata["providers"]
            ):
                raise ValueError(f"missing operator preparation for task {key}")
            bound = []
            for row, manifest in zip(metadata["providers"], manifests, strict=True):
                if not isinstance(row, dict) or set(row) != {
                    "factory",
                    "options",
                    "source",
                }:
                    raise ValueError("invalid operator preparation provider")
                description = OperatorPreparation.from_dict(
                    {"factory": row["factory"], "options": row["options"]}
                )
                identity = digest(row)
                if identity not in self.loaders:
                    factory, _ = _factory(description, row["source"])
                    loader = factory(
                        phase="load",
                        device=self.device,
                        options=plain(description.options),
                    )
                    if callable(getattr(loader, "close", None)):
                        self.stack.callback(_close_owner, loader, self.device)
                    if not all(
                        callable(getattr(loader, name, None))
                        for name in ("bind", "close")
                    ):
                        raise TypeError(
                            "operator preparation loader has an incomplete lifecycle"
                        )
                    self.loaders[identity] = loader
                binding = self.loaders[identity].bind(
                    manifest, program.operator_binaries.__getitem__
                )
                if not callable(getattr(binding, "activate", None)):
                    raise TypeError("prepared operator task has no activation")
                bound.append(binding)
            self.bindings[program_identity, key] = tuple(bound)

    def wrap(self, program: Any, key: str, function: Any) -> Any:
        metadata = program.bindings.get("operator_preparation")
        if not metadata:
            return function
        identity = digest(metadata), key
        if identity not in self.bindings:
            raise ValueError(f"operator preparation task was not loaded: {key}")
        bindings = self.bindings[identity]

        def call(*args: Any, **kwargs: Any) -> Any:
            with self.lock:
                if self.closed:
                    raise RuntimeError("operator preparations are closed")
                self.active += 1
            try:
                with ExitStack() as scope:
                    for binding in bindings:
                        scope.enter_context(binding.activate())
                    return function(*args, **kwargs)
            finally:
                with self.lock:
                    self.active -= 1

        return call

    def close(self) -> None:
        with self.lock:
            if self.active:
                raise RuntimeError("cannot close active operator preparations")
            if self.closed:
                return
            self.closed = True
            self.bindings.clear()
            try:
                self.stack.close()
            finally:
                self.loaders.clear()
