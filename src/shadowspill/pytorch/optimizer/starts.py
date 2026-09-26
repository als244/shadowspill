"""What each optimizer-state entry starts at, read from how the optimizer makes it.

Discovery runs an optimizer's first step on storage-free parameters, so the
state it creates has a shape and a dtype but no values. Where the values would
have started is still in the step: every entry is made by some operation before
the update first writes to it, and that operation says -- a constant, or a copy
of its parameter at the entry's own precision, as a higher-precision master copy
is. `StartRecorder` watches the step and keeps, for every tensor it makes, what
that tensor starts as; `start_of` answers for one tensor once the step is done.

An entry made from anything else -- the gradient, say -- has no value before the
first step, and says so rather than being given one.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode


@dataclass(frozen=True, slots=True)
class ConstantStart:
    """Every element starts at one value."""

    value: bool | int | float


@dataclass(frozen=True, slots=True)
class ParameterStart:
    """A copy of the parameter, at the entry's own dtype."""


@dataclass(frozen=True, slots=True)
class ValueStart:
    """A value the step built on the host from data, kept exactly."""

    value: torch.Tensor


@dataclass(frozen=True, slots=True)
class HeldStart:
    """The value the optimizer already holds for the entry, as after loading
    a checkpoint into it."""


@dataclass(frozen=True, slots=True)
class NoStart:
    """No value before the first step, and why."""

    reason: str


StateStart = ConstantStart | ParameterStart | ValueStart | HeldStart | NoStart

#: An allocation nothing has written yet: its first write says what it starts at.
_UNWRITTEN = NoStart("allocated and never written")
_GRADIENT = NoStart("made from the gradient")

_CONSTANTS = {
    "zeros": 0,
    "zeros_like": 0,
    "new_zeros": 0,
    "ones": 1,
    "ones_like": 1,
    "new_ones": 1,
}
_FILLED = {"full", "full_like", "new_full", "scalar_tensor"}
_ALLOCATIONS = {
    "empty",
    "empty_like",
    "new_empty",
    "empty_strided",
    "new_empty_strided",
}
_COPIES = {"_to_copy", "clone", "_to_dtype", "_to_copy_dtype", "contiguous"}
_LIFTS = {"lift_fresh", "lift_fresh_copy"}


# torch ships the mode class without annotations, hence the two ignores.
class StartRecorder(TorchDispatchMode):  # type: ignore[no-untyped-call]
    """Keeps what every tensor an optimizer step makes starts as.

    Seeded with the parameters and their gradients, which is what an entry can
    be a copy of. A view starts as what it views; a copy as what it copies; an
    allocation that nothing has written as its first write. Every other
    operation's result has no start of its own, which only matters if the
    optimizer keeps it as state.
    """

    def __init__(
        self,
        parameters: Iterable[torch.Tensor],
        gradients: Iterable[torch.Tensor],
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self._starts: dict[int, tuple[weakref.ref[torch.Tensor], StateStart]] = {}
        for parameter in parameters:
            self._set(parameter, ParameterStart())
        for gradient in gradients:
            self._set(gradient, _GRADIENT)

    def start_of(self, tensor: torch.Tensor) -> StateStart:
        start = self._get(tensor)
        return NoStart("made before the step") if start is None else start

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        output = func(*args, **kwargs)
        name = func.overloadpacket.__name__
        if _writes_its_first_argument(func):
            target = args[0] if args else None
            if isinstance(target, torch.Tensor) and self._get(target) is _UNWRITTEN:
                self._set(target, self._first_write(name, args))
            return output
        if isinstance(output, torch.Tensor):
            self._set(output, self._made(func, name, args, kwargs, output))
        return output

    def _made(
        self,
        func: Any,
        name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: torch.Tensor,
    ) -> StateStart:
        # A lift hands over a tensor built from data outside the dispatcher;
        # its schema calls it an alias, but of nothing the step made.
        if name in _LIFTS:
            if _storage_free(output):
                return NoStart(f"made by {name} with no value")
            return ValueStart(output.detach().clone())
        if getattr(func, "is_view", False) or name in {"detach", "alias"}:
            return self._source(args[0] if args else None)
        if name in _CONSTANTS:
            return ConstantStart(_CONSTANTS[name])
        if name in _FILLED:
            value = kwargs.get("fill_value", args[-1] if args else None)
            if name == "scalar_tensor":
                value = args[0] if args else kwargs.get("s")
            if isinstance(value, (bool, int, float)):
                return ConstantStart(value)
            return NoStart(f"made by {name} from a value that is not a number")
        if name in _ALLOCATIONS:
            return _UNWRITTEN
        if name in _COPIES:
            return self._source(args[0] if args else None)
        if name == "copy":
            return self._source(args[1] if len(args) > 1 else None)
        return NoStart(f"made by {name}")

    def _first_write(self, name: str, args: tuple[Any, ...]) -> StateStart:
        if name == "zero_":
            return ConstantStart(0)
        if name == "fill_" and len(args) > 1:
            value = args[1]
            if isinstance(value, (bool, int, float)):
                return ConstantStart(value)
            return self._source(value)
        if name == "copy_" and len(args) > 1:
            return self._source(args[1])
        return NoStart(f"first written by {name}")

    def _source(self, value: object) -> StateStart:
        if not isinstance(value, torch.Tensor):
            return NoStart("made from something that is not a tensor")
        start = self._get(value)
        if start is None:
            return NoStart("made from a tensor the step did not make")
        return start

    def _set(self, tensor: torch.Tensor, start: StateStart) -> None:
        self._starts[id(tensor)] = (weakref.ref(tensor), start)

    def _get(self, tensor: torch.Tensor) -> StateStart | None:
        entry = self._starts.get(id(tensor))
        if entry is None or entry[0]() is not tensor:
            return None
        return entry[1]


def _writes_its_first_argument(func: Any) -> bool:
    arguments = func._schema.arguments
    if not arguments:
        return False
    alias = arguments[0].alias_info
    return alias is not None and alias.is_write


def _storage_free(tensor: torch.Tensor) -> bool:
    return tensor.is_meta or type(tensor).__name__ == "FakeTensor"


__all__ = [
    "ConstantStart",
    "HeldStart",
    "NoStart",
    "ParameterStart",
    "StartRecorder",
    "StateStart",
    "ValueStart",
]
