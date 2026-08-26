"""Exact fixed-shape guards for public forward and training inputs."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils._pytree import TreeSpec, tree_flatten, treespec_dumps

from shadowspill.errors import InputGuardError, PlanningError

from .contracts import TensorSpec


@dataclass(frozen=True, slots=True)
class TensorGuard:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    layout: torch.layout
    requires_grad: bool
    storage_offset: int
    storage_nbytes: int
    alias_ordinal: int

    def validate(self, value: object, *, path: str) -> None:
        if not isinstance(value, torch.Tensor):
            raise InputGuardError(f"{path} must be a tensor")
        actual = (
            tuple(value.shape),
            tuple(value.stride()),
            value.dtype,
            value.layout,
            bool(value.requires_grad),
            int(value.storage_offset()),
            int(value.untyped_storage().nbytes()),
        )
        expected = (
            self.shape,
            self.stride,
            self.dtype,
            self.layout,
            self.requires_grad,
            self.storage_offset,
            self.storage_nbytes,
        )
        if actual != expected:
            raise InputGuardError(
                f"{path} geometry differs: expected shape={self.shape}, "
                f"stride={self.stride}, dtype={self.dtype}, "
                f"layout={self.layout}, requires_grad={self.requires_grad}; "
                f"got shape={actual[0]}, stride={actual[1]}, dtype={actual[2]}, "
                f"layout={actual[3]}, requires_grad={actual[4]}, "
                f"storage_offset={actual[5]}, storage_nbytes={actual[6]}"
            )
        if value.device.type not in {"cpu", "cuda"}:
            raise InputGuardError(f"{path} must be a CPU or CUDA tensor")

    def identity(self) -> dict[str, object]:
        return {
            "kind": "tensor",
            "shape": self.shape,
            "stride": self.stride,
            "dtype": str(self.dtype),
            "layout": str(self.layout),
            "requires_grad": self.requires_grad,
            "storage_offset": self.storage_offset,
            "storage_nbytes": self.storage_nbytes,
            "alias_ordinal": self.alias_ordinal,
        }


@dataclass(frozen=True, slots=True)
class StaticGuard:
    value: Any
    type_name: str

    def validate(self, value: object, *, path: str) -> None:
        if _qualified_type(value) != self.type_name or not _equal(value, self.value):
            raise InputGuardError(
                f"{path} static value differs: expected {self.value!r} "
                f"({self.type_name}), got {value!r} ({_qualified_type(value)})"
            )

    def identity(self) -> dict[str, object]:
        return {
            "kind": "static",
            "type": self.type_name,
            "repr": repr(self.value),
        }


LeafGuard = TensorGuard | StaticGuard


@dataclass(frozen=True, slots=True)
class InputSignature:
    """One immutable pytree structure and its position-specific leaf guards."""

    tree_spec: TreeSpec
    leaves: tuple[LeafGuard, ...]

    @property
    def digest(self) -> str:
        payload = {
            "tree": treespec_dumps(self.tree_spec),
            "leaves": [leaf.identity() for leaf in self.leaves],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def validate(self, values: object, *, position: int | None = None) -> None:
        leaves, tree_spec = tree_flatten(values)
        prefix = "inputs" if position is None else f"microbatch {position}"
        if tree_spec != self.tree_spec:
            raise InputGuardError(
                f"{prefix} structure differs from the captured template"
            )
        for index, (guard, value) in enumerate(zip(self.leaves, leaves, strict=True)):
            guard.validate(value, path=f"{prefix} leaf {index}")
        storage_ordinals: dict[tuple[str, int], int] = {}
        for index, (guard, value) in enumerate(zip(self.leaves, leaves, strict=True)):
            if not isinstance(guard, TensorGuard) or not isinstance(
                value, torch.Tensor
            ):
                continue
            storage = value.untyped_storage()
            key = (value.device.type, storage._cdata)
            ordinal = storage_ordinals.setdefault(key, len(storage_ordinals))
            if ordinal != guard.alias_ordinal:
                raise InputGuardError(
                    f"{prefix} leaf {index} storage alias relationship differs"
                )


def capture_input_signature(values: object) -> InputSignature:
    """Capture a complete fixed-shape tensor/static pytree signature."""

    leaves, tree_spec = tree_flatten(values)
    guards: list[LeafGuard] = []
    storage_ordinals: dict[tuple[str, int], int] = {}
    for index, value in enumerate(leaves):
        if isinstance(value, TensorSpec):
            alias_ordinal = len(storage_ordinals)
            storage_ordinals[("spec", index)] = alias_ordinal
            guards.append(
                TensorGuard(
                    shape=value.shape,
                    stride=value.resolved_stride,
                    dtype=value.dtype,
                    layout=value.layout,
                    requires_grad=value.requires_grad,
                    storage_offset=0,
                    storage_nbytes=value.storage_nbytes,
                    alias_ordinal=alias_ordinal,
                )
            )
            continue
        if isinstance(value, torch.Tensor):
            if value.layout is not torch.strided:
                raise PlanningError(
                    f"example input leaf {index} must use strided layout"
                )
            if value.device.type not in {"cpu", "meta"}:
                raise PlanningError(
                    f"example input leaf {index} must be CPU or meta before planning"
                )
            storage = value.untyped_storage()
            storage_key = (value.device.type, storage._cdata)
            alias_ordinal = storage_ordinals.setdefault(
                storage_key, len(storage_ordinals)
            )
            guards.append(
                TensorGuard(
                    shape=tuple(value.shape),
                    stride=tuple(value.stride()),
                    dtype=value.dtype,
                    layout=value.layout,
                    requires_grad=bool(value.requires_grad),
                    storage_offset=int(value.storage_offset()),
                    storage_nbytes=int(storage.nbytes()),
                    alias_ordinal=alias_ordinal,
                )
            )
            continue
        try:
            saved = copy.deepcopy(value)
        except BaseException as exc:
            raise PlanningError(
                f"example input leaf {index} cannot be preserved"
            ) from exc
        if not _equal(value, saved):
            raise PlanningError(
                f"example input leaf {index} has no stable equality contract"
            )
        guards.append(StaticGuard(saved, _qualified_type(saved)))
    return InputSignature(tree_spec=tree_spec, leaves=tuple(guards))


def capture_training_signatures(
    example_inputs: Sequence[Sequence[Any]],
) -> tuple[InputSignature, ...]:
    """Capture one signature per fixed gradient-accumulation position."""

    if not isinstance(example_inputs, (list, tuple)) or not example_inputs:
        raise PlanningError("example_inputs must be a non-empty outer sequence")
    signatures: list[InputSignature] = []
    for position, microbatch in enumerate(example_inputs):
        if not isinstance(microbatch, (list, tuple)):
            raise PlanningError(f"example_inputs[{position}] must be a list or tuple")
        signatures.append(capture_input_signature(microbatch))
    return tuple(signatures)


def validate_training_inputs(
    values: Sequence[Sequence[Any]], signatures: tuple[InputSignature, ...]
) -> None:
    """Guard the complete step before executing or mutating any task."""

    if not isinstance(values, (list, tuple)):
        raise InputGuardError("training inputs must be an outer list or tuple")
    if len(values) != len(signatures):
        raise InputGuardError(
            "gradient-accumulation count differs: expected "
            f"{len(signatures)}, got {len(values)}"
        )
    for position, (microbatch, signature) in enumerate(
        zip(values, signatures, strict=True)
    ):
        if not isinstance(microbatch, (list, tuple)):
            raise InputGuardError(f"microbatch {position} must be a list or tuple")
        signature.validate(microbatch, position=position)


def _qualified_type(value: object) -> str:
    type_ = type(value)
    return f"{type_.__module__}.{type_.__qualname__}"


def _equal(left: object, right: object) -> bool:
    try:
        result = left == right
    except BaseException:
        return False
    return isinstance(result, bool) and result
