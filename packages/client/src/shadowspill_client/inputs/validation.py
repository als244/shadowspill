"""Validate every Datum and global limit before selecting any executable Plan."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from shadowspill_client import BatchConfig
from shadowspill_client._records import number
from shadowspill_client.protocol.definitions import Participation
from shadowspill_client.protocol.inputs import InputSchema
from shadowspill_client.protocol.payload import BatchSnapshot
from shadowspill_client.protocol.tensors import TensorPayload

from .values import ValidatedTensor, validate_values

type Path = tuple[str, ...]


def _flatten(
    value: Any, path: Path, paths: set[Path], index: int
) -> dict[Path, TensorPayload]:
    if isinstance(value, Mapping):
        if path in paths:
            raise ValueError(f"Datum {index}: expected a tensor at {path}")
        result = {}
        for name, child in value.items():
            child_path = (*path, name)
            if not any(p[: len(child_path)] == child_path for p in paths):
                raise ValueError(f"Datum {index}: unknown input field {child_path}")
            result.update(_flatten(child, child_path, paths, index))
        return result
    if path in paths and isinstance(value, TensorPayload):
        return {path: value}
    raise ValueError(f"Datum {index}: invalid field {path}")


def combined_schema(model: InputSchema, objective: InputSchema) -> InputSchema:
    return InputSchema(
        fields=[
            replace(
                field,
                path=(tree, *field.path),
                validity=None if field.validity is None else (tree, *field.validity),
            )
            for tree, schema in (("inputs", model), ("loss_inputs", objective))
            for field in schema.fields
        ],
        capacities=[
            replace(capacity, path=(tree, *capacity.path) if capacity.path else ())
            for tree, schema in (("inputs", model), ("loss_inputs", objective))
            for capacity in schema.capacities
        ],
    )


@dataclass(frozen=True)
class ValidatedBatch:
    owner: BatchSnapshot
    fields: tuple[Mapping[Path, ValidatedTensor], ...]
    dimensions: tuple[Mapping[str, int], ...]
    capacities: Mapping[str, int]
    schema: InputSchema
    config: BatchConfig

    def require_context(self, schema: InputSchema, config: BatchConfig) -> None:
        if self.owner.closed:
            raise RuntimeError("input snapshot is closed")
        if self.schema != schema or self.config.limits != config.limits:
            raise ValueError("batch was validated against a different input contract")

    def participating_positions(self, rule: Participation) -> int:
        if not rule.path:
            return len(self.fields)
        path = ("loss_inputs", *rule.path)
        if any(path not in values for values in self.fields):
            raise ValueError(f"missing participation field: {path}")
        return sum(values[path].count_valid() for values in self.fields)


def validate_batch(
    snapshot: BatchSnapshot,
    schema: InputSchema,
    batch: BatchConfig,
    *,
    required_paths: Sequence[Path] | None = None,
) -> ValidatedBatch:
    """Validate owned inputs without allocating another full batch of casts.

    Required paths describe the selected computation/output closure; omitted
    optional fields in that closure use declared defaults. Known unused fields
    may be supplied and are validated too. The snapshot must stay owned until
    all packing/transfer consumers finish. Validation uses bounded chunk scratch.
    """
    if snapshot.closed:
        raise RuntimeError("input snapshot is closed")
    definitions = {tuple(field.path): field for field in schema.fields}
    selected = set(definitions if required_paths is None else required_paths)
    if selected - definitions.keys():
        raise ValueError("required paths name unknown input fields")
    capacity_names = {capacity.name for capacity in schema.capacities}
    if batch.limits.keys() != capacity_names:
        raise ValueError(f"batch limits must name exactly {sorted(capacity_names)}")
    limits = {name: number(value, name) for name, value in batch.limits.items()}
    # Reject a record-count overflow before allocating per-Datum bookkeeping.
    for capacity in schema.capacities:
        if (
            capacity.reduction == "count"
            and len(snapshot.datums) > limits[capacity.name]
        ):
            raise ValueError(
                f"{capacity.name}: supplied {len(snapshot.datums)} "
                f"exceeds limit {batch.limits[capacity.name]}"
            )
    all_values, all_dimensions = [], []
    for index, datum in enumerate(snapshot.datums):
        paths = set(definitions)
        provided = _flatten(datum.inputs, ("inputs",), paths, index)
        provided.update(_flatten(datum.loss_inputs, ("loss_inputs",), paths, index))
        dimensions: dict[str, int] = {}
        for path, tensor in provided.items():
            field = definitions[path]
            if len(tensor.shape) != len(field.shape):
                raise ValueError(f"Datum {index} {path}: rank differs from schema")
            for actual, declared in zip(tensor.shape, field.shape, strict=True):
                if isinstance(declared, str):
                    if declared in dimensions and dimensions[declared] != actual:
                        raise ValueError(
                            f"Datum {index}: aligned dimension {declared!r} differs"
                        )
                    dimensions[declared] = actual
                elif actual != declared:
                    raise ValueError(
                        f"Datum {index} {path}: dimension {actual} "
                        f"differs from {declared}"
                    )
        values = {}
        for path, field in definitions.items():
            supplied = provided.get(path)
            if supplied is not None:
                values[path] = ValidatedTensor(
                    field.dtype, supplied.shape, source=supplied
                )
            elif path in selected:
                if field.required:
                    raise ValueError(f"Datum {index}: missing required field {path}")
                if (
                    missing := {d for d in field.shape if isinstance(d, str)}
                    - dimensions.keys()
                ):
                    raise ValueError(
                        f"Datum {index} {path}: cannot infer default dimensions "
                        f"{sorted(missing)}"
                    )
                shape = tuple(
                    dimensions[d] if isinstance(d, str) else d for d in field.shape
                )
                assert isinstance(field.default, (bool, int, float))
                values[path] = ValidatedTensor(
                    field.dtype, shape, default=field.default
                )
        # Validate masks before consumers, independently of field declaration
        # order. A mask applies only when the operation selects it: a target-only
        # forward must validate every target, even if unused loss masks arrive.
        for masked in (False, True):
            for path, value in values.items():
                field = definitions[path]
                if (field.validity is not None) != masked:
                    continue
                validity = (
                    values[tuple(field.validity)]
                    if field.validity is not None and tuple(field.validity) in selected
                    else None
                )
                label = f"Datum {index} {path}"
                if value.source is not None:
                    validate_values(value.source, field, label, validity=validity)
                elif validity is None or validity.count_valid():
                    scalar = np.asarray(value.default)
                    validate_values(
                        TensorPayload(scalar.dtype.name, (), scalar.tobytes()),
                        field,
                        label,
                    )
                if validity is not None:
                    assert isinstance(field.padding, (bool, int, float))
                    padding = np.asarray(field.padding)
                    validate_values(
                        TensorPayload(padding.dtype.name, (), padding.tobytes()),
                        replace(field, finite=True),
                        f"{label} padding",
                    )
                    values[path] = replace(
                        value, validity=validity, padding=field.padding
                    )
        all_values.append(MappingProxyType(values))
        all_dimensions.append(MappingProxyType(dimensions))
    totals = {}
    for capacity in schema.capacities:
        if capacity.reduction == "count":
            total = len(all_values)
        else:
            assert capacity.axis is not None
            sizes = [
                values[tuple(capacity.path)].shape[capacity.axis]
                for values in all_values
                if tuple(capacity.path) in values
            ]
            total = sum(sizes) if capacity.reduction == "sum" else max(sizes, default=0)
        if total > limits[capacity.name]:
            raise ValueError(
                f"{capacity.name}: supplied {total} "
                f"exceeds limit {batch.limits[capacity.name]}"
            )
        totals[capacity.name] = total
    return ValidatedBatch(
        snapshot,
        tuple(all_values),
        tuple(all_dimensions),
        MappingProxyType(totals),
        schema,
        batch,
    )
