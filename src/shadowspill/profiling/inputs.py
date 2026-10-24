"""Bounded, alias-preserving task inputs, with explicit caller overrides."""

from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch

from shadowspill.artifacts.program import Program
from shadowspill.errors import CaptureError


def prepare_inputs(
    program: Program,
    key: str,
    device: torch.device,
    *,
    supplied: Sequence[Any] | None = None,
    slots: Sequence[Mapping[str, Any]] = (),
    values: Mapping[str, Any] | None = None,
) -> tuple[Any, ...]:
    description = program.bindings["task_inputs"][key]
    if supplied is not None:
        if len(supplied) != description["count"]:
            raise ValueError(f"{key}: supplied input count differs from the task")
        return tuple(supplied)
    arguments: list[Any] = [None] * description["count"]
    for position, value in description["static_values"]:
        arguments[position] = value
    for slot in slots:
        if values is not None and slot["object_id"] in values:
            arguments[slot["leaf_index"]] = values[slot["object_id"]]
    geometries = dict(
        zip(
            description["tensor_positions"], description["tensor_geometry"], strict=True
        )
    )
    groups: dict[int, list[int]] = {}
    for position, group in zip(
        description["tensor_positions"], description["tensor_aliases"], strict=True
    ):
        groups.setdefault(group, []).append(position)
    provenance = description.get("provenance", ())
    for positions in groups.values():
        missing = [position for position in positions if arguments[position] is None]
        if not missing:
            continue
        existing = [
            arguments[position]
            for position in positions
            if arguments[position] is not None
        ]
        geometry = geometries[positions[0]]
        target = device if geometry["device_type"] == "cuda" else torch.device("cpu")
        capacity = max(_extent(geometries[position]) for position in positions)
        storage = (
            existing[0].untyped_storage()
            if existing
            else torch.zeros(
                capacity, dtype=torch.uint8, device=target
            ).untyped_storage()
        )
        for position in missing:
            geometry = geometries[position]
            value = torch.empty(0, dtype=_dtype(geometry), device=target).set_(
                storage,
                geometry["storage_offset"],
                geometry["shape"],
                geometry["stride"],
            )
            source = provenance[position] if provenance else {}
            if not existing:
                _initialize(value, source)
            arguments[position] = value
    return tuple(arguments)


def _dtype(geometry: Mapping[str, Any]) -> torch.dtype:
    return cast(torch.dtype, getattr(torch, geometry["dtype"].removeprefix("torch.")))


def _extent(geometry: Mapping[str, Any]) -> int:
    span = (
        0
        if 0 in geometry["shape"]
        else 1
        + sum(
            (size - 1) * stride
            for size, stride in zip(geometry["shape"], geometry["stride"], strict=True)
        )
    )
    return int((geometry["storage_offset"] + span) * _dtype(geometry).itemsize)


def _initialize(value: torch.Tensor, source: Mapping[str, Any]) -> None:
    if "value" in source:
        value.copy_(
            torch.as_tensor(source["value"], dtype=value.dtype, device=value.device)
        )
    elif source.get("role") in {"control", "constant", "optimizer_hyperparameter"}:
        raise CaptureError(
            f"task input {source.get('source')} needs caller-supplied values"
        )
    elif value.dtype in {torch.float32, torch.float64, torch.float16, torch.bfloat16}:
        if source.get("role") in {"optimizer_state", "gradient"}:
            value.zero_()
        else:
            value.uniform_(
                -0.02,
                0.02,
                generator=torch.Generator(device=value.device).manual_seed(0),
            )
    else:
        value.zero_()
