"""Restore optimizer checkpoints without materializing planned CUDA state."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class OptimizerCheckpointTensor:
    """One checkpoint tensor paired with its existing optimizer-owned view."""

    name: str
    destination: torch.Tensor
    source: torch.Tensor


@dataclass(frozen=True, slots=True)
class OptimizerCheckpointRestore:
    """Prepared in-place optimizer restore and its tensor payloads."""

    initialized: bool
    tensors: tuple[OptimizerCheckpointTensor, ...]


def restore_optimizer_checkpoint_structure(
    named_parameters: Mapping[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    checkpoint: Mapping[str, object],
) -> OptimizerCheckpointRestore:
    """Restore public optimizer structure while retaining every tensor object.

    ``Optimizer.load_state_dict()`` casts checkpoint tensors to each parameter's
    device.  A planned model deliberately exposes host-only CUDA placeholders,
    so that generic behavior would manufacture unplanned device allocations.
    This adapter performs the same parameter-ID reconciliation but preserves
    the optimizer's admitted tensor objects and returns the bytes to copy into
    their runtime-owned storage.
    """

    if set(checkpoint) != {"state", "param_groups"}:
        raise RuntimeError("optimizer state_dict keys differ")
    saved_state = checkpoint["state"]
    saved_groups = checkpoint["param_groups"]
    if not isinstance(saved_state, Mapping) or not isinstance(
        saved_groups, Sequence
    ) or isinstance(saved_groups, str | bytes):
        raise TypeError("optimizer state and param_groups must be containers")
    groups: tuple[object, ...] = tuple(saved_groups)
    if len(groups) != len(optimizer.param_groups):
        raise RuntimeError("optimizer checkpoint parameter-group count differs")

    name_by_parameter = {
        id(parameter): name for name, parameter in named_parameters.items()
    }
    parameter_by_saved_id: dict[object, torch.Tensor] = {}
    restored_groups: list[dict[str, object]] = []
    tensors: list[OptimizerCheckpointTensor] = []
    for group_index, (current_group, saved_group_value) in enumerate(
        zip(optimizer.param_groups, groups, strict=True)
    ):
        if not isinstance(saved_group_value, Mapping):
            raise TypeError("optimizer parameter group must be a mapping")
        saved_group = dict(saved_group_value)
        saved_parameters = saved_group.pop("params", None)
        if not _is_sequence(saved_parameters):
            raise TypeError("optimizer parameter-group params must be a sequence")
        current_parameters = tuple(current_group.get("params", ()))
        if len(saved_parameters) != len(current_parameters):
            raise RuntimeError("optimizer checkpoint parameter inventory differs")
        for saved_id, parameter in zip(
            saved_parameters, current_parameters, strict=True
        ):
            previous = parameter_by_saved_id.setdefault(saved_id, parameter)
            if previous is not parameter:
                raise RuntimeError("optimizer checkpoint parameter ID is ambiguous")
        current_values = {
            key: value for key, value in current_group.items() if key != "params"
        }
        restored = _restore_value(
            current_values,
            saved_group,
            f"optimizer_group.{group_index}",
            tensors,
        )
        if not isinstance(restored, dict):
            raise AssertionError("optimizer group restore changed container type")
        restored_groups.append(restored)

    restored_state: list[tuple[torch.Tensor, object]] = []
    for saved_id, saved_value in saved_state.items():
        parameter = parameter_by_saved_id.get(saved_id)
        if parameter is None:
            raise RuntimeError(
                f"optimizer checkpoint state key {saved_id!r} is not a parameter"
            )
        parameter_name = name_by_parameter.get(id(parameter))
        if parameter_name is None:
            raise RuntimeError("optimizer checkpoint parameter has no model name")
        current_value = optimizer.state.get(parameter)
        restored_state.append(
            (
                parameter,
                _restore_value(
                    current_value,
                    saved_value,
                    f"optimizer.{parameter_name}",
                    tensors,
                ),
            )
        )

    optimizer.state.clear()
    for parameter, state in restored_state:
        optimizer.state[parameter] = state
    for current_group, restored in zip(
        optimizer.param_groups, restored_groups, strict=True
    ):
        parameters = current_group["params"]
        current_group.clear()
        current_group.update(restored)
        current_group["params"] = parameters
    return OptimizerCheckpointRestore(bool(saved_state), tuple(tensors))


def _restore_value(
    current: object,
    saved: object,
    path: str,
    tensors: list[OptimizerCheckpointTensor],
) -> object:
    if isinstance(saved, torch.Tensor):
        if not isinstance(current, torch.Tensor):
            raise RuntimeError(
                f"optimizer checkpoint tensor {path!r} has no existing storage"
            )
        if saved.shape != current.shape or saved.dtype != current.dtype:
            raise RuntimeError(
                f"optimizer checkpoint tensor {path!r} has incompatible geometry"
            )
        tensors.append(OptimizerCheckpointTensor(path, current, saved))
        return current
    if isinstance(saved, Mapping):
        current_mapping = current if isinstance(current, Mapping) else {}
        return {
            key: _restore_value(
                current_mapping.get(key),
                value,
                f"{path}.{key}",
                tensors,
            )
            for key, value in saved.items()
        }
    if isinstance(saved, list):
        current_list = current if isinstance(current, list) else []
        return [
            _restore_value(
                current_list[index] if index < len(current_list) else None,
                value,
                f"{path}.{index}",
                tensors,
            )
            for index, value in enumerate(saved)
        ]
    if isinstance(saved, tuple):
        current_tuple = current if isinstance(current, tuple) else ()
        return tuple(
            _restore_value(
                current_tuple[index] if index < len(current_tuple) else None,
                value,
                f"{path}.{index}",
                tensors,
            )
            for index, value in enumerate(saved)
        )
    return copy.deepcopy(saved)


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


__all__ = [
    "OptimizerCheckpointRestore",
    "OptimizerCheckpointTensor",
    "restore_optimizer_checkpoint_structure",
]
