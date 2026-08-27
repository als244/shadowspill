"""Deterministic task-local values for isolated structural profiling."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import torch

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import (
    GraphArtifact,
    TaskInputProvenance,
    TaskInputRole,
)

REPRESENTATIVE_VALUE_POLICY = "shadowspill.task-values/v5"

_REFERENCE_ROLES = frozenset(
    {
        TaskInputRole.PARAMETER,
        TaskInputRole.BUFFER,
        TaskInputRole.CONSTANT,
        TaskInputRole.OPTIMIZER_STATE,
        TaskInputRole.OPTIMIZER_HYPERPARAMETER,
    }
)
_REQUIRED_REFERENCE_ROLES = _REFERENCE_ROLES - {
    TaskInputRole.OPTIMIZER_STATE,
}


@dataclass(frozen=True, slots=True)
class RepresentativeInputSummary:
    """Content-free provenance for one materialized task argument."""

    position: int
    role: TaskInputRole
    source: str | None
    value_policy: str
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    alias_group: int
    consumer_targets: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "position": self.position,
            "role": self.role.value,
            "source": self.source,
            "value_policy": self.value_policy,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "storage_offset": self.storage_offset,
            "alias_group": self.alias_group,
            "consumer_targets": list(self.consumer_targets),
        }

    @classmethod
    def from_dict(cls, value: object) -> RepresentativeInputSummary:
        if not isinstance(value, dict):
            raise ValueError("representative input summary must be an object")
        try:
            return cls(
                position=int(value["position"]),
                role=TaskInputRole(str(value["role"])),
                source=(None if value["source"] is None else str(value["source"])),
                value_policy=str(value["value_policy"]),
                dtype=str(value["dtype"]),
                shape=tuple(int(item) for item in value["shape"]),
                stride=tuple(int(item) for item in value["stride"]),
                storage_offset=int(value["storage_offset"]),
                alias_group=int(value["alias_group"]),
                consumer_targets=tuple(str(item) for item in value["consumer_targets"]),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "representative input summary has an invalid schema"
            ) from exc


@dataclass(frozen=True, slots=True)
class RepresentativeInputSet:
    """One independently materialized structural contract input set."""

    structural_contract_key: str
    policy_version: str
    probe_index: int
    arguments: tuple[object, ...]
    summaries: tuple[RepresentativeInputSummary, ...]


def materialize_representative_inputs(
    artifact: GraphArtifact,
    *,
    device_ordinal: int,
    probe_index: int = 0,
    allocation_check: Callable[[str], None] | None = None,
) -> RepresentativeInputSet:
    """Materialize exact state/user values and deterministic anonymous values."""

    _validate_representative_artifact(artifact)
    if probe_index < 0:
        raise ValueError("representative input probe index must be non-negative")
    positions_by_group = _alias_group_positions(artifact)
    arguments: list[object | None] = [None] * len(artifact.example_arguments)
    summaries: list[RepresentativeInputSummary | None] = [None] * len(arguments)
    for group, positions in sorted(positions_by_group.items()):
        for position, target, summary in _materialize_alias_group(
            artifact,
            group,
            positions,
            device_ordinal=device_ordinal,
            probe_index=probe_index,
            allocation_check=allocation_check,
        ):
            arguments[position] = target
            summaries[position] = summary
    if any(value is None for value in arguments) or any(
        value is None for value in summaries
    ):
        raise AssertionError("representative task input materialization is incomplete")
    return RepresentativeInputSet(
        structural_contract_key=artifact.compatibility_digest,
        policy_version=REPRESENTATIVE_VALUE_POLICY,
        probe_index=probe_index,
        arguments=tuple(value for value in arguments if value is not None),
        summaries=tuple(value for value in summaries if value is not None),
    )


def _validate_representative_artifact(artifact: GraphArtifact) -> None:
    if len(artifact.example_arguments) != len(artifact.input_provenance):
        raise CaptureError(
            "task input provenance differs from the compiled tensor contract"
        )
    if len(artifact.tensor_argument_alias_groups) != len(artifact.example_arguments):
        raise CaptureError("task input alias groups differ from argument arity")


def _alias_group_positions(artifact: GraphArtifact) -> dict[int, list[int]]:
    positions: dict[int, list[int]] = {}
    for position, group in enumerate(artifact.tensor_argument_alias_groups):
        positions.setdefault(group, []).append(position)
    return positions


def _materialize_alias_group(
    artifact: GraphArtifact,
    group: int,
    positions: list[int],
    *,
    device_ordinal: int,
    probe_index: int,
    allocation_check: Callable[[str], None] | None,
) -> tuple[tuple[int, torch.Tensor, RepresentativeInputSummary], ...]:
    examples = tuple(artifact.example_arguments[position] for position in positions)
    if any(not isinstance(value, torch.Tensor) for value in examples):
        raise CaptureError("compiled task tensor contract contains a static argument")
    tensors = tuple(value for value in examples if isinstance(value, torch.Tensor))
    target_device = _representative_device(tensors, device_ordinal)
    owner = torch.empty(
        max(_minimum_storage_bytes(value) for value in tensors),
        dtype=torch.uint8,
        device=target_device,
    )
    if allocation_check is not None:
        allocation_check(
            "materialize representative input "
            f"for contract {artifact.compatibility_digest}, alias group {group}"
        )
    values: list[tuple[int, torch.Tensor, RepresentativeInputSummary]] = []
    for position, example in zip(positions, tensors, strict=True):
        target = torch.empty(0, dtype=example.dtype, device=target_device).set_(
            owner.untyped_storage(),
            int(example.storage_offset()),
            tuple(example.shape),
            tuple(example.stride()),
        )
        provenance = artifact.input_provenance[position]
        policy = _populate_value(
            target,
            provenance,
            structural_contract_key=artifact.compatibility_digest,
            position=position,
            probe_index=probe_index,
        )
        target.requires_grad_(bool(example.requires_grad))
        values.append(
            (
                position,
                target,
                RepresentativeInputSummary(
                    position=position,
                    role=provenance.role,
                    source=provenance.source,
                    value_policy=policy,
                    dtype=str(example.dtype),
                    shape=tuple(example.shape),
                    stride=tuple(example.stride()),
                    storage_offset=int(example.storage_offset()),
                    alias_group=group,
                    consumer_targets=provenance.consumer_targets,
                ),
            )
        )
    return tuple(values)


def _representative_device(
    tensors: tuple[torch.Tensor, ...],
    device_ordinal: int,
) -> torch.device:
    device_types = {value.device.type for value in tensors}
    if len(device_types) != 1:
        raise CaptureError("one task input alias group spans multiple devices")
    device_type = next(iter(device_types))
    return (
        torch.device("cuda", device_ordinal)
        if device_type == "cuda"
        else torch.device(device_type)
    )


def _minimum_storage_bytes(value: torch.Tensor) -> int:
    if value.layout is not torch.strided:
        raise CaptureError("representative task inputs require strided tensors")
    if any(dimension == 0 for dimension in value.shape):
        elements = int(value.storage_offset())
    elif value.ndim == 0:
        elements = int(value.storage_offset()) + 1
    else:
        elements = (
            int(value.storage_offset())
            + 1
            + sum(
                (int(dimension) - 1) * int(stride)
                for dimension, stride in zip(value.shape, value.stride(), strict=True)
            )
        )
    return max(0, elements * value.element_size())


def _populate_value(
    target: torch.Tensor,
    provenance: TaskInputProvenance,
    *,
    structural_contract_key: str,
    position: int,
    probe_index: int,
) -> str:
    reference = provenance.representative_value
    if reference is not None:
        _validate_reference(
            reference,
            target,
            structural_contract_key=structural_contract_key,
            position=position,
            provenance=provenance,
        )
        target.copy_(reference, non_blocking=False)
        return (
            "authentic_control"
            if provenance.role is TaskInputRole.CONTROL
            else "authentic"
        )
    if provenance.role in _REQUIRED_REFERENCE_ROLES:
        raise _value_error(
            structural_contract_key,
            position,
            provenance,
            target,
            "authentic initialized value is unavailable",
        )
    if provenance.role is TaskInputRole.OPTIMIZER_STATE:
        target.zero_()
        return "lazy_optimizer_state_zero"
    if not target.is_floating_point() and not target.is_complex():
        raise _value_error(
            structural_contract_key,
            position,
            provenance,
            target,
            "integer/boolean inputs require an authentic producer-derived or "
            "caller-supplied value",
        )

    seed = _seed(structural_contract_key, position, probe_index)
    try:
        if target.is_floating_point():
            generator = torch.Generator(device=target.device).manual_seed(seed)
            target.normal_(mean=0.0, std=1.0, generator=generator)
            return "deterministic_normal_0_1"
        if target.is_complex():
            generator = torch.Generator(device=target.device).manual_seed(seed)
            torch.view_as_real(target).normal_(
                mean=0.0,
                std=1.0,
                generator=generator,
            )
            return "deterministic_complex_normal_0_1"
        raise AssertionError("floating/complex representative policy is incomplete")
    except BaseException as exc:
        raise _value_error(
            structural_contract_key,
            position,
            provenance,
            target,
            f"synthetic value construction failed: {exc}",
        ) from exc


def _validate_reference(
    reference: torch.Tensor,
    target: torch.Tensor,
    *,
    structural_contract_key: str,
    position: int,
    provenance: TaskInputProvenance,
) -> None:
    if (
        tuple(reference.shape) != tuple(target.shape)
        or tuple(reference.stride()) != tuple(target.stride())
        or reference.dtype != target.dtype
    ):
        raise _value_error(
            structural_contract_key,
            position,
            provenance,
            target,
            "authentic value geometry differs from the structural contract",
        )


def _value_error(
    structural_contract_key: str,
    position: int,
    provenance: TaskInputProvenance,
    target: torch.Tensor,
    detail: str,
) -> CaptureError:
    consumers = provenance.consumer_targets or ("<output-only>",)
    return CaptureError(
        "representative task input is invalid: "
        f"contract={structural_contract_key}, position={position}, "
        f"role={provenance.role.value}, dtype={target.dtype}, "
        f"consumers={consumers}: {detail}"
    )


def _seed(structural_contract_key: str, position: int, probe_index: int) -> int:
    encoded = (
        f"{REPRESENTATIVE_VALUE_POLICY}:{structural_contract_key}:"
        f"{position}:{probe_index}"
    ).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") & (
        (1 << 63) - 1
    )


