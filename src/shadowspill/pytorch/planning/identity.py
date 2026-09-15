"""The identity a step build has before any capture.

Everything the step archive keys on is known on the CPU model: the caller's
export bypass key, the model's structure, the inputs' signatures, the
optimizer's type, step code and hyperparameters, the request's own settings,
the machine the program is built for and the environment its tasks are
profiled under. The data ordering is the one fact that differs between the
programs one capture produces, so it enters the key last.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from shadowspill.pytorch.guards import capture_training_signatures
from shadowspill.pytorch.optimizer.artifacts import (
    code_identity,
    optimizer_step_identity,
    optimizer_type_name,
    optimizer_value_identity,
)
from shadowspill.pytorch.partition import PartitionSpec
from shadowspill.pytorch.profiling.metadata import training_profiling_metadata
from shadowspill.pytorch.runtime_adapter.runtime import PlanMemory
from shadowspill.schema import artifact_schema
from shadowspill.step import StepDataOrdering

_SCHEMA = artifact_schema("step_identity")


def machine_identity(memory: PlanMemory) -> dict[str, object]:
    """The facts of the machine that enter a step program."""

    return {
        "execution_budget_bytes": memory.execution_budget,
        "spill_budget_bytes": memory.spill_budget,
        "dynamic_scratch_reserve_bytes": memory.dynamic_scratch_reserve_bytes,
        "execution_capacity_bytes": memory.execution.capacity,
        "execution_physical_capacity_bytes": memory.execution.physical_capacity,
        "spill_capacity_bytes": memory.spill.capacity,
        "execution_device": memory.execution_device,
    }


def step_identity(
    model: nn.Module,
    *,
    objective: Callable[..., Any],
    build_optimizer: Callable[[Any], torch.optim.Optimizer],
    hyperparams: Sequence[str],
    example_inputs: Sequence[Sequence[Any]],
    partition: PartitionSpec,
    profiling_metadata: Sequence[object] | None,
    optimizer_ordering: str,
    allocation_probe_seeds: int,
    allocation_probe_repetitions: int,
    export_bypass_key: str,
    machine: Mapping[str, object],
    environment: Mapping[str, object],
) -> dict[str, object]:
    """What one capture is the same as, before it runs.

    The optimizer is built over the model's parameters to read its type,
    step code and hyperparameters; that is what the build does again over the
    pool's copy, and a factory that allocates state on construction would do
    so twice.
    """

    optimizer = build_optimizer(model.parameters())
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must return a torch.optim.Optimizer")
    signatures = capture_training_signatures(example_inputs)
    workloads = training_profiling_metadata(
        profiling_metadata, microbatch_count=len(example_inputs)
    )
    return {
        "schema": _SCHEMA,
        "export_bypass_key": export_bypass_key,
        "model": {
            "modules": [
                (name, f"{type(module).__module__}.{type(module).__qualname__}")
                for name, module in model.named_modules()
            ],
            "parameters": [
                (name, tuple(item.shape), str(item.dtype), tuple(item.stride()))
                for name, item in model.named_parameters()
            ],
            "buffers": [
                (name, tuple(item.shape), str(item.dtype), tuple(item.stride()))
                for name, item in model.named_buffers()
            ],
        },
        "objective": code_identity(objective),
        "inputs": [item.digest for item in signatures],
        "profiling_metadata": [item.digest for item in workloads],
        "optimizer": {
            "type": optimizer_type_name(optimizer),
            "step": optimizer_step_identity(optimizer),
            "groups": [
                optimizer_value_identity(
                    {key: value for key, value in group.items() if key != "params"}
                )
                for group in optimizer.param_groups
            ],
        },
        "hyperparams": list(hyperparams),
        "partition": partition if isinstance(partition, str) else repr(partition),
        "optimizer_ordering": optimizer_ordering,
        "allocation_probes": {
            "seeds": allocation_probe_seeds,
            "repetitions": allocation_probe_repetitions,
        },
        "machine": dict(machine),
        "environment": dict(environment),
    }


def step_key(identity: Mapping[str, object], data_ordering: StepDataOrdering) -> str:
    """The archive key of one ordering's program of a step identity."""

    encoded = json.dumps(
        {**identity, "data_ordering": data_ordering.to_dict()},
        sort_keys=True,
        separators=(",", ":"),
        default=_encode,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _encode(value: object) -> object:
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"step identity holds a value JSON cannot carry: {value!r}")


__all__ = ["machine_identity", "step_identity", "step_key"]
