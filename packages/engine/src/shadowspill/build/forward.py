"""Public forward builder: model call to a persistent unprofiled Program."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import replace
from typing import Any

import torch
from torch import nn

from shadowspill.artifacts.program import Program
from shadowspill.build.lowering.forward.program import lower_forward
from shadowspill.parallelism import Parallelism
from shadowspill.preparation import OperatorPreparation
from shadowspill.preparation.lifecycle import BuildPreparations
from shadowspill.pytorch.accelerator import resolve_device
from shadowspill.pytorch.chunking import ChunkPolicy, ChunkRegions
from shadowspill.pytorch.partition import PartitionSpec
from shadowspill.resources import Resource
from shadowspill.resources.description import validate_resources

from .capture import capture_forward_model
from .common import build_setup, compile_artifacts
from .finalize import finalize_tasks
from .parallelism import attach_parallelism, model_parallelism
from .resources import attach_resources


def build_forward(
    model: nn.Module,
    *,
    example_args: Sequence[Any] = (),
    example_kwargs: Mapping[str, Any] | None = None,
    forward_fn: Callable[..., Any] | None = None,
    preserve_buffers: Sequence[str] = (),
    partition: PartitionSpec = "auto",
    chunking: ChunkPolicy | ChunkRegions | None = None,
    parallelism: Parallelism | None = None,
    resources: Mapping[str, Resource] | None = None,
    operator_preparation: Sequence[OperatorPreparation] = (),
    rank_map: Mapping[int, int] | None = None,
    target: str | torch.device = "cuda",
) -> Program:
    """Build fixed implementations without profiling, pools or live training state.

    CPU and meta models are accepted. The CUDA backend finalizes kernels on
    one representative GPU. Planning the returned artifact is CPU-only.
    By default capture calls ``model(*args, **kwargs)``. ``forward_fn`` instead
    receives ``(model, *args, **kwargs)``, preserving registered parameter names.
    ``operator_preparation`` supplies explicit provider factories that capture
    selected binaries and restore them before measurement/runtime admission.
    ``preserve_buffers`` names buffers whose functional updates may contribute
    to outputs but must not be written back. Each invocation starts from the
    current registered values; temporary values use ordinary planned storage.
    """
    spec, parameters = model_parallelism(model, parallelism)
    validate_resources(resources or {}, spec)
    device = resolve_device(target)
    assert device.index is not None
    with closing(BuildPreparations(operator_preparation, device)) as preparations:
        captured = capture_forward_model(
            model,
            example_args,
            dict(example_kwargs or {}),
            forward_fn=forward_fn,
            partition=partition,
            device_index=device.index,
            preserve_buffers=preserve_buffers,
        )
        if chunking is not None:
            from .chunking import chunk_forward

            captured = chunk_forward(captured, chunking)
        compiled = compile_artifacts(captured.artifacts, device)
        program = lower_forward(
            captured,
            tuple(compiled[item.compatibility_digest] for item in captured.artifacts),
            setup=build_setup(device),
        )
        program = replace(
            program,
            bindings={**program.bindings, "compiled_device_index": device.index},
        )
        if spec.world_size > 1:
            _validate_distributed_mutations(program)
        program = attach_resources(
            attach_parallelism(program, spec, parameters), resources or {}
        )
        return finalize_tasks(
            program, device, rank_map=rank_map, preparations=preparations
        )


def _validate_distributed_mutations(program: Program) -> None:
    # Generic storage keeps buffers whole on each worker. Their mutations are
    # local, unlike parameter writes that would need an explicit synchronization
    # and ownership contract. Check aliases, including parameter/buffer ties.
    aliases = {obj.object_id: obj.alias_group_id for obj in program.objects}
    registrations = program.bindings["registrations"]
    buffers = {
        aliases[item["object_id"]] for item in registrations if not item["parameter"]
    } - {aliases[item["object_id"]] for item in registrations if item["parameter"]}
    if any(
        aliases[mutation.object_id] not in buffers
        for task in program.tasks
        for mutation in task.mutations
    ):
        raise NotImplementedError(
            "distributed forward may mutate only rank-local model buffers"
        )
