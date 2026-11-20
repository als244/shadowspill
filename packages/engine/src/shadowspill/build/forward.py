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
from shadowspill.operator_preparation import OperatorPreparation
from shadowspill.operator_preparation.lifecycle import BuildPreparations
from shadowspill.parallelism import Parallelism
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
    """Capture one forward call of a model as a Program.

    The call is captured, split into stages, and each task's code finalized
    on one representative GPU of the target. The model's parameters may be on
    the CPU, the meta device or the target; build does not change them, and
    the Program holds none of their values. Nothing is profiled, no pools are
    opened and no persistent state is initialized; planning the result needs
    no GPU.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    example_args : sequence, default ()
        Positional arguments of one representative call. Runs supply
        arguments of the same signature: tensor values may differ, but not
        structure, shapes, dtypes or other Python values.
    example_kwargs : mapping of str to object, optional
        Keyword arguments of that call.
    forward_fn : callable, optional
        Function to capture instead of ``model(*args, **kwargs)``, such as a
        validation loss, called as ``forward_fn(model, *args, **kwargs)``.
        Parameter names are kept. Only the build needs it.
    preserve_buffers : sequence of str, default ()
        Buffers whose in-place updates may feed the outputs but are not
        written back: each run starts from their current values.
        Parameter-backed buffers are refused.
    partition : str or PartitionPolicy, default "auto"
        Stage boundaries: ``"auto"``, one stage per block of a repeated
        module sequence, such as each layer, with what comes before and after
        as stages of their own; ``"whole"``, one stage; or a
        `PartitionPolicy`.
    chunking : ChunkPolicy or ChunkRegions, optional
        Execution regions and chunks within stages, which bound temporary
        memory. By default none.
    parallelism : Parallelism, optional
        Ownership across ranks. By default one rank.
    resources : mapping of str to Resource, optional
        Shared resources the model's operators use, by name.
    operator_preparation : sequence of OperatorPreparation, default ()
        Adapters that save the external kernels each task selects while it is
        finalized, and restore them before measurement and execution.
    rank_map : mapping of int to int, optional
        Launcher rank to logical rank, when ``resources`` need every worker
        during build.
    target : str or torch.device, default "cuda"
        The device to finalize code on. Measurement and execution use the
        same process-local device ordinal.

    Returns
    -------
    Program
        The forward Program. Its runs return the model's outputs in their
        original structure.

    Raises
    ------
    CaptureError
        If the call cannot be captured as a fixed graph, or the partition is
        invalid.
    CompilationError
        If a task's code cannot be finalized.
    NotImplementedError
        If a distributed call changes anything but rank-local model buffers.
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
