"""Explicit completed-gradient tasks for full-parameter observations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.optimizer.artifacts import (
    OptimizerCapture,
    OptimizerTask,
    OptimizerTensorBinding,
    OptimizerTensorRole,
)
from shadowspill.pytorch.optimizer.bindings import optimizer_input_provenance

from ._optimizer import UpdateLayout


@torch.library.custom_op(
    "shadowspill_training::sum_gradient_", mutates_args=("gradient",)
)
def sum_gradient_(gradient: torch.Tensor, group: str) -> None:
    result = torch.ops._c10d_functional.all_reduce_(gradient, "sum", group)
    torch.ops._c10d_functional.wait_tensor(result)


@sum_gradient_.register_fake
def _sum_gradient_fake(gradient: torch.Tensor, group: str) -> None:
    return None


def before_observations(
    captured: OptimizerCapture, layouts: Mapping[str, UpdateLayout]
) -> OptimizerCapture:
    """SUM once before metrics; subsequent updates only take their owned slice."""
    bindings = {item.name: item for item in captured.bindings}
    seen: set[str] = set()
    mutations: set[str] = set()
    tasks = []
    for task in captured.update_tasks:
        names = tuple(
            name
            for name in task.binding_names
            if bindings[name].role is OptimizerTensorRole.GRADIENT
            and name not in seen
            and layouts[name.removeprefix("gradient.")].gradient_group_name is not None
        )
        seen.update(names)
        if names:
            incoming = tuple(replace(bindings[name], mutable=True) for name in names)
            groups = tuple(
                layouts[name.removeprefix("gradient.")].gradient_group_name
                for name in names
            )

            def reduce(
                *values: torch.Tensor, groups: tuple[str | None, ...] = groups
            ) -> tuple[torch.Tensor, ...]:
                for value, group in zip(values, groups, strict=True):
                    assert group is not None
                    sum_gradient_(value, group)
                return values

            examples = tuple(item.tensor for item in incoming)
            with torch.no_grad():
                graph = make_fx(reduce, tracing_mode="fake")(*examples)
                artifact = GraphArtifact.capture(
                    kind="optimizer",
                    graph_module=graph,
                    example_inputs=examples,
                    input_provenance=optimizer_input_provenance(incoming, {}),
                )
            tasks.append(
                OptimizerTask(artifact, names, names, task.completion_stage_index)
            )
            mutations.update(names)
        tasks.append(task)
    return replace(
        captured,
        update_tasks=tuple(tasks),
        bindings=tuple(
            replace(item, mutable=True) if item.name in mutations else item
            for item in captured.bindings
        ),
        mutation_names=tuple(
            dict.fromkeys((*captured.mutation_names, *sorted(mutations)))
        ),
    )


def initialize_missing_gradients(
    captured: OptimizerCapture, layouts: Mapping[str, UpdateLayout]
) -> OptimizerCapture:
    """Missing local contributions are explicit zero outputs, not initial state.

    Every replica gets the corresponding task occurrence. A replica with an
    existing gradient returns its input unchanged; another creates zeros. The
    later SUM/reduce-scatter therefore has identical participation everywhere.
    """
    pending = {
        "gradient." + name
        for name, layout in layouts.items()
        if layout.needs_gradient_initialization
    }
    bindings = {item.name: item for item in captured.bindings}
    tasks = []
    for task in captured.update_tasks:
        for name in task.binding_names:
            if name not in pending:
                continue
            pending.remove(name)
            binding = bindings[name]
            layout = layouts[name.removeprefix("gradient.")]
            examples: tuple[torch.Tensor, ...]
            names: tuple[str, ...]
            if layout.local_gradient:
                examples = (binding.tensor,)
                names = (name,)

                def initialize(
                    *values: torch.Tensor, binding: OptimizerTensorBinding = binding
                ) -> tuple[torch.Tensor, ...]:
                    return values
            else:
                examples = ()
                names = ()

                def initialize(
                    *values: torch.Tensor, binding: OptimizerTensorBinding = binding
                ) -> tuple[torch.Tensor, ...]:
                    return (
                        torch.empty_strided(
                            binding.tensor.shape,
                            binding.tensor.stride(),
                            dtype=binding.tensor.dtype,
                            device=binding.tensor.device,
                        ).zero_(),
                    )

            mode = getattr(binding.tensor, "fake_mode", None)
            from contextlib import nullcontext

            with mode if mode is not None else nullcontext(), torch.no_grad():
                graph = make_fx(initialize, tracing_mode="fake")(*examples)
                artifact = GraphArtifact.capture(
                    kind="optimizer", graph_module=graph, example_inputs=examples
                )
            tasks.append(
                OptimizerTask(
                    artifact,
                    names,
                    (),
                    task.completion_stage_index,
                    output_names=(name,),
                )
            )
        tasks.append(task)
    if pending:
        raise ValueError(
            f"unused distributed gradient initialization: {sorted(pending)}"
        )
    return replace(captured, update_tasks=tuple(tasks))
