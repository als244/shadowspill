"""Version-pinned PyTorch 2.13 Export and AOTAutograd boundary."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from functorch.compile import make_boxed_func  # type: ignore[import-untyped]
from torch._functorch.aot_autograd import aot_function
from torch._functorch.partitioners import min_cut_rematerialization_partition
from torch.export.graph_signature import OutputKind
from torch.utils._pytree import tree_flatten

from .capture import (
    AotGraphPair,
    GraphArtifact,
    ObjectiveSchema,
    capture_objective_schema,
    normalize_objective_result,
)
from .contracts import CaptureError, ObjectiveError, ObjectiveResult


@dataclass(frozen=True, slots=True)
class ExportCapture:
    """One functional Export graph plus exact flattened example arguments."""

    exported_program: torch.export.ExportedProgram
    flat_inputs: tuple[object, ...]
    user_output_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TrainingObjectiveCapture:
    """Exported objective semantics before stage-local differentiation."""

    exported: ExportCapture
    capture_module: nn.Module
    objective_schema: ObjectiveSchema


@dataclass(frozen=True, slots=True)
class TrainingCapture(TrainingObjectiveCapture):
    """Objective export plus whole-graph AOT alternatives for oracle use."""

    save_pair: AotGraphPair
    recompute_pair: AotGraphPair


class _ObjectiveModule(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        objective: Callable[..., torch.Tensor | ObjectiveResult],
        schema: ObjectiveSchema,
    ) -> None:
        super().__init__()
        self.model = model
        self.objective = objective
        self.schema = schema

    def forward(self, *args: Any) -> tuple[torch.Tensor, ...]:
        loss, metrics = normalize_objective_result(
            self.objective(self.model, *args), require_grad=False
        )
        leaves, tree_spec = tree_flatten(metrics)
        if tree_spec != self.schema.metric_tree_spec:
            raise ObjectiveError("objective metric structure changed during capture")
        tensor_metrics = tuple(
            leaves[position].detach()
            for position in self.schema.tensor_metric_positions
        )
        return (loss, *tensor_metrics)


def _export(module: nn.Module, inputs: Sequence[Any]) -> ExportCapture:
    try:
        exported = torch.export.export(module, tuple(inputs), strict=True)
        exported = exported.run_decompositions({})
    except BaseException as exc:
        raise CaptureError(f"strict PyTorch export failed: {exc}") from exc
    flatten = getattr(exported, "_graph_module_flat_inputs", None)
    if not callable(flatten):
        raise CaptureError(
            "PyTorch 2.13 ExportedProgram flat-input adapter is unavailable"
        )
    flat_inputs = tuple(flatten(tuple(inputs), {}))
    user_outputs = tuple(
        index
        for index, spec in enumerate(exported.graph_signature.output_specs)
        if spec.kind == OutputKind.USER_OUTPUT
    )
    if not user_outputs:
        raise CaptureError("exported graph has no user output")
    return ExportCapture(
        exported_program=exported,
        flat_inputs=flat_inputs,
        user_output_indices=user_outputs,
    )


def capture_forward(module: nn.Module, inputs: Sequence[Any]) -> ExportCapture:
    """Strictly export an inference graph while keeping all state explicit."""

    return _export(module, inputs)


def capture_training(
    model: nn.Module,
    objective: Callable[..., torch.Tensor | ObjectiveResult],
    microbatch: Sequence[Any],
) -> TrainingCapture:
    """Capture objective plus save-all and min-cut recomputation graph pairs."""

    objective_capture = capture_training_objective(model, objective, microbatch)
    save_pair = _capture_pair(objective_capture.exported, recomputation=False)
    recompute_pair = _capture_pair(objective_capture.exported, recomputation=True)
    return TrainingCapture(
        exported=objective_capture.exported,
        capture_module=objective_capture.capture_module,
        objective_schema=objective_capture.objective_schema,
        save_pair=save_pair,
        recompute_pair=recompute_pair,
    )


def capture_training_objective(
    model: nn.Module,
    objective: Callable[..., torch.Tensor | ObjectiveResult],
    microbatch: Sequence[Any],
) -> TrainingObjectiveCapture:
    """Export an objective without constructing an unused whole-model VJP."""

    probe_loss, probe_metrics = normalize_objective_result(
        objective(model, *microbatch), require_grad=True
    )
    del probe_loss
    schema = capture_objective_schema(probe_metrics)
    capture_module = _ObjectiveModule(model, objective, schema)
    exported = _export(capture_module, microbatch)
    return TrainingObjectiveCapture(
        exported=exported,
        capture_module=capture_module,
        objective_schema=schema,
    )


def inference_artifact(capture: ExportCapture) -> GraphArtifact:
    """Create the structural task ABI for a functional Export graph."""

    return GraphArtifact.capture(
        kind="inference",
        graph_module=capture.exported_program.graph_module,
        example_inputs=capture.flat_inputs,
    )


def _capture_pair(capture: ExportCapture, *, recomputation: bool) -> AotGraphPair:
    eager_output = capture.exported_program.graph_module(*capture.flat_inputs)
    return capture_graph_pair(
        capture.exported_program.graph_module,
        capture.flat_inputs,
        original_output=eager_output,
        recomputation=recomputation,
        root_output_positions=(capture.user_output_indices[0],),
        specialize_unit_tangents=True,
    )


def capture_graph_pair(
    graph_module: torch.fx.GraphModule,
    inputs: Sequence[object],
    *,
    original_output: object,
    recomputation: bool,
    root_output_positions: tuple[int, ...] | None = None,
    specialize_unit_tangents: bool = False,
) -> AotGraphPair:
    """Differentiate one functional graph with a flat tensor/static ABI."""

    capture_inputs = tuple(
        value.detach().requires_grad_(value.requires_grad)
        if isinstance(value, torch.Tensor)
        else value
        for value in inputs
    )
    captured: dict[str, GraphArtifact] = {}

    def forward_compiler(
        graph_module: torch.fx.GraphModule, example_inputs: Sequence[object]
    ) -> Any:
        captured["forward"] = GraphArtifact.capture(
            kind="forward",
            graph_module=graph_module,
            example_inputs=tuple(example_inputs),
        )
        return make_boxed_func(graph_module.forward)

    def backward_compiler(
        graph_module: torch.fx.GraphModule, example_inputs: Sequence[object]
    ) -> Any:
        captured["backward"] = GraphArtifact.capture(
            kind="backward",
            graph_module=graph_module,
            example_inputs=tuple(example_inputs),
        )
        return make_boxed_func(graph_module.forward)

    try:
        aot: Any = aot_function
        if recomputation:
            compiled = aot(
                graph_module,
                fw_compiler=forward_compiler,
                bw_compiler=backward_compiler,
                partition_fn=min_cut_rematerialization_partition,
            )
        else:
            compiled = aot(
                graph_module,
                fw_compiler=forward_compiler,
                bw_compiler=backward_compiler,
            )
        outputs = compiled(*capture_inputs)
        output_values, _ = tree_flatten(outputs)
        if root_output_positions is None:
            roots = tuple(
                value
                for value in output_values
                if isinstance(value, torch.Tensor)
                and value.requires_grad
                and (value.is_floating_point() or value.is_complex())
            )
        else:
            roots = tuple(output_values[index] for index in root_output_positions)
        if not roots:
            raise CaptureError("training stage has no differentiable output")
        differentiable_inputs = tuple(
            value
            for value in capture_inputs
            if isinstance(value, torch.Tensor) and value.requires_grad
        )
        if not differentiable_inputs:
            raise CaptureError("training graph has no differentiable inputs or state")
        torch.autograd.grad(
            roots,
            differentiable_inputs,
            grad_outputs=tuple(torch.ones_like(root) for root in roots),
            allow_unused=True,
            materialize_grads=True,
        )
    except CaptureError:
        raise
    except BaseException as exc:
        mode = "recomputation" if recomputation else "save"
        raise CaptureError(f"AOTAutograd {mode} capture failed: {exc}") from exc
    if set(captured) != {"forward", "backward"}:
        raise CaptureError("AOTAutograd did not emit a complete graph pair")
    original_output_count = len(tree_flatten(original_output)[0])
    saved = max(0, captured["forward"].output_count - original_output_count)
    specialized_count = 0
    backward = captured["backward"]
    if specialize_unit_tangents:
        backward, specialized_count = _specialize_terminal_unit_tangents(
            backward, tuple(roots)
        )
    return AotGraphPair(
        forward=captured["forward"],
        backward=backward,
        recomputation=recomputation,
        saved_value_count=saved,
        specialized_unit_tangent_count=specialized_count,
    )


def _specialize_terminal_unit_tangents(
    backward: GraphArtifact, roots: tuple[torch.Tensor, ...]
) -> tuple[GraphArtifact, int]:
    """Replace terminal scalar cotangent inputs with device unit constants."""

    count = len(roots)
    if count == 0 or count > len(backward.example_arguments):
        raise CaptureError("terminal tangent specialization arity is invalid")
    tangents = backward.example_arguments[-count:]
    if any(
        not isinstance(tangent, torch.Tensor) or tangent.ndim != 0
        for tangent in tangents
    ):
        raise CaptureError("terminal objective cotangent must be a scalar tensor")
    if any(root.ndim != 0 for root in roots):
        raise CaptureError("terminal objective root must be a scalar tensor")

    graph_module = copy.deepcopy(backward.graph_module)
    placeholders = tuple(
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    )
    if len(placeholders) != len(backward.example_arguments):
        raise CaptureError("backward placeholder count changed before specialization")
    anchors = placeholders[:-count]
    if not anchors:
        # A device-relative scalar cannot be constructed without either a
        # tensor anchor or a backend-specific device literal. Preserve the
        # explicit tangent ABI for this degenerate graph.
        return backward, 0
    anchor = anchors[0]
    for placeholder, tangent in zip(
        placeholders[-count:], tangents, strict=True
    ):
        assert isinstance(tangent, torch.Tensor)
        with graph_module.graph.inserting_after(placeholder):
            unit = graph_module.graph.call_function(
                torch.ops.aten.new_ones.default,
                args=(anchor, []),
                kwargs={"dtype": tangent.dtype},
            )
        unit.meta = dict(placeholder.meta)
        placeholder.replace_all_uses_with(unit)
        graph_module.graph.erase_node(placeholder)
    graph_module.graph.lint()
    graph_module.recompile()
    return (
        GraphArtifact.capture(
            kind="backward",
            graph_module=graph_module,
            example_inputs=backward.example_arguments[:-count],
        ),
        count,
    )
