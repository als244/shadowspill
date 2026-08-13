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
from torch.export.graph_signature import ExportGraphSignature, InputKind, OutputKind
from torch.utils._pytree import tree_flatten

from .capture import (
    AotGraphPair,
    GraphArtifact,
    ObjectiveSchema,
    TaskInputProvenance,
    TaskInputRole,
    capture_objective_schema,
    normalize_objective_result,
)
from .contracts import CaptureError, ObjectiveError, ObjectiveResult
from .output_contract import ExplicitMutation, StorageRootKind


@dataclass(frozen=True, slots=True)
class ExportCapture:
    """One functional Export graph plus exact flattened example arguments."""

    exported_program: torch.export.ExportedProgram
    flat_inputs: tuple[object, ...]
    user_output_indices: tuple[int, ...]
    mutations: tuple[ExportMutation, ...]


@dataclass(frozen=True, slots=True)
class ExportMutation:
    """One Export signature output that replaces explicit input state."""

    output_index: int
    input_index: int
    kind: OutputKind
    target: str


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
    mutations = _export_mutations(exported.graph_signature)
    return ExportCapture(
        exported_program=exported,
        flat_inputs=flat_inputs,
        user_output_indices=user_outputs,
        mutations=mutations,
    )


def _export_mutations(
    signature: ExportGraphSignature,
) -> tuple[ExportMutation, ...]:
    """Normalize Export's target/name mutation maps into dense positions."""

    input_specs = tuple(signature.input_specs)
    output_specs = tuple(signature.output_specs)
    mutable_kinds = {
        OutputKind.BUFFER_MUTATION,
        OutputKind.PARAMETER_MUTATION,
        OutputKind.USER_INPUT_MUTATION,
    }
    result: list[ExportMutation] = []
    for output_index, output in enumerate(output_specs):
        if output.kind not in mutable_kinds:
            continue
        target = output.target
        if not isinstance(target, str) or not target:
            raise CaptureError("Export mutation output has no target")
        candidates: list[int] = []
        for input_index, input_spec in enumerate(input_specs):
            argument_name = getattr(input_spec.arg, "name", None)
            if output.kind is OutputKind.USER_INPUT_MUTATION:
                matches = (
                    input_spec.kind is InputKind.USER_INPUT and argument_name == target
                )
            else:
                expected = (
                    InputKind.BUFFER
                    if output.kind is OutputKind.BUFFER_MUTATION
                    else InputKind.PARAMETER
                )
                matches = input_spec.kind is expected and input_spec.target == target
            if matches:
                candidates.append(input_index)
        if len(candidates) != 1:
            raise CaptureError(
                "Export mutation target does not resolve to exactly one input: "
                f"output={output_index}, kind={output.kind.name}, "
                f"target={target!r}, candidates={candidates}"
            )
        result.append(ExportMutation(output_index, candidates[0], output.kind, target))
    return tuple(result)


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
        explicit_mutations=_explicit_mutations(capture),
    )


def export_capture_digest(capture: ExportCapture) -> str:
    """Return the stable semantic/input ABI of one freshly exported graph."""

    return GraphArtifact.input_compatibility_digest(
        graph_module=capture.exported_program.graph_module,
        example_inputs=capture.flat_inputs,
        explicit_mutations=_explicit_mutations(capture),
    )


def _capture_pair(capture: ExportCapture, *, recomputation: bool) -> AotGraphPair:
    graph_module = capture.exported_program.graph_module
    eager_output = graph_module(*capture.flat_inputs)
    return capture_graph_pair(
        graph_module,
        capture.flat_inputs,
        original_output=eager_output,
        recomputation=recomputation,
        root_output_positions=(capture.user_output_indices[0],),
        specialize_unit_tangents=True,
        explicit_mutations=_explicit_mutations(capture),
    )


def capture_graph_pair(
    graph_module: torch.fx.GraphModule,
    inputs: Sequence[object],
    *,
    original_output: object,
    recomputation: bool,
    root_output_positions: tuple[int, ...] | None = None,
    specialize_unit_tangents: bool = False,
    explicit_mutations: tuple[ExplicitMutation, ...] = (),
    input_provenance: tuple[TaskInputProvenance, ...] | None = None,
) -> AotGraphPair:
    """Differentiate one functional graph with a flat tensor/static ABI."""

    normalized_mutations = _tensor_only_mutations(explicit_mutations, tuple(inputs))
    tensor_provenance = (
        None
        if input_provenance is None
        else tuple(
            item
            for value, item in zip(inputs, input_provenance, strict=True)
            if isinstance(value, torch.Tensor)
        )
    )
    capture_inputs = tuple(
        value.detach().requires_grad_(value.requires_grad)
        if isinstance(value, torch.Tensor)
        else value
        for value in inputs
    )
    captured: dict[str, GraphArtifact] = {}
    raw_backward: dict[str, object] = {}

    def forward_compiler(
        graph_module: torch.fx.GraphModule, example_inputs: Sequence[object]
    ) -> Any:
        captured["forward"] = GraphArtifact.capture(
            kind="forward",
            graph_module=graph_module,
            example_inputs=tuple(example_inputs),
            explicit_mutations=normalized_mutations,
            input_provenance=tensor_provenance,
        )
        return make_boxed_func(graph_module.forward)

    def backward_compiler(
        graph_module: torch.fx.GraphModule, example_inputs: Sequence[object]
    ) -> Any:
        raw_backward["graph_module"] = graph_module
        raw_backward["example_inputs"] = tuple(example_inputs)
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
    if set(captured) != {"forward"} or set(raw_backward) != {
        "graph_module",
        "example_inputs",
    }:
        raise CaptureError("AOTAutograd did not emit a complete graph pair")
    original_output_count = len(tree_flatten(original_output)[0])
    saved = max(0, captured["forward"].output_count - original_output_count)
    backward_graph = raw_backward["graph_module"]
    backward_inputs = raw_backward["example_inputs"]
    if not isinstance(backward_graph, torch.fx.GraphModule) or not isinstance(
        backward_inputs, tuple
    ):
        raise CaptureError("AOTAutograd backward capture has an invalid type")
    backward_provenance = _backward_input_provenance(
        captured["forward"],
        original_output_count=original_output_count,
        saved_value_count=saved,
        backward_argument_count=len(backward_inputs),
    )
    backward = GraphArtifact.capture(
        kind="backward",
        graph_module=backward_graph,
        example_inputs=backward_inputs,
        input_provenance=backward_provenance,
    )
    specialized_count = 0
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


def _backward_input_provenance(
    forward: GraphArtifact,
    *,
    original_output_count: int,
    saved_value_count: int,
    backward_argument_count: int,
) -> tuple[TaskInputProvenance, ...]:
    """Project saved forward values and terminal tangents onto backward inputs."""

    if saved_value_count > backward_argument_count:
        raise CaptureError("AOT backward has fewer arguments than saved values")
    views = {item.leaf_index: item for item in forward.storage_contract.output_views}
    result: list[TaskInputProvenance] = []
    for offset in range(saved_value_count):
        leaf_index = original_output_count + offset
        view = views.get(leaf_index)
        if view is None:
            result.append(
                TaskInputProvenance(
                    TaskInputRole.RESIDUAL,
                    f"forward_output_{leaf_index}",
                )
            )
            continue
        root = forward.storage_contract.roots[view.root_id]
        if root.kind is StorageRootKind.INPUT:
            assert root.source_input is not None
            try:
                source = forward.input_provenance[root.source_input]
            except IndexError as exc:
                raise CaptureError(
                    "saved forward input root is outside the task ABI"
                ) from exc
            result.append(_saved_input_view_provenance(source, view))
        else:
            result.append(
                TaskInputProvenance(
                    TaskInputRole.RESIDUAL,
                    f"forward_output_{leaf_index}",
                )
            )
    result.extend(
        TaskInputProvenance(TaskInputRole.TANGENT, f"tangent_{index}")
        for index in range(backward_argument_count - saved_value_count)
    )
    return tuple(result)


def rebind_backward_input_provenance(
    pair: AotGraphPair,
    forward: GraphArtifact,
) -> tuple[TaskInputProvenance, ...]:
    """Rebuild saved-value provenance for one occurrence-local forward ABI."""

    original_output_count = forward.output_count - pair.saved_value_count
    if original_output_count < 0:
        raise CaptureError("AOT graph pair has an invalid saved-value count")
    return _backward_input_provenance(
        forward,
        original_output_count=original_output_count,
        saved_value_count=pair.saved_value_count,
        backward_argument_count=pair.backward.argument_count,
    )


def _saved_input_view_provenance(
    source: TaskInputProvenance,
    view: Any,
) -> TaskInputProvenance:
    """Preserve authentic values through an AOT-saved input view."""

    reference = source.representative_value
    if reference is None:
        return source
    itemsize = reference.element_size()
    if view.dtype != str(reference.dtype) or view.offset_bytes % itemsize:
        raise CaptureError(
            "saved input view is incompatible with its representative storage: "
            f"source={source.source}, dtype={view.dtype}, "
            f"offset_bytes={view.offset_bytes}"
        )
    storage_bytes = reference.untyped_storage().nbytes()
    if view.offset_bytes + view.span_bytes > storage_bytes:
        raise CaptureError(
            "saved input view exceeds its representative storage: "
            f"source={source.source}, required={view.offset_bytes + view.span_bytes}, "
            f"available={storage_bytes}"
        )
    # This helper runs while the enclosing AOT capture owns a FakeTensorMode.
    # The occurrence-local reference is intentionally a real CPU tensor, so
    # construct its metadata-only view below Python dispatch.
    with torch._C._DisableTorchDispatch():
        result = torch.empty(0, dtype=reference.dtype, device=reference.device)
        result.set_(
            reference.untyped_storage(),
            view.offset_bytes // itemsize,
            view.shape,
            view.stride,
        )
    return TaskInputProvenance(
        source.role,
        source.source,
        source.consumer_targets,
        result,
    )


def _explicit_mutations(capture: ExportCapture) -> tuple[ExplicitMutation, ...]:
    return tuple(
        ExplicitMutation(item.input_index, item.output_index, item.target)
        for item in capture.mutations
    )


def _tensor_only_mutations(
    mutations: tuple[ExplicitMutation, ...],
    inputs: tuple[object, ...],
) -> tuple[ExplicitMutation, ...]:
    """Translate a mixed positional ABI to AOT's tensor-only forward ABI."""

    tensor_position = {
        original: compact
        for compact, original in enumerate(
            index
            for index, value in enumerate(inputs)
            if isinstance(value, torch.Tensor)
        )
    }
    result: list[ExplicitMutation] = []
    for mutation in mutations:
        try:
            position = tensor_position[mutation.input_position]
        except KeyError as exc:
            raise CaptureError(
                "functional mutation target is not an AOT tensor input"
            ) from exc
        result.append(
            ExplicitMutation(position, mutation.output_leaf_index, mutation.target)
        )
    return tuple(result)


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
    for placeholder, tangent in zip(placeholders[-count:], tangents, strict=True):
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
            input_provenance=backward.input_provenance[:-count],
        ),
        count,
    )
