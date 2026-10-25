"""Version-pinned PyTorch 2.13 Export and AOTAutograd boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import torch
import torch.nn as nn
from functorch.compile import make_boxed_func  # type: ignore[import-untyped]
from torch._functorch import config as functorch_config
from torch._functorch.aot_autograd import aot_function
from torch.export.graph_signature import ExportGraphSignature, InputKind, OutputKind
from torch.utils._pytree import tree_flatten, tree_unflatten

from shadowspill.errors import CaptureError, ObjectiveError
from shadowspill.pytorch.capture.artifacts import (
    AotGraphPair,
    GraphArtifact,
    ObjectiveSchema,
    TaskInputProvenance,
    capture_inference_artifact,
    capture_objective_schema,
    normalize_objective_result,
)
from shadowspill.pytorch.capture.retention import (
    RetentionPolicy,
    RetentionRecord,
    RetentionSummary,
    partition_function,
)
from shadowspill.pytorch.capture.storage import ExplicitMutation, StorageRootKind
from shadowspill.pytorch.contracts import ObjectiveResult
from shadowspill.pytorch.representations import is_wrapper, tensor_components
from shadowspill.task.inputs import TaskInputRole

from .effects import normalize_export_effects
from .functional import functionalize_logical_export
from .torch_deprecations import copy_graph_module, quiet_leaf_spec_deprecation


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


class _GraphPairCollector:
    """Capture AOT compiler callbacks without mixing them into orchestration."""

    def __init__(
        self,
        mutations: tuple[ExplicitMutation, ...],
        input_provenance: tuple[TaskInputProvenance, ...] | None,
    ) -> None:
        self._mutations = mutations
        self._input_provenance = input_provenance
        self.forward: GraphArtifact | None = None
        self.backward_graph: torch.fx.GraphModule | None = None
        self.backward_inputs: tuple[object, ...] | None = None

    def compile_forward(
        self,
        graph_module: torch.fx.GraphModule,
        example_inputs: Sequence[object],
    ) -> Any:
        self.forward = GraphArtifact.capture(
            kind="forward",
            graph_module=graph_module,
            example_inputs=tuple(example_inputs),
            explicit_mutations=self._mutations,
            input_provenance=self._input_provenance,
        )
        return make_boxed_func(graph_module.forward)

    def compile_backward(
        self,
        graph_module: torch.fx.GraphModule,
        example_inputs: Sequence[object],
    ) -> Any:
        self.backward_graph = graph_module
        self.backward_inputs = tuple(example_inputs)
        return make_boxed_func(graph_module.forward)

    def require_complete(
        self,
    ) -> tuple[GraphArtifact, torch.fx.GraphModule, tuple[object, ...]]:
        if (
            self.forward is None
            or self.backward_graph is None
            or self.backward_inputs is None
        ):
            raise CaptureError("AOTAutograd did not emit a complete graph pair")
        return self.forward, self.backward_graph, self.backward_inputs


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

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        loss, metrics = normalize_objective_result(
            self.objective(self.model, *args, **kwargs), require_grad=False
        )
        leaves, tree_spec = tree_flatten(metrics)
        if tree_spec != self.schema.metric_tree_spec:
            raise ObjectiveError("objective metric structure changed during capture")
        tensor_metrics = tuple(
            leaves[position].detach()
            for position in self.schema.tensor_metric_positions
        )
        return (loss, *tensor_metrics)


def _flatten_inputs(
    exported: torch.export.ExportedProgram,
    inputs: Sequence[Any],
    kwargs: Mapping[str, Any] | None = None,
) -> tuple[object, ...]:
    flatten = getattr(exported, "_graph_module_flat_inputs", None)
    if not callable(flatten):
        raise CaptureError(
            "PyTorch 2.13 ExportedProgram flat-input adapter is unavailable"
        )
    return tuple(flatten(tuple(inputs), dict(kwargs or {})))


def _functional_copy(
    destination: torch.Tensor,
    source: torch.Tensor,
    non_blocking: bool = False,
) -> torch.Tensor:
    """What a copy into a value is worth, as a value.

    Writing into a slice of a value is how several models assemble one:
    an output is allocated and its halves written in place. Capture
    functionalizes that write into `aten.copy`, whose result is the
    destination's shape and type holding the source's numbers -- and
    which has no derivative, because the operator it came from is a
    mutation and mutations are not differentiated.

    Read as a value it is differentiable, and this says how: the source,
    shaped and typed like the destination it was written into. Without it
    a model that assembles a tensor by writing into it captures for a
    forward pass and refuses for a backward one.
    """

    return source.to(destination.dtype).expand_as(destination).clone()


def _repair_constant_targets(exported: torch.export.ExportedProgram) -> None:
    """Give lifted constants valid attribute names before Export unlift.

    A selected forward callback can make Dynamo name an inline tensor through
    a Python closure, including indexing such as __closure__[1]. Export
    later interprets that string as an attribute path and fails. Constants have
    no source-model state name to preserve: rename only those invalid targets,
    keeping the exact tensors, input positions, aliases and parameter names.
    """
    used = {
        str(spec.target).split(".")[0]
        for spec in exported.graph_signature.input_specs
        if spec.target is not None
    }
    renamed: dict[str, str] = {}
    for spec in exported.graph_signature.input_specs:
        target = spec.target
        if (
            spec.kind is not InputKind.CONSTANT_TENSOR
            or not isinstance(target, str)
            or all(
                part.isidentifier() or part.isdecimal() for part in target.split(".")
            )
        ):
            continue
        if target not in renamed:
            index = len(renamed)
            candidate = f"_shadowspill_constant_{index}"
            while candidate in used:
                index += 1
                candidate = f"_shadowspill_constant_{index}"
            used.add(candidate)
            exported.constants[candidate] = exported.constants.pop(target)
            renamed[target] = candidate
        spec.target = renamed[target]


def _export(
    module: nn.Module,
    inputs: Sequence[Any],
    kwargs: Mapping[str, Any] | None = None,
) -> ExportCapture:
    try:
        with quiet_leaf_spec_deprecation():
            exported = torch.export.export(
                module, tuple(inputs), kwargs=dict(kwargs or {}), strict=True
            )
            _repair_constant_targets(exported)
            # Decomposition unwraps tensor subclasses before differentiation,
            # losing their logical gradient contract (and can mark integer
            # payloads requires_grad). Preserve logical operators until AOT.
            flat_inputs = _flatten_inputs(exported, inputs, kwargs)
            if any(is_wrapper(value) for value in flat_inputs):
                exported = functionalize_logical_export(
                    exported, flat_inputs, _functional_copy
                )
            else:
                exported = exported.run_decompositions(
                    {torch.ops.aten.copy.default: _functional_copy}
                )
            normalize_export_effects(exported)
    except BaseException as exc:
        raise CaptureError(f"strict PyTorch export failed: {exc}") from exc
    flat_inputs = _flatten_inputs(exported, inputs, kwargs)
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
    """Normalize Export's target/name mutation maps into contiguous positions."""

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


def capture_forward(
    module: nn.Module,
    inputs: Sequence[Any],
    *,
    kwargs: Mapping[str, Any] | None = None,
) -> ExportCapture:
    """Strictly export an inference graph while keeping all state explicit."""

    return _export(module, inputs, kwargs)


def capture_training(
    model: nn.Module,
    objective: Callable[..., torch.Tensor | ObjectiveResult],
    microbatch: Sequence[Any],
) -> TrainingCapture:
    """Capture the objective plus its whole-graph ``save`` and ``recompute`` pairs."""

    objective_capture = capture_training_objective(model, objective, microbatch)
    save_pair = _capture_pair(objective_capture.exported, memory_budget=1.0)
    recompute_pair = _capture_pair(objective_capture.exported, memory_budget=0.0)
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
    *,
    kwargs: Mapping[str, Any] | None = None,
) -> TrainingObjectiveCapture:
    """Export an objective without constructing an unused whole-model VJP."""

    try:
        probe_loss, probe_metrics = normalize_objective_result(
            objective(model, *microbatch, **dict(kwargs or {})), require_grad=True
        )
        del probe_loss
        schema = capture_objective_schema(probe_metrics)
        capture_module = _ObjectiveModule(model, objective, schema)
        exported = _export(capture_module, microbatch, kwargs)
    except (CaptureError, ObjectiveError):
        raise
    except BaseException as error:
        raise CaptureError(f"training objective capture failed: {error}") from error
    return TrainingObjectiveCapture(
        exported=exported,
        capture_module=capture_module,
        objective_schema=schema,
    )


def rebind_training_objective(
    capture: TrainingObjectiveCapture,
    microbatch: Sequence[Any],
    *,
    kwargs: Mapping[str, Any] | None = None,
) -> TrainingObjectiveCapture:
    """The same exported objective bound to another position's example inputs.

    Positions with one input structure share the export; only the flattened
    example arguments differ, and those are cheap.
    """

    exported = replace(
        capture.exported,
        flat_inputs=_flatten_inputs(
            capture.exported.exported_program, microbatch, kwargs
        ),
    )
    return replace(capture, exported=exported)


def bind_training_objective(
    program: torch.export.ExportedProgram,
    model: nn.Module,
    schema: ObjectiveSchema,
    microbatch: Sequence[Any],
) -> TrainingObjectiveCapture:
    """Bind an explicitly loaded export to fresh fake state without exporting."""

    for spec in program.graph_signature.input_specs:
        if spec.kind not in {InputKind.PARAMETER, InputKind.BUFFER}:
            continue
        if not isinstance(spec.target, str) or not spec.target.startswith("model."):
            raise CaptureError("prepared training state target is not rooted at model")
        name = spec.target.removeprefix("model.")
        # Export stores non-persistent buffers alongside lifted constants, while
        # persistent buffers and parameters live in state_dict. Both kinds of
        # buffer must bind to this model, never to the archived meta placeholder.
        state = (
            program.constants
            if spec.kind is InputKind.BUFFER and not spec.persistent
            else program.state_dict
        )
        state[spec.target] = (
            model.get_parameter(name)
            if spec.kind is InputKind.PARAMETER
            else model.get_buffer(name)
        )
    # Partition policies inspect the source module hierarchy. The loaded graph
    # is the objective; no Python objective is called during admission.
    module = nn.Module()
    module.add_module("model", model)
    return TrainingObjectiveCapture(
        ExportCapture(
            program,
            _flatten_inputs(program, microbatch),
            tuple(
                i
                for i, spec in enumerate(program.graph_signature.output_specs)
                if spec.kind is OutputKind.USER_OUTPUT
            ),
            _export_mutations(program.graph_signature),
        ),
        module,
        schema,
    )


def inference_artifact(capture: ExportCapture) -> GraphArtifact:
    """Create the structural task contract for a functional Export graph."""

    return capture_inference_artifact(
        graph_module=capture.exported_program.graph_module,
        example_inputs=capture.flat_inputs,
        explicit_mutations=_explicit_mutations(capture),
    )


def export_capture_digest(capture: ExportCapture) -> str:
    """Return the stable semantic/input contract of one exported graph."""

    return GraphArtifact.input_compatibility_digest(
        graph_module=capture.exported_program.graph_module,
        example_inputs=capture.flat_inputs,
        explicit_mutations=_explicit_mutations(capture),
    )


def _capture_pair(capture: ExportCapture, *, memory_budget: float) -> AotGraphPair:
    graph_module = capture.exported_program.graph_module
    eager_output = graph_module(*capture.flat_inputs)
    return capture_graph_pair(
        graph_module,
        capture.flat_inputs,
        original_output=eager_output,
        memory_budget=memory_budget,
        root_output_positions=(capture.user_output_indices[0],),
        specialize_unit_tangents=True,
        explicit_mutations=_explicit_mutations(capture),
    )


def capture_graph_pair(
    graph_module: torch.fx.GraphModule,
    inputs: Sequence[object],
    *,
    original_output: object,
    memory_budget: float = 1.0,
    retention: RetentionPolicy | None = None,
    root_output_positions: tuple[int, ...] | None = None,
    specialize_unit_tangents: bool = False,
    explicit_mutations: tuple[ExplicitMutation, ...] = (),
    input_provenance: tuple[TaskInputProvenance, ...] | None = None,
) -> AotGraphPair:
    """Differentiate one functional graph with a flat tensor/static signature.

    ``memory_budget`` is the partition's. ``1.0`` retains for the backward
    what ``retention`` says is expensive to regenerate and regenerates the
    rest; ``0.0`` retains the inputs alone; a value between the two is
    PyTorch's knapsack between them. ``None`` is the library's default policy
    (:mod:`shadowspill.pytorch.capture.retention`).
    """

    if not 0.0 <= memory_budget <= 1.0:
        raise ValueError("memory_budget must be between zero and one")
    normalized_mutations = _tensor_only_mutations(explicit_mutations, tuple(inputs))
    if any(is_wrapper(value) for value in inputs):
        offsets = []
        offset = 0
        for value in inputs:
            if isinstance(value, torch.Tensor):
                offsets.append(offset)
                offset += sum(1 for _ in tensor_components(value))
        normalized_mutations = tuple(
            replace(item, input_position=offsets[item.input_position])
            for item in normalized_mutations
        )
    logical_provenance = _tensor_input_provenance(inputs, input_provenance)
    tensor_provenance = physical_input_provenance(inputs, input_provenance)
    capture_inputs = _capture_inputs(inputs)
    collector = _GraphPairCollector(normalized_mutations, tensor_provenance)
    record = RetentionRecord()
    roots = _execute_aot_capture(
        graph_module,
        capture_inputs,
        collector,
        partition=partition_function(
            memory_budget, retention or RetentionPolicy(), record
        ),
        memory_budget=memory_budget,
        root_output_positions=root_output_positions,
    )
    forward, backward_graph, backward_inputs = collector.require_complete()
    pair = _build_graph_pair(
        forward,
        backward_graph,
        backward_inputs,
        roots,
        original_output,
        retention=record.require(),
        specialize_unit_tangents=specialize_unit_tangents,
    )
    if any(is_wrapper(value) for value in inputs):
        components = tuple(
            (index, path)
            for index, value in enumerate(inputs)
            if isinstance(value, torch.Tensor)
            for path, _ in tensor_components(value)
        )
        pair = replace(
            pair,
            forward=replace(pair.forward, input_components=components),
            gradient_provenance=logical_provenance
            or tuple(
                TaskInputProvenance(TaskInputRole.USER_INPUT)
                for value in inputs
                if isinstance(value, torch.Tensor)
            ),
        )
    return pair


def physical_input_provenance(
    inputs: Sequence[object], provenance: tuple[TaskInputProvenance, ...] | None
) -> tuple[TaskInputProvenance, ...] | None:
    """Expand logical input provenance in AOT's physical-component order."""
    if provenance is None:
        return None
    expanded = []
    for value, item in zip(inputs, provenance, strict=True):
        if not isinstance(value, torch.Tensor):
            continue
        references = (
            dict(tensor_components(item.representative_value))
            if item.representative_value is not None
            else {}
        )
        for path, _component in tensor_components(value):
            expanded.append(replace(item, representative_value=references.get(path)))
    return tuple(expanded)


# A backward graph is captured for one tangent layout and is then called
# directly: nothing stands between a plan and the compiled artifact. The
# compiler's own guess is that a tangent arrives strided exactly like the
# forward output it belongs to, and its runtime wrapper restrides every
# incoming gradient that disagrees. A stage boundary has no such wrapper --
# the gradient one task publishes is the gradient the next task is handed --
# so a guessed layout that the producing task does not yield is read as a
# wrong stride inside the compiled kernel. Restriding at the boundary is not
# open to us either: geometry sizes objects, alias extents and offsets before
# any task is compiled, so a copy nobody planned has nowhere to live. Pinning
# the guess off asks for the canonical memory format of the output, which is
# a function of the graph's structure rather than of the strides one capture
# happened to produce.
_PINNED_AOT_CONFIG: Mapping[str, Any] = {
    "guess_tangent_strides_as_outputs": False,
    # Compiler ordering tokens belong inside tasks, not in model inputs or
    # saved activations. Keep with_effects chains in the compiled graphs.
    "unlift_effect_tokens": True,
}


def _execute_aot_capture(
    graph_module: torch.fx.GraphModule,
    capture_inputs: tuple[object, ...],
    collector: _GraphPairCollector,
    *,
    partition: Callable[..., tuple[torch.fx.GraphModule, torch.fx.GraphModule]],
    memory_budget: float,
    root_output_positions: tuple[int, ...] | None,
) -> tuple[torch.Tensor, ...]:
    try:
        with functorch_config.patch(**_PINNED_AOT_CONFIG):
            aot: Any = aot_function
            compiled = cast(
                Callable[..., object],
                aot(
                    _rooted_forward(graph_module, root_output_positions),
                    fw_compiler=collector.compile_forward,
                    bw_compiler=collector.compile_backward,
                    partition_fn=partition,
                ),
            )
            outputs = compiled(*capture_inputs)
            roots = _differentiable_roots(outputs, root_output_positions)
            _trigger_backward_capture(roots, capture_inputs)
        return roots
    except CaptureError:
        raise
    except BaseException as exc:
        raise CaptureError(
            f"AOTAutograd capture at partition budget {memory_budget:g} failed: {exc}"
        ) from exc


def _rooted_forward(
    graph_module: torch.fx.GraphModule,
    root_output_positions: tuple[int, ...] | None,
) -> Callable[..., object]:
    """Keep non-root values as forward outputs without creating cotangents.

    Selecting outputs only at torch.autograd.grad is too late: AOT has already
    built a backward accepting gradients for every differentiable output.
    """
    if root_output_positions is None:
        return graph_module
    selected = frozenset(root_output_positions)

    def forward(*args: object) -> object:
        leaves, spec = tree_flatten(graph_module(*args))
        return tree_unflatten(
            [
                value.detach()
                if isinstance(value, torch.Tensor) and index not in selected
                else value
                for index, value in enumerate(leaves)
            ],
            spec,
        )

    return forward


def _differentiable_roots(
    outputs: object,
    root_output_positions: tuple[int, ...] | None,
) -> tuple[torch.Tensor, ...]:
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
    if not roots or not all(isinstance(root, torch.Tensor) for root in roots):
        raise CaptureError("training stage has no differentiable tensor output")
    return roots


def _trigger_backward_capture(
    roots: tuple[torch.Tensor, ...],
    capture_inputs: tuple[object, ...],
) -> None:
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


def _build_graph_pair(
    forward: GraphArtifact,
    backward_graph: torch.fx.GraphModule,
    backward_inputs: tuple[object, ...],
    roots: tuple[torch.Tensor, ...],
    original_output: object,
    *,
    retention: RetentionSummary,
    specialize_unit_tangents: bool,
) -> AotGraphPair:
    original_output_count = len(tree_flatten(original_output)[0])
    saved = max(0, forward.output_count - original_output_count)
    backward_provenance = _backward_input_provenance(
        forward,
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
            backward, roots
        )
    return AotGraphPair(
        forward=forward,
        backward=backward,
        retention=retention,
        saved_value_count=saved,
        specialized_unit_tangent_count=specialized_count,
    )


def _tensor_input_provenance(
    inputs: Sequence[object],
    provenance: tuple[TaskInputProvenance, ...] | None,
) -> tuple[TaskInputProvenance, ...] | None:
    if provenance is None:
        return None
    return tuple(
        item
        for value, item in zip(inputs, provenance, strict=True)
        if isinstance(value, torch.Tensor)
    )


def _capture_inputs(inputs: Sequence[object]) -> tuple[object, ...]:
    return tuple(
        value.detach().requires_grad_(value.requires_grad)
        if isinstance(value, torch.Tensor)
        else value
        for value in inputs
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
                    "saved forward input root is outside the task contract"
                ) from exc
            result.append(_saved_input_view_provenance(source, view))
        else:
            result.append(
                TaskInputProvenance(
                    (
                        TaskInputRole.RESIDUAL
                        if _is_continuous_dtype_name(view.dtype)
                        else TaskInputRole.CONTROL
                    ),
                    f"forward_output_{leaf_index}",
                )
            )
    result.extend(
        TaskInputProvenance(TaskInputRole.TANGENT, f"tangent_{index}")
        for index in range(backward_argument_count - saved_value_count)
    )
    return tuple(result)


def _is_continuous_dtype_name(dtype_name: str) -> bool:
    dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise CaptureError(f"AOT saved value has unknown dtype {dtype_name!r}")
    return bool(dtype.is_floating_point or dtype.is_complex)


def rebind_backward_input_provenance(
    pair: AotGraphPair,
    forward: GraphArtifact,
) -> tuple[TaskInputProvenance, ...]:
    """Rebuild saved-value provenance for one occurrence-local forward contract."""

    original_output_count = forward.output_count - pair.saved_value_count
    if original_output_count < 0:
        raise CaptureError("AOT graph pair has an invalid saved-value count")
    # A backward that accumulates takes the running gradients as further
    # arguments. They come from the plan rather than from the forward, so
    # projecting the forward cannot rebuild them; carry them across instead.
    priors = tuple(
        item
        for item in pair.backward.input_provenance
        if item.role is TaskInputRole.GRADIENT
    )
    if priors and pair.backward.input_provenance[-len(priors) :] != priors:
        raise CaptureError(
            "accumulated gradient arguments are not the backward's last ones"
        )
    return (
        *_backward_input_provenance(
            forward,
            original_output_count=original_output_count,
            saved_value_count=pair.saved_value_count,
            backward_argument_count=pair.backward.argument_count - len(priors),
        ),
        *priors,
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
        result = reference.as_strided(
            view.shape,
            view.stride,
            view.offset_bytes // itemsize,
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
    """Translate a mixed positional signature to AOT's tensor-only forward contract."""

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

    graph_module = copy_graph_module(backward.graph_module)
    placeholders = tuple(
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    )
    if len(placeholders) != len(backward.example_arguments):
        raise CaptureError("backward placeholder count changed before specialization")
    anchors = placeholders[:-count]
    if not anchors:
        # A device-relative scalar cannot be constructed without either a
        # tensor anchor or a backend-specific device literal. Preserve the
        # explicit tangent contract for this degenerate graph.
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
