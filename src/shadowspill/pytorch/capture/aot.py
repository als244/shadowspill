"""Version-pinned PyTorch 2.13 Export and AOTAutograd boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any, cast

import torch
import torch.nn as nn
from functorch.compile import make_boxed_func  # type: ignore[import-untyped]
from torch._functorch import config as functorch_config
from torch._functorch.aot_autograd import aot_function
from torch._functorch.partitioners import min_cut_rematerialization_partition
from torch._guards import detect_fake_mode
from torch._prims_common import get_computation_dtype
from torch.export.graph_signature import ExportGraphSignature, InputKind, OutputKind
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils._pytree import tree_flatten

from shadowspill.errors import CaptureError, ObjectiveError
from shadowspill.pytorch.capture.artifacts import (
    AotGraphPair,
    GraphArtifact,
    ObjectiveSchema,
    TaskInputProvenance,
    capture_objective_schema,
    normalize_objective_result,
)
from shadowspill.pytorch.capture.storage import ExplicitMutation, StorageRootKind
from shadowspill.pytorch.contracts import ObjectiveResult
from shadowspill.task.inputs import TaskInputRole

from .accumulate import ACCUMULATE_MATMUL, ADDING_INTO
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


def _flatten_inputs(
    exported: torch.export.ExportedProgram, inputs: Sequence[Any]
) -> tuple[object, ...]:
    flatten = getattr(exported, "_graph_module_flat_inputs", None)
    if not callable(flatten):
        raise CaptureError(
            "PyTorch 2.13 ExportedProgram flat-input adapter is unavailable"
        )
    return tuple(flatten(tuple(inputs), {}))


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


def _export(module: nn.Module, inputs: Sequence[Any]) -> ExportCapture:
    try:
        with quiet_leaf_spec_deprecation():
            exported = torch.export.export(module, tuple(inputs), strict=True)
            exported = exported.run_decompositions(
                {torch.ops.aten.copy.default: _functional_copy}
            )
    except BaseException as exc:
        raise CaptureError(f"strict PyTorch export failed: {exc}") from exc
    flat_inputs = _flatten_inputs(exported, inputs)
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

    try:
        probe_loss, probe_metrics = normalize_objective_result(
            objective(model, *microbatch), require_grad=True
        )
        del probe_loss
        schema = capture_objective_schema(probe_metrics)
        capture_module = _ObjectiveModule(model, objective, schema)
        exported = _export(capture_module, microbatch)
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
    capture: TrainingObjectiveCapture, microbatch: Sequence[Any]
) -> TrainingObjectiveCapture:
    """The same exported objective bound to another position's example inputs.

    Positions with one input structure share the export; only the flattened
    example arguments differ, and those are cheap.
    """

    exported = replace(
        capture.exported,
        flat_inputs=_flatten_inputs(capture.exported.exported_program, microbatch),
    )
    return replace(capture, exported=exported)


def inference_artifact(capture: ExportCapture) -> GraphArtifact:
    """Create the structural task contract for a functional Export graph."""

    return GraphArtifact.capture(
        kind="inference",
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
    activation_memory_budget: float | None = None,
    root_output_positions: tuple[int, ...] | None = None,
    specialize_unit_tangents: bool = False,
    explicit_mutations: tuple[ExplicitMutation, ...] = (),
    input_provenance: tuple[TaskInputProvenance, ...] | None = None,
) -> AotGraphPair:
    """Differentiate one functional graph with a flat tensor/static signature."""

    _validate_activation_budget(recomputation, activation_memory_budget)
    normalized_mutations = _tensor_only_mutations(explicit_mutations, tuple(inputs))
    tensor_provenance = _tensor_input_provenance(inputs, input_provenance)
    capture_inputs = _capture_inputs(inputs)
    collector = _GraphPairCollector(normalized_mutations, tensor_provenance)
    roots = _execute_aot_capture(
        graph_module,
        capture_inputs,
        collector,
        recomputation=recomputation,
        activation_memory_budget=activation_memory_budget,
        root_output_positions=root_output_positions,
    )
    forward, backward_graph, backward_inputs = collector.require_complete()
    return _build_graph_pair(
        forward,
        backward_graph,
        backward_inputs,
        roots,
        original_output,
        recomputation=recomputation,
        specialize_unit_tangents=specialize_unit_tangents,
    )


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
_PINNED_TANGENT_LAYOUT: Mapping[str, Any] = {"guess_tangent_strides_as_outputs": False}


def _execute_aot_capture(
    graph_module: torch.fx.GraphModule,
    capture_inputs: tuple[object, ...],
    collector: _GraphPairCollector,
    *,
    recomputation: bool,
    activation_memory_budget: float | None,
    root_output_positions: tuple[int, ...] | None,
) -> tuple[torch.Tensor, ...]:
    try:
        with functorch_config.patch(**_PINNED_TANGENT_LAYOUT):
            compiled = _aot_callable(
                graph_module,
                collector,
                recomputation=recomputation,
                activation_memory_budget=activation_memory_budget,
            )
            outputs = compiled(*capture_inputs)
            roots = _differentiable_roots(outputs, root_output_positions)
            _trigger_backward_capture(roots, capture_inputs)
        return roots
    except CaptureError:
        raise
    except BaseException as exc:
        mode = "recomputation" if recomputation else "save"
        raise CaptureError(f"AOTAutograd {mode} capture failed: {exc}") from exc


def _aot_callable(
    graph_module: torch.fx.GraphModule,
    collector: _GraphPairCollector,
    *,
    recomputation: bool,
    activation_memory_budget: float | None,
) -> Callable[..., object]:
    aot: Any = aot_function
    if not recomputation:
        return cast(
            Callable[..., object],
            aot(
                graph_module,
                fw_compiler=collector.compile_forward,
                bw_compiler=collector.compile_backward,
            ),
        )
    return cast(
        Callable[..., object],
        aot(
            graph_module,
            fw_compiler=collector.compile_forward,
            bw_compiler=collector.compile_backward,
            partition_fn=_min_cut_partitioner(activation_memory_budget),
        ),
    )


def _min_cut_partitioner(
    activation_memory_budget: float | None,
) -> Callable[..., tuple[torch.fx.GraphModule, torch.fx.GraphModule]]:
    """Bind a memory budget to the lazy AOT partition callback itself.

    ``aot_function`` does not partition when its callable is constructed.  It
    partitions on the first invocation, after the caller's construction scope
    has returned.  The budget must therefore be scoped inside the callback
    that AOT invokes; wrapping callable construction silently observes the
    ambient Functorch default instead.
    """

    def partition(
        joint_module: torch.fx.GraphModule,
        joint_inputs: object,
        **kwargs: Any,
    ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        if activation_memory_budget is None:
            return min_cut_rematerialization_partition(
                joint_module,
                joint_inputs,
                **kwargs,
            )
        with functorch_config.patch(activation_memory_budget=activation_memory_budget):
            return min_cut_rematerialization_partition(
                joint_module,
                joint_inputs,
                **kwargs,
            )

    return partition


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
    recomputation: bool,
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
        recomputation=recomputation,
        saved_value_count=saved,
        specialized_unit_tangent_count=specialized_count,
    )


def _validate_activation_budget(
    recomputation: bool,
    activation_memory_budget: float | None,
) -> None:
    if activation_memory_budget is None:
        return
    if not recomputation:
        raise ValueError("activation_memory_budget requires the min-cut partitioner")
    if not 0.0 <= activation_memory_budget <= 1.0:
        raise ValueError("activation_memory_budget must be between zero and one")


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


def accumulate_gradient_outputs(
    backward: GraphArtifact,
    leaf_indices: Sequence[int],
    *,
    round_accumulation_once: bool = False,
) -> GraphArtifact:
    """Return a backward that adds its gradients onto ones it is given.

    Every microbatch after the first contributes to a gradient that already
    exists. Left alone, the graph hands back a fresh gradient and something
    outside it has to do the addition, which puts real device work between
    tasks where no plan accounts for it.

    So the graph takes the running gradient as an argument and returns the sum.
    This adds onto outputs rather than rewriting the operations that produce
    them, so it holds for any backward: whatever computed the gradient, its
    result is what gets added to. The addition is in place, so the running
    gradient keeps its storage. It is declared as a mutation of that argument,
    which is how the runtime knows the returned gradient is the argument rather
    than a new one.

    The compiler folds the add into the kernel it generates for whatever
    produced the contribution -- a reduction, say -- so nothing is written
    twice. A matrix multiply is a library call that writes its result before
    anything can read it, so a gradient one computes, moved at most by views,
    is added by the multiply itself instead (:func:`accumulate_matmul_`):
    ``C = A @ B + C`` into the running gradient, where the device has a kernel
    for it at these dtypes. That rounds the sum once. Adding after the
    multiply rounds it once too when the running gradient is at the dtype the
    multiply sums at, and the two agree; when it is narrower -- bf16 gradients
    -- adding after rounds the product first. The multiply adds such a
    gradient only with ``round_accumulation_once``, which trades agreement
    with adding after for the one rounding and the pass it saves.
    """

    if not leaf_indices:
        return backward
    graph_module = copy_graph_module(backward.graph_module)
    graph = graph_module.graph
    output_node = next(node for node in graph.nodes if node.op == "output")
    outputs = list(output_node.args[0])
    for leaf in leaf_indices:
        if leaf < 0 or leaf >= len(outputs) or outputs[leaf] is None:
            raise CaptureError(f"backward has no gradient output at leaf {leaf}")

    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    if len(placeholders) != len(backward.example_arguments):
        raise CaptureError("backward placeholder count changed before accumulation")
    # The arguments this adds have to belong to the same fake mode as the ones
    # already there, and the caller need not be inside that mode.
    fake_mode = detect_fake_mode(backward.example_arguments)
    anchor = placeholders[-1]
    priors: list[torch.Tensor] = []
    mutations: list[ExplicitMutation] = []
    replaced: list[torch.fx.Node] = []
    for offset, leaf in enumerate(leaf_indices):
        produced = outputs[leaf]
        value = produced.meta.get("val")
        if not isinstance(value, torch.Tensor):
            raise CaptureError(f"backward gradient at leaf {leaf} has no geometry")
        with graph.inserting_after(anchor):
            prior = graph.placeholder(f"shadowspill_prior_grad_{leaf}")
        prior.meta = dict(produced.meta)
        with fake_mode if fake_mode is not None else nullcontext():
            _record_value(prior, torch.zeros_like(value))
        anchor = prior
        adding = _add_in_multiply(
            graph, produced, prior, outputs, fake_mode, round_accumulation_once
        )
        if adding is not None:
            outputs[leaf] = prior
            replaced.extend(adding)
        else:
            with graph.inserting_before(output_node):
                total = graph.call_function(
                    torch.ops.aten.add_.Tensor, args=(prior, produced)
                )
            total.meta = dict(produced.meta)
            outputs[leaf] = total
        priors.append(prior.meta["val"])
        mutations.append(
            ExplicitMutation(
                input_position=len(backward.example_arguments) + offset,
                output_leaf_index=leaf,
                target=f"shadowspill_prior_grad_{leaf}",
            )
        )
    output_node.args = (tuple(outputs),)
    for node in replaced:
        graph.erase_node(node)
    graph.lint()
    graph_module.recompile()
    return GraphArtifact.capture(
        kind="backward",
        graph_module=graph_module,
        example_inputs=(*backward.example_arguments, *priors),
        explicit_mutations=tuple(mutations),
        input_provenance=(
            *backward.input_provenance,
            *(TaskInputProvenance(role=TaskInputRole.GRADIENT) for _ in leaf_indices),
        ),
    )


def cast_gradient_outputs(
    backward: GraphArtifact,
    leaf_indices: Sequence[int],
    dtype: torch.dtype,
) -> GraphArtifact:
    """Return a backward whose gradients at ``leaf_indices`` come out at ``dtype``.

    A backward computes a gradient at the dtype of what it is the gradient of.
    One kept at another dtype -- fp32 gradients of bf16 weights, say -- comes
    out at that one: a gradient the step creates is created at it, and the
    accumulating form, derived from this one, adds each contribution into the
    running gradient there.

    Which operation computed a gradient decides how. One an operation
    returned at ``dtype`` already -- a kernel asked for fp32 weight gradients
    -- reaches the output converted to the parameter's dtype, since autograd
    gives a parameter its gradient at the parameter's dtype; the conversion is
    dropped and the value the operation returned kept. One a matrix multiply
    computes, moved at most by views and copies on its way out, is written at
    ``dtype`` by the multiply itself where PyTorch has a kernel for it (fp32
    from fp16 or bf16 operands on CUDA, say): its products are summed at
    ``dtype`` and never rounded to the operands'. Any other is what the
    operations computed, cast as it leaves.
    """

    graph_module = copy_graph_module(backward.graph_module)
    graph = graph_module.graph
    output_node = next(node for node in graph.nodes if node.op == "output")
    outputs = list(output_node.args[0])
    fake_mode = detect_fake_mode(backward.example_arguments)
    changed = False
    for leaf in leaf_indices:
        produced = outputs[leaf] if 0 <= leaf < len(outputs) else None
        if produced is None or not isinstance(produced.meta.get("val"), torch.Tensor):
            raise CaptureError(f"backward gradient at leaf {leaf} has no geometry")
        value = produced.meta["val"]
        if value.dtype == dtype:
            continue
        changed = True
        unconverted = _before_conversion(graph, produced, outputs, dtype, fake_mode)
        if unconverted is not None:
            outputs[leaf] = unconverted
            continue
        if _write_at_dtype(graph, produced, outputs, dtype, fake_mode):
            continue
        with graph.inserting_before(output_node):
            cast = graph.call_function(
                torch.ops.prims.convert_element_type.default, args=(produced, dtype)
            )
        cast.meta = dict(produced.meta)
        with fake_mode if fake_mode is not None else nullcontext():
            _record_value(cast, value.to(dtype))
        outputs[leaf] = cast
    if not changed:
        return backward
    output_node.args = (tuple(outputs),)
    graph.lint()
    graph_module.recompile()
    return GraphArtifact.capture(
        kind="backward",
        graph_module=graph_module,
        example_inputs=backward.example_arguments,
        input_provenance=backward.input_provenance,
    )


#: Operations that only convert a tensor's dtype.
_CONVERTING = frozenset(
    {
        torch.ops.prims.convert_element_type.default,
        torch.ops.aten._to_copy.default,
    }
)


def _before_conversion(
    graph: torch.fx.Graph,
    produced: torch.fx.Node,
    outputs: Sequence[object],
    dtype: torch.dtype,
    fake_mode: Any,
) -> torch.fx.Node | None:
    """The gradient ``produced`` is converted from, where that is at ``dtype``.

    It is when ``produced`` is a dtype conversion, moved at most, that nothing
    else reads, of a value at ``dtype`` already: the moves are applied to that
    value instead, and the conversion goes. Returns the node that now gives the
    gradient, or ``None``.
    """

    chain = [produced]
    while chain[-1].target in _MOVING_ONLY:
        source = chain[-1].args[0]
        if not isinstance(source, torch.fx.Node):
            return None
        chain.append(source)
    conversion = chain[-1]
    source = conversion.args[0] if conversion.args else None
    if (
        conversion.target not in _CONVERTING
        or set(conversion.kwargs) - {"dtype"}
        or not isinstance(source, torch.fx.Node)
        or not isinstance(source.meta.get("val"), torch.Tensor)
        or source.meta["val"].dtype != dtype
        or outputs.count(produced) != 1
        or any(len(node.users) != 1 for node in chain)
    ):
        return None
    conversion.replace_all_uses_with(source)
    graph.erase_node(conversion)
    with fake_mode if fake_mode is not None else nullcontext():
        for node in reversed(chain[:-1]):
            args, kwargs = torch.fx.node.map_arg(
                (node.args, node.kwargs), lambda item: item.meta["val"]
            )
            _record_value(node, cast(Callable[..., Any], node.target)(*args, **kwargs))
    return chain[0] if len(chain) > 1 else source


#: Matrix multiplies beside the overload of each that sums its products and
#: writes its result at a dtype it is given, over the same arguments.
_WRITING_AT_DTYPE: dict[object, torch._ops.OpOverload] = {
    torch.ops.aten.mm.default: torch.ops.aten.mm.dtype,
    torch.ops.aten.bmm.default: torch.ops.aten.bmm.dtype,
    torch.ops.aten.addmm.default: torch.ops.aten.addmm.dtype,
    torch.ops.aten.baddbmm.default: torch.ops.aten.baddbmm.dtype,
}

#: Operations that only move values -- views and copies of one tensor --
#: which a gradient may pass through between what computes it and the
#: backward's output.
_MOVING_ONLY = frozenset(
    {
        torch.ops.aten.alias.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.t.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
    }
)


def _write_at_dtype(
    graph: torch.fx.Graph,
    produced: torch.fx.Node,
    outputs: Sequence[object],
    dtype: torch.dtype,
    fake_mode: Any,
) -> bool:
    """Have the matrix multiply that computes ``produced`` write it at ``dtype``.

    It does when ``produced`` is a multiply's result, moved at most, that
    nothing else reads, and PyTorch has a kernel that writes that multiply at
    ``dtype`` from these operands on their device -- which dtypes it accepts
    is the operator's own check. Every node from the multiply to the output
    then carries ``dtype``. Returns whether it did.
    """

    chain = [produced]
    while chain[-1].target in _MOVING_ONLY:
        source = chain[-1].args[0]
        if not isinstance(source, torch.fx.Node):
            return False
        chain.append(source)
    multiply = chain[-1]
    writing = _WRITING_AT_DTYPE.get(multiply.target)
    if (
        writing is None
        or outputs.count(produced) != 1
        or any(len(node.users) != 1 for node in chain)
    ):
        return False
    operands = torch.fx.node.map_arg(multiply.args, lambda item: item.meta["val"])
    device = cast(torch.Tensor, operands[0]).device
    schema = writing._schema
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        f"{schema.name}.{schema.overload_name}",
        torch._C._dispatch_key_for_device(device.type),
    ):
        return False
    with fake_mode if fake_mode is not None else nullcontext():
        try:
            value = writing(*operands, dtype, **multiply.kwargs)
        except RuntimeError:  # not a dtype it writes from these operands
            return False
        with graph.inserting_after(multiply):
            written = graph.call_function(
                writing, args=(*multiply.args, dtype), kwargs=dict(multiply.kwargs)
            )
        written.meta = dict(multiply.meta)
        _record_value(written, value)
        multiply.replace_all_uses_with(written)
        graph.erase_node(multiply)
        for node in reversed(chain[:-1]):
            args, kwargs = torch.fx.node.map_arg(
                (node.args, node.kwargs), lambda item: item.meta["val"]
            )
            _record_value(node, cast(Callable[..., Any], node.target)(*args, **kwargs))
    return True


#: Matrix multiplies that can add their result into a running gradient
#: themselves, whatever dtype they write.
_ADDS_INTO = frozenset(
    {
        torch.ops.aten.mm.default,
        torch.ops.aten.mm.dtype,
        torch.ops.aten.bmm.default,
        torch.ops.aten.bmm.dtype,
    }
)


def _add_in_multiply(
    graph: torch.fx.Graph,
    produced: torch.fx.Node,
    prior: torch.fx.Node,
    outputs: Sequence[object],
    fake_mode: Any,
    round_accumulation_once: bool,
) -> list[torch.fx.Node] | None:
    """Have the matrix multiply that computes ``produced`` add it into ``prior``.

    It does when ``produced`` is a multiply's result, moved at most, that
    nothing else reads; when every move can be undone on ``prior`` as a view,
    so the multiply's result lands where ``produced`` would have been added;
    and when PyTorch has an in-place kernel for it on the device that accepts
    these dtypes -- the operator's own check. A running gradient narrower than
    the dtype the multiply sums at is added so only with
    ``round_accumulation_once``. Returns the nodes it replaced, for the caller
    to erase once the output no longer reads them, or ``None``.
    """

    chain = [produced]
    while chain[-1].target in _MOVING_ONLY:
        source = chain[-1].args[0]
        if not isinstance(source, torch.fx.Node):
            return None
        chain.append(source)
    multiply = chain[-1]
    if (
        multiply.target not in _ADDS_INTO
        or outputs.count(produced) != 1
        or any(len(node.users) != 1 for node in chain)
    ):
        return None
    running = cast(torch.Tensor, prior.meta["val"])
    adding = ADDING_INTO[cast(torch.Tensor, multiply.meta["val"]).dim()]
    schema = adding._schema
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        f"{schema.name}.{schema.overload_name}",
        torch._C._dispatch_key_for_device(running.device.type),
    ):
        return None
    left, right = (cast(torch.fx.Node, item) for item in multiply.args[:2])
    operands = cast(torch.Tensor, left.meta["val"]).dtype
    if not round_accumulation_once and running.dtype != get_computation_dtype(operands):
        return None
    undoing = [undo for undo in map(_undo_move, chain[:-1]) if undo is not None]
    # Which dtypes the multiply writes at is the operator's own check, made by
    # its overload that takes one: the in-place overloads' fake forms are
    # decompositions that misread it.
    writing = cast(
        Callable[..., torch.Tensor],
        _WRITING_AT_DTYPE.get(multiply.target, multiply.target),
    )
    with fake_mode if fake_mode is not None else nullcontext():
        try:
            views = [running]
            for target, arguments in undoing:
                views.append(target(views[-1], *arguments))
            writing(left.meta["val"], right.meta["val"], running.dtype)
        except RuntimeError:  # a move no view undoes, or a dtype it cannot write
            return None
    with graph.inserting_before(multiply):
        view = prior
        for (target, arguments), value in zip(undoing, views[1:], strict=True):
            view = graph.call_function(target, args=(view, *arguments))
            _record_value(view, value)
        graph.call_function(ACCUMULATE_MATMUL, args=(view, left, right))
    return chain


def _undo_move(
    node: torch.fx.Node,
) -> tuple[Callable[..., Any], tuple[Any, ...]] | None:
    """The view that undoes the move ``node`` makes: the operation and the
    arguments after the tensor, or ``None`` for a copy, which moves nothing."""

    target = node.target
    if target is torch.ops.aten.clone.default:
        return None
    if target in (torch.ops.aten.t.default, torch.ops.aten.transpose.int):
        return cast(Callable[..., Any], target), tuple(node.args[1:])
    if target is torch.ops.aten.permute.default:
        axes = cast(Sequence[int], node.args[1])
        order = [axis % len(axes) for axis in axes]
        return torch.ops.aten.permute.default, (
            [order.index(axis) for axis in range(len(order))],
        )
    source = cast(torch.Tensor, cast(torch.fx.Node, node.args[0]).meta["val"])
    return torch.ops.aten.view.default, (list(source.shape),)


def _record_value(node: torch.fx.Node, value: torch.Tensor) -> None:
    """Record what ``node`` now computes, as graph capture records it."""

    node.meta["val"] = value
    node.meta["tensor_meta"] = _extract_tensor_metadata(value)


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
