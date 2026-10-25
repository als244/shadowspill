"""Version-pinned AOTAutograd capture of one forward/backward graph pair."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

import torch
from functorch.compile import make_boxed_func  # type: ignore[import-untyped]
from torch._functorch import config as functorch_config
from torch._functorch.aot_autograd import aot_function
from torch.utils._pytree import tree_flatten, tree_unflatten

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import (
    AotGraphPair,
    GraphArtifact,
    TaskInputProvenance,
)
from shadowspill.pytorch.capture.retention import (
    RetentionPolicy,
    RetentionRecord,
    RetentionSummary,
    partition_function,
)
from shadowspill.pytorch.capture.storage import ExplicitMutation
from shadowspill.pytorch.representations import is_wrapper, tensor_components
from shadowspill.task.inputs import TaskInputRole

from .export import ExportCapture, _explicit_mutations
from .provenance import (
    _backward_input_provenance,
    _tensor_input_provenance,
    physical_input_provenance,
)
from .torch_deprecations import copy_graph_module


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


def _capture_inputs(inputs: Sequence[object]) -> tuple[object, ...]:
    return tuple(
        value.detach().requires_grad_(value.requires_grad)
        if isinstance(value, torch.Tensor)
        else value
        for value in inputs
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
