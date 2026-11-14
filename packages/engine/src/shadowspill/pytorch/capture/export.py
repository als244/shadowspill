"""Strict PyTorch export, functionalization and explicit mutation signatures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.export.graph_signature import ExportGraphSignature, InputKind, OutputKind

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import (
    GraphArtifact,
    capture_inference_artifact,
)
from shadowspill.pytorch.capture.storage import ExplicitMutation
from shadowspill.pytorch.representations import is_wrapper

from .effects import normalize_export_effects
from .functional import functionalize_logical_export
from .torch_deprecations import quiet_leaf_spec_deprecation


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


def _explicit_mutations(capture: ExportCapture) -> tuple[ExplicitMutation, ...]:
    return tuple(
        ExplicitMutation(item.input_index, item.output_index, item.target)
        for item in capture.mutations
    )
