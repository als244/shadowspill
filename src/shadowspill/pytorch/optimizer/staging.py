"""Map registered parameters to semantic training-stage ownership."""

from __future__ import annotations

from collections.abc import Collection

from torch.export.graph_signature import InputKind, InputSpec

from shadowspill.errors import CaptureError
from shadowspill.pytorch.graph_pairs.artifacts import (
    DifferentiatedStage,
    PartitionedTrainingCapture,
    parameter_gradient_leaves,
)


def training_parameter_stage_owners(
    captures: tuple[PartitionedTrainingCapture, ...],
    parameter_names: Collection[str],
) -> dict[str, tuple[int, ...]]:
    """Return the training stages whose backward passes contribute each parameter.

    Export makes parameters explicit root inputs.  Stage partitioning preserves
    that provenance in :class:`StageValueSource`, so optimizer grouping can use
    the same semantic stage boundaries without inspecting module-name patterns
    or runtime allocation behavior.
    """

    known = frozenset(parameter_names)
    owners: dict[str, set[int]] = {}
    expected_stage_count: int | None = None
    for capture in captures:
        if expected_stage_count is None:
            expected_stage_count = len(capture.stages)
        elif len(capture.stages) != expected_stage_count:
            raise CaptureError(
                "microbatch positions produced different training-stage counts"
            )
        input_specs = tuple(
            capture.training.exported.exported_program.graph_signature.input_specs
        )
        for stage_index, stage in enumerate(capture.stages):
            for name in _stage_parameter_names(stage, input_specs, known):
                owners.setdefault(name, set()).add(stage_index)
    return {name: tuple(sorted(indices)) for name, indices in owners.items()}


def _stage_parameter_names(
    stage: DifferentiatedStage,
    input_specs: tuple[InputSpec, ...],
    known: frozenset[str],
) -> tuple[str, ...]:
    result: list[str] = []
    for source in stage.example.stage.input_sources:
        if source is None or source.root_input_index is None:
            continue
        try:
            spec = input_specs[source.root_input_index]
        except IndexError as exc:
            raise CaptureError(
                "stage parameter provenance refers outside the Export contract"
            ) from exc
        if spec.kind is not InputKind.PARAMETER:
            continue
        result.append(_optimizer_parameter_name(spec, known))
    return tuple(result)


def _optimizer_parameter_name(spec: InputSpec, known: frozenset[str]) -> str:
    target = spec.target
    if not isinstance(target, str) or not target.startswith("model."):
        raise CaptureError(
            f"objective Export parameter target is not rooted at model: {target!r}"
        )
    name = target.removeprefix("model.")
    if name not in known:
        raise CaptureError(
            f"stage parameter {name!r} is absent from the optimizer model"
        )
    return name


def training_parameters_with_gradients(
    captures: tuple[PartitionedTrainingCapture, ...],
    parameter_names: Collection[str],
) -> frozenset[str]:
    """Return the parameters a captured backward actually produces gradients for.

    ``requires_grad`` says a parameter *may* be trained. It does not say the
    objective reaches it. A weight whose only use is to produce integers --
    an index into keys, a routing choice -- has no gradient through it
    however it is flagged, and eager training skips such a parameter because
    its ``grad`` is ``None``.

    A plan has to know the same thing before it runs. Otherwise it reserves
    a gradient object nobody writes, the optimizer declares state for it,
    and the first step refuses on a fetch of an object that was never
    produced.
    """

    known = frozenset(parameter_names)
    trained: set[str] = set()
    for capture in captures:
        input_specs = tuple(
            capture.training.exported.exported_program.graph_signature.input_specs
        )
        for stage in capture.stages:
            sources = stage.example.stage.input_sources
            for variant in stage.graph_pairs.variants:
                for position in parameter_gradient_leaves(variant.pair):
                    if position >= len(sources):
                        continue
                    source = sources[position]
                    if source is None or source.root_input_index is None:
                        continue
                    try:
                        spec = input_specs[source.root_input_index]
                    except IndexError as exc:
                        raise CaptureError(
                            "stage parameter provenance refers outside the "
                            "Export contract"
                        ) from exc
                    if spec.kind is not InputKind.PARAMETER:
                        continue
                    trained.add(_optimizer_parameter_name(spec, known))
    return frozenset(trained)


__all__ = [
    "training_parameter_stage_owners",
    "training_parameters_with_gradients",
]
