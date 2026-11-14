"""Construct graph pairs for one structural stage contract."""

from __future__ import annotations

from collections.abc import Sequence

from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.capture.artifacts import GraphArtifact

from ..partition.artifacts import StageExample
from .artifacts import GraphPairVariant, TaskGraphPairs
from .policy import GraphPairPolicy

#: The two endpoints of the partition budget, in the order they are built.
GRAPH_PAIR_VARIANTS: tuple[str, ...] = ("save", "recompute")
_MEMORY_BUDGETS = {"save": 1.0, "recompute": 0.0}


def validate_graph_pair_variants(variants: Sequence[str]) -> tuple[str, ...]:
    """The variants named for a stage's graph pairs, in the build's own order.

    The graph-pair store keys its entries by them, and
    :func:`~shadowspill.pytorch.graph_pairs.training.partition_training_capture`
    takes them as ``graph_pair_variants``, both by default. A caller may name
    fewer than both endpoints, but never none, and never a name that is not an
    endpoint.
    """

    chosen = tuple(variants)
    unknown = sorted(set(chosen) - set(_MEMORY_BUDGETS))
    if unknown:
        raise ValueError(
            f"unknown graph-pair variants {', '.join(unknown)}; choose from"
            f" {', '.join(GRAPH_PAIR_VARIANTS)}"
        )
    if not chosen or len(set(chosen)) != len(chosen):
        raise ValueError("graph_pair_variants must name at least one distinct variant")
    return tuple(name for name in GRAPH_PAIR_VARIANTS if name in chosen)


def forward_sink_variants(variants: Sequence[str]) -> tuple[str, ...]:
    """The variants a sink of the forward phase is built with.

    A stage no later forward stage reads produces values only the backward
    consumes, and graph-pair selection forces such a group to ``save``:
    recomputing it would recompute it from nothing. The planner requires
    that option to exist, so it is built for a sink whatever was asked.
    """

    return validate_graph_pair_variants(tuple(dict.fromkeys(("save", *variants))))


def build_default_graph_pairs(
    example: StageExample,
    roots: tuple[int, ...],
    *,
    specialize_unit_tangents: bool,
    graph_pair_policy: GraphPairPolicy,
    variants: Sequence[str] = GRAPH_PAIR_VARIANTS,
) -> TaskGraphPairs:
    """Capture independent saved-state cuts for the two named endpoints.

    ``graph_pair_policy.save`` controls the save cut; ``.recompute`` controls
    the recompute cut, including retained additions. The numeric budgets label
    the base endpoints, not actual saved bytes. Both policies are bound inside
    the lazy partition callback, so ambient configuration cannot change them.

    ``variants`` names the endpoints to capture. A request whose search can
    select only one of them asks for that one alone: nothing is captured,
    compiled or profiled for a choice nobody can make, and the reference
    that fixes the stage-boundary contract is then that variant.
    """

    stage = example.stage
    structural_contract = GraphArtifact.input_compatibility_digest(
        graph_module=stage.graph_module,
        example_inputs=example.inputs,
        explicit_mutations=stage.mutations,
        input_provenance=stage.input_provenance,
    )
    chosen = validate_graph_pair_variants(variants)
    built = tuple(
        GraphPairVariant(
            option_id,
            _MEMORY_BUDGETS[option_id],
            capture_graph_pair(
                stage.graph_module,
                example.inputs,
                memory_budget=_MEMORY_BUDGETS[option_id],
                retention=graph_pair_policy.variant(option_id),
                original_output=example.output,
                root_output_positions=roots,
                specialize_unit_tangents=specialize_unit_tangents,
                explicit_mutations=stage.mutations,
                input_provenance=stage.input_provenance,
            ),
        )
        for option_id in chosen
    )
    return TaskGraphPairs(
        structural_contract,
        roots,
        built,
        reference_option_id="save" if "save" in chosen else chosen[0],
    )


__all__ = [
    "GRAPH_PAIR_VARIANTS",
    "build_default_graph_pairs",
    "forward_sink_variants",
    "validate_graph_pair_variants",
]
