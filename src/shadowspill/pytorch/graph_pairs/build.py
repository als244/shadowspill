"""Construct graph pairs for one structural stage contract."""

from __future__ import annotations

from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.retention import RetentionPolicy

from ..partition.artifacts import StageExample
from .artifacts import GraphPairVariant, TaskGraphPairs


def build_default_graph_pairs(
    example: StageExample,
    roots: tuple[int, ...],
    *,
    specialize_unit_tangents: bool,
    retention: RetentionPolicy,
) -> TaskGraphPairs:
    """Capture the two endpoints of the partition budget.

    ``save`` is budget ``1.0``: the min-cut over saved bytes under
    ``retention``, which regenerates every memory-bound value and retains the
    rest. ``recompute`` is budget ``0.0``, which retains the stage's inputs
    alone. Both budgets are bound inside the lazy partition callback so
    ambient Functorch configuration cannot alter a structural contract.

    The returned record and every downstream consumer support an arbitrary
    ordered number of variants, so a budget between the two is a variant this
    builder could emit without changing partitioning, caching, lowering,
    diagnostics or the canonical ShadowSpillProgram representation.
    """

    stage = example.stage
    structural_contract = GraphArtifact.input_compatibility_digest(
        graph_module=stage.graph_module,
        example_inputs=example.inputs,
        explicit_mutations=stage.mutations,
        input_provenance=stage.input_provenance,
    )
    variants = tuple(
        GraphPairVariant(
            option_id,
            memory_budget,
            capture_graph_pair(
                stage.graph_module,
                example.inputs,
                memory_budget=memory_budget,
                retention=retention,
                original_output=example.output,
                root_output_positions=roots,
                specialize_unit_tangents=specialize_unit_tangents,
                explicit_mutations=stage.mutations,
                input_provenance=stage.input_provenance,
            ),
        )
        for option_id, memory_budget in (("save", 1.0), ("recompute", 0.0))
    )
    return TaskGraphPairs(structural_contract, roots, variants)


__all__ = ["build_default_graph_pairs"]
