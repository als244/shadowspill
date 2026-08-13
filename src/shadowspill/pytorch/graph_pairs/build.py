"""Construct graph-pair portfolios for one structural stage ABI."""

from __future__ import annotations

from ..aot import capture_graph_pair
from ..capture import GraphArtifact
from ..partition.artifacts import StageExample
from .artifacts import GraphPairPortfolio, GraphPairVariant


def build_default_portfolio(
    example: StageExample,
    roots: tuple[int, ...],
    *,
    specialize_unit_tangents: bool,
) -> GraphPairPortfolio:
    """Capture the established default and runtime-optimized min-cut choices.

    The returned portfolio and every downstream consumer support an arbitrary
    ordered number of variants. Intermediate min-cut budgets can therefore be
    added here without changing partitioning, caching, lowering, diagnostics,
    or the canonical Program representation.

    The historical ``recompute`` choice uses PyTorch's min-cut budget ``1.0``.
    It is fixed explicitly here so ambient Functorch configuration cannot alter
    the structural ABI. It is not mislabeled as the ``0.0`` full-recompute
    endpoint.
    """

    stage = example.stage
    structural_abi = GraphArtifact.input_compatibility_digest(
        graph_module=stage.graph_module,
        example_inputs=example.inputs,
        explicit_mutations=stage.mutations,
        input_provenance=stage.input_provenance,
    )
    variants = (
        GraphPairVariant(
            "save",
            None,
            capture_graph_pair(
                stage.graph_module,
                example.inputs,
                recomputation=False,
                original_output=example.output,
                root_output_positions=roots,
                specialize_unit_tangents=specialize_unit_tangents,
                explicit_mutations=stage.mutations,
                input_provenance=stage.input_provenance,
            ),
        ),
        GraphPairVariant(
            "recompute",
            1.0,
            capture_graph_pair(
                stage.graph_module,
                example.inputs,
                recomputation=True,
                activation_memory_budget=1.0,
                original_output=example.output,
                root_output_positions=roots,
                specialize_unit_tangents=specialize_unit_tangents,
                explicit_mutations=stage.mutations,
                input_provenance=stage.input_provenance,
            ),
        ),
    )
    return GraphPairPortfolio(structural_abi, roots, variants)


__all__ = ["build_default_portfolio"]
