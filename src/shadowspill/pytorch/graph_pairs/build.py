"""Construct graph-pair portfolios for one structural stage ABI."""

from __future__ import annotations

from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.capture.artifacts import GraphArtifact

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

    PyTorch's min-cut budget ``0.0`` is the full-recompute endpoint.  Fix it
    explicitly so ambient Functorch configuration cannot alter the structural
    ABI.  The opposite endpoint, ``1.0``, retains the full saved-value set and
    therefore must not be exposed as recomputation.
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
            0.0,
            capture_graph_pair(
                stage.graph_module,
                example.inputs,
                recomputation=True,
                activation_memory_budget=0.0,
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
