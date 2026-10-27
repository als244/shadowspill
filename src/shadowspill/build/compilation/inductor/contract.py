"""The storage contract a compiled callable publishes to its caller."""

from __future__ import annotations

from torch._inductor.graph import GraphLowering
from torch.fx import GraphModule

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.storage import (
    MutationBinding,
    OutputView,
    StorageRoot,
    StorageRootKind,
    TaskStorageContract,
)

from .manifest import _GraphLoweringManifest
from .outputs import _graph_input_positions, _lower_graph_outputs, _select_graph_outputs
from .roots import (
    _build_executable_roots,
    _lowered_output_views,
    _project_mutations,
    _visible_output_indices,
)
from .values import _copy_root, _copy_view


def _project_callable_contract(
    optimized_graph: GraphModule,
    inner_contract: TaskStorageContract,
    semantic_contract: TaskStorageContract,
) -> TaskStorageContract:
    """Remove compiler-private saved outputs using Inductor contract metadata."""

    roots, output_views = _project_visible_outputs(
        optimized_graph,
        inner_contract,
        semantic_contract,
    )
    mutations = _project_callable_mutations(
        semantic_contract.mutations,
        roots,
        output_views,
    )
    return TaskStorageContract.build(roots, output_views, mutations)


def _project_visible_outputs(
    optimized_graph: GraphModule,
    inner_contract: TaskStorageContract,
    semantic_contract: TaskStorageContract,
) -> tuple[tuple[StorageRoot, ...], tuple[OutputView, ...]]:
    visible = _visible_output_indices(optimized_graph)
    inner_view_by_leaf = {view.leaf_index: view for view in inner_contract.output_views}
    visible_views = tuple(
        inner_view_by_leaf[index] for index in visible if index in inner_view_by_leaf
    )
    semantic_views = tuple(
        sorted(semantic_contract.output_views, key=lambda view: view.leaf_index)
    )
    if len(visible_views) != len(semantic_views):
        raise CaptureError(
            "Inductor callable-visible tensor output count changed: "
            f"semantic={len(semantic_views)}, executable={len(visible_views)}, "
            f"visible_indices={visible}"
        )
    inner_root_by_id = {root.root_id: root for root in inner_contract.roots}
    selected_root_ids = tuple(dict.fromkeys(view.root_id for view in visible_views))
    root_index = {original: index for index, original in enumerate(selected_root_ids)}
    roots = tuple(
        _copy_root(inner_root_by_id[original], root_index[original])
        for original in selected_root_ids
    )
    output_views = tuple(
        _copy_view(executable, semantic.leaf_index, root_index[executable.root_id])
        for executable, semantic in zip(visible_views, semantic_views, strict=True)
    )
    return roots, output_views


def _project_callable_mutations(
    mutations: tuple[MutationBinding, ...],
    roots: tuple[StorageRoot, ...],
    output_views: tuple[OutputView, ...],
) -> tuple[MutationBinding, ...]:
    view_by_leaf = {view.leaf_index: view for view in output_views}
    root_by_id = {root.root_id: root for root in roots}
    for mutation in mutations:
        leaf = mutation.replacement_output_leaf
        if leaf is not None:
            view = view_by_leaf.get(leaf)
            if view is None:
                raise CaptureError(
                    "Inductor removed a functional mutation replacement: "
                    f"leaf={leaf}, target={mutation.argument_name}"
                )
            root = root_by_id[view.root_id]
            if (
                root.kind is StorageRootKind.INPUT
                and root.source_input != mutation.input_position
            ):
                raise CaptureError(
                    "functional mutation replacement aliases another input: "
                    f"leaf={leaf}, expected={mutation.input_position}, "
                    f"actual={root.source_input}"
                )
    return mutations


def _graph_lowering_contract(
    graph: GraphLowering,
    optimized_graph: GraphModule,
    inner_contract: TaskStorageContract,
    semantic_contract: TaskStorageContract,
) -> _GraphLoweringManifest:
    """Project Inductor's returned buffers into an executable storage contract."""

    visible, outputs = _select_graph_outputs(
        graph, optimized_graph, inner_contract, semantic_contract
    )
    input_position_by_name = _graph_input_positions(graph, optimized_graph)
    records = _lower_graph_outputs(
        graph,
        outputs,
        visible,
        inner_contract,
        semantic_contract,
    )
    roots, allocations = _build_executable_roots(graph, records, input_position_by_name)
    output_views = _lowered_output_views(records, roots)
    mutations = _project_mutations(semantic_contract, roots, output_views)
    contract = TaskStorageContract.build(roots, output_views, mutations)
    return _GraphLoweringManifest(contract, allocations)
