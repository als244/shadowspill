"""Package child AOT code under the complete row task's storage contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from shadowspill.errors import CompilationError
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.chunking.executable import ChunkedExecutable
from shadowspill.pytorch.numerics import compose_numerical_contracts
from shadowspill.task.manifest import ExecutableRootAllocation, build_manifest
from shadowspill.task.storage import StorageRootKind

from ..allocation_behavior import composed_allocation_behavior
from .api import BuiltTask
from .inductor.manifest import toolchain


def compile_chunk_task(
    artifact: GraphArtifact,
    compiled: Mapping[str, BuiltTask],
) -> BuiltTask:
    from shadowspill.pytorch.executable import restore_executable

    task = artifact.chunking
    assert task is not None
    numerical_contracts = compose_numerical_contracts(
        compiled[child.compatibility_digest].numerical_contracts
        for _, child in (*task.children, *task.accumulating_children)
    )
    storage = task.storage_contract
    manifest = build_manifest(
        artifact.storage_contract,
        storage,
        storage,
        tuple(
            ExecutableRootAllocation(
                root.root_id,
                root.minimum_span_bytes if root.kind is StorageRootKind.FRESH else 0,
            )
            for root in storage.roots
        ),
        capture_ns=0,
        toolchain={
            **toolchain(numerical_contracts),
            "row_composition": artifact.compatibility_digest,
        },
    )
    dependencies: list[dict[str, str]] = []
    children: dict[int, Any] = {}
    accumulating_children: dict[int, Any] = {}
    host_outputs: tuple[int, ...] | None = None
    for artifacts, functions in (
        (task.children, children),
        (task.accumulating_children, accumulating_children),
    ):
        for size, child in artifacts:
            built = compiled[child.compatibility_digest]
            locations = tuple(built.memory["off_device_output_leaves"])
            if host_outputs is not None and locations != host_outputs:
                raise CompilationError("row children disagree on host output locations")
            host_outputs = locations
            dependencies.extend(built.dependencies)
            functions[size] = restore_executable(
                built.payload,
                dependencies=built.dependencies,
                numerical_contracts=built.numerical_contracts,
            )
    assert host_outputs is not None
    host_roots = {
        view.root_id for view in storage.output_views if view.leaf_index in host_outputs
    }
    if any(
        view.root_id in host_roots and view.leaf_index not in host_outputs
        for view in storage.output_views
    ):
        raise CompilationError("one row output root has mixed host/device views")
    function = ChunkedExecutable(
        {
            **task.identity,
            "storage": storage.to_dict(),
            "host_output_roots": sorted(host_roots),
            "output_count": artifact.output_count,
            "input_offsets": [x.storage_offset for x in artifact.tensor_inputs],
        },
        children,
        accumulating_children,
    )
    unique = {tuple(sorted(item.items())): item for item in dependencies}
    return BuiltTask(
        manifest,
        function.serialize(),
        {
            "known_workspace_bytes": 0,
            "allocations": [],
            "unresolved": ["complete row composition requires allocator measurement"],
            "off_device_output_leaves": host_outputs,
            "allocation_behavior": composed_allocation_behavior(
                compiled[child.compatibility_digest].memory
                for _, child in (*task.children, *task.accumulating_children)
            ),
        },
        tuple(unique.values()),
        numerical_contracts=numerical_contracts,
    )
