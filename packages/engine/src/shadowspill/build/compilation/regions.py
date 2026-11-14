"""Package locally composed execution under one outer task contract."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from shadowspill.errors import CompilationError
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.chunking.region_executable import RegionExecutable
from shadowspill.pytorch.executable import restore_executable
from shadowspill.pytorch.numerics import compose_numerical_contracts
from shadowspill.task.manifest import ExecutableRootAllocation, build_manifest
from shadowspill.task.storage import StorageRootKind

from .api import BuiltTask
from .inductor.manifest import toolchain


def compile_region_task(
    artifact: GraphArtifact, compiled: Mapping[str, BuiltTask]
) -> BuiltTask:
    task = artifact.regions
    assert task is not None
    metadata = deepcopy(task.identity)
    children = {}
    dependencies = {}
    contracts = compose_numerical_contracts(
        compiled[child.compatibility_digest].numerical_contracts
        for child in task.children
    )
    # A checkpoint can retain an outer host input directly, without a child
    # producing it. Those passthroughs still have no device allocation.
    host_slots = {
        index
        for index, geometry in enumerate(artifact.tensor_inputs)
        if geometry.device_type == "cpu"
    }
    for index, child in enumerate(task.children):
        built = compiled[child.compatibility_digest]
        instruction = metadata["regions"][index]
        host_leaves = set(built.memory["off_device_output_leaves"])
        roots = {
            view["root_id"]
            for view in instruction["boundary"]["storage"]["output_views"]
            if view["leaf_index"] in host_leaves
        }
        instruction["boundary"]["host_output_roots"] = sorted(roots)
        for view in instruction["boundary"]["storage"]["output_views"]:
            if view["root_id"] in roots and view["leaf_index"] not in host_leaves:
                raise CompilationError(
                    "one execution output root has mixed host/device views"
                )
        host_slots.update(instruction["outputs"][leaf] for leaf in host_leaves)
        children[index] = restore_executable(
            built.payload,
            dependencies=built.dependencies,
            numerical_contracts=built.numerical_contracts,
        )
        dependencies.update(
            {tuple(sorted(item.items())): item for item in built.dependencies}
        )
    host_leaves = {
        index
        for index, item in enumerate(metadata["outputs"])
        if item.get("slot") in host_slots
    }
    storage = task.storage_contract
    host_roots = {
        view.root_id for view in storage.output_views if view.leaf_index in host_leaves
    }
    if any(
        view.root_id in host_roots and view.leaf_index not in host_leaves
        for view in storage.output_views
    ):
        raise CompilationError("one checkpoint output root has mixed host/device views")
    metadata["boundary"]["host_output_roots"] = sorted(host_roots)
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
            **toolchain(contracts),
            "region_composition": artifact.compatibility_digest,
        },
    )
    return BuiltTask(
        manifest,
        RegionExecutable(metadata, children).serialize(),
        {
            "known_workspace_bytes": 0,
            "allocations": [],
            "unresolved": [
                "complete execution region composition requires allocator measurement"
            ],
            "off_device_output_leaves": tuple(sorted(host_leaves)),
        },
        tuple(dependencies.values()),
        numerical_contracts=contracts,
    )
