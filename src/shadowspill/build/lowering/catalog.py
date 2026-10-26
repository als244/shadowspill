"""Reusable compiled task identities and geometry, without timing records."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from shadowspill.artifacts._values import digest
from shadowspill.artifacts.program import TaskDefinition
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.task.geometry import TaskStorageGeometry
from shadowspill.task.storage import TaskStorageContract

from ..compilation import BuiltTask
from ..metadata import task_inputs


class CompiledTasks:
    def __init__(self, compiled: Mapping[str, BuiltTask], setup: dict[str, Any]):
        self.compiled = compiled
        self.setup = setup
        self.definitions = {
            "invocation_start": TaskDefinition(
                "invocation_start", "control", 0, "Invocation start"
            )
        }
        self.executables: dict[str, bytes] = {}
        self.manifests: dict[str, Any] = {}
        self.memory: dict[str, Any] = {}
        self.inputs: dict[str, Any] = {}
        self.dependencies: dict[str, Any] = {}

    def key(self, artifact: GraphArtifact, paths: tuple[str, ...] = ()) -> str:
        item = self.compiled[artifact.compatibility_digest]
        key = digest(
            {
                "graph": artifact.compatibility_digest,
                "implementation": item.manifest.compatibility_digest,
                "setup": self.setup,
                "dependencies": item.dependencies,
                "compiler_input_offsets": item.input_offsets,
            }
        )
        definition = self.definitions.get(key)
        if definition is None:
            definition = TaskDefinition(
                key,
                key,
                workspace_bytes=(
                    item.memory["known_workspace_bytes"]
                    if not item.memory["unresolved"]
                    else None
                ),
            )
            self.executables[key] = item.payload
            self.manifests[key] = item.manifest.to_dict()
            self.memory[key] = item.memory
            self.inputs[key] = task_inputs(artifact)
            self.dependencies[key] = item.dependencies
        paths = tuple(dict.fromkeys((*definition.module_paths, *paths)))
        self.definitions[key] = replace(
            definition,
            module_paths=paths,
            name=f"{' / '.join(paths)} {artifact.kind}".strip(),
        )
        return key

    def contract(self, artifact: GraphArtifact) -> TaskStorageContract:
        item = self.compiled[artifact.compatibility_digest]
        return item.manifest.storage_contract.without_device_storage(
            item.memory["off_device_output_leaves"]
        )

    def geometry(
        self, artifact: GraphArtifact, metadata_digest: str | None = None
    ) -> TaskStorageGeometry:
        item = self.compiled[artifact.compatibility_digest]
        contract = self.contract(artifact)
        host_roots = {
            view.root_id
            for view in contract.output_views
            if view.leaf_index in item.memory["off_device_output_leaves"]
        }
        sizes = tuple(
            0 if allocation.root_id in host_roots else allocation.requested_bytes
            for allocation in item.manifest.root_allocations
        )
        return TaskStorageGeometry(
            contract.compatibility_digest,
            sizes,
            sizes,
            tuple(
                (view.leaf_index, view.offset_bytes) for view in contract.output_views
            ),
        )

    def bindings(self) -> dict[str, Any]:
        return {
            "provider": "torch-inductor-aot",
            "manifests": self.manifests,
            "compiler_memory": self.memory,
            "task_inputs": self.inputs,
            "operator_dependencies": self.dependencies,
        }
