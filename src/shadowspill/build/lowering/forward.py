"""Emit forward IR from compiler geometry, with no task measurements."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any

from torch.utils._pytree import treespec_dumps

from shadowspill.artifacts._values import digest
from shadowspill.artifacts.program import Program, TaskDefinition
from shadowspill.ir import DeviceSpec
from shadowspill.pytorch.lowering.forward.artifacts import ForwardTaskGeometry
from shadowspill.pytorch.lowering.forward.objects import register_forward_objects
from shadowspill.pytorch.lowering.forward.residency import derive_forward_residency
from shadowspill.pytorch.lowering.forward.tasks import emit_forward_tasks
from shadowspill.task.geometry import TaskStorageGeometry

from ..capture import ForwardCapture
from ..compilation import BuiltTask
from ..metadata import module_paths, root_sources, signature_record, task_inputs


def lower_forward(
    captured: ForwardCapture, compiled: tuple[BuiltTask, ...], *, setup: dict[str, Any]
) -> Program:
    keys = tuple(
        digest(
            {
                "graph": artifact.compatibility_digest,
                "implementation": item.manifest.compatibility_digest,
                "setup": setup,
            }
        )
        for artifact, item in zip(captured.artifacts, compiled, strict=True)
    )
    geometry = ForwardTaskGeometry(
        tuple(item.manifest.storage_contract for item in compiled),
        tuple(TaskStorageGeometry.from_manifest(item.manifest) for item in compiled),
        keys,
        "invocation_start",
    )
    device_id = "cuda_0"
    objects = register_forward_objects(
        captured.model, captured.partitioned, device_id=device_id
    )
    graph = emit_forward_tasks(
        captured.partitioned, captured.artifacts, objects, geometry, device_id=device_id
    )
    initial, final = derive_forward_residency(objects, graph)
    definitions = {
        "invocation_start": TaskDefinition(
            "invocation_start", "control", 0, "Invocation start"
        )
    }
    for key, stage, item in zip(
        keys, captured.partitioned.stages, compiled, strict=True
    ):
        paths = module_paths(stage.stage.graph_module)
        if key in definitions:
            paths = tuple(dict.fromkeys((*definitions[key].module_paths, *paths)))
        definitions[key] = TaskDefinition(
            key,
            key,
            workspace_bytes=(
                item.memory["known_workspace_bytes"]
                if not item.memory["unresolved"]
                else None
            ),
            name=" / ".join(paths) or stage.stage.module_target,
            module_paths=paths,
        )
    sources: dict[str, list[str]] = defaultdict(list)
    for registration in objects.registrations:
        sources[objects.catalog.alias_id(registration.object_id)].append(
            registration.name
        )
    roots = root_sources(captured.exported)
    for slot in objects.root_input_slots:
        name = roots[slot.leaf_index]["name"]
        names = sources[objects.catalog.alias_id(slot.object_id)]
        if name and name not in names:
            names.append(name)
    return Program(
        devices=(DeviceSpec(device_id, "process_0", "cuda", 0),),
        alias_groups=objects.catalog.alias_groups(),
        objects=objects.catalog.objects(),
        tasks=graph.tasks,
        task_definitions=tuple(definitions.values()),
        initial_residency=initial,
        final_residency=final,
        name=type(captured.model).__name__,
        setup=setup,
        sources=sources,
        executables={
            key: item.payload for key, item in zip(keys, compiled, strict=True)
        },
        bindings={
            "kind": "forward",
            "provider": "torch-inductor-aot",
            "entrypoints": [asdict(item) for item in graph.entrypoints],
            "registrations": [asdict(item) for item in objects.registrations],
            "root_inputs": [asdict(item) for item in objects.root_input_slots],
            "root_sources": roots,
            "input_signature": signature_record(captured.input_signature),
            "public_outputs": graph.public_outputs,
            "output_literals": captured.output_literals,
            "output_tree": treespec_dumps(captured.output_tree),
            "manifests": {
                key: item.manifest.to_dict()
                for key, item in zip(keys, compiled, strict=True)
            },
            "compiler_memory": {
                key: item.memory for key, item in zip(keys, compiled, strict=True)
            },
            "task_inputs": {
                key: task_inputs(artifact)
                for key, artifact in zip(keys, captured.artifacts, strict=True)
            },
        },
    )
