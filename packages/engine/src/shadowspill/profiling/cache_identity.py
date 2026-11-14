"""Measurement identities include finalized code and representative input lineage."""

from collections.abc import Mapping, Sequence
from typing import Any

from shadowspill.artifacts._values import digest, plain
from shadowspill.artifacts.program import Program
from shadowspill.artifacts.tasks import stage_jobs
from shadowspill.task.inputs import REPRESENTATIVE_VALUE_POLICY
from shadowspill.task.profiling import ProfilingOptions


def task_identities(
    program: Program,
    jobs: Sequence[Mapping[str, Any]],
    *,
    options: ProfilingOptions | None,
    environment: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    definitions = {item.profile_key: item for item in program.task_definitions}
    inputs = program.bindings["task_inputs"]
    producers: dict[str, tuple[Mapping[str, Any], int]] = {}
    component: Mapping[str, Any]
    job: Mapping[str, Any]
    for job in stage_jobs(program, None):
        if "forward" in job:
            component = job["forward"]
            passed = {slot["object_id"] for slot in component["inputs"]}
            for slot in component["outputs"]:
                if slot["object_id"] not in passed:
                    producers.setdefault(
                        slot["object_id"], (component, slot["leaf_index"])
                    )

    def lineage(component: Mapping[str, Any]) -> str:
        key = component["profile_key"]
        controls = []
        provenance = inputs[key].get("provenance", ())
        for slot in component["inputs"]:
            position, object_id = slot["leaf_index"], slot["object_id"]
            source = provenance[position] if provenance else {}
            if (
                source.get("role") == "control"
                and "value" not in source
                and "value_bytes" not in source
                and object_id in producers
            ):
                producer, leaf = producers[object_id]
                controls.append((position, leaf, lineage(producer)))
        return digest(
            {
                "implementation": definitions[key].compatibility_digest,
                "inputs": inputs[key],
                "controls": controls,
            }
        )

    contexts: dict[str, dict[str, Any] | None] = {}
    for job in jobs:
        if "forward" not in job:
            component = job["optimizer"] if "optimizer" in job else job["gradient"]
            contexts.setdefault(component["profile_key"], None)
            continue
        forward = job["forward"]
        context = lineage(forward)
        contexts.setdefault(forward["profile_key"], {"forward": context})
        if (backward := job.get("backward")) is not None:
            sources = {
                slot["object_id"]: (kind, slot["leaf_index"])
                for kind in ("inputs", "outputs")
                for slot in forward[kind]
            }
            contexts.setdefault(
                backward["profile_key"],
                {
                    "forward": context,
                    "bindings": [
                        (slot["leaf_index"], sources[slot["object_id"]])
                        for slot in backward["inputs"]
                        if slot["object_id"] in sources
                    ],
                },
            )
    policy = None if options is None else options.to_dict()
    # A record with more samples can satisfy a smaller requested minimum.
    if policy is not None:
        policy.pop("minimum_samples")
    return {
        key: {
            "schema": "shadowspill.task-measurement-request/v1",
            "profile_key": key,
            # Finalization includes selected Inductor/Triton binaries and
            # provider preparation manifests. Serialized Python payload bytes
            # also contain build-local names/paths and are not this identity.
            "implementation": definitions[key].compatibility_digest,
            "setup": plain(program.setup),
            "manifest": {
                name: plain(value)
                for name, value in program.bindings["manifests"][key].items()
                if name != "contract_capture_ns"
            },
            "inputs": plain(inputs[key]),
            "input_context": context,
            "representative_value_policy": REPRESENTATIVE_VALUE_POLICY,
            "numerical_contracts": plain(
                program.bindings.get("task_metadata", {})
                .get(key, {})
                .get("numerical_contracts", ())
            ),
            "resources": {
                name: value.to_dict() for name, value in program.resources.items()
            },
            "resource_sources": plain(program.bindings.get("resource_sources", {})),
            "environment": plain(environment),
            "profiling_options": policy,
        }
        for key, context in contexts.items()
    }
