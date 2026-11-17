"""Choose one measured implementation for equivalent physical workers."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from shadowspill.artifacts._values import digest, plain
from shadowspill.artifacts.plan import Plan
from shadowspill.artifacts.pool_memory import PoolMemoryProfile
from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.artifacts.program import Program
from shadowspill.communication.info import CommunicationInfo
from shadowspill.training.preparation.capture.templates import TemplateUse


def plan_sources(plans: Mapping[int, Mapping[str, Plan]]) -> set[str]:
    """Retain a representative even when its original candidate was not selected."""
    return {
        key
        for worker in plans.values()
        for plan in worker.values()
        for key in (plan.program.bindings.get("preparation_template", {}).get("id"),)
        if key is not None
    }


def _ownership(value: Mapping[str, Any]) -> dict[str, Any]:
    # These describe staging/publication ownership, not the captured function.
    # In particular, contributes_loss, output geometry and placement stay exact.
    return {
        key: item
        for key, item in value.items()
        if key not in {"logical_rank", "buffer"}
    }


def computation_identity(program: Program) -> str:
    """Compare source computation and hardware, independently of tuning winners.

    All ranks execute the representative's complete binary/profile/Plan, never a
    schedule paired with a different local implementation. The pre-finalization
    task digest binds the graph, ABI, compiler and providers. Other physical IR,
    runtime bindings, model declarations and hardware must also match. Distinct
    loss ownership, model shards, shapes or hardware therefore stay separate.
    """
    value = program.to_dict()
    bindings = value["bindings"]
    template = bindings["preparation_template"]
    template.pop("id")
    template["binding"] = _ownership(template["binding"])
    if "training_operation" in bindings:
        bindings["training_operation"] = _ownership(bindings["training_operation"])
    finalized = program.bindings.get("finalized_tasks", {})
    for row in value["task_definitions"]:
        key = row["profile_key"]
        if key in finalized:
            identity = finalized[key]
            if digest(identity) != row["compatibility_digest"] or identity[
                "inductor_kernels"
            ] != program.bindings.get("inductor_kernels", {}).get(key):
                raise ValueError(
                    f"task {key}: invalid finalized implementation identity"
                )
            row["compatibility_digest"] = identity["task"]
            # Device placement is a resolved property of the original task.
            row["output_devices"] = plain(identity["output_devices"])
            if key in value["executables"]:
                value["executables"][key] = identity["task"]
    for name in (
        "finalized_tasks",
        "triton_choices",
        "inductor_kernels",
        "operator_preparation",
        "manifests",
        "allocation_coverage",
    ):
        bindings.pop(name, None)
    value.pop("operator_binaries", None)
    return digest(value)


def representative_sources(
    sources: Mapping[str, ProfiledProgram],
    pool_memory: Mapping[int, PoolMemoryProfile | None] | None = None,
    communication: Mapping[str, CommunicationInfo] | None = None,
) -> dict[str, str]:
    """Map equivalent captures to one deterministic, measured implementation.

    Rank-local input bindings and initialization remain in the setup inventory.
    Timing noise does not create additional searches or diverging schedules.
    Pool calibration, when supplied, must match before a Plan can be shared.
    """
    groups: dict[tuple[str, str | None, str | None, str | None], str] = {}
    result = {}
    ordered = sorted(
        sources,
        key=lambda key: (
            sources[key].program.bindings["preparation_template"]["binding"][
                "logical_rank"
            ],
            key,
        ),
    )
    for key in ordered:
        source = sources[key]
        rank = source.program.bindings["preparation_template"]["binding"][
            "logical_rank"
        ]
        pool = None if pool_memory is None else pool_memory[rank]
        # Explicit hypothetical per-worker costs are different planning inputs.
        # Ordinary measurement noise is not; use the representative's evidence.
        estimated = any(
            task.metadata.get("duration_kind") == "estimated"
            for task in source.task_profiles.values()
        )
        comm = None if communication is None else communication[key]
        identity = (
            computation_identity(source.program),
            None if pool is None else digest(pool.to_dict()),
            digest(source.profiles.to_dict()) if estimated else None,
            digest(comm.to_dict())
            if comm is not None and comm.metadata.get("timing_kind") == "estimated"
            else None,
        )
        result[key] = groups.setdefault(identity, key)
    return result


def representative_use(
    use: TemplateUse, representatives: Mapping[str, str]
) -> TemplateUse:
    return replace(use, template=representatives[use.template])
