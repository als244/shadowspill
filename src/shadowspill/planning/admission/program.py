"""Complete observed memory may drive placement; estimates remain simulation-only."""

from shadowspill.artifacts.entrypoints import entrypoint
from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.ir import ObjectRole, ShadowSpillProgram
from shadowspill.planner.admission import AdmissionFacts
from shadowspill.profiling.records import measured_task

from .facts import build_admission_facts
from .outputs import output_bindings_for_entrypoints


def caller_output_aliases(program: ShadowSpillProgram) -> frozenset[str]:
    sizes = {item.alias_group_id: item.size_bytes for item in program.alias_groups}
    return frozenset(
        item.alias_group_id
        for item in program.objects
        if item.role is ObjectRole.OUTPUT and sizes[item.alias_group_id]
    )


def program_admission(
    profiled: ProfiledProgram, costed: ShadowSpillProgram, usable_bytes: int
) -> AdmissionFacts | None:
    if profiled.estimated_memory or not profiled.program.executables:
        return None
    evidence = {}
    for definition in profiled.program.task_definitions:
        if definition.profile_key == "invocation_start":
            continue
        record = profiled.task_profiles[definition.profile_key].metadata
        if "allocation_contract" not in record or "compiled_layout" not in record:
            return None
        measurement = measured_task(
            profiled.task_profiles[definition.profile_key],
            profiled.program.bindings["manifests"][definition.profile_key],
        )
        evidence[definition.compatibility_digest] = measurement.allocation_trace
    bindings = output_bindings_for_entrypoints(
        costed.tasks,
        tuple(entrypoint(item) for item in profiled.program.bindings["entrypoints"]),
        {item.object_id: item.alias_group_id for item in costed.objects},
    )
    return build_admission_facts(
        costed,
        execution_pool_bytes=usable_bytes,
        object_capacity_bytes=usable_bytes,
        output_bindings=bindings,
        allocation_traces_by_compatibility=evidence,
    )
